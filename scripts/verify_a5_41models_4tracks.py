#!/usr/bin/env python3
"""Verify all catalog + preset models across A5 SKUs, 4 feature tracks & CP physics.

[Role]
  Audit every core catalog model and architecture preset across
  ``Ascend950DT_9582``, ``Ascend950PR_958b``, and the 4 commercial
  ``Ascend950DT`` derived SKUs, verifying ``raw_eager -> merged -> unmerged_pr
  -> expected_ideal`` waterfall monotonicity (preserving slower transport
  candidates and their measured crossover when selecting shape-optimal ideal),
  strict topological DAG validity and zero ``ResourceEnvelope`` / unmodeled
  hits on all ranks' lowered graphs, 100% distinct structural layer-signature
  coverage, per-rank and per-layer cross-rank DCP Attention FLOPs, collective
  wire traffic, and exact-bounded KV HBM conservation (plus explicit refusal of
  upstream-unsupported replicated-cache DCP), and ``mindie-profiler`` structure
  ingress compatibility.

[Invariants]
  * Exact inventory parity against ``CORE_CATALOG_ENTRIES | ARCHITECTURE_PRESET_CATALOG``
    and ``list_models()`` with zero silent drops; every requested unit produces
    a row even on failure before returning non-zero exit code.
  * Every duration checked is finite and strictly positive (``math.isfinite``
    and ``> 0.0``), and every derived SKU checks observed ``memory_bytes``.
  * For every KV-reading LLM supporting DCP, every rank and every layer
    individually conserve cross-rank Attention FLOPs within 2% tile tolerance,
    emit positive DCP collective wire traffic on every rank, and satisfy both
    lower and upper KV HBM bounds.
  * ``buffer_units["l2"] == 1`` across all base and derived A5 SKUs.

[Timeouts & Locks]
  Bounded ``ProcessPoolExecutor`` sized via ``runtime_advisor.resolve_jobs``;
  zero device, network, or file-lock side effects.

[Side-Effect Policy]
  Read-only verification; writes optional JSON summary only when ``--output``
  is passed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import dataclasses
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mindie_sim.backends.ascend import profiles as ascend_profiles  # noqa: E402
from mindie_sim.catalog.context_parallel import (  # noqa: E402
    cp_interleave,
    is_v41_replicated_cache_model,
    kv_cache_replication_info,
    layer_dcp_factors,
    owned_positions,
)
from mindie_sim.catalog.fleet import ARCHITECTURE_PRESET_CATALOG, CORE_CATALOG_ENTRIES  # noqa: E402
from mindie_sim.compose import compose_case  # noqa: E402
from mindie_sim.model_catalog import CATALOG, load_model  # noqa: E402
from mindie_sim.prediction.capacity import Memory  # noqa: E402
from mindie_sim.prediction.efficiency import _lower_and_run_graph  # noqa: E402
from mindie_sim.prediction.operator_analysis import target_graphs  # noqa: E402
from mindie_sim.prediction.structure import parse_case  # noqa: E402
from mindie_sim.runtime_advisor import resolve_jobs  # noqa: E402
from mindie_sim.schema.common import ModelingError, dtype_bytes  # noqa: E402
from mindie_sim.structure import export_structure  # noqa: E402

TRACKS = ("raw_eager", "merged", "unmerged_pr", "expected_ideal")
PRIMARY_SKUS = ("Ascend950DT_9582", "Ascend950PR_958b")
COMMERCIAL_DERIVED_SKUS = (
    ("425-96", 28, 96 * 1024),
    ("486-96", 32, 96 * 1024),
    ("547-96", 36, 96 * 1024),
    ("547-144", 36, 144 * 1024),
)
MONOTONIC_TOL_MS = 1e-5
FLOPS_CONSERVATION_TOL = 0.02
KV_CONSERVATION_TOL = 0.02
AUDIT_LAYERS = 4
INLINE_SIGNATURE_MAX_LAYERS = 8
MIN_CATALOG_MODELS = 39
WORLD_DEVICES = 8
EXPECTED_ASCEND_PROFILES = 16
DECODE_CONTEXT = 2048
DECODE_SEQS = 4
CONSERVATION_CONTEXT = 512
CONSERVATION_SEQS = 2
SUBMISSION_KERNEL_LAUNCH_MS = 0.062
SUBMISSION_GRAPH_REPLAY_MS = 0.30
BYTES_PER_MIB = 1024 * 1024
DECLARED_BODY_INPUT_ROOTS = frozenset({"analysis_input"})
DCP_COLLECTIVE_RECIPES = frozenset({
    "dcp_query_allgather",
    "dcp_dp_query_allgather",
    "dcp_output_alltoall",
    "dcp_pcp_allgather",
    "dcp_kv_allgather",
    "dcp_prefill_kv_gather",
    "candidate_gather",
})
DCP_COLLECTIVE_AXES = frozenset({
    "cp",
    "dcp",
    "dcp_tp",
    "dcp_pcp",
    "dcp_dp",
    "dcp_a2a",
    "tp",
    "pcp",
})
DTYPE_BYTES_FP8 = 1
DTYPE_BYTES_BF16 = 2
DTYPE_BYTES_FP32 = 4
FAST_LSE_WIRE_BYTES = 4
GENERIC_LSE_WIRE_BYTES = 8
CANDIDATE_ENTRY_BYTES = 4
CANDIDATE_PAIR_BYTES = 8
SFA_GENERIC_MAX_HEADS = 64
SFA_PADDED_HEADS = 128
WIRE_BOUND_TOL = 1.0

ALL_CATALOG_MODELS = tuple(sorted(
    set(CORE_CATALOG_ENTRIES.keys()) | set(ARCHITECTURE_PRESET_CATALOG.keys())
))


def _expected_inventory_keys() -> set[str]:
    """Independent expected model + preset inventory grounded in on-disk config files."""
    pkg_root = Path(_REPO_ROOT) / "mindie_sim"
    disk_keys = (
        {p.stem for p in (pkg_root / "model_configs").glob("*.json")}
        | {p.stem for p in (pkg_root / "catalog" / "configs").glob("*.json")}
    )
    fleet_keys = set(CORE_CATALOG_ENTRIES.keys()) | set(ARCHITECTURE_PRESET_CATALOG.keys())
    catalog_keys = set(CATALOG.keys()) | set(ARCHITECTURE_PRESET_CATALOG.keys())
    if fleet_keys != catalog_keys:
        raise RuntimeError(
            f"inventory_drift:fleet_only={sorted(fleet_keys - catalog_keys)}:"
            f"catalog_only={sorted(catalog_keys - fleet_keys)}"
        )
    if fleet_keys != disk_keys:
        raise RuntimeError(
            f"inventory_disk_drift:missing={sorted(disk_keys - fleet_keys)}:"
            f"extra={sorted(fleet_keys - disk_keys)}"
        )
    return fleet_keys


def _layer_signature(plan: Any) -> tuple[Any, ...]:
    """Full physical LayerPlan signature (drops raw position, relativizes owner IDs)."""
    idx = int(getattr(plan, "index", 0))
    sig: list[tuple[str, Any]] = []
    for field in dataclasses.fields(plan):
        if field.name in ("index", "native_layer_id"):
            continue
        val = getattr(plan, field.name)
        if field.name.endswith("layer_id"):
            ival = int(val)
            val = (
                "absent"
                if ival < 0
                else ("self" if ival == idx else ("earlier" if ival < idx else "later"))
            )
        sig.append((field.name, val))
    return tuple(sig)


def _signature_coverage_prefix(spec: Any) -> tuple[int, int]:
    """Return ``(min_prefix_covering_all_signatures, distinct_signature_count)``."""
    plans = tuple(spec.layer_plans)
    all_sigs = {_layer_signature(p) for p in plans}
    seen: set[tuple[Any, ...]] = set()
    min_prefix = len(plans)
    for idx, plan in enumerate(plans, 1):
        seen.add(_layer_signature(plan))
        if seen == all_sigs:
            min_prefix = idx
            break
    return min_prefix, len(all_sigs)


def _require_finite_positive(value: Any, *, tag: str) -> float:
    """Reject NaN, +/-inf, and non-positive durations everywhere."""
    val = float(value)
    if not math.isfinite(val) or val <= 0.0:
        raise RuntimeError(f"invalid_non_positive_or_non_finite_duration:{tag}:{val}")
    return val


def _canonical_parallel_for_model(spec: Any) -> dict[str, int]:
    """Deterministic canonical 8-die parallelism across ALL model layers (zero fallback)."""
    plans = tuple(spec.layer_plans)
    has_moe = any(plan.ffn == "moe" for plan in plans)
    has_mla = any(plan.attention == "mla" for plan in plans)
    ep = WORLD_DEVICES if has_moe else 1
    if has_mla:
        # Canonical MLA / DSA / CSA default: Pure Attention DP + MoE EP (zero KV replication)
        return {"dp": WORLD_DEVICES, "tp": 1, "ep": ep, "num_devices": WORLD_DEVICES}

    gqa_kv_heads = {
        int(plan.num_kv_heads) for plan in plans
        if plan.attention == "gqa" and int(plan.num_kv_heads) > 0
    }
    linear_heads = {
        int(getattr(plan, "linear_value_heads", 0) or getattr(plan, "linear_heads", 0) or 1)
        for plan in plans if plan.attention in {"gdn", "kda"}
    }
    q_heads = {int(plan.num_heads) for plan in plans if int(plan.num_heads) > 0}
    if int(spec.num_heads) > 0:
        q_heads.add(int(spec.num_heads))

    for tp in (8, 4, 2, 1):
        if any(h % tp != 0 for h in q_heads):
            continue
        if any(kv % tp != 0 for kv in gqa_kv_heads):
            continue
        if any(lh % tp != 0 for lh in linear_heads):
            continue
        dp = WORLD_DEVICES // tp
        return {"dp": dp, "tp": tp, "ep": ep, "num_devices": WORLD_DEVICES}
    return {"dp": WORLD_DEVICES, "tp": 1, "ep": ep, "num_devices": WORLD_DEVICES}


def _assert_clean_graph(
    graphs: Any,
    rep: dict[str, Any],
    *,
    tag: str,
    external_roots: frozenset[str] = DECLARED_BODY_INPUT_ROOTS,
) -> None:
    """Assert zero forbidden recipes, unique op_ids, and strict topological DAG across ALL ranks."""
    if not graphs:
        raise RuntimeError(f"empty_replica_graphs:{tag}")
    for rank, ops in graphs.items():
        if not ops:
            raise RuntimeError(f"empty_rank_graph:{tag}:rank={rank}")
        seen: set[str] = set()
        for op in ops:
            if op.op_id in seen:
                raise RuntimeError(f"duplicate_op_id:{tag}:rank={rank}:{op.op_id}")
            recipe = str(getattr(op, "recipe", "") or "")
            if "ResourceEnvelope" in recipe or "unmodeled" in recipe:
                raise RuntimeError(f"forbidden_graph_op:{tag}:rank={rank}:{op.op_id}:{recipe}")
            for dep in getattr(op, "deps", ()):
                if dep not in seen and dep not in external_roots:
                    raise RuntimeError(
                        f"undefined_or_inverted_dep:{tag}:rank={rank}:{op.op_id}->{dep}"
                    )
            seen.add(op.op_id)
    text = json.dumps(rep, default=str)
    if "ResourceEnvelope" in text or "unmodeled_operator_mechanism" in text:
        raise RuntimeError(f"forbidden_mechanism:{tag}")


def _layer_attention_flops_by_rank(
    graphs: Any,
    *,
    tag: str,
    layer_plans: tuple[Any, ...],
    tp: int,
    tp_gather: int,
) -> tuple[dict[int, dict[str, float]], dict[str, int]]:
    """Aggregate useful attention FLOPs per physical rank/layer without trusting pair metadata."""
    out: dict[int, dict[str, float]] = {}
    layer_indices: dict[str, int] = {}
    for rank, ops in graphs.items():
        per_layer: dict[str, float] = {}
        for op in ops:
            if op.family != "attention":
                continue
            fl = float(op.flops)
            if not math.isfinite(fl) or fl < 0.0:
                raise RuntimeError(f"invalid_attention_flops:{tag}:rank={rank}:{op.op_id}:{fl}")
            layer_idx = int(getattr(op, "layer", -1))
            layer_key = op.op_id.rsplit(".", 1)[0] if "." in op.op_id else op.op_id
            if 0 <= layer_idx < len(layer_plans):
                layer_indices[layer_key] = layer_idx
                plan = layer_plans[layer_idx]
                gathered_h = (int(plan.num_heads) // max(1, tp)) * max(1, tp_gather)
                meta = dict(getattr(op, "metadata", ()))
                if (
                    plan.indexer
                    and SFA_GENERIC_MAX_HEADS < gathered_h < SFA_PADDED_HEADS
                    and int(meta.get("sfa_padded_query_heads") or 0) == SFA_PADDED_HEADS
                ):
                    fl *= float(gathered_h) / float(SFA_PADDED_HEADS)
            per_layer[layer_key] = per_layer.get(layer_key, 0.0) + fl
        out[int(rank)] = per_layer
    return out, layer_indices


def _expected_dcp_rank_shares(
    plan: Any,
    *,
    context: int,
    dcp_degree: int,
    interleave: int,
) -> tuple[float, ...]:
    """Independently derive per-rank attention pair shares from geometry and context."""
    degree = max(1, int(dcp_degree))
    if degree == 1:
        return (1.0,)
    if plan.indexer or plan.cache_kind != "dense":
        return tuple(1.0 / float(degree) for _ in range(degree))
    visible = context + 1
    if int(plan.window) > 0:
        visible = min(visible, int(plan.window))
    owned = tuple(
        float(owned_positions(0, visible - 1, degree, interleave, cp_rank))
        for cp_rank in range(degree)
    )
    total = sum(owned)
    if total <= 0.0:
        return tuple(1.0 / float(degree) for _ in range(degree))
    return tuple(v / total for v in owned)


def _validate_rank_keys(
    graphs: Any,
    *,
    tag: str,
    dp: int,
    tp: int,
    require_all_world_ranks: bool,
) -> None:
    """Reject missing replicas or impossible physical ranks outside ``[0, dp * tp)``."""
    world = dp * tp
    expected_ranks = set(range(world)) if require_all_world_ranks else ({0} if len(graphs) == 1 else {idx * tp for idx in range(dp)})
    ranks = {int(r) for r in graphs}
    if not graphs or ranks != expected_ranks or any(r < 0 or r >= world for r in ranks):
        raise RuntimeError(
            f"impossible_physical_ranks:{tag}:got={sorted(ranks)}:expected={sorted(expected_ranks)}:world={world}"
        )


def _expected_dcp_wire_bounds(
    recipe: str,
    *,
    model: Any,
    plan: Any,
    seqs: int,
    dcp_dp: int,
    dcp_tp: int,
    dcp_degree: int,
) -> tuple[float, float]:
    """Independent analytical (min, max) wire bytes derived purely from spec geometry."""
    act_b = float(dtype_bytes(model.activation_dtype))
    local_h = max(1, int(plan.num_heads) // max(1, dcp_tp))
    if plan.attention == "mla":
        rope_dim = (
            int(plan.qk_rope_head_dim)
            if int(plan.cache_row_elements) > int(plan.kv_lora_rank)
            else 0
        )
        q_dim = int(plan.kv_lora_rank) + rope_dim
    else:
        q_dim = int(plan.head_dim)
    v_dim = int(plan.kv_lora_rank) if plan.attention == "mla" else int(plan.head_dim)
    if recipe == "dcp_dp_query_allgather":
        exact = float(seqs * local_h * q_dim) * act_b * float(max(0, dcp_dp - 1))
        return exact, exact
    if recipe == "dcp_query_allgather":
        exact = float(seqs * local_h * q_dim) * act_b * float(max(0, dcp_tp - 1))
        return exact, exact
    if recipe == "dcp_output_alltoall":
        lo = float(seqs * local_h) * (float(v_dim) * act_b + float(FAST_LSE_WIRE_BYTES)) * float(dcp_degree - 1)
        hi = float(seqs * local_h) * (float(v_dim) * act_b + float(GENERIC_LSE_WIRE_BYTES)) * float(dcp_degree - 1)
        return lo, hi
    if recipe == "candidate_gather":
        topk = max(1, int(plan.index_topk))
        lo = float(seqs * topk * CANDIDATE_ENTRY_BYTES * (dcp_degree - 1))
        hi = float(seqs * max(1, dcp_dp) * topk * CANDIDATE_PAIR_BYTES * (dcp_degree - 1))
        return lo, hi
    lo = float(seqs * local_h * v_dim) * act_b * float(dcp_degree - 1)
    hi = float(seqs * max(1, dcp_dp) * local_h * (v_dim * act_b + GENERIC_LSE_WIRE_BYTES)) * float(dcp_degree - 1)
    return lo, hi


def _validate_dcp_collectives(
    dcp_graphs: Any,
    *,
    tag: str,
    model: Any,
    layer_plans: tuple[Any, ...],
    expected_layer_indices: set[int],
    dcp_dp: int,
    dcp_tp: int,
    dcp_pcp: int,
    dcp_degree: int,
) -> None:
    """Assert per-layer DCP collectives, legal axes, unique groups, and analytical wire bounds."""
    dcp_world = dcp_dp * dcp_tp * dcp_pcp
    allowed_axes = set(DCP_COLLECTIVE_AXES)
    if dcp_tp <= 1:
        allowed_axes -= {"tp", "dcp_tp"}
    if dcp_pcp <= 1:
        allowed_axes -= {"pcp", "dcp_pcp"}
    if dcp_dp <= 1:
        allowed_axes -= {"dcp_dp"}
    for rank, ops in sorted(dcp_graphs.items()):
        r = int(rank)
        seen_layers: set[int] = set()
        for op in ops:
            recipe = str(getattr(op, "recipe", "") or "")
            if recipe not in DCP_COLLECTIVE_RECIPES:
                continue
            layer_idx = int(getattr(op, "layer", -1))
            if not (0 <= layer_idx < len(layer_plans)):
                raise RuntimeError(f"invalid_dcp_collective_layer:{tag}:rank={r}:{op.op_id}:{layer_idx}")
            seen_layers.add(layer_idx)
            meta = dict(getattr(op, "metadata", ()))
            axis = str(meta.get("group_axis") or "")
            if axis not in allowed_axes:
                raise RuntimeError(
                    f"invalid_dcp_collective_group_axis:{tag}:rank={r}:{op.op_id}:{axis}:allowed={sorted(allowed_axes)}"
                )
            group_ranks = meta.get("group_ranks")
            if group_ranks is not None:
                members = tuple(int(x) for x in group_ranks)
                if (
                    len(members) < 2
                    or len(set(members)) != len(members)
                    or r not in members
                    or any(x < 0 or x >= dcp_world for x in members)
                ):
                    raise RuntimeError(
                        f"invalid_dcp_collective_group_ranks:{tag}:rank={r}:{op.op_id}:{members}:world={dcp_world}"
                    )
            lb = float(getattr(op, "link_bytes", 0.0))
            lo_wire, hi_wire = _expected_dcp_wire_bounds(
                recipe, model=model, plan=layer_plans[layer_idx],
                seqs=CONSERVATION_SEQS, dcp_dp=dcp_dp, dcp_tp=dcp_tp, dcp_degree=dcp_degree,
            )
            if not math.isfinite(lb) or lb <= 0.0 or lb < lo_wire - WIRE_BOUND_TOL or lb > hi_wire + WIRE_BOUND_TOL:
                raise RuntimeError(
                    f"dcp_collective_wire_out_of_bounds:{tag}:rank={r}:{op.op_id}:got={lb}:bounds=({lo_wire},{hi_wire})"
                )
        missing_layers = sorted(expected_layer_indices - seen_layers)
        if missing_layers:
            raise RuntimeError(
                f"missing_dcp_collectives_on_layers:{tag}:rank={r}:missing={missing_layers}:seen={sorted(seen_layers)}"
            )


def _check_per_rank_per_layer_conservation(
    *,
    tag: str,
    cs_dcp: Any,
    base_graphs: Any,
    dcp_graphs: Any,
    base_dp: int,
    base_tp: int,
    dcp_dp: int,
    dcp_tp: int,
    dcp_degree: int,
    require_all_world_ranks: bool,
) -> float:
    """Assert every rank & every KV layer individually conserve FLOPs, groups, and wire bounds."""
    _validate_rank_keys(base_graphs, tag=f"{tag}:base", dp=base_dp, tp=base_tp, require_all_world_ranks=False)
    _validate_rank_keys(
        dcp_graphs, tag=f"{tag}:dcp", dp=dcp_dp, tp=dcp_tp, require_all_world_ranks=require_all_world_ranks,
    )
    layer_plans = tuple(cs_dcp.model.layer_plans)
    base_scale = (float(base_dp) / float(len(base_graphs))) * float(base_tp)
    dcp_scale = float(dcp_dp * dcp_tp) / float(len(dcp_graphs))

    base_by_rank, base_idx = _layer_attention_flops_by_rank(
        base_graphs, tag=f"{tag}:base", layer_plans=layer_plans, tp=base_tp, tp_gather=1,
    )
    dcp_tp_gather = min(dcp_tp, dcp_degree)
    dcp_by_rank, dcp_idx = _layer_attention_flops_by_rank(
        dcp_graphs, tag=f"{tag}:dcp", layer_plans=layer_plans, tp=dcp_tp, tp_gather=dcp_tp_gather,
    )
    base_layers = sorted({lk for r_map in base_by_rank.values() for lk in r_map})
    dcp_layers = sorted({lk for r_map in dcp_by_rank.values() for lk in r_map})
    if not base_layers:
        raise RuntimeError(f"zero_attention_layers:{tag}")
    if dcp_layers != base_layers:
        raise RuntimeError(f"attention_layer_set_mismatch:{tag}:base={base_layers}:dcp={dcp_layers}")

    interleave = int(cp_interleave(cs_dcp.runtime))
    total_base = 0.0
    total_dcp = 0.0
    sorted_dcp_ranks = sorted(dcp_by_rank)
    expected_layer_indices: set[int] = set()
    for lk in base_layers:
        layer_base = sum(base_by_rank[r].get(lk, 0.0) for r in sorted(base_by_rank)) * base_scale
        if not math.isfinite(layer_base) or layer_base <= 0.0:
            raise RuntimeError(f"invalid_base_layer_flops:{tag}:{lk}:{layer_base}")
        l_idx = dcp_idx.get(lk, base_idx.get(lk, -1))
        if not (0 <= l_idx < len(layer_plans)):
            raise RuntimeError(f"unresolved_attention_layer_index:{tag}:{lk}:{l_idx}")
        expected_layer_indices.add(l_idx)
        shares = _expected_dcp_rank_shares(
            layer_plans[l_idx], context=CONSERVATION_CONTEXT, dcp_degree=dcp_degree, interleave=interleave,
        )
        layer_dcp_raw = 0.0
        for rank_pos, r in enumerate(sorted_dcp_ranks):
            r_fl = dcp_by_rank[r].get(lk, 0.0)
            share_r = shares[rank_pos] if len(sorted_dcp_ranks) == len(shares) else 1.0 / float(len(sorted_dcp_ranks))
            expected_r = (layer_base / dcp_scale) * share_r
            if expected_r == 0.0:
                if r_fl != 0.0:
                    raise RuntimeError(f"unexpected_nonzero_idle_owner_flops:{tag}:rank={r}:{lk}:{r_fl}")
            else:
                if r_fl <= 0.0:
                    raise RuntimeError(f"zero_rank_layer_flops:{tag}:rank={r}:{lk}")
                rel_r = abs(r_fl - expected_r) / expected_r
                if rel_r > FLOPS_CONSERVATION_TOL:
                    raise RuntimeError(
                        f"rank_flops_imbalance:{tag}:rank={r}:{lk}:{r_fl:.1f}_vs_{expected_r:.1f}:rel={rel_r:.4f}"
                    )
            layer_dcp_raw += r_fl
        layer_dcp = layer_dcp_raw * dcp_scale
        rel = abs(layer_dcp - layer_base) / layer_base
        if not math.isfinite(rel) or rel > FLOPS_CONSERVATION_TOL:
            raise RuntimeError(
                f"layer_flops_not_conserved:{tag}:{lk}:{layer_dcp:.1f}_vs_{layer_base:.1f}:rel={rel:.4f}"
            )
        total_base += layer_base
        total_dcp += layer_dcp

    _validate_dcp_collectives(
        dcp_graphs,
        tag=tag,
        model=cs_dcp.model,
        layer_plans=layer_plans,
        expected_layer_indices=expected_layer_indices,
        dcp_dp=dcp_dp,
        dcp_tp=dcp_tp,
        dcp_pcp=int(cs_dcp.runtime.parallel.pcp),
        dcp_degree=dcp_degree,
    )
    return round(total_dcp / total_base, 6)


def _check_kv_bounds(
    *,
    tag: str,
    cs_base: Any,
    cs_dcp: Any,
    sliced: tuple[Any, ...],
    dcp_degree: int,
) -> float:
    """Enforce finite, exact-bounded KV HBM and physical rank ownership across every rank."""
    base_bytes = tuple(float(v) for v in Memory(cs_base).cache_bytes(CONSERVATION_CONTEXT, padded=False))
    dcp_bytes = tuple(float(v) for v in Memory(cs_dcp).cache_bytes(CONSERVATION_CONTEXT, padded=False))
    if (
        not base_bytes
        or not dcp_bytes
        or any(not math.isfinite(v) or v <= 0.0 for v in base_bytes)
        or any(not math.isfinite(v) or v <= 0.0 for v in dcp_bytes)
    ):
        raise RuntimeError(f"invalid_kv_cache_bytes:{tag}:base={base_bytes}:dcp={dcp_bytes}")
    kv1 = max(base_bytes)
    kv2_max = max(dcp_bytes)
    kv2_min = min(dcp_bytes)
    pure_dense_kv = all(
        p.attention in {"mla", "gqa"} and p.cache_kind == "dense" and not p.indexer
        for p in sliced
    )
    expected_shard = kv1 / float(dcp_degree)
    if pure_dense_kv:
        tol = max(1.0, KV_CONSERVATION_TOL * expected_shard)
        if abs(kv2_max - expected_shard) > tol or abs(kv2_min - expected_shard) > tol:
            raise RuntimeError(
                f"tp_dcp_kv_hbm_out_of_bounds:{tag}:min={kv2_min}:max={kv2_max}:expected={expected_shard}"
            )
    else:
        if kv2_min < expected_shard - 1.0 or kv2_max >= kv1:
            raise RuntimeError(
                f"tp_dcp_hybrid_kv_hbm_out_of_bounds:{tag}:min={kv2_min}:max={kv2_max}:kv1={kv1}"
            )
    p = cs_dcp.runtime.parallel
    expected_world = max(1, int(p.dp)) * max(1, int(p.tp)) * max(1, int(p.pcp))
    rank_alloc = Memory(cs_dcp).rank_cache_allocation(CONSERVATION_CONTEXT)
    if len(rank_alloc) != expected_world:
        raise RuntimeError(
            f"missing_physical_kv_ranks:{tag}:got={len(rank_alloc)}:expected={expected_world}"
        )
    rank_valid = tuple(
        sum(float(v) for v in row["valid"]["total_bytes_per_stage_rank"])
        for row in rank_alloc
    )
    if any(not math.isfinite(v) or v < 0.0 for v in rank_valid):
        raise RuntimeError(f"invalid_rank_kv_allocation:{tag}:{rank_valid}")
    interleave = int(cp_interleave(cs_dcp.runtime))
    first_kv = next((pl for pl in sliced if pl.attention in {"mla", "gqa"}), sliced[0])
    pcp_f, tp_g, dp_w = layer_dcp_factors(first_kv, p)
    total_valid = sum(rank_valid)
    if pure_dense_kv:
        for r_idx, v in enumerate(rank_valid):
            tp_i = r_idx % max(1, int(p.tp))
            pcp_i = (r_idx // max(1, int(p.tp))) % max(1, int(p.pcp))
            dp_i = r_idx // max(1, int(p.tp) * int(p.pcp))
            cp_r = ((dp_i % dp_w) * pcp_f + (pcp_i % pcp_f)) * tp_g + (tp_i % tp_g)
            owned = (
                owned_positions(0, CONSERVATION_CONTEXT - 1, dcp_degree, interleave, cp_r)
                if CONSERVATION_CONTEXT > 0 else 0
            )
            exp_r = kv1 * (float(owned) / float(CONSERVATION_CONTEXT)) if CONSERVATION_CONTEXT > 0 else 0.0
            if owned == 0:
                if v != 0.0:
                    raise RuntimeError(f"unexpected_nonzero_idle_kv_rank:{tag}:rank={r_idx}:{v}")
            else:
                tol_r = max(1.0, KV_CONSERVATION_TOL * exp_r)
                if abs(v - exp_r) > tol_r:
                    raise RuntimeError(
                        f"rank_kv_mismatch:{tag}:rank={r_idx}:got={v}:expected={exp_r}"
                    )
        if abs(total_valid - kv1) > max(1.0, KV_CONSERVATION_TOL * kv1):
            raise RuntimeError(f"world_kv_total_mismatch:{tag}:sum={total_valid}:expected={kv1}")
    else:
        lo_world = kv1 * (1.0 - KV_CONSERVATION_TOL)
        hi_world = kv1 * float(dcp_degree) * (1.0 + KV_CONSERVATION_TOL)
        if total_valid < lo_world or total_valid > hi_world:
            raise RuntimeError(
                f"hybrid_world_kv_out_of_bounds:{tag}:sum={total_valid}:bounds=({lo_world},{hi_world})"
            )
    return round(kv2_max / kv1, 6)


def _verify_dcp_conservation(model_key: str, spec: Any, *, audit_layers: int) -> dict[str, Any]:
    """Verify per-rank, per-layer Attention FLOPs, DCP collectives, durations, and KV HBM bounds."""
    sliced = tuple(spec.layer_plans[:min(audit_layers, len(spec.layer_plans))])
    has_kv = any(p.attention in {"mla", "gqa"} and int(p.num_heads) >= 2 for p in sliced)
    if not has_kv:
        return {"checked": False, "reason": "no_kv_attention_layers"}
    if is_v41_replicated_cache_model(spec):
        has_moe = any(p.ffn == "moe" for p in sliced)
        ep2 = 2 if has_moe else 1
        for kw in (
            dict(dp=1, tp=2, ep=ep2, dcp=2, num_devices=2),
            dict(dp=2, tp=1, pcp=4, dcp=4, ep=WORLD_DEVICES if has_moe else 1, num_devices=WORLD_DEVICES),
        ):
            try:
                compose_case(
                    model_key, hardware=PRIMARY_SKUS[0], layers=audit_layers,
                    va_recommended=False, **kw,
                )
            except ModelingError as exc:
                if "v41_replicated_cache_requires_dcp_1" not in str(exc):
                    raise RuntimeError(f"unexpected_v41_dcp_error:{model_key}:{exc}") from exc
            else:
                raise RuntimeError(f"v41_dcp_must_be_refused:{model_key}:{kw}")
        return {
            "checked": True,
            "dcp_refusal_verified": True,
            "reason": "v41_replicated_cache_requires_dcp_1",
        }

    has_moe = any(p.ffn == "moe" for p in sliced)
    ep2 = 2 if has_moe else 1

    # 1) Universal DP-cohort DCP=2 vs Pure DP=2 (every KV-reading model)
    c_dp1 = compose_case(
        model_key, hardware=PRIMARY_SKUS[0], dp=2, tp=1, ep=ep2, dcp=1,
        num_devices=2, layers=audit_layers, va_recommended=False,
    )
    c_dp2 = compose_case(
        model_key, hardware=PRIMARY_SKUS[0], dp=2, tp=1, ep=ep2, dcp=2,
        num_devices=2, layers=audit_layers, va_recommended=False,
    )
    cs_dp1, cs_dp2 = parse_case(c_dp1), parse_case(c_dp2)
    g1, _, rep1 = _lower_and_run_graph(
        cs_dp1, phase="decode", tokens=1, context=CONSERVATION_CONTEXT,
        sequences=CONSERVATION_SEQS, scope="body", include_submission=False, fast_span_only=True,
    )
    g2, _, rep2 = _lower_and_run_graph(
        cs_dp2, phase="decode", tokens=1, context=CONSERVATION_CONTEXT,
        sequences=CONSERVATION_SEQS, scope="body", include_submission=False, fast_span_only=True,
    )
    _assert_clean_graph(g1, rep1, tag=f"{model_key}:dp_dcp1")
    _assert_clean_graph(g2, rep2, tag=f"{model_key}:dp_dcp2")
    dp1_ms = _require_finite_positive(rep1["duration_ms"], tag=f"{model_key}:dp_dcp1")
    dp2_ms = _require_finite_positive(rep2["duration_ms"], tag=f"{model_key}:dp_dcp2")
    dp_ratio = _check_per_rank_per_layer_conservation(
        tag=f"{model_key}:dp_dcp2", cs_dcp=cs_dp2, base_graphs=g1, dcp_graphs=g2,
        base_dp=2, base_tp=1, dcp_dp=2, dcp_tp=1, dcp_degree=2, require_all_world_ranks=True,
    )
    result: dict[str, Any] = {
        "checked": True,
        "dp_dcp1_ms": round(dp1_ms, 6),
        "dp_dcp2_ms": round(dp2_ms, 6),
        "dp_dcp2_flops_ratio": dp_ratio,
        "dp_dcp2_kv_ratio": _check_kv_bounds(
            tag=f"{model_key}:dp_dcp2", cs_base=cs_dp1, cs_dcp=cs_dp2,
            sliced=sliced, dcp_degree=2,
        ),
    }

    # 2) For MLA / 1-KV-head models, also verify TP=2 DCP=2 vs TP=2 DCP=1 across BOTH physical ranks
    is_mla_or_1kv = any(
        p.attention == "mla" or (p.attention == "gqa" and int(p.num_kv_heads) == 1)
        for p in sliced
    )
    if is_mla_or_1kv:
        c_tp1 = compose_case(
            model_key, hardware=PRIMARY_SKUS[0], dp=1, tp=2, ep=ep2, dcp=1,
            num_devices=2, layers=audit_layers, va_recommended=False,
        )
        c_tp2 = compose_case(
            model_key, hardware=PRIMARY_SKUS[0], dp=1, tp=2, ep=ep2, dcp=2,
            num_devices=2, layers=audit_layers, va_recommended=False,
        )
        cs_tp1, cs_tp2 = parse_case(c_tp1), parse_case(c_tp2)
        if not c_tp1["runtime"]["metadata"].get("kv_cache_replicated_tp"):
            raise RuntimeError(f"missing_kv_cache_replicated_tp_flag:{model_key}")
        if c_tp2["runtime"]["metadata"].get("kv_cache_replicated_tp"):
            raise RuntimeError(f"unexpected_kv_cache_replicated_tp_under_dcp:{model_key}")
        gt1, _, rept1 = _lower_and_run_graph(
            cs_tp1, phase="decode", tokens=1, context=CONSERVATION_CONTEXT,
            sequences=CONSERVATION_SEQS, scope="body", include_submission=False, fast_span_only=True,
        )
        gt2, _, rept2 = _lower_and_run_graph(
            cs_tp2, phase="decode", tokens=1, context=CONSERVATION_CONTEXT,
            sequences=CONSERVATION_SEQS, scope="body", include_submission=False, fast_span_only=True,
        )
        _assert_clean_graph(gt1, rept1, tag=f"{model_key}:tp_dcp1")
        _assert_clean_graph(gt2, rept2, tag=f"{model_key}:tp_dcp2")
        gt2_all = dict(gt2)
        if 1 not in gt2_all:
            peer_g = target_graphs(
                cs_tp2, phase="decode", tokens=1, context=CONSERVATION_CONTEXT,
                sequences=CONSERVATION_SEQS, scope="body", physical_rank=1,
            )
            gt2_all[1] = next(iter(peer_g.values()))
        result["tp_dcp1_ms"] = round(_require_finite_positive(rept1["duration_ms"], tag=f"{model_key}:tp_dcp1"), 6)
        result["tp_dcp2_ms"] = round(_require_finite_positive(rept2["duration_ms"], tag=f"{model_key}:tp_dcp2"), 6)
        result["tp_dcp2_flops_ratio"] = _check_per_rank_per_layer_conservation(
            tag=f"{model_key}:tp_dcp2", cs_dcp=cs_tp2, base_graphs=gt1, dcp_graphs=gt2_all,
            base_dp=1, base_tp=2, dcp_dp=1, dcp_tp=2, dcp_degree=2, require_all_world_ranks=True,
        )
        result["tp_dcp2_kv_ratio"] = _check_kv_bounds(
            tag=f"{model_key}:tp_dcp2", cs_base=cs_tp1, cs_dcp=cs_tp2,
            sliced=sliced, dcp_degree=2,
        )
    return result


def _eval_expected_ideal_candidates(
    model_key: str,
    sku: str,
    par: dict[str, int],
    *,
    audit_layers: int,
    unmerged_mechs: set[str],
    ideal_raw: dict[str, Any],
    ideal_ms: float,
) -> tuple[float, dict[str, Any] | None]:
    """Select shape-optimal ideal among valid MoE transports while preserving slower crossovers."""
    ideal_mechs = {
        m["id"]
        for m in (ideal_raw["runtime"]["metadata"].get("feature_track_report") or {}).get(
            "active_mechanisms", ()
        )
    }
    if "dynamic_mc2_dispatch" not in ideal_mechs or "dynamic_mc2_dispatch" in unmerged_mechs:
        return ideal_ms, None
    alt_raw = compose_case(
        model_key, hardware=sku, feature_track="expected_ideal",
        _stack_phase="decode", layers=audit_layers,
        exclude_optimizations=("dynamic_mc2_dispatch",), **par,
    )
    alt_graphs, _, alt_rep = _lower_and_run_graph(
        parse_case(alt_raw), phase="decode", tokens=1, context=DECODE_CONTEXT,
        sequences=DECODE_SEQS, scope="body", include_submission=False, fast_span_only=True,
    )
    _assert_clean_graph(alt_graphs, alt_rep, tag=f"{model_key}:{sku}:expected_ideal_alltoall")
    alt_ms = _require_finite_positive(
        alt_rep["duration_ms"], tag=f"{model_key}:{sku}:expected_ideal_alltoall",
    )
    selected = "dynamic_mc2_dispatch" if ideal_ms <= alt_ms else "alltoall"
    return min(ideal_ms, alt_ms), {
        "dynamic_mc2_ms": round(ideal_ms, 6),
        "alltoall_ms": round(alt_ms, 6),
        "selected": selected,
        "crossover_delta_ms": round(ideal_ms - alt_ms, 6),
    }


def _audit_single_model(model_key: str) -> dict[str, Any]:
    spec = load_model(model_key)
    sig_prefix, distinct_sigs = _signature_coverage_prefix(spec)
    audit_layers = (
        min(max(AUDIT_LAYERS, sig_prefix), spec.layers)
        if sig_prefix <= INLINE_SIGNATURE_MAX_LAYERS
        else min(AUDIT_LAYERS, spec.layers)
    )
    is_dit = model_key == "hunyuan-video-dit" or getattr(spec, "architecture", "") == "diffusion"
    sku_results: dict[str, Any] = {}
    observed_layers_by_sku: dict[str, set[int]] = {}

    for sku in PRIMARY_SKUS:
        sku_observed: set[int] = set()
        if is_dit:
            raw = compose_case(model_key, hardware=sku, tp=WORLD_DEVICES, ep=1,
                               num_devices=WORLD_DEVICES, layers=audit_layers)
            cs = parse_case(raw)
            graphs, _, rep = _lower_and_run_graph(
                cs, phase="prefill", tokens=256, context=0, sequences=1,
                scope="body", include_submission=False, fast_span_only=True,
            )
            _assert_clean_graph(graphs, rep, tag=f"{model_key}:{sku}:dit")
            sku_observed.update(
                int(op.layer) for ops in graphs.values() for op in ops if int(op.layer) >= 0
            )
            observed_layers_by_sku[sku] = sku_observed
            dur = _require_finite_positive(rep["duration_ms"], tag=f"{model_key}:{sku}:dit")
            sku_results[sku] = {
                "dit_body_ms": dur,
                "track_exemption": "diffusion_architecture_single_track_prefill",
            }
            continue

        par = _canonical_parallel_for_model(spec)
        track_ms: dict[str, float] = {}
        submitted_ms: dict[str, float] = {}
        kv_rep_info: dict[str, Any] = {}
        unmerged_mechs: set[str] = set()
        ideal_candidates: dict[str, Any] | None = None

        for trk in TRACKS:
            raw = compose_case(
                model_key, hardware=sku, feature_track=trk,
                _stack_phase="decode", layers=audit_layers, **par,
            )
            cs = parse_case(raw)
            if not kv_rep_info:
                kv_rep_info = kv_cache_replication_info(cs.model, cs.runtime.parallel)
                if kv_rep_info["kv_cache_replicated_tp"]:
                    raise RuntimeError(
                        f"canonical_parallel_must_not_replicate_kv:{model_key}:{par}"
                    )
            graphs, _, rep = _lower_and_run_graph(
                cs, phase="decode", tokens=1, context=DECODE_CONTEXT, sequences=DECODE_SEQS,
                scope="body", include_submission=False, fast_span_only=True,
            )
            _assert_clean_graph(graphs, rep, tag=f"{model_key}:{sku}:{trk}")
            sku_observed.update(
                int(op.layer) for ops in graphs.values() for op in ops if int(op.layer) >= 0
            )
            dur = _require_finite_positive(rep["duration_ms"], tag=f"{model_key}:{sku}:{trk}")
            if trk == "unmerged_pr":
                unmerged_mechs = {
                    m["id"]
                    for m in (raw["runtime"]["metadata"].get("feature_track_report") or {}).get(
                        "active_mechanisms", ()
                    )
                }
            if trk == "expected_ideal":
                dur, ideal_candidates = _eval_expected_ideal_candidates(
                    model_key, sku, par, audit_layers=audit_layers,
                    unmerged_mechs=unmerged_mechs, ideal_raw=raw, ideal_ms=dur,
                )
            track_ms[trk] = dur
            if trk in ("raw_eager", "merged"):
                raw_sub = dict(raw)
                raw_sub["runtime"] = {
                    **dict(raw["runtime"]),
                    "submission": {
                        "kernel_launch_ms": SUBMISSION_KERNEL_LAUNCH_MS,
                        "graph_replay_ms": SUBMISSION_GRAPH_REPLAY_MS,
                    },
                }
                _, _, rep_sub = _lower_and_run_graph(
                    parse_case(raw_sub), phase="decode", tokens=1, context=DECODE_CONTEXT,
                    sequences=DECODE_SEQS, scope="body", include_submission=True,
                    fast_span_only=True,
                )
                submitted_ms[trk] = _require_finite_positive(
                    rep_sub["duration_ms"], tag=f"{model_key}:{sku}:{trk}:submitted",
                )

        if submitted_ms["merged"] > submitted_ms["raw_eager"] + MONOTONIC_TOL_MS:
            raise RuntimeError(
                f"non_monotonic_submitted:{model_key}:{sku}:"
                f"raw_eager={submitted_ms['raw_eager']:.6f}<merged={submitted_ms['merged']:.6f}"
            )
        for hi, lo in zip(TRACKS[1:-1], TRACKS[2:]):
            if track_ms[lo] > track_ms[hi] + MONOTONIC_TOL_MS:
                raise RuntimeError(
                    f"non_monotonic:{model_key}:{sku}:{hi}={track_ms[hi]:.6f}<{lo}={track_ms[lo]:.6f}"
                )
        extra_sig_ms: float | None = None
        if sig_prefix > audit_layers:
            raw_full = compose_case(
                model_key, hardware=sku, feature_track="merged",
                _stack_phase="decode", layers=sig_prefix, **par,
            )
            g_full, _, rep_full = _lower_and_run_graph(
                parse_case(raw_full), phase="decode", tokens=1, context=DECODE_CONTEXT,
                sequences=DECODE_SEQS, scope="body", include_submission=False, fast_span_only=True,
            )
            _assert_clean_graph(g_full, rep_full, tag=f"{model_key}:{sku}:full_signatures")
            sku_observed.update(
                int(op.layer) for ops in g_full.values() for op in ops if int(op.layer) >= 0
            )
            extra_sig_ms = _require_finite_positive(
                rep_full["duration_ms"], tag=f"{model_key}:{sku}:full_signatures",
            )
        observed_layers_by_sku[sku] = sku_observed
        sku_entry: dict[str, Any] = {
            "canonical_parallel": par,
            "kv_replication": kv_rep_info,
            "tracks_ms": track_ms,
            "submitted_ms": submitted_ms,
        }
        if ideal_candidates is not None:
            sku_entry["ideal_transport_candidates"] = ideal_candidates
        if extra_sig_ms is not None:
            sku_entry["full_signature_merged_ms"] = extra_sig_ms
        sku_results[sku] = sku_entry

    dcp_conservation = (
        {"checked": False, "reason": "diffusion_model"}
        if is_dit else _verify_dcp_conservation(model_key, spec, audit_layers=audit_layers)
    )
    observed_common = (
        set.intersection(*observed_layers_by_sku.values())
        if observed_layers_by_sku else set()
    )
    executed_sig_layers = len(observed_common)
    covered_sigs = len({
        _layer_signature(spec.layer_plans[idx])
        for idx in observed_common
        if 0 <= idx < len(spec.layer_plans)
    })
    if covered_sigs != distinct_sigs:
        raise RuntimeError(
            f"incomplete_signature_coverage:{model_key}:covered={covered_sigs}:total={distinct_sigs}:"
            f"observed={sorted(observed_common)}"
        )
    return {
        "model": model_key,
        "status": "passed",
        "declared_layers": int(spec.layers),
        "audited_layers": int(audit_layers),
        "signature_coverage_layers": executed_sig_layers,
        "total_layer_signatures": int(distinct_sigs),
        "covered_layer_signatures": int(covered_sigs),
        "skus": sku_results,
        "dcp_conservation": dcp_conservation,
    }


def _audit_single_model_safe(model_key: str) -> dict[str, Any]:
    """Never abort the remaining fleet on a single-model failure."""
    try:
        return _audit_single_model(model_key)
    except Exception as exc:
        return {
            "model": model_key,
            "status": "failed",
            "error": f"{type(exc).__name__}:{exc}",
        }


def _verify_derived_skus_and_parallelism() -> dict[str, Any]:
    profiles = ascend_profiles()
    if len(profiles) != EXPECTED_ASCEND_PROFILES:
        raise RuntimeError(f"ascend_profiles_count_changed:{len(profiles)}")
    derived_checked = {}
    for alias, exp_aic, exp_hbm_mib in COMMERCIAL_DERIVED_SKUS:
        raw = compose_case(
            "qwen3.6-35b-a3b", hardware=alias, tp=4, dp=2, ep=WORLD_DEVICES,
            num_devices=WORLD_DEVICES, layers=AUDIT_LAYERS,
        )
        dev = raw["machine"]["device"]
        obs_aic = int(dev["compute_units"])
        obs_hbm_mib = int(dev["memory_bytes"]) // BYTES_PER_MIB
        if obs_aic != exp_aic:
            raise RuntimeError(f"derived_aic_mismatch:{alias}:obs={obs_aic}:exp={exp_aic}")
        if obs_hbm_mib != exp_hbm_mib:
            raise RuntimeError(f"derived_hbm_mismatch:{alias}:obs={obs_hbm_mib}:exp={exp_hbm_mib}")
        if int(dev["buffer_units"]["l2"]) != 1:
            raise RuntimeError(f"derived_l2_unit_violation:{alias}")
        cs = parse_case(raw)
        graphs, _, rep = _lower_and_run_graph(
            cs, phase="decode", tokens=1, context=4096, sequences=8,
            scope="body", include_submission=False, fast_span_only=True,
        )
        _assert_clean_graph(graphs, rep, tag=f"derived:{alias}")
        dur = _require_finite_positive(rep["duration_ms"], tag=f"derived:{alias}")
        derived_checked[alias] = {
            "aic": obs_aic, "hbm_mib": obs_hbm_mib, "decode_body_ms": dur,
        }

    # All canonical parallelism regimes + explicit TP-replicated transparency + V4.1 DCP refusal
    par_checks: dict[str, Any] = {}
    for tag, model, kw, ph, tok, ctx, expect_rep in (
        ("PureDP_EP", "deepseek-v3.2", dict(dp=8, tp=1, ep=8, num_devices=8), "decode", 1, 4096, False),
        ("DP_LocalTP", "glm-5.3-flash", dict(dp=8, tp=1, ep=8, linear_proj_tp=8, o_proj_tp=8, lm_head_tp=8, num_devices=8), "decode", 1, 4096, False),
        ("TP_DCP", "kimi-k3.1", dict(dp=1, tp=8, dcp=8, ep=8, num_devices=8), "decode", 1, 16384, False),
        ("PCP_V41_C2_DCP1", "deepseek-v4.1-flash", dict(dp=2, tp=1, pcp=4, dcp=1, ep=8, num_devices=8), "prefill", 1024, 0, False),
        ("PCP_KeyPool", "glm-5.3-flash", dict(dp=2, tp=1, pcp=4, dcp=4, ep=8, num_devices=8), "prefill", 1024, 0, False),
        ("TP_EP_ReplicatedKV", "kimi-k3.1", dict(dp=1, tp=8, dcp=1, ep=8, num_devices=8), "decode", 1, 4096, True),
    ):
        raw = compose_case(
            model, hardware=PRIMARY_SKUS[0], feature_track="merged",
            _stack_phase=ph, layers=AUDIT_LAYERS, **kw,
        )
        rep_flag = bool(raw["runtime"]["metadata"].get("kv_cache_replicated_tp"))
        if rep_flag is not expect_rep:
            raise RuntimeError(
                f"kv_replication_flag_mismatch:{tag}:got={rep_flag}:expected={expect_rep}"
            )
        cs = parse_case(raw)
        graphs, _, rep = _lower_and_run_graph(
            cs, phase=ph, tokens=tok, context=ctx, sequences=4,
            scope="body", include_submission=False, fast_span_only=True,
        )
        _assert_clean_graph(graphs, rep, tag=f"parallel:{tag}")
        dur = _require_finite_positive(rep["duration_ms"], tag=f"parallel:{tag}")
        par_checks[tag] = {
            "duration_ms": dur,
            "kv_cache_replicated_tp": rep_flag,
        }

    # Explicit negative control: V4.1 replicated-cache PCP with DCP=4 MUST raise ModelingError
    try:
        compose_case(
            "deepseek-v4.1-flash", hardware=PRIMARY_SKUS[0], feature_track="merged",
            _stack_phase="prefill", layers=AUDIT_LAYERS,
            dp=2, tp=1, pcp=4, dcp=4, ep=8, num_devices=8,
        )
    except ModelingError as exc:
        if "v41_replicated_cache_requires_dcp_1" not in str(exc):
            raise RuntimeError(f"unexpected_v41_pcp_dcp4_error:{exc}") from exc
        par_checks["PCP_V41_C2_DCP4_Refused"] = {"refused": True, "error": str(exc)}
    else:
        raise RuntimeError("v41_pcp_dcp4_must_be_refused")

    raw_s = compose_case(
        "qwen3.8-flash-next", hardware="a5", feature_track="unmerged_pr",
        tp=4, dp=2, ep=8, num_devices=8, layers=AUDIT_LAYERS, mtp_proposal_tokens=1,
    )
    struct_doc = export_structure(
        raw_s, phase="prefill", tokens=1024, context=0, sequences=1,
        scope="model", draft_prepare=True,
    )
    try:
        from mindie_profiler.sim_structure_ingress import apply_structure_ingress  # type: ignore[import-not-found]
        ingress_report: dict[str, Any] = {}
        apply_structure_ingress(ingress_report, (), structure=struct_doc, sim_case=raw_s)
        if not ingress_report:
            raise RuntimeError("profiler_ingress_returned_empty")
        si = ingress_report.get("structure_ingress")
        sm = ingress_report.get("structure_mapping")
        ses = ingress_report.get("subgraph_efficiency_summary")
        si_status = str(si.get("status", "")) if isinstance(si, dict) else ""
        sm_status = str(sm.get("status", "")) if isinstance(sm, dict) else ""
        if si_status in {"blocked", "unavailable"} or sm_status in {"blocked", "unavailable"}:
            profiler_status = f"blocked:{si_status or sm_status}"
        elif not isinstance(si, dict) or not si.get("declared_execution_preserved"):
            raise RuntimeError(f"profiler_ingress_execution_not_preserved:{si}")
        elif not isinstance(ses, dict) or int(ses.get("subgraph_count", 0) or 0) <= 0:
            raise RuntimeError(f"profiler_ingress_zero_subgraphs:{ses}")
        elif int(ses.get("evaluated_count", 0) or 0) > 0:
            profiler_status = "verified"
        else:
            profiler_status = "ingested_structure_only"
    except ImportError:
        profiler_status = "blocked:mindie_profiler_not_installed"

    return {
        "base_profile_count": len(profiles),
        "derived_skus": derived_checked,
        "parallel_modes": par_checks,
        "structure_subgraphs": len(struct_doc.get("subgraphs") or ()),
        "profiler_ingress_status": profiler_status,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", default="auto",
                        help="worker count or 'auto' (resolved via runtime_advisor)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    expected_keys = _expected_inventory_keys()
    actual_keys = set(ALL_CATALOG_MODELS)
    if actual_keys != expected_keys:
        raise RuntimeError(
            f"catalog_inventory_mismatch:missing={sorted(expected_keys - actual_keys)}:"
            f"extra={sorted(actual_keys - expected_keys)}"
        )
    total = len(ALL_CATALOG_MODELS)
    if total < MIN_CATALOG_MODELS:
        raise RuntimeError(f"catalog_model_count_below_floor:{total}<{MIN_CATALOG_MODELS}")

    extra_error: str | None = None
    try:
        extra = _verify_derived_skus_and_parallelism()
    except Exception as exc:
        extra_error = f"{type(exc).__name__}:{exc}"
        extra = {"profiler_ingress_status": "failed:derived_or_parallel_error"}

    workers = resolve_jobs(args.workers, task_count=total)
    if workers == 1:
        raw_rows = [_audit_single_model_safe(m) for m in ALL_CATALOG_MODELS]
    else:
        rows_by_model: dict[str, dict[str, Any]] = {}
        pool_error: str | None = None
        pool: ProcessPoolExecutor | None = None
        try:
            pool = ProcessPoolExecutor(max_workers=workers)
            futures = {pool.submit(_audit_single_model_safe, m): m for m in ALL_CATALOG_MODELS}
            for fut in as_completed(futures):
                m = futures[fut]
                try:
                    rows_by_model[m] = fut.result()
                except Exception as exc:
                    rows_by_model[m] = {
                        "model": m,
                        "status": "failed",
                        "error": f"{type(exc).__name__}:{exc}",
                    }
        except Exception as exc:
            pool_error = f"{type(exc).__name__}:{exc}"
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        raw_rows = [
            rows_by_model.get(
                m,
                {
                    "model": m,
                    "status": "failed",
                    "error": pool_error or "worker_pool_terminated_before_completion",
                },
            )
            for m in ALL_CATALOG_MODELS
        ]

    model_rows: list[dict[str, Any]] = []
    for expected_model, row in zip(ALL_CATALOG_MODELS, raw_rows, strict=True):
        if not isinstance(row, dict) or row.get("model") != expected_model:
            reported = row.get("model") if isinstance(row, dict) else None
            model_rows.append({
                "model": expected_model,
                "reported_model": reported,
                "status": "failed",
                "error": f"model_identity_mismatch:expected={expected_model}:got={reported!r}",
            })
        else:
            model_rows.append(row)

    passed = sum(1 for row in model_rows if row.get("status") == "passed")
    failed = sum(1 for row in model_rows if row.get("status") == "failed")
    blocked = sum(1 for row in model_rows if row.get("status") == "blocked")
    profiler_blocked = 1 if str(extra.get("profiler_ingress_status", "")).startswith("blocked:") else 0
    total_blocked = blocked + profiler_blocked
    total_failed = failed + (1 if extra_error else 0)
    identities_exact = [row.get("model") for row in model_rows] == list(ALL_CATALOG_MODELS)
    overall_pass = (
        identities_exact
        and passed + failed + blocked == total
        and total_failed == 0
        and total_blocked == 0
    )

    summary: dict[str, Any] = {
        "status": "pass" if overall_pass else "fail",
        "accounting": {
            "total": total,
            "expected_inventory_count": len(expected_keys),
            "passed": passed,
            "failed": failed,
            "blocked": total_blocked,
            "model_blocked": blocked,
            "profiler_ingress_blocked": profiler_blocked,
            "derived_and_parallel_error": extra_error,
        },
        "coverage": {
            "declared_layers_total": sum(int(r.get("declared_layers", 0)) for r in model_rows),
            "distinct_layer_signatures_total": sum(int(r.get("total_layer_signatures", 0)) for r in model_rows),
            "distinct_layer_signatures_covered": sum(int(r.get("covered_layer_signatures", 0)) for r in model_rows),
        },
        "workers_used": workers,
        "models_verified": passed,
        "tracks": list(TRACKS),
        "primary_skus": list(PRIMARY_SKUS),
        **extra,
        "wall_seconds": round(time.perf_counter() - t0, 3),
        "models": model_rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(
            (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, sort_keys=True))
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
