"""Executable ``mindie-bench-subgraph-plan/1`` builder and writer-clock/geometry helpers.

[Role] Translate onboarding cliff/SKU plans into executable
  ``mindie-bench-subgraph-plan/1`` plans with exact ``ParallelSpec`` world
  sizes, canonical ``mindie-profiler`` operator bindings + tensor geometry, and
  ``sim_case`` bindings; also parse ``run_id.<arm>.<mode>.t<trial>`` suffixes,
  derive ``clock_operator`` strictly from writer timing evidence, filter
  multi-mode writer observations without collisions, and extract executed
  geometry.
[Invariants]
  * Reuses ``required_ranks_for_plan`` / ``validate_parallel_topology`` (never
    divergent world-size arithmetic).
  * Never exports unmodeled ``family="sku_suite"`` nodes; every point maps to a
    real executable binding + full tensor geometry.
  * Never overwrites writer-measured ``clock_operator`` with the plan's clock;
    fails closed with ``clock_operator_plan_mismatch`` when no writer mode
    matches.
[Timeouts & Locks] Pure in-memory plan/record helpers; zero I/O, locks, or
  subprocesses.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from mindie_sim.schema.common import ModelingError

from .contracts import VALID_CLOCK_OPERATORS

_DEFAULT_ATTENTION_HEADS = 16
_DEFAULT_ATTENTION_KV_HEADS = 2
_DEFAULT_HEAD_DIM = 128
_DEFAULT_HIDDEN_WIDTH = 4096
_DEFAULT_CONTEXT_FALLBACK = 2048
_DEFAULT_SIM_MODEL = "deepseek-v2-lite"
_INDEXER_SIM_MODEL = "deepseek-v4.1-flash"
_SUBGRAPH_PLAN_SCHEMA = "mindie-bench-subgraph-plan/1"
_TRIAL_SUFFIX_RE = re.compile(r"\.[^.]+\.t\d+$")
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")
_KNOWN_ARMS = frozenset({"base", "aclgraph"})
_BUSY_UNION_TIMING_MODES = frozenset({
    "stream_saturated", "busy_union", "event_busy_union", "eager_device", "aclgraph_device",
})
_ELAPSED_SPAN_TIMING_MODES = frozenset({"eager", "aclgraph", "elapsed_span", "event_elapsed_span"})
_BUSY_UNION_BOUNDARIES = frozenset({"busy_union", "event_busy_union"})
_SPARSE_INDEXER_FAMILIES = frozenset({"sparse_indexer", "lightning_indexer", "indexer"})
_FAMILY_BINDINGS: Mapping[str, tuple[str, str, str, str]] = {
    "rmsnorm": ("rmsnorm", "rmsnorm", "rms_norm", "aclnnRmsNorm"),
    "add_rmsnorm": ("add_rmsnorm", "add_rmsnorm", "add_rms_norm", "aclnnAddRmsNorm"),
    "layernorm": ("layernorm", "layernorm", "layer_norm", "aclnnLayerNorm"),
    "attention": ("attention", "fused_infer_attention", "fused_infer_attention", "aclnnFusedInferAttentionScoreV3"),
    "sparse_indexer": ("indexer", "quant_lightning_indexer", "lightning_indexer", "aclnnQuantLightningIndexerV2"),
    "lightning_indexer": ("indexer", "quant_lightning_indexer", "lightning_indexer", "aclnnQuantLightningIndexerV2"),
    "indexer": ("indexer", "quant_lightning_indexer", "lightning_indexer", "aclnnQuantLightningIndexerV2"),
    "topk": ("topk", "index_topk", "indexer_topk", "aclnnTopkV2"),
    "swiglu": ("swiglu", "swiglu", "swiglu", "aclnnSwiGlu"),
    "geglu": ("geglu", "geglu", "geglu", "aclnnGeGluV2"),
    "rotary": ("rotary", "rotary", "rotary_embedding", "aclnnApplyRotaryPosEmbV2"),
    "gemm": ("gemm", "linear", "matmul", "aclnnMatmul"),
    "matmul": ("gemm", "linear", "matmul", "aclnnMatmul"),
    "linear": ("gemm", "linear", "matmul", "aclnnMatmul"),
    "grouped_gemm": ("grouped_gemm", "grouped_matmul", "grouped_matmul", "aclnnGroupedMatmulV5"),
    "routing": ("moe_init_routing", "moe_init_routing_v3", "moe_init_routing", "aclnnMoeInitRoutingV3"),
    "moe_init_routing": ("moe_init_routing", "moe_init_routing_v3", "moe_init_routing", "aclnnMoeInitRoutingV3"),
    "cast": ("cast", "cast", "dtype_cast", "aclnnCast"),
    "elementwise": ("elementwise", "residual_add", "vector_add", "aclnnAdd"),
}


def quantile_ns(sorted_ns: Sequence[int], q: float) -> int:
    """Compute linear-interpolated quantile over sorted integer ns samples."""
    if not sorted_ns:
        return 0
    if len(sorted_ns) == 1:
        return int(sorted_ns[0])
    pos = q * (len(sorted_ns) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return int(sorted_ns[lo])
    return int(round(sorted_ns[lo] * (1.0 - (pos - lo)) + sorted_ns[hi] * (pos - lo)))


def parse_run_id_and_arm(
    run_id: str,
    timing_mode: str = "",
    trial: Any = None,
    source: str = "",
    execution: Mapping[str, Any] | None = None,
) -> tuple[str, str]:
    """Strip ``.<arm>.<mode>.t<trial>`` or ``.<mode>.t<trial>`` to recover ``(base_run_id, arm)``."""
    clean = str(run_id or "run-0").strip()
    if timing_mode and trial is not None and not isinstance(trial, bool):
        suffix = f".{timing_mode}.t{int(trial)}"
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)]
    else:
        clean = _TRIAL_SUFFIX_RE.sub("", clean)
    arm = ""
    for candidate in _KNOWN_ARMS:
        marker = f".{candidate}"
        if clean.endswith(marker):
            clean = clean[: -len(marker)]
            arm = candidate
            break
    if not arm and isinstance(execution, Mapping):
        if str(execution.get("graph") or "").strip().lower() == "aclgraph":
            arm = "aclgraph"
    if not arm and source:
        prefix = source.rsplit("/", 1)[-1].split(".", 1)[0].strip().lower()
        if prefix in _KNOWN_ARMS:
            arm = prefix
    return clean or "run-0", arm or "base"


def extract_shape_dims(shape: Mapping[str, Any], plan_key: Mapping[str, Any]) -> dict[str, Any]:
    """Extract canonical integer onboarding coordinates from executed shape + plan key."""
    dims: dict[str, Any] = {}
    merged = {**dict(plan_key), **dict(shape)}
    for dst, candidates in (
        ("context_len", ("context_len", "context", "kv_len", "seq_len", "scanned_rows")),
        ("tokens", ("tokens", "m", "num_tokens")),
        ("batch_size", ("batch_size", "batch", "b", "sequences")),
        ("query_tokens", ("query_tokens", "q_len", "q")),
        ("topk", ("topk", "top_k", "topk_items")),
        ("batch_x_kv_heads", ("batch_x_kv_heads",)),
    ):
        for key in candidates:
            val = merged.get(key)
            if val is not None and not isinstance(val, bool):
                try:
                    dims[dst] = int(val)
                    break
                except (TypeError, ValueError):
                    continue
    dims.setdefault("batch_size", 1)
    dims.setdefault("query_tokens", 1)
    dims["parallel_topology"] = str(merged.get("parallel_topology") or "tp1")
    return dims


def parse_topo_options(topo: str) -> dict[str, int]:
    """Parse canonical topology tokens into ``compose_options``."""
    return {k: int(v) for k, v in _TOPO_TOKEN_RE.findall(str(topo or "tp1").lower())} or {"tp": 1}


def resolve_point_binding_and_shape(
    plan_doc: Mapping[str, Any],
    pt: Mapping[str, Any],
    *,
    sku: str,
) -> tuple[str, str, str, str, str, dict[str, Any], dict[str, Any]]:
    """Return ``(prof_fam, prof_recipe, prof_builder, impl, sim_fam, shape, sim_case)``."""
    plan_family = str(plan_doc.get("family") or "").strip() or "rmsnorm"
    axis = str(pt.get("axis") or "context_len").strip()
    val = int(pt["value"])
    batch = max(1, int(pt.get("batch_size", 1)))
    q_tok = max(1, int(pt.get("query_tokens", 1)))
    topo = str(pt.get("parallel_topology") or "tp1").strip()
    topk_val = int(pt["topk"]) if pt.get("topk") is not None else None
    topo_opts = parse_topo_options(topo)

    if topk_val is not None or plan_family in _SPARSE_INDEXER_FAMILIES:
        pfam, prec, pbld, impl = _FAMILY_BINDINGS["indexer"]
        sim_fam, model_name = "indexer", _INDEXER_SIM_MODEL
        ctx_len = val if axis == "context_len" else max(val, _DEFAULT_CONTEXT_FALLBACK)
        shape: dict[str, Any] = {
            "tokens": max(1, batch * q_tok), "width": _DEFAULT_HEAD_DIM,
            "heads": _DEFAULT_ATTENTION_HEADS, "kv_heads": 1, "head_dim": _DEFAULT_HEAD_DIM,
            "seq_len": ctx_len, "context_len": ctx_len,
            "batch_size": batch, "sequences": batch, "query_tokens": q_tok, "q_len": q_tok,
            "parallel_topology": topo,
        }
        if topk_val is not None:
            shape["top_k"] = topk_val
            shape["topk"] = topk_val
    elif plan_family == "sku_suite":
        model_name = _DEFAULT_SIM_MODEL
        if axis in {"context_len", "batch_x_kv_heads"}:
            pfam, prec, pbld, impl = (
                ("attention", "flash_attention_prefill", "fusion_attention", "aclnnFlashAttentionVarLenScoreV2")
                if q_tok > 1 else _FAMILY_BINDINGS["attention"]
            )
            sim_fam = "attention"
            ctx_len = val if axis == "context_len" else _DEFAULT_CONTEXT_FALLBACK
            shape = {
                "tokens": max(1, batch * q_tok),
                "heads": _DEFAULT_ATTENTION_HEADS, "kv_heads": _DEFAULT_ATTENTION_KV_HEADS,
                "head_dim": _DEFAULT_HEAD_DIM, "seq_len": ctx_len, "context_len": ctx_len,
                "batch_size": batch, "sequences": val if axis == "batch_x_kv_heads" else batch,
                "query_tokens": q_tok, "q_len": q_tok, "parallel_topology": topo,
            }
            if axis == "batch_x_kv_heads":
                shape["batch_x_kv_heads"] = val
        else:
            pfam, prec, pbld, impl = _FAMILY_BINDINGS["rmsnorm"]
            sim_fam = "rmsnorm"
            shape = {
                "tokens": val, "width": _DEFAULT_HIDDEN_WIDTH,
                "batch_size": batch, "sequences": batch,
                "query_tokens": q_tok, "q_len": q_tok, "parallel_topology": topo,
            }
    elif plan_family in _FAMILY_BINDINGS:
        pfam, prec, pbld, impl = _FAMILY_BINDINGS[plan_family]
        sim_fam = pfam
        model_name = _INDEXER_SIM_MODEL if sim_fam in {"indexer", "layernorm", "rotary"} else _DEFAULT_SIM_MODEL
        ctx_len = val if axis == "context_len" else _DEFAULT_CONTEXT_FALLBACK
        tok_val = val if axis == "tokens" else max(1, batch * q_tok)
        shape = {
            "tokens": tok_val, "width": _DEFAULT_HIDDEN_WIDTH,
            "m": tok_val, "n": _DEFAULT_HIDDEN_WIDTH, "k": _DEFAULT_HIDDEN_WIDTH,
            "heads": _DEFAULT_ATTENTION_HEADS, "kv_heads": _DEFAULT_ATTENTION_KV_HEADS,
            "head_dim": _DEFAULT_HEAD_DIM, "seq_len": ctx_len,
            "batch_size": batch, "sequences": val if axis == "batch_x_kv_heads" else batch,
            "query_tokens": q_tok, "q_len": q_tok, "parallel_topology": topo, axis: val,
        }
    else:
        pfam, prec, pbld = plan_family, plan_family, ""
        impl, sim_fam, model_name = str(plan_doc.get("target_id") or plan_family), plan_family, _DEFAULT_SIM_MODEL
        shape = {
            "tokens": val if axis == "tokens" else max(1, batch * q_tok),
            "width": _DEFAULT_HIDDEN_WIDTH, "batch_size": batch, "sequences": batch,
            "query_tokens": q_tok, "q_len": q_tok, "parallel_topology": topo, axis: val,
        }

    custom_sim = pt.get("sim_case") or plan_doc.get("sim_case")
    sim_case = (
        dict(custom_sim)
        if isinstance(custom_sim, Mapping)
        else {"compose_model": model_name, "compose_hardware": sku, "compose_options": topo_opts}
    )
    return pfam, prec, pbld, impl, sim_fam, shape, sim_case


def build_profiler_subgraph_plan(
    plan_doc: Mapping[str, Any], *, sku: str, split_tag: str = ""
) -> dict[str, Any]:
    """Translate an onboarding cliff/SKU plan into executable ``mindie-bench-subgraph-plan/1``."""
    from .cliff_planner import required_ranks_for_plan, validate_parallel_topology

    if str(plan_doc.get("schema") or "") == _SUBGRAPH_PLAN_SCHEMA:
        out = dict(plan_doc)
        out.setdefault("plan_id", str(plan_doc.get("plan_sha256") or plan_doc.get("plan_id") or "onboard"))
        out.setdefault("plan_sha256", str(plan_doc.get("plan_sha256") or ""))
        out.setdefault("sku", sku)
        return out

    world = required_ranks_for_plan(plan_doc)
    plan_sha = str(plan_doc["plan_sha256"])
    plan_id = f"{plan_sha}:{split_tag}" if split_tag else plan_sha
    cases: list[dict[str, Any]] = []
    for split_key in ("train_points", "heldout_points", "abi_rejection_points"):
        for pt in plan_doc.get(split_key) or ():
            if not isinstance(pt, Mapping) or not pt.get("point_id"):
                continue
            pid = str(pt["point_id"])
            topo = str(pt.get("parallel_topology") or "tp1")
            pt_ranks = validate_parallel_topology(topo)
            pfam, prec, pbld, impl, _sim_fam, shape, _sim_case = resolve_point_binding_and_shape(
                plan_doc, pt, sku=sku
            )
            node: dict[str, Any] = {
                "node_id": f"{pid}:op0", "kind": "operator",
                "op": pbld or prec, "family": pfam, "recipe": prec,
                "implementation": impl, "dtype": "bf16", "shape": dict(shape),
            }
            if pbld:
                node["builder"] = pbld
            case_entry: dict[str, Any] = {
                "case_id": pid, "split": str(pt.get("split") or "heldout"), "level": "operator",
                "phase": "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode",
                "archetype": str(pt.get("regime_tag") or "onboarding"),
                "stage": "s1_joint" if pt_ranks > 1 else "s0_local",
                "clock_operator": str(pt.get("clock_operator") or "elapsed_span"),
                "implementation": impl, "dtype": "bf16", "shape": dict(shape),
                "plan_key": {"point_id": pid, "family": pfam, "recipe": prec, "abi": impl, "parallel_topology": topo},
                "nodes": [node], "edges": [],
            }
            if pt_ranks > 1:
                case_entry["group_ranks"] = list(range(pt_ranks))
                case_entry["parallel"] = parse_topo_options(topo)
            cases.append(case_entry)
    return {
        "schema": _SUBGRAPH_PLAN_SCHEMA, "plan_id": plan_id, "plan_sha256": plan_sha,
        "target_id": str(plan_doc.get("target_id") or ""), "sku": sku,
        "stage": "s1_joint" if world > 1 else "s0_local", "world_size": world,
        "execution_modes": ["eager"], "cases": cases,
    }


def derive_writer_clock_operator(
    group: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    cid: str,
    tmode: str,
) -> str:
    """Derive clock operator strictly from profiler writer timing evidence (never plan)."""
    explicit: set[str] = set()
    for row in group:
        exec_map = row.get("execution") if isinstance(row.get("execution"), Mapping) else {}
        ident_map = row.get("identity") if isinstance(row.get("identity"), Mapping) else {}
        ident_exec = ident_map.get("execution") if isinstance(ident_map.get("execution"), Mapping) else {}
        for cand in (
            row.get("clock_operator"), exec_map.get("clock_operator"),
            ident_map.get("clock_operator"), ident_exec.get("clock_operator"),
            summary.get("clock_operator"),
        ):
            val = str(cand or "").strip()
            if val:
                if val not in VALID_CLOCK_OPERATORS:
                    raise ModelingError(f"profiler_writer_clock_operator_invalid:{cid}:{val}")
                explicit.add(val)
    if len(explicit) > 1:
        raise ModelingError(f"profiler_writer_clock_operator_conflict:{cid}:{sorted(explicit)}")
    if explicit:
        return next(iter(explicit))
    boundary_name = str(group[0].get("boundary") or "device_kernel").strip().lower()
    mode_low = str(tmode or "").strip().lower()
    if boundary_name in _BUSY_UNION_BOUNDARIES or mode_low in _BUSY_UNION_TIMING_MODES:
        return "busy_union"
    if mode_low in _ELAPSED_SPAN_TIMING_MODES:
        return "elapsed_span"
    return "busy_union" if boundary_name == "device_kernel" else "elapsed_span"


def extract_observed_shape(summary: Mapping[str, Any], first: Mapping[str, Any]) -> dict[str, Any]:
    """Return the richest non-placeholder executed/requested shape emitted by the writer."""
    ident = first.get("identity") if isinstance(first.get("identity"), Mapping) else {}
    key_map = first.get("key") if isinstance(first.get("key"), Mapping) else {}
    for cand in (
        summary.get("actual_shape"), summary.get("requested_shape"),
        first.get("actual_shape"), first.get("requested_shape"),
        ident.get("shape"), key_map.get("shape"),
    ):
        if isinstance(cand, Mapping) and cand and dict(cand) != {"declared": True}:
            return dict(cand)
    return {}


def select_mode_candidates(
    candidates: Sequence[tuple[str, str, str, list[Mapping[str, Any]]]],
    *,
    cid: str,
    plan_pt: Mapping[str, Any] | None,
) -> list[tuple[str, str, str, list[Mapping[str, Any]]]]:
    """Filter multi-mode groups to the plan's clock/timing mode; fail closed on mismatch."""
    plan_clk = str((plan_pt or {}).get("clock_operator") or "").strip()
    if not plan_clk:
        return list(candidates)
    matched = [item for item in candidates if item[2] == plan_clk]
    if not matched:
        found = sorted({item[2] for item in candidates})
        raise ModelingError(
            f"clock_operator_plan_mismatch:{cid}:plan={plan_clk}:measured={','.join(found)}"
        )
    if len(matched) == 1:
        return matched
    pref_mode = "eager" if plan_clk == "elapsed_span" else "stream_saturated"
    scored = sorted(
        matched,
        key=lambda it: (0 if it[0] == "base" else 1, 0 if it[1].lower() == pref_mode else 1, it[0], it[1]),
    )
    return [scored[0]]


__all__ = [
    "build_profiler_subgraph_plan",
    "derive_writer_clock_operator",
    "extract_observed_shape",
    "extract_shape_dims",
    "parse_run_id_and_arm",
    "parse_topo_options",
    "quantile_ns",
    "resolve_point_binding_and_shape",
    "select_mode_candidates",
]
