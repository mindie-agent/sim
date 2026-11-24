"""Profiler subgraph plan lowering, observation normalization, dispatch, and HostQueue leases.

[Role] Bridge ``mindie-sim`` onboarding cliff plans with ``mindie-profiler``
  execution: lower L1/L2/L3 cases into executable ``mindie-bench-subgraph-plan/1``
  (including real 1-layer ``default_spine()`` L3 steps, L2 ``attention +
  v_expand`` subgraphs, and unpadded L1 ``matmul`` / ``grouped_matmul`` cliffs),
  normalize real ``mindie-profiler`` JSONL/receipt outputs into sealed
  onboarding rows (preserving ``kv_store(seq_len=1)`` safety on L3 and GEMM /
  ``grouped_gemm`` operand dtypes/quant modes), invoke ``mindie-profiler`` bench
  subprocesses, resolve strict profiler builders, and manage ``HostQueue`` leases.
[Invariants]
  * ``extract_shape_dims`` resolves the primary ``attention``/``indexer`` node
    before falling back to ``node_shapes``, preventing auxiliary nodes such as
    ``kv_store`` (``seq_len=1``) from clobbering L3 ``context_len``.
  * Verifies physical executed geometry (``sequences``, ``q_len``, ``block_table``,
    ``seq_len``, GEMM/``grouped_gemm`` ``(m, n, k, matrix_groups, dtype,
    weight_dtype, quant_mode)``) before sealing capture digests.
  * ``TimeoutError`` always propagates before ``OSError``; ``HostQueue`` leases
    always release in ``finally`` and report release failures via
    ``receipt_warning``.
[Timeouts & Locks] Bounded subprocesses; ``lru_cache`` on pure 1-layer lowering;
  zero local locks or device contexts.
[Side-Effect Policy] Writes only caller-requested capture files and ephemeral
  temp files cleaned on exit.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from mindie_sim.schema.common import ModelingError

from .contracts import (
    DEFAULT_CLEANUP_GRACE_S, DEFAULT_CLOCK_OPERATOR, DEFAULT_GATE_SUBPROCESS_TIMEOUT_S, NS_PER_MS,
    STANDARD_LEVEL_BOUNDARY_KINDS, VALID_CLOCK_OPERATORS, assert_sanitized_payload,
    canonical_json_bytes, read_bounded_json, required_ranks_for_plan, sanitize_lease_for_receipt,
    sanitize_public_alias, sanitize_public_capture_uri, sanitize_public_run_id,
    sanitize_public_warning, sha256_bytes, subprocess_window_kwargs, validate_parallel_topology,
    write_atomic_bytes, write_atomic_json,
)
from .whitebox_runner import resolve_sku_profile, temporary_scaffold_profile

_DEFAULT_ATTENTION_HEADS = 16
_INDEXER_ATTENTION_HEADS = 32
_MLA_LATENT_DIM = 512
_MLA_ROPE_DIM = 64
_MLA_HEAD_DIM = 128
_MLA_BLOCK_SIZE = 128
_MLA_HIDDEN_WIDTH = 2048
_MLA_BF16_BYTES = 2
_MOE_TOPK_EXPERTS = 6
_MOE_EXPERT_BANK = 64
_MOE_GATE_UP_WIDTH = 2816
_DEFAULT_HEAD_DIM = 128
_DEFAULT_HIDDEN_WIDTH = 4096
_DEFAULT_CONTEXT_FALLBACK = 2048
_L2_CONTEXT_MIN = 24576
_L3_CONTEXT_MIN = 200000
_MIN_COLLECTIVE_BYTES = 4
_DEFAULT_COLLECTIVE_BYTES = 4096
_RING_ALLREDUCE_FACTOR = 2
_MIN_DCP_PARTIALS = 2
_MIN_GEMM_RANK = 3
_MIN_2D_RANK = 2
_BLOCK_TABLE_PAIR_LEN = 2
_LOWER_CACHE_MAXSIZE = 64
_P50_Q = 0.50
_P99_Q = 0.99
_COORDINATOR_RPC_TIMEOUT_S = 45
_DEFAULT_LEASE_TTL_S = 600
_PARTIAL_ABI_EXIT_CODE = 2
_SUBGRAPH_PLAN_SCHEMA = "mindie-bench-subgraph-plan/1"
_OBSERVATION_SCHEMA = "mindie-bench-observation/1"
_RECEIPT_SCHEMA = "mindie-bench-receipt/1"
_SUBGRAPH_RUN_SCHEMA = "mindie-bench-subgraph-run/1"
_ARCH35_MARKER = "35"
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")
_TRIAL_SUFFIX_RE = re.compile(r"\.[^.]+\.t\d+$")
_L2_REGIME_TAGS = frozenset({"high_batch_decode_2d", "mtp_decode_2d", "chunked_prefill_2d"})
_SPARSE_INDEXER_FAMILIES = frozenset({"sparse_indexer", "lightning_indexer", "indexer"})
_GEMM_FAMILIES = frozenset({"gemm", "grouped_gemm", "gate_up_projection", "down_projection"})
_ABI_REJECT_STATUSES = frozenset({"unsupported_binding", "abi_rejected", "unmodeled"})
_BUSY_UNION_MODES = frozenset({"stream_saturated", "busy_union", "event_busy_union", "eager_device", "aclgraph_device"})
_VALID_LEASE_OUTCOMES = frozenset({"", "success", "ok"})
_VALID_LEASE_STATUSES = frozenset({"", "ready", "granted", "acquired", "active", "ok"})
_CAPTURE_SHAPE_KEYS = frozenset({
    "tokens", "sequences", "batch_size", "q_len", "query_tokens", "seq_len", "kv_len", "context_len",
    "page_count", "block_size", "block_table", "heads", "kv_heads", "head_dim", "latent", "rope", "width",
    "top_k", "topk", "sparse_count", "m", "n", "k", "e", "matrix_groups", "partial_count", "nbytes", "group_size",
    "dtype", "input_dtype", "weight_dtype", "output_dtype", "quant_mode", "quantization", "scale_dtype", "has_scales",
    "query_dtype", "cache_dtype", "query_layout", "input_layout", "paged", "parallel_topology", "batch_x_kv_heads",
})


def parse_topo_options(topo: str) -> dict[str, int]:
    return {k: int(v) for k, v in _TOPO_TOKEN_RE.findall(str(topo or "tp1").lower())} or {"tp": 1}


def assign_plan_point_level(plan_doc: Mapping[str, Any], pt: Mapping[str, Any]) -> str:
    explicit = str(pt.get("level") or "").strip().upper()
    if explicit in STANDARD_LEVEL_BOUNDARY_KINDS:
        return explicit
    if str(plan_doc.get("family") or "").strip() != "sku_suite" or str(pt.get("split") or "").strip().lower() != "heldout":
        return "L1"
    topo, tag, val = str(pt.get("parallel_topology") or "tp1").strip().lower(), str(pt.get("regime_tag") or ""), int(pt.get("value") or 0)
    if topo != "tp1" or (tag == "off_grid_heldout" and val >= _L3_CONTEXT_MIN):
        return "L3"
    if tag in _L2_REGIME_TAGS or (tag == "off_grid_heldout" and _L2_CONTEXT_MIN <= val < _L3_CONTEXT_MIN):
        return "L2"
    return "L1"


def _page_count(kv_len: int, block_size: int = _MLA_BLOCK_SIZE) -> int:
    return max(1, (int(kv_len) + int(block_size) - 1) // int(block_size))


def _decode_mla_recipe_for_sku(sku: str, *, profile_override: Mapping[str, Any] | None = None) -> str:
    prof = resolve_sku_profile(sku, profile_override)
    if prof is None:
        raise ModelingError(f"unknown_sku_profile_for_lowering:{sku}")
    dev = prof.get("device") if isinstance(prof.get("device"), Mapping) else {}
    ca = dev.get("compute_arch") if isinstance(dev.get("compute_arch"), Mapping) else {}
    preset = str(ca.get("preset") or "").strip()
    if not preset:
        raise ModelingError(f"sku_profile_missing_arch_preset:{sku}")
    return "ascend_paged_mla_decode_arch35" if _ARCH35_MARKER in preset else "ascend_paged_mla_decode_arch22"


@lru_cache(maxsize=_LOWER_CACHE_MAXSIZE)
def _lower_one_layer_cached(sku: str, prof_sha: str, topo: str, phase: str, q_tok: int, ctx_len: int, seqs: int) -> tuple[dict[str, Any], ...]:
    """Unplugged socket: hardcoded default model override and silent bf16 fallback purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: lower caller-supplied sim_case directly (never hardcoded default model) "
        "and require explicit non-empty op.dtype (zero silent bf16 fallback)"
    )


def _lower_one_layer_step_ops(sku: str, topo: str, phase: str, q_tok: int, ctx_len: int, seqs: int, *, profile_override: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], ...]:
    with temporary_scaffold_profile(sku, profile_override):
        prof = resolve_sku_profile(sku, profile_override)
        if not isinstance(prof, Mapping):
            raise ModelingError(f"unknown_sku_profile_for_lowering:{sku}")
        return _lower_one_layer_cached(sku, sha256_bytes(canonical_json_bytes(prof)), topo, phase, q_tok, ctx_len, seqs)


def _lowered_op_to_profiler_node(pid: str, op_rec: Mapping[str, Any], *, attn_shape: Mapping[str, Any], total_tokens: int, local_heads: int, topo_opts: Mapping[str, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Unplugged socket: fake fallback elementwise stub, shadow _DTYPE_BYTES, and silent bf16 fallbacks purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: resolve profiler builders strictly via ProbedAbiRegistry / "
        "resolve_profiler_builder_strict, raise UnmodeledCapabilityError for unknown families "
        "(zero fake elementwise stub), use schema.common.dtype_bytes, and require explicit dtypes"
    )


def _resolve_family_shape(
    pbld: str, *, tok_val: int, width: int, tp: int, ep: int,
    local_heads: int, topo: str, topo_opts: dict[str, int],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Build the shape dict and adjusted topo options for a registered family binding."""
    if pbld == "matmul":
        return (
            {"tokens": tok_val, "m": tok_val, "n": width, "k": max(1, width // tp), "dtype": "bf16", "parallel_topology": topo},
            {**topo_opts, "graph_capture_mode": "none"},
        )
    if pbld == "grouped_matmul":
        loc_e = max(1, _MOE_EXPERT_BANK // ep)
        return (
            {"tokens": tok_val, "m": tok_val * _MOE_TOPK_EXPERTS, "n": max(1, _MOE_GATE_UP_WIDTH // tp), "k": width, "e": loc_e, "matrix_groups": loc_e, "dtype": "bf16", "parallel_topology": topo},
            {**topo_opts, "graph_capture_mode": "none"},
        )
    if pbld == "rotary_embedding":
        return (
            {"tokens": tok_val, "heads": local_heads, "head_dim": _DEFAULT_HEAD_DIM, "parallel_topology": topo},
            topo_opts,
        )
    return {"tokens": tok_val, "width": width, "parallel_topology": topo}, topo_opts


def resolve_point_binding_and_shape(
    plan_doc: Mapping[str, Any], pt: Mapping[str, Any], *, sku: str, profile_override: Mapping[str, Any] | None = None,
) -> tuple[str, str, str, str, str, dict[str, Any], dict[str, Any]]:
    """Unplugged socket: AG014 model-name width dispatch, _FAMILY_BINDINGS whitelist, and hardcoded default models purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: query canonical ProbedAbiRegistry (schema/abi.py) for (op_family, arch, dtypes), "
        "derive width structurally from caller-supplied sim_case/plan_doc, and never use hardcoded model defaults"
    )


def _build_case_nodes(pid: str, *, sku: str, pt: Mapping[str, Any], level: str, pfam: str, prec: str, pbld: str, impl: str, shape: Mapping[str, Any], topo_opts: Mapping[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Unplugged socket: L3 lowering ignoring caller sim_case purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: lower caller-supplied sim_case directly for L3 case nodes and "
        "require explicit operator dtypes from lowered OpSpec"
    )


def resolve_point_measured_binding(plan_doc: Mapping[str, Any], pt: Mapping[str, Any], *, sku: str, profile_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Unplugged socket: depends on purged point binding and silent bf16 GEMM dtype fallback."""
    raise NotImplementedError(
        "LEGACY_PURGED: resolve measured binding via ProbedAbiRegistry and caller sim_case "
        "with explicit non-empty GEMM dtype (zero silent bf16 fallback)"
    )


def build_profiler_subgraph_plan(plan_doc: Mapping[str, Any], *, sku: str, split_tag: str = "", acquisition_id: str = "", profile_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if str(plan_doc.get("schema") or "") == _SUBGRAPH_PLAN_SCHEMA:
        return {**dict(plan_doc), "sku": sku, **({"acquisition_id": acquisition_id} if acquisition_id else {})}
    world = required_ranks_for_plan(plan_doc)
    plan_sha = str(plan_doc["plan_sha256"])
    cases: list[dict[str, Any]] = []
    with temporary_scaffold_profile(sku, profile_override):
        for split_key in ("train_points", "heldout_points", "abi_rejection_points"):
            for pt in plan_doc.get(split_key) or ():
                if not isinstance(pt, Mapping) or not pt.get("point_id"):
                    continue
                pid, topo = str(pt["point_id"]), str(pt.get("parallel_topology") or "tp1")
                pt_ranks, topo_opts = validate_parallel_topology(topo), parse_topo_options(topo)
                pfam, prec, pbld, impl, _sf, shape, _sc = resolve_point_binding_and_shape(plan_doc, pt, sku=sku, profile_override=profile_override)
                level = assign_plan_point_level(plan_doc, pt)
                nodes, edges, _nb, _tids = _build_case_nodes(pid, sku=sku, pt=pt, level=level, pfam=pfam, prec=prec, pbld=pbld, impl=impl, shape=shape, topo_opts=topo_opts)
                entry: dict[str, Any] = {
                    "case_id": pid, "split": str(pt.get("split") or "heldout"), "level": level, "boundary": STANDARD_LEVEL_BOUNDARY_KINDS[level],
                    "phase": "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode", "stage": "s1_joint" if pt_ranks > 1 else "s0_local",
                    "clock_operator": str(pt.get("clock_operator") or DEFAULT_CLOCK_OPERATOR), "implementation": impl, "dtype": "bf16",
                    "shape": dict(shape), "batch": int(shape.get("sequences", pt.get("batch_size", 1))),
                    "plan_key": {"point_id": pid, "axis": str(pt.get("axis") or "context_len"), "family": pfam, "recipe": prec, "abi": impl, "parallel_topology": topo, "level": level, "batch_size": int(pt.get("batch_size", 1))},
                    "nodes": nodes, "edges": edges,
                }
                if pbld in {"absorbed_mla", "fused_infer_attention"}:
                    entry["paged"] = True
                if shape.get("query_layout"):
                    entry["layout"] = str(shape["query_layout"])
                if world > 1:
                    entry["group_ranks"] = list(range(pt_ranks))
                if pt_ranks > 1:
                    entry["parallel"] = topo_opts
                cases.append(entry)
    return {"schema": _SUBGRAPH_PLAN_SCHEMA, "plan_id": f"{plan_sha}:{split_tag}" if split_tag else plan_sha, "plan_sha256": plan_sha, "target_id": str(plan_doc.get("target_id") or ""), "sku": sku, "stage": "s1_joint" if world > 1 else "s0_local", "world_size": world, "execution_modes": ["eager"], "cases": cases, **({"acquisition_id": acquisition_id} if acquisition_id else {})}


def _int_from(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for k in keys:
        v = mapping.get(k)
        if v is not None and not isinstance(v, bool):
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
    return None


def _norm_op_key(raw: str, cid: str) -> str:
    s = str(raw or "").strip()
    return s.split(":", 1)[1].strip() if s.startswith(f"{cid}:") else s


def _primary_geometry_source(actual_shape: Mapping[str, Any], measured_binding: Mapping[str, Any] | None, *, cid: str) -> Mapping[str, Any]:
    raw_ns = actual_shape.get("node_shapes")
    if not isinstance(raw_ns, Mapping):
        return actual_shape
    norm_ns = {_norm_op_key(str(k), cid): v for k, v in raw_ns.items() if isinstance(v, Mapping)}
    if isinstance(measured_binding, Mapping):
        for nb in measured_binding.get("node_bindings") or ():
            if isinstance(nb, Mapping) and str(nb.get("family") or "") in {"attention", "indexer"}:
                oid = str(nb.get("op_id") or "").strip()
                if oid in norm_ns:
                    return norm_ns[oid]
    for nval in norm_ns.values():
        if any(nval.get(mk) is not None for mk in ("heads", "kv_heads", "block_table", "top_k", "sparse_count", "paged")):
            return nval
    return actual_shape


def extract_shape_dims(
    shape: Mapping[str, Any], plan_key: Mapping[str, Any], *, plan_pt: Mapping[str, Any] | None = None,
    expected_shape: Mapping[str, Any] | None = None, measured_binding: Mapping[str, Any] | None = None, cid: str = "",
) -> dict[str, Any]:
    """Extract canonical onboarding coordinates from executed shape without ``kv_store(seq_len=1)`` clobber."""
    dims: dict[str, Any] = {}
    axis = str((plan_pt or {}).get("axis") or plan_key.get("axis") or "").strip()
    eff = dict(shape)
    primary = _primary_geometry_source(shape, measured_binding, cid=cid or "case")
    if primary is not shape:
        eff.update(primary)
    dims["query_tokens"] = _int_from(eff, "q_len", "query_tokens", "q") or _int_from(plan_key, "q_len", "query_tokens", "q") or 1
    ctx_exp = _int_from(eff, "context_len", "context", "kv_len_prior")
    seq_v = _int_from(eff, "seq_len", "kv_len", "scanned_rows")
    exp_seq, exp_ctx = _int_from(expected_shape or {}, "seq_len", "kv_len"), _int_from(expected_shape or {}, "context_len")
    if seq_v is not None:
        dims["kv_len"] = seq_v
    if ctx_exp is not None:
        dims["context_len"] = ctx_exp
    elif seq_v is not None and exp_seq is not None and exp_ctx is not None and seq_v == exp_seq:
        dims["context_len"] = exp_ctx
    elif seq_v is not None:
        dims["context_len"] = seq_v
    elif (pk_ctx := _int_from(plan_key, "context_len", "seq_len", "context")) is not None:
        dims["context_len"] = pk_ctx
    tok_exp, m_v = _int_from(eff, "tokens", "num_tokens"), _int_from(eff, "m")
    exp_m, exp_tok = _int_from(expected_shape or {}, "m"), _int_from(expected_shape or {}, "tokens")
    tok_v = tok_exp if tok_exp is not None else (exp_tok if (m_v is not None and exp_m is not None and exp_tok is not None and m_v == exp_m) else (m_v if m_v is not None else _int_from(plan_key, "tokens", "m", "num_tokens")))
    if tok_v is not None:
        dims["tokens"] = tok_v
    seq_cnt = _int_from(eff, "sequences", "batch_size", "batch", "b")
    kv_h = _int_from(eff, "kv_heads")
    bxkv = _int_from(eff, "batch_x_kv_heads") or (seq_cnt * kv_h if (seq_cnt is not None and kv_h is not None and axis == "batch_x_kv_heads") else None)
    if bxkv is not None:
        dims["batch_x_kv_heads"] = bxkv
    if seq_cnt is not None:
        dims["sequences"] = seq_cnt
    dims["batch_size"] = (_int_from(plan_pt or {}, "batch_size") or _int_from(plan_key, "batch_size") or 1) if axis == "batch_x_kv_heads" else (seq_cnt or _int_from(plan_key, "batch_size", "sequences") or 1)
    if (tk := _int_from(eff, "sparse_count", "top_k", "topk")) is not None:
        dims["topk"] = tk
    dims["parallel_topology"] = str(eff.get("parallel_topology") or plan_key.get("parallel_topology") or (plan_pt or {}).get("parallel_topology") or "tp1")
    return dims


def _assert_geo_eq(cid: str, field: str, expected: Any, actual: Any) -> None:
    if expected is not None and actual is not None and actual != expected:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:{field}:expected={expected}:actual={actual}")


def _verify_gemm_actual_fields(n_act: Mapping[str, Any], mb: Mapping[str, Any], *, cid: str, oid: str) -> None:
    for dim in ("m", "n", "k"):
        _assert_geo_eq(cid, f"{dim}:{oid}", _int_from(mb, dim), _int_from(n_act, dim))
    _assert_geo_eq(cid, f"matrix_groups:{oid}", _int_from(mb, "matrix_groups") or 1, _int_from(n_act, "matrix_groups", "e"))
    exp_dt = str(mb.get("dtype") or "").strip().lower()
    exp_wdt = str(mb.get("weight_dtype") or exp_dt).strip().lower()
    exp_q = str(mb.get("quant_mode") or "none").strip().lower()
    got_dt = str(n_act.get("dtype") or n_act.get("input_dtype") or "").strip().lower()
    got_wdt = str(n_act.get("weight_dtype") or "").strip().lower()
    got_q = str(n_act.get("quant_mode") or n_act.get("quantization") or "").strip().lower()
    _assert_geo_eq(cid, f"dtype:{oid}", exp_dt or None, got_dt or None)
    _assert_geo_eq(cid, f"weight_dtype:{oid}", exp_wdt or None, got_wdt or None)
    if got_q and got_q not in {"", "none", "dense", exp_q, exp_dt}:
        raise ModelingError(f"profiler_actual_geometry_mismatch:{cid}:quant_mode:{oid}:expected={exp_q}:actual={got_q}")


def verify_actual_geometry_matches_plan(
    actual_shape: Mapping[str, Any], plan_pt: Mapping[str, Any], *, expected_shape: Mapping[str, Any], cid: str, measured_binding: Mapping[str, Any] | None = None,
) -> None:
    if not actual_shape:
        raise ModelingError(f"profiler_actual_shape_missing:{cid}")
    geo = _primary_geometry_source(actual_shape, measured_binding, cid=cid)
    exp_seqs = _int_from(expected_shape, "sequences")
    _assert_geo_eq(cid, "q_len", max(1, int(plan_pt.get("query_tokens", 1))), _int_from(geo, "q_len", "query_tokens"))
    _assert_geo_eq(cid, "sequences", exp_seqs, _int_from(geo, "sequences"))
    bt = geo.get("block_table")
    if isinstance(bt, Sequence) and not isinstance(bt, (str, bytes)) and len(bt) >= _BLOCK_TABLE_PAIR_LEN and type(bt[0]) is int:
        _assert_geo_eq(cid, "block_table_sequences", exp_seqs, int(bt[0]))
    _assert_geo_eq(cid, "seq_len", _int_from(expected_shape, "seq_len", "kv_len"), _int_from(geo, "seq_len", "kv_len"))
    if isinstance(measured_binding, Mapping):
        raw_ns = actual_shape.get("node_shapes")
        norm_ns = {_norm_op_key(str(k), cid): v for k, v in raw_ns.items() if isinstance(v, Mapping)} if isinstance(raw_ns, Mapping) else {}
        if measured_binding.get("node_bindings"):
            for nb in measured_binding["node_bindings"]:
                if isinstance(nb, Mapping) and str(nb.get("family") or "") in _GEMM_FAMILIES and str(nb.get("op_id") or "") in norm_ns:
                    oid = str(nb["op_id"])
                    _verify_gemm_actual_fields(norm_ns[oid], nb, cid=cid, oid=oid)
        elif str(measured_binding.get("family") or "") in _GEMM_FAMILIES:
            _verify_gemm_actual_fields(geo, measured_binding, cid=cid, oid=str(measured_binding.get("target_op_id") or "op0"))


def _quantile_ns(sorted_ns: Sequence[int], q: float) -> int:
    if not sorted_ns:
        return 0
    if len(sorted_ns) == 1:
        return int(sorted_ns[0])
    pos = q * (len(sorted_ns) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return int(sorted_ns[lo]) if lo == hi else int(round(sorted_ns[lo] * (1.0 - (pos - lo)) + sorted_ns[hi] * (pos - lo)))


def _sanitize_capture_shape(raw: Any) -> Any:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        ks = str(k)
        if ks == "node_shapes" and isinstance(v, Mapping):
            out["node_shapes"] = {str(nk): _sanitize_capture_shape(nv) for nk, nv in v.items() if isinstance(nv, Mapping)}
        elif ks in _CAPTURE_SHAPE_KEYS:
            out[ks] = v
    return out


def normalize_profiler_records(
    raw_records: Sequence[Mapping[str, Any]], *, plan_doc: Mapping[str, Any], sku: str,
    endpoint_alias: str = "", capture_out_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize raw ``mindie-profiler`` JSONL/receipt rows into sealed onboarding rows & ABI rows."""
    if not raw_records:
        return [], []
    is_prof = any(str(r.get("schema") or "") in {_OBSERVATION_SCHEMA, _RECEIPT_SCHEMA, _SUBGRAPH_RUN_SCHEMA} or r.get("record_type") in {"observation", "case_complete", "receipt"} for r in raw_records)
    if not is_prof:
        return [dict(r) for r in raw_records if str(r.get("status") or "").lower() not in _ABI_REJECT_STATUSES], [
            {"point_id": str(r.get("point_id") or r.get("case_id") or ""), "rejected": True, "abi_error": str(r.get("abi_error") or r.get("reason") or ""), "executed_shape": dict(r.get("executed_shape") or r.get("actual_shape") or {})}
            for r in raw_records if str(r.get("status") or "").lower() in _ABI_REJECT_STATUSES
        ]
    plan_pts = {str(p["point_id"]): p for sk in ("train_points", "heldout_points") for p in (plan_doc.get(sk) or ()) if isinstance(p, Mapping) and p.get("point_id")}
    buckets: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    receipts: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    abi_rows: list[dict[str, Any]] = []
    for rec in raw_records:
        cid = str(rec.get("case_id") or rec.get("point_id") or "").strip()
        status = str(rec.get("status") or "ok").strip().lower()
        rtype = str(rec.get("record_type") or "").strip().lower()
        if status in _ABI_REJECT_STATUSES:
            abi_rows.append({"point_id": cid, "rejected": True, "abi_error": str(rec.get("reason") or rec.get("abi_error") or ""), "executed_shape": dict(rec.get("executed_shape") or rec.get("actual_shape") or {})})
            continue
        split = str(rec.get("split") or (plan_pts.get(cid) or {}).get("split") or "heldout").strip().lower()
        if rtype == "receipt" or str(rec.get("schema") or "") == _SUBGRAPH_RUN_SCHEMA:
            receipts.setdefault((cid, split), []).append(rec)
        elif rtype == "observation" or rec.get("duration_ms") is not None:
            if status not in {"", "ok", "passed", "pass"}:
                raise ModelingError(f"profiler_observation_status_invalid:{cid}:status={status}")
            buckets.setdefault((cid, split), []).append(rec)

    clean_ep = sanitize_public_alias(endpoint_alias or sku, sku)
    plan_family = str(plan_doc.get("family") or "").strip()
    out_rows: list[dict[str, Any]] = []
    for (cid, split), group in sorted(buckets.items()):
        plan_pt = plan_pts.get(cid)
        binfo = resolve_point_measured_binding(plan_doc, plan_pt, sku=sku) if plan_pt is not None else None
        rcpts = receipts.get((cid, split), [])
        # Extract executed shape + top-level node_shapes (matching profiler@57dc39c1 _with_node_shapes)
        actual_shape: dict[str, Any] = {}
        for src in (*rcpts, *group):
            cand = src.get("actual_shape") or src.get("executed_shape")
            if isinstance(cand, Mapping) and cand:
                actual_shape = dict(cand)
            if isinstance(src.get("node_shapes"), Mapping) and src["node_shapes"]:
                actual_shape["node_shapes"] = dict(src["node_shapes"])
            if actual_shape:
                break
        if plan_pt is not None and binfo is not None:
            verify_actual_geometry_matches_plan(actual_shape, plan_pt, expected_shape=binfo["expected_shape"], cid=cid, measured_binding=binfo["measured_binding"])
        raise NotImplementedError(
            "LEGACY_PURGED: fail closed in normalize_profiler_records on unrecognized "
            "clock_operator, missing physical rank identity, or missing event_ids "
            "(zero silent rank-0 / DEFAULT_CLOCK_OPERATOR / synthetic event_ids)"
        )
    assert_sanitized_payload(out_rows, field_name="profiler_normalized_rows")
    return out_rows, abi_rows


def validate_host_queue_grant(lease_info: Mapping[str, Any] | None, *, required_devices: int, min_ttl_seconds: int) -> tuple[str, int, list[int], int]:
    req_npus = int(required_devices)
    info = dict(lease_info or {})
    lease_id, fence = str(info.get("lease_id") or "").strip(), info.get("fence_token")
    if req_npus < 1 or not lease_id or type(fence) is not int or fence <= 0:
        raise ModelingError("host_queue_lease_missing_lease_id_or_positive_fence_token")
    raw_devs = info.get("devices")
    if not isinstance(raw_devs, Sequence) or isinstance(raw_devs, (str, bytes)):
        raise ModelingError("host_queue_lease_missing_granted_devices")
    devs = [int(d) for d in raw_devs if type(d) is int and d >= 0]
    if len(devs) != len(raw_devs) or len(set(devs)) != req_npus:
        raise ModelingError(f"host_queue_lease_device_count_mismatch:expected={req_npus}:granted={devs}")
    ttl = info.get("ttl_seconds")
    if type(ttl) is not int or ttl < max(1, int(min_ttl_seconds)):
        raise ModelingError(f"host_queue_lease_ttl_below_subprocess_plus_grace:ttl={ttl!r}:required={min_ttl_seconds}")
    return lease_id, fence, devs, ttl


def _run_coordinator_lease_cmd(*, action: str, endpoint_alias: str, payload: Mapping[str, Any], timeout_s: int, coordinator_bin: str = "") -> dict[str, Any]:
    exe = coordinator_bin.strip() or shutil.which("mindie-coordinator") or ""
    if not exe:
        raise ModelingError("coordinator_cli_not_found:mindie-coordinator")
    cmd = [exe, "lease", "--action", action, "--alias", endpoint_alias, "--json", json.dumps(dict(payload), sort_keys=True)]
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False, **subprocess_window_kwargs())
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"coordinator_lease_{action}_timeout:timeout_s={timeout_s}") from exc
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"coordinator_lease_{action}_os_error:{exc}") from exc
    if done.returncode != 0:
        raise ModelingError(f"coordinator_lease_{action}_failed:rc={done.returncode}")
    parsed = json.loads(done.stdout or "{}")
    if not isinstance(parsed, Mapping):
        raise ModelingError(f"coordinator_lease_{action}_non_mapping_response")
    return dict(parsed)


@contextmanager
def acquire_coordinator_host_queue_lease(
    endpoint_alias: str, *, npu_count: int = 1, devices: Sequence[int] | None = None,
    ttl_seconds: int = _DEFAULT_LEASE_TTL_S, min_subprocess_timeout_s: int = 0,
    cleanup_grace_s: int = DEFAULT_CLEANUP_GRACE_S, rpc_timeout_s: int = _COORDINATOR_RPC_TIMEOUT_S, coordinator_bin: str = "",
) -> Iterator[dict[str, Any]]:
    alias, req_npus, eff_ttl = endpoint_alias.strip(), int(npu_count), int(ttl_seconds)
    min_req = int(min_subprocess_timeout_s) + int(cleanup_grace_s) if int(min_subprocess_timeout_s) > 0 else 1
    if not alias or req_npus < 1 or eff_ttl < min_req:
        raise ModelingError(f"host_queue_lease_invalid_request:npu={req_npus}:ttl={eff_ttl}:min={min_req}")
    req_pl = {"npu_count": req_npus, "ttl_seconds": eff_ttl, "purpose": "mindie-sim-onboarding", **({"devices": [int(d) for d in devices]} if devices else {})}
    acq = _run_coordinator_lease_cmd(action="acquire", endpoint_alias=alias, payload=req_pl, timeout_s=int(rpc_timeout_s), coordinator_bin=coordinator_bin)
    data = acq.get("data") if isinstance(acq.get("data"), Mapping) else acq
    lease_id, fence = str(data.get("lease_id") or "").strip(), data.get("fence_token")
    rec: dict[str, Any] = {"lease_mode": "coordinator_host_queue", "lease_id": lease_id, "fence_token": fence if type(fence) is int else 0, "devices": [], "ttl_seconds": eff_ttl, "receipt_unconfirmed": False, "receipt_warning": ""}
    if bool(acq.get("receipt_unconfirmed") or data.get("receipt_unconfirmed")) or acq.get("receipt_warning") or data.get("receipt_warning"):
        rec["receipt_unconfirmed"] = True
        rec["receipt_warning"] = sanitize_public_warning(acq.get("receipt_warning") or data.get("receipt_warning") or "host_queue_lease_acquire_unconfirmed")
    try:
        if str(acq.get("outcome") or "").lower() not in _VALID_LEASE_OUTCOMES or str(acq.get("status") or "").lower() not in _VALID_LEASE_STATUSES:
            raise ModelingError("host_queue_lease_acquire_rejected")
        _, _, devs, val_ttl = validate_host_queue_grant({**dict(data), "lease_id": lease_id, "fence_token": fence, "ttl_seconds": data.get("ttl_seconds", eff_ttl)}, required_devices=req_npus, min_ttl_seconds=min_req)
        rec["devices"], rec["ttl_seconds"] = devs, val_ttl
        yield rec
    finally:
        if lease_id and type(fence) is int and fence > 0:
            try:
                _run_coordinator_lease_cmd(action="release", endpoint_alias=alias, payload={"lease_id": lease_id, "fence_token": fence}, timeout_s=int(rpc_timeout_s), coordinator_bin=coordinator_bin)
            except TimeoutError:
                raise
            except (ModelingError, OSError) as exc:
                rec["receipt_unconfirmed"] = True
                rec["receipt_warning"] = sanitize_public_warning(f"host_queue_lease_release_unconfirmed:{exc}")


def run_profiler_bench_subprocess(
    plan_doc: Mapping[str, Any], *, sku: str, profiler_root: str | Path | None = None,
    endpoint_alias: str = "", allowed_phy_ids: Sequence[int] = (), fence_token: int | None = None,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S, capture_out_dir: str | Path | None = None,
    profile_override: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute ``mindie_profiler bench subgraph`` locally and normalize observations & ABI rejections."""
    sg_plan = build_profiler_subgraph_plan(plan_doc, sku=sku, profile_override=profile_override)
    world = max(1, int(sg_plan.get("world_size") or 1))
    env = dict(os.environ)
    if profiler_root is not None:
        env["PYTHONPATH"] = str(Path(profiler_root).resolve()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if allowed_phy_ids:
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(int(d)) for d in allowed_phy_ids)
    if fence_token is not None:
        env["MINDIE_LEASE_FENCE_TOKEN"] = str(int(fence_token))
    raw_records: list[Mapping[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="msim-prof-") as tmp_dir:
        tmp = Path(tmp_dir)
        plan_f = write_atomic_json(tmp / "plan.json", sg_plan)
        obs_f, out_f = tmp / "obs.jsonl", tmp / "out.json"
        for rk in range(world):
            rk_env = {**env, "RANK": str(rk), "WORLD_SIZE": str(world)}
            rk_obs = tmp / f"obs.rank{rk}.jsonl" if world > 1 else obs_f
            try:
                done = subprocess.run([sys.executable, "-B", "-m", "mindie_profiler", "bench", "subgraph", "--plan", str(plan_f), "--observations", str(rk_obs), "--out", str(out_f)], env=rk_env, capture_output=True, text=True, timeout=timeout_s, check=False, **subprocess_window_kwargs())
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"profiler_bench_timeout:{timeout_s}s") from exc
            except TimeoutError:
                raise
            except OSError as exc:
                raise ModelingError(f"profiler_bench_os_error:{exc}") from exc
            if done.returncode not in (0, _PARTIAL_ABI_EXIT_CODE):
                raise ModelingError(f"profiler_bench_failed:rc={done.returncode}:{(done.stderr or '').strip()}")
            if rk_obs.is_file():
                for line in rk_obs.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        raw_records.append(json.loads(line))
            rk_out = Path(f"{out_f}.rank{rk}.json") if world > 1 else out_f
            if rk_out.is_file():
                doc = json.loads(rk_out.read_text(encoding="utf-8"))
                for rec in doc.get("records") or ():
                    if isinstance(rec, Mapping):
                        raw_records.append({**dict(rec), "schema": _SUBGRAPH_RUN_SCHEMA, "record_type": "receipt", "run_id": doc.get("run_id", ""), "rank": rk})
    return normalize_profiler_records(raw_records, plan_doc=plan_doc, sku=sku, endpoint_alias=endpoint_alias, capture_out_dir=capture_out_dir)


def acquire_stage3_observations(
    plan_doc: Mapping[str, Any], *, sku: str, raw_source: str | Path | Sequence[Mapping[str, Any]] | None = None,
    custom_runner: Any = None, live_profiler: bool = False, endpoint_alias: str = "",
    profiler_root: str | Path | None = None, coordinator_bin: str = "", capture_root: str | Path | None = None,
    profile_override: Mapping[str, Any] | None = None, abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
    lease_acquirer: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], int]:
    """Acquire and normalize Stage-3 silicon rows via custom runner, live profiler + HostQueue, or offline replay."""
    req_npus = required_ranks_for_plan(plan_doc)
    lease_pub: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    abi_obs = [dict(r) for r in (abi_rejection_results or ()) if isinstance(r, Mapping)]
    if custom_runner is not None:
        raw_out = list(custom_runner(plan_doc))
        rows, norm_abi = normalize_profiler_records(raw_out, plan_doc=plan_doc, sku=sku, endpoint_alias=endpoint_alias, capture_out_dir=capture_root)
        abi_obs = abi_obs or norm_abi
    elif live_profiler:
        min_ttl = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S + DEFAULT_CLEANUP_GRACE_S
        lctx = lease_acquirer(sku, min_ttl) if lease_acquirer else acquire_coordinator_host_queue_lease(endpoint_alias, npu_count=req_npus, ttl_seconds=min_ttl, min_subprocess_timeout_s=DEFAULT_GATE_SUBPROCESS_TIMEOUT_S, coordinator_bin=coordinator_bin)
        with lctx as linfo:
            _, fence, devs, _ = validate_host_queue_grant(linfo, required_devices=req_npus, min_ttl_seconds=min_ttl)
            rows, norm_abi = run_profiler_bench_subprocess(plan_doc, sku=sku, profiler_root=profiler_root, endpoint_alias=endpoint_alias, allowed_phy_ids=devs, fence_token=fence, capture_out_dir=capture_root, profile_override=profile_override)
            abi_obs = abi_obs or norm_abi
        lease_pub = sanitize_lease_for_receipt(linfo)
    elif raw_source is not None:
        raw_obs = read_bounded_json(raw_source) if isinstance(raw_source, (str, Path)) else raw_source
        raw_list = list(raw_obs.get("cases") or raw_obs.get("observations") or ()) if isinstance(raw_obs, Mapping) else list(raw_obs)
        rows, norm_abi = normalize_profiler_records(raw_list, plan_doc=plan_doc, sku=sku, endpoint_alias=endpoint_alias, capture_out_dir=capture_root)
        abi_obs = abi_obs or norm_abi
    return rows, abi_obs, lease_pub, req_npus


def resolve_profiler_builder_strict(family: str, recipe: str, impl: str, *, dtypes: Sequence[str] = (), meta: Mapping[str, Any] | None = None, profiler_root: str | Path | None = None) -> tuple[str, str]:
    added_path, saved = "", {}
    if profiler_root is not None:
        rstr = str(Path(profiler_root).resolve())
        for mn in list(sys.modules):
            if mn == "mindie_profiler" or mn.startswith("mindie_profiler."):
                saved[mn] = sys.modules.pop(mn)
        if rstr not in sys.path:
            sys.path.insert(0, rstr)
            added_path = rstr
    try:
        try:
            from mindie_profiler.bench.operators import BenchUnmodeledOperatorError, resolve_builder_strict
        except ImportError as exc:
            raise ModelingError(f"profiler_owner_unavailable:{exc}") from exc
        try:
            b, abi = resolve_builder_strict(family, recipe, impl, dtypes=tuple(dtypes), meta=meta or {})
        except BenchUnmodeledOperatorError as exc:
            raise ModelingError(f"profiler_builder_unmodeled:{exc}") from exc
        return str(b), str(abi)
    finally:
        if added_path and added_path in sys.path:
            sys.path.remove(added_path)
        if profiler_root is not None:
            for mn in list(sys.modules):
                if mn == "mindie_profiler" or mn.startswith("mindie_profiler."):
                    sys.modules.pop(mn, None)
            sys.modules.update(saved)
