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
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any

from mindie_sim.export.operator_matrix import (
    is_onboarding_family,
    resolve_builder_strict as _canonical_resolve_builder_strict,
    resolve_onboarding_builder_and_abi,
)
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError, dtype_bytes

from .contracts import (
    DEFAULT_CLEANUP_GRACE_S, DEFAULT_CLOCK_OPERATOR, DEFAULT_GATE_SUBPROCESS_TIMEOUT_S, NS_PER_MS,
    STANDARD_LEVEL_BOUNDARY_KINDS, VALID_CLOCK_OPERATORS, assert_sanitized_payload,
    canonical_json_bytes, read_bounded_json, required_ranks_for_plan, sanitize_lease_for_receipt,
    sanitize_public_alias, sanitize_public_capture_uri, sanitize_public_run_id,
    sanitize_public_warning, sha256_bytes, subprocess_window_kwargs, validate_parallel_topology,
    write_atomic_bytes, write_atomic_json,
)
from .whitebox_runner import resolve_sku_profile, temporary_scaffold_profile

_DEFAULT_ATTENTION_HEADS, _INDEXER_ATTENTION_HEADS = 16, 32
_MLA_LATENT_DIM, _MLA_ROPE_DIM, _MLA_HEAD_DIM, _MLA_BLOCK_SIZE = 512, 64, 128, 128
_MLA_HIDDEN_WIDTH, _MLA_BF16_BYTES, _MLA_DENSE_INTERMEDIATE, _MLA_MOE_INTERMEDIATE = 2048, 2, 10944, 1408
_MLA_SHARED_EXPERTS, _MLA_PROBE_LAYERS, _MLA_PROBE_VOCAB = 2, 2, 102400
_MLA_INDEX_Q_LORA_RANK, _MLA_INDEX_DEFAULT_TOPK, _MLA_FIRST_K_DENSE_REPLACE = 1536, 2048, 1
_MOE_TOPK_EXPERTS, _MOE_EXPERT_BANK, _MOE_GATE_UP_WIDTH = 6, 64, 2816
_DEFAULT_HEAD_DIM, _DEFAULT_HIDDEN_WIDTH, _DEFAULT_CONTEXT_FALLBACK = 128, 4096, 2048
_L2_CONTEXT_MIN, _L3_CONTEXT_MIN = 24576, 200000
_MIN_COLLECTIVE_BYTES, _DEFAULT_COLLECTIVE_BYTES, _RING_ALLREDUCE_FACTOR = 4, 4096, 2
_MIN_DCP_PARTIALS, _MIN_GEMM_RANK, _MIN_2D_RANK, _BLOCK_TABLE_PAIR_LEN = 2, 3, 2, 2
_LOWER_CACHE_MAXSIZE, _P50_Q, _P99_Q = 64, 0.50, 0.99
_COORDINATOR_RPC_TIMEOUT_S, _DEFAULT_LEASE_TTL_S, _PARTIAL_ABI_EXIT_CODE = 45, 600, 2
_SUBGRAPH_PLAN_SCHEMA = "mindie-bench-subgraph-plan/1"
_BENCH_PLAN_V2_SCHEMA = "mindie-bench-plan/2"
_OBSERVATION_SCHEMA = "mindie-bench-observation/1"
_OBSERVATION_V2_SCHEMA = "mindie-bench-observation/2"
_RECEIPT_SCHEMA = "mindie-bench-receipt/1"
_SUBGRAPH_RUN_SCHEMA = "mindie-bench-subgraph-run/1"
_ARCH35_MARKER = "35"
_ARCH_A3_PREFIX = "Ascend910_93"
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")
_TRIAL_SUFFIX_RE = re.compile(r"\.[^.]+\.t\d+$")
_L2_REGIME_TAGS = frozenset({"high_batch_decode_2d", "mtp_decode_2d", "chunked_prefill_2d"})
_SPARSE_INDEXER_FAMILIES = frozenset({"sparse_indexer", "lightning_indexer", "indexer"})
_GEMM_FAMILIES = frozenset({"gemm", "grouped_gemm", "gate_up_projection", "down_projection"})
_RESIDUAL_ADD_RECIPES = frozenset({"residual_add", "vector_add", "add"})
_COLLECTIVE_FAMILIES = frozenset({"dcp_allgather", "allgather", "allreduce", "alltoall", "reduce_scatter"})
_DCP_PACK_RECIPES = frozenset({"dcp_query_pack", "dcp_output_pack"})
_ABI_REJECT_STATUSES = frozenset({"unsupported_binding", "abi_rejected", "unmodeled"})
_BUSY_UNION_MODES = frozenset({"stream_saturated", "busy_union", "event_busy_union"})
_HOST_FALLBACK_MODES = frozenset({"host_fallback", "perf_counter_host_fallback"})
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


def _arch_for_sku(sku: str, *, profile_override: Mapping[str, Any] | None = None) -> str:
    prof = resolve_sku_profile(sku, profile_override)
    if prof is None:
        raise ModelingError(f"unknown_sku_profile_for_lowering:{sku}")
    arch_raw = str(prof.get("architecture") or "").strip().upper()
    if arch_raw in {"A2", "A3", "A5"}:
        return arch_raw
    dev = prof.get("device") if isinstance(prof.get("device"), Mapping) else {}
    ca = dev.get("compute_arch") if isinstance(dev.get("compute_arch"), Mapping) else {}
    preset = str(ca.get("preset") or "").strip()
    if not preset:
        raise ModelingError(f"sku_profile_missing_arch_preset:{sku}")
    if _ARCH35_MARKER in preset:
        return "A5"
    return "A3" if str(sku).startswith(_ARCH_A3_PREFIX) else "A2"


def _structural_mla_probe_config(*, indexed: bool = False, topk: int | None = None, hidden_size: int | None = None) -> dict[str, Any]:
    base = {"model_type": "structural_mla", "intermediate_size": _MLA_DENSE_INTERMEDIATE, "kv_lora_rank": _MLA_LATENT_DIM, "qk_nope_head_dim": _MLA_HEAD_DIM, "qk_rope_head_dim": _MLA_ROPE_DIM, "v_head_dim": _MLA_HEAD_DIM, "vocab_size": _MLA_PROBE_VOCAB, "torch_dtype": "bfloat16"}
    if indexed:
        return {**base, "name": "structural-indexed-mla-probe", "hidden_size": int(hidden_size or _DEFAULT_HIDDEN_WIDTH), "num_hidden_layers": 1, "num_attention_heads": _INDEXER_ATTENTION_HEADS, "num_key_value_heads": _INDEXER_ATTENTION_HEADS, "q_lora_rank": _MLA_INDEX_Q_LORA_RANK, "_probe_index_topk": int(topk or _MLA_INDEX_DEFAULT_TOPK)}
    return {**base, "name": "structural-mla-probe", "hidden_size": int(hidden_size or _MLA_HIDDEN_WIDTH), "num_hidden_layers": _MLA_PROBE_LAYERS, "num_attention_heads": _DEFAULT_ATTENTION_HEADS, "num_key_value_heads": 1, "n_routed_experts": _MOE_EXPERT_BANK, "n_shared_experts": _MLA_SHARED_EXPERTS, "num_experts_per_tok": _MOE_TOPK_EXPERTS, "moe_intermediate_size": _MLA_MOE_INTERMEDIATE, "first_k_dense_replace": _MLA_FIRST_K_DENSE_REPLACE}


def _refresh_probe_single_spine(case_dict: dict[str, Any], *, indexed: bool, index_dtype: str) -> dict[str, Any]:
    from mindie_sim.compose import resolve_case_contracts

    rt = dict(case_dict.get("runtime") or {})
    meta = dict(rt.get("metadata") or {})
    ss = dict(meta.get("single_spine") or {})
    ci = ss.get("contract_inputs")
    if not isinstance(ci, Mapping):
        return case_dict
    ci_copy = dict(ci)
    if indexed:
        ci_copy["requested_options"] = {**dict(ci_copy.get("requested_options") or {}), "indexer_kv_dtype": index_dtype}
    ss["contract_inputs"] = ci_copy
    updated = {**case_dict, "runtime": {**rt, "metadata": {**meta, "single_spine": ss}}}
    sc, qp, lp = resolve_case_contracts(updated)
    if qp is not None and lp is not None:
        ss.update({"quant_plan_digest": qp.plan_digest, "quant_provenance": qp.provenance, "kv_cache_abi_count": len(qp.kv_cache_abi), "index_cache_abi_count": len(qp.index_cache_abi), "recurrent_state_abi_count": len(qp.recurrent_state_abi), "prepared_weights": [i.allocation_id for i in qp.prepared_weights], "packed_groups_count": len(qp.packed_groups), "lowering_plan_digest": lp.plan_digest, "allowed_rewrite_tracks": list(lp.allowed_rewrite_tracks), "recipe_bindings_count": len(lp.recipe_bindings), "scenario_contract_digest": sc.digest() if sc is not None else ""})
        updated = {**updated, "runtime": {**rt, "metadata": {**meta, "single_spine": ss}}}
    return updated


def _compose_structural_probe_case(
    sku: str, topo_opts: Mapping[str, Any], *, indexed: bool = False,
    topk: int | None = None, hidden_size: int | None = None,
    single_layer: bool = False, index_dtype: str = "bf16",
    index_head_dim: int | None = None,
) -> dict[str, Any]:
    from mindie_sim.compose import compose_case as compose_case_dict

    cfg = _structural_mla_probe_config(indexed=indexed, topk=topk, hidden_size=hidden_size)
    idx_topk = int(cfg.pop("_probe_index_topk", _MLA_INDEX_DEFAULT_TOPK))
    raw = compose_case_dict(model=cfg, hardware=sku, **dict(topo_opts))
    md = dict(raw["model"])
    if indexed:
        idx_dim = int(index_head_dim or _DEFAULT_HEAD_DIM)
        md["mla"] = {**dict(md.get("mla") or {}), "index_topk": idx_topk, "index_head_dim": idx_dim, "index_n_heads": _INDEXER_ATTENTION_HEADS}
        md["quant"] = {**dict(md.get("quant") or {}), "index_dtype": index_dtype}
        lp0 = dict(md["layer_plans"][0])
        lp0.update({"indexer": True, "index_heads": _INDEXER_ATTENTION_HEADS, "index_dim": idx_dim, "index_topk": idx_topk, "owns_index_k": True, "owns_topk": True})
        md["layer_plans"] = [lp0]
    if single_layer:
        md["layers"], md["body_layers"], md["layer_plans"] = 1, 1, list(md["layer_plans"])[:1]
    out_case = {**raw, "model": md}
    if indexed or single_layer:
        out_case = _refresh_probe_single_spine(out_case, indexed=indexed, index_dtype=index_dtype)
    return out_case


def _materialize_one_layer_case(sku: str, topo: str, sim_case_json: str) -> Any:
    from mindie_sim.compose import compose_case as compose_case_dict
    from mindie_sim.prediction.structure import parse_case

    topo_opts = parse_topo_options(topo)
    if sim_case_json:
        raw_sc = json.loads(sim_case_json)
        if not isinstance(raw_sc, Mapping):
            raise ModelingError("profiler_sim_case_invalid_mapping")
        if "compose_model" in raw_sc:
            opts = {**topo_opts, **dict(raw_sc.get("compose_options") or {})}
            raw = compose_case_dict(model=raw_sc["compose_model"], hardware=str(raw_sc.get("compose_hardware", sku)), **opts)
            overrides = raw_sc.get("model_overrides")
            if isinstance(overrides, Mapping) and overrides:
                from .whitebox_runner import WhiteboxSpineRunner
                raw = WhiteboxSpineRunner._apply_model_overrides(raw, overrides)
        else:
            raw = dict(raw_sc)
        md = dict(raw["model"])
        md["layers"], md["body_layers"], md["layer_plans"] = 1, 1, list(md["layer_plans"])[:1]
        return parse_case({**raw, "model": md})
    return parse_case(_compose_structural_probe_case(sku, topo_opts, single_layer=True))


@lru_cache(maxsize=_LOWER_CACHE_MAXSIZE)
def _lower_one_layer_cached(
    sku: str, prof_sha: str, topo: str, phase: str, q_tok: int, ctx_len: int, seqs: int, sim_case_json: str = "",
) -> tuple[dict[str, Any], ...]:
    del prof_sha
    from mindie_sim.prediction.results import LoweringRequest, StepRequest
    from mindie_sim.prediction.spine import default_spine

    case = _materialize_one_layer_case(sku, topo, sim_case_json)
    prep = default_spine().lower(
        case,
        LoweringRequest(step=StepRequest(phase=phase, tokens=int(q_tok), context=int(ctx_len), sequences=int(seqs), scope="body"), tier="mainline"),  # type: ignore[arg-type]
    )
    out: list[dict[str, Any]] = []
    for op in prep.graph.executable[0].ops:
        op_dt = str(op.dtype or "").strip().lower()
        if not op_dt:
            raise ModelingError(f"lowered_op_missing_dtype:{op.op_id}")
        meta = dict(op.metadata)
        nabi = meta.get("native_attention_abi") if isinstance(meta.get("native_attention_abi"), Mapping) else {}
        pgeo = meta.get("paged_mla_geometry") if isinstance(meta.get("paged_mla_geometry"), Mapping) else {}
        out.append({
            "op_id": str(op.op_id), "family": str(op.family), "recipe": str(op.recipe),
            "dtype": op_dt, "quant_mode": str(meta.get("quant_mode") or meta.get("quantization") or "none").strip().lower(),
            "local_shape": tuple(int(x) if isinstance(x, (int, float)) else x for x in (op.local_shape or ())),
            "deps": tuple(str(d) for d in (op.deps or ())), "link_bytes": int(round(float(op.link_bytes or 0.0))),
            "matrix_groups": int(meta.get("matrix_weight_bank_groups") or meta.get("matrix_groups") or 1),
            "matrix_group_rows": tuple(int(x) for x in meta["matrix_group_rows"]) if isinstance(meta.get("matrix_group_rows"), Sequence) and not isinstance(meta.get("matrix_group_rows"), (str, bytes)) else (),
            "group_axis": str(meta.get("group_axis") or ""),
            "query_layout": str(nabi.get("query_layout") or ""), "cache_layout": str(nabi.get("cache_layout") or ""),
            "output_layout": str(nabi.get("output_layout") or ""),
            "kv_len": int(pgeo["kv_lengths"][0]) if pgeo.get("kv_lengths") else 0,
            "page_count": int(pgeo["page_counts"][0]) if pgeo.get("page_counts") else 0,
            "partial_count": int(meta["partial_count"]) if meta.get("partial_count") is not None else 0,
        })
    return tuple(out)


def _lower_one_layer_step_ops(
    sku: str, topo: str, phase: str, q_tok: int, ctx_len: int, seqs: int,
    *, sim_case: Mapping[str, Any] | None = None, profile_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    with temporary_scaffold_profile(sku, profile_override):
        prof = resolve_sku_profile(sku, profile_override)
        if not isinstance(prof, Mapping):
            raise ModelingError(f"unknown_sku_profile_for_lowering:{sku}")
        sc_json = canonical_json_bytes(sim_case).decode("utf-8") if isinstance(sim_case, Mapping) else ""
        return _lower_one_layer_cached(sku, sha256_bytes(canonical_json_bytes(prof)), topo, phase, q_tok, ctx_len, seqs, sc_json)


def _require_op_dtype(op_rec: Mapping[str, Any], *, oid: str) -> str:
    dt = str(op_rec.get("dtype") or "").strip().lower()
    if not dt:
        raise ModelingError(f"lowered_op_missing_dtype:{oid}")
    return dt


def _lowered_collective_node(
    nid: str, oid: str, fam: str, rec: str, dt: str, op_rec: Mapping[str, Any],
    *, arch: str, topo_opts: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    tp, dcp = max(1, int(topo_opts.get("tp", 1))), max(1, int(topo_opts.get("dcp", 1)))
    pcp = max(1, int(topo_opts.get("pcp", 1)))
    raw_link = int(op_rec.get("link_bytes") or 0)
    grp_axis = "dcp" if fam == "dcp_allgather" else (str(op_rec.get("group_axis") or "") or ("dcp" if "dcp" in rec else ("pcp" if "pcp" in rec else "tp")))
    grp_sz = dcp if grp_axis in {"dcp", "cp"} else (pcp if grp_axis == "pcp" else tp)
    _pf, _pr, bld, impl = resolve_onboarding_builder_and_abi(arch, fam)
    if fam in {"dcp_allgather", "allgather"}:
        nbytes = max(_MIN_COLLECTIVE_BYTES, (raw_link // max(1, grp_sz - 1)) if raw_link > 0 else _DEFAULT_COLLECTIVE_BYTES)
    elif fam == "allreduce":
        nbytes = max(_MIN_COLLECTIVE_BYTES, ((raw_link * grp_sz) // max(1, _RING_ALLREDUCE_FACTOR * (grp_sz - 1))) if (raw_link > 0 and grp_sz > 1) else (raw_link or _DEFAULT_COLLECTIVE_BYTES))
    else:
        nbytes = max(_MIN_COLLECTIVE_BYTES, raw_link or _DEFAULT_COLLECTIVE_BYTES)
    return (
        {"node_id": nid, "op_id": oid, "kind": "collective", "op": bld, "builder": bld, "family": fam, "recipe": rec, "implementation": impl, "dtype": dt, "group": grp_axis, "group_size": grp_sz, "nbytes": nbytes, "shape": {"dtype": dt, "nbytes": nbytes, "group_size": grp_sz}},
        {"op_id": oid, "family": fam, "recipe": rec, "builder": bld, "dtype": dt, "nbytes": nbytes, "group_size": grp_sz},
    )


def _lowered_op_to_profiler_node(
    pid: str, op_rec: Mapping[str, Any], *, attn_shape: Mapping[str, Any], total_tokens: int,
    local_heads: int, topo_opts: Mapping[str, int], arch: str = "A2",
) -> tuple[dict[str, Any], dict[str, Any]]:
    oid, fam, rec, loc = str(op_rec["op_id"]), str(op_rec["family"]), str(op_rec["recipe"]), tuple(op_rec["local_shape"])
    dt = _require_op_dtype(op_rec, oid=oid)
    dcp, nid = max(1, int(topo_opts.get("dcp", 1))), f"{pid}:{oid}"
    if fam == "attention":
        sh = dict(attn_shape)
        if op_rec["kv_len"] > 0:
            sh["seq_len"] = sh["kv_len"] = int(op_rec["kv_len"])
            sh["page_count"] = int(op_rec["page_count"]) or _page_count(int(op_rec["kv_len"]))
        if op_rec["query_layout"]:
            sh["query_layout"] = sh["input_layout"] = op_rec["query_layout"]
            sh["cache_layout"] = op_rec["cache_layout"] or "PA_BSND"
            sh["output_layout"] = op_rec["output_layout"] or "NBSD"
        impl = resolve_onboarding_builder_and_abi(arch, "attention")[3]
        return (
            {"node_id": nid, "op_id": oid, "kind": "operator", "op": "absorbed_mla", "builder": "absorbed_mla", "family": "attention", "recipe": rec, "implementation": impl, "dtype": dt, "paged": True, "layout": sh.get("query_layout", "TND"), "shape": sh},
            {"op_id": oid, "family": "attention", "recipe": rec, "builder": "absorbed_mla", "paged": True, "heads": sh["heads"], "kv_heads": 1, "latent": _MLA_LATENT_DIM, "rope": _MLA_ROPE_DIM, "kv_len": sh["seq_len"], "page_count": sh["page_count"], "query_layout": sh.get("query_layout", "TND")},
        )
    if fam in _GEMM_FAMILIES and len(loc) >= _MIN_GEMM_RANK:
        m_d, n_d, k_d, mg = int(loc[0]), int(loc[1]), int(loc[2]), max(1, int(op_rec["matrix_groups"]))
        qm = str(op_rec.get("quant_mode") or "none").strip().lower()
        if mg > 1:
            impl = resolve_onboarding_builder_and_abi(arch, "grouped_gemm")[3]
            raw_gl = op_rec.get("matrix_group_rows") or ()
            q_base, q_rem = divmod(max(0, m_d), mg)
            glist = [int(x) for x in raw_gl] if (len(raw_gl) == mg and sum(int(x) for x in raw_gl) == max(0, m_d)) else ([q_base + 1] * q_rem + [q_base] * (mg - q_rem))
            return (
                {"node_id": nid, "op_id": oid, "kind": "operator", "op": "grouped_matmul", "builder": "grouped_matmul", "family": "grouped_gemm", "recipe": "grouped_matmul", "implementation": impl, "dtype": dt, "group_list": glist, "group_list_type": 1, "shape": {"m": m_d, "n": n_d, "k": k_d, "e": mg, "matrix_groups": mg, "dtype": dt}},
                {"op_id": oid, "family": fam, "recipe": rec, "builder": "grouped_matmul", "dtype": dt, "quant_mode": qm, "m": m_d, "n": n_d, "k": k_d, "matrix_groups": mg},
            )
        impl = resolve_onboarding_builder_and_abi(arch, "gemm")[3]
        return (
            {"node_id": nid, "op_id": oid, "kind": "operator", "op": "matmul", "builder": "matmul", "family": fam, "recipe": rec, "implementation": impl, "dtype": dt, "shape": {"m": m_d, "n": n_d, "k": k_d, "matrix_groups": 1, "dtype": dt}},
            {"op_id": oid, "family": fam, "recipe": rec, "builder": "matmul", "dtype": dt, "quant_mode": qm, "m": m_d, "n": n_d, "k": k_d, "matrix_groups": 1},
        )
    if fam in {"rmsnorm", "add_rmsnorm", "swiglu"}:
        toks = int(loc[0]) if len(loc) >= _MIN_2D_RANK else total_tokens
        width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
        _pf, _pr, bld, impl = resolve_onboarding_builder_and_abi(arch, fam)
        return {"node_id": nid, "op_id": oid, "kind": "operator", "op": bld, "builder": bld, "family": fam, "recipe": rec, "implementation": impl, "dtype": dt, "shape": {"tokens": toks, "width": width}}, {"op_id": oid, "family": fam, "recipe": rec, "builder": bld, "width": width}
    if fam == "elementwise" and rec == "rotary":
        impl = resolve_onboarding_builder_and_abi(arch, "rotary")[3]
        return {"node_id": nid, "op_id": oid, "kind": "operator", "op": "rotary_embedding", "builder": "rotary_embedding", "family": "elementwise", "recipe": "rotary", "implementation": impl, "dtype": dt, "shape": {"tokens": total_tokens, "heads": local_heads, "head_dim": _MLA_ROPE_DIM}}, {"op_id": oid, "family": "elementwise", "recipe": "rotary", "builder": "rotary_embedding"}
    if fam == "elementwise" and rec in _DCP_PACK_RECIPES:
        _pf, _pr, bld, impl = resolve_onboarding_builder_and_abi(arch, rec)
        width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
        return {"node_id": nid, "op_id": oid, "kind": "operator", "op": bld, "builder": bld, "family": "elementwise", "recipe": rec, "implementation": impl, "dtype": dt, "shape": {"tokens": total_tokens, "width": width}}, {"op_id": oid, "family": "elementwise", "recipe": rec, "builder": bld, "width": width}
    if fam == "elementwise" and rec in _RESIDUAL_ADD_RECIPES:
        width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
        return {"node_id": nid, "op_id": oid, "kind": "operator", "op": "vector_add", "builder": "vector_add", "family": "elementwise", "recipe": rec, "implementation": "torch.add", "dtype": dt, "shape": {"tokens": total_tokens, "width": width}}, {"op_id": oid, "family": "elementwise", "recipe": rec, "builder": "vector_add", "width": width}
    if fam == "kv_compress_epilog":
        toks = int(loc[0]) if len(loc) >= _MIN_2D_RANK else total_tokens
        bw = int(loc[-1]) if loc else (_MLA_LATENT_DIM + _MLA_ROPE_DIM) * _MLA_BF16_BYTES
        ew = max(1, int(bw / dtype_bytes(dt)))
        impl = resolve_onboarding_builder_and_abi(arch, "kv_compress_epilog")[3]
        return {"node_id": nid, "op_id": oid, "kind": "operator", "op": "kv_compress_epilog", "builder": "kv_compress_epilog", "family": fam, "recipe": rec, "implementation": impl, "dtype": dt, "shape": {"tokens": toks, "width": ew, "byte_width": bw}}, {"op_id": oid, "family": fam, "recipe": rec, "builder": "kv_compress_epilog", "dtype": dt, "width": ew}
    if fam in _COLLECTIVE_FAMILIES:
        return _lowered_collective_node(nid, oid, fam, rec, dt, op_rec, arch=arch, topo_opts=topo_opts)
    if fam == "attention_merge":
        toks = int(loc[0]) if len(loc) >= _MIN_GEMM_RANK else total_tokens
        heads = int(loc[1]) if len(loc) >= _MIN_GEMM_RANK else local_heads
        hdim = int(loc[2]) if len(loc) >= _MIN_GEMM_RANK else _MLA_LATENT_DIM
        partials = int(op_rec.get("partial_count") or 0) or max(_MIN_DCP_PARTIALS, dcp)
        impl = resolve_onboarding_builder_and_abi(arch, "attention_merge")[3]
        return {"node_id": nid, "op_id": oid, "kind": "operator", "op": "dcp_lse_attention_merge", "builder": "dcp_lse_attention_merge", "family": fam, "recipe": rec, "implementation": impl, "dtype": dt, "shape": {"tokens": toks, "heads": heads, "kv_heads": 1, "head_dim": hdim, "partial_count": partials, "dtype": dt}}, {"op_id": oid, "family": fam, "recipe": rec, "builder": "dcp_lse_attention_merge", "dtype": dt, "partial_count": partials}
    if fam in _SPARSE_INDEXER_FAMILIES:
        impl = resolve_onboarding_builder_and_abi(arch, "indexer")[3]
        return {"node_id": nid, "op_id": oid, "kind": "operator", "op": "lightning_indexer", "builder": "lightning_indexer", "family": "indexer", "recipe": rec, "implementation": impl, "dtype": dt, "shape": {"tokens": total_tokens, "heads": _INDEXER_ATTENTION_HEADS, "head_dim": _DEFAULT_HEAD_DIM}}, {"op_id": oid, "family": "indexer", "recipe": rec, "builder": "lightning_indexer", "dtype": dt}
    raise UnmodeledCapabilityError(f"profiler_builder_unmodeled:family={fam}:recipe={rec}:op_id={oid}")


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


def _structural_width_for_point(
    plan_doc: Mapping[str, Any], pt: Mapping[str, Any], *, sim_fam: str, custom_sim: Mapping[str, Any] | None,
) -> int:
    for src in (pt, plan_doc):
        if src.get("hidden_size"):
            return int(src["hidden_size"])
        if src.get("width"):
            return int(src["width"])
    if isinstance(custom_sim, Mapping):
        mdl = custom_sim.get("model")
        if isinstance(mdl, Mapping) and (mdl.get("hidden_size") or mdl.get("hidden")):
            return int(mdl.get("hidden_size") or mdl.get("hidden"))
    return _DEFAULT_HIDDEN_WIDTH if sim_fam in {"indexer", "layernorm", "rotary"} else _MLA_HIDDEN_WIDTH


def resolve_point_binding_and_shape(
    plan_doc: Mapping[str, Any], pt: Mapping[str, Any], *, sku: str, profile_override: Mapping[str, Any] | None = None,
) -> tuple[str, str, str, str, str, dict[str, Any], dict[str, Any]]:
    arch = _arch_for_sku(sku, profile_override=profile_override)
    plan_family = str(plan_doc.get("family") or "rmsnorm").strip()
    axis, val = str(pt.get("axis") or "context_len").strip(), int(pt["value"])
    batch, q_tok = max(1, int(pt.get("batch_size", 1))), max(1, int(pt.get("query_tokens", 1)))
    topo = str(pt.get("parallel_topology") or "tp1").strip()
    topk_val = int(pt["topk"]) if pt.get("topk") is not None else None
    is_abi_reject = str(pt.get("split") or "").strip().lower() == "abi_rejection"
    topo_opts: dict[str, Any] = parse_topo_options(topo)
    tp, dcp, pcp, ep = max(1, int(topo_opts.get("tp", 1))), max(1, int(topo_opts.get("dcp", 1))), max(1, int(topo_opts.get("pcp", 1))), max(1, int(topo_opts.get("ep", 1)))
    local_heads = max(1, _DEFAULT_ATTENTION_HEADS // tp)
    level = assign_plan_point_level(plan_doc, pt)
    raw_custom = pt.get("sim_case") or plan_doc.get("sim_case")
    custom_sim: Mapping[str, Any] | None = raw_custom if isinstance(raw_custom, Mapping) else None
    probe_indexed, probe_single_layer, probe_hidden, idx_dt = False, False, None, "bf16"
    if topk_val is not None or plan_family in _SPARSE_INDEXER_FAMILIES:
        cm_q = custom_sim.get("model", {}).get("quant", {}) if isinstance(custom_sim, Mapping) and isinstance(custom_sim.get("model"), Mapping) and isinstance(custom_sim["model"].get("quant"), Mapping) else {}
        exp_dt = str(pt.get("query_dtype") or pt.get("cache_dtype") or plan_doc.get("query_dtype") or plan_doc.get("cache_dtype") or cm_q.get("index_dtype") or "").strip().lower()
        is_q_idx = (exp_dt in {"int8", "fp8", "fp8_e4m3"}) if exp_dt else (plan_family == "sparse_indexer" and arch == "A5")
        idx_dt = (exp_dt if exp_dt in {"int8", "fp8_e4m3"} else "fp8_e4m3") if is_q_idx else "bf16"
        pfam, prec, pbld, impl = resolve_onboarding_builder_and_abi(arch, "indexer", prefer_quant_indexer=is_q_idx)
        sim_fam, probe_indexed = "indexer", True
        ctx_len = val if axis == "context_len" else max(val, _DEFAULT_CONTEXT_FALLBACK)
        kv_len = ctx_len if is_abi_reject else (ctx_len + q_tok)
        shape: dict[str, Any] = {"tokens": max(1, batch * q_tok), "heads": _INDEXER_ATTENTION_HEADS, "kv_heads": 1, "head_dim": _DEFAULT_HEAD_DIM, "seq_len": kv_len, "kv_len": kv_len, "context_len": ctx_len, "batch_size": batch, "sequences": batch, "query_tokens": q_tok, "q_len": q_tok, "query_dtype": idx_dt, "cache_dtype": idx_dt, "input_layout": "TND", "query_layout": "TND", "parallel_topology": topo}
        if topk_val is not None:
            shape.update({"top_k": topk_val, "sparse_count": topk_val, "enforce_exact_topk": True})
    elif (plan_family == "sku_suite" and axis in {"context_len", "batch_x_kv_heads"}) or plan_family == "attention":
        is_single_q_local = q_tok == 1 and tp == 1 and dcp == 1 and pcp == 1
        attn_rec = _decode_mla_recipe_for_sku(sku, profile_override=profile_override) if is_single_q_local else "latent_attention0"
        pfam, prec, pbld = ("attention", attn_rec, "absorbed_mla")
        impl = resolve_onboarding_builder_and_abi(arch, "attention")[3]
        sim_fam = "attention"
        ctx_len = val if axis == "context_len" else _DEFAULT_CONTEXT_FALLBACK
        seqs = val if axis == "batch_x_kv_heads" else batch
        kv_len = ctx_len + q_tok
        if dcp > 1:
            la = next((o for o in _lower_one_layer_step_ops(sku, topo, "prefill" if q_tok > 1 else "decode", q_tok, ctx_len, seqs, sim_case=custom_sim, profile_override=profile_override) if o["family"] == "attention"), None)
            if la and len(la["local_shape"]) >= _MIN_2D_RANK:
                kv_len = int(la["local_shape"][1])
        qlay = "BNSD" if is_single_q_local else "TND"
        shape = {"tokens": max(1, seqs * q_tok), "sequences": seqs, "batch_size": batch, "query_tokens": q_tok, "q_len": q_tok, "seq_len": kv_len, "kv_len": kv_len, "context_len": ctx_len, "page_count": _page_count(kv_len), "heads": local_heads, "kv_heads": 1, "head_dim": _MLA_LATENT_DIM, "latent": _MLA_LATENT_DIM, "kv_lora_rank": _MLA_LATENT_DIM, "rope": _MLA_ROPE_DIM, "qk_rope_head_dim": _MLA_ROPE_DIM, "block_size": _MLA_BLOCK_SIZE, "paged": True, "query_layout": qlay, "input_layout": qlay, "cache_layout": "PA_BSND", "output_layout": "NBSD" if is_single_q_local else "TND", "parallel_topology": topo}
        if axis == "batch_x_kv_heads":
            shape["batch_x_kv_heads"] = val
        probe_single_layer = level == "L3" and plan_family == "sku_suite"
    elif plan_family == "sku_suite":
        pfam, prec, pbld, impl = resolve_onboarding_builder_and_abi(arch, "rmsnorm")
        sim_fam = "rmsnorm"
        shape = {"tokens": val, "width": _MLA_HIDDEN_WIDTH, "parallel_topology": topo}
    elif is_onboarding_family(plan_family):
        pfam, prec, pbld, impl = resolve_onboarding_builder_and_abi(arch, plan_family)
        sim_fam = pfam
        width = _structural_width_for_point(plan_doc, pt, sim_fam=sim_fam, custom_sim=custom_sim)
        probe_hidden = width
        tok_val = val if axis == "tokens" else max(1, (val if axis == "batch_x_kv_heads" else batch) * q_tok)
        shape, topo_opts = _resolve_family_shape(
            pbld, tok_val=tok_val, width=width, tp=tp, ep=ep,
            local_heads=local_heads, topo=topo, topo_opts=topo_opts,
        )
    else:
        pfam, prec, pbld, impl, sim_fam = plan_family, plan_family, "", str(plan_doc.get("target_id") or plan_family), plan_family
        shape = {"tokens": val if axis == "tokens" else max(1, batch * q_tok), "width": _structural_width_for_point(plan_doc, pt, sim_fam=sim_fam, custom_sim=custom_sim), "parallel_topology": topo}
    if custom_sim is not None:
        sim_case = dict(custom_sim)
    else:
        ln_probe = (sim_fam == "layernorm")
        sim_case = _compose_structural_probe_case(
            sku, topo_opts, indexed=(probe_indexed or ln_probe),
            topk=topk_val if (probe_indexed and not is_abi_reject) else None,
            hidden_size=probe_hidden, single_layer=(probe_single_layer or ln_probe),
            index_dtype=idx_dt, index_head_dim=(probe_hidden if ln_probe else None),
        )
    return pfam, prec, pbld, impl, sim_fam, shape, sim_case


def _build_case_nodes(
    pid: str, *, sku: str, pt: Mapping[str, Any], level: str, pfam: str, prec: str,
    pbld: str, impl: str, shape: Mapping[str, Any], topo_opts: Mapping[str, int],
    sim_case: Mapping[str, Any] | None = None, profile_override: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    arch = _arch_for_sku(sku, profile_override=profile_override)
    total_tokens = max(1, int(shape.get("tokens", 1)))
    local_heads = max(1, int(shape.get("heads", _DEFAULT_ATTENTION_HEADS // max(1, int(topo_opts.get("tp", 1))))))
    node_dt = str(shape.get("dtype") or shape.get("query_dtype") or "bf16").strip().lower()
    if level == "L1" or pbld != "absorbed_mla":
        node: dict[str, Any] = {"node_id": f"{pid}:op0", "kind": "operator", "op": pbld or prec, "family": pfam, "recipe": prec, "implementation": impl, "dtype": node_dt, "shape": dict(shape)}
        if pbld:
            node["builder"] = pbld
        if pbld in {"absorbed_mla", "fused_infer_attention"}:
            node["paged"] = True
        if shape.get("query_layout"):
            node["layout"] = str(shape["query_layout"])
        return [node], [], [], []
    if level == "L2":
        attn_id = "s0.l0.paged_mla_decode" if prec in {"ascend_paged_mla_decode_arch22", "ascend_paged_mla_decode_arch35"} else "s0.l0.latent_attention0"
        v_m = total_tokens * local_heads
        gmm_impl = resolve_onboarding_builder_and_abi(arch, "grouped_gemm")[3]
        nodes = [
            {"node_id": f"{pid}:{attn_id}", "op_id": attn_id, "kind": "operator", "op": pbld, "builder": pbld, "family": pfam, "recipe": prec, "implementation": impl, "dtype": node_dt, "paged": True, "layout": str(shape.get("query_layout") or "TND"), "shape": dict(shape)},
            {"node_id": f"{pid}:s0.l0.v_expand", "op_id": "s0.l0.v_expand", "kind": "operator", "op": "grouped_matmul", "builder": "grouped_matmul", "family": "grouped_gemm", "recipe": "grouped_matmul", "implementation": gmm_impl, "dtype": node_dt, "group_list": [total_tokens] * local_heads, "group_list_type": 1, "shape": {"m": v_m, "n": _MLA_HEAD_DIM, "k": _MLA_LATENT_DIM, "e": local_heads, "matrix_groups": local_heads, "dtype": node_dt}},
        ]
        nbs = [
            {"op_id": attn_id, "family": "attention", "recipe": prec, "builder": pbld, "paged": True, "heads": local_heads, "kv_heads": 1, "latent": _MLA_LATENT_DIM, "rope": _MLA_ROPE_DIM, "kv_len": int(shape["seq_len"]), "page_count": int(shape["page_count"]), "query_layout": str(shape.get("query_layout") or "TND")},
            {"op_id": "s0.l0.v_expand", "family": "gemm", "recipe": "v_expand", "builder": "grouped_matmul", "dtype": node_dt, "quant_mode": "none", "m": v_m, "n": _MLA_HEAD_DIM, "k": _MLA_LATENT_DIM, "matrix_groups": local_heads},
        ]
        return nodes, [{"from": f"{pid}:{attn_id}", "to": f"{pid}:s0.l0.v_expand", "kind": "data"}], nbs, [attn_id, "s0.l0.v_expand"]
    lowered = _lower_one_layer_step_ops(
        sku, str(pt.get("parallel_topology") or "tp1"),
        "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode",
        max(1, int(pt.get("query_tokens", 1))),
        int(shape.get("context_len", pt["value"])),
        max(1, int(shape.get("sequences", pt.get("batch_size", 1)))),
        sim_case=sim_case, profile_override=profile_override,
    )
    op_ids = {r["op_id"] for r in lowered}
    nodes, edges, nbs, tids = [], [], [], []
    for rec in lowered:
        nd, nb = _lowered_op_to_profiler_node(pid, rec, attn_shape=shape, total_tokens=total_tokens, local_heads=local_heads, topo_opts=topo_opts, arch=arch)
        nodes.append(nd)
        nbs.append(nb)
        tids.append(rec["op_id"])
        edges.extend({"from": f"{pid}:{d}", "to": f"{pid}:{rec['op_id']}", "kind": "stream"} for d in rec["deps"] if d in op_ids)
    return nodes, edges, nbs, tids


def resolve_point_measured_binding(plan_doc: Mapping[str, Any], pt: Mapping[str, Any], *, sku: str, profile_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    with temporary_scaffold_profile(sku, profile_override):
        pfam, prec, pbld, impl, sim_fam, shape, sim_case = resolve_point_binding_and_shape(plan_doc, pt, sku=sku, profile_override=profile_override)
        level = assign_plan_point_level(plan_doc, pt)
        topo_opts = parse_topo_options(str(pt.get("parallel_topology") or "tp1"))
        pid = str(pt.get("point_id") or "pt")
        has_custom = isinstance(pt.get("sim_case") or plan_doc.get("sim_case"), Mapping)
        target_op_id = (
            ("s0.l0.paged_mla_decode" if prec in {"ascend_paged_mla_decode_arch22", "ascend_paged_mla_decode_arch35"} else "s0.l0.latent_attention0") if sim_fam == "attention"
            else ("s0.l0.norm1" if sim_fam == "rmsnorm" else ("" if (sim_fam == "indexer" and has_custom) else ("s0.l0.index_lightning" if sim_fam == "indexer" else ("s0.l1.expert_up" if pbld == "grouped_matmul" else ("s0.l0.output" if pbld == "matmul" else "")))))
        )
        _nodes, _edges, nbs, tids = _build_case_nodes(
            pid, sku=sku, pt=pt, level=level, pfam=pfam, prec=prec, pbld=pbld, impl=impl,
            shape=shape, topo_opts=topo_opts, sim_case=sim_case, profile_override=profile_override,
        )
    binding: dict[str, Any] = {"builder": pbld, "family": sim_fam, "recipe": prec, "implementation": impl, "target_op_id": target_op_id}
    for k in ("paged", "heads", "kv_heads", "head_dim", "latent", "rope", "width", "kv_len", "page_count", "query_layout", "input_layout", "query_dtype", "cache_dtype", "sparse_count", "topk", "dtype", "m", "n", "k", "matrix_groups"):
        if k in shape:
            binding[k] = shape[k]
    if pbld in {"matmul", "grouped_matmul"} or sim_fam in _GEMM_FAMILIES:
        gemm_dt = str(shape.get("dtype") or "").strip().lower()
        if not gemm_dt:
            raise ModelingError(f"gemm_binding_dtype_missing:{pid}")
        binding["dtype"] = gemm_dt
        binding.setdefault("weight_dtype", str(shape.get("weight_dtype") or gemm_dt).strip().lower())
        binding.setdefault("quant_mode", str(shape.get("quant_mode") or "none").strip().lower())
        binding.setdefault("matrix_groups", int(shape.get("matrix_groups") or shape.get("e") or 1))
    if nbs:
        binding["node_bindings"] = nbs
    if tids:
        binding["timed_op_ids"] = tids
    return {"level": level, "phase": "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode", "target_family": sim_fam, "target_op_id": target_op_id, "target_op_ids": list(tids) if level == "L2" else [], "measured_binding": binding, "sim_case": sim_case, "expected_shape": shape, "profiler_family": pfam, "profiler_recipe": prec, "profiler_builder": pbld, "implementation": impl}


def _plan_backend_kind(plan_doc: Mapping[str, Any], pt: Mapping[str, Any]) -> str:
    if (bk := str(pt.get("backend_kind") or plan_doc.get("backend_kind") or "").strip().lower()):
        return bk
    sc = pt.get("sim_case") or plan_doc.get("sim_case")
    if isinstance(sc, Mapping):
        if isinstance(mdl := sc.get("model"), Mapping) and (att := str(mdl.get("attention") or "").strip().lower()):
            return att
        if sc.get("compose_model"):
            from mindie_sim.export.operator_matrix import resolve_model_spec_public
            return str(getattr(resolve_model_spec_public(sc["compose_model"]), "attention", "") or "").lower()
    return ""


def build_profiler_subgraph_plan(plan_doc: Mapping[str, Any], *, sku: str, split_tag: str = "", acquisition_id: str = "", profile_override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if str(plan_doc.get("schema") or "") == _SUBGRAPH_PLAN_SCHEMA:
        return {**dict(plan_doc), "sku": sku, **({"acquisition_id": acquisition_id} if acquisition_id else {})}
    world, plan_sha, cases = required_ranks_for_plan(plan_doc), str(plan_doc["plan_sha256"]), []
    with temporary_scaffold_profile(sku, profile_override):
        for split_key in ("train_points", "heldout_points", "abi_rejection_points"):
            for pt in plan_doc.get(split_key) or ():
                if not isinstance(pt, Mapping) or not pt.get("point_id"):
                    continue
                pid, topo = str(pt["point_id"]), str(pt.get("parallel_topology") or "tp1")
                pt_ranks, topo_opts = validate_parallel_topology(topo, backend_kind=_plan_backend_kind(plan_doc, pt)), parse_topo_options(topo)
                pfam, prec, pbld, impl, _sf, shape, sim_case = resolve_point_binding_and_shape(plan_doc, pt, sku=sku, profile_override=profile_override)
                level = assign_plan_point_level(plan_doc, pt)
                nodes, edges, _nb, _tids = _build_case_nodes(pid, sku=sku, pt=pt, level=level, pfam=pfam, prec=prec, pbld=pbld, impl=impl, shape=shape, topo_opts=topo_opts, sim_case=sim_case, profile_override=profile_override)
                entry_dt = str(shape.get("dtype") or shape.get("query_dtype") or "bf16").strip().lower()
                entry: dict[str, Any] = {
                    "case_id": pid, "split": str(pt.get("split") or "heldout"), "level": level, "boundary": STANDARD_LEVEL_BOUNDARY_KINDS[level],
                    "phase": "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode", "stage": "s1_joint" if pt_ranks > 1 else "s0_local",
                    "clock_operator": str(pt.get("clock_operator") or DEFAULT_CLOCK_OPERATOR), "implementation": impl, "dtype": entry_dt,
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


def _extract_row_event_ids(row: Mapping[str, Any]) -> list[str]:
    raw = row.get("event_ids")
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        cleaned = [str(ev).strip() for ev in raw if str(ev).strip()]
        if cleaned:
            return cleaned
    sample_id = str(row.get("sample_id") or "").strip()
    return [sample_id] if sample_id else []


def _resolve_attributed_event_ids(
    p50_row: Mapping[str, Any], rcpts: Sequence[Mapping[str, Any]], group: Sequence[Mapping[str, Any]], cid: str,
) -> list[str]:
    win_rank = p50_row.get("rank")
    cands = [p50_row, *(r for r in rcpts if r.get("rank") == win_rank), *(r for r in rcpts if r.get("rank") != win_rank), *(r for r in group if r is not p50_row)]
    for rec in cands:
        if ev := _extract_row_event_ids(rec):
            return ev
    raise ModelingError(f"profiler_observation_event_ids_missing:{cid}")


def _is_qualified_observation(row: Mapping[str, Any]) -> bool:
    if row.get("qualified") is False:
        return False
    tmode = str(row.get("timing_mode") or "").strip().lower()
    tsrc = str(row.get("timer_source") or "").strip().lower()
    return tmode not in _HOST_FALLBACK_MODES and tsrc not in _HOST_FALLBACK_MODES


def _expected_ranks_for_case(plan_doc: Mapping[str, Any], cid: str, plan_pt: Mapping[str, Any] | None) -> tuple[int, set[int] | None]:
    world = max(1, int(plan_doc.get("world_size") or 1))
    for case in (plan_doc.get("cases") or ()):
        if not isinstance(case, Mapping) or str(case.get("case_id") or "") != cid:
            continue
        ck = case.get("canonical_key") if isinstance(case.get("canonical_key"), Mapping) else {}
        grp = case.get("group_ranks") or ck.get("ordered_rank_placement") or next(
            (nd["group_ranks"] for nd in (case.get("nodes") or ()) if isinstance(nd, Mapping) and nd.get("group_ranks")), None,
        )
        if isinstance(grp, Sequence) and not isinstance(grp, (str, bytes)) and grp:
            rset = {int(r) for r in grp}
            return max(world, max(rset) + 1), rset
        return world, (set(range(world)) if world > 1 else None)
    topo_w = validate_parallel_topology(str(plan_pt.get("parallel_topology") or "tp1")) if plan_pt is not None else 1
    if topo_w > 1:
        return max(world, topo_w), set(range(topo_w))
    return world, (set(range(world)) if world > 1 else None)


def _row_clock_and_modes(row: Mapping[str, Any], cid: str) -> tuple[str, str, str]:
    tmode = str(row.get("timing_mode") or "").strip().lower()
    cap = str(row.get("capture_mode") or ("aclgraph" if "aclgraph" in tmode else "eager")).strip().lower()
    raw_clock = str(row.get("clock_operator") or "").strip()
    if raw_clock:
        if raw_clock not in VALID_CLOCK_OPERATORS:
            raise ModelingError(f"profiler_clock_operator_invalid:{cid}:{raw_clock!r}")
        return tmode, cap, raw_clock
    if not tmode:
        raise ModelingError(f"profiler_clock_operator_missing:{cid}")
    return tmode, cap, ("busy_union" if tmode in _BUSY_UNION_MODES else DEFAULT_CLOCK_OPERATOR)


def normalize_profiler_records(
    raw_records: Sequence[Mapping[str, Any]], *, plan_doc: Mapping[str, Any], sku: str,
    endpoint_alias: str = "", capture_out_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Normalize raw ``mindie-profiler`` JSONL/receipt rows into sealed onboarding rows & ABI rows."""
    if not raw_records:
        return [], []
    is_prof = any(str(r.get("schema") or "") in {_OBSERVATION_SCHEMA, _OBSERVATION_V2_SCHEMA, _RECEIPT_SCHEMA, _SUBGRAPH_RUN_SCHEMA} or r.get("record_type") in {"observation", "case_complete", "receipt"} for r in raw_records)
    if not is_prof:
        return [dict(r) for r in raw_records if str(r.get("status") or "").lower() not in _ABI_REJECT_STATUSES], [
            {"point_id": str(r.get("point_id") or r.get("case_id") or ""), "rejected": True, "abi_error": str(r.get("abi_error") or r.get("reason") or ""), "executed_shape": dict(r.get("executed_shape") or r.get("actual_shape") or {})}
            for r in raw_records if str(r.get("status") or "").lower() in _ABI_REJECT_STATUSES
        ]
    plan_pts = {str(p["point_id"]): p for sk in ("train_points", "heldout_points") for p in (plan_doc.get(sk) or ()) if isinstance(p, Mapping) and p.get("point_id")}
    raw_by_case: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    receipts: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    abi_rows: list[dict[str, Any]] = []
    for rec in raw_records:
        cid = str(rec.get("case_id") or rec.get("point_id") or "").strip()
        status, rtype = str(rec.get("status") or "ok").strip().lower(), str(rec.get("record_type") or "").strip().lower()
        if status in _ABI_REJECT_STATUSES:
            abi_rows.append({"point_id": cid, "rejected": True, "abi_error": str(rec.get("reason") or rec.get("abi_error") or ""), "executed_shape": dict(rec.get("executed_shape") or rec.get("actual_shape") or {})})
            continue
        split = str(rec.get("split") or (plan_pts.get(cid) or {}).get("split") or "heldout").strip().lower()
        acq_k = str(rec.get("acquisition_id") or rec.get("acquisition_root") or "").strip()
        if rtype == "receipt" or str(rec.get("schema") or "") == _SUBGRAPH_RUN_SCHEMA:
            receipts.setdefault((cid, split, acq_k), []).append(rec)
        elif rtype == "observation" or rec.get("duration_ms") is not None or rec.get("duration_us") is not None:
            if status not in {"", "ok", "passed", "pass"}:
                raise ModelingError(f"profiler_observation_status_invalid:{cid}:status={status}")
            raw_by_case.setdefault((cid, split, acq_k), []).append(rec)

    buckets: dict[tuple[str, str, str, str, str, str], list[Mapping[str, Any]]] = {}
    for (cid, split, acq_k), raw_group in sorted(raw_by_case.items()):
        qual_group = [r for r in raw_group if _is_qualified_observation(r)]
        if not qual_group:
            raise ModelingError(f"profiler_observation_unqualified:{cid}")
        for r in qual_group:
            tmode, cap, ck = _row_clock_and_modes(r, cid)
            buckets.setdefault((cid, split, acq_k, tmode, cap, ck), []).append(r)

    clean_ep, plan_family, out_rows = sanitize_public_alias(endpoint_alias or sku, sku), str(plan_doc.get("family") or "").strip(), []
    for (cid, split, acq_k, _tmode, _cap, clock_op), group in sorted(buckets.items()):
        plan_pt = plan_pts.get(cid)
        binfo = resolve_point_measured_binding(plan_doc, plan_pt, sku=sku) if plan_pt is not None else None
        rcpts = receipts.get((cid, split, acq_k)) or []
        if not rcpts and acq_k and (unkeyed := receipts.get((cid, split, ""))):
            obs_rids = {_TRIAL_SUFFIX_RE.sub("", str(g.get("run_id") or "").strip()) for g in group}
            rcpts = [r for r in unkeyed if not (rr := _TRIAL_SUFFIX_RE.sub("", str(r.get("run_id") or "").strip())) or any(o == rr or o.startswith(f"{rr}.") for o in obs_rids)]
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
        exp_world, exp_ranks = _expected_ranks_for_case(plan_doc, cid, plan_pt)
        by_slot: dict[Any, list[tuple[int, int, Mapping[str, Any]]]] = {}
        ranks_seen: set[int] = set()
        for idx_g, row in enumerate(group):
            dur_ms = float(row["duration_ms"]) if row.get("duration_ms") is not None else (float(row.get("duration_us") or 0.0) / 1000.0)
            if dur_ms <= 0.0:
                raise ModelingError(f"profiler_sample_duration_invalid:{cid}:duration_ms={dur_ms}")
            raw_rank = row.get("rank")
            if raw_rank is None or isinstance(raw_rank, bool):
                raise ModelingError(f"profiler_observation_rank_missing:{cid}")
            try:
                rank_int = int(raw_rank)
            except (TypeError, ValueError) as exc:
                raise ModelingError(f"profiler_observation_rank_invalid:{cid}:rank={raw_rank!r}") from exc
            if rank_int < 0 or (exp_ranks is not None and rank_int not in exp_ranks) or (exp_world > 1 and rank_int >= exp_world):
                raise ModelingError(f"profiler_observation_rank_invalid:{cid}:rank={raw_rank!r}")
            ranks_seen.add(rank_int)
            slot_key: Any = (row.get("trial"), row.get("repeat_index")) if ("trial" in row or "repeat_index" in row) else idx_g
            by_slot.setdefault(slot_key, []).append((int(round(dur_ms * NS_PER_MS)), rank_int, row))
        if exp_ranks is not None:
            for entries in by_slot.values():
                got_r = {item[1] for item in entries}
                if got_r != exp_ranks:
                    raise ModelingError(f"profiler_observation_rank_set_incomplete:{cid}:expected={sorted(exp_ranks)}:got={sorted(got_r)}")
        slot_peaks = sorted((max(entries, key=lambda item: (item[0], item[1])) for entries in by_slot.values()), key=lambda item: (item[0], item[1]))
        durs_ns = [item[0] for item in slot_peaks]
        p50_ns, p99_ns = _quantile_ns(durs_ns, _P50_Q), _quantile_ns(durs_ns, _P99_Q)
        p50_row, first = min(slot_peaks, key=lambda item: abs(item[0] - p50_ns))[2], group[0]
        event_ids = _resolve_attributed_event_ids(p50_row, rcpts, group, cid)
        level = binfo["level"] if binfo is not None else str(first.get("level") or "L1").upper()
        explicit_acq = acq_k or str(first.get("acquisition_id") or first.get("acquisition_root") or ((rcpts[0].get("acquisition_id") or rcpts[0].get("acquisition_root")) if rcpts else "") or "").strip()
        base_acq = str((plan_doc.get("train_lineage_root") if split == "train" else plan_doc.get("heldout_lineage_root")) or f"acq-{split}")
        acq_id = explicit_acq or (f"{base_acq}-{level.lower()}" if split == "heldout" else base_acq)
        base_rid = _TRIAL_SUFFIX_RE.sub("", str(first.get("run_id") or f"run-{cid}"))
        pub_rid = sanitize_public_run_id(base_rid)
        timed_nodes = list(next((r["timed_nodes"] for r in (*rcpts, *group) if isinstance(r.get("timed_nodes"), Sequence) and not isinstance(r.get("timed_nodes"), (str, bytes)) and r["timed_nodes"]), ()))
        dims = extract_shape_dims(actual_shape, {"point_id": cid, "axis": str((plan_pt or {}).get("axis") or "")}, plan_pt=plan_pt, expected_shape=binfo["expected_shape"] if binfo else None, measured_binding=binfo["measured_binding"] if binfo else None, cid=cid)
        clean_shape = _sanitize_capture_shape(actual_shape)
        cap_obj = {"schema": "mindie-sim.onboarding-capture/1", "case_id": cid, "sku": sku, "split": split, "level": level, "acquisition_root": acq_id, "run_uid": pub_rid, "endpoint_alias": clean_ep, "clock_operator": clock_op, "measured_ns": p50_ns, "measured_p99_ns": p99_ns, "device_ranks": sorted(ranks_seen), "event_ids": event_ids, "actual_shape": clean_shape, "timed_nodes": timed_nodes}
        cap_bytes = canonical_json_bytes(cap_obj)
        raw_sha = sha256_bytes(cap_bytes)
        if capture_out_dir is not None:
            write_atomic_bytes(Path(capture_out_dir) / f"{raw_sha}.json", cap_bytes)
        row_out: dict[str, Any] = {
            "case_id": cid, "point_id": cid, "sku": sku, "split": split, "level": level, "status": "ok",
            "acquisition_root": acq_id, "run_uid": pub_rid, "raw_sha256": raw_sha,
            "capture_uri": sanitize_public_capture_uri("", raw_sha256=raw_sha, default_prefix=f"capture://{split}"),
            "device_ranks": sorted(ranks_seen), "event_ids": event_ids,
            "measured_ns": p50_ns, "measured_p99_ns": p99_ns, "clock_operator": clock_op,
            "busy_union_ns": p50_ns if clock_op == "busy_union" else None,
            "elapsed_span_ns": p50_ns if clock_op == "elapsed_span" else None,
            "actual_shape": clean_shape, "timed_nodes": timed_nodes, **dims,
        }
        if binfo is not None:
            row_out.update({"sim_case": binfo["sim_case"], "phase": binfo["phase"], "target_family": binfo["target_family"], "target_op_id": binfo["target_op_id"], "target_op_ids": binfo["target_op_ids"], "measured_binding": binfo["measured_binding"], "implementation": binfo["implementation"]})
        elif plan_family and plan_family != "sku_suite":
            row_out["target_family"] = plan_family
        out_rows.append(row_out)
    assert_sanitized_payload(out_rows, field_name="profiler_normalized_rows")
    return out_rows, abi_rows


def validate_host_queue_grant(lease_info: Mapping[str, Any] | None, *, required_devices: int, min_ttl_seconds: int) -> tuple[str, int, list[int], int]:
    req_npus, info = int(required_devices), dict(lease_info or {})
    lease_id, fence, raw_devs, ttl = str(info.get("lease_id") or "").strip(), info.get("fence_token"), info.get("devices"), info.get("ttl_seconds")
    if req_npus < 1 or not lease_id or type(fence) is not int or fence <= 0:
        raise ModelingError("host_queue_lease_missing_lease_id_or_positive_fence_token")
    if not isinstance(raw_devs, Sequence) or isinstance(raw_devs, (str, bytes)):
        raise ModelingError("host_queue_lease_missing_granted_devices")
    devs = [int(d) for d in raw_devs if type(d) is int and d >= 0]
    if len(devs) != len(raw_devs) or len(set(devs)) != req_npus:
        raise ModelingError(f"host_queue_lease_device_count_mismatch:expected={req_npus}:granted={devs}")
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
    warn = acq.get("receipt_warning") or data.get("receipt_warning")
    unconf = bool(acq.get("receipt_unconfirmed") or data.get("receipt_unconfirmed") or warn)
    rec: dict[str, Any] = {"lease_mode": "coordinator_host_queue", "lease_id": lease_id, "fence_token": fence if type(fence) is int else 0, "devices": [], "ttl_seconds": eff_ttl, "receipt_unconfirmed": unconf, "receipt_warning": sanitize_public_warning(warn or "host_queue_lease_acquire_unconfirmed") if unconf else ""}
    try:
        if str(acq.get("outcome") or "").lower() not in _VALID_LEASE_OUTCOMES or str(acq.get("status") or "").lower() not in _VALID_LEASE_STATUSES:
            raise ModelingError("host_queue_lease_acquire_rejected")
        _, _, rec["devices"], rec["ttl_seconds"] = validate_host_queue_grant({**dict(data), "lease_id": lease_id, "fence_token": fence, "ttl_seconds": data.get("ttl_seconds", eff_ttl)}, required_devices=req_npus, min_ttl_seconds=min_req)
        yield rec
    finally:
        if lease_id and type(fence) is int and fence > 0:
            try:
                _run_coordinator_lease_cmd(action="release", endpoint_alias=alias, payload={"lease_id": lease_id, "fence_token": fence}, timeout_s=int(rpc_timeout_s), coordinator_bin=coordinator_bin)
            except TimeoutError:
                raise
            except (ModelingError, OSError) as exc:
                rec["receipt_unconfirmed"], rec["receipt_warning"] = True, sanitize_public_warning(f"host_queue_lease_release_unconfirmed:{exc}")


def _plan_required_ranks(plan_doc: Mapping[str, Any]) -> int:
    req = max(required_ranks_for_plan(plan_doc), int(plan_doc.get("world_size") or 1))
    for case in (plan_doc.get("cases") or ()):
        if not isinstance(case, Mapping):
            continue
        ck = case.get("canonical_key") if isinstance(case.get("canonical_key"), Mapping) else {}
        for grp in (case.get("group_ranks"), ck.get("ordered_rank_placement"), *(nd.get("group_ranks") for nd in (case.get("nodes") or ()) if isinstance(nd, Mapping))):
            if isinstance(grp, Sequence) and not isinstance(grp, (str, bytes)) and grp:
                req = max(req, max(int(r) for r in grp) + 1, len(grp))
    return req


def _collect_rank_outputs(tmp: Path, out_f: Path, obs_f: Path, world: int, cdir: Path | None = None) -> list[Mapping[str, Any]]:
    raw: list[Mapping[str, Any]] = []
    mf_path = (out_f if out_f.is_file() else (cdir / "campaign_manifest.json")) if cdir is not None else None
    if cdir is not None and mf_path is not None and mf_path.is_file():
        doc = json.loads(mf_path.read_text(encoding="utf-8"))
        if isinstance(doc, Mapping) and str(doc.get("schema") or "") == "mindie-bench-campaign/1":
            from mindie_sim.observations.campaign_reconcile import verify_campaign_integrity
            camp = verify_campaign_integrity(doc, base_dir=cdir)
            for sp in sorted({(cdir / rel).resolve() for rel in [camp.observations_jsonl, *(s.path for s in camp.run_shards)]}):
                if sp.is_file():
                    raw.extend(json.loads(ln) for ln in sp.read_text(encoding="utf-8").splitlines() if ln.strip())
            if raw:
                return raw
    for rk in range(world):
        rk_obs, rk_out = (tmp / f"obs.rank{rk}.jsonl" if world > 1 else obs_f), (Path(f"{out_f}.rank{rk}.json") if world > 1 else out_f)
        if rk_obs.is_file():
            raw.extend(json.loads(ln) for ln in rk_obs.read_text(encoding="utf-8").splitlines() if ln.strip())
        if rk_out.is_file():
            doc = json.loads(rk_out.read_text(encoding="utf-8"))
            raw.extend({**dict(r), "schema": _SUBGRAPH_RUN_SCHEMA, "record_type": "receipt", "run_id": doc.get("run_id", ""), "rank": rk} for r in (doc.get("records") or ()) if isinstance(r, Mapping))
    return raw


def _launch_ranks_concurrent(world: int, cmd_for_rank: Any, *, env: Mapping[str, str], tmp: Path, timeout_s: int) -> None:
    mport = ""
    if world > 1:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            mport = str(int(sock.getsockname()[1]))
    procs, fhs, deadline = [], [], time.monotonic() + float(timeout_s)
    try:
        for rk in range(world):
            rk_env = {**env, "RANK": str(rk), "LOCAL_RANK": str(rk), "WORLD_SIZE": str(world), **({"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": mport} if world > 1 else {})}
            fo, fe = (tmp / f"rk{rk}.out.log").open("w", encoding="utf-8"), (tmp / f"rk{rk}.err.log").open("w", encoding="utf-8")
            fhs.extend([fo, fe])
            procs.append((rk, subprocess.Popen(cmd_for_rank(rk), env=rk_env, stdout=fo, stderr=fe, text=True, **subprocess_window_kwargs()), tmp / f"rk{rk}.err.log"))
        for rk, proc, err_log in procs:
            try:
                rc = proc.wait(timeout=max(0.05, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(f"profiler_bench_timeout:{timeout_s}s:rank={rk}") from exc
            if rc not in (0, _PARTIAL_ABI_EXIT_CODE):
                raise ModelingError(f"profiler_bench_failed:rank={rk}:rc={rc}:{err_log.read_text(encoding='utf-8', errors='replace').strip() if err_log.is_file() else ''}")
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"profiler_bench_os_error:{exc}") from exc
    finally:
        for _, proc, _ in procs:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
        for fh in fhs:
            try:
                fh.close()
            except OSError:
                pass


def run_profiler_bench_subprocess(
    plan_doc: Mapping[str, Any], *, sku: str, profiler_root: str | Path | None = None,
    endpoint_alias: str = "", allowed_phy_ids: Sequence[int] = (), fence_token: int | None = None,
    timeout_s: int = DEFAULT_GATE_SUBPROCESS_TIMEOUT_S, capture_out_dir: str | Path | None = None,
    profile_override: Mapping[str, Any] | None = None, campaign_dir: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute coordinated ``mindie_profiler bench`` (concurrent S1 ranks; zero pipe/rank deadlock)."""
    is_v2 = str(plan_doc.get("schema") or "") == _BENCH_PLAN_V2_SCHEMA
    sg_plan = dict(plan_doc) if is_v2 else build_profiler_subgraph_plan(plan_doc, sku=sku, profile_override=profile_override)
    world, env = _plan_required_ranks(sg_plan), dict(os.environ)
    if profiler_root is not None:
        env["PYTHONPATH"] = str(Path(profiler_root).resolve()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if allowed_phy_ids:
        env["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(int(d)) for d in allowed_phy_ids)
    if fence_token is not None:
        env["MINDIE_LEASE_FENCE_TOKEN"] = str(int(fence_token))
    with tempfile.TemporaryDirectory(prefix="msim-prof-") as tmp_dir:
        tmp = Path(tmp_dir)
        plan_f, obs_f, out_f = write_atomic_json(tmp / "plan.json", sg_plan), tmp / "obs.jsonl", tmp / "out.json"
        cdir = (Path(campaign_dir) if campaign_dir is not None else (tmp / "campaign")) if (is_v2 or campaign_dir is not None) else None
        if cdir is not None:
            cdir.mkdir(parents=True, exist_ok=True)
            _launch_ranks_concurrent(world, lambda _rk: [sys.executable, "-B", "-m", "mindie_profiler", "bench", "run-plan", "--plan", str(plan_f), "--world-size", str(world), "--campaign-dir", str(cdir), "--out", str(out_f)], env=env, tmp=tmp, timeout_s=timeout_s)
        else:
            _launch_ranks_concurrent(world, lambda rk: [sys.executable, "-B", "-m", "mindie_profiler", "bench", "subgraph", "--plan", str(plan_f), "--observations", str(tmp / f"obs.rank{rk}.jsonl" if world > 1 else obs_f), "--out", str(out_f)], env=env, tmp=tmp, timeout_s=timeout_s)
        raw = _collect_rank_outputs(tmp, out_f, obs_f, world, cdir=cdir)
    return normalize_profiler_records(raw, plan_doc=plan_doc, sku=sku, endpoint_alias=endpoint_alias, capture_out_dir=capture_out_dir)


def acquire_stage3_observations(
    plan_doc: Mapping[str, Any], *, sku: str, raw_source: str | Path | Sequence[Mapping[str, Any]] | None = None,
    custom_runner: Any = None, live_profiler: bool = False, endpoint_alias: str = "",
    profiler_root: str | Path | None = None, coordinator_bin: str = "", capture_root: str | Path | None = None,
    profile_override: Mapping[str, Any] | None = None, abi_rejection_results: Sequence[Mapping[str, Any]] | None = None,
    lease_acquirer: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], int]:
    """Acquire and normalize Stage-3 silicon rows via custom runner, live profiler + HostQueue, or offline replay."""
    req_npus, lease_pub, rows = _plan_required_ranks(plan_doc), {}, []
    abi_obs = [dict(r) for r in (abi_rejection_results or ()) if isinstance(r, Mapping)]
    if custom_runner is not None:
        rows, norm_abi = normalize_profiler_records(list(custom_runner(plan_doc)), plan_doc=plan_doc, sku=sku, endpoint_alias=endpoint_alias, capture_out_dir=capture_root)
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
    if profiler_root is None:
        try:
            b, abi = _canonical_resolve_builder_strict(family, recipe, impl, tuple(dtypes), meta or {})
            return str(b), str(abi)
        except UnmodeledCapabilityError as exc:
            raise ModelingError(f"profiler_builder_unmodeled:{exc}") from exc
    root_path = Path(profiler_root).resolve()
    if not (root_path / "mindie_profiler" / "bench" / "operators.py").is_file():
        raise ModelingError("profiler_owner_unavailable:missing:operators.py")
    saved_path, saved_mods = list(sys.path), {k: sys.modules.pop(k) for k in list(sys.modules) if k == "mindie_profiler" or k.startswith("mindie_profiler.")}
    sys.path.insert(0, str(root_path))
    try:
        mod = importlib.import_module(".".join(("mindie_profiler", "bench", "operators")))
    except Exception as exc:
        raise ModelingError(f"profiler_owner_unavailable:{exc}") from exc
    finally:
        sys.path[:] = saved_path
        for k in [m for m in sys.modules if m == "mindie_profiler" or m.startswith("mindie_profiler.")]:
            sys.modules.pop(k, None)
        sys.modules.update(saved_mods)
    err_cls, resolver = getattr(mod, "BenchUnmodeledOperatorError", RuntimeError), getattr(mod, "resolve_builder_strict", None)
    if not callable(resolver):
        raise ModelingError("profiler_owner_unavailable:resolve_builder_strict_missing")
    try:
        b, abi = resolver(family, recipe, impl, dtypes=tuple(dtypes), meta=meta or {})
        return str(b), str(abi)
    except err_cls as exc:
        raise ModelingError(f"profiler_builder_unmodeled:{exc}") from exc
