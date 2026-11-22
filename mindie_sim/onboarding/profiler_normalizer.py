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
_DEFAULT_SIM_MODEL = "deepseek-v2-lite"
_INDEXER_SIM_MODEL = "deepseek-v4.1-flash"
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
_DTYPE_BYTES: Mapping[str, int] = {"fp32": 4, "float32": 4, "int32": 4, "bf16": 2, "fp16": 2, "float16": 2, "int8": 1, "uint8": 1, "fp8_e4m3": 1, "mxfp8": 1}
_CAPTURE_SHAPE_KEYS = frozenset({
    "tokens", "sequences", "batch_size", "q_len", "query_tokens", "seq_len", "kv_len", "context_len",
    "page_count", "block_size", "block_table", "heads", "kv_heads", "head_dim", "latent", "rope", "width",
    "top_k", "topk", "sparse_count", "m", "n", "k", "e", "matrix_groups", "partial_count", "nbytes", "group_size",
    "dtype", "input_dtype", "weight_dtype", "output_dtype", "quant_mode", "quantization", "scale_dtype", "has_scales",
    "query_dtype", "cache_dtype", "query_layout", "input_layout", "paged", "parallel_topology", "batch_x_kv_heads",
})
_FAMILY_BINDINGS: Mapping[str, tuple[str, str, str, str]] = {
    "rmsnorm": ("rmsnorm", "rmsnorm", "rms_norm", "aclnnRmsNorm"),
    "add_rmsnorm": ("add_rmsnorm", "add_rmsnorm", "add_rms_norm", "aclnnAddRmsNorm"),
    "layernorm": ("layernorm", "layernorm", "layer_norm", "aclnnLayerNorm"),
    "attention": ("attention", "ascend_paged_mla_decode_arch22", "absorbed_mla", "aclnnFusedInferAttentionScoreV4"),
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
    del prof_sha
    from mindie_sim.compose import compose_case as compose_case_dict
    from mindie_sim.prediction.results import StepRequest
    from mindie_sim.prediction.spine import default_spine
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.stack_registry import LoweringRequest

    raw = compose_case_dict(model=_DEFAULT_SIM_MODEL, hardware=sku, **parse_topo_options(topo))
    md = dict(raw["model"])
    md["layers"], md["body_layers"], md["layer_plans"] = 1, 1, list(md["layer_plans"])[:1]
    prep = default_spine().lower(
        parse_case({**raw, "model": md}),
        LoweringRequest(step=StepRequest(phase=phase, tokens=int(q_tok), context=int(ctx_len), sequences=int(seqs), scope="body"), tier="mainline"),  # type: ignore[arg-type]
    )
    out: list[dict[str, Any]] = []
    for op in prep.graph.executable[0].ops:
        meta = dict(op.metadata)
        nabi = meta.get("native_attention_abi") if isinstance(meta.get("native_attention_abi"), Mapping) else {}
        pgeo = meta.get("paged_mla_geometry") if isinstance(meta.get("paged_mla_geometry"), Mapping) else {}
        out.append({
            "op_id": str(op.op_id), "family": str(op.family), "recipe": str(op.recipe),
            "dtype": str(op.dtype or "bf16"), "quant_mode": str(meta.get("quant_mode") or meta.get("quantization") or "none").strip().lower(),
            "local_shape": tuple(int(x) if isinstance(x, (int, float)) else x for x in (op.local_shape or ())),
            "deps": tuple(str(d) for d in (op.deps or ())), "link_bytes": int(round(float(op.link_bytes or 0.0))),
            "matrix_groups": int(meta.get("matrix_weight_bank_groups") or meta.get("matrix_groups") or 1),
            "group_axis": str(meta.get("group_axis") or ""),
            "query_layout": str(nabi.get("query_layout") or ""), "cache_layout": str(nabi.get("cache_layout") or ""),
            "output_layout": str(nabi.get("output_layout") or ""),
            "kv_len": int(pgeo["kv_lengths"][0]) if pgeo.get("kv_lengths") else 0,
            "page_count": int(pgeo["page_counts"][0]) if pgeo.get("page_counts") else 0,
        })
    return tuple(out)


def _lower_one_layer_step_ops(sku: str, topo: str, phase: str, q_tok: int, ctx_len: int, seqs: int, *, profile_override: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], ...]:
    with temporary_scaffold_profile(sku, profile_override):
        prof = resolve_sku_profile(sku, profile_override)
        if not isinstance(prof, Mapping):
            raise ModelingError(f"unknown_sku_profile_for_lowering:{sku}")
        return _lower_one_layer_cached(sku, sha256_bytes(canonical_json_bytes(prof)), topo, phase, q_tok, ctx_len, seqs)


def _lowered_op_to_profiler_node(pid: str, op_rec: Mapping[str, Any], *, attn_shape: Mapping[str, Any], total_tokens: int, local_heads: int, topo_opts: Mapping[str, int]) -> tuple[dict[str, Any], dict[str, Any]]:
    oid, fam, rec, loc = str(op_rec["op_id"]), str(op_rec["family"]), str(op_rec["recipe"]), tuple(op_rec["local_shape"])
    tp, dcp, nid = max(1, int(topo_opts.get("tp", 1))), max(1, int(topo_opts.get("dcp", 1))), f"{pid}:{oid}"
    if fam == "attention":
        sh = dict(attn_shape)
        if op_rec["kv_len"] > 0:
            sh["seq_len"] = sh["kv_len"] = int(op_rec["kv_len"])
            sh["page_count"] = int(op_rec["page_count"]) or _page_count(int(op_rec["kv_len"]))
        if op_rec["query_layout"]:
            sh["query_layout"] = sh["input_layout"] = op_rec["query_layout"]
            sh["cache_layout"] = op_rec["cache_layout"] or "PA_BSND"
            sh["output_layout"] = op_rec["output_layout"] or "NBSD"
        return (
            {"node_id": nid, "op_id": oid, "kind": "operator", "op": "absorbed_mla", "builder": "absorbed_mla", "family": "attention", "recipe": rec, "implementation": "aclnnFusedInferAttentionScoreV4", "dtype": "bf16", "paged": True, "layout": sh.get("query_layout", "TND"), "shape": sh},
            {"op_id": oid, "family": "attention", "recipe": rec, "builder": "absorbed_mla", "paged": True, "heads": sh["heads"], "kv_heads": 1, "latent": _MLA_LATENT_DIM, "rope": _MLA_ROPE_DIM, "kv_len": sh["seq_len"], "page_count": sh["page_count"], "query_layout": sh.get("query_layout", "TND")},
        )
    if fam in _GEMM_FAMILIES and len(loc) >= _MIN_GEMM_RANK:
        m_d, n_d, k_d, mg = int(loc[0]), int(loc[1]), int(loc[2]), max(1, int(op_rec["matrix_groups"]))
        dt, qm = str(op_rec.get("dtype") or "bf16").lower(), str(op_rec.get("quant_mode") or "none").lower()
        if mg > 1:
            return (
                {"node_id": nid, "op_id": oid, "kind": "operator", "op": "grouped_matmul", "builder": "grouped_matmul", "family": "grouped_gemm", "recipe": "grouped_matmul", "implementation": "aclnnGroupedMatmulV5", "dtype": dt, "group_list": [max(1, m_d // mg)] * mg, "group_list_type": 1, "shape": {"m": m_d, "n": n_d, "k": k_d, "e": mg, "matrix_groups": mg, "dtype": dt}},
                {"op_id": oid, "family": fam, "recipe": rec, "builder": "grouped_matmul", "dtype": dt, "quant_mode": qm, "m": m_d, "n": n_d, "k": k_d, "matrix_groups": mg},
            )
        return (
            {"node_id": nid, "op_id": oid, "kind": "operator", "op": "matmul", "builder": "matmul", "family": fam, "recipe": rec, "implementation": "aclnnMatmul", "dtype": dt, "shape": {"m": m_d, "n": n_d, "k": k_d, "matrix_groups": 1, "dtype": dt}},
            {"op_id": oid, "family": fam, "recipe": rec, "builder": "matmul", "dtype": dt, "quant_mode": qm, "m": m_d, "n": n_d, "k": k_d, "matrix_groups": 1},
        )
    if fam in {"rmsnorm", "add_rmsnorm", "swiglu"}:
        toks, width = (int(loc[0]) if len(loc) >= _MIN_2D_RANK else total_tokens), (int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH)
        bld = {"rmsnorm": "rms_norm", "add_rmsnorm": "add_rms_norm", "swiglu": "swiglu"}[fam]
        impl = {"rmsnorm": "aclnnRmsNorm", "add_rmsnorm": "aclnnAddRmsNorm", "swiglu": "aclnnSwiGlu"}[fam]
        return {"node_id": nid, "op_id": oid, "kind": "operator", "op": bld, "builder": bld, "family": fam, "recipe": rec, "implementation": impl, "dtype": "bf16", "shape": {"tokens": toks, "width": width}}, {"op_id": oid, "family": fam, "recipe": rec, "builder": bld, "width": width}
    if fam == "elementwise" and rec == "rotary":
        return {"node_id": nid, "op_id": oid, "kind": "operator", "op": "rotary_embedding", "builder": "rotary_embedding", "family": "elementwise", "recipe": "rotary", "implementation": "aclnnApplyRotaryPosEmbV2", "dtype": "bf16", "shape": {"tokens": total_tokens, "heads": local_heads, "head_dim": _MLA_ROPE_DIM}}, {"op_id": oid, "family": "elementwise", "recipe": "rotary", "builder": "rotary_embedding"}
    if fam == "kv_compress_epilog":
        toks = int(loc[0]) if len(loc) >= _MIN_2D_RANK else total_tokens
        dt = str(op_rec["dtype"] or "bf16").lower()
        bw = int(loc[-1]) if loc else (_MLA_LATENT_DIM + _MLA_ROPE_DIM) * _MLA_BF16_BYTES
        ew = max(1, bw // _DTYPE_BYTES.get(dt, _MLA_BF16_BYTES))
        return {"node_id": nid, "op_id": oid, "kind": "operator", "op": "kv_compress_epilog", "builder": "kv_compress_epilog", "family": fam, "recipe": rec, "implementation": "aclnnKvCompressEpilog", "dtype": dt, "shape": {"tokens": toks, "width": ew, "byte_width": bw}}, {"op_id": oid, "family": fam, "recipe": rec, "builder": "kv_compress_epilog", "dtype": dt, "width": ew}
    if fam in {"dcp_allgather", "allreduce"}:
        dt, raw_link = str(op_rec["dtype"] or "bf16").lower(), int(op_rec["link_bytes"])
        grp_axis = "dcp" if fam == "dcp_allgather" else (op_rec["group_axis"] or "tp")
        grp_sz = dcp if grp_axis in {"dcp", "cp"} else tp
        if fam == "dcp_allgather":
            nbytes = max(_MIN_COLLECTIVE_BYTES, (raw_link // max(1, dcp - 1)) if raw_link > 0 else _DEFAULT_COLLECTIVE_BYTES)
            bld, impl = "all_gather", "hcclAllGather"
        else:
            nbytes = max(_MIN_COLLECTIVE_BYTES, ((raw_link * grp_sz) // max(1, _RING_ALLREDUCE_FACTOR * (grp_sz - 1))) if (raw_link > 0 and grp_sz > 1) else (raw_link or _DEFAULT_COLLECTIVE_BYTES))
            bld, impl = "all_reduce", "hcclAllReduce"
        return {"node_id": nid, "op_id": oid, "kind": "collective", "op": bld, "builder": bld, "family": fam, "recipe": rec, "implementation": impl, "dtype": dt, "group": grp_axis, "group_size": grp_sz, "nbytes": nbytes, "shape": {"dtype": dt, "nbytes": nbytes, "group_size": grp_sz}}, {"op_id": oid, "family": fam, "recipe": rec, "builder": bld, "dtype": dt, "nbytes": nbytes, "group_size": grp_sz}
    if fam == "attention_merge":
        toks = int(loc[0]) if len(loc) >= _MIN_GEMM_RANK else total_tokens
        heads = int(loc[1]) if len(loc) >= _MIN_GEMM_RANK else local_heads
        hdim = int(loc[2]) if len(loc) >= _MIN_GEMM_RANK else _MLA_LATENT_DIM
        dt, partials = str(op_rec["dtype"] or "fp32").lower(), max(_MIN_DCP_PARTIALS, dcp)
        return {"node_id": nid, "op_id": oid, "kind": "operator", "op": "dcp_lse_attention_merge", "builder": "dcp_lse_attention_merge", "family": fam, "recipe": rec, "implementation": "aclnnRingAttentionUpdate", "dtype": dt, "shape": {"tokens": toks, "heads": heads, "kv_heads": 1, "head_dim": hdim, "partial_count": partials, "dtype": dt}}, {"op_id": oid, "family": fam, "recipe": rec, "builder": "dcp_lse_attention_merge", "dtype": dt, "partial_count": partials}
    width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
    return {"node_id": nid, "op_id": oid, "kind": "operator", "op": "vector_add", "builder": "vector_add", "family": "elementwise", "recipe": "residual_add", "implementation": "aclnnAdd", "dtype": "bf16", "shape": {"tokens": total_tokens, "width": width}}, {"op_id": oid, "family": fam, "recipe": rec, "builder": "vector_add"}


def resolve_point_binding_and_shape(
    plan_doc: Mapping[str, Any], pt: Mapping[str, Any], *, sku: str, profile_override: Mapping[str, Any] | None = None,
) -> tuple[str, str, str, str, str, dict[str, Any], dict[str, Any]]:
    plan_family = str(plan_doc.get("family") or "rmsnorm").strip()
    axis, val = str(pt.get("axis") or "context_len").strip(), int(pt["value"])
    batch, q_tok = max(1, int(pt.get("batch_size", 1))), max(1, int(pt.get("query_tokens", 1)))
    topo = str(pt.get("parallel_topology") or "tp1").strip()
    topk_val = int(pt["topk"]) if pt.get("topk") is not None else None
    is_abi_reject = str(pt.get("split") or "").strip().lower() == "abi_rejection"
    topo_opts = parse_topo_options(topo)
    tp, dcp, pcp, ep = max(1, topo_opts.get("tp", 1)), max(1, topo_opts.get("dcp", 1)), max(1, topo_opts.get("pcp", 1)), max(1, topo_opts.get("ep", 1))
    local_heads = max(1, _DEFAULT_ATTENTION_HEADS // tp)
    level = assign_plan_point_level(plan_doc, pt)
    model_overrides: dict[str, Any] = {}
    if topk_val is not None or plan_family in _SPARSE_INDEXER_FAMILIES:
        pfam, prec, pbld, impl = _FAMILY_BINDINGS["indexer"]
        sim_fam, model_name = "indexer", _INDEXER_SIM_MODEL
        ctx_len = val if axis == "context_len" else max(val, _DEFAULT_CONTEXT_FALLBACK)
        kv_len = ctx_len if is_abi_reject else (ctx_len + q_tok)
        shape: dict[str, Any] = {"tokens": max(1, batch * q_tok), "heads": _INDEXER_ATTENTION_HEADS, "kv_heads": 1, "head_dim": _DEFAULT_HEAD_DIM, "seq_len": kv_len, "kv_len": kv_len, "context_len": ctx_len, "batch_size": batch, "sequences": batch, "query_tokens": q_tok, "q_len": q_tok, "query_dtype": "bf16", "cache_dtype": "bf16", "input_layout": "TND", "query_layout": "TND", "parallel_topology": topo}
        if topk_val is not None:
            shape.update({"top_k": topk_val, "sparse_count": topk_val, "enforce_exact_topk": True})
            if not is_abi_reject:
                model_overrides = {"index_topk": topk_val, "index_dtype": "bf16"}
    elif (plan_family == "sku_suite" and axis in {"context_len", "batch_x_kv_heads"}) or plan_family == "attention":
        is_single_q_local = q_tok == 1 and tp == 1 and dcp == 1 and pcp == 1
        attn_rec = _decode_mla_recipe_for_sku(sku, profile_override=profile_override) if is_single_q_local else "latent_attention0"
        pfam, prec, pbld, impl = ("attention", attn_rec, "absorbed_mla", "aclnnFusedInferAttentionScoreV4")
        sim_fam, model_name = "attention", _DEFAULT_SIM_MODEL
        ctx_len = val if axis == "context_len" else _DEFAULT_CONTEXT_FALLBACK
        seqs = val if axis == "batch_x_kv_heads" else batch
        kv_len = ctx_len + q_tok
        if dcp > 1:
            la = next((o for o in _lower_one_layer_step_ops(sku, topo, "prefill" if q_tok > 1 else "decode", q_tok, ctx_len, seqs, profile_override=profile_override) if o["family"] == "attention"), None)
            if la and len(la["local_shape"]) >= _MIN_2D_RANK:
                kv_len = int(la["local_shape"][1])
        qlay = "BNSD" if is_single_q_local else "TND"
        shape = {"tokens": max(1, seqs * q_tok), "sequences": seqs, "batch_size": batch, "query_tokens": q_tok, "q_len": q_tok, "seq_len": kv_len, "kv_len": kv_len, "context_len": ctx_len, "page_count": _page_count(kv_len), "heads": local_heads, "kv_heads": 1, "head_dim": _MLA_LATENT_DIM, "latent": _MLA_LATENT_DIM, "kv_lora_rank": _MLA_LATENT_DIM, "rope": _MLA_ROPE_DIM, "qk_rope_head_dim": _MLA_ROPE_DIM, "block_size": _MLA_BLOCK_SIZE, "paged": True, "query_layout": qlay, "input_layout": qlay, "cache_layout": "PA_BSND", "output_layout": "NBSD" if is_single_q_local else "TND", "parallel_topology": topo}
        if axis == "batch_x_kv_heads":
            shape["batch_x_kv_heads"] = val
        if level == "L3" and plan_family == "sku_suite":
            model_overrides = {"layers": 1}
    elif plan_family == "sku_suite":
        pfam, prec, pbld, impl = _FAMILY_BINDINGS["rmsnorm"]
        sim_fam, model_name = "rmsnorm", _DEFAULT_SIM_MODEL
        shape = {"tokens": val, "width": _MLA_HIDDEN_WIDTH, "parallel_topology": topo}
    elif plan_family in _FAMILY_BINDINGS:
        pfam, prec, pbld, impl = _FAMILY_BINDINGS[plan_family]
        sim_fam = pfam
        model_name = _INDEXER_SIM_MODEL if sim_fam in {"indexer", "layernorm", "rotary"} else _DEFAULT_SIM_MODEL
        width = _MLA_HIDDEN_WIDTH if model_name == _DEFAULT_SIM_MODEL else _DEFAULT_HIDDEN_WIDTH
        tok_val = val if axis == "tokens" else max(1, (val if axis == "batch_x_kv_heads" else batch) * q_tok)
        if pbld == "matmul":
            shape = {"tokens": tok_val, "m": tok_val, "n": width, "k": max(1, width // tp), "dtype": "bf16", "parallel_topology": topo}
            topo_opts = {**topo_opts, "graph_capture_mode": "none"}
        elif pbld == "grouped_matmul":
            loc_e = max(1, _MOE_EXPERT_BANK // ep)
            shape = {"tokens": tok_val, "m": tok_val * _MOE_TOPK_EXPERTS, "n": max(1, _MOE_GATE_UP_WIDTH // tp), "k": width, "e": loc_e, "matrix_groups": loc_e, "dtype": "bf16", "parallel_topology": topo}
            topo_opts = {**topo_opts, "graph_capture_mode": "none"}
        elif pbld == "rotary_embedding":
            shape = {"tokens": tok_val, "heads": local_heads, "head_dim": _DEFAULT_HEAD_DIM, "parallel_topology": topo}
        else:
            shape = {"tokens": tok_val, "width": width, "parallel_topology": topo}
    else:
        pfam, prec, pbld, impl, sim_fam, model_name = plan_family, plan_family, "", str(plan_doc.get("target_id") or plan_family), plan_family, _DEFAULT_SIM_MODEL
        shape = {"tokens": val if axis == "tokens" else max(1, batch * q_tok), "width": _MLA_HIDDEN_WIDTH, "parallel_topology": topo}
    custom_sim = pt.get("sim_case") or plan_doc.get("sim_case")
    sim_case = dict(custom_sim) if isinstance(custom_sim, Mapping) else {"compose_model": model_name, "compose_hardware": sku, "compose_options": topo_opts, **({"model_overrides": model_overrides} if model_overrides else {})}
    return pfam, prec, pbld, impl, sim_fam, shape, sim_case


def _build_case_nodes(pid: str, *, sku: str, pt: Mapping[str, Any], level: str, pfam: str, prec: str, pbld: str, impl: str, shape: Mapping[str, Any], topo_opts: Mapping[str, int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    total_tokens = max(1, int(shape.get("tokens", 1)))
    local_heads = max(1, int(shape.get("heads", _DEFAULT_ATTENTION_HEADS // max(1, int(topo_opts.get("tp", 1))))))
    if level == "L1" or pbld != "absorbed_mla":
        node: dict[str, Any] = {"node_id": f"{pid}:op0", "kind": "operator", "op": pbld or prec, "family": pfam, "recipe": prec, "implementation": impl, "dtype": "bf16", "shape": dict(shape)}
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
        nodes = [
            {"node_id": f"{pid}:{attn_id}", "op_id": attn_id, "kind": "operator", "op": pbld, "builder": pbld, "family": pfam, "recipe": prec, "implementation": impl, "dtype": "bf16", "paged": True, "layout": str(shape.get("query_layout") or "TND"), "shape": dict(shape)},
            {"node_id": f"{pid}:s0.l0.v_expand", "op_id": "s0.l0.v_expand", "kind": "operator", "op": "grouped_matmul", "builder": "grouped_matmul", "family": "grouped_gemm", "recipe": "grouped_matmul", "implementation": "aclnnGroupedMatmulV5", "dtype": "bf16", "group_list": [total_tokens] * local_heads, "group_list_type": 1, "shape": {"m": v_m, "n": _MLA_HEAD_DIM, "k": _MLA_LATENT_DIM, "e": local_heads, "matrix_groups": local_heads, "dtype": "bf16"}},
        ]
        nbs = [
            {"op_id": attn_id, "family": "attention", "recipe": prec, "builder": pbld, "paged": True, "heads": local_heads, "kv_heads": 1, "latent": _MLA_LATENT_DIM, "rope": _MLA_ROPE_DIM, "kv_len": int(shape["seq_len"]), "page_count": int(shape["page_count"]), "query_layout": str(shape.get("query_layout") or "TND")},
            {"op_id": "s0.l0.v_expand", "family": "gemm", "recipe": "v_expand", "builder": "grouped_matmul", "dtype": "bf16", "quant_mode": "none", "m": v_m, "n": _MLA_HEAD_DIM, "k": _MLA_LATENT_DIM, "matrix_groups": local_heads},
        ]
        return nodes, [{"from": f"{pid}:{attn_id}", "to": f"{pid}:s0.l0.v_expand", "kind": "data"}], nbs, [attn_id, "s0.l0.v_expand"]
    lowered = _lower_one_layer_step_ops(sku, str(pt.get("parallel_topology") or "tp1"), "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode", max(1, int(pt.get("query_tokens", 1))), int(shape.get("context_len", pt["value"])), max(1, int(shape.get("sequences", pt.get("batch_size", 1)))))
    op_ids = {r["op_id"] for r in lowered}
    nodes, edges, nbs, tids = [], [], [], []
    for rec in lowered:
        nd, nb = _lowered_op_to_profiler_node(pid, rec, attn_shape=shape, total_tokens=total_tokens, local_heads=local_heads, topo_opts=topo_opts)
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
        target_op_id = (
            ("s0.l0.paged_mla_decode" if prec in {"ascend_paged_mla_decode_arch22", "ascend_paged_mla_decode_arch35"} else "s0.l0.latent_attention0") if sim_fam == "attention"
            else ("s0.l0.norm1" if sim_fam == "rmsnorm" else ("s0.l2.index_quant_lightning" if sim_fam == "indexer" else ("s0.l1.expert_up" if pbld == "grouped_matmul" else ("s0.l0.output" if pbld == "matmul" else ""))))
        )
        _nodes, _edges, nbs, tids = _build_case_nodes(pid, sku=sku, pt=pt, level=level, pfam=pfam, prec=prec, pbld=pbld, impl=impl, shape=shape, topo_opts=topo_opts)
    binding: dict[str, Any] = {"builder": pbld, "family": sim_fam, "recipe": prec, "implementation": impl, "target_op_id": target_op_id}
    for k in ("paged", "heads", "kv_heads", "head_dim", "latent", "rope", "width", "kv_len", "page_count", "query_layout", "input_layout", "query_dtype", "cache_dtype", "sparse_count", "topk", "dtype", "m", "n", "k", "matrix_groups"):
        if k in shape:
            binding[k] = shape[k]
    if pbld in {"matmul", "grouped_matmul"} or sim_fam in _GEMM_FAMILIES:
        binding.setdefault("dtype", str(shape.get("dtype") or "bf16"))
        binding.setdefault("weight_dtype", str(shape.get("weight_dtype") or binding["dtype"]))
        binding.setdefault("quant_mode", str(shape.get("quant_mode") or "none"))
        binding.setdefault("matrix_groups", int(shape.get("matrix_groups") or shape.get("e") or 1))
    if nbs:
        binding["node_bindings"] = nbs
    if tids:
        binding["timed_op_ids"] = tids
    return {"level": level, "phase": "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode", "target_family": sim_fam, "target_op_id": target_op_id, "target_op_ids": list(tids) if level == "L2" else [], "measured_binding": binding, "sim_case": sim_case, "expected_shape": shape, "profiler_family": pfam, "profiler_recipe": prec, "profiler_builder": pbld, "implementation": impl}


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
        by_slot: dict[Any, list[int]] = {}
        ranks_seen: set[int] = set()
        for idx_g, row in enumerate(group):
            dur_ms = float(row.get("duration_ms") or 0.0)
            if dur_ms <= 0.0:
                raise ModelingError(f"profiler_sample_duration_invalid:{cid}:duration_ms={dur_ms}")
            ranks_seen.add(int(row["rank"]) if row.get("rank") is not None else 0)
            by_slot.setdefault(row.get("repeat_index", row.get("trial", idx_g)), []).append(int(round(dur_ms * NS_PER_MS)))
        durs_ns = sorted(max(vals) for vals in by_slot.values())
        p50_ns, p99_ns = _quantile_ns(durs_ns, _P50_Q), _quantile_ns(durs_ns, _P99_Q)
        first = group[0]
        tmode = str(first.get("timing_mode") or "eager").strip().lower()
        clock_op = str(first.get("clock_operator") or ("busy_union" if tmode in _BUSY_UNION_MODES else DEFAULT_CLOCK_OPERATOR)).strip()
        if clock_op not in VALID_CLOCK_OPERATORS:
            clock_op = DEFAULT_CLOCK_OPERATOR
        level = binfo["level"] if binfo is not None else str(first.get("level") or "L1").upper()
        explicit_acq = str(first.get("acquisition_id") or (rcpts[0].get("acquisition_id") if rcpts else "") or "").strip()
        base_acq = str((plan_doc.get("train_lineage_root") if split == "train" else plan_doc.get("heldout_lineage_root")) or f"acq-{split}")
        acq_id = explicit_acq or (f"{base_acq}-{level.lower()}" if split == "heldout" else base_acq)
        base_rid = _TRIAL_SUFFIX_RE.sub("", str(first.get("run_id") or f"run-{cid}"))
        pub_rid = sanitize_public_run_id(base_rid)
        timed_nodes = list(next((r["timed_nodes"] for r in (*rcpts, *group) if isinstance(r.get("timed_nodes"), Sequence) and not isinstance(r.get("timed_nodes"), (str, bytes)) and r["timed_nodes"]), ()))
        dims = extract_shape_dims(actual_shape, {"point_id": cid, "axis": str((plan_pt or {}).get("axis") or "")}, plan_pt=plan_pt, expected_shape=binfo["expected_shape"] if binfo else None, measured_binding=binfo["measured_binding"] if binfo else None, cid=cid)
        clean_shape = _sanitize_capture_shape(actual_shape)
        cap_obj = {"schema": "mindie-sim.onboarding-capture/1", "case_id": cid, "sku": sku, "split": split, "level": level, "acquisition_root": acq_id, "run_uid": pub_rid, "endpoint_alias": clean_ep, "clock_operator": clock_op, "measured_ns": p50_ns, "measured_p99_ns": p99_ns, "device_ranks": sorted(ranks_seen), "event_ids": [f"{pub_rid}-s", f"{pub_rid}-e"], "actual_shape": clean_shape, "timed_nodes": timed_nodes}
        cap_bytes = canonical_json_bytes(cap_obj)
        raw_sha = sha256_bytes(cap_bytes)
        if capture_out_dir is not None:
            write_atomic_bytes(Path(capture_out_dir) / f"{raw_sha}.json", cap_bytes)
        row_out: dict[str, Any] = {
            "case_id": cid, "point_id": cid, "sku": sku, "split": split, "level": level, "status": "ok",
            "acquisition_root": acq_id, "run_uid": pub_rid, "raw_sha256": raw_sha,
            "capture_uri": sanitize_public_capture_uri("", raw_sha256=raw_sha, default_prefix=f"capture://{split}"),
            "device_ranks": sorted(ranks_seen), "event_ids": [f"{pub_rid}-s", f"{pub_rid}-e"],
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
