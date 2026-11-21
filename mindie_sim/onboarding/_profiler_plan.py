"""Executable ``mindie-bench-subgraph-plan/1`` builder and writer-clock/geometry helpers.

[Role] Translate onboarding cliff/SKU plans into executable L1/L2/L3
  ``mindie-bench-subgraph-plan/1`` plans with exact ``ParallelSpec`` world
  sizes, batch-preserving canonical ``mindie-profiler`` operator bindings
  (matched 1:1 to lowered ``mindie-sim`` recipes, layouts, and tensor
  geometry), and ``sim_case`` / ``measured_binding`` metadata; also parse
  ``run_id.<arm>.<mode>.t<trial>`` suffixes, derive ``clock_operator`` strictly
  from writer timing evidence, filter multi-mode writer observations without
  collisions, extract executed geometry, and verify builder-reported geometry
  against the plan before sealing any capture.
[Invariants]
  * Reuses ``required_ranks_for_plan`` / ``validate_parallel_topology`` (never
    divergent world-size arithmetic).
  * Never exports unmodeled ``family="sku_suite"`` nodes; every point maps to
    real executable L1/L2/L3 nodes + full batch-preserving tensor geometry.
  * Never overwrites writer-measured ``clock_operator`` or builder-reported
    ``actual_shape`` with the plan; fails closed with
    ``clock_operator_plan_mismatch`` or ``profiler_actual_geometry_mismatch``
    before sealing.
[Timeouts & Locks] Pure in-memory plan/record helpers; zero I/O, locks, or
  subprocesses.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import re
from typing import Any

from mindie_sim.backends import ascend_profiles
from mindie_sim.schema.common import ModelingError

from .contracts import VALID_CLOCK_OPERATORS

_DEFAULT_ATTENTION_HEADS = 16
_INDEXER_ATTENTION_HEADS = 32
_DEFAULT_ATTENTION_KV_HEADS = 1
_MLA_LATENT_DIM = 512
_MLA_ROPE_DIM = 64
_MLA_HEAD_DIM = 128
_MLA_BLOCK_SIZE = 128
_MLA_HIDDEN_WIDTH = 2048
_MLA_Q_PROJ_OUT = 3072
_MLA_BF16_BYTES = 2
_DEFAULT_HEAD_DIM = 128
_DEFAULT_HIDDEN_WIDTH = 4096
_DEFAULT_CONTEXT_FALLBACK = 2048
_L2_CONTEXT_MIN = 24576
_L3_CONTEXT_MIN = 200000
_ARCH35_SKU_MARKER = "950"
_ARCH35_PRESET_MARKER = "35"
_DEFAULT_SIM_MODEL = "deepseek-v2-lite"
_INDEXER_SIM_MODEL = "deepseek-v4.1-flash"
_SUBGRAPH_PLAN_SCHEMA = "mindie-bench-subgraph-plan/1"
_TRIAL_SUFFIX_RE = re.compile(r"\.[^.]+\.t\d+$")
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")
_KNOWN_ARMS = frozenset({"base", "aclgraph"})
_L2_REGIME_TAGS = frozenset({"high_batch_decode_2d", "mtp_decode_2d", "chunked_prefill_2d"})
_LEVEL_BOUNDARY_KINDS: Mapping[str, str] = {
    "L1": "device_kernel",
    "L2": "fx_region",
    "L3": "model_step",
}
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


def _int_from(mapping: Mapping[str, Any], *keys: str) -> int | None:
    for key in keys:
        val = mapping.get(key)
        if val is not None and not isinstance(val, bool):
            try:
                return int(val)
            except (TypeError, ValueError):
                continue
    return None


def extract_shape_dims(
    shape: Mapping[str, Any],
    plan_key: Mapping[str, Any],
    *,
    plan_pt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract canonical integer onboarding coordinates strictly from executed shape + plan key."""
    dims: dict[str, Any] = {}
    axis = str((plan_pt or {}).get("axis") or plan_key.get("axis") or "").strip()
    merged = {**dict(plan_key), **dict(shape)}

    ctx_val = _int_from(shape, "context_len", "seq_len", "context", "kv_len", "scanned_rows")
    if ctx_val is None:
        ctx_val = _int_from(plan_key, "context_len", "seq_len", "context", "kv_len", "scanned_rows")
    if ctx_val is not None:
        dims["context_len"] = ctx_val

    tok_val = _int_from(shape, "tokens", "m", "num_tokens")
    if tok_val is None:
        tok_val = _int_from(plan_key, "tokens", "m", "num_tokens")
    if tok_val is not None:
        dims["tokens"] = tok_val

    q_val = _int_from(shape, "q_len", "query_tokens", "q")
    if q_val is None:
        q_val = _int_from(plan_key, "q_len", "query_tokens", "q")
    dims["query_tokens"] = q_val if q_val is not None else 1

    seq_val = _int_from(shape, "sequences", "batch_size", "batch", "b")
    kv_val = _int_from(shape, "kv_heads")
    bxkv_val = _int_from(shape, "batch_x_kv_heads")
    if bxkv_val is None and seq_val is not None and kv_val is not None and axis == "batch_x_kv_heads":
        bxkv_val = seq_val * kv_val
    if bxkv_val is not None:
        dims["batch_x_kv_heads"] = bxkv_val

    if seq_val is not None:
        dims["sequences"] = seq_val
    if axis == "batch_x_kv_heads":
        b_plan = _int_from(plan_pt or {}, "batch_size") or _int_from(plan_key, "batch_size") or 1
        dims["batch_size"] = b_plan
    elif seq_val is not None:
        dims["batch_size"] = seq_val
    else:
        dims["batch_size"] = _int_from(plan_key, "batch_size", "batch", "b", "sequences") or 1

    tk_val = _int_from(shape, "top_k", "topk", "topk_items")
    if tk_val is None:
        tk_val = _int_from(plan_key, "top_k", "topk", "topk_items")
    if tk_val is not None:
        dims["topk"] = tk_val

    dims["parallel_topology"] = str(merged.get("parallel_topology") or "tp1")
    return dims


def parse_topo_options(topo: str) -> dict[str, int]:
    """Parse canonical topology tokens into ``compose_options``."""
    return {k: int(v) for k, v in _TOPO_TOKEN_RE.findall(str(topo or "tp1").lower())} or {"tp": 1}


def _decode_mla_recipe_for_sku(sku: str) -> str:
    prof = ascend_profiles().get(sku)
    preset = ""
    if isinstance(prof, Mapping):
        dev = prof.get("device")
        if isinstance(dev, Mapping):
            ca = dev.get("compute_arch")
            if isinstance(ca, Mapping) and ca.get("preset"):
                preset = str(ca["preset"]).strip()
    if _ARCH35_PRESET_MARKER in preset or _ARCH35_SKU_MARKER in str(sku).lower():
        return "ascend_paged_mla_decode_arch35"
    return "ascend_paged_mla_decode_arch22"


def assign_plan_point_level(plan_doc: Mapping[str, Any], pt: Mapping[str, Any]) -> str:
    """Deterministically assign ``L1`` / ``L2`` / ``L3`` for a plan point."""
    explicit = str(pt.get("level") or "").strip().upper()
    if explicit in _LEVEL_BOUNDARY_KINDS:
        return explicit
    if str(plan_doc.get("family") or "").strip() != "sku_suite":
        return "L1"
    if str(pt.get("split") or "").strip().lower() != "heldout":
        return "L1"
    topo = str(pt.get("parallel_topology") or "tp1").strip().lower()
    tag = str(pt.get("regime_tag") or "").strip()
    val = int(pt.get("value") or 0)
    if topo != "tp1" or (tag == "off_grid_heldout" and val >= _L3_CONTEXT_MIN):
        return "L3"
    if tag in _L2_REGIME_TAGS or (tag == "off_grid_heldout" and _L2_CONTEXT_MIN <= val < _L3_CONTEXT_MIN):
        return "L2"
    return "L1"


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
    tp = max(1, int(topo_opts.get("tp", 1)))
    dcp = max(1, int(topo_opts.get("dcp", 1)))
    pcp = max(1, int(topo_opts.get("pcp", 1)))
    local_mla_heads = max(1, _DEFAULT_ATTENTION_HEADS // tp)
    decode_mla_rec = _decode_mla_recipe_for_sku(sku)
    custom_sim = pt.get("sim_case") or plan_doc.get("sim_case")
    is_default_mla_model = not (
        isinstance(custom_sim, Mapping)
        and custom_sim.get("compose_model")
        and str(custom_sim["compose_model"]) != _DEFAULT_SIM_MODEL
    )

    if topk_val is not None or plan_family in _SPARSE_INDEXER_FAMILIES:
        pfam, prec, pbld, impl = _FAMILY_BINDINGS["indexer"]
        sim_fam, model_name = "indexer", _INDEXER_SIM_MODEL
        ctx_len = val if axis == "context_len" else max(val, _DEFAULT_CONTEXT_FALLBACK)
        shape: dict[str, Any] = {
            "tokens": max(1, batch * q_tok), "width": _DEFAULT_HEAD_DIM,
            "heads": _INDEXER_ATTENTION_HEADS, "kv_heads": 1, "head_dim": _DEFAULT_HEAD_DIM,
            "seq_len": ctx_len, "context_len": ctx_len,
            "batch_size": batch, "sequences": batch, "query_tokens": q_tok, "q_len": q_tok,
            "parallel_topology": topo,
        }
        if topk_val is not None:
            shape["top_k"] = topk_val
            shape["topk"] = topk_val
            shape["sparse_count"] = topk_val
            shape["enforce_exact_topk"] = True
    elif (plan_family == "sku_suite" and axis in {"context_len", "batch_x_kv_heads"}) or (
        plan_family == "attention" and is_default_mla_model
    ):
        is_single_q_local = q_tok == 1 and tp == 1 and dcp == 1 and pcp == 1
        attn_rec = decode_mla_rec if is_single_q_local else "latent_attention0"
        pfam, prec, pbld, impl = ("attention", attn_rec, "absorbed_mla", "aclnnFusedInferAttentionScoreV4")
        sim_fam, model_name = "attention", _DEFAULT_SIM_MODEL
        ctx_len = val if axis == "context_len" else _DEFAULT_CONTEXT_FALLBACK
        seqs = val if axis == "batch_x_kv_heads" else batch
        total_tokens = max(1, seqs * q_tok)
        shape = {
            "tokens": total_tokens, "sequences": seqs, "batch_size": batch,
            "query_tokens": q_tok, "q_len": q_tok,
            "seq_len": ctx_len, "context_len": ctx_len,
            "heads": local_mla_heads, "kv_heads": _DEFAULT_ATTENTION_KV_HEADS,
            "head_dim": _MLA_LATENT_DIM, "latent": _MLA_LATENT_DIM, "kv_lora_rank": _MLA_LATENT_DIM,
            "rope": _MLA_ROPE_DIM, "qk_rope_head_dim": _MLA_ROPE_DIM,
            "block_size": _MLA_BLOCK_SIZE, "paged": True, "parallel_topology": topo,
        }
        if axis == "batch_x_kv_heads":
            shape["batch_x_kv_heads"] = val
    elif plan_family == "sku_suite":
        pfam, prec, pbld, impl = _FAMILY_BINDINGS["rmsnorm"]
        sim_fam, model_name = "rmsnorm", _DEFAULT_SIM_MODEL
        shape = {
            "tokens": val, "width": _MLA_HIDDEN_WIDTH,
            "batch_size": batch, "sequences": batch,
            "query_tokens": q_tok, "q_len": q_tok, "parallel_topology": topo,
        }
    elif plan_family == "attention":
        pfam, prec, pbld, impl = ("attention", "fused_infer_attention", "fused_infer_attention", "aclnnFusedInferAttentionScoreV3")
        sim_fam, model_name = "attention", _DEFAULT_SIM_MODEL
        ctx_len = val if axis == "context_len" else _DEFAULT_CONTEXT_FALLBACK
        seqs = val if axis == "batch_x_kv_heads" else batch
        shape = {
            "tokens": max(1, seqs * q_tok), "sequences": seqs, "batch_size": batch,
            "query_tokens": q_tok, "q_len": q_tok,
            "seq_len": ctx_len, "context_len": ctx_len,
            "heads": local_mla_heads, "kv_heads": _DEFAULT_ATTENTION_KV_HEADS,
            "head_dim": _DEFAULT_HEAD_DIM, "block_size": _MLA_BLOCK_SIZE,
            "paged": True, "parallel_topology": topo,
        }
        if axis == "batch_x_kv_heads":
            shape["batch_x_kv_heads"] = val
    elif plan_family in _FAMILY_BINDINGS:
        pfam, prec, pbld, impl = _FAMILY_BINDINGS[plan_family]
        sim_fam = pfam
        model_name = _INDEXER_SIM_MODEL if sim_fam in {"indexer", "layernorm", "rotary"} else _DEFAULT_SIM_MODEL
        width = _MLA_HIDDEN_WIDTH if model_name == _DEFAULT_SIM_MODEL else _DEFAULT_HIDDEN_WIDTH
        ctx_len = val if axis == "context_len" else _DEFAULT_CONTEXT_FALLBACK
        seqs = val if axis == "batch_x_kv_heads" else batch
        tok_val = val if axis == "tokens" else max(1, seqs * q_tok)
        shape = {
            "tokens": tok_val, "width": width, "m": tok_val, "n": width, "k": width,
            "heads": local_mla_heads, "kv_heads": _DEFAULT_ATTENTION_KV_HEADS,
            "head_dim": _DEFAULT_HEAD_DIM, "seq_len": ctx_len,
            "batch_size": batch, "sequences": seqs,
            "query_tokens": q_tok, "q_len": q_tok, "parallel_topology": topo, axis: val,
        }
    else:
        pfam, prec, pbld = plan_family, plan_family, ""
        impl, sim_fam, model_name = str(plan_doc.get("target_id") or plan_family), plan_family, _DEFAULT_SIM_MODEL
        shape = {
            "tokens": val if axis == "tokens" else max(1, batch * q_tok),
            "width": _MLA_HIDDEN_WIDTH, "batch_size": batch, "sequences": batch,
            "query_tokens": q_tok, "q_len": q_tok, "parallel_topology": topo, axis: val,
        }

    sim_case = (
        dict(custom_sim)
        if isinstance(custom_sim, Mapping)
        else {"compose_model": model_name, "compose_hardware": sku, "compose_options": topo_opts}
    )
    return pfam, prec, pbld, impl, sim_fam, shape, sim_case


def resolve_point_measured_binding(
    plan_doc: Mapping[str, Any],
    pt: Mapping[str, Any],
    *,
    sku: str,
) -> dict[str, Any]:
    """Build exact 1:1 replay target & measured-operator binding metadata for ``pt``."""
    pfam, prec, pbld, impl, sim_fam, shape, sim_case = resolve_point_binding_and_shape(
        plan_doc, pt, sku=sku
    )
    level = assign_plan_point_level(plan_doc, pt)
    if sim_fam == "attention":
        target_op_id = (
            "s0.l0.paged_mla_decode"
            if prec in {"ascend_paged_mla_decode_arch22", "ascend_paged_mla_decode_arch35"}
            else "s0.l0.latent_attention0"
        )
    elif sim_fam == "rmsnorm":
        target_op_id = "s0.l0.norm1"
    elif sim_fam == "indexer":
        target_op_id = "s0.l2.index_quant_lightning"
    else:
        target_op_id = ""
    target_op_ids = [target_op_id, "s0.l0.v_expand"] if (level == "L2" and sim_fam == "attention" and target_op_id) else []
    binding: dict[str, Any] = {
        "builder": pbld, "family": sim_fam, "recipe": prec, "implementation": impl,
        "target_op_id": target_op_id,
    }
    for key in ("paged", "heads", "kv_heads", "head_dim", "latent", "rope", "width"):
        if key in shape:
            binding[key] = shape[key]
    return {
        "level": level,
        "phase": "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode",
        "target_family": sim_fam,
        "target_op_id": target_op_id,
        "target_op_ids": target_op_ids,
        "measured_binding": binding,
        "sim_case": sim_case,
        "expected_shape": shape,
        "profiler_family": pfam,
        "profiler_recipe": prec,
        "profiler_builder": pbld,
        "implementation": impl,
    }


def _build_case_nodes_and_edges(
    pid: str,
    *,
    level: str,
    pfam: str,
    prec: str,
    pbld: str,
    impl: str,
    shape: Mapping[str, Any],
    topo_opts: Mapping[str, int],
    pt_ranks: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    is_paged_attn = pbld in {"absorbed_mla", "fused_infer_attention"}
    total_tokens = max(1, int(shape.get("tokens", 1)))
    tp = max(1, int(topo_opts.get("tp", 1)))
    local_heads = max(1, int(shape.get("heads", _DEFAULT_ATTENTION_HEADS // tp)))

    if level == "L1" or pbld != "absorbed_mla":
        node: dict[str, Any] = {
            "node_id": f"{pid}:op0", "kind": "operator",
            "op": pbld or prec, "family": pfam, "recipe": prec,
            "implementation": impl, "dtype": "bf16", "shape": dict(shape),
        }
        if pbld:
            node["builder"] = pbld
        if is_paged_attn:
            node["paged"] = True
        return [node], []

    attn_node: dict[str, Any] = {
        "node_id": f"{pid}:attn", "kind": "operator",
        "op": pbld, "builder": pbld, "family": pfam, "recipe": prec,
        "implementation": impl, "dtype": "bf16", "paged": True, "shape": dict(shape),
    }
    v_exp_m = total_tokens * local_heads
    v_exp_node: dict[str, Any] = {
        "node_id": f"{pid}:v_expand", "kind": "operator",
        "op": "matmul", "builder": "matmul", "family": "gemm", "recipe": "v_expand",
        "implementation": "aclnnMatmul", "dtype": "bf16",
        "shape": {
            "m": v_exp_m, "n": _MLA_HEAD_DIM, "k": _MLA_LATENT_DIM,
            "tokens": v_exp_m, "width": _MLA_LATENT_DIM,
        },
    }
    if level == "L2":
        return [attn_node, v_exp_node], [{"from": f"{pid}:attn", "to": f"{pid}:v_expand", "kind": "data"}]

    norm_node: dict[str, Any] = {
        "node_id": f"{pid}:norm1", "kind": "operator",
        "op": "rms_norm", "builder": "rms_norm", "family": "rmsnorm", "recipe": "rmsnorm",
        "implementation": "aclnnRmsNorm", "dtype": "bf16",
        "shape": {"tokens": total_tokens, "width": _MLA_HIDDEN_WIDTH},
    }
    q_proj_node: dict[str, Any] = {
        "node_id": f"{pid}:q_proj", "kind": "operator",
        "op": "matmul", "builder": "matmul", "family": "gemm", "recipe": "q_proj",
        "implementation": "aclnnMatmul", "dtype": "bf16",
        "shape": {
            "m": total_tokens, "n": max(1, _MLA_Q_PROJ_OUT // tp), "k": _MLA_HIDDEN_WIDTH,
            "tokens": total_tokens, "width": _MLA_HIDDEN_WIDTH,
        },
    }
    out_k = local_heads * _MLA_HEAD_DIM
    out_node: dict[str, Any] = {
        "node_id": f"{pid}:output", "kind": "operator",
        "op": "matmul", "builder": "matmul", "family": "gemm", "recipe": "output",
        "implementation": "aclnnMatmul", "dtype": "bf16",
        "shape": {
            "m": total_tokens, "n": _MLA_HIDDEN_WIDTH, "k": out_k,
            "tokens": total_tokens, "width": out_k,
        },
    }
    nodes = [norm_node, q_proj_node, attn_node, v_exp_node, out_node]
    edges = [
        {"from": f"{pid}:norm1", "to": f"{pid}:q_proj", "kind": "data"},
        {"from": f"{pid}:q_proj", "to": f"{pid}:attn", "kind": "stream"},
        {"from": f"{pid}:attn", "to": f"{pid}:v_expand", "kind": "data"},
        {"from": f"{pid}:v_expand", "to": f"{pid}:output", "kind": "data"},
    ]
    if pt_ranks > 1 and tp > 1:
        nbytes = total_tokens * _MLA_HIDDEN_WIDTH * _MLA_BF16_BYTES
        nodes.append({
            "node_id": f"{pid}:output_reduce", "kind": "collective",
            "op": "all_reduce", "builder": "all_reduce", "family": "allreduce",
            "recipe": "output_reduce", "implementation": "hcclAllReduce", "dtype": "bf16",
            "group": "tp", "group_size": tp, "nbytes": nbytes,
            "shape": {"tokens": total_tokens, "width": _MLA_HIDDEN_WIDTH, "nbytes": nbytes},
        })
        edges.append({"from": f"{pid}:output", "to": f"{pid}:output_reduce", "kind": "data"})
    return nodes, edges


def build_profiler_subgraph_plan(
    plan_doc: Mapping[str, Any], *, sku: str, split_tag: str = "", acquisition_id: str = ""
) -> dict[str, Any]:
    """Translate an onboarding cliff/SKU plan into executable L1/L2/L3 ``mindie-bench-subgraph-plan/1``."""
    from .cliff_planner import required_ranks_for_plan, validate_parallel_topology

    if str(plan_doc.get("schema") or "") == _SUBGRAPH_PLAN_SCHEMA:
        out = dict(plan_doc)
        out.setdefault("plan_id", str(plan_doc.get("plan_sha256") or plan_doc.get("plan_id") or "onboard"))
        out.setdefault("plan_sha256", str(plan_doc.get("plan_sha256") or ""))
        out.setdefault("sku", sku)
        if acquisition_id:
            out["acquisition_id"] = acquisition_id
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
            axis = str(pt.get("axis") or "context_len").strip()
            topo = str(pt.get("parallel_topology") or "tp1")
            pt_ranks = validate_parallel_topology(topo)
            topo_opts = parse_topo_options(topo)
            pfam, prec, pbld, impl, _sim_fam, shape, _sim_case = resolve_point_binding_and_shape(
                plan_doc, pt, sku=sku
            )
            level = assign_plan_point_level(plan_doc, pt)
            nodes, edges = _build_case_nodes_and_edges(
                pid, level=level, pfam=pfam, prec=prec, pbld=pbld, impl=impl,
                shape=shape, topo_opts=topo_opts, pt_ranks=pt_ranks,
            )
            case_entry: dict[str, Any] = {
                "case_id": pid, "split": str(pt.get("split") or "heldout"),
                "level": level, "boundary": _LEVEL_BOUNDARY_KINDS[level],
                "phase": "prefill" if int(pt.get("query_tokens", 1)) > 1 else "decode",
                "archetype": str(pt.get("regime_tag") or "onboarding"),
                "stage": "s1_joint" if pt_ranks > 1 else "s0_local",
                "clock_operator": str(pt.get("clock_operator") or "elapsed_span"),
                "implementation": impl, "dtype": "bf16", "shape": dict(shape),
                "batch": int(shape.get("sequences", pt.get("batch_size", 1))),
                "plan_key": {
                    "point_id": pid, "axis": axis, "family": pfam, "recipe": prec,
                    "abi": impl, "parallel_topology": topo, "level": level,
                    "batch_size": int(pt.get("batch_size", 1)),
                },
                "nodes": nodes, "edges": edges,
            }
            if pbld in {"absorbed_mla", "fused_infer_attention"}:
                case_entry["paged"] = True
            if world > 1:
                case_entry["group_ranks"] = list(range(pt_ranks))
            if pt_ranks > 1:
                case_entry["parallel"] = topo_opts
            cases.append(case_entry)
    plan_out: dict[str, Any] = {
        "schema": _SUBGRAPH_PLAN_SCHEMA, "plan_id": plan_id, "plan_sha256": plan_sha,
        "target_id": str(plan_doc.get("target_id") or ""), "sku": sku,
        "stage": "s1_joint" if world > 1 else "s0_local", "world_size": world,
        "execution_modes": ["eager"], "cases": cases,
    }
    if acquisition_id:
        plan_out["acquisition_id"] = acquisition_id
    return plan_out


def verify_actual_geometry_matches_plan(
    actual_shape: Mapping[str, Any],
    plan_pt: Mapping[str, Any],
    *,
    expected_shape: Mapping[str, Any],
    cid: str,
) -> None:
    """Compare builder-reported executed geometry against the plan BEFORE sealing a capture."""
    if not actual_shape:
        return
    axis = str(plan_pt.get("axis") or "").strip()
    exp_val = int(plan_pt["value"])
    exp_q = max(1, int(plan_pt.get("query_tokens", 1)))
    exp_seqs = _int_from(expected_shape, "sequences")
    exp_tokens = _int_from(expected_shape, "tokens")

    got_seqs = _int_from(actual_shape, "sequences")
    if exp_seqs is not None and got_seqs is not None and got_seqs != exp_seqs:
        raise ModelingError(
            f"profiler_actual_geometry_mismatch:{cid}:sequences:expected={exp_seqs}:actual={got_seqs}"
        )
    got_q = _int_from(actual_shape, "q_len", "query_tokens")
    if got_q is not None and got_q != exp_q:
        raise ModelingError(
            f"profiler_actual_geometry_mismatch:{cid}:q_len:expected={exp_q}:actual={got_q}"
        )
    got_toks = _int_from(actual_shape, "tokens")
    if exp_tokens is not None and got_toks is not None and got_toks != exp_tokens:
        raise ModelingError(
            f"profiler_actual_geometry_mismatch:{cid}:tokens:expected={exp_tokens}:actual={got_toks}"
        )
    if axis == "context_len":
        got_ctx = _int_from(actual_shape, "seq_len", "context_len")
        if got_ctx is not None and got_ctx != exp_val:
            raise ModelingError(
                f"profiler_actual_geometry_mismatch:{cid}:seq_len:expected={exp_val}:actual={got_ctx}"
            )
    if axis == "batch_x_kv_heads":
        got_bxkv = _int_from(actual_shape, "batch_x_kv_heads")
        got_kv = _int_from(actual_shape, "kv_heads")
        if got_bxkv is not None and got_bxkv != exp_val:
            raise ModelingError(
                f"profiler_actual_geometry_mismatch:{cid}:batch_x_kv_heads:expected={exp_val}:actual={got_bxkv}"
            )
        if got_seqs is not None and got_kv is not None and got_seqs * got_kv != exp_val:
            raise ModelingError(
                f"profiler_actual_geometry_mismatch:{cid}:batch_x_kv_heads:expected={exp_val}:actual={got_seqs * got_kv}"
            )
    if plan_pt.get("topk") is not None:
        exp_tk = int(plan_pt["topk"])
        got_tk = _int_from(actual_shape, "top_k", "topk", "sparse_count")
        if got_tk is not None and got_tk != exp_tk:
            raise ModelingError(
                f"profiler_actual_geometry_mismatch:{cid}:top_k:expected={exp_tk}:actual={got_tk}"
            )
    for field in ("kv_heads", "heads", "head_dim", "latent", "width"):
        exp_f = _int_from(expected_shape, field)
        got_f = _int_from(actual_shape, field)
        if exp_f is not None and got_f is not None and got_f != exp_f:
            raise ModelingError(
                f"profiler_actual_geometry_mismatch:{cid}:{field}:expected={exp_f}:actual={got_f}"
            )


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


def extract_observed_shape(
    summary: Mapping[str, Any], first: Mapping[str, Any], comp: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return the richest non-placeholder builder-reported actual/requested shape."""
    ident = first.get("identity") if isinstance(first.get("identity"), Mapping) else {}
    key_map = first.get("key") if isinstance(first.get("key"), Mapping) else {}
    comp_map = comp if isinstance(comp, Mapping) else {}
    for cand in (
        comp_map.get("actual_shape"), summary.get("actual_shape"), first.get("actual_shape"),
        comp_map.get("executed_shape"), summary.get("executed_shape"), first.get("executed_shape"),
        summary.get("requested_shape"), first.get("requested_shape"),
        ident.get("shape"), key_map.get("shape"),
    ):
        if isinstance(cand, Mapping) and cand and dict(cand) != {"declared": True}:
            return dict(cand)
    return {}


def extract_abi_attempted_shape(rec: Mapping[str, Any], planned_pt: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
    """Normalize writer-reported attempted geometry for an ABI rejection; return None if absent."""
    hint = rec.get("unmodeled_hint") if isinstance(rec.get("unmodeled_hint"), Mapping) else {}
    raw: Mapping[str, Any] | None = None
    for cand in (
        rec.get("executed_shape"), rec.get("attempted_shape"), rec.get("actual_shape"),
        rec.get("requested_shape"), rec.get("shape"), hint.get("executed_shape"), hint.get("shape"),
    ):
        if isinstance(cand, Mapping) and cand and dict(cand) != {"declared": True}:
            raw = cand
            break
    if raw is None:
        return None
    out = dict(raw)
    axis = str((planned_pt or {}).get("axis") or "").strip()
    if "context_len" not in out and _int_from(out, "seq_len") is not None:
        out["context_len"] = _int_from(out, "seq_len")
    if "topk" not in out and _int_from(out, "top_k", "sparse_count") is not None:
        out["topk"] = _int_from(out, "top_k", "sparse_count")
    if "query_tokens" not in out and _int_from(out, "q_len") is not None:
        out["query_tokens"] = _int_from(out, "q_len")
    if "batch_size" not in out and axis != "batch_x_kv_heads" and _int_from(out, "sequences") is not None:
        out["batch_size"] = _int_from(out, "sequences")
    return out


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
    "assign_plan_point_level",
    "build_profiler_subgraph_plan",
    "derive_writer_clock_operator",
    "extract_abi_attempted_shape",
    "extract_observed_shape",
    "extract_shape_dims",
    "parse_run_id_and_arm",
    "parse_topo_options",
    "quantile_ns",
    "resolve_point_binding_and_shape",
    "resolve_point_measured_binding",
    "select_mode_candidates",
    "verify_actual_geometry_matches_plan",
]
