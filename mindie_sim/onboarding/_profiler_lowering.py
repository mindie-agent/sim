"""1-layer white-box spine lowering to executable ``mindie-profiler`` subgraph nodes.

[Role] Lower 1-layer ``mindie-sim`` model-step DAGs via ``default_spine()`` and
  translate every lowered ``OpSpec`` (MLA attention, GEMM / grouped GEMM,
  RMSNorm / SwiGLU / Rotary, KV compress epilog, DCP ``all_gather``,
  ``dcp_lse_attention_merge``, TP/CP ``all_reduce``) into exact 1:1
  ``mindie-profiler`` subgraph nodes and ``measured_binding`` entries.
[Invariants]
  * L3 ``model_step`` cases are lowered from the exact same 1-layer
    ``default_spine().lower(...)`` graph that ``WhiteboxSpineRunner`` replays.
  * Preserves exact collective ``dtype``, payload ``nbytes``, ``group_size``,
    ``attention_merge`` ``partial_count``, and GEMM ``(m, n, k, matrix_groups)``.
  * Bounded ``lru_cache`` avoids repeated lowering across plan points.
[Timeouts & Locks] Pure in-memory lowering helpers; zero I/O, locks, or
  subprocesses.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
import hashlib
import re
from typing import Any

from mindie_sim.schema.common import ModelingError

from .contracts import canonical_json_bytes

_MLA_LATENT_DIM = 512
_MLA_ROPE_DIM = 64
_MLA_BLOCK_SIZE = 128
_MLA_HIDDEN_WIDTH = 2048
_MLA_BF16_BYTES = 2
_MIN_COLLECTIVE_BYTES = 4
_DEFAULT_COLLECTIVE_BYTES = 4096
_RING_ALLREDUCE_FACTOR = 2
_MIN_DCP_PARTIALS = 2
_MIN_GEMM_SHAPE_RANK = 3
_MIN_2D_SHAPE_RANK = 2
_LOWER_CACHE_MAXSIZE = 64
_ARCH35_PRESET_MARKER = "35"
_DEFAULT_SIM_MODEL = "deepseek-v2-lite"
_TOPO_TOKEN_RE = re.compile(r"(tp|pp|dp|ep|pcp|dcp)(\d+)")
_DTYPE_ELEMENT_BYTES: Mapping[str, int] = {
    "fp32": 4, "float32": 4, "int32": 4,
    "bf16": 2, "fp16": 2, "float16": 2,
    "int8": 1, "uint8": 1, "fp8_e4m3": 1, "mxfp8": 1,
}


def _dtype_element_bytes(dtype: str) -> int:
    return int(_DTYPE_ELEMENT_BYTES.get(str(dtype or "bf16").strip().lower(), _MLA_BF16_BYTES))


def parse_topo_options(topo: str) -> dict[str, int]:
    """Parse canonical topology tokens into ``compose_options``."""
    return {k: int(v) for k, v in _TOPO_TOKEN_RE.findall(str(topo or "tp1").lower())} or {"tp": 1}


def _preset_from_profile(prof: Mapping[str, Any] | None) -> str:
    if not isinstance(prof, Mapping):
        return ""
    dev = prof.get("device")
    if isinstance(dev, Mapping):
        ca = dev.get("compute_arch")
        if isinstance(ca, Mapping) and ca.get("preset"):
            return str(ca["preset"]).strip()
    return ""


def _decode_mla_recipe_for_sku(
    sku: str,
    *,
    plan_doc: Mapping[str, Any] | None = None,
    profile_override: Mapping[str, Any] | None = None,
) -> str:
    from ._sku_calibration import resolve_sku_profile

    prof = resolve_sku_profile(sku, profile_override)
    if prof is None:
        raise ModelingError(f"unknown_sku_profile_for_lowering:{sku}")
    prof_preset = _preset_from_profile(prof)
    if not prof_preset:
        raise ModelingError(f"sku_profile_missing_arch_preset:{sku}")
    if isinstance(plan_doc, Mapping):
        plan_preset = str(plan_doc.get("arch_preset") or "").strip()
        if plan_preset and plan_preset != prof_preset:
            raise ModelingError(
                f"plan_arch_preset_profile_conflict:{sku}:plan={plan_preset}:profile={prof_preset}"
            )
    if _ARCH35_PRESET_MARKER in prof_preset:
        return "ascend_paged_mla_decode_arch35"
    return "ascend_paged_mla_decode_arch22"


def _page_count(kv_len: int, block_size: int = _MLA_BLOCK_SIZE) -> int:
    return max(1, (int(kv_len) + int(block_size) - 1) // int(block_size))


@lru_cache(maxsize=_LOWER_CACHE_MAXSIZE)
def _lower_one_layer_step_ops_cached(
    sku: str, prof_sha: str, topo: str, phase: str, q_tok: int, ctx_len: int, seqs: int
) -> tuple[dict[str, Any], ...]:
    """Lower a 1-layer ``deepseek-v2-lite`` step via ``default_spine()`` so L3 profiler == L3 replay."""
    del prof_sha
    from mindie_sim.compose import compose_case as compose_case_dict
    from mindie_sim.prediction.results import StepRequest
    from mindie_sim.prediction.spine import default_spine
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.stack_registry import LoweringRequest

    opts = parse_topo_options(topo)
    raw = compose_case_dict(model=_DEFAULT_SIM_MODEL, hardware=sku, **opts)
    model_d = dict(raw["model"])
    model_d["layers"] = 1
    model_d["body_layers"] = 1
    model_d["layer_plans"] = list(model_d["layer_plans"])[:1]
    case = parse_case({**raw, "model": model_d})
    prep = default_spine().lower(
        case,
        LoweringRequest(
            step=StepRequest(
                phase=phase,  # type: ignore[arg-type]
                tokens=int(q_tok),
                context=int(ctx_len),
                sequences=int(seqs),
                scope="body",
            ),
            tier="mainline",
        ),
    )
    out: list[dict[str, Any]] = []
    for op in prep.graph.executable[0].ops:
        meta = dict(op.metadata)
        nabi = meta.get("native_attention_abi") if isinstance(meta.get("native_attention_abi"), Mapping) else {}
        pgeo = meta.get("paged_mla_geometry") if isinstance(meta.get("paged_mla_geometry"), Mapping) else {}
        out.append({
            "op_id": str(op.op_id),
            "family": str(op.family),
            "recipe": str(op.recipe),
            "dtype": str(op.dtype or "bf16"),
            "quant_mode": str(meta.get("quant_mode") or meta.get("quantization") or "none").strip().lower(),
            "local_shape": tuple(int(x) if isinstance(x, (int, float)) else x for x in (op.local_shape or ())),
            "deps": tuple(str(d) for d in (op.deps or ())),
            "link_bytes": int(round(float(op.link_bytes or 0.0))),
            "matrix_groups": int(meta.get("matrix_groups") or 1),
            "group_axis": str(meta.get("group_axis") or ""),
            "query_layout": str(nabi.get("query_layout") or ""),
            "cache_layout": str(nabi.get("cache_layout") or ""),
            "output_layout": str(nabi.get("output_layout") or ""),
            "kv_len": int(pgeo["kv_lengths"][0]) if pgeo.get("kv_lengths") else 0,
            "page_count": int(pgeo["page_counts"][0]) if pgeo.get("page_counts") else 0,
        })
    return tuple(out)


def _lower_one_layer_step_ops(
    sku: str,
    topo: str,
    phase: str,
    q_tok: int,
    ctx_len: int,
    seqs: int,
    *,
    profile_override: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Lower a 1-layer step against the exact SKU profile (never silently substituting B4)."""
    from ._sku_calibration import resolve_sku_profile, temporary_scaffold_profile

    with temporary_scaffold_profile(sku, profile_override):
        prof = resolve_sku_profile(sku, profile_override)
        if not isinstance(prof, Mapping):
            raise ModelingError(f"unknown_sku_profile_for_lowering:{sku}")
        prof_sha = hashlib.sha256(canonical_json_bytes(prof)).hexdigest()
        return _lower_one_layer_step_ops_cached(sku, prof_sha, topo, phase, q_tok, ctx_len, seqs)


def _lowered_attn_or_gemm_node(
    nid: str,
    op_rec: Mapping[str, Any],
    *,
    attn_shape: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    op_id = str(op_rec["op_id"])
    fam = str(op_rec["family"])
    rec = str(op_rec["recipe"])
    loc = tuple(op_rec["local_shape"])
    mg = max(1, int(op_rec["matrix_groups"]))

    if fam == "attention":
        sh = dict(attn_shape)
        if op_rec["kv_len"] > 0:
            sh["seq_len"] = int(op_rec["kv_len"])
            sh["kv_len"] = int(op_rec["kv_len"])
            sh["page_count"] = int(op_rec["page_count"]) or _page_count(int(op_rec["kv_len"]))
        elif len(loc) >= _MIN_2D_SHAPE_RANK and int(loc[1]) > 0:
            sh["seq_len"] = int(loc[1])
            sh["kv_len"] = int(loc[1])
            sh["page_count"] = _page_count(int(loc[1]))
        if op_rec["query_layout"]:
            sh["query_layout"] = op_rec["query_layout"]
            sh["input_layout"] = op_rec["query_layout"]
            sh["cache_layout"] = op_rec["cache_layout"] or "PA_BSND"
            sh["output_layout"] = op_rec["output_layout"] or "NBSD"
        node: dict[str, Any] = {
            "node_id": nid, "op_id": op_id, "kind": "operator",
            "op": "absorbed_mla", "builder": "absorbed_mla", "family": "attention",
            "recipe": rec, "implementation": "aclnnFusedInferAttentionScoreV4",
            "dtype": "bf16", "paged": True, "layout": sh.get("query_layout", "TND"),
            "shape": sh,
        }
        binding = {
            "op_id": op_id, "family": "attention", "recipe": rec, "builder": "absorbed_mla",
            "paged": True, "heads": sh["heads"], "kv_heads": 1, "latent": _MLA_LATENT_DIM,
            "rope": _MLA_ROPE_DIM, "kv_len": sh["seq_len"], "page_count": sh["page_count"],
            "query_layout": sh.get("query_layout", "TND"),
        }
        return node, binding

    if fam in {"gemm", "gate_up_projection", "down_projection"} and len(loc) >= _MIN_GEMM_SHAPE_RANK:
        m_dim, n_dim, k_dim = int(loc[0]), int(loc[1]), int(loc[2])
        op_dtype = str(op_rec.get("dtype") or "bf16").strip().lower()
        op_quant = str(op_rec.get("quant_mode") or "none").strip().lower()
        if mg > 1:
            per_grp = max(1, m_dim // mg)
            node = {
                "node_id": nid, "op_id": op_id, "kind": "operator",
                "op": "grouped_matmul", "builder": "grouped_matmul", "family": "grouped_gemm",
                "recipe": "grouped_matmul", "implementation": "aclnnGroupedMatmulV5",
                "dtype": op_dtype, "group_list": [per_grp] * mg, "group_list_type": 1,
                "shape": {"m": m_dim, "n": n_dim, "k": k_dim, "e": mg, "matrix_groups": mg, "dtype": op_dtype},
            }
            return node, {
                "op_id": op_id, "family": fam, "recipe": rec, "builder": "grouped_matmul",
                "dtype": op_dtype, "quant_mode": op_quant,
                "m": m_dim, "n": n_dim, "k": k_dim, "matrix_groups": mg,
            }
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator",
            "op": "matmul", "builder": "matmul", "family": fam, "recipe": rec,
            "implementation": "aclnnMatmul", "dtype": op_dtype,
            "shape": {"m": m_dim, "n": n_dim, "k": k_dim, "matrix_groups": 1, "dtype": op_dtype},
        }
        return node, {
            "op_id": op_id, "family": fam, "recipe": rec, "builder": "matmul",
            "dtype": op_dtype, "quant_mode": op_quant,
            "m": m_dim, "n": n_dim, "k": k_dim, "matrix_groups": 1,
        }
    return None


def _lowered_op_to_profiler_node(
    pid: str,
    op_rec: Mapping[str, Any],
    *,
    attn_shape: Mapping[str, Any],
    total_tokens: int,
    local_heads: int,
    topo_opts: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert one lowered ``OpSpec`` summary into a profiler node + node binding."""
    op_id = str(op_rec["op_id"])
    fam = str(op_rec["family"])
    rec = str(op_rec["recipe"])
    loc = tuple(op_rec["local_shape"])
    tp = max(1, int(topo_opts.get("tp", 1)))
    dcp = max(1, int(topo_opts.get("dcp", 1)))
    nid = f"{pid}:{op_id}"

    attn_or_gemm = _lowered_attn_or_gemm_node(nid, op_rec, attn_shape=attn_shape)
    if attn_or_gemm is not None:
        return attn_or_gemm

    if fam == "rmsnorm":
        toks = int(loc[0]) if len(loc) >= _MIN_2D_SHAPE_RANK else total_tokens
        width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "rms_norm", "builder": "rms_norm",
            "family": "rmsnorm", "recipe": rec, "implementation": "aclnnRmsNorm", "dtype": "bf16",
            "shape": {"tokens": toks, "width": width},
        }
        return node, {"op_id": op_id, "family": "rmsnorm", "recipe": rec, "builder": "rms_norm", "width": width}

    if fam == "add_rmsnorm":
        toks = int(loc[0]) if len(loc) >= _MIN_2D_SHAPE_RANK else total_tokens
        width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "add_rms_norm", "builder": "add_rms_norm",
            "family": "add_rmsnorm", "recipe": rec, "implementation": "aclnnAddRmsNorm", "dtype": "bf16",
            "shape": {"tokens": toks, "width": width},
        }
        return node, {"op_id": op_id, "family": "add_rmsnorm", "recipe": rec, "builder": "add_rms_norm", "width": width}

    if fam == "swiglu":
        toks = int(loc[0]) if len(loc) >= _MIN_2D_SHAPE_RANK else total_tokens
        width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "swiglu", "builder": "swiglu",
            "family": "swiglu", "recipe": rec, "implementation": "aclnnSwiGlu", "dtype": "bf16",
            "shape": {"tokens": toks, "width": width},
        }
        return node, {"op_id": op_id, "family": "swiglu", "recipe": rec, "builder": "swiglu", "width": width}

    if fam == "elementwise" and rec == "rotary":
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "rotary_embedding", "builder": "rotary_embedding",
            "family": "elementwise", "recipe": "rotary", "implementation": "aclnnApplyRotaryPosEmbV2", "dtype": "bf16",
            "shape": {"tokens": total_tokens, "heads": local_heads, "head_dim": _MLA_ROPE_DIM},
        }
        return node, {"op_id": op_id, "family": "elementwise", "recipe": "rotary", "builder": "rotary_embedding"}

    if fam == "kv_compress_epilog":
        toks = int(loc[0]) if len(loc) >= _MIN_2D_SHAPE_RANK else total_tokens
        op_dt = str(op_rec["dtype"] or "bf16").strip().lower()
        elem_b = _dtype_element_bytes(op_dt)
        byte_width = int(loc[-1]) if loc else (_MLA_LATENT_DIM + _MLA_ROPE_DIM) * _MLA_BF16_BYTES
        elem_width = max(1, byte_width // elem_b)
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "kv_compress_epilog", "builder": "kv_compress_epilog",
            "family": "kv_compress_epilog", "recipe": rec, "implementation": "aclnnKvCompressEpilog", "dtype": op_dt,
            "shape": {"tokens": toks, "width": elem_width, "byte_width": byte_width},
        }
        return node, {"op_id": op_id, "family": fam, "recipe": rec, "builder": "kv_compress_epilog", "dtype": op_dt, "width": elem_width, "byte_width": byte_width}

    if fam == "dcp_allgather":
        raw_link = int(op_rec["link_bytes"])
        comm_dt = str(op_rec["dtype"] or "bf16").strip().lower()
        send_nbytes = max(
            _MIN_COLLECTIVE_BYTES,
            (raw_link // max(1, dcp - 1)) if raw_link > 0 else _DEFAULT_COLLECTIVE_BYTES,
        )
        node = {
            "node_id": nid, "op_id": op_id, "kind": "collective", "op": "all_gather", "builder": "all_gather",
            "family": "dcp_allgather", "recipe": rec, "implementation": "hcclAllGather", "dtype": comm_dt,
            "group": "dcp", "group_size": dcp, "nbytes": send_nbytes,
            "shape": {
                "dtype": comm_dt, "nbytes": send_nbytes, "link_bytes": raw_link, "group_size": dcp,
            },
        }
        return node, {
            "op_id": op_id, "family": fam, "recipe": rec, "builder": "all_gather",
            "dtype": comm_dt, "nbytes": send_nbytes, "group_size": dcp,
        }

    if fam == "attention_merge":
        toks = int(loc[0]) if len(loc) >= _MIN_GEMM_SHAPE_RANK else total_tokens
        heads = int(loc[1]) if len(loc) >= _MIN_GEMM_SHAPE_RANK else local_heads
        hdim = int(loc[2]) if len(loc) >= _MIN_GEMM_SHAPE_RANK else _MLA_LATENT_DIM
        merge_dt = str(op_rec["dtype"] or "fp32").strip().lower()
        partials = max(_MIN_DCP_PARTIALS, dcp)
        node = {
            "node_id": nid, "op_id": op_id, "kind": "operator", "op": "dcp_lse_attention_merge",
            "builder": "dcp_lse_attention_merge", "family": "attention_merge", "recipe": rec,
            "implementation": "aclnnRingAttentionUpdate", "dtype": merge_dt,
            "shape": {"tokens": toks, "heads": heads, "kv_heads": 1, "head_dim": hdim, "partial_count": partials, "dtype": merge_dt},
        }
        return node, {
            "op_id": op_id, "family": fam, "recipe": rec, "builder": "dcp_lse_attention_merge",
            "dtype": merge_dt, "partial_count": partials,
        }

    if fam == "allreduce":
        grp_axis = op_rec["group_axis"] or "tp"
        grp_sz = dcp if grp_axis == "cp" else tp
        comm_dt = str(op_rec["dtype"] or "bf16").strip().lower()
        raw_link = int(op_rec["link_bytes"])
        if raw_link > 0 and grp_sz > 1:
            payload_nbytes = max(
                _MIN_COLLECTIVE_BYTES,
                (raw_link * grp_sz) // max(1, _RING_ALLREDUCE_FACTOR * (grp_sz - 1)),
            )
        else:
            payload_nbytes = max(_MIN_COLLECTIVE_BYTES, raw_link or _DEFAULT_COLLECTIVE_BYTES)
        node = {
            "node_id": nid, "op_id": op_id, "kind": "collective", "op": "all_reduce", "builder": "all_reduce",
            "family": "allreduce", "recipe": rec, "implementation": "hcclAllReduce", "dtype": comm_dt,
            "group": grp_axis, "group_size": grp_sz, "nbytes": payload_nbytes,
            "shape": {
                "dtype": comm_dt, "nbytes": payload_nbytes, "link_bytes": raw_link, "group_size": grp_sz,
            },
        }
        return node, {
            "op_id": op_id, "family": fam, "recipe": rec, "builder": "all_reduce",
            "dtype": comm_dt, "nbytes": payload_nbytes, "group_size": grp_sz,
        }

    width = int(loc[-1]) if loc else _MLA_HIDDEN_WIDTH
    node = {
        "node_id": nid, "op_id": op_id, "kind": "operator", "op": "vector_add", "builder": "vector_add",
        "family": "elementwise", "recipe": "residual_add", "implementation": "aclnnAdd", "dtype": "bf16",
        "shape": {"tokens": total_tokens, "width": width},
    }
    return node, {"op_id": op_id, "family": fam, "recipe": rec, "builder": "vector_add"}


__all__ = [
    "_decode_mla_recipe_for_sku",
    "_dtype_element_bytes",
    "_lower_one_layer_step_ops",
    "_lowered_attn_or_gemm_node",
    "_lowered_op_to_profiler_node",
    "_page_count",
    "parse_topo_options",
]
