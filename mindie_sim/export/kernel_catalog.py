"""Domain-standard Ascend submodule, macro-module, and kernel catalog classifier.

[Role] Map lowered ``mindie_sim`` operators into:
  (1) domain-standard Ascend submodules (zero ``Other``),
  (2) architectural macro-modules for Sheet 1 executive duration & share % tables,
  (3) multi-ecosystem Ascend kernel identities (``CANN aclnn/AscendC``,
      ``cannbot-dsl``, ``Triton-Ascend``, ``torch_npu Custom Op``, ``HCCL / MC2``),
      readiness statuses, source references, and rank-local tensor contracts,
      preferring ``export_op_structure`` as single source of truth when available.
[Invariants] Zero banned internal Python simulator recipe class names ever escape
  into ``AscendKernelDescriptor.ascend_kernel``. Every operator classifies into a
  valid non-``Other`` submodule and exactly one macro-module.
[Timeouts & Locks] Pure deterministic in-memory lookup; no I/O, locks, or device
  contexts.
[Side-Effect Policy] Read-only over ``OpSpec`` and ``Case`` metadata.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from mindie_sim.schema.case import Case
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure_contracts import predicted_contract
from mindie_sim.structure_implementations import native_implementation

from .models import (
    ATTN_4D_RANK,
    ATTN_5D_RANK,
    BACKEND_CANN,
    BACKEND_CANNBOT_DSL,
    BACKEND_HCCL_MC2,
    BACKEND_TORCH_NPU,
    BACKEND_TRITON,
    BANNED_IMPLEMENTATION_NAMES,
    COLLECTIVE_FAMILIES,
    DEFAULT_SPARSE_INDEX_TOPK,
    GEMM_SHAPE_RANK,
    KERNEL_STATUS_EXISTING,
    KERNEL_STATUS_EXTEND_TILING,
    KERNEL_STATUS_FRAMEWORK,
    KERNEL_STATUS_NEW_OP,
    KERNEL_STATUS_NEW_OR_EXTEND,
    MACRO_MOD_COLLECTIVES,
    MACRO_MOD_DENSE_NORM,
    MACRO_MOD_DSA_LI,
    MACRO_MOD_ENGRAM,
    MACRO_MOD_GATED_O_PROJ,
    MACRO_MOD_KDA_GDN,
    MACRO_MOD_LATENT_ATTN,
    MACRO_MOD_MLA_PROLOG,
    MACRO_MOD_MOE_ROUTED,
    MACRO_MOD_MOE_ROUTER,
    MACRO_MOD_MOE_SHARED,
    MACRO_MOD_SPEC_HEAD,
    MACRO_MOD_VISION,
    MATRIX_2D_RANK,
    STRUCTURE_BACKEND_MAP,
    STRUCTURE_READINESS_MAP,
    SUB_ATTN_RES,
    SUB_DENSE_FFN,
    SUB_DSA_LI,
    SUB_EMBED_TAIL,
    SUB_ENGRAM_PLE,
    SUB_GATED_MLA_O,
    SUB_GDN,
    SUB_KDA,
    SUB_LATENT_ATTN,
    SUB_MLA_PROLOG,
    SUB_MOE_ROUTED,
    SUB_MOE_ROUTER,
    SUB_MOE_SHARED,
    SUB_STD_ATTN,
    SUB_VISION,
    AscendKernelDescriptor,
    OperatorExportRow,
)

_LAYER_PATTERN = re.compile(r"(?:^|\.)l(\d+)(?:\.|$)")
_CONTEXT_KV_PATTERN = re.compile(r"context_kv(\d+)(?:\.|$)")
_MOE_EP_COMM_FAMILIES = frozenset({
    "dispatch", "combine", "mc2", "mc2_protocol", "mc2_transport",
})
_DTYPE_UPPER_MAP: dict[str, str] = {
    "bf16": "BF16", "bfloat16": "BF16", "fp16": "FP16", "float16": "FP16",
    "fp32": "FP32", "float32": "FP32", "fp8": "FP8", "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8", "mxfp4": "MXFP4", "mx_fp4": "MXFP4", "fp4": "MXFP4",
    "fp4_e2m1": "MXFP4", "mxfp8": "MXFP8", "mx_fp8": "MXFP8",
    "int8": "INT8", "int4": "INT4", "int32": "INT32", "int64": "INT64",
}


def extract_layer_index(op_id: str, op_layer: int | None = None) -> int | None:
    """Resolve integer decoder layer index from ``OpSpec.layer`` or ``op_id``."""
    if op_layer is not None and op_layer >= 0:
        return int(op_layer)
    match = _LAYER_PATTERN.search(op_id)
    if match is not None:
        return int(match.group(1))
    ctx_match = _CONTEXT_KV_PATTERN.search(op_id)
    return int(ctx_match.group(1)) if ctx_match is not None else None


def strip_op_prefix(op_id: str) -> str:
    """Return layer-relative suffix (e.g. ``s0.l3.index_q`` -> ``index_q``)."""
    text = str(op_id or "")
    text = re.sub(r"^draft\d+\.", "", text)
    text = re.sub(r"^context_kv\d+\.", "", text)
    text = re.sub(r"^s\d+\.l\d+\.", "", text)
    text = re.sub(r"^l\d+\.", "", text)
    return text


def classify_submodule(
    op_id: str,
    family: str,
    recipe: str = "",
    *,
    layer_attention: str = "",
    layer_ffn: str = "",
) -> str:
    """Classify an operator into a domain-standard Ascend stage (ZERO ``Other``!)."""
    oid = str(op_id or "").lower()
    suffix = strip_op_prefix(op_id).lower()
    fam = str(family or "").strip().lower()
    rec = str(recipe or "").strip().lower()
    attn = str(layer_attention or "").strip().lower()
    ffn = str(layer_ffn or "").strip().lower()

    if _is_embed_or_tail_op(oid, suffix, fam, rec):
        return SUB_EMBED_TAIL
    if any(tok in suffix or tok in rec for tok in ("vision", "aligner", "projector")):
        return SUB_VISION
    if any(tok in suffix or tok in fam or tok in rec for tok in ("engram", "ple_")):
        return SUB_ENGRAM_PLE
    if _is_shared_expert_op(suffix, rec):
        return SUB_MOE_SHARED
    if _is_attn_res_or_norm_op(suffix, fam, rec):
        return SUB_ATTN_RES
    if _is_dsa_indexer_op(suffix, fam, rec):
        return SUB_DSA_LI
    if _is_mla_prolog_op(suffix, fam, rec, attn):
        return SUB_MLA_PROLOG
    if _is_latent_attn_op(suffix, fam, rec, attn):
        return SUB_LATENT_ATTN
    if _is_gated_mla_output_op(suffix, fam, rec, attn):
        return SUB_GATED_MLA_O
    if _is_linear_attn_op(suffix, fam, rec, attn):
        return SUB_GDN if _is_gdn_op(suffix, fam, rec, attn) else SUB_KDA
    if "megamoe" in rec or "dispatch_ffn_combine" in rec:
        return SUB_MOE_ROUTED
    if _is_moe_router_dispatch_op(suffix, fam, rec):
        return SUB_MOE_ROUTER
    if _is_moe_routed_op(suffix, fam, rec):
        return SUB_MOE_ROUTED
    if _is_dense_ffn_op(suffix, fam, rec, ffn):
        return SUB_DENSE_FFN
    if _is_std_attn_op(suffix, fam, rec, attn):
        return SUB_STD_ATTN
    if "ffn" in suffix:
        return SUB_MOE_ROUTED if ffn == "moe" else SUB_DENSE_FFN
    if attn == "kda":
        return SUB_KDA
    if attn == "gdn":
        return SUB_GDN
    if attn in {"mla", "mla_dsa", "dsa"}:
        return SUB_LATENT_ATTN
    return SUB_ATTN_RES


def classify_macro_module(op: OperatorExportRow) -> str:
    """Assign an operator to its single executive architectural macro-module."""
    if op.model_scope == "draft":
        return MACRO_MOD_SPEC_HEAD
    sub = op.submodule
    if sub == SUB_ENGRAM_PLE:
        return MACRO_MOD_ENGRAM
    if sub == SUB_VISION:
        return MACRO_MOD_VISION
    if sub == SUB_MOE_SHARED:
        return MACRO_MOD_MOE_SHARED

    suffix = strip_op_prefix(op.op_id).lower()
    fam = str(op.family or "").strip().lower()
    rec = str(op.recipe or "").strip().lower()
    kern_l = str(op.ascend_kernel or "").strip().lower()
    if "megamoe" in rec or "megamoe" in kern_l or "dispatch_ffn_combine" in kern_l or "dispatch_ffn_combine" in rec:
        return MACRO_MOD_MOE_ROUTED
    is_moe_ep_comm = sub in {SUB_MOE_ROUTER, SUB_MOE_ROUTED} and (
        suffix in {"dispatch", "unpack", "combine"}
        or "dispatch" in rec
        or "combine" in rec
        or fam in _MOE_EP_COMM_FAMILIES
    )
    if sub == SUB_MOE_ROUTER or is_moe_ep_comm:
        return MACRO_MOD_MOE_ROUTER
    if op.task_type == "HCCL" or fam in COLLECTIVE_FAMILIES:
        return MACRO_MOD_COLLECTIVES
    if sub == SUB_EMBED_TAIL:
        return MACRO_MOD_SPEC_HEAD
    if sub in {SUB_KDA, SUB_GDN}:
        return MACRO_MOD_KDA_GDN
    if sub == SUB_MLA_PROLOG:
        return MACRO_MOD_MLA_PROLOG
    if sub == SUB_DSA_LI:
        return MACRO_MOD_DSA_LI
    if sub in {SUB_LATENT_ATTN, SUB_STD_ATTN}:
        return MACRO_MOD_LATENT_ATTN
    if sub == SUB_GATED_MLA_O:
        return MACRO_MOD_GATED_O_PROJ
    if sub == SUB_MOE_ROUTED:
        return MACRO_MOD_MOE_ROUTED
    return MACRO_MOD_DENSE_NORM


def _is_embed_or_tail_op(oid: str, suffix: str, fam: str, rec: str) -> bool:
    if fam in {"embedding", "logits", "acceptance"}:
        return True
    if oid.startswith("final.") or suffix.startswith(("final_", "logits", "embedding")):
        return True
    if suffix in {
        "sp_shard", "sp_output_gather", "accept", "features", "features_gather",
        "context_norm", "query_embedding", "markov_lookup", "markov", "markov_bias",
    }:
        return True
    return any(tok in rec for tok in (
        "draft_features", "draft_context_norm", "draft_query_embedding",
        "markov", "vocab_logits", "speculative_acceptance", "hc_lowrank_head",
    ))


def _is_shared_expert_op(suffix: str, rec: str) -> bool:
    return suffix.startswith("shared.") or suffix.startswith("shared_gate") or "shared_gate" in rec


def _is_attn_res_or_norm_op(suffix: str, fam: str, rec: str) -> bool:
    if fam in {"attn_res_prefix", "mhc"} or "attn_res" in suffix or ".hc_" in f".{suffix}":
        return True
    if suffix in {
        "norm1", "norm2", "input_layernorm", "post_attention_layernorm",
        "attention.residual", "ffn.residual", "residual",
    }:
        return True
    return rec in {
        "rmsnorm", "add_rmsnorm", "residual_add", "grouped_gemma_rmsnorm",
        "hc_silu", "hc_values_fp32", "hc_projection", "hc_lowrank_down_inject",
        "hc_lowrank_up",
    }


def _is_dsa_indexer_op(suffix: str, fam: str, rec: str) -> bool:
    if fam in {"indexer", "index_projection", "key_pool", "indexer_quant_cache"}:
        return True
    if suffix.startswith(("index_", "candidate_", "cann_keypool", "cann_key_pool", "cann_pool_key")):
        return True
    return any(tok in rec for tok in (
        "lightning_indexer", "index_scores", "index_topk",
        "candidate_", "keypool", "key_pool", "csa_summary",
    ))


def _is_mla_prolog_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if "mla_prolog" in suffix or "mla_prolog" in rec:
        return True
    if suffix in {
        "q_a", "q_a.quantize", "kv_a", "kv_a.quantize", "qkv_a", "qkv_a.quantize",
        "q_norm", "q_b", "q_b.quantize", "kv_norm", "q_rotary", "kv_rotary",
        "kv_store", "kv_scatter", "q_absorb",
    }:
        return True
    if suffix == "rotary" and attn in {"mla", "mla_dsa", "dsa", ""}:
        return True
    return suffix == "attention.sp_gather" and attn in {"mla", "mla_dsa", "dsa"}


def _is_latent_attn_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if suffix in {
        "dcp_query_pack", "dcp_query_allgather", "latent_attention_batch",
        "mqsmla", "dcp_output_pack", "dcp_output_alltoall", "dcp_lse_merge",
        "v_expand", "output_inverse_rotary",
    }:
        return True
    return fam in {"attention", "attention_merge"} and attn in {"mla", "mla_dsa", "dsa"}


def _is_gated_mla_output_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if suffix in {
        "gated_output", "grouped_output_a", "grouped_output_a.quantize",
        "grouped_output_b", "grouped_output_b.quantize",
    }:
        return True
    if attn in {"mla", "mla_dsa", "dsa"} and (
        suffix in {
            "output_gate", "output", "output.quantize", "output_reduce",
            "output_reduce.reduce_scatter", "output_reduce.reduce_scatter.pad",
            "tp_attn", "tp_attn.reduce_scatter",
        }
        or suffix.startswith("output_reduce.")
    ):
        return True
    return False


def _is_gdn_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if attn == "gdn":
        return True
    return any(tok in f"{suffix} {fam} {rec}" for tok in (
        "gdn", "linear_qkvz", "linear_beta_alpha", "decode_recurrent_scan",
        "alpha_contiguous", "beta_contiguous", "z_norm_materialize",
    ))


def _is_linear_attn_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if attn in {"kda", "gdn"} and suffix == "attention.sp_gather":
        return True
    if fam in {
        "recurrent_scan", "recurrent_vector", "gdn_gating", "gdn_prepare",
        "gdn_state", "gdn_output", "short_convolution", "causal_conv1d_fn",
    }:
        return True
    return (
        suffix.startswith((
            "linear_", "recurrent_", "chunk_kda", "recover_ssm", "short_conv",
            "decode_q_l2norm", "decode_k_l2norm", "decode_recurrent", "delta_gates",
            "qk_l2norm", "core_output_", "beta_contiguous", "alpha_contiguous",
            "convolution_input_", "q_flat", "k_flat", "v_flat", "qkv_concatenate",
            "z_norm_materialize",
        ))
        or any(tok in rec for tok in ("kda", "gdn", "recover_ssm", "short_conv", "causal_conv"))
    )


def _is_moe_router_dispatch_op(suffix: str, fam: str, rec: str) -> bool:
    if fam in {"router", "moe_init_routing"}:
        return True
    if suffix in {
        "router", "moe_gating_topk", "latent_down", "dispatch",
        "dispatch.received_quantize", "route_pack",
    }:
        return True
    return rec in {"router", "moe_gating_topk", "route_pack"} or suffix.startswith("dispatch.")


def _is_moe_routed_op(suffix: str, fam: str, rec: str) -> bool:
    if fam == "grouped_gemm":
        return True
    if suffix in {
        "expert_up", "expert_act", "expert_down", "expert_down.quantize",
        "unpack", "combine", "latent_norm", "latent_up", "join", "route_unpack",
    }:
        return True
    return any(tok in rec for tok in (
        "grouped_gate_up", "grouped_down", "grouped_situ",
        "grouped_swiglu", "megamoe", "expert_add", "route_unpack",
    ))


def _is_dense_ffn_op(suffix: str, fam: str, rec: str, ffn: str) -> bool:
    if fam in {"gate_up_projection", "down_projection", "swiglu", "situ"}:
        return True
    if suffix in {
        "ffn.sp_gather", "gate_up", "gate_up.quantize", "swiglu", "situ",
        "down", "down.quantize", "tp", "tp.reduce_scatter",
    }:
        return True
    return rec in {"dense_gate_up", "dense_down"} or (ffn == "dense" and "ffn" in suffix)


def _is_std_attn_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if fam in {"qkv_projection", "output_projection", "attention", "rotary", "cache_store"}:
        return True
    return suffix in {
        "attention.sp_gather", "qkv", "qkv.quantize", "qk_norm", "qkv.value", "qkv.gate",
        "rope", "rotary", "cache", "attention", "output_gate", "output", "output.quantize",
        "tp_attn", "tp_attn.reduce_scatter", "output_reduce", "output_reduce.reduce_scatter",
    }


def classify_stream_index(stream_name: str, family: str, op_id: str) -> int:
    """Resolve physical Ascend stream bucket (0=Compute, 1=MSO Aux, 2=HCCL/Comm)."""
    lower = str(stream_name or "").strip().lower()
    fam = str(family or "").strip().lower()
    if lower in {"shared_stream", "aux", "mso", "shared", "stream1", "1"}:
        return 1
    if lower in {"comm", "hccl", "csa", "stream2", "2", "cp_decode_comm_stream"} or fam in COLLECTIVE_FAMILIES:
        return 2
    if "shared_expert" in op_id.lower() and lower not in {"main", ""}:
        return 1
    return 0


def classify_task_type(
    family: str, flops: float, vector_ops: float, link_bytes: float,
) -> str:
    """Classify task core type (``HCCL``, ``MIX_AIC``, ``AI_CORE``, ``AI_VECTOR_CORE``)."""
    if str(family or "").strip().lower() in COLLECTIVE_FAMILIES or link_bytes > 0.0:
        return "HCCL"
    if flops > 0.0 and vector_ops > 0.0:
        return "MIX_AIC"
    if flops > 0.0:
        return "AI_CORE"
    return "AI_VECTOR_CORE"


def _format_dtype_upper(raw: str) -> str:
    tok = str(raw or "bf16").strip().lower()
    return _DTYPE_UPPER_MAP.get(tok, tok.upper() or "BF16")


def _format_tensor_contract(
    op_id: str,
    family: str,
    recipe: str,
    shape: Sequence[int],
    dtype: str,
    meta: Mapping[str, Any],
    case: Case,
    layer_plan: Any | None,
) -> str:
    """Format a readable rank-local tensor contract (never bare ambiguous element counts)."""
    suffix = strip_op_prefix(op_id).lower()
    fam = str(family or "").lower()
    act_dt = _format_dtype_upper(str(meta.get("activation_format") or meta.get("input_format") or dtype or "bf16"))
    w_dt = _format_dtype_upper(str(meta.get("weight_format") or dtype or "bf16"))
    out_dt = _format_dtype_upper(str(meta.get("output_format") or dtype or "bf16"))
    dims = tuple(int(d) for d in shape)
    tp = max(1, int(case.runtime.parallel.tp))

    if fam == "grouped_gemm" and len(dims) == GEMM_SHAPE_RANK:
        m_dim, n_dim, k_dim = dims
        e_local = int(meta.get("matrix_weight_bank_groups") or meta.get("local_experts") or 0)
        if e_local <= 0 and layer_plan is not None:
            ep = max(1, int(case.runtime.parallel.ep))
            e_local = max(1, int(getattr(layer_plan, "num_experts", 0) or 0) // ep)
        w_shape = f"[{e_local},{n_dim},{k_dim}]" if e_local > 0 else f"[{n_dim},{k_dim}]"
        return f"x:[{m_dim},{k_dim}] {act_dt} × W:{w_shape} {w_dt} -> y:[{m_dim},{n_dim}] {out_dt}"

    if len(dims) == GEMM_SHAPE_RANK and (
        fam in {
            "gemm", "projection", "gate_up_projection", "down_projection",
            "qkv_projection", "output_projection", "router", "logits", "index_projection",
        }
        or meta.get("program_kind") == "matmul"
    ):
        m_dim, n_dim, k_dim = dims
        if suffix in {"q_absorb", "v_expand"}:
            return f"x:[{m_dim},{k_dim}] {act_dt} × W_bmm:[{k_dim},{n_dim}] {w_dt} -> y:[{m_dim},{n_dim}] {out_dt}"
        return f"x:[{m_dim},{k_dim}] {act_dt} × W:[{n_dim},{k_dim}] {w_dt} -> y:[{m_dim},{n_dim}] {out_dt}"

    if fam == "attention":
        if len(dims) == ATTN_5D_RANK:
            q_tok, ctx, nh, dqk, dv = dims
            return f"Q:[{q_tok},{nh},{dqk}] {act_dt} × KV:[{ctx},1,{dqk}] -> O:[{q_tok},{nh},{dv}] {out_dt}"
        if len(dims) == ATTN_4D_RANK:
            q_tok, ctx, nh, hd = dims
            return f"Q:[{q_tok},{nh},{hd}] {act_dt} × KV:[{ctx},{hd}] -> O:[{q_tok},{nh},{hd}] {out_dt}"
        if len(dims) == GEMM_SHAPE_RANK:
            q_tok, nh, dv = dims
            topk = int(getattr(layer_plan, "index_topk", 0) or 0) if layer_plan is not None else 0
            kv_txt = f"SparseKV(topk={topk})" if topk > 0 else "PagedKV"
            return f"Q:[{q_tok},{nh},{dv}] {act_dt} × {kv_txt} -> O:[{q_tok},{nh},{dv}] {out_dt}"

    if fam in {"allgather", "reduce_scatter", "reducescatter", "allreduce", "alltoall", "mc2_protocol"}:
        if len(dims) >= MATRIX_2D_RANK:
            sh_txt = ",".join(str(d) for d in dims)
            return f"comm:[{sh_txt}] {act_dt} (TP={tp}, EP={case.runtime.parallel.ep})"
        if len(dims) == 1:
            rows = int(meta.get("graph_active_query_rows") or 0)
            if rows > 0 and dims[0] % rows == 0:
                return f"comm:[{rows},{dims[0] // rows}] {act_dt} (TP={tp})"
            return f"comm:[{dims[0]}] {act_dt} (TP={tp})"

    if len(dims) == 1 and dims[0] > 0:
        rows = int(meta.get("graph_active_query_rows") or 0)
        hidden = int(case.model.hidden_size or 0)
        if rows > 0 and dims[0] % rows == 0:
            return f"x:[{rows},{dims[0] // rows}] {act_dt} -> y:[{rows},{dims[0] // rows}] {out_dt}"
        if hidden > 0 and dims[0] % hidden == 0:
            return f"x:[{dims[0] // hidden},{hidden}] {act_dt} -> y:[{dims[0] // hidden},{hidden}] {out_dt}"
        return f"x:[{dims[0]}] {act_dt}"

    if dims:
        sh_txt = ",".join(str(d) for d in dims)
        return f"x:[{sh_txt}] {act_dt} -> y:[{sh_txt}] {out_dt}"
    return f"scalar {act_dt}"


def _format_source_ref(resolved_sources: Sequence[Any], default_ref: str) -> str:
    """Format canonical repository/path source reference."""
    parts: list[str] = []
    for src in resolved_sources:
        if isinstance(src, Mapping):
            repo = str(src.get("repository") or "").strip()
            rel = str(src.get("relative_path") or "").strip()
        else:
            repo = str(getattr(src, "repository", "") or "").strip()
            rel = str(getattr(src, "relative_path", "") or "").strip()
        if rel:
            parts.append(f"{repo}/{rel}" if repo else rel)
    if parts:
        return "; ".join(dict.fromkeys(parts))
    return default_ref


def _native_kernel_from_structure(
    op_id: str, spec: OpSpec | None, shape: Sequence[int], dtype: str,
) -> str:
    if spec is None:
        return ""
    meta = dict(spec.metadata) if spec.metadata else {}
    op_dict = {
        "op_id": op_id,
        "family": spec.family,
        "recipe": spec.recipe,
        "metadata": meta,
        "local_shape": list(spec.local_shape or shape),
        "padded_shape": list(spec.padded_shape or shape),
        "shape": list(shape),
        "dtype": dtype or spec.dtype or "bf16",
        "phase": spec.phase or "decode",
        "source_boundary": spec.source_boundary or "",
    }
    try:
        contract = predicted_contract(op_dict)
        nat = native_implementation(op_dict, contract)
    except Exception:
        return ""
    if isinstance(nat, Mapping):
        kinds = nat.get("native_kinds") or ()
        if kinds:
            return str(kinds[0])
    return ""


def _extract_native_candidate(struct_node: Mapping[str, Any] | None) -> str:
    """Read primary native kernel symbol from ``export_structure`` node/candidates."""
    if not isinstance(struct_node, Mapping):
        return ""
    nat_exec = struct_node.get("native_execution")
    if isinstance(nat_exec, Mapping):
        sel = nat_exec.get("selected")
        if isinstance(sel, Mapping) and sel.get("native"):
            cand = str(sel["native"]).strip()
            if cand and cand not in BANNED_IMPLEMENTATION_NAMES:
                return cand
    for raw_list in (
        struct_node.get("native_candidates"),
        (struct_node.get("implementation") or {}).get("native_kinds")
        if isinstance(struct_node.get("implementation"), Mapping) else (),
    ):
        if isinstance(raw_list, str):
            raw_list = (raw_list,)
        for item in raw_list or ():
            cand = str(item.get("native") if isinstance(item, Mapping) else item or "").strip()
            if cand and cand not in BANNED_IMPLEMENTATION_NAMES:
                return cand
    return ""


def resolve_ascend_kernel_identity(
    op_id: str,
    spec: OpSpec | None,
    family: str,
    recipe: str,
    shape: Sequence[int],
    dtype: str,
    case: Case,
    layer_plan: Any | None,
    resolved_sources: Sequence[Any],
    struct_node: Mapping[str, Any] | None = None,
    subgraph_info: Mapping[str, Any] | None = None,
) -> AscendKernelDescriptor:
    """Map an op using ``export_structure`` first, with clean semantic fallbacks."""
    suffix = strip_op_prefix(op_id).lower()
    fam = str(family or "").lower()
    rec = str(recipe or "").lower()
    meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
    track = str(dict(case.runtime.metadata).get("feature_track_report", {}).get("feature_track", "") or "")
    is_ideal = track == "expected_ideal"
    is_raw_eager = track == "raw_eager"
    is_quant_mm = str(meta.get("activation_format") or "").lower() in {"int8", "fp8", "fp8_e4m3", "fp8_e5m2"}
    is_mxfp4_w = str(meta.get("weight_format") or "").lower() in {"mxfp4", "mx_fp4", "fp4", "fp4_e2m1"}
    struct_nat = _extract_native_candidate(struct_node)
    reg_spec = None
    if not isinstance(struct_node, Mapping) or not struct_nat:
        from mindie_sim.structure_implementations import _lookup_registry_spec
        reg_spec = _lookup_registry_spec(fam, rec)
    if not struct_nat and reg_spec is not None and reg_spec.native_kinds:
        struct_nat = str(reg_spec.native_kinds[0])
    nat_kind = struct_nat or _native_kernel_from_structure(op_id, spec, shape, dtype)
    contract_txt = _format_tensor_contract(op_id, fam, rec, shape, dtype, meta, case, layer_plan)

    role, kernel, backend, status, def_src, needs_new, new_spec, canon_key = _classify_kernel_tuple(
        suffix=suffix, fam=fam, rec=rec, meta=meta, nat_kind=nat_kind,
        is_ideal=is_ideal, is_raw_eager=is_raw_eager,
        is_quant_mm=is_quant_mm, is_mxfp4_w=is_mxfp4_w,
        layer_plan=layer_plan,
    )
    if isinstance(struct_node, Mapping):
        preserve_qualified = bool(struct_nat) and kernel.startswith(f"{struct_nat} (")
        if (
            struct_nat
            and not needs_new
            and not preserve_qualified
            and not (is_raw_eager and (fam == "rmsnorm" or suffix == "linear_output_norm_gate"))
        ):
            kernel = "aclnnLayerNorm_LayerNormV3" if (fam == "layernorm" and struct_nat in {"aclnnLayerNorm", "LayerNormV3"}) else struct_nat
        raw_backend = str(struct_node.get("backend_stack") or "").strip().lower()
        if raw_backend in STRUCTURE_BACKEND_MAP:
            backend = STRUCTURE_BACKEND_MAP[raw_backend]
        raw_readiness = str(struct_node.get("readiness") or "").strip().lower()
        if raw_readiness in STRUCTURE_READINESS_MAP:
            mapped_status, mapped_new = STRUCTURE_READINESS_MAP[raw_readiness]
            if mapped_new or not needs_new:
                status = mapped_status if not new_spec else f"{mapped_status} {new_spec}"
                needs_new = mapped_new
        struct_src = str(struct_node.get("source_ref") or "").strip()
        if struct_src:
            def_src = struct_src
    elif reg_spec is not None:
        if reg_spec.backend_stack in STRUCTURE_BACKEND_MAP and not needs_new:
            backend = STRUCTURE_BACKEND_MAP[reg_spec.backend_stack]
        if reg_spec.source_ref and not def_src:
            def_src = reg_spec.source_ref
    if isinstance(subgraph_info, Mapping):
        f_status = str(subgraph_info.get("fusion_status") or "").strip().lower()
        if f_status == "fused_new_kernel_required" and not needs_new:
            reason = str(subgraph_info.get("reason") or "").strip()
            status = f"{KERNEL_STATUS_NEW_OP} {reason}".strip()
            needs_new = True
            new_spec = new_spec or reason
        elif f_status == "fused_framework_enable" and status == KERNEL_STATUS_EXISTING:
            status = KERNEL_STATUS_FRAMEWORK
    if kernel in BANNED_IMPLEMENTATION_NAMES or not kernel:
        kernel = "aclnnCustomOp"
    return AscendKernelDescriptor(
        logical_role=role,
        ascend_kernel=kernel,
        kernel_backend=backend,
        kernel_status=status,
        source_ref=_format_source_ref(resolved_sources, def_src),
        tensor_contract=contract_txt,
        needs_new_kernel=needs_new,
        new_kernel_spec=new_spec,
        canonical_op_key=canon_key,
    )


def _classify_kernel_tuple(
    *,
    suffix: str,
    fam: str,
    rec: str,
    meta: Mapping[str, Any],
    nat_kind: str,
    is_ideal: bool,
    is_raw_eager: bool = False,
    is_quant_mm: bool,
    is_mxfp4_w: bool,
    layer_plan: Any | None,
) -> tuple[str, str, str, str, str, bool, str, str]:
    mm_kernel = "QuantBatchMatmulV3" if is_quant_mm else "MatMulV2"
    mm_src = "vllm-ascend/vllm_ascend/ops/linear.py"

    if suffix.endswith(".pad") or suffix == "pad":
        return (
            "TP-Aligned Sequence Padding", "PadV3 / TensorMove", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/distributed/parallel_state.py",
            False, "", suffix,
        )
    if suffix == "sp_shard":
        return (
            "Sequence-Parallel Token Shard Split", "Slice / Split", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/distributed/parallel_state.py",
            False, "", "sp_shard",
        )
    if suffix == "markov_bias":
        return (
            "Draft Markov Logits Bias Add", "aclnnAdd", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py",
            False, "", "markov_bias",
        )
    if fam == "attn_res_prefix" or "attn_res" in suffix:
        return (
            "Block Attention Residual Prefix (Norm+Score+Softmax+BMM)",
            "vllm_ascend_attn_res_prefix",
            BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} AttnRes Cross-Layer Fused Prefix",
            "vllm-ascend/vllm_ascend/models/kimi_k3/attn_res.py",
            True,
            "Fused block residual norm+projection+softmax+weighted sum across block history",
            suffix,
        )
    if fam in {"all_gather_matmul", "matmul_reduce_scatter"}:
        is_ag = fam == "all_gather_matmul"
        return (
            f"Fused MC2 {'AllGather + MatMul' if is_ag else 'MatMul + ReduceScatter'} ({suffix})",
            "aclnnAllGatherMatmulV2" if is_ag else "aclnnMatmulReduceScatterV2",
            BACKEND_HCCL_MC2,
            KERNEL_STATUS_FRAMEWORK,
            "vllm-ascend/vllm_ascend/compilation/passes/mm_reduce_scatter_fusion_pass.py",
            False, "", fam,
        )
    if fam == "mhc" or ".hc_" in f".{suffix}":
        return _classify_mhc_tuple(suffix, fam, rec, mm_kernel, layer_plan, is_ideal=is_ideal)
    if any(tok in suffix or tok in fam or tok in rec for tok in ("engram", "ple_")):
        return _classify_engram_ple_tuple(suffix, fam, rec, mm_kernel)
    if fam in {"rmsnorm", "add_rmsnorm", "add_rmsnorm_quant", "rmsnorm_quant", "layernorm", "quantize"}:
        return _classify_norm_quant_tuple(suffix, fam, rec, meta, nat_kind, is_raw_eager=is_raw_eager)
    if "mla_prolog" in suffix or "mla_prolog" in rec:
        return (
            "Fused MLA Prolog (Q/KV-A + RMSNorm + Q-B + W_UK Absorb + KV Cache)",
            "aclnnMlaPrologV3",
            BACKEND_CANN,
            f"{KERNEL_STATUS_FRAMEWORK} (96->128 head pad & NoPE P^-1)",
            "vllm-ascend/vllm_ascend/attention/mla_v1.py",
            False, "", "mla_prolog_v3",
        )
    if suffix in {"q_absorb", "v_expand"}:
        role = "W_UK Absorbed Q BMM" if suffix == "q_absorb" else "W_UV Value Expand BMM"
        return (
            role, "BatchMatMulV2 (npu_transpose_batchmatmul)", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/attention/mla_v1.py",
            False, "", suffix,
        )
    if fam == "attention":
        return _classify_attention_tuple(suffix, rec, nat_kind, layer_plan)
    if fam == "attention_merge" or suffix == "dcp_lse_merge":
        return (
            "DCP Attention LSE Merge", "aclnnAttentionUpdate", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/attention/context_parallel/mla_cp.py",
            False, "", "dcp_lse_merge",
        )
    if _is_dsa_indexer_op(suffix, fam, rec):
        return _classify_indexer_tuple(suffix, fam, rec, mm_kernel, layer_plan)
    if _is_linear_attn_op(suffix, fam, rec, "kda" if "kda" in rec else ""):
        return _classify_linear_attn_tuple(
            suffix, fam, rec, meta, is_ideal, mm_kernel,
            is_raw_eager=is_raw_eager, layer_plan=layer_plan,
        )
    if _is_moe_router_dispatch_op(suffix, fam, rec) or _is_moe_routed_op(suffix, fam, rec) or _is_shared_expert_op(suffix, rec):
        return _classify_moe_tuple(suffix, fam, rec, meta, nat_kind, is_ideal, is_quant_mm, is_mxfp4_w, mm_kernel)
    if fam in COLLECTIVE_FAMILIES:
        return _classify_collective_tuple(suffix, fam, rec)
    return _classify_proj_and_misc_tuple(suffix, fam, nat_kind, mm_kernel, mm_src)


def _classify_proj_and_misc_tuple(
    suffix: str, fam: str, nat_kind: str, mm_kernel: str, mm_src: str,
) -> tuple[str, str, str, str, str, bool, str, str]:
    if fam in {"cache_store", "kv_compress_epilog", "kv_cache_write"} or suffix in {"kv_store", "kv_scatter", "cache"}:
        return (
            "Paged KV Cache Scatter Write", "aclnnScatterPaKvCache", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/attention/mla_v1.py",
            False, "", "kv_store",
        )
    if fam == "rotary" or suffix in {"rotary", "rope", "q_rotary", "kv_rotary", "output_inverse_rotary"}:
        return (
            f"Rotary Position Embedding ({suffix})", "InplacePartialRotaryMul", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/rotary_embedding.py",
            False, "", suffix,
        )
    if fam in {"swiglu", "situ"}:
        if fam == "situ":
            return (
                "K3 SiTU Gated Activation (beta=4, linear_beta=25)",
                "K3 SituAndMul",
                BACKEND_TRITON,
                f"{KERNEL_STATUS_NEW_OP} K3 SituAndMul(beta=4, linear_beta=25)",
                "vllm-ascend/vllm_ascend/models/kimi_k3/modeling_kimi_k3.py",
                True,
                "Fused K3 SituAndMul activation (non-standard SiLU variant not in shipping CANN SwiGlu)",
                suffix,
            )
        return (
            "SwiGLU Activation", "SwiGlu", BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/ops/activation.py", False, "", suffix,
        )
    if fam in {"embedding", "engram_gather"}:
        return (
            "Token / Engram Table Gather", "aclnnEmbedding (GatherV2)", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm/vllm/model_executor/layers/vocab_parallel_embedding.py",
            False, "", suffix,
        )
    if fam == "logits":
        role = "Draft Markov Logits Head" if "markov" in suffix else "Vocab Parallel LM Head GEMM"
        return (
            role, mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "vllm/vllm/model_executor/layers/logits_processor.py", False, "", suffix,
        )
    if fam == "acceptance":
        return (
            "Speculative Verification & Token Acceptance",
            "speculative_verify_accept",
            BACKEND_TORCH_NPU,
            KERNEL_STATUS_FRAMEWORK,
            "vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py",
            False, "", "accept",
        )
    if fam in {"gemm", "projection", "gate_up_projection", "down_projection",
               "qkv_projection", "output_projection"}:
        role_map = {
            "q_a": "Q-A Low-Rank Projection Linear",
            "kv_a": "KV-A Compressed Projection Linear",
            "qkv_a": "Fused Q/KV-A Low-Rank Projection",
            "q_b": "Q-B Head Expansion Linear",
            "qkv": "Fused QKV Projection Linear",
            "output_gate": "Gated-MLA Output Gate Linear (W_g)",
            "output": "Attention Output Projection (O-Proj)",
            "grouped_output_a": "Grouped Low-Rank O-Proj Stage A",
            "grouped_output_b": "Grouped Low-Rank O-Proj Stage B",
            "gate_up": "Dense FFN Gate+Up Projection",
            "down": "Dense FFN Down Projection",
            "features": "DSpark Context Feature Projection",
        }
        return (
            role_map.get(suffix, f"Linear Projection ({suffix})"),
            mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, mm_src, False, "", suffix,
        )
    if suffix == "gated_output":
        return (
            "Gated-MLA Sigmoid × Attention Output", "aclnnSigmoid + aclnnMul", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/attention/mla_v1.py",
            False, "", "gated_output",
        )
    if suffix in {"dcp_query_pack", "dcp_output_pack", "candidate_pack"} or fam in {"tensor_copy", "copy", "tensor_fill"}:
        return (
            f"Tensor Pack / Layout Materialize ({suffix})", "TensorMove / Reshape", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/attention/context_parallel/mla_cp.py",
            False, "", suffix,
        )
    eager_vec_map = {
        "cast": ("Dtype Cast", "Cast"),
        "pow": ("Elementwise Power", "Pows"),
        "pows": ("Elementwise Power", "Pows"),
        "reducemean": ("Reduction Mean", "ReduceMean"),
        "rsqrt": ("Reciprocal Sqrt", "Rsqrt"),
        "sigmoid": ("Sigmoid Activation", "Sigmoid"),
        "mul": ("Elementwise Multiply", "Mul"),
        "adds": ("Scalar Bias Add", "Adds"),
        "add": ("Elementwise Add", "Add"),
    }
    if suffix in eager_vec_map or fam in eager_vec_map:
        e_role, e_kern = eager_vec_map.get(suffix) or eager_vec_map[fam]
        return (
            e_role, e_kern, BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/ops/common.py", False, "", suffix or fam,
        )
    return (
        f"Elementwise / Aux Op ({suffix or fam})",
        nat_kind if (nat_kind and nat_kind not in BANNED_IMPLEMENTATION_NAMES) else "aclnnAdd / VectorOp",
        BACKEND_CANN, KERNEL_STATUS_EXISTING,
        "vllm-ascend/vllm_ascend/ops/common.py", False, "", suffix or fam,
    )


def _classify_norm_quant_tuple(
    suffix: str, fam: str, rec: str, meta: Mapping[str, Any], nat_kind: str,
    *, is_raw_eager: bool = False,
) -> tuple[str, str, str, str, str, bool, str, str]:
    out_fmt = str(meta.get("output_format") or "").lower()
    is_mx = out_fmt in {"fp8_e4m3", "mxfp8"} and str(meta.get("activation_scale_format") or "") == "fp8_e8m0"
    if fam == "layernorm":
        return (
            "Indexer K LayerNorm (1-kernel FP32-UB)" if "index" in suffix else "LayerNorm (1-kernel FP32-UB)",
            "aclnnLayerNorm_LayerNormV3", BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/ops/layernorm.py", False, "", suffix,
        )
    if fam == "add_rmsnorm_quant":
        kern = "AddRmsNormDynamicMxQuant" if is_mx else "AddRmsNormDynamicQuant"
        return (
            "Fused Residual Add + RMSNorm + Dynamic Quant", kern, BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/layernorm.py", False, "", suffix,
        )
    if fam == "rmsnorm_quant":
        kern = "RmsNormDynamicMxQuant" if is_mx else "RmsNormDynamicQuant"
        return (
            "Fused RMSNorm + Dynamic Quant", kern, BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/layernorm.py", False, "", suffix,
        )
    if fam == "add_rmsnorm":
        return (
            "Fused Residual Add + RMSNorm", "AddRmsNorm", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/layernorm.py", False, "", suffix,
        )
    if fam == "quantize":
        kern = "DynamicMxQuant" if is_mx else "DynamicQuant"
        return (
            f"Dynamic Activation Quantize ({suffix})", kern, BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/quantization/w8a8_dynamic.py",
            False, "", suffix,
        )
    role_map = {
        "q_norm": "Q Latent RMSNorm",
        "kv_norm": "KV Latent RMSNorm",
        "latent_norm": "MoE Routed Latent RMSNorm",
        "final_norm": "Final Model RMSNorm",
        "context_norm": "DSpark Context RMSNorm",
        "qk_norm": "Q/K Head RMSNorm",
    }
    if is_raw_eager or "unfused" in rec or "eager" in rec:
        return (
            f"Unfused Eager {role_map.get(suffix, f'RMSNorm ({suffix})')} (8 kernels)",
            "Cast->Pows->ReduceMean->Adds->Rsqrt->Mul->Cast",
            BACKEND_TORCH_NPU, KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/ops/layernorm.py", False, "", suffix,
        )
    return (
        role_map.get(suffix, f"RMSNorm ({suffix})"), "RmsNorm", BACKEND_CANN,
        KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/layernorm.py", False, "", suffix,
    )


def _classify_attention_tuple(
    suffix: str, rec: str, nat_kind: str, layer_plan: Any | None,
) -> tuple[str, str, str, str, str, bool, str, str]:
    topk = int(getattr(layer_plan, "index_topk", 0) or 0) if layer_plan is not None else 0
    if "mixed_quant_sparse" in rec or suffix == "mqsmla":
        return (
            f"Mixed-Quant Sparse Flash MLA (topk={topk or DEFAULT_SPARSE_INDEX_TOPK})",
            "aclnnMixedQuantSparseFlashMla", BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "cann/ops-transformer/src/mc2_or_sfa/mixed_quant_sparse_flash_mla.cpp",
            False, "", "mqsmla",
        )
    if topk > 0 or "sparse" in rec or "paired_mla" in rec:
        return (
            f"Sparse Flash Attention (TopK={topk or DEFAULT_SPARSE_INDEX_TOPK})",
            "aclnnSparseFlashAttention", BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/attention/sfa_v1.py",
            False, "", "latent_attention_batch",
        )
    return (
        "Fused Infer Attention Score v2 (FIA v2)",
        "aclnnFusedInferAttentionScoreV2", BACKEND_CANN, KERNEL_STATUS_EXISTING,
        "vllm-ascend/vllm_ascend/attention/mla_v1.py", False, "", suffix,
    )


def _classify_indexer_tuple(
    suffix: str, fam: str, rec: str, mm_kernel: str, layer_plan: Any | None,
) -> tuple[str, str, str, str, str, bool, str, str]:
    topk = int(getattr(layer_plan, "index_topk", 0) or 0) if layer_plan is not None else DEFAULT_SPARSE_INDEX_TOPK
    src = "vllm-ascend/vllm_ascend/attention/sfa_v1.py"
    if suffix in {"index_scores", "quant_lightning_indexer"} or fam == "indexer":
        kern = "aclnnPoolKeyIndexer" if "pool_key" in rec else "aclnnQuantLightningIndexer"
        return (
            f"DSA Lightning Indexer Score (topk={topk})", kern, BACKEND_CANN,
            KERNEL_STATUS_EXISTING, src, False, "", "index_scores",
        )
    if suffix == "index_topk":
        return (
            f"DSA Local TopK-{topk} Selection", "aclnnTopk", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, src, False, "", "index_topk",
        )
    if suffix == "candidate_merge":
        return (
            f"DSA Cross-DCP TopK-{topk} Candidate Merge", "aclnnTopk / CandidateMerge", BACKEND_CANN,
            KERNEL_STATUS_FRAMEWORK, src, False, "", "candidate_merge",
        )
    if suffix == "candidate_gather":
        return (
            "DSA Candidate Indices AllGather (DCP)", "hcom_allGather_", BACKEND_HCCL_MC2,
            KERNEL_STATUS_EXISTING, src, False, "", "candidate_gather",
        )
    if suffix == "index_store" or fam in {"indexer_quant_cache", "kv_cache_write"}:
        return (
            "DSA Indexer K Cache Scatter Write", "aclnnScatterPaKvCache", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, src, False, "", "index_store",
        )
    if suffix == "index_norm":
        return ("DSA Indexer K LayerNorm", "LayerNormV3", BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", "index_norm")
    if suffix == "index_query_quantize":
        return ("DSA Indexer Query Dynamic Quantize", "DynamicQuant", BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", "index_query_quantize")
    if suffix in {"index_k", "index_q", "index_weights", "index_wq", "index_wk"}:
        labels = {
            "index_k": "DSA Indexer Key Projection (W_k)",
            "index_q": "DSA Indexer Query Projection (W_q)",
            "index_weights": "DSA Indexer Head Weights Linear",
            "index_wq": "DSA Indexer Query Projection",
            "index_wk": "DSA Indexer Key Projection",
        }
        return (labels[suffix], mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", suffix)
    if suffix == "candidate_pack":
        return ("DSA Candidate Indices Pack", "TensorMove / GatherV2", BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", "candidate_pack")
    return (f"DSA Indexer Aux ({suffix})", "TensorMove / GatherV2", BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", suffix)


def _classify_mhc_tuple(
    suffix: str, fam: str, rec: str, mm_kernel: str, layer_plan: Any | None, *, is_ideal: bool,
) -> tuple[str, str, str, str, str, bool, str, str]:
    hc_lowrank = int(getattr(layer_plan, "hc_lowrank", 0) or 0) if layer_plan is not None else 0
    hc_sinkhorn = int(getattr(layer_plan, "hc_sinkhorn_iters", 0) or 0) if layer_plan is not None else 0
    is_lowrank = (
        (hc_lowrank > 0 and hc_sinkhorn == 0)
        or rec in {"hc_gate_mix", "hc_combine", "hc_combine_rmsnorm", "hc_silu", "hc_lowrank_down_inject", "hc_lowrank_up"}
    )
    src = (
        "vllm-ascend/vllm_ascend/models/qwen4_exp/hyper_connection.py"
        if is_lowrank else "vllm-ascend/vllm_ascend/ops/mhc.py"
    )
    if rec == "hc_silu" or suffix.endswith("hc_silu"):
        return (f"mHC Low-Rank SiLU ({suffix})", "Silu", BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", suffix)
    if fam == "rmsnorm" or rec == "grouped_gemma_rmsnorm" or suffix.endswith("hc_norm"):
        return (f"mHC Grouped RMSNorm ({suffix})", "RmsNorm", BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", suffix)
    if fam == "gemm" or rec in {"hc_lowrank_down_inject", "hc_lowrank_up", "hc_projection", "hc_coefficients_fused"}:
        return (f"mHC Hyper-Connection Projection ({suffix})", mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", suffix)
    if rec in {"hc_combine", "hc_combine_rmsnorm"} or suffix.endswith("hc_post"):
        kern = "Qwen4ExpHcCombine" if is_lowrank else "mhc_post_combine"
        return (
            f"mHC Post Combine ({suffix})", kern, BACKEND_CANN if is_lowrank else BACKEND_TRITON,
            KERNEL_STATUS_NEW_OP if (is_lowrank and is_ideal) else KERNEL_STATUS_FRAMEWORK,
            src, bool(is_lowrank and is_ideal),
            "UB-resident fused Qwen4ExpHcCombine eliminating strided HBM round-trips" if (is_lowrank and is_ideal) else "",
            suffix,
        )
    if is_lowrank:
        return (
            f"mHC Low-Rank Gated Residual Mix ({suffix})", "Qwen4ExpGatedResidualMix", BACKEND_CANN,
            KERNEL_STATUS_NEW_OP if is_ideal else KERNEL_STATUS_FRAMEWORK,
            src, bool(is_ideal),
            "UB-resident fused Qwen4ExpGatedResidualMix (Norm+Down+SiLU+Up+Mix)" if is_ideal else "",
            suffix,
        )
    kern = "mhc_sinkhorn_mix_kernel" if ("sinkhorn" in suffix or "mix" in suffix) else "HcPre"
    return (
        f"mHC Hyper-Connection ({suffix})", kern,
        BACKEND_TRITON if fam == "mhc" else BACKEND_CANN,
        KERNEL_STATUS_FRAMEWORK if fam == "mhc" else KERNEL_STATUS_EXISTING,
        src, False, "", suffix,
    )


def _classify_linear_attn_tuple(
    suffix: str, fam: str, rec: str, meta: Mapping[str, Any], is_ideal: bool, mm_kernel: str,
    *, is_raw_eager: bool = False, layer_plan: Any | None = None,
) -> tuple[str, str, str, str, str, bool, str, str]:
    kda_src = "vllm/vllm/third_party/flash_linear_attention/ops/kda.py"
    if suffix.startswith("recurrent_scan") or "recurrent_kda" in rec:
        has_spec = bool(meta.get("kda_recurrent_params", {}).get("recover_ssm")) or is_ideal
        if is_ideal:
            return (
                "Fused Recurrent KDA (Spec Verify + Persistent State)",
                "flash_kda_persistent", BACKEND_CANNBOT_DSL,
                f"{KERNEL_STATUS_NEW_OR_EXTEND} Spec 2D State-Slot Writeback & Persistent UB",
                "cann/cannbot-dsl/samples/attention/flash_kda/flash_kda_persistent.py",
                True,
                "Persistent cannbot-dsl / AscendC recurrent KDA with 2D accepted-token state slot writeback",
                "recurrent_scan",
            )
        return (
            "Fused Recurrent KDA Decode", "fused_recurrent_kda", BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} Speculative KDA State-Slot Writeback" if has_spec else KERNEL_STATUS_EXISTING,
            kda_src, has_spec,
            "Speculative 2D state-slot writeback extension for fused_recurrent_kda" if has_spec else "",
            "recurrent_scan",
        )
    if suffix == "decode_recurrent_scan" or fam == "recurrent_vector":
        return (
            "Fused Recurrent GDN Decode", "RecurrentGatedDeltaRule", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/gdn.py", False, "", "decode_recurrent_scan",
        )
    if "recover_ssm" in suffix or "recover_ssm" in rec:
        return (
            "Speculative Accepted-Token SSM State Commit (RecoverSSM)",
            "recover_ssm_state_commit", BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} Speculative Accepted-State Commit Kernel",
            "vllm-ascend/vllm_ascend/ops/linear_attention/recover_ssm.py",
            True,
            "Compact accepted-token SSM/KDA state replay & single HBM writeback commit kernel",
            "recover_ssm_commit",
        )
    if suffix == "short_convolution" or fam in {"short_convolution", "causal_conv1d_fn"}:
        return (
            "Causal Conv1d + SiLU (Kernel=4)", "CausalConv1d", BACKEND_CANN,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} Spec Accepted-Token Rolling Conv Update" if is_ideal else KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/ops/causal_conv1d.py", is_ideal,
            "Speculative multi-token rolling conv_state update for CausalConv1d" if is_ideal else "",
            "short_convolution",
        )
    if suffix == "linear_output_norm_gate":
        if is_raw_eager or (not is_ideal and ("unfused" in rec or "eager" in rec)):
            return (
                "Unfused Eager RMSNorm + Sigmoid Gate (11 kernels)",
                "Cast->Pows->ReduceMean->Adds->Rsqrt->Mul->Cast->Mul->Sigmoid->Mul->Cast",
                BACKEND_TORCH_NPU, KERNEL_STATUS_EXISTING, kda_src, False, "", "linear_output_norm_gate",
            )
        return (
            "Fused RMSNorm + Sigmoid Output Gate", "FusedRMSNormGated", BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} FusedRMSNormGated(eps=1e-5, sigmoid)",
            kda_src, True,
            "Fused RMSNorm + sigmoid gate kernel on [T, Hv, Dv] replacing 11-kernel eager chain",
            "linear_output_norm_gate",
        )
    if "l2norm" in suffix:
        return (
            f"Q/K L2 Normalization ({suffix})", "l2norm_fwd_kernel2", BACKEND_TRITON,
            KERNEL_STATUS_EXISTING, "vllm/vllm/third_party/flash_linear_attention/ops/l2norm.py",
            False, "", suffix,
        )
    if suffix == "delta_gates" or fam == "gdn_gating":
        return (
            "Delta Decay Gate Transform", "fused_gdn_gating_kernel", BACKEND_TRITON,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/gdn.py", False, "", "delta_gates",
        )
    if fam in {"reduce_scatter", "allgather", "allreduce"}:
        return _classify_collective_tuple(suffix, fam, rec)
    if fam in {"gemm", "projection"} or suffix.startswith("linear_"):
        is_gdn_proj = (
            suffix in {"linear_qkvz", "linear_beta_alpha"}
            or "gdn" in rec
            or (layer_plan is not None and str(getattr(layer_plan, "attention", "")).lower() == "gdn")
        )
        lin_roles = {
            "linear_qkvbfg_a": "KDA Fused Q/K/V/Beta/Decay-A/Gate Projection",
            "linear_decay_b": "KDA Low-Rank Decay-B Projection",
            "linear_gate_b": "KDA Low-Rank Gate-B Projection",
            "linear_output": "GDN Output Projection (O-Proj)" if is_gdn_proj else "KDA Output Projection (O-Proj)",
            "linear_qkvz": "GDN Fused QKVZ Projection",
            "linear_beta_alpha": "GDN Beta/Alpha Gate Projection",
        }
        proj_src = (
            "vllm-ascend/vllm_ascend/ops/gdn.py"
            if is_gdn_proj else "vllm-ascend/vllm_ascend/models/kimi_k3/kda.py"
        )
        return (
            lin_roles.get(suffix, f"Linear Attention Projection ({suffix})"),
            mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING,
            proj_src, False, "", suffix,
        )
    return (
        f"Linear Attention Layout / Copy ({suffix})", "TensorMove / Slice", BACKEND_CANN,
        KERNEL_STATUS_EXISTING, kda_src, False, "", suffix,
    )


def _classify_moe_tuple(
    suffix: str, fam: str, rec: str, meta: Mapping[str, Any], nat_kind: str,
    is_ideal: bool, is_quant_mm: bool, is_mxfp4_w: bool, mm_kernel: str,
) -> tuple[str, str, str, str, str, bool, str, str]:
    moe_src = "vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py"
    if suffix == "router" or fam == "router":
        role = "Shared Expert Gate Linear" if "shared" in suffix else "MoE Router Gate Linear (W_router)"
        return (role, mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", suffix)
    if suffix == "moe_gating_topk":
        return (
            "MoE Sigmoid + Correction Bias TopK Gating", "MoeGatingTopK", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, moe_src, False, "", "moe_gating_topk",
        )
    if suffix in {"latent_down", "latent_up"}:
        role = "MoE Routed Latent Down Projection (H->Hm)" if suffix == "latent_down" else "MoE Routed Latent Up Projection (Hm->H)"
        return (role, mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", suffix)
    if suffix == "latent_norm":
        return ("MoE Routed Latent RMSNorm", "RmsNorm", BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", "latent_norm")
    if "megamoe" in rec or "dispatch_ffn_combine" in rec:
        return (
            "Fused MegaMoE (Dispatch + GMM1 + Act + GMM2 + Combine)",
            "dispatch_ffn_combine (MegaMoE)", BACKEND_CANN,
            f"{KERNEL_STATUS_EXTEND_TILING} MXFP4 MegaMoE Compact Tiling & AIV/Cube Ping-Pong",
            "vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py", True,
            "Fused MegaMoE dispatch_ffn_combine supporting packed MXFP4 uint8 + K3 SiTU + compact tiling",
            "megamoe_arch35",
        )
    if suffix == "dispatch" or "dispatch" in rec:
        kern = nat_kind if (nat_kind and "Dispatch" in nat_kind) else "MoeDistributeDispatchV2"
        return ("MoE EP Token Distribute Dispatch (MC2)", kern, BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, moe_src, False, "", "dispatch")
    if suffix in {"unpack", "combine"} or "combine" in rec:
        kern = nat_kind if (nat_kind and "Combine" in nat_kind) else "MoeDistributeCombineV2"
        return ("MoE EP Token Distribute Combine (MC2)", kern, BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, moe_src, False, "", "unpack")
    if suffix in {"expert_up", "expert_down"} or fam == "grouped_gemm":
        is_up = suffix == "expert_up" or "gate_up" in rec
        q_tag = "MXFP4" if is_mxfp4_w else ("W8A8" if is_quant_mm else "BF16")
        role = f"Routed Expert {'Gate+Up GMM1' if is_up else 'Down GMM2'} ({q_tag})"
        if is_ideal and is_mxfp4_w:
            return (
                role, "GroupedMatmul (MXFP4 Compact Tiling)", BACKEND_CANN,
                f"{KERNEL_STATUS_EXTEND_TILING} MXFP4 GroupedMatmul Compact Tiling",
                "cann/ops-transformer/src/mc2/grouped_matmul.cpp", True,
                f"MXFP4 group=32 compact tiling + AIV/Cube double-buffering for {role}",
                suffix,
            )
        return (role, "GroupedMatmul", BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", suffix)
    if suffix == "expert_act" or suffix == "shared.situ":
        if fam == "situ" or "situ" in rec:
            return (
                "Routed Expert K3 SiTU+Mul Activation" if suffix == "expert_act" else "Shared Expert K3 SiTU+Mul Activation",
                "K3 SituAndMul", BACKEND_TRITON,
                f"{KERNEL_STATUS_NEW_OP} K3 SituAndMul(beta=4, linear_beta=25) + MXQuant",
                "vllm-ascend/vllm_ascend/models/kimi_k3/modeling_kimi_k3.py", True,
                "Fused K3 SituAndMul(beta=4, linear_beta=25) + optional DynamicMxQuant epilogue",
                suffix,
            )
        return ("Routed Expert SwiGLU Activation", "SwiGlu", BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", suffix)
    if suffix == "join":
        return (
            "MoE Routed + Shared Residual Add", "aclnnAdd", BACKEND_CANN,
            KERNEL_STATUS_FRAMEWORK if is_ideal else KERNEL_STATUS_EXISTING,
            moe_src, False, "", "join",
        )
    if suffix.startswith("shared."):
        if fam in COLLECTIVE_FAMILIES:
            return _classify_collective_tuple(suffix, fam, rec)
        s_roles = {
            "shared.gate_up": "Shared Expert Gate+Up Linear (MSO Stream 1)",
            "shared.down": "Shared Expert Down Linear (MSO Stream 1)",
            "shared.swiglu": "Shared Expert SwiGLU (MSO Stream 1)",
        }
        kern = "SwiGlu" if "swiglu" in suffix else mm_kernel
        st = f"{KERNEL_STATUS_FRAMEWORK} (MSO Stream 1 Overlap)" if is_ideal else KERNEL_STATUS_EXISTING
        return (s_roles.get(suffix, f"Shared Expert ({suffix})"), kern, BACKEND_CANN, st, moe_src, False, "", suffix)
    return (f"MoE Aux ({suffix})", "aclnnAdd / VectorOp", BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", suffix)


def _classify_engram_ple_tuple(
    suffix: str, fam: str, rec: str, mm_kernel: str,
) -> tuple[str, str, str, str, str, bool, str, str]:
    src = "vllm-ascend/vllm_ascend/ops/engram.py"
    if fam in {"engram_rh2d", "h2d"} or "host_uva" in suffix or "rh2d" in rec:
        return (
            "PLE / Engram Host DRAM UVA H2D Gather",
            "aclrtMemcpyAsync (Host->Device UVA H2D)",
            BACKEND_CANN,
            KERNEL_STATUS_FRAMEWORK,
            src, False, "", suffix,
        )
    if fam in {"engram_query", "engram_ids"} or "ngram" in suffix:
        return (
            "PLE / Engram N-Gram Hash ID Lookup",
            "ple_ngram_hash_lookup",
            BACKEND_TORCH_NPU,
            KERNEL_STATUS_FRAMEWORK,
            src, False, "", suffix,
        )
    if fam in {"engram_apply"} or "gate" in suffix:
        return (
            "PLE / Engram Gated Residual Fusion",
            "ple_gated_fusion_kernel",
            BACKEND_TRITON,
            KERNEL_STATUS_FRAMEWORK,
            src, False, "", suffix,
        )
    if fam in {"conv1d", "short_convolution"} or "conv" in suffix:
        return (
            "PLE / Engram Depthwise Short Conv1d",
            "CausalConv1d",
            BACKEND_CANN,
            KERNEL_STATUS_EXISTING,
            src, False, "", suffix,
        )
    if fam == "quantize" or suffix.endswith(".quantize"):
        return (
            "PLE / Engram Activation Quantize",
            "DynamicMxQuant",
            BACKEND_CANN,
            KERNEL_STATUS_EXISTING,
            src, False, "", suffix,
        )
    if fam in {"gemm", "projection"} or "proj" in suffix:
        return (
            "PLE / Engram KV Projection",
            mm_kernel,
            BACKEND_CANN,
            KERNEL_STATUS_EXISTING,
            src, False, "", suffix,
        )
    if fam in COLLECTIVE_FAMILIES or "allreduce" in suffix or "broadcast" in suffix:
        return _classify_collective_tuple(suffix, fam, rec)
    return (
        f"PLE / Engram Table Gather ({suffix})",
        "aclnnEmbedding (GatherV2)",
        BACKEND_CANN,
        KERNEL_STATUS_EXISTING,
        src, False, "", suffix,
    )


def _classify_collective_tuple(
    suffix: str, fam: str, rec: str,
) -> tuple[str, str, str, str, str, bool, str, str]:
    hccl_src = "vllm-ascend/vllm_ascend/distributed/parallel_state.py"
    if fam == "allgather" or "allgather" in rec:
        role = "DCP Query AllGather" if "dcp_query" in suffix else (
            "Sequence-Parallel AllGather" if "sp_gather" in suffix or "sp_output" in suffix else (
                "Vocab Logits AllGather" if "logits" in suffix else f"HCCL AllGather ({suffix})"
            )
        )
        return (role, "hcom_allGather_", BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, hccl_src, False, "", suffix)
    if fam in {"reduce_scatter", "reducescatter"} or "reduce_scatter" in rec:
        role = "Sequence-Parallel ReduceScatter" if ("sp_scatter" in suffix or "reduce_scatter" in suffix) else f"HCCL ReduceScatter ({suffix})"
        return (role, "hcom_reduceScatter_", BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, hccl_src, False, "", suffix)
    if fam in {"alltoall", "alltoallv"} or "alltoall" in rec:
        role = "DCP Output AllToAllV" if "dcp_output" in suffix else f"HCCL AllToAllV ({suffix})"
        return (role, "hcom_allToAllV_", BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, hccl_src, False, "", suffix)
    return (
        f"TP Sum AllReduce ({suffix})", "hcom_allReduce_", BACKEND_HCCL_MC2,
        KERNEL_STATUS_EXISTING, hccl_src, False, "", suffix,
    )
