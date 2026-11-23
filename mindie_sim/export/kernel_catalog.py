"""Domain-standard Ascend submodule, macro-module, and kernel catalog classifier.

[Role] Map lowered ``mindie_sim`` operators into:
  (1) domain-standard Ascend submodules (zero ``Other``),
  (2) architectural macro-modules for Sheet 1 executive duration & share % tables,
  (3) single-source architecture-aware Ascend kernel identities projected from
      ``NativeCallBinding`` / ``export_op_structure`` / typed operator family
      capabilities, with honest empty ``ascend_kernel`` for logical-only rows
      and zero guessed source file paths.
[Invariants] Zero banned internal Python simulator recipe names, internal stage
  names (``step5_*``), cross-generation aliases (A5 MLA ``FusedInferAttentionScore``,
  A5 ``dispatch_ffn_combine``, or cross-gen executor tags), or generic fallback
  strings escape into ``ascend_kernel``. Every operator classifies into a valid
  non-``Other`` submodule and one macro-module.
[Timeouts & Locks] Pure deterministic in-memory lookup; no I/O, locks, or device
  contexts.
[Side-Effect Policy] Read-only over ``OpSpec``, ``NativeCallBinding``, and
  ``Case`` metadata.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import re
from typing import Any

from mindie_sim.prediction.native_binding import bind_native_call
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure_contracts import predicted_contract
from mindie_sim.structure_implementations import _lookup_registry_spec, native_implementation

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
    MACRO_MOD_DENSE_FFN,
    MACRO_MOD_DRAFT_LAYERS,
    MACRO_MOD_DSA_LI,
    MACRO_MOD_ENGRAM,
    MACRO_MOD_GATED_O_PROJ,
    MACRO_MOD_KDA_GDN,
    MACRO_MOD_LATENT_ATTN,
    MACRO_MOD_MLA_PROLOG,
    MACRO_MOD_MOE_ROUTED,
    MACRO_MOD_MOE_ROUTER,
    MACRO_MOD_MOE_SHARED,
    MACRO_MOD_RESIDUAL_NORM,
    MACRO_MOD_VERIFY_HEAD,
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
_NAMESPACE_BACKEND_MAP: dict[str, str] = {
    "aclnn": BACKEND_CANN,
    "cann_kernel": BACKEND_CANN,
    "hccl": BACKEND_HCCL_MC2,
    "triton": BACKEND_TRITON,
    "torch": BACKEND_TORCH_NPU,
}
_INTERNAL_SYMBOL_PREFIXES = (
    "step5_", "qsa_dual_", "ascend_paged_mla_", "ascend_sparse_flash_",
)
_INTERNAL_ARCH_TAG_RE = re.compile(r"_arch\d+$")
_REJECTED_SYMBOLS = frozenset({
    "MegaMoeArch35", "megamoe_arch35",
})
_EAGER_VECTOR_MAP: dict[str, tuple[str, str]] = {
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
_PROJ_ROLES: dict[str, str] = {
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
_NORM_ROLES: dict[str, str] = {
    "q_norm": "Q Latent RMSNorm",
    "kv_norm": "KV Latent RMSNorm",
    "latent_norm": "MoE Routed Latent RMSNorm",
    "final_norm": "Final Model RMSNorm",
    "context_norm": "DSpark Context RMSNorm",
    "qk_norm": "Q/K Head RMSNorm",
}
_INDEXER_PROJ_ROLES: dict[str, str] = {
    "index_k": "DSA Indexer Key Projection (W_k)",
    "index_q": "DSA Indexer Query Projection (W_q)",
    "index_weights": "DSA Indexer Head Weights Linear",
    "index_wq": "DSA Indexer Query Projection",
    "index_wk": "DSA Indexer Key Projection",
}
_PURE_GLUE_FAMILIES = frozenset({
    "allreduce", "all_reduce", "tp_allreduce", "allgather", "all_gather",
    "reduce_scatter", "reducescatter", "alltoall", "alltoallv",
    "collective", "hccl", "p2p", "quantize",
    "tensor_copy", "copy", "view", "reshape", "slice", "permute",
    "transpose", "cat", "concat", "contiguous", "tensor_fill", "pad", "split",
})
_GLUE_ROLE_MARKERS = (
    "contiguous", "materialize", "_zero", "_copy", "_flat", "concatenate",
    "sp_gather", "sp_scatter", "sp_shard", "sp_output_gather",
    "reduce_scatter", "output_reduce", "linear_reduce",
    "dcp_query_", "dcp_output_", "candidate_pack", "candidate_gather",
    "logits_gather", "features_gather",
)
_SCOPE_ATOM_PREFIXES = ("attention.", "ffn.", "shared.", "final.")


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
    """Classify an operator into a domain-standard Ascend stage (ZERO ``Other``)."""
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
    """Assign an operator to its single executive L2 architectural macro-module."""
    if op.model_scope == "draft":
        return MACRO_MOD_DRAFT_LAYERS
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
    if any(tok in s for tok in ("megamoe", "dispatch_ffn_combine") for s in (rec, kern_l)):
        return MACRO_MOD_MOE_ROUTED
    is_moe_ep_comm = sub in {SUB_MOE_ROUTER, SUB_MOE_ROUTED} and (
        suffix in {"dispatch", "unpack", "combine"}
        or "dispatch" in rec or "combine" in rec or fam in _MOE_EP_COMM_FAMILIES
    )
    if sub == SUB_MOE_ROUTER or is_moe_ep_comm:
        return MACRO_MOD_MOE_ROUTER
    if op.task_type == "HCCL" or fam in COLLECTIVE_FAMILIES:
        return MACRO_MOD_COLLECTIVES
    sub_map = {
        SUB_EMBED_TAIL: MACRO_MOD_VERIFY_HEAD,
        SUB_KDA: MACRO_MOD_KDA_GDN,
        SUB_GDN: MACRO_MOD_KDA_GDN,
        SUB_MLA_PROLOG: MACRO_MOD_MLA_PROLOG,
        SUB_DSA_LI: MACRO_MOD_DSA_LI,
        SUB_LATENT_ATTN: MACRO_MOD_LATENT_ATTN,
        SUB_STD_ATTN: MACRO_MOD_LATENT_ATTN,
        SUB_GATED_MLA_O: MACRO_MOD_GATED_O_PROJ,
        SUB_MOE_ROUTED: MACRO_MOD_MOE_ROUTED,
        SUB_DENSE_FFN: MACRO_MOD_DENSE_FFN,
    }
    return sub_map.get(sub, MACRO_MOD_RESIDUAL_NORM)


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
    return suffix.startswith(("shared.", "shared_gate")) or "shared_gate" in rec


def _is_attn_res_or_norm_op(suffix: str, fam: str, rec: str) -> bool:
    if fam in {"attn_res_prefix", "mhc"} or "attn_res" in suffix or ".hc_" in f".{suffix}":
        return True
    if suffix in {
        "norm1", "norm2", "input_layernorm", "post_attention_layernorm",
        "attention.residual", "ffn.residual", "residual",
    }:
        return True
    if rec.endswith("_rmsnorm") and "grouped" in rec:
        return True
    return rec in {
        "rmsnorm", "add_rmsnorm", "residual_add", "hc_silu", "hc_values_fp32",
        "hc_projection", "hc_lowrank_down_inject", "hc_lowrank_up",
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
    return attn in {"mla", "mla_dsa", "dsa"} and (
        suffix in {
            "output_gate", "output", "output.quantize", "output_reduce",
            "output_reduce.reduce_scatter", "output_reduce.reduce_scatter.pad",
            "tp_attn", "tp_attn.reduce_scatter",
        }
        or suffix.startswith("output_reduce.")
    )


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


def resolve_comm_domain_tag(
    op_id: str, family: str, recipe: str, meta: Mapping[str, Any], case: Case,
) -> str:
    """Resolve the exact active collective communication domain tag."""
    suffix = strip_op_prefix(op_id).lower()
    fam = str(family or "").strip().lower()
    rec = str(recipe or "").strip().lower()
    axis = str(meta.get("comm_axis") or meta.get("group_axis") or "").strip().lower()
    explicit_group = int(meta.get("comm_group_size") or 0)
    p = case.runtime.parallel
    tp = explicit_group if (explicit_group > 0 and axis == "tp") else max(1, int(p.tp))
    ep = explicit_group if (explicit_group > 0 and axis == "ep") else max(1, int(p.ep))
    dcp = explicit_group if (explicit_group > 0 and axis in {"dcp", "cp"}) else max(1, int(p.cp))
    pcp = explicit_group if (explicit_group > 0 and axis == "pcp") else max(1, int(p.pcp))
    pp = explicit_group if (explicit_group > 0 and axis == "pp") else max(1, int(p.pp))
    is_comm = (
        fam in COLLECTIVE_FAMILIES
        or fam in {"all_gather_matmul", "matmul_reduce_scatter"}
        or "megamoe" in rec or "dispatch_ffn_combine" in rec
        or suffix.startswith(("dcp_", "pcp_", "sp_"))
        or any(tok in suffix for tok in ("sp_gather", "sp_scatter", "reduce_scatter"))
        or any(tok in rec for tok in ("allgather", "allreduce", "alltoall"))
    )
    if not is_comm:
        return ""
    if (axis == "ep" or fam in _MOE_EP_COMM_FAMILIES
            or suffix in {"dispatch", "unpack", "combine"}
            or any(tok in rec for tok in ("dispatch", "combine", "megamoe"))):
        return f"EP={ep}"
    if axis in {"dcp", "cp"} or suffix.startswith("dcp_") or "dcp_" in rec or "candidate_gather" in suffix:
        return f"DCP={dcp}"
    if axis == "pcp" or suffix.startswith("pcp_"):
        return f"PCP={pcp}"
    if axis == "pp" or fam == "p2p":
        return f"PP={pp}"
    if (any(tok in suffix for tok in ("sp_gather", "sp_scatter", "sp_output", "reduce_scatter"))
            or fam in {"reduce_scatter", "reducescatter", "all_gather_matmul", "matmul_reduce_scatter"}):
        return f"TP={tp} (SP)"
    return f"TP={tp}"


def _atom_scope_prefix(suffix: str) -> str:
    return next((pfx for pfx in _SCOPE_ATOM_PREFIXES if suffix.startswith(pfx)), "")


def _is_pure_glue_op(suf: str, fam: str, rec: str, meta: Mapping[str, Any] | None = None) -> bool:
    if meta:
        if meta.get("is_glue") or meta.get("view_only"):
            return True
        if (meta.get("comm_axis") or meta.get("group_axis")) and fam not in {
            "all_gather_matmul", "matmul_reduce_scatter", "dispatch", "combine", "mc2",
        } and suf not in {"dispatch", "unpack", "combine"}:
            return True
    if suf in {"pad", "tp", "tp_attn", "shared.tp"} or suf.endswith((".quantize", ".pad")):
        return True
    if fam in _PURE_GLUE_FAMILIES and suf not in {
        "dispatch", "unpack", "combine", "expert_up", "expert_down", "expert_act",
    }:
        return True
    if any(tok in suf for tok in _GLUE_ROLE_MARKERS):
        return True
    return any(tok in rec for tok in ("contiguous", "materialize", "tensor_copy", "view_copy"))


def _canonicalize_single_atom(
    raw_suffix: str, fam: str, rec: str, meta: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    suf = strip_op_prefix(raw_suffix).lower()
    if not suf or _is_pure_glue_op(suf, fam, rec, meta):
        return ()
    scope = _atom_scope_prefix(suf)
    if "mla_prolog" in suf or "mla_prolog" in rec:
        return ("q_a", "kv_a", "q_norm", "q_b", "kv_norm", "q_rotary", "kv_rotary", "rotary", "kv_store", "q_absorb")
    if "megamoe" in rec or "dispatch_ffn_combine" in rec or "dispatch_ffn_combine" in suf:
        return ("moe_dispatch", "expert_up", "expert_act", "expert_down", "moe_combine")
    if suf == "dispatch" or fam == "dispatch":
        return ("moe_dispatch",)
    if suf in {"unpack", "combine"} or fam == "combine":
        return ("moe_combine",)
    if suf == "qkv_a":
        return ("q_a", "kv_a")
    if suf in {"q_rotary", "kv_rotary"}:
        return (suf,)
    if suf in {"rotary", "rope"}:
        return ("rotary",)
    if suf in {"kv_scatter", "kv_store", "cache"}:
        return ("kv_store",)
    if fam == "mhc" or ".hc_" in f".{suf}":
        return (f"{scope}mhc_combine",) if ("hc_post" in suf or "hc_combine" in rec) else (f"{scope}mhc_mix",)
    if "attn_res" in suf or fam == "attn_res_prefix":
        return (f"{scope}attn_res",)
    if suf in {"norm1", "input_layernorm"}:
        return ("pre_attn_norm",)
    if suf in {"norm2", "post_attention_layernorm"}:
        return ("pre_moe_norm",)
    if suf in {"quant_lightning_indexer", "index_scores"} or fam == "indexer":
        return ("index_scores",)
    if suf in {"latent_attention_batch", "mqsmla", "attention"} or fam == "attention":
        return ("core_attention",)
    if suf == "dcp_lse_merge" or fam == "attention_merge":
        return ("dcp_lse_merge",)
    if suf in {"output", "o_proj"}:
        return ("attn_output",)
    if suf == "linear_output":
        return ("linear_output",)
    if suf in {"linear_output_norm_gate", "fused_linear_output_epilogue"}:
        return ("linear_output_norm_gate",)
    if suf.startswith("recurrent_scan") or suf == "decode_recurrent_scan" or fam in {"recurrent_scan", "recurrent_vector"}:
        return ("recurrent_state_scan",)
    if "recover_ssm" in suf:
        return ("recover_ssm",)
    if suf == "short_convolution" or fam in {"short_convolution", "causal_conv1d_fn"}:
        return ("short_convolution",)
    return (suf,)


def extract_semantic_atoms(
    op_id: str,
    family: str,
    recipe: str = "",
    absorbed_source_ops: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Extract canonical unlowered semantic atoms."""
    fam = str(family or "").strip().lower()
    rec = str(recipe or "").strip().lower()
    meta = dict(metadata) if isinstance(metadata, Mapping) else {}
    atoms: list[str] = []
    for key in ("semantic_atoms", "absorbed_atoms"):
        explicit = meta.get(key)
        if isinstance(explicit, (list, tuple)):
            for item in explicit:
                tok = str(item or "").strip().lower()
                if tok and tok not in atoms:
                    atoms.append(tok)
    if atoms:
        return tuple(atoms)
    for src in absorbed_source_ops or ():
        for atom in _canonicalize_single_atom(str(src), "", ""):
            if atom not in atoms:
                atoms.append(atom)
    if fam in {"all_gather_matmul", "matmul_reduce_scatter"}:
        for atom in _canonicalize_single_atom(op_id, "gemm", rec, meta):
            if atom not in atoms:
                atoms.append(atom)
        return tuple(atoms)
    if not atoms:
        for atom in _canonicalize_single_atom(op_id, fam, rec, meta):
            if atom not in atoms:
                atoms.append(atom)
    return tuple(atoms)


def _format_tensor_contract(
    op_id: str, family: str, recipe: str, shape: Sequence[int],
    dtype: str, meta: Mapping[str, Any], case: Case, layer_plan: Any | None,
) -> str:
    """Format a readable rank-local tensor contract."""
    suffix = strip_op_prefix(op_id).lower()
    fam = str(family or "").lower()
    act_dt = _format_dtype_upper(str(meta.get("activation_format") or meta.get("input_format") or dtype or "bf16"))
    w_dt = _format_dtype_upper(str(meta.get("weight_format") or dtype or "bf16"))
    out_dt = _format_dtype_upper(str(meta.get("output_format") or dtype or "bf16"))
    dims = tuple(int(d) for d in shape)
    domain_tag = resolve_comm_domain_tag(op_id, fam, recipe, meta, case)
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
            "all_gather_matmul", "matmul_reduce_scatter",
        }
        or meta.get("program_kind") == "matmul"
    ):
        m_dim, n_dim, k_dim = dims
        if suffix in {"q_absorb", "v_expand"}:
            return f"x:[{m_dim},{k_dim}] {act_dt} × W_bmm:[{k_dim},{n_dim}] {w_dt} -> y:[{m_dim},{n_dim}] {out_dt}"
        comm_sfx = f" ({domain_tag})" if domain_tag else ""
        return f"x:[{m_dim},{k_dim}] {act_dt} × W:[{n_dim},{k_dim}] {w_dt} -> y:[{m_dim},{n_dim}] {out_dt}{comm_sfx}"
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
    if fam in COLLECTIVE_FAMILIES or domain_tag:
        tag = domain_tag or f"TP={max(1, int(case.runtime.parallel.tp))}"
        if len(dims) >= MATRIX_2D_RANK:
            return f"comm:[{','.join(str(d) for d in dims)}] {act_dt} ({tag})"
        if len(dims) == 1:
            rows = int(meta.get("graph_active_query_rows") or 0)
            if rows > 0 and dims[0] % rows == 0:
                return f"comm:[{rows},{dims[0] // rows}] {act_dt} ({tag})"
            return f"comm:[{dims[0]}] {act_dt} ({tag})"
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


def _resolve_arch_family(case: Case) -> str:
    """Resolve canonical Ascend architecture family strictly from machine schema."""
    fam = str(getattr(case.machine, "arch_family", "") or "").strip().upper()
    if fam in {"A5", "A3", "A2"}:
        return fam
    devices = getattr(case.machine, "devices", ()) or getattr(case.machine, "dies", ()) or ()
    for dev in devices:
        dev_fam = str(getattr(dev, "arch_family", "") or "").strip().upper()
        if dev_fam in {"A5", "A3", "A2"}:
            return dev_fam
        preset = str(getattr(getattr(dev, "compute_arch", None), "preset", "") or "").lower()
        if "35" in preset:
            return "A5"
        if "22" in preset:
            return "A3"
        if "20" in preset:
            return "A2"
    return fam


def _clean_native_symbol(raw: Any) -> str:
    """Reject internal simulator stage/recipe names and normalize display aliases."""
    sym = str(raw or "").strip()
    if not sym:
        return ""
    if sym in BANNED_IMPLEMENTATION_NAMES or sym in _REJECTED_SYMBOLS:
        return ""
    low = sym.lower()
    if low.startswith(_INTERNAL_SYMBOL_PREFIXES) or _INTERNAL_ARCH_TAG_RE.search(low):
        return ""
    if sym.endswith("GatedResidualMix") and not sym.startswith("FusedMHC"):
        return "FusedMHCGatedResidualMix"
    if sym.endswith("HcCombine") and not sym.startswith("FusedMHC"):
        return "FusedMHCCombine"
    return sym


def _format_source_record(raw: Any) -> str:
    if isinstance(raw, Mapping):
        repo = str(raw.get("repository") or "").strip()
        rel = str(raw.get("relative_path") or raw.get("path") or "").strip()
    else:
        repo = str(getattr(raw, "repository", "") or "").strip()
        rel = str(getattr(raw, "relative_path", "") or getattr(raw, "path", "") or "").strip()
    if not rel:
        return ""
    return f"{repo}/{rel}" if (repo and not rel.startswith(f"{repo}/")) else rel


def _format_source_ref(resolved_sources: Sequence[Any], bound_source: str, fallback_ref: str) -> str:
    """Format canonical repository/path source reference without guessed paths."""
    parts = [formatted for src in resolved_sources if (formatted := _format_source_record(src))]
    if bound_source:
        parts.append(bound_source)
    if parts:
        return "; ".join(dict.fromkeys(parts))
    return fallback_ref


def _extract_binding_identity(
    spec: OpSpec | None,
    meta: Mapping[str, Any],
    native_binding: Mapping[str, Any] | Any | None = None,
) -> tuple[str, str, str]:
    """Project (kernel_name, source_ref, backend) from NativeCallBinding / spec metadata."""
    bdict: Mapping[str, Any] = {}
    if native_binding is not None:
        bdict = native_binding.to_dict() if hasattr(native_binding, "to_dict") else (
            native_binding if isinstance(native_binding, Mapping) else {}
        )
    elif isinstance(meta.get("native_call"), Mapping):
        bdict = meta["native_call"]
    elif isinstance(meta.get("native_call_binding"), Mapping):
        bdict = meta["native_call_binding"]
    elif spec is not None:
        try:
            bound = bind_native_call(spec)
            if bound is not None:
                bdict = bound.to_dict()
        except ModelingError:
            bdict = {}

    api = bdict.get("api") if isinstance(bdict.get("api"), Mapping) else {}
    kernels = bdict.get("kernels") or ()
    first_kernel = ""
    for item in kernels:
        if isinstance(item, Mapping):
            cand = _clean_native_symbol(item.get("name"))
            if cand:
                first_kernel = cand
                break
    api_symbol = _clean_native_symbol(api.get("aclnn_symbol"))
    api_entry = _clean_native_symbol(api.get("entry"))
    meta_kind = _clean_native_symbol(meta.get("native_kind"))
    meta_sym = _clean_native_symbol(meta.get("aclnn_symbol"))
    # Prefer concise kernel/symbol identity over dotted Python module paths.
    kernel = (
        first_kernel
        or meta_kind
        or api_symbol
        or meta_sym
        or (api_entry if ("." not in api_entry) else "")
    )
    source = _format_source_record(bdict.get("source")) or _format_source_record(meta.get("native_source"))
    ns = str(api.get("namespace") or "").strip().lower()
    backend = _NAMESPACE_BACKEND_MAP.get(ns, "")
    return kernel, source, backend


def _extract_struct_native(struct_node: Mapping[str, Any] | None) -> str:
    """Read primary declared native symbol from ``export_structure`` node."""
    if not isinstance(struct_node, Mapping):
        return ""
    nat_exec = struct_node.get("native_execution")
    if isinstance(nat_exec, Mapping):
        sel = nat_exec.get("selected")
        if isinstance(sel, Mapping):
            cand = _clean_native_symbol(sel.get("native"))
            if cand:
                return cand
    impl = struct_node.get("implementation")
    if isinstance(impl, Mapping):
        for item in impl.get("native_kinds") or ():
            cand = _clean_native_symbol(item)
            if cand:
                return cand
    return ""


def _native_kernel_from_structure(
    op_id: str, spec: OpSpec | None, shape: Sequence[int], dtype: str,
) -> str:
    if spec is None:
        return ""
    op_dict = {
        "op_id": op_id,
        "family": spec.family,
        "recipe": spec.recipe,
        "metadata": dict(spec.metadata) if spec.metadata else {},
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
        for item in nat.get("native_kinds") or ():
            cand = _clean_native_symbol(item)
            if cand:
                return cand
    return ""


def _format_situ_params(meta: Mapping[str, Any], layer_plan: Any | None) -> str:
    raw_params = meta.get("activation_params")
    params_map: dict[str, Any] = {}
    if isinstance(raw_params, Mapping):
        params_map = dict(raw_params)
    elif isinstance(raw_params, (tuple, list)):
        for entry in raw_params:
            if isinstance(entry, (tuple, list)) and len(entry) == 2:
                params_map[str(entry[0])] = entry[1]
    beta = params_map.get("beta", meta.get("activation_situ_beta"))
    lin_beta = params_map.get("linear_beta", meta.get("activation_situ_linear_beta"))
    if beta is None and layer_plan is not None:
        beta = getattr(layer_plan, "activation_situ_beta", None)
    if lin_beta is None and layer_plan is not None:
        lin_beta = getattr(layer_plan, "activation_situ_linear_beta", None)
    parts: list[str] = []
    if beta is not None:
        b_val = float(beta)
        parts.append(f"beta={int(b_val) if b_val.is_integer() else b_val:g}")
    if lin_beta is not None:
        lb_val = float(lin_beta)
        parts.append(f"linear_beta={int(lb_val) if lb_val.is_integer() else lb_val:g}")
    return f"({', '.join(parts)})" if parts else ""


def _disambiguate_scoped_role(suffix: str, role: str) -> str:
    """Prefix repeated per-layer roles with their exact topological scope."""
    if role.startswith("["):
        return role
    if "attn_res" in suffix:
        for pfx, tag in (("attention.", "[Pre-Attn]"), ("ffn.", "[Pre-MoE/FFN]"), ("final.", "[Final]")):
            if suffix.startswith(pfx):
                return f"{tag} {role}"
    if ".hc_" in f".{suffix}":
        if suffix.startswith("attention."):
            return f"[Post-Attn] {role}" if "hc_post" in suffix else f"[Pre-Attn] {role}"
        if suffix.startswith("ffn."):
            return f"[Post-MoE/FFN] {role}" if "hc_post" in suffix else f"[Pre-MoE/FFN] {role}"
        if suffix.startswith("final."):
            return f"[Final] {role}"
    scoped = {
        "norm1": "[Pre-Attn]", "input_layernorm": "[Pre-Attn]",
        "norm2": "[Pre-MoE/FFN]", "post_attention_layernorm": "[Pre-MoE/FFN]",
        "attention.sp_gather": "[Attn SP]", "ffn.sp_gather": "[MoE/FFN SP]",
        "shared.sp_gather": "[Shared SP]", "sp_output_gather": "[Final SP]",
        "attention.sp_scatter": "[Attn SP]", "output_reduce.reduce_scatter": "[Attn SP]",
        "tp_attn.reduce_scatter": "[Attn SP]", "ffn.sp_scatter": "[MoE/FFN SP]",
        "tp.reduce_scatter": "[MoE/FFN SP]",
    }
    tag = scoped.get(suffix)
    return f"{tag} {role}" if tag else role


def _project_attention(
    *, suffix: str, rec: str, meta: Mapping[str, Any], bound_kernel: str,
    arch_family: str, is_c8_kv: bool, layer_plan: Any | None,
) -> tuple[str, str, str, str, bool, str, str]:
    topk = int(getattr(layer_plan, "index_topk", 0) or 0) if layer_plan is not None else 0
    layer_attn = str(getattr(layer_plan, "attention", "") or "").lower() if layer_plan is not None else ""
    is_mla = (
        layer_attn in {"mla", "mla_dsa", "dsa"}
        or "mla" in rec or "mla" in suffix or "latent" in suffix
        or bool(meta.get("paged_mla_geometry") or meta.get("indexed_attention_geometry") or meta.get("mqsmla_geometry"))
    )
    is_mqs = (
        bound_kernel in {"MixedQuantSparseFlashMla", "aclnnMixedQuantSparseFlashMla"}
        or "mixed_quant_sparse" in rec or suffix == "mqsmla" or bool(meta.get("mqsmla_geometry"))
    )
    is_sparse = (
        is_mqs or bound_kernel in {"SparseFlashMla", "aclnnSparseFlashAttention", "SparseFlashAttention"}
        or topk > 0 or "sparse" in rec or "paired_mla" in rec or bool(meta.get("indexed_attention_geometry"))
    )
    if is_mqs:
        return (
            f"Mixed-Quant Sparse Flash MLA (topk={topk or DEFAULT_SPARSE_INDEX_TOPK})",
            "MixedQuantSparseFlashMla", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "mqsmla",
        )
    if is_sparse:
        kern = bound_kernel if bound_kernel in {"SparseFlashMla", "aclnnSparseFlashAttention"} else "SparseFlashMla"
        return (
            f"Sparse Flash MLA (TopK={topk or DEFAULT_SPARSE_INDEX_TOPK})",
            kern, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "latent_attention_batch",
        )
    if arch_family == "A5" and is_mla:
        # A5 dense MLA (including DSpark draft) binds FlashMla — never FIA.
        kern = bound_kernel if (bound_kernel and "FusedInferAttentionScore" not in bound_kernel) else "FlashMla"
        return ("Flash MLA Decode Attention", kern, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix or "attention")
    if arch_family in {"A2", "A3"} and is_c8_kv:
        return (
            "Fused Infer Attention Score v3 (C8 INT8 KV)",
            "aclnnFusedInferAttentionScoreV3 (C8 INT8 KV)", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix,
        )
    kern = bound_kernel or ("FusedInferAttentionScore" if arch_family == "A5" else "aclnnFusedInferAttentionScoreV2")
    return ("Fused Infer Attention Score", kern, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)


def _project_fused_mc2(
    *, meta: Mapping[str, Any], bound_kernel: str, arch_family: str, is_int4_w: bool, is_mxfp4_w: bool,
) -> tuple[str, str, str, str, bool, str, str]:
    fused_lvl = int(meta.get("fused_mc2_level") or meta.get("enable_fused_mc2") or 0)
    if arch_family == "A5":
        kern = bound_kernel if (bound_kernel and "dispatch_ffn_combine" not in bound_kernel) else "aclnnMegaMoe"
        needs_new = bool(meta.get("abi_variant_extension")) or is_mxfp4_w
        status = (
            f"{KERNEL_STATUS_EXTEND_TILING} MXFP4 MegaMoE Compact Tiling"
            if needs_new else KERNEL_STATUS_EXISTING
        )
        spec_txt = "Fused aclnnMegaMoe supporting packed MXFP4 + gated activation compact tiling" if needs_new else ""
        return ("Fused MegaMoE (Dispatch + GMM1 + Act + GMM2 + Combine)", kern, BACKEND_CANN, status, needs_new, spec_txt, "megamoe")
    if arch_family == "A3":
        if bound_kernel in {"aclnnMegaMoe", "dispatch_ffn_combine", "dispatch_ffn_combine_w4_a8"}:
            kern = bound_kernel
        elif fused_lvl == 2:
            kern = "aclnnMegaMoe"
        else:
            kern = "dispatch_ffn_combine_w4_a8" if is_int4_w else "dispatch_ffn_combine"
        role = (
            "Fused MegaMoE (A3 Dispatch + GMM1 + Act + GMM2 + Combine)"
            if kern == "aclnnMegaMoe"
            else f"Fused MoE MC2 ({kern})"
        )
        return (role, kern, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", kern)
    return ("Fused MoE MC2", bound_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "fused_mc2")


def _project_collective(suffix: str, fam: str, rec: str, bound_kernel: str) -> tuple[str, str, str, str, bool, str, str]:
    if fam == "allgather" or "allgather" in rec:
        role = (
            "DCP Query AllGather" if "dcp_query" in suffix
            else "Sequence-Parallel AllGather" if ("sp_gather" in suffix or "sp_output" in suffix)
            else "Vocab Logits AllGather" if "logits" in suffix else f"HCCL AllGather ({suffix})"
        )
        return (role, bound_kernel or "hcom_allGather_", BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, False, "", suffix)
    if fam in {"reduce_scatter", "reducescatter"} or "reduce_scatter" in rec:
        role = "Sequence-Parallel ReduceScatter" if ("sp_scatter" in suffix or "reduce_scatter" in suffix) else f"HCCL ReduceScatter ({suffix})"
        return (role, bound_kernel or "hcom_reduceScatter_", BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, False, "", suffix)
    if fam in {"alltoall", "alltoallv"} or "alltoall" in rec:
        role = "DCP Output AllToAllV" if "dcp_output" in suffix else f"HCCL AllToAllV ({suffix})"
        return (role, bound_kernel or "hcom_allToAllV_", BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, False, "", suffix)
    return (f"TP Sum AllReduce ({suffix})", bound_kernel or "hcom_allReduce_", BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, False, "", suffix)


def _project_norm_quant(
    suffix: str, fam: str, rec: str, meta: Mapping[str, Any], bound_kernel: str,
    *, is_raw_eager: bool, is_a2_or_a3: bool,
) -> tuple[str, str, str, str, bool, str, str]:
    out_fmt = str(meta.get("output_format") or "").lower()
    is_mx = (not is_a2_or_a3) and out_fmt in {"fp8_e4m3", "mxfp8"} and str(meta.get("activation_scale_format") or "") == "fp8_e8m0"
    if fam == "layernorm":
        role = "Indexer K LayerNorm (1-kernel FP32-UB)" if "index" in suffix else "LayerNorm (1-kernel FP32-UB)"
        return (role, "aclnnLayerNorm_LayerNormV3", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if fam == "add_rmsnorm_quant":
        kern = bound_kernel or ("AddRmsNormDynamicMxQuant" if is_mx else ("aclnnAddRmsNormDynamicQuant" if is_a2_or_a3 else "AddRmsNormDynamicQuant"))
        return ("Fused Residual Add + RMSNorm + Dynamic Quant", kern, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if fam == "rmsnorm_quant":
        kern = bound_kernel or ("RmsNormDynamicMxQuant" if is_mx else ("aclnnRmsNormDynamicQuant" if is_a2_or_a3 else "RmsNormDynamicQuant"))
        return ("Fused RMSNorm + Dynamic Quant", kern, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if fam == "add_rmsnorm":
        return ("Fused Residual Add + RMSNorm", bound_kernel or "AddRmsNorm", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if fam == "quantize":
        kern = bound_kernel or ("DynamicMxQuant" if is_mx else ("aclnnDynamicQuant" if is_a2_or_a3 else "DynamicQuant"))
        return (f"Dynamic Activation Quantize ({suffix})", kern, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if is_raw_eager or "unfused" in rec or "eager" in rec:
        return (
            f"Unfused Eager {_NORM_ROLES.get(suffix, f'RMSNorm ({suffix})')} (8 kernels)",
            "Cast->Pows->ReduceMean->Adds->Rsqrt->Mul->Cast",
            BACKEND_TORCH_NPU, KERNEL_STATUS_EXISTING, False, "", suffix,
        )
    return (_NORM_ROLES.get(suffix, f"RMSNorm ({suffix})"), bound_kernel or "RmsNorm", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)


def _project_linear_attn(
    suffix: str, fam: str, rec: str, meta: Mapping[str, Any], bound_kernel: str,
    mm_kernel: str, *, is_ideal: bool, is_raw_eager: bool, layer_plan: Any | None,
) -> tuple[str, str, str, str, bool, str, str]:
    if suffix.startswith("recurrent_scan") or "recurrent_kda" in rec:
        has_spec = bool(meta.get("kda_recurrent_params", {}).get("recover_ssm")) or is_ideal
        if is_ideal:
            return (
                "Fused Recurrent KDA (Spec Verify + Persistent State)",
                bound_kernel or "flash_kda_persistent", BACKEND_CANNBOT_DSL,
                f"{KERNEL_STATUS_NEW_OR_EXTEND} Spec 2D State-Slot Writeback & Persistent UB",
                True, "Persistent cannbot-dsl / AscendC recurrent KDA with 2D state-slot writeback", "recurrent_scan",
            )
        return (
            "Fused Recurrent KDA Decode", bound_kernel or "fused_recurrent_kda", BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} Speculative KDA State-Slot Writeback" if has_spec else KERNEL_STATUS_EXISTING,
            has_spec, "Speculative 2D state-slot writeback extension for fused_recurrent_kda" if has_spec else "", "recurrent_scan",
        )
    if suffix == "decode_recurrent_scan" or fam == "recurrent_vector":
        return ("Fused Recurrent GDN Decode", bound_kernel or "RecurrentGatedDeltaRule", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "decode_recurrent_scan")
    if "recover_ssm" in suffix or "recover_ssm" in rec:
        return (
            "Speculative Accepted-Token SSM State Commit (RecoverSSM)",
            bound_kernel or "recover_ssm_state_commit", BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} Speculative Accepted-State Commit Kernel",
            True, "Compact accepted-token SSM/KDA state replay & single HBM writeback commit kernel", "recover_ssm_commit",
        )
    if suffix == "short_convolution" or fam in {"short_convolution", "causal_conv1d_fn"}:
        return (
            "Causal Conv1d + SiLU (Kernel=4)", bound_kernel or "CausalConv1d", BACKEND_CANN,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} Spec Accepted-Token Rolling Conv Update" if is_ideal else KERNEL_STATUS_EXISTING,
            is_ideal, "Speculative multi-token rolling conv_state update for CausalConv1d" if is_ideal else "", "short_convolution",
        )
    if suffix == "linear_output_norm_gate":
        if is_raw_eager or (not is_ideal and ("unfused" in rec or "eager" in rec)):
            return (
                "Unfused Eager RMSNorm + Sigmoid Gate (11 kernels)",
                "Cast->Pows->ReduceMean->Adds->Rsqrt->Mul->Cast->Mul->Sigmoid->Mul->Cast",
                BACKEND_TORCH_NPU, KERNEL_STATUS_EXISTING, False, "", "linear_output_norm_gate",
            )
        return (
            "Fused RMSNorm + Sigmoid Output Gate", bound_kernel or "FusedRMSNormGated", BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} FusedRMSNormGated(eps=1e-5, sigmoid)",
            True, "Fused RMSNorm + sigmoid gate kernel replacing eager chain", "linear_output_norm_gate",
        )
    if "l2norm" in suffix:
        return (f"Q/K L2 Normalization ({suffix})", bound_kernel or "l2norm_fwd_kernel2", BACKEND_TRITON, KERNEL_STATUS_EXISTING, False, "", suffix)
    if suffix == "delta_gates" or fam == "gdn_gating":
        return ("Delta Decay Gate Transform", bound_kernel or "fused_gdn_gating_kernel", BACKEND_TRITON, KERNEL_STATUS_EXISTING, False, "", "delta_gates")
    if fam in COLLECTIVE_FAMILIES:
        return _project_collective(suffix, fam, rec, bound_kernel)
    if fam in {"gemm", "projection"} or suffix.startswith("linear_"):
        is_gdn = suffix in {"linear_qkvz", "linear_beta_alpha"} or "gdn" in rec or (
            layer_plan is not None and str(getattr(layer_plan, "attention", "")).lower() == "gdn"
        )
        lin_roles = {
            "linear_qkvbfg_a": "KDA Fused Q/K/V/Beta/Decay-A/Gate Projection",
            "linear_decay_b": "KDA Low-Rank Decay-B Projection",
            "linear_gate_b": "KDA Low-Rank Gate-B Projection",
            "linear_output": "GDN Output Projection (O-Proj)" if is_gdn else "KDA Output Projection (O-Proj)",
            "linear_qkvz": "GDN Fused QKVZ Projection",
            "linear_beta_alpha": "GDN Beta/Alpha Gate Projection",
        }
        return (lin_roles.get(suffix, f"Linear Attention Projection ({suffix})"), bound_kernel or mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    return (f"Linear Attention Layout / Copy ({suffix})", bound_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)


def _project_moe(
    suffix: str, fam: str, rec: str, meta: Mapping[str, Any], bound_kernel: str,
    mm_kernel: str, *, arch_family: str, is_ideal: bool, is_quant_mm: bool,
    is_mxfp4_w: bool, is_int4_w: bool, layer_plan: Any | None,
) -> tuple[str, str, str, str, bool, str, str]:
    is_a2_or_a3 = arch_family in {"A2", "A3"}
    if "megamoe" in rec or "dispatch_ffn_combine" in rec:
        return _project_fused_mc2(
            meta=meta, bound_kernel=bound_kernel, arch_family=arch_family,
            is_int4_w=is_int4_w, is_mxfp4_w=is_mxfp4_w,
        )
    if suffix == "router" or fam == "router":
        role = "Shared Expert Gate Linear" if "shared" in suffix else "MoE Router Gate Linear (W_router)"
        return (role, bound_kernel or mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if suffix == "moe_gating_topk" or fam == "moe_init_routing":
        default_k = "aclnnMoeInitRoutingQuantV2" if (is_a2_or_a3 and is_quant_mm and fam == "moe_init_routing") else "MoeGatingTopK"
        return ("MoE Sigmoid + Correction Bias TopK Gating", bound_kernel or default_k, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix or "moe_gating_topk")
    if suffix in {"latent_down", "latent_up"}:
        role = "MoE Routed Latent Down Projection (H->Hm)" if suffix == "latent_down" else "MoE Routed Latent Up Projection (Hm->H)"
        return (role, bound_kernel or mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if suffix == "latent_norm":
        return ("MoE Routed Latent RMSNorm", bound_kernel or "RmsNorm", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "latent_norm")
    if suffix == "dispatch" or "dispatch" in rec:
        kern = bound_kernel if (bound_kernel and "Dispatch" in bound_kernel) else "MoeDistributeDispatchV2"
        return ("MoE EP Token Distribute Dispatch (MC2)", kern, BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, False, "", "dispatch")
    if suffix in {"unpack", "combine"} or "combine" in rec:
        kern = bound_kernel if (bound_kernel and "Combine" in bound_kernel) else "MoeDistributeCombineV2"
        return ("MoE EP Token Distribute Combine (MC2)", kern, BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, False, "", "unpack")
    if suffix in {"expert_up", "expert_down"} or fam == "grouped_gemm":
        is_up = suffix == "expert_up" or "gate_up" in rec
        if is_a2_or_a3:
            q_tag = "W4A8 INT4" if is_int4_w else ("W8A8 INT8" if is_quant_mm else "BF16")
            default_k = (
                "aclnnGroupedMatmulV4 (INT4 FRACTAL_NZ)" if is_int4_w
                else ("aclnnGroupedMatmulV4 (INT8)" if is_quant_mm else "GroupedMatmul")
            )
            return (f"Routed Expert {'Gate+Up GMM1' if is_up else 'Down GMM2'} ({q_tag})", bound_kernel or default_k, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
        q_tag = "MXFP4" if is_mxfp4_w else ("W8A8" if is_quant_mm else "BF16")
        role = f"Routed Expert {'Gate+Up GMM1' if is_up else 'Down GMM2'} ({q_tag})"
        if is_ideal and is_mxfp4_w:
            return (
                role, bound_kernel or "GroupedMatmul (MXFP4 Compact Tiling)", BACKEND_CANN,
                f"{KERNEL_STATUS_EXTEND_TILING} MXFP4 GroupedMatmul Compact Tiling", True,
                f"MXFP4 group=32 compact tiling + AIV/Cube double-buffering for {role}", suffix,
            )
        return (role, bound_kernel or "GroupedMatmul", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if suffix in {"expert_act", "shared.situ"} or fam in {"swiglu", "situ"}:
        if fam == "situ" or "situ" in rec:
            p_str = _format_situ_params(meta, layer_plan)
            role = f"{'Routed' if suffix == 'expert_act' else 'Shared'} Expert SiTU+Mul Activation {p_str}".strip()
            return (
                role, bound_kernel or "SituAndMul", BACKEND_TRITON,
                f"{KERNEL_STATUS_NEW_OP} SituAndMul{p_str} + Quant", True,
                f"Fused SituAndMul{p_str} + optional dynamic quant epilogue", suffix,
            )
        act_kern = bound_kernel or ("aclnnSwiGluQuant" if (is_a2_or_a3 and is_quant_mm) else "SwiGlu")
        return ("Routed Expert SwiGLU Activation", act_kern, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if suffix == "join":
        return ("MoE Routed + Shared Residual Add", bound_kernel or "aclnnAdd", BACKEND_CANN, KERNEL_STATUS_FRAMEWORK if is_ideal else KERNEL_STATUS_EXISTING, False, "", "join")
    if suffix.startswith("shared."):
        if fam in COLLECTIVE_FAMILIES:
            return _project_collective(suffix, fam, rec, bound_kernel)
        s_roles = {
            "shared.gate_up": "Shared Expert Gate+Up Linear (MSO Stream 1)",
            "shared.down": "Shared Expert Down Linear (MSO Stream 1)",
            "shared.swiglu": "Shared Expert SwiGLU (MSO Stream 1)",
        }
        kern = bound_kernel or ("SwiGlu" if "swiglu" in suffix else mm_kernel)
        st = f"{KERNEL_STATUS_FRAMEWORK} (MSO Stream 1 Overlap)" if is_ideal else KERNEL_STATUS_EXISTING
        return (s_roles.get(suffix, f"Shared Expert ({suffix})"), kern, BACKEND_CANN, st, False, "", suffix)
    return (f"MoE Aux ({suffix})", bound_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)


def _project_engram_mhc_indexer(
    suffix: str, fam: str, rec: str, bound_kernel: str, mm_kernel: str,
    *, is_ideal: bool, layer_plan: Any | None,
) -> tuple[str, str, str, str, bool, str, str] | None:
    if fam == "mhc" or ".hc_" in f".{suffix}":
        hc_lr = int(getattr(layer_plan, "hc_lowrank", 0) or 0) if layer_plan is not None else 0
        hc_sk = int(getattr(layer_plan, "hc_sinkhorn_iters", 0) or 0) if layer_plan is not None else 0
        is_lr = (hc_lr > 0 and hc_sk == 0) or rec in {
            "hc_gate_mix", "hc_combine", "hc_combine_rmsnorm", "hc_silu", "hc_lowrank_down_inject", "hc_lowrank_up",
        }
        if rec == "hc_silu" or suffix.endswith("hc_silu"):
            return (f"mHC Low-Rank SiLU ({suffix})", bound_kernel or "Silu", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
        if fam == "rmsnorm" or (rec.endswith("_rmsnorm") and "grouped" in rec) or suffix.endswith("hc_norm"):
            return (f"mHC Grouped RMSNorm ({suffix})", bound_kernel or "RmsNorm", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
        if fam == "gemm" or rec in {"hc_lowrank_down_inject", "hc_lowrank_up", "hc_projection", "hc_coefficients_fused"}:
            return (f"mHC Hyper-Connection Projection ({suffix})", bound_kernel or mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
        if rec in {"hc_combine", "hc_combine_rmsnorm"} or suffix.endswith("hc_post"):
            kern = bound_kernel or ("FusedMHCCombine" if is_lr else "mhc_post_combine")
            return (
                f"mHC Post Combine ({suffix})", kern, BACKEND_CANN if is_lr else BACKEND_TRITON,
                KERNEL_STATUS_NEW_OP if (is_lr and is_ideal) else KERNEL_STATUS_FRAMEWORK,
                bool(is_lr and is_ideal), "UB-resident fused FusedMHCCombine" if (is_lr and is_ideal) else "", suffix,
            )
        if is_lr:
            return (
                f"mHC Low-Rank Gated Residual Mix ({suffix})", bound_kernel or "FusedMHCGatedResidualMix", BACKEND_CANN,
                KERNEL_STATUS_NEW_OP if is_ideal else KERNEL_STATUS_FRAMEWORK,
                bool(is_ideal), "UB-resident fused FusedMHCGatedResidualMix" if is_ideal else "", suffix,
            )
        kern = bound_kernel or ("mhc_sinkhorn_mix_kernel" if ("sinkhorn" in suffix or "mix" in suffix) else "HcPre")
        return (f"mHC Hyper-Connection ({suffix})", kern, BACKEND_TRITON if fam == "mhc" else BACKEND_CANN, KERNEL_STATUS_FRAMEWORK if fam == "mhc" else KERNEL_STATUS_EXISTING, False, "", suffix)

    if any(tok in suffix or tok in fam or tok in rec for tok in ("engram", "ple_")):
        if fam in {"engram_rh2d", "h2d"} or "host_uva" in suffix or "rh2d" in rec:
            return ("PLE / Engram Host DRAM UVA H2D Gather", "aclrtMemcpyAsync (Host->Device UVA H2D)", BACKEND_CANN, KERNEL_STATUS_FRAMEWORK, False, "", suffix)
        if fam in {"engram_query", "engram_ids"} or "ngram" in suffix:
            return ("PLE / Engram N-Gram Hash ID Lookup", bound_kernel or "ple_ngram_hash_lookup", BACKEND_TORCH_NPU, KERNEL_STATUS_FRAMEWORK, False, "", suffix)
        if fam == "engram_apply" or "gate" in suffix:
            return ("PLE / Engram Gated Residual Fusion", bound_kernel or "ple_gated_fusion_kernel", BACKEND_TRITON, KERNEL_STATUS_FRAMEWORK, False, "", suffix)
        if fam in {"conv1d", "short_convolution"} or "conv" in suffix:
            return ("PLE / Engram Depthwise Short Conv1d", bound_kernel or "CausalConv1d", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
        if fam == "quantize" or suffix.endswith(".quantize"):
            return ("PLE / Engram Activation Quantize", bound_kernel or "DynamicMxQuant", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
        if fam in {"gemm", "projection"} or "proj" in suffix:
            return ("PLE / Engram KV Projection", bound_kernel or mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
        if fam in COLLECTIVE_FAMILIES or "allreduce" in suffix or "broadcast" in suffix:
            return _project_collective(suffix, fam, rec, bound_kernel)
        return (f"PLE / Engram Table Gather ({suffix})", bound_kernel or "aclnnEmbedding (GatherV2)", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)

    if _is_dsa_indexer_op(suffix, fam, rec):
        topk = int(getattr(layer_plan, "index_topk", 0) or 0) if layer_plan is not None else DEFAULT_SPARSE_INDEX_TOPK
        if suffix in {"index_scores", "quant_lightning_indexer"} or fam == "indexer":
            kern = bound_kernel or ("aclnnPoolKeyIndexer" if "pool_key" in rec else "aclnnQuantLightningIndexer")
            return (f"DSA Lightning Indexer Score (topk={topk})", kern, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "index_scores")
        if suffix == "index_topk":
            return (f"DSA Local TopK-{topk} Selection", bound_kernel or "aclnnTopk", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "index_topk")
        if suffix == "candidate_merge":
            return (f"DSA Cross-DCP TopK-{topk} Candidate Merge", bound_kernel or "aclnnTopk", BACKEND_CANN, KERNEL_STATUS_FRAMEWORK, False, "", "candidate_merge")
        if suffix == "candidate_gather":
            return ("DSA Candidate Indices AllGather (DCP)", bound_kernel or "hcom_allGather_", BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, False, "", "candidate_gather")
        if suffix == "index_store" or fam in {"indexer_quant_cache", "kv_cache_write"}:
            return ("DSA Indexer K Cache Scatter Write", bound_kernel or "aclnnScatterPaKvCache", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "index_store")
        if suffix == "index_norm":
            return ("DSA Indexer K LayerNorm", "aclnnLayerNorm_LayerNormV3", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "index_norm")
        if suffix == "index_query_quantize":
            return ("DSA Indexer Query Dynamic Quantize", bound_kernel or "DynamicQuant", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "index_query_quantize")
        if suffix in _INDEXER_PROJ_ROLES:
            return (_INDEXER_PROJ_ROLES[suffix], bound_kernel or mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
        # Unbound internal/aux indexer stages (e.g. step5_csa_summary_update) stay honest empty unless bound.
        return (f"DSA Indexer Aux ({suffix})", bound_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    return None


def _project_family_tuple(
    *, suffix: str, fam: str, rec: str, meta: Mapping[str, Any], bound_kernel: str,
    arch_family: str, is_ideal: bool, is_raw_eager: bool, is_quant_mm: bool,
    is_mxfp4_w: bool, is_int4_w: bool, is_c8_kv: bool, layer_plan: Any | None,
) -> tuple[str, str, str, str, bool, str, str]:
    is_a2_or_a3 = arch_family in {"A2", "A3"}
    mm_kernel = (
        "aclnnQuantMatmulV4 (INT8)" if (is_a2_or_a3 and is_quant_mm)
        else ("QuantBatchMatmulV3" if is_quant_mm else "MatMulV2")
    )
    if fam == "attn_res_prefix" or "attn_res" in suffix:
        return (
            "Block Attention Residual Prefix (Norm+Score+Softmax+BMM)",
            bound_kernel or "vllm_ascend_attn_res_prefix", BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} AttnRes Cross-Layer Fused Prefix", True,
            "Fused block residual norm+projection+softmax+weighted sum across block history", suffix,
        )
    if fam in {"all_gather_matmul", "matmul_reduce_scatter"}:
        is_ag = fam == "all_gather_matmul"
        return (
            f"Fused MC2 {'AllGather + MatMul' if is_ag else 'MatMul + ReduceScatter'} ({suffix})",
            bound_kernel or ("aclnnAllGatherMatmulV2" if is_ag else "aclnnMatmulReduceScatterV2"),
            BACKEND_HCCL_MC2, KERNEL_STATUS_FRAMEWORK, False, "", suffix or fam,
        )
    special = _project_engram_mhc_indexer(
        suffix, fam, rec, bound_kernel, mm_kernel, is_ideal=is_ideal, layer_plan=layer_plan,
    )
    if special is not None:
        return special
    if fam in {"rmsnorm", "add_rmsnorm", "add_rmsnorm_quant", "rmsnorm_quant", "layernorm", "quantize"}:
        return _project_norm_quant(suffix, fam, rec, meta, bound_kernel, is_raw_eager=is_raw_eager, is_a2_or_a3=is_a2_or_a3)
    if "mla_prolog" in suffix or "mla_prolog" in rec:
        return (
            "Fused MLA Prolog (Q/KV-A + RMSNorm + Q-B + W_UK Absorb + KV Cache)",
            bound_kernel or "aclnnMlaPrologV3", BACKEND_CANN,
            f"{KERNEL_STATUS_FRAMEWORK} (96->128 head pad & NoPE P^-1)", False, "", "mla_prolog_v3",
        )
    if fam == "attention":
        return _project_attention(
            suffix=suffix, rec=rec, meta=meta, bound_kernel=bound_kernel,
            arch_family=arch_family, is_c8_kv=is_c8_kv, layer_plan=layer_plan,
        )
    if fam == "attention_merge" or suffix == "dcp_lse_merge":
        return ("DCP Attention LSE Merge", bound_kernel or "aclnnAttentionUpdate", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "dcp_lse_merge")
    if _is_linear_attn_op(suffix, fam, rec, "kda" if "kda" in rec else ""):
        return _project_linear_attn(
            suffix, fam, rec, meta, bound_kernel, mm_kernel,
            is_ideal=is_ideal, is_raw_eager=is_raw_eager, layer_plan=layer_plan,
        )
    if _is_moe_router_dispatch_op(suffix, fam, rec) or _is_moe_routed_op(suffix, fam, rec) or _is_shared_expert_op(suffix, rec):
        return _project_moe(
            suffix, fam, rec, meta, bound_kernel, mm_kernel,
            arch_family=arch_family, is_ideal=is_ideal, is_quant_mm=is_quant_mm,
            is_mxfp4_w=is_mxfp4_w, is_int4_w=is_int4_w, layer_plan=layer_plan,
        )
    if fam in COLLECTIVE_FAMILIES:
        return _project_collective(suffix, fam, rec, bound_kernel)
    if fam in {"cache_store", "kv_compress_epilog", "kv_cache_write"} or suffix in {"kv_store", "kv_scatter", "cache"}:
        return ("Paged KV Cache Scatter Write", bound_kernel or "aclnnScatterPaKvCache", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", "kv_store")
    if fam == "rotary" or suffix in {"rotary", "rope", "q_rotary", "kv_rotary", "output_inverse_rotary"}:
        return (f"Rotary Position Embedding ({suffix})", bound_kernel or "InplacePartialRotaryMul", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if fam in {"swiglu", "situ"}:
        if fam == "situ":
            p_str = _format_situ_params(meta, layer_plan)
            return (
                f"SiTU Gated Activation {p_str}".strip(), bound_kernel or "SituAndMul", BACKEND_TRITON,
                f"{KERNEL_STATUS_NEW_OP} SituAndMul{p_str}", True,
                f"Fused SituAndMul{p_str} activation (non-standard gated variant)", suffix,
            )
        return ("SwiGLU Activation", bound_kernel or "SwiGlu", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if fam in {"embedding", "engram_gather"}:
        return ("Token / Engram Table Gather", bound_kernel or "aclnnEmbedding (GatherV2)", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if fam == "logits":
        role = "Draft Markov Logits Head" if "markov" in suffix else "Vocab Parallel LM Head GEMM"
        return (role, bound_kernel or mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if fam == "acceptance":
        return ("Speculative Verification & Token Acceptance", bound_kernel or "speculative_verify_accept", BACKEND_TORCH_NPU, KERNEL_STATUS_FRAMEWORK, False, "", "accept")
    if suffix in {"q_absorb", "v_expand"}:
        role = "W_UK Absorbed Q BMM" if suffix == "q_absorb" else "W_UV Value Expand BMM"
        return (role, bound_kernel or "BatchMatMulV2 (npu_transpose_batchmatmul)", BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if fam in {"gemm", "projection", "gate_up_projection", "down_projection", "qkv_projection", "output_projection"}:
        return (_PROJ_ROLES.get(suffix, f"Linear Projection ({suffix})"), bound_kernel or mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix)
    if suffix in _EAGER_VECTOR_MAP or fam in _EAGER_VECTOR_MAP:
        e_role, e_kern = _EAGER_VECTOR_MAP.get(suffix) or _EAGER_VECTOR_MAP[fam]
        return (e_role, bound_kernel or e_kern, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix or fam)
    # Logical-only or unbound aux rows keep their logical role and empty ascend_kernel (zero fabrication).
    return (f"Elementwise / Aux Op ({suffix or fam})", bound_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, False, "", suffix or fam)


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
    *,
    native_binding: Mapping[str, Any] | Any | None = None,
) -> AscendKernelDescriptor:
    """Project one architecture-aware Ascend kernel identity from producer bindings."""
    suffix = strip_op_prefix(op_id).lower()
    fam = str(family or "").lower()
    rec = str(recipe or "").lower()
    meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
    arch_family = _resolve_arch_family(case)
    track = str(dict(case.runtime.metadata).get("feature_track_report", {}).get("feature_track", "") or "")
    is_ideal = track == "expected_ideal"
    is_raw_eager = track == "raw_eager"
    is_quant_mm = str(meta.get("activation_format") or "").lower() in {"int8", "fp8", "fp8_e4m3", "fp8_e5m2"}
    w_fmt_l = str(meta.get("weight_format") or "").lower()
    is_mxfp4_w = w_fmt_l in {"mxfp4", "mx_fp4", "fp4", "fp4_e2m1"}
    is_int4_w = w_fmt_l in {"int4", "w4a8"}
    kv_raw = str(
        (getattr(case.model.quant, "kv_dtype", "") if getattr(case.model, "quant", None) is not None else "")
        or getattr(case.model, "kv_dtype", "") or ""
    ).lower()
    is_c8_kv = kv_raw in {"int8", "c8"}

    bound_kernel, bound_source, bound_backend = _extract_binding_identity(spec, meta, native_binding)
    struct_nat = _extract_struct_native(struct_node)
    reg_spec = _lookup_registry_spec(fam, rec) if (not bound_kernel and not struct_nat) else None
    reg_nat = _clean_native_symbol(reg_spec.native_kinds[0]) if (reg_spec and reg_spec.native_kinds) else ""
    fallback_src = ""
    if isinstance(struct_node, Mapping) and struct_node.get("source_ref"):
        fallback_src = str(struct_node["source_ref"]).strip()
    elif reg_spec is not None and reg_spec.source_ref:
        fallback_src = str(reg_spec.source_ref).strip()

    effective_bound = bound_kernel or struct_nat or reg_nat or _native_kernel_from_structure(op_id, spec, shape, dtype)
    # Cross-generation guard: never let legacy/generic structure symbols override A5 MLA or A5 MegaMoE.
    if arch_family == "A5":
        if "dispatch_ffn_combine" in effective_bound:
            effective_bound = ""
        if fam == "attention" and "FusedInferAttentionScore" in effective_bound:
            layer_attn = str(getattr(layer_plan, "attention", "") or "").lower() if layer_plan is not None else ""
            if (layer_attn in {"mla", "mla_dsa", "dsa"} or "mla" in rec or "mla" in suffix
                    or "latent" in suffix or meta.get("paged_mla_geometry") or meta.get("indexed_attention_geometry")):
                effective_bound = ""

    contract_txt = _format_tensor_contract(op_id, fam, rec, shape, dtype, meta, case, layer_plan)
    role, kernel, backend, status, needs_new, new_spec, canon_key = _project_family_tuple(
        suffix=suffix, fam=fam, rec=rec, meta=meta, bound_kernel=effective_bound,
        arch_family=arch_family, is_ideal=is_ideal, is_raw_eager=is_raw_eager,
        is_quant_mm=is_quant_mm, is_mxfp4_w=is_mxfp4_w, is_int4_w=is_int4_w,
        is_c8_kv=is_c8_kv, layer_plan=layer_plan,
    )
    if bound_backend and not needs_new and not (is_raw_eager and fam == "rmsnorm"):
        backend = bound_backend
    if isinstance(struct_node, Mapping):
        raw_backend = str(struct_node.get("backend_stack") or "").strip().lower()
        if raw_backend in STRUCTURE_BACKEND_MAP and not bound_backend and not (is_raw_eager and fam == "rmsnorm"):
            backend = STRUCTURE_BACKEND_MAP[raw_backend]
        raw_readiness = str(struct_node.get("readiness") or "").strip().lower()
        if raw_readiness in STRUCTURE_READINESS_MAP:
            mapped_status, mapped_new = STRUCTURE_READINESS_MAP[raw_readiness]
            if mapped_new or not needs_new:
                status = mapped_status if not new_spec else f"{mapped_status} {new_spec}"
                needs_new = mapped_new
    elif reg_spec is not None and reg_spec.backend_stack in STRUCTURE_BACKEND_MAP and not needs_new and not bound_backend:
        backend = STRUCTURE_BACKEND_MAP[reg_spec.backend_stack]

    if isinstance(subgraph_info, Mapping):
        f_status = str(subgraph_info.get("fusion_status") or "").strip().lower()
        if f_status == "fused_new_kernel_required" and not needs_new:
            reason = str(subgraph_info.get("reason") or "").strip()
            status = f"{KERNEL_STATUS_NEW_OP} {reason}".strip()
            needs_new = True
            new_spec = new_spec or reason
        elif f_status == "fused_framework_enable" and status == KERNEL_STATUS_EXISTING:
            status = KERNEL_STATUS_FRAMEWORK

    kernel = _clean_native_symbol(kernel)
    return AscendKernelDescriptor(
        logical_role=_disambiguate_scoped_role(suffix, role),
        ascend_kernel=kernel,
        kernel_backend=backend,
        kernel_status=status,
        source_ref=_format_source_ref(resolved_sources, bound_source, fallback_src),
        tensor_contract=contract_txt,
        needs_new_kernel=needs_new,
        new_kernel_spec=new_spec,
        canonical_op_key=canon_key,
    )
