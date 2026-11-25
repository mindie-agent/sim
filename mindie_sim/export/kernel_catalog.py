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

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.catalog.quant import projection_quant
from mindie_sim.schema.abi import AbiQueryKey, AbiSupportVerdict, ProbedAbiEntry
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import KernelBinding, OpSpec
from mindie_sim.structure_implementations import operator_birth_certificate

from .models import (
    ATTN_4D_RANK,
    ATTN_5D_RANK,
    BACKEND_ASCENDC,
    BACKEND_CANN,
    BACKEND_CATLASS,
    BACKEND_HCCL_MC2,
    BACKEND_TILELANG,
    BACKEND_TORCH_NPU,
    BACKEND_TRITON,
    BANNED_IMPLEMENTATION_NAMES,
    COLLECTIVE_FAMILIES,
    GEMM_SHAPE_RANK,
    KERNEL_STATUS_EXISTING,
    KERNEL_STATUS_EXTEND_TILING,
    KERNEL_STATUS_FRAMEWORK,
    KERNEL_STATUS_NEW_OP,
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
    MAX_SOURCE_REF_SUMMARY_LEN,
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
_MXFP4_DTYPE_TOKENS = frozenset({"mxfp4", "mx_fp4", "mxfp4_e2m1", "fp4", "fp4_e2m1"})
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
    if fam in {"grouped_gemm", "grouped_moe"}:
        return True
    if suffix in {
        "expert_up", "expert_act", "expert_down", "expert_down.quantize",
        "unpack", "combine", "latent_norm", "latent_up", "join", "route_unpack",
    }:
        return True
    return any(tok in rec for tok in (
        "grouped_gate_up", "grouped_down", "grouped_situ",
        "grouped_swiglu", "megamoe", "dispatch_ffn_combine", "expert_add", "route_unpack",
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
    tok = str(raw or "").strip().lower()
    if not tok:
        raise ModelingError("missing_operator_dtype_for_display")
    return _DTYPE_UPPER_MAP.get(tok, tok.upper())


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
    return False


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


def _resolve_expert_weight_format(
    spec: OpSpec | None,
    meta: Mapping[str, Any],
    case: Case,
    layer_plan: Any | None,
    *,
    allow_model_fallback: bool = True,
) -> str:
    """Resolve expert weight precision format from ABI, metadata, or layer quant."""
    abi = getattr(spec, "precision_abi", None) if spec is not None else None
    if abi is not None:
        w_dt = str(getattr(abi, "weight_dtype", "") or "").strip().lower()
        if w_dt:
            return w_dt
        lin_m = str(getattr(abi, "linear_method", "") or "").strip().lower()
        if lin_m in _MXFP4_DTYPE_TOKENS:
            return "mxfp4"
    meta_w = str(meta.get("weight_format") or meta.get("weight_dtype") or "").strip().lower()
    if meta_w:
        return meta_w
    if not allow_model_fallback:
        return ""
    model_quant = getattr(case.model, "quant", None)
    if model_quant is not None:
        pq = projection_quant(model_quant, layer_plan) if layer_plan is not None else model_quant
        exp_w = str(getattr(pq, "expert_weight_format", "") or "").strip().lower()
        if exp_w:
            return exp_w
    return ""


def _is_expert_matrix_op(op_id: str, family: str, recipe: str) -> bool:
    suffix = strip_op_prefix(op_id).lower()
    fam = str(family or "").strip().lower()
    rec = str(recipe or "").strip().lower()
    return (
        fam in {"grouped_gemm", "grouped_moe"}
        or suffix in {"expert_up", "expert_down"}
        or "megamoe" in rec
        or "dispatch_ffn_combine" in rec
    )


def resolve_operator_display_dtype(
    op_id: str,
    spec: OpSpec | None,
    family: str,
    recipe: str,
    raw_dtype: str,
    case: Case,
    layer_plan: Any | None = None,
    *,
    absorbed_source_ops: Sequence[str] = (),
    op_spec_by_key: Mapping[tuple[int, str], OpSpec] | None = None,
) -> str:
    """Resolve public display ``dtype`` surfacing ``mxfp4`` on MXFP4 expert operators."""
    base_dt = str(raw_dtype or "").strip()
    if not base_dt:
        raise ModelingError(f"missing_operator_display_dtype:{op_id}")
    meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
    direct_w = _resolve_expert_weight_format(
        spec, meta, case, layer_plan, allow_model_fallback=False,
    )
    if direct_w in _MXFP4_DTYPE_TOKENS:
        return "mxfp4"
    if _is_expert_matrix_op(op_id, family, recipe):
        if _resolve_expert_weight_format(spec, meta, case, layer_plan, allow_model_fallback=True) in _MXFP4_DTYPE_TOKENS:
            return "mxfp4"
        rec_low = str(recipe or "").strip().lower()
        if "megamoe" in rec_low or "dispatch_ffn_combine" in rec_low:
            mega_geo = meta.get("megamoe_geometry") if isinstance(meta.get("megamoe_geometry"), Mapping) else {}
            comp_dt = str(
                meta.get("compute_dtype")
                or mega_geo.get("compute_dtype")
                or meta.get("compute_format")
                or ""
            ).strip().lower()
            if comp_dt in {"fp8_e4m3", "fp8_e5m2", "int8", "fp8", "mxfp8"}:
                return comp_dt
    if absorbed_source_ops and op_spec_by_key:
        for src_id in absorbed_source_ops:
            src_spec = op_spec_by_key.get((0, str(src_id)))
            if src_spec is None:
                continue
            src_meta = dict(src_spec.metadata) if src_spec.metadata else {}
            src_is_exp = _is_expert_matrix_op(src_spec.op_id, src_spec.family, src_spec.recipe)
            src_w = _resolve_expert_weight_format(
                src_spec, src_meta, case, layer_plan, allow_model_fallback=src_is_exp,
            )
            if src_w in _MXFP4_DTYPE_TOKENS:
                return "mxfp4"
    return base_dt


def _format_tensor_contract(
    op_id: str, family: str, recipe: str, shape: Sequence[int],
    dtype: str, meta: Mapping[str, Any], case: Case, layer_plan: Any | None,
) -> str:
    """Format a readable rank-local tensor contract."""
    suffix = strip_op_prefix(op_id).lower()
    fam = str(family or "").lower()
    rec = str(recipe or "").lower()
    act_dt = _format_dtype_upper(str(meta.get("activation_format") or meta.get("input_format") or dtype or ""))
    w_dt = _format_dtype_upper(str(meta.get("weight_format") or dtype or ""))
    out_dt = _format_dtype_upper(str(meta.get("output_format") or dtype or ""))
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
    if "megamoe" in rec or "dispatch_ffn_combine" in rec:
        exp_w = _resolve_expert_weight_format(None, meta, case, layer_plan)
        w_label = _format_dtype_upper(exp_w or dtype or "")
        act_label = _format_dtype_upper(
            str(meta.get("activation_format") or meta.get("input_format") or meta.get("compute_format") or dtype or "")
        )
        tag = domain_tag or f"EP={max(1, int(case.runtime.parallel.ep))}"
        sh_txt = ",".join(str(d) for d in dims) if dims else "tokens"
        return f"x:[{sh_txt}] {act_label} × W:{w_label} -> y:[{sh_txt}] {out_dt} ({tag})"
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
        arch_fam = str(getattr(getattr(dev, "compute_arch", None), "arch_family", "") or "").strip().upper()
        if arch_fam in {"A5", "A3", "A2"}:
            return arch_fam
    raise ModelingError("missing_structured_arch_family")



_RETIRED_DSL_TOKEN = "cann" + "bot"
_CANONICAL_BACKEND_MAP: dict[str, str] = {
    "cann": BACKEND_CANN,
    "aclnn": BACKEND_CANN,
    "cann_aclnn": BACKEND_CANN,
    "cann aclnn/ascendc": BACKEND_CANN,
    "ascendc": BACKEND_ASCENDC,
    "vllm_ascend_csrc": BACKEND_ASCENDC,
    "tilelang": BACKEND_TILELANG,
    "catlass": BACKEND_CATLASS,
    "triton": BACKEND_TRITON,
    "triton_ascend": BACKEND_TRITON,
    "triton-ascend": BACKEND_TRITON,
    "torch_npu": BACKEND_TORCH_NPU,
    "torch": BACKEND_TORCH_NPU,
    "pytorch-npu": BACKEND_TORCH_NPU,
    "torch_npu custom op": BACKEND_TORCH_NPU,
    "third_party_wheel": BACKEND_TORCH_NPU,
    "host": BACKEND_TORCH_NPU,
    "hccl": BACKEND_HCCL_MC2,
    "mc2": BACKEND_HCCL_MC2,
    "hccl/mc2": BACKEND_HCCL_MC2,
    "hccl / mc2": BACKEND_HCCL_MC2,
    "hccl_mc2": BACKEND_HCCL_MC2,
    "custom-ops": BACKEND_ASCENDC,
    "custom_ops": BACKEND_ASCENDC,
    "view_reshape": BACKEND_CANN,
    "logical": BACKEND_CANN,
}
_SOURCE_REPO_BACKEND_MAP: dict[str, str] = {
    "ascend/tilelang-ascend": BACKEND_TILELANG,
    "ascend/catlass": BACKEND_CATLASS,
    "ascend/triton-ascend": BACKEND_TRITON,
}
_BIRTH_READINESS_MAP: dict[str, tuple[str, bool]] = {
    "shipping": (KERNEL_STATUS_EXISTING, False),
    "proposed_pr": (KERNEL_STATUS_FRAMEWORK, False),
    "synthesized_target": (KERNEL_STATUS_NEW_OP, True),
    "tiler_extension": (KERNEL_STATUS_EXTEND_TILING, True),
}


def _clean_native_symbol(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text or text in BANNED_IMPLEMENTATION_NAMES:
        return ""
    if _RETIRED_DSL_TOKEN in text.lower():
        raise ModelingError(f"banned_legacy_dsl_kernel_symbol:{text!r}")
    return text


def _map_canonical_backend(raw: Any, *, family: str = "") -> str:
    tok = str(raw or "").strip().lower()
    if tok:
        if _RETIRED_DSL_TOKEN in tok:
            raise ModelingError(f"banned_legacy_dsl_backend:{raw!r}")
        mapped = _CANONICAL_BACKEND_MAP.get(tok)
        if mapped:
            return mapped
    if str(family or "").strip().lower() in COLLECTIVE_FAMILIES:
        return BACKEND_HCCL_MC2
    return BACKEND_CANN


def _format_source_entry(src: Any) -> str:
    if src is None:
        return ""
    if isinstance(src, str):
        return src.strip()
    if isinstance(src, Mapping):
        repo = str(src.get("repository") or "").strip()
        rel = str(src.get("relative_path") or src.get("path") or "").strip()
        sym = str(src.get("symbol") or "").strip()
        if repo and rel:
            return f"{repo}:{rel}" + (f"#{sym}" if sym else "")
        return rel or repo
    repo = str(getattr(src, "repository", "") or "").strip()
    rel = str(getattr(src, "relative_path", "") or getattr(src, "path", "") or "").strip()
    if repo and rel:
        return f"{repo}:{rel}"
    return rel or repo


def _collect_source_ref(
    resolved_sources: Sequence[Any],
    *explicit_sources: Any,
) -> str:
    seen: list[str] = []
    for item in (*explicit_sources, *(resolved_sources or ())):
        txt = _format_source_entry(item)
        if txt and txt not in seen:
            seen.append(txt)
    joined = "; ".join(seen)
    ellipsis = "..."
    if len(joined) > MAX_SOURCE_REF_SUMMARY_LEN:
        return joined[: MAX_SOURCE_REF_SUMMARY_LEN - len(ellipsis)] + ellipsis
    return joined


def _extract_native_binding_fields(raw_binding: Any) -> tuple[str, str, str]:
    """Read ``(symbol, backend_token, source_ref)`` strictly from a structured native call."""
    if raw_binding is None:
        return "", "", ""
    if isinstance(raw_binding, Mapping):
        api = raw_binding.get("api") if isinstance(raw_binding.get("api"), Mapping) else {}
        sym = (
            _clean_native_symbol(api.get("aclnn_symbol"))
            or _clean_native_symbol(api.get("entry"))
            or _clean_native_symbol(raw_binding.get("aclnn_symbol"))
            or _clean_native_symbol(raw_binding.get("entry"))
            or _clean_native_symbol(raw_binding.get("symbol"))
            or _clean_native_symbol(raw_binding.get("kernel_id"))
        )
        if not sym:
            for item in raw_binding.get("kernels") or ():
                if isinstance(item, Mapping):
                    sym = _clean_native_symbol(item.get("name") or item.get("symbol"))
                    if sym:
                        break
        if not sym:
            for cand in api.get("candidates") or ():
                sym = _clean_native_symbol(cand)
                if sym:
                    break
        backend = str(api.get("namespace") or raw_binding.get("backend") or "").strip()
        source = _format_source_entry(raw_binding.get("source"))
        return sym, backend, source
    sym = (
        _clean_native_symbol(getattr(raw_binding, "aclnn_symbol", ""))
        or _clean_native_symbol(getattr(raw_binding, "entry", ""))
        or _clean_native_symbol(getattr(raw_binding, "symbol", ""))
        or _clean_native_symbol(getattr(raw_binding, "kernel_id", ""))
    )
    backend = str(getattr(raw_binding, "namespace", "") or getattr(raw_binding, "backend", "") or "").strip()
    source = _format_source_entry(getattr(raw_binding, "source", None))
    return sym, backend, source


def _resolve_binding_origin(case: Case, op_id: str, family: str, recipe: str) -> str:
    rt_meta = dict(case.runtime.metadata) if case.runtime.metadata else {}
    raw_origins = rt_meta.get("recipe_binding_origins")
    if isinstance(raw_origins, Mapping):
        origins = {str(k): str(v) for k, v in raw_origins.items()}
    elif isinstance(raw_origins, (list, tuple)):
        origins = {
            str(pair[0]): str(pair[1])
            for pair in raw_origins
            if isinstance(pair, (list, tuple)) and len(pair) == MATRIX_2D_RANK
        }
    else:
        return ""
    suffix = strip_op_prefix(op_id)
    for key in (op_id, suffix, f"{family}:{recipe}", recipe, family, "stack"):
        if key and key in origins:
            return origins[key]
    return ""


def _lookup_probed_abi_entry(
    symbol: str, arch_family: str, track: str,
) -> ProbedAbiEntry | None:
    if not symbol:
        return None
    registry = default_abi_registry()
    candidates = list(registry.lookup(symbol, arch_family, available_only=False))
    if not candidates and not symbol.startswith("aclnn"):
        candidates = list(registry.lookup(f"aclnn{symbol}", arch_family, available_only=False))
    if not candidates:
        snap = registry.get_snapshot(arch_family)
        candidates = [e for e in snap.entries if symbol in e.device_kernel_expansion]
    if not candidates:
        return None
    allow_unmerged = track in {"proposed", "target"}
    for entry in candidates:
        if entry.available_on_device and entry.vllm_ascend_track == "merged":
            return entry
    if allow_unmerged:
        for entry in candidates:
            if entry.available_on_device:
                return entry
    return candidates[0]


def _extract_struct_provenance(
    op_id: str,
    spec: OpSpec | None,
    family: str,
    recipe: str,
    shape: Sequence[int],
    dtype: str,
    struct_node: Mapping[str, Any] | None,
) -> tuple[str, str, str, str, str]:
    """Return ``(symbol, backend, readiness, source_ref, role)`` from structure metadata."""
    sym = backend = readiness = source_ref = role = ""
    if isinstance(struct_node, Mapping):
        impl = struct_node.get("implementation")
        if isinstance(impl, Mapping):
            for kind in impl.get("native_kinds") or ():
                sym = _clean_native_symbol(kind)
                if sym:
                    break
        if not sym:
            for cand in struct_node.get("native_candidates") or ():
                if isinstance(cand, Mapping) and cand.get("selected"):
                    sym = _clean_native_symbol(cand.get("native"))
                    if sym:
                        break
        backend = str(struct_node.get("backend_stack") or "").strip()
        readiness = str(struct_node.get("readiness") or "").strip().lower()
        source_ref = str(struct_node.get("source_ref") or "").strip()
        role = str(struct_node.get("submodule") or "").strip()
        if sym:
            return sym, backend, readiness, source_ref, role
    if not shape:
        return sym, backend, readiness, source_ref, role
    op_dict = {
        "op_id": op_id,
        "family": family,
        "recipe": recipe,
        "dtype": str(dtype or (spec.dtype if spec is not None else "") or "bf16"),
        "local_shape": tuple(int(d) for d in shape),
        "metadata": dict(spec.metadata) if (spec is not None and spec.metadata) else {},
    }
    cert = operator_birth_certificate(op_dict, strict=False)
    if cert.readiness == "unmodeled":
        return sym, backend, readiness, source_ref, role
    cert_sym = _clean_native_symbol(cert.native_kinds[0]) if cert.native_kinds else ""
    return (
        sym or cert_sym,
        backend or str(cert.backend_stack or ""),
        readiness or str(cert.readiness or ""),
        source_ref or str(cert.source_ref or ""),
        role or str(cert.submodule or ""),
    )


def _project_from_abi_verdict(
    verdict: AbiSupportVerdict,
    *,
    logical_role: str,
    tensor_contract: str,
    canonical_op_key: str,
    binding_origin: str,
    resolved_sources: Sequence[Any],
    extra_sources: Sequence[Any],
) -> AscendKernelDescriptor:
    admitted = verdict.status in {"admitted_shipping", "admitted_proposed_pr"}
    is_shipping = verdict.status == "admitted_shipping" and verdict.available_on_device
    if is_shipping:
        status, needs_new, new_spec = KERNEL_STATUS_EXISTING, False, ""
    elif verdict.status == "admitted_proposed_pr":
        status, needs_new, new_spec = KERNEL_STATUS_FRAMEWORK, False, verdict.proposed_pr_ref
    else:
        reason = verdict.rejection_reason or f"abi_rejected:{verdict.abi_id}"
        status, needs_new, new_spec = f"{KERNEL_STATUS_NEW_OP} ({reason})", True, reason
    kernel = _clean_native_symbol(verdict.symbol_or_kernel) if admitted else ""
    backend = _map_canonical_backend(verdict.backend_kind)
    src = _collect_source_ref(resolved_sources, verdict.vllm_ascend_ref, verdict.source_ref, *extra_sources)
    return AscendKernelDescriptor(
        logical_role=logical_role,
        ascend_kernel=kernel,
        kernel_backend=backend,
        kernel_status=status,
        source_ref=src,
        tensor_contract=tensor_contract,
        needs_new_kernel=needs_new,
        new_kernel_spec=new_spec,
        canonical_op_key=canonical_op_key,
        is_shipping_abi=is_shipping,
        abi_entry_id=verdict.matched_entry_id,
        binding_origin=binding_origin,
    )


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
    """Pure read-only projection from KernelBinding / NativeCall / ProbedAbiRegistry / binding_origins."""
    arch_family = _resolve_arch_family(case)
    meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
    fam = str(family or "").strip()
    rec = str(recipe or "").strip()
    suffix = strip_op_prefix(op_id)
    rt_meta = dict(case.runtime.metadata) if case.runtime.metadata else {}
    track = str(rt_meta.get("feature_track") or "shipping").strip().lower()
    binding_origin = _resolve_binding_origin(case, op_id, fam, rec)
    contract_txt = _format_tensor_contract(op_id, fam, rec, shape, dtype, meta, case, layer_plan)
    canon_key = f"{fam or 'op'}:{rec or suffix}"

    kb = getattr(spec, "kernel_binding", None) if spec is not None else None
    has_birth = isinstance(kb, KernelBinding)
    birth_sym = _clean_native_symbol(kb.kernel_id) if has_birth else ""
    birth_backend = str(kb.backend or "").strip() if has_birth else ""
    if has_birth and birth_backend.lower() in {"", "custom-ops", "custom_ops"} and kb.source_ref is not None:
        repo_key = str(getattr(kb.source_ref, "repository", "") or "").strip().lower()
        birth_backend = _SOURCE_REPO_BACKEND_MAP.get(repo_key, birth_backend)
    birth_readiness = str(kb.readiness or "").strip().lower() if has_birth else ""
    birth_source = _format_source_entry(kb.source_ref) if has_birth else ""
    birth_role = str(kb.submodule or "").strip() if has_birth else ""

    nb_obj = (
        native_binding
        or meta.get("native_call_binding")
        or meta.get("native_call")
        or (struct_node.get("native_call") if isinstance(struct_node, Mapping) else None)
    )
    nb_sym, nb_backend, nb_source = _extract_native_binding_fields(nb_obj)
    meta_impl = meta.get("implementation_binding") if isinstance(meta.get("implementation_binding"), Mapping) else {}
    meta_api = meta.get("native_api") if isinstance(meta.get("native_api"), Mapping) else {}
    meta_sym = (
        (_clean_native_symbol(meta.get("fused_kernel_name")) if bool(meta.get("target_rewrite_fused")) else "")
        or _clean_native_symbol(meta.get("aclnn_symbol"))
        or _clean_native_symbol(meta.get("hccl_entry"))
        or _clean_native_symbol(meta_api.get("aclnn_symbol") or meta_api.get("entry"))
        or _clean_native_symbol(meta_impl.get("implementation"))
        or _clean_native_symbol(meta.get("native_kind"))
    )
    meta_source = _format_source_entry(meta_impl.get("source") or meta.get("source_reference"))
    st_sym, st_backend, st_readiness, st_source, st_role = _extract_struct_provenance(
        op_id, spec, fam, rec, shape, dtype, struct_node,
    )
    role = birth_role or st_role or (f"{fam}:{rec}" if rec else (fam or suffix))

    verdict = meta.get("abi_verdict")
    if not isinstance(verdict, AbiSupportVerdict) and isinstance(meta.get("abi_query"), AbiQueryKey):
        verdict = default_abi_registry().verify(meta["abi_query"])
    if isinstance(verdict, AbiSupportVerdict):
        return _project_from_abi_verdict(
            verdict,
            logical_role=role,
            tensor_contract=contract_txt,
            canonical_op_key=canon_key,
            binding_origin=binding_origin,
            resolved_sources=resolved_sources,
            extra_sources=(birth_source, nb_source, meta_source, st_source),
        )

    is_target_fused = bool(meta.get("target_rewrite_fused"))
    declared_sym = birth_sym or nb_sym or meta_sym or st_sym
    abi_entry = (
        None
        if (is_target_fused or (has_birth and birth_readiness in {"synthesized_target", "tiler_extension"}))
        else _lookup_probed_abi_entry(declared_sym, arch_family, track)
    )

    is_shipping_abi = False
    abi_entry_id = ""
    abi_source = ""
    new_spec = ""
    status = KERNEL_STATUS_EXISTING
    needs_new = False
    kernel = declared_sym
    backend_raw = birth_backend or nb_backend or str(meta_api.get("namespace") or "") or st_backend

    if abi_entry is not None:
        abi_entry_id = abi_entry.entry_id
        abi_source = abi_entry.vllm_ascend_ref or _format_source_entry(abi_entry.source_ref)
        if abi_entry.available_on_device:
            kernel = _clean_native_symbol(abi_entry.symbol_or_kernel) or declared_sym
            backend_raw = abi_entry.backend_kind
            is_shipping_abi = abi_entry.vllm_ascend_track == "merged"
            if not is_shipping_abi:
                status = KERNEL_STATUS_FRAMEWORK
        else:
            kernel = ""
            backend_raw = abi_entry.backend_kind
            needs_new = True
            new_spec = abi_entry.probe_domain_note or f"unavailable_on_{arch_family}:{abi_entry.abi_id}"
            status = f"{KERNEL_STATUS_NEW_OP} ({new_spec})"
    elif has_birth and birth_readiness in _BIRTH_READINESS_MAP:
        status, needs_new = _BIRTH_READINESS_MAP[birth_readiness]
    elif st_readiness in STRUCTURE_READINESS_MAP:
        status, needs_new = STRUCTURE_READINESS_MAP[st_readiness]

    if is_target_fused:
        is_shipping_abi = False
        needs_new = True
        status = KERNEL_STATUS_NEW_OP
        new_spec = new_spec or str(meta.get("fusion_reason") or meta.get("fused_kernel_name") or "").strip()

    if isinstance(subgraph_info, Mapping):
        f_status = str(subgraph_info.get("fusion_status") or "").strip().lower()
        if f_status == "fused_new_kernel_required" and not needs_new:
            reason = str(subgraph_info.get("reason") or "").strip()
            needs_new = True
            is_shipping_abi = False
            new_spec = new_spec or reason
            status = f"{KERNEL_STATUS_NEW_OP} {reason}".strip()
        elif f_status == "fused_framework_enable" and status == KERNEL_STATUS_EXISTING:
            status = KERNEL_STATUS_FRAMEWORK

    return AscendKernelDescriptor(
        logical_role=role,
        ascend_kernel=kernel,
        kernel_backend=_map_canonical_backend(backend_raw, family=fam),
        kernel_status=status,
        source_ref=_collect_source_ref(resolved_sources, birth_source, abi_source, nb_source, meta_source, st_source),
        tensor_contract=contract_txt,
        needs_new_kernel=needs_new,
        new_kernel_spec=new_spec,
        canonical_op_key=canon_key,
        is_shipping_abi=is_shipping_abi,
        abi_entry_id=abi_entry_id,
        binding_origin=binding_origin,
    )
