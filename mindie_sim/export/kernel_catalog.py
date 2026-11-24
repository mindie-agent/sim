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

from mindie_sim.catalog.quant import projection_quant
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

from .models import (
    ATTN_4D_RANK,
    ATTN_5D_RANK,
    COLLECTIVE_FAMILIES,
    GEMM_SHAPE_RANK,
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
    """Unplugged socket for pure read-only Ascend kernel identity projection."""
    raise NotImplementedError(
        "LEGACY_PURGED: heuristic string-sniffing kernel_catalog fabricator deleted; "
        "project strictly read-only from OpOccurrence.native_binding / "
        "OpSpec.metadata['native_call'] + ProbedAbiRegistry.verify() + "
        "StackLoweringPlan.binding_origins (fail-fast on missing provenance)."
    )
