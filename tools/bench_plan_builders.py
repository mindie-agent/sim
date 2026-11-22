"""Strict 1-to-1 operator builder resolver for subgraph benchmark plans.

[Role]
  Map every non-control ``(family, recipe, implementation)`` triple emitted by
  ``mindie_sim`` structure export to its canonical device-side benchmark builder
  and upstream CANN / Triton ABI, validate physical fabric routes, and resolve
  complete collective rank cosets without silent exception swallowing or rank
  truncation.
[Invariants]
  * Control-only nodes (``barrier``, ``event``, ``hccl_boundary``) map strictly
    to ``kind="boundary"`` and never masquerade as device kernels.
  * Every non-control operator MUST match an explicit semantic rule; unknown
    triples raise ``UnmodeledCapabilityError`` carrying a structured
    ``mindie-unmodeled-structure-hint/1`` JSON payload.
  * ``fabric_route_ranks`` rejects unknown routes, invalid participant counts,
    truncated intra-package/intra-node sets, and unavailable physical ranks
    with ``BenchGeometryError``.
  * ``resolve_collective_cosets`` preserves all cosets for the axis and raises
    ``BenchTopologyError`` on invalid/unknown topology axes instead of
    swallowing exceptions.
[Timeouts & Locks]
  Pure in-memory dispatch; O(1) per operator, no I/O or locks.
[Side-Effect Policy]
  Stateless pure functions; never imports device runtimes or mutates inputs.
"""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from mindie_sim.public.errors import UnmodeledCapabilityError

ONBOARDING_HINT_SCHEMA = "mindie-unmodeled-structure-hint/1"
CONTROL_FAMILIES = frozenset({"barrier", "event", "hccl_boundary"})
BOUNDARY_BUILDER = "graph_boundary"
SUPPORTED_FABRIC_ROUTES = frozenset({
    "intra_package_die", "intra_node_hccs", "inter_node_roce",
})


class BenchTopologyError(ValueError):
    """Structured error raised when collective topology resolution fails."""


class BenchGeometryError(ValueError):
    """Structured error raised when a fabric route cannot be physically realized."""


COLLECTIVE_BUILDERS = frozenset({
    "all_reduce", "quantized_all_reduce", "all_gather", "reduce_scatter",
    "all_to_all", "all_to_all_v", "mc2_dispatch", "mc2_combine",
    "mc2_dispatch_v3", "mc2_combine_v3", "megamoe_fused_mc2", "p2p_send_recv",
})

BUILDERS = frozenset({
    *COLLECTIVE_BUILDERS, BOUNDARY_BUILDER,
    "all_gather_matmul", "matmul_reduce_scatter",
    "add_rms_norm_quant", "rms_norm_quant", "add_rms_norm", "gemma_rms_norm",
    "rms_norm", "layer_norm", "swiglu_oai", "swiglu", "geglu", "gelu",
    "situ_activation", "silu_mul", "sigmoid", "softmax",
    "dequant_swiglu_quant", "swiglu_dynamic_quant", "dequant_situ_quant",
    "dynamic_quant", "grouped_matmul_swiglu_quant", "grouped_matmul",
    "mla_prolog_v3", "mla_prolog_v2", "mla_kv_compressor",
    "dcp_lse_attention_merge", "mixed_quant_sparse_mla",
    "sparse_flash_attention", "absorbed_mla", "block_sparse_attention_c8",
    "incre_flash_attention", "fusion_attention", "fused_infer_attention",
    "pool_key_indexer", "lightning_indexer", "indexer_quant_cache",
    "key_pool", "kv_compress_epilog", "histogram_v2",
    "scatter_pa_kv_cache", "scatter_index_cache",
    "moe_gating_topk_hash", "moe_gating_topk", "indexer_topk",
    "moe_init_routing", "moe_token_permute", "strided_gather",
    "moe_token_unpermute", "split_qkv_rmsnorm_rope", "rotary_embedding",
    "causal_conv1d_fn", "causal_conv1d_update", "fused_gdn_gating",
    "chunk_gated_delta_rule", "chunk_gdn_fwd_prepare", "chunk_gdn_fwd_h",
    "chunk_gdn_fwd_o", "recurrent_gdn_recover_ssm",
    "recurrent_gated_delta_rule", "fused_recurrent_kda", "chunk_kda_fwd",
    "kda_chunk_products", "kda_wy_transform", "kda_chunk_output",
    "kda_triangular_solve", "ssm_state_gather", "ssm_state_scatter",
    "hc_pre", "hc_post", "hc_combine_norm", "mhc_stream_mix",
    "attn_res_prefix", "ple_lookup", "engram_ngram_hash", "engram_uva_h2d",
    "gated_mul_add", "vocab_embedding", "rejection_sampler",
    "categorical_sampler", "quant_matmul", "matmul", "l2_norm",
    "rms_norm_gated", "vector_add", "muls_add", "vector_mul_gate",
    "vector_pack_indices", "index_add", "tensor_pad", "dtype_cast",
    "tensor_copy", "tensor_fill",
})

_QUANT_DTYPES = frozenset({
    "int8", "int4", "w4a8", "w4a16", "fp8_e4m3", "fp8_e5m2", "mxfp8", "mxfp4",
})
_DENSE_GEMM_FAMILIES = frozenset({
    "gemm", "qkv_projection", "output_projection", "gate_up_projection",
    "down_projection", "index_projection", "gate_projection", "projection",
    "router", "logits", "attention_qk", "attention_pv", "vision_merge",
})
_UNQUANTIZED_GEMM_ROLES = frozenset({
    "router", "logits", "attention_qk", "attention_pv", "vision_merge",
})
_ELEMENTWISE_ADD_RECIPES = frozenset({
    "residual_add", "expert_add", "draft_feature_add",
    "catalog_engram_residual_add", "markov_bias", "attention_residual_mix",
    "catalog_aligner_bias", "catalog_qwen_merger_bias", "catalog_vision_bias",
    "catalog_vision_residual", "gemma_gamma_shift", "router_correction_bias",
})
_ELEMENTWISE_MUL_GATE_RECIPES = frozenset({
    "attention_gate", "gated_output", "shared_gate_apply",
    "index_key_basis_scale", "index_query_basis_scale",
    "route_normalize", "router_group_mask",
})
_DIRECT_FAMILY_BUILDERS: dict[str, tuple[str, str]] = {
    "allgather": ("all_gather", "hcclAllGather"), "engram_broadcast": ("all_gather", "hcclAllGather"),
    "dcp_allgather": ("all_gather", "hcclAllGather"), "engram_ids": ("all_to_all_v", "hcclAlltoAllV"),
    "reduce_scatter": ("reduce_scatter", "hcclReduceScatter"),
    "pp_p2p": ("p2p_send_recv", "hcclSendRecv"), "cache_transfer": ("p2p_send_recv", "hcclSendRecv"),
    "state_publish": ("p2p_send_recv", "hcclSendRecv"), "state_transfer": ("p2p_send_recv", "hcclSendRecv"),
    "add_rmsnorm_quant": ("add_rms_norm_quant", "aclnnAddRmsNormDynamicQuantV2"),
    "rmsnorm_quant": ("rms_norm_quant", "aclnnRmsNormDynamicQuant"),
    "add_rmsnorm": ("add_rms_norm", "aclnnAddRmsNorm"), "layernorm": ("layer_norm", "aclnnLayerNorm"),
    "geglu": ("geglu", "aclnnGeGluV2"), "gelu": ("gelu", "aclnnGelu"),
    "situ": ("situ_activation", "aclnnSitu"), "sigmoid": ("sigmoid", "aclnnSigmoid"),
    "softmax": ("softmax", "aclnnSoftmaxV2"), "attention_softmax": ("softmax", "aclnnSoftmaxV2"),
    "dequant_swiglu_quant": ("dequant_swiglu_quant", "aclnnDequantSwigluQuant"),
    "attention_merge": ("dcp_lse_attention_merge", "aclnnRingAttentionUpdate"),
    "indexer_quant_cache": ("indexer_quant_cache", "aclnnIndexerQuantCache"),
    "key_pool": ("key_pool", "aclnnKeyPool"), "kv_compress_epilog": ("kv_compress_epilog", "aclnnKvCompressEpilog"),
    "histogram": ("histogram_v2", "aclnnHistogramV2"), "moe_init_routing": ("moe_init_routing", "aclnnMoeInitRoutingV3"),
    "causal_conv1d_fn": ("causal_conv1d_fn", "aclnnCausalConv1d"),
    "short_convolution": ("causal_conv1d_update", "aclnnCausalConv1dUpdate"),
    "conv1d": ("causal_conv1d_update", "aclnnCausalConv1dUpdate"),
    "gdn_gating": ("fused_gdn_gating", "fused_gdn_gating_kernel"),
    "gdn_prepare": ("chunk_gdn_fwd_prepare", "aclnnChunkGatedDeltaRuleFwdPrepare"),
    "gdn_state": ("chunk_gdn_fwd_h", "aclnnChunkFwdH"), "gdn_output": ("chunk_gdn_fwd_o", "aclnnChunkFwdO"),
    "triangular_solve": ("kda_triangular_solve", "solve_tril_16x16_kernel"),
    "state_gather": ("ssm_state_gather", "_gather_ssm_states_kernel"),
    "state_scatter": ("ssm_state_scatter", "_scatter_ssm_states_kernel"),
    "attn_res_prefix": ("attn_res_prefix", "aclnnAttnResFwd"),
    "engram_gather": ("ple_lookup", "aclnnEmbedding"), "engram_query": ("engram_ngram_hash", "engram_ngram_ids_kernel"),
    "engram_rh2d": ("engram_uva_h2d", "aclrtMemcpyAsync"), "engram_apply": ("gated_mul_add", "aclnnMul"),
    "embedding": ("vocab_embedding", "aclnnEmbedding"),
    "sampling": ("categorical_sampler", "_categorical_sample_kernel"),
    "confidence": ("categorical_sampler", "_categorical_sample_kernel"),
    "token_commit": ("categorical_sampler", "_categorical_sample_kernel"),
    "moe_ffn": ("megamoe_fused_mc2", "aclnnMegaMoe"),
    "index_add": ("index_add", "aclnnIndexAdd"), "scatter_add": ("index_add", "aclnnIndexAdd"),
    "gather": ("strided_gather", "aclnnIndexSelect"), "strided_gather": ("strided_gather", "aclnnIndexSelect"),
    "tensor_select": ("strided_gather", "aclnnIndexSelect"),
    "pad": ("tensor_pad", "aclnnPadV3"), "cast": ("dtype_cast", "aclnnCast"),
    "tensor_copy": ("tensor_copy", "aclnnInplaceCopy"), "tensor_materialization": ("tensor_copy", "aclnnInplaceCopy"),
    "tensor_view": ("tensor_copy", "aclnnInplaceCopy"),
    "tensor_fill": ("tensor_fill", "aclnnInplaceZero"), "zero_fill": ("tensor_fill", "aclnnInplaceZero"),
    "fill": ("tensor_fill", "aclnnInplaceZero"),
}


def resolve_builder_strict(
    family: str,
    recipe: str,
    impl: str,
    dtypes: Sequence[str],
    meta: Mapping[str, Any] | None = None,
    *,
    model_id: str = "",
    layer_idx: int = -1,
    local_shape: Sequence[Any] | None = None,
) -> tuple[str, str]:
    """Return ``(builder_name, upstream_abi)`` or raise ``UnmodeledCapabilityError``."""
    meta_dict = dict(meta or {})
    fam = str(family or "").strip()
    rec = str(recipe or "").strip()
    imp = str(impl or "").strip()
    if fam in CONTROL_FAMILIES:
        return (BOUNDARY_BUILDER, fam)
    has_quant = (
        any(str(d) in _QUANT_DTYPES for d in dtypes)
        or str(meta_dict.get("weight_format") or "") in _QUANT_DTYPES
        or str(meta_dict.get("dequant") or "") not in {"", "none"}
    )

    if fam == "sp_matmul_collective" or rec == "sp_matmul_collective_pipeline":
        direction = str(meta_dict.get("sp_collective_kind")
                        or ("allgather" if meta_dict.get("pipeline_gather")
                            and not meta_dict.get("pipeline_scatter")
                            else "reduce_scatter"))
        if "all_gather" in direction or "allgather" in direction:
            return ("all_gather_matmul", "aclnnAllGatherMatmul")
        return ("matmul_reduce_scatter", "aclnnMatmulReduceScatter")
    if fam == "mc2_protocol":
        if int(meta_dict.get("mc2_mode") or 0) in {2, 3}:
            return ("megamoe_fused_mc2", "aclnnMegaMoe")
        native_kind = str(meta_dict.get("native_kind") or "")
        is_v2 = ("V2" in imp or "v2" in rec or native_kind.endswith("V2"))
        if "Dispatch" in imp or "dispatch" in rec or "Dispatch" in native_kind:
            if is_v2:
                return ("mc2_dispatch", "aclnnMoeDistributeDispatchV2")
            return ("mc2_dispatch_v3", "aclnnMoeDistributeDispatchV3")
        if "Combine" in imp or "combine" in rec or "Combine" in native_kind:
            if is_v2:
                return ("mc2_combine", "aclnnMoeDistributeCombineV2")
            return ("mc2_combine_v3", "aclnnMoeDistributeCombineV3")
    if fam in {"alltoall", "alltoallv"}:
        if int(meta_dict.get("mc2_mode") or 0) in {2, 3}:
            return ("megamoe_fused_mc2", "aclnnMegaMoe")
        if rec == "arch22_v2_declared_peer_channel":
            native_kind = str(meta_dict.get("native_kind") or "")
            if "Combine" in imp or "Combine" in native_kind:
                return ("mc2_combine", "aclnnMoeDistributeCombineV2")
            return ("mc2_dispatch", "aclnnMoeDistributeDispatchV2")
        if rec == "expert_dispatch":
            return ("mc2_dispatch", "aclnnMoeDistributeDispatchV2")
        if rec == "expert_combine":
            return ("mc2_combine", "aclnnMoeDistributeCombineV2")
        if rec in {"all_to_all", "dcp_output_alltoall"}:
            return ("all_to_all", "hcclAlltoAll")
        if rec in {"alltoall", "all_to_all_v", "expert_quantized_dispatch",
                   "expert_scale_dispatch"} or fam == "alltoallv":
            return ("all_to_all_v", "hcclAlltoAllV")
    if fam in {"allreduce", "tp_allreduce"}:
        if (meta_dict.get("quantized_all_reduce")
                or meta_dict.get("quantized_all_reduce_applied")
                or rec == "quantized_all_reduce"):
            return ("quantized_all_reduce", "hcclAllReduceInt8")
        return ("all_reduce", "hcclAllReduce")
    if fam == "rmsnorm":
        return ("gemma_rms_norm", "aclnnGemmaRmsNorm") if "gemma" in rec else ("rms_norm", "aclnnRmsNorm")
    if fam == "swiglu":
        return ("swiglu_oai", "aclnnSwiGluOai") if "oai" in rec else ("swiglu", "aclnnSwiGlu")
    if fam == "activation" and rec == "hc_silu":
        return ("silu_mul", "aclnnSilu")
    if fam == "quantize":
        if rec == "swiglu_quantize":
            return ("swiglu_dynamic_quant", "aclnnSwigluQuant")
        if rec == "situ_quantize":
            return ("dequant_situ_quant", "aclnnDequantSituQuant")
        if rec in {"dynamic_quantize", "index_query_quantize", "quantize"}:
            return ("dynamic_quant", "aclnnDynamicQuantV2")
    if fam == "grouped_gemm" or (fam == "gemm" and rec in {"grouped_output_a", "grouped_output_b"}):
        if meta_dict.get("onchip_handoff_to"):
            return ("grouped_matmul_swiglu_quant", "aclnnGroupedMatmulSwigluQuantWeightNZ")
        return ("grouped_matmul", "aclnnGroupedMatmulV5")
    if fam == "mla" and rec == "draft_context_mla_prolog_v3":
        return ("mla_prolog_v3", "aclnnMlaPrologV3WeightNz")
    if fam == "mla_prolog" or rec in {"mla_prolog_v2", "mla_prolog", "mla_preprocess"}:
        return ("mla_prolog_v2", "aclnnMlaPrologV2WeightNz")
    if fam == "compressed_attention" and rec == "aclnn_compressor":
        return ("mla_kv_compressor", "aclnnCompressor")
    if fam == "attention":
        if rec == "aclnn_mixed_quant_sparse_flash_mla" or imp == "aclnnMixedQuantSparseFlashMla":
            return ("mixed_quant_sparse_mla", "aclnnMixedQuantSparseFlashMla")
        if rec in {"sparse_flash_attention", "dsa_sparse_attention"} or imp == "SparseFlashAttention":
            return ("sparse_flash_attention", "aclnnSparseFlashAttention")
        if rec in {"ascend_paged_mla_decode_arch22", "ascend_paged_mla_decode_arch35",
                   "latent_attention0", "latent_attention_batch", "paired_mla_attention_analytic"}:
            return ("absorbed_mla", "aclnnFusedInferAttentionScoreV4")
        if str(meta_dict.get("cache_layout") or "") == "int8_fa3" or rec in {"gbsa_attention", "generic_block_sparse_attention"}:
            return ("block_sparse_attention_c8", "aclnnGenericBlockSparseAttention")
        if rec == "flash_attention_decode":
            return ("incre_flash_attention", "aclnnFusedInferAttentionScoreV3")
        if rec in {"flash_attention", "flash_attention_prefill"}:
            return ("fusion_attention", "aclnnFlashAttentionVarLenScoreV2")
        if rec == "fused_infer_attention":
            return ("fused_infer_attention", "aclnnFusedInferAttentionScoreV3")
    if fam == "lightning_indexer" and rec == "bf16_lightning_indexer_v2":
        return ("lightning_indexer", "aclnnLightningIndexerV2")
    if fam == "indexer":
        if rec == "cann_pool_key_indexer" or imp == "PoolKeyIndexer":
            return ("pool_key_indexer", "aclnnPoolKeyIndexer")
        if rec in {"quant_lightning_indexer", "index_scores", "index_online_topk"}:
            return ("lightning_indexer", "aclnnQuantLightningIndexerV2")
    if fam == "kv_cache_write":
        if rec == "scatter_pa_kv_cache":
            return ("scatter_pa_kv_cache", "aclnnScatterPaKvCache")
        if rec == "qsa_index_cache_write":
            return ("scatter_index_cache", "aclnnScatterNdUpdate")
    if fam == "cache_store":
        if rec in {"kv_store", "draft_context_kv_scatter"}:
            return ("scatter_pa_kv_cache", "aclnnScatterPaKvCache")
        if rec in {"index_store", "cann_keypool_cache_write"}:
            return ("scatter_index_cache", "aclnnScatterNdUpdate")
    if fam == "topk":
        if rec == "moe_gating_topk":
            return (("moe_gating_topk_hash", "aclnnMoeGatingTopKHash") if "Hash" in imp
                    else ("moe_gating_topk", "aclnnMoeGatingTopKSoftmaxV2"))
        if rec in {"index_topk", "index_online_topk", "router_expert_topk",
                   "router_group_score_top2", "router_group_topk", "candidate_merge"}:
            return ("indexer_topk", "aclnnTopkV2")
    if fam == "routing":
        if rec == "moe_init_routing_v3":
            return ("moe_init_routing", "aclnnMoeInitRoutingV3")
        if rec == "route_pack":
            return ("moe_token_permute", "aclnnMoeTokenPermute")
        if rec == "router_weight_gather":
            return ("strided_gather", "aclnnGatherElements")
        if rec == "route_unpack":
            return (("megamoe_fused_mc2", "aclnnMegaMoe") if int(meta_dict.get("mc2_mode") or 0) == 3
                    else ("moe_token_unpermute", "aclnnMoeTokenUnpermute"))
    if fam == "rotary" or (fam == "elementwise" and rec == "rotary"):
        if rec in {"split_qkv_rmsnorm_mrope_fused", "split_qkv_rmsnorm_rope_fused", "split_qkv_rmsnorm_rope_vnorm_fused"}:
            return ("split_qkv_rmsnorm_rope", "split_qkv_rmsnorm_mrope_kernel")
        return ("rotary_embedding", "aclnnApplyRotaryPosEmbV2")
    if fam == "elementwise" and rec == "short_convolution":
        return ("causal_conv1d_update", "aclnnCausalConv1dUpdate")
    if fam == "elementwise" and rec == "delta_gates":
        return ("fused_gdn_gating", "fused_gdn_gating_kernel")
    if fam == "gdn_prefill" and rec == "gdn_phase6_arch22":
        return ("chunk_gated_delta_rule", "aclnnChunkGatedDeltaRuleFwd")
    if fam == "recurrent_vector":
        if "recover_ssm" in rec or bool(meta_dict.get("recover_ssm")):
            return ("recurrent_gdn_recover_ssm", "aclnnRecurrentGatedDeltaRule")
        return ("recurrent_gated_delta_rule", "aclnnRecurrentGatedDeltaRule")
    if fam in {"recurrent_scan", "kda_recurrent"}:
        if rec in {"vllm_ascend_recurrent_kda_arch22", "vllm_ascend_recurrent_kda_arch35"} or imp == "RecurrentKda":
            return ("fused_recurrent_kda", "aclnnRecurrentKda")
        if "gdn_state_contract" in meta_dict or "pcp_segment" in meta_dict:
            return ("chunk_gated_delta_rule", "aclnnChunkGatedDeltaRuleFwd")
        if rec.startswith("recurrent_scan") or rec in {"chunk_kda_fwd", "chunk_kda_fwd_arch22"} or fam == "kda_recurrent":
            return ("chunk_kda_fwd", "aclnnChunkKdaFwd")
    if fam == "recurrent_local":
        if rec.startswith("chunk_products"):
            return ("kda_chunk_products", "aclnnChunkKdaFwdPrepare")
        if rec.startswith("wy_transform"):
            return ("kda_wy_transform", "recompute_w_u_fwd_kernel")
    if fam == "recurrent_output" and rec.startswith("recurrent_output"):
        return ("kda_chunk_output", "aclnnChunkKdaFwdFinalize")
    if fam == "copy" and rec == "cann_keypool_state_gather":
        return ("ssm_state_gather", "_gather_ssm_states_kernel")
    if fam == "elementwise" and rec == "recover_ssm_commit":
        return ("ssm_state_scatter", "_scatter_ssm_states_kernel")
    if fam == "mhc":
        if rec in {"hc_pre_fused", "hc_pre", "hc_pre_mix"}:
            return ("hc_pre", "aclnnHcPre")
        if rec in {"hc_post_fused", "hc_post", "hc_post_mix"}:
            return ("hc_post", "aclnnHcPost")
        if rec in {"hc_combine_norm", "hc_combine_rmsnorm"}:
            return ("hc_combine_norm", "_hc_combine_norm_kernel")
        if rec == "mc2_routed_expert_program":
            return ("megamoe_fused_mc2", "aclnnMegaMoe")
        if rec in {"hc_inverse_rms", "hc_normalize_scale_bias_gate", "hc_sinkhorn", "hc_gate_mix", "hc_combine"}:
            return ("mhc_stream_mix", f"mhc_{rec}_kernel")
    if fam == "acceptance" and rec == "speculative_acceptance":
        return ("rejection_sampler", "rejection_sample_kernel")
    if rec == "mc2_routed_expert_program":
        return ("megamoe_fused_mc2", "aclnnMegaMoe")
    if fam in _DENSE_GEMM_FAMILIES:
        if has_quant and fam not in _UNQUANTIZED_GEMM_ROLES:
            return ("quant_matmul", "aclnnQuantMatmulWeightNz")
        return ("matmul", "aclnnMatmul")
    if fam == "elementwise":
        if rec in {"decode_q_l2norm", "decode_k_l2norm", "phase6_q_l2norm", "phase6_k_l2norm", "qk_l2norm"} or "l2norm" in rec:
            return ("l2_norm", "l2norm_fwd_kernel2")
        if rec == "linear_output_norm_gate":
            return ("rms_norm_gated", "_layer_norm_fwd_1pass_kernel_npu")
        if rec in _ELEMENTWISE_ADD_RECIPES:
            return ("vector_add", "aclnnAdd")
        if rec in {"muls_add", "moe_muls_add"}:
            return ("muls_add", "muls_add_kernel")
        if rec in _ELEMENTWISE_MUL_GATE_RECIPES:
            return ("vector_mul_gate", "aclnnMul")
        if rec in {"gdn_packed_prepare", "candidate_blocks", "candidate_pack"}:
            return ("vector_pack_indices", "aclnnIndexSelect")
        if rec in {"dcp_output_pack", "dcp_query_pack"}:
            return ("tensor_copy", "aclnnInplaceCopy")
    if fam in _DIRECT_FAMILY_BUILDERS:
        return _DIRECT_FAMILY_BUILDERS[fam]
    if fam == "copy":
        return ("tensor_copy", "aclnnInplaceCopy")

    hint = {
        "schema": ONBOARDING_HINT_SCHEMA, "model_id": str(model_id), "layer_idx": int(layer_idx),
        "family": fam, "recipe": rec, "implementation": imp,
        "local_shape": [int(x) if isinstance(x, int) else x for x in (local_shape or ())],
        "dtypes": [str(d) for d in dtypes],
        "metadata": {str(k): v for k, v in meta_dict.items()
                     if isinstance(v, (str, int, float, bool, type(None)))},
        "agent_action": "inspect upstream kernel ABI in vllm-ascend and register explicit PhysicalRecipe + bench operator builder",
    }
    err = UnmodeledCapabilityError(
        capability=f"{fam}:{rec}:{imp}", entry="export_subgraph_bench_plan",
        code="unmodeled_bench_operator", message=json.dumps(hint, sort_keys=True),
        hints=(hint,),
        minimal_replay_input={"model_id": str(model_id), "layer_idx": int(layer_idx),
                              "family": fam, "recipe": rec, "implementation": imp},
    )
    err.payload = hint
    raise err


_DEFAULT_HEAD_DIM, _MLA_LATENT_DIM, _Q_VIEW_RANK_3D, _Q_VIEW_RANK_4D, _KV_HEAD_AXIS = 128, 512, 3, 4, -2


def attention_shape_dims(
    raw_meta: Mapping[str, Any], dims: Sequence[int], *,
    sequences: int = 1, kv_length: int | None = None, phase: str = "decode",
) -> dict[str, int]:
    """Extract caller-faithful attention shape with explicit sequences, q_len, and kv_len."""
    seqs_val = max(1, int(sequences or 1))
    is_prefill = 1 if str(phase) == "prefill" else 0
    gqa = raw_meta.get("gqa_decode") or raw_meta.get("gqa_prefill")
    if isinstance(gqa, Mapping):
        reqs = gqa.get("requests") or ()
        s_cnt = len(reqs) if reqs else seqs_val
        tokens = sum(int(q) for q, _ in reqs) if reqs else (int(dims[0]) if dims else 0)
        seq_len = max((int(q) + int(c) for q, c in reqs), default=max(1, int(kv_length or tokens)))
        q_len = max((int(q) for q, _ in reqs), default=max(1, tokens // max(1, s_cnt)))
        heads = int(gqa.get("query_heads") or 1)
        return {"tokens": tokens, "sequences": s_cnt, "q_len": q_len, "kv_len": seq_len, "is_prefill": is_prefill,
                "heads": heads, "kv_heads": int(gqa.get("kv_heads") or heads), "seq_len": seq_len,
                "head_dim": int(gqa.get("head_dim") or _DEFAULT_HEAD_DIM)}
    mla = raw_meta.get("paged_mla_geometry")
    if isinstance(mla, Mapping):
        kvs, ctxs = mla.get("kv_lengths") or (), mla.get("contexts") or ()
        s_cnt, tokens = len(kvs) or len(ctxs) or seqs_val, int(dims[0]) if dims else (len(kvs) or len(ctxs) or seqs_val)
        heads = int(mla.get("logical_heads") or mla.get("heads") or 1)
        seq_len = (max((int(v) for v in kvs), default=1) if kvs
                   else max((int(c) + 1 for c in ctxs), default=max(1, int(kv_length or 1))))
        abi = raw_meta.get("native_attention_abi")
        latent = int(abi.get("latent_width") or _MLA_LATENT_DIM) if isinstance(abi, Mapping) else _MLA_LATENT_DIM
        return {"tokens": tokens, "sequences": s_cnt, "q_len": max(1, tokens // max(1, s_cnt)), "kv_len": seq_len,
                "is_prefill": is_prefill, "heads": heads, "kv_heads": 1, "seq_len": seq_len, "head_dim": latent}
    idx = raw_meta.get("indexed_attention_geometry")
    if isinstance(idx, Mapping):
        seqs = idx.get("sequences") or ()
        s_cnt = len(seqs) if seqs else seqs_val
        tokens = sum(int(q) for q, _ in seqs) if seqs else int(idx.get("query") or (dims[0] if dims else 0))
        seq_len = (max((int(q) + int(c) for q, c in seqs), default=1) if seqs
                   else max(1, int(idx.get("query") or 0) + int(idx.get("context") or 0)))
        q_len = max((int(q) for q, _ in seqs), default=max(1, tokens // max(1, s_cnt)))
        return {"tokens": tokens, "sequences": s_cnt, "q_len": q_len, "kv_len": seq_len, "is_prefill": is_prefill,
                "heads": int(idx.get("heads") or 1), "kv_heads": 1, "seq_len": seq_len,
                "head_dim": int(idx.get("latent") or _MLA_LATENT_DIM)}
    tv = raw_meta.get("attention_tensor_view")
    if isinstance(tv, Mapping):
        q_sh = tuple(int(x) for x in (tv.get("query_shape") or ()) if isinstance(x, int))
        k_sh = tuple(int(x) for x in (tv.get("key_shape") or tv.get("cache_head_shape") or ()) if isinstance(x, int))
        kvs = tv.get("kv_lengths") or ()
        s_cnt, tokens = (len(kvs) if kvs else seqs_val), (q_sh[0] if q_sh else (int(dims[0]) if dims else 0))
        heads = q_sh[1] if len(q_sh) >= 2 else 1
        head_dim = q_sh[-1] if len(q_sh) >= _Q_VIEW_RANK_3D else _DEFAULT_HEAD_DIM
        kv_heads = k_sh[_KV_HEAD_AXIS] if len(k_sh) >= 2 else 1
        seq_len = (max((int(v) for v in kvs), default=1) if kvs else (k_sh[0] if len(k_sh) == _Q_VIEW_RANK_3D else (
            int(kv_length) if kv_length is not None else (int(dims[1]) if len(dims) >= _Q_VIEW_RANK_4D else max(1, tokens)))))
        return {"tokens": tokens, "sequences": s_cnt, "q_len": max(1, tokens // max(1, s_cnt)), "kv_len": seq_len,
                "is_prefill": is_prefill, "heads": heads, "kv_heads": kv_heads, "seq_len": seq_len, "head_dim": head_dim}
    tokens = int(dims[0]) if dims else 0
    heads = int(raw_meta.get("heads") or (dims[2] if len(dims) >= _Q_VIEW_RANK_4D else (dims[1] if len(dims) > 1 else 1)))
    seq_len = (int(kv_length) if kv_length is not None else (int(dims[1]) if len(dims) >= _Q_VIEW_RANK_4D
               else int(raw_meta.get("seq_len") or (dims[2] if len(dims) > 2 else max(1, tokens)))))
    return {"tokens": tokens, "sequences": seqs_val, "q_len": max(1, tokens // seqs_val), "kv_len": seq_len,
            "is_prefill": is_prefill, "heads": heads, "kv_heads": int(raw_meta.get("key_heads") or heads),
            "seq_len": seq_len, "head_dim": int(dims[3]) if len(dims) > _Q_VIEW_RANK_3D else int(raw_meta.get("head_dim") or _DEFAULT_HEAD_DIM)}


_FABRIC_OP_FAMILIES = {
    "all_reduce": "allreduce", "quantized_all_reduce": "allreduce",
    "all_gather": "allgather", "reduce_scatter": "reduce_scatter",
    "all_to_all": "alltoall", "all_to_all_v": "alltoallv",
    "p2p_send_recv": "pp_p2p", "send_recv": "pp_p2p",
}
_ROCE_STRICT_MARKERS = ("roce", "rdma", "scaleout", "scale_out")
_CROSS_NODE_LINK_MARKERS = (*_ROCE_STRICT_MARKERS, "inter_node", "inter_pod")


def resolve_fabric_collective_abi(collective_op: str) -> tuple[str, str]:
    """Resolve ``(builder, upstream_abi)`` for a fabric collective via ``resolve_builder_strict``."""
    op = str(collective_op or "").strip()
    family = _FABRIC_OP_FAMILIES.get(op)
    if family is None:
        raise BenchTopologyError(f"fabric_collective_unknown:{op}")
    return resolve_builder_strict(family, op, "", ("fp32",))


def _inter_link_groups(links: Sequence[Any], rank_node: Mapping[int, int]) -> list[list[list[int]]]:
    strict_groups: list[list[list[int]]] = []
    generic_groups: list[list[list[int]]] = []
    for lnk in links:
        text = (f"{lnk.get('channel', '')} {lnk.get('name', '')}" if isinstance(lnk, Mapping)
                else f"{getattr(lnk, 'channel', '')} {getattr(lnk, 'name', '')}").lower()
        raw_members = lnk.get("members", ()) if isinstance(lnk, Mapping) else getattr(lnk, "members", ())
        members = sorted({int(r) for r in (raw_members or ()) if int(r) in rank_node})
        nids = sorted({rank_node[r] for r in members})
        if len(nids) < 2:
            continue
        grouped = [[r for r in members if rank_node[r] == nid] for nid in nids]
        if any(m in text for m in _ROCE_STRICT_MARKERS):
            strict_groups.append(grouped)
        elif any(m in text for m in _CROSS_NODE_LINK_MARKERS):
            generic_groups.append(grouped)
    return strict_groups + generic_groups


def _machine_topology(hardware: Mapping[str, Any], machine: Any | None) -> tuple[list[list[int]], list[list[int]], list[list[list[int]]], bool, set[int] | None]:
    if machine is not None:
        dies = list(getattr(machine, "dies", ()) or ())
        node_ids = sorted({int(getattr(d, "node", 0)) for d in dies})
        pkg_keys = sorted({(int(getattr(d, "node", 0)), int(getattr(d, "package", 0))) for d in dies})
        by_node = [sorted(int(d.rank) for d in dies if int(getattr(d, "node", 0)) == n) for n in node_ids]
        by_pkg = [sorted(int(d.rank) for d in dies if (int(getattr(d, "node", 0)), int(getattr(d, "package", 0))) == k) for k in pkg_keys]
        rank_node = {int(d.rank): int(getattr(d, "node", 0)) for d in dies}
        inter = _inter_link_groups(list(getattr(machine, "links", ()) or ()), rank_node)
        return by_node, by_pkg, inter, bool(inter), {int(d.rank) for d in dies}
    raw_by_node = hardware.get("ranks_by_node")
    raw_by_pkg = hardware.get("package_ranks")
    raw_ranks = hardware.get("ranks")
    avail = {int(r) for r in raw_ranks} if raw_ranks is not None else None
    if raw_by_node is not None:
        by_node = [[int(r) for r in grp] for grp in raw_by_node]
        by_pkg = [[int(r) for r in grp] for grp in (raw_by_pkg or ())]
        rank_node = {r: idx for idx, grp in enumerate(by_node) for r in grp}
        declared_links = hardware.get("declared_links")
        inter = _inter_link_groups(declared_links, rank_node) if declared_links is not None else []
        has_inter = bool(inter) if declared_links is not None else bool(hardware.get("has_inter_node_link", len(by_node) >= 2))
        if avail is None:
            avail = {r for grp in by_node for r in grp}
        return by_node, by_pkg, inter, has_inter, avail
    return [], [], [], bool(hardware.get("has_inter_node_link", True)), avail


def fabric_route_ranks(
    route: str, participants: int, hardware: Mapping[str, Any], *,
    available_ranks: Sequence[int] | None = None, machine: Any | None = None,
) -> list[int]:
    """Construct verified physical rank IDs guaranteed to traverse ``route``."""
    route_key = str(route or "").strip()
    if route_key not in SUPPORTED_FABRIC_ROUTES:
        raise BenchGeometryError(f"fabric_route_unknown:{route_key}")
    if isinstance(participants, bool) or not isinstance(participants, int) or participants < 2:
        raise BenchGeometryError(f"fabric_participants_invalid:{participants}")
    raw_pkg = int(machine.dies_per_package) if machine is not None else hardware.get("dies_per_package")
    raw_node = int(machine.ranks_per_node) if machine is not None else hardware.get("dies_per_node")
    if isinstance(raw_pkg, bool) or not isinstance(raw_pkg, int) or raw_pkg < 1:
        raise BenchGeometryError(f"fabric_dies_per_package_invalid:{raw_pkg}")
    if isinstance(raw_node, bool) or not isinstance(raw_node, int) or raw_node < raw_pkg:
        raise BenchGeometryError(f"fabric_dies_per_node_invalid:{raw_node}:per_pkg={raw_pkg}")
    if raw_node % raw_pkg != 0:
        raise BenchGeometryError(f"fabric_node_not_divisible_by_package:dies_per_node={raw_node},dies_per_package={raw_pkg}")
    by_node, by_pkg, inter_groups, has_inter, declared_avail = _machine_topology(hardware, machine)
    if route_key == "intra_package_die":
        if raw_pkg < 2 or participants > raw_pkg:
            raise BenchGeometryError(f"fabric_intra_package_unrealizable:participants={participants},dies_per_package={raw_pkg}")
        if by_pkg:
            pkg = next((grp for grp in by_pkg if len(grp) >= participants), None)
            if pkg is None:
                raise BenchGeometryError(f"fabric_intra_package_unrealizable:participants={participants}")
            ranks = list(pkg[:participants])
        else:
            ranks = list(range(participants))
    elif route_key == "intra_node_hccs":
        if participants > raw_node:
            raise BenchGeometryError(f"fabric_intra_node_exceeds_node:participants={participants},dies_per_node={raw_node}")
        packages_per_node = raw_node // raw_pkg
        if packages_per_node < 2:
            raise BenchGeometryError(f"fabric_intra_node_hccs_requires_multiple_packages:packages={packages_per_node}")
        if by_node:
            node0 = by_node[0]
            if len(node0) < participants:
                raise BenchGeometryError(f"fabric_intra_node_exceeds_node:participants={participants},node_dies={len(node0)}")
            node0_pkgs = [grp for grp in by_pkg if grp and grp[0] in set(node0)]
            ranks = ([grp[0] for grp in node0_pkgs[:participants]]
                     if len(node0_pkgs) >= participants else list(node0[:participants]))
        else:
            ranks = ([idx * raw_pkg for idx in range(participants)]
                     if participants <= packages_per_node else list(range(participants)))
    else:
        declared_nodes = len(by_node) if by_node else int(hardware.get("nodes") or 1)
        if declared_nodes < 2 or not has_inter:
            raise BenchTopologyError(f"fabric_inter_node_requires_multi_node:nodes={declared_nodes}")
        half, rem = participants // 2, participants - (participants // 2)
        if inter_groups:
            chosen_pair = next(
                ((ga, gb) for grp in inter_groups for i, ga in enumerate(grp)
                 for gb in grp[i + 1 :] if len(ga) >= half and len(gb) >= rem),
                None,
            )
            if chosen_pair is None:
                raise BenchGeometryError(f"fabric_inter_node_half_exceeds_node:half={rem}")
            ranks = [*chosen_pair[0][:half], *chosen_pair[1][:rem]]
        elif by_node:
            if len(by_node[0]) < half or len(by_node[1]) < rem:
                raise BenchGeometryError(f"fabric_inter_node_half_exceeds_node:half={rem},node_dies={len(by_node[0])}")
            ranks = [*by_node[0][:half], *by_node[1][:rem]]
        else:
            if rem > raw_node:
                raise BenchGeometryError(f"fabric_inter_node_half_exceeds_node:half={rem},dies_per_node={raw_node}")
            ranks = [*range(half), *range(raw_node, raw_node + rem)]
    effective_avail = ({int(r) for r in available_ranks} if available_ranks is not None else declared_avail)
    if effective_avail is not None:
        missing = [r for r in ranks if r not in effective_avail]
        if missing:
            raise BenchGeometryError(f"fabric_route_ranks_unavailable:route={route_key},missing={missing}")
    return ranks


LADDER_KV_FAMILIES = frozenset({
    "attention", "compressed_attention", "indexer", "indexer_quant_cache",
    "kv_compress_epilog", "cache_store", "key_pool", "attn_res_prefix",
    "recurrent_scan", "recurrent_vector", "recurrent_local", "recurrent_output",
    "kda_recurrent", "short_convolution", "rotary",
})


def is_ladder_kv_node(node: Mapping[str, Any]) -> bool:
    fam = str(node.get("family") or "")
    if fam in LADDER_KV_FAMILIES:
        return True
    meta = dict((node.get("operator") or {}).get("metadata") or {})
    return str(meta.get("group_axis") or "") in {"cp", "pcp"}


def resolve_collective_cosets(
    topology: Any, metadata: Mapping[str, Any], stage: int = 0,
) -> list[list[int]]:
    """Resolve all physical rank cosets across every internal and ``external_dp`` replica."""
    if topology is None:
        raise BenchTopologyError("collective_topology_missing")
    raw_axis = str(metadata.get("group_axis") or "tp").strip()
    axis = "cp" if raw_axis == "dcp" else raw_axis
    stage_idx = int(stage)
    parallel = getattr(topology, "parallel", None)
    dp = max(1, int(getattr(parallel, "dp", 1) or 1))
    ext_dp = max(1, int(getattr(parallel, "external_dp", 1) or 1))
    pp = max(1, int(getattr(parallel, "pp", 1) or 1))
    total_dp = ext_dp * dp
    if axis == "dp":
        cosets: list[list[int]] = []
        for ext in range(ext_dp):
            try:
                per_rep = [tuple(int(r) for r in topology.ranks(ext * dp + r, stage_idx)) for r in range(dp)]
            except Exception as exc:
                raise BenchTopologyError(f"collective_topology_dp_failed:stage={stage_idx}:{exc}") from exc
            if not per_rep or not per_rep[0]:
                raise BenchTopologyError(f"collective_topology_dp_empty:stage={stage_idx},ext={ext}")
            width = len(per_rep[0])
            if any(len(row) != width for row in per_rep):
                raise BenchTopologyError(f"collective_topology_dp_ragged:stage={stage_idx},ext={ext}")
            cosets.extend([per_rep[r][off] for r in range(dp)] for off in range(width))
        return cosets
    if axis == "pp":
        cosets = []
        try:
            for replica in range(total_dp):
                per_stage = [tuple(int(r) for r in topology.ranks(replica, s)) for s in range(pp)]
                if not per_stage or not per_stage[0]:
                    raise BenchTopologyError(f"collective_topology_pp_empty:replica={replica}")
                width = len(per_stage[0])
                if any(len(row) != width for row in per_stage):
                    raise BenchTopologyError(f"collective_topology_pp_ragged:replica={replica}")
                cosets.extend([per_stage[s][off] for s in range(pp)] for off in range(width))
        except BenchTopologyError:
            raise
        except Exception as exc:
            raise BenchTopologyError(f"collective_topology_pp_failed:{exc}") from exc
        return cosets
    seen: set[tuple[int, ...]] = set()
    cosets = []
    from mindie_sim.prediction.sharding.local_tp import expert_tp_groups
    for replica in range(total_dp):
        try:
            if not topology.ranks(replica, stage_idx):
                raise BenchTopologyError(f"collective_topology_missing_ranks:axis={axis},stage={stage_idx},replica={replica}")
            groups = (expert_tp_groups(topology, replica=replica, stage=stage_idx)
                      if axis == "etp"
                      else topology.groups(axis, replica, stage_idx, metadata=metadata))
        except BenchTopologyError:
            raise
        except Exception as exc:
            raise BenchTopologyError(
                f"collective_topology_resolution_failed:axis={axis},stage={stage_idx},replica={replica}:{exc}"
            ) from exc
        for grp in groups:
            key = tuple(int(r) for r in grp)
            if not key:
                raise BenchTopologyError(f"collective_topology_empty_group:axis={axis},stage={stage_idx},replica={replica}")
            if key not in seen:
                seen.add(key)
                cosets.append(list(key))
    if not cosets:
        raise BenchTopologyError(f"collective_topology_no_groups:axis={axis},stage={stage_idx}")
    return cosets


def resolve_collective_group_ranks(
    topology: Any, metadata: Mapping[str, Any], stage: int = 0, *, rank: int | None = None,
) -> list[int] | None:
    """Return the coset matching declared ``group_ranks`` or physical ``rank`` (strict on mismatch)."""
    if topology is None:
        return None
    cosets = resolve_collective_cosets(topology, metadata, stage)
    declared = metadata.get("group_ranks")
    meta_rank = metadata.get("physical_rank")
    if meta_rank is None:
        meta_rank = metadata.get("rank")
    if declared is not None:
        members = tuple(int(r) for r in declared)
        matched = next((grp for grp in cosets if len(grp) == len(members) and set(grp) == set(members)), None)
        if matched is None:
            raise BenchTopologyError(f"collective_declared_group_unmatched:declared={list(members)},cosets={cosets}")
        if meta_rank is not None and int(meta_rank) not in matched:
            raise BenchTopologyError(f"collective_metadata_rank_not_in_declared_group:rank={meta_rank},declared={list(members)}")
        if rank is not None and int(rank) not in matched:
            raise BenchTopologyError(f"collective_rank_not_in_declared_group:rank={rank},declared={list(members)}")
        return list(matched)
    target = int(rank) if rank is not None else int(meta_rank if meta_rank is not None else 0)
    chosen = next((grp for grp in cosets if target in grp), None)
    if chosen is None:
        raise BenchTopologyError(f"collective_rank_not_in_cosets:rank={target},stage={stage},cosets={cosets}")
    return list(chosen)


def collapse_device_kernel_groups(
    nodes: Sequence[Mapping[str, Any]],
    subgraphs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collapse multi-phase device_kernel_group records into one physical invocation."""
    buckets: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for node in nodes:
        if str(node.get("family") or "") in CONTROL_FAMILIES:
            continue
        dkg = dict((node.get("operator") or {}).get("metadata") or {}).get("device_kernel_group")
        if isinstance(dkg, Mapping) and dkg.get("id"):
            gkey = (int(node.get("replica", 0)), int(node.get("stage", 0)),
                    int(node.get("layer", -1)), str(dkg["id"]))
            buckets.setdefault(gkey, []).append(node)
    alias: dict[str, str] = {}
    collapsed_by_owner: dict[str, dict[str, Any]] = {}
    removed_ids: set[str] = set()
    for members in buckets.values():
        owner = next(
            (m for m in reversed(members)
             if dict(dict((m.get("operator") or {}).get("metadata") or {}).get("device_kernel_group") or {}).get("exit")),
            members[-1],
        )
        owner_id = str(owner["node_id"])
        member_ids = {str(m["node_id"]) for m in members}
        for mid in member_ids:
            alias[mid] = owner_id
            if mid != owner_id:
                removed_ids.add(mid)
        ext_deps = list(dict.fromkeys(
            str(d) for m in members for d in (m.get("deps") or ()) if str(d) not in member_ids
        ))
        work = {k: sum(float((m.get("physical_work") or {}).get(k, 0.0)) for m in members)
                for k in ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops", "link_bytes")}
        mflops: dict[str, float] = {}
        rwork: dict[str, float] = {}
        for m in members:
            for dt, val in dict((m.get("physical_work") or {}).get("matrix_flops") or {}).items():
                mflops[str(dt)] = mflops.get(str(dt), 0.0) + float(val)
            for rk, val in dict((m.get("physical_work") or {}).get("resource_work") or {}).items():
                rwork[str(rk)] = rwork.get(str(rk), 0.0) + float(val)
        work.update(matrix_flops=mflops, resource_work=rwork, launches=1)
        op_copy = dict(owner.get("operator") or {})
        meta_copy = dict(op_copy.get("metadata") or {})
        meta_copy.pop("device_kernel_group", None)
        op_copy["metadata"] = meta_copy
        sig = (owner["physical_signature"] if len(members) == 1
               else "dkg:" + "|".join(str(m["physical_signature"]) for m in members))
        collapsed_by_owner[owner_id] = {
            **dict(owner), "deps": ext_deps, "operator": op_copy,
            "physical_work": work, "physical_signature": sig,
        }
    if not collapsed_by_owner:
        return [dict(n) for n in nodes], [dict(s) for s in subgraphs]
    out_nodes: list[dict[str, Any]] = []
    for node in nodes:
        nid = str(node["node_id"])
        if nid in removed_ids:
            continue
        rec = dict(collapsed_by_owner.get(nid, node))
        rec["deps"] = list(dict.fromkeys(
            alias.get(str(d), str(d)) for d in (rec.get("deps") or ())
            if alias.get(str(d), str(d)) != nid
        ))
        out_nodes.append(rec)
    out_subs: list[dict[str, Any]] = []
    for sub in subgraphs:
        srec = dict(sub)
        srec["node_ids"] = list(dict.fromkeys(
            alias.get(str(nid), str(nid)) for nid in (sub.get("node_ids") or ())
        ))
        out_subs.append(srec)
    return out_nodes, out_subs


__all__ = [
    "BOUNDARY_BUILDER", "BUILDERS", "COLLECTIVE_BUILDERS", "CONTROL_FAMILIES",
    "ONBOARDING_HINT_SCHEMA", "SUPPORTED_FABRIC_ROUTES",
    "BenchGeometryError", "BenchTopologyError", "UnmodeledCapabilityError",
    "attention_shape_dims", "collapse_device_kernel_groups", "fabric_route_ranks",
    "is_ladder_kv_node", "resolve_builder_strict", "resolve_fabric_collective_abi",
    "resolve_collective_cosets", "resolve_collective_group_ranks",
]
