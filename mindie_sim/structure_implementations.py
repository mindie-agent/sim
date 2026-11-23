"""Authoritative (family, recipe) birth-certificate registry and native counterparts.

[Role]
  Resolve every semantic operator to a typed ``OperatorRecipeSpec`` carrying its
  ``block``, ``submodule``, ``backend_stack``, ``native_kinds``, ``readiness``,
  and ``source_ref``, and expose native implementation candidates without
  fabricating execution claims or observed timings.

[Invariants]
  * Every AST-declared ``family`` in ``mindie_sim`` resolves through ONE closed
    registry with ``block`` in ``VALID_BLOCKS``, non-banned ``submodule``
    (``"Other"`` is forbidden), ``backend_stack`` in ``VALID_BACKEND_STACKS``,
    ``readiness`` in ``VALID_READINESS``, and non-empty ``source_ref``.
  * Every executed operator resolves to non-empty ``native_kinds``; pure
    zero-kernel sync/layout views are tagged ``backend_stack="view_reshape"``.
  * A counterpart is neither kernel selection nor occurrence evidence; only an
    explicit selection fact promotes a candidate to ``selected=True``.

[Timeouts & Locks]
  Constant per-operator dictionary lookup; no I/O, device access or locks.

[Side-Effect Policy]
  Pure immutable metadata; no profiler, capture or device dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any

from mindie_sim.structure_contracts import predicted_contract, typed_values
from mindie_sim.structure.native import (
    MIXED_QUANT_NATIVE_KIND, shared_kv_native_name,
)

VALID_BLOCKS = frozenset({"embedding", "attention", "ffn", "logits", "spec_verify"})
VALID_BACKEND_STACKS = frozenset({
    "aclnn", "tilelang", "triton", "torch_npu", "hccl", "mc2", "view_reshape",
})
VALID_READINESS = frozenset({"existing", "framework_enable", "new_kernel_required"})
BANNED_SUBMODULES = frozenset({"", "other", "unknown", "none", "n/a", "misc"})
_UNSET = object()

_CONTRACT_NATIVE = {
    "native_tensor_abi", "gated_norm_active_rows", "attention_active_qkv",
    "sparse_attention_active", "kv_cache_store_active_elements",
    "dynamic_quant_active", "add_rmsnorm_dynamic_quant_active", "rmsnorm_mxquant_active",
    "layernorm_active", "transpose_batchmatmul_active",
    "attention_paged_mla_qk_rope",
}
_NATIVE_CONTRACT = {
    "ChunkGatedDeltaRuleFwdPrepare", "ChunkFwdH", "ChunkFwdO", "CausalConv1d",
    "DynamicMxQuant", "AddRmsNormDynamicMxQuant",
}
_QUANT_ACT_FORMATS = frozenset({"int8", "fp8_e4m3", "fp8_e5m2"})
_WEIGHT_ONLY_QUANT_FORMATS = frozenset({"int4", "w4a8", "mxfp4", "fp4", "fp4_e2m1", "int8"})
_ZERO_KERNEL_FAMILIES = frozenset({
    "barrier", "event", "release", "tensor_view", "attention_metadata",
})
_MATMUL_FAMILIES = frozenset({
    "gemm", "grouped_gemm", "qkv_projection", "o_projection", "output_projection",
    "gate_projection", "gate_up_projection", "down_projection", "projection",
    "index_projection", "router", "logits", "vision_projector",
})
_PRESERVE_DIGIT_SUFFIX = re.compile(
    r"(?:1d|2d|3d|fp32|fp16|bf16|int8|int4|v2|v3|v4|top2|arch35|arch22|p2p|mc2|c2|h2d|d2h|k2)$"
)
_INDEXED_SUFFIX = re.compile(r"(?:_seg\d+|_chunk\d+|_stage\d+|_\d+|\d+)$")


@dataclass(frozen=True)
class OperatorRecipeSpec:
    """Closed birth certificate for one ``(family, recipe)`` primitive."""

    block: str
    submodule: str
    backend_stack: str
    native_kinds: tuple[str, ...]
    readiness: str
    source_ref: str
    conditions: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.block not in VALID_BLOCKS:
            raise ValueError(f"invalid_operator_block:{self.block}")
        if str(self.submodule or "").strip().lower() in BANNED_SUBMODULES:
            raise ValueError(f"banned_operator_submodule:{self.submodule}")
        if self.backend_stack not in VALID_BACKEND_STACKS:
            raise ValueError(f"invalid_backend_stack:{self.backend_stack}")
        if self.readiness not in VALID_READINESS:
            raise ValueError(f"invalid_readiness:{self.readiness}")
        if not str(self.source_ref or "").strip():
            raise ValueError("empty_source_ref")
        if self.backend_stack != "view_reshape" and not self.native_kinds:
            raise ValueError(f"empty_native_kinds_for_executed_op:{self.submodule}")


def _s(block: str, submodule: str, stack: str, kinds: tuple[str, ...],
       source_ref: str, readiness: str = "existing",
       conditions: tuple[tuple[str, Any], ...] = ()) -> OperatorRecipeSpec:
    return OperatorRecipeSpec(block, submodule, stack, kinds, readiness, source_ref, conditions)


def _expand_recipes(groups: tuple[tuple[str, tuple[str, ...], OperatorRecipeSpec], ...]) -> dict[tuple[str, str], OperatorRecipeSpec]:
    return {(family, recipe): spec for family, recipes, spec in groups for recipe in recipes}


def _expand_gemms(groups: tuple[tuple[tuple[str, ...], str, str, str], ...]) -> dict[str, tuple[str, str, str]]:
    return {recipe: (block, submodule, ref) for recipes, block, submodule, ref in groups for recipe in recipes}


def _normalize_recipe_stem(recipe: str) -> str:
    text = str(recipe or "").strip()
    if not text or _PRESERVE_DIGIT_SUFFIX.search(text):
        return text
    return _INDEXED_SUFFIX.sub("", text) or text


# Exact ``(family, recipe)`` overrides (checked raw first, then stem-normalized).
_RECIPE_REGISTRY: dict[tuple[str, str], OperatorRecipeSpec] = _expand_recipes((
    ("attention_merge", ("dcp_lse_merge",), _s("attention", "DCP Attention LSE Merge", "triton", ("dcp_lse_merge_kernel", "AttentionLseMerge"), "vllm-ascend/vllm_ascend/attention/context_parallel/dcp.py")),
    ("elementwise", ("dcp_query_pack", "dcp_output_pack"), _s("attention", "DCP Tensor Pack", "aclnn", ("Cat", "TensorMove"), "vllm-ascend/vllm_ascend/attention/context_parallel/dcp.py")),
    ("elementwise", ("attention_gate", "gated_output"), _s("attention", "Attention Output Gate", "aclnn", ("Mul", "Sigmoid"), "vllm-ascend/vllm_ascend/attention/attention_v1.py")),
    ("elementwise", ("rotary",), _s("attention", "Rotary Position Embedding", "aclnn", ("RotaryPositionEmbedding", "InplacePartialRotaryMul"), "cann/ops-transformer/attention/rope")),
    ("elementwise", ("compress_pool", "index_compress_pool"), _s("attention", "KV Compression Pooling", "tilelang", ("Compressor", "KeyPoolCompress"), "vllm-ascend/vllm_ascend/models/deepseek_v4/compressor.py", "framework_enable")),
    ("elementwise", ("index_key_basis_scale", "index_query_basis_scale"), _s("attention", "DSA Indexer Basis Scale", "aclnn", ("Mul", "Muls"), "vllm-ascend/vllm_ascend/attention/sfa_v1.py")),
    ("elementwise", ("qk_l2norm",), _s("attention", "Recurrent QK L2Norm", "triton", ("l2norm_fwd_kernel2",), "vllm-ascend/vllm_ascend/ops/gdn.py", conditions=(("scope", "declared_qk_normalization; native_row_layout_not_selected"),))),
    ("elementwise", ("decode_q_l2norm", "decode_k_l2norm"), _s("attention", "Recurrent QK L2Norm", "triton", ("l2norm_fwd_kernel2", "L2Normalize"), "vllm-ascend/vllm_ascend/ops/gdn.py")),
    ("elementwise", ("delta_gates",), _s("attention", "Recurrent Delta Gate", "triton", ("fused_gdn_gating_kernel", "FusedKdaGating"), "vllm-ascend/vllm_ascend/ops/gdn.py")),
    ("elementwise", ("short_convolution",), _s("attention", "Causal Short Conv1D", "aclnn", ("CausalConv1d", "CausalConv1dUpdate"), "vllm-ascend/vllm_ascend/ops/causal_conv1d.py")),
    ("elementwise", ("linear_output_norm_gate",), _s("attention", "Gated RMSNorm", "triton", ("layer_norm_gated_fwd_kernel", "_layer_norm_fwd_1pass_kernel_npu", "FusedRMSNormGated"), "vllm-ascend/vllm_ascend/ops/gdn.py")),
    ("elementwise", ("linear_output_norm_pow",), _s("attention", "Eager Gated RMSNorm", "aclnn", ("Pows",), "torch_npu/aclnn/pows")),
    ("elementwise", ("linear_output_norm_mean",), _s("attention", "Eager Gated RMSNorm", "aclnn", ("ReduceMean",), "torch_npu/aclnn/mean")),
    ("elementwise", ("linear_output_norm_add_eps",), _s("attention", "Eager Gated RMSNorm", "aclnn", ("Adds", "Add"), "torch_npu/aclnn/adds")),
    ("elementwise", ("linear_output_norm_rsqrt",), _s("attention", "Eager Gated RMSNorm", "aclnn", ("Rsqrt",), "torch_npu/aclnn/rsqrt")),
    ("elementwise", ("linear_output_norm_mul", "linear_output_norm_weight_mul", "linear_output_gate_mul"), _s("attention", "Eager Gated RMSNorm", "aclnn", ("Mul",), "torch_npu/aclnn/mul")),
    ("elementwise", ("linear_output_gate_sigmoid",), _s("attention", "Eager Gated RMSNorm", "aclnn", ("Sigmoid",), "torch_npu/aclnn/sigmoid")),
    ("cast", ("linear_output_norm_cast_in", "linear_output_gate_cast_in", "linear_output_norm_cast_out"), _s("attention", "Eager Gated RMSNorm", "aclnn", ("Cast",), "torch_npu/aclnn/cast")),
    ("elementwise", ("kda_pcp_restore", "pcp_hidden_restore", "pcp_kv_restore", "pcp_compressed_kv_restore", "pcp_keypool_hidden_restore"), _s("attention", "PCP Sequence Reorder", "aclnn", ("IndexSelect", "TensorMove"), "vllm-ascend/vllm_ascend/attention/context_parallel/pcp.py")),
    ("elementwise", ("kda_pcp_scatter",), _s("attention", "PCP Sequence Scatter", "aclnn", ("Slice", "TensorMove"), "vllm-ascend/vllm_ascend/ops/kda_pcp.py")),
    ("elementwise", ("pcp_c2_ring_write",), _s("attention", "PCP C2 Ring Write", "aclnn", ("ScatterUpdate", "TensorMove"), "vllm-ascend/vllm_ascend/attention/context_parallel/dsa_v41_cp.py")),
    ("elementwise", ("candidate_pack", "candidate_blocks"), _s("attention", "DSA Candidate Pack", "aclnn", ("Cat", "TensorMove"), "vllm-ascend/vllm_ascend/attention/sfa_v1.py")),
    ("copy", ("paged_mla_query_pad_nope", "paged_mla_query_pad_rope"), _s("attention", "Paged MLA Query Padding", "aclnn", ("PadV3", "TensorMove"), "vllm-ascend/vllm_ascend/attention/mla_v1.py")),
    ("copy", ("cann_keypool_nonempty_inputs", "cann_keypool_state_gather"), _s("attention", "KeyPool State Gather", "aclnn", ("GatherV2", "TensorMove"), "vllm-ascend/vllm_ascend/attention/indexer_kpool.py")),
    ("cache_store", ("kv_store", "compressed_store", "index_store", "cann_keypool_cache_write"), _s("attention", "KV/Index Cache Write", "aclnn", ("ReshapeAndCache", "ScatterPaKvCache"), "cann/ops-transformer/attention/reshape_and_cache")),
    ("kv_compress_epilog", ("kv_store", "compressed_store"), _s("attention", "KV Compress Cache Epilog", "aclnn", ("aclnnKvCompressEpilog", "ReshapeAndCache"), "cann/ops-transformer/attention/kv_compress_epilog")),
    ("indexer", ("index_scores",), _s("attention", "DSA Indexer Score", "aclnn", ("LightningIndexer", "QuantLightningIndexerV2"), "cann/ops-transformer/attention/quant_lightning_indexer_v2")),
    ("indexer", ("index_online_topk",), _s("attention", "DSA Indexer Online TopK", "tilelang", ("QuantSparseLightningIndexer", "LightningIndexerTopK"), "vllm-ascend/vllm_ascend/attention/sfa_v1.py", "framework_enable")),
    ("topk", ("index_topk", "candidate_merge"), _s("attention", "DSA Indexer TopK", "aclnn", ("TopKV2", "LightningIndexer"), "cann/ops-transformer/attention/quant_lightning_indexer_v2")),
    ("sigmoid", ("attention_q_gate_sigmoid", "linear_beta_sigmoid"), _s("attention", "Attention Gate Sigmoid", "aclnn", ("Sigmoid",), "torch_npu/aclnn/sigmoid")),
    ("elementwise", ("residual_add",), _s("ffn", "Residual Connection", "aclnn", ("Add",), "torch_npu/aclnn/add")),
    ("elementwise", ("expert_add",), _s("ffn", "MoE Routed-Shared Combine", "aclnn", ("Add",), "vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py")),
    ("elementwise", ("shared_gate_apply",), _s("ffn", "Shared Expert Gate", "aclnn", ("Sigmoid", "Mul"), "vllm-ascend/vllm_ascend/ops/fused_moe/shared_expert.py")),
    ("elementwise", ("route_normalize", "router_correction_bias", "router_group_mask", "router_routed_scale"), _s("ffn", "MoE Router Postprocess", "aclnn", ("MoeGatingTopKSoftmaxV2", "MoeGatingTopK", "Mul"), "vllm-ascend/vllm_ascend/ops/fused_moe/moe_router.py")),
    ("cast", ("router_input_cast",), _s("ffn", "MoE Router Precision Cast", "aclnn", ("Cast",), "vllm-ascend/vllm_ascend/ops/fused_moe/moe_router.py")),
    ("sigmoid", ("router_sigmoid_scores",), _s("ffn", "MoE Router Sigmoid", "aclnn", ("Sigmoid",), "vllm-ascend/vllm_ascend/ops/fused_moe/moe_router.py")),
    ("routing", ("route_pack",), _s("ffn", "MoE Token Permute", "aclnn", ("MoeInitRoutingV2", "MoeInitRoutingV3", "MoeTokenPermute"), "cann/ops-transformer/moe/moe_init_routing", conditions=(("scope", "declared_permutation_and_token_pack"), ("native_version_and_optional_quantization_not_selected", True)))),
    ("routing", ("route_unpack",), _s("ffn", "MoE Token Unpermute", "aclnn", ("MoeTokenUnpermute", "MoeFinalizeRoutingV2"), "cann/ops-transformer/moe/moe_finalize_routing", conditions=(("scope", "declared_inverse_permutation_and_weighted_reduction"), ("optional_residual_not_included", True)))),
    ("routing", ("router_weight_gather",), _s("ffn", "MoE Router Weight Gather", "aclnn", ("GatherD", "IndexSelect"), "vllm-ascend/vllm_ascend/ops/fused_moe/moe_router.py")),
    ("topk", ("router_topk", "router_expert_topk", "router_group_topk", "router_group_score_top2"), _s("ffn", "MoE Router TopK", "aclnn", ("MoeGatingTopKSoftmaxV2", "MoeGatingTopK", "TopKV2"), "cann/ops-transformer/moe/moe_gating_top_k")),
    ("topk", ("hash_routing",), _s("ffn", "MoE Hash Routing", "aclnn", ("MoeHashRouting", "GatherV2"), "vllm-ascend/vllm_ascend/ops/fused_moe/moe_router.py", "framework_enable")),
    ("quantize", ("swiglu_quantize",), _s("ffn", "Fused SwiGLU Quantization", "aclnn", ("SwiGluQuant", "DynamicQuantV2"), "cann/ops-transformer/moe/swiglu_quant")),
    ("quantize", ("situ_quantize",), _s("ffn", "Fused SiTU MX Quantization", "aclnn", ("SituMxQuant", "DynamicMxQuant"), "vllm-ascend/csrc/moe/situ_mx_quant", "new_kernel_required")),
    ("quantize", ("geglu_quantize",), _s("ffn", "Fused GeGLU Quantization", "aclnn", ("GeluQuant", "DynamicQuantV2"), "vllm-ascend/vllm_ascend/ops/activation.py")),
    ("grouped_gemm", ("megamoe_arch35",), _s("ffn", "MegaMoE Fused MC2", "mc2", ("MegaMoeArch35", "MoeDistributeDispatchV2"), "cann/ops-transformer/mc2/megamoe_arch35")),
    ("activation", ("hc_silu",), _s("attention", "Hyper-Connection SiLU", "aclnn", ("Silu", "Swish"), "vllm-ascend/vllm_ascend/models/qwen4_exp/hyper_connection.py")),
    ("cast", ("hc_values_fp32",), _s("attention", "Hyper-Connection FP32 Cast", "aclnn", ("Cast",), "vllm-ascend/vllm_ascend/models/deepseek_v4/mhc.py")),
    ("cast", ("index_weights_fp16", "index_weights_fp32"), _s("attention", "Indexer Head Weight Cast", "aclnn", ("Cast",), "vllm-ascend/vllm_ascend/attention/sfa_v1.py")),
    ("cast", ("residual_values_fp32", "attention_residual_output_cast"), _s("attention", "Attention Residual Cast", "aclnn", ("Cast",), "vllm-ascend/vllm_ascend/models/kimi_k3.py")),
    ("elementwise", ("attention_residual_mix",), _s("attention", "Attention Residual Mix", "aclnn", ("BatchMatMulV2", "Mul"), "vllm-ascend/vllm_ascend/models/kimi_k3.py")),
    ("softmax", ("attention_residual_softmax",), _s("attention", "Attention Residual Softmax", "aclnn", ("SoftmaxV2",), "vllm-ascend/vllm_ascend/models/kimi_k3.py")),
    ("mhc", ("hc_pre_mix", "hc_normalize_scale_bias_gate", "hc_sinkhorn", "hc_inverse_rms"), _s("attention", "Hyper-Connection Pre-Mix", "aclnn", ("HcPre", "RmsNorm", "SoftmaxV2"), "vllm-ascend/csrc/moe/hc_pre", "framework_enable")),
    ("mhc", ("hc_post_mix",), _s("attention", "Hyper-Connection Post-Mix", "aclnn", ("HcPost", "Add"), "vllm-ascend/csrc/moe/hc_post", "framework_enable")),
    ("mhc", ("hc_combine", "hc_combine_rmsnorm"), _s("attention", "Hyper-Connection Low-Rank Combine", "aclnn", ("Qwen4ExpHcCombine",), "vllm-ascend/vllm_ascend/models/qwen4_exp/hyper_connection.py", "new_kernel_required")),
    ("mhc", ("hc_gate_mix",), _s("attention", "Hyper-Connection Low-Rank Mix", "aclnn", ("Qwen4ExpGatedResidualMix",), "vllm-ascend/vllm_ascend/models/qwen4_exp/hyper_connection.py", "new_kernel_required")),
    ("elementwise", ("catalog_engram_residual_add",), _s("attention", "Engram Residual Add", "aclnn", ("Add",), "vllm-ascend/vllm_ascend/models/deepseek_v4/engram.py")),
    ("conv1d", ("ple_short_conv",), _s("attention", "PLE Depthwise Conv1D", "aclnn", ("CausalConv1d", "Conv1D"), "vllm-ascend/vllm_ascend/models/engram/ple.py")),
    ("allgather", ("dcp_query_allgather",), _s("attention", "DCP Query AllGather", "hccl", ("hcom_allGather",), "cann/hccl/allgather")),
    ("allgather", ("kda_pcp_gather", "pcp_kv_gather", "pcp_compressed_kv_gather", "pcp_index_gather", "pcp_hidden_gather", "pcp_keypool_hidden_gather", "pcp_conv_tail_gather", "pcp_gdn_state_gather", "pcp_gdn_hupdate_gather", "pcp_c2_ring_tail_gather", "pcp_ple_conv_state_step"), _s("attention", "PCP Context AllGather", "hccl", ("hcom_allGather",), "cann/hccl/allgather")),
    ("allgather", ("ring_allgather",), _s("ffn", "Sequence Parallel AllGather", "hccl", ("hcom_allGather",), "cann/hccl/allgather")),
    ("reduce_scatter", ("kda_pcp_reduce_scatter",), _s("attention", "KDA PCP ReduceScatter", "hccl", ("hcom_reduceScatter",), "cann/hccl/reduce_scatter")),
    ("reduce_scatter", ("ring_reduce_scatter",), _s("ffn", "Sequence Parallel ReduceScatter", "hccl", ("hcom_reduceScatter",), "cann/hccl/reduce_scatter")),
    ("allreduce", ("kda_pcp_allreduce", "linear_reduce", "output_reduce"), _s("attention", "Attention TP AllReduce", "hccl", ("hcom_allReduce",), "cann/hccl/allreduce")),
    ("allreduce", ("catalog_tp_allreduce", "ring_allreduce", "moe_output_reduce", "expert_tensor_reduce"), _s("ffn", "TP AllReduce", "hccl", ("hcom_allReduce",), "cann/hccl/allreduce")),
    ("alltoall", ("dcp_output_alltoall",), _s("attention", "DCP Output AllToAll", "hccl", ("hcom_alltoall",), "cann/hccl/alltoall")),
    ("alltoall", ("expert_dispatch", "expert_combine"), _s("ffn", "MoE EP AllToAll", "hccl", ("hcom_alltoall", "hcom_alltoallv"), "cann/hccl/alltoall")),
    ("pad", ("sequence_parallel_right_pad",), _s("embedding", "Sequence Parallel Input Pad", "aclnn", ("PadV3",), "vllm-ascend/vllm_ascend/distributed/sp.py")),
    ("pad", ("sequence_parallel_reduce_scatter_pad",), _s("ffn", "Sequence Parallel ReduceScatter Pad", "aclnn", ("PadV3",), "vllm-ascend/vllm_ascend/distributed/sp.py")),
    ("tensor_select", ("select_hidden",), _s("logits", "Logits Token Select", "aclnn", ("IndexSelect", "GatherV2"), "vllm/model_executor/layers/logits_processor.py")),
    ("quantize", ("index_key_quantize", "index_query_quantize"), _s("attention", "DSA Indexer Quantization", "aclnn", ("DynamicQuantV2", "DynamicMxQuant"), "cann/ops-transformer/quant/dynamic_quant")),
    ("quantize", ("kda_pcp_gather_quant", "pcp_hidden_gather_quant", "pcp_keypool_hidden_gather_quant"), _s("attention", "PCP Gather Quantization", "aclnn", ("DynamicQuantV2",), "vllm-ascend/vllm_ascend/attention/context_parallel/pcp.py")),
    ("vector", ("kda_pcp_gather_dequant", "pcp_hidden_gather_dequant", "pcp_keypool_hidden_gather_dequant"), _s("attention", "PCP Gather Dequantization", "aclnn", ("Dequantize", "Mul"), "vllm-ascend/vllm_ascend/attention/context_parallel/pcp.py")),
    ("recurrent_local", ("chunk_products", "wy_transform"), _s("attention", "Recurrent Chunk Local Prepare", "triton", ("ChunkGatedDeltaRuleFwdPrepare", "ChunkKdaFwdPrepare"), "fla-npu/ops/chunk.py")),
    ("triangular_solve", ("triangular_solve",), _s("attention", "Recurrent WY Triangular Solve", "triton", ("SolveTril", "ChunkGatedDeltaRuleFwdPrepare"), "fla-npu/ops/solve_tril.py")),
    ("recurrent_scan", ("recurrent_scan",), _s("attention", "Recurrent State Scan", "triton", ("ChunkFwdH", "RecurrentGatedDeltaRule", "RecurrentKda"), "fla-npu/ops/chunk_h.py")),
    ("recurrent_scan", ("pcp_gdn_hupdate", "pcp_gdn_prefix_scan", "pcp_gdn_fwd_h_rerun"), _s("attention", "PCP GDN Prefix Scan", "triton", ("pcp_gdn_prefix_scan_kernel", "ChunkFwdH"), "vllm-ascend/vllm_ascend/ops/triton/fla/chunk.py")),
    ("recurrent_output", ("recurrent_output",), _s("attention", "Recurrent Chunk Output", "triton", ("ChunkFwdO", "ChunkKdaFwdO"), "fla-npu/ops/chunk_o.py")),
    ("short_convolution", ("ascend_causal_conv1d_update",), _s("attention", "Causal Conv1D Update", "aclnn", ("CausalConv1dUpdate", "CausalConv1d"), "vllm-ascend/vllm_ascend/ops/causal_conv1d.py")),
    ("state_gather", ("gather_recurrent_state",), _s("attention", "Recurrent State Gather", "aclnn", ("GatherV2", "IndexSelect"), "vllm-ascend/vllm_ascend/ops/gdn.py")),
    ("state_scatter", ("scatter_recurrent_state",), _s("attention", "Recurrent State Scatter", "aclnn", ("ScatterUpdate", "ScatterNdUpdate"), "vllm-ascend/vllm_ascend/ops/gdn.py")),
    ("tensor_fill", ("core_output_zero",), _s("attention", "Recurrent Output Buffer Fill", "aclnn", ("Fill", "ZerosLike"), "torch_npu/aclnn/fill")),
))

# Domain submodule + source_ref mapping for GEMM / projection recipes.
_GEMM_SUBMODULES: dict[str, tuple[str, str, str]] = _expand_gemms((
    (("q_a", "q_b", "kv_a", "qkv_a", "q_proj"), "attention", "MLA Input Projection", "vllm-ascend/vllm_ascend/attention/mla_v1.py"),
    (("q_absorb", "v_expand"), "attention", "MLA Latent Absorption", "vllm-ascend/vllm_ascend/attention/sfa_v1.py"),
    (("output", "output_gate", "attention_output"), "attention", "Attention Output Projection", "vllm-ascend/vllm_ascend/attention/attention_v1.py"),
    (("grouped_output_a", "grouped_output_b"), "attention", "MLA Grouped Output Projection", "vllm-ascend/vllm_ascend/attention/dsa_v41.py"),
    (("qkv_projection", "attention_q_gate_projection"), "attention", "Attention QKV Projection", "vllm-ascend/vllm_ascend/attention/attention_v1.py"),
    (("linear_qkvz", "linear_beta_alpha"), "attention", "GDN Linear Input Projection", "vllm-ascend/vllm_ascend/ops/gdn.py"),
    (("linear_qkvbfg_a", "linear_decay_b", "linear_gate_b"), "attention", "KDA Linear Input Projection", "vllm-ascend/vllm_ascend/models/kimi_k3/kda.py"),
    (("linear_output",), "attention", "Linear Attention Output Projection", "vllm-ascend/vllm_ascend/ops/gdn.py"),
    (("index_q", "index_k", "index_wq", "index_wk", "index_weights", "index_k_weights", "index_pool_gate", "index_query_hadamard", "index_key_hadamard"), "attention", "DSA Indexer Projection", "vllm-ascend/vllm_ascend/attention/sfa_v1.py"),
    (("compress_values", "compress_gates", "index_compress_values", "index_compress_gates"), "attention", "KV Compression Projection", "vllm-ascend/vllm_ascend/models/deepseek_v4/compressor.py"),
    (("hc_coefficients_fused", "hc_projection", "hc_lowrank_up", "hc_lowrank_down_inject", "hc_lowrank_head_up", "hc_lowrank_head_down"), "attention", "Hyper-Connection Projection", "vllm-ascend/vllm_ascend/models/deepseek_v4/mhc.py"),
    (("attention_residual_scores",), "attention", "Attention Residual Score Projection", "vllm-ascend/vllm_ascend/models/kimi_k3.py"),
    (("engram_gate_projection", "engram_q_gate", "engram_k_gate", "ple_kv_proj"), "attention", "Engram/PLE Projection", "vllm-ascend/vllm_ascend/models/deepseek_v4/engram.py"),
    (("dense_gate_up",), "ffn", "Dense FFN Gate-Up Projection", "vllm-ascend/vllm_ascend/ops/linear.py"),
    (("dense_down",), "ffn", "Dense FFN Down Projection", "vllm-ascend/vllm_ascend/ops/linear.py"),
    (("latent_projection",), "ffn", "MoE Latent Projection", "vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py"),
    (("router", "shared_gate"), "ffn", "MoE Router Gate", "vllm-ascend/vllm_ascend/ops/fused_moe/moe_router.py"),
    (("grouped_gate_up", "balanced_grouped_gate_up", "grouped_down", "balanced_grouped_down"), "ffn", "MoE Expert Grouped GEMM", "cann/ops-transformer/moe/grouped_matmul"),
    (("vocab_logits", "lm_head"), "logits", "LM Head Logits", "vllm/model_executor/layers/vocab_parallel_embedding.py"),
))

# Authoritative 100%-complete fallback table covering EVERY AST family in mindie_sim.
_FAMILY_REGISTRY: dict[str, OperatorRecipeSpec] = {
    "acceptance": _s("spec_verify", "Speculative Token Acceptance", "aclnn", ("RejectionSampler", "ArgMaxV2"), "vllm/v1/spec_decode/rejection_sampler.py"),
    "activation": _s("ffn", "Elementwise Activation", "aclnn", ("Silu", "Gelu"), "torch_npu/aclnn/activation"),
    "add": _s("ffn", "Elementwise Add", "aclnn", ("Add",), "torch_npu/aclnn/add"),
    "add_rmsnorm": _s("attention", "Residual Add + RMSNorm", "aclnn", ("AddRmsNorm",), "cann/ops-transformer/norm/add_rms_norm"),
    "add_rmsnorm_quant": _s("attention", "Fused Add + RMSNorm + Quant", "aclnn", ("AddRmsNormQuant", "AddRmsNormDynamicQuant", "AddRmsNormDynamicMxQuant"), "cann/ops-transformer/norm/add_rms_norm_quant"),
    "all_gather_matmul": _s("ffn", "MC2 AllGather + MatMul", "mc2", ("aclnnAllGatherMatmulV2",), "vllm-ascend/vllm_ascend/compilation/passes/mm_reduce_scatter_fusion_pass.py", "framework_enable"),
    "allgather": _s("attention", "HCCL AllGather", "hccl", ("hcom_allGather",), "cann/hccl/allgather"),
    "allreduce": _s("ffn", "HCCL AllReduce", "hccl", ("hcom_allReduce",), "cann/hccl/allreduce"),
    "alltoall": _s("ffn", "HCCL AllToAll", "hccl", ("hcom_alltoall",), "cann/hccl/alltoall"),
    "alltoallv": _s("ffn", "HCCL AllToAllV", "hccl", ("hcom_alltoallv",), "cann/hccl/alltoallv"),
    "attention": _s("attention", "Flash/Paged Attention Core", "aclnn", ("FusedInferAttentionScore",), "cann/ops-transformer/attention/fused_infer_attention_score"),
    "attention_merge": _s("attention", "Attention Split LSE Merge", "triton", ("dcp_lse_merge_kernel", "AttentionLseMerge"), "vllm-ascend/vllm_ascend/attention/context_parallel/dcp.py"),
    "attention_metadata": _s("attention", "Attention Slot Metadata", "view_reshape", (), "vllm-ascend/vllm_ascend/attention/attention_v1.py"),
    "attention_pv": _s("attention", "Attention Score-Value MatMul", "aclnn", ("BatchMatMulV2", "FusedInferAttentionScore"), "cann/ops-transformer/attention/fused_infer_attention_score"),
    "attention_qk": _s("attention", "Attention Query-Key MatMul", "aclnn", ("BatchMatMulV2", "FusedInferAttentionScore"), "cann/ops-transformer/attention/fused_infer_attention_score"),
    "attention_residual": _s("attention", "Attention Residual Prefix", "aclnn", ("AttnResFwd",), "vllm-ascend/models/kimi_k3.py"),
    "attention_softmax": _s("attention", "Attention Score Softmax", "aclnn", ("SoftmaxV2",), "cann/ops-transformer/attention/softmax"),
    "attn_res_prefix": _s("attention", "Attention Residual Prefix", "aclnn", ("AttnResFwd",), "vllm-ascend/csrc/attention/attn_res_fwd"),
    "barrier": _s("attention", "Stream Synchronization Barrier", "view_reshape", (), "torch_npu/runtime/event_sync"),
    "cache_load": _s("attention", "KV Cache Load", "aclnn", ("GatherV2", "IndexSelect"), "cann/ops-transformer/attention/reshape_and_cache"),
    "cache_store": _s("attention", "KV/Index Cache Write", "aclnn", ("ReshapeAndCache", "ScatterPaKvCache"), "cann/ops-transformer/attention/reshape_and_cache"),
    "cache_transfer": _s("attention", "Cross-Stage Cache Transfer", "hccl", ("hcom_send", "hcom_receive"), "cann/hccl/send_recv"),
    "cast": _s("attention", "Dtype Precision Cast", "aclnn", ("Cast",), "torch_npu/aclnn/cast"),
    "causal_conv1d": _s("attention", "Causal Conv1D", "aclnn", ("CausalConv1d",), "vllm-ascend/vllm_ascend/ops/causal_conv1d.py"),
    "causal_conv1d_fn": _s("attention", "Causal Conv1D Prefill", "aclnn", ("CausalConv1d",), "vllm-ascend/vllm_ascend/ops/causal_conv1d.py"),
    "compressed_attention": _s("attention", "Compressed KV Attention", "aclnn", ("SparseFlashAttention",), "vllm-ascend/vllm_ascend/attention/dsa_v41.py"),
    "confidence": _s("spec_verify", "Speculative Draft Confidence", "aclnn", ("SoftmaxV2", "TopKV2"), "vllm/v1/spec_decode/eagle.py"),
    "conv1d": _s("attention", "Depthwise Conv1D", "aclnn", ("CausalConv1d", "Conv1D"), "vllm-ascend/vllm_ascend/ops/causal_conv1d.py"),
    "conv_state_copy": _s("attention", "Conv1D History State Copy", "aclnn", ("TensorMove", "ScatterUpdate"), "vllm-ascend/vllm_ascend/ops/causal_conv1d.py"),
    "copy": _s("attention", "Device Tensor Copy", "aclnn", ("TensorMove", "GatherV2"), "torch_npu/aclnn/copy"),
    "dcp_allgather": _s("attention", "DCP Query AllGather", "hccl", ("hcom_allGather",), "cann/hccl/allgather"),
    "dequant_swiglu_quant": _s("ffn", "Dequant SwiGLU Quant", "aclnn", ("DequantSwigluQuant",), "cann/ops-transformer/moe/dequant_swiglu_quant"),
    "down_projection": _s("ffn", "FFN Down Projection", "aclnn", ("MatMulV2", "MatMulV3"), "vllm-ascend/vllm_ascend/ops/linear.py"),
    "dsa": _s("attention", "DeepSeek Sparse Attention", "aclnn", ("SparseFlashAttention",), "cann/ops-transformer/attention/sparse_flash_attention"),
    "elementwise": _s("attention", "Elementwise Vector Op", "aclnn", ("Add", "Mul"), "torch_npu/aclnn/elementwise"),
    "embedding": _s("embedding", "Token Embedding Lookup", "aclnn", ("GatherV2", "Embedding"), "vllm/model_executor/layers/vocab_parallel_embedding.py"),
    "engram_apply": _s("attention", "Engram Memory Gate Apply", "aclnn", ("Mul", "Sigmoid"), "vllm-ascend/vllm_ascend/models/deepseek_v4/engram.py"),
    "engram_broadcast": _s("attention", "Engram TP Broadcast", "hccl", ("hcom_broadcast", "hcom_allGather"), "cann/hccl/broadcast"),
    "engram_gather": _s("attention", "Engram Table Gather", "aclnn", ("GatherV2", "IndexSelect"), "vllm-ascend/vllm_ascend/models/deepseek_v4/engram.py"),
    "engram_ids": _s("attention", "Engram N-Gram ID Exchange", "hccl", ("hcom_allGather",), "vllm-ascend/vllm_ascend/models/deepseek_v4/engram.py"),
    "engram_query": _s("attention", "Engram N-Gram Hash Query", "aclnn", ("GatherV2", "Mod"), "vllm-ascend/vllm_ascend/models/deepseek_v4/engram.py"),
    "engram_rh2d": _s("attention", "Engram Host-to-Device Fetch", "torch_npu", ("aclrtMemcpyAsync",), "vllm-ascend/vllm_ascend/models/deepseek_v4/engram.py"),
    "event": _s("attention", "Stream Event Marker", "view_reshape", (), "torch_npu/runtime/event_sync"),
    "fill": _s("attention", "Tensor Fill", "aclnn", ("Fill",), "torch_npu/aclnn/fill"),
    "gate_projection": _s("attention", "Gated Linear Projection", "aclnn", ("MatMulV2", "MatMulV3"), "vllm-ascend/vllm_ascend/ops/linear.py"),
    "gate_up_projection": _s("ffn", "FFN Gate-Up Projection", "aclnn", ("MatMulV2", "MatMulV3"), "vllm-ascend/vllm_ascend/ops/linear.py"),
    "gdn": _s("attention", "Gated DeltaNet Core", "triton", ("ChunkGatedDeltaRuleFwd", "RecurrentGatedDeltaRule"), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "gdn_gating": _s("attention", "GDN Delta Gating", "triton", ("fused_gdn_gating_kernel",), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "gdn_output": _s("attention", "GDN Chunk Output", "triton", ("ChunkFwdO",), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "gdn_prepare": _s("attention", "GDN Chunk Prepare", "triton", ("ChunkGatedDeltaRuleFwdPrepare",), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "gdn_state": _s("attention", "GDN Chunk State Scan", "triton", ("ChunkFwdH",), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "geglu": _s("ffn", "FFN GeGLU Activation", "aclnn", ("Gelu", "Mul"), "torch_npu/aclnn/gelu"),
    "gelu": _s("ffn", "FFN GELU Activation", "aclnn", ("Gelu",), "torch_npu/aclnn/gelu"),
    "gemm": _s("attention", "Dense Linear Projection", "aclnn", ("MatMulV2", "MatMulV3"), "cann/ops-nn/matmul"),
    "grouped_gemm": _s("ffn", "MoE Grouped Expert GEMM", "aclnn", ("GroupedMatmul",), "cann/ops-transformer/moe/grouped_matmul"),
    "hbm_load": _s("attention", "Device HBM Load", "aclnn", ("TensorMove",), "torch_npu/aclnn/copy"),
    "hccl_boundary": _s("ffn", "HCCL Collective Boundary", "hccl", ("hcom_allReduce",), "cann/hccl/collective"),
    "hccl_local": _s("ffn", "HCCL Local Collective", "hccl", ("hcom_allReduce",), "cann/hccl/collective"),
    "hccl_signal": _s("ffn", "HCCL Collective Signal", "hccl", ("hcom_barrier",), "cann/hccl/collective"),
    "histogram": _s("ffn", "MoE Expert Token Histogram", "aclnn", ("Histogram", "MoeInitRoutingV3"), "cann/ops-transformer/moe/moe_init_routing"),
    "index_pool": _s("attention", "DSA KeyPool Compression", "triton", ("PoolKeyCompress", "KeyPoolCompress"), "vllm-ascend/vllm_ascend/attention/indexer_kpool.py"),
    "index_projection": _s("attention", "DSA Indexer Projection", "aclnn", ("MatMulV2", "MatMulV3"), "vllm-ascend/vllm_ascend/attention/sfa_v1.py"),
    "indexer": _s("attention", "DSA Lightning Indexer", "aclnn", ("LightningIndexer", "QuantLightningIndexerV2"), "cann/ops-transformer/attention/quant_lightning_indexer_v2"),
    "indexer_quant_cache": _s("attention", "DSA Indexer Quant Cache", "aclnn", ("aclnnIndexerQuantCache",), "cann/ops-transformer/attention/indexer_quant_cache"),
    "indexer_slot_manage": _s("attention", "DSA Offload Slot Manager", "aclnn", ("FusedLightningIndexerManage",), "vllm-ascend/vllm_ascend/attention/offload.py", "framework_enable"),
    "kda": _s("attention", "Kimi Delta Attention Core", "triton", ("ChunkKdaFwd", "RecurrentKda"), "vllm-ascend/vllm_ascend/ops/kimi_kda.py"),
    "key_pool": _s("attention", "DSA KeyPool Producer", "triton", ("PoolKeyCompress",), "vllm-ascend/vllm_ascend/attention/indexer_kpool.py"),
    "kv_cache_write": _s("attention", "Paged KV Cache Scatter", "aclnn", ("aclnnScatterPaKvCache", "ReshapeAndCache"), "cann/ops-transformer/attention/scatter_pa_kv_cache"),
    "kv_cast": _s("attention", "KV Cache Precision Cast", "aclnn", ("Cast",), "torch_npu/aclnn/cast"),
    "kv_compress_epilog": _s("attention", "KV Compress Cache Epilog", "aclnn", ("aclnnKvCompressEpilog",), "cann/ops-transformer/attention/kv_compress_epilog"),
    "kv_offload_d2h": _s("attention", "KV Offload D2H Transfer", "torch_npu", ("aclrtMemcpyAsync",), "vllm-ascend/vllm_ascend/kv_connector/memfabric.py"),
    "kv_offload_exchange": _s("attention", "KV Offload Exchange", "torch_npu", ("aclrtMemcpyAsync",), "vllm-ascend/vllm_ascend/kv_connector/memfabric.py"),
    "kv_offload_h2d": _s("attention", "KV Offload H2D Transfer", "torch_npu", ("aclrtMemcpyAsync",), "vllm-ascend/vllm_ascend/kv_connector/memfabric.py"),
    "kv_transfer": _s("attention", "PD KV Handoff Transfer", "hccl", ("hcom_send", "hcom_receive"), "vllm-ascend/vllm_ascend/kv_connector/mooncake.py"),
    "latent_projection": _s("ffn", "MoE Latent Projection", "aclnn", ("MatMulV2", "MatMulV3"), "vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py"),
    "layernorm": _s("attention", "LayerNorm", "aclnn", ("LayerNormV3", "aclnnLayerNorm"), "cann/ops-nn/layer_norm_v3"),
    "lightning_indexer": _s("attention", "DSA Lightning Indexer", "aclnn", ("LightningIndexer", "QuantLightningIndexerV2"), "cann/ops-transformer/attention/quant_lightning_indexer_v2"),
    "lm_head": _s("logits", "LM Head Logits", "aclnn", ("MatMulV2", "MatMulV3"), "vllm/model_executor/layers/vocab_parallel_embedding.py"),
    "logits": _s("logits", "LM Head Logits", "aclnn", ("MatMulV2", "MatMulV3"), "vllm/model_executor/layers/vocab_parallel_embedding.py"),
    "matmul_reduce_scatter": _s("ffn", "MC2 MatMul + ReduceScatter", "mc2", ("aclnnMatmulReduceScatterV2",), "vllm-ascend/vllm_ascend/compilation/passes/mm_reduce_scatter_fusion_pass.py", "framework_enable"),
    "mc2_protocol": _s("ffn", "MC2 Fused Dispatch/Combine", "mc2", ("MoeDistributeDispatchV2", "MoeDistributeCombineV2"), "cann/ops-transformer/mc2"),
    "mhc": _s("attention", "Hyper-Connection Residual Mix", "aclnn", ("HcPre", "HcPost"), "vllm-ascend/vllm_ascend/models/deepseek_v4/mhc.py", "framework_enable"),
    "mla": _s("attention", "MLA Prolog / Latent Attention", "aclnn", ("MlaPrologV3", "aclnnMlaPrologV4WeightNz"), "cann/ops-transformer/attention/mla_prolog_v3"),
    "mla_decode": _s("attention", "MLA Decode Attention", "aclnn", ("FusedInferAttentionScore", "SparseFlashAttention"), "cann/ops-transformer/attention/fused_infer_attention_score"),
    "moe": _s("ffn", "MoE Expert Block", "aclnn", ("GroupedMatmul",), "cann/ops-transformer/moe/grouped_matmul"),
    "moe_align": _s("ffn", "MoE Block Token Align", "aclnn", ("MoeInitRoutingV3", "MoeTokenPermute"), "cann/ops-transformer/moe/moe_init_routing"),
    "moe_combine": _s("ffn", "MoE Expert Combine", "mc2", ("MoeDistributeCombineV2", "MoeFinalizeRoutingV2"), "cann/ops-transformer/mc2"),
    "moe_init_routing": _s("ffn", "MoE Init Routing & Permute", "aclnn", ("MoeInitRoutingV3", "MoeInitRoutingV2"), "cann/ops-transformer/moe/moe_init_routing"),
    "multiply": _s("attention", "Elementwise Multiply", "aclnn", ("Mul",), "torch_npu/aclnn/mul"),
    "mxfp4_weight_supply": _s("ffn", "MXFP4 Expert Weight Supply", "aclnn", ("GroupedMatmulSituQuant", "GroupedMatmul"), "vllm-ascend/csrc/moe/grouped_matmul_situ_quant", "new_kernel_required"),
    "native_fusion": _s("attention", "Native Fused Region", "aclnn", ("MlaPrologV3", "HcPre", "HcPost"), "vllm-ascend/csrc/fusion"),
    "norm_gate": _s("attention", "Gated RMSNorm", "triton", ("layer_norm_gated_fwd_kernel", "_layer_norm_fwd_1pass_kernel_npu"), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "o_projection": _s("attention", "Attention Output Projection", "aclnn", ("MatMulV2", "MatMulV3"), "vllm-ascend/vllm_ascend/attention/attention_v1.py"),
    "offload": _s("attention", "Host Offload Transfer", "torch_npu", ("aclrtMemcpyAsync",), "vllm-ascend/vllm_ascend/kv_connector/memfabric.py"),
    "output_projection": _s("attention", "Attention Output Projection", "aclnn", ("MatMulV2", "MatMulV3"), "vllm-ascend/vllm_ascend/attention/attention_v1.py"),
    "pad": _s("ffn", "Sequence Parallel Tensor Pad", "aclnn", ("PadV3",), "vllm-ascend/vllm_ascend/distributed/sp.py"),
    "ple_projection": _s("attention", "PLE Linear Projection", "aclnn", ("MatMulV2", "MatMulV3"), "vllm-ascend/vllm_ascend/models/engram/ple.py"),
    "pooled_indexer": _s("attention", "DSA Pooled Key Indexer", "triton", ("PoolKeyIndexer",), "vllm-ascend/vllm_ascend/attention/indexer_kpool.py"),
    "pp_p2p": _s("ffn", "Pipeline Parallel P2P Send/Recv", "hccl", ("hcom_send", "hcom_receive"), "cann/hccl/send_recv"),
    "prefetch": _s("ffn", "Weight Prefetch", "torch_npu", ("CmoPrefetch",), "vllm-ascend/vllm_ascend/ops/prefetch.py"),
    "prefix_sum": _s("ffn", "Expert Group Prefix Sum", "aclnn", ("Cumsum",), "torch_npu/aclnn/cumsum"),
    "projection": _s("ffn", "Linear Projection", "aclnn", ("MatMulV2", "MatMulV3"), "vllm-ascend/vllm_ascend/ops/linear.py"),
    "qkv_preprocess": _s("attention", "QKV Norm + RoPE Preprocess", "aclnn", ("QkvRmsNormRopeCache",), "cann/ops-transformer/attention/qkv_norm_rope"),
    "qkv_projection": _s("attention", "Attention QKV Projection", "aclnn", ("MatMulV2", "MatMulV3"), "vllm-ascend/vllm_ascend/attention/attention_v1.py"),
    "quantize": _s("ffn", "Dynamic Activation Quantization", "aclnn", ("DynamicQuantV2", "DynamicMxQuant"), "cann/ops-transformer/quant/dynamic_quant"),
    "recover_ssm": _s("attention", "Speculative SSM State Commit", "triton", ("RecoverSsmStateCommit", "RecurrentGatedDeltaRule"), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "recurrent_local": _s("attention", "Recurrent Chunk Local Prepare", "triton", ("ChunkGatedDeltaRuleFwdPrepare", "ChunkKdaFwdPrepare"), "fla-npu/ops/chunk.py"),
    "recurrent_output": _s("attention", "Recurrent Chunk Output", "triton", ("ChunkFwdO", "ChunkKdaFwdO"), "fla-npu/ops/chunk_o.py"),
    "recurrent_scan": _s("attention", "Recurrent State Scan", "triton", ("ChunkFwdH", "RecurrentGatedDeltaRule", "RecurrentKda"), "fla-npu/ops/chunk_h.py"),
    "recurrent_vector": _s("attention", "Recurrent Decode Vector Scan", "triton", ("RecurrentGatedDeltaRule",), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "reduce": _s("attention", "Vector Reduction", "aclnn", ("ReduceMean", "ReduceSum"), "torch_npu/aclnn/reduce"),
    "reduce_scatter": _s("ffn", "HCCL ReduceScatter", "hccl", ("hcom_reduceScatter",), "cann/hccl/reduce_scatter"),
    "release": _s("attention", "Buffer Release Marker", "view_reshape", (), "torch_npu/runtime/memory_release"),
    "relu": _s("ffn", "ReLU Activation", "aclnn", ("Relu",), "torch_npu/aclnn/relu"),
    "rewrite_fused": _s("attention", "TileLang Fused Region", "tilelang", ("TileLangFusedKernel",), "tilelang-ascend/kernels/fused_regions", "framework_enable"),
    "rmsnorm": _s("attention", "RMSNorm", "aclnn", ("RmsNorm",), "cann/ops-transformer/norm/rms_norm"),
    "rmsnorm_quant": _s("attention", "Fused RMSNorm + Dynamic Quant", "aclnn", ("RmsNormQuant", "AddRmsNormQuant"), "cann/ops-transformer/norm/rms_norm_quant"),
    "rotary": _s("attention", "Rotary Position Embedding", "aclnn", ("InplacePartialRotaryMul", "_triton_rope_siso"), "cann/ops-transformer/attention/rope", conditions=(("native_layout_and_rotary_partition_not_selected", True),)),
    "router": _s("ffn", "MoE Router Gate", "aclnn", ("MatMulV2", "MatMulV3"), "vllm-ascend/vllm_ascend/ops/fused_moe/moe_router.py"),
    "routing": _s("ffn", "MoE Token Routing", "aclnn", ("MoeInitRoutingV3", "MoeTokenPermute"), "cann/ops-transformer/moe/moe_init_routing"),
    "sampling": _s("logits", "Token Sampling", "aclnn", ("TopKTopP", "ArgMaxV2"), "vllm/v1/sample/sampler.py"),
    "short_convolution": _s("attention", "Causal Short Conv1D", "aclnn", ("CausalConv1dUpdate", "CausalConv1d"), "vllm-ascend/vllm_ascend/ops/causal_conv1d.py"),
    "sigmoid": _s("attention", "Elementwise Sigmoid", "aclnn", ("Sigmoid",), "torch_npu/aclnn/sigmoid"),
    "silu": _s("ffn", "SiLU Activation", "aclnn", ("Silu",), "torch_npu/aclnn/silu"),
    "situ": _s("ffn", "MoE SiTU Activation", "aclnn", ("SituMxQuant", "Silu"), "vllm-ascend/csrc/moe/situ_mx_quant", "new_kernel_required"),
    "situ_mx_quant": _s("ffn", "Fused SiTU MX Quantization", "aclnn", ("SituMxQuant",), "vllm-ascend/csrc/moe/situ_mx_quant", "new_kernel_required"),
    "softmax": _s("attention", "Softmax", "aclnn", ("SoftmaxV2",), "torch_npu/aclnn/softmax"),
    "state_gather": _s("attention", "Recurrent State Gather", "aclnn", ("GatherV2", "IndexSelect"), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "state_publication": _s("attention", "State Publication Barrier", "view_reshape", (), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "state_publish": _s("attention", "State Publish Barrier", "view_reshape", (), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "state_scatter": _s("attention", "Recurrent State Scatter", "aclnn", ("ScatterUpdate", "ScatterNdUpdate"), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "state_transfer": _s("attention", "Cross-Rank Recurrent State Transfer", "hccl", ("hcom_send", "hcom_receive"), "vllm-ascend/vllm_ascend/ops/gdn.py"),
    "swiglu": _s("ffn", "FFN SwiGLU Activation", "aclnn", ("SwiGlu", "Swiglu", "SwiGLU"), "cann/ops-transformer/activation/swiglu"),
    "tensor_copy": _s("attention", "Device Tensor Copy / Concat", "aclnn", ("TensorMove", "ConcatD"), "torch_npu/aclnn/copy"),
    "tensor_fill": _s("attention", "Device Tensor Fill", "aclnn", ("Fill", "ZerosLike"), "torch_npu/aclnn/fill"),
    "tensor_materialization": _s("attention", "Tensor Contiguous Materialization", "aclnn", ("TensorMove", "Transpose"), "torch_npu/aclnn/contiguous"),
    "tensor_select": _s("logits", "Tensor Index Select", "aclnn", ("IndexSelect", "GatherV2"), "torch_npu/aclnn/index_select"),
    "tensor_view": _s("attention", "Zero-Copy Tensor View", "view_reshape", (), "torch_npu/tensor_view"),
    "token_commit": _s("spec_verify", "Speculative Token Commit", "aclnn", ("ScatterUpdate", "IndexPut"), "vllm/v1/spec_decode/rejection_sampler.py"),
    "topk": _s("ffn", "TopK Selection", "aclnn", ("TopKV2", "MoeGatingTopK"), "torch_npu/aclnn/topk"),
    "tp_allreduce": _s("attention", "TP AllReduce", "hccl", ("hcom_allReduce",), "cann/hccl/allreduce"),
    "transfer": _s("attention", "Device Data Transfer", "torch_npu", ("aclrtMemcpyAsync",), "torch_npu/runtime/memcpy"),
    "triangular_solve": _s("attention", "Recurrent WY Triangular Solve", "triton", ("SolveTril", "ChunkGatedDeltaRuleFwdPrepare"), "fla-npu/ops/solve_tril.py"),
    "vector": _s("attention", "Vector Pointwise Kernel", "aclnn", ("Mul", "Add"), "torch_npu/aclnn/vector"),
    "vision_merge": _s("embedding", "Vision Token Merge", "aclnn", ("ConcatD", "Reshape"), "vllm/model_executor/models/vision.py"),
    "vision_patch_embed": _s("embedding", "Vision Patch Embedding", "aclnn", ("Conv3D", "Conv2D"), "vllm/model_executor/models/vision.py"),
    "vision_projector": _s("embedding", "Vision Multimodal Projector", "aclnn", ("MatMulV2", "MatMulV3"), "vllm/model_executor/models/vision.py"),
    "zero_fill": _s("attention", "Zero Buffer Fill", "aclnn", ("ZerosLike", "Fill"), "torch_npu/aclnn/fill"),
}


def _lookup_registry_spec(family: str, recipe: str) -> OperatorRecipeSpec | None:
    exact = _RECIPE_REGISTRY.get((family, recipe))
    if exact is not None:
        return exact
    stem = _normalize_recipe_stem(recipe)
    if stem != recipe:
        by_stem = _RECIPE_REGISTRY.get((family, stem))
        if by_stem is not None:
            return by_stem
    return _FAMILY_REGISTRY.get(family)


def _infer_backend_stack(kinds: tuple[str, ...], default_stack: str) -> str:
    if not kinds:
        return default_stack
    first = kinds[0]
    lower = first.lower()
    if lower.startswith("hcom_"):
        return "hccl"
    if any(tok in lower for tok in ("moedistribute", "megamoe")):
        return "mc2"
    if lower.startswith(("dsl.", "tilelang.")) or first in {"Compressor", "QuantSparseLightningIndexer", "TileLangFusedKernel"}:
        return "tilelang"
    if (lower.endswith("_kernel") or "_kernel" in lower
            or first.startswith(("ChunkGatedDeltaRule", "ChunkFwd", "ChunkKda", "RecurrentKda",
                                 "RecurrentGatedDeltaRule", "PoolKey", "SolveTril"))):
        return "triton"
    if lower.startswith("aclrt") or first in {"CmoPrefetch"}:
        return "torch_npu"
    return "aclnn" if default_stack == "view_reshape" else default_stack


def _matmul_kinds_and_conditions(op: dict[str, Any], meta: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    if op["family"] == "grouped_gemm":
        return ("GroupedMatmul",), {
            "optional_bias": bool(meta.get("bias")),
            "epilogue_scope": "declared_matrix_and_dequantization_only",
        }
    ntv = meta.get("native_tensor_view")
    if isinstance(ntv, dict) and ntv.get("kind") == "transpose_batchmatmul":
        return ("TransposeBatchMatMul",), {"optional_bias": bool(meta.get("bias"))}
    act_fmt, wt_fmt = meta.get("activation_format"), meta.get("weight_format")
    operand_fmts = {k: meta.get(k) for k in ("activation_format", "weight_format", "output_format") if k in meta}
    if act_fmt in _QUANT_ACT_FORMATS:
        return ("QuantBatchMatmulV3", "QuantBatchMatmulV4"), {
            "native_version_not_selected": True, "optional_bias": bool(meta.get("bias")),
            "operand_formats": operand_fmts,
        }
    if wt_fmt in _WEIGHT_ONLY_QUANT_FORMATS:
        return ("WeightQuantBatchMatmulV2",), {"optional_bias": bool(meta.get("bias")), "operand_formats": operand_fmts}
    if int(meta.get("matrix_groups", 1) or 1) > 1:
        return ("BatchMatMulV2", "TransposeBatchMatMul"), {"optional_bias": bool(meta.get("bias"))}
    return ("MatMulV2", "MatMulV3"), {"optional_bias": bool(meta.get("bias"))}


def _is_matmul_op(op: dict[str, Any], meta: dict[str, Any]) -> bool:
    return meta.get("program_kind") == "matmul" or op.get("family") in _MATMUL_FAMILIES


def _contract_kinds(contract) -> tuple[str, ...]:
    if not contract:
        return ()
    if contract[0] in _CONTRACT_NATIVE:
        return (contract[1],)
    if contract[0] in _NATIVE_CONTRACT:
        return (contract[0],)
    return ()


def _attention_kinds(meta: dict[str, Any]) -> tuple[tuple[str, ...], dict[str, Any]]:
    view = meta.get("attention_tensor_view", {})
    kind = view.get("kind")
    kinds = (("SparseFlashAttention",) if kind == "indexed_latent" else
             (MIXED_QUANT_NATIVE_KIND,) if kind == "mixed_quant_shared_kv" else
             (shared_kv_native_name(view),) if kind == "shared_kv" else
             ("FusedInferAttentionScore",))
    return kinds, {"native_layout_dispatch_and_auxiliary_outputs_not_selected": True}


def native_implementation(op, contract):
    meta = {**op["metadata"].get("semantic_metadata", {}), **op["metadata"]}
    explicit = meta.get("native_implementation")
    if explicit:
        return explicit
    if op.get("source_boundary") == "call" or op["family"] == "barrier":
        return None
    if op.get("source_boundary") == "kernel" and isinstance(meta.get("native_kind"), str):
        return {
            "native_kinds": (meta["native_kind"],),
            "selection": "declared_source_kernel_boundary",
            "basis": "implementation_binding",
            "conditions": {},
        }
    native_kinds = meta.get("native_kinds")
    if isinstance(native_kinds, str):
        native_kinds = (native_kinds,)
    if not native_kinds and isinstance(meta.get("native_kind"), str):
        native_kinds = (meta["native_kind"],)
    if native_kinds:
        kinds = tuple(str(kind) for kind in native_kinds if str(kind).strip())
        if kinds:
            return {
                "native_kinds": kinds,
                "selection": "compatible_semantic_boundary; runtime_selection_not_implied",
                "basis": "explicit_native_identity",
                "conditions": {"native_identity_not_selected": True},
            }
    family, recipe = str(op.get("family") or ""), str(op.get("recipe") or "")
    kinds = _contract_kinds(contract)
    conditions: dict[str, Any] = {}
    if not kinds and family == "attention":
        kinds, conditions = _attention_kinds(meta)
    if not kinds:
        exact = _RECIPE_REGISTRY.get((family, recipe)) or _RECIPE_REGISTRY.get(
            (family, _normalize_recipe_stem(recipe)))
        if exact is not None:
            kinds, conditions = exact.native_kinds, dict(exact.conditions)
    if not kinds and _is_matmul_op(op, meta):
        kinds, conditions = _matmul_kinds_and_conditions(op, meta)
    if not kinds:
        spec = _FAMILY_REGISTRY.get(family)
        if spec is not None:
            kinds, conditions = spec.native_kinds, dict(spec.conditions)
    if not kinds:
        return None
    return {
        "native_kinds": kinds,
        "selection": "compatible_semantic_boundary; runtime_selection_not_implied",
        "basis": "declared_semantics_and_tensor_boundary",
        "conditions": conditions,
    }


def operator_birth_certificate(
    operator: dict[str, Any],
    contract=_UNSET,
    *,
    block: str | None = None,
    resolved_kinds: tuple[str, ...] | None = None,
) -> OperatorRecipeSpec:
    """Resolve the closed ``OperatorRecipeSpec`` birth certificate for one node."""
    if contract is _UNSET:
        contract = predicted_contract(operator)
    meta = {**operator.get("metadata", {}).get("semantic_metadata", {}),
            **operator.get("metadata", {})}
    family = str(operator.get("family") or "")
    recipe = str(operator.get("recipe") or "")
    stem = _normalize_recipe_stem(recipe)
    base = _lookup_registry_spec(family, recipe) or _s(
        "attention", f"Primitive {family or 'op'}", "aclnn", ("CustomOp",),
        "mindie-sim/structure_implementations")
    if _is_matmul_op(operator, meta) and (family, recipe) not in _RECIPE_REGISTRY:
        gemm_info = _GEMM_SUBMODULES.get(recipe) or _GEMM_SUBMODULES.get(stem)
        if gemm_info is not None:
            base = replace(base, block=gemm_info[0], submodule=gemm_info[1], source_ref=gemm_info[2])
    resolved_block = (
        block if block in VALID_BLOCKS
        else meta.get("structure_block") if meta.get("structure_block") in VALID_BLOCKS
        else "logits" if (str(operator.get("op_id") or "").endswith(".final_norm") or recipe == "final_norm")
        else base.block
    )
    if operator.get("source_boundary") == "call" or family in _ZERO_KERNEL_FAMILIES:
        return replace(base, block=resolved_block, backend_stack="view_reshape", native_kinds=())
    kinds = (resolved_kinds if resolved_kinds is not None
             else tuple(_implementation_kinds(native_implementation(operator, contract)))) or base.native_kinds
    stack = _infer_backend_stack(kinds, base.backend_stack)
    return replace(base, block=resolved_block, backend_stack=stack, native_kinds=kinds)


_STATUS_COMPATIBLE = "compatible_not_selected"
_STATUS_DECLARED = "declared_native_implementation"
_NOT_SELECTED = ("not_selected", "unselected", "compatible", "unresolved",
                 "undefined", "none", "candidate", "alternative")


def _implementation_kinds(implementation):
    """Return the declared native names without treating them as selected."""
    if isinstance(implementation, str):
        return [implementation]
    if not isinstance(implementation, dict):
        return []
    kinds = implementation.get("native_kinds")
    if kinds is None and implementation.get("native_kind") is not None:
        kinds = (implementation["native_kind"],)
    if kinds is None:
        return []
    if isinstance(kinds, str):
        kinds = (kinds,)
    return [kind for kind in kinds if isinstance(kind, str) and kind]


def _first_native(implementation):
    kinds = _implementation_kinds(implementation)
    return kinds[0] if len(kinds) == 1 else None


def _selection_fact(implementation, *, source):
    """Interpret a declared selection without inferring it from a name count."""
    if not isinstance(implementation, dict) or implementation.get("selected") is False:
        return None
    selection = implementation.get("selection")
    if selection is None:
        return None
    if isinstance(selection, dict):
        if selection.get("selected") is False:
            return None
        basis = str(selection.get("basis") or selection.get("kind") or "")
        if (selection.get("selected") is True
                or basis in {"selected_software_plan", "declared_selection"}
                or selection.get("native")):
            native = (_first_native(implementation)
                      or (selection.get("native") if isinstance(selection.get("native"), str) else None)
                      or (selection.get("implementation") if isinstance(selection.get("implementation"), str) else None))
            return {"native": native, "selected_by": source,
                    "evidence": basis or "declared_selection", "plan": selection}
        return None
    if isinstance(selection, str):
        text = selection.strip()
        if not text or any(token in text.lower() for token in _NOT_SELECTED):
            return None
        return {"native": _first_native(implementation) or text,
                "selected_by": source, "evidence": text, "plan": {"selection": text}}
    if selection is True or implementation.get("selected") is True:
        return {"native": _first_native(implementation),
                "selected_by": source, "evidence": "declared_selection", "plan": None}
    return None


def _fusion_boundary(meta):
    boundary = meta.get("fusion_boundary")
    return boundary if isinstance(boundary, dict) else None


def _fusion_selection(meta):
    boundary = _fusion_boundary(meta)
    if not boundary:
        return None
    native = _first_native(boundary.get("native"))
    plan = {key: boundary.get(key) for key in
            ("implementation", "program_profile", "submitted_kernels",
             "semantic_phase_count", "timing_basis") if boundary.get(key) is not None}
    return {"native": native, "selected_by": "operator.metadata.fusion_boundary",
            "evidence": "selected_software_plan", "plan": plan}


def _selected_native(operator):
    """A declared selection is evidence, never an observed execution."""
    meta = operator.get("metadata", {})
    native_call = meta.get("native_call")
    if isinstance(native_call, dict):
        api = native_call.get("api") if isinstance(native_call.get("api"), dict) else {}
        native = (api.get("entry") or api.get("aclnn_symbol")
                  or native_call.get("entry") or native_call.get("aclnn_symbol"))
        if native:
            return {"native": str(native), "selected_by": "operator.metadata.native_call",
                    "evidence": "declared_native_call_boundary", "plan": None}
    if meta.get("aclnn_symbol"):
        return {"native": str(meta["aclnn_symbol"]),
                "selected_by": "operator.metadata.aclnn_symbol",
                "evidence": "declared_native_call_boundary", "plan": None}
    binding = meta.get("implementation_binding")
    if isinstance(binding, dict) and binding.get("implementation"):
        return {"native": str(binding["implementation"]),
                "selected_by": "operator.metadata.implementation_binding",
                "evidence": "declared_selected_implementation", "plan": None}
    explicit = meta.get("native_implementation")
    if isinstance(explicit, dict):
        fact = _selection_fact(explicit, source="operator.metadata.native_implementation")
        if fact is not None:
            return fact
    return _fusion_selection(meta)


def _mark_selected(candidates, selected):
    if selected is None:
        return candidates
    for candidate in candidates:
        if selected.get("native") and candidate["native"] == selected["native"]:
            candidate["selected"] = True
            candidate["status"] = "selected_declared"
    return candidates


def _candidate(operator, kind, implementation, *, explicit, index, region_id=None, owner_id=None):
    op_id = operator.get("op_id") if operator else None
    owner = owner_id or region_id or op_id or "operator"
    conditions = implementation.get("conditions") if isinstance(implementation, dict) else None
    return {
        "candidate_id": f"{owner}/native:{kind or index}",
        "native": kind,
        "status": _STATUS_DECLARED if explicit else _STATUS_COMPATIBLE,
        "selected": False, "observed": False,
        "basis": (implementation.get("basis") if isinstance(implementation, dict) else None)
                 or "declared_semantics_and_tensor_boundary",
        "conditions": conditions or {},
        "gaps": [] if kind else [{"code": "native_name_unresolved", "detail": {"owner": owner}}],
    }


def _execution(operator, candidates, selected):
    if selected is not None:
        match = next((row for row in candidates
                      if selected.get("native") and row["native"] == selected["native"]), None)
        native = selected.get("native")
        return {
            "kind": "source_selected_native" if native else "source_selected_plan",
            "selection_state": "selected_declared",
            "selected": {**selected, "candidate_id": match["candidate_id"] if match else None},
            "observed": False,
            "reason": ("declared_software_selection; not a captured occurrence"
                       if native else "a software plan is selected but no unique native API is named"),
        }
    if operator.get("source_boundary") == "call":
        return {"kind": "expanded_source_call", "selection_state": "expanded",
                "selected": None, "observed": False,
                "reason": "children carry the native boundaries; the call is not a kernel"}
    if not candidates:
        return {"kind": "unresolved_native_boundary", "selection_state": "unresolved",
                "selected": None, "observed": False,
                "reason": "no declared native candidate; see representation gaps"}
    state = "single_compatible_unselected" if len(candidates) == 1 else "multiple_compatible_unselected"
    return {"kind": "candidate_set_only", "selection_state": state,
            "selected": None, "observed": False,
            "reason": "declared compatible alternatives are not a runtime selection"}


def native_candidates(operator, contract=_UNSET, *, owner_id=None):
    """Return ``(candidates, execution, gaps)`` for one semantic operator."""
    if contract is _UNSET:
        contract = predicted_contract(operator)
    implementation = native_implementation(operator, contract)
    meta = operator.get("metadata", {})
    explicit = bool(meta.get("native_implementation"))
    kinds = _implementation_kinds(implementation)
    fusion = _fusion_boundary(meta)
    if not kinds and fusion is not None:
        implementation = fusion.get("native") or {}
        explicit = False
        kinds = _implementation_kinds(implementation)
    candidates = [_candidate(operator, kind, implementation, explicit=explicit,
                             index=index, owner_id=owner_id)
                  for index, kind in enumerate(kinds)]
    selected = _selected_native(operator)
    _mark_selected(candidates, selected)
    execution = _execution(operator, candidates, selected)
    gaps = []
    if not kinds:
        gaps.append({
            "code": "native_candidate_unmodeled",
            "detail": {"family": operator.get("family"), "recipe": operator.get("recipe"),
                       "native_tensor_abi": "declared" if contract is not None else "unavailable"},
        })
    if (selected is not None and selected.get("native")
            and not any(candidate["selected"] for candidate in candidates)):
        gaps.append({"code": "selected_native_not_in_candidates",
                     "detail": {"selected": selected["native"],
                                "candidates": [candidate["native"] for candidate in candidates]}})
    return candidates, execution, gaps


def operator_structure(operator, contract=_UNSET, *, semantic_refs=None,
                       owner_id=None, context=None):
    """Recognize one operator's values, birth certificate and native candidates."""
    if contract is _UNSET:
        contract = predicted_contract(operator)
    values = typed_values(operator, contract, context=context)
    candidates, execution, gaps = native_candidates(operator, contract, owner_id=owner_id)
    cert = operator_birth_certificate(
        operator, contract, resolved_kinds=tuple(c["native"] for c in candidates if c.get("native")))
    if values.get("selection") and not execution.get("plan_selection"):
        execution["plan_selection"] = values["selection"]
    unique = []
    for gap in (*values["gaps"], *gaps):
        if gap not in unique:
            unique.append(gap)
    return {
        "contract": contract, "values": values,
        "semantic_refs": list(semantic_refs) if semantic_refs else None,
        "native_candidates": candidates, "native_execution": execution,
        "birth_certificate": cert, "gaps": unique,
    }


def region_structure(implementation, *, region_id, semantic_refs=None):
    """Describe one declared fused/cross-layer region and its native boundary."""
    kinds = _implementation_kinds(implementation)
    explicit = bool(kinds)
    candidates = [_candidate(None, kind, implementation or {}, explicit=explicit,
                             index=index, region_id=region_id)
                  for index, kind in enumerate(kinds)]
    selected = _selection_fact(implementation, source="subgraph.declared_implementation")
    _mark_selected(candidates, selected)
    if selected is not None:
        execution = {
            "kind": "declared_native_region" if selected.get("native") else "declared_region_plan",
            "selection_state": "selected_declared",
            "selected": {**selected,
                         "candidate_id": next((c["candidate_id"] for c in candidates if c["selected"]), None)},
            "observed": False,
            "reason": ("declared region implementation; not a captured occurrence"
                       if selected.get("native") else "a region plan is selected without a unique native API"),
        }
    elif candidates:
        state = "single_compatible_unselected" if len(candidates) == 1 else "multiple_compatible_unselected"
        execution = {"kind": "candidate_set_only", "selection_state": state,
                     "selected": None, "observed": False,
                     "reason": "declared native names are correspondences, not a selection"}
    else:
        execution = {"kind": "semantic_region_only", "selection_state": "unresolved",
                     "selected": None, "observed": False,
                     "reason": "no declared native region implementation"}
    return {"semantic_refs": list(semantic_refs) if semantic_refs else None,
            "native_candidates": candidates, "native_execution": execution}
