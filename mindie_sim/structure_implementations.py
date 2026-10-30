"""Native counterparts for already-declared compound semantic primitives.

[Role] Expose known implementation boundaries without manufacturing subgraphs.
[Invariants] A counterpart is neither kernel selection nor tensor/occurrence
  evidence. Optional epilogues with unmodeled work are not silently included.
[Timeouts & Locks] Constant per-operator metadata work; no I/O or locks.
[Side-Effect Policy] Pure metadata; no profiler, capture or device dependency.
"""


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
_SEMANTIC = {
    "rmsnorm": ("RmsNorm",),
    "add_rmsnorm": ("AddRmsNorm",),
    "swiglu": ("SwiGlu", "Swiglu", "SwiGLU"),
    "gdn_gating": ("fused_gdn_gating_kernel",),
    "recurrent_vector": ("RecurrentGatedDeltaRule",),
    "layernorm": ("LayerNormV3",),
}


def native_implementation(op, contract):
    meta = {**op["metadata"].get("semantic_metadata", {}), **op["metadata"]}
    explicit = meta.get("native_implementation")
    if explicit:
        return explicit
    if op.get("source_boundary") == "call" or op["family"] == "barrier":
        return None
    kinds = ()
    if contract:
        if contract[0] in _CONTRACT_NATIVE:
            kinds = (contract[1],)
        elif contract[0] in _NATIVE_CONTRACT:
            kinds = (contract[0],)
    if not kinds:
        kinds = _SEMANTIC.get(op["family"], ())
    conditions = {}
    if not kinds and meta.get("program_kind") == "matmul":
        if op["family"] == "grouped_gemm":
            kinds = ("GroupedMatmul",)
            conditions = {"optional_bias": bool(meta.get("bias")),
                          "epilogue_scope": "declared_matrix_and_dequantization_only"}
        elif meta.get("activation_format") in {"int8", "fp8_e4m3", "fp8_e5m2"}:
            kinds = ("QuantBatchMatmulV3", "QuantBatchMatmulV4")
            conditions = {"native_version_not_selected": True,
                          "optional_bias": bool(meta.get("bias")),
                          "operand_formats": {
                              key: meta[key] for key in
                              ("activation_format", "weight_format", "output_format")}}
    if not kinds and op["family"] == "attention":
        kind = meta.get("attention_tensor_view", {}).get("kind")
        kinds = (("SparseFlashAttention",) if kind == "indexed_latent" else
                 ("SparseAttnSharedkv",) if kind == "shared_kv" else
                 ("FusedInferAttentionScore",))
        conditions = {"native_layout_dispatch_and_auxiliary_outputs_not_selected": True}
    if not kinds and op["recipe"] == "qk_l2norm":
        kinds = ("l2norm_fwd_kernel2",)
        conditions = {"scope": "declared_qk_normalization; native_row_layout_not_selected"}
    if not kinds and op["family"] == "rotary":
        kinds = ("InplacePartialRotaryMul", "_triton_rope_siso")
        conditions = {"native_layout_and_rotary_partition_not_selected": True}
    if not kinds and op["family"] == "routing" and op["recipe"] == "route_pack":
        kinds = ("MoeInitRoutingV2", "MoeInitRoutingV3", "MoeTokenPermute")
        conditions = {"scope": "declared_permutation_and_token_pack",
                      "native_version_and_optional_quantization_not_selected": True}
    if not kinds and op["family"] == "routing" and op["recipe"] == "route_unpack":
        kinds = ("MoeTokenUnpermute", "MoeFinalizeRoutingV2")
        conditions = {"scope": "declared_inverse_permutation_and_weighted_reduction",
                      "optional_residual_not_included": True}
    if not kinds:
        return None
    return {
        "native_kinds": kinds,
        "selection": "compatible_semantic_boundary; runtime_selection_not_implied",
        "basis": "declared_semantics_and_tensor_boundary",
        "conditions": conditions,
    }
