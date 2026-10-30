"""Shared tensor contracts for declared operator structures.

[Role] Describe shape/dtype contracts independently of latency estimation.
[Invariants] Contracts preserve the native ABI and declared conditions. A tensor
  match does not establish a vendor implementation, layer identity or timing.
  Unsupported or absent descriptors remain unknown.
[Timeouts & Locks] Bounded per-operator metadata arithmetic; no I/O or locks.
[Side-Effect Policy] Pure records; no profiler dependency or device contexts.
"""
import math

from .structure_matrix_contracts import matrix_comparison_scope, matrix_contract
from .structure_native_contracts import (
    attention_contract, cache_contract, gated_norm_contract, gemma_norm_contract,
    layer_norm_contract, transpose_batchmatmul_contract, add_rmsnorm_bias_contract,
)


GDN_KINDS = {"gdn_prepare": "ChunkGatedDeltaRuleFwdPrepare",
             "gdn_state": "ChunkFwdH", "gdn_output": "ChunkFwdO"}


def quant_contract(op):
    meta, shape, family = op["metadata"], tuple(op["local_shape"]), op["family"]
    required = {"input_format", "output_format", "activation_scale_shape",
                "activation_scale_format"}
    if not required <= meta.keys() or len(shape) != 2:
        return None
    source, output, scale_dtype = (
        meta["input_format"], meta["output_format"], meta["activation_scale_format"])
    scale_shape = tuple(meta["activation_scale_shape"])
    if output == "int8" and scale_dtype == "fp32" and scale_shape == (shape[0],):
        x, y, scale = (shape, source, "ND"), (shape, output, "ND"), (scale_shape, scale_dtype, "ND")
        if family == "quantize":
            return "dynamic_quant_active", "DynamicQuant", x, y, scale
        if family == "add_rmsnorm_quant" and meta.get("norm_has_weight") is True:
            if meta.get("norm_weight_offset", 0.0) != 0.0:
                return None
            gamma = ((shape[-1],), source, "ND")
            affine = gamma if meta.get("norm_has_bias") else ((), None, None)
            return ("add_rmsnorm_dynamic_quant_active", "AddRmsNormDynamicQuant",
                    (x, x, gamma, affine), (y, x, scale))
        return None
    if output != "fp8_e4m3" or scale_dtype != "fp8_e8m0":
        return None
    if family == "rmsnorm_quant":
        if (meta.get("norm_has_weight") is not True or meta.get("norm_weight_offset", 0.0)
                or shape[-1] % 64 or scale_shape != (shape[0], shape[-1] // 64, 2)):
            return None
        return ("rmsnorm_mxquant_active", "RmsNormDynamicMxQuant",
                (shape, source, "ND"), ((shape[-1],), source, "ND"),
                (shape, output, "ND"), (scale_shape, scale_dtype, "NCL"))
    kind = "DynamicMxQuant" if family == "quantize" else "AddRmsNormDynamicMxQuant"
    return kind, shape, source, output, scale_shape, scale_dtype


def gdn_contract(op):
    """Pinned native ABI shapes; flags still require independent caller evidence."""
    meta = op["metadata"]
    lengths, heads, keys = meta["lengths"], meta["heads"], meta["key_heads"]
    tokens = sum(lengths)
    dk, dv = meta.get("head_dim_k", meta.get("dk", 128)), meta.get("head_dim_v", meta.get("dv", 128))
    chunks = sum(math.ceil(length / 64) for length in lengths)
    qk, v = (1, keys, tokens, dk), (1, heads, tokens, dv)
    g, cu, ci = (1, heads, tokens), (len(lengths) + 1,), (chunks * 2,)
    h = (1, heads, chunks, dk, dv)
    family = op["family"]
    if family == "gdn_prepare":
        inputs = (qk, qk, v, g, g, cu, ci)
        outputs = (g, (1, heads, tokens, dk), v, (1, heads, tokens, 64),
                   qk, qk, (1, keys, tokens), (1, keys, tokens), g)
        ins = ("bf16",) * 3 + (meta["gate_dtype"],) * 2 + ("int64",) * 2
        outs = ("fp32",) + ("bf16",) * 5 + ("fp32",) * 3
    elif family == "gdn_state":
        initial = ((len(lengths), heads, dv, dk) if meta["state_v_first"]
                   else (len(lengths), heads, dk, dv))
        inputs = (qk, (1, heads, tokens, dk), v, g, initial, cu, ci)
        outputs = (h, v, (len(lengths) * heads * dk * dv,))
        ins = ("bf16",) * 3 + (meta["gate_dtype"], "fp32") + ("int64",) * 2
        outs = ("bf16", "bf16", "fp32")
        if not meta["use_initial_state"] or not meta["output_final_state"]:
            return None
    else:
        inputs, outputs = (qk, qk, v, h, g, cu, ci), ((tokens * heads * dv,),)
        ins = ("bf16",) * 4 + (meta["gate_dtype"],) + ("int64",) * 2
        outs = ("bf16",)
    return GDN_KINDS[family], inputs, outputs, ins, outs


def conv_contract(op):
    """Active FN descriptors; full cache capacity and values are not inferred."""
    meta = op["metadata"]
    tokens, dim, width = sum(meta["lengths"]), meta["dim"], meta["width"]
    batch = len(meta["lengths"])
    state = bool(meta["state_rows"] and meta["state_history_slots"])
    cache, flags = meta["cache_indices"] is not None, meta["has_initial_state"] is not None
    shapes = ((tokens, dim), (width, dim), (dim,) if meta["bias"] else (),
              (meta["state_history_slots"], dim) if state else (),
              (batch + 1,), (batch,) if cache else (), (batch,) if flags else ())
    metadata_dtypes = meta["metadata_dtypes"]
    dtypes = (meta["dtype"], meta["dtype"], meta["dtype"] if meta["bias"] else None,
              meta["dtype"] if state else None, metadata_dtypes["query_start_loc"],
              metadata_dtypes["cache_indices"] if cache else None,
              metadata_dtypes["has_initial_state"] if flags else None)
    return "CausalConv1d", shapes, ((tokens, dim),), dtypes, (meta["dtype"],)


def short_conv_active_contract(op):
    """Source caller's active tensors, without inventing its native index ABI."""
    meta, shape = op["metadata"], tuple(op["local_shape"])
    if (op["phase"] != "decode" or len(shape) != 2
            or meta.get("caller_implementation") != "vllm_ascend_gdn_decode"):
        return None
    history = meta.get("state_history_slots", 0)
    if not history:
        return None
    dtype, dim = op["dtype"], shape[-1]
    tensor = lambda dims: (dims, dtype, "ND")
    return ("causal_conv1d_active", "CausalConv1d", tensor(shape),
            tensor((meta["kernel_width"], dim)),
            tensor((dim,)) if meta["bias"] else ((), None, None),
            tensor((history, dim)), tensor(shape))


def _recurrent_state_dtype(op):
    meta = op["metadata"]
    state_dtype = meta.get("state_dtype")
    if (op["dtype"] == "bf16" and state_dtype in {"bf16", "fp32"}
            and meta.get("state_element_bytes") == {"bf16": 2, "fp32": 4}[state_dtype]):
        return state_dtype
    return None


def comparison_scope(op):
    family = op["family"]
    if op["metadata"].get("native_norm_kind"):
        return ("selected_native_normalization_tensor_boundary; explicit_caller_beta_and_external_gamma_shift; "
                "scalar_values_and_capture_checkpoint_identity_not_established; "
                + op["metadata"]["norm_auxiliary_scope"])
    if op.get("source_boundary") == "call":
        return "source_call; expand_to_kernel_boundaries_before_tensor_comparison"
    if op["metadata"].get("native_tensor_view", {}).get("kind") == "transpose_batchmatmul":
        return ("canonical_batch_matrix_geometry_and_operand_precision; "
                "physical_permutation_and_native_dispatch_not_compared")
    if op["metadata"].get("program_kind") == "matmul":
        return matrix_comparison_scope(op)
    if family == "layernorm":
        return ("active_input_affine_and_primary_output_descriptors; "
                "mean_rstd_auxiliary_outputs_epsilon_and_dispatch_not_compared")
    if op["recipe"] == "linear_output_norm_gate":
        if op["metadata"].get("recurrent_attention_kind") == "kda":
            return ("active_gated_norm_rows_width_weight_dtype_and_format; "
                    "leading_tensor_axes_epsilon_activation_residual_bias_and_auxiliary_stats_not_compared; "
                    "no_kernel_implementation_or_occurrence_proof")
        return ("native_tensor_abi; norm_kind_grouping_weight_gate_declared; "
                "epsilon_gate_activation_and_norm_order_not_observed")
    if family == "rmsnorm" and op["metadata"].get("norm_weight_offset") == 1.0:
        return ("native_tensor_abi; gemma_weight_offset_declared_from_model_source; "
                "epsilon_and_weight_values_not_observed")
    if op["metadata"].get("attention_tensor_view", {}).get("kind") in {
        "indexed_latent", "shared_kv",
    }:
        return ("active_query_output_and_cache_head_descriptors; "
                "cache_capacity_page_size_sequence_and_index_tensors_sink_and_dispatch_not_compared; "
                "no_kernel_implementation_or_occurrence_proof")
    if op["metadata"].get("attention_tensor_view", {}).get("kind") == "paged_gqa":
        return {
            "comparison_scope": "active_paged_attention_tensor_descriptors",
            "basis": "independently_declared_paged_GQA_storage_and_query_geometry",
            "compared": ["native_operator_type", "query_shape_dtype_format",
                         "cache_block_size_packed_row_width_dtype_format",
                         "main_output_shape_dtype_format", "block_table_batch_count"],
            "not_compared": ["cache_capacity", "block_table_column_count_and_values",
                             "sequence_length_array_values", "kv_head_factorization",
                             "mask_descriptor_and_values", "optional_auxiliary_tensors",
                             "scale", "sparse_mode", "causal_flag", "logical_input_layout",
                             "phase", "context_length", "verify_or_draft_role"],
            "conditions": "Context and execution geometry remain independent declarations; "
                          "tensor equality does not identify a layer or verify/draft invocation.",
            "occurrence_identity": False,
        }
    if op["metadata"].get("attention_tensor_view", {}).get("kind") == "paged_latent":
        return {
            "comparison_scope": "active_paged_mla_rotary_tensor_descriptors",
            "basis": "declared_source_BNSD_query_NBSD_output_and_shared_paged_latent_cache",
            "compared": ["native_operator_type", "native_BNSD_query_shape_dtype_format",
                         "native_BNSD_query_rope_shape_dtype_format",
                         "cache_heads_block_size_width_dtype_format",
                         "native_NBSD_output_shape_dtype_format", "block_table_batch_count"],
            "not_compared": ["cache_capacity", "block_table_column_count_and_values",
                             "sequence_length_values", "scale", "sparse_mode", "causal_flag",
                             "auxiliary_output", "phase", "context_length", "verify_or_draft_role",
                             "captured_binary_identity", "captured_host_split_dispatch"],
            "conditions": "Active tensor equality does not establish captured sequence lengths, "
                          "host dispatch, binary identity, or occurrence identity. Context and "
                          "dispatch still determine the independently declared physical work.",
            "occurrence_identity": False,
        }
    if family == "attention":
        return ("active_qkv_and_primary_output_descriptors; "
                "mask_auxiliary_tensors_sequence_values_layout_scalar_and_dispatch_not_compared; "
                "no_kernel_implementation_or_occurrence_proof")
    if family == "cache_store":
        return ("active_key_value_element_counts_dtypes_and_slot_count; "
                "native_storage_layout_cache_capacity_slot_dtype_values_and_scalar_flags_not_compared; "
                "no_kernel_implementation_or_occurrence_proof")
    if family in GDN_KINDS:
        return "native_tensor_abi; per_sequence_lengths_and_dispatch_flags_declared_not_observed"
    if family == "causal_conv1d_fn":
        return ("active_tensor_descriptors; cache_capacity_rows_not_compared; "
                "run_mode_activation_lengths_cache_indices_initial_flags_declared_not_observed")
    if family == "short_convolution":
        return ("source_caller_active_input_weight_bias_output_and_per_sequence_history_descriptors; "
                "state_bank_capacity_index_prefix_array_abi_scalar_values_and_dispatch_not_compared; "
                "no_native_full_ABI_or_occurrence_identity_proof")
    if family == "moe_init_routing":
        return ("active_BF16_INT32_inputs_and_four_MXFP8_INT32_INT64_E8M0_outputs; "
                "expert_ids_sort_order_quant_flags_scalar_parameters_and_binary_dispatch_not_compared")
    if family == "recurrent_vector":
        supported = _recurrent_state_dtype(op) is not None
        return {
            "comparison_scope": "active_recurrent_gdn_geometry",
            "basis": "declared_BF16_operands_and_BF16_or_FP32_state",
            "contract_support": "supported" if supported else "unsupported_recurrent_gdn_dtype",
            "declared_operand_dtype": op["dtype"],
            "declared_state_dtype": op["metadata"].get("state_dtype"),
            "declared_state_element_bytes": op["metadata"].get("state_element_bytes"),
            "compared": ["query_token_count", "query_key_heads", "value_heads",
                         "key_width", "value_width", "sequence_count", "state_dtype"],
            "not_compared": ["state_capacity", "state_indices_values", "sequence_length_values",
                             "accepted_token_values", "scale", "phase", "verify_or_draft_role"],
            "occurrence_identity": False,
        }
    if family == "tensor_copy":
        return "tensor_shapes; slice_offsets_not_observed"
    return "tensor_contract; analytical_program_not_binary_tiling"


def predicted_contract(op):
    meta = op["metadata"]
    shape = tuple(op["local_shape"])
    family = op["family"]
    if op.get("source_boundary") == "call":
        return None
    if (family == "tensor_materialization" and op["recipe"] == "tensor_copy"
            and meta.get("native_kind") == "TensorMove"
            and meta.get("native_tensor_layout") == "ND"):
        tensor = (shape, op["dtype"], "ND")
        return "native_tensor_abi", "TensorMove", (tensor,), (tensor,)
    if meta.get("native_tensor_view", {}).get("kind") == "transpose_batchmatmul":
        return transpose_batchmatmul_contract(op)
    if family == "layernorm":
        return layer_norm_contract(op)
    if op["recipe"] == "linear_output_norm_gate":
        return gated_norm_contract(op)
    if family == "attention":
        return attention_contract(op)
    if family == "cache_store":
        return cache_contract(op)
    if op["family"] in GDN_KINDS:
        return gdn_contract(op)
    if op["family"] == "causal_conv1d_fn":
        return conv_contract(op)
    if family == "short_convolution":
        return short_conv_active_contract(op)
    if family == "moe_init_routing":
        from .prediction.family_moe_init import tensor_contract
        return tensor_contract(op)
    if meta.get("program_kind") == "matmul":
        return matrix_contract(op)
    if family in {"quantize", "add_rmsnorm_quant", "rmsnorm_quant"}:
        return quant_contract(op)
    if family == "dequant_swiglu_quant":
        from .prediction.family_grouped_quant import tensor_contract as grouped_contract
        grouped = grouped_contract(op)
        if grouped is not None:
            return grouped
        from .prediction.family_shared_quant import tensor_contract
        return tensor_contract(op)
    if op["family"] == "gdn_gating":
        return ("gdn_gating", shape)
    if op["family"] == "recurrent_vector":
        # The retained native scalar-decay ABI supports BF16 Q/K/V/beta.
        # Equal byte widths do not make FP16 and BF16 state interchangeable.
        state_dtype = _recurrent_state_dtype(op)
        if state_dtype is None:
            return None
        return ("recurrent_vector", tuple(meta[k] for k in ("tokens", "key_heads", "heads", "dk", "dv")),
                len(meta.get("sequence_tokens", (meta["tokens"],))), state_dtype)
    if op["family"] == "tensor_copy" and meta.get("source_layout"):
        layout = meta["source_layout"]
        if len(layout["storage_shape"]) == 2 and len(layout["shape"]) == 2:
            return ("slice_copy", tuple(layout["storage_shape"]), tuple(layout["shape"]),
                    meta["input_format"], meta["output_format"])
    if family in {"rmsnorm", "add_rmsnorm"}:
        if meta.get("native_norm_kind") == "AddRmsNormBias":
            return add_rmsnorm_bias_contract(op)
        semantic = {**meta, **meta.get("semantic_metadata", {})}
        if "norm_has_weight" not in semantic:
            return None
        if semantic.get("norm_weight_offset", 0.0) != 0.0:
            return gemma_norm_contract(op)
        return family, shape, op["dtype"], bool(semantic["norm_has_weight"])
    if family == "swiglu" and shape:
        return "swiglu", shape, op["dtype"]
    if family == "relu" and shape:
        return "relu", shape, op["dtype"]
    return None
