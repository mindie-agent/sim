"""Observed MoE dispatch/combine/SwiGLU and quantization operator contracts.

[Role] Validate observed MoE and dynamic quantization kernel contracts.
[Invariants] Preserves exact scale-group and slot-arity constraints.
[Timeouts & Locks] Pure in-memory checks; no I/O or locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

import math

QUANT_FUSION_TYPES = {"DynamicQuant", "DynamicQuantV2", "AddRmsNormDynamicQuant", "RmsNormCast"}
_FLOAT = {"bf16", "fp16"}
_EMPTY = ((), None, None)
QUANT_FUSION_NATIVE_SLOT_ARITIES = {
    "DynamicQuant": {(1, 2)},
    "DynamicQuantV2": {(1, 3)},
    "AddRmsNormDynamicQuant": {(3, 5), (4, 5)},
    "RmsNormCast": {(2, 2)},
}
ADD_RMSNORM_REQUIRED_INPUT_COUNT = 3
ADD_RMSNORM_REQUIRED_OUTPUT_INDICES = (0, 2, 3)


def _issue(kind, reason, detail, missing=()):
    return {
        "category": "capture_metadata_missing" if missing else "contract_support_gap",
        "reason": reason, "detail": detail, "operator_type": kind,
        **({"missing_fields": list(missing)} if missing else {}),
    }


def quant_fusion_contract(native):
    _, kind, inputs, outputs = native
    if (len(inputs), len(outputs)) not in QUANT_FUSION_NATIVE_SLOT_ARITIES[kind]:
        return None, _issue(kind, "unsupported_native_tensor_slots",
                            "Captured slot counts do not match the retained native quantization ABI.")
    required_in = (range(min(len(inputs), ADD_RMSNORM_REQUIRED_INPUT_COUNT))
                   if kind == "AddRmsNormDynamicQuant" else range(len(inputs)))
    required_out = (ADD_RMSNORM_REQUIRED_OUTPUT_INDICES
                    if kind == "AddRmsNormDynamicQuant" else range(len(outputs)))
    missing = [f"input[{i}].shape" for i in required_in if not inputs[i][0]]
    missing += [f"output[{i}].shape" for i in required_out if not outputs[i][0]]
    if missing:
        return None, _issue(kind, "missing_active_tensor_shapes",
                            "Quantization input, result and scale shapes are required.", missing)
    x = inputs[0]
    if len(x[0]) != 2 or any(dim <= 0 for dim in x[0]) or x[1] not in _FLOAT or x[2] != "ND":
        return None, _issue(kind, "unsupported_quantization_geometry",
                            "The verified native projection requires a floating ND row/channel input.")
    rows, width = x[0]
    if kind == "RmsNormCast":
        if (inputs[1] != ((width,), x[1], "ND") or outputs[0] != x
                or outputs[1] != (x[0], "fp32", "ND")):
            return None, _issue(kind, "inconsistent_rmsnorm_cast_geometry",
                                "RmsNormCast must expose both the original-dtype and FP32 normalized outputs.")
        return native, None
    y, scale = (x[0], "int8", "ND"), ((rows,), "fp32", "ND")
    if kind == "DynamicQuantV2":
        # The exported offset is a materialized result, even when its values
        # happen to be zero. Its work cannot be folded into the 1/2-output ABI.
        offset = ((rows,), "fp32", "ND")
        if outputs != (y, scale, offset):
            return None, _issue(kind, "inconsistent_dynamic_quant_v2_geometry",
                                "DynamicQuantV2 retains separate per-row FP32 scale and offset outputs.")
        return ("dynamic_quant_v2_active", kind, x, y, scale, offset), None
    if kind == "DynamicQuant":
        if outputs != (y, scale):
            return None, _issue(kind, "inconsistent_dynamic_quant_geometry",
                                "The retained dynamic INT8 path requires one FP32 scale per input row.")
        return ("dynamic_quant_active", kind, x, y, scale), None
    optional = inputs[3] if len(inputs) == 4 else _EMPTY
    if (inputs[1] != x or inputs[2] != ((width,), x[1], "ND")
            or optional not in {_EMPTY, ((width,), x[1], "ND")}
            or (outputs[0], outputs[2], outputs[3]) != (y, x, scale)
            or outputs[1] != ((1,), "int8", "ND") or outputs[4] != ((1,), "fp32", "ND")):
        return None, _issue(
            kind, "inconsistent_add_rmsnorm_dynamic_quant_geometry",
            "The verified single-output-mask branch retains matching residual/weight "
            "geometry, optional affine input, INT8 result and per-row FP32 scales.")
    return ("add_rmsnorm_dynamic_quant_active", kind,
            (*inputs[:3], optional), (y, x, scale)), None


def quant_fusion_scope(kind):
    scope = {
        "occurrence_identity": False,
        "timing_scope": "inclusive_fused_subgraph; no_member_time_split",
        "unobserved": ["tensor_values", "scalar_dispatch_arguments", "module_or_layer_identity"],
        "not_compared": ["epsilon", "scale_values", "rounding_mode"],
    }
    if kind in {"DynamicQuant", "DynamicQuantV2"}:
        return {**scope, "kind": "dynamic_int8_tensor_descriptors",
                "compared": ["native_operator_type", "input_output_and_scale_slots",
                             *(["materialized_offset_slot"] if kind == "DynamicQuantV2" else [])]}
    if kind == "RmsNormCast":
        return {**scope, "kind": "native_tensor_slots",
                "compared": ["native_operator_type", "all_input_and_output_slots"]}
    return {
        **scope, "kind": "add_rmsnorm_dynamic_int8_descriptors",
        "compared": ["native_operator_type", "input_residual_weight_optional_affine_slots",
                     "primary_quantized_residual_and_scale_slots"],
        "not_compared": [*scope["not_compared"], "auxiliary_mask_output_values"],
    }


MOE_FUSION_TYPES = {
    "GroupedMatmulSwigluQuant", "GroupedMatmulSwigluQuantV2",
    "GroupedMatmulSituQuant", "DequantSwigluQuant", "RmsNormDynamicMxQuant",
}

# Retained native ABI arity, packed-weight rank and MX microscale block.
GROUPED_SWIGLU_INPUT_SLOTS = 5
GROUPED_SWIGLU_V2_INPUT_SLOTS = 6
GROUPED_SITU_INPUT_SLOTS = 5
GROUPED_WEIGHT_RANK = 5
DEQUANT_SWIGLU_INPUT_SLOTS = 7
DEQUANT_REQUIRED_INPUT_SLOTS = 3
MX_MICROSCALE_BLOCK = 64




def _tensor(slot, rank, dtypes, layout="ND"):
    return (len(slot[0]) == rank and all(dim > 0 for dim in slot[0])
            and slot[1] in dtypes and slot[2] == layout)


def _missing(kind, inputs, outputs, ins, outs):
    fields = [f"input[{i}].shape" for i in ins if not inputs[i][0]]
    fields += [f"output[{i}].shape" for i in outs if not outputs[i][0]]
    return ({
        "category": "capture_metadata_missing", "reason": "missing_active_tensor_shapes",
        "operator_type": kind, "missing_fields": fields,
        "detail": "The compared grouped-FFN/epilogue tensor shapes are absent.",
    } if fields else None)


def _grouped(kind, inputs, outputs):
    expected = (GROUPED_SWIGLU_INPUT_SLOTS if kind == "GroupedMatmulSwigluQuant"
                else GROUPED_SWIGLU_V2_INPUT_SLOTS)
    if len(inputs) != expected or len(outputs) != 2:
        return None, _issue(kind, "unsupported_native_tensor_slots",
                            "Only the retained single stacked expert-weight ABI is supported.")
    missing = _missing(kind, inputs, outputs, range(expected), range(2))
    if missing:
        return None, missing
    x, result, out_scale = inputs[0], outputs[0], outputs[1]
    if kind == "GroupedMatmulSwigluQuant":
        weight, weight_scale, x_scale, groups = inputs[1:]
    else:
        x_scale, groups, weight, weight_scale, assist = inputs[1:]
        if assist[0] != weight_scale[0] or assist[1:] != ("fp32", "ND"):
            return None, _issue(kind, "unsupported_grouped_assistance_geometry",
                                "The retained W4A8 implementation requires per-expert/channel assistance.")
    if (not _tensor(x, 2, {"int8"}) or not _tensor(result, 2, {"int8"})
            or x[0][0] != result[0][0] or not _tensor(groups, 1, {"int64"})
            or x_scale != ((x[0][0],), "fp32", "ND") or out_scale != x_scale
            or not _tensor(weight, GROUPED_WEIGHT_RANK, {"int8", "int4", "DT_INT4"}, "FRACTAL_NZ")
            or weight[0][0] != groups[0][0]
            or not _tensor(weight_scale, 2, {"fp32", "bf16", "fp16", "uint64", "UINT64", "DT_UINT64"})
            or weight_scale[0] != (groups[0][0], 2 * result[0][1])
            or math.prod(weight[0][1:]) != x[0][1] * 2 * result[0][1]):
        return None, _issue(
            kind, "unsupported_grouped_mlp_parameter_geometry",
            "Input/output channels, expert bank, packed weight element count and "
            "per-token/per-channel scales must describe the verified grouped SwiGLU ABI.")
    weight_dtype = "int4" if weight[1] in {"DT_INT4", "int4"} else weight[1]
    return ("grouped_mlp_parameters_active", kind,
            (groups[0][0], x[0][1], result[0][1]),
            (x[1], weight_dtype, result[1])), None


def _grouped_situ(kind, inputs, outputs):
    """Single stacked MX A8W4 ABI from the retained native source delivery."""
    if len(inputs) != GROUPED_SITU_INPUT_SLOTS or len(outputs) != 2:
        return None, _issue(
            kind, "unsupported_native_tensor_slots",
            "The verified stacked GroupedMatmulSituQuant ABI has five inputs and "
            "two outputs; dynamic per-expert tensor lists are not flattened.")
    missing = _missing(kind, inputs, outputs, range(GROUPED_SITU_INPUT_SLOTS), range(2))
    if missing:
        return None, missing
    x, x_scale, weight, weight_scale, groups = inputs
    result, out_scale = outputs
    if (not _tensor(x, 2, {"fp8_e4m3"})
            or not _tensor(result, 2, {"fp8_e4m3"})
            or not _tensor(groups, 1, {"int64"})):
        return None, _issue(
            kind, "unsupported_grouped_situ_geometry",
            "The retained MX A8W4 interface requires ND FP8 input/output and "
            "an INT64 expert-group vector.")
    rows, width = result[0]
    experts, k = groups[0][0], x[0][1]
    if (x[0][0] != rows or k % MX_MICROSCALE_BLOCK or width % MX_MICROSCALE_BLOCK
            or x_scale != ((rows, k // MX_MICROSCALE_BLOCK, 2), "fp8_e8m0", "ND")
            or weight != ((experts, 2 * width, k // 2),
                          "fp4_e2m1", "FRACTAL_NZ_C0_16")
            or weight_scale != ((experts, 2 * width, k // MX_MICROSCALE_BLOCK, 2), "fp8_e8m0", "ND")
            or out_scale != ((rows, width // MX_MICROSCALE_BLOCK, 2), "fp8_e8m0", "ND")):
        return None, _issue(
            kind, "unsupported_grouped_situ_parameter_geometry",
            "Stacked FP4 packed weight axes, paired E8M0 scales, routed row axes "
            "and the channel-halving SiTU output must agree.")
    return ("grouped_mlp_parameters_active", kind,
            (experts, k, width), ("fp8_e4m3", "mxfp4", "fp8_e4m3")), None


def _dequant_swiglu(kind, inputs, outputs):
    if len(inputs) != DEQUANT_SWIGLU_INPUT_SLOTS or len(outputs) != 2:
        return None, _issue(kind, "unsupported_native_tensor_slots",
                            "The verified dequantize/SwiGLU/quantize ABI has seven inputs and two outputs.")
    missing = _missing(kind, inputs, outputs, range(DEQUANT_REQUIRED_INPUT_SLOTS), range(2))
    if missing:
        return None, missing
    x, w_scale, x_scale = inputs[:3]
    result, scale = outputs
    if any(slot != _EMPTY for slot in inputs[3:]):
        return None, _issue(
            kind, "dequant_swiglu_optional_boundary_unsupported",
            "Bias, explicit quantization scales/offsets or grouped indices select "
            "a different epilogue boundary from the retained dynamic-quantization path.")
    if (not _tensor(x, 2, {"int32"}) or not _tensor(result, 2, {"int8"})
            or x[0] != (result[0][0], result[0][1] * 2)
            or w_scale != ((x[0][1],), "fp32", "ND")
            or x_scale != ((x[0][0],), "fp32", "ND") or scale != x_scale):
        return None, _issue(kind, "unsupported_dequant_swiglu_geometry",
                            "The verified epilogue halves the channel width and retains per-row scales.")
    return ("dequant_swiglu_quant_active", kind, x, w_scale, x_scale, result, scale), None


def _rmsnorm_mxquant(kind, inputs, outputs):
    if len(inputs) not in {2, 3} or len(outputs) != 3:
        return None, _issue(kind, "unsupported_native_tensor_slots",
                            "RMS/MX quantization requires x, gamma, optional beta and three outputs.")
    missing = _missing(kind, inputs, outputs, range(2), range(2))
    if missing:
        return None, missing
    x, gamma = inputs[:2]
    result, scale, auxiliary = outputs
    if not _tensor(x, 2, _FLOAT):
        return None, _issue(kind, "unsupported_rmsnorm_mxquant_geometry",
                            "Expected a floating ND matrix.")
    rows, width = x[0]
    pairs = (width + MX_MICROSCALE_BLOCK - 1) // MX_MICROSCALE_BLOCK
    full = len(inputs) == 3
    beta = inputs[2] if full else _EMPTY
    disabled_rstd = ((0,) if full else (), "fp32", "ND")
    if (gamma[0] != (width,) or gamma[1] not in {x[1], "fp32"} or gamma[2] != "ND"
            or beta not in {_EMPTY, gamma} or result != (x[0], "fp8_e4m3", "ND")
            or scale != ((rows, pairs, 2), "fp8_e8m0", "ND" if full else "NCL")
            or auxiliary not in {_EMPTY, disabled_rstd, ((rows, 1), "fp32", "ND")}):
        return None, _issue(kind, "unsupported_rmsnorm_mxquant_geometry",
                            "RMS weights, paired E8M0 scales and rstd must match the recorded ABI.")
    if beta != _EMPTY or auxiliary == ((rows, 1), "fp32", "ND"):
        return ("native_tensor_abi", kind, inputs, outputs), None
    return ("rmsnorm_mxquant_active", kind, x, gamma, result, scale), None


def moe_fusion_contract(native):
    _, kind, inputs, outputs = native
    if kind in {"GroupedMatmulSwigluQuant", "GroupedMatmulSwigluQuantV2"}:
        return _grouped(kind, inputs, outputs)
    if kind == "GroupedMatmulSituQuant":
        return _grouped_situ(kind, inputs, outputs)
    if kind == "DequantSwigluQuant":
        return _dequant_swiglu(kind, inputs, outputs)
    if kind == "RmsNormDynamicMxQuant":
        return _rmsnorm_mxquant(kind, inputs, outputs)
    return None, _issue(
        kind, "unsupported_moe_native_type", "No grouped FFN projection is declared.")


def moe_fusion_scope(kind):
    common = {
        "occurrence_identity": False,
        "timing_scope": "inclusive_fused_subgraph; no_member_time_split",
        "unobserved": ["tensor_values", "scalar_dispatch_arguments", "module_or_layer_identity"],
    }
    if kind in {"GroupedMatmulSwigluQuant", "GroupedMatmulSwigluQuantV2"}:
        return {
            **common, "kind": "grouped_mlp_channel_and_precision_descriptors",
            "compared": ["native_operator_type", "expert_bank_size", "input_output_channels",
                         "input_weight_output_dtypes"],
            "not_compared": ["routed_token_count", "expert_group_sizes_and_indices",
                             "weight_packing", "quantization_scale_representation",
                             "quantization_scale_values", "activation_variant_and_limit",
                             "transpose_and_dequantization_flags"],
            "excluded_semantic_work": ["routing", "expert_down_projection", "combine"],
        }
    if kind == "GroupedMatmulSituQuant":
        return {
            **common, "kind": "grouped_situ_channel_and_precision_descriptors",
            "compared": ["native_operator_type", "expert_bank_size", "input_output_channels",
                         "input_weight_output_dtypes"],
            "validated_observed": ["stacked_FP4_NZ_C0_16_packed_axes",
                                   "input_weight_output_paired_E8M0_scales",
                                   "equal_input_output_row_axes"],
            "not_compared": ["routed_token_count", "expert_group_sizes_and_indices",
                             "group_list_type", "beta", "linear_beta", "scale_values",
                             "native_binary_or_tiling"],
            "work_assumption": (
                "A channel-domain match is conditional on the independently declared "
                "routing workload; it does not establish observed expert rows or weight reads."),
            "excluded_semantic_work": ["input_quantization", "routing",
                                       "expert_down_projection", "combine"],
        }
    if kind == "DequantSwigluQuant":
        return {
            **common, "kind": "dequant_swiglu_quant_tensor_descriptors",
            "compared": ["native_operator_type", "input_output_slots", "weight_and_token_scale_slots"],
            "not_compared": ["activation_side", "activation_variant_and_limit", "rounding_mode",
                             "quantization_scale_values"],
            "excluded_semantic_work": ["input_matrix_multiplication"],
        }
    if kind == "RmsNormDynamicMxQuant":
        return {
            **common, "kind": "rmsnorm_mxquant_tensor_descriptors",
            "compared": ["native_operator_type", "input_weight_output_and_scale_slots",
                         "materialized_beta_and_rstd_descriptors_when_present"],
            "not_compared": ["epsilon", "rounding_mode", "scale_values"],
        }
    return {
        **common, "kind": "native_implementation_boundary_only", "compared": [],
        "not_compared": ["native_implementation_abi", "active_tensor_geometry"],
    }
