"""Observed contracts for captured normalization and QKV fusion kernels.

[Role]
  Project captured native tensor descriptors for normalization, gated
  normalization and QKV-fusion kernels into active observable contracts.

[Invariants]
  * Scalar flags, tensor values and missing output slots remain unobserved.
  * A native tensor match never establishes fusion flags or layer occurrence.
  * The SituMxQuant, Kimi gated-norm and QKV slot rules stay distinct.

[Timeouts & Locks]
  Pure per-row projection; no waits or locks.

[Side-Effect Policy]
  No I/O, simulation import or device-runtime initialization.
"""
from __future__ import annotations

import math

from .abi import native_tensor_contract, native_tensor_diagnostic


NORM_GATE = "_layer_norm_fwd_1pass_kernel_npu"
KIMI_NORM_GATE = "layer_norm_gated_fwd_kernel"
QKV_NORM_ROPE = "split_qkv_rmsnorm_mrope_kernel"
FUSED_TYPES = {
    NORM_GATE, KIMI_NORM_GATE, QKV_NORM_ROPE, "GemmaRmsNorm", "AddRmsNormBias", "SituMxQuant",
}
UNOBSERVED_FIELDS = {
    NORM_GATE: ("eps", "norm_before_gate", "gate_activation", "is_rms_norm", "group_size", "strides"),
    KIMI_NORM_GATE: (
        "eps", "gate_activation", "is_rms_norm", "bias", "residual",
        "mean", "rstd", "strides", "compiled_kernel_flags",
    ),
    QKV_NORM_ROPE: (
        "eps", "mrope_sections", "is_interleaved", "partial_rope",
        "output_slot_identity", "unrecorded_output_tensors",
    ),
    "GemmaRmsNorm": ("eps", "weight_values"),
    "AddRmsNormBias": ("eps", "weight_values"),
    "SituMxQuant": ("beta", "linear_beta", "activate_left", "axis", "rounding_mode"),
}
SITU_MX_CHANNEL_DIVISOR = 128
SITU_MX_SCALE_BLOCK_WIDTH = 64
QKV_NORM_ROPE_REQUIRED_OUTPUT_SLOTS = 4


def _situ_mx_contract(inputs, outputs):
    if len(inputs) != 1 or len(outputs) != 2:
        return None
    x, (y, scales) = inputs[0], outputs
    if (len(x[0]) != 2 or x[1:] != ("bf16", "ND")
            or x[0][-1] <= 0 or x[0][-1] % SITU_MX_CHANNEL_DIVISOR):
        return None
    rows, width = x[0][0], x[0][-1] // 2
    if (rows <= 0 or y[0] != (rows, width) or y[1] not in {"fp8_e4m3", "fp8_e5m2"}
            or y[2] != "ND" or scales != ((rows, width // SITU_MX_SCALE_BLOCK_WIDTH, 2), "fp8_e8m0", "ND")):
        return None
    return "situ_mxquant_active", "SituMxQuant", x, y, scales


def _kimi_gated_rows(inputs, outputs):
    """The KDA wrapper flattens leading axes and exposes only the primary output."""
    if len(inputs) != 3 or len(outputs) != 1:
        return None
    x, gate, weight = inputs
    if (len(x[0]) < 2 or len(gate[0]) < 2 or not x[0][-1]
            or gate[0][-1] != x[0][-1] or math.prod(gate[0]) != math.prod(x[0])
            or gate[1:] != x[1:] or outputs[0] != x
            or weight != ((x[0][-1],), *x[1:])):
        return None
    rows = math.prod(x[0][:-1])
    return ("gated_norm_active_rows", KIMI_NORM_GATE,
            ((rows, x[0][-1]), *x[1:]), weight)


def _normalization_slots(kind, inputs, outputs):
    """Validate the retained native pattern; other valid exporter ABIs stay unknown."""
    if kind == NORM_GATE:
        if len(inputs) != 3 or len(outputs) != 2:
            return False
        x, weight, gate = inputs
        y, statistics = outputs
        if len(x[0]) != 2 or weight[0] != (x[0][-1],) or gate != x or y != x:
            return False
        return (weight[1:] == x[1:]
                and statistics == ((x[0][0],), "fp32", "ND"))
    if kind == "GemmaRmsNorm":
        if len(inputs) != 2 or len(outputs) != 2:
            return False
        x, weight = inputs
        y, statistics = outputs
        return (len(x[0]) == 2 and weight == ((x[0][-1],), *x[1:])
                and y == x and statistics == ((x[0][0], 1), "fp32", "ND"))
    if kind == "AddRmsNormBias":
        if len(inputs) != 4 or len(outputs) != 3:
            return False
        x, residual, weight, bias = inputs
        y, statistics, residual_output = outputs
        return (len(x[0]) == 2 and residual == x and y == x and residual_output == x
                and weight == ((x[0][-1],), *x[1:])
                and bias in {((), None, None), weight}
                and statistics == ((x[0][0], 1), "fp32", "ND"))
    return False


def observed_fused_contract(row):
    if row.kind not in FUSED_TYPES:
        return None
    contract = native_tensor_contract(row)
    if contract is None:
        return None
    _, _, inputs, outputs = contract
    if row.kind == QKV_NORM_ROPE:
        # The retained A3/A5 exports contain one output, while the wrapper
        # returns Q/K/V/gate. Q and gate even share shape, so slot identity
        # cannot be recovered from this descriptor.
        return None
    if row.kind == "SituMxQuant":
        return _situ_mx_contract(inputs, outputs)
    present = [slot for slot in (*inputs, *outputs) if slot != ((), None, None)]
    if any(slot[2] != "ND" or slot[1] not in {"bf16", "fp16", "fp32"} for slot in present):
        return None
    if row.kind == KIMI_NORM_GATE:
        return _kimi_gated_rows(inputs, outputs)
    return contract if _normalization_slots(row.kind, inputs, outputs) else None


def fused_contract_scope(row):
    if row.kind not in FUSED_TYPES:
        return None
    scope = {
        "comparison_scope": "native_tensor_slots; scalar_flags_and_tensor_values_not_observed",
        "unobserved_fields": list(UNOBSERVED_FIELDS[row.kind]),
        "basis": "captured_tensor_descriptors",
        "semantic_limits": [
            "Tensor equality does not establish compiled flags, numerical equivalence or layer occurrence.",
        ],
    }
    if row.kind == "SituMxQuant":
        scope["comparison_scope"] = (
            "active_input_output_and_MX_scale_descriptors; "
            "activation_scalar_parameters_and_quantization_mode_not_compared")
        scope["semantic_limits"].append(
            "The native kind establishes a SiTU/MX boundary; beta, linear_beta and "
            "activate_left require an independent caller declaration and are not inferred from shape.")
    elif row.kind == KIMI_NORM_GATE:
        scope["comparison_scope"] = (
            "active_row_geometry_weight_dtype_and_ND_descriptors; "
            "auxiliary_outputs_and_scalar_flags_not_compared; no_full_native_abi_equality")
        scope["semantic_limits"].append(
            "The KDA wrapper reshapes x and gate to rows by final feature dimension, "
            "restores the primary output shape, and discards mean/rstd. This comparison "
            "does not alias the Qwen normalization kernel or establish the compiled RMS/gate flags.")
        scope["compared_input_slots"] = ["x", "gate", "weight"]
        scope["compared_output_slots"] = ["primary_output"]
    elif row.kind == QKV_NORM_ROPE:
        scope["comparison_scope"] = "incomplete_native_output_slots; no_contract_equality"
        scope["required_output_slots"] = QKV_NORM_ROPE_REQUIRED_OUTPUT_SLOTS
        scope["observed_output_slots"] = len(row.output_shapes) if row.output_shapes else 0
        scope["semantic_limits"].append(
            "The referenced wrapper declares Q/K/V/gate outputs; the retained export cannot identify them.")
    elif row.kind == "AddRmsNormBias":
        scope["semantic_limits"].append(
            "The fourth slot distinguishes absent beta from a channel-wise affine beta. "
            "Caller-side weight shifts and the checkpoint configuration remain unknown.")
    elif row.kind == "GemmaRmsNorm":
        scope["semantic_limits"].append(
            "The native Gemma operator identity is retained; it is not aliased to ordinary RmsNorm.")
    return scope


def fused_contract_diagnostic(row):
    if row.kind not in FUSED_TYPES:
        return None
    descriptor = native_tensor_contract(row)
    diagnostic = native_tensor_diagnostic(row)
    if descriptor is None:
        return {
            **diagnostic,
            "operator_type": row.kind,
            "unobserved_fields": list(UNOBSERVED_FIELDS[row.kind]),
        }
    if row.kind == QKV_NORM_ROPE and len(descriptor[3]) < 4:
        return {
            "category": "capture_metadata_missing",
            "reason": "incomplete_native_tensor_slots",
            "detail": ("The QKV normalization/rotary wrapper returns Q, K, V and gate tensors, "
                       "but this capture records fewer than four outputs. Q and gate can have "
                       "the same shape; missing slots and output identity cannot be inferred."),
            "operator_type": row.kind,
            "observed_output_slots": len(descriptor[3]),
            "required_output_slots": QKV_NORM_ROPE_REQUIRED_OUTPUT_SLOTS,
            "captured_descriptor": descriptor,
            "unobserved_fields": list(UNOBSERVED_FIELDS[row.kind]),
        }
    if observed_fused_contract(row) is None:
        return {
            "category": "contract_support_gap",
            "reason": "unsupported_tensor_contract",
            "detail": "The complete captured descriptors differ from the verified native fusion pattern.",
            "operator_type": row.kind,
            "unobserved_fields": list(UNOBSERVED_FIELDS[row.kind]),
        }
    return {
        "category": "tensor_contract_available",
        "reason": "observed_tensor_contract_available",
        "detail": (
            "Active x/gate/weight/output row geometry is available; auxiliary statistics, "
            "scalar flags and tensor values are not compared."
            if row.kind == KIMI_NORM_GATE else
            "All compared native tensor slots are present; scalar flags and tensor values are not compared."),
        "operator_type": row.kind,
        "unobserved_fields": list(UNOBSERVED_FIELDS[row.kind]),
    }
