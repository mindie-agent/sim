"""Observed native tensor contracts shared by profiling consumers.

[Role] Translate complete captured tensor descriptors into comparable contracts.
[Invariants] Shape matches do not prove implementation, flags or model-layer
  identity. Missing evidence returns no contract; observed timing is not used.
[Timeouts & Locks] Pure in-memory projection; no waits or locks.
[Side-Effect Policy] No I/O or simulation dependency.
"""

from __future__ import annotations

from .abi import DTYPES, native_tensor_contract
from .fused_contracts import (
    FUSED_TYPES, fused_contract_diagnostic, fused_contract_scope, observed_fused_contract,
)
from .attention_contracts import (
    ATTENTION_CACHE_TYPES, attention_contract_diagnostic, attention_contract_scope,
    observed_attention_contract,
)
from .fusion_contracts import (
    FUSION_TYPES, fusion_contract_diagnostic, fusion_contract_scope, observed_fusion_contract,
)
from .layout_contracts import (
    LAYOUT_BODY_TYPES, layout_contract_diagnostic, layout_contract_scope, observed_layout_contract,
)
from .router_contracts import (
    ROUTER_TYPES, router_contract_diagnostic, router_contract_scope, observed_router_contract,
)
from .matrix_contracts import (
    MATRIX_BODY_TYPES, matrix_contract_diagnostic, matrix_contract_scope, observed_matrix_contract,
)

GDN_NATIVE_TYPES = {"ChunkGatedDeltaRuleFwdPrepare", "ChunkFwdH", "ChunkFwdO"}
MATRIX_TYPES = {"MatMulV2", "MatMulV3"}
SWIGLU_TYPES = {"Swiglu", "SwiGLU", "SwiGlu"}
NATIVE_BODY_TYPES = (FUSION_TYPES | FUSED_TYPES | ATTENTION_CACHE_TYPES
                     | LAYOUT_BODY_TYPES | ROUTER_TYPES | MATRIX_BODY_TYPES)
SUPPORTED_TYPES = GDN_NATIVE_TYPES | MATRIX_TYPES | SWIGLU_TYPES | NATIVE_BODY_TYPES | {
    "CausalConv1d", "DynamicMxQuant", "AddRmsNormDynamicMxQuant",
    "fused_gdn_gating_kernel", "Slice",
    "RmsNorm", "AddRmsNorm", "Relu",
}
_ABSENT = {"", "N/A", "NULL", "DT_UNDEFINED"}

MX_SCALE_BLOCK_ELEMENTS = 64
MX_SCALE_BLOCK_ROUNDING_OFFSET = 63


def _mx_quant_contract(row):
    native = native_tensor_contract(row)
    if native is None:
        return None
    inputs, outputs = native[2:]
    residual = row.kind == "AddRmsNormDynamicMxQuant"
    if len(inputs) != (3 if residual else 1) or len(outputs) not in (
        {3, 4} if residual else {2}
    ):
        return None
    shape, dtype, layout = inputs[0]
    if (len(shape) != 2 or min(shape) <= 0 or layout != "ND"
            or dtype not in {"bf16", "fp16", "fp32"}):
        return None
    m, k = shape
    scale_index = 2 if residual else 1
    if (outputs[0] != (shape, "fp8_e4m3", "ND")
            or outputs[scale_index] != (
                (m, (k + MX_SCALE_BLOCK_ROUNDING_OFFSET) // MX_SCALE_BLOCK_ELEMENTS, 2),
                "fp8_e8m0", "NCL" if residual else "ND")):
        return None
    if residual:
        if (inputs[1] != inputs[0] or inputs[2] != ((k,), dtype, "ND")
                or outputs[1] != inputs[0]):
            return None
        # The retained fused ABI has an optional, typed-empty rstd slot. It
        # must not hide an unvalidated materialized auxiliary output.
        if len(outputs) == 4 and outputs[3] not in {
            ((), "fp32", "ND"), ((), None, None),
        }:
            return None
    return row.kind, shape, dtype, outputs[0][1], outputs[scale_index][0], "fp8_e8m0"


def observed_contract(row):
    if row.kind in MATRIX_BODY_TYPES:
        return observed_matrix_contract(row)
    if row.kind in ROUTER_TYPES:
        return observed_router_contract(row)
    if row.kind in LAYOUT_BODY_TYPES:
        return observed_layout_contract(row)
    if row.kind in FUSION_TYPES:
        return observed_fusion_contract(row)
    if row.kind in FUSED_TYPES:
        return observed_fused_contract(row)
    if row.kind in ATTENTION_CACHE_TYPES:
        return observed_attention_contract(row)
    if row.kind in {"DynamicMxQuant", "AddRmsNormDynamicMxQuant"}:
        return _mx_quant_contract(row)
    inputs, outputs = row.input_shapes, row.output_shapes
    if not inputs or not outputs:
        return None
    if len(row.input_dtypes) < len(inputs) or len(row.output_dtypes) < len(outputs):
        return None
    if any(shape and dtype in {"", "N/A", "NULL", "DT_UNDEFINED"}
           for shapes, dtypes in ((inputs, row.input_dtypes), (outputs, row.output_dtypes))
           for shape, dtype in zip(shapes, dtypes)):
        return None
    if row.kind in GDN_NATIVE_TYPES:
        return (row.kind, inputs, outputs, tuple(DTYPES.get(d, d) for d in row.input_dtypes),
                tuple(DTYPES.get(d, d) for d in row.output_dtypes))
    if row.kind == "CausalConv1d":
        if len(inputs) < 7 or len(inputs[0]) != 2 or len(inputs[1]) != 2:
            return None
        if inputs[3] and len(inputs[3]) != 3:
            return None
        shapes = (*inputs[:3], inputs[3][1:] if inputs[3] else (), *inputs[4:7])
        return (row.kind, shapes, outputs, tuple(DTYPES.get(d, d) for d in row.input_dtypes[:7]),
                tuple(DTYPES.get(d, d) for d in row.output_dtypes))
    if row.kind in MATRIX_TYPES:
        if len(inputs) < 2 or len(inputs[0]) != 2 or len(outputs) != 1 or len(outputs[0]) != 2:
            return None
        dtypes = tuple(DTYPES.get(value) for value in (*row.input_dtypes[:2], row.output_dtypes[0]))
        if None in dtypes:
            return None
        m, k = inputs[0]
        om, n = outputs[0]
        if om != m:
            return None
        formats = row.input_formats
        if (len(inputs) > 3 or inputs[1] != (n, k)
                or formats[:2] != ("ND", "ND") or row.output_formats != ("ND",)):
            return None
        if len(inputs) == 3 and inputs[2]:
            bias_dtype = DTYPES.get(row.input_dtypes[2])
            if (inputs[2] != (n,) or formats[2:] != ("ND",)
                    or bias_dtype is None):
                return None
            return ("matrix_bias", (m, n, k), *dtypes, inputs[2], bias_dtype)
        scale = (), ""
        return ("matrix", (m, n, k), *dtypes, *scale)
    if row.kind == "fused_gdn_gating_kernel" and len(inputs) >= 3:
        return ("gdn_gating", inputs[1])
    if (row.kind == "Slice" and len(inputs[0]) in {2, 3}
            and len(outputs) == 1 and len(outputs[0]) == len(inputs[0])):
        # Captured geometry does not include index tensor values or source offsets.
        # Preserve rank so an equivalent-element view cannot become an exact match.
        if any(out <= 0 or out > source for source, out in zip(inputs[0], outputs[0])):
            return None
        dtypes = DTYPES.get(row.input_dtypes[0]), DTYPES.get(row.output_dtypes[0])
        return None if None in dtypes else ("slice_copy", inputs[0], outputs[0], *dtypes)
    if row.kind in {"RmsNorm", "AddRmsNorm"}:
        residual = row.kind == "AddRmsNorm"
        base_inputs = 2 if residual else 1
        if len(inputs) not in {base_inputs, base_inputs + 1} or not inputs[0]:
            return None
        dtype = DTYPES.get(row.input_dtypes[0])
        if dtype is None or outputs[0] != inputs[0] or DTYPES.get(row.output_dtypes[0]) != dtype:
            return None
        if residual and (inputs[1] != inputs[0] or DTYPES.get(row.input_dtypes[1]) != dtype):
            return None
        has_weight = len(inputs) > base_inputs and bool(inputs[-1])
        if has_weight and (inputs[-1] != (inputs[0][-1],) or DTYPES.get(row.input_dtypes[-1]) != dtype):
            return None
        if len(row.input_formats) < len(inputs) or len(row.output_formats) < len(outputs):
            return None
        if any(value != "ND" for value in (*row.input_formats, *row.output_formats)):
            return None
        return ("add_rmsnorm" if residual else "rmsnorm", inputs[0], dtype, has_weight)
    if row.kind == "Relu":
        if len(inputs) != 1 or len(outputs) != 1 or not inputs[0]:
            return None
        dtype = DTYPES.get(row.input_dtypes[0])
        if (dtype is None or outputs[0] != inputs[0]
                or DTYPES.get(row.output_dtypes[0]) != dtype
                or row.input_formats != ("ND",) or row.output_formats != ("ND",)):
            return None
        return ("relu", inputs[0], dtype)
    if row.kind in SWIGLU_TYPES:
        if len(inputs) != 1 or len(outputs) != 1 or not inputs[0] or not outputs[0]:
            return None
        dtype = DTYPES.get(row.input_dtypes[0])
        if (dtype is None or DTYPES.get(row.output_dtypes[0]) != dtype
                or inputs[0][:-1] != outputs[0][:-1] or inputs[0][-1] != 2 * outputs[0][-1]
                or row.input_formats != ("ND",) or row.output_formats != ("ND",)):
            return None
        return ("swiglu", outputs[0], dtype)
    return None


def observed_contract_diagnostic(row):
    """Explain evidence/support gaps without guessing a declaration or occurrence."""
    missing, unparsed = [], []
    event = getattr(row, "event", None)
    for field in ("input_shapes", "output_shapes"):
        if not getattr(row, field):
            raw = str(getattr(event, field, "")).strip().strip('"')
            (missing if raw in _ABSENT else unparsed).append(field)
    if unparsed:
        return {
            "category": "parser_contract_gap", "reason": "unparsed_tensor_shapes",
            "detail": "The source has nonempty tensor descriptors that normalization cannot parse.",
            "fields": unparsed, "operator_type": row.kind,
        }
    if row.kind in MATRIX_BODY_TYPES:
        return matrix_contract_diagnostic(row)
    if row.kind in ROUTER_TYPES:
        return router_contract_diagnostic(row)
    if row.kind in LAYOUT_BODY_TYPES:
        return layout_contract_diagnostic(row)
    if row.kind in FUSION_TYPES:
        return fusion_contract_diagnostic(row)
    if row.kind in FUSED_TYPES:
        return fused_contract_diagnostic(row)
    if row.kind in ATTENTION_CACHE_TYPES:
        return attention_contract_diagnostic(row)
    for direction in ("input", "output"):
        shapes = getattr(row, f"{direction}_shapes") or ()
        dtypes = getattr(row, f"{direction}_dtypes")
        if len(dtypes) < len(shapes) or any(
                shape and dtype in _ABSENT for shape, dtype in zip(shapes, dtypes)):
            missing.append(f"{direction}_dtypes")
        if row.kind in MATRIX_TYPES | SWIGLU_TYPES | {
            "RmsNorm", "AddRmsNorm", "Relu", "DynamicMxQuant", "AddRmsNormDynamicMxQuant",
        }:
            formats = getattr(row, f"{direction}_formats")
            if len(formats) < len(shapes) or any(
                    shape and layout in _ABSENT for shape, layout in zip(shapes, formats)):
                missing.append(f"{direction}_formats")
    if missing:
        return {
            "category": "capture_metadata_missing", "reason": "missing_tensor_metadata",
            "detail": "The captured tensor fields needed by this contract are absent or incomplete.",
            "missing_fields": missing, "operator_type": row.kind,
        }
    if row.kind not in SUPPORTED_TYPES:
        return {
            "category": "contract_support_gap", "reason": "unsupported_operator_contract",
            "detail": "Tensor descriptors are present, but this operator type has no observed contract.",
            "operator_type": row.kind,
        }
    if observed_contract(row) is None:
        return {
            "category": "contract_support_gap", "reason": "unsupported_tensor_contract",
            "detail": ("The operator type is supported, but its captured geometry, dtype or layout "
                       "does not satisfy the implemented tensor contract."),
            "operator_type": row.kind,
        }
    return {
        "category": "tensor_contract_available", "reason": "observed_tensor_contract_available",
        "detail": "Captured tensor descriptors form a contract; this does not identify an occurrence.",
        "operator_type": row.kind,
    }


def observed_contract_scope(row):
    """Expose the comparison boundary independently of successful attribution."""
    if row.kind in MATRIX_BODY_TYPES:
        return matrix_contract_scope(row)
    if row.kind in ROUTER_TYPES:
        return router_contract_scope(row)
    if row.kind in LAYOUT_BODY_TYPES:
        return layout_contract_scope(row)
    if row.kind in FUSION_TYPES:
        return fusion_contract_scope(row)
    if row.kind in FUSED_TYPES:
        return fused_contract_scope(row)
    if row.kind in ATTENTION_CACHE_TYPES:
        return attention_contract_scope(row)
    return {
        "comparison_scope": "captured_tensor_contract",
        "basis": "captured_tensor_descriptors",
        "semantic_limits": [
            "Tensor equality alone does not establish layer occurrence or implementation flags.",
        ],
    }
