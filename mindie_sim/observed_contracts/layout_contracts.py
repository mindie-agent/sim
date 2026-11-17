"""Observed contracts for native normalization and transposed batch-matrix layouts.

[Role]
  Project captured normalization and transposed batch-matrix descriptors into
  partial contracts that preserve the observed precision and batch axes.

[Invariants]
  * Rank, slot and dtype checks precede any partial semantic projection.
  * Physical permutations, scalar parameters and auxiliary values stay
    unobserved rather than inferred from a typed geometry match.
  * A support gap is reported instead of inventing native dispatch.

[Timeouts & Locks]
  Bounded per-event tuple arithmetic; no waits or locks.

[Side-Effect Policy]
  Pure metadata projection; no source or capture I/O.
"""
from __future__ import annotations

from .abi import native_contract_issue, native_tensor_contract


LAYOUT_BODY_TYPES = {"LayerNormV3", "TransposeBatchMatMul"}
_FLOAT = {"bf16", "fp16", "fp32"}


def _issue(kind, reason, detail):
    return {"category": "contract_support_gap", "reason": reason,
            "detail": detail, "operator_type": kind}


def _project(row):
    issue = native_contract_issue(row)
    if issue is not None:
        return None, issue
    _, kind, inputs, outputs = native_tensor_contract(row)
    if kind == "LayerNormV3":
        if len(inputs) != 3 or len(outputs) != 3:
            return None, _issue(kind, "unsupported_native_tensor_slots",
                                "LayerNormV3 requires x, gamma, beta and y, mean, rstd slots.")
        x, gamma, beta = inputs
        if (len(x[0]) != 2 or not all(dim > 0 for dim in x[0])
                or x[1] not in _FLOAT or x[2] != "ND"
                or gamma[0] != (x[0][-1],) or gamma != beta
                or gamma[1] not in _FLOAT or gamma[2] != "ND" or outputs[0] != x
                or any(slot[0] != (x[0][0], 1) or slot[1:] != ("fp32", "ND")
                       for slot in outputs[1:])):
            return None, _issue(kind, "unsupported_layernorm_geometry",
                                "Verified LayerNorm has ND rows, matching affine channels, "
                                "a primary result and one FP32 mean/rstd per row.")
        return ("layernorm_active", kind, x, gamma, beta, outputs[0]), None
    if len(inputs) != 2 or len(outputs) != 1:
        return None, _issue(kind, "unsupported_native_tensor_slots",
                            "The retained TransposeBatchMatMul boundary has two inputs and one output.")
    x, weight = inputs
    y = outputs[0]
    if (any(len(slot[0]) != 3 or not all(dim > 0 for dim in slot[0])
            or slot[1] not in _FLOAT or slot[2] != "ND"
            for slot in (x, weight, y))):
        return None, _issue(kind, "unsupported_batch_matrix_geometry",
                            "The supported projection needs complete rank-three floating ND operands.")
    batch, k, n = weight[0]
    rows = y[0][0]
    if y[0] != (rows, batch, n) or x[0] not in {
        (rows, batch, k), (batch, rows, k),
    }:
        return None, _issue(kind, "inconsistent_batch_matrix_axes",
                            "Per-head weights and the input/output axes do not describe "
                            "a retained transposed batch matrix branch.")
    return ("transpose_batchmatmul_active", kind, (batch, rows, n, k),
            x[1], weight[1], y[1]), None


def observed_layout_contract(row):
    return _project(row)[0] if row.kind in LAYOUT_BODY_TYPES else None


def layout_contract_diagnostic(row):
    contract, issue = _project(row)
    if issue is not None:
        return issue
    return {"category": "supported_contract", "reason": "native_active_projection",
            "detail": "Native tensor descriptors support the declared partial comparison domain.",
            "operator_type": row.kind, "contract_family": contract[0]}


def layout_contract_scope(row):
    scope = {
        "comparison_scope": ("active_primary_normalization_descriptors"
                             if row.kind == "LayerNormV3" else "canonical_batch_matrix_geometry"),
        "unobserved_fields": ["epsilon", "tensor_values", "module_or_layer_identity"]
                             if row.kind == "LayerNormV3" else
                             ["permutation_scalar_arguments", "physical_strides",
                              "module_or_layer_identity"],
        "not_compared": ["auxiliary_mean_rstd_values_and_stores"]
                        if row.kind == "LayerNormV3" else
                        ["physical_input_axis_order", "native_dispatch"],
        "observed_input_shapes": row.input_shapes,
        "observed_output_shapes": row.output_shapes,
        "semantic_limits": "A typed geometry match does not prove binary selection or occurrence.",
    }
    return scope
