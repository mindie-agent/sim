"""Observed contracts for expert-selection router tensor geometry.

[Role]
  Project verified TopK and Hash router tensor slots into active selection
  contracts while keeping logits generation outside the observed boundary.

[Invariants]
  * Hash routing still consumes logits and returns routing weights.
  * Tensor shapes never reveal score functions, renormalization or expert
    values.
  * The retained fifth Hash input is accepted only when explicitly undefined.

[Timeouts & Locks]
  Pure bounded per-row metadata operations; no locks.

[Side-Effect Policy]
  No I/O, simulation imports, measurements or mutations.
"""
from __future__ import annotations

from .abi import native_tensor_contract, native_tensor_diagnostic


ROUTER_TYPES = {"MoeGatingTopK", "MoeGatingTopKHash"}
_EMPTY = ((), None, None)
TOPK_ROUTER_INPUT_SLOT_ARITIES = (2,)
HASH_ROUTER_INPUT_SLOT_ARITIES = (4, 5)


def _issue(row, reason, detail, *, missing=False):
    return {
        "category": "capture_metadata_missing" if missing else "contract_support_gap",
        "operator_type": row.kind, "reason": reason, "detail": detail,
    }


def _project(row):
    native = native_tensor_contract(row)
    if native is None:
        return None, native_tensor_diagnostic(row)
    _, kind, inputs, outputs = native
    expected = (TOPK_ROUTER_INPUT_SLOT_ARITIES if kind == "MoeGatingTopK"
                else HASH_ROUTER_INPUT_SLOT_ARITIES)
    if kind not in ROUTER_TYPES or len(inputs) not in expected or len(outputs) != 3:
        return None, _issue(row, "unsupported_router_native_slots",
                            "TopK requires 2/3 slots; Hash supports the verified 4/3 "
                            "interface and the retained 5/3 form with an undefined extra input.")
    if len(inputs) == 5 and inputs[4] != _EMPTY:
        return None, _issue(row, "unverified_router_extra_input",
                            "A materialized fifth Hash input has no verified source interpretation.")
    logits, bias = inputs[:2]
    weights, indices, auxiliary = outputs
    if not logits[0] or not weights[0] or not indices[0]:
        return None, _issue(row, "missing_router_active_shapes",
                            "Logits, selected weights and selected IDs require captured shapes.",
                            missing=True)
    if (len(logits[0]) != 2 or len(weights[0]) != 2 or len(indices[0]) != 2
            or any(dim <= 0 for slot in (logits, weights, indices) for dim in slot[0])
            or logits[1] not in {"bf16", "fp16", "fp32"} or logits[2] != "ND"
            or weights[0][0] != logits[0][0] or weights[0][1] > logits[0][1]
            or weights[1:] != logits[1:] or indices != (weights[0], "int32", "ND")
            or bias not in {_EMPTY, ((logits[0][1],), logits[1], "ND")}
            or auxiliary != (logits[0], "fp32", "ND")):
        return None, _issue(row, "unsupported_router_active_geometry",
                            "The verified interface preserves token rows and score dtype, "
                            "with INT32 selected IDs, optional per-expert bias and FP32 auxiliary scores.")
    mode = "score_topk"
    if kind == "MoeGatingTopKHash":
        token_ids, table = inputs[2:4]
        if token_ids == table == _EMPTY:
            pass
        elif (token_ids[0] == (logits[0][0],) and token_ids[1] in {"int32", "int64"}
              and token_ids[2] == "ND" and len(table[0]) == 2 and table[0][0] > 0
              and table[0][1] == weights[0][1] and table[1] in {"int32", "int64"}
              and table[2] == "ND" and bias == _EMPTY):
            mode = "hash_table"
        else:
            return None, _issue(row, "unsupported_router_hash_inputs",
                                "Hash token IDs and the token-to-expert table must occur together "
                                "with the same selected-expert width and no correction bias.")
    return ("router_selection_active", kind, mode, logits, weights, indices), None


def observed_router_contract(row):
    return _project(row)[0] if row.kind in ROUTER_TYPES else None


def router_contract_diagnostic(row):
    contract, issue = _project(row)
    return issue or {
        "category": "tensor_contract_available", "reason": "observed_tensor_contract_available",
        "operator_type": row.kind,
        "detail": "Router active descriptors are available; scalar semantics remain unobserved.",
    }


def router_contract_scope(row):
    return {
        "kind": "router_active_selection_descriptors",
        "compared": ["native_operator_type", "hash_table_presence", "logits_shape_dtype_layout",
                     "selected_weight_shape_dtype_layout", "selected_id_shape_dtype_layout"],
        "not_compared": ["score_function", "renorm", "group_count", "k_group", "bias_values",
                         "bias_presence", "routed_scaling_factor", "eps", "out_flag",
                         "token_ids", "hash_table_capacity_dtype_and_values"],
        "normalization_scope": (
            "Hash source call uses renorm=0; any declared top-k renormalization is external"
            if row.kind == "MoeGatingTopKHash" else
            "TopK supports internal renormalization; the scalar mode is not captured"),
        "excluded_semantic_work": ["router_logits_matrix_multiplication"],
        "capture_abi_limit": (
            "The retained fifth input is undefined and excluded; no materialized extension is accepted"
            if row.kind == "MoeGatingTopKHash" else None),
        "occurrence_identity": False,
    }
