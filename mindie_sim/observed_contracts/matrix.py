"""Observed matrix and router operator contracts.

[Role] Validate observed GEMM, grouped GEMM, and MoE router contracts.
[Invariants] Exact slot, rank, and dtype verification against declared ABI.
[Timeouts & Locks] Pure in-memory checks; no I/O or locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from .vocabulary import native_tensor_contract, native_tensor_diagnostic


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


from .vocabulary import native_tensor_contract, native_tensor_diagnostic


MATRIX_BODY_TYPES = {"QuantBatchMatmulV3", "QuantBatchMatmulV4", "GroupedMatmul"}
_DTYPES = {"DT_INT4": "int4", "INT4": "int4", "DT_FLOAT4_E2M1": "mxfp4",
           "UINT64": "uint64", "DT_UINT64": "uint64"}
_MX_WEIGHT_DTYPES = {"mxfp4", "fp4_e2m1"}

# Retained native ABI arity and NZ/MX tile geometry.
QUANT_MATRIX_V3_INPUT_SLOTS = 4
QUANT_MATRIX_V4_INPUT_SLOTS = 5
GROUPED_MATRIX_FIXED_INPUT_SLOTS = 8
GROUPED_MATRIX_TOKEN_SCALE_INPUT_SLOTS = 9
MX_MICROSCALE_BLOCK = 64
MX_MICROSCALE_CEIL_OFFSET = MX_MICROSCALE_BLOCK - 1
NZ_WEIGHT_K_TILE = 16
NZ_WEIGHT_N_TILE = 32
NZ_TRANSPOSED_K_TILE = 32
NZ_TRANSPOSED_N_TILE = 16
NZ_TRANSPOSED_K_CEIL_OFFSET = NZ_TRANSPOSED_K_TILE - 1
NZ_TRANSPOSED_N_CEIL_OFFSET = NZ_TRANSPOSED_N_TILE - 1
GROUPED_MX_K_TILE = 32
GROUPED_MX_N_TILE = 16


def _mx_dtype(value):
    """One canonical MXFP4 spelling for both literal and normalized sources."""
    return "mxfp4" if value in _MX_WEIGHT_DTYPES else value


def _slots(row):
    native = native_tensor_contract(row)
    if native is None:
        return None
    return tuple(tuple((shape, _DTYPES.get(dtype, dtype), layout)
                       for shape, dtype, layout in side) for side in native[2:])




def _nd_mx_v3(row, inputs, outputs, m, n, k):
    """Retained ND MX weights have matching N-major or K-major scale axes."""
    x, weight, ws, xs = inputs
    y = outputs[0]
    groups = (k + MX_MICROSCALE_CEIL_OFFSET) // MX_MICROSCALE_BLOCK
    axes_match = ((weight[0] == (n, k) and ws[0] == (n, groups, 2))
                  or (weight[0] == (k, n) and ws[0] == (groups, n, 2)))
    if (not axes_match or x[1] != "fp8_e4m3" or weight[1] != "fp8_e4m3"
            or y[1] not in {"bf16", "fp16", "fp32"}
            or ws[1:] != ("fp8_e8m0", "ND")
            or xs != ((m, groups, 2), "fp8_e8m0", "ND")):
        return None, _issue(row, "unsupported_nd_mx_matrix_axes_or_scales",
                            "ND FP8 V3 requires matching weight/weight-scale axes "
                            "[N,K]/[N,ceil(K/64),2] or [K,N]/[ceil(K/64),N,2], "
                            "paired E8M0 token scales and a floating output.")
    return ("matrix", (m, n, k), x[1], _mx_dtype(weight[1]), y[1], xs[0], xs[1]), None


def _dense(row, inputs, outputs):
    expected = (QUANT_MATRIX_V3_INPUT_SLOTS if row.kind == "QuantBatchMatmulV3"
                else QUANT_MATRIX_V4_INPUT_SLOTS)
    accumulator = row.kind == "QuantBatchMatmulV3" and len(inputs) == 3
    if (len(inputs) != expected and not accumulator) or len(outputs) != 1:
        return None, _issue(row, "unsupported_quant_matrix_slots",
                            "Supported retained interfaces are V3 four-input dequantized or "
                            "three-input INT32 accumulator output and V4 five inputs. "
                            "Each has one output; additional epilogues are not discarded.")
    x, weight, y = inputs[0], inputs[1], outputs[0]
    if (len(x[0]) != 2 or len(y[0]) != 2 or x[0][0] != y[0][0]
            or any(v <= 0 for v in (*x[0], *y[0]))):
        return None, _issue(row, "unsupported_quant_matrix_geometry",
                            "This projection requires positive 2-D input and output with equal rows.")
    m, k = x[0]
    _, n = y[0]
    if (row.kind == "QuantBatchMatmulV3" and not accumulator
            and x[2] == weight[2] == y[2] == "ND"):
        return _nd_mx_v3(row, inputs, outputs, m, n, k)
    if x[2] != "ND" or y[2] != "ND" or weight[2] != "FRACTAL_NZ":
        return None, _issue(row, "unsupported_quant_matrix_layout",
                            "Only retained ND activations/output with source-defined "
                            "NZ weights or the separate ND FP8 paired-scale branch pass.")
    groups = (k + MX_MICROSCALE_CEIL_OFFSET) // MX_MICROSCALE_BLOCK
    if row.kind == "QuantBatchMatmulV3":
        if (k % NZ_WEIGHT_K_TILE or n % NZ_WEIGHT_N_TILE
                or weight[0] != (n // NZ_WEIGHT_N_TILE, k // NZ_WEIGHT_K_TILE,
                                 NZ_WEIGHT_K_TILE, NZ_WEIGHT_N_TILE)):
            return None, _issue(row, "inconsistent_quant_matrix_weight_axes",
                                "V3 non-transposed NZ weight axes are [N/32,K/16,16,32].")
        if accumulator:
            # shared_experts.py leaves pertoken_scale=None and requests INT32;
            # the following dequant/activation kernel owns output rescaling.
            if (x[1] == weight[1] == "int8" and y[1] == "int32"
                    and inputs[2][0] == (n,) and inputs[2][1] in {"bf16", "fp32"}
                    and inputs[2][2] == "ND"):
                return ("matrix", (m, n, k), "int8", "int8", "int32", (), ""), None
            return None, _issue(
                row, "unsupported_quant_matrix_accumulator_boundary",
                "The three-input branch requires INT8 operands, an N-wide weight-scale "
                "descriptor and INT32 output. A missing token scale does not imply a "
                "complete floating-point dequantization.")
        ws, xs = inputs[2:4]
        if x[1] == weight[1] == "int8":
            valid = (ws[0] == (n,) and ws[1] in {"bf16", "fp32"} and ws[2] == "ND"
                     and xs == ((m,), "fp32", "ND")
                     and (y[1] == "bf16" or y[1] == "fp16" and ws[1] == "fp32"))
        elif x[1] == weight[1] == "fp8_e4m3":
            valid = (ws == ((groups, n, 2), "fp8_e8m0", "ND")
                     and xs == ((m, groups, 2), "fp8_e8m0", "ND")
                     and y[1] in {"bf16", "fp16", "fp32"})
        else:
            valid = False
    else:
        xs, ws, bias = inputs[2:5]
        if bias[0]:
            return None, _issue(row, "quant_matrix_bias_boundary_not_declared",
                                "The observed V4 bias epilogue requires a corresponding declared program.")
        if weight[0] != ((k + NZ_TRANSPOSED_K_CEIL_OFFSET) // NZ_TRANSPOSED_K_TILE,
                         (n + NZ_TRANSPOSED_N_CEIL_OFFSET) // NZ_TRANSPOSED_N_TILE,
                         NZ_TRANSPOSED_N_TILE, NZ_TRANSPOSED_K_TILE):
            return None, _issue(row, "inconsistent_quant_matrix_weight_axes",
                                "The retained transposed NZ weight has [ceil(K/32),ceil(N/16),16,32] axes.")
        valid = (
            x[1] == "fp8_e4m3" and weight[1] in _MX_WEIGHT_DTYPES
            and xs == ((m, groups, 2), "fp8_e8m0", "ND")
            and ws == ((n, groups, 2), "fp8_e8m0", "ND")
            and y[1] in {"bf16", "fp16"}
            and bias[1] in {None, y[1]} and bias[2] in {None, "ND"})
    if not valid:
        return None, _issue(row, "unsupported_quant_matrix_scale_or_precision",
                            "Operand/output precision and both scale descriptors must satisfy "
                            "the retained INT8 per-token/per-channel or MX paired-scale interface.")
    return ("matrix", (m, n, k), x[1], _mx_dtype(weight[1]), y[1], xs[0], xs[1]), None


def _grouped(row, inputs, outputs):
    if (len(inputs) not in {GROUPED_MATRIX_FIXED_INPUT_SLOTS,
                            GROUPED_MATRIX_TOKEN_SCALE_INPUT_SLOTS}
            or len(outputs) != 1):
        return None, _issue(row, "unsupported_grouped_matrix_slots",
                            "Only one activation, one expert bank and one output are supported; "
                            "dynamic multi-tensor lists are not flattened.")
    x, weight, bias, scale, offset, anti_scale, anti_offset, group_list = (
        inputs[:GROUPED_MATRIX_FIXED_INPUT_SLOTS])
    y = outputs[0]
    if (len(x[0]) != 2 or len(y[0]) != 2 or x[0][0] != y[0][0]
            or any(v <= 0 for v in (*x[0], *y[0])) or x[2] != y[2] or x[2] != "ND"
            or len(group_list[0]) != 1 or group_list[0][0] <= 0
            or group_list[1:] != ("int64", "ND")):
        return None, _issue(row, "unsupported_grouped_matrix_geometry",
                            "Grouped rows require ND input/output and an explicit INT64 expert-group vector.")
    m, k = x[0]
    n, experts = y[0][1], group_list[0][0]
    xs = (inputs[GROUPED_MATRIX_FIXED_INPUT_SLOTS]
          if len(inputs) == GROUPED_MATRIX_TOKEN_SCALE_INPUT_SLOTS else ((), None, None))
    mxfp4 = x[1] == "fp8_e4m3" and weight[1] in _MX_WEIGHT_DTYPES
    if offset[0] or anti_offset[0] or anti_scale[0] and not mxfp4:
        return None, _issue(row, "grouped_matrix_antiquant_boundary_not_declared",
                            "Materialized offset/antiquant interfaces are separate semantic branches.")
    if mxfp4:
        # The A5 W4A8 MX wrapper routes E8M0 weight microscales through the
        # antiquant_scale input; it is not an affine offset/correction branch.
        groups = (k + MX_MICROSCALE_CEIL_OFFSET) // MX_MICROSCALE_BLOCK
        valid = (
            k % MX_MICROSCALE_BLOCK == n % GROUPED_MX_N_TILE == 0
            and weight[0] == (experts, k // GROUPED_MX_K_TILE, n // GROUPED_MX_N_TILE,
                              GROUPED_MX_N_TILE, GROUPED_MX_K_TILE)
            and weight[2] == "FRACTAL_NZ" and not bias[0] and not scale[0]
            and anti_scale == ((experts, n, groups, 2), "fp8_e8m0", "NCHW")
            and xs == ((m, groups, 2), "fp8_e8m0", "NCL") and y[1] == "bf16")
    elif weight[0] == (experts, k, n) and weight[2] == "FRACTAL_NZ":
        if x[1] == weight[1] == "int8":
            valid = (not bias[0] and scale == ((experts, n), "bf16", "ND")
                     and xs == ((m,), "fp32", "ND") and y[1] == "bf16")
        elif x[1] == "int8" and weight[1] == "int4":
            valid = (bias == ((experts, n), "fp32", "ND")
                     and scale == ((experts, 1, n), "uint64", "NCL")
                     and xs == ((m,), "fp32", "ND") and y[1] == "bf16")
        else:
            valid = False
    elif (weight[0] == (experts, k, n) and weight[2] == "NCL"
          and x[1] == weight[1] == y[1] == "bf16"):
        valid = not any(slot[0] for slot in (bias, scale, xs))
    elif (weight[0] == (experts, n // NZ_WEIGHT_N_TILE, k // NZ_WEIGHT_K_TILE,
                        NZ_WEIGHT_K_TILE, NZ_WEIGHT_N_TILE)
          and not n % NZ_WEIGHT_N_TILE and not k % NZ_WEIGHT_K_TILE
          and weight[2] == "FRACTAL_NZ"
          and x[1] == weight[1] == "fp8_e4m3"):
        groups = (k + MX_MICROSCALE_CEIL_OFFSET) // MX_MICROSCALE_BLOCK
        valid = (not bias[0] and scale == ((experts, groups, n, 2), "fp8_e8m0", "NCHW")
                 and xs == ((m, groups, 2), "fp8_e8m0", "NCL") and y[1] == "bf16")
    else:
        valid = False
    if not valid:
        return None, _issue(row, "unsupported_grouped_matrix_precision_or_axes",
                            "The expert bank, grouped scale and per-token scale must match "
                            "a retained BF16, INT8, INT4, MXFP8 or MXFP8/MXFP4 interface.")
    return ("grouped_matrix", (experts, m, n, k), x[1], _mx_dtype(weight[1]), y[1],
            xs[0], xs[1] if xs[0] else ""), None


def _project(row):
    slots = _slots(row)
    if slots is None:
        return None, native_tensor_diagnostic(row)
    if row.kind == "GroupedMatmul":
        return _grouped(row, *slots)
    if row.kind in {"QuantBatchMatmulV3", "QuantBatchMatmulV4"}:
        return _dense(row, *slots)
    return None, _issue(row, "unsupported_matrix_native_type", "No matrix projection is declared.")


def observed_matrix_contract(row):
    return _project(row)[0]


def matrix_contract_diagnostic(row):
    contract, issue = _project(row)
    return issue or {
        "category": "tensor_contract_available", "reason": "observed_tensor_contract_available",
        "operator_type": row.kind,
        "detail": "Source-defined operand and scale geometry is available; dispatch and values remain unknown.",
    }


def matrix_contract_scope(row):
    grouped = row.kind == "GroupedMatmul"
    slots = _slots(row)
    accumulator = (row.kind == "QuantBatchMatmulV3" and slots is not None
                   and len(slots[0]) == 3 and len(slots[1]) == 1
                   and slots[1][0][1] == "int32")
    scope = {
        "kind": "grouped_matrix_active_geometry" if grouped else "quantized_matrix_active_geometry",
        "compared": ["logical_M_N_K", "activation_weight_output_precision",
                     "activation_scale_shape_dtype", *(["expert_bank_count"] if grouped else [])],
        "validated_observed": ["native_operator_type", "weight_storage_axes_layout",
                               "weight_scale_shape_dtype_layout", "activation_output_layout",
                               "supported_bias_or_quant_correction_branch"],
        "not_compared": ["native_binary_or_tiling", "scale_values", "scalar_flags",
                         "weight_scale_storage_representation",
                         *(["per_expert_row_counts", "active_expert_values", "group_list_values",
                            "quant_correction_bias_values"] if grouped else [])],
        "conditions": (
            "Declared routed work may use probability moments; equal total rows do not prove "
            "the captured expert histogram or active weight reads. The retained A5 MXFP4 "
            "branch uses N-major E8M0 microscales in the antiquant_scale slot, with no "
            "materialized offset or bias." if grouped else
            "The retained V4 five-slot capture orders scales before optional bias; "
            "the public ten-slot graph definition and ND-only MX documentation are not "
            "claimed as the captured NZ custom-dtype implementation identity."
            if row.kind == "QuantBatchMatmulV4" else
            "V3 INT8 per-channel/per-token and MX paired-scale interfaces are checked separately."),
        "occurrence_identity": False,
    }
    if accumulator:
        scope["output_boundary"] = "integer_accumulator"
        scope["compared"] = ["logical_M_N_K", "activation_weight_output_precision"]
        scope["not_compared"].extend(["subsequent_dequantization", "subsequent_activation"])
        scope["conditions"] = (
            "The source shared-expert path requests INT32 with pertoken_scale=None. "
            "Its separate dequantization/activation kernel is outside this boundary; "
            "the weight-scale descriptor does not establish floating-point output.")
    if (row.kind == "QuantBatchMatmulV3" and slots is not None
            and len(slots[0]) == 4 and slots[0][1][2] == "ND"):
        scope["validated_observed"].append("matching_N_major_or_K_major_weight_and_scale_axes")
        scope["conditions"] = (
            "ND FP8 V3 is projected only when weight axes and both E8M0 scale "
            "descriptors agree with the measured input/output geometry. Native "
            "transpose flags, binary identity and tiling remain unobserved.")
    if grouped:
        scope["row_count_basis"] = "recorded_tensor_extent"
        scope["required_runtime_evidence"] = [
            "group_list_type", "group_list_values", "active_rows_within_allocated_rows"]
        if slots is not None and len(slots[0]) >= 8 and slots[0][7][0] == (1,):
            scope["single_bank_condition"] = (
                "A single bank remains a grouped interface. Dense equivalence requires "
                "independent evidence that its group list covers all recorded rows; "
                "neither bank count nor matching M proves that condition.")
    return scope
