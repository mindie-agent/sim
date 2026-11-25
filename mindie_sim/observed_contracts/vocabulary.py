"""Observed contract vocabulary, carriers, ABI helpers, targets and layout rules.

[Role] Own canonical observed operator vocabulary, contract error types, ABI
  carriers, target resolution, and layout compatibility checks.
[Invariants] Shared SSOT across simulator and profiler contract verification.
[Timeouts & Locks] Pure in-memory contract declarations; no I/O or locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

class StructureTargetError(ValueError):
    """A declared structure carrier is not a valid target set."""


from collections.abc import Mapping as _AbcMapping
from typing import Any


CARRIER_MAX_NESTING_DEPTH = 64
CARRIER_INLINE_LEAF_FLOOR = (1 << 2) + 2
CARRIER_INLINE_LEAF_DEPTH = 1 << CARRIER_INLINE_LEAF_FLOOR

_SCALAR_TYPES = (str, int, float, bool, type(None))


def _snapshot_structure_carrier(value: Any, name: str = "structure metadata", _depth: int = 0) -> Any:
    """Copy one carrier into plain str-keyed containers with bounded depth."""
    if _depth > CARRIER_MAX_NESTING_DEPTH:
        raise ProfilerUsageError(f"{name} exceeds container nesting limit")
    vtype = type(value)
    if vtype in _SCALAR_TYPES:
        return value
    inline_leaf = _depth < CARRIER_INLINE_LEAF_DEPTH
    if vtype is dict:
        out = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ProfilerUsageError(f"{name} keys must be plain strings")
            out[key] = (
                item if inline_leaf and type(item) in _SCALAR_TYPES
                else _snapshot_structure_carrier(item, name, _depth + 1)
            )
        return out
    if vtype is list or vtype is tuple:
        if inline_leaf:
            return [
                item if type(item) in _SCALAR_TYPES
                else _snapshot_structure_carrier(item, name, _depth + 1)
                for item in value
            ]
        return [_snapshot_structure_carrier(item, name, _depth + 1) for item in value]
    if isinstance(value, _AbcMapping):
        try:
            _ = len(value)
            out = {}
            for key in value:
                if type(key) is not str:
                    raise ProfilerUsageError(f"{name} keys must be plain strings")
                out[key] = _snapshot_structure_carrier(value[key], name, _depth + 1)
            return out
        except TimeoutError:
            raise
        except ProfilerUsageError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProfilerUsageError(f"invalid {name} mapping protocol") from exc
    if isinstance(value, (list, tuple)):
        try:
            _ = len(value)
            return [_snapshot_structure_carrier(item, name, _depth + 1) for item in value]
        except TimeoutError:
            raise
        except ProfilerUsageError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProfilerUsageError(f"invalid {name} sequence protocol") from exc
    return value


FIELDS = {
    "matrix": ("family", "geometry", "input_dtype", "weight_dtype", "output_dtype",
               "scale_shape", "scale_dtype"),
    "matrix_bias": ("family", "geometry", "input_dtype", "weight_dtype", "output_dtype",
                    "bias_shape", "bias_dtype"),
    "grouped_matrix": ("family", "geometry", "input_dtype", "weight_dtype", "output_dtype",
                       "scale_shape", "scale_dtype"),
    "gdn_gating": ("family", "shape", "input_slots", "output_slots"),
    "slice_copy": ("family", "input_shape", "output_shape", "input_dtype", "output_dtype"),
    "rmsnorm": ("family", "shape", "dtype", "has_weight"),
    "add_rmsnorm": ("family", "shape", "dtype", "has_weight"),
    "DynamicMxQuant": ("family", "shape", "input_dtype", "output_dtype", "scale_shape",
                       "scale_dtype"),
    "AddRmsNormDynamicMxQuant": (
        "family", "shape", "input_dtype", "output_dtype", "scale_shape", "scale_dtype"),
    "native_tensor_abi": ("family", "native_operator", "input_slots", "output_slots"),
    "attention_active_qkv": ("family", "native_operator", "qkv_slots", "main_output_slots"),
    "attention_paged_active_qkv": (
        "family", "native_operator", "query_slot",
        "cache_block_size_packed_row_width_dtype_format", "main_output_slot",
        "block_table_batch_count"),
    "attention_paged_mla_qk_rope": (
        "family", "native_operator", "query_slot", "query_rope_slot",
        "cache_heads_block_size_width_dtype_format", "output_slot",
        "block_table_batch_count"),
    "kv_cache_store_active": ("family", "native_operator", "kv_slots", "slot_mapping",
                              "cache_dtypes_layouts"),
    "kv_cache_store_active_elements": ("family", "native_operator", "kv_elements_dtype",
                                       "slot_mapping_shape"),
    "sparse_attention_active": ("family", "native_operator", "query_slots",
                                "cache_head_slots", "output_slot"),
    "gated_norm_active_rows": ("family", "native_operator", "rows_slot", "weight_slot"),
    "lightning_indexer_active": ("family", "native_operator", "query_and_weight_slots",
                                 "key_cache_head_slot", "sparse_index_output_slot"),
    "mla_prolog_active": ("family", "native_operator", "token_rows_slot",
                          "parameter_shape_dtype_slots", "cache_head_slots", "output_slots"),
    "quant_lightning_indexer_active": (
        "family", "native_operator", "query_weight_scale_slots",
        "key_cache_and_scale_head_slots", "sparse_index_output_slot"),
    "hyper_connection_active": (
        "family", "native_operator", "canonical_input_slots", "primary_output_slots"),
    "compressor_active": (
        "family", "native_operator", "token_slot", "parameter_slots",
        "state_channel_slot", "rope_width_slots", "output_width_slot"),
    "recurrent_kda_parameters_active": (
        "family", "native_operator", "gate_beta_parameter_slots",
        "qkv_dtype_format", "state_dtype_format", "output_dtype_format"),
    "grouped_mlp_parameters_active": (
        "family", "native_operator", "expert_and_channel_geometry",
        "input_weight_output_dtypes"),
    "dequant_swiglu_quant_active": (
        "family", "native_operator", "input_slot", "weight_scale_slot", "token_scale_slot",
        "output_slot", "output_scale_slot"),
    "rmsnorm_mxquant_active": (
        "family", "native_operator", "input_slot", "weight_slot", "output_slot", "scale_slot"),
    "dynamic_quant_active": (
        "family", "native_operator", "input_slot", "output_slot", "scale_slot"),
    "dynamic_quant_v2_active": (
        "family", "native_operator", "input_slot", "output_slot", "scale_slot"),
    "add_rmsnorm_dynamic_quant_active": (
        "family", "native_operator", "input_residual_weight_affine_slots",
        "primary_output_slots"),
    "situ_mxquant_active": (
        "family", "native_operator", "input_slot", "output_slot", "scale_slot"),
    "causal_conv1d_active": (
        "family", "native_operator", "input_slot", "weight_slot", "output_slot"),
    "layernorm_active": (
        "family", "native_operator", "input_slot", "weight_slot", "bias_slot", "output_slot"),
    "transpose_batchmatmul_active": (
        "family", "native_operator", "batch_geometry", "input_dtype", "weight_dtype",
        "output_dtype"),
    "router_selection_active": (
        "family", "native_operator", "selection_mode", "logits_slot", "weights_slot",
        "indices_slot", "text_bias_slot", "vision_bias_slot"),
}

NATIVE_FAMILIES = {
    "native_tensor_abi", "attention_active_qkv", "attention_paged_active_qkv",
    "attention_paged_mla_qk_rope", "kv_cache_store_active",
    "kv_cache_store_active_elements",
    "sparse_attention_active", "gated_norm_active_rows",
    "lightning_indexer_active", "mla_prolog_active",
    "quant_lightning_indexer_active", "hyper_connection_active", "compressor_active",
    "recurrent_kda_parameters_active",
    "grouped_mlp_parameters_active", "dequant_swiglu_quant_active", "rmsnorm_mxquant_active",
    "dynamic_quant_active", "dynamic_quant_v2_active", "add_rmsnorm_dynamic_quant_active",
    "situ_mxquant_active", "causal_conv1d_active",
    "layernorm_active", "transpose_batchmatmul_active",
    "router_selection_active",
}

# Public aliases: the canonical vocabulary keeps the historical names above.
CONTRACT_FIELDS = FIELDS
NATIVE_CONTRACT_FAMILIES = NATIVE_FAMILIES

ONBOARDING_SCHEMA = "mindie-unmodeled-structure-hint/1"
ONBOARDING_STATUS = "unmodeled_in_sim"
SSOT_OWNER_REPO = "mindie-sim"
ONBOARDING_TARGETS = (
    "mindie_sim/structure_native_contracts.py",
    "mindie_sim/observed_contracts/attention_contracts.py",
    "mindie_sim/structure_subgraphs.py",
    "mindie_sim/subgraph_math.py",
)
AGENT_DIRECTIVE = (
    "Onboard this operator/subgraph ONCE in mindie-sim (declared contract + "
    "observed_contracts + subgraph_math). Do NOT add local operator/subgraph "
    "contract rules to mindie-profiler."
)
_CANDIDATE_TOKENS = (
    ("attention", "attention"), ("flash", "attention"), ("mla", "attention"),
    ("index", "indexer"), ("topk", "indexer"), ("moe", "moe_ffn"),
    ("expert", "moe_ffn"), ("swiglu", "dense_ffn"), ("rmsnorm", "norm_quant"),
    ("quant", "norm_quant"), ("conv", "recurrent_gdn_kda"), ("gdn", "recurrent_gdn_kda"),
    ("kda", "recurrent_gdn_kda"), ("hc", "mhc_residual"), ("residual", "mhc_residual"),
)


def infer_subgraph_candidate(observed_kernel_name=None, observed_op_type=None):
    """Infer one canonical semantic family from an observed operator name only."""
    text = " ".join(str(value) for value in (observed_op_type, observed_kernel_name) if value)
    lowered = text.casefold()
    for token, candidate in _CANDIDATE_TOKENS:
        if token in lowered:
            return candidate
    return None


def build_unmodeled_onboarding_hint(
    *,
    observed_kernel_name=None,
    observed_op_type=None,
    observed_abi_summary=None,
    inferred_subgraph_candidate=None,
):
    """Build the deterministic single-onboarding-point payload for one gap."""
    candidate = inferred_subgraph_candidate
    if candidate is None:
        candidate = infer_subgraph_candidate(observed_kernel_name, observed_op_type)
    return {
        "schema": ONBOARDING_SCHEMA,
        "status": ONBOARDING_STATUS,
        "observed_kernel_name": observed_kernel_name,
        "observed_op_type": observed_op_type,
        "observed_abi_summary": observed_abi_summary,
        "inferred_subgraph_candidate": candidate,
        "ssot_owner_repo": SSOT_OWNER_REPO,
        "onboarding_targets": list(ONBOARDING_TARGETS),
        "agent_directive": AGENT_DIRECTIVE,
    }


def observed_contract_diagnostic(row):
    """Delegate to the canonical observed diagnostic without an import cycle."""
    from .signatures import observed_contract_diagnostic as _diagnostic

    return _diagnostic(row)


def mismatch_diagnostic(contract, declared):
    """Explain a declared-contract mismatch without proposing a mapping."""
    family = contract[0]
    same_family = [
        other for other in declared if other[0] == family
        and (family not in NATIVE_FAMILIES or other[1] == contract[1])
    ]
    if not same_family:
        return {
            "category": "declared_structure_mismatch",
            "reason": "declared_operator_family_absent",
            "detail": "No declared contract in the selected rank/scope has this operator family.",
        }
    # Exact equality is handled by the mapper. These comparisons only explain
    # which declaration needs inspection; no nearest node is ever selected.
    names = FIELDS.get(family, ("family", "shape"))
    differences = []
    for other in same_family:
        fields = [names[index] if index < len(names) else f"field_{index}"
                  for index in range(max(len(contract), len(other)))
                  if index >= len(contract) or index >= len(other)
                  or contract[index] != other[index]]
        differences.append((fields, other))
    differences.sort(key=lambda pair: len(pair[0]))
    best = differences[0][0]
    same_geometry = [fields for fields, other in differences if other[1] == contract[1]]
    precision_fields = {
        "input_dtype", "weight_dtype", "output_dtype", "bias_dtype", "scale_shape",
        "scale_dtype",
    }
    if family in {"matrix", "matrix_bias", "grouped_matrix"} and any(
        fields and set(fields) <= precision_fields for fields in same_geometry
    ):
        reason = "declared_dtype_or_quantization_mismatch"
    elif any(field in best for field in ("geometry", "shape", "input_shape", "output_shape")):
        reason = "declared_geometry_or_scope_mismatch"
    else:
        reason = "declared_tensor_descriptor_mismatch"
    return {
        "category": "declared_structure_mismatch", "reason": reason,
        "detail": ("Compare the selected execution geometry, precision and scope with the "
                   "capture; tensor descriptors alone do not establish prefill/decode or "
                   "draft ownership."),
        "near_contracts": [
            {"contract": other, "differing_fields": fields}
            for fields, other in differences[:3]
        ],
    }


from typing import Any

_MISSING_TEXT = frozenset({"", "N/A", "NULL", "NONE", "UNDEFINED"})
_UNDEFINED_DTYPES = frozenset({"", "N/A", "NULL", "NONE", "UNDEFINED", "DT_UNDEFINED"})
_ATTENTION_AUX_DTYPES = frozenset(
    {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "bool"}
)
_NON_CONTIGUOUS_FORMATS = frozenset(
    {"FRACTAL_NZ", "FRACTAL_Z", "FRACTAL_Z_3D", "NC1HWC0", "NDC1HWC0", "C1HWNCOC0"}
)
_COARSE_SHAPE_KEYS = frozenset(
    {"local", "padded", "launch", "payload", "active_query", "nz_storage", "nz_logical_storage"}
)
_OUTPUT_SHAPE_KEYS = frozenset({"output", "out", "y", "outputs"})
_PRIMARY_INPUT_KEYS = ("input", "x", "lhs", "activation", "query", "q")
_TAIL_INPUT_ROLES = frozenset({"key_tail", "value_tail"})
_PARTIAL_CACHE_ROLES = frozenset({"cache_head", "rope_cache_head", "slot"})
_INPUT_ROLE_ORDER: dict[str, int] = {
    "input": 0,
    "x": 0,
    "lhs": 0,
    "activation": 0,
    "query": 0,
    "q": 0,
    "weight": 1,
    "rhs": 1,
    "w": 1,
    "key": 2,
    "k": 2,
    "key_tail": 2,
    "cache": 2,
    "value": 3,
    "v": 3,
    "value_tail": 3,
    "query_rope": 4,
    "rope_cache": 5,
    "residual": 6,
    "gamma": 7,
    "norm_weight": 7,
    "beta": 8,
    "norm_bias": 8,
    "gate": 9,
    "scale": 10,
    "weight_scale": 10,
    "bias": 11,
    "input_scale": 12,
    "activation_scale": 12,
    "group_list": 13,
    "indices": 14,
    "sparse_indices": 14,
}
_OUTPUT_ROLE_ORDER: dict[str, int] = {"output": 0, "out": 0, "y": 0}
_AUX_ONLY_ROLES = frozenset(
    {
        "group_list",
        "input_scale",
        "activation_scale",
        "scale",
        "weight_scale",
        "indices",
        "sparse_indices",
    }
)
_QUANTIZE_FAMILIES = frozenset({"quantize", "add_rmsnorm_quant", "rmsnorm_quant"})
_ROLE_DTYPE_KEYS: dict[str, tuple[str, ...]] = {
    "input": ("input_dtype", "activation_dtype", "activation_format", "x_dtype", "lhs_dtype"),
    "x": ("x_dtype", "input_dtype", "activation_dtype", "activation_format"),
    "lhs": ("lhs_dtype", "input_dtype", "activation_dtype", "activation_format"),
    "activation": ("activation_dtype", "activation_format", "input_dtype"),
    "weight": ("weight_dtype", "weight_format", "rhs_dtype", "w_dtype"),
    "rhs": ("rhs_dtype", "weight_dtype", "weight_format"),
    "w": ("w_dtype", "weight_dtype", "weight_format"),
    "bias": ("bias_dtype",),
    "residual": ("residual_dtype", "input_dtype"),
    "gate": ("gate_dtype", "input_dtype"),
    "gamma": ("gamma_dtype", "norm_parameter_dtype", "parameter_dtype"),
    "beta": ("beta_dtype", "norm_parameter_dtype", "parameter_dtype"),
    "norm_weight": ("norm_weight_dtype", "norm_parameter_dtype", "parameter_dtype"),
    "norm_bias": ("norm_bias_dtype", "norm_parameter_dtype", "parameter_dtype"),
    "query": ("query_dtype",),
    "q": ("query_dtype",),
    "query_rope": ("query_rope_dtype", "query_dtype"),
    "key": ("key_dtype", "kv_dtype", "cache_dtype"),
    "k": ("key_dtype", "kv_dtype", "cache_dtype"),
    "key_tail": ("key_dtype", "kv_dtype", "cache_dtype"),
    "value": ("value_dtype", "kv_dtype", "cache_dtype"),
    "v": ("value_dtype", "kv_dtype", "cache_dtype"),
    "value_tail": ("value_dtype", "kv_dtype", "cache_dtype"),
    "cache": ("cache_dtype", "kv_dtype", "key_dtype"),
    "cache_head": ("cache_dtype", "kv_dtype", "key_dtype"),
    "rope_cache": ("rope_cache_dtype", "cache_dtype", "key_dtype"),
    "rope_cache_head": ("rope_cache_dtype", "cache_dtype", "key_dtype"),
    "group_list": ("group_list_dtype",),
    "input_scale": ("input_scale_dtype", "activation_scale_dtype", "activation_scale_format", "scale_dtype"),
    "activation_scale": ("activation_scale_dtype", "activation_scale_format", "input_scale_dtype", "scale_dtype"),
    "scale": ("scale_dtype", "weight_scale_dtype"),
    "weight_scale": ("weight_scale_dtype", "scale_dtype"),
    "indices": ("index_dtype",),
    "sparse_indices": ("index_dtype",),
}
_DECLARED_INPUT_DTYPE_ROLES: tuple[tuple[str, frozenset[str]], ...] = (
    ("weight_dtype", frozenset({"weight", "rhs", "w"})),
    ("weight_format", frozenset({"weight", "rhs", "w"})),
    ("rhs_dtype", frozenset({"weight", "rhs", "w"})),
    ("w_dtype", frozenset({"weight", "rhs", "w"})),
    ("key_dtype", frozenset({"key", "k", "key_tail"})),
    ("value_dtype", frozenset({"value", "v", "value_tail"})),
    (
        "cache_dtype",
        frozenset(
            {
                "key",
                "k",
                "key_tail",
                "value",
                "v",
                "value_tail",
                "cache",
                "cache_head",
                "rope_cache",
                "rope_cache_head",
            }
        ),
    ),
    ("kv_dtype", frozenset({"key", "k", "key_tail", "value", "v", "value_tail", "cache", "cache_head"})),
    ("bias_dtype", frozenset({"bias"})),
    ("norm_parameter_dtype", frozenset({"gamma", "beta", "norm_weight", "norm_bias"})),
    ("parameter_dtype", frozenset({"gamma", "beta", "norm_weight", "norm_bias"})),
    ("group_list_dtype", frozenset({"group_list"})),
    ("scale_dtype", frozenset({"scale", "weight_scale", "input_scale", "activation_scale"})),
    ("weight_scale_dtype", frozenset({"scale", "weight_scale"})),
    ("input_scale_dtype", frozenset({"input_scale", "activation_scale"})),
    ("activation_scale_dtype", frozenset({"input_scale", "activation_scale"})),
    ("activation_scale_format", frozenset({"input_scale", "activation_scale"})),
    ("index_dtype", frozenset({"indices", "sparse_indices"})),
)


import hashlib
import json
from collections.abc import Mapping, Sequence

ABI_SCHEMA = "mindie-native-tensor-abi/1"

DTYPES = {
    "DT_BF16": "bf16", "DT_FLOAT16": "fp16", "FLOAT": "fp32", "INT8": "int8",
    "INT64": "int64", "INT32": "int32", "BOOL": "bool", "DT_UNDEFINED": None,
    "DT_FLOAT8_E4M3FN": "fp8_e4m3", "DT_FLOAT8_E5M2": "fp8_e5m2",
    "DT_FLOAT8_E8M0": "fp8_e8m0", "DT_FLOAT4_E2M1": "fp4_e2m1",
    "DT_INT4": "int4", "INT4": "int4", "DT_UINT4": "uint4", "UINT4": "uint4",
    "UINT64": "uint64", "DT_UINT64": "uint64", "UINT32": "uint32",
    "DT_UINT32": "uint32", "UINT8": "uint8", "DT_UINT8": "uint8",
    "BF16": "bf16", "BFP16": "bf16", "FP16": "fp16", "DT_FLOAT": "fp32",
    "HALF": "fp16", "DT_HALF": "fp16", "DOUBLE": "fp64", "DT_DOUBLE": "fp64",
    "INT16": "int16", "DT_INT16": "int16", "UINT16": "uint16", "DT_UINT16": "uint16",
    "DT_BOOL": "bool", "FP8_E4M3FN": "fp8_e4m3",
    "bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "fp64": "fp64",
    "int8": "int8", "int4": "int4", "int64": "int64", "int32": "int32",
    "uint64": "uint64", "uint32": "uint32", "uint4": "uint4", "bool": "bool",
    "FLOAT16": "fp16", "DT_INT8": "int8", "DT_INT32": "int32",
    "DT_INT64": "int64", "DT_UINT1": "uint1", "UNDEFINED": None,
    "bfloat16": "bf16", "bfloat16_t": "bf16", "float16": "fp16",
    "float32": "fp32", "float64": "fp64", "float8_e4m3fn": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3", "float8_e5m2": "fp8_e5m2", "fp8_e5m2": "fp8_e5m2",
    "fp8_e8m0": "fp8_e8m0", "fp8_e8m0fnu": "fp8_e8m0", "e8m0": "fp8_e8m0",
    "fp4_e2m1": "fp4_e2m1", "dt_float4_e2m1": "fp4_e2m1", "fp4": "fp4_e2m1",
    "mxfp4": "fp4_e2m1", "e2m1": "fp4_e2m1",
    "dt_float8_e4m3fn": "fp8_e4m3", "dt_float8_e8m0": "fp8_e8m0",
    "dt_int4": "int4", "uint8": "uint8", "uint1": "uint1",
    "DT_BFLOAT16": "bf16", "DT_FLOAT32": "fp32", "DT_FLOAT64": "fp64",
    "DT_HIFLOAT8": "hif8", "HIFLOAT8": "hif8", "HIF8": "hif8",
    "hifloat8": "hif8", "hif8": "hif8",
    "FLOAT4_E2M1": "fp4_e2m1", "FP4_E2M1": "fp4_e2m1",
    "MXFP4": "fp4_e2m1", "float4_e2m1": "fp4_e2m1",
}
_MISSING = {"", "N/A", "NULL", "UNDEFINED", "DT_UNDEFINED"}
_EMPTY_DTYPES = {"UNDEFINED", "DT_UNDEFINED"}
_EMPTY_FORMATS = {"NULL", "UNDEFINED"}


def _missing_layout(value):
    return not isinstance(value, str) or value.strip().upper() in _MISSING


def _undefined_dtype(value):
    """True only for an explicitly declared undefined dtype.

    ``None`` is a present JSON null (an explicit undefined declaration).
    A blank/whitespace string or ``N/A`` is a missing field, not a
    declaration: ``TensorSlot.state`` already reports it as incomplete, and
    the ABI must preserve that gap instead of upgrading it to undefined.
    """
    if value is None:
        return True
    return str(value).strip().upper() in {"UNDEFINED", "DT_UNDEFINED", "NULL"}


def _undefined_layout(value):
    """True only for an explicitly declared undefined layout spelling."""
    if value is None:
        return True
    return str(value).strip().upper() in {"NULL", "UNDEFINED", "DT_UNDEFINED"}


def native_contract_issue(row):
    """Return a concrete missing/unsupported descriptor reason, or None."""
    for direction in ("input", "output"):
        shapes = getattr(row, f"{direction}_shapes")
        if shapes is None or not shapes:
            return {
                "category": "capture_metadata_missing", "reason": "missing_tensor_metadata",
                "detail": "Native tensor ABI requires captured input and output descriptors.",
                "missing_fields": [f"{direction}_shapes"], "operator_type": row.kind,
            }
        dtypes = getattr(row, f"{direction}_dtypes")
        formats = getattr(row, f"{direction}_formats")
        if len(shapes) != len(dtypes) or len(shapes) != len(formats):
            return {
                "category": "capture_metadata_missing", "reason": "incomplete_native_tensor_slots",
                "detail": "Native shape, dtype and format arrays must describe the same ordered slots.",
                "direction": direction, "slot_counts": {
                    "shapes": len(shapes), "dtypes": len(dtypes), "formats": len(formats)},
                "operator_type": row.kind,
            }
        for index, (shape, dtype, layout) in enumerate(zip(shapes, dtypes, formats)):
            if any(not isinstance(value, int) or value < 0 for value in shape):
                return {
                    "category": "capture_metadata_missing", "reason": "unknown_tensor_dimensions",
                    "detail": "Negative or noninteger native dimensions cannot establish geometry.",
                    "direction": direction, "slot": index, "operator_type": row.kind,
                }
            if not shape and _undefined_dtype(dtype) and _undefined_layout(layout):
                continue
            missing = []
            if _norm_dtype(dtype) is None:
                missing.append("dtype")
            if _missing_layout(layout):
                missing.append("format")
            if missing:
                return {
                    "category": "capture_metadata_missing", "reason": "missing_tensor_metadata",
                    "detail": "Only explicitly undefined empty slots may omit dtype and format.",
                    "missing_fields": [f"{direction}[{index}].{field}" for field in missing],
                    "operator_type": row.kind,
                }
    return None


def native_tensor_contract(row, *, kind=None):
    """Return full native tensor slots; unknown nonempty dtype names stay literal."""
    if native_contract_issue(row) is not None:
        return None
    sides = []
    for direction in ("input", "output"):
        slots = []
        for shape, dtype, layout in zip(
            getattr(row, f"{direction}_shapes"), getattr(row, f"{direction}_dtypes"),
            getattr(row, f"{direction}_formats"),
        ):
            empty = (not shape and _undefined_dtype(dtype)
                     and _undefined_layout(layout))
            slots.append((tuple(shape), None if empty else _norm_dtype(dtype),
                          None if empty else layout))
        sides.append(tuple(slots))
    return ("native_tensor_abi", kind or row.kind, *sides)


def native_tensor_diagnostic(row):
    return native_contract_issue(row) or {
        "category": "tensor_contract_available", "reason": "observed_tensor_contract_available",
        "detail": "All captured native tensor slots are preserved; scalar values and dispatch are not.",
        "operator_type": row.kind,
    }


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _norm_dtype(value):
    """Canonical native dtype for a descriptor or signature; preserve unknown literals."""
    if not isinstance(value, str):
        return None
    text = value.strip().strip('"')
    if text.upper() in _UNDEFINED_DTYPES:
        return None
    return DTYPES.get(text.upper(), DTYPES.get(text.lower().replace("-", "_"), text))


def _slot_inputs(slots, direction):
    """Normalize every original slot without dropping position/presence.

    Records retain the original position, presence flags, raw dtype/layout and
    all known fields.  A missing/invalid field is an issue on that exact slot,
    never a deleted position.  ``present=false`` is an explicit absent slot;
    explicit undefined, typed-empty scalar and missing fields remain distinct.
    """
    records, issues = [], []
    for index, slot in enumerate(slots or ()):
        base = {"position": index}
        if isinstance(slot, Mapping):
            if slot.get("present") is False:
                records.append({
                    **base, "present": False, "shape": None, "dtype": None,
                    "layout": None, "dtype_present": False,
                    "layout_present": False, "quality": "absent", "issues": [],
                })
                continue
            shape_present = "shape" in slot
            shape = slot.get("shape")
            dtype_raw = slot.get("dtype_raw", slot.get("dtype"))
            dtype_present = (bool(slot.get("dtype_present"))
                             if "dtype_present" in slot else "dtype" in slot)
            if "layout" in slot:
                layout_raw = slot.get("layout_raw", slot.get("layout"))
                layout_present = (bool(slot.get("layout_present"))
                                  if "layout_present" in slot else True)
            elif "format" in slot:
                layout_raw = slot.get("format")
                layout_present = (bool(slot.get("layout_present"))
                                  if "layout_present" in slot else True)
            else:
                layout_raw, layout_present = None, False
        elif (isinstance(slot, Sequence)
              and not isinstance(slot, (str, bytes)) and len(slot) == 3):
            shape_present = True
            shape, dtype_raw, layout_raw = slot
            dtype_present = layout_present = True
        else:
            records.append({
                **base, "present": True, "shape": None, "dtype": None,
                "layout": None, "dtype_present": False,
                "layout_present": False, "quality": "invalid",
                "issues": [{"reason": "tensor_slot_not_a_descriptor"}],
            })
            issues.append({"slot": f"{direction}[{index}]",
                           "reason": "tensor_slot_not_a_descriptor"})
            continue
        slot_issues = []
        if not shape_present:
            slot_issues.append({"reason": "missing_tensor_shape"})
            shape = None
        if not isinstance(shape, (list, tuple)) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in shape
        ):
            slot_issues.append({"reason": "unknown_tensor_dimensions"})
            shape = list(shape) if isinstance(shape, (list, tuple)) else None
        else:
            shape = [int(value) for value in shape]
        dtype = _norm_dtype(dtype_raw)
        if not dtype_present:
            slot_issues.append({"reason": "missing_dtype"})
        if not layout_present:
            slot_issues.append({"reason": "missing_layout"})
        if not slot_issues:
            empty = not shape
            undefined_dtype = _undefined_dtype(dtype_raw)
            undefined_layout = _undefined_layout(layout_raw)
            if empty and undefined_dtype and undefined_layout:
                quality = "undefined"
            elif empty:
                if dtype is None or _missing_layout(layout_raw):
                    slot_issues.append({"reason": "missing_dtype_or_layout"})
                    quality = "missing"
                else:
                    quality = "typed_scalar"
            else:
                if dtype is None:
                    slot_issues.append({"reason": "missing_dtype"})
                if _missing_layout(layout_raw):
                    slot_issues.append({"reason": "missing_layout"})
                quality = "tensor" if not slot_issues else "missing"
        else:
            quality = "missing"
        record = {
            **base, "present": True, "shape": shape,
            "dtype": dtype if dtype_present else None,
            "layout": layout_raw if layout_present else None,
            "dtype_present": dtype_present, "layout_present": layout_present,
            "dtype_raw": dtype_raw, "layout_raw": layout_raw,
            "quality": quality, "issues": slot_issues,
        }
        records.append(record)
        for issue in slot_issues:
            issues.append({"slot": f"{direction}[{index}]",
                           "reason": issue["reason"]})
    return records, issues


def _canonical_slot(record):
    if record.get("quality") == "absent":
        return {"present": False}
    if record.get("quality") == "undefined":
        # Raw DT_UNDEFINED/NULL spellings stay in dtype_raw/layout_raw; the
        # semantic identity is the same explicit undefined slot.
        return {"shape": [], "dtype": None, "layout": None}
    return {
        "shape": list(record.get("shape") or []),
        "dtype": record.get("dtype"),
        "layout": record.get("layout"),
    }


def native_abi_from_slots(native_operator, inputs, outputs, *, strict_operator=True):
    """Build one lossless, idempotent ABI qualification from ordered slots.

    Every supplied slot position/presence is retained even on failure.  Only a
    complete qualification receives a digest; passing an already-qualified
    complete record back through this function reproduces the same canonical
    slots/digest.  Passing an incomplete record back cannot delete its gap or
    upgrade it to complete.  Known dtype aliases are normalized centrally;
    unknown names stay literal.
    """
    operator = str(native_operator or "").strip()
    in_slots, in_issues = _slot_inputs(inputs or (), "inputs")
    out_slots, out_issues = _slot_inputs(outputs or (), "outputs")
    issues = [*in_issues, *out_issues]
    if not operator:
        issues.insert(0, {"slot": "native_operator",
                          "reason": "missing_native_operator"})
    if not in_slots and not in_issues:
        issues.append({"slot": "inputs", "reason": "missing_native_input_slots"})
    if not out_slots and not out_issues:
        issues.append({"slot": "outputs", "reason": "missing_native_output_slots"})
    if issues:
        return {
            "schema": ABI_SCHEMA,
            "status": "incomplete" if strict_operator else "candidate",
            "native_operator": operator or None,
            "inputs": in_slots, "outputs": out_slots,
            "slot_count": {"inputs": len(in_slots), "outputs": len(out_slots)},
            "issues": issues, "digest": None, "verified": False,
            "declared_digest": None, "computed_digest": None,
        }
    payload = {
        "schema": ABI_SCHEMA, "native_operator": operator,
        "inputs": [_canonical_slot(record) for record in in_slots],
        "outputs": [_canonical_slot(record) for record in out_slots],
    }
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return {
        **payload, "status": "complete", "inputs": in_slots,
        "outputs": out_slots,
        "slot_count": {"inputs": len(in_slots), "outputs": len(out_slots)},
        "digest": f"sha256:{digest}", "verified": True,
        "declared_digest": None, "computed_digest": f"sha256:{digest}",
    }


def native_abi_from_contract(contract, *, strict_operator=True):
    """Build one native ABI from a declared ordered tensor contract.

    Captured cohort contracts use ``["native_tensor_abi", operator, inputs,
    outputs]`` where each slot is ``[shape, dtype, layout]``.  Those slots are
    the same ordered positions ``native_abi_from_slots`` already qualifies;
    this adapter only establishes the direction boundary.  A structural family
    contract (``["grouped_matrix", ...]`` or ``["slice_copy", ...]``) is not a
    native ABI and returns ``None`` instead of being promoted to one.
    """
    if not isinstance(contract, (list, tuple)) or len(contract) != 4:
        return None
    if str(contract[0] or "") != "native_tensor_abi":
        return None
    operator = contract[1]
    inputs, outputs = contract[2], contract[3]
    if not isinstance(operator, str) or not operator.strip():
        return None
    if not isinstance(inputs, (list, tuple)) or not isinstance(
            outputs, (list, tuple)):
        return None
    return native_abi_from_slots(operator, inputs, outputs,
                                 strict_operator=strict_operator)


def _row_slots(row, direction):
    shapes = getattr(row, f"{direction}_shapes", None)
    dtypes = getattr(row, f"{direction}_dtypes", None)
    formats = getattr(row, f"{direction}_formats", None)
    if shapes is None:
        return None
    count = max(len(shapes or ()), len(dtypes or ()), len(formats or ()))
    slots = []
    for index in range(count):
        dtype = dtypes[index] if dtypes is not None and index < len(dtypes) else None
        layout = formats[index] if formats is not None and index < len(formats) else None
        slots.append({
            "shape": shapes[index] if index < len(shapes) else None,
            "dtype": dtype,
            "layout": layout,
            "dtype_present": (dtypes is not None and index < len(dtypes)
                              and dtype is not None),
            "layout_present": (formats is not None and index < len(formats)
                               and layout is not None),
        })
    return slots


def native_abi_record(row, *, kind=None):
    """Return observed slots preserving every raw position, or a concrete gap.

    Local tensor gaps keep all other known slots; they are never collapsed to
    ``inputs=None``.  The original canonical payload for a fully known record
    remains unchanged so previously retained digests stay stable.
    """
    operator = kind if kind is not None else getattr(row, "kind", None)
    inputs = _row_slots(row, "input")
    outputs = _row_slots(row, "output")
    if inputs is None and outputs is None:
        return {
            "schema": ABI_SCHEMA, "status": "incomplete",
            "native_operator": operator,
            "inputs": [], "outputs": [],
            "slot_count": {"inputs": 0, "outputs": 0},
            "issues": [{"slot": "inputs",
                        "reason": "missing_tensor_metadata"}],
            "digest": None, "verified": False,
            "declared_digest": None, "computed_digest": None,
        }
    abi = native_abi_from_slots(operator, inputs or (), outputs or ())
    issue = native_contract_issue(row)
    if issue is not None:
        extra = {
            "slot": issue.get("missing_fields") or issue.get("direction") or "abi",
            "reason": issue.get("reason", "incomplete_native_tensor_slots"),
        }
        if issue.get("detail"):
            extra["detail"] = issue["detail"]
        issues = [*abi.get("issues", ()), extra]
        return {
            **abi, "status": "incomplete", "issues": issues,
            "digest": None, "verified": False, "computed_digest": None,
        }
    return abi


def native_abi_digest(record):
    """Return a complete verified digest, or ``None`` for gaps/conflicts."""
    if not isinstance(record, Mapping):
        return None
    if record.get("status") != "complete" or not record.get("verified"):
        return None
    digest = record.get("digest")
    return digest if isinstance(digest, str) and digest.startswith("sha256:") else None


def native_abi_identity(record):
    """Return only the operator and content hash used for exact pairing."""
    if not isinstance(record, Mapping):
        return None
    digest = native_abi_digest(record)
    operator = record.get("native_operator")
    if digest is None or not isinstance(operator, str) or not operator:
        return None
    return {"native_operator": operator, "native_abi_sha256": digest}


from collections import defaultdict
from collections.abc import Mapping, Sequence



class StructureTargets:
    def __init__(self, nodes, subgraphs, native_calls=None, *, _validated: bool = False):
        if not _validated:
            nodes = _snapshot_structure_carrier(nodes, "structure nodes")
            subgraphs = _snapshot_structure_carrier(subgraphs, "structure subgraphs")
            if native_calls is not None:
                native_calls = _snapshot_structure_carrier(native_calls, "structure native_calls")
        self.nodes = nodes
        self.subgraphs = {}
        for record in subgraphs:
            key = record.get("subgraph_id")
            members = record.get("node_ids", ())
            if not key or key in nodes or key in self.subgraphs:
                raise ProfilerUsageError(f"structure subgraph_id missing or duplicated: {key}")
            if not members or len(set(members)) != len(members) or any(
                member not in nodes for member in members
            ):
                raise ProfilerUsageError(f"subgraph has invalid semantic members: {key}")
            for field in ("replica", "stage"):
                if any(nodes[member].get(field) != record.get(field) for member in members):
                    raise ProfilerUsageError(f"subgraph crosses declared {field}: {key}")
            rec_phase = record.get("phase")
            for member in members:
                mem_phase = nodes[member].get("phase")
                if (
                    mem_phase != rec_phase
                    and not {mem_phase, rec_phase} <= {"decode", "draft"}
                    and not {mem_phase, rec_phase} <= {"prefill", "draft", "draft_prefill"}
                ):
                    raise ProfilerUsageError(f"subgraph crosses declared phase: {key}")
            if record.get("layer") is not None and any(
                nodes[member].get("layer") != record["layer"] for member in members
            ):
                raise ProfilerUsageError(f"subgraph has inconsistent declared layer: {key}")
            self.subgraphs[key] = record
        self.all = {**nodes, **self.subgraphs}
        self.native_implementations = defaultdict(list)
        self.native_calls = defaultdict(list)
        self.native_call_entries = defaultdict(list)
        self.native_call_kernels = defaultdict(list)
        self.native_call_observed = defaultdict(list)
        self.native_call_alias_names = set()
        self.native_bindings = {}
        if native_calls is not None and (
            not isinstance(native_calls, Sequence) or isinstance(native_calls, (str, bytes))
        ):
            raise ProfilerUsageError("structure native_calls must be a sequence of mappings")
        for record in native_calls or ():
            if not isinstance(record, Mapping):
                raise ProfilerUsageError("structure native_calls entries must be mappings")
            node_id = record.get("node_id")
            if not isinstance(node_id, str) or not node_id or node_id not in self.nodes:
                raise ProfilerUsageError(
                    f"structure native_calls references unknown node_id: {node_id!r}"
                )
            call_id = record.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                raise ProfilerUsageError(
                    f"structure native_calls entry for {node_id} requires non-empty call_id"
                )
            prev_binding = self.native_bindings.get(node_id)
            if prev_binding is not None and prev_binding.get("call_id") != call_id:
                raise ProfilerUsageError(
                    f"structure native_calls has conflicting bindings for {node_id}"
                )
            self.native_bindings[node_id] = record
            if node_id not in self.native_calls[call_id]:
                self.native_calls[call_id].append(node_id)
            api = record.get("api")
            if api is not None and not isinstance(api, Mapping):
                raise ProfilerUsageError(f"structure native_calls api must be a mapping: {node_id}")
            api = api or {}
            for entry in (api.get("entry"), api.get("aclnn_symbol"), *(api.get("candidates") or ())):
                if entry and node_id not in self.native_call_entries[entry]:
                    self.native_call_entries[entry].append(node_id)
            kernels = record.get("kernels", ())
            if not isinstance(kernels, Sequence) or isinstance(kernels, (str, bytes)):
                raise ProfilerUsageError(
                    f"structure native_calls kernels must be a sequence: {node_id}"
                )
            for kernel in kernels:
                if not isinstance(kernel, Mapping):
                    raise ProfilerUsageError(
                        f"structure native_calls kernel entry must be a mapping: {node_id}"
                    )
                name = kernel.get("name")
                if name and node_id not in self.native_call_kernels[name]:
                    self.native_call_kernels[name].append(node_id)
            aliases = record.get("observed_aliases", ())
            if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
                raise ProfilerUsageError(
                    f"structure native_calls observed_aliases must be a sequence: {node_id}"
                )
            for alias in aliases:
                if not isinstance(alias, Mapping):
                    raise ProfilerUsageError(
                        f"structure native_calls observed_alias must be a mapping: {node_id}"
                    )
                raw = (
                    alias.get("rank_id"),
                    alias.get("global_task_id"),
                    alias.get("stream_id"),
                    alias.get("task_id"),
                    alias.get("connection_id"),
                    alias.get("device_id"),
                    alias.get("start_ns"),
                    alias.get("end_ns"),
                )
                logical = alias.get("logical_rank", alias.get("rank_index"))
                if logical is not None and str(self.nodes[node_id].get("replica")) != str(logical):
                    continue
                if all(value is not None for value in raw):
                    raw_key = tuple(str(value) for value in raw)
                    if node_id not in self.native_call_observed[raw_key]:
                        self.native_call_observed[raw_key].append(node_id)
                for field in ("compute_name", "comm_name", "name", "op_type"):
                    if alias.get(field):
                        self.native_call_alias_names.add(str(alias[field]))
        for key, target in self.all.items():
            implementation = target.get("implementation") or {}
            kinds = set(implementation.get("native_kinds", ()))
            if implementation.get("native_kind"):
                kinds.add(implementation["native_kind"])
            for kind in sorted(kinds):
                self.native_implementations[kind].append(key)
        self._classification = {}

    def describe_implementation(self, kind, targets):
        """Describe a declared native boundary without claiming a tensor match."""
        targets = list(dict.fromkeys(targets))
        node_ids, subgraph_ids = self.split(targets)
        bindings = [self.native_bindings[key] for key in targets if key in self.native_bindings]
        call_ids = {row["call_id"] for row in bindings if row.get("call_id")}
        native_call = None
        if len(call_ids) == 1:
            row = bindings[0]
            api = row.get("api") or {}
            owner_keys = {
                (
                    str(self.all.get(key, {}).get("replica")),
                    str(self.all.get(key, {}).get("stage")),
                    (self.native_bindings[key].get("timing_owner") or {}).get("op_id")
                    or self.native_bindings[key].get("op_id")
                    or key,
                )
                for key in targets
                if key in self.native_bindings
            }
            unique_timing_owner = row.get("timing_owner") if len(owner_keys) == 1 else None
            native_call = {
                "call_id": row["call_id"],
                "entry": api.get("entry"),
                "aclnn_symbol": api.get("aclnn_symbol"),
                "kernels": [
                    kernel.get("name")
                    for kernel in row.get("kernels", ())
                    if isinstance(kernel, dict) and kernel.get("name")
                ],
                "evidence": row.get("evidence") or {},
                "timing_owner": unique_timing_owner,
                "declared_target_count": len(bindings),
                "tensor_match_implied": False,
                "occurrence_binding_implied": False,
            }
        return {
            "basis": "declared_native_implementation",
            "native_operator_type": kind,
            "declared_target_count": len(targets),
            "declared_node_ids": node_ids[:64],
            "declared_subgraph_ids": subgraph_ids[:64],
            "targets_with_contract": sum(
                self.all[key].get("contract") is not None for key in targets),
            "event_matchable_targets": sum(
                self.all[key].get("event_matchable", True) for key in targets),
            "coverage_kinds": sorted({
                self.all[key].get("coverage", {}).get(
                    "kind", "declared_node_boundary" if key in self.nodes else "complete_subgraph")
                for key in targets
            }),
            "common": self.classify(targets)["common"],
            "native_call": native_call,
            "tensor_match_implied": False,
            "occurrence_binding_implied": False,
        }

    def members(self, key):
        return self.subgraphs[key]["node_ids"] if key in self.subgraphs else [key]

    def enclosing_scopes(self, keys, fields):
        """Keep both sides of a cross-block fusion without duplicating an event."""
        scopes = set()
        for key in keys:
            for member in self.members(key):
                node = self.nodes[member]
                meta = node.get("operator", {}).get("metadata", {})
                semantic = {**meta.get("semantic_metadata", {}), **meta}
                owners = [node, *semantic.get("fused_origin_scopes", ())]
                for owner in owners:
                    scope = {**owner, "replica": node.get("replica")}
                    if (scope.get("layer") is not None and scope["layer"] >= 0
                            and scope.get(fields[-1]) is not None):
                        scopes.add(tuple(scope.get(field) for field in fields))
        return scopes

    def scope(self, row):
        """Expand a declared evidence scope, preserving full fusion containment."""
        node_ids = row.get("node_ids", ())
        subgraph_ids = row.get("subgraph_ids", ())
        if not node_ids and not subgraph_ids:
            raise ProfilerUsageError("evidence scope requires node_ids or subgraph_ids")
        if any(key not in self.nodes for key in node_ids) or any(
            key not in self.subgraphs for key in subgraph_ids
        ):
            raise ProfilerUsageError("evidence scope references an unknown node or subgraph")
        members = set(node_ids)
        for key in subgraph_ids:
            members.update(self.members(key))
        return members | {
            key for key, record in self.subgraphs.items()
            if set(record["node_ids"]) <= members
        }

    def split(self, keys):
        return (
            [key for key in keys if key in self.nodes],
            [key for key in keys if key in self.subgraphs],
        )

    def classify(self, candidates):
        """Keep only properties shared by every compatible declared target."""
        key = tuple(candidates)
        if key not in self._classification:
            targets = [self.all[item] for item in candidates]
            common = {}
            for field in ("block", "role"):
                values = {target.get(field) for target in targets}
                if len(values) == 1 and None not in values:
                    common[field] = next(iter(values))
            structure = dict(targets[0].get("structure", {})) if targets else {}
            structure = {
                field: value for field, value in structure.items()
                if value is not None and all(
                    target.get("structure", {}).get(field) == value for target in targets
                )
            }
            if structure:
                common["structure"] = structure
            self._classification[key] = {
                "basis": "common_to_all_declared_tensor_candidates",
                "common": common,
                "occurrence_binding_implied": False,
            }
        return self._classification[key]


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
                or gamma[1] not in {x[1], "fp32"} or gamma[2] != "ND" or outputs[0] != x
                or any(slot[0] != (x[0][0], 1) or slot[1:] != gamma[1:]
                       for slot in outputs[1:])):
            return None, _issue(kind, "unsupported_layernorm_geometry",
                                "Verified LayerNorm has ND rows, matching affine channels, "
                                "a primary result and mean/rstd in the affine dtype per row.")
        return ("layernorm_active", kind, x, gamma, beta, outputs[0]), None
    if len(inputs) != 2 or len(outputs) != 1:
        return None, _issue(kind, "unsupported_native_tensor_slots",
                            "The retained TransposeBatchMatMul boundary has two inputs and one output.")
    x, weight = inputs
    y = outputs[0]
    if (x[1] not in _FLOAT or any(
            len(slot[0]) != 3 or min(slot[0]) <= 0 or slot[1:] != (x[1], "ND")
            for slot in (x, weight, y))):
        return None, _issue(kind, "unsupported_batch_matrix_geometry",
                            "The supported projection needs rank-three ND inputs and output of the same floating dtype.")
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


ProfilerUsageError = StructureTargetError
