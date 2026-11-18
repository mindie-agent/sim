"""BIND1 native-call signature vocabulary: tokens, dtype aliases and roles.

[Role]
  Own the literal vocabulary used to parse captured semicolon-delimited tensor
  lanes and to align declared BIND1 ``typed`` roles with observed ABI slots.

[Invariants]
  * Missing/undefined spellings stay distinct from a parsed empty shape.
  * Dtype aliases normalize to canonical lower-case tokens; unknown spellings
    fall back to lower-cased text with a leading ``dt_`` stripped.
  * Role ranks are unique within a direction; duplicate ranks are ambiguous.

[Timeouts & Locks]
  Pure immutable module constants and one in-memory normalization helper.

[Side-Effect Policy]
  No I/O or mutation of caller state; imports no profiler runtime modules.
"""

from __future__ import annotations

from typing import Any

_MISSING_TEXT = frozenset({"", "N/A", "NULL", "NONE", "UNDEFINED"})
_UNDEFINED_DTYPES = frozenset({"", "N/A", "NULL", "NONE", "UNDEFINED", "DT_UNDEFINED"})
_ATTENTION_AUX_DTYPES = frozenset(
    {"int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "bool"}
)
_DTYPE_TOKENS = {
    **dict.fromkeys(("BF16", "BFP16", "BFLOAT16", "DT_BF16", "DT_BFLOAT16"), "bf16"),
    **dict.fromkeys(("FP16", "HALF", "FLOAT16", "DT_FLOAT16", "DT_HALF"), "fp16"),
    **dict.fromkeys(("FP32", "FLOAT", "FLOAT32", "DT_FLOAT", "DT_FLOAT32"), "fp32"),
    **dict.fromkeys(("FP64", "DOUBLE", "FLOAT64", "DT_DOUBLE", "DT_FLOAT64"), "fp64"),
    **dict.fromkeys(("INT4", "DT_INT4"), "int4"),
    **dict.fromkeys(("INT8", "DT_INT8"), "int8"),
    **dict.fromkeys(("INT16", "DT_INT16"), "int16"),
    **dict.fromkeys(("INT32", "DT_INT32"), "int32"),
    **dict.fromkeys(("INT64", "DT_INT64"), "int64"),
    **dict.fromkeys(("UINT8", "DT_UINT8"), "uint8"),
    **dict.fromkeys(("UINT16", "DT_UINT16"), "uint16"),
    **dict.fromkeys(("UINT32", "DT_UINT32"), "uint32"),
    **dict.fromkeys(("UINT64", "DT_UINT64"), "uint64"),
    **dict.fromkeys(("BOOL", "DT_BOOL"), "bool"),
    **dict.fromkeys(("HIF8", "HIFLOAT8", "DT_HIFLOAT8"), "hif8"),
    **dict.fromkeys(("FP8_E4M3", "FP8_E4M3FN", "FLOAT8_E4M3FN", "DT_FLOAT8_E4M3FN"), "fp8_e4m3"),
    **dict.fromkeys(("FP8_E5M2", "FLOAT8_E5M2", "DT_FLOAT8_E5M2"), "fp8_e5m2"),
    **dict.fromkeys(("FP8_E8M0", "FLOAT8_E8M0", "DT_FLOAT8_E8M0"), "fp8_e8m0"),
    **dict.fromkeys(("MXFP4", "FLOAT4_E2M1", "DT_FLOAT4_E2M1"), "mxfp4"),
}
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


def _norm_dtype(raw: Any) -> str | None:
    text = str(raw or "").strip().strip('"')
    upper = text.upper()
    if upper in _UNDEFINED_DTYPES:
        return None
    return _DTYPE_TOKENS.get(upper, text.lower().removeprefix("dt_"))
