"""xLLM custom operator declarations and input validation.

[Role] Declare xLLM custom operator and input contracts.
[Invariants] Vendor names select explicit occurrences. Shape and dtype come from the occurrence.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

from math import prod
import sys
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


XLLM_VENDOR = "custom_xllm_math"
FUSED_GROUPED_RECIPE = "moe_grouped_matmul_swiglu_quant"
FP8_GROUPED_OWNER_RECIPE = "ascend_mxfp8_grouped_swiglu_quant"

_ATTENTION_DTYPES = ("bf16", "fp16")
_QUANT_OUTPUT_DTYPES = ("int8", "fp8_e4m3", "fp8_e5m2")

XLLM_CUSTOM_OPS: dict[str, dict[str, Any]] = {
    "smla_va": {
        "native_kind": "SmlaVa",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/smla_va",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("smla_va", "mla_attention", "paged_mla_attention", "attention"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("op_name", "soc_family"),
        "atomic_occurrence": True,
    },
    "sparse_flash_attention_lse": {
        "native_kind": "SparseFlashAttentionLse",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/sparse_flash_attention_lse",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("sparse_flash_attention_lse", "indexed_attention", "mla_attention",
                     "attention"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("op_name", "soc_family", "return_lse"),
        "atomic_occurrence": True,
    },
    "moe_grouped_matmul_swiglu_quant": {
        "native_kind": "MoeGroupedMatmulSwigluQuant",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/moe_grouped_matmul_swiglu_quant",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("moe_grouped_matmul_swiglu_quant", "grouped_gemm"),
        "input_dtypes": ("int8",),
        "attributes": ("op_name", "soc_family", "weight_scale_format",
                       "output_format"),
        "atomic_occurrence": False,
        "requires_declared_fusion": True,
        "quantization": "int8_per_channel_float_scale",
        "fp8_owner_recipe": FP8_GROUPED_OWNER_RECIPE,
    },
    "mega_gdn_decode": {
        "native_kind": "MegaGdnDecode",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/mega_gdn_decode",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("mega_gdn_decode", "gdn_decode", "recurrent_vector",
                     "recurrent_scan"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("op_name", "soc_family"),
        "atomic_occurrence": True,
    },
    "onerec_final_beam_select": {
        "native_kind": "OnerecFinalBeamSelect",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/onerec_final_beam_select",
        "supported_socs": ("Ascend910B", "Ascend910_93", "Ascend950"),
        "families": ("onerec_final_beam_select", "beam_select", "elementwise"),
        "input_dtypes": ("bf16", "fp16", "fp32"),
        "attributes": ("op_name", "soc_family", "beam_width"),
        "atomic_occurrence": True,
    },
    "reshape_and_cache_a5": {
        "native_kind": "ReshapeAndCacheA5",
        "vendor": XLLM_VENDOR,
        "source_path": "third_party/xllm_ops/reshape_and_cache_a5",
        "supported_socs": ("Ascend950",),
        "families": ("reshape_and_cache_a5", "kv_cache_write", "cache_store"),
        "input_dtypes": ("bf16", "fp16", "fp8_e4m3", "fp8_e5m2", "int8"),
        "attributes": ("op_name", "soc_family", "kv_page_size"),
        "atomic_occurrence": True,
        "a5_only": True,
    },
}


CAPABILITIES = frozenset({
    "explicit_caller_implementation",
    "custom_xllm_math",
    "A2",
    "A3",
    "A5",
    "atomic_occurrence_lowering",
    "dtype_checked",
    "declared_work_preserved",
    "fused_grouped_gemm_requires_declared_region",
    "int8_grouped_gemm",
    "fp8_mx32_owned_by_f_matrix",
})


# Generic attention, grouped GEMM, cache store and elementwise families
# require an explicit implementation.
_EXACT_SEMANTIC_ALIASES = {
    "smla_va": "smla_va",
    "mla_attention": "smla_va",
    "paged_mla_attention": "smla_va",
    "sparse_flash_attention_lse": "sparse_flash_attention_lse",
    "indexed_attention": "sparse_flash_attention_lse",
    "moe_grouped_matmul_swiglu_quant": "moe_grouped_matmul_swiglu_quant",
    "mega_gdn_decode": "mega_gdn_decode",
    "gdn_decode": "mega_gdn_decode",
    "onerec_final_beam_select": "onerec_final_beam_select",
    "beam_select": "onerec_final_beam_select",
    "reshape_and_cache_a5": "reshape_and_cache_a5",
    "kv_cache_write": "reshape_and_cache_a5",
}


def _custom_ops() -> dict[str, dict[str, Any]]:
    pkg = sys.modules.get(__package__ or "")
    return getattr(pkg, "XLLM_CUSTOM_OPS", XLLM_CUSTOM_OPS) if pkg is not None else XLLM_CUSTOM_OPS


def capabilities() -> dict[str, Any]:
    """Machine-readable registry of the declarations behind this wrapper."""
    from mindie_sim.prediction.implementations import IMPLEMENTATIONS

    # Registry entries can refer to the original module during the move.
    modules = {__package__, "mindie_sim.prediction.family_xllm_ops"}
    implementations = {
        name: item for name, item in IMPLEMENTATIONS.items() if item.module in modules}
    selector_operators = {
        operator for item in implementations.values() for operator in item.operators}
    ops = _custom_ops()
    return {
        "vendor": XLLM_VENDOR,
        "capabilities": tuple(sorted(CAPABILITIES)),
        "implementations": tuple(implementations),
        "semantic_aliases": dict(_EXACT_SEMANTIC_ALIASES),
        "operators": {
            name: {
                "native_kind": spec["native_kind"],
                "families": tuple(spec["families"]),
                "selector_families": tuple(
                    family for family in spec["families"]
                    if family in selector_operators),
                "direct_only_families": tuple(
                    family for family in spec["families"]
                    if family not in selector_operators),
                "supported_socs": tuple(spec["supported_socs"]),
                "input_dtypes": tuple(spec["input_dtypes"]),
                "attributes": tuple(spec["attributes"]),
                "atomic_occurrence": bool(spec["atomic_occurrence"]),
                "requires_declared_fusion": bool(
                    spec.get("requires_declared_fusion")),
                "quantization": spec.get("quantization"),
                "fp8_owner_recipe": spec.get("fp8_owner_recipe"),
            }
            for name, spec in ops.items()
        },
    }


def _declared_dtype(op: OpSpec, meta: dict[str, Any]) -> str:
    dtype = op.dtype or meta.get("input_format") or meta.get("compute_format")
    if not dtype:
        raise ModelingError("xllm_custom_op_requires_declared_input_dtype")
    return str(dtype)


def _shape(op: OpSpec) -> tuple[int, ...]:
    shape = tuple(int(value) for value in op.local_shape)
    if not shape or any(value < 0 for value in shape):
        raise ModelingError("xllm_custom_op_requires_nonnegative_local_shape")
    return shape


def _tokens_width(op: OpSpec) -> tuple[int, int]:
    shape = _shape(op)
    return shape[0], prod(shape[1:]) if len(shape) > 1 else 1


def _attention_heads(op: OpSpec, meta: dict[str, Any]) -> int:
    shape = _shape(op)
    if len(shape) >= 5:
        heads = shape[2]
    elif len(shape) == 4:
        heads = shape[2]
    elif len(shape) == 3:
        heads = shape[1]
    else:
        heads = int(meta.get("heads", 1) or 1)
    if heads < 1:
        raise ModelingError("xllm_attention_requires_positive_heads")
    return heads


def _resolve_op_name(op: OpSpec, explicit_op_name: str | None) -> str:
    ops = _custom_ops()
    if explicit_op_name is not None:
        clean = str(explicit_op_name).strip()
        if clean not in ops:
            raise ModelingError(f"xllm_custom_op_unknown:{clean}")
        return clean
    # Exact vendor semantic name only.  A shared family such as
    # ``elementwise`` or ``grouped_gemm`` is not a declaration that this
    # vendor operator runs; it must be named by the caller.
    for candidate in (op.recipe, op.family):
        if candidate in _EXACT_SEMANTIC_ALIASES and _EXACT_SEMANTIC_ALIASES[candidate] in ops:
            return _EXACT_SEMANTIC_ALIASES[candidate]
    raise ModelingError(f"xllm_custom_op_requires_explicit_name:{op.family}:{op.recipe}")


def _check_family(op: OpSpec, spec: dict[str, Any], resolved: str) -> None:
    if op.family in spec["families"] or op.recipe in spec["families"]:
        return
    raise ModelingError(f"xllm_custom_op_family_mismatch:{resolved}:{op.family}")


def _fused_declared(op: OpSpec, meta: dict[str, Any], resolved: str) -> bool:
    scopes = meta.get("fused_origin_scopes") or ()
    return op.recipe == resolved or (
        isinstance(scopes, (tuple, list)) and len(scopes) >= 3)
