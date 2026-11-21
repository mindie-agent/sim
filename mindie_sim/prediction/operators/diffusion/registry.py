"""MindIE-SD operator declarations.

[Role] Declare MindIE-SD diffusion operator selection.
[Invariants] Only exact names or explicit declarations select custom operators.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

import sys
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


MINDIE_SD_VENDOR = "mindie_sd"


_ALL_SOCS = ("Ascend910B", "Ascend910_93", "Ascend950")
_ATTENTION_DTYPES = ("bf16", "fp16")
_ATTN_FAMILIES = ("rainfusion", "ada_bsa", "eagle_bsa", "eagle_qbsa", "indexed_attention", "gqa_attention")

MINDIE_SD_CUSTOM_OPS: dict[str, dict[str, Any]] = {
    "rainfusion": {
        "native_kind": "RainFusionBlockSparseAttention", "bound": "cube",
        "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_rainfusion"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("sparsity_ratio", "block_size"),
        "sparse": True,
    },
    "ada_bsa": {
        "native_kind": "AdaBlockSparseAttention", "bound": "cube",
        "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_ada_bsa"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("sparsity_ratio", "block_size"),
        "sparse": True,
    },
    "eagle_bsa": {
        "native_kind": "EagleBlockSparseAttention", "bound": "cube",
        "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_eagle_bsa"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("sparsity_ratio", "block_size"),
        "sparse": True,
    },
    "eagle_qbsa": {
        "native_kind": "EagleQuantBlockSparseAttention", "bound": "cube",
        "supported_socs": _ALL_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_eagle_qbsa"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("sparsity_ratio", "block_size", "kv_dtype"),
        "sparse": True, "quantized_cache": True,
    },
    "adalayernorm": {
        "native_kind": "AdaLayerNormFused", "bound": "memory",
        "supported_socs": _ALL_SOCS,
        "families": ("adalayernorm", "mindie_sd_adalayernorm", "rmsnorm", "add_rmsnorm"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("gamma_beta_count",),
    },
    "norm_rope_concat": {
        "native_kind": "RmsNormRopeConcatFused", "bound": "memory",
        "supported_socs": _ALL_SOCS,
        "families": ("norm_rope_concat", "mindie_sd_norm_rope_concat", "rmsnorm", "elementwise"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": (),
    },
    "laurel": {
        "native_kind": "LaurelLowRankResidual", "bound": "cube",
        "supported_socs": _ALL_SOCS,
        "families": ("laurel", "mindie_sd_laurel", "gemm", "elementwise"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("low_rank",),
    },
    "quant_four_over_six_a5": {
        "native_kind": "QuantFourOverSixA5", "bound": "vector",
        "supported_socs": ("Ascend950",),
        "families": ("quant_four_over_six_a5", "mindie_sd_quant_four_over_six_a5",
                     "quantize", "elementwise"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": (),
        "a5_only": True,
    },
    "eagle_ffn": {
        "native_kind": "CatlassMatmulAddFfn", "bound": "cube",
        "supported_socs": _ALL_SOCS,
        "families": ("eagle_ffn", "mindie_sd_eagle_ffn", "gemm", "ffn"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": (),
    },
}


CAPABILITIES = frozenset({
    "explicit_caller_implementation",
    "mindie_sd",
    "A2",
    "A3",
    "A5",
    "block_sparse_attention",
    "declared_sparsity",
    "declared_work_preserved",
    "ordinary_resource_envelope",
    "a5_mxfp4_quantization",
})


# Shared families require an explicit implementation.
_EXACT_SEMANTIC_ALIASES = {
    "rainfusion": "rainfusion",
    "ada_bsa": "ada_bsa",
    "eagle_bsa": "eagle_bsa",
    "eagle_qbsa": "eagle_qbsa",
    "adalayernorm": "adalayernorm",
    "norm_rope_concat": "norm_rope_concat",
    "laurel": "laurel",
    "quant_four_over_six_a5": "quant_four_over_six_a5",
    "eagle_ffn": "eagle_ffn",
}


def _custom_ops() -> dict[str, dict[str, Any]]:
    pkg = sys.modules.get(__package__ or "")
    return getattr(pkg, "MINDIE_SD_CUSTOM_OPS", MINDIE_SD_CUSTOM_OPS) if pkg is not None else MINDIE_SD_CUSTOM_OPS


def capabilities() -> dict[str, Any]:
    """Machine-readable registry of the declarations behind this wrapper."""
    from mindie_sim.prediction.implementations import IMPLEMENTATIONS

    # Registry entries can refer to the original module during the move.
    modules = {__package__, "mindie_sim.prediction.family_mindie_sd_ops"}
    implementations = {
        name: item for name, item in IMPLEMENTATIONS.items() if item.module in modules}
    selector_operators = {
        operator for item in implementations.values() for operator in item.operators}
    ops = _custom_ops()
    return {
        "vendor": MINDIE_SD_VENDOR,
        "capabilities": tuple(sorted(CAPABILITIES)),
        "implementations": tuple(implementations),
        "semantic_aliases": dict(_EXACT_SEMANTIC_ALIASES),
        "operators": {
            name: {
                "native_kind": spec["native_kind"],
                "bound": spec["bound"],
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
                "sparse": bool(spec.get("sparse")),
                "quantized_cache": bool(spec.get("quantized_cache")),
            }
            for name, spec in ops.items()
        },
    }


def _normalize_op_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    clean = str(raw).strip().lower().removeprefix("mindie_sd_")
    return clean if clean in _custom_ops() else str(raw).strip()


def _resolve_op_name(op: OpSpec, explicit_op_name: str | None) -> str:
    ops = _custom_ops()
    norm = _normalize_op_name(explicit_op_name)
    if norm is not None:
        if norm not in ops:
            raise ModelingError(f"mindie_sd_custom_op_unknown:{explicit_op_name}")
        return norm
    for candidate in (op.recipe, op.family):
        if candidate in _EXACT_SEMANTIC_ALIASES and _EXACT_SEMANTIC_ALIASES[candidate] in ops:
            return _EXACT_SEMANTIC_ALIASES[candidate]
        normalized = _normalize_op_name(candidate)
        if normalized in ops:
            return normalized
    raise ModelingError(
        f"mindie_sd_custom_op_requires_explicit_name:{op.family}:{op.recipe}")
