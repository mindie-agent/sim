"""MindIE-SD operator declarations.

[Role] Declare MindIE-SD diffusion operator selection.
[Invariants] Only exact names or explicit declarations select custom operators.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

import sys
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
MINDIE_SD_VENDOR = "mindie_sd"

_MINDIE_SD_REPO = "Ascend/MindIE-SD"
_MINDIE_SD_REV = "9fe772dfb90696e21b5e5d4e92ca07dffab84cf4"

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository=_MINDIE_SD_REPO,
        revision=_MINDIE_SD_REV,
        relative_path="csrc/plugin/rainfusionattention.cpp",
        source_sha256="b93f7038895fc9090b347f6da793c7b744c48d4b55a8d3e6115fb5f5e8eaeaf1",
    ),
    SourceRef(
        repository=_MINDIE_SD_REPO,
        revision=_MINDIE_SD_REV,
        relative_path="csrc/ops/ada_block_sparse_attention/op_host/ada_block_sparse_attention_def.cpp",
        source_sha256="105e0eea08d20f51d095c4d36a4d26e423bdf8189575118519b802c1c6adb49c",
    ),
    SourceRef(
        repository=_MINDIE_SD_REPO,
        revision=_MINDIE_SD_REV,
        relative_path="csrc/ops/eagle_block_sparse_attention/op_host/eagle_block_sparse_attention_def.cpp",
        source_sha256="dbb1022ab1620c1fb1f0e34440e180085b23a53722f6ff9dfc88b58084d65551",
    ),
    SourceRef(
        repository=_MINDIE_SD_REPO,
        revision=_MINDIE_SD_REV,
        relative_path="csrc/ops/eagle_quant_block_sparse_attention/op_host/eagle_quant_block_sparse_attention_def.cpp",
        source_sha256="db5020ae4ef6eb4dbcd7dbd4756354653c8a2372872e9bf8ac021b47d20cd385",
    ),
    SourceRef(
        repository=_MINDIE_SD_REPO,
        revision=_MINDIE_SD_REV,
        relative_path="csrc/plugin/adalayernorm.cpp",
        source_sha256="d862775cf5193b94b411b0c1edc7ac9e85429407b7f7326a4dbd960d951ce68f",
    ),
    SourceRef(
        repository=_MINDIE_SD_REPO,
        revision=_MINDIE_SD_REV,
        relative_path="csrc/ops/norm_rope_concat/op_host/norm_rope_concat_def.cpp",
        source_sha256="6c7fd0d1473073d593117176d7cd0d2b5c0deea8c73ae9c50e9e87fe14c57552",
    ),
    SourceRef(
        repository=_MINDIE_SD_REPO,
        revision=_MINDIE_SD_REV,
        relative_path="csrc/plugin/register_ops.cpp",
        source_sha256="861fc97a28905b4e9c2ea39f8cd5641725b7b21f355ca8bb237446b5ae86c08a",
    ),
    SourceRef(
        repository=_MINDIE_SD_REPO,
        revision=_MINDIE_SD_REV,
        relative_path="csrc/ops/quant_four_over_six_a5/op_host/quant_four_over_six_a5_def.cpp",
        source_sha256="0266e56885835192fa88810cbf861a5ceb6bb1cf8d7a72ec07324e874c6cef68",
    ),
    SourceRef(
        repository=_MINDIE_SD_REPO,
        revision=_MINDIE_SD_REV,
        relative_path="csrc/ops/eagle_ffn/op_host/eagle_ffn_def.cpp",
        source_sha256="b742ae16a54ac6c84d1452fd29fc1dc8c3fe948c295b00c6389e5ae3e7529880",
    ),
)

_ALL_SOCS = ("Ascend910B", "Ascend910_93", "Ascend950")
_ARCH22_SOCS = ("Ascend910B", "Ascend910_93")
_ARCH35_SOCS = ("Ascend950",)
_ATTENTION_DTYPES = ("bf16", "fp16")
_QUANT_ATTENTION_DTYPES = ("bf16", "fp16", "int8")
_NORM_ROPE_DTYPES = ("bf16", "fp16", "fp32")
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
        "supported_socs": _ARCH22_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_ada_bsa"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("sparsity_ratio", "block_size"),
        "sparse": True,
    },
    "eagle_bsa": {
        "native_kind": "EagleBlockSparseAttention", "bound": "cube",
        "supported_socs": _ARCH22_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_eagle_bsa"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": ("sparsity_ratio", "block_size"),
        "sparse": True,
    },
    "eagle_qbsa": {
        "native_kind": "EagleQuantBlockSparseAttention", "bound": "cube",
        "supported_socs": _ARCH35_SOCS, "families": (*_ATTN_FAMILIES, "mindie_sd_eagle_qbsa"),
        "input_dtypes": _QUANT_ATTENTION_DTYPES,
        "attributes": ("sparsity_ratio", "block_size", "kv_dtype"),
        "sparse": True, "quantized_cache": True, "a5_only": True,
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
        "input_dtypes": _NORM_ROPE_DTYPES,
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
        "supported_socs": _ARCH35_SOCS,
        "families": ("quant_four_over_six_a5", "mindie_sd_quant_four_over_six_a5",
                     "quantize", "elementwise"),
        "input_dtypes": ("bf16",),
        "attributes": (),
        "a5_only": True,
    },
    "eagle_ffn": {
        "native_kind": "CatlassMatmulAddFfn", "bound": "cube",
        "supported_socs": _ARCH35_SOCS,
        "unfused_fallback_socs": _ARCH22_SOCS,
        "families": ("eagle_ffn", "mindie_sd_eagle_ffn", "gemm", "ffn"),
        "input_dtypes": _ATTENTION_DTYPES,
        "attributes": (),
        "a5_only": True,
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

    modules = {__package__}
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
