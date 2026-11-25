"""MindIE-SD diffusion operator entry points.

[Role] Expose diffusion lowering, typed recipes and capability declarations.
[Invariants] Public entry points and private helpers retain their original module-level names.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

import math as math
from dataclasses import replace as replace
from math import prod as prod
from typing import Any, Literal
from typing import Mapping as Mapping

from mindie_sim.schema.common import (
    ModelingError as ModelingError,
    align_up as align_up,
    dtype_bytes as dtype_bytes,
)
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec as OpSpec

from mindie_sim.prediction.mechanism import OperatorPlan as OperatorPlan, PlannedRecipe, Work as Work
from mindie_sim.prediction.program import KernelProgram as KernelProgram, Stage as Stage
from .attention import (
    _norm_rope_contract as _norm_rope_contract,
    _sparse_contract as _sparse_contract,
    _sparse_shape as _sparse_shape,
)
from .block_ops import (
    _adalayernorm_contract as _adalayernorm_contract,
    _eagle_ffn_contract as _eagle_ffn_contract,
    _laurel_contract as _laurel_contract,
    _quant_four_over_six_contract as _quant_four_over_six_contract,
)
from .contracts import (
    _as_shape as _as_shape,
    _build_contract_plan,
    _bytes as _bytes,
    _contract as _contract,
    _contract_work as _contract_work,
    _input_dtype as _input_dtype,
    _positive_int as _positive_int,
    _ratio as _ratio,
    _read as _read,
    _role as _role,
    _shape as _shape,
    _tokens_width as _tokens_width,
)
from .lowering import (
    compute_mindie_sd_effective_denoising_steps,
    lower_op,
)
from .registry import (
    CAPABILITIES,
    MINDIE_SD_CUSTOM_OPS,
    MINDIE_SD_VENDOR,
    _ALL_SOCS as _ALL_SOCS,
    _ATTENTION_DTYPES as _ATTENTION_DTYPES,
    _ATTN_FAMILIES as _ATTN_FAMILIES,
    _EXACT_SEMANTIC_ALIASES as _EXACT_SEMANTIC_ALIASES,
    _normalize_op_name as _normalize_op_name,
    _resolve_op_name as _resolve_op_name,
    capabilities,
)


TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"

_MINDIE_SD_REPO = "Ascend/MindIE-SD"
_MINDIE_SD_REV = "9fe772dfb90696e21b5e5d4e92ca07dffab84cf4"

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository=_MINDIE_SD_REPO,
        revision=_MINDIE_SD_REV,
        relative_path="csrc/plugin/register_ops.cpp",
        source_sha256="861fc97a28905b4e9c2ea39f8cd5641725b7b21f355ca8bb237446b5ae86c08a",
    ),
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


def recipes() -> dict[str, Any]:
    """Register one typed plan per composite MindIE-SD occurrence."""
    return {f"mindie_sd_{name}": PlannedRecipe(_build_contract_plan)
            for name in MINDIE_SD_CUSTOM_OPS}


__all__ = [
    "CAPABILITIES",
    "MINDIE_SD_CUSTOM_OPS",
    "MINDIE_SD_VENDOR",
    "SOURCES",
    "TARGET_BACKEND",
    "capabilities",
    "compute_mindie_sd_effective_denoising_steps",
    "lower_op",
    "recipes",
]
