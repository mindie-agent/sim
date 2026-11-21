"""xLLM custom operator entry points.

[Role] Expose xLLM lowering, typed recipes and capability declarations.
[Invariants] Public entry points and private helpers retain their original module-level names.
[Timeouts & Locks] Bounded local computation. No subprocesses or external locks.
[Side-Effect Policy] Creates local records only. No I/O or device access.
"""

from __future__ import annotations

from dataclasses import replace as replace
from math import prod as prod
from typing import Any

from mindie_sim.schema.common import (
    ModelingError as ModelingError,
    align_up as align_up,
    dtype_bytes as dtype_bytes,
)
from mindie_sim.schema.operators import OpSpec as OpSpec

from ..mechanism import OperatorPlan as OperatorPlan, PlannedRecipe, Work as Work
from ..program import Allocation as Allocation, KernelProgram as KernelProgram, Stage as Stage
from .cache import _build_cache_a5, _cache_row_bytes as _cache_row_bytes
from .contracts import (
    CAPABILITIES,
    FP8_GROUPED_OWNER_RECIPE as FP8_GROUPED_OWNER_RECIPE,
    FUSED_GROUPED_RECIPE,
    XLLM_CUSTOM_OPS,
    XLLM_VENDOR,
    _ATTENTION_DTYPES as _ATTENTION_DTYPES,
    _EXACT_SEMANTIC_ALIASES as _EXACT_SEMANTIC_ALIASES,
    _QUANT_OUTPUT_DTYPES as _QUANT_OUTPUT_DTYPES,
    _attention_heads as _attention_heads,
    _check_family as _check_family,
    _declared_dtype as _declared_dtype,
    _fused_declared as _fused_declared,
    _resolve_op_name as _resolve_op_name,
    _shape as _shape,
    _tokens_width as _tokens_width,
    capabilities,
)
from .gdn import (
    _build_mega_gdn_decode,
    _mega_gdn_config as _mega_gdn_config,
    _shape_meta as _shape_meta,
)
from .grouped_matmul import (
    _build_fused_grouped,
    _fused_config as _fused_config,
    _fused_epilogue_work as _fused_epilogue_work,
    _fused_quant_contract as _fused_quant_contract,
    _matmul_dims as _matmul_dims,
    fused_region_contract,
)
from .lowering import lower_op


def recipes() -> dict[str, Any]:
    """Register the physical plans for the grouped, recurrent and cache calls."""
    return {
        FUSED_GROUPED_RECIPE: PlannedRecipe(_build_fused_grouped),
        "mega_gdn_decode": PlannedRecipe(_build_mega_gdn_decode),
        "reshape_and_cache_a5_plan": PlannedRecipe(_build_cache_a5),
    }


__all__ = [
    "CAPABILITIES",
    "FUSED_GROUPED_RECIPE",
    "XLLM_CUSTOM_OPS",
    "XLLM_VENDOR",
    "capabilities",
    "fused_region_contract",
    "lower_op",
    "recipes",
]
