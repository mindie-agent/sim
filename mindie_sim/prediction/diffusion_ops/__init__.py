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
from typing import Any
from typing import Mapping as Mapping

from mindie_sim.schema.common import (
    ModelingError as ModelingError,
    align_up as align_up,
    dtype_bytes as dtype_bytes,
)
from mindie_sim.schema.operators import OpSpec as OpSpec

from ..mechanism import OperatorPlan as OperatorPlan, PlannedRecipe, Work as Work
from ..program import KernelProgram as KernelProgram, Stage as Stage
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
    lower_mindie_sd_custom_op,
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


def recipes() -> dict[str, Any]:
    """Register one typed plan per composite MindIE-SD occurrence."""
    return {f"mindie_sd_{name}": PlannedRecipe(_build_contract_plan)
            for name in MINDIE_SD_CUSTOM_OPS}


__all__ = [
    "CAPABILITIES",
    "MINDIE_SD_CUSTOM_OPS",
    "MINDIE_SD_VENDOR",
    "capabilities",
    "compute_mindie_sd_effective_denoising_steps",
    "lower_mindie_sd_custom_op",
    "lower_op",
    "recipes",
]
