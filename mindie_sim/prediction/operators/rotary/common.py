"""Unified rotary position embedding work, pipeline spec, and AIV lowering.

[Role] Provide the shared shape/byte work conservation, ``FusedPipelineSpec``
  and finite AIV tile lowering for standard rotate-half RoPE across Ascend
  architectures without ISA-specific leaf duplication.
[Invariants] Preserves explicitly declared ``OpSpec`` work verbatim for traffic
  profile rewrites; derives conserved HBM/MTE2/MTE3/vector work from local
  shape and dtype when standalone shape-only ops are provided. Binds sustainable
  AIV DRAM ceiling via ``RecipeContext.service_rate`` without ``service_modes``
  capability sniffing.
[Timeouts & Locks] O(1) work arithmetic and bounded AIV tile staging; no I/O
  or locks.
[Side-Effect Policy] Pure immutable ``OperatorPlan`` and descriptor records.
"""
from __future__ import annotations

from dataclasses import replace
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ...physical_recipe_base import work_from_op
from ...program import RecipeContext
from ...results import (
    FusedPipelineDescriptor,
    FusedPipelineSpec,
    PipelinePhase,
    WorkItem,
)
from ...vector_pipeline import AIVTiledVectorAndGatherRecipe
from ..contracts import OperatorPlanningContext

IMPLEMENTATION = "rotary_position_embedding"
UPSTREAM_REVISION = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
KERNEL_SOURCE_SHA256 = (
    "5160d08c053ba32694df0c699d31ddc9b67a26d3ae4545f4fb9d6122efc3400e"
)
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision=UPSTREAM_REVISION,
        relative_path=(
            "posembedding/rotary_position_embedding/op_kernel/"
            "rotary_position_embedding.cpp"
        ),
        source_sha256=KERNEL_SOURCE_SHA256,
    ),
)
SUPPORTED_ABIS: frozenset[str] = frozenset({
    "aclnnRotaryPositionEmbedding",
})

# Rotate-half per element (ops-transformer rotate_half.h:124 & rotate_half_base.h:294-296):
# Muls(sin, -1, halfD) = 0.5 + Mul(x, cos) = 1.0 + Mul(xNew, sin) = 1.0 + Add(xNew, x) = 1.0.
ROTARY_OPS_PER_ELEMENT = 3.5
COS_SIN_TENSOR_COUNT = 2
BSND_MIN_RANK = 3
TWO_DIM_RANK = 2
MS_PER_SECOND = 1000.0
AIV_DRAM_SERVICE_MODE = "mte2_aiv_dram"

_AIV_VECTOR_RECIPE = AIVTiledVectorAndGatherRecipe()


def _has_declared_work(work: Work) -> bool:
    return any(v > 0.0 for v in work.matrix.values()) or any(
        v > 0.0 for v in work.resources.values()
    )


def _derive_rotary_work(op: OpSpec) -> Work:
    """Derive exact rotate-half byte/vector work from local shape and dtype."""
    shape = tuple(int(dim) for dim in (op.local_shape or op.semantic_shape or ()))
    if not shape or any(dim < 0 for dim in shape):
        raise ModelingError(f"rotary_invalid_shape:{op.op_id}:{shape}")
    if any(dim == 0 for dim in shape):
        return Work()
    elem_bytes = dtype_bytes(op.dtype or "bf16")
    elements = math.prod(shape)
    rotary_dim = shape[-1]
    if len(shape) >= BSND_MIN_RANK:
        tokens = math.prod(shape[:-2])
    elif len(shape) == TWO_DIM_RANK:
        tokens = shape[0]
    else:
        tokens = 1
    tensor_bytes = float(elements * elem_bytes)
    coeff_bytes = float(COS_SIN_TENSOR_COUNT * tokens * rotary_dim * elem_bytes)
    read_bytes = tensor_bytes + coeff_bytes
    write_bytes = tensor_bytes
    return Work(resources={
        "hbm": read_bytes + write_bytes,
        "mte2": read_bytes,
        "mte3": write_bytes,
        "vector": float(elements * ROTARY_OPS_PER_ELEMENT),
    })


def _bind_aiv_dram_ceiling(context: RecipeContext) -> RecipeContext:
    """Apply declared sustainable AIV DRAM ceiling via named service rate lookup."""
    rate = context.service_rate(AIV_DRAM_SERVICE_MODE)
    if rate is None:
        return context
    hbm_name = context.resource_names.get("hbm")
    if hbm_name is None or hbm_name not in context.rates:
        return context
    ceiling = min(rate, float(context.rates[hbm_name]) * MS_PER_SECOND)
    return replace(
        context,
        service_modes={**context.service_modes, "hbm": ("hbm", ceiling)},
    )


def _pipeline_spec(work: Work, dtype: str) -> FusedPipelineSpec:
    hbm_total = float(work.resources.get("hbm", 0.0))
    mte2_bytes = float(
        work.resources.get("mte2", work.resources.get("mte2_aiv", 0.0))
    )
    mte3_bytes = float(work.resources.get("mte3", 0.0))
    vec_ops = float(work.resources.get("vector", 0.0))
    # Match physical_recipe_base.io_weight / vector_pipeline.py HBM read/write split.
    write_hbm = min(hbm_total, max(0.0, mte3_bytes)) if hbm_total > 0.0 else 0.0
    read_hbm = max(0.0, hbm_total - write_hbm)
    phases = (
        PipelinePhase(
            phase_id="tile_load",
            after=(),
            work=(
                WorkItem("hbm", "bytes", read_hbm, dtype),
                WorkItem("mte2_aiv", "bytes", mte2_bytes, dtype),
            ),
        ),
        PipelinePhase(
            phase_id="tile_compute",
            after=("tile_load",),
            work=(WorkItem("vector", "op", vec_ops, dtype),),
        ),
        PipelinePhase(
            phase_id="tile_store",
            after=("tile_compute",),
            work=(
                WorkItem("hbm", "bytes", write_hbm, dtype),
                WorkItem("mte3", "bytes", mte3_bytes, dtype),
            ),
        ),
    )
    return FusedPipelineSpec(
        family="rotary",
        implementation=IMPLEMENTATION,
        phases=phases,
        sources=SOURCES,
        repeats=1,
    )


def build(op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
    """Build the conserved AIV rotary ``OperatorPlan`` with attached ``FusedPipelineSpec``."""
    declared = work_from_op(op)
    work = declared if _has_declared_work(declared) else _derive_rotary_work(op)
    dtype = str(op.dtype or "bf16")
    spec = _pipeline_spec(work, dtype)
    return OperatorPlan(
        work=work,
        program=_AIV_VECTOR_RECIPE.lower,
        metadata={
            "implementation": IMPLEMENTATION,
            "arch_id": str(ctx.compute_arch.arch_id),
            "governed_family": "rotary",
        },
        pipeline_spec=spec,
        context_binder=_bind_aiv_dram_ceiling,
        traffic_rewrite_capable=True,
    )


def describe(
    op: OpSpec,
    ctx: OperatorPlanningContext,
    plan: OperatorPlan,
    /,
) -> FusedPipelineDescriptor | None:
    """Project an occurrence-bound ``FusedPipelineDescriptor`` from ``plan``."""
    spec = plan.pipeline_spec
    if not isinstance(spec, FusedPipelineSpec):
        return None
    return FusedPipelineDescriptor(
        owner=ctx.occurrence,
        family=spec.family,
        implementation=spec.implementation,
        member_op_ids=(op.op_id,),
        absorbed_source_op_ids=(),
        phases=spec.phases,
        sources=spec.sources,
        repeats=spec.repeats,
    )


__all__ = [
    "IMPLEMENTATION",
    "SOURCES",
    "SUPPORTED_ABIS",
    "build",
    "describe",
]
