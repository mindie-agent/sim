"""Shared finite-stage lowering for the five physical mechanism classes.

[Role] Partition an operator's declared resource demand into named physical
  phases and bounded tile pipelines without changing its total work.
[Invariants] Every phase consumes only the resources assigned to it; the sum of
  issued stage work equals the declared plan work per physical resource. A tile
  is a finite occurrence, never a steady-state throughput claim.
[Timeouts & Locks] Bounded phase/tile counts, no I/O or locks.
[Side-Effect Policy] Returns immutable programs and submits only through the
  caller's Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, Work
from .program import KernelProgram, PipelineProgram, Stage

ZERO = 0
ONE = 1
ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
MAX_TILES = 64
MIN_TILE_BYTES = 4096
DEFAULT_AIV_TILES = 64
DEFAULT_PIPELINE_SLOTS = 2
IO_HBM_CATEGORY = "hbm"
IO_READ_CATEGORY = "mte2"
IO_WRITE_CATEGORY = "mte3"
COMPUTE_CATEGORIES = ("cube", "vector")
LINK_CATEGORY = "link"


@dataclass(frozen=True)
class Phase:
    name: str
    weights: Mapping[str, float]
    after: tuple[str, ...] = ()


def _fraction(value: float, total: float) -> float:
    return ZERO_FLOAT if total <= ZERO_FLOAT else float(value) / float(total)


def work_from_op(op) -> Work:
    """Recover the declared physical work without re-pricing the operator."""
    matrix = dict(getattr(op, "matrix_flops", ()) or ())
    resources = {}
    for field, role in (("hbm_bytes", "hbm"), ("mte2_bytes", "mte2"),
                        ("mte3_bytes", "mte3"), ("vector_ops", "vector"),
                        ("link_bytes", "link")):
        value = float(getattr(op, field, ZERO_FLOAT) or ZERO_FLOAT)
        if value:
            resources[role] = resources.get(role, ZERO_FLOAT) + value
    metadata = dict(getattr(op, "metadata", ()) or ())
    for role, value in dict(metadata.get("resource_work", {})).items():
        resources[role] = resources.get(role, ZERO_FLOAT) + float(value)
    return Work(matrix, resources)


def _work_keys(context):
    names = dict(context.resource_names)
    by_category: dict[str, list[str]] = {}
    seen: set[str] = set()
    for role, name in names.items():
        if name in context.work and name not in seen:
            by_category.setdefault(role, []).append(name)
            seen.add(name)
    # Service-mode and physical prefixes use names that are not logical roles.
    for key in context.work:
        category = None
        if key.startswith("memory@"):
            category = "hbm"
        elif key.startswith("vector"):
            category = "vector"
        elif key.startswith("matrix@"):
            category = "cube"
        elif key.startswith("mte2"):
            category = "mte2"
        elif key.startswith("mte3"):
            category = "mte3"
        elif key.startswith("link"):
            category = "link"
        if category is not None and key not in by_category.setdefault(category, []):
            by_category[category].append(key)
    return by_category


def _phase_amounts(context, phases: Sequence[Phase]) -> list[dict[str, float]]:
    """Allocate each physical resource across phases by declared weights."""
    by_category = _work_keys(context)
    totals = {category: math.fsum(float(context.work[name]) for name in names)
              for category, names in by_category.items()}
    result = [dict() for _phase in phases]
    for category, names in by_category.items():
        weights = [max(ZERO_FLOAT, float(phase.weights.get(category, ZERO_FLOAT)))
                   for phase in phases]
        denominator = math.fsum(weights)
        if denominator <= ZERO_FLOAT:
            continue
        allocated = ZERO_FLOAT
        for index, weight in enumerate(weights):
            amount = (totals[category] - allocated
                      if index == len(weights) - ONE
                      else totals[category] * weight / denominator)
            allocated += amount
            share = amount / float(len(names))
            for name in names:
                result[index][name] = result[index].get(name, ZERO_FLOAT) + share
    return result


def phased_program(context, phases: Sequence[Phase], *,
                   minimum_ms: float | None = None) -> KernelProgram:
    """Lower named phases with explicit dependencies and conserved work."""
    if not phases:
        return KernelProgram((Stage("empty", keep_empty=True),),
                             minimum_ms=(context.minimum_ms
                                         if minimum_ms is None else minimum_ms))
    amounts = _phase_amounts(context, phases)
    stages = []
    for phase, amount in zip(phases, amounts):
        work = {name: value for name, value in amount.items()
                if value > ZERO_FLOAT}
        stages.append(Stage(phase.name, work, after=phase.after,
                            keep_empty=True))
    return KernelProgram(tuple(stages),
                         minimum_ms=(context.minimum_ms
                                     if minimum_ms is None else minimum_ms))


def _tile_count(context, *, cap: int = MAX_TILES) -> int:
    op = context.op
    units = int(context.resource_units.get(
        "vector", max(ONE, context.core_count)))
    bytes_to_cover = max(float(getattr(op, "mte2_bytes", ZERO_FLOAT) or ZERO_FLOAT),
                         float(getattr(op, "hbm_bytes", ZERO_FLOAT) or ZERO_FLOAT))
    needed = max(ONE, int(math.ceil(bytes_to_cover / MIN_TILE_BYTES)))
    return max(ONE, min(cap, units, needed))


def tiled_phased_program(context, phases: Sequence[Phase], *,
                         slots: int = DEFAULT_PIPELINE_SLOTS, cap: int = MAX_TILES) -> KernelProgram:
    """Run the same phase DAG as a bounded finite tile pipeline."""
    count = _tile_count(context, cap=cap)
    if count <= ONE:
        return phased_program(context, phases)
    amounts = _phase_amounts(context, phases)

    def tile(index):
        stages = []
        for phase, amount in zip(phases, amounts):
            start = float(index) / count
            end = float(index + ONE) / count
            work = {name: value * (end - start)
                    for name, value in amount.items()
                    if value > ZERO_FLOAT}
            stages.append(Stage(phase.name, work, after=phase.after,
                                keep_empty=True))
        return stages

    pipeline = PipelineProgram(count, max(ONE, int(slots)), tile,
                               minimum_ms=context.minimum_ms)
    return KernelProgram((Stage("setup", latency=context.setup_ms),
                          Stage("tiles", after=("setup",), program=pipeline)),
                         minimum_ms=context.minimum_ms)


def io_weight(context, category: str) -> float:
    """Fraction of HBM traffic that is a payload read or write."""
    op = context.op
    total = float(getattr(op, "hbm_bytes", ZERO_FLOAT) or ZERO_FLOAT)
    if total <= ZERO_FLOAT:
        return ONE_FLOAT
    write = float(getattr(op, "mte3_bytes", ZERO_FLOAT) or ZERO_FLOAT)
    if category == IO_HBM_CATEGORY:
        return ONE_FLOAT
    if category == "hbm_write":
        return min(ONE_FLOAT, max(ZERO_FLOAT, write / total))
    return min(ONE_FLOAT, max(ZERO_FLOAT, (total - write) / total))


def build_plan(recipe, op, geometry) -> OperatorPlan:
    """Expose the canonical plan before a lowering context exists."""
    del geometry
    work = work_from_op(op)
    return OperatorPlan(work, recipe.lower, metadata={
        "physical_class": type(recipe).__name__,
        "stage_basis": "declared_physical_phase_partition",
    })


def require_mechanism(value, name: str):
    if value is None:
        raise ModelingError(f"physical_mechanism_required:{name}")
    return value
