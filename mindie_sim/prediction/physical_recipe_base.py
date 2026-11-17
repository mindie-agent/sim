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
    return 0.0 if total <= 0.0 else float(value) / float(total)


def work_from_op(op) -> Work:
    """Recover the declared physical work without re-pricing the operator."""
    matrix = dict(getattr(op, "matrix_flops", ()) or ())
    resources = {}
    for field, role in (("hbm_bytes", "hbm"), ("mte2_bytes", "mte2"),
                        ("mte3_bytes", "mte3"), ("vector_ops", "vector"),
                        ("link_bytes", "link")):
        value = float(getattr(op, field, 0.0) or 0.0)
        if value:
            resources[role] = resources.get(role, 0.0) + value
    metadata = dict(getattr(op, "metadata", ()) or ())
    for role, value in dict(metadata.get("resource_work", {})).items():
        resources[role] = resources.get(role, 0.0) + float(value)
    return Work(matrix, resources)


#: Canonical phase category for each declared physical resource role.  Both
#: split MTE2 read pipes belong to one ``mte2`` phase category because they
#: serve the same operand-fetch phases, but each pipe conserves its own work.
CATEGORY_BY_ROLE = {
    "hbm": "hbm",
    "cube": "cube",
    "vector": "vector",
    "mte2": "mte2",
    "mte2_aic": "mte2",
    "mte2_aiv": "mte2",
    "mte3": "mte3",
    "link": "link",
    "mte1": "mte1",
    "mte1_l0a": "mte1",
    "mte1_l0b": "mte1",
    "fixpipe": "fixpipe",
}


def _category(role: str) -> str | None:
    """One canonical phase category per physical resource, or ``None``."""
    if role in CATEGORY_BY_ROLE:
        return CATEGORY_BY_ROLE[role]
    for prefix, category in (("memory@", "hbm"), ("matrix@", "cube"),
                             ("mte2", "mte2"), ("mte3", "mte3"),
                             ("vector", "vector"), ("link", "link"),
                             ("mte1", "mte1"), ("fixpipe", "fixpipe")):
        if role.startswith(prefix):
            return category
    # Topology-owned link resources carry a namespace and endpoint identity
    # (``<scope>:link:<channel>:<rank>:<endpoint>``); their category is the
    # link token, not the caller's namespace.
    tokens = role.split(":")
    if "link" in tokens:
        return "link"
    return None


def _unmodeled(context, reason: str, name: str):
    family = str(getattr(context.op, "family", ""))
    recipe = str(getattr(context.op, "recipe", ""))
    raise ModelingError(
        f"unmodeled_operator_mechanism:{family}:{recipe}:{reason}:{name}")


def _resource_categories(context) -> dict[str, str]:
    """Map every nonzero declared resource to exactly one phase category."""
    by_name: dict[str, str] = {}
    for role, name in dict(context.resource_names).items():
        category = _category(str(role))
        previous = by_name.get(str(name))
        if category is None:
            continue
        if previous is not None and previous != category:
            _unmodeled(context, "resource_category_conflict", str(name))
        by_name[str(name)] = category
    result: dict[str, str] = {}
    for name, value in dict(context.work).items():
        if float(value) <= 0.0:
            continue
        key = str(name)
        if key.startswith("host") or key.startswith("scheduler@"):
            # Host submission latency is not a device pipe and is owned by the
            # Engine's submission boundary, not by a device phase.
            continue
        category = by_name.get(key) or _category(key)
        if category is None:
            _unmodeled(context, "uncategorized_resource", key)
        result[key] = category
    return result


def _phase_amounts(context, phases: Sequence[Phase]) -> list[dict[str, float]]:
    """Allocate each physical resource independently across declared phases.

    ``W[s, r] = W[r] * alpha[s, c(r)] / sum_j alpha[j, c(r)]``.  The per-name
    loop is deliberately outside the phase loop: two directed links or two
    split read pipes keep their own work, so congestion and conservation are
    per physical resource rather than an equal share of a category total.
    """
    categories = _resource_categories(context)
    result = [dict() for _phase in phases]
    for name, category in categories.items():
        total = float(context.work[name])
        if total <= 0.0:
            continue
        weights = [max(0.0, float(phase.weights.get(category, 0.0)))
                   for phase in phases]
        denominator = math.fsum(weights)
        if denominator <= 0.0:
            _unmodeled(context, "resource_without_consuming_phase", name)
        positive = [index for index, weight in enumerate(weights)
                    if weight > 0.0]
        last = positive[-1]
        allocated = 0.0
        for index, weight in enumerate(weights):
            if index == last:
                amount = total - allocated
            elif weight <= 0.0:
                amount = 0.0
            else:
                amount = total * weight / denominator
            allocated += amount
            if amount > 0.0:
                result[index][name] = result[index].get(name, 0.0) + amount
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
                if value > 0.0}
        stages.append(Stage(phase.name, work, after=phase.after,
                            keep_empty=True))
    return KernelProgram(tuple(stages),
                         minimum_ms=(context.minimum_ms
                                     if minimum_ms is None else minimum_ms))


def _tile_count(context, *, cap: int = MAX_TILES) -> int:
    op = context.op
    units = int(context.resource_units.get(
        "vector", max(1, context.core_count)))
    bytes_to_cover = max(float(getattr(op, "mte2_bytes", 0.0) or 0.0),
                         float(getattr(op, "hbm_bytes", 0.0) or 0.0))
    needed = max(1, int(math.ceil(bytes_to_cover / MIN_TILE_BYTES)))
    return max(1, min(cap, units, needed))


def tiled_phased_program(context, phases: Sequence[Phase], *,
                         slots: int = DEFAULT_PIPELINE_SLOTS, cap: int = MAX_TILES) -> KernelProgram:
    """Run the same phase DAG as a bounded finite tile pipeline."""
    count = _tile_count(context, cap=cap)
    if count <= 1:
        return phased_program(context, phases)
    amounts = _phase_amounts(context, phases)

    def tile(index):
        stages = []
        for phase, amount in zip(phases, amounts):
            start = float(index) / count
            end = float(index + 1) / count
            work = {name: value * (end - start)
                    for name, value in amount.items()
                    if value > 0.0}
            stages.append(Stage(phase.name, work, after=phase.after,
                                keep_empty=True))
        return stages

    pipeline = PipelineProgram(count, max(1, int(slots)), tile,
                               minimum_ms=context.minimum_ms)
    return KernelProgram((Stage("setup", latency=context.setup_ms),
                          Stage("tiles", after=("setup",), program=pipeline)),
                         minimum_ms=context.minimum_ms)


def io_weight(context, category: str) -> float:
    """Fraction of HBM traffic that is a payload read or write."""
    op = context.op
    total = float(getattr(op, "hbm_bytes", 0.0) or 0.0)
    if total <= 0.0:
        return 1.0
    write = float(getattr(op, "mte3_bytes", 0.0) or 0.0)
    if category == IO_HBM_CATEGORY:
        return 1.0
    if category == "hbm_write":
        return min(1.0, max(0.0, write / total))
    return min(1.0, max(0.0, (total - write) / total))


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
