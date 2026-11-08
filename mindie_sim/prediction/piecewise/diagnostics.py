"""Explain piecewise regimes, dominant pipes, boundaries and fidelity.

[Role]
  Export a bounded, data-traceable diagnostic view of a compiled subgraph:
  the selected segment, guard cells, dominant/tied pipes, work, crossover
  thresholds, materialisation and the declared fidelity class.

[Invariants]
  * Diagnostics never invent a measured threshold or a serving latency.  A
    reported crossover is a root of declared expressions; an unsupported root
    is reported as absent with its reason.
  * Fidelity labels keep ``resource_lower_bound``, isolated device time and
    graph/serving spans distinct.
  * Export is bounded by an explicit entry limit and contains no request ids
    or model labels in the performance key.
  * Boundary distances are reported on the discrete cell, not smoothed into a
    continuous seam.

[Timeouts & Locks]
  Bounded per-operator work and crossover solving.  No I/O or locks.

[Side-Effect Policy]
  Returns immutable report records; no filesystem/network writes.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError
from .compiler import (
    AnalysisDomain, CompiledOperator, CompiledSubgraph, OperatorCost,
    WorkloadGeometry,
)
from .expressions import as_fraction
from .guards import Crossover, PipeDemand, PipeDominanceGuard


@dataclass(frozen=True)
class OperatorDiagnostic:
    name: str
    family: str
    segment: int
    regime: tuple[tuple[str, int], ...]
    guards: tuple[dict[str, Any], ...]
    dominant: str
    tied: tuple[str, ...]
    duration_ms: Fraction
    work: tuple[tuple[str, Fraction], ...]
    times: tuple[tuple[str, Fraction], ...]
    crossovers: tuple[Crossover, ...]
    materialization: tuple[tuple[str, Fraction], ...]
    formula: str
    fidelity: str = "isolated_device_time"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "family": self.family,
            "segment": self.segment, "regime": dict(self.regime),
            "guards": list(self.guards), "dominant": self.dominant,
            "tied": list(self.tied), "duration_ms": str(self.duration_ms),
            "work": {key: str(value) for key, value in self.work},
            "times_ms": {key: str(value) for key, value in self.times},
            "crossovers": [item.to_dict() for item in self.crossovers],
            "materialization": {key: str(value) for key, value in self.materialization},
            "formula": self.formula, "fidelity": self.fidelity,
        }


@dataclass(frozen=True)
class DiagnosticReport:
    instance_id: str
    kind: str
    total_ms: Fraction
    operators: tuple[OperatorDiagnostic, ...]
    fidelity: str = "resource_lower_bound"
    assumptions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id, "kind": self.kind,
            "total_ms": str(self.total_ms), "fidelity": self.fidelity,
            "assumptions": list(self.assumptions),
            "operators": [operator.to_dict() for operator in self.operators],
        }


def _crossovers(operator: CompiledOperator, geometry: WorkloadGeometry,
                state: Any, cost: OperatorCost) -> tuple[Crossover, ...]:
    found: list[Crossover] = []
    if not cost.tied_pipes:
        return ()
    try:
        env = operator._env(geometry, state)
    except ModelingError:
        return ()
    coordinate = "T"
    pinned = {key: value for key, value in env.items() if key != coordinate}
    for stage_index in range(len(operator.program.stages)):
        guard = operator.dominance(geometry, state, stage=stage_index)
        try:
            times = guard.times(env)
        except ModelingError:
            continue
        peak = max(times.values())
        leaders = [pipe for pipe, value in times.items() if value == peak]
        for leader in leaders:
            for other in operator.program.stages[stage_index]:
                if other.pipe == leader:
                    continue
                try:
                    found.extend(guard.crossover(leader, other.pipe, coordinate,
                                                 at=pinned))
                except ModelingError:
                    continue
    return tuple(found)


def explain(cost_function: Any, geometry: WorkloadGeometry,
            state: Any = None) -> DiagnosticReport:
    state = state if state is not None else _empty_state()
    compiled = cost_function.compiled
    evaluation = compiled.evaluate(geometry, state)
    records = []
    for operator, cost in zip(compiled.operators, evaluation.operators):
        guards = tuple({
            "axis": axis.name,
            "coordinate": axis.coordinate,
            "cell": cell.to_dict(),
        } for axis, cell in zip(operator.axes, cost.guard_cells))
        records.append(OperatorDiagnostic(
            name=cost.name, family=cost.family, segment=cost.segment_index,
            regime=cost.regime, guards=guards, dominant=cost.dominant_pipe,
            tied=cost.tied_pipes, duration_ms=cost.duration_ms,
            work=cost.work, times=cost.times,
            crossovers=_crossovers(operator, geometry, state, cost),
            materialization=cost.materialization, formula=cost.formula,
            fidelity=cost.fidelity))
    return DiagnosticReport(
        compiled.instance.instance_id, compiled.instance.kind,
        evaluation.total_ms, tuple(records),
        assumptions=("declared structural formulas", "no serving floors",
                     "single Engine for contention"))


def coverage_summary(instantiated: Any, *, limit: int = 4096) -> dict[str, Any]:
    if limit < 1:
        raise ModelingError("diagnostic_limit_invalid")
    instances = instantiated.subgraphs[:limit]
    entries = [{"instance_id": instance.instance_id, "kind": instance.kind,
                "variant": instance.variant, "stage": instance.stage,
                "owner_layer": instance.owner_layer,
                "fingerprint": instance.fingerprint()}
               for instance in instances]
    report = instantiated.coverage
    return {
        "instances": entries,
        "truncated": len(instantiated.subgraphs) > len(instances),
        "kinds": sorted({instance.kind for instance in instances}),
        "holes": [hole.__dict__ for hole in report.holes],
        "complete": report.complete,
    }


def regime_summary(cost: OperatorCost) -> str:
    axes = ", ".join(f"{name}={value}" for name, value in cost.regime)
    return f"{cost.name}[{axes}] -> {cost.dominant_pipe} ({cost.duration_ms})"


def boundary_distance(operator: CompiledOperator, axis_name: str, value: int) -> dict[str, Any]:
    for axis in operator.axes:
        if axis.name != axis_name:
            continue
        cell = axis.cell(axis.index(value))
        lower = cell.lower
        upper = cell.upper
        exact = as_fraction(value)
        return {
            "axis": axis_name, "value": value,
            "lower": None if lower is None else str(lower),
            "upper": None if upper is None else str(upper),
            "distance_lower": None if lower is None else str(exact - lower),
            "distance_upper": None if upper is None else str(upper - exact),
        }
    raise ModelingError(f"segment_axis_missing:{operator.name}:{axis_name}")


def _empty_state():
    from mindie_sim.prediction.subgraphs.state import StateView
    return StateView()
