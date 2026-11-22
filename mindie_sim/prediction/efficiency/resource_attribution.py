"""Exposed-memory attribution and fixed-plan counterfactuals.

[Role] Propagate the four ordered service clocks through a realized physical
  stage DAG, attribute the terminal clocks to Cube/Vector/HBM/Link, and report
  the exact same-plan zero-HBM counterfactual when an execution boundary
  reconciles with the plan.  Service times, never byte/FLOP proportions, drive
  the propagation.
[Invariants] A clock is ``(all, without_communication, compute, cube)``; a
  physical stage contributes ``(max(C,V,H,L), max(C,V,H), max(C,V), C)`` and
  serial composition adds service times instead of rescaling them.  Terminal
  clocks are componentwise maxima over the realized DAG sinks.  HBM exposure is
  ``T_baseline - T_same_plan_zero_hbm``; when the boundary or counterfactual is
  not available the result is ``None``, never a fabricated proportion.  Buckets
  remain nonnegative and telescope to the reported span when the graph is total.
[Timeouts & Locks] Pure arithmetic over bounded node sets; no I/O or locks.
[Side-Effect Policy] Returns new mappings; does not mutate reports or plans.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.efficiency.exposed_breakdown import (
    CLOCK_GAP_ABS_FLOOR_MS,
    CLOCK_GAP_LIMIT_FRACTION,
    ExposedClocks,
    finish_breakdown,
    roofline_clocks,
)
from mindie_sim.prediction.efficiency.roofline_graph import (
    RESOURCES,
    PhysicalWorkGraph,
    PhysicalWorkNode,
    physical_graph_order,
)

DEFAULT_CUBE_COVER = 1.0
DEFAULT_VECTOR_COVER = 1.0
EXPOSURE_RECONCILE_FRACTION = 1e-3
EXPOSURE_RECONCILE_FLOOR_MS = 1e-6


@dataclass(frozen=True)
class ExposedHbm:
    memory_ms: float
    cube_covered_ms: float
    vector_covered_ms: float
    exposed_ms: float
    basis: str

    def to_dict(self) -> dict:
        return {
            "memory_ms": self.memory_ms,
            "cube_covered_ms": self.cube_covered_ms,
            "vector_covered_ms": self.vector_covered_ms,
            "exposed_ms": self.exposed_ms,
            "basis": self.basis,
        }


def estimate_exposed_hbm(memory_ms: float, cube_ms: float, vector_ms: float, *,
                         cube_cover_fraction: float,
                         vector_cover_fraction: float) -> float:
    """Return ``max(0, mem - eta_C*cube - eta_V*vector)`` without clipping inputs."""
    values = (memory_ms, cube_ms, vector_ms)
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ModelingError("exposed_hbm_requires_nonnegative_finite_service")
    for name, value in (("cube_cover_fraction", cube_cover_fraction),
                        ("vector_cover_fraction", vector_cover_fraction)):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ModelingError(f"exposed_hbm_cover_fraction_invalid:{name}")
    covered = (float(cube_cover_fraction) * float(cube_ms)
               + float(vector_cover_fraction) * float(vector_ms))
    return max(0.0, float(memory_ms) - covered)


def join_clocks(terminal_clocks: Sequence[ExposedClocks]) -> ExposedClocks:
    """Componentwise maximum of terminal clocks; empty terminals are all zero."""
    clocks = tuple(terminal_clocks)
    if not clocks:
        return ExposedClocks(0.0, 0.0, 0.0, 0.0)
    for clock in clocks:
        if not isinstance(clock, ExposedClocks):
            raise ModelingError("resource_attribution_clock_required")
    return ExposedClocks(
        max(clock.all_ms for clock in clocks),
        max(clock.without_communication_ms for clock in clocks),
        max(clock.compute_ms for clock in clocks),
        max(clock.cube_ms for clock in clocks),
    )


def advance_clocks(predecessor_clocks: Sequence[ExposedClocks],
                   local: ExposedClocks) -> ExposedClocks:
    """Join predecessor clocks, then add this stage's service clocks.

    The addition is serial service-time composition; it never rescales, divides
    or redistributes a memory phase into a compute phase.
    """
    if not isinstance(local, ExposedClocks):
        raise ModelingError("resource_attribution_local_clock_required")
    joined = join_clocks(predecessor_clocks)
    return ExposedClocks(
        joined.all_ms + local.all_ms,
        joined.without_communication_ms + local.without_communication_ms,
        joined.compute_ms + local.compute_ms,
        joined.cube_ms + local.cube_ms,
    )


def node_local_clocks(node: PhysicalWorkNode) -> ExposedClocks:
    """Service clocks of one physical work node, including explicit control."""
    if not isinstance(node, PhysicalWorkNode):
        raise ModelingError("physical_local_clocks_require_physical_node")
    service = roofline_clocks(
        cube_ms=node.service_ms("cube"),
        vector_ms=node.service_ms("vector"),
        memory_ms=node.service_ms("hbm"),
        communication_ms=node.service_ms("link"),
    )
    control = max(0.0, node.ideal_ms - service.all_ms)
    return ExposedClocks(
        service.all_ms + control,
        service.without_communication_ms + control,
        service.compute_ms + control,
        service.cube_ms,
    )


def _clock_order(nodes: Iterable[str],
                 predecessors: Mapping[str, Sequence[str]]) -> tuple[str, ...]:
    keys = tuple(nodes)
    successors = {node_id: [] for node_id in keys}
    indegree = {}
    for node_id in keys:
        indegree[node_id] = len(predecessors.get(node_id, ()))
        for predecessor in predecessors.get(node_id, ()):
            successors[predecessor].append(node_id)
    ready = [node_id for node_id in keys if indegree[node_id] == 0]
    order: list[str] = []
    cursor = 0
    while cursor < len(ready):
        node_id = ready[cursor]
        cursor += 1
        order.append(node_id)
        for successor in successors[node_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    if len(order) != len(keys):
        raise ModelingError("realized_clock_dag_cycle")
    return tuple(order)


def realized_clock_dag(
    local_by_node: Mapping[str, ExposedClocks],
    predecessors: Mapping[str, Sequence[str]] | None = None,
    *,
    allow_external_predecessors: bool = False,
) -> tuple[dict[str, ExposedClocks], ExposedClocks]:
    """Propagate local service clocks over one realized physical-stage DAG.

    ``predecessors`` must contain the actual data, stream, state and
    collective-rendezvous ordering.  Unknown predecessors are an error unless
    ``allow_external_predecessors`` is set, which is the explicit cut boundary
    case.  Returns ``(clocks_by_node, terminal_clocks)`` where the terminal is
    the componentwise join over DAG sinks.
    """
    for node_id, local in local_by_node.items():
        if not isinstance(local, ExposedClocks):
            raise ModelingError(
                f"realized_clock_dag_local_clock_required:{node_id}")
    normalized: dict[str, tuple[str, ...]] = {}
    for node_id in local_by_node:
        selected = []
        for predecessor in (predecessors or {}).get(node_id, ()):
            predecessor = str(predecessor)
            if predecessor not in local_by_node:
                if allow_external_predecessors:
                    continue
                raise ModelingError(
                    f"realized_clock_dag_unknown_predecessor:{predecessor}")
            if predecessor not in selected:
                selected.append(predecessor)
        normalized[node_id] = tuple(selected)
    order = _clock_order(local_by_node, normalized)
    clocks: dict[str, ExposedClocks] = {}
    successors = {node_id: 0 for node_id in local_by_node}
    for node_id in order:
        clocks[node_id] = advance_clocks(
            [clocks[predecessor] for predecessor in normalized[node_id]],
            local_by_node[node_id])
        for predecessor in normalized[node_id]:
            successors[predecessor] += 1
    terminals = [clocks[node_id] for node_id in order
                 if successors[node_id] == 0]
    return clocks, join_clocks(terminals)


def fixed_plan_exposure(plan: PhysicalWorkGraph) -> Mapping[str, object]:
    """Same-plan zero-HBM counterfactual under the graph's dependency order."""
    if not isinstance(plan, PhysicalWorkGraph):
        raise ModelingError("fixed_plan_exposure_requires_physical_graph")
    nodes = {node.node_id: node for node in plan.nodes}
    predecessors = plan.predecessors()
    order = physical_graph_order(plan)
    baseline: dict[str, float] = {}
    zero_hbm: dict[str, float] = {}
    for node_id in order:
        node = nodes[node_id]
        baseline[node_id] = node.ideal_ms + max(
            (baseline[predecessor]
             for predecessor in predecessors.get(node_id, ())),
            default=0.0)
        zero_own = float(node.occurrences) * max(
            node.launch_ms,
            node.fixed_ms + max(node.cube_ms, node.vector_ms, node.link_ms))
        zero_hbm[node_id] = zero_own + max(
            (zero_hbm[predecessor]
             for predecessor in predecessors.get(node_id, ())),
            default=0.0)
    baseline_span = max(baseline.values(), default=0.0)
    zero_span = max(zero_hbm.values(), default=0.0)
    delta = exposed_hbm_delta_ms(baseline_span, zero_span)
    return {
        "baseline_ms": baseline_span,
        "zero_hbm_service_ms": zero_span,
        "delta_h_ms": (delta if delta is not None else 0.0),
        "available": delta is not None,
        "basis": "same_plan_same_dependencies_zero_hbm_service",
    }


def _boundary_reconciles(first: float, second: float) -> bool:
    tolerance = max(EXPOSURE_RECONCILE_FLOOR_MS,
                    EXPOSURE_RECONCILE_FRACTION * max(abs(first), abs(second)))
    return abs(first - second) <= tolerance


def exposed_hbm_delta_ms(baseline_ms, zero_hbm_ms) -> float | None:
    """Return ``T_baseline - T_same_plan_zero_hbm`` or ``None`` if unavailable."""
    numbers = []
    for value in (baseline_ms, zero_hbm_ms):
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number < 0.0:
            return None
        numbers.append(number)
    baseline, zero = numbers
    tolerance = max(EXPOSURE_RECONCILE_FLOOR_MS,
                    EXPOSURE_RECONCILE_FRACTION * max(baseline, zero))
    if zero > baseline + tolerance:
        return None
    return max(0.0, baseline - zero)


def attribute_stage_dag(
    plan: PhysicalWorkGraph,
    *,
    boundary: str = "stage_dag",
    duration_ms: float | None = None,
) -> Mapping[str, object]:
    """Attribute service and exposed time over one realized stage DAG.

    ``duration_ms`` is the executed boundary.  When supplied it must reconcile
    with the plan's own critical path; otherwise the attribution is unavailable
    rather than rescaled.  With ``duration_ms=None`` the plan's own modeled span
    is used and the evidence basis says so explicitly.
    """
    if not isinstance(plan, PhysicalWorkGraph):
        raise ModelingError("attribute_stage_dag_requires_physical_graph")
    local = {node.node_id: node_local_clocks(node) for node in plan.nodes}
    clocks, terminal = realized_clock_dag(local, plan.predecessors())
    modeled_span = terminal.all_ms
    counterfactual = dict(fixed_plan_exposure(plan))
    if duration_ms is None:
        span, available, evidence = modeled_span, True, "modeled"
    else:
        try:
            span = float(duration_ms)
        except (TypeError, ValueError):
            raise ModelingError(
                "attribute_stage_dag_duration_invalid") from None
        if not math.isfinite(span) or span < 0.0:
            raise ModelingError("attribute_stage_dag_duration_invalid")
        available = _boundary_reconciles(modeled_span, span)
        evidence = "executed"
        if not available:
            counterfactual.update(
                available=False,
                unavailable_reason="unreconciled_execution_boundary")
    memory_service = math.fsum(node.service_ms("hbm") for node in plan.nodes)
    result: dict[str, object] = {
        "available": bool(available),
        "boundary": str(boundary),
        "evidence": evidence,
        "clocks_by_node": clocks,
        "terminal_clocks": terminal,
        "modeled_span_ms": modeled_span,
        "reported_span_ms": span,
        "memory_service_ms": memory_service,
        "exposed_hbm_estimate_ms": None,
        "fixed_plan_counterfactual": counterfactual,
        "buckets": None,
        "basis": "four_clock_realized_stage_dag_service_propagation",
    }
    if not available:
        result["unavailable_reason"] = "unreconciled_stage_dag_boundary"
        return result
    gap_limit = max(span * CLOCK_GAP_LIMIT_FRACTION, CLOCK_GAP_ABS_FLOOR_MS)
    if abs(span - terminal.all_ms) > gap_limit:
        result.update(
            available=False,
            unavailable_reason="unreconciled_service_clock_boundary",
            basis="unavailable_service_clock_boundary")
        return result
    control_ms = math.fsum(
        max(0.0,
            node.ideal_ms - max(node.service_ms(resource)
                                for resource in RESOURCES))
        for node in plan.nodes)
    breakdown = finish_breakdown(
        terminal, duration_ms=span, boundary=str(boundary),
        control_ms=control_ms)
    values = breakdown["exposed_breakdown_ms"]
    service = {
        resource: math.fsum(node.service_ms(resource) for node in plan.nodes)
        for resource in RESOURCES
    }
    result["buckets"] = {
        "cube": {"service_ms": service["cube"], "exposed_ms": values["cube"]},
        "vector": {"service_ms": service["vector"],
                   "exposed_ms": values["vector"]},
        "hbm": {"memory_service_ms": service["hbm"],
                "exposed_ms": values["memory"]},
        "link": {"service_ms": service["link"],
                 "exposed_ms": values["communication"]},
        "control": {
            "launch_or_idle_ms": max(0.0, span - modeled_span),
            "fixed_or_launch_service_ms": control_ms,
        },
    }
    delta = exposed_hbm_delta_ms(
        span if evidence == "executed" else counterfactual["baseline_ms"],
        counterfactual["zero_hbm_service_ms"])
    if evidence == "executed" and delta is not None:
        counterfactual.update(
            baseline_ms=span,
            delta_h_ms=delta,
            available=True,
            basis="executed_baseline_same_plan_zero_hbm_service")
    result["exposed_hbm_estimate_ms"] = delta
    result["fixed_plan_counterfactual"] = counterfactual
    if evidence == "executed" and delta is None:
        result["exposed_hbm_unavailable_reason"] = (
            "zero_hbm_counterfactual_not_available")
    return result


def attribute_resource_exposure(report: Mapping[str, object],
                                plan: PhysicalWorkGraph) -> Mapping[str, object]:
    """Attach four physical buckets and the fixed-plan HBM counterfactual."""
    if not isinstance(plan, PhysicalWorkGraph):
        raise ModelingError("resource_attribution_requires_physical_graph")
    span_raw = report.get("graph_duration_ms",
                          report.get("duration_ms", 0.0))
    try:
        span = float(span_raw or 0.0)
    except (TypeError, ValueError):
        raise ModelingError(
            "resource_attribution_duration_invalid") from None
    if not math.isfinite(span) or span < 0.0:
        raise ModelingError("resource_attribution_duration_invalid")
    if span <= 0.0:
        return {
            "available": False,
            "buckets": None,
            "memory_service_ms": math.fsum(
                node.service_ms("hbm") for node in plan.nodes),
            "exposed_hbm_estimate_ms": None,
            "fixed_plan_counterfactual": {
                **fixed_plan_exposure(plan),
                "available": False,
                "unavailable_reason": "execution_boundary_missing",
            },
            "basis": "unavailable_execution_boundary_missing",
        }
    attribution = attribute_stage_dag(
        plan, boundary="device_step", duration_ms=span)
    counterfactual = dict(attribution["fixed_plan_counterfactual"])
    if not attribution["available"]:
        return {
            "available": False,
            "buckets": None,
            "memory_service_ms": attribution["memory_service_ms"],
            "exposed_hbm_estimate_ms": None,
            "fixed_plan_counterfactual": counterfactual,
            "basis": "unavailable_unreconciled_execution_boundary",
        }
    logical = float(report.get("logical_hbm_bytes",
                               report.get("hbm_bytes", 0.0))
                    or 0.0)
    physical = float(report.get("hbm_bytes", 0.0) or 0.0)
    l2_hit = float(report.get("l2_hit_bytes", 0.0) or 0.0)
    buckets = dict(attribution["buckets"])
    buckets["hbm"] = {
        "logical_bytes": logical,
        "physical_hbm_bytes": physical,
        "l2_hit_bytes": l2_hit,
        "memory_service_ms": attribution["memory_service_ms"],
        "exposed_hbm_estimate_ms": attribution["exposed_hbm_estimate_ms"],
        "exposure_basis": "fixed_plan_zero_hbm_counterfactual",
        "exposed_ms": buckets["hbm"]["exposed_ms"],
    }
    return {
        "available": True,
        "buckets": buckets,
        "memory_service_ms": attribution["memory_service_ms"],
        "exposed_hbm_estimate_ms": attribution["exposed_hbm_estimate_ms"],
        "fixed_plan_counterfactual": counterfactual,
        "basis": ("four_clock_realized_stage_dag_and_"
                  "fixed_plan_zero_hbm_counterfactual"),
    }
