"""Critical-path roofline and topology-resource utilization.

[Role] Fold finite physical operator nodes over the dependency DAG, combine the
  longest path with whole-device resource conservation, and expose per-link
  utilization against the actual topology capacity.
[Invariants] Every node's target bound is at least its 1.0x ideal bound. The
  step bound is at least the dependency path and every shared-resource cut.
  Link utilization is reported raw and is never clipped to hide accounting
  errors.
[Timeouts & Locks] Bounded DAG/sort work; no I/O or locks.
[Side-Effect Policy] Pure immutable arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

from mindie_sim.schema.common import ModelingError

ZERO = 0
ONE = 1
ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
MS_PER_S = 1000.0
UTILIZATION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class RooflineTargets:
    cube: float = ONE_FLOAT
    vector: float = ONE_FLOAT
    hbm: float = ONE_FLOAT
    link: float = ONE_FLOAT

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= ZERO_FLOAT
               or value > ONE_FLOAT
               for value in (self.cube, self.vector, self.hbm, self.link)):
            raise ModelingError("roofline_target_efficiency_must_be_in_unit_interval")

    def for_resource(self, resource: str) -> float:
        return {
            "cube": self.cube,
            "vector": self.vector,
            "hbm": self.hbm,
            "link": self.link,
        }.get(resource, ONE_FLOAT)


@dataclass(frozen=True)
class PhysicalWorkNode:
    node_id: str = ""
    cube_ms: float = ZERO_FLOAT
    vector_ms: float = ZERO_FLOAT
    hbm_ms: float = ZERO_FLOAT
    link_ms: float = ZERO_FLOAT
    launch_ms: float = ZERO_FLOAT
    fixed_ms: float = ZERO_FLOAT
    predecessors: tuple[str, ...] = ()

    def __post_init__(self):
        if any(not math.isfinite(value) or value < ZERO_FLOAT
               for value in (self.cube_ms, self.vector_ms, self.hbm_ms,
                             self.link_ms, self.launch_ms, self.fixed_ms)):
            raise ModelingError("physical_work_node_duration_must_be_nonnegative")

    @property
    def ideal_ms(self) -> float:
        return max(self.launch_ms,
                   self.fixed_ms + max(self.cube_ms, self.vector_ms,
                                       self.hbm_ms, self.link_ms))

    def target_ms(self, targets: RooflineTargets) -> float:
        raw = max(
            self.launch_ms,
            self.fixed_ms + max(self.cube_ms / targets.cube,
                                self.vector_ms / targets.vector,
                                self.hbm_ms / targets.hbm,
                                self.link_ms / targets.link),
        )
        return max(self.ideal_ms, raw)


@dataclass(frozen=True)
class PhysicalWorkGraph:
    nodes: tuple[PhysicalWorkNode, ...] = ()
    edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ModelingError("physical_work_graph_duplicate_node")
        known = set(ids)
        for before, after in self.edges:
            if before not in known or after not in known:
                raise ModelingError("physical_work_graph_edge_unknown_node")

    def predecessors(self) -> dict[str, tuple[str, ...]]:
        result = {node.node_id: tuple(node.predecessors) for node in self.nodes}
        for before, after in self.edges:
            result[after] = tuple(dict.fromkeys((*result[after], before)))
        return result


@dataclass(frozen=True)
class RooflineResult:
    physical_ideal_ms: float
    target_duration_ms: float
    critical_path_ideal_ms: float
    critical_path_target_ms: float
    resource_totals: Mapping[str, float]
    resource_cut_bounds: Mapping[str, float]
    node_ideal_ms: Mapping[str, float]
    node_target_ms: Mapping[str, float]
    critical_path_nodes: tuple[str, ...]
    target_step_basis: str = "dag_critical_path_and_resource_conservation"

    @property
    def target_step_duration_ms(self) -> float:
        return self.target_duration_ms


@dataclass(frozen=True)
class LinkUtilization:
    utilization_by_resource: Mapping[str, float]
    capacity_by_resource: Mapping[str, float]
    wire_by_resource: Mapping[str, float]
    span_ms: float
    maximum: float
    violations: tuple[str, ...]
    nominal_peak_utilization: float = ZERO_FLOAT
    configured_capacity_utilization: float = ZERO_FLOAT
    useful_payload_throughput: float = ZERO_FLOAT
    bus_wire_throughput: float = ZERO_FLOAT

    def __post_init__(self):
        if self.span_ms < ZERO_FLOAT:
            raise ModelingError("link_utilization_span_must_be_nonnegative")


def operator_roofline_ms(node: PhysicalWorkNode,
                         targets: RooflineTargets) -> float:
    """Return one operator's target-efficiency critical-path bound."""
    if not isinstance(node, PhysicalWorkNode):
        raise ModelingError("operator_roofline_requires_physical_node")
    return node.target_ms(targets)


def _longest_path(nodes: Mapping[str, PhysicalWorkNode],
                  predecessors: Mapping[str, tuple[str, ...]],
                  weights: Mapping[str, float]) -> tuple[float, tuple[str, ...]]:
    best: dict[str, float] = {}
    path: dict[str, tuple[str, ...]] = {}
    visiting: set[str] = set()

    def visit(node_id: str) -> tuple[float, tuple[str, ...]]:
        if node_id in best:
            return best[node_id], path[node_id]
        if node_id in visiting:
            raise ModelingError("physical_work_graph_cycle")
        visiting.add(node_id)
        finish, chosen = ZERO_FLOAT, ()
        for predecessor in predecessors.get(node_id, ()):
            candidate, candidate_path = visit(predecessor)
            if candidate > finish:
                finish, chosen = candidate, candidate_path
        visiting.remove(node_id)
        total = weights[node_id] + finish
        best[node_id] = total
        path[node_id] = (*chosen, node_id)
        return total, path[node_id]

    maximum, critical = ZERO_FLOAT, ()
    for node_id in nodes:
        candidate, candidate_path = visit(node_id)
        if candidate > maximum:
            maximum, critical = candidate, candidate_path
    return maximum, critical


def critical_path_roofline(graph: PhysicalWorkGraph,
                           targets: RooflineTargets) -> RooflineResult:
    """Compute dependency and whole-device resource roofline bounds."""
    if not isinstance(graph, PhysicalWorkGraph):
        raise ModelingError("critical_path_roofline_requires_physical_graph")
    if not isinstance(targets, RooflineTargets):
        raise ModelingError("critical_path_roofline_requires_targets")
    nodes = {node.node_id: node for node in graph.nodes}
    predecessors = graph.predecessors()
    ideal_weights = {node_id: node.ideal_ms for node_id, node in nodes.items()}
    target_weights = {node_id: node.target_ms(targets)
                      for node_id, node in nodes.items()}
    cp_ideal, critical = _longest_path(nodes, predecessors, ideal_weights)
    cp_target, target_critical = _longest_path(nodes, predecessors, target_weights)
    resources = ("cube", "vector", "hbm", "link")
    resource_totals = {
        resource: math.fsum(getattr(node, f"{resource}_ms")
                            for node in graph.nodes)
        for resource in resources
    }
    resource_cut_bounds = {
        resource: (total / targets.for_resource(resource)
                   if total > ZERO_FLOAT else ZERO_FLOAT)
        for resource, total in resource_totals.items()
    }
    physical_ideal = max(cp_ideal, *resource_totals.values(), ZERO_FLOAT)
    target_bound = max(cp_target, *resource_cut_bounds.values(), ZERO_FLOAT)
    target_bound = max(physical_ideal, target_bound)
    return RooflineResult(
        physical_ideal_ms=physical_ideal,
        target_duration_ms=target_bound,
        critical_path_ideal_ms=cp_ideal,
        critical_path_target_ms=cp_target,
        resource_totals=resource_totals,
        resource_cut_bounds=resource_cut_bounds,
        node_ideal_ms=ideal_weights,
        node_target_ms=target_weights,
        critical_path_nodes=(critical if cp_ideal >= cp_target else target_critical),
    )


def link_utilization(wire_by_resource: Mapping[str, float],
                     capacity_by_resource: Mapping[str, float],
                     span_ms: float) -> LinkUtilization:
    """Return raw per-resource utilization and any capacity violation."""
    span = float(span_ms)
    if not math.isfinite(span) or span < ZERO_FLOAT:
        raise ModelingError("link_utilization_span_invalid")
    utilization = {}
    violations = []
    for resource, wire in wire_by_resource.items():
        capacity = float(capacity_by_resource.get(resource, ZERO_FLOAT))
        if capacity <= ZERO_FLOAT:
            raise ModelingError(f"link_utilization_capacity_missing:{resource}")
        value = float(wire) / (capacity * span) if span > ZERO_FLOAT else ZERO_FLOAT
        utilization[resource] = value
        if value > ONE_FLOAT + UTILIZATION_TOLERANCE:
            violations.append(resource)
    maximum = max(utilization.values(), default=ZERO_FLOAT)
    total_wire = math.fsum(float(value) for value in wire_by_resource.values())
    total_capacity = math.fsum(float(value) for value in capacity_by_resource.values())
    return LinkUtilization(
        utilization_by_resource=utilization,
        capacity_by_resource={key: float(value)
                              for key, value in capacity_by_resource.items()},
        wire_by_resource={key: float(value)
                          for key, value in wire_by_resource.items()},
        span_ms=span,
        maximum=maximum,
        violations=tuple(violations),
        nominal_peak_utilization=maximum,
        configured_capacity_utilization=maximum,
        useful_payload_throughput=(total_wire / span if span > ZERO_FLOAT else ZERO_FLOAT),
        bus_wire_throughput=(total_capacity / span if span > ZERO_FLOAT else ZERO_FLOAT),
    )


def graph_from_rows(rows) -> PhysicalWorkGraph:
    """Build a physical invocation DAG from analysis rows."""
    nodes = tuple(
        PhysicalWorkNode(
            node_id=str(row.get("physical_id") or row.get("op_id") or index),
            cube_ms=float(row.get("t_cube_ideal_ms", ZERO_FLOAT) or ZERO_FLOAT),
            vector_ms=float(row.get("t_vec_ideal_ms", ZERO_FLOAT) or ZERO_FLOAT),
            hbm_ms=float(row.get("t_hbm_ideal_ms", ZERO_FLOAT) or ZERO_FLOAT),
            link_ms=float(row.get("t_link_ideal_ms", ZERO_FLOAT) or ZERO_FLOAT),
            launch_ms=float(row.get("fixed_ms", ZERO_FLOAT) or ZERO_FLOAT),
        )
        for index, row in enumerate(rows)
    )
    return PhysicalWorkGraph(nodes)
