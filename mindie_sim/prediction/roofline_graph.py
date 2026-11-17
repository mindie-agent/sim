"""Critical-path roofline and topology-resource utilization.

[Role] Fold finite physical work nodes over the realized occurrence dependency
  DAG, combine the longest path with per-physical-resource conservation, and
  expose per-link utilization against the actual topology capacity.  A node
  carries its occurrence multiplicity and, optionally, the physical identity of
  each service class so independent devices are never summed into one logical
  category.
[Invariants] Every node's target bound is at least its 1.0x ideal bound and its
  occurrence multiplicity scales service, launch and fixed work.  The step
  bound is ``max(longest path, max_r total_r / rate_r)`` where ``r`` is a
  physical resource identity.  Edges come from declared data, stream, state and
  collective-rendezvous dependencies; this module never invents an edge.
  Link utilization is reported raw and is never clipped to hide accounting
  errors.
[Timeouts & Locks] Bounded DAG/sort work; no I/O or locks.
[Side-Effect Policy] Pure immutable arithmetic.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable, Mapping

from mindie_sim.schema.common import ModelingError

ZERO = 0
ONE = 1
ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
MS_PER_S = 1000.0
UTILIZATION_TOLERANCE = 1e-9
RESOURCES: tuple[str, ...] = ("cube", "vector", "hbm", "link")
RESOURCE_FIELDS: Mapping[str, str] = {
    "cube": "cube_ms",
    "vector": "vector_ms",
    "hbm": "hbm_ms",
    "link": "link_ms",
}
OCCURRENCE_ID_KEYS: tuple[str, ...] = ("occurrence_id", "node_id", "id")
ROW_ID_KEYS: tuple[str, ...] = ("physical_id", "op_id", "layer")
PREDECESSOR_KEYS: tuple[str, ...] = ("predecessors", "predecessor_ids", "deps")
SERVICE_ROW_KEYS: Mapping[str, tuple[str, ...]] = {
    "cube": ("cube_ms", "t_cube_ideal_ms"),
    "vector": ("vector_ms", "t_vec_ideal_ms"),
    "hbm": ("hbm_ms", "t_hbm_ideal_ms"),
    "link": ("link_ms", "t_link_ideal_ms"),
}
OCCURRENCE_KEYS: tuple[str, ...] = ("occurrences", "occurrence_count", "count")
FIXED_KEYS: tuple[str, ...] = ("fixed_ms", "fixed_launch_ms")
LAUNCH_KEYS: tuple[str, ...] = ("launch_ms",)
RESOURCE_ID_SUFFIXES: tuple[str, ...] = ("_resource", "_physical_id", "_device")


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
    """One actual execution occurrence (or an explicit serial multiplicity).

    ``resource_ids`` maps a logical service class (cube/vector/hbm/link) to the
    physical device identity that serves it.  Absent identities fall back to the
    logical class name, which is the degenerate single-device case.  The default
    multiplicity is one; grouped rows that already fold ``N`` identical
    occurrences may declare ``occurrences=N`` and are then serialized at this
    DAG position by construction, never deduplicated with other rows.
    """

    node_id: str = ""
    cube_ms: float = ZERO_FLOAT
    vector_ms: float = ZERO_FLOAT
    hbm_ms: float = ZERO_FLOAT
    link_ms: float = ZERO_FLOAT
    launch_ms: float = ZERO_FLOAT
    fixed_ms: float = ZERO_FLOAT
    predecessors: tuple[str, ...] = ()
    occurrences: int = ONE
    resource_ids: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        raw_ids = self.resource_ids
        if isinstance(raw_ids, Mapping):
            normalized = tuple(sorted(
                (str(name), str(identity))
                for name, identity in raw_ids.items()))
        elif raw_ids:
            normalized = tuple(
                (str(name), str(identity)) for name, identity in raw_ids)
        else:
            normalized = ()
        object.__setattr__(self, "resource_ids", normalized)
        if any(not math.isfinite(value) or value < ZERO_FLOAT
               for value in (self.cube_ms, self.vector_ms, self.hbm_ms,
                             self.link_ms, self.launch_ms, self.fixed_ms)):
            raise ModelingError("physical_work_node_duration_must_be_nonnegative")
        if (isinstance(self.occurrences, bool)
                or not isinstance(self.occurrences, int)
                or self.occurrences < ONE):
            raise ModelingError(
                "physical_work_node_occurrences_must_be_positive_integer")
        seen: dict[str, str] = {}
        for resource, identity in self.resource_ids:
            if resource not in RESOURCE_FIELDS:
                raise ModelingError(
                    f"physical_work_node_resource_unknown:{resource}")
            if not identity:
                raise ModelingError(
                    "physical_work_node_resource_identity_must_be_nonempty")
            if resource in seen:
                raise ModelingError(
                    f"physical_work_node_resource_duplicate:{resource}")
            seen[resource] = identity

    def service_ms(self, resource: str) -> float:
        """Total service of one logical class across this node's occurrences."""
        if resource not in RESOURCE_FIELDS:
            raise ModelingError(f"physical_work_node_resource_unknown:{resource}")
        return (float(self.occurrences)
                * float(getattr(self, RESOURCE_FIELDS[resource])))

    def physical_resource_id(self, resource: str) -> str:
        """Physical identity serving ``resource`` for this occurrence."""
        if resource not in RESOURCE_FIELDS:
            raise ModelingError(f"physical_work_node_resource_unknown:{resource}")
        for name, identity in self.resource_ids:
            if name == resource:
                return identity
        return resource

    @property
    def ideal_ms(self) -> float:
        service = max(self.cube_ms, self.vector_ms, self.hbm_ms, self.link_ms)
        return float(self.occurrences) * max(
            self.launch_ms, self.fixed_ms + service)

    def target_ms(self, targets: RooflineTargets) -> float:
        if not isinstance(targets, RooflineTargets):
            raise ModelingError("physical_work_node_targets_required")
        service = max(
            self.cube_ms / targets.cube,
            self.vector_ms / targets.vector,
            self.hbm_ms / targets.hbm,
            self.link_ms / targets.link,
        )
        raw = max(self.launch_ms, self.fixed_ms + service)
        per_occurrence = max(
            max(self.launch_ms, self.fixed_ms + max(
                self.cube_ms, self.vector_ms, self.hbm_ms, self.link_ms)),
            raw)
        return float(self.occurrences) * per_occurrence


@dataclass(frozen=True)
class PhysicalWorkGraph:
    nodes: tuple[PhysicalWorkNode, ...] = ()
    edges: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ModelingError("physical_work_graph_duplicate_node")
        known = set(ids)
        for node in self.nodes:
            for before in node.predecessors:
                if before not in known:
                    raise ModelingError(
                        "physical_work_graph_edge_unknown_node")
        for before, after in self.edges:
            if before not in known or after not in known:
                raise ModelingError("physical_work_graph_edge_unknown_node")

    def predecessors(self) -> dict[str, tuple[str, ...]]:
        result = {node.node_id: tuple(dict.fromkeys(node.predecessors))
                  for node in self.nodes}
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


def _topological_order(nodes: Mapping[str, PhysicalWorkNode],
                       predecessors: Mapping[str, tuple[str, ...]]) -> tuple[str, ...]:
    successors = {node_id: [] for node_id in nodes}
    indegree = {}
    for node_id in nodes:
        indegree[node_id] = len(predecessors.get(node_id, ()))
        for predecessor in predecessors.get(node_id, ()):
            if predecessor not in nodes:
                raise ModelingError("physical_work_graph_edge_unknown_node")
            successors[predecessor].append(node_id)
    ready = deque(node_id for node_id in nodes if indegree[node_id] == ZERO)
    order: list[str] = []
    while ready:
        node_id = ready.popleft()
        order.append(node_id)
        for successor in successors[node_id]:
            indegree[successor] -= ONE
            if indegree[successor] == ZERO:
                ready.append(successor)
    if len(order) != len(nodes):
        raise ModelingError("physical_work_graph_cycle")
    return tuple(order)


def physical_graph_order(graph: PhysicalWorkGraph) -> tuple[str, ...]:
    """Topological order of a physical work graph; a cycle is an error."""
    if not isinstance(graph, PhysicalWorkGraph):
        raise ModelingError("physical_graph_order_requires_physical_graph")
    nodes = {node.node_id: node for node in graph.nodes}
    return _topological_order(nodes, graph.predecessors())


def _longest_path(nodes: Mapping[str, PhysicalWorkNode],
                  predecessors: Mapping[str, tuple[str, ...]],
                  weights: Mapping[str, float]) -> tuple[float, tuple[str, ...]]:
    order = _topological_order(nodes, predecessors)
    best: dict[str, float] = {}
    path: dict[str, tuple[str, ...]] = {}
    for node_id in order:
        finish, chosen = ZERO_FLOAT, ()
        for predecessor in predecessors.get(node_id, ()):
            if best[predecessor] > finish:
                finish, chosen = best[predecessor], path[predecessor]
        best[node_id] = weights[node_id] + finish
        path[node_id] = (*chosen, node_id)
    maximum, critical = ZERO_FLOAT, ()
    for node_id in order:
        if best[node_id] > maximum:
            maximum, critical = best[node_id], path[node_id]
    return maximum, critical


def _resource_totals(
        nodes: Iterable[PhysicalWorkNode]) -> tuple[dict[str, str], dict[str, float]]:
    """Per-physical-resource service totals, keyed by physical identity."""
    nodes = tuple(nodes)
    explicit = any(node.resource_ids for node in nodes)
    logical_by_key: dict[str, str] = {}
    values_by_key: dict[str, list[float]] = {}
    for node in nodes:
        for resource in RESOURCES:
            service = node.service_ms(resource)
            if service <= ZERO_FLOAT:
                continue
            key = node.physical_resource_id(resource)
            previous = logical_by_key.get(key)
            if previous is not None and previous != resource:
                raise ModelingError(
                    f"physical_resource_identity_conflict:{key}")
            logical_by_key[key] = resource
            values_by_key.setdefault(key, []).append(service)
    if not explicit:
        for resource in RESOURCES:
            logical_by_key.setdefault(resource, resource)
            values_by_key.setdefault(resource, [])
    totals = {key: math.fsum(values)
              for key, values in values_by_key.items()}
    return logical_by_key, totals


def critical_path_roofline(graph: PhysicalWorkGraph,
                           targets: RooflineTargets) -> RooflineResult:
    """Compute dependency and per-physical-resource roofline bounds.

    The graph must already contain the realized occurrence nodes and their
    data/stream/state/collective-rendezvous predecessors.  This function does
    not create missing edges, and its bound is only as sound as the supplied
    graph and pipe rates; the construction is the caller's responsibility.
    """
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
    logical_by_key, resource_totals = _resource_totals(graph.nodes)
    resource_cut_bounds = {
        key: (total / targets.for_resource(logical_by_key[key])
              if total > ZERO_FLOAT else ZERO_FLOAT)
        for key, total in resource_totals.items()
    }
    physical_ideal = max(cp_ideal, max(resource_totals.values(), default=ZERO_FLOAT))
    target_bound = max(cp_target, max(resource_cut_bounds.values(),
                                      default=ZERO_FLOAT))
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


def _row_value(row: Mapping[str, object], keys: Iterable[str], default=None):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _nonnegative_float(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ModelingError(
            f"graph_from_occurrences_service_invalid:{label}") from None
    if not math.isfinite(number) or number < ZERO_FLOAT:
        raise ModelingError(f"graph_from_occurrences_service_invalid:{label}")
    return number


def _row_ids(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def _row_resource_ids(row: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    raw = row.get("resource_ids")
    pairs: list[tuple[str, str]] = []
    if isinstance(raw, Mapping):
        pairs.extend((str(name), str(identity))
                     for name, identity in raw.items())
    elif raw:
        pairs.extend((str(name), str(identity))
                     for name, identity in raw)
    for resource in RESOURCES:
        for suffix in RESOURCE_ID_SUFFIXES:
            identity = row.get(f"{resource}{suffix}")
            if identity:
                pairs.append((resource, str(identity)))
                break
    return tuple(pairs)


def _occurrence_node_id(row: Mapping[str, object], index: int,
                        used: set[str]) -> str:
    explicit = _row_value(row, OCCURRENCE_ID_KEYS, None)
    if explicit is None:
        base = str(_row_value(row, ROW_ID_KEYS, f"occurrence{index}"))
        candidate = f"{base}#{index}"
        while candidate in used:
            candidate = f"{candidate}#"
        used.add(candidate)
        return candidate
    base = str(explicit)
    candidate = base
    suffix = ONE + ONE
    while candidate in used:
        candidate = f"{base}#{suffix}"
        suffix += ONE
    used.add(candidate)
    return candidate


def graph_from_occurrences(
    rows: Iterable[Mapping[str, object]],
    edges: Iterable[tuple[str, str]] = (),
    *,
    service_by_id: Mapping[str, Mapping[str, object]] | None = None,
) -> PhysicalWorkGraph:
    """Build a physical work DAG from ACTUAL execution occurrence rows.

    Each row is one execution occurrence.  Service work is read from
    ``cube_ms``/``t_cube_ideal_ms`` (likewise vector, hbm, link),
    ``fixed_ms``/``fixed_launch_ms`` and ``launch_ms``; ``occurrences`` is an
    explicit serial multiplicity and is never used to merge rows.  Duplicate
    explicit identities receive an occurrence suffix instead of disappearing.

    Dependency identity is read from ``predecessors``/``predecessor_ids``/
    ``deps``.  When the rows do not carry it, the caller MUST pass ``edges``;
    this function never invents data, stream, state or rendezvous edges.  Rows
    that carry both are unioned.  A row predecessor that does not identify
    another row is an external boundary input and is cut, not fabricated.
    Rows without an occurrence identity receive an index-qualified identity
    (``<physical_id>#<row index>``); pass explicit occurrence ids when an edge
    must name one occurrence.  ``service_by_id`` optionally supplies the
    service fields for an occurrence identity, which lets a caller join a
    realized timeline to its priced occurrence rows without collapsing either.
    ``service_by_id`` keys are occurrence identities; supply unique explicit
    ids when more than one occurrence shares a physical identity.
    """
    materialized = tuple(rows)
    if not materialized:
        return PhysicalWorkGraph((), tuple(edges or ()))
    edge_list = tuple((str(before), str(after)) for before, after in (edges or ()))
    has_dependency_field = tuple(
        _row_value(row, PREDECESSOR_KEYS, None) is not None
        for row in materialized)
    if len(materialized) > ONE and not edge_list and not all(has_dependency_field):
        raise ModelingError("graph_from_occurrences_requires_explicit_edges")
    prepared = []
    used: set[str] = set()
    for index, row in enumerate(materialized):
        node_id = _occurrence_node_id(row, index, used)
        prepared.append((node_id, row, _row_ids(
            _row_value(row, PREDECESSOR_KEYS, None))))
    known = {node_id for node_id, _row, _predecessors in prepared}
    nodes = []
    for node_id, row, raw_predecessors in prepared:
        service = {
            resource: _row_value(row, SERVICE_ROW_KEYS[resource], ZERO_FLOAT)
            for resource in RESOURCES
        }
        if service_by_id is not None and node_id in service_by_id:
            supplied = service_by_id[node_id]
            for resource in RESOURCES:
                value = _row_value(supplied, SERVICE_ROW_KEYS[resource], None)
                if value is not None:
                    service[resource] = value
        count = _row_value(row, OCCURRENCE_KEYS, ONE)
        if (isinstance(count, bool) or not isinstance(count, int)
                or count < ONE):
            raise ModelingError(
                f"graph_from_occurrences_occurrences_invalid:{node_id}")
        nodes.append(PhysicalWorkNode(
            node_id=node_id,
            cube_ms=_nonnegative_float(service["cube"], f"{node_id}:cube"),
            vector_ms=_nonnegative_float(service["vector"], f"{node_id}:vector"),
            hbm_ms=_nonnegative_float(service["hbm"], f"{node_id}:hbm"),
            link_ms=_nonnegative_float(service["link"], f"{node_id}:link"),
            fixed_ms=_nonnegative_float(
                _row_value(row, FIXED_KEYS, ZERO_FLOAT), f"{node_id}:fixed"),
            launch_ms=_nonnegative_float(
                _row_value(row, LAUNCH_KEYS, ZERO_FLOAT), f"{node_id}:launch"),
            predecessors=tuple(
                predecessor for predecessor in raw_predecessors
                if predecessor in known),
            occurrences=count,
            resource_ids=_row_resource_ids(row),
        ))
    return PhysicalWorkGraph(tuple(nodes), edge_list)


def graph_from_rows(rows) -> PhysicalWorkGraph:
    """Degenerate compatibility caller: one edgeless node per grouped row.

    This retains the historical row fields only.  It ignores ``occurrences``,
    predecessor identity and physical resource identity, so it cannot recover
    serial chains, repeated layers or independent devices.  New integration
    must call :func:`graph_from_occurrences` with actual occurrence rows and
    explicit edges instead.
    """
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
