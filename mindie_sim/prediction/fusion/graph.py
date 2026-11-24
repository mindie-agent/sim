"""Typed operator projection shared by the Day-0 graph optimizer.

[Role] Carry immutable tensor/operator records as a *projection* of the
  canonical selected ``OpSpec`` DAG (see ``fusion_bridge``), provide forward
  navigation, boundary predicates and fail-closed state validation.
[Invariants] One producer per value; ordered dependencies; storage bytes are
  physical and names never select policy. ``successors`` always follows the
  forward data/event direction. State readers/writers require declared effects
  and an ordered version relation. Legality remains in feasibility.
[Timeouts & Locks] Finite record validation only; no I/O, clocks or locks.
[Side-Effect Policy] Pure construction and read-only graph queries.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4, "tf32": 4, "int64": 8,
               "int32": 4, "int16": 2, "int8": 1, "uint8": 1, "fp8": 1,
               "fp8_e4m3": 1, "fp8_e5m2": 1, "int4": 1, "fp4": 1}
TILE_ROWS = (1, 2, 4, 8, 16, 32, 64)
UB_ALIGNMENT = 32
CUBE_KINDS = frozenset({"matmul", "grouped_matmul", "attention", "mla_prolog"})
REDUCTION_KINDS = frozenset({"rms_norm", "add_rms_norm", "rms_norm_quant",
                             "add_rms_norm_quant", "reduce", "softmax",
                             "dynamic_quant", "quant"})
STATE_KINDS = frozenset({"state_update", "recurrent", "cache_write"})
PARAM_ROLES = frozenset({"weight", "state", "parameter", "index"})
STATE_ROLES = frozenset({"state"})
LAYOUTS = frozenset({"contiguous", "nd", "nz", "strided", "view"})


class FeasibilityError(ValueError):
    """A structural graph contract is violated (not a rejected candidate)."""


def _align(value: int, alignment: int = UB_ALIGNMENT) -> int:
    if alignment < 1:
        raise FeasibilityError("alignment_must_be_positive")
    return int(math.ceil(max(0, value) / alignment) * alignment)


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def dtype_bytes(dtype: str) -> int:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise FeasibilityError(f"dtype_storage_unknown:{dtype}") from exc


@dataclass(frozen=True)
class TensorValue:
    """One physical value with storage (not logical element) size.

    ``sharded`` states that the byte count is already the local rank shard, so
    a second tensor-parallel division is a conservation error. ``layout`` and
    ``alias_of`` are carried from the canonical descriptor; they are never
    invented by a recipe.
    """
    value_id: str
    dtype: str
    shape: tuple[int, ...]
    role: str = "activation"
    producer: str | None = None
    storage_bytes: int | None = None
    alignment_bytes: int = UB_ALIGNMENT
    state_version: int | None = None
    public: bool = False
    sharded: bool = False
    layout: str = "contiguous"
    storage_shape: tuple[int, ...] | None = None
    alias_of: str | None = None
    state_key: str = ""

    def __post_init__(self):
        if not self.value_id or not self.dtype or not self.shape:
            raise FeasibilityError("tensor_identity_and_shape_required")
        if any(dim < 0 for dim in self.shape) or self.alignment_bytes < 1:
            raise FeasibilityError(f"tensor_shape_or_alignment_invalid:{self.value_id}")
        if self.role not in {"activation", "residual", "scale", "weight", "state",
                             "index", "output", "parameter"}:
            raise FeasibilityError(f"tensor_role_unknown:{self.value_id}")
        if (self.storage_bytes is not None and self.storage_bytes < 0) or (
                self.state_version is not None and self.state_version < 0):
            raise FeasibilityError(f"tensor_storage_or_version_invalid:{self.value_id}")
        if self.layout not in LAYOUTS:
            raise FeasibilityError(f"tensor_layout_unknown:{self.value_id}:{self.layout}")
        if self.storage_shape is not None and any(d < 0 for d in self.storage_shape):
            raise FeasibilityError(f"tensor_storage_shape_invalid:{self.value_id}")

    @property
    def nbytes(self) -> int:
        return (int(self.storage_bytes) if self.storage_bytes is not None
                else _numel(self.shape) * dtype_bytes(self.dtype))

    @property
    def materialized(self) -> bool:
        return (self.role in {"weight", "state", "parameter", "index", "output"}
                or self.public)

    @property
    def is_state(self) -> bool:
        return self.role in STATE_ROLES or self.state_version is not None


@dataclass(frozen=True)
class OpNode:
    """A typed semantic operator projected from one selected ``OpSpec``.

    ``work`` is the declared physical service, one entry per resource. Recipes
    must preserve every entry except a proven eliminated HBM access.
    """
    op_id: str
    kind: str
    resource: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    deps: tuple[str, ...] = ()
    params: tuple[tuple[str, Any], ...] = ()
    stream: str = "main"
    stage: int = 0
    phase: str = "decode"
    collective: bool = False
    capture_boundary: str = ""
    state_effect: str = ""
    work: tuple[tuple[str, float], ...] = ()
    family: str = ""
    source_boundary: str = ""
    dtype: str = ""
    layout: str = "contiguous"
    numerical_policy: str = "exact"
    priority: int = 0
    owner: int = 0
    #: False when the projection lacks a declared dtype/shape/storage fact; a
    #: missing semantic fact is ineligible, never a silently fabricated tensor.
    semantic_complete: bool = True
    #: A source-kernel/native planner owns its program; the optimizer may not
    #: replace it with a synthetic fused chain.
    opaque: bool = False

    def __post_init__(self):
        if not self.op_id or not self.kind:
            raise FeasibilityError("operator_identity_required")
        if self.resource not in {"vector", "cube", "mix", "host"}:
            raise FeasibilityError(f"operator_resource_unknown:{self.op_id}")
        if self.state_effect and self.state_effect not in {"read", "write", "read_write"}:
            raise FeasibilityError(f"operator_state_effect_invalid:{self.op_id}")
        if self.numerical_policy not in {"exact", "relaxed"}:
            raise FeasibilityError(f"operator_numerical_policy_invalid:{self.op_id}")
        if self.priority < 0 or self.owner < 0:
            raise FeasibilityError(f"operator_priority_or_owner_invalid:{self.op_id}")
        if any(amount < 0 or not math.isfinite(amount) for _, amount in self.work):
            raise FeasibilityError(f"operator_work_invalid:{self.op_id}")

    def param(self, name: str, default: Any = None) -> Any:
        return dict(self.params).get(name, default)

    def work_dict(self) -> dict[str, float]:
        return dict(self.work)


@dataclass(frozen=True)
class Constraint:
    name: str
    status: str
    detail: str

    def record(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def state_ports(graph: "TypedGraph", op: OpNode) -> tuple[str, ...]:
    """State-typed input/output values touched by ``op``."""
    return tuple(v for v in (*op.inputs, *op.outputs)
                 if graph.values[v].is_state or graph.values[v].role in STATE_ROLES)


def state_keys(graph: "TypedGraph", op: OpNode) -> tuple[str, ...]:
    keys = {graph.values[v].state_key for v in state_ports(graph, op)
            if graph.values[v].state_key}
    key = op.param("state_key")
    if isinstance(key, str) and key:
        keys.add(key)
    return tuple(sorted(keys))


def _state_read_versions(graph: "TypedGraph", op: OpNode) -> dict[str, int | None]:
    """Version observed by each state target's read port, compared per key.

    A node may touch several unrelated state targets in one instruction; there
    is no meaningful global min over those versions.  Each target keeps its own
    observed version so ``{a: 7, b: 1}`` never compares ``7`` with ``1``.
    """
    versions: dict[str, int | None] = {}
    declared_key = op.param("state_key")
    fallback = declared_key if isinstance(declared_key, str) else ""
    for value_id in op.inputs:
        value = graph.values[value_id]
        if not value.is_state:
            continue
        key = value.state_key or fallback or value_id
        previous = versions.get(key)
        if previous is None or value.state_version is None:
            versions[key] = value.state_version
        else:
            versions[key] = min(previous, value.state_version)
    return versions


def _state_write_versions(graph: "TypedGraph", op: OpNode) -> dict[str, int | None]:
    """Version produced by each state target's write port, compared per key."""
    versions: dict[str, int | None] = {}
    declared_key = op.param("state_key")
    fallback = declared_key if isinstance(declared_key, str) else ""
    for value_id in op.outputs:
        value = graph.values[value_id]
        if not value.is_state:
            continue
        key = value.state_key or fallback or value_id
        previous = versions.get(key)
        if previous is None or value.state_version is None:
            versions[key] = value.state_version
        else:
            versions[key] = max(previous, value.state_version)
    return versions


def _forward_edges(graph: "TypedGraph") -> dict[str, set[str]]:
    edges: dict[str, set[str]] = {op_id: set() for op_id in graph.ops}
    consumers: dict[str, list[str]] = {}
    for op in graph.ops.values():
        for dep in op.deps:
            if dep in edges and dep != op.op_id:
                edges[dep].add(op.op_id)
        for value_id in op.inputs:
            consumers.setdefault(value_id, []).append(op.op_id)
    for op in graph.ops.values():
        for value_id in op.outputs:
            for consumer in consumers.get(value_id, ()):
                if consumer != op.op_id:
                    edges[op.op_id].add(consumer)
    return edges


def _reachable(edges: Mapping[str, set[str]], starts: Iterable[str]) -> set[str]:
    seen: set[str] = set()
    stack = list(starts)
    while stack:
        for nxt in edges.get(stack.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


@dataclass(frozen=True)
class TypedGraph:
    """Topologically ordered operator projection with one producer per value."""
    rows: int
    values: Mapping[str, TensorValue]
    ops: Mapping[str, OpNode]
    public_outputs: tuple[str, ...] = ()
    provenance: str = "typed_projection"
    external_inputs: tuple[str, ...] = ()

    def __post_init__(self):
        if self.rows < 1 or not self.ops or not isinstance(self.values, Mapping):
            raise FeasibilityError("graph_requires_rows_values_and_operators")

    def validate(self) -> tuple[Constraint, ...]:
        seen: set[str] = set()
        for op_id, op in self.ops.items():
            if op_id != op.op_id:
                raise FeasibilityError(f"graph_operator_key_mismatch:{op_id}")
            for dep in op.deps:
                if dep not in seen and dep not in self.external_inputs:
                    raise FeasibilityError(f"graph_dependency_not_before_operator:{op_id}:{dep}")
            for value_id in op.inputs:
                value = self.values.get(value_id)
                if value is None or (value.producer is not None and value.producer not in seen):
                    raise FeasibilityError(f"graph_input_invalid:{op_id}:{value_id}")
            for value_id in op.outputs:
                value = self.values.get(value_id)
                if value is None or value.producer != op_id:
                    raise FeasibilityError(f"graph_value_producer_mismatch:{value_id}")
            seen.add(op_id)
        for value_id in self.public_outputs:
            if value_id not in self.values:
                raise FeasibilityError(f"graph_public_output_unknown:{value_id}")
        for value in self.values.values():
            if value.producer is not None and value.producer not in self.ops:
                raise FeasibilityError(f"graph_value_producer_unknown:{value.value_id}")
        self._validate_state()
        return (Constraint("graph_topology", "satisfied",
                           "unique producers and ordered dependencies"),
                Constraint("state_ordering", "satisfied",
                           "declared effects and ordered state versions"))

    def _validate_state(self) -> None:
        """Fail closed on missing effects and unordered same-state access.

        Two accesses to one state are ordered in a DAG iff either ``first``
        reaches ``second`` or ``second`` reaches ``first``.  Demanding both
        directions would require a cycle and rejects every valid chain.
        """
        edges = _forward_edges(self)
        for op in self.ops.values():
            ports = state_ports(self, op)
            stateful = bool(ports) or op.kind in STATE_KINDS
            unversioned = bool(op.param("unversioned_state"))
            if stateful and not op.state_effect and not unversioned:
                raise FeasibilityError(f"state_effect_missing:{op.op_id}")
            if not stateful and op.state_effect and not unversioned:
                raise FeasibilityError(f"state_effect_without_state_port:{op.op_id}")
            if op.state_effect in {"read", "read_write"}:
                read_ports = [self.values[v] for v in op.inputs
                              if self.values[v].is_state]
                missing = (not read_ports
                           or any(value.state_version is None
                                  for value in read_ports))
                if missing and not (op.opaque and unversioned):
                    raise FeasibilityError(f"state_read_version_missing:{op.op_id}")
            if op.state_effect in {"write", "read_write"}:
                versions = [self.values[v].state_version for v in op.outputs
                            if self.values[v].is_state]
                missing = (not versions or any(v is None for v in versions))
                if missing:
                    # An unversioned write is an opaque barrier, never a
                    # synthesized ``0 -> 1`` transition.
                    if not (op.opaque and unversioned):
                        raise FeasibilityError(
                            f"state_write_version_missing:{op.op_id}")
                else:
                    reads = _state_read_versions(self, op)
                    writes = _state_write_versions(self, op)
                    # Compare a read-modify-write per target: advancing ``a``
                    # says nothing about the version of an unrelated ``b``.
                    for key, written in writes.items():
                        observed = reads.get(key)
                        if observed is not None and written is not None \
                                and written <= observed:
                            raise FeasibilityError(
                                f"state_version_not_advanced:{key}:{op.op_id}")
        by_key: dict[str, list[str]] = {}
        for op in self.ops.values():
            for key in state_keys(self, op):
                by_key.setdefault(key, []).append(op.op_id)
        for key, members in by_key.items():
            if len(members) < 2:
                continue
            reach = {member: _reachable(edges, (member,)) for member in members}
            for first in members:
                for second in members:
                    if first == second:
                        continue
                    forward = second in reach[first]
                    backward = first in reach[second]
                    if not (forward or backward):
                        raise FeasibilityError(
                            f"state_hazard_unordered:{key}:{first}:{second}")
            for first in members:
                for second in members:
                    if first == second or second not in reach[first]:
                        continue
                    written = _state_write_versions(self, self.ops[first])
                    reads = _state_read_versions(self, self.ops[second])
                    writes = _state_write_versions(self, self.ops[second])
                    # Check each target independently: a successor that reads
                    # ``a`` and writes ``b`` still must not move ``b`` back.
                    for key in written.keys() & (reads.keys() | writes.keys()):
                        version = written[key]
                        if version is None:
                            continue
                        for observed in (reads.get(key), writes.get(key)):
                            if observed is not None and observed < version:
                                raise FeasibilityError(
                                    "state_version_not_monotone:"
                                    f"{key}:{first}:{second}")

    def consumers(self, value_id: str) -> tuple[str, ...]:
        return _graph_index(self)["consumers"].get(value_id, ())

    def successors(self, op_id: str) -> tuple[str, ...]:
        """Forward neighbours only: explicit event successors and readers."""
        if op_id not in self.ops:
            _ = self.ops[op_id]
        return _graph_index(self)["successors"][op_id]

    def predecessors(self, op_id: str) -> tuple[str, ...]:
        """Backward neighbours only: explicit event predecessors and producers."""
        if op_id not in self.ops:
            _ = self.ops[op_id]
        return _graph_index(self)["predecessors"][op_id]

    def topological_order(self) -> tuple[str, ...]:
        ordered = _graph_index(self)["topo_order"]
        if ordered is None:
            raise FeasibilityError("graph_forward_dependency_cycle")
        return ordered


_STAMP_SLOT = 0; _INDEX_SLOT = 1


def _graph_index(graph: TypedGraph) -> dict[str, Any]:
    stamp = (
        tuple((op_id, op.inputs, op.outputs, op.deps)
              for op_id, op in graph.ops.items()),
        tuple(graph.values.keys()),
        graph.public_outputs,
    )
    cached = getattr(graph, "_cached_idx", None)
    if cached is not None and cached[_STAMP_SLOT] == stamp:
        return cached[_INDEX_SLOT]
    consumers_lists: dict[str, list[str]] = {}
    dep_children: dict[str, list[str]] = {}
    for op in graph.ops.values():
        for value_id in dict.fromkeys(op.inputs):
            consumers_lists.setdefault(value_id, []).append(op.op_id)
        for dep in dict.fromkeys(op.deps):
            if dep in graph.ops and dep != op.op_id:
                dep_children.setdefault(dep, []).append(op.op_id)
    consumers = {vid: tuple(ops) for vid, ops in consumers_lists.items()}
    successors: dict[str, tuple[str, ...]] = {}
    predecessors: dict[str, tuple[str, ...]] = {}
    fwd_edges: dict[str, tuple[str, ...]] = {}
    rev_lists: dict[str, list[str]] = {op_id: [] for op_id in graph.ops}
    for op_id, op in graph.ops.items():
        succ: list[str] = []
        seen_s: set[str] = set()
        for value_id in op.outputs:
            for consumer in consumers.get(value_id, ()):
                if consumer != op_id and consumer not in seen_s:
                    seen_s.add(consumer)
                    succ.append(consumer)
        for child in dep_children.get(op_id, ()):
            if child not in seen_s:
                seen_s.add(child)
                succ.append(child)
        succ_t = tuple(succ)
        successors[op_id] = succ_t
        fwd_edges[op_id] = succ_t
        for child in succ_t:
            rev_lists[child].append(op_id)
        pred: list[str] = []
        seen_p: set[str] = set()
        for dep in op.deps:
            if dep != op_id and dep not in seen_p:
                seen_p.add(dep)
                pred.append(dep)
        for value_id in op.inputs:
            val = graph.values.get(value_id)
            producer = val.producer if val is not None else None
            if producer is not None and producer != op_id and producer not in seen_p:
                seen_p.add(producer)
                pred.append(producer)
        predecessors[op_id] = tuple(pred)
    pending = {op_id: len(predecessors[op_id]) for op_id in graph.ops}
    ready = deque(op_id for op_id, count in pending.items() if not count)
    ordered: list[str] = []
    while ready:
        current = ready.popleft()
        ordered.append(current)
        for nxt in successors[current]:
            pending[nxt] -= _INDEX_SLOT
            if not pending[nxt]:
                ready.append(nxt)
    topo_order = tuple(ordered) if len(ordered) == len(graph.ops) else None
    topo_rank = ({op_id: idx for idx, op_id in enumerate(topo_order)}
                 if topo_order is not None else None)
    idx = {
        "consumers": consumers,
        "successors": successors,
        "predecessors": predecessors,
        "fwd_edges": fwd_edges,
        "rev_edges": {op_id: tuple(items) for op_id, items in rev_lists.items()},
        "topo_order": topo_order,
        "topo_rank": topo_rank,
        "public_set": frozenset(graph.public_outputs),
        "value_rank": {vid: i for i, vid in enumerate(graph.values)},
    }
    object.__setattr__(graph, "_cached_idx", (stamp, idx))
    return idx


def is_connected(graph: TypedGraph, members: Iterable[str]) -> bool:
    chosen = set(members)
    if not chosen:
        return False
    start = next(iter(chosen))
    seen, stack = {start}, [start]
    while stack:
        current = stack.pop()
        for nxt in (set(graph.predecessors(current)) | set(graph.successors(current))) & chosen:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen == chosen


def is_convex(graph: TypedGraph, members: Iterable[str]) -> bool:
    """No forward path may leave the region and re-enter it."""
    chosen = set(members)
    if not chosen:
        return False
    idx = _graph_index(graph)
    topo_rank = idx["topo_rank"]
    fwd_edges = idx["fwd_edges"]
    rev_edges = idx["rev_edges"]
    if topo_rank is not None and all(m in topo_rank for m in chosen):
        if len(chosen) == _INDEX_SLOT:
            return True
        min_r = min(topo_rank[m] for m in chosen)
        max_r = max(topo_rank[m] for m in chosen)
        fwd_seen: set[str] = set()
        stack = list(chosen)
        while stack:
            for nxt in fwd_edges.get(stack.pop(), ()):
                if nxt not in fwd_seen and topo_rank[nxt] <= max_r:
                    fwd_seen.add(nxt)
                    stack.append(nxt)
        outside_fwd = fwd_seen - chosen
        if not outside_fwd:
            return True
        rev_seen: set[str] = set()
        stack = list(chosen)
        while stack:
            for prv in rev_edges.get(stack.pop(), ()):
                if prv not in rev_seen and topo_rank[prv] >= min_r:
                    if prv in outside_fwd:
                        return False
                    rev_seen.add(prv)
                    stack.append(prv)
        return True
    fwd = _reachable(fwd_edges, chosen)
    rev = _reachable(rev_edges, chosen)
    return not ((fwd & rev) - chosen)


def _is_public(graph: TypedGraph, value_id: str) -> bool:
    return graph.values[value_id].public or value_id in _graph_index(graph)["public_set"]


def external_readers(graph: TypedGraph, members: Iterable[str]) -> dict[str, tuple[str, ...]]:
    chosen = set(members)
    found: dict[str, tuple[str, ...]] = {}
    for op_id in chosen:
        for value_id in graph.ops[op_id].outputs:
            outside = tuple(c for c in graph.consumers(value_id) if c not in chosen)
            if outside or _is_public(graph, value_id):
                found[value_id] = outside
    return found


def region_causal_order(graph: TypedGraph, members: Iterable[str]) -> tuple[str, ...]:
    """Topological order restricted to the region; rejects a broken region."""
    chosen = set(members)
    pending = {op_id: len(set(graph.predecessors(op_id)) & chosen) for op_id in chosen}
    ready = sorted(op_id for op_id, count in pending.items() if not count)
    ordered: list[str] = []
    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for successor in graph.successors(current):
            if successor in chosen:
                pending[successor] -= 1
                if not pending[successor]:
                    ready.append(successor)
    if len(ordered) != len(chosen):
        raise FeasibilityError("region_causal_cycle")
    return tuple(ordered)
