"""Framework-layer general N-stream scheduling and per-stream core control.

[Role] Model arbitrary N >= 1 named execution streams per rank with explicit
  record/wait ``StreamEvent`` synchronization and per-stream AIC/AIV core
  control (``disjoint`` core-slice partitioning or ``shared_cap`` limits).
[Invariants] Whole-graph microbatch splitting (DBO) is never performed at the
  framework layer. Every ``StreamPlan`` declares unique ASCII lanes including
  ``main``. Disjoint AIC/AIV budgets never exceed die counts or overlap owners;
  coupled ``aic + aiv`` placements stay proportional for C11 ``_engine_accel``
  compatibility. Every cross-stream dependency is covered by transitive
  same-stream FIFO + event reachability, and the combined graph is a strict DAG.
[Timeouts & Locks] O(|V| + |E|) graph validation and deterministic owner
  binding; no I/O or locks.
[Side-Effect Policy] Returns new immutable OpSpec, Stage, Program and
  StreamOwners records; never mutates inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import heapq
import math
from typing import Any, Literal, TypeAlias

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import DatapathCapabilitySpec
from mindie_sim.schema.operators import OpSpec

StreamId: TypeAlias = str
StreamIsolation: TypeAlias = Literal["disjoint", "shared_cap"]
MAIN_STREAM: StreamId = "main"

AIC_ROLES: frozenset[str] = frozenset({"cube", "mte1", "mte2_aic", "fixpipe"})
AIV_ROLES: frozenset[str] = frozenset({
    "vector",
    "mte2_aiv",
    "mte3",
    "reg_data_writes",
    "reg_ub_accesses",
    "reg_ub_stores",
    "reg_alu_lanes",
    "reg_special_lanes",
    "reg_reduce_lanes",
    "reg_shuffle_lanes",
    "scalar_store_issue",
})


def validate_stream_id(value: str, field_name: str = "stream_id") -> StreamId:
    """Validate a non-empty ASCII stream identifier without whitespace."""
    if (
        type(value) is not str
        or not value
        or not value.isascii()
        or any(ch.isspace() for ch in value)
    ):
        raise ModelingError(f"stream_id_invalid:{field_name}:{value!r}")
    return value


def _optional_nonnegative_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ModelingError(f"stream_core_budget_invalid:{field_name}:{value!r}")
    return value


@dataclass(frozen=True, slots=True)
class StreamCoreBudget:
    """Per-stream AIC/AIV core-count budget and isolation mode (``控核``)."""

    aic: int | None = None
    aiv: int | None = None
    isolation: StreamIsolation = "disjoint"

    def __post_init__(self) -> None:
        _optional_nonnegative_int(self.aic, "aic")
        _optional_nonnegative_int(self.aiv, "aiv")
        if self.isolation not in {"disjoint", "shared_cap"}:
            raise ModelingError(f"stream_isolation_invalid:{self.isolation!r}")


@dataclass(frozen=True, slots=True)
class StreamLane:
    """One named execution lane inside a rank ``StreamPlan``."""

    stream_id: StreamId
    budget: StreamCoreBudget = StreamCoreBudget()
    priority: int = 0

    def __post_init__(self) -> None:
        validate_stream_id(self.stream_id, "stream_id")
        if type(self.priority) is not int or self.priority < 0:
            raise ModelingError(
                f"stream_priority_invalid:{self.stream_id}:{self.priority!r}"
            )


@dataclass(frozen=True, slots=True)
class StreamEvent:
    """Explicit cross-stream synchronization event: record on producer, wait on consumer."""

    event_id: str
    record_stream: StreamId
    record_after: str
    wait_stream: StreamId
    wait_before: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ModelingError("stream_event_id_required")
        validate_stream_id(self.record_stream, "record_stream")
        validate_stream_id(self.wait_stream, "wait_stream")
        if not self.record_after:
            raise ModelingError(f"stream_event_record_after_required:{self.event_id}")
        if not self.wait_before:
            raise ModelingError(f"stream_event_wait_before_required:{self.event_id}")


@dataclass(frozen=True, slots=True)
class StreamPlan:
    """Rank-local assignment of operators to N >= 1 streams, core budgets, and events."""

    rank: int
    lanes: tuple[StreamLane, ...]
    assignments: tuple[tuple[str, StreamId], ...] = ()
    events: tuple[StreamEvent, ...] = ()

    def __post_init__(self) -> None:
        if not self.lanes:
            raise ModelingError("stream_plan_lanes_required")
        seen: set[str] = set()
        for lane in self.lanes:
            if lane.stream_id in seen:
                raise ModelingError(f"stream_plan_duplicate_lane:{lane.stream_id}")
            seen.add(lane.stream_id)
        if MAIN_STREAM not in seen:
            raise ModelingError("stream_plan_main_lane_required")

    def lane_ids(self) -> tuple[StreamId, ...]:
        """Return declared stream identifiers in declaration order."""
        return tuple(lane.stream_id for lane in self.lanes)

    def lane(self, stream_id: StreamId) -> StreamLane:
        """Look up one declared lane by stream identifier."""
        for item in self.lanes:
            if item.stream_id == stream_id:
                return item
        raise ModelingError(f"stream_plan_unknown_lane:{stream_id}")


@dataclass(frozen=True, slots=True)
class StreamOwners:
    """Bound physical core slices and effective core limits for one stream lane."""

    aic: tuple[int, ...]
    aiv: tuple[int, ...]
    aic_limit: int
    aiv_limit: int
    isolation: StreamIsolation = "disjoint"
    pool_aic: tuple[int, ...] = ()
    pool_aiv: tuple[int, ...] = ()


def _resolve_disjoint_counts(
    lanes: list[StreamLane],
    *,
    total: int,
    kind: Literal["aic", "aiv"],
) -> dict[StreamId, int]:
    unbounded = [lane for lane in lanes if getattr(lane.budget, kind) is None]
    explicit = sum(
        int(getattr(lane.budget, kind))
        for lane in lanes
        if getattr(lane.budget, kind) is not None
    )
    if explicit > total:
        raise ModelingError(f"stream_core_budget_{kind}_exceeded:{explicit}>{total}")
    if len(unbounded) > 1:
        raise ModelingError(f"stream_disjoint_ambiguous_unbounded_{kind}")
    remainder = total - explicit
    return {
        lane.stream_id: (
            remainder if getattr(lane.budget, kind) is None
            else int(getattr(lane.budget, kind))
        )
        for lane in lanes
    }


def bind_stream_owners(
    plan: StreamPlan,
    datapath: DatapathCapabilitySpec,
    /,
) -> dict[StreamId, StreamOwners]:
    """Bind every lane of ``plan`` to deterministic physical owners and core limits."""
    disjoint_lanes = [l for l in plan.lanes if l.budget.isolation == "disjoint"]
    shared_lanes = [l for l in plan.lanes if l.budget.isolation == "shared_cap"]

    aic_counts = _resolve_disjoint_counts(
        disjoint_lanes, total=datapath.aic_count, kind="aic"
    )
    aiv_counts = _resolve_disjoint_counts(
        disjoint_lanes, total=datapath.aiv_count, kind="aiv"
    )

    aic_alloc: dict[StreamId, list[int]] = {l.stream_id: [] for l in disjoint_lanes}
    aiv_alloc: dict[StreamId, list[int]] = {l.stream_id: [] for l in disjoint_lanes}
    used_aic: set[int] = set()
    used_aiv: set[int] = set()

    if datapath.aic_aiv_pairs:
        pairs = list(datapath.aic_aiv_pairs)
        cursor = 0
        # Pass 1: preserve AIC->AIV pair affinity across all disjoint lanes first.
        for lane in disjoint_lanes:
            sid = lane.stream_id
            need_aic = aic_counts[sid]
            need_aiv = aiv_counts[sid]
            take = min(need_aic, max(0, len(pairs) - cursor))
            for aic_id, paired_aivs in pairs[cursor : cursor + take]:
                aic_alloc[sid].append(aic_id)
                used_aic.add(aic_id)
                for aiv_id in paired_aivs:
                    if len(aiv_alloc[sid]) < need_aiv and aiv_id not in used_aiv:
                        aiv_alloc[sid].append(aiv_id)
                        used_aiv.add(aiv_id)
            cursor += take
        # Pass 2: fill any remaining AIC/AIV counts from unclaimed die cores.
        rem_aic = [i for i in range(datapath.aic_count) if i not in used_aic]
        rem_aiv = [i for i in range(datapath.aiv_count) if i not in used_aiv]
        for lane in disjoint_lanes:
            sid = lane.stream_id
            deficit_aic = aic_counts[sid] - len(aic_alloc[sid])
            if deficit_aic > 0:
                picked = rem_aic[:deficit_aic]
                aic_alloc[sid].extend(picked)
                used_aic.update(picked)
                rem_aic = rem_aic[deficit_aic:]
            deficit_aiv = aiv_counts[sid] - len(aiv_alloc[sid])
            if deficit_aiv > 0:
                picked = rem_aiv[:deficit_aiv]
                aiv_alloc[sid].extend(picked)
                used_aiv.update(picked)
                rem_aiv = rem_aiv[deficit_aiv:]
    else:
        aic_pos = 0
        aiv_pos = 0
        for lane in disjoint_lanes:
            sid = lane.stream_id
            n_aic = aic_counts[sid]
            n_aiv = aiv_counts[sid]
            picked_aic = list(range(aic_pos, aic_pos + n_aic))
            picked_aiv = list(range(aiv_pos, aiv_pos + n_aiv))
            aic_alloc[sid] = picked_aic
            aiv_alloc[sid] = picked_aiv
            used_aic.update(picked_aic)
            used_aiv.update(picked_aiv)
            aic_pos += n_aic
            aiv_pos += n_aiv

    owners_by_id: dict[StreamId, StreamOwners] = {}
    for lane in disjoint_lanes:
        sid = lane.stream_id
        aic_ids = tuple(aic_alloc[sid])
        aiv_ids = tuple(aiv_alloc[sid])
        owners_by_id[sid] = StreamOwners(
            aic=aic_ids,
            aiv=aiv_ids,
            aic_limit=len(aic_ids),
            aiv_limit=len(aiv_ids),
            isolation="disjoint",
        )

    avail_aic_ids = tuple(i for i in range(datapath.aic_count) if i not in used_aic)
    avail_aiv_ids = tuple(i for i in range(datapath.aiv_count) if i not in used_aiv)
    avail_aic = len(avail_aic_ids)
    avail_aiv = len(avail_aiv_ids)

    for lane in shared_lanes:
        sid = lane.stream_id
        aic_lim = avail_aic if lane.budget.aic is None else lane.budget.aic
        aiv_lim = avail_aiv if lane.budget.aiv is None else lane.budget.aiv
        if aic_lim > datapath.aic_count:
            raise ModelingError(
                f"stream_shared_cap_aic_exceeded:{sid}:{aic_lim}>{datapath.aic_count}"
            )
        if aic_lim > avail_aic:
            raise ModelingError(
                f"stream_shared_cap_aic_exceeds_shared_pool:{sid}:{aic_lim}>{avail_aic}"
            )
        if aiv_lim > datapath.aiv_count:
            raise ModelingError(
                f"stream_shared_cap_aiv_exceeded:{sid}:{aiv_lim}>{datapath.aiv_count}"
            )
        if aiv_lim > avail_aiv:
            raise ModelingError(
                f"stream_shared_cap_aiv_exceeds_shared_pool:{sid}:{aiv_lim}>{avail_aiv}"
            )
        has_disjoint_pool = bool(used_aic or used_aiv)
        owners_by_id[sid] = StreamOwners(
            aic=(),
            aiv=(),
            aic_limit=aic_lim,
            aiv_limit=aiv_lim,
            isolation="shared_cap",
            pool_aic=avail_aic_ids if has_disjoint_pool else (),
            pool_aiv=avail_aiv_ids if has_disjoint_pool else (),
        )

    return {lane.stream_id: owners_by_id[lane.stream_id] for lane in plan.lanes}


def proportional_coupled_placement(
    owners: StreamOwners,
    datapath: DatapathCapabilitySpec,
    *,
    stream_id: StreamId = MAIN_STREAM,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the largest proportional ``(aic, aiv)`` slice safe for C11 ``_engine_accel``."""
    aic_ids = owners.aic if owners.isolation == "disjoint" else owners.pool_aic
    aiv_ids = owners.aiv if owners.isolation == "disjoint" else owners.pool_aiv
    if not aic_ids or not aiv_ids:
        raise ModelingError(
            f"stream_coupled_placement_zero:{stream_id}:{len(aic_ids)}:{len(aiv_ids)}"
        )
    gcd = math.gcd(datapath.aic_count, datapath.aiv_count)
    step_aic = datapath.aic_count // gcd
    step_aiv = datapath.aiv_count // gcd
    groups = min(len(aic_ids) // step_aic, len(aiv_ids) // step_aiv)
    if groups <= 0:
        raise ModelingError(
            f"stream_coupled_placement_zero:{stream_id}:{len(aic_ids)}:{len(aiv_ids)}"
        )
    return aic_ids[: groups * step_aic], aiv_ids[: groups * step_aiv]


def _remap_units(existing: tuple[int, ...] | None, pool: tuple[int, ...]) -> tuple[int, ...]:
    if not existing:
        return pool
    seen: list[int] = []
    for unit in existing:
        mapped = pool[unit % len(pool)]
        if mapped not in seen:
            seen.append(mapped)
    return tuple(seen)


def _cap_rate(
    caps: dict[str, float],
    res: str,
    rates: Mapping[str, float],
    limit: int,
    total: int,
) -> None:
    if limit >= total or total <= 0:
        return
    base_rate = rates.get(res)
    if base_rate is None or base_rate <= 0.0:
        return
    bound = float(base_rate) * (float(limit) / float(total))
    prev = caps.get(res)
    caps[res] = bound if prev is None else min(prev, bound)


def bind_stage_to_stream(
    stage: Any,
    owners: StreamOwners,
    datapath: DatapathCapabilitySpec,
    *,
    resource_names: Mapping[str, str],
    resource_units: Mapping[str, int],
    rates: Mapping[str, float],
    stream_id: StreamId = MAIN_STREAM,
) -> Any:
    """Bind one ``Stage`` to its lane's physical core owners and rate limits."""
    role_by_res = {phys: role for role, phys in resource_names.items()}
    aic_res = [
        res for res, amt in stage.work.items()
        if amt > 0.0 and role_by_res.get(res) in AIC_ROLES
    ]
    aiv_res = [
        res for res, amt in stage.work.items()
        if amt > 0.0 and role_by_res.get(res) in AIV_ROLES
    ]
    other_res = [
        res for res, amt in stage.work.items()
        if amt > 0.0 and res not in aic_res and res not in aiv_res
    ]

    if aic_res and owners.aic_limit == 0:
        raise ModelingError(f"stream_zero_aic_budget:{stream_id}:{stage.name}")
    if aiv_res and owners.aiv_limit == 0:
        raise ModelingError(f"stream_zero_aiv_budget:{stream_id}:{stage.name}")

    placement = dict(stage.placement)
    caps = dict(stage.caps)
    coupled_has_unit_res = stage.coupled and any(
        int(resource_units.get(role_by_res.get(res, ""), 1)) <= 1
        for res in other_res
    )

    if owners.isolation == "disjoint":
        if stage.coupled and aic_res and aiv_res:
            aic_slice, aiv_slice = proportional_coupled_placement(
                owners, datapath, stream_id=stream_id
            )
            if coupled_has_unit_res and len(aic_slice) < datapath.aic_count:
                for res in aic_res:
                    _cap_rate(caps, res, rates, len(aic_slice), datapath.aic_count)
                for res in aiv_res:
                    _cap_rate(caps, res, rates, len(aiv_slice), datapath.aiv_count)
            else:
                for res in aic_res:
                    placement[res] = aic_slice
                for res in aiv_res:
                    placement[res] = aiv_slice
        else:
            if aic_res:
                if coupled_has_unit_res and len(owners.aic) < datapath.aic_count:
                    for res in aic_res:
                        _cap_rate(caps, res, rates, len(owners.aic), datapath.aic_count)
                else:
                    for res in aic_res:
                        placement[res] = _remap_units(placement.get(res), owners.aic)
            if aiv_res:
                if coupled_has_unit_res and len(owners.aiv) < datapath.aiv_count:
                    for res in aiv_res:
                        _cap_rate(caps, res, rates, len(owners.aiv), datapath.aiv_count)
                else:
                    for res in aiv_res:
                        placement[res] = _remap_units(placement.get(res), owners.aiv)
    else:
        for res in aic_res:
            _cap_rate(caps, res, rates, owners.aic_limit, datapath.aic_count)
            if owners.pool_aic and not coupled_has_unit_res:
                placement[res] = _remap_units(placement.get(res), owners.pool_aic)
        for res in aiv_res:
            _cap_rate(caps, res, rates, owners.aiv_limit, datapath.aiv_count)
            if owners.pool_aiv and not coupled_has_unit_res:
                placement[res] = _remap_units(placement.get(res), owners.pool_aiv)
        if stage.coupled and aic_res and aiv_res and (owners.pool_aic or owners.pool_aiv):
            if not coupled_has_unit_res and owners.pool_aic and owners.pool_aiv:
                aic_slice, aiv_slice = proportional_coupled_placement(
                    owners, datapath, stream_id=stream_id
                )
                for res in aic_res:
                    placement[res] = aic_slice
                for res in aiv_res:
                    placement[res] = aiv_slice

    nested = (
        apply_stream_owners_to_program(
            stage.program,
            owners,
            datapath,
            resource_names=resource_names,
            resource_units=resource_units,
            rates=rates,
            stream_id=stream_id,
        )
        if stage.program is not None else None
    )
    if (
        placement == dict(stage.placement)
        and caps == dict(stage.caps)
        and nested is stage.program
    ):
        return stage
    return replace(stage, placement=placement, caps=caps, program=nested)


def check_work_against_stream_budget(
    work: Mapping[str, float],
    owners: StreamOwners,
    *,
    resource_names: Mapping[str, str],
    stream_id: StreamId = MAIN_STREAM,
    label: str = "op",
) -> None:
    """Reject AIC or AIV work placed on a stream whose core budget for that engine is 0."""
    role_by_res = {phys: role for role, phys in resource_names.items()}
    if owners.aic_limit == 0 and any(
        amt > 0.0 and role_by_res.get(res) in AIC_ROLES for res, amt in work.items()
    ):
        raise ModelingError(f"stream_zero_aic_budget:{stream_id}:{label}")
    if owners.aiv_limit == 0 and any(
        amt > 0.0 and role_by_res.get(res) in AIV_ROLES for res, amt in work.items()
    ):
        raise ModelingError(f"stream_zero_aiv_budget:{stream_id}:{label}")


def apply_stream_owners_to_program(
    program: Any,
    owners: StreamOwners,
    datapath: DatapathCapabilitySpec,
    *,
    resource_names: Mapping[str, str],
    resource_units: Mapping[str, int],
    rates: Mapping[str, float],
    stream_id: StreamId = MAIN_STREAM,
) -> Any:
    """Recursively bind all stages of ``program`` and its wrappers to ``owners``."""
    from .program import (
        KernelProgram,
        NativeProgram,
        ParallelProgram,
        PipelineProgram,
        PipelineTemplateProgram,
    )

    kwargs = {
        "resource_names": resource_names,
        "resource_units": resource_units,
        "rates": rates,
        "stream_id": stream_id,
    }
    if isinstance(program, KernelProgram):
        bound = tuple(
            bind_stage_to_stream(stage, owners, datapath, **kwargs)
            for stage in program.stages
        )
        return replace(program, stages=bound)
    if isinstance(program, ParallelProgram):
        members = tuple(
            apply_stream_owners_to_program(member, owners, datapath, **kwargs)
            for member in program.programs
        )
        return replace(program, programs=members)
    if isinstance(program, PipelineProgram):
        orig_tile = program.tile

        def bound_tile(index: int):
            return tuple(
                bind_stage_to_stream(stage, owners, datapath, **kwargs)
                for stage in orig_tile(index)
            )

        return replace(program, tile=bound_tile)
    if isinstance(program, PipelineTemplateProgram):
        phases = tuple(
            replace(
                phase,
                stages=tuple(
                    bind_stage_to_stream(stage, owners, datapath, **kwargs)
                    for stage in phase.stages
                ),
            )
            for phase in program.phases
        )
        return replace(program, phases=phases)
    if isinstance(program, NativeProgram):
        inner = apply_stream_owners_to_program(program.program, owners, datapath, **kwargs)
        return program if inner is program.program else replace(program, program=inner)
    if hasattr(program, "body") and getattr(program, "body", None) is not None:
        inner = apply_stream_owners_to_program(program.body, owners, datapath, **kwargs)
        return program if inner is program.body else replace(program, body=inner)
    if hasattr(program, "program") and getattr(program, "program", None) is not None:
        inner = apply_stream_owners_to_program(program.program, owners, datapath, **kwargs)
        return program if inner is program.program else replace(program, program=inner)
    return program


def _topological_order(
    n_nodes: int,
    edges: list[set[int]],
    error_code: str,
) -> list[int]:
    indegree = [0] * n_nodes
    for targets in edges:
        for dst in targets:
            indegree[dst] += 1
    heap = [i for i in range(n_nodes) if indegree[i] == 0]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        node = heapq.heappop(heap)
        order.append(node)
        for nxt in edges[node]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                heapq.heappush(heap, nxt)
    if len(order) != n_nodes:
        raise ModelingError(error_code)
    return order


def lower_stream_edges(
    ops: tuple[OpSpec, ...],
    plan: StreamPlan,
    /,
) -> tuple[OpSpec, ...]:
    """Bind N streams, verify transitive FIFO+event reachability, and enforce DAG order."""
    op_ids = tuple(op.op_id for op in ops)
    index_of = {op_id: i for i, op_id in enumerate(op_ids)}
    known = set(op_ids)
    valid_lanes = set(plan.lane_ids())

    stream_of: dict[str, StreamId] = {}
    for op_id, stream_id in plan.assignments:
        if op_id not in known:
            raise ModelingError(f"stream_unknown_operator:{op_id}")
        if stream_id not in valid_lanes:
            raise ModelingError(f"stream_unknown_lane:{op_id}:{stream_id}")
        if op_id in stream_of:
            raise ModelingError(f"stream_duplicate_assignment:{op_id}")
        stream_of[op_id] = stream_id
    for op_id in op_ids:
        stream_of.setdefault(op_id, MAIN_STREAM)

    stream_order: dict[StreamId, list[int]] = {sid: [] for sid in plan.lane_ids()}
    for i, op_id in enumerate(op_ids):
        stream_order[stream_of[op_id]].append(i)

    seen_events: set[str] = set()
    event_waits_for: dict[str, list[str]] = {op_id: [] for op_id in op_ids}
    n_ops = len(op_ids)
    sync_edges: list[set[int]] = [set() for _ in range(n_ops)]

    for s_indices in stream_order.values():
        for prev_i, next_i in zip(s_indices, s_indices[1:]):
            sync_edges[prev_i].add(next_i)

    for event in plan.events:
        if event.event_id in seen_events:
            raise ModelingError(f"stream_duplicate_event:{event.event_id}")
        seen_events.add(event.event_id)
        if event.record_stream not in valid_lanes:
            raise ModelingError(
                f"stream_event_unknown_record_lane:{event.event_id}:{event.record_stream}"
            )
        if event.wait_stream not in valid_lanes:
            raise ModelingError(
                f"stream_event_unknown_wait_lane:{event.event_id}:{event.wait_stream}"
            )
        if event.record_after not in known:
            raise ModelingError(f"stream_unknown_event_record:{event.record_after}")
        if stream_of[event.record_after] != event.record_stream:
            raise ModelingError(
                f"stream_event_record_stream_mismatch:{event.event_id}:{event.record_after}"
            )
        rec_i = index_of[event.record_after]
        for target in event.wait_before:
            if target not in known:
                raise ModelingError(f"stream_unknown_event_wait:{target}")
            if stream_of[target] != event.wait_stream:
                raise ModelingError(
                    f"stream_event_wait_stream_mismatch:{event.event_id}:{target}"
                )
            event_waits_for[target].append(event.record_after)
            sync_edges[rec_i].add(index_of[target])

    # Transitive reachability in the same-stream FIFO + StreamEvent subgraph.
    sync_topo = _topological_order(n_ops, sync_edges, "stream_event_cycle")
    ancestors = [1 << i for i in range(n_ops)]
    for u in sync_topo:
        mask_u = ancestors[u]
        for v in sync_edges[u]:
            ancestors[v] |= mask_u

    for i, op in enumerate(ops):
        consumer_stream = stream_of[op.op_id]
        for dep in op.deps:
            if dep not in known:
                continue
            if stream_of[dep] == consumer_stream:
                continue
            if not (ancestors[i] & (1 << index_of[dep])):
                raise ModelingError(
                    f"stream_cross_dependency_missing_event:{dep}->{op.op_id}"
                )

    combined_edges: list[set[int]] = [set(adj) for adj in sync_edges]
    for i, op in enumerate(ops):
        for dep in op.deps:
            if dep in known:
                combined_edges[index_of[dep]].add(i)
    topo_order = _topological_order(n_ops, combined_edges, "stream_event_cycle")

    lowered: list[OpSpec] = []
    for i in topo_order:
        op = ops[i]
        extra_deps = event_waits_for[op.op_id]
        merged_deps = (
            tuple(dict.fromkeys((*op.deps, *extra_deps))) if extra_deps else op.deps
        )
        lowered.append(replace(op, deps=merged_deps, stream=stream_of[op.op_id]))
    return tuple(lowered)


__all__ = [
    "AIC_ROLES",
    "AIV_ROLES",
    "MAIN_STREAM",
    "StreamCoreBudget",
    "StreamEvent",
    "StreamId",
    "StreamIsolation",
    "StreamLane",
    "StreamOwners",
    "StreamPlan",
    "apply_stream_owners_to_program",
    "bind_stage_to_stream",
    "bind_stream_owners",
    "check_work_against_stream_budget",
    "lower_stream_edges",
    "proportional_coupled_placement",
    "validate_stream_id",
]
