"""Vector sub-pipe lowering and prepared-row max-min epoch solver.

[Role] Lower typed AIV vector instruction groups onto ``vec_issue`` plus
  declared ``vec_{alu,sfu,reduce}`` sub-pipes without a competing aggregate
  ``vector`` row, and provide the canonical Python prepared-row weighted
  max-min epoch allocator/integrator mirrored bit-for-bit in C11.
[Invariants] ``target_eff`` scales issue and sub-pipe throughput while leaving
  hardware result latency cycles invariant. Prepared-row conservation holds:
  every flow consumes at most ``z_left`` and delivers ``sum(dz) * work_per_z``
  to declared counters with residual snapping within ``RESIDUAL_SNAP_ULPS``.
[Timeouts & Locks] Bounded progressive-filling loops with ``guard <= n + 1``;
  no I/O, threads or locks.
[Side-Effect Policy] Pure value builders except explicit in-place ``z_left``
  and ``counters`` integration inside ``integrate_prepared_epoch``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError

RESIDUAL_SNAP_ULPS = 4.0
POLICY_FAIR = 0
POLICY_FIFO = 1
MS_PER_SECOND = 1000.0
VALID_VECTOR_PIPES = frozenset({"vec_alu", "vec_sfu", "vec_reduce"})


@dataclass(frozen=True, slots=True)
class VectorInstructionGroup:
    """One typed AIV instruction group targeting ``vec_issue`` and a sub-pipe."""

    group_id: str
    pipe: Literal["vec_alu", "vec_sfu", "vec_reduce"]
    count: int
    issue_slots: int = 1
    initiation_cycles: int = 1
    result_latency_cycles: int = 0
    dependencies: tuple[str, ...] = ()
    owner: int = 0
    last_issue_mark: bool = True

    def __post_init__(self) -> None:
        if not self.group_id:
            raise ModelingError("vector_group_id_required")
        if self.pipe not in VALID_VECTOR_PIPES:
            raise ModelingError(f"vector_group_pipe_invalid:{self.pipe!r}")
        if (
            self.count < 0 or self.issue_slots < 1
            or self.initiation_cycles < 1 or self.result_latency_cycles < 0
            or self.owner < 0
        ):
            raise ModelingError(f"vector_group_counts_invalid:{self.group_id}")


@dataclass(frozen=True, slots=True)
class BoundVectorOp:
    """Lowered ``vec_issue + vec_{alu,sfu,reduce}`` work without a competing total vector row."""

    owner: int
    issue_work: float
    pipe_work: Mapping[str, float]
    vector_projection: float
    raw_latency_ms: float
    span_ms: float
    stages: tuple[Any, ...] = ()


def bind_vector_groups(
    groups: Sequence[VectorInstructionGroup],
    *,
    issue_rate_per_ms: float,
    pipe_rates_per_ms: Mapping[str, float],
    clock_hz: float,
    target_eff: float = 1.0,
    resource_names: Mapping[str, str] | None = None,
) -> BoundVectorOp:
    """Lower AIV instruction groups onto ``vec_issue`` and declared sub-pipes.

    ``target_eff`` scales issue and sub-pipe throughput while leaving hardware
    result latency cycles invariant. The legacy ``vector`` total is computed as
    a pure reporting projection and never occupies a competing resource row.
    """
    from ..program.stages import Stage

    if issue_rate_per_ms <= 0.0 or clock_hz <= 0.0 or not (0.0 < target_eff <= 1.0):
        raise ModelingError("bind_vector_rates_invalid")
    eff_issue = issue_rate_per_ms * target_eff
    eff_pipes = {
        pipe: float(rate) * target_eff
        for pipe, rate in pipe_rates_per_ms.items()
    }
    res_map = dict(resource_names or {})
    if not groups:
        return BoundVectorOp(
            owner=0, issue_work=0.0, pipe_work={}, vector_projection=0.0,
            raw_latency_ms=0.0, span_ms=0.0, stages=(),
        )
    owner = groups[0].owner
    by_id: dict[str, VectorInstructionGroup] = {}
    for grp in groups:
        if grp.owner != owner:
            raise ModelingError(f"vector_group_owner_mismatch:{grp.group_id}")
        if grp.group_id in by_id:
            raise ModelingError(f"vector_group_duplicate:{grp.group_id}")
        if grp.pipe not in eff_pipes or eff_pipes[grp.pipe] <= 0.0:
            raise ModelingError(f"vector_subpipe_missing:{grp.pipe}")
        by_id[grp.group_id] = grp

    cycle_ms = MS_PER_SECOND / float(clock_hz)
    total_issue = 0.0
    pipe_totals: dict[str, float] = {}
    stages: list[Stage] = []
    raw_ready_by_id: dict[str, float] = {}
    max_raw_lat = 0.0

    for grp in groups:
        for dep in grp.dependencies:
            if dep not in by_id:
                raise ModelingError(f"vector_group_unknown_dep:{grp.group_id}:{dep}")
        start_ms = max((raw_ready_by_id[d] for d in grp.dependencies), default=0.0)
        issue_w = float(grp.count * grp.issue_slots)
        exec_w = float(grp.count * grp.initiation_cycles)
        total_issue += issue_w
        pipe_totals[grp.pipe] = pipe_totals.get(grp.pipe, 0.0) + exec_w
        thru_ms = max(issue_w / eff_issue, exec_w / eff_pipes[grp.pipe]) if grp.count > 0 else 0.0
        raw_lat_ms = float(grp.result_latency_cycles) * cycle_ms
        if raw_lat_ms > max_raw_lat:
            max_raw_lat = raw_lat_ms
        if grp.count > 1 and raw_lat_ms > 0.0 and grp.last_issue_mark:
            per_inst_ms = thru_ms / float(grp.count)
            last_issue_ms = start_ms + float(grp.count - 1) * per_inst_ms
            raw_ready = max(start_ms + thru_ms, last_issue_ms + raw_lat_ms)
        else:
            raw_ready = max(start_ms + thru_ms, start_ms + raw_lat_ms) if grp.count > 0 else start_ms
        raw_ready_by_id[grp.group_id] = raw_ready

        issue_res = res_map.get("vec_issue", "vec_issue")
        pipe_res = res_map.get(grp.pipe, grp.pipe)
        deps_after = tuple(
            f"{d}:raw" if by_id[d].result_latency_cycles > 0 else d
            for d in grp.dependencies
        )
        pl = {issue_res: (owner,), pipe_res: (owner,)}
        if grp.count > 1 and raw_lat_ms > 0.0 and grp.last_issue_mark:
            prefix_count = grp.count - 1
            stages.append(Stage(
                f"{grp.group_id}:prefix",
                work={
                    issue_res: float(prefix_count * grp.issue_slots) / eff_issue,
                    pipe_res: float(prefix_count * grp.initiation_cycles) / eff_pipes[grp.pipe],
                },
                after=deps_after,
                placement=pl,
                coupled=True,
            ))
            stages.append(Stage(
                grp.group_id,
                work={
                    issue_res: float(grp.issue_slots) / eff_issue,
                    pipe_res: float(grp.initiation_cycles) / eff_pipes[grp.pipe],
                },
                after=(f"{grp.group_id}:prefix",),
                placement=pl,
                coupled=True,
            ))
            stages.append(Stage(
                f"{grp.group_id}:raw",
                latency=raw_lat_ms,
                after=(f"{grp.group_id}:prefix",),
            ))
        else:
            stages.append(Stage(
                grp.group_id,
                work={
                    issue_res: issue_w / eff_issue,
                    pipe_res: exec_w / eff_pipes[grp.pipe],
                } if grp.count > 0 else {},
                after=deps_after,
                placement=pl if grp.count > 0 else {},
                coupled=grp.count > 0,
            ))
            if raw_lat_ms > 0.0 and grp.count > 0:
                stages.append(Stage(
                    f"{grp.group_id}:raw",
                    latency=raw_lat_ms,
                    after=(grp.group_id,),
                ))

    if not any(g.dependencies for g in groups):
        span_ms = max(
            total_issue / eff_issue,
            max((w / eff_pipes[p] for p, w in pipe_totals.items()), default=0.0),
            max(raw_ready_by_id.values(), default=0.0),
        )
    else:
        span_ms = max(raw_ready_by_id.values(), default=0.0)

    return BoundVectorOp(
        owner=owner,
        issue_work=total_issue,
        pipe_work=pipe_totals,
        vector_projection=sum(pipe_totals.values()),
        raw_latency_ms=max_raw_lat,
        span_ms=span_ms,
        stages=tuple(stages),
    )


@dataclass(frozen=True, slots=True)
class PreparedCapacityRow:
    """One normalized capacity row matching C11 ``msim_capacity_row``."""

    row_id: int
    scope: int = 0
    unit: int = 0
    policy: int = POLICY_FAIR
    capacity: float = 1.0


@dataclass(frozen=True, slots=True)
class PreparedTerm:
    """One nonnegative ``ms_per_z`` coefficient linking a flow to a capacity row."""

    row: int
    flow: int
    ms_per_z: float


@dataclass(frozen=True, slots=True)
class PreparedDelivery:
    """One counter delivery coefficient ``work_per_z`` for a flow."""

    counter: int
    flow: int
    work_per_z: float


@dataclass(slots=True)
class PreparedProgress:
    """Mutable progress cursor for one flow matching C11 ``msim_progress``."""

    sequence: int
    task: int
    priority: int
    placement_id: int
    member_map: int
    z_left: float
    weight: float
    max_dz_ms: float = math.inf


def allocate_prepared_epoch(
    rows: Sequence[PreparedCapacityRow],
    progs: Sequence[PreparedProgress],
    terms: Sequence[PreparedTerm],
) -> list[float]:
    """Reference prepared-row weighted max-min matching C11 ``msim_allocate_epoch``."""
    n_rows = len(rows)
    n_progs = len(progs)
    dz = [0.0] * n_progs
    if n_rows == 0 or n_progs == 0:
        return dz

    coalesced: dict[tuple[int, int], float] = {}
    for term in terms:
        if term.ms_per_z > 0.0:
            key = (term.flow, term.row)
            coalesced[key] = coalesced.get(key, 0.0) + float(term.ms_per_z)

    order = sorted(
        range(n_progs),
        key=lambda f: (progs[f].priority, progs[f].sequence, progs[f].placement_id, progs[f].member_map, f),
    )
    flow_terms: list[list[tuple[int, float]]] = [[] for _ in range(n_progs)]
    for (flow, row), coeff in sorted(coalesced.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        flow_terms[flow].append((row, coeff))

    fifo_winner: dict[int, int] = {}
    for f in order:
        if progs[f].z_left <= 0.0 or progs[f].weight <= 0.0 or progs[f].max_dz_ms <= 0.0:
            continue
        for row, coeff in flow_terms[f]:
            if coeff > 0.0 and rows[row].policy == POLICY_FIFO and row not in fifo_winner:
                fifo_winner[row] = f

    rem_cap = [float(r.capacity) for r in rows]
    by_prio: dict[int, list[int]] = {}
    for f in order:
        by_prio.setdefault(progs[f].priority, []).append(f)

    for prio in sorted(by_prio):
        free: list[int] = []
        for f in by_prio[prio]:
            p = progs[f]
            if p.z_left <= 0.0 or p.weight <= 0.0 or p.max_dz_ms <= 0.0:
                continue
            blocked = False
            for row, coeff in flow_terms[f]:
                if coeff > 0.0:
                    if rows[row].policy == POLICY_FIFO and fifo_winner.get(row) != f:
                        blocked = True
                        break
                    if rem_cap[row] <= 0.0:
                        blocked = True
                        break
            if not blocked:
                free.append(f)
        guard = 0
        while free and guard <= n_progs + 1:
            guard += 1
            slope = [0.0] * n_rows
            for f in free:
                w_f = progs[f].weight
                for row, coeff in flow_terms[f]:
                    slope[row] += coeff * w_f
            delta = math.inf
            bounds = [math.inf] * n_rows
            for j in range(n_rows):
                if slope[j] > 0.0:
                    b = rem_cap[j] / slope[j]
                    bounds[j] = b
                    if b < delta:
                        delta = b
            for f in free:
                rem_v = (progs[f].max_dz_ms - dz[f]) / progs[f].weight
                if rem_v < delta:
                    delta = rem_v
            if not math.isfinite(delta) or delta < 0.0:
                break
            for f in free:
                dz[f] += delta * progs[f].weight
            saturated: set[int] = set()
            for j in range(n_rows):
                if slope[j] > 0.0:
                    nv = rem_cap[j] - delta * slope[j]
                    rem_cap[j] = nv if nv > 0.0 else 0.0
                    if bounds[j] <= delta or rem_cap[j] <= 0.0:
                        saturated.add(j)
            next_free: list[int] = []
            for f in free:
                if dz[f] >= progs[f].max_dz_ms:
                    continue
                if any(row in saturated for row, _coeff in flow_terms[f]):
                    continue
                next_free.append(f)
            free = next_free
    return dz


def integrate_prepared_epoch(
    progs: Sequence[PreparedProgress],
    deliveries: Sequence[PreparedDelivery],
    dz_per_ms: Sequence[float],
    dt_ms: float,
    counters: list[float],
) -> list[float]:
    """Integrate one epoch ``dt_ms`` matching C11 ``msim_integrate_epoch``."""
    if not math.isfinite(dt_ms) or dt_ms < 0.0:
        raise ModelingError(f"prepared_epoch_dt_invalid:{dt_ms!r}")
    dz_steps = [0.0] * len(progs)
    for f, p in enumerate(progs):
        rate = float(dz_per_ms[f])
        if p.z_left <= 0.0 or rate <= 0.0:
            continue
        step = dt_ms * rate
        if step >= p.z_left:
            used = p.z_left
            p.z_left = 0.0
        else:
            residual = p.z_left - step
            if residual <= RESIDUAL_SNAP_ULPS * math.ulp(p.z_left):
                used = p.z_left
                p.z_left = 0.0
            else:
                used = step
                p.z_left = residual
        dz_steps[f] = used
    for d in deliveries:
        step = dz_steps[d.flow]
        if step > 0.0:
            counters[d.counter] += step * d.work_per_z
    return dz_steps


__all__ = [
    "BoundVectorOp",
    "POLICY_FAIR",
    "POLICY_FIFO",
    "PreparedCapacityRow",
    "PreparedDelivery",
    "PreparedProgress",
    "PreparedTerm",
    "RESIDUAL_SNAP_ULPS",
    "VALID_VECTOR_PIPES",
    "VectorInstructionGroup",
    "allocate_prepared_epoch",
    "bind_vector_groups",
    "integrate_prepared_epoch",
]
