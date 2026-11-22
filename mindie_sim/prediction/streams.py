"""Framework-layer general N-stream scheduling and per-stream core control.

[Role] Model arbitrary N >= 1 named execution streams per rank with explicit
  record/wait ``StreamEvent`` synchronization and per-stream AIC/AIV core
  control (``disjoint`` core-slice partitioning or ``shared_cap`` aggregate
  lane quotas).
[Invariants] Whole-graph microbatch splitting (DBO) is never performed at the
  framework layer. Every ``StreamPlan`` declares unique ASCII lanes including
  ``main``. Disjoint AIC/AIV budgets never exceed die counts or overlap owners;
  coupled ``aic + aiv`` placements preserve narrow tile widths and stay
  proportional for C11 ``_engine_accel`` compatibility. Cross-stream sync uses
  ``O(|V| * |queues|)`` vector clocks keyed by ``(stage, stream_id)``.
[Timeouts & Locks] O(|V| * |queues| + |E|) validation and deterministic owner
  binding; no I/O or locks.
[Side-Effect Policy] Returns new immutable OpSpec, Stage, Program, Resource
  and StreamOwners records; never mutates inputs.
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
STREAM_QUOTA_PREFIX = "stream_quota@"

AIC_ROLES: frozenset[str] = frozenset({"cube", "mte1", "mte2_aic", "fixpipe"})
AIV_ROLES: frozenset[str] = frozenset({
    "vector", "mte2_aiv", "mte3", "reg_data_writes", "reg_ub_accesses",
    "reg_ub_stores", "reg_alu_lanes", "reg_special_lanes", "reg_reduce_lanes",
    "reg_shuffle_lanes", "scalar_store_issue",
})
AIC_BUFFER_POOLS: frozenset[str] = frozenset({"l1", "l0a", "l0b", "l0c"})
AIV_BUFFER_POOLS: frozenset[str] = frozenset({"ub"})


def validate_stream_id(value: str, field_name: str = "stream_id") -> StreamId:
    """Validate a non-empty ASCII stream identifier without whitespace."""
    if type(value) is not str or not value or not value.isascii() or any(ch.isspace() for ch in value):
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
    lanes: list[StreamLane], *, total: int, kind: Literal["aic", "aiv"]
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
    plan: StreamPlan, datapath: DatapathCapabilitySpec, /
) -> dict[StreamId, StreamOwners]:
    """Bind every lane of ``plan`` to deterministic physical owners and core limits."""
    disjoint_lanes = [l for l in plan.lanes if l.budget.isolation == "disjoint"]
    shared_lanes = [l for l in plan.lanes if l.budget.isolation == "shared_cap"]
    aic_counts = _resolve_disjoint_counts(disjoint_lanes, total=datapath.aic_count, kind="aic")
    aiv_counts = _resolve_disjoint_counts(disjoint_lanes, total=datapath.aiv_count, kind="aiv")

    aic_alloc: dict[StreamId, list[int]] = {l.stream_id: [] for l in disjoint_lanes}
    aiv_alloc: dict[StreamId, list[int]] = {l.stream_id: [] for l in disjoint_lanes}
    used_aic: set[int] = set()
    used_aiv: set[int] = set()

    if datapath.aic_aiv_pairs:
        pairs = list(datapath.aic_aiv_pairs)
        cursor = 0
        for lane in disjoint_lanes:
            sid = lane.stream_id
            need_aic, need_aiv = aic_counts[sid], aiv_counts[sid]
            take = min(need_aic, max(0, len(pairs) - cursor))
            for aic_id, paired_aivs in pairs[cursor : cursor + take]:
                aic_alloc[sid].append(aic_id)
                used_aic.add(aic_id)
                for aiv_id in paired_aivs:
                    if len(aiv_alloc[sid]) < need_aiv and aiv_id not in used_aiv:
                        aiv_alloc[sid].append(aiv_id)
                        used_aiv.add(aiv_id)
            cursor += take
        rem_aic = [i for i in range(datapath.aic_count) if i not in used_aic]
        rem_aiv = [i for i in range(datapath.aiv_count) if i not in used_aiv]
        for lane in disjoint_lanes:
            sid = lane.stream_id
            d_aic = aic_counts[sid] - len(aic_alloc[sid])
            if d_aic > 0:
                picked = rem_aic[:d_aic]
                aic_alloc[sid].extend(picked)
                used_aic.update(picked)
                rem_aic = rem_aic[d_aic:]
            d_aiv = aiv_counts[sid] - len(aiv_alloc[sid])
            if d_aiv > 0:
                picked = rem_aiv[:d_aiv]
                aiv_alloc[sid].extend(picked)
                used_aiv.update(picked)
                rem_aiv = rem_aiv[d_aiv:]
    else:
        aic_pos = aiv_pos = 0
        for lane in disjoint_lanes:
            sid = lane.stream_id
            n_aic, n_aiv = aic_counts[sid], aiv_counts[sid]
            p_aic = list(range(aic_pos, aic_pos + n_aic))
            p_aiv = list(range(aiv_pos, aiv_pos + n_aiv))
            aic_alloc[sid], aiv_alloc[sid] = p_aic, p_aiv
            used_aic.update(p_aic)
            used_aiv.update(p_aiv)
            aic_pos += n_aic
            aiv_pos += n_aiv

    owners_by_id: dict[StreamId, StreamOwners] = {}
    for lane in disjoint_lanes:
        sid = lane.stream_id
        aic_ids, aiv_ids = tuple(aic_alloc[sid]), tuple(aiv_alloc[sid])
        owners_by_id[sid] = StreamOwners(
            aic=aic_ids, aiv=aiv_ids,
            aic_limit=len(aic_ids), aiv_limit=len(aiv_ids),
            isolation="disjoint",
        )

    avail_aic_ids = tuple(i for i in range(datapath.aic_count) if i not in used_aic)
    avail_aiv_ids = tuple(i for i in range(datapath.aiv_count) if i not in used_aiv)
    avail_aic, avail_aiv = len(avail_aic_ids), len(avail_aiv_ids)
    has_disjoint = bool(used_aic or used_aiv)

    for lane in shared_lanes:
        sid = lane.stream_id
        aic_lim = avail_aic if lane.budget.aic is None else lane.budget.aic
        aiv_lim = avail_aiv if lane.budget.aiv is None else lane.budget.aiv
        if aic_lim > datapath.aic_count:
            raise ModelingError(f"stream_shared_cap_aic_exceeded:{sid}:{aic_lim}>{datapath.aic_count}")
        if aic_lim > avail_aic:
            raise ModelingError(f"stream_shared_cap_aic_exceeds_shared_pool:{sid}:{aic_lim}>{avail_aic}")
        if aiv_lim > datapath.aiv_count:
            raise ModelingError(f"stream_shared_cap_aiv_exceeded:{sid}:{aiv_lim}>{datapath.aiv_count}")
        if aiv_lim > avail_aiv:
            raise ModelingError(f"stream_shared_cap_aiv_exceeds_shared_pool:{sid}:{aiv_lim}>{avail_aiv}")
        owners_by_id[sid] = StreamOwners(
            aic=(), aiv=(), aic_limit=aic_lim, aiv_limit=aiv_lim,
            isolation="shared_cap",
            pool_aic=avail_aic_ids if has_disjoint else (),
            pool_aiv=avail_aiv_ids if has_disjoint else (),
        )
    return {lane.stream_id: owners_by_id[lane.stream_id] for lane in plan.lanes}


def stream_quota_resources(
    owners_by_stream: Mapping[StreamId, StreamOwners],
    datapath: DatapathCapabilitySpec,
    *,
    resource_names: Mapping[str, str],
    rates: Mapping[str, float],
    work_units: Mapping[str, str] = (),  # type: ignore[assignment]
    replica: int = 0,
    stage: int = 0,
    namespace: str = "",
) -> tuple[tuple[Any, ...], dict[tuple[StreamId, str], str]]:
    """Build aggregate lane-quota ``Resource`` objects for ``shared_cap`` lanes."""
    from mindie_sim.prediction.engine import Resource

    unit_map = dict(work_units) if work_units else {}
    resources: list[Any] = []
    quota_map: dict[tuple[StreamId, str], str] = {}
    for sid, owners in owners_by_stream.items():
        if owners.isolation != "shared_cap":
            continue
        for role_set, limit, total, pool in (
            (AIC_ROLES, owners.aic_limit, datapath.aic_count, owners.pool_aic),
            (AIV_ROLES, owners.aiv_limit, datapath.aiv_count, owners.pool_aiv),
        ):
            effective_pool = len(pool) if pool else total
            if limit <= 0 or limit >= effective_pool or total <= 0:
                continue
            for role in sorted(role_set):
                phys = resource_names.get(role)
                if not phys or phys not in rates:
                    continue
                base_rate = float(rates[phys])
                if base_rate <= 0.0:
                    continue
                qname = f"{namespace}{STREAM_QUOTA_PREFIX}r{replica}:s{stage}:{sid}:{role}"
                q_rate = base_rate * (float(limit) / float(total))
                resources.append(Resource(
                    qname, q_rate, unit_map.get(role, "work"), "fair", units=1
                ))
                quota_map[(sid, phys)] = qname
    return tuple(resources), quota_map


def proportional_coupled_placement(
    owners: StreamOwners,
    datapath: DatapathCapabilitySpec,
    *,
    stream_id: StreamId = MAIN_STREAM,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the largest proportional ``(aic, aiv)`` slice safe for C11 ``_engine_accel``."""
    aic_ids = owners.aic if owners.isolation == "disjoint" else owners.pool_aic
    aiv_ids = owners.aiv if owners.isolation == "disjoint" else owners.pool_aiv
    return _proportional_pair(
        aic_ids, aiv_ids, datapath.aic_count, datapath.aiv_count, stream_id=stream_id
    )


def _proportional_pair(
    aic_ids: tuple[int, ...],
    aiv_ids: tuple[int, ...],
    aic_total: int,
    aiv_total: int,
    *,
    stream_id: StreamId,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if not aic_ids or not aiv_ids:
        raise ModelingError(
            f"stream_coupled_placement_zero:{stream_id}:{len(aic_ids)}:{len(aiv_ids)}"
        )
    gcd = math.gcd(aic_total, aiv_total)
    step_aic = aic_total // gcd
    step_aiv = aiv_total // gcd
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


def validate_stream_plan_ranks(
    stream_plans: Mapping[int, StreamPlan] | tuple[StreamPlan, ...] | list[StreamPlan] | None,
    active_replicas: set[int] | frozenset[int],
) -> tuple[StreamPlan, ...]:
    """Normalize and validate that every ``StreamPlan.rank`` matches an active replica."""
    if not stream_plans:
        return ()
    plans = tuple(stream_plans.values()) if isinstance(stream_plans, Mapping) else tuple(stream_plans)
    seen: set[int] = set()
    unmatched: list[int] = []
    for plan in plans:
        rank = int(plan.rank)
        if rank in seen:
            raise ModelingError("stream_duplicate_rank_plan")
        seen.add(rank)
        if rank not in active_replicas:
            unmatched.append(rank)
    if unmatched:
        raise ModelingError(
            "stream_plan_unmatched_rank:" + ",".join(str(r) for r in sorted(unmatched))
        )
    return plans


def _remap_allocations(
    allocs: tuple[Any, ...], owners: StreamOwners, datapath: DatapathCapabilitySpec,
    pool_names: Mapping[str, str] | None, pool_units: Mapping[str, int] | None,
) -> tuple[Any, ...]:
    aic_pool = owners.aic if owners.isolation == "disjoint" else owners.pool_aic
    aiv_pool = owners.aiv if owners.isolation == "disjoint" else owners.pool_aiv
    if not allocs or not pool_names or not (aic_pool or aiv_pool):
        return allocs
    logical_by_phys = {phys: name for name, phys in pool_names.items()}
    punits = pool_units or {}
    remapped, changed = [], False
    for alloc in allocs:
        logical = logical_by_phys.get(alloc.pool, alloc.pool)
        target_pool: tuple[int, ...] | None = None
        if logical in AIC_BUFFER_POOLS and aic_pool and int(punits.get(logical, datapath.aic_count)) == datapath.aic_count:
            target_pool = aic_pool
        elif logical in AIV_BUFFER_POOLS and aiv_pool and int(punits.get(logical, datapath.aiv_count)) == datapath.aiv_count:
            target_pool = aiv_pool
        if target_pool is not None and alloc.units:
            new_units = _remap_units(alloc.units, target_pool)
            if new_units != alloc.units:
                remapped.append(replace(alloc, units=new_units))
                changed = True
                continue
        remapped.append(alloc)
    return tuple(remapped) if changed else allocs


def _cap_rate(caps: dict[str, float], res: str, rates: Mapping[str, float], limit: int, total: int) -> None:
    if limit >= total or total <= 0:
        return
    base_rate = rates.get(res)
    if base_rate is None or base_rate <= 0.0:
        return
    bound = float(base_rate) * (float(limit) / float(total))
    prev = caps.get(res)
    caps[res] = bound if prev is None else min(prev, bound)


def check_work_against_stream_budget(
    work: Mapping[str, float], owners: StreamOwners, *,
    resource_names: Mapping[str, str], stream_id: StreamId = MAIN_STREAM, label: str = "op",
) -> None:
    """Reject AIC or AIV work placed on a stream whose core budget for that engine is 0."""
    role_by_res = {phys: role for role, phys in resource_names.items()}
    if owners.aic_limit == 0 and any(amt > 0.0 and role_by_res.get(res) in AIC_ROLES for res, amt in work.items()):
        raise ModelingError(f"stream_zero_aic_budget:{stream_id}:{label}")
    if owners.aiv_limit == 0 and any(amt > 0.0 and role_by_res.get(res) in AIV_ROLES for res, amt in work.items()):
        raise ModelingError(f"stream_zero_aiv_budget:{stream_id}:{label}")


def bind_stage_to_stream(
    stage: Any, owners: StreamOwners, datapath: DatapathCapabilitySpec, *,
    resource_names: Mapping[str, str], resource_units: Mapping[str, int],
    rates: Mapping[str, float], stream_id: StreamId = MAIN_STREAM,
    quota_resources: Mapping[str, str] | None = None,
    pool_names: Mapping[str, str] | None = None, pool_units: Mapping[str, int] | None = None,
) -> Any:
    """Bind one ``Stage`` to its lane's physical core owners and aggregate lane quotas."""
    from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage

    role_by_res = {phys: role for role, phys in resource_names.items()}
    aic_res = [r for r, a in stage.work.items() if a > 0.0 and role_by_res.get(r) in AIC_ROLES]
    aiv_res = [r for r, a in stage.work.items() if a > 0.0 and role_by_res.get(r) in AIV_ROLES]
    other_res = [r for r, a in stage.work.items() if a > 0.0 and r not in aic_res and r not in aiv_res]
    if aic_res and owners.aic_limit == 0:
        raise ModelingError(f"stream_zero_aic_budget:{stream_id}:{stage.name}")
    if aiv_res and owners.aiv_limit == 0:
        raise ModelingError(f"stream_zero_aiv_budget:{stream_id}:{stage.name}")
    work, placement, caps, coupled = dict(stage.work), dict(stage.placement), dict(stage.caps), bool(stage.coupled)
    coupled_has_unit_res = coupled and any(int(resource_units.get(role_by_res.get(r, ""), 1)) <= 1 for r in other_res)
    bind_kw = dict(
        resource_names=resource_names, resource_units=resource_units, rates=rates,
        stream_id=stream_id, quota_resources=quota_resources,
        pool_names=pool_names, pool_units=pool_units,
    )

    if owners.isolation == "disjoint":
        if coupled and aic_res and aiv_res:
            aic_pool, aiv_pool = proportional_coupled_placement(
                owners, datapath, stream_id=stream_id
            )
            orig_aic = next((placement[r] for r in aic_res if r in placement), None)
            orig_aiv = next((placement[r] for r in aiv_res if r in placement), None)
            mapped_aic = _remap_units(orig_aic, owners.aic) if orig_aic else aic_pool
            mapped_aiv = _remap_units(orig_aiv, owners.aiv) if orig_aiv else aiv_pool
            aic_slice, aiv_slice = _proportional_pair(
                mapped_aic, mapped_aiv, datapath.aic_count, datapath.aiv_count,
                stream_id=stream_id,
            )
            if coupled_has_unit_res and len(aic_slice) < datapath.aic_count:
                for res in aic_res:
                    placement.pop(res, None)
                    _cap_rate(caps, res, rates, len(aic_slice), datapath.aic_count)
                for res in aiv_res:
                    placement.pop(res, None)
                    _cap_rate(caps, res, rates, len(aiv_slice), datapath.aiv_count)
            else:
                for res in aic_res:
                    placement[res] = aic_slice
                for res in aiv_res:
                    placement[res] = aiv_slice
        else:
            for res_list, pool, total in (
                (aic_res, owners.aic, datapath.aic_count),
                (aiv_res, owners.aiv, datapath.aiv_count),
            ):
                for res in res_list:
                    mapped = _remap_units(placement.get(res), pool)
                    if coupled_has_unit_res and len(mapped) < total:
                        placement.pop(res, None)
                        _cap_rate(caps, res, rates, len(mapped), total)
                    else:
                        placement[res] = mapped
    else:
        qmap = quota_resources or {}
        if not coupled and len(stage.work) > 1 and any(res in qmap for res in (*aic_res, *aiv_res)):
            branches = []
            for idx, (res, amt) in enumerate(stage.work.items()):
                sub_work = {res: amt}
                sub_pl = {res: placement[res]} if res in placement else {}
                sub_caps = {res: caps[res]} if res in caps else {}
                sub_coupled = False
                is_aic, is_aiv = res in aic_res, res in aiv_res
                if is_aic or is_aiv:
                    limit = owners.aic_limit if is_aic else owners.aiv_limit
                    total = datapath.aic_count if is_aic else datapath.aiv_count
                    pool = owners.pool_aic if is_aic else owners.pool_aiv
                    orig = sub_pl.get(res)
                    if res in qmap:
                        sub_work[qmap[res]] = amt
                        sub_coupled = True
                        if orig is not None and len(orig) < total:
                            _cap_rate(sub_caps, res, rates, min(len(orig), limit), total)
                        if pool:
                            sub_pl[res] = _remap_units(orig, pool)
                        else:
                            sub_pl.pop(res, None)
                    else:
                        if pool:
                            sub_pl[res] = _remap_units(orig, pool)
                        _cap_rate(sub_caps, res, rates, min(len(orig), limit) if orig else limit, total)
                branches.append(KernelProgram((Stage(
                    f"{stage.name}:b{idx}", sub_work, caps=sub_caps,
                    placement=sub_pl, coupled=sub_coupled, priority=stage.priority,
                ),)))
            work_prog = ParallelProgram(tuple(branches))
            nested_inner = (
                apply_stream_owners_to_program(stage.program, owners, datapath, **bind_kw)
                if stage.program is not None else None
            )
            final_prog = (
                KernelProgram((
                    Stage(f"{stage.name}:work", program=work_prog),
                    Stage(f"{stage.name}:body", after=(f"{stage.name}:work",), program=nested_inner),
                ))
                if nested_inner is not None else work_prog
            )
            acq = _remap_allocations(stage.acquire, owners, datapath, pool_names, pool_units)
            rel = _remap_allocations(stage.free, owners, datapath, pool_names, pool_units)
            return replace(stage, work={}, placement={}, caps={}, acquire=acq, free=rel, program=final_prog)

        for res_list, limit, total, pool in (
            (aic_res, owners.aic_limit, datapath.aic_count, owners.pool_aic),
            (aiv_res, owners.aiv_limit, datapath.aiv_count, owners.pool_aiv),
        ):
            for res in res_list:
                orig = placement.get(res)
                if res in qmap:
                    work[qmap[res]] = float(stage.work[res])
                    coupled = True
                    if orig is not None and len(orig) < total:
                        _cap_rate(caps, res, rates, min(len(orig), limit), total)
                    if pool and not coupled_has_unit_res:
                        placement[res] = _remap_units(orig, pool)
                    else:
                        placement.pop(res, None)
                else:
                    eff_lim = min(len(orig), limit) if orig else limit
                    _cap_rate(caps, res, rates, eff_lim, total)
                    if pool and not coupled_has_unit_res:
                        placement[res] = _remap_units(orig, pool)
                    elif coupled_has_unit_res:
                        placement.pop(res, None)
        if coupled and aic_res and aiv_res and owners.pool_aic and owners.pool_aiv and not coupled_has_unit_res:
            aic_pool, aiv_pool = proportional_coupled_placement(owners, datapath, stream_id=stream_id)
            orig_aic = next((placement[r] for r in aic_res if r in placement), None)
            orig_aiv = next((placement[r] for r in aiv_res if r in placement), None)
            mapped_aic = _remap_units(orig_aic, owners.pool_aic) if orig_aic else aic_pool
            mapped_aiv = _remap_units(orig_aiv, owners.pool_aiv) if orig_aiv else aiv_pool
            aic_slice, aiv_slice = _proportional_pair(
                mapped_aic, mapped_aiv, datapath.aic_count, datapath.aiv_count, stream_id=stream_id
            )
            for res in aic_res:
                placement[res] = aic_slice
            for res in aiv_res:
                placement[res] = aiv_slice

    acquire = _remap_allocations(stage.acquire, owners, datapath, pool_names, pool_units)
    free = _remap_allocations(stage.free, owners, datapath, pool_names, pool_units)
    nested = (
        apply_stream_owners_to_program(stage.program, owners, datapath, **bind_kw)
        if stage.program is not None else None
    )
    if (
        work == dict(stage.work) and placement == dict(stage.placement)
        and caps == dict(stage.caps) and coupled == stage.coupled
        and acquire is stage.acquire and free is stage.free and nested is stage.program
    ):
        return stage
    return replace(stage, work=work, placement=placement, caps=caps, coupled=coupled, acquire=acquire, free=free, program=nested)


def _bind_pipeline_phase(phase: Any, owners: StreamOwners, datapath: DatapathCapabilitySpec, kwargs: dict[str, Any]) -> Any:
    stages = tuple(bind_stage_to_stream(s, owners, datapath, **kwargs) for s in phase.stages)
    residues = tuple(
        tuple(bind_stage_to_stream(s, owners, datapath, **kwargs) for s in group)
        for group in phase.residue_stages
    )
    return replace(phase, stages=stages, residue_stages=residues)


def apply_stream_owners_to_program(
    program: Any, owners: StreamOwners, datapath: DatapathCapabilitySpec, *,
    resource_names: Mapping[str, str], resource_units: Mapping[str, int],
    rates: Mapping[str, float], stream_id: StreamId = MAIN_STREAM,
    quota_resources: Mapping[str, str] | None = None,
    pool_names: Mapping[str, str] | None = None, pool_units: Mapping[str, int] | None = None,
) -> Any:
    """Recursively bind all stages of ``program`` and its wrappers to ``owners``."""
    from mindie_sim.prediction.program import (
        KernelProgram, NativeProgram, ParallelProgram,
        PipelineProgram, PipelineTemplateProgram,
    )

    kwargs = dict(
        resource_names=resource_names, resource_units=resource_units, rates=rates,
        stream_id=stream_id, quota_resources=quota_resources,
        pool_names=pool_names, pool_units=pool_units,
    )
    if isinstance(program, KernelProgram):
        return replace(program, stages=tuple(bind_stage_to_stream(s, owners, datapath, **kwargs) for s in program.stages))
    if isinstance(program, ParallelProgram):
        return replace(program, programs=tuple(apply_stream_owners_to_program(m, owners, datapath, **kwargs) for m in program.programs))
    if isinstance(program, PipelineProgram):
        orig_tile = program.tile
        return replace(program, tile=lambda idx: tuple(bind_stage_to_stream(s, owners, datapath, **kwargs) for s in orig_tile(idx)))
    if isinstance(program, PipelineTemplateProgram):
        tmpl = program.template
        bound_tmpl = replace(
            tmpl,
            setup=tuple(bind_stage_to_stream(s, owners, datapath, **kwargs) for s in tmpl.setup),
            drain=tuple(bind_stage_to_stream(s, owners, datapath, **kwargs) for s in tmpl.drain),
            phases=tuple(_bind_pipeline_phase(p, owners, datapath, kwargs) for p in tmpl.phases),
            motif_phases=tuple(_bind_pipeline_phase(p, owners, datapath, kwargs) for p in tmpl.motif_phases),
        )
        return replace(program, template=bound_tmpl)
    if isinstance(program, NativeProgram):
        inner = apply_stream_owners_to_program(program.program, owners, datapath, **kwargs)
        return program if inner is program.program else replace(program, program=inner)
    for attr in ("body", "program"):
        child = getattr(program, attr, None)
        if child is not None:
            inner = apply_stream_owners_to_program(child, owners, datapath, **kwargs)
            return program if inner is child else replace(program, **{attr: inner})
    return program


def _topological_order(n_nodes: int, edges: list[set[int]], error_code: str) -> list[int]:
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


def lower_stream_edges(ops: tuple[OpSpec, ...], plan: StreamPlan, /) -> tuple[OpSpec, ...]:
    """Bind N streams, verify O(|V|*|queues|) transitive FIFO+event reachability, and keep DAG order."""
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

    # Key FIFO queues by (op.stage, stream_id) matching execution.py (replica, stage, stream).
    queue_order: dict[tuple[int, StreamId], list[int]] = {}
    queue_id_of: list[int] = []
    pos_in_queue: list[int] = []
    queue_ids: dict[tuple[int, StreamId], int] = {}
    for i, op in enumerate(ops):
        qkey = (int(op.stage), stream_of[op.op_id])
        qid = queue_ids.get(qkey)
        if qid is None:
            qid = len(queue_ids)
            queue_ids[qkey] = qid
            queue_order[qkey] = []
        queue_id_of.append(qid)
        pos_in_queue.append(len(queue_order[qkey]))
        queue_order[qkey].append(i)

    seen_events: set[str] = set()
    event_waits_for: dict[str, list[str]] = {op_id: [] for op_id in op_ids}
    n_ops = len(op_ids)
    sync_edges: list[set[int]] = [set() for _ in range(n_ops)]

    for q_indices in queue_order.values():
        for prev_i, next_i in zip(q_indices, q_indices[1:]):
            sync_edges[prev_i].add(next_i)

    for event in plan.events:
        if event.event_id in seen_events:
            raise ModelingError(f"stream_duplicate_event:{event.event_id}")
        seen_events.add(event.event_id)
        if event.record_stream not in valid_lanes:
            raise ModelingError(f"stream_event_unknown_record_lane:{event.event_id}:{event.record_stream}")
        if event.wait_stream not in valid_lanes:
            raise ModelingError(f"stream_event_unknown_wait_lane:{event.event_id}:{event.wait_stream}")
        if event.record_after not in known:
            raise ModelingError(f"stream_unknown_event_record:{event.record_after}")
        if stream_of[event.record_after] != event.record_stream:
            raise ModelingError(f"stream_event_record_stream_mismatch:{event.event_id}:{event.record_after}")
        rec_i = index_of[event.record_after]
        for target in event.wait_before:
            if target not in known:
                raise ModelingError(f"stream_unknown_event_wait:{target}")
            if stream_of[target] != event.wait_stream:
                raise ModelingError(f"stream_event_wait_stream_mismatch:{event.event_id}:{target}")
            event_waits_for[target].append(event.record_after)
            sync_edges[rec_i].add(index_of[target])

    # Collect same-stage cross-stream data dependencies; fast-path skip vector clocks when none exist.
    cross_deps: list[tuple[int, int]] = []
    for i, op in enumerate(ops):
        c_stage, c_stream = int(op.stage), stream_of[op.op_id]
        for dep in op.deps:
            if dep not in known:
                continue
            d_idx = index_of[dep]
            if int(ops[d_idx].stage) == c_stage and stream_of[dep] != c_stream:
                cross_deps.append((d_idx, i))

    if cross_deps:
        sync_topo = _topological_order(n_ops, sync_edges, "stream_event_cycle")
        vclock: list[dict[int, int]] = [{queue_id_of[i]: pos_in_queue[i]} for i in range(n_ops)]
        for u in sync_topo:
            vc_u = vclock[u]
            for v in sync_edges[u]:
                vc_v = vclock[v]
                for qid, pos in vc_u.items():
                    if pos > vc_v.get(qid, -1):
                        vc_v[qid] = pos
        for d_idx, c_idx in cross_deps:
            if vclock[c_idx].get(queue_id_of[d_idx], -1) < pos_in_queue[d_idx]:
                raise ModelingError(
                    f"stream_cross_dependency_missing_event:{op_ids[d_idx]}->{op_ids[c_idx]}"
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
        merged_deps = tuple(dict.fromkeys((*op.deps, *extra_deps))) if extra_deps else op.deps
        lowered.append(replace(op, deps=merged_deps, stream=stream_of[op.op_id]))
    return tuple(lowered)


__all__ = [
    "AIC_BUFFER_POOLS", "AIC_ROLES", "AIV_BUFFER_POOLS", "AIV_ROLES",
    "MAIN_STREAM", "STREAM_QUOTA_PREFIX", "StreamCoreBudget", "StreamEvent",
    "StreamId", "StreamIsolation", "StreamLane", "StreamOwners", "StreamPlan",
    "apply_stream_owners_to_program", "bind_stage_to_stream", "bind_stream_owners",
    "check_work_against_stream_budget", "lower_stream_edges",
    "proportional_coupled_placement", "stream_quota_resources", "validate_stream_id",
    "validate_stream_plan_ranks",
]
