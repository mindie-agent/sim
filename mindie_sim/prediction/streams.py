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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass, replace
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
    "vector", "mte2_aiv", "mte3", "vec_issue", "vec_alu", "vec_sfu", "vec_reduce",
    "reg_data_writes", "reg_ub_accesses", "reg_ub_stores", "reg_alu_lanes",
    "reg_special_lanes", "reg_reduce_lanes", "reg_shuffle_lanes",
    "scalar_store_issue",
})
AIC_BUFFER_POOLS: frozenset[str] = frozenset({
    "l1", "l0a", "l0b", "l0c", "bt", "fb", "bias_table", "fixpipe_buffer",
})
AIV_BUFFER_POOLS: frozenset[str] = frozenset({"ub"})
_AIC_OWNER_AXES: frozenset[str] = frozenset({"aic", "cube"}) | AIC_ROLES
_AIV_OWNER_AXES: frozenset[str] = frozenset({"aiv", "vector"}) | AIV_ROLES
_EXTRA_AIV_PIPES: frozenset[str] = frozenset({"ub_copy", "aiv_simt", "sort_topk"})
_PIPELINE_VALIDATE_WINDOW = 64
_HOMOGENEOUS_VALIDATE_TILES = 2


def _role_executors_for_datapath(
    datapath: DatapathCapabilitySpec | None,
) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """Derive ``(aic_only, aiv_only, dual_executor)`` resource roles from ``datapath``."""
    aic = set(AIC_ROLES)
    aiv = set(AIV_ROLES) | _EXTRA_AIV_PIPES
    dual: set[str] = set()
    if datapath is None:
        return frozenset(aic), frozenset(aiv), frozenset()
    declared = getattr(getattr(datapath, "compute_arch", None), "declared", None)
    for svc in getattr(declared, "instruction_services", ()):
        axis = str(getattr(svc, "owner_axis", ""))
        names = (svc.name, svc.resource) if svc.resource else (svc.name,)
        if axis in _AIC_OWNER_AXES or axis in aic:
            aic.update(name for name in names if name and name not in aiv)
        elif axis in _AIV_OWNER_AXES or axis in aiv:
            aiv.update(name for name in names if name and name not in aic)
    executors_by_unit: dict[str, set[str]] = {}
    for path in getattr(datapath, "usable_paths", ()):
        unit = getattr(path, "unit", "")
        executor = getattr(path, "executor", "")
        if unit and executor in {"aic", "aiv"}:
            executors_by_unit.setdefault(unit, set()).add(executor)
    for unit, executors in executors_by_unit.items():
        if executors == {"aic"}:
            aic.add(unit)
            aiv.discard(unit)
            dual.discard(unit)
        elif executors == {"aiv"}:
            aiv.add(unit)
            aic.discard(unit)
            dual.discard(unit)
        elif executors == {"aic", "aiv"}:
            aic.discard(unit)
            aiv.discard(unit)
            dual.add(unit)
    return frozenset(aic), frozenset(aiv), frozenset(dual)


def owner_roles_for_datapath(
    datapath: DatapathCapabilitySpec | None,
) -> tuple[frozenset[str], frozenset[str]]:
    """Derive single-executor AIC and AIV resource roles from ``datapath`` declarations."""
    aic, aiv, _ = _role_executors_for_datapath(datapath)
    return aic, aiv


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

    def units_for(self, axis: Literal["aic", "aiv"]) -> tuple[int, ...]:
        """Return the physical unit pool bound to ``axis`` for this lane."""
        if axis == "aic":
            return self.aic if self.isolation == "disjoint" else self.pool_aic
        return self.aiv if self.isolation == "disjoint" else self.pool_aiv

    def pool_for_role(
        self,
        role: str,
        *,
        datapath: DatapathCapabilitySpec | None = None,
        buffer_pool: bool = False,
    ) -> tuple[int, ...]:
        """Return the lane's AIC or AIV core pool for a resource role or SRAM pool."""
        if buffer_pool:
            aic_set, aiv_set, dual_set = AIC_BUFFER_POOLS, AIV_BUFFER_POOLS, frozenset()
        else:
            aic_set, aiv_set, dual_set = _role_executors_for_datapath(datapath)
        if role in aic_set:
            return self.units_for("aic")
        if role in aiv_set:
            return self.units_for("aiv")
        if role in dual_set:
            return self.units_for("aiv") or self.units_for("aic")
        return ()

    def paired_owners(
        self,
        datapath: DatapathCapabilitySpec,
        *,
        require_full_pair: bool = False,
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """Resolve physical ``(aic_id, paired_aivs)`` tuples admitted by this lane."""
        if self.isolation == "disjoint":
            aic_base, aiv_base = self.aic, self.aiv
            aic_rem, aiv_rem = len(aic_base), len(aiv_base)
        else:
            aic_base = self.pool_aic or tuple(range(datapath.aic_count))
            aiv_base = self.pool_aiv or tuple(range(datapath.aiv_count))
            aic_rem, aiv_rem = self.aic_limit, self.aiv_limit
        if aic_rem <= 0 or aiv_rem <= 0 or not aic_base or not aiv_base:
            return ()
        if datapath.aic_aiv_pairs:
            aic_set, aiv_set = set(aic_base), set(aiv_base)
            pairs: list[tuple[int, tuple[int, ...]]] = []
            for aic_id, paired_aivs in datapath.aic_aiv_pairs:
                if aic_rem <= 0 or aiv_rem <= 0:
                    break
                if aic_id not in aic_set:
                    continue
                matched = tuple(v for v in paired_aivs if v in aiv_set)
                if not matched:
                    continue
                if require_full_pair:
                    if len(matched) != len(paired_aivs) or len(matched) > aiv_rem:
                        continue
                    chosen = matched
                else:
                    chosen = matched[:aiv_rem]
                pairs.append((aic_id, chosen))
                aic_rem -= 1
                aiv_rem -= len(chosen)
            return tuple(pairs)
        aic_ids = aic_base[:aic_rem]
        aiv_ids = aiv_base[:aiv_rem]
        if len(aiv_ids) >= len(aic_ids) and len(aiv_ids) % len(aic_ids) == 0:
            ratio = len(aiv_ids) // len(aic_ids)
            return tuple(
                (aic_id, aiv_ids[i * ratio : (i + 1) * ratio])
                for i, aic_id in enumerate(aic_ids)
            )
        if require_full_pair:
            return ()
        return tuple((aic_id, (aiv_ids[i % len(aiv_ids)],)) for i, aic_id in enumerate(aic_ids))


def resolve_aic_aiv_pairs(
    datapath: DatapathCapabilitySpec,
    owners: StreamOwners | None = None,
    *,
    require_full_pair: bool = False,
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    """Return physical ``(aic_id, paired_aivs)`` groups from ``datapath.aic_aiv_pairs``."""
    if owners is not None:
        return owners.paired_owners(datapath, require_full_pair=require_full_pair)
    return datapath.aic_aiv_pairs


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
    aic_roles, aiv_roles, dual_roles = _role_executors_for_datapath(datapath)
    for sid, owners in owners_by_stream.items():
        if owners.isolation != "shared_cap":
            continue
        lane_aic_roles = aic_roles | (dual_roles if owners.aiv_limit == 0 else frozenset())
        lane_aiv_roles = aiv_roles | (dual_roles if owners.aiv_limit > 0 else frozenset())
        for role_set, limit, total, pool in (
            (lane_aic_roles, owners.aic_limit, datapath.aic_count, owners.pool_aic),
            (lane_aiv_roles, owners.aiv_limit, datapath.aiv_count, owners.pool_aiv),
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
    pool_len = len(pool)
    pool_set = set(pool)
    slot_map: dict[int, int] = {}
    pending: list[int] = []
    for unit in pool:
        if 0 <= unit < pool_len and unit not in slot_map:
            slot_map[unit] = unit
        else:
            pending.append(unit)
    leftovers: list[int] = []
    for unit in pending:
        rem = unit % pool_len
        if rem not in slot_map:
            slot_map[rem] = unit
        else:
            leftovers.append(unit)
    if leftovers:
        empty_slots = (idx for idx in range(pool_len) if idx not in slot_map)
        for idx, unit in zip(empty_slots, leftovers):
            slot_map[idx] = unit
    seen: list[int] = []
    for unit in existing:
        mapped = unit if unit in pool_set else slot_map[unit % pool_len]
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


def bind_allocations_to_stream(
    allocs: tuple[Any, ...],
    owners: StreamOwners,
    datapath: DatapathCapabilitySpec,
    *,
    pool_names: Mapping[str, str] | None = None,
    pool_units: Mapping[str, int] | None = None,
    stream_id: StreamId = MAIN_STREAM,
    label: str = "stage",
) -> tuple[Any, ...]:
    """Bind private per-owner buffer grants/releases onto the lane's physical core pool."""
    if not allocs:
        return allocs
    logical_by_phys = {phys: name for name, phys in (pool_names or {}).items()}
    punits = pool_units or {}
    aic_pool = owners.units_for("aic")
    aiv_pool = owners.units_for("aiv")
    remapped, changed = [], False
    for alloc in allocs:
        logical = logical_by_phys.get(alloc.pool, alloc.pool)
        target_pool: tuple[int, ...] | None = None
        if logical in AIC_BUFFER_POOLS:
            if owners.aic_limit == 0 and getattr(alloc, "size", 0) > 0:
                raise ModelingError(f"stream_zero_aic_budget:{stream_id}:{label}")
            unit_count = int(punits.get(logical, punits.get(alloc.pool, datapath.aic_count)))
            if aic_pool and unit_count == datapath.aic_count:
                target_pool = aic_pool
        elif logical in AIV_BUFFER_POOLS:
            if owners.aiv_limit == 0 and getattr(alloc, "size", 0) > 0:
                raise ModelingError(f"stream_zero_aiv_budget:{stream_id}:{label}")
            unit_count = int(punits.get(logical, punits.get(alloc.pool, datapath.aiv_count)))
            if aiv_pool and unit_count == datapath.aiv_count:
                target_pool = aiv_pool
        if target_pool is not None:
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


def _dual_role_axis(
    role: str,
    res: str,
    owners: StreamOwners | None,
    datapath: DatapathCapabilitySpec | None,
    resource_units: Mapping[str, int],
) -> Literal["aic", "aiv"]:
    if owners is not None:
        if owners.aiv_limit == 0 and owners.aic_limit > 0:
            return "aic"
        if owners.aic_limit == 0 and owners.aiv_limit > 0:
            return "aiv"
    unit_count = resource_units.get(role, resource_units.get(res))
    if (
        datapath is not None
        and unit_count is not None
        and int(unit_count) == datapath.aic_count
        and datapath.aic_count != datapath.aiv_count
    ):
        return "aic"
    return "aiv"


def check_work_against_stream_budget(
    work: Mapping[str, float], owners: StreamOwners, *,
    resource_names: Mapping[str, str], stream_id: StreamId = MAIN_STREAM, label: str = "op",
    datapath: DatapathCapabilitySpec | None = None,
) -> None:
    """Reject AIC or AIV work placed on a stream whose core budget for that engine is 0."""
    aic_roles, aiv_roles, dual_roles = _role_executors_for_datapath(datapath)
    role_by_res = {phys: role for role, phys in resource_names.items()}
    if owners.aic_limit == 0 and any(amt > 0.0 and role_by_res.get(res, res) in aic_roles for res, amt in work.items()):
        raise ModelingError(f"stream_zero_aic_budget:{stream_id}:{label}")
    if owners.aiv_limit == 0 and any(amt > 0.0 and role_by_res.get(res, res) in aiv_roles for res, amt in work.items()):
        raise ModelingError(f"stream_zero_aiv_budget:{stream_id}:{label}")
    if (
        owners.aic_limit == 0 and owners.aiv_limit == 0
        and any(amt > 0.0 and role_by_res.get(res, res) in dual_roles for res, amt in work.items())
    ):
        raise ModelingError(f"stream_zero_aiv_budget:{stream_id}:{label}")


def _bind_disjoint_stage(
    aic_res: list[str], aiv_res: list[str],
    placement: dict[str, tuple[int, ...]],
    owners: StreamOwners, datapath: DatapathCapabilitySpec, *,
    stream_id: StreamId, coupled: bool,
) -> dict[str, tuple[int, ...]]:
    pool_by_res: dict[str, tuple[int, ...]] = {}
    if coupled and aic_res and aiv_res:
        aic_pool, aiv_pool = proportional_coupled_placement(owners, datapath, stream_id=stream_id)
        orig_aic = next((placement[r] for r in aic_res if r in placement), None)
        orig_aiv = next((placement[r] for r in aiv_res if r in placement), None)
        mapped_aic = _remap_units(orig_aic, owners.aic) if orig_aic else aic_pool
        mapped_aiv = _remap_units(orig_aiv, owners.aiv) if orig_aiv else aiv_pool
        aic_slice, aiv_slice = _proportional_pair(
            mapped_aic, mapped_aiv, datapath.aic_count, datapath.aiv_count, stream_id=stream_id,
        )
        for res in aic_res:
            placement[res] = aic_slice
            pool_by_res[res] = owners.aic
        for res in aiv_res:
            placement[res] = aiv_slice
            pool_by_res[res] = owners.aiv
        return pool_by_res
    for res_list, pool in ((aic_res, owners.aic), (aiv_res, owners.aiv)):
        for res in res_list:
            placement[res] = _remap_units(placement.get(res), pool)
            pool_by_res[res] = pool
    return pool_by_res


def _rebind_cohort_metadata(
    stage: Any,
    placement: Mapping[str, tuple[int, ...]],
    pool_by_res: Mapping[str, tuple[int, ...]],
) -> tuple[tuple[str, ...], dict[str, tuple[tuple[int, ...], ...]]]:
    widened = list(stage.cohort_widened)
    members = dict(stage.cohort_members)
    if not widened and not members:
        return (), {}
    cohort_size = int(getattr(stage, "cohort_size", 1))
    for res, rows in list(members.items()):
        if res not in placement:
            members.pop(res, None)
            if res in widened:
                widened.remove(res)
            continue
        target_units = placement[res]
        if target_units == stage.placement.get(res):
            continue
        if len(set(rows)) <= 1:
            members[res] = (target_units,) * cohort_size
            continue
        pool = pool_by_res.get(res, target_units)
        remapped = tuple(_remap_units(tuple(row), pool) for row in rows)
        flat = tuple(u for row in remapped for u in row)
        if (
            len({len(r) for r in remapped}) == 1
            and len(set(flat)) == len(flat)
            and set(flat) == set(target_units)
        ):
            members[res] = remapped
        elif len(target_units) >= cohort_size and len(target_units) % cohort_size == 0:
            stride = len(target_units) // cohort_size
            members[res] = tuple(
                target_units[i * stride : (i + 1) * stride] for i in range(cohort_size)
            )
        else:
            members[res] = (target_units,) * cohort_size
            if res in widened:
                widened.remove(res)
    kept_widened = tuple(
        res for res in widened
        if res in placement and len(placement[res]) >= cohort_size
        and len(placement[res]) % cohort_size == 0
    )
    return kept_widened, members


def _split_shared_uncoupled_stage(
    stage: Any, aic_res: list[str], aiv_res: list[str],
    placement: dict[str, tuple[int, ...]], caps: dict[str, float],
    owners: StreamOwners, datapath: DatapathCapabilitySpec, *,
    rates: Mapping[str, float], qmap: Mapping[str, str],
    bind_kw: dict[str, Any], pool_names: Mapping[str, str] | None,
    pool_units: Mapping[str, int] | None, stream_id: StreamId,
) -> Any:
    from mindie_sim.prediction.program.stages import KernelProgram, ParallelProgram, Stage

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
    acq = bind_allocations_to_stream(stage.acquire, owners, datapath, pool_names=pool_names,
                                     pool_units=pool_units, stream_id=stream_id, label=stage.name)
    rel = bind_allocations_to_stream(stage.free, owners, datapath, pool_names=pool_names,
                                     pool_units=pool_units, stream_id=stream_id, label=stage.name)
    return replace(
        stage, work={}, placement={}, caps={}, acquire=acq, free=rel,
        cohort_widened=(), cohort_members={}, program=final_prog,
    )


def _bind_shared_stage(
    stage: Any, aic_res: list[str], aiv_res: list[str],
    work: dict[str, float], placement: dict[str, tuple[int, ...]],
    caps: dict[str, float], owners: StreamOwners, datapath: DatapathCapabilitySpec, *,
    rates: Mapping[str, float], qmap: Mapping[str, str],
    stream_id: StreamId, coupled: bool,
) -> tuple[bool, dict[str, tuple[int, ...]]]:
    pool_by_res: dict[str, tuple[int, ...]] = {}
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
                if pool:
                    placement[res] = _remap_units(orig, pool)
                    pool_by_res[res] = pool
                else:
                    placement.pop(res, None)
            else:
                eff_lim = min(len(orig), limit) if orig else limit
                _cap_rate(caps, res, rates, eff_lim, total)
                if pool:
                    placement[res] = _remap_units(orig, pool)
                    pool_by_res[res] = pool
    if coupled and aic_res and aiv_res and owners.pool_aic and owners.pool_aiv:
        aic_pool, aiv_pool = proportional_coupled_placement(owners, datapath, stream_id=stream_id)
        orig_aic = next((placement[r] for r in aic_res if r in placement), None)
        orig_aiv = next((placement[r] for r in aiv_res if r in placement), None)
        mapped_aic = _remap_units(orig_aic, owners.pool_aic) if orig_aic else aic_pool
        mapped_aiv = _remap_units(orig_aiv, owners.pool_aiv) if orig_aiv else aiv_pool
        aic_slice, aiv_slice = _proportional_pair(
            mapped_aic, mapped_aiv, datapath.aic_count, datapath.aiv_count, stream_id=stream_id,
        )
        for res in aic_res:
            placement[res] = aic_slice
            pool_by_res[res] = owners.pool_aic
        for res in aiv_res:
            placement[res] = aiv_slice
            pool_by_res[res] = owners.pool_aiv
    return coupled, pool_by_res


def bind_stage_to_stream(
    stage: Any, owners: StreamOwners, datapath: DatapathCapabilitySpec, *,
    resource_names: Mapping[str, str], resource_units: Mapping[str, int],
    rates: Mapping[str, float], stream_id: StreamId = MAIN_STREAM,
    quota_resources: Mapping[str, str] | None = None,
    pool_names: Mapping[str, str] | None = None, pool_units: Mapping[str, int] | None = None,
) -> Any:
    """Bind one ``Stage`` to its lane's physical core owners and aggregate lane quotas."""
    aic_roles, aiv_roles, dual_roles = _role_executors_for_datapath(datapath)
    role_by_res = {phys: role for role, phys in resource_names.items()}
    aic_res: list[str] = []
    aiv_res: list[str] = []
    for res, amt in stage.work.items():
        if amt <= 0.0:
            continue
        role = role_by_res.get(res, res)
        if role in aic_roles:
            aic_res.append(res)
        elif role in aiv_roles:
            aiv_res.append(res)
        elif role in dual_roles:
            axis = _dual_role_axis(role, res, owners, datapath, resource_units)
            (aic_res if axis == "aic" else aiv_res).append(res)
    if aic_res and owners.aic_limit == 0:
        raise ModelingError(f"stream_zero_aic_budget:{stream_id}:{stage.name}")
    if aiv_res and owners.aiv_limit == 0:
        raise ModelingError(f"stream_zero_aiv_budget:{stream_id}:{stage.name}")
    work, placement, caps, coupled = dict(stage.work), dict(stage.placement), dict(stage.caps), bool(stage.coupled)
    bind_kw = dict(
        resource_names=resource_names, resource_units=resource_units, rates=rates,
        stream_id=stream_id, quota_resources=quota_resources,
        pool_names=pool_names, pool_units=pool_units,
    )

    if owners.isolation == "disjoint":
        pool_by_res = _bind_disjoint_stage(
            aic_res, aiv_res, placement, owners, datapath,
            stream_id=stream_id, coupled=coupled,
        )
    else:
        qmap = quota_resources or {}
        if not coupled and len(stage.work) > 1 and any(res in qmap for res in (*aic_res, *aiv_res)):
            return _split_shared_uncoupled_stage(
                stage, aic_res, aiv_res, placement, caps, owners, datapath,
                rates=rates, qmap=qmap, bind_kw=bind_kw,
                pool_names=pool_names, pool_units=pool_units, stream_id=stream_id,
            )
        coupled, pool_by_res = _bind_shared_stage(
            stage, aic_res, aiv_res, work, placement, caps, owners, datapath,
            rates=rates, qmap=qmap, stream_id=stream_id, coupled=coupled,
        )

    cohort_widened, cohort_members = _rebind_cohort_metadata(stage, placement, pool_by_res)
    acquire = bind_allocations_to_stream(stage.acquire, owners, datapath, pool_names=pool_names,
                                         pool_units=pool_units, stream_id=stream_id, label=stage.name)
    free = bind_allocations_to_stream(stage.free, owners, datapath, pool_names=pool_names,
                                      pool_units=pool_units, stream_id=stream_id, label=stage.name)
    nested = (
        apply_stream_owners_to_program(stage.program, owners, datapath, **bind_kw)
        if stage.program is not None else None
    )
    if (
        work == dict(stage.work) and placement == dict(stage.placement)
        and caps == dict(stage.caps) and coupled == stage.coupled
        and cohort_widened == stage.cohort_widened
        and cohort_members == dict(stage.cohort_members)
        and acquire is stage.acquire and free is stage.free and nested is stage.program
    ):
        return stage
    return replace(
        stage, work=work, placement=placement, caps=caps, coupled=coupled,
        cohort_widened=cohort_widened, cohort_members=cohort_members,
        acquire=acquire, free=free, program=nested,
    )


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
    from mindie_sim.prediction.program.pipeline import (
        PipelineProgram, PipelineTemplateProgram,
    )
    from mindie_sim.prediction.program.stages import (
        KernelProgram, NativeProgram, ParallelProgram, Stage,
    )

    kwargs = dict(
        resource_names=resource_names, resource_units=resource_units, rates=rates,
        stream_id=stream_id, quota_resources=quota_resources,
        pool_names=pool_names, pool_units=pool_units,
    )
    binder = getattr(program, "bind_stream_owners", None)
    if callable(binder):
        return binder(owners, datapath, **kwargs)
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
    if is_dataclass(program):
        for attr in ("body", "program"):
            child = getattr(program, attr, None)
            if child is not None:
                inner = apply_stream_owners_to_program(child, owners, datapath, **kwargs)
                return program if inner is child else replace(program, **{attr: inner})
        raw_stages = getattr(program, "stages", None)
        if isinstance(raw_stages, tuple) and raw_stages and all(isinstance(s, Stage) for s in raw_stages):
            bound_stages = tuple(bind_stage_to_stream(s, owners, datapath, **kwargs) for s in raw_stages)
            return replace(program, stages=bound_stages)
        raw_progs = getattr(program, "programs", None)
        if isinstance(raw_progs, tuple) and raw_progs:
            bound_progs = tuple(apply_stream_owners_to_program(m, owners, datapath, **kwargs) for m in raw_progs)
            return replace(program, programs=bound_progs)
    return program


def _validate_stage_allocations(
    stage: Any, owners: StreamOwners, datapath: DatapathCapabilitySpec | None, *,
    pool_names: Mapping[str, str] | None, pool_units: Mapping[str, int] | None,
    stream_id: StreamId,
) -> None:
    if not (stage.acquire or stage.free):
        return
    logical_by_phys = {phys: name for name, phys in (pool_names or {}).items()}
    punits = pool_units or {}
    aic_count = datapath.aic_count if datapath is not None else 0
    aiv_count = datapath.aiv_count if datapath is not None else 0
    for alloc in (*stage.acquire, *stage.free):
        logical = logical_by_phys.get(alloc.pool, alloc.pool)
        if logical in AIC_BUFFER_POOLS:
            if owners.aic_limit == 0 and getattr(alloc, "size", 0) > 0:
                raise ModelingError(f"stream_zero_aic_budget:{stream_id}:{stage.name}")
            allowed = owners.units_for("aic")
            expected = aic_count
        elif logical in AIV_BUFFER_POOLS:
            if owners.aiv_limit == 0 and getattr(alloc, "size", 0) > 0:
                raise ModelingError(f"stream_zero_aiv_budget:{stream_id}:{stage.name}")
            allowed = owners.units_for("aiv")
            expected = aiv_count
        else:
            continue
        unit_count = int(punits.get(logical, punits.get(alloc.pool, expected)))
        if allowed and expected > 0 and unit_count == expected:
            if not alloc.units or not set(alloc.units).issubset(allowed):
                raise ModelingError(
                    f"stage_allocation_outside_stream_owners:{stream_id}:{stage.name}:{logical}:{alloc.units}"
                )


def _validate_stage_cohort(
    stage: Any,
    allowed_by_res: Mapping[str, tuple[int, ...]],
    *,
    stream_id: StreamId,
) -> None:
    cohort_members = getattr(stage, "cohort_members", None) or {}
    cohort_widened = getattr(stage, "cohort_widened", ()) or ()
    for res in cohort_widened:
        if res not in stage.placement:
            raise ModelingError(f"stage_cohort_widened_missing_placement:{stream_id}:{stage.name}:{res}")
    for res, rows in cohort_members.items():
        placed = stage.placement.get(res)
        if not placed:
            raise ModelingError(f"stage_cohort_members_missing_placement:{stream_id}:{stage.name}:{res}")
        placed_set = set(placed)
        allowed = allowed_by_res.get(res, ())
        allowed_set = set(allowed) if allowed else None
        flat = tuple(u for row in rows for u in row)
        if not set(flat).issubset(placed_set):
            raise ModelingError(f"stage_cohort_members_outside_placement:{stream_id}:{stage.name}:{res}")
        if allowed_set is not None and not set(flat).issubset(allowed_set):
            raise ModelingError(f"stage_cohort_members_outside_stream_owners:{stream_id}:{stage.name}:{res}")
        if len(set(rows)) > 1 or res in cohort_widened:
            if len({len(r) for r in rows}) != 1 or len(set(flat)) != len(flat) or set(flat) != placed_set:
                raise ModelingError(f"stage_cohort_members_not_disjoint_union:{stream_id}:{stage.name}:{res}")


def validate_stage_owners(
    stage: Any,
    datapath: DatapathCapabilitySpec | None,
    *,
    resource_names: Mapping[str, str],
    resource_units: Mapping[str, int] = (),  # type: ignore[assignment]
    owners: StreamOwners | None = None,
    stream_id: StreamId = MAIN_STREAM,
    pool_names: Mapping[str, str] | None = None,
    pool_units: Mapping[str, int] | None = None,
    require_placement: bool = True,
) -> None:
    """Validate post-binding per-owner placements and buffer units on one ``Stage``.

    Global/fabric/control stages (HBM, L2, links, host, sync, zero-work) remain
    valid with empty ``placement``.
    """
    aic_roles, aiv_roles, dual_roles = _role_executors_for_datapath(datapath)
    role_by_res = {phys: role for role, phys in resource_names.items()}
    runits = dict(resource_units) if resource_units else {}
    has_quota = any(STREAM_QUOTA_PREFIX in res and amt > 0.0 for res, amt in stage.work.items())
    allowed_by_res: dict[str, tuple[int, ...]] = {}
    for res, amt in stage.work.items():
        if amt <= 0.0 or STREAM_QUOTA_PREFIX in res:
            continue
        role = role_by_res.get(res, res)
        if role in aic_roles:
            axis: Literal["aic", "aiv"] = "aic"
        elif role in aiv_roles:
            axis = "aiv"
        elif role in dual_roles:
            axis = _dual_role_axis(role, res, owners, datapath, runits)
        else:
            continue
        total = (
            (datapath.aic_count if axis == "aic" else datapath.aiv_count)
            if datapath is not None else int(runits.get(role, runits.get(res, 0)))
        )
        limit = (owners.aic_limit if axis == "aic" else owners.aiv_limit) if owners is not None else total
        allowed = owners.units_for(axis) if owners is not None else ()
        allowed_by_res[res] = allowed
        if owners is not None and limit == 0:
            raise ModelingError(f"stream_zero_{axis}_budget:{stream_id}:{stage.name}")
        placed = stage.placement.get(res)
        if not placed:
            if owners is not None and owners.isolation == "shared_cap" and not allowed:
                if res in stage.caps or has_quota or (not require_placement and limit >= total):
                    continue
            if owners is None and not require_placement:
                continue
            raise ModelingError(f"stage_missing_per_owner_placement:{stage.name}:{role}")
        if allowed:
            if not set(placed).issubset(allowed):
                raise ModelingError(
                    f"stage_placement_outside_stream_owners:{stream_id}:{stage.name}:{role}:{placed}"
                )
        elif total > 0 and any(u < 0 or u >= total for u in placed):
            raise ModelingError(
                f"stage_placement_owner_out_of_range:{stage.name}:{role}:{placed}"
            )
    _validate_stage_cohort(stage, allowed_by_res, stream_id=stream_id)
    if owners is not None:
        _validate_stage_allocations(
            stage, owners, datapath,
            pool_names=pool_names, pool_units=pool_units, stream_id=stream_id,
        )
    if stage.program is not None:
        validate_program_owners(
            stage.program, datapath,
            resource_names=resource_names, resource_units=runits, owners=owners,
            stream_id=stream_id, pool_names=pool_names, pool_units=pool_units,
            require_placement=require_placement,
        )


def _ordered_stages(stages: Sequence[Any]) -> tuple[Any, ...]:
    if len(stages) <= 1:
        return tuple(stages)
    by_name = {stage.name: idx for idx, stage in enumerate(stages)}
    edges: list[set[int]] = [set() for _ in stages]
    has_edges = False
    for idx, stage in enumerate(stages):
        for dep in getattr(stage, "after", ()):
            pred = by_name.get(dep)
            if pred is not None and pred != idx:
                edges[pred].add(idx)
                has_edges = True
    if not has_edges:
        return tuple(stages)
    order = _topological_order(len(stages), edges, "stage_dependency_cycle")
    return tuple(stages[idx] for idx in order)


def _validate_stage_sequence_grants(
    stages: Sequence[Any],
    *,
    stream_id: StreamId,
    live: Mapping[str, tuple[str, frozenset[int]]] | None = None,
) -> dict[str, tuple[str, frozenset[int]]]:
    active = {} if live is None else dict(live)
    for stage in _ordered_stages(stages):
        for alloc in stage.acquire:
            active[alloc.name] = (alloc.pool, frozenset(alloc.units))
        for alloc in stage.free:
            grant = active.get(alloc.name)
            if grant is not None:
                if (alloc.pool, frozenset(alloc.units)) != grant:
                    raise ModelingError(
                        f"stage_allocation_release_owner_mismatch:{stream_id}:{stage.name}:{alloc.name}"
                    )
                active.pop(alloc.name, None)
    return active


def _validate_pipeline_program(
    program: Any,
    datapath: DatapathCapabilitySpec | None,
    *,
    stream_id: StreamId,
    vkw: Mapping[str, Any],
) -> None:
    count = int(program.count)
    if count <= 0:
        return
    prefix = (
        min(count, _HOMOGENEOUS_VALIDATE_TILES)
        if program.homogeneous else min(count, _PIPELINE_VALIDATE_WINDOW)
    )
    live: dict[str, tuple[str, frozenset[int]]] = {}
    for idx in range(prefix):
        tile_stages = tuple(program.tile(idx))
        live = _validate_stage_sequence_grants(tile_stages, stream_id=stream_id, live=live)
        for stage in tile_stages:
            validate_stage_owners(stage, datapath, **vkw)
    if count > prefix:
        tail_stages = tuple(program.tile(count - 1))
        _validate_stage_sequence_grants(tail_stages, stream_id=stream_id)
        for stage in tail_stages:
            validate_stage_owners(stage, datapath, **vkw)


def validate_program_owners(
    program: Any,
    datapath: DatapathCapabilitySpec | None,
    *,
    resource_names: Mapping[str, str],
    resource_units: Mapping[str, int] = (),  # type: ignore[assignment]
    owners: StreamOwners | None = None,
    stream_id: StreamId = MAIN_STREAM,
    pool_names: Mapping[str, str] | None = None,
    pool_units: Mapping[str, int] | None = None,
    require_placement: bool = True,
) -> None:
    """Recursively validate post-binding per-owner placements and grant/release parity."""
    from mindie_sim.prediction.program.pipeline import (
        PipelineProgram, PipelineTemplateProgram,
    )
    from mindie_sim.prediction.program.stages import (
        KernelProgram, NativeProgram, ParallelProgram, Stage,
    )

    vkw = dict(
        resource_names=resource_names, resource_units=resource_units, owners=owners,
        stream_id=stream_id, pool_names=pool_names, pool_units=pool_units,
        require_placement=require_placement,
    )
    if isinstance(program, KernelProgram):
        _validate_stage_sequence_grants(program.stages, stream_id=stream_id)
        for stage in program.stages:
            validate_stage_owners(stage, datapath, **vkw)
        return
    if isinstance(program, ParallelProgram):
        for child in program.programs:
            validate_program_owners(child, datapath, **vkw)
        return
    if isinstance(program, PipelineProgram):
        _validate_pipeline_program(program, datapath, stream_id=stream_id, vkw=vkw)
        return
    if isinstance(program, PipelineTemplateProgram):
        tmpl = program.template
        live = _validate_stage_sequence_grants(tmpl.setup, stream_id=stream_id)
        for stage in tmpl.setup:
            validate_stage_owners(stage, datapath, **vkw)
        for phase in (*tmpl.phases, *tmpl.motif_phases):
            phase_live = _validate_stage_sequence_grants(phase.stages, stream_id=stream_id, live=live)
            for stage in phase.stages:
                validate_stage_owners(stage, datapath, **vkw)
            for group in phase.residue_stages:
                _validate_stage_sequence_grants(group, stream_id=stream_id, live=live)
                for stage in group:
                    validate_stage_owners(stage, datapath, **vkw)
            live = phase_live
        _validate_stage_sequence_grants(tmpl.drain, stream_id=stream_id, live=live)
        for stage in tmpl.drain:
            validate_stage_owners(stage, datapath, **vkw)
        return
    if isinstance(program, NativeProgram):
        validate_program_owners(program.program, datapath, **vkw)
        return
    for attr in ("body", "program"):
        child = getattr(program, attr, None)
        if child is not None:
            validate_program_owners(child, datapath, **vkw)
            return
    raw_stages = getattr(program, "stages", None)
    if isinstance(raw_stages, tuple) and raw_stages and all(isinstance(s, Stage) for s in raw_stages):
        _validate_stage_sequence_grants(raw_stages, stream_id=stream_id)
        for stage in raw_stages:
            validate_stage_owners(stage, datapath, **vkw)
        return
    raw_progs = getattr(program, "programs", None)
    if isinstance(raw_progs, tuple) and raw_progs:
        for child in raw_progs:
            validate_program_owners(child, datapath, **vkw)


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
    "apply_stream_owners_to_program", "bind_allocations_to_stream",
    "bind_stage_to_stream", "bind_stream_owners",
    "check_work_against_stream_budget", "lower_stream_edges",
    "owner_roles_for_datapath", "proportional_coupled_placement",
    "resolve_aic_aiv_pairs", "stream_quota_resources", "validate_program_owners",
    "validate_stage_owners", "validate_stream_id", "validate_stream_plan_ranks",
]
