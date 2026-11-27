"""Fragment allocation planning.

[Role] Resolve allocation scopes, bindings, holds and releases before admission.
[Invariants] Concurrent stage and program allocations must fit each physical unit.
[Timeouts & Locks] Finite scan of fragment allocation declarations. No waits or locks.
[Side-Effect Policy] Builds an allocation plan in memory before submission.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

from mindie_sim.schema.operators import AllocationEffect, ProgramFragment

from ..engine import Allocation, Event
from .state import FragmentContractError

_CAPACITY_ROUNDING_ULPS = 4


@dataclass(frozen=True)
class _PlannedAllocation:
    """One acquire/release lifetime; release boundaries stay per allocation."""

    name: str
    allocation: Allocation
    lifetime: str
    external_hold: Event | None = None
    local_release: bool = False
    #: Original producer-owned buffer identity/binding (owner, storage,
    #: ranges, generation, event, rank, stage, host_bytes, unit). It stays on
    #: this allocation until its real free; it is never applied globally to a
    #: different K/V entry.
    memory: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _AllocationPlan:
    stage: tuple[_PlannedAllocation, ...] = ()
    program: tuple[_PlannedAllocation, ...] = ()


def _hashable_value(value: Any) -> Any:
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _normalize_half_open_ranges(
    raw_ranges: Any, size: float,
) -> tuple[tuple[float, float], ...]:
    """Normalize explicit half-open byte intervals using the Memory ledger convention."""
    if raw_ranges is None:
        return ((0.0, size),) if size > 0.0 else ()
    if isinstance(raw_ranges, (int, float)) and not isinstance(raw_ranges, bool):
        val = float(raw_ranges)
        return ((0.0, val),) if val > 0.0 else ()
    items = tuple(raw_ranges)
    if not items:
        return ((0.0, size),) if size > 0.0 else ()
    if all(isinstance(item, (int, float)) and not isinstance(item, bool)
           for item in items):
        nums = [float(item) for item in items]
        if len(nums) == 1:
            return ((0.0, nums[0]),) if nums[0] > 0.0 else ()
        if len(nums) % 2:
            return tuple((0.0, n) for n in nums if n > 0.0)
        return tuple(
            (nums[i], nums[i + 1])
            for i in range(0, len(nums), 2)
            if nums[i + 1] > nums[i]
        )
    return tuple(
        (float(lo), float(hi))
        for lo, hi in items
        if float(hi) > float(lo)
    )


def _union_interval_bytes(intervals: list[tuple[float, float]]) -> float:
    ordered = sorted((lo, hi) for lo, hi in intervals if hi > lo)
    merged: list[tuple[float, float]] = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return float(sum(hi - lo for lo, hi in merged))


def _memory_grant_spec(self, planned: _PlannedAllocation) -> dict[str, Any]:
    allocation = planned.allocation
    spec: dict[str, Any] = {
        "name": allocation.name,
        "bytes": float(allocation.size),
    }
    if planned.memory:
        spec.update(planned.memory)
    spec["stage"] = int(spec.get("stage", self.memory_stage))
    spec.setdefault("replica", self.memory_replica)
    return spec


def _check_memory_backed_capacity(
    self, memory_planned: list[_PlannedAllocation],
) -> None:
    """Validate concurrent memory-backed allocations by physical-range union."""
    grants = [_memory_grant_spec(self, item) for item in memory_planned]
    grant_decision = getattr(self.memory, "grant_decision", None)
    if callable(grant_decision):
        decision = grant_decision(
            self.memory_replica, grants, stage=self.memory_stage)
        if decision.get("ok"):
            return
        if decision.get("reason") == "invalid_or_duplicate":
            raise FragmentContractError(
                "fragment_allocation_memory_binding_invalid")
        if decision.get("permanent"):
            pool_name = memory_planned[0].allocation.pool
            shortfall = decision.get("shortfall_bytes_per_stage_rank") or ()
            raise FragmentContractError(
                f"fragment_allocation_concurrent_exceeds_capacity:"
                f"{pool_name}:memory:{shortfall}")
    # Standalone physical-range union catches permanent overflows even when
    # existing live holders alias one of the candidate's backings (or when a
    # custom routing ledger does not expose grant_decision).
    hbm_ranges: dict[tuple, list[tuple[float, float]]] = {}
    host_ranges: dict[tuple, list[tuple[float, float]]] = {}
    pool_by_stage_rank: dict[tuple[int, int], str] = {}
    for planned, spec in zip(memory_planned, grants):
        stage_idx = int(spec["stage"])
        rank_raw = spec.get("rank")
        rank_idx = 0 if rank_raw is None else int(rank_raw)
        storage = str(spec.get("storage") or "")
        tier = str(spec.get("tier") or "hbm").lower()
        backing_key = str(
            spec.get("alias_of")
            or spec.get("physical_backing_id")
            or spec.get("physical_key")
            or spec["name"]
        )
        generation = _hashable_value(spec.get("generation"))
        size = float(spec["bytes"])
        ranges = _normalize_half_open_ranges(spec.get("ranges"), size)
        host_bytes = float(spec.get("host_bytes", 0.0) or 0.0)
        host_key = (stage_idx, rank_idx, storage, backing_key, generation)
        if tier == "host":
            host_iv = ranges or (((0.0, host_bytes),) if host_bytes > 0.0 else ())
            if host_iv:
                host_ranges.setdefault(host_key, []).extend(host_iv)
            continue
        if tier not in {"hbm", ""}:
            continue
        hbm_key = (stage_idx, rank_idx, storage, tier, backing_key, generation)
        pool_by_stage_rank[(stage_idx, rank_idx)] = planned.allocation.pool
        if ranges:
            hbm_ranges.setdefault(hbm_key, []).extend(ranges)
        if host_bytes > 0.0:
            host_ranges.setdefault(host_key, []).append((0.0, host_bytes))

    hbm_by_stage_rank: dict[tuple[int, int], float] = {}
    for (stage_idx, rank_idx, *_), intervals in hbm_ranges.items():
        sr = (stage_idx, rank_idx)
        hbm_by_stage_rank[sr] = hbm_by_stage_rank.get(sr, 0.0) + _union_interval_bytes(intervals)

    rank_caps = getattr(self.memory, "rank_capacity", None)
    for (stage_idx, rank_idx), amount in hbm_by_stage_rank.items():
        pool_name = pool_by_stage_rank[(stage_idx, rank_idx)]
        if (rank_caps is not None
                and 0 <= self.memory_replica < len(rank_caps)
                and 0 <= stage_idx < len(rank_caps[self.memory_replica])
                and 0 <= rank_idx < len(rank_caps[self.memory_replica][stage_idx])):
            cap = float(rank_caps[self.memory_replica][stage_idx][rank_idx])
        else:
            cap = float(self.engine.pools[pool_name])
        if math.isfinite(cap) and amount > cap + _CAPACITY_ROUNDING_ULPS * math.ulp(cap):
            raise FragmentContractError(
                f"fragment_allocation_concurrent_exceeds_capacity:"
                f"{pool_name}:rank{rank_idx}:{amount}:{cap}")

    host_caps = getattr(self.memory, "host_dram_capacity", None)
    if host_ranges and host_caps is not None and 0 <= self.memory_replica < len(host_caps):
        host_cap = float(host_caps[self.memory_replica])
        host_total = sum(_union_interval_bytes(ivs) for ivs in host_ranges.values())
        if math.isfinite(host_cap) and host_total > host_cap + _CAPACITY_ROUNDING_ULPS * math.ulp(host_cap):
            raise FragmentContractError(
                f"fragment_allocation_concurrent_exceeds_capacity:"
                f"host:replica{self.memory_replica}:{host_total}:{host_cap}")


def _plan_allocations(
    self, fragment: ProgramFragment, conditions: Mapping[str, bool],
    holds: Mapping[str, Any], sequence: int,
) -> _AllocationPlan:
    acquires: dict[str, tuple[AllocationEffect, Allocation]] = {}
    releases: dict[str, tuple[AllocationEffect, Allocation]] = {}
    ready_holds = {str(name): self._as_event(event, f"hold:{name}")
                   for name, event in holds.items()}
    for effect in fragment.allocations:
        if not self._active(effect.condition, conditions, effect.name):
            continue
        pool = self._pool(effect.pool)
        if effect.size > self.engine.pools[pool]:
            raise FragmentContractError(
                f"fragment_allocation_exceeds_capacity:"
                f"{effect.name}:{effect.size}:{pool}")
        if any(unit >= self.engine.pool_units[pool] for unit in effect.units):
            raise FragmentContractError(
                f"fragment_allocation_unit_out_of_range:"
                f"{effect.name}:{pool}")
        allocation = Allocation(
            f"{self.prefix}{fragment.name}#{sequence}:{effect.name}", pool,
            effect.size, effect.units)
        target = acquires if effect.action == "acquire" else releases
        if effect.name in target:
            raise FragmentContractError(
                f"fragment_allocation_duplicate:{effect.name}")
        target[effect.name] = effect, allocation
    unknown_holds = set(ready_holds) - set(acquires)
    if unknown_holds:
        raise FragmentContractError(
            f"fragment_hold_unknown:{sorted(unknown_holds)[0]}")
    stage: list[_PlannedAllocation] = []
    program: list[_PlannedAllocation] = []
    for name, (effect, allocation) in acquires.items():
        release = releases.get(name)
        if release is not None:
            other = release[0]
            if (other.size != effect.size or other.units != effect.units
                    or other.pool != effect.pool
                    or other.lifetime != effect.lifetime):
                raise FragmentContractError(
                    f"fragment_allocation_release_mismatch:{name}")
            if name in ready_holds:
                raise FragmentContractError(
                    f"fragment_allocation_hold_conflict:{name}")
        elif name not in ready_holds:
            raise FragmentContractError(
                f"fragment_allocation_release_missing:{name}")
        planned = _PlannedAllocation(
            name=name, allocation=allocation, lifetime=effect.lifetime,
            external_hold=ready_holds.get(name),
            local_release=release is not None,
            memory=self._allocation_binding(name))
        if effect.lifetime in {"stage", "slot"}:
            stage.append(planned)
        elif effect.lifetime == "program":
            program.append(planned)
    missing_grant = set(releases) - set(acquires)
    if missing_grant:
        raise FragmentContractError(
            f"fragment_allocation_grant_missing:{sorted(missing_grant)[0]}")
    # Program allocations stay live across repeats while the current repeat
    # also holds its stage/slot allocations. If that concurrent set cannot
    # fit, the fragment can never be admitted; reject instead of stalling.
    concurrent: dict[tuple[str, int], float] = {}
    memory_planned: list[_PlannedAllocation] = []
    for planned in tuple(program) + tuple(stage):
        allocation = planned.allocation
        if self.memory is not None and allocation.pool in self.memory_pools:
            memory_planned.append(planned)
            continue
        for unit in allocation.units:
            key = allocation.pool, unit
            concurrent[key] = concurrent.get(key, 0.0) + allocation.size
    for (pool, unit), amount in concurrent.items():
        if amount > self.engine.pools[pool] + _CAPACITY_ROUNDING_ULPS * math.ulp(
                float(self.engine.pools[pool])):
            raise FragmentContractError(
                f"fragment_allocation_concurrent_exceeds_capacity:"
                f"{pool}:unit{unit}:{amount}:{self.engine.pools[pool]}")
    if memory_planned:
        _check_memory_backed_capacity(self, memory_planned)
    return _AllocationPlan(tuple(stage), tuple(program))
