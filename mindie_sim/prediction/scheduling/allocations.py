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
    for planned in tuple(program) + tuple(stage):
        allocation = planned.allocation
        for unit in allocation.units:
            key = allocation.pool, unit
            concurrent[key] = concurrent.get(key, 0.0) + allocation.size
    for (pool, unit), amount in concurrent.items():
        if amount > self.engine.pools[pool] + _CAPACITY_ROUNDING_ULPS * math.ulp(
                float(self.engine.pools[pool])):
            raise FragmentContractError(
                f"fragment_allocation_concurrent_exceeds_capacity:"
                f"{pool}:unit{unit}:{amount}:{self.engine.pools[pool]}")
    return _AllocationPlan(tuple(stage), tuple(program))
