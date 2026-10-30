"""Incremental resource and allocation simulation.

[Role] Execute causal work on explicitly shared physical resources.
[Invariants] Service conserves work. Completion releases successors and owned
  allocations. Finished tasks are removed; event handles, not a global history,
  carry dependencies across submissions. Resource units are work per millisecond.
[Timeouts & Locks] Finite virtual events only; no threads, sleeps or I/O.
[Side-Effect Policy] Mutates this engine and the event handles it creates.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import heapq
import math
from typing import Callable, Mapping, Sequence

from .ownership import unit_rates


class ExecutionError(ValueError):
    pass


@dataclass(frozen=True)
class Resource:
    name: str
    rate: float
    unit: str = "work"
    policy: str = "fair"
    units: int = 1

    def __post_init__(self):
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise ExecutionError(f"resource_rate_invalid:{self.name}")
        if self.policy not in {"fair", "fifo"}:
            raise ExecutionError(f"resource_policy_invalid:{self.policy}")
        if self.units < 1:
            raise ExecutionError(f"resource_units_invalid:{self.name}")


@dataclass(frozen=True)
class Allocation:
    name: str
    pool: str
    size: float
    units: tuple[int, ...] = (0,)


@dataclass(eq=False)
class Event:
    sequence: int
    name: str
    time: float | None = None
    _waiters: list[int] = field(default_factory=list, repr=False)

    @property
    def done(self) -> bool:
        return self.time is not None


@dataclass
class _Task:
    event: Event
    remaining: dict[str, float]
    demand: dict[str, float]
    waiting: int
    release: float
    latency: float
    caps: dict[str, float]
    acquire: tuple[Allocation, ...]
    free: tuple[Allocation, ...]
    owner: str
    metadata: dict
    placement: dict
    coupled: bool = False
    callback: Callable[[Event], None] | None = None
    start: float | None = None
    service: dict[str, float] = field(default_factory=dict)


def _fair_rates(capacity: float, caps: Mapping[int, float]) -> dict[int, float]:
    """Max-min allocation redistributes unused service from capped producers."""
    result: dict[int, float] = {}
    remaining = capacity
    ordered = sorted(caps.items(), key=lambda item: (item[1], item[0]))
    for index, (key, cap) in enumerate(ordered):
        share = remaining / (len(ordered) - index)
        rate = min(cap, share)
        result[key] = rate
        remaining -= rate
    return result


class Engine:
    def __init__(
        self, resources: Sequence[Resource], *,
        pools: Mapping[str, float] | None = None, trace_limit: int = 0,
        pool_units: Mapping[str, int] | None = None,
    ):
        self.resources = {resource.name: resource for resource in resources}
        self.pools = dict(pools or {})
        self.pool_units = {name: (pool_units or {}).get(name, 1) for name in self.pools}
        self.allocated = dict.fromkeys(self.pools, 0.0)
        self.peak_allocated = dict(self.allocated)
        self.unit_allocated = {}
        self.peak_unit_allocated = {}
        self.allocations: dict[str, Allocation] = {}
        self.now = 0.0
        self.serviced = dict.fromkeys(self.resources, 0.0)
        self.busy_ms = dict.fromkeys(self.resources, 0.0)
        self._tasks: dict[int, _Task] = {}
        self._promises: dict[int, Event] = {}
        self._ready: set[int] = set()
        self._active: set[int] = set()
        self._releases: list[tuple[float, int]] = []
        self._sequence = 0
        self.completed_count = 0
        self.peak_tasks = 0
        self.trace = deque(maxlen=trace_limit)
        self.trace_dropped = 0

    def extend(self, resources=(), pools=None, pool_units=None):
        """Add a disjoint physical pool to the same virtual clock."""
        for resource in resources:
            if resource.name in self.resources:
                raise ExecutionError(f"resource_already_declared:{resource.name}")
            self.resources[resource.name] = resource
            self.serviced[resource.name] = self.busy_ms[resource.name] = 0
        for name, capacity in (pools or {}).items():
            if name in self.pools:
                raise ExecutionError(f"pool_already_declared:{name}")
            self.pools[name] = capacity
            self.pool_units[name] = (pool_units or {}).get(name, 1)
            self.allocated[name] = self.peak_allocated[name] = 0

    def submit(
        self, name: str, work: Mapping[str, float] | None = None, *,
        after: Sequence[Event] = (), release: float | None = None,
        latency: float = 0.0, caps: Mapping[str, float] | None = None,
        acquire: Sequence[Allocation] = (), free: Sequence[Allocation] = (),
        owner: str = "", metadata: Mapping | None = None,
        on_complete: Callable[[Event], None] | None = None,
        placement: Mapping[str, Sequence[int]] | None = None,
        coupled: bool = False,
    ) -> Event:
        demands = {key: float(value) for key, value in (work or {}).items() if value}
        for key, value in demands.items():
            if key not in self.resources:
                raise ExecutionError(f"resource_missing:{key}")
            if value < 0 or not math.isfinite(value):
                raise ExecutionError(f"invalid_work:{name}:{key}")
        if latency < 0 or not math.isfinite(latency):
            raise ExecutionError(f"invalid_latency:{name}")
        limits = dict(caps or {})
        if any(value <= 0 or not math.isfinite(value) for value in limits.values()):
            raise ExecutionError(f"invalid_supply_rate:{name}")
        placement = {key: tuple(units) for key, units in (placement or {}).items()}
        for key, units in placement.items():
            if not units or len(set(units)) != len(units) or any(
                unit < 0 or unit >= self.resources[key].units for unit in units
            ):
                raise ExecutionError(f"resource_unit_out_of_range:{key}")
        for allocation in acquire:
            if allocation.size < 0 or not math.isfinite(allocation.size):
                raise ExecutionError(f"invalid_allocation:{allocation.name}")
            if allocation.pool not in self.pools:
                raise ExecutionError(f"pool_missing:{allocation.pool}")
            if allocation.size > self.pools[allocation.pool]:
                raise ExecutionError(f"allocation_exceeds_capacity:{allocation.name}")
            if not allocation.units or len(set(allocation.units)) != len(allocation.units) or any(
                unit < 0 or unit >= self.pool_units[allocation.pool] for unit in allocation.units
            ):
                raise ExecutionError(f"buffer_unit_out_of_range:{allocation.pool}")
        dependencies = tuple(dict.fromkeys(after))
        pending = tuple(event for event in dependencies if not event.done)
        if any(event.sequence not in self._tasks and event.sequence not in self._promises
               for event in pending):
            raise ExecutionError(f"dependency_not_submitted:{name}")
        self._sequence += 1
        event = Event(self._sequence, name)
        release = max(
            self.now, self.now if release is None else release,
            max((event.time for event in dependencies if event.done), default=self.now),
        )
        task = _Task(
            event, dict(demands), demands, len(pending), release, latency, limits,
            tuple(acquire), tuple(free), owner, dict(metadata or {}), placement,
            coupled=coupled, callback=on_complete,
        )
        self._tasks[event.sequence] = task
        for predecessor in pending:
            predecessor._waiters.append(event.sequence)
        if not pending:
            self._queue(task)
        self.peak_tasks = max(self.peak_tasks, len(self._tasks))
        return event

    def future(self, name: str) -> Event:
        """A completion handle for a lazily generated finite program."""
        self._sequence += 1
        event = Event(self._sequence, name)
        self._promises[event.sequence] = event
        return event

    def resolve(self, event: Event) -> None:
        del self._promises[event.sequence]
        event.time = self.now
        self._wake(event)

    def _wake(self, event: Event) -> None:
        for successor in event._waiters:
            dependent = self._tasks[successor]
            dependent.waiting -= 1
            if not dependent.waiting:
                self._queue(dependent)
        event._waiters.clear()

    def _queue(self, task: _Task) -> None:
        if task.release <= self.now:
            self._ready.add(task.event.sequence)
        else:
            heapq.heappush(self._releases, (task.release, task.event.sequence))

    def _can_allocate(self, task: _Task) -> bool:
        amounts = {}
        for item in task.acquire:
            if item.name in self.allocations:
                raise ExecutionError(f"allocation_already_live:{item.name}")
            for unit in item.units:
                key = item.pool, unit
                amounts[key] = amounts.get(key, 0.0) + item.size
        return all(self.unit_allocated.get(key, 0) + amount <= self.pools[key[0]]
                   for key, amount in amounts.items())

    def _complete(self, task: _Task) -> None:
        key = task.event.sequence
        task.event.time = self.now
        for item in task.free:
            live = self.allocations.pop(item.name)
            self.allocated[live.pool] -= live.size * len(live.units)
            for unit in live.units:
                pair = live.pool, unit
                self.unit_allocated[pair] -= live.size
                if not self.unit_allocated[pair]:
                    del self.unit_allocated[pair]
        self._active.discard(key)
        del self._tasks[key]
        self.completed_count += 1
        if self.trace.maxlen:
            if len(self.trace) == self.trace.maxlen:
                self.trace_dropped += 1
            self.trace.append({
                "task": task.event.name, "owner": task.owner,
                "start_ms": task.start, "end_ms": self.now,
                "demand": task.demand, "serviced": task.service,
                "metadata": task.metadata,
                "placement": task.placement,
                "coupled": task.coupled,
            })
        self._wake(task.event)
        if task.callback is not None:
            task.callback(task.event)

    def _settle(self) -> None:
        while self._releases and self._releases[0][0] <= self.now:
            _, key = heapq.heappop(self._releases)
            self._ready.add(key)
        changed = True
        while changed:
            changed = False
            for key in sorted(self._ready):
                task = self._tasks[key]
                if not self._can_allocate(task):
                    continue
                self._ready.remove(key)
                for item in task.acquire:
                    self.allocations[item.name] = item
                    self.allocated[item.pool] += item.size * len(item.units)
                    for unit in item.units:
                        pair = item.pool, unit
                        self.unit_allocated[pair] = self.unit_allocated.get(pair, 0) + item.size
                        self.peak_unit_allocated[pair] = max(
                            self.peak_unit_allocated.get(pair, 0), self.unit_allocated[pair])
                        self.peak_allocated[item.pool] = max(
                            self.peak_allocated[item.pool], self.unit_allocated[pair])
                task.start = self.now
                self._active.add(key)
                changed = True
            for key in sorted(self._active):
                task = self._tasks[key]
                if not task.remaining and self.now >= task.start + task.latency:
                    self._complete(task)
                    changed = True

    def _rates(self) -> dict[str, dict[int, float]]:
        contenders: dict[str, dict[int, float]] = {}
        for key in self._active:
            task = self._tasks[key]
            if task.coupled and task.remaining:
                from .coupled import coupled_rates
                return coupled_rates(self.resources,
                                     {key: self._tasks[key] for key in self._active})
            for name in task.remaining:
                contenders.setdefault(name, {})[key] = task.caps.get(name, math.inf)
        result = {}
        for name, caps in contenders.items():
            resource = self.resources[name]
            if resource.policy == "fifo":
                winner = min(caps)
                result[name] = {winner: min(resource.rate, caps[winner])}
            else:
                pinned = {key: self._tasks[key].placement[name]
                          for key in caps if name in self._tasks[key].placement}
                result[name] = (unit_rates(resource.rate, resource.units, caps, pinned)
                                if pinned else _fair_rates(resource.rate, caps))
        return result

    def advance(
        self, *, until: Sequence[Event] = (), horizon: float = math.inf,
        first: bool = True,
    ) -> tuple[Event, ...]:
        """Advance to a target completion or arrival without skipping releases."""
        if horizon < self.now:
            raise ExecutionError("time_cannot_move_backwards")
        while True:
            self._settle()
            complete = tuple(event for event in until if event.done)
            if (complete and first) or (until and len(complete) == len(until)):
                return complete
            if not self._tasks and self._promises:
                raise ExecutionError("execution_has_unresolved_programs")
            if not self._tasks or self.now >= horizon:
                return complete
            rates = self._rates()
            dt = min(
                horizon - self.now,
                self._releases[0][0] - self.now if self._releases else math.inf,
            )
            for name, consumers in rates.items():
                for key, rate in consumers.items():
                    dt = min(dt, self._tasks[key].remaining[name] / rate)
            for key in self._active:
                task = self._tasks[key]
                if task.start + task.latency > self.now:
                    dt = min(dt, task.start + task.latency - self.now)
            if not math.isfinite(dt):
                names = ",".join(task.event.name for task in self._tasks.values())
                raise ExecutionError(f"execution_stalled:{names[:300]}")
            next_time = self.now + dt
            if dt > 0 and next_time == self.now:
                next_time = math.nextafter(self.now, math.inf)
            for name, consumers in rates.items():
                self.busy_ms[name] += dt
                for key, rate in consumers.items():
                    task = self._tasks[key]
                    remaining = task.remaining[name]
                    used = min(remaining, dt * rate)
                    residual = remaining - used
                    # used is clamped to [0, remaining]; ulp is monotonic on
                    # nonnegative floats, including subnormal values.
                    if residual <= 4 * math.ulp(remaining):
                        used, residual = remaining, 0.0
                    task.service[name] = task.service.get(name, 0.0) + used
                    self.serviced[name] += used
                    if residual:
                        task.remaining[name] = residual
                    else:
                        del task.remaining[name]
            self.now = next_time

    def summary(self) -> dict:
        return {
            "time_ms": self.now, "serviced": dict(self.serviced),
            "resource_busy_ms": dict(self.busy_ms),
            "peak_allocated": dict(self.peak_allocated),
            "live_allocated": dict(self.allocated),
            "peak_allocated_per_unit": {
                f"{pool}:unit{unit}": value
                for (pool, unit), value in self.peak_unit_allocated.items()},
            "completed_tasks": self.completed_count, "live_tasks": len(self._tasks),
            "live_programs": len(self._promises),
            "peak_live_tasks": self.peak_tasks,
            "retained_trace_events": len(self.trace), "trace_dropped": self.trace_dropped,
        }
