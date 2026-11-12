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


@dataclass(slots=True)
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
    allocation_demands: tuple[tuple[tuple[str, int], float], ...]
    coupled: bool = False
    callback: Callable[[Event], None] | None = None
    start: float | None = None
    service: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _PreparedTask:
    """Validated static service; each submission owns its remaining work."""

    demand: dict[str, float]
    latency: float
    caps: dict[str, float]
    acquire: tuple[Allocation, ...]
    free: tuple[Allocation, ...]
    placement: dict[str, tuple[int, ...]]
    coupled: bool
    allocation_demands: tuple[tuple[tuple[str, int], float], ...]


_EMPTY_TASK = _PreparedTask({}, 0.0, {}, (), (), {}, False, ())


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
        # Ready and completable heaps are sequence-ordered. Running tasks stay
        # out of the completable heap until work and latency are both done, so
        # a zero-work chain does not re-sort every overlapping active task.
        self._ready_heap: list[int] = []
        self._ready_seen: set[int] = set()
        self._completable: list[int] = []
        self._completable_seen: set[int] = set()
        self._active: set[int] = set()
        self._releases: list[tuple[float, int]] = []
        self._sequence = 0
        self.completed_count = 0
        self.peak_tasks = 0
        self.trace = deque(maxlen=trace_limit)
        self.trace_dropped = 0
        # Isolated-schedule trace. A private program records the exact advance
        # steps that touched only its own replica resources; identical ranks and
        # later invocations replay those steps instead of rebuilding tasks.
        self._trace_prefix: str | None = None
        self._trace_token: str | None = None
        self._trace_steps: list | None = None
        self._trace_peaks: dict | None = None
        self._trace_clean = True
        self._trace_private = True
        self._script_pending = None
        self._script_pending_key = None

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
        if (work is None and latency == 0.0 and caps is None and not acquire
                and not free and placement is None and not coupled):
            prepared = _EMPTY_TASK
        else:
            prepared = self._prepare(
                name, work, latency=latency, caps=caps, acquire=acquire, free=free,
                placement=placement, coupled=coupled,
            )
        return self._submit_prepared(
            prepared, name, after=after, release=release, owner=owner,
            metadata=metadata, on_complete=on_complete,
        )

    def _prepare(
        self, name, work=None, *, latency=0.0, caps=None, acquire=(), free=(),
        placement=None, coupled=False,
    ) -> _PreparedTask:
        """Validate a template against this engine's physical resource binding."""
        if (work is None and latency == 0 and caps is None and acquire == ()
                and free == () and placement is None and not coupled):
            return _EMPTY_TASK
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
        allocation_demands = {}
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
            for unit in allocation.units:
                key = allocation.pool, unit
                allocation_demands[key] = allocation_demands.get(key, 0.0) + allocation.size
        return _PreparedTask(demands, latency, limits, tuple(acquire), tuple(free),
                             placement, coupled, tuple(allocation_demands.items()))

    def _submit_prepared(
        self, prepared, name, *, after=(), release=None, owner="", metadata=None,
        on_complete=None,
    ) -> Event:
        # Dependencies and allocations remain live invocation state. Only static
        # work, supply and placement validation can be shared by repeated tiles.
        dependencies = tuple(after)
        if len(dependencies) > 1:
            dependencies = tuple(dict.fromkeys(dependencies))
        pending = []
        release = max(self.now, self.now if release is None else release)
        for predecessor in dependencies:
            if predecessor.time is not None:
                release = max(release, predecessor.time)
            else:
                if predecessor.sequence not in self._tasks and predecessor.sequence not in self._promises:
                    raise ExecutionError(f"dependency_not_submitted:{name}")
                pending.append(predecessor)
        self._sequence += 1
        event = Event(self._sequence, name)
        task = _Task(
            event, dict(prepared.demand), prepared.demand, len(pending), release,
            prepared.latency, prepared.caps, prepared.acquire, prepared.free,
            owner, dict(metadata or {}), prepared.placement, prepared.allocation_demands,
            coupled=prepared.coupled, callback=on_complete,
        )
        self._tasks[event.sequence] = task
        for predecessor in pending:
            predecessor._waiters.append(event.sequence)
        if not pending:
            self._queue(task)
        self.peak_tasks = max(self.peak_tasks, len(self._tasks))
        self._note_traced_submit(name, prepared)
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

    def _push_ready(self, key: int) -> None:
        if key in self._ready_seen:
            return
        self._ready_seen.add(key)
        heapq.heappush(self._ready_heap, key)

    def _push_completable(self, key: int) -> None:
        if key in self._completable_seen:
            return
        self._completable_seen.add(key)
        heapq.heappush(self._completable, key)

    def _queue(self, task: _Task) -> None:
        if task.release <= self.now:
            self._push_ready(task.event.sequence)
        else:
            heapq.heappush(self._releases, (task.release, task.event.sequence))

    def _can_allocate(self, task: _Task) -> bool:
        for item in task.acquire:
            if item.name in self.allocations:
                raise ExecutionError(f"allocation_already_live:{item.name}")
        # Static amounts belong to the prepared stage; only current occupancy
        # is reevaluated as other streams acquire and release the same units.
        for key, amount in task.allocation_demands:
            if self.unit_allocated.get(key, 0) + amount > self.pools[key[0]]:
                return False
        return True

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
        self._completable_seen.discard(key)
        del self._tasks[key]
        self.completed_count += 1
        if self.trace.maxlen:
            if len(self.trace) == self.trace.maxlen:
                self.trace_dropped += 1
            self.trace.append({
                "task": task.event.name, "owner": task.owner,
                "start_ms": task.start, "end_ms": self.now,
                "demand": dict(task.demand), "serviced": task.service,
                "metadata": task.metadata,
                "placement": dict(task.placement),
                "coupled": task.coupled,
            })
        self._wake(task.event)
        if task.callback is not None:
            task.callback(task.event)

    def _start_ready(self, key: int) -> None:
        task = self._tasks[key]
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
        if self._trace_prefix and task.event.name.startswith(self._trace_prefix):
            token = self._trace_token
            peaks = self._trace_peaks
            for item in task.acquire:
                if token not in item.pool:
                    continue
                for unit in item.units:
                    val = self.unit_allocated[item.pool, unit]
                    norm = item.pool.replace(token, "@r*", 1), unit
                    if val > peaks.get(norm, 0.0):
                        peaks[norm] = val
        if not task.remaining and self.now >= task.start + task.latency:
            self._push_completable(key)

    def _settle(self) -> None:
        while self._releases and self._releases[0][0] <= self.now:
            _, key = heapq.heappop(self._releases)
            self._push_ready(key)
        for key in self._active:
            task = self._tasks[key]
            if not task.remaining and self.now >= task.start + task.latency:
                self._push_completable(key)
        while True:
            deferred: list[int] = []
            while self._ready_heap:
                key = self._ready_heap[0]
                task = self._tasks.get(key)
                heapq.heappop(self._ready_heap)
                self._ready_seen.discard(key)
                if task is None:
                    continue
                if not self._can_allocate(task):
                    deferred.append(key)
                    continue
                self._start_ready(key)
            for key in deferred:
                self._push_ready(key)
            if not self._completable:
                return
            # Drain the snapshot before starting tasks woken by these callbacks.
            # Lower sequence numbers still complete first, matching sorted order.
            batch = [heapq.heappop(self._completable) for _ in range(len(self._completable))]
            for key in batch:
                self._completable_seen.discard(key)
                task = self._tasks.get(key)
                if task is None or task.remaining or self.now < task.start + task.latency:
                    continue
                self._complete(task)

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
            if self._script_pending is not None and self.replay_ready():
                self._drain_script()
                continue
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
            tracing = self._trace_prefix is not None
            adds = []
            busy = []
            if tracing:
                prefix = self._trace_prefix
                for key in self._active:
                    if not self._tasks[key].event.name.startswith(prefix):
                        self._trace_clean = False
                        break
                if self._releases:
                    rel_t, rel_key = self._releases[0]
                    rel_task = self._tasks.get(rel_key)
                    if rel_t <= self.now + dt and (
                            rel_task is None or not rel_task.event.name.startswith(prefix)):
                        self._trace_clean = False
            for name, consumers in rates.items():
                own_resource = tracing and self._trace_token in name
                if tracing and not own_resource:
                    self._trace_private = False
                    self._trace_clean = False
                if own_resource:
                    busy.append(name.replace(self._trace_token, "@r*", 1))
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
                    if self.trace.maxlen:
                        task.service[name] = task.service.get(name, 0.0) + used
                    self.serviced[name] += used
                    if own_resource and task.event.name.startswith(self._trace_prefix):
                        adds.append((name.replace(self._trace_token, "@r*", 1), used))
                    if residual:
                        task.remaining[name] = residual
                    else:
                        del task.remaining[name]
            if tracing and self._trace_clean:
                self._trace_steps.append((dt, tuple(busy), tuple(adds)))
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

    def _note_traced_submit(self, name: str, prepared: _PreparedTask) -> None:
        prefix = self._trace_prefix
        if prefix is None:
            return
        if (prepared.demand or prepared.acquire) and not name.startswith(prefix):
            self._trace_clean = False
        token = self._trace_token
        if token is None:
            return
        for key in prepared.demand:
            if token not in key:
                self._trace_private = False
                self._trace_clean = False
        for item in prepared.acquire:
            if token not in item.pool:
                self._trace_private = False
                self._trace_clean = False

    def begin_trace(self, prefix: str, token: str) -> None:
        """Record one private program's advance steps under ``prefix``."""
        self._trace_prefix = prefix
        self._trace_token = token
        self._trace_steps = []
        self._trace_peaks = {}
        self._trace_clean = True
        self._trace_private = True

    def end_trace(self):
        """Stop recording. The script is reusable only when every step stayed private."""
        clean = self._trace_clean and self._trace_private
        steps = tuple(self._trace_steps or ())
        peaks = tuple((pool, unit, value) for (pool, unit), value in (self._trace_peaks or {}).items())
        self._trace_prefix = None
        self._trace_token = None
        self._trace_steps = None
        self._trace_peaks = None
        return clean, steps, peaks

    def replay_ready(self) -> bool:
        """True when no live demand can change an isolated program's rates or dt."""
        if self._trace_prefix is not None:
            return False
        for task in self._tasks.values():
            if task.remaining or task.allocation_demands:
                return False
        for key in self._active:
            task = self._tasks[key]
            if task.latency and task.start is not None and self.now < task.start + task.latency:
                return False
        return True

    def queue_script(self, key, steps, members, peaks) -> None:
        """Replay ``steps`` once for every ``(rank token, completion)`` member."""
        self._script_pending_key = key
        self._script_pending = (steps, list(members), peaks)

    def _drain_script(self) -> None:
        steps, members, peaks = self._script_pending
        self._script_pending = None
        self._script_pending_key = None
        for dt, busy, adds in steps:
            next_time = self.now + dt
            if dt > 0 and next_time == self.now:
                next_time = math.nextafter(self.now, math.inf)
            for name in busy:
                for token, _complete in members:
                    concrete = name.replace("@r*", token, 1)
                    if concrete in self.busy_ms:
                        self.busy_ms[concrete] += dt
            for name, used in adds:
                for token, _complete in members:
                    concrete = name.replace("@r*", token, 1)
                    if concrete in self.serviced:
                        self.serviced[concrete] += used
            self.now = next_time
        for pool, unit, value in peaks:
            for token, _complete in members:
                concrete = pool.replace("@r*", token, 1)
                if concrete not in self.peak_allocated:
                    continue
                pair = concrete, unit
                if value > self.peak_unit_allocated.get(pair, 0.0):
                    self.peak_unit_allocated[pair] = value
                if value > self.peak_allocated[concrete]:
                    self.peak_allocated[concrete] = value
        for _token, complete in members:
            if complete.time is None:
                self.resolve(complete)
