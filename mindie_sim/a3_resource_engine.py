"""Event-driven shared-resource service engine (incremental executable model).

[Role]
  Advance an append-only dependency DAG of work tasks over explicitly named
  resource scopes.  A task is admitted when dependencies/release are visible
  and any declared reservation set can be allocated; it is served only when the
  live policy gives it a positive rate.  Admission and first service time are
  therefore distinct witnesses.

[Invariants]
  * One persistent, monotonic timeline.  ``run`` never moves clock backwards,
    never skips an intervening release/service event when asked to advance, and
    retains active work across calls (background service is not restarted).
  * Pending/active/release-waiting work are persistent structures; completing a
    task visits only direct successors.  No event scans completed history or
    revalidates the whole DAG.  ``add_task`` requires known dependencies, so
    append-only registration constructs a DAG in dependency order.
  * Equal-share divides live demanders at each event; FIFO winner uses
    ``(priority, release_ms, enqueue_order, task_id)``.  Serviced work equals
    demand; tiny positive demand is real work, not a tolerance decision.
  * Optional explicit ``reservations`` hold capacity from admission to whole
    task completion.  Fixed mode caps the owner's own rate and gives others the
    unreserved pool; minimum mode adds a pool share.  Residency is separate
    from ``serviced`` work counters.  No implicit/default reservations.
  * A zero-demand task completes at its dependency/release frontier and its
    completion can satisfy successors in the same event loop.
  * ``until``/``until_any`` stop after the complete same-time frontier, so the
    earliest target timestamp is returned; unrelated active background work
    keeps its partially served remaining work for a later call.
  * The default result remains API-compatible: full trace, full completed map
    and full deadline history.  ``compact=True`` and/or an integer ``cursor``
    opt into bounded delta output so callers do not reserialize old history.

[Timeouts & Locks]
  Deterministic in-memory event loop bounded by a task-count guard plus the
  caller's ``horizon_ms``.  No wall-clock sleeps, threads or locks.

[Side-Effect Policy]
  In-memory state only.  No filesystem, network, device or global mutation.
"""
from __future__ import annotations

import bisect
import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

_EPS = 1e-12


def _is_roundoff_residual(
    left: float, remaining: float, used: float, demand: float,
) -> bool:
    """True only for a few-ulp residual, never for genuine small demand."""
    if left <= 0.0:
        return True
    tolerance = 4.0 * max(
        math.ulp(remaining), math.ulp(used), math.ulp(abs(demand)),
    )
    return left <= tolerance


class ResourceError(ValueError):
    """Stable resource-model validation failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)


@dataclass(frozen=True)
class ResourceSpec:
    name: str
    capacity: float
    unit: str = "unit"
    scope: str = "shared"
    policy: str = "equal_share"

    def __post_init__(self) -> None:
        if not self.name:
            raise ResourceError("resource_name_empty")
        if not math.isfinite(self.capacity) or self.capacity <= 0:
            raise ResourceError(f"resource_capacity_invalid:{self.name}")
        if self.policy not in {"equal_share", "fifo"}:
            raise ResourceError(f"resource_policy_unsupported:{self.policy}")


@dataclass
class WorkTask:
    task_id: str
    demands: dict[str, float]
    dependencies: tuple[str, ...] = ()
    release_ms: float = 0.0
    priority: int = 0
    owner: str = ""
    enqueue_order: int = 0
    deadline_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    remaining: dict[str, float] = field(default_factory=dict)
    start_ms: float | None = None
    admitted_ms: float | None = None
    end_ms: float | None = None
    started: bool = False
    completed: bool = False
    service_intervals: list[tuple[str, float, float, float]] = field(default_factory=list)
    # Optional capacity held from admission to whole-task completion.  It is a
    # residency claim, never a service-work counter.
    reservations: dict[str, float] = field(default_factory=dict)
    reservation_mode: str = "fixed"


@dataclass(frozen=True)
class ResourceTrace:
    task_id: str
    start_ms: float
    end_ms: float
    demands: tuple[tuple[str, float], ...]
    owner: str
    ready_ms: float | None = None
    deadline_ms: float | None = None
    slack_ms: float | None = None
    service: tuple[tuple[str, float, float, float], ...] = ()
    residency: tuple[tuple[str, float, float, float], ...] = ()
    reservation_mode: str = "fixed"

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id, "start_ms": self.start_ms, "end_ms": self.end_ms,
            "duration_ms": self.end_ms - self.start_ms, "demands": dict(self.demands),
            "owner": self.owner, "ready_ms": self.ready_ms,
            "deadline_ms": self.deadline_ms, "slack_ms": self.slack_ms,
            "service": [
                {"resource": name, "start_ms": start, "end_ms": end, "rate": rate}
                for name, start, end, rate in self.service
            ],
        }
        if self.residency:
            payload["residency"] = [
                {"resource": name, "start_ms": start, "end_ms": end, "amount": amount}
                for name, start, end, amount in self.residency
            ]
            payload["reservation_mode"] = self.reservation_mode
        return payload


class ResourceEngine:
    """Persistent shared-resource scheduler with incremental output support."""

    def __init__(
        self,
        resources: Mapping[str, ResourceSpec] | Sequence[ResourceSpec],
    ) -> None:
        if isinstance(resources, Mapping):
            specs = dict(resources)
        else:
            specs = {item.name: item for item in resources}
        if not specs:
            raise ResourceError("resources_empty")
        self.resources: dict[str, ResourceSpec] = specs
        self.tasks: dict[str, WorkTask] = {}
        self.completed_at: dict[str, float] = {}
        self.trace: list[ResourceTrace] = []
        self.serviced: dict[str, float] = {name: 0.0 for name in specs}
        self.now_ms: float = 0.0
        self._enqueue_counter: int = 0
        # Persistent scheduler state.  ``_pending``/``_active`` preserve
        # registration/admission order for deterministic same-time output.
        self._pending: dict[str, WorkTask] = {}
        self._active: dict[str, WorkTask] = {}
        self._outstanding: dict[str, int] = {}
        self._dependents: dict[str, list[str]] = {}
        self._ready: list[tuple[int, str]] = []
        self._releases: list[tuple[float, int, str]] = []
        self._reservation_wait: list[tuple[int, float, int, str]] = []
        self._held: dict[str, float] = {}
        self._completion_log: list[str] = []
        self._completion_times: list[float] = []
        self._deadline_slack: dict[str, float] = {}
        self._makespan: float | None = None

    def completed_ms(self, task_id: str) -> float | None:
        """O(1) lookup in the complete persistent map, independent of deltas."""
        return self.completed_at.get(task_id)

    def discard_completed_history(self, keep: Sequence[str] = ()) -> None:
        """Release consumed history after a predictive step has copied its delta.

        Keep preparation/draft witnesses explicitly named by the caller plus
        dependencies of live tasks. Old trace cursors are invalid after this
        operation. Standalone engine users retain history unless they opt in.
        """
        retained = set(keep)
        for task in self._pending.values():
            retained.update(task.dependencies)
        removed = set(self.completed_at) - retained
        for task_id in removed:
            self.tasks.pop(task_id, None)
            self.completed_at.pop(task_id, None)
            self._outstanding.pop(task_id, None)
            self._deadline_slack.pop(task_id, None)
        self.trace = [row for row in self.trace if row.task_id not in removed]
        kept = [(task_id, time) for task_id, time in
                zip(self._completion_log, self._completion_times) if task_id not in removed]
        self._completion_log = [row[0] for row in kept]
        self._completion_times = [row[1] for row in kept]

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def add_task(
        self,
        task_id: str,
        demands: Mapping[str, float],
        *,
        dependencies: Sequence[str] = (),
        release_ms: float = 0.0,
        priority: int = 0,
        owner: str = "",
        deadline_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        reservations: Mapping[str, float] | None = None,
        reservation_mode: str = "fixed",
    ) -> WorkTask:
        if not task_id or task_id in self.tasks:
            raise ResourceError(f"task_id_invalid_or_duplicate:{task_id}")
        clean: dict[str, float] = {}
        for name, raw in demands.items():
            if name not in self.resources:
                raise ResourceError(f"resource_unknown:{name}")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ResourceError(f"task_demand_invalid:{task_id}:{name}") from exc
            if not math.isfinite(value) or value < 0:
                raise ResourceError(f"task_demand_invalid:{task_id}:{name}")
            if value > 0:
                clean[name] = value
        if reservation_mode not in {"fixed", "minimum"}:
            raise ResourceError(f"task_reservation_mode_invalid:{task_id}")
        reserved: dict[str, float] = {}
        for name, raw in (reservations or {}).items():
            if name not in self.resources:
                raise ResourceError(f"resource_unknown:{name}")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ResourceError(
                    f"task_reservation_invalid:{task_id}:{name}"
                ) from exc
            if not math.isfinite(value) or value < 0:
                raise ResourceError(f"task_reservation_invalid:{task_id}:{name}")
            if value > self.resources[name].capacity + _EPS:
                raise ResourceError(
                    f"resource_reservation_exceeds_capacity:{task_id}:{name}"
                )
            if value > 0:
                reserved[name] = value
        if not math.isfinite(float(release_ms)) or float(release_ms) < 0:
            raise ResourceError(f"task_release_invalid:{task_id}")
        if deadline_ms is not None:
            if not math.isfinite(float(deadline_ms)) or float(deadline_ms) < 0:
                raise ResourceError(f"task_deadline_invalid:{task_id}")
        for dep in dependencies:
            if dep not in self.tasks and dep not in self.completed_at:
                raise ResourceError(f"task_dependency_unknown:{task_id}:{dep}")
            if dep == task_id:
                raise ResourceError(f"task_self_dependency:{task_id}")
        task = WorkTask(
            task_id=task_id, demands=clean, dependencies=tuple(dependencies),
            release_ms=float(release_ms), priority=int(priority), owner=owner,
            enqueue_order=self._enqueue_counter,
            deadline_ms=float(deadline_ms) if deadline_ms is not None else None,
            metadata=dict(metadata or {}), remaining=dict(clean),
            reservations=reserved, reservation_mode=reservation_mode,
        )
        self._enqueue_counter += 1
        self.tasks[task_id] = task
        self._pending[task_id] = task
        unique_deps = tuple(dict.fromkeys(dependencies))
        outstanding = 0
        for dep in unique_deps:
            dep_task = self.tasks.get(dep)
            if dep_task is not None and not dep_task.completed:
                outstanding += 1
                self._dependents.setdefault(dep, []).append(task_id)
        self._outstanding[task_id] = outstanding
        if outstanding == 0:
            self._schedule(task, self.now_ms)
        return task

    # ------------------------------------------------------------------
    # Frontier and dependency transitions
    # ------------------------------------------------------------------
    def _schedule(self, task: WorkTask, now: float) -> None:
        if task.completed or task.started:
            return
        if self._outstanding.get(task.task_id, 0) != 0:
            return
        if task.release_ms <= now + _EPS:
            heapq.heappush(self._ready, (task.enqueue_order, task.task_id))
        else:
            heapq.heappush(
                self._releases,
                (task.release_ms, task.enqueue_order, task.task_id),
            )

    def _can_reserve(self, task: WorkTask) -> bool:
        for resource, amount in task.reservations.items():
            spec = self.resources.get(resource)
            if spec is None:
                raise ResourceError(f"resource_unknown:{resource}")
            if amount > spec.capacity + _EPS:
                raise ResourceError(
                    f"resource_reservation_exceeds_capacity:"
                    f"{task.task_id}:{resource}"
                )
            if self._held.get(resource, 0.0) + amount > spec.capacity + _EPS:
                return False
        return True

    def _admit(self, task: WorkTask, now: float) -> None:
        task.started = True
        if task.admitted_ms is None:
            task.admitted_ms = now
        for resource, amount in task.reservations.items():
            if amount > 0.0:
                self._held[resource] = self._held.get(resource, 0.0) + amount
        if not task.remaining:
            task.start_ms = now
            self._complete(task, now)
        else:
            self._active[task.task_id] = task

    def _drain_frontier(self, now: float) -> None:
        """Admit all work visible at ``now`` and cascade zero-work completions."""
        while True:
            moved = False
            while self._releases and self._releases[0][0] <= now + _EPS:
                _, _, task_id = heapq.heappop(self._releases)
                task = self.tasks.get(task_id)
                if task is None or task.completed or task.started:
                    continue
                if self._outstanding.get(task_id, 0) != 0:
                    continue
                heapq.heappush(self._ready, (task.enqueue_order, task_id))
                moved = True
            if self._ready:
                _, task_id = heapq.heappop(self._ready)
                task = self.tasks.get(task_id)
                if task is None or task.completed or task.started:
                    continue
                if self._outstanding.get(task_id, 0) != 0:
                    continue
                if task.reservations and not self._can_reserve(task):
                    heapq.heappush(self._reservation_wait, (
                        task.priority, task.release_ms,
                        task.enqueue_order, task.task_id,
                    ))
                    continue
                self._admit(task, now)
                continue
            while self._reservation_wait:
                _, _, _, task_id = self._reservation_wait[0]
                task = self.tasks.get(task_id)
                if (task is None or task.completed or task.started
                        or self._outstanding.get(task_id, 0) != 0):
                    heapq.heappop(self._reservation_wait)
                    continue
                if not self._can_reserve(task):
                    break
                heapq.heappop(self._reservation_wait)
                self._admit(task, now)
                moved = True
                break
            if moved:
                continue
            break

    def _complete(self, task: WorkTask, now: float) -> None:
        task.completed = True
        task.end_ms = now
        task.remaining = {key: 0.0 for key in task.demands}
        self.completed_at[task.task_id] = now
        slack = None if task.deadline_ms is None else task.deadline_ms - now
        admitted = task.admitted_ms if task.admitted_ms is not None else now
        residency = tuple(
            (name, float(admitted), now, amount)
            for name, amount in sorted(task.reservations.items())
            if amount > 0.0
        )
        self.trace.append(ResourceTrace(
            task.task_id,
            float(task.start_ms if task.start_ms is not None else now), now,
            tuple(sorted(task.demands.items())), task.owner,
            task.admitted_ms, task.deadline_ms, slack,
            tuple(task.service_intervals), residency, task.reservation_mode,
        ))
        if slack is not None:
            self._deadline_slack[task.task_id] = slack
        for resource, amount in task.reservations.items():
            left = self._held.get(resource, 0.0) - amount
            if left <= _EPS:
                self._held.pop(resource, None)
            else:
                self._held[resource] = left
        self._pending.pop(task.task_id, None)
        self._active.pop(task.task_id, None)
        self._completion_log.append(task.task_id)
        self._completion_times.append(now)
        if self._makespan is None or now > self._makespan:
            self._makespan = now
        for child_id in self._dependents.pop(task.task_id, ()):
            child = self.tasks.get(child_id)
            if child is None or child.completed:
                continue
            count = self._outstanding.get(child_id, 0) - 1
            if count <= 0:
                self._outstanding[child_id] = 0
                self._schedule(child, now)
            else:
                self._outstanding[child_id] = count

    def _next_release_time(self) -> float | None:
        while self._releases:
            release, _, task_id = self._releases[0]
            task = self.tasks.get(task_id)
            if (task is None or task.completed or task.started
                    or self._outstanding.get(task_id, 0) != 0):
                heapq.heappop(self._releases)
                continue
            return release
        return None

    # ------------------------------------------------------------------
    # Service
    # ------------------------------------------------------------------
    def _compute_shares(self) -> dict[tuple[str, str], float]:
        active = list(self._active.values())
        users: dict[str, list[WorkTask]] = {}
        for task in active:
            for resource, remaining in task.remaining.items():
                # Positive demand is real work even below the float-comparison
                # epsilon used for event collisions.
                if remaining > 0.0:
                    users.setdefault(resource, []).append(task)
        shares: dict[tuple[str, str], float] = {}
        for resource in set(users) | set(self._held):
            spec = self.resources.get(resource)
            if spec is None:
                raise ResourceError(f"resource_unknown:{resource}")
            held = self._held.get(resource, 0.0)
            if held > spec.capacity + _EPS:
                raise ResourceError(
                    f"resource_reservation_over_capacity:{resource}"
                )
            pool = max(0.0, spec.capacity - held)
            demanders = users.get(resource, [])
            if not demanders:
                continue
            fixed_ids = {
                task.task_id for task in demanders
                if task.reservation_mode == "fixed"
                and task.reservations.get(resource, 0.0) > 0.0
            }
            pool_users = [
                task for task in demanders if task.task_id not in fixed_ids
            ]
            if spec.policy == "fifo":
                winner = min(pool_users, key=lambda item: (
                    item.priority, item.release_ms, item.enqueue_order, item.task_id,
                )) if pool_users else None
                for task in demanders:
                    if task.task_id in fixed_ids:
                        rate = task.reservations.get(resource, 0.0)
                    else:
                        own = (
                            task.reservations.get(resource, 0.0)
                            if task.reservation_mode == "minimum" else 0.0
                        )
                        rate = own + (pool if task is winner else 0.0)
                    if rate > 0.0:
                        shares[(resource, task.task_id)] = rate
            else:
                per = pool / len(pool_users) if pool_users else 0.0
                for task in demanders:
                    if task.task_id in fixed_ids:
                        rate = task.reservations.get(resource, 0.0)
                    else:
                        own = (
                            task.reservations.get(resource, 0.0)
                            if task.reservation_mode == "minimum" else 0.0
                        )
                        rate = own + per
                    if rate > 0.0:
                        shares[(resource, task.task_id)] = rate
        return shares

    def _service(
        self, dt: float, shares: Mapping[tuple[str, str], float],
        now: float,
    ) -> list[WorkTask]:
        finished: list[WorkTask] = []
        for task in self._active.values():
            for resource, remaining in task.remaining.items():
                if remaining <= 0.0:
                    task.remaining[resource] = 0.0
                    continue
                rate = shares.get((resource, task.task_id), 0.0)
                if rate <= 0:
                    continue
                if task.start_ms is None:
                    task.start_ms = now
                used = rate * dt
                if used >= remaining:
                    used = remaining
                    left = 0.0
                else:
                    left = remaining - used
                if used <= 0.0:
                    # Subnormal underflow still serves the whole residual.
                    used = remaining
                    left = 0.0
                if left > 0.0 and _is_roundoff_residual(
                    left, remaining, used, task.demands.get(resource, 0.0),
                ):
                    left = 0.0
                task.remaining[resource] = left
                service_end = now + min(used / rate, dt)
                intervals = task.service_intervals
                if (intervals and intervals[-1][0] == resource
                        and abs(intervals[-1][2] - now) <= _EPS
                        and abs(intervals[-1][3] - rate) <= _EPS):
                    intervals[-1] = (resource, intervals[-1][1], service_end, rate)
                else:
                    intervals.append((resource, now, service_end, rate))
                self.serviced[resource] = self.serviced.get(resource, 0.0) + used
            if all(value <= 0.0 for value in task.remaining.values()):
                finished.append(task)
        return finished

    # ------------------------------------------------------------------
    # Public run / result projection
    # ------------------------------------------------------------------
    def run(
        self, *, start_ms: float | None = None, horizon_ms: float = 1e12,
        until: Sequence[str] | None = None, until_any: bool = False,
        compact: bool = False, cursor: int | None = None,
    ) -> dict[str, Any]:
        """Advance the persistent timeline and project a result snapshot.

        ``start_ms=None`` resumes at the monotonic frontier.  An earlier value
        is clamped by the frontier.  A later value is an advance request: the
        engine services already active work and processes every intervening
        release/completion event instead of jumping over them.  ``until``
        stops after the target(s) complete and the complete same-time frontier
        has been finalized; background work keeps its remaining service state.

        The default (``compact=False``, ``cursor=None``) returns full history.
        ``compact=True`` omits historical ``completed_at_ms``/``trace`` and
        deadline copies, including only this call's new completions plus all
        ``until`` targets for lookup compatibility.  ``cursor`` is the trace
        cursor returned as ``trace_cursor``; when supplied, ``trace`` is the
        delta from that cursor.  Compact/cursor results also expose
        ``new_completed_at_ms``, ``trace_start_cursor`` and ``trace_cursor``.
        """
        requested = float(self.now_ms if start_ms is None else start_ms)
        now = float(self.now_ms)
        if now < 0 or not math.isfinite(now):
            raise ResourceError("start_ms_invalid")
        if requested < 0 or not math.isfinite(requested):
            raise ResourceError("start_ms_invalid")
        targets = tuple(str(item) for item in (until or ()))
        for target in targets:
            if target not in self.tasks and target not in self.completed_at:
                raise ResourceError(f"until_task_unknown:{target}")
        if cursor is not None:
            if (isinstance(cursor, bool) or not isinstance(cursor, int)
                    or cursor < 0 or cursor > len(self.trace)):
                raise ResourceError(f"trace_cursor_invalid:{cursor}")
        run_trace_start = len(self.trace)
        run_completion_start = len(self._completion_log)
        trace_start = cursor if cursor is not None else (
            run_trace_start if compact else 0
        )
        guard_limit = max(10000, len(self.tasks) * 100 + 1000)
        guard = 0
        while True:
            guard += 1
            if guard > guard_limit:
                raise ResourceError("resource_engine_no_progress")
            self._drain_frontier(now)
            if targets and (
                (until_any and any(item in self.completed_at for item in targets))
                or (not until_any
                    and all(item in self.completed_at for item in targets))
            ):
                break
            if not self._pending:
                break
            if not self._active:
                next_release = self._next_release_time()
                if next_release is None:
                    raise ResourceError("resource_engine_deadlock")
                if next_release > horizon_ms:
                    raise ResourceError(f"resource_horizon_exceeded:{horizon_ms}")
                now = max(now, next_release)
                continue
            shares = self._compute_shares()
            dt = math.inf
            serviced_any = False
            for task in self._active.values():
                for resource, remaining in task.remaining.items():
                    if remaining <= 0.0:
                        continue
                    rate = shares.get((resource, task.task_id), 0.0)
                    if rate > 0:
                        serviced_any = True
                        dt = min(dt, remaining / rate)
            if not serviced_any or not math.isfinite(dt):
                raise ResourceError("resource_engine_no_service")
            next_release = self._next_release_time()
            if next_release is not None and next_release > now + _EPS:
                dt = min(dt, next_release - now)
            if not math.isfinite(dt):
                raise ResourceError("resource_engine_no_service")
            if dt <= 0:
                dt = _EPS
            next_now = now + dt
            if next_now > horizon_ms:
                raise ResourceError(f"resource_horizon_exceeded:{horizon_ms}")
            finished = self._service(dt, shares, now)
            now = next_now
            for task in sorted(finished, key=lambda item: (
                item.enqueue_order, item.task_id,
            )):
                if not task.completed:
                    self._complete(task, now)
        self.now_ms = now
        return self._project(
            requested=requested, targets=targets, compact=compact,
            cursor=cursor, run_trace_start=run_trace_start,
            run_completion_start=run_completion_start,
            trace_start=trace_start,
        )

    def _project(
        self, *, requested: float, targets: tuple[str, ...], compact: bool,
        cursor: int | None, run_trace_start: int,
        run_completion_start: int, trace_start: int,
    ) -> dict[str, Any]:
        new_ids = self._completion_log[run_completion_start:]
        new_completed = {task_id: self.completed_at[task_id] for task_id in new_ids}
        if compact:
            completed = dict(new_completed)
            for target in targets:
                if target in self.completed_at:
                    completed[target] = self.completed_at[target]
            completed_out = dict(sorted(completed.items()))
            completed_this = dict(sorted(new_completed.items()))
            slack_ids = [
                task_id for task_id in new_ids
                if task_id in self._deadline_slack
            ]
            for target in targets:
                if target in self._deadline_slack and target not in new_completed:
                    slack_ids.append(target)
            deadline_slack = {
                task_id: self._deadline_slack[task_id] for task_id in slack_ids
            }
        else:
            completed_out = dict(sorted(self.completed_at.items()))
            first = bisect.bisect_left(
                self._completion_times, requested - _EPS,
            )
            completed_this = dict(sorted(
                (task_id, self.completed_at[task_id])
                for task_id in self._completion_log[first:]
            ))
            deadline_slack = dict(self._deadline_slack)
        result: dict[str, Any] = {
            "status": "modeled",
            "start_ms": requested,
            "resume_ms": self.now_ms,
            "makespan_ms": (
                self._makespan if self._makespan is not None else self.now_ms
            ),
            "target_completed_at_ms": {
                key: self.completed_at[key] for key in targets
                if key in self.completed_at
            },
            "drain_complete": not self._pending,
            "remaining_tasks": list(self._pending),
            "completed_at_ms": completed_out,
            "completed_this_run_ms": completed_this,
            "trace": [item.to_dict() for item in self.trace[trace_start:]],
            "deadline_slack_ms": deadline_slack,
            "deadline_misses": [
                task_id for task_id, slack in deadline_slack.items()
                if slack is not None and slack < -_EPS
            ],
            "serviced": dict(sorted(self.serviced.items())),
            "resources": {
                name: {"capacity": spec.capacity, "unit": spec.unit,
                       "scope": spec.scope, "policy": spec.policy}
                for name, spec in self.resources.items()
            },
        }
        if compact or cursor is not None:
            result.update({
                "compact": bool(compact),
                "new_completed_at_ms": dict(sorted(new_completed.items())),
                "trace_start_cursor": trace_start,
                "trace_cursor": len(self.trace),
            })
        return result


__all__ = [
    "ResourceError", "ResourceSpec", "WorkTask", "ResourceTrace", "ResourceEngine",
]
