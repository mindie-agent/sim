"""Canonical continuous-time resource-sharing Engine.

[Role] Single canonical Engine owning continuous-time resource arbitration,
  periodic-stream execution, and C11 fast-coupled dispatch.
[Invariants] Monotone event clock, exact resource conservation, and zero
  divergent shadow simulation tracks.
[Timeouts & Locks] Honors step deadlines; single-threaded event loop.
[Side-Effect Policy] Pure in-memory simulation."""
from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
import math

from .engine_types import ExecutionError, _structural_value

_PERIODIC_EPOCH_MISS_MULTIPLIER = 2
_PERIODIC_COLD_ISSUE_MULTIPLIER = 2
_MAX_PERIODIC_LIVE_TASKS = 4096
_MAX_PERIODIC_RUNS = 64
_MAX_PERIODIC_LIVE_SLOTS = 128
_BINARY64_EXACT_INT_LIMIT = 1 << 53
_BINARY64_DYADIC_DEN_LIMIT = 1 << 500


def _advance_periodic_clock_single_step(now: float, dt: float, cycles: int) -> float:
    """Exact binary64 repeated addition ``now + dt`` (with nextafter on underflow) in O(log cycles)."""
    if dt <= 0.0 or cycles <= 0:
        return now
    rem = int(cycles)
    while rem > 0:
        if now == 0.0:
            now = dt
            rem -= 1
            continue
        _, exp = math.frexp(now)
        next_binade = math.ldexp(1.0, exp)
        one = now + dt
        if one == now:
            one = math.nextafter(now, math.inf)
        if one >= next_binade or rem == 1:
            now = one
            rem -= 1
            continue
        two = one + dt
        if two == one:
            two = math.nextafter(one, math.inf)
        d1 = one - now
        d2 = two - one
        if d1 <= 0.0 or d1 != d2 or two >= next_binade:
            now = one
            rem -= 1
            continue
        jump = min(rem, int((next_binade - now) // d1))
        while jump > 1 and now + jump * d1 >= next_binade:
            jump -= 1
        if jump <= 1:
            now = one
            rem -= 1
        else:
            now = now + jump * d1
            rem -= jump
    return now


def _advance_periodic_clock_exact(now: float, steps: tuple[float, ...], cycles: int) -> float:
    """Bit-exact periodic clock advance avoiding O(cycles) loops for large skip counts."""
    if cycles <= 0 or not steps:
        return now
    if all(dt >= 0.0 and math.isfinite(dt) for dt in steps) and now >= 0.0 and math.isfinite(now):
        now_num, now_den = now.as_integer_ratio()
        ratios = [dt.as_integer_ratio() for dt in steps]
        max_den = max(now_den, *(den for _, den in ratios))
        if max_den <= _BINARY64_DYADIC_DEN_LIMIT:
            i_now = now_num * (max_den // now_den)
            i_sum = sum(num * (max_den // den) for num, den in ratios)
            i_end = i_now + i_sum * cycles
            if i_end <= _BINARY64_EXACT_INT_LIMIT:
                return i_end / max_den
        if len(steps) == 1:
            return _advance_periodic_clock_single_step(now, steps[0], cycles)
    from ..engine_periodic.frontier import _advance_periodic_clock
    return _advance_periodic_clock(now, steps, cycles)


def _admission_l2_digest(view) -> str:
    """Return a deterministic digest of live L2 residency, reservations, and dirty state."""
    fingerprint = getattr(view, "residency_fingerprint", None)
    return fingerprint() if callable(fingerprint) else ""


def _counter_delta(current, previous):
    return {key: delta for key, value in current.items() if (delta := value - previous.get(key, 0.0))}


class _EnginePeriodicMixin:
    def _allocation_frontier(self) -> tuple[tuple, tuple, tuple]:
        """Cached ``(allocations, pools, grants)`` structural residue.

        Allocation occupancy and the persistent grant map only change when a
        task acquires/releases a buffer or a memory grant is taken/returned;
        ``_alloc_state_version`` is bumped at exactly those boundaries.  The
        previous view is reused otherwise, so the periodic observer no longer
        re-walks every persistent model weight/KV grant on every micro-step.
        """
        cached = self._periodic_alloc_cache
        version = self._alloc_state_version
        if cached is not None and cached[0] == version:
            return cached[1], cached[2], cached[3]
        allocation_shapes = {}
        for item in self.allocations.values():
            allocation_shapes[(item.pool, item.size, item.units)] = (
                allocation_shapes.get((item.pool, item.size, item.units), 0) + 1)
        allocations = tuple(sorted(
            (pool, size, units, count)
            for (pool, size, units), count in allocation_shapes.items()))
        pools = tuple(sorted(
            (name, round(self.allocated.get(name, 0.0), 12),
             tuple(sorted((unit, round(amount, 12)) for unit, amount
                          in self.unit_allocated.get(name, {}).items())))
            for name in self.pools))
        grants = tuple(sorted(
            (name, record["replica"],
             tuple(sorted((str(key), _structural_value(value))
                          for key, value in record["binding"].items()
                          if str(key) != "last_use")))
            for name, record in self._memory_grants.items()))
        self._periodic_alloc_cache = (version, allocations, pools, grants)
        return allocations, pools, grants

    def _periodic_state(self, epoch: str | None = None) -> tuple:
        """Tile-index-independent exact structural view for the detector."""
        def task_key(task):
            metadata = self._periodic_tasks.get(task.event.sequence)
            if metadata is not None:
                run_id, tile, ordinal, kind = metadata
                run = self._periodic_runs.get(run_id, {})
                return ("periodic", run_id, tile - run.get("finished", 0),
                        ordinal, kind)
            return ("opaque", task.event.name)

        tasks = tuple(sorted(
            (task_key(task),
             tuple(sorted((key, value) for key, value in task.remaining.items()
                          if value > 0)),
             task.waiting,
             round(max(0.0, task.release - self.now), 12),
             (None if task.start is None else
              round(max(0.0, task.start + task.latency - self.now), 12)),
             task.priority,
             tuple(sorted(task.caps.items())),
             tuple(sorted(task.placement.items())),
             task.coupled, task.owner)
            for task in self._tasks.values()))
        runs = tuple(sorted(
            (run_id, run.get("phase_ordinal", 0), run.get("phase_end", 0),
             tuple(run.get("slots", ())),
             tuple(sorted((a.pool, a.size, a.units)
                          for a in run.get("persistent_allocations", ()))))
            for run_id, run in self._periodic_runs.items()))
        allocations, pools, grants = self._allocation_frontier()
        releases = tuple(
            (round(when - self.now, 12), key) for when, key in self._releases)
        # Keep the frozen 5-tuple adapter shape so engine_periodic can intern
        # the static task template. Memory budget/grants are structural too,
        # so they ride in the pool part rather than changing the outer arity.
        pools = (*pools, ("@memory", 0.0,
                          (((self._memory_epoch() if epoch is None else epoch), 0.0,
                            (("grants", grants),))),))
        return (tasks, runs, allocations, pools, releases)

    def _memory_epoch(self) -> str:
        """Exact shared-budget generation for one settled frontier.

        A real grant/release/hold/retire changes ``budget_version``. Using the
        live version (instead of a constant 0) makes an externally changed
        admission frontier invalidate the previous periodic proof.
        """
        ledgers = dict(self._memory_ledgers)
        if self.memory is not None:
            ledgers[id(self.memory)] = self.memory

        def version(ledger):
            value = getattr(ledger, "budget_version", None)
            if value is not None:
                return int(value)
            pools = getattr(ledger, "pools", None)
            if isinstance(pools, _AbcMapping):
                return sum(int(getattr(pool, "budget_version", 0))
                           for pool in pools.values())
            return 0

        parts = sorted(str(version(ledger)) for ledger in ledgers.values())
        base = "mem:" + ":".join(parts or ("0",))
        l2_dig = _admission_l2_digest(getattr(self, "_admission_memory_view", None))
        return f"{base}|l2:{l2_dig}" if l2_dig else base

    def _progressable_periodic_run(self, periods: int) -> bool:
        """Cheap necessary condition for any further periodic evidence.

        A skip needs ``evidence_periods`` repeated periods *after* the current
        landmark, and each period advances the live window by its slot count
        (one retired and one issued tile per slot step).  A run that can no
        longer issue that many tiles cannot produce one more comparison, so
        the observer must not pay a full structural validation for it.
        """
        for run in self._periodic_runs.values():
            slots = tuple(run.get("slots", ()))
            if not slots:
                continue
            end = int(run.get("phase_end", run.get("total", 0)))
            if end - int(run.get("next", 0)) > periods * len(slots):
                return True
        return False

    def _crossed_external_or_opaque(self, previous, stop_at: float) -> bool:
        """Did a real external task act inside the observed interval?

        Blocked downstream tasks (``waiting > 0``) and passive lease holders
        (started, zero remaining work, a release beyond ``stop_at``) consume no
        pipe rate and cannot change the next interval, so they are never
        mistaken for interference.  A non-periodic task that actually completed
        in ``(previous_now, now]`` is tracked by a plain counter and always
        sets the flag.
        """
        if self._external_completions > previous["external_completions"]:
            return True
        if self._releases:
            stop_at = min(stop_at, self._releases[0][0])
        for task in self._tasks.values():
            if task.event.sequence in self._periodic_tasks:
                continue
            if task.waiting > 0:
                # Still blocked on an unresolved predecessor: no pipe rate.
                continue
            if (task.start is not None and not task.remaining
                    and task.release > stop_at
                    and (self.now >= task.start + task.latency
                         or task.start + task.latency > stop_at)):
                # Passive lease holder: cannot release before ``stop_at``.
                continue
            if bool(task.remaining) or bool(task.acquire) or bool(task.free):
                return True
        return False

    def _periodic_stop(self, horizon: float) -> float:
        if self._releases:
            horizon = min(horizon, self._releases[0][0])
        for key in self._latency_active:
            if key in self._periodic_tasks:
                continue
            task = self._tasks[key]
            anchor = task.first_service if task.latency_from_first_service else task.start
            if anchor is not None and anchor + task.latency > self.now:
                horizon = min(horizon, anchor + task.latency)
        return horizon

    def _periodic_epoch_stalled(self, detector) -> bool:
        """Return True once the active periodic epoch cannot match within ``max_lag``."""
        epoch_key = tuple(
            (run_id, run.get("phase_ordinal", 0), run.get("phase_end", 0))
            for run_id, run in sorted(self._periodic_runs.items())
            if run.get("slots")
        )
        miss_budget = _PERIODIC_EPOCH_MISS_MULTIPLIER * detector.history_limit
        any_retired = any(
            int(run.get("finished", 0)) > 0
            for run in self._periodic_runs.values()
            if run.get("slots")
        )
        issue_limit = (
            detector.max_lag
            if any_retired
            else _PERIODIC_COLD_ISSUE_MULTIPLIER * detector.max_lag
        )
        if epoch_key != self._periodic_stall_epoch:
            if (self._periodic_stall_obs >= miss_budget
                    or self._periodic_no_issue_streak > issue_limit):
                self._periodic_last = None
            self._periodic_stall_epoch = epoch_key
            self._periodic_stall_obs = 0
            self._periodic_last_next_sum = -1
            self._periodic_no_issue_streak = 0
        if (self._periodic_stall_obs >= miss_budget
                or self._periodic_no_issue_streak > issue_limit):
            return True
        next_sum = sum(
            int(run.get("next", 0))
            for run in self._periodic_runs.values()
            if run.get("slots")
        )
        if next_sum == self._periodic_last_next_sum:
            self._periodic_no_issue_streak += 1
            return self._periodic_no_issue_streak > issue_limit
        self._periodic_last_next_sum = next_sum
        self._periodic_no_issue_streak = 0
        return False

    def _periodic_observe(self, stop_at: float):
        """Read-only periodic frontier observation on the settled Engine."""
        if not self._periodic_runs:
            return None
        if any(run.get("indexed") for run in self._periodic_runs.values()):
            # G1 selects an indexed catalog by absolute panel. The ordered
            # phase cursor and cycle correspondence are only proven for the
            # legacy ordered template; G3 owns an exact indexed preflight.
            # Until then the ordinary exact Engine advance is authoritative.
            self._periodic_last = None
            self._periodic_detector = None
            return None
        # The detector validates the whole structural tuple recursively. Keep
        # this observation bounded to an active cohort that can actually be
        # compared; an oversized ordinary graph keeps the detailed Engine path
        # instead of paying O(live tasks) validation on every step.
        if len(self._tasks) > _MAX_PERIODIC_LIVE_TASKS:
            return None
        if (len(self._periodic_runs) > _MAX_PERIODIC_RUNS
                or sum(len(tuple(run.get("slots", ())))
                       for run in self._periodic_runs.values()) > _MAX_PERIODIC_LIVE_SLOTS):
            return None
        detector = self._periodic_detector
        if detector is None:
            try:
                from ..engine_periodic import CycleDetector
            except ImportError:
                return None
            detector = CycleDetector()
            self._periodic_detector = detector
        if not self._progressable_periodic_run(detector.evidence_periods):
            # No run can reach the next comparable period, so no skip is
            # possible here.  Keep the measured interval baseline: the next
            # real observation then aggregates this whole span exactly.
            return None
        if self._periodic_epoch_stalled(detector):
            return None
        if (self._periodic_last is not None
                and self._external_completions
                > self._periodic_last["external_completions"]):
            # Discard the crossed interval so the next observation can start
            # fresh once external completions settle.
            self._periodic_last = None
            return None
        previous = self._periodic_last
        stop_at = self._periodic_stop(stop_at)
        landmark = (self._periodic_stall_epoch, self._periodic_last_next_sum)
        clock_steps = None
        if previous is not None:
            clock_steps = previous["clock_steps"]
            clock = self._periodic_clock_step
            if (clock_steps is not None and len(clock_steps) < detector.max_lag
                    and clock is not None and clock[0] == self.advance_steps == previous["advance_steps"] + 1
                    and not self._crossed_external_or_opaque(previous, stop_at)):
                clock_steps.append(clock[1])
            else:
                clock_steps = None
            # A valid cycle must issue tiles. Observe full structural evidence
            # at issue landmarks and retain the arithmetic of intermediate steps.
            if landmark == previous["landmark"]:
                previous["clock_steps"] = clock_steps
                previous["advance_steps"] = self.advance_steps
                return None
        from ..engine_periodic import Cursor, IntervalDelta, SettledFrontier
        epoch = self._memory_epoch()
        state = self._periodic_state(epoch)
        cursors = tuple(
            Cursor(run_id, run.get("phase_ordinal", 0),
                   run.get("phase_end", 0), run.get("next", 0),
                   run.get("finished", 0), tuple(run.get("slots", ())))
            for run_id, run in sorted(self._periodic_runs.items()))
        interval = None
        if clock_steps:
            issued = {
                run_id: (run.get("next", 0) - previous["runs"].get(
                    run_id, (0, 0))[0],
                    run.get("finished", 0) - previous["runs"].get(
                    run_id, (0, 0))[1])
                for run_id, run in self._periodic_runs.items()
            }
            interval = IntervalDelta(
                dt=max(0.0, self.now - previous["now"]),
                issued_retired_by_run=issued,
                sequence_count=max(0, self._sequence - previous["sequence"]),
                service_by_resource=_counter_delta(self.serviced, previous["serviced"]),
                busy_by_resource=_counter_delta(self.busy_ms, previous["busy"]),
                deliveries_by_counter=_counter_delta(self.delivered, previous["delivered"]),
                hbm_accounting=_counter_delta(self._hbm_accounting_serviced, previous["hbm_accounting"]),
                clock_steps=tuple(clock_steps),
                allocation_event_count=max(
                    0, self.allocation_event_count
                    - previous["allocation_event_count"]),
                per_unit_peaks=dict(self.peak_allocated),
                logical_completions=max(
                    0, self.completed_count - previous["completed"]),
                crossed_external_or_opaque=False,
            )
        frame = SettledFrontier(
            now=self.now, epoch=epoch, normalized_state=state,
            cursors=cursors, sequence=self._sequence)
        skip = self._periodic_detector.observe(
            frame, interval, stop_at=stop_at)
        if skip is not None:
            self._periodic_stall_obs = 0
            self._periodic_last_next_sum = -1
            self._periodic_no_issue_streak = 0
        else:
            self._periodic_stall_obs += 1
        self._periodic_last = {
            "now": self.now, "sequence": self._sequence,
            "serviced": dict(self.serviced), "busy": dict(self.busy_ms),
            "delivered": dict(self.delivered), "hbm_accounting": dict(self._hbm_accounting_serviced),
            "advance_steps": self.advance_steps, "clock_steps": [], "landmark": landmark,
            "allocation_event_count": self.allocation_event_count,
            "completed": self.completed_count,
            "external_completions": self._external_completions,
            "runs": {run_id: (run.get("next", 0), run.get("finished", 0))
                     for run_id, run in self._periodic_runs.items()},
        }
        return skip

    def _apply_periodic_skip(self, skip) -> None:
        """Apply one helper-approved skip to this Engine's live window.

        The helper's ``delta`` is per period; ``total_delta`` applies the exact
        cycle count. Public Event objects/sequences stay identical. Live tasks
        keep their remaining work (the frontier is homogeneous over the skipped
        cycles) but their start/release times advance by the skipped span;
        already completed predecessors keep their original completion time.
        """
        total = skip.total_delta()
        dt = float(total.dt)
        if dt < 0 or not math.isfinite(dt):
            raise ExecutionError("periodic_skip_period_invalid")
        start_before = self.now
        if skip.delta.clock_steps:
            self.now = _advance_periodic_clock_exact(self.now, skip.delta.clock_steps, skip.cycles)
            dt = self.now - start_before
        else:
            self.now += dt
        wave_rec = getattr(self, "_wave_recorder", None)
        if wave_rec is not None:
            per_cycle = getattr(skip, "delta", None)
            cycle_dt = float(per_cycle.dt) if per_cycle is not None else dt
            cycle_count = int(getattr(skip, "cycles", 1) or 1)
            svc_pat, ho_pat = wave_rec.extract_cycle_pattern(
                start_before - cycle_dt, start_before,
            )
            if svc_pat or ho_pat:
                wave_rec.on_period(
                    start_before, cycle_dt, cycle_count,
                    template_id="periodic_skip", state_digest="skip",
                    service_pattern=svc_pat, handoff_pattern=ho_pat,
                )
            elif hasattr(wave_rec, "_service_by_pipe") and any(
                float(amt) > 0.0
                for amt in dict(getattr(per_cycle, "busy_by_resource", None) or ()).values()
            ):
                raise ExecutionError("periodic_skip_missing_proved_cycle_pattern")
        self._sequence += int(total.sequence_count)
        self.allocation_event_count += int(total.allocation_event_count)
        self.completed_count += int(total.logical_completions)
        for resource, amount in dict(total.service_by_resource).items():
            if resource in self.serviced:
                self.serviced[resource] += float(amount)
        for resource, amount in dict(total.busy_by_resource).items():
            if resource in self.busy_ms:
                self.busy_ms[resource] += float(amount)
        for counters, delta in ((self.delivered, total.deliveries_by_counter),
                                (self._hbm_accounting_serviced, total.hbm_accounting)):
            for name, amount in delta.items():
                counters[name] = counters.get(name, 0.0) + amount
        for pool, peak in dict(total.per_unit_peaks).items():
            self.peak_allocated[pool] = max(
                self.peak_allocated.get(pool, 0.0), float(peak))
        for correspondence in skip.correspondence.runs:
            run = self._periodic_runs.get(correspondence.run_id)
            if run is None:
                continue
            run["phase_ordinal"] = correspondence.phase_ordinal
            run["phase_end"] = correspondence.phase_end_exclusive
            run["next"] = correspondence.next_issue_after
            run["finished"] = correspondence.retired_after
            offset = correspondence.tile_offset
            run["live_tiles"] = {
                tile + offset for tile in run.get("live_tiles", ())}
            run["retire_tiles"] = {
                seq: tile + offset
                for seq, tile in run.get("retire_tiles", {}).items()}
            run["slots"] = tuple(sorted(
                tile - correspondence.retired_after
                for tile in run["live_tiles"]))
        for sequence, (run_id, tile, ordinal, kind) in tuple(
                self._periodic_tasks.items()):
            correspondence = next(
                (item for item in skip.correspondence.runs
                 if item.run_id == run_id), None)
            if correspondence is not None:
                self._periodic_tasks[sequence] = (
                    run_id, tile + correspondence.tile_offset, ordinal, kind)
        for key in self._periodic_tasks:
            task = self._tasks[key]
            if task.start is not None:
                task.start += dt
                task.event.started = task.start
            if task.first_service is not None:
                task.first_service += dt
            if task.ready_at is not None:
                task.ready_at += dt
        self._periodic_last = None


from contextlib import contextmanager
from contextvars import ContextVar
import ctypes
import heapq
import math
import time
from typing import Any, Mapping, Sequence

from ..runtime.coupled import coupled_rates, _coupled_pair_fast
from ..ownership import unit_rates
from .arbiter import component_rates, _fast_eligible_task
from ._engine_hbm_bus import _hbm_entry_done_since_bind, _sync_task_hbm_accounting
from .contracts import (
    AllocationBinding, AllocationKey, BindingFrame, EventKey, OccurrenceKey,
    PortBinding, Submission,
)
from .engine_catalog import _EngineCatalogMixin
from .engine_lifecycle import _EngineLifecycleMixin
from .engine_pair_cached import _EnginePairCachedMixin
from .engine_pair_fast import _EnginePairFastMixin
from .engine_prepare import _EnginePrepareMixin
from .engine_solo_step import _EngineSoloStepMixin, _emit_wave_service
from .engine_submit import _EngineSubmitMixin
from .engine_types import (
    Allocation, Event, ExecutionError, PreparedStage, Resource,
    _EMPTY_CAPS, _EMPTY_FROZENSET, _EMPTY_METADATA, _EMPTY_PLACEMENT,
    _EMPTY_TASK, _FULL_FINITE_PREFLIGHT, _PreparedTask, _Task,
    _binding_for, _chain_callbacks, _compute_pool_unit_masks,
    _memory_binding, _structural_value, _units_to_mask,
)
from .state import RuntimeState


def _stop_window(now: float, horizon: float, releases, latency_active, tasks) -> float:
    dt = horizon - now
    if releases:
        rel_dt = releases[0][0] - now
        if rel_dt < dt:
            dt = rel_dt
    if latency_active:
        if len(latency_active) == 1:
            ltask = tasks[next(iter(latency_active))]
            anchor = ltask.first_service if ltask.latency_from_first_service else ltask.start
            if anchor is not None and anchor + ltask.latency > now:
                lat_dt = anchor + ltask.latency - now
                if lat_dt < dt:
                    dt = lat_dt
        else:
            for lkey in latency_active:
                ltask = tasks[lkey]
                anchor = ltask.first_service if ltask.latency_from_first_service else ltask.start
                if anchor is not None and anchor + ltask.latency > now:
                    lat_dt = anchor + ltask.latency - now
                    if lat_dt < dt:
                        dt = lat_dt
    return dt


def _completed_targets(until_tuple, until_two_0, until_two_1, first):
    if until_two_0 is not None:
        d0 = until_two_0.time is not None
        d1 = until_two_1.time is not None
        if d0 and d1:
            return until_tuple, True
        if d0 or d1:
            complete = (until_two_0,) if d0 else (until_two_1,)
            if first:
                return complete, True
        else:
            complete = ()
    elif first:
        complete = ()
        for event in until_tuple:
            if event.time is not None:
                return tuple(ev for ev in until_tuple if ev.time is not None), True
    else:
        complete = tuple(ev for ev in until_tuple if ev.time is not None)
        if len(complete) == len(until_tuple):
            return complete, True
    return complete, False


DEADLINE_POLL_MASK = 255
#: Cooperative per-candidate/outer sweep deadline.  Every pricing loop polls
#: this value; no signal handler, thread or subprocess is involved.
ACTIVE_DEADLINE_MONO: ContextVar[float | None] = ContextVar(
    "mindie_sim_deadline_mono", default=None)


def check_active_deadline() -> None:
    deadline = ACTIVE_DEADLINE_MONO.get()
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError("simulation_deadline_exceeded")


_CACHED_MIXED_PLAN_KIND = 3
_CACHED_MIXED_REM_IDX = 6
_TRIPLE_ACTIVE_COUNT = 3

_NATIVE_EPOCH_STEP_LIMIT = 1 << 20
_STALL_NAME_LIMIT = 300
_SERVICE_ROUNDING_ULPS = 4

#: Default no-progress guard of one Engine advance (structural, not physics).
DEFAULT_ZERO_PROGRESS_LIMIT = 100_000


class Engine(
    _EnginePrepareMixin, _EngineSubmitMixin, _EngineLifecycleMixin,
    _EnginePeriodicMixin, _EnginePairCachedMixin, _EnginePairFastMixin,
    _EngineSoloStepMixin, _EngineCatalogMixin, RuntimeState,
):
    """One clock and one mutable causal root; see module invariants."""

    def __init__(
        self, resources: Sequence[Resource], *,
        pools: Mapping[str, float] | None = None, trace_limit: int = 0,
        pool_units: Mapping[str, int] | None = None,
        zero_progress_limit: int = DEFAULT_ZERO_PROGRESS_LIMIT,
        backend: str | None = None, accel_eager_attach: bool = False,
    ):
        if not isinstance(zero_progress_limit, int) or isinstance(
                zero_progress_limit, bool) or zero_progress_limit < 1:
            raise ExecutionError("zero_progress_limit_invalid")
        resource_map = {}
        for resource in resources:
            if not isinstance(resource, Resource):
                raise ExecutionError("resource_requires_Resource")
            if resource.name in resource_map:
                raise ExecutionError(f"resource_already_declared:{resource.name}")
            resource_map[resource.name] = resource
        super().__init__(
            resources=resource_map, pools=pools, trace_limit=trace_limit,
            pool_units=pool_units, zero_progress_limit=zero_progress_limit)
        self._memory_context: dict | None = None
        self._memory_context_fast: tuple | None = None
        self._stage_id_prepared: dict[int, tuple[Any, _PreparedTask]] = {}
        self._stage_proto_prepared: dict[tuple, tuple] = {}
        self._compiled_programs: dict[int, tuple[Any, Any]] = {}
        self._seen_floor_peaks: set[int] = set()
        self._empty_task: _PreparedTask = _EMPTY_TASK
        self.memory_ledger_error: str | None = None
        from .accel_bridge import make_backend
        self._accel, self._accel_telemetry = make_backend(
            backend, eager_attach=bool(accel_eager_attach))
        self._fifo_live = 0
        self._inline_blocked = 0
        self._fifo_names: frozenset[str] = frozenset()
        self._fifo_resource_count = -1
        self._native_slots: dict[int, int] = {}
        self._native_seqs: set[int] = set()
        self._native_active = 0
        self._native_seeded = False
        self._wave_recorder = None
        self._admission_memory_view = None
        self._allocation_versions: dict[str, int] = {}
        self._resource_version: int = 0
        self._hbm_bus_aliases: dict[str, Any] = {}
        self._hbm_bus_epoch: int = 0
        self._hbm_acct_live: int = 0
        self._hbm_accounting_serviced: dict[str, float] = {}
        self._periodic_clock_step: tuple[int, float] | None = None
        self._pending_hbm_retrofits: dict[str, tuple[Any, bool]] = {}
        self._discover_hbm_bus_aliases()
        if self._accel is not None and getattr(self._accel, "mode", "") == "c":
            self._accel.try_attach(self)

    def attach_wave_recorder(self, recorder) -> None:
        """Attach a read-only ``WaveRecorder`` observer without altering schedules."""
        self._wave_recorder = recorder
        if recorder is not None:
            for rname in self.resources:
                recorder.register_pipe(rname)

    def detach_wave_recorder(self):
        """Detach and return any active ``WaveRecorder`` observer."""
        recorder = self._wave_recorder
        self._wave_recorder = None
        return recorder

    @property
    def admission_memory_view(self):
        """Return the canonical per-die ``AdmissionMemoryView`` attached to this Engine."""
        return self._admission_memory_view

    def attach_admission_memory_view(self, view) -> None:
        """Attach the canonical per-die ``AdmissionMemoryView`` for L2 state tracking."""
        self._admission_memory_view = view

    def bump_allocation_version(self, allocation_id: str) -> int:
        """Advance and return the monotonic version counter for ``allocation_id``."""
        key = str(allocation_id)
        ver = self._allocation_versions.get(key, 0) + 1
        self._allocation_versions[key] = ver
        return ver

    @property
    def accel_backend(self) -> str:
        if self._accel is None:
            return self._accel_telemetry["accel_backend"]
        return self._accel.status()["accel_backend"]

    @property
    def accel_reason(self):
        if self._accel is None:
            return self._accel_telemetry.get("backend_reason")
        return self._accel.status().get("backend_reason")

    def accel_status(self) -> dict:
        status = dict(self._accel_telemetry)
        if self._accel is not None:
            status.update(self._accel.status())
        status["accel_backend"] = self.accel_backend
        status["backend_reason"] = self.accel_reason
        status["fallback_reason"] = status.get("fallback_reason", self.accel_reason)
        # ``advance_steps`` counts every integration step of this Engine; the
        # kernel reports how many of them it performed.  Their difference is the
        # honest Python step count, never a build-availability guess.
        native = int(status.get("native_steps") or 0)
        status["native_steps"] = native
        status["python_steps"] = max(0, int(self.advance_steps) - native)
        status.setdefault("requested", self._accel_telemetry.get("requested"))
        return status

    def _frontier_next_release(self) -> float:
        return self._releases[0][0] if self._releases else math.inf

    def _native_deliver(self, sequence: int) -> None:
        if self._accel is not None and sequence in self._native_slots:
            self._accel.deliver.append(sequence)

    def _complete_native(self, task: _Task, moment: float) -> None:
        self._complete(task, moment=moment, native=True)

    def fifo_resource_names(self) -> frozenset[str]:
        """Declared ``fifo`` resource names, memoized on the resource count."""
        if len(self.resources) != self._fifo_resource_count:
            self._fifo_names = frozenset(
                name for name, resource in self.resources.items()
                if resource.policy == "fifo")
            self._fifo_resource_count = len(self.resources)
        return self._fifo_names

    def task_blocks_native(self, task: _Task) -> bool:
        """True when one live row can never be mirrored by the kernel.

        A cheap, exact-at-submission predicate: allocation bookkeeping needs a
        Python start boundary, an ordinary completion callback needs a Python
        completion, and an occurrence row owns Python metrics.  A zero count is
        the necessary condition for adopting a frontier, so a positive one lets
        every later re-adoption attempt return in constant time instead of
        auditing thousands of rows.
        """
        if (task.acquire or task.free or task.allocation_demands
                or task.start_callback is not None or task.latency_from_first_service):
            return True
        if task.callback is not None and not task.native_hook:
            return True
        return bool(task.metadata) and "occurrence" in task.metadata

    def task_uses_fifo(self, task: _Task) -> bool:
        """True when this row can be ordered by a FIFO resource."""
        names = self.fifo_resource_names()
        if not names:
            return False
        demand = task.demand
        if not demand:
            return False
        for name in demand:
            if name in names:
                return True
        return False

    def extend(self, resources=(), pools=None, pool_units=None, *, _discover_hbm: bool = True):
        """Add a disjoint physical pool to the same virtual clock.

        A second ``Execution``/``Memory`` binding may re-declare the exact
        resources and pools an adopted Engine already owns.  Re-declaration is
        accepted only when the declaration is identical; a different rate,
        policy, unit count or capacity is still a real conflict and raises.
        """
        added_resource = False
        for resource in resources:
            existing = self.resources.get(resource.name)
            if existing is not None:
                if existing != resource:
                    raise ExecutionError(
                        f"resource_redeclared_differently:{resource.name}")
                continue
            self.resources[resource.name] = resource
            self.serviced[resource.name] = self.busy_ms[resource.name] = 0
            added_resource = True
        for name, capacity in (pools or {}).items():
            units = (pool_units or {}).get(name, 1)
            if name in self.pools:
                if self.pools[name] != capacity or self.pool_units[name] != units:
                    raise ExecutionError(f"pool_redeclared_differently:{name}")
                continue
            self.pools[name] = capacity
            self.pool_units[name] = units
            self.allocated[name] = self.peak_allocated[name] = 0
            self._pool_unit_masks[name] = 0
            self._alloc_state_version += 1
        self._seen_floor_peaks.clear()
        self._clear_prepared_caches()
        if added_resource:
            self._resource_version += 1
            if _discover_hbm:
                self._discover_hbm_bus_aliases()
        if (added_resource and self._accel is not None
                and self._accel.enabled
                and not self._tasks and not self._native_slots):
            self._accel.enabled = False
            self._accel.try_attach(self)

    def bind_memory(self, memory, *, pools: Sequence[str] = ("hbm",)) -> bool:
        """Idempotently bind the single default ledger for this Engine.

        A different already-bound ledger is never overwritten; callers that
        need PD endpoints on one Engine must submit with ``memory=`` so each
        real task carries its own ledger. Returns True only when this Engine
        now exposes ``memory`` as the default.
        """
        if memory is None:
            return self.memory is None
        if self.memory is not None and self.memory is not memory:
            return False
        already = (self.memory is memory
                   and getattr(memory, "runtime_state", None) is self)
        self.memory = memory
        self.memory_pools.update(str(pool) for pool in pools)
        self._memory_pools_frozen = frozenset(self.memory_pools)
        self._memory_ledgers[id(memory)] = memory
        if already:
            # Re-binding the same ledger to the same root is a no-op: the
            # capacity domain is already adopted and claimed here.
            return True
        binder = getattr(memory, "bind_runtime", None)
        if binder is not None:
            try:
                binder(self)
            except ValueError as error:
                # A declared envelope whose fixed allocations exceed the
                # physical die is over-subscribed, not an invalid runtime.
                # Memory keeps its own real oom accounting; skip only the
                # capacity domain so construction is not turned into a ledger
                # failure. Any other binding conflict is still raised.
                if "pool_capacity_invalid" not in str(error):
                    raise
                self.memory_ledger_error = str(error)
        return True

    @contextmanager
    def memory_scope(self, memory=None, *, replica: int = 0, stage: int = 0,
                     pools: Sequence[str] = ("hbm",), bindings=None):
        """Attach a real ledger/binding to tasks submitted inside this block.

        ``program.start``/``Stage.submit`` are T01/T10 interfaces without a
        memory parameter. This explicit, short-lived scope is captured into
        each ``_Task`` at submission, so later advance steps and a second
        conductor on the same Engine cannot rewrite an earlier task's ledger.
        """
        previous = self._memory_context
        previous_fast = self._memory_context_fast
        ledger = memory if memory is not None else self.memory
        if ledger is not None:
            self._memory_ledgers[id(ledger)] = ledger
        if pools.__class__ is frozenset:
            pools_frozen = pools
        else:
            pool_key = tuple(sorted({str(pool) for pool in pools}))
            if not pool_key:
                pools_frozen = _EMPTY_FROZENSET
            else:
                pools_frozen = self._engine_pool_sets.get(pool_key)
                if pools_frozen is None:
                    pools_frozen = frozenset(pool_key)
                    self._engine_pool_sets[pool_key] = pools_frozen
                    self._engine_pool_set_ids.add(id(pools_frozen))
        norm_bindings = dict(bindings) if bindings else {}
        rep_i = int(replica)
        stg_i = int(stage)
        context = {
            "memory": ledger,
            "pools": pools_frozen,
            "replica": rep_i, "stage": stg_i,
            "bindings": norm_bindings,
        }
        self._memory_context = context
        self._memory_context_fast = (
            ledger, pools_frozen, norm_bindings if norm_bindings else None,
            rep_i, stg_i,
        )
        try:
            yield context
        finally:
            self._memory_context = previous
            self._memory_context_fast = previous_fast

    def memory_grant(self, name: str) -> dict | None:
        """Return the live T08 binding for one granted allocation name."""
        record = self._memory_grants.get(str(name))
        if record is None:
            return None
        return {
            "name": str(name), "replica": record["replica"],
            "ledger": record["ledger"], "binding": dict(record["binding"]),
        }

    def _rates(self) -> dict[str, dict[int, float]]:
        return component_rates(
            self.resources, self._tasks, self._active,
            coupled_solver=coupled_rates,
            coupled_active=self._coupled_active,
            placed_active=self._placed_active,
            allow_native=self._accel is not None)

    def advance(
        self, *, until: Sequence[Event] = (), horizon: float = math.inf,
        first: bool = True, max_steps: int | None = None,
    ) -> tuple[Event, ...]:
        """Advance to a target completion or arrival without skipping releases."""
        if horizon < self.now:
            raise ExecutionError("time_cannot_move_backwards")
        if max_steps is not None and (not isinstance(max_steps, int)
                                      or isinstance(max_steps, bool)
                                      or max_steps < 1):
            raise ExecutionError("max_steps_invalid")
        until_tuple = until if until.__class__ is tuple else tuple(until)
        n_until = len(until_tuple)
        until_one = until_tuple[0] if n_until == 1 else None
        until_two_0 = until_tuple[0] if n_until == 2 else None
        until_two_1 = until_tuple[1] if n_until == 2 else None
        self.advance_calls += 1
        steps = 0
        active = self._active
        tasks = self._tasks
        releases = self._releases
        latency_active = self._latency_active
        busy_ms = self.busy_ms
        serviced = self.serviced
        completable = self._completable
        completable_seen = self._completable_seen
        resources = self.resources
        pair_cache = self._pair_rate_cache
        _settle = self._settle
        _step_solo_fast = self._step_solo_fast
        while True:
            if (steps & DEADLINE_POLL_MASK) == 0 and ACTIVE_DEADLINE_MONO.get() is not None:
                check_active_deadline()
            accel = self._accel
            if accel is not None and accel.start_conflict(self, until_tuple):
                accel.disable(self, "start_boundary")
                accel.made_progress = True
            _settle()
            if until_one is not None:
                if until_one.time is not None:
                    return until_tuple
                complete = ()
            elif n_until:
                complete, reached = _completed_targets(
                    until_tuple, until_two_0, until_two_1, first)
                if reached:
                    return complete
            else:
                complete = ()
            if not tasks and self._promises:
                raise ExecutionError("execution_has_unresolved_programs")
            now = self.now
            if not tasks or now >= horizon:
                return complete
            if max_steps is not None and steps >= max_steps:
                raise ExecutionError("execution_step_budget_exhausted")
            if self._placed_active or self._fifo_live:
                self._settle_asymmetric_cohorts()
            if self._periodic_runs:
                skipped = self._periodic_observe(horizon)
                if skipped is not None:
                    self._apply_periodic_skip(skipped)
                    steps += 1
                    continue
            accel = self._accel
            if accel is not None and (self._native_seqs or accel.pending
                                      or accel.made_progress):
                consumed = accel.advance_epoch(
                    self, horizon,
                    (max_steps - steps) if max_steps is not None else _NATIVE_EPOCH_STEP_LIMIT,
                    until_tuple)
                if consumed:
                    steps += consumed
                    continue
                if accel.made_progress:
                    continue
            if not self._periodic_runs:
                n_act = len(active)
                if n_act == 0:
                    dt = horizon - now
                    if releases:
                        rel_dt = releases[0][0] - now
                        if rel_dt < dt:
                            dt = rel_dt
                    if not math.isfinite(dt):
                        names = ",".join(t.event.name for t in tasks.values())
                        raise ExecutionError(f"execution_stalled:{names[:_STALL_NAME_LIMIT]}")
                    next_time = now + dt
                    if dt > 0 and next_time == now:
                        next_time = math.nextafter(now, math.inf)
                    self.advance_steps += 1
                    steps += 1
                    self.now = next_time
                    continue
                wave_rec = self._wave_recorder
                solo_key = 0
                solo_task = None
                if n_act == 1:
                    solo_key = next(iter(active))
                    solo_task = tasks[solo_key]
                elif n_act == 2:
                    it_a = iter(active)
                    k0, k1 = next(it_a), next(it_a)
                    t0, t1 = tasks[k0], tasks[k1]
                    rem0, rem1 = t0.remaining, t1.remaining
                    if not rem0:
                        solo_key, solo_task = k1, t1
                    elif not rem1:
                        solo_key, solo_task = k0, t0
                    elif (t0.priority == t1.priority
                          and _fast_eligible_task(resources, t0)
                          and _fast_eligible_task(resources, t1)):
                        next_pair = self._step_active_pair(
                            k0, t0, k1, t1, now, horizon, releases, latency_active,
                            tasks, resources, pair_cache, wave_rec)
                        if next_pair is not None:
                            self.advance_steps += 1
                            steps += 1
                            self.now = next_pair
                            continue
                elif n_act == _TRIPLE_ACTIVE_COUNT:
                    next_triple = self._step_active_triple(
                        active, now, horizon, releases, latency_active, tasks,
                        resources, wave_rec)
                    if next_triple is not None:
                        self.advance_steps += 1
                        steps += 1
                        self.now = next_triple
                        continue
                if solo_task is not None and _fast_eligible_task(resources, solo_task):
                    next_solo = _step_solo_fast(solo_key, solo_task, now, horizon, max_steps)
                    if next_solo is not None:
                        if solo_task.hbm_accounting:
                            _sync_task_hbm_accounting(self, solo_task)
                        if solo_task.deliveries:
                            self._update_task_deliveries(solo_task)
                        self.advance_steps += 1
                        steps += 1
                        self.now = next_solo
                        continue
            next_time = self._step_shared_service(
                now, horizon, releases, latency_active, tasks, busy_ms, serviced,
                completable, completable_seen)
            self.advance_steps += 1
            steps += 1
            self.now = next_time

    def _step_active_pair(
        self, k0, t0, k1, t1, now, horizon, releases, latency_active,
        tasks, resources, pair_cache, wave_rec,
    ) -> float | None:
        rem0, rem1 = t0.remaining, t1.remaining
        p0, p1 = t0.prepared, t1.prepared
        if (p0 is not None and p1 is not None
                and not t0.service and not t1.service
                and len(rem0) == len(p0.demand)
                and len(rem1) == len(p1.demand)):
            hit = pair_cache.get((id(p0), id(p1)))
            if hit is not None and hit[0] is p0 and hit[1] is p1:
                cplan = hit[2]
                if cplan.__class__ is tuple:
                    if (cplan[0] != _CACHED_MIXED_PLAN_KIND
                            or (rem1 if cplan[1] else rem0)[cplan[_CACHED_MIXED_PLAN_KIND]]
                            == cplan[_CACHED_MIXED_REM_IDX]):
                        dt = _stop_window(now, horizon, releases, latency_active, tasks)
                        watched = (
                            ((k0, t0, dict(t0.service)), (k1, t1, dict(t1.service)))
                            if (wave_rec is not None or t0.deliveries or t1.deliveries
                                or t0.hbm_accounting or t1.hbm_accounting) else ()
                        )
                        next_time = self._step_cached_pair(cplan, k0, t0, k1, t1, now, dt)
                        if watched:
                            self._emit_fast_step_observers(wave_rec, watched, now, next_time)
                        return next_time
                    cp_fast = _coupled_pair_fast(resources, k0, t0, k1, t1)
                elif cplan is False:
                    cp_fast = _coupled_pair_fast(resources, k0, t0, k1, t1)
                else:
                    cp_fast = None
            else:
                cp_fast = _coupled_pair_fast(resources, k0, t0, k1, t1)
                self._cache_pair_plan(p0, p1, t0, cp_fast)
        else:
            cp_fast = _coupled_pair_fast(resources, k0, t0, k1, t1)
        if cp_fast is not None:
            dt = _stop_window(now, horizon, releases, latency_active, tasks)
            watched = (
                ((k0, t0, dict(t0.service)), (k1, t1, dict(t1.service)))
                if (wave_rec is not None or t0.deliveries or t1.deliveries
                    or t0.hbm_accounting or t1.hbm_accounting) else ()
            )
            next_time = self._step_pair_fast(cp_fast, now, dt)
            if watched:
                self._emit_fast_step_observers(wave_rec, watched, now, next_time)
            return next_time
        return None

    def _step_active_triple(
        self, active, now, horizon, releases, latency_active, tasks, resources, wave_rec,
    ) -> float | None:
        it_a = iter(active)
        k0, k1, k2 = next(it_a), next(it_a), next(it_a)
        t0, t1, t2 = tasks[k0], tasks[k1], tasks[k2]
        rem0, rem1, rem2 = t0.remaining, t1.remaining, t2.remaining
        if not (t0.priority == t1.priority == t2.priority
                    and _fast_eligible_task(resources, t0)
                    and _fast_eligible_task(resources, t1)
                    and _fast_eligible_task(resources, t2)):
            return None
        if not rem0 or not rem1 or not rem2:
            if not rem0:
                ka, ta, kb, tb = k1, t1, k2, t2
            elif not rem1:
                ka, ta, kb, tb = k0, t0, k2, t2
            else:
                ka, ta, kb, tb = k0, t0, k1, t1
            cp_fast = _coupled_pair_fast(resources, ka, ta, kb, tb)
            if cp_fast is not None:
                dt = _stop_window(now, horizon, releases, latency_active, tasks)
                watched = (
                    ((ka, ta, dict(ta.service)), (kb, tb, dict(tb.service)))
                    if (wave_rec is not None or ta.deliveries or tb.deliveries
                        or ta.hbm_accounting or tb.hbm_accounting) else ()
                )
                next_time = self._step_pair_fast(cp_fast, now, dt)
                if watched:
                    self._emit_fast_step_observers(wave_rec, watched, now, next_time)
                return next_time
            return None
        ks, ts = 0, None
        if not t0.coupled and len(rem0) == 1:
            rcand = next(iter(rem0))
            if rcand not in rem1 and rcand not in rem2:
                ks, ts, rs = k0, t0, rcand
                ka, ta, kb, tb = k1, t1, k2, t2
        if ts is None and not t1.coupled and len(rem1) == 1:
            rcand = next(iter(rem1))
            if rcand not in rem0 and rcand not in rem2:
                ks, ts, rs = k1, t1, rcand
                ka, ta, kb, tb = k0, t0, k2, t2
        if ts is None and not t2.coupled and len(rem2) == 1:
            rcand = next(iter(rem2))
            if rcand not in rem0 and rcand not in rem1:
                ks, ts, rs = k2, t2, rcand
                ka, ta, kb, tb = k0, t0, k1, t1
        if ts is not None:
            cp_fast = _coupled_pair_fast(resources, ka, ta, kb, tb)
            if cp_fast is not None:
                prep_s = ts.prepared
                if prep_s is not None and prep_s.solo_rates is not None:
                    rate_s = prep_s.solo_rates[rs]
                else:
                    res_s = resources[rs]
                    if res_s.policy == "fifo":
                        rate_s = res_s.rate
                    elif ts.placement and rs in ts.placement:
                        rate_s = res_s.rate / res_s.units * len(ts.placement[rs])
                    else:
                        rate_s = res_s.rate
                    if ts.caps and rs in ts.caps and ts.caps[rs] < rate_s:
                        rate_s = ts.caps[rs]
                dt = _stop_window(now, horizon, releases, latency_active, tasks)
                watched = (
                    ((ka, ta, dict(ta.service)), (kb, tb, dict(tb.service)),
                     (ks, ts, dict(ts.service)))
                    if (wave_rec is not None or ta.deliveries or tb.deliveries or ts.deliveries
                        or ta.hbm_accounting or tb.hbm_accounting or ts.hbm_accounting) else ()
                )
                next_time = self._step_pair_fast(cp_fast, now, dt, ks, ts, rs, rate_s)
                if watched:
                    self._emit_fast_step_observers(wave_rec, watched, now, next_time)
                return next_time
        return None

    def _step_shared_service(
        self, now, horizon, releases, latency_active, tasks, busy_ms, serviced,
        completable, completable_seen,
    ) -> float:
        rates = self._rates()
        dt = _stop_window(now, horizon, releases, latency_active, tasks)
        stop_dt = dt
        for name, consumers in rates.items():
            for key, rate in consumers.items():
                if rate == 0.0:
                    raise ExecutionError(f"service_rate_underflow:{name}")
                work_dt = tasks[key].remaining[name] / rate
                if work_dt < dt:
                    dt = work_dt
        if not math.isfinite(dt):
            names = ",".join(task.event.name for task in tasks.values())
            raise ExecutionError(f"execution_stalled:{names[:_STALL_NAME_LIMIT]}")
        if self._periodic_runs:
            self._periodic_clock_step = (self.advance_steps + 1, dt) if dt < stop_dt else None
        next_time = now + dt
        if dt > 0 and next_time == now:
            next_time = math.nextafter(now, math.inf)
        wave_rec = self._wave_recorder
        for name, consumers in rates.items():
            busy_ms[name] += dt
            for key, rate in consumers.items():
                task = tasks[key]
                remaining = task.remaining[name]
                scaled = dt * rate
                if scaled >= remaining:
                    used, residual = remaining, 0.0
                else:
                    residual = remaining - scaled
                    if residual <= _SERVICE_ROUNDING_ULPS * math.ulp(remaining):
                        used, residual = remaining, 0.0
                    else:
                        used = scaled
                if task.first_service is None and used > 0.0:
                    task.first_service = now
                service = task.service
                service[name] = service.get(name, 0.0) + used
                serviced[name] += used
                if wave_rec is not None and dt > 0.0:
                    _emit_wave_service(
                        wave_rec, key, name, task.placement, now, next_time, used,
                    )
                if residual:
                    task.remaining[name] = residual
                else:
                    del task.remaining[name]
                    if not task.remaining and task.latency <= 0.0:
                        completable_seen.add(key)
                        heapq.heappush(completable, key)
                if task.hbm_accounting:
                    _sync_task_hbm_accounting(self, task)
                if task.deliveries:
                    self._update_task_deliveries(task)
        return next_time

    def service_of(self, event: Event) -> dict[str, float]:
        """Read-only per-task service; live native slots export accrued work."""
        task = self._tasks.get(event.sequence)
        if task is None:
            return {}
        slot = self._native_slots.get(event.sequence)
        if slot is not None and self._accel is not None and self._accel.arena is not None:
            names = tuple(task.remaining)
            n = len(names)
            buf = (ctypes.c_double * max(2 * n, 1))()
            got = int(self._accel.arena.library.msim_ctx_export(
                self._accel.arena.ctx, slot, buf, 2 * n))
            if got == n:
                merged = dict(task.service)
                for idx, name in enumerate(names):
                    merged[name] = merged.get(name, 0.0) + float(buf[n + idx])
                if task.hbm_accounting:
                    live_rem = {name: float(buf[idx]) for idx, name in enumerate(names)}
                    credited = task.hbm_credited or {}
                    for entry in task.hbm_accounting:
                        base = entry[0]
                        done_since = _hbm_entry_done_since_bind(live_rem, entry)
                        delta = done_since - credited.get(base, 0.0)
                        if delta > 0.0:
                            merged[base] = merged.get(base, 0.0) + delta
                return merged
        merged = dict(task.service)
        if task.split_children:
            for child_ev in task.split_children:
                if not child_ev.done:
                    for rn, amt in self.service_of(child_ev).items():
                        merged[rn] = merged.get(rn, 0.0) + amt
        return merged

    def _accel_coverage(self) -> tuple[int, int, str | None]:
        if self._accel is not None:
            accel = self._accel
            if accel.enabled and accel.native_steps > 0:
                reason = None
            elif accel.enabled:
                reason = "no_native_window"
            else:
                reason = accel.reason or accel.disabled_reason or "not_attached"
            return accel.native_calls, accel.native_steps, reason
        telemetry = self._accel_telemetry
        return (telemetry.get("native_calls", 0),
                telemetry.get("native_steps", 0),
                telemetry.get("backend_reason"))

    def summary(self) -> dict:
        native_calls, native_steps, fallback_reason = self._accel_coverage()
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
            "blocked_tasks": len(self._ready_heap),
            "advance_calls": self.advance_calls, "advance_steps": self.advance_steps,
            "python_steps": max(0, int(self.advance_steps) - int(
                (self.accel_status().get("native_steps") or 0))),
            "allocation_events": self.allocation_event_count,
            "retained_allocation_events": len(self.allocation_trace),
            "retained_trace_events": len(self.trace), "trace_dropped": self.trace_dropped,
            "accel_backend": self.accel_backend,
            "accel_reason": self.accel_reason,
            "native_calls": native_calls, "native_steps": native_steps,
            "fallback_reason": fallback_reason,
        }


__all__ = [
    "ACTIVE_DEADLINE_MONO", "Allocation", "AllocationBinding", "AllocationKey",
    "BindingFrame", "Engine", "Event", "EventKey", "ExecutionError",
    "OccurrenceKey", "PortBinding", "PreparedStage", "Resource", "Submission",
    "check_active_deadline", "unit_rates",
    "_EMPTY_CAPS", "_EMPTY_FROZENSET", "_EMPTY_METADATA", "_EMPTY_PLACEMENT",
    "_EMPTY_TASK", "_FULL_FINITE_PREFLIGHT", "_PreparedTask", "_Task",
    "_binding_for", "_chain_callbacks", "_compute_pool_unit_masks",
    "_memory_binding", "_structural_value", "_units_to_mask",
]
