"""Periodic steady-state frontier observation and verified cycle skipping.

[Role] Build tile-index-independent structural snapshots of settled Engine
  frontiers and apply CycleDetector-approved skips without altering Event order.
[Invariants] Any external/opaque work, memory-epoch change or indexed catalog
  invalidates skipping and preserves exact step-by-step execution.
[Timeouts & Locks] Bounded structural inspection (`<= 512` live tasks); no I/O.
[Side-Effect Policy] Advances clock, counters and live window cursors on valid skip.
"""
from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
import math

from .engine_types import ExecutionError, _structural_value

_PERIODIC_EPOCH_MISS_MULTIPLIER = 2
_PERIODIC_COLD_ISSUE_MULTIPLIER = 2


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
        if len(self._tasks) > 512:
            return None
        if (len(self._periodic_runs) > 64
                or sum(len(tuple(run.get("slots", ())))
                       for run in self._periodic_runs.values()) > 128):
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
            # The detector only accepts an interval with no external/opaque
            # completion (``crossed_external_or_opaque``).  Since the baseline
            # was recorded, real external work has completed, so this interval
            # can never become evidence.  Skipping the observation is a sound
            # O(1) rejection, not a lost proof.
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
        from ..engine_periodic.frontier import _advance_periodic_clock
        if skip.delta.clock_steps:
            self.now = _advance_periodic_clock(self.now, skip.delta.clock_steps, skip.cycles)
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
            if not svc_pat:
                cycle_busy = (
                    dict(per_cycle.busy_by_resource)
                    if per_cycle is not None else dict(total.busy_by_resource)
                )
                svc_pat = tuple(
                    (rname, 0.0, float(busy)) for rname, busy in cycle_busy.items() if busy > 0.0
                )
            wave_rec.on_period(
                start_before, cycle_dt, cycle_count,
                template_id="periodic_skip", state_digest="skip",
                service_pattern=svc_pat, handoff_pattern=ho_pat,
            )
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
