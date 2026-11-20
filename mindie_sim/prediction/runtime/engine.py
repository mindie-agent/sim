"""Unique causal Engine: clock, event frontier and transaction settlement.

[Role] Execute the one selected physical catalog as real causal work on shared
  physical resources. The Engine is the only writer of its RuntimeState clock,
  frontier, event sequences and transaction commits.
[Invariants] Service conserves work. Completion wakes only actual successors and
  releases declared allocations. FIFO is non-preemptive; fair service is capped
  max-min over the actual physical units. No second runtime, script replay or
  old-executor fallback exists. Resource units are work per millisecond.
[Timeouts & Locks] Finite virtual events only; no threads, sleeps or I/O. A
  bounded settle/step budget reports zero progress instead of spinning.
[Side-Effect Policy] Mutates the inherited RuntimeState and the Event handles it
  creates. A callback failure aborts advance; the engine is not promised reusable
  after a contract error.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import heapq
import math
from typing import Any, Mapping, Sequence

from ..coupled import coupled_rates, _coupled_pair_fast
from ..ownership import unit_rates
from .arbiter import component_rates, _fast_eligible_task
from .contracts import (
    AllocationBinding, AllocationKey, BindingFrame, EventKey, OccurrenceKey,
    PortBinding, Submission,
)
from .engine_catalog import _EngineCatalogMixin
from .engine_lifecycle import _EngineLifecycleMixin
from .engine_pair_cached import _EnginePairCachedMixin
from .engine_pair_fast import _EnginePairFastMixin
from .engine_periodic_mixin import _EnginePeriodicMixin
from .engine_prepare import _EnginePrepareMixin
from .engine_solo_step import _EngineSoloStepMixin
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


DEADLINE_POLL_MASK = 255
_CACHED_MIXED_PLAN_KIND = 3
_CACHED_MIXED_REM_IDX = 6
_TRIPLE_ACTIVE_COUNT = 3

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

    def add_hbm_bus(self, bus) -> tuple[Resource, Resource]:
        """Declare a die's 2-row directional HBM bus resources on this Engine."""
        rows = (
            Resource(bus.row_r, 1.0, unit="row_fraction", policy="fair", units=1),
            Resource(bus.row_w, 1.0, unit="row_fraction", policy="fair", units=1),
        )
        self.extend(rows)
        return rows

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
        if self._accel is not None:
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

    def extend(self, resources=(), pools=None, pool_units=None):
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
        self._compiled_programs.clear()
        self._stage_id_prepared.clear()
        self._stage_proto_prepared.clear()
        self._pair_rate_cache.clear()
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
        pools_frozen = (
            pools if pools.__class__ is frozenset
            else frozenset(str(pool) for pool in pools)
        )
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
            placed_active=self._placed_active)

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
        _step_pair_fast = self._step_pair_fast
        _step_cached_pair = self._step_cached_pair
        _cache_pair_plan = self._cache_pair_plan
        _step_solo_fast = self._step_solo_fast
        _isfinite = math.isfinite
        _ulp = math.ulp
        _nextafter = math.nextafter
        _heappush = heapq.heappush
        _inf = math.inf
        while True:
            if (steps & DEADLINE_POLL_MASK) == 0:
                from ..evaluation import ACTIVE_DEADLINE_MONO, check_active_deadline
                if ACTIVE_DEADLINE_MONO.get() is not None:
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
            elif until_two_0 is not None:
                d0 = until_two_0.time is not None
                d1 = until_two_1.time is not None
                if d0 and d1:
                    return until_tuple
                if d0 or d1:
                    complete = (until_two_0,) if d0 else (until_two_1,)
                    if first:
                        return complete
                else:
                    complete = ()
            elif first and n_until:
                complete = ()
                for event in until_tuple:
                    if event.time is not None:
                        return tuple(ev for ev in until_tuple if ev.time is not None)
            elif n_until:
                complete = tuple(ev for ev in until_tuple if ev.time is not None)
                if len(complete) == n_until:
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
                self._split_asymmetric_cohorts()
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
                    (max_steps - steps) if max_steps is not None else (1 << 20),
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
                    if not _isfinite(dt):
                        names = ",".join(t.event.name for t in tasks.values())
                        raise ExecutionError(f"execution_stalled:{names[:300]}")
                    next_time = now + dt
                    if dt > 0 and next_time == now:
                        next_time = _nextafter(now, _inf)
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
                                            or (rem1[cplan[_CACHED_MIXED_PLAN_KIND]] if cplan[1] else rem0[cplan[_CACHED_MIXED_PLAN_KIND]]) == cplan[_CACHED_MIXED_REM_IDX]):
                                        dt = _stop_window(now, horizon, releases, latency_active, tasks)
                                        watched = (
                                            ((k0, t0, dict(t0.service)), (k1, t1, dict(t1.service)))
                                            if (wave_rec is not None or t0.deliveries or t1.deliveries) else ()
                                        )
                                        next_time = _step_cached_pair(cplan, k0, t0, k1, t1, now, dt)
                                        if watched:
                                            self._emit_fast_step_observers(wave_rec, watched, now, next_time)
                                        self.advance_steps += 1
                                        steps += 1
                                        self.now = next_time
                                        continue
                                    cp_fast = _coupled_pair_fast(resources, k0, t0, k1, t1)
                                elif cplan is False:
                                    cp_fast = _coupled_pair_fast(resources, k0, t0, k1, t1)
                                else:
                                    cp_fast = None
                            else:
                                cp_fast = _coupled_pair_fast(resources, k0, t0, k1, t1)
                                _cache_pair_plan(p0, p1, t0, cp_fast)
                        else:
                            cp_fast = _coupled_pair_fast(resources, k0, t0, k1, t1)
                        if cp_fast is not None:
                            dt = _stop_window(now, horizon, releases, latency_active, tasks)
                            watched = (
                                ((k0, t0, dict(t0.service)), (k1, t1, dict(t1.service)))
                                if (wave_rec is not None or t0.deliveries or t1.deliveries) else ()
                            )
                            next_time = _step_pair_fast(cp_fast, now, dt)
                            if watched:
                                self._emit_fast_step_observers(wave_rec, watched, now, next_time)
                            self.advance_steps += 1
                            steps += 1
                            self.now = next_time
                            continue
                elif n_act == _TRIPLE_ACTIVE_COUNT:
                    it_a = iter(active)
                    k0, k1, k2 = next(it_a), next(it_a), next(it_a)
                    t0, t1, t2 = tasks[k0], tasks[k1], tasks[k2]
                    rem0, rem1, rem2 = t0.remaining, t1.remaining, t2.remaining
                    if (t0.priority == t1.priority == t2.priority
                            and _fast_eligible_task(resources, t0)
                            and _fast_eligible_task(resources, t1)
                            and _fast_eligible_task(resources, t2)):
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
                                    if (wave_rec is not None or ta.deliveries or tb.deliveries) else ()
                                )
                                next_time = _step_pair_fast(cp_fast, now, dt)
                                if watched:
                                    self._emit_fast_step_observers(wave_rec, watched, now, next_time)
                                self.advance_steps += 1
                                steps += 1
                                self.now = next_time
                                continue
                        else:
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
                                        if (wave_rec is not None or ta.deliveries or tb.deliveries or ts.deliveries) else ()
                                    )
                                    next_time = _step_pair_fast(cp_fast, now, dt, ks, ts, rs, rate_s)
                                    if watched:
                                        self._emit_fast_step_observers(wave_rec, watched, now, next_time)
                                    self.advance_steps += 1
                                    steps += 1
                                    self.now = next_time
                                    continue
                if solo_task is not None and _fast_eligible_task(resources, solo_task):
                    next_solo = _step_solo_fast(solo_key, solo_task, now, horizon, max_steps)
                    if next_solo is not None:
                        if solo_task.deliveries:
                            self._update_task_deliveries(solo_task)
                        self.advance_steps += 1
                        steps += 1
                        self.now = next_solo
                        continue
            rates = self._rates()
            dt = _stop_window(now, horizon, releases, latency_active, tasks)
            for name, consumers in rates.items():
                for key, rate in consumers.items():
                    if rate == 0.0:
                        raise ExecutionError(f"service_rate_underflow:{name}")
                    work_dt = tasks[key].remaining[name] / rate
                    if work_dt < dt:
                        dt = work_dt
            if not _isfinite(dt):
                names = ",".join(task.event.name for task in tasks.values())
                raise ExecutionError(f"execution_stalled:{names[:300]}")
            next_time = now + dt
            if dt > 0 and next_time == now:
                next_time = _nextafter(now, _inf)
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
                        if residual <= 4 * _ulp(remaining):
                            used, residual = remaining, 0.0
                        else:
                            used = scaled
                    if task.first_service is None and used > 0.0:
                        task.first_service = now
                    service = task.service
                    service[name] = service.get(name, 0.0) + used
                    serviced[name] += used
                    if wave_rec is not None and dt > 0.0:
                        pl_units = task.placement.get(name) if task.placement else None
                        owner_u = pl_units[0] if pl_units else 0
                        wave_rec.on_service(key, name, owner_u, now, next_time, used)
                    if residual:
                        task.remaining[name] = residual
                    else:
                        del task.remaining[name]
                        if not task.remaining and task.latency <= 0.0:
                            completable_seen.add(key)
                            _heappush(completable, key)
                    if task.deliveries:
                        self._update_task_deliveries(task)
            self.advance_steps += 1
            steps += 1
            self.now = next_time

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
                return merged
        return dict(task.service)

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
    "Allocation", "AllocationBinding", "AllocationKey", "BindingFrame",
    "Engine", "Event", "EventKey", "ExecutionError", "OccurrenceKey",
    "PortBinding", "PreparedStage", "Resource", "Submission", "unit_rates",
    "_EMPTY_CAPS", "_EMPTY_FROZENSET", "_EMPTY_METADATA", "_EMPTY_PLACEMENT",
    "_EMPTY_TASK", "_FULL_FINITE_PREFLIGHT", "_PreparedTask", "_Task",
    "_binding_for", "_chain_callbacks", "_compute_pool_unit_masks",
    "_memory_binding", "_structural_value", "_units_to_mask",
]
