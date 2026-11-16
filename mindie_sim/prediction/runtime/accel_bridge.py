"""ctypes bridge: backend choice, kernel lifecycle and epoch dispatch.

[Role] Select ``python`` / ``c`` / ``auto`` once per Engine, validate the loaded
  library, own one per-Engine kernel context, and translate a callback-free
  numeric epoch into the canonical Engine.
[Invariants] No scheduler, cache or persistent service state lives here.  The
  Python Engine keeps clock authority, events, callbacks, grants and the memory
  ledger.  A non-zero kernel status is fatal (``KernelFault``) and is never
  replayed through Python.  ``auto`` keeps tiny workloads on pure Python with
  zero native calls and zero compilation.  Under ``auto`` a missing toolchain or
  unloadable library is an explicit ``python_fallback``; under explicit ``c`` it
  is a structured error.
[Timeouts & Locks] One process-local library memo guarded by the cache
  coordinator's cross-process lock across build and ``ctypes.CDLL``; normal
  calls are bounded by the caller's step budget, while an output drain uses
  budget zero and is bounded by the fixed wire buffers (each call publishes at
  least one record).  The dispatch loop adds no threads, sleeps or I/O after
  attach.
[Side-Effect Policy] Loads the cached immutable library and mutates the Engine
  only at the same points as the Python reference.
"""
from __future__ import annotations

import ctypes
import math
import os
import weakref

from . import accel_build, accel_cache, accel_loader
from .accel_loader import AccelUnavailable, load_library
from .accel_sync import (
    adopt_frontier, deliver_slots, drain_output, effective_units, explicit_units,
    finish_handback, flush_pending, reclaim_arena, solver_capacity_exceeded,
    sync_back, task_usage,
)
from .accel_tables import (
    ABI_VERSION, Arena, B_IDLE, B_OUTPUT, E_ARITH, E_STALL, E_STATE, INT32_MAX,
    KernelFault, MAX_L, MAX_OWN, MAX_R, MAX_T, MAX_U,
    _free_ctx_box, is_inert_trace_entry, needs_external_start,
    pure_task, resource_table, wide_integer,
)

#: ``auto`` keeps smaller Engines on pure Python: zero build and marshalling.
AUTO_MIN_TASKS = 64

#: Callback-free numeric windows run in C unless MINDIE_SIM_ACCEL_EPOCH=0.
EPOCH_ENABLED = os.environ.get("MINDIE_SIM_ACCEL_EPOCH", "1") != "0"
def load_accel_cdll(status: dict | None = None):
    """Build (when needed) and map under one cache lock."""
    return accel_loader.load_accel_cdll(status)
def resolve_mode(requested: str | None) -> str:
    value = requested
    if value is None:
        value = os.environ.get("MINDIE_SIM_ACCEL")
    if value is None:
        value = os.environ.get("MINDIE_SIM_ENGINE_BACKEND")
    if value is None:
        value = "auto"
    value = str(value).strip().lower()
    if value in {"python", "py", "off", "0"}:
        return "python"
    if value in {"c", "c11", "native", "on", "1"}:
        return "c"
    if value in {"auto", ""}:
        return "auto"
    raise ValueError(f"engine_backend_invalid:{value}")
class NativeBackend:
    """One Engine's immutable library handle, kernel context and telemetry."""
    __slots__ = (
        "library", "arena", "status_dict", "backend", "reason", "abi",
        "native_calls", "native_steps", "native_boundaries", "fallbacks",
        "mode", "enabled", "res_names", "pending", "deliver", "drops",
        "external", "max_tasks", "live", "seeded", "disabled_reason", "error",
        "made_progress", "engine_slots", "dormant", "handing_back", "_finalizer",
        "deps", "usage", "links", "units", "edges", "storage", "epoch_links",
        "epoch_units", "epoch_storage", "epoch_tasks", "epoch_edges",
        "_engine_finalizer", "__weakref__", "applying", "callback_dirty",
        "coupled_seqs", "pending_handback", "draining", "aborted_publication",
        "needs_settle", "fair_prio", "has_fair_res",
    )
    def __init__(self, status: dict | None, mode: str = "auto") -> None:
        self.library = None
        self.arena = None
        self.status_dict = dict(status or {})
        self.backend = "python_fallback"
        self.reason = "auto_deferred" if mode == "auto" else None
        self.abi = ABI_VERSION
        self.native_calls = 0
        self.native_steps = 0
        self.native_boundaries = 0
        self.fallbacks = 0
        self.mode = mode
        self.enabled = False
        self.res_names: tuple[str, ...] = ()
        self.pending: list = []
        self.deliver: list[int] = []
        self.drops: list[int] = []
        self.external: list[int] = []
        self.max_tasks = 0
        self.live = 0
        self.seeded = False
        self.disabled_reason = "not_attached"
        self.error: str | None = None
        self.made_progress = False
        self.engine_slots: dict[int, int] = {}
        self.dormant = not EPOCH_ENABLED
        self.handing_back = False
        self._finalizer = None
        self._engine_finalizer = None
        self.deps: dict[int, tuple] = {}
        self.usage: dict[int, tuple[int, int, int, int]] = {}
        self.links = self.units = self.edges = self.storage = 0
        self.epoch_links = self.epoch_units = self.epoch_tasks = 0
        self.epoch_edges = self.epoch_storage = 0
        self.applying = False
        self.callback_dirty = False
        self.coupled_seqs: set[int] = set()
        self.pending_handback = False
        self.draining = False
        self.aborted_publication = False
        self.needs_settle = False
        self.fair_prio: int | None = None
        self.has_fair_res = False
    # -- lifecycle -----------------------------------------------------
    def try_attach(self, engine) -> bool:
        if self.enabled:
            return True
        if self.mode == "auto" and engine._sequence < AUTO_MIN_TASKS:
            self.disabled_reason = "auto_tiny_workload"
            return False
        if self.arena is None:
            try:
                # Keep the cache lock across the hit/build and the in-process
                # mapping: a concurrent publisher may add a 17th generation
                # and prune, but it cannot unlink this path before CDLL maps it.
                with accel_cache.cache_lock(accel_build.cache_root()):
                    try:
                        status = accel_build.build_accel()
                    except TimeoutError:
                        raise
                    except (OSError, RuntimeError) as error:
                        status = {"ok": False,
                                  "reason": f"build_unavailable:{error}"}
                    if not status.get("ok"):
                        self.reason = (status.get("reason")
                                       or "native_build_unavailable")
                        self.status_dict = dict(status)
                        self.disabled_reason = self.reason
                        if self.mode == "c":
                            raise AccelUnavailable(self.reason, status)
                        return False
                    self.status_dict = dict(status)
                    self.library = load_library(status)
            except TimeoutError:
                raise
            except (AccelUnavailable, OSError, MemoryError, KernelFault) as error:
                return self._attach_failed(error)
            try:
                self.arena = Arena(self.library)
                self._finalizer = weakref.finalize(self, Arena.close, self.arena)
            except TimeoutError:
                raise
            except (AccelUnavailable, OSError, MemoryError, KernelFault) as error:
                return self._attach_failed(error)
        else:
            self.arena.reset()
            self.seeded = False
        rates, units, policy = resource_table(engine)
        if len(rates) < 1 or len(rates) > MAX_R:
            self.disabled_reason = "resource_count"
            self.dormant = True
            return False
        for rate, unit in zip(rates, units):
            if not math.isfinite(rate) or rate <= 0.0 or not 1 <= unit <= MAX_U:
                self.disabled_reason = "resource_shape"
                self.dormant = True
                return False
        try:
            self.arena.configure(rates, units, policy)
        except (KernelFault, MemoryError) as error:
            return self._attach_failed(error)
        self.res_names = tuple(engine.resources)
        self.has_fair_res = any(p == 0 for p in policy)
        self.fair_prio = None
        self.enabled = True
        self.dormant = False
        self.backend = "c_native" if EPOCH_ENABLED else "c_native_probe"
        self.reason = None if EPOCH_ENABLED else "epoch_disabled_by_env"
        self.disabled_reason = None if EPOCH_ENABLED else "epoch_disabled_by_env"
        self._bind_engine(engine)
        return True
    def _bind_engine(self, engine) -> None:
        """Register a cycle-free finalizer that only retains ``(lib, box)``."""
        if self.arena is None:
            return
        if self._engine_finalizer is not None:
            self._engine_finalizer.detach()
        self._engine_finalizer = weakref.finalize(
            engine, _free_ctx_box, self.library, self.arena.ctx_box)
    def _attach_failed(self, error) -> bool:
        reason = getattr(error, "reason", None) or str(error)
        self.reason = self.disabled_reason = reason
        if self._engine_finalizer is not None:
            self._engine_finalizer.detach()
            self._engine_finalizer = None
        if self._finalizer is not None:
            self._finalizer()
        self._finalizer = None
        if self.arena is not None:
            self.arena.close()
        self.arena = None
        self.library = None
        if self.mode == "c":
            raise AccelUnavailable(reason, self.status_dict)
        return False
    def disable(self, engine, reason: str = "mixed_frontier") -> None:
        """Return every kernel-owned row to the reference frontier."""
        was_enabled = self.enabled
        self.enabled = False
        self.dormant = True
        self.disabled_reason = reason
        self.needs_settle = False
        if was_enabled:
            self.pending_handback = True
            finish_handback(self, engine)
    def _sync_back(self, engine) -> None:
        """Exact hand-back owned by :mod:`accel_sync`."""
        sync_back(self, engine)
    # -- eligibility ---------------------------------------------------
    def _adopt(self, engine, skip: int) -> bool:
        """Mirror the unstarted pure frontier when ``auto`` attaches."""
        return adopt_frontier(self, engine, skip)
    def eligible(self, engine, task) -> bool:
        if self.handing_back:
            return False
        if self.dormant:
            return False
        if engine._periodic_runs:
            self.disabled_reason = "periodic_frontier"
            return False
        if wide_integer(task):
            if self.enabled:
                self.disable(engine, "wide_integer")
            else:
                self.disabled_reason = "wide_integer"
            return False
        if not pure_task(engine, task):
            return False
        if (self.enabled and self.has_fair_res and self.fair_prio is not None
                and task.priority != self.fair_prio):
            self.disable(engine, "mixed_priority")
            return False
        if not self.enabled:
            if engine._sequence < AUTO_MIN_TASKS:
                return False
            if engine._active or engine._native_slots:
                self.disabled_reason = "auto_busy"
                return False
            if not self.try_attach(engine):
                return False
            if (self.mode == "auto"
                    and not self._adopt(engine, task.event.sequence)):
                reason = ("capacity_exceeded"
                          if self.disabled_reason == "capacity_exceeded"
                          else "auto_mixed_frontier")
                self.disable(engine, reason)
                return False
        new_units = effective_units(engine, task)
        new_storage = explicit_units(task)
        coupled = bool(self.coupled_seqs) or task.coupled
        if (len(engine._native_seqs) + 1 > MAX_T
                or self.epoch_links + len(task.remaining) > MAX_L
                or self.epoch_storage + new_storage > INT32_MAX
                or self.epoch_edges + task.waiting > MAX_OWN
                or solver_capacity_exceeded(
                    len(engine._native_seqs) + 1,
                    self.links + len(task.remaining),
                    self.units + new_units, coupled)):
            self.disable(engine, "capacity_exceeded")
            return False
        return True
    @staticmethod
    def needs_external_start(task) -> bool:
        return needs_external_start(task)
    def queue(self, engine, event, task, pending) -> None:
        pending = tuple(pending or ())
        self.pending.append((event, task, pending))
        self.deps[event.sequence] = pending
        use = task_usage(engine, task, pending)
        self.usage[event.sequence] = use
        self.links += use[0]
        self.units += use[1]
        self.edges += use[2]
        self.storage += use[3]
        self.epoch_links += use[0]
        self.epoch_units += use[1]
        self.epoch_edges += use[2]
        self.epoch_storage += use[3]
        self.epoch_tasks += 1
        if task.coupled:
            self.coupled_seqs.add(event.sequence)
        if self.has_fair_res and self.fair_prio is None:
            self.fair_prio = task.priority
        self.engine_slots = engine._native_slots
        self.live += 1
        if self.live > self.max_tasks:
            self.max_tasks = self.live
    def note_external_predecessors(self, engine, pending) -> None:
        if not pending or not engine._native_slots:
            return
        slots = engine._native_slots
        for predecessor in pending:
            slot = slots.get(predecessor.sequence)
            if slot is not None:
                self.external.append(slot)
    def release(self, engine, sequence: int) -> None:
        slot = engine._native_slots.pop(sequence, None)
        if slot is None:
            return
        use = self.usage.pop(sequence, None)
        if use is not None:
            self.links -= use[0]
            self.units -= use[1]
            self.edges -= use[2]
            self.storage -= use[3]
        self.deps.pop(sequence, None)
        self.coupled_seqs.discard(sequence)
        self.drops.append(slot)
        if self.live:
            self.live -= 1
    def start_conflict(self, engine, until=()) -> bool:
        """True when a native start cannot wake a Python observer mid-epoch."""
        if not self.enabled or not engine._native_seqs:
            return False
        stopping = {event.sequence for event in until if not event.done}
        for observers in engine._start_waiters.values():
            for observer in observers:
                if (observer.sequence in stopping or observer._waiters
                        or observer._callbacks):
                    return True
        for seq in engine._native_seqs:
            t_obj = engine._tasks.get(seq)
            obs = t_obj.start_observer if t_obj is not None else None
            if obs is not None and (obs.sequence in stopping or obs._waiters or obs._callbacks):
                return True
        return False
    def _mixed_reason(self) -> str:
        return "auto_mixed_frontier" if self.mode == "auto" else "mixed_frontier"
    def on_callback_registered(self, engine, event) -> None:
        """a live mirrored row acquired a Python completion observer.

        The kernel can no longer run through that row's completion: it cannot
        invoke the callback at the exact instant and it would publish the epoch
        clock past it.  Hand the whole frontier back before the next native
        step, deferring the transfer when this registration happens inside
        ``_apply`` (the current completion batch must finish first).
        """
        if not self.enabled or event.done:
            return
        if event.sequence not in engine._native_seqs:
            return
        if self.applying:
            self.callback_dirty = True
            return
        self.disable(engine, self._mixed_reason())
        self.made_progress = True
    def _sync_counters(self, engine) -> None:
        """publish counters before another call can fail (no prefix too)."""
        if self.arena is None:
            return
        busy, serv = self.arena.counters()
        for i, name in enumerate(self.res_names):
            engine.busy_ms[name] = busy[i]
            engine.serviced[name] = serv[i]
    def _apply_prefix(self, engine, error: KernelFault) -> None:
        """publish the kernel-accepted prefix of a failing epoch."""
        prefix = getattr(error, "prefix", None)
        if prefix is None or not getattr(error, "has_prefix", False):
            # even an accepted-nothing failure must publish the native
            # counters already earned by an earlier successful call.
            self._sync_counters(engine)
            return
        now, _boundary, steps, ncomp, nstart = prefix
        self.native_calls += 1
        self.native_steps += steps
        self._sync_counters(engine)
        engine.now = now
        engine.advance_steps += steps
        self._apply(engine, nstart, ncomp)
        finish_handback(self, engine)
        self.needs_settle = False
        self.made_progress = bool(steps or ncomp or nstart)
    # -- epoch ---------------------------------------------------------
    def _reclaim(self, engine) -> None:
        reclaim_arena(self, engine)
    def _flush(self, engine) -> None:
        flush_pending(self, engine)
    def _deliver(self) -> None:
        deliver_slots(self)
    def activate(self, engine, sequence: int) -> None:
        slot = engine._native_slots.get(sequence)
        if slot is None:
            return
        buf = (ctypes.c_int32 * 1)(slot)
        rc = self.library.msim_ctx_activate(
            self.arena.ctx, buf, 1, engine.now, self.arena.st, 8)
        if rc != 0:
            raise KernelFault(int(self.arena.st[1]), "activate")
    def _trace_blocks_epoch(self, engine) -> bool:
        """a bounded trace retaining a release-sensitive row stays Python.

        ``Engine._complete`` appends the name/owner/metadata/demand/service and
        placement references to ``engine.trace``.  If a native epoch integrates
        past the retained row's own completion instant, a later completion
        evicts it at the batch-final clock and runs user finalization with a
        future ``engine.now``/counters.  Hand the frontier back instead; Python
        evicts the entry at its exact completion instant.
        """
        if not engine._trace_maxlen:
            return False
        if not any(not is_inert_trace_entry(row) for row in engine._trace_dq):
            return False
        self.disable(engine, "trace_retention")
        self.made_progress = True
        return True
    def _handback_mixed(self, engine) -> None:
        """Disable a frontier that is no longer entirely kernel-mirrored."""
        self.disable(engine, "mixed_frontier")
        self.made_progress = True
        self.needs_settle = False
    def _settle_instant(self, engine, force: bool = False) -> None:
        """Publish same-instant admission after a callback completion.

        The kernel holds awakened successors until Python published them.
        """
        rounds = 0
        while True:
            if self.dormant or not self.enabled:
                self.needs_settle = False
                return
            if not force and not self.needs_settle and not self.pending:
                return
            force = False
            if self._trace_blocks_epoch(engine):
                self.needs_settle = False
                return
            if (len(engine._native_seqs) != len(engine._tasks)
                    or tuple(engine.resources) != self.res_names):
                self._handback_mixed(engine)
                return
            self.engine_slots = engine._native_slots
            self._flush(engine)
            if not engine._native_slots:
                self.needs_settle = False
                return
            if len(engine._native_slots) != len(engine._tasks):
                self._handback_mixed(engine)
                return
            if not self.seeded:
                self.arena.seed([engine.busy_ms[n] for n in self.res_names],
                                [engine.serviced[n] for n in self.res_names])
                self.seeded = True
            self.needs_settle = False  # consume the previous round's hold.
            try:
                now, boundary, steps, ncomp, nstart = self.arena.run(
                    engine.now, engine.now,
                    engine._frontier_next_release(), 1)
            except KernelFault as error:
                self._apply_prefix(engine, error)
                self._translate_fault(error)
            self.native_calls += 1
            self.native_steps += steps
            engine.now, engine.advance_steps = now, engine.advance_steps + steps
            self._sync_counters(engine)
            self._apply(engine, nstart, ncomp)
            if boundary == B_OUTPUT and (self.enabled or self.pending_handback):
                steps += drain_output(self, engine)
            finish_handback(self, engine)
            self.made_progress = self.made_progress or bool(steps or ncomp or nstart)
            if not self.enabled or self.pending_handback:
                self.needs_settle = False
                return
            if not self.needs_settle and not self.pending:
                return
            rounds += 1
            if rounds > engine.zero_progress_limit:
                from .engine import ExecutionError
                raise ExecutionError("execution_zero_progress")
    def admit(self, engine) -> None:
        """Flush queued rows and start every ready row at ``engine.now``."""
        self._settle_instant(engine, force=True)
    def _translate_fault(self, error: KernelFault) -> None:
        """Map the kernel arithmetic status to Python's exact underflow error."""
        st = self.arena.st if self.arena is not None else None
        index = (int(st[7]) - 1) if st is not None else -1
        if error.code == E_ARITH and 0 <= index < len(self.res_names):
            from .engine import ExecutionError
            raise ExecutionError(
                f"service_rate_underflow:{self.res_names[index]}")
        raise error
    def advance_epoch(self, engine, horizon: float, budget: int, until) -> int:
        """Run one callback-free epoch; returns the consumed step count."""
        self.made_progress = False
        if self.dormant or not self.enabled:
            return 0
        if not engine._native_seqs and not self.pending:
            return 0
        if self._trace_blocks_epoch(engine):
            return 0
        if (len(engine._native_seqs) != len(engine._tasks)
                or tuple(engine.resources) != self.res_names):
            # Sync back when Python-owned tasks or resource changes mix the frontier.
            self.disable(engine, "mixed_frontier")
            self.made_progress = True
            return 0
        self.engine_slots = engine._native_slots
        self._flush(engine)
        if not engine._native_slots:
            return 0
        if len(engine._native_slots) != len(engine._tasks):
            self.disable(engine, "mixed_frontier")
            self.made_progress = True
            return 0
        if not self.seeded:
            self.arena.seed([engine.busy_ms[n] for n in self.res_names],
                            [engine.serviced[n] for n in self.res_names])
            self.seeded = True
        stop = [engine._native_slots[event.sequence] for event in until
                if not event.done and event.sequence in engine._native_slots]
        if stop:
            self.arena.call("msim_ctx_mark_external", stop)
        try:
            now, boundary, steps, ncomp, nstart = self.arena.run(
                engine.now, horizon, engine._frontier_next_release(), budget)
        except KernelFault as error:
            self._apply_prefix(engine, error)
            self._translate_fault(error)
        self.native_calls += 1
        self.native_steps += steps
        engine.now = now
        engine.advance_steps += steps
        self._sync_counters(engine)
        self._apply(engine, nstart, ncomp)
        if boundary == B_OUTPUT and (self.enabled or self.pending_handback):
            steps += drain_output(self, engine)
        finish_handback(self, engine)
        if boundary:
            self.native_boundaries += 1
        self.made_progress = bool(steps or ncomp or nstart)
        if self.needs_settle or self.pending:
            self._settle_instant(engine)
        if not self.made_progress and boundary == B_IDLE and engine._native_seqs:
            self.fallbacks += 1
            raise KernelFault(E_STALL, "idle", f"mirrored={len(engine._native_slots)}")
        return steps
    def _apply(self, engine, nstart: int, ncomp: int) -> None:
        """Publish one kernel result batch exactly once."""
        tasks = engine._tasks
        slots = engine._native_slots
        self.applying = True
        try:
            for seq, moment in self.arena.starts(nstart):
                task = tasks.get(seq)
                if task is None:
                    continue
                task.start = moment
                task.event.started = moment
                engine._active.add(seq)
                engine._native_active += 1
                # The arbiter fast paths read these live counters; a kernel
                # start must publish the same state ``_start_ready`` would have.
                if task.coupled:
                    engine._coupled_active += 1
                if task.placement:
                    engine._placed_active += 1
                if task.latency > 0.0:
                    engine._latency_active.add(seq)
                if len(tasks) > engine.peak_tasks:
                    engine.peak_tasks = len(tasks)
                for observer in engine._start_waiters.pop(seq, ()):
                    observer.started = observer.time = moment
                    engine._promises.pop(observer.sequence, None)
                    engine._wake(observer)
                    for callback in observer._callbacks or ():
                        callback(observer)
                    observer._callbacks = None
            for seq, moment, svc_start in self.arena.completions(ncomp):
                use = self.usage.pop(seq, None)
                if use is not None:
                    self.links -= use[0]
                    self.units -= use[1]
                    self.edges -= use[2]
                    self.storage -= use[3]
                self.deps.pop(seq, None)
                self.coupled_seqs.discard(seq)
                if slots.pop(seq, None) is None:
                    continue
                task = tasks.get(seq)
                if task is None:
                    continue
                if moment != engine.now and task.callback:
                    # no observer with a future clock/counter pair.
                    raise KernelFault(E_STATE, "causal_publication")
                nlink = len(task.remaining)
                for res, amount in self.arena.services(svc_start, nlink):
                    name = self.res_names[res]
                    task.service[name] = amount
                engine._complete_native(task, moment)
            self.live = len(slots)
            # never overwrite an earlier chunk's hold with False.
            self.needs_settle = self.needs_settle or bool(self.arena.last_held)
        except BaseException:
            # The accepted batch is only partially published; mark kernel residue
            # non-exportable so callback-triggered disable cannot raise stale_slot.
            self.aborted_publication = True
            raise
        finally:
            self.applying = False
            if self.callback_dirty:
                self.callback_dirty = False
                if self.enabled:
                    # A callback was attached to a still-live mirrored row by a
                    # callback in this batch: hand back before the next run.
                    self.disable(engine, self._mixed_reason())
                    self.made_progress = True
    def status(self) -> dict:
        status = dict(self.status_dict)
        # Honest label: ``c_native`` means numeric steps really ran in C; an
        # attached kernel that never found a callback-free window is ``idle``.
        if self.enabled and self.native_steps > 0:
            backend, reason = "c_native", None
        elif self.enabled:
            backend, reason = "c_native_ready", "no_native_window"
        else:
            backend = "python_fallback"
            reason = self.reason or self.disabled_reason or "not_attached"
        status.update({
            "accel_backend": backend, "backend_reason": reason,
            "abi": self.abi, "native_calls": self.native_calls,
            "native_steps": self.native_steps,
            "native_boundaries": self.native_boundaries,
            "fallbacks": self.fallbacks,
            "fallback_reason": reason, "mirrored_live": self.live,
            "mode": self.mode,
            "enabled": self.enabled, "epoch_enabled": EPOCH_ENABLED,
            "disabled_reason": self.disabled_reason,
        })
        return status
    def close(self) -> None:
        if self._engine_finalizer is not None:
            self._engine_finalizer.detach()
            self._engine_finalizer = None
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None
        if self.arena is not None:
            self.arena.close()
            self.arena = None
        self.enabled = False
#: Counter keys every backend reports, so ``Engine.accel_status()`` has one
#: stable shape whether or not a kernel is attached.
_IDLE_COUNTERS = {
    "native_calls": 0, "native_steps": 0, "native_boundaries": 0,
    "fallbacks": 0, "mirrored_live": 0,
}

def make_backend(requested: str | None = None):
    """Return ``(backend_or_none, telemetry)`` for one Engine."""
    mode = resolve_mode(requested)
    if mode == "python":
        return None, {
            "accel_backend": "python", "backend_reason": "requested_python",
            "requested": "python", "epoch_enabled": False,
            "enabled": False, "disabled_reason": "requested_python",
            **_IDLE_COUNTERS,
        }
    if mode == "c":
        try:
            status = accel_build.build_accel()
        except TimeoutError:
            raise
        except (OSError, RuntimeError) as error:
            raise AccelUnavailable(f"build_unavailable:{error}") from error
        if not status.get("ok"):
            raise AccelUnavailable(status.get("reason") or "native_build_unavailable",
                                   status)
        try:
            backend = NativeBackend(status, mode)
        except (AccelUnavailable, KernelFault, MemoryError) as error:
            raise AccelUnavailable(getattr(error, "reason", str(error)), status)
        return backend, {
            "accel_backend": "c_native" if EPOCH_ENABLED else "c_native_probe",
            "backend_reason": None if EPOCH_ENABLED else "epoch_disabled_by_env",
            "requested": mode, "build": status, "epoch_enabled": EPOCH_ENABLED,
            **_IDLE_COUNTERS,
        }
    # ``auto`` defers compilation until a workload is large enough to amortize it.
    backend = NativeBackend(None, "auto")
    return backend, {
        "accel_backend": "python", "backend_reason": "auto_deferred",
        "requested": mode, "build": None, "epoch_enabled": EPOCH_ENABLED,
        "enabled": False, "disabled_reason": "auto_deferred",
        **_IDLE_COUNTERS,
    }
def accel_status(engine=None) -> dict:
    """Honest backend telemetry: discovery, library and live coverage."""
    if engine is not None:
        return engine.accel_status()
    status = accel_build.build_accel(check_only=True)
    reason = status.get("reason") or "library_not_built"
    return {
        "accel_backend": status.get("backend") or "python_fallback",
        "backend_reason": reason, "fallback_reason": reason,
        "toolchain_ready": bool(status.get("ok")),
        "build": status, "process": {"pid": os.getpid()},
    }
