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

from collections import deque
import ctypes
import math
import os
import weakref

from . import accel_build, accel_cache, accel_loader
from .accel_sync import resolve_mode  # noqa: F401  (re-exported)
from .accel_loader import AccelUnavailable, load_library
from .accel_sync import (
    adopt_frontier, can_rearm, deliver_slots, drain_output, effective_units,
    explicit_units, finish_handback, flush_pending, note_rearm_rejection,
    purge_adopted, reclaim_arena, solver_capacity_exceeded, sync_back,
    task_usage,
)
from .accel_tables import (
    ABI_VERSION, Arena, B_BUDGET, B_CALLBACK, B_HORIZON, B_IDLE, B_OUTPUT,
    B_STOP, E_ARITH, E_STALL, E_STATE, INT32_MAX, KernelFault, MAX_L, MAX_OWN,
    MAX_T, MAX_U, _free_ctx_box, _inert_pool_set, _stable_task_payloads,
    is_inert_trace_entry, needs_external_start, pure_task, resource_table,
    wide_integer,
)

#: ``auto`` keeps smaller Engines on pure Python: zero build and marshalling.
AUTO_MIN_TASKS = 64
AUTO_GRAPH_MIN_TASKS = 16
#: Dynamic growable C11 resource table ceiling (matching ``MSIM_MAX_RES``).
MAX_RES_DYNAMIC = 65536
#: Bounded prepared-stage compatibility memo and ``auto`` overhead guards.
_PREP_CACHE_MAX = 2048
_AUTO_CHURN_MIN_CALLS = 512
_AUTO_CHURN_MIN_RESOURCES = 64
#: Guard only true degenerate callback thrashing (<= 1 step/epoch); healthy hybrid
#: recurrent graphs average ~2-3 steps per advance_epoch and stay faster in C11.
_AUTO_CHURN_MIN_STEPS_PER_CALL = 1.5
#: Stop re-arming ``auto`` when repeated handbacks average too few C steps/call.
_THRASH_HANDBACK_LIMIT = 4
_THRASH_MIN_STEPS_PER_CALL = 8
#: Bounded persistent hybrid adoption limits.
_HYBRID_ADOPT_TASK_CAP = 32768
_HYBRID_PROMOTE_BFS_CAP = 512
_WINDOW_REARM_STEP_COOLDOWN = 4
_WINDOW_REARM_MAX_COOLDOWN = 64
_ACTIVATE_ST_CELLS = 8
_INITIAL_AUTO_REASONS = frozenset({
    "not_attached", "auto_ready", "auto_deferred", "auto_tiny_workload",
})
#: Frontier-shaped reasons: recoverable in ``auto`` once the boundary drains.
#: Explicit ``c`` and capability failures stay disabled for the Engine's life.
_TRANSIENT_AUTO_REASONS = frozenset({
    "auto_mixed_frontier", "auto_busy", "periodic_frontier", "mixed_frontier",
    "start_boundary", "trace_retention", "mixed_priority", "alloc_boundary",
    "cohort_symmetry_break",
})
_TRANSIENT_C_REASONS = frozenset({"alloc_boundary"})

#: Callback-free numeric windows run in C unless MINDIE_SIM_ACCEL_EPOCH=0.
EPOCH_ENABLED = os.environ.get("MINDIE_SIM_ACCEL_EPOCH", "1") != "0"


def _remaining_shapes_valid(engine, task) -> bool:
    """Validate remaining work, caps and unit placements for C11 mirroring."""
    prep = task.prepared
    if (prep is not None and prep.static_stable
            and task.caps is prep.caps and task.placement is prep.placement
            and len(task.remaining) == len(prep.demand)
            and not task.service):
        resources = engine.resources
        placement = task.placement
        if placement:
            for name in task.remaining:
                units = placement.get(name)
                if units is None:
                    continue
                n_u = len(units)
                if n_u < 1 or n_u > MAX_U:
                    return False
                res_obj = resources[name]
                if res_obj.policy == "fifo" and n_u < res_obj.units:
                    return False
        if task.coupled and placement and len(task.remaining) > 1:
            from .accel_terms import coupled_placement_proportional
            return coupled_placement_proportional(engine, task)
        return True
    resources = engine.resources
    placement = task.placement
    caps = task.caps
    for name, work in task.remaining.items():
        res_obj = resources.get(name)
        if res_obj is None or not math.isfinite(work) or work <= 0.0:
            return False
        cap = caps.get(name, math.inf)
        if math.isnan(cap) or cap <= 0.0:
            return False
        units = placement.get(name) if placement else None
        if units is None:
            continue
        if len(units) < 1 or len(units) > MAX_U:
            return False
        limit = res_obj.units
        if res_obj.policy == "fifo" and len(units) < limit:
            return False
        if any(type(unit) is not int or unit < 0 or unit >= limit for unit in units):
            return False
    if task.coupled and task.placement and len(task.remaining) > 1:
        from .accel_terms import coupled_placement_proportional
        return coupled_placement_proportional(engine, task)
    return True


def _task_c11_compatible(
    engine, task, *, started: bool, allow_non_hook_cb: bool,
    allow_free_only: bool = False,
) -> bool:
    """True when ``task`` can execute bit-identically inside the C11 arena."""
    if engine._periodic_runs:
        return False
    if task.start_callback is not None or task.latency_from_first_service:
        return False
    if not started:
        if task.acquire or task.allocation_demands:
            return False
        if task.free and not allow_free_only:
            return False
    if task.callback is not None and not task.native_hook and not allow_non_hook_cb:
        return False
    if task.cohort_widened and len(task.remaining) > MAX_RES_DYNAMIC:
        return False
    if not _stable_task_payloads(task, engine):
        return False
    if task.metadata is not None and "occurrence" in task.metadata:
        return False
    try:
        if not math.isfinite(task.release):
            return False
        if not math.isfinite(task.latency) or task.latency < 0.0:
            return False
    except (OverflowError, TypeError):
        return False
    if wide_integer(task):
        return False
    return _remaining_shapes_valid(engine, task)
def load_accel_cdll(status: dict | None = None):
    """Build (when needed) and map under one cache lock."""
    return accel_loader.load_accel_cdll(status)
class NativeBackend:
    """One Engine's immutable library handle, kernel context and telemetry."""
    __slots__ = (
        "library", "arena", "status_dict", "backend",
        "reason", "abi", "native_calls", "native_steps", "epoch_calls",
        "native_epochs_executed", "native_boundaries", "fallbacks", "mode", "enabled",
        "res_names", "pending", "deliver", "drops",
        "external", "callback_marks", "max_tasks", "live", "seeded",
        "disabled_reason", "error", "made_progress", "engine_slots",
        "dormant", "recoverable", "handing_back", "rearm_after",
        "rearm_failures", "rearm_rejections", "_finalizer", "deps",
        "usage", "links", "units", "edges",
        "storage", "epoch_links", "epoch_units", "epoch_storage",
        "epoch_tasks", "epoch_edges", "_engine_finalizer", "__weakref__",
        "applying", "callback_dirty", "coupled_seqs", "pending_handback",
        "draining", "aborted_publication", "needs_settle", "fair_prio",
        "has_fair_res", "min_tasks", "res_index", "touched_res_ids", "res_version",
        "handbacks", "partial_window", "window_has_non_hook_cb",
        "window_rearm_after", "window_rearm_cooldown", "pending_ext_seqs",
        "start_obs_seqs", "cohort_seqs", "cohort_checked_seq",
        "_prep_ok_cache", "_prep_cache_ver",
        "_trace_dq_id", "_trace_seen_appends", "_trace_seen_ext_mut",
        "_trace_seen_len", "_trace_has_non_inert", "_cohort_stop_key",
    )
    def __init__(self, status: dict | None, mode: str = "auto", *,
                 eager_attach: bool = False) -> None:
        self.library = None
        self.arena = None
        self.status_dict = dict(status or {})
        self.backend = "python_fallback"
        self.reason = "auto_deferred" if mode == "auto" else None
        self.abi = ABI_VERSION
        self.native_calls = 0
        self.native_steps = 0
        self.epoch_calls = 0
        self.native_epochs_executed = 0
        self.native_boundaries = 0
        self.fallbacks = 0
        self.mode = mode
        self.min_tasks = AUTO_GRAPH_MIN_TASKS if mode == "auto" and eager_attach else AUTO_MIN_TASKS
        self.enabled = False
        self.res_names: tuple[str, ...] = ()
        self.res_index: dict[str, int] = {}
        self.touched_res_ids: set[int] = set()
        self.res_version: int = -1
        self.pending: list = []
        self.deliver: list[int] = []
        self.drops: list[int] = []
        self.external: list[int] = []
        self.callback_marks: list[int] = []
        self.max_tasks = 0
        self.live = 0
        self.seeded = False
        self.disabled_reason = "not_attached"
        self.error: str | None = None
        self.made_progress = False
        self.engine_slots: dict[int, int] = {}
        self.dormant = not EPOCH_ENABLED
        self.recoverable = False
        self.handing_back = False
        self.rearm_after = 0
        self.rearm_failures = 0
        self.rearm_rejections = 0
        self.handbacks = 0
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
        self.partial_window = False
        self.window_has_non_hook_cb = False
        self.window_rearm_after = 0
        self.window_rearm_cooldown = 0
        self.pending_ext_seqs: set[int] = set()
        self.start_obs_seqs: set[int] = set()
        self.cohort_seqs: set[int] = set()
        self.cohort_checked_seq = 0
        self._prep_ok_cache: dict[int, tuple[object, bool]] = {}
        self._trace_dq_id: int = 0
        self._trace_seen_appends: int = 0
        self._trace_seen_ext_mut: int = 0
        self._trace_seen_len: int = 0
        self._trace_has_non_inert: bool = False
        self._cohort_stop_key = None
        self._prep_cache_ver = -1
    # -- lifecycle -----------------------------------------------------
    def try_attach(self, engine) -> bool:
        if self.enabled:
            return True
        if self.mode == "auto" and engine._sequence < self.min_tasks:
            self.disabled_reason = "auto_tiny_workload"
            return False
        reuse_config = False
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
            self._cohort_stop_key = None
            self.seeded = False
            reuse_config = (
                bool(self.res_names)
                and self.res_version == engine._resource_version
                and len(engine.resources) == len(self.res_names)
            )
        if not reuse_config:
            rates, units, policy = resource_table(engine)
            if len(rates) < 1 or len(rates) > MAX_RES_DYNAMIC:
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
            self._cohort_stop_key = None
            self.res_names = tuple(engine.resources)
            self.res_index = {name: i for i, name in enumerate(self.res_names)}
            self.res_version = engine._resource_version
            self.has_fair_res = bool(policy)
        self.touched_res_ids.clear()
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
    def disable(self, engine, reason: str = "mixed_frontier", *,
                count_handback: bool = True) -> None:
        """Return rows to Python; ``auto`` keeps a frontier reason recoverable."""
        was_enabled = self.enabled
        self.enabled = False
        self.dormant = True
        self.partial_window = False
        self.window_has_non_hook_cb = False
        self.pending_ext_seqs.clear()
        self.start_obs_seqs.clear()
        self.recoverable = (
            (self.mode == "auto" and reason in _TRANSIENT_AUTO_REASONS)
            or (self.mode == "c" and reason in _TRANSIENT_C_REASONS)
        )
        self.disabled_reason = reason
        self.needs_settle = False
        if was_enabled:
            if count_handback:
                self.handbacks += 1
                if (self.recoverable and self.mode == "auto"
                        and self.handbacks >= _THRASH_HANDBACK_LIMIT
                        and self.native_steps < self.native_calls * _THRASH_MIN_STEPS_PER_CALL):
                    self.recoverable = False
            # A real hand-back is a legitimate transition: the next pure burst
            # may adopt immediately, so the failure backoff restarts here.
            self.rearm_failures = 0
            self.rearm_after = 0
            self.pending_handback = True
            finish_handback(self, engine)
    def _release_window(self, engine, reason: str = "alloc_boundary") -> None:
        """Pause a partial native window at an exact causal boundary without thrash penalty."""
        self.enabled = False
        self.dormant = True
        self.partial_window = False
        self.window_has_non_hook_cb = False
        self.pending_ext_seqs.clear()
        self.start_obs_seqs.clear()
        self.recoverable = (
            self.mode == "auto"
            or (self.mode == "c" and reason in _TRANSIENT_C_REASONS)
        )
        self.disabled_reason = reason
        self.needs_settle = False
        if reason == "cohort_symmetry_break":
            self._bump_window_cooldown(engine)
        else:
            self.window_rearm_after = 0
            self.window_rearm_cooldown = 0
        self.pending_handback = True
        finish_handback(self, engine)
    def _sync_back(self, engine) -> None:
        """Exact hand-back owned by :mod:`accel_sync`."""
        sync_back(self, engine)
    # -- eligibility ---------------------------------------------------
    def _adopt(self, engine, skip: int) -> bool:
        """Mirror the unstarted pure frontier when ``auto`` attaches."""
        return adopt_frontier(self, engine, skip)
    def _rearmable(self, engine) -> bool:
        return can_rearm(self, engine)
    def _hybrid_allowed(self, engine) -> bool:
        return (
            self.mode == "auto"
            and engine._trace_maxlen == 0
            and not engine._periodic_runs
        )
    def _can_try_window_adopt(self, engine) -> bool:
        """Cheap gate for adopting the settled active frontier inside ``advance``."""
        if (self.enabled or self.handing_back or not EPOCH_ENABLED
                or engine._periodic_runs or not engine._tasks
                or engine._native_slots or engine._native_seqs or self.pending):
            return False
        if self.mode == "auto":
            if (engine._sequence < self.min_tasks
                    or len(engine._tasks) < self.min_tasks
                    or engine._trace_maxlen > 0):
                return False
            reason = self.disabled_reason
            if reason is not None and reason not in _TRANSIENT_AUTO_REASONS and reason not in _INITIAL_AUTO_REASONS:
                return False
        elif self.mode == "c":
            if not self.recoverable or engine._trace_maxlen > 0:
                return False
        else:
            return False
        return engine.advance_steps >= self.window_rearm_after
    def _bump_window_cooldown(self, engine) -> None:
        step = max(_WINDOW_REARM_STEP_COOLDOWN, self.window_rearm_cooldown * 2)
        self.window_rearm_cooldown = min(_WINDOW_REARM_MAX_COOLDOWN, step)
        self.window_rearm_after = engine.advance_steps + self.window_rearm_cooldown
    def _build_all_predecessors(self, engine) -> dict[int, list]:
        """Map every live task sequence to all undone predecessor Events/promises."""
        tasks = engine._tasks
        preds: dict[int, list] = {}
        for t_obj in tasks.values():
            ev = t_obj.event
            for wseq in ev._waiters:
                if wseq in tasks:
                    preds.setdefault(wseq, []).append(ev)
        for promise in engine._promises.values():
            if promise.done:
                continue
            for wseq in promise._waiters:
                if wseq in tasks:
                    preds.setdefault(wseq, []).append(promise)
        return preds
    def _collect_window_candidates(self, engine):
        """Collect all active tasks plus all compatible unstarted tasks into one persistent arena."""
        tasks = engine._tasks
        allow_cb = self.mode == "auto"
        has_fair = bool(engine.resources)
        active_tasks = []
        fair_prio = None
        has_non_hook_cb = False
        for seq in sorted(engine._active):
            t_obj = tasks.get(seq)
            if t_obj is None or t_obj.start is None or t_obj.event.done:
                return None
            if not t_obj.remaining and engine.now >= t_obj.start + t_obj.latency:
                return None
            if not _task_c11_compatible(
                engine, t_obj, started=True, allow_non_hook_cb=allow_cb,
                allow_free_only=True,
            ):
                return None
            if has_fair:
                if fair_prio is None:
                    fair_prio = t_obj.priority
                elif t_obj.priority != fair_prio:
                    return None
            if t_obj.callback is not None and not t_obj.native_hook:
                has_non_hook_cb = True
            active_tasks.append(t_obj)
        all_preds = self._build_all_predecessors(engine)
        window_list = list(active_tasks)
        in_window = {t_obj.event.sequence for t_obj in active_tasks}
        preds_by_seq: dict[int, list] = {t_obj.event.sequence: [] for t_obj in active_tasks}
        nlinks = sum(len(t_obj.remaining) for t_obj in active_tasks)
        nunits = sum(effective_units(engine, t_obj) for t_obj in active_tasks)
        nstore = sum(explicit_units(t_obj) for t_obj in active_tasks)
        nedges = 0
        coupled = any(t_obj.coupled for t_obj in active_tasks)
        if (nstore > INT32_MAX or nlinks > MAX_L
                or solver_capacity_exceeded(len(active_tasks), nlinks, nunits, coupled)):
            return None
        start_waiters = engine._start_waiters
        for seq, w_obj in sorted(tasks.items()):
            if seq in in_window or len(window_list) >= _HYBRID_ADOPT_TASK_CAP:
                continue
            if w_obj.start is not None or w_obj.event.done:
                continue
            if w_obj.start_observer is not None or seq in start_waiters:
                continue
            if has_fair and fair_prio is not None and w_obj.priority != fair_prio:
                continue
            if not _task_c11_compatible(
                engine, w_obj, started=False, allow_non_hook_cb=allow_cb,
                allow_free_only=allow_cb,
            ):
                continue
            p_list = all_preds.get(seq, ())
            if len(p_list) != w_obj.waiting:
                continue
            w_links = len(w_obj.remaining)
            w_units = effective_units(engine, w_obj)
            w_store = explicit_units(w_obj)
            w_edges = len(p_list)
            w_coupled = coupled or w_obj.coupled
            if (len(window_list) + 1 > MAX_T
                    or nlinks + w_links > MAX_L
                    or nedges + w_edges > MAX_OWN
                    or nstore + w_store > INT32_MAX
                    or solver_capacity_exceeded(
                        len(window_list) + 1, nlinks + w_links, nunits + w_units, w_coupled)):
                break
            in_window.add(seq)
            preds_by_seq[seq] = list(p_list)
            window_list.append(w_obj)
            nlinks += w_links
            nunits += w_units
            nstore += w_store
            nedges += w_edges
            coupled = w_coupled
            if w_obj.callback is not None and not w_obj.native_hook:
                has_non_hook_cb = True
        ext_boundary = {
            t_obj.event.sequence
            for t_obj in window_list
            if any(wseq not in in_window and wseq in tasks for wseq in t_obj.event._waiters)
        }
        return active_tasks, window_list, preds_by_seq, ext_boundary, has_non_hook_cb
    def _activate_native_tasks(self, engine, active_tasks) -> None:
        """Activate already-started Python tasks inside the live C11 arena."""
        slots = engine._native_slots
        by_start: dict[float, list[int]] = {}
        for t_obj in active_tasks:
            seq = t_obj.event.sequence
            engine._latency_active.discard(seq)
            by_start.setdefault(float(t_obj.start), []).append(slots[seq])
        for start_moment, act_slots in by_start.items():
            buf = (ctypes.c_int32 * len(act_slots))(*act_slots)
            rc = self.library.msim_ctx_activate(
                self.arena.ctx, buf, len(act_slots), start_moment,
                self.arena.st, _ACTIVATE_ST_CELLS)
            if rc != 0:
                raise KernelFault(int(self.arena.st[1]), "activate")
        engine._native_active += len(active_tasks)
    def _mark_ext_boundary_slots(self, engine, boundary_seqs) -> None:
        """Mark mirrored slots that have unmirrored Python waiters."""
        if not boundary_seqs:
            return
        slots = engine._native_slots
        ext_slots = [slots[s] for s in boundary_seqs if s in slots]
        if ext_slots:
            self.arena.call("msim_ctx_mark_external", ext_slots)
            self.arena.call("msim_ctx_mark_callback", ext_slots)
    def _has_live_cohorts(self, engine) -> bool:
        """O(1)-amortized check for whether any live task has ``cohort_size > 1``."""
        tasks = engine._tasks
        if self.cohort_seqs:
            self.cohort_seqs.intersection_update(tasks)
        cur_seq = engine._sequence
        prev_seq = self.cohort_checked_seq
        if cur_seq > prev_seq:
            if prev_seq == 0 or (cur_seq - prev_seq) > len(tasks):
                for seq, t_obj in tasks.items():
                    if t_obj.cohort_size > 1:
                        self.cohort_seqs.add(seq)
            else:
                for seq in range(prev_seq + 1, cur_seq + 1):
                    t_obj = tasks.get(seq)
                    if t_obj is not None and t_obj.cohort_size > 1:
                        self.cohort_seqs.add(seq)
            self.cohort_checked_seq = cur_seq
        return bool(self.cohort_seqs)
    def _adopt_active_window(self, engine) -> bool:
        """Adopt the settled active frontier (+ all compatible unstarted tasks) into C11."""
        from . import accel_terms as _at
        if self._trace_blocks_epoch(engine):
            return False
        if self._has_live_cohorts(engine) and _at.has_asymmetric_cohort(engine, self.cohort_seqs):
            return False
        collected = self._collect_window_candidates(engine)
        if collected is None:
            return False
        active_tasks, window_list, preds_by_seq, ext_boundary, has_non_hook_cb = collected
        if not self.try_attach(engine):
            return False
        for t_obj in window_list:
            seq = t_obj.event.sequence
            self.queue(engine, t_obj.event, t_obj, tuple(preds_by_seq[seq]))
            engine._native_seqs.add(seq)
        purge_adopted(engine, (t_obj.event.sequence for t_obj in window_list if t_obj.start is None))
        self.engine_slots = engine._native_slots
        self._flush_with_ext(engine)
        self._activate_native_tasks(engine, active_tasks)
        self._mark_ext_boundary_slots(engine, ext_boundary)
        self.partial_window = len(engine._native_seqs) < len(engine._tasks)
        self.window_has_non_hook_cb = has_non_hook_cb
        self.window_rearm_after = 0
        self.window_rearm_cooldown = 0
        return True
    def _collect_promoted_descendants(self, engine, new_active, allow_cb: bool):
        """Bounded local BFS for newly-submitted unstarted compatible children of ``new_active``."""
        tasks = engine._tasks
        mirrored = engine._native_seqs
        start_waiters = engine._start_waiters
        promoted = list(new_active)
        in_batch = {t_obj.event.sequence for t_obj in new_active}
        preds_by_seq: dict[int, list] = {t_obj.event.sequence: [] for t_obj in new_active}
        pred_hits: dict[int, list] = {}
        queue = deque(new_active)
        while queue and len(promoted) < _HYBRID_PROMOTE_BFS_CAP:
            cur = queue.popleft()
            for wseq in cur.event._waiters:
                if wseq in mirrored or wseq in in_batch:
                    continue
                w_obj = tasks.get(wseq)
                if w_obj is None or w_obj.start is not None or w_obj.event.done:
                    continue
                hits = pred_hits.setdefault(wseq, [])
                hits.append(cur.event)
                if len(hits) < w_obj.waiting:
                    continue
                if w_obj.start_observer is not None or wseq in start_waiters:
                    continue
                if self.has_fair_res and self.fair_prio is not None and w_obj.priority != self.fair_prio:
                    continue
                if (self._fast_prepared_c11_compatible(engine, w_obj) is not True
                        and not _task_c11_compatible(
                            engine, w_obj, started=False, allow_non_hook_cb=allow_cb,
                            allow_free_only=allow_cb,
                        )):
                    continue
                in_batch.add(wseq)
                preds_by_seq[wseq] = hits
                promoted.append(w_obj)
                queue.append(w_obj)
        return promoted, in_batch, preds_by_seq
    def _promote_hybrid_tasks(self, engine) -> bool:
        """Incrementally admit newly Python-started tasks into the STILL-LIVE C11 arena."""
        if len(engine._active) <= engine._native_active:
            return True
        tasks = engine._tasks
        mirrored = engine._native_seqs
        new_active = [tasks[seq] for seq in sorted(engine._active) if seq not in mirrored]
        if not new_active:
            return True
        allow_cb = self.mode == "auto"
        for t_obj in new_active:
            if t_obj.start is None or t_obj.event.done:
                return False
            if not t_obj.remaining and engine.now >= t_obj.start + t_obj.latency:
                return False
            if not _task_c11_compatible(
                engine, t_obj, started=True, allow_non_hook_cb=allow_cb,
                allow_free_only=True,
            ):
                return False
            if self.has_fair_res and self.fair_prio is not None and t_obj.priority != self.fair_prio:
                return False
        promoted, in_batch, preds_by_seq = self._collect_promoted_descendants(
            engine, new_active, allow_cb,
        )
        slots = engine._native_slots
        extra_edge_count = sum(
            1 for t_obj in promoted
            for wseq in dict.fromkeys(t_obj.event._waiters)
            if wseq in slots and wseq not in in_batch
        )
        add_links = sum(len(t_obj.remaining) for t_obj in promoted)
        add_units = sum(effective_units(engine, t_obj) for t_obj in promoted)
        add_store = sum(explicit_units(t_obj) for t_obj in promoted)
        add_edges = extra_edge_count + sum(len(preds_by_seq[t_obj.event.sequence]) for t_obj in promoted)
        coupled = bool(self.coupled_seqs) or any(t_obj.coupled for t_obj in promoted)
        if (len(mirrored) + len(promoted) > MAX_T
                or self.epoch_links + add_links > MAX_L
                or self.epoch_storage + add_store > INT32_MAX
                or self.epoch_edges + add_edges > MAX_OWN
                or solver_capacity_exceeded(
                    len(mirrored) + len(promoted),
                    self.links + add_links, self.units + add_units, coupled)):
            # Clean periodic arena reset when monotonic edge/link counters reach ceiling.
            self._release_window(engine, "alloc_boundary")
            return False
        self.engine_slots = slots
        if self.pending:
            self._flush_with_ext(engine)
        for t_obj in promoted:
            seq = t_obj.event.sequence
            self.queue(engine, t_obj.event, t_obj, tuple(preds_by_seq[seq]))
            mirrored.add(seq)
            if t_obj.callback is not None and not t_obj.native_hook:
                self.window_has_non_hook_cb = True
        purge_adopted(engine, (t_obj.event.sequence for t_obj in promoted if t_obj.start is None))
        self._flush_with_ext(engine)
        extra_edges = [
            (slots[t_obj.event.sequence], slots[wseq], wseq)
            for t_obj in promoted
            for wseq in dict.fromkeys(t_obj.event._waiters)
            if wseq in slots and wseq not in in_batch
        ]
        if extra_edges:
            self.arena.link(extra_edges)
            self.epoch_edges += len(extra_edges)
        ext_boundary = [
            t_obj.event.sequence for t_obj in promoted
            if any(wseq not in mirrored and wseq in tasks for wseq in t_obj.event._waiters)
        ]
        self._activate_native_tasks(engine, new_active)
        self._mark_ext_boundary_slots(engine, ext_boundary)
        return True
    def _all_unmirrored_alloc_only(self, engine) -> bool:
        """True when every non-mirrored task is only an allocation-boundary task."""
        mirrored = engine._native_seqs
        for seq, t_obj in engine._tasks.items():
            if seq in mirrored:
                continue
            if not (t_obj.acquire or t_obj.free or t_obj.allocation_demands):
                return False
            if not _task_c11_compatible(
                engine, t_obj, started=True, allow_non_hook_cb=False,
                allow_free_only=True,
            ):
                return False
        return True
    def _fast_prepared_c11_compatible(self, engine, task) -> bool | None:
        """O(1) compatibility check for repeated freshly-submitted ``PreparedStage`` tasks."""
        prep = task.prepared
        if prep is None or task.start is not None or task.service:
            return None
        if (task.start_callback is not None or task.acquire or task.free
                or task.allocation_demands or task.memory_ledger is not None
                or task.memory_bindings is not None or task.memory_grant_names):
            return False
        if task.callback is not None and not task.native_hook:
            return False
        meta = task.metadata
        if meta is not None and (type(meta) is not dict or meta):
            return None
        if task.owner is not None and type(task.owner) is not str:
            return False
        if type(task.waiting) is not int or task.waiting < 0:
            return False
        rel = task.release
        if (type(rel) is not float and type(rel) is not int) or not math.isfinite(rel):
            return False
        if (type(task.memory_granted) is not bool
                or type(task.memory_replica) is not int
                or type(task.memory_stage) is not int
                or not _inert_pool_set(task.memory_pools, engine)):
            return False
        ev = task.event
        waiters = ev._waiters
        if (type(ev.name) is not str or type(ev.sequence) is not int
                or ev.time is not None or ev.started is not None
                or ev._callbacks is not None or type(waiters) is not list):
            return None
        for w in waiters:
            if type(w) is not int:
                return None
        if (task.demand is not prep.demand or task.caps is not prep.caps
                or task.placement is not prep.placement
                or task.cohort_widened is not prep.cohort_widened
                or task.cohort_members is not prep.cohort_members):
            return None
        rem = task.remaining
        pd = prep.demand
        if type(rem) is not dict or len(rem) != len(pd):
            return None
        for rk, rv in rem.items():
            rt = type(rv)
            if type(rk) is not str or (rt is not float and rt is not int):
                return None
            if pd.get(rk) != rv:
                return None
        if prep.coupled and prep.placement and engine._partial_placed_live:
            return None
        ver = engine._resource_version
        if self._prep_cache_ver != ver:
            self._prep_ok_cache.clear()
            self._prep_cache_ver = ver
        pid = id(prep)
        hit = self._prep_ok_cache.get(pid)
        if hit is not None and hit[0] is prep:
            return hit[1]
        ok = _task_c11_compatible(
            engine, task, started=False, allow_non_hook_cb=False,
            allow_free_only=False,
        )
        if len(self._prep_ok_cache) >= _PREP_CACHE_MAX:
            self._prep_ok_cache.clear()
        self._prep_ok_cache[pid] = (prep, ok)
        return ok
    def eligible(self, engine, task) -> bool:
        if self.handing_back or self.partial_window:
            return False
        if self.dormant:
            if not can_rearm(self, engine):
                return False
        if engine._periodic_runs:
            self.disabled_reason = "periodic_frontier"
            if not self.enabled:
                self.dormant = True
                self.recoverable = self.mode == "auto"
            return False
        if wide_integer(task):
            if self.enabled:
                self.disable(engine, "wide_integer")
            else:
                self.disabled_reason = "wide_integer"
            return False
        fast_ok = self._fast_prepared_c11_compatible(engine, task)
        if fast_ok is False:
            return False
        if fast_ok is None and not _task_c11_compatible(
            engine, task, started=False, allow_non_hook_cb=False,
            allow_free_only=False,
        ):
            return False
        if (self.enabled and self.has_fair_res and self.fair_prio is not None
                and task.priority != self.fair_prio):
            self.disable(engine, "mixed_priority")
            return False
        if not self.enabled:
            if engine._sequence < self.min_tasks:
                return False
            if engine._active or engine._native_slots or engine._completable:
                # Transient: the next pure burst may adopt.  Staying dormant
                # keeps the per-task preflight out of the hot Python path.
                self.disabled_reason = "auto_busy"
                self.dormant = self.mode == "auto"
                self.recoverable = self.mode == "auto"
                return False
            if not self.try_attach(engine):
                return False
            if (self.mode == "auto"
                    and not self._adopt(engine, task.event.sequence)):
                reason = ("capacity_exceeded"
                          if self.disabled_reason == "capacity_exceeded"
                          else "auto_mixed_frontier")
                # Failed pre-execution adoption is NOT a real execution handback.
                self.disable(engine, reason, count_handback=False)
                if reason != "capacity_exceeded":
                    note_rearm_rejection(self, engine)
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
            if (self.mode == "auto" and self._hybrid_allowed(engine)
                    and len(engine._native_seqs) + 1 <= MAX_T
                    and len(task.remaining) <= MAX_L
                    and new_storage <= INT32_MAX
                    and task.waiting <= MAX_OWN
                    and not solver_capacity_exceeded(
                        len(engine._native_seqs) + 1,
                        self.links + len(task.remaining),
                        self.units + new_units, coupled)):
                self._release_window(engine, "alloc_boundary")
            else:
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
        if task.start_observer is not None:
            self.start_obs_seqs.add(event.sequence)
        if task.cohort_size > 1:
            self.cohort_seqs.add(event.sequence)
        if self.has_fair_res and self.fair_prio is None:
            self.fair_prio = task.priority
        self.engine_slots = engine._native_slots
        self.live += 1
        if self.live > self.max_tasks:
            self.max_tasks = self.live
    def note_external_predecessors(self, engine, pending) -> None:
        if not pending or (not engine._native_slots and not engine._native_seqs):
            return
        slots = engine._native_slots
        mark_cb = self.partial_window or self._hybrid_allowed(engine)
        for predecessor in pending:
            seq = predecessor.sequence
            slot = slots.get(seq)
            if slot is not None:
                self.external.append(slot)
                if mark_cb:
                    self.callback_marks.append(slot)
            elif seq in engine._native_seqs:
                self.pending_ext_seqs.add(seq)
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
        self.start_obs_seqs.discard(sequence)
        self.drops.append(slot)
        if self.live:
            self.live -= 1
    def start_conflict(self, engine, until=()) -> bool:
        """True when a native start cannot wake a Python observer mid-epoch."""
        if not self.enabled or not engine._native_seqs:
            if self._can_try_window_adopt(engine):
                # Trigger settled-frontier adoption inside ``advance_epoch`` right
                # after ``Engine._settle`` drains zero-work completions/callbacks.
                self.made_progress = True
            return False
        if not engine._start_waiters and not self.start_obs_seqs:
            return False
        stopping = {event.sequence for event in until if not event.done}
        for observers in engine._start_waiters.values():
            for observer in observers:
                if (observer.sequence in stopping or observer._waiters
                        or observer._callbacks):
                    return True
        if self.start_obs_seqs:
            tasks = engine._tasks
            for seq in self.start_obs_seqs:
                t_obj = tasks.get(seq)
                obs = t_obj.start_observer if t_obj is not None else None
                if obs is not None and (obs.sequence in stopping or obs._waiters or obs._callbacks):
                    return True
        return False
    def _mixed_reason(self) -> str:
        return "auto_mixed_frontier" if self.mode == "auto" else "mixed_frontier"
    def can_publish_control(self, engine) -> bool:
        """Allow one zero-work Python grant at a native publication boundary."""
        return (self.applying and self.enabled
                and len(engine._native_seqs) == len(engine._tasks) - 1)
    def on_callback_registered(self, engine, event, *, native_hook: bool = False) -> None:
        """A live mirrored row acquired a Python completion observer.

        An undeclared hook was not marked when the task was mirrored. The
        kernel can no longer run through that row's completion: it cannot
        invoke the callback at the exact instant and it would publish the epoch
        clock past it.  Hand the whole frontier back before the next native
        step, deferring the transfer when this registration happens inside
        ``_apply`` (the current completion batch must finish first).  A
        declared ``native_hook=True`` observer instead marks ``MSIM_F_CALLBACK``
        directly on the mirrored slot so the kernel halts at that exact
        completion instant without abandoning native execution.
        """
        if not self.enabled or event.done:
            return
        if event.sequence not in engine._native_seqs:
            return
        if native_hook:
            slot = engine._native_slots.get(event.sequence)
            if slot is not None:
                self.callback_marks.append(slot)
            return
        if self.applying:
            self.callback_dirty = True
            return
        self.disable(engine, self._mixed_reason())
        self.made_progress = True
    def _sync_counters(self, engine, *, full: bool = False) -> None:
        if self.arena is None:
            return
        if engine._hbm_acct_live:
            from .accel_terms import sync_native_hbm_accounting
            sync_native_hbm_accounting(self, engine)
        busy, serv = self.arena.counters_raw()
        acct = engine._hbm_accounting_serviced
        names = self.res_names
        eng_busy = engine.busy_ms
        eng_serv = engine.serviced
        indices = (
            range(len(names))
            if (full or not self.touched_res_ids)
            else self.touched_res_ids
        )
        if acct:
            for i in indices:
                name = names[i]
                eng_busy[name] = busy[i]
                eng_serv[name] = serv[i] + acct.get(name, 0.0)
        else:
            for i in indices:
                name = names[i]
                eng_busy[name] = busy[i]
                eng_serv[name] = serv[i]
        if engine._delivery_live:
            from .accel_terms import sync_native_deliveries
            sync_native_deliveries(self, engine)
    def _apply_prefix(self, engine, error: KernelFault) -> None:
        """publish the kernel-accepted prefix of a failing epoch."""
        prefix = getattr(error, "prefix", None)
        if prefix is None or not getattr(error, "has_prefix", False):
            self._sync_counters(engine, full=True)
            return
        now, _boundary, steps, ncomp, nstart = prefix
        self.native_calls += 1
        self.native_steps += steps
        if steps > 0:
            self.native_epochs_executed += 1
        self._sync_counters(engine, full=True)
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
    def _flush_with_ext(self, engine) -> None:
        flush_pending(self, engine)
        if self.pending_ext_seqs:
            slots = engine._native_slots
            ext_slots = [slots[s] for s in self.pending_ext_seqs if s in slots]
            self.pending_ext_seqs.clear()
            if ext_slots:
                self.arena.call("msim_ctx_mark_external", ext_slots)
                if self.partial_window or self._hybrid_allowed(engine):
                    self.arena.call("msim_ctx_mark_callback", ext_slots)
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
        dq = engine._trace_dq
        cur_id = id(dq)
        cur_appends = engine._trace_appends
        cur_ext = engine._trace_view._ext_mut
        cur_len = len(dq)
        if not cur_len:
            self._trace_dq_id = cur_id
            self._trace_seen_appends = cur_appends
            self._trace_seen_ext_mut = cur_ext
            self._trace_seen_len = 0
            self._trace_has_non_inert = False
            return False
        if (not self._trace_has_non_inert
                and cur_id == self._trace_dq_id
                and cur_ext == self._trace_seen_ext_mut):
            delta = cur_appends - self._trace_seen_appends
            if delta == 0 and cur_len == self._trace_seen_len:
                return False
            if 0 < delta <= cur_len and cur_len <= self._trace_seen_len + delta:
                non_inert = False
                for idx in range(cur_len - delta, cur_len):
                    if not is_inert_trace_entry(dq[idx]):
                        non_inert = True
                        break
                self._trace_seen_appends = cur_appends
                self._trace_seen_len = cur_len
                if not non_inert:
                    return False
                self._trace_has_non_inert = True
                self.disable(engine, "trace_retention")
                self.made_progress = True
                return True
        non_inert = any(not is_inert_trace_entry(row) for row in dq)
        self._trace_dq_id = cur_id
        self._trace_seen_appends = cur_appends
        self._trace_seen_ext_mut = cur_ext
        self._trace_seen_len = cur_len
        self._trace_has_non_inert = non_inert
        if not non_inert:
            return False
        self.disable(engine, "trace_retention")
        self.made_progress = True
        return True
    def _handback_mixed(self, engine) -> None:
        """Disable a frontier that is no longer entirely kernel-mirrored."""
        self.disable(engine, "mixed_frontier")
        self.made_progress = True
        self.needs_settle = False
    def _can_refresh_after_drain(self, engine) -> bool:
        """True when resource table growth only requires draining live slots first."""
        if not engine._native_slots:
            return False
        names = tuple(engine.resources)
        if not 1 <= len(names) <= MAX_RES_DYNAMIC:
            return False
        rates, units, policy = resource_table(engine)
        if not all(math.isfinite(rate) and rate > 0 and 1 <= unit <= MAX_U
                   for rate, unit in zip(rates, units)):
            return False
        if policy:
            priorities = {task.priority for _, task, _ in self.pending}
            if len(priorities) > 1:
                return False
        return True
    def _refresh_resources(self, engine) -> bool:
        if (self.res_version == engine._resource_version
                and len(engine.resources) == len(self.res_names)):
            return True
        names = tuple(engine.resources)
        if names == self.res_names:
            self.res_version = engine._resource_version
            return True
        if engine._native_slots or not 1 <= len(names) <= MAX_RES_DYNAMIC:
            return False
        rates, units, policy = resource_table(engine)
        if not all(math.isfinite(rate) and rate > 0 and 1 <= unit <= MAX_U
                   for rate, unit in zip(rates, units)):
            return False
        has_fair = bool(policy)
        priorities = {task.priority for _, task, _ in self.pending} if has_fair else set()
        if len(priorities) > 1:
            return False
        # A nested program can declare its directional rows after its ready
        # event was queued. Unflushed rows have no native resource indices yet.
        self.arena.configure(rates, units, policy)
        self._cohort_stop_key = None
        self.res_names = names
        self.res_index = {name: i for i, name in enumerate(names)}
        self.res_version = engine._resource_version
        self.touched_res_ids.clear()
        self.has_fair_res = has_fair
        self.fair_prio = next(iter(priorities), None)
        self.seeded = False
        return True
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
            if not self._refresh_resources(engine):
                if self._can_refresh_after_drain(engine):
                    self._release_window(engine, "alloc_boundary")
                    self.made_progress = True
                elif self.partial_window:
                    self._release_window(engine, "mixed_frontier")
                    self.made_progress = True
                else:
                    self._handback_mixed(engine)
                return
            if not self.partial_window and len(engine._native_seqs) != len(engine._tasks):
                if self._hybrid_allowed(engine):
                    self.partial_window = True
                else:
                    self._handback_mixed(engine)
                    return
            self.engine_slots = engine._native_slots
            self._flush_with_ext(engine)
            if not engine._native_slots:
                self.needs_settle = False
                return
            if not self.partial_window and len(engine._native_slots) != len(engine._tasks):
                if self._hybrid_allowed(engine):
                    self.partial_window = True
                else:
                    self._handback_mixed(engine)
                    return
            if self.partial_window:
                self.needs_settle = False
                return
            if not self.seeded:
                acct = engine._hbm_accounting_serviced
                self.arena.seed([engine.busy_ms[n] for n in self.res_names],
                                [engine.serviced[n] - acct.get(n, 0.0) for n in self.res_names])
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
            if steps > 0:
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
    def _prepare_hybrid_epoch(self, engine) -> bool:
        """Keep the live C11 arena synchronized when unmirrored Python tasks exist."""
        if len(engine._native_seqs) == len(engine._tasks):
            self.partial_window = False
            return True
        if not self.partial_window and not self._hybrid_allowed(engine):
            if self._all_unmirrored_alloc_only(engine):
                self._release_window(engine, "alloc_boundary")
            else:
                self.disable(engine, "mixed_frontier")
            self.made_progress = True
            return False
        if not self._promote_hybrid_tasks(engine):
            if self.enabled:
                if self._hybrid_allowed(engine):
                    self._release_window(engine, "mixed_frontier")
                else:
                    self.disable(engine, "mixed_frontier")
            self.made_progress = True
            return False
        self.partial_window = len(engine._native_seqs) < len(engine._tasks)
        return True
    def advance_epoch(self, engine, horizon: float, budget: int, until) -> int:
        """Run one callback-free epoch; returns the consumed step count."""
        self.made_progress = False
        if self.dormant or not self.enabled:
            if budget <= 1 or not self._can_try_window_adopt(engine) or not engine._active:
                return 0
            if not self._adopt_active_window(engine):
                self._bump_window_cooldown(engine)
                return 0
        if not engine._native_seqs and not self.pending:
            return 0
        if self._trace_blocks_epoch(engine):
            return 0
        from . import accel_terms as _at
        has_cohorts = self._has_live_cohorts(engine)
        if has_cohorts and _at.has_asymmetric_cohort(engine, self.cohort_seqs):
            if self._hybrid_allowed(engine):
                self._release_window(engine, "cohort_symmetry_break")
            else:
                self.disable(engine, "cohort_symmetry_break")
            self.made_progress = True
            return 0
        if not self._refresh_resources(engine):
            if self._can_refresh_after_drain(engine):
                self._release_window(engine, "alloc_boundary")
            elif self.partial_window:
                self._release_window(engine, "mixed_frontier")
            else:
                self.disable(engine, "mixed_frontier")
            self.made_progress = True
            return 0
        if not self._prepare_hybrid_epoch(engine):
            return 0
        self.engine_slots = engine._native_slots
        self._flush_with_ext(engine)
        if not engine._native_slots:
            if engine._tasks and self.partial_window:
                self._release_window(engine)
                self.made_progress = True
            return 0
        if not self.partial_window and len(engine._native_slots) != len(engine._tasks):
            if not self._prepare_hybrid_epoch(engine):
                return 0
        if not self.seeded:
            acct = engine._hbm_accounting_serviced
            self.arena.seed([engine.busy_ms[n] for n in self.res_names],
                            [engine.serviced[n] - acct.get(n, 0.0) for n in self.res_names])
            self.seeded = True
        has_cohorts = self._has_live_cohorts(engine)
        if has_cohorts:
            stop = _at.stop_slots_for_epoch(
                engine, until, cohort_seqs=self.cohort_seqs, backend=self,
            )
        else:
            slots = engine._native_slots
            stop = [slots[e.sequence] for e in until if not e.done and e.sequence in slots]
        if stop:
            self.arena.call("msim_ctx_mark_external", stop)
        _at.prepare_obs_bridge(self, engine)
        track_window = self.partial_window or self.window_has_non_hook_cb
        alloc_ver_before = engine._alloc_state_version if track_window else 0
        ready_seen_before = set(engine._ready_seen) if track_window else set()
        cohort_horizon = (
            _at.cohort_break_horizon(engine, horizon, cohort_seqs=self.cohort_seqs)
            if has_cohorts else horizon
        )
        next_rel = engine._frontier_next_release()
        eff_horizon = min(cohort_horizon, next_rel)
        if track_window and next_rel <= engine.now:
            self._release_window(engine)
            self.made_progress = True
            return 0
        try:
            now, boundary, steps, ncomp, nstart = self.arena.run(
                engine.now, eff_horizon, next_rel, budget)
        except KernelFault as error:
            _at.drain_obs_bridge(self, engine)
            self._apply_prefix(engine, error)
            self._translate_fault(error)
        self.native_calls += 1
        self.epoch_calls += 1
        self.native_steps += steps
        if steps > 0:
            self.native_epochs_executed += 1
        engine.now, engine.advance_steps = now, engine.advance_steps + steps
        if steps > 0:
            self._sync_counters(engine)
        obs_prog = _at.drain_obs_bridge(self, engine)
        self._apply(engine, nstart, ncomp)
        if boundary == B_OUTPUT and (self.enabled or self.pending_handback):
            steps += drain_output(self, engine)
        finish_handback(self, engine)
        if boundary:
            self.native_boundaries += 1
        self.made_progress = bool(steps or ncomp or nstart or obs_prog)
        if (self.mode == "auto" and self.enabled
                and len(self.res_names) >= _AUTO_CHURN_MIN_RESOURCES
                and self.epoch_calls >= _AUTO_CHURN_MIN_CALLS
                and self.native_steps < self.epoch_calls * _AUTO_CHURN_MIN_STEPS_PER_CALL):
            self.disable(engine, "auto_callback_churn", count_handback=False)
            self.recoverable = False
            return steps
        self.partial_window = bool(self.enabled and len(engine._native_seqs) < len(engine._tasks))
        caller_horizon_hit = (
            boundary in (B_HORIZON, B_BUDGET, B_STOP)
            and (boundary != B_HORIZON or engine.now >= cohort_horizon)
        )
        if self.enabled and (
            (self.window_has_non_hook_cb and caller_horizon_hit)
            or (not engine._native_slots and engine._tasks)
            or (self.partial_window and boundary == B_IDLE and not self.made_progress)
            or not self._refresh_resources(engine)
        ):
            self._release_window(engine)
            if not steps:
                self.made_progress = True
            return steps
        if self.enabled and self.partial_window:
            self.needs_settle = False
            if not steps and (
                boundary in (B_CALLBACK, B_HORIZON, B_STOP)
                or engine._completable
                or engine._ready_seen != ready_seen_before
                or (engine._alloc_state_version != alloc_ver_before and engine._ready_heap)
                or (engine._releases and engine._releases[0][0] <= engine.now)
            ):
                self.made_progress = True
            return steps
        if self.needs_settle or self.pending:
            self._settle_instant(engine)
        if not self.made_progress and boundary == B_IDLE and engine._native_seqs:
            self.fallbacks += 1
            raise KernelFault(E_STALL, "idle", f"mirrored={len(engine._native_slots)}")
        return steps
    def _apply(self, engine, nstart: int, ncomp: int) -> None:
        if engine._wave_recorder is not None and nstart > 0:
            from .accel_terms import record_native_starts
            record_native_starts(self, engine, nstart, ncomp)
        tasks, slots = engine._tasks, engine._native_slots
        self.applying = True
        try:
            if nstart > 0:
                for seq, moment in self.arena.starts(nstart):
                    task = tasks.get(seq)
                    if task is None:
                        continue
                    task.start = task.event.started = moment
                    engine._active.add(seq)
                    engine._native_active += 1
                    if task.coupled:
                        engine._coupled_active += 1
                    if task.placement:
                        engine._placed_active += 1
                    if task.latency > 0.0:
                        engine._latency_active.add(seq)
                    if len(tasks) > engine.peak_tasks:
                        engine.peak_tasks = len(tasks)
                    obs1 = task.start_observer
                    if obs1 is not None:
                        task.start_observer = None
                        self.start_obs_seqs.discard(seq)
                        obs1.started = obs1.time = moment
                        engine._promises.pop(obs1.sequence, None)
                        engine._wake(obs1)
                        for callback in obs1._callbacks or ():
                            callback(obs1)
                        obs1._callbacks = None
                    if engine._start_waiters:
                        for observer in engine._start_waiters.pop(seq, ()):
                            observer.started = observer.time = moment
                            engine._promises.pop(observer.sequence, None)
                            engine._wake(observer)
                            for callback in observer._callbacks or ():
                                callback(observer)
                            observer._callbacks = None
            if ncomp > 0:
                comp_buf = self.arena.comp
                svc_buf = self.arena.svc
                res_names = self.res_names
                usage_pop = self.usage.pop
                deps_pop = self.deps.pop
                coupled_discard = self.coupled_seqs.discard
                start_obs_discard = self.start_obs_seqs.discard
                slots_pop = slots.pop
                tasks_get = tasks.get
                complete_native = engine._complete_native
                from .accel_tables import COMP_STRIDE, SVC_STRIDE, _PACK_U64, _U64_MASK, _UNPACK_F64
                for idx in range(ncomp):
                    base = idx * COMP_STRIDE
                    seq = int(comp_buf[base])
                    moment = _UNPACK_F64(_PACK_U64(int(comp_buf[base + 1]) & _U64_MASK))[0]
                    svc_start = int(comp_buf[base + 2])
                    use = usage_pop(seq, None)
                    if use is not None:
                        self.links -= use[0]
                        self.units -= use[1]
                        self.edges -= use[2]
                        self.storage -= use[3]
                    deps_pop(seq, None)
                    coupled_discard(seq)
                    start_obs_discard(seq)
                    if slots_pop(seq, None) is None:
                        continue
                    task = tasks_get(seq)
                    if task is None:
                        continue
                    if moment != engine.now and task.callback:
                        # no observer with a future clock/counter pair.
                        raise KernelFault(E_STATE, "causal_publication")
                    nlink = len(task.remaining)
                    if nlink:
                        t_svc = task.service
                        for j in range(nlink):
                            sbase = (svc_start + j) * SVC_STRIDE
                            r_name = res_names[int(svc_buf[sbase])]
                            t_svc[r_name] = t_svc.get(r_name, 0.0) + float(svc_buf[sbase + 1])
                    complete_native(task, moment)
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
            "native_epochs_executed": self.native_epochs_executed,
            "native_boundaries": self.native_boundaries,
            "fallbacks": self.fallbacks,
            "fallback_reason": reason, "mirrored_live": self.live,
            "mode": self.mode,
            "enabled": self.enabled, "epoch_enabled": EPOCH_ENABLED,
            "disabled_reason": self.disabled_reason,
        })
        return status
    def close(self) -> None:
        self._prep_ok_cache.clear()
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
_IDLE_COUNTERS = {"native_calls": 0, "native_steps": 0, "native_epochs_executed": 0,
                  "native_boundaries": 0, "fallbacks": 0, "mirrored_live": 0}

def make_backend(requested: str | None = None, *, eager_attach: bool = False):
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
    eager = bool(eager_attach and accel_loader.toolchain_or_cache_ready())
    backend = NativeBackend(None, "auto", eager_attach=eager)
    reason = "auto_ready" if eager else "auto_deferred"
    return backend, {
        "accel_backend": "python", "backend_reason": reason,
        "requested": mode, "build": None, "epoch_enabled": EPOCH_ENABLED,
        "enabled": False, "disabled_reason": reason, **_IDLE_COUNTERS,
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
