"""Frontier task submission, promises, causal boundaries and callback hooks.

[Role] Materialize prepared tasks into live ``_Task`` records, wire predecessor
  edges, coordinate optional C11 arena mirroring and manage promises/observers.
[Invariants] Every submitted dependency must already be live or promised; live
  mirrored tasks hand back cleanly if a Python completion callback attaches.
[Timeouts & Locks] Pure heap/dict operations; no threads, sleeps, locks or I/O.
[Side-Effect Policy] Mutates the Engine's live frontier and native mirror sets.
"""
from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
from dataclasses import replace
import heapq
from typing import Callable, Sequence

from . import frontier
from .contracts import OccurrenceKey
from .engine_prepare import _refresh_prepared_for_hbm_bus
from .engine_types import (
    Event, ExecutionError, _EMPTY_FROZENSET, _EMPTY_METADATA, _Task,
    _chain_callbacks,
)
from .memory_physics import (
    AdmissionMemoryView, CACHE_POLICY_DISABLE, CACHE_POLICY_LAST_USE, SectorCoverage, bind_memory_access,
)
from .l2_residency import _coverage_overlap, _coverage_runs, _merged_bytes, _residency_runs


def _backing_identity(pattern) -> tuple[int, str, int]:
    """Return canonical ``(die, backing_allocation, tensor_version)`` for coherence."""
    key = pattern.reuse_key
    if key is not None:
        return key.die, key.backing_allocation, key.tensor_version
    return pattern.die, pattern.tensor_id, pattern.version


class _EngineSubmitMixin:
    def _submit_prepared(
        self, prepared, name, *, after=(), release=None, owner="", metadata=None,
        on_complete=None, on_start=None, memory=None, memory_pools=None,
        memory_bindings=None, acquire_override=None, free_override=None,
        native_hook: bool = False, deliveries=(), inline_control: bool = False,
    ) -> Event:
        # Dependencies and allocations remain live invocation state. Only static
        # work, supply and placement validation can be shared by repeated tiles.
        prepared = _refresh_prepared_for_hbm_bus(self, prepared, name)
        now = self.now
        tasks = self._tasks
        pending_one = None
        pending = None
        if after:
            pending_one, pending, release = self._pending_predecessors(name, after, release, tasks)
        if release is None or release < now:
            release = now
        fast_ctx = self._memory_context_fast
        if (not metadata and memory is None and memory_pools is None
                and memory_bindings is None):
            the_metadata = _EMPTY_METADATA
            if fast_ctx is not None:
                ledger, pools, bindings, memory_replica, memory_stage = fast_ctx
            elif not self.memory_pools:
                ledger = self.memory
                pools = _EMPTY_FROZENSET
                bindings = None
                memory_replica = memory_stage = 0
            else:
                ledger = self.memory
                if ledger is not None:
                    self._memory_ledgers[id(ledger)] = ledger
                pools = self._memory_pools_frozen
                bindings = None
                memory_replica = memory_stage = 0
        else:
            the_metadata, ledger, pools, bindings, memory_replica, memory_stage = (
                self._resolve_task_memory(metadata, memory, memory_pools, memory_bindings))
        acq = prepared.acquire if acquire_override is None else acquire_override
        fr = prepared.free if free_override is None else free_override
        alloc_demands = prepared.allocation_demands
        # Only a task that really participates in memory accounting retains the
        # ledger and pools.  A pure compute/comm row must stay inert so a
        # serving Engine that owns one global ``Memory`` ledger does not
        # disable native execution for the whole decode frontier.
        # Only the ledger is dropped for a non-accounting row.  The declared
        # pool set is retained verbatim: callers may hold a reference to it and
        # an explicit caller pool must keep the row in Python.
        task_ledger = (ledger if (acq or fr or alloc_demands
                                  or bindings is not None) else None)
        self._sequence += 1
        event = Event(self._sequence, name)
        waiting_count = (1 if pending_one is not None
                         else (0 if pending is None else len(pending)))
        deliv_tuple = (
            tuple((str(cid), float(amt)) for cid, amt in deliveries if float(amt) > 0.0)
            if deliveries else ()
        )
        wave_rec = self._wave_recorder
        task = _Task(
            event, prepared.demand.copy() if prepared.demand else {},
            prepared.demand,
            waiting_count, release,
            prepared.latency, prepared.caps,
            acq, fr,
            owner, the_metadata, prepared.placement, alloc_demands,
            prepared.priority, prepared.coupled, on_complete, on_start,
            None, {}, False, task_ledger, memory_replica, memory_stage, pools,
            bindings, (), prepared.cohort_size, prepared, None, native_hook,
            prepared.cohort_widened, prepared.cohort_members,
            deliv_tuple, {} if deliv_tuple else None, False,
            release if waiting_count <= 1 else None,
            pending_one.name if pending_one is not None else "",
            "publish" if pending_one is not None else "",
            None, prepared.latency_from_first_service, None,
            list(pending) if (wave_rec is not None and pending is not None) else None,
            prepared.hbm_accounting,
            {} if prepared.hbm_accounting else None,
        )
        tasks[event.sequence] = task
        if deliv_tuple:
            self._delivery_live += 1
        if prepared.hbm_accounting:
            self._hbm_acct_live += 1
        if wave_rec is not None:
            preds = (
                (pending_one.sequence,) if pending_one is not None
                else (tuple(p.sequence for p in pending) if pending is not None else ())
            )
            occ = the_metadata.get("occurrence") if the_metadata else None
            wave_rec.on_submit(
                event.sequence, name,
                aliases=(owner,) if owner else (),
                predecessors=preds,
                occurrence=occ,
            )
        if self.task_uses_fifo(task):
            self._fifo_live += 1
        if self.task_blocks_native(task):
            self._inline_blocked += 1
        if the_metadata:
            occurrence = the_metadata.get("occurrence")
            if occurrence.__class__ is OccurrenceKey:
                record = self.occurrence_metrics.setdefault(
                    str(occurrence.launch),
                    {"occurrence": occurrence, "submitted": 0, "completed": 0,
                     "service": {}, "started": 0})
                record["submitted"] += 1
        if pending_one is not None:
            pending_one._waiters.append(event.sequence)
        elif pending is not None:
            for predecessor in pending:
                predecessor._waiters.append(event.sequence)
        accel = self._accel
        if (inline_control and accel is not None and accel.can_publish_control(self)
                and not waiting_count and release <= now and not self._completable
                and not (task.remaining or task.latency)
                and self._can_allocate(task)):
            # C has paused at the grant's dependency boundary. A zero-work
            # allocation can settle now using the ordinary Python ledger.
            self._start_ready(event.sequence)
            self._completable.pop()
            self._complete(task)
            return event
        if (accel is not None
                and accel.eligible(self, task)
                and len(self._native_seqs) == len(tasks) - 1):
            pending_tup = ((pending_one,) if pending_one is not None
                           else (tuple(pending) if pending is not None else ()))
            accel.queue(self, event, task, pending_tup)
            self._native_seqs.add(event.sequence)
            if not accel.needs_external_start(task):
                if len(tasks) > self.peak_tasks:
                    self.peak_tasks = len(tasks)
                return event
        elif accel is not None and (pending_one is not None or pending is not None):
            accel.note_external_predecessors(
                self, (pending_one,) if pending_one is not None else pending)
        if pending_one is None and pending is None:
            if release <= now:
                sequence = event.sequence
                if sequence not in self._ready_seen:
                    self._ready_seen.add(sequence)
                    heapq.heappush(self._ready_heap, (prepared.priority, sequence))
            else:
                heapq.heappush(self._releases, (release, event.sequence))
        if len(tasks) > self.peak_tasks:
            self.peak_tasks = len(tasks)
        return event

    def _pending_predecessors(self, name, after, release, tasks):
        pending_one = None
        pending = None
        acls = after.__class__
        if acls is Event:
            moment = after.time
            if moment is not None:
                if release is None or moment > release:
                    release = moment
            else:
                s0 = after.sequence
                if s0 not in tasks and s0 not in self._promises:
                    raise ExecutionError(f"dependency_not_submitted:{name}")
                pending_one = after
        elif (acls is tuple or acls is list) and len(after) == 1:
            p0 = after[0]
            moment = p0.time
            if moment is not None:
                if release is None or moment > release:
                    release = moment
            else:
                s0 = p0.sequence
                if s0 not in tasks and s0 not in self._promises:
                    raise ExecutionError(f"dependency_not_submitted:{name}")
                pending_one = p0
        else:
            dependencies = after if acls is tuple else tuple(after)
            if len(dependencies) > 1:
                dependencies = tuple(dict.fromkeys(dependencies))
            for predecessor in dependencies:
                moment = predecessor.time
                if moment is not None:
                    if release is None or moment > release:
                        release = moment
                else:
                    sequence = predecessor.sequence
                    if sequence not in tasks and sequence not in self._promises:
                        raise ExecutionError(f"dependency_not_submitted:{name}")
                    if pending_one is None and pending is None:
                        pending_one = predecessor
                    elif pending is None:
                        pending = [pending_one, predecessor]
                        pending_one = None
                    else:
                        pending.append(predecessor)
        return pending_one, pending, release

    def _resolve_task_memory(self, metadata, memory, memory_pools, memory_bindings):
        context = self._memory_context
        the_metadata = dict(metadata) if metadata else {}
        if (context is None and memory is None and memory_pools is None
                and memory_bindings is None and not self.memory_pools
                and not the_metadata):
            ledger = self.memory
            pools = _EMPTY_FROZENSET
            bindings = None
            memory_replica = memory_stage = 0
        else:
            if memory is None and context is not None:
                memory = context.get("memory")
            if memory_pools is None:
                memory_pools = (context.get("pools") if context is not None
                                else the_metadata.get("memory_pools"))
            if memory_bindings is None and context is not None:
                memory_bindings = context.get("bindings")
            if memory_bindings is None:
                declared_bindings = the_metadata.get("memory_bindings")
                if declared_bindings.__class__ is dict or isinstance(
                        declared_bindings, _AbcMapping):
                    memory_bindings = declared_bindings
            ledger = memory if memory is not None else self.memory
            if ledger is not None:
                self._memory_ledgers[id(ledger)] = ledger
            if memory_pools is None:
                pools = self._memory_pools_frozen
            elif memory_pools.__class__ is frozenset:
                pools = memory_pools
            elif memory_pools:
                if isinstance(memory_pools, str):
                    memory_pools = (memory_pools,)
                pools = frozenset(str(pool) for pool in memory_pools)
            else:
                pools = _EMPTY_FROZENSET
            if not memory_bindings:
                bindings = None
            elif (context is not None
                  and memory_bindings is context.get("bindings")):
                bindings = memory_bindings
            else:
                bindings = {str(key): dict(value)
                            for key, value in memory_bindings.items()}
            memory_replica = int(the_metadata.get(
                "memory_replica", context.get("replica", 0) if context else 0))
            memory_stage = int(the_metadata.get(
                "memory_stage", context.get("stage", 0) if context else 0))
        return the_metadata, ledger, pools, bindings, memory_replica, memory_stage

    def _submit_host_launch(
        self, prepared, name: str, obs_name: str, after, owner: str = "",
    ) -> tuple[Event, Event]:
        """Submit a host launch task and its start_boundary observer in one step."""
        prepared = _refresh_prepared_for_hbm_bus(self, prepared, name)
        now = self.now
        tasks = self._tasks
        pending_one = None
        pending = None
        release = None
        if after.__class__ is Event:
            moment = after.time
            if moment is not None:
                release = moment
            else:
                s0 = after.sequence
                if s0 not in tasks and s0 not in self._promises:
                    raise ExecutionError(f"dependency_not_submitted:{name}")
                pending_one = after
        elif after:
            dependencies = after if after.__class__ is tuple else tuple(after)
            if len(dependencies) == 2 and dependencies[0] is dependencies[1]:
                dependencies = (dependencies[0],)
            elif len(dependencies) > 2:
                dependencies = tuple(dict.fromkeys(dependencies))
            for predecessor in dependencies:
                moment = predecessor.time
                if moment is not None:
                    if release is None or moment > release:
                        release = moment
                else:
                    sequence = predecessor.sequence
                    if sequence not in tasks and sequence not in self._promises:
                        raise ExecutionError(f"dependency_not_submitted:{name}")
                    if pending_one is None and pending is None:
                        pending_one = predecessor
                    elif pending is None:
                        pending = [pending_one, predecessor]
                        pending_one = None
                    else:
                        pending.append(predecessor)
        if release is None or release < now:
            release = now
        fast_ctx = self._memory_context_fast
        if fast_ctx is not None:
            ledger, pools, bindings, memory_replica, memory_stage = fast_ctx
        elif not self.memory_pools:
            ledger = self.memory
            pools = _EMPTY_FROZENSET
            bindings = None
            memory_replica = memory_stage = 0
        else:
            ledger = self.memory
            if ledger is not None:
                self._memory_ledgers[id(ledger)] = ledger
            pools = self._memory_pools_frozen
            bindings = None
            memory_replica = memory_stage = 0
        seq_launch = self._sequence + 1
        seq_obs = seq_launch + 1
        self._sequence = seq_obs
        launch = Event(seq_launch, name)
        obs = Event(seq_obs, obs_name)
        self._promises[seq_obs] = obs
        waiting_count = (1 if pending_one is not None
                         else (0 if pending is None else len(pending)))
        host_accounting = bool(
            prepared.acquire or prepared.free
            or prepared.allocation_demands or bindings is not None)
        task = _Task(
            launch, prepared.demand.copy() if prepared.demand else {},
            prepared.demand,
            waiting_count, release,
            prepared.latency, prepared.caps,
            prepared.acquire, prepared.free,
            owner, _EMPTY_METADATA, prepared.placement,
            prepared.allocation_demands,
            prepared.priority, prepared.coupled, None, None,
            None, {}, False,
            ledger if host_accounting else None, memory_replica,
            memory_stage, pools,
            bindings, (), prepared.cohort_size, prepared, obs, False,
            prepared.cohort_widened, prepared.cohort_members,
            hbm_accounting=prepared.hbm_accounting,
            hbm_credited={} if prepared.hbm_accounting else None,
        )
        tasks[seq_launch] = task
        if prepared.hbm_accounting:
            self._hbm_acct_live += 1
        wave_rec = self._wave_recorder
        if wave_rec is not None:
            preds = (
                (pending_one.sequence,) if pending_one is not None
                else (tuple(p.sequence for p in pending) if pending is not None else ())
            )
            wave_rec.on_submit(
                seq_launch, name,
                aliases=(owner,) if owner else (),
                predecessors=preds,
            )
        if self.task_uses_fifo(task):
            self._fifo_live += 1
        if self.task_blocks_native(task):
            self._inline_blocked += 1
        if pending_one is not None:
            pending_one._waiters.append(seq_launch)
        elif pending is not None:
            for predecessor in pending:
                predecessor._waiters.append(seq_launch)
        if pending_one is not None or pending is not None:
            if self._accel is not None:
                self._accel.note_external_predecessors(
                    self, (pending_one,) if pending_one is not None else pending)
        elif release <= now:
            if seq_launch not in self._ready_seen:
                self._ready_seen.add(seq_launch)
                heapq.heappush(self._ready_heap, (prepared.priority, seq_launch))
        else:
            heapq.heappush(self._releases, (release, seq_launch))
        if len(tasks) > self.peak_tasks:
            self.peak_tasks = len(tasks)
        return launch, obs

    def _instant_done_event(self, name: str) -> Event:
        """Create an already-completed causal boundary at ``self.now``."""
        self._sequence += 1
        now = self.now
        return Event(self._sequence, name, time=now, started=now)

    def future(self, name: str) -> Event:
        """A completion handle for a lazily generated finite program."""
        self._sequence += 1
        event = Event(self._sequence, name)
        self._promises[event.sequence] = event
        return event

    def boundary(self, name: str, *, after: Sequence[Event] = ()) -> Event:
        """A real zero-work causal boundary, not a fabricated duration.

        Host-submit, launch-admitted, state-visible and release markers can use
        this event instead of reusing an unrelated completion. The boundary
        becomes done with ``after`` and owns no resource work or allocation.
        """
        return self.submit(name, after=tuple(after))

    def start_boundary(self, task: Event, name: str = "") -> Event:
        """An event at the referenced task's actual start, not its completion.

        Use it to publish host submission or launch admission at the true
        boundary. An already-started task returns a completed historical
        marker at ``task.started``; an unsubmitted promise is rejected.
        """
        if not isinstance(task, Event):
            raise ExecutionError("start_boundary_requires_event")
        label = name or task.name + ":started"
        if task.started is not None:
            self._sequence += 1
            event = Event(self._sequence, label, time=task.started,
                          started=task.started)
            return event
        t_obj = self._tasks.get(task.sequence)
        if t_obj is None:
            raise ExecutionError(f"start_boundary_requires_active_task:{task.name}")
        self._sequence += 1
        event = Event(self._sequence, label)
        self._promises[event.sequence] = event
        self._start_waiters.setdefault(task.sequence, []).append(event)
        return event

    def resolve(self, event: Event) -> None:
        del self._promises[event.sequence]
        event.time = self.now
        self._wake(event)
        callbacks = event._callbacks
        if callbacks:
            event._callbacks = None
            for callback in callbacks:
                callback(event)

    def on_complete(
        self, event: Event, callback: Callable[[Event], None], *,
        native_hook: bool = False,
    ) -> None:
        """Observe one event's completion without adding a resolver task.

        A live task appends to its own callback chain; a promise stores the
        callback and fires it exactly at ``resolve``; an already finished event
        fires immediately at the current call boundary.  No event time, order
        or dependency changes.  ``native_hook=True`` marks a pure framework
        boundary observer so the C11 kernel halts at that completion instant
        instead of handing back the whole frontier.
        """
        if event.time is not None:
            callback(event)
            return
        task = self._tasks.get(event.sequence)
        if task is not None:
            previous = task.callback
            task.callback = (callback if previous is None
                             else _chain_callbacks(previous, callback))
            if native_hook:
                if previous is None:
                    task.native_hook = True
            else:
                task.native_hook = False
            if previous is None and self._accel is not None:
                self._accel.on_callback_registered(
                    self, event, native_hook=native_hook)
            elif not native_hook and self._accel is not None:
                self._accel.on_callback_registered(
                    self, event, native_hook=False)
            return
        if event.sequence in self._promises:
            if event._callbacks is None:
                event._callbacks = [callback]
            else:
                event._callbacks.append(callback)
            return
        raise ExecutionError(f"callback_target_not_submitted:{event.name}")

    def _bound_access_work_and_place(self, bound):
        res_rates: dict[str, float] = {}
        if bound.pipe_row_id and bound.pipe_row_id in self.resources:
            res_p = self.resources[bound.pipe_row_id]
            res_rates[bound.pipe_row_id] = res_p.rate / res_p.units
        if bound.l2_row_id and bound.l2_row_id in self.resources:
            res_rates[bound.l2_row_id] = self.resources[bound.l2_row_id].rate
        raw_work = bound.stage_work(resource_rates=res_rates)
        if bound.l2_row_id and bound.l2_row_id not in self.resources:
            raw_work.pop(bound.l2_row_id, None)
        place = {k: tuple(v) for k, v in bound.stage_placement().items() if k in raw_work}
        delivs = tuple(
            (str(c.counter_id), float(c.work_per_progress))
            for c in bound.deliveries if float(c.work_per_progress) > 0.0
        )
        return raw_work, place, delivs

    def _submit_allocation_boundary(self, name, stage, after, *, owner=""):
        """Keep zero-work template grants and releases at native completion boundaries."""
        done = self.future(name + ":visible")
        prepared = self.prepared_stage(stage, stage.acquire, stage.free)

        def publish(ready):
            event = self._submit_prepared(prepared, name, after=(ready,), owner=owner,
                                          metadata=dict(stage.metadata), inline_control=True)
            self.on_complete(event, lambda _: self.resolve(done))

        self.submit(name + ":allocation_ready", after=after, owner=owner,
                    on_complete=publish, native_hook=True)
        return done

    def submit_bound_access(
        self,
        name: str,
        bound,
        *,
        after: Sequence[Event] = (),
        release: float | None = None,
        owner: str = "",
        priority: int = 0,
        memory_state=None,
        pattern=None,
        on_delivery=None,
    ) -> Event:
        """Submit a coupled transfer and publish live residency after its delivery."""
        if memory_state is None or pattern is None:
            return self._submit_access_flow(name, bound, after=after, release=release,
                                            owner=owner, priority=priority)

        if not isinstance(memory_state, AdmissionMemoryView) or bound.bind_config is None:
            return self._submit_custom_access(name, bound, memory_state, pattern, after, release, owner, priority)

        done = self.future(name)
        aligned = bound.ledger.aligned_transfer_bytes
        bus = bound.bind_config[0]
        admit_extra = 0

        def admit(event, pat=None):
            nonlocal admit_extra
            if pat is None:
                pat = pattern() if callable(pattern) else pattern
            resident = memory_state.resident_coverage_for(pat, aligned)
            coverage = pat.runs.sector_coverage()
            wb_die, wb_alloc, wb_ver = _backing_identity(pat)
            if pat.direction == "read" and not coverage.empty:
                pending_wbs = memory_state.pop_matching_writebacks(wb_die, wb_alloc, wb_ver, coverage)
                if pending_wbs:
                    wb_entries = tuple(
                        (item.die, item.key.backing_allocation, item.key.tensor_version,
                         item.coverage if not item.coverage.empty else SectorCoverage(intervals=item.ranges),
                         item.bytes)
                        for item in pending_wbs
                    )
                    admit_extra += sum(item.bytes for item in pending_wbs)
                    self._launch_access_writebacks(
                        name + ":dirty_writeback", wb_entries, (event,),
                        memory_state, bus, owner, priority)
                wbs = getattr(self, "_inflight_writebacks", None)
                wb_pending = wbs.get((memory_state, wb_die, wb_alloc, wb_ver)) if wbs else None
                wb_waits = _overlapping_writebacks(wb_pending, coverage, resident) if wb_pending else ()
                if wb_waits:
                    def retry_wb(ready):
                        resolved = pat
                        if not callable(pattern) and pat.reuse_key is not None:
                            reuse = memory_state.resolve_reuse_key(pat.reuse_key)
                            if memory_state.is_resident(reuse):
                                resolved = replace(pat, reuse_key=reuse)
                        admit(ready, resolved)

                    self.submit(name + ":wait_writeback", {}, after=wb_waits, owner=owner, priority=priority,
                                on_complete=retry_wb, native_hook=True)
                    return

            key = pat.reuse_key
            identity = None
            cacheable = key is not None and aligned > 0 and pat.cache_policy != CACHE_POLICY_DISABLE
            if cacheable and key.die == pat.die and pat.direction == "read":
                identity = memory_state, key.die, key.backing_allocation, key.tensor_version
            fills = getattr(self, "_inflight_l2_fills", None)
            pending = fills.get(identity) if fills else None
            working_set = max(aligned if pat.reuse_distance_bytes is None else pat.reuse_distance_bytes,
                              _merged_bytes(coverage))
            waits = (_overlapping_fills(pending, coverage, resident)
                     if pending and working_set <= memory_state._preview_effective_l2(key) else ())
            if waits:
                def retry(ready):
                    # Primitive callers may name different reuse streams for
                    # the same allocation. Template lifetimes pin their key.
                    resolved = pat
                    if not callable(pattern):
                        reuse = memory_state.resolve_reuse_key(key)
                        if memory_state.is_resident(reuse):
                            resolved = replace(pat, reuse_key=reuse)
                    admit(ready, resolved)

                self.submit(name + ":wait_fill", {}, after=waits, owner=owner, priority=priority,
                            on_complete=retry, native_hook=True)
                return

            live = _rebind_access(bound, pat, memory_state, commit=False)
            if cacheable:
                # Other streams see the working-set pressure during the copy.
                memory_state._activate_key(key)
            ledger = live.ledger
            if not ledger.l2_hit_bytes:
                resident = SectorCoverage()
            fill = None
            if identity is not None and ledger.dram_sector_bytes and pat.cache_policy != CACHE_POLICY_LAST_USE:
                if working_set <= memory_state._preview_effective_l2(key):
                    fill = self.future(name + ":l2_fill")
                    if fills is None:
                        fills = self._inflight_l2_fills = {}
                    fills.setdefault(identity, {})[fill] = coverage, resident

            def publish(extra):
                if fill is not None:
                    del fills[identity][fill]
                    if not fills[identity]:
                        del fills[identity]
                    self.resolve(fill)
                if on_delivery is not None:
                    on_delivery(ledger, resident, extra)
                self.resolve(done)

            def complete(transfer):
                extra_self = _settle_access(memory_state, pat, ledger, resident, coverage)
                drained = memory_state.drain_writebacks(bus.die)
                entries = []
                if extra_self:
                    entries.append((wb_die, wb_alloc, wb_ver, coverage, extra_self))
                for item in drained:
                    item_cov = item.coverage if not item.coverage.empty else SectorCoverage(intervals=item.ranges)
                    entries.append((item.die, item.key.backing_allocation, item.key.tensor_version,
                                    item_cov, item.bytes))
                extra = sum(amt for _, _, _, _, amt in entries)
                if entries:
                    self._launch_access_writebacks(
                        name + ":dirty_writeback", entries, (transfer,),
                        memory_state, bus, owner, priority,
                        on_done=lambda: publish(admit_extra + extra))
                else:
                    publish(admit_extra)

            done.started = event.time
            self._submit_access_flow(name, live, after=(event,), owner=owner, priority=priority, callback=complete)

        self.submit(name + ":admit", {}, after=after, release=release, owner=owner, priority=priority,
                    on_complete=admit, native_hook=True)
        return done

    def _launch_access_writebacks(
        self, tag, entries, after_events, memory_state, bus, owner, priority, on_done=None,
    ):
        total = sum(amt for _, _, _, _, amt in entries)
        wb_done = self.future(tag + ":done")
        wbs = getattr(self, "_inflight_writebacks", None)
        if wbs is None:
            wbs = self._inflight_writebacks = {}
        registered = []
        for die, alloc, version, cov, _ in entries:
            ident = memory_state, die, alloc, version
            bucket = wbs.setdefault(ident, {})
            bucket[wb_done] = bucket[wb_done].union(cov) if wb_done in bucket else cov
            registered.append(ident)
        registered = tuple(dict.fromkeys(registered))

        def finish(_ev):
            for ident in registered:
                bucket = wbs.get(ident)
                if bucket is not None:
                    bucket.pop(wb_done, None)
                    if not bucket:
                        del wbs[ident]
            self.resolve(wb_done)
            if on_done is not None:
                on_done()

        work = {term.row_id: term.ms_per_progress for term in bus.terms_for("write", total)
                if term.ms_per_progress}
        self.submit(tag, work, after=after_events, owner=owner, priority=priority,
                    coupled=len(work) > 1, deliveries=((bus.ctr_w, total),),
                    on_complete=finish, native_hook=True)
        return wb_done

    def _submit_custom_access(self, name, bound, state, pattern, after, release, owner, priority):
        """Keep caller-owned admission protocols at their original start boundary."""
        work, placement, deliveries = self._bound_access_work_and_place(bound)

        def admit(event):
            if bound.bind_config is None:
                state.admit_access(pattern, bound.ledger.aligned_transfer_bytes)
                return
            live = _rebind_access(bound, pattern, state, commit=True)
            live_work, live_place, live_deliveries = self._bound_access_work_and_place(live)
            task = self._tasks[event.sequence]
            had_fifo, had_deliveries = self.task_uses_fifo(task), bool(task.deliveries)
            task.demand, task.remaining, task.placement = live_work, dict(live_work), live_place
            task.coupled, task.prepared = bool(live_work), None
            task.deliveries, task.delivered_acc = live_deliveries, {} if live_deliveries else None
            self._fifo_live += self.task_uses_fifo(task) - had_fifo
            self._delivery_live += bool(live_deliveries) - had_deliveries

        return self.submit(
            name, work=work, after=after, release=release, owner=owner, priority=priority,
            placement=placement, coupled=bool(work), on_start=admit, deliveries=deliveries,
        )

    def _submit_access_flow(self, name, bound, *, after, owner, priority, release=None, callback=None):
        work, placement, deliveries = self._bound_access_work_and_place(bound)
        return self.submit(
            name, work=work, after=after, release=release, owner=owner, priority=priority,
            placement=placement, coupled=bool(work), deliveries=deliveries,
            on_complete=callback, native_hook=callback is not None,
        )

    def _wake(self, event: Event, *, native: bool = False) -> None:
        frontier.wake_successors(self, event, native=native)
    def _push_ready(self, key: int) -> None:
        frontier.push_ready(self, key)
    def _push_completable(self, key: int) -> None:
        frontier.push_completable(self, key)
    def _queue(self, task: _Task) -> None:
        frontier.queue_ready(self, task)


def _rebind_access(bound, pattern, state, *, commit):
    bus, hit, miss, l2, pipe_row, l2_row, rules, nominal = bound.bind_config
    return bind_memory_access(
        pattern, state, hbm_bus=bus, copy_hit_bytes_per_ms=hit, copy_miss_bytes_per_ms=miss,
        l2_bytes_per_ms=l2, pipe_row_id=pipe_row, l2_row_id=l2_row, burst_rules=rules,
        nominal_pipe_bytes_per_ms=nominal, commit=commit,
    )


def _overlapping_writebacks(pending, requested, resident):
    waits = []
    for done, wb_cov in pending.items():
        overlap = _coverage_overlap(requested, wb_cov)
        if overlap:
            if resident.empty:
                waits.append(done)
            else:
                excluded = (_coverage_overlap(requested, resident) + _coverage_overlap(wb_cov, resident)
                            - _coverage_overlap(requested.union(wb_cov), resident))
                if overlap > excluded:
                    waits.append(done)
    return tuple(waits)


def _overlapping_fills(pending, requested, resident):
    waits = []
    for done, (filling, previously_resident) in pending.items():
        overlap = _coverage_overlap(requested, filling)
        if overlap:
            available = resident.union(previously_resident)
            excluded = (_coverage_overlap(requested, available) + _coverage_overlap(filling, available)
                        - _coverage_overlap(requested.union(filling), available))
            if overlap > excluded:
                waits.append(done)
    return tuple(waits)


def _settle_access(state, pattern, ledger, resident, coverage):
    aligned = ledger.aligned_transfer_bytes
    if pattern.direction == "read" and not resident.empty:
        current = state.resident_coverage_for(pattern, aligned)
        if current != resident and _coverage_overlap(coverage, current.union(resident)) > _coverage_overlap(coverage, current):
            # Previously warm sectors that were evicted in flight were never
            # fetched. Only the cold part can return to L2.
            cold = tuple(run for warm, run in _residency_runs(tuple(_coverage_runs(coverage)),
                                                              tuple(_coverage_runs(resident))) if not warm)
            if not cold:
                return 0
            pattern = replace(pattern, runs=replace(pattern.runs, runs=cold))
    if pattern.direction == "write" and ledger.dram_sector_bytes:
        # The original flow already published these bytes to HBM.
        pattern = replace(pattern, visibility=replace(pattern.visibility, live_out=True))
    _, miss = state.admit_access(pattern, aligned)
    if pattern.direction == "write" and ledger.l2_hit_bytes and miss:
        # Reservation or stream pressure can remove capacity during the write.
        return _merged_bytes(coverage)
    return 0
