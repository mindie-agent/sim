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
import heapq
from typing import Callable, Sequence

from . import frontier
from .contracts import OccurrenceKey
from .engine_types import (
    Event, ExecutionError, _EMPTY_FROZENSET, _EMPTY_METADATA, _Task,
    _chain_callbacks,
)


class _EngineSubmitMixin:
    def _submit_prepared(
        self, prepared, name, *, after=(), release=None, owner="", metadata=None,
        on_complete=None, on_start=None, memory=None, memory_pools=None,
        memory_bindings=None, acquire_override=None, free_override=None,
        native_hook: bool = False, deliveries=(),
    ) -> Event:
        # Dependencies and allocations remain live invocation state. Only static
        # work, supply and placement validation can be shared by repeated tiles.
        now = self.now
        tasks = self._tasks
        pending_one = None
        pending = None
        if after:
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
            release if waiting_count == 0 else None,
            pending_one.name if pending_one is not None else "",
            "publish" if pending_one is not None else "",
        )
        tasks[event.sequence] = task
        wave_rec = self._wave_recorder
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

    def _submit_host_launch(
        self, prepared, name: str, obs_name: str, after, owner: str = "",
    ) -> tuple[Event, Event]:
        """Submit a host launch task and its start_boundary observer in one step."""
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
        )
        tasks[seq_launch] = task
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
        if self._accel is not None and (pending_one is not None or pending is not None):
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

    def on_complete(self, event: Event, callback: Callable[[Event], None]) -> None:
        """Observe one event's completion without adding a resolver task.

        A live task appends to its own callback chain; a promise stores the
        callback and fires it exactly at ``resolve``; an already finished event
        fires immediately at the current call boundary.  No event time, order
        or dependency changes.
        """
        if event.time is not None:
            callback(event)
            return
        task = self._tasks.get(event.sequence)
        if task is not None:
            previous = task.callback
            task.callback = (callback if previous is None
                             else _chain_callbacks(previous, callback))
            if not event.done and self._accel is not None:
                self._accel.on_callback_registered(self, event)
            return
        if event.sequence in self._promises:
            if event._callbacks is None:
                event._callbacks = [callback]
            else:
                event._callbacks.append(callback)
            return
        raise ExecutionError(f"callback_target_not_submitted:{event.name}")

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
    ) -> Event:
        """Submit one ``BoundAccess`` as a single coupled, owner-pinned flow."""
        res_rates: dict[str, float] = {}
        if bound.pipe_row_id and bound.pipe_row_id in self.resources:
            res_p = self.resources[bound.pipe_row_id]
            res_rates[bound.pipe_row_id] = res_p.rate / res_p.units
        if bound.l2_row_id and bound.l2_row_id in self.resources:
            res_rates[bound.l2_row_id] = self.resources[bound.l2_row_id].rate
        work = bound.stage_work(resource_rates=res_rates)
        deliveries = tuple((c.counter_id, c.work_per_progress) for c in bound.deliveries)
        on_start = None
        if memory_state is not None and pattern is not None:
            aligned = bound.ledger.aligned_transfer_bytes
            on_start = lambda _ev, ms=memory_state, pat=pattern, ab=aligned: ms.admit_access(pat, ab)
        return self.submit(
            name, work=work, after=after, release=release, owner=owner,
            placement=bound.stage_placement(), coupled=bool(work),
            priority=priority, on_start=on_start, deliveries=deliveries,
        )

    def _wake(self, event: Event, *, native: bool = False) -> None:
        frontier.wake_successors(self, event, native=native)
    def _push_ready(self, key: int) -> None:
        frontier.push_ready(self, key)
    def _push_completable(self, key: int) -> None:
        frontier.push_completable(self, key)
    def _queue(self, task: _Task) -> None:
        frontier.queue_ready(self, task)
