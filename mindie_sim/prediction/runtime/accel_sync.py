"""Adoption, output-drain and hand-back ownership of the mirrored frontier.

[Role] Own the two state journeys of the C accelerator: mirroring an unstarted
  pure Python frontier into the kernel (``adopt_frontier``) and returning every
  kernel-owned row to the canonical Python Engine (``sync_back``).  Also owns
  ``drain_output``: publish every same-instant output continuation without
  consuming caller step budget or integrating another interval.
[Invariants] Exactly one frontier owns a task at a time.  Adoption removes the
  adopted sequences from Python's ready/release heaps, so the external release
  boundary handed to the kernel only describes non-mirrored work.  Hand-back
  restores the reference state from the exported kernel residue: ``remaining``
  keeps only strictly positive keys, cumulative ``service`` receipts are
  preserved, readiness is recomputed from the real predecessor Events, and a
  future release is re-inserted exactly once.  No step is ever replayed.
  Capacity preflight counts the solver's *effective* unit memberships: an
  omitted placement consumes every declared unit, exactly like ``unit_len()``
  in the kernel, while placement-pool storage counts explicit tuples only.
[Timeouts & Locks] One bounded ctypes export per live row and one bounded
  native drain call per output batch; no threads, sleeps, locks or I/O.
[Side-Effect Policy] Mutates only the Engine frontier and the backend's own
  bookkeeping tables.
"""
from __future__ import annotations

import ctypes
import heapq
import math
import os

from .accel_tables import (
    B_OUTPUT, E_STATE, INT32_MAX, KernelFault, MAX_COEF, MAX_F, MAX_L, MAX_MROW,
    MAX_OWN, MAX_SENT, MAX_T, MAX_UENT, pure_task, task_flags,
)


def finish_handback(backend, engine) -> None:
    """Export only after every accepted output record was published.

    ``disable()`` records the request during ``_apply()`` or a drain loop and
    returns without touching the kernel.  The outermost publication exit calls
    this entry point; a nested ``_apply``/drain leaves the request pending.
    A batch that aborted before publishing every accepted record never
    exports its partially popped kernel state, so the caller sees the primary
    exception instead of a secondary ``stale_slot`` error.
    """
    if backend.aborted_publication:
        return
    if not backend.pending_handback:
        return
    if backend.applying or backend.draining:
        return
    backend.pending_handback = False
    backend.handing_back = True
    try:
        sync_back(backend, engine)
    finally:
        backend.handing_back = False
    assert not (backend.pending or engine._native_seqs or engine._native_slots)


def deliver_slots(backend) -> None:
    """Publish every queued dropped-slot notification in one kernel call."""
    slots = [backend.engine_slots[sequence] for sequence in backend.deliver
             if sequence in backend.engine_slots]
    backend.deliver.clear()
    if slots:
        backend.arena.call("msim_ctx_deliver", slots)


def reclaim_arena(backend, engine) -> None:
    """Reset the arena at an empty boundary and rebuild live accounting."""
    backend.arena.reset()
    backend.seeded = False
    backend.deliver.clear()
    backend.drops.clear()
    backend.external.clear()
    backend.usage = {
        event.sequence: task_usage(engine, task, pending)
        for event, task, pending in backend.pending}
    backend.links = sum(use[0] for use in backend.usage.values())
    backend.units = sum(use[1] for use in backend.usage.values())
    backend.edges = sum(use[2] for use in backend.usage.values())
    backend.storage = sum(use[3] for use in backend.usage.values())
    backend.epoch_links = backend.links
    backend.epoch_units = backend.units
    backend.epoch_edges = backend.edges
    backend.epoch_storage = backend.storage
    backend.epoch_tasks = len(backend.pending)
    backend.coupled_seqs = {
        event.sequence for event, task, _pending in backend.pending
        if task.coupled}
    backend.fair_prio = (backend.pending[0][1].priority
                         if (backend.has_fair_res and backend.pending) else None)


def flush_pending(backend, engine) -> None:
    """Mirror every queued row into the kernel at the current instant."""
    if backend.external:
        backend.arena.call("msim_ctx_mark_external", backend.external)
        backend.external.clear()
    if backend.drops:
        backend.arena.call("msim_ctx_drop", backend.drops)
        backend.drops.clear()
    if (not engine._native_slots and backend.pending
            and (backend.epoch_storage > 65536 or backend.epoch_links > MAX_L
                 or backend.epoch_tasks > MAX_T)):
        reclaim_arena(backend, engine)
    if not backend.pending:
        if backend.deliver:
            deliver_slots(backend)
        return
    index = {name: i for i, name in enumerate(backend.res_names)}
    slots = engine._native_slots
    rows = []
    for event, task, pending in backend.pending:
        wset = set(task.cohort_widened) if (task.cohort_size > 1 and task.cohort_widened) else ()
        links = []
        for name, remaining in task.remaining.items():
            units = task.placement.get(name, ()) if task.placement else ()
            links.append((index[name], remaining,
                          task.caps.get(name, math.inf), units, name in wset))
        flags = task_flags(task, engine)
        task.waiting = sum(
            1 for predecessor in pending if not predecessor.done)
        rows.append((
            event.sequence, flags, int(task.priority), int(task.cohort_size),
            task.waiting, engine.now, task.latency, task.release, links, 0))
    assigned = backend.arena.submit(rows)
    edges = []
    for (event, _task, pending), slot in zip(backend.pending, assigned):
        slots[event.sequence] = slot
        del pending
    for (event, _task, pending) in backend.pending:
        for predecessor in pending:
            pred_slot = slots.get(predecessor.sequence)
            if pred_slot is not None:
                edges.append((pred_slot, slots[event.sequence], event.sequence))
    if edges:
        backend.arena.link(edges)
    backend.pending.clear()
    if backend.deliver:
        deliver_slots(backend)


def drain_output(backend, engine) -> int:
    """Publish every remaining same-instant output batch.

    Each drain call passes ``max_steps=0`` and ``horizon=ctx->now``, so it
    consumes no caller step budget and can never integrate a positive-dt
    interval.  A callback registration during ``_apply`` records a pending
    hand-back through ``disable``; the loop keeps publishing every
    already-accepted record and only the outermost exit exports the genuinely
    live residue, after which Python settles rows the callback submitted.
    A failing allocation inside a
    zero-budget call follows the same exactly-once prefix publication and
    error translation as every other native call.  The kernel itself
    resumes an interrupted completion snapshot, so this loop just supplies the
    next zero-budget output window.  A disable requested mid-batch keeps
    the loop alive until C has emitted every already-accepted record, and only
    then exports the genuinely live residue.
    """
    extra = 0
    backend.draining = True
    try:
        while backend.enabled or backend.pending_handback:
            try:
                now, boundary, steps, ncomp, nstart = backend.arena.run(
                    engine.now, engine.now, engine._frontier_next_release(), 0)
            except KernelFault as error:
                backend._apply_prefix(engine, error)
                backend._translate_fault(error)
            backend.native_calls += 1
            backend.native_steps += steps
            extra += steps
            engine.now = now
            engine.advance_steps += steps
            backend._sync_counters(engine)
            from .accel_terms import drain_obs_bridge, record_native_starts
            record_native_starts(backend, engine, nstart)
            drain_obs_bridge(backend, engine)
            backend._apply(engine, nstart, ncomp)
            # Every completion and trace eviction published by this
            # chunk must settle before another drain call can admit an awakened
            # child.  A retained release-sensitive row hands back instead.
            if backend._trace_blocks_epoch(engine):
                break
            if boundary != B_OUTPUT:
                break
    finally:
        backend.draining = False
        finish_handback(backend, engine)
    return extra


def resolve_mode(requested: str | None) -> str:
    """Resolve ``python`` / ``c`` / ``auto`` from the argument or environment."""
    value = requested
    for source in ("MINDIE_SIM_ACCEL", "MINDIE_SIM_ENGINE_BACKEND"):
        if value is not None:
            break
        value = os.environ.get(source)
    text = str(value if value is not None else "auto").strip().lower()
    if text in {"python", "py", "off", "0"}:
        return "python"
    if text in {"c", "c11", "native", "on", "1"}:
        return "c"
    if text in {"auto", ""}:
        return "auto"
    raise ValueError(f"engine_backend_invalid:{text}")


#: Minimum live pure rows an ``auto`` Engine must hold before a re-adoption is
#: even attempted; mirrors ``accel_bridge.AUTO_MIN_TASKS`` without importing it.
REARM_MIN_TASKS = 64
#: A rejected re-adoption scans the live frontier, so repeated failures back off
#: geometrically (bounded) instead of running once per submitted row.
REARM_BACKOFF_CAP = 1024
REARM_BACKOFF_SHIFT = 4
#: Bounded futile probes before ``auto`` stops retrying a frontier that is
#: structurally unmirrorable (a pending promise edge or a Python start row).
REARM_REJECT_LIMIT = 4


def rearm_delay(sequence: int, failures: int) -> int:
    """Next sequence threshold after ``failures`` rejected re-adoptions."""
    shift = min(int(failures), REARM_BACKOFF_SHIFT)
    return int(sequence) + min(REARM_BACKOFF_CAP, REARM_MIN_TASKS << shift)


def note_rearm_rejection(backend, engine) -> None:
    """Record one rejected adoption and stop after the bounded probe budget."""
    backend.rearm_failures += 1
    backend.rearm_after = rearm_delay(engine._sequence, backend.rearm_failures)
    backend.rearm_rejections += 1
    if backend.rearm_rejections >= REARM_REJECT_LIMIT:
        backend.recoverable = False


def can_rearm(backend, engine) -> bool:
    """Cheap, exact gate for a recoverable ``auto`` hand-back.

    Adoption needs a settled frontier (no live or mirrored native row), a
    workload that amortizes the call, and a sequence position past the current
    backoff window.  ``_promises`` are separate completion handles, never
    pending work, so they are never inspected or subtracted.
    """
    if not backend.recoverable or backend.mode != "auto":
        return False
    if backend.handing_back or backend.enabled or backend.pending:
        return False
    if engine._active or engine._native_slots or engine._native_seqs:
        return False
    if engine._inline_blocked:
        return False
    if engine._sequence < REARM_MIN_TASKS:
        return False
    return (engine._sequence >= backend.rearm_after
            and len(engine._tasks) >= REARM_MIN_TASKS)


def solver_capacity_exceeded(tasks: int, links: int, units: int,
                             coupled: bool) -> bool:
    """Conservative preflight of every remaining C solver scratch limit.

    ``links`` is an upper bound on flow and coefficient counts (one flow and at
    most one coefficient per live link); ``units`` is the *effective* unit
    membership (default placements consume every declared unit) and therefore
    bounds both the independent overlap scratch and the coupled owner/entry
    tables.  A frontier that fails this test stays in Python, never faults.
    """
    if tasks > MAX_T or links > MAX_L or units > MAX_UENT:
        return True
    if coupled and (tasks > MAX_F or links > MAX_F or links > MAX_COEF
                    or links > MAX_MROW or units > MAX_SENT):
        return True
    return False


def explicit_units(task) -> int:
    """Placement-pool cells stored for the live links (explicit tuples only)."""
    placement = task.placement or {}
    return sum(len(placement[name]) for name in task.remaining
               if name in placement)


def effective_units(engine, task) -> int:
    """Solver unit memberships for every live work demand on one task.

    An omitted placement means "all declared units" to ``unit_len()`` in the
    kernel, so it must contribute ``resource.units`` here as well.  Counting
    only explicit tuples undercounted the overlap/coupled solver demand and let
    the kernel fault after admission instead of declining in preflight.
    """
    placement = task.placement or {}
    total = 0
    for name in task.remaining:
        units = placement.get(name)
        if units is None:
            resource = engine.resources.get(name)
            total += resource.units if resource is not None else 0
        else:
            total += len(units)
    return total


def task_usage(engine, task, pending) -> tuple[int, int, int, int]:
    """``(links, effective_units, edges, explicit_storage)`` for one row."""
    return (len(task.remaining), effective_units(engine, task), len(pending),
            explicit_units(task))


def purge_adopted(engine, sequences) -> None:
    """Remove newly mirrored sequences from the Python-owned frontier heaps."""
    adopted = set(sequences)
    if not adopted:
        return
    if engine._releases:
        kept = [entry for entry in engine._releases if entry[1] not in adopted]
        if len(kept) != len(engine._releases):
            engine._releases[:] = kept
            heapq.heapify(engine._releases)
    if engine._ready_heap:
        kept = [entry for entry in engine._ready_heap if entry[1] not in adopted]
        if len(kept) != len(engine._ready_heap):
            engine._ready_heap[:] = kept
            heapq.heapify(engine._ready_heap)
    engine._ready_seen.difference_update(adopted)


def adopt_frontier(backend, engine, skip: int) -> bool:
    """Mirror the unstarted pure frontier when ``auto`` attaches."""
    items = [task for _, task in sorted(engine._tasks.items())
             if task.event.sequence != skip]
    if not items:
        return True
    if engine._active or engine._native_slots or backend.pending:
        return False
    if backend.has_fair_res and len({task.priority for task in items}) > 1:
        backend.disabled_reason = "mixed_priority"
        return False
    for task in items:
        if task.start is not None or task.event.done or not pure_task(engine, task):
            return False
    predecessors = {task.event.sequence: [] for task in items}
    for task in items:
        for successor in task.event._waiters:
            if successor in predecessors:
                predecessors[successor].append(task.event)
    for task in items:
        if task.waiting != len(predecessors[task.event.sequence]):
            return False
    nlinks = sum(len(task.remaining) for task in items)
    nunits = sum(effective_units(engine, task) for task in items)
    nstore = sum(explicit_units(task) for task in items)
    nedges = sum(len(preds) for preds in predecessors.values())
    coupled = any(task.coupled for task in items)
    if nedges > MAX_OWN or nstore > INT32_MAX:
        backend.disabled_reason = "capacity_exceeded"
        return False
    if solver_capacity_exceeded(len(items), nlinks, nunits, coupled):
        backend.disabled_reason = "capacity_exceeded"
        return False
    for task in items:
        backend.queue(engine, task.event, task,
                      tuple(predecessors[task.event.sequence]))
        engine._native_seqs.add(task.event.sequence)
    # The kernel now owns these releases too: leaving them in Python would pass
    # an already-consumed boundary back as ``next_release``.
    purge_adopted(engine, (task.event.sequence for task in items))
    return True


def _queue_release(engine, task, releases: set[int]) -> None:
    """Insert one future release exactly once (sequence-deduplicated)."""
    sequence = task.event.sequence
    if sequence in releases:
        return
    heapq.heappush(engine._releases, (task.release, sequence))
    releases.add(sequence)


def requeue_pending(engine, task, pending, releases: set[int]) -> None:
    """Recompute readiness for one never-submitted mirrored row."""
    waiting = sum(1 for predecessor in pending or () if not predecessor.done)
    task.waiting = waiting
    if waiting or task.start is not None:
        return
    sequence = task.event.sequence
    if task.release <= engine.now:
        if sequence not in engine._ready_seen:
            engine._ready_seen.add(sequence)
            heapq.heappush(engine._ready_heap, (task.priority, sequence))
    else:
        _queue_release(engine, task, releases)


def sync_back(backend, engine) -> None:
    """Return every kernel-owned row to the reference frontier."""
    arena = backend.arena
    now = engine.now
    if backend.seeded and arena is not None:
        # callbacks below must observe this run's accepted counters.
        backend._sync_counters(engine)
    if arena is not None:
        for sequence, slot in tuple(engine._native_slots.items()):
            task = engine._tasks.get(sequence)
            if task is None:
                continue
            names = tuple(task.remaining)
            n = len(names)
            buf = (ctypes.c_double * max(2 * n, 1))()
            got = int(arena.library.msim_ctx_export(arena.ctx, slot, buf, 2 * n))
            if got < 0 or got != n:
                # an already-completed C slot must have been settled by
                # its accepted completion record, never turned back into
                # positive Python remaining work or silently dropped.
                raise KernelFault(E_STATE, "export", f"stale_slot:{sequence}")
            for index, name in enumerate(names):
                task.service[name] = task.service.get(name, 0.0) + float(
                    buf[n + index])
                remaining = float(buf[index])
                # The Python reference deletes an exhausted key; a stored 0.0
                # re-enters the coupled solver as a zero scale.
                if remaining > 0.0:
                    task.remaining[name] = remaining
                else:
                    task.remaining.pop(name, None)
    releases = {entry[1] for entry in engine._releases}
    settled: list[int] = []
    for sequence in tuple(engine._native_slots):
        task = engine._tasks.get(sequence)
        if task is None:
            continue
        task.waiting = sum(
            1 for predecessor in backend.deps.get(sequence, ())
            if not predecessor.done)
        if task.start is None:
            if not task.waiting:
                if task.release <= now:
                    if sequence not in engine._ready_seen:
                        engine._ready_seen.add(sequence)
                        heapq.heappush(engine._ready_heap,
                                       (task.priority, sequence))
                else:
                    _queue_release(engine, task, releases)
        elif not task.event.done:
            was_active = sequence in engine._active
            engine._active.add(sequence)
            if task.latency > 0.0:
                engine._latency_active.add(sequence)
            if not was_active:
                if task.coupled:
                    engine._coupled_active += 1
                if task.placement:
                    engine._placed_active += 1
            if not task.remaining and now >= task.start + task.latency:
                # this row already belonged to the kernel's current
                # completion snapshot.  Complete it in sequence order after the
                # hand-back, before Python may start successors awakened by
                # earlier records (``engine.py``'s snapshot-drain order).
                settled.append(sequence)
    for _event, task, pending in tuple(backend.pending):
        requeue_pending(engine, task, pending, releases)
    engine._native_seqs.clear()
    engine._native_slots.clear()
    backend.pending.clear()
    backend.deliver.clear()
    backend.drops.clear()
    backend.external.clear()
    backend.engine_slots = engine._native_slots
    backend.live = 0
    backend.deps.clear()
    backend.usage.clear()
    backend.coupled_seqs.clear()
    backend.fair_prio = None
    backend.callback_dirty = False
    backend.links = backend.units = backend.edges = 0
    backend.storage = 0
    backend.epoch_links = backend.epoch_units = 0
    backend.epoch_storage = 0
    backend.epoch_tasks = backend.epoch_edges = 0
    # every native table is gone, so these already-complete snapshot
    # members wake the restored Python frontier exactly once.  Doing this here
    # (rather than leaving them in ``_completable``) keeps the trace order:
    # the interrupted batch settles before awakened successors may start.
    for sequence in sorted(settled):
        task = engine._tasks.get(sequence)
        if task is None or task.event.done:
            continue
        if task.remaining or now < task.start + task.latency:
            continue
        engine._complete(task, moment=now)
    # Postcondition: no callback above may leave work in the hand-back.
    assert not backend.pending
    assert not engine._native_seqs
    assert not engine._native_slots
