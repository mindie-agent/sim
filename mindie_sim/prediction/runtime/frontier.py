"""Dependency frontier and bounded event-interest bookkeeping.

[Role] Own the ready/release/completion ordering of one RuntimeState and the
  subscription of successor events to their predecessors.
[Invariants] Every heap entry is a live task sequence; stale entries are
  discarded on pop. Release order is (time, sequence); admission order is
  (priority, sequence). No second scheduler exists.
[Timeouts & Locks] None; callers bound settle rounds through the Engine.
[Side-Effect Policy] Mutates only the frontier containers passed in.
"""
from __future__ import annotations

import heapq
import math


def queue_ready(state, task) -> None:
    """Queue one dependency-satisfied task at its release boundary."""
    key = task.event.sequence
    if task.release <= state.now:
        push_ready(state, key)
    else:
        heapq.heappush(state._releases, (task.release, key))


def push_ready(state, key: int) -> None:
    task = state._tasks.get(key)
    if task is None or key in state._ready_seen:
        return
    state._ready_seen.add(key)
    heapq.heappush(state._ready_heap, (task.priority, key))


def pop_ready(state):
    """Pop the next admissible (priority, sequence) task, skipping stale keys."""
    while state._ready_heap:
        _, key = heapq.heappop(state._ready_heap)
        state._ready_seen.discard(key)
        task = state._tasks.get(key)
        if task is not None:
            return task
    return None


def push_completable(state, key: int) -> None:
    if key not in state._completable_seen:
        state._completable_seen.add(key)
        heapq.heappush(state._completable, key)


def take_completable(state):
    """Drain a stable screenshot of already-finished tasks."""
    batch = [heapq.heappop(state._completable)
             for _ in range(len(state._completable))]
    for key in batch:
        state._completable_seen.discard(key)
    return batch


def pop_due_releases(state) -> tuple[int, ...]:
    """Release all tasks whose declared release time is at the current tick."""
    due = []
    while state._releases and state._releases[0][0] <= state.now:
        _, key = heapq.heappop(state._releases)
        due.append(key)
    return tuple(due)


def register_after(state, event, after) -> tuple[float, tuple]:
    """Attach one new event to already-submitted predecessors.

    Returns the earliest release boundary implied by completed predecessors and
    the pending predecessor tuple. A predecessor that is neither live nor a
    registered promise is a contract error communicated as ``dependency_missing``.
    """
    release = state.now
    pending = []
    for predecessor in after:
        if predecessor.time is not None:
            release = max(release, predecessor.time)
        else:
            sequence = predecessor.sequence
            if sequence not in state._tasks and sequence not in state._promises:
                raise ValueError("dependency_not_submitted")
            pending.append(predecessor)
    for predecessor in pending:
        predecessor._waiters.append(event.sequence)
    return release, tuple(pending)


def wake_successors(state, event, *, native: bool = False) -> None:
    """Decrement only the actual successors of a completed event.

    A successor mirrored into the C kernel is owned there: a kernel completion
    already decremented its pending count, while a Python completion must
    hand the delivery to the kernel instead of touching the Python heaps.
    """
    slots = getattr(state, "_native_seqs", None)
    for successor in event._waiters:
        dependent = state._tasks.get(successor)
        if dependent is None:
            continue
        if slots is not None and successor in slots:
            if not native:
                state._native_deliver(successor)
            continue
        dependent.waiting -= 1
        if not dependent.waiting:
            dependent.blocker_event = event.name
            dependent.ready_at = state.now
            if not dependent.blocker_reason:
                dependent.blocker_reason = "publish"
            queue_ready(state, dependent)
    event._waiters.clear()


def next_release(state) -> float:
    return state._releases[0][0] if state._releases else math.inf


def live_witness(state, limit: int = 12) -> str:
    """Bounded stable witness for stall/deadlock diagnostics."""
    rows = []
    for key in sorted(state._tasks, key=lambda sequence: (
            state._tasks[sequence].priority, sequence))[:limit]:
        task = state._tasks[key]
        rows.append(
            f"{task.event.name}#{key}:remaining={dict(task.remaining)}:"
            f"waiting={task.waiting}:release={task.release - state.now:.6g}")
    return ",".join(rows) or "empty"
