"""Fragment work admission and repeat execution.

[Role] Submit fragments with state checks, ordered repeats and real release events.
[Invariants] Checks precede admission. Repeats preserve work and allocation lifetimes.
[Timeouts & Locks] Homogeneous repeats can batch. Other repeats issue within the
  expansion limit.
[Side-Effect Policy] Submits to the caller's Engine and updates the conductor's live
  records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from mindie_sim.schema.operators import ProgramFragment

from ..engine import Event
from .state import (
    _STATE_ACTIONS_WRITE,
    FragmentContractError,
    UnsupportedFragmentSemantics,
)


@dataclass(frozen=True)
class FragmentHandle:
    """Completion handle for one fragment occurrence, not a retained history."""

    fragment: ProgramFragment
    event: Event
    release_events: tuple[Event, ...] = ()
    grant_by_name: tuple[tuple[str, Event], ...] = ()
    release_by_name: tuple[tuple[str, Event], ...] = ()
    batched: bool = False
    repeat_count: int = 1
    sequence: int = 0


@dataclass
class _LiveRecord:
    fragment: ProgramFragment
    sequence: int
    handle: FragmentHandle | None = None
    started_ms: float | None = None


def _submit_fragment(
    self, fragment: ProgramFragment, *,
    after: Sequence[Event] = (), inputs: Mapping[str, Any] | None = None,
    conditions: Mapping[str, bool] | None = None,
    holds: Mapping[str, Any] | None = None,
    event_bindings: Mapping[str, Any] | None = None,
    owner: str = "",
) -> FragmentHandle:
    """Admit one fragment. Work advances only through ``engine.advance``."""
    reasons = self.check(fragment, holds=holds)
    if reasons:
        raise UnsupportedFragmentSemantics(";".join(reasons))
    conditions = dict(conditions or {})
    holds = dict(holds or {})
    event_bindings = dict(event_bindings or {})
    self._validate_binding(fragment)
    stream = fragment.occurrence.stream if fragment.occurrence else ""
    if stream and stream not in self._streams \
            and len(self._streams) >= self.stream_limit:
        raise FragmentContractError(
            f"fragment_stream_limit_exceeded:{stream}")
    # Every rejection that depends on the event registry, allocation plan,
    # bindings or repeat bound happens before the first Engine submission
    # and before the conductor's sequence/live handle changes. The whole
    # event_out batch is checked against old names plus all new names at
    # once; no port is registered (or evicted) until the batch is known to
    # be publishable.
    output_generations = self._preflight_event_publication(
        fragment, event_bindings)
    base = self._base_dependencies(fragment, after, inputs)
    priority = self._priority(fragment)
    writes = self._active_effects(fragment, conditions, _STATE_ACTIONS_WRITE)
    reads = self._active_effects(fragment, conditions, {"read"})
    prospective_sequence = self._sequence + 1
    plan = self._plan_allocations(
        fragment, conditions, holds, prospective_sequence)
    output_plan = self._plan_event_outputs(
        fragment, plan, conditions, event_bindings)
    work = self._work(fragment)
    batchable = self._batchable(fragment, holds)
    if not batchable and fragment.repeats > self.repeat_limit:
        raise UnsupportedFragmentSemantics(
            "fragment_repeats_exceed_expansion_bound:"
            f"{fragment.repeats}:{self.repeat_limit}")
    self._sequence = prospective_sequence
    sequence = prospective_sequence
    name = f"{self.prefix}{fragment.name}#{sequence}"
    record = _LiveRecord(fragment, sequence)

    def check_reads(event: Event) -> None:
        at = event.started if event.started is not None else self.engine.now
        if record.started_ms is None:
            record.started_ms = at
        for effect in reads:
            self.state.check_read(effect.target, effect.version, at=at)

    def apply_writes(event: Event) -> None:
        at = event.time if event.time is not None else self.engine.now
        if writes:
            self.state.apply(writes, at=at, sequence=event.sequence)

    grant_by_name: dict[str, Event] = {}
    release_by_name: dict[str, Event] = {}
    release_events: list[Event] = []
    if batchable:
        def finish(event: Event) -> None:
            apply_writes(event)
            self._finish(record, event)

        work_complete = self._submit_task(
            name, self._work(fragment, fragment.repeats), base,
            (), (), priority, fragment.coupled, check_reads, finish, owner)
        batched = True
    else:
        batched = False
        work_complete = self.engine.future(name + ":work_complete")

        def finalize(event: Event) -> None:
            apply_writes(event)
            work_complete.started = (
                record.started_ms if record.started_ms is not None
                else event.started)
            self.engine.resolve(work_complete)
            self._finish(record, event)

        def issue(index: int, predecessors: Sequence[Event]) -> Event:
            first = index == 0
            last = index == fragment.repeats - 1
            # Program allocations are part of the first real admission
            # boundary, not a separate earlier grant. This gives all-or-
            # wait admission on one Engine: an unavailable Q cannot leave P
            # held by a half-admitted fragment while another fragment takes
            # Q. Program allocations stay live because the first repeat
            # does not free them; each is still released independently.
            acquire = (
                tuple(item.allocation for item in plan.program)
                if first else ()) + tuple(
                    item.allocation for item in plan.stage)
            free = tuple(item.allocation for item in plan.stage
                         if item.local_release)

            def completed(event: Event) -> None:
                if last:
                    finalize(event)
                else:
                    issue(index + 1, (event,))

            allocations = (plan.program if first else ()) + tuple(plan.stage)
            bindings = {
                planned.allocation.name: planned.memory
                for planned in allocations
                if planned.memory is not None}
            task = self._submit_task(
                f"{name}:r{index}", work, predecessors,
                acquire, free, priority, fragment.coupled,
                check_reads, completed, owner,
                memory_bindings=bindings)
            if first:
                for grant_index, planned in enumerate(plan.program):
                    grant_by_name[planned.name] = self.engine.start_boundary(
                        task, f"{name}:program_grant{grant_index}")
                if fragment.repeats == 1:
                    for grant_index, planned in enumerate(plan.stage):
                        grant_by_name[planned.name] = (
                            self.engine.start_boundary(
                                task, f"{name}:stage_grant{grant_index}"))
            return task

        issue(0, base)

        # One release task per allocation. Independent holds never merge
        # into one max-release; a local program release is not delayed by
        # an unrelated external hold.
        for index, planned in enumerate(plan.program):
            dependencies = ((work_complete,) if planned.external_hold is None
                            else (work_complete, planned.external_hold))
            release = self.engine.submit(
                f"{name}:release{index}", after=dependencies,
                free=(planned.allocation,), owner=owner)
            release_by_name[planned.name] = release
            release_events.append(release)
        for index, planned in enumerate(plan.stage):
            if planned.external_hold is None:
                if planned.local_release and fragment.repeats == 1:
                    release_by_name.setdefault(planned.name, work_complete)
                continue
            release = self.engine.submit(
                f"{name}:stage_release{index}",
                after=(work_complete, planned.external_hold),
                free=(planned.allocation,), owner=owner)
            release_by_name[planned.name] = release
            release_events.append(release)

    handle = FragmentHandle(
        fragment=fragment, event=work_complete,
        release_events=tuple(release_events),
        grant_by_name=tuple(grant_by_name.items()),
        release_by_name=tuple(release_by_name.items()),
        batched=batched, repeat_count=fragment.repeats, sequence=sequence)
    record.handle = handle
    self._live[work_complete.sequence] = record
    output_events = self._resolve_output_events(
        fragment, output_plan, work_complete, grant_by_name,
        release_by_name, event_bindings)
    self._record_outputs(fragment, output_events, output_generations)
    if stream:
        self._streams[stream] = work_complete
    return handle
