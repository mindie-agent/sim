"""Live-scope four-clock observer for the causal Engine.

[Role]
  Maintain the four ordered clocks for submitted/resolved events, serial or
  collective edges, fluid service intervals, periodic skips and terminal joins
  of one execution scope, then reduce them with the pure exposed-time helper.

[Invariants]
  * Event clocks join componentwise over real predecessors; internal waits add
    no separate duration and a resource wait names its actual blocker.
  * Periodic skip advances all four clocks by the proved cycle correspondence
    and delta; a changed attribution structure invalidates the skip.
  * Reference-counted side tables release entries once no live successor or
    scope needs them; no all-token timeline is retained.
  * Attribution is observation only: no service, rate or Engine state changes.

[Timeouts & Locks]
  Single-threaded callbacks; no locks, I/O or background work.

[Side-Effect Policy]
  Mutates only this observer's bounded scope tables.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from ..exposed_breakdown import (
    ExposedClocks, advance_clocks, control_clocks, cube_clocks,
    finish_breakdown, join_clocks, memory_clocks, roofline_clocks,
    unavailable_breakdown, vector_clocks, communication_clocks,
)
from ..exposed_program import (
    ROLE_COMMUNICATION, ROLE_CONTROL, ROLE_COUPLED, ROLE_CUBE, ROLE_MEMORY,
    ROLE_VECTOR, StageDescriptor, declarations_from_program,
)
from .exposed_intervals import IntervalClocks, bind_interval, interval_clocks


ZERO = 0.0
ONE = 1
FIRST = 0
INT_ZERO = 0
MAX_EVENTS = 32768
MAX_SCOPES = 64
SCOPE_DEVICE_STEP = "device_step"


@dataclass
class _Scope:
    start_ms: float
    inputs: tuple[ExposedClocks, ...] = ()
    events: dict[Any, ExposedClocks] = field(default_factory=dict)
    event_time: dict[Any, float] = field(default_factory=dict)
    metadata: dict[Any, dict[str, Any]] = field(default_factory=dict)
    releases: dict[Any, tuple[ExposedClocks, ...]] = field(default_factory=dict)
    refcounts: Counter = field(default_factory=Counter)
    service: ExposedClocks = field(default_factory=lambda: ExposedClocks(ZERO, ZERO, ZERO, ZERO))
    control_ms: float = ZERO
    opaque_control_ms: float = ZERO
    terminals: list[Any] = field(default_factory=list)
    periodic_delta: ExposedClocks | None = None
    periodic_repeats: int = INT_ZERO


def _clock_for_declaration(declaration: Any) -> tuple[ExposedClocks, float, float]:
    if isinstance(declaration, StageDescriptor):
        control, opaque = declaration.control_ms, declaration.opaque_control_ms
    elif isinstance(declaration, Mapping):
        serial_after = tuple(declaration.get("serial_after", ()) or ())
        del serial_after
        role = str(declaration.get("role", ROLE_CONTROL))
        cube = float(declaration.get("cube_ms", ZERO) or ZERO)
        vector = float(declaration.get("vector_ms", ZERO) or ZERO)
        memory = float(declaration.get("memory_ms", ZERO) or ZERO)
        communication = float(declaration.get("communication_ms", ZERO) or ZERO)
        control = float(declaration.get("control_ms", ZERO) or ZERO)
        opaque = float(declaration.get("opaque_control_ms", ZERO) or ZERO)
        if role == ROLE_COUPLED:
            clocks = roofline_clocks(cube_ms=cube, vector_ms=vector,
                                     memory_ms=memory,
                                     communication_ms=communication)
        elif role == ROLE_CUBE:
            clocks = cube_clocks(cube)
        elif role == ROLE_VECTOR:
            clocks = vector_clocks(vector)
        elif role == ROLE_MEMORY:
            clocks = memory_clocks(memory)
        elif role == ROLE_COMMUNICATION:
            clocks = communication_clocks(communication)
        else:
            clocks = vector_clocks(vector)
        clocks, control, opaque = bind_interval(
            interval=IntervalClocks(clocks), control_ms=control,
            opaque_control_ms=opaque)
        return clocks, control, opaque
    else:
        raise ModelingError("exposed_observer_declaration_unsupported")
    role = declaration.role
    if role == ROLE_COUPLED:
        clocks = roofline_clocks(
            cube_ms=declaration.cube_ms, vector_ms=declaration.vector_ms,
            memory_ms=declaration.memory_ms,
            communication_ms=declaration.communication_ms)
    elif role == ROLE_CUBE:
        clocks = cube_clocks(declaration.cube_ms)
    elif role == ROLE_VECTOR:
        clocks = vector_clocks(declaration.vector_ms)
    elif role == ROLE_MEMORY:
        clocks = memory_clocks(declaration.memory_ms)
    elif role == ROLE_COMMUNICATION:
        clocks = communication_clocks(declaration.communication_ms)
    else:
        clocks = control_clocks(declaration.control_ms)
    clocks, control, opaque = bind_interval(
        interval=IntervalClocks(clocks), control_ms=control,
        opaque_control_ms=opaque)
    return clocks, control, opaque


def _edge_keys(after: Any) -> tuple[Any, ...]:
    if after is None:
        return ()
    if isinstance(after, (str, bytes)):
        return (after,)
    if isinstance(after, Mapping):
        return tuple(after.keys())
    try:
        return tuple(after)
    except TypeError:
        return (after,)


class ExposedObserver:
    """Bounded side-table observer implementing the public contract."""

    def __init__(self, *, max_events: int = MAX_EVENTS) -> None:
        self.max_events = max(ONE, int(max_events))
        self._scopes: dict[Any, _Scope] = {}
        self._active: Any = None

    def begin_scope(self, scope_id: Any, start_ms: float, inputs: Any = ()) -> None:
        clocks = ()
        seeded: dict[Any, ExposedClocks] = {}
        if isinstance(inputs, Mapping):
            seeded = {key: value for key, value in inputs.items()
                      if isinstance(value, ExposedClocks)}
            clocks = tuple(seeded.values())
        elif isinstance(inputs, (list, tuple)):
            clocks = tuple(value for value in inputs
                           if isinstance(value, ExposedClocks))
        elif isinstance(inputs, ExposedClocks):
            clocks = (inputs,)
        self._scopes[scope_id] = _Scope(float(start_ms), clocks, events=seeded)
        self._active = scope_id
        if len(self._scopes) > MAX_SCOPES:
            oldest = next(iter(self._scopes))
            if oldest != scope_id:
                self._scopes.pop(oldest, None)

    def _scope(self, scope_id: Any | None = None) -> _Scope:
        key = self._active if scope_id is None else scope_id
        if key not in self._scopes:
            raise ModelingError(f"exposed_observer_scope_unknown:{key}")
        return self._scopes[key]

    def on_submit(self, event: Any, *, after: Any = (), release: Any = (),
                 declaration: Any = None) -> None:
        scope = self._scope()
        predecessors = [scope.events[key] for key in _edge_keys(after)
                        if key in scope.events]
        for key in _edge_keys(release):
            predecessors.extend(scope.releases.get(key, ()))
        duration, control, opaque = _clock_for_declaration(declaration)
        scope.events[event] = advance_clocks(predecessors, duration)
        scope.control_ms = max(scope.control_ms, control)
        scope.opaque_control_ms = max(scope.opaque_control_ms, opaque)
        for key in _edge_keys(after):
            if key in scope.events:
                scope.refcounts[key] += ONE
        if len(scope.events) > self.max_events:
            oldest = next(iter(scope.events))
            scope.events.pop(oldest, None)
            scope.event_time.pop(oldest, None)

    def on_service_interval(
        self, *, start_ms: float, end_ms: float, services: Mapping[str, float],
        offered_capacities: Mapping[str, float], blockers: Any = (),
    ) -> IntervalClocks:
        if end_ms < start_ms:
            raise ModelingError("exposed_observer_interval_regressed")
        interval = interval_clocks(
            delivered_work=services, offered_capacities=offered_capacities,
            blockers=blockers)
        scope = self._scope()
        elapsed = float(end_ms) - float(start_ms)
        clocks = interval.clocks
        if clocks.all_ms > ZERO:
            scale = elapsed / clocks.all_ms
            clocks = ExposedClocks(
                clocks.all_ms * scale, clocks.without_communication_ms * scale,
                clocks.compute_ms * scale, clocks.cube_ms * scale)
        scope.service = ExposedClocks(
            scope.service.all_ms + clocks.all_ms,
            scope.service.without_communication_ms + clocks.without_communication_ms,
            scope.service.compute_ms + clocks.compute_ms,
            scope.service.cube_ms + clocks.cube_ms)
        return interval

    def on_resolve(self, event: Any, *, after: Any = (), moment_ms: float) -> ExposedClocks:
        scope = self._scope()
        predecessors = [scope.events[key] for key in _edge_keys(after)
                        if key in scope.events]
        base = scope.events.get(event)
        clocks = join_clocks([*predecessors, *([base] if base else [])])
        scope.events[event] = clocks
        scope.event_time[event] = float(moment_ms)
        for key in _edge_keys(after):
            scope.refcounts[key] -= ONE
            if scope.refcounts[key] <= INT_ZERO:
                scope.refcounts.pop(key, None)
        return clocks

    def release(self, event: Any) -> None:
        scope = self._scope()
        clock = scope.events.get(event)
        if clock is not None:
            scope.releases[event] = (clock,)

    def on_periodic_skip(self, *, correspondence: Any, repeat: int,
                         clock_delta: Any) -> None:
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < ONE:
            raise ModelingError("exposed_observer_repeat_invalid")
        scope = self._scope()
        if isinstance(clock_delta, ExposedClocks):
            delta = clock_delta
        elif isinstance(clock_delta, Mapping):
            delta = ExposedClocks(
                float(clock_delta.get("all_ms", ZERO)),
                float(clock_delta.get("without_communication_ms", ZERO)),
                float(clock_delta.get("compute_ms", ZERO)),
                float(clock_delta.get("cube_ms", ZERO)))
        else:
            delta = ExposedClocks(float(clock_delta), float(clock_delta),
                                  float(clock_delta), float(clock_delta))
        scaled = ExposedClocks(
            delta.all_ms * repeat, delta.without_communication_ms * repeat,
            delta.compute_ms * repeat, delta.cube_ms * repeat)
        scope.periodic_delta = scaled
        scope.periodic_repeats = repeat
        pairs = correspondence.items() if isinstance(correspondence, Mapping) else correspondence
        for source, target in pairs:
            clock = scope.events.get(source)
            if clock is not None:
                scope.events[target] = ExposedClocks(
                    clock.all_ms + scaled.all_ms,
                    clock.without_communication_ms + scaled.without_communication_ms,
                    clock.compute_ms + scaled.compute_ms,
                    clock.cube_ms + scaled.cube_ms)

    def finish_scope(self, scope_id: Any, terminal: Any) -> dict[str, Any]:
        scope = self._scope(scope_id)
        terminals = _edge_keys(terminal)
        terminal_clocks = [scope.events[key] for key in terminals
                           if key in scope.events]
        if terminal_clocks:
            clocks = join_clocks(terminal_clocks)
        elif scope.service.all_ms > ZERO:
            clocks = scope.service
        else:
            return unavailable_breakdown(
                "exposed_observer_no_terminal_clock", boundary=SCOPE_DEVICE_STEP)
        duration = clocks.all_ms
        result = finish_breakdown(
            clocks, duration_ms=duration, boundary=SCOPE_DEVICE_STEP,
            control_ms=scope.control_ms,
            opaque_control_ms=scope.opaque_control_ms)
        result["exposed_breakdown_basis"]["periodic_skips"] = int(
            scope.periodic_repeats)
        return result

    def scope_events(self, scope_id: Any) -> dict[Any, ExposedClocks]:
        return dict(self._scope(scope_id).events)

    def program_scope(
        self, scope_id: Any, start_ms: float, descriptors: Sequence[Any],
        *, inputs: Any = (),
    ) -> Any:
        """Drive a serial program's declared descriptors for a compact scope."""
        self.begin_scope(scope_id, start_ms, inputs)
        previous: tuple[Any, ...] = ()
        terminal: Any = None
        for index, descriptor in enumerate(descriptors):
            event = f"{scope_id}:stage{index}"
            self.on_submit(event, after=previous, declaration=descriptor)
            previous = (event,)
            terminal = event
        return terminal
