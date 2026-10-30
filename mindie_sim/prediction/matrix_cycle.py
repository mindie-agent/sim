"""Nominal dependency periods for finite refill and micro-consumer windows.

[Role] Evaluate scalar completion dependencies of a regular RefillPipeline.
[Invariants] No resources, events or contention are simulated here. A jump is
  permitted only after the complete normalized dependency state repeats. Input
  floats use exact binary integer ticks, so cycle detection needs no tolerance.
  Periodic work includes its phase in that state; arbitrary finite tail panels
  continue from the actual dependency state without draining a pattern.
[Timeouts & Locks] O(pattern + tail + refill_slots + consumer_slots) state; at
  most max_steps panels to find a cycle, then less than one period plus finite
  tails. Descriptors are retained, never expanded by repetition count.
[Side-Effect Policy] Pure arithmetic; no I/O, Engine mutation or calibration.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Sequence

from mindie_sim.schema.common import ModelingError


@dataclass(frozen=True)
class MatrixCycle:
    span_ms: float
    prologue_ms: float
    prologue_panels: int
    period_ms: float
    period_panels: int
    stepped_panels: int
    jumped_panels: int


class _Dependencies:
    def __init__(self, refill_slots: int, consumer_slots: int):
        self.loaded = self.operand = self.cube = 0
        self.panels = deque([0] * refill_slots, maxlen=refill_slots)
        self.consumers = deque([0] * consumer_slots, maxlen=consumer_slots)

    def advance(self, load: int, steps: tuple[tuple[int, int], ...]) -> None:
        self.loaded = max(self.loaded, self.panels[0]) + load
        for operand, cube in steps:
            self.operand = max(self.operand, self.loaded, self.consumers[0]) + operand
            self.cube = max(self.cube, self.operand) + cube
            self.consumers.append(self.cube)
        self.panels.append(self.cube)

    def normalized(self) -> tuple[int, ...]:
        return (
            self.loaded - self.cube, self.operand - self.cube,
            *(value - self.cube for value in self.panels),
            *(value - self.cube for value in self.consumers),
        )

    def translate(self, delta: int) -> None:
        self.loaded += delta
        self.operand += delta
        self.cube += delta
        self.panels = deque((v + delta for v in self.panels), maxlen=self.panels.maxlen)
        self.consumers = deque(
            (v + delta for v in self.consumers), maxlen=self.consumers.maxlen)


def _duration(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ModelingError("matrix_cycle_requires_finite_nonnegative_durations")
    return value


def matrix_cycle(
    load_ms: float,
    micro_steps: Sequence[tuple[float, float]],
    panels: int,
    refill_slots: int = 2,
    consumer_slots: int = 2,
    *,
    tail_load_ms: float | None = None,
    tail_micro_steps: Sequence[tuple[float, float]] | None = None,
    max_steps: int = 4096,
) -> MatrixCycle:
    """Return the exact isolated three-stream dependency span.

    Each panel loads serially, then its micros pass through serial operand and
    cube streams. A load waits for the panel ``refill_slots`` positions earlier
    to retire. An operand waits for the micro ``consumer_slots`` positions
    earlier to retire. Each micro retires at cube completion.

    ``panels`` counts uniform full panels; the optional tail is additional.
    ``prologue_*`` identifies a witnessed cycle's starting completion, not a
    minimal warmup or a separate resource stage. If a finite workload finishes
    before a repeated state is found, period fields are zero and the full-panel
    span is reported as its prologue. Contention and finite allocation capacity
    beyond the declared windows remain the caller's Engine responsibility.
    """
    if (isinstance(panels, bool) or not isinstance(panels, int) or panels < 0
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in (refill_slots, consumer_slots, max_steps))):
        raise ModelingError("matrix_cycle_requires_integer_counts_and_positive_windows")
    steps = tuple((_duration(a), _duration(b)) for a, b in micro_steps)
    if not steps:
        raise ModelingError("matrix_cycle_requires_micro_steps")
    if (tail_load_ms is None) != (tail_micro_steps is None):
        raise ModelingError("matrix_cycle_requires_complete_tail")
    tail = None
    if tail_micro_steps is not None:
        tail = tuple((_duration(a), _duration(b)) for a, b in tail_micro_steps)
        if not tail:
            raise ModelingError("matrix_cycle_requires_tail_micro_steps")
    return periodic_matrix_cycle(
        ((_duration(load_ms), steps),), panels, refill_slots, consumer_slots,
        tail=((_duration(tail_load_ms), tail),) if tail is not None else (),
        max_steps=max_steps,
    )


def periodic_matrix_cycle(
    pattern: Sequence[tuple[float, Sequence[tuple[float, float]]]],
    repeats: int,
    refill_slots: int = 2,
    consumer_slots: int = 2,
    *,
    tail: Sequence[tuple[float, Sequence[tuple[float, float]]]] = (),
    max_steps: int = 4096,
) -> MatrixCycle:
    """Evaluate a periodic panel schedule with continuous finite dependencies.

    A panel is ``(load_ms, ((operand_ms, cube_ms), ...))``. ``pattern`` runs
    ``repeats`` times, followed by the finite ``tail``. The dependency queues
    remain live across pattern boundaries; no warmup/drain is inserted there.
    State repetition requires the same next-pattern position as well as the
    same normalized completion times. All counts in MatrixCycle are panels,
    not whole patterns. The tail is stepped from the jumped dependency state.

    This is the existing nominal RefillPipeline recurrence, not a contention
    engine. A finite budget bounds cycle discovery; memory depends on the
    declared pattern, tail and windows, never on the number of repetitions.
    An empty pattern is allowed only with zero repeats (a pure finite tail).
    """
    if (isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 0
            or any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                   for v in (refill_slots, consumer_slots, max_steps))):
        raise ModelingError("matrix_cycle_requires_integer_counts_and_positive_windows")

    def parse(panels):
        result = []
        for load, micros in panels:
            steps = tuple((_duration(a), _duration(b)) for a, b in micros)
            if not steps:
                raise ModelingError("matrix_cycle_requires_micro_steps")
            result.append((_duration(load), steps))
        return tuple(result)

    pattern, tail = parse(pattern), parse(tail)
    if repeats and not pattern:
        raise ModelingError("matrix_cycle_requires_nonempty_repeated_pattern")
    values = (value for load, steps in (*pattern, *tail)
              for value in (load, *(v for step in steps for v in step)))
    # Binary float denominators are powers of two, so max is their exact LCM.
    denominator = max((value.as_integer_ratio()[1] for value in values), default=1)

    def ticks(value):
        numerator, divisor = value.as_integer_ratio()
        return numerator * (denominator // divisor)

    pattern_ticks = tuple((ticks(load), tuple((ticks(a), ticks(b)) for a, b in steps))
                          for load, steps in pattern)
    tail_ticks = tuple((ticks(load), tuple((ticks(a), ticks(b)) for a, b in steps))
                       for load, steps in tail)
    panels = repeats * len(pattern_ticks)
    state = _Dependencies(refill_slots, consumer_slots)
    anchor = (0, state.normalized())
    anchor_panel = anchor_time = 0
    power, distance = 1, 0
    done = stepped = jumped = 0
    period = period_time = 0
    prologue_panel = prologue_time = 0
    while done < panels:
        if not period and stepped >= max_steps:
            raise ModelingError(f"matrix_cycle_period_not_found_within_budget:{max_steps}")
        load, micros = pattern_ticks[done % len(pattern_ticks)]
        state.advance(load, micros)
        done += 1
        stepped += 1
        if period:
            continue
        distance += 1
        normalized = (done % len(pattern_ticks), state.normalized())
        if normalized == anchor:
            period, period_time = distance, state.cube - anchor_time
            prologue_panel, prologue_time = anchor_panel, anchor_time
            cycles = (panels - done) // period
            jumped = cycles * period
            state.translate(cycles * period_time)
            done += jumped
        elif distance == power:
            # Brent retains one witness, rather than historical state snapshots.
            anchor, anchor_panel, anchor_time = normalized, done, state.cube
            power *= 2
            distance = 0
    if not period:
        prologue_panel, prologue_time = panels, state.cube
    for load, micros in tail_ticks:
        state.advance(load, micros)
        stepped += 1
    return MatrixCycle(
        span_ms=state.cube / denominator,
        prologue_ms=prologue_time / denominator,
        prologue_panels=prologue_panel,
        period_ms=period_time / denominator,
        period_panels=period,
        stepped_panels=stepped,
        jumped_panels=jumped,
    )
