"""Periodic finite-panel arithmetic checked against independent recurrences."""
from fractions import Fraction
from itertools import product

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix_cycle import matrix_cycle, periodic_matrix_cycle
from mindie_sim.prediction.program import Stage
from mindie_sim.prediction.refill import RefillPipeline


def literal(pattern, repeats, tail, refill_slots, consumer_slots):
    """Independent literal panel/micro history oracle, only used for short tests."""
    load_done = [Fraction(0)]
    operand_done = [Fraction(0)]
    cube_done = [Fraction(0)]
    panel_done = []
    for load, micros in tuple(pattern) * repeats + tuple(tail):
        retired = panel_done[-refill_slots] if len(panel_done) >= refill_slots else 0
        load_done.append(max(load_done[-1], retired) + Fraction(load))
        for operand, cube in micros:
            reusable = cube_done[-consumer_slots] if len(cube_done) > consumer_slots else 0
            operand_done.append(max(operand_done[-1], load_done[-1], reusable) + Fraction(operand))
            cube_done.append(max(cube_done[-1], operand_done[-1]) + Fraction(cube))
        panel_done.append(cube_done[-1])
    return float(cube_done[-1])


def engine_span(pattern, repeats, tail, refill_slots, consumer_slots):
    engine = Engine([Resource(role, 1.0) for role in ('load', 'operand', 'cube')])
    count = len(pattern) * repeats

    def panel(index):
        load, micros = pattern[index % len(pattern)] if index < count else tail[index - count]

        def steps():
            for part, (operand, cube) in enumerate(micros):
                yield (Stage('operand', {'operand': operand}),
                       Stage('cube', {'cube': cube}, after=('operand',))), part + 1 == len(micros)
        return Stage('load', {'load': load}), steps(), ()

    done = RefillPipeline(count + len(tail), refill_slots, consumer_slots, panel).start(
        engine, 'periodic', engine.submit('ready', release=7))
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.summary()['live_tasks'] == engine.summary()['live_programs'] == 0
    return done.time - 7


PATTERNS = (
    ((1., ((.25, 2.), (3., .5))), (.125, ((.5, .25),)), (4., ((0., 1.), (2., .5)))),
    ((.1, ((.01, .03),)), (.3, ((.05, .09), (.01, .12)))),
    ((0., ((0., 0.),)), (0., ((0., 1.),))),
)
TAIL = ((1.75, ((2., .25), (.125, 3.))), (.5, ((.25, .125),)), (0., ((1., .5),)))


@pytest.mark.parametrize('pattern', PATTERNS)
def test_periodic_and_finite_tail_match_literal_recurrence_across_slots(pattern):
    for repeats, refill, consumer in product((0, 1, 2, 7, 37), (1, 2, 3), (1, 2, 5)):
        result = periodic_matrix_cycle(pattern, repeats, refill, consumer, tail=TAIL)
        assert result.span_ms == literal(pattern, repeats, TAIL, refill, consumer)
        assert result.stepped_panels + result.jumped_panels == repeats * len(pattern) + len(TAIL)
        if result.period_panels:
            assert result.period_panels % len(pattern) == 0


@pytest.mark.parametrize('refill,consumer', [(1, 1), (2, 2), (3, 4)])
def test_variable_micro_lengths_use_actual_engine_dependencies(refill, consumer):
    pattern = PATTERNS[0]
    result = periodic_matrix_cycle(pattern, 9, refill, consumer, tail=TAIL)
    assert result.span_ms == pytest.approx(engine_span(pattern, 9, TAIL, refill, consumer))


def test_phase_index_prevents_a_false_cycle_when_normalized_times_already_match():
    # With one slot and zero micro work the normalized state is identical after
    # every panel, but the next panel load differs. A phase-free cycle key fails.
    pattern = tuple((float(load), ((0., 0.),)) for load in (1, 2, 5))
    result = periodic_matrix_cycle(pattern, 100_000, 1, 1)
    assert result.span_ms == 800_000
    assert result.period_panels == 3
    assert result.stepped_panels < 20
    assert result.jumped_panels > 299_980


def test_pattern_boundary_never_drains_shared_refill_and_consumer_windows():
    pattern = ((1., ((0., 2.),)), (1., ((0., 2.),)))
    result = periodic_matrix_cycle(pattern, 10)
    one = periodic_matrix_cycle(pattern, 1)
    assert result.span_ms == 41
    assert result.span_ms < one.span_ms * 10
    assert result.span_ms == engine_span(pattern, 10, (), 2, 2)


@pytest.mark.parametrize('length', [15, 16, 80])
def test_large_refill_and_scale_periods_jump_without_expanding_panels(length):
    # Distinct 3/5 or 16/80 source periods are explicit input, not averaged away.
    pattern = tuple((float(10 + (20 if i % 3 == 0 else 0) + (10 if i % 5 == 0 else 0)),
                     ((.25, 1.), (.5, 2.))) for i in range(length))
    repeats = 100_000
    tail = ((17., ((.5, 1.),)), (11., ((.25, 2.),)))
    result = periodic_matrix_cycle(pattern, repeats, tail=tail)
    expected = repeats * sum(load for load, _ in pattern) + 28 + 2.25
    assert result.span_ms == expected
    assert result.period_panels % length == 0
    assert result.stepped_panels < length * 8 + len(tail)
    assert result.stepped_panels + result.jumped_panels == repeats * length + len(tail)


def test_uniform_api_keeps_identical_cycle_witness_and_arbitrary_tail_is_continuous():
    full = (1.5, ((.25, 2.), (1., .5)))
    tail = (.75, ((.5, .125),))
    uniform = matrix_cycle(*full, 10_000, 3, 2, tail_load_ms=tail[0], tail_micro_steps=tail[1])
    periodic = periodic_matrix_cycle((full,), 10_000, 3, 2, tail=(tail,))
    assert uniform == periodic
    pure_tail = periodic_matrix_cycle((), 0, tail=TAIL)
    assert pure_tail.span_ms == literal((), 0, TAIL, 2, 2)
    assert pure_tail.prologue_panels == pure_tail.prologue_ms == 0
    assert periodic_matrix_cycle((), 0).span_ms == 0
