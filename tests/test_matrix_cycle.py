"""Compare nominal cycle arithmetic with actual finite Engine dependencies."""
from itertools import product

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix_cycle import matrix_cycle
from mindie_sim.prediction.program import Stage
from mindie_sim.prediction.refill import RefillPipeline
from mindie_sim.schema.common import ModelingError


def _engine_span(load, micros, panels, refill_slots, consumer_slots, tail=None):
    engine = Engine([Resource(name, 1.0) for name in ("load", "operand", "cube")])

    def panel(index):
        panel_load, panel_micros = tail if index == panels else (load, micros)

        def steps():
            for part, (operand, cube) in enumerate(panel_micros):
                yield (
                    Stage("operand", {"operand": operand}),
                    Stage("cube", {"cube": cube}, after=("operand",)),
                ), part == len(panel_micros) - 1

        return Stage("load", {"load": panel_load}), steps(), ()

    ready = engine.submit("ready", release=7.0)
    end = RefillPipeline(
        panels + int(tail is not None), refill_slots, consumer_slots, panel,
    ).start(engine, "matrix", ready)
    engine.advance(until=(end,))
    assert engine.summary()["live_tasks"] == 0
    return end.time - 7.0


def test_cycle_matches_actual_refill_with_unequal_micro_work_and_tail_windows():
    for panels, refill_slots, consumer_slots, load, micros in product(
        (1, 2, 7, 31), (1, 2, 3), (1, 2, 4), (0.25, 3.0),
        (((1.0, 1.0),), ((0.25, 2.0), (3.0, 0.5), (0.0, 1.0))),
    ):
        tail = (1.75, ((2.0, 0.25), (0.125, 3.0)))
        result = matrix_cycle(
            load, micros, panels, refill_slots, consumer_slots,
            tail_load_ms=tail[0], tail_micro_steps=tail[1],
        )
        expected = _engine_span(
            load, micros, panels, refill_slots, consumer_slots, tail)
        assert result.span_ms == pytest.approx(expected)
        assert result.stepped_panels + result.jumped_panels == panels + 1


def test_three_stages_with_two_refill_slots_are_not_naive_roofline():
    actual = matrix_cycle(1.0, ((1.0, 1.0),), 4, 2, 2)
    assert actual.span_ms == 7.0
    assert actual.span_ms == _engine_span(1.0, ((1.0, 1.0),), 4, 2, 2)
    assert actual.span_ms > 3.0 + (4 - 1) * 1.0


def test_zero_operand_special_case_preserves_the_overlap_ranking():
    balanced = matrix_cycle(1.0, ((0.0, 0.5), (0.0, 0.5)), 20)
    faster_copy = matrix_cycle(0.1, ((0.0, 0.75), (0.0, 0.75)), 20)
    assert balanced.span_ms == pytest.approx(21.0)
    assert faster_copy.span_ms == pytest.approx(30.1)
    assert balanced.span_ms < faster_copy.span_ms


def test_ten_million_panels_use_a_proven_period_and_keep_exact_tail_state():
    count = 10_000_000
    result = matrix_cycle(
        1.0, ((1.0, 1.0),), count, 2, 2,
        tail_load_ms=0.5, tail_micro_steps=((0.5, 0.5),),
    )
    # Even full counts finish at 3*N/2+1. Their last load/operand/cube
    # complete at 3*N/2-1, 3*N/2, 3*N/2+1. The final short panel can
    # overlap the previous cube and then retires half a unit later.
    assert result.span_ms == 3 * count / 2 + 1.5
    assert result.period_panels == 2
    assert result.period_ms == 3.0
    assert result.stepped_panels < 16
    assert result.jumped_panels > count - 16
    assert result.stepped_panels + result.jumped_panels == count + 1


def test_zero_full_panels_and_zero_duration_cycles_do_not_invent_a_prologue():
    tail = (2.0, ((1.0, 3.0), (2.0, 1.0)))
    result = matrix_cycle(
        4.0, ((1.0, 2.0),), 0,
        tail_load_ms=tail[0], tail_micro_steps=tail[1],
    )
    assert result.prologue_ms == result.prologue_panels == 0
    assert result.span_ms == _engine_span(4.0, ((1.0, 2.0),), 0, 2, 2, tail)
    empty = matrix_cycle(4.0, ((1.0, 2.0),), 0)
    assert empty.span_ms == empty.stepped_panels == 0
    zero = matrix_cycle(0.0, ((0.0, 0.0),), 10_000_000)
    assert zero.span_ms == 0.0
    assert zero.stepped_panels < 4


def test_a_missing_cycle_fails_within_the_explicit_work_budget():
    with pytest.raises(ModelingError, match="period_not_found_within_budget:1"):
        matrix_cycle(1.0, ((1.0, 1.0),), 10_000_000, max_steps=1)
    finite = matrix_cycle(1.0, ((1.0, 1.0),), 1, max_steps=1)
    assert finite.span_ms == 3.0
    assert finite.period_panels == 0
