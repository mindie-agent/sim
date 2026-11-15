"""O(1) recurrent phase construction, closed-form sums and indexed ABI."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mindie_sim.prediction.recurrent import _sum_valid_powers
from mindie_sim.prediction.recurrent_template import (
    lower_recurrent_scan, recurrent_scan_classes,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


def context(tokens=10, chunk=4, count=None, *, setup=0.0, minimum=0.0):
    chunks = count if count is not None else (tokens + chunk - 1) // chunk
    op = OpSpec("scan", "recurrent_scan", "prefill", 0, 0, (), (tokens,),
                (tokens,), (tokens,), (tokens,), metadata=(
                    ("chunks", chunks), ("chunk", chunk), ("tokens", tokens),
                    ("heads", 4), ("chunk_vector_ops", 100.0)))
    return SimpleNamespace(
        op=op, work={"cube": 1000.0, "vector": 500.0, "hbm": 300.0},
        resource_names={"cube": "cube", "vector": "vector"},
        rates={"cube": 1000.0, "vector": 1000.0}, core_count=4,
        minimum_ms=minimum, setup_ms=setup, minimum_provenance={},
        native_work={}, service_identity={},
    )


def test_two_classes_conserve_every_declared_work_term():
    ctx = context()
    phases = recurrent_scan_classes(ctx)
    assert [phase.kind for phase in phases] == ["recurrent_full", "recurrent_tail"]
    assert len(phases) == 2
    aggregate = {"cube": 0.0, "vector": 0.0, "hbm": 0.0}
    for repeats, phase in ((2, phases[0]), (1, phases[1])):
        for stage in phase.stages:
            for key, value in stage.work.items():
                if key in aggregate:
                    aggregate[key] += repeats * value
    assert aggregate["cube"] == pytest.approx(ctx.work["cube"])
    assert aggregate["vector"] == pytest.approx(ctx.work["vector"])
    assert aggregate["hbm"] == pytest.approx(ctx.work["hbm"])


def test_indexed_template_is_constant_time_and_binds_setup_correctly():
    ctx = context(tokens=6554, chunk=64, setup=0.25, minimum=0.5)
    program = lower_recurrent_scan(ctx)
    template = program.template
    assert template.indexed_panels == 103
    assert template.setup_ms == 0.25
    assert template.minimum_ms == 0.5
    assert template.tile_factory_calls <= 2
    assert len(template.phases) == 2
    tail_start = 6554 // 64
    assert template.index_of(0) == (0, tail_start, 0, 0, 0)
    assert template.index_of(tail_start) == (1, 103, 0, tail_start, 0)
    with pytest.raises(ModelingError, match="index_out_of_range"):
        template.index_of(103)


def test_tail_phase_uses_the_real_partial_width():
    phases = recurrent_scan_classes(context(tokens=10, chunk=4))
    full, tail = phases
    assert full.stages[-1].work["cube"] == pytest.approx(1000.0 * 0.2)
    assert tail.stages[-1].work["cube"] == pytest.approx(1000.0 * 0.1)
    assert [stage.name for stage in tail.stages] == ["c1", "delta", "c2_update"]
    assert tail.stages[1].after == ("c1",)
    assert tail.stages[2].after == ("delta",)


def test_closed_form_sums_match_the_small_finite_loop():
    for tokens in (0, 1, 3, 64, 65, 6554):
        for chunk in (1, 4, 64):
            for power in (2, 3):
                expected = sum(min(chunk, tokens - offset) ** power
                               for offset in range(0, tokens, chunk))
                assert _sum_valid_powers(tokens, chunk, power) == expected
