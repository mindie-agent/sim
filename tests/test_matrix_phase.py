"""Independent traffic and causal controls for explicit matrix phase abstraction."""
from dataclasses import replace
import math

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import HierarchicalMatmul
from mindie_sim.prediction.matrix_mapping import mapping_bound
from mindie_sim.prediction.matrix_phase import PhaseMatmul
from mindie_sim.prediction.program import RecipeBook
from test_matrix_epilogue import quantized_context
from test_matrix_mapping import setup


def execute(context, recipe, *, offset=7, trace_limit=0):
    engine = Engine(
        [Resource(name, rate, units=context.resource_units.get(name, 1))
         for name, rate in context.rates.items()],
        pools=context.pool_capacities,
        pool_units={name: context.core_count for name in context.pool_capacities},
        trace_limit=trace_limit)
    complete = recipe.lower(context).start(
        engine, "matrix", engine.submit("ready", release=offset))
    engine.advance(horizon=offset)
    assert not any(engine.serviced.values())
    engine.advance(until=(complete,))
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert not any(engine.allocated.values())
    return engine, complete.time - offset


@pytest.mark.parametrize("m,n,k,groups", [(7, 5, 3, 1), (2, 3, 2, 3)])
@pytest.mark.parametrize("reuse_inputs,reuse_weights", [(True, True), (False, True),
                                                       (True, False), (False, False)])
def test_phase_and_detailed_preserve_ragged_grouped_physical_work(
        m, n, k, groups, reuse_inputs, reuse_weights):
    _, context = setup(m=m * groups, n=n, k=k, cores=3)
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(),
                 m=m * groups, n=n, k=k, weight_count=groups,
                 quant=QuantSpec("bf16"), category="dense_mlp", phase="decode")
    context = replace(context, op=op, work={"cube": op.flops})
    phase = PhaseMatmul(((4, 4, 2),), 2, 2, reuse_inputs=reuse_inputs,
                        reuse_weights=reuse_weights)
    detailed = HierarchicalMatmul(4, 4, 2, 2, 2, reuse_inputs=reuse_inputs,
                                  reuse_weights=reuse_weights)
    compact, compact_ms = execute(context, phase)
    full, _ = execute(context, detailed)
    a, b, c = groups * m * k * 2, groups * n * k * 2, groups * m * n * 2
    expected = {
        "cube": 2 * groups * m * n * k,
        "hbm": a * (1 if reuse_inputs else math.ceil(n / 4))
               + b * (1 if reuse_weights else math.ceil(m / 4)) + c,
        "mte2": a * math.ceil(n / 4) + b * math.ceil(m / 4),
        "mte1": a * math.ceil(n / 2) + b * math.ceil(m / 2), "mte3": c,
    }
    assert compact.serviced == pytest.approx(expected)
    assert compact.serviced == pytest.approx(full.serviced)
    assert mapping_bound(context, detailed) <= compact_ms
    assert not any(compact.peak_allocated.values())


def test_phase_does_not_claim_detailed_k_overlap_or_fabricate_buffer_residency():
    records = []
    _, context = setup(m=4, n=2, k=6, cores=1, records=records)
    context = replace(context, rates={"cube": 16, "hbm": 1e12, "mte2": 1e12,
                                     "mte1": 8, "mte3": 1e12})
    phase = PhaseMatmul(((4, 2, 2),), 2, 2)
    compact, compact_ms = execute(context, phase)
    detailed, detailed_ms = execute(context, HierarchicalMatmul(4, 2, 2, 2, 2))
    # MTE1 needs 96/8=12ms and cube needs 96/16=6ms. The serial phase
    # model takes 18ms; detailed interleaving overlaps them and takes 13ms.
    assert compact_ms == pytest.approx(18, abs=1e-8)
    assert detailed_ms == pytest.approx(13, abs=1e-8)
    assert compact.serviced == pytest.approx(detailed.serviced)
    assert not any(compact.peak_allocated.values())
    assert records[0]["selected"]["tile_m"] == 4
    assert records[0]["buffer_model"].startswith("selected_tile_capacity_screening_only")


def test_phase_owner_traffic_matches_independent_ragged_tile_assignment():
    _, context = setup(m=7, n=5, k=3, cores=3)
    engine, _ = execute(context, PhaseMatmul(((4, 4, 2),), 2, 2), trace_limit=100)
    expected = {owner: 0 for owner in range(3)}
    index = 0
    for row in range(0, 7, 4):
        for column in range(0, 5, 4):
            expected[index % 3] += (min(4, 7 - row) + min(4, 5 - column)) * 3 * 2
            index += 1
    actual = {}
    for event in engine.trace:
        if "mte2" in event["demand"]:
            owner, = event["placement"]["mte2"]
            actual[owner] = actual.get(owner, 0) + event["serviced"]["mte2"]
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize("k", [2, 6])
def test_phase_output_scales_follow_matrix_and_do_not_repeat_over_k(k):
    _, context = quantized_context(k=k)
    engine, _ = execute(context, PhaseMatmul(((4, 4, 2),), 2, 2), trace_limit=100)
    detailed, _ = execute(context, HierarchicalMatmul(4, 4, 2, 2, 2))
    assert engine.serviced == pytest.approx(detailed.serviced)
    assert engine.serviced["hbm"] == pytest.approx(16 * k + 124)
    assert engine.serviced["mte2"] == pytest.approx(22 * k + 88)
    assert engine.serviced["vector"] == pytest.approx(120)
    by_owner = {}
    for event in engine.trace:
        if event["task"].endswith((":matrix", ":epilogue", ":store")):
            key, phase = event["task"].rsplit(":", 1)
            by_owner.setdefault(key, {})[phase] = event
    for phases in by_owner.values():
        assert phases["matrix"]["end_ms"] <= phases["epilogue"]["start_ms"]
        assert phases["epilogue"]["end_ms"] <= phases["store"]["start_ms"]


def test_phase_task_window_does_not_grow_with_k_and_public_recipe_remains_explicit():
    book = RecipeBook.from_mapping({"programs": {"matmul": {"kind": "phase_matmul",
        "candidates": [[4, 4, 2]], "compute_m": 2, "compute_n": 2}}})
    _, short_context = setup(m=7, n=5, k=6, cores=3)
    _, long_context = setup(m=7, n=5, k=60000, cores=3)
    short, _ = execute(short_context, book.select(short_context.op))
    long, _ = execute(long_context, book.select(long_context.op))
    assert long.serviced["cube"] == 10000 * short.serviced["cube"]
    assert long.serviced["mte1"] == 10000 * short.serviced["mte1"]
    assert long.summary()["completed_tasks"] == short.summary()["completed_tasks"]
    assert long.peak_tasks == short.peak_tasks < 30
