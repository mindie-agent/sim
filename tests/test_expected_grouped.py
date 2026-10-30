"""Per-bank ownership controls for fractional probability-routing moments."""
from dataclasses import replace
import math

import pytest

from mindie_sim.prediction.expected_grouped import ExpectedGroupedMatmul
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.prediction.matrix_steady import SteadyMatmul
from test_matrix_epilogue import quantized_context


def hardware(rows, active, rates):
    _, context = quantized_context(m=rows, n=64, k=256,
                                   groups=active, cores=8, rates=rates)
    context = replace(context, pool_capacities={key: 1024*1024 for key in context.pool_capacities})
    engine = Engine([Resource(name, rate, units=context.resource_units[name])
                     for name, rate in context.rates.items()], pools=context.pool_capacities,
                    pool_units={key: 8 for key in context.pool_capacities})
    return engine, context


def run(*, rows=2560, active=512, matrix=None, expected=True, rates=None):
    engine, context = hardware(rows, active, rates)
    meta = dict(context.op.metadata)
    meta.update(matrix_groups=1, matrix_mapping_domain="aggregate_experts",
                routing_basis="probability_moments", expected_rows=rows,
                expected_active_experts=active)
    op = replace(context.op, metadata=tuple(meta.items()))
    records = []
    context = replace(context, op=op, record_mapping=records.append)
    matrix = matrix or TiledMatmul(128, 128, 64)
    recipe = ExpectedGroupedMatmul(matrix) if expected else matrix
    completion = recipe.lower(context).start(engine, "grouped", engine.submit("ready", release=3))
    engine.advance(horizon=2)
    assert not any(engine.serviced.values())
    engine.advance(until=(completion,))
    assert not engine._tasks and not engine._promises
    assert not engine.allocations
    assert engine.allocated == pytest.approx(dict.fromkeys(engine.allocated, 0), abs=1e-9)
    return engine, context, records, completion.time-3


@pytest.mark.parametrize("rows,active", ((2560, 512), (6400, 512), (17, 6.25), (1, .75)))
def test_bank_grid_does_not_reload_all_expert_weights_per_aggregate_m_tile(rows, active):
    new, context, records, _ = run(rows=rows, active=active)
    old, _, _, _ = run(rows=rows, active=active, expected=False)
    op = context.op
    # N fits one tile; each bank's conditional row count fits one M tile.
    # Therefore all inputs/scales/weights pass through MTE2 exactly once.
    assert math.ceil(rows/active) <= 128
    assert new.serviced["mte2"] == pytest.approx(op.mte2_bytes)
    weight = dict(op.metadata)["packed_weight_bytes"]
    assert old.serviced["mte2"] == pytest.approx(
        op.mte2_bytes + (math.ceil(rows/128)-1)*weight)
    for role, amount in {"hbm": op.hbm_bytes, "cube": op.flops,
                         "vector": op.vector_ops, "mte3": op.mte3_bytes}.items():
        assert new.serviced[role] == pytest.approx(amount)
        assert old.serviced[role] == pytest.approx(amount)
    assert records[0]["bank_slots"] == math.ceil(active)
    assert records[0]["semantic_rows"] == rows


def test_integer_equal_bank_control_matches_independent_grouped_program():
    rates = {"cube": 1e8, "vector": 1e8, "hbm": 1e8,
             "mte2": 1e7, "mte1": 1e8, "mte3": 1e8}
    actual, _, _, duration = run(rows=24, active=8, rates=rates)
    engine, context = hardware(24, 8, rates)
    done = TiledMatmul(128, 128, 64).lower(context).start(engine, "integer", engine.submit("ready"))
    engine.advance(until=(done,))
    assert actual.serviced == pytest.approx(engine.serviced)
    assert duration == pytest.approx(done.time)


def test_compact_selected_program_retains_fractional_work_and_bandwidth_response():
    durations, service = [], []
    for scale in (.5, 1, 2):
        rates = {"cube": 1e15, "vector": 1e15, "hbm": 1e14,
                 "mte2": 1e5*scale, "mte1": 1e15, "mte3": 1e15}
        engine, context, records, duration = run(rows=17, active=6.25,
                                                matrix=SteadyMatmul(), rates=rates)
        durations.append(duration)
        service.append(engine.serviced)
        assert engine.serviced["hbm"] == pytest.approx(context.op.hbm_bytes)
        assert engine.serviced["vector"] == pytest.approx(context.op.vector_ops)
        assert engine.serviced["cube"] == pytest.approx(context.op.flops)
    assert durations[0] > durations[1] > durations[2]
    assert service[0] == pytest.approx(service[1])
    assert service[0] == pytest.approx(service[2])
