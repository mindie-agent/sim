"""Actual matrix service follows split pipes even when a legacy pipe coexists."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import HierarchicalMatmul, TiledMatmul
from mindie_sim.prediction.matrix_mapping import MappedMatmul, mapping_bound
from mindie_sim.prediction.matrix_phase import PhaseMatmul
from mindie_sim.prediction.matrix_steady import SteadyMatmul
from test_matrix_epilogue import quantized_context


RECIPES = (
    TiledMatmul(1, 1, 1),
    HierarchicalMatmul(1, 1, 1, 1, 1),
    MappedMatmul(((1, 1, 1),), 1, 1),
    PhaseMatmul(((1, 1, 1),), 1, 1),
    SteadyMatmul(((1, 1, 1),), 1, 1),
)


def execute(recipe, *, aic=1, aiv=2, generic=1e-6):
    rates = dict.fromkeys(("cube", "vector", "hbm", "mte1", "mte3"), 1e12)
    rates.update(mte2=generic, mte2_aic=aic, mte2_aiv=aiv)
    _, context = quantized_context(m=1, n=1, k=1, groups=1, cores=1, rates=rates)
    units = {**context.resource_units, "vector": 2, "mte2_aiv": 2, "mte3": 2}
    context = replace(context, resource_units=units)
    engine = Engine([Resource(name, rate, units=units[name]) for name, rate in rates.items()],
                    pools=context.pool_capacities,
                    pool_units={name: 1 for name in context.pool_capacities})
    completion = recipe.lower(context).start(engine, "matrix", engine.submit("ready", release=7))
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(completion,))
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return completion.time - 7, engine, context


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda r: type(r).__name__)
def test_declared_aic_operands_and_aiv_scales_do_not_use_legacy_generic_pipe(recipe):
    duration, engine, context = execute(recipe)
    # One INT8 A/B element, followed by two FP32 scales on two AIVs.
    # The 2-byte operand takes 2 ms, the 8-byte scale takes 4 ms.
    assert duration == pytest.approx(6, abs=1e-8)
    assert engine.serviced["mte2"] == 0
    assert engine.serviced["mte2_aic"] == 2
    assert engine.serviced["mte2_aiv"] == 8
    assert engine.serviced["hbm"] == 12
    assert engine.serviced["cube"] == 2
    assert engine.serviced["vector"] == 4
    assert mapping_bound(context, HierarchicalMatmul(1, 1, 1, 1, 1)) == 4
    fast_generic, other, _ = execute(recipe, generic=1e12)
    assert fast_generic == duration
    assert other.serviced == engine.serviced


@pytest.mark.parametrize("recipe", RECIPES, ids=lambda r: type(r).__name__)
@pytest.mark.parametrize("axis,expected", (("aic", (8, 6, 5)), ("aiv", (10, 6, 4))))
def test_independent_pipe_scan_changes_only_service_time_and_preserves_work(recipe, axis, expected):
    results = [execute(recipe, **{axis: (1 if axis == "aic" else 2) * scale})
               for scale in (.5, 1, 2)]
    assert [result[0] for result in results] == pytest.approx(expected, abs=1e-8)
    assert all(result[1].serviced == results[1][1].serviced for result in results)
