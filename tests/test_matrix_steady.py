"""Compact matrix work and nominal recurrence compared with actual detailed execution."""
from dataclasses import replace

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import HierarchicalMatmul
from mindie_sim.prediction.matrix_steady import SteadyMatmul
from mindie_sim.prediction.program import RecipeContext


def _run(shape, recipe, *, cores=1, groups=1, load_rate=24, cube_rate=32, operand_rate=1e12):
    m, n, k = shape
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(),
                 m=m, n=n, k=k, weight_count=groups, quant=QuantSpec("bf16"),
                 category="dense_mlp", phase="prefill")
    rates = {"cube": cube_rate, "hbm": 1e12, "mte2": load_rate,
             "mte1": operand_rate, "mte3": 1e12}
    pools = {"l1": 1e7, "l0a": 1e7, "l0b": 1e7, "l0c": 1e7}
    units = {key: (1 if key == "hbm" else cores) for key in rates}
    context = RecipeContext(
        op, {"cube": op.flops}, {name: name for name in rates}, rates, cores,
        pool_names={name: name for name in pools}, pool_capacities=pools,
        pool_units={name: cores for name in pools}, resource_units=units)
    engine = Engine([Resource(name, rate, units=units[name]) for name, rate in rates.items()],
                    pools=pools, pool_units={name: cores for name in pools})
    done = recipe.lower(context).start(engine, "matrix", engine.submit("ready", release=3))
    engine.advance(horizon=3)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert not any(engine.allocated.values())
    return done.time - 3, engine


def test_steady_matrix_retains_overlap_sensitive_hardware_ranking():
    times = []
    for cube, load in ((32, 24), (32 / 1.5, 240)):
        detailed, de = _run((4, 2, 40), HierarchicalMatmul(4, 2, 2, 2, 2),
                            cube_rate=cube, load_rate=load)
        steady, se = _run((4, 2, 40), SteadyMatmul(((4, 2, 2),), 2, 2),
                          cube_rate=cube, load_rate=load)
        assert steady == pytest.approx(detailed, abs=1e-8)
        assert se.serviced == pytest.approx(de.serviced)
        assert se.completed_count < de.completed_count / 5
        times.append(steady)
    assert times == pytest.approx([21, 30.1], abs=1e-8)


@pytest.mark.parametrize("shape,cores,groups", [
    ((4, 2, 1), 1, 1), ((3, 2, 7), 1, 1), ((4, 3, 8), 1, 1),
    ((9, 7, 13), 3, 1), ((12, 7, 13), 4, 3), ((8, 2, 400), 2, 2),
])
def test_steady_matrix_preserves_tail_group_owner_work(shape, cores, groups):
    detailed, de = _run(shape, HierarchicalMatmul(4, 4, 4, 2, 2),
                        cores=cores, groups=groups, operand_rate=16)
    steady, se = _run(shape, SteadyMatmul(((4, 4, 4),), 2, 2),
                      cores=cores, groups=groups, operand_rate=16)
    assert se.serviced == pytest.approx(de.serviced)
    assert steady > 0
    if cores == groups == 1:
        assert steady == pytest.approx(detailed, abs=1e-8)


def test_steady_matrix_task_count_is_independent_of_k_panel_count():
    _, small = _run((4, 2, 40), SteadyMatmul(((4, 2, 2),), 2, 2))
    _, large = _run((4, 2, 20_000_000), SteadyMatmul(((4, 2, 2),), 2, 2))
    assert large.completed_count == small.completed_count
    assert large.peak_tasks == small.peak_tasks
    assert large.serviced["cube"] == pytest.approx(320_000_000)


@pytest.mark.parametrize("shape,groups", [
    ((11, 17, 7), 1), ((12, 17, 11), 3), ((4, 43, 40), 2),
])
def test_owner_run_preserves_nominal_serial_outputs_and_all_tail_work(shape, groups):
    kwargs = dict(groups=groups, operand_rate=20, cube_rate=32, load_rate=24)
    serial, before = _run(
        shape, SteadyMatmul(((4, 4, 4),), 2, 2, compact_output_tiles=False), **kwargs)
    compact, after = _run(shape, SteadyMatmul(((4, 4, 4),), 2, 2), **kwargs)
    assert compact == pytest.approx(serial, rel=1e-12, abs=1e-9)
    assert after.serviced == pytest.approx(before.serviced)
    assert after.completed_count < before.completed_count
    assert after.peak_unit_allocated == before.peak_unit_allocated


def test_owner_run_task_window_does_not_scale_with_output_tile_count():
    recipe = SteadyMatmul(((4, 4, 4),), 2, 2)
    _, small = _run((12, 8, 40), recipe, cores=2)
    _, large = _run((12000, 8, 40), recipe, cores=2)
    assert large.completed_count == small.completed_count
    assert large.peak_tasks == small.peak_tasks
    assert large.serviced["cube"] == pytest.approx(2 * 12000 * 8 * 40)


@pytest.mark.parametrize("k,expected", [(5, 12.775), (6, 15.55)])
def test_short_k_tail_load_overlaps_the_first_operand(k, expected):
    kwargs = dict(load_rate=3.2, cube_rate=320, operand_rate=32)
    detailed, de = _run((2, 2, k), HierarchicalMatmul(2, 2, 4, 2, 2), **kwargs)
    steady, se = _run((2, 2, k), SteadyMatmul(((2, 2, 4),), 2, 2), **kwargs)
    # The short final refill starts while the first operand is being copied.
    # Taking that first operand out of the steady body falsely serializes it.
    assert detailed == pytest.approx(expected, abs=1e-8)
    assert steady == pytest.approx(detailed, abs=1e-8)
    assert se.serviced == pytest.approx(de.serviced)


def _owned_context(cores):
    op = gemm_op(
        op_id="matrix", deps=(), stage=0, layer=0, requests=(),
        m=4 * cores, n=2, k=40, quant=QuantSpec("bf16"),
        category="dense_mlp", phase="prefill")
    rates = {"cube": 32 * cores, "hbm": 1e12, "mte2": 24 * cores,
             "mte1": 1e12, "mte3": 1e12}
    units = {key: 1 if key == "hbm" else cores for key in rates}
    pools = {"l1": 48, "l0a": 16, "l0b": 16, "l0c": 32}
    context = RecipeContext(
        op, {"cube": op.flops}, {name: name for name in rates}, rates, cores,
        pool_names={name: name for name in pools}, pool_capacities=pools,
        pool_units={name: cores for name in pools}, resource_units=units)
    engine = Engine(
        [Resource(name, rate, units=units[name]) for name, rate in rates.items()],
        pools=pools, pool_units={name: cores for name in pools}, trace_limit=200)
    return engine, context


def test_steady_residency_blocks_a_second_invocation_until_the_owner_releases():
    engine, context = _owned_context(1)
    recipe = SteadyMatmul(((4, 2, 2),), 2, 2)
    first = recipe.lower(context).start(engine, "first", engine.submit("ready"))
    second = recipe.lower(context).start(
        engine, "second", engine.submit("later-ready", release=5))
    engine.advance(until=(first,))
    assert first.time == pytest.approx(21.0, abs=1e-8)
    assert not second.done
    engine.advance(until=(second,))
    assert second.time == pytest.approx(42.0, abs=1e-8)
    second_load = next(row for row in engine.trace
                       if row["task"].startswith("second:") and row["task"].endswith(":load"))
    assert second_load["start_ms"] >= first.time - 1e-9
    assert engine.peak_unit_allocated == {
        (pool, 0): size for pool, size in context.pool_capacities.items()}
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == 0


def test_external_cube_work_slows_only_its_owner_and_preserves_all_traffic():
    stores = []
    for external in (False, True):
        engine, context = _owned_context(2)
        done = SteadyMatmul(((4, 2, 2),), 2, 2).lower(context).start(
            engine, "matrix", engine.submit("ready"))
        events = [done]
        if external:
            events.append(engine.submit(
                "external", {"cube": 640}, release=3, placement={"cube": (1,)}))
        engine.advance(until=tuple(events), first=False)
        per_owner = {
            row["placement"]["mte3"][0]: row["end_ms"]
            for row in engine.trace if "mte3" in row["demand"]
        }
        stores.append(per_owner)
        assert engine.serviced["cube"] == pytest.approx(context.op.flops + 640 * external)
        assert engine.serviced["hbm"] == pytest.approx(context.op.hbm_bytes)
        assert all(size <= context.pool_capacities[pool]
                   for (pool, _), size in engine.peak_unit_allocated.items())
        assert not any(engine.allocated.values())
        assert engine.summary()["live_tasks"] == 0
    assert stores[1][0] == pytest.approx(stores[0][0], abs=1e-8)
    assert stores[1][1] > stores[0][1]


@pytest.mark.parametrize("tile_k", [2, 4])
def test_quantized_grouped_tail_keeps_scale_work_on_the_output_boundary(tile_k):
    from test_matrix_epilogue import quantized_context

    results = []
    for recipe in (
        HierarchicalMatmul(4, 4, tile_k, 2, 2),
        SteadyMatmul(((4, 4, tile_k),), 2, 2),
    ):
        engine, context = quantized_context()
        done = recipe.lower(context).start(engine, "quantized", engine.submit("ready"))
        engine.advance(until=(done,))
        results.append(engine.serviced)
        assert not any(engine.allocated.values())
        assert engine.summary()["live_tasks"] == 0
    # Scale traffic is once per output tile even when the final K panel is short.
    assert results[1] == pytest.approx(results[0])
    assert results[1] == pytest.approx({
        "cube": 360, "vector": 120, "hbm": 220,
        "mte2": 220, "mte1": 228, "mte3": 60,
    })
