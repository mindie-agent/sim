"""Independent traffic, ownership, capacity and causal controls for matrix mapping."""
from dataclasses import replace
import itertools

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import HierarchicalMatmul
from mindie_sim.prediction.matrix_mapping import MappedMatmul, owner_tiles
from mindie_sim.prediction.program import RecipeContext


def setup(m=8, n=8, k=6, *, cores=2, pools=None, copy_rate=16, records=None):
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(),
                 m=m, n=n, k=k, quant=QuantSpec("bf16"), category="dense_mlp",
                 phase="prefill")
    rates = {"cube": 128, "hbm": 1024, "mte2": copy_rate, "mte1": 1024, "mte3": 1024}
    pools = pools or {"l1": 256, "l0a": 32, "l0b": 32, "l0c": 256}
    engine = Engine([Resource(name, rate, units=1 if name == "hbm" else cores)
                     for name, rate in rates.items()], pools=pools,
                    pool_units={name: cores for name in pools})
    context = RecipeContext(
        op, {"cube": op.flops}, {name: name for name in rates}, rates, cores,
        pool_names={name: name for name in pools}, pool_capacities=pools,
        resource_units={name: 1 if name == "hbm" else cores for name in rates},
        record_mapping=records.append if records is not None else None)
    return engine, context


def run(recipe, **kwargs):
    engine, context = setup(**kwargs)
    done = recipe.lower(context).start(engine, "matrix", engine.submit("ready", release=7))
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return engine, done.time - 7


def recipe(tm=4, tn=4, *, slots=2, compute_slots=2):
    return HierarchicalMatmul(tm, tn, 2, 2, 2, slots, compute_slots)


def test_refill_reuse_reduces_only_the_outer_private_boundary():
    small, small_ms = run(recipe(2, 2))
    large, large_ms = run(recipe(4, 4))
    # A/B are 96 bytes each. The 4x4 refill uses each operand twice;
    # four 2x2 consumers still use each operand four times on MTE1.
    assert small.serviced["mte2"] == large.serviced["mte1"] == 768
    assert large.serviced["mte2"] == 384
    assert small.serviced["mte1"] == large.serviced["mte1"]
    assert small.serviced["cube"] == large.serviced["cube"] == 768
    assert small.serviced["hbm"] == large.serviced["hbm"] == 320
    assert large_ms < small_ms


def test_ragged_three_axis_tiles_preserve_all_boundaries_and_release_buffers():
    engine, _ = run(recipe(), m=7, n=5, k=3)
    a, b, c = 7 * 3 * 2, 5 * 3 * 2, 7 * 5 * 2
    assert engine.serviced == pytest.approx({
        "cube": 2 * 7 * 5 * 3, "hbm": a + b + c,
        "mte2": a * 2 + b * 2, "mte1": a * 3 + b * 4, "mte3": c})
    assert engine.peak_allocated["l1"] <= 256 * 2


def test_refill_double_buffer_overlaps_compute_without_deleting_copies():
    serial, serial_ms = run(recipe(slots=1), copy_rate=128)
    pipelined, pipelined_ms = run(recipe(slots=2), copy_rate=128)
    assert pipelined.serviced == pytest.approx(serial.serviced)
    assert pipelined_ms < serial_ms
    assert pipelined_ms > 768 / 128


def test_macro_accumulator_residency_blocks_the_next_output_wave_until_store():
    # A single macro accumulator fits exactly. Prefetch may proceed, but a
    # second output wave cannot overwrite the still-live first accumulator.
    engine, _ = run(recipe(), pools={"l1": 128, "l0a": 16, "l0b": 16, "l0c": 64})
    assert max(engine.peak_unit_allocated[("l0c", unit)] for unit in range(2)) == 64


def test_mapping_capacity_screening_and_private_service_select_different_work():
    mapped = MappedMatmul(((2, 2, 2), (4, 4, 2)), 2, 2)
    records = []
    roomy, _ = run(mapped, records=records)
    assert records[0]["selected"]["tile_m"] == 4
    constrained, _ = run(mapped, pools={"l1": 256, "l0a": 32, "l0b": 32, "l0c": 16})
    assert constrained.serviced["mte2"] == 2 * roomy.serviced["mte2"]
    assert constrained.serviced["cube"] == roomy.serviced["cube"]
    assert constrained.serviced["mte1"] == roomy.serviced["mte1"]


def test_small_work_does_not_borrow_idle_owners_in_mapping_or_execution():
    engine, context = setup(m=1, n=8, k=2, cores=4, copy_rate=4)
    mapped = MappedMatmul(((2, 2, 2), (4, 4, 2)), 2, 2)
    selected, record = mapped.select(context)
    assert selected.tile_n == 2
    bounds = {tuple(row["tile"]): row["resource_bound_ms"] for row in record["candidates"]}
    assert bounds[(4, 4, 2)] > bounds[(2, 2, 2)]
    engine, _ = run(mapped, m=1, n=8, k=2, cores=4, copy_rate=4)
    assert engine.serviced["mte2"] == 48


def test_owner_aggregate_matches_independent_tile_enumeration():
    for m, n, tm, tn, cores in itertools.product((1, 5, 8), (2, 7, 9), (2, 4), (2, 4), (1, 3, 5)):
        expected = [[0, 0, 0, 0] for _ in range(cores)]
        index = 0
        for row in range(0, m, tm):
            for col in range(0, n, tn):
                a, b = min(tm, m - row), min(tn, n - col)
                values = expected[index % cores]
                for axis, amount in enumerate((1, a, b, a * b)):
                    values[axis] += amount
                index += 1
        assert owner_tiles(m, n, tm, tn, cores) == tuple(tuple(v) for v in expected if v[0])


def test_same_mapping_runs_on_unified_memory_without_inventing_copy_pipes():
    engine, context = setup()
    names = {role: name for role, name in context.resource_names.items()
             if role in {"cube", "hbm"}}
    context = replace(context, resource_names=names, pool_names={}, pool_capacities={})
    engine = Engine([Resource(name, context.rates[name], units=2) for name in names.values()])
    done = recipe().lower(context).start(engine, "gpu", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.serviced == {"cube": 768, "hbm": 320}


def test_micro_pipeline_crosses_k_without_repeating_its_fill():
    _, context = setup(m=4, n=2, k=6, cores=1)
    rates = {"cube": 16, "hbm": 1e12, "mte2": 1e12, "mte1": 8, "mte3": 1e12}
    context = replace(context, rates=rates)
    engine = Engine([Resource(name, rate) for name, rate in rates.items()],
                    pools=context.pool_capacities)
    done = recipe(4, 2).lower(context).start(engine, "matrix", engine.submit("ready"))
    engine.advance(until=(done,))
    # Six operand transfers of 16B at 8B/ms, each feeding 1ms compute.
    # The compute overlaps the next operand across all three K panels:
    # one 1ms drain, not three separate panel drains (15ms).
    assert done.time == pytest.approx(13, abs=1e-8)


def test_long_k_keeps_only_the_active_micro_window():
    short, _ = run(recipe(), k=8)
    long, _ = run(recipe(), k=128)
    assert long.serviced["cube"] == 16 * short.serviced["cube"]
    assert long.peak_tasks == short.peak_tasks
    assert long.peak_allocated == short.peak_allocated


def test_refill_prefetch_window_is_independent_of_micro_consumer_window():
    _, context = setup(m=4, n=4, k=4, cores=1,
                       pools={"l1": 64, "l0a": 16, "l0b": 16, "l0c": 64})
    rates = {"cube": 16, "hbm": 1e12, "mte2": 32 / 3, "mte1": 1e12, "mte3": 1e12}
    context = replace(context, rates=rates)
    engine = Engine([Resource(name, rate) for name, rate in rates.items()],
                    pools=context.pool_capacities)
    done = recipe().lower(context).start(engine, "matrix", engine.submit("ready"))
    engine.advance(until=(done,))
    # Refill 0 runs 0..3, refill 1 runs 3..6 while the first four
    # micro computes run 3..7. The second four computes finish at 11.
    assert done.time == pytest.approx(11, abs=1e-8)


def test_inner_operand_ownership_prevents_a_false_large_tile_winner():
    _, context = setup(m=3, n=3, k=2, cores=2)
    rates = {"cube": 1e9, "hbm": 1e9, "mte2": 8, "mte1": 8, "mte3": 1e9}
    context = replace(context, rates=rates)
    selected, record = MappedMatmul(((2, 2, 2), (4, 4, 2)), 2, 2).select(context)
    bounds = {tuple(row["tile"]): row["resource_bound_ms"] for row in record["candidates"]}
    assert bounds == {(2, 2, 2): 7, (4, 4, 2): 12}
    assert selected.tile_m == 2


def test_batched_matrices_keep_their_weight_and_accumulator_domains():
    engine, context = setup(m=6, n=3, k=2, cores=2)
    op = gemm_op(op_id="heads", deps=(), stage=0, layer=0, requests=(),
                 m=6, n=3, k=2, weight_count=3, quant=QuantSpec("bf16"),
                 category="dense_mlp", phase="decode")
    context = replace(context, op=op, work={"cube": op.flops})
    done = recipe(4, 4).lower(context).start(engine, "heads", engine.submit("ready"))
    engine.advance(until=(done,))
    # Three independent [2,2] @ [2,3] matrices. Each matrix reads only
    # its own B. An aggregate [6,2] @ [2,3] tile is not their layout.
    assert engine.serviced == pytest.approx(
        {"cube": 72, "hbm": 96, "mte2": 60, "mte1": 84, "mte3": 36})
    assert max(v for (pool, _), v in engine.peak_unit_allocated.items() if pool == "l0c") <= 48
    assert not any(engine.allocated.values())


def test_grouped_owner_aggregate_matches_independent_tile_enumeration():
    for m, n, cores, groups in itertools.product((1, 3, 8), (2, 5), (1, 3, 5), (2, 7)):
        expected = [[0, 0, 0, 0] for _ in range(cores)]
        index = 0
        for _ in range(groups):
            for row in range(0, m, 4):
                for col in range(0, n, 2):
                    a, b = min(4, m - row), min(2, n - col)
                    for axis, amount in enumerate((1, a, b, a * b)):
                        expected[index % cores][axis] += amount
                    index += 1
        assert owner_tiles(m, n, 4, 2, cores, groups) == tuple(tuple(v) for v in expected if v[0])
