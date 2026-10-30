"""Independent traffic and owner/dependency controls for matrix output paths."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import HierarchicalMatmul
from mindie_sim.prediction.matrix_output import PartitionedMatrixOutput
from test_matrix_epilogue import quantized_context


def setup(*, m=3, n=5, k=4, groups=1, rates=None):
    rates = rates or {"hbm": 1024, "cube": 128, "mte1": 128, "mte2_aic": 64,
                      "mte2_aiv": 64, "vector": 32, "mte3": 128, "fixpipe": 64}
    _, context = quantized_context(m=m, n=n, k=k, groups=groups, cores=1, rates=rates)
    units = {name: 2 if name in {"vector", "mte2_aiv", "mte3"} else 1
             for name in rates}
    pools = {**context.pool_capacities, "ub": 4096}
    pool_units = {name: 2 if name == "ub" else 1 for name in pools}
    context = replace(context, resource_units=units, pool_capacities=pools,
                      pool_names={name: name for name in pools}, pool_units=pool_units)
    engine = Engine([Resource(name, rate, units=units[name]) for name, rate in rates.items()],
                    pools=pools, pool_units=pool_units, trace_limit=2000)
    return engine, context


OUTPUT = PartitionedMatrixOutput("fixpipe", 2, 2, 8)


def test_output_handoff_charges_padded_publication_and_private_scale_replication():
    engine, context = setup()
    done = HierarchicalMatmul(4, 8, 2, 2, 2).lower(
        context, output=OUTPUT, serial_tiles=True).start(
            engine, "partitioned", engine.submit("ready", release=7))
    engine.advance(horizon=6)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    # M=3,N=5,K=4: 12B A,20B B,12B row scales,20B column scales,
    # 30B useful BF16 output. Two vectors each read the same 20B column scale.
    assert engine.serviced == pytest.approx({
        "hbm": 94, "cube": 120, "mte1": 76, "mte2_aic": 32,
        "mte2_aiv": 52, "vector": 60, "mte3": 30, "fixpipe": 80,
    })
    assert engine.peak_unit_allocated[("ub", 0)] == 160
    assert engine.peak_unit_allocated[("ub", 1)] == 160
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_one_row_uses_one_vector_and_next_tile_waits_for_output_acknowledgement():
    rates = dict.fromkeys(("hbm", "cube", "mte1", "mte2_aic", "mte2_aiv", "mte3"), 1e12)
    rates.update(vector=8, fixpipe=32)
    engine, context = setup(m=1, n=8, k=2, rates=rates)
    done = HierarchicalMatmul(2, 4, 2, 2, 2).lower(
        context, output=OUTPUT, serial_tiles=True).start(
            engine, "serial-output", engine.submit("ready"))
    engine.advance(until=(done,))
    # Per tile: 32B publication / 32B/ms + 16ops / (8ops/ms / 2 owners).
    # The inactive second AIV cannot supply the first, and the next tile waits.
    assert done.time == pytest.approx(10, abs=1e-8)
    assert engine.serviced["vector"] == 32
    assert engine.serviced["fixpipe"] == 64
    assert engine.serviced["mte3"] == 16
    vector_rows = [row for row in engine.trace
                   if row.get("demand", {}).get("vector")]
    assert vector_rows
    assert all(row["placement"]["vector"] == (0,) for row in vector_rows)


def test_explicit_owner_rectangles_preserve_group_work_without_cross_group_weight_reuse():
    engine, context = setup(m=6, n=5, k=4, groups=2)
    # One owner, independent equal-row experts; N split differs from generic tiling.
    sizes = ((3, 2), (3, 3), (3, 2), (3, 3))
    done = HierarchicalMatmul(4, 4, 2, 2, 2).lower(
        context, owner_tiles=((4, iter(sizes)),), output=OUTPUT,
        serial_tiles=True).start(engine, "source-order", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.serviced["cube"] == 240
    assert engine.serviced["vector"] == 120
    assert engine.serviced["hbm"] == pytest.approx(188)
    assert engine.serviced["mte2_aic"] == 88  # A copied twice per expert, B once.
    assert engine.serviced["fixpipe"] == 160
    assert not any(engine.allocated.values())
