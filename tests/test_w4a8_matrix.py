"""Independent source rectangles and shared detailed/steady physical oracles."""
import math

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.w4a8_matrix import matrix_program, owner_rectangles
from mindie_sim.prediction.w4a8_schedule import down_schedule, gate_schedule, total_work


def _schedule(rows=(1, 0, 3), n=320, k=640, cores=1, down=False):
    factory = down_schedule if down else gate_schedule
    return factory(sum(rows), n, k, rows, aic_owners=cores, aiv_owners=2 * cores,
                   fixpipe=True)


def _run(schedule, *, compact=True, modes=False, operand_rate=128):
    cores = schedule.aic_owners
    op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(),
                 m=schedule.m, n=schedule.n, k=schedule.k, quant=QuantSpec("bf16"),
                 category="dense_mlp", phase="decode")
    rates = {"cube": cores, "hbm": 1e12, "mte2_aic": 512 * cores,
             "mte2_aiv": 512 * cores, "mte1": operand_rate * cores,
             "mte3": 1e9, "fixpipe": 1e9, "vector": 1e9}
    units = {key: 1 if key == "hbm" else (2 * cores if key in
             {"vector", "mte2_aiv", "mte3"} else cores) for key in rates}
    pools = {"l1": 512 * 1024, "l0a": 64 * 1024,
             "l0b": 64 * 1024, "l0c": 128 * 1024}
    modes = ({"mte1_l0a": ("mte1", 2000 * rates["mte1"]),
              "mte1_l0b": ("mte1", 1000 * rates["mte1"])} if modes else {})
    context = RecipeContext(
        op, {}, {key: key for key in rates}, rates, cores,
        pool_names={key: key for key in pools}, pool_capacities=pools,
        pool_units={key: cores for key in pools}, resource_units=units,
        compute_peaks={"int4": 1e9 * cores}, service_modes=modes)
    engine = Engine([Resource(key, value, units=units[key]) for key, value in rates.items()],
                    pools=pools, pool_units={key: cores for key in pools}, trace_limit=300)
    done = matrix_program(context, schedule, compact=compact).start(
        engine, "matrix", engine.submit("ready", release=7))
    engine.advance(horizon=7)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert not any(engine.allocated.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert all(value <= pools[pool] for (pool, _), value in engine.peak_unit_allocated.items())
    return done.time - 7, engine


@pytest.mark.parametrize("rows,n,cores,down", [
    ((1, 0, 3), 320, 1, False), ((1,) * 8 + (0,) * 248, 512, 20, False),
    ((3, 0, 129, 2), 320, 4, False), ((1,) * 8 + (0,) * 248, 6144, 20, True),
])
def test_source_rectangles_preserve_expert_boundaries_and_rolling_owners(rows, n, cores, down):
    schedule = _schedule(rows, n, cores=cores, down=down)
    expected = [[] for _ in range(cores)]
    ordinal = 0
    for rows in rows:
        for m in range(0, rows * 2, schedule.base_m):
            for column in range(0, n, schedule.base_n):
                expected[ordinal % cores].append((min(schedule.base_m, rows * 2 - m),
                                                   min(schedule.base_n, n - column)))
                ordinal += 1
    assert [list(owner_rectangles(schedule, owner)) for owner in range(cores)] == expected


@pytest.mark.parametrize("down,modes,k", [(False, False, 128), (False, True, 640),
                                        (True, True, 512), (True, False, 768)])
def test_steady_and_detailed_match_physical_work_and_serial_owner_boundary(down, modes, k):
    schedule = _schedule(k=k, down=down)
    detailed_ms, detailed = _run(schedule, compact=False, modes=modes)
    compact_ms, compact = _run(schedule, modes=modes)
    assert compact.serviced == pytest.approx(detailed.serviced)
    assert compact_ms == pytest.approx(detailed_ms, abs=1e-7)
    matrix = total_work(schedule.matrix)
    assert compact.serviced["hbm"] == pytest.approx(matrix.resources["hbm"])
    assert compact.serviced["mte2_aic"] == pytest.approx(matrix.resources["mte2"])
    assert compact.serviced["vector"] == pytest.approx(matrix.resources["vector"])
    assert compact.serviced["fixpipe"] == pytest.approx(4 * schedule.m * schedule.n)
    assert compact.serviced["mte3"] == 0
    # Independent inner-grid bytes, with A's explicit directional instruction
    # twice as fast as B on the same MTE1 resource in this synthetic control.
    a = b = 0
    for owner in range(schedule.aic_owners):
        for rows, cols in owner_rectangles(schedule, owner):
            a += 0.5 * rows * k * math.ceil(cols / 128)
            b += 0.5 * cols * k * math.ceil(rows / min(schedule.base_m, 128))
    assert compact.serviced["mte1"] == pytest.approx((a / 2 if modes else a) + b)


def test_owner_tiles_remain_independent_bank_loads_and_k_compression_is_bounded():
    times = []
    for rate in (64, 128, 256):
        elapsed, engine = _run(_schedule(cores=4), operand_rate=rate)
        times.append(elapsed)
        assert engine.serviced["hbm"] > 0
    assert times[0] > times[1] > times[2]
    _, small = _run(_schedule(k=512, cores=4))
    _, large = _run(_schedule(k=256 * 1_000_000, cores=4))
    assert large.completed_count == small.completed_count
    assert large.peak_tasks == small.peak_tasks
    assert large.serviced["fixpipe"] == small.serviced["fixpipe"]
    assert large.serviced["vector"] == small.serviced["vector"]


def test_group_count_bulk_copy_includes_idle_aivs_without_256_active_weight_banks():
    schedule = _schedule((1,) * 8 + (0,) * 248, n=6144, k=256, cores=20, down=True)
    pre = total_work(schedule.pre)
    assert pre.resources["mte2"] == 8 * 256 + 40 * 256 * 8
    assert pre.resources["vector"] == 6.5 * 8 * 256 + 40 * 2 * 256
    _, engine = _run(schedule, modes=True)
    assert engine.serviced["mte2_aic"] == pytest.approx(
        8 * 0.5 * 6144 * 256 + 96 * 8 * 512 + 96 * 256)
    assert engine.serviced["mte1"] > 0
