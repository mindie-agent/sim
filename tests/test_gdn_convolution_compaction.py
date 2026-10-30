"""Independent serial-task and physical-service controls for convolution runs."""
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.gdn_decode_programs import conv_plan, conv_schedule, conv_task_runs, update_geometry
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.schema.operators import OpSpec


def run(lengths, dim, units, *, compact, width=4, bias=False, speculative=True,
        rates=None, contender=0):
    shape = (sum(lengths), dim)
    op = OpSpec("conv", "vector", "decode", 0, 0, (), shape, shape, shape, shape,
                dtype="bf16", metadata=(("sequence_tokens", lengths), ("kernel_width", width),
                    ("bias", bias), ("speculative", speculative),
                    ("state_history_slots", 2 + max(lengths))))
    values = {"hbm": 1e8, "mte2_aiv": 1e8, "mte2_aic": 1e8,
              "vector": 1e8, "mte3": 1e8, **(rates or {})}
    counts = {role: units if role != "hbm" else 1 for role in values}
    geometry = Geometry(max(1, units // 2), counts, {"ub": 256 * 1024}, {"ub": units})
    plan = conv_plan(op, geometry, compact=compact)
    engine = Engine([Resource(role, rate, units=counts[role]) for role, rate in values.items()],
                    pools={"ub": 256 * 1024}, pool_units={"ub": units}, trace_limit=20000)
    context = RecipeContext(plan.work.apply(op), {}, {role: role for role in values}, values,
                            geometry.compute_units, resource_units=counts,
                            pool_names={"ub": "ub"}, pool_units={"ub": units})
    ready = engine.submit("input", release=3)
    complete = plan.lower(context).start(engine, "conv", ready)
    other = engine.submit("external", {"hbm": contender}, after=(ready,))
    engine.advance(horizon=2)
    assert not any(engine.serviced.values())
    engine.advance(until=(complete,))
    engine.advance(until=(other,))
    assert not engine._tasks and not engine._promises and not engine.allocations
    assert not any(engine.allocated.values())
    assert all(e["start_ms"] >= 3 for e in engine.trace if e["demand"])
    assert engine.peak_allocated["ub"] <= 188416 * units
    assert engine.serviced["mte2_aic"] == 0
    record = conv_schedule(lengths, dim, units, width=width, bias=bias,
                           speculative=speculative, compact=compact)
    return plan, engine, complete.time - 3, record


@pytest.mark.parametrize("lengths,dim,units", (
    ((4,) * 128, 2560, 64), ((4,) * 40, 1280, 64),
    ((4, 3, 2) * 10, 1297, 4), ((4, 4, 3, 3, 2, 2), 513, 8),
))
def test_runs_preserve_source_round_robin_owners_and_tail_lengths(lengths, dim, units):
    tile, columns, blocks = update_geometry(len(lengths), dim, units)
    for owner in range(blocks):
        expected = [(lengths[index // columns], min(tile, dim - (index % columns) * tile))
                    for index in range(owner, len(lengths) * columns, blocks)]
        actual = [(q, n) for q, n, repeat in conv_task_runs(
            lengths, dim, tile, columns, blocks, owner) for _ in range(repeat)]
        assert actual == expected


@pytest.mark.parametrize("width,speculative,bias", ((2, False, False), (3, False, True),
                                                       (4, False, False), (4, True, True)))
@pytest.mark.parametrize("limited", ("mte2_aiv", "mte3", "vector"))
def test_one_owner_serial_repetition_preserves_duration_and_service(width, speculative, bias, limited):
    args = ((4,) * 12, 512, 1)
    detailed = run(*args, compact=False, width=width, speculative=speculative,
                   bias=bias, rates={limited: 1e6})
    compact = run(*args, compact=True, width=width, speculative=speculative,
                  bias=bias, rates={limited: 1e6})
    assert compact[2] == pytest.approx(detailed[2], rel=1e-10)
    assert compact[1].serviced == pytest.approx(detailed[1].serviced, rel=1e-11)
    assert compact[1].completed_count < detailed[1].completed_count / 5


@pytest.mark.parametrize("lengths,dim,units,contender", (
    ((4,) * 128, 2560, 64, 0), ((4,) * 40, 1280, 64, 0),
    ((4, 4, 3, 3, 3) * 4, 513, 4, 1e6), ((4,) * 17, 512, 4, 1e6),
))
def test_shared_bandwidth_conserves_every_addressed_payload_under_compaction(lengths, dim, units, contender):
    detailed = run(lengths, dim, units, compact=False, contender=contender, rates={"hbm": 1e6})
    compact = run(lengths, dim, units, compact=True, contender=contender, rates={"hbm": 1e6})
    for plan, engine, duration, record in (detailed, compact):
        reads = sum(9 + 2 * q for q in lengths) * dim * 2
        writes = sum(2 + 2 * q for q in lengths) * dim * 2
        assert engine.serviced["hbm"] == pytest.approx(reads + writes + contender)
        assert engine.serviced["mte2_aiv"] == pytest.approx(reads)
        assert engine.serviced["mte3"] == pytest.approx(writes)
        assert engine.serviced["vector"] == pytest.approx(sum(5 + 17 * q for q in lengths) * dim)
        assert duration >= (reads + writes) / 1e6 - 1e-10
        assert plan.work.resources["hbm"] == reads + writes
    assert compact[1].serviced == pytest.approx(detailed[1].serviced, rel=1e-11)
    # This symmetric case has the same repetition count on every owner: the
    # complete resource schedule scales uniformly, so duration is exact too.
    if len(lengths) == 128:
        assert compact[2] == pytest.approx(detailed[2], rel=1e-10)
        assert compact[3]["logical_tasks"] == 640
        assert compact[3]["resource_cycles"] == 64
        assert compact[3]["read_descriptors"] == 10880
        assert compact[3]["write_descriptors"] == 6400


def test_aiv_bandwidth_scan_retains_physical_work_and_one_fixed_owner_buffer():
    controls = [run((4,) * 32, 2560, 4, compact=True, rates={"mte2_aiv": 1e5 * scale})
                for scale in (.5, 1, 2)]
    assert controls[0][2] > controls[1][2] > controls[2][2]
    for _, engine, _, _ in controls:
        assert engine.peak_allocated["ub"] == 188416
        assert engine.peak_unit_allocated == {("ub", owner): 188416 for owner in range(4)}
        assert engine.serviced == pytest.approx(controls[0][1].serviced, rel=1e-11)
