"""Address hazards, queue backpressure, physical traffic and nested lifetimes."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.catalog.ops import vector_op
from mindie_sim.prediction.pointwise import TiledPointwise
from mindie_sim.prediction.program import KernelProgram, RecipeContext, Stage
from mindie_sim.prediction.transactions import QueuedStore


def store(**changes):
    base = QueuedStore(
        rows=2, cols=2, row_stride_bytes=4, element_bytes=2, offset_bytes=0, participants=1,
        issue_resource="issue", issue_rate=1000, memory_resource="memory",
        granule_bytes=8, outstanding=2, response_ms=.01)
    return replace(base, **changes)


def run(program, *, bandwidth=1e20, background=0):
    engine = Engine([Resource("issue", 1000), Resource("memory", bandwidth)], trace_limit=100)
    if background:
        engine.submit("other_memory_consumer", {"memory": background})
    done = program.start(engine, "store", engine.submit("ready"))
    engine.advance(until=(done,))
    return done.time, engine


def test_same_address_block_serializes_responses_not_all_independent_transfers():
    same, engine = run(store())
    separate, _ = run(store(row_stride_bytes=8))
    serial, _ = run(store(row_stride_bytes=8, outstanding=1))
    # With two slots, issues at 1,2,12,22 us and one shared granule return at
    # 11,21,31,41 us. Two independent granules permit responses at 11,21,22,32.
    assert same == pytest.approx(.041)
    assert separate == pytest.approx(.032)
    assert serial == pytest.approx(.044)
    assert engine.serviced == {"issue": 4, "memory": 8}
    assert engine.summary()["live_tasks"] == 0


def test_memory_contention_keeps_slots_occupied_until_real_completion():
    uncontended, _ = run(store(row_stride_bytes=8), bandwidth=100)
    contended, engine = run(store(row_stride_bytes=8), bandwidth=100, background=100)
    assert contended > uncontended
    transfers = [row for row in engine.trace if row["task"].endswith(":response")]
    assert sum(row["serviced"]["memory"] for row in transfers) == 8
    # A fixed timer must not release the slot while the payload is still queued.
    first = next(row for row in engine.trace if row["task"].endswith("request0:response"))
    third = next(row for row in engine.trace if row["task"].endswith("request2:issue"))
    assert third["start_ms"] >= first["end_ms"]


def test_queue_working_set_is_bounded_and_nested_stage_releases_after_its_child():
    allocation = Allocation("output", "ub", 16)
    program = KernelProgram((
        Stage("write", acquire=(allocation,), free=(allocation,),
              program=store(rows=1024, row_stride_bytes=128)),
    ))
    engine = Engine([Resource("issue", 1000), Resource("memory", 1e20)],
                    pools={"ub": 16}, trace_limit=5)
    done = program.start(engine, "kernel", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.serviced == {"issue": 2048, "memory": 4096}
    assert engine.peak_tasks < 20
    assert engine.peak_allocated["ub"] == 16 and engine.allocated["ub"] == 0
    writes = [row for row in engine.trace if row["task"].endswith(":response")]
    freed = next(row for row in engine.trace if row["task"].endswith(":free"))
    assert freed["start_ms"] >= max(row["end_ms"] for row in writes)
    assert not engine._tasks and not engine._promises


def test_distinct_unit_address_phases_do_not_create_per_tile_global_barriers():
    rates = {"hbm": 1e20, "vector": 1e20, "issue": 3}
    engine = Engine([Resource(key, value) for key, value in rates.items()])
    op = vector_op(op_id="store", deps=(), stage=0, layer=0, requests=(), elements=81,
                   ops_per_element=0, dtype="bf16", family="store", phase="primitive",
                   recipe="tiled", shape=(9, 9))
    context = RecipeContext(op, {}, {key: key for key in rates}, rates, 3)
    recipe = TiledPointwise(0, ({
        "bytes_per_element": 2, "resource": "issue",
        "queue": {"granule_bytes": 16, "outstanding": 2, "response_ms": 5, "row_control_ms": 1},
    },), tile_rows=3, tile_cols=3)
    done = recipe.lower(context).start(engine, "pointwise", engine.submit("ready"))
    engine.advance(until=(done,))
    # Address-enumerated tile spans for units 0/1/2 are [38,38,35], [38,35,38],
    # [35,38,34]. Independent workers take 111; a barrier at each tile gives 114.
    assert done.time == pytest.approx(111)
    assert engine.serviced["issue"] == 81 and engine.serviced["hbm"] == 162
