"""Compact owner runs retain source work and finite nominal dependencies."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.operators import kda_prepare, kda_prepare_compact
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from test_kda_prepare import execute, hardware_case, params


def local_case(cores=2, *, memory=1e30):
    case = hardware_case(units=cores, split_copy=True)
    case["machine"]["device"]["rates"].update({
        "memory": memory, "mte2_aic": 1e12, "mte2_aiv": 1e12,
        "mte1": 4e12, "mte3": 1e12, "vector": 3e12,
        "reg_special_lanes": 8e11, "fixpipe": 2e12,
    })
    case["machine"]["device"]["resource_units"]["mte3"] = 2 * cores
    return case


@pytest.mark.parametrize("lengths,heads,cores", [
    ((64,), 2, 2), ((128,), 6, 2), ((64, 128, 64), 2, 2), ((128,), 66, 32),
])
def test_compact_work_and_noncontended_owner_schedule_match_expanded(lengths, heads, cores):
    case = local_case(cores)
    expanded = execute(params(lengths, heads, compact=False), case)
    compact = execute(params(lengths, heads, compact=True), case)
    assert compact.now == pytest.approx(expanded.now, rel=1e-10)
    assert compact.serviced == pytest.approx(expanded.serviced, rel=1e-10, abs=1e-12)
    assert compact.summary()["live_tasks"] == 0
    assert compact.summary()["live_programs"] == 0
    assert compact.completed_count < expanded.completed_count / 20


def test_single_gate_owner_nominal_includes_shared_hbm_arbitration():
    # Exact nominal is computed by the real Engine; independent stage maxima
    # plus a bulk-byte bound would miss load/store competition here.
    case = parse_case(local_case(1, memory=1e8))
    op = kda_prepare.make_op(params((128,), 2))
    execution = Execution(case, CostModel(case))
    context = execution._context(op, 0, "none")
    geometry = kda_prepare.parse_params(params((128,), 2))
    program = kda_prepare._gate_pipeline(0, geometry, context, 2)
    nominal = kda_prepare_compact._nominal(("shared-hbm-control",), program, context)
    ready = execution.engine.submit("start")
    done = program.start(execution.engine, "direct", ready)
    execution.engine.advance(until=(done,))
    assert nominal == pytest.approx(execution.engine.now, rel=1e-12)
    memory_bound = context.bind(kda_prepare.gate_work(
        replace(geometry, heads=1, key_heads=1)))["memory@r0:s0"] / 1e5
    assert nominal >= memory_bound * (1 - 1e-12)


def test_gate_barrier_ready_and_empty_owners_survive_compaction():
    op = kda_prepare.make_op(params((64, 128), 2, compact=True))
    report = verify_operator(local_case(8), op, kda_prepare.build, input_ready_ms=7)
    assert report["status"] == "passed"
    engine = execute(params((64, 128), 2, compact=True),
                     local_case(8), trace_limit=300)
    gate = [row for row in engine.trace if ":gate:program:group" in row["task"]]
    prepare = [row for row in engine.trace if ":prepare:program:group" in row["task"]]
    assert len({row["task"].split(":gate:program:")[1].split(":")[0]
                for row in gate}) == 4
    assert len({row["task"].split(":prepare:program:")[1].split(":")[0]
                for row in prepare}) == 3
    assert min(row["start_ms"] for row in prepare) >= max(row["end_ms"] for row in gate)
    assert engine.summary()["live_tasks"] == 0


def test_contention_is_resource_coupled_and_long_geometry_keeps_active_state():
    short = execute(params((128,), 2, compact=True), local_case(2, memory=1e9))
    long = execute(params((8192,), 2, compact=True), local_case(2, memory=1e9))
    fast = execute(params((8192,), 2, compact=True), local_case(2))
    assert long.now > fast.now
    assert long.now >= long.serviced["memory@r0:s0"] / 1e6 * (1 - 1e-12)
    assert short.summary()["peak_live_tasks"] == long.summary()["peak_live_tasks"]
    assert short.completed_count == long.completed_count
    assert long.summary()["live_tasks"] == long.summary()["live_programs"] == 0


def test_nominal_cache_does_not_reuse_a_different_special_rate():
    value = params((64,), 2, compact=True)
    fast = execute(value, local_case())
    slower = local_case()
    slower["machine"]["device"]["rates"]["reg_special_lanes"] /= 10
    warm = execute(value, slower)
    kda_prepare_compact._NOMINAL.clear()
    cold = execute(value, slower)
    assert warm.now == pytest.approx(cold.now, rel=1e-12)
    assert warm.now > fast.now
