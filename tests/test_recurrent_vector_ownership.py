"""Physical recurrence traffic, fixed copy ownership and late-input contracts."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def run_scan(*, topology="split", sequences=(1, 1, 2), aiv_units=4,
             copy_units=2, read_scale=1, aic_scale=1, ready_ms=0):
    parsed = parse_case(case())
    layer = replace(parsed.model.layer_plans[0], attention="gdn", linear_key_heads=1,
                    linear_value_heads=1, linear_heads=1, linear_key_head_dim=3,
                    linear_head_dim=2, conv_kernel=3, recurrent_state_dtype="fp32")
    rates = {"hbm": 1e15, "vector": 1e15}
    if topology != "unified":
        rates.update(mte2=2e6 * read_scale, mte3=1e15)
    if topology in {"split", "split_only"}:
        rates.update(mte2=1e15, mte2_aiv=2e6 * read_scale, mte2_aic=2e6 * aic_scale)
        if topology == "split_only":
            del rates["mte2"]
    device = replace(parsed.machine.dies[0], rates=tuple(rates.items()),
                     compute_units=aiv_units, resource_units=tuple(
                         (role, copy_units if role.startswith("mte") else aiv_units)
                         for role in rates), buffers=(("ub", 1024),),
                     buffer_units=(("ub", aiv_units),))
    parsed = replace(parsed, model=replace(parsed.model, layers=1, layer_plans=(layer,)),
                     machine=replace(parsed.machine, dies=(device,)),
                     runtime=replace(parsed.runtime, pp_layer_partition=(1,)))
    views = tuple(RequestView(str(i), q, 16, False, "decode")
                  for i, q in enumerate(sequences))
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode")
    scan = replace(next(op for op in ops if op.family == "recurrent_vector"), deps=("input",))
    execution = Execution(parsed, CostModel(parsed), include_submission=False, trace_limit=2000)
    input_ready = execution.engine.submit("input", release=ready_ms)
    scheduled = execution.add(
        IterationWork((scan,), IterationPlan(views), 1, sum(sequences), sum(sequences),
                      None, "body", "test"), 0, 0, external_inputs={"input": input_ready})
    if ready_ms:
        execution.engine.advance(horizon=ready_ms / 2)
        assert not any(execution.engine.serviced.values())
    execution.engine.advance(until=(scheduled.completion,))
    return scan, execution.engine, scheduled.completion.time - ready_ms


@pytest.mark.parametrize("topology", ("split", "split_only"))
def test_split_aiv_binding_retains_input_state_and_checkpoint_bytes(topology):
    scan, engine, _ = run_scan(topology=topology, ready_ms=9)
    # Three states are loaded once, four token/head updates checkpoint state
    # and output. Q/K/V plus decay/beta are consumed at every token.
    assert scan.mte2_bytes == 3 * 3 * 2 * 4 + 4 * ((2 * 3 + 2) * 2 + 6)
    assert scan.mte3_bytes == 4 * (3 * 2 * 4 + 2 * 2)
    assert engine.serviced["memory@r0:s0"] == scan.mte2_bytes + scan.mte3_bytes
    assert engine.serviced["mte2_aiv@r0:s0"] == scan.mte2_bytes
    assert engine.serviced["mte3@r0:s0"] == scan.mte3_bytes
    assert engine.serviced.get("mte2@r0:s0", 0) == engine.serviced["mte2_aic@r0:s0"] == 0
    assert engine.serviced["vector@r0:s0"] == 4 * 7 * 3 * 2
    assert all(event["start_ms"] >= 9 for event in engine.trace if event["demand"])
    assert not engine.allocations and not any(engine.allocated.values())
    assert not engine._tasks and not engine._promises


@pytest.mark.parametrize("aiv_units,copy_units", ((4, 2), (8, 4), (4, 4), (4, 8)))
def test_ragged_active_aiv_work_stays_on_its_physical_copy_units(aiv_units, copy_units):
    scan, engine, duration = run_scan(aiv_units=aiv_units, copy_units=copy_units)
    reads, writes = {}, {}
    for event in engine.trace:
        for role, destination in (("mte2_aiv", reads), ("mte3", writes)):
            name = role + "@r0:s0"
            amount = event["demand"].get(name, 0)
            if amount:
                owners = tuple(event["placement"][name])
                for owner in owners:
                    destination[owner] = destination.get(owner, 0) + amount / len(owners)
    expected_reads, expected_writes = {}, {}
    per_owner_reads, per_owner_writes = (46, 46, 68), (28, 28, 56)
    for owner in range(3):
        if copy_units >= aiv_units:
            width = copy_units // aiv_units
            pipes = range(owner * width, (owner + 1) * width)
        else:
            pipes = (owner // (aiv_units // copy_units),)
        for pipe in pipes:
            expected_reads[pipe] = expected_reads.get(pipe, 0) + per_owner_reads[owner] / len(pipes)
            expected_writes[pipe] = expected_writes.get(pipe, 0) + per_owner_writes[owner] / len(pipes)
    assert reads == expected_reads and writes == expected_writes
    assert sum(reads.values()) == scan.mte2_bytes
    # Four owners share two pipes: the first token reads 92 B on pipe 0 and
    # 46 B on pipe 1. The second reads only 22 B on pipe 1; idle cores cannot help.
    if (aiv_units, copy_units) == (4, 2):
        assert duration == pytest.approx((92 + 22) / 1000, abs=1e-8)


def test_aiv_bandwidth_sweep_changes_only_its_real_read_service():
    results = [run_scan(read_scale=scale) for scale in (.5, 1, 2)]
    durations = [duration for _, _, duration in results]
    assert durations[0] > durations[1] > durations[2]
    assert durations == pytest.approx((.228, .114, .057), abs=1e-8)
    assert results[0][1].serviced == results[1][1].serviced == results[2][1].serviced
    # AIC is not an owner of a vector-only recurrence.
    assert run_scan(aic_scale=.01)[2] == durations[1] == run_scan(aic_scale=100)[2]


def test_generic_pipe_and_unified_gpu_keep_their_declared_memory_semantics():
    generic = [run_scan(topology="generic", read_scale=scale) for scale in (.5, 1, 2)]
    assert [duration for _, _, duration in generic] == pytest.approx((.228, .114, .057), abs=1e-8)
    for scan, engine, _ in generic:
        assert engine.serviced["mte2@r0:s0"] == scan.mte2_bytes
        assert engine.serviced["mte3@r0:s0"] == scan.mte3_bytes
        assert engine.serviced["memory@r0:s0"] == scan.hbm_bytes
    scan, engine, duration = run_scan(topology="unified", ready_ms=1)
    assert engine.serviced["memory@r0:s0"] == scan.hbm_bytes
    assert engine.serviced["vector@r0:s0"] == scan.vector_ops
    assert duration > 0
    assert not any(name.startswith("mte") for name in engine.serviced)


def test_multiple_ragged_waves_release_state_and_preserve_every_checkpoint():
    scan, engine, _ = run_scan(sequences=(1, 4, 2, 3, 1))
    assert engine.serviced["mte2_aiv@r0:s0"] == 5 * 24 + 11 * 22 == scan.mte2_bytes
    assert engine.serviced["mte3@r0:s0"] == 11 * 28 == scan.mte3_bytes
    assert engine.serviced["vector@r0:s0"] == 11 * 42 == scan.vector_ops
    assert not engine.allocations and not any(engine.unit_allocated.values())
    assert not engine._tasks and not engine._promises
