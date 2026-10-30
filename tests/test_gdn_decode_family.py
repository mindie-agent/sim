"""Source caller invocations and resource work against independent decode oracles."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.gdn_decode_programs import CONV, GATING, update_geometry
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_operator_mechanisms import recurrent_case


def source_case(lengths=(4,) * 4, *, keys=4, heads=12, units=64, queue=False):
    parsed = recurrent_case()
    layer = replace(parsed.model.layer_plans[0], linear_key_heads=keys,
                    linear_value_heads=heads, linear_heads=heads,
                    linear_key_head_dim=128, linear_head_dim=128, conv_kernel=4)
    kernels = {"lowerings": {"gdn_decode": "vllm_ascend_gdn_decode"}}
    rates = {"hbm": 4e12, "vector": 1e13, "mte2": 4e12,
             "mte2_aic": 4e12, "mte2_aiv": 4e12, "mte3": 4e12}
    if queue:
        rates["scalar_store_issue"] = 6.4e9
        kernels["lowerings"]["gdn_decode"] = {
            "implementation": "vllm_ascend_gdn_decode",
            "options": {"gating_beta_store": {"resource": "scalar_store_issue", "queue": {
                "outstanding": 8, "granule_bytes": 128, "response_ms": 0.0001}}}}
    device = replace(parsed.machine.dies[0], rates=tuple(rates.items()), compute_units=units // 2,
                     resource_units=tuple((r, units) for r in rates))
    parsed = replace(parsed, model=replace(parsed.model, layer_plans=(layer,)),
                     runtime=replace(parsed.runtime, kernels=kernels),
                     machine=replace(parsed.machine, dies=(device,)))
    views = tuple(RequestView(str(i), q, 4096, False, "decode") for i, q in enumerate(lengths))
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode")
    return parsed, views, ops


def execute_one(parsed, views, op):
    op = replace(op, deps=("ready",))
    execution = Execution(parsed, CostModel(parsed), include_submission=False, trace_limit=10000)
    ready = execution.engine.submit("ready", release=2)
    complete = execution.add(IterationWork((op,), IterationPlan(views), 1, 0, 0, None, "body", "test"),
                             0, 0, external_inputs={"ready": ready}).completion
    execution.engine.advance(horizon=1)
    assert not any(execution.engine.serviced.values())
    execution.engine.advance(until=(complete,))
    assert not execution.engine._tasks and not execution.engine._promises
    return execution.engine, complete.time - 2


def test_decode_source_is_on_ordinary_frontend_with_separate_native_launches():
    parsed, _, ops = source_case()
    assert sum(op.recipe == CONV for op in ops) == sum(op.recipe == GATING for op in ops) == 1
    assert sum(op.family == "recurrent_vector" for op in ops) == 1
    assert sum("decode_q_l2norm" in op.recipe or "decode_k_l2norm" in op.recipe for op in ops) == 2
    copies = [op for op in ops if op.family == "tensor_copy"]
    assert len(copies) == 9
    assert all(dict(op.metadata)["invocation_class"] == "vector" for op in copies)
    assert sum(op.family == "tensor_fill" for op in ops) == 1
    assert not any(op.recipe == "qk_l2norm" for op in ops)
    ids = {op.op_id for op in ops}
    assert all(dep in ids or dep == "" for op in ops for dep in op.deps)


@pytest.mark.parametrize("batch,dim,expected", ((4, 1280, (192, 7, 28)),
                                                 (4, 2560, (512, 5, 20)),
                                                 (40, 1280, (1024, 2, 64)),
                                                 (32, 2560, (512, 5, 64))))
def test_update_dispatch_matches_canonical_choice_and_observed_blocknum(batch, dim, expected):
    assert update_geometry(batch, dim, 64) == expected


def test_speculative_convolution_keeps_history_reread_and_every_query_row():
    parsed, views, ops = source_case(lengths=(4, 3), keys=2, heads=6)
    op = next(op for op in ops if op.recipe == CONV)
    prepared, _ = RecipeBook.from_mapping(parsed.runtime.kernels).prepare(op, Geometry.from_devices(parsed.machine.dies))
    # Weight+initial history+input, then speculative history shift and reread:
    # width 4, prior history 3, two preserved positions and q current inputs.
    read = ((4 + 3 + 4 + 2 + 4) + (4 + 3 + 3 + 2 + 3)) * 1280 * 2
    write = ((4 + 2 + 4) + (3 + 2 + 3)) * 1280 * 2
    engine, _ = execute_one(parsed, views, op)
    assert engine.serviced["mte2_aiv@r0:s0"] == pytest.approx(read)
    assert engine.serviced["mte3@r0:s0"] == pytest.approx(write)
    assert engine.serviced["memory@r0:s0"] == pytest.approx(read + write)
    assert prepared.hbm_bytes == read + write
    assert engine.serviced["vector@r0:s0"] == pytest.approx(((5 + 4 * 17) + (5 + 3 * 17)) * 1280)
    assert engine.serviced["mte2@r0:s0"] == engine.serviced["mte2_aic@r0:s0"] == 0
    assert all(event["start_ms"] >= 2 for event in engine.trace if event["demand"])


@pytest.mark.parametrize("queue", (False, True))
def test_gating_masks_memory_but_not_vector_tiles_and_keeps_fp32_dma(queue):
    parsed, views, ops = source_case(queue=queue)
    op = next(op for op in ops if op.recipe == GATING)
    engine, _ = execute_one(parsed, views, op)
    assert engine.serviced["mte2_aiv@r0:s0"] == 16 * 12 * 4 + 64 * 12 * 8
    assert engine.serviced["mte3@r0:s0"] == 16 * 12 * (4 if queue else 6)
    assert engine.serviced["memory@r0:s0"] == 16 * 12 * 10 + 64 * 12 * 8
    assert engine.serviced["vector@r0:s0"] == 64 * 2 * (13 * 64 * 8 + 2 * 8)
    if queue:
        assert engine.serviced["scalar_store_issue@r0:s0"] == 16 * 12


def test_padded_gate_copies_precede_actual_slice_and_core_stays_unpadded():
    parsed, views, _ = source_case(lengths=(1,), heads=12, keys=4)
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode", tokens=16)
    copies = [op for op in ops if op.recipe in {"beta_contiguous", "alpha_contiguous"}]
    assert len(copies) == 2
    assert all(op.local_shape == (16, 12) and op.mte2_bytes == 16 * 12 * 2 for op in copies)
    gating = next(op for op in ops if op.recipe == GATING)
    assert gating.local_shape == (1, 12)
    assert next(op for op in ops if op.recipe == CONV).local_shape == (1, 2560)
    assert dict(next(op for op in ops if op.family == "recurrent_vector").metadata)["tokens"] == 1


def test_logical_gating_grid_reuses_physical_owners_without_borrowing_cores():
    parsed, views, ops = source_case(lengths=(4,), units=40)
    op = next(op for op in ops if op.recipe == GATING)
    op = replace(op, metadata=tuple({**dict(op.metadata), "logical_programs": 64}.items()))
    engine, _ = execute_one(parsed, views, op)
    assert engine.serviced["vector@r0:s0"] == 64 * 2 * (13 * 64 * 8 + 2 * 8)
    placements = [event["placement"]["vector@r0:s0"] for event in engine.trace
                  if "vector@r0:s0" in event["demand"]]
    assert all(len(p) == 1 and p[0] < 40 for p in placements)
    # Two programs, two column tiles each run on owner zero, in sequence.
    owner0 = [event for event in engine.trace
              if event["demand"].get("vector@r0:s0") and event["placement"]["vector@r0:s0"] == (0,)]
    assert len(owner0) == 4
    assert all(a["end_ms"] <= b["start_ms"] for a, b in zip(owner0, owner0[1:]))


def test_ordinary_graph_finishes_and_known_vector_copies_pay_one_invocation():
    from mindie_sim.schema.invocation import InvocationTiming
    from mindie_sim.prediction.operator_analysis import operator_costs
    parsed, views, ops = source_case()
    device = replace(parsed.machine.dies[0], buffers=(("ub", 256 * 1024),),
                     buffer_units=(("ub", 64),),
                     kernel_invocations=(("vector", InvocationTiming(minimum_ms=.003)),))
    parsed = replace(parsed, machine=replace(parsed.machine, dies=(device,)))
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    ready = execution.engine.submit("input", release=5)
    complete = execution.add(IterationWork(tuple(ops), IterationPlan(views), 1, 16, 16, None, "body", "test"),
                             0, 0, external_inputs={"": ready}).completion
    execution.engine.advance(until=(complete,))
    assert complete.done and execution.engine.serviced["host@r0:s0"] == 0
    assert not execution.engine._tasks and not execution.engine._promises
    assert not any(execution.engine.allocated.values())
    copies = [op for op in ops if op.family == "tensor_copy"]
    costs = operator_costs(parsed, copies)
    assert all(row["duration_ms"] == pytest.approx(.003) for row in costs["operators"])


def test_convolution_aiv_bandwidth_changes_service_time_not_work():
    parsed, views, ops = source_case(lengths=(4,) * 8)
    op = next(op for op in ops if op.recipe == CONV)
    durations, totals = [], []
    for scale in (.5, 1, 2):
        device = parsed.machine.dies[0]
        rates = tuple((key, (2e8 * scale if key == "mte2_aiv" else value))
                      for key, value in device.rates)
        changed = replace(parsed, machine=replace(parsed.machine, dies=(replace(device, rates=rates),)))
        engine, duration = execute_one(changed, views, op)
        durations.append(duration)
        totals.append(engine.serviced)
    assert durations[0] > durations[1] > durations[2]
    assert all(totals[0][key] == pytest.approx(totals[1][key]) == pytest.approx(totals[2][key])
               for key in totals[0])


def test_beta_issue_queue_cannot_borrow_an_idle_shared_physical_pipe():
    parsed, views, ops = source_case(lengths=(4,) * 32, heads=1, keys=1, units=4, queue=True)
    device = parsed.machine.dies[0]
    device = replace(device,
                     rates=tuple((key, 4000 if key == "scalar_store_issue" else rate)
                                 for key, rate in device.rates),
                     resource_units=tuple((key, 2 if key == "scalar_store_issue" else count)
                                          for key, count in device.resource_units))
    parsed = replace(parsed, machine=replace(parsed.machine, dies=(device,)))
    op = next(op for op in ops if op.recipe == GATING)
    meta = dict(op.metadata)
    meta["beta_store"]["queue"]["response_ms"] = 0
    op = replace(op, metadata=tuple(meta.items()))
    engine, duration = execute_one(parsed, views, op)
    # Only logical owners 0 and 1 have valid rows; both belong to issue pipe 0.
    # 128 stores / (4000/2 stores/s) = 64 ms, not 32 ms using the idle pipe 1.
    assert duration == pytest.approx(64, abs=.0001)
    events = [e for e in engine.trace if e["demand"].get("scalar_store_issue@r0:s0")]
    assert len(events) == 128
    assert all(e["placement"]["scalar_store_issue@r0:s0"] == (0,) for e in events)


def test_arch35_recurrence_is_explicit_and_does_not_expand_a3_buffer_capacity():
    from mindie_sim.prediction.gdn_recurrent import IMPLEMENTATION
    from mindie_sim.prediction.operator_analysis import operator_costs
    parsed, views, ops = source_case()
    device = replace(parsed.machine.dies[0], buffers=(("ub", 192 * 1024),),
                     buffer_units=(("ub", 64),))
    parsed = replace(parsed, machine=replace(parsed.machine, dies=(device,)))
    legacy = next(op for op in ops if op.family == "recurrent_vector")
    assert dict(legacy.metadata)["recurrence_selection"] == "analytical"
    assert operator_costs(parsed, (legacy,))["operators"][0]["live_tasks"] == 0
    binding = {"implementation": "vllm_ascend_gdn_decode",
               "options": {"recurrent_implementation": "arch35_fp32"}}
    kernels = {"lowerings": {"gdn_decode": binding}}
    selected = replace(parsed, runtime=replace(parsed.runtime, kernels=kernels))
    ops, _ = Frontend().body(selected.model, selected.runtime, views, mode="decode")
    native = next(op for op in ops if op.family == "recurrent_vector")
    assert native.recipe == IMPLEMENTATION
    assert dict(native.metadata)["recurrence_selection"] == "arch35_fp32"
    with pytest.raises(ValueError, match="allocation_exceeds_capacity"):
        operator_costs(selected, (native,))
    selected = replace(selected, machine=replace(selected.machine, dies=(
        replace(device, buffers=(("ub", 256 * 1024),)),)))
    result = operator_costs(selected, (native,))["operators"][0]
    assert result["live_tasks"] == 0 and not any(result["remaining_buffers"].values())
