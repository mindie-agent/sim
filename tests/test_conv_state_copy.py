"""Source caller copies retain state extent, physical ownership and readiness."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, target_graphs
from mindie_sim.prediction.conv_state_copy import IMPLEMENTATION, build
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_mechanism_contract import hardware_case

ROOT = Path(__file__).resolve().parents[1]


def source_case(batch=16):
    raw = json.loads((ROOT / "examples/cases/high-concurrency-20261030/glm-flash-ws9" /
                      f"a5_glm53flash_tp8_c{batch}_p4k.json").read_text())
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})["conv_state_copy"] = {
        "implementation": IMPLEMENTATION, "options": {"state_slots": 6}}
    return raw


@pytest.mark.parametrize("batch", (16, 32, 64))
def test_full_glm_frontend_binds_two_copies_per_kda_layer_and_preserves_inner_convolution(batch):
    raw = source_case(batch)
    old = deepcopy(raw)
    old["runtime"]["kernels"]["lowerings"].pop("conv_state_copy")
    old_ops = target_graphs(old, scope="body", phase="decode", tokens=4, context=4096,
                            sequences=batch)[0]
    ops = target_graphs(raw, scope="body", phase="decode", tokens=4, context=4096,
                        sequences=batch)[0]
    copies = [op for op in ops if op.recipe == IMPLEMENTATION]
    assert len(copies) == 68
    before = {op.op_id: op for op in old_ops if op.recipe == "short_convolution"}
    assert len(before) == 34
    by_id = {op.op_id: op for op in ops}
    for op in copies:
        assert op.local_shape == (batch, 6, 3072)
        assert dict(op.metadata)["state_strides"] == (1, 6)
        prefix = op.op_id.rsplit(".", 1)[0]
        conv = by_id[prefix + ".short_convolution"]
        assert (conv.flops, conv.vector_ops, conv.hbm_bytes) == (
            before[conv.op_id].flops, before[conv.op_id].vector_ops, before[conv.op_id].hbm_bytes)
        assert prefix + ".convolution_state_gather" in conv.deps
        assert prefix + ".convolution_output_zero" in conv.deps
        if dict(op.metadata)["direction"] == "scatter":
            assert op.deps == (conv.op_id,)
            followers = [candidate for candidate in ops if candidate.layer == op.layer
                         and candidate.family in {"recurrent_scan", "recurrent_kda"}]
            assert followers
            pending = [dep for candidate in followers for dep in candidate.deps]
            ancestors = set()
            while pending:
                dep = pending.pop()
                if dep not in ancestors and dep in by_id:
                    ancestors.add(dep)
                    pending.extend(by_id[dep].deps)
            assert op.op_id in ancestors
    assert raw == source_case(batch)


def physical_case():
    raw = hardware_case()
    raw["machine"]["device"].update({
        "rates": {"memory": 1e9, "vector": 1e9, "mte2": 1e9,
                  "mte2_aic": 1e9, "mte2_aiv": 4e6, "mte3": 8e6},
        "resource_units": {"vector": 4, "mte2_aic": 4, "mte2_aiv": 4, "mte3": 4},
        "buffers": {"ub": 4096}, "buffer_units": {"ub": 4}})
    raw["runtime"]["kernels"] = {"lowerings": {"conv_state_copy": {
        "implementation": IMPLEMENTATION, "options": {"state_slots": 3}}}}
    return parse_case(raw)


def copy_op(direction, *, active=(True, True), dim=257, slots=3):
    shape = (len(active), slots, dim)
    return OpSpec("copy", "conv_state_copy", "decode", 0, 0, (), shape, shape, shape, shape,
                  dtype="bf16", recipe=IMPLEMENTATION, source_boundary="kernel", deps=("input",),
                  metadata=(("direction", direction), ("active_requests", active),
                            ("cache_index_dtype", "int32"), ("query_start_dtype", "int64"),
                            ("invocation_class", "vector")))


@pytest.mark.parametrize("direction", ("gather", "scatter"))
@pytest.mark.parametrize("active", ((True, True), (False, True), (False, False)))
def test_copy_tail_and_inactive_masks_match_source_memory_direction(direction, active):
    parsed, op = physical_case(), copy_op(direction, active=active)
    plan = build(op, Geometry.from_devices(parsed.machine.dies))
    row = operator_costs(parsed, (op,), input_ready_ms=7)["operators"][0]
    # Four 256-element programs/request, with a three-element last block.
    reads = (2 if direction == "scatter" else sum(active)) * 771 * 2
    writes = (sum(active) if direction == "scatter" else 2) * 771 * 2
    packed_indices = 8 if direction == "gather" else 0
    assert plan.work.resources["hbm"] == reads + writes + packed_indices + 8 * 20
    assert plan.work.resources["mte2_aiv"] == reads
    assert plan.work.resources["mte3"] == writes + packed_indices
    assert not row["service_before_input"]
    assert all(abs(value) < 1e-8 for value in row["service_delta"].values())
    assert row["live_tasks"] == 0
    assert all(value == 0 for value in row["remaining_buffers"].values())


def test_one_logical_program_cannot_borrow_idle_owners_and_has_one_read_then_write():
    parsed = physical_case()
    op = copy_op("scatter", active=(True,), dim=17, slots=3)
    execution = Execution(parsed, CostModel(parsed), include_submission=False, trace_limit=1000)
    ready = execution.engine.submit("input", release=9)
    iteration = IterationWork((op,), IterationPlan((RequestView("r", 1, 0, False, "decode"),)),
                              1, 1, 1, None, "body", "copy_control")
    complete = execution.add(iteration, 0, 0, external_inputs={"input": ready}).completion
    execution.engine.advance(horizon=8)
    assert not any(execution.engine.serviced.values())
    execution.engine.advance(until=(complete,))
    engine = execution.engine
    assert complete.time - 9 == pytest.approx(20 / 1e6 + 102 / 1000 + 102 / 2000)
    stages = [row for row in engine.trace if row["demand"]]
    assert all(set(units) == {0} for row in stages for units in row["placement"].values())
    assert not engine.allocations and not engine._tasks and not engine._promises
    assert engine.peak_unit_allocated == {("ub@r0:s0", 0): 512}


def test_late_source_gather_and_writeback_order_and_payload_service():
    raw = source_case()
    ops = target_graphs(raw, scope="body", phase="decode", tokens=4,
                        context=4096, sequences=16)[0]
    selected = [op for op in ops if op.layer == 0 and op.recipe in {
        IMPLEMENTATION, "short_convolution", "convolution_output_zero"}]
    names = {op.op_id for op in selected}
    external = {dep: 11 for op in selected for dep in op.deps if dep not in names}
    report = graph_costs(raw, selected, external_inputs=external)
    times = {row["op_id"]: row for row in report["timeline"]}
    gather = times["s0.l0.convolution_state_gather"]
    conv = times["s0.l0.short_convolution"]
    scatter = times["s0.l0.convolution_state_scatter"]
    assert gather["ready_ms"] >= 11
    assert conv["ready_ms"] >= gather["complete_ms"]
    assert scatter["ready_ms"] >= conv["complete_ms"]
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
