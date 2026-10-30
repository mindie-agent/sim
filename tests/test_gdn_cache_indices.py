"""Request-index materialization is independent of query rows and graph padding."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.gdn_decode_programs import CONV, GATING
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan
from test_gdn_decode_family import execute_one, source_case


def selected(lengths, stride, *, padded_tokens=None):
    parsed, views, _ = source_case(lengths=lengths)
    kernels = {"lowerings": {"gdn_decode": {
        "implementation": "vllm_ascend_gdn_decode",
        "options": {"cache_index_row_stride": stride}}}}
    parsed = replace(parsed, runtime=replace(parsed.runtime, kernels=kernels))
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode", tokens=padded_tokens)
    return parsed, views, ops


@pytest.mark.parametrize("lengths,padding", (((4,) * 4, 32), ((1,) * 4, 16), ((4, 2, 3, 4), 32)))
def test_first_column_reads_only_one_int32_per_request(lengths, padding):
    parsed, views, ops = selected(lengths, 4, padded_tokens=padding)
    copy = next(op for op in ops if op.recipe == "state_indices_contiguous")
    assert copy.local_shape == (4, 1)
    assert copy.mte2_bytes == copy.mte3_bytes == 16
    meta = dict(copy.metadata)
    assert meta["source_layout"]["strides"] == (4, 1)
    assert meta["source_output"] == "causal_cache_indices"
    assert meta["input_format"] == meta["output_format"] == "int32"
    conv = next(op for op in ops if op.recipe == CONV)
    assert copy.op_id in conv.deps
    assert dict(conv.metadata)["cache_index_row_stride"] == 4
    assert all(copy.op_id not in op.deps for op in ops if op.op_id != conv.op_id)
    engine, _ = execute_one(parsed, views, copy)
    assert engine.serviced["memory@r0:s0"] == 32
    assert engine.serviced["mte2@r0:s0"] + engine.serviced["mte2_aiv@r0:s0"] == 16
    assert engine.serviced["mte3@r0:s0"] == 16


@pytest.mark.parametrize("lengths,stride", (((4,) * 4, 1), ((4,), 4), ((1,), 4)))
def test_contiguous_or_singleton_index_view_needs_no_materialization(lengths, stride):
    _, _, ops = selected(lengths, stride)
    assert not any(op.recipe == "state_indices_contiguous" for op in ops)


def test_index_materialization_waits_for_global_input_and_only_gates_convolution():
    parsed, views, ops = selected((4,) * 4, 4)
    copy = next(op for op in ops if op.recipe == "state_indices_contiguous")
    conv = next(op for op in ops if op.recipe == CONV)
    gating = next(op for op in ops if op.recipe == GATING)
    execution = Execution(parsed, CostModel(parsed), include_submission=False, trace_limit=100000)
    ready = execution.engine.submit("input", release=5)
    completion = execution.add(IterationWork(tuple(ops), IterationPlan(views), 1, 16, 16, None, "body", "test"),
                               0, 0, external_inputs={"": ready}).completion
    execution.engine.advance(horizon=4)
    assert not any(execution.engine.serviced.values())
    execution.engine.advance(until=(completion,))
    assert completion.done and copy.op_id in conv.deps and copy.op_id not in gating.deps
    assert not execution.engine._tasks and not execution.engine._promises
    assert not any(execution.engine.allocated.values())
