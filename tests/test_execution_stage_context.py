"""Pending cohort work shares hardware inputs without sharing device service."""
from dataclasses import replace

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.structure import parse_case
from test_mechanism_contract import hardware_case
from test_resource_services import copy_op, iteration


def test_pending_context_storage_scales_by_stage_while_each_replica_keeps_its_rate():
    raw = hardware_case()
    raw["machine"]["num_devices"] = 2
    raw["runtime"]["dp"] = 2
    parsed = parse_case(raw)
    second = parsed.machine.dies[1]
    second = replace(second, rates=tuple({**dict(second.rates), "hbm": 2e6}.items()))
    parsed = replace(parsed, machine=replace(parsed.machine, dies=(parsed.machine.dies[0], second)))
    contexts = {0: [], 1: []}

    class ObservedExecution(Execution):
        def _context(self, op, replica, scope, **options):
            context = super()._context(op, replica, scope, **options)
            contexts[replica].append(context)
            return context

    execution = ObservedExecution(parsed, CostModel(parsed), include_submission=False)
    ops = tuple(copy_op(2000, op_id=str(i)) for i in range(128))
    steps = execution.add_group({replica: iteration(ops) for replica in contexts}, 7)
    shared_fields = ("resource_names", "resource_units", "work_units", "compute_peaks",
                     "pool_names", "pool_units", "pool_capacities", "service_modes", "store_queues")
    for replica, records in contexts.items():
        for name in shared_fields:
            assert len({id(getattr(record, name)) for record in records}) == 1
        assert len({id(record.native_work) for record in records}) == len(ops)
        assert records[0].resource_names["hbm"] == f"memory@r{replica}:s0"
    execution.engine.advance(until=tuple(step.completion for step in steps.values()), first=False)
    assert steps[0].completion.time == 7 + 128 * 2
    assert steps[1].completion.time == 7 + 128
    assert execution.engine.serviced["memory@r0:s0"] == 256000
    assert execution.engine.serviced["memory@r1:s0"] == 256000
    assert not execution.engine.summary()["live_tasks"]


def test_operator_precision_alias_extends_only_its_local_binding():
    raw = hardware_case()
    raw["machine"]["device"]["compute_peaks"]["fp16"] = 4e6
    parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    op = copy_op(0, flops=2000, matrix_flops=(("float16", 2000),), dtype="float16")
    context = execution._context(op, 0, "none")
    assert context.compute_peaks["float16"] == 4e6
    assert "float16" not in execution.stage_contexts[0, 0].compute_peaks
    step = execution.add(iteration((op,)), 0, 3)
    execution.engine.advance(until=(step.completion,))
    assert step.completion.time == pytest.approx(3.5)
    assert execution.engine.serviced["matrix@r0:s0"] == .5
