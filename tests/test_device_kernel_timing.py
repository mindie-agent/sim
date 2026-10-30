"""Hardware completion constraints preserve causal boundaries and real work."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs, predict, target_operators
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import IterationWork
from test_graph_analysis import op
from test_mechanism_contract import hardware_case
from test_mla_family_selection import mla_case


def timed_case():
    raw = hardware_case()
    raw["machine"]["device"]["kernel_invocations"] = {
        "vector": {"minimum_ms": .0025, "provenance": {"basis": "causal_control"}},
        "cube": {"minimum_ms": .005, "provenance": {"basis": "causal_control"}},
    }
    return raw


def vector(name="v", *, amount=1, **kwargs):
    return replace(op(name, **kwargs), vector_ops=amount)


def test_one_completion_constraint_across_operator_graph_and_serving():
    raw = timed_case()
    raw["workload"]["requests"][0]["arrival_ms"] = 7
    operation = vector()
    lower = lambda plan, *_: IterationWork((operation,), plan, 1, 1, 1, None,
                                          "prefill", "device_floor")
    isolated = operator_costs(raw, (operation,), input_ready_ms=7)["operators"][0]
    graph = graph_costs(raw, (operation,), input_ready_ms=7)
    serving = predict(raw, lower=lower)
    assert isolated["duration_ms"] == pytest.approx(.0025)
    assert graph["complete_ms"] == serving["requests"][0]["finish_ms"] == 7.0025
    assert isolated["serviced"] == {"vector@r0:s0": 1}
    assert isolated["service_before_input"] == {}
    assert graph["execution"]["live_tasks"] == serving["execution"]["live_programs"] == 0
    large = operator_costs(raw, (vector(amount=100),))["operators"][0]
    assert large["duration_ms"] == pytest.approx(.1)


def test_independent_kernels_overlap_and_fused_boundary_pays_once():
    raw = timed_case()
    split = (vector("a"), vector("b", deps=("a",)))
    independent = (split[0], vector("b", stream="second"))
    fused = (vector("ab", amount=2),)
    serial = graph_costs(raw, split)
    concurrent = graph_costs(raw, independent)
    combined = graph_costs(raw, fused)
    assert serial["duration_ms"] == pytest.approx(.005)
    assert concurrent["duration_ms"] == combined["duration_ms"] == pytest.approx(.0025)
    assert serial["execution"]["work"] == combined["execution"]["work"]
    mixed = graph_costs(raw, (replace(vector(), flops=1),))
    assert mixed["duration_ms"] == pytest.approx(.005)  # max, never 2.5+5 us
    assert mixed["parameters"]["invocation_models"][0]["invocations"] == 1


def test_host_submission_and_runtime_override_do_not_stack_device_defaults():
    raw = timed_case()
    raw["machine"]["device"]["launch_us"] = 10
    raw["runtime"]["submission"] = {"graph_replay_ms": .03}
    operations = (vector("a"), vector("b", deps=("a",)))
    eager = graph_costs(raw, operations, include_submission=True)
    replay = graph_costs(raw, operations, include_submission=True, replay_scope="full")
    assert eager["duration_ms"] == pytest.approx(.0225)
    assert replay["duration_ms"] == pytest.approx(.035)
    raw["runtime"]["kernels"] = {"invocations": [{"match": {}, "minimum_ms": 0}]}
    overridden = graph_costs(raw, operations)
    assert overridden["duration_ms"] == pytest.approx(.002)
    assert overridden["execution"]["work"] == eager["execution"]["work"]


def test_no_implicit_aiv_on_dma_or_join_and_per_device_parameter_override():
    raw = timed_case()
    operations = (op("dma", memory=1), replace(op("join", deps=("dma",)), family="barrier"))
    result = graph_costs(raw, operations)
    assert result["duration_ms"] == pytest.approx(.001)
    assert result["parameters"].get("invocation_models", []) == []
    raw["machine"]["devices"] = [{"kernel_invocations": {"vector": {"minimum_ms": .004}}}]
    machine = Hardware.from_mapping(raw["machine"])
    assert Hardware.from_mapping(machine.to_dict()) == machine
    assert dict(machine.dies[0].kernel_invocations)["cube"].minimum_ms == .005
    assert operator_costs(raw, (vector(),))["operator_sum_ms"] == pytest.approx(.004)
    raw["machine"]["devices"][0]["kernel_invocations"]["vector"] = {"minimum_ms": 0}
    assert operator_costs(raw, (vector(),))["operator_sum_ms"] == pytest.approx(.001)


def test_source_call_expansion_charges_physical_kernels_not_join_or_tile():
    raw = mla_case(splits=3)
    call = next(operation for operation in target_operators(
        raw, phase="decode", tokens=1, context=16) if operation.source_boundary == "call")
    before = operator_costs(raw, (call,))
    raw["machine"]["device"]["kernel_invocations"] = {
        "cube": {"minimum_ms": 100}, "vector": {"minimum_ms": 100}}
    after = operator_costs(raw, (call,))
    assert after["operator_sum_ms"] == 200
    assert sum(model["invocations"] for row in after["operators"]
               for model in row["basis"]["invocation_models"]) == 2
    assert [row["serviced"] for row in after["operators"]] == [
        row["serviced"] for row in before["operators"]]
