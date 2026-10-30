"""Public invocation boundaries checked against causal resource arithmetic."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs, predict, target_operators
from mindie_sim.prediction.cost import primitive_key
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.primitives import PrimitiveStore
from mindie_sim.prediction.program import KernelProgram, RecipeBook
from mindie_sim.prediction.services import ResourceService
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork
from test_communication_topology import exchange
from test_graph_analysis import op
from test_joint_collective_execution import cohort_case
from test_mechanism_contract import hardware_case
from test_mla_family_selection import mla_case
from test_resource_services import copy_op, curve


def declare(raw, *, startup=3, minimum=0, match=None):
    raw.setdefault("runtime", {}).setdefault("kernels", {})["invocations"] = [{
        "match": match or {}, "startup_ms": startup, "minimum_ms": minimum,
        "provenance": {"kind": "independent_causal_control"},
    }]
    return raw


def lower_copy(plan, model, runtime, machine):
    return IterationWork(
        (copy_op(4000),), plan, 1, 1, 1, None, "prefill", "invocation_control")


def test_a5_ordinary_class_minima_are_inclusive_once_and_keep_dma_outside():
    raw = hardware_case()
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1}
    mixed = replace(op("mixed", flops=2), vector_ops=1)
    small = operator_costs(raw, (mixed,), input_ready_ms=7)["operators"][0]
    assert small["duration_ms"] == pytest.approx(.001578)
    assert small["service_before_input"] == {}
    # A combined Cube/Vector kernel takes the maximum class minimum once;
    # it does not pay both or append that value after useful resource work.
    large = replace(mixed, hbm_bytes=4e8)
    result = graph_costs(raw, (large,), input_ready_ms=7, trace_limit=100)
    assert result["duration_ms"] == pytest.approx(.1)
    assert result["execution"]["serviced"]["memory@r0:s0"] == 4e8
    assert not any(event["task"].endswith(":startup") for event in result["events"])
    dma = operator_costs(raw, (op("dma", memory=400),))["operators"][0]
    assert dma["duration_ms"] == pytest.approx(1e-7)
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


def test_a5_quantized_and_empty_classes_are_explicit_boundaries_not_stacked_floors():
    raw = hardware_case()
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1}
    # Native dtype/scales do not establish a library's ND vs NZ dispatch.
    # The call owner declares the measured class; ordinary quantized/mixed
    # programs retain their ordinary Cube/Vector envelope.
    plain = replace(op("quant", flops=2), vector_ops=4, dtype="int8")
    declared = replace(plain, metadata=(("invocation_class", "w8a8_nd_bf16"),))
    empty = replace(op("empty"), recipe="empty", source_boundary="kernel",
                    metadata=(("invocation_class", "empty_aic"),))
    recipes = RecipeBook({"empty": PlannedRecipe(
        lambda op, geometry: OperatorPlan(Work(), lambda context: KernelProgram(())))})
    for operation, expected in ((plain, .001578), (declared, .003037), (empty, .000439)):
        report = operator_costs(raw, (operation,), recipes=recipes, input_ready_ms=3)
        assert report["operator_sum_ms"] == pytest.approx(expected)
        assert report["operators"][0]["service_before_input"] == {}
    # A large declared call pays max(body, minimum), not body + floor.
    big = replace(declared, hbm_bytes=4e8)
    report = graph_costs(raw, (big,), input_ready_ms=3, isolated=False)
    assert report["duration_ms"] == pytest.approx(.1)
    assert report["execution"]["serviced"]["memory@r0:s0"] == 4e8
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


def test_a5_custom_execution_class_and_runtime_rule_replace_ordinary_assumption():
    raw = hardware_case()
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1,
                      "device": {"kernel_invocations": {"custom_tiny_add": {"minimum_ms": .000876}}}}
    custom = replace(op("tiny"), vector_ops=1, metadata=(("invocation_class", "custom_tiny_add"),))
    assert operator_costs(raw, (custom,))["operators"][0]["duration_ms"] == pytest.approx(.000876)
    raw = declare(raw, startup=0, minimum=0)
    disabled = operator_costs(raw, (replace(custom, metadata=()),))["operators"][0]
    assert disabled["duration_ms"] < 1e-6


@pytest.mark.parametrize("minimum,expected", ((2, 7), (8, 8)))
def test_public_operator_graph_and_predict_share_original_ready_floor(minimum, expected):
    raw = declare(hardware_case(), minimum=minimum)
    raw["workload"]["requests"][0]["arrival_ms"] = 7
    operation = copy_op(4000)
    isolated = operator_costs(raw, (operation,), input_ready_ms=7)["operators"][0]
    graph = graph_costs(raw, (operation,), input_ready_ms=7)
    serving = predict(raw, lower=lower_copy)
    # One 1000 B/ms service supplies 4000 B after 3 ms startup. The completion
    # floor starts at input readiness; it is not delayed until startup ends.
    assert isolated["duration_ms"] == graph["duration_ms"] == expected
    assert serving["requests"][0]["finish_ms"] == graph["complete_ms"] == 7 + expected
    assert isolated["serviced"] == {"memory@r0:s0": 4000}
    assert graph["execution"]["serviced"]["memory@r0:s0"] == 4000
    assert serving["execution"]["serviced"]["memory@r0:s0"] == 4000
    assert isolated["service_before_input"] == {}
    assert graph["execution"]["live_tasks"] == serving["execution"]["live_tasks"] == 0


def test_primitive_completion_and_declared_floor_are_one_original_ready_constraint():
    raw = declare(hardware_case(), startup=3, minimum=8)
    operation = copy_op(4000)
    parsed = parse_case(raw)
    with PrimitiveStore(":memory:") as store:
        store.record(
            primitive_key(parsed, operation, "none"), duration_ms=12,
            run_id="constraint-control", repeat_index=0, source="independent-oracle")
        report = graph_costs(raw, (operation,), input_ready_ms=7, primitives=store)
    assert report["complete_ms"] == 19  # 7 + max(12, 8, 3 + 4)
    assert report["execution"]["serviced"]["memory@r0:s0"] == 4000
    assert report["isolated"]["operator_sum_ms"] == 12


def test_host_launch_and_graph_replay_stay_outside_per_kernel_startup():
    raw = declare(hardware_case(), startup=3)
    raw["machine"]["device"]["launch_us"] = 2000
    raw["runtime"]["submission"] = {"graph_replay_ms": 3}
    operators = (
        op("first", memory=1000),
        op("second", memory=1000, deps=("first",)),
    )
    device = graph_costs(raw, operators, isolated=False)
    eager = graph_costs(raw, operators, include_submission=True,
                        replay_scope="none", isolated=False, trace_limit=100)
    graph = graph_costs(raw, operators, include_submission=True,
                        replay_scope="full", isolated=False, trace_limit=100)
    # Body1 occupies [5,6], body2 [9,10]. The second eager host submission
    # [2,4] overlaps the first kernel instead of being added after it.
    assert device["duration_ms"] == 8
    assert eager["duration_ms"] == 10
    assert graph["duration_ms"] == 11
    assert eager["execution"]["serviced"]["host@r0:s0"] == 4
    assert graph["execution"]["serviced"]["host@r0:s0"] == 3
    assert sum(row["task"].endswith(":submit") for row in eager["events"]) == 2
    assert sum(row["task"].endswith(":submit") for row in graph["events"]) == 1
    assert eager["execution"]["serviced"]["memory@r0:s0"] == 2000
    assert graph["execution"]["serviced"]["memory@r0:s0"] == 2000


def test_source_mla_call_charges_two_kernels_and_never_the_logical_join():
    raw = mla_case(splits=3)
    call = next(op for op in target_operators(raw, phase="decode", tokens=1, context=16)
                if op.source_boundary == "call")
    external = {key: 5 for key in call.deps}
    plain = graph_costs(raw, (call,), external_inputs=external, isolated=False)
    timed = declare(deepcopy(raw), startup=1)
    graph = graph_costs(timed, (call,), external_inputs=external,
                        isolated=False, trace_limit=1000)
    isolated = operator_costs(timed, (call,))
    before = operator_costs(raw, (call,))
    assert graph["duration_ms"] == pytest.approx(plain["duration_ms"] + 2)
    assert isolated["operator_sum_ms"] == pytest.approx(before["operator_sum_ms"] + 2)
    assert graph["parameters"]["invocation_models"][0]["invocations"] == 2
    assert sum(row["task"].endswith(":startup") for row in graph["events"]) == 2
    assert graph["execution"]["serviced"] == plain["execution"]["serviced"]
    assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0


@pytest.mark.parametrize("minimum,idle_end", ((0, 9), (20, 20)))
def test_idle_collective_receiver_has_startup_without_global_completion_floor(minimum, idle_end):
    raw = declare(cohort_case(2), startup=5, minimum=minimum,
                  match={"recipe": "idle_receiver"})
    graphs = {
        0: (replace(exchange(0, axis="ep"), recipe="idle_receiver"),),
        1: (exchange(4, axis="ep"),),
    }
    report = graph_costs(raw, graphs)
    rows = {row["id"]: row for row in report["timeline"]}
    # The receiving rank's real call enters at5ms. Four bytes then use the
    # physical link until9ms; only its local floor may keep rank0 until20ms.
    assert rows["r0:exchange"]["complete_ms"] == idle_end
    assert rows["r1:exchange"]["complete_ms"] == 9
    assert sum(v for k, v in report["execution"]["serviced"].items()
               if k.startswith("link:")) == 4
    assert report["parameters"]["invocation_models"][0]["invocations"] == 1
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert report["isolated"]["collective_cohorts"][0]["duration_ms"] == idle_end


def test_inclusive_resource_response_rejects_a_second_same_boundary_startup():
    raw = declare(hardware_case(), startup=2)
    recipes = RecipeBook({"copy": ResourceService(curve())})
    with pytest.raises(ValueError, match="duplicates_resource_service_boundary"):
        operator_costs(raw, (copy_op(2000),), recipes=recipes)


def test_outside_response_domain_retains_declared_analytic_invocation():
    raw = declare(hardware_case(), startup=2)
    recipes = RecipeBook({"copy": ResourceService(curve())})
    report = operator_costs(raw, (copy_op(4000),), recipes=recipes)
    row = report["operators"][0]
    # The response curve is only defined on1000..3000 B; outside that range
    # its startup is absent, so the independent declared2ms is not a duplicate.
    assert row["duration_ms"] == 6
    assert row["basis"]["service_models"][0]["queries"] == {"analytic_outside_domain": 1}
    assert row["serviced"] == {"memory@r0:s0": 4000}
