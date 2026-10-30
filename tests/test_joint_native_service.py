"""Observed native service is a declared queue, distinct from physical byte service."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs, predict
from mindie_sim.schema.common import ModelingError
from test_device_kernel_groups import fork
from test_invocation_binding import lower_copy
from test_resource_services import copy_op
from test_service_regimes import declared, response_rule


def joint_case():
    rule = response_rule(match={"recipe": copy_op(4000).recipe},
                         response={"fixed_us": 2000}, execution={
        "kind": "joint_native", "resource": "observed_copy", "reference_rate": 1e9,
        "native_kernel": "DeclaredNativeCopy"})
    raw = declared(rule)
    device = raw["machine"]["device"]
    device["rates"]["observed_copy"] = 1e9
    device["work_units"] = {"observed_copy": "busy_ns"}
    device["resource_units"] = {"observed_copy": 1}
    return raw


def test_joint_service_can_replace_high_analytic_bound_without_relabeling_bytes():
    raw = joint_case()
    # Full native response includes its device startup/tail exactly once.
    raw["runtime"]["kernels"]["invocations"] = [
        {"match": {"recipe": copy_op(4000).recipe}, "startup_ms": 5, "minimum_ms": 9}]
    raw["workload"]["requests"][0]["arrival_ms"] = 7
    row = operator_costs(raw, (copy_op(4000),), input_ready_ms=7)["operators"][0]
    graph = graph_costs(raw, (copy_op(4000),), input_ready_ms=7)
    serving = predict(raw, lower=lower_copy)
    assert row["duration_ms"] == graph["duration_ms"] == 2
    assert serving["requests"][0]["finish_ms"] == graph["complete_ms"] == 9
    assert row["analytical_reference_work"]["hbm"] == 4000
    assert row["serviced"] == row["declared_service"] == {"observed_copy@r0:s0": 2e6}
    assert not row["service_delta"] and not row["service_before_input"]
    assert "invocation_models" not in row["basis"]
    assert graph["execution"]["work"]["hbm_bytes"] == 4000
    assert graph["execution"]["work_basis"].startswith("logical_and_analytical_reference")
    assert graph["parameters"]["joint_native_services"][0]["analytical_reference_work"]["hbm"] == 4000
    assert not graph["execution"]["live_tasks"] and not graph["execution"]["live_programs"]


def test_replaced_primitive_and_source_program_are_reported_as_reference_only():
    from mindie_sim.prediction.cost import primitive_key
    from mindie_sim.prediction.primitives import PrimitiveStore
    from mindie_sim.prediction.structure import parse_case
    raw = joint_case()
    with PrimitiveStore(":memory:") as store:
        store.record(primitive_key(parse_case(raw), copy_op(4000), "none"),
                     duration_ms=20, run_id="reference", repeat_index=0, source="synthetic_control")
        row = operator_costs(raw, (copy_op(4000),), primitives=store)["operators"][0]
        graph = graph_costs(raw, (copy_op(4000),), primitives=store)
    assert row["duration_ms"] == 2
    assert row["program"] == "JointNativeService"
    assert row["joint_native_service"]["reference_primitive_ms"] == 20
    assert row["basis"]["operator_occurrences"] == {
        "measured_primitive": 0, "analytic_work": 0, "joint_native_service": 1}
    assert "primitive_completion_parameters" not in row["basis"]["calibration"]
    assert row["basis"]["joint_native_services"][0]["replaced_primitive_count"] == 1
    assert graph["timeline"][0]["program"] == "JointNativeService"


def test_joint_queue_contends_across_streams_and_keeps_dependences_and_rate_scans():
    operators = tuple(replace(copy_op(4000), op_id=str(i), stream=str(i)) for i in range(3))
    for factor in (.5, 1, 2):
        raw = joint_case()
        raw["machine"]["device"]["rates"]["observed_copy"] *= factor
        result = graph_costs(raw, operators, input_ready_ms=7, isolated=False)
        assert result["duration_ms"] == pytest.approx(6 / factor)
        assert result["execution"]["serviced"]["observed_copy@r0:s0"] == 6e6
    last = replace(operators[2], deps=("0", "1"))
    result = graph_costs(joint_case(), (*operators[:2], last), input_ready_ms=7)
    assert result["duration_ms"] == 6


def test_unmatched_domain_still_executes_analytic_program_and_different_queue_overlaps():
    raw = joint_case()
    miss = graph_costs(raw, (copy_op(5000),), isolated=False)
    assert miss["duration_ms"] == 5
    assert "joint_native_services" not in miss["parameters"]
    raw["machine"]["device"]["rates"]["other_copy"] = 1e9
    raw["machine"]["device"]["work_units"]["other_copy"] = "busy_ns"
    raw["machine"]["device"]["resource_units"]["other_copy"] = 1
    rule = deepcopy(raw["runtime"]["kernels"]["service_regimes"][0])
    rule.update(id="other", match={"recipe": "second_copy"})
    rule["execution"]["resource"] = "other_copy"
    raw["runtime"]["kernels"]["service_regimes"].append(rule)
    a = replace(copy_op(4000), op_id="a", stream="a")
    b = replace(copy_op(4000), op_id="b", stream="b", recipe="second_copy")
    result = graph_costs(raw, (a, b), isolated=False)
    assert result["duration_ms"] == 2


def test_independent_dp_devices_do_not_share_the_observed_service_queue():
    raw = joint_case()
    raw["machine"]["num_devices"] = 2
    raw["runtime"]["dp"] = 2
    result = graph_costs(raw, {0: (copy_op(4000),), 1: (copy_op(4000),)}, isolated=False)
    assert result["duration_ms"] == 2
    assert result["execution"]["serviced"]["observed_copy@r0:s0"] == 2e6
    assert result["execution"]["serviced"]["observed_copy@r1:s0"] == 2e6


def test_joint_service_cannot_hide_native_multiphase_or_network_program():
    raw = joint_case()
    rule = raw["runtime"]["kernels"]["service_regimes"][0]
    rule["match"] = {"kernel": "NativeDispatch"}
    rule.pop("coordinates")
    rule.pop("domain")
    with pytest.raises(ModelingError, match="single_local_kernel"):
        graph_costs(raw, fork(), isolated=False)
    rule["match"] = {"recipe": copy_op(4000).recipe}
    with pytest.raises(ModelingError, match="single_local_kernel"):
        operator_costs(raw, (replace(copy_op(4000), family="collective"),))
