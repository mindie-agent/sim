"""Observable response regimes preserve actual common-engine execution."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs, predict
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from test_device_kernel_groups import fork
from test_graph_analysis import op
from test_invocation_binding import lower_copy
from test_mechanism_contract import hardware_case
from test_resource_services import copy_op


def declared(rule, raw=None):
    raw = deepcopy(raw or hardware_case())
    raw.setdefault("runtime", {}).setdefault("kernels", {})["service_regimes"] = [rule]
    return raw


def response_rule(**changes):
    return {"id": "observed-copy", "match": {"family": "copy"},
            "coordinates": {"bytes": {"path": "work.hbm"}},
            "domain": {"bytes": {"min": 1000, "max": 4000}},
            "response": {"fixed_us": 1000, "terms": [{"coordinate": "bytes", "us_per_unit": 2,
                "scale_with": {"resource": "hbm", "reference_rate": 1e6}}]},
            "provenance": {"kind": "synthetic_causal_control"}, **changes}


def test_observable_operator_graph_and_serving_use_same_late_ready_inclusive_response():
    raw = declared(response_rule(match={"recipe": copy_op(4000).recipe}))
    raw["workload"]["requests"][0]["arrival_ms"] = 7
    isolated = operator_costs(raw, (copy_op(4000),), input_ready_ms=7)
    graph = graph_costs(raw, (copy_op(4000),), input_ready_ms=7)
    serving = predict(raw, lower=lower_copy)
    assert isolated["operator_sum_ms"] == graph["duration_ms"] == 9
    assert graph["complete_ms"] == serving["requests"][0]["finish_ms"] == 16
    assert {k: v for k, v in graph["execution"]["serviced"].items() if v} == {"memory@r0:s0": 4000}
    assert isolated["operators"][0]["service_before_input"] == {}
    assert graph["parameters"]["service_regimes"][0]["queries"] == {"observable_response": 1}
    assert "observable_kernel_regimes" in graph["parameters"]["calibration"]
    assert serving["execution"]["live_tasks"] == serving["execution"]["live_programs"] == 0


def test_effective_rate_scales_without_inflating_native_work_or_ignoring_shared_contention():
    rule = response_rule(match={"recipe": copy_op(4000).recipe})
    spans = []
    for factor in (.5, 1, 2):
        raw = declared(rule)
        raw["machine"]["device"]["rates"]["memory"] *= factor
        result = graph_costs(raw, (copy_op(4000),), input_ready_ms=3, isolated=False)
        spans.append(result["duration_ms"])
        assert {k: v for k, v in result["execution"]["serviced"].items() if v} == {"memory@r0:s0": 4000}
    assert spans == [17, 9, 5]
    # Physical contention can exceed the observed isolated response. It is not
    # replaced by an independent callback that ignores simultaneous work.
    raw = declared(response_rule(response={"fixed_us": 4500}, match={"recipe": copy_op(4000).recipe}))
    operators = tuple(replace(copy_op(4000), op_id=str(i), stream=str(i)) for i in range(3))
    result = graph_costs(raw, operators, isolated=False)
    assert result["duration_ms"] == 12
    assert {k: v for k, v in result["execution"]["serviced"].items() if v} == {"memory@r0:s0": 12000}


def test_native_group_prices_one_response_without_stacking_it_on_phases_or_startup():
    raw = declared({"id": "native", "match": {"kernel": "NativeDispatch"},
                    "response": {"fixed_us": 10000}, "provenance": {"boundary": "device_kernel"}})
    raw["runtime"]["kernels"]["invocations"] = [{"match": {"kernel": "NativeDispatch"},
                                                    "startup_ms": 3}]
    result = graph_costs(raw, fork(), input_ready_ms=11, isolated=False)
    assert result["complete_ms"] == 21  # max(10 response, 3 startup + 6 body)
    assert result["parameters"]["service_regimes"][0]["queries"] == {"observable_response": 1}
    assert result["execution"]["serviced"]["memory@r0:s0"] == 5000
    phases = operator_costs(raw, fork())
    assert phases["operator_sum_ms"] == 8
    assert all("service_regimes" not in row["basis"] for row in phases["operators"])


def test_prepared_source_schedule_selects_group_service_and_domain_keeps_analytic_path():
    def plan(operation, geometry):
        work = Work(resources={"hbm": 2000})
        return OperatorPlan(work, lambda c: KernelProgram((Stage("read", c.bind(work)),)),
                            metadata={"group_visits": (2, 4), "branch": "small"})
    operation = replace(op("source"), recipe="group-source", source_boundary="kernel")
    recipes = RecipeBook({"group-source": PlannedRecipe(plan)})
    rule = {"id": "small-groups", "match": {"metadata.program_execution.branch": "small"},
            "coordinates": {"groups": {"path": "metadata.program_execution.group_visits", "reduce": "max"}},
            "domain": {"groups": {"min": 2, "max": 4}},
            "response": {"terms": [{"coordinate": "groups", "us_per_unit": 3000}]}}
    result = graph_costs(declared(rule), (operation,), recipes=recipes, isolated=False)
    assert result["duration_ms"] == 12
    assert {k: v for k, v in result["execution"]["serviced"].items() if v} == {"memory@r0:s0": 2000}
    rule["domain"]["groups"]["max"] = 3
    outside = graph_costs(declared(rule), (operation,), recipes=recipes, isolated=False)
    assert outside["duration_ms"] == 2
    assert outside["parameters"]["service_regimes"][0]["queries"] == {"analytic_outside_domain": 1}


def test_parameter_provenance_and_model_labels_do_not_change_calculated_response():
    raw = declared(response_rule(match={"recipe": copy_op(4000).recipe}))
    control = graph_costs(raw, (copy_op(4000),), isolated=False)
    raw["model"]["name"] = "unseen architecture label"
    raw["runtime"]["kernels"]["service_regimes"][0]["provenance"]["validation_observed_us"] = 1e9
    changed = graph_costs(raw, (copy_op(4000),), isolated=False)
    assert control["duration_ms"] == changed["duration_ms"] == 9
    assert control["execution"]["serviced"] == changed["execution"]["serviced"]
