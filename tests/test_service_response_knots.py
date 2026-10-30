"""Unseen physical points traverse the same Engine and explicit identity gates."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs, predict
from mindie_sim.prediction.response_functions import ResponseFunction
from mindie_sim.schema.operators import IterationWork
from test_joint_native_service import joint_case
from test_resource_services import copy_op


def curve_case(*, outside="analytic", joint=True):
    raw = joint_case()
    rule = raw["runtime"]["kernels"]["service_regimes"][0]
    rule.update(match={"hardware": "arbitrary", "recipe": "copy", "dtype": "bf16",
        "layout": "contiguous", "replay_scope": "none", "implementation": {},
        "metadata.protocol": "retained"},
        coordinates={"bytes": {"path": "work.hbm"},
                     "context": {"path": "metadata.context_tokens"}},
        domain={"bytes": {"min": 500, "max": 8000}, "context": {"min": 4096, "max": 4096}},
        response={"kind": "piecewise_linear", "coordinate": "bytes",
                  "knots": [[1000, 4000], [2000, 6000], [4000, 9000]], "outside": outside})
    if not joint:
        del rule["execution"]
    return raw


def operation(amount, *, context=4096):
    return copy_op(amount, metadata=(("context_tokens", context), ("protocol", "retained")))


@pytest.mark.parametrize("joint", (False, True))
def test_unseen_points_interpolate_in_real_engine_with_late_input_and_unchanged_work(joint):
    raw = curve_case(joint=joint)
    for amount, duration in ((1000, 4), (1500, 5), (2000, 6), (3000, 7.5), (4000, 9)):
        op = operation(amount)
        isolated = operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]
        graph = graph_costs(raw, (op,), input_ready_ms=7, isolated=False)
        assert isolated["duration_ms"] == graph["duration_ms"] == duration
        assert graph["complete_ms"] == 7 + duration
        assert isolated["operator"]["hbm_bytes"] == amount
        record = graph["parameters"]["service_regimes"][0]
        assert record["response_relations"] == {"anchor" if amount in {1000, 2000, 4000} else "interpolated": 1}
        assert record["coordinate_ranges"]["bytes"] == {"min": amount, "max": amount}
        assert not isolated["service_before_input"]
        assert not graph["execution"]["live_tasks"] and not graph["execution"]["live_programs"]


def test_explicit_extrapolation_is_distinct_from_analytic_outside_curve_and_application_domain():
    safe = graph_costs(curve_case(), (operation(5000),), isolated=False)
    assert safe["duration_ms"] == 5
    assert safe["parameters"]["service_regimes"][0]["queries"] == {"analytic_outside_knots": 1}
    assert safe["parameters"]["regime_selection"]["unselected"] == ["observed-copy"]
    extrapolated = graph_costs(curve_case(outside="linear"), (operation(5000),), isolated=False)
    assert extrapolated["duration_ms"] == 10.5
    assert extrapolated["parameters"]["service_regimes"][0]["response_relations"] == {"extrapolated": 1}
    limited = graph_costs(curve_case(outside="linear"), (operation(9000),), isolated=False)
    assert limited["duration_ms"] == 9
    assert limited["parameters"]["service_regimes"][0]["queries"] == {"analytic_outside_domain": 1}


def test_fixed_context_and_protocol_misses_never_borrow_a_response_from_another_domain():
    context_miss = graph_costs(curve_case(), (operation(1500, context=4097),), isolated=False)
    assert context_miss["duration_ms"] == 1.5
    assert context_miss["parameters"]["service_regimes"][0]["queries"] == {"analytic_outside_domain": 1}
    changed = replace(operation(1500), metadata=(("protocol", "different"), ("context_tokens", 4096)))
    report = graph_costs(curve_case(), (changed,), isolated=False)
    assert report["duration_ms"] == 1.5
    record = report["parameters"]["service_regimes"][0]
    assert record["queries"] == {"physical_identity_mismatch": 1}
    assert record["mismatched_axes"] == {"metadata.protocol": 1}
    assert report["parameters"]["calibration"] == "none"
    assert report["parameters"]["regime_selection"]["unmatched"] == ["observed-copy"]


def test_two_unseen_points_share_joint_queue_and_coordinate_report_is_bounded():
    ops = (replace(operation(1500), op_id="a", stream="a"),
           replace(operation(3000), op_id="b", stream="b"))
    result = graph_costs(curve_case(), ops, isolated=False)
    assert result["duration_ms"] == 12.5
    assert result["execution"]["serviced"]["observed_copy@r0:s0"] == 12.5e6
    record = result["parameters"]["service_regimes"][0]
    assert record["response_relations"] == {"interpolated": 2}
    assert record["coordinate_ranges"]["bytes"] == {"min": 1500, "max": 3000}
    assert result["execution"]["work"]["hbm_bytes"] == 4500


def test_serving_context_growth_uses_unseen_coordinate_without_refitting_or_point_rules():
    raw = curve_case()
    rule = raw["runtime"]["kernels"]["service_regimes"][0]
    rule.update(coordinates={"context": {"path": "metadata.context_tokens"}},
                domain={"context": {"min": 1, "max": 3}},
                response={"kind": "piecewise_linear", "coordinate": "context",
                          "knots": [[1, 4000], [3, 9000]]})
    original = deepcopy(rule)
    raw["workload"]["requests"][0].update(output_tokens=3, arrival_ms=7)

    def growing_context(plan, model, runtime, machine):
        view = plan.requests[0]
        op = operation(1000, context=view.kv_len)
        return IterationWork((op,), plan, 1, 1, 1, None, view.phase, "synthetic_context_growth")

    result = predict(raw, lower=growing_context)
    assert result["requests"][0]["finish_ms"] == 7 + 4 + 6.5 + 9
    record = result["parameters"]["service_regimes"][0]
    assert record["response_relations"] == {"anchor": 2, "interpolated": 1}
    assert record["coordinate_ranges"] == {"context": {"min": 1, "max": 3}}
    assert raw["runtime"]["kernels"]["service_regimes"][0] == original


def test_pure_response_evaluation_matches_engine_but_does_not_consume_observed_values():
    raw = curve_case()
    rule = raw["runtime"]["kernels"]["service_regimes"][0]
    model = ResponseFunction.from_mapping(rule["response"], rule["coordinates"])
    for observed_us in (7500, 15000):
        predicted = model.evaluate({"bytes": 3000, "context": 4096})
        error = predicted.duration_us - observed_us
        assert predicted.duration_us == 7500
        assert error == 7500 - observed_us
    result = graph_costs(raw, (operation(3000),), isolated=False)
    assert result["duration_ms"] * 1000 == model.evaluate({"bytes": 3000}).duration_us
