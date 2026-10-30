"""Unseen TP and token geometry use the real frontend and one physical Engine."""
from copy import deepcopy
from dataclasses import replace
from itertools import product

import pytest

from mindie_sim import graph_costs, operator_costs
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.response_functions import ResponseFunction
from mindie_sim.prediction.services import service_identity
from mindie_sim.prediction.structure import parse_case
from test_prediction_system import case


def geometry_case(tp):
    raw = case(tp=tp)
    raw["model"].update(hidden_size=128, num_heads=16, num_kv_heads=8,
                        intermediate_size=256, vocab_size=256)
    raw["runtime"].update(vllm_impl="declared-test-runtime", vllm_version="fixture-v1")
    device = raw["machine"]["device"]
    device["rates"]["matrix_service"] = 1e9
    device["work_units"] = {"matrix_service": "busy_ns"}
    device["resource_units"] = {"matrix_service": 1}
    return raw


def lowered(raw, recipe, rows, *, context=4096):
    operation = next(op for op in target_operators(raw,
        phase="decode" if rows == 1 else "prefill", tokens=rows, context=context)
        if op.recipe == recipe)
    # Isolate the frontend's unchanged physical operator from the rest of the
    # transformer graph; input readiness is supplied explicitly by the caller.
    return replace(operation, deps=())


def rule_for(raw, operation, response):
    identity = service_identity(parse_case(raw), operation)
    return {"id": "geometry-response", "match": {
        **{key: identity[key] for key in (
            "runtime", "revision", "dtype", "layout", "implementation", "replay_scope")},
        "hardware": identity["hardware"]["key"], "recipe": operation.recipe},
        "coordinates": {"m": {"path": "local_shape.0"}, "n": {"path": "local_shape.1"},
            "k": {"path": "local_shape.2"}, "flops": {"path": "work.cube"},
            "memory_bytes": {"path": "work.hbm"}},
        "response": response,
        "execution": {"kind": "joint_native", "resource": "matrix_service",
            "reference_rate": 1e9, "native_kernel": "DeclaredPhysicalProgram"},
        "provenance": {"basis": "declared_supply_control_not_hardware_calibration"}}


def bind(raw, rule):
    raw["runtime"]["kernels"] = {"service_regimes": [deepcopy(rule)]}
    return raw


def roofline():
    return {"kind": "roofline", "fixed_us": 3, "services": [
        {"coordinate": "flops", "resource": "cube", "dtype": "bf16", "utilization": .5},
        {"coordinate": "memory_bytes", "resource": "hbm", "utilization": .25}]}


@pytest.mark.parametrize("recipe", ("qkv_projection", "attention_output", "dense_down"))
def test_same_supply_parameters_follow_tp2_to_tp8_and_large_prefill_without_point_rules(recipe):
    source = geometry_case(2)
    rule = rule_for(source, lowered(source, recipe, 1), roofline())
    original = deepcopy(rule)
    shapes = {}
    for tp, rows in product((2, 8), (1, 4096)):
        raw = bind(geometry_case(tp), rule)
        op = lowered(raw, recipe, rows)
        m, n, k = op.local_shape
        shapes[tp, rows] = (m, n, k)
        # Independent dense BF16 arithmetic and compulsory operand traffic.
        assert op.flops == 2 * m * n * k
        assert op.hbm_bytes == 2 * (m * k + k * n + m * n)
        expected_ms = .003 + max(2 * m * n * k / 5e8,
                                 2 * (m * k + k * n + m * n) / 2.5e8) * 1000
        result = graph_costs(raw, (op,), input_ready_ms=7)
        isolated = operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]
        assert result["duration_ms"] == pytest.approx(expected_ms)
        assert isolated["duration_ms"] == pytest.approx(expected_ms)
        assert result["complete_ms"] == pytest.approx(7 + expected_ms)
        assert result["execution"]["work"]["flops"] == op.flops
        assert result["execution"]["work"]["hbm_bytes"] == op.hbm_bytes
        assert not isolated["service_before_input"]
        record = result["parameters"]["service_regimes"][0]
        assert record["response_relations"] == {"physical_supply": 1}
        assert record["coordinate_ranges"]["m"] == {"min": rows, "max": rows}
    assert shapes[2, 1][1] * shapes[2, 1][2] == 4 * shapes[8, 1][1] * shapes[8, 1][2]
    assert rule == original


def test_context_changes_actual_kv_traffic_and_supply_bottleneck_without_refitting():
    source = geometry_case(2)
    operation = lowered(source, "flash_attention", 1)
    rule = rule_for(source, operation, roofline())
    predictions, memory_work = [], []
    for context in (1024, 8192):
        raw = bind(geometry_case(8), rule)
        op = lowered(raw, "flash_attention", 1, context=context)
        result = graph_costs(raw, (op,))
        expected = .003 + 1000 * max(op.flops / 5e8, op.hbm_bytes / 2.5e8)
        assert result["duration_ms"] == pytest.approx(expected)
        predictions.append(result["duration_ms"])
        memory_work.append(result["execution"]["work"]["hbm_bytes"])
    assert predictions[1] > predictions[0]
    assert memory_work[1] > memory_work[0]
    # This large-M control is compute limited: changing only matrix supply
    # changes duration, while declared bytes and arithmetic remain identical.
    raw = geometry_case(8)
    op = lowered(raw, "qkv_projection", 4096)
    bind(raw, rule_for(raw, op, roofline()))
    before = graph_costs(raw, (op,))
    raw["machine"]["device"]["compute_peaks"]["bf16"] *= 2
    after = graph_costs(raw, (op,))
    assert after["duration_ms"] - .003 == pytest.approx((before["duration_ms"] - .003) / 2)
    assert after["execution"]["work"] == before["execution"]["work"]


@pytest.mark.parametrize("recipe", ("qkv_projection", "attention_output"))
def test_complete_multi_axis_training_grid_interpolates_unseen_local_shapes_in_engine(recipe):
    # A synthetic multilinear oracle establishes interpolation mechanics; these
    # parameters are not presented as an observed hardware performance surface.
    oracle = lambda m, n, k: 3 + .01 * m * n * k + .001 * m * n
    axes = ([1, 8192], [32, 128], [16, 128])
    response = {"kind": "multilinear", "grid": [
        {"coordinate": name, "knots": points} for name, points in zip(("m", "n", "k"), axes)],
        "values_us": [oracle(*point) for point in product(*axes)]}
    source = geometry_case(2)
    rule = rule_for(source, lowered(source, recipe, 1), response)
    rule["coordinates"] = {name: rule["coordinates"][name] for name in ("m", "n", "k")}
    for tp in (2, 4, 8):
        raw = bind(geometry_case(tp), rule)
        op = lowered(raw, recipe, 4096)
        result = graph_costs(raw, (op,), input_ready_ms=11)
        assert result["duration_ms"] == pytest.approx(oracle(*op.local_shape) / 1000)
        record = result["parameters"]["service_regimes"][0]
        assert record["response_relations"] == {"interpolated": 1}
        assert result["execution"]["work"]["hbm_bytes"] == op.hbm_bytes
    function = ResponseFunction.from_mapping(response, rule["coordinates"])
    assert function.evaluate({"m": 1, "n": 32, "k": 16}).basis == "anchor"
    assert function.evaluate({"m": 16384, "n": 64, "k": 32}).duration_us is None
    extrapolated = ResponseFunction.from_mapping({**response, "outside": "linear"}, rule["coordinates"])
    answer = extrapolated.evaluate({"m": 16384, "n": 64, "k": 32})
    assert answer.basis == "extrapolated"
    assert answer.duration_us == pytest.approx(oracle(16384, 64, 32))
