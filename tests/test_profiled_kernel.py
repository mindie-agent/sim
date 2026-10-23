"""Consume a DB produced from the real A3 golden CSV excerpt by profiler."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.a3_cost import PredictiveCostModel
from mindie_sim.a3_calibration_store import SQLiteCalibrationStore
from mindie_sim.a3_measurement import record_prediction_misses
from mindie_sim.a3_ops import OpSpec
from mindie_sim.a3_types import MachineSpec
from mindie_sim.a3_types import A3InputError
from mindie_sim.operator_estimation import create_operator_estimator, operator_plan_report
from mindie_sim.operator_plan import EstimateContext
from mindie_sim.profiled_kernel import ProfiledKernelProvider

FIXTURE = Path(__file__).parent / "fixtures" / "profiled_kernel"


def _inputs():
    source = json.loads((FIXTURE / "source.json").read_text())
    binding = {"variant_id": source["variant_id"], "family": "cast", "recipe": source["signature"]["name"],
               "phase": "observed", "launch_shape": [8192], "dtype": "int32", "layout": "ND"}
    config = {"db": str(FIXTURE / "observed.sqlite"), "environment_key": source["context"]["environment_key"], "bindings": [binding]}
    machine = json.loads((Path(__file__).parents[1] / "examples/a3_real_hardware_predict.json").read_text())["machine"]
    return source, binding, config, machine


def test_real_captured_duration_enters_existing_resource_engine_without_duplicate_service():
    source, binding, config, machine = _inputs()
    report = operator_plan_report({
        "machine": machine, "estimators": {"profiled_kernels": config, "strict": True},
        "context": {"timing_scope": "profiled_device_kernel"},
        "operators": [{"op_id": "captured-cast", "family": "cast", "phase": "observed",
                       "recipe": binding["recipe"], "launch_shape": [8192], "dtype": "int32", "layout": "ND",
                       "metadata": {"profile_variant": source["variant_id"]}}],
    })
    # The existing host/device dispatch term is separate from the captured
    # kernel; service is replaced once rather than added to an analytic cost.
    assert report["elapsed_ms"] == pytest.approx(
        source["median_ns"] / 1e6 + report["operators"][0]["launch_ms"]
    )
    op = report["operators"][0]
    assert op["operator_estimation"]["status"] == "observed_plan"
    assert op["operator_estimation"]["not_npu_measured"] is False
    assert op["calibration"]["status"] == "profiled_kernel_observed"
    assert op["calibration"]["observations_used"] > 0
    assert op["execution_plan"]["provenance"]["source_class"] == "observed"
    assert op["execution_plan"]["resource_totals"] == {}


def test_observed_cost_does_not_cross_hardware_scope_or_shape():
    _, binding, config, raw_machine = _inputs()
    provider = ProfiledKernelProvider(config)
    machine = MachineSpec.from_mapping(raw_machine)
    op = OpSpec("cast", "cast", "observed", 0, 0, (), (8192,), (8192,), (8192,), (8192,),
                recipe=binding["recipe"], dtype="int32", layout="ND")
    assert provider.estimate(op, machine, EstimateContext()).plan is None
    context = EstimateContext(timing_scope="profiled_device_kernel")
    assert provider.estimate(op, replace(machine, hardware_key="Ascend950DT_9582"), context).plan is None
    assert provider.estimate(replace(op, launch_shape=(4096,)), machine, context).plan is None
    assert provider.estimate(op, machine, context).plan is not None


def test_observed_kernel_is_not_submitted_as_a_new_measurement(tmp_path):
    store = SQLiteCalibrationStore(tmp_path / "calibration.sqlite")
    result = record_prediction_misses(store, {"traces": [{"ops": [
        {"family": "cast", "calibration_status": "profiled_kernel_observed", "launch_shape": [8192]},
        {"family": "cast", "calibration_status": "uncalibrated", "launch_shape": [4096]},
    ]}]})
    assert result["recorded_requests"] == 1
    assert result["skipped_calibrated"] == 1


def test_explicit_observed_only_operator_cannot_fall_back_to_zero_cost():
    source, binding, config, machine = _inputs()
    config["environment_key"] = "different-software-stack"
    with pytest.raises(A3InputError, match="profile_variant_unavailable"):
        operator_plan_report({
            "machine": machine, "estimators": {"profiled_kernels": config},
            "context": {"timing_scope": "profiled_device_kernel"},
            "operators": [{"op_id": "captured-cast", "family": "cast", "phase": "observed",
                           "recipe": binding["recipe"], "launch_shape": [8192], "dtype": "int32", "layout": "ND",
                           "metadata": {"profile_variant": source["variant_id"]}}],
        })
