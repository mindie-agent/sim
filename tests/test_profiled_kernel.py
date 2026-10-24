"""Consume a DB produced from the real A3 golden CSV excerpt by profiler."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.a3_cost import CalibrationObservation, CalibrationStore, PredictiveCostModel
from mindie_sim.a3_calibration_store import SQLiteCalibrationStore
from mindie_sim.a3_measurement import record_prediction_misses
from mindie_sim.a3_ops import OpSpec
from mindie_sim.a3_types import MachineSpec
from mindie_sim.a3_types import A3InputError
from mindie_sim.operator_estimation import create_operator_estimator, operator_plan_report
from mindie_sim.operator_plan import EstimateContext
from mindie_sim.profiled_kernel import CAPTURED_TIMING_SCOPE, ProfiledKernelProvider

FIXTURE = Path(__file__).parent / "fixtures" / "profiled_kernel"

# A second real variant of the same capture: ``aclnnRepeat_TileAiCore_Tile``
# reads ``8192,2,64`` BF16 and writes ``8192,2,128`` BF16, and was captured
# twice (6520 ns / 13740 ns).  Its declared HBM work follows from those real
# tensor shapes, so the operator declaration is derivable, not invented.
TILE_VARIANT = "3bb058898b3341c85048d7e7f33decb5"
TILE_RECIPE = "aclnnRepeat_TileAiCore_Tile"
TILE_READ_BYTES = 8192 * 2 * 64 * 2
TILE_WRITE_BYTES = 8192 * 2 * 128 * 2


def _inputs():
    source = json.loads((FIXTURE / "source.json").read_text())
    binding = {"variant_id": source["variant_id"], "family": "cast", "recipe": source["signature"]["name"],
               "phase": "observed", "launch_shape": [8192], "dtype": "int32", "layout": "ND",
               "timing_scope": CAPTURED_TIMING_SCOPE}
    config = {"db": str(FIXTURE / "observed.sqlite"), "environment_key": source["context"]["environment_key"], "bindings": [binding]}
    machine = json.loads((Path(__file__).parents[1] / "examples/a3_real_hardware_predict.json").read_text())["machine"]
    return source, binding, config, machine


def _cast_operator(op_id="captured-cast", **extra):
    """The captured Cast kernel reads 8192 INT32 and writes 8192 INT64."""
    operator = {
        "op_id": op_id, "family": "cast", "phase": "observed",
        "recipe": json.loads((FIXTURE / "source.json").read_text())["signature"]["name"],
        "launch_shape": [8192], "dtype": "int32", "layout": "ND",
        "hbm_bytes": float(8192 * 4 + 8192 * 8), "vector_ops": 8192.0,
        "metadata": {"profile_variant": json.loads((FIXTURE / "source.json").read_text())["variant_id"]},
    }
    operator.update(extra)
    return operator


def test_real_captured_duration_enters_existing_resource_engine_without_duplicate_service():
    source, binding, config, machine = _inputs()
    report = operator_plan_report({
        "machine": machine, "estimators": {"profiled_kernels": config, "strict": True},
        "operators": [_cast_operator()],
    })
    # The existing host/device dispatch term is separate from the captured
    # kernel; service is replaced once rather than added to an analytic cost.
    # This kernel's declared work is far below its captured duration, so the
    # captured median is the binding completion floor.
    assert report["elapsed_ms"] == pytest.approx(
        source["median_ns"] / 1e6 + report["operators"][0]["launch_ms"]
    )
    op = report["operators"][0]
    assert op["operator_estimation"]["status"] == "observed_plan"
    assert op["operator_estimation"]["not_npu_measured"] is False
    assert op["calibration"]["status"] == "profiled_kernel_observed"
    assert op["calibration"]["observations_used"] > 0
    assert op["execution_plan"]["provenance"]["source_class"] == "observed"
    # The captured kernel is not a free shared-stage task: it charges the
    # operator's own declared work, flagged as a proxy the capture cannot
    # confirm.  ``mindie-sim`` never reports this as measured occupancy.
    assert op["execution_plan"]["resource_totals"] == {
        "hbm": float(8192 * 4 + 8192 * 8), "vector": 8192.0,
    }
    provenance = op["execution_plan"]["provenance"]
    assert provenance["occupancy_basis"] == "declared_work_proxy"
    assert provenance["resource_occupancy"] == "not_captured"
    assert provenance["timing_scope"] == CAPTURED_TIMING_SCOPE
    serviced = report["execution"]["serviced"]
    assert serviced["hbm@s0"] == pytest.approx(float(8192 * 4 + 8192 * 8))
    assert serviced["vector@s0"] == pytest.approx(8192.0)


def test_captured_kernel_contends_for_the_stage_it_shares():
    """A concurrent analytic operator is delayed by the captured kernel's work.

    Before the declared-work demands existed the captured plan held no shared
    resource, so this DAG finished in exactly the analytic operator's own time.
    """
    source, binding, config, machine = _inputs()
    config = dict(config, bindings=[dict(
        binding, variant_id=TILE_VARIANT, recipe=TILE_RECIPE,
        family="copy_layout", launch_shape=[8192, 2, 64], dtype="bf16",
    )])
    analytic = {
        "op_id": "v0", "family": "cast", "phase": "prefill", "recipe": "analytic_vector",
        "launch_shape": [8192], "dtype": "bf16", "layout": "ND",
        "vector_ops": 2.0e9, "hbm_bytes": 4.0e8,
    }
    captured = {
        "op_id": "tile", "family": "copy_layout", "phase": "observed",
        "recipe": TILE_RECIPE, "launch_shape": [8192, 2, 64], "dtype": "bf16",
        "layout": "ND", "hbm_bytes": float(TILE_READ_BYTES + TILE_WRITE_BYTES),
        "metadata": {"profile_variant": TILE_VARIANT},
    }
    alone = operator_plan_report({
        "machine": machine, "estimators": {"profiled_kernels": config},
        "operators": [analytic],
    })
    shared = operator_plan_report({
        "machine": machine, "estimators": {"profiled_kernels": config},
        "operators": [captured, analytic],
    })
    assert shared["operators"][0]["operator_estimation"]["status"] == "observed_plan"
    assert shared["elapsed_ms"] > alone["elapsed_ms"]
    assert shared["execution"]["serviced"]["hbm@s0"] == pytest.approx(
        4.0e8 + TILE_READ_BYTES + TILE_WRITE_BYTES
    )


def test_real_variant_evidence_keeps_calls_and_observed_spread():
    """The real twice-captured Tile variant: 6520 / 13740 ns around a 10130 median."""
    _, binding, config, raw_machine = _inputs()
    provider = ProfiledKernelProvider(dict(config, bindings=[dict(
        binding, variant_id=TILE_VARIANT, recipe=TILE_RECIPE,
        family="copy_layout", launch_shape=[8192, 2, 64], dtype="bf16",
    )]))
    op = OpSpec("tile", "copy_layout", "observed", 0, 0, (), (8192, 2, 64), (8192, 2, 64),
                (8192, 2, 64), (8192, 2, 64), recipe=TILE_RECIPE, dtype="bf16", layout="ND",
                hbm_bytes=float(TILE_READ_BYTES + TILE_WRITE_BYTES))
    plan = provider.estimate(op, MachineSpec.from_mapping(raw_machine), EstimateContext()).plan
    provenance = plan.provenance
    assert provenance["captured_calls"] == 2
    assert provenance["captured_min_ns"] == 6520
    assert provenance["captured_max_ns"] == 13740
    assert provenance["captured_median_ns"] == 10130.0
    # One inventory row aggregates both calls; the capture does not record
    # whether they are independent trials, so that stays unknown rather than
    # being asserted in either direction.
    assert provenance["aggregated_observation_rows"] == 1
    assert provenance["independent_trials"] is None
    assert plan.stages[0].minimum_duration_ms == pytest.approx(10130.0 / 1e6)


def test_observed_cost_does_not_cross_hardware_scope_or_shape():
    _, binding, config, raw_machine = _inputs()
    provider = ProfiledKernelProvider(config)
    machine = MachineSpec.from_mapping(raw_machine)
    op = OpSpec("cast", "cast", "observed", 0, 0, (), (8192,), (8192,), (8192,), (8192,),
                recipe=binding["recipe"], dtype="int32", layout="ND",
                hbm_bytes=float(8192 * 4 + 8192 * 8), vector_ops=8192.0)
    context = EstimateContext()
    assert provider.estimate(op, replace(machine, hardware_key="Ascend950DT_9582"), context).plan is None
    assert provider.estimate(replace(op, launch_shape=(4096,)), machine, context).plan is None
    assert provider.estimate(op, machine, context).plan is not None
    # The captured scope belongs to the binding, so the run's own measurement
    # protocol neither enables nor disables it.
    unscoped = ProfiledKernelProvider(dict(config, bindings=[
        {key: value for key, value in binding.items() if key != "timing_scope"}
    ]))
    estimate = unscoped.estimate(op, machine, context)
    assert estimate.plan is None
    assert estimate.unsupported_reason == "profile_binding_timing_scope_unsupported"


def test_captured_binding_and_existing_exact_calibration_coexist_in_one_run():
    """A captured binding does not disqualify the run's measured calibration."""
    _, _, config, raw_machine = _inputs()
    machine = MachineSpec.from_mapping(raw_machine)
    provider = create_operator_estimator({"profiled_kernels": config})
    gemm = OpSpec("gemm", "gemm", "prefill", 0, 0, (), (4096, 7168, 2048), (4096, 7168, 2048),
                  (4096, 7168, 2048), (4096, 7168, 2048), recipe="npu_matmul",
                  dtype="bf16", flops=2.0 * 4096 * 7168 * 2048)
    bare = PredictiveCostModel(machine, timing_scope="device_event_warm")
    observation = CalibrationObservation(
        op_family="gemm", phase="prefill", recipe="npu_matmul",
        shape=tuple(sorted(bare.shape_features(gemm).items())),
        duration_ms=0.420, analytic_ms=0.350, source_class="measured",
        source="a3-campaign", machine_key=bare.machine_key(),
        features=tuple(sorted(bare.calibration_features().items())),
        timing_scope="device_event_warm",
    )
    model = PredictiveCostModel(
        machine, calibration=CalibrationStore([observation]),
        timing_scope="device_event_warm", op_estimator=provider,
    )
    captured = model.cost(OpSpec(
        "captured-cast", "cast", "observed", 0, 0, (), (8192,), (8192,), (8192,), (8192,),
        recipe=json.loads((FIXTURE / "source.json").read_text())["signature"]["name"],
        dtype="int32", layout="ND", hbm_bytes=float(8192 * 4 + 8192 * 8), vector_ops=8192.0,
        metadata=(("profile_variant", json.loads((FIXTURE / "source.json").read_text())["variant_id"]),),
    ))
    assert captured.operator_estimation["status"] == "observed_plan"
    assert captured.calibration.status == "profiled_kernel_observed"
    calibrated = model.cost(gemm)
    assert calibrated.calibration.status == "calibrated_exact"
    assert calibrated.calibration.observations_used == 1
    assert calibrated.operator_estimation["reason"] == "existing_calibration_precedence"


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
            "operators": [_cast_operator()],
        })


def test_captured_operator_without_declared_work_is_refused():
    """A capture records duration, not occupancy: there is nothing to schedule."""
    _, binding, config, raw_machine = _inputs()
    with pytest.raises(A3InputError, match="operator_baseline_work_required"):
        operator_plan_report({
            "machine": raw_machine, "estimators": {"profiled_kernels": config},
            "operators": [_cast_operator(hbm_bytes=0.0, vector_ops=0.0)],
        })
    provider = ProfiledKernelProvider(config)
    bare = OpSpec("cast", "cast", "observed", 0, 0, (), (8192,), (8192,), (8192,), (8192,),
                  recipe=binding["recipe"], dtype="int32", layout="ND")
    estimate = provider.estimate(bare, MachineSpec.from_mapping(raw_machine), EstimateContext())
    assert estimate.plan is None
    assert estimate.unsupported_reason == "profile_occupancy_undeclared_for_shared_stage"


@pytest.mark.parametrize("contradiction", [
    {"witness_only": True},
    {"serial_ms": 1.0},
])
def test_explicit_capture_contradicting_the_same_declaration_is_refused(contradiction):
    """Non-executing or explicitly priced operators cannot also be captured."""
    source, _, config, machine = _inputs()
    metadata = {"profile_variant": source["variant_id"], **contradiction}
    with pytest.raises(A3InputError, match="profile_variant_contradicts_declaration"):
        operator_plan_report({
            "machine": machine, "estimators": {"profiled_kernels": config},
            "operators": [_cast_operator(metadata=metadata)],
        })


def test_fused_communication_capture_is_superseded_visibly_not_dropped():
    """The topology model keeps the price; the request stays on the record."""
    source, binding, config, raw_machine = _inputs()
    provider = create_operator_estimator({"profiled_kernels": config})
    model = PredictiveCostModel(
        MachineSpec.from_mapping(raw_machine), timing_scope="device_event_warm",
        op_estimator=provider,
    )
    dispatch = OpSpec(
        "dispatch", "moe_comm_fused", "decode", 0, 0, (), (8192,), (8192,), (8192,), (8192,),
        recipe=binding["recipe"], dtype="bf16", layout="ND", link_bytes=4.0e6,
        metadata=(("profile_variant", source["variant_id"]),),
    )
    costed = model.cost(dispatch)
    assert costed.operator_estimation["status"] == "preserved"
    assert costed.operator_estimation["reason"] == "communication_owned_by_existing_model"
    assert costed.operator_estimation["superseded_profile_variant"] == source["variant_id"]
    assert costed.estimated_ms > 0
