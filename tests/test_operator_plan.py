"""Independent arithmetic and integration checks; no hardware measurements."""
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_sim.a3_cost import CalibrationResult, PredictiveCostModel
from mindie_sim.a3_ops import OpSpec
from mindie_sim.a3_resources import ResourceEngine, add_costed_ops, build_resource_specs
from mindie_sim.a3_resource_engine import ResourceSpec
from mindie_sim.a3_types import A3InputError, MachineSpec
from mindie_sim.operator_plan import (
    CompositeEstimator, EstimateContext, ExecutionStage, KernelEstimate, KernelPlan,
    evaluate_kernel_plan, register_kernel_plan,
)
from mindie_sim.operator_estimation import _operator


def machine():
    return MachineSpec.from_mapping({
        "name": "arithmetic", "arch_family": "A5", "num_devices": 1,
        "dies_per_package": 1,
        "device": {"hbm_gib": 1, "cube_flops_per_s": 1000,
                   "vector_ops_per_s": 1000, "hbm_bw_bytes_per_s": 1000,
                   "mte_bw_bytes_per_s": 1000, "launch_us": 1000},
    })


def operator(name="op", deps=()):
    return OpSpec(name, "gemm", "prefill", 0, 0, (), (1, 1, 1), (1, 1, 1),
                  (1, 1, 1), (1, 1, 1), flops=2, deps=deps, dtype="bf16")


def plan(stages=None, includes_dispatch=False):
    return KernelPlan("arithmetic", "1", stages or (
        ExecutionStage("a", (("cube", 2.0),)),
        ExecutionStage("b", (("vector", 3.0),), ("a",)),
    ), {"source": "synthetic_fixture"}, includes_dispatch=includes_dispatch)


class Provider:
    def __init__(self, value=None):
        self.value = value or plan()

    def estimate(self, op, machine, context):
        return KernelEstimate(self.value)


def test_finite_stage_floor_is_concurrent_and_all_leaves_complete():
    value = plan((
        ExecutionStage("a", (("hbm", 2.0),), minimum_duration_ms=5.0),
        ExecutionStage("independent", (("cube", 7.0),)),
        ExecutionStage("b", (("vector", 3.0),), ("a",)),
    ))
    result = evaluate_kernel_plan(value, machine(), EstimateContext(), launch_ms=1)
    # dispatch 1 + max(independent 7, max(hbm2,floor5)+vector3) = 9.
    assert result["elapsed_ms"] == pytest.approx(9)
    assert value.resource_totals() == {"hbm": 2, "cube": 7, "vector": 3}


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_invalid_physical_demands_rejected(value):
    with pytest.raises(A3InputError):
        ExecutionStage("bad", (("cube", value),))


def test_unknown_units_duplicate_names_and_forward_dependencies_rejected():
    with pytest.raises(A3InputError):
        ExecutionStage("bad", (("teraflops_per_second", 1),))
    with pytest.raises(A3InputError):
        plan((ExecutionStage("a"), ExecutionStage("a")))
    with pytest.raises(A3InputError):
        plan((ExecutionStage("a", deps=("future",)), ExecutionStage("future")))
    with pytest.raises(A3InputError):
        KernelEstimate()


def test_missing_or_invalid_fallback_work_is_not_a_launch_only_success():
    raw = {"op_id": "unsupported", "family": "gemm", "launch_shape": [1024, 1024, 1024]}
    with pytest.raises(A3InputError, match="baseline_work_required"):
        _operator(raw)
    for value in (-1, float("nan"), float("inf"), True):
        with pytest.raises(A3InputError, match="operator_work_invalid"):
            _operator({**raw, "flops": value})


def test_external_dependency_release_and_dispatch_are_preserved():
    specs, maps = build_resource_specs(machine())
    engine = ResourceEngine(specs)
    engine.add_task("producer", {maps[0]["vector"]: 4})
    register_kernel_plan(engine, plan(), maps[0], task_id="op",
                         dependencies=("producer",), start_ms=7, launch_ms=1)
    result = engine.run()
    assert result["completed_at_ms"]["op"] == pytest.approx(13)


def test_rate_cap_retains_actual_resource_pressure_while_other_tasks_run():
    engine = ResourceEngine({"cube": ResourceSpec("cube", 100)})
    value = plan((ExecutionStage("limited", (("cube", 100.0),),
                                 minimum_duration_ms=10,
                                 rate_limits=(("cube", 10.0),)),))
    register_kernel_plan(engine, value, {"cube": "cube"}, task_id="a")
    engine.add_task("b", {"cube": 900}, release_ms=1)
    result = engine.run()
    assert result["completed_at_ms"]["a"] == pytest.approx(10)
    # a occupies 10 units/ms until t10: b serves 810 units from t1..10,
    # then its final 90 units use the released 100 units/ms capacity.
    assert result["completed_at_ms"]["b"] == pytest.approx(10.9)
    assert result["serviced"]["cube"] == pytest.approx(1000)


def test_rate_caps_require_positive_existing_resource_work():
    for rates in ((("missing", 1),), (("cube", 0),), (("cube", True),)):
        with pytest.raises(A3InputError):
            ExecutionStage("bad", (("cube", 1),), rate_limits=rates)


def test_parent_work_and_dispatch_charged_once_and_default_unchanged():
    m = machine()
    base = PredictiveCostModel(m).cost(operator())
    assert "execution_plan" not in base.to_dict()
    model = PredictiveCostModel(m, op_estimator=Provider())
    cost = model.cost(operator())
    assert cost.estimated_ms == pytest.approx(6)
    assert cost.analytic_ms == base.analytic_ms
    assert cost.operator_estimation["baseline_ms"] == base.estimated_ms
    specs, maps = build_resource_specs(m)
    engine = ResourceEngine(specs)
    add_costed_ops(engine, [cost], maps, prefix="x", start_ms=0)
    result = engine.run()
    assert result["completed_at_ms"]["x:op"] == pytest.approx(6)
    assert result["serviced"][maps[0]["cube"]] == pytest.approx(2)
    assert result["serviced"][maps[0]["launch"]] == pytest.approx(1)


def test_whole_kernel_including_dispatch_does_not_add_launch():
    value = plan((ExecutionStage("whole", minimum_duration_ms=6),), includes_dispatch=True)
    cost = PredictiveCostModel(machine(), op_estimator=Provider(value)).cost(operator())
    assert cost.estimated_ms == 6
    assert cost.launch_ms == 0


def test_fallback_strict_and_existing_calibration_priority():
    class Unsupported:
        def estimate(self, *args):
            return KernelEstimate(unsupported_reason="dtype_not_supported")

    m = machine()
    base = PredictiveCostModel(m).cost(operator())
    fallback = PredictiveCostModel(m, op_estimator=CompositeEstimator((Unsupported(),))).cost(operator())
    assert fallback.estimated_ms == base.estimated_ms
    assert fallback.operator_estimation["status"] == "fallback"
    with pytest.raises(A3InputError, match="dtype_not_supported"):
        PredictiveCostModel(m, op_estimator=CompositeEstimator((Unsupported(),), strict=True)).cost(operator())

    class Measured:
        def factor_for(self, **kwargs):
            return CalibrationResult("calibrated_exact", 2, 1, True, False, False, "fixture")

    class MustNotRun:
        def estimate(self, *args):
            raise AssertionError("calibrated whole kernel must win")

    result = PredictiveCostModel(m, calibration=Measured(), op_estimator=MustNotRun()).cost(operator())
    assert result.execution_plan is None
    assert result.operator_estimation["reason"] == "existing_calibration_precedence"


def test_existing_fusion_stage_plan_keeps_precedence():
    op = replace(operator(), metadata=(("fusion_stages", ({"flops": 2},)),))
    cost = PredictiveCostModel(machine(), op_estimator=Provider()).cost(op)
    assert cost.execution_plan is None
    assert cost.operator_estimation["reason"] == "existing_stage_plan_precedence"


def test_explicit_serial_cost_is_preserved():
    op = replace(operator(), metadata=(("serial_ms", 7),))
    cost = PredictiveCostModel(machine(), op_estimator=Provider()).cost(op)
    assert cost.execution_plan is None
    assert cost.operator_estimation["reason"] == "explicit_serial_cost_precedence"


def test_cli_mixed_vector_cube_dag_and_default_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result_path = tmp_path/"result.json"
    run = subprocess.run([
        sys.executable, "-m", "mindie_sim", "operator-plan", "--input",
        str(root/"examples/operator-plans/mixed.json"), "--output", str(result_path),
    ], cwd=root, text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stdout[-2000:]+run.stderr[-2000:]
    result = json.loads(result_path.read_text())
    assert result["status"] == "modeled"
    assert [op["operator_estimation"]["status"] for op in result["operators"]] == ["analytic_plan"]*3
    trace = {event["task_id"]: event for event in result["execution"]["trace"]}
    assert trace["operators:gemm"]["end_ms"] <= min(
        event["start_ms"] for key, event in trace.items() if key.startswith("operators:post:plan:")
    )
    assert result["elapsed_ms"] == pytest.approx(sum(op["estimated_ms"] for op in result["operators"]))
    run = subprocess.run([sys.executable, "-c",
                          "import mindie_sim,sys; assert 'torch_npu' not in sys.modules"],
                         cwd=root, text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stderr
