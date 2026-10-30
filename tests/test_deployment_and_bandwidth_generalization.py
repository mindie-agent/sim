"""Real public execution controls for deployment resource axes and fixed work."""
from copy import deepcopy

import pytest

from mindie_sim import predict
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import IterationWork, OpSpec
from tools.deployment_generalization import compare_sweep, effective_resources, scaled_case
from test_prediction_system import case


def control_case(*, pd=False):
    raw = case(prompt=4, output=3)
    raw["machine"] = {"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 1}
    if pd:
        raw["pools"] = {name: {"machine": deepcopy(raw["machine"]),
                               "runtime": deepcopy(raw["runtime"])}
                        for name in ("prefill", "decode")}
        raw["connector"] = {"bytes_per_s": 128000, "latency_us": 0}
    return raw


def resource_lower(axis):
    def lower(plan, model, runtime, machine):
        shape = (plan.valid_tokens,)
        work = ({"flops": 1e12} if axis == "aic" else
                {"vector_ops": 1e12} if axis == "aiv" else
                {"hbm_bytes": 1e10} if axis == "hbm" else {"flops": 1})
        op = OpSpec("control", "gemm" if axis == "aic" else "vector", "body", 0, 0,
                    tuple(v.request_id for v in plan.requests), shape, shape, shape, shape,
                    dtype="bf16", **work)
        return IterationWork((op,), plan, 1, plan.valid_tokens, plan.valid_tokens,
                             None, "body", "resource_control")
    return lower


@pytest.mark.parametrize("axis", ("aic", "aiv", "hbm", "network"))
@pytest.mark.parametrize("pd", (False, True))
def test_public_serving_axes_preserve_work_and_show_strict_bottleneck_response(axis, pd):
    if axis == "network" and not pd:
        # A one-device colocated case has no inter-node traffic: a plateau is
        # physically required, and must not be faked by changing another rate.
        raw = control_case()
    else:
        raw = control_case(pd=pd)
    reports = [predict(scaled_case(raw, axis, scale), lower=resource_lower(axis))
               for scale in (.5, 1, 2)]
    times = [r["metrics"]["makespan_ms"] for r in reports]
    if axis == "network" and not pd:
        assert times[0] == times[1] == times[2]
        assert compare_sweep(reports)["status"] == "plateau"
    else:
        assert times[0] > times[1] > times[2]
        assert compare_sweep(reports)["status"] == "strict_response"
    for result in reports:
        assert result["requests"][0]["status"] == "served"
        assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
        if pd:
            assert result["execution"]["handoff_bytes"] == 512
        else:
            assert result["execution"]["work"] == reports[1]["execution"]["work"]


@pytest.mark.parametrize("sku", ("Ascend910_9362", "Ascend950DT_9582"))
def test_axes_change_effective_independent_pipes_and_modes_without_cross_axis_leakage(sku):
    raw = control_case()
    raw["machine"] = {"backend": "ascend", "sku": sku, "num_devices": 16, "ranks_per_node": 8}
    original = Hardware.from_mapping(raw["machine"])
    for axis in ("network", "hbm", "aic", "aiv"):
        scaled = scaled_case(raw, axis, .5)
        hardware = Hardware.from_mapping(scaled["machine"])
        first, changed = original.dies[0], hardware.dies[0]
        assert first.memory_bytes == changed.memory_bytes
        assert first.compute_units == changed.compute_units
        assert first.resource_units == changed.resource_units
        modes = dict(changed.service_modes)
        for mode, (resource, rate) in first.service_modes:
            ratio = dict(changed.rates)[resource] / dict(first.rates)[resource]
            assert modes[mode] == (resource, rate * ratio)
        if axis == "network":
            assert first == changed
            for before, after in zip(original.links, hardware.links):
                ratio = .5 if before.channel == "d2d_inter_node" else 1
                assert after.capacity_bytes_per_ms == before.capacity_bytes_per_ms * ratio
                assert after.latency_us == before.latency_us
        elif axis == "aic":
            assert dict(changed.compute_peaks) == {key: value * .5 for key, value in first.compute_peaks}
            assert dict(changed.rates)["vector"] == dict(first.rates)["vector"]
        elif axis == "aiv":
            assert changed.compute_peaks == first.compute_peaks
            assert dict(changed.rates)["hbm"] == dict(first.rates)["hbm"]
        else:
            assert dict(changed.rates)["hbm"] == dict(first.rates)["hbm"] * .5
            assert changed.compute_peaks == first.compute_peaks
    assert effective_resources(raw)["colocated"]["ranks"] == 16


def test_colocated_inter_node_scan_changes_actual_collective_service():
    raw = control_case()
    raw["machine"].update(num_devices=4, ranks_per_node=2)
    raw["runtime"]["tp"] = 4

    def collective(plan, model, runtime, machine):
        shape = (plan.valid_tokens,)
        op = OpSpec("reduce", "allreduce", "body", 0, 0,
                    tuple(v.request_id for v in plan.requests), shape, shape, shape, shape,
                    link_bytes=25e6, metadata=(("group_axis", "tp"),))
        return IterationWork((op,), plan, 1, plan.valid_tokens, plan.valid_tokens,
                             None, "body", "collective_control")

    results = [predict(scaled_case(raw, "network", scale), lower=collective)
               for scale in (.5, 1, 2)]
    assert compare_sweep(results)["status"] == "strict_response"
    for result in results:
        assert result["execution"]["work"] == results[1]["execution"]["work"]
        assert any(value > 0 for name, value in result["execution"]["serviced"].items()
                   if "d2d_inter_node" in name)
        assert result["execution"]["live_tasks"] == 0


def test_full_dense_capacity_planner_uses_real_ep1_geometry_and_keeps_cross_node_pp():
    from mindie_sim.model_catalog import load_model
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.structure import parse_case
    from tools.deployment_generalization import plan_scheme

    plan = plan_scheme("qwen38", "a3", "multi", "colocated")
    parsed = parse_case(plan["case"])
    assert parsed.model.layer_plans == load_model("qwen3.8-27b").layer_plans
    assert parsed.runtime.parallel.ep == 1
    assert parsed.runtime.parallel.pp > 1
    assert parsed.machine.ranks_per_node < parsed.runtime.parallel.world_size
    assert Memory(parsed).resize(0, "r0", 130)
