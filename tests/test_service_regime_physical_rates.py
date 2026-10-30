"""A joint native call remains conditional on its explicitly fixed supplies."""
from copy import deepcopy
import math

import pytest

from mindie_sim import operator_costs, target_graphs
from mindie_sim.prediction.structure import parse_case
from test_attention_decode import hardware


def attention_case():
    raw = hardware()
    raw["machine"] = {"backend": "ascend", "sku": "Ascend910_9362", "num_devices": 1,
                      "device": {"rates": {"native_call_service": 1e9},
                                 "resource_units": {"native_call_service": 1},
                                 "work_units": {"native_call_service": "busy_ns"}}}
    raw["model"].update(hidden_size=1024, num_heads=4, num_kv_heads=1,
                        head_dim=256, intermediate_size=2048)
    device = parse_case(raw).machine.dies[0]
    rates, peaks = dict(device.rates), dict(device.compute_peaks)
    raw["runtime"]["kernels"] = {
        "recipes": {"flash_attention_decode": {"kind": "staged_gqa_decode"}},
        "service_regimes": [{
            "id": "fixed-device-native-fai",
            "match": {"hardware": "Ascend910_9362", "recipe": "flash_attention_decode",
                      "physical_rates.resources.hbm": rates["hbm"],
                      "physical_rates.resources.mte2_aic": rates["mte2_aic"],
                      "physical_rates.compute_peaks.bf16": peaks["bf16"]},
            "response": {"fixed_us": 1000},
            "execution": {"kind": "joint_native", "resource": "native_call_service",
                          "reference_rate": 1e9, "native_kernel": "DeclaredFAIControl"}}]}
    return raw


def execute(raw):
    # Real staged decode work, kept small enough for an in-memory contract test.
    graphs = target_graphs(raw, scope="body", phase="decode", tokens=4,
                           context=32, sequences=1)
    op = next(op for ops in graphs.values() for op in ops
              if op.recipe == "flash_attention_decode")
    return operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]


@pytest.mark.parametrize("section,role,axis", (
    ("rates", "hbm", "physical_rates.resources.hbm"),
    ("rates", "mte2_aic", "physical_rates.resources.mte2_aic"),
    ("compute_peaks", "bf16", "physical_rates.compute_peaks.bf16"),
))
def test_same_sku_supply_change_restores_the_complete_analytical_program(section, role, axis):
    raw = attention_case()
    rule = raw["runtime"]["kernels"]["service_regimes"][0]
    baseline = execute(raw)
    assert baseline["program"] == "JointNativeService" and baseline["duration_ms"] == 1
    assert baseline["analytical_reference_work"]["hbm"] > 0

    changed = deepcopy(raw)
    changed["machine"]["device"].setdefault(section, {})[role] = rule["match"][axis] / 2
    result = execute(changed)
    plain = deepcopy(changed)
    plain["runtime"]["kernels"]["service_regimes"] = []
    analytical = execute(plain)
    assert result["program"] == analytical["program"] == "StagedGQADecode"
    assert result["duration_ms"] == analytical["duration_ms"]
    assert result["serviced"] == analytical["serviced"]
    assert result["declared_service"] == analytical["declared_service"]
    assert result["basis"]["service_regimes"][0]["queries"] == {"physical_identity_mismatch": 1}
    assert not result["service_before_input"] and result["live_tasks"] == 0


def test_service_mode_guard_uses_exact_current_rate_and_unguarded_rules_keep_their_behavior():
    raw = attention_case()
    raw["machine"]["device"]["service_modes"] = {
        "copy_lane": {"resource": "vector", "rate": 1e12}}
    rule = raw["runtime"]["kernels"]["service_regimes"][0]
    rule["match"]["physical_rates.service_modes.copy_lane"] = ["vector", 1e12]
    assert execute(raw)["program"] == "JointNativeService"
    raw["machine"]["device"]["service_modes"]["copy_lane"]["rate"] = math.nextafter(1e12, math.inf)
    assert execute(raw)["program"] == "StagedGQADecode"
    rule["match"] = {key: value for key, value in rule["match"].items()
                     if not key.startswith("physical_rates")}
    raw["machine"]["device"]["rates"]["hbm"] = 1e9
    unchanged = execute(raw)
    assert unchanged["program"] == "JointNativeService" and unchanged["duration_ms"] == 1
