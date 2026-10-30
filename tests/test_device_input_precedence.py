"""Explicit configuration changes must reach actual physical service."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import operator_costs, study
from mindie_sim.schema.hardware import Hardware
from test_graph_analysis import op
from test_mechanism_contract import hardware_case


def ascend_case(device):
    raw = hardware_case()
    raw["machine"] = {
        "backend": "ascend", "sku": "Ascend950DT_9582", "device": device,
    }
    return raw


def memory_bound_case(device):
    raw = ascend_case(device)
    # Weight service must exceed the independent device invocation minimum.
    # The original 8-wide toy is entirely floor-bound on this real A5 profile.
    raw["model"].update(hidden_size=2048, num_heads=16, num_kv_heads=4,
                        head_dim=128, intermediate_size=4096)
    return raw


def test_neutral_memory_override_wins_over_native_alias_in_real_study():
    raw = memory_bound_case({"hbm_bw_bytes_per_s": 4e12})
    original = deepcopy(raw)
    result = study([
        {"id": "base", "case": raw},
        {"id": "slower", "case": raw, "overrides": {
            "machine": {"device": {"rates": {"hbm": 2e12}}}}},
    ], [{"id": "decode", "phase": "decode", "tokens": 1, "context": 16}])
    base, slower = result["evaluations"]
    assert result["status"] == "completed"
    assert base["operators"]["work"] == slower["operators"]["work"]
    assert slower["duration_ms"] > base["duration_ms"]
    assert raw == original
    memory = op("read", memory=4e9)
    first = operator_costs(raw, (memory,))["operators"][0]
    raw["machine"]["device"]["rates"] = {"hbm": 2e12}
    second = operator_costs(raw, (memory,))["operators"][0]
    assert first["serviced"] == second["serviced"] == {"memory@r0:s0": 4e9}
    assert (first["duration_ms"], second["duration_ms"]) == (1, 2)


def test_study_canonicalizes_each_hardware_override_before_recursive_merge():
    raw = memory_bound_case({"rates": {"hbm": 4e12}})
    result = study([
        {"id": "base", "case": raw},
        *({"id": name, "case": raw,
           "overrides": {"machine": {"device": declaration}}}
          for name, declaration in (
              ("neutral", {"rates": {"hbm": 2e12}}),
              ("native", {"hbm_bw_bytes_per_s": 2e12}),
              ("alias", {"rates": {"memory": 2e12}}),
          )),
    ], [{"id": "decode", "phase": "decode", "tokens": 1, "context": 16}])
    assert result["status"] == "completed"
    base, *overrides = result["evaluations"]
    assert all(row["duration_ms"] > base["duration_ms"] for row in overrides)
    assert all(row["duration_ms"] == overrides[0]["duration_ms"] for row in overrides)
    assert all(row["operators"]["work"] == base["operators"]["work"] for row in overrides)


def test_native_compute_transfer_and_capacity_override_profile_at_the_boundary():
    raw = ascend_case({
        "cube_flops_per_s": 1e12, "mte1_bw_bytes_per_s": 2e12,
        "aic_count": 8, "l1_bytes": 65536,
    })
    matrix = op("compute", flops=1e9)
    transfer = replace(op("inner"), metadata=(("resource_work", {"mte1": 4e9}),))
    assert operator_costs(raw, (matrix,))["operator_sum_ms"] == 1
    assert operator_costs(raw, (transfer,))["operator_sum_ms"] == 2
    machine = Hardware.from_mapping(raw["machine"])
    assert machine.dies[0].compute_units == 8
    assert dict(machine.dies[0].buffers)["l1"] == 65536
    assert Hardware.from_mapping(machine.to_dict()) == machine


def test_per_device_sparse_native_and_neutral_overrides_preserve_other_resources():
    raw = ascend_case({
        "rates": {"hbm": 4e12}, "compute_peaks": {"bf16": 4e12},
        "memory_bytes": 8 * 1024**3,
    })["machine"]
    raw.update(num_devices=2, devices=[
        {"hbm_bw_bytes_per_s": 1e12, "cube_flops_per_s": 1e12,
         "hbm_gib": 4, "aic_count": 8},
        {"rates": {"memory": 2e12, "hbm": 3e12},
         "compute_peaks": {"bf16": 2e12}, "buffers": {"l1": 65536}},
    ])
    machine = Hardware.from_mapping(raw)
    a, b = machine.dies
    assert [dict(d.rates)["hbm"] for d in machine.dies] == [1e12, 3e12]
    assert [d.peak("bf16") for d in machine.dies] == [1e12, 2e12]
    assert a.memory_bytes == 4 * 1024**3 and b.memory_bytes == 8 * 1024**3
    assert a.compute_units == 8 and b.compute_units == 32
    assert dict(a.rates)["vector"] == dict(b.rates)["vector"]
    assert dict(a.buffers)["l0c"] == dict(b.buffers)["l0c"]
    assert Hardware.from_mapping(machine.to_dict()) == machine
    local_case = hardware_case()
    local_case["machine"] = machine.to_dict()
    local_case["runtime"] = {"dp": 2}
    first = operator_costs(local_case, (op("read", memory=3e9),), replica=0)
    second = operator_costs(local_case, (op("read", memory=3e9),), replica=1)
    assert first["operator_sum_ms"] == pytest.approx(3)
    assert second["operator_sum_ms"] == pytest.approx(1)


def test_machine_memory_default_controls_capacity_unless_device_overrides_it():
    raw = ascend_case({})["machine"]
    raw.update(hbm_gib=7, num_devices=2, devices=[{}, {"hbm_gib": 3}])
    machine = Hardware.from_mapping(raw)
    assert [device.memory_bytes for device in machine.dies] == [7 * 1024**3, 3 * 1024**3]
    raw["device"]["memory_bytes"] = 5 * 1024**3
    machine = Hardware.from_mapping(raw)
    assert [device.memory_bytes for device in machine.dies] == [5 * 1024**3, 3 * 1024**3]
