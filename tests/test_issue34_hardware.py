"""Pure hardware-SKU derivation, ownership scaling and efficiency curves."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.hardware_derivation import derive_hardware_sku
from mindie_sim.hardware_efficiency import (
    EfficiencyCurve, apply_mte2_ceiling, coerce_curve,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from issue34_helpers import hardware


def a5():
    return Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950DT_9582",
                                  "num_devices": 1})


def test_aic_scaling_touches_only_owned_supplies_and_units():
    base = a5()
    base_die = base.dies[0]
    derived = derive_hardware_sku(base, sku="X16", aic_count=16,
                                  memory_bytes=64 * 2 ** 30)
    die = derived.dies[0]
    assert derived.sku == "X16" and die.compute_units == 16
    assert dict(die.compute_peaks)["bf16"] == pytest.approx(
        dict(base_die.compute_peaks)["bf16"] / 2)
    assert dict(die.compute_peaks)["fp8_e4m3"] == pytest.approx(
        dict(base_die.compute_peaks)["fp8_e4m3"] / 2)
    assert dict(die.resource_units)["vector"] == 32
    assert dict(die.buffer_units)["l1"] == 16
    assert dict(die.buffers) == dict(base_die.buffers)
    assert dict(die.rates)["hbm"] == pytest.approx(dict(base_die.rates)["hbm"])
    assert dict(die.rates)["vector"] == pytest.approx(
        dict(base_die.rates)["vector"] / 2)
    assert die.memory_bytes == 64 * 2 ** 30
    assert [(link.name, link.capacity_bytes_per_ms) for link in derived.links] == [
        (link.name, link.capacity_bytes_per_ms) for link in base.links]


def test_aiv_override_and_roundtrip_idempotence():
    derived = derive_hardware_sku(a5(), sku="X", aic_count=16, aiv_count=16)
    die = derived.dies[0]
    assert die.compute_units == 16 and dict(die.resource_units)["vector"] == 16
    assert dict(die.buffer_units)["ub"] == 16
    parsed = Hardware.from_mapping(derived.to_dict())
    assert parsed.dies[0].compute_peaks == die.compute_peaks
    assert parsed.dies[0].resource_units == die.resource_units
    noop = derive_hardware_sku(derived, sku="X")
    assert noop.dies[0].compute_units == die.compute_units
    assert noop.dies[0].compute_peaks == die.compute_peaks
    assert noop.dies[0].rates == die.rates


def test_hbm_peak_and_curve_interpolate_then_clamp():
    curve = EfficiencyCurve("aic_count", ((8, 0.5), (32, 1.0)))
    derived = derive_hardware_sku(
        a5(), sku="X", aic_count=16, hbm_peak_bytes_per_s=2e12,
        hbm_efficiency=curve)
    rate = dict(derived.dies[0].rates)["hbm"]
    assert curve.resolve(16) == pytest.approx(2 / 3)
    assert rate == pytest.approx(2e12 * 2 / 3)
    assert curve.resolve(1) == 0.5 and curve.resolve(64) == 1.0
    with pytest.raises(ModelingError, match="efficiency_curve_knots_not_increasing"):
        EfficiencyCurve("aic_count", ((8, 0.5), (8, 0.6)))
    with pytest.raises(ModelingError, match="efficiency_curve_knot_y_invalid"):
        EfficiencyCurve("aic_count", ((8, 0.0),))
    with pytest.raises(ModelingError, match="efficiency_curve_axis_unsupported"):
        EfficiencyCurve("tokens", ((8, 0.5),))
    assert coerce_curve(0.8).resolve(1) == 0.8


def test_mte2_ceiling_is_a_shared_coupling_cap():
    capped = apply_mte2_ceiling({"mte2_aic": 80.0, "mte2_aiv": 40.0}, 60.0)
    assert capped["mte2_aic"] + capped["mte2_aiv"] == pytest.approx(60.0)
    assert capped["mte2_aic"] == pytest.approx(40.0)
    assert capped["mte2_aiv"] == pytest.approx(20.0)
    derived = derive_hardware_sku(
        a5(), sku="X", aic_count=16, mte2_ceiling_bytes_per_s=1e12)
    rates = dict(derived.dies[0].rates)
    assert rates["mte2_aic"] + rates["mte2_aiv"] <= 1e12 + 1e-3


def test_a5_derivative_supply_alias_and_heterogeneous_core_rejection():
    derived = derive_hardware_sku(a5(), sku="A5X", aic_count=16)
    aliases = dict(derived.dies[0].compute_peak_aliases)
    assert aliases["mxfp4"] == "fp8_e4m3"
    assert derived.dies[0].peak("mxfp4") == pytest.approx(
        derived.dies[0].peak("fp8_e4m3"))
    heterogeneous = replace(hardware(2), dies=(
        replace(hardware(2).dies[0], rank=0),
        replace(hardware(2).dies[1], rank=1, compute_units=8,
                resource_units=(("vector", 8),))))
    with pytest.raises(ModelingError, match="requires_per_device_core_mapping"):
        derive_hardware_sku(heterogeneous, sku="H", aic_count=4)
