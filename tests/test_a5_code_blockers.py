"""Regression tests for the verified A5 code blockers (B-1/P5, P6, B-4, P15, B-5).

Every assertion here is an A5 capability declaration or a priced kernel, not a
fixture-shaped constant check:

* ``B-1/P5`` - an A5 fp8 GEMM must be priced from the declared 2x bf16 peak.
* ``P6``    - ``compute_peaks`` must come from the SKU profile (or a caller map).
* ``B-4``   - every machine builder must derive one vector peak (2.70336e13).
* ``P15``   - a campaign that names an A5 SKU must get the A5 machine, or fail.
* ``B-5``   - an exact-but-unregistered SKU must say so precisely.
* ``M``     - ``MachineLimits`` must accept the four declared A5 blocks and
              refuse a block whose declared peak contradicts its components.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_sim.a3_cost import PredictiveCostModel
from mindie_sim.a3_operator_campaign import CampaignConfig
from mindie_sim.a3_ops import OpSpec
from mindie_sim.a3_physical_calibration import (
    MachineLimits, PhysicalCalibrationError,
)
from mindie_sim.a3_real_data import machine_from_environment
from mindie_sim.a3_types import A3InputError, MachineSpec
from mindie_sim.sku_profile import (
    AMBIGUOUS_SKU_REASON, resolve_device_capability_profile,
)

FIXTURES = Path(__file__).parent / "fixtures" / "mindie_sim"
A5_SKUS = ("Ascend950DT_9582", "Ascend950DT_9581", "Ascend950PR_9579")
#: Declared A5 cube capability: 32 AIC x 8192 (bf16) / 16384 (fp8) x 1.65 GHz.
A5_BF16_PEAK = 32 * 8192 * 1.65e9
A5_FP8_PEAK = 32 * 16384 * 1.65e9
#: A5 vector engine: 64 AIV x 256 element-ops/cycle x 1.65 GHz.
A5_VECTOR_OPS = 64 * 256 * 1.65e9


def _gemm_5120(dtype: str = "fp8_e4m3") -> OpSpec:
    width = 5120
    return OpSpec(
        op_id="gemm.5120", family="gemm", phase="prefill", stage=0, layer=0,
        requests=(), semantic_shape=(width, width, width),
        local_shape=(width, width, width), padded_shape=(width, width, width),
        launch_shape=(width, width, width), flops=2.0 * width ** 3,
        hbm_bytes=3.0 * width ** 2 * 1.0, dtype=dtype,
        layout="contiguous", recipe="torch_mm_preallocated",
    )


def _price(machine: MachineSpec, dtype: str) -> float:
    return PredictiveCostModel(machine, compute_dtype=dtype).cost(
        _gemm_5120(dtype)
    ).analytic_ms


def _a5_blocks() -> list[dict]:
    document = json.loads((FIXTURES / "a5-machine-blocks.json").read_text())
    return document["blocks"]


# ---------------------------------------------------------------------------
# B-1/P5 - fp8 peak is the declared peak, never the bf16 base
# ---------------------------------------------------------------------------


def test_a5_fp8_peak_is_two_bf16_and_prices_5120_gemm_at_03303_ms():
    machine = MachineSpec.from_mapping({"sku": "Ascend950DT_9581"})
    peaks = dict(machine.compute_peaks)
    assert machine.peak_flops_for("fp8_e4m3") == pytest.approx(A5_FP8_PEAK)
    assert peaks["fp8_e4m3"] == pytest.approx(2.0 * peaks["bf16"])
    # The blocked number: 2*5120^3 FLOPs at 8.650752e14 flops/s = 0.310304 ms,
    # plus the 0.02 ms launch intercept.  Priced at the bf16 peak it is 0.640606
    # (the old silent 2x), which is exactly the bf16 price of the same kernel.
    service_ms = 2.0 * 5120 ** 3 / A5_FP8_PEAK * 1000.0
    assert service_ms == pytest.approx(0.310304, abs=1e-6)
    assert _price(machine, "fp8_e4m3") == pytest.approx(0.330303, abs=1e-6)
    assert _price(machine, "fp8_e4m3") == pytest.approx(service_ms + 0.02, rel=1e-12)
    # bf16 keeps its old price: only the fp8 declaration changed.
    assert _price(machine, "bf16") == pytest.approx(0.640606, abs=1e-6)


def test_undeclared_compute_dtype_is_a_loud_rejection_not_the_bf16_peak():
    # A2 declares bf16/fp16/fp32/int8 cube rates and no fp8 at all.  The old
    # fallback priced fp8 as bf16; now the absent capability is a rejection.
    a2 = MachineSpec.from_mapping({"sku": "Ascend910B4"})
    assert "fp8_e4m3" not in dict(a2.compute_peaks)
    with pytest.raises(A3InputError, match="machine_compute_peak_undeclared:fp8_e4m3"):
        a2.peak_flops_for("fp8_e4m3")
    a5 = MachineSpec.from_mapping({"sku": "Ascend950DT_9581"})
    with pytest.raises(A3InputError, match="machine_compute_peak_undeclared:int4"):
        a5.peak_flops_for("int4")


def test_fp8_spelling_alias_resolves_inside_the_declared_map():
    machine = MachineSpec.from_mapping({
        "name": "declared", "num_devices": 1,
        "device": {
            "hbm_gib": 1, "cube_flops_per_s": 1e14,
            "vector_ops_per_s": 1e13, "hbm_bw_bytes_per_s": 1e12,
            "mte_bw_bytes_per_s": 2e12,
        },
        "compute_peaks": {"fp8": 2e14},
    })
    assert machine.peak_flops_for("fp8_e4m3") == 2e14
    assert machine.peak_flops_for("fp8_e5m2") == 2e14


# ---------------------------------------------------------------------------
# P6 - compute_peaks come from the profile (or an explicit caller override)
# ---------------------------------------------------------------------------


def test_compute_peaks_are_derived_from_the_sku_profile():
    machine = MachineSpec.from_mapping({"sku": "Ascend950DT_9582"})
    peaks = dict(machine.compute_peaks)
    assert peaks["bf16"] == pytest.approx(A5_BF16_PEAK)
    assert peaks["fp8_e4m3"] == pytest.approx(A5_FP8_PEAK)
    assert peaks["fp32"] == pytest.approx(32 * 4096 * 1.65e9)
    assert peaks["int8"] == pytest.approx(A5_FP8_PEAK)
    # The bf16 peak is the same value the per-die cube baseline carries.
    assert peaks["bf16"] == pytest.approx(machine.dies[0].cube_flops_per_s)

    profile = resolve_device_capability_profile(soc="Ascend950DT_9581")
    override = MachineSpec.from_profile(
        profile, compute_peaks={"bf16": 1e14, "fp8_e4m3": 2e14},
    )
    assert override.peak_flops_for("fp8_e4m3") == 2e14
    assert override.peak_flops_for("bf16") == 1e14


def test_a2_a3_declared_peaks_are_unchanged_golden_derivations():
    # A2 catalog clock is 1500 MHz without a runtime snapshot.
    a2 = MachineSpec.from_mapping({"sku": "Ascend910B4"})
    assert a2.dies[0].cube_flops_per_s == pytest.approx(20 * 8192 * 1.5e9)
    assert dict(a2.compute_peaks)["bf16"] == pytest.approx(20 * 8192 * 1.5e9)
    a3 = MachineSpec.from_mapping({"sku": "Ascend910_9362"})
    assert a3.dies[0].cube_flops_per_s == pytest.approx(20 * 8192 * 1.65e9)
    assert dict(a3.compute_peaks)["bf16"] == pytest.approx(20 * 8192 * 1.65e9)


# ---------------------------------------------------------------------------
# B-4 - one vector-peak derivation for every builder
# ---------------------------------------------------------------------------


def test_every_machine_builder_agrees_on_the_a5_vector_peak():
    catalog = MachineSpec.from_mapping({"sku": "Ascend950DT_9582"})
    measurement = machine_from_environment({
        "device_name": "Ascend950DT_9582", "visible_devices": "0,1",
    })
    assert A5_VECTOR_OPS == pytest.approx(2.70336e13)
    assert catalog.dies[0].vector_ops_per_s == pytest.approx(A5_VECTOR_OPS)
    assert measurement.dies[0].vector_ops_per_s == pytest.approx(A5_VECTOR_OPS)
    # The deleted second derivation was exactly 2x; no builder may return it.
    assert catalog.dies[0].vector_ops_per_s == pytest.approx(2.70336e13)
    assert dict(measurement.metadata)["vector_capability_source"].startswith(
        "legacy_measurement_nominal"
    )


def test_a3_builders_and_machine_block_share_one_vector_peak():
    # 40 AIV x 128 element-ops/cycle x 1.65 GHz = 8.448e12, the value the A3
    # machine block declares; the 2x measurement nominal was 1.6896e13.
    catalog = MachineSpec.from_mapping({"sku": "Ascend910_9362"})
    measurement = machine_from_environment({"device_name": "Ascend910_9362"})
    assert catalog.dies[0].vector_ops_per_s == pytest.approx(8.448e12)
    assert measurement.dies[0].vector_ops_per_s == pytest.approx(8.448e12)


# ---------------------------------------------------------------------------
# P15 - the campaign machine follows the SKU, or fails on a contradiction
# ---------------------------------------------------------------------------


def test_campaign_a5_sku_only_declaration_derives_the_a5_machine():
    config = CampaignConfig.from_mapping({"machine": {"sku": "Ascend950DT_9582"}})
    spec = config.machine_spec()
    assert config.to_dict()["machine"]["arch_family"] == "A5"
    assert spec.arch_family == "A5"
    assert spec.hardware_key == "Ascend950DT_9582"
    assert spec.dies_per_package == 1
    assert len(spec.dies) == 128
    assert spec.dies[0].aic_count == 32
    assert spec.dies[0].vector_ops_per_s == pytest.approx(A5_VECTOR_OPS)
    assert spec.dies[0].hbm_bw_bytes_per_s == pytest.approx(4.0e12)
    peaks = dict(spec.compute_peaks)
    assert peaks["bf16"] == pytest.approx(A5_BF16_PEAK)
    assert peaks["fp8_e4m3"] == pytest.approx(A5_FP8_PEAK)
    assert {link.channel for link in spec.links} == {
        "d2d_intra_node", "d2d_inter_node",
    }


def test_campaign_machine_contradicting_the_sku_is_refused():
    with pytest.raises(
        A3InputError, match="campaign_machine_sku_family_conflict:Ascend950DT_9582",
    ):
        CampaignConfig.from_mapping({
            "machine": {"sku": "Ascend950DT_9582", "arch_family": "A3"},
        })
    with pytest.raises(
        A3InputError, match="campaign_machine_sku_peak_conflict:Ascend950DT_9582",
    ):
        CampaignConfig.from_mapping({
            "machine": {
                "sku": "Ascend950DT_9582",
                "device": {"cube_flops_per_s": 2.70336e14},
            },
        })
    with pytest.raises(A3InputError, match="campaign_machine_sku_unresolved"):
        CampaignConfig.from_mapping({"machine": {"sku": "Ascend950DT_9599"}})


def test_campaign_default_machine_stays_the_a3_baseline():
    config = CampaignConfig.from_mapping({})
    spec = config.machine_spec()
    assert spec.arch_family == "A3"
    assert spec.dies_per_package == 2
    assert len(spec.dies) == 128
    assert spec.dies[0].vector_ops_per_s == pytest.approx(1.6896e13)
    assert dict(spec.compute_peaks)["bf16"] == pytest.approx(2.70336e14)
    # The A3 declaration keeps its own fp8==bf16 assumption; only an A5 SKU is
    # re-derived from the A5 profile (that is where the 2x fp8 capability is).
    assert dict(spec.compute_peaks)["fp8_e4m3"] == pytest.approx(2.70336e14)


# ---------------------------------------------------------------------------
# B-5 - an exact-but-unregistered SKU says exactly that
# ---------------------------------------------------------------------------


def test_exact_unregistered_sku_reports_a_precise_reason():
    profile = resolve_device_capability_profile(soc="Ascend950DT_9599")
    assert profile.status == "ambiguous_sku"
    assert profile.unmodeled_reason == "sku_not_registered:Ascend950DT_9599"
    assert profile.resolution_sources == ("unregistered_exact_soc",)
    # No capability was granted: the unresolvable part still cannot be priced.
    assert profile.cube_spec.aic_count_per_die is None
    assert profile.peak_cube_ops_per_s("bf16") is None

    for family in ("A5", "Ascend910", "Ascend910B", "Ascend950"):
        ambiguous = resolve_device_capability_profile(soc=family)
        assert ambiguous.unmodeled_reason == AMBIGUOUS_SKU_REASON, family
        assert ambiguous.resolution_sources == ("ambiguous_family_only",)


# ---------------------------------------------------------------------------
# A5 SKU registration and the four declared machine blocks
# ---------------------------------------------------------------------------


def test_a5_skus_are_registered_with_the_same_declared_capability():
    for sku in A5_SKUS:
        profile = resolve_device_capability_profile(soc=sku)
        assert profile.status != "ambiguous_sku", sku
        assert profile.clock_and_topology_spec.soc_version == sku
        assert profile.clock_and_topology_spec.arch_family == "A5"
        assert profile.cube_spec.aic_count_per_die == 32
        assert profile.vector_spec.aiv_count_per_die == 64
        assert profile.peak_cube_ops_per_s("bf16") == pytest.approx(A5_BF16_PEAK)
        assert profile.peak_cube_ops_per_s("fp8_e4m3") == pytest.approx(A5_FP8_PEAK)


def test_machine_limits_accept_every_declared_a5_machine_block():
    blocks = _a5_blocks()
    assert len(blocks) == 4
    for block in blocks:
        limits = MachineLimits.from_mapping(block)
        assert limits.hardware_key == block["hardware_key"]
        assert limits.arch_family == "A5"
        # The loader re-derives aic x ops x sustained clock; the declared peak
        # must agree exactly (it does for both dtypes of both SKUs).
        assert limits.peak_flops == pytest.approx(block["peak_flops"], rel=1e-12)
        assert limits.vec_peak_ops_per_s == pytest.approx(
            block["vec_peak_ops_per_s"], rel=1e-12
        )
        dtype = "fp8_e4m3" if block["compute_dtype"] == "fp8" else "bf16"
        machine = MachineSpec.from_mapping({"sku": block["hardware_key"]})
        assert machine.peak_flops_for(dtype) == pytest.approx(limits.peak_flops)
    # The fp8 block is the declared 2x bf16 peak, per SKU.
    by_sku: dict[str, dict[str, float]] = {}
    for block in blocks:
        by_sku.setdefault(block["sku"], {})[block["compute_dtype"]] = MachineLimits.from_mapping(
            block
        ).peak_flops
    for sku, peaks in by_sku.items():
        assert peaks["fp8"] == pytest.approx(2.0 * peaks["bf16"]), sku


def test_machine_limits_reject_a_block_whose_declared_peak_contradicts_it():
    fp8 = next(
        block for block in _a5_blocks()
        if block["sku"] == "Ascend950DT_9581" and block["compute_dtype"] == "fp8"
    )
    with pytest.raises(PhysicalCalibrationError, match="peak_flops_contradicts_components"):
        MachineLimits.from_mapping({**fp8, "peak_flops": A5_BF16_PEAK})
    # The vector peak was documentation before: a 2x declaration was silently
    # replaced by the loader instead of being refused (B-4).
    with pytest.raises(PhysicalCalibrationError, match="vec_peak_contradicts_components"):
        MachineLimits.from_mapping({**fp8, "vec_peak_ops_per_s": 2.0 * A5_VECTOR_OPS})
