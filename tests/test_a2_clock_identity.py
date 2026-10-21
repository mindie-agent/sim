"""A2/A3 hardware-identity contracts: clock precedence, declared vector capability.

Every fixture used here is a real capture from the standing boards
(``tests/fixtures/mindie_sim/npu-smi-a2-*.txt`` = 173.125.1.2 / 910B4,
``npu-smi-a3-*.txt`` = the A3 910_9362 host).  The tests assert derivations from
those captures and the catalog entries corrected from live evidence; they never
assert that a new measurement happened and they never invent a clock band or a
vector ops/cycle count.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from mindie_sim import hardware as hw
from mindie_sim import hardware_probe as hp
from mindie_sim import thermal as th
from mindie_sim.hardware_common import parse_mhz_range
from mindie_sim.sku_profile import infer_arch_family

FIXTURES = Path(__file__).parent / "fixtures" / "mindie_sim"


def _a2_config() -> hw.PlatformConfig:
    return hw.parse_platform_ini(FIXTURES / "Ascend910B4.ini")


def _a2_snapshot() -> hw.NpuSmiSnapshot:
    return hw.parse_npu_smi_snapshot(
        (FIXTURES / "npu-smi-a2-info.txt").read_text(encoding="utf-8"),
        common_text=(FIXTURES / "npu-smi-a2-common.txt").read_text(encoding="utf-8"),
        board_text=(FIXTURES / "npu-smi-a2-board.txt").read_text(encoding="utf-8"),
    )


def _a3_snapshot() -> hw.NpuSmiSnapshot:
    return hw.parse_npu_smi_snapshot(
        (FIXTURES / "npu-smi-a3-info.txt").read_text(encoding="utf-8"),
        common_text=(FIXTURES / "npu-smi-a3-common.txt").read_text(encoding="utf-8"),
        board_text=(FIXTURES / "npu-smi-a3-board.txt").read_text(encoding="utf-8"),
    )


def test_cann_91_npu_arch_2201_on_an_a2_board_stays_a2() -> None:
    # CANN 9.1.0 tags the A2 Ascend910B4.ini with NpuArch=2201, the same arch id
    # A3 parts declare.  A non-catalog A2 SOC is where the hazard becomes live:
    # no exact-SKU lookup can mask the inference any more.
    assert infer_arch_family("Ascend910B4", npu_arch="2201") == "A2"
    assert infer_arch_family("Ascend910B4-2", npu_arch="2201") == "A2"
    # Genuine A3 evidence still selects A3: a 910_93/910C SKU token or dual die.
    assert infer_arch_family("Ascend910_9362", npu_arch="2201") == "A3"
    assert infer_arch_family("Ascend910C", npu_arch="2201") == "A3"
    assert infer_arch_family("Ascend910B4", npu_arch="2201", dual_die=True) == "A3"
    # The shared arch id alone is never A3 (it used to be, for every 2201 part).
    assert infer_arch_family(None, npu_arch="2201") != "A3"

    config = dataclasses.replace(_a2_config(), soc_version="Ascend910B4-2", npu_arch="2201")
    profile = hw.resolve_device_capability_profile(config=config)
    assert profile.clock_and_topology_spec.arch_family == "A2"
    assert profile.clock_and_topology_spec.dies_per_package == 1
    assert profile.clock_and_topology_spec.npu_arch == "2201"
    assert profile.cube_spec.aic_count_per_die == 20
    assert profile.vector_spec.aiv_count_per_die == 40


def test_clock_precedence_is_override_then_snapshot_then_configured() -> None:
    snapshot = _a2_snapshot()
    profile = hw.resolve_device_capability_profile(soc="Ascend910B4", snapshot=snapshot)
    clock = profile.clock_and_topology_spec
    assert (clock.configured_clock_mhz, clock.runtime_clock_mhz, clock.runtime_current_clock_mhz) == (
        1500.0,
        1650.0,
        800.0,
    )
    assert profile.effective_clock_mhz() == 1650.0
    assert clock.clock_source() == "npu_smi_runtime"
    assert profile.peak_cube_ops_per_s("bf16") == pytest.approx(20 * 8192 * 1.65e9, rel=1e-12)
    declared = next(item for item in profile.discrepancies if item["field"] == "configured_clock_mhz")
    assert declared["authority"] == "npu_smi_runtime"
    assert declared["platform_config_clock_mhz"] == 1500.0
    assert declared["runtime_aicore_freq_mhz"] == 1650.0
    assert declared["observed_current_freq_mhz"] == 800.0

    # An explicit caller override outranks the snapshot, and the declared .ini
    # clock is still reported untouched next to both runtime numbers.
    override = hw.resolve_device_capability_profile(
        soc="Ascend910B4", snapshot=snapshot, clock_override_mhz=1600.0
    )
    assert override.effective_clock_mhz() == 1600.0
    assert override.clock_and_topology_spec.clock_source() == "caller_override"
    assert override.clock_and_topology_spec.configured_clock_mhz == 1500.0
    assert override.clock_and_topology_spec.runtime_clock_mhz == 1650.0
    assert override.peak_cube_ops_per_s("bf16") == pytest.approx(20 * 8192 * 1.6e9, rel=1e-12)
    override_entry = next(item for item in override.discrepancies if item["field"] == "effective_clock_mhz")
    assert override_entry["authority"] == "caller_override"
    assert override_entry["caller_override_clock_mhz"] == 1600.0
    assert override_entry["platform_config_clock_mhz"] == 1500.0
    assert override_entry["runtime_aicore_freq_mhz"] == 1650.0

    # An override that matches the declared clock is not a disagreement at all.
    agreeing = hw.resolve_device_capability_profile(soc="Ascend910B4", clock_override_mhz=1500.0)
    assert agreeing.discrepancies == ()
    assert agreeing.effective_clock_mhz() == 1500.0
    assert agreeing.clock_and_topology_spec.clock_source() == "caller_override"

    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(hw.HardwareError) as caught:
            hw.resolve_device_capability_profile(soc="Ascend910B4", clock_override_mhz=bad)
        assert caught.value.reason == "sku_invalid_clock_override"


def test_clock_without_snapshot_or_override_is_the_configured_one() -> None:
    profile = hw.resolve_device_capability_profile(soc="Ascend910B4")
    clock = profile.clock_and_topology_spec
    assert clock.configured_clock_mhz == 1500.0
    assert clock.runtime_clock_mhz is None
    assert clock.clock_override_mhz is None
    assert profile.effective_clock_mhz() == 1500.0
    assert clock.clock_source() == "platform_config"
    assert profile.peak_cube_ops_per_s("bf16") == pytest.approx(20 * 8192 * 1.5e9, rel=1e-12)
    # No snapshot means no runtime evidence: silence, not a fabricated mismatch.
    assert [item for item in profile.discrepancies if "clock" in str(item["field"])] == []


def test_vector_peak_is_declared_or_absent_never_inferred_from_simd_width() -> None:
    profile = hw.resolve_device_capability_profile(soc="Ascend910B4")
    vector = profile.vector_spec
    # vec_calc_size=128 is a SIMD width in bytes, not an ops/cycle count, and no
    # board source declares an A2/A3 AIV ops/cycle figure (B8).
    assert vector.simd_width_bytes == 128
    assert dict(vector.ops_per_cycle_per_aiv) == {}
    assert vector.ops_per_cycle_per_die("bf16") is None
    assert vector.ops_per_cycle_per_die("fp16") is None
    assert profile.peak_vector_ops_per_s("bf16") is None
    assert profile.to_dict()["vector_spec"]["ops_per_cycle_per_aiv"] == {}
    # The same is true for A3: the fixtures' 256 is a repo constant, not evidence.
    a3 = hw.resolve_device_capability_profile(soc="Ascend910_9362")
    assert a3.vector_spec.ops_per_cycle_per_die("bf16") is None

    # A caller that owns the measurement declares it explicitly; dtype aliases
    # resolve inside that one map and an undeclared dtype never borrows a value.
    declared = hw.resolve_device_capability_profile(
        soc="Ascend910B4",
        sku_profile={"vector_spec": {"ops_per_cycle_per_aiv": {"fp16": 256}}},
    )
    assert declared.vector_spec.ops_per_cycle_per_die("float16") == pytest.approx(40 * 256)
    assert declared.peak_vector_ops_per_s("fp16") == pytest.approx(40 * 256 * 1.5e9, rel=1e-12)
    assert declared.vector_spec.ops_per_cycle_per_die("bf16") is None
    assert declared.peak_vector_ops_per_s("fp8") is None


def test_cube_peak_helpers_refuse_undeclared_dtypes() -> None:
    profile = hw.resolve_device_capability_profile(soc="Ascend910B4", snapshot=_a2_snapshot())
    assert profile.peak_cube_ops_per_s("fp8") is None
    assert profile.peak_cube_ops_per_s("bf16", clock_mhz=1500.0) == pytest.approx(20 * 8192 * 1.5e9)
    assert profile.peak_cube_ops_per_s("bf16") == pytest.approx(20 * 8192 * 1.65e9, rel=1e-12)


def test_sustained_clock_band_stays_unmeasured_until_a_caller_supplies_one() -> None:
    for profile in (
        hw.resolve_device_capability_profile(soc="Ascend910B4"),
        hw.resolve_device_capability_profile(soc="Ascend910_9362"),
    ):
        clock = profile.clock_and_topology_spec
        assert clock.sustained_clock_mhz_range == ()
        assert clock.sustained_clock_source is None
        assert profile.to_dict()["clock_and_topology_spec"]["sustained_clock_mhz_range"] == []

    supplied = hw.resolve_device_capability_profile(
        soc="Ascend910B4", sustained_mhz_range=(1200.0, 1600.0)
    )
    assert supplied.clock_and_topology_spec.sustained_clock_mhz_range == (1200.0, 1600.0)
    assert supplied.clock_and_topology_spec.sustained_clock_source == "caller_supplied"

    # hardware_report forwards no band by default and surfaces the clock
    # precedence result; an explicit band is the only way to populate it.
    report = hw.hardware_report(soc="Ascend910B4", snapshot=_a2_snapshot())
    assert report["effective_clock_mhz"] == 1650.0
    assert report["effective_clock_source"] == "npu_smi_runtime"
    reported = report["device_capability_profile"]["clock_and_topology_spec"]
    assert reported["sustained_clock_mhz_range"] == []
    assert reported["sustained_clock_source"] is None
    measured = hw.hardware_report(soc="Ascend910B4", sustained_mhz_range=(1200.0, 1600.0))
    measured_clock = measured["device_capability_profile"]["clock_and_topology_spec"]
    assert measured_clock["sustained_clock_mhz_range"] == [1200.0, 1600.0]
    assert measured_clock["sustained_clock_source"] == "caller_supplied"

    # The thermal report keeps the same convention: no band -> no sustained
    # reference times, and the frequency table is the short-burst band only.
    thermal = th.thermal_bounds(cube_reference_cycles=1000.0)
    assert thermal["status"] == "partial_reference_only"
    assert thermal["sustained_mhz_range"] == []
    assert thermal["sustained_reference_us"] is None
    assert thermal["short_burst_reference_us"] == [
        pytest.approx(1000.0 / 1640.0),
        pytest.approx(1000.0 / 1650.0),
    ]
    assert thermal["frequencies_mhz"] == [1640.0, 1650.0]
    measured_thermal = th.thermal_bounds(
        cube_reference_cycles=1000.0, sustained_mhz_range=(1200.0, 1600.0)
    )
    assert measured_thermal["sustained_mhz_range"] == [1200.0, 1600.0]
    assert measured_thermal["sustained_reference_us"] == [
        pytest.approx(1000.0 / 1200.0),
        pytest.approx(1000.0 / 1600.0),
    ]

    # The cross-check only claims "the configured clock is not sustained" when a
    # band actually exists.
    quiet = hw.cross_check_capacities(_a2_config(), _a2_snapshot(), simulator_ub_kib=None)
    assert "platform_clock_is_not_sustained_clock" not in quiet["flags"]
    flagged = hw.cross_check_capacities(
        _a2_config(), _a2_snapshot(), simulator_ub_kib=None, sustained_mhz_range=(1200.0, 1300.0)
    )
    assert "platform_clock_is_not_sustained_clock" in flagged["flags"]
    # Backward-compatible parsing: ordered pairs and the CLI "lo:hi" spelling.
    assert parse_mhz_range([1600.0, 1200.0]) == (1200.0, 1600.0)
    assert parse_mhz_range("1250:1300") == (1250.0, 1300.0)
    assert parse_mhz_range(None) is None
    assert parse_mhz_range(()) is None
    assert parse_mhz_range("  ") is None
    for bad in ((0.0, 1300.0), (float("nan"), 1300.0), (1.0, 2.0, 3.0), "fast:slow"):
        with pytest.raises(hw.HardwareError) as caught:
            hw.resolve_device_capability_profile(soc="Ascend910B4", sustained_mhz_range=bad)
        assert caught.value.reason == "sku_invalid_sustained_mhz_range"


def test_a3_9362_catalog_reflects_the_live_board_core_counts() -> None:
    # Live evidence, both captured on the standing A3 host (mindie-qwen35hist):
    #   npu-smi -t common  -> Chip Count 2, Aicore Count 20 per chip
    #   torch_npu props    -> cube_core_num=20, vector_core_num=40
    profile = hw.resolve_device_capability_profile(soc="Ascend910_9362")
    assert profile.cube_spec.aic_count_per_die == 20
    assert profile.vector_spec.aiv_count_per_die == 40
    assert profile.clock_and_topology_spec.dies_per_package == 2
    assert profile.clock_and_topology_spec.configured_clock_mhz == 1650.0
    assert profile.cube_spec.ops_per_cycle_per_die("bf16") == 20 * 8192
    # 176160768 B (live CANN l2_size / torch_npu L2_cache_size) = 168 MiB.
    assert profile.memory_hierarchy_spec.l2_capacity_mib == pytest.approx(176160768 / 1024 ** 2)
    assert profile.memory_hierarchy_spec.l2_capacity_mib == 168.0
    assert profile.to_dict()["memory_hierarchy_spec"]["l2_capacity_mib"] == 168.0

    # The 1200 GB/s HBM peak is deliberately unchanged and labelled unverified:
    # live memory-bound GEMMs imply 1.45-1.64 TB/s while their weights fit in L2.
    memory = profile.memory_hierarchy_spec
    assert memory.hbm_peak_bw_gbps == 1200.0
    assert memory.hbm_peak_bw_verified is False
    assert memory.sustained_hbm_bw_gbps is None
    assert profile.to_dict()["memory_hierarchy_spec"]["hbm_peak_bw_verified"] is False

    a2 = hw.resolve_device_capability_profile(soc="Ascend910B4")
    assert a2.memory_hierarchy_spec.hbm_peak_bw_gbps == 800.0
    assert a2.memory_hierarchy_spec.hbm_peak_bw_verified is False

    # A measured sustained figure can be carried separately (arbitrary test
    # input, not a board claim) without touching the declared peak or its flag.
    measured = hw.resolve_device_capability_profile(
        soc="Ascend910_9362",
        sku_profile={"memory_hierarchy_spec": {"sustained_hbm_bw_gbps": 1234.0}},
    )
    assert measured.memory_hierarchy_spec.sustained_hbm_bw_gbps == 1234.0
    assert measured.memory_hierarchy_spec.hbm_peak_bw_gbps == 1200.0
    assert measured.memory_hierarchy_spec.hbm_peak_bw_verified is False


def test_a5_profile_is_unchanged_by_the_a2_identity_work() -> None:
    a5 = hw.resolve_device_capability_profile(soc="Ascend950DT_9581")
    assert a5.clock_and_topology_spec.arch_family == "A5"
    assert a5.cube_spec.aic_count_per_die == 32
    assert a5.vector_spec.aiv_count_per_die == 64
    assert a5.vector_spec.simd_width_bytes == 256
    assert a5.clock_and_topology_spec.configured_clock_mhz == 1650.0
    assert a5.effective_clock_mhz() == 1650.0
    assert a5.memory_hierarchy_spec.hbm_peak_bw_gbps == 4000.0
    # No A5 entry claims HBM verification or a declared AIV ops/cycle figure.
    assert a5.memory_hierarchy_spec.hbm_peak_bw_verified is None
    assert dict(a5.vector_spec.ops_per_cycle_per_aiv) == {}
    assert [item for item in a5.discrepancies if "clock" in str(item["field"])] == []


def test_probe_keeps_rated_and_dvfs_clocks_separate() -> None:
    a2 = _a2_snapshot()
    assert a2.rated_aicore_freq_mhz == 1650.0
    assert a2.observed_aicore_freq_mhz == 800.0
    payload = a2.to_dict()
    assert payload["rated_aicore_freq_mhz"] == 1650.0
    assert payload["observed_aicore_freq_mhz"] == 800.0

    a3 = _a3_snapshot()
    assert len(a3.chips) == 2
    assert a3.rated_aicore_freq_mhz == 1650.0
    assert a3.observed_aicore_freq_mhz == 800.0

    # An info-only capture has no `-t common` block: no clock is fabricated.
    info_only = hp.load_npu_smi_snapshot(FIXTURES / "npu-smi-a2-info.txt")
    assert info_only.chips == ()
    assert info_only.rated_aicore_freq_mhz is None
    assert info_only.observed_aicore_freq_mhz is None
