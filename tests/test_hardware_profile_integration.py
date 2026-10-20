"""Targeted contracts for the integrated hardware profile, thermal and probe.

These tests use the repository's captured CANN/``npu-smi`` fixtures and the
explicit A5 C8 FlashMLA Golden workload fixture.  They do not connect to a live
NPU and do not claim that a new board measurement happened on this machine.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import pytest

from mindie_sim import hardware as hw
from mindie_sim import hardware_probe as hp
from mindie_sim import thermal as th
from mindie_sim.a3_types import _profile_defaults

FIXTURES = Path(__file__).parent / "fixtures" / "mindie_sim"
GOLDEN = th.A5_FLASHMLA_C8_GOLDEN_WORKLOAD


def _a2_config() -> hw.PlatformConfig:
    return hw.parse_platform_ini(FIXTURES / "Ascend910B4.ini")


def _a5_config() -> hw.PlatformConfig:
    return hw.parse_platform_ini(FIXTURES / "Ascend950DT_9581.ini")


def _a2_snapshot() -> hw.NpuSmiSnapshot:
    """The real 910B4 capture: ``Aicore Freq`` 1650 (rated), ``curFreq`` 800."""
    return hw.parse_npu_smi_snapshot(
        (FIXTURES / "npu-smi-a2-info.txt").read_text(encoding="utf-8"),
        common_text=(FIXTURES / "npu-smi-a2-common.txt").read_text(encoding="utf-8"),
        board_text=(FIXTURES / "npu-smi-a2-board.txt").read_text(encoding="utf-8"),
    )


def _a3_capture_dir(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("npu-smi-a3-info.txt", "npu-smi-a3-common.txt", "npu-smi-a3-board.txt"):
        shutil.copy(FIXTURES / name, destination / name)
    return destination


def test_cube_capability_has_one_unit_explicit_source() -> None:
    a2 = hw.resolve_device_capability_profile(config=_a2_config())
    assert a2.cube_spec.ops_per_cycle_per_die("bf16") == 20 * 8192
    assert a2.cube_spec.ops_per_cycle_per_die("fp8") is None
    assert "throughput" not in a2.cube_spec.to_dict()

    a5 = hw.resolve_device_capability_profile(config=_a5_config())
    assert a5.cube_spec.ops_per_cycle_per_die("fp8") == 32 * 16384
    assert a5.cube_spec.ops_per_cycle_per_die("fp8_e4m3") == 32 * 16384
    assert a5.cube_spec.ops_per_cycle_per_die("bf16") == 32 * 8192
    assert a5.cube_spec.ops_per_cycle_per_die("int4") is None


def test_a2_profile_derives_cube_peak_from_the_verified_runtime_clock() -> None:
    # Real 910B4 capture: the .ini declares cube_freq=1500 while npu-smi reports
    # "Aicore Freq" 1650 (rated, reached under load) and "curFreq" 800 (DVFS).
    profile = hw.resolve_device_capability_profile(soc="Ascend910B4", snapshot=_a2_snapshot())
    clock = profile.clock_and_topology_spec
    assert clock.configured_clock_mhz == 1500.0
    assert clock.runtime_clock_mhz == 1650.0
    assert clock.runtime_current_clock_mhz == 800.0
    assert profile.effective_clock_mhz() == 1650.0
    assert clock.clock_source() == "npu_smi_runtime"
    assert profile.peak_cube_ops_per_s("bf16") == pytest.approx(20 * 8192 * 1.65e9, rel=1e-12)
    entry = next(item for item in profile.discrepancies if item["field"] == "configured_clock_mhz")
    assert entry["platform_config_clock_mhz"] == 1500.0
    assert entry["runtime_aicore_freq_mhz"] == 1650.0
    assert entry["observed_current_freq_mhz"] == 800.0
    assert entry["caller_override_clock_mhz"] is None
    assert entry["authority"] == "npu_smi_runtime"


def test_a2_profile_without_a_snapshot_reports_the_configured_clock_only() -> None:
    profile = hw.resolve_device_capability_profile(soc="Ascend910B4")
    clock = profile.clock_and_topology_spec
    assert clock.configured_clock_mhz == 1500.0
    assert clock.runtime_clock_mhz is None
    assert clock.runtime_current_clock_mhz is None
    assert profile.effective_clock_mhz() == 1500.0
    assert clock.clock_source() == "platform_config"
    assert profile.peak_cube_ops_per_s("bf16") == pytest.approx(20 * 8192 * 1.5e9, rel=1e-12)
    assert [item for item in profile.discrepancies if "clock" in str(item["field"])] == []


def test_custom_dtype_overrides_and_a3_default_formula_agree() -> None:
    custom = hw.resolve_device_capability_profile(
        soc="Ascend950DT_9581",
        sku_profile={
            "cube_spec": {
                "ops_per_cycle_per_aic": {
                    "bf16": 4096,
                    "fp8_e4m3": 2048,
                    "fp8_e5m2": 1024,
                }
            }
        },
    )
    assert custom.cube_spec.ops_per_cycle_per_die("bf16") == 32 * 4096
    assert custom.cube_spec.ops_per_cycle_per_die("fp8") == 32 * 2048
    assert custom.cube_spec.ops_per_cycle_per_die("int4") is None

    # A3 keeps its existing per-AIC map * count formula; the normalized accessor
    # is numerically identical, so the core lane can switch without a value
    # change.
    for profile in (
        hw.resolve_device_capability_profile(config=_a2_config()),
        hw.resolve_device_capability_profile(config=_a5_config()),
    ):
        defaults = _profile_defaults(profile, {})
        expected = profile.cube_spec.ops_per_cycle_per_die("bf16") * (
            profile.clock_and_topology_spec.configured_clock_mhz * 1e6
        )
        assert defaults["cube_flops_per_s"] == pytest.approx(expected, rel=1e-12)


def test_generic_thermal_has_no_implicit_a5_workload() -> None:
    unmodeled = th.thermal_bounds()
    assert unmodeled["status"] == "unmodeled"
    assert unmodeled["unmodeled_reason"] == th.MISSING_THERMAL_INPUT_REASON
    assert unmodeled["cube_reference_cycles"] is None
    assert unmodeled["clock_table"] == []
    assert unmodeled["gap_sensitivity"] == []
    assert unmodeled["workload_origin"] is None

    no_peaks = th.thermal_bounds(
        fp8_flops=GOLDEN["fp8_flops"],
        bf16_flops=GOLDEN["bf16_flops"],
    )
    assert no_peaks["status"] == "unmodeled"
    assert no_peaks["unmodeled_reason"] == th.MISSING_CUBE_PEAK_REASON
    assert no_peaks["cube_reference_cycles"] is None
    assert no_peaks["unsupported_dtypes"] == ["fp8", "bf16"]


def test_golden_fixture_is_explicit_and_origin_labeled() -> None:
    thermal = th.thermal_bounds(**GOLDEN)
    assert thermal["status"] == "ok"
    assert thermal["cube_reference_cycles"] == pytest.approx(442918.5024, rel=1e-12)
    assert thermal["reference_cube_cycles_per_die"] == pytest.approx(442918.5024, rel=1e-12)
    assert thermal["reference_mhz_for_cycles"] is None
    assert thermal["workload_origin"] == "A5_C8_FLASHMLA_GOLDEN_FIXTURE"
    assert thermal["cycle_basis"] == "flops_over_ops_per_cycle"
    assert thermal["measurement_mhz"] is None
    assert thermal["wall_us"] is None
    assert thermal["counted_us"] is None
    assert thermal["gap_us"] is None
    # No sustained band is supplied, so the frequency table is the caller's
    # short-burst band only (the old 1250-1300 ladder was a fabricated default).
    assert thermal["sustained_mhz_range"] == []
    assert thermal["sustained_reference_us"] is None
    assert thermal["frequencies_mhz"] == [1640.0, 1650.0]
    assert thermal["sensitivity"]["frequencies_mhz"] == thermal["frequencies_mhz"]
    assert {row["frequency_mhz"] for row in thermal["clock_table"]} == set(thermal["frequencies_mhz"])
    assert thermal["ideal_hardware_lower_bound_us"] == pytest.approx(442918.5024 / 1650.0, rel=1e-12)
    assert thermal["ideal_hardware_lower_bound_mhz"] == 1650.0
    assert thermal["required_mhz_for_target"]["reference_cube_mhz"] == pytest.approx(
        442918.5024 / 350.0, rel=1e-12
    )


def test_custom_clock_report_keeps_one_frequency_basis() -> None:
    thermal = th.thermal_bounds(
        wall_cycles=1000.0,
        counted_cycles=800.0,
        cube_reference_cycles=1000.0,
        sustained_mhz_range=(900.0, 1000.0),
        short_burst_mhz_range=(1050.0, 1100.0),
        measurement_mhz=1000.0,
    )
    assert thermal["status"] == "ok"
    assert thermal["reference_mhz_for_cycles"] is None
    assert thermal["frequencies_mhz"] == [900.0, 1000.0, 1050.0, 1100.0]
    assert thermal["sensitivity"]["frequencies_mhz"] == [900.0, 1000.0, 1050.0, 1100.0]
    assert thermal["wall_us"] == pytest.approx(1.0, rel=1e-12)
    assert thermal["counted_us"] == pytest.approx(0.8, rel=1e-12)
    assert thermal["gap_us"] == pytest.approx(0.2, rel=1e-12)
    for row in thermal["clock_table"]:
        mhz = row["frequency_mhz"]
        assert row["wall_us"] == pytest.approx(1000.0 / mhz, rel=1e-12)
        assert row["counted_us"] == pytest.approx(800.0 / mhz, rel=1e-12)
        assert row["reference_lower_bound_us"] == pytest.approx(1000.0 / mhz, rel=1e-12)
    assert thermal["ideal_hardware_lower_bound_us"] == pytest.approx(1000.0 / 1100.0, rel=1e-12)
    assert thermal["ideal_hardware_lower_bound_mhz"] == 1100.0

    no_clock = th.thermal_bounds(
        wall_cycles=1000.0,
        counted_cycles=800.0,
        cube_reference_cycles=1000.0,
        sustained_mhz_range=(900.0, 1000.0),
        short_burst_mhz_range=(1050.0, 1100.0),
    )
    assert no_clock["wall_us"] is None
    assert no_clock["counted_us"] is None
    assert no_clock["gap_us"] is None
    assert no_clock["clock_table"][1]["wall_us"] == pytest.approx(1.0, rel=1e-12)
    assert "no caller-supplied measurement_mhz" in no_clock["time_basis"]["top_level_times"]


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"sustained_mhz_range": (float("nan"), 1300.0)}, "thermal_clock_range_invalid"),
        ({"short_burst_mhz_range": (1000.0, float("inf"))}, "thermal_clock_range_invalid"),
        ({"extra_frequencies_mhz": (float("nan"),)}, "thermal_frequency_invalid"),
        ({"extra_frequencies_mhz": (0.0,)}, "thermal_frequency_invalid"),
        ({"measurement_mhz": float("nan")}, "thermal_measurement_mhz_invalid"),
        ({"measurement_mhz": -1.0}, "thermal_measurement_mhz_invalid"),
    ],
)
def test_nonfinite_or_nonpositive_clock_inputs_are_rejected(kwargs, reason) -> None:
    with pytest.raises(hw.HardwareError) as caught:
        th.thermal_bounds(cube_reference_cycles=1000.0, **kwargs)
    assert caught.value.reason == reason


def test_invalid_measurement_pair_and_order_are_errors() -> None:
    with pytest.raises(hw.HardwareError) as caught:
        th.thermal_bounds(wall_cycles=1000.0)
    assert caught.value.reason == "thermal_cycles_pair_incomplete"

    with pytest.raises(hw.HardwareError) as caught:
        th.thermal_bounds(wall_cycles=100.0, counted_cycles=200.0)
    assert caught.value.reason == "counted_cycles_exceed_wall_cycles"


def test_explicit_cycle_basis_survives_unsupported_fp8() -> None:
    a2 = _a2_config()
    report = hw.hardware_report(config=a2, fp8_flops=1e9, cube_reference_cycles=1000.0)
    thermal = report["thermal"]
    assert thermal["status"] == "partial_reference_only"
    assert thermal["unmodeled_reason"] is None
    assert thermal["cube_reference_cycles"] == 1000.0
    assert thermal["unsupported_dtypes"] == ["fp8"]
    assert thermal["peak_sources"]["fp8"] is None

    unmodeled = hw.hardware_report(config=a2, fp8_flops=1e9)["thermal"]
    assert unmodeled["status"] == "unmodeled"
    assert unmodeled["unmodeled_reason"] == "unsupported_dtype_or_missing_sku_cube_throughput"
    assert unmodeled["cube_reference_cycles"] is None
    assert unmodeled["unsupported_dtypes"] == ["fp8"]


def test_unsupported_dtype_with_pmu_pair_is_measurement_only() -> None:
    thermal = hw.hardware_report(
        config=_a2_config(),
        fp8_flops=1e9,
        wall_cycles=1000.0,
        counted_cycles=800.0,
        measurement_mhz=1000.0,
        sustained_mhz_range=(900.0, 1000.0),
        short_burst_mhz_range=(1050.0, 1100.0),
    )["thermal"]
    assert thermal["status"] == "partial_measurement_only"
    assert thermal["cube_reference_cycles"] is None
    assert thermal["unsupported_dtypes"] == ["fp8"]
    assert thermal["wall_us"] == pytest.approx(1.0, rel=1e-12)
    assert thermal["counted_us"] == pytest.approx(0.8, rel=1e-12)
    assert thermal["sensitivity"]["frequencies_mhz"] == [900.0, 1000.0, 1050.0, 1100.0]


def test_hardware_report_uses_profile_peaks_and_explicit_overrides() -> None:
    a2 = hw.hardware_report(config=_a2_config(), bf16_flops=12884901888.0)["thermal"]
    assert a2["status"] == "partial_reference_only"
    assert a2["cube_reference_cycles"] == pytest.approx(12884901888.0 / (20 * 8192), rel=1e-12)
    assert a2["peak_sources"]["bf16"] == "sku_profile.cube_spec.ops_per_cycle_per_die"

    a5 = hw.hardware_report(config=_a5_config(), bf16_flops=12884901888.0)["thermal"]
    assert a5["cube_reference_cycles"] == pytest.approx(12884901888.0 / (32 * 8192), rel=1e-12)

    override = hw.hardware_report(
        config=_a5_config(),
        bf16_flops=12884901888.0,
        bf16_ops_per_cycle=1000.0,
    )["thermal"]
    assert override["cube_reference_cycles"] == pytest.approx(12884901888.0 / 1000.0, rel=1e-12)
    assert override["peak_sources"]["bf16"] == "caller_explicit"


def test_generic_hardware_report_and_explicit_golden_report() -> None:
    generic = hw.hardware_report(config=_a5_config())
    thermal = generic["thermal"]
    assert thermal["status"] == "unmodeled"
    assert thermal["unmodeled_reason"] == th.MISSING_THERMAL_INPUT_REASON
    assert thermal["cube_reference_cycles"] is None
    assert thermal["workload_origin"] is None

    explicit = hw.hardware_report(config=_a5_config(), **GOLDEN)["thermal"]
    assert explicit["status"] == "ok"
    assert explicit["cube_reference_cycles"] == pytest.approx(442918.5024, rel=1e-12)
    assert explicit["workload_origin"] == "A5_C8_FLASHMLA_GOLDEN_FIXTURE"
    assert explicit["peak_sources"]["fp8"] == "caller_explicit"
    assert explicit["wall_us"] is None


def test_explicit_cycle_basis_models_thermal_even_when_sku_is_ambiguous() -> None:
    report = hw.hardware_report(
        soc="Ascend910",
        cube_reference_cycles=1000.0,
        sustained_mhz_range=(900.0, 1000.0),
        short_burst_mhz_range=(1050.0, 1100.0),
    )
    assert report["status"] == "ambiguous_sku"
    thermal = report["thermal"]
    assert thermal["status"] == "partial_reference_only"
    assert thermal["unmodeled_reason"] is None
    assert thermal["cube_reference_cycles"] == 1000.0
    assert thermal["frequencies_mhz"] == [900.0, 1000.0, 1050.0, 1100.0]

    unmodeled = hw.hardware_report(soc="Ascend910")["thermal"]
    assert unmodeled["status"] == "unmodeled"
    assert unmodeled["unmodeled_reason"] == hw.AMBIGUOUS_SKU_REASON
    assert unmodeled["cube_reference_cycles"] is None


def test_hardware_report_ini_text_and_conflict() -> None:
    text = (FIXTURES / "Ascend950DT_9581.ini").read_text(encoding="utf-8")
    report = hw.hardware_report(ini_text=text)
    assert report["soc"]["soc_version"] == "Ascend950DT_9581"
    assert report["thermal"]["status"] == "unmodeled"

    with pytest.raises(hw.HardwareError) as caught:
        hw.hardware_report(config=_a5_config(), ini_text=text)
    assert caught.value.reason == "hardware_config_and_ini_text_conflict"


def test_probe_directory_load_preserves_a3_dual_die_phy_ids(tmp_path: Path) -> None:
    capture = _a3_capture_dir(tmp_path / "a3-capture")
    snapshot = hw.load_npu_smi_snapshot(capture)
    assert len(snapshot.devices) == 16
    assert snapshot.phy_ids == tuple(range(16))
    assert snapshot.chip_count == 2
    assert snapshot.dual_die is True
    assert snapshot.hbm_totals_mib == (65536,)
    assert snapshot.board["Product Name"] == "IT22HMDA_4_S"
    assert [(chip.npu_id, chip.chip_id, chip.temp_c) for chip in snapshot.chips] == [
        (0, 0, 51.0),
        (0, 1, 51.0),
    ]
    payload = snapshot.to_dict()
    assert payload["phy_ids"] == list(range(16))
    assert len(payload["devices"]) == 16

    direct = hp.load_npu_smi_snapshot(capture)
    assert direct.phy_ids == snapshot.phy_ids
    parsed = hw.parse_npu_smi_snapshot(
        (FIXTURES / "npu-smi-a3-info.txt").read_text(encoding="utf-8"),
        common_text=(FIXTURES / "npu-smi-a3-common.txt").read_text(encoding="utf-8"),
        board_text=(FIXTURES / "npu-smi-a3-board.txt").read_text(encoding="utf-8"),
    )
    assert parsed.phy_ids == tuple(range(16))


def test_probe_directory_role_detection_and_errors(tmp_path: Path) -> None:
    capture = _a3_capture_dir(tmp_path / "a3")
    (capture / "notes.txt").write_text("not an npu-smi capture\n", encoding="utf-8")
    assert len(hw.load_npu_smi_snapshot(capture).devices) == 16

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(hw.HardwareError) as caught:
        hw.load_npu_smi_snapshot(empty)
    assert caught.value.reason == "npu_smi_info_missing"

    with pytest.raises(hw.HardwareError) as caught:
        hw.load_npu_smi_snapshot(tmp_path / "missing")
    assert caught.value.reason == "npu_smi_path_missing"

    ambiguous = tmp_path / "ambiguous"
    ambiguous.mkdir()
    shutil.copy(FIXTURES / "npu-smi-a2-info.txt", ambiguous / "a2-info.txt")
    shutil.copy(FIXTURES / "npu-smi-a3-info.txt", ambiguous / "a3-info.txt")
    with pytest.raises(hw.HardwareError) as caught:
        hw.load_npu_smi_snapshot(ambiguous)
    assert caught.value.reason == "npu_smi_info_ambiguous"


def test_probe_common_keeps_each_chip_with_its_npu_and_ignores_mcu_tail() -> None:
    text = "\n".join(
        [
            "\tNPU ID                         : 0",
            "\tChip Count                     : 2",
            "\tChip ID                        : 0",
            "\tAicore Freq(MHZ)               : 1650",
            "\tAicore Count                   : 20",
            "\tTemperature(C)                 : 50",
            "\tChip ID                        : 1",
            "\tAicore Freq(MHZ)               : 1650",
            "\tAicore Count                   : 20",
            "\tTemperature(C)                 : 51",
            "\tNPU ID                         : 1",
            "\tChip Count                     : 2",
            "\tChip ID                        : 0",
            "\tAicore Freq(MHZ)               : 1800",
            "\tAicore Count                   : 24",
            "\tTemperature(C)                 : 60",
            "\tChip ID                        : 1",
            "\tAicore Freq(MHZ)               : 1800",
            "\tAicore Count                   : 24",
            "\tTemperature(C)                 : 61",
            "\tChip Name                      : mcu",
            "\tTemperature(C)                 : 40",
        ]
    )
    chips = hp.parse_npu_smi_common(text)
    assert [(chip.npu_id, chip.chip_id, chip.aicore_freq_mhz, chip.temp_c) for chip in chips] == [
        (0, 0, 1650.0, 50.0),
        (0, 1, 1650.0, 51.0),
        (1, 0, 1800.0, 60.0),
        (1, 1, 1800.0, 61.0),
    ]


def test_probe_single_file_modes_remain_supported() -> None:
    info = hw.load_npu_smi_snapshot(FIXTURES / "npu-smi-a3-info.txt")
    assert len(info.devices) == 16
    assert info.chips == ()
    assert info.board == {}

    common = hw.load_npu_smi_snapshot(FIXTURES / "npu-smi-a3-common.txt")
    assert common.devices == ()
    assert [(chip.npu_id, chip.chip_id) for chip in common.chips] == [(0, 0), (0, 1)]

    board = hw.load_npu_smi_snapshot(FIXTURES / "npu-smi-a3-board.txt")
    assert board.devices == ()
    assert board.chips == ()
    assert board.board["Product Name"] == "IT22HMDA_4_S"


def test_thermal_reference_and_target_arithmetic_is_frequency_indexed() -> None:
    thermal = th.thermal_bounds(
        cube_reference_cycles=1000.0,
        target_us=100.0,
        sustained_mhz_range=(900.0, 1000.0),
        short_burst_mhz_range=(1050.0, 1100.0),
    )
    assert thermal["status"] == "partial_reference_only"
    assert thermal["required_mhz_for_target"]["reference_cube_mhz"] == pytest.approx(10.0)
    assert thermal["sustained_reference_us"] == pytest.approx([1000.0 / 900.0, 1000.0 / 1000.0])
    assert not math.isnan(thermal["reference_lower_bounds_us"]["900"])
