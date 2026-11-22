"""Regression coverage for L0 primitive calibration producer and sealed tables.

[Role] Verify producer, schema, privacy, quantile, burst-turn, and shipped 4-SKU
  sealed-table invariants on top of the locked Single-Spine primitive contract.
[Invariants] Sealed tables preserve certified upper bounds, monotone quantiles,
  privacy hygiene, partial-mode evidence, directional PCIe caps, honest P10
  algorithms, non-extrapolated burst-turn bounds, and realistic P01/P03 lowering.
[Timeouts & Locks] Pure Tier-0 in-memory/file tests; no network or device calls.
[Side-Effect Policy] Read-only over packaged tables; no host filesystem writes.
"""
from __future__ import annotations

import json
import math
import re

import pytest

from _support.hardware import machine
from mindie_sim.primitives import default_primitives_dir, load_manifest, load_primitives, lower_primitive
from mindie_sim.prediction.runtime.memory_physics import AddressRun
from mindie_sim.schema.datapath import datapath_for_die
from tools._bench_l0_bounds import (
    MIN_CALIBRATED_SAMPLES,
    PCIE_GEN4X16_UNI_GBPS,
    PCIE_GEN5X16_UNI_GBPS,
    directional_pcie_bps,
    summarize_metric,
)
from tools.bench_l0_primitives import (
    assert_no_privacy_leak,
    build_primitive_table_payload,
    unwrap_raw_measurements,
)

pytestmark = pytest.mark.tier0

CALIBRATED_FLEET_SKUS = ("Ascend910B4", "Ascend910_9362", "Ascend910_9382", "Ascend950DT_9582")
UNMEASURED_FLEET_SKUS = ("Ascend950PR_9579",)
PRIVATE_PATTERN = re.compile(
    r"(?<![\d.])(?:\d{1,3}[._-]){3}\d{1,3}(?![\d.])|@|/(?:Users|home|root)/|\$\{?HOME\b"
)


def _dp(sku: str):
    return datapath_for_die(machine(sku).dies[0])


@pytest.mark.parametrize("sku", CALIBRATED_FLEET_SKUS)
def test_calibrated_fleet_tables_lower_strided_p01_and_fixpipe_p03_cleanly(sku):
    dp = _dp(sku)
    table = load_primitives(sku, datapath=dp, require_calibrated=True)
    assert table.provenance.calibrated
    assert "silicon_profile_conflict" not in table.cell("P01_datacopy").parameters
    strided = lower_primitive(
        "P01_datacopy",
        {"runs": (AddressRun(0, 4, 64, 512),)},
        table,
        datapath=dp,
        allow_uncalibrated_declaration=True,
    )
    assert strided.bound_access is not None and strided.bound_access.ledger.logical_tensor_bytes == 256
    fix = lower_primitive(
        "P03_fixpipe",
        {"m": 64, "n": 128, "dst": "gm"},
        table,
        datapath=dp,
        allow_uncalibrated_declaration=True,
    )
    assert any("fixpipe" in st.work for st in fix.stages)
    assert table.cell("P10_sort_topk").parameters.get("sort_algorithm") == "sort32_mrgsort4"
    p13 = table.cell("P13_crosscore_sync")
    assert p13.status == "calibrated" and p13.band("mode2_paired_ns").p50 > 0.0
    if dp.arch_id == "npu_arch_3510":
        assert p13.band("mode4_intra_block_ns").p50 != p13.band("mode2_paired_ns").p50


def test_unmeasured_sku_is_explicit_and_all_packaged_files_have_zero_privacy_leaks():
    manifest = load_manifest()
    pdir = default_primitives_dir()
    for sku in UNMEASURED_FLEET_SKUS:
        dp = _dp(sku)
        table = load_primitives(sku, datapath=dp)
        assert not table.provenance.calibrated
        assert all(cell.status == "unmeasured" for cell in table.primitives.values())
    for sku in CALIBRATED_FLEET_SKUS + UNMEASURED_FLEET_SKUS:
        entry = manifest["tables"][sku]
        table_text = (pdir / entry["file"]).read_text(encoding="utf-8")
        cap_text = (pdir / "captures" / entry["file"].replace(".json", ".capture.json")).read_text(encoding="utf-8")
        assert not PRIVATE_PATTERN.search(table_text)
        assert not PRIVATE_PATTERN.search(cap_text)
        assert_no_privacy_leak(table_text, label=sku)
        assert_no_privacy_leak(cap_text, label=f"{sku}:cap")
        ini_name = json.loads(table_text)["primitives"]["P01_datacopy"]["parameters"]["ini_filename"]
        assert "/" not in ini_name and "\\" not in ini_name


def test_summarize_metric_rejects_short_nan_and_preserves_monotone_quantiles():
    band_short, diag_short = summarize_metric([10.0] * (MIN_CALIBRATED_SAMPLES - 1), certified_bound=20.0)
    assert diag_short["status"] == "insufficient_or_invalid_samples" and "p50" not in band_short
    band_nan, diag_nan = summarize_metric([10.0] * MIN_CALIBRATED_SAMPLES + [math.nan], certified_bound=20.0)
    assert diag_nan["status"] == "insufficient_or_invalid_samples" and "p50" not in band_nan
    band_zero, diag_zero = summarize_metric([0.0] * MIN_CALIBRATED_SAMPLES, certified_bound=0.0, bound_kind="latency_lower")
    assert diag_zero["status"] == "insufficient_or_invalid_samples" and "p50" not in band_zero
    # Exact tie values that previously triggered 1-ULP p95 > p99 under x*(1-f) + x*f
    tie_val = 122248996401531.69
    band_tie, diag_tie = summarize_metric([tie_val] * 16, certified_bound=122880000000000.0)
    assert diag_tie["status"] == "calibrated"
    assert band_tie["p05"] <= band_tie["p50"] <= band_tie["p95"] <= band_tie["p99"] <= band_tie["certified_upper_capacity"]


def test_directional_pcie_caps_unwrap_and_partial_p06_evidence_and_alpha_conflict():
    assert directional_pcie_bps({"uni_peak_bw_gbps": 64.0, "bi_peak_bw_gbps": 128.0, "protocol": "PCIe_Gen5x16"}) == pytest.approx(PCIE_GEN5X16_UNI_GBPS * 1e9)
    assert directional_pcie_bps({"uni_peak_bw_gbps": 64.0, "bi_peak_bw_gbps": 128.0, "protocol": "PCIe_Gen4x16"}) == pytest.approx(PCIE_GEN4X16_UNI_GBPS * 1e9)
    assert directional_pcie_bps({"uni_peak_bw_gbps": 64.0, "bi_peak_bw_gbps": 112.0, "protocol": "HCCS"}) == pytest.approx(64.0e9)
    nested = {"tag": "campaign", "payload": {"device_name": "Ascend910B4", "P06_mmad": {"bf16_tflops": [1.0]}}}
    assert unwrap_raw_measurements(nested)["device_name"] == "Ascend910B4"
    raw = {
        "device_name": "Ascend910B4",
        "ini_filename": "/usr/local/Ascend/opp/platform_config/Ascend910B4.ini",
        "P01_datacopy": {
            "hbm_read_bytes_per_s": [6.5e11] * 12,
            "hbm_write_bytes_per_s": [6.2e11] * 12,
            "hbm_rw_alpha": [1.32] * 12,
        },
        "P06_mmad": {"cube_only_modes": {"bf16": [1.2e13] * 12}},
    }
    payload = build_primitive_table_payload(
        sku="Ascend910B4", cann_version="9.2.1", endpoint_alias="operator@node.example.invalid",
        campaign_id="camp-integrity-test", raw_measurements=raw,
    )
    from mindie_sim.primitives import parse_primitive_table
    parse_primitive_table(payload)
    assert "operator" not in payload["provenance"]["capture_uri"] and "example" not in payload["provenance"]["capture_uri"]
    p06 = payload["primitives"]["P06_mmad"]
    assert p06["modes"]["bf16"]["status"] == "calibrated" and p06["modes"]["bf16"]["evidence_id"].startswith("capture://")
    assert payload["primitives"]["P01_datacopy"]["parameters"]["ini_filename"] == "Ascend910B4.ini"
    p01_diag = payload["primitives"]["P01_datacopy"]["parameters"]["metric_diagnostics"]["hbm_rw_alpha"]
    assert p01_diag["status"] == "p99_exceeds_nominal_conflict" and p01_diag["observed_p50"] == pytest.approx(1.32)
    assert "p50" not in payload["primitives"]["P01_datacopy"]["metrics"]["hbm_rw_alpha"]

    for sku in CALIBRATED_FLEET_SKUS:
        tbl = load_primitives(sku, datapath=_dp(sku), require_calibrated=True)
        assert tbl.cann_version and tbl.cann_version != "unmeasured"
        bt = tbl.cell("P01_datacopy").parameters.get("burst_turn_penalty") or ()
        assert all(int(r.get("min_row_bytes", 0)) > 0 for r in bt)


def test_l0_metrology_cold_ring_zero_trial_p07_p10_p15_p18_p20_regressions():
    """Lock fixes #7-#12: cold-ring touched footprint, zero-duration trials, P07/P10 units, P15/P18/P20 guards."""
    from types import SimpleNamespace
    from mindie_sim import compose_case, parse_case
    from mindie_sim.prediction.collectives.topology import Topology
    from mindie_sim.prediction.execution.step_helpers import _calibrated_stars_spec
    from tools._bench_l0_probes import _safe_rate, certify_cold_ring, collect_event_ms

    # #7: 16 KiB probe touching 3.25 MiB prefix on 192 MiB L2 is NOT cold unless evicted.
    l2_192m = 192 * 1024 * 1024
    assert certify_cold_ring(3_407_872, 16_384, l2_192m, evicted_bytes=0) is False
    assert certify_cold_ring(3_407_872, 640 * 1024 * 1024, l2_192m, evicted_bytes=640 * 1024 * 1024) is True

    # #8: Zero-duration trial is preserved (not dropped) and rejected by summarize_metric.
    class _FakeEvt:
        def __init__(self, ms: float):
            self._ms = ms
        def elapsed_time(self, _other) -> float:
            return self._ms
    evs = [(_FakeEvt(1.0), None)] * 8 + [(_FakeEvt(0.0), None)]
    ms_list = collect_event_ms(evs, inner=1)
    assert len(ms_list) == 9 and ms_list[-1] == 0.0
    rates = [_safe_rate(1e6, dt) for dt in ms_list]
    band, diag = summarize_metric(rates, certified_bound=2e9)
    assert diag["status"] == "insufficient_or_invalid_samples" and diag["invalid_samples"] == 1 and "p50" not in band

    # #9: P07 (179.13 ns for 8192-elem MulAddDst ops_per_element=2; 89.56 ns for 1-op) and P10 (1.5082 us for N=1024).
    dp9582 = _dp("Ascend950DT_9582")
    tbl9582 = load_primitives("Ascend950DT_9582", datapath=dp9582, require_calibrated=True)
    assert tbl9582.cell("P07_vec_alu").parameters.get("work_unit") == "elementary_alu_ops_per_s"
    assert tbl9582.cell("P10_sort_topk").parameters.get("work_unit") == "elem_passes_per_s"
    r_p07_fma = lower_primitive(
        "P07_vec_alu", {"elements": 8192, "ops_per_element": 2, "clock_hz": 1.65e9}, tbl9582, datapath=dp9582,
        allow_uncalibrated_declaration=True,
    )
    assert 170.0 <= r_p07_fma.analytical_ms * 1e6 <= 190.0
    r_p07_1op = lower_primitive(
        "P07_vec_alu", {"elements": 8192, "ops_per_element": 1, "clock_hz": 1.65e9}, tbl9582, datapath=dp9582,
        allow_uncalibrated_declaration=True,
    )
    assert 85.0 <= r_p07_1op.analytical_ms * 1e6 <= 95.0
    r_p10 = lower_primitive("P10_sort_topk", {"n": 1024}, tbl9582, datapath=dp9582, allow_uncalibrated_declaration=True)
    assert 1.4 <= r_p10.analytical_ms * 1e3 <= 1.65

    # #10: P15 AIV min-kernel (1.810 us) never binds as STARS dispatch for AIC GEMMs.
    dp9382 = _dp("Ascend910_9382")
    tbl9382 = load_primitives("Ascend910_9382", datapath=dp9382, require_calibrated=True)
    assert tbl9382.cell("P15_rts_stars").parameters.get("issue_interval_includes_min_kernel") is True
    assert _calibrated_stars_spec(dp9382.stars_spec, tbl9382, submission_mode="eager", owner_kind="aic") is None

    # #11: P18 2-rank intra-package AR eager startup never leaks to cross-package AG, 4-rank RS, or blocked stack.
    cs4 = parse_case(compose_case("qwen3.8-27b", hardware="Ascend910_9382", tp=4, ep=1, num_devices=4, layers=1, quant="none"))
    topo = Topology(cs4.machine, cs4.runtime.parallel, primitive_table=tbl9382)
    assert topo.p18_matched_startup_us("allreduce", (0, 1), stack="eager") > 100.0
    assert topo.p18_matched_startup_us("allgather", (0, 2), stack="eager") == 0.0
    assert topo.p18_matched_startup_us("reduce_scatter", (0, 1, 2, 3), stack="eager") == 0.0
    assert topo.p18_matched_startup_us("allreduce", (0, 1), stack="") == 0.0

    # #12: P20 selects sealed directional D2H startup (~65.59 us on 9362), not H2D alias.
    dp9362 = _dp("Ascend910_9362")
    tbl9362 = load_primitives("Ascend910_9362", datapath=dp9362, require_calibrated=True)
    r_p20 = lower_primitive(
        "P20_host_dma_sync",
        {"payload_bytes": 4 * 1024 * 1024, "direction": "d2h", "memory_kind": "pinned", "node": 0, "root_complex_id": "rc0", "die": 0},
        tbl9362, datapath=dp9362, allow_uncalibrated_declaration=True,
    )
    startup_us = sum(st.latency * 1000.0 for st in r_p20.stages)
    assert startup_us == pytest.approx(tbl9362.cell("P20_host_dma_sync").band("pinned_d2h_startup_ns").p50 * 1e-3, rel=1e-4)
    assert startup_us > 50.0
