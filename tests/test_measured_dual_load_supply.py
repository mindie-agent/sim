"""Measured MAC service consumed through the ordinary tiled operator path."""
import csv
import gzip
import json
import os
from pathlib import Path
import statistics

import pytest

from mindie_sim import operator_costs
from mindie_sim.__main__ import parser, run
from mindie_sim.compose import compose_case
from mindie_sim.observations.calibration import (
    apply_profiler_supply,
    import_profiler_supply,
)
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case

HERE = Path(__file__).parent
SUPPLY = HERE / "fixtures" / "r64_qbmm_dual_load" / "supply.json"
HARDWARE = "Ascend950DT_9582"
N, K = 6144, 16384
HELDOUT_DC = Path(os.environ.get(
    "MINDIE_TEST_R64_GLM51_HELDOUT_DC",
    "/Volumes/data/mindie-agent-workstreams/a5-multinode-fleet-rack06-20261107/"
    "out/glm51_dp16tp1ep16/dc_1k_c32/177.25.21.1/"
    "dp4_pp0_tp0_dcp0_ep4_rank0_1470_20261107100720369_ascend_pt/kernel_details.csv.gz"))
HELDOUT_PF = Path(os.environ.get(
    "MINDIE_TEST_R64_GLM51_HELDOUT_PF",
    "/Volumes/data/mindie-agent-workstreams/a5-multinode-fleet-rack06-20261107/"
    "out/glm51_dp16tp1ep16/pf_2k_c1/177.25.21.1/"
    "dp6_pp0_tp0_dcp0_ep6_rank0_1466_20261107101241633_ascend_pt/kernel_details.csv.gz"))


def _case(owners=None):
    composed = compose_case(
        "glm-5.1", HARDWARE, tp=1, dp=1, ep=1, quant="W8A8_MXFP8",
        prompt_tokens=1024, output_tokens=1, concurrency=1, layers=1,
        num_devices=1, ranks_per_node=1,
    )
    bound = apply_profiler_supply(
        composed, SUPPLY, recipe="output", hardware=HARDWARE, owners=owners)
    return parse_case(bound)


def _price(tokens, owners=None, *, dtype="fp8_e4m3", n=N, k=K):
    from dataclasses import replace

    case = _case(owners)
    ops = target_operators(case, phase="prefill", tokens=tokens, context=0)
    op = next(op for op in ops if op.recipe == "output")
    if (n, k) != (N, K):
        op = replace(op, local_shape=(tokens, n, k),
                     semantic_shape=(tokens, n, k), padded_shape=(tokens, n, k),
                     launch_shape=(tokens, n, k))
    if dtype != op.dtype:
        op = replace(op, dtype=dtype)
    result = operator_costs(case, (op,))
    return result["operators"][0], case, op


def _service(row):
    return row["basis"]["service_regimes"][0]


def _heldout_medians(path, m, owners):
    rows = []
    with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if "QuantBatchMatmulV3" not in row["Name"]:
                continue
            try:
                shape = row["Input Shapes"].strip('"').split(";")[0]
                rm, rk = map(int, shape.split(","))
                out = row["Output Shapes"].strip('"').split(";")[0]
                om, rn = map(int, out.split(","))
                block = int(row["Block Num"])
            except (TypeError, ValueError):
                continue
            if (rm, rk, om, rn, block) != (m, K, m, N, owners):
                continue
            rows.append((
                float(row["Duration(us)"]), float(row["aic_mac_time(us)"]),
                float(row["aic_mte2_time(us)"]), float(row["aic_mac_ratio"]),
                float(row["cube_utilization(%)"]),
            ))
    if not rows:
        pytest.skip(f"heldout rows unavailable: {path}")
    elapsed, mac, mte2, ratio, cube = [statistics.median(values) for values in zip(*rows)]
    issued = 2 * ((m + 15) // 16 * 16) * K * N
    return {
        "samples": len(rows), "elapsed_us": elapsed, "mac_us": mac,
        "mte2_us": mte2, "mac_ratio": ratio, "cube_pct": cube,
        "issued_flops": issued,
    }


def test_train_fixture_is_a_measured_median_pair_with_real_issued_work():
    imported = import_profiler_supply(SUPPLY, recipe="output", hardware=HARDWARE, owners=24)
    supply = imported["supply"]
    assert imported["source_identity"]["source_scope"] == "r64"
    full, tail = supply["saturated"], supply["underfilled"]
    assert (full["event_count"], tail["event_count"]) == (78, 1326)
    assert full["logical_flops"] == full["issued_flops"] == 206158430208
    assert tail["logical_flops"] == 201326592
    assert tail["issued_flops"] == 3221225472
    assert full["rate_flops_per_s"] == pytest.approx(862.118639267344e12, rel=1e-9)
    assert tail["rate_flops_per_s"] == pytest.approx(255.1263640107714e12, rel=1e-9)
    assert full["pipe_times_ns"]["aic_mac_time"] == 239130
    assert tail["pipe_times_ns"]["aic_mac_time"] == 12626
    assert imported["service_regimes"][0]["execution"]["kind"] == "mac_service"
    assert imported["service_regimes"][0]["execution"]["tile_n"] == 256


def test_unseen_m2_and_m2048_choose_measured_mac_rate_and_issued_work():
    tail, _, _ = _price(2)
    tail_service = _service(tail)
    assert tail_service["execution_modes"] == {"mac_service": 1}
    assert tail_service["queries"] == {"observable_response": 1}
    assert tail_service["response_relations"] == {"measured_underfilled": 1}
    assert tail_service["issued_work"] == 3221225472
    assert tail_service["active_owners"] == 24
    assert tail_service["response_us_max"] == pytest.approx(12.626, rel=1e-9)
    assert tail["duration_ms"] == pytest.approx(0.033162, rel=2e-3)

    full, _, _ = _price(2048)
    full_service = _service(full)
    assert full_service["response_relations"] == {"measured_saturated": 1}
    assert full_service["issued_work"] == 412316860416
    assert full_service["active_owners"] == 32
    assert full_service["response_us_max"] == pytest.approx(478.26, rel=1e-9)
    assert full["duration_ms"] == pytest.approx(0.595628, rel=2e-3)


def test_train_geometry_and_owner_conditions_are_actual_not_a_work_envelope():
    train_full, _, _ = _price(1024)
    service = _service(train_full)
    assert service["response_us_max"] == pytest.approx(239.13, rel=1e-9)
    assert service["issued_work"] == 206158430208
    assert service["active_owners"] == 32
    # A mismatched explicit owner count is not silently reclassified to the
    # nearest measured load; the ordinary analytical service stays in place.
    wrong, _, _ = _price(2, 32)
    wrong_service = _service(wrong)
    assert wrong_service.get("response_relations") is None
    assert wrong_service.get("response_us_max") is None
    assert wrong["duration_ms"] == pytest.approx(0.0339516, rel=2e-3)


def test_unrelated_hardware_dtype_and_geometry_keep_the_analytical_path():
    composed = compose_case(
        "glm-5.1", HARDWARE, tp=1, dp=1, ep=1, quant="W8A8_MXFP8",
        prompt_tokens=1024, output_tokens=1, concurrency=1, layers=1,
        num_devices=1, ranks_per_node=1,
    )
    bound = apply_profiler_supply(
        composed, SUPPLY, recipe="output", hardware="Ascend910_9392", owners=32)
    other_case = parse_case(bound)
    op = next(op for op in target_operators(
        other_case, phase="prefill", tokens=2048, context=0)
        if op.recipe == "output")
    other_row = operator_costs(other_case, (op,))["operators"][0]
    other_service = _service(other_row)
    assert other_service["queries"] == {"physical_identity_mismatch": 1}
    assert other_service.get("response_relations") is None
    bare_case = parse_case(composed)
    bare_op = next(op for op in target_operators(
        bare_case, phase="prefill", tokens=2048, context=0)
        if op.recipe == "output")
    bare_row = operator_costs(bare_case, (bare_op,))["operators"][0]
    # No recipe/tiling mutation leaked from the off-domain import.
    assert other_row["duration_ms"] == pytest.approx(bare_row["duration_ms"], rel=1e-12)
    assert other_row["serviced"] == bare_row["serviced"]

    bf16_row, _, _ = _price(2048, dtype="bf16")
    bf16_service = _service(bf16_row)
    assert bf16_service["queries"] == {"physical_identity_mismatch": 1}
    assert bf16_service.get("response_relations") is None

    # Different N/K geometry is outside the exact measured selector.
    geometry_row, _, _ = _price(2048, n=4096)
    geometry_service = _service(geometry_row)
    assert geometry_service["queries"] == {"analytic_outside_domain": 1}
    assert geometry_service.get("response_relations") is None


def test_import_rejects_unavailable_supply_and_requires_explicit_hardware():
    with pytest.raises(ValueError, match="requires_explicit_hardware_identity"):
        import_profiler_supply(SUPPLY)
    unavailable = {
        "schema_version": "mindie-profiler-measured-supply/1",
        "status": "not_available",
        "limitations": [{"code": "measured_supply_no_pair"}],
    }
    with pytest.raises(ValueError, match="not_available"):
        import_profiler_supply(unavailable, hardware=HARDWARE)
    imported = import_profiler_supply(
        unavailable, hardware=HARDWARE, required=False)
    assert imported["service_regimes"] == []
    assert imported["status"] == "not_available"


def test_existing_operators_cli_applies_the_measured_mac_service():
    composed = compose_case(
        "glm-5.1", HARDWARE, tp=1, dp=1, ep=1, quant="W8A8_MXFP8",
        prompt_tokens=1024, output_tokens=1, concurrency=1, layers=1,
        num_devices=1, ranks_per_node=1,
    )
    case_path = Path("/tmp/sat1_sim_cli_case.json")
    case_path.write_text(json.dumps(composed), encoding="utf-8")
    args = parser().parse_args([
        "operators", str(case_path), "--phase", "prefill", "--tokens", "2",
        "--supply", str(SUPPLY), "--supply-recipe", "output",
        "--supply-hardware", HARDWARE,
    ])
    report = run(args)
    row = next(row for row in report["operators"]
               if row["operator"].get("recipe") == "output")
    service = row["basis"]["service_regimes"][0]
    assert service["execution_modes"] == {"mac_service": 1}
    assert service["response_relations"] == {"measured_underfilled": 1}
    assert service["issued_work"] == 3221225472
    assert service["active_owners"] == 24
    assert service["response_us_max"] == pytest.approx(12.626, rel=1e-9)
    # The CLI case's selected program is SteadyMatmul; the measured MAC is
    # applied without replacing that implementation or its MTE2 layout.
    assert row["duration_ms"] == pytest.approx(0.034059, rel=2e-3)


@pytest.mark.skipif(not HELDOUT_DC.is_file() or not HELDOUT_PF.is_file(),
                    reason="set the two independent installed captures to run acceptance")
def test_independent_heldout_captures_compare_after_prediction_only():
    # Forecast first, using only the train supply fixture.
    m2, _, _ = _price(2)
    m2048, _, _ = _price(2048)
    m2_service = _service(m2)
    m2048_service = _service(m2048)
    predicted = {
        2: (m2_service["response_us_max"], m2_service["issued_work"],
            m2_service["active_owners"], m2["duration_ms"] * 1000),
        2048: (m2048_service["response_us_max"], m2048_service["issued_work"],
               m2048_service["active_owners"], m2048["duration_ms"] * 1000),
    }
    # Only now read the independent captures' genuine PMU counters.
    actual = {
        2: _heldout_medians(HELDOUT_DC, 2, 24),
        2048: _heldout_medians(HELDOUT_PF, 2048, 32),
    }
    for m, truth in actual.items():
        mac, issued, owners, elapsed = predicted[m]
        assert issued == truth["issued_flops"]
        assert owners == (24 if m == 2 else 32)
        assert mac == pytest.approx(truth["mac_us"], rel=0.01)
        # Elapsed is a separate MTE2/other-pipe result. It is reported, not
        # silently used to refit the MAC rate.
        assert elapsed > 0 and truth["elapsed_us"] > 0
        assert abs(elapsed / truth["elapsed_us"] - 1) < 0.35

def test_physical_issue_class_rejects_unsupported_m_tails():
    from mindie_sim.prediction.piecewise.compiler import measured_mac_rate

    rule = import_profiler_supply(
        SUPPLY, recipe="output", hardware=HARDWARE, required=True,
    )["service_regimes"][0]
    under = 255.1263640107714e9
    full = 862.118639267344e9
    expected = {
        1: under, 2: under, 16: under,
        17: None, 128: None, 129: None, 512: None,
        1024: full, 2048: full,
    }
    for m, want in expected.items():
        issued = 2 * ((m + 15) // 16 * 16) * K * N
        got = measured_mac_rate(rule, issued, k=K, n=N)
        if want is None:
            assert got is None, (m, got)
        else:
            assert float(got) == pytest.approx(want, rel=1e-12), m

def test_generic_issue_class_rejects_unsupported_tails_and_keeps_full_region():
    for m in (17, 128, 129, 512):
        row, _, _ = _price(m)
        service = _service(row)
        assert service.get("response_relations") is None
        assert service.get("response_us_max") is None
    for m, issued in ((1024, 206158430208), (2048, 412316860416)):
        row, _, _ = _price(m)
        service = _service(row)
        assert service["response_relations"] == {"measured_saturated": 1}
        assert service["active_owners"] == 32
        assert service["issued_work"] == issued

def test_sparse_mte1_resource_service_imports_and_prices_pf2048():
    imported = import_profiler_supply(
        SUPPLY, recipe="output", hardware=HARDWARE, required=True)
    resources = imported["resource_services"]
    assert len(resources) == 1
    record = resources[0]
    assert record["resource"] == "mte1"
    assert record["native_type"] == "KvQuantSparseFlashAttention"
    assert record["source_identity"]["source_scope"] == "r64"
    assert record["hardware"] == HARDWARE
    assert record["owners"] == 32 and record["mix_block_num"] == 64
    assert record["measured_q_tokens"] == 1024
    assert record["measured_critical_bytes"] == 93782016
    assert record["measured_total_bytes"] == 1566572544
    assert record["event_count"] == 78
    assert record["median_mte1_us"] == pytest.approx(270.2975, rel=1e-9)
    assert record["rate_bytes_per_s"] == pytest.approx(346.9585e9, rel=1e-5)

    from mindie_sim.observations.resource_services import select_resource_service
    source = imported["source_identity"]
    pf = select_resource_service(
        resources, hardware=HARDWARE, source_identity=source,
        q_len=2048, context=0, kv_len=2048, causal=True, owners=32,
        phase="prefill", op_state="dynamic")
    assert pf is not None
    assert pf["query_critical_bytes"] == 373948416
    assert pf["time_us"] == pytest.approx(1077.790031, rel=1e-6)
    dc = select_resource_service(
        resources, hardware=HARDWARE, source_identity=source,
        q_len=2, context=1056, kv_len=1057, causal=True, owners=32,
        phase="decode", op_state="static")
    assert dc is None

    composed = compose_case(
        "glm-5.1", HARDWARE, tp=1, dp=1, ep=1, quant="W8A8_MXFP8",
        prompt_tokens=1024, output_tokens=1, concurrency=1, layers=1,
        num_devices=1, ranks_per_node=1,
    )
    bound = apply_profiler_supply(
        composed, SUPPLY, recipe="output", hardware=HARDWARE)
    meta = bound["runtime"]["metadata"]["profiler_measured_supply"]
    assert meta["resource_services"][0]["resource"] == "mte1"
    assert meta["resource_services"][0]["rate_bytes_per_s"] == pytest.approx(
        346.9585e9, rel=1e-5)
