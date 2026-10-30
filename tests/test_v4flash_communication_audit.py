"""Independent arithmetic and provenance checks for the V4Flash comm audit.

The synthetic rows below use the retained 46-column kernel_details schema; they
exercise interval identity and class separation, not a fabricated external
schema.  Committed scoreboard/case data supply the full-model invariants.
"""
from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit_v4flash_communication.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("audit_v4flash_communication", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_tool()


def row(name, type_name, start, duration, *, core="COMMUNICATION", stream="N/A", device="8"):
    values = {column: "N/A" for column in audit.CAPTURE_COLUMNS}
    values.update({
        "Device_id": device, "Task ID": "1", "Stream ID": stream, "Name": name,
        "Type": type_name, "Accelerator Core": core, "Start Time(us)": start,
        "Duration(us)": duration, "Wait Time(us)": "0.000",
        "Input Shapes": "N/A", "Output Shapes": "N/A",
    })
    return values


def write_capture(path: Path, rows) -> Path:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=audit.CAPTURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_interval_union_and_intersection_use_half_open_ns():
    assert audit.union_ns([(0, 10), (5, 15), (20, 30)]) == 25
    assert audit.union_ns([(0, 10), (10, 20)]) == 20
    assert audit.intersection_ns([(0, 10), (20, 30)], [(3, 12), (25, 40)]) == 12
    assert audit.intersection_ns([(0, 10)], [(10, 20)]) == 0
    assert audit.intersection_ns([(0, 10), (2, 6)], [(3, 8), (5, 12)]) == 7
    assert audit.intersection_ns([(0, 20)], [(0, 10), (5, 15)]) == 15
    assert audit.intersection_ns([(0, 10), (2, 6)], [(3, 4)]) == 1


def test_native_alias_aicpu_are_separated_and_duplicate_is_not_charged(tmp_path):
    native_reduce = row("hcom_allReduce__503_1_1", "hcom_allReduce_", "1000.000", "2000.000")
    aicpu_reduce = row("allreduceAicpuKernel", "allreduceAicpuKernel", "1500.000", "1000.000",
                       core="AI_CPU", stream="92")
    native_alltoall = row("hcom_alltoallv__703_1_1", "hcom_alltoallv_", "4000.000", "3000.000")
    alias_alltoall = row("AivKernel", "hcom_alltoallv_", "4000.000", "3000.000", stream="41")
    native_gather = row("hcom_allGather__703_1_1", "hcom_allGather_", "8000.000", "1000.000")
    alias_gather = row("AivKernel", "hcom_allGather_", "8000.000", "1000.000", stream="41")
    aicpu_gather = row("allgatherAicpuKernel", "allgatherAicpuKernel", "8500.000", "500.000",
                       core="AI_CPU", stream="92")
    duplicate = dict(native_alltoall)
    capture = write_capture(tmp_path / "capture.csv", [
        native_reduce, aicpu_reduce, native_alltoall, alias_alltoall,
        duplicate, native_gather, alias_gather, aicpu_gather,
    ])
    report = audit.audit_capture(capture)
    assert report["rows"] == 8
    assert report["duplicate_full_identity_rows"] == 1
    assert report["classes"]["native"] == {
        "count": 3, "sum_ms": 6.0, "union_ms": 6.0,
        "min_ms": 1.0, "median_ms": 2.0, "max_ms": 3.0,
    }
    assert report["projection_candidates"]["rows"] == 2
    assert report["projection_candidates"]["exact_native_interval_matches"] == 2
    assert report["projection_candidates"]["mirror_interval_ms"] == 4.0
    assert report["projection_candidates"]["mirror_groups"] == 2
    assert report["aicpu"]["rows"] == 2
    assert report["aicpu"]["sum_ms"] == 1.5
    assert report["aicpu"]["intersection_with_native_ms"] == 1.5
    assert report["union_intersection"]["native_sum_ms"] == 6.0
    assert report["union_intersection"]["all_rows_sum_ms"] == 11.5
    assert report["union_intersection"]["all_rows_union_ms"] == 6.0
    assert report["union_intersection"]["all_rows_overlap_ms"] == 5.5
    assert report["native_communication_streams"] == {"503": 1, "703": 2}
    assert report["allgather_by_stream"] == {
        "703": {"count": 1, "sum_ms": 1.0, "median_ms": 1.0},
    }


def test_equal_timestamps_need_same_rank_and_one_to_one_pair(tmp_path):
    capture = write_capture(tmp_path / "capture.csv", [
        row("hcom_allGather__1", "hcom_allGather_", "1000", "1000"),
        row("AivKernel", "hcom_allGather_", "1000", "1000", device="9"),
        row("hcom_allGather__2", "hcom_allGather_", "3000", "1000"),
        row("hcom_allGather__3", "hcom_allGather_", "3000", "1000"),
        row("AivKernel", "hcom_allGather_", "3000", "1000"),
        row("hcom_allGather__4", "hcom_allGather_", "5000", "1000"),
        row("AivKernel", "hcom_allGather_", "5000", "1000"),
    ])
    report = audit.audit_capture(capture)
    candidates = report["projection_candidates"]
    assert candidates["exact_native_interval_matches"] == 1
    assert candidates["unmatched_rows"] == 1
    assert candidates["ambiguous_rows"] == 1
    assert candidates["paired_interval_union_ms"] == 1
    assert candidates["mirror_interval_ms"] == 3
    assert report["union_intersection"]["native_mirror_exact_overlap_ms"] == 1


def test_overlapping_aicpu_rows_report_outside_union_not_sum(tmp_path):
    capture = write_capture(tmp_path / "capture.csv", [
        row("hcom_allReduce__1", "hcom_allReduce_", "1000", "2000"),
        row("allreduceAicpuKernel", "allreduceAicpuKernel", "2000", "2000", core="AI_CPU"),
        row("allreduceAicpuKernel", "allreduceAicpuKernel", "2500", "2000", core="AI_CPU"),
    ])
    report = audit.audit_capture(capture)
    assert report["aicpu"]["sum_ms"] == 4
    assert report["aicpu"]["intersection_with_native_ms"] == 1
    assert report["aicpu"]["outside_native_ms"] == 1.5
    assert report["union_intersection"]["native_aicpu_union_ms"] == 3.5


def test_declared_tp8_ep8_contract_separates_self_copy_from_remote_link():
    entry = json.loads((ROOT / audit.DEFAULT_INPUT).read_text())
    conditions = next(row for row in entry["entries"]
                      if row["id"] == audit.MODEL_ID)["observed"]["conditions"]
    contract = audit._contract(ROOT / audit.DEFAULT_CASE, ROOT / audit.DEFAULT_CATALOG,
                               conditions)
    assert contract["source_assignment_partitions"] == 8
    assert contract["assignments_per_source"] == 3072
    assert contract["self_assignments_per_source"] == 384
    assert contract["self_bytes_per_source_per_layer"] == 3_145_728
    assert contract["remote_bytes_per_source_per_layer"] == 22_020_096
    assert contract["whole_cohort_remote_bytes_per_layer"] == 176_160_768
    assert contract["allreduce_bytes_per_rank_per_call"] == 58_720_256
    assert contract["draft_features_gather_bytes"] == 29_360_128
    assert contract["logits_gather_bytes"] == 226_240
    assert contract["predicted_link_bytes_per_rank_total"] == 7_134_737_344
    assert contract["predicted_group_link_bytes"] == {
        "allreduce_two_groups": 5_167_382_528,
        "expert_dispatch": 968_884_224,
        "expert_combine": 968_884_224,
        "draft_features_gather": 29_360_128,
        "logits_gather": 226_240,
    }
    assert contract["collective_recipe_bindings"] == []
    assert contract["algorithm_scope"] == "resource_envelope_default"
    assert contract["round_readiness_modeled"] is False
    assert contract["self_copy_modeled_as_link"] is False


@pytest.mark.skipif(not audit.DEFAULT_CAPTURE.exists(), reason="retained capture not mounted")
def test_retained_capture_reconciles_exactly_once():
    report = audit.build_audit(ROOT / audit.DEFAULT_CAPTURE,
                               scoreboard_path=ROOT / audit.DEFAULT_SCOREBOARD,
                               input_path=ROOT / audit.DEFAULT_INPUT,
                               prediction_path=ROOT / audit.DEFAULT_PREDICTION,
                               case_path=ROOT / audit.DEFAULT_CASE,
                               catalog_path=ROOT / audit.DEFAULT_CATALOG,
                               repository=ROOT)
    capture = report["source"]["capture"]
    assert capture["rows"] == 4064
    assert capture["span_ms"] == 575.235407
    assert capture["activity_sum_ms"] == 609.513832
    assert capture["rank_scope"]["device_ids"] == ["8"]
    assert capture["classes"]["native"]["sum_ms"] == 111.032451
    assert capture["projection_candidates"]["mirror_interval_ms"] == 38.2382
    assert capture["projection_candidates"]["exact_native_interval_matches"] == 178
    assert capture["projection_candidates"]["unmatched_rows"] == 0
    assert capture["projection_candidates"]["ambiguous_rows"] == 0
    assert capture["aicpu"]["intersection_with_native_ms"] == 30.629491
    assert capture["union_intersection"]["all_rows_union_ms"] == 129.60612
    assert capture["alltoallv_group_count"] == 44
    assert capture["alltoallv_ordinal_groups"] == [
        {"ordinal": 1, "count": 44, "sum_ms": 3.009098, "median_ms": 0.065989},
        {"ordinal": 2, "count": 44, "sum_ms": 6.03904, "median_ms": 0.129958},
        {"ordinal": 3, "count": 44, "sum_ms": 23.343473, "median_ms": 0.463141},
    ]
    reconciliation = report["reconciliation"]
    assert reconciliation["observed_minus_predicted_ms"] == pytest.approx(
        reconciliation["observed_ms"] - reconciliation["predicted_ms"], abs=1e-6)
    assert reconciliation["native_minus_predicted_ms"] == pytest.approx(
        111.032451 - reconciliation["predicted_ms"], abs=1e-6)
    assert reconciliation["predicted_total_equals_link_bytes"] is True
