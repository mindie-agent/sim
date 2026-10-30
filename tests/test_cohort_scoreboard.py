"""Real native cohorts retain independent windows, references and signed errors."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim.observations.cohort_scoreboard import summarize_cohorts


ROOT = Path(__file__).resolve().parents[1]


def retained():
    return json.loads((ROOT / "tests/fixtures/verification/qwen38-native-cohorts.json").read_text())


def test_actual_qwen_cohorts_retain_measured_totals_and_native_call_passes():
    source = retained()
    result = summarize_cohorts(source, label="selected", provenance={})
    families = {row["native_kind"]: row for row in result["windows"][0]["families"]}
    assert families["MatMulV3"]["observed_ms"] == pytest.approx(458.381475)
    assert families["QuantBatchMatmulV3"]["observed_ms"] == pytest.approx(714.307065)
    assert sum(row["complete_calls"] for row in families.values()) == 344
    assert sum(row["passed_cohorts"] for row in families.values()) == 8
    assert families["MatMulV3"]["predicted_ms"] == pytest.approx(471.914144, abs=1e-6)
    assert families["MatMulV3"]["sum_absolute_descriptor_gap_ms"] > abs(
        families["MatMulV3"]["signed_gap_ms"])


def test_references_and_repeated_windows_cannot_inflate_passing_denominator():
    source = retained()
    reference = deepcopy(source["windows"][0]["cohorts"][0])
    reference["status"] = "reference_only"
    source["windows"][0]["cohorts"] = [reference]
    second = deepcopy(source["windows"][0])
    second["rank_id"] = "other_rank"
    source["windows"].append(second)
    report = summarize_cohorts(source, label="reference", provenance={})
    assert len(report["windows"]) == 2
    for window in report["windows"]:
        family = window["families"][0]
        assert family["complete_calls"] == 48
        assert family["passed_cohorts"] == 0
        assert family["comparison_status"] == "reference_only"
        assert family["predicted_ms"] > 0


def test_clipped_activity_is_not_compared_to_complete_kernel_calls():
    source = retained()
    row = source["windows"][0]["cohorts"][0]
    row["observed"]["event_count"] += 1
    row["observed"]["clipped_event_count"] += 1
    row["observed"]["activity_sum_ns"] += 1000000
    report = summarize_cohorts(source, label="clipped", provenance={})
    family = next(f for f in report["windows"][0]["families"]
                  if f["native_kind"] == "MatMulV3")
    assert family["complete_calls"] == 184
    assert family["observed_ms"] == pytest.approx(458.381475)
    assert family["predicted_ms"] == pytest.approx(471.914144, abs=1e-6)
