"""Criteria preserve complete-event counts, conditional scope and residual work."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from mindie_sim.observations.scoreboard import gap, model_scoreboard, render_markdown
from mindie_sim.observations.scoreboard_accuracy import (
    annotate_call, annotate_kimi_report, step_fields,
)
from test_model_scoreboard import glm_entry


def descriptor(identity, observed_us, predicted_us, count):
    observed, predicted = observed_us / 1000, predicted_us / 1000
    return {
        "id": identity, "family": "GroupedMatmul", "status": "conditional_reference",
        "occurrences": count, "observed_mean_ms": observed,
        "predicted_ms": predicted, "observed_sum_ms": observed * count,
        "predicted_sum_ms": predicted * count,
        "sum_error": gap(predicted * count, observed * count),
    }


@pytest.mark.parametrize("observed_us,predicted_us,status", [
    (50, 65, "within_launch_floor_tolerance"),
    (50, 65.001, "outside_tolerance"),
    (50.001, 65.001, "outside_tolerance"),
    (100, 115, "within_relative_tolerance"),
])
def test_homogeneous_group_uses_real_sum_count_and_shared_threshold(
        observed_us, predicted_us, status):
    entry = glm_entry()
    row = {
        "name": "norm", "comparison_key": "norm",
        "identity": {"shape": [1, 1024], "dtype": "bf16"},
        "boundary": "complete_native_event", "scope": "window",
        "statistic": "sum", "occurrences": 20, "duration_ms": observed_us * 20 / 1000,
    }
    entry["observed"]["groups"] = [row]
    entry["predicted"]["groups"] = [{**row, "duration_ms": predicted_us * 20 / 1000}]
    original = deepcopy(entry)
    report = model_scoreboard([entry])
    group = report["models"][0]["group_comparisons"][0]
    assert group["observed_per_call_us"] == pytest.approx(observed_us)
    assert group["predicted_per_call_us"] == pytest.approx(predicted_us)
    assert group["per_call_diff_us"] == pytest.approx(predicted_us - observed_us)
    assert group["abs_error_per_call_us"] == pytest.approx(abs(predicted_us - observed_us))
    assert group["descriptive_tier"]["accuracy_status"] == status
    # A descriptive scoreboard cannot promote itself into scored native cohorts.
    assert group["accuracy_pass"] is None
    assert group["accuracy_status"] == "reference_only"
    assert group["signed_gap_ms"] == pytest.approx((predicted_us - observed_us) * 20 / 1000)
    assert entry == original
    markdown = render_markdown(report)
    assert status in markdown and "outside_tolerance" in markdown


def test_small_call_allowance_does_not_change_whole_step_or_endpoint_criterion():
    entry = glm_entry()
    entry["observed"]["step"]["duration_ms"] = 0.04
    entry["predicted"]["step"]["duration_ms"] = 0.05
    call = annotate_call({
        "status": "conditional_closed_call_blocks", "closed_block_comparison": True,
        "occurrences": 10, "observed_sum_ms": 0.4, "predicted_sum_ms": 0.5,
    })
    assert call["descriptive_tier"]["accuracy_status"] == "within_launch_floor_tolerance"
    report = model_scoreboard([entry])
    step = report["models"][0]["step_comparison"]
    assert step["step_accuracy_pass"] is False
    assert step["absolute_gap_percent"] == pytest.approx(25)
    ranged = step_fields({
        "status": "conditional_range", "scenarios": [
            {"status": "conditional", "absolute_gap_percent": 14},
            {"status": "conditional", "absolute_gap_percent": 16},
        ],
    })
    assert ranged["step_accuracy_pass"] is False
    assert [row["step_accuracy_pass"] for row in ranged["step_accuracy_endpoints"]] == [True, False]


def test_missing_count_and_coarse_native_program_mismatch_do_not_invent_per_call():
    row = annotate_call({
        "status": "conditional_reference", "descriptor_groups": 1,
        "observed_mean_ms": 0.01, "predicted_weighted_mean_ms": 0.02,
        "observed_sum_ms": 100, "predicted_sum_ms": 200,
    })
    assert row["observed_per_call_us"] is row["predicted_per_call_us"] is None
    assert row["per_call_diff_us"] is row["descriptive_tier"] is None
    assert row["per_call_ineligible_reason"] == "retained_call_count_unavailable"
    entry = glm_entry()
    entry["observed"]["family_groups"] = [{
        "name": "matrix", "duration_ms": 10, "occurrences": 7,
        "statistic": "mean_of_step_activity_sums",
    }]
    entry["predicted"]["groups"] = [{
        "name": "projection", "duration_ms": 2, "occurrences": 2,
    }]
    entry["family_bindings"] = [{
        "id": "matrix", "observed_group": "matrix", "predicted_groups": ["projection"],
        "assumptions": ["Native and program boundaries differ."],
        "boundary_assumptions": ["Whole-family activity diagnostic only."],
    }]
    group = model_scoreboard([entry])["models"][0]["group_comparisons"][0]
    assert group["signed_gap_ms"] == -8
    assert group["observed_per_call_us"] is group["predicted_per_call_us"] is None
    assert group["descriptive_tier"] is None
    assert group["per_call_ineligible_reason"] == "native_and_program_grouping_not_equivalent"


def mixed_report():
    # The cumulative gaps cancel although one population misses the criterion.
    rows = [descriptor("small", 10, 24, 1000), descriptor("large", 100, 80, 700)]
    return {
        "schema": "mindie-sim.kimi-family-scoreboard/2", "rows": rows,
        "family_totals": [{
            "family": "GroupedMatmul", "status": "conditional_reference",
            "descriptor_groups": 2, "occurrences": 1700,
            "observed_sum_ms": 80, "predicted_sum_ms": 80,
            "sum_error": gap(80, 80), "descriptor_absolute_residual_sum_ms": 28,
        }],
        "grouped_reference_domains": [],
    }


def test_mixed_descriptor_mean_cannot_hide_large_failure_or_cumulative_cancellation():
    original = mixed_report()
    preserved = deepcopy(original)
    report = annotate_kimi_report(original)
    total = report["family_totals"][0]
    assert total["descriptive_tier"] is None
    assert total["accuracy_pass"] is None
    assert total["observed_per_call_us"] == total["predicted_per_call_us"]
    diagnostic = total["descriptor_diagnostics"]
    assert diagnostic["statuses"] == {
        "within_launch_floor_tolerance": 1, "outside_tolerance": 1,
    }
    assert diagnostic["signed_cumulative_gap_ms"] == 0
    assert diagnostic["absolute_descriptor_cumulative_gap_ms"] == 28
    assert total["sum_error"]["absolute_gap_ms"] == 0
    assert total["descriptor_absolute_residual_sum_ms"] == 28
    assert original == preserved
    assert annotate_kimi_report(report) == report


def test_unresolved_native_stage_has_no_per_call_pair_or_zero_residual():
    original = mixed_report()
    row = original["rows"][0]
    row.update(status="not_comparable", predicted_sum_ms=None, sum_error=None)
    original["rows"] = [row]
    total = original["family_totals"][0]
    total.update(status="not_comparable", descriptor_groups=1, predicted_sum_ms=None,
                 sum_error=None, descriptor_absolute_residual_sum_ms=None)
    report = annotate_kimi_report(original)
    assert report["rows"][0]["predicted_per_call_us"] is None
    assert report["rows"][0]["descriptive_tier"] is None
    assert report["family_totals"][0]["descriptor_diagnostics"]["signed_cumulative_gap_ms"] is None


@pytest.mark.parametrize("stage_name", ["GATE_PREPARE", "FWD_H"])
def test_retained_kda_source_stage_keeps_descriptive_tier_without_scored_acceptance(stage_name):
    fixture = (Path(__file__).parents[1] / "tests/fixtures/verification"
               / "kda-prepare-stages.json")
    retained = json.loads(fixture.read_text())
    stage = next(row for row in retained["stages"] if row["stage"] == stage_name)
    row = {
        "family": f"KDA closed-call stage {stage_name}",
        "status": "conditional_source_phase",
        "closed_block_comparison": True,
        "occurrences": stage["observed_calls"],
        "observed_sum_ms": stage["observed_sum_ms"],
        "predicted_sum_ms": stage["compact_predicted_sum_ms"],
        "sum_error": gap(stage["compact_predicted_sum_ms"], stage["observed_sum_ms"]),
    }
    entry = glm_entry()
    entry["id"] = "kimi-source-stage"
    del entry["observed"]["step"]
    entry["observed"]["unavailable_reasons"] = ["source_stage_is_not_a_model_step"]
    entry["call_comparisons"] = [row]
    original = deepcopy(entry)

    report = model_scoreboard([entry])
    result = report["models"][0]["call_comparisons"][0]
    assert result["occurrences"] == 276
    assert result["observed_per_call_us"] == pytest.approx(stage["observed_sum_ms"] * 1000 / 276)
    assert result["predicted_per_call_us"] == pytest.approx(
        stage["compact_predicted_per_call_ms"] * 1000)
    assert result["per_call_diff_us"] == pytest.approx(stage["signed_gap_ms"] * 1000 / 276)
    assert result["accuracy_status"] == "reference_only"
    assert result["accuracy_pass"] is None
    assert result["descriptive_tier"]["accuracy_tier"] == "relative_15pct"
    assert result["descriptive_tier"]["accuracy_status"] == "outside_tolerance"
    assert "per_call_ineligible_reason" not in result
    assert result["sum_error"] == row["sum_error"]
    assert report["coverage"]["conditional_step_comparisons"] == 0
    assert report["models"][0]["step_comparison"]["step_accuracy_pass"] is None
    assert report["descriptive_residuals"]["signed_gap_ms"] is None
    assert report["descriptive_residuals"]["absolute_gap_ms"] is None
    assert entry == original


def test_tool_refreshes_saved_kimi_report_without_prediction_or_raw_mutation(tmp_path):
    path = Path(__file__).parents[1] / "tools/model_scoreboard.py"
    spec = importlib.util.spec_from_file_location("scoreboard_accuracy_tool", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    incoming, output, markdown = [tmp_path / name for name in ("input.json", "output.json", "output.md")]
    original = mixed_report()
    incoming.write_text(json.dumps(original))
    assert tool.main(["--kimi-family-report", str(incoming), "-o", str(output),
                      "--markdown", str(markdown)]) == 0
    report = json.loads(output.read_text())
    assert json.loads(incoming.read_text()) == original
    assert report["accuracy_refresh"]["execution"] == "retained_predictions_not_reexecuted"
    assert report["rows"][1]["descriptive_tier"]["accuracy_pass"] is False
    assert "outside_tolerance" in markdown.read_text()
    assert "28.000000" in markdown.read_text()
