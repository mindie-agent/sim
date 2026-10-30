"""Real retained family inventory remains conserved across report refreshes."""
import json
from pathlib import Path

import pytest


def test_retained_qwen_family_mapping_conserves_all_costs(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from refresh_model_scoreboard import family_mapping

    report = json.loads((root / "tests/fixtures/verification/"
        "qwen38-family-costs.json").read_text())
    result = family_mapping(report)
    assert result["classified_ms"] + result["unassigned_ms"] == pytest.approx(
        report["costs"]["operator_sum_ms"])
    assignments = result["assignments"]
    assert len(assignments) == len(report["costs"]["families"])
    by_recipe = {row["recipe"]: row["semantic_family"] for row in assignments}
    assert by_recipe["flash_attention"] == "attention"
    assert by_recipe["recurrent_scan0"] == "gdn"
    assert by_recipe["linear_qkvz"] == "matmul"


def test_mixed_native_call_membership_requires_declared_boundary(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from refresh_model_scoreboard import family_mapping

    report = {"costs": {"operator_sum_ms": 11, "families": [
        {"family": "native_fusion", "recipe": "ascend_w4a8_msd",
         "occurrences": 1, "isolated_ms": 7},
        {"family": "native_fusion", "recipe": "declared_native_fusion",
         "occurrences": 1, "isolated_ms": 4},
    ]}}
    raw = family_mapping(report)
    declared = family_mapping(report, {"ascend_w4a8_msd": "grouped_gemm"})
    assert raw["unassigned_ms"] == 11
    assert declared["classified_ms"] == 7
    assert declared["unassigned_ms"] == 4
    assert declared["groups"] == {"grouped_gemm": ["ascend_w4a8_msd"]}


def test_latest_timeout_is_retained_separately_from_older_completed_body(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from retain_body_scoreboard import retain, section

    # This exact captured GLM pair previously passed, then failed the same
    # 90-second runtime budget. The newest failure must survive consumption.
    report = retain(root / "tests/fixtures/body-scoreboard-latest-timeout.json")
    completed, latest = report["rows"][0], report["latest_attempts"][0]
    assert completed["case_id"] == latest["case_id"]
    assert completed["strict_body_15pct"]
    assert latest["comparison"]["predicted_ms"] is None
    assert "strict_body_15pct" not in latest
    assert latest["prediction_failure"]["reason"] == "body_prediction_timeout"
    assert latest["prediction_failure"]["timeout_s"] == 90
    assert latest["runtime_identity"] != completed["runtime_identity"]
    assert report["current_counts"]["passed"] == 1
    assert report["latest_attempt_counts"] == {
        "count": 1, "completed": 0, "failed": 1, "passed": 0}
    rendered = section(report, "receipt.json")
    assert "**0/1 completed**" in rendered and "**1 failed**" in rendered
    assert "body_prediction_timeout" in rendered


def test_native_service_evaluation_relation_survives_compaction(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from retain_body_scoreboard import compact_row
    source = {
        'display_status': 'compared', 'calibration_origin': 'observed',
        'calibration_evidence': {'evaluation_relation': 'in_sample',
                                 'body_residual_used': False},
        'comparison': {'status': 'compared', 'observed_ms': {'mean': 10},
                       'predicted_ms': 9},
    }
    result = compact_row(source)
    assert result['calibration_evidence'] == source['calibration_evidence']
    assert result['strict_body_15pct']
    assert result['relative_error'] == -.1


def test_compact_publication_preserves_failed_latest_and_archived_selection(monkeypatch, tmp_path):
    from hashlib import sha256
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / 'tools'))
    from retain_body_scoreboard import retain
    source = root / 'tests/fixtures/body-scoreboard-latest-timeout.json'
    full = source.read_bytes()
    archive = tmp_path / 'full.json'
    archive.write_bytes(full)
    compact = json.loads(full)
    compact.pop('current_selection_policy')
    compact.pop('coverage_status')
    compact.update(publication_format='mindie-priority-scoreboard-compact/1',
        full_evidence={'path': str(archive), 'sha256': sha256(full).hexdigest(),
                       'size_bytes': len(full)})
    path = tmp_path / 'compact.json'
    path.write_text(json.dumps(compact))
    expected, actual = retain(source), retain(path)
    for key in ('selection_policy', 'coverage_status', 'rows', 'latest_attempt_counts'):
        assert actual[key] == expected[key]
    archive.write_bytes(full + b' ')
    with pytest.raises(ValueError, match='identity_mismatch'):
        retain(path)
