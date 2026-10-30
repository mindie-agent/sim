"""Preserved capture boundaries and independent analytical callback predictions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim.__main__ import main
from mindie_sim.observations.call_alignment import compare_calls
from mindie_sim.observations.captures import iter_conditions
from mindie_sim.observations.store import digest_file
from test_capture_adapters import activation_protocol, grouped_protocol


ROOT = Path(__file__).resolve().parents[1]


def activation_case():
    return json.loads((ROOT / "examples/calls/activation-nominal.json").read_text())


def test_real_activation_calls_keep_event_span_and_reuse_physical_predictions():
    protocol = activation_protocol()
    original = list(iter_conditions(protocol))
    result = compare_calls(activation_case(), protocol, include_submission=True)
    assert result["status"] == "completed"
    assert result["summary"]["samples"] == 270
    assert result["summary"]["sample_roles"] == {"unassigned": 270}
    assert result["summary"]["unique_predictions"] < len(original) * 2
    assert result["validation"]["strict_pairs"] == 0
    for row, source in zip(result["conditions"], original):
        assert row["identity"] == source["identity"]
        assert row["canary"] == source["metadata"].get("canary")
        for comparison in row["comparisons"]:
            assert comparison["observation_boundary"] == source["identity"]["boundary"]
            assert comparison["prediction_boundary"] != comparison["observation_boundary"]
            assert not comparison["strict_identity_equivalence"]
            assert comparison["signed_difference_ms"] == pytest.approx(
                comparison["predicted_duration_ms"] - comparison["observed_duration_ms"])
    burst = next(row for row in result["conditions"] if row["boundary"] == "npu_event_burst")
    source = next(row for row in original if row["condition_id"] == burst["condition_id"])
    # The adapter already restored whole-event time; do not multiply it again.
    from statistics import median
    assert burst["statistics"]["sample_median_ms"] == median(
        sample["duration_ms"] for sample in source["samples"])
    assert all(p["execution"]["live_tasks"] == 0 for p in result["predictions"])


def test_event_durations_and_canaries_never_change_predictions(tmp_path):
    protocol = activation_protocol()
    reference = compare_calls(activation_case(), protocol)
    changed = deepcopy(protocol)
    for index, capture in enumerate(changed["captures"]):
        raw = json.loads(Path(capture["path"]).read_text())
        raw["canaries"] = {name: {"ok": False, "note": "independent canary change"}
                           for name in raw["canaries"]}
        for cell in raw["cells"].values():
            cell["samples_ms"] = [value * 17 for value in cell["samples_ms"]]
            cell["per_pass_ms"] = [[value * 17 for value in trial]
                                   for trial in cell["per_pass_ms"]]
            cell["burst_samples_ms"] = [value * 17 for value in cell["burst_samples_ms"]]
        path = tmp_path / f"capture{index}.json"
        path.write_text(json.dumps(raw))
        capture.update(path=str(path), sha256=digest_file(path))
    altered = compare_calls(activation_case(), changed)
    assert altered["predictions"] == reference["predictions"]
    for before, after in zip(reference["conditions"], altered["conditions"]):
        assert after["statistics"]["sample_median_ms"] == pytest.approx(
            before["statistics"]["sample_median_ms"] * 17)


def test_actual_grouped_representation_is_not_substituted_with_bf16_nd():
    result = compare_calls(activation_case(), grouped_protocol())
    assert result["status"] == "completed"
    assert result["summary"]["samples"] == 240
    assert result["summary"]["unique_predictions"] == 6
    for row in result["conditions"]:
        assert row["alignment"]["quantization"]["compute"] == "int8"
        assert row["alignment"]["api"]["weight"] == "int8 [E,K,N] FRACTAL_NZ"
        assert row["alignment"]["identified_historical_compiled_kernel"] is None
        assert row["alignment"]["matrix"]["core_recipe"] == "mapped_matmul"
    assert {row["identity"]["execution"]["invocations"]
            for row in result["conditions"]} == {1, 8}
    assert result["validation"]["strict_pairs"] == 0


def test_budget_retains_observations_and_cli_returns_partial_report(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(activation_protocol()))
    output = tmp_path / "comparison.json"
    code = main(["compare-calls", str(ROOT / "examples/calls/activation-nominal.json"),
                 str(protocol), "--max-predictions", "1", "-o", str(output)])
    result = json.loads(output.read_text())
    assert code == 2
    assert result["summary"]["samples"] == 270
    assert result["summary"]["prediction_evaluations"] == 1
    assert result["summary"]["condition_statuses"]["prediction_failed"] > 0
    assert any(comparison.get("reason") == "prediction_budget_exceeded"
               for row in result["conditions"] for comparison in row["comparisons"])
