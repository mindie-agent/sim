"""Actual historical call samples, source identity and boundary reconciliation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from statistics import median

import pytest

from mindie_sim.observations.historical import call_inventory, call_observations, compare_call


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures/observations"


def protocol():
    return json.loads((FIXTURES / "gdn-call-protocol.json").read_text())


def test_real_event_rows_keep_trials_call_boundaries_and_original_canary():
    raw = json.loads((FIXTURES / "gdn-call-measure.json").read_text())
    observations = list(call_observations(protocol(), root=ROOT))
    for source, observed in zip(raw["rows"], observations, strict=True):
        durations = [sample["duration_ms"] for sample in observed["samples"]]
        assert durations == source["samples_ms"]
        assert observed["statistics"]["sample_median_ms"] == median(source["samples_ms"])
        trial_medians = [median(source["samples_ms"][:20]), median(source["samples_ms"][20:])]
        assert observed["statistics"]["median_of_trial_medians_ms"] == median(trial_medians)
        assert [sample["trial"] for sample in observed["samples"]] == [0] * 20 + [1] * 20
        assert observed["reported_statistics"]["median"] == source["stat_median"]
    by_role = {row["identity"]["conditions"]["role"]: row for row in observations}
    assert by_role["scan_prefill"]["canary"]["ok"] is False
    assert len(by_role["scan_prefill"]["samples"]) == 40
    assert by_role["conv_prefill"]["identity"]["conditions"]["in_bytes"] == 4096 * 10240 * 2
    assert by_role["conv_prefill"]["identity"]["geometry"]["conv_width"] == 10240


def test_wrapper_measurement_cannot_score_a_single_kernel_or_different_geometry():
    observed = next(call_observations(protocol(), root=ROOT))
    predicted = {key: deepcopy(observed[key]) for key in ("observation_id", "boundary", "identity")}
    actual = observed["statistics"]["median_of_trial_medians_ms"]
    predicted["duration_ms"] = actual * 1.1
    assert compare_call(predicted, observed)["comparison"]["relative_error"] == pytest.approx(.1)
    kernel_only = dict(predicted, boundary="device_kernel")
    assert compare_call(kernel_only, observed)["comparison"] is None
    assert compare_call(kernel_only, observed)["mismatches"] == ["boundary"]
    different = deepcopy(predicted)
    different["identity"]["conditions"]["batch"] = 4
    assert compare_call(different, observed)["comparison"] is None
    assert compare_call(different, observed)["mismatches"] == ["identity"]
    recurrent = next(row for row in call_observations(protocol(), root=ROOT)
                     if row["identity"]["conditions"]["role"] == "scan_decode")
    without_normalization = {
        key: deepcopy(recurrent[key]) for key in ("observation_id", "boundary", "identity")
    }
    without_normalization["identity"]["call_contract"]["components"].pop(0)
    assert compare_call(without_normalization, recurrent)["comparison"] is None


def test_changed_capture_is_not_reused_under_an_old_source_identity(tmp_path):
    selected = protocol()
    changed = tmp_path / "measure.json"
    raw = json.loads((FIXTURES / "gdn-call-measure.json").read_text())
    raw["rows"][0]["samples_ms"][0] *= 2
    changed.write_text(json.dumps(raw))
    selected["capture"]["path"] = str(changed)
    with pytest.raises(ValueError, match="historical_source_changed"):
        list(call_observations(selected, root=ROOT))


def test_failed_raw_call_remains_visible_without_becoming_zero_cost(tmp_path):
    selected = protocol()
    raw = json.loads((FIXTURES / "gdn-call-measure.json").read_text())
    failed = dict(raw["rows"][0], status="failed", error_first_line="recorded warm-up failure")
    for key in tuple(failed):
        if key.startswith("stat_") or key == "samples_ms":
            failed.pop(key)
    raw["rows"] = [failed]
    path = tmp_path / "failed-call.json"
    path.write_text(json.dumps(raw))
    selected["capture"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    observed = next(call_observations(selected, root=ROOT))
    assert observed["statistics"] is None
    assert observed["samples"] == []
    assert observed["failure"]["error_first_line"] == failed["error_first_line"]
    predicted = {key: observed[key] for key in ("observation_id", "boundary", "identity")}
    assert compare_call(dict(predicted, duration_ms=0), observed)["comparison"] is None


def test_inventory_counts_events_separately_from_conditions_and_keeps_bounded_preview():
    inventory = call_inventory(protocol(), root=ROOT, preview_limit=2)
    assert inventory["condition_rows"] == 5
    assert inventory["event_samples"] == sum(
        len(row["samples_ms"])
        for row in json.loads((FIXTURES / "gdn-call-measure.json").read_text())["rows"])
    assert inventory["original_canary_outcomes"] == {"passed": 4, "failed": 1}
    assert len(inventory["observation_preview"]) == 2
    assert inventory["calibration_bindings"] == inventory["prediction_comparisons"] == 0
