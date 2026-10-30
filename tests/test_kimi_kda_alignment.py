"""Real native neighbors establish call units independently of timing values."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.kimi_kda_alignment import compare
from tools.model_scoreboard_kimi import prediction_packet, predict_worker


ROOT = Path(__file__).resolve().parents[1]


def inputs():
    data = ROOT / "tests/fixtures/verification"
    blocks = json.loads((data / "kda-call-blocks.json").read_text())
    blocks["runs"] = blocks["runs"][:2]
    observed = json.loads((data / "kimi-native-descriptors.json").read_text())
    case = json.loads((ROOT / "examples/cases/kimi-k3-a5-current-checkpoint-conditional.json").read_text())
    packet = prediction_packet("kda", observed, case, {}, context=32768,
                               kda_tokens=8192, matrix_recipe="steady_matmul")
    return blocks, predict_worker(packet)


def test_actual_closed_call_blocks_reconcile_core_and_stage_sums():
    blocks, prediction = inputs()
    result = compare(blocks, prediction)
    observed = sum(event["duration_ms"] for run in blocks["runs"] for event in run["events"])
    predicted = 2 * prediction["rows"][0]["prediction"]["per_call_ms"]
    assert result["matched_blocks"] == 2
    assert result["observed_activity_sum_ms"] == pytest.approx(observed)
    assert result["predicted_activity_sum_ms"] == pytest.approx(predicted)
    assert result["sum_error"]["signed_gap_ms"] == pytest.approx(predicted - observed)
    assert sum(stage["observed_activity_sum_ms"] for stage in result["stages"]) == pytest.approx(observed)
    assert sum(stage["predicted_activity_sum_ms"] for stage in result["stages"]) == pytest.approx(predicted)
    assert result["geometry"]["source_valid_token_interval"] == [8129, 8192]
    assert result["geometry"]["heads"] == 12
    # Retained native-event means remain separate; this comparison recomputes
    # selected call blocks instead of dividing or trusting global summaries.
    blocks["closed_run_activity"]["count"] = 999
    blocks["activity"]["mean_ms"] = 123456
    second = compare(blocks, prediction)
    assert second["matched_blocks"] == 2
    assert second["sum_error"] == result["sum_error"]


def test_missing_boundary_or_different_geometry_cannot_become_a_paired_call():
    blocks, prediction = inputs()
    blocks["runs"][0]["semantic_boundary_anchors"]["following_output_projection"] = False
    result = compare(blocks, prediction)
    assert result["coverage"] == {"observed_runs": 2, "matched": 1, "unpaired": 1}
    assert result["unpaired_run_ids"] == [blocks["runs"][0]["id"]]
    changed = deepcopy(prediction)
    changed["rows"][0]["scenario_descriptors"]["input_shapes"][0][1] = 4096
    assert compare(blocks, changed)["status"] == "incomparable"
