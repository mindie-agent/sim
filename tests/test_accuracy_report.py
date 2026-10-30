"""Independent residual/ranking oracles and the actual retained GPU protocol."""
from copy import deepcopy
import json
from pathlib import Path
from statistics import median

import pytest

from mindie_sim.observations.accuracy import accuracy_report
from tools.accuracy_report import retained_copy_campaign


ROOT = Path(__file__).resolve().parents[1]


def point(name, predicted, *, tokens=1, configuration=1, level="operator", occurrences=1):
    return {
        "point_id": name, "duration_ms": predicted, "occurrences": occurrences,
        "identity": {
            "level": level, "boundary": "declared_device_program",
            "hardware": {"resource": "oracle"}, "implementation": {"recipe": "oracle"},
            "phase": "prefill", "shape": {"tokens": tokens},
            "dtype": "bf16", "layout": "contiguous", "execution": {"workers": configuration},
        },
    }


def sample(prediction, observed, *, run="run0", split="heldout", suffix="0"):
    return {
        "sample_id": f"{prediction['point_id']}:{run}:{suffix}",
        "identity": deepcopy(prediction["identity"]), "duration_ms": observed,
        "run_id": run, "split": split,
        "source": f"independent_arithmetic_oracle:{prediction['point_id']}:{run}:{suffix}",
    }


def test_signed_cancellation_preserves_absolute_repeated_error_and_declared_tolerance():
    first, second = point("first", 12, tokens=1, occurrences=3), point("second", 8, tokens=2, occurrences=3)
    report = accuracy_report({
        "predictions": [first, second], "observations": [sample(first, 10), sample(second, 10)],
        "tolerances": {"relative": .1},
        "passed": True,
    })
    totals = report["domain_residuals"][0]
    assert totals["signed_residual_ms"] == 0
    assert totals["absolute_residual_ms"] == 12
    assert totals["selected_observed_sum_ms"] == 60
    assert report["tolerance_outcomes"] == {"within": 0, "outside": 2, "undeclared": 0}
    assert [row["comparison"]["allowed_error_ms"] for row in report["points"]] == [1, 1]


def test_repeated_sampling_cannot_outweigh_independent_runs_and_training_is_excluded():
    prediction = point("p", 21)
    samples = [sample(prediction, 10, run="a", suffix=str(index)) for index in range(20)]
    samples.extend([sample(prediction, 20, run="b"), sample(prediction, 40, run="c"),
                    sample(prediction, 10000, run="training", split="train")])
    used = sample(prediction, 1, run="leaked_training")
    samples.append(used)
    campaign = {
        "predictions": [prediction], "observations": samples,
        "training_sample_ids": [used["sample_id"]],
    }
    first = accuracy_report(campaign)
    paired = first["points"][0]
    assert paired["observed_ms"] == 20
    assert paired["heldout_samples"] == 22 and paired["heldout_runs"] == 3
    assert paired["comparison"]["within_tolerance"] is None
    assert first["coverage"]["excluded_observations"] == {"split_train": 1, "used_for_training": 1}
    campaign["observations"][-2]["duration_ms"] *= 100
    assert accuracy_report(campaign)["points"] == first["points"]


def test_different_boundaries_and_incomplete_identities_remain_unpaired():
    operator = point("operator", 2)
    graph = point("graph", 2, level="subgraph")
    unknown = point("unknown", 3, tokens=2)
    del unknown["identity"]["hardware"]
    unknown_boundary = point("unknown_boundary", 1)
    unknown_boundary["identity"]["boundary"] = None
    wrong_boundary = point("wrapper", 2)
    wrong_boundary["identity"]["boundary"] = "host_wrapper"
    source = sample(operator, 1)
    report = accuracy_report({
        "predictions": [graph, unknown, unknown_boundary, wrong_boundary],
        "observations": [source, sample(unknown_boundary, 1)],
        "tolerances": {"relative": 1},
    })
    assert report["coverage"]["paired_heldout_points"] == 0
    assert report["coverage"]["unpaired_heldout_identities"] == 1
    assert report["points"][0]["reason"] == "no_matching_observation"
    assert report["points"][1]["reason"] == "incomplete_identity"
    assert report["points"][2]["missing_identity_fields"] == ["boundary"]
    assert report["points"][3]["reason"] == "no_matching_observation"
    assert report["coverage"]["excluded_observations"] == {"incomplete_identity": 1}
    assert all(row["comparison"] is None for row in report["points"])
    empty = accuracy_report({"predictions": [operator], "observations": []})
    assert empty["points"][0]["predicted_ms"] == 2
    assert empty["tolerance_outcomes"]["within"] == 0


def test_trend_reports_growth_error_and_reversal_without_using_training_shapes():
    small, large = point("small", 5, tokens=1), point("large", 12, tokens=2)
    campaign = {
        "predictions": [small, large], "observations": [sample(small, 10), sample(large, 20)],
        "relations": [{"id": "tokens", "kind": "trend", "axis": "shape.tokens",
                       "points": ["large", "small"]}],
    }
    relation = accuracy_report(campaign)["relations"][0]
    assert relation["status"] == "agrees"
    assert relation["segments"][0]["ratio_relative_error"] == pytest.approx(.2)
    large["duration_ms"] = 4
    assert accuracy_report(campaign)["relations"][0]["status"] == "disagrees"
    campaign["observations"][0]["split"] = "train"
    relation = accuracy_report(campaign)["relations"][0]
    assert relation["status"] == "not_assessed"
    assert relation["unpaired_points"] == ["small"]


def test_configuration_ranking_exposes_reversal_ties_and_uncontrolled_domain_changes():
    first, second, third = (point("one", 7, configuration=1), point("two", 6, configuration=2),
                            point("three", 4, configuration=3))
    campaign = {
        "predictions": [first, second, third],
        "observations": [sample(first, 8), sample(second, 5), sample(third, 5.02)],
        "tolerances": {"order_absolute_ms": .1},
        "relations": [{"id": "workers", "kind": "ranking", "vary": ["execution.workers"],
                       "points": ["one", "two", "three"]}],
    }
    relation = accuracy_report(campaign)["relations"][0]
    assert relation["status"] == "unresolved_ties"
    assert relation["pair_outcomes"] == {"agrees": 2, "observed_tie": 1}
    first["duration_ms"] = 3
    relation = accuracy_report(campaign)["relations"][0]
    assert relation["status"] == "disagrees"
    assert relation["pair_outcomes"]["reversed"] == 2
    first["identity"]["hardware"] = {"resource": "different"}
    campaign["observations"][0]["identity"] = deepcopy(first["identity"])
    assert accuracy_report(campaign)["relations"][0]["status"] == "domain_mismatch"


def test_duplicate_measurement_cannot_be_relabelled_as_training_and_heldout():
    prediction = point("p", 1)
    original = sample(prediction, 1, split="train")
    with pytest.raises(ValueError, match="conflicting_accuracy_sample_or_split"):
        accuracy_report({
            "predictions": [prediction],
            "observations": [original, dict(original, split="heldout")],
        })


def test_retained_gpu_batch_uses_raw_holdout_and_frozen_engine_not_reported_passes(tmp_path):
    path = ROOT / "data/measurements/gpu/service-curve-validation-20261028.json"
    retained = json.loads(path.read_text())
    retained["passed"] = True
    retained["scores_by_work"] = [{"median_relative_error": 0, "accepted": True}]
    retained["heldout_engine"] = []
    altered = tmp_path / "ignored-prior-verdict.json"
    altered.write_text(json.dumps(retained))
    campaign = retained_copy_campaign(altered, root=ROOT, tolerances={"relative": .1})
    result = accuracy_report(campaign)
    paired = [row for row in result["points"] if row["status"] == "paired"]
    raw = json.loads((ROOT / retained["source"]).read_text())
    heldout = [sample for sample in raw["samples"] if sample["split"] == "heldout"]
    expected = median(row["duration_ms"] * row["repetitions"] for row in heldout)
    assert len(paired) == 1
    assert paired[0]["observed_ms"] == pytest.approx(expected)
    assert paired[0]["heldout_samples"] == 5 and paired[0]["heldout_runs"] == 1
    assert {row["metadata"]["trial"] for row in paired[0]["observation_samples"]} == set(range(5))
    assert paired[0]["comparison"]["relative_error"] == pytest.approx(-.003238146446031376)
    assert paired[0]["sample_relative_error_range"] == pytest.approx(
        [-.06161831738244, .02904259849327])
    assert paired[0]["run_relative_error_range"] == pytest.approx(
        [paired[0]["comparison"]["relative_error"]] * 2)
    assert result["coverage"]["excluded_observations"] == {"split_train": 15}
    assert result["coverage"]["unpaired_predictions"] == 3
    assert all(relation["status"] == "not_assessed" for relation in result["relations"])
