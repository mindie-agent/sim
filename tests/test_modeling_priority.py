"""Refinement priority follows repeated impact, not percentage or cancellation."""
from copy import deepcopy

import pytest

from tools.modeling_priority import cost_ranking, residual_ranking


def pair(recipe, predicted_us, observed_us, calls, *, capture_calls=1,
         contract_capture_calls=1, contract_model_calls=None):
    return {
        "recipe": recipe, "shape": [16], "dtype": "bf16",
        "source_lines": [2], "match": "tensor_contract; dispatch_declared",
        "predicted_us": predicted_us, "observed_median_us": observed_us,
        "model_calls": calls, "capture_calls": capture_calls,
        "contract_model_calls": calls if contract_model_calls is None else contract_model_calls,
        "contract_capture_calls": contract_capture_calls,
    }


def comparison(pairs, total):
    return {
        "boundary": "sum_of_isolated_device_programs", "operator_sum_ms": total,
        "model": "structural", "hardware": "declared", "software": {},
        "phase": "prefill", "tokens_per_sequence": 16, "sequences": 1,
        "selected_families": [], "comparisons": pairs,
    }


def test_cost_ranking_uses_all_occurrences_and_keeps_variant_references():
    def row(family, recipe, calls, duration):
        return {"operator": {"family": family, "recipe": recipe},
                "occurrences": calls, "total_ms": calls * duration}
    report = {
        "boundary": "sum_of_isolated_device_programs", "operator_sum_ms": 25,
        "operator_occurrences": 104,
        "operators": [row("gemm", "projection", 2, 3),
                      row("norm", "rms", 100, .1),
                      row("gemm", "projection", 1, 4),
                      row("attention", "online", 1, 5)],
    }
    before = deepcopy(report)
    ranked = cost_ranking(report)["ranked_families"]
    assert [r["total_ms"] for r in ranked] == [10, 10, 5]
    assert ranked[0]["operator_indices"] == [0, 2]
    assert ranked[0]["occurrences"] == 3
    assert ranked[1]["occurrences"] == 100
    assert [r["share_percent"] for r in ranked] == [40, 40, 20]
    assert [r["cumulative_share_percent"] for r in ranked] == [40, 80, 100]
    assert report == before


def test_large_percentage_on_tiny_operator_does_not_outrank_repeated_residual():
    report = comparison([
        pair("large_under", 1000, 1100, 10),
        pair("large_over", 1100, 1000, 10),
        pair("small", 1, 2, 20),
    ], 25)
    before = deepcopy(report)
    result = residual_ranking(report)
    assert [r["recipe"] for r in result["ranked_pairs"]] == [
        "large_under", "large_over", "small"]
    assert result["ranked_pairs"][-1]["operator_error_percent"] == -50
    assert result["signed_residual_ms"] == pytest.approx(-.02)
    assert result["absolute_residual_ms"] == pytest.approx(2.02)
    assert result["absolute_residual_percent_of_reference"] == pytest.approx(8.08)
    assert result["unpaired_or_ambiguous_predicted_ms"] == pytest.approx(3.98)
    assert report == before


def test_ambiguous_contracts_and_missing_observations_do_not_become_zero_error():
    ambiguous = [
        pair("same_op", 1000, 1100, 10, capture_calls=5, contract_capture_calls=10),
        pair("same_op", 1000, 1200, 10, capture_calls=5, contract_capture_calls=10),
        pair("branch_a", 2000, 2100, 6, contract_model_calls=10),
        pair("branch_b", 3000, 2100, 4, contract_model_calls=10),
    ]
    result = residual_ranking(comparison(ambiguous, 34))
    assert result["one_to_one_pairs"] == 0
    assert result["ambiguous_pairs_excluded_from_sum"] == 4
    assert result["unpaired_or_ambiguous_predicted_ms"] == 34
    assert result["signed_residual_ms"] is None
    assert result["absolute_residual_ms"] is None
    # Candidate impacts stay visible without summing a Cartesian set of pairs.
    assert len(result["ranked_pairs"]) == 4
    empty = residual_ranking(comparison([], 0))
    assert empty["absolute_residual_ms"] is None
    assert empty["paired_predicted_share_percent"] is None
