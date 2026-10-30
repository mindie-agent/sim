"""Project the shared per-call criterion onto retained scoreboard boundaries.

[Role] Add conditional diagnostic tiers and independent whole-step acceptance.
[Invariants] A reference never becomes a scored cohort. Per-call fields require
  retained counts. Mixed descriptors are tiered separately, never by pooled mean.
  No criterion changes predicted time or removes cumulative residuals.
[Timeouts & Locks] Linear work in finite report rows; no locks or waits.
[Side-Effect Policy] Pure copies of supplied report mappings; no I/O or devices.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

from ..cohort_accuracy import accuracy_fields, accuracy_policy


CONDITIONAL_STATUSES = {
    "conditional", "conditional_reference", "conditional_closed_call_blocks",
    "conditional_semantic_stage_projection", "conditional_source_phase",
}


def step_fields(pair):
    """The whole step always uses 15%, including steps shorter than 50 us."""
    result = {
        "step_accuracy_tolerance": {"unit": "ratio", "value": 0.15},
        "step_accuracy_scope": "conditional_whole_step_not_heldout_validation",
        "step_accuracy_pass": None,
        "step_accuracy_status": "unscored",
    }
    if pair["status"] == "conditional_range":
        endpoints = [step_fields(row) for row in pair["scenarios"]]
        result["step_accuracy_endpoints"] = endpoints
        if endpoints and all(row["step_accuracy_pass"] is not None for row in endpoints):
            result["step_accuracy_pass"] = all(row["step_accuracy_pass"] for row in endpoints)
    elif pair["status"] == "conditional" and pair.get("absolute_gap_percent") is not None:
        result["step_accuracy_pass"] = pair["absolute_gap_percent"] <= 15
    if result["step_accuracy_pass"] is not None:
        result["step_accuracy_status"] = (
            "within_relative_tolerance" if result["step_accuracy_pass"] else "outside_tolerance")
    return result


def call_fields(row, *, homogeneous=False):
    """Retain per-call arithmetic, with a separate descriptive-only criterion."""
    count = row.get("occurrences")
    has_count = isinstance(count, (int, float)) and math.isfinite(count) and count > 0
    paired = row.get("status") in CONDITIONAL_STATUSES
    observed = row.get("observed_mean_ms") if has_count else None
    predicted = row.get("predicted_weighted_mean_ms", row.get("predicted_ms")) if has_count else None
    if has_count and observed is None and row.get("observed_sum_ms") is not None:
        observed = row["observed_sum_ms"] / count
    if has_count and predicted is None and row.get("predicted_sum_ms") is not None and paired:
        predicted = row["predicted_sum_ms"] / count
    if not paired:
        predicted = None
    record = {
        "status": "reference_only" if paired else "unscored",
        "observed": {"duration_ns": {"mean": None if observed is None else observed * 1e6}},
        "prediction": {"per_call_ms": predicted},
    }
    fields = accuracy_fields(record)
    fields["accuracy_scope"] = "conditional_diagnostic_excluded_from_scored_cohort_denominator"
    fields["descriptive_tier"] = None
    if not has_count:
        fields["per_call_ineligible_reason"] = "retained_call_count_unavailable"
    elif not paired or predicted is None or observed is None:
        fields["per_call_ineligible_reason"] = "matching_complete_call_prediction_unavailable"
    elif not homogeneous:
        fields["per_call_ineligible_reason"] = "mixed_or_unconfirmed_descriptor_population"
    elif observed <= 0:
        fields["per_call_ineligible_reason"] = "positive_complete_event_mean_unavailable"
    else:
        fields["descriptive_tier"] = {
            **accuracy_fields({**record, "status": "compared"}),
            "scope": "conditional_numeric_diagnostic_not_scored_cohort_acceptance",
        }
    return fields


def annotate_call(row, *, homogeneous=None):
    if homogeneous is None:
        homogeneous = row.get("descriptor_groups") == 1 or row.get("closed_block_comparison") is True
    return {**deepcopy(row), **call_fields(row, homogeneous=homogeneous)}


def group_fields(row):
    """Only a homogeneous matched group supports a descriptive per-call tier."""
    if row.get("comparison_level") == "coarse_semantic_family":
        return {
            **call_fields({"status": row["status"]}),
            "per_call_ineligible_reason": "native_and_program_grouping_not_equivalent",
        }
    left, right = row.get("observed") or {}, row.get("predicted") or {}
    same_count = (left.get("occurrences") == right.get("occurrences")
                  and left.get("occurrences") is not None)
    counts = left["occurrences"] if same_count else None
    summed = (row.get("comparison_level") == "semantic_family"
              or left.get("statistic") == right.get("statistic") == "sum")
    mean = left.get("statistic") == right.get("statistic") == "mean"
    values = {
        "status": row["status"], "occurrences": counts if summed or mean else None,
        "observed_sum_ms": left.get("duration_ms") if summed else None,
        "predicted_sum_ms": right.get("duration_ms") if summed else None,
        "observed_mean_ms": left.get("duration_ms") if mean else None,
        "predicted_weighted_mean_ms": right.get("duration_ms") if mean else None,
    }
    return call_fields(values, homogeneous=(
        row.get("comparison_level") != "semantic_family" and bool(left.get("identity"))))


def policy():
    return {
        "per_call": accuracy_policy(),
        "whole_step": {
            "relative_tolerance": 0.15,
            "scope": "conditional_whole_step_not_heldout_validation",
        },
        "conditional_references": (
            "Per-call numeric diagnostics are separate from scored cohort acceptance; "
            "accuracy_pass remains null. Mixed descriptors are not tiered by pooled mean."),
        "small_call_tolerance": (
            "An acceptance allowance, not measured launch-overhead attribution or a timing floor."),
    }


def annotate_kimi_report(original):
    """Refresh saved descriptor reports without re-executing their predictions."""
    report = deepcopy(original)
    report["accuracy_policy"] = policy()
    report["rows"] = [annotate_call(row, homogeneous=True) for row in report["rows"]]
    for field in ("family_totals", "grouped_reference_domains"):
        for total in report.get(field, ()):
            members = [row for row in report["rows"]
                       if (row.get("family") == total.get("family") if field == "family_totals"
                           else row.get("matrix_groups") == total.get("matrix_groups"))]
            total.update(call_fields(total, homogeneous=len(members) == 1))
            tiers = [row["descriptive_tier"] for row in members if row["descriptive_tier"]]
            residuals = [row["sum_error"] for row in members if row.get("sum_error") is not None]
            total["descriptor_diagnostics"] = {
                "scope": "descriptive_only_not_scored_cohort_denominator",
                "descriptor_count": len(members),
                "tiered_descriptor_count": len(tiers),
                "untiered_descriptor_count": len(members) - len(tiers),
                "statuses": dict(Counter(row["accuracy_status"] for row in tiers)),
                "tiers": dict(Counter(row["accuracy_tier"] for row in tiers)),
                "residual_descriptor_count": len(residuals),
                "signed_cumulative_gap_ms": (
                    math.fsum(row["signed_gap_ms"] for row in residuals) if residuals else None),
                "absolute_descriptor_cumulative_gap_ms": (
                    math.fsum(row["absolute_gap_ms"] for row in residuals) if residuals else None),
            }
    return report


def render_kimi_markdown(report):
    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    lines = [
        "# Kimi descriptor-conditioned criterion", "",
        "Conditional numeric diagnostics only; these references do not enter scored "
        "cohort acceptance. Observed means above 50 us use relative 15%; means at or "
        "below 50 us use absolute 15 us. No predicted duration is changed.", "",
        "| Family | Calls | Observed cumulative ms | Predicted cumulative ms | Signed / absolute descriptor gap ms | Descriptor tier outcomes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in report["family_totals"]:
        diagnostic = row["descriptor_diagnostics"]
        lines.append(
            f"| {row['family']} | {row['occurrences']} | {number(row['observed_sum_ms'])} | "
            f"{number(row.get('predicted_sum_ms'))} | "
            f"{number((row.get('sum_error') or {}).get('signed_gap_ms'))} / "
            f"{number(row.get('descriptor_absolute_residual_sum_ms'))} | "
            f"{diagnostic['statuses']}; untiered {diagnostic['untiered_descriptor_count']} |")
    lines.extend([
        "", "Family means do not select a tier for mixed descriptors. Cumulative activity "
        "is separate from a whole-model step; descriptor residuals retain cancellation.", "",
        "| Descriptor | Family | Calls | Observed us/call | Predicted us/call | Signed / absolute us/call | Conditional tier outcome |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in report["rows"]:
        tier = row["descriptive_tier"]
        outcome = tier["accuracy_status"] if tier else row.get("per_call_ineligible_reason", "unscored")
        lines.append(
            f"| {row['id']} | {row['family']} | {row.get('occurrences', 'unavailable')} | "
            f"{number(row['observed_per_call_us'])} | {number(row['predicted_per_call_us'])} | "
            f"{number(row['per_call_diff_us'])} / {number(row['abs_error_per_call_us'])} | {outcome} |")
    return "\n".join(lines) + "\n"
