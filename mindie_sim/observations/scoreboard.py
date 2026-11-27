"""Observation accuracy scoreboards and cohort roll-ups.

[Role] Aggregate per-operator and cohort-level prediction accuracy into
  structured release scoreboards.
[Invariants] Deterministic quantiles and exact sample counts.
[Timeouts & Locks] Pure in-memory aggregation; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

from ..cohort_accuracy import (
    NATIVE_ACCEPTANCE_POLICY, accuracy_fields, accuracy_policy,
    native_acceptance, native_acceptance_policy,
)
from ..coverage import (
    comparison_coverage, inventory_coverage, record_linkage,
)


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


NATIVE_SCOREBOARD_SCHEMA = "mindie-sim.native-comparison-scoreboard/1"


def native_comparison_scoreboard(comparison, *, inventory=None, captures=None,
                                 ledger=None, records=None, captures_manifest=None,
                                 record_dispositions=None, record_sources=None,
                                 source_registry=None, captures_index=None,
                                 source_edges=None, label=None, source=None):
    """Project one profiler native comparison into the public sim slice.

    The downstream report reuses the comparison owner's paired items.  It never
    re-prices, medians a failure away, or counts an unpaired side as correct.
    """
    acceptance = native_acceptance(comparison)
    linkage_inputs = (record_dispositions, record_sources, source_registry,
                      captures_index, captures_manifest)
    if inventory is None and any(value is not None for value in (
            captures, ledger, records, *linkage_inputs)):
        raise ValueError("native_scoreboard_inventory_required_for_capture_inputs")
    if record_dispositions is None and any(value is not None for value in (
            record_sources, source_registry, captures_index, source_edges)):
        raise ValueError("native_scoreboard_record_dispositions_required_for_linkage")
    inventory_report = None
    coverage = None
    linkage = None
    if inventory is not None:
        inventory_report = inventory_coverage(
            inventory, captures, ledger=ledger, records=records,
            captures_manifest=captures_manifest)
        if record_dispositions is not None:
            linkage = record_linkage(
                acceptance, dispositions=record_dispositions,
                record_sources=record_sources,
                source_registry=source_registry,
                captures_index=captures_index,
                captures_manifest=captures_manifest,
                source_edges=source_edges)
        coverage = comparison_coverage(
            inventory_report, acceptance, linkage=linkage)
    counts = acceptance["counts"]
    policy = native_acceptance_policy()
    report = {
        "schema": NATIVE_SCOREBOARD_SCHEMA,
        "status": acceptance["status"],
        "label": label,
        "source_schema": acceptance["source_schema"],
        "source_status": acceptance["source_status"],
        "source": source,
        "source_inputs": deepcopy(comparison.get("inputs")),
        "source_boundary_policy": deepcopy(acceptance["boundary_policy"]),
        "source_interpretation": acceptance["source_interpretation"],
        "policy": policy,
        "tolerance": acceptance["tolerance"],
        "coverage": deepcopy(acceptance["coverage"]),
        "coverage_reconciled": acceptance["coverage_reconciled"],
        "subset_scope": deepcopy(acceptance["subset_scope"]),
        "parent_capture": acceptance["parent_capture"],
        "partial_coverage": acceptance["partial_coverage"],
        "truncated": acceptance["truncated"],
        "window": deepcopy(acceptance["window"]),
        "windows": deepcopy(acceptance["windows"]),
        "source_identity": deepcopy(acceptance["source_identity"]),
        "observation_counts": deepcopy(acceptance["observation_counts"]),
        "legacy_evidence": {
            "accepted_as_this_round": [NATIVE_ACCEPTANCE_POLICY],
            "rejected_as_this_round": list(policy["excluded_evidence"]),
            "historical_absolute_15us_pass_accepted": False,
            "in_sample_or_mean_reproduction_accepted": False,
            "training_sample_resplit_used_as_generalization": False,
            "diagnostic_tolerance_used_as_formal": False,
            "work_only_counted_as_measured_duration": False,
            "source_status_authoritative": True,
        },
        "denominators": {
            "accuracy": {
                "basis": "paired_comparable_native_items",
                "paired_items": counts["paired_items"],
                "duration_denominator_items": counts["duration_denominator_items"],
                "duration_denominator_basis": (
                    "within_tolerance+outside_tolerance_items_with_real_predicted_duration"),
                "work_only": counts["work_only"],
                "within": counts["within"],
                "outside": counts["outside"],
                "observation_samples": counts["observation_samples"],
                "condition_gap_items_in_denominator": counts[
                    "condition_gap_items_in_denominator"],
                "unpaired_observations": counts["unpaired_observations"],
                "unpaired_predictions": counts["unpaired_predictions"],
                "ambiguous": counts["ambiguous"],
                "source_status": acceptance["source_status"],
                "partial_coverage": acceptance["partial_coverage"],
                "truncated": acceptance["truncated"],
                "excluded_from_duration_denominator": {
                    "work_only_items": counts["work_only"],
                    "unpaired_observations": counts["unpaired_observations"],
                    "unpaired_predictions": counts["unpaired_predictions"],
                    "ambiguous": counts["ambiguous"],
                },
            },
            "inventory": coverage,
        },
        "counts": acceptance["source_counts"],
        "cohorts": summarize_native_comparison(acceptance),
        "items": acceptance["items"],
        "unpaired_observations": acceptance["unpaired_observations"],
        "unpaired_predictions": acceptance["unpaired_predictions"],
        "ambiguous": acceptance["ambiguous"],
        "failure_matrix": acceptance["failure_matrix"],
        "inventory": inventory_report,
        "record_linkage": linkage,
        "consumer_contract": {
            "schema": NATIVE_SCOREBOARD_SCHEMA,
            "status": acceptance["status"],
            "source_status": acceptance["source_status"],
            "accuracy_denominator": (
                "denominators.accuracy.duration_denominator_items"),
            "accuracy_counts": "denominators.accuracy",
            "duration_denominator": (
                "denominators.accuracy.duration_denominator_items "
                "(within_tolerance+outside_tolerance items with real predicted duration)"),
            "paired_items": (
                "denominators.accuracy.paired_items (all ABI/boundary pairs, "
                "including work_only)"),
            "work_only": (
                "denominators.accuracy.work_only and items[].status=work_only; "
                "no predicted duration, numerical error or within/outside verdict"),
            "precision_denominator": (
                "record_linkage.independent_evidence.precision_denominator_paired_items "
                "equals denominators.accuracy.duration_denominator_items"),
            "items": "items[].formal_allowed_error_ms and items[].samples[]",
            "unpaired": "unpaired_observations/unpaired_predictions/ambiguous",
            "failure_matrix": "failure_matrix",
            "record_linkage": "record_linkage",
            "inventory": "inventory",
            "coverage": "coverage/coverage_reconciled/subset_scope/parent_capture",
        },
        "structure_accuracy": {
            "computed": False,
            "reason": "Unknown, unpaired or ambiguous structural items never enter a "
                      "correctness denominator.",
        },
        "non_claims": list(acceptance["non_claims"]) + [
            "independent_native_slice_only_not_7337_record_validation",
            "no_historical_absolute_15us_or_in_sample_mean_pass",
            "no_training_sample_resplit_as_generalization",
            "no_median_or_family_average_substitution",
            "no_work_only_item_in_duration_denominator",
            "no_unpaired_or_ambiguous_item_in_accuracy_denominator",
            "no_item_only_pass_when_source_is_partial_truncated_or_inconclusive",
            "no_diagnostic_tolerance_as_formal_acceptance",
            "record_match_is_identity_linkage_not_per_call_certification",
            "duplicate_or_derived_records_do_not_increase_accuracy_denominator",
        ],
        "interpretation": list(acceptance["interpretation"]) + [
            "The reported status describes this slice only; a complete candidate still "
            "requires T19 full validation.",
        ],
    }
    return report


def render_native_comparison_markdown(report):
    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    def table_row(*values):
        return "| " + " | ".join(f"{value}" for value in values) + " |"

    accuracy = report["denominators"]["accuracy"]
    lines = [
        "# Independent native comparison",
        "",
        f"Slice status: `{report['status']}`; source comparison status: "
        f"`{report['source_status']}`.",
    ]
    lines += [
        "",
        "Criterion: `tolerance.formal` gives `abs(predicted - observed) / observed "
        "<= 10%` for every item and every retained sample that has a real predicted "
        "duration. A diagnostic tolerance is retained for diagnosis only and never "
        "changes the formal verdict. There is no absolute-error exemption, no median "
        "substitution and no family mean. Work-only items retain observation/work "
        "facts but have no duration verdict or numerical error. A historical "
        "`absolute_15us` or in-sample mean reproduction is not accepted here.",
        "",
        "## Accuracy denominator",
        "",
        "| Paired ABI items | Duration denominator | Within | Outside | Work only | "
        "Samples | Condition gaps | Unpaired observations | Unpaired predictions | "
        "Ambiguous |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.append(table_row(*(accuracy[key] for key in (
        "paired_items", "duration_denominator_items", "within", "outside", "work_only",
        "observation_samples", "condition_gap_items_in_denominator",
        "unpaired_observations", "unpaired_predictions", "ambiguous",
    ))))
    lines += [
        "",
        "`paired_items` is the ABI/boundary pair set (includes work-only items). "
        "`duration_denominator_items` is `within + outside`: only items with a real "
        "predicted duration and formal verdict. Work-only items are not passes and "
        "not failures. Unpaired and ambiguous sides stay below and never become "
        "correct by being unknown.",
        "",
        "## Source coverage",
        "",
    ]
    window = report.get("window") or {}
    source_identity = report.get("source_identity") or {}
    window_kinds = sorted({str(row.get("kind"))
                           for row in (report.get("windows") or ())})
    lines += [
        f"- Source status `{report['source_status']}`; partial_coverage="
        f"`{report['partial_coverage']}`; truncated=`{report['truncated']}`; "
        f"window kind `{window.get('kind', 'unavailable')}`; "
        f"window count `{window.get('count', 'unavailable')}`; "
        f"window kinds `{window_kinds}`.",
        f"- Source identity: `{source_identity.get('path', 'unavailable')}`; "
        f"sha256 `{source_identity.get('sha256', 'unavailable')}`; "
        f"scope `{source_identity.get('sha256_scope', 'unavailable')}`.",
        "- Local item verdicts are kept even when the source window is incomplete; "
        "that does not make the slice passed.",
    ]
    diagnostic = (report.get("tolerance") or {}).get("diagnostic")
    if diagnostic:
        lines.append(
            "- Diagnostic tolerance retained separately: "
            f"relative={diagnostic.get('relative')}, "
            f"absolute_ms={diagnostic.get('absolute_ms')}; it never changes the "
            "formal verdict.")
    inventory = report.get("inventory")
    if inventory:
        ledger = inventory["ledger"]
        identity = inventory.get("identity_ledger") or {}
        recovery = inventory.get("a5_record_recovery") or {}
        flags = recovery.get("flags", {})
        validated = flags.get("full_per_call_10pct_validated") or {}
        validated_text = (", ".join(f"{key}:{value}" for key, value in validated.items())
                          if validated else "unavailable")
        lines += [
            "",
            "## 7337-record inventory identity",
            "",
            "The historical ledger is carried as record identity, not as independent "
            "captures and not as this slice's accuracy denominator.",
            "",
            "| Ledger records | A5 records | Captures (manifest) | A5 captures | "
            "Source-set duplicate extras | Legacy duplicate extras |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        lines.append(table_row(
            ledger["total_records"], ledger.get("a5_scope_records"), ledger.get("captures_total"),
            ledger.get("a5_capture_count"), ledger.get("source_set_duplicate_extra_records"),
            ledger.get("legacy_a5_duplicate_extra_records")
            if ledger.get("legacy_a5_duplicate_extra_records") is not None else "n/a",
        ))
        lines += [
            "",
            f"- Capture count source: `{ledger.get('capture_count_source')}`; actual "
            f"manifest version retained in `inventory.capture_manifest`.",
            f"- Legacy identity-ledger duplicate groups: "
            f"{identity.get('duplicate_groups', 'unavailable')}; multi-record groups: "
            f"{identity.get('multi_record_duplicate_groups', 'unavailable')}.",
            f"- A5 runtime state not restored: "
            f"{report['inventory'].get('unrecovered', {}).get('runtime_state_not_restored_records')}; "
            f"A5 current 10% per-call validated flags: {validated_text}.",
            f"- Comparison coverage: {report['denominators']['inventory']['statement']}",
        ]
    linkage = report.get("record_linkage")
    if linkage:
        independent = linkage["independent_evidence"]
        lines += [
            "",
            "## Ledger record linkage",
            "",
            "Record matching is identity/coverage linkage, not per-call certification. "
            "Duplicate source-set references and derived rows remain visible but do "
            "not increase the paired-item precision denominator.",
            "",
            "| Link status | Paired ABI items | Duration denominator | Work only | "
            "Matched items | Unmatched | Conflict | Matched audit records | "
            "Independent captures | Duplicate refs | Derived | Precision denominator |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: |",
        ]
        lines.append(table_row(
            linkage["status"],
            *(linkage["comparison"][key] for key in (
                "paired_items", "duration_denominator_items", "work_only_items",
                "matched_items", "unmatched_items", "conflict_items",
            )),
            *(independent[key] for key in (
                "matched_audit_records", "independent_capture_count", "duplicate_reference_records",
                "derived_records", "precision_denominator_paired_items",
            )),
        ))
        for row in linkage["items"]:
            if row["status"] != "matched":
                lines.append(
                    f"- {row['item_id']}: {row['status']} ({row.get('reason')}); "
                    f"matched records {row['matched_record_count']}.")
    if report["partial_coverage"] or report["truncated"]:
        lines.append(
            "- Cohort row `all_items_within` is local item evidence only; the overall "
            "slice remains incomplete because the source window is partial/truncated.")
    cohort_text = render_native_comparison(report.get("cohorts", ()))
    if cohort_text:
        lines.extend(cohort_text.splitlines())
    lines += [
        "",
        "## Per-item results",
        "",
        "| Item | Operator | Boundary | Observed ms | Predicted ms | Allowed ms | "
        "Relative error | Samples within/total | Verdict |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in report["items"]:
        samples = item.get("samples", ())
        within = sum(sample["within_tolerance"] for sample in samples)
        lines.append(table_row(
            item["item_id"], item["identity"].get("native_operator"), item["boundary"].get("kind"),
            number(item["observed_ms"]), number(item.get("predicted_ms")),
            number(item.get("formal_allowed_error_ms")), number(item.get("relative_error")),
            f"{within}/{len(samples)}", item["status"],
        ))
    work_only_items = [item for item in report["items"]
                       if item["status"] == "work_only"]
    if work_only_items:
        lines += [
            "",
            "## Work-only items (no duration verdict)",
            "",
            "These ABI pairs retain the real observation and physical work facts. "
            "No predicted duration, formal allowed error or within/outside result is "
            "invented, so they are excluded from the duration denominator.",
            "",
            "| Item | Operator | Observed ms | Cube MACs | Issued MACs | Tensor bytes | Reason |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for item in work_only_items:
            work = item.get("work") or {}
            cube = work.get("cube") or {}
            byte_values = work.get("bytes") or {}
            lines.append(table_row(
                item["item_id"], item["identity"].get("native_operator"), number(item["observed_ms"]),
                number(cube.get("macs")), number(cube.get("issued_macs")), number(byte_values.get("total")),
                item.get("comparison", {}).get("reason", "unavailable"),
            ))
    if report["unpaired_observations"] or report["unpaired_predictions"] \
            or report["ambiguous"]:
        lines += [
            "",
            "## Unpaired and ambiguous sides",
            "",
            "| Side | Identity / ids | Value ms | Reason |",
            "| --- | --- | ---: | --- |",
        ]
        for row in report["unpaired_observations"]:
            lines.append(table_row("observation", row.get("observation_ids") or row.get("request_id"),
                                   number(row.get("observed_ms")), row.get("reason")))
        for row in report["unpaired_predictions"]:
            lines.append(table_row("prediction", row.get("prediction_id"),
                                   number(row.get("predicted_ms")), row.get("reason")))
        for row in report["ambiguous"]:
            lines.append(table_row("ambiguous", row.get("request_id"), "unavailable", row.get("reason")))
    if report["failure_matrix"]:
        lines += [
            "",
            "## Failure matrix",
            "",
            "| Root cause | Reason | Count | Item ids | Claim |",
            "| --- | --- | ---: | --- | --- |",
        ]
        for row in report["failure_matrix"]:
            item_ids = ", ".join(row.get("item_ids", ())[:5])
            lines.append(table_row(row.get("root_cause"), row.get("reason"),
                                   row.get("count"), item_ids, row.get("claim")))
    lines += ["", "## Non-claims", ""]
    lines += [f"- {value}" for value in report["non_claims"]]
    return "\n".join(lines) + "\n"


from collections import Counter
from collections.abc import Mapping
import json

from ..cohort_accuracy import accuracy_fields, accuracy_policy


def summarize_cohorts(report, *, label, provenance):
    windows = []
    for window in report["windows"]:
        groups = {}
        for cohort in window["cohorts"]:
            status = cohort["status"]
            key = (cohort["native_operator_type"], status)
            group = groups.setdefault(key, {
                "native_kind": key[0], "comparison_status": status,
                "cohorts": 0, "complete_calls": 0, "passed_cohorts": 0,
                "tier_counts": Counter(), "failures": [],
                "observed_ms": 0.0, "predicted_ms": 0.0,
                "sum_absolute_descriptor_gap_ms": 0.0,
                "unpriced_cohorts": 0,
            })
            group["cohorts"] += 1
            count = cohort["observed"]["complete_event_count"]
            group["complete_calls"] += count
            observed = cohort["observed"]["complete_activity_sum_ns"] / 1e6
            group["observed_ms"] += observed
            predicted_call = cohort.get("prediction", {}).get("per_call_ms")
            if predicted_call is None:
                group["unpriced_cohorts"] += 1
            else:
                predicted = predicted_call * count
                group["predicted_ms"] += predicted
                group["sum_absolute_descriptor_gap_ms"] += abs(predicted - observed)
            fields = accuracy_fields(cohort)
            if fields["accuracy_pass"] is not None:
                group["passed_cohorts"] += int(fields["accuracy_pass"])
                group["tier_counts"][fields["accuracy_tier"]] += 1
            if fields["accuracy_pass"] is False:
                group["failures"].append({
                    "cohort_id": cohort["cohort_id"],
                    "contract": cohort["contract"],
                    "complete_calls": count,
                    "absolute_activity_gap_ms": fields["abs_error_per_call_us"] * count / 1000,
                    **fields,
                })
        families = []
        for group in groups.values():
            group["tier_counts"] = dict(group["tier_counts"])
            group["failures"].sort(key=lambda row: -row["absolute_activity_gap_ms"])
            group["failed_cohorts"] = len(group["failures"])
            if group["unpriced_cohorts"]:
                group.update(predicted_ms=None, signed_gap_ms=None,
                             absolute_net_gap_ms=None, signed_gap_percent=None,
                             sum_absolute_descriptor_gap_ms=None)
            else:
                gap = group["predicted_ms"] - group["observed_ms"]
                group.update(signed_gap_ms=gap, absolute_net_gap_ms=abs(gap),
                             signed_gap_percent=(
                                 100 * gap / group["observed_ms"] if group["observed_ms"] else None))
            families.append(group)
        windows.append({
            "rank_id": window["rank_id"], "window": window["window"],
            "source_identity": window["source_identity"],
            "families": sorted(families, key=lambda row: -row["observed_ms"]),
        })
    return {
        "label": label, "report_source": provenance,
        "graph_id": report["graph_id"], "structure_source": report.get("structure_source", {}),
        "declared_execution": report["declared_execution"],
        "declared_kernel_bindings": report.get("declared_kernel_bindings", {}),
        "accuracy_policy": accuracy_policy(),
        "scope": "retained_native_complete_calls; isolated_predictions; not_whole_model",
        "aggregation": "separate_variant_rank_and_window; no_cross_report_sum",
        "windows": windows,
    }


def render_cohort_campaigns(campaigns):
    if not campaigns:
        return ""
    lines = [
        "\n## Selected native-call campaigns\n",
        "These retained comparisons do not change any full-step row. Each variant, "
        "rank and capture window stays separate. Activity sums use complete events; "
        "clipped rows are excluded. Family net cancellation does not hide failing "
        "descriptors. References are excluded from the pass denominator.\n",
        "| Campaign / rank / window | Native family | Status | Calls | Observed ms | Predicted ms | "
        "Signed gap ms | Sum absolute descriptor gaps ms | Per-call tiers passed |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    for campaign in campaigns:
        for index, window in enumerate(campaign["windows"]):
            for family in window["families"]:
                passed = (f"{family['passed_cohorts']}/{family['cohorts']}"
                          if family["comparison_status"] == "compared" else "unscored")
                lines.append(
                    f"| {campaign['label']} / {window['rank_id']} / {index} | {family['native_kind']} | "
                    f"{family['comparison_status']} | {family['complete_calls']} | "
                    f"{number(family['observed_ms'])} | {number(family['predicted_ms'])} | "
                    f"{number(family['signed_gap_ms'])} | "
                    f"{number(family['sum_absolute_descriptor_gap_ms'])} | {passed} |")
    for campaign in campaigns:
        lines += [
            "",
            f"- {campaign['label']} report SHA256: `{campaign['report_source']['sha256']}`; "
            f"graph: `{campaign['graph_id']}`. Execution and capture identities remain in JSON.",
        ]
    return "\n".join(lines) + "\n"


def _native_cohort(item):
    identity = item.get("identity")
    boundary = item.get("boundary")
    if not isinstance(identity, Mapping) or not isinstance(boundary, Mapping):
        raise ValueError("native_acceptance_item_requires_identity_and_boundary")
    fusion = boundary.get("fusion")
    members = (fusion.get("members") if isinstance(fusion, Mapping) else None)
    if members is None:
        members = identity.get("fusion_members") or ()
    return {
        "native_operator": identity.get("native_operator"),
        "native_abi_sha256": identity.get("native_abi_sha256"),
        "boundary_kind": boundary.get("kind"),
        "hardware": identity.get("hardware"),
        "model": identity.get("model"),
        "quant": identity.get("quant"),
        "phase": identity.get("phase"),
        "implementation": identity.get("implementation"),
        "deployment_image_id": identity.get("deployment_image_id"),
        "fusion_members": sorted(str(member) for member in members),
    }


def summarize_native_comparison(acceptance):
    """Group independent native items by exact cohort identity, never by name.

    The group row is a count of individual verdicts.  It is not a mean, so a
    failing item cannot disappear inside a cohort total.
    """
    if not isinstance(acceptance, Mapping):
        raise ValueError("native_comparison_summary_requires_acceptance")
    groups = {}
    for item in acceptance.get("items", ()):
        identity = _native_cohort(item)
        key = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        groups.setdefault(key, {"identity": identity, "items": []})["items"].append(item)
    rows = []
    for group in groups.values():
        items = group["items"]
        within = [item for item in items if item["status"] == "within_tolerance"]
        outside = [item for item in items if item["status"] == "outside_tolerance"]
        work_only = [item for item in items if item["status"] == "work_only"]
        timed = within + outside
        failing_samples = [
            sample.get("observation_id")
            for item in outside for sample in item["samples"]
            if not sample["within_tolerance"]
        ]
        observed = [item["observed_ms"] for item in items]
        timed_observed = [item["observed_ms"] for item in timed]
        predicted = [item["predicted_ms"] for item in timed]
        worst = (max(abs(item["relative_error"]) for item in timed)
                 if timed else None)
        rows.append({
            "cohort": group["identity"],
            "paired_items": len(items),
            "duration_denominator_items": len(timed),
            "work_only": len(work_only),
            "within": len(within),
            "outside": len(outside),
            "observation_samples": sum(len(item["samples"]) for item in timed),
            "observed_ms_range": [min(observed), max(observed)] if observed else None,
            "timed_observed_ms_range": (
                [min(timed_observed), max(timed_observed)] if timed_observed else None),
            "predicted_ms_range": [min(predicted), max(predicted)] if predicted else None,
            "worst_absolute_relative_error": worst,
            "failing_item_ids": [item["item_id"] for item in outside],
            "failing_sample_ids": [str(value) for value in failing_samples],
            "work_only_item_ids": [item["item_id"] for item in work_only],
            "work_only_reasons": sorted({
                str(item.get("comparison", {}).get("reason"))
                for item in work_only}),
            "status": (
                "contains_failures" if outside else
                "contains_work_only" if work_only and within else
                "work_only" if work_only else
                "all_items_within" if within else "empty"),
            "source_status": acceptance.get("source_status"),
            "partial_coverage": acceptance.get("partial_coverage"),
            "truncated": acceptance.get("truncated"),
            "aggregation": "per_item_verdicts_not_group_mean; predicted_range_and_error_from_timed_set",
        })
    return sorted(rows, key=lambda row: (
        row["status"] != "contains_failures",
        -(row["worst_absolute_relative_error"]
          if row["worst_absolute_relative_error"] is not None else -1.0),
        row["cohort"]["native_operator"],
        row["cohort"]["native_abi_sha256"] or ""))


def render_native_comparison(rows):
    if not rows:
        return ""
    lines = [
        "\n## Independent native cohorts\n",
        "Each row counts individual same-ABI item verdicts. A cohort mean, family "
        "average or signed cancellation is not a pass criterion; failing item IDs "
        "remain below. Work-only items keep observation/work facts but contribute "
        "no duration verdict or numerical error.\n",
        "| Operator | ABI (sha256) | Boundary | Items (within/outside/work-only) | "
        "Duration denominator | Samples | Observed ms range | Predicted ms range | "
        "Worst relative error | Status |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"

    def span(values):
        return (" .. ".join(number(value) for value in values)
                if values else "unavailable")

    for row in rows:
        cohort = row["cohort"]
        abi = (cohort["native_abi_sha256"] or "unavailable")
        if abi.startswith("sha256:"):
            abi = abi[7:19]
        lines.append(
            f"| {cohort['native_operator']} | {abi} | {cohort['boundary_kind']} | "
            f"{row['paired_items']} ({row['within']}/{row['outside']}/"
            f"{row['work_only']}) | "
            f"{row['duration_denominator_items']} | "
            f"{row['observation_samples']} | "
            f"{span(row['observed_ms_range'])} | "
            f"{span(row['predicted_ms_range'])} | "
            f"{number(row['worst_absolute_relative_error'])} | {row['status']} |")
    if rows and (rows[0].get("partial_coverage") or rows[0].get("truncated")):
        lines.append(
            "- Source status `" + str(rows[0].get("source_status")) + "`; "
            "`all_items_within` is local item evidence only, not a slice pass.")
    if any(row["work_only_item_ids"] for row in rows):
        lines.append(
            "- Work-only items have no predicted duration or formal error; they stay "
            "outside the duration denominator. Reasons: "
            + ", ".join(sorted({reason for row in rows
                                for reason in row["work_only_reasons"]})) + ".")
    if any(row["failing_item_ids"] for row in rows):
        lines.append("")
    for row in rows:
        if row["failing_item_ids"]:
            lines.append(f"- {row['cohort']['native_operator']} failing items: "
                         + ", ".join(row["failing_item_ids"]))
    return "\n".join(lines) + "\n"


import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

from ..coverage import source_identity


CONDITION_FIELDS = (
    "model", "hardware", "parallel", "phase", "tokens", "context", "sequences",
    "precision", "branch",
)
PARTITION_CONDITION_FIELDS = (
    "active_sequences", "padding_sequences",
)


def gap(predicted, observed):
    """Keep signed and absolute errors, with the observed value as denominator."""
    if not all(math.isfinite(value) and value >= 0 for value in (predicted, observed)):
        raise ValueError("scoreboard_times_must_be_finite_nonnegative")
    delta = predicted - observed
    return {
        "signed_gap_ms": delta, "absolute_gap_ms": abs(delta),
        "signed_gap_percent": 100 * delta / observed if observed else None,
        "absolute_gap_percent": 100 * abs(delta) / observed if observed else None,
    }


def _differences(observed, predicted, path=""):
    if isinstance(observed, dict) and isinstance(predicted, dict):
        return [
            item
            for key in sorted(observed.keys() | predicted.keys())
            for item in _differences(observed.get(key), predicted.get(key),
                                     f"{path}.{key}" if path else key)
        ]
    return [] if observed == predicted else [{
        "field": path, "observed": observed, "predicted": predicted,
    }]


def _unknowns(value, path):
    if value is None or value == {} or value == []:
        return [path]
    if isinstance(value, dict):
        return [item for key, nested in value.items()
                for item in _unknowns(nested, f"{path}.{key}")]
    if isinstance(value, (list, tuple)):
        return [item for index, nested in enumerate(value)
                for item in _unknowns(nested, f"{path}[{index}]")]
    return []


def _expected_rank_keys(left, right, *, use_sequences=True):
    if use_sequences:
        for side in (left, right):
            seq = side.get("sequences")
            if isinstance(seq, dict) and seq:
                return tuple(sorted(seq))
    for side in (left, right):
        parallel = side.get("parallel")
        if isinstance(parallel, dict):
            dp = parallel.get("dp")
            if isinstance(dp, int) and not isinstance(dp, bool) and dp > 0:
                return tuple(str(rank) for rank in range(dp))
    return None


def _aligned_condition_pair(left, right, key):
    lval, rval = left.get(key), right.get(key)
    if (key in ("sequences", *PARTITION_CONDITION_FIELDS)
            and isinstance(lval, dict) != isinstance(rval, dict)):
        rank_keys = _expected_rank_keys(
            left, right, use_sequences=(key != "sequences"))
        if rank_keys is not None:
            if not isinstance(lval, dict) and lval is not None:
                lval = {rank: lval for rank in rank_keys}
            if not isinstance(rval, dict) and rval is not None:
                rval = {rank: rval for rank in rank_keys}
    return lval, rval


def _step_comparison(entry):
    observed, predicted = entry["observed"], entry.get("predicted", {})
    if predicted.get("scenarios"):
        return _scenario_comparison(entry)
    measured, simulated = observed.get("step"), predicted.get("step")
    result = {
        "status": "unavailable",
        "observed_ms": measured.get("duration_ms") if measured else None,
        "predicted_ms": simulated.get("duration_ms") if simulated else None,
        "observed_boundary": measured.get("boundary") if measured else None,
        "predicted_boundary": simulated.get("boundary") if simulated else None,
        "signed_gap_ms": None, "absolute_gap_ms": None,
        "signed_gap_percent": None, "absolute_gap_percent": None,
        "formal_accuracy": "not_assessed",
    }
    if result["observed_ms"] is None or result["predicted_ms"] is None:
        result["reasons"] = (
            ([] if result["observed_ms"] is not None else
             observed.get("unavailable_reasons", ["measured_step_not_identified"]))
            + ([] if result["predicted_ms"] is not None else
               predicted.get("unavailable_reasons", ["matching_prediction_not_available"])))
        return result
    if predicted.get("identity_issues"):
        return {**result, "status": "incomparable",
                "reasons": predicted["identity_issues"]}
    left, right = observed.get("conditions", {}), predicted.get("conditions", {})
    fields = CONDITION_FIELDS + tuple(
        key for key in PARTITION_CONDITION_FIELDS
        if key in left or key in right
    )
    unknown = sorted({item for key in fields
                      for side in (left, right)
                      for item in _unknowns(side.get(key), key)})
    differences = [item for key in fields
                   for item in _differences(*_aligned_condition_pair(left, right, key), key)]
    if unknown or differences:
        return {**result, "status": "incomparable",
                "reasons": [f"unknown_condition:{key}" for key in unknown]
                           + [f"condition_mismatch:{item['field']}" for item in differences],
                "condition_differences": differences}
    if (not measured.get("comparison_scope")
            or measured["comparison_scope"] != simulated.get("comparison_scope")):
        return {**result, "status": "incomparable",
                "reasons": ["step_comparison_scope_mismatch"]}
    assumptions = list(entry.get("assumptions", ())) + list(predicted.get("assumptions", ()))
    if measured["boundary"] != simulated["boundary"]:
        relation = entry.get("boundary_relation", {})
        if (relation.get("observed") != measured["boundary"]
                or relation.get("predicted") != simulated["boundary"]
                or not relation.get("assumptions")):
            return {**result, "status": "incomparable",
                    "reasons": ["timing_boundary_relation_not_declared"]}
        assumptions.extend(relation["assumptions"])
    return {
        **result, "status": "conditional", "reasons": [],
        "comparison_scope": measured["comparison_scope"],
        "assumptions": assumptions,
        **gap(result["predicted_ms"], result["observed_ms"]),
    }


def _scenario_comparison(entry):
    """Retain explicit endpoint scenarios; do not invent a median context."""
    observed, predicted = entry["observed"], entry["predicted"]
    bounds = observed.get("context_range")
    unknown_context = observed.get("conditions", {}).get("context") is None
    rows = []
    for scenario in predicted["scenarios"]:
        context = scenario.get("conditions", {}).get("context")
        current = deepcopy(entry)
        current["predicted"] = scenario
        if unknown_context:
            current["observed"]["conditions"]["context"] = context
        pair = _step_comparison(current)
        pair["context"] = context
        rows.append(pair)
    valid = (unknown_context and bounds and len(rows) == 2
             and all(row["status"] == "conditional" for row in rows)
             and sorted(row["context"] for row in rows) == sorted(bounds))
    observed_ms = observed.get("step", {}).get("duration_ms")
    result = {
        "status": "conditional_range" if valid else "incomparable",
        "observed_ms": observed_ms, "predicted_ms": None,
        "signed_gap_ms": None, "absolute_gap_ms": None,
        "signed_gap_percent": None, "absolute_gap_percent": None,
        "formal_accuracy": "not_assessed", "scenarios": rows,
        "reasons": [] if valid else ["context_endpoint_scenarios_not_comparable"],
        "assumptions": list(entry.get("assumptions", ())),
        "interpretation": "two_context_endpoint_predictions_not_a_proven_interior_latency_bound",
    }
    qualifiers = {}
    for row in rows:
        for assumption in row.get("assumptions", ()):
            qualifiers.setdefault(assumption, []).append(row["context"])
        if row["status"] != "conditional":
            result["reasons"].extend(
                f"Context {row['context']}: {reason}" for reason in row.get("reasons", ()))
    for assumption, contexts in qualifiers.items():
        if assumption in result["assumptions"]:
            continue
        result["assumptions"].append(
            assumption if len(contexts) == len(rows) else
            f"Context {', '.join(str(context) for context in contexts)}: {assumption}")
    if valid:
        for field in ("predicted_ms", "signed_gap_ms", "absolute_gap_ms",
                      "signed_gap_percent", "absolute_gap_percent"):
            values = [row[field] for row in rows]
            result[field.replace("_ms", "_range_ms").replace("_percent", "_range_percent")] = [
                min(values), max(values)]
    hashes = sorted({scenario.get("runtime", {}).get("package_content_sha256", "unknown")
                     for scenario in predicted["scenarios"]})
    result["scenario_runtime_hashes"] = hashes
    if len(hashes) > 1:
        result["assumptions"].append(
            "Endpoint package hashes differ; the endpoint spread is not attributed solely to context.")
    return result


def _leaders(rows, top):
    ordered = sorted(rows, key=lambda row: (
        -max(row["duration_range_ms"]) if "duration_range_ms" in row
        else -row["duration_ms"], row["name"]))
    return deepcopy(ordered[:top])


def _predicted_leaders(predicted, top):
    scenarios = predicted.get("scenarios")
    if not scenarios:
        return _leaders(predicted.get("groups", ()), top)
    by_key = {}
    for index, scenario in enumerate(scenarios):
        for row in scenario.get("groups", ()):
            by_key.setdefault((row["name"], row.get("family")), []).append((index, row))
    leaders = []
    for (name, family), members in by_key.items():
        values = [row["duration_ms"] for _, row in members]
        leaders.append({
            "name": name, "family": family, "duration_range_ms": [min(values), max(values)],
            "statistic": ("explicit_context_endpoint_predictions"
                          if len(members) == len(scenarios) else
                          "partial_context_predictions_missing_endpoint"),
            "boundary": "sum_of_isolated_device_programs", "scope": "declared_model_probe",
            "scenario_indices": [index for index, _ in members],
            "scenario_coverage": f"{len(members)}/{len(scenarios)}",
        })
    return _leaders(leaders, top)


def _coarse_family_comparisons(entry, step):
    """Explicit family associations retain native grouping and statistic changes.

    This diagnostic level intentionally permits unequal native/program counts.
    It cannot claim tensor-exact accuracy or price mixed wrapper activity as a
    physical link. Membership and every selected statistic remain in the row.
    """
    observed = {row["name"]: row for row in entry["observed"].get("family_groups", ())}
    prediction = entry.get("predicted", {})
    scenarios = prediction.get("scenarios", [prediction])
    results = []
    for binding in entry.get("family_bindings", ()):
        left = observed.get(binding["observed_group"])
        row = {
            "comparison_key": binding["id"], "comparison_level": "coarse_semantic_family",
            "status": "unavailable", "formal_accuracy": "not_assessed",
            "observed_ms": left.get("duration_ms") if left else None,
            "predicted_ms": None, "signed_gap_ms": None, "absolute_gap_ms": None,
            "observed": deepcopy(left), "predicted_groups": binding["predicted_groups"],
            "assumptions": list(binding["assumptions"]),
        }
        members = []
        for scenario in scenarios:
            selected = [group for group in scenario.get("groups", ())
                        if group["name"] in binding["predicted_groups"]]
            missing = set(binding["predicted_groups"]) - {group["name"] for group in selected}
            members.append({
                "context": scenario.get("conditions", {}).get("context"),
                "duration_ms": math.fsum(group["duration_ms"] for group in selected) if not missing else None,
                "missing_groups": sorted(missing), "groups": deepcopy(selected),
            })
        row["prediction_scenarios"] = members
        values = [member["duration_ms"] for member in members]
        if values and all(value is not None for value in values):
            if len(values) == 1:
                row["predicted_ms"] = values[0]
            else:
                row["predicted_range_ms"] = [min(values), max(values)]
        if not left:
            row["reason"] = "observed_family_not_retained"
        elif any(value is None for value in values) or not values:
            row["reason"] = "predicted_family_members_not_retained"
        elif not left.get("comparable", True):
            row.update(status="incomparable", reason=left["reason"])
        elif step["status"] not in {"conditional", "conditional_range"}:
            row.update(status="incomparable", reason="parent_step_conditions_not_comparable")
        elif not binding.get("boundary_assumptions") or not left.get("statistic"):
            row.update(status="incomparable", reason="family_statistic_or_boundary_not_declared")
        else:
            row["assumptions"].extend(binding["boundary_assumptions"])
            residuals = [gap(value, left["duration_ms"]) for value in values]
            if len(values) == 1:
                row.update(status="conditional", **residuals[0])
            else:
                row["status"] = "conditional_range"
                for field in residuals[0]:
                    observations = [item[field] for item in residuals]
                    key = field.replace("_ms", "_range_ms").replace("_percent", "_range_percent")
                    row[key] = ([min(observations), max(observations)]
                                if all(item is not None for item in observations) else None)
        results.append(row)
    return results


def _group_comparisons(entry):
    """Pair only declared homogeneous group identities, never names alone."""
    observed, predicted = {}, {}
    for side, output in ((entry["observed"], observed), (entry.get("predicted", {}), predicted)):
        for row in side.get("groups", ()):
            key = row.get("comparison_key")
            if key is None:
                continue
            if key in output:
                raise ValueError(f"duplicate_scoreboard_group_identity:{key}")
            output[key] = row
    rows = []
    for key in sorted(observed.keys() | predicted.keys()):
        left, right = observed.get(key), predicted.get(key)
        row = {"comparison_key": key, "status": "unavailable",
               "observed": left, "predicted": right}
        if left and right:
            differences = _differences(left.get("identity"), right.get("identity"))
            if (_unknowns(left.get("identity"), "identity")
                    or _unknowns(right.get("identity"), "identity") or differences):
                row.update(status="incomparable", reason="group_contract_mismatch",
                           condition_differences=differences)
            elif not left.get("boundary") or left.get("boundary") != right.get("boundary"):
                row.update(status="incomparable", reason="group_boundary_mismatch")
            elif any(_unknowns(left.get(field), field)
                     or _unknowns(right.get(field), field)
                     or left[field] != right[field]
                     for field in ("statistic", "scope", "occurrences")):
                row.update(status="incomparable", reason="group_statistical_scope_mismatch")
            else:
                row.update(status="conditional",
                           **gap(right["duration_ms"], left["duration_ms"]))
        rows.append(row)
    return rows


def _semantic_family_comparisons(entry, step):
    """Coarse declared family mappings are useful without tensor-exact claims."""
    comparisons = []
    for binding in entry.get("semantic_family_comparisons", ()):
        row = {"comparison_key": binding["id"], "comparison_level": "semantic_family",
               "status": "incomparable", "assumptions": list(binding["assumptions"]),
               "formal_accuracy": "not_assessed"}
        selected = {}
        missing = []
        for side in ("observed", "predicted"):
            names = binding[side + "_groups"]
            candidates = entry.get(side, {}).get("groups", ())
            groups = [item for item in candidates if item["name"] in names]
            if not names or len(groups) != len(names) or {item["name"] for item in groups} != set(names):
                missing.append(side)
                continue
            selected[side] = groups
            row[side] = {
                "groups": names, "duration_ms": math.fsum(item["duration_ms"] for item in groups),
                "occurrences": (sum(item["occurrences"] for item in groups)
                                if all(item.get("occurrences") is not None for item in groups)
                                else None),
                "boundaries": sorted({item["boundary"] for item in groups}),
                "scope": step.get("comparison_scope"),
            }
        if missing:
            row.update(status="unavailable", reason="family_groups_missing:" + ",".join(missing))
        elif step["status"] != "conditional":
            row["reason"] = "parent_step_conditions_not_comparable"
        elif any(item.get("aggregation") != "sum" or item.get("occurrences") is None
                 for groups in selected.values() for item in groups):
            row["reason"] = "family_requires_summed_activity_and_explicit_occurrences"
        elif row["observed"]["occurrences"] != row["predicted"]["occurrences"]:
            row["reason"] = "family_occurrence_count_mismatch"
        else:
            relation = binding.get("boundary_relation", {})
            if (row["observed"]["boundaries"] != [relation.get("observed")]
                    or row["predicted"]["boundaries"] != [relation.get("predicted")]
                    or not relation.get("assumptions")):
                row["reason"] = "family_boundary_relation_not_declared"
            else:
                row.update(status="conditional", **gap(
                    row["predicted"]["duration_ms"], row["observed"]["duration_ms"]))
                row["assumptions"].extend(relation["assumptions"])
        comparisons.append(row)
    return comparisons


def model_scoreboard(entries, *, top=5):
    """Report actual coverage; never rank models or fill missing error with zero."""
    if top < 1:
        raise ValueError("scoreboard_top_must_be_positive")
    models, references = [], []
    for entry in entries:
        comparison = _step_comparison(entry)
        comparison.update(step_fields(comparison))
        groups = (_group_comparisons(entry) + _semantic_family_comparisons(entry, comparison)
                  + _coarse_family_comparisons(entry, comparison))
        for group in groups:
            group.update(group_fields(group))
        groups.sort(key=lambda row: (
            row["status"] not in {"conditional", "conditional_range"},
            -(row.get("absolute_gap_ms") or max(row.get("absolute_gap_range_ms") or [0])),
            row["comparison_key"]))
        destination = references if entry.get("role") == "reference" else models
        destination.append({
            "id": entry["id"], "label": entry.get("label", entry["id"]),
            "role": entry.get("role", "requested_model"),
            "observed": deepcopy(entry["observed"]),
            "predicted": deepcopy(entry.get("predicted", {})),
            "step_comparison": comparison,
            "observed_top": _leaders(entry["observed"].get("groups", ()), top),
            "predicted_top": _predicted_leaders(entry.get("predicted", {}), top),
            "group_comparisons": groups,
            "call_comparisons": [annotate_call(row) for row in entry.get("call_comparisons", ())],
            "sources": deepcopy(entry.get("sources", [])),
            "limitations": list(entry.get("limitations", [])),
        })
    compared = [row["step_comparison"] for row in models
                if row["step_comparison"]["status"] == "conditional"]
    ranged = sum(row["step_comparison"]["status"] == "conditional_range" for row in models)
    return {
        "schema": "mindie-sim.model-scoreboard/1",
        "accuracy_policy": policy(),
        "models": models,
        "references": references,
        "coverage": {
            "models": len(models), "conditional_step_comparisons": len(compared) + ranged,
            "unavailable_or_incomparable": len(models) - len(compared) - ranged,
        },
        "descriptive_residuals": {
            "scope": "single_value_step_comparisons_only_not_model_ranking",
            "range_models_reported_separately": ranged,
            "signed_gap_ms": math.fsum(row["signed_gap_ms"] for row in compared) if compared else None,
            "absolute_gap_ms": math.fsum(row["absolute_gap_ms"] for row in compared) if compared else None,
        },
        "interpretation": {
            "accuracy": "descriptive_conditioned_comparison_not_heldout_validation",
            "cost_leaders": "activity_and_isolated_costs_are_not_additive_wall_span_components",
            "missing": "unavailable_is_not_zero_error_and_never_blocks_analytic_simulation",
        },
    }


def prediction_from_probe(probe, *, conditions, comparison_scope):
    """Adapt the public convergence worker's actual graph and isolated ledger."""
    if probe.get("status") != "modeled":
        return {"status": probe.get("status", "unknown"), "conditions": conditions,
                "failure": deepcopy(probe), "unavailable_reasons": [
                    probe.get("error", probe.get("reason", "prediction_not_modeled"))]}
    graph, identity = probe.get("graph"), probe.get("execution_identity", {})
    actual = identity.get("conditions", {})
    issues = []
    if not identity:
        issues.append("prediction_execution_identity_not_retained")
    else:
        if identity.get("effective_case_key") != probe["effective_case_key"]:
            issues.append("prediction_execution_case_key_mismatch")
        if actual.get("parallel") != probe.get("parallel"):
            issues.append("prediction_execution_parallel_mismatch")
        issues.extend(f"prediction_label_mismatch:{item['field']}"
                      for item in _differences(conditions, actual))
    return {
        "status": "modeled",
        "conditions": deepcopy(actual), "claimed_conditions": deepcopy(conditions),
        "execution_identity": deepcopy(identity), "identity_issues": issues,
        "runtime": deepcopy(probe["runtime"]),
        "runtime_integrity": deepcopy(probe.get("runtime_integrity")),
        "assumptions": (
            ["Package sources changed during this probe; before/after hashes are retained."]
            if probe.get("runtime_integrity", {}).get("status") == "changed" else []),
        "case_key": probe["effective_case_key"],
        "step": ({"duration_ms": graph["duration_ms"], "boundary": graph["boundary"],
                  "comparison_scope": comparison_scope, "statistic": "deterministic_prediction"}
                 if graph else None),
        "isolated_sum_ms": probe["costs"]["operator_sum_ms"],
        "groups": [
            {"name": row["recipe"], "family": row["family"],
             "duration_ms": row["isolated_ms"], "occurrences": row["occurrences"],
             "boundary": "sum_of_isolated_device_programs",
             "statistic": "sum", "aggregation": "sum", "scope": "declared_model_probe"}
            for row in probe["costs"]["families"]
        ],
        "execution": ({key: graph[key] for key in ("live_tasks", "live_programs")}
                      if graph else {}),
        "unavailable_reasons": [] if graph else ["prediction_has_no_graph_span"],
    }


def render_markdown(report):
    def number(value):
        return "unavailable" if value is None else f"{value:.6f}"
    def interval(pair, key):
        span = pair.get(key.replace("_ms", "_range_ms").replace("_percent", "_range_percent"))
        return " .. ".join(number(value) for value in span) if span else number(pair.get(key))
    def per_call_table(items, key):
        values = [
            "", "Conditional per-call diagnostics are excluded from scored cohort acceptance.",
            "", "| Key | Observed us/call | Predicted us/call | Signed us/call | Absolute us/call | Conditional tier outcome / unavailable reason |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for item in items:
            tier = item.get("descriptive_tier")
            outcome = (tier["accuracy_status"] if tier else
                       item.get("per_call_ineligible_reason", "unscored"))
            values.append(
                f"| {item[key]} | "
                + " | ".join(number(item.get(field)) for field in (
                    "observed_per_call_us", "predicted_per_call_us",
                    "per_call_diff_us", "abs_error_per_call_us"))
                + f" | {outcome} |")
        return values
    lines = [
        "# Model timing scoreboard", "",
        "Descriptive comparisons only. Activity sums and isolated costs are not "
        "additive decompositions of a step span.", "",
        "Whole steps retain relative 15% tolerance. Homogeneous conditional calls use "
        "the shared observed-mean tiers: above 50 us, relative 15%; at or below 50 us, "
        "absolute 15 us. The small-call allowance does not establish measured launch "
        "overhead and never changes predicted time. References do not enter scored "
        "cohort acceptance.", "",
        "| Model | Observed step ms | Predicted graph ms | Signed gap ms | Absolute gap ms | Signed gap % | Absolute gap % | Status | Step 15% criterion |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in report["models"]:
        pair = row["step_comparison"]
        lines.append("| " + " | ".join([row["label"]] + [
            interval(pair, key) for key in ("observed_ms", "predicted_ms",
                                         "signed_gap_ms", "absolute_gap_ms", "signed_gap_percent",
                                         "absolute_gap_percent")
        ] + [pair["status"], pair.get("step_accuracy_status", "unscored")]) + " |")
    if report.get("references"):
        lines.extend(["", "Reference captures below are not replacements for requested models."])
    for row in report["models"] + report.get("references", []):
        pair = row["step_comparison"]
        prefix = "Reference: " if row["role"] == "reference" else ""
        lines.extend(["", "## " + prefix + row["label"], ""])
        if row["role"] == "reference":
            lines.append(f"Observed step: {number(pair['observed_ms'])} ms; comparison {pair['status']}.")
        lines.extend(f"- {reason}" for reason in pair.get("reasons", ()))
        lines.extend(f"- Assumption: {reason}" for reason in pair.get("assumptions", ()))
        lines.extend(f"- Limitation: {reason}" for reason in row["limitations"])
        for side in ("observed", "predicted"):
            lines.extend(["", f"### {side.title()} cost leaders", "",
                          "| Group | ms | Statistic | Activity domain | Boundary / scope | Scenario coverage |",
                          "| --- | ---: | --- | --- | --- | --- |"])
            for item in row[side + "_top"]:
                duration = (" .. ".join(number(value) for value in item["duration_range_ms"])
                            if "duration_range_ms" in item else number(item["duration_ms"]))
                lines.append(f"| {item['name']} | {duration} | "
                             f"{item.get('statistic', 'unspecified')} | "
                             f"{item.get('activity_domain', 'as_declared')} | "
                             f"{item.get('boundary', '')} / {item.get('scope', '')} | "
                             f"{item.get('scenario_coverage', 'not_scenario_based')} |")
            if not row[side + "_top"]:
                lines.append("| unavailable | unavailable | | | | |")
        if row["group_comparisons"]:
            lines.extend(["", "### Explicit group comparisons", "",
                          "| Key | Observed ms | Predicted ms | Signed gap ms | Absolute gap ms | Status / reason |",
                          "| --- | ---: | ---: | ---: | ---: | --- |"])
            for item in row["group_comparisons"]:
                values = {
                    **item,
                    "observed_ms": item.get("observed_ms", (item.get("observed") or {}).get("duration_ms")),
                    "predicted_ms": item.get("predicted_ms", (item.get("predicted") or {}).get("duration_ms")),
                }
                lines.append(
                    f"| {item['comparison_key']} | "
                    + " | ".join(interval(values, key) for key in (
                        "observed_ms", "predicted_ms", "signed_gap_ms", "absolute_gap_ms"))
                    + f" | {item['status']} {item.get('reason', '')} |")
            lines.append("")
            assumptions = {}
            for item in row["group_comparisons"]:
                for value in item.get("assumptions", ()):
                    assumptions.setdefault(value, []).append(item["comparison_key"])
            lines.extend(f"- {', '.join(keys)} assumption: {value}"
                         for value, keys in assumptions.items())
            lines.extend(per_call_table(row["group_comparisons"], "comparison_key"))
        else:
            lines.extend(["", "No homogeneous observed/predicted group pair is declared; "
                          "cost-leader names alone do not establish equal work."])
        if row["call_comparisons"]:
            lines.extend(["", "### Descriptor-conditioned calls and capture totals", "",
                          "These are native-family activity populations over the mixed capture, "
                          "not a reconstructed model step. Descriptor-conditioned predictions "
                          "use the actual retained occurrence count only for cumulative activity.",
                          "", "| Family | Calls | Observed mean / P50 ms | Predicted single / weighted mean ms | Observed cumulative ms | Predicted cumulative ms | Signed / absolute cumulative gap ms | Status |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"])
            for item in row["call_comparisons"]:
                error = item.get("sum_error") or {}
                lines.append(
                    f"| {item['family']} | {item['occurrences']} | "
                    f"{number(item['observed_mean_ms'])} / {number(item.get('observed_p50_ms'))} | "
                    f"{number(item.get('predicted_weighted_mean_ms'))} | "
                    f"{number(item['observed_sum_ms'])} | {number(item.get('predicted_sum_ms'))} | "
                    f"{number(error.get('signed_gap_ms'))} / "
                    f"{number(error.get('absolute_gap_ms'))} | {item['status']} |")
            lines.append("")
            for item in row["call_comparisons"]:
                lines.append(f"- {item['family']}: {item['assumption']}")
                if item.get("descriptor_absolute_residual_sum_ms") is not None:
                    lines.append(
                        f"- {item['family']} sum of absolute residuals across descriptor "
                        f"populations: {number(item['descriptor_absolute_residual_sum_ms'])} ms; "
                        "this also exposes cancellation within the cumulative family total.")
                if item.get("semantic_reference_ms") is not None:
                    lines.append(
                        f"- {item['family']} complete semantic reference: "
                        f"{number(item['semantic_reference_ms'])} ms; its native launch stage "
                        "is unresolved, so it is not multiplied by launch count.")
            lines.extend(per_call_table(row["call_comparisons"], "family"))
        for scenario in row["predicted"].get("scenarios", ()):
            context = scenario["conditions"]["context"]
            lines.extend(["", f"### Predicted context {context} cost leaders", "",
                          "| Group | Isolated sum ms | Occurrences |",
                          "| --- | ---: | ---: |"])
            for item in _leaders(scenario.get("groups", ()), 5):
                lines.append(f"| {item['name']} | {number(item['duration_ms'])} | "
                             f"{item['occurrences']} |")
        audit = row["observed"].get("communication_audit")
        if audit:
            lines.extend(["", "### Communication activity overlap audit", "",
                          "| Export domain | Raw events | Activity sum ms | Interval union ms |",
                          "| --- | ---: | ---: | ---: |"])
            for key in ("native_communication", "aicpu_activity"):
                item = audit[key]
                lines.append(f"| {key} | {item['raw_events']} | "
                             f"{number(item['raw_activity_sum_ms'])} | "
                             f"{number(item['interval_union_ms'])} |")
            overlap = audit["overlap"]
            lines.extend(["", f"Cross-domain intersection: "
                          f"{number(overlap['native_aicpu_intersection_ms'])} ms; "
                          f"either-domain union: {number(overlap['either_domain_interval_union_ms'])} ms.",
                          "", audit["duplicate_identity_audit"]["identity_limit"],
                          "No event was removed. Wait columns remain separate from duration; "
                          "AICPU activities are not additional physical link work."])
    return "\n".join(lines) + "\n"


def _read_json(path):
    path = Path(path)
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ValueError("native_scoreboard_metadata_exceeds_50_mib")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv=None, *, prog="python -m mindie_sim.observations.scoreboard"):
    """Publish one independent native comparison slice as JSON and markdown."""
    parser = argparse.ArgumentParser(
        prog=prog,
        description=("Project a profiler native comparison (formal/diagnostic "
                     "tolerance plus source coverage) into the sim scoreboard."))
    parser.add_argument(
        "comparison", type=Path, help="mindie-profiler-native-comparison/1 JSON")
    parser.add_argument("--inventory", type=Path, help="7337-ledger inventory summary JSON")
    parser.add_argument("--captures", type=Path, help="capture recovery summary JSON")
    parser.add_argument("--captures-manifest", type=Path,
                        help="T03 capture manifest JSON (actual version)")
    parser.add_argument("--ledger", type=Path, help="scope_map_7337.csv identity ledger")
    parser.add_argument("--records", type=Path, help="a5_records.csv recovery fields")
    parser.add_argument("--record-dispositions", type=Path,
                        help="T03 record_dispositions_7337.csv")
    parser.add_argument("--record-sources", type=Path,
                        help="T03 record_sources.csv")
    parser.add_argument("--source-registry", type=Path,
                        help="T03 source_registry.csv")
    parser.add_argument("--captures-index", type=Path,
                        help="T03 captures/index_all.csv")
    parser.add_argument("--source-edges", type=Path,
                        help="declared item -> source_id/path edge JSON")
    parser.add_argument("--label", help="campaign label retained in the report")
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args(argv)
    review = native_comparison_scoreboard(
        _read_json(args.comparison),
        inventory=args.inventory, captures=args.captures,
        ledger=args.ledger, records=args.records,
        captures_manifest=args.captures_manifest,
        record_dispositions=args.record_dispositions,
        record_sources=args.record_sources,
        source_registry=args.source_registry,
        captures_index=args.captures_index,
        source_edges=(_read_json(args.source_edges)
                      if args.source_edges is not None else None),
        label=args.label, source=source_identity(args.comparison))
    text = json.dumps(review, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if args.markdown:
        args.markdown.write_text(
            render_native_comparison_markdown(review), encoding="utf-8")
    return {"failed": 3, "incomplete": 4, "conditional": 4,
            "empty": 5}.get(review["status"], 0)


if __name__ == "__main__":
    raise SystemExit(main())
