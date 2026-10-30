"""Keep selected native-call evidence beside, but outside, model-step totals.

[Role] Summarize existing cohort comparisons for the model scoreboard.
[Invariants] No prediction is executed or adjusted. Captures, ranks, windows and
  variants are never pooled. Per-descriptor failures survive family cancellation.
  Reference rows cannot enter the compared-cohort pass denominator.
[Timeouts & Locks] Linear in a bounded supplied cohort report; no I/O or locks.
[Side-Effect Policy] Returns fresh report records; no mutation of source data.
"""
from collections import Counter

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
