"""Rank existing operator reports without changing predictions.

[Role] Expose repeated cost and conditional residual contributions.
[Invariants] Each report keeps its own denominator and matching limits. Signed
  cancellation never substitutes for absolute residual; unpaired is not zero.
[Timeouts & Locks] One bounded report read at a time; no locks or simulation.
[Side-Effect Policy] Reads explicit reports and writes only requested output.
"""
from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

from mindie_sim.__main__ import emit, read_json


def percent(value, total):
    return 100 * value / total if total else None


def cost_ranking(report):
    """Family totals prioritize inspection, while original rows retain identity."""
    groups = {}
    for index, row in enumerate(report["operators"]):
        op = row["operator"]
        key = op["family"], op["recipe"]
        group = groups.setdefault(key, {
            "family": key[0], "recipe": key[1], "operator_indices": [],
            "occurrences": 0, "total_ms": 0.0,
        })
        group["operator_indices"].append(index)
        group["occurrences"] += row["occurrences"]
        group["total_ms"] += row["total_ms"]
    total = report["operator_sum_ms"]
    cumulative = 0.0
    ranked = sorted(groups.values(), key=lambda row: -row["total_ms"])
    for row in ranked:
        cumulative += row["total_ms"]
        row["share_percent"] = percent(row["total_ms"], total)
        row["cumulative_share_percent"] = percent(cumulative, total)
    return {
        "kind": "predicted_cost",
        "boundary": report["boundary"],
        "graph_scope": report.get("graph_scope"),
        "tokens_per_sequence": report.get("tokens_per_sequence"),
        "sequences": report.get("sequences"),
        "reference_sum_ms": total,
        "operator_occurrences": report["operator_occurrences"],
        "ranked_families": ranked,
        "interpretation": "Predicted isolated cost share, not measured error or graph sensitivity.",
    }


def residual_ranking(report):
    """Project paired medians to modeled repetitions; exclude ambiguous matches."""
    total = report["operator_sum_ms"]
    rows = []
    for index, pair in enumerate(report["comparisons"]):
        count = pair["model_calls"]
        predicted = pair["predicted_us"] * count / 1000
        observed = pair["observed_median_us"] * count / 1000
        delta = predicted - observed
        # The comparator may produce a Cartesian set of candidate matches.
        # Only one predicted variant and one captured signature can be summed.
        unique = (count == pair["contract_model_calls"]
                  and pair["capture_calls"] == pair["contract_capture_calls"])
        rows.append({
            "comparison_index": index,
            "recipe": pair["recipe"], "shape": pair["shape"], "dtype": pair["dtype"],
            "source_lines": pair["source_lines"], "match": pair["match"],
            "model_calls": count, "capture_calls": pair["capture_calls"],
            "one_to_one_contract": unique,
            "predicted_contribution_ms": predicted,
            "observed_median_times_model_calls_ms": observed,
            "operator_error_percent": percent(
                pair["predicted_us"] - pair["observed_median_us"], pair["observed_median_us"]),
            "signed_residual_ms": delta, "absolute_residual_ms": abs(delta),
            "absolute_residual_percent_of_reference": percent(abs(delta), total),
        })
    paired = [row for row in rows if row["one_to_one_contract"]]
    paired_ms = math.fsum(row["predicted_contribution_ms"] for row in paired)
    signed = math.fsum(row["signed_residual_ms"] for row in paired) if paired else None
    absolute = math.fsum(row["absolute_residual_ms"] for row in paired) if paired else None
    return {
        "kind": "conditional_residual",
        "boundary": report["boundary"],
        "model": report["model"], "hardware": report["hardware"],
        "software": report["software"],
        "phase": report["phase"], "tokens_per_sequence": report["tokens_per_sequence"],
        "sequences": report["sequences"], "selected_families": report["selected_families"],
        "reference_sum_ms": total,
        "one_to_one_pairs": len(paired),
        "ambiguous_pairs_excluded_from_sum": len(rows) - len(paired),
        "paired_predicted_ms": paired_ms,
        "paired_predicted_share_percent": percent(paired_ms, total),
        "unpaired_or_ambiguous_predicted_ms": total - paired_ms,
        "signed_residual_ms": signed,
        "absolute_residual_ms": absolute,
        "absolute_residual_percent_of_reference": (
            percent(absolute, total) if absolute is not None else None),
        "ranked_pairs": sorted(rows, key=lambda row: -row["absolute_residual_ms"]),
        "interpretation": (
            "Conditional kernel-median residual times modeled calls. One-to-one tensor "
            "matching does not establish dispatch identity. The denominator is this "
            "comparison's selected isolated sum, not another report's model total. "
            "No whole-model accuracy, uncertainty bound or critical-path sensitivity is inferred."
        ),
    }


def summarize(report):
    if report["schema"] == "mindie-sim.operators/1":
        return cost_ranking(report)
    if report["schema"] == "mindie-sim.operator-comparison/1":
        return residual_ranking(report)
    raise ValueError("requires_operator_cost_or_comparison_report")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reports", nargs="+")
    parser.add_argument("-o", "--output")
    args = parser.parse_args(argv)
    summaries = []
    for name in args.reports:
        path = Path(name)
        result = summarize(read_json(path))
        with path.open("rb") as stream:
            result["input_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        summaries.append({"input": str(path), **result})
    emit({"schema": "mindie-sim.modeling-priority/1", "reports": summaries}, args.output)


if __name__ == "__main__":
    main()
