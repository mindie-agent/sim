"""Descriptive same-boundary residuals beside independent accuracy validation.

[Role] Compare explicitly identified predictions with retained observations.
[Invariants] Descriptive matches never promote sample splits or certify accuracy.
  Unknown semantics remain unmatched; predictions always survive missing data.
[Timeouts & Locks] One observation pass; retain matching scalar durations only.
[Side-Effect Policy] Pure report construction; no calibration or source writes.
"""
from collections import Counter, defaultdict
from statistics import median

from .accuracy import _duration, _identity_key, accuracy_report


def comparison_report(campaign):
    """Keep all-split descriptive residuals distinct from heldout validation.

    Observations may be an iterator. An explicit prediction identity is the
    only join key; this function never finds a neighboring shape or recipe.
    """
    predictions = list(campaign["predictions"])
    selected = {_identity_key(point["identity"])[0] for point in predictions}
    groups, seen, boundaries = {}, set(), set()

    def collect():
        for sample in campaign.get("observations", ()):
            key, absent = _identity_key(sample["identity"])
            boundaries.add(sample["identity"].get("boundary"))
            if key in selected and not absent and sample["sample_id"] not in seen:
                seen.add(sample["sample_id"])
                group = groups.setdefault(key, {
                    "runs": defaultdict(list), "splits": Counter(), "source_preview": [],
                    "unresolved_semantics": Counter(),
                })
                unresolved = sample.get("metadata", {}).get("unresolved_semantics", ())
                if unresolved:
                    group["unresolved_semantics"].update(unresolved)
                else:
                    _duration(sample["duration_ms"], observation=True)
                    group["runs"][sample["run_id"]].append(sample["duration_ms"])
                    group["splits"][sample["split"]] += 1
                    if len(group["source_preview"]) < 5:
                        group["source_preview"].append({
                            "sample_id": sample["sample_id"], "source": sample["source"],
                            "split": sample["split"],
                        })
            yield sample

    validation = accuracy_report({**campaign, "predictions": predictions, "observations": collect()})
    points = []
    for prediction in predictions:
        key, absent = _identity_key(prediction["identity"])
        group = groups.get(key)
        point = {
            "point_id": prediction["point_id"], "identity": prediction["identity"],
            "predicted_ms": prediction["duration_ms"], "status": "unpaired",
            "comparison": None,
        }
        points.append(point)
        if absent:
            point.update(reason="incomplete_identity", missing_identity_fields=absent)
        elif not group:
            point["reason"] = "no_exact_observation_identity"
        elif not group["runs"]:
            point.update(reason="unresolved_observation_semantics",
                         unresolved_semantics=dict(group["unresolved_semantics"]))
        else:
            medians = {run: median(values) for run, values in group["runs"].items()}
            observed = median(medians.values())
            residual = prediction["duration_ms"] - observed
            point.update(
                status="descriptive_match", observed_ms=observed,
                statistic="median_of_run_medians",
                acquisition_runs=len(medians), samples=sum(group["splits"].values()),
                sample_splits=dict(group["splits"]), run_medians_ms=medians,
                observation_preview=group["source_preview"],
                unresolved_semantics=dict(group["unresolved_semantics"]),
                comparison={
                    "signed_residual_ms": residual, "absolute_residual_ms": abs(residual),
                    "relative_error": residual / observed,
                },
                interpretation="descriptive_all_split_comparison_not_independent_validation",
            )
    return {
        "schema": "mindie-sim.comparison-report/1",
        "descriptive": {
            "matched_points": sum(point["status"] == "descriptive_match" for point in points),
            "unpaired_points": sum(point["status"] == "unpaired" for point in points),
            "available_boundaries": sorted(value for value in boundaries if value is not None),
            "points": points,
            "interpretation": (
                "Descriptive residuals include original unassigned/training observations. "
                "They have no tolerance or acceptance result and do not establish generalization."
            ),
        },
        "validation": validation,
    }
