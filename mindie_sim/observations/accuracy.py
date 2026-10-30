"""Independent accuracy, scaling-trend and configuration-order comparisons.

[Role] Compare declared operator, subgraph or model points with heldout timings.
[Invariants] Identities and boundaries match exactly; training data never score
  validation. Signed cancellation cannot hide absolute residual. Empty coverage
  is unassessed, never a pass, and never disables an analytical prediction.
[Timeouts & Locks] Finite input points/samples; bounded ranking counterexamples.
[Side-Effect Policy] Pure report construction; no fitting, simulation or I/O.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import json
import math
from statistics import median


IDENTITY_FIELDS = (
    "level", "boundary", "hardware", "implementation", "phase", "shape",
    "dtype", "layout", "execution",
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _identity_key(identity):
    absent = [field for field in IDENTITY_FIELDS if not identity.get(field)]
    if absent:
        return None, sorted(set(absent))
    if identity["level"] not in {"operator", "subgraph", "model"}:
        raise ValueError("accuracy_level_requires_operator_subgraph_or_model")
    return _canonical(identity), []


def _tolerance(values):
    for name, value in values.items():
        if name not in {"absolute_ms", "relative", "order_absolute_ms", "order_relative"}:
            raise ValueError(f"unknown_accuracy_tolerance:{name}")
        if not math.isfinite(value) or value < 0:
            raise ValueError("accuracy_tolerances_must_be_finite_nonnegative")
    return dict(values)


def _duration(value, *, observation=False):
    if not math.isfinite(value) or value < 0 or (observation and value == 0):
        raise ValueError("accuracy_duration_must_be_finite_and_positive_for_observations")
    return value


def _bound(observed, tolerance):
    if not {"absolute_ms", "relative"} & tolerance.keys():
        return None
    return max(tolerance.get("absolute_ms", 0), tolerance.get("relative", 0) * observed)


def _direction(first, second, tolerance):
    tie = max(tolerance.get("order_absolute_ms", 0),
              tolerance.get("order_relative", 0) * max(first, second))
    delta = second - first
    return 0 if abs(delta) <= tie else (1 if delta > 0 else -1)


def _order_pair(first, second, tolerance):
    observed = _direction(first["observed_ms"], second["observed_ms"], tolerance)
    predicted = _direction(first["predicted_ms"], second["predicted_ms"], tolerance)
    status = ("observed_tie" if not observed else "predicted_tie" if not predicted
              else "agrees" if observed == predicted else "reversed")
    return {
        "first": first["point_id"], "second": second["point_id"],
        "observed_direction": observed, "predicted_direction": predicted,
        "status": status,
    }


def _at(identity, path):
    value = identity
    for name in path.split("."):
        value = value[name]
    return value


def _fixed_domain(identity, axes):
    value = deepcopy(identity)
    for path in axes:
        parts = path.split(".")
        parent = value
        for name in parts[:-1]:
            parent = parent[name]
        del parent[parts[-1]]
    return _canonical(value)


def _relation(specification, points, tolerance):
    ids = specification["points"]
    rows = [points[name] for name in ids]
    result = {
        "id": specification["id"], "kind": specification["kind"], "points": ids,
        "order_tolerance": {name: tolerance.get(name, 0)
                            for name in ("order_absolute_ms", "order_relative")},
    }
    missing = [row["point_id"] for row in rows if row["status"] != "paired"]
    if missing or len(rows) < 2:
        return {**result, "status": "not_assessed", "unpaired_points": missing,
                "reason": "requires_at_least_two_fully_paired_heldout_points"}
    kind = specification["kind"]
    axes = [specification["axis"]] if kind == "trend" else specification["vary"]
    if (not axes or any(axis.split(".")[0] in {"level", "boundary", "phase"} for axis in axes)
            or (kind == "trend" and not axes[0].startswith("shape."))
            or (kind == "ranking" and any(axis == "shape" or axis.startswith("shape.")
                                          for axis in axes))):
        raise ValueError("accuracy_relation_axes_must_preserve_boundary_phase_and_ranking_workload")
    if len({_fixed_domain(row["identity"], axes) for row in rows}) != 1:
        return {**result, "status": "domain_mismatch", "varying_axes": axes,
                "reason": "identities_differ_outside_declared_axes"}
    result["varying_axes"] = axes
    if kind == "trend":
        rows = sorted(rows, key=lambda row: _at(row["identity"], axes[0]))
        coordinates = [_at(row["identity"], axes[0]) for row in rows]
        if (any(not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in coordinates) or len(set(coordinates)) != len(coordinates)):
            raise ValueError("accuracy_trend_requires_distinct_finite_numeric_coordinates")
        comparisons = []
        for left, right in zip(rows, rows[1:]):
            pair = _order_pair(left, right, tolerance)
            observed_ratio = right["observed_ms"] / left["observed_ms"]
            predicted_ratio = (right["predicted_ms"] / left["predicted_ms"]
                               if left["predicted_ms"] else None)
            pair.update({
                "observed_ratio": observed_ratio, "predicted_ratio": predicted_ratio,
                "ratio_relative_error": (predicted_ratio / observed_ratio - 1
                                         if predicted_ratio is not None else None),
            })
            comparisons.append(pair)
        counts = Counter(pair["status"] for pair in comparisons)
        result["segments"] = comparisons
    elif kind == "ranking":
        counts, differences = Counter(), []
        for index, left in enumerate(rows):
            for right in rows[index + 1:]:
                pair = _order_pair(left, right, tolerance)
                counts[pair["status"]] += 1
                if pair["status"] != "agrees" and len(differences) < 20:
                    differences.append(pair)
        result.update({
            "observed_order": [row["point_id"] for row in sorted(rows, key=lambda row: row["observed_ms"])],
            "predicted_order": [row["point_id"] for row in sorted(rows, key=lambda row: row["predicted_ms"])],
            "counterexample_preview": differences,
        })
    else:
        raise ValueError(f"unknown_accuracy_relation_kind:{kind}")
    result["pair_outcomes"] = dict(counts)
    result["status"] = ("disagrees" if counts["reversed"] or counts["predicted_tie"]
                        else "unresolved_ties" if counts["observed_tie"] else "agrees")
    return result


def accuracy_report(campaign):
    """Compare independent records; input acceptance flags are never consumed."""
    tolerance = _tolerance(campaign.get("tolerances", {}))
    trained = set(campaign.get("training_sample_ids", ()))
    observations, excluded, seen = defaultdict(list), Counter(), {}
    observed_keys, heldout_keys = Counter(), set()
    for sample in campaign.get("observations", ()):
        key, absent = _identity_key(sample["identity"])
        sample_id = sample["sample_id"]
        unresolved = tuple(sorted(sample.get("metadata", {}).get("unresolved_semantics") or ()))
        signature = (key, sample["duration_ms"], sample["run_id"], sample["split"],
                     sample["source"], unresolved)
        if sample_id in seen:
            if signature != seen[sample_id]:
                raise ValueError(f"conflicting_accuracy_sample_or_split:{sample_id}")
            continue
        seen[sample_id] = signature
        if key is not None:
            observed_keys[key] += 1
        if absent:
            excluded["incomplete_identity"] += 1
        elif sample.get("metadata", {}).get("unresolved_semantics"):
            excluded["unresolved_semantics"] += 1
        elif sample["split"] != "heldout":
            excluded[f"split_{sample['split']}"] += 1
        elif sample_id in trained:
            excluded["used_for_training"] += 1
        else:
            _duration(sample["duration_ms"], observation=True)
            observations[key].append(sample)
            heldout_keys.add(key)
    points, domains, selected_keys = {}, {}, set()
    for prediction in campaign["predictions"]:
        point_id, identity = prediction["point_id"], prediction["identity"]
        if point_id in points:
            raise ValueError(f"duplicate_accuracy_prediction_point:{point_id}")
        key, absent = _identity_key(identity)
        predicted = _duration(prediction["duration_ms"])
        occurrences = prediction.get("occurrences", 1)
        if not isinstance(occurrences, int) or occurrences < 1:
            raise ValueError("accuracy_occurrences_must_be_positive_integer")
        paired = observations.get(key, ())
        point = {
            "point_id": point_id, "identity": identity, "predicted_ms": predicted,
            "occurrences": occurrences, "status": "paired" if paired else "unpaired",
            "comparison": None,
        }
        points[point_id] = point
        if not paired:
            point["reason"] = ("incomplete_identity" if absent else
                               "no_eligible_heldout_samples" if observed_keys[key]
                               else "no_matching_observation")
            point["missing_identity_fields"] = absent
            continue
        selected_keys.add(key)
        runs = defaultdict(list)
        for sample in paired:
            runs[sample["run_id"]].append(sample["duration_ms"])
        run_medians = {run: median(values) for run, values in runs.items()}
        observed = median(run_medians.values())
        error = predicted - observed
        bound = _bound(observed, tolerance)
        point.update({
            "observed_ms": observed, "statistic": "median_of_run_medians",
            "heldout_samples": len(paired), "heldout_runs": len(runs),
            "run_medians_ms": run_medians,
            "run_relative_error_range": [
                min(predicted / duration - 1 for duration in run_medians.values()),
                max(predicted / duration - 1 for duration in run_medians.values()),
            ],
            "sample_relative_error_range": [
                min(predicted / sample["duration_ms"] - 1 for sample in paired),
                max(predicted / sample["duration_ms"] - 1 for sample in paired),
            ],
            "comparison": {
                "signed_residual_ms": error, "absolute_residual_ms": abs(error),
                "relative_error": error / observed,
                "repeated_signed_residual_ms": occurrences * error,
                "repeated_absolute_residual_ms": occurrences * abs(error),
                "allowed_error_ms": bound,
                "within_tolerance": abs(error) <= bound if bound is not None else None,
            },
            "observation_samples": [{
                name: sample[name] for name in
                ("sample_id", "run_id", "duration_ms", "source", "metadata") if name in sample
            } for sample in paired],
        })
        domain = {name: value for name, value in identity.items() if name != "shape"}
        aggregate = domains.setdefault(_canonical(domain), {
            "identity": domain, "paired_points": 0, "occurrences": 0,
            "signed_residual_ms": 0.0, "absolute_residual_ms": 0.0,
            "selected_observed_sum_ms": 0.0,
        })
        aggregate["paired_points"] += 1
        aggregate["occurrences"] += occurrences
        aggregate["signed_residual_ms"] += occurrences * error
        aggregate["absolute_residual_ms"] += occurrences * abs(error)
        aggregate["selected_observed_sum_ms"] += occurrences * observed
    paired_points = [row for row in points.values() if row["status"] == "paired"]
    verdicts = Counter(row["comparison"]["within_tolerance"] for row in paired_points)
    return {
        "schema": "mindie-sim.accuracy-report/1",
        "tolerances": tolerance,
        "coverage": {
            "predicted_points": len(points), "paired_heldout_points": len(paired_points),
            "unpaired_predictions": len(points) - len(paired_points),
            "unpaired_heldout_identities": len(heldout_keys - selected_keys),
            "excluded_observations": dict(excluded),
        },
        "tolerance_outcomes": {
            "within": verdicts[True], "outside": verdicts[False], "undeclared": verdicts[None],
        },
        "points": list(points.values()), "domain_residuals": list(domains.values()),
        "relations": [_relation(specification, points, tolerance)
                      for specification in campaign.get("relations", ())],
        "interpretation": (
            "Residual sums cover only selected paired cases and declared occurrences; "
            "they are not whole-model accuracy, a confidence interval or graph critical-path error. "
            "No prediction is disabled and no fitting occurs."
        ),
        "provenance": campaign.get("provenance", {}),
        "limits": campaign.get("limits", []),
    }
