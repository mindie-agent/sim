"""Observed vs predicted accuracy comparison and historical baselines.

[Role] Compute signed/absolute error metrics, compare predicted vs measured
  traces, and load historical accuracy baselines.
[Invariants] Transparent error reporting; never hides missing captures.
[Timeouts & Locks] In-memory evaluation; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from statistics import median

from .captures import digest_file, json_items, json_top_fields


def _header(path):
    wanted = {"schema", "env", "geometry", "trials", "samples_per_condition", "warmup"}
    return json_top_fields(path, wanted)


def _checked_source(root, reference):
    path = Path(root) / reference["path"]
    actual = digest_file(path)
    if actual != reference["sha256"]:
        raise ValueError(f"historical_source_changed:{path}")
    return path


def call_observations(protocol, *, root="."):
    """Yield actual call rows under a reviewed producer protocol, including failures.

    A protocol supplies interpretation absent from the capture, tied to the
    exact producer hash. It never supplies timing, model correction or split.
    """
    path = _checked_source(root, protocol["capture"])
    _checked_source(root, protocol["producer"])
    for reference in protocol.get("metadata_sources", ()):
        _checked_source(root, reference)
    header = _header(path)
    if header.get("schema") != protocol["capture_schema"]:
        raise ValueError("historical_capture_schema_does_not_match_protocol")
    trials = header["trials"]
    for index, row in enumerate(json_items(path, "rows.item")):
        role = row["role"]
        contract = protocol["calls"][role]
        conditions = {key: value for key, value in row.items()
                      if key not in {"samples_ms", "canary", "model_geometry", "status",
                                     "error_type", "error_first_line", "trace_tail"}
                      and not key.startswith("stat_")}
        geometry = header["geometry"].get(row.get("model_geometry"))
        identity = {
            "hardware": protocol["hardware"],
            "implementation": row["impl"],
            "producer_sha256": protocol["producer"]["sha256"],
            "conditions": conditions,
            "geometry": geometry,
            "call_contract": contract,
        }
        observation = {
            "schema": "mindie-sim.operator-call-observation/1",
            "observation_id": f"{protocol['capture']['sha256']}:{index}",
            "source": f"{path}#/rows/{index}",
            "source_sha256": protocol["capture"]["sha256"],
            "producer": protocol["producer"],
            "boundary": protocol["boundary"],
            "identity": identity,
            "capture_environment": header["env"],
            "model_geometry_label": row.get("model_geometry"),
            "status": row["status"],
            "canary": row.get("canary"),
            "split": "unassigned",
        }
        if row["status"] != "measured":
            observation["failure"] = {key: row[key] for key in
                                      ("error_type", "error_first_line", "trace_tail")
                                      if key in row}
            observation["samples"] = []
            observation["statistics"] = None
            yield observation
            continue
        samples = row["samples_ms"]
        if (not samples or any(not math.isfinite(value) or value <= 0 for value in samples)
                or len(samples) != row["stat_n"] or len(samples) % trials):
            raise ValueError(f"historical_event_samples_conflict:{path}#/rows/{index}")
        per_trial = len(samples) // trials
        # The retained producer concatenates equal-sized trials. An event is
        # a call within a trial, not an independently restarted experiment.
        observation["samples"] = [
            {"trial": number // per_trial, "repeat": number % per_trial, "duration_ms": value}
            for number, value in enumerate(samples)
        ]
        observation["statistics"] = {
            "sample_count": len(samples), "trial_count": trials,
            "sample_median_ms": median(samples),
            "median_of_trial_medians_ms": median(
                median(samples[start:start + per_trial])
                for start in range(0, len(samples), per_trial)),
            "minimum_ms": min(samples), "maximum_ms": max(samples),
        }
        # This producer used the upper central order statistic for an even
        # sample count. Preserve it separately from the normalized median.
        observation["reported_statistics"] = {
            key.removeprefix("stat_"): value for key, value in row.items()
            if key.startswith("stat_")
        }
        yield observation


def compare_call(prediction, observation):
    """Compare an explicitly paired program only at an identical call boundary."""
    differences = []
    for field in ("observation_id", "boundary", "identity"):
        if prediction.get(field) != observation[field]:
            differences.append(field)
    if observation["status"] != "measured":
        differences.append("observation_not_measured")
    result = {
        "observation_id": observation["observation_id"], "source": observation["source"],
        "comparable": not differences, "mismatches": differences,
        "statistic": "median_of_trial_medians_ms", "comparison": None,
    }
    if differences:
        return result
    predicted = prediction["duration_ms"]
    if not math.isfinite(predicted) or predicted < 0:
        raise ValueError("historical_comparison_requires_finite_nonnegative_prediction")
    observed = observation["statistics"]["median_of_trial_medians_ms"]
    result["comparison"] = {
        "predicted_ms": predicted, "observed_ms": observed,
        "error_ms": predicted - observed, "relative_error": predicted / observed - 1,
    }
    return result


def call_inventory(protocol, *, root=".", preview_limit=5):
    """Summarize readable conditions without accepting calibration or accuracy."""
    roles, statuses, canaries = Counter(), Counter(), Counter()
    events = conditions = 0
    rows = []
    for row in call_observations(protocol, root=root):
        conditions += 1
        roles[row["identity"]["conditions"]["role"]] += 1
        statuses[row["status"]] += 1
        canary = row["canary"]
        canaries["unknown" if not canary or "ok" not in canary
                 else "passed" if canary["ok"] else "failed"] += 1
        events += len(row["samples"])
        if len(rows) < preview_limit:
            rows.append({
                "observation_id": row["observation_id"], "source": row["source"],
                "role": row["identity"]["conditions"]["role"],
                "status": row["status"], "boundary": row["boundary"],
                "statistics": row["statistics"],
                "reported_median_ms": row.get("reported_statistics", {}).get("median"),
            })
    return {
        "capture": protocol["capture"], "condition_rows": conditions,
        "event_samples": events, "roles": dict(roles), "statuses": dict(statuses),
        "original_canary_outcomes": dict(canaries), "observation_preview": rows,
        "unobserved_roles": sorted(set(protocol["calls"]) - set(roles)),
        "calibration_bindings": 0, "prediction_comparisons": 0,
        "acceptance": "reader_and_call_boundary_only",
    }


from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
import json
import math
from statistics import median

from ..cohort_accuracy import (
    NATIVE_COMPARISON_SCHEMA, native_acceptance, native_acceptance_policy,
)


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


NATIVE_ACCURACY_SCHEMA = "mindie-sim.native-accuracy-report/1"


def native_accuracy_report(comparison):
    """Project the profiler's independent native comparison into accuracy rows.

    Every point keeps the source prediction, observation range, per-sample
    verdicts and failure status.  A mean or a median cannot replace a sample.
    """
    acceptance = native_acceptance(comparison)
    counts = acceptance["counts"]
    points = []
    for item in acceptance["items"]:
        if item["status"] == "work_only":
            points.append({
                "point_id": item["item_id"],
                "identity": item["identity"],
                "boundary": item["boundary"],
                "predicted_ms": None,
                "observed_ms": item["observed_ms"],
                "observation_range_ms": item["observation_range_ms"],
                "observation_samples": item["observation_samples"],
                "statistic": item.get("statistic"),
                "status": "work_only",
                "duration_available": False,
                "within_tolerance": None,
                "relative_error": None,
                "absolute_residual_ms": None,
                "formal_allowed_error_ms": None,
                "condition_gaps": item["condition_gaps"],
                "samples": [],
                "work": deepcopy(item.get("work")),
                "unresolved_reason": item.get("comparison", {}).get("reason"),
                "interpretation": "work_only_retained_not_scored_no_duration_or_error",
            })
            continue
        points.append({
            "point_id": item["item_id"],
            "identity": item["identity"],
            "boundary": item["boundary"],
            "predicted_ms": item["predicted_ms"],
            "observed_ms": item["observed_ms"],
            "observation_range_ms": item["observation_range_ms"],
            "observation_samples": item["observation_samples"],
            "statistic": item.get("statistic"),
            "status": "paired",
            "duration_available": True,
            "within_tolerance": item["within_tolerance"],
            "relative_error": item["relative_error"],
            "absolute_residual_ms": item["absolute_residual_ms"],
            "formal_allowed_error_ms": item["formal_allowed_error_ms"],
            "condition_gaps": item["condition_gaps"],
            "samples": item["samples"],
            "interpretation": "per_item_native_acceptance_not_heldout_generalization",
        })
    return {
        "schema": NATIVE_ACCURACY_SCHEMA,
        "source_schema": acceptance["source_schema"],
        "source_status": acceptance["source_status"],
        "status": acceptance["status"],
        "policy": native_acceptance_policy(),
        "tolerances": acceptance["tolerance"],
        "coverage": {
            "paired_items": counts["paired_items"],
            "duration_denominator_items": counts["duration_denominator_items"],
            "work_only_items": counts["work_only"],
            "observation_samples": counts["observation_samples"],
            "condition_gap_items_in_denominator": counts[
                "condition_gap_items_in_denominator"],
            "unpaired_observations": counts["unpaired_observations"],
            "unpaired_predictions": counts["unpaired_predictions"],
            "ambiguous": counts["ambiguous"],
            "source_status": acceptance["source_status"],
            "partial_coverage": acceptance["partial_coverage"],
            "truncated": acceptance["truncated"],
            "coverage_reconciled": acceptance["coverage_reconciled"],
            "subset_scope": deepcopy(acceptance["subset_scope"]),
            "parent_capture": acceptance["parent_capture"],
            "window_count": len(acceptance.get("windows") or ()),
            "source_identity": acceptance["source_identity"],
            "structure_accuracy": "not_computed",
        },
        "source_coverage": deepcopy(acceptance["coverage"]),
        "tolerance_outcomes": {
            "within": counts["within"],
            "outside": counts["outside"],
            "work_only": counts["work_only"],
            "duration_denominator_items": counts["duration_denominator_items"],
            "condition_gap_items_in_denominator": counts[
                "condition_gap_items_in_denominator"],
            "unpaired_observations": counts["unpaired_observations"],
            "unpaired_predictions": counts["unpaired_predictions"],
            "ambiguous": counts["ambiguous"],
            "source_status": acceptance["source_status"],
            "partial_coverage": acceptance["partial_coverage"],
            "truncated": acceptance["truncated"],
        },
        "points": points,
        "unpaired_observations": acceptance["unpaired_observations"],
        "unpaired_predictions": acceptance["unpaired_predictions"],
        "ambiguous": acceptance["ambiguous"],
        "failure_matrix": acceptance["failure_matrix"],
        "non_claims": list(acceptance["non_claims"]) + [
            "this_report_is_not_a_7337_record_or_heldout_generalization_result",
            "work_only_items_have_no_predicted_duration_or_numerical_error",
        ],
        "limits": [
            "ABI/boundary pairs are retained for all items; only items with a real "
            "predicted duration enter the tolerance denominator.",
            "Work-only items retain observation and work facts with no duration error; "
            "they are not counted as within or outside.",
            "Unpaired and ambiguous sides are retained outside the tolerance denominator.",
            "Source partial/truncated/inconclusive coverage remains incomplete; local "
            "all-within item verdicts do not make the slice passed.",
            "Diagnostic tolerance is retained separately and never changes formal status.",
            "Historical absolute_15us, in-sample and mean reproductions are excluded.",
        ],
    }


def accuracy_report(campaign):
    """Compare independent records; input acceptance flags are never consumed."""
    if isinstance(campaign, Mapping) and campaign.get("schema") == NATIVE_COMPARISON_SCHEMA:
        return native_accuracy_report(campaign)
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


_CLOCK_TELEMETRY_KEYS = (
    "observed_aicore_freq_mhz",
    "aicore_freq_mhz",
    "device_aicore_freq_mhz",
)


def extract_observed_aicore_freq_mhz(envelope: Mapping | None) -> float | None:
    """Extract observed AICore clock (MHz) from a silicon capture or msprof envelope."""
    if not isinstance(envelope, Mapping):
        return None
    payload = envelope.get("payload")
    silicon = payload.get("observed_silicon") if isinstance(payload, Mapping) else None
    containers = [container for container in (silicon, envelope, payload) if isinstance(container, Mapping)]
    for sub in ("observed_silicon", "device", "hardware", "metadata", "boundary", "telemetry", "clock"):
        nested = envelope.get(sub)
        if isinstance(nested, Mapping):
            containers.append(nested)
    for container in containers:
        for key in _CLOCK_TELEMETRY_KEYS:
            raw = container.get(key)
            if raw is None or isinstance(raw, bool):
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if math.isfinite(val) and val > 0.0:
                return val
    return None


def apply_capture_observed_clock(
    case_dict: Mapping, envelope: Mapping | None,
) -> dict:
    """Return a copy of ``case_dict`` aligned with any capture-observed AICore clock."""
    out = deepcopy(dict(case_dict))
    observed = extract_observed_aicore_freq_mhz(envelope)
    if observed is None:
        return out
    from mindie_sim.backends.ascend import apply_observed_aicore_clock, profiles

    known = profiles()
    for section_key in ("machine", "hardware"):
        section = out.get(section_key)
        if not isinstance(section, dict):
            continue
        device = section.setdefault("device", {})
        if not isinstance(device, dict) or "observed_aicore_freq_mhz" in device:
            return out
        sku = section.get("sku")
        if sku in known and ("compute_peaks" in device or "rates" in device):
            section["device"] = apply_observed_aicore_clock(
                {**device, "observed_aicore_freq_mhz": observed}, known[sku],
            )
        else:
            device["observed_aicore_freq_mhz"] = observed
        return out
    if isinstance(out.get("device"), dict) and "observed_aicore_freq_mhz" not in out["device"]:
        out["device"]["observed_aicore_freq_mhz"] = observed
    return out


from collections import Counter, defaultdict
from collections.abc import Mapping
import json
from statistics import median



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


NATIVE_IDENTITY_FIELDS = (
    "native_operator", "native_abi_sha256", "boundary",
)
NATIVE_CONDITION_FIELDS = (
    "hardware", "phase", "model", "quant", "deployment_image_id",
    "implementation", "execution", "attrs", "api", "version",
)
_NATIVE_STRUCTURAL_FIELDS = (
    "native_operator", "native_abi_sha256", "boundary", "scope", "expansion",
    "rank", "device_id", "stream", "clock_domain", "window_id",
    "fusion_members",
)


def _native_conditions(identity):
    """Return the full declared condition mapping, including new keys/null.

    New campaign records carry ``identity.conditions``.  The legacy top-level
    fields are accepted as a fallback and are merged only when the complete
    mapping does not already declare that key, so an explicit null survives.
    """
    raw = identity.get("conditions")
    conditions = dict(raw) if isinstance(raw, Mapping) else {}
    # Legacy callers express a condition at top level; an explicit top-level
    # value (including null) is the caller's declared fact and overrides the
    # frozen mapping rather than being silently ignored.
    for key in NATIVE_CONDITION_FIELDS:
        if key in identity:
            conditions[key] = identity[key]
    return conditions


def _native_flat_conditions(value):
    flat = {}

    def visit(item, path):
        if isinstance(item, Mapping):
            if not item:
                flat[path or "conditions"] = {}
                return
            for key, child in item.items():
                visit(child, f"{path}.{key}" if path else str(key))
            return
        flat[path or "conditions"] = item

    visit(dict(value or {}), "")
    return flat


def _native_identity(record):
    identity = record.get("identity")
    if not isinstance(identity, Mapping):
        return None, ["identity"]
    raw_boundary = identity.get("boundary")
    boundary = raw_boundary.get("kind") if isinstance(raw_boundary, Mapping) else raw_boundary
    fusion_members = identity.get("fusion_members")
    if fusion_members is None and isinstance(raw_boundary, Mapping):
        fusion = raw_boundary.get("fusion")
        fusion_members = fusion.get("members") if isinstance(fusion, Mapping) else ()
    conditions = _native_conditions(identity)
    values = {
        "native_operator": identity.get("native_operator"),
        "native_abi_sha256": identity.get("native_abi_sha256"),
        "boundary": boundary,
        "scope": identity.get("scope"),
        "expansion": identity.get("expansion"),
        "rank": identity.get("rank"),
        "device_id": identity.get("device_id"),
        "stream": identity.get("stream"),
        "clock_domain": identity.get("clock_domain"),
        "window_id": identity.get("window_id"),
        "conditions": conditions,
        "fusion_members": tuple(sorted(fusion_members or ())),
    }
    for key in NATIVE_CONDITION_FIELDS:
        values[key] = conditions.get(key)
    missing = [field for field in NATIVE_IDENTITY_FIELDS
               if not values.get(field)]
    if not conditions.get("hardware"):
        missing.append("hardware")
    if missing:
        return None, missing
    return values, []


def _native_key(identity):
    conditions = _native_conditions(identity)
    payload = {
        key: identity.get(key) for key in _NATIVE_STRUCTURAL_FIELDS
    }
    payload["conditions"] = conditions
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _native_identity_differences(left, right):
    differences = []
    for field in _NATIVE_STRUCTURAL_FIELDS:
        if left.get(field) != right.get(field):
            differences.append(field)
    left_flat = _native_flat_conditions(_native_conditions(left))
    right_flat = _native_flat_conditions(_native_conditions(right))
    for path in sorted(set(left_flat) | set(right_flat)):
        if (path not in left_flat or path not in right_flat
                or left_flat[path] != right_flat[path]):
            name = path.split(".", 1)[0]
            if name not in differences:
                differences.append(name)
    return sorted(differences)


def _native_observation_detail(group):
    durations = [sample["duration_ms"] for sample in group["samples"]]
    return {
        "identity": group["identity"],
        "request_ids": sorted(group["request_ids"]),
        "observation_ids": [sample["sample_id"] for sample in group["samples"]],
        "observed_ms": median(durations),
        "samples": len(group["samples"]),
        "observation_range_ms": [min(durations), max(durations)],
    }


def native_comparison_report(campaign):
    """Descriptive same-boundary native residuals with both unpaired sides.

    This report never assigns a tolerance verdict.  Pairing requires the exact
    content-addressed native ABI, operator, boundary, rank/stream and declared
    conditions.  A kernel or API name is not a key, and a fused observation is
    not split among its semantic members.
    """
    observations = list(campaign.get("observations", ()))
    predictions = list(campaign.get("predictions", ()))
    groups, missing_observations, seen_samples = {}, [], {}
    for sample in observations:
        identity, missing = _native_identity(sample)
        if missing:
            missing_observations.append({
                "sample_id": sample.get("sample_id"), "reason": "incomplete_identity",
                "missing_identity_fields": missing,
            })
            continue
        metadata = sample.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("unresolved_semantics"):
            missing_observations.append({
                "sample_id": sample.get("sample_id"),
                "reason": "unresolved_observation_semantics",
            })
            continue
        _duration(sample["duration_ms"], observation=True)
        key = _native_key(identity)
        sample_id = sample.get("sample_id")
        signature = (key, sample["duration_ms"], sample.get("run_id"),
                     sample.get("source"))
        if sample_id in seen_samples:
            if seen_samples[sample_id] != signature:
                missing_observations.append({
                    "sample_id": sample_id,
                    "reason": "conflicting_native_observation",
                })
            continue
        seen_samples[sample_id] = signature
        group = groups.setdefault(key, {
            "identity": identity, "samples": [],
            "request_ids": set(), "fusion_members": identity["fusion_members"],
        })
        group["samples"].append(sample)
        if sample.get("request_id"):
            group["request_ids"].add(sample["request_id"])
    by_request = {}
    for key, group in groups.items():
        for request_id in group["request_ids"]:
            by_request.setdefault(request_id, set()).add(key)
    ambiguous_requests = {
        request_id for request_id, keys in by_request.items() if len(keys) > 1
    }
    groups_by_key = {}
    for key, group in groups.items():
        groups_by_key.setdefault(key, []).append(group)
    predictions_by_key, predictions_by_request, unpaired_predictions = {}, {}, []
    for prediction in predictions:
        identity, missing = _native_identity(prediction)
        request_id = prediction.get("request_id")
        if missing:
            unpaired_predictions.append({
                "prediction_id": prediction.get("prediction_id"),
                "reason": "incomplete_identity",
                "missing_identity_fields": missing,
            })
            continue
        status = str(prediction.get("status") or "modeled")
        if status not in {"modeled", "predicted", "ok"}:
            reason = prediction.get("reason") or prediction.get("unmodeled_reason")
            unpaired_predictions.append({
                "prediction_id": prediction.get("prediction_id"),
                "reason": f"prediction_{status}" + (f":{reason}" if reason else ""),
            })
            continue
        try:
            duration = _duration(prediction["duration_ms"])
        except (KeyError, TypeError, ValueError):
            unpaired_predictions.append({
                "prediction_id": prediction.get("prediction_id"),
                "reason": "invalid_prediction_duration",
            })
            continue
        row = {
            "prediction_id": prediction.get("prediction_id"),
            "request_id": request_id, "identity": identity,
            "duration_ms": duration,
            "evidence": list(prediction.get("evidence") or ()),
        }
        if request_id:
            predictions_by_request.setdefault(request_id, []).append(row)
        else:
            predictions_by_key.setdefault(_native_key(identity), []).append(row)
    points, not_comparable, used = [], [], set()
    paired_keys, not_comparable_keys = set(), set()
    for key, group in groups.items():
        identity = group["identity"]
        if group["request_ids"] & ambiguous_requests:
            not_comparable.append({
                "reason": "ambiguous_request_identity",
                **_native_observation_detail(group),
            })
            not_comparable_keys.add(key)
            continue
        by_id = [
            row for request_id in group["request_ids"]
            for row in predictions_by_request.get(request_id, ())
        ]
        if not by_id and len(groups_by_key.get(key, [])) > 1:
            not_comparable.append({
                "reason": "ambiguous_observation_candidates",
                **_native_observation_detail(group),
                "candidate_request_ids": [
                    candidate["request_ids"] for candidate in groups_by_key[key]
                ],
            })
            not_comparable_keys.add(key)
            continue
        candidates = by_id or predictions_by_key.get(key, [])
        if not candidates:
            continue
        exact = [row for row in candidates
                 if _native_key(row["identity"]) == key]
        if not exact:
            differences = {
                row["prediction_id"]: _native_identity_differences(
                    identity, row["identity"])
                for row in candidates
            }
            fields = sorted({
                field for values in differences.values() for field in values})
            reason = ("fusion_boundary_mismatch" if fields == ["fusion_members"]
                      else "request_identity_conflict:" + ",".join(fields))
            not_comparable.append({
                "reason": reason,
                **_native_observation_detail(group),
                "prediction_ids": [row["prediction_id"] for row in candidates],
                "identity_differences": differences,
            })
            not_comparable_keys.add(key)
            continue
        candidates = exact
        if len(candidates) > 1:
            not_comparable.append({
                "reason": "ambiguous_predictions",
                **_native_observation_detail(group),
                "prediction_ids": [row["prediction_id"] for row in candidates],
            })
            not_comparable_keys.add(key)
            used.update(row["prediction_id"] for row in candidates)
            continue
        prediction = candidates[0]
        used.add(prediction["prediction_id"])
        observed = median(sample["duration_ms"] for sample in group["samples"])
        residual = prediction["duration_ms"] - observed
        sample_differences = []
        for sample in group["samples"]:
            sample_residual = prediction["duration_ms"] - sample["duration_ms"]
            sample_differences.append({
                "sample_id": sample["sample_id"],
                "observed_ms": sample["duration_ms"],
                "signed_residual_ms": sample_residual,
                "absolute_residual_ms": abs(sample_residual),
                "relative_error": sample_residual / sample["duration_ms"],
                "denominator": "observed_ms",
            })
        points.append({
            "point_id": "native-point:" + _native_key(identity)[:32],
            "request_id": prediction.get("request_id"),
            "identity": identity,
            "prediction_id": prediction["prediction_id"],
            "observed_ms": observed,
            "predicted_ms": prediction["duration_ms"],
            "statistic": "median_of_observations",
            "samples": len(group["samples"]),
            "runs": len({sample.get("run_id") for sample in group["samples"]}),
            "observation_range_ms": [
                min(sample["duration_ms"] for sample in group["samples"]),
                max(sample["duration_ms"] for sample in group["samples"]),
            ],
            "comparison": {
                "signed_residual_ms": residual,
                "absolute_residual_ms": abs(residual),
                "relative_error": residual / observed,
                "denominator": "observed_ms",
                "sample_differences": sample_differences,
                "worst_absolute_relative_error": max(
                    abs(row["relative_error"]) for row in sample_differences),
            },
            "prediction_evidence": prediction["evidence"],
            "interpretation": "descriptive_same_boundary_difference_not_acceptance",
        })
        paired_keys.add(key)
    unpaired_observations = []
    for key, group in groups.items():
        if key in paired_keys or key in not_comparable_keys:
            continue
        unpaired_observations.append({
            "request_ids": sorted(group["request_ids"]),
            "identity": group["identity"],
            "observation_ids": [sample["sample_id"] for sample in group["samples"]],
            "observed_ms": median(sample["duration_ms"] for sample in group["samples"]),
            "samples": len(group["samples"]),
            "reason": "no_exact_prediction_identity",
        })
    for rows in (*predictions_by_key.values(), *predictions_by_request.values()):
        for row in rows:
            if row["prediction_id"] not in used:
                unpaired_predictions.append({
                    "prediction_id": row["prediction_id"],
                    "request_id": row.get("request_id"),
                    "identity": row["identity"],
                    "predicted_ms": row["duration_ms"],
                    "reason": "no_exact_observation_identity",
                })
    status = ("completed" if points and not not_comparable and not unpaired_observations
              and not unpaired_predictions and not missing_observations
              else "partial")
    return {
        "schema": "mindie-sim.native-comparison/1",
        "status": status,
        "cohort_role": campaign.get("cohort_role", "unspecified"),
        "fixed_boundary": {
            "join": "native_operator+native_abi_sha256+boundary+rank/stream+conditions",
            "denominator": "observed_ms",
            "name_only_allowed": False,
            "family_average_allowed": False,
            "fusion_member_allocation_allowed": False,
        },
        "coverage": {
            "observations": len(observations),
            "predictions": len(predictions),
            "paired": len(points),
            "not_comparable": len(not_comparable),
            "not_comparable_observation_samples": sum(
                int(entry.get("samples", 0)) for entry in not_comparable),
            "unpaired_observations": len(unpaired_observations),
            "unpaired_predictions": len(unpaired_predictions),
            "excluded_observations": len(missing_observations),
        },
        "points": points,
        "not_comparable": not_comparable,
        "unpaired_observations": unpaired_observations,
        "unpaired_predictions": unpaired_predictions,
        "excluded_observations": missing_observations,
        "non_claims": [
            "descriptive_residual_not_acceptance",
            "no_name_only_pairing",
            "no_family_average_pass",
            "no_fused_member_allocation",
        ],
        "interpretation": [
            "Residuals are descriptive same-boundary differences, not acceptance.",
            "A name-only or family-level match is never emitted.",
            "A fused observation retains its semantic members and is not allocated.",
            "Unpaired and unresolved sides remain visible.",
        ],
    }
