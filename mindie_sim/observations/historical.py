"""Reuse historical operator calls without importing their fitted predictors.

[Role] Read retained event samples as independent comparison observations.
[Invariants] Source hashes and call boundaries are explicit. A wrapper event is
  not a device-kernel duration. Canary outcomes and failed rows remain visible;
  neither controls admission to simulation. No historical fit enters prediction.
[Timeouts & Locks] Streaming JSON with one active condition; no device or locks.
[Side-Effect Policy] Reads the caller's capture and producer only.
"""
from collections import Counter
import math
from pathlib import Path
from statistics import median

import ijson
from ijson.common import ObjectBuilder

from .store import digest_file, json_items


def _header(path):
    wanted = {"schema", "env", "geometry", "trials", "samples_per_condition", "warmup"}
    result, builder, root, depth = {}, None, None, 0
    with Path(path).open("rb") as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if builder is None:
                if prefix not in wanted:
                    continue
                if event not in {"start_map", "start_array"}:
                    result[prefix] = value
                    continue
                root, builder, depth = prefix, ObjectBuilder(), 0
            builder.event(event, value)
            if event in {"start_map", "start_array"}:
                depth += 1
            elif event in {"end_map", "end_array"}:
                depth -= 1
                if depth == 0:
                    result[root], builder = builder.value, None
    return result


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
