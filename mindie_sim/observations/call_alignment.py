"""Conditional comparisons of preserved full callbacks and analytical graphs.

[Role] Join reviewed callback structure with independent call predictions.
[Invariants] Event identities, canaries and sample roles are immutable. Different
  timing boundaries produce descriptive differences, never accuracy acceptance.
  Durations do not enter builders, prediction keys, resources or submission.
[Timeouts & Locks] Stream one condition; bounded condition/prediction ledgers.
[Side-Effect Policy] Reads audited captures; no calibration or device execution.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from statistics import median

from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from .captures import iter_conditions
from .historical import call_observations


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def _statistics(samples):
    if not samples:
        return None
    values = [sample["duration_ms"] for sample in samples]
    trials = defaultdict(list)
    for sample in samples:
        trial = sample.get("trial", sample.get("metadata", {}).get("trial", 0))
        trials[sample.get("run_id"), trial].append(sample["duration_ms"])
    return {
        "sample_count": len(values), "trial_count": len(trials),
        "sample_median_ms": median(values),
        "median_of_trial_medians_ms": median(median(v) for v in trials.values()),
        "minimum_ms": min(values), "maximum_ms": max(values),
    }


def _capture_record(condition):
    samples = condition["samples"]
    return {
        "condition_id": condition["condition_id"], "source": condition["source"],
        "identity": condition["identity"],
        "boundary": condition["identity"].get("boundary"),
        "capture_status": condition["status"], "metadata": condition["metadata"],
        "family": condition["metadata"].get("variant", "grouped_gemm"),
        "canary": condition["metadata"].get("canary"),
        "sample_roles": dict(Counter(s["split"] for s in samples)),
        "run_ids": sorted({s["run_id"] for s in samples}),
        "statistics": _statistics(samples),
    }


def _gdn_record(observation):
    return {
        "condition_id": observation["observation_id"], "source": observation["source"],
        "identity": observation["identity"], "boundary": observation["boundary"],
        "capture_status": observation["status"],
        "family": observation["identity"]["conditions"]["role"],
        "metadata": {key: observation[key] for key in
                     ("producer", "capture_environment", "model_geometry_label",
                      "reported_statistics", "failure") if key in observation},
        "canary": observation["canary"],
        "sample_roles": {observation["split"]: len(observation["samples"])},
        "run_ids": [observation["source_sha256"]],
        "statistics": observation["statistics"],
    }


def _activation(condition):
    from .call_bindings_activation import bind_activation_condition
    binding = bind_activation_condition(condition)
    return (binding.call, binding.ledger,
            {"invocations": binding.invocations, "mode": binding.mode})


def _gdn(observation):
    from .gdn_binding import bind_observation
    binding = bind_observation(observation)
    if binding["status"] != "bound":
        raise ModelingError("; ".join(binding["mismatches"]))
    call = binding["call"]
    return call, {**call.ledger, "unresolved": list(call.limitations)}, {
        "invocations": 1, "mode": "eager"}


def _grouped(condition):
    from .grouped_binding import bind_grouped_condition
    binding = bind_grouped_condition(condition)
    return binding.call, binding.ledger, {
        "invocations": binding.invocations, "mode": binding.mode}


# Adapters preserve original schemas; builders remain independently callable.
ADAPTERS = {
    "activation_calls": (iter_conditions, _capture_record, _activation),
    "grouped_calls": (iter_conditions, _capture_record, _grouped),
    "historical-call-protocol/1": (call_observations, _gdn_record, _gdn),
}


def _diagnostics(rows):
    groups = defaultdict(list)
    for row in rows:
        for comparison in row["comparisons"]:
            if "relative_difference" in comparison and comparison["relative_difference"] is not None:
                groups[row["family"], row["boundary"], comparison["scenario"]].append(comparison)
    return [{
        "family": family, "observation_boundary": boundary, "scenario": scenario,
        "conditions": len(values),
        "median_relative_difference": median(v["relative_difference"] for v in values),
        "minimum_relative_difference": min(v["relative_difference"] for v in values),
        "maximum_relative_difference": max(v["relative_difference"] for v in values),
        "interpretation": "descriptive_condition_distribution_not_workload_accuracy",
    } for (family, boundary, scenario), values in sorted(groups.items())]


def compare_calls(case, protocol, *, root=".", include_submission=False,
                  submission=None, max_predictions=512, max_conditions=4096):
    """Report conditional event/graph differences without fitting either side.

    The optional submitted scenario uses declared host queue parameters. It is
    an alternative scenario, not a confidence bound on the original event.
    Unsupported callback semantics remain rows with their original observations.
    """
    case = parse_case(case)
    case_sha256 = _hash(case.to_dict())
    adapter = protocol.get("adapter", protocol.get("schema"))
    if adapter not in ADAPTERS:
        raise ModelingError(f"call_alignment_protocol_unsupported:{adapter}")
    if max_predictions < 1 or max_conditions < 1:
        raise ModelingError("call_alignment_requires_positive_budgets")
    reader, record, bind = ADAPTERS[adapter]
    rows, predictions, cache = [], [], {}
    scenarios = [("device", False)] + ([("submitted", True)] if include_submission else [])
    failures, evaluations = 0, 0
    for observation in reader(protocol, root=root):
        if len(rows) >= max_conditions:
            raise ModelingError("call_alignment_condition_budget_exceeded")
        row = record(observation)
        row.update(status="unsupported", comparisons=[])
        rows.append(row)
        # A failed capture does not prevent a declaration from being simulated.
        # Only missing/unimplemented arguments prevent this adapter binding.
        try:
            call, ledger, schedule = bind(observation)
        except ValueError as exc:
            row["reason"] = str(exc)
            continue
        row["alignment"] = ledger
        row["status"] = "predicted"
        for scenario, with_submission in scenarios:
            key = _hash({"case_sha256": case_sha256, "identity": row["identity"],
                         "ledger": ledger, "schedule": schedule, "scenario": scenario,
                         "submission": dict(submission or {})})
            if key not in cache:
                if evaluations >= max_predictions:
                    cache[key] = {"status": "failed", "reason": "prediction_budget_exceeded"}
                else:
                    evaluations += 1
                    try:
                        result = call_costs(
                            case, call.ops, recipes=call.recipes, **schedule,
                            include_submission=with_submission, submission=submission,
                        )
                        prediction = {
                            "prediction_id": key, "scenario": scenario,
                            "status": "modeled", **result,
                        }
                        predictions.append(prediction)
                        cache[key] = prediction
                    except ValueError as exc:
                        cache[key] = {"status": "failed", "reason": str(exc)}
            prediction = cache[key]
            comparison = {
                "scenario": scenario, "status": prediction["status"],
                "strict_identity_equivalence": False,
                "observation_boundary": row["boundary"],
            }
            if prediction["status"] == "failed":
                comparison["reason"] = prediction["reason"]
                row["status"] = "prediction_failed"
                failures += 1
            else:
                comparison.update(
                    prediction_id=key, prediction_boundary=prediction["boundary"],
                    predicted_duration_ms=prediction["duration_ms"])
                statistics = row["statistics"]
                if statistics:
                    observed = statistics["sample_median_ms"]
                    delta = prediction["duration_ms"] - observed
                    comparison.update(
                        reference_statistic="sample_median_ms",
                        observed_duration_ms=observed, signed_difference_ms=delta,
                        absolute_difference_ms=abs(delta),
                        relative_difference=delta / observed if observed > 0 else None,
                        interpretation="conditional_boundary_difference",
                    )
                else:
                    comparison["interpretation"] = "prediction_without_timing_samples"
            row["comparisons"].append(comparison)
    statuses = Counter(row["status"] for row in rows)
    roles = Counter()
    for row in rows:
        roles.update(row["sample_roles"])
    return {
        "schema": "mindie-sim.call-alignment/1",
        "status": "completed" if statuses["predicted"] == len(rows) else "partial",
        "adapter": adapter, "case_sha256": case_sha256,
        "machine": case.machine.to_dict(),
        "protocol": protocol,
        "scenario_declarations": {
            "include_submission": include_submission, "submission": dict(submission or {}),
            "note": "Hardware launch service remains from the supplied case; no residual fitting.",
        },
        "summary": {
            "conditions": len(rows), "condition_statuses": dict(statuses),
            "samples": sum(roles.values()), "sample_roles": dict(roles),
            "unique_predictions": len(predictions), "prediction_evaluations": evaluations,
            "failed_condition_scenarios": failures,
        },
        "validation": {
            "status": "unassessed", "strict_pairs": 0,
            "reason": "Declared device/submitted graph spans retain a different boundary from raw events.",
        },
        "interpretation": [
            "Complete callback dataflow is aligned only within each binder's declared domain.",
            "Source references do not identify the historical loaded native binary.",
            "Host allocator, internal dispatch and event-enqueue gaps are not inferred from residuals.",
            "Device and submitted scenarios are declarations, not statistical lower/upper bounds.",
            "Original canaries, sample roles and event spans remain unchanged.",
            "Conditional differences diagnose mechanisms; no tolerance or accuracy pass is assigned.",
        ],
        "diagnostics": _diagnostics(rows),
        "predictions": predictions, "conditions": rows,
    }
