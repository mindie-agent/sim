"""Strict execution-plan identity and report-only whole-kernel replay.

Reuse the operator-plan execution and legacy resource engine. Observations never
replace costs or fit stage parameters. Only a declared measured, non-synthetic,
heldout observation with the complete current identity contributes to accuracy.
No device, network, database or filesystem operations are performed here.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from .a3_cost import PredictiveCostModel
from .a3_ops import OpSpec
from .a3_resources import ResourceEngine, add_costed_ops, build_resource_specs
from .a3_types import A3InputError, MachineSpec
from .operator_estimation import _operator, create_operator_estimator, operator_plan_report
from .operator_plan import EstimateContext, KernelPlan

_PROTOCOL_FIELDS = {
    "timing_scope", "timer", "execution_mode", "calls_per_sample", "warmup",
    "completion", "includes_dispatch", "cache_state",
}
_EVIDENCE_FIELDS = {
    "sample_id", "identity", "identity_sha256", "split", "source_class", "synthetic", "duration_ms", "source",
}


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise A3InputError("execution_replay_non_json_or_nonfinite_input") from exc


def identity_sha256(identity: Mapping[str, Any]) -> str:
    """Stable hash of a complete identity, not an op ID or a shape lookup key."""
    return "sha256:" + hashlib.sha256(_canonical(identity).encode()).hexdigest()


def _text(value: Any, reason: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise A3InputError(reason)
    return value


def _mapping(raw: Any, allowed: set[str], reason: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) - allowed:
        raise A3InputError(reason)
    return raw


def _sequence(raw: Any, reason: str) -> Sequence[Any]:
    if not isinstance(raw, (tuple, list)):
        raise A3InputError(reason)
    return raw


def _protocol(raw: Mapping[str, Any], context: EstimateContext) -> dict[str, Any]:
    value = _mapping(raw, _PROTOCOL_FIELDS, "execution_replay_protocol_fields_invalid")
    if set(value) != _PROTOCOL_FIELDS:
        raise A3InputError("execution_replay_protocol_incomplete")
    for key in ("timing_scope", "timer", "execution_mode", "warmup", "completion", "cache_state"):
        _text(value[key], "execution_replay_protocol_field_invalid:" + key)
    if value["timing_scope"] != context.timing_scope:
        raise A3InputError("execution_replay_protocol_context_scope_mismatch")
    # The current operator-plan includes one device dispatch and one isolated
    # call. Do not reinterpret a batch, host span, or service-only observation.
    if (context.timing_scope != "device_event_warm" or value["timer"] != "device_event"
            or value["execution_mode"] != "eager" or value["warmup"] != "warm"
            or type(value["calls_per_sample"]) is not int or value["calls_per_sample"] != 1
            or value["completion"] != "isolated_whole_kernel" or value["includes_dispatch"] is not True):
        raise A3InputError("execution_replay_protocol_unsupported")
    return dict(value)


def execution_plan_identity(
    plan: KernelPlan, op: OpSpec, machine: MachineSpec, context: EstimateContext,
    *, estimator_config: Mapping[str, Any], protocol: Mapping[str, Any], launch_ms: float,
) -> dict[str, Any]:
    """Conservatively lock the full plan, effective hardware and declared config.

    Source descriptions and local provider paths are deliberately retained. A
    provenance-only/configuration change may refuse an otherwise similar plan;
    this identity is not a promise of semantic equivalence across environments.
    Sample ID, evidence source/outcome and external op dependencies are excluded.
    """
    if not isinstance(plan, KernelPlan):
        raise A3InputError("execution_replay_plan_required")
    declared_protocol = _protocol(protocol, context)
    if (isinstance(launch_ms, bool) or not isinstance(launch_ms, (int, float))
            or not math.isfinite(launch_ms) or launch_ms < 0):
        raise A3InputError("execution_replay_launch_invalid")
    operator = op.to_dict()
    for key in ("op_id", "deps", "requests", "notes"):
        operator.pop(key)
    operator["declared_dtype"] = operator["dtype"]
    operator["dtype"] = _text(
        op.dtype or context.activation_dtype or context.compute_dtype, "execution_replay_dtype_missing",
    )
    _text(op.layout, "execution_replay_layout_missing")
    hardware = machine.to_dict()
    result = {
        "schema_version": 1, "plan": plan.to_dict(), "operator": operator,
        "context": asdict(context), "hardware": hardware,
        "hardware_sha256": identity_sha256(hardware),
        "estimator_config": dict(estimator_config),
        "timing_scope": context.timing_scope, "protocol": declared_protocol,
        "launch_ms": float(launch_ms), "duration_unit": "ms",
    }
    # Return a detached JSON object, so caller-owned mutable provenance cannot
    # later change the identity already used by a report or observation.
    return json.loads(_canonical(result))


def _legacy_isolated(cost: Any, machine: MachineSpec, context: EstimateContext) -> float:
    resources, stages = build_resource_specs(
        machine, stages=cost.op.stage + 1, compute_dtype=context.compute_dtype,
    )
    engine = ResourceEngine(resources)
    isolated = replace(cost, op=replace(cost.op, deps=()))
    ids = add_costed_ops(engine, (isolated,), stages, prefix="isolated", start_ms=0.0)
    result = engine.run(until=tuple(ids.values()))
    return float(result["completed_at_ms"][ids[cost.op.op_id]])


def _shared_span(op: OpSpec, cost: Any, report: Mapping[str, Any], completion_ids: Mapping[str, str]) -> dict[str, Any]:
    execution = report["execution"]
    task = "operators:" + op.op_id
    completed = execution["completed_at_ms"]
    ready = max((float(completed[completion_ids[dep]]) for dep in op.deps), default=0.0)
    end = float(completed[completion_ids[op.op_id]])
    # Use exact IDs instead of startswith(op_id), which conflates user IDs
    # such as "x" and "x:neighbor". The parent gate is zero work for fine plans.
    ids = {task, completion_ids[op.op_id]}
    if cost.execution_plan is not None:
        ids.update(f"{task}:plan:{stage.name}" for stage in cost.execution_plan.stages)
        ids.add(task + ":dispatch")
    else:
        metadata = dict(cost.op.metadata)
        ids.update(f"{task}:comm{i}" for i, _ in enumerate(metadata.get("comm_stages", ())))
        ids.update(f"{task}:stage{i}:{stage.get('name', i)}"
                   for i, stage in enumerate(metadata.get("fusion_stages", ())))
    events = [event for event in execution["trace"] if event["task_id"] in ids]
    first = min(float(event["start_ms"]) for event in events)
    return {
        "dependency_ready_ms": ready, "first_admission_ms": first, "complete_ms": end,
        "span_from_ready_ms": end - ready,
        "ready_to_first_admission_ms": first - ready,
        "span_from_first_admission_ms": end - first,
        "event_task_ids": [event["task_id"] for event in events],
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _errors(rows: Sequence[Mapping[str, Any]], prediction: str) -> dict[str, Any]:
    durations = [row["duration_ms"] for row in rows]
    errors = [abs(row[prediction] - row["duration_ms"]) for row in rows]
    percents = [error / duration * 100 for error, duration in zip(errors, durations)]
    # Positive finite inputs can still overflow naive sums. Normalize before
    # taking incremental means; averaging huge relative errors by duration can
    # otherwise lose a tiny-duration contribution through cancellation.
    def mean(values: Sequence[float]) -> float:
        result = 0.0
        for index, value in enumerate(values):
            result += (value - result) / (index + 1)
        return result

    scale = max(durations, default=1.0)
    wape = (mean([error / scale for error in errors])
            / mean([duration / scale for duration in durations]) * 100) if rows else None
    if wape is not None and not math.isfinite(wape):
        raise A3InputError("execution_replay_accuracy_numeric_range_exceeded")
    return {
        "count": len(rows), "wape_pct": wape,
        "mean_absolute_error_ms": mean(errors) if rows else None,
        "p50_absolute_error_pct": _percentile(percents, .5),
        "p95_absolute_error_pct": _percentile(percents, .95),
    }


def _review_evidence(
    raw: Sequence[Any], operators: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_identity: dict[str, list[Mapping[str, Any]]] = {}
    for row in operators:
        if row["identity_sha256"] is not None:
            by_identity.setdefault(row["identity_sha256"], []).append(row)
    sample_counts = Counter(row.get("sample_id") for row in raw
                            if isinstance(row, Mapping) and isinstance(row.get("sample_id"), str))
    reviewed: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        result: dict[str, Any] = {"index": index, "status": "rejected"}
        try:
            record = _mapping(item, _EVIDENCE_FIELDS, "execution_replay_evidence_fields_invalid")
            if set(record) != _EVIDENCE_FIELDS:
                raise A3InputError("execution_replay_evidence_incomplete")
            sample_id = _text(record["sample_id"], "execution_replay_evidence_sample_id_invalid")
            result["sample_id"] = sample_id
            if sample_counts[sample_id] != 1:
                raise A3InputError("execution_replay_evidence_duplicate_sample_id")
            _text(record["source"], "execution_replay_evidence_source_missing")
            if record["split"] not in {"train", "heldout"}:
                raise A3InputError("execution_replay_evidence_split_invalid")
            if (record["source_class"] not in {"measured", "synthetic"}
                    or not isinstance(record["synthetic"], bool)
                    or record["synthetic"] != (record["source_class"] == "synthetic")):
                raise A3InputError("execution_replay_evidence_source_class_synthetic_mismatch")
            duration = record["duration_ms"]
            if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or not math.isfinite(duration) or duration <= 0):
                raise A3InputError("execution_replay_evidence_duration_invalid")
            identity = record["identity"]
            if not isinstance(identity, Mapping):
                raise A3InputError("execution_replay_evidence_identity_missing")
            fingerprint = identity_sha256(identity)
            if fingerprint != record["identity_sha256"]:
                raise A3InputError("execution_replay_evidence_identity_hash_mismatch")
            candidates = by_identity.get(fingerprint)
            if not candidates or any(identity != row["identity"] for row in candidates):
                raise A3InputError("execution_replay_evidence_identity_not_in_current_plans")
            if any(not math.isfinite(abs(candidates[0][key] - duration) / duration * 100)
                   for key in ("legacy_isolated_ms", "fine_isolated_ms")):
                raise A3InputError("execution_replay_evidence_error_percentage_unrepresentable")
            result.update({
                "status": "matched", "sample_id": sample_id, "identity_sha256": fingerprint,
                "source": record["source"], "source_class": record["source_class"],
                "synthetic": record["synthetic"], "split": record["split"],
                "duration_ms": float(duration), "matched_op_ids": [row["op_id"] for row in candidates],
                "legacy_isolated_ms": candidates[0]["legacy_isolated_ms"],
                "fine_isolated_ms": candidates[0]["fine_isolated_ms"],
                "accuracy_eligible": not record["synthetic"] and record["split"] == "heldout",
                "comparison_scope": "isolated_whole_kernel; report only, never stage fitting",
            })
        except (A3InputError, TypeError, OverflowError) as exc:
            result["reason"] = getattr(exc, "reason", None) or str(exc)
        reviewed.append(result)
    matched = [row for row in reviewed if row["status"] == "matched"]
    heldout = [row for row in matched if row["accuracy_eligible"]]
    train = [row for row in matched if row["source_class"] == "measured" and row["split"] == "train"]
    return reviewed, {
        "status": "declared_measured_heldout_comparison" if heldout else "no_matched_measured_heldout",
        "scope": "declared split and provenance; collection authenticity and training independence are not certified",
        "heldout_sample_ids": [row["sample_id"] for row in heldout],
        "train_sample_ids": [row["sample_id"] for row in train],
        "synthetic_matched_count": sum(row["synthetic"] for row in matched),
        "heldout": {"legacy_isolated": _errors(heldout, "legacy_isolated_ms"),
                    "fine_isolated": _errors(heldout, "fine_isolated_ms")},
        "train_diagnostics": {"legacy_isolated": _errors(train, "legacy_isolated_ms"),
                              "fine_isolated": _errors(train, "fine_isolated_ms")},
        "rejected_count": len(reviewed) - len(matched),
    }


def execution_replay_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Replay an operator-plan payload without modifying its execution costs."""
    wrapper = _mapping(payload, {"schema_version", "operator_plan", "protocol", "evidence"},
                       "execution_replay_input_fields_invalid")
    if type(wrapper.get("schema_version")) is not int or wrapper["schema_version"] != 1:
        raise A3InputError("execution_replay_schema_version_unsupported")
    raw = wrapper.get("operator_plan")
    if not isinstance(raw, Mapping):
        raise A3InputError("execution_replay_operator_plan_missing")
    context = EstimateContext(**dict(raw.get("context", {})))
    protocol = _protocol(wrapper.get("protocol"), context)
    observations = _sequence(wrapper.get("evidence", ()), "execution_replay_evidence_list_invalid")
    # Existing parsing, providers, costs and graph execution remain authoritative.
    shared = operator_plan_report(raw, include_task_ids=True)
    completion_ids = shared.pop("task_ids")
    machine = MachineSpec.from_mapping(raw.get("machine"))
    ops = tuple(_operator(item) for item in raw["operators"])
    kwargs = {
        "compute_dtype": context.compute_dtype, "weight_dtype": context.weight_dtype,
        "activation_dtype": context.activation_dtype, "timing_scope": context.timing_scope,
        "quantization": context.quantization,
    }
    legacy = PredictiveCostModel(machine, **kwargs)
    fine = PredictiveCostModel(machine, op_estimator=create_operator_estimator(raw["estimators"]), **kwargs)
    baselines = [legacy.cost(op) for op in ops]
    selected_costs = [fine.cost(op) for op in ops]
    rows = []
    for op, baseline, selected in zip(ops, baselines, selected_costs):
        plan = selected.execution_plan
        identity = execution_plan_identity(
            plan, op, machine, context, estimator_config=raw["estimators"],
            protocol=protocol, launch_ms=selected.launch_ms,
        ) if plan is not None else None
        span = _shared_span(op, selected, shared, completion_ids)
        isolated = float(selected.estimated_ms) if plan is not None else None
        extra = span["span_from_ready_ms"] - isolated if isolated is not None else None
        if extra is not None and abs(extra) <= 4 * (
            math.ulp(span["complete_ms"]) + math.ulp(span["dependency_ready_ms"]) + math.ulp(isolated)
        ):
            extra = 0.0
        rows.append({
            "op_id": op.op_id, "family": op.family,
            "operator_estimation": dict(selected.operator_estimation or {}),
            "identity": identity, "identity_sha256": identity_sha256(identity) if identity else None,
            "legacy_estimate_ms": baseline.estimated_ms,
            "legacy_isolated_ms": _legacy_isolated(baseline, machine, context),
            "fine_isolated_ms": isolated, "shared_execution": span,
            "extra_execution_span_ms": extra,
        })
    evidence, accuracy = _review_evidence(observations, rows)
    selected_count = sum(row["identity"] is not None for row in rows)
    fallback = [row for row in rows if row["identity"] is None]
    return {
        "status": "modeled", "command": "execution-replay", "schema_version": 1,
        "input_sha256": hashlib.sha256(_canonical(payload).encode()).hexdigest(),
        "effective_input": json.loads(_canonical(payload)), "protocol": protocol,
        "scope": "report-only, single eager warm device-event whole-kernel call including device dispatch",
        "not_npu_measured_by_sim": True, "costs_modified_by_evidence": False,
        "units": {"durations": "ms", "error_percentages": "percent"},
        "extra_execution_span_definition": "shared completion minus external dependency-ready time minus fine isolated time; signed difference, not causal contention attribution",
        "coverage": {"operator_records": len(rows), "fine_plan_records": selected_count,
                     "fine_record_fraction": selected_count / len(rows),
                     "fallback_or_preserved_records": len(fallback),
                     "reasons": dict(Counter(row["operator_estimation"].get("reason", "unspecified") for row in fallback)),
                     "scope": "operator records, not time coverage"},
        "operators": rows, "evidence": evidence, "accuracy": accuracy,
        "operator_plan": shared,
        "limitations": ["strict identity locks source text and local profile/provider paths",
                        "no database import, cost replacement, parameter fit or stage-duration attribution",
                        "batched, graph, host-wall and dispatch-excluded protocols are unsupported"],
    }


__all__ = ["identity_sha256", "execution_plan_identity", "execution_replay_report"]
