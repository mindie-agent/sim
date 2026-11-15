"""Typed immutable launch/SOL/feature reports and their semantic checks.

[Role] Own the report records returned by `launch_*`, `prediction.sol` and
  `feature_analysis`, their canonical JSON/Markdown projections and the named
  contract checks L01-L04/S01-S05.  They project existing Case/Engine data.
[Invariants] Frozen tuples only; finite JSON doubles.  Launch bucket IDs
  partition `setting_ids`; parents/shadow IDs/leaves resolve; secrets are
  `[REDACTED]`.  Service is work/peak; load sums terms; SOL is
  max(critical path, resource load) and never exceeds the same-plan Engine
  prediction beyond 1e-12 ms.  Available request latencies are positive.
  Markdown never recomputes physics.
[Timeouts & Locks] Bounded in-memory serialization/validation; no locks.
[Side-Effect Policy] No I/O, environment reads or device contexts.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import json
import math
import re
from typing import Any

SOL_TOLERANCE_MS = 1e-12
ARITHMETIC_TOLERANCE = 1e-9
_SECRET_MARKERS = ("api_key", "apikey", "secret", "password", "credential",
                    "authorization", "bearer")
_CREDENTIAL_RE = re.compile(r"bearer |sk-|password=|api_key=|apikey=", re.IGNORECASE)
_R = dataclass(frozen=True, slots=True)


class ReportValidationError(ValueError):
    """A report failed a named semantic contract check."""

    def __init__(self, errors):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors) or "report_invalid")


_FIELD_NAMES: dict[type, tuple[str, ...]] = {}
_SCALAR_TYPES = frozenset({str, int, bool})
_ZERO_INT = 0; _ZERO_FLOAT = 0.0


def _plain(value):
    """JSON-safe projection: tuples->lists, mappings->dicts, finite doubles."""
    cls = type(value)
    if value is None or cls in _SCALAR_TYPES:
        return value
    if cls is float:
        if not math.isfinite(value):
            raise ValueError("report_number_must_be_finite")
        return 0.0 if value == 0 else value
    if cls is dict:
        return {str(key): _plain(item) for key, item in value.items()}
    if cls is list or cls is tuple:
        return [_plain(item) for item in value]
    if not isinstance(value, type) and is_dataclass(value):
        names = _FIELD_NAMES.get(cls)
        if names is None:
            names = tuple(field.name for field in fields(cls))
            _FIELD_NAMES[cls] = names
        return {name: _plain(getattr(value, name)) for name in names}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("report_number_must_be_finite")
        return _ZERO_FLOAT if value == _ZERO_INT else value
    if isinstance(value, (str, int, bool)):
        return value
    raise ValueError(f"report_value_not_json:{cls.__name__}")


to_dict = _plain


def canonical_json(report) -> str:
    """One canonical JSON object (sorted keys, compact separators, LF end)."""
    return json.dumps(_plain(report), sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False) + "\n"


def _close(left, right, tolerance=ARITHMETIC_TOLERANCE):
    return abs(float(left) - float(right)) <= tolerance * max(
        1.0, abs(float(left)), abs(float(right)))


def _as_dict(value):
    return value if isinstance(value, dict) else _plain(value)

@_R
class Span:
    source: str; line_start: int; column_start: int; line_end: int; column_end: int
@_R
class Diagnostic:
    code: str; severity: str; message: str; setting_id: str | None = None; op_ids: tuple[str, ...] = ()
@_R
class Setting:
    id: str; parent_id: str | None; kind: str; name: str; pointer: str | None; value: Any
    redacted: bool; span: Span; effective: bool; shadowed_by: str | None; effect: str
    targets: tuple[str, ...]; rule_id: str | None; reason: str
@_R
class RequiredInput:
    name: str; reason: str; affected_metrics: tuple[str, ...] = ()
@_R
class SourceIdentity:
    name: str; revision: str | None = None; sha256: str | None = None; source_kind: str = "user"
@_R
class Patch:
    path: str; before: Any; after: Any; reason: str
@_R
class Context:
    case_id: str; plan_id: str; scope: str; phase: str; tokens: int; sequences: int
    context_tokens: int; boundary: str; coverage: str; sources: tuple[SourceIdentity, ...]
@_R
class Demand:
    physical_resource_id: str; owner_units: tuple[int, ...]; mode: str; work: float
    work_unit: str; peak_per_ms: float; peak_source: str; service_ms: float
@_R
class ResourceLoad:
    physical_resource_id: str; terms: tuple[Demand, ...]; load_bound_ms: float
@_R
class LocalBound:
    id: str; member_ops: tuple[str, ...]; demands: tuple[Demand, ...]
    resource_bound_ms: float; phase_precedence_bound_ms: float; lower_bound_ms: float
@_R
class Bound:
    status: str; dispatch_policy: str; critical_path_ms: float | None
    critical_path_ops: tuple[str, ...]; resource_load_ms: float | None; sol_ms: float | None
    resources: tuple[ResourceLoad, ...]; local_bounds: tuple[LocalBound, ...]
    unavailable_reasons: tuple[str, ...] = (); kind: str = "necessary_lower_bound"
    achievability_claimed: bool = False; uses_empirical_derating: bool = False
    work_basis: str = "selected_issued_work"; floor_bases: tuple[str, ...] = ()
    empirical_floor_ms: float | None = None
@_R
class Prediction:
    plan_id: str; predicted_ms: float; backend: dict; includes_host_submission: bool
    timing_basis: str
@_R
class GapEvidence:
    cause: str; basis: str; op_ids: tuple[str, ...]; description: str; delta_ms: float | None
@_R
class RequestMetric:
    name: str; status: str; request_id: str | None; scope: str; statistic: str
    sol_ms: float | None; predicted_ms: float | None; fixed_decisions: bool; reason: str | None
@_R
class Counterfactual:
    id: str; feature: str; status: str; baseline_plan_id: str; candidate_plan_id: str | None
    patches: tuple[Patch, ...]; preconditions: tuple[str, ...]; predicted_ms: float | None
    sol_ms: float | None; saved_ms: float | None; sol_saved_ms: float | None; reason: str | None
@_R
class LaunchParseReport:
    schema_version: str; status: str; source: SourceIdentity; input_format: str
    command_index: int; framework_version: str | None; adapter_id: str | None; strict: bool
    setting_ids: tuple[str, ...]; applied: tuple[Setting, ...]
    neutral_operational: tuple[Setting, ...]; unmodeled_or_unsupported: tuple[Setting, ...]
    required_inputs: tuple[RequiredInput, ...]; effective_overrides: tuple[Patch, ...]
    assumptions: tuple[str, ...]; case: dict | None; coverage: str
    diagnostics: tuple[Diagnostic, ...]
@_R
class SolComparisonReport:
    schema_version: str; status: str; context: Context; bound: Bound
    physical_dispatch_bound: Bound | None; prediction: Prediction | None
    efficiency_vs_sol_pct: float | None; gap_evidence: tuple[GapEvidence, ...]
    request_metrics: tuple[RequestMetric, ...]; counterfactuals: tuple[Counterfactual, ...]
    diagnostics: tuple[Diagnostic, ...]; counterfactuals_additive: bool = False

    def throughput_tokens_per_s(self) -> float | None:
        """Derived decode throughput; not serialized as a new contract field."""
        if self.prediction is None or self.prediction.predicted_ms <= 0:
            return None
        return (self.context.sequences * self.context.tokens * 1000.0
                / self.prediction.predicted_ms)


def _secret_paths(value, path=""):
    if isinstance(value, dict):
        return [child for key, item in value.items()
                for child in ([f"{path}/{key}"]
                              if (any(marker in str(key).lower() for marker in _SECRET_MARKERS)
                                  or str(key).lower().endswith("token")) and item != "[REDACTED]"
                              else []) + _secret_paths(item, f"{path}/{key}")]
    if isinstance(value, (list, tuple)):
        return [child for index, item in enumerate(value)
                for child in _secret_paths(item, f"{path}/{index}")]
    return []


def _all_strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for key, item in value.items()
                for text in [str(key)] + _all_strings(item)]
    if isinstance(value, (list, tuple)):
        return [text for item in value for text in _all_strings(item)]
    return []


def _credential_string(text):
    return bool(_CREDENTIAL_RE.search(str(text)))


def _has_credential_in_payload(value) -> bool:
    if value is None or type(value) in _SCALAR_TYPES:
        return bool(_CREDENTIAL_RE.search(value)) if type(value) is str else False
    if isinstance(value, str):
        return bool(_CREDENTIAL_RE.search(value))
    if type(value) in (dict, list, tuple):
        try:
            return bool(_CREDENTIAL_RE.search(json.dumps(value, ensure_ascii=False)))
        except TypeError:
            pass
    if isinstance(value, dict):
        for key, item in value.items():
            if _credential_string(key) or _has_credential_in_payload(item):
                return True
        return False
    if isinstance(value, (list, tuple)):
        for item in value:
            if _has_credential_in_payload(item):
                return True
        return False
    return False


def _span_errors(span, label, errors):
    keys = ("line_start", "column_start", "line_end", "column_end")
    if not isinstance(span, dict) or not span.get("source") or any(
            not isinstance(span.get(key), int) or span[key] < 1 for key in keys):
        errors.append(f"{label}:span_invalid"); return
    if (span["line_end"], span["column_end"]) < (span["line_start"], span["column_start"]):
        errors.append(f"{label}:span_order")


def _launch_report_dict(report):
    if isinstance(report, dict):
        return report, None
    if type(report) is LaunchParseReport and type(report.case) is dict:
        try:
            dumped = json.dumps(report.case, allow_nan=False, ensure_ascii=False)
        except (TypeError, ValueError):
            return _plain(report), None
        shallow = _plain(LaunchParseReport(
            report.schema_version, report.status, report.source, report.input_format,
            report.command_index, report.framework_version, report.adapter_id,
            report.strict, report.setting_ids, report.applied,
            report.neutral_operational, report.unmodeled_or_unsupported,
            report.required_inputs, report.effective_overrides, report.assumptions,
            None, report.coverage, report.diagnostics))
        shallow["case"] = report.case
        return shallow, dumped
    return _as_dict(report), None


def validate_launch_report(report) -> tuple[str, ...]:
    """L01-L03 plus status/redaction checks; raise on the first invalid report."""
    data, dumped_case = _launch_report_dict(report)
    errors = []
    if data.get("schema_version") != "launch-parse/1":
        errors.append("L00:schema_version")
    status = data.get("status")
    if status not in {"completed", "partial", "needs_inputs", "failed"}:
        errors.append("L00:status")
    ids = list(data.get("setting_ids", ()))
    if len(ids) != len(set(ids)):
        errors.append("L01:setting_ids_duplicate")
    buckets = {name: list(data.get(name, ())) for name in
               ("applied", "neutral_operational", "unmodeled_or_unsupported")}
    all_ids = [item.get("id") for rows in buckets.values() for item in rows]
    if len(all_ids) != len(set(all_ids)):
        errors.append("L01:bucket_ids_not_disjoint")
    if set(ids) != set(all_ids):
        errors.append("L01:bucket_ids_do_not_partition_setting_ids")
    known = set(ids)
    for bucket, rows in buckets.items():
        for index, row in enumerate(rows):
            label = f"L01:{bucket}[{index}]"
            if not row.get("id"):
                errors.append(f"{label}:id_missing"); continue
            for key in ("parent_id", "shadowed_by"):
                reference = row.get(key)
                if reference is not None and reference not in known:
                    errors.append(f"{label}:{key}_unresolved:{reference}")
            if row.get("effective") is True and row.get("shadowed_by") is not None:
                errors.append(f"{label}:effective_setting_shadowed")
            if row.get("effective") is False and row.get("shadowed_by") is None:
                errors.append(f"{label}:ineffective_setting_without_shadow")
            _span_errors(row.get("span"), label, errors)
            allowed = {"applied": {"modeled", "no_op", "shadowed"},
                       "neutral_operational": {"operational", "shadowed"},
                       "unmodeled_or_unsupported": {"unmodeled", "shadowed"}}[bucket]
            if row.get("effect") not in allowed:
                errors.append(f"{label}:effect_invalid")
            if bucket == "applied" and not row.get("targets"):
                errors.append(f"{label}:applied_requires_targets")
            if bucket == "neutral_operational" and row.get("targets"):
                errors.append(f"{label}:operational_requires_no_targets")
            if row.get("kind") == "json_leaf" and not str(row.get("pointer") or "").startswith("/"):
                errors.append(f"{label}:json_leaf_pointer")
            if row.get("redacted") is True and row.get("value") != "[REDACTED]":
                errors.append(f"{label}:secret_not_redacted")
            if not row.get("redacted") and _secret_paths(row.get("value")):
                errors.append(f"{label}:secret_path_not_redacted")
    for index, item in enumerate(data.get("diagnostics", ())):
        label = f"L03:diagnostics[{index}]"
        if not item.get("code") or not item.get("message"):
            errors.append(f"{label}:missing_code_or_message")
        text = str(item.get("message", "")).lower()
        if any(marker in text for marker in ("bearer ", "sk-", "password=")):
            errors.append(f"{label}:secret_in_message")
    payloads = [data.get("case"), data.get("effective_overrides"),
                data.get("assumptions")]
    for index, payload in enumerate(payloads):
        has_cred = (bool(_CREDENTIAL_RE.search(dumped_case))
                    if index == _ZERO_INT and dumped_case is not None
                    else _has_credential_in_payload(payload))
        if has_cred:
            errors.append(f"L03:payload[{index}]:secret_in_payload")
    required, unsupported = data.get("required_inputs", ()), buckets["unmodeled_or_unsupported"]
    rules = {
        "completed": (data.get("case") is not None and not required and not unsupported
                      and data.get("coverage") == "exact_declared_inputs"),
        "needs_inputs": (data.get("case") is None and bool(required)
                         and data.get("coverage") == "unresolved"),
        "partial": (data.get("case") is not None and bool(unsupported) and not required
                    and data.get("coverage") == "conditional_modeled_subset"
                    and not data.get("strict")),
        "failed": (data.get("case") is None and bool(data.get("diagnostics"))),
    }
    if status in rules and not rules[status]:
        errors.append(f"L02:status_invariant:{status}")
    if errors:
        raise ReportValidationError(errors)
    return tuple(errors)


def validate_sol_report(report) -> tuple[str, ...]:
    """S01-S05 arithmetic/reference checks for the comparison report."""
    data, errors = _as_dict(report), []
    if data.get("schema_version") != "sol-comparison/1":
        errors.append("S00:schema_version")
    bound, context = data.get("bound") or {}, data.get("context") or {}
    prediction = data.get("prediction")
    for index, row in enumerate(bound.get("resources", ())):
        label, total = f"S01:resources[{index}]", 0.0
        for term_index, term in enumerate(row.get("terms", ())):
            if term.get("physical_resource_id") != row.get("physical_resource_id"):
                errors.append(f"{label}:term_resource_mismatch:{term_index}")
            peak = float(term.get("peak_per_ms") or 0)
            if peak <= 0:
                errors.append(f"{label}:peak_nonpositive:{term_index}"); continue
            if not _close(term.get("service_ms"), float(term.get("work") or 0) / peak):
                errors.append(f"{label}:service_not_work_over_peak:{term_index}")
            total += float(term.get("service_ms") or 0)
        if not _close(row.get("load_bound_ms"), total):
            errors.append(f"{label}:load_not_sum_of_terms")
    resource_load = max((float(row.get("load_bound_ms") or 0)
                         for row in bound.get("resources", ())), default=0.0)
    if bound.get("status") == "available":
        if not _close(bound.get("resource_load_ms"), resource_load):
            errors.append("S02:resource_load_not_max_row")
        if not _close(bound.get("sol_ms"),
                      max(float(bound.get("critical_path_ms") or 0), resource_load),
                      SOL_TOLERANCE_MS):
            errors.append("S02:sol_not_max_cp_and_load")
    for index, local in enumerate(bound.get("local_bounds", ())):
        grouped = {}
        for item in local.get("demands", ()):
            key = item.get("physical_resource_id")
            grouped[key] = grouped.get(key, 0.0) + float(item.get("service_ms") or 0)
        resource = max(grouped.values(), default=0.0)
        expected = max(resource, float(local.get("phase_precedence_bound_ms") or 0))
        if (not _close(local.get("resource_bound_ms"), resource)
                or not _close(local.get("lower_bound_ms"), expected)):
            errors.append(f"S02:local[{index}]:bounds")
    if prediction is not None:
        if prediction.get("plan_id") != context.get("plan_id"):
            errors.append("S03:plan_id_mismatch")
        predicted, sol = float(prediction.get("predicted_ms") or 0), bound.get("sol_ms")
        if sol is not None and float(sol) > predicted + SOL_TOLERANCE_MS:
            errors.append("S03:bound_exceeds_prediction")
        if predicted > 0 and sol is not None:
            if not _close(data.get("efficiency_vs_sol_pct"), 100.0 * float(sol) / predicted):
                errors.append("S03:efficiency_not_100_sol_over_predicted")
        elif sol == 0 and data.get("efficiency_vs_sol_pct") is not None:
            errors.append("S03:zero_over_zero_efficiency_must_be_null")
    for index, counter in enumerate(data.get("counterfactuals", ())):
        label = f"S04:counterfactuals[{index}]"
        if counter.get("baseline_plan_id") != context.get("plan_id"):
            errors.append(f"{label}:baseline_plan_mismatch")
        deltas = (counter.get("saved_ms"), counter.get("sol_saved_ms"))
        if counter.get("status") == "evaluated":
            if not counter.get("candidate_plan_id"):
                errors.append(f"{label}:candidate_plan_missing")
            candidate_values = (counter.get("predicted_ms"), counter.get("sol_ms"))
            if None in deltas or None in candidate_values:
                errors.append(f"{label}:evaluated_delta_missing")
            elif any(not isinstance(value, (int, float)) or not math.isfinite(float(value))
                     for value in candidate_values + deltas) or any(
                         float(value) < 0 for value in candidate_values):
                errors.append(f"{label}:nonfinite_or_negative_value")
            else:
                if float(counter["sol_ms"]) > float(counter["predicted_ms"]) + SOL_TOLERANCE_MS:
                    errors.append(f"{label}:candidate_bound_exceeds_prediction")
                if prediction is not None and bound.get("sol_ms") is not None:
                    expected = (float(prediction["predicted_ms"]) - float(counter["predicted_ms"]),
                                float(bound["sol_ms"]) - float(counter["sol_ms"]))
                    if any(not _close(a, b) for a, b in zip(deltas, expected)):
                        errors.append(f"{label}:saved_arithmetic")
        else:
            if any(value is not None for value in deltas):
                errors.append(f"{label}:delta_must_be_null")
            if not counter.get("reason"):
                errors.append(f"{label}:reason_required")
    for index, metric in enumerate(data.get("request_metrics", ())):
        label, values = f"S05:request_metrics[{index}]", (metric.get("sol_ms"),
                                                         metric.get("predicted_ms"))
        if metric.get("status") == "available":
            if (metric.get("scope") not in {"single_request_no_queue", "fixed_trace"}
                    or metric.get("fixed_decisions") is not True):
                errors.append(f"{label}:scope_or_fixed_decisions")
            if None in values:
                errors.append(f"{label}:values_required")
            elif not all(isinstance(value, (int, float))
                         and math.isfinite(float(value)) for value in values):
                errors.append(f"{label}:nonfinite_latency")
            elif float(values[0]) <= 0 or float(values[1]) <= 0:
                errors.append(f"{label}:nonpositive_latency")
            elif (metric.get("name") == "ttft"
                  and float(metric["predicted_ms"])
                  < float(metric["sol_ms"]) - SOL_TOLERANCE_MS):
                errors.append(f"{label}:ttft_below_prefill_sol")
            elif (metric.get("name") == "tpot"
                  and float(metric["sol_ms"])
                  > float(metric["predicted_ms"]) + SOL_TOLERANCE_MS):
                errors.append(f"{label}:tpot_below_decode_sol")
        elif metric.get("status") == "unavailable" and (
                None not in values or not metric.get("reason")):
            errors.append(f"{label}:unavailable_values_or_reason")
    if data.get("status") == "completed" and (
            prediction is None or bound.get("status") != "available"):
        errors.append("S03:completed_requires_prediction_and_available_bound")
    if errors:
        raise ReportValidationError(errors)
    return tuple(errors)

def _md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |"
              for row in rows]
    return "\n".join(lines)


def launch_markdown(report) -> str:
    if isinstance(report, LaunchParseReport):
        rows = [(item.id, item.name, _plain(item.value), item.effect,
                 ",".join(item.targets or ()) or "-", item.reason)
                for bucket in (report.applied, report.neutral_operational,
                               report.unmodeled_or_unsupported)
                for item in bucket]
        src_name = report.source.name if report.source is not None else None
        lines = [f"# Launch inspection: {report.status}", "",
                 f"- Source: `{src_name}`",
                 f"- Adapter: `{report.adapter_id}`",
                 f"- Coverage: `{report.coverage}`", "",
                 _md_table(("ID", "Setting", "Value", "Effect", "Targets", "Reason"), rows)]
        lines += [f"\n## Required inputs\n\n- `{item.name}`: {item.reason}"
                  for item in report.required_inputs]
        lines += [f"\n## Diagnostics\n\n- `{item.code}` ({item.severity}): {item.message}"
                  for item in report.diagnostics]
        return "\n".join(lines) + "\n"
    data = _as_dict(report)
    rows = [(item.get("id"), item.get("name"), item.get("value"), item.get("effect"),
             ",".join(item.get("targets") or ()) or "-", item.get("reason"))
            for bucket in ("applied", "neutral_operational", "unmodeled_or_unsupported")
            for item in data.get(bucket, ())]
    lines = [f"# Launch inspection: {data.get('status')}", "",
             f"- Source: `{(data.get('source') or {}).get('name')}`",
             f"- Adapter: `{data.get('adapter_id')}`", f"- Coverage: `{data.get('coverage')}`", "",
             _md_table(("ID", "Setting", "Value", "Effect", "Targets", "Reason"), rows)]
    lines += [f"\n## Required inputs\n\n- `{item['name']}`: {item['reason']}"
              for item in data.get("required_inputs", ())]
    lines += [f"\n## Diagnostics\n\n- `{item['code']}` ({item['severity']}): {item['message']}"
              for item in data.get("diagnostics", ())]
    return "\n".join(lines) + "\n"


def _counterfactual_table(rows):
    headers = ("Feature", "Status", "Candidate", "Predicted ms", "SOL ms", "Saved ms",
               "SOL saved ms", "Reason")
    return _md_table(headers, [(item.get("feature"), item.get("status"),
                                item.get("candidate_plan_id"), item.get("predicted_ms"),
                                item.get("sol_ms"), item.get("saved_ms"),
                                item.get("sol_saved_ms"), item.get("reason")) for item in rows])


def sol_markdown(report) -> str:
    data = _as_dict(report)
    bound, prediction, context = (data.get("bound") or {}, data.get("prediction") or {},
                                  data.get("context") or {})
    throughput = (float(context.get("sequences") or 0) * float(context.get("tokens") or 0)
                  * 1000.0 / float(prediction["predicted_ms"])
                  if prediction.get("predicted_ms") else None)
    lines = [f"# SOL comparison: {data.get('status')}", "",
             f"- Plan: `{context.get('plan_id')}`",
             f"- SOL ({bound.get('dispatch_policy')}): `{bound.get('sol_ms')}` ms",
             f"- Critical path: `{bound.get('critical_path_ms')}` ms",
             f"- Resource load: `{bound.get('resource_load_ms')}` ms",
             f"- Engine prediction: `{prediction.get('predicted_ms')}` ms",
             f"- Efficiency vs SOL: `{data.get('efficiency_vs_sol_pct')}` %",
             f"- Derived throughput: `{throughput}` tokens/s", "",
             _md_table(("Constraint", "Load ms", "Terms"),
                       [(item.get("physical_resource_id"), item.get("load_bound_ms"),
                         len(item.get("terms") or ())) for item in bound.get("resources", ())])]
    lines += [f"\n## Gap evidence\n\n- `{item['cause']}` ({item['basis']}, "
              f"{item['delta_ms']} ms): {item['description']}" for item in data.get("gap_evidence", ())]
    if data.get("counterfactuals"):
        lines += ["\n## Feature deltas", "", _counterfactual_table(data["counterfactuals"])]
    return "\n".join(lines) + "\n"

def feature_markdown(rows) -> str:
    """Markdown for a counterfactual tuple or a SolComparisonReport."""
    data = _as_dict(rows)
    items = data.get("counterfactuals") if isinstance(data, dict) else data
    return "# Feature counterfactuals\n\n" + _counterfactual_table(items or ()) + "\n"


__all__ = [
    "ARITHMETIC_TOLERANCE", "SOL_TOLERANCE_MS", "Bound", "Context", "Counterfactual",
    "Demand", "Diagnostic", "GapEvidence", "LaunchParseReport", "LocalBound",
    "Patch", "Prediction", "ReportValidationError", "RequestMetric", "RequiredInput",
    "ResourceLoad", "Setting", "SolComparisonReport", "SourceIdentity", "Span",
    "canonical_json", "feature_markdown", "launch_markdown", "sol_markdown", "to_dict",
    "validate_launch_report", "validate_sol_report",
]
