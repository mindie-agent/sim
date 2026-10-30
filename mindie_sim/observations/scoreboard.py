"""Descriptive model comparisons with explicit workload and timing boundaries.

[Role] Assemble measured step spans, independent predictions and cost leaders.
[Invariants] Capture spans and activity sums never become step latency. Unknown
  conditions remain unknown; binary provenance is not a simulation admission
  gate. Signed cancellation cannot hide absolute residual. No fitting occurs.
[Timeouts & Locks] Finite, already aggregated records; no waits or locks.
[Side-Effect Policy] Pure report construction, no devices, files or calibration.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .scoreboard_accuracy import annotate_call, group_fields, policy, step_fields


CONDITION_FIELDS = (
    "model", "hardware", "parallel", "phase", "tokens", "context", "sequences",
    "precision", "branch",
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
    unknown = sorted({item for key in CONDITION_FIELDS
                      for side in (left, right)
                      for item in _unknowns(side.get(key), key)})
    differences = [item for key in CONDITION_FIELDS
                   for item in _differences(left.get(key), right.get(key), key)]
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
