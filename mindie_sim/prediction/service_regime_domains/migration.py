"""Evaluate explicit service migration bindings.

[Role] Check fixed effects, work associations and replacement evidence.
[Invariants] Candidate conditions and provenance stay distinct from physical delivery.
[Timeouts & Locks] Bounded by the supplied program and services. No waits or locks.
[Side-Effect Policy] Returns migration evidence and coverage diagnostics.
"""
from __future__ import annotations

from collections.abc import Mapping

from mindie_sim.schema.common import ModelingError
from ..response_functions import (
    nonnegative,
    positive,
    provenance_roles,
    same_declared_value,
)

from .work import (
    _attr,
    _evaluate_work_reference,
    _fragment_for,
    _resolve_actual_work,
    _resource_identity,
    _service_stage_overlap,
    _stage,
    _undelivered_stage_work,
)


_MISSING = object()
_FIXED_EVIDENCE_KINDS = frozenset({
    "legacy_whole_call", "whole_call", "whole-call", "observed_replay",
    "legacy_observation"})


def _condition_check(conditions, context):
    """Use T06's declared-condition semantics; never a weaker local matcher."""
    if not conditions:
        return {"applies": True, "checks": [], "policy": "no_declared_conditions"}
    from mindie_sim.observations.calibration import service_conditions_match
    return service_conditions_match({"conditions": dict(conditions)}, dict(context or {}))


def _actual_native(plan, rule_execution, context):
    values = [plan.get("native_boundary")]
    identity = _attr(context, "service_identity", {}) or {}
    values.append(identity.get("kernel") if isinstance(identity, Mapping) else None)
    if isinstance(rule_execution, Mapping):
        values.append(rule_execution.get("native_kernel"))
    present = sorted({str(value).strip() for value in values if value})
    if len(present) > 1:
        return None, present
    return (present[0] if present else None), present


def _service_problem(record, kind, native):
    problems = []
    if record["boundary"] not in {"resource_phase", "device_kernel"}:
        problems.append("service_boundary_must_be_resource_phase_or_device_kernel")
    if not record["phase"]:
        problems.append("service_requires_phase")
    if not record["work_unit"]:
        problems.append("service_requires_work_unit")
    try:
        positive(record["effective_rate"])
    except (ModelingError, TypeError):
        problems.append("service_effective_rate_requires_positive_finite")
    ident = record["identification"]
    if kind == "independent_service":
        if record["scope"] != "independent_resource":
            problems.append("service_scope_does_not_match_replacement_kind")
        if (ident.get("status") != "identified"
                or ident.get("resource_identified") is not True):
            problems.append("service_does_not_declare_identified_resource")
        if len(record["resources"]) != 1:
            problems.append("independent_service_requires_exactly_one_resource")
    else:
        if record["scope"] != "joint_phase":
            problems.append("service_scope_does_not_match_replacement_kind")
        if ident.get("status") != "joint_identified":
            problems.append("joint_service_does_not_declare_joint_identified_status")
        if not record["resources"]:
            problems.append("joint_service_requires_resources")
        if not record["native_boundary"]:
            problems.append("joint_service_requires_native_boundary")
    if native and record["native_boundary"] and record["native_boundary"] != native:
        problems.append("service_native_boundary_does_not_match_actual_program")
    return problems


def _fixed_effect_text(record):
    identification = record.get("identification") or {}
    provenance = record.get("provenance") or {}
    fit = provenance.get("fit") if isinstance(provenance.get("fit"), Mapping) else {}
    return (identification.get("included_fixed_effects")
            or provenance.get("included_effects")
            or (fit or {}).get("included_fixed_effects"))


def _fixed_occurrence_stage(plan, stage_name):
    """Actual stage carrying the fixed latency for one declared fixed row.

    T04's real resource-service lowering puts the resolved startup on a
    latency-only ``setup`` stage before the work ``service`` stage.  A fixed
    component therefore belongs to that actual occurrence, not to every work
    term that shares its stage.
    """
    by_name = {getattr(stage, "name", None): stage
               for stage in plan.get("stages", ())}
    stage = by_name.get(stage_name)
    if stage is None:
        return None
    if not (getattr(stage, "work", {}) or {}) and getattr(stage, "latency", 0) > 0:
        return stage_name
    for name in getattr(stage, "after", ()) or ():
        prior = by_name.get(name)
        if (prior is not None and not (getattr(prior, "work", {}) or {})
                and getattr(prior, "latency", 0) > 0):
            return name
    return stage_name


def _first_invocation_stage(plan):
    for stage in plan.get("stages", ()):
        if (not (getattr(stage, "work", {}) or {})
                and getattr(stage, "latency", 0) > 0
                and getattr(stage, "fragment", None) is None):
            return getattr(stage, "name", None)
    stages = tuple(plan.get("stages", ()) or ())
    return getattr(stages[0], "name", None) if stages else None


def _invocation_evidence(plan):
    """Read T04's actual invocation timing from the selected context."""
    context = plan.get("context")
    timing = _attr(context, "invocation_timing")
    if timing is None:
        return None
    try:
        startup_us = nonnegative(_attr(timing, "startup_ms", 0)) * 1000.0
        minimum_us = nonnegative(_attr(timing, "minimum_ms", 0)) * 1000.0
    except ModelingError as exc:
        raise ModelingError("migration_actual_invocation_timing_invalid") from exc
    provenance = _attr(timing, "provenance", {}) or {}
    return {"startup_us": startup_us, "minimum_us": minimum_us,
            "provenance": dict(provenance) if isinstance(provenance, Mapping) else {}}


def _fixed_check_stage(plan, stage_name):
    """Work stage whose phase/resources own a fixed component.

    A fixed row may name the latency-only invocation stage (the actual fixed
    occurrence) or the work stage.  Resource and phase checks always use the
    associated work stage, never the latency-only setup itself.
    """
    stage = _stage(plan, stage_name)
    if stage is None:
        return None
    if (getattr(stage, "work", {}) or {}) or _fragment_for(plan, stage) is not None:
        return stage
    for candidate in plan.get("stages", ()):
        if stage_name in (getattr(candidate, "after", ()) or ()) and (
                (getattr(candidate, "work", {}) or {})
                or _fragment_for(plan, candidate) is not None):
            return candidate
    return stage


def _invocation_fixed_occurrence(plan, invocation):
    """One independently declared T04 invocation startup, charged once."""
    if not invocation or invocation["startup_us"] <= 0:
        return None
    occurrence_stage = _first_invocation_stage(plan)
    return {
        "kind": "invocation_startup", "service_id": None,
        "stage": occurrence_stage, "actual_stage": occurrence_stage,
        "phase": None, "fixed_us": invocation["startup_us"],
        "native_boundary": "",
        "included_fixed_effects": (
            "T04 actual invocation startup declared separately from the "
            "selected service formula and charged once"),
        "fixed_components": "invocation_startup",
        "covers_invocation_startup": True,
        "invocation_startup_us": invocation["startup_us"],
    }


def _fixed_rows(plan, services, stage_work_phases, legacy_fixed, invocation=None):
    rows, occurrences, conflicts, notes = [], [], [], []
    names = {stage["name"] for stage in plan.get("program_view") or ()}
    if not plan.get("fixed"):
        if legacy_fixed > 0 or any(
                term.get("kind") == "zero_slope"
                for term in plan.get("decomposition", {}).get("terms", ())):
            conflicts.append("legacy_fixed_component_unbound")
        else:
            occurrence = _invocation_fixed_occurrence(plan, invocation)
            if occurrence is not None:
                occurrences.append(occurrence)
        return rows, occurrences, conflicts, notes
    if not names:
        conflicts.append("fixed_binding_requires_actual_stage")
        return rows, occurrences, conflicts, notes
    seen = set()
    for item in plan["fixed"]:
        sid = item["service_id"]
        record = services.get(sid)
        stage = item.get("stage")
        if stage is None and len(names) == 1:
            stage = next(iter(names))
        if stage is None or stage not in names:
            conflicts.append("fixed_binding_requires_actual_stage:" + sid)
            continue
        if record is None:
            conflicts.append("unknown_fixed_service:" + sid)
            continue
        actual_stage = _fixed_occurrence_stage(plan, stage)
        check_stage = _fixed_check_stage(plan, stage)
        phases = stage_work_phases.get(getattr(check_stage, "name", stage), set())
        if phases and record["phase"] not in phases:
            conflicts.append("fixed_phase_does_not_match_stage_work:" + sid)
        if record["resources"] and not _service_stage_overlap(
                plan, check_stage, record["resources"]):
            conflicts.append("fixed_service_resource_not_on_actual_stage:" + sid)
        if (actual_stage, sid) in seen:
            conflicts.append("fixed_boundary_duplicated:" + str(actual_stage) + ":" + sid)
            continue
        seen.add((actual_stage, sid))
        covers_invocation = bool(
            invocation and invocation["startup_us"] > 0
            and record["fixed_us"] >= invocation["startup_us"] - 1e-9)
        rows.append({"service_id": sid, "stage": stage,
                     "phase": record["phase"], "new_fixed_us": record["fixed_us"],
                     "native_boundary": record["native_boundary"]})
        if record["fixed_us"] > 0:
            occurrences.append({
                "kind": "service_fixed", "service_id": sid, "stage": stage,
                "actual_stage": actual_stage, "phase": record["phase"],
                "fixed_us": record["fixed_us"],
                "native_boundary": record["native_boundary"],
                "included_fixed_effects": _fixed_effect_text(record),
                "fixed_components": record["identification"].get("fixed_components"),
                "covers_invocation_startup": covers_invocation,
                "invocation_startup_us": (invocation or {}).get("startup_us"),
            })
    by_stage = {}
    for occurrence in occurrences:
        if occurrence["kind"] != "service_fixed":
            continue
        by_stage.setdefault(occurrence["actual_stage"], []).append(
            occurrence["service_id"])
    for stage, ids in by_stage.items():
        if len(ids) > 1:
            conflicts.append("multiple_fixed_components_on_stage:" + str(stage))
    if invocation and invocation["startup_us"] > 0 and not any(
            occurrence["covers_invocation_startup"] for occurrence in occurrences):
        occurrence = _invocation_fixed_occurrence(plan, invocation)
        if occurrence is not None:
            occurrences.append(occurrence)
    if legacy_fixed == 0 and rows:
        notes.append("new_fixed_without_legacy_fixed")
    return rows, occurrences, conflicts, notes


def _identification_problems(record):
    ident = record.get("identification") or {}
    fit = (record.get("provenance") or {}).get("fit") or {}
    rate_source = ident.get("rate_source")
    fixed_source = ident.get("fixed_source")
    train_work = (ident.get("train_work") or fit.get("training_work")
                  or fit.get("train_work") or [])
    try:
        distinct = len({float(work) for work in train_work})
    except (TypeError, ValueError):
        distinct = 0
    problems = []
    if rate_source == "fitted_with_declared_fixed":
        # Fixed is known/declared; one work point identifies the remaining rate.
        rate_ok = distinct >= 1
        fixed_ok = fixed_source == "declared"
    elif rate_source == "declared":
        # Rate is known/declared; one work point identifies the remaining fixed.
        rate_ok = True
        fixed_ok = fixed_source == "fitted" and ident.get("fixed_identified") is True
    elif rate_source == "fitted":
        # Rate and fixed are both unknown: two work sizes are required.
        rate_ok = distinct >= 2
        fixed_ok = fixed_source == "fitted" and ident.get("fixed_identified") is True
    else:
        rate_ok = False
        fixed_ok = False
    if not rate_ok:
        problems.append("replacement_rate_not_independently_identified")
    if not fixed_ok:
        problems.append("replacement_fixed_not_independently_identified")
    return problems


def evaluate_migration_binding(decomposition, rule_match, rule_id, plan, services, *,
                               execution=None, old_provenance_negative=False,
                               old_training_tokens=(), coordinate_declarations=None):
    """One evaluation of actual stage/work/fixed ownership and source roles.

    ``plan`` carries read-only ``KernelProgram``/``Stage``/``RecipeContext``
    references. Old term indices and formulas are diagnostics; actual stage
    work, unit, owner and boundary are the authority. Missing program/work or
    training evidence is pending/replay, never ready.
    """
    execution = execution if isinstance(execution, Mapping) else {}
    if not plan.get("program_view") and (plan["work"] or plan["fixed"]):
        return {"status": "awaiting_program_binding",
                "blockers": ["migration_program_binding_required"],
                "conflicts": [], "unresolved_conditions": [],
                "coverage": {"complete": False, "program": None,
                             "work_terms": [], "external_work": [], "fixed": []},
                "service_evidence": {}, "conditions": {},
                "diagnostics": {"notes": ["actual_program_stage_reference_required"]},
                "scope": {"kind": plan["scope"],
                          "slice_conditions": dict(plan.get("slice_conditions") or {})},
                "replacement_scope": None}
    context = plan.get("context")
    native, native_values = _actual_native(plan, execution, context)
    kind, scope = plan["replacement_kind"], plan["scope"]
    context_values = plan.get("condition_context") or {}
    slice_conditions = plan.get("slice_conditions") or {}
    conflicts, unresolved, notes, rate_rows = [], [], [], []
    identification_pending, service_evidence = [], {}
    referenced = {item["service_id"] for item in plan["work"]}
    referenced |= {item["service_id"] for item in plan["fixed"]}
    for sid in sorted(referenced):
        record = services.get(sid)
        if record is None:
            conflicts.append("unknown_service:" + sid)
            continue
        problems = _service_problem(record, kind, native)
        provenance = record["provenance"]
        roles = provenance_roles(provenance)
        training = roles["training"] or roles["other"]
        source_problems = []
        if not training:
            source_problems.append("replacement_service_requires_training_sources")
        source_problems.extend(_identification_problems(record))
        legacy_overlap = sorted(set(training) & set(old_training_tokens))
        if old_provenance_negative and legacy_overlap:
            notes.append("legacy_training_point_shared_with_independent_identification")
        if str(provenance.get("kind") or "").lower() in _FIXED_EVIDENCE_KINDS:
            source_problems.append("replacement_service_is_not_independent_evidence")
        checks = _condition_check(record["conditions"], context_values)
        missing_paths = {check["path"] for check in checks["checks"]
                         if not check["present"]}
        for check in checks["checks"]:
            if not check["present"]:
                unresolved.append({"scope": "service", "service_id": sid, **check})
            elif not check["matched"]:
                conflicts.append("service_condition_mismatch:" + sid + ":" + check["path"])
        for path, expected in record["conditions"].items():
            if path == "regime_rule_id":
                if str(expected) != str(rule_id):
                    conflicts.append("service_regime_rule_id_mismatch:" + sid)
                continue
            if path in {"native_boundary", "native_kernel"} or path in missing_paths:
                continue
            rule_expected = rule_match.get(path, _MISSING)
            slice_expected = slice_conditions.get(path, _MISSING)
            if rule_expected is not _MISSING and same_declared_value(rule_expected, expected):
                continue
            if scope == "slice" and slice_expected is not _MISSING and same_declared_value(
                    slice_expected, expected):
                continue
            conflicts.append("service_condition_outside_rule_scope:" + sid + ":" + path)
        service_evidence[sid] = {
            "service_id": sid, "scope": record["scope"],
            "resources": list(record["resources"]), "phase": record["phase"],
            "work_unit": record["work_unit"], "boundary": record["boundary"],
            "effective_rate": record["effective_rate"],
            "aggregate_rate": record.get("aggregate_rate"),
            "per_owner_rate": record.get("per_owner_rate"),
            "rate_scope": record["rate_scope"], "active_units": record.get("active_units"),
            "fixed_us": record["fixed_us"], "native_boundary": record["native_boundary"],
            "identification": dict(record["identification"]),
            "rate_source": record["identification"].get("rate_source"),
            "fixed_source": record["identification"].get("fixed_source"),
            "train_work": record["identification"].get("train_work"),
            "fixed_components": record["identification"].get("fixed_components"),
            "included_fixed_effects": (
                record["identification"].get("included_fixed_effects")
                or (record["provenance"] or {}).get("included_effects")
                or (record["provenance"].get("fit") or {}).get("included_fixed_effects")),
            "provenance": dict(provenance), "source_roles": roles,
            "legacy_training_overlap": legacy_overlap,
            "conditions": checks, "problems": sorted(set(problems)),
            "source_problems": sorted(set(source_problems)),
            "valid": not problems, "identification_ready": not source_problems}
        conflicts.extend(sid + ":" + problem for problem in problems)
        identification_pending.extend(sid + ":" + problem for problem in source_problems)
    rule_check = _condition_check(rule_match or {}, context_values)
    slice_check = (_condition_check(slice_conditions, context_values) if slice_conditions
                   else {"applies": True, "checks": [], "policy": "no_slice_conditions"})
    for scope_name, check in (("rule", rule_check), ("slice", slice_check)):
        for item in check["checks"]:
            if not item["present"]:
                unresolved.append({"scope": scope_name, **item})
            elif not item["matched"]:
                conflicts.append(scope_name + "_condition_mismatch:" + item["path"])
    if len(native_values) > 1:
        conflicts.append("actual_native_boundary_conflict:" + ",".join(native_values))
    terms = [term for term in decomposition.get("terms", ())
             if term.get("kind") in {"resource_rate", "unidentified_slope",
                                     "declared_resource_supply"}]
    legacy_fixed = float(decomposition.get("fixed_us", 0))
    stage_map = {getattr(stage, "name", None): stage
                 for stage in plan.get("stages", ())}
    resolved = _resolve_actual_work(plan)
    actual_instances = resolved["instances"]
    invocation = _invocation_evidence(plan)
    undelivered_work = _undelivered_stage_work(plan, resolved)
    evaluations, bound_stages, instance_services = [], {}, {}
    for item in plan["work"]:
        evaluation = _evaluate_work_reference(
            plan, item, stage_map, services, service_evidence, context, native,
            resolved)
        evaluations.append(evaluation)
        if evaluation["instance"] is not None:
            instance_services.setdefault(evaluation["instance"], set()).add(
                item["service_id"])
            bound_stages.setdefault(item["stage"], set()).add(item["phase"])
        if evaluation["undelivered"]:
            projected = evaluation["projected"] or {}
            undelivered_work.append({
                "stage": item["stage"], "resource": item["resource"],
                "physical_resource": projected.get("resource"),
                "reason": projected.get("basis", "actual_work_instance_missing")})
        prefix = ("term" + str(item["term_index"]) + ":"
                  if item["term_index"] is not None
                  else "external:" + str(item["stage"]) + ":")
        for problem in evaluation["problems"]:
            if problem == "binding_service_identification_pending":
                identification_pending.append(prefix + item["service_id"])
            elif problem in {"actual_work_instance_missing",
                             "actual_work_instance_ambiguous",
                             "actual_native_unit_missing"}:
                continue
            else:
                conflicts.append(prefix + problem)
    deduplicated, seen_undelivered = [], set()
    for entry in undelivered_work:
        key = (entry.get("stage"),
               entry.get("physical_resource") or entry.get("resource"))
        if key in seen_undelivered:
            continue
        seen_undelivered.add(key)
        deduplicated.append(entry)
    undelivered_work = deduplicated
    by_term = {}
    for evaluation in evaluations:
        index = evaluation["item"]["term_index"]
        if index is not None:
            by_term.setdefault(index, []).append(evaluation)
    work_rows, missing_terms, unbound_terms = [], [], []
    for term in terms:
        index = term["index"]
        bindings = by_term.get(index, [])
        row = {"term_index": index, "coordinate": term["coordinate"],
               "coordinate_declaration": dict(
                   (coordinate_declarations or {}).get(term["coordinate"], {})),
               "legacy_resource": term.get("resource"),
               "legacy_dtype": term.get("dtype"),
               "legacy_us_per_unit": term.get("us_per_unit"),
               "bound": False, "unbound": not bindings, "bindings": [],
               "problems": []}
        if len(bindings) > 1:
            row["problems"].append("work_term_bound_multiple_times")
            conflicts.append("term" + str(index) + ":work_term_bound_multiple_times")
        for evaluation in bindings:
            row["bindings"].append(dict(evaluation["item"]))
            row["problems"].extend(evaluation["problems"])
        if bindings:
            row["bound"] = (len(bindings) == 1 and not row["problems"]
                            and not bindings[0]["undelivered"]
                            and not bindings[0]["stage_missing"])
        else:
            term_aliases = _resource_identity(plan, term.get("resource"))[2]
            matching = [entry for entry in actual_instances.values()
                        if (_resource_identity(plan, entry["resource"])[2]
                            & term_aliases)]
            covered = any(
                instance_services.get(key)
                for key, entry in actual_instances.items()
                if (_resource_identity(plan, entry["resource"])[2]
                    & term_aliases))
            row["actual_resource_covered"] = bool(covered)
            if covered:
                row["diagnostic_only"] = True
            elif matching and any(entry["amount"] > 0 for entry in matching):
                missing_terms.append(index)
            else:
                row["diagnostic_only"] = True
                unbound_terms.append(index)
        work_rows.append(row)
    external_rows = []
    for evaluation in evaluations:
        item = evaluation["item"]
        if item["term_index"] is not None:
            continue
        row = {"stage": item["stage"], "service_id": item["service_id"],
               "resource": item["resource"], "bound": bool(evaluation["bound"]),
               "problems": list(evaluation["problems"])}
        if evaluation["undelivered"]:
            row["undelivered"] = True
        external_rows.append(row)
    rate_rows = []
    for evaluation in evaluations:
        if evaluation["rate_row"] is None:
            continue
        instance = evaluation["instance"]
        evaluation["rate_row"]["charged_once"] = len(
            instance_services.get(instance, set())) <= 1
        rate_rows.append(evaluation["rate_row"])
    uncovered_actual_work, real_zero_work, zero_work_with_service = [], [], []
    for key, instance in sorted(actual_instances.items()):
        label = str(key[0]) + ":" + str(instance["resource"])
        bound_services = instance_services.get(key, set())
        if instance.get("ambiguous"):
            conflicts.append("actual_work_instance_ambiguous:" + label)
            continue
        if instance["amount"] > 0 and not bound_services:
            uncovered_actual_work.append({
                "stage": key[0], "resource": instance["resource"],
                "physical_resource": instance["physical_resource"],
                "amount": instance["amount"], "unit": instance["unit"],
                "repeats": instance["repeats"],
                "blocker": "actual_work_uncovered:" + label})
        if instance["amount"] == 0:
            if bound_services:
                conflicts.append("service_bound_for_real_zero_work:" + label)
                zero_work_with_service.append(label)
            else:
                real_zero_work.append({"stage": key[0],
                                       "resource": instance["resource"],
                                       "unit": instance["unit"]})
        if len(bound_services) > 1:
            conflicts.append("actual_work_bound_to_multiple_services:" + label)
    plan_for_fixed = dict(plan, decomposition=decomposition)
    fixed_rows, fixed_occurrences, fixed_conflicts, fixed_notes = _fixed_rows(
        plan_for_fixed, services, bound_stages, legacy_fixed,
        invocation=invocation)
    conflicts.extend(fixed_conflicts)
    notes.extend(fixed_notes)
    for row in fixed_rows:
        evidence = service_evidence.get(row["service_id"], {})
        if not evidence.get("identification_ready", True):
            identification_pending.append(
                "fixed:" + row["stage"] + ":" + row["service_id"])
    owned_fixed = {(occurrence["actual_stage"], occurrence["service_id"])
                   for occurrence in fixed_occurrences
                   if occurrence["kind"] == "service_fixed"}
    for item in plan["work"]:
        record = services.get(item["service_id"])
        if record and record["fixed_us"] > 0 and item["stage"]:
            occurrence_stage = _fixed_occurrence_stage(plan, item["stage"])
            if (occurrence_stage, item["service_id"]) not in owned_fixed:
                conflicts.append("stage_fixed_component_not_owned:"
                                 + item["stage"] + ":" + item["service_id"])
    complete = (not conflicts and not unresolved and not identification_pending
                and not uncovered_actual_work
                and not zero_work_with_service and not undelivered_work
                and all(evaluation["bound"] for evaluation in evaluations)
                and (legacy_fixed <= 0 or bool(fixed_rows)))
    diagnostic_combination = ("max_by_resource"
                              if decomposition.get("response_class") == "resource_phase_supply"
                              else "sum")
    coverage = {
        "combination": "actual_program_dependencies",
        "diagnostic_combination": diagnostic_combination,
        "program": plan.get("program_view"),
        "work_terms": work_rows, "external_work": external_rows,
        "actual_work_instances": list(actual_instances.values()),
        "uncovered_actual_work": uncovered_actual_work,
        "undelivered_work": undelivered_work,
        "real_zero_work": real_zero_work,
        "zero_work_with_service": zero_work_with_service,
        "fixed": fixed_rows, "fixed_occurrences": fixed_occurrences,
        "invocation": (None if invocation is None else {
            **invocation,
            "fixed_occurrence_count": len(fixed_occurrences),
            "service_fixed_covers_startup": any(
                occurrence["kind"] == "service_fixed"
                and occurrence["covers_invocation_startup"]
                for occurrence in fixed_occurrences)}),
        "complete": complete,
        "fixed_total_us": sum(occurrence["fixed_us"]
                              for occurrence in fixed_occurrences),
        "legacy_fixed_us": legacy_fixed,
    }
    stage_missing = any(evaluation["stage_missing"] for evaluation in evaluations)
    if conflicts:
        status = "conflict"
    elif unresolved:
        status = "awaiting_conditions"
    elif identification_pending:
        status = "observation_replay"
    elif stage_missing:
        status = "awaiting_program_binding"
    elif complete and scope == "slice":
        status = "slice_ready"
    elif complete:
        status = "ready"
    elif (any(evaluation["instance"] is not None for evaluation in evaluations)
          or external_rows or fixed_rows):
        status = "partial"
    else:
        status = "no_binding"
    blockers = sorted(set(conflicts))
    blockers.extend(sorted(set(identification_pending)))
    blockers.extend(sorted({"condition_context_missing:" + str(item.get("path"))
                            for item in unresolved}))
    blockers.extend("legacy_work_term_unbound:" + str(index)
                    for index in missing_terms)
    blockers.extend(item["blocker"] for item in uncovered_actual_work)
    blockers.extend("actual_work_undelivered:" + str(item["stage"]) + ":"
                    + str(item["resource"]) for item in undelivered_work)
    if status == "awaiting_program_binding":
        blockers.append("migration_stage_binding_required")
    if legacy_fixed > 0 and not fixed_rows:
        blockers.append("legacy_fixed_component_unbound")
    if status == "no_binding":
        blockers.append("migration_binding_plan_requires_actual_work")
    return {
        "status": status, "blockers": sorted(set(blockers)),
        "conflicts": sorted(set(conflicts)),
        "unresolved_conditions": unresolved,
        "coverage": coverage, "service_evidence": service_evidence,
        "conditions": {"rule": rule_check, "slice": slice_check,
                       "services": {sid: item["conditions"]
                                    for sid, item in service_evidence.items()}},
        "diagnostics": {
            "old_provenance_negative": bool(old_provenance_negative),
            "old_training_tokens": sorted(old_training_tokens),
            "actual_native_boundary": native,
            "native_boundary_sources": native_values,
            "actual_resources": dict(
                _attr(context, "resource_names", {}) or {}),
            "actual_invocation": invocation,
            "rate_comparison": rate_rows,
            "fixed_comparison": {
                "legacy_fixed_us": legacy_fixed,
                "service_fixed_total_us": sum(
                    occurrence["fixed_us"] for occurrence in fixed_occurrences),
                "difference_is_diagnostic": True},
            "fixed_occurrence_count": len(fixed_occurrences),
            "unbound_legacy_terms": unbound_terms,
            "undelivered_work_instances": undelivered_work,
            "notes": sorted(set(notes + [
                "old_term_indexes_and_formula_are_diagnostic",
                "old_rate_and_fixed_differences_are_diagnostic",
                "actual_work_owner_and_fixed_boundary_are_shared_projection",
                "validation_or_diagnostic_references_do_not_veto_new_training"])),
        },
        "scope": {"kind": scope, "slice_conditions": dict(slice_conditions)},
        "replacement_scope": ("slice" if scope == "slice" else
                              ("independent_resource" if kind == "independent_service"
                               else "joint_phase")),
    }
