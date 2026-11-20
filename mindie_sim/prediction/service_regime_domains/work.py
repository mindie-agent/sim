"""Physical work resolution for service migration.

[Role] Resolve declared work references against actual stage fragments.
[Invariants] Work is resolved by ownership, resource and native unit before rate scaling.
[Timeouts & Locks] Bounded by the program stages and work references. No waits or locks.
[Side-Effect Policy] Returns work and diagnostic records from caller-supplied data.
"""
from __future__ import annotations

from collections.abc import Mapping


def _attr(value, name, default=None):
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _stage(plan, name):
    for stage in plan.get("stages", ()):
        if getattr(stage, "name", None) == name:
            return stage
    return None


def _fragment_for(plan, stage):
    if stage is None:
        return None
    indexed = (plan.get("fragments") or {}).get(getattr(stage, "name", ""))
    if indexed is not None:
        return indexed
    return (getattr(stage, "fragment", None)
            or (plan.get("occurrences") or {}).get(getattr(stage, "name", "")))


def _resource_identity(plan, resource):
    """Resolve one logical role or physical key through the actual context.

    A service or binding may name the logical role (``hbm``) or the physical
    resource key produced by ``RecipeContext.placement`` (``hbm@r0:s0``).  Both
    spellings must resolve to the same physical occurrence; otherwise the same
    work, owner partition or fixed component would be judged once per spelling.
    """
    if resource is None:
        return None, None, frozenset()
    text = str(resource)
    context = plan.get("context")
    names = dict(_attr(context, "resource_names", {}) or {})
    modes = dict(_attr(context, "service_modes", {}) or {})
    reverse = {str(name): str(role) for role, name in names.items()}
    if text in modes:
        role = str(modes[text][0])
    elif text in reverse:
        role = reverse[text]
    else:
        role = text
    physical = str(names.get(role, text))
    return role, physical, frozenset({text, role, physical})


def _stage_resource_aliases(plan, stage):
    """Logical and physical names actually issued by one selected stage."""
    resources = set()
    fragment = _fragment_for(plan, stage)
    if fragment is not None and hasattr(fragment, "work"):
        for item in fragment.work.issued():
            resources.update(_resource_identity(plan, item.resource)[2])
    for key in (getattr(stage, "work", {}) or {}):
        resources.update(_resource_identity(plan, key)[2])
    return resources


def _service_stage_overlap(plan, stage, resources):
    aliases = _stage_resource_aliases(plan, stage)
    return any(_resource_identity(plan, resource)[2] & aliases
               for resource in resources)


def _resolve_actual_work(plan):
    """One dereference of the selected program's stage/fragment/context work.

    The returned instances are the only authority for work amount, native unit,
    repeat count, physical resource and actual owner partition.  Term bindings,
    external bindings, coverage and fixed checks all consume this same map.
    """
    instances, fragments = {}, {}
    for stage in plan.get("stages", ()):
        name = getattr(stage, "name", "")
        fragment = _fragment_for(plan, stage)
        fragments[name] = fragment
        if fragment is None or not hasattr(fragment, "work"):
            continue
        repeats = int(getattr(fragment, "repeats", 1))
        for item in fragment.work.issued():
            role, physical, _aliases = _resource_identity(plan, item.resource)
            key = (name, physical)
            entry = {"stage": name, "resource": item.resource,
                     "logical_resource": role, "physical_resource": physical,
                     "unit": item.unit, "repeats": repeats,
                     "amount": float(item.amount) * repeats,
                     "owner_units": _owner_units(plan, stage, physical),
                     "fragment": getattr(fragment, "name", "")}
            if key in instances:
                instances[key]["ambiguous"] = True
            else:
                instances[key] = entry
    return {"instances": instances, "fragments": fragments}


def _stage_work(plan, stage, resource, multiplicity, work_unit=None,
                resolved=None):
    """Read one work reference from the already resolved actual instance."""
    role, physical, aliases = _resource_identity(plan, resource)
    result = {"amount": None, "unit": None,
              "basis": "actual_work_instance_missing",
              "repeats": None, "owner_units": None, "instance": None,
              "delivered": False, "logical_resource": role,
              "resource": physical, "fragment": None}
    if stage is None:
        result["basis"] = "stage_missing"
        return result
    resolved = resolved if resolved is not None else _resolve_actual_work(plan)
    stage_name = getattr(stage, "name", "")
    fragment = resolved["fragments"].get(stage_name)
    result["fragment"] = getattr(fragment, "name", None)
    matched = []
    for key, entry in resolved["instances"].items():
        if key[0] != stage_name:
            continue
        item_role, item_physical, _item_aliases = _resource_identity(
            plan, entry["resource"])
        if (key[1] == physical or entry["resource"] in aliases
                or item_role == role):
            matched.append(entry)
    if len(matched) != 1:
        result["basis"] = ("actual_work_instance_ambiguous" if matched
                           else "actual_work_instance_missing")
        result["repeats"] = int(getattr(fragment, "repeats", 1))
        return result
    entry = matched[0]
    if entry.get("ambiguous"):
        result["basis"] = "actual_work_instance_ambiguous"
        result["repeats"] = entry["repeats"]
        return result
    if not entry["unit"]:
        result["basis"] = "actual_native_unit_missing"
        result["repeats"] = entry["repeats"]
        result["owner_units"] = entry["owner_units"]
        return result
    basis = ("program_fragment_real_zero" if entry["amount"] == 0
             else "program_fragment_work")
    if work_unit is not None and work_unit != entry["unit"]:
        basis = "declared_unit_conflict"
    result.update({
        "amount": entry["amount"], "unit": entry["unit"], "basis": basis,
        "repeats": entry["repeats"],
        "multiplicity_matches": int(multiplicity) == entry["repeats"],
        "owner_units": entry["owner_units"],
        "instance": (stage_name, entry["physical_resource"]),
        "logical_resource": entry["logical_resource"],
        "resource": entry["physical_resource"],
        "fragment": entry["fragment"], "delivered": True,
    })
    return result


def _owner_units(plan, stage, resource):
    """Actual owner partition keyed by either physical key or logical role."""
    role, physical, aliases = _resource_identity(plan, resource)
    placement = dict(getattr(stage, "placement", {}) or {})
    for key in (physical, role):
        if key in placement and placement[key]:
            return len(tuple(placement[key]))
    for key, units in placement.items():
        if key in aliases and units:
            return len(tuple(units))
    context = plan.get("context")
    units = _attr(context, "resource_units", {}) or {}
    for key in (role, physical):
        if key in units and int(units[key] or 0):
            return int(units[key])
    return None


def _undelivered_stage_work(plan, resolved=None):
    """Enumerate positive work whose authoritative fragment/unit is absent."""
    resolved = resolved if resolved is not None else _resolve_actual_work(plan)
    undelivered = []
    for stage in plan.get("stages", ()):
        fragment = resolved["fragments"].get(getattr(stage, "name", ""))
        issued = set()
        if fragment is not None and hasattr(fragment, "work"):
            for item in fragment.work.issued():
                issued |= _resource_identity(plan, item.resource)[2]
        for resource, amount in (getattr(stage, "work", {}) or {}).items():
            try:
                positive_work = float(amount) > 0
            except (TypeError, ValueError):
                positive_work = False
            _role, physical, aliases = _resource_identity(plan, resource)
            if positive_work and not (issued & aliases):
                undelivered.append({
                    "stage": getattr(stage, "name", ""),
                    "resource": resource,
                    "physical_resource": physical,
                    "declared_engine_work": float(amount),
                    "declared_unit": (getattr(stage, "work_units", {}) or {}).get(resource),
                    "reason": "authoritative_fragment_work_instance_or_native_unit_missing",
                })
    return undelivered


def _evaluate_work_reference(plan, item, stage_map, services, service_evidence,
                             context, native, resolved):
    """Resolve one work row against the same actual instance used by coverage.

    Delivery is decided before any rate arithmetic: a missing fragment, a
    missing/ambiguous work instance or a missing native unit returns an
    undelivered result with no amount.  Every rate scope consumes this result.
    """
    result = {"item": item, "problems": [], "undelivered": False,
              "stage_missing": False, "projected": None, "instance": None,
              "rate_row": None, "bound": False, "phase": None,
              "record": None, "service_id": item["service_id"]}
    if item["stage"] is None:
        result["stage_missing"] = True
        return result
    stage = stage_map.get(item["stage"])
    if stage is None:
        result["problems"].append("work_stage_not_in_program:" + str(item["stage"]))
        return result
    record = services.get(item["service_id"])
    if record is None:
        result["problems"].append("unknown_service:" + item["service_id"])
        return result
    result["record"] = record
    evidence = service_evidence.get(item["service_id"], {})
    if not evidence.get("valid", False):
        result["problems"].append("binding_service_invalid")
    elif not evidence.get("identification_ready", False):
        result["problems"].append("binding_service_identification_pending")
    projected = _stage_work(plan, stage, item["resource"], item["multiplicity"],
                            item["work_unit"], resolved=resolved)
    result["projected"] = projected
    if not projected["delivered"]:
        result["undelivered"] = True
        result["problems"].append(projected["basis"])
        return result
    result["instance"] = projected["instance"]
    if projected["basis"] == "declared_unit_conflict":
        result["problems"].append("binding_work_unit_does_not_match_actual_stage")
    binding_role, _binding_physical, binding_aliases = _resource_identity(
        plan, item["resource"])
    _actual_role, _actual_physical, actual_aliases = _resource_identity(
        plan, projected["resource"])
    service_aliases = set()
    for resource in record["resources"]:
        service_aliases.update(_resource_identity(plan, resource)[2])
    if service_aliases:
        if not (service_aliases & actual_aliases):
            result["problems"].append("actual_work_resource_not_covered_by_service")
        if not (service_aliases & binding_aliases):
            result["problems"].append("actual_stage_resource_not_covered_by_service")
    if projected["basis"] == "program_fragment_real_zero":
        result["problems"].append("service_bound_for_real_zero_work")
    if projected.get("multiplicity_matches") is False:
        result["problems"].append("multiplicity_does_not_match_actual_occurrence")
    if projected["unit"] != record["work_unit"]:
        result["problems"].append("stage_work_unit_does_not_match_service")
    if item["phase"] != record["phase"]:
        result["problems"].append("binding_phase_does_not_match_service")
    if item["native_boundary"]:
        if native and item["native_boundary"] != native:
            result["problems"].append(
                "binding_native_boundary_does_not_match_actual_program")
        elif record["native_boundary"] and item["native_boundary"] != record["native_boundary"]:
            result["problems"].append(
                "binding_native_boundary_does_not_match_service")
    work_scope = item["work_scope"] or record["rate_scope"]
    rate = (record.get("aggregate_rate") if work_scope == "aggregate"
            else record.get("per_owner_rate"))
    if rate is None:
        result["problems"].append("service_scope_rate_missing")
    owner_units = projected.get("owner_units")
    if work_scope == "per_owner":
        if owner_units is None:
            result["problems"].append("per_owner_work_requires_actual_owner_units")
        elif int(record.get("active_units") or 0) != int(owner_units):
            result["problems"].append(
                "per_owner_active_units_do_not_match_actual_owner_units")
    work_amount = projected["amount"]
    if work_scope == "per_owner":
        work_amount = (work_amount / owner_units) if owner_units else None
    native_work_map = _attr(context, "native_work", {}) or {}
    native_work = native_work_map.get(item["resource"])
    if native_work is None:
        native_work = native_work_map.get(binding_role)
    result["rate_row"] = {
        "term_index": item["term_index"], "service_id": item["service_id"],
        "stage": item["stage"], "work_scope": work_scope,
        "actual_work": work_amount, "aggregate_work": projected["amount"],
        "actual_unit": projected["unit"], "owner_units": owner_units,
        "rate_used": rate, "amount_basis": projected["basis"],
        "multiplicity": item["multiplicity"],
        "actual_repeats": projected.get("repeats"),
        "native_work": native_work,
        "native_work_difference": (
            None if native_work is None
            else projected["amount"] - native_work),
        "instance": list(projected["instance"]), "charged_once": True,
        "difference_is_diagnostic": True}
    result["phase"] = item["phase"]
    result["bound"] = not result["problems"]
    return result
