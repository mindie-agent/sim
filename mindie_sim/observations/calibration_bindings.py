"""Applicability and operator bindings for calibrated resource services.

[Role]
  Match service conditions and attach the selected service to an operator.
[Invariants]
  Bindings preserve the declared work unit, phase and applicability domain.
  Local communication services use their shared applicability contract.
[Timeouts & Locks]
  In-memory matching and binding. No I/O or locks.
[Side-Effect Policy]
  Returns match reports, binding dictionaries and replaced operators.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import math

from .resource_services import (
    RESOURCE_SERVICE_BINDING_SCHEMA,
    NormalizedResourceService,
    identity_digest,
    normalize_resource_service,
)


_CONDITION_MISSING = object()


def _condition_value(context, path):
    value = context
    for part in str(path).split("."):
        if isinstance(value, Mapping):
            if part not in value:
                return _CONDITION_MISSING
            value = value[part]
        elif (isinstance(value, (list, tuple)) and part.isdigit()
              and int(part) < len(value)):
            value = value[int(part)]
        else:
            return _CONDITION_MISSING
    return value


def _condition_matches(actual, expected, path=""):
    if str(path).endswith("image_digest"):
        return identity_digest(actual) == identity_digest(expected)
    if str(path) == "dtype":
        return str(actual).lower() == str(expected).lower()
    if isinstance(expected, Mapping) and set(expected) <= {"min", "max"}:
        try:
            number = float(actual)
            if "min" in expected and number < float(expected["min"]):
                return False
            if "max" in expected and number > float(expected["max"]):
                return False
            return True
        except (TypeError, ValueError):
            return False
    if (isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and isinstance(expected, (int, float))
            and not isinstance(expected, bool)):
        return math.isclose(float(actual), float(expected),
                            rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return (actual.keys() == expected.keys()
                and all(_condition_matches(actual[key], expected[key], path)
                        for key in expected))
    if (isinstance(actual, (list, tuple))
            and isinstance(expected, (list, tuple))):
        return (len(actual) == len(expected)
                and all(_condition_matches(a, b, path)
                        for a, b in zip(actual, expected)))
    return actual == expected


def service_conditions_match(service, context):
    """Check only the conditions the service actually declares.

    Extra keys in a broader physical contract (unrelated rates, modes, owners or
    environment fields) are not identity.  A declared rate/mode/unit/environment
    condition must still match, so a real change in the depended-on supply is
    rejected.  ``conditions`` may use dotted paths such as
    ``resource_units.mte2_aiv`` or ``rates.hbm``.
    """
    if not isinstance(context, Mapping):
        raise ValueError("service_condition_context_requires_mapping")
    conditions = getattr(service, "conditions", None)
    if conditions is None and isinstance(service, Mapping):
        conditions = service.get("conditions")
    if conditions is None:
        raise ValueError("service_condition_service_requires_conditions")
    if not isinstance(conditions, Mapping):
        raise ValueError("service_condition_conditions_requires_mapping")
    checks = []
    for path, expected in conditions.items():
        actual = _condition_value(context, path)
        present = actual is not _CONDITION_MISSING
        matched = present and _condition_matches(actual, expected, path)
        checks.append({
            "path": str(path),
            "expected": deepcopy(expected),
            "actual": deepcopy(actual) if present else None,
            "present": present,
            "matched": matched,
            "optional": False,
        })
    optional = getattr(service, "optional_conditions", None)
    if optional is None and isinstance(service, Mapping):
        optional = service.get("optional_conditions")
    if optional:
        if not isinstance(optional, Mapping):
            raise ValueError("service_optional_conditions_requires_mapping")
        for path, expected in optional.items():
            actual = _condition_value(context, path)
            present = actual is not _CONDITION_MISSING
            if not present or actual is None:
                checks.append({
                    "path": str(path), "expected": deepcopy(expected),
                    "actual": None, "present": present, "matched": True,
                    "optional": True,
                })
                continue
            matched = _condition_matches(actual, expected, path)
            checks.append({
                "path": str(path), "expected": deepcopy(expected),
                "actual": deepcopy(actual), "present": True,
                "matched": matched, "optional": True,
            })
    return {
        "applies": all(check["matched"] for check in checks),
        "checks": checks,
        "policy": ("only_declared_condition_paths; unrelated physical "
                   "contract keys are ignored"),
    }


def _condition_failure_reason(service, result):
    failed = [check for check in result["checks"] if not check["matched"]]
    if not failed:
        return "conditions_matched"
    parts = []
    for check in failed:
        actual = check["actual"] if check["present"] else "<missing>"
        parts.append(f"{check['path']}={actual!r}!={check['expected']!r}")
    return service.service_id + ":" + ";".join(parts)


def select_resource_service_candidate(services, context, *, service_id=None):
    """Narrow by optional ID, then apply the one condition matcher to all.

    Explicit ID, single candidate and automatic selection all pass through the
    same condition check.  A missing ID, missing condition, value mismatch or
    ambiguity raises a specific error; there is no nominal fallback.
    """
    if not isinstance(context, Mapping):
        raise ValueError("resource_service_context_requires_mapping")
    candidates = list(services or ())
    if service_id is not None:
        candidates = [item for item in candidates
                      if item.service_id == service_id]
        if not candidates:
            raise ValueError("resource_service_id_not_found:" + str(service_id))
    checked = []
    selected = []
    for service in candidates:
        result = service_conditions_match(service, context)
        checked.append((service, result))
        if result["applies"]:
            selected.append(service)
    if len(selected) == 1:
        return selected[0]
    if not selected:
        reasons = ";".join(
            _condition_failure_reason(service, result)
            for service, result in checked)
        raise ValueError("resource_service_conditions_have_no_match:" + reasons)
    raise ValueError("resource_service_conditions_are_ambiguous:"
                     + ",".join(item.service_id for item in selected))


def resource_service_binding(service, *, resource_role=None, phase=None,
                             work_unit=None, conditions=None):
    """Generate the exact T04/T14 binding for one normalized service.

    The binding names the service, the actual physical resource role, the
    declared work unit and the phase.  If the service declares applicability
    conditions, a real placement context must be supplied and match.  The
    produced mapping is the one consumed by ``Execution._context`` /
    ``_apply_resource_service_program``; it is not a whole-call duration.
    """
    if not isinstance(service, NormalizedResourceService):
        service = normalize_resource_service(service)
    if service.scope != "independent_resource" or service.kind != "resource_rate":
        raise ValueError("resource_service_binding_requires_independent_rate")
    role = str(resource_role or service.resource or "")
    if role not in service.resources:
        raise ValueError("resource_service_binding_role_not_declared:" + role)
    unit = str(work_unit or service.work_unit)
    if unit != service.work_unit:
        raise ValueError(
            "resource_service_binding_work_unit_mismatch:"
            + service.service_id + ":" + unit + ":" + service.work_unit)
    selected_phase = str(phase or service.phase)
    if service.phase and selected_phase != service.phase:
        raise ValueError(
            "resource_service_binding_phase_mismatch:"
            + service.service_id + ":" + selected_phase + ":" + service.phase)
    if service.conditions:
        if conditions is None:
            raise ValueError(
                "resource_service_binding_requires_conditions_context:"
                + service.service_id)
        if not service_conditions_match(service, conditions)["applies"]:
            raise ValueError(
                "resource_service_binding_conditions_do_not_match:"
                + service.service_id)
    source = deepcopy(service.provenance.get("source")
                      or service.provenance.get("sources"))
    return {
        "schema": RESOURCE_SERVICE_BINDING_SCHEMA,
        "service_id": service.service_id,
        "model_id": service.model_id,
        "resource_role": role,
        "work_unit": unit,
        "phase": selected_phase,
        "rate_scope": service.rate_scope,
        "active_units": service.active_units,
        "fixed_us": service.fixed_us,
        "effective_rate": service.effective_rate,
        "formula": service.formula,
        "conditions": deepcopy(service.conditions),
        "optional_conditions": deepcopy(service.optional_conditions),
        "conditions_checked": bool(service.conditions or service.optional_conditions),
        "boundary": service.boundary,
        "resource": service.resource,
        "resources": list(service.resources),
        "startup_us": service.startup_us,
        "response_us": service.response_us,
        "work_scope": service.rate_scope,
        "aggregate_rate": service.aggregate_rate,
        "per_owner_rate": service.per_owner_rate,
        "identification": deepcopy(service.identification),
        "rate_representation": deepcopy(service.rate_representation),
        "normalization": {**deepcopy(service.normalization),
                          "applied_once": True},
        "startup_once": True,
        "fixed_once": True,
        "formula": service.formula,
        "source": source,
        "production_prediction": True,
        "legacy": False,
    }


def attach_operator_resource_service_binding(op, binding):
    """Attach a previously selected binding to an OpSpec without re-selecting."""
    if not isinstance(binding, Mapping):
        raise ValueError("resource_service_binding_requires_mapping")
    metadata = [(key, value) for key, value in tuple(op.metadata)
                if key != "resource_service"]
    previous = dict(op.metadata).get("resource_service")
    if previous is not None and previous != dict(binding):
        raise ValueError("resource_service_binding_already_present")
    metadata.append(("resource_service", dict(binding)))
    return replace(op, metadata=tuple(metadata))


def bind_operator_resource_service(op, service, *, resource_role=None,
                                   phase=None, work_unit=None, conditions=None):
    """Return a new OpSpec carrying the exact T04/T14 resource-service binding.

    The actual selected producer calls this once after it has chosen the
    program/phase/context; T14 can pass the returned op to the common
    Execution.  The binding is never inferred later from a model label or a
    whole-call observation.
    """
    binding = resource_service_binding(
        service, resource_role=resource_role, phase=phase,
        work_unit=work_unit, conditions=conditions)
    binding.update({
        "op_id": getattr(op, "op_id", None),
        "op_family": getattr(op, "family", None),
        "op_stage": getattr(op, "stage", None),
        "op_phase": getattr(op, "phase", None),
    })
    return attach_operator_resource_service_binding(op, binding)


def bind_a5_local_service(record, *, domain, op=None, resource_role=None,
                          phase=None, include_lineage=False):
    """Select an accepted A5COMM local service and expose one T04 binding.

    The A5COMM applicability/presence/lineage functions remain the single
    authority.  This wrapper only turns the already-normalized service and its
    measured domain into the fields T04/T09/PW-FAMILIES/T15 consume.
    """
    from mindie_sim.observations.communication import (
        a5_local_service_applicability_key, a5_local_service_lineage_key,
        require_a5_local_service_applicability,
        require_a5_local_service_domain,
    )
    if not isinstance(record, Mapping):
        raise ValueError("a5_local_service_record_requires_mapping")
    if include_lineage:
        require_a5_local_service_domain(record, domain)
    else:
        require_a5_local_service_applicability(record, domain)
    service = normalize_resource_service(record["service"])
    binding = resource_service_binding(
        service, resource_role=resource_role or service.resource,
        phase=phase or service.phase, work_unit=service.work_unit,
        conditions=domain)
    provenance = dict(service.provenance)
    binding.update({
        "service_schema": record.get("schema"),
        "service_semantics": record.get("service_semantics"),
        "domain_key": deepcopy(record.get("domain_key") or {}),
        "applicability_key": a5_local_service_applicability_key(domain),
        "lineage_key": a5_local_service_lineage_key(domain),
        "evidence_gaps": list(record.get("evidence_gaps") or ()),
        "heldout_max_abs_relative_error": record.get(
            "heldout_max_abs_relative_error"),
        "link_transit_identified": bool(
            provenance.get("link_transit_identified", False)),
        "hop_scaling": provenance.get("hop_scaling", "not_derived"),
        "shared_domain": deepcopy(provenance.get("shared_domain")),
        "transport_window": deepcopy(provenance.get("transport_window")),
        "direction": service.conditions.get("direction"),
        "actual_phase": getattr(op, "phase", None) if op is not None else None,
        "physical_resource_role": (resource_role or service.resource),
    })
    if record.get("source") is not None:
        binding["source"] = deepcopy(record.get("source"))
    if op is None:
        return binding
    binding.update({
        "op_id": getattr(op, "op_id", None),
        "op_family": getattr(op, "family", None),
        "op_stage": getattr(op, "stage", None),
        "op_phase": getattr(op, "phase", None),
    })
    return attach_operator_resource_service_binding(op, binding)


def bind_a5_local_service_operator(record, op, *, domain, **kwargs):
    return bind_a5_local_service(record, domain=domain, op=op, **kwargs)
