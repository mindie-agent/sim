"""Service handoff normalization and migration reporting helpers.

[Role] Normalize diagnostic handoffs, links and provenance evidence.
[Invariants] Service evidence retains its declared scope and authority.
[Timeouts & Locks] Bounded by supplied records. No I/O, waits or locks.
[Side-Effect Policy] Returns diagnostic records without binding a production service.
"""
from collections.abc import Mapping
import json

from mindie_sim.schema.common import ModelingError
from .response_functions import nonnegative as _nonnegative


MISSING = object()
MIGRATION_SCHEMA = "mindie-sim.regime-migration-readiness/1"
HANDOFF_SCHEMA = "mindie-sim.regime-resource-phase-handoff/1"
MIGRATION_STATUSES = frozenset({
    "resource_phase_ready", "slice_ready", "awaiting_binding_plan",
    "awaiting_resource_service", "awaiting_program_binding", "awaiting_conditions",
    "partial", "conflict", "link_rejected", "observation_replay"})
_PUBLIC_STATUS = {
    "ready": "resource_phase_ready", "slice_ready": "slice_ready", "partial": "partial",
    "awaiting_conditions": "awaiting_conditions", "conflict": "conflict",
    "awaiting_program_binding": "awaiting_program_binding",
    "no_binding": "awaiting_binding_plan", "not_eligible": "observation_replay",
    "observation_replay": "observation_replay"}
_IDENTIFIABILITY_KEYS = frozenset({
    "resource_identifiability", "limitations", "boundary", "parameter_boundary",
    "scaling_assumption", "response_semantics", "combination",
    "measurement_identity", "assignment", "data_roles"})
_NEGATIVE_IDENTIFIABILITY_MARKERS = (
    "not independent", "no independent", "not independently", "does not identify",
    "not identify", "no guessed", "not assigned to a guessed", "not a hardware",
    "not a bandwidth", "effective call response only", "engineered", "inclusive",
    "whole native", "whole-call", "whole call", "total duration", "not measured")


def _provenance_identifiability(provenance):
    """Preserve old evidence; its negative result does not gate a new service."""
    evidence = []
    for key in sorted(_IDENTIFIABILITY_KEYS):
        value = provenance.get(key) if isinstance(provenance, Mapping) else None
        if value is None:
            continue
        text = value if isinstance(value, str) else json.dumps(
            value, sort_keys=True, ensure_ascii=False, default=str)
        lowered = text.lower()
        marker = next((item for item in _NEGATIVE_IDENTIFIABILITY_MARKERS
                       if item in lowered), None)
        if marker is not None:
            evidence.append({"key": key, "marker": marker, "text": text})
    return {"negative": bool(evidence), "evidence": evidence}


def _split_links(links):
    """Separate explicit binding plans from old candidate-only lists."""
    if links is None:
        return {}, {}
    if not isinstance(links, Mapping):
        raise ModelingError("migration_links_require_mapping")
    plans, legacy = {}, {}
    for rule_id, value in links.items():
        if not isinstance(rule_id, str) or not rule_id:
            raise ModelingError("migration_link_requires_rule_id")
        if isinstance(value, Mapping):
            plans[rule_id] = value
            continue
        ids = (value,) if isinstance(value, str) else tuple(value) if isinstance(
            value, (list, tuple)) else None
        if ids is None or any(not isinstance(item, str) or not item for item in ids):
            raise ModelingError("migration_link_requires_service_id_sequence:" + rule_id)
        if len(set(ids)) != len(ids):
            raise ModelingError("migration_link_repeats_service:" + rule_id)
        legacy[rule_id] = ids
    return plans, legacy


def _service_field(service, name, default=MISSING):
    if isinstance(service, Mapping):
        return service.get(name, default)
    return getattr(service, name, default)


def normalize_service_handoff(service):
    """Read one already-normalized T06 handoff without re-normalizing it."""
    service_id = _service_field(service, "service_id")
    if service_id is MISSING:
        service_id = _service_field(service, "id")
    if not isinstance(service_id, str) or not service_id:
        raise ModelingError("migration_service_handoff_requires_id")
    resources = _service_field(service, "resources", ())
    if isinstance(resources, str):
        resources = (resources,)
    if not isinstance(resources, (list, tuple)):
        raise ModelingError("migration_service_handoff_requires_resources:" + service_id)
    conditions = _service_field(service, "conditions", {}) or {}
    identification = _service_field(service, "identification", {}) or {}
    provenance = _service_field(service, "provenance", {}) or {}
    if not all(isinstance(item, Mapping) for item in (conditions, identification, provenance)):
        raise ModelingError("migration_service_handoff_requires_mappings:" + service_id)
    native_boundary = (_service_field(service, "native_boundary", None)
                       or _service_field(service, "native_kernel", None)
                       or conditions.get("native_boundary")
                       or conditions.get("native_kernel") or "")
    return {
        "service_id": service_id,
        "scope": str(_service_field(service, "scope", "")),
        "resources": tuple(str(item) for item in resources),
        "phase": str(_service_field(service, "phase", "")),
        "work_unit": str(_service_field(service, "work_unit", "")),
        "boundary": str(_service_field(service, "boundary", "")),
        "fixed_us": _nonnegative(_service_field(service, "fixed_us", 0)),
        "effective_rate": _service_field(service, "effective_rate"),
        "aggregate_rate": _service_field(service, "aggregate_rate"),
        "per_owner_rate": _service_field(service, "per_owner_rate"),
        "rate_scope": str(_service_field(service, "rate_scope", "")),
        "active_units": _service_field(service, "active_units", 1),
        "conditions": dict(conditions),
        "identification": dict(identification),
        "provenance": dict(provenance),
        "native_boundary": str(native_boundary).strip(),
    }


def _target_authority(kind):
    return ("resource_rate_service" if kind == "independent_service"
            else "joint_phase_service")


def _no_plan_record(raw, decomposition, mode, old_evidence):
    rule_id = raw["id"]
    response_class = decomposition["response_class"]
    target, replacement_scope = "diagnostic_observation_replay", None
    blockers = list(decomposition.get("blockers", ()))
    if response_class == "resource_phase_supply":
        status, target = "resource_phase_ready", "resource_phase_supply"
        replacement_scope = "declared_physical_rates"
        blockers += ["t04_executor_must_bind_declared_supply_as_resource_work"]
        requires = ["T04"]
    elif old_evidence["negative"] and decomposition["decomposable"]:
        status = "observation_replay"
        blockers += ["explicit_observation_replay_only",
                     "provenance_declares_no_independent_resource_identifiability",
                     "t06_normalized_resource_service_required_for_switch"]
        requires = ["T06", "T04"]
    elif mode == "joint_native":
        status = "observation_replay"
        blockers += ["explicit_observation_replay_only",
                     "joint_native_total_duration_must_not_become_a_busy_ns_rate",
                     "t06_normalized_joint_or_resource_service_required_for_switch"]
        requires = ["T06", "T04"]
    elif mode == "mac_service" or response_class == "measured_issue_class":
        status = "observation_replay"
        blockers += ["explicit_observation_replay_only",
                     "measured_issue_class_is_not_a_shape_generalization",
                     "t06_normalized_shape_service_required_for_switch"]
        requires = ["T06", "T04"]
    elif decomposition["decomposable"]:
        status, target = "awaiting_resource_service", "resource_rate_service"
        blockers += ["t06_normalized_resource_service_not_linked"]
        requires = ["T06", "T04"]
    else:
        status = "observation_replay"
        blockers += ["explicit_observation_replay_only",
                     "whole_call_response_requires_explicit_observation_replay"]
        requires = []
    return {
        "rule_id": rule_id, "execution_mode": mode,
        "response_class": response_class,
        "production_authority": {
            "completion_floor": "legacy_completion_floor",
            "joint_native": "legacy_joint_native_busy_ns",
            "mac_service": "legacy_mac_service"}.get(mode, "legacy_" + mode),
        "target_authority": target, "status": status,
        "replacement_scope": replacement_scope,
        "decomposition": decomposition, "linked_services": [],
        "blockers": sorted(set(blockers)),
        "provenance_evidence": list(old_evidence["evidence"]),
        "coverage": None, "conditions": None, "diagnostics": None,
        "service_evidence": {}, "scope": None, "requires": requires}
