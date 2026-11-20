"""Role-ledger validation and resource service training packets.

[Role]
  Partition historical cohorts, validate contracts and fit eligible trials.
[Invariants]
  In-sample cohorts alone supply training observations. Frozen transfers
  and undeclared cohorts keep their declared roles.
[Timeouts & Locks]
  Reads an explicit local ledger path. No locks.
[Side-Effect Policy]
  Reads metadata and builds reports without opening referenced captures.
"""
from __future__ import annotations

from collections.abc import Mapping

from .service_fitting_regression import fit_resource_service


_ROLE_IN_SAMPLE = "in_sample"
_ROLE_FROZEN_TRANSFER = "frozen_training_function_transfer_to_new_capture"
_ROLE_UNDECLARED = "not_declared"
_ROLE_ALIASES = {
    _ROLE_IN_SAMPLE: _ROLE_IN_SAMPLE,
    _ROLE_FROZEN_TRANSFER: "frozen_transfer",
    _ROLE_UNDECLARED: "not_declared",
}
_ROLE_SCOPES = frozenset({"independent_resource", "joint_phase"})
_PHASE_WINDOWS = frozenset({"resource_phase", "resource", "phase"})
_JOINT_ROLE_WINDOWS = frozenset({"joint", "mixed", "overlapping", "joint_phase"})
_FORBIDDEN_ROLE_WINDOWS = frozenset({
    "step", "body", "request", "serving", "e2e", "engine_step", "body_step",
    "whole_step", "whole_body",
})
_ROLE_NEEDED_EVIDENCE = (
    "scope", "resources", "phase", "work", "work_unit",
    "resource_duration_us", "owner_units", "boundary",
)
_ROLE_SERVICE_BOUNDARIES = frozenset({"resource_phase", "device_kernel"})


def load_role_ledger(path):
    """Read a JSONL cohort ledger without opening any referenced capture."""
    import json
    from pathlib import Path
    records = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return tuple(records)


def calibration_role_partitions(records):
    """Partition historical cohorts by their declared role, without relabeling.

    ``in_sample`` alone is eligible for parameter fitting.  Frozen transfer
    cohorts remain independent transfer evidence; ``not_declared`` rows are
    neither training nor a holdout.
    """
    if not isinstance(records, (list, tuple)):
        raise ValueError("calibration_role_records_require_sequence")
    parts = {"in_sample": [], "frozen_transfer": [], "not_declared": []}
    seen = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("calibration_role_record_requires_mapping")
        cohort_id = str(record.get("cohort_id") or "").strip()
        if not cohort_id:
            raise ValueError("calibration_role_record_requires_cohort_id")
        if cohort_id in seen:
            raise ValueError("calibration_role_record_duplicate_cohort_id")
        seen.add(cohort_id)
        role = str(record.get("role") or "").strip()
        if role not in _ROLE_ALIASES:
            raise ValueError("calibration_role_unknown:" + role)
        parts[_ROLE_ALIASES[role]].append(record)
    counts = {key: len(value) for key, value in parts.items()}
    return {
        "in_sample": tuple(parts["in_sample"]),
        "frozen_transfer": tuple(parts["frozen_transfer"]),
        "not_declared": tuple(parts["not_declared"]),
        "counts": counts,
        "total": sum(counts.values()),
        "policy": ("in_sample_only_for_fit; frozen_transfer_independent; "
                   "not_declared_neither_training_nor_holdout"),
    }


def _canonical(value):
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _role_conditions(record, *, conditions, hardware):
    """Merge record-level and call-level conditions, rejecting conflicts."""
    raw = record.get("conditions")
    if raw is None:
        declared = {}
    elif not isinstance(raw, Mapping):
        raise ValueError("resource_service_role_record_conditions_requires_mapping")
    else:
        declared = dict(raw)
    merged = dict(declared)
    if hardware:
        declared_hardware = declared.get("hardware")
        if declared_hardware and str(declared_hardware) != str(hardware):
            raise ValueError("resource_service_role_record_hardware_mismatch")
        merged["hardware"] = str(hardware)
    if conditions is not None:
        if not isinstance(conditions, Mapping):
            raise ValueError("resource_service_role_record_conditions_requires_mapping")
        for path, expected in conditions.items():
            if path in declared and _canonical(declared[path]) != _canonical(expected):
                raise ValueError(
                    "resource_service_role_record_condition_conflict:" + str(path))
            merged[path] = expected
    return declared, merged


def _role_resources(record):
    raw = record.get("resources")
    if raw is None and record.get("resource") is not None:
        raw = [record["resource"]]
    elif isinstance(raw, str):
        raw = [raw]
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("resource_service_role_record_resources_invalid")
    values = tuple(str(item).strip() for item in raw)
    if any(not item for item in values) or len(set(values)) != len(values):
        raise ValueError("resource_service_role_record_resources_invalid")
    return values


def _role_contract(record, *, resource, phase, work_unit, boundary,
                   rate_scope, active_units, conditions, hardware, scope):
    """Extract one source observation contract exactly as declared.

    ``state`` is one of ``candidate``, ``joint``, ``excluded`` or ``gap``.
    The caller parameter is a selector and a source-level declaration; it is
    never silently substituted for a missing per-record contract field.
    """
    cohort_id = str(record["cohort_id"])
    declared = {
        "cohort_id": cohort_id,
        "case_id": record.get("case_id"),
        "source_path": (record.get("source_path") or record.get("source")),
        "locator": (record.get("locator") or record.get("example_locator")
                    or record.get("capture_locator")),
        "scope": record.get("scope"),
        "resources": _role_resources(record),
        "resource": record.get("resource"),
        "phase": record.get("phase"),
        "work_unit": record.get("work_unit"),
        "work": record.get("work"),
        "duration_us": record.get("duration_us"),
        "resource_duration_us": record.get("resource_duration_us"),
        "boundary": record.get("boundary"),
        "rate_scope": record.get("rate_scope"),
        "active_units": (record.get("active_units")
                         if record.get("active_units") is not None else
                         record.get("owner_units")
                         if record.get("owner_units") is not None else
                         record.get("owners")),
        "conditions": record.get("conditions") or {},
        "provenance": record.get("provenance") or {},
        "window": (record.get("window")
                   or (record.get("provenance") or {}).get("window")),
    }
    base = {
        "cohort_id": cohort_id,
        "case_id": record.get("case_id"),
        "source_path": declared["source_path"],
        "declared": declared,
    }
    scope_decl = str(record.get("scope") or "").strip()
    missing = []
    if scope_decl not in _ROLE_SCOPES:
        missing.append("scope")
        scope_decl = None
    resources = declared["resources"]
    if resources is None:
        missing.append("resources")
    elif scope == "independent_resource" and resource not in resources:
        return {**base, "state": "excluded", "reason": "resource_not_selected"}
    phase_decl = str(record.get("phase") or "").strip()
    if not phase_decl:
        missing.append("phase")
    elif phase_decl != phase:
        return {**base, "state": "excluded", "reason": "phase_not_selected"}
    unit_decl = str(record.get("work_unit") or "").strip()
    if not unit_decl:
        missing.append("work_unit")
    elif unit_decl != work_unit:
        raise ValueError(
            "resource_service_role_record_work_unit_mismatch:"
            + unit_decl + "!=" + work_unit)
    work = record.get("work")
    if work is None:
        missing.append("work")
    provenance = record.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        raise ValueError("resource_service_role_record_provenance_requires_mapping")
    provenance = dict(provenance)
    window = (provenance.get("window") or provenance.get("observation_kind")
              or record.get("window"))
    if window is not None:
        window = str(window).strip().lower()
    if window in _FORBIDDEN_ROLE_WINDOWS:
        return {**base, "state": "gap", "reason":
                "whole_step_or_body_window_is_not_a_service_phase",
                "missing": ["resource_phase_window"]}
    declared_duration = record.get("resource_duration_us")
    duration_field = None
    if declared_duration is not None:
        duration_selector = declared_duration
        duration_field = "resource_duration_us"
    elif record.get("duration_us") is not None:
        if scope_decl == "joint_phase":
            duration_selector = record["duration_us"]
            duration_field = "duration_us_joint_window"
        elif scope_decl == "independent_resource" and window in _PHASE_WINDOWS:
            duration_selector = record["duration_us"]
            duration_field = "duration_us_resource_phase_alias"
        else:
            duration_selector = None
            missing.append("resource_duration_us")
    else:
        duration_selector = None
        missing.append("resource_duration_us")
    if scope_decl == "independent_resource" and window in _JOINT_ROLE_WINDOWS:
        return {**base, "state": "gap",
                "reason": "mixed_window_declared_as_independent_resource",
                "missing": ["independent_resource_window"]}
    declared_boundary = str(record.get("boundary") or "").strip()
    if declared_boundary and declared_boundary not in _ROLE_SERVICE_BOUNDARIES:
        return {**base, "state": "gap", "reason": "boundary_invalid",
                "missing": ["boundary"]}
    if boundary is not None:
        if declared_boundary and declared_boundary != boundary:
            raise ValueError(
                "resource_service_role_record_boundary_conflict:"
                + declared_boundary + "!=" + boundary)
        selected_boundary = boundary
    else:
        selected_boundary = declared_boundary or "resource_phase"
    record_rate_scope = record.get("rate_scope")
    if record_rate_scope is not None:
        record_rate_scope = str(record_rate_scope).strip()
        if record_rate_scope not in {"aggregate", "per_owner"}:
            return {**base, "state": "gap", "reason": "rate_scope_invalid",
                    "missing": ["rate_scope"]}
    if (rate_scope is not None and record_rate_scope is not None
            and record_rate_scope != rate_scope):
        raise ValueError(
            "resource_service_role_record_owner_scope_conflict:rate_scope")
    selected_rate_scope = record_rate_scope or rate_scope or "aggregate"
    record_units = (record.get("active_units")
                    if record.get("active_units") is not None else
                    record.get("owner_units")
                    if record.get("owner_units") is not None else
                    record.get("owners"))
    if record_units is not None:
        if isinstance(record_units, bool) or not isinstance(record_units, int) \
                or record_units < 1:
            return {**base, "state": "gap", "reason": "active_units_invalid",
                    "missing": ["active_units"]}
    if (active_units is not None and record_units is not None
            and int(record_units) != int(active_units)):
        raise ValueError(
            "resource_service_role_record_owner_scope_conflict:active_units")
    selected_units = int(record_units or active_units or 1)
    declared_conditions, merged_conditions = _role_conditions(
        record, conditions=conditions, hardware=hardware)
    if missing:
        return {**base, "state": "gap",
                "reason": "missing_explicit_resource_phase_work",
                "missing": missing}
    if resources is not None and scope_decl == "independent_resource" \
            and tuple(resources) != (resource,):
        return {**base, "state": "gap",
                "reason": "independent_resource_requires_exactly_the_selected_resource",
                "missing": ["resources"]}
    contract = {
        "cohort_id": cohort_id,
        "scope": scope_decl,
        "resources": tuple(resources),
        "phase": phase_decl,
        "work_unit": unit_decl,
        "work": work,
        "duration_us": duration_selector,
        "duration_field": duration_field,
        "boundary": selected_boundary,
        "rate_scope": selected_rate_scope,
        "active_units": selected_units,
        "conditions": merged_conditions,
        "declared_conditions": declared_conditions,
        "provenance": provenance,
        "window": window,
        "supply": record.get("supply", provenance.get("supply")),
        "startup_us": (record.get("startup_us")
                       if record.get("startup_us") is not None
                       else provenance.get("startup_us")),
        "source_path": declared["source_path"],
        "locator": declared["locator"],
        "case_id": record.get("case_id"),
    }
    state = "joint" if scope_decl == "joint_phase" else "candidate"
    return {**base, "state": state, "contract": contract, "declared": declared}


def _role_gap_row(extracted, *, reason=None, missing=None):
    return {
        "cohort_id": extracted["cohort_id"],
        "case_id": extracted.get("case_id"),
        "source_path": extracted.get("source_path"),
        "reason": reason or extracted.get("reason")
        or "missing_explicit_resource_phase_work",
        "missing": list(missing if missing is not None
                        else extracted.get("missing") or ()),
        "declared": extracted.get("declared"),
    }


def _role_joint_evidence(contract):
    return {
        "cohort_id": contract["cohort_id"],
        "case_id": contract.get("case_id"),
        "scope": contract["scope"],
        "resources": list(contract["resources"]),
        "phase": contract["phase"],
        "work_unit": contract["work_unit"],
        "work": contract["work"],
        "duration_us": contract["duration_us"],
        "duration_field": contract["duration_field"],
        "boundary": contract["boundary"],
        "rate_scope": contract["rate_scope"],
        "active_units": contract["active_units"],
        "conditions": contract["conditions"],
        "window": contract["window"],
        "provenance": contract["provenance"],
        "source_path": contract["source_path"],
        "locator": contract["locator"],
        "resource_identified": False,
    }


def _role_service_source(contract):
    return {
        "cohort_id": contract["cohort_id"],
        "case_id": contract.get("case_id"),
        "source_path": contract["source_path"],
        "locator": contract.get("locator"),
        "window": contract.get("window"),
        "duration_field": contract.get("duration_field"),
        "work_unit": contract.get("work_unit"),
        "rate_scope": contract.get("rate_scope"),
        "active_units": contract.get("active_units"),
    }


def fit_resource_service_from_role_records(
        records, *, resource, phase, work_unit, boundary=None,
        rate_scope=None, active_units=None, conditions=None, hardware=None,
        scope="independent_resource", supply=None, startup_us=None,
        response_us=0):
    """Fit one normalized service only from explicit in-sample source work.

    The caller's ``resource/phase/work_unit`` select one homogeneous source
    observation contract; they do not fill in a missing record field.  Each
    training record keeps its declared scope, resource set, phase, window,
    work unit, owner scope, boundary, conditions and provenance.  A joint or
    mixed window is retained as joint evidence and is never promoted to an
    independent resource rate.  ``duration_us`` aliases
    ``resource_duration_us`` only when the source declares the same resource
    phase window; otherwise the record stays underdetermined.

    Frozen transfer and not-declared records are preserved separately and
    never enter the fit.  The fit itself still runs through
    ``fit_resource_service -> normalize_resource_service``.
    """
    partition = calibration_role_partitions(records)
    resource = str(resource).strip()
    phase = str(phase).strip()
    work_unit = str(work_unit).strip()
    if not resource or not phase or not work_unit:
        raise ValueError("role_service_fit_requires_resource_phase_work_unit")
    scope = str(scope).strip()
    if scope not in _ROLE_SCOPES:
        raise ValueError("role_service_fit_scope_unknown")
    rate_scope = None if rate_scope is None else str(rate_scope).strip()
    if rate_scope is not None and rate_scope not in {"aggregate", "per_owner"}:
        raise ValueError("role_service_fit_rate_scope_unknown")
    boundary = None if boundary is None else str(boundary).strip()
    if boundary is not None and boundary not in _ROLE_SERVICE_BOUNDARIES:
        raise ValueError("role_service_fit_boundary_invalid")
    candidates = []
    joint = []
    underdetermined = []
    excluded = []
    participation = []
    for record in partition["in_sample"]:
        extracted = _role_contract(
            record, resource=resource, phase=phase, work_unit=work_unit,
            boundary=boundary, rate_scope=rate_scope,
            active_units=active_units, conditions=conditions,
            hardware=hardware, scope=scope)
        if extracted["state"] == "excluded":
            excluded.append({
                "cohort_id": extracted["cohort_id"],
                "case_id": extracted.get("case_id"),
                "source_path": extracted.get("source_path"),
                "reason": extracted["reason"],
                "declared": extracted.get("declared"),
            })
            participation.append({
                "cohort_id": extracted["cohort_id"],
                "case_id": extracted.get("case_id"),
                "status": "excluded",
                "reason": extracted["reason"],
            })
            continue
        if extracted["state"] == "joint":
            contract = extracted["contract"]
            joint.append(_role_joint_evidence(contract))
            if scope == "independent_resource":
                underdetermined.append(_role_gap_row(
                    extracted, reason="joint_window_not_independent_resource",
                    missing=["independent_resource_measurement"]))
                participation.append({
                    "cohort_id": extracted["cohort_id"],
                    "case_id": extracted.get("case_id"),
                    "status": "joint_evidence_only",
                    "reason": "joint_window_not_independent_resource",
                    "scope": "joint_phase",
                })
            else:
                candidates.append(contract)
                participation.append({
                    "cohort_id": extracted["cohort_id"],
                    "case_id": extracted.get("case_id"),
                    "status": "pending_training",
                    "scope": "joint_phase",
                })
            continue
        if extracted["state"] == "gap":
            underdetermined.append(_role_gap_row(extracted))
            participation.append({
                "cohort_id": extracted["cohort_id"],
                "case_id": extracted.get("case_id"),
                "status": "underdetermined",
                "reason": extracted.get("reason"),
                "missing": list(extracted.get("missing") or ()),
            })
            continue
        if scope == "joint_phase":
            underdetermined.append(_role_gap_row(
                extracted, reason="independent_resource_not_a_joint_phase",
                missing=["joint_phase_measurement"]))
            participation.append({
                "cohort_id": extracted["cohort_id"],
                "case_id": extracted.get("case_id"),
                "status": "excluded",
                "reason": "independent_resource_not_a_joint_phase",
            })
            continue
        candidates.append(extracted["contract"])
        participation.append({
            "cohort_id": extracted["cohort_id"],
            "case_id": extracted.get("case_id"),
            "status": "pending_training",
            "scope": extracted["contract"]["scope"],
        })

    base = {
        "schema": "mindie-sim.role-service-fit/2",
        "role_counts": partition["counts"],
        "total": partition["total"],
        "training_ids": [record["cohort_id"]
                         for record in partition["in_sample"]],
        "frozen_transfer_ids": [record["cohort_id"]
                                for record in partition["frozen_transfer"]],
        "not_declared_ids": [record["cohort_id"]
                            for record in partition["not_declared"]],
        "excluded": excluded,
        "underdetermined": underdetermined,
        "joint_evidence": joint,
        "training_participation": participation,
        "policy": partition["policy"],
        "scope": scope,
        "needed_evidence": list(_ROLE_NEEDED_EVIDENCE),
        "identifiable_combination": (
            "fixed_us + work/effective_rate; from a whole-call total only "
            "this linear combination is identifiable"),
    }
    def _finalize_participation(status, reason=None):
        for item in participation:
            if item["status"] == "pending_training":
                item["status"] = status
                if reason is not None:
                    item["reason"] = reason

    if not candidates:
        _finalize_participation("not_fitted")
        if scope == "independent_resource" and joint:
            reason = "joint_window_not_independent_resource"
        elif scope == "joint_phase" and not joint:
            reason = "no_in_sample_joint_phase_with_explicit_work"
        else:
            reason = "no_in_sample_record_with_explicit_resource_phase_work"
        return {**base, "status": "underdetermined", "reason": reason,
                "service": None}

    # The selected records must describe one physical packet.  A different
    # window/unit/owner/condition set is not silently averaged into it.
    for key in ("work_unit", "boundary", "rate_scope", "active_units",
                "conditions"):
        values = {_canonical(item[key]) for item in candidates}
        if len(values) != 1:
            raise ValueError(
                "resource_service_role_record_contract_not_homogeneous:" + key)
    if scope == "joint_phase":
        resource_sets = {tuple(item["resources"]) for item in candidates}
        if len(resource_sets) != 1:
            raise ValueError(
                "resource_service_role_record_contract_not_homogeneous:resources")
    windows = {item["window"] for item in candidates}
    if len(windows) != 1:
        raise ValueError(
            "resource_service_role_record_contract_not_homogeneous:window")
    common_window = next(iter(windows))
    if scope == "independent_resource" and common_window in _JOINT_ROLE_WINDOWS:
        raise ValueError(
            "resource_service_role_record_contract_not_homogeneous:window")
    declared_supplies = {_canonical(item["supply"])
                         for item in candidates
                         if item.get("supply") is not None}
    if supply is not None and declared_supplies and \
            _canonical(supply) not in declared_supplies:
        raise ValueError("resource_service_role_record_supply_conflict")
    if len(declared_supplies) > 1:
        raise ValueError("resource_service_role_record_supply_conflict")
    if supply is None and declared_supplies:
        supply = candidates[0]["supply"]
    declared_startups = {_canonical(item["startup_us"])
                         for item in candidates
                         if item.get("startup_us") is not None}
    if startup_us is not None and declared_startups and \
            _canonical(startup_us) not in declared_startups:
        raise ValueError("resource_service_role_record_startup_conflict")
    if len(declared_startups) > 1:
        raise ValueError("resource_service_role_record_startup_conflict")
    if startup_us is None and declared_startups:
        startup_us = candidates[0]["startup_us"]

    distinct_work = {item["work"] for item in candidates}
    if len(distinct_work) < 2 and supply is None and startup_us is None:
        for item in candidates:
            underdetermined.append({
                "cohort_id": item["cohort_id"],
                "case_id": item.get("case_id"),
                "source_path": item["source_path"],
                "reason": "fit_requires_two_work_sizes_or_declared_rate",
                "missing": ["second_training_work", "declared_supply"],
                "declared": None,
            })
        _finalize_participation(
            "not_fitted", "fit_requires_two_work_sizes_or_declared_rate")
        return {**base, "status": "underdetermined",
                "reason": "fit_requires_two_work_sizes_or_declared_rate",
                "service": None}

    resources = ([resource] if scope == "independent_resource"
                 else list(candidates[0]["resources"]))
    packet = {
        "id": f"role-fit:{resource}:{phase}",
        "scope": scope,
        "resources": resources,
        "phase": phase,
        "work_unit": work_unit,
        "boundary": candidates[0]["boundary"],
        "rate_scope": candidates[0]["rate_scope"],
        "active_units": candidates[0]["active_units"],
        "conditions": candidates[0]["conditions"],
        "samples": [{
            "work": item["work"],
            "duration_us": item["duration_us"],
            "run_id": item["cohort_id"],
            "repeat_index": 0,
            "split": "train",
            "source": str(item["source_path"] or item["cohort_id"]),
        } for item in candidates],
        "provenance": {
            "window": common_window or (
                "mixed" if scope == "joint_phase" else "resource_phase"),
            "observation_kind": (
                "joint_phase" if scope == "joint_phase"
                else "explicit_resource_phase"),
            "boundary_kind": candidates[0]["boundary"],
            "scope": "independent" if scope == "independent_resource" else "joint",
            "role_ledger": {
                "training_role": _ROLE_IN_SAMPLE,
                "role_counts": dict(partition["counts"]),
                "training_sources": [_role_service_source(item)
                                      for item in candidates],
            },
        },
    }
    if supply is not None:
        packet["supply"] = dict(supply)
    if startup_us is not None:
        packet["startup_us"] = startup_us
        packet["response_us"] = response_us
    if scope == "joint_phase":
        packet["identification"] = {"resource_identified": False,
                                    "status": "joint_identified"}
    try:
        fitted = fit_resource_service(packet, hardware=hardware)
    except ValueError as error:
        text = str(error)
        if "requires_two_work_sizes_or_a_declared_rate" in text:
            _finalize_participation(
                "not_fitted", "fit_requires_two_work_sizes_or_declared_rate")
            return {**base, "status": "underdetermined",
                    "reason": "fit_requires_two_work_sizes_or_declared_rate",
                    "service": None}
        raise
    service = dict(fitted["service"])
    provenance = dict(service.get("provenance") or {})
    provenance.update({
        "training_role": "in_sample",
        "role_counts": dict(partition["counts"]),
        "frozen_transfer_role": "independent_transfer_not_in_fit",
        "not_declared_role": "excluded_neither_training_nor_holdout",
        "role_sources": [_role_service_source(item) for item in candidates],
        "source_contract": {
            "scope": scope,
            "resources": resources,
            "phase": phase,
            "work_unit": work_unit,
            "boundary": candidates[0]["boundary"],
            "rate_scope": candidates[0]["rate_scope"],
            "active_units": candidates[0]["active_units"],
            "conditions": candidates[0]["conditions"],
            "window": common_window,
            "duration_fields": sorted({
                item["duration_field"] for item in candidates
                if item.get("duration_field")}),
        },
    })
    service["provenance"] = provenance
    _finalize_participation("trained")
    return {
        **base,
        "status": "fitted",
        "service": service,
        "fit": fitted.get("fit"),
        "formula": fitted.get("formula"),
    }
