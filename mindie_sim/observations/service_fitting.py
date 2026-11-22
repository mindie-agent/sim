"""Resource-service role contracts, regression fitting and orchestration.

[Role] Define role fitting contracts, solve non-negative physical service
  regressions, and emit calibrated service models.
[Invariants] Conserves physical units and reports exact fit residuals.
[Timeouts & Locks] Pure numeric fitting; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections.abc import Mapping



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


_HASH_CHUNK_BYTES = 1024 * 1024


def _sha256_file(path):
    import hashlib
    from pathlib import Path
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def load_role_ledger(path):
    """Read a JSONL cohort ledger without opening any referenced capture."""
    import json
    from pathlib import Path
    records = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
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


def _declared_role_contract(record, cohort_id):
    return {
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


def _role_contract(record, *, resource, phase, work_unit, boundary,
                   rate_scope, active_units, conditions, hardware, scope):
    """Extract one source observation contract exactly as declared.

    ``state`` is one of ``candidate``, ``joint``, ``excluded`` or ``gap``.
    The caller parameter is a selector and a source-level declaration; it is
    never silently substituted for a missing per-record contract field.
    """
    cohort_id = str(record["cohort_id"])
    declared = _declared_role_contract(record, cohort_id)
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


def _role_fit_candidates(
    records, *, resource, phase, work_unit, boundary, rate_scope,
    active_units, conditions, hardware, scope,
):
    candidates = []
    joint = []
    underdetermined = []
    excluded = []
    participation = []
    for record in records:
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
    return candidates, joint, underdetermined, excluded, participation


def _finalize_role_participation(participation, status, reason=None):
    for item in participation:
        if item["status"] == "pending_training":
            item["status"] = status
            if reason is not None:
                item["reason"] = reason


def _resolve_homogeneous_role_fit(candidates, *, scope, supply, startup_us):
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
    return common_window, supply, startup_us


def _role_fit_packet(
    candidates, counts, *, resource, phase, work_unit, scope, common_window,
    supply, startup_us, response_us,
):
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
                "role_counts": dict(counts),
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
    return packet, resources


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
    candidates, joint, underdetermined, excluded, participation = _role_fit_candidates(
        partition["in_sample"], resource=resource, phase=phase, work_unit=work_unit,
        boundary=boundary, rate_scope=rate_scope, active_units=active_units,
        conditions=conditions, hardware=hardware, scope=scope)

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

    if not candidates:
        _finalize_role_participation(participation, "not_fitted")
        if scope == "independent_resource" and joint:
            reason = "joint_window_not_independent_resource"
        elif scope == "joint_phase" and not joint:
            reason = "no_in_sample_joint_phase_with_explicit_work"
        else:
            reason = "no_in_sample_record_with_explicit_resource_phase_work"
        return {**base, "status": "underdetermined", "reason": reason,
                "service": None}

    common_window, supply, startup_us = _resolve_homogeneous_role_fit(
        candidates, scope=scope, supply=supply, startup_us=startup_us)

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
        _finalize_role_participation(
            participation, "not_fitted", "fit_requires_two_work_sizes_or_declared_rate")
        return {**base, "status": "underdetermined",
                "reason": "fit_requires_two_work_sizes_or_declared_rate",
                "service": None}

    packet, resources = _role_fit_packet(
        candidates, partition["counts"], resource=resource, phase=phase, work_unit=work_unit,
        scope=scope, common_window=common_window, supply=supply,
        startup_us=startup_us, response_us=response_us)
    try:
        fitted = fit_resource_service(packet, hardware=hardware)
    except ValueError as error:
        text = str(error)
        if "requires_two_work_sizes_or_a_declared_rate" in text:
            _finalize_role_participation(
                participation, "not_fitted", "fit_requires_two_work_sizes_or_declared_rate")
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
    _finalize_role_participation(participation, "trained")
    return {
        **base,
        "status": "fitted",
        "service": service,
        "fit": fitted.get("fit"),
        "formula": fitted.get("formula"),
    }


from collections import defaultdict
from collections.abc import Mapping
import json
import math
from statistics import median

from mindie_sim.observations.resource_services import (
    normalize_resource_service,
    resolve_nominal_rate,
)
from mindie_sim.prediction.service_regime_domains.services import ServiceCurve


_FORBIDDEN_SERVICE_BOUNDARIES = frozenset({
    "step", "body", "request", "serving", "e2e", "engine_step", "body_step",
    "whole_step", "whole_body",
})
_TARGET_KEYS = frozenset({
    "target_ms", "target_us", "target_duration_ms", "target_duration_us",
    "body_residual_ms", "step_residual_ms", "residual_ms",
})
_FORBIDDEN_WINDOWS = frozenset({
    "step", "body", "serving", "e2e", "engine_step", "body_step",
    "whole_step", "whole_body", "request",
})
_JOINT_ONLY_WINDOWS = frozenset({"joint", "mixed", "overlapping"})
_FIXED_TERM_TOLERANCE_US = 1e-9


def nonnegative_affine(points):
    """Identify startup + work/rate, constraining startup to be nonnegative."""
    n = len(points)
    if n < 2 or len({x for x, _ in points}) < 2:
        raise ValueError("rate_fit_requires_training_trials_at_distinct_work_sizes")
    if any(not math.isfinite(v) or v <= 0 for point in points for v in point):
        raise ValueError("primitive_fit_requires_positive_finite_work_and_duration")
    x = sum(p[0] for p in points) / n
    y = sum(p[1] for p in points) / n
    variance = sum((p[0] - x) ** 2 for p in points)
    slope = sum((a - x) * (b - y) for a, b in points) / variance
    intercept = y - slope * x
    if intercept < 0:
        intercept = 0
        slope = sum(a * b for a, b in points) / sum(a ** 2 for a, _ in points)
    if slope <= 0:
        raise ValueError("observations_do_not_identify_a_positive_service_rate")
    return intercept, 1 / slope


def fit_service_curve(packet):
    groups = defaultdict(lambda: defaultdict(list))
    repeats = {}
    training_sources = set()
    heldout = []
    for sample in packet["samples"]:
        work, duration = sample["work"], sample["duration_ms"]
        if not all(math.isfinite(v) and v > 0 for v in (work, duration)):
            raise ValueError("primitive_fit_requires_positive_finite_work_and_duration")
        key = sample["run_id"], sample["repeat_index"], work
        record = duration, sample["split"]
        if key in repeats:
            if repeats[key] != record:
                raise ValueError("conflicting_primitive_repeat_or_split")
            continue
        repeats[key] = record
        if sample["split"] == "train":
            groups[work][sample["run_id"]].append(duration)
            training_sources.add(sample["source"])
        elif sample["split"] == "heldout":
            heldout.append(sample)
        else:
            raise ValueError("service_sample_split_requires_train_or_heldout")
    knots = tuple((work, median(median(values) for values in runs.values()))
                  for work, runs in sorted(groups.items()))
    startup, rate = nonnegative_affine(knots)
    # A multi-regime ladder (small OneShot plateau + asymptotic large-payload
    # slope) can put the affine intercept above the smallest measured duration.
    # The curve startup must remain below every served duration, so cap it at
    # the tightest per-knot residual. The affine reference keeps the raw fit.
    curve_startup = startup
    if curve_startup >= min(t for _, t in knots):
        curve_startup = max(0.0, min(t - x / rate for x, t in knots))
    model = ServiceCurve(
        packet["resource"], packet["work_unit"], packet["identity"], packet["boundary"],
        packet["active_units"], curve_startup, knots, {
            **packet.get("provenance", {}),
            "training_sources": sorted(training_sources),
            "training_trials": sum(len(runs) for runs in groups.values()),
            "training_samples": sum(len(values) for runs in groups.values() for values in runs.values()),
            "aggregation": "median_of_run_medians_at_each_work",
            "startup_estimator": "nonnegative_affine_on_training_knots",
        })
    rows = []
    for sample in heldout:
        estimate = model.estimate(sample["work"])
        predicted = estimate.duration_ms if estimate else None
        rows.append({
            "work": sample["work"], "observed_ms": sample["duration_ms"],
            "predicted_ms": predicted, "source": sample["source"], "run_id": sample["run_id"],
            "repeat_index": sample["repeat_index"],
            "basis": estimate.basis if estimate else "outside_domain",
            "relative_error": predicted / sample["duration_ms"] - 1 if estimate else None,
        })
    return {"model": model.to_dict(), "model_id": model.model_id,
            "affine_reference": {"startup_ms": startup, "units_per_ms": rate},
            "heldout": rows}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"resource_service_fit_{name}_requires_number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"resource_service_fit_{name}_requires_positive_finite")
    return result


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"resource_service_fit_{name}_requires_number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"resource_service_fit_{name}_requires_nonnegative_finite")
    return result


def _packet_resources(packet):
    raw = packet.get("resources")
    if raw is None:
        raw = [packet["resource"]] if packet.get("resource") else None
    elif isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("resource_service_fit_requires_a_resource_list")
    resources = tuple(str(item).strip() for item in raw)
    if any(not item for item in resources) or len(set(resources)) != len(resources):
        raise ValueError("resource_service_fit_requires_distinct_resources")
    return resources


def _sample_record(sample):
    if not isinstance(sample, Mapping):
        raise ValueError("resource_service_fit_sample_requires_mapping")
    for key in _TARGET_KEYS:
        if sample.get(key) is not None:
            raise ValueError("resource_service_fit_rejects_target_duration:" + key)
    work = _positive(sample.get("work"), "work")
    has_us = sample.get("duration_us") is not None
    has_ms = sample.get("duration_ms") is not None
    if has_us and has_ms:
        raise ValueError("resource_service_fit_sample_requires_one_duration_unit")
    if has_us:
        duration_us = _positive(sample["duration_us"], "duration_us")
    elif has_ms:
        duration_us = _positive(sample["duration_ms"], "duration_ms") * 1000
    else:
        raise ValueError("resource_service_fit_sample_requires_duration")
    run_id = sample.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("resource_service_fit_sample_requires_run_id")
    repeat_index = sample.get("repeat_index", 0)
    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
        raise ValueError("resource_service_fit_sample_repeat_index_requires_integer")
    if repeat_index < 0:
        raise ValueError("resource_service_fit_sample_repeat_index_requires_nonnegative")
    split = sample.get("split")
    if split not in {"train", "heldout"}:
        raise ValueError("resource_service_fit_sample_split_requires_train_or_heldout")
    source = sample.get("source")
    if source is not None and not isinstance(source, str):
        raise ValueError("resource_service_fit_sample_source_requires_text")
    return {
        "work": work,
        "duration_us": duration_us,
        "run_id": run_id.strip(),
        "repeat_index": repeat_index,
        "split": split,
        "source": source or run_id.strip(),
    }


def _training_knots(samples):
    groups = defaultdict(lambda: defaultdict(list))
    repeats = {}
    training_sources = set()
    training_runs = set()
    heldout = []
    training_samples = 0
    for sample in samples:
        key = sample["run_id"], sample["repeat_index"], sample["work"]
        record = sample["duration_us"], sample["split"]
        if key in repeats:
            if repeats[key] != record:
                raise ValueError("conflicting_resource_service_repeat_or_split")
            continue
        repeats[key] = record
        if sample["split"] == "train":
            groups[sample["work"]][sample["run_id"]].append(sample["duration_us"])
            training_sources.add(sample["source"])
            training_runs.add(sample["run_id"])
            training_samples += 1
        else:
            heldout.append(sample)
    if not groups:
        raise ValueError("resource_service_fit_requires_training_samples")
    knots = tuple(
        (work, median(median(values) for values in runs.values()))
        for work, runs in sorted(groups.items())
    )
    run_work_cells = sum(len(runs) for runs in groups.values())
    return {
        "knots": knots,
        "heldout": heldout,
        "training_sources": sorted(training_sources),
        "training_runs": len(training_runs),
        "training_run_work_cells": run_work_cells,
        "training_samples": training_samples,
    }


def _cap_fixed(fixed_us, rate_per_us, knots):
    if fixed_us < 0:
        fixed_us = 0.0
    capped = fixed_us >= min(duration for _, duration in knots)
    if capped:
        fixed_us = max(0.0, min(duration - work / rate_per_us
                               for work, duration in knots))
    return fixed_us, capped


def _declared_fixed(knots, effective_rate):
    candidates = [duration - work * 1e6 / effective_rate
                  for work, duration in knots]
    fixed_us = median(candidates)
    if fixed_us < -_FIXED_TERM_TOLERANCE_US:
        raise ValueError(
            "resource_service_observations_contradict_declared_rate")
    return max(0.0, fixed_us)


def _resource_fit_contract(packet, hardware):
    scope = str(packet.get("scope") or "").strip()
    if scope not in {"independent_resource", "joint_phase"}:
        raise ValueError("resource_service_fit_scope_unknown")
    resources = _packet_resources(packet)
    if scope == "independent_resource" and len(resources) != 1:
        raise ValueError(
            "resource_service_mixed_window_cannot_identify_independent_resource")
    phase = str(packet.get("phase") or "").strip()
    if not phase:
        raise ValueError("resource_service_fit_requires_phase")
    work_unit = str(packet.get("work_unit") or "").strip()
    if not work_unit:
        raise ValueError("resource_service_fit_requires_work_unit")
    boundary = str(packet.get("boundary") or "").strip()
    if boundary in _FORBIDDEN_SERVICE_BOUNDARIES:
        raise ValueError(
            "resource_service_fit_boundary_must_be_resource_phase_or_device_kernel")
    if boundary not in {"resource_phase", "device_kernel"}:
        raise ValueError(
            "resource_service_fit_boundary_must_be_resource_phase_or_device_kernel")

    provenance = packet.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        raise ValueError("resource_service_fit_provenance_requires_mapping")
    provenance = dict(provenance)
    for key in ("window", "observation_kind", "boundary_kind", "scope"):
        declared = str(provenance.get(key, "")).strip().lower()
        if declared in _FORBIDDEN_WINDOWS:
            raise ValueError(
                "resource_service_fit_rejects_step_or_body_window:" + key)
        if scope == "independent_resource" and declared in _JOINT_ONLY_WINDOWS:
            raise ValueError(
                "resource_service_mixed_window_cannot_identify_independent_resource:"
                + key)

    conditions = packet.get("conditions") or {}
    if not isinstance(conditions, Mapping):
        raise ValueError("resource_service_fit_conditions_requires_mapping")
    conditions = dict(conditions)
    if hardware:
        declared_hardware = conditions.get("hardware")
        if declared_hardware and str(declared_hardware) != str(hardware):
            raise ValueError("resource_service_hardware_mismatch")
        conditions["hardware"] = str(hardware)

    active_units = packet.get("active_units", 1)
    if isinstance(active_units, bool) or not isinstance(active_units, int) \
            or active_units < 1:
        raise ValueError("resource_service_fit_active_units_requires_positive_integer")
    rate_scope = str(packet.get("rate_scope", "aggregate")).strip()
    if rate_scope not in {"aggregate", "per_owner"}:
        raise ValueError("resource_service_fit_rate_scope_unknown")

    return {
        "scope": scope, "resources": resources, "phase": phase, "work_unit": work_unit,
        "boundary": boundary, "active_units": active_units, "rate_scope": rate_scope,
        "conditions": conditions,
    }, provenance


def _fit_resource_parameters(packet, contract, knots, declared_supply, nominal_rates, hardware):
    declared_response = _nonnegative(packet.get("response_us", 0), "response_us")
    declared_startup = packet.get("startup_us")
    if declared_startup is not None:
        declared_startup = _nonnegative(declared_startup, "startup_us")
    affine_reference = None
    startup_capped = False
    nominal_rate = None
    nominal_multiplier = None
    declared_supply_kind = None
    if declared_supply is not None:
        if declared_startup is not None:
            raise ValueError(
                "resource_service_fit_supply_and_startup_are_mutually_exclusive")
        if not isinstance(declared_supply, Mapping):
            raise ValueError("resource_service_fit_supply_requires_mapping")
        declared_supply_kind = str(declared_supply.get("kind") or "").strip()
        rate_record = {
            "id": str(packet.get("id") or packet.get("service_id")
                      or "resource-service"),
            "scope": contract["scope"],
            "resources": list(contract["resources"]),
            "phase": contract["phase"],
            "work_unit": contract["work_unit"],
            "boundary": contract["boundary"],
            "startup_us": 0,
            "response_us": 0,
            "active_units": contract["active_units"],
            "rate_scope": contract["rate_scope"],
            "supply": dict(declared_supply),
            "conditions": contract["conditions"],
            "identification": {"status": "declared",
                               "resource_identified": False},
        }
        rate_service = normalize_resource_service(
            rate_record, nominal_rates=nominal_rates, hardware=hardware)
        effective_rate = rate_service.effective_rate
        nominal_rate = rate_service.nominal_rate
        nominal_multiplier = rate_service.nominal_multiplier
        fixed_us = _declared_fixed(knots, effective_rate)
        rate_source = "declared"
        response_us = declared_response
        if response_us > fixed_us + 1e-9:
            raise ValueError(
                "resource_service_fit_response_exceeds_observed_fixed_term")
        startup_us = max(0.0, fixed_us - response_us)
        separable = bool(packet.get("startup_response_separable",
                                    response_us > 0))
    elif declared_startup is not None:
        fixed_us = declared_startup + declared_response
        service_work = [(work, duration - fixed_us)
                        for work, duration in knots]
        if any(service_us <= 0 for _, service_us in service_work):
            raise ValueError(
                "resource_service_observations_contradict_declared_startup")
        slope_us_per_work = (
            sum(work * service_us for work, service_us in service_work)
            / sum(work * work for work, _ in service_work))
        if slope_us_per_work <= 0:
            raise ValueError(
                "observations_do_not_identify_a_positive_service_rate")
        rate_per_us = 1 / slope_us_per_work
        effective_rate = rate_per_us * 1e6
        if nominal_rates is not None and len(contract["resources"]) == 1:
            nominal_rate = resolve_nominal_rate(
                contract["resources"][0], nominal_rates, work_unit=contract["work_unit"],
                hardware=hardware, rate_scope=contract["rate_scope"])
        nominal_multiplier = (effective_rate / nominal_rate
                              if nominal_rate is not None else None)
        affine_reference = {
            "fixed_us": fixed_us,
            "fixed_source": "declared",
            "rate_per_us": rate_per_us,
            "rate_per_s": effective_rate,
        }
        rate_source = "fitted_with_declared_fixed"
        startup_us = declared_startup
        response_us = declared_response
        separable = True
    else:
        if len(knots) < 2:
            raise ValueError(
                "resource_service_fit_requires_two_work_sizes_or_a_declared_rate")
        affine_fixed, rate_per_us = nonnegative_affine(knots)
        effective_rate = rate_per_us * 1e6
        fixed_us, startup_capped = _cap_fixed(affine_fixed, rate_per_us, knots)
        if nominal_rates is not None and len(contract["resources"]) == 1:
            nominal_rate = resolve_nominal_rate(
                contract["resources"][0], nominal_rates, work_unit=contract["work_unit"],
                hardware=hardware, rate_scope=contract["rate_scope"])
        nominal_multiplier = (effective_rate / nominal_rate
                              if nominal_rate is not None else None)
        affine_reference = {
            "fixed_us": affine_fixed,
            "rate_per_us": rate_per_us,
            "rate_per_s": effective_rate,
        }
        rate_source = "fitted"
        response_us = declared_response
        if response_us > fixed_us + 1e-9:
            raise ValueError(
                "resource_service_fit_response_exceeds_observed_fixed_term")
        startup_us = max(0.0, fixed_us - response_us)
        separable = bool(packet.get("startup_response_separable",
                                    response_us > 0))

    return {
        "declared_startup": declared_startup, "declared_supply_kind": declared_supply_kind,
        "affine_reference": affine_reference, "startup_capped": startup_capped,
        "nominal_rate": nominal_rate, "nominal_multiplier": nominal_multiplier,
        "effective_rate": effective_rate, "fixed_us": fixed_us, "rate_source": rate_source,
        "response_us": response_us, "startup_us": startup_us, "separable": separable,
    }


def _resource_fit_identification(packet, scope, rate_source, declared_supply_kind):
    identification = packet.get("identification") or {}
    if not isinstance(identification, Mapping):
        raise ValueError("resource_service_fit_identification_requires_mapping")
    identification = dict(identification)
    fitted_rate = rate_source in {"fitted", "fitted_with_declared_fixed"}
    if "resource_identified" in identification:
        resource_identified = bool(identification["resource_identified"])
        if resource_identified and not fitted_rate:
            raise ValueError(
                "resource_service_declared_rate_cannot_be_reported_as_identified")
    else:
        resource_identified = (scope == "independent_resource" and fitted_rate)
    if scope == "joint_phase" and resource_identified:
        raise ValueError("joint_service_cannot_identify_an_internal_resource")
    status = identification.get("status")
    if status is None:
        if scope == "joint_phase":
            status = "joint_identified"
        elif declared_supply_kind == "nominal":
            status = "nominal_specification"
        elif resource_identified:
            status = "identified"
        elif rate_source == "declared":
            status = "partially_identified"
        else:
            status = "declared_effective_service"
    return identification, status, resource_identified


def _fitted_resource_record(packet, contract, provenance, grouped, parameters, declared_supply):
    scope = contract["scope"]
    rate_source = parameters["rate_source"]
    declared_supply_kind = parameters["declared_supply_kind"]
    separable = parameters["separable"]
    nominal_rate = parameters["nominal_rate"]
    effective_rate = parameters["effective_rate"]
    fixed_us = parameters["fixed_us"]
    declared_startup = parameters["declared_startup"]
    knots = grouped["knots"]
    identification, status, resource_identified = _resource_fit_identification(
        packet, scope, rate_source, declared_supply_kind)
    train_work = [work for work, _ in knots]
    heldout_work = sorted({sample["work"] for sample in grouped["heldout"]})
    fixed_source = "declared" if declared_startup is not None else "fitted"
    if declared_startup is not None:
        included_fixed = (
            "startup and response are declared separately; no fixed term is "
            "fitted from the observation")
    elif separable:
        included_fixed = (
            "declared response is separated; the remaining observed fixed term "
            "may include startup, occupancy or tail effects and is counted once")
    else:
        included_fixed = (
            "startup, occupancy and tail effects are combined in the observed "
            "fixed term and are counted once")
    if scope == "joint_phase":
        parameter_class = "joint_phase_service"
    elif declared_supply_kind == "nominal":
        parameter_class = "nominal_specification"
    else:
        parameter_class = "effective_resource_service"
    if scope == "joint_phase":
        rate_interpretation = (
            "joint_phase_service_rate_not_an_isolated_internal_resource_rate")
    elif declared_supply_kind == "nominal":
        rate_interpretation = (
            "declared_nominal_rate_with_observed_fixed_term_not_a_calibrated_service")
    elif resource_identified:
        rate_interpretation = "identified_effective_resource_service"
    else:
        rate_interpretation = (
            "declared_effective_resource_service_not_identified_as_isolated_peak")
    fit_report = {
        "rate_source": rate_source,
        "declared_supply_kind": declared_supply_kind,
        "fixed_source": fixed_source,
        "fixed_us": fixed_us,
        "startup_us": parameters["startup_us"],
        "response_us": parameters["response_us"],
        "effective_rate": effective_rate,
        "rate_scope": contract["rate_scope"],
        "nominal_rate": nominal_rate,
        "nominal_multiplier": parameters["nominal_multiplier"],
        "affine_reference": parameters["affine_reference"],
        "startup_capped": parameters["startup_capped"],
        "training_work": train_work,
        "training_runs": grouped["training_runs"],
        "training_run_work_cells": grouped["training_run_work_cells"],
        "training_samples": grouped["training_samples"],
        "aggregation": "median_of_run_medians_at_each_work",
        "included_fixed_effects": included_fixed,
        "residual_us": {
            "max_abs": max(abs(duration - (fixed_us + work * 1e6 / effective_rate))
                           for work, duration in knots),
            "mean_abs": sum(abs(duration - (fixed_us + work * 1e6 / effective_rate))
                            for work, duration in knots) / len(knots),
        },
    }
    if declared_supply is not None:
        service_supply = dict(declared_supply)
        if nominal_rate is not None and service_supply.get("nominal_rate") is None:
            service_supply["nominal_rate"] = nominal_rate
    else:
        service_supply = {
            "kind": "absolute_rate",
            "rate": effective_rate,
            **({"nominal_rate": nominal_rate} if nominal_rate is not None else {}),
        }
    service_record = {
        "id": str(packet.get("id") or packet.get("service_id")
                  or "resource-service"),
        "scope": scope,
        "resources": list(contract["resources"]),
        "phase": contract["phase"],
        "work_unit": contract["work_unit"],
        "boundary": contract["boundary"],
        "startup_us": parameters["startup_us"],
        "response_us": parameters["response_us"],
        "active_units": contract["active_units"],
        "rate_scope": contract["rate_scope"],
        "supply": service_supply,
        "conditions": contract["conditions"],
        "identification": {
            **identification,
            "status": status,
            "resource_identified": resource_identified,
            "rate_source": rate_source,
            "fixed_identified": declared_startup is None,
            "fixed_source": fixed_source,
            "startup_response_separable": separable,
            "fixed_components": (
                "declared_startup_and_response"
                if declared_startup is not None else
                "startup_plus_declared_response"
                if separable else "combined_fixed"),
            "train_work": train_work,
            "heldout_work": heldout_work,
            "training_work_range": [min(train_work), max(train_work)],
            "formula_domain": "all_nonnegative_work",
            "concurrency_scope": identification.get(
                "concurrency_scope", "isolated_invocation"),
            "included_fixed_effects": included_fixed,
        },
        "provenance": {
            **provenance,
            "fit": {**dict(provenance.get("fit") or {}), **fit_report},
            "training_sources": grouped["training_sources"],
            "parameter_class": parameter_class,
            **({"rate_interpretation": rate_interpretation}
               if "rate_interpretation" not in provenance else {}),
            "normalized_rate_interpretation": rate_interpretation,
            "concurrency": "not_fitted; one invocation boundary",
            "included_effects": included_fixed,
        },
    }
    return service_record, fit_report


def fit_resource_service(packet, *, nominal_rates=None, hardware=None):
    """Fit one fixed term plus steady-state rate from independent trials.

    A declared ``supply`` fixes the rate and calibrates only the fixed term from
    one or more work sizes.  Otherwise at least two distinct training work sizes
    are required to identify startup and rate.  A joint/mixed window is fitted
    only as a joint-phase service; it never emits an internal resource rate.
    Sample ``work`` uses the declared ``rate_scope`` (aggregate or per-owner);
    it is not silently converted between the two scopes.
    """
    if not isinstance(packet, Mapping):
        raise ValueError("resource_service_fit_requires_mapping")
    for key in _TARGET_KEYS:
        if packet.get(key) is not None:
            raise ValueError("resource_service_fit_rejects_target_duration:" + key)
    samples = packet.get("samples")
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("resource_service_fit_requires_samples")
    records = [_sample_record(sample) for sample in samples]

    contract, provenance = _resource_fit_contract(packet, hardware)
    grouped = _training_knots(records)
    declared_supply = packet.get("supply")
    parameters = _fit_resource_parameters(
        packet, contract, grouped["knots"], declared_supply, nominal_rates, hardware)
    service_record, fit_report = _fitted_resource_record(
        packet, contract, provenance, grouped, parameters, declared_supply)
    normalized = normalize_resource_service(service_record, hardware=hardware)
    heldout_rows = []
    for sample in grouped["heldout"]:
        prediction = normalized.prediction(sample["work"])
        predicted_us = prediction["duration_us"]
        heldout_rows.append({
            "work": sample["work"],
            "observed_us": sample["duration_us"],
            "predicted_us": predicted_us,
            "relative_error": predicted_us / sample["duration_us"] - 1,
            "source": sample["source"],
            "run_id": sample["run_id"],
            "repeat_index": sample["repeat_index"],
            "basis": prediction["basis"],
            "inside_training_range": prediction["inside_training_range"],
            "extrapolated": prediction["extrapolated"],
        })
    return {
        "schema": "mindie-sim.resource-service-fit/1",
        "service": normalized.to_dict(),
        "model_id": normalized.model_id,
        "formula": normalized.formula,
        "fit": fit_report,
        "heldout": heldout_rows,
        "normalization": normalized.normalization,
        "source": (provenance.get("source") or provenance.get("sources")
                   or packet.get("source")),
    }


from collections import defaultdict
from collections.abc import Mapping
import json
import math
from statistics import median

from mindie_sim.observations.resource_services import (
    normalize_resource_service,
    resolve_nominal_rate,
)
from mindie_sim.prediction.service_regime_domains.services import ServiceCurve

from .service_fitting_capture import (
    _capture_float,
    _capture_int,
    capture_image_identity,
    load_native_cohort_validation,
    read_capture_locators,
)
from .service_fitting_capture import (
    bind_capture_native_plan,
    capture_measurement,
    join_capture_measurement_and_plan,
    logical_capture_work,
    parse_capture_abi,
)


ACTUAL_EVIDENCE_SCHEMA = "mindie-sim.actual-resource-evidence/1"


def _capture_record_cohort(row):
    import json as _json
    fields = {}
    for key in ("contract_kind", "contract_match_status", "contract_checks",
                "contract_notes", "family_role", "family_status",
                "archive_event_count", "archive_mean_us", "cohort_window",
                "global_signature_count", "global_signature_mean_us",
                "full_cohort_window_reproduced", "clock_domain", "step_state",
                "locator_status", "locator_notes", "ambiguities"):
        value = row.get(key)
        if key in {"contract_checks", "contract_notes", "cohort_window",
                   "clock_domain", "step_state", "locator_notes",
                   "ambiguities"}:
            try:
                fields[key] = _json.loads(value) if value else None
            except (TypeError, ValueError):
                fields[key] = value
        else:
            fields[key] = value
    return fields


def _capture_locator(row):
    return {
        "csv_line": _capture_int(row.get("example_locator_csv_line")),
        "row_index": _capture_int(row.get("example_locator_row_index")),
        "task_id": row.get("example_locator_task_id"),
        "stream_id": row.get("example_locator_stream_id"),
        "device_id": row.get("example_locator_device_id"),
        "raw_start_us": _capture_float(row.get("raw_start_us")),
        "raw_duration_us": _capture_float(row.get("raw_duration_us")),
        "raw_end_us": _capture_float(row.get("raw_end_us")),
        "raw_op_type": row.get("raw_op_type"),
        "raw_input_shapes": row.get("raw_input_shapes"),
        "raw_output_shapes": row.get("raw_output_shapes"),
        "abi_signature": row.get("abi_signature"),
        "source_path": row.get("source_path"),
    }


def _locator_checks(raw, locator):
    checks = []
    for key, raw_key in (("device_id", "Device_id"),
                         ("task_id", "Task ID"),
                         ("stream_id", "Stream ID")):
        expected = str(locator.get(key) or "")
        actual = str(raw.get(raw_key) or "")
        checks.append({"name": key, "ok": expected == actual,
                       "expected": expected, "actual": actual})
    start = _capture_float(raw.get("Start Time(us)"))
    duration = _capture_float(raw.get("Duration(us)"))
    expected_start = locator.get("raw_start_us")
    expected_duration = locator.get("raw_duration_us")
    checks.append({
        "name": "start_time",
        "ok": (start is not None and expected_start is not None
               and abs(start - expected_start) <= 1e-6),
        "expected": expected_start, "actual": start,
    })
    checks.append({
        "name": "duration",
        "ok": (duration is not None and expected_duration is not None
               and abs(duration - expected_duration) <= 1e-6),
        "expected": expected_duration, "actual": duration,
    })
    abi = parse_capture_abi(raw)["abi_signature"]
    expected_abi = str(locator.get("abi_signature") or "")
    expected_sections = expected_abi.split("|")
    actual_sections = abi.split("|")
    tokens_match = (len(expected_sections) == len(actual_sections)
                    and all(
                        [token for token in expected.split(";") if token]
                        == [token for token in actual.split(";") if token]
                        for expected, actual in zip(expected_sections,
                                                    actual_sections)))
    checks.append({
        "name": "abi_signature", "ok": tokens_match,
        "expected": expected_abi, "actual": abi,
        "slot_compaction": tokens_match and abi != expected_abi,
        "detail": ("present_tensor_tokens_match"
                   + (";raw_field_compacts_empty_optional_slots"
                      if tokens_match and abi != expected_abi else "")),
    })
    return checks


def _read_actual_capture_groups(selected, native_by_cohort, excluded, *, with_sha256):
    groups_by_capture = {}
    for record in selected:
        cohort_id = record["cohort_id"]
        row = native_by_cohort.get(cohort_id)
        if row is None:
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "native_validation_row_missing",
                "source_path": record.get("source_path"),
            })
            continue
        if str(row.get("calibration_relation") or "") != _ROLE_IN_SAMPLE:
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "native_validation_role_mismatch",
                "declared_role": row.get("calibration_relation"),
                "source_path": record.get("source_path"),
            })
            continue
        if str(row.get("case_id") or "") != str(record.get("case_id") or ""):
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "case_id_mismatch",
                "ledger_case": record.get("case_id"),
                "native_case": row.get("case_id"),
                "source_path": record.get("source_path"),
            })
            continue
        if str(row.get("source_path") or "") != str(record.get("source_path") or ""):
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "source_path_mismatch",
                "ledger_source": record.get("source_path"),
                "native_source": row.get("source_path"),
            })
            continue
        locator = _capture_locator(row)
        if not locator["csv_line"]:
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "locator_line_missing",
                "source_path": row.get("source_path"),
            })
            continue
        groups_by_capture.setdefault(locator["source_path"], []).append(
            (record, row, locator))
    capture_files = {}
    raw_by_key = {}
    for source_path, items in groups_by_capture.items():
        lines = [item[2]["csv_line"] for item in items]
        try:
            raw = read_capture_locators(source_path, lines,
                                        with_sha256=with_sha256)
        except ValueError as error:
            for record, row, locator in items:
                excluded.append({
                    "cohort_id": record["cohort_id"],
                    "role": record.get("role"),
                    "reason": "capture_unavailable",
                    "detail": str(error),
                    "source_path": source_path,
                })
            continue
        capture_files[source_path] = {
            "path": raw["path"], "bytes": raw["bytes"],
            "sha256": raw["sha256"], "rows_scanned": raw["rows_scanned"],
            "image": capture_image_identity(source_path),
        }
        for record, row, locator in items:
            raw_row = raw["rows"].get(locator["csv_line"])
            if raw_row is None:
                excluded.append({
                    "cohort_id": record["cohort_id"],
                    "role": record.get("role"),
                    "reason": "locator_row_missing",
                    "source_path": source_path,
                    "csv_line": locator["csv_line"],
                })
                continue
            raw_by_key[record["cohort_id"]] = raw_row
    return capture_files, raw_by_key


def _join_actual_capture_rows(selected, native_by_cohort, capture_files, raw_by_key):
    joined = []
    for record in selected:
        cohort_id = record["cohort_id"]
        row = native_by_cohort.get(cohort_id)
        if row is None or cohort_id not in raw_by_key:
            continue
        raw_row = raw_by_key[cohort_id]
        locator = _capture_locator(row)
        image = capture_files.get(locator["source_path"], {}).get("image") or {}
        contract = _capture_record_cohort(row)
        measurement = capture_measurement(raw_row)
        work = logical_capture_work(raw_row)
        checks = _locator_checks(raw_row, locator)
        locator_verified = all(item["ok"] for item in checks)
        abi_slot_compaction = any(
            item.get("slot_compaction") for item in checks)
        exact = str(row.get("contract_match_status") or "") == "exact"
        plan = bind_capture_native_plan(raw_row,
                                        image_id=image.get("image_id"))
        pairing = join_capture_measurement_and_plan(measurement, plan)
        missing = []
        if not locator_verified:
            missing.append("raw_locator_and_abi_signature_verified")
        if not exact:
            missing.append("exact_geometry_contract")
        if str(row.get("full_cohort_window_reproduced") or "") != "True":
            missing.append("reproduced_full_cohort_window")
        if work.get("logical_work") is None:
            missing.append("logical_physical_work")
        if plan.get("status") != "predicted":
            missing.append("selected_native_program")
            missing.extend(
                "native_plan:" + str(item) for item in plan.get("reasons") or ())
        else:
            missing.append("dispatch_and_tiling_identity")
            missing.append("issued_work_measured")
        missing.append("owner_aggregation_and_active_owner_count")
        missing.append("independent_pipe_measurement")
        eligible = False
        if not locator_verified:
            reason = "raw_locator_or_abi_signature_mismatch"
        elif not exact:
            reason = "geometry_contract_not_exact"
        elif work.get("logical_work") is None:
            reason = "logical_physical_work_not_derivable"
        elif plan.get("status") != "predicted":
            reason = "selected_native_program_not_established"
        elif not plan.get("source_established"):
            missing.append("same_binary_source_identity")
            reason = "binary_source_not_established"
        else:
            reason = "measured_pipe_aggregation_and_dispatch_not_established"
        joined.append({
            "cohort_id": cohort_id,
            "case_id": record.get("case_id"),
            "role": record.get("role"),
            "source_path": locator["source_path"],
            "locator": locator,
            "native_validation": contract,
            "capture_image": image,
            "locator_checks": checks,
            "locator_verified": locator_verified,
            "abi_slot_compaction": abi_slot_compaction,
            "raw_measurement": measurement,
            "logical_work": work,
            "native_plan": plan,
            "measurement_plan_join": pairing,
            "training_eligibility": {
                "eligible": eligible,
                "reason": reason,
                "missing": list(dict.fromkeys(missing)),
            },
            "duplicates": {
                "source": "T03 native_cohort_validation global_signature_count",
                "global_signature_count": _capture_int(
                    row.get("global_signature_count")),
                "archive_event_count": _capture_int(
                    row.get("archive_event_count")),
                "global_signature_mean_us": _capture_float(
                    row.get("global_signature_mean_us")),
                "example_row_is_one_occurrence": True,
            },
        })
    return joined


def _summarize_actual_evidence_groups_and_pairs(joined):
    groups = {}
    for item in joined:
        key = (str(item["capture_image"].get("image_id")),
               str(item["raw_measurement"]["scheduling"].get("kernel_type")),
               str(item["native_validation"].get("contract_match_status")))
        group = groups.setdefault(key, {
            "image_id": key[0], "native_type": key[1],
            "contract_match_status": key[2], "cohorts": [],
        })
        group["cohorts"].append(item["cohort_id"])
    for group in groups.values():
        group.update(count=len(group["cohorts"]), fit_eligible=False, missing=[
            "independent_pipe_work_and_owner_aggregation",
            "same_dispatch_tiling_identity",
            "at_least_two_independent_work_sizes_for_one_resource",
        ])
    pairs = []
    by_type = {}
    for item in joined:
        by_type.setdefault(
            (item["source_path"],
             item["raw_measurement"]["scheduling"].get("kernel_type")),
            []).append(item)
    for (source_path, native_type), items in sorted(
            by_type.items(), key=lambda pair: (str(pair[0][0]), str(pair[0][1]))):
        if len(items) < 2:
            continue
        pair = items[:2]
        first, second = pair
        same_work = (first["logical_work"].get("logical_work")
                     == second["logical_work"].get("logical_work"))
        pairs.append({
            "source_path": source_path, "native_type": native_type,
            "cohort_ids": [item["cohort_id"] for item in pair],
            "csv_lines": [item["locator"]["csv_line"] for item in pair],
            "task_ids": [item["locator"]["task_id"] for item in pair],
            "stream_ids": [item["locator"]["stream_id"] for item in pair],
            "logical_work": [item["logical_work"].get("logical_work") for item in pair],
            "block_num": [item["raw_measurement"].get("block_num") for item in pair],
            "kernel_duration_us": [item["raw_measurement"].get("kernel_duration_us") for item in pair],
            "same_work": same_work,
            "same_owner_condition": False,
            "two_point_fit_eligible": False,
            "reason": ("overlapping_joint_kernel_window; different_work_and_"
                       "owner_conditions; measured_pipe_aggregation_not_declared"),
        })
    return list(groups.values()), pairs


def build_actual_evidence(role_records, native_validation, *,
                          with_sha256=False, max_cohorts=None,
                          role="in_sample"):
    """Join role ledger -> T03 validation -> same-boundary raw PMU rows.

    The function reads only ``in_sample`` captures.  Frozen-transfer and
    not-declared cohorts are registered but their raw captures are not opened.
    Every join check, participation record, exclusion, duplicate counter and
    source hash is returned so the training set is reproducible.
    """
    import hashlib
    if isinstance(native_validation, Mapping):
        native_rows = list(native_validation.get("rows") or ())
        native_meta = {key: native_validation.get(key)
                       for key in ("path", "sha256", "bytes", "count", "counts")}
    else:
        native_rows = list(native_validation)
        native_meta = {"path": None, "sha256": None, "bytes": None,
                       "count": len(native_rows), "counts": {}}
    native_by_cohort = {str(row.get("cohort_id")): row for row in native_rows}
    partition = calibration_role_partitions(role_records)
    selected = list(partition["in_sample"])
    if max_cohorts is not None:
        selected = selected[:int(max_cohorts)]
    registration = {
        "role_counts": dict(partition["counts"]),
        "total": partition["total"],
        **{f"{name}_ids": [record["cohort_id"] for record in partition[name]]
           for name in ("in_sample", "frozen_transfer", "not_declared")},
        "policy": partition["policy"],
        "read_capture_roles": [role],
    }
    excluded = []
    for name, reason in (
        ("frozen_transfer", "role_frozen_transfer_not_training"),
        ("not_declared", "role_not_declared_neither_training_nor_holdout"),
    ):
        excluded.extend({
            "cohort_id": record["cohort_id"], "role": record.get("role"),
            "reason": reason, "source_path": record.get("source_path"),
        } for record in partition[name])
    capture_files, raw_by_key = _read_actual_capture_groups(
        selected, native_by_cohort, excluded, with_sha256=with_sha256)
    joined = _join_actual_capture_rows(selected, native_by_cohort, capture_files, raw_by_key)
    group_rows, pairs = _summarize_actual_evidence_groups_and_pairs(joined)
    counts = {}
    for item in joined:
        key = item["native_validation"].get("contract_match_status")
        counts[key] = counts.get(key, 0) + 1
    reproduced = sum(
        1 for item in joined
        if item["native_validation"].get("full_cohort_window_reproduced")
        in {"True", True})
    training_inputs = []
    for item in joined:
        if item["training_eligibility"]["eligible"]:
            training_inputs.append({
                "cohort_id": item["cohort_id"], "role": item["role"],
                "scope": "independent_resource",
                "resources": [item["native_plan"].get("owner_phases", {}).get(
                    "mte2", {}).get("resource", "mte2")],
                "phase": "load", "work_unit": "bytes", "work": item["native_plan"].get("issued", {}).get("mte2_bytes"),
                "duration_us": item["raw_measurement"].get("kernel_duration_us"),
                "boundary": "device_kernel", "provenance": {"window": "resource_phase"},
            })
    result = {
        "schema": ACTUAL_EVIDENCE_SCHEMA,
        "role_registration": registration,
        "native_validation": native_meta,
        "captures": capture_files,
        "participation": joined,
        "excluded": excluded,
        "counts": {
            "in_sample_joined": len(joined),
            "in_sample_excluded": len(excluded) - len(
                partition["frozen_transfer"]) - len(partition["not_declared"]),
            "contract_match_status": counts,
            "reproduced_full_cohort_window": reproduced,
            "not_reproduced_full_cohort_window": len(joined) - reproduced,
            "abi_slot_compaction": sum(
                1 for item in joined if item.get("abi_slot_compaction")),
        },
        "groups": group_rows,
        "real_pairs": pairs,
        "training_inputs": training_inputs,
        "service": None,
        "fit": {
            "status": "underdetermined",
            "reason": ("no_in_sample_observation_has_an_independent_pipe_"
                       "measurement_with_declared_owner_aggregation_and_"
                       "verified_dispatch"),
            "training_inputs": len(training_inputs),
            "needed_evidence": [
                "selected_native_program_and_same_dispatch_tiling",
                "measured_issued_work_per_owner",
                "declared_owner_aggregation_and_active_owner_count",
                "independent_resource_pipe_measurement",
                "resource_duration_us_for_one_phase",
            ],
        },
        "policy": {
            "read_capture_roles": ["in_sample"],
            "frozen_transfer_capture_read": False,
            "not_declared_capture_read": False,
            "sealed_holdout_read": False,
            "cpu_simulated_duration_in_training": False,
            "target_duration_in_training": False,
            "logical_work_is_issued_work": False,
            "block_num_is_owner_count": False,
            "overlapping_pipe_times_added": False,
        },
    }
    canonical = json.loads(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    result["canonical_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")).hexdigest()
    return result


def fit_actual_training_evidence(role_path, native_validation_path, *,
                                 with_sha256=False, max_cohorts=None):
    """Load the real role/native evidence and return one reproducible report."""
    import hashlib
    from pathlib import Path
    role_path = Path(role_path).expanduser().resolve()
    if not role_path.is_file():
        raise ValueError("role_ledger_source_missing:" + str(role_path))
    records = load_role_ledger(role_path)
    validation = load_native_cohort_validation(native_validation_path)
    evidence = build_actual_evidence(
        records, validation, with_sha256=with_sha256,
        max_cohorts=max_cohorts)
    evidence["role_ledger"] = {
        "path": str(role_path),
        "sha256": _sha256_file(role_path),
        "bytes": role_path.stat().st_size,
        "count": len(records),
    }
    canonical = json.loads(json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), allow_nan=False))
    evidence["canonical_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")).hexdigest()
    if evidence["training_inputs"]:
        fitted = fit_resource_service_from_role_records(
            evidence["training_inputs"], resource="mte2", phase="load",
            work_unit="bytes", scope="independent_resource")
        evidence["fit"] = {
            "status": fitted.get("status"),
            "service": fitted.get("service"),
            "fit": fitted.get("fit"),
            "reason": fitted.get("reason"),
            "training_participation": fitted.get("training_participation"),
        }
    return evidence


__all__ = [
    "ACTUAL_EVIDENCE_SCHEMA",
    "build_actual_evidence",
    "bind_capture_native_plan",
    "capture_image_identity",
    "capture_measurement",
    "fit_actual_training_evidence",
    "fit_resource_service_from_role_records",
    "join_capture_measurement_and_plan",
    "load_native_cohort_validation",
    "load_role_ledger",
    "logical_capture_work",
    "parse_capture_abi",
    "read_capture_locators",
]
