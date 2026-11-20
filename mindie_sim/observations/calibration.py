"""Explicit bindings from profiler observations to primitive/service parameters.

[Role] Reuse retained observations through the current primitive identity,
  import source-measured dual-load matrix service into a normal Case, and load
  one normalized resource/phase profile from built-in or user JSON.
[Invariants] A human-supplied binding declares hardware, implementation, shape
  and split. Filename resemblance and an old eligibility bit never bind a row.
  A dual-load import copies measured pipe rates, never a target latency, and
  leaves owner/work-domain misses to the normal analytical path. A resource
  profile resolves each nominal/effective rate once, keeps startup/response
  separate, and never silently substitutes a nominal rate for an incompatible
  explicit calibration. Joint-phase records never claim internal resource rates.
[Timeouts & Locks] Bounded reads through ProfilerCatalogue; streaming C7
  extraction for the measured-supply fragment; bounded local profile files.
[Side-Effect Policy] Writes only the explicitly writable PrimitiveStore and
  returns new case/rule dictionaries; profile export writes only an explicit path.
"""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

from mindie_sim.prediction.primitives import PrimitiveKey
from .resource_services import (
    RESOURCE_SERVICE_BINDING_SCHEMA,
    RESOURCE_SERVICE_PROFILE_SCHEMA,
    RESOURCE_SERVICE_SCHEMA,
    NormalizedResourceService,
    c7_applicability,
    identity_digest,
    normalize_resource_service,
)

_A5_REEXPORTS = frozenset({
    "A5_ARCH_FAMILY", "A5_HARDWARE_KEY", "A5_SKU",
    "EMBEDDED_A5_CAMPAIGNS", "EMBEDDED_A5_MC2_CAMPAIGNS",
    "A5CommCalibrationBundle", "bind_service_curve", "build_a5_comm_recipe",
    "build_a5_followup_manifest", "collect_native_service", "compute_a5_contention_slowdown",
    "evaluate_a5_mc2_chains", "fit_a5_service_curves", "ingest_a5_campaign_data",
    "resolve_expansion_regime", "to_service_regime",
})


def __getattr__(name: str):
    if name in _A5_REEXPORTS:
        from . import communication as _comm
        return getattr(_comm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

PROFILER_SUPPLY_SCHEMA = "mindie-profiler-measured-supply/1"


__all__ = [
    "A5_ARCH_FAMILY", "A5_HARDWARE_KEY", "A5_SKU",
    "EMBEDDED_A5_CAMPAIGNS", "EMBEDDED_A5_MC2_CAMPAIGNS",
    "A5CommCalibrationBundle", "bind_service_curve", "build_a5_comm_recipe",
    "build_a5_followup_manifest", "collect_native_service", "compute_a5_contention_slowdown",
    "evaluate_a5_mc2_chains", "fit_a5_service_curves", "import_profiler",
    "import_profiler_supply", "apply_profiler_supply",
    "ingest_a5_campaign_data", "resolve_expansion_regime", "to_service_regime",
    "apply_resource_profile", "bind_operator_resource_service",
    "export_resource_profile", "load_resource_profile",
    "attach_operator_resource_service_binding", "bind_a5_local_service",
    "bind_a5_local_service_operator",
    "resource_service_binding", "select_resource_service_candidate",
    "service_conditions_match",
]


def import_profiler(catalogue, store, bindings, *, max_samples=1000):
    from .store import ProfilerCatalogue

    imported = 0
    with ProfilerCatalogue(catalogue) as source:
        for binding in bindings:
            key = PrimitiveKey.from_mapping(binding["key"])
            rows = list(source.samples(binding["variant"], limit=max_samples + 1))
            if len(rows) > max_samples:
                raise ValueError("profiler_binding_exceeds_sample_budget:select_a_capture_subset")
            if not rows:
                raise ValueError("profiler_binding_has_no_raw_events")
            for row in rows:
                identity = row["source"] + ("::" + row["member"] if row["member"] else "")
                store.record(
                    key, duration_ms=row["duration_ms"],
                    run_id=binding["run_id"] + ":" + identity, repeat_index=row["row"],
                    source=f"{identity}#row={row['row']}", split=binding["split"])
                imported += 1
    return {"imported_samples": imported, "inventory": store.inventory()}


def _measured_supply_payload(source):
    if isinstance(source, dict):
        payload = source
        if payload.get("schema_version") == PROFILER_SUPPLY_SCHEMA:
            return payload
        enrichment = payload.get("enrichment") if isinstance(payload.get("enrichment"), dict) else {}
        pmu = enrichment.get("pmu_observations") if isinstance(enrichment.get("pmu_observations"), dict) else {}
        supply = pmu.get("measured_supply")
        if isinstance(supply, dict):
            return supply
        raise ValueError("profiler_report_has_no_measured_supply")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError("profiler_supply_source_missing:" + str(path))
    if path.stat().st_size <= 50 * 1024 ** 2:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if data.get("schema_version") == PROFILER_SUPPLY_SCHEMA:
                return data
            enrichment = data.get("enrichment") if isinstance(data.get("enrichment"), dict) else {}
            pmu = enrichment.get("pmu_observations") if isinstance(enrichment.get("pmu_observations"), dict) else {}
            if isinstance(pmu.get("measured_supply"), dict):
                return pmu["measured_supply"]
    import ijson

    with path.open("rb") as stream:
        for item in ijson.items(
                stream, "enrichment.pmu_observations.measured_supply",
                use_float=True):
            if isinstance(item, dict):
                return item
    # A compact export may contain only the measured-supply object itself.
    with path.open("rb") as stream:
        for item in ijson.items(stream, "measured_supply", use_float=True):
            if isinstance(item, dict):
                return item
    raise ValueError("profiler_report_has_no_measured_supply")


def _profiler_source_description(source):
    if isinstance(source, Mapping):
        return {"kind": "in_memory",
                "schema_version": source.get("schema_version")}
    path = Path(source).expanduser().resolve()
    return {"kind": "file", "path": str(path), "bytes": path.stat().st_size}


def import_profiler_supply(source, *, recipe=None, kernel=None, hardware=None,
                           owners=None, required=True, require_hardware=True):
    """Return the normal service-regime rule from one measured dual-load pair.

    ``source`` is a C7 report path/mapping or an already-extracted
    ``mindie-profiler-measured-supply/1`` mapping.  The function never reads a
    target duration as a service cost; it copies only the measured pipe rates,
    active-owner counts and declared work domains.
    """
    if require_hardware and not hardware:
        raise ValueError("profiler_supply_requires_explicit_hardware_identity")
    supply = _measured_supply_payload(source)
    resources = deepcopy(supply.get("resource_services") or ())
    if supply.get("status") != "measured_pair" or not isinstance(supply.get("pair"), dict):
        if resources:
            return {"service_regimes": [], "supply": None,
                    "status": supply.get("status"),
                    "source": _profiler_source_description(source),
                    "source_identity": dict(supply.get("source_identity") or {}),
                    "resource_services": resources}
        if required:
            reasons = [item.get("code") for item in supply.get("limitations", ())
                       if isinstance(item, dict)]
            raise ValueError("profiler_measured_supply_not_available:"
                             + ",".join(str(code) for code in reasons if code))
        return {"service_regimes": [], "supply": supply, "status": supply.get("status"),
                "source": _profiler_source_description(source),
                "resource_services": []}
    pair = deepcopy(supply["pair"])
    rule = pair.get("service_regime")
    if not isinstance(rule, dict):
        raise ValueError("profiler_measured_supply_pair_requires_service_regime")
    match = dict(rule.get("match") or {})
    match.pop("kernel", None)
    match.pop("recipe", None)
    if recipe:
        match["recipe"] = str(recipe)
    elif kernel:
        match["kernel"] = str(kernel)
    elif not match.get("kernel"):
        match["kernel"] = str(pair.get("identity", {}).get("native_type") or "")
    if hardware:
        match["hardware"] = str(hardware)
    rule["match"] = match
    # Validate with the same public response parser the runtime uses.
    from mindie_sim.prediction.response_functions import ResponseFunction

    coordinates = rule.get("coordinates") or {}
    if not isinstance(coordinates, dict) or not coordinates:
        raise ValueError("profiler_supply_regime_requires_coordinates")
    if owners is not None:
        if isinstance(owners, bool) or not isinstance(owners, int) or owners <= 0:
            raise ValueError("profiler_supply_owners_must_be_positive_integer")
        owner_coordinate = coordinates.get("owners")
        owner_coordinate = dict(owner_coordinate) if isinstance(owner_coordinate, dict) else {
            "path": "metadata.service_owners"}
        owner_coordinate["default"] = owners
        coordinates = {**coordinates, "owners": owner_coordinate}
        rule["coordinates"] = coordinates
        rule["provenance"] = {
            **dict(rule.get("provenance") or {}),
            "declared_owner_default": owners,
            "declared_owner_basis": "caller_selected_implementation_active_owners",
        }
    ResponseFunction.from_mapping(rule.get("response"), coordinates)
    return {
        "service_regimes": [rule],
        "supply": pair,
        "status": "measured_pair",
        "source": _profiler_source_description(source),
        "source_identity": dict(supply.get("source_identity") or {}),
        "resource_services": resources,
        "source_schema": supply.get("schema_version"),
        "scope": supply.get("scope"),
    }


def _normalize_profiler_resource_services(resources, *, hardware, source,
                                          source_identity):
    normalized = []
    for raw in resources or ():
        if not isinstance(raw, Mapping):
            raise ValueError("profiler_resource_service_requires_mapping")
        rate = raw.get("rate_bytes_per_s")
        if rate is None:
            continue
        owners = raw.get("owners")
        if isinstance(owners, bool) or not isinstance(owners, int) or owners < 1:
            raise ValueError("profiler_resource_service_requires_positive_owners")
        resource = str(raw.get("resource") or "").strip()
        if not resource:
            raise ValueError("profiler_resource_service_requires_resource")
        declared_hardware = str(raw.get("hardware") or hardware or "").strip()
        if not declared_hardware:
            raise ValueError("profiler_resource_service_requires_hardware")
        hardware_gap = bool(
            hardware and raw.get("hardware")
            and str(raw["hardware"]) != str(hardware))
        c7_record = ({**dict(raw), "hardware": declared_hardware}
                     if not raw.get("hardware") else dict(raw))
        applicability = c7_applicability(c7_record)
        conditions = dict(applicability["conditions"])
        optional_conditions = dict(applicability["optional_conditions"])
        geometry = dict(raw.get("geometry_binding") or {})
        source_sha = str(raw.get("source_sha256") or "")
        identifier = (f"profiler:{resource}:"
                      f"{raw.get('shape_class') or 'default'}:"
                      f"{source_sha[:12] or 'unhashed'}")
        source_scope = str(raw.get("scope") or "")
        effective_rate = float(rate)
        rate_conversion = "per_owner_rate_preserved_critical_owner_work"
        model = normalize_resource_service({
            "id": identifier,
            "scope": "independent_resource",
            "resources": [resource],
            "phase": str(raw.get("phase") or "prefill"),
            "work_unit": "bytes",
            "boundary": "resource_phase",
            "startup_us": 0,
            "response_us": 0,
            "active_units": owners,
            "rate_scope": "per_owner",
            "supply": {"kind": "absolute_rate", "rate": effective_rate},
            "conditions": conditions,
            "optional_conditions": optional_conditions,
            "identification": {
                "status": "identified",
                "resource_identified": True,
                "rate_source": "measured_pmu",
                "source_identity": deepcopy(
                    raw.get("source_identity") or source_identity or {}),
                **({"train_work": [int(raw["measured_critical_bytes"])]}
                   if raw.get("measured_critical_bytes") else {}),
            },
            "provenance": {
                "kind": "profiler_resource_service",
                "source": deepcopy(source),
                "source_sha256": source_sha,
                "source_rate_bytes_per_s": float(rate),
                "source_rate_scope": source_scope,
                "rate_conversion": rate_conversion,
                "source_measured_critical_bytes": raw.get("measured_critical_bytes"),
                "source_identity": deepcopy(
                    raw.get("source_identity") or source_identity or {}),
                "geometry_binding": geometry,
                "c7_work_domain": deepcopy(applicability["work_domain"]),
                "not_elapsed_rate": bool(
                    dict(raw.get("provenance") or {}).get("not_elapsed_rate")),
                "selection_policy": "c7_applicability_match",
                "requested_hardware": str(hardware) if hardware else None,
                "applicability_gap": (
                    "hardware_mismatch" if hardware_gap else None),
                "production_prediction": True,
            },
        })
        normalized.append(model)
    return tuple(normalized)


def _legacy_diagnostic_comparison(rules, normalized):
    rows = []
    normalized_ids = [item.service_id for item in normalized]
    for rule in rules:
        response = dict(rule.get("response") or {})
        rows.append({
            "rule_id": rule.get("id"),
            "response_kind": response.get("kind", "affine"),
            "legacy_fixed_us": response.get("fixed_us"),
            "legacy_rate_terms": deepcopy(response.get("terms") or response.get("loads") or ()),
            "normalized_service_ids": normalized_ids,
            "relation": "legacy_diagnostic_only_not_used_for_production",
            "difference_is_diagnostic": True,
        })
    return {
        "rules": rows,
        "normalized_parameters": [{
            "service_id": item.service_id,
            "resource": item.resource,
            "work_unit": item.work_unit,
            "rate_scope": item.rate_scope,
            "active_units": item.active_units,
            "effective_rate": item.effective_rate,
            "fixed_us": item.fixed_us,
        } for item in normalized],
        "production_prediction": False,
    }


def apply_profiler_supply(case, source, *, recipe=None, kernel=None,
                          hardware=None, owners=None, required=True,
                          require_hardware=True, legacy_diagnostics=False):
    """Attach normalized C7 resource services; legacy rules stay diagnostic.

    Ordinary prediction never receives the imported ``service_regimes``.
    Those rules are preserved under ``runtime.metadata`` with their source and
    ``production_prediction=false``.  Only an explicit
    ``legacy_diagnostics=True`` replay materializes them into
    ``runtime.kernels.service_regimes``.
    """
    if not isinstance(case, dict):
        raise ValueError("profiler_supply_case_requires_mapping")
    imported = import_profiler_supply(
        source, recipe=recipe, kernel=kernel, hardware=hardware,
        owners=owners, required=required, require_hardware=require_hardware)
    if not imported["service_regimes"] and not imported.get("resource_services"):
        return deepcopy(case)
    result = deepcopy(case)
    runtime = result.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("profiler_supply_case_runtime_requires_mapping")
    metadata = runtime.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("profiler_supply_case_runtime_metadata_requires_mapping")
    source_desc = deepcopy(imported.get("source") or {})
    source_identity = deepcopy(imported.get("source_identity") or {})
    pair = imported.get("supply") or {}
    normalized = _normalize_profiler_resource_services(
        imported.get("resource_services") or (),
        hardware=hardware, source=source_desc, source_identity=source_identity)
    legacy_rules = deepcopy(imported.get("service_regimes") or [])
    for rule in legacy_rules:
        rule["production_prediction"] = False
        rule["diagnostic_only"] = True
        rule["diagnostic_source"] = deepcopy(source_desc)
        rule["source_identity"] = deepcopy(source_identity)
    execution = (dict(legacy_rules[0].get("execution") or {})
                 if legacy_rules else {})
    if normalized:
        profile = {
            "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
            # Top-level hardware is the source/profile identity, not the
            # caller's requested placement.  A mismatched placement is an
            # explicit applicability gap checked at selection time.
            "hardware": (str(normalized[0].conditions.get("hardware") or "") or None),
            "services": [item.to_dict() for item in normalized],
            "source": source_desc,
            "provenance": {
                "kind": "profiler_measured_supply",
                "source_identity": source_identity,
                "production_prediction": True,
            },
        }
        existing = metadata.get("resource_service_profile")
        if existing is None:
            metadata["resource_service_profile"] = profile
        else:
            if not isinstance(existing, dict):
                raise ValueError("profiler_supply_existing_profile_requires_mapping")
            merged = deepcopy(existing)
            services = list(merged.get("services") or ())
            have = {item.get("id") for item in services if isinstance(item, Mapping)}
            services.extend(item for item in profile["services"]
                            if item.get("id") not in have)
            merged["services"] = services
            metadata["resource_service_profile"] = merged
    metadata["profiler_measured_supply"] = {
        "schema_version": imported.get("source_schema"),
        "scope": imported.get("scope"),
        "source": source_desc,
        "identity": deepcopy(pair.get("identity")),
        "service_ratio": deepcopy(pair.get("service_ratio")),
        "declared_owner_default": owners,
        "source_identity": source_identity,
        "resource_services": deepcopy(imported.get("resource_services") or ()),
        "normalized_resource_services": [item.to_dict() for item in normalized],
        "legacy_service_regimes": legacy_rules,
        "legacy_production_prediction": False,
        "legacy_diagnostics_replay": bool(legacy_diagnostics),
        "diagnostic_comparison": _legacy_diagnostic_comparison(
            legacy_rules, normalized),
        "applicability_gaps": [
            {"service_id": item.service_id,
             "gap": item.provenance.get("applicability_gap")}
            for item in normalized
            if item.provenance.get("applicability_gap")],
        "mac_execution": execution or None,
        "mac_execution_production_prediction": False,
    }
    if legacy_diagnostics:
        kernels = runtime.setdefault("kernels", {})
        if not isinstance(kernels, dict):
            raise ValueError("profiler_supply_case_kernels_requires_mapping")
        regimes = list(kernels.get("service_regimes") or ())
        # The runtime parser accepts the original rule schema; our
        # production_prediction/diagnostic markers stay only in metadata.
        regimes.extend(deepcopy(imported.get("service_regimes") or ()))
        kernels["service_regimes"] = regimes
        metadata["legacy_diagnostics"] = True
    return result

_RESOURCE_PROFILE_MAX_BYTES = 50 * 1024 ** 2


def _resource_profile_payload(source, *, source_kind=None):
    if isinstance(source, Mapping):
        payload = deepcopy(dict(source))
        origin = {
            "kind": source_kind or "in_memory",
            "label": str(payload.get("name") or "resource_service_profile"),
        }
        return payload, origin
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError("resource_profile_source_missing:" + str(path))
    size = path.stat().st_size
    if size > _RESOURCE_PROFILE_MAX_BYTES:
        raise ValueError("resource_profile_source_exceeds_50_mib")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("resource_profile_source_requires_utf8_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("resource_profile_source_requires_object")
    return payload, {
        "kind": source_kind or "file",
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": size,
    }


def _profile_origin(source):
    """Return a preserved loaded-profile origin, never a caller request."""
    if not isinstance(source, Mapping):
        return None
    if source.get("schema") != RESOURCE_SERVICE_PROFILE_SCHEMA:
        return None
    origin = source.get("source")
    if not isinstance(origin, Mapping):
        return None
    kind = origin.get("kind")
    if kind not in {"file", "builtin", "in_memory", "user"}:
        return None
    if kind == "file" and not (origin.get("path") and origin.get("sha256")):
        return None
    return deepcopy(dict(origin))


def load_resource_profile(source, *, nominal_rates=None, hardware=None,
                          source_kind=None):
    """Load one built-in or user resource-service profile through one path.

    ``source`` is a mapping (built-in profile) or a JSON file path (user
    profile).  Exactly the same normalization runs for both.  Every declared
    multiplier resolves against an explicit nominal rate; an incompatible or
    incomplete profile raises instead of falling back to a nominal service.
    """
    preserved_origin = (_profile_origin(source)
                        if source_kind is None else None)
    raw, origin = _resource_profile_payload(source, source_kind=source_kind)
    if preserved_origin is not None:
        origin = preserved_origin
    schema = raw.get("schema")
    if schema == RESOURCE_SERVICE_PROFILE_SCHEMA:
        services = raw.get("services")
        profile_hardware = raw.get("hardware")
        profile_scope = raw.get("scope") or "resource_and_phase_services"
        profile_provenance = raw.get("provenance") or {}
        profile_source = raw.get("source")
    elif schema == RESOURCE_SERVICE_SCHEMA:
        services = [raw]
        profile_hardware = None
        profile_scope = "single_resource_service"
        profile_provenance = raw.get("provenance") or {}
        profile_source = None
    else:
        raise ValueError("resource_profile_schema_unknown")
    if not isinstance(services, (list, tuple)) or not services:
        raise ValueError("resource_profile_requires_at_least_one_service")
    if not isinstance(profile_provenance, Mapping):
        raise ValueError("resource_profile_provenance_requires_mapping")
    try:
        profile_provenance = json.loads(json.dumps(
            dict(profile_provenance), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("resource_profile_provenance_requires_json_values") from exc
    if hardware and profile_hardware and str(profile_hardware) != str(hardware):
        raise ValueError("resource_profile_hardware_mismatch")
    effective_hardware = str(hardware or profile_hardware or "") or None
    normalized = []
    identifiers = set()
    for index, service in enumerate(services):
        if not isinstance(service, Mapping):
            raise ValueError(f"resource_profile_service_{index}_requires_mapping")
        item = deepcopy(dict(service))
        conditions = item.get("conditions") or {}
        if not isinstance(conditions, Mapping):
            raise ValueError("resource_profile_service_conditions_requires_mapping")
        conditions = dict(conditions)
        if effective_hardware and not conditions.get("hardware"):
            conditions["hardware"] = effective_hardware
        item["conditions"] = conditions
        provenance = item.get("provenance") or {}
        if not isinstance(provenance, Mapping):
            raise ValueError("resource_profile_service_provenance_requires_mapping")
        provenance = dict(provenance)
        provenance.setdefault("profile_source", origin)
        item["provenance"] = provenance
        model = normalize_resource_service(
            item, nominal_rates=nominal_rates, hardware=effective_hardware)
        if model.service_id in identifiers:
            raise ValueError("resource_profile_requires_distinct_service_ids")
        identifiers.add(model.service_id)
        normalized.append(model)
    formulas = {model.service_id: model.formula for model in normalized}
    return {
        "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
        "source": origin,
        "profile_source": deepcopy(profile_source),
        "hardware": effective_hardware,
        "scope": profile_scope,
        "provenance": deepcopy(profile_provenance),
        "services": [model.to_dict() for model in normalized],
        "formulas": formulas,
        "service_ids": [model.service_id for model in normalized],
        "normalization": {
            "applied_once": True,
            "service_count": len(normalized),
            "nominal_rates_supplied": nominal_rates is not None,
        },
    }


def export_resource_profile(profile, *, path=None):
    """Export normalized parameters and formulas; write only when asked."""
    if not isinstance(profile, Mapping):
        raise ValueError("resource_profile_export_requires_mapping")
    schema = profile.get("schema")
    if schema == RESOURCE_SERVICE_PROFILE_SCHEMA:
        payload = deepcopy(dict(profile))
        services = payload.get("services")
        if not isinstance(services, (list, tuple)) or not services:
            raise ValueError("resource_profile_export_requires_services")
        models = [normalize_resource_service(service) for service in services]
        if len({model.service_id for model in models}) != len(models):
            raise ValueError("resource_profile_requires_distinct_service_ids")
        payload["services"] = [model.to_dict() for model in models]
        payload["formulas"] = {model.service_id: model.formula for model in models}
        payload["service_ids"] = [model.service_id for model in models]
        payload["normalization"] = {
            **dict(payload.get("normalization") or {}),
            "applied_once": True,
            "service_count": len(models),
        }
    elif schema == RESOURCE_SERVICE_SCHEMA:
        model = normalize_resource_service(profile)
        payload = {
            "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
            "source": {"kind": "in_memory", "label": model.service_id},
            "services": [model.to_dict()],
            "formulas": {model.service_id: model.formula},
            "service_ids": [model.service_id],
            "normalization": {"applied_once": True, "service_count": 1},
        }
    else:
        raise ValueError("resource_profile_export_requires_loaded_or_service_schema")
    try:
        text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("resource_profile_export_requires_json_values") from exc
    if path is None:
        return payload
    target = Path(path).expanduser().resolve()
    data = (text + "\n").encode("utf-8")
    target.write_bytes(data)
    return {
        "path": str(target),
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "profile": payload,
    }


def apply_resource_profile(case, source, *, nominal_rates=None, hardware=None):
    """Attach a normalized profile to a new Case metadata handoff.

    In-memory and file profiles use the identical load/normalization path, so
    top-level hardware propagation and service-condition conflicts behave the
    same.  An already-loaded profile keeps its original source identity.  The
    function does not rewrite recipes, service regimes or hardware rates; a
    later consumer owns how a service enters its program.
    """
    if not isinstance(case, Mapping):
        raise ValueError("resource_profile_case_requires_mapping")
    profile = load_resource_profile(
        source, nominal_rates=nominal_rates, hardware=hardware)
    result = deepcopy(dict(case))
    runtime = result.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("resource_profile_case_runtime_requires_mapping")
    metadata = runtime.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("resource_profile_case_runtime_metadata_requires_mapping")
    metadata["resource_service_profile"] = deepcopy(profile)
    metadata["resource_service_profile_source"] = deepcopy(profile.get("source"))
    metadata["resource_service_profile_hardware"] = profile.get("hardware")
    return result


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
