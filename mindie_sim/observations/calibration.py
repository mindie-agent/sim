"""Explicit bindings from profiler observations to primitive/service parameters.

[Role] Reuse retained observations through the current primitive identity and
  import source-measured dual-load matrix service into a normal Case.
[Invariants] A human-supplied binding declares hardware, implementation, shape
  and split. Filename resemblance and an old eligibility bit never bind a row.
  A dual-load import copies measured pipe rates, never a target latency, and
  leaves owner/work-domain misses to the normal analytical path.
[Timeouts & Locks] Bounded reads through ProfilerCatalogue; streaming C7
  extraction for the measured-supply fragment; transactional store.
[Side-Effect Policy] Writes only the explicitly writable PrimitiveStore and
  returns new case/rule dictionaries.
"""
from copy import deepcopy
import json
from pathlib import Path

from mindie_sim.prediction.primitives import PrimitiveKey
from .a5_comm_calibration import (
    A5_ARCH_FAMILY, A5_HARDWARE_KEY, A5_SKU,
    EMBEDDED_A5_CAMPAIGNS, EMBEDDED_A5_MC2_CAMPAIGNS,
    A5CommCalibrationBundle, bind_service_curve, build_a5_comm_recipe,
    build_a5_followup_manifest, collect_native_service, compute_a5_contention_slowdown,
    evaluate_a5_mc2_chains, fit_a5_service_curves, ingest_a5_campaign_data,
    resolve_expansion_regime, to_service_regime,
)

PROFILER_SUPPLY_SCHEMA = "mindie-profiler-measured-supply/1"


__all__ = [
    "A5_ARCH_FAMILY", "A5_HARDWARE_KEY", "A5_SKU",
    "EMBEDDED_A5_CAMPAIGNS", "EMBEDDED_A5_MC2_CAMPAIGNS",
    "A5CommCalibrationBundle", "bind_service_curve", "build_a5_comm_recipe",
    "build_a5_followup_manifest", "collect_native_service", "compute_a5_contention_slowdown",
    "evaluate_a5_mc2_chains", "fit_a5_service_curves", "import_profiler",
    "import_profiler_supply", "apply_profiler_supply",
    "ingest_a5_campaign_data", "resolve_expansion_regime", "to_service_regime",
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
                    "source_identity": dict(supply.get("source_identity") or {}),
                    "resource_services": resources}
        if required:
            reasons = [item.get("code") for item in supply.get("limitations", ())
                       if isinstance(item, dict)]
            raise ValueError("profiler_measured_supply_not_available:"
                             + ",".join(str(code) for code in reasons if code))
        return {"service_regimes": [], "supply": supply, "status": supply.get("status"),
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
        "source_identity": dict(supply.get("source_identity") or {}),
        "resource_services": resources,
        "source_schema": supply.get("schema_version"),
        "scope": supply.get("scope"),
    }


def apply_profiler_supply(case, source, *, recipe=None, kernel=None,
                          hardware=None, owners=None, required=True,
                          require_hardware=True):
    """Return a new Case mapping with the measured dual-load rule attached."""
    if not isinstance(case, dict):
        raise ValueError("profiler_supply_case_requires_mapping")
    imported = import_profiler_supply(
        source, recipe=recipe, kernel=kernel, hardware=hardware,
        owners=owners, required=required, require_hardware=require_hardware)
    if not imported["service_regimes"] and not imported.get("resource_services"):
        return deepcopy(case)
    result = deepcopy(case)
    runtime = result.setdefault("runtime", {})
    kernels = runtime.setdefault("kernels", {})
    regimes = list(kernels.get("service_regimes") or ())
    regimes.extend(imported["service_regimes"])
    kernels["service_regimes"] = regimes
    # A certified C7 can be resource-only (zero measured dual-load MAC rules,
    # one real resource_services record).  That valid binding must reach the
    # normal case metadata without inventing a MAC rule.
    rule = (imported["service_regimes"][0]
            if imported["service_regimes"] else None)
    execution = dict(rule.get("execution") or {}) if rule is not None else {}
    metadata = runtime.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("profiler_supply_case_runtime_metadata_requires_mapping")
    pair = imported.get("supply") or {}
    metadata["profiler_measured_supply"] = {
        "schema_version": imported.get("source_schema"),
        "scope": imported.get("scope"),
        "identity": deepcopy(pair.get("identity")),
        "service_ratio": deepcopy(pair.get("service_ratio")),
        "declared_owner_default": owners,
        "source_identity": deepcopy(imported.get("source_identity") or {}),
        "resource_services": deepcopy(imported.get("resource_services") or ()),
        "mac_execution": execution or None,
    }
    return result
