"""Canonical pre-use execution-protocol identity for operator curve artifacts.

[Role]
  Own the single canonical protocol view used by raw curve grouping, normalized
  query context, artifact export and public forecast query.  Preserve declared
  execution facts (mode, per-block invocation counts, capture/host/replay
  counts, output lifetime/pool/output policy, timing scope and timer settings)
  so eager and graph protocols cannot be conflated.

[Invariants]
  * Outcome/provenance fields (address counts, run IDs, timing values,
    capture outcome, observed API/backend, duplicates) never define
    applicability.
  * Unknown/missing physical protocol fields are not defaulted: strict equality
    of the canonical protocol is required by consumers.  A legacy artifact with
    no `mode` cannot match a graph-mode query.
  * This module does not divide block intervals into per-invocation service,
    infer a pure device model, or alter declared host/idle/lifetime semantics.

[Timeouts & Locks]
  Pure bounded dict/JSON normalization; no I/O or locks.

[Side-Effect Policy]
  No side effects.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

# Fields that describe observations, outcomes or collection provenance rather
# than the pre-use execution contract.  Unknown fields are preserved.
_NON_PHYSICAL_KEYS = {
    "run_id", "run_ids", "measurement_id", "request_id", "case_id",
    "condition_id", "split", "campaign_id",
    "capture_unique_address_count", "eager_unique_address_count",
    "unique_address_count", "capture_outcome",
    "duration_ms", "duration", "median_ms", "mean_ms", "min_ms", "max_ms",
    "raw_samples_ms", "trial_medians_ms", "trials_data",
    "observed", "observed_api", "observed_backend", "observed_context",
    "observed_at", "profiler", "start", "end", "timestamp",
    "not_a3_measured", "seed", "execution_kind", "worker_recipe",
    "requested_backend", "backend", "source_class",
    "requested_accumulation_dtype",
}


def _is_non_physical(key: str) -> bool:
    if key in _NON_PHYSICAL_KEYS:
        return True
    if key.startswith("observed") or key.startswith("profiler"):
        return True
    if "address" in key:
        return True
    if key.endswith("_ms") or "duration" in key or "elapsed" in key:
        return True
    return False


def _clean(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _clean(item)
            for key, item in value.items()
            if not _is_non_physical(str(key))
        }
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    return value


def canonical_protocol(protocol: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the physical pre-use protocol view (outcomes removed)."""
    if not protocol:
        return {}
    return _clean(dict(protocol))


def source_derived_protocol(production: Mapping[str, Any],
                            request: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize one source measurement protocol for DB and curve export.

    Explicit source facts win.  ``timing_scope`` may come from the surrounding
    production record.  A single raw-repeat list is one independent trial only
    when no explicit trial count/records exist.  Absent cache/mode fields stay
    absent; no placeholder is invented.
    """
    production = dict(production or {})
    request = dict(request or {})
    source = dict(production.get("protocol") or {})
    protocol = dict(source)
    timing_scope = (production.get("timing_scope")
                    or source.get("timing_scope"))
    if timing_scope not in (None, ""):
        protocol["timing_scope"] = timing_scope
    raw = production.get("raw_samples_ms") or ()
    if "warmup" not in protocol and "warmup" in request:
        protocol["warmup"] = request.get("warmup")
    if "iterations" not in protocol:
        if source.get("iterations") is not None:
            protocol["iterations"] = source.get("iterations")
        elif raw:
            protocol["iterations"] = len(raw)
        elif request.get("iterations") is not None:
            protocol["iterations"] = request.get("iterations")
    explicit_trials = None
    for source_value in (source.get("trials"), production.get("trials"),
                         request.get("trials")):
        if isinstance(source_value, (list, tuple)):
            explicit_trials = len(source_value)
        elif isinstance(source_value, int) and not isinstance(source_value, bool) and source_value > 0:
            explicit_trials = source_value
        if explicit_trials is not None:
            break
    protocol["trials"] = explicit_trials if explicit_trials is not None else (
        1 if raw else protocol.get("trials"))
    if protocol.get("trials") is None:
        protocol.pop("trials", None)
    if "preallocated" not in protocol and source.get("preallocated") is not None:
        protocol["preallocated"] = source.get("preallocated")
    if "measurement_scope" not in protocol and source.get("measurement_scope") not in (None, ""):
        protocol["measurement_scope"] = source.get("measurement_scope")
    return protocol


def protocol_key(protocol: Mapping[str, Any] | None) -> dict[str, Any]:
    """Alias used by artifact/query paths; same canonical view everywhere."""
    return canonical_protocol(protocol)


def protocol_signature(protocol: Mapping[str, Any] | None,
                       observed: Mapping[str, Any] | None = None) -> str:
    """Raw-analysis grouping signature: physical protocol + observed execution.

    Observed API/backend is retained only as execution evidence for separating
    analysis groups; it is not part of the pre-use artifact/query protocol key.
    """
    payload: dict[str, Any] = {"protocol": canonical_protocol(protocol)}
    evidence = {
        str(key): observed.get(key)
        for key in ("observed_api", "observed_backend", "execution_kind")
        if observed and observed.get(key) is not None
    }
    if evidence:
        payload["observed_execution"] = evidence
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def protocol_equal(left: Mapping[str, Any] | None,
                   right: Mapping[str, Any] | None) -> bool:
    """Strict canonical equality; absence never wildcards a known field."""
    return canonical_protocol(left) == canonical_protocol(right)


__all__ = [
    "canonical_protocol", "protocol_key",
    "protocol_signature", "protocol_equal",
]
