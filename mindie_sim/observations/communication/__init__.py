"""Communication observation ingestion, calibration, binding, and local service API.

[Role] Expose the neutral ``mindie_sim.observations.communication`` facade for
  golden summaries, campaign ingestion, service-curve fitting, engine binding,
  normalized stage adaptation, and native local service validation.
[Invariants] Importing this package performs no calibration, file I/O, or device
  work; submodules load lazily on first attribute access and preserve identity
  across re-exports.
[Timeouts & Locks] No waits, locks, or subprocesses.
[Side-Effect Policy] Lazy module attribute lookup only; no I/O on import.
"""
from __future__ import annotations

_EXPORTS = {
    # golden
    "A5_ARCH_FAMILY": "golden",
    "A5_HARDWARE_KEY": "golden",
    "A5_NATIVE_GROUP2": "golden",
    "A5_SKU": "golden",
    "EMBEDDED_A5_CAMPAIGNS": "golden",
    "EMBEDDED_A5_MC2_CAMPAIGNS": "golden",
    "a5_native_group2_curve_ms": "golden",
    # ingest / normalization
    "ingest_a5_campaign_data": "ingest",
    "resolve_expansion_regime": "normalization",
    # binding / normalized / recipes
    "NormalizedServiceStage": "normalized",
    "bind_service_curve": "binding",
    "build_a5_comm_recipe": "recipes",
    "build_normalized_service_recipe": "normalized",
    "build_normalized_service_stage": "normalized",
    "normalized_service_duration_ms": "normalized",
    "service_curve_from_normalized": "normalized",
    "to_service_regime": "binding",
    # calibration
    "A5CommCalibrationBundle": "calibration",
    "fit_a5_service_curves": "calibration",
    # analysis
    "build_a5_followup_manifest": "analysis",
    "compute_a5_contention_slowdown": "analysis",
    "evaluate_a5_mc2_chains": "analysis",
    # local service / local domain
    "A5_COMM_LOCAL_SERVICE_SCHEMA": "local_service",
    "A5_NATIVE_CHILD_EVIDENCE_SCHEMA": "local_service",
    "NATIVE_A5_SERVICE_SOURCE": "local_service",
    "a5_local_service_applicability_key": "local_domain",
    "a5_local_service_domain_key": "local_domain",
    "a5_local_service_lineage_key": "local_domain",
    "a5_local_service_prediction": "local_service",
    "a5_local_service_supply_change": "local_service",
    "collect_native_service": "local_service",
    "fit_a5_native_local_service": "local_service",
    "native_child_stage_evidence": "local_service",
    "require_a5_local_service_applicability": "local_domain",
    "require_a5_local_service_domain": "local_domain",
}

__all__ = sorted(_EXPORTS)


def _load_submodule(submodule: str):
    if submodule == "golden":
        from . import golden as module
    elif submodule == "ingest":
        from . import ingest as module
    elif submodule == "normalization":
        from . import normalization as module
    elif submodule == "normalized":
        from . import normalized as module
    elif submodule == "binding":
        from . import binding as module
    elif submodule == "recipes":
        from . import recipes as module
    elif submodule == "calibration":
        from . import calibration as module
    elif submodule == "analysis":
        from . import analysis as module
    elif submodule == "local_service":
        from . import local_service as module
    elif submodule == "local_domain":
        from . import local_domain as module
    else:  # pragma: no cover - guarded by _EXPORTS
        raise AttributeError(f"unknown communication submodule: {submodule!r}")
    return module


def __getattr__(name: str):
    submodule = _EXPORTS.get(name)
    if submodule is not None:
        module = _load_submodule(submodule)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
