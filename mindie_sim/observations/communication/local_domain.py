"""Applicability and lineage domain keys for native local communication services.

[Role] Canonicalize target-environment applicability keys separately from
  calibration-source lineage keys and reject cross-domain or conflicting
  hardware respecifications.
[Invariants] Target applicability and source lineage are compared separately so
  a matching source hash never masks a changed runtime/CANN/algorithm condition
  and prediction remains a target-applicability check.
[Timeouts & Locks] Pure in-memory validation; no locks, timers, or I/O.
[Side-Effect Policy] Pure functions; never mutates caller domain mappings.
"""
from __future__ import annotations

from typing import Mapping

from mindie_sim.schema.common import ModelingError
from .golden import A5_ARCH_FAMILY, A5_HARDWARE_KEY, A5_SKU
from .normalization import (
    NATIVE_WINDOW_ADMITTED_REPEATS,
    NATIVE_WINDOW_WARMUP_CALLS,
)

_NATIVE_LOCAL_DIRECTION = {"all_gather": "read", "reduce_scatter": "send"}
_NATIVE_LOCAL_COLLECTIVES = tuple(_NATIVE_LOCAL_DIRECTION)
_NATIVE_LOCAL_COLLECTIVE_SET = frozenset(_NATIVE_LOCAL_COLLECTIVES)

_APPLICABILITY_KEYS = (
    ("platform", "family", "platform.family"),
    ("platform", "variant", "platform.variant"),
    ("platform", "sku", "platform.sku"),
    ("platform", "key", "platform.key"),
    ("platform", "arch_family", "platform.arch_family"),
    ("topology", "kind", "topology.kind"),
    ("topology", "scope", "topology.scope"),
    ("topology", "group_size", "topology.group_size"),
    ("topology", "host", "topology.host"),
    ("topology", "nodes", "topology.nodes"),
    ("topology", "nic", "topology.nic"),
    ("software", "stack", "software.stack"),
    ("software", "cann", "software.cann"),
    ("software", "execution_core", "software.execution_core"),
    ("software", "torch", "software.torch"),
    ("software", "torch_npu", "software.torch_npu"),
    ("software", "hccl_algorithm", "software.hccl_algorithm"),
    ("software", "hccl_protocol", "software.hccl_protocol"),
    ("transport_window", "kind", "transport_window.kind"),
    ("transport_window", "phase", "transport_window.phase"),
    ("transport_window", "warmup_calls", "transport_window.warmup_calls"),
    ("transport_window", "admitted_repeats", "transport_window.admitted_repeats"),
    ("transport_window", "concurrent_collectives",
     "transport_window.concurrent_collectives"),
    ("units", "work", "units.work"),
    ("units", "duration", "units.duration"),
    ("units", "rate", "units.rate"),
    ("units", "rate_scope", "units.rate_scope"),
    ("units", "active_units", "units.active_units"),
    ("units", "dtype", "units.dtype"),
    ("units", "service_work_unit", "units.service_work_unit"),
    ("shared_domain", "id", "shared_domain.id"),
    ("shared_domain", "kind", "shared_domain.kind"),
    ("shared_domain", "scope", "shared_domain.scope"),
    ("shared_domain", "members", "shared_domain.members"),
    ("shared_domain", "host", "shared_domain.host"),
    ("shared_domain", "inter_node", "shared_domain.inter_node"),
    ("shared_domain", "nic", "shared_domain.nic"),
)
_APPLICABILITY_KEY_SET = frozenset(key for _, _, key in _APPLICABILITY_KEYS) | {
    "direction", "hardware",
}
_LINEAGE_KEYS = (
    ("source", "summary_sha256", "source.summary_sha256"),
    ("source", "manifest_sha256", "source.manifest_sha256"),
    ("source", "versions_sha256", "source.versions_sha256"),
    ("source", "environment_image_id", "source.environment_image_id"),
)
_LINEAGE_KEY_SET = frozenset(key for _, _, key in _LINEAGE_KEYS)

__all__ = [
    "a5_local_service_applicability_key",
    "a5_local_service_domain_key",
    "a5_local_service_lineage_key",
    "require_a5_local_service_applicability",
    "require_a5_local_service_domain",
]


def _flatten_sections(mapping, paths):
    flat = {}
    for outer, inner, flat_key in paths:
        section = mapping.get(outer)
        if isinstance(section, Mapping) and inner in section:
            flat[flat_key] = section[inner]
    return flat


def _applicability_sections(domain):
    if "conditions" in domain and isinstance(domain["conditions"], Mapping):
        return domain["conditions"]
    if "applicability" in domain and isinstance(domain["applicability"], Mapping):
        return domain["applicability"]
    if any(isinstance(domain.get(outer), Mapping)
           for outer, _, _ in _APPLICABILITY_KEYS):
        return domain
    return None


def _complete_applicability_identity(flat):
    """Add scalar hardware and reject a hardware/platform identity conflict."""
    hardware = flat.get("hardware")
    platform_key = flat.get("platform.key")
    platform_sku = flat.get("platform.sku")
    expected = platform_key or platform_sku
    if hardware is not None and expected is not None and str(hardware) != str(expected):
        raise ValueError("a5_local_service_hardware_identity_conflict")
    return flat


def a5_local_service_applicability_key(domain):
    """Canonical target-applicability key; source lineage is separate."""
    if not isinstance(domain, Mapping):
        raise ValueError("a5_local_service_domain_requires_mapping")
    if "applicability_key" in domain:
        return _complete_applicability_identity(
            {str(key): value for key, value in
             dict(domain["applicability_key"]).items()})
    conditions = _applicability_sections(domain)
    if conditions is not None:
        flat = _flatten_sections(conditions, _APPLICABILITY_KEYS)
        if "hardware" in conditions:
            flat["hardware"] = conditions["hardware"]
        if "direction" in conditions:
            flat["direction"] = conditions["direction"]
        return _complete_applicability_identity(flat)
    if not any(isinstance(value, Mapping) for value in domain.values()):
        flat = {str(key): value for key, value in domain.items()
                if key in _APPLICABILITY_KEY_SET}
        return _complete_applicability_identity(flat)
    return {}


def a5_local_service_lineage_key(domain):
    """Canonical calibration-source lineage, not target applicability."""
    if not isinstance(domain, Mapping):
        raise ValueError("a5_local_service_domain_requires_mapping")
    if "lineage_key" in domain:
        return {str(key): value for key, value in dict(domain["lineage_key"]).items()}
    source = domain.get("source")
    if isinstance(source, Mapping):
        flat = _flatten_sections({"source": source}, _LINEAGE_KEYS)
        if flat:
            return flat
        return {key: source.get(inner) for outer, inner, key in _LINEAGE_KEYS
                if outer == "source" and inner in source}
    if not any(isinstance(value, Mapping) for value in domain.values()):
        return {str(key): value for key, value in domain.items()
                if key in _LINEAGE_KEY_SET}
    return {}


def a5_local_service_domain_key(domain):
    """Backward-compatible merged key; require() checks the two parts apart."""
    return {**a5_local_service_applicability_key(domain),
            **a5_local_service_lineage_key(domain)}


def _declared_applicability(record):
    if not isinstance(record, Mapping):
        raise ValueError("a5_local_service_record_requires_mapping")
    declared_app = dict(record.get("applicability_key") or {})
    declared_lineage = dict(record.get("lineage_key") or {})
    if not declared_app and not declared_lineage:
        declared = dict(record.get("domain_key") or {})
        declared_app = {key: value for key, value in declared.items()
                        if key in _APPLICABILITY_KEY_SET}
        declared_lineage = {key: value for key, value in declared.items()
                            if key in _LINEAGE_KEY_SET}
    if not declared_app and not declared_lineage:
        raise ValueError("a5_local_service_record_requires_declared_domain")
    return declared_app, declared_lineage


def _presence_mismatches(declared, requested):
    missing = sorted(key for key in declared if key not in requested)
    changed = sorted(key for key in declared if key in requested
                     and requested[key] != declared[key])
    return missing, changed


def require_a5_local_service_applicability(record, domain):
    """Match the target environment only; calibration lineage is separate.

    This is the local counterpart of T06 ``service_conditions_match``:
    declared paths must be present and equal; extra requested paths and
    source lineage are not applicability.
    """
    declared_app, _declared_lineage = _declared_applicability(record)
    requested_app = a5_local_service_applicability_key(domain)
    missing, changed = _presence_mismatches(declared_app, requested_app)
    if missing or changed:
        raise ModelingError(
            "a5_local_service_domain_mismatch:applicability:"
            + ",".join(missing + changed))
    return declared_app


def require_a5_local_service_domain(record, domain):
    """Reject a reuse/supply respecification outside the measured domain.

    Applicability (target environment) and lineage (calibration source) are
    compared separately so a source hash cannot stand in for a changed runtime
    condition, and a shared source cannot hide a changed torch/CANN/algorithm.
    """
    declared_app, declared_lineage = _declared_applicability(record)
    require_a5_local_service_applicability(record, domain)
    requested_lineage = a5_local_service_lineage_key(domain)
    missing_lineage, changed_lineage = _presence_mismatches(
        declared_lineage, requested_lineage)
    if missing_lineage or changed_lineage:
        raise ModelingError(
            "a5_local_service_domain_mismatch:lineage:"
            + ",".join(missing_lineage + changed_lineage))
    return {**declared_app, **declared_lineage}


def _native_local_domain(source_identity, *, group_size, stack, cann,
                         direction, dtype):
    software_versions = source_identity.get("python_packages") or {}
    shared = {
        "id": f"ub_mesh:{source_identity['host']}:W{int(group_size)}",
        "kind": "ub_mesh", "scope": "intra_node",
        "members": int(group_size), "host": source_identity["host"],
        "inter_node": False, "nic": "na",
    }
    window = {
        "kind": "native_single_target_call", "phase": "aicpu_child",
        "warmup_calls": NATIVE_WINDOW_WARMUP_CALLS,
        "admitted_repeats": NATIVE_WINDOW_ADMITTED_REPEATS,
        "profiled_ranks": [0, int(group_size) - 1],
        "clock_domain": "device", "concurrent_collectives": False,
        "start": "unique hccl target-call marker/range",
        "end": "target-call completion",
    }
    conditions = {
        # ``hardware`` stays the T06/profile hardware key string; the
        # structured identity lives under ``platform`` so conditions can be
        # consumed by service_conditions_match without a key/type conflict.
        "hardware": A5_HARDWARE_KEY,
        "platform": {"family": A5_ARCH_FAMILY, "variant": "DT",
                     "sku": A5_SKU, "key": A5_HARDWARE_KEY,
                     "arch_family": A5_ARCH_FAMILY},
        "topology": {"kind": "ub_mesh", "scope": "intra_node",
                     "group_size": int(group_size),
                     "host": source_identity["host"], "nodes": 1, "nic": "na"},
        "software": {
            "stack": str(stack), "cann": str(cann),
            "execution_core": "aicpu",
            "torch": software_versions.get("torch"),
            "torch_npu": software_versions.get("torch_npu"),
            "hccl_algorithm": source_identity.get("hccl_algorithm"),
            "hccl_protocol": source_identity.get("hccl_protocol"),
        },
        "transport_window": window,
        "units": {"work": "bytes_per_rank_input", "duration": "ms",
                  "rate": "bytes_per_second", "rate_scope": "per_owner",
                  "active_units": int(group_size), "dtype": dtype,
                  "service_work_unit": "bytes"},
        "direction": direction,
        "shared_domain": shared,
    }
    measurement = {
        "kind": "native_target_call", "phase": "aicpu_child",
        "resource": "aicpu_child",
        "work_coordinate": "per_rank_input_bytes", "duration_unit": "ms",
        "source_kind": "native_ai_cpu_child",
        "service_semantics": "aicpu_child_collective_phase",
        "complete_collective_call": False, "complete_directed_call": False,
        "link_transit_identified": False, "hop_scaling": "not_derived",
        "algorithm_identified": False, "direction_observed": False,
    }
    domain = {
        **conditions,
        "source": source_identity,
        "measurement_boundary": measurement,
        "provenance": {
            "direction_basis": "hccl_source_template_native_direction",
            "direction_observed": False,
            "algorithm_identified": False,
            "shared_domain_basis": "declared_intrahost_ub_mesh_and_group_size",
            "transport_window_basis": "native_target_call_manifest",
        },
    }
    applicability_key = a5_local_service_applicability_key(conditions)
    lineage_key = a5_local_service_lineage_key({"source": source_identity})
    domain_key = {**applicability_key, **lineage_key}
    return domain, domain_key, applicability_key, lineage_key


def _local_service_evidence_gaps(collective, group_size):
    return [
        "link_transit_is_not_separated_from_the_aicpu_child_phase",
        "per_hop_link_price_is_not_identified",
        f"measured_only:{collective}:W{int(group_size)}:bf16:aicpu:intrahost:rack08:cann9.2.0",
        "cross_node_aiv_and_other_skus_are_unmeasured",
        "pr_variant_is_unmeasured",
        "hccl_algorithm_and_protocol_are_not_exposed_by_the_python_abi",
        "heldout_split_is_by_payload_not_by_environment",
    ]
