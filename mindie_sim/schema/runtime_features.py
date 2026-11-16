"""Canonical runtime/parallel feature normalization and validation.

[Role]
  Provide one pure normalization entry point for the Issue #34 runtime and
  parallel feature surface: local projection widths, collective startup mode,
  matrix compute policy, prefix pooling/read modes, homogeneous graph reuse and
  redundant expert slots.  Parsing (``Case``/``Software``/``ParallelSpec``), the
  CLI and JSON aliases all converge here.

[Invariants]
  * Inputs are never mutated; every function returns a fresh copy.
  * Only the declared canonical names are emitted.  ``kda_proj_tp`` is an input
    alias of ``linear_proj_tp``; equal duplicates are accepted, unequal ones
    raise ``runtime_feature_conflict:<canonical>``.
  * Booleans are not integers, widths/counts are nonnegative integers and
    modes come from fixed registries.
  * ``runtime.routing.redundant_experts_per_rank`` is lifted into
    ``runtime.parallel`` and removed before routing-policy parsing.
  * The narrow top-level ``software.recover_ssm`` alias accepts only that key
    and must agree with ``runtime.recover_ssm`` when both are present.

[Timeouts & Locks]
  Pure metadata arithmetic; no I/O, locks or background work.

[Side-Effect Policy]
  Read-only: no filesystem, network or global mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from mindie_sim.schema.common import ModelingError


ZERO: int = 0
ONE: int = 1
EMPTY_PREFIX_CACHE_KEY: str = ""
DEFAULT_PREFIX_CACHE_KEY: str = "default-prefix"

COLLECTIVE_STARTUP_MODES = frozenset(
    {"legacy", "ring", "log2", "direct", "concurrent", "topology"}
)
COLLECTIVE_STARTUP_NORMALIZED = MappingProxyType({
    "legacy": "legacy",
    "ring": "ring",
    "log2": "log2",
    "direct": "concurrent",
    "concurrent": "concurrent",
    "topology": "topology",
})
MATRIX_COMPUTE_POLICIES = frozenset({"auto", "declared"})
PREFIX_CACHE_MODES = frozenset({"private", "shared"})
PREFIX_READ_MODES = frozenset({"auto", "per_sequence", "batch_shared"})
HOMOGENEOUS_GRAPH_REUSE_MODES = frozenset({"off", "auto", "require"})
EXPERT_SLOT_POLICIES = frozenset({"balanced_expectation", "explicit"})
EXPERT_SLOT_SCOPES = frozenset({"all", "target"})
EXPERT_SLOT_POLICY_KEY = "expert_slot_policy"
EXPERT_SLOT_MAP_KEY = "expert_slot_map"
EXPERT_SLOT_SCOPE_KEY = "expert_slot_scope"
REDUNDANT_EXPERTS_KEY = "redundant_experts_per_rank"
LOCAL_PARALLEL_DEFAULTS = MappingProxyType({
    "o_proj_tp": ONE,
    "linear_proj_tp": ONE,
    "lm_head_tp": ONE,
    REDUNDANT_EXPERTS_KEY: ZERO,
})
RUNTIME_MODE_DEFAULTS = MappingProxyType({
    "collective_startup_mode": "legacy",
    "matrix_compute_policy": "auto",
    "prefix_cache_mode": "private",
    "prefix_read_mode": "auto",
    "homogeneous_graph_reuse": "off",
})
KDA_PROJ_TP_ALIAS = "kda_proj_tp"
LINEAR_PROJ_TP = "linear_proj_tp"
PARALLEL_FEATURE_NAMES = frozenset(LOCAL_PARALLEL_DEFAULTS) | {KDA_PROJ_TP_ALIAS}
RUNTIME_FEATURE_NAMES = frozenset(RUNTIME_MODE_DEFAULTS) | {"recover_ssm"}
SOFTWARE_ALIAS_KEYS = frozenset({"recover_ssm"})
SOFTWARE_ALIAS = "software"
ROUTING_KEY = "routing"
PARALLEL_KEY = "parallel"
RUNTIME_KEY = "runtime"
SLOT_KEYS = (EXPERT_SLOT_POLICY_KEY, EXPERT_SLOT_MAP_KEY, EXPERT_SLOT_SCOPE_KEY,
             REDUNDANT_EXPERTS_KEY)


def _feature_int(value: Any, name: str, *, minimum: int = ZERO) -> int:
    if isinstance(value, bool):
        raise ModelingError(f"runtime_feature_invalid:{name}:{value!r}")
    if isinstance(value, float):
        if not value.is_integer():
            raise ModelingError(f"runtime_feature_invalid:{name}:{value!r}")
        value = int(value)
    if not isinstance(value, int) or value < minimum:
        raise ModelingError(f"runtime_feature_invalid:{name}:{value!r}")
    return int(value)


def _feature_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ModelingError(f"runtime_feature_invalid:{name}:{value!r}")
    return value


def _feature_mode(value: Any, name: str, allowed: frozenset[str]) -> str:
    if not isinstance(value, str):
        raise ModelingError(f"runtime_feature_invalid:{name}:{value!r}")
    mode = value.strip().lower()
    if mode not in allowed:
        raise ModelingError(f"runtime_feature_unsupported:{name}:{value}")
    return COLLECTIVE_STARTUP_NORMALIZED.get(mode, mode)


def _parallel_int(raw: Mapping[str, Any], canonical: str) -> tuple[bool, int]:
    present = canonical in raw
    value: Any = raw.get(canonical)
    alias_present = canonical == LINEAR_PROJ_TP and KDA_PROJ_TP_ALIAS in raw
    if alias_present:
        alias_value = _feature_int(raw[KDA_PROJ_TP_ALIAS], KDA_PROJ_TP_ALIAS)
        if present:
            declared = _feature_int(value, canonical)
            if declared != alias_value:
                raise ModelingError(f"runtime_feature_conflict:{canonical}")
            return True, declared
        return True, alias_value
    if not present:
        return False, LOCAL_PARALLEL_DEFAULTS[canonical]
    return True, _feature_int(value, canonical)


def normalize_parallel_features(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with canonical local projection widths and slot count."""
    if not isinstance(raw, Mapping):
        raise ModelingError("runtime_feature_invalid:parallel")
    result = dict(raw)
    for canonical in LOCAL_PARALLEL_DEFAULTS:
        present, value = _parallel_int(raw, canonical)
        if present:
            result[canonical] = value
    result.pop(KDA_PROJ_TP_ALIAS, None)
    return result


def normalize_runtime_features(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize runtime feature fields without mutating the caller."""
    if not isinstance(raw, Mapping):
        raise ModelingError("runtime_feature_invalid:runtime")
    result = dict(raw)
    parallel = result.get(PARALLEL_KEY)
    if parallel is None:
        parallel = {}
    elif not isinstance(parallel, Mapping):
        raise ModelingError("runtime_feature_invalid:parallel")
    routing = result.get(ROUTING_KEY)
    if routing is not None and not isinstance(routing, Mapping):
        raise ModelingError("runtime_feature_invalid:routing")
    routing_copy = dict(routing or {})
    normalized_parallel = normalize_parallel_features(parallel)
    if REDUNDANT_EXPERTS_KEY in routing_copy:
        lifted = _feature_int(routing_copy[REDUNDANT_EXPERTS_KEY], REDUNDANT_EXPERTS_KEY)
        if REDUNDANT_EXPERTS_KEY in parallel:
            declared = _feature_int(parallel[REDUNDANT_EXPERTS_KEY], REDUNDANT_EXPERTS_KEY)
            if declared != lifted:
                raise ModelingError(f"runtime_feature_conflict:{REDUNDANT_EXPERTS_KEY}")
        normalized_parallel[REDUNDANT_EXPERTS_KEY] = lifted
        routing_copy.pop(REDUNDANT_EXPERTS_KEY)
    if normalized_parallel:
        result[PARALLEL_KEY] = normalized_parallel
    if routing is not None:
        result[ROUTING_KEY] = routing_copy
    overlap_aliases = ("enable_shared_expert_overlap", "overlap_shared_expert")
    for alias in overlap_aliases:
        if alias in result:
            result["multistream_overlap_shared_expert"] = _feature_bool(result.pop(alias), alias)
    dbo_aliases = ("dbo", "dual_batch_overlap", "enable_dual_batch_overlap")
    for alias in dbo_aliases:
        if alias in result:
            result["enable_dbo"] = _feature_bool(result.pop(alias), alias)
    for name in RUNTIME_MODE_DEFAULTS:
        if name in result:
            result[name] = _feature_mode(result[name], name, _MODES[name])
    if "recover_ssm" in result:
        value = result["recover_ssm"]
        if not isinstance(value, bool):
            raise ModelingError("runtime_recover_ssm_invalid")
        result["recover_ssm"] = value
    return result


_MODES = {
    "collective_startup_mode": COLLECTIVE_STARTUP_MODES,
    "matrix_compute_policy": MATRIX_COMPUTE_POLICIES,
    "prefix_cache_mode": PREFIX_CACHE_MODES,
    "prefix_read_mode": PREFIX_READ_MODES,
    "homogeneous_graph_reuse": HOMOGENEOUS_GRAPH_REUSE_MODES,
}


def _merge_software_alias(raw: dict[str, Any]) -> None:
    alias = raw.pop(SOFTWARE_ALIAS, None)
    if alias is None:
        return
    if not isinstance(alias, Mapping):
        raise ModelingError("case_software_alias_invalid")
    unknown = set(alias) - SOFTWARE_ALIAS_KEYS
    if unknown:
        raise ModelingError(
            "case_software_alias_unknown:" + ",".join(sorted(unknown)))
    runtime = raw.get(RUNTIME_KEY)
    if runtime is None:
        runtime = {}
    elif not isinstance(runtime, Mapping):
        raise ModelingError("runtime_feature_invalid:runtime")
    merged = dict(runtime)
    for key, value in alias.items():
        if key in merged:
            left = _feature_bool(merged[key], key)
            right = _feature_bool(value, key)
            if left != right:
                raise ModelingError(f"runtime_feature_conflict:{key}")
        merged[key] = _feature_bool(value, key)
    raw[RUNTIME_KEY] = merged


def normalize_case_runtime(raw_case: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a case mapping's runtime/parallel feature declarations."""
    if not isinstance(raw_case, Mapping):
        raise ModelingError("case_invalid")
    result = dict(raw_case)
    _merge_software_alias(result)
    runtime = result.get(RUNTIME_KEY)
    if runtime is None:
        return result
    if not isinstance(runtime, Mapping):
        raise ModelingError("runtime_feature_invalid:runtime")
    result[RUNTIME_KEY] = normalize_runtime_features(runtime)
    return result


def split_routing_features(
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split logical routing from physical expert-slot declarations."""
    if not isinstance(raw, Mapping):
        raise ModelingError("routing_invalid")
    logical: dict[str, Any] = {}
    slots: dict[str, Any] = {}
    for key, value in raw.items():
        if key in SLOT_KEYS:
            slots[key] = value
        else:
            logical[key] = value
    if EXPERT_SLOT_POLICY_KEY in slots:
        slots[EXPERT_SLOT_POLICY_KEY] = _feature_mode(
            slots[EXPERT_SLOT_POLICY_KEY], EXPERT_SLOT_POLICY_KEY, EXPERT_SLOT_POLICIES)
    if EXPERT_SLOT_SCOPE_KEY in slots:
        slots[EXPERT_SLOT_SCOPE_KEY] = _feature_mode(
            slots[EXPERT_SLOT_SCOPE_KEY], EXPERT_SLOT_SCOPE_KEY, EXPERT_SLOT_SCOPES)
    if EXPERT_SLOT_MAP_KEY in slots and not isinstance(slots[EXPERT_SLOT_MAP_KEY], Mapping):
        raise ModelingError("runtime_feature_invalid:expert_slot_map")
    return logical, slots


def validate_runtime_features(runtime: Any) -> None:
    """Validate an already constructed ``Software`` instance."""
    parallel = runtime.parallel
    for name in ("o_proj_tp", "linear_proj_tp", "lm_head_tp"):
        _feature_int(getattr(parallel, name, LOCAL_PARALLEL_DEFAULTS[name]), name, minimum=ONE)
    _feature_int(getattr(parallel, REDUNDANT_EXPERTS_KEY, ZERO), REDUNDANT_EXPERTS_KEY)
    for name, allowed in _MODES.items():
        _feature_mode(getattr(runtime, name), name, allowed)
    _feature_bool(getattr(runtime, "recover_ssm", False), "recover_ssm")
    routing = getattr(runtime, "routing", {}) or {}
    if REDUNDANT_EXPERTS_KEY in routing:
        raise ModelingError(
            f"runtime_feature_conflict:{REDUNDANT_EXPERTS_KEY}")


def canonical_prefix_cache_key(mode: str, key: str | None) -> str:
    """Resolve the effective physical prefix identity for one case."""
    if mode == "shared" and not (key or "").strip():
        return DEFAULT_PREFIX_CACHE_KEY
    return (key or "").strip()


def normalize_prefix_cache_key(value: Any) -> str:
    """Prefix identities are plain strings; whitespace is not significant."""
    if value is None:
        return EMPTY_PREFIX_CACHE_KEY
    if not isinstance(value, str):
        raise ModelingError(f"runtime_feature_invalid:prefix_cache_key:{value!r}")
    return value.strip()
