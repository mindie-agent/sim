"""Canonical normalization, validation, and regime resolution for communication campaigns.

[Role] Validate campaign rows, hardware selectors, canary/status fields,
  expansion-regime evidence, and measurement/window descriptors.
[Invariants] Observed or profiler evidence alone may confirm AICPU/AIV; a
  requested mode alone never promotes. Status/canary failures and non-matching
  hardware identities raise explicitly.
[Timeouts & Locks] Pure in-memory normalization with no waits, locks, or I/O.
[Side-Effect Policy] Pure functions; never mutates input mappings or launches
  device work.
"""
from __future__ import annotations

import json
from typing import Mapping

from mindie_sim.schema.common import ModelingError
from .golden import A5_ARCH_FAMILY, A5_HARDWARE_KEY, A5_SKU

INTRA_NODE_MAX_RANKS = 8
NATIVE_WINDOW_WARMUP_CALLS = 3
NATIVE_WINDOW_ADMITTED_REPEATS = 5

_EXPANSION_MODES = frozenset({"aicpu", "aiv"})
_CANARY_OK = frozenset({
    "all_ranks_matched", "ok", "passed", "matched", "match", "completed",
})
_SCOPES = {
    "intra_node_ub": "intra_node_ub",
    "intra": "intra_node_ub",
    "intra_node": "intra_node_ub",
    "ub": "intra_node_ub",
    "cross_node_urma": "cross_node_urma",
    "cross_node": "cross_node_urma",
    "cross": "cross_node_urma",
    "urma": "cross_node_urma",
    "cross_node_urma_far": "cross_node_urma_far",
    "far": "cross_node_urma_far",
}
_PAIR_PARTS = {"0->1", "0->4", "0->8", "0->24"}

__all__ = [
    "INTRA_NODE_MAX_RANKS",
    "NATIVE_WINDOW_ADMITTED_REPEATS",
    "NATIVE_WINDOW_WARMUP_CALLS",
    "resolve_expansion_regime",
]


class _Resolution(dict):
    """Mapping whose resolved fields are also readable as attributes."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _normalized_text(value):
    return "".join(character for character in str(value).lower()
                   if character.isalnum())


def _is_a5(hardware_key, arch_family=None):
    if arch_family is not None and str(arch_family).strip().upper() != A5_ARCH_FAMILY:
        return False
    key = str(hardware_key or "").strip()
    if key in {A5_HARDWARE_KEY, A5_SKU}:
        return True
    folded = key.lower()
    return (folded in {A5_HARDWARE_KEY.lower(), A5_SKU.lower(), "a5",
                       "ascend950dt"} or folded.startswith("ascend950dt_9582"))


def _require_a5(hardware_key, arch_family=None):
    if not _is_a5(hardware_key, arch_family):
        raise ModelingError("a5_calibration_rejects_non_a5_hardware")


def _mode(value):
    text = str(value or "").strip().lower()
    if text in {"", "unspecified", "not_observed", "default", "auto", "none"}:
        return "unspecified"
    if text in {"aicpu", "ai_cpu", "cpu"}:
        return "aicpu"
    if text in {"aiv", "ai_vector_core", "vector", "aivector"}:
        return "aiv"
    return text


def _observed_mode(value):
    text = str(value or "").strip().lower()
    return text if text in _EXPANSION_MODES else "not_observed"


def _profiler_modes(evidence):
    """Extract explicitly named expansion engines from profiler text."""
    modes, stack = set(), [evidence]
    while stack:
        if modes >= _EXPANSION_MODES:
            break
        item = stack.pop()
        if isinstance(item, Mapping):
            for key, value in item.items():
                stack.extend((key, value))
            continue
        if isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
            continue
        if item is None or item is True or item is False:
            continue
        text = _normalized_text(item)
        if "aicpu" in text:
            modes.add("aicpu")
        if "aiv" in text or "aivectorcore" in text:
            modes.add("aiv")
    return modes


def resolve_expansion_regime(requested_expansion_mode, observed_expansion_mode, *,
                             profiler_evidence=None):
    """Resolve one expansion regime without promoting an unconfirmed request."""
    raw_req = _mode(requested_expansion_mode)
    observed = _observed_mode(observed_expansion_mode)
    requested = raw_req if raw_req in _EXPANSION_MODES else "unspecified"
    evidence = _profiler_modes(profiler_evidence)
    regime, conflict = "unspecified", False
    if observed in _EXPANSION_MODES:
        regime = observed
        conflict = bool(evidence and evidence != {observed})
    elif len(evidence) == 1:
        regime = next(iter(evidence))
    elif evidence:
        conflict = True
    confirmed = regime in _EXPANSION_MODES and not conflict
    result = {
        "requested_expansion_mode": requested,
        "observed_expansion_mode": observed,
        "regime": regime if confirmed else "unspecified",
        "expansion_unconfirmed": not confirmed,
        "profiler_confirmed": bool(evidence) and regime in evidence and not conflict,
        "profiler_modes": sorted(evidence),
        "profiler_evidence": profiler_evidence,
        "confirmed_by": ("observed" if observed in _EXPANSION_MODES and not conflict
                         else "profiler" if confirmed else None),
        "evidence_conflict": conflict,
        "requested_mismatch": (observed in _EXPANSION_MODES
                               and requested in _EXPANSION_MODES
                               and observed != requested),
    }
    result["unconfirmed_tag"] = (None if confirmed else
                                 f"requested_{requested}_unconfirmed"
                                 if requested in _EXPANSION_MODES else None)
    return _Resolution(result)


def _collective(value):
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "allreduce": "all_reduce", "all_reduce_only": "all_reduce",
        "allgather": "all_gather", "all_gather_only": "all_gather",
        "reduce_scatter_only": "reduce_scatter", "reduce_scatter_v": "reduce_scatter",
        "alltoall": "all_to_all", "all_to_all_only": "all_to_all",
        "alltoallv": "all_to_allv", "all_to_allv_only": "all_to_allv",
        "p2p": "p2p_send_recv", "p2p_only": "p2p_send_recv",
        "send_recv": "p2p_send_recv", "point_to_point": "p2p_send_recv",
    }
    return aliases.get(text, text)


def _dtype(value):
    text = str(value or "bf16").strip().lower().replace("_", "-")
    if text in {"fp8", "fp8e4m3", "fp8-e4m3", "e4m3", "fp8-e4m3fn"}:
        return "fp8-e4m3"
    if text in {"int8", "w8a8", "s8"}:
        return "int8"
    if text in {"bf16", "bfloat16"}:
        return "bf16"
    return text


def _explicit_scope(value):
    text = str(value or "").strip().lower()
    if text in _SCOPES:
        return _SCOPES[text]
    for token, scope in (("far", "cross_node_urma_far"), ("intra", "intra_node_ub"),
                         ("cross", "cross_node_urma"), ("urma", "cross_node_urma"),
                         ("2node", "cross_node_urma"), ("4node", "cross_node_urma")):
        if token in text:
            return scope
    return None


def _scope(value, world_size):
    return _explicit_scope(value) or (
        "cross_node_urma" if int(world_size) > INTRA_NODE_MAX_RANKS
        else "intra_node_ub")


def _explicit_pair(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return f"{int(value[0])}->{int(value[1])}"
        except (TypeError, ValueError):
            return None
    if isinstance(value, Mapping):
        source, destination = value.get("source"), value.get("destination")
        if source is not None and destination is not None:
            return f"{int(source)}->{int(destination)}"
    text = str(value or "").strip().lower()
    if text in _PAIR_PARTS:
        return text
    compact = "".join(text.replace("->", ">").replace("-", ">").replace(":", ">")
                      .replace("_", ">").split())
    if ">" not in compact:
        return None
    left, right = compact.split(">", 1)
    return f"{left}->{right}" if left.isdigit() and right.isdigit() else None


def _pair(value, collective, scope, world_size):
    if collective != "p2p_send_recv":
        return None
    if explicit := _explicit_pair(value):
        return explicit
    text = str(value or "").strip().lower()
    if "far" in text or scope == "cross_node_urma_far":
        return "0->24"
    if scope == "intra_node_ub":
        return "0->1"
    if "cross" in text or int(world_size) > INTRA_NODE_MAX_RANKS:
        return "0->8"
    if "intra" in text:
        return "0->1"
    return "0->8" if scope.startswith("cross") else "0->1"


def _pair_scope(pair, world_size):
    if pair == "0->1":
        return "intra_node_ub"
    if pair == "0->24":
        return "cross_node_urma_far"
    return ("cross_node_urma"
            if pair and int(world_size) > INTRA_NODE_MAX_RANKS else None)


def _split(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"train", "heldout"}:
        return text
    raise ValueError("service_sample_split_requires_train_or_heldout")


def _protocol_key(value):
    if value is None:
        return None
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
            if isinstance(value, Mapping) else str(value))


def _same_canary(expected, observed):
    if expected is None or observed is None:
        return True
    if isinstance(expected, Mapping) and isinstance(observed, Mapping):
        return all(key in observed and _same_canary(value, observed[key])
                   for key, value in expected.items())
    if (isinstance(expected, (int, float)) and isinstance(observed, (int, float))
            and not isinstance(expected, bool) and not isinstance(observed, bool)):
        return float(expected) == float(observed)
    return expected == observed or str(expected) == str(observed)


def _canary_issue(value):
    if value is None or value is True:
        return None
    if value is False:
        return "canary_failed"
    if isinstance(value, str):
        return None if value.strip().lower() in _CANARY_OK else f"canary={value!r}"
    if isinstance(value, Mapping):
        expected, observed = value.get("expected"), value.get("observed")
        if expected is not None and not _same_canary(expected, observed):
            return f"canary={expected!r}!={observed!r}"
    return None


def _status_issue(document):
    for key in ("status", "state"):
        if key in document and str(document[key]).strip().lower() not in {"completed", "ok"}:
            return f"{key}={document[key]!r}"
    if any(str(status).strip().lower() not in {"completed", "ok"}
           for status in document.get("rank_status") or ()):
        return f"rank_status={document['rank_status']!r}"
    for key in ("runs", "rank_reports", "cases"):
        children = document.get(key) or ()
        for child in children if isinstance(children, (list, tuple)) else ():
            if isinstance(child, Mapping) and (issue := _status_issue(child)):
                return issue
    return None


def _validate_row(document, label):
    if issue := _status_issue(document):
        raise ValueError(f"a5_campaign_not_completed:{label}:{issue}")
    for key in ("canary_ok", "canary_passed"):
        if key in document and document[key] is not True:
            raise ValueError(f"a5_campaign_canary_failed:{label}:{key}={document[key]!r}")
    for key in ("canary", "correctness"):
        issue = _canary_issue(document.get(key))
        if issue:
            raise ValueError(f"a5_campaign_canary_failed:{label}:{key}:{issue}")
    for target in (document, document.get("machine"), document.get("hardware"),
                   document.get("stack_binding")):
        if isinstance(target, Mapping) and any(
            key in target for key in ("hardware_key", "sku", "arch_family",
                                      "key", "architecture")
        ):
            _require_a5(
                target.get("hardware_key") or target.get("sku")
                or target.get("key") or A5_HARDWARE_KEY,
                target.get("arch_family") or target.get("architecture"))


_validate_campaign = _validate_row


def _selectors(document, row=None):
    row = row or {}
    row_proto, doc_proto = row.get("protocol"), document.get("protocol")
    return {
        "stack": row.get("stack", document.get("stack")),
        "cann": (row.get("cann") or (row.get("machine") or {}).get("cann")
                 or (row.get("stack_binding") or {}).get("cann")
                 or document.get("cann")
                 or (document.get("machine") or {}).get("cann")
                 or (document.get("stack_binding") or {}).get("cann")),
        "algorithm": (row.get("algorithm")
                      or (row_proto.get("algorithm")
                          if isinstance(row_proto, Mapping) else None)
                      or document.get("algorithm")
                      or (doc_proto.get("algorithm")
                          if isinstance(doc_proto, Mapping) else None)),
        "protocol": row_proto if "protocol" in row else doc_proto,
    }


def _merge_obs(parent, child):
    merged = {**parent, **child}
    if "observed_expansion_mode" in child or "observed_execution" in child:
        merged["observed_expansion_mode"] = child.get(
            "observed_expansion_mode", child.get("observed_execution"))
    sel = _selectors(child)
    for key in ("stack", "cann", "algorithm"):
        if sel[key] is not None:
            merged[key] = sel[key]
    return merged


def _canonical_semantic(value):
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def _fallback_measurement(collective, boundary, algorithm_identified=False):
    """Explicit campaign fallback; never claims a native AI_CPU child."""
    if boundary == "resource_phase":
        return {
            "kind": "resource_phase", "phase": "collective_phase",
            "resource": "collective_phase",
            "work_coordinate": "captured_per_rank_payload",
            "duration_unit": "ms", "source_kind": "campaign_resource_phase",
            "service_semantics": "collective_phase_service",
            "complete_collective_call": False, "complete_directed_call": False,
            "link_transit_identified": False, "hop_scaling": "not_derived",
            "algorithm_identified": bool(algorithm_identified),
            "direction_observed": False, "explicit": False,
        }
    if collective == "p2p_send_recv":
        return {
            "kind": "device_call", "phase": "directed_p2p_call",
            "resource": "directed_p2p_call",
            "work_coordinate": "captured_per_rank_payload",
            "duration_unit": "ms", "source_kind": "campaign_device_call",
            "service_semantics": "directed_p2p_call",
            "complete_collective_call": False, "complete_directed_call": True,
            "link_transit_identified": False, "hop_scaling": "not_derived",
            "algorithm_identified": bool(algorithm_identified),
            "direction_observed": True, "explicit": False,
        }
    return {
        "kind": "device_call", "phase": "collective_call",
        "resource": "collective_call",
        "work_coordinate": "captured_per_rank_payload",
        "duration_unit": "ms", "source_kind": "campaign_device_call",
        "service_semantics": "whole_collective_call_envelope",
        "complete_collective_call": True, "complete_directed_call": False,
        "link_transit_identified": False, "hop_scaling": "not_derived",
        "algorithm_identified": bool(algorithm_identified),
        "direction_observed": False, "explicit": False,
    }


def _validate_measurement_window(measurement, window):
    phase = measurement.get("phase")
    kind = window.get("kind")
    if phase == "aicpu_child" and kind != "native_single_target_call":
        raise ValueError("a5_measurement_transport_window_mismatch")
    if phase in {"collective_call", "directed_p2p_call", "collective_phase"} \
            and kind == "native_single_target_call":
        raise ValueError("a5_measurement_transport_window_mismatch")


def _fallback_transport_window(measurement):
    if measurement.get("phase") == "aicpu_child":
        return {
            "kind": "native_single_target_call", "phase": "aicpu_child",
            "warmup_calls": NATIVE_WINDOW_WARMUP_CALLS,
            "admitted_repeats": NATIVE_WINDOW_ADMITTED_REPEATS,
            "profiled_ranks": "0,group_size-1",
            "clock_domain": "device", "concurrent_collectives": False,
        }
    return {
        "kind": "campaign_steady_state", "phase": "per_op",
        "warmup_calls": None, "admitted_repeats": None,
        "clock_domain": "device", "concurrent_collectives": False,
    }
