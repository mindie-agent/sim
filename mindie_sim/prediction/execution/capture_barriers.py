"""One capture-barrier vocabulary for fused and standalone operators.

[Role] Normalize every declared replay barrier (explicit blocker aliases,
  capture flags, dynamic shapes, unstable pointers, mutating state effects) so
  the fusion bridge and the canonical capture audit make the same decision.
[Invariants] A blocker declared in any supported spelling survives contraction
  and blocks capture for an operator that stays standalone.  A read-only state
  effect is a graph input, never a barrier; a mutating one always is.
[Timeouts & Locks] Finite metadata scanning; no I/O, clocks or locks.
[Side-Effect Policy] Pure functions over immutable operator metadata.
"""
from __future__ import annotations

from typing import Any, Mapping

DYNAMIC_SHAPE_KEYS = (
    "dynamic_shape", "dynamic_dims", "shape_is_dynamic", "graph_dynamic_shape",
    "dynamic_shape_dims", "shape_dynamic", "symbolic_shape",
    "dynamic_shape_symbols",
)
CAPTURE_FLAG_KEYS = (
    *DYNAMIC_SHAPE_KEYS, "host_barrier", "host_sync", "synchronize",
    "cache_store", "cache_mutation", "mutates_cache", "cache_write",
    "capture_boundary", "capture_ineligible",
)
CAPTURE_BLOCKER_KEYS = ("explicit_capture_blockers", "capture_blockers",
                        "capture_blocker")
STATE_EFFECT_KEYS = ("state_effects", "gdn_state_effects")
_CACHE_BOUNDARY_KEYS = ("cache_store", "cache_mutation", "mutates_cache",
                        "cache_write")
_MUTATING_ACTIONS = frozenset({"write", "read_write", "write_candidate",
                               "commit", "rollback"})


def _iter_state_effects(metadata: Mapping):
    """Yield every declared state effect across all supported metadata keys."""
    meta = dict(metadata)
    for key in STATE_EFFECT_KEYS:
        raw = meta.get(key)
        if not raw:
            continue
        if isinstance(raw, (tuple, list)):
            yield from raw
        else:
            yield raw


def _effect_is_mutating(effect: Any) -> bool:
    if hasattr(effect, "to_dict") and not isinstance(effect, Mapping):
        effect = effect.to_dict()
    if not isinstance(effect, Mapping):
        return True
    return str(effect.get("action") or "write") in _MUTATING_ACTIONS


def has_mutating_state_effect(metadata: Mapping) -> bool:
    """True when a declared state effect writes, not merely reads."""
    return any(_effect_is_mutating(effect)
               for effect in _iter_state_effects(metadata))


def pointer_unstable(metadata: Mapping) -> bool:
    meta = dict(metadata)
    return bool(meta.get("pointer_stable") is False
                or meta.get("unstable_pointer")
                or meta.get("pointer_unstable")
                or meta.get("pointer_stability") in {"unstable", False}
                or meta.get("capture_pointer_stable") is False)


def declared_blockers(metadata: Mapping) -> tuple[str, ...]:
    """Every explicit blocker token under any supported spelling."""
    meta = dict(metadata)
    found: list[str] = []
    for key in CAPTURE_BLOCKER_KEYS:
        raw = meta.get(key)
        if isinstance(raw, str):
            found.append(raw)
        elif isinstance(raw, (tuple, list, set, frozenset)):
            found.extend(str(item) for item in raw)
    return tuple(dict.fromkeys(found))


def normalized_capture_metadata(member_specs: tuple) -> dict:
    """Union capture-blocking facts of every contracted member."""
    found: dict[str, object] = {}
    blockers: list[str] = []
    effects: list[object] = []
    seen_effects: set[tuple] = set()
    for spec in member_specs:
        meta = dict(spec.metadata)
        for key in CAPTURE_FLAG_KEYS:
            if meta.get(key):
                found[key] = meta[key]
        for effect in _iter_state_effects(meta):
            if hasattr(effect, "to_dict") and not isinstance(effect, Mapping):
                effect = effect.to_dict()
            if isinstance(effect, Mapping):
                record = dict(effect)
                identity = ("mapping", tuple(sorted((str(k), repr(v))
                                                    for k, v in record.items())))
                normalized: object = record
            else:
                identity = ("opaque", repr(effect))
                normalized = effect
            if identity not in seen_effects:
                seen_effects.add(identity)
                effects.append(normalized)
        if pointer_unstable(meta):
            found["pointer_stable"] = False
        blockers.extend(declared_blockers(meta))
    if effects:
        found["state_effects"] = tuple(effects)
    if blockers:
        found["capture_blockers"] = tuple(dict.fromkeys(blockers))
    return found


def standalone_capture_blockers(op, node=None) -> tuple[str, ...]:
    """Formatted capture blockers for one operator before replay evaluation."""
    meta = dict(getattr(op, "metadata", ()) or ())
    op_id = str(getattr(op, "op_id", ""))
    found: list[str] = []
    if getattr(op, "phase", "") == "prefill":
        found.append(f"prefill_capture_unsupported:{op_id}")
    work = meta.get("resource_work")
    if (isinstance(work, Mapping) and work.get("host")) \
            or meta.get("host_barrier") or meta.get("host_sync") \
            or meta.get("synchronize"):
        found.append(f"host_barrier:{op_id}")
    if getattr(op, "family", "") in {"allreduce", "allgather", "reduce_scatter"} \
            or (node is not None and node.collective):
        found.append(f"collective:{op_id}")
    if node is not None and str(node.state_effect) in _MUTATING_ACTIONS:
        found.append(f"state_write:{op_id}")
    if any(meta.get(key) for key in _CACHE_BOUNDARY_KEYS):
        found.append(f"capture_boundary_state:{op_id}")
    if has_mutating_state_effect(meta):
        found.append(f"capture_boundary_state:{op_id}")
    if bool(getattr(op, "capture_ineligible", False)) \
            or meta.get("capture_ineligible"):
        found.append(f"capture_ineligible:{op_id}")
    for item in declared_blockers(meta):
        found.append(f"capture_blocker:{item}")
    if meta.get("capture_boundary") \
            or (node is not None and node.capture_boundary):
        found.append(f"capture_boundary:{op_id}")
    if node is not None and node.resource == "host":
        found.append(f"host_scalar_dependency:{op_id}")
    if not tuple(getattr(op, "local_shape", ()) or ()) or any(
            meta.get(key) for key in DYNAMIC_SHAPE_KEYS):
        found.append(f"dynamic_shape:{op_id}")
    if pointer_unstable(meta):
        found.append(f"unstable_pointer:{op_id}")
    return tuple(dict.fromkeys(found))
