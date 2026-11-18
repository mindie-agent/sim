"""Compact projections from engine telemetry into graph-report fields.

[Role] Serialize trace-retention counters, the selected matrix wrapper and the
  C11 backend status into the stable subset a graph report publishes.
[Invariants] Projection never mutates the engine or operator metadata and never
  invents a backend: every field is read from the live object as-is.  A missing
  or unknown wrapper/status yields an empty/compact record, not a guess.
[Timeouts & Locks] O(1) attribute reads; no I/O, threads or device access.
[Side-Effect Policy] Pure functions over already-materialized metadata.
"""
from __future__ import annotations

from collections.abc import Mapping


def bounded_event_profile(engine) -> dict:
    """Separate completed physical events from optional debug-trace retention."""
    retained = len(engine.trace)
    return {
        "completed_physical_events": engine.completed_count,
        "completed_tasks": engine.completed_count,
        "trace_retained": retained,
        "trace_dropped": engine.trace_dropped,
        "trace_events": retained + engine.trace_dropped,
    }


def matrix_wrapper_projection(metadata) -> dict:
    """Publish only the selected wrapper's semantic kind/source role.

    The nested metadata is the explicit source-selection result; absent it the
    ordinary matrix row keeps no kind. This is attribution, not a second op.
    """
    wrapper = metadata.get("matrix_wrapper")
    if not isinstance(wrapper, Mapping):
        return {}
    native = wrapper.get("native_kind")
    if not isinstance(native, str) or not native:
        return {}
    result = {"native_kind": native}
    role = wrapper.get("source_role")
    if isinstance(role, str) and role:
        result["source_role"] = role
    source = wrapper.get("source")
    basis = (wrapper.get("source_basis")
             or (source.get("basis") if isinstance(source, Mapping) else None))
    if isinstance(basis, str) and basis:
        result["source_basis"] = basis
    return result


def accel_status(view) -> dict:
    """Stable compact subset of ``Engine.accel_status()`` for graph output."""
    return {
        "accel_backend": view.get("accel_backend"),
        "native_steps": int(view.get("native_steps") or 0),
        "python_steps": int(view.get("python_steps") or 0),
        "fallback_reason": view.get("fallback_reason"),
        "mode": view.get("mode"),
        "enabled": bool(view.get("enabled")),
    }
