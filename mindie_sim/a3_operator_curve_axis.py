"""Declared curve-axis and physical-shape coupling semantics.

[Role]
  Own canonical handling of declared native/generic series axes whose physical
  shape is derived rather than a single raw dimension (for example q_kv and
  tokens_per_expert).  Adaptors use this for condition identity and public
  query matching; no caller value is copied from an artifact.

[Invariants]
  * q_kv means q == kv == value; kv is never held fixed while q varies.
  * tokens_per_expert means routed_tokens == value * experts and the balanced
    expert histogram is [value] * experts.  Unbalanced histograms stay distinct
    evidence and are not relabelled.
  * RoPE seq maps to the canonical tokens axis; source axis is retained as
    evidence by the attention adapter.
  * dtype_probe is not a physical numeric axis; it cannot silently become q.
  * Unknown derived axes are unsupported, never fallback to another dimension.
  * Query validation uses the original full shape; missing/incorrect coupling
    is unsupported.

[Timeouts & Locks]
  Pure bounded arithmetic.

[Side-Effect Policy]
  No side effects.
"""
from __future__ import annotations

from typing import Any, Mapping

DERIVED = {
    "q_kv": ("q", "kv"),
    "tokens_per_expert": ("routed_tokens",),
}
NON_NUMERIC = {"dtype_probe"}


def fixed_shape(axis: str, shape: Mapping[str, Any]) -> dict[str, Any]:
    """Shape keys that stay fixed while ``axis`` varies."""
    clean = {str(k): v for k, v in dict(shape).items()}
    for key in DERIVED.get(axis, (axis,)):
        clean.pop(key, None)
    return clean


def axis_value(axis: str, shape: Mapping[str, Any]) -> int | None:
    """Return the declared axis value from a full physical shape."""
    values = {str(k): int(v) for k, v in dict(shape).items()}
    if axis == "q_kv":
        q, kv = values.get("q"), values.get("kv")
        return q if q is not None and q == kv and q > 0 else None
    if axis == "tokens_per_expert":
        routed, experts = values.get("routed_tokens"), values.get("experts")
        if routed is None or not experts or experts <= 0 or routed % experts:
            return None
        return routed // experts
    if axis in NON_NUMERIC:
        return None
    value = values.get(axis)
    return value if value is not None and value > 0 else None


def normalize_shape(axis: str, value: int,
                    shape: Mapping[str, Any]) -> dict[str, int]:
    """Apply declared derived-axis coupling to a full point shape."""
    result = {str(k): int(v) for k, v in dict(shape).items()}
    if axis == "q_kv":
        result["q"] = int(value)
        result["kv"] = int(value)
    elif axis == "tokens_per_expert":
        experts = int(result.get("experts") or 0)
        if experts <= 0:
            raise ValueError("tokens_per_expert requires positive experts")
        result["routed_tokens"] = int(value) * experts
    elif axis in NON_NUMERIC:
        return result
    elif axis in result:
        result[axis] = int(value)
    else:
        raise ValueError(f"axis {axis!r} is not present in physical shape")
    return result


def validate_series(axis: str, shape: Mapping[str, Any],
                    declared_value: int) -> tuple[bool, str]:
    try:
        value = axis_value(axis, shape)
    except (TypeError, ValueError):
        return False, "shape_not_integer"
    if value is None:
        if axis in NON_NUMERIC:
            return True, "non_numeric_probe"
        return False, f"declared_axis_coupling_invalid:{axis}"
    if int(value) != int(declared_value):
        return False, f"declared_value_mismatch:{value}!={declared_value}"
    return True, "ok"


def query_shape(curve: Mapping[str, Any],
                shape: Mapping[str, Any]) -> tuple[dict[str, Any] | None,
                                                   int | None, str]:
    """Return fixed physical shape, x query and reason for a full query shape."""
    axis = str(curve.get("axis") or "")
    value = axis_value(axis, shape)
    if value is None:
        return None, None, f"unsupported_or_invalid_axis:{axis}"
    fixed = fixed_shape(axis, shape)
    if fixed != dict(curve.get("fixed_shape") or {}):
        return None, None, "fixed_shape_mismatch"
    return fixed, int(value), "ok"


__all__ = ["DERIVED", "NON_NUMERIC", "fixed_shape", "axis_value",
           "normalize_shape", "validate_series", "query_shape"]
