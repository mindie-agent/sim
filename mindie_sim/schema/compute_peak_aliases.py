"""Validated compute-peak supply aliases.

[Role]
  Distinguish a declared *supply alias* (a reporting name that reuses another
  native peak) from a native operand capability.  A packed MXFP4 checkpoint may
  therefore publish ``compute_peaks["mxfp4"]`` on a derivative SKU while the
  Cube still executes native FP8.

[Invariants]
  * Aliases are acyclic, point directly at a native peak/mode and carry the
    exact same finite positive value as their target.
  * An alias name is never admitted as a native operand mode; resolving it for
    reporting returns the native target.
  * Native peak projection removes validated alias keys before architecture
    capability validation, so the native-mode rules stay strict.

[Timeouts & Locks]
  Pure mapping arithmetic; no I/O or locks.

[Side-Effect Policy]
  Read-only; callers own serialization and provenance.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from mindie_sim.schema.common import ModelingError


def _items(value: Any, reason: str) -> tuple[tuple[str, Any], ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        return tuple(value.items())
    if isinstance(value, (tuple, list)):
        result: list[tuple[str, Any]] = []
        for item in value:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise ModelingError(reason)
            result.append((item[0], item[1]))
        return tuple(result)
    raise ModelingError(reason)


def _peak_map(peaks: Any) -> dict[str, float]:
    try:
        values = dict(_items(peaks, "compute_peaks_invalid"))
    except (TypeError, ValueError) as exc:
        raise ModelingError("compute_peaks_invalid") from exc
    return {str(name): float(value) for name, value in values.items()}


def _alias_map(aliases: Any) -> dict[str, str]:
    values = _items(aliases, "compute_peak_aliases_invalid")
    result: dict[str, str] = {}
    for name, target in values:
        if not isinstance(name, str) or not name:
            raise ModelingError(f"compute_peak_alias_invalid:{name!r}")
        if not isinstance(target, str) or not target:
            raise ModelingError(f"compute_peak_alias_invalid:{name}")
        if name in result and result[name] != target:
            raise ModelingError(f"compute_peak_alias_conflict:{name}")
        result[name] = target
    return result


def resolve_compute_peak_name(
    dtype: str, peaks: Any, aliases: Any,
) -> str:
    """Return the native peak name for ``dtype`` (one validated hop chain)."""
    name = str(dtype)
    mapping = _alias_map(aliases)
    seen: set[str] = set()
    while name in mapping:
        if name in seen:
            raise ModelingError(f"compute_peak_alias_cycle:{name}")
        seen.add(name)
        name = mapping[name]
    return name


def native_compute_peak_items(
    peaks: Any, aliases: Any,
) -> tuple[tuple[str, float], ...]:
    """Project declared peaks onto native supplies, preserving declaration order."""
    values = _peak_map(peaks)
    alias_names = set(_alias_map(aliases))
    return tuple((name, value) for name, value in values.items()
                 if name not in alias_names)


def validate_compute_peak_aliases(
    peaks: Any, aliases: Any, compute_arch: Any = None,
) -> None:
    """Validate aliases against declared peaks and native compute modes."""
    values = _peak_map(peaks)
    mapping = _alias_map(aliases)
    if not mapping:
        return
    native_modes = bool(getattr(compute_arch, "has_native_modes", lambda: False)())
    for name, target in mapping.items():
        if name not in values:
            raise ModelingError(f"compute_peak_alias_undeclared_peak:{name}")
        if target in mapping:
            seen = {name}
            cursor = target
            while cursor in mapping:
                if cursor in seen:
                    raise ModelingError(f"compute_peak_alias_cycle:{name}")
                seen.add(cursor)
                cursor = mapping[cursor]
            raise ModelingError(f"compute_peak_alias_chain:{name}->{target}")
        if target not in values:
            raise ModelingError(
                f"compute_peak_alias_unknown_target:{name}->{target}")
        value = values[name]
        target_value = values[target]
        if (not math.isfinite(value) or value <= 0.0
                or not math.isfinite(target_value) or target_value <= 0.0):
            raise ModelingError(f"compute_peak_alias_invalid_value:{name}")
        if not math.isclose(value, target_value, rel_tol=0.0, abs_tol=0.0):
            raise ModelingError(
                f"compute_peak_alias_value_mismatch:{name}:{value}!={target_value}")
        if native_modes:
            try:
                compute_arch.native_mode("matrix", target)
            except ModelingError as exc:
                raise ModelingError(
                    f"compute_peak_alias_target_not_native:{name}->{target}") from exc
            try:
                compute_arch.native_mode("matrix", name)
            except ModelingError:
                continue
            raise ModelingError(f"compute_peak_alias_is_native_mode:{name}")
