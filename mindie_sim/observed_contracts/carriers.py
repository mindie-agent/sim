"""Bounded snapshotting of arbitrary declared structure carriers.

[Role]
  Copy read-only ``Mapping``/``Sequence`` structure payloads into plain
  dictionaries and lists without mutating the caller's object, so one observed
  projection owns an immutable snapshot.

[Invariants]
  * Scalars are preserved by identity; dictionary keys must be plain strings.
  * Nesting depth is bounded; deeper payloads raise ``StructureTargetError``.
  * ``TimeoutError`` always propagates untouched before any data-error handler.
  * A mapping/sequence protocol that raises is reported as an invalid carrier,
    never silently truncated.

[Timeouts & Locks]
  One bounded recursive copy; no waits or locks.

[Side-Effect Policy]
  Pure in-memory allocation; the input carrier is never mutated.
"""
from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
from typing import Any

from .errors import StructureTargetError as ProfilerUsageError

CARRIER_MAX_NESTING_DEPTH = 64
CARRIER_INLINE_LEAF_FLOOR = (1 << 2) + 2
CARRIER_INLINE_LEAF_DEPTH = 1 << CARRIER_INLINE_LEAF_FLOOR

_SCALAR_TYPES = (str, int, float, bool, type(None))


def _snapshot_structure_carrier(value: Any, name: str = "structure metadata", _depth: int = 0) -> Any:
    """Copy one carrier into plain str-keyed containers with bounded depth."""
    if _depth > CARRIER_MAX_NESTING_DEPTH:
        raise ProfilerUsageError(f"{name} exceeds container nesting limit")
    vtype = type(value)
    if vtype in _SCALAR_TYPES:
        return value
    inline_leaf = _depth < CARRIER_INLINE_LEAF_DEPTH
    if vtype is dict:
        out = {}
        for key, item in value.items():
            if type(key) is not str:
                raise ProfilerUsageError(f"{name} keys must be plain strings")
            out[key] = (
                item if inline_leaf and type(item) in _SCALAR_TYPES
                else _snapshot_structure_carrier(item, name, _depth + 1)
            )
        return out
    if vtype is list or vtype is tuple:
        if inline_leaf:
            return [
                item if type(item) in _SCALAR_TYPES
                else _snapshot_structure_carrier(item, name, _depth + 1)
                for item in value
            ]
        return [_snapshot_structure_carrier(item, name, _depth + 1) for item in value]
    if isinstance(value, _AbcMapping):
        try:
            _ = len(value)
            out = {}
            for key in value:
                if type(key) is not str:
                    raise ProfilerUsageError(f"{name} keys must be plain strings")
                out[key] = _snapshot_structure_carrier(value[key], name, _depth + 1)
            return out
        except TimeoutError:
            raise
        except ProfilerUsageError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProfilerUsageError(f"invalid {name} mapping protocol") from exc
    if isinstance(value, (list, tuple)):
        try:
            _ = len(value)
            return [_snapshot_structure_carrier(item, name, _depth + 1) for item in value]
        except TimeoutError:
            raise
        except ProfilerUsageError:
            raise
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProfilerUsageError(f"invalid {name} sequence protocol") from exc
    return value


__all__ = ["CARRIER_MAX_NESTING_DEPTH", "_snapshot_structure_carrier"]
