"""Validation and exact comparison of periodic snapshots.

[Role] Validate structural values and freeze them for periodic evidence.
[Invariants] Structural equality compares concrete types and exact values.
  The floating-point tolerance applies only to measured quantities.
[Timeouts & Locks] Validation depth is bounded. No locks or I/O.
[Side-Effect Policy] Owns frozen values and adjusts only caller-provided counters.
"""
from __future__ import annotations

import math
import sys
from collections.abc import Hashable, Mapping
from typing import Any


class PeriodicContractError(ValueError):
    """The caller supplied an unsafe or incomplete periodic contract."""


# Only floating measurement rounding may be tolerated.  Structural values never
# use this tolerance; they go through ``structural_equal`` below.
_ULP_SLOP = 8
_MAX_STATE_DEPTH = 64
# A canonical tree adds a tagged tuple around every input container, so its
# depth is bounded by roughly twice the validated input depth.
_MAX_CANONICAL_DEPTH = 2 * _MAX_STATE_DEPTH + 8
_TEMPLATE_REPR_LIMIT = 256
_MAPPING_ENTRY_OVERHEAD = 64


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(name: str, value: Any, *, minimum: int = 0) -> int:
    if not _is_int(value) or value < minimum:
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    return value


def _require_float(
    name: str,
    value: Any,
    *,
    positive: bool = False,
    allow_infinite: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    number = float(value)
    if math.isnan(number) or (math.isinf(number) and not allow_infinite):
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    if positive and number <= 0:
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    return number


def _require_run_id(value: Any, *, name: str = "run_id") -> Hashable:
    # The frozen interface uses ``complete.sequence`` as the run id.  Keeping
    # this to concrete scalar keys avoids an accidental opaque/unhashable id.
    if not isinstance(value, (int, str)) or isinstance(value, bool):
        raise PeriodicContractError(f"periodic_{name}_invalid:{value!r}")
    return value


def _check_structural(
    value: Any,
    *,
    path: str = "normalized_state",
    depth: int = 0,
    counter: list[int] | None = None,
) -> None:
    """Reject opaque state so identity cannot silently degrade to a digest."""
    if counter is not None:
        counter[0] += 1
    if depth > _MAX_STATE_DEPTH:
        raise PeriodicContractError(f"periodic_state_too_deep:{path}")
    if value is None or isinstance(value, (bool, str, bytes)):
        return
    if _is_int(value):
        return
    if isinstance(value, float):
        if math.isnan(value):
            raise PeriodicContractError(f"periodic_state_nan:{path}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _check_structural(item, path=f"{path}[{index}]", depth=depth + 1, counter=counter)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _check_structural(key, path=f"{path}.key", depth=depth + 1, counter=counter)
            _check_structural(item, path=f"{path}[{key!r}]", depth=depth + 1, counter=counter)
        return
    if isinstance(value, (set, frozenset)):
        for item in value:
            _check_structural(item, path=f"{path}{{item}}", depth=depth + 1, counter=counter)
        return
    raise PeriodicContractError(f"periodic_state_opaque:{path}:{type(value).__name__}")


def _numeric_close(first: float, second: float) -> bool:
    """Equality for measured floats, allowing only a few representation ulps."""
    if first == second:
        return True
    if not (math.isfinite(first) and math.isfinite(second)):
        return False
    scale = max(abs(first), abs(second))
    return abs(first - second) <= _ULP_SLOP * math.ulp(scale)


def _structural_equal(first: Any, second: Any) -> bool:
    """Exact typed structural equality.

    Python ``==`` is deliberately not used as a shortcut: ``True``, ``1`` and
    ``1.0`` are distinct typed values, and the same mutable container may be
    observed at two different times.  Containers are compared recursively by
    exact type/content; sets and mappings are matched without relying on
    Python hash/equality merging.
    """
    if type(first) is not type(second):
        return False
    kind = type(first)
    if first is None:
        return True
    if kind is bool or kind is int or kind is str or kind is bytes:
        return first == second
    if kind is float:
        if math.isnan(first) or math.isnan(second):
            return False
        return first == second
    if isinstance(first, type):
        return first is second
    if kind is tuple or kind is list:
        return len(first) == len(second) and all(
            _structural_equal(left, right)
            for left, right in zip(first, second))
    if isinstance(first, Mapping):
        if len(first) != len(second):
            return False
        unmatched = list(second.items())
        for key, value in first.items():
            for index, (other_key, other_value) in enumerate(unmatched):
                if (_structural_equal(key, other_key)
                        and _structural_equal(value, other_value)):
                    del unmatched[index]
                    break
            else:
                return False
        return True
    if isinstance(first, (set, frozenset)):
        if len(first) != len(second):
            return False
        unmatched = list(second)
        for item in first:
            for index, other in enumerate(unmatched):
                if _structural_equal(item, other):
                    del unmatched[index]
                    break
            else:
                return False
        return True
    return False


def structural_equal(first: Any, second: Any) -> bool:
    """Public, exact structural comparison used by the detector and tests."""
    _check_structural(first)
    _check_structural(second)
    return _structural_equal(first, second)


def _first_difference(first: Any, second: Any, path: str = "state") -> str | None:
    """Return the first unequal structural component for adapter diagnostics."""
    if first is second:
        return None
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        if not (isinstance(first, Mapping) and isinstance(second, Mapping)):
            return f"{path}:mapping_type"
        if first.keys() != second.keys():
            left = sorted(map(repr, first.keys() - second.keys()))
            right = sorted(map(repr, second.keys() - first.keys()))
            return f"{path}.keys:left_only={left[:4]}:right_only={right[:4]}"
        for key in first:
            found = _first_difference(first[key], second[key], f"{path}[{key!r}]")
            if found is not None:
                return found
        return None
    if isinstance(first, (set, frozenset)) or isinstance(second, (set, frozenset)):
        if not (isinstance(first, (set, frozenset)) and isinstance(second, (set, frozenset))):
            return f"{path}:set_type"
        if first != second:
            return f"{path}:set_value"
        return None
    if type(first) is not type(second):
        return f"{path}:type:{type(first).__name__}!={type(second).__name__}"
    if isinstance(first, (tuple, list)):
        if len(first) != len(second):
            return f"{path}.length:{len(first)}!={len(second)}"
        for index, (left, right) in enumerate(zip(first, second)):
            found = _first_difference(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if first != second:
        return f"{path}:{first!r}!={second!r}"
    return None


def structural_difference(first: Any, second: Any) -> str | None:
    """Public first-difference diagnostic; values are already validated."""
    _check_structural(first)
    _check_structural(second)
    return _first_difference(first, second)


def _bounded_repr(value: Any) -> str:
    """Bound diagnostic text; display text is never used as identity."""
    try:
        text = repr(value)
    except Exception:  # pragma: no cover - builtin state only
        return "<unrepresentable>"
    if len(text) > _TEMPLATE_REPR_LIMIT:
        return text[:_TEMPLATE_REPR_LIMIT] + "..."
    return text


def _measure_retained(value: Any, *, seen: set[int] | None = None) -> int:
    """Approximate bytes owned by a validated value graph.

    Shared sub-objects inside one observation are counted once.  The result is
    a retention lower bound used for cache admission, not an RSS measurement.
    """
    if seen is None:
        seen = set()
    ident = id(value)
    if ident in seen:
        return 0
    seen.add(ident)
    try:
        size = sys.getsizeof(value)
    except TypeError:  # pragma: no cover - builtin state only
        size = 0
    kind = type(value)
    if kind is tuple or kind is list:
        for item in value:
            size += _measure_retained(item, seen=seen)
    elif kind is set or kind is frozenset:
        for item in value:
            size += _measure_retained(item, seen=seen)
    elif isinstance(value, Mapping):
        # ``MappingProxyType`` only reports its own small wrapper size, so the
        # payload/table share is charged per entry here.  Keys and values are
        # measured once per shared identity.
        size += _MAPPING_ENTRY_OVERHEAD * len(value)
        for key, item in value.items():
            size += _measure_retained(key, seen=seen)
            size += _measure_retained(item, seen=seen)
    return size


def _check_canonical_snapshot(
    value: Any, *, path: str = "template", depth: int = 0,
) -> None:
    """Reject anything that is not an internally tagged immutable type tree.

    ``from_validated`` is a public factory for an already validated snapshot.
    Rechecking its tags/container identities here keeps an external caller from
    bypassing the interner's type-exact boundary by supplying a raw builtin tree
    in the snapshot slot.
    """
    if depth > _MAX_CANONICAL_DEPTH:
        raise PeriodicContractError(f"periodic_snapshot_too_deep:{path}")
    if type(value) is not tuple or not value:
        raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
    tag = value[0]
    if type(tag) is not str:
        raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
    if tag == "none":
        if len(value) != 1:
            raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
        return
    if tag in {"bool", "int", "float", "str", "bytes"}:
        if len(value) != 2:
            raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
        payload = value[1]
        expected = {
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "bytes": bytes,
        }[tag]
        if type(payload) is not expected:
            raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
        if tag == "float" and math.isnan(payload):
            raise PeriodicContractError(f"periodic_snapshot_nan:{path}")
        return
    if tag in {"tuple", "list"}:
        if len(value) != 2 or type(value[1]) is not tuple:
            raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
        for index, item in enumerate(value[1]):
            _check_canonical_snapshot(
                item, path=f"{path}[{index}]", depth=depth + 1)
        return
    if tag == "dict" or tag == "mapping":
        if tag == "dict":
            if len(value) != 2 or type(value[1]) is not tuple:
                raise PeriodicContractError(
                    f"periodic_template_snapshot_invalid:{path}")
            items = value[1]
        else:
            # The real emitted class is retained for exact type identity.  Its
            # metaclass may legitimately be ``ABCMeta`` (e.g. UserDict), so the
            # check is "class object and a concrete Mapping subclass", never
            # ``type(cls) is type``.  A forged tag cannot smuggle in ``dict``
            # or an unrelated/abstract class.
            cls = value[1] if len(value) == 3 else None
            if (
                len(value) != 3
                or not isinstance(cls, type)
                or cls is dict
                or not issubclass(cls, Mapping)
                or getattr(cls, "__abstractmethods__", ())
                or type(value[2]) is not tuple
            ):
                raise PeriodicContractError(
                    f"periodic_template_snapshot_invalid:{path}")
            items = value[2]
        for index, pair in enumerate(items):
            if type(pair) is not tuple or len(pair) != 2:
                raise PeriodicContractError(
                    f"periodic_template_snapshot_invalid:{path}[{index}]")
            _check_canonical_snapshot(
                pair[0], path=f"{path}[{index}].key", depth=depth + 1)
            _check_canonical_snapshot(
                pair[1], path=f"{path}[{index}].value", depth=depth + 1)
        return
    if tag in {"set", "frozenset"}:
        if len(value) != 2 or type(value[1]) is not tuple:
            raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")
        for index, item in enumerate(value[1]):
            _check_canonical_snapshot(
                item, path=f"{path}{{item}}[{index}]", depth=depth + 1)
        return
    raise PeriodicContractError(f"periodic_template_snapshot_invalid:{path}")


def _freeze_snapshot(
    value: Any, *, path: str = "snapshot", depth: int = 0,
    counter: list[int] | None = None, sizes: list[int] | None = None,
) -> tuple:
    """Deep-freeze a validated value into an immutable typed canonical form.

    Every container carries an explicit type tag; bool/int/float are separate
    tags.  The returned tree contains only tuples and immutable scalars, so a
    later caller mutation cannot rewrite history and canonical ``==`` cannot
    merge Python-equal but differently typed values.

    ``sizes`` optionally accumulates the real ``sys.getsizeof`` total of the
    produced tree.  The tree is being built anyway, so charging each new node
    there costs one C call per node and lets a historical frame report its
    retained bytes without a second recursive pass over the same nodes.
    """
    if depth > _MAX_STATE_DEPTH:
        raise PeriodicContractError(f"periodic_snapshot_too_deep:{path}")
    if counter is not None:
        counter[0] += 1
    kind = type(value)
    if value is None:
        result = ("none",)
    elif kind is bool:
        result = ("bool", value)
    elif kind is int:
        result = ("int", value)
    elif kind is float:
        if math.isnan(value):
            raise PeriodicContractError(f"periodic_snapshot_nan:{path}")
        result = ("float", value)
    elif kind is str:
        result = ("str", value)
    elif kind is bytes:
        result = ("bytes", value)
    elif kind is tuple:
        result = ("tuple", tuple(
            _freeze_snapshot(item, path=f"{path}[{index}]", depth=depth + 1,
                             counter=counter, sizes=sizes)
            for index, item in enumerate(value)))
    elif kind is list:
        result = ("list", tuple(
            _freeze_snapshot(item, path=f"{path}[{index}]", depth=depth + 1,
                             counter=counter, sizes=sizes)
            for index, item in enumerate(value)))
    elif isinstance(value, Mapping):
        items = [
            (_freeze_snapshot(key, path=f"{path}.key", depth=depth + 1,
                              counter=counter, sizes=sizes),
             _freeze_snapshot(item, path=f"{path}[{key!r}]", depth=depth + 1,
                              counter=counter, sizes=sizes))
            for key, item in value.items()
        ]
        items.sort(key=lambda pair: repr(pair[0]))
        if kind is dict:
            result = ("dict", tuple(items))
        else:
            # Keep the real mapping class (identity, not a display name) so two
            # same-repr mapping types can never share a template key.
            result = ("mapping", kind, tuple(items))
    elif kind is set:
        result = ("set", tuple(sorted(
            (_freeze_snapshot(item, path=f"{path}{{item}}", depth=depth + 1,
                              counter=counter, sizes=sizes)
             for item in value),
            key=repr)))
    elif kind is frozenset:
        result = ("frozenset", tuple(sorted(
            (_freeze_snapshot(item, path=f"{path}{{item}}", depth=depth + 1,
                              counter=counter, sizes=sizes)
             for item in value),
            key=repr)))
    else:
        raise PeriodicContractError(
            f"periodic_snapshot_opaque:{path}:{kind.__name__}")
    if sizes is not None:
        sizes[0] += sys.getsizeof(result)
    return result
