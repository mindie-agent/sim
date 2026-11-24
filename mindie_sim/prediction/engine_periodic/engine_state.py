"""Static and dynamic parts of the Engine frontier.

[Role] Split, validate and freeze the restricted Engine state shape.
[Invariants] The static key preserves real scalar types. Every changed dynamic
  field is validated and retained with its original type.
[Timeouts & Locks] Bounded by the supplied frontier. No locks or I/O.
[Side-Effect Policy] Builds immutable values and adjusts only caller-provided counters.
"""
from __future__ import annotations

import math
import sys
from typing import Any

from .snapshots import PeriodicContractError, _check_structural, _freeze_snapshot


_ENGINE_STATE_FIELDS = 5
_DYNAMIC_TASK_FIELDS = 4
_STATIC_TASK_NODE_OVERHEAD = 4



def _freeze_engine_dynamic(
    value: Any, counter: list[int] | None = None,
    sizes: list[int] | None = None,
) -> tuple:
    """Validate and freeze the known engine dynamic residue in one pass.

    ``sizes`` accumulates the real retained size of the frozen tree as it is
    produced.  A historical engine frame therefore knows its byte cost without
    a second recursive ``sys.getsizeof`` walk over the same nodes.
    """
    if type(value) is not tuple or len(value) != _ENGINE_STATE_FIELDS:
        raise PeriodicContractError("periodic_engine_dynamic_shape_invalid")
    task_dynamic, runs, allocations, pools, releases = value
    for part in (task_dynamic, runs, allocations, pools, releases):
        if type(part) is not tuple:
            raise PeriodicContractError("periodic_engine_dynamic_part_invalid")
    if counter is not None:
        counter[0] += _ENGINE_STATE_FIELDS

    def charged(node):
        # Charge the node we just built, in the same pass.  This is the real
        # sys.getsizeof total of the retained frozen tree, not an estimate.
        if sizes is not None:
            sizes[0] += sys.getsizeof(node)
        return node

    frozen_tasks = []
    for entry in task_dynamic:
        if type(entry) is not tuple or len(entry) != _DYNAMIC_TASK_FIELDS:
            raise PeriodicContractError("periodic_engine_task_dynamic_invalid")
        remaining, waiting, release, start = entry
        if type(remaining) is not tuple:
            raise PeriodicContractError("periodic_engine_remaining_invalid")
        frozen_remaining = []
        for pair in remaining:
            if type(pair) is not tuple or len(pair) != 2:
                raise PeriodicContractError("periodic_engine_remaining_pair_invalid")
            key, amount = pair
            if type(key) is not str:
                raise PeriodicContractError("periodic_engine_remaining_key_invalid")
            if type(amount) is not int and type(amount) is not float:
                raise PeriodicContractError("periodic_engine_remaining_value_invalid")
            if type(amount) is float and not math.isfinite(amount):
                raise PeriodicContractError("periodic_engine_remaining_value_invalid")
            amount_tag = "float" if type(amount) is float else "int"
            frozen_remaining.append(charged(
                ("tuple", (charged(("str", key)),
                           charged((amount_tag, amount))))))
            if counter is not None:
                counter[0] += 2
        if type(waiting) is not int or waiting < 0:
            raise PeriodicContractError("periodic_engine_waiting_invalid")
        frozen_offsets = []
        for offset in (release, start):
            if offset is None:
                frozen_offsets.append(charged(("none",)))
                continue
            if type(offset) is not int and type(offset) is not float:
                raise PeriodicContractError("periodic_engine_offset_invalid")
            if type(offset) is float and not math.isfinite(offset):
                raise PeriodicContractError("periodic_engine_offset_invalid")
            offset_tag = "float" if type(offset) is float else "int"
            frozen_offsets.append(charged((offset_tag, offset)))
        frozen_tasks.append(charged(
            ("tuple", (
                ("tuple", charged(tuple(frozen_remaining))),
                charged(("int", waiting)),
                frozen_offsets[0],
                frozen_offsets[1],
            ))))
        if counter is not None:
            counter[0] += _DYNAMIC_TASK_FIELDS
    frozen_tail = tuple(
        _freeze_snapshot(part, path="engine_dynamic_tail", counter=counter,
                         sizes=sizes)
        for part in (runs, allocations, pools, releases))
    # Charge exactly the reference-only containers this function creates on
    # the retained path: the task tuple, its tag wrapper, the 1-tuple head, the
    # concatenated body and the outer result.  Leaves were charged above.
    tasks_tuple = charged(tuple(frozen_tasks))
    tasks_wrapper = charged(("tuple", tasks_tuple))
    body = charged((tasks_wrapper,) + frozen_tail)
    result = charged(("tuple", body))
    return result


def _engine_scalar_key(value: Any, counter: list[int], *, path: str) -> tuple:
    """Type-exact scalar key used by the restricted Engine static schema."""
    counter[0] += 1
    kind = type(value)
    if value is None:
        return ("none",)
    if kind is bool:
        return ("bool", value)
    if kind is int:
        return ("int", value)
    if kind is float:
        if not math.isfinite(value):
            raise PeriodicContractError(
                f"periodic_engine_static_invalid:{path}")
        return ("float", value)
    if kind is str:
        return ("str", value)
    if kind is bytes:
        return ("bytes", value)
    raise PeriodicContractError(f"periodic_engine_static_invalid:{path}")


def _engine_caps_key(value: Any, counter: list[int]) -> tuple:
    if type(value) is not tuple:
        raise PeriodicContractError("periodic_engine_static_caps_invalid")
    result = []
    for pair in value:
        if type(pair) is not tuple or len(pair) != 2:
            raise PeriodicContractError("periodic_engine_static_caps_invalid")
        name, amount = pair
        if type(name) is not str:
            raise PeriodicContractError("periodic_engine_static_caps_invalid")
        if type(amount) is int:
            amount_key = ("int", amount)
        elif type(amount) is float:
            if not math.isfinite(amount):
                raise PeriodicContractError(
                    "periodic_engine_static_caps_invalid")
            amount_key = ("float", amount)
        else:
            raise PeriodicContractError("periodic_engine_static_caps_invalid")
        result.append((("str", name), amount_key))
        counter[0] += 2
    return tuple(result)


def _engine_tuple_key(value: Any, counter: list[int]) -> tuple:
    """Exact key for small tuple identity/placement fields of the static schema."""
    if type(value) is not tuple:
        raise PeriodicContractError("periodic_engine_static_key_invalid")
    result = []
    for item in value:
        if type(item) is tuple:
            result.append(_engine_tuple_key(item, counter))
        elif type(item) is int:
            result.append(("int", item))
            counter[0] += 1
        elif type(item) is str:
            result.append(("str", item))
            counter[0] += 1
        else:
            raise PeriodicContractError("periodic_engine_static_key_invalid")
    counter[0] += 1
    return tuple(result)


def _engine_static_key(static: Any, counter: list[int]) -> tuple:
    """Build the reusable exact lookup key for the restricted Engine static part.

    It validates the known schema directly and tags every scalar with its real
    builtin type, so equal-looking ``bool``/``int``/``float`` values can never
    share a key.  A hit therefore needs no canonical-tree rebuild: the cached
    fixed tree is returned as-is.
    """
    if type(static) is not tuple:
        raise PeriodicContractError("periodic_engine_static_shape_invalid")
    result = []
    for task in static:
        if type(task) is not tuple or len(task) != 6:
            raise PeriodicContractError("periodic_engine_static_task_invalid")
        key, priority, caps, placement, coupled, owner = task
        if type(priority) is not int or type(coupled) is not bool:
            raise PeriodicContractError("periodic_engine_static_task_invalid")
        result.append((
            _engine_tuple_key(key, counter),
            priority,
            _engine_caps_key(caps, counter),
            _engine_tuple_key(placement, counter),
            coupled,
            _engine_scalar_key(owner, counter, path="owner"),
        ))
        counter[0] += _STATIC_TASK_NODE_OVERHEAD
    return tuple(result)


def split_engine_state(state: tuple) -> tuple[tuple, tuple] | None:
    """Split the current Engine structural 5-tuple into static/dynamic parts.

    The known form is the T04 adapter's tile-index-independent state:
    ``(tasks, runs, allocations, pools, releases)`` with each task
    ``(key, remaining, waiting, release, start, priority, caps, placement,
    coupled, owner)``.  The static fields are interned; only remaining/waiting/
    release/start plus the small runs/allocations/pools/releases part is
    validated on every frame.  Any other shape, or static data that is not
    hashable/immutable, returns ``None`` and keeps the full validation path.
    """
    if type(state) is not tuple or len(state) != _ENGINE_STATE_FIELDS:
        return None
    tasks, runs, allocations, pools, releases = state
    if not all(type(part) is tuple for part in (tasks, runs, allocations, pools, releases)):
        return None
    static_parts = []
    dynamic_parts = []
    for task in tasks:
        if type(task) is not tuple or len(task) != 10:
            return None
        (key, remaining, waiting, release, start, priority, caps, placement, coupled, owner) = task
        static_parts.append((key, priority, caps, placement, coupled, owner))
        dynamic_parts.append((remaining, waiting, release, start))
    static = tuple(static_parts)
    try:
        hash(static)
    except TypeError:
        return None
    dynamic = (tuple(dynamic_parts), runs, allocations, pools, releases)
    return static, dynamic
