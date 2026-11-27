"""Static and dynamic parts of the Engine frontier.

[Role] Split, validate and freeze the restricted Engine state shape.
[Invariants] The static key preserves real scalar types. Every changed dynamic
  field is validated and retained with its original type.
[Timeouts & Locks] Bounded by the supplied frontier. No locks or I/O.
[Side-Effect Policy] Builds immutable values and adjusts only caller-provided counters.
"""
from __future__ import annotations

from functools import lru_cache
import math
from typing import Any

from .snapshots import (
    PeriodicContractError,
    _EMPTY_TUPLE_NODE,
    _FALSE_NODE,
    _INT_TUPLE_MAX_LEN,
    _NONE_NODE,
    _SMALL_INT_LIMIT,
    _SMALL_INT_NODES,
    _TRUE_NODE,
    _TUPLE_BASE_SIZE,
    _TUPLE_PTR_SIZE,
    _ZERO_FLOAT_NODE,
    _check_structural,
    _freeze_int_tuple,
    _freeze_snapshot,
    _measure_retained,
)


_ENGINE_STATE_FIELDS = 5
_DYNAMIC_TASK_FIELDS = 4
_STATIC_TASK_NODE_OVERHEAD = 4
_PAIR_NODE_SIZE = _TUPLE_BASE_SIZE + _TUPLE_PTR_SIZE * 2
_SINGLE_NODE_SIZE = _TUPLE_BASE_SIZE + _TUPLE_PTR_SIZE
_REMAINING_ENTRY_SIZE = _PAIR_NODE_SIZE * 3
_TASK_NODE_BASE_SIZE = _PAIR_NODE_SIZE * 2
_BLOCKED_EMPTY_TASK_SIZE = (
    _TUPLE_BASE_SIZE + _PAIR_NODE_SIZE + _SINGLE_NODE_SIZE + _TASK_NODE_BASE_SIZE
)
_BLOCKED_EMPTY_TASK_NODES = tuple(
    ("tuple", (_EMPTY_TUPLE_NODE, _SMALL_INT_NODES[w], _ZERO_FLOAT_NODE, _NONE_NODE))
    for w in range(_SMALL_INT_LIMIT)
)
_PERIODIC_KEY_ARITY = 5
_OPAQUE_KEY_ARITY = 2
_STATIC_TASK_FIELDS = 6
_STATIC_NODE_CACHE_LIMIT = 512
_INTERN_STR_MAX_LEN = 64
_CAP_ENTRY_NODES = 3
_PLACEMENT_PAIR_NODES = 3
_TASK_SCALAR_NODES = 3
_INT_TYPE_SET = frozenset((int,))
_SIX_TUPLE_SIZE = _TUPLE_BASE_SIZE + _TUPLE_PTR_SIZE * _STATIC_TASK_FIELDS
_FIVE_TUPLE_SIZE = _TUPLE_BASE_SIZE + _TUPLE_PTR_SIZE * _PERIODIC_KEY_ARITY
_PAIR_BODY_SIZE = _TUPLE_BASE_SIZE + _TUPLE_PTR_SIZE * _OPAQUE_KEY_ARITY


@lru_cache(maxsize=_STATIC_NODE_CACHE_LIMIT)
def _intern_str_node(text: str) -> tuple[str, str]:
    """Share canonical ``("str", text)`` leaves for short resource/kind labels."""
    return ("str", text)


def _str_node(text: str) -> tuple[str, str]:
    """Intern only short labels so rejected oversized payloads never linger in caches."""
    return _intern_str_node(text) if len(text) <= _INTERN_STR_MAX_LEN else ("str", text)


def _clear_engine_state_caches() -> None:
    """Release all process-local canonical node caches on ``reset_state_caches()``."""
    _intern_str_node.cache_clear()
    _freeze_placement_pair.cache_clear()
    _freeze_placement_tuple.cache_clear()
    _freeze_int_tuple.cache_clear()


@lru_cache(maxsize=_STATIC_NODE_CACHE_LIMIT)
def _freeze_placement_pair(resource: str, units: tuple[int, ...]) -> tuple:
    """Cache canonical placement entry ``("tuple", (("str", r), ("tuple", ints)))``."""
    units_node = (
        _EMPTY_TUPLE_NODE
        if not units
        else (
            _freeze_int_tuple(units)
            if len(units) <= _INT_TUPLE_MAX_LEN
            else (
                "tuple",
                tuple(
                    _SMALL_INT_NODES[u] if 0 <= u < _SMALL_INT_LIMIT else ("int", u)
                    for u in units
                ),
            )
        )
    )
    return ("tuple", (_str_node(resource), units_node))


@lru_cache(maxsize=_STATIC_NODE_CACHE_LIMIT)
def _freeze_placement_tuple(placement: tuple[tuple[str, tuple[int, ...]], ...]) -> tuple:
    """Cache canonical placement tuple after exact-type validation."""
    if not placement:
        return _EMPTY_TUPLE_NODE
    return ("tuple", tuple(_freeze_placement_pair(r, u) for r, u in placement))


def _freeze_static_key_node(
    key: tuple, seen: set[int],
) -> tuple[tuple, int, int]:
    """Freeze a validated task key and return ``(node, node_count, retained_bytes)``."""
    kn = len(key)
    if kn == _OPAQUE_KEY_ARITY and type(key[0]) is str and type(key[1]) is str:
        n0 = _str_node(key[0])
        n1 = _str_node(key[1])
        added = _PAIR_NODE_SIZE + _PAIR_BODY_SIZE
        if id(n0) not in seen:
            added += _measure_retained(n0, seen=seen)
        if id(n1) not in seen:
            added += _measure_retained(n1, seen=seen)
        return ("tuple", (n0, n1)), _OPAQUE_KEY_ARITY + 1, added
    if (
        kn == _PERIODIC_KEY_ARITY
        and type(key[0]) is str
        and type(key[1]) is int
        and type(key[2]) is int
        and type(key[3]) is int
        and type(key[4]) is str
    ):
        v1, v2, v3 = key[1], key[2], key[3]
        n0 = _str_node(key[0])
        n1 = _SMALL_INT_NODES[v1] if 0 <= v1 < _SMALL_INT_LIMIT else ("int", v1)
        n2 = _SMALL_INT_NODES[v2] if 0 <= v2 < _SMALL_INT_LIMIT else ("int", v2)
        n3 = _SMALL_INT_NODES[v3] if 0 <= v3 < _SMALL_INT_LIMIT else ("int", v3)
        n4 = _str_node(key[4])
        added = _PAIR_NODE_SIZE + _FIVE_TUPLE_SIZE
        for leaf in (n0, n1, n2, n3, n4):
            if id(leaf) not in seen:
                added += _measure_retained(leaf, seen=seen)
        return ("tuple", (n0, n1, n2, n3, n4)), _PERIODIC_KEY_ARITY + 1, added
    sub = [0]
    node = _freeze_snapshot(key, path="template.key", counter=sub)
    return node, sub[0], _measure_retained(node, seen=seen)


def _freeze_static_caps_node(
    caps: tuple, seen: set[int],
) -> tuple[tuple, int, int]:
    """Freeze validated task caps and return ``(node, node_count, retained_bytes)``."""
    if not caps:
        added = _measure_retained(_EMPTY_TUPLE_NODE, seen=seen) if id(_EMPTY_TUPLE_NODE) not in seen else 0
        return _EMPTY_TUPLE_NODE, 1, added
    sub_caps = []
    nodes = 1
    added = _PAIR_NODE_SIZE
    for cname, camt in caps:
        k_node = _str_node(cname)
        if type(camt) is float:
            a_node = (
                _ZERO_FLOAT_NODE
                if (camt == 0.0 and not math.copysign(1.0, camt) < 0.0)
                else ("float", camt)
            )
        else:
            a_node = (
                _SMALL_INT_NODES[camt]
                if 0 <= camt < _SMALL_INT_LIMIT
                else ("int", camt)
            )
        sub_caps.append(("tuple", (k_node, a_node)))
        nodes += _CAP_ENTRY_NODES
        added += _PAIR_NODE_SIZE + _PAIR_BODY_SIZE
        if id(k_node) not in seen:
            added += _measure_retained(k_node, seen=seen)
        if id(a_node) not in seen:
            added += _measure_retained(a_node, seen=seen)
    inner = tuple(sub_caps)
    added += _TUPLE_BASE_SIZE + _TUPLE_PTR_SIZE * len(inner)
    return ("tuple", inner), nodes, added


def _freeze_static_placement_node(
    placement: tuple, seen: set[int],
) -> tuple[tuple, int, int]:
    """Freeze validated task placement and return ``(node, node_count, retained_bytes)``."""
    if not placement:
        added = _measure_retained(_EMPTY_TUPLE_NODE, seen=seen) if id(_EMPTY_TUPLE_NODE) not in seen else 0
        return _EMPTY_TUPLE_NODE, 1, added
    pl_nodes = 1
    for pair in placement:
        if (
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is str
            and len(pair[0]) <= _INTERN_STR_MAX_LEN
            and type(pair[1]) is tuple
            and len(pair[1]) <= _INT_TUPLE_MAX_LEN
            and (not pair[1] or set(map(type, pair[1])) == _INT_TYPE_SET)
        ):
            pl_nodes += _PLACEMENT_PAIR_NODES + len(pair[1])
        else:
            sub = [0]
            node = _freeze_snapshot(placement, path="template.placement", counter=sub)
            return node, sub[0], _measure_retained(node, seen=seen)
    pl_node = _freeze_placement_tuple(placement)
    added = _measure_retained(pl_node, seen=seen) if id(pl_node) not in seen else 0
    return pl_node, pl_nodes, added


def _freeze_static_owner_node(owner: Any, seen: set[int]) -> tuple[tuple, int]:
    """Freeze validated scalar owner and return ``(node, retained_bytes)``."""
    if owner is None:
        node = _NONE_NODE
    else:
        okind = type(owner)
        if okind is str:
            node = _str_node(owner)
        elif okind is int:
            node = _SMALL_INT_NODES[owner] if 0 <= owner < _SMALL_INT_LIMIT else ("int", owner)
        elif okind is bool:
            node = _TRUE_NODE if owner else _FALSE_NODE
        elif okind is float:
            node = (
                _ZERO_FLOAT_NODE
                if (owner == 0.0 and not math.copysign(1.0, owner) < 0.0)
                else ("float", owner)
            )
        else:
            node = ("bytes", owner)
    added = _measure_retained(node, seen=seen) if id(node) not in seen else 0
    return node, added


def _freeze_engine_static(static: tuple, counter: list[int]) -> tuple[tuple, int, set[int]]:
    """Freeze a validated Engine static tuple and return ``(snapshot, retained_bytes, seen)``."""
    nodes = 1
    seen: set[int] = set()
    # Account for the shared tag string ``"tuple"`` once, matching ``_measure_retained``.
    retained = _measure_retained("tuple", seen=seen)
    frozen_tasks = []
    for task in static:
        key, priority, caps, placement, coupled, owner = task
        nodes += _TASK_SCALAR_NODES
        key_node, k_nodes, k_bytes = _freeze_static_key_node(key, seen)
        nodes += k_nodes
        retained += k_bytes
        prio_node = (
            _SMALL_INT_NODES[priority]
            if 0 <= priority < _SMALL_INT_LIMIT
            else ("int", priority)
        )
        if id(prio_node) not in seen:
            retained += _measure_retained(prio_node, seen=seen)
        caps_node, c_nodes, c_bytes = _freeze_static_caps_node(caps, seen)
        nodes += c_nodes
        retained += c_bytes
        pl_node, p_nodes, p_bytes = _freeze_static_placement_node(placement, seen)
        nodes += p_nodes
        retained += p_bytes
        coupled_node = _TRUE_NODE if coupled else _FALSE_NODE
        if id(coupled_node) not in seen:
            retained += _measure_retained(coupled_node, seen=seen)
        owner_node, o_bytes = _freeze_static_owner_node(owner, seen)
        nodes += 1
        retained += o_bytes + _PAIR_NODE_SIZE + _SIX_TUPLE_SIZE
        frozen_tasks.append(
            ("tuple", (key_node, prio_node, caps_node, pl_node, coupled_node, owner_node))
        )
    tasks_tuple = tuple(frozen_tasks)
    retained += _PAIR_NODE_SIZE + (_TUPLE_BASE_SIZE + _TUPLE_PTR_SIZE * len(tasks_tuple))
    counter[0] += nodes
    return ("tuple", tasks_tuple), retained, seen


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

    track_size = sizes is not None
    added_size = 0
    seen_blocked_mask = 0
    frozen_tasks = []
    for entry in task_dynamic:
        if type(entry) is not tuple or len(entry) != _DYNAMIC_TASK_FIELDS:
            raise PeriodicContractError("periodic_engine_task_dynamic_invalid")
        remaining, waiting, release, start = entry
        if (
            remaining == ()
            and type(remaining) is tuple
            and type(waiting) is int
            and 0 <= waiting < _SMALL_INT_LIMIT
            and type(release) is float
            and release == 0.0
            and not math.copysign(1.0, release) < 0.0
            and start is None
        ):
            frozen_tasks.append(_BLOCKED_EMPTY_TASK_NODES[waiting])
            if track_size:
                bit = 1 << waiting
                if not (seen_blocked_mask & bit):
                    seen_blocked_mask |= bit
                    added_size += _BLOCKED_EMPTY_TASK_SIZE
            if counter is not None:
                counter[0] += _DYNAMIC_TASK_FIELDS
            continue
        if type(remaining) is not tuple:
            raise PeriodicContractError("periodic_engine_remaining_invalid")
        if remaining:
            frozen_remaining = []
            for pair in remaining:
                if type(pair) is not tuple or len(pair) != 2:
                    raise PeriodicContractError("periodic_engine_remaining_pair_invalid")
                key, amount = pair
                if type(key) is not str:
                    raise PeriodicContractError("periodic_engine_remaining_key_invalid")
                amt_type = type(amount)
                if amt_type is float:
                    if not math.isfinite(amount):
                        raise PeriodicContractError("periodic_engine_remaining_value_invalid")
                    amt_node = ("float", amount)
                elif amt_type is int:
                    amt_node = _SMALL_INT_NODES[amount] if 0 <= amount < _SMALL_INT_LIMIT else ("int", amount)
                else:
                    raise PeriodicContractError("periodic_engine_remaining_value_invalid")
                frozen_remaining.append(("tuple", (("str", key), amt_node)))
            rem_tuple = tuple(frozen_remaining)
            rem_node = ("tuple", rem_tuple)
            n_rem = len(rem_tuple)
            if counter is not None:
                counter[0] += n_rem * 2
            if track_size:
                added_size += n_rem * _REMAINING_ENTRY_SIZE + _TUPLE_BASE_SIZE + _TUPLE_PTR_SIZE * n_rem
        else:
            rem_node = _EMPTY_TUPLE_NODE
            if track_size:
                added_size += _TUPLE_BASE_SIZE
        if type(waiting) is not int or waiting < 0:
            raise PeriodicContractError("periodic_engine_waiting_invalid")
        wait_node = _SMALL_INT_NODES[waiting] if waiting < _SMALL_INT_LIMIT else ("int", waiting)
        if release is None:
            rel_node = _NONE_NODE
            if track_size:
                added_size += _SINGLE_NODE_SIZE
        else:
            rel_type = type(release)
            if rel_type is float:
                if not math.isfinite(release):
                    raise PeriodicContractError("periodic_engine_offset_invalid")
                rel_node = _ZERO_FLOAT_NODE if (release == 0.0 and not math.copysign(1.0, release) < 0.0) else ("float", release)
            elif rel_type is int:
                rel_node = _SMALL_INT_NODES[release] if 0 <= release < _SMALL_INT_LIMIT else ("int", release)
            else:
                raise PeriodicContractError("periodic_engine_offset_invalid")
            if track_size:
                added_size += _PAIR_NODE_SIZE
        if start is None:
            start_node = _NONE_NODE
            if track_size:
                added_size += _SINGLE_NODE_SIZE
        else:
            start_type = type(start)
            if start_type is float:
                if not math.isfinite(start):
                    raise PeriodicContractError("periodic_engine_offset_invalid")
                start_node = _ZERO_FLOAT_NODE if (start == 0.0 and not math.copysign(1.0, start) < 0.0) else ("float", start)
            elif start_type is int:
                start_node = _SMALL_INT_NODES[start] if 0 <= start < _SMALL_INT_LIMIT else ("int", start)
            else:
                raise PeriodicContractError("periodic_engine_offset_invalid")
            if track_size:
                added_size += _PAIR_NODE_SIZE
        frozen_tasks.append(("tuple", (rem_node, wait_node, rel_node, start_node)))
        if track_size:
            added_size += _TASK_NODE_BASE_SIZE
        if counter is not None:
            counter[0] += _DYNAMIC_TASK_FIELDS
    if track_size:
        sizes[0] += added_size
    frozen_tail = tuple(
        _freeze_snapshot(part, path="engine_dynamic_tail", counter=counter,
                         sizes=sizes)
        for part in (runs, allocations, pools, releases))
    # Charge exactly the reference-only containers this function creates on
    # the retained path: the task tuple, its tag wrapper, the concatenated
    # body and the outer result.  Leaves were charged above.
    tasks_tuple = tuple(frozen_tasks)
    tasks_wrapper = ("tuple", tasks_tuple)
    body = (tasks_wrapper,) + frozen_tail
    result = ("tuple", body)
    if track_size:
        sizes[0] += (
            (_TUPLE_BASE_SIZE + _TUPLE_PTR_SIZE * len(tasks_tuple))
            + _PAIR_NODE_SIZE
            + (_TUPLE_BASE_SIZE + _TUPLE_PTR_SIZE * len(body))
            + _PAIR_NODE_SIZE
        )
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
    if not value:
        counter[0] += 1
        return ()
    if type(value[0]) is int:
        all_int = True
        for item in value:
            if type(item) is not int:
                all_int = False
                break
        if all_int:
            # Exact-int fast path (e.g. 31-unit placement tuples): index-0 string
            # tag cannot collide with mixed tuple keys whose elements are tuples.
            counter[0] += len(value) + 1
            return ("ints", value)
    result = []
    for item in value:
        kind = type(item)
        if kind is tuple:
            result.append(_engine_tuple_key(item, counter))
        elif kind is int:
            result.append(_SMALL_INT_NODES[item] if 0 <= item < _SMALL_INT_LIMIT else ("int", item))
            counter[0] += 1
        elif kind is str:
            result.append(("str", item))
            counter[0] += 1
        else:
            raise PeriodicContractError("periodic_engine_static_key_invalid")
    counter[0] += 1
    return tuple(result)


def _engine_task_key(value: Any, counter: list[int]) -> tuple:
    """Fast-path exact key for Engine task keys (periodic 5-tuple or opaque 2-tuple)."""
    if type(value) is tuple:
        n = len(value)
        if n == _OPAQUE_KEY_ARITY and type(value[0]) is str and type(value[1]) is str:
            counter[0] += _OPAQUE_KEY_ARITY + 1
            return (("str", value[0]), ("str", value[1]))
        if (
            n == _PERIODIC_KEY_ARITY
            and type(value[0]) is str
            and type(value[1]) is int
            and type(value[2]) is int
            and type(value[3]) is int
            and type(value[4]) is str
        ):
            v1, v2, v3 = value[1], value[2], value[3]
            counter[0] += _PERIODIC_KEY_ARITY + 1
            return (
                ("str", value[0]),
                _SMALL_INT_NODES[v1] if 0 <= v1 < _SMALL_INT_LIMIT else ("int", v1),
                _SMALL_INT_NODES[v2] if 0 <= v2 < _SMALL_INT_LIMIT else ("int", v2),
                _SMALL_INT_NODES[v3] if 0 <= v3 < _SMALL_INT_LIMIT else ("int", v3),
                ("str", value[4]),
            )
    return _engine_tuple_key(value, counter)


def _engine_placement_key(value: Any, counter: list[int]) -> tuple:
    """Fast-path exact key for Engine placement ``((resource, (unit, ...)), ...)``."""
    if type(value) is not tuple:
        raise PeriodicContractError("periodic_engine_static_key_invalid")
    if not value:
        counter[0] += 1
        return ()
    result = []
    for pair in value:
        if (
            type(pair) is tuple
            and len(pair) == 2
            and type(pair[0]) is str
            and type(pair[1]) is tuple
        ):
            units = pair[1]
            if not units:
                result.append((("str", pair[0]), ()))
                counter[0] += _OPAQUE_KEY_ARITY + 1
                continue
            if units:
                if len(units) == 1:
                    if type(units[0]) is int:
                        result.append((_str_node(pair[0]), ("ints", units)))
                        counter[0] += 1 + _OPAQUE_KEY_ARITY + 1
                        continue
                elif set(map(type, units)) == _INT_TYPE_SET:
                    result.append((_str_node(pair[0]), ("ints", units)))
                    counter[0] += len(units) + _OPAQUE_KEY_ARITY + 1
                    continue
        return _engine_tuple_key(value, counter)
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
        if type(task) is not tuple or len(task) != _STATIC_TASK_FIELDS:
            raise PeriodicContractError("periodic_engine_static_task_invalid")
        key, priority, caps, placement, coupled, owner = task
        if type(priority) is not int or type(coupled) is not bool:
            raise PeriodicContractError("periodic_engine_static_task_invalid")
        result.append((
            _engine_task_key(key, counter),
            priority,
            () if (type(caps) is tuple and not caps) else _engine_caps_key(caps, counter),
            _engine_placement_key(placement, counter),
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
