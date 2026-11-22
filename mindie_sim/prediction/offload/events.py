"""Event ownership for state offload accesses.

[Role] Bind and release transfer and producer events for iteration work.
[Invariants] Event bindings follow the access identity and work lifetime.
[Timeouts & Locks] Bounded by active work. No I/O, waits or locks.
[Side-Effect Policy] Updates the caller's access registry.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import weakref

from .accesses import StateAccess, _accesses_from_ops, _unique_accesses


def _work_ref(work: Any):
    try:
        return weakref.ref(work)
    except TypeError:
        def _strong() -> Any:
            return work
        return _strong


def _access_registry(execution: Any, work: Any) -> dict:
    """Issued-access registry for one actual IterationWork invocation.

    The same work shares layer/draft/fallback accounting; a later batch has a
    new work object and is charged independently even for the same request or
    op name.  A weak reference prevents the registry from owning the batch.
    """
    batches = getattr(execution, "_offload_access_batches", None)
    if batches is None:
        batches = {}
        execution._offload_access_batches = batches
    for key, (reference, _registry) in list(batches.items()):
        if reference() is None:
            batches.pop(key, None)
    entry = batches.get(id(work))
    if entry is not None and entry[0]() is work:
        return entry[1]
    registry: dict = {}
    batches[id(work)] = (_work_ref(work), registry)
    return registry


def _accesses_for_work(execution: Any, work: Any) -> tuple[StateAccess, ...]:
    """One actual access set for all four paths of this IterationWork."""
    bindings = getattr(execution, "_offload_actual_accesses", None)
    if bindings is None:
        bindings = {}
        execution._offload_actual_accesses = bindings
    for key, (reference, _accesses) in list(bindings.items()):
        if reference() is None:
            bindings.pop(key, None)
    entry = bindings.get(id(work))
    if entry is not None and entry[0]() is work:
        return entry[1]
    accesses = _accesses_from_ops(work)
    bindings[id(work)] = (_work_ref(work), accesses)
    return accesses


def release_offload_access(execution: Any, work: Any) -> int:
    """Drop one batch's issued registry and actual Event bindings.

    This is the explicit batch-end hook T04 calls after real last use.  Event
    history must not accumulate with the number of iterations.
    """
    batches = getattr(execution, "_offload_access_batches", None) or {}
    entry = batches.pop(id(work), None)
    bindings = getattr(execution, "_offload_actual_accesses", None)
    if bindings is not None:
        bindings.pop(id(work), None)
    event_bindings = getattr(execution, "_offload_access_events", None)
    if event_bindings is not None:
        event_bindings.pop(id(work), None)
    return len(entry[1]) if entry is not None else 0


def _event_table(source: Any) -> dict:
    """Normalize actual access->Event bindings without inspecting Event.name.

    Accepted shapes are an ``access key``/``StateAccess``/``(producer,
    output)``/``producer`` mapping, or an iterable of rows carrying ``event``
    plus those identity fields.  This is the same physical identity used by
    ``StateAccess.key``; display names are irrelevant.
    """
    table = {"key": {}, "producer_output": {}, "producer": {}}
    if source is None:
        return table
    if isinstance(source, Mapping):
        items = source.items()
    else:
        items = ((None, row) for row in source)
    for key, value in items:
        if key is None:
            if not isinstance(value, Mapping):
                continue
            row = dict(value)
            event = row.get("event")
            if event is None:
                continue
            access = row.get("access")
            if isinstance(access, StateAccess):
                table["key"][access.key] = event
                continue
            identity = row.get("key")
            if isinstance(identity, (tuple, list)) and identity:
                table["key"][tuple(identity)] = event
                continue
            producer = str(row.get("producer") or "")
            output = str(row.get("output") or "")
            occurrence = str(row.get("occurrence") or "")
            if producer and output:
                table["producer_output"][(producer, output)] = event
            elif producer:
                table["producer"][producer] = event
            elif occurrence:
                table["producer"][occurrence] = event
            continue
        if value is None:
            continue
        if isinstance(key, StateAccess):
            table["key"][key.key] = value
        elif isinstance(key, (tuple, list)):
            identity = tuple(key)
            if (len(identity) == 2
                    and all(isinstance(item, str) for item in identity)):
                table["producer_output"][identity] = value
            else:
                table["key"][identity] = value
        elif isinstance(key, str):
            table["producer"][key] = value
    return table


def _merge_event_tables(*tables) -> dict:
    merged = {"key": {}, "producer_output": {}, "producer": {}}
    for table in tables:
        if not table:
            continue
        for field in merged:
            merged[field].update(table.get(field, {}))
    return merged


def _has_event_bindings(table: Mapping | None) -> bool:
    return bool(table and any(table.get(field) for field in
                              ("key", "producer_output", "producer")))


def _event_for_access(access: StateAccess, table: Mapping | None) -> Any:
    """Resolve the actual producer.output Event for one physical access."""
    if not table:
        return None
    event = table.get("key", {}).get(access.key)
    if event is not None:
        return event
    event = table.get("producer_output", {}).get(
        (access.producer, access.output))
    if event is not None:
        return event
    return table.get("producer", {}).get(access.producer)


def bind_offload_access_events(execution: Any, work: Any, bindings: Any) -> int:
    """Bind the same batch's actual producer.output Events once.

    All four schedule_* paths consume this table through their optional
    ``access_events`` argument or this explicit per-work registration.  A
    later real output can be added before the unissued write is submitted.
    """
    events = getattr(execution, "_offload_access_events", None)
    if events is None:
        events = {}
        execution._offload_access_events = events
    for key, (reference, _table) in list(events.items()):
        if reference() is None:
            events.pop(key, None)
    current = _work_event_table(execution, work)
    table = _merge_event_tables(current, _event_table(bindings))
    if not _has_event_bindings(table):
        return 0
    events[id(work)] = (_work_ref(work), table)
    return sum(len(rows) for rows in table.values())


def _work_event_table(execution: Any, work: Any) -> dict:
    events = getattr(execution, "_offload_access_events", None)
    if not events:
        return {}
    entry = events.get(id(work))
    if entry is None or entry[0]() is not work:
        return {}
    return entry[1]


def _registered_events(execution: Any, work: Any, accesses) -> tuple[Any, ...]:
    registry = _access_registry(execution, work)
    events = []
    for access in _unique_accesses(accesses):
        events.extend(registry.get(access.key, ()))
    return tuple(dict.fromkeys(events))


def _register_access_events(execution: Any, work: Any, accesses, events) -> None:
    if not events:
        return
    registry = _access_registry(execution, work)
    for access in _unique_accesses(accesses):
        registry[access.key] = tuple(events)
