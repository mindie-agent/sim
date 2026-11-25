"""Actual-access binding for one physical Execution.add occurrence.

[Role] Project the selected program's real per-occurrence/per-port state
  accesses into the existing SERVICE ``StateAccess`` fields while retaining
  the producer's complete binding metadata for T08/Memory.
[Invariants] An explicit empty tuple means true zero. A missing field comes
  from the corresponding producer, never from a layer-first entry, a global
  ``self.state_accesses`` slot or an arbitrary Event name. Requests are the
  actual non-padding lanes; byte/row spans keep their declared unit.
[Timeouts & Locks] Pure functions over the current IterationWork; no I/O.
[Side-Effect Policy] Reads OpSpec/IterationPlan and returns immutable records.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from mindie_sim.prediction.engine import Event

_PROJECTED_FIELDS = (
    "occurrence", "producer", "request", "stage", "layer", "direction",
    "position", "output", "storage", "state_version", "span", "span_format",
    "byte_span", "byte_span_format", "span_unit", "persistent", "temporary",
    "rank", "scope", "bytes", "source", "reader", "grant", "decision",
    "last_use",
)
# Producer-owned metadata that T08/Memory needs but SERVICE does not accept.
_MEMORY_FIELDS = (
    "unit", "row_stride", "row_span", "byte_span", "span_format", "line_bytes",
    "line_segments", "byte_segments", "ranges_complete", "coverage_bytes",
    "purpose", "action", "port", "owner", "persistent", "temporary",
    "representation", "tier", "rows", "request_slot", "physical_key",
    "generation", "host_bytes", "source_ready", "grant_ready",
    "decision_event", "last_use_event",
)
_CACHE_STORAGE_PREFIXES = ("kv:", "kv/", "gqa_kv", "draft_kv", "mla_kv", "paged_latent_kv", "index_pool_tail")


@dataclass(frozen=True)
class BoundAccess:
    """One real access plus the producer binding preserved for Memory."""

    access: Any
    memory: tuple[tuple[str, Any], ...] = ()

    def memory_dict(self) -> dict[str, Any]:
        return dict(self.memory)

    @property
    def key(self) -> tuple:
        return self.access.key


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item))
                            for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _request_rows(work: Any) -> tuple[str, tuple[str, ...], set[str]]:
    plan = getattr(work, "plan", None)
    views = tuple(getattr(plan, "requests", ()) or ())
    padding = {str(view.request_id) for view in views
               if getattr(view, "padding", False)}
    valid = tuple(str(view.request_id) for view in views
                  if not getattr(view, "padding", False))
    return "", valid, padding


def _first_valid(requests: Sequence[str], padding: set[str]) -> str:
    for request in requests:
        if str(request) not in padding:
            return str(request)
    return str(requests[0]) if requests else "*"


def _wrap_span_sequence(raw: Any) -> Any:
    """Wrap a single flat numeric span tuple into a 1-element sequence."""
    if isinstance(raw, (tuple, list)) and raw and isinstance(raw[0], int) and not isinstance(raw[0], bool):
        return (raw,)
    return raw


def _bytes_and_span(
    record: Mapping, op: Any, direction: str,
) -> tuple[float, tuple, str, tuple, str, str | None]:
    value = record.get("bytes")
    if value is None:
        value = record.get("write_bytes" if direction == "write" else "read_bytes")
    if value is None:
        row_bytes = record.get("row_bytes")
        rows = record.get("rows")
        if row_bytes is not None and rows is not None:
            value = float(row_bytes) * int(rows)
    if value is None:
        meta = dict(getattr(op, "metadata", ()) or ())
        value = meta.get("state_write_bytes" if direction == "write"
                         else "state_read_bytes", 0.0)
    from mindie_sim.prediction.offload.accesses import _canonical_span_format, _normalize_span
    span_format = str(record.get("span_format") or "half_open")
    byte_fmt = str(record.get("byte_span_format") or span_format)
    raw_byte_span = record.get("byte_span")
    norm_byte_span: tuple = ()
    canonical_byte_fmt = "half_open"
    if raw_byte_span:
        norm_byte_span = _normalize_span(_wrap_span_sequence(raw_byte_span), span_format=byte_fmt)
        canonical_byte_fmt = _canonical_span_format(norm_byte_span, byte_fmt)
    declared_unit = record.get("unit") or record.get("span_unit")
    span = record.get("span")
    if span:
        normalized = _normalize_span(_wrap_span_sequence(span), span_format=span_format)
        canonical_fmt = _canonical_span_format(normalized, span_format)
        return (
            float(value or 0.0), normalized, canonical_fmt,
            norm_byte_span, canonical_byte_fmt, declared_unit,
        )
    row_span = record.get("row_span")
    if row_span:
        normalized = _normalize_span(_wrap_span_sequence(row_span), span_format=span_format)
        canonical_fmt = _canonical_span_format(normalized, span_format)
        return (
            float(value or 0.0), normalized, canonical_fmt,
            norm_byte_span, canonical_byte_fmt, declared_unit or "rows",
        )
    if norm_byte_span:
        return (
            float(value or 0.0), norm_byte_span, canonical_byte_fmt,
            norm_byte_span, canonical_byte_fmt, declared_unit or "bytes",
        )
    return float(value or 0.0), (), "half_open", (), "half_open", declared_unit


def _normalise_one(record: Any, op: Any, work: Any, *,
                   occurrence: str) -> BoundAccess | None:
    from mindie_sim.prediction.offload import StateAccess
    if isinstance(record, StateAccess):
        data = record.to_dict()
    elif isinstance(record, Mapping):
        data = dict(record)
    else:
        data = {
            key: getattr(record, key)
            for key in _PROJECTED_FIELDS + _MEMORY_FIELDS
            if hasattr(record, key)
        }
    meta = dict(getattr(op, "metadata", ()) or ())
    _unused, valid, padding = _request_rows(work)
    stage = int(getattr(op, "stage", data.get("stage", 0)))
    layer = int(getattr(op, "layer", data.get("layer", -1)))
    raw_requests = getattr(op, "requests", ()) or data.get("requests", ())
    if isinstance(raw_requests, str):
        raw_requests = (raw_requests,)
    requests = tuple(str(item) for item in (raw_requests or ()))
    request = data.get("request")
    if request in (None, "", "*"):
        if len(valid) == 1:
            request = valid[0]
        elif len(requests) == 1:
            request = str(requests[0])
        else:
            request = _first_valid(requests, padding) if requests else "*"
    direction = str(data.get("direction") or "read")
    if direction not in {"read", "write", "alias"}:
        raise ValueError(f"state_access_direction_invalid:{direction}")
    storage = str(data.get("storage") or meta.get(
        "state_layout", "kv" if meta.get("cache_kind") else "state"))
    position = data.get("position")
    if not position:
        low_storage = storage.strip().lower()
        if (low_storage == "kv"
                or any(low_storage.startswith(prefix) for prefix in _CACHE_STORAGE_PREFIXES)
                or meta.get("cache_kind")):
            position = "cache"
        else:
            position = "local"
    output = str(data.get("output") or meta.get("cache_kind") or "")
    producer = str(data.get("producer") or occurrence)
    (bytes_value, span, canonical_fmt,
     norm_byte_span, canonical_byte_fmt, inferred_unit) = _bytes_and_span(data, op, direction)
    version = data.get("state_version")
    if version is None:
        key = ("state_write_versions" if direction == "write"
               else "state_read_versions")
        versions = meta.get(key)
        if isinstance(versions, (tuple, list)) and versions:
            version = versions[0]
    rank = data.get("rank")
    if rank is None:
        rank = meta.get("rank")
    scope = str(data.get("scope") or "target")
    temporary = bool(data.get("temporary", False))
    raw_persistent = data.get("persistent")
    persistent = (not temporary) if raw_persistent is None else bool(raw_persistent)
    access = StateAccess(
        occurrence=str(data.get("occurrence") or occurrence),
        producer=producer,
        request=str(request),
        stage=stage,
        layer=layer,
        direction=direction,
        position=str(position),
        output=output,
        storage=storage,
        state_version=(None if version is None else int(version)),
        span=span,
        span_format=canonical_fmt,
        byte_span=norm_byte_span,
        byte_span_format=canonical_byte_fmt,
        span_unit=str(inferred_unit or ""),
        persistent=persistent,
        temporary=temporary,
        rank=(None if rank is None else int(rank)),
        scope=scope,
        bytes=bytes_value,
        source=str(data.get("source") or ""),
        reader=str(data.get("reader") or ""),
        grant=str(data.get("grant") or ""),
        decision=str(data.get("decision") or ""),
        last_use=str(data.get("last_use") or ""),
    )
    memory = {key: _freeze(data[key]) for key in _MEMORY_FIELDS
              if key in data and data[key] is not None}
    if inferred_unit is not None and "unit" not in memory:
        memory["unit"] = str(inferred_unit)
    return BoundAccess(access, tuple(sorted(memory.items())))


def _fallback(op: Any, work: Any) -> tuple[BoundAccess, ...]:
    from mindie_sim.prediction.offload import _accesses_from_ops
    occurrence = str(getattr(op, "op_id", ""))
    projected = []
    for access in _accesses_from_ops(work, (op,)):
        data = access.to_dict()
        bound = _normalise_one(data, op, work, occurrence=occurrence)
        if bound is not None:
            projected.append(bound)
    return tuple(projected)


def _records_for_op(op: Any, work: Any, records: Any) -> tuple[BoundAccess, ...]:
    occurrence = str(getattr(op, "op_id", ""))
    return tuple(
        bound for bound in (
            _normalise_one(record, op, work, occurrence=occurrence)
            for record in tuple(records or ()))
        if bound is not None)


def bind_op_accesses(work: Any, op: Any, *,
                     state_accesses: Any = None) -> tuple[BoundAccess, ...]:
    """Bind the following order: explicit op entry, producer metadata, fallback.

    An explicit empty entry is true zero and never falls back. A field missing
    inside one access is completed by the corresponding producer record; a
    missing *field* is not the same as an explicitly empty access set.
    """
    occurrence = str(getattr(op, "op_id", ""))
    if state_accesses is not None:
        if isinstance(state_accesses, Mapping):
            if occurrence in state_accesses:
                return _records_for_op(op, work, state_accesses[occurrence])
        else:
            return _records_for_op(op, work, state_accesses)
    meta = dict(getattr(op, "metadata", ()) or ())
    if "state_accesses" in meta or "memory_accesses" in meta:
        return _records_for_op(
            op, work, meta.get("state_accesses", meta.get("memory_accesses")))
    return _fallback(op, work)


def bind_work_accesses(work: Any, physical_ops: Sequence[Any], *,
                       state_accesses: Any = None
                       ) -> dict[str, tuple[BoundAccess, ...]]:
    """Bind every actual physical occurrence to its real access tuple."""
    return {
        str(getattr(op, "op_id", "")):
            bind_op_accesses(work, op, state_accesses=state_accesses)
        for op in physical_ops
    }


def memory_bindings_for(op: Any, bindings: Any) -> dict[str, dict] | None:
    """Resolve this occurrence's per-allocation Memory bindings.

    Accepted producer shapes are a flat ``allocation -> binding`` mapping, an
    ``op_id -> {allocation: binding}`` mapping, or a sequence of binding rows
    carrying ``occurrence``/``op_id`` and ``allocation``/``name``.
    """
    if bindings is not None:
        occurrence = str(getattr(op, "op_id", ""))
        if isinstance(bindings, Mapping):
            if occurrence in bindings and isinstance(bindings[occurrence], Mapping):
                return {str(name): dict(value)
                        for name, value in bindings[occurrence].items()}
            flat = {str(name): dict(value) for name, value in bindings.items()
                    if isinstance(value, Mapping)}
            if flat and all(isinstance(value, Mapping)
                            for value in bindings.values()):
                return flat
        else:
            result = {}
            for row in tuple(bindings):
                if not isinstance(row, Mapping):
                    continue
                owner = str(row.get("occurrence") or row.get("op_id") or "")
                if owner and owner != occurrence:
                    continue
                name = row.get("allocation", row.get("name"))
                if name:
                    result[str(name)] = {
                        str(key): value for key, value in row.items()
                        if key not in {"occurrence", "op_id", "allocation", "name"}}
            if result:
                return result
    raw_meta = getattr(op, "metadata", ()) or ()
    if isinstance(raw_meta, Mapping):
        raw = raw_meta.get("memory_bindings", raw_meta.get("memory_binding"))
    else:
        raw = None
        for key, val in raw_meta:
            if key == "memory_bindings":
                raw = val
                break
            if key == "memory_binding" and raw is None:
                raw = val
    if isinstance(raw, Mapping):
        return {str(name): dict(value) for name, value in raw.items()
                if isinstance(value, Mapping)}
    return None


def event_for(value: Any, ids: Mapping[str, Event]) -> Event | None:
    """Resolve an actual control edge, never by guessing an Event name."""
    if isinstance(value, Event):
        return value
    if isinstance(value, str) and value in ids:
        return ids[value]
    return None


def mandatory_edges(access: StateAccess, ids: Mapping[str, Event],
                    memory: Mapping[str, Any] | None = None
                    ) -> tuple[Event, ...]:
    """Source-ready + destination-grant + decision edges for one access.

    Actual Event/string bindings from the producer are used first. A missing
    binding contributes no fabricated edge; the corresponding producer
    completion is used only when the producer is an actual submitted
    occurrence. All returned edges are a union, never a single global barrier.
    """
    memory = dict(memory or ())
    edges = []
    for key in ("source_ready", "grant_ready", "decision_event"):
        value = memory.get(key)
        edge = event_for(value, ids)
        if edge is not None:
            edges.append(edge)
    if access.source:
        source = event_for(access.source, ids)
        if source is not None:
            edges.append(source)
    if access.grant:
        grant = event_for(access.grant, ids)
        if grant is not None:
            edges.append(grant)
    if access.decision:
        decision = event_for(access.decision, ids)
        if decision is not None:
            edges.append(decision)
    if not access.source and not memory.get("source_ready"):
        producer = event_for(access.producer, ids)
        if producer is not None:
            edges.append(producer)
    return tuple(dict.fromkeys(edges))


def service_projection(bound: BoundAccess):
    """Only the fields the existing SERVICE helpers accept."""
    return bound.access
