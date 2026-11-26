"""State access records and DDR/SSD tier projections.

[Role] Normalize state access identities and project tier traffic.
[Invariants] Accesses retain their physical producers, storage and ranges.
[Timeouts & Locks] Bounded by selected operators and active accesses. No waits or locks.
[Side-Effect Policy] Builds records from declared state and operator metadata.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from .policy import OffloadPolicy


_INDEX_POOL_TAIL_COMPONENT_BYTES = 4
_STRIDED_SPAN_ARITY = 4
_RECURRENT_STATE_3D_RANK = 3
_RECURRENT_STATE_4D_RANK = 4
_CANONICAL_SPAN_EXPAND_LIMIT = 65536
_PERIODIC_MIN_FULL_PERIODS = 3
_SPAN_FORMATS = frozenset({"half_open", "start_count", "start_width_stride_count"})
_PERSISTENT_STATE_POSITIONS = frozenset({"cache", "state", "tail", "layout"})
_TRANSIENT_STORAGE_PREFIXES = (
    "moe.act:", "moe.rx:", "act:", "activation:", "scratch:", "workspace:",
    "tmp:", "temp:", "caller:",
)
_TRANSIENT_STORAGE_SUFFIXES = (":staging", "@candidate", ":tmp", ":temp")


def is_persistent_offload_access(access: "StateAccess") -> bool:
    """True strictly for persistent cache/state storage eligible for DDR/SSD offload."""
    if access.temporary or not access.persistent:
        return False
    if str(access.position) not in _PERSISTENT_STATE_POSITIONS:
        return False
    storage = str(access.storage or "").strip().lower()
    if any(storage.startswith(prefix) for prefix in _TRANSIENT_STORAGE_PREFIXES):
        return False
    if any(storage.endswith(suffix) for suffix in _TRANSIENT_STORAGE_SUFFIXES):
        return False
    return True


def _canonical_span_format(normalized: tuple[tuple[int, ...], ...], fmt: str) -> str:
    """Canonical span format after normalizing contiguous spans to ``half_open``."""
    if fmt == "start_width_stride_count" and any(
        len(item) == _STRIDED_SPAN_ARITY for item in normalized
    ):
        return "start_width_stride_count"
    return "half_open"


def _normalize_span(
    value, *, span_format: str = "half_open",
) -> tuple[tuple[int, ...], ...]:
    """Stable hashable span identity across half_open, start_count, and strided forms.

    - ``"half_open"``: ``(start, end)`` pairs with ``0 <= start <= end``.
    - ``"start_count"``: ``(start, count)`` pairs with ``start >= 0, count >= 0``,
      converted once at the boundary to canonical half-open ``(start, start + count)``.
    - ``"start_width_stride_count"``: symbolic ``(start, width, stride, count)``
      descriptors with ``start >= 0, width > 0, count > 0, stride >= width``;
      when every descriptor is contiguous (``count == 1`` or ``stride == width``),
      canonicalizes directly to ``half_open`` ``(start, start + width * count)``.
    """
    fmt = str(span_format or "half_open")
    if fmt not in _SPAN_FORMATS:
        raise ValueError(f"state_access_span_format_invalid:{fmt}")
    if not value:
        return ()
    if fmt == "start_width_stride_count":
        quads: list[tuple[int, int, int, int]] = []
        for item in value:
            if isinstance(item, (str, bytes)):
                raise ValueError("state_access_span_range_invalid")
            raw = tuple(item)
            if any(isinstance(part, bool) for part in raw):
                raise ValueError("state_access_span_range_invalid")
            if len(raw) != _STRIDED_SPAN_ARITY:
                raise ValueError("state_access_span_strided_requires_quad")
            start, width, stride, count = (
                int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]),
            )
            if start < 0 or width <= 0 or count <= 0 or stride < width:
                raise ValueError("state_access_span_range_invalid")
            quads.append((start, width, width if count == 1 else stride, count))
        if all(count == 1 or stride == width for _, width, stride, count in quads):
            return tuple((start, start + width * count) for start, width, _, count in quads)
        return tuple(quads)
    normalized: list[tuple[int, ...]] = []
    for item in value:
        if isinstance(item, (str, bytes)):
            raise ValueError("state_access_span_range_invalid")
        raw = tuple(item)
        if any(isinstance(part, bool) for part in raw):
            raise ValueError("state_access_span_range_invalid")
        if len(raw) != 2:
            raise ValueError("state_access_span_range_requires_pair")
        first, second = int(raw[0]), int(raw[1])
        if fmt == "start_count":
            if first < 0 or second < 0:
                raise ValueError("state_access_span_range_invalid")
            normalized.append((first, first + second))
        else:
            if first < 0 or second < first:
                raise ValueError("state_access_span_range_invalid")
            normalized.append((first, second))
    return tuple(normalized)


def _run_end(run: tuple[int, int, int, int]) -> int:
    start, width, stride, count = run
    return start + (count - 1) * stride + width


def _pop_first_interval(
    run: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    start, width, stride, count = run
    if count <= 1:
        return None
    rem_count = count - 1
    return (start + stride, width, width if rem_count == 1 else stride, rem_count)


def _merge_boundary_touching(
    cluster: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    merged: list[tuple[int, int, int, int]] = []
    for run in cluster:
        start, width, stride, count = run
        if merged and _run_end(merged[-1]) == start:
            ps, pw, pd, pc = merged.pop()
            if pc > 1:
                prefix_count = pc - 1
                merged.append((ps, pw, pw if prefix_count == 1 else pd, prefix_count))
                tail_start = ps + prefix_count * pd
            else:
                tail_start = ps
            combined_width = pw + width
            merged.append((tail_start, combined_width, combined_width, 1))
            rest = _pop_first_interval((start, width, stride, count))
            if rest is not None:
                merged.append(rest)
        else:
            merged.append(run)
    return merged


def _coalesce_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    coalesced: list[tuple[int, int]] = []
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            coalesced.append((cur_start, cur_end))
            cur_start, cur_end = start, end
    coalesced.append((cur_start, cur_end))
    return coalesced


def _intervals_to_runs(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    return [(start, end - start, end - start, 1) for start, end in intervals]


def _zone_intervals(
    runs: list[tuple[int, int, int, int]], lo: int, hi: int,
) -> list[tuple[int, int]]:
    raw: list[tuple[int, int]] = []
    for start, width, stride, count in runs:
        if start >= hi or _run_end((start, width, stride, count)) <= lo:
            continue
        if count == 1:
            raw.append((max(start, lo), min(start + width, hi)))
            continue
        k_lo = max(0, (lo - start - width) // stride + 1)
        k_hi = min(count - 1, (hi - 1 - start) // stride)
        for idx in range(k_lo, k_hi + 1):
            a = start + idx * stride
            raw.append((max(a, lo), min(a + width, hi)))
    return _coalesce_intervals(raw)


def _uniform_periodic_runs(
    tile: list[tuple[int, int]], period: int, repeat_count: int,
) -> list[tuple[int, int, int, int]]:
    if not tile or repeat_count <= 0:
        return []
    if len(tile) == 1:
        a, b = tile[0]
        width = b - a
        return [(a, width, period if repeat_count > 1 else width, repeat_count)]
    first_width = tile[0][1] - tile[0][0]
    sub_stride = tile[1][0] - tile[0][0]
    uniform = (
        len(tile) * sub_stride == period
        and all(end - start == first_width for start, end in tile)
        and all(
            tile[i + 1][0] - tile[i][0] == sub_stride
            for i in range(len(tile) - 1)
        )
    )
    if uniform:
        return [(tile[0][0], first_width, sub_stride, len(tile) * repeat_count)]
    return [
        (start, end - start, period if repeat_count > 1 else end - start, repeat_count)
        for start, end in tile
    ]


def _merge_periodic_zone_runs(
    runs: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merge same-track continuations and adjacent touching runs after periodic zoning."""
    merged: list[tuple[int, int, int, int]] = []
    track_index: dict[tuple[int, int, int], int] = {}
    for run in _merge_boundary_touching(runs):
        start, width, stride, count = run
        if count >= 2:
            track = (width, stride, start % stride)
            prev_idx = track_index.get(track)
            if prev_idx is not None:
                ps, pw, pd, pc = merged[prev_idx]
                if start == ps + pc * pd:
                    merged[prev_idx] = (ps, pw, pd, pc + count)
                    continue
            track_index[track] = len(merged)
        merged.append(run)
    return merged


def _coalesce_periodic_cluster(
    cluster: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]] | None:
    """Coalesce arbitrarily large periodic/interleaved strided clusters in O(zones * period_work)."""
    strides = sorted({stride for _, width, stride, count in cluster if count >= 2 and stride > width})
    if not strides:
        return None
    period = math.lcm(*strides)
    work_per_period = sum(
        period // stride for _, width, stride, count in cluster if count >= 2 and stride > width
    )
    if period <= 0 or work_per_period > _CANONICAL_SPAN_EXPAND_LIMIT:
        return None
    points = sorted({pt for run in cluster for pt in (run[0], _run_end(run))})
    emitted: list[tuple[int, int, int, int]] = []
    for lo, hi in zip(points, points[1:]):
        active = [run for run in cluster if run[0] < hi and _run_end(run) > lo]
        if not active:
            continue
        if any(run[3] == 1 for run in active):
            emitted.append((lo, hi - lo, hi - lo, 1))
            continue
        full_periods = (hi - lo) // period
        if full_periods < _PERIODIC_MIN_FULL_PERIODS:
            emitted.extend(_intervals_to_runs(_zone_intervals(active, lo, hi)))
            continue
        mid_lo = lo + period
        two_p = _zone_intervals(active, mid_lo, mid_lo + 2 * period)
        if not two_p:
            emitted.extend(_intervals_to_runs(_zone_intervals(active, lo, mid_lo)))
            emitted.extend(_intervals_to_runs(_zone_intervals(active, lo + (full_periods - 1) * period, hi)))
            continue
        if len(two_p) == 1 and two_p[0] == (mid_lo, mid_lo + 2 * period):
            mid_hi = lo + (full_periods - 1) * period
            emitted.extend(_intervals_to_runs(_zone_intervals(active, lo, mid_lo)))
            emitted.append((mid_lo, mid_hi - mid_lo, mid_hi - mid_lo, 1))
            emitted.extend(_intervals_to_runs(_zone_intervals(active, mid_hi, hi)))
            continue
        gap = mid_lo if two_p[0][0] > mid_lo else two_p[0][1]
        mid_periods = (hi - gap) // period - 1
        if mid_periods <= 0:
            emitted.extend(_intervals_to_runs(_zone_intervals(active, lo, hi)))
            continue
        mid_hi = gap + mid_periods * period
        emitted.extend(_intervals_to_runs(_zone_intervals(active, lo, gap)))
        emitted.extend(_uniform_periodic_runs(_zone_intervals(active, gap, gap + period), period, mid_periods))
        emitted.extend(_intervals_to_runs(_zone_intervals(active, mid_hi, hi)))
    return _merge_periodic_zone_runs(emitted)


def _coalesce_run_cluster(
    cluster: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    if len(cluster) <= 1:
        return cluster
    if sum(run[3] for run in cluster) <= _CANONICAL_SPAN_EXPAND_LIMIT:
        intervals = [
            (start + idx * stride, start + idx * stride + width)
            for start, width, stride, count in cluster
            for idx in range(count)
        ]
        return _intervals_to_runs(_coalesce_intervals(intervals))
    if all(_run_end(left) == right[0] for left, right in zip(cluster, cluster[1:])):
        return _merge_boundary_touching(cluster)
    periodic = _coalesce_periodic_cluster(cluster)
    if periodic is not None:
        return periodic
    return sorted(set(cluster))


def _canonical_span_key(
    descriptors: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, int, int, int], ...]:
    """Canonical representation invariant to half_open vs strided descriptor encoding."""
    if not descriptors:
        return ()
    runs: list[tuple[int, int, int, int]] = []
    for item in descriptors:
        if len(item) == 2:
            start, end = int(item[0]), int(item[1])
            if end > start:
                width = end - start
                runs.append((start, width, width, 1))
        else:
            start, width, stride, count = (
                int(item[0]), int(item[1]), int(item[2]), int(item[3]),
            )
            if count == 1 or stride == width:
                span_width = width * count
                runs.append((start, span_width, span_width, 1))
            else:
                runs.append((start, width, stride, count))
    if not runs:
        return ()
    runs.sort()
    separated: list[tuple[int, int, int, int]] = []
    cluster = [runs[0]]
    cluster_end = _run_end(runs[0])
    for run in runs[1:]:
        if run[0] <= cluster_end:
            cluster.append(run)
            cluster_end = max(cluster_end, _run_end(run))
        else:
            separated.extend(_coalesce_run_cluster(cluster))
            cluster = [run]
            cluster_end = _run_end(run)
    separated.extend(_coalesce_run_cluster(cluster))
    queue = deque(separated)
    canonical: list[tuple[int, int, int, int]] = []
    while queue:
        start, width, stride, count = queue.popleft()
        while queue:
            ns, nw, nd, nc = queue[0]
            if nw != width:
                break
            if count == 1:
                new_stride = ns - start
                if nc >= 2 and nd == new_stride:
                    queue.popleft()
                    stride, count = new_stride, 1 + nc
                else:
                    rest = _pop_first_interval(queue.popleft())
                    if rest is not None:
                        queue.appendleft(rest)
                    stride, count = new_stride, 2
            elif ns != start + count * stride:
                break
            elif nc >= 2 and nd == stride:
                queue.popleft()
                count += nc
            else:
                rest = _pop_first_interval(queue.popleft())
                if rest is not None:
                    queue.appendleft(rest)
                count += 1
        canonical.append((start, width, stride, count))
    return tuple(canonical)


@dataclass(frozen=True)
class StateAccess:
    """Instance binding of one real state read/write/alias to an occurrence.

    This is a private projection over existing OpSpec/TensorValue/StateEffect
    data, not a new public IR.  ``bytes`` is the actual issued transfer volume
    for this direction/span; ``read_bytes``/``write_bytes`` are retained for
    callers that carry one record per direction.
    """

    occurrence: str
    producer: str
    request: str
    stage: int
    layer: int
    direction: str
    position: str = "cache"
    output: str = ""
    storage: str = ""
    state_version: int | None = None
    span: tuple[tuple[int, ...], ...] = ()
    rank: int | None = None
    scope: str = "target"
    bytes: float = 0.0
    source: str = ""
    reader: str = ""
    grant: str = ""
    decision: str = ""
    last_use: str = ""
    span_format: str = "half_open"
    byte_span: tuple[tuple[int, ...], ...] = ()
    byte_span_format: str = "half_open"
    span_unit: str = ""
    persistent: bool = True
    temporary: bool = False

    def __post_init__(self):
        fmt = str(self.span_format or "half_open")
        norm_span = _normalize_span(self.span, span_format=fmt)
        object.__setattr__(self, "span", norm_span)
        object.__setattr__(self, "span_format", _canonical_span_format(norm_span, fmt))
        bfmt = str(self.byte_span_format or "half_open")
        norm_bspan = _normalize_span(self.byte_span, span_format=bfmt) if self.byte_span else ()
        object.__setattr__(self, "byte_span", norm_bspan)
        object.__setattr__(self, "byte_span_format", _canonical_span_format(norm_bspan, bfmt))
        object.__setattr__(self, "span_unit", str(self.span_unit or ""))
        object.__setattr__(self, "persistent", bool(self.persistent))
        object.__setattr__(self, "temporary", bool(self.temporary))
        if self.direction not in {"read", "write", "alias"}:
            raise ValueError(f"state_access_direction_invalid:{self.direction}")
        if not self.occurrence or not self.producer:
            raise ValueError("state_access_requires_occurrence_and_producer")
        if isinstance(self.bytes, bool) or not math.isfinite(float(self.bytes)) or float(self.bytes) < 0:
            raise ValueError("state_access_bytes_invalid")
        object.__setattr__(self, "bytes", float(self.bytes))

    @property
    def key(self) -> tuple:
        # One outer occurrence can carry several physical writers.  The
        # producer, stage, and exact byte coverage are part of the physical
        # access identity so disjoint slices within one cache line are never
        # collapsed while equivalent span encodings deduplicate identically.
        return (self.occurrence, self.producer, self.stage, self.output,
                self.storage, self.state_version,
                _canonical_span_key(self.span),
                _canonical_span_key(self.byte_span),
                self.span_unit, self.direction, self.rank, self.request,
                self.scope)

    def to_dict(self) -> dict:
        return {
            "occurrence": self.occurrence, "producer": self.producer,
            "request": self.request, "stage": self.stage, "layer": self.layer,
            "direction": self.direction, "position": self.position,
            "output": self.output, "storage": self.storage,
            "state_version": self.state_version, "span": self.span,
            "span_format": self.span_format,
            "byte_span": self.byte_span,
            "byte_span_format": self.byte_span_format,
            "span_unit": self.span_unit,
            "persistent": self.persistent,
            "temporary": self.temporary,
            "rank": self.rank, "scope": self.scope, "bytes": self.bytes,
            "source": self.source, "reader": self.reader, "grant": self.grant,
            "decision": self.decision, "last_use": self.last_use,
        }


def _as_access(value: Any) -> StateAccess:
    if isinstance(value, StateAccess):
        return value
    if isinstance(value, Mapping):
        data = dict(value)
        if "bytes" not in data:
            data["bytes"] = data.get("write_bytes", data.get("read_bytes", 0.0))
        data.pop("read_bytes", None)
        data.pop("write_bytes", None)
        data.setdefault("position", "cache")
        data.setdefault("output", "")
        data.setdefault("storage", "")
        data.setdefault("scope", "target")
        fmt = str(data.get("span_format") or "half_open")
        norm_span = _normalize_span(data.get("span", ()), span_format=fmt)
        data["span"] = norm_span
        data["span_format"] = _canonical_span_format(norm_span, fmt)
        raw_bspan = data.get("byte_span", ())
        bfmt = str(data.get("byte_span_format") or fmt)
        norm_bspan = _normalize_span(raw_bspan, span_format=bfmt) if raw_bspan else ()
        data["byte_span"] = norm_bspan
        data["byte_span_format"] = _canonical_span_format(norm_bspan, bfmt)
        data["span_unit"] = str(data.get("span_unit") or data.pop("unit", "") or "")
        temporary = bool(data.get("temporary", False))
        raw_persistent = data.get("persistent")
        data["temporary"] = temporary
        data["persistent"] = (not temporary) if raw_persistent is None else bool(raw_persistent)
        data.setdefault("rank", None)
        data.setdefault("state_version", None)
        data.setdefault("source", "")
        data.setdefault("reader", "")
        data.setdefault("grant", "")
        data.setdefault("decision", "")
        data.setdefault("last_use", "")
        return StateAccess(**data)
    raise TypeError("state_access_mapping_or_StateAccess_required")


def _as_accesses(values) -> tuple[StateAccess, ...]:
    return tuple(_as_access(value) for value in (values or ()))


def _unique_accesses(accesses) -> tuple[StateAccess, ...]:
    seen = set()
    result = []
    for access in accesses:
        if access.key in seen:
            continue
        seen.add(access.key)
        result.append(access)
    return tuple(result)


def _access_bytes(accesses, direction: str) -> float:
    return float(sum(access.bytes for access in _unique_accesses(accesses)
                     if access.direction == direction))


def _physical_producer_output(meta: Mapping, occurrence: str,
                              default_output: str = "") -> tuple[str, str]:
    """Read the producer/output port carried by the selected op.

    Only producer-owned fields are consumed.  A generic metadata key named
    ``producer`` is not guessed to be a state port; explicit lowering owners
    supply ``state_producer``/``state_output`` or ``state_producer_output``.
    When none is present the selected occurrence itself is the writer.
    """
    raw = meta.get("state_producer_output")
    producer = ""
    output = ""
    if isinstance(raw, Mapping):
        producer = str(raw.get("producer") or "")
        output = str(raw.get("output") or "")
    elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
        producer, output = str(raw[0] or ""), str(raw[1] or "")
    producer = producer or str(
        meta.get("state_producer") or meta.get("cache_producer") or "")
    output = output or str(
        meta.get("state_output") or meta.get("cache_output") or "")
    return producer or occurrence, output or default_output


def _accesses_from_ops(work: Any, physical_ops=None) -> tuple[StateAccess, ...]:
    """Compatibility binding from real selected op metadata only.

    It never walks the model layer list, a token total or a capacity delta.
    Missing state-byte/output/storage bindings stay absent; the lowering owner
    must supply them through the explicit ``state_accesses`` argument.
    """
    ops = (tuple(physical_ops) if physical_ops is not None
           else tuple(getattr(work, "ops", ()) or ()))
    if not ops:
        return ()
    all_ops = tuple(getattr(work, "ops", ()) or ()) or ops
    cache_views_by_owner: dict[tuple[int, int, str], Mapping] = {}
    for candidate in all_ops:
        if str(getattr(candidate, "family", "")) != "cache_store":
            continue
        c_meta = dict(getattr(candidate, "metadata", ()) or ())
        ctv = c_meta.get("cache_tensor_view")
        if isinstance(ctv, Mapping):
            c_scope = "draft" if c_meta.get("producer_scope") == "draft" else "target"
            c_key = (int(getattr(candidate, "stage", 0)), int(getattr(candidate, "layer", -1)), c_scope)
            cache_views_by_owner.setdefault(c_key, ctv)
    plan = getattr(work, "plan", None)
    plan_views = tuple(getattr(plan, "requests", ()) or ())
    padding_ids = {str(view.request_id) for view in plan_views
                   if getattr(view, "padding", False)}
    views_by_id = {str(view.request_id): view for view in plan_views}
    result = []
    for op in ops:
        meta = dict(getattr(op, "metadata", ()) or ())
        stage = int(getattr(op, "stage", 0))
        layer = int(getattr(op, "layer", -1))
        raw_requests = tuple(getattr(op, "requests", ()) or ("*",))
        requests = tuple(request for request in raw_requests
                         if str(request) not in padding_ids)
        if not requests:
            continue
        scope = "draft" if meta.get("producer_scope") == "draft" else "target"
        occurrence = str(getattr(op, "op_id", ""))
        kind = str(meta.get("cache_kind", ""))
        producer, bound_output = _physical_producer_output(
            meta, occurrence, kind)
        if kind == "index_pool_tail":
            retained = int(meta.get("retained_rows", 0) or 0)
            width = int(meta.get("index_width", 0) or 0)
            write = float(meta.get("write_bytes", 0.0) or 0.0)
            if write <= 0 and retained and width:
                write = float(retained * width * 2 * _INDEX_POOL_TAIL_COMPONENT_BYTES)
            if write <= 0:
                write = float(getattr(op, "mte3_bytes", 0.0) or 0.0)
            if write > 0:
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=requests[0] if requests else "*", stage=stage,
                    layer=layer, direction="write", position="tail",
                    output=bound_output or kind, storage=kind, bytes=write,
                    span=((0, retained),) if retained else (),
                    scope=scope, source=producer,
                ))
            continue
        if bool(meta.get("shared_owner", False)):
            continue
        if kind and str(getattr(op, "family", "")) == "cache_store":
            _append_cache_store_accesses(
                result, op, meta, raw_requests, requests, views_by_id,
                bound_output, kind, occurrence, producer, stage, layer, scope)
            continue
        sibling_ctv = cache_views_by_owner.get((stage, layer, scope))
        gqa = meta.get("gqa_decode") or meta.get("gqa_prefill")
        if not isinstance(gqa, Mapping):
            gqa = _mixed_gqa_contract(op, meta, sibling_ctv)
        if isinstance(gqa, Mapping):
            _append_gqa_history_accesses(
                result, meta, gqa, sibling_ctv, raw_requests, padding_ids,
                bound_output, occurrence, producer, stage, layer, scope)
            continue
        contract = meta.get("gdn_state_contract")
        if contract:
            _append_gdn_state_accesses(
                result, op, meta, contract, raw_requests, padding_ids,
                bound_output, occurrence, producer, stage, layer, scope)
            continue
        read = float(meta.get("state_read_bytes", 0.0) or 0.0)
        write = float(meta.get("state_write_bytes", 0.0) or 0.0)
        if read > 0:
            for request in requests:
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=request, stage=stage, layer=layer,
                    direction="read", position="state",
                    output=bound_output,
                    storage=str(meta.get("state_layout", "state")),
                    state_version=(meta.get("state_read_versions") or (None,))[0]
                    if isinstance(meta.get("state_read_versions"), (tuple, list)) else None,
                    bytes=read / max(1, len(requests)), scope=scope,
                    source=producer,
                ))
        if write > 0:
            for request in requests:
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=request, stage=stage, layer=layer,
                    direction="write", position="state",
                    output=bound_output,
                    storage=str(meta.get("state_layout", "state")),
                    state_version=(meta.get("state_commit_versions") or (None,))[0]
                    if isinstance(meta.get("state_commit_versions"), (tuple, list)) else None,
                    bytes=write / max(1, len(requests)), scope=scope,
                    source=producer,
                ))
    return tuple(result)


def _request_cache_store_rows(view: Any, *, scope: str) -> int:
    """Logical rows written to KV cache by one request view (0 for padding)."""
    if getattr(view, "padding", False):
        return 0
    logical = int(getattr(view, "logical_q_len", getattr(view, "q_len", 0)) or 0)
    if scope == "target" and getattr(view, "draft_verify", True):
        proposal = int(getattr(view, "draft_proposal_tokens", 0) or 0)
        if proposal > 0:
            if getattr(view, "phase", "") == "decode":
                return max(logical, 1 + proposal)
            return logical + proposal
    return max(0, logical)


def _draft_cache_store_rows(meta: Mapping) -> dict[str, int]:
    """Extract per-request draft row counts attached by ``_tag_draft_ops``."""
    raw = meta.get("draft_rows_by_request")
    if isinstance(raw, Mapping):
        return {str(req): max(0, int(cnt or 0)) for req, cnt in raw.items()}
    if isinstance(raw, (tuple, list)):
        rows: dict[str, int] = {}
        for item in raw:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                rows[str(item[0])] = max(0, int(item[1] or 0))
        return rows
    return {}


def _append_cache_store_accesses(
    result, op, meta, raw_requests, requests, views_by_id,
    bound_output, kind, occurrence, producer, stage, layer, scope,
):
    write = float(getattr(op, "mte3_bytes", 0.0) or 0.0)
    if write <= 0.0:
        write = float(getattr(op, "hbm_bytes", 0.0) or 0.0) / 2.0
    ctv = meta.get("cache_tensor_view")
    row_bytes = float(ctv.get("cache_row_bytes", 0.0) or 0.0) if isinstance(ctv, Mapping) else 0.0
    if scope == "draft":
        draft_rows = _draft_cache_store_rows(meta)
        if draft_rows:
            if row_bytes > 0.0:
                for req in requests:
                    req_bytes = float(draft_rows.get(str(req), 0)) * row_bytes
                    if req_bytes > 0.0:
                        result.append(StateAccess(
                            occurrence=occurrence, producer=producer,
                            request=req, stage=stage, layer=layer,
                            direction="write", position="cache",
                            output=bound_output or kind, storage=kind,
                            bytes=req_bytes, scope=scope, source=producer,
                        ))
                return
            if write > 0.0:
                total_draft = 0
                for req in raw_requests:
                    key = str(req)
                    if key in draft_rows:
                        total_draft += draft_rows[key]
                    elif key in views_by_id and getattr(views_by_id[key], "padding", False):
                        total_draft += max(0, int(getattr(views_by_id[key], "q_len", 0) or 0))
                if total_draft > 0:
                    for req in requests:
                        req_bytes = write * (float(draft_rows.get(str(req), 0)) / float(total_draft))
                        if req_bytes > 0.0:
                            result.append(StateAccess(
                                occurrence=occurrence, producer=producer,
                                request=req, stage=stage, layer=layer,
                                direction="write", position="cache",
                                output=bound_output or kind, storage=kind,
                                bytes=req_bytes, scope=scope, source=producer,
                            ))
                    return

    all_active_known = all(str(req) in views_by_id for req in requests)
    all_raw_known = all(str(req) in views_by_id for req in raw_requests)

    if row_bytes > 0.0 and scope == "target" and all_active_known:
        for req in requests:
            rows = _request_cache_store_rows(views_by_id[str(req)], scope=scope)
            req_bytes = float(rows) * row_bytes
            if req_bytes <= 0.0:
                continue
            result.append(StateAccess(
                occurrence=occurrence, producer=producer,
                request=req, stage=stage, layer=layer,
                direction="write", position="cache",
                output=bound_output or kind, storage=kind,
                bytes=req_bytes, scope=scope, source=producer,
            ))
        return

    if all_raw_known and write > 0.0:
        total_phys = 0
        for req in raw_requests:
            view = views_by_id[str(req)]
            if getattr(view, "padding", False):
                total_phys += max(0, int(getattr(view, "q_len", 0) or 0))
            else:
                total_phys += _request_cache_store_rows(view, scope=scope)
        if total_phys > 0:
            for req in requests:
                rows = _request_cache_store_rows(views_by_id[str(req)], scope=scope)
                req_bytes = write * (float(rows) / float(total_phys))
                if req_bytes <= 0.0:
                    continue
                result.append(StateAccess(
                    occurrence=occurrence, producer=producer,
                    request=req, stage=stage, layer=layer,
                    direction="write", position="cache",
                    output=bound_output or kind, storage=kind,
                    bytes=req_bytes, scope=scope, source=producer,
                ))
            return

    if write <= 0.0:
        return
    for req in requests:
        result.append(StateAccess(
            occurrence=occurrence, producer=producer,
            request=req, stage=stage, layer=layer,
            direction="write", position="cache",
            output=bound_output or kind, storage=kind,
            bytes=write / max(1, len(requests)), scope=scope,
            source=producer,
        ))


def _is_plain_gqa_cache_representations(reprs: Any) -> bool:
    """True when ``cache_representations`` declares only plain unpacked rows."""
    if not isinstance(reprs, Mapping):
        return False
    for cfg in reprs.values():
        if not isinstance(cfg, Mapping):
            return False
        if int(cfg.get("group_size", 0) or 0) > 0:
            return False
        scale = str(cfg.get("scale_dtype", "none") or "none").strip().lower()
        if scale not in {"", "none"}:
            return False
        if int(cfg.get("suffix_elements", 0) or 0) > 0:
            return False
        if int(cfg.get("packed_elements", 1) or 1) != 1:
            return False
        if int(cfg.get("alignment", 1) or 1) > 1:
            return False
        if str(cfg.get("dtype", "") or "").strip().lower() == "fp4_e2m1":
            return False
    return True


def _plain_cache_tensor_row_bytes(ctv: Mapping | None) -> float | None:
    """Return exact plain unpacked KV row bytes from a sibling ``cache_tensor_view``.

    Returns ``None`` if ``ctv`` is absent, or ``0.0`` if ``ctv`` declares
    packed, block-quantized, scaled, or padded rows that must stay gated.
    """
    if not isinstance(ctv, Mapping):
        return None
    if int(ctv.get("cache_group_size", 0) or 0) > 0:
        return 0.0
    scale = str(ctv.get("cache_scale_dtype", "none") or "none").strip().lower()
    if scale not in {"", "none"}:
        return 0.0
    key_dtype = str(ctv.get("key_dtype") or "").strip().lower()
    val_dtype = str(ctv.get("value_dtype") or "").strip().lower()
    if not key_dtype or not val_dtype or "fp4_e2m1" in {key_dtype, val_dtype}:
        return 0.0
    key_shape = ctv.get("key_shape")
    if not isinstance(key_shape, (tuple, list)) or len(key_shape) < _RECURRENT_STATE_3D_RANK:
        return None
    kv_heads = int(key_shape[1] or 0)
    head_dim = int(key_shape[2] or 0)
    if kv_heads <= 0 or head_dim <= 0:
        return None
    from mindie_sim.schema.common import dtype_bytes
    plain = float(kv_heads * head_dim) * (
        float(dtype_bytes(key_dtype)) + float(dtype_bytes(val_dtype))
    )
    declared = float(ctv.get("cache_row_bytes", 0.0) or 0.0)
    if declared > 0.0 and abs(declared - plain) > 1e-6:
        return 0.0
    return declared if declared > 0.0 else plain


def _mixed_gqa_contract(
    op: Any, meta: Mapping, sibling_ctv: Mapping | None,
) -> dict[str, Any] | None:
    """Build a plain GQA history contract for mixed prefill+decode attention ops."""
    if str(getattr(op, "family", "")) != "attention":
        return None
    req_shapes = meta.get("request_shapes")
    if not req_shapes:
        return None
    cp_pairs = meta.get("cp_per_rank_pairs") or ()
    cp = len(cp_pairs) if isinstance(cp_pairs, (tuple, list)) and cp_pairs else 1
    if cp > 1 or float(meta.get("selection_read_bytes", 0.0) or 0.0) > 0.0:
        return None
    if meta.get("prefix_shared_rows") or meta.get("prefix_read") or meta.get("sparse_prefix_unknown"):
        return None
    # Detect sliding-window truncation when local_shape records fewer rows than full causal kv_len.
    local_shape = tuple(getattr(op, "local_shape", ()) or ())
    full_rows = sum(
        int(pair[0] or 0) + int(pair[1] or 0)
        for pair in req_shapes
        if isinstance(pair, (tuple, list)) and len(pair) >= 2
    )
    if len(local_shape) >= 2 and full_rows > 0 and int(local_shape[1] or 0) < full_rows:
        return None
    atv = meta.get("attention_tensor_view")
    kv_heads = 0
    head_dim = 0
    if isinstance(atv, Mapping):
        if str(atv.get("kv_storage") or "") == "none":
            return None
        kv_heads = int(atv.get("local_kv_heads", 0) or 0)
        head_dim = int(atv.get("head_dim", 0) or 0)
    if (kv_heads <= 0 or head_dim <= 0) and isinstance(sibling_ctv, Mapping):
        key_shape = sibling_ctv.get("key_shape")
        if isinstance(key_shape, (tuple, list)) and len(key_shape) >= _RECURRENT_STATE_3D_RANK:
            kv_heads = int(key_shape[1] or 0)
            head_dim = int(key_shape[2] or 0)
    if kv_heads <= 0 or head_dim <= 0:
        return None
    return {
        "requests": tuple(req_shapes),
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "window": int(meta.get("window", 0) or 0),
        "cp": cp,
    }


def _append_gqa_history_accesses(
    result, meta, gqa, sibling_ctv, raw_requests, padding_ids, bound_output,
    occurrence, producer, stage, layer, scope,
):
    """Bind plain dense/ragged GQA historical KV read spans from GQA contracts.

    Packed/quantized K/V, shared prefixes, sliding windows, and CP ownership
    bind through explicit physical ``memory_accesses``/``state_accesses`` emitted
    by the attention lowering owner (``W2-OP01`` ``gqa_contracts.py``).
    """
    req_shapes = tuple(gqa.get("requests", ()) or ())
    kv_heads = int(gqa.get("kv_heads", 0) or 0)
    head_dim = int(gqa.get("head_dim", 0) or 0)
    if not req_shapes or kv_heads <= 0 or head_dim <= 0:
        return
    if int(gqa.get("window", 0) or 0) > 0 or int(gqa.get("cp", 1) or 1) > 1:
        return
    atv = meta.get("attention_tensor_view")
    if isinstance(atv, Mapping) and atv.get("cache_representations"):
        if not _is_plain_gqa_cache_representations(atv["cache_representations"]):
            return
    ctv_row_bytes = _plain_cache_tensor_row_bytes(sibling_ctv)
    if ctv_row_bytes == 0.0:
        return
    from mindie_sim.schema.common import dtype_bytes
    if ctv_row_bytes is not None and ctv_row_bytes > 0.0:
        row_bytes = ctv_row_bytes
    elif isinstance(atv, Mapping) and atv.get("key_dtype") and atv.get("value_dtype"):
        row_bytes = float(kv_heads * head_dim) * (
            float(dtype_bytes(str(atv["key_dtype"])))
            + float(dtype_bytes(str(atv["value_dtype"])))
        )
    else:
        kv_dtype = str(gqa.get("kv_dtype") or "bf16")
        row_bytes = float(2 * kv_heads * head_dim) * float(dtype_bytes(kv_dtype))
    if row_bytes <= 0.0:
        return
    for slot, shape_pair in enumerate(req_shapes):
        if not isinstance(shape_pair, (tuple, list)) or len(shape_pair) < 2:
            continue
        context = int(shape_pair[1] or 0)
        if context <= 0:
            continue
        req = (
            str(raw_requests[slot])
            if slot < len(raw_requests)
            else (str(raw_requests[0]) if len(raw_requests) == 1 else f"req:{slot}")
        )
        if req in padding_ids:
            continue
        result.append(StateAccess(
            occurrence=occurrence, producer=producer,
            request=req, stage=stage, layer=layer,
            direction="read", position="cache",
            output=bound_output or "gqa_kv", storage="gqa_kv",
            span=((0, context),),
            scope=scope, bytes=float(context) * row_bytes,
            source=producer,
        ))


def _gdn_value_input_shapes(meta: Mapping, contract: Mapping) -> tuple[tuple[int, ...], ...]:
    """Extract typed tensor layout shapes declared on a GDN state op."""
    shapes: list[tuple[int, ...]] = []
    for raw in (meta.get("state_value_inputs"), contract.get("value_inputs")):
        if not isinstance(raw, (tuple, list)):
            continue
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            layout = entry.get("layout")
            shape = layout.get("shape") if isinstance(layout, Mapping) else None
            if isinstance(shape, (tuple, list)) and shape:
                shapes.append(tuple(int(dim or 0) for dim in shape))
    return tuple(shapes)


def _gdn_conv_channels(op: Any, meta: Mapping, contract: Mapping) -> int:
    """Resolve conv channel width without misreading flattened 1D ``local_shape``."""
    for shape in _gdn_value_input_shapes(meta, contract):
        if len(shape) >= 2 and shape[-1] > 0:
            return int(shape[-1])
    local_shape = tuple(getattr(op, "local_shape", ()) or ())
    if len(local_shape) >= 2 and int(local_shape[-1] or 0) > 0:
        return int(local_shape[-1])
    return int(meta.get("channels", 0) or 0)


def _gdn_recurrent_state_bytes(op: Any, meta: Mapping, contract: Mapping, element: float) -> float:
    """Resolve per-request recurrent state bytes from contract dims or typed tensor shape."""
    heads = int(meta.get("heads", contract.get("heads", 0)) or 0)
    dk = int(meta.get("dk", contract.get("dk", 0)) or 0)
    dv = int(meta.get("dv", contract.get("dv", 0)) or 0)
    if heads > 0 and dk > 0 and dv > 0:
        return float(heads * dk * dv) * element
    local_shape = tuple(int(dim or 0) for dim in (getattr(op, "local_shape", ()) or ()))
    candidates = (local_shape,) + _gdn_value_input_shapes(meta, contract)
    for shape in candidates:
        if len(shape) >= _RECURRENT_STATE_4D_RANK:
            elems = shape[1] * shape[2] * shape[3]
            if elems > 0:
                return float(elems) * element
        if len(shape) == _RECURRENT_STATE_3D_RANK:
            elems = shape[0] * shape[1] * shape[2]
            if elems > 0:
                return float(elems) * element
    return 0.0


def _append_gdn_state_accesses(
    result, op, meta, contract, raw_requests, padding_ids, bound_output,
    occurrence, producer, stage, layer, scope,
):
    # The original-slot contract is the physical access authority.
    # Local chunk banks carry scope="local" and never hit host cache tiers;
    # gather/scatter split cache_read and cache_publish directions explicitly.
    contract_scope = str(contract.get("scope") or "").strip().lower()
    target_kind = str(contract.get("target_kind") or "").strip().lower()
    if contract_scope == "local" or target_kind == "local":
        return
    allow_read = contract_scope != "cache_publish"
    allow_write = contract_scope != "cache_read"

    from mindie_sim.schema.common import dtype_bytes
    is_conv = (
        target_kind == "conv"
        or str(getattr(op, "family", "")) == "short_convolution"
    )
    if is_conv:
        state_dtype = str(
            meta.get("state_dtype") or contract.get("conv_state_dtype") or "bf16"
        )
        storage = str(
            meta.get("state_layout")
            or contract.get("conv_state_layout")
            or contract.get("state_layout")
            or "state"
        )
    else:
        state_dtype = str(
            contract.get("state_dtype") or meta.get("state_dtype") or "fp32"
        )
        storage = str(contract.get("state_layout", "state"))
    element = float(meta.get("state_element_bytes", 0.0) or 0.0)
    if not element:
        element = float(dtype_bytes(state_dtype))
    state_bytes = 0.0 if is_conv else _gdn_recurrent_state_bytes(op, meta, contract, element)
    history_slots = int(meta.get("state_history_slots", contract.get("state_history_slots", 0)) or 0)
    kernel_width = int(meta.get("kernel_width", contract.get("conv_kernel", 0)) or 0)
    base_history = (kernel_width - 1) if kernel_width > 1 else history_slots
    channels = _gdn_conv_channels(op, meta, contract) if is_conv else 0
    conv_speculative = bool(meta.get("speculative", False))
    conv_row_bytes = float(channels) * element if (is_conv and channels > 0 and base_history > 0) else 0.0
    queries = tuple(contract.get("queries", ()) or meta.get("sequence_tokens", ()) or ())
    read_versions = tuple(contract.get("read_versions", ()))
    checkpoints = tuple(contract.get("checkpoints", ()))
    targets = tuple(contract.get("targets", ()))
    publish_versions = tuple(contract.get("publish_versions", ()))
    slot_rows = []
    for slot, request in enumerate(raw_requests):
        if str(request) in padding_ids:
            continue
        slot_rows.append({
            "slot": slot, "request": request,
            "target": (targets[slot] if slot < len(targets)
                       else (bound_output or "state")),
            "read_version": (read_versions[slot]
                             if slot < len(read_versions) else None),
            "publish_version": (publish_versions[slot]
                                if slot < len(publish_versions)
                                else None),
            "checkpoint": (int(checkpoints[slot])
                           if slot < len(checkpoints) else 0),
            "query": (int(queries[slot]) if slot < len(queries) else 1),
        })
    if conv_row_bytes > 0.0:
        for row in slot_rows:
            row["read"] = (
                float(base_history) * conv_row_bytes
                if (allow_read and row["read_version"] is not None) else 0.0
            )
            if allow_write and row["checkpoint"] > 0:
                write_slots = (
                    base_history + max(0, row["query"] - 1)
                    if (conv_speculative and kernel_width > 1)
                    else base_history
                )
                row["write"] = float(write_slots) * conv_row_bytes
            else:
                row["write"] = 0.0
    else:
        for row in slot_rows:
            row["read"] = (
                state_bytes
                if (allow_read and state_bytes > 0.0 and row["read_version"] is not None)
                else 0.0
            )
            row["write"] = (
                state_bytes * row["checkpoint"]
                if (allow_write and state_bytes > 0.0) else 0.0
            )
    aggregate_read = float(meta.get("state_read_bytes", 0.0) or 0.0)
    aggregate_write = float(meta.get("state_write_bytes", 0.0) or 0.0)
    if state_bytes > 0.0 or conv_row_bytes > 0.0:
        contract_read = sum(row["read"] for row in slot_rows)
        contract_write = sum(row["write"] for row in slot_rows)
        checks = []
        if allow_read:
            checks.append(("read", aggregate_read, contract_read))
        if allow_write:
            checks.append(("write", aggregate_write, contract_write))
        for direction, aggregate, total in checks:
            if aggregate > 0.0 and abs(aggregate - total) > 1e-9:
                raise ValueError(
                    f"state_access_contract_total_mismatch:{direction}:"
                    f"{aggregate}:{total}")
    elif slot_rows:
        if allow_read and aggregate_read > 0.0:
            for row in slot_rows:
                row["read"] = aggregate_read / len(slot_rows)
        if allow_write and aggregate_write > 0.0:
            for row in slot_rows:
                row["write"] = aggregate_write / len(slot_rows)
    for row in slot_rows:
        if row["read"] > 0.0:
            result.append(StateAccess(
                occurrence=occurrence, producer=producer,
                request=row["request"], stage=stage, layer=layer,
                direction="read", position="state",
                output=row["target"],
                storage=storage,
                state_version=row["read_version"], span=((0, 1),),
                scope=scope, bytes=row["read"], source=producer))
        if row["write"] > 0.0:
            result.append(StateAccess(
                occurrence=occurrence, producer=producer,
                request=row["request"], stage=stage, layer=layer,
                direction="write", position="state",
                output=row["target"],
                storage=storage,
                state_version=row["publish_version"],
                span=((0, 1),), scope=scope, bytes=row["write"],
                source=producer, decision=producer))


def project_state_tiers(
    accesses: Any, policy: OffloadPolicy, *, expected_bytes: float | None = None,
) -> tuple[dict, ...]:
    """Project persistent cache/state accesses into per-occurrence issued tier bytes.

    Transient local activations (e.g. ``position="local"`` or ``moe.act:*``)
    are excluded from host-tier offload projection while remaining intact in the
    general execution memory ledger.
    """
    bound = tuple(
        access for access in _unique_accesses(_as_accesses(accesses))
        if is_persistent_offload_access(access)
    )
    if expected_bytes is not None:
        if (isinstance(expected_bytes, bool)
                or not isinstance(expected_bytes, (int, float))
                or not math.isfinite(float(expected_bytes))
                or float(expected_bytes) < 0.0):
            raise ValueError("state_access_total_bytes_invalid")
        if abs(float(expected_bytes) - sum(access.bytes for access in bound)) > 1e-9:
            raise ValueError("state_access_total_bytes_mismatch")
    records = []
    for access in bound:
        if access.direction == "alias":
            ddr = ssd = 0.0
        else:
            read = access.direction == "read"
            scale = (1.0 - policy.mtp_kvc_hit_rate) if read else 1.0
            ddr = access.bytes * policy.kvc_ddr_ratio * scale
            ssd = access.bytes * policy.kvc_ssd_ratio * scale
        records.append({
            **access.to_dict(),
            "ddr_bytes": ddr, "ssd_bytes": ssd, "hbm_bytes": ddr + ssd,
        })
    return tuple(records)
