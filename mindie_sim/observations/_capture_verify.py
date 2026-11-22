"""On-disk capture path resolution and streaming consistency verification.

[Role] Resolve optional local capture files from capture URIs or source
  references and verify streaming SHA-256 digests, device ranks, clock
  domains/operators, and boundary events across JSON, JSONL, Chrome traces, and profiler CSV.
[Invariants] Fails closed with ``ModelingError`` on unsafe capture URI paths,
  digest mismatches, rank mismatches, or clock/event mismatches on either
  header metadata or matched event rows; never loads whole large files.
[Timeouts & Locks] Synchronous 64 KiB chunked streaming; zero locks or I/O
  outside read-only access to the caller-supplied capture path.
[Side-Effect Policy] Strictly read-only over local capture files.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import partial
import gzip
from pathlib import Path, PurePosixPath
import re
from typing import Any

from mindie_sim.schema.common import ModelingError

CAPTURE_URI_SCHEME = "capture://"
_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})
_CAPTURE_EVENT_KEYS = frozenset({"events", "event_ids"})
_CAPTURE_CONTAINERS = frozenset({"boundary", "metadata", "payload"})
_NS_PER_US = 1000
_NS_PER_MS = 1000 * _NS_PER_US
_CSV_TIME_SCALE = {"ns": 1, "us": _NS_PER_US, "ms": _NS_PER_MS}
_CAPTURE_ROUNDING_NS = 1
_FIRST_SOURCE_ROW = 1
# msprof packs the device ID below the track type and host PID.
_ASCEND_DEVICE_ID_MASK = 0x1F
_ASCEND_TRACK_TYPE_MASK = 0x3E0
_ASCEND_HARDWARE_TRACK = 0x1C0
_TRACE_DEVICE_LABEL = re.compile(r"(?:device|npu|rank)(?: id)?[ _:-]+(?P<rank>[0-9]+)", re.IGNORECASE)
_TRACE_RANK_KEYS = ("device_rank", "rank_id", "rank", "device_id", "device", "Device_id", "Device Id", "Device ID")
_EVENT_ID_KEYS = ("event_id", "id", "name")
_TASK_ID_KEYS = ("Task Id", "Task ID", "task_id", "taskId")
_CONNECTION_ID_KEYS = ("Connection Id", "Connection ID", "connection_id", "connectionId")
_STREAM_ID_KEYS = ("Stream Id", "Stream ID", "Physic Stream Id", "stream_id", "streamId", "stream")
_JSON_ROW_KEYS = frozenset((*_EVENT_ID_KEYS, *_TRACE_RANK_KEYS, *_TASK_ID_KEYS, *_CONNECTION_ID_KEYS,
                            *_STREAM_ID_KEYS, "ph", "pid", "tid", "args"))
_SELECTOR_PREFIXES = ("row:", "task:", "conn:", "stream:", "dev:")
_JSON_START_KEYS = ("start_time", "start_ts", "start", "timestamp", "ts")
_JSON_END_KEYS = ("end_time", "end_ts", "end")
_JSON_DURATION_KEYS = ("duration", "dur")
_JSON_TIME_KEYS = frozenset(
    field for names in (_JSON_START_KEYS, _JSON_END_KEYS, _JSON_DURATION_KEYS)
    for name in names for field in (name, *(f"{name}_{unit}" for unit in _CSV_TIME_SCALE))
) | {"unit", "time_unit"}
_CAPTURE_META_KEYS = frozenset({
    "device_ranks", "ranks", "profiled_ranks", "device_rank", "rank",
    "clock_domain", "clock_operator", "start_event", "end_event",
})
L3_BOUNDARY_KINDS = frozenset({"model_step", "service_step", "e2e_step", "device_only_step", "host_device_step"})

_KernelOccurrence = tuple[int, int]  # (device rank, one-based source row)
_RankWindows = Mapping[int, tuple[Decimal, Decimal]]


@dataclass(frozen=True, slots=True)
class CaptureVerificationReceipt:
    selected_kernels: frozenset[_KernelOccurrence]
    window_kernels: frozenset[_KernelOccurrence] | None = None


@dataclass(frozen=True, slots=True)
class _CaptureRow:
    rank: int
    record: int
    names: frozenset[str]
    start: Decimal | None
    end: Decimal

    @property
    def occurrence(self) -> _KernelOccurrence:
        return self.rank, self.record


def resolve_capture_file(
    capture_uri: str, source_ref: tuple[tuple[str, str], ...], *,
    capture_root: Path | None, capture_files: Mapping[str, str | Path] | None,
    case_id: str, raw_sha256: str,
) -> Path | None:
    """Resolve an optional local capture file without storing host paths in manifests."""
    if capture_files:
        for key in (case_id, capture_uri, raw_sha256.lower()):
            if key and key in capture_files:
                resolved = Path(capture_files[key])
                if not resolved.is_file():
                    raise ModelingError(f"capture_file_missing:{case_id}")
                return resolved
    if capture_root is None:
        return None
    if capture_uri and capture_uri.startswith(CAPTURE_URI_SCHEME):
        rel = capture_uri[len(CAPTURE_URI_SCHEME):].lstrip("/")
        rel_path = PurePosixPath(rel)
        if ".." in rel_path.parts or not rel_path.parts:
            raise ModelingError(f"capture_uri_path_unsafe:{case_id}")
        candidates = [
            capture_root / rel_path,
            *([capture_root / f"{rel}.json"] if not rel_path.suffix else []),
            capture_root / rel_path.name,
        ]
        for cand in candidates:
            if cand.is_file():
                return cand
    if rel_ref := dict(source_ref).get("relative_path"):
        cand = capture_root / PurePosixPath(rel_ref)
        if cand.is_file():
            return cand
    return None


def _collect_event_names(raw_events: Any) -> set[str]:
    names: set[str] = set()
    if isinstance(raw_events, Sequence) and not isinstance(raw_events, (str, bytes)):
        for item in raw_events:
            if isinstance(item, str) and item.strip():
                names.add(item.strip())
            elif isinstance(item, Mapping):
                for field in ("event_id", "id", "name"):
                    if isinstance(val := item.get(field), str) and val.strip():
                        names.add(val.strip())
    return names


def _row_event_and_clock_info(
    row: Mapping[str, Any],
) -> tuple[set[str], list[Any], dict[str, list[Any]], bool]:
    ev_names: set[str] = set()
    row_clocks: list[Any] = []
    item_clocks: dict[str, list[Any]] = {}
    has_boundary = False
    for container in (row, row.get("boundary"), row.get("metadata")):
        if not isinstance(container, Mapping):
            continue
        if "clock_operator" in container:
            row_clocks.append(container["clock_operator"])
        for field in ("event_id", "id", "name", "start_event", "end_event"):
            if isinstance(val := container.get(field), str) and val.strip():
                ev_names.add(val.strip())
                if field in ("start_event", "end_event"):
                    has_boundary = True
        for key in ("event_ids", "events"):
            raw_ev = container.get(key)
            ev_names |= _collect_event_names(raw_ev)
            if isinstance(raw_ev, Sequence) and not isinstance(raw_ev, (str, bytes)):
                for item in raw_ev:
                    if isinstance(item, Mapping) and "clock_operator" in item:
                        for ev in _collect_event_names((item,)):
                            item_clocks.setdefault(ev, []).append(item["clock_operator"])
    return ev_names, row_clocks, item_clocks, has_boundary


def _iter_capture_rows(
    events: Iterator[tuple[str, Any]], first: tuple[str, Any], *,
    read_value: Callable[..., Any], skip_value: Callable[..., Any], trace: bool = False,
) -> Iterator[tuple[bool, Mapping[str, Any]]]:
    """Yield individual trace rows and compact headers from the streaming parser."""
    if first[0] == "start_array":
        while (item := next(events))[0] != "end_array":
            row = read_value(events, item)
            if isinstance(row, Mapping):
                yield trace or _is_trace_row(row), row
            elif trace:
                raise ModelingError("capture_trace_requires_object_rows")
            elif isinstance(row, str) and not trace:
                yield False, {"event_id": row}
        return
    if first[0] != "start_map":
        raise ModelingError("capture_json_requires_object_or_array")
    header = {}
    while (entry := next(events))[0] != "end_map":
        key, item = str(entry[1]), next(events)
        kind, _ = item
        if key in _CAPTURE_META_KEYS or key in _JSON_ROW_KEYS or key in _JSON_TIME_KEYS:
            header[key] = read_value(events, item)
        elif key == "traceEvents":
            if kind != "start_array":
                raise ModelingError("capture_trace_requires_array")
            yield from _iter_capture_rows(events, item, read_value=read_value, skip_value=skip_value, trace=True)
        elif key in _CAPTURE_EVENT_KEYS and kind == "start_array":
            yield from _iter_capture_rows(events, item, read_value=read_value, skip_value=skip_value)
        elif key in _CAPTURE_CONTAINERS and kind == "start_map":
            yield from _iter_capture_rows(events, item, read_value=read_value, skip_value=skip_value)
        else:
            skip_value(events, item)
    if header:
        yield _is_trace_row(header), header


def _is_trace_row(row: Mapping[str, Any]) -> bool:
    return "ph" in row and ("pid" in row or "ts" in row)


def _iter_jsonl_capture_rows(rows: Iterator[Mapping[str, Any]]) -> Iterator[tuple[bool, Mapping[str, Any]]]:
    for row in rows:
        yield _is_trace_row(row), row
        for key in ("events", "traceEvents"):
            if isinstance(nested := row.get(key), list):
                for item in nested:
                    if isinstance(item, Mapping):
                        yield key == "traceEvents" or _is_trace_row(item), item


def _trace_args(row: Mapping[str, Any]) -> Mapping[str, Any]:
    args = row.get("args")
    return args if isinstance(args, Mapping) else {}


def _trace_rank(row: Mapping[str, Any]) -> int | None:
    args = _trace_args(row)
    ranks = {int(value) for source in (row, args) for key in _TRACE_RANK_KEYS
             if (value := source.get(key)) is not None and str(value).isascii() and str(value).isdecimal()}
    if len(ranks) > 1:
        raise ModelingError("capture_rank_mismatch:conflicting_trace_rank_fields")
    return next(iter(ranks), None)


def _trace_process_rank(row: Mapping[str, Any]) -> int | None:
    label = str(_trace_args(row).get("name", ""))
    rank = _trace_rank(row)
    pid = row.get("pid")
    track_rank = None
    if device := _TRACE_DEVICE_LABEL.fullmatch(label):
        track_rank = int(device["rank"])
    elif (label == "Ascend Hardware" and type(pid) is int and pid >= _ASCEND_HARDWARE_TRACK
          and pid & _ASCEND_TRACK_TYPE_MASK == _ASCEND_HARDWARE_TRACK):
        track_rank = pid & _ASCEND_DEVICE_ID_MASK
    if rank is not None and track_rank is not None and rank != track_rank:
        raise ModelingError("capture_rank_mismatch:conflicting_trace_process_rank_fields")
    return rank if rank is not None else track_rank


def _capture_metadata(
    rows: Iterator[tuple[bool, Mapping[str, Any]]], target_events: frozenset[str] | None,
    expected_clock_operator: str | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    found: set[str] = set()
    rank_columns: set[int] = set()
    for trace, row in rows:
        if trace:
            processes = meta.setdefault("_trace_processes", {})
            if row.get("ph") == "M" and row.get("name") == "process_name":
                pid = row.get("pid")
                if isinstance(pid, (int, str)) and (rank := _trace_process_rank(row)) is not None:
                    if processes.setdefault(pid, rank) != rank:
                        raise ModelingError("capture_rank_mismatch:conflicting_trace_process_ranks")
            continue
        if _json_kernel_start(row) is not None:
            meta["_timed_json_rows"] = True
            continue
        ev_names, row_clocks, item_clocks, has_boundary = _row_event_and_clock_info(row)
        matched = ev_names if target_events is None else ev_names & target_events
        is_header = has_boundary or not any(key in row for key in ("event_id", "id", "name"))
        if not (is_header or matched):
            continue
        found.update(matched)
        if expected_clock_operator is not None and any(
            clock != expected_clock_operator
            for clock in (*row_clocks, *(clock for ev in matched for clock in item_clocks.get(ev, ())))
        ):
            meta["_clock_mismatch"] = True
        if is_header:
            if (ranks := _capture_ranks(row)) is not None:
                meta["_header_ranks"] = ranks
            for key in _CAPTURE_META_KEYS & row.keys():
                if key in meta and meta[key] != row[key] and key != "clock_operator":
                    kind = {"start_event": "event", "end_event": "event", "clock_domain": "clock_domain"}.get(key, "rank")
                    raise ModelingError(f"capture_{kind}_mismatch:header_conflict:{key}")
                meta.setdefault(key, row[key])
            for key in ("boundary", "metadata"):
                if key in row:
                    meta.setdefault(key, row[key])
        elif (ranks := _capture_ranks(row)) is not None:
            rank_columns.update(ranks)
    if rank_columns and _capture_ranks(meta) is None:
        meta["device_ranks"] = sorted(rank_columns)
    meta["event_ids"] = sorted(found)
    return meta


def _read_capture_rows(
    path: Path, *, ijson_mod: Any, iter_json_events_fn: Callable[..., Any],
    read_value_fn: Callable[..., Any], skip_value_fn: Callable[..., Any],
) -> Iterator[tuple[bool, Mapping[str, Any]]]:
    with path.open("rb") as handle:
        events = iter_json_events_fn(handle, ijson_mod=ijson_mod)
        yield from _iter_capture_rows(events, next(events), read_value=read_value_fn, skip_value=skip_value_fn)


def _extract_capture_meta(
    path: Path, *, ijson_mod: Any = None,
    stream_jsonl_fn: Callable[[Path], Any],
    iter_json_events_fn: Callable[..., Any],
    read_value_fn: Callable[..., Any],
    skip_value_fn: Callable[..., Any],
    target_events: frozenset[str] | None = None,
    expected_clock_operator: str | None = None,
) -> dict[str, Any]:
    def rows() -> Iterator[tuple[bool, Mapping[str, Any]]]:
        if path.suffix.lower() in _JSONL_SUFFIXES:
            yield from _iter_jsonl_capture_rows(stream_jsonl_fn(path))
        else:
            yield from _read_capture_rows(path, ijson_mod=ijson_mod, iter_json_events_fn=iter_json_events_fn,
                                          read_value_fn=read_value_fn, skip_value_fn=skip_value_fn)

    meta = _capture_metadata(rows(), target_events, expected_clock_operator)
    if "_trace_processes" in meta or meta.get("_timed_json_rows"):
        meta["_capture_rows"] = rows
    return meta


def _capture_ranks(meta: Mapping[str, Any]) -> set[int] | None:
    for container in (meta, meta.get("boundary"), meta.get("metadata")):
        if not isinstance(container, Mapping):
            continue
        for key in ("device_ranks", "ranks", "profiled_ranks"):
            raw = container.get(key)
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                ranks = {int(x) for x in raw if not isinstance(x, bool) and isinstance(x, int)}
                return ranks if len(ranks) == len(raw) else set()
        for key in ("device_rank", "rank"):
            raw = container.get(key)
            if isinstance(raw, int) and not isinstance(raw, bool):
                return {raw}
    return None


def _csv_time_field(fields: Mapping[str, str], kinds: tuple[str, ...]) -> tuple[str, int] | None:
    for kind in kinds:
        for unit, scale in _CSV_TIME_SCALE.items():
            for label in (f"{kind}({unit})", f"{kind}{unit}"):
                if label in fields:
                    return fields[label], scale
    return None


def _csv_capture_rows(
    path: Path, case_id: str, ranks: tuple[int, ...], target_events: frozenset[str], seen_ranks: set[int],
    windows: _RankWindows | None,
) -> Iterator[_CaptureRow]:
    open_csv = gzip.open if path.suffix.lower() == ".gz" else Path.open
    with open_csv(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        fields = {"".join(c for c in label.lower() if c.isalnum() or c in "()"): label
                  for label in reader.fieldnames or ()}
        rank_col = next((fields[key] for key in ("rankid", "rank", "devicerank", "deviceid", "device")
                         if key in fields), None)
        event_cols = [fields[key] for key in ("eventid", "event", "eventname", "name", "opname", "kernelname", "taskid", "optype", "id")
                      if key in fields]
        start_col = _csv_time_field(fields, ("starttime", "taskstarttime", "timestamp", "start"))
        end_col = _csv_time_field(fields, ("endtime", "taskendtime", "end"))
        duration_col = _csv_time_field(fields, ("duration", "durationtime", "taskduration"))
        total_col = _csv_time_field(fields, ("totaltime",))
        if not rank_col:
            raise ModelingError(f"capture_rank_mismatch:{case_id}:rank_column_absent")
        if not event_cols or not (
            start_col and (duration_col or end_col) or total_col and "count" in fields
        ):
            raise ModelingError(f"capture_csv_schema_unsupported:{case_id}")
        for record, row in enumerate(reader, start=_FIRST_SOURCE_ROW):
            rank = int(row[rank_col].strip())
            if rank not in ranks:
                continue
            seen_ranks.add(rank)
            names = frozenset()
            if windows is None:
                names = target_events.intersection((f"row={record}", *(row[col].strip() for col in event_cols)))
                if not names:
                    continue
            if not start_col:
                if int(row[fields["count"]]) != 1 or total_col is None:
                    raise ModelingError(f"capture_csv_aggregate_event:{case_id}")
                yield _CaptureRow(rank, record, names, None, _capture_time_ns(row, total_col))
                continue
            start = _capture_time_ns(row, start_col)
            if not _in_window(windows, rank, start):
                continue
            end = _capture_time_ns(row, end_col) if end_col else start + _capture_time_ns(row, duration_col)
            if duration_col and abs(end - start - _capture_time_ns(row, duration_col)) > _CAPTURE_ROUNDING_NS:
                raise ModelingError(f"capture_csv_interval_inconsistent:{case_id}")
            yield _CaptureRow(rank, record, names, start, end)


def _in_window(windows: _RankWindows | None, rank: int, start: Decimal) -> bool:
    if windows is None:
        return True
    left, right = windows[rank]
    # A launch at the endpoint belongs to the following window.
    return left <= start < right


def _timed_rank_duration(
    case_id: str, format_name: str, clock_operator: str, left: Decimal, right: Decimal,
    intervals: Iterable[tuple[Decimal, Decimal]],
) -> Decimal:
    if clock_operator == "elapsed_span":
        return right - left
    if clock_operator != "busy_union":
        raise ModelingError(f"capture_{format_name}_clock_unsupported:{case_id}")
    busy, cursor = Decimal(0), left
    for start, end in sorted(intervals):
        end = min(end, right)
        busy += max(0, end - max(start, cursor))
        cursor = max(cursor, end)
    return busy


def _event_field_values(sources: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> set[str]:
    return {str(value).strip() for source in sources for key in keys
            if type(value := source.get(key)) in (str, int) and str(value).strip()}


def _json_event_names(
    row: Mapping[str, Any], record: int, rank: int, target_events: frozenset[str],
) -> frozenset[str]:
    sources = (row, _trace_args(row))
    tasks = _event_field_values(sources, _TASK_ID_KEYS)
    connections = _event_field_values(sources, _CONNECTION_ID_KEYS)
    streams = _event_field_values(sources, _STREAM_ID_KEYS) or _event_field_values((row,), ("tid",))
    raw = _event_field_values(sources, _EVENT_ID_KEYS) | tasks | connections
    qualified = {
        f"row:{record}", *(f"task:{task}" for task in tasks), *(f"conn:{conn}" for conn in connections),
        *(selector for stream in streams for task in tasks
          for selector in (f"stream:{stream}:task:{task}", f"dev:{rank}:stream:{stream}:task:{task}")),
    }
    return target_events.intersection(qualified | {name for name in raw if not name.startswith(_SELECTOR_PREFIXES)})


def _json_time_field(row: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[str, int] | None:
    for unit, scale in _CSV_TIME_SCALE.items():
        for key in keys:
            if (field := f"{key}_{unit}") in row:
                return field, scale
    for key in keys:
        if key in row:
            unit = row.get("time_unit", row.get("unit", "us" if "ts" in row and "dur" in row else "ns"))
            return key, _CSV_TIME_SCALE[str(unit).lower()]
    return None


def _json_kernel_start(row: Mapping[str, Any]) -> tuple[str, int] | None:
    if ("start_event" in row or "end_event" in row) and not any(key in row for key in _EVENT_ID_KEYS):
        return None
    return _json_time_field(row, _JSON_START_KEYS)


def _capture_time_ns(row: Mapping[str, Any], field: tuple[str, int] | None) -> Decimal:
    if field is None:
        raise ValueError("capture_time_field_absent")
    column, scale = field
    value = Decimal(str(row[column]).strip()) * scale
    if not value.is_finite():
        raise ValueError("non_finite_capture_time")
    return value


def _json_capture_rows(
    case_id: str, boundary: Any, meta: Mapping[str, Any], target_events: frozenset[str], trace_format: bool,
    seen_ranks: set[int], windows: _RankWindows | None,
) -> Iterator[_CaptureRow]:
    processes = meta.get("_trace_processes", {})
    header_ranks = meta.get("_header_ranks", ())
    default_rank = next(iter(header_ranks)) if len(header_ranks) == 1 and not trace_format else None
    source = (row for trace, row in meta["_capture_rows"]() if trace == trace_format)
    for record, row in enumerate(source, start=_FIRST_SOURCE_ROW):
        if trace_format and row.get("ph") != "X":
            continue
        start_field = ("ts", _NS_PER_US) if trace_format else _json_kernel_start(row)
        if start_field is None:
            continue
        process_rank = processes.get(row.get("pid"), default_rank)
        explicit_rank = _trace_rank(row)
        rank = explicit_rank if explicit_rank is not None else process_rank
        if rank is None and not trace_format:
            raise ModelingError(f"capture_rank_mismatch:{case_id}:timed_json_event_rank_absent")
        if rank not in boundary.device_ranks:
            continue
        seen_ranks.add(rank)
        if trace_format and process_rank is not None and rank != process_rank:
            raise ModelingError(f"capture_rank_mismatch:{case_id}:trace_process_conflict")
        names = _json_event_names(row, record, rank, target_events) if windows is None else frozenset()
        if windows is None and not names:
            continue
        for key in ("clock_operator", "clock_domain"):
            expected = getattr(boundary, key)
            if names and row.get(key, expected) != expected:
                raise ModelingError(f"capture_{key}_mismatch:{case_id}:expected={expected}")
        start = _capture_time_ns(row, start_field)
        if not _in_window(windows, rank, start):
            continue
        end_field = None if trace_format else _json_time_field(row, _JSON_END_KEYS)
        dur_field = ("dur", _NS_PER_US) if trace_format else _json_time_field(row, _JSON_DURATION_KEYS)
        end = _capture_time_ns(row, end_field) if end_field else start + _capture_time_ns(row, dur_field)
        if end_field and dur_field and abs(end - start - _capture_time_ns(row, dur_field)) > _CAPTURE_ROUNDING_NS:
            raise ModelingError(f"capture_json_interval_inconsistent:{case_id}")
        yield _CaptureRow(rank, record, names, start, end)


def _selected_rows(
    case_id: str, format_name: str, boundary: Any, target_events: frozenset[str],
    rows: Iterable[_CaptureRow], seen_ranks: set[int],
) -> dict[int, list[_CaptureRow]]:
    selected: dict[int, list[_CaptureRow]] = {rank: [] for rank in boundary.device_ranks}
    found: dict[int, set[str]] = {rank: set() for rank in selected}
    endpoints = {boundary.start_event.strip(), boundary.end_event.strip()}
    for row in rows:
        if row.end <= (row.start if row.start is not None else 0):
            reason = "invalid_interval" if format_name == "trace" else "nonpositive_duration"
            raise ModelingError(f"capture_{format_name}_{reason}:{case_id}")
        if repeated := row.names & found[row.rank]:
            if format_name != "csv" or (repeated := repeated & endpoints):
                reason = "ambiguous_endpoint" if format_name == "csv" else "ambiguous_event"
                raise ModelingError(f"capture_{format_name}_{reason}:{case_id}:{','.join(sorted(repeated))}")
        selected[row.rank].append(row)
        found[row.rank].update(row.names)
    if missing_ranks := selected.keys() - seen_ranks:
        raise ModelingError(f"capture_rank_mismatch:{case_id}:missing={sorted(missing_ranks)}")
    for rank, names in found.items():
        if missing := target_events - names:
            raise ModelingError(f"capture_event_mismatch:{case_id}:rank={rank}:missing={','.join(sorted(missing))}")
    return selected


def _verify_kernel_count(case_id: str, rank: int, count: int) -> None:
    if count != 1:
        raise ModelingError(f"capture_device_kernel_requires_one_occurrence:{case_id}:rank={rank}:count={count}")


def _verify_timed_capture(
    case_id: str, boundary: Any, format_name: str, target_events: frozenset[str],
    rows: Callable[[_RankWindows | None], Iterator[_CaptureRow]], seen_ranks: set[int],
) -> tuple[int, CaptureVerificationReceipt]:
    if not boundary.device_ranks:
        raise ModelingError(f"capture_rank_mismatch:{case_id}:requested_ranks_absent")
    start_name, end_name = boundary.start_event.strip(), boundary.end_event.strip()
    kind = getattr(boundary, "boundary_kind", "")
    single_kernel, step = kind == "device_kernel", kind in L3_BOUNDARY_KINDS
    selected = _selected_rows(case_id, format_name, boundary, target_events, rows(None), seen_ranks)
    if single_kernel:
        for rank, local_rows in selected.items():
            _verify_kernel_count(case_id, rank, len(local_rows))
        if start_name != end_name:
            raise ModelingError(f"capture_device_kernel_endpoint_mismatch:{case_id}")
    occurrences = frozenset(row.occurrence for local_rows in selected.values() for row in local_rows)
    if any(row.start is None for local_rows in selected.values() for row in local_rows):
        if not single_kernel and (len(occurrences) != 1 or len(target_events) != 1):
            raise ModelingError(f"capture_csv_timestamps_absent:{case_id}")
        return round(max(row.end for local_rows in selected.values() for row in local_rows)), CaptureVerificationReceipt(occurrences)

    windows: dict[int, tuple[Decimal, Decimal]] = {}
    for rank, local_rows in selected.items():
        left = next(row.start for row in local_rows if start_name in row.names)
        right = next(row.end for row in local_rows if end_name in row.names)
        if right <= left or any(row.start < left or row.end > right for row in local_rows):
            raise ModelingError(f"capture_{format_name}_event_outside_boundary:{case_id}")
        windows[rank] = left, right

    operator = boundary.clock_operator
    census: set[_KernelOccurrence] = set()
    spans: dict[int, list[tuple[Decimal, Decimal]]] = {rank: [] for rank in windows}
    if single_kernel or step:
        for row in rows(windows):
            if row.end < row.start:
                raise ModelingError(f"capture_{format_name}_nonpositive_duration:{case_id}")
            if row.end == row.start:
                continue
            if single_kernel and row.occurrence not in occurrences:
                raise ModelingError(f"capture_device_kernel_multiple_occurrences:{case_id}:rank={row.rank}")
            if step:
                census.add(row.occurrence)
                if operator == "busy_union":
                    spans[row.rank].append((row.start, row.end))
    if not step and operator == "busy_union":
        spans = {rank: [(row.start, row.end) for row in local_rows] for rank, local_rows in selected.items()}
    duration = round(max(_timed_rank_duration(case_id, format_name, operator, left, right, spans[rank])
                         for rank, (left, right) in windows.items()))
    return duration, CaptureVerificationReceipt(occurrences, frozenset(census) if step else None)


def _verify_duration(case_id: str, observed_ns: int, measured_ns: int | None) -> None:
    if measured_ns is None or abs(observed_ns - measured_ns) > _CAPTURE_ROUNDING_NS:
        raise ModelingError(f"capture_duration_mismatch:{case_id}:declared={measured_ns}:actual={observed_ns}")


def verify_capture_file_consistency(
    case_id: str, boundary: Any, path: Path, *, ijson_mod: Any = None,
    digest_fn: Callable[[Path], str],
    stream_chunk_bytes: int,
    extract_meta_fn: Callable[..., dict[str, Any]],
    measured_ns: int | None = None,
) -> CaptureVerificationReceipt | None:
    """Verify the linked raw capture before scoring the declared observation.

    For multiple ranks, the longest verified local window represents the capture
    duration. CSV ``row=N`` and trace ``row:N`` are one-based source records.
    Trace ``task:``, ``conn:``, ``stream:<id>:task:``, and
    ``dev:<rank>:stream:<id>:task:`` selectors are tied to their source fields.
    """
    try:
        actual_sha = digest_fn(path)
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"capture_file_unreadable:{case_id}") from exc
    expected_sha = str(boundary.raw_sha256).strip().lower()
    if actual_sha != expected_sha:
        raise ModelingError(f"capture_digest_mismatch:{case_id}:expected={expected_sha}:actual={actual_sha}")
    start_name, end_name = str(boundary.start_event).strip(), str(boundary.end_event).strip()
    target_events = frozenset({start_name, end_name, *(ev.strip() for ev in boundary.event_ids if ev.strip())})
    read_errors = (OSError, EOFError, StopIteration, ValueError, InvalidOperation, csv.Error, AttributeError, KeyError)
    if json_error := getattr(ijson_mod, "JSONError", None):
        read_errors += (json_error,)
    receipt = None
    seen_ranks: set[int] = set()
    try:
        if path.name.lower().endswith((".csv", ".csv.gz")):
            rows = partial(_csv_capture_rows, path, case_id, boundary.device_ranks, target_events, seen_ranks)
            duration, receipt = _verify_timed_capture(case_id, boundary, "csv", target_events, rows, seen_ranks)
            _verify_duration(case_id, duration, measured_ns)
            return receipt
        if path.suffix.lower() not in _JSON_SUFFIXES:
            with path.open("rb") as handle:
                if not handle.read(stream_chunk_bytes).lstrip().startswith((b"{", b"[")):
                    raise ModelingError(f"capture_format_unsupported:{case_id}")
        meta = extract_meta_fn(
            path, ijson_mod=ijson_mod,
            target_events=target_events,
            expected_clock_operator=boundary.clock_operator,
        )
        if "_capture_rows" in meta:
            trace_format = "_trace_processes" in meta
            rows = partial(_json_capture_rows, case_id, boundary, meta, target_events, trace_format, seen_ranks)
            duration, receipt = _verify_timed_capture(
                case_id, boundary, "trace" if trace_format else "json", target_events, rows, seen_ranks,
            )
            _verify_duration(case_id, duration, measured_ns)
            meta["event_ids"] = sorted(target_events)
    except TimeoutError:
        raise
    except ModelingError:
        raise
    except read_errors as exc:
        raise ModelingError(f"capture_file_unreadable:{case_id}") from exc
    if meta.get("_clock_mismatch"):
        raise ModelingError(f"capture_clock_operator_mismatch:{case_id}:expected={boundary.clock_operator}")
    cap_ranks = _capture_ranks(meta)
    if cap_ranks is not None and cap_ranks != set(boundary.device_ranks):
        raise ModelingError(f"capture_rank_mismatch:{case_id}:expected={sorted(boundary.device_ranks)}:actual={sorted(cap_ranks)}")
    cap_events, matched_start, matched_end = set(), False, False
    for container in (meta, meta.get("boundary"), meta.get("metadata")):
        if not isinstance(container, Mapping):
            continue
        if "clock_operator" in container and container["clock_operator"] != boundary.clock_operator:
            raise ModelingError(f"capture_clock_operator_mismatch:{case_id}:expected={boundary.clock_operator}")
        if (cap_clock := container.get("clock_domain")) is not None and str(cap_clock).strip() != str(boundary.clock_domain).strip():
            raise ModelingError(f"capture_clock_domain_mismatch:{case_id}:expected={boundary.clock_domain}:actual={cap_clock}")
        if (cap_start := container.get("start_event")) is not None:
            if str(cap_start).strip() != start_name:
                raise ModelingError(f"capture_event_mismatch:{case_id}:start_event")
            matched_start = True
        if (cap_end := container.get("end_event")) is not None:
            if str(cap_end).strip() != end_name:
                raise ModelingError(f"capture_event_mismatch:{case_id}:end_event")
            matched_end = True
        for key in ("event_ids", "events"):
            raw_ev = container.get(key)
            cap_events |= _collect_event_names(raw_ev)
            if isinstance(raw_ev, Sequence) and not isinstance(raw_ev, (str, bytes)):
                for item in raw_ev:
                    if isinstance(item, Mapping) and "clock_operator" in item and (_collect_event_names((item,)) & target_events):
                        if item["clock_operator"] != boundary.clock_operator:
                            raise ModelingError(f"capture_clock_operator_mismatch:{case_id}:expected={boundary.clock_operator}")
    if not matched_start and start_name not in cap_events:
        raise ModelingError(f"capture_event_mismatch:{case_id}:start_event_absent:{start_name}")
    if not matched_end and end_name not in cap_events:
        raise ModelingError(f"capture_event_mismatch:{case_id}:end_event_absent:{end_name}")
    if boundary.event_ids:
        known = cap_events | ({start_name} if matched_start else set()) | ({end_name} if matched_end else set())
        if missing := [ev for ev in boundary.event_ids if ev.strip() not in known]:
            raise ModelingError(f"capture_event_mismatch:{case_id}:missing={','.join(missing)}")
    if receipt is None and getattr(boundary, "boundary_kind", "") == "device_kernel":
        raise ModelingError(f"capture_device_kernel_occurrence_unavailable:{case_id}")
    return receipt
