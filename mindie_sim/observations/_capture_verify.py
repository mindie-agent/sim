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

from collections.abc import Callable, Iterator, Mapping, Sequence
import csv
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
# msprof packs the device ID below the track type and host PID.
_ASCEND_DEVICE_ID_MASK = 0x1F
_ASCEND_TRACK_TYPE_MASK = 0x3E0
_ASCEND_HARDWARE_TRACK = 0x1C0
_TRACE_DEVICE_LABEL = re.compile(r"(?:device|npu|rank)(?: id)?[ _:-]+(?P<rank>[0-9]+)", re.IGNORECASE)
_TRACE_RANK_KEYS = ("device_rank", "rank_id", "rank", "device_id", "Device_id", "Device Id", "Device ID")
_TRACE_EVENT_ID_KEYS = ("event_id", "id", "name", "Task Id", "connection_id")
_CAPTURE_META_KEYS = frozenset({
    "device_ranks", "ranks", "profiled_ranks", "device_rank", "rank",
    "clock_domain", "clock_operator", "start_event", "end_event",
})


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
        if trace:
            yield True, {}
        while (item := next(events))[0] != "end_array":
            row = read_value(events, item)
            if isinstance(row, Mapping):
                yield trace or "ph" in row and ("pid" in row or "ts" in row), row
            elif isinstance(row, str) and not trace:
                yield False, {"event_id": row}
        return
    if first[0] != "start_map":
        raise ModelingError("capture_json_requires_object_or_array")
    header = {}
    while (entry := next(events))[0] != "end_map":
        key, item = str(entry[1]), next(events)
        kind, _ = item
        if key in _CAPTURE_META_KEYS or key in {"event_id", "id", "name"}:
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
        yield False, header


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
    if path.suffix.lower() in _JSONL_SUFFIXES:
        return _capture_metadata(((False, row) for row in stream_jsonl_fn(path)), target_events, expected_clock_operator)
    rows = partial(_read_capture_rows, path, ijson_mod=ijson_mod, iter_json_events_fn=iter_json_events_fn,
                   read_value_fn=read_value_fn, skip_value_fn=skip_value_fn)
    meta = _capture_metadata(rows(), target_events, expected_clock_operator)
    if "_trace_processes" in meta:
        meta["_trace_rows"] = rows
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


def _csv_time_ns(row: Mapping[str, str], field: tuple[str, int]) -> Decimal:
    column, scale = field
    value = Decimal(row[column].strip()) * scale
    if not value.is_finite():
        raise ValueError("non_finite_csv_time")
    return value


def _csv_capture_rows(
    path: Path, case_id: str, ranks: set[int], target_events: frozenset[str],
) -> Iterator[tuple[int, frozenset[str], Decimal | None, Decimal]]:
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
        seen_ranks = set()
        for record, row in enumerate(reader, start=1):
            rank = int(row[rank_col].strip())
            if rank not in ranks:
                continue
            seen_ranks.add(rank)
            row_id = f"row={record}"
            names = target_events.intersection(row[col].strip() for col in event_cols)
            if row_id in target_events:
                names = names | {row_id}
            if not names:
                continue
            if not start_col:
                if int(row[fields["count"]]) != 1 or total_col is None:
                    raise ModelingError(f"capture_csv_aggregate_event:{case_id}")
                yield rank, names, None, _csv_time_ns(row, total_col)
                continue
            start = _csv_time_ns(row, start_col)
            end = _csv_time_ns(row, end_col) if end_col else start + _csv_time_ns(row, duration_col)
            if duration_col and abs(end - start - _csv_time_ns(row, duration_col)) > _CAPTURE_ROUNDING_NS:
                raise ModelingError(f"capture_csv_interval_inconsistent:{case_id}")
            yield rank, names, start, end
        if ranks - seen_ranks:
            raise ModelingError(f"capture_rank_mismatch:{case_id}:missing={sorted(ranks - seen_ranks)}")


def _csv_capture_duration(
    case_id: str, boundary: Any, path: Path, target_events: frozenset[str],
) -> int:
    start_name, end_name = boundary.start_event.strip(), boundary.end_event.strip()
    ranks = set(boundary.device_ranks)
    if not ranks:
        raise ModelingError(f"capture_rank_mismatch:{case_id}:requested_ranks_absent")
    timed = {rank: [] for rank in ranks}
    untimed = []
    found: dict[int, set[str]] = {rank: set() for rank in ranks}
    endpoint_counts: set[tuple[int, str]] = set()
    for rank, names, start, end in _csv_capture_rows(path, case_id, ranks, target_events):
        if start is None:
            untimed.append(end)
        else:
            timed[rank].append((names, start, end))
        if end <= (start if start is not None else 0):
            raise ModelingError(f"capture_csv_nonpositive_duration:{case_id}")
        found[rank].update(names)
        for name in names & {start_name, end_name}:
            key = (rank, name)
            if key in endpoint_counts:
                raise ModelingError(f"capture_csv_ambiguous_endpoint:{case_id}:{name}")
            endpoint_counts.add(key)
    for rank, names in found.items():
        if missing := target_events - names:
            raise ModelingError(f"capture_event_mismatch:{case_id}:rank={rank}:missing={','.join(sorted(missing))}")
    if untimed:
        if len(untimed) != 1 or len(target_events) != 1 or any(timed.values()):
            raise ModelingError(f"capture_csv_timestamps_absent:{case_id}")
        return round(untimed[0])

    return round(max(_timed_rank_duration(case_id, "csv", boundary.clock_operator, start_name, end_name, rows)
                     for rows in timed.values()))


def _timed_rank_duration(
    case_id: str, format_name: str, clock_operator: str, start_name: str, end_name: str,
    rows: list[tuple[frozenset[str], Decimal, Decimal]],
) -> Decimal:
    left = next(start for names, start, _ in rows if start_name in names)
    right = next(end for names, _, end in rows if end_name in names)
    if any(start < left or end > right for _, start, end in rows) or right <= left:
        raise ModelingError(f"capture_{format_name}_event_outside_boundary:{case_id}")
    if clock_operator == "elapsed_span":
        return right - left
    if clock_operator != "busy_union":
        raise ModelingError(f"capture_{format_name}_clock_unsupported:{case_id}")
    busy, cursor = Decimal(0), left
    for start, end in sorted((start, end) for _, start, end in rows):
        busy += max(0, end - max(start, cursor))
        cursor = max(cursor, end)
    return busy


def _trace_capture_duration(
    case_id: str, boundary: Any, meta: Mapping[str, Any], target_events: frozenset[str],
) -> int:
    ranks = set(boundary.device_ranks)
    if not ranks:
        raise ModelingError(f"capture_rank_mismatch:{case_id}:requested_ranks_absent")
    timed = {rank: [] for rank in ranks}
    found: dict[int, set[str]] = {rank: set() for rank in ranks}
    seen_ranks = set()
    for trace, row in meta["_trace_rows"]():
        if not trace or row.get("ph") != "X":
            continue
        process_rank = meta["_trace_processes"].get(row.get("pid"))
        explicit_rank = _trace_rank(row)
        rank = explicit_rank if explicit_rank is not None else process_rank
        if rank not in ranks:
            continue
        seen_ranks.add(rank)
        names = target_events.intersection(
            str(value).strip() for source in (row, _trace_args(row)) for key in _TRACE_EVENT_ID_KEYS
            if isinstance(value := source.get(key), (str, int)) and not isinstance(value, bool)
        )
        if not names:
            continue
        if process_rank is not None and rank != process_rank:
            raise ModelingError(f"capture_rank_mismatch:{case_id}:trace_process_conflict")
        if row.get("clock_operator", boundary.clock_operator) != boundary.clock_operator:
            raise ModelingError(f"capture_clock_operator_mismatch:{case_id}:expected={boundary.clock_operator}")
        if row.get("clock_domain", boundary.clock_domain) != boundary.clock_domain:
            raise ModelingError(f"capture_clock_domain_mismatch:{case_id}:expected={boundary.clock_domain}")
        if repeated := names & found[rank]:
            raise ModelingError(f"capture_trace_ambiguous_event:{case_id}:{','.join(sorted(repeated))}")
        start = Decimal(str(row["ts"])) * _NS_PER_US
        duration = Decimal(str(row["dur"])) * _NS_PER_US
        if not start.is_finite() or not duration.is_finite() or duration.is_signed() or duration.is_zero():
            raise ModelingError(f"capture_trace_invalid_interval:{case_id}")
        found[rank].update(names)
        timed[rank].append((names, start, start + duration))
    if missing_ranks := ranks - seen_ranks:
        raise ModelingError(f"capture_rank_mismatch:{case_id}:missing={sorted(missing_ranks)}")
    for rank, names in found.items():
        if missing := target_events - names:
            raise ModelingError(f"capture_event_mismatch:{case_id}:rank={rank}:missing={','.join(sorted(missing))}")
    return round(max(_timed_rank_duration(case_id, "trace", boundary.clock_operator,
                                         boundary.start_event.strip(), boundary.end_event.strip(), rows)
                     for rows in timed.values()))


def _verify_duration(case_id: str, observed_ns: int, measured_ns: int | None) -> None:
    if measured_ns is None or abs(observed_ns - measured_ns) > _CAPTURE_ROUNDING_NS:
        raise ModelingError(f"capture_duration_mismatch:{case_id}:declared={measured_ns}:actual={observed_ns}")


def verify_capture_file_consistency(
    case_id: str, boundary: Any, path: Path, *, ijson_mod: Any = None,
    digest_fn: Callable[[Path], str],
    stream_chunk_bytes: int,
    extract_meta_fn: Callable[..., dict[str, Any]],
    measured_ns: int | None = None,
) -> None:
    """Verify the linked raw capture before scoring the declared observation.

    Chrome and CSV timing are compared in nanoseconds. A CSV row=N event selects
    the Nth data record when native task names or IDs repeat. For multiple
    ranks, the longest verified local window represents the capture duration.
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
    try:
        if path.name.lower().endswith((".csv", ".csv.gz")):
            duration = _csv_capture_duration(case_id, boundary, path, target_events)
            _verify_duration(case_id, duration, measured_ns)
            return
        if path.suffix.lower() not in _JSON_SUFFIXES:
            with path.open("rb") as handle:
                if not handle.read(stream_chunk_bytes).lstrip().startswith((b"{", b"[")):
                    raise ModelingError(f"capture_format_unsupported:{case_id}")
        meta = extract_meta_fn(
            path, ijson_mod=ijson_mod,
            target_events=target_events,
            expected_clock_operator=boundary.clock_operator,
        )
        if "_trace_rows" in meta:
            duration = _trace_capture_duration(case_id, boundary, meta, target_events)
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
