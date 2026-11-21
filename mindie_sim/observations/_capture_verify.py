"""On-disk capture path resolution and streaming consistency verification.

[Role] Resolve optional local capture files from capture URIs or source
  references and verify streaming SHA-256 digests, device ranks, clock
  domains/operators, and boundary event identifiers across JSON, JSONL, and native profiler CSV.
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
import gzip
from pathlib import Path, PurePosixPath
from typing import Any

from mindie_sim.schema.common import ModelingError

CAPTURE_URI_SCHEME = "capture://"
_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})
_CAPTURE_EVENT_KEYS = frozenset({"events", "event_ids", "traceEvents"})
_CAPTURE_CONTAINERS = frozenset({"boundary", "metadata", "payload"})
_NS_PER_US = 1000
_NS_PER_MS = 1000 * _NS_PER_US
_CSV_TIME_SCALE = {"ns": 1, "us": _NS_PER_US, "ms": _NS_PER_MS}
_CAPTURE_ROUNDING_NS = 1
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
    read_value: Callable[..., Any], skip_value: Callable[..., Any],
) -> Iterator[Mapping[str, Any]]:
    """Yield individual trace rows and compact headers from the streaming parser."""
    if first[0] == "start_array":
        while (item := next(events))[0] != "end_array":
            row = read_value(events, item)
            if isinstance(row, Mapping):
                yield row
            elif isinstance(row, str):
                yield {"event_id": row}
        return
    if first[0] != "start_map":
        raise ModelingError("capture_json_requires_object_or_array")
    header = {}
    while (entry := next(events))[0] != "end_map":
        key, item = str(entry[1]), next(events)
        if key in _CAPTURE_META_KEYS or key in {"event_id", "id", "name"}:
            header[key] = read_value(events, item)
        elif key in _CAPTURE_EVENT_KEYS and item[0] == "start_array":
            yield from _iter_capture_rows(events, item, read_value=read_value, skip_value=skip_value)
        elif key in _CAPTURE_CONTAINERS and item[0] == "start_map":
            yield from _iter_capture_rows(events, item, read_value=read_value, skip_value=skip_value)
        else:
            skip_value(events, item)
    if header:
        yield header


def _capture_metadata(
    rows: Iterator[Mapping[str, Any]], target_events: frozenset[str] | None,
    expected_clock_operator: str | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    found: set[str] = set()
    rank_columns: set[int] = set()
    for row in rows:
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
        return _capture_metadata(stream_jsonl_fn(path), target_events, expected_clock_operator)
    with path.open("rb") as handle:
        events = iter_json_events_fn(handle, ijson_mod=ijson_mod)
        rows = _iter_capture_rows(events, next(events), read_value=read_value_fn, skip_value=skip_value_fn)
        return _capture_metadata(rows, target_events, expected_clock_operator)


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

    return round(max(_csv_rank_duration(case_id, boundary.clock_operator, start_name, end_name, rows)
                     for rows in timed.values()))


def _csv_rank_duration(
    case_id: str, clock_operator: str, start_name: str, end_name: str,
    rows: list[tuple[frozenset[str], Decimal, Decimal]],
) -> Decimal:
    left = next(start for names, start, _ in rows if start_name in names)
    right = next(end for names, _, end in rows if end_name in names)
    if any(start < left or end > right for _, start, end in rows) or right <= left:
        raise ModelingError(f"capture_csv_event_outside_boundary:{case_id}")
    if clock_operator == "elapsed_span":
        return right - left
    if clock_operator != "busy_union":
        raise ModelingError(f"capture_csv_clock_unsupported:{case_id}")
    busy, cursor = Decimal(0), left
    for start, end in sorted((start, end) for _, start, end in rows):
        busy += max(0, end - max(start, cursor))
        cursor = max(cursor, end)
    return busy


def verify_capture_file_consistency(
    case_id: str, boundary: Any, path: Path, *, ijson_mod: Any = None,
    digest_fn: Callable[[Path], str],
    stream_chunk_bytes: int,
    extract_meta_fn: Callable[..., dict[str, Any]],
    measured_ns: int | None = None,
) -> None:
    """Verify the linked raw capture before scoring the declared observation.

    CSV timing is compared at nanosecond precision. A row=N event selects
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
            if measured_ns is None or abs(duration - measured_ns) > _CAPTURE_ROUNDING_NS:
                raise ModelingError(f"capture_duration_mismatch:{case_id}:declared={measured_ns}:actual={duration}")
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
