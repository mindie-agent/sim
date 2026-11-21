"""On-disk capture path resolution and streaming consistency verification.

[Role] Resolve optional local capture files from capture URIs or source
  references and verify streaming SHA-256 digests, device ranks, clock
  domains/operators, and boundary event identifiers across JSON and JSONL.
[Invariants] Fails closed with ``ModelingError`` on unsafe capture URI paths,
  digest mismatches, rank mismatches, or clock/event mismatches on either
  header metadata or matched event rows; never loads whole large files.
[Timeouts & Locks] Synchronous 64 KiB chunked streaming; zero locks or I/O
  outside read-only access to the caller-supplied capture path.
[Side-Effect Policy] Strictly read-only over local capture files.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from mindie_sim.schema.common import ModelingError

CAPTURE_URI_SCHEME = "capture://"
_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})
_CAPTURE_META_KEYS = frozenset({
    "device_ranks", "ranks", "profiled_ranks", "device_rank", "rank",
    "clock_domain", "clock_operator", "start_event", "end_event", "event_ids", "events",
    "boundary", "metadata",
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


def _append_unique_clock(bucket: list[Any], value: Any) -> None:
    for existing in bucket:
        if type(existing) is type(value) and existing == value:
            return
    bucket.append(value)


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
        merged: dict[str, Any] = {}
        events_found: set[str] = set()
        event_clocks: dict[str, list[Any]] = {}
        boundary_clocks: list[Any] = []
        clock_mismatch = False
        meta_keys = _CAPTURE_META_KEYS - {"events", "event_ids"}
        for row in stream_jsonl_fn(path):
            for key in meta_keys:
                if key in row and key not in merged:
                    merged[key] = row[key]
            row_events, row_clocks, item_clocks, has_boundary = _row_event_and_clock_info(row)
            if target_events is not None:
                matched_row = row_events & target_events
                if matched_row:
                    events_found |= matched_row
                if expected_clock_operator is not None:
                    if row_clocks and (has_boundary or not row_events or matched_row):
                        if any(clk != expected_clock_operator for clk in row_clocks):
                            clock_mismatch = True
                            break
                    for ev, clocks in item_clocks.items():
                        if ev in target_events and any(clk != expected_clock_operator for clk in clocks):
                            clock_mismatch = True
                            break
                    if clock_mismatch:
                        break
                    continue
                if row_clocks:
                    for ev in matched_row:
                        bucket = event_clocks.setdefault(ev, [])
                        for clk in row_clocks:
                            _append_unique_clock(bucket, clk)
                    if has_boundary or not row_events:
                        for clk in row_clocks:
                            _append_unique_clock(boundary_clocks, clk)
                for ev, clocks in item_clocks.items():
                    if ev in target_events:
                        bucket = event_clocks.setdefault(ev, [])
                        for clk in clocks:
                            _append_unique_clock(bucket, clk)
            else:
                events_found |= row_events
                if row_clocks:
                    for ev in row_events:
                        bucket = event_clocks.setdefault(ev, [])
                        for clk in row_clocks:
                            _append_unique_clock(bucket, clk)
                    if has_boundary or not row_events:
                        for clk in row_clocks:
                            _append_unique_clock(boundary_clocks, clk)
                for ev, clocks in item_clocks.items():
                    bucket = event_clocks.setdefault(ev, [])
                    for clk in clocks:
                        _append_unique_clock(bucket, clk)
        if events_found:
            merged["event_ids"] = sorted(events_found)
        if event_clocks:
            merged["_event_clocks"] = event_clocks
        if boundary_clocks:
            merged["_boundary_clocks"] = boundary_clocks
        if clock_mismatch:
            merged["_clock_mismatch"] = True
        return merged
    meta: dict[str, Any] = {}
    with path.open("rb") as handle:
        events = iter_json_events_fn(handle, ijson_mod=ijson_mod)
        if next(events)[0] != "start_map":
            return meta
        while (entry := next(events))[0] != "end_map":
            key = str(entry[1])
            meta[key] = read_value_fn(events) if key in _CAPTURE_META_KEYS else skip_value_fn(events)
            if key not in _CAPTURE_META_KEYS:
                meta.pop(key, None)
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


def verify_capture_file_consistency(
    case_id: str, boundary: Any, path: Path, *, ijson_mod: Any = None,
    digest_fn: Callable[[Path], str],
    stream_chunk_bytes: int,
    extract_meta_fn: Callable[..., dict[str, Any]],
) -> None:
    """Fail closed when an on-disk capture file mismatches digest, rank, clock, or events."""
    try:
        actual_sha = digest_fn(path)
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"capture_file_unreadable:{case_id}") from exc
    expected_sha = str(boundary.raw_sha256).strip().lower()
    if actual_sha != expected_sha:
        raise ModelingError(f"capture_digest_mismatch:{case_id}:expected={expected_sha}:actual={actual_sha}")
    is_json_file = path.suffix.lower() in _JSON_SUFFIXES
    if not is_json_file:
        with path.open("rb") as handle:
            is_json_file = handle.read(stream_chunk_bytes).lstrip().startswith((b"{", b"["))
    if not is_json_file:
        return
    start_name, end_name = str(boundary.start_event).strip(), str(boundary.end_event).strip()
    target_events = frozenset({start_name, end_name, *(ev.strip() for ev in boundary.event_ids if ev.strip())})
    try:
        meta = extract_meta_fn(
            path, ijson_mod=ijson_mod,
            target_events=target_events,
            expected_clock_operator=boundary.clock_operator,
        )
    except TimeoutError:
        raise
    except ModelingError as exc:
        raise ModelingError(f"capture_file_unreadable:{case_id}:{exc.reason}") from exc
    except (OSError, StopIteration) as exc:
        raise ModelingError(f"capture_file_unreadable:{case_id}") from exc
    if meta.get("_clock_mismatch"):
        raise ModelingError(f"capture_clock_operator_mismatch:{case_id}:expected={boundary.clock_operator}")
    cap_ranks = _capture_ranks(meta)
    if cap_ranks is not None and cap_ranks != set(boundary.device_ranks):
        raise ModelingError(f"capture_rank_mismatch:{case_id}:expected={sorted(boundary.device_ranks)}:actual={sorted(cap_ranks)}")
    cap_events, matched_start, matched_end = set(), False, False
    ev_clocks = meta.get("_event_clocks") or {}
    for clk in (*meta.get("_boundary_clocks", ()), *(c for ev in target_events for c in ev_clocks.get(ev, ()))):
        if clk != boundary.clock_operator:
            raise ModelingError(f"capture_clock_operator_mismatch:{case_id}:expected={boundary.clock_operator}")
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
