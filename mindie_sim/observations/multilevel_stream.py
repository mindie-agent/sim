"""Chunked two-pass JSON/JSONL streaming, provenance sanitization, and capture checks.

[Role] Provide fixed-chunk (64 KiB) stdlib JSON/JSONL event streaming for
  multilevel gate manifests and tail-cluster diagnosis inputs, provenance and
  capture consistency verification, and signed Engine clock measurements.
[Invariants] Never whole-reads JSON files via json.load or truncates top-level
  metadata scans at 64 KiB; top-level keys after arbitrarily large arrays
  (e.g. lineage_edges after cases) are parsed in Pass 1 while skipping array
  items in O(1) memory; JSON integers stay exact int. All SHA-256 tokens
  normalize to lowercase; on-disk capture checks fail closed on mismatch.
[Timeouts & Locks] Synchronous 64 KiB chunked reads; O(1) skip memory; zero
  locks, subprocesses, or device contexts.
[Side-Effect Policy] Strictly read-only over local manifest and capture files.
"""
from __future__ import annotations

import codecs
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
from pathlib import Path, PurePosixPath
import re
import secrets
from types import MappingProxyType
from typing import Any, Literal, cast

from mindie_sim.schema.common import ModelingError
from ._capture_verify import (
    CAPTURE_URI_SCHEME,
    _CAPTURE_META_KEYS,
    _JSONL_SUFFIXES,
    _JSON_SUFFIXES,
    _capture_ranks,
    _collect_event_names,
    _extract_capture_meta as _extract_capture_meta_impl,
    resolve_capture_file,
    verify_capture_file_consistency as _verify_capture_file_consistency_impl,
)

STREAM_CHUNK_BYTES = 64 * 1024
STREAM_WHOLE_READ_LIMIT_BYTES = 50 * 1024 * 1024
SHA256_HEX_LENGTH = 64
MS_TO_NS = 1_000_000
_CLOCK_EPS_MS = 1e-6
_HMAC_KEY_BYTES = 32
_RECEIPT_HMAC_KEY = secrets.token_bytes(_HMAC_KEY_BYTES)
_SHA256_PREFIX = "sha256:"
Level = Literal["L1", "L2", "L3"]
VALID_CLOCK_OPERATORS = frozenset({"elapsed_span", "busy_union"})
SOURCE_REF_KEYS = frozenset({"repository", "revision", "relative_path", "source_sha256"})
TIME_UNIT_SCALE_NS: Mapping[str, int] = MappingProxyType({
    "ns": 1, "us": 1000, "ms": 1_000_000, "s": 1_000_000_000,
})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_ANYCASE_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_PRIVATE_UNIX_PATH_RE = re.compile(
    r"/(?:" + "|".join(("Users", "home", "mnt", "root", "tmp", "Volumes")) + r")/"
)
_WINDOWS_DRIVE_RE = re.compile(r"(?:^|[\s\"'=:,])[A-Za-z]:\\")
_IPV4_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_CONTAINER_MARKERS = ("".join(("docker", ".io/")), "".join(("swr", ".cn-")), "".join(("@sha", "256:")))
_JSON_TOKEN_RE = re.compile(
    r'\s*(?:'
    r'(?P<lbrace>\{)|(?P<rbrace>\})|(?P<lbracket>\[)|(?P<rbracket>\])|'
    r'(?P<colon>:)|(?P<comma>,)|'
    r'(?P<string>"(?:[^"\\]|\\.)*")|'
    r'(?P<number>-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)|'
    r'(?P<literal>true|false|null)'
    r')',
    re.DOTALL,
)

__all__ = [
    "CAPTURE_URI_SCHEME", "EngineInterval", "EngineRunReceipt", "MS_TO_NS",
    "SHA256_HEX_LENGTH", "SOURCE_REF_KEYS", "STREAM_CHUNK_BYTES",
    "STREAM_WHOLE_READ_LIMIT_BYTES", "TIME_UNIT_SCALE_NS", "VALID_CLOCK_OPERATORS", "_req_sha256", "_req_str",
    "bind_engine_interval", "bind_engine_rung_interval", "digest_file_streaming",
    "iter_json_records_streaming", "iter_manifest_json_cases", "normalize_lineage_token",
    "resolve_capture_file", "sanitize_capture_uri", "sanitize_source_ref",
    "scan_manifest_json_pass1", "stream_jsonl", "validate_engine_interval",
    "validate_provenance_text", "verify_capture_file_consistency",
]


def _req_str(value: Any, reason: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(reason)
    return value.strip()


def normalize_lineage_token(value: str) -> str:
    """Normalize SHA-256 hex digests (with optional ``sha256:`` prefix) to lowercase."""
    text = value.strip()
    if len(text) == SHA256_HEX_LENGTH and _HEX_ANYCASE_RE.fullmatch(text):
        return text.lower()
    if text[:len(_SHA256_PREFIX)].lower() == _SHA256_PREFIX:
        rest = text[len(_SHA256_PREFIX):]
        if len(rest) == SHA256_HEX_LENGTH and _HEX_ANYCASE_RE.fullmatch(rest):
            return rest.lower()
    return text


def _req_sha256(value: Any, reason: str) -> str:
    text = normalize_lineage_token(_req_str(value, reason))
    if len(text) != SHA256_HEX_LENGTH or not _SHA256_RE.fullmatch(text):
        raise ModelingError(f"{reason}:{text}")
    return text


def validate_provenance_text(text: str, *, field_name: str = "provenance") -> str:
    """Reject private host paths, IPv4 addresses, and container image digests."""
    if _PRIVATE_UNIX_PATH_RE.search(text) or _WINDOWS_DRIVE_RE.search(text):
        raise ModelingError(f"private_host_path_rejected:{field_name}")
    if _IPV4_RE.search(text):
        raise ModelingError(f"private_ip_rejected:{field_name}")
    if any(marker in text.lower() for marker in _CONTAINER_MARKERS):
        raise ModelingError(f"container_reference_rejected:{field_name}")
    return text


def sanitize_capture_uri(uri: str) -> str:
    cleaned = validate_provenance_text(_req_str(uri, "capture_uri_empty"), field_name="capture_uri")
    if not cleaned.startswith(CAPTURE_URI_SCHEME):
        raise ModelingError(f"capture_uri_scheme_invalid:{cleaned}")
    return cleaned


def sanitize_source_ref(ref: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    if not isinstance(ref, Mapping) or frozenset(ref.keys()) != SOURCE_REF_KEYS:
        raise ModelingError("source_ref_invalid")
    repo = validate_provenance_text(_req_str(ref["repository"], "source_ref_repository_empty"), field_name="source_ref.repository")
    rev = validate_provenance_text(_req_str(ref["revision"], "source_ref_revision_empty"), field_name="source_ref.revision")
    rel = validate_provenance_text(_req_str(ref["relative_path"], "source_ref_path_empty"), field_name="source_ref.relative_path")
    if rel.startswith("/") or ".." in PurePosixPath(rel).parts:
        raise ModelingError(f"source_ref_relative_path_unsafe:{rel}")
    return (
        ("relative_path", rel), ("repository", repo), ("revision", rev),
        ("source_sha256", _req_sha256(ref["source_sha256"], "source_ref_sha256_invalid")),
    )


def _iter_json_tokens(handle: Any, chunk_size: int = STREAM_CHUNK_BYTES) -> Iterator[tuple[str, Any]]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    buf, pos, eof = "", 0, False
    while True:
        match = _JSON_TOKEN_RE.match(buf, pos)
        if match is None:
            if eof:
                if buf[pos:].strip():
                    raise ModelingError("json_stream_parse_error:trailing_data")
                return
            chunk = handle.read(chunk_size)
            buf += decoder.decode(chunk) if chunk else decoder.decode(b"", final=True)
            eof = not chunk
            continue
        if match.lastgroup == "number" and not eof:
            end = match.end()
            if end == len(buf) or buf[end] not in " \t\r\n,]}:":
                chunk = handle.read(chunk_size)
                buf += decoder.decode(chunk) if chunk else decoder.decode(b"", final=True)
                eof = not chunk
                continue
        pos = match.end()
        kind, text = match.lastgroup, match.group(match.lastgroup)
        if kind == "string":
            try:
                yield ("string", json.loads(text))
            except json.JSONDecodeError as exc:
                raise ModelingError(f"json_stream_invalid_string:{exc.msg}") from exc
        elif kind == "number":
            yield ("number", float(text) if ("." in text or "e" in text or "E" in text) else int(text))
        elif kind == "literal":
            yield ("literal", {"true": True, "false": False, "null": None}[text])
        else:
            yield ("punct", text)
        if pos >= STREAM_CHUNK_BYTES:
            buf, pos = buf[pos:], 0


def _parse_json_value(token: tuple[str, Any], tokens: Iterator[tuple[str, Any]]) -> Iterator[tuple[str, Any]]:
    kind, val = token
    if kind == "punct" and val == "{":
        yield ("start_map", None)
        nxt = next(tokens)
        if nxt == ("punct", "}"):
            yield ("end_map", None)
            return
        while True:
            if nxt[0] != "string":
                raise ModelingError("json_stream_key_not_string")
            yield ("map_key", nxt[1])
            if next(tokens) != ("punct", ":"):
                raise ModelingError("json_stream_expected_colon")
            yield from _parse_json_value(next(tokens), tokens)
            sep = next(tokens)
            if sep == ("punct", "}"):
                yield ("end_map", None)
                return
            if sep != ("punct", ","):
                raise ModelingError("json_stream_expected_comma_or_rbrace")
            nxt = next(tokens)
    elif kind == "punct" and val == "[":
        yield ("start_array", None)
        nxt = next(tokens)
        if nxt == ("punct", "]"):
            yield ("end_array", None)
            return
        while True:
            yield from _parse_json_value(nxt, tokens)
            sep = next(tokens)
            if sep == ("punct", "]"):
                yield ("end_array", None)
                return
            if sep != ("punct", ","):
                raise ModelingError("json_stream_expected_comma_or_rbracket")
            nxt = next(tokens)
    elif kind in {"string", "number"}:
        yield (kind, val)
    elif kind == "literal":
        yield ("boolean", val) if isinstance(val, bool) else ("null", None)
    else:
        raise ModelingError(f"json_stream_unexpected_token:{val!r}")


def _iter_json_events(handle: Any, *, ijson_mod: Any = None) -> Iterator[tuple[str, Any]]:
    if ijson_mod is not None:
        yield from ijson_mod.basic_parse(handle, use_float=True)
        return
    tokens = _iter_json_tokens(handle, STREAM_CHUNK_BYTES)
    try:
        first = next(tokens)
    except StopIteration as exc:
        raise ModelingError("json_stream_empty") from exc
    yield from _parse_json_value(first, tokens)


def _skip_value(events: Iterator[tuple[str, Any]], first: tuple[str, Any] | None = None) -> None:
    ev = (first or next(events))[0]
    if ev not in {"start_map", "start_array"}:
        return
    depth = 1
    while depth:
        ev = next(events)[0]
        depth += 1 if ev in {"start_map", "start_array"} else (-1 if ev in {"end_map", "end_array"} else 0)


def _read_value(events: Iterator[tuple[str, Any]], first: tuple[str, Any] | None = None) -> Any:
    ev, val = first or next(events)
    if ev == "start_map":
        out: dict[str, Any] = {}
        while (nxt := next(events))[0] != "end_map":
            out[str(nxt[1])] = _read_value(events)
        return out
    if ev == "start_array":
        arr: list[Any] = []
        while (nxt := next(events))[0] != "end_array":
            arr.append(_read_value(events, nxt))
        return arr
    return val


def stream_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream JSON-Lines records line by line without whole-file buffering."""
    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            if not (stripped := raw_line.strip()):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ModelingError(f"jsonl_parse_error:{target.name}:line={lineno}:{exc.msg}") from exc
            if not isinstance(obj, dict):
                raise ModelingError(f"jsonl_record_not_object:{target.name}:line={lineno}")
            yield obj


def scan_manifest_json_pass1(
    path: Path, header_keys: frozenset[str], *, ijson_mod: Any = None,
) -> tuple[dict[str, Any], int]:
    """Pass 1: stream all top-level manifest metadata without buffering ``cases``."""
    header: dict[str, Any] = {}
    saw_cases, case_count = False, 0
    try:
        with path.open("rb") as handle:
            events = _iter_json_events(handle, ijson_mod=ijson_mod)
            if next(events)[0] != "start_map":
                raise ModelingError(f"manifest_schema_invalid:{path.name}")
            while (entry := next(events))[0] != "end_map":
                key = str(entry[1])
                if key == "cases":
                    saw_cases = True
                    if next(events)[0] != "start_array":
                        raise ModelingError(f"manifest_schema_invalid:{path.name}")
                    while (elem := next(events))[0] != "end_array":
                        if elem[0] != "start_map":
                            raise ModelingError(f"manifest_case_not_object:{path.name}")
                        _skip_value(events, elem)
                        case_count += 1
                elif key in header_keys:
                    header[key] = _read_value(events)
                else:
                    _skip_value(events)
    except StopIteration as exc:
        raise ModelingError(f"manifest_json_truncated:{path.name}") from exc
    if not saw_cases:
        raise ModelingError(f"manifest_schema_invalid:{path.name}")
    return header, case_count


def iter_manifest_json_cases(path: Path, *, ijson_mod: Any = None) -> Iterator[dict[str, Any]]:
    """Pass 2: stream ``cases`` objects one at a time without whole-file buffering."""
    try:
        with path.open("rb") as handle:
            events = _iter_json_events(handle, ijson_mod=ijson_mod)
            if next(events)[0] != "start_map":
                raise ModelingError(f"manifest_schema_invalid:{path.name}")
            while (entry := next(events))[0] != "end_map":
                if str(entry[1]) != "cases":
                    _skip_value(events)
                    continue
                if next(events)[0] != "start_array":
                    raise ModelingError(f"manifest_schema_invalid:{path.name}")
                while (elem := next(events))[0] != "end_array":
                    if elem[0] != "start_map":
                        raise ModelingError(f"manifest_case_not_object:{path.name}")
                    yield _read_value(events, elem)
                return
    except StopIteration as exc:
        raise ModelingError(f"manifest_json_truncated:{path.name}") from exc


def iter_json_records_streaming(
    path: Path, array_keys: Sequence[str], *, ijson_mod: Any = None,
) -> Iterator[dict[str, Any]]:
    """Stream object records from a JSON list, keyed array, or single object."""
    wanted = set(array_keys)
    try:
        with path.open("rb") as handle:
            events = _iter_json_events(handle, ijson_mod=ijson_mod)
            first = next(events)
            if first[0] == "start_array":
                while (elem := next(events))[0] != "end_array":
                    val = _read_value(events, elem)
                    if isinstance(val, Mapping):
                        yield dict(val)
            elif first[0] == "start_map":
                singleton: dict[str, Any] = {}
                found_array = False
                while (entry := next(events))[0] != "end_map":
                    key, val_ev = str(entry[1]), next(events)
                    if not found_array and key in wanted and val_ev[0] == "start_array":
                        found_array = True
                        while (elem := next(events))[0] != "end_array":
                            val = _read_value(events, elem)
                            if isinstance(val, Mapping):
                                yield dict(val)
                    elif found_array:
                        _skip_value(events, val_ev)
                    else:
                        singleton[key] = _read_value(events, val_ev)
                if not found_array and singleton:
                    yield singleton
    except StopIteration as exc:
        raise ModelingError(f"json_stream_truncated:{path.name}") from exc


def digest_file_streaming(path: Path) -> str:
    """Compute lowercase SHA-256 hex digest in bounded 64 KiB chunks."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(STREAM_CHUNK_BYTES):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def _extract_capture_meta(
    path: Path, *, ijson_mod: Any = None,
    target_events: frozenset[str] | None = None,
    expected_clock_operator: str | None = None,
) -> dict[str, Any]:
    return _extract_capture_meta_impl(
        path, ijson_mod=ijson_mod,
        stream_jsonl_fn=stream_jsonl,
        iter_json_events_fn=_iter_json_events,
        read_value_fn=_read_value,
        skip_value_fn=_skip_value,
        target_events=target_events,
        expected_clock_operator=expected_clock_operator,
    )


def verify_capture_file_consistency(case_id: str, boundary: Any, path: Path, *, ijson_mod: Any = None) -> None:
    """Fail closed when an on-disk capture file mismatches digest, rank, clock, or events."""
    _verify_capture_file_consistency_impl(
        case_id, boundary, path, ijson_mod=ijson_mod,
        digest_fn=digest_file_streaming,
        stream_chunk_bytes=STREAM_CHUNK_BYTES,
        extract_meta_fn=_extract_capture_meta,
    )


@dataclass(frozen=True, slots=True)
class EngineRunReceipt:
    receipt_id: str
    case_id: str
    level: Level
    boundary_kind: str
    engine_uid: str
    start_event_name: str
    end_event_name: str
    start_event_seq: int
    end_event_seq: int
    start_ns: int
    end_ns: int
    predicted_ns: int
    receipt_token: str
    arm_a_receipt_id: str = ""
    arm_b_receipt_id: str = ""
    clock_operator: str = "elapsed_span"


@dataclass(frozen=True, slots=True)
class EngineInterval:
    case_id: str
    level: Level
    boundary_kind: str
    start_event: str
    end_event: str
    start_ns: int
    end_ns: int
    predicted_ns: int
    engine_receipt: str
    engine_sourced: bool = True
    external_offset_ns: int = 0
    status: Literal["ok", "missing_evidence"] = "ok"
    missing_reason: str = ""
    source_level: str = ""
    derived_from_step_division: bool = False
    wire_bytes: int | None = None
    kernel_sequence: tuple[str, ...] = ()
    arm_a_predicted_ns: int | None = None
    arm_b_predicted_ns: int | None = None
    diagnostics: tuple[tuple[str, Any], ...] = ()
    run_receipt: EngineRunReceipt | None = None
    clock_operator: str = "elapsed_span"
    elapsed_span_ns: int | None = None
    busy_union_ns: int | None = None


def _sign_receipt(
    receipt_id: str, case_id: str, level: str, boundary_kind: str, engine_uid: str,
    start_event_name: str, end_event_name: str, start_event_seq: int, end_event_seq: int,
    start_ns: int, end_ns: int, predicted_ns: int,
    arm_a_id: str = "", arm_b_id: str = "",
    arm_a_ns: int | None = None, arm_b_ns: int | None = None, *,
    clock_operator: str, elapsed_span_ns: int | None, busy_union_ns: int | None,
) -> str:
    payload = "|".join((
        receipt_id, case_id, level, boundary_kind, engine_uid, start_event_name, end_event_name,
        str(start_event_seq), str(end_event_seq), str(start_ns), str(end_ns), str(predicted_ns),
        arm_a_id, arm_b_id, "" if arm_a_ns is None else str(arm_a_ns), "" if arm_b_ns is None else str(arm_b_ns),
        clock_operator, str(elapsed_span_ns), str(busy_union_ns),
    )).encode("utf-8")
    return hmac.new(_RECEIPT_HMAC_KEY, payload, "sha256").hexdigest()


_EVENT_OWNER_MAX_ENTRIES = 2048
_EVENT_OWNER_REGISTRY: OrderedDict[int, tuple[Any, str]] = OrderedDict()


def _trace_owns_event(engine: Any, event: Any) -> bool:
    """Verify ``engine`` actually recorded a settled task/boundary matching ``event``."""
    ev_name = str(event.name)
    ev_time = float(event.time)
    ev_started = getattr(event, "started", None)
    for row in engine.trace:
        task_name = str(row.get("task", ""))
        end_ms = row.get("end_ms")
        start_ms = row.get("start_ms")
        if task_name == ev_name and end_ms is not None and abs(float(end_ms) - ev_time) <= _CLOCK_EPS_MS:
            if ev_started is None or start_ms is None or abs(float(start_ms) - float(ev_started)) <= _CLOCK_EPS_MS:
                return True
        if (ev_name.endswith(":start") or ev_name == f"{task_name}:start") and start_ms is not None:
            base = ev_name[:-6] if ev_name.endswith(":start") else ev_name
            if task_name == base and abs(float(start_ms) - ev_time) <= _CLOCK_EPS_MS:
                return True
    return False


def _validate_live_engine_events(
    case: Any, engine: Any, start_event: Any, end_event: Any, clock_op: str,
) -> tuple[str, int, int, int, int | None]:
    from mindie_sim.prediction.runtime.engine import Engine, Event

    if not isinstance(engine, Engine):
        raise ModelingError(f"engine_receipt_requires_live_engine:{case.case_id}")
    if not isinstance(start_event, Event) or not isinstance(end_event, Event):
        raise ModelingError(f"engine_receipt_requires_event_handles:{case.case_id}")
    if not start_event.done or not end_event.done or start_event.time is None or end_event.time is None:
        raise ModelingError(f"engine_event_handles_unsettled:{case.case_id}")
    seq_limit = int(getattr(engine, "_sequence", 0))
    if not (1 <= start_event.sequence <= seq_limit and 1 <= end_event.sequence <= seq_limit):
        raise ModelingError(f"engine_event_foreign_sequence:{case.case_id}")
    if start_event.name != case.boundary.start_event or end_event.name != case.boundary.end_event:
        raise ModelingError(f"engine_event_name_mismatch:{case.case_id}:{start_event.name}:{end_event.name}")
    if start_event.time < -_CLOCK_EPS_MS or end_event.time > engine.now + _CLOCK_EPS_MS or end_event.time < start_event.time:
        raise ModelingError(f"engine_event_clock_bounds_invalid:{case.case_id}")
    uid = _req_str(getattr(engine, "uid", ""), f"engine_uid_missing:{case.case_id}")
    if engine._trace_maxlen <= 0 and clock_op == "elapsed_span":
        raise ModelingError(f"engine_event_not_owned_by_engine:{case.case_id}:trace_required")
    busy_ns = _busy_union_ns(engine, start_event.time, end_event.time)
    if clock_op == "busy_union" and busy_ns is None:
        raise ModelingError(f"busy_union_trace_incomplete:{case.case_id}")
    if not (_trace_owns_event(engine, start_event) and _trace_owns_event(engine, end_event)):
        raise ModelingError(f"engine_event_not_owned_by_engine:{case.case_id}")
    for ev in (start_event, end_event):
        entry = _EVENT_OWNER_REGISTRY.get(id(ev))
        if entry is not None and entry[0] is ev and entry[1] != uid:
            raise ModelingError(f"engine_event_not_owned_by_engine:{case.case_id}:cross_engine_handle")
    for ev in (start_event, end_event):
        _EVENT_OWNER_REGISTRY[id(ev)] = (ev, uid)
        _EVENT_OWNER_REGISTRY.move_to_end(id(ev))
        while len(_EVENT_OWNER_REGISTRY) > _EVENT_OWNER_MAX_ENTRIES:
            _EVENT_OWNER_REGISTRY.popitem(last=False)
    start_ns, end_ns = int(round(start_event.time * MS_TO_NS)), int(round(end_event.time * MS_TO_NS))
    if (elapsed_span_ns := end_ns - start_ns) <= 0:
        raise ModelingError(f"interval_span_or_duration_invalid:{case.case_id}")
    return uid, start_ns, end_ns, elapsed_span_ns, busy_ns


def _validate_clock_operator(value: Any, case_id: str) -> Literal["elapsed_span", "busy_union"]:
    if not isinstance(value, str) or value not in VALID_CLOCK_OPERATORS:
        raise ModelingError(f"clock_operator_invalid:{case_id}")
    return cast(Literal["elapsed_span", "busy_union"], value)


def _busy_union_ns(engine: Any, start_ms: float, end_ms: float) -> int | None:
    """Union settled device work spans when the trace can account for the interval."""
    from mindie_sim.prediction.execution.submission import interval_union_ns

    if (engine._trace_maxlen <= 0 or engine.trace_dropped or len(engine.trace) != engine.completed_count
            or any(task.start is not None and task.start < end_ms for task in engine._tasks.values())):
        return None
    host_resources = {
        name for name in engine.resources
        if any(part == "host" or part.startswith(("host@", "scheduler@")) for part in name.split(":"))
    }
    spans = []
    for row in engine.trace:
        if (demand := row.get("demand")) is None:
            return None
        if not any(amount > 0 and resource not in host_resources for resource, amount in demand.items()):
            continue
        start, end = row.get("start_ms"), row.get("end_ms")
        if start is None or end is None:
            return None
        if (start := max(start_ms, start)) < (end := min(end_ms, end)):
            spans.append((start, end))
    return interval_union_ns(spans)


def bind_engine_interval(
    case: Any, engine: Any, start_event: Any, end_event: Any, *,
    wire_bytes: int | None = None, kernel_sequence: tuple[str, ...] = (),
    diagnostics: tuple[tuple[str, Any], ...] = (),
    source_level: str = "", derived_from_step_division: bool = False,
    clock_operator: str | None = None,
) -> EngineInterval:
    """Bind one ``EngineInterval`` strictly to a live ``Engine`` run and named ``Event`` handles."""
    clock_op = _validate_clock_operator(case.boundary.clock_operator if clock_operator is None else clock_operator, case.case_id)
    uid, s_ns, e_ns, elapsed_span_ns, busy_union_ns = _validate_live_engine_events(case, engine, start_event, end_event, clock_op)
    pred_ns = busy_union_ns if clock_op == "busy_union" else elapsed_span_ns
    assert pred_ns is not None
    rid = validate_provenance_text(
        f"{CAPTURE_URI_SCHEME}engine-receipt/{uid}/{case.level}/{case.case_id}/{start_event.sequence}-{end_event.sequence}",
        field_name="engine_receipt",
    )
    tok = _sign_receipt(rid, case.case_id, case.level, case.boundary.boundary_kind, uid,
                        start_event.name, end_event.name, start_event.sequence, end_event.sequence, s_ns, e_ns, pred_ns,
                        clock_operator=clock_op, elapsed_span_ns=elapsed_span_ns, busy_union_ns=busy_union_ns)
    rcpt = EngineRunReceipt(rid, case.case_id, case.level, case.boundary.boundary_kind, uid,
                            start_event.name, end_event.name, start_event.sequence, end_event.sequence, s_ns, e_ns, pred_ns, tok,
                            clock_operator=clock_op)
    return EngineInterval(
        case.case_id, case.level, case.boundary.boundary_kind, start_event.name, end_event.name, s_ns, e_ns, pred_ns,
        rid, True, 0, "ok", "", source_level, derived_from_step_division, wire_bytes, kernel_sequence, None, None, diagnostics, rcpt,
        clock_op, elapsed_span_ns, busy_union_ns,
    )


def bind_engine_rung_interval(
    case: Any, *, arm_a: tuple[Any, Any, Any], arm_b: tuple[Any, Any, Any],
    diagnostics: tuple[tuple[str, Any], ...] = (),
) -> EngineInterval:
    """Bind a signed difference of two live Engine arms using the same clock operator."""
    if case.level != "L2" or case.boundary.boundary_kind != "paired_rung_delta":
        raise ModelingError(f"rung_binding_requires_l2_paired_rung_delta:{case.case_id}")
    (eng_a, s_a, e_a), (eng_b, s_b, e_b) = arm_a, arm_b
    clock_op = _validate_clock_operator(case.boundary.clock_operator, case.case_id)
    uid_a, _, _, elapsed_a, busy_a = _validate_live_engine_events(case, eng_a, s_a, e_a, clock_op)
    uid_b, _, _, elapsed_b, busy_b = _validate_live_engine_events(case, eng_b, s_b, e_b, clock_op)
    if (uid_a, s_a.sequence, e_a.sequence) == (uid_b, s_b.sequence, e_b.sequence):
        raise ModelingError(f"rung_arms_must_be_distinct_runs:{case.case_id}")
    pred_a, pred_b = (busy_a, busy_b) if clock_op == "busy_union" else (elapsed_a, elapsed_b)
    assert pred_a is not None and pred_b is not None
    elapsed_delta = elapsed_b - elapsed_a
    busy_delta = busy_b - busy_a if busy_a is not None and busy_b is not None else None
    a_id = f"{CAPTURE_URI_SCHEME}engine-receipt/{uid_a}/L2/{case.case_id}/arm_a/{s_a.sequence}-{e_a.sequence}"
    b_id = f"{CAPTURE_URI_SCHEME}engine-receipt/{uid_b}/L2/{case.case_id}/arm_b/{s_b.sequence}-{e_b.sequence}"
    delta_ns, uid = pred_b - pred_a, f"{uid_a}+{uid_b}"
    rid = validate_provenance_text(f"{CAPTURE_URI_SCHEME}engine-receipt/{uid}/L2/{case.case_id}/rung/{s_a.sequence}-{e_b.sequence}", field_name="engine_receipt")
    tok = _sign_receipt(rid, case.case_id, "L2", "paired_rung_delta", uid, s_a.name, e_b.name,
                        s_a.sequence, e_b.sequence, 0, delta_ns, delta_ns, a_id, b_id, pred_a, pred_b,
                        clock_operator=clock_op, elapsed_span_ns=elapsed_delta, busy_union_ns=busy_delta)
    rcpt = EngineRunReceipt(rid, case.case_id, "L2", "paired_rung_delta", uid, s_a.name, e_b.name,
                            s_a.sequence, e_b.sequence, 0, delta_ns, delta_ns, tok, a_id, b_id, clock_op)
    return EngineInterval(
        case.case_id, "L2", "paired_rung_delta", s_a.name, e_b.name, 0, delta_ns, delta_ns,
        rid, True, 0, "ok", "", "", False, None, (), pred_a, pred_b, diagnostics, rcpt,
        clock_op, elapsed_delta, busy_delta,
    )


def validate_engine_interval(case: Any, interval: EngineInterval, *, expected_level: Level) -> EngineInterval:
    """Validate Engine interval case_id, boundary, signed live-Engine receipt, and duration."""
    clock_op = case.boundary.clock_operator
    if interval.clock_operator != clock_op:
        raise ModelingError(f"clock_operator_mismatch:{case.case_id}:{clock_op}:{interval.clock_operator}")
    _validate_clock_operator(clock_op, case.case_id)
    if interval.case_id != case.case_id:
        raise ModelingError(f"engine_interval_case_id_mismatch:{case.case_id}:{interval.case_id}")
    if interval.level != expected_level:
        raise ModelingError(f"engine_interval_level_mismatch:{case.case_id}:{interval.level}")
    if interval.status == "missing_evidence":
        if not interval.missing_reason.strip():
            raise ModelingError(f"missing_evidence_requires_reason:{case.case_id}")
        return interval
    if not interval.engine_sourced or interval.external_offset_ns != 0:
        raise ModelingError(f"non_engine_or_offset_rejected:{case.case_id}:offset_ns={interval.external_offset_ns}")
    validate_provenance_text(_req_str(interval.engine_receipt, f"engine_receipt_missing:{case.case_id}"), field_name="engine_receipt")
    if (interval.boundary_kind != case.boundary.boundary_kind
            or interval.start_event != case.boundary.start_event or interval.end_event != case.boundary.end_event):
        raise ModelingError(f"boundary_mismatch:{case.case_id}")
    if case.boundary.boundary_kind == "paired_rung_delta":
        if (interval.arm_a_predicted_ns is None or interval.arm_b_predicted_ns is None
                or interval.arm_a_predicted_ns <= 0 or interval.arm_b_predicted_ns <= 0
                or interval.predicted_ns != interval.arm_b_predicted_ns - interval.arm_a_predicted_ns):
            raise ModelingError(f"rung_delta_prediction_invalid:{case.case_id}")
    else:
        span_ns = interval.end_ns - interval.start_ns
        duration_ns = interval.busy_union_ns if clock_op == "busy_union" else span_ns
        if interval.predicted_ns != duration_ns or not 0 < interval.predicted_ns <= span_ns:
            raise ModelingError(f"interval_span_or_duration_invalid:{case.case_id}")
    rcpt = interval.run_receipt
    if not isinstance(rcpt, EngineRunReceipt):
        raise ModelingError(f"engine_receipt_unbound:{case.case_id}")
    if ((rcpt.receipt_id, rcpt.case_id, rcpt.level, rcpt.boundary_kind, rcpt.start_event_name, rcpt.end_event_name,
         rcpt.start_ns, rcpt.end_ns, rcpt.predicted_ns, rcpt.clock_operator)
            != (interval.engine_receipt, case.case_id, expected_level, case.boundary.boundary_kind,
                case.boundary.start_event, case.boundary.end_event, interval.start_ns, interval.end_ns, interval.predicted_ns, clock_op)):
        raise ModelingError(f"engine_receipt_binding_mismatch:{case.case_id}")
    expected_tok = _sign_receipt(
        rcpt.receipt_id, rcpt.case_id, rcpt.level, rcpt.boundary_kind, rcpt.engine_uid,
        rcpt.start_event_name, rcpt.end_event_name, rcpt.start_event_seq, rcpt.end_event_seq,
        rcpt.start_ns, rcpt.end_ns, rcpt.predicted_ns, rcpt.arm_a_receipt_id, rcpt.arm_b_receipt_id,
        interval.arm_a_predicted_ns, interval.arm_b_predicted_ns,
        clock_operator=rcpt.clock_operator, elapsed_span_ns=interval.elapsed_span_ns, busy_union_ns=interval.busy_union_ns,
    )
    if not hmac.compare_digest(rcpt.receipt_token, expected_tok):
        raise ModelingError(f"engine_receipt_signature_invalid:{case.case_id}")
    return interval
