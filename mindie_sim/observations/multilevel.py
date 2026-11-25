"""Multilevel streaming capture reader, drilldown and verification gates.

[Role] Stream multilevel observation JSONL records, run drilldown diagnostics,
  and evaluate multilevel accuracy gates.
[Invariants] Bounded streaming I/O; strict gate failure propagation.
[Timeouts & Locks] Local streaming file reads only; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mindie_sim.schema.common import ModelingError

DRILLDOWN_LEVELS = ("L1_in_L3", "L2_in_L3")


@dataclass(slots=True)
class _CancellationAudit:
    component_count: int = 0
    absolute_error_ns: int = 0
    signed_error_ns: int = 0

    def add(self, error_ns: int) -> None:
        self.component_count += 1
        self.absolute_error_ns += abs(error_ns)
        self.signed_error_ns += error_ns

    def to_dict(self) -> dict[str, int | float]:
        cancellation_ns = self.absolute_error_ns - abs(self.signed_error_ns)
        return {
            "component_count": self.component_count,
            "cancellation_ns": cancellation_ns,
            "cancellation_ratio": cancellation_ns / max(1.0, self.absolute_error_ns),
        }


def score_vertical_drilldown(
    step_predicted_ns: int, step_measured_ns: int, components: Sequence[tuple[str, str, int, int]],
    exceeds_tolerance: Callable[[int, int], bool],
) -> dict[str, Any]:
    audits = {level: _CancellationAudit() for level in DRILLDOWN_LEVELS}
    failed_components = []
    seen = set()
    duplicate = False
    for level, name, predicted_ns, measured_ns in components:
        if level not in audits:
            raise ModelingError(f"invalid_drilldown_level:{level}")
        duplicate |= (level, name) in seen
        seen.add((level, name))
        audits[level].add(predicted_ns - measured_ns)
        if measured_ns <= 0 or predicted_ns <= 0 or exceeds_tolerance(predicted_ns, measured_ns):
            failed_components.append({
                "level_tag": level, "name": name, "predicted_ns": predicted_ns, "measured_ns": measured_ns,
            })
    step_passed = min(step_predicted_ns, step_measured_ns) > 0 and not exceeds_tolerance(step_predicted_ns, step_measured_ns)
    missing_levels = [level for level, audit in audits.items() if not audit.component_count]
    reason = ("empty_component_census" if not components else
              "duplicate_component" if duplicate else
              "component_levels_missing" if missing_levels else
              "component_pointwise_failure" if failed_components else
              "step_pointwise_failure" if not step_passed else "")
    return {
        "passed": not reason,
        "failure_reason": reason,
        "missing_levels": missing_levels,
        "step_passed": step_passed,
        "failed_components": failed_components,
        "levels": {level: audit.to_dict() for level, audit in audits.items()},
    }


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
from .captures import (
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


def _iter_json_object_array(
    events: Iterator[tuple[str, Any]], path: Path,
) -> Iterator[dict[str, Any]]:
    try:
        while (elem := next(events))[0] != "end_array":
            val = _read_value(events, elem)
            if isinstance(val, Mapping):
                yield dict(val)
    except StopIteration as exc:
        raise ModelingError(f"json_stream_truncated:{path.name}") from exc


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
                yield from _iter_json_object_array(events, path)
            elif first[0] == "start_map":
                singleton: dict[str, Any] = {}
                found_array = False
                while (entry := next(events))[0] != "end_map":
                    key, val_ev = str(entry[1]), next(events)
                    if not found_array and key in wanted and val_ev[0] == "start_array":
                        found_array = True
                        yield from _iter_json_object_array(events, path)
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
    device_ranks: tuple[int, ...] = ()


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
    device_ranks: tuple[int, ...] = (),
) -> str:
    payload = "|".join((
        receipt_id, case_id, level, boundary_kind, engine_uid, start_event_name, end_event_name,
        str(start_event_seq), str(end_event_seq), str(start_ns), str(end_ns), str(predicted_ns),
        arm_a_id, arm_b_id, "" if arm_a_ns is None else str(arm_a_ns), "" if arm_b_ns is None else str(arm_b_ns),
        clock_operator, str(elapsed_span_ns), str(busy_union_ns), ",".join(str(r) for r in device_ranks),
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
    for ev in (start_event, end_event):
        if ev.runtime_uid != uid:
            raise ModelingError(f"engine_event_not_owned_by_engine:{case.case_id}:cross_engine_handle")
    if engine._trace_maxlen <= 0 and clock_op == "elapsed_span":
        raise ModelingError(f"engine_event_not_owned_by_engine:{case.case_id}:trace_required")
    busy_ns = _busy_union_ns(
        engine, start_event.time, end_event.time,
        device_ranks=case.boundary.device_ranks,
        start_event_name=start_event.name, end_event_name=end_event.name,
    )
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
    if busy_ns is not None:
        busy_ns = min(busy_ns, elapsed_span_ns)
    return uid, start_ns, end_ns, elapsed_span_ns, busy_ns


def _validate_clock_operator(value: Any, case_id: str) -> Literal["elapsed_span", "busy_union"]:
    if not isinstance(value, str) or value not in VALID_CLOCK_OPERATORS:
        raise ModelingError(f"clock_operator_invalid:{case_id}")
    return cast(Literal["elapsed_span", "busy_union"], value)


_HOST_RESOURCE_EXACT = frozenset({"host", "host_memory", "host_vector", "scheduler"})
_HOST_RESOURCE_PREFIXES = ("host@", "host_memory@", "host_vector@", "scheduler@")
_SHARED_FABRIC_TAGS = frozenset({"link", "port", "pair", "host_rc"})
_RESOURCE_DIE_RE = re.compile(r"(?:@r\d+\.p\d+\.d|@d|@die|@rank|:die|:port)(\d+)(?::|$)")
_RESOURCE_TAIL_DIE_RE = re.compile(r":d(\d+)(?::|$)")
_RESOURCE_PAIR_RE = re.compile(r":(?:pair)?(\d+)-?>(\d+)(?::|$)")
_RESOURCE_STAGE_RE = re.compile(r"@r(\d+):s(\d+)(?::|$)")
_RESOURCE_REPLICA_RE = re.compile(r"@r(\d+)(?::|$)")


def _is_host_resource(name: str) -> bool:
    return any(part in _HOST_RESOURCE_EXACT or part.startswith(_HOST_RESOURCE_PREFIXES) for part in name.split(":"))


def _is_shared_fabric_resource(name: str) -> bool:
    return any(part in _SHARED_FABRIC_TAGS for part in name.split(":"))


def _replica_stage_maps(
    resources: Iterable[str], host_resources: set[str], participating: tuple[int, ...],
) -> tuple[dict[tuple[int, int], tuple[int, ...]], dict[tuple[int, str], tuple[int, ...]], bool]:
    explicit_ranks: set[int] = set()
    stage_slots: set[tuple[int, int]] = set()
    replica_only: set[int] = set()
    for name in resources:
        if name in host_resources:
            continue
        if (die := _RESOURCE_DIE_RE.search(name)) is not None:
            explicit_ranks.add(int(die.group(1)))
        elif (pair := _RESOURCE_PAIR_RE.search(name)) is not None:
            explicit_ranks.update((int(pair.group(1)), int(pair.group(2))))
        elif (td := _RESOURCE_TAIL_DIE_RE.search(name)) is not None:
            explicit_ranks.add(int(td.group(1)))
        elif (st := _RESOURCE_STAGE_RE.search(name)) is not None:
            stage_slots.add((int(st.group(1)), int(st.group(2))))
        elif (rp := _RESOURCE_REPLICA_RE.search(name)) is not None:
            replica_only.add(int(rp.group(1)))
    has_any_rank_tag = bool(explicit_ranks or stage_slots or replica_only)
    all_reps = {rep for rep, _ in stage_slots} | replica_only
    if not all_reps:
        return {}, {}, has_any_rank_tag
    n_stages = (max(stage for _, stage in stage_slots) + 1) if stage_slots else 1
    n_slots = (max(all_reps) + 1) * n_stages
    world = max(max((*participating, *explicit_ranks)) + 1, n_slots)
    width = world // n_slots if world % n_slots == 0 else 1
    stage_map = {
        (rep, stage): tuple(range((rep * n_stages + stage) * width, (rep * n_stages + stage + 1) * width))
        for rep in range(max(all_reps) + 1)
        for stage in range(n_stages)
    }
    replica_map: dict[tuple[int, str], tuple[int, ...]] = {}
    for rep in range(max(all_reps) + 1):
        replica_map[(rep, "h2d")] = stage_map[(rep, 0)]
        replica_map[(rep, "d2h")] = stage_map[(rep, n_stages - 1)]
        replica_map[(rep, "")] = tuple(r for s in range(n_stages) for r in stage_map[(rep, s)])
    return stage_map, replica_map, has_any_rank_tag


def _demand_ranks(
    demand: Mapping[str, float], host_resources: set[str],
    stage_map: Mapping[tuple[int, int], tuple[int, ...]],
    replica_map: Mapping[tuple[int, str], tuple[int, ...]],
    has_any_rank_tag: bool, participating_set: set[int], *, tagged_only: bool = False,
) -> set[int]:
    device_res = [r for r, amount in demand.items() if amount > 0 and r not in host_resources]
    if not device_res:
        return set()
    specific: set[int] = set()
    replica_ranks: set[int] = set()
    has_specific = has_replica = False
    for resource in device_res:
        if (die := _RESOURCE_DIE_RE.search(resource)) is not None:
            has_specific = True
            specific.add(int(die.group(1)))
        elif (pair := _RESOURCE_PAIR_RE.search(resource)) is not None:
            has_specific = True
            specific.update((int(pair.group(1)), int(pair.group(2))))
        elif (td := _RESOURCE_TAIL_DIE_RE.search(resource)) is not None:
            has_specific = True
            specific.add(int(td.group(1)))
        elif (st := _RESOURCE_STAGE_RE.search(resource)) is not None:
            has_specific = True
            specific.update(stage_map.get((int(st.group(1)), int(st.group(2))), ()))
        elif (rp := _RESOURCE_REPLICA_RE.search(resource)) is not None:
            has_replica = True
            prefix = resource.split("@", 1)[0].rsplit(":", 1)[-1]
            role = prefix if prefix in {"h2d", "d2h"} else ""
            replica_ranks.update(replica_map.get((int(rp.group(1)), role), ()))
    if has_specific:
        return specific & participating_set
    if has_replica:
        return replica_ranks & participating_set
    if tagged_only or (has_any_rank_tag and all(_is_shared_fabric_resource(r) for r in device_res)):
        return set()
    return set(participating_set)


def _boundary_time(row: Mapping[str, Any], event_name: str) -> float | None:
    task_name = str(row.get("task", ""))
    if event_name.endswith(":start") and task_name == event_name.removesuffix(":start"):
        val = row.get("start_ms")
        return None if val is None else float(val)
    if task_name == event_name:
        val = row.get("end_ms")
        return None if val is None else float(val)
    return None


def _rank_windows(
    trace: Iterable[Mapping[str, Any]], start_ms: float, end_ms: float, participating: tuple[int, ...],
    start_event_name: str | None, end_event_name: str | None,
    resolve_tagged: Callable[[Mapping[str, float]], set[int]],
) -> dict[int, tuple[float, float]]:
    windows = {rank: (start_ms, end_ms) for rank in participating}
    if not (start_event_name and end_event_name):
        return windows
    starts: dict[int, list[float]] = {rank: [] for rank in participating}
    ends: dict[int, list[float]] = {rank: [] for rank in participating}
    for row in trace:
        demand = row.get("demand")
        if not demand or not (ranks := resolve_tagged(demand)):
            continue
        if (t_s := _boundary_time(row, start_event_name)) is not None and start_ms - _CLOCK_EPS_MS <= t_s <= end_ms + _CLOCK_EPS_MS:
            for rank in ranks:
                starts[rank].append(t_s)
        if (t_e := _boundary_time(row, end_event_name)) is not None and start_ms - _CLOCK_EPS_MS <= t_e <= end_ms + _CLOCK_EPS_MS:
            for rank in ranks:
                ends[rank].append(t_e)
    for rank in participating:
        if len(starts[rank]) == 1 and len(ends[rank]) == 1 and starts[rank][0] <= ends[rank][0]:
            windows[rank] = (starts[rank][0], ends[rank][0])
    return windows


def _union_spans_ns(spans: list[tuple[float, float]]) -> int:
    if not spans:
        return 0
    ordered = sorted(spans)
    merged: list[float] = []
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            if end > cur_end:
                cur_end = end
        else:
            merged.append(cur_end - cur_start)
            cur_start, cur_end = start, end
    merged.append(cur_end - cur_start)
    return int(round(math.fsum(merged) * MS_TO_NS))


def _busy_union_ns(
    engine: Any, start_ms: float, end_ms: float,
    device_ranks: Sequence[int] | None = None, *,
    start_event_name: str | None = None, end_event_name: str | None = None,
) -> int | None:
    """Union settled device work spans per participating rank and take the max."""
    if engine._trace_maxlen <= 0 or engine.trace_dropped or len(engine.trace) != engine.completed_count:
        return None
    participating = tuple(dict.fromkeys(device_ranks)) if device_ranks is not None else (0,)
    if not participating:
        return None
    participating_set = set(participating)
    host_resources = {name for name in engine.resources if _is_host_resource(name)}
    stage_map, replica_map, has_any_rank_tag = _replica_stage_maps(engine.resources, host_resources, participating)
    resolve = partial(
        _demand_ranks, host_resources=host_resources, stage_map=stage_map,
        replica_map=replica_map, has_any_rank_tag=has_any_rank_tag, participating_set=participating_set,
    )
    windows = _rank_windows(
        engine.trace, start_ms, end_ms, participating, start_event_name, end_event_name,
        partial(resolve, tagged_only=True),
    )
    for task in engine._tasks.values():
        if task.start is not None and any(task.start < windows[rank][1] for rank in resolve(task.demand)):
            return None
    spans_by_rank: dict[int, list[tuple[float, float]]] = {rank: [] for rank in participating}
    for row in engine.trace:
        if (demand := row.get("demand")) is None:
            return None
        if not (targets := resolve(demand)):
            continue
        start, end = row.get("start_ms"), row.get("end_ms")
        if start is None or end is None:
            return None
        for rank in targets:
            left, right = windows[rank]
            if (clipped_s := max(left, start)) < (clipped_e := min(right, end)):
                spans_by_rank[rank].append((clipped_s, clipped_e))
    return max(_union_spans_ns(spans) for spans in spans_by_rank.values())


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
    ranks = tuple(case.boundary.device_ranks)
    tok = _sign_receipt(rid, case.case_id, case.level, case.boundary.boundary_kind, uid,
                        start_event.name, end_event.name, start_event.sequence, end_event.sequence, s_ns, e_ns, pred_ns,
                        clock_operator=clock_op, elapsed_span_ns=elapsed_span_ns, busy_union_ns=busy_union_ns,
                        device_ranks=ranks)
    rcpt = EngineRunReceipt(rid, case.case_id, case.level, case.boundary.boundary_kind, uid,
                            start_event.name, end_event.name, start_event.sequence, end_event.sequence, s_ns, e_ns, pred_ns, tok,
                            clock_operator=clock_op, device_ranks=ranks)
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
    ranks = tuple(case.boundary.device_ranks)
    tok = _sign_receipt(rid, case.case_id, "L2", "paired_rung_delta", uid, s_a.name, e_b.name,
                        s_a.sequence, e_b.sequence, 0, delta_ns, delta_ns, a_id, b_id, pred_a, pred_b,
                        clock_operator=clock_op, elapsed_span_ns=elapsed_delta, busy_union_ns=busy_delta,
                        device_ranks=ranks)
    rcpt = EngineRunReceipt(rid, case.case_id, "L2", "paired_rung_delta", uid, s_a.name, e_b.name,
                            s_a.sequence, e_b.sequence, 0, delta_ns, delta_ns, tok, a_id, b_id, clock_op, ranks)
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
    expected_ranks = tuple(case.boundary.device_ranks)
    if rcpt.device_ranks and rcpt.device_ranks != expected_ranks:
        raise ModelingError(f"engine_receipt_binding_mismatch:{case.case_id}")
    expected_tok = _sign_receipt(
        rcpt.receipt_id, rcpt.case_id, rcpt.level, rcpt.boundary_kind, rcpt.engine_uid,
        rcpt.start_event_name, rcpt.end_event_name, rcpt.start_event_seq, rcpt.end_event_seq,
        rcpt.start_ns, rcpt.end_ns, rcpt.predicted_ns, rcpt.arm_a_receipt_id, rcpt.arm_b_receipt_id,
        interval.arm_a_predicted_ns, interval.arm_b_predicted_ns,
        clock_operator=rcpt.clock_operator, elapsed_span_ns=interval.elapsed_span_ns, busy_union_ns=interval.busy_union_ns,
        device_ranks=expected_ranks,
    )
    if not hmac.compare_digest(rcpt.receipt_token, expected_tok):
        raise ModelingError(f"engine_receipt_signature_invalid:{case.case_id}")
    return interval


from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
import math
from pathlib import Path, PurePosixPath
import statistics
from types import MappingProxyType
from typing import Any, Literal, Protocol, overload, runtime_checkable

from mindie_sim.observations.captures import L1_COMPOSITE_BOUNDARY_KINDS, L3_BOUNDARY_KINDS, CaptureVerificationReceipt
from mindie_sim.observations.captures import verify_capture_file_consistency as _verify_capture
from mindie_sim.schema.common import ModelingError

try:
    import ijson
except ImportError:  # pragma: no cover - stdlib fallback when ijson is absent
    ijson = None

DEFAULT_THRESHOLD_PCT = 10.0
DEFAULT_ATOL_NS = 2000
TAIL_QUANTILE_P90 = 0.9
NEAR_ZERO_DELTA_MIN_RATIO = 0.05
DEFAULT_NEAR_ZERO_DELTA_NS = 500
PERCENT_SCALE = 100.0
REQUIRED_LEVELS: tuple[Literal["L1", "L2", "L3"], ...] = ("L1", "L2", "L3")
VALID_LEVELS = frozenset(REQUIRED_LEVELS)
VALID_SPLITS = frozenset({"train", "heldout"})
_CALLER_PREDICTION_KEYS = frozenset({"predicted_ns", "predicted_ms", "predicted_us", "predicted_duration", "predicted_s"})
MANIFEST_HEADER_KEYS = frozenset(
    {"manifest_id", "schema_version", "lineage_edges", "source_uri", "source_sha256", "strict_three_level_blind"}
)
L1_BOUNDARY_KINDS = L1_COMPOSITE_BOUNDARY_KINDS | {"device_kernel"}
L2_BOUNDARY_KINDS = frozenset({"fx_region", "dispatch_gmm_combine", "periodic_block",
                               "paired_rung_delta", "subgraph_block", "npu_event_burst"})
FORBIDDEN_L2_BOUNDARY_KINDS = L3_BOUNDARY_KINDS | {"layer_step_quotient", "step_divided_by_layers", "l3_clone"}

SplitName = Literal["train", "heldout"]
ResidualStatus = Literal["ok", "missing_evidence", "invalid_evidence", "near_zero_delta", "debug_only", "counterfactual_excluded"]
CohortStatus = Literal["pass", "fail", "insufficient_evidence"]


@dataclass(frozen=True, slots=True)
class CaptureBoundary:
    raw_sha256: str
    acquisition_uid: str
    run_uid: str
    parent_digests: tuple[str, ...]
    event_ids: tuple[str, ...]
    device_ranks: tuple[int, ...]
    clock_domain: str
    start_event: str
    end_event: str
    boundary_kind: str
    unit: str
    capture_uri: str = ""
    source_ref: tuple[tuple[str, str], ...] = ()
    session_id: str = ""
    clock_operator: Literal["elapsed_span", "busy_union"] = "elapsed_span"


@dataclass(frozen=True, slots=True)
class CohortKey:
    level: Level
    cohort_name: str
    sku: str
    phase: str
    mode: str
    shape_regime: str
    parallel_topology: str
    def as_id(self) -> str:
        return "|".join((self.level, self.cohort_name, self.sku, self.phase, self.mode, self.shape_regime, self.parallel_topology))
    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GateCase:
    case_id: str
    level: Level
    split: SplitName
    boundary: CaptureBoundary
    input_digest: str
    stack_digest: str
    implementation: str
    cohort: CohortKey
    measured_ns: int | None
    debug_cross_level: bool = False
    is_counterfactual: bool = False
    has_matched_capture: bool = True
    rung_arm_a_measured_ns: int | None = None
    rung_arm_b_measured_ns: int | None = None
    block_capture_present: bool = True
    regime_dims: tuple[tuple[str, str], ...] = ()
    payload: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class Residual:
    case_id: str
    level: str
    cohort: CohortKey
    acquisition_root: str
    measured_ns: int
    predicted_ns: int
    engine_receipt: str
    signed_error: float
    absolute_percentage_error: float
    status: ResidualStatus = "ok"
    excluded_reason: str = ""
    boundary_kind: str = ""
    signed_delta_ns: int | None = None
    arm_a_signed_error: float | None = None
    arm_b_signed_error: float | None = None
    regime_dims: tuple[tuple[str, str], ...] = ()
    clock_operator: str = "elapsed_span"
    elapsed_span_ns: int | None = None
    busy_union_ns: int | None = None
    capture_scope: tuple[str, ...] = ()
    capture_events: frozenset[str] = frozenset()
    capture_ranks: frozenset[int] = frozenset()
    expected_components: tuple[str, ...] = ()
    capture_verification: CaptureVerificationReceipt | None = None
    split: SplitName = "heldout"
    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["signed_error"] = None if math.isnan(self.signed_error) else self.signed_error
        out["absolute_percentage_error"] = None if math.isnan(self.absolute_percentage_error) else self.absolute_percentage_error
        out["regime_dims"] = dict(self.regime_dims)
        out["capture_events"] = sorted(self.capture_events)
        out["capture_ranks"] = sorted(self.capture_ranks)
        if (verification := out["capture_verification"]) is not None:
            verification["selected_kernels"] = sorted(verification["selected_kernels"])
            if verification["window_kernels"] is not None:
                verification["window_kernels"] = sorted(verification["window_kernels"])
        return out


@dataclass(frozen=True, slots=True)
class FrozenFitParameter:
    parameter_id: str
    training_digests: tuple[str, ...]
    source_uri: str = ""


@dataclass(frozen=True, slots=True)
class FrozenFitManifest:
    fit_id: str
    schema_version: str
    parameters: tuple[FrozenFitParameter, ...]
    lineage_edges: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class Manifest:
    manifest_id: str
    schema_version: str
    cases: Sequence[GateCase]
    lineage_edges: tuple[tuple[str, str], ...] = ()
    source_uri: str = ""
    source_sha256: str = ""
    strict_three_level_blind: bool = True


@dataclass(frozen=True, slots=True)
class SplitIndex:
    case_roots: Mapping[str, str]
    component_splits: Mapping[str, str]
    train_components: frozenset[str]
    heldout_components: frozenset[str]
    heldout_roots_by_level: Mapping[str, frozenset[str]]
    debug_only_cases: frozenset[str]
    fit_ancestor_components: Mapping[str, frozenset[str]]
    capture_receipts: Mapping[str, CaptureVerificationReceipt] = field(default_factory=dict)
    def root_for(self, case_id: str) -> str:
        return self.case_roots[case_id]


@dataclass(frozen=True, slots=True)
class CohortSummary:
    cohort: CohortKey
    status: CohortStatus
    n_roots: int
    n_rows: int
    mape_pct: float | None
    medape_pct: float | None
    median_signed_pct: float | None
    p90_ape_pct: float | None
    max_ape_pct: float | None
    missing: int
    excluded_reasons: tuple[tuple[str, int], ...]
    threshold_pct: float = DEFAULT_THRESHOLD_PCT
    failed_metrics: tuple[str, ...] = ()
    tail_case_ids: tuple[str, ...] = ()
    acquisition_roots: tuple[str, ...] = ()
    evaluated_ok_case_ids: tuple[str, ...] = ()
    atol_ns: float = DEFAULT_ATOL_NS
    pointwise_failed_count: int = 0
    complete_drilldowns: int = 0
    drilldowns: tuple[tuple[str, dict[str, Any]], ...] = ()
    diagnostic_only: bool = False
    @property
    def passed(self) -> bool:
        return self.status == "pass"
    @property
    def tail_exceeded(self) -> bool:
        return self.pointwise_failed_count > 0


@runtime_checkable
class EvaluationRunner(Protocol):
    def run_l1(self, case: GateCase) -> EngineInterval: ...
    def run_l2(self, case: GateCase) -> EngineInterval: ...
    def run_l3(self, case: GateCase) -> EngineInterval: ...


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}
    def find(self, node: str) -> str:
        root = self._parent.setdefault(node, node)
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[node] != root:
            node, self._parent[node] = self._parent[node], root
        return root
    def union(self, left: str, right: str) -> None:
        rl, rr = self.find(left), self.find(right)
        if rl != rr:
            self._parent[rr if rl < rr else rl] = rl if rl < rr else rr


class _LazyManifestCases(Sequence[GateCase]):
    """Streaming disk-backed ``GateCase`` sequence that avoids buffering cases."""

    __slots__ = ("_path", "_is_jsonl", "_allow_invalid_durations", "_count")

    def __init__(self, path: Path, *, is_jsonl: bool, allow_invalid_durations: bool, count: int) -> None:
        self._path, self._is_jsonl = path, is_jsonl
        self._allow_invalid_durations, self._count = allow_invalid_durations, count

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[GateCase]:
        if self._is_jsonl:
            for row in stream_jsonl(self._path):
                if row.get("record_type", "case") not in {"manifest_header", "lineage_edge"}:
                    yield parse_gate_case(row, allow_invalid_durations=self._allow_invalid_durations)
            return
        for row in iter_manifest_json_cases(self._path, ijson_mod=ijson):
            yield parse_gate_case(row, allow_invalid_durations=self._allow_invalid_durations)

    @overload
    def __getitem__(self, index: int) -> GateCase: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[GateCase, ...]: ...

    def __getitem__(self, index: int | slice) -> GateCase | tuple[GateCase, ...]:
        if isinstance(index, slice):
            indices = range(*index.indices(self._count))
            wanted, collected, max_idx = set(indices), {}, max(indices, default=-1)
            for pos, item in enumerate(self):
                if pos in wanted:
                    collected[pos] = item
                if pos >= max_idx:
                    break
            return tuple(collected[i] for i in indices)
        target = index + self._count if index < 0 else index
        if target < 0 or target >= self._count:
            raise IndexError(index)
        for pos, item in enumerate(self):
            if pos == target:
                return item
        raise IndexError(index)


def _parse_ranks(raw: Any) -> tuple[int, ...]:
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise ModelingError("device_ranks_must_be_nonempty_sequence")
    ranks = [item for item in raw if not isinstance(item, bool) and isinstance(item, int) and item >= 0]
    if len(ranks) != len(raw) or len(set(ranks)) != len(ranks):
        raise ModelingError(f"device_rank_invalid:{raw!r}")
    return tuple(ranks)


def _parse_str_tuple(raw: Any, reason: str, *, normalize_digests: bool = False) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ModelingError(reason)
    items = [validate_provenance_text(_req_str(x, reason), field_name=reason) for x in raw]
    return tuple(normalize_lineage_token(x) if normalize_digests else x for x in items)


def _to_ns(value: Any, unit: str, reason: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ModelingError(reason)
    return int(round(float(value) * TIME_UNIT_SCALE_NS[unit]))


def _fstr(mapping: Mapping[str, Any], key: str, cid: str, *, check_prov: bool = False) -> str:
    val = _req_str(mapping.get(key), f"{key}_missing:{cid}")
    return validate_provenance_text(val, field_name=key) if check_prov else val


def _parse_lineage_edges(raw: Any) -> tuple[tuple[str, str], ...]:
    if not raw:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ModelingError("lineage_edges_invalid")
    edges = []
    for item in raw:
        pair = (item.get("parent"), item.get("child")) if isinstance(item, Mapping) else (
            (item[0], item[1]) if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2 else None
        )
        if pair is None:
            raise ModelingError("lineage_edge_entry_invalid")
        edges.append((
            normalize_lineage_token(_req_str(pair[0], "lineage_edge_parent_missing")),
            normalize_lineage_token(_req_str(pair[1], "lineage_edge_child_missing")),
        ))
    return tuple(edges)


def parse_gate_case(raw: Mapping[str, Any], *, allow_invalid_durations: bool = False) -> GateCase:
    """Validate and convert one raw manifest record into a ``GateCase``."""
    if not isinstance(raw, Mapping):
        raise ModelingError("gate_case_must_be_mapping")
    cid = validate_provenance_text(_req_str(raw.get("case_id"), "case_id_missing"), field_name="case_id")
    p_raw = raw.get("payload") or {}
    if not _CALLER_PREDICTION_KEYS.isdisjoint(raw) or (
        isinstance(p_raw, Mapping) and not _CALLER_PREDICTION_KEYS.isdisjoint(p_raw)
    ):
        raise ModelingError(f"caller_predicted_duration_forbidden:{cid}")
    level, split = _req_str(raw.get("level"), "level_missing"), _req_str(raw.get("split"), "split_missing")
    b_raw, c_raw = raw.get("boundary"), raw.get("cohort")
    if level not in VALID_LEVELS or split not in VALID_SPLITS:
        raise ModelingError(f"case_level_or_split_invalid:{cid}:{level}:{split}")
    if not isinstance(b_raw, Mapping) or not isinstance(c_raw, Mapping):
        raise ModelingError(f"boundary_or_cohort_missing:{cid}")
    unit = _fstr(b_raw, "unit", cid).lower()
    if unit not in TIME_UNIT_SCALE_NS:
        raise ModelingError(f"unit_unsupported:{unit}")
    sess = validate_provenance_text(_req_str(b_raw["session_id"], f"session_id_invalid:{cid}"), field_name="session_id") if b_raw.get("session_id") else ""
    boundary = CaptureBoundary(
        raw_sha256=_req_sha256(b_raw.get("raw_sha256"), f"raw_sha256_invalid:{cid}"),
        acquisition_uid=_fstr(b_raw, "acquisition_uid", cid, check_prov=True),
        run_uid=_fstr(b_raw, "run_uid", cid, check_prov=True),
        parent_digests=_parse_str_tuple(b_raw.get("parent_digests"), f"parent_digests_invalid:{cid}", normalize_digests=True),
        event_ids=_parse_str_tuple(b_raw.get("event_ids"), f"event_ids_invalid:{cid}"),
        device_ranks=_parse_ranks(b_raw.get("device_ranks")),
        clock_domain=_fstr(b_raw, "clock_domain", cid, check_prov=True),
        start_event=_fstr(b_raw, "start_event", cid), end_event=_fstr(b_raw, "end_event", cid),
        boundary_kind=_fstr(b_raw, "boundary_kind", cid), unit=unit,
        capture_uri=sanitize_capture_uri(str(b_raw["capture_uri"])) if b_raw.get("capture_uri") else "",
        source_ref=sanitize_source_ref(b_raw["source_ref"]) if b_raw.get("source_ref") is not None else (),
        session_id=sess,
        clock_operator=_validate_clock_operator(b_raw.get("clock_operator", "elapsed_span"), cid),
    )
    cohort = CohortKey(
        level=level, cohort_name=_fstr(c_raw, "cohort_name", cid), sku=_fstr(c_raw, "sku", cid),  # type: ignore[arg-type]
        phase=_fstr(c_raw, "phase", cid), mode=_fstr(c_raw, "mode", cid),
        shape_regime=_fstr(c_raw, "shape_regime", cid), parallel_topology=_fstr(c_raw, "parallel_topology", cid),
    )
    arm_a = _to_ns(raw["rung_arm_a_measured_ns"], "ns", f"rung_arm_a_invalid:{cid}") if raw.get("rung_arm_a_measured_ns") is not None else None
    arm_b = _to_ns(raw["rung_arm_b_measured_ns"], "ns", f"rung_arm_b_invalid:{cid}") if raw.get("rung_arm_b_measured_ns") is not None else None
    measured_ns = (
        _to_ns(raw["measured_ns"], "ns", f"measured_ns_invalid:{cid}") if raw.get("measured_ns") is not None
        else (_to_ns(raw["measured_duration"], unit, f"measured_duration_invalid:{cid}") if raw.get("measured_duration") is not None else None)
    )
    if not allow_invalid_durations:
        if boundary.boundary_kind == "paired_rung_delta":
            if arm_a is None or arm_b is None or arm_a <= 0 or arm_b <= 0:
                raise ModelingError(f"rung_arms_must_be_positive:{cid}")
            measured_ns = (arm_b - arm_a) if measured_ns is None else measured_ns
        elif measured_ns is None or measured_ns <= 0:
            raise ModelingError(f"measured_duration_must_be_positive:{cid}")
    r_raw = raw.get("regime_dims") or {}
    if not isinstance(r_raw, Mapping) or not isinstance(p_raw, Mapping):
        raise ModelingError(f"case_metadata_invalid:{cid}")
    return GateCase(
        case_id=cid, level=level, split=split, boundary=boundary,  # type: ignore[arg-type]
        input_digest=_fstr(raw, "input_digest", cid), stack_digest=_fstr(raw, "stack_digest", cid),
        implementation=_fstr(raw, "implementation", cid), cohort=cohort, measured_ns=measured_ns,
        debug_cross_level=bool(raw.get("debug_cross_level", False)),
        is_counterfactual=bool(raw.get("is_counterfactual", False)),
        has_matched_capture=bool(raw.get("has_matched_capture", True)),
        rung_arm_a_measured_ns=arm_a, rung_arm_b_measured_ns=arm_b,
        block_capture_present=bool(raw.get("block_capture_present", True)),
        regime_dims=tuple(sorted((str(k), validate_provenance_text(str(v), field_name=f"regime_dims.{k}")) for k, v in r_raw.items())),
        payload=tuple(sorted((str(k), validate_provenance_text(v, field_name=f"payload.{k}") if isinstance(v, str) else v) for k, v in p_raw.items())),
    )


def stream_manifest(
    path: str | Path, *, allow_invalid_durations: bool = False,
    max_whole_read_bytes: int = STREAM_WHOLE_READ_LIMIT_BYTES,
) -> Manifest:
    """Two-pass streaming loader for JSON/JSONL manifests without buffering cases."""
    del max_whole_read_bytes
    target, extra_edges = Path(path), []
    is_jsonl = target.suffix.lower() in {".jsonl", ".ndjson"}
    if is_jsonl:
        header, case_count = {}, 0
        for row in stream_jsonl(target):
            rtype = row.get("record_type", "case")
            if rtype == "manifest_header":
                header.update({k: v for k, v in row.items() if k in MANIFEST_HEADER_KEYS})
            elif rtype == "lineage_edge":
                extra_edges.extend(_parse_lineage_edges([row]))
            else:
                case_count += 1
    else:
        header, case_count = scan_manifest_json_pass1(target, MANIFEST_HEADER_KEYS, ijson_mod=ijson)
    return Manifest(
        manifest_id=_req_str(header.get("manifest_id", target.stem), "manifest_id_invalid"),
        schema_version=_req_str(header.get("schema_version", "mindie-sim.multilevel-manifest/1"), "schema_version_invalid"),
        cases=_LazyManifestCases(target, is_jsonl=is_jsonl, allow_invalid_durations=allow_invalid_durations, count=case_count),
        lineage_edges=(*_parse_lineage_edges(header.get("lineage_edges")), *extra_edges),
        source_uri=sanitize_capture_uri(str(header["source_uri"])) if header.get("source_uri") else "",
        source_sha256=_req_sha256(header["source_sha256"], "manifest_sha256_invalid") if header.get("source_sha256") else "",
        strict_three_level_blind=bool(header.get("strict_three_level_blind", True)),
    )


def _verify_case_capture(case: GateCase, path: Path) -> CaptureVerificationReceipt | None:
    return _verify_capture(
        case.case_id, case.boundary, path, measured_ns=case.measured_ns, ijson_mod=ijson,
        digest_fn=digest_file_streaming, stream_chunk_bytes=STREAM_CHUNK_BYTES,
        extract_meta_fn=_extract_capture_meta,
    )


def validate_lineage(
    manifest: Manifest | str | Path, fits: FrozenFitManifest, *,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = False,
) -> SplitIndex:
    """Compute normalized transitive lineage closure, verify disk captures, and isolate splits."""
    resolved = stream_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    cap_root = Path(capture_root) if capture_root is not None else None
    uf, seen_cases, acq_uids = _UnionFind(), set(), set()
    compact_rows: list[tuple[str, str, str, bool, str, str, tuple[str, ...]]] = []
    capture_receipts: dict[str, CaptureVerificationReceipt] = {}

    for case in resolved.cases:
        if case.case_id in seen_cases:
            raise ModelingError(f"duplicate_case_id:{case.case_id}")
        seen_cases.add(case.case_id)
        b = case.boundary
        raw_sha = _req_sha256(b.raw_sha256, f"raw_sha256_invalid:{case.case_id}")
        acq_uid = _req_str(b.acquisition_uid, f"acquisition_uid_missing:{case.case_id}")
        run_uid = _req_str(b.run_uid, f"run_uid_missing:{case.case_id}")
        session_scope = (b.session_id.strip() if b.session_id else "") or run_uid
        if require_capture_files or cap_root is not None or capture_files is not None:
            cap_path = resolve_capture_file(
                b.capture_uri, b.source_ref, capture_root=cap_root,
                capture_files=capture_files, case_id=case.case_id, raw_sha256=raw_sha,
            )
            if cap_path is None:
                if require_capture_files:
                    raise ModelingError(f"capture_file_missing:{case.case_id}")
            elif (receipt := _verify_case_capture(case, cap_path)) is not None:
                capture_receipts[case.case_id] = receipt
            if cap_root is not None and b.source_ref:
                ref_map = dict(b.source_ref)
                ref_path = cap_root / PurePosixPath(ref_map["relative_path"])
                if ref_path.is_file() and digest_file_streaming(ref_path) != _req_sha256(ref_map["source_sha256"], f"source_ref_sha256_invalid:{case.case_id}"):
                    raise ModelingError(f"capture_source_ref_digest_mismatch:{case.case_id}")
        tokens = [raw_sha, acq_uid, run_uid, *(normalize_lineage_token(t) for t in b.parent_digests if t and t.strip())]
        uf.find(tokens[0])
        for tok in tokens[1:]:
            uf.union(tokens[0], tok)
        acq_uids.add(acq_uid)
        compact_rows.append((case.case_id, case.level, case.split, case.debug_cross_level, tokens[0], session_scope, b.event_ids))

    for parent, child in (*resolved.lineage_edges, *fits.lineage_edges):
        uf.union(normalize_lineage_token(parent), normalize_lineage_token(child))
    for param in fits.parameters:
        if param.source_uri:
            sanitize_capture_uri(param.source_uri)
        for digest in param.training_digests:
            if digest and digest.strip():
                uf.find(normalize_lineage_token(digest))

    rep_to_acq: dict[str, list[str]] = {}
    for uid in acq_uids:
        rep_to_acq.setdefault(uf.find(uid), []).append(uid)
    canonical_root = lambda tok: min(rep_to_acq[rep]) if (rep := uf.find(tok)) in rep_to_acq else rep

    # Qualify local event IDs by (acquisition_root, session_scope) so distinct
    # captures reusing local IDs never erroneously merge.
    for _cid, _lvl, _spl, _dbg, anchor, session_scope, ev_ids in compact_rows:
        acq_root = canonical_root(anchor)
        for ev in ev_ids:
            if ev and ev.strip():
                uf.union(anchor, f"event::{acq_root}::{session_scope}::{ev.strip()}")
    rep_to_acq.clear()
    for uid in acq_uids:
        rep_to_acq.setdefault(uf.find(uid), []).append(uid)

    case_roots, comp_splits, comp_cases = {}, {}, {}
    train_comps, heldout_comps, debug_only = set(), set(), set()
    heldout_levels_by_root: dict[str, set[str]] = {}
    heldout_roots_by_level: dict[str, set[str]] = {"L1": set(), "L2": set(), "L3": set()}
    for cid, level, split, is_dbg, anchor, _session_scope, _ev_ids in compact_rows:
        root = canonical_root(anchor)
        case_roots[cid] = root
        comp_cases.setdefault(root, []).append(cid)
        if root in comp_splits and comp_splits[root] != split:
            raise ModelingError(f"lineage_split_leakage:root={root}:cases={','.join(sorted(comp_cases[root]))}")
        comp_splits[root] = split
        if split == "train":
            train_comps.add(root)
        else:
            heldout_comps.add(root)
            if is_dbg:
                debug_only.add(cid)
            else:
                heldout_levels_by_root.setdefault(root, set()).add(level)
                heldout_roots_by_level[level].add(root)

    fit_ancestors: dict[str, frozenset[str]] = {}
    for param in fits.parameters:
        roots = frozenset(canonical_root(normalize_lineage_token(d)) for d in param.training_digests if d and d.strip())
        if roots & heldout_comps:
            raise ModelingError(f"lineage_fit_heldout_leakage:parameter={param.parameter_id}:roots={','.join(sorted(roots & heldout_comps))}")
        fit_ancestors[param.parameter_id] = roots
    if resolved.strict_three_level_blind:
        for root, levels in sorted(heldout_levels_by_root.items()):
            if len(levels) > 1:
                raise ModelingError(f"lineage_cross_level_root_overlap:root={root}:levels={','.join(sorted(levels))}")
    return SplitIndex(
        case_roots=MappingProxyType(case_roots), component_splits=MappingProxyType(comp_splits),
        train_components=frozenset(train_comps), heldout_components=frozenset(heldout_comps),
        heldout_roots_by_level=MappingProxyType({k: frozenset(v) for k, v in heldout_roots_by_level.items()}),
        debug_only_cases=frozenset(debug_only), fit_ancestor_components=MappingProxyType(fit_ancestors),
        capture_receipts=MappingProxyType(capture_receipts),
    )


def run_l1(case: GateCase, runner: EvaluationRunner) -> EngineInterval:
    """Execute one independent Level-1 operator/collective case on the Engine."""
    if case.level != "L1" or case.boundary.boundary_kind not in L1_BOUNDARY_KINDS:
        raise ModelingError(f"run_l1_contract_invalid:{case.case_id}:{case.level}:{case.boundary.boundary_kind}")
    return validate_engine_interval(case, runner.run_l1(case), expected_level="L1")


def run_l2(case: GateCase, runner: EvaluationRunner) -> EngineInterval:
    """Execute one independent Level-2 subgraph/block/rung case on the Engine."""
    if case.level != "L2":
        raise ModelingError(f"run_l2_level_mismatch:{case.case_id}:{case.level}")
    if case.boundary.boundary_kind in FORBIDDEN_L2_BOUNDARY_KINDS:
        raise ModelingError(f"l2_derived_from_l3_forbidden:{case.case_id}:{case.boundary.boundary_kind}")
    if not case.block_capture_present or case.boundary.boundary_kind == "missing_block_capture":
        return EngineInterval(case.case_id, "L2", case.boundary.boundary_kind, case.boundary.start_event,
                              case.boundary.end_event, 0, 0, 0, "", status="missing_evidence",
                              missing_reason="block_capture_absent", clock_operator=case.boundary.clock_operator)
    if case.boundary.boundary_kind not in L2_BOUNDARY_KINDS:
        raise ModelingError(f"run_l2_boundary_invalid:{case.case_id}:{case.boundary.boundary_kind}")
    interval = runner.run_l2(case)
    if interval.derived_from_step_division or interval.source_level == "L3":
        raise ModelingError(f"l2_cloned_or_divided_from_l3_forbidden:{case.case_id}")
    return validate_engine_interval(case, interval, expected_level="L2")


def run_l3(case: GateCase, runner: EvaluationRunner) -> EngineInterval:
    """Execute one independent Level-3 model/service step case on the Engine."""
    if case.level != "L3" or case.boundary.boundary_kind not in L3_BOUNDARY_KINDS:
        raise ModelingError(f"run_l3_contract_invalid:{case.case_id}:{case.level}:{case.boundary.boundary_kind}")
    return validate_engine_interval(case, runner.run_l3(case), expected_level="L3")


def _make_residual(
    case: GateCase, root: str, *, measured_ns: int, interval: EngineInterval | None = None,
    signed_error: float = math.nan, status: ResidualStatus = "ok",
    excluded_reason: str = "", signed_delta_ns: int | None = None,
    arm_a_err: float | None = None, arm_b_err: float | None = None,
    capture_verification: CaptureVerificationReceipt | None = None,
) -> Residual:
    scope, events, ranks, expected = (), frozenset(), frozenset(), ()
    if case.level == "L3" or case.debug_cross_level:
        b = case.boundary
        scope = (case.split, b.raw_sha256.lower(), b.acquisition_uid, b.run_uid,
                 b.session_id or b.run_uid, b.clock_domain)
        events, ranks = frozenset(b.event_ids), frozenset(b.device_ranks)
    if case.level == "L3":
        expected = _parse_str_tuple(dict(case.payload).get("component_case_ids"),
                                    f"invalid_drilldown_component_ids:{case.case_id}")
    return Residual(
        case.case_id, case.level, case.cohort, root, measured_ns,
        interval.predicted_ns if interval else 0, interval.engine_receipt if interval else "",
        signed_error, math.nan if math.isnan(signed_error) else abs(signed_error),
        status, excluded_reason, case.boundary.boundary_kind, signed_delta_ns, arm_a_err, arm_b_err, case.regime_dims,
        case.boundary.clock_operator, interval.elapsed_span_ns if interval else None, interval.busy_union_ns if interval else None,
        scope, events, ranks, expected, capture_verification, case.split,
    )


def evaluate(
    manifest: Manifest | str | Path, fits: FrozenFitManifest, runner: EvaluationRunner, *,
    target_split: Literal["heldout", "train", "all"] = "heldout",
    near_zero_delta_ns: int = DEFAULT_NEAR_ZERO_DELTA_NS,
    capture_root: str | Path | None = None,
    capture_files: Mapping[str, str | Path] | None = None,
    require_capture_files: bool = False,
) -> Iterator[Residual]:
    """Verify lineage and captures, then score each case on the Engine.

    A step declares its L1 and L2 case IDs in payload.component_case_ids.
    The inner cases are debug_cross_level rows from the same raw capture and
    session. Each level must cover the physical kernel census from the capture.
    """
    resolved = stream_manifest(manifest) if isinstance(manifest, (str, Path)) else manifest
    split_index = validate_lineage(
        resolved, fits, capture_root=capture_root,
        capture_files=capture_files, require_capture_files=require_capture_files,
    )
    receipt_levels: dict[str, set[str]] = {}
    dispatch = {"L1": run_l1, "L2": run_l2, "L3": run_l3}
    for case in resolved.cases:
        if target_split != "all" and case.split != target_split:
            continue
        root = split_index.root_for(case.case_id)
        is_debug = case.debug_cross_level or case.case_id in split_index.debug_only_cases
        capture_verification = split_index.capture_receipts.get(case.case_id)
        if case.is_counterfactual and not case.has_matched_capture:
            yield _make_residual(case, root, measured_ns=int(case.measured_ns or 0), status="counterfactual_excluded", excluded_reason="counterfactual_without_matched_capture")
            continue
        is_rung = case.boundary.boundary_kind == "paired_rung_delta"
        arm_a, arm_b = case.rung_arm_a_measured_ns, case.rung_arm_b_measured_ns
        if is_rung and (arm_a is None or arm_b is None or arm_a <= 0 or arm_b <= 0):
            yield _make_residual(case, root, measured_ns=int(case.measured_ns or 0), status="invalid_evidence", excluded_reason="invalid_rung_arm_duration")
            continue
        if not is_rung and (case.measured_ns is None or case.measured_ns <= 0):
            yield _make_residual(case, root, measured_ns=int(case.measured_ns or 0), status="invalid_evidence", excluded_reason="zero_or_missing_measured_duration")
            continue
        measured_ns = (case.measured_ns if case.measured_ns is not None else (arm_b - arm_a)) if is_rung else case.measured_ns  # type: ignore[operator]
        assert measured_ns is not None
        interval = dispatch[case.level](case, runner)
        if interval.status == "missing_evidence":
            yield _make_residual(case, root, measured_ns=measured_ns, status="missing_evidence", excluded_reason=interval.missing_reason)
            continue
        for rcpt_key in (interval.engine_receipt, *(f"engine_uid:{u}" for u in (interval.run_receipt.engine_uid.split("+") if interval.run_receipt else ()))):
            seen = receipt_levels.setdefault(rcpt_key, set())
            seen.add(case.level)
            if len(seen) > 1:
                raise ModelingError(f"cross_level_engine_receipt_clone:{rcpt_key}")
        if is_rung:
            assert arm_a is not None and arm_b is not None and interval.arm_a_predicted_ns is not None and interval.arm_b_predicted_ns is not None
            delta_ns = interval.predicted_ns - measured_ns
            near_zero = measured_ns == 0 or abs(measured_ns) <= max(near_zero_delta_ns, int(max(abs(arm_a), abs(arm_b)) * NEAR_ZERO_DELTA_MIN_RATIO))
            status: ResidualStatus = "debug_only" if is_debug else ("near_zero_delta" if near_zero else "ok")
            reason = "debug_cross_level_not_blind_acceptance" if is_debug else ("near_zero_rung_delta" if near_zero else "")
            yield _make_residual(
                case, root, measured_ns=measured_ns, interval=interval,
                signed_error=math.nan if (is_debug or near_zero) else delta_ns / float(measured_ns),
                status=status, excluded_reason=reason, signed_delta_ns=delta_ns,
                arm_a_err=(interval.arm_a_predicted_ns - arm_a) / float(arm_a),
                arm_b_err=(interval.arm_b_predicted_ns - arm_b) / float(arm_b),
                capture_verification=capture_verification,
            )
        else:
            yield _make_residual(
                case, root, measured_ns=measured_ns, interval=interval,
                signed_error=(interval.predicted_ns - measured_ns) / float(measured_ns),
                status="debug_only" if is_debug else "ok",
                excluded_reason="debug_cross_level_not_blind_acceptance" if is_debug else "",
                capture_verification=capture_verification,
            )


def _validate_tolerance(threshold_pct: float, atol_ns: float) -> None:
    if type(atol_ns) not in (int, float) or not 0 <= atol_ns <= DEFAULT_ATOL_NS:
        raise ModelingError("physical_atol_ns_out_of_bounds")
    if type(threshold_pct) not in (int, float) or not 0 < threshold_pct <= DEFAULT_THRESHOLD_PCT:
        raise ModelingError("physical_threshold_pct_out_of_bounds")


def exceeds_physical_tolerance(
    predicted_ns: int | float, measured_ns: int | float, *, threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    atol_ns: float = DEFAULT_ATOL_NS,
) -> bool:
    """Check pointwise error against the relative tolerance and capped timer floor."""
    _validate_tolerance(threshold_pct, atol_ns)
    return abs(predicted_ns - measured_ns) > max(threshold_pct / PERCENT_SCALE * abs(measured_ns), atol_ns)


def evaluate_vertical_drilldown(
    step_predicted_ns: int, step_measured_ns: int, components: Sequence[tuple[str, str, int, int]], *,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT, atol_ns: float = DEFAULT_ATOL_NS,
) -> dict[str, Any]:
    """Score a step and its same-capture operator families and blocks."""
    exceeds = partial(exceeds_physical_tolerance, threshold_pct=threshold_pct, atol_ns=atol_ns)
    return score_vertical_drilldown(step_predicted_ns, step_measured_ns, components, exceeds)


def _step_census(step: Residual, candidates: Iterable[Residual]) -> tuple[str, list[tuple[str, str, int, int]]]:
    expected = set(step.expected_components)
    if not expected or len(expected) != len(step.expected_components):
        return "missing_or_invalid_component_census", []
    receipt = step.capture_verification
    if receipt is None or not (census := receipt.window_kernels):  # Includes clipped busy-union contributors.
        return "step_physical_census_unavailable", []
    inner, verified, coverage = [], [], {level: set() for level in DRILLDOWN_LEVELS}
    outside_step = bundled_l1 = False
    for row in candidates:
        proof = row.capture_verification
        kernels = proof.selected_kernels if proof else frozenset()
        in_window = bool(kernels) and kernels <= census
        if row.case_id not in expected and not in_window:
            continue
        inner.append(row.case_id)
        bundled_l1 |= row.level == "L1" and len(kernels) != len({rank for rank, _ in kernels})
        if not (in_window and row.status == "debug_only" and row.engine_receipt and row.measured_ns > 0):
            outside_step |= bool(kernels) and not in_window
            continue
        verified.append(row)
        coverage[f"{row.level}_in_L3"].update(kernels)
    parts = [(f"{row.level}_in_L3", row.case_id, row.predicted_ns, row.measured_ns) for row in verified]
    if bundled_l1:
        return "l1_drilldown_requires_individual_occurrences", parts
    if set(inner) != expected or len(inner) != len(expected):
        return "component_census_mismatch", parts
    if outside_step:
        return "component_kernels_outside_step", parts
    if len(verified) != len(inner):
        return "component_evidence_unavailable", parts
    if {part[0] for part in parts} != set(DRILLDOWN_LEVELS):
        return "component_levels_missing", parts
    if any(kernels != census for kernels in coverage.values()):
        return "component_physical_kernels_incomplete", parts
    return "", parts


def _score_step_censuses(
    steps: Iterable[Residual], inner: Mapping[tuple[str, ...], list[Residual]],
    exceeds: Callable[[int, int], bool],
) -> tuple[int, tuple[tuple[str, dict[str, Any]], ...]]:
    reports, complete = [], 0
    for step in steps:
        reason, components = _step_census(step, inner.get(step.capture_scope, ()))
        report = score_vertical_drilldown(step.predicted_ns, step.measured_ns, components, exceeds)
        if reason:
            report.update(passed=False, failure_reason=reason)
        else:
            complete += 1
        reports.append((step.case_id, report))
    return complete, tuple(reports)


@dataclass(slots=True)
class _CohortRows:
    apes: list[float] = field(default_factory=list)
    signed: list[float] = field(default_factory=list)
    roots: set[str] = field(default_factory=set)
    tails: list[str] = field(default_factory=list)
    evaluated: set[str] = field(default_factory=set)
    excluded: Counter[str] = field(default_factory=Counter)
    steps: list[Residual] = field(default_factory=list)
    train_steps: list[Residual] = field(default_factory=list)
    missing: int = 0
    independent: int = 0


def summarize_cohorts(
    residuals: Iterable[Residual], *, threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    strict_pointwise: bool = True, atol_ns: float = DEFAULT_ATOL_NS,
) -> dict[CohortKey, CohortSummary]:
    """Report descriptive percentages and score pointwise evidence at each level."""
    _validate_tolerance(threshold_pct, atol_ns)
    exceeds = partial(exceeds_physical_tolerance, threshold_pct=threshold_pct, atol_ns=atol_ns)
    buckets: dict[CohortKey, _CohortRows] = {}
    inner: dict[tuple[str, ...], list[Residual]] = {}
    for row in residuals:
        diagnostic = row.level in {"L1", "L2"} and bool(row.capture_scope)
        if diagnostic:
            inner.setdefault(row.capture_scope, []).append(row)
        bucket = buckets.get(row.cohort)
        if bucket is None:
            bucket = buckets[row.cohort] = _CohortRows()
        non_heldout = row.split != "heldout"
        bucket.independent += not diagnostic and not non_heldout
        if not diagnostic and not non_heldout and row.status == "ok" and math.isfinite(row.absolute_percentage_error):
            bucket.apes.append(row.absolute_percentage_error * PERCENT_SCALE)
            bucket.signed.append(row.signed_error * PERCENT_SCALE)
            bucket.roots.add(row.acquisition_root)
            bucket.evaluated.add(row.case_id)
            if exceeds(row.predicted_ns, row.measured_ns):
                bucket.tails.append(row.case_id)
            if row.level == "L3":
                bucket.steps.append(row)
        else:
            if non_heldout and row.level == "L3" and row.status == "ok" and math.isfinite(row.absolute_percentage_error):
                bucket.train_steps.append(row)
            reason = row.excluded_reason or ("train_split_not_blind_acceptance" if non_heldout else row.status)
            bucket.excluded[reason] += 1
            bucket.missing += not diagnostic and not non_heldout and row.status in {"missing_evidence", "invalid_evidence"}

    summaries: dict[CohortKey, CohortSummary] = {}
    for key, bucket in buckets.items():
        complete, drilldowns = _score_step_censuses(bucket.steps, inner, exceeds)
        _, train_drilldowns = _score_step_censuses(bucket.train_steps, inner, exceeds)
        failed = (("max_ape_pct",) if strict_pointwise and bucket.tails else ())
        if any(report["failed_components"] for _, report in drilldowns):
            failed += ("vertical_drilldown",)
        apes = sorted(bucket.apes)
        status: CohortStatus = "fail" if failed else (
            "insufficient_evidence" if not apes or bucket.missing or complete != len(bucket.steps) else "pass"
        )
        p90_idx = math.ceil(TAIL_QUANTILE_P90 * len(apes)) - 1
        summaries[key] = CohortSummary(
            key, status, len(bucket.roots), len(apes), statistics.fmean(apes) if apes else None,
            statistics.median(apes) if apes else None, statistics.median(bucket.signed) if apes else None,
            apes[p90_idx] if apes else None, apes[-1] if apes else None,
            bucket.missing, tuple(sorted(bucket.excluded.items())), threshold_pct, failed,
            tuple(bucket.tails), tuple(sorted(bucket.roots)), tuple(sorted(bucket.evaluated)),
            atol_ns=atol_ns, pointwise_failed_count=len(bucket.tails), complete_drilldowns=complete,
            drilldowns=(*drilldowns, *train_drilldowns), diagnostic_only=not bucket.independent,
        )
    return summaries


def gate_decision(
    summaries: Mapping[CohortKey, CohortSummary], *,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    atol_ns: float = DEFAULT_ATOL_NS,
    required_levels: Sequence[Level] = REQUIRED_LEVELS,
    require_disjoint_level_roots: bool = True,
    expected_case_ids: Iterable[str] | None = None,
    evaluated_case_ids: Iterable[str | Residual] | None = None,
    calibrated_count: int | None = None,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Compute the release verdict from cohort results and optional planned coverage.

    Coverage comes from scored residuals recorded in summaries. The
    evaluated_case_ids argument is retained for caller compatibility. Physical
    tolerance settings must match those used to score the cohorts.
    """
    _validate_tolerance(threshold_pct, atol_ns)
    if any(atol_ns != v.atol_ns or threshold_pct != v.threshold_pct for v in summaries.values()):
        raise ModelingError("cohort_physical_tolerance_mismatch")
    missing_planned_cases = []
    if expected_case_ids is not None:
        evaluated = {case_id for summary in summaries.values() for case_id in summary.evaluated_ok_case_ids}
        missing_planned_cases = sorted(set(expected_case_ids) - evaluated)
    pointwise_failures = {
        k.as_id() for k, v in summaries.items()
        if v.pointwise_failed_count > 0
    }
    failed = sorted(pointwise_failures | {k.as_id() for k, v in summaries.items() if v.status == "fail"})
    insufficient = sorted(
        k.as_id() for k, v in summaries.items()
        if not v.diagnostic_only and (v.status == "insufficient_evidence"
                                     or k.level == "L3" and v.complete_drilldowns != v.n_rows)
    )
    tail_warnings = sorted(k.as_id() for k, v in summaries.items() if v.tail_exceeded)
    covered_levels: set[str] = set()
    root_levels: dict[str, set[str]] = {}
    for key, summary in summaries.items():
        if summary.n_rows > 0 and summary.n_roots > 0:
            covered_levels.add(key.level)
            for root in summary.acquisition_roots:
                root_levels.setdefault(root, set()).add(key.level)
    missing_levels = sorted(lvl for lvl in required_levels if lvl not in covered_levels)
    overlapping_roots = sorted(r for r, lvls in root_levels.items() if len(lvls) > 1)
    incomplete = (
        not covered_levels or insufficient or missing_levels or missing_planned_cases
        or (require_disjoint_level_roots and overlapping_roots)
        or plan_only or (calibrated_count is not None and calibrated_count <= 0)
    )
    status = "fail" if failed else ("insufficient_evidence" if incomplete else "pass")
    return {
        "status": status, "cohort_count": len(summaries),
        "covered_levels": sorted(covered_levels), "missing_levels": missing_levels,
        "overlapping_level_roots": overlapping_roots,
        "failed_cohorts": failed, "insufficient_cohorts": insufficient, "tail_warning_cohorts": tail_warnings,
        "pointwise_threshold_exceeded": sorted(pointwise_failures),
        "vertical_drilldowns": {case_id: report for summary in summaries.values() for case_id, report in summary.drilldowns},
        "missing_planned_cases": missing_planned_cases, "plan_only": plan_only, "calibrated_count": calibrated_count,
    }
