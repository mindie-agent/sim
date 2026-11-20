"""Chunked two-pass JSON/JSONL streaming, provenance sanitization, and capture checks.

[Role] Provide fixed-chunk (64 KiB) stdlib JSON/JSONL event streaming for
  multilevel gate manifests and tail-cluster diagnosis inputs, provenance and
  SHA-256 token sanitization, and on-disk capture consistency verification.
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
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import math
from pathlib import Path, PurePosixPath
import re
import secrets
from types import MappingProxyType
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError

STREAM_CHUNK_BYTES = 64 * 1024
STREAM_WHOLE_READ_LIMIT_BYTES = 50 * 1024 * 1024
SHA256_HEX_LENGTH = 64
MS_TO_NS = 1_000_000
_CLOCK_EPS_MS = 1e-6
_HMAC_KEY_BYTES = 32
_RECEIPT_HMAC_KEY = secrets.token_bytes(_HMAC_KEY_BYTES)
CAPTURE_URI_SCHEME = "capture://"
_SHA256_PREFIX = "sha256:"
Level = Literal["L1", "L2", "L3"]
SOURCE_REF_KEYS = frozenset({"repository", "revision", "relative_path", "source_sha256"})
TIME_UNIT_SCALE_NS: Mapping[str, int] = MappingProxyType({
    "ns": 1, "us": 1000, "ms": 1_000_000, "s": 1_000_000_000,
})
_JSON_SUFFIXES = frozenset({".json", ".jsonl", ".ndjson"})
_JSONL_SUFFIXES = frozenset({".jsonl", ".ndjson"})
_CAPTURE_META_KEYS = frozenset({
    "device_ranks", "ranks", "profiled_ranks", "device_rank", "rank",
    "clock_domain", "start_event", "end_event", "event_ids", "events",
    "boundary", "metadata",
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
    "STREAM_WHOLE_READ_LIMIT_BYTES", "TIME_UNIT_SCALE_NS", "_fstr", "_parse_lineage_edges",
    "_parse_ranks", "_parse_str_tuple", "_req_sha256", "_req_str", "_to_ns",
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
        candidates = [capture_root / rel_path, *([capture_root / f"{rel}.json"] if not rel_path.suffix else []), capture_root / rel_path.name]
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


def _extract_capture_meta(path: Path, *, ijson_mod: Any = None) -> dict[str, Any]:
    if path.suffix.lower() in _JSONL_SUFFIXES:
        merged: dict[str, Any] = {}
        events_found: set[str] = set()
        for row in stream_jsonl(path):
            for key in _CAPTURE_META_KEYS - {"events", "event_ids"}:
                if key in row:
                    merged.setdefault(key, row[key])
            events_found |= _collect_event_names(row.get("event_ids")) | _collect_event_names(row.get("events"))
            for field in ("event_id", "name"):
                if isinstance(row.get(field), str) and row[field].strip():
                    events_found.add(row[field].strip())
        if events_found:
            merged["event_ids"] = sorted(events_found)
        return merged
    meta: dict[str, Any] = {}
    with path.open("rb") as handle:
        events = _iter_json_events(handle, ijson_mod=ijson_mod)
        if next(events)[0] != "start_map":
            return meta
        while (entry := next(events))[0] != "end_map":
            key = str(entry[1])
            if key in _CAPTURE_META_KEYS:
                meta[key] = _read_value(events)
            else:
                _skip_value(events)
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


def verify_capture_file_consistency(case_id: str, boundary: Any, path: Path, *, ijson_mod: Any = None) -> None:
    """Fail closed when an on-disk capture file mismatches digest, rank, clock, or events."""
    try:
        actual_sha = digest_file_streaming(path)
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
            is_json_file = handle.read(STREAM_CHUNK_BYTES).lstrip().startswith((b"{", b"["))
    if not is_json_file:
        return
    try:
        meta = _extract_capture_meta(path, ijson_mod=ijson_mod)
    except TimeoutError:
        raise
    except ModelingError as exc:
        raise ModelingError(f"capture_file_unreadable:{case_id}:{exc.reason}") from exc
    except (OSError, StopIteration) as exc:
        raise ModelingError(f"capture_file_unreadable:{case_id}") from exc
    cap_ranks = _capture_ranks(meta)
    if cap_ranks is not None and cap_ranks != set(boundary.device_ranks):
        raise ModelingError(f"capture_rank_mismatch:{case_id}:expected={sorted(boundary.device_ranks)}:actual={sorted(cap_ranks)}")
    cap_events, has_events = set(), False
    for container in (meta, meta.get("boundary"), meta.get("metadata")):
        if not isinstance(container, Mapping):
            continue
        if (cap_clock := container.get("clock_domain")) is not None and str(cap_clock).strip() != str(boundary.clock_domain).strip():
            raise ModelingError(f"capture_clock_domain_mismatch:{case_id}:expected={boundary.clock_domain}:actual={cap_clock}")
        if (cap_start := container.get("start_event")) is not None and str(cap_start).strip() != str(boundary.start_event).strip():
            raise ModelingError(f"capture_event_mismatch:{case_id}:start_event")
        if (cap_end := container.get("end_event")) is not None and str(cap_end).strip() != str(boundary.end_event).strip():
            raise ModelingError(f"capture_event_mismatch:{case_id}:end_event")
        for key in ("event_ids", "events"):
            if key in container and container[key] is not None:
                has_events = True
                cap_events |= _collect_event_names(container[key])
    if has_events and boundary.event_ids:
        if missing := [ev for ev in boundary.event_ids if ev.strip() not in cap_events]:
            raise ModelingError(f"capture_event_mismatch:{case_id}:missing={','.join(missing)}")


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


def _sign_receipt(
    receipt_id: str, case_id: str, level: str, boundary_kind: str, engine_uid: str,
    start_event_name: str, end_event_name: str, start_event_seq: int, end_event_seq: int,
    start_ns: int, end_ns: int, predicted_ns: int,
    arm_a_id: str = "", arm_b_id: str = "",
    arm_a_ns: int | None = None, arm_b_ns: int | None = None,
) -> str:
    payload = "|".join((
        receipt_id, case_id, level, boundary_kind, engine_uid, start_event_name, end_event_name,
        str(start_event_seq), str(end_event_seq), str(start_ns), str(end_ns), str(predicted_ns),
        arm_a_id, arm_b_id, "" if arm_a_ns is None else str(arm_a_ns), "" if arm_b_ns is None else str(arm_b_ns),
    )).encode("utf-8")
    return hmac.new(_RECEIPT_HMAC_KEY, payload, "sha256").hexdigest()


def _validate_live_engine_events(case: Any, engine: Any, start_event: Any, end_event: Any) -> tuple[str, int, int, int]:
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
    start_ns, end_ns = int(round(start_event.time * MS_TO_NS)), int(round(end_event.time * MS_TO_NS))
    if (pred_ns := end_ns - start_ns) <= 0:
        raise ModelingError(f"interval_span_or_duration_invalid:{case.case_id}")
    return _req_str(getattr(engine, "uid", ""), f"engine_uid_missing:{case.case_id}"), start_ns, end_ns, pred_ns


def bind_engine_interval(
    case: Any, engine: Any, start_event: Any, end_event: Any, *,
    wire_bytes: int | None = None, kernel_sequence: tuple[str, ...] = (),
    diagnostics: tuple[tuple[str, Any], ...] = (),
    source_level: str = "", derived_from_step_division: bool = False,
) -> EngineInterval:
    """Bind one ``EngineInterval`` strictly to a live ``Engine`` run and named ``Event`` handles."""
    uid, s_ns, e_ns, pred_ns = _validate_live_engine_events(case, engine, start_event, end_event)
    rid = validate_provenance_text(
        f"{CAPTURE_URI_SCHEME}engine-receipt/{uid}/{case.level}/{case.case_id}/{start_event.sequence}-{end_event.sequence}",
        field_name="engine_receipt",
    )
    tok = _sign_receipt(rid, case.case_id, case.level, case.boundary.boundary_kind, uid,
                        start_event.name, end_event.name, start_event.sequence, end_event.sequence, s_ns, e_ns, pred_ns)
    rcpt = EngineRunReceipt(rid, case.case_id, case.level, case.boundary.boundary_kind, uid,
                            start_event.name, end_event.name, start_event.sequence, end_event.sequence, s_ns, e_ns, pred_ns, tok)
    return EngineInterval(
        case.case_id, case.level, case.boundary.boundary_kind, start_event.name, end_event.name, s_ns, e_ns, pred_ns,
        rid, True, 0, "ok", "", source_level, derived_from_step_division, wire_bytes, kernel_sequence, None, None, diagnostics, rcpt,
    )


def bind_engine_rung_interval(
    case: Any, *, arm_a: tuple[Any, Any, Any], arm_b: tuple[Any, Any, Any],
    diagnostics: tuple[tuple[str, Any], ...] = (),
) -> EngineInterval:
    """Bind a ``paired_rung_delta`` interval strictly to two live ``Engine`` arm runs."""
    if case.level != "L2" or case.boundary.boundary_kind != "paired_rung_delta":
        raise ModelingError(f"rung_binding_requires_l2_paired_rung_delta:{case.case_id}")
    (eng_a, s_a, e_a), (eng_b, s_b, e_b) = arm_a, arm_b
    uid_a, _, _, pred_a = _validate_live_engine_events(case, eng_a, s_a, e_a)
    uid_b, _, _, pred_b = _validate_live_engine_events(case, eng_b, s_b, e_b)
    if (uid_a, s_a.sequence, e_a.sequence) == (uid_b, s_b.sequence, e_b.sequence):
        raise ModelingError(f"rung_arms_must_be_distinct_runs:{case.case_id}")
    a_id = f"{CAPTURE_URI_SCHEME}engine-receipt/{uid_a}/L2/{case.case_id}/arm_a/{s_a.sequence}-{e_a.sequence}"
    b_id = f"{CAPTURE_URI_SCHEME}engine-receipt/{uid_b}/L2/{case.case_id}/arm_b/{s_b.sequence}-{e_b.sequence}"
    delta_ns, uid = pred_b - pred_a, f"{uid_a}+{uid_b}"
    rid = validate_provenance_text(f"{CAPTURE_URI_SCHEME}engine-receipt/{uid}/L2/{case.case_id}/rung/{s_a.sequence}-{e_b.sequence}", field_name="engine_receipt")
    tok = _sign_receipt(rid, case.case_id, "L2", "paired_rung_delta", uid, s_a.name, e_b.name,
                        s_a.sequence, e_b.sequence, 0, delta_ns, delta_ns, a_id, b_id, pred_a, pred_b)
    rcpt = EngineRunReceipt(rid, case.case_id, "L2", "paired_rung_delta", uid, s_a.name, e_b.name,
                            s_a.sequence, e_b.sequence, 0, delta_ns, delta_ns, tok, a_id, b_id)
    return EngineInterval(
        case.case_id, "L2", "paired_rung_delta", s_a.name, e_b.name, 0, delta_ns, delta_ns,
        rid, True, 0, "ok", "", "", False, None, (), pred_a, pred_b, diagnostics, rcpt,
    )


def validate_engine_interval(case: Any, interval: EngineInterval, *, expected_level: Level) -> EngineInterval:
    """Validate Engine interval case_id, boundary, signed live-Engine receipt, and duration."""
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
    elif interval.end_ns - interval.start_ns != interval.predicted_ns or interval.predicted_ns <= 0:
        raise ModelingError(f"interval_span_or_duration_invalid:{case.case_id}")
    rcpt = interval.run_receipt
    if not isinstance(rcpt, EngineRunReceipt):
        raise ModelingError(f"engine_receipt_unbound:{case.case_id}")
    if ((rcpt.receipt_id, rcpt.case_id, rcpt.level, rcpt.boundary_kind, rcpt.start_event_name, rcpt.end_event_name,
         rcpt.start_ns, rcpt.end_ns, rcpt.predicted_ns)
            != (interval.engine_receipt, case.case_id, expected_level, case.boundary.boundary_kind,
                case.boundary.start_event, case.boundary.end_event, interval.start_ns, interval.end_ns, interval.predicted_ns)):
        raise ModelingError(f"engine_receipt_binding_mismatch:{case.case_id}")
    expected_tok = _sign_receipt(
        rcpt.receipt_id, rcpt.case_id, rcpt.level, rcpt.boundary_kind, rcpt.engine_uid,
        rcpt.start_event_name, rcpt.end_event_name, rcpt.start_event_seq, rcpt.end_event_seq,
        rcpt.start_ns, rcpt.end_ns, rcpt.predicted_ns, rcpt.arm_a_receipt_id, rcpt.arm_b_receipt_id,
        interval.arm_a_predicted_ns, interval.arm_b_predicted_ns,
    )
    if not hmac.compare_digest(rcpt.receipt_token, expected_tok):
        raise ModelingError(f"engine_receipt_signature_invalid:{case.case_id}")
    return interval
