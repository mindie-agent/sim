"""Observation capture ingestion, verification, schema validation and storage.

[Role] Load, validate, verify, and persist profiling observation captures
  and kernel API requirement manifests.
[Invariants] Rejects corrupt or mismatched capture metadata fail-closed.
[Timeouts & Locks] Local JSON/filesystem access only; no locks.
[Side-Effect Policy] Writes only to caller-supplied store directories."""
from __future__ import annotations

PROFILER_API_REQUIREMENT = (
    "Profiling imports require the companion profiler development source: "
    "python -m pip install -e /path/to/profiler. Released mindie-profiler 0.5.0 "
    "does not provide normalized/tensor_contracts; the profiling extra alone "
    "is insufficient until a compatible release is available."
)


def kernel_rows(path, **kwargs):
    try:
        from mindie_profiler.normalized import kernel_rows as read_rows
    except ModuleNotFoundError as exc:
        if exc.name not in {"mindie_profiler", "mindie_profiler.normalized"}:
            raise
        raise ModuleNotFoundError(PROFILER_API_REQUIREMENT) from exc
    yield from read_rows(path, **kwargs)


import json
import math
from pathlib import Path

try:
    import ijson
except ImportError:  # pragma: no cover - stdlib fallback when ijson is absent
    ijson = None


class _Moments:
    """Constant-memory exact moments; no invented quantiles from chunk stats."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf

    def add(self, value):
        self.count += 1
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def summary(self):
        return {} if not self.count else {
            "count": self.count, "mean": self.total / self.count,
            "min": self.minimum, "max": self.maximum,
        }


def _stream_service_observation(stream, wanted):
    request_fields = {"ok", "completion_tokens", "decode_ms", "ttft_ms", "n_text_chunks"}
    record, request = {}, {}
    tpot, last_text_e2e = _Moments(), _Moments()
    observed_requests = successful = exact_usage = output_tokens = 0
    # Parse scalar leaves so token arrays and full response text are never
    # accumulated, even in captures hundreds of megabytes in size.
    for prefix, event, value in ijson.parse(stream, use_float=True):
        if prefix == "per_request.item" and event == "start_map":
            request = {}
        elif prefix == "per_request.item" and event == "end_map":
            observed_requests += 1
            if request.get("ok"):
                successful += 1
                tokens = request.get("completion_tokens")
                decode = request.get("decode_ms")
                first = request.get("ttft_ms")
                if tokens is not None:
                    exact_usage += 1
                    output_tokens += tokens
                    if tokens > 1 and decode is not None:
                        tpot.add(decode / (tokens - 1))
                if first is not None and decode is not None:
                    last_text_e2e.add(first + decode)
        elif prefix.startswith("per_request.item.") and event in {"number", "boolean", "null"}:
            key = prefix.removeprefix("per_request.item.")
            if key in request_fields:
                request[key] = value
        keys = prefix.split(".")
        if keys[0] not in wanted or len(keys) > 2 or event not in {"string", "number", "boolean", "null"}:
            continue
        if len(keys) == 1:
            record[keys[0]] = value
        else:
            record.setdefault(keys[0], {})[keys[1]] = value
    return (record, tpot, last_text_e2e, observed_requests, successful,
            exact_usage, output_tokens)


def _summarize_service_document(doc, wanted):
    record = {}
    tpot, last_text_e2e = _Moments(), _Moments()
    observed_requests = successful = exact_usage = output_tokens = 0
    if isinstance(doc, dict):
        for key in wanted:
            if key not in doc:
                continue
            val = doc[key]
            if not isinstance(val, (dict, list)):
                record[key] = val
            elif isinstance(val, dict):
                for subkey, subval in val.items():
                    if not isinstance(subval, (dict, list)):
                        record.setdefault(key, {})[subkey] = subval
        per_req = doc.get("per_request")
        if isinstance(per_req, list):
            for item in per_req:
                if not isinstance(item, dict):
                    continue
                observed_requests += 1
                if item.get("ok") is True:
                    successful += 1
                    tokens = item.get("completion_tokens")
                    decode = item.get("decode_ms")
                    first = item.get("ttft_ms")
                    if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
                        exact_usage += 1
                        output_tokens += tokens
                        if tokens > 1 and isinstance(decode, (int, float)) and not isinstance(decode, bool):
                            tpot.add(decode / (tokens - 1))
                    if (isinstance(first, (int, float)) and not isinstance(first, bool)
                            and isinstance(decode, (int, float)) and not isinstance(decode, bool)):
                        last_text_e2e.add(first + decode)
    return (record, tpot, last_text_e2e, observed_requests, successful,
            exact_usage, output_tokens)


def service_observation(path):
    wanted = {"label", "model", "input_len", "output_len", "concurrency", "num_prompts",
              "n_ok", "n_failed", "wall_s", "ttft_ms", "tpot_ms", "e2e_ms",
              "output_tokens_per_s", "total_output_tokens", "output_throughput",
              "output_throughput_tok_s", "token_exact"}
    if ijson is not None:
        with Path(path).open("rb") as stream:
            stats = _stream_service_observation(stream, wanted)
    else:
        with Path(path).open("rb") as stream:
            doc = json.load(stream)
        stats = _summarize_service_document(doc, wanted)
    (record, tpot, last_text_e2e, observed_requests, successful,
     exact_usage, output_tokens) = stats
    if "n_ok" not in record or "ttft_ms" not in record:
        raise ValueError("not_a_raw_service_measurement")
    # The historical bench_stream producer divides by nonempty SSE events,
    # which can carry multiple speculative tokens. Preserve that original
    # statistic, but normalize TPOT only from actual usage and elapsed decode.
    record["reported_metrics"] = {key: record.pop(key) for key in
                                  ("tpot_ms", "e2e_ms", "output_throughput_tok_s",
                                   "output_tokens_per_s", "output_throughput", "total_output_tokens")
                                  if key in record}
    record["tpot_ms"] = tpot.summary()
    record["e2e_ms"] = last_text_e2e.summary()
    complete_usage = successful == exact_usage == record["n_ok"] and successful > 0
    record["normalization"] = {
        "per_request_records": observed_requests,
        "successful_requests": successful, "requests_with_usage": exact_usage,
        "tpot": "per-request decode_ms / (usage.completion_tokens - 1)",
        "e2e": "first text latency + first-to-last text elapsed time",
        "quantiles": "not reconstructed from source chunk-interval quantiles",
    }
    wall = record.get("wall_s")
    record["output_tps"] = output_tokens / wall if complete_usage and wall and wall > 0 else None
    record["source"] = str(Path(path).resolve())
    record["boundary"] = "client_stream"
    return record


def compare(prediction, observed, *, boundary, statistics=("p50", "mean")):
    if observed.get("boundary") != boundary:
        raise ValueError("observation_boundary_conflicts_with_declared_source_boundary")
    rows = []
    for name in ("ttft_ms", "tpot_ms", "e2e_ms"):
        actual = observed.get(name, {})
        for statistic in statistics:
            measured = actual.get(statistic, actual.get("median") if statistic == "p50" else None)
            estimated = prediction["metrics"].get(name, {}).get(statistic)
            if measured is None or estimated is None:
                continue
            rows.append({"metric": name, "statistic": statistic, "predicted": estimated,
                         "observed": measured, "error": estimated - measured,
                         "relative_error": (estimated / measured - 1) if measured else None})
    measured, estimated = observed.get("output_tps"), prediction["metrics"].get("output_tps")
    if measured is not None and estimated is not None:
        rows.append({"metric": "output_tps", "statistic": "aggregate", "predicted": estimated,
                     "observed": measured, "error": estimated - measured,
                     "relative_error": estimated / measured - 1 if measured else None})
    matching = prediction.get("metric_boundary") == boundary
    return {"schema": "mindie-sim.validation/1", "source": observed.get("source"),
            "prediction_boundary": prediction.get("metric_boundary"), "observation_boundary": boundary,
            "comparable": matching, "rows": rows, "normalization": observed.get("normalization"),
            "boundary_difference": None if matching else
            "Client observations include transport and frontend work outside device token commitment."}


def fit_resource_rate(samples):
    """Fit a primitive's nonnegative setup + work/rate from training trials."""
    from .service_fitting import nonnegative_affine
    train = [sample for sample in samples if sample["split"] == "train"]
    heldout = [sample for sample in samples if sample["split"] == "heldout"]
    n = len(train)
    if any(not math.isfinite(s[key]) or s[key] <= 0 for s in samples
           for key in ("work_bytes", "duration_ms")):
        raise ValueError("primitive_fit_requires_positive_finite_work_and_duration")
    intercept, rate = nonnegative_affine([(s["work_bytes"], s["duration_ms"]) for s in train])
    return {"bytes_per_ms": rate, "setup_ms": intercept, "train_trials": n,
            "heldout": [{"work_bytes": s["work_bytes"], "observed_ms": s["duration_ms"],
                         "predicted_ms": intercept + s["work_bytes"] / rate,
                         "relative_error": (intercept + s["work_bytes"] / rate) / s["duration_ms"] - 1}
                        for s in heldout]}


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
from mindie_sim.schema.parallel import DieIndex, GlobalRank, RankCoord

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
_TRACE_GLOBAL_RANK_LABEL = re.compile(r"rank(?: id)?[ _:-]+(?P<rank>[0-9]+)", re.IGNORECASE)
_TRACE_LOCAL_DEVICE_LABEL = re.compile(r"(?:device|npu)(?: id)?[ _:-]+(?P<device>[0-9]+)", re.IGNORECASE)
_TRACE_GLOBAL_RANK_KEYS = ("device_rank", "rank_id", "rank")
_TRACE_LOCAL_DEVICE_KEYS = ("device_id", "device", "Device_id", "Device Id", "Device ID")
_TRACE_RANK_KEYS = (*_TRACE_GLOBAL_RANK_KEYS, *_TRACE_LOCAL_DEVICE_KEYS)
_TRACE_HOST_CATEGORIES = frozenset({"cpu_op", "python_function", "cann_api", "host"})
_EVENT_ID_KEYS = ("event_id", "id", "name")
_TASK_ID_KEYS = ("Task Id", "Task ID", "task_id", "taskId")
_CONNECTION_ID_KEYS = ("Connection Id", "Connection ID", "connection_id", "connectionId")
_CALL_ID_KEYS = ("Call Id", "Call ID", "call_id", "callId", "Host Call Id", "host_call_id", "aclnn_call_id", "Api Id", "api_id")
_STREAM_ID_KEYS = ("Stream Id", "Stream ID", "Physic Stream Id", "stream_id", "streamId", "stream")
_JSON_ROW_KEYS = frozenset((*_EVENT_ID_KEYS, *_TRACE_RANK_KEYS, *_TASK_ID_KEYS, *_CONNECTION_ID_KEYS,
                            *_STREAM_ID_KEYS, *_CALL_ID_KEYS, "ph", "pid", "tid", "args"))
_HOST_API_PREFIXES = ("aclnn", "aclop", "hccl", "Hccl", "aten::", "npu_", "ops.")
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
L1_COMPOSITE_BOUNDARY_KINDS = frozenset({"aclnn_call", "native_composite", "hccl_api", "single_operator",
                                        "collective_call", "npu_event_call"})
L3_BOUNDARY_KINDS = frozenset({"model_step", "service_step", "e2e_step", "device_only_step", "host_device_step"})

_KernelOccurrence = tuple[int, int]  # (device rank, one-based source row)
_RankWindows = Mapping[int, tuple[Decimal, Decimal]]
_HostThread = tuple[int | str, int | str]
_HostLaunch = tuple[_HostThread, Decimal, Decimal]


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
    stream: str | None = None
    call_id: str | None = None
    connection_id: str | None = None

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
    candidates = []
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
    if rel_ref := dict(source_ref).get("relative_path"):
        candidates.append(capture_root / PurePosixPath(rel_ref))
    return next((candidate for candidate in candidates if candidate.is_file()), None)


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


def _trace_coords(row: Mapping[str, Any]) -> tuple[GlobalRank | None, DieIndex | None]:
    args = _trace_args(row)
    global_ranks = {
        int(value) for source in (row, args) for key in _TRACE_GLOBAL_RANK_KEYS
        if (value := source.get(key)) is not None and not isinstance(value, bool)
        and str(value).isascii() and str(value).isdecimal()
    }
    local_devices = {
        int(value) for source in (row, args) for key in _TRACE_LOCAL_DEVICE_KEYS
        if (value := source.get(key)) is not None and not isinstance(value, bool)
        and str(value).isascii() and str(value).isdecimal()
    }
    for source in (row, args):
        if isinstance(coord := source.get("rank_coord"), RankCoord):
            global_ranks.add(int(coord.global_rank))
            local_devices.add(int(coord.die_index))
    if len(global_ranks) > 1 or len(local_devices) > 1:
        raise ModelingError("capture_rank_mismatch:conflicting_trace_rank_fields")
    g_rank = GlobalRank(next(iter(global_ranks))) if global_ranks else None
    l_dev = DieIndex(next(iter(local_devices))) if local_devices else None
    return g_rank, l_dev


def _trace_rank(row: Mapping[str, Any]) -> int | None:
    g_rank, l_dev = _trace_coords(row)
    return int(g_rank) if g_rank is not None else (int(l_dev) if l_dev is not None else None)


def _trace_process_coords(row: Mapping[str, Any]) -> tuple[GlobalRank | None, DieIndex | None]:
    label = str(_trace_args(row).get("name", ""))
    row_rank, row_device = _trace_coords(row)
    pid = row.get("pid")
    track_rank = GlobalRank(int(m["rank"])) if (m := _TRACE_GLOBAL_RANK_LABEL.fullmatch(label)) else None
    track_device: DieIndex | None = None
    if device := _TRACE_LOCAL_DEVICE_LABEL.fullmatch(label):
        track_device = DieIndex(int(device["device"]))
    elif (label == "Ascend Hardware" and type(pid) is int and pid >= _ASCEND_HARDWARE_TRACK
          and pid & _ASCEND_TRACK_TYPE_MASK == _ASCEND_HARDWARE_TRACK):
        track_device = DieIndex(pid & _ASCEND_DEVICE_ID_MASK)
    if row_rank is not None and track_rank is not None and row_rank != track_rank:
        raise ModelingError("capture_rank_mismatch:conflicting_trace_process_rank_fields")
    if row_device is not None and track_device is not None and row_device != track_device:
        raise ModelingError("capture_rank_mismatch:conflicting_trace_process_rank_fields")
    return (row_rank if row_rank is not None else track_rank,
            row_device if row_device is not None else track_device)


def _record_trace_pid_rank(meta: dict[str, Any], pid: int | str, row: Mapping[str, Any]) -> None:
    args = _trace_args(row)
    g_ranks = {
        int(v) for src in (row, args) for k in _TRACE_GLOBAL_RANK_KEYS
        if (v := src.get(k)) is not None and not isinstance(v, bool)
        and str(v).isascii() and str(v).isdecimal()
    }
    for src in (row, args):
        if isinstance(coord := src.get("rank_coord"), RankCoord):
            g_ranks.add(int(coord.global_rank))
    if len(g_ranks) != 1:
        return
    r_val = next(iter(g_ranks))
    pid_ranks = meta.setdefault("_trace_pid_ranks", {})
    if pid not in pid_ranks:
        pid_ranks[pid] = r_val
    elif pid_ranks[pid] != r_val:
        pid_ranks[pid] = None


_TRACE_HOST_PROCESS_TOKENS = frozenset({"python", "python3", "pytorch", "cpu", "host", "cann", "ascendcl", "runtime"})


def _is_host_process_label(label: str) -> bool:
    low = label.strip().lower()
    if not low or low == "ascend hardware":
        return False
    first = low.split()[0]
    return low in _TRACE_HOST_PROCESS_TOKENS or first in _TRACE_HOST_PROCESS_TOKENS


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
            host_processes = meta.setdefault("_trace_host_processes", set())
            pid = row.get("pid")
            if row.get("ph") == "M" and row.get("name") == "process_name" and isinstance(pid, (int, str)):
                label = str(_trace_args(row).get("name", ""))
                if _is_host_process_label(label):
                    if pid in processes:
                        raise ModelingError("capture_rank_mismatch:conflicting_trace_process_ranks")
                    host_processes.add(pid)
                else:
                    coords = _trace_process_coords(row)
                    if coords[0] is not None or coords[1] is not None:
                        if pid in host_processes or processes.setdefault(pid, coords) != coords:
                            raise ModelingError("capture_rank_mismatch:conflicting_trace_process_ranks")
                    else:
                        if pid in processes:
                            raise ModelingError("capture_rank_mismatch:conflicting_trace_process_ranks")
                        host_processes.add(pid)
            elif row.get("ph") == "X" and isinstance(pid, (int, str)):
                _record_trace_pid_rank(meta, pid, row)
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


def _extract_capture_meta(
    path: Path, *, ijson_mod: Any = None, stream_jsonl_fn: Callable[[Path], Any],
    iter_json_events_fn: Callable[..., Any], read_value_fn: Callable[..., Any], skip_value_fn: Callable[..., Any],
    target_events: frozenset[str] | None = None,
    expected_clock_operator: str | None = None,
) -> dict[str, Any]:
    def rows() -> Iterator[tuple[bool, Mapping[str, Any]]]:
        if path.suffix.lower() in _JSONL_SUFFIXES:
            yield from _iter_jsonl_capture_rows(stream_jsonl_fn(path))
        else:
            with path.open("rb") as handle:
                events = iter_json_events_fn(handle, ijson_mod=ijson_mod)
                yield from _iter_capture_rows(events, next(events), read_value=read_value_fn, skip_value=skip_value_fn)

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
    path: Path, case_id: str, ranks: tuple[int, ...], seen_ranks: set[int],
    windows: _RankWindows | None, target_events: frozenset[str] | None,
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
        stream_cols = [fields[key] for key in ("streamid", "physicstreamid", "logicstreamid", "stream") if key in fields]
        call_cols = [fields[key] for key in ("callid", "hostcallid", "aclnncallid", "apiid") if key in fields]
        connection_cols = [fields["connectionid"]] if "connectionid" in fields else []
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
            if target_events is not None:
                names = target_events.intersection((f"row={record}", *(row[col].strip() for col in event_cols)))
                if not names:
                    continue
            start = _capture_time_ns(row, start_col) if start_col else None
            window = windows[rank] if windows is not None else None
            if start is None:
                if int(row[fields["count"]]) != 1 or total_col is None:
                    raise ModelingError(f"capture_csv_aggregate_event:{case_id}")
                end = _capture_time_ns(row, total_col)
            else:
                if window and start >= window[1]:
                    continue
                end = _capture_time_ns(row, end_col) if end_col else start + _capture_time_ns(row, duration_col)
                if duration_col and abs(end - start - _capture_time_ns(row, duration_col)) > _CAPTURE_ROUNDING_NS:
                    raise ModelingError(f"capture_csv_interval_inconsistent:{case_id}")
                if end < start:
                    raise ModelingError(f"capture_csv_nonpositive_duration:{case_id}")
                if window and end <= window[0]:
                    continue
            yield _CaptureRow(rank, record, names, start, end, next(_event_field_ids((row,), stream_cols), None),
                              _unique_event_id((row,), call_cols), _unique_event_id((row,), connection_cols))


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
        start, end = max(start, left), min(end, right)
        busy += max(0, end - max(start, cursor))
        cursor = max(cursor, end)
    return busy


def _event_field_ids(sources: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> Iterator[str]:
    return (str(value).strip() for source in sources for key in keys
            if type(value := source.get(key)) in (str, int) and str(value).strip())


def _unique_event_id(sources: Iterable[Mapping[str, Any]], keys: Iterable[str]) -> str | None:
    values = set(_event_field_ids(sources, keys))
    return values.pop() if len(values) == 1 else None


def _json_event_names(
    row: Mapping[str, Any], record: int, rank: int, target_events: frozenset[str],
    *, local_device: int | None = None,
) -> frozenset[str]:
    sources = (row, _trace_args(row))
    tasks = set(_event_field_ids(sources, _TASK_ID_KEYS))
    connections = set(_event_field_ids(sources, _CONNECTION_ID_KEYS))
    streams = set(_event_field_ids(sources, _STREAM_ID_KEYS)) or set(_event_field_ids((row,), ("tid",)))
    raw = set(_event_field_ids(sources, _EVENT_ID_KEYS)) | tasks | connections
    dev_ids = {local_device} if local_device is not None else {rank}
    qualified = {
        f"row:{record}", *(f"task:{task}" for task in tasks), *(f"conn:{conn}" for conn in connections),
        *(f"stream:{stream}:task:{task}" for stream in streams for task in tasks),
        *(f"dev:{dev}:stream:{stream}:task:{task}" for dev in dev_ids for stream in streams for task in tasks),
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
    case_id: str, boundary: Any, meta: Mapping[str, Any], trace_format: bool, seen_ranks: set[int],
    windows: _RankWindows | None, target_events: frozenset[str] | None,
) -> Iterator[_CaptureRow]:
    processes = meta.get("_trace_processes", {})
    host_processes = meta.get("_trace_host_processes", ())
    pid_ranks = meta.get("_trace_pid_ranks", {})
    header_ranks = meta.get("_header_ranks", ())
    default_rank = next(iter(header_ranks)) if len(header_ranks) == 1 and not trace_format else None
    source = (row for trace, row in meta["_capture_rows"]() if trace == trace_format)
    for record, row in enumerate(source, start=_FIRST_SOURCE_ROW):
        if trace_format and row.get("ph") != "X":
            continue
        pid = row.get("pid")
        if trace_format and (pid in host_processes
                             or str(row.get("cat") or "").strip().lower() in _TRACE_HOST_CATEGORIES):
            continue
        start_field = ("ts", _NS_PER_US) if trace_format else _json_kernel_start(row)
        if start_field is None:
            continue
        proc_rank, proc_device = processes.get(pid, (default_rank, None))
        row_rank, row_device = _trace_coords(row)
        if trace_format:
            if proc_rank is not None and row_rank is not None and row_rank != proc_rank:
                raise ModelingError(f"capture_rank_mismatch:{case_id}:trace_process_conflict")
            if proc_device is not None and row_device is not None and row_device != proc_device:
                raise ModelingError(f"capture_rank_mismatch:{case_id}:trace_process_conflict")
        mapped_pid_rank = pid_ranks.get(pid) if trace_format and pid in processes else None
        resolved_rank = (
            row_rank if row_rank is not None
            else proc_rank if proc_rank is not None
            else mapped_pid_rank if mapped_pid_rank is not None
            else row_device if row_device is not None
            else proc_device
        )
        rank = int(resolved_rank) if resolved_rank is not None else None
        local_dev = (int(row_device) if row_device is not None
                     else int(proc_device) if proc_device is not None else None)
        if rank is None and not trace_format:
            raise ModelingError(f"capture_rank_mismatch:{case_id}:timed_json_event_rank_absent")
        if rank not in boundary.device_ranks:
            continue
        seen_ranks.add(rank)
        names = frozenset()
        if target_events is not None:
            names = _json_event_names(row, record, rank, target_events, local_device=local_dev)
            if not names:
                continue
            for key in ("clock_operator", "clock_domain"):
                expected = getattr(boundary, key)
                if row.get(key, expected) != expected:
                    raise ModelingError(f"capture_{key}_mismatch:{case_id}:expected={expected}")
        start = _capture_time_ns(row, start_field)
        window = windows[rank] if windows is not None else None
        if window and start >= window[1]:
            continue
        end_field = None if trace_format else _json_time_field(row, _JSON_END_KEYS)
        dur_field = ("dur", _NS_PER_US) if trace_format else _json_time_field(row, _JSON_DURATION_KEYS)
        end = _capture_time_ns(row, end_field) if end_field else start + _capture_time_ns(row, dur_field)
        if end_field and dur_field and abs(end - start - _capture_time_ns(row, dur_field)) > _CAPTURE_ROUNDING_NS:
            raise ModelingError(f"capture_json_interval_inconsistent:{case_id}")
        if end < start:
            reason = "invalid_interval" if trace_format and windows is None else "nonpositive_duration"
            raise ModelingError(f"capture_{'trace' if trace_format else 'json'}_{reason}:{case_id}")
        if window and end <= window[0]:
            continue
        sources = (row, _trace_args(row))
        stream = next(_event_field_ids(sources, _STREAM_ID_KEYS), None) or _unique_event_id((row,), ("tid",))
        yield _CaptureRow(rank, record, names, start, end, stream, _unique_event_id(sources, _CALL_ID_KEYS),
                          _unique_event_id(sources, _CONNECTION_ID_KEYS))


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


def _host_trace_rows(meta: Mapping[str, Any]) -> Iterator[tuple[_HostThread, Mapping[str, Any]]]:
    host_processes = meta.get("_trace_host_processes", ())
    for trace, row in meta["_capture_rows"]():
        if not trace or row.get("ph") not in ("X", "s"):
            continue
        pid, tid = row.get("pid"), row.get("tid")
        if (type(pid) in (int, str) and type(tid) in (int, str)
                and pid not in meta["_trace_processes"]
                and (pid in host_processes or _trace_rank(row) is None)):
            yield (pid, tid), row


def _host_trace_interval(row: Mapping[str, Any]) -> tuple[Decimal, Decimal]:
    start = _capture_time_ns(row, ("ts", _NS_PER_US))
    return start, start + _capture_time_ns(row, ("dur", _NS_PER_US)) if row["ph"] == "X" else start


def _is_host_api_name(raw_name: Any) -> bool:
    stem = str(raw_name or "").strip().rsplit("@", 1)[-1]
    return stem.startswith(_HOST_API_PREFIXES)


def _host_launch_id(row: Mapping[str, Any]) -> str | None:
    sources = row, _trace_args(row)
    conn = _unique_event_id(sources, _CONNECTION_ID_KEYS)
    return _unique_event_id(sources, ("id",)) if conn is None and row["ph"] == "s" else conn


def _trace_call_connections(meta: Mapping[str, Any], connections: set[str]) -> set[str] | None:
    candidates: dict[tuple[str, str], _HostLaunch | None] = {}
    has_launches = False
    for track, row in _host_trace_rows(meta):
        if (conn := _host_launch_id(row)) is not None:
            has_launches = True
            if conn in connections:
                key, launch = (row["ph"], conn), (track, *_host_trace_interval(row))
                if candidates.setdefault(key, launch) != launch:
                    candidates[key] = None
    single_connection = connections if len(connections) == 1 else None
    if not has_launches:
        return single_connection
    launches = [launch for conn in connections
                if (launch := candidates.get(("X", conn), candidates.get(("s", conn)))) is not None]
    if len(launches) != len(connections) or len(threads := {track for track, _, _ in launches}) != 1:
        return None
    if any(end < start for _, start, end in launches):
        return None
    thread = threads.pop()
    left, right = min(start for _, start, _ in launches), max(end for _, _, end in launches)
    owner = shortest = None
    for track, row in _host_trace_rows(meta):
        if track != thread or row["ph"] != "X" or not _is_host_api_name(row.get("name")) or _host_launch_id(row) in connections:
            continue
        start, end = _host_trace_interval(row)
        if start <= left and right <= end:
            duration = end - start
            if shortest is None or duration < shortest:
                owner, shortest = (start, end), duration
            elif duration == shortest:
                owner = None
        elif start < end and any(start <= begin <= finish <= end for _, begin, finish in launches):
            return None  # A proper subset belongs to another API span.
    if owner is None:
        return single_connection if shortest is None else None
    spans, flows = set(), set()
    left, right = owner
    for track, row in _host_trace_rows(meta):
        if track == thread and (conn := _host_launch_id(row)) is not None:
            start, end = _host_trace_interval(row)
            if left <= start <= end <= right:
                (spans if row["ph"] == "X" else flows).add(conn)
    flows -= spans
    if flows:  # A launch span takes precedence even when its flow starts inside the owner.
        for _, row in _host_trace_rows(meta):
            if row["ph"] == "X":
                flows.discard(_host_launch_id(row))
    return spans | flows


def _verify_l1_ownership(
    case_id: str, selected: Mapping[int, list[_CaptureRow]], occurrences: frozenset[_KernelOccurrence],
    all_rows: Iterable[_CaptureRow], trace_meta: Mapping[str, Any] | None,
) -> None:
    owners: dict[int, tuple[set[str], set[str]]] = {}
    for rank, rows in selected.items():
        if any(not (row.call_id or row.connection_id) for row in rows):
            raise ModelingError(f"capture_l1_call_ownership_unverified:{case_id}:rank={rank}")
        calls = {row.call_id for row in rows if row.call_id}
        connections = {row.connection_id for row in rows if row.connection_id}
        if len(calls) > 1:
            raise ModelingError(f"capture_l1_call_ownership_unverified:{case_id}:rank={rank}")
        if calls:
            call_conns = {row.connection_id for row in rows if row.call_id and row.connection_id}
            trace_conns = (_trace_call_connections(trace_meta, connections)
                           if trace_meta is not None and connections else None)
            owned_conns = (trace_conns | connections) if trace_conns is not None else connections
            reachable = (trace_conns | call_conns) if trace_conns is not None else call_conns
            if not all(row.call_id in calls or row.connection_id in reachable for row in rows):
                raise ModelingError(f"capture_l1_call_ownership_unverified:{case_id}:rank={rank}")
            owners[rank] = calls, owned_conns
        else:
            owned_conns = (_trace_call_connections(trace_meta, connections) if trace_meta is not None
                           else connections if len(connections) == 1 else None)
            if owned_conns is None or not (connections <= owned_conns):
                raise ModelingError(f"capture_l1_call_ownership_unverified:{case_id}:rank={rank}")
            owners[rank] = set(), owned_conns
    for row in all_rows:
        if row.rank in owners and row.occurrence not in occurrences and row.end > (row.start or 0):
            owned_calls, owned_conns = owners[row.rank]
            if (row.call_id and row.call_id in owned_calls) or (row.connection_id and row.connection_id in owned_conns):
                raise ModelingError(f"capture_l1_call_ownership_unverified:{case_id}:rank={row.rank}")


def _verify_paired_rung_capture(
    case_id: str, boundary: Any, format_name: str,
    selected: Mapping[int, list[_CaptureRow]], occurrences: frozenset[_KernelOccurrence],
    rows: Callable[[_RankWindows | None, frozenset[str] | None], Iterator[_CaptureRow]],
    *, arm_a_measured_ns: int | None, arm_b_measured_ns: int | None,
) -> tuple[int, CaptureVerificationReceipt]:
    start_name, end_name = boundary.start_event.strip(), boundary.end_event.strip()
    if start_name == end_name:
        raise ModelingError(f"capture_paired_rung_arms_ambiguous:{case_id}")
    arm_pairs: dict[int, tuple[_CaptureRow, _CaptureRow]] = {}
    for rank, local_rows in selected.items():
        row_a = next(row for row in local_rows if start_name in row.names)
        row_b = next(row for row in local_rows if end_name in row.names)
        if row_a.occurrence == row_b.occurrence:
            raise ModelingError(f"capture_paired_rung_arms_ambiguous:{case_id}:rank={rank}")
        arm_pairs[rank] = (row_a, row_b)
    operator = boundary.clock_operator
    if any(row.start is None for local_rows in selected.values() for row in local_rows):
        actual_a = round(max(row_a.end for row_a, _ in arm_pairs.values()))
        actual_b = round(max(row_b.end for _, row_b in arm_pairs.values()))
    else:
        win_a = {rank: (row_a.start, row_a.end) for rank, (row_a, _) in arm_pairs.items()}
        win_b = {rank: (row_b.start, row_b.end) for rank, (_, row_b) in arm_pairs.items()}
        spans_a: dict[int, list[tuple[Decimal, Decimal]]] = {r: [] for r in win_a}
        spans_b: dict[int, list[tuple[Decimal, Decimal]]] = {r: [] for r in win_b}
        if operator == "busy_union":
            for r_row in rows(win_a, None):
                if r_row.end > r_row.start:
                    spans_a[r_row.rank].append((r_row.start, r_row.end))
            for r_row in rows(win_b, None):
                if r_row.end > r_row.start:
                    spans_b[r_row.rank].append((r_row.start, r_row.end))
        actual_a = round(max(_timed_rank_duration(case_id, format_name, operator, left, right, spans_a[r])
                             for r, (left, right) in win_a.items()))
        actual_b = round(max(_timed_rank_duration(case_id, format_name, operator, left, right, spans_b[r])
                             for r, (left, right) in win_b.items()))
    if arm_a_measured_ns is not None and abs(actual_a - arm_a_measured_ns) > _CAPTURE_ROUNDING_NS:
        raise ModelingError(f"capture_duration_mismatch:{case_id}:arm_a:declared={arm_a_measured_ns}:actual={actual_a}")
    if arm_b_measured_ns is not None and abs(actual_b - arm_b_measured_ns) > _CAPTURE_ROUNDING_NS:
        raise ModelingError(f"capture_duration_mismatch:{case_id}:arm_b:declared={arm_b_measured_ns}:actual={actual_b}")
    return actual_b - actual_a, CaptureVerificationReceipt(occurrences)


def _verify_timed_capture(
    case_id: str, boundary: Any, format_name: str, target_events: frozenset[str],
    rows: Callable[[_RankWindows | None, frozenset[str] | None], Iterator[_CaptureRow]], seen_ranks: set[int],
    trace_meta: Mapping[str, Any] | None = None, *,
    arm_a_measured_ns: int | None = None, arm_b_measured_ns: int | None = None,
) -> tuple[int, CaptureVerificationReceipt]:
    if not boundary.device_ranks:
        raise ModelingError(f"capture_rank_mismatch:{case_id}:requested_ranks_absent")
    start_name, end_name = boundary.start_event.strip(), boundary.end_event.strip()
    kind = getattr(boundary, "boundary_kind", "")
    single_kernel, step = kind == "device_kernel", kind in L3_BOUNDARY_KINDS
    composite = kind in L1_COMPOSITE_BOUNDARY_KINDS
    selected = _selected_rows(case_id, format_name, boundary, target_events, rows(None, target_events), seen_ranks)
    occurrences = frozenset(row.occurrence for local_rows in selected.values() for row in local_rows)
    if kind == "paired_rung_delta":
        return _verify_paired_rung_capture(
            case_id, boundary, format_name, selected, occurrences, rows,
            arm_a_measured_ns=arm_a_measured_ns, arm_b_measured_ns=arm_b_measured_ns,
        )
    if single_kernel:
        for rank, local_rows in selected.items():
            if (count := len(local_rows)) != 1:
                raise ModelingError(f"capture_device_kernel_requires_one_occurrence:{case_id}:rank={rank}:count={count}")
        if start_name != end_name:
            raise ModelingError(f"capture_device_kernel_endpoint_mismatch:{case_id}")
    if composite:
        _verify_l1_ownership(case_id, selected, occurrences, rows(None, None), trace_meta)
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
    scan_busy = operator == "busy_union" and not (single_kernel or composite)
    census: set[_KernelOccurrence] = set()
    spans: dict[int, list[tuple[Decimal, Decimal]]] = {rank: [] for rank in windows}
    if single_kernel or step or scan_busy:
        for row in rows(windows, None):
            if row.end == row.start:
                continue
            left, right = windows[row.rank]
            launched_inside = left <= row.start < right
            if single_kernel and row.occurrence not in occurrences:
                selected_stream = selected[row.rank][0].stream
                independent = selected_stream is not None and row.stream is not None and row.stream != selected_stream
                overlap = min(row.end, right) - max(row.start, left)
                if not independent and (launched_inside or overlap > _CAPTURE_ROUNDING_NS):
                    raise ModelingError(f"capture_device_kernel_multiple_occurrences:{case_id}:rank={row.rank}")
            if step and (launched_inside or operator == "busy_union" and row.end - left > _CAPTURE_ROUNDING_NS):
                census.add(row.occurrence)
            if scan_busy:
                spans[row.rank].append((row.start, row.end))
    if not scan_busy and operator == "busy_union":
        spans = {rank: [(row.start, row.end) for row in local_rows] for rank, local_rows in selected.items()}
    duration = round(max(_timed_rank_duration(case_id, format_name, operator, left, right, spans[rank])
                         for rank, (left, right) in windows.items()))
    return duration, CaptureVerificationReceipt(occurrences, frozenset(census) if step else None)


def verify_capture_file_consistency(
    case_id: str, boundary: Any, path: Path, *, ijson_mod: Any = None,
    digest_fn: Callable[[Path], str],
    stream_chunk_bytes: int,
    extract_meta_fn: Callable[..., dict[str, Any]],
    measured_ns: int | None = None,
    arm_a_measured_ns: int | None = None,
    arm_b_measured_ns: int | None = None,
) -> CaptureVerificationReceipt | None:
    """Bind one-based source rows and field-qualified events to a raw capture.

    For multiple ranks, the longest verified local window represents the capture duration.
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
    receipt, rows, meta = None, None, {}
    format_name = "csv"
    seen_ranks: set[int] = set()
    try:
        if path.name.lower().endswith((".csv", ".csv.gz")):
            rows = partial(_csv_capture_rows, path, case_id, boundary.device_ranks, seen_ranks)
        else:
            if path.suffix.lower() not in _JSON_SUFFIXES:
                with path.open("rb") as handle:
                    if not handle.read(stream_chunk_bytes).lstrip().startswith((b"{", b"[")):
                        raise ModelingError(f"capture_format_unsupported:{case_id}")
            meta = extract_meta_fn(path, ijson_mod=ijson_mod, target_events=target_events,
                                   expected_clock_operator=boundary.clock_operator)
            trace_format = "_trace_processes" in meta
            format_name = "trace" if trace_format else "json"
            if "_capture_rows" in meta:
                rows = partial(_json_capture_rows, case_id, boundary, meta, trace_format, seen_ranks)
        if rows is not None:
            duration, receipt = _verify_timed_capture(
                case_id, boundary, format_name, target_events, rows, seen_ranks,
                trace_meta=meta if format_name == "trace" else None,
                arm_a_measured_ns=arm_a_measured_ns, arm_b_measured_ns=arm_b_measured_ns,
            )
            if measured_ns is None or abs(duration - measured_ns) > _CAPTURE_ROUNDING_NS:
                raise ModelingError(f"capture_duration_mismatch:{case_id}:declared={measured_ns}:actual={duration}")
            if format_name == "csv":
                return receipt
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
    cap_events, matched = set(meta["event_ids"]), set()
    endpoints = {"start_event": start_name, "end_event": end_name}
    for container in (meta, meta.get("boundary"), meta.get("metadata")):
        if not isinstance(container, Mapping):
            continue
        if "clock_operator" in container and container["clock_operator"] != boundary.clock_operator:
            raise ModelingError(f"capture_clock_operator_mismatch:{case_id}:expected={boundary.clock_operator}")
        if (cap_clock := container.get("clock_domain")) is not None and str(cap_clock).strip() != str(boundary.clock_domain).strip():
            raise ModelingError(f"capture_clock_domain_mismatch:{case_id}:expected={boundary.clock_domain}:actual={cap_clock}")
        for key, expected in endpoints.items():
            if (declared := container.get(key)) is not None:
                if str(declared).strip() != expected:
                    raise ModelingError(f"capture_event_mismatch:{case_id}:{key}")
                matched.add(key)
    for key, expected in endpoints.items():
        if key not in matched and expected not in cap_events:
            raise ModelingError(f"capture_event_mismatch:{case_id}:{key}_absent:{expected}")
    known = cap_events | {endpoints[key] for key in matched}
    if missing := [ev for ev in boundary.event_ids if ev.strip() not in known]:
        raise ModelingError(f"capture_event_mismatch:{case_id}:missing={','.join(missing)}")
    if receipt is None:
        kind = getattr(boundary, "boundary_kind", "")
        if kind == "device_kernel":
            raise ModelingError(f"capture_device_kernel_occurrence_unavailable:{case_id}")
        if kind == "paired_rung_delta":
            raise ModelingError(f"capture_paired_rung_rows_unavailable:{case_id}")
        if kind in L1_COMPOSITE_BOUNDARY_KINDS:
            raise ModelingError(f"capture_l1_call_ownership_unverified:{case_id}:no_device_rows")
    return receipt


import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

try:
    import ijson
except ImportError:  # pragma: no cover - stdlib fallback when ijson is absent
    ijson = None


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_prefix(value, parts):
    if not parts:
        yield value
        return
    head, *tail = parts
    if head == "item":
        if isinstance(value, list):
            for item in value:
                yield from _iter_prefix(item, tail)
    elif isinstance(value, dict) and head in value:
        yield from _iter_prefix(value[head], tail)


def json_items(path, prefix=""):
    if ijson is not None:
        with Path(path).open("rb") as stream:
            yield from ijson.items(stream, prefix, use_float=True)
        return
    with Path(path).open("rb") as stream:
        doc = json.load(stream)
    parts = [part for part in prefix.split(".") if part] if prefix else []
    yield from _iter_prefix(doc, parts)


def json_kvitems(path, prefix=""):
    if ijson is not None:
        with Path(path).open("rb") as stream:
            yield from ijson.kvitems(stream, prefix, use_float=True)
        return
    parts = [part for part in prefix.split(".") if part] if prefix else []
    for item in json_items(path, ".".join(parts)):
        if isinstance(item, dict):
            yield from item.items()


def json_top_fields(path, wanted):
    if ijson is not None:
        from ijson.common import ObjectBuilder

        result, builder, selected, depth = {}, None, None, 0
        with Path(path).open("rb") as stream:
            for prefix, event, value in ijson.parse(stream, use_float=True):
                if builder is None:
                    if prefix not in wanted:
                        continue
                    if event not in {"start_map", "start_array"}:
                        result[prefix] = value
                        continue
                    selected, builder, depth = prefix, ObjectBuilder(), 0
                builder.event(event, value)
                if event in {"start_map", "start_array"}:
                    depth += 1
                elif event in {"end_map", "end_array"}:
                    depth -= 1
                    if not depth:
                        result[selected], builder = builder.value, None
        return result
    with Path(path).open("rb") as stream:
        doc = json.load(stream)
    if not isinstance(doc, dict):
        return {}
    return {key: doc[key] for key in wanted if key in doc}


def profiler_rows(path):
    from .captures import kernel_rows

    for row in kernel_rows(path):
        event = row.event
        yield {
            "line": row.line, "duration_ms": row.duration_ns / 1e6,
            "kernel": row.name, "operator": row.kind,
            "input_shapes": event.input_shapes.strip('"'),
            "input_dtypes": event.input_dtypes, "input_formats": event.input_formats,
            "output_shapes": event.output_shapes.strip('"'),
            "output_dtypes": event.output_dtypes, "device": event.device_id,
            "stream": row.stream, "core": row.core,
            "block_dim": "" if row.blocks is None else str(row.blocks),
            "boundary": "profiled_device_kernel",
        }


def source_kind(path):
    path = Path(path)
    # Source role is a discovery aid. Neither this classification nor a
    # filename ever authorizes applying a timing to an operator.
    if path.name.endswith(".heldout.json"):
        return "validation"
    if path.suffix == ".csv":
        with path.open("rb") as stream:
            header = stream.readline(16384)
        if b"Duration(us)" in header or b"Task Duration(us)" in header:
            return "profiler"
    if ("mindie-sim-data/sim/" in str(path) or "predict" in path.name
            or "true128k-results-full" in str(path)):
        return "prediction"
    if path.suffix in {".json", ".jsonl", ".ndjson"}:
        return "structured_source"
    if path.suffix in {".sqlite", ".db"}:
        return "database"
    return "supporting_source"


class ObservationIndex:
    def __init__(self, path):
        self.connection = sqlite3.connect(path, timeout=5)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS sources(
              path TEXT PRIMARY KEY, bytes INTEGER, mtime_ns INTEGER,
              sha256 TEXT, kind TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS catalogues(
              path TEXT PRIMARY KEY, kind TEXT NOT NULL, schema_json TEXT NOT NULL);
        """)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def register(self, path, kind=None):
        path = Path(path).resolve()
        stat = path.stat()
        previous = self.connection.execute(
            "SELECT bytes,mtime_ns,sha256,kind FROM sources WHERE path=?", (str(path),)).fetchone()
        role = kind or source_kind(path)
        if previous and previous[:2] == (stat.st_size, stat.st_mtime_ns) and previous[3] == role:
            return
        self.connection.execute("INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?)",
                                (str(path), stat.st_size, stat.st_mtime_ns, digest_file(path), role))

    def scan(self, root):
        count = 0
        for directory, names, files in os.walk(root):
            names[:] = sorted(name for name in names if name not in {".git", ".venv", "__pycache__"})
            for name in sorted(files):
                path = Path(directory, name)
                if path.is_file() and path.suffix not in {".pyc", ".pid", ".cpid", ".stop"}:
                    self.register(path)
                    count += 1
                    if count % 200 == 0:
                        self.connection.commit()
        self.connection.commit()
        return self.inventory()

    def attach_profiler_catalogue(self, path):
        path = Path(path).resolve()
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"sources", "variants", "events", "costs"} <= names:
                raise ValueError("external_profiler_schema_not_supported")
            schema = {row[0]: row[1] for row in db.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='table'")}
        self.register(path, "profiler_catalogue")
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO catalogues VALUES(?,?,?)",
                                    (str(path), "profiler_variants/1", json.dumps(schema, sort_keys=True)))

    def inventory(self):
        return {kind: {"files": count, "bytes": size}
                for kind, count, size in self.connection.execute(
                    "SELECT kind,COUNT(*),SUM(bytes) FROM sources GROUP BY kind")}

    def find_sources(self, pattern, limit=50):
        return [dict(zip(("path", "bytes", "sha256", "kind"), row)) for row in
                self.connection.execute("SELECT path,bytes,sha256,kind FROM sources WHERE path LIKE ? LIMIT ?",
                                        ("%" + pattern + "%", min(limit, 1000)))]


class ProfilerCatalogue:
    """Read the retained 2026 capture database with no dependency on old code."""

    def __init__(self, path):
        self.connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
        self.connection.execute("PRAGMA busy_timeout=5000")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def variants(self, *, limit=50, after=None):
        sql = """SELECT v.id,v.signature,v.context,v.capture,c.calls,c.median_ns,c.min_ns,c.max_ns
                 FROM variants v JOIN costs c ON c.variant=v.id WHERE v.id>? ORDER BY v.id LIMIT ?"""
        cursor = self.connection.execute(sql, (bytes.fromhex(after) if after else b"", min(limit, 1000)))
        for key, signature, context, capture, calls, median, minimum, maximum in cursor:
            yield {"id": key.hex(), "signature": json.loads(signature), "context": json.loads(context),
                   "capture": capture, "calls": calls, "median_ms": median / 1e6,
                   "min_ms": minimum / 1e6, "max_ms": maximum / 1e6}

    def samples(self, variant, limit=1000, *, query_seconds=5):
        # Old events has no variant index. Read through a bounded cursor rather
        # than building an in-memory reverse index or modifying the raw store.
        deadline = time.monotonic() + query_seconds
        self.connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        try:
            cursor = self.connection.execute(
                "SELECT e.duration_ns,s.path,s.member,e.row_number FROM events e "
                "JOIN sources s ON s.id=e.source WHERE e.variant=? LIMIT ?",
                (bytes.fromhex(variant), min(limit, 10000)))
            yield from ({"duration_ms": duration / 1e6, "source": path, "member": member, "row": row}
                        for duration, path, member, row in cursor)
        finally:
            self.connection.set_progress_handler(None, 0)


from collections import Counter
from itertools import groupby
import json
import math
from pathlib import Path



def _checked(root, reference):
    path = Path(root) / reference["path"]
    if digest_file(path) != reference["sha256"]:
        raise ValueError(f"capture_source_hash_mismatch:{path}")
    return path


def _fields(path, wanted):
    return json_top_fields(path, wanted)


def _identity(manifest, header, *, level, boundary, implementation, phase,
              shape, dtype, layout, execution):
    return {
        "level": level, "boundary": boundary,
        "hardware": {**manifest["hardware"], "device": header["device"]},
        "implementation": {
            "producer_sha256": manifest["producer"]["sha256"],
            "software": {key: header[key] for key in ("torch", "torch_npu", "cann") if key in header},
            **implementation,
        },
        "phase": phase, "shape": shape, "dtype": dtype, "layout": layout,
        "execution": execution,
    }


def _condition(capture, path, pointer, identity, metadata, durations, *,
               pointers=None, samples_per_trial=None, status="measured"):
    samples = []
    for index, duration in enumerate(durations):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"capture_event_duration_invalid:{path}#{pointer}:{index}")
        location = pointers[index] if pointers else f"{pointer}/{index}"
        sample_metadata = {
            **metadata, "condition_id": f"{capture['sha256']}:{pointer}", "repeat": index,
        }
        if samples_per_trial:
            sample_metadata.update(trial=index // samples_per_trial,
                                   repeat=index % samples_per_trial)
        samples.append({
            "sample_id": f"{capture['sha256']}:{location}",
            "identity": identity, "duration_ms": duration,
            "run_id": capture["sha256"], "split": "unassigned",
            "source": f"{path}#{location}", "metadata": sample_metadata,
        })
    return {
        "condition_id": f"{capture['sha256']}:{pointer}",
        "source": f"{path}#{pointer}", "identity": identity, "status": status,
        "metadata": metadata, "samples": samples,
    }


def _activation(manifest, capture, path):
    header = _fields(path, {"schema", "device", "visible", "torch", "torch_npu", "canaries"})
    if header["schema"] != "a3-activation-measure/1":
        raise ValueError("activation_capture_schema_mismatch")
    for key, row in json_kvitems(path, "cells"):
        if True:
            variant = manifest["variants"][row["variant"]]
            prefix = "/cells/" + key.replace("~", "~0").replace("/", "~1")
            metadata = {
                "condition_label": key, "variant": row["variant"],
                "context": {name: row[name] for name in ("hidden", "batch", "seq", "scope")},
                "context_role": "producer_labels_not_additional_tensor_dimensions",
                "canary": header["canaries"].get(row["variant"]),
                "canary_drift_pct": row.get("canary_drift_pct"),
                "source_status": row["status"],
                "training_use": "not_assigned_for_current_prediction; historical_fit_reuse_not_excluded",
                "limitations": manifest["limitations"],
            }
            base = dict(
                level=variant["level"], implementation={"api": variant["api"], "variant": row["variant"]},
                phase="decode" if row["scope"] == "decode_step" else "prefill",
                shape={"tokens": row["tokens"], "intermediate": row["intermediate"]},
                dtype=row["dtype"], layout=variant["layout"],
            )
            execution = {
                "mode": "eager", "timer": "npu_events", "invocations": 1,
                "input_values": "zeros", "output_lifetime": "discard_after_call",
                "warmup_calls": 10, "timer_interleaving": "event_call_event",
            }
            identity = _identity(manifest, header, boundary="npu_event_call", execution=execution, **base)
            if row["status"] != "measured":
                yield _condition(capture, path, prefix, identity,
                                 {**metadata, "error": row.get("error")}, (), status=row["status"])
                continue
            passes = row["per_pass_ms"]
            samples = [value for values in passes for value in values]
            if samples != row["samples_ms"] or len(samples) != row["n"]:
                raise ValueError(f"activation_trial_samples_conflict:{key}")
            pointers = [f"{prefix}/per_pass_ms/{trial}/{repeat}"
                        for trial, values in enumerate(passes) for repeat in range(len(values))]
            yield _condition(capture, path, prefix + "/per_pass_ms", identity, metadata,
                             samples, pointers=pointers, samples_per_trial=len(passes[0]))
            count = row["burst_ops"]
            if len(row["burst_samples_ms"]) != row["burst_reps"]:
                raise ValueError(f"activation_burst_samples_conflict:{key}")
            burst_identity = _identity(
                manifest, header, boundary="npu_event_burst",
                execution={**execution, "invocations": count, "warmup_calls": 3,
                           "timer_interleaving": "event_calls_event"}, **{**base, "level": "subgraph"})
            yield _condition(
                capture, path, prefix + "/burst_samples_ms", burst_identity,
                {**metadata, "recorded_units": "batch_elapsed_ms_divided_by_invocations",
                 "duration_conversion": "recorded_per_call_ms_times_burst_ops",
                 "recorded_rounding_ms": 0.000001},
                [value * count for value in row["burst_samples_ms"]])


_GROUPED_PROBE_SCHEMA_RE = re.compile(r"^(?:[a-z0-9]+-)*grouped-graph-probe/1$")


def _grouped(manifest, capture, path):
    header = _fields(path, {"schema", "device", "torch", "torch_npu", "cann", "timing"})
    if not _GROUPED_PROBE_SCHEMA_RE.fullmatch(str(header.get("schema") or "")):
        raise ValueError("grouped_capture_schema_mismatch")
    for index, row in enumerate(json_items(path, "rows.item")):
        prefix = f"/rows/{index}"
        metadata = {
            "condition_label": row["label"], "projection_label": row.get("projection"),
            "canary": row.get("canary"), "source_status": row["status"],
            "training_use": "not_assigned_for_current_prediction",
            "limitations": manifest["limitations"],
        }
        if row["status"] != "measured":
            yield _condition(capture, path, prefix, {},
                             {**metadata, "error": row.get("error"), "shape": row.get("shape")},
                             (), status=row["status"])
            continue
        shape = row["shape"]
        counts = [shape["tokens_per_expert"]] * shape["groups"]
        counts += [0] * (shape["experts"] - shape["groups"])
        if sum(counts) != shape["tokens"]:
            raise ValueError("grouped_producer_histogram_conflict")
        contract = row["contract"]
        def _contract_dtype(key: str, default: str) -> str:
            raw_val = str(contract.get(key) or "").strip()
            return raw_val.split()[0] if raw_val else default
        base = dict(
            implementation={"api": contract["api"], "contract": contract},
            phase="primitive",
            shape={"tokens": shape["tokens"], "k": shape["k"], "n": shape["n"],
                   "experts": shape["experts"], "expert_counts": counts},
            dtype={
                "input": _contract_dtype("x", "int8"),
                "weight": _contract_dtype("weight", "int8"),
                "scale": _contract_dtype("scale", "float32"),
                "per_token_scale": _contract_dtype("per_token_scale", "float32"),
                "output": _contract_dtype("out_dtype", "bfloat16"),
            },
            layout={"input": "ND", "weight": "FRACTAL_NZ", "output": "ND"},
        )
        for field, mode, invocations in (
            ("eager_per_call", "eager", 1), ("graph_replay_1call", "graph", 1),
            ("graph_replay_8calls", "graph", 8),
        ):
            execution = {
                "mode": mode, "timer": "npu_events", "invocations": invocations,
                "warmup_calls": header["timing"]["warmup"], "activation_quant": "outside_timed_region",
                "output_lifetime": "captured_static_buffers" if mode == "graph" else "discard_after_call",
                "capture_warmup_calls": 3 if mode == "graph" else 0,
                "dependency": "serial",
            }
            identity = _identity(
                manifest, header, level="subgraph" if mode == "graph" else "operator",
                boundary="npu_event_graph_replay" if mode == "graph" else "npu_event_call",
                execution=execution, **base)
            timing = row.get(field)
            if timing is None:
                yield _condition(capture, path, prefix + "/" + field, identity,
                                 {**metadata, "error": row.get("graph_error")},
                                 (), status="unavailable")
                continue
            if (len(timing["raw_samples_ms"]) != timing["sample_count"]
                    or timing["sample_count"] != header["timing"]["trials"] * header["timing"]["samples_per_trial"]):
                raise ValueError("grouped_trial_samples_conflict")
            yield _condition(capture, path, prefix + "/" + field + "/raw_samples_ms", identity,
                             metadata, timing["raw_samples_ms"],
                             samples_per_trial=header["timing"]["samples_per_trial"])


def _csv_rows(path):
    previous = None
    # A marker defines a boundary even when it carries no measured duration.
    for row in kernel_rows(path, include_zero_duration=True):
        if previous is not None and row.start_ns < previous:
            raise ValueError(f"capture_csv_start_times_not_monotone:{path}:{row.line}")
        previous = row.start_ns
        yield row


def _ledger():
    return {"duration_ns": 0, "calls": Counter(), "source_preview": [], "source_count": 0,
            "contaminated": False, "devices": set()}


def _timestamp_group(manifest, rows):
    ledger, markers = _ledger(), 0
    for row in rows:
        name = row.name
        if name == manifest["marker"]:
            markers += 1
            continue
        grouped = name in manifest["kernels"]
        if not grouped and name != manifest["attention_kernel"]:
            continue
        if not row.input_shapes or not row.input_shapes[0]:
            raise ValueError(f"capture_tensor_shape_missing:{row.source}:{row.line}")
        tokens = row.input_shapes[0][0]
        limit = manifest["max_grouped_tokens"] if grouped else manifest["max_attention_query_tokens"]
        if tokens > limit:
            ledger["contaminated"] = True
        if not grouped:
            continue
        event = row.event
        signature = {
            "kernel": name, "input_shapes": event.input_shapes.strip('"'),
            "input_dtypes": event.input_dtypes, "input_formats": event.input_formats,
            "output_shapes": event.output_shapes.strip('"'),
            "output_dtypes": event.output_dtypes, "output_formats": event.output_formats,
        }
        ledger["calls"][json.dumps(signature, sort_keys=True)] += 1
        ledger["duration_ns"] += row.duration_ns
        if len(ledger["source_preview"]) < 5:
            ledger["source_preview"].append(row.line)
        ledger["source_count"] += 1
        ledger["devices"].add(event.device_id)
    if markers > 1:
        raise ValueError("served_grouped_duplicate_step_marker_timestamp")
    return ledger, markers


def _served_condition(manifest, capture, path, ledger, index, begin, end):
    identity = {
        "level": "subgraph", "boundary": "profiled_kernel_sum",
        "hardware": {**manifest["hardware"], "profile_device_ids": sorted(ledger["devices"])},
        "implementation": {
            "kernels": manifest["kernels"], "runtime": manifest["runtime"],
            "producer_sha256": manifest["producer"]["sha256"],
        },
        "phase": "decode",
        "shape": {"calls": [dict(json.loads(signature), occurrences=count)
                           for signature, count in sorted(ledger["calls"].items())]},
        "dtype": "per_tensor_in_call_descriptors", "layout": "per_tensor_in_call_descriptors",
        "execution": {"mode": "FULL_DECODE_ONLY", "aggregation": "sum_of_selected_kernel_durations",
                      "step_marker": manifest["marker"]},
    }
    metadata = {
        "condition_label": capture["label"], "step": index, "start_ns": begin, "end_ns": end,
        "source_line_preview": ledger["source_preview"], "source_line_count": ledger["source_count"],
        "training_use": "not_assigned_for_current_prediction",
        "unresolved_semantics": ["group_list_tensor_values", "loaded_kernel_binary_hash"],
        "limitations": manifest["limitations"],
        "selection": "exclude_intervals_with_prefill_sized_grouped_or_attention_calls",
    }
    return _condition(
        capture, path, f"/intervals/{index}", identity, metadata,
        () if ledger["contaminated"] else (ledger["duration_ns"] / 1e6,),
        status="prefill_contaminated" if ledger["contaminated"] else "measured")


def _served_grouped(manifest, capture, path):
    """Keep one active interval; equal-start kernels belong after that marker."""
    current, begin, markers = _ledger(), None, 0
    for start, rows in groupby(_csv_rows(path), key=lambda row: row.start_ns):
        pending, marker = _timestamp_group(manifest, rows)
        if marker:
            if begin is not None and current["calls"]:
                yield _served_condition(manifest, capture, path, current, markers - 1, begin, start)
            begin, current, markers = start, _ledger(), markers + 1
        if begin is not None:
            current["calls"].update(pending["calls"])
            current["duration_ns"] += pending["duration_ns"]
            current["devices"].update(pending["devices"])
            current["contaminated"] |= pending["contaminated"]
            current["source_preview"].extend(pending["source_preview"][:5 - len(current["source_preview"])])
            current["source_count"] += pending["source_count"]
    if markers < 2:
        raise ValueError("served_grouped_step_markers_missing")


ADAPTERS = {
    "activation_calls": _activation,
    "grouped_calls": _grouped,
    "served_grouped_kernels": _served_grouped,
}


def iter_conditions(manifest, *, root="."):
    """Read original conditions under a source-pinned interpretation manifest."""
    if manifest["schema"] != "mindie-sim.capture-protocol/1":
        raise ValueError("capture_protocol_schema_unknown")
    _checked(root, manifest["producer"])
    for reference in manifest.get("supporting_sources", ()):
        _checked(root, reference)
    adapter = ADAPTERS[manifest["adapter"]]
    for capture in manifest["captures"]:
        yield from adapter(manifest, capture, _checked(root, capture))


def iter_observations(manifest, *, root="."):
    """Yield the existing accuracy sample contract, with unassigned original use."""
    for condition in iter_conditions(manifest, root=root):
        yield from condition["samples"]


def capture_inventory(manifest, *, root=".", preview_limit=5):
    conditions, samples, boundaries, statuses = 0, 0, Counter(), Counter()
    identities, runs, sources, previews = set(), set(), set(), []
    for condition in iter_conditions(manifest, root=root):
        conditions += 1
        statuses[condition["status"]] += 1
        identity = condition["identity"]
        boundaries[identity.get("boundary", "unknown")] += 1
        if identity:
            identities.add(json.dumps(identity, sort_keys=True))
        samples += len(condition["samples"])
        for sample in condition["samples"]:
            runs.add(sample["run_id"])
            sources.add(sample["source"].split("#")[0])
        if len(previews) < preview_limit:
            previews.append({key: value for key, value in condition.items() if key != "samples"})
    return {
        "schema": "mindie-sim.capture-inventory/1",
        "adapter": manifest["adapter"], "source_captures": len(manifest["captures"]),
        "timing_conditions": conditions, "distinct_identities": len(identities),
        "event_or_kernel_sum_samples": samples, "acquisition_runs": len(runs),
        "conditions_by_boundary": dict(boundaries), "condition_statuses": dict(statuses),
        "sample_splits": {"unassigned": samples}, "condition_preview": previews,
        "sources": sorted(sources), "limitations": manifest["limitations"],
    }
