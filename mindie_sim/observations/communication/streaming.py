"""Bounded streaming JSON projection and weighted timing aggregation.

[Role] Stream very large communication capture JSON files in fixed 64 KiB chunks
  without materializing raw per-iteration arrays or unbounded sample buffers.
[Invariants] Working memory stays O(1) over streamed sample buckets and folded
  timing quantiles; a pure-stdlib chunked parser replaces ``ijson`` whenever it
  is absent; repeat/split conflicts raise immediately.
[Timeouts & Locks] Bounded synchronous file reads; no timers, locks, or
  subprocesses.
[Side-Effect Policy] Read-only over the input stream or path; never writes files
  or mutates calibration state.
"""
from __future__ import annotations

import codecs
from copy import deepcopy
import json
import re
from typing import Mapping

from .normalization import (
    _dtype,
    _mode,
    _observed_mode,
    _protocol_key,
    _selectors,
    _split,
    _validate_row,
)

STREAM_CHUNK_BYTES = 65536
_STREAM_COMPACT_THRESHOLD_BYTES = 1_048_576
_TIMING_COMPACT_LIMIT = 256
_STREAM_BUCKET_LIMIT = 4096
_STREAM_ARRAY_LIMIT = 65536
_UNICODE_HEX_DIGITS = 4
_HEX_RADIX = 16
_SURROGATE_ESCAPE_LEN = 6
_UTF16_HIGH_SURROGATE_MIN = 0xD800
_UTF16_HIGH_SURROGATE_MAX = 0xDBFF
_UTF16_LOW_SURROGATE_BASE = 0xDC00
_UTF16_SURROGATE_SHIFT = 10
_UTF16_SUPPLEMENTARY_BASE = 0x10000

_TIMING_KEYS = frozenset({
    "per_op_ms", "median_ms", "duration_ms", "ms", "per_op_ms_summary",
    "raw_iterations_us",
})
_SAMPLE_KEYS = frozenset({
    "split", "run_id", "repeat_index", "status", "state", "canary_ok",
    "canary_passed", "canary", "correctness", "hardware_key", "sku",
    "arch_family", "hardware", "single", "boundary", "observed_expansion_mode",
    "observed_execution", "requested_expansion_mode", "profiler_evidence",
    "stack", "cann", "algorithm", "protocol", "machine", "stack_binding",
    "payload_bytes", "size_bytes", "work", "dtype",
    "measurement", "transport_window", "direction_observed",
    "algorithm_identified",
})
_STREAM_SKIP = frozenset({
    "per_op_ms_by_trial", "per_rank_median_ms", "clock_samples",
    "clock_summary", "metadata_before", "metadata_after", "rank_map",
    "devices", "env",
})
_JSON_TOKEN = re.compile(
    r'\s*(?:'
    r'(?P<lbrace>\{)|(?P<rbrace>\})|(?P<lbracket>\[)|(?P<rbracket>\])|'
    r'(?P<colon>:)|(?P<comma>,)|'
    r'(?P<string>"(?:[^"\\]|\\.)*")|'
    r'(?P<number>-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)|'
    r'(?P<literal>true|false|null)'
    r')', re.DOTALL)
_ESCAPES = {
    '"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f",
    "n": "\n", "r": "\r", "t": "\t",
}

__all__ = [
    "STREAM_CHUNK_BYTES",
    "_fold_timings",
    "_iter_json_events",
    "_read_stream_value",
    "_stream_json",
    "_weighted_median",
]


def _decode_json_string(text):
    if "\\" not in text:
        return text
    result, index = [], 0
    while index < len(text):
        character = text[index]
        if character != "\\":
            result.append(character)
            index += 1
            continue
        escape = text[index + 1]
        index += 2
        if escape in _ESCAPES:
            result.append(_ESCAPES[escape])
        elif escape == "u":
            code = int(text[index:index + _UNICODE_HEX_DIGITS], _HEX_RADIX)
            index += _UNICODE_HEX_DIGITS
            if (_UTF16_HIGH_SURROGATE_MIN <= code <= _UTF16_HIGH_SURROGATE_MAX
                    and text[index:index + 2] == "\\u"):
                low = int(text[index + 2:index + _SURROGATE_ESCAPE_LEN], _HEX_RADIX)
                index += _SURROGATE_ESCAPE_LEN
                code = (_UTF16_SUPPLEMENTARY_BASE
                        + ((code - _UTF16_HIGH_SURROGATE_MIN) << _UTF16_SURROGATE_SHIFT)
                        + (low - _UTF16_LOW_SURROGATE_BASE))
            result.append(chr(code))
        else:
            raise ValueError("invalid json string escape")
    return "".join(result)


def _iter_json_tokens(handle, chunk_size=STREAM_CHUNK_BYTES):
    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer, position, eof = "", 0, False
    while True:
        match = _JSON_TOKEN.match(buffer, position)
        if match is None:
            if eof:
                if buffer[position:].strip():
                    raise ValueError("invalid trailing json stream data")
                return
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += decoder.decode(chunk)
            else:
                eof = True
                buffer += decoder.decode(b"", final=True)
            continue
        if match.lastgroup == "number" and not eof:
            end = match.end()
            # A number may continue in the next chunk; never commit a prefix
            # such as "1" while the stream still holds ".5".
            if end == len(buffer) or buffer[end] not in " \t\r\n,]}":
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += decoder.decode(chunk)
                else:
                    eof = True
                    buffer += decoder.decode(b"", final=True)
                continue
        position = match.end()
        kind = match.lastgroup
        text = match.group(kind)
        if kind == "string":
            yield ("string", _decode_json_string(text[1:-1]))
        elif kind == "number":
            yield ("number", float(text))
        elif kind == "literal":
            yield ("literal", {"true": True, "false": False, "null": None}[text])
        else:
            yield ("punct", text)
        if position > _STREAM_COMPACT_THRESHOLD_BYTES:
            buffer = buffer[position:]
            position = 0


def _parse_json_value(token, tokens):
    kind, value = token
    if kind == "punct" and value == "{":
        yield ("start_map", None)
        token = next(tokens)
        if token == ("punct", "}"):
            yield ("end_map", None)
            return
        while True:
            if token[0] != "string":
                raise ValueError("json object key must be a string")
            yield ("map_key", token[1])
            if next(tokens) != ("punct", ":"):
                raise ValueError("json object key requires a colon")
            yield from _parse_json_value(next(tokens), tokens)
            separator = next(tokens)
            if separator == ("punct", "}"):
                yield ("end_map", None)
                return
            if separator != ("punct", ","):
                raise ValueError("json object requires a comma or closing brace")
            token = next(tokens)
    elif kind == "punct" and value == "[":
        yield ("start_array", None)
        token = next(tokens)
        if token == ("punct", "]"):
            yield ("end_array", None)
            return
        while True:
            yield from _parse_json_value(token, tokens)
            separator = next(tokens)
            if separator == ("punct", "]"):
                yield ("end_array", None)
                return
            if separator != ("punct", ","):
                raise ValueError("json array requires a comma or closing bracket")
            token = next(tokens)
    elif kind == "string":
        yield ("string", value)
    elif kind == "number":
        yield ("number", value)
    elif kind == "literal":
        yield ("boolean", value) if isinstance(value, bool) else ("null", None)
    else:
        raise ValueError("json value expected")


def _iter_json_events(handle, chunk_size=STREAM_CHUNK_BYTES):
    tokens = _iter_json_tokens(handle, chunk_size)
    yield from _parse_json_value(next(tokens), tokens)


def _skip_stream_value(events, item=None):
    event = (item or next(events))[0]
    if event not in {"start_map", "start_array"}:
        return
    depth = 1
    while depth:
        event = next(events)[0]
        if event in {"start_map", "start_array"}:
            depth += 1
        elif event in {"end_map", "end_array"}:
            depth -= 1


def _sample_ms(sample):
    for key in ("median_ms", "per_op_ms", "duration_ms", "ms", "per_op_ms_summary"):
        val = sample.get(key)
        if isinstance(val, Mapping):
            val = val.get("median")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    raw = [(float(x) / 1000.0, 1) for x in (sample.get("raw_iterations_us") or ())
           if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return _weighted_median(raw) if raw else None


def _fold_timings(values, item, weight=1):
    if item is None:
        return
    values.append((float(item), int(weight)))
    if len(values) > _TIMING_COMPACT_LIMIT:
        values.sort(key=lambda pair: pair[0])
        merged = []
        for val, wt in values:
            if merged and merged[-1][0] == val:
                merged[-1] = (val, merged[-1][1] + wt)
            else:
                merged.append((val, wt))
        if len(merged) > _STREAM_BUCKET_LIMIT:
            raise ValueError("a5_stream_timing_quantile_limit_exceeded")
        values[:] = merged


def _weighted_median(values):
    if not values:
        return None
    pairs = sorted(values, key=lambda pair: pair[0])
    total = sum(wt for _, wt in pairs)
    lo, hi, acc, lo_val = (total - 1) // 2, total // 2, 0, None
    for val, wt in pairs:
        nxt = acc + wt
        if lo_val is None and nxt > lo:
            lo_val = val
        if nxt > hi:
            return (lo_val + val) / 2.0
        acc = nxt
    return pairs[-1][0]


def _semantic_key(value):
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          allow_nan=False)
    except (TypeError, ValueError):
        return repr(value)


def _bucket_add(buckets, sample, index):
    pay = sample.get("payload_bytes", sample.get("size_bytes", sample.get("work")))
    obs = sample.get("observed_expansion_mode", sample.get("observed_execution"))
    req, ev = sample.get("requested_expansion_mode"), sample.get("profiler_evidence")
    sel = _selectors({}, sample)
    sp_ex = _split(sample.get("split"))
    key = (
        "repeat_index" in sample, "run_id" in sample, sample.get("run_id") or None,
        _dtype(sample.get("dtype")) if sample.get("dtype") else None,
        int(pay) if pay is not None else None,
        bool(sample["single"]) if sample.get("single") is not None else None,
        "boundary" in sample, sample.get("boundary"),
        "requested_expansion_mode" in sample,
        _mode(req) if req is not None else None,
        obs is not None, _observed_mode(obs) if obs is not None else None,
        "profiler_evidence" in sample, _protocol_key(ev),
        "stack" in sample, sel["stack"],
        sel["cann"], sel["algorithm"],
        "protocol" in sample, _protocol_key(sel["protocol"]),
        _semantic_key(sample.get("measurement")),
        _semantic_key(sample.get("transport_window")),
        sample.get("direction_observed"), sample.get("algorithm_identified"),
    )
    entry = buckets.get(key)
    if entry is None:
        if len(buckets) >= _STREAM_BUCKET_LIMIT:
            raise ValueError("a5_stream_sample_identity_limit_exceeded")
        entry = {}
        buckets[key] = entry
    ms, r = _sample_ms(sample), int(sample.get("repeat_index", index) or 0)
    if ms is not None:
        for o_sp, (_, _, o_seen, o_runs) in entry.items():
            hit, p_ms = (True, o_seen[r]) if r in o_seen else next(
                ((True, v) for lo, hi, v in o_runs
                 if v is not Ellipsis and lo <= r <= hi), (False, None))
            if hit:
                if (p_ms is None or p_ms != ms
                        or (sp_ex is not None and o_sp is not None and o_sp != sp_ex)):
                    raise ValueError("conflicting_primitive_repeat_or_split")
                if o_sp == sp_ex:
                    return
    part = entry.get(sp_ex)
    if part is None:
        tmpl = {k: v for k, v in sample.items()
                if k in _SAMPLE_KEYS and k not in _TIMING_KEYS and k != "split"}
        if sp_ex is not None:
            tmpl["split"] = sp_ex
        part = [tmpl, [], {}, []]
        entry[sp_ex] = part
    _, vals, seen, runs = part
    if ms is not None:
        if not runs:
            runs.append([r, r, ms])
        elif runs[-1][2] is Ellipsis:
            runs[-1][0], runs[-1][1] = min(runs[-1][0], r), max(runs[-1][1], r)
        elif r == runs[-1][1] + 1:
            if ms == runs[-1][2] or runs[-1][2] is None:
                runs[-1][1] = r
            elif len(runs) < _TIMING_COMPACT_LIMIT:
                runs.append([r, r, ms])
            else:
                runs[-1][1], runs[-1][2] = r, None
                while len(runs) > 1 and runs[-2][1] + 1 == runs[-1][0]:
                    runs[-2][1], runs[-2][2] = runs[-1][1], None
                    runs.pop()
        else:
            seen[r] = ms
            if len(seen) >= _STREAM_BUCKET_LIMIT:
                merged = []
                for lo, hi, v in sorted(runs + [[k, k, w] for k, w in seen.items()]):
                    if merged and merged[-1][1] + 1 == lo:
                        merged[-1][1] = hi
                        merged[-1][2] = v if merged[-1][2] == v else None
                    else:
                        merged.append([lo, hi, v])
                runs[:] = ([[merged[0][0], merged[-1][1], Ellipsis]]
                           if len(merged) >= _STREAM_BUCKET_LIMIT else merged)
                seen.clear()
    _fold_timings(vals, ms)


def _read_stream_samples(events):
    """Consume a samples array in bounded O(1) working memory."""
    item = next(events)
    raw_samples, buckets, timings, has_evidence, idx = [], {}, [], False, 0

    def _consume(sample):
        nonlocal raw_samples, has_evidence, idx
        cur = idx
        idx += 1
        if not isinstance(sample, Mapping):
            return
        _validate_row(sample, f"stream:sample[{cur}]")
        keys = set(sample)
        if not (keys & (_SAMPLE_KEYS | _TIMING_KEYS)):
            return
        if keys & _SAMPLE_KEYS:
            has_evidence = True
        _fold_timings(timings, _sample_ms(sample))
        clean = {k: sample[k] for k in keys & (_SAMPLE_KEYS | _TIMING_KEYS)}
        if raw_samples is not None:
            raw_samples.append((cur, clean))
            if len(raw_samples) > _TIMING_COMPACT_LIMIT:
                for p_idx, prev in raw_samples:
                    _bucket_add(buckets, prev, p_idx)
                raw_samples = None
        else:
            _bucket_add(buckets, clean, cur)

    if item[0] != "start_array":
        value = _read_stream_value(events, item)
        for sample in (value if isinstance(value, list) else ()):
            _consume(sample)
    else:
        while True:
            entry = next(events)
            if entry[0] == "end_array":
                break
            _consume(_read_stream_value(events, entry))
    if not has_evidence:
        return [], _weighted_median(timings)
    if raw_samples is not None:
        return [s for _, s in raw_samples], None
    return [{**tmpl, **({"per_op_ms": _weighted_median(vals)} if vals else {}),
             "_repeats": (runs, seen), "_weights": list(vals)}
            for entry in buckets.values()
            for tmpl, vals, seen, runs in entry.values()], None


def _read_stream_value(events, item=None):
    item = item or next(events)
    if item[0] == "start_map":
        result, s_med, r_med = {}, None, None
        while True:
            entry = next(events)
            if entry[0] == "end_map":
                fb = s_med if s_med is not None else r_med
                cur = result.get("per_op_ms_summary")
                if fb is not None and not (
                    isinstance(cur, Mapping)
                    and isinstance(cur.get("median"), (int, float))
                ):
                    result["per_op_ms_summary"] = {"median": fb}
                samples = result.get("samples")
                if isinstance(samples, list):
                    inherited = {key: result[key] for key in (
                        "measurement", "transport_window", "direction_observed",
                        "algorithm_identified") if key in result}
                    for sample in samples:
                        if isinstance(sample, dict):
                            for key, value in inherited.items():
                                sample.setdefault(key, deepcopy(value))
                return result
            key = entry[1]
            if key in _STREAM_SKIP:
                _skip_stream_value(events)
            elif key == "samples":
                filtered, s_med = _read_stream_samples(events)
                if filtered:
                    result["samples"] = filtered
            elif key == "raw_iterations_us":
                raw_entry = next(events)
                if raw_entry[0] != "start_array":
                    _skip_stream_value(events, raw_entry)
                else:
                    vals = []
                    while (elem := next(events))[0] != "end_array":
                        val = _read_stream_value(events, elem)
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            _fold_timings(vals, float(val) / 1000.0)
                    r_med = _weighted_median(vals)
            else:
                result[key] = _read_stream_value(events)
    if item[0] == "start_array":
        result = []
        while True:
            entry = next(events)
            if entry[0] == "end_array":
                return result
            if len(result) >= _STREAM_ARRAY_LIMIT:
                raise ValueError("a5_stream_array_limit_exceeded")
            result.append(_read_stream_value(events, entry))
    return item[1]


def _stream_json(path):
    """Stream one JSON document into the bounded ingest projection."""
    with open(path, "rb") as handle:
        try:
            import ijson
        except ImportError:
            events = _iter_json_events(handle)
        else:
            events = ijson.basic_parse(handle, use_float=True)
        return _read_stream_value(events)
