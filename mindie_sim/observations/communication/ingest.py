"""Streaming communication capture ingestion and chain parsing.

[Role] Stream HCCL profiling traces and parse causal collective chains.
[Invariants] Bounded streaming memory; zero fabrication.
[Timeouts & Locks] Local read-only file streaming; no locks.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

import codecs
from copy import deepcopy
import json
import re
from typing import Mapping

from .schema import (
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


def _inherit_sample_fields(result):
    samples = result.get("samples")
    if isinstance(samples, list):
        inherited = {key: result[key] for key in (
            "measurement", "transport_window", "direction_observed",
            "algorithm_identified") if key in result}
        for sample in samples:
            if isinstance(sample, dict):
                for key, value in inherited.items():
                    sample.setdefault(key, deepcopy(value))


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
                _inherit_sample_fields(result)
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


from collections import defaultdict
import math
from statistics import median
from typing import Mapping

from .schema import (
    _canonical_semantic,
    _dtype,
    _protocol_key,
    _validate_row,
    resolve_expansion_regime,
)



def _chain_row(values, campaign, origin, default_dtype, default_observed,
               default_requested, default_evidence, default_run_id=None,
               default_repeat=0):
    if isinstance(values, (list, tuple)) and len(values) == 10:
        (dtype, ep, experts, tokens, dispatch, gate,
         swiglu, down, combine, chain_ms) = values
        norm_dtype = _dtype(dtype)
        legs = {
            "dispatch": float(dispatch), "gate_up": float(gate),
            "swiglu_quant" if norm_dtype != "bf16" else "swiglu": float(swiglu),
            "down": float(down), "combine": float(combine),
        }
        return _chain_record(
            campaign, origin, norm_dtype, int(ep), int(experts), int(tokens),
            legs, float(chain_ms), f"{norm_dtype}-ep{ep}-e{experts}-g{tokens}",
            default_observed, default_requested, default_evidence,
            run_id=default_run_id, repeat_index=default_repeat)
    if not isinstance(values, Mapping):
        raise ValueError(f"a5_mc2_chain_row_invalid:{campaign}")
    row_status = values.get("status", values.get("state"))
    if row_status is not None and str(row_status).strip().lower() not in {"ok", "completed"}:
        raise ValueError(f"a5_mc2_chain_row_not_completed:{campaign}:{row_status!r}")
    _validate_row(values, f"{campaign}:{values.get('condition', values.get('chain_id'))}")
    legs = {str(key): float(value) for key, value in (values.get("legs_ms") or {}).items()}
    if not legs:
        raise ValueError(f"a5_mc2_chain_row_missing_legs:{campaign}")
    chain_ms = values.get("chain_ms", values.get("chain_p50_ms", values.get("pipeline_ms")))
    if chain_ms is None:
        raise ValueError(f"a5_mc2_chain_row_missing_chain:{campaign}")
    dtype = _dtype(values.get("dtype") or default_dtype)
    swiglu = legs.get("swiglu_quant", legs.get("swiglu"))
    normalized = {
        "dispatch": legs.get("dispatch", 0.0),
        "gate_up": legs.get("gate_up", legs.get("gmm_gate_up", 0.0)),
        "down": legs.get("down", legs.get("gmm_down", 0.0)),
        "combine": legs.get("combine", 0.0),
    }
    normalized["swiglu_quant" if dtype != "bf16" else "swiglu"] = swiglu or 0.0
    ep = values.get("ranks") or values.get("ep") or values.get("world_size") or 1
    rep = int(values.get("repeat_index")
              if values.get("repeat_index") is not None else default_repeat)
    return _chain_record(
        campaign, origin, dtype, int(ep), values.get("experts"),
        values.get("global_tokens"), normalized, float(chain_ms),
        values.get("condition") or values.get("chain_id") or campaign,
        values.get("observed_expansion_mode", default_observed),
        values.get("requested_expansion_mode", default_requested),
        values.get("profiler_evidence", default_evidence),
        run_id=values.get("run_id") or default_run_id, repeat_index=rep,
        chain_id=values.get("chain_id") or None)


def _chain_record(campaign, origin, dtype, ep, experts, tokens, legs, chain_ms,
                  condition, observed="not_observed", requested="unspecified",
                  evidence=None, *, run_id=None, repeat_index=0, chain_id=None):
    total = sum(legs.values())
    if not math.isfinite(chain_ms) or chain_ms <= 0 or total <= 0:
        raise ValueError(f"a5_mc2_chain_row_nonpositive:{campaign}:{condition}")
    resolved = resolve_expansion_regime(requested, observed, profiler_evidence=evidence)
    result = {
        "campaign": campaign, "chain_id": chain_id or f"{campaign}:{condition}",
        "condition": condition, "run_id": run_id or campaign,
        "repeat_index": int(repeat_index or 0),
        "dtype": dtype, "ep": ep, "world_size": ep,
        "experts": int(experts) if experts is not None else None,
        "global_tokens": int(tokens) if tokens is not None else None,
        "legs_ms": legs, "chain_p50_ms": chain_ms, "sum_leg_p50_ms": total,
        "overlap_ratio": chain_ms / total, "source": origin,
        "profiler_evidence": evidence,
        "stages": ("quant", "all_to_allv_dispatch", "grouped_matmul",
                   "all_to_allv_combine"),
        "stage_legs_ms": {
            "quant": legs.get("swiglu_quant", legs.get("swiglu", 0.0)),
            "all_to_allv_dispatch": legs["dispatch"],
            "grouped_matmul": legs["gate_up"] + legs["down"],
            "all_to_allv_combine": legs["combine"],
        },
    }
    for key in ("requested_expansion_mode", "observed_expansion_mode", "regime",
                "expansion_unconfirmed", "unconfirmed_tag", "profiler_confirmed"):
        result[key] = resolved[key]
    return result


def _bf16_chains(document, campaign, origin, observed, requested, evidence,
                 default_run_id=None, default_repeat=0):
    by_condition = defaultdict(dict)
    for case_id, stages in (document.get("median_ms_by_case") or {}).items():
        kind = case_id.rsplit("-", 1)[-1]
        by_condition[case_id[:-(len(kind) + 1)]][kind] = stages
    rows = []
    for condition in sorted(
        by_condition, key=lambda value: tuple(int(part[1:]) for part in value.split("-"))
    ):
        stages = by_condition[condition]

        def pick(kind, stage, *, _stages=stages):
            values = [value for value in (_stages.get(kind, {}).get(stage) or {}).values()
                      if isinstance(value, (int, float))]
            return median(values) if values else None

        dispatch = pick("dispatch", "dispatch")
        combine = pick("combine", "combine")
        gate = pick("gmm", "gmm_gate_up")
        down = pick("gmm", "gmm_down")
        swiglu = pick("gmm", "swiglu")
        chain_ms = pick("pipeline", "pipeline")
        if None in (dispatch, combine, gate, down, swiglu, chain_ms):
            continue
        expert_text, token_text = condition.split("-")
        legs = {"dispatch": dispatch, "gate_up": gate, "swiglu": swiglu,
                "down": down, "combine": combine}
        rows.append(_chain_record(
            campaign, origin, _dtype(document.get("dtype") or "bf16"),
            int(document.get("ranks_requested") or 1),
            int(expert_text[1:]), int(token_text[1:]), legs, chain_ms, condition,
            observed, requested, evidence, run_id=default_run_id,
            repeat_index=default_repeat))
    return rows


def _campaign_chains(document, label, origin):
    observed = document.get(
        "observed_expansion_mode",
        document.get("observed_execution", "not_observed"))
    requested = document.get("requested_expansion_mode", "unspecified")
    evidence = document.get("profiler_evidence")
    default_run_id = document.get("run_id") or label
    default_repeat = int(document.get("repeat_index", 0) or 0)
    if "median_ms_by_case" in document:
        return _bf16_chains(
            document, label, origin, observed, requested, evidence,
            default_run_id, default_repeat)
    values = (document.get("chains")
              or (document.get("provenance") or {}).get("chain_validation")
              or document.get("chain_validation"))
    if not values:
        return []
    return [_chain_row(
        value, label, origin, _dtype(document.get("dtype") or "bf16"),
        observed, requested, evidence, default_run_id, default_repeat)
        for value in values]


def _dedupe_conditions(rows, keep_meta=False):
    grouped, weights, metas, intervals = {}, {}, {}, defaultdict(list)
    for row in rows:
        reps, wts, sp = row.pop("_repeats", None), row.pop("_weights", None), row["split"]
        eff_sp = sp or "train"
        base = (
            row["campaign"], row["run_id"], row["collective"], row["world_size"],
            row["placement_scope"], row["dtype"], row["payload_bytes"],
            row["single"], row["pair"], row["regime"], row.get("unconfirmed_tag"),
            row.get("profiler_confirmed"), row.get("stack"), row.get("cann"),
            row.get("algorithm"), _protocol_key(row.get("protocol")),
            row.get("boundary"), _canonical_semantic(row.get("measurement")),
            _canonical_semantic(row.get("transport_window")),
            row.get("direction_observed"), row.get("algorithm_identified"),
        )
        spans = (
            [(lo, hi, (ms, eff_sp)) for lo, hi, ms in reps[0]]
            + [(r, r, (ms, eff_sp)) for r, ms in reps[1].items()]
            if reps else [(row["repeat_index"], row["repeat_index"],
                           (row["per_op_ms"], eff_sp))]
        )
        if any(
            lo <= p_hi and p_lo <= hi and (
                rec[1] != p_rec[1] or (
                    rec[0] is not Ellipsis and p_rec[0] is not Ellipsis
                    and (rec[0] is None or p_rec[0] is None or rec[0] != p_rec[0])
                )
            )
            for lo, hi, rec in spans for p_lo, p_hi, p_rec in intervals[base]
        ):
            raise ValueError("conflicting_primitive_repeat_or_split")
        if intervals[base] and all(
            rec[0] is not None and rec[0] is not Ellipsis
            and any(p_rec == rec and p_lo <= lo and hi <= p_hi
                    for p_lo, p_hi, p_rec in intervals[base])
            for lo, hi, rec in spans
        ):
            continue
        intervals[base].extend(spans)
        c_key = base + (None, sp)
        key = c_key if reps or c_key in weights else base + (row["repeat_index"], sp)
        if reps and c_key not in weights:
            prevs = [k for k in grouped if k[:-2] == base and k[-1] == sp]
            if prevs:
                wts = [(grouped[k]["per_op_ms"], 1) for k in prevs] + list(wts or ())
                row["repeat_index"] = min(
                    [row["repeat_index"]]
                    + [grouped[k]["repeat_index"] for k in prevs])
                for k in prevs:
                    grouped.pop(k, None)
        if key == c_key:
            cur = weights.setdefault(c_key, [])
            for val, wt in (wts or ((row["per_op_ms"], 1),)):
                _fold_timings(cur, val, wt)
            row["per_op_ms"], row["_weights"] = _weighted_median(cur), list(cur)
            if keep_meta and reps:
                m = metas.setdefault(c_key, ([], {}))
                m[0].extend(reps[0])
                m[1].update(reps[1])
                row["_repeats"] = m
            if (prev := grouped.get(c_key)) is not None:
                prev["per_op_ms"], prev["_weights"] = row["per_op_ms"], row["_weights"]
                prev["repeat_index"] = min(prev["repeat_index"], row["repeat_index"])
                continue
        grouped[key] = row
    return [grouped[k] for k in sorted(
        grouped,
        key=lambda k: str(k[:2] + (grouped[k]["repeat_index"],) + k[2:-2] + (k[-1],)))]


def _dedupe_chains(rows):
    seen = {}
    for row in rows:
        key = (
            row["campaign"], row["condition"], row["dtype"], row["ep"],
            row["experts"], row["global_tokens"], round(row["chain_p50_ms"], 9),
            row["regime"], round(float(row["sum_leg_p50_ms"]), 9),
            row.get("run_id"), row.get("repeat_index"), row.get("chain_id"),
            tuple(sorted((name, round(float(value), 9))
                         for name, value in (row.get("legs_ms") or {}).items())),
        )
        seen.setdefault(key, row)
    return [seen[key] for key in sorted(seen, key=str)]


from collections import defaultdict
from itertools import chain
import json
import os
from pathlib import Path
from statistics import median
from typing import Mapping

from mindie_sim.schema.common import ModelingError
from .schema import (
    A5_ARCH_FAMILY,
    A5_HARDWARE_KEY,
    EMBEDDED_A5_CAMPAIGNS,
    EMBEDDED_A5_MC2_CAMPAIGNS,
)
from .schema import (
    _collective,
    _dtype,
    _explicit_pair,
    _explicit_scope,
    _fallback_measurement,
    _fallback_transport_window,
    _merge_obs,
    _observed_mode,
    _pair,
    _pair_scope,
    _protocol_key,
    _require_a5,
    _scope,
    _selectors,
    _split,
    _validate_campaign,
    _validate_measurement_window,
    _validate_row,
    resolve_expansion_regime,
)

_LARGE_JSON_BYTES = 50 * 1024 * 1024
_OTHER_JSON_MAX_BYTES = 8 * 1024 * 1024
_KNOWN_A5_ARTIFACTS = tuple((
    "2026-10-21/comm-rack06/a5-comm-rack06-calibration.json "
    "2026-10-21/comm-rack08/a5-comm-rack08-calibration.json "
    "2026-10-22/comm-w16/artifacts/a5-comm-w16-calibration.json "
    "2026-10-22/comm-w32/artifacts/a5-comm-w32-calibration.json "
    "2026-10-21/mc2/a5mc2-rack06-r1/a5mc2-ep2.json "
    "2026-10-21/mc2/a5mc2-rack06-r1/a5mc2-ep4.json "
    "2026-10-21/mc2/a5mc2-rack08-r1/a5mc2-ep2.json "
    "2026-10-21/mc2/a5mc2-rack08-r1/a5mc2-ep4.json "
    "2026-10-22/mc2-quant/results/a5-mc2-quant-chain-validation.json "
    "2026-10-22/mc2-quant/results/a5-mc2-quant-comm-lane.json"
).split())
_EXPAND_KEYS = frozenset({
    "split", "run_id", "repeat_index", "single", "boundary",
    "observed_expansion_mode", "observed_execution",
    "requested_expansion_mode", "profiler_evidence", "stack",
    "cann", "algorithm", "protocol", "machine", "stack_binding",
    "payload_bytes", "size_bytes", "work", "dtype",
    "measurement", "transport_window", "direction_observed",
    "algorithm_identified",
})



def _load_json(path):
    path = Path(path)
    try:
        size = path.stat().st_size
    except TimeoutError:
        raise
    except OSError as exc:
        raise ValueError(f"a5_campaign_source_unreadable:{path}") from exc
    if size > _LARGE_JSON_BYTES:
        try:
            return _stream_json(path)
        except TimeoutError:
            raise
        except Exception as exc:
            raise ValueError(f"a5_campaign_source_unreadable:{path}") from exc
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except TimeoutError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"a5_campaign_source_unreadable:{path}") from exc


def _row_duration(row):
    for key in ("median_ms", "per_op_ms", "duration_ms", "ms", "per_op_ms_summary"):
        val = row.get(key)
        if isinstance(val, Mapping):
            val = val.get("median")
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
    if row.get("samples"):
        vals = [v for s in row["samples"]
                if isinstance(s, Mapping) and (v := _row_duration(s)) is not None]
        if vals:
            return float(median(vals))
    raw = [float(x) / 1000.0 for x in (row.get("raw_iterations_us") or ())
           if isinstance(x, (int, float)) and not isinstance(x, bool)]
    return float(median(raw)) if raw else None


def _base_row(document, label, origin, collective, world, scope, pair, payload,
              duration, dtype, single, row=None, requested=None, observed=None,
              evidence=None, run_id=None, repeat_index=0, split=None):
    row = row or {}
    boundary = row.get("boundary", document.get("boundary"))
    if boundary not in (None, "device_kernel", "resource_phase"):
        raise ModelingError("service_curve_requires_device_or_resource_phase_boundary")
    resolved = resolve_expansion_regime(
        row.get("requested_expansion_mode", requested),
        row.get("observed_expansion_mode", row.get("observed_execution", observed)),
        profiler_evidence=row.get("profiler_evidence", evidence))
    algorithm_identified = (
        row.get("algorithm", document.get("algorithm")) is not None)
    declared_measurement = row.get("measurement", document.get("measurement"))
    if isinstance(declared_measurement, Mapping):
        measurement = dict(declared_measurement)
        measurement.setdefault("explicit", True)
    else:
        measurement = _fallback_measurement(
            collective, boundary, algorithm_identified)
    declared_window = row.get("transport_window", document.get("transport_window"))
    window = (dict(declared_window) if isinstance(declared_window, Mapping)
              else _fallback_transport_window(measurement))
    _validate_measurement_window(measurement, window)
    return {
        "campaign": label, "collective": collective, "world_size": int(world),
        "placement_scope": scope, "dtype": _dtype(row.get("dtype") or dtype),
        "payload_bytes": int(payload), "per_op_ms": float(duration),
        "single": bool(single), "pair": pair,
        **{key: resolved[key] for key in (
            "observed_expansion_mode", "requested_expansion_mode", "regime",
            "expansion_unconfirmed", "unconfirmed_tag", "profiler_confirmed")},
        "profiler_evidence": row.get("profiler_evidence", evidence),
        **_selectors(document, row),
        "canary": row.get("canary", row.get("correctness")),
        "run_id": run_id or label, "repeat_index": int(repeat_index or 0),
        "split": split, "source": origin, "boundary": boundary,
        "measurement": measurement, "transport_window": window,
        "direction_observed": bool(measurement.get("direction_observed", False)),
        "algorithm_identified": bool(measurement.get(
            "algorithm_identified", algorithm_identified)),
        **{k: row[k] for k in ("_repeats", "_weights") if k in row},
    }


def _rank_rows(document, origin, p2p_pair=None, default_run_id=None, default_repeat=0):
    protocol = (document.get("protocol")
                if isinstance(document.get("protocol"), Mapping) else {})
    dtype = _dtype(document.get("dtype") or protocol.get("dtype"))
    world = (document.get("world_size")
             or len(str(document.get("visible_devices", "")).split(",")))
    placement = document.get("placement", "")
    default_scope = _explicit_scope(document.get("placement_scope"))
    observed = document.get(
        "observed_expansion_mode",
        document.get("observed_execution", "not_observed"))
    requested = document.get("requested_expansion_mode", "unspecified")
    evidence = document.get("profiler_evidence")
    rows = []
    for case in document.get("cases") or ():
        _validate_row(case, f"{origin}:{case.get('case_id')}")
        collective = _collective(case.get("collective") or case.get("mode"))
        declared_scope = (
            _explicit_scope(case.get("placement_scope")) or default_scope)
        scope = (declared_scope or _explicit_scope(case.get("placement"))
                 or _scope(case.get("placement") or placement, world))
        explicit = (_explicit_pair(case.get("p2p_pair"))
                    or _explicit_pair(case.get("pair"))
                    or _explicit_pair(document.get("p2p_pair"))
                    or _explicit_pair(p2p_pair))
        pair = explicit if collective == "p2p_send_recv" else None
        if collective == "p2p_send_recv":
            scope = scope if declared_scope else (_pair_scope(pair, world) or scope)
            pair = pair or _pair(
                case.get("p2p_scope") or protocol.get("p2p_scope"),
                collective, scope, world)
        run_id = (case.get("run_id") or document.get("run_id") or default_run_id
                  or Path(origin).stem or document.get("label") or "rank")
        repeat = case.get("repeat_index", document.get("repeat_index", default_repeat))
        split = _split(case.get("split", document.get("split")))
        samples = case.get("samples") or ()
        for index, sample in enumerate(samples):
            if isinstance(sample, Mapping):
                _validate_row(sample, f"{origin}:{case.get('case_id')}:sample[{index}]")
        expand = bool(samples) and any(
            isinstance(sample, Mapping) and (_EXPAND_KEYS & set(sample))
            for sample in samples)
        if expand:
            inh_rep = ("repeat_index" in case or "repeat_index" in document
                       or default_repeat != 0)
            for index, sample in enumerate(samples):
                duration = _row_duration(sample)
                sample_payload = sample.get(
                    "payload_bytes", sample.get(
                        "size_bytes", sample.get(
                            "work", case.get(
                                "payload_bytes", case.get(
                                    "size_bytes", case.get("work"))))))
                if duration is None or sample_payload is None:
                    continue
                reps, rep = sample.get("_repeats"), sample.get("repeat_index")
                if rep is None and reps:
                    if inh_rep:
                        if (len(reps[0]) > 1 or reps[1]
                                or (reps[0] and reps[0][0][2] in (None, Ellipsis))):
                            raise ValueError("conflicting_primitive_repeat_or_split")
                        rep = int(repeat or 0)
                        sample = {
                            **sample,
                            "_repeats": ([[rep, rep, reps[0][0][2]]], {})
                            if reps[0] else reps,
                        }
                    else:
                        rep = reps[0][0][0] if reps[0] else index
                elif rep is None:
                    rep = repeat if inh_rep else index
                rows.append(_base_row(
                    document, document.get("label") or Path(origin).stem, origin,
                    collective, world, scope, pair, sample_payload, duration, dtype,
                    sample.get("single", case.get("single", document.get("single"))),
                    _merge_obs(case, sample), run_id=sample.get("run_id", run_id),
                    repeat_index=rep, split=_split(sample.get("split", split)),
                    requested=requested, observed=observed, evidence=evidence))
            continue
        duration = _row_duration(case)
        payload = case.get("payload_bytes", case.get("size_bytes", case.get("work")))
        if duration is None or payload is None:
            raise ValueError(
                f"a5_campaign_case_has_no_median:{origin}:{case.get('case_id')}")
        rows.append(_base_row(
            document, document.get("label") or Path(origin).stem, origin,
            collective, world, scope, pair, payload, duration, dtype,
            case.get("single", document.get("single")), case, run_id=run_id,
            repeat_index=repeat, split=split, requested=requested,
            observed=observed, evidence=evidence))
    return rows


def _parent_document(path, document):
    origin = str(path) if path is not None else "memory"
    stem = (path.name[:-len(".parent.json")]
            if path is not None and path.name.endswith(".parent.json")
            else path.stem if path is not None
            else str(document.get("label") or document.get("run_id") or "parent"))
    rank_files = sorted(path.parent.glob(stem + ".rank*.json")) if path is not None else ()
    reports = [(report, origin) for report in document.get("rank_reports") or ()
               if isinstance(report, Mapping)]
    if not rank_files and not reports:
        return [(document, origin)]
    protocol, placement = document.get("protocol") or {}, document.get("placement", "")
    world = document.get("world_size") or 1
    pair = _explicit_pair(document.get("p2p_pair")) or _pair(
        protocol.get("p2p_scope"), "p2p_send_recv", _scope(placement, world), world)
    grouped = defaultdict(list)
    parent_run = str(document.get("run_id") or document.get("label") or stem)
    inherited_keys = (
        "observed_expansion_mode", "observed_execution",
        "requested_expansion_mode", "profiler_evidence", "stack", "cann",
        "algorithm", "protocol", "machine", "stack_binding", "split",
        "boundary", "single",
    )
    defaults = {key: document[key] for key in inherited_keys if key in document}
    for rank_document, rank_origin in chain(
        ((_load_json(item), str(item)) for item in rank_files), reports
    ):
        rank_document = _merge_obs(defaults, rank_document)
        _validate_campaign(rank_document, rank_origin)
        rank_rows = _rank_rows(
            rank_document, rank_origin, pair, default_run_id=parent_run,
            default_repeat=document.get("repeat_index", 0))
        for row in _dedupe_conditions(rank_rows, keep_meta=True):
            grouped[(
                row["collective"], row["payload_bytes"], bool(row["single"]),
                row["split"], row["pair"], row["dtype"], row["placement_scope"],
                row["regime"], row.get("boundary"), row["run_id"],
                row["repeat_index"], row.get("stack"), row.get("cann"),
                row.get("algorithm"), _protocol_key(row.get("protocol")),
            )].append(row)
    rows = []
    for _key, group in sorted(grouped.items(), key=str):
        if len(group) < int(world):
            raise NotImplementedError(
                "LEGACY_PURGED: verify complete rank coverage before stamping all_ranks_matched in communication/ingest.py"
            )
        med = median(row["per_op_ms"] for row in group)
        base = {k: v for k, v in group[0].items()
                if k not in {"_repeats", "_weights"} or len(group) == 1}
        extra = (
            {"_weights": [(med, sum(w for _, w in group[0]["_weights"]))]}
            if len(group) > 1 and "_weights" in group[0] else {}
        )
        rows.append({
            **base,
            "campaign": document.get("label") or stem,
            "per_op_ms": med,
            **extra,
            "canary": "all_ranks_matched",
            "source": origin,
        })
    return [({**document, "conditions": rows,
              "label": document.get("label") or stem}, origin)]


def _list_document(values, origin):
    if not values:
        return []
    if all(isinstance(value, Mapping) for value in values):
        if any({"label", "conditions", "series", "chains", "families"} & set(value)
               for value in values):
            return [(value, origin) for value in values]
    raise ValueError(f"a5_campaign_source_list_not_a_campaign:{origin}")


def _expand_path(path):
    document = _load_json(path)
    if isinstance(document, list):
        return _list_document(document, str(path))
    if not isinstance(document, Mapping):
        raise ValueError(f"a5_campaign_source_requires_json_object:{path}")
    if (document.get("role") == "parent" or path.name.endswith(".parent.json")
            or ("rank_reports" in document and "median_ms_by_case" not in document)):
        return _parent_document(path, document)
    if document.get("role") == "rank" or (
        "cases" in document and "median_ms_by_case" not in document
    ):
        label = document.get("label") or path.stem
        return [({
            **document,
            "conditions": _rank_rows(
                document, str(path), _explicit_pair(document.get("p2p_pair"))),
            "label": label,
        }, str(path))]
    return [(document, str(path))]


def _known_root(root):
    return ("mindie-sim-data" in {part.lower() for part in Path(root).parts}
            and Path(root).name.lower() == "a5")


def _directory_documents(root):
    root = Path(root)
    if _known_root(root):
        files = [root / relative for relative in _KNOWN_A5_ARTIFACTS]
        files = [path for path in files if path.is_file()]
        if files:
            return [document for path in files for document in _expand_path(path)]
    calibration, chains, comms, parents, ranks, others = [], [], [], [], [], []
    for path in sorted(root.rglob("*.json")):
        if path.name.startswith("metadata_"):
            continue
        try:
            size = path.stat().st_size
        except TimeoutError:
            raise
        except OSError as exc:
            raise ValueError(f"a5_campaign_source_unreadable:{path}") from exc
        name = path.name.lower()
        if "calibration" in name:
            calibration.append(path)
        elif "chain-validation" in name or "chain_validation" in name:
            chains.append(path)
        elif "comm-lane" in name or "comm_lane" in name:
            comms.append(path)
        elif name.endswith(".parent.json"):
            parents.append(path)
        elif ".rank" in name:
            ranks.append(path)
        elif size <= _OTHER_JSON_MAX_BYTES:
            others.append(path)
    files = calibration + chains + comms
    if not calibration:
        shadowed = {(path.parent, path.name[:-len(".parent.json")]) for path in parents}
        files += parents
        files += [path for path in ranks
                  if (path.parent, path.name.split(".rank")[0]) not in shadowed]
    files = files or others
    if not files:
        raise ValueError(f"a5_campaign_source_has_no_json:{root}")
    return [document for path in files for document in _expand_path(path)]


def _source_documents(source):
    if source is None:
        embedded = (tuple(EMBEDDED_A5_CAMPAIGNS.values())
                    + tuple(EMBEDDED_A5_MC2_CAMPAIGNS.values()))
        return [(_campaign, _campaign.get("source")
                 or f"embedded:{_campaign.get('label', 'a5')}")
                for _campaign in embedded]
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.is_dir():
            return _directory_documents(path)
        if path.is_file():
            return _expand_path(path)
        raise ValueError(f"a5_campaign_source_missing:{path}")
    if isinstance(source, Mapping):
        if source.get("role") == "parent" or (
            "rank_reports" in source and "median_ms_by_case" not in source
        ):
            return _parent_document(None, source)
        if source.get("role") == "rank" or (
            "cases" in source and "median_ms_by_case" not in source
            and "conditions" not in source
        ):
            label = source.get("label") or source.get("run_id") or "rank"
            return [({
                **source,
                "conditions": _rank_rows(
                    source, "memory", _explicit_pair(source.get("p2p_pair"))),
                "label": label,
            }, "memory")]
        if source and all(
            isinstance(value, Mapping)
            and ({"label", "conditions", "series", "chains", "rank_reports", "cases"}
                 & set(value))
            for value in source.values()
        ):
            return [doc for value in source.values() for doc in _source_documents(value)]
        if "legs_ms" in source and ("condition" in source or "chain_ms" in source):
            return _list_document([source], "memory")
        if ({"payload_bytes", "per_op_ms"} <= set(source)
                or {"size_bytes", "median_ms"} <= set(source)):
            return _list_document([source], "memory")
        return [(source, "memory")]
    if isinstance(source, (list, tuple)):
        if (source and all(isinstance(value, Mapping) for value in source)
                and any({"label", "conditions", "series", "chains", "families",
                         "rank_reports", "cases"} & set(value) for value in source)):
            return [doc for value in source for doc in _source_documents(value)]
        return _list_document(list(source), "memory")
    raise ValueError("a5_campaign_source_requires_path_or_mapping")


def _condition_rows(row, document, label, origin, default_dtype, default_world,
                    default_scope, default_pair, default_observed, default_requested,
                    default_evidence):
    _validate_row(row, f"{label}:{row.get('collective', row.get('mode'))}")
    collective = _collective(row.get("collective") or row.get("mode"))
    world = int(row.get("world_size") or row.get("ranks")
                or row.get("group_size") or default_world or 1)
    placement = row.get("placement") or document.get("placement", "")
    declared_scope = _explicit_scope(row.get("placement_scope")) or default_scope
    scope = declared_scope or _explicit_scope(placement) or _scope("", world)
    explicit = _explicit_pair(row.get("p2p_pair")) or _explicit_pair(row.get("pair"))
    pair = explicit if collective == "p2p_send_recv" else None
    if collective == "p2p_send_recv":
        scope = scope if declared_scope else (_pair_scope(pair, world) or scope)
        pair = pair or default_pair or _pair(
            row.get("p2p_scope"), collective, scope, world)
    payload = row.get("payload_bytes", row.get("size_bytes", row.get("work")))
    duration = _row_duration(row)
    run_id = row.get("run_id") or document.get("run_id") or label
    repeat = row.get("repeat_index", document.get("repeat_index", 0))
    split = _split(row.get("split", document.get("split")))
    samples = row.get("samples") or ()
    for index, sample in enumerate(samples):
        if isinstance(sample, Mapping):
            _validate_row(sample, f"{label}:sample[{index}]")
    expand = bool(samples) and any(
        isinstance(sample, Mapping) and (_EXPAND_KEYS & set(sample))
        for sample in samples)
    if not expand and (payload is None or duration is None):
        raise ValueError(f"a5_campaign_condition_missing_payload_or_duration:{label}")
    if expand:
        result = []
        for index, sample in enumerate(samples):
            sample_duration = _row_duration(sample)
            sample_payload = sample.get(
                "payload_bytes", sample.get("size_bytes", sample.get("work", payload)))
            if sample_duration is None or sample_payload is None:
                continue
            rep = sample.get(
                "repeat_index",
                sample["_repeats"][0][0][0]
                if sample.get("_repeats") and sample["_repeats"][0] else index)
            result.append(_base_row(
                document, label, origin, collective, world, scope, pair,
                sample_payload, sample_duration, default_dtype,
                sample.get("single", row.get("single", document.get("single"))),
                _merge_obs(row, sample), run_id=sample.get("run_id", run_id),
                repeat_index=rep, split=_split(sample.get("split", split)),
                requested=default_requested, observed=default_observed,
                evidence=default_evidence))
        return result
    return [_base_row(
        document, label, origin, collective, world, scope, pair, payload,
        duration, default_dtype, row.get("single", document.get("single")),
        row, run_id=run_id, repeat_index=repeat, split=split,
        requested=default_requested, observed=default_observed,
        evidence=default_evidence)]


def _add_series(rows, document, label, origin, dtype, common, series, single=False):
    collective, world, scope, pair, points = series
    collective = _collective(collective)
    pair = pair or _pair("", collective, scope, world)
    for payload, duration in points:
        rows.append(_base_row(
            document, label, origin, collective, world, scope, pair, payload,
            duration, dtype, bool(single or document.get("single")), **common))


def _campaign_conditions(document, label, origin):
    protocol = (document.get("protocol")
                if isinstance(document.get("protocol"), Mapping) else {})
    dtype = _dtype(document.get("dtype") or protocol.get("dtype"))
    world = document.get("world_size") or document.get("ranks")
    scope = _explicit_scope(document.get("placement_scope"))
    observed = document.get(
        "observed_expansion_mode",
        document.get("observed_execution", "not_observed"))
    requested = document.get("requested_expansion_mode", "unspecified")
    evidence = document.get("profiler_evidence")
    common = {
        "run_id": document.get("run_id") or label,
        "repeat_index": document.get("repeat_index", 0),
        "split": _split(document.get("split")),
        "requested": requested, "observed": observed, "evidence": evidence,
    }
    rows = []
    for row in document.get("conditions") or document.get("comm_conditions") or ():
        rows.extend(_condition_rows(
            row, document, label, origin, dtype, world, scope, None,
            observed, requested, evidence))
    for series in document.get("series") or ():
        _add_series(rows, document, label, origin, dtype, common, series)
    raw_points = document.get("raw_points") or ()
    if raw_points:
        lane = {}
        for index, point in enumerate(raw_points):
            _validate_row(point, f"{label}:raw_points[{index}]")
            observed_point = point.get(
                "observed_expansion_mode", point.get("observed_execution"))
            resolved_point = resolve_expansion_regime(
                point.get("requested_expansion_mode", requested),
                point.get("observed_expansion_mode",
                          point.get("observed_execution", observed)),
                profiler_evidence=point.get("profiler_evidence", evidence))
            sel = _selectors(document, point)
            key = (
                _dtype(point.get("dtype")), int(point.get("ep")),
                int(point.get("link_bytes")),
                bool(point.get("single", document.get("single", False))),
                _split(point.get("split", document.get("split"))),
                point.get("run_id"), point.get("repeat_index"),
                observed_point, point.get("requested_expansion_mode"),
                point.get("algorithm"),
                point.get("boundary", document.get("boundary")),
                resolved_point["regime"], resolved_point["unconfirmed_tag"],
                resolved_point["profiler_confirmed"],
                (sel["stack"], sel["cann"], sel["algorithm"],
                 _protocol_key(sel["protocol"])),
            )
            lane.setdefault(key, (point, []))[1].append(float(point["ms"]))
        for key, (point, values) in sorted(lane.items(), key=str):
            (lane_dtype, series_world, payload, single, split, run_id,
             repeat_index, observed_point, requested_point,
             _, _, _, _, _, _) = key
            rows.append(_base_row(
                document, label, origin, "all_to_allv", series_world,
                _scope("", series_world), None, payload, median(values),
                lane_dtype, single, point,
                run_id=run_id or common["run_id"],
                repeat_index=(repeat_index if repeat_index is not None
                              else common["repeat_index"]),
                split=split if split is not None else common["split"],
                requested=(requested_point if requested_point is not None
                           else common["requested"]),
                observed=(observed_point if observed_point is not None
                          else common["observed"]),
                evidence=point.get("profiler_evidence", evidence)))
    for lane_dtype, series_world, points in document.get("all_to_allv_series") or ():
        _add_series(
            rows, document, label, origin, lane_dtype, common,
            ("all_to_allv", series_world, _scope("", series_world), None, points))
    for series in document.get("single_anchors") or ():
        _add_series(rows, document, label, origin, dtype, common, series, True)
    return rows


def ingest_a5_campaign_data(source=None):
    """Normalize communication campaigns into steady-state conditions and launch anchors."""
    campaigns, conditions, anchors, chains = [], [], [], []
    for document, origin in _source_documents(source):
        label = str(document.get("label") or document.get("run_id")
                    or document.get("campaign") or Path(origin).stem)
        if (document.get("schema_version") == "mindie-mc2-launch/1"
                and origin not in {"memory", "embedded"}):
            label = f"{Path(origin).parent.name}:{label}"
        _validate_campaign(document, label)
        _require_a5(
            document.get("hardware_key") or document.get("sku") or A5_HARDWARE_KEY,
            document.get("arch_family"))
        protocol = document.get("protocol")
        campaigns.append({
            "label": label,
            "status": document.get("status", document.get("state", "completed")),
            "kind": document.get("kind", "calibration"),
            "hardware_key": document.get("hardware_key", A5_HARDWARE_KEY),
            "arch_family": document.get("arch_family", A5_ARCH_FAMILY),
            "stack": document.get("stack"), "date": document.get("date"),
            "cann": document.get("cann") or (document.get("machine") or {}).get("cann"),
            "algorithm": (document.get("algorithm")
                          or (protocol.get("algorithm")
                              if isinstance(protocol, Mapping) else None)),
            "protocol": protocol, "placement": document.get("placement"),
            "nodes": list(document.get("nodes") or ()),
            "dtype": _dtype(document.get("dtype")),
            "observed_expansion_mode": _observed_mode(
                document.get("observed_expansion_mode")),
            "canary": document.get("canary"), "source": origin,
        })
        for row in _campaign_conditions(document, label, origin):
            (anchors if row["single"] else conditions).append(row)
        chains.extend(_campaign_chains(document, label, origin))
    conditions, anchors = _dedupe_conditions(conditions), _dedupe_conditions(anchors)
    chains = _dedupe_chains(chains)
    campaigns.sort(key=lambda item: item["label"])
    return {
        "schema": "a5-campaign-ingestion/1", "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "campaigns": campaigns,
        "conditions": conditions, "rows": conditions,
        "single_op_anchors": anchors, "mc2_chains": chains,
        "campaign_labels": [campaign["label"] for campaign in campaigns],
    }


def _coerce_ingested(ingested):
    if ingested is None:
        return ingest_a5_campaign_data()
    if isinstance(ingested, Mapping) and ingested.get("schema") == "a5-campaign-ingestion/1":
        return ingested
    return ingest_a5_campaign_data(ingested)
