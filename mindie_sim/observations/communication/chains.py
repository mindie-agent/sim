"""MC2 chain parsing and repeat/split-aware observation deduplication.

[Role] Normalize decomposed MC2 chain rows from tuple, mapping, and per-case
  median tables, and deduplicate condition and chain observations without
  losing repeat or split conflicts.
[Invariants] Non-positive chain totals and failed statuses raise. Conflicting
  repeat timings or train/heldout splits on the same observation identity raise
  immediately; exact duplicates collapse idempotently.
[Timeouts & Locks] Pure in-memory normalization and deduplication; no I/O or
  locks.
[Side-Effect Policy] Pure functions over in-memory row lists.
"""
from __future__ import annotations

from collections import defaultdict
import math
from statistics import median
from typing import Mapping

from .normalization import (
    _canonical_semantic,
    _dtype,
    _protocol_key,
    _validate_row,
    resolve_expansion_regime,
)
from .streaming import _fold_timings, _weighted_median

__all__ = [
    "_bf16_chains",
    "_campaign_chains",
    "_chain_record",
    "_chain_row",
    "_dedupe_chains",
    "_dedupe_conditions",
]


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

        def pick(kind, stage):
            values = [value for value in (stages.get(kind, {}).get(stage) or {}).values()
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
