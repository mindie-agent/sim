"""Per-rank HBM capacity and service lifecycle windows.

[Role] Project KV lifetimes against the allocation ledger, resolve draft
  geometry and select jointly feasible SLA candidates.
[Invariants] Capacity is checked against simultaneous per-rank occupancy,
  including delayed release and shared prefix ownership when configured.
[Timeouts & Locks] In-memory projections and finite concurrency searches.
[Side-Effect Policy] Cases and allocation ledgers are read-only. Baseline gain
  calculation updates the supplied result rows.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from mindie_sim.catalog.speculative_state import resolve_speculative_proposals
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.model import RequestSpec
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.workload import RequestView

from ..capacity import Memory
from ..scheduler import DraftPolicy
from ..structure import parse_case
from .analysis import scale_case_hardware
from .peaks import DEFAULT_CONTEXT, DEFAULT_OUTPUT_BUDGET, _check_pos_float


HBM_BYTES_PRECISION_DIGITS = 3
HBM_UTILIZATION_PRECISION_DIGITS = 9


def _stage_rank_capacity_facts(memory: Memory) -> list[dict[str, Any]]:
    """Read the T08 per-rank supply from ``rank_capacity_plans``/``to_dict``.

    ``headroom_bytes`` is the AllocationPlan supply after resident weights,
    workspace, temporary representation, reserves and any live selected/shared/
    grant allocations.  It is therefore the same bound the T08 admission ledger
    checks, not a second private allocator.
    """
    case, p = memory.case, memory.case.runtime.parallel
    dies, width = case.machine.dies, max(1, int(p.tp) * int(p.pcp))
    dp_inner = int(p.dp)
    ext_dp = max(1, int(getattr(p, "external_dp", 1) or 1))
    util = float(case.runtime.gpu_memory_utilization)
    rows: list[dict[str, Any]] = []
    for ext_idx in range(ext_dp):
        for replica in range(dp_inner):
            physical = memory.physical_usage(replica)["hbm_bytes_per_stage_rank"]
            phys_replica = ext_idx * dp_inner + replica
            for stage in range(int(p.pp)):
                for rank, plan in enumerate(memory.rank_capacity_plans(
                        replica, stage, include_cache=False)):
                    report = plan.to_dict()
                    allocations = {str(item["name"]): float(item["bytes"])
                                   for item in report.get("allocations", ())}
                    index = (phys_replica * int(p.pp) + stage) * width + rank
                    in_plan = (allocations.get("selected_plan", 0.0)
                               + allocations.get("shared_pages", 0.0)
                               + allocations.get("grants", 0.0)
                               + allocations.get("grant_readers", 0.0))
                    occupied = (float(physical[stage][rank])
                                if ext_idx == 0 and rank < len(physical[stage]) else 0.0)
                    external_held = max(0.0, occupied - in_plan)
                    fixed_bytes = float(report.get("peak_bytes") or 0.0)
                    die_hbm = float(dies[index].hbm_bytes) if index < len(dies) else 0.0
                    if ext_idx == 0:
                        cap_bytes = float(report.get("capacity_bytes") or 0.0)
                        headroom = float(report.get("headroom_bytes") or 0.0)
                    else:
                        cap_bytes = die_hbm * util
                        headroom = cap_bytes - fixed_bytes
                    rows.append({
                        "replica": phys_replica, "stage": stage, "rank": rank,
                        "basis": str(report.get("basis", "per_rank_stage_physical_allocations")),
                        "capacity_bytes": cap_bytes,
                        "fixed_bytes": fixed_bytes,
                        "external_held_bytes": round(external_held, HBM_BYTES_PRECISION_DIGITS),
                        "supply_bytes": round(headroom - external_held, HBM_BYTES_PRECISION_DIGITS),
                        "workspace_bytes": allocations.get("activation_workspace", 0.0),
                        "fixed_allocations": allocations,
                        "hbm_bytes_per_die": die_hbm,
                    })
    return rows


def capacity_window(
    case: CaseSpec | Mapping[str, Any], *, phase: str = "decode",
    context: int = DEFAULT_CONTEXT,
    tokens_per_seq: int = 1, sequences: int = 1,
    output_budget: int = DEFAULT_OUTPUT_BUDGET,
    emitted_tokens_per_seq: float | None = None, async_scheduling: bool | None = None,
    hbm_capacity_scale: float = 1.0, memory: Memory | None = None,
) -> dict[str, Any]:
    """Project one candidate B onto T08's per-rank supply and the real KV windows.

    The returned record binds the stage context/query/emitted facts used by the
    latency, SLA, capacity, counterfactual, advise, guardrail, topology and PD
    consumers.  ``supply_bytes`` is already net of weights, workspace and fixed
    reserve; ``active_kv_bytes`` is the executing window, ``retiring_kv_bytes``
    the delayed-release hold, and ``peak_bytes`` the largest simultaneous
    occupancy.  ``active_headroom_bytes`` is retained only as an explicitly
    labelled active-window value, never as the complete-service headroom.
    """
    if phase not in {"decode", "prefill"}:
        raise ValueError(f"capacity_window_phase_invalid:{phase}")
    if isinstance(context, bool) or not isinstance(context, int) or context < 0:
        raise ValueError(f"capacity_window_context_invalid:{context}")
    if isinstance(tokens_per_seq, bool) or not isinstance(tokens_per_seq, int) or tokens_per_seq < 1:
        raise ValueError(f"capacity_window_tokens_per_seq_invalid:{tokens_per_seq}")
    if isinstance(sequences, bool) or not isinstance(sequences, int) or sequences < 1:
        raise ValueError(f"capacity_window_sequences_invalid:{sequences}")
    if isinstance(output_budget, bool) or not isinstance(output_budget, int) or output_budget < 0:
        raise ValueError(f"capacity_window_output_budget_invalid:{output_budget}")
    base_case = case if isinstance(case, CaseSpec) else parse_case(case)
    if async_scheduling is not None:
        base_case = replace(base_case, runtime=replace(
            base_case.runtime, async_scheduling=bool(async_scheduling)))
    scaled = (scale_case_hardware(base_case, hbm_capacity_scale=hbm_capacity_scale)
              if hbm_capacity_scale != 1.0 else base_case)
    q_tokens, b = max(1, int(tokens_per_seq)), int(sequences)
    workspace_tokens = max(1, b * q_tokens)
    ledger = memory if memory is not None else Memory(scaled, workspace_tokens=workspace_tokens)
    if memory is not None and memory.case != scaled:
        raise ValueError("capacity_window_ledger_case_mismatch")
    prefill = phase == "prefill"
    output_eff = 0 if prefill else max(0, int(output_budget))
    if prefill:
        emit_tokens = float(q_tokens)
        delayed_steps = 0
        lookahead_tokens = 0
        steps_per_wave = 1
        retiring_waves = 0
        staggered_delayed_b = 0
        wave_delayed_b = 0
        active_tokens = context + q_tokens
        retiring_tokens = active_tokens
        wave_tokens = active_tokens
    else:
        emit_tokens = (float(q_tokens) if emitted_tokens_per_seq is None
                       else max(1e-6, float(emitted_tokens_per_seq)))
        delayed_steps = max(0, int(ledger.async_delayed_release_steps))
        raw_lookahead = max(0, int(ledger.async_in_flight_lookahead_steps)) * q_tokens
        max_span_tokens = max(q_tokens, output_eff)
        lookahead_tokens = min(raw_lookahead, max(0, max_span_tokens - q_tokens))
        steps_per_wave = max(1, math.ceil(max(1, output_eff) / emit_tokens))
        retiring_waves = (max(1, math.ceil(delayed_steps / steps_per_wave))
                          if delayed_steps > 0 else 0)
        active_tokens = max(1, context + max_span_tokens)
        retiring_tokens = max(1, context + max_span_tokens)
        wave_tokens = max(1, context + min(
            max_span_tokens, max(1, delayed_steps) * q_tokens + lookahead_tokens))
        staggered_delayed_b = 0
        wave_delayed_b = 0
        if b >= 2 and delayed_steps > 0 and output_eff > 0:
            staggered_delayed_b = max(1, math.ceil(b * delayed_steps / steps_per_wave))
            wave_delayed_b = retiring_waves * b
    rows = _capacity_window_rank_rows(
        scaled, ledger, phase, context, q_tokens, b, delayed_steps,
        active_tokens, retiring_tokens, wave_tokens,
        staggered_delayed_b, wave_delayed_b)
    stages: list[dict[str, Any]] = []
    for stage_index in range(int(scaled.runtime.parallel.pp)):
        stage_rows = [row for row in rows if int(row["stage"]) == stage_index]
        if not stage_rows:
            continue
        active = max(float(row["active_kv_bytes"]) for row in stage_rows)
        retiring = max(float(row["retiring_kv_bytes"]) for row in stage_rows)
        batch_wave = max(float(row["batch_wave_kv_bytes"]) for row in stage_rows)
        wave_retiring = max(float(row["wave_retiring_kv_bytes"]) for row in stage_rows)
        staging = max(float(row["staging_bytes"]) for row in stage_rows)
        peak = max(float(row["peak_kv_bytes"]) for row in stage_rows)
        supply = min(float(row["supply_bytes"]) for row in stage_rows)
        headroom = min(float(row["headroom_bytes"]) for row in stage_rows)
        stages.append({
            "stage": stage_index,
            "supply_bytes": round(supply, HBM_BYTES_PRECISION_DIGITS),
            "capacity_bytes": round(max(float(row["capacity_bytes"]) for row in stage_rows), HBM_BYTES_PRECISION_DIGITS),
            "fixed_bytes": round(max(float(row["fixed_bytes"]) for row in stage_rows), HBM_BYTES_PRECISION_DIGITS),
            "external_held_bytes": round(max(float(row["external_held_bytes"]) for row in stage_rows), HBM_BYTES_PRECISION_DIGITS),
            "workspace_bytes": round(max(float(row["workspace_bytes"]) for row in stage_rows), HBM_BYTES_PRECISION_DIGITS),
            "active_kv_bytes": round(active, HBM_BYTES_PRECISION_DIGITS),
            "retiring_kv_bytes": round(retiring, HBM_BYTES_PRECISION_DIGITS),
            "batch_wave_kv_bytes": round(batch_wave, HBM_BYTES_PRECISION_DIGITS),
            "wave_retiring_kv_bytes": round(wave_retiring, HBM_BYTES_PRECISION_DIGITS),
            "staging_bytes": round(staging, HBM_BYTES_PRECISION_DIGITS),
            "peak_bytes": round(peak, HBM_BYTES_PRECISION_DIGITS),
            "used_bytes": round(max(float(row["used_bytes"]) for row in stage_rows), HBM_BYTES_PRECISION_DIGITS),
            "headroom_bytes": round(headroom, HBM_BYTES_PRECISION_DIGITS),
            "active_headroom_bytes": round(min(float(row["active_headroom_bytes"]) for row in stage_rows), HBM_BYTES_PRECISION_DIGITS),
            "hbm_bytes_per_die": round(min(float(row["hbm_bytes_per_die"]) for row in stage_rows), HBM_BYTES_PRECISION_DIGITS),
            "hbm_utilization_ratio": round(max(float(row["hbm_utilization_ratio"]) for row in stage_rows), HBM_UTILIZATION_PRECISION_DIGITS),
        })
    fits = bool(rows) and all(float(row["peak_kv_bytes"]) <=
                              float(row["supply_bytes"]) + max(1e-9, abs(float(row["supply_bytes"])) * 1e-12)
                              for row in rows)
    return {
        "schema": "mindie-sim.capacity-window/1",
        "phase": phase, "context_tokens": int(context),
        "query_tokens_per_sequence": q_tokens, "emitted_tokens_per_sequence": emit_tokens,
        "output_budget": int(output_budget), "sequences": b,
        "async_scheduling": bool(scaled.runtime.async_scheduling),
        "hbm_capacity_scale": float(hbm_capacity_scale),
        "window": {
            "active_tokens_per_sequence": int(active_tokens),
            "retiring_tokens_per_sequence": int(retiring_tokens),
            "wave_tokens_per_sequence": int(wave_tokens),
            "lookahead_tokens": int(lookahead_tokens),
            "delayed_release_steps": int(delayed_steps),
            "steps_per_wave": int(steps_per_wave),
            "retiring_waves": int(retiring_waves),
            "staggered_delayed_sequences": int(staggered_delayed_b),
            "wave_delayed_sequences": int(wave_delayed_b),
        },
        "supply_bytes": round(min((float(row["supply_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "fixed_bytes": round(max((float(row["fixed_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "external_held_bytes": round(max((float(row["external_held_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "workspace_bytes": round(max((float(row["workspace_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "weight_bytes_per_die": round(max((
            float(row["fixed_allocations"].get("weights", 0.0)) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "active_kv_bytes": round(max((float(row["active_kv_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "retiring_kv_bytes": round(max((float(row["retiring_kv_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "batch_wave_kv_bytes": round(max((float(row["batch_wave_kv_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "wave_retiring_kv_bytes": round(max((float(row["wave_retiring_kv_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "staging_bytes": round(max((float(row["staging_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "peak_bytes": round(max((float(row["peak_kv_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "headroom_bytes": round(min((float(row["headroom_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "active_headroom_bytes": round(min((float(row["active_headroom_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "stages": stages, "stage_rank": rows, "fits": fits,
    }


def _window_request_views(
    scaled: CaseSpec,
    sequences: int,
    tokens: int,
    *,
    phase: str,
    context: int,
    q_tokens: int,
    id_prefix: str,
    start_offset: int = 0,
) -> tuple[RequestView, ...]:
    """Build per-sequence ``RequestView`` objects preserving workload prefix keys."""
    if sequences <= 0:
        return ()
    if phase == "decode":
        seq_q = max(1, int(q_tokens))
        seq_ctx = max(0, int(tokens) - seq_q)
    else:
        seq_q = max(1, int(tokens) - int(context))
        seq_ctx = int(context)
    reqs = tuple(getattr(getattr(scaled, "workload", None), "requests", ()) or ())
    views: list[RequestView] = []
    for index in range(int(sequences)):
        src = reqs[(int(start_offset) + index) % len(reqs)] if reqs else None
        pfx_tok = (
            min(int(getattr(src, "prefix_tokens", 0) or 0), seq_ctx)
            if src is not None else 0
        )
        pfx_key = str(getattr(src, "prefix_cache_key", "") or "") if src is not None else ""
        views.append(RequestView(
            request_id=f"{id_prefix}{index}",
            q_len=seq_q,
            context=seq_ctx,
            samples=True,
            phase=phase,
            prefix_tokens=pfx_tok,
            prefix_cache_key=pfx_key,
        ))
    return tuple(views)


def _receipt_dynamic_by_rank(
    receipt: Any,
    *,
    include_staging: bool,
    c8_by_key: Mapping[tuple[int, int, int], float],
) -> dict[tuple[int, int, int], tuple[float, float]]:
    """Extract per-``(replica, stage, rank)`` projected ``(cache_bytes, staging_bytes)``."""
    out: dict[tuple[int, int, int], tuple[float, float]] = {}
    for die in receipt.dies:
        key = (int(die.replica), int(die.stage), int(die.rank))
        c8_b = c8_by_key.get(key, 0.0)
        cache_b = (
            max(0.0, float(die.kv_cache_allocated_bytes) - c8_b)
            + float(die.index_cache_allocated_bytes)
            + float(die.recurrent_state_bytes)
            + float(die.draft_cache_bytes)
        )
        staging_b = float(die.staging_bytes) if include_staging else 0.0
        out[key] = (cache_b, staging_b)
    return out


def _capacity_window_rank_rows(
    scaled, ledger, phase, context, q_tokens, b, delayed_steps,
    active_tokens, retiring_tokens, wave_tokens, staggered_delayed_b, wave_delayed_b,
):
    rows = _stage_rank_capacity_facts(ledger)
    dp_inner = max(1, int(scaled.runtime.parallel.dp))
    c8_by_key = {
        (int(row["replica"]) % dp_inner, int(row["stage"]), int(row["rank"])): float(
            row["fixed_allocations"].get("c8_static_channel_scale", 0.0)
        )
        for row in rows
    }
    include_staging = ledger.kvc_hbm_ratio < 1.0
    shared_prefix = getattr(
        scaled.runtime, "prefix_cache_mode", "private") == "shared"
    zero_map: dict[tuple[int, int, int], tuple[float, float]] = {}

    def _project_views(views: tuple[RequestView, ...]) -> dict[tuple[int, int, int], tuple[float, float]]:
        if not views:
            return zero_map
        return _receipt_dynamic_by_rank(
            ledger.resident_memory_receipt(phase=phase, request_views=views),
            include_staging=include_staging,
            c8_by_key=c8_by_key,
        )

    def _project_uniform(tokens: int, sequences: int) -> dict[tuple[int, int, int], tuple[float, float]]:
        if sequences <= 0:
            return zero_map
        width = q_tokens if phase == "decode" else None
        return _receipt_dynamic_by_rank(
            ledger.resident_memory_receipt(
                phase=phase, sequence_tokens=int(tokens),
                batch_per_dp=int(sequences), query_tokens=width,
            ),
            include_staging=include_staging,
            c8_by_key=c8_by_key,
        )

    if shared_prefix:
        act_views = _window_request_views(
            scaled, b, active_tokens, phase=phase, context=context,
            q_tokens=q_tokens, id_prefix="act", start_offset=0)
        act_map = _project_views(act_views)
        if delayed_steps > 0:
            ret_views = _window_request_views(
                scaled, staggered_delayed_b, retiring_tokens, phase=phase,
                context=context, q_tokens=q_tokens, id_prefix="ret",
                start_offset=b)
            wav_views = _window_request_views(
                scaled, b, wave_tokens, phase=phase, context=context,
                q_tokens=q_tokens, id_prefix="wav", start_offset=0)
            wret_views = _window_request_views(
                scaled, wave_delayed_b, retiring_tokens, phase=phase,
                context=context, q_tokens=q_tokens, id_prefix="wret",
                start_offset=b)
            ret_map = _project_views(ret_views)
            wav_map = _project_views(wav_views)
            wret_map = _project_views(wret_views)
            stag_union = _project_views(act_views + ret_views) if ret_views else act_map
            wave_union = _project_views(wav_views + wret_views) if wret_views else wav_map
        else:
            ret_map, wav_map, wret_map = zero_map, act_map, zero_map
            stag_union, wave_union = act_map, act_map
    else:
        act_map = _project_uniform(active_tokens, b)
        if delayed_steps > 0:
            ret_map = _project_uniform(retiring_tokens, staggered_delayed_b)
            wav_map = _project_uniform(wave_tokens, b)
            wret_map = _project_uniform(retiring_tokens, wave_delayed_b)
        else:
            ret_map, wav_map, wret_map = zero_map, act_map, zero_map
        stag_union, wave_union = None, None

    for row in rows:
        key = (int(row["replica"]) % dp_inner, int(row["stage"]), int(row["rank"]))
        active, active_stg = act_map.get(key, (0.0, 0.0))
        retiring, _ = ret_map.get(key, (0.0, 0.0))
        batch_wave, wave_stg = wav_map.get(key, (0.0, 0.0))
        wave_retiring, _ = wret_map.get(key, (0.0, 0.0))
        if stag_union is not None and wave_union is not None:
            stag_cache, _ = stag_union.get(key, (0.0, 0.0))
            wave_cache, _ = wave_union.get(key, (0.0, 0.0))
            stag_total = stag_cache + active_stg
            wave_total = wave_cache + wave_stg
        else:
            stag_total = active + retiring + active_stg
            wave_total = batch_wave + wave_retiring + wave_stg
        if wave_total > stag_total:
            peak, staging = wave_total, wave_stg
        else:
            peak, staging = stag_total, active_stg
        used = float(row["fixed_bytes"]) + float(row["external_held_bytes"]) + peak
        row.update({
            "active_kv_bytes": round(active, HBM_BYTES_PRECISION_DIGITS),
            "retiring_kv_bytes": round(retiring, HBM_BYTES_PRECISION_DIGITS),
            "batch_wave_kv_bytes": round(batch_wave, HBM_BYTES_PRECISION_DIGITS),
            "wave_retiring_kv_bytes": round(wave_retiring, HBM_BYTES_PRECISION_DIGITS),
            "staging_bytes": round(staging, HBM_BYTES_PRECISION_DIGITS),
            "peak_kv_bytes": round(peak, HBM_BYTES_PRECISION_DIGITS),
            "headroom_bytes": round(float(row["supply_bytes"]) - peak, HBM_BYTES_PRECISION_DIGITS),
            "active_headroom_bytes": round(float(row["supply_bytes"]) - active - active_stg, HBM_BYTES_PRECISION_DIGITS),
            "used_bytes": round(used, HBM_BYTES_PRECISION_DIGITS),
            "hbm_utilization_ratio": round(max(0.0, min(1.0, (
                used) / max(float(row["hbm_bytes_per_die"]), 1e-12))), HBM_UTILIZATION_PRECISION_DIGITS),
        })
    return rows


def _vram_max_sequences_per_dp(
    case: CaseSpec, *, context: int, tokens_per_seq: int, output_budget: int,
    hbm_capacity_scale: float = 1.0, emitted_tokens_per_seq: float | None = None,
    async_scheduling: bool | None = None, phase: str = "decode",
    memory: Memory | None = None,
) -> int:
    """Solve max concurrent sequences per DP replica with the shared capacity window."""
    base_case = replace(case, runtime=replace(case.runtime, async_scheduling=bool(async_scheduling))) if async_scheduling is not None else case
    scaled = scale_case_hardware(base_case, hbm_capacity_scale=hbm_capacity_scale) if hbm_capacity_scale != 1.0 else base_case
    q_tokens = max(1, int(tokens_per_seq))
    rt_cap = min(int(scaled.runtime.max_num_seqs), max(1, int(scaled.runtime.max_num_batched_tokens) // q_tokens))

    def fits(b: int) -> bool:
        batch_memory = (
            memory if memory is not None
            and int(getattr(memory, "workspace_tokens", 0) or 0) == max(1, b * q_tokens)
            else None
        )
        return capacity_window(
            scaled, phase=phase, context=context, tokens_per_seq=q_tokens,
            sequences=b, output_budget=output_budget,
            emitted_tokens_per_seq=emitted_tokens_per_seq,
            async_scheduling=None, memory=batch_memory,
        )["fits"]

    if not fits(1):
        return 0
    lo, hi, best = 1, max(1, rt_cap), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        best, lo, hi = (mid, mid + 1, hi) if fits(mid) else (best, lo, mid - 1)
    return best


def _resolve_speculative_tokens(
    case: CaseSpec, *, output_budget: int | None = None, tokens_per_seq: int | None,
    emitted_tokens_per_seq: float | None, phase: str = "decode",
    request_id: str | None = None,
) -> tuple[int, float]:
    """Resolve query tokens per sequence and committed tokens per step aligned with DraftPolicy."""
    if phase == "prefill":
        if tokens_per_seq is None:
            reqs = tuple(getattr(getattr(case, "workload", None), "requests", ()) or ())
            if reqs and reqs[0].request_id != "__structure_probe__":
                prompt_tok = int(getattr(reqs[0], "prompt_tokens", 1) or 1)
                prefix_tok = int(getattr(reqs[0], "prefix_tokens", 0) or 0)
                resolved_q = max(1, prompt_tok - prefix_tok)
            else:
                resolved_q = 1
        else:
            resolved_q = int(tokens_per_seq)
        if isinstance(tokens_per_seq, bool) or resolved_q < 1:
            raise ValueError(f"tokens_per_seq_must_be_positive_int:{tokens_per_seq}")
        return resolved_q, float(resolved_q)
    if isinstance(output_budget, bool):
        raise ValueError(f"output_budget_must_be_int_or_none:{output_budget}")
    bounded_output = int(output_budget) if (output_budget is not None and int(output_budget) >= 1) else None
    reqs = tuple(getattr(getattr(case, "workload", None), "requests", ()) or ())
    matched_req = (
        next((r for r in reqs if getattr(r, "request_id", None) == request_id), None)
        if request_id is not None else None
    )
    if matched_req is not None:
        req_out = (
            bounded_output if bounded_output is not None
            else max(1, int(getattr(matched_req, "output_tokens", 1) or 1))
        )
        req = replace(matched_req, output_tokens=req_out)
    else:
        if bounded_output is not None:
            req_out = bounded_output
        else:
            req_out = (
                max(1, int(getattr(reqs[0], "output_tokens", 1) or 1))
                if reqs and reqs[0].request_id != "__structure_probe__"
                else 1
            )
        req = RequestSpec(request_id=request_id or "sla", prompt_tokens=1, output_tokens=req_out, arrival_ms=0.0)
    dp_pol = DraftPolicy.for_request(case, req)
    policy_proposal = dp_pol.proposal if dp_pol.proposal > 0 else 0
    policy_mean = dp_pol.mean if dp_pol.proposal > 0 else 1.0
    if policy_proposal == 0:
        contract_prop = resolve_speculative_proposals(case.runtime, phase="decode")
        if contract_prop > 0:
            policy_proposal = contract_prop
            raw_mean = dict(case.runtime.mtp_acceptance_by_output_tokens).get(
                req.output_tokens, case.runtime.mtp_acceptance_length,
            )
            policy_mean = min(float(policy_proposal + 1), max(1.0, float(raw_mean)))
    max_prop = (
        min(policy_proposal, max(0, bounded_output - 1))
        if bounded_output is not None else policy_proposal
    )
    resolved_q = int(tokens_per_seq) if tokens_per_seq is not None else (1 + max_prop if max_prop > 0 else 1)
    if isinstance(tokens_per_seq, bool) or resolved_q < 1:
        raise ValueError(f"tokens_per_seq_must_be_positive_int:{tokens_per_seq}")
    if emitted_tokens_per_seq is not None:
        return resolved_q, _check_pos_float("emitted_tokens_per_seq", emitted_tokens_per_seq)
    if policy_proposal > 0:
        eff_prop = max(0, min(resolved_q - 1, max_prop))
        if eff_prop == 0:
            return resolved_q, 1.0
        if dp_pol.distribution:
            expected = sum(
                (1.0 + float(min(eff_prop, acc))) * float(prob)
                for acc, prob in enumerate(dp_pol.distribution)
            )
        elif dp_pol.trace:
            expected = sum(
                1.0 + float(min(eff_prop, acc)) for acc in dp_pol.trace
            ) / float(len(dp_pol.trace))
        else:
            expected = 1.0 + (float(policy_mean) - 1.0) * (float(eff_prop) / float(policy_proposal))
        return resolved_q, min(float(resolved_q), expected)
    return resolved_q, float(resolved_q)


def select_jointly_feasible_candidate(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the throughput-maximizing candidate that passed both SLA and capacity."""
    feasible = [record for record in records if record and bool(record.get("feasible"))]
    if not feasible:
        return None

    def _objective(record: Mapping[str, Any]) -> tuple[float, int]:
        b = int(record["b"])
        step_ms = float(record.get("step_ms") or 0.0)
        tps = (float(b) / step_ms) if step_ms > 0.0 else math.inf
        return (tps, b)

    return max(feasible, key=_objective)


def apply_sla_baseline_gains(results: Sequence[dict[str, Any]],
                            base_r: Mapping[str, Any]) -> None:
    """Populate baseline-relative throughput gains in place.

    Extracted so a caller that evaluates each counterfactual in its own worker
    (one flat pool over ``(sku, scenario)``) reproduces exactly the fields a
    single-call solve publishes. Each throughput gain requires a positive
    baseline throughput at the same boundary.
    """
    base_tps = base_r["tps_at_sla"]
    base_feas_tps = base_r["tps_feasible"]
    base_b = max(1, base_r["b_sla"])
    for row in results:
        row["tps_gain_vs_baseline"] = (
            row["tps_at_sla"] / base_tps if base_tps > 0 else None)
        row["tps_feasible_gain_vs_baseline"] = (
            row["tps_feasible"] / base_feas_tps if base_feas_tps > 0 else None)
        row["concurrency_gain_vs_baseline"] = (
            (row["b_sla"] / base_b) if base_r["b_sla"] > 0
            else ((base_r["step_ms_at_b1"] / row["step_ms_at_b1"])
                  if row["step_ms_at_b1"] > 0.0 else 0.0))
