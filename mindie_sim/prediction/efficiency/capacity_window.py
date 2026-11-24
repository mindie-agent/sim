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

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.model import RequestSpec
from mindie_sim.schema.speculative import SpeculativeSpec

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
    rows: list[dict[str, Any]] = []
    for replica in range(int(p.dp)):
        physical = memory.physical_usage(replica)["hbm_bytes_per_stage_rank"]
        for stage in range(int(p.pp)):
            for rank, plan in enumerate(memory.rank_capacity_plans(
                    replica, stage, include_cache=False)):
                report = plan.to_dict()
                allocations = {str(item["name"]): float(item["bytes"])
                               for item in report.get("allocations", ())}
                index = (replica * int(p.pp) + stage) * width + rank
                in_plan = (allocations.get("selected_plan", 0.0)
                           + allocations.get("shared_pages", 0.0)
                           + allocations.get("grants", 0.0))
                occupied = float(physical[stage][rank]) if rank < len(physical[stage]) else 0.0
                external_held = max(0.0, occupied - in_plan)
                rows.append({
                    "replica": replica, "stage": stage, "rank": rank,
                    "basis": str(report.get("basis", "per_rank_stage_physical_allocations")),
                    "capacity_bytes": float(report.get("capacity_bytes") or 0.0),
                    "fixed_bytes": float(report.get("peak_bytes") or 0.0),
                    "external_held_bytes": round(external_held, HBM_BYTES_PRECISION_DIGITS),
                    "supply_bytes": round(float(report.get("headroom_bytes") or 0.0)
                                          - external_held, HBM_BYTES_PRECISION_DIGITS),
                    "workspace_bytes": allocations.get("activation_workspace", 0.0),
                    "fixed_allocations": allocations,
                    "hbm_bytes_per_die": (float(dies[index].hbm_bytes)
                                          if index < len(dies) else 0.0),
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
        lookahead_tokens = max(0, int(ledger.async_in_flight_lookahead_steps)) * q_tokens
        steps_per_wave = max(1, math.ceil(max(1, output_eff) / emit_tokens))
        retiring_waves = (max(1, math.ceil(delayed_steps / steps_per_wave))
                          if delayed_steps > 0 else 0)
        active_tokens = max(1, context + max(q_tokens, output_eff) + lookahead_tokens)
        retiring_tokens = max(1, context + max(q_tokens, output_eff))
        wave_tokens = max(1, context + min(
            max(q_tokens, output_eff), max(1, delayed_steps) * q_tokens) + lookahead_tokens)
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
        "peak_bytes": round(max((float(row["peak_kv_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "headroom_bytes": round(min((float(row["headroom_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "active_headroom_bytes": round(min((float(row["active_headroom_bytes"]) for row in rows), default=0.0), HBM_BYTES_PRECISION_DIGITS),
        "stages": stages, "stage_rank": rows, "fits": fits,
    }


def _capacity_window_rank_rows(
    scaled, ledger, phase, context, q_tokens, b, delayed_steps,
    active_tokens, retiring_tokens, wave_tokens, staggered_delayed_b, wave_delayed_b,
):
    shared_prefix = getattr(
        scaled.runtime, "prefix_cache_mode", "private") == "shared"
    prefix_tokens = (
        int(scaled.workload.requests[0].prefix_tokens)
        if shared_prefix and scaled.workload.requests else 0)
    if shared_prefix:
        from ..capacity.features import batch_cache_bytes

        def batch_stage_bytes(sequences: int, tokens: int) -> list[float]:
            if sequences <= 0:
                return [0.0] * int(scaled.runtime.parallel.pp)
            if phase == "decode":
                # The KV horizon grows with emitted tokens, but the verified
                # query width stays q_tokens; passing the horizon as q_len
                # would charge one recurrent checkpoint per future token.
                seq_q = max(1, int(q_tokens))
                seq_ctx = max(0, int(tokens) - seq_q)
            else:
                seq_q = max(1, int(tokens) - context)
                seq_ctx = context
            return [float(value) for value in batch_cache_bytes(
                scaled, sequences=sequences, context=seq_ctx,
                tokens_per_seq=seq_q,
                prefix_tokens=min(prefix_tokens, seq_ctx),
                phase=phase, padded=True).total_bytes]

        kv_active = batch_stage_bytes(b, active_tokens)
        kv_retiring = (batch_stage_bytes(staggered_delayed_b, retiring_tokens)
                       if delayed_steps > 0 else [0.0] * len(kv_active))
        kv_wave = (batch_stage_bytes(b, wave_tokens)
                   if delayed_steps > 0 else list(kv_active))
        kv_wave_retiring = (
            batch_stage_bytes(wave_delayed_b, retiring_tokens)
            if wave_delayed_b > 0 else [0.0] * len(kv_active))
    else:
        def ledger_stage_bytes(tokens: int) -> list[float]:
            # Decode always forwards the scheduled verify width; the ledger default only
            # applies to prefill windows.
            width = q_tokens if phase == "decode" else None
            return [float(value) for value in ledger.dp_cohort_cache_bytes(
                tokens, padded=True, query_tokens=width)]

        kv_active = ledger_stage_bytes(active_tokens)
        kv_retiring = (ledger_stage_bytes(retiring_tokens)
                       if delayed_steps > 0 else [0.0] * len(kv_active))
        kv_wave = (ledger_stage_bytes(wave_tokens)
                   if delayed_steps > 0 else list(kv_active))
        kv_wave_retiring = kv_retiring
    rows = _stage_rank_capacity_facts(ledger)
    for row in rows:
        stage = int(row["stage"])
        if shared_prefix:
            active = kv_active[stage]
            retiring = kv_retiring[stage]
            batch_wave = kv_wave[stage]
            wave_retiring = kv_wave_retiring[stage]
        else:
            active = b * kv_active[stage]
            retiring = staggered_delayed_b * kv_retiring[stage]
            batch_wave = b * kv_wave[stage]
            wave_retiring = wave_delayed_b * kv_retiring[stage]
        peak = max(active + retiring, batch_wave + wave_retiring)
        used = float(row["fixed_bytes"]) + float(row["external_held_bytes"]) + peak
        row.update({
            "active_kv_bytes": round(active, HBM_BYTES_PRECISION_DIGITS),
            "retiring_kv_bytes": round(retiring, HBM_BYTES_PRECISION_DIGITS),
            "batch_wave_kv_bytes": round(batch_wave, HBM_BYTES_PRECISION_DIGITS),
            "wave_retiring_kv_bytes": round(wave_retiring, HBM_BYTES_PRECISION_DIGITS),
            "peak_kv_bytes": round(peak, HBM_BYTES_PRECISION_DIGITS),
            "headroom_bytes": round(float(row["supply_bytes"]) - peak, HBM_BYTES_PRECISION_DIGITS),
            "active_headroom_bytes": round(float(row["supply_bytes"]) - active, HBM_BYTES_PRECISION_DIGITS),
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


def _resolve_speculative_tokens(case: CaseSpec, *, output_budget: int, tokens_per_seq: int | None, emitted_tokens_per_seq: float | None) -> tuple[int, float]:
    """Resolve query tokens per sequence and committed tokens per step aligned with DraftPolicy."""
    req = RequestSpec(request_id="sla", prompt_tokens=1, output_tokens=max(1, output_budget), arrival_ms=0.0)
    dp_pol = DraftPolicy.for_request(case, req)
    policy_proposal = dp_pol.proposal if dp_pol.proposal > 0 else 0
    policy_mean = dp_pol.mean if dp_pol.proposal > 0 else 1.0
    resolved_q = int(tokens_per_seq) if tokens_per_seq is not None else (1 + policy_proposal if policy_proposal > 0 else 1)
    if isinstance(tokens_per_seq, bool) or resolved_q < 1:
        raise ValueError(f"tokens_per_seq_must_be_positive_int:{tokens_per_seq}")
    if emitted_tokens_per_seq is not None:
        return resolved_q, _check_pos_float("emitted_tokens_per_seq", emitted_tokens_per_seq)
    if resolved_q == 1 and policy_proposal > 0 and tokens_per_seq == 1:
        return resolved_q, 1.0
    return resolved_q, (min(float(resolved_q), float(policy_mean)) if (policy_proposal > 0 and resolved_q > 1) else float(resolved_q))


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
