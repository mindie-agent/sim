"""Disaggregated Prefill/Decode instance-ratio sweep.

[Role] Compose one prefill-stage rate, one KV handoff rate and one decode-stage
  rate into sustainable pipeline throughput and a Pareto frontier over instance
  counts.
[Invariants] Stage metrics come from one declared stage case each; a pipeline
  rate is the minimum stage rate. No cross-configuration metric stitching.
[Timeouts & Locks] Pure in-memory arithmetic; no I/O, devices or locks.
[Side-Effect Policy] Returns new dictionaries; inputs are never mutated.
"""
from __future__ import annotations

import math

from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    DEFAULT_SLA_SCENARIOS, _lower_and_run_graph, _resolve_speculative_tokens,
    solve_sla_sizing,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.sweep_resize import check_pos_float, check_pos_int

DEFAULT_PROMPT_TOKENS = 4096
DEFAULT_OUTPUT_TOKENS = 1024
DEFAULT_TTFT_SLA_MS = 3000.0
DEFAULT_KV_TRANSFER_BW_GBPS = 100.0
DEFAULT_SLA_MS = 30.0
MS_PER_S = 1000.0
PARETO_EPS = 1e-12


def _pool_kv_hbm_ms(parsed_pool: CaseSpec, memory: Memory, tokens: int) -> float:
    parallel = parsed_pool.runtime.parallel
    width = parallel.tp * parallel.pcp
    best = 0.0
    for stage, stage_bytes in enumerate(memory.cache_bytes(tokens, padded=False)):
        dies = (parsed_pool.machine.dies[stage * width:(stage + 1) * width]
                or (parsed_pool.machine.dies[0],))
        hbm_rate_ms = min(float(dict(die.rates)["hbm"]) for die in dies) / 1000.0
        best = max(best, float(stage_bytes) / max(hbm_rate_ms, 1e-12))
    return best


def sweep_pd_disaggregation_ratios(
    prefill_case: CaseSpec | Mapping[str, Any],
    decode_case: CaseSpec | Mapping[str, Any], *, total_dies: int,
    prompt_tokens: int = DEFAULT_PROMPT_TOKENS,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    decode_context: int | None = None,
    ttft_sla_ms: float = DEFAULT_TTFT_SLA_MS,
    tpot_sla_ms: float = DEFAULT_SLA_MS,
    kv_transfer_bw_gbps: float = DEFAULT_KV_TRANSFER_BW_GBPS,
) -> dict[str, Any]:
    """Sweep disaggregated P/D instance counts and KV handoff bottlenecks."""
    total = check_pos_int("total_dies", total_dies, minimum=1)
    prompt = check_pos_int("prompt_tokens", prompt_tokens, minimum=1)
    output = check_pos_int("output_tokens", output_tokens, minimum=1)
    eff_decode = (prompt if decode_context is None
                  else check_pos_int("decode_context", decode_context, minimum=0))
    check_pos_float("ttft_sla_ms", ttft_sla_ms)
    check_pos_float("tpot_sla_ms", tpot_sla_ms)
    check_pos_float("kv_transfer_bw_gbps", kv_transfer_bw_gbps)
    p_parsed, d_parsed = parse_case(prefill_case), parse_case(decode_case)
    p_par, d_par = p_parsed.runtime.parallel, d_parsed.runtime.parallel
    p_dies, d_dies = p_par.world_size, d_par.world_size
    p_dp, d_dp = int(p_par.dp), int(d_par.dp)
    p_mem = Memory(p_parsed, workspace_tokens=prompt)
    p_kv = p_mem.cache_bytes(prompt + 1, padded=True)
    prefill_hbm_ok = bool(
        prompt <= p_parsed.runtime.max_model_len
        and (p_parsed.runtime.enable_chunked_prefill
             or prompt <= p_parsed.runtime.max_num_batched_tokens)
        and all(capacity >= 0.0 and capacity >= kv
                for row in p_mem.capacity for capacity, kv in zip(row, p_kv)))
    p_slots = (
        min([int(p_parsed.runtime.max_num_seqs)]
            + [int(math.floor(max(0.0, float(capacity)) / float(kv)))
               for row in p_mem.capacity
               for capacity, kv in zip(row, p_kv) if kv > 0])
        if prefill_hbm_ok else 0)
    _, _, p_graph = _lower_and_run_graph(
        p_parsed, phase="prefill", tokens=prompt, context=0, sequences=1,
        scope="body", run_graph=True)
    assert p_graph is not None
    p_step = float(p_graph["duration_ms"])
    prefill_ok = bool(p_step <= ttft_sla_ms)
    d_mem = Memory(d_parsed)
    decode_tokens = output - 1
    if decode_tokens <= 0:
        kv_bytes = 0.0
        kv_ms = 0.0
        rps_p = ((p_dp * MS_PER_S) / max(p_step, 1e-12)) if (
            prefill_ok and prefill_hbm_ok) else 0.0
        handoff_tpot = 0.0
        decode_budget = tpot_sla_ms
        b_sla = 0
        b_vram = int(d_parsed.runtime.max_num_seqs)
        b_feasible = 0
        b1_step = 0.0
        decode_step = 0.0
        effective_tpot = 0.0
        first_itl = 0.0
        tpot_ok = True
        rps_d = rps_p
        raw_rps_d = rps_p
        decode_tps = rps_d * float(output)
        infeasible = ("prefill_hbm_oom" if not prefill_hbm_ok
                      else "prefill_ttft_sla_exceeded" if not prefill_ok
                      else None)
    else:
        kv_bytes = float(sum(d_mem.cache_bytes(prompt, padded=False))
                         * d_par.tp * d_par.pcp)
        wire_ms = kv_bytes / (float(kv_transfer_bw_gbps) * 1e6)
        p_hbm_ms = _pool_kv_hbm_ms(p_parsed, p_mem, prompt)
        d_hbm_ms = _pool_kv_hbm_ms(d_parsed, d_mem, prompt)
        kv_ms = max(wire_ms, p_hbm_ms, d_hbm_ms)
        per_rep = min(
            MS_PER_S / max(p_step, 1e-12),
            (float(p_slots) * MS_PER_S) / max(p_step + kv_ms, 1e-12))
        rps_p = (p_dp * per_rep) if (prefill_ok and prefill_hbm_ok) else 0.0
        handoff_tpot = kv_ms / float(decode_tokens)
        decode_budget = tpot_sla_ms - handoff_tpot
        _, d_emit_est = _resolve_speculative_tokens(
            d_parsed, output_budget=output, tokens_per_seq=None,
            emitted_tokens_per_seq=None)
        step_budget = decode_budget * max(1.0, float(d_emit_est))
        d_sla = solve_sla_sizing(
            d_parsed,
            sla_ms=(max(step_budget, PARETO_EPS) if step_budget > 0
                    else tpot_sla_ms * max(1.0, float(d_emit_est))),
            context=eff_decode, phase="decode", output_budget=output,
            scenarios=[DEFAULT_SLA_SCENARIOS[0]])
        d_scen = d_sla["scenarios"][0]
        emitted = float(d_sla["emitted_tokens_per_seq"])
        length_ok = bool(max(prompt, eff_decode) + output
                         <= d_parsed.runtime.max_model_len)
        b_vram = int(d_scen["b_vram_max"]) if length_ok else 0
        b_sla = int(d_scen["b_sla"]) if decode_budget > 0.0 else 0
        b_feasible = int(d_scen["b_feasible"])
        b1_step = float(d_scen["step_ms_at_b1"])
        decode_step = (float(d_scen["step_ms_feasible"])
                       if b_feasible >= 1
                       and d_scen.get("step_ms_feasible") is not None
                       else b1_step)
        effective_tpot = decode_step / max(emitted, 1e-12) + handoff_tpot
        first_itl = kv_ms + decode_step
        tpot_ok = bool(b_feasible >= 1 and effective_tpot <= tpot_sla_ms)
        rps_d = ((b_feasible * d_dp * emitted * MS_PER_S)
                 / (max(decode_step, 1e-12) * float(decode_tokens))
                 if tpot_ok else 0.0)
        decode_tps = rps_d * float(output)
        infeasible = ("prefill_hbm_oom" if not prefill_hbm_ok
                      else "prefill_ttft_sla_exceeded" if not prefill_ok
                      else "decode_max_model_len_exceeded" if not length_ok
                      else "decode_hbm_oom" if b_vram < 1
                      else "effective_tpot_sla_exceeded" if not tpot_ok
                      else None)
        raw_rps_d = ((max(1, int(d_scen["b_feasible"])) * d_dp
                      * emitted * MS_PER_S)
                     / (max(decode_step, 1e-12) * float(decode_tokens)))
    rows: list[dict[str, Any]] = []
    for n_p in range(1, (total // p_dies) + 1):
        max_nd = (total - n_p * p_dies) // d_dies
        for n_d in range(1, max_nd + 1):
            used = n_p * p_dies + n_d * d_dies
            r_p = n_p * rps_p
            r_kv = r_p if decode_tokens <= 0 else n_p * (
                MS_PER_S / max(kv_ms, 1e-9))
            r_d = r_p if decode_tokens <= 0 else n_d * rps_d
            feasible = bool(infeasible is None and min(r_p, r_kv, r_d) > 0.0)
            r_sys = min(r_p, r_kv, r_d) if feasible else 0.0
            sustainable = r_sys * float(output)
            if decode_tokens <= 0:
                bottleneck = "prefill"
            else:
                stage_rates = (("prefill", r_p), ("kv_transfer", r_kv),
                               ("decode", r_d if decode_budget > 0.0
                                else n_d * raw_rps_d))
                bottleneck = min(stage_rates, key=lambda item: item[1])[0]
            rows.append({
                "n_prefill": n_p, "n_decode": n_d,
                "pd_ratio": f"{n_p}P:{n_d}D",
                "prefill_dies": n_p * p_dies, "decode_dies": n_d * d_dies,
                "used_dies": used, "idle_dies": total - used,
                "prefill_rps": r_p, "kv_transfer_rps": r_kv,
                "decode_rps": r_d, "sustainable_rps": r_sys,
                "sustainable_tps": sustainable,
                "sustainable_tps_per_die": sustainable / float(used),
                "decode_step_ms": decode_step,
                "effective_tpot_ms": effective_tpot,
                "first_decode_itl_ms": first_itl,
                "bottleneck_stage": bottleneck, "feasible": feasible,
                "infeasible_reason": infeasible, "reason": infeasible,
                "metrics_boundary": "one_pipeline_stage_rate_composition",
            })
    feasible_rows = [row for row in rows if row["feasible"]]
    for row in rows:
        if not row["feasible"]:
            row["pareto_optimal"] = False
            continue
        dominated = False
        for other in feasible_rows:
            if other is row:
                continue
            ge_all = (float(other["sustainable_tps"])
                      >= float(row["sustainable_tps"]) - PARETO_EPS
                      and int(other["used_dies"]) <= int(row["used_dies"]))
            strict = (float(other["sustainable_tps"])
                      > float(row["sustainable_tps"]) + PARETO_EPS
                      or int(other["used_dies"]) < int(row["used_dies"]))
            if ge_all and strict:
                dominated = True
                break
        row["pareto_optimal"] = not dominated
    front = [row for row in rows if row.get("pareto_optimal")]
    best = max(front, key=lambda row: (row["sustainable_tps"],
                                       -row["used_dies"]), default=None)
    return {
        "schema": "mindie-sim.pd-sweep/1", "total_dies": total,
        "prompt_tokens": prompt, "output_tokens": output,
        "decode_context": eff_decode, "ttft_sla_ms": ttft_sla_ms,
        "tpot_sla_ms": tpot_sla_ms,
        "kv_transfer_bw_gbps": kv_transfer_bw_gbps,
        "kv_bytes_per_request": kv_bytes,
        "kv_transfer_ms_per_request": kv_ms, "decode_step_ms": decode_step,
        "effective_tpot_ms": effective_tpot,
        "first_decode_itl_ms": first_itl,
        "prefill_instance": {
            "dies_per_instance": p_dies, "parallel": p_par.to_dict(),
            "step_ms": p_step, "hbm_ok": prefill_hbm_ok,
            "ttft_sla_ok": prefill_ok, "rps_per_instance": rps_p,
        },
        "decode_instance": {
            "dies_per_instance": d_dies, "parallel": d_par.to_dict(),
            "b_sla": b_sla, "b_vram_max": b_vram, "b_feasible": b_feasible,
            "step_ms_at_b1": b1_step, "step_ms_feasible": decode_step,
            "effective_tpot_ms": effective_tpot,
            "first_decode_itl_ms": first_itl, "tpot_sla_ok": tpot_ok,
            "tps_per_instance": decode_tps, "rps_per_instance": rps_d,
        },
        "candidates": rows, "ratios": rows, "pareto_frontier": front,
        "best_configuration": best,
        "plan_selection": "fixed_stage_case_per_pipeline",
        "composition":
            "separate_prefill_and_decode_stage_programs_bound_by_min_stage_rate",
        "comparison_boundary":
            "stage_rates_are_composed_for_one_pipeline_configuration_not_stitched_per_metric",
    }


__all__ = ["sweep_pd_disaggregation_ratios"]
