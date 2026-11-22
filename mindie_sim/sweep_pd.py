"""Disaggregated Prefill/Decode 3-tier pooled sweep and auto-optimal deployment solver.

[Role] Compose one prefill-stage rate, one shard-aware KV handoff / 3-tier pool
  rate (``direct_blocking_d2d``, ``layerwise_overlap_d2d``, ``pooled_dram_ssd``),
  and one decode-stage rate into sustainable pipeline throughput, and jointly
  search PD Disaggregated vs Colocated deployments with MTP under user SLAs.
[Invariants] Zero guessed prompt/output/TTFT/TPOT/SSD defaults (Domain 1 fail-fast).
  KV/Indexer bytes derive strictly from ``disaggregated.handoff.compute_pd_handoff_plan``
  and default wire/host rates derive strictly from the active SKU's declared
  links and L0 ``P20`` / PCIe / DDR profile (zero hardcoded ``100 GB/s``).
  A pipeline rate is the minimum stage rate; no cross-configuration stitching.
[Timeouts & Locks] Pure in-memory bounded search and arithmetic; no I/O or locks.
[Side-Effect Policy] Returns new dictionaries; inputs are never mutated.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import math
from typing import Any

from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.disaggregated.handoff import compute_pd_handoff_plan
from mindie_sim.prediction.efficiency import (
    DEFAULT_SLA_SCENARIOS,
    _lower_and_run_graph,
    _resolve_speculative_tokens,
    solve_sla_sizing,
)
from mindie_sim.prediction.offload.policy import (
    _DEFAULT_SSD_BANDWIDTH_UTILIZATION,
    _DEFAULT_SSD_LATENCY_US,
    missing_scenario_error,
    resolve_sku_host_dma_rates,
)
from mindie_sim.prediction.speculation import compute_speculative_accept_step
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.sweep_resize import check_pos_float, check_pos_int

MS_PER_S = 1000.0
US_PER_MS = 1000.0
BYTES_PER_MS_PER_GBPS = 1_000_000.0
PARETO_EPS = 1e-12
DEFAULT_SPECULATIVE_ACCEPT_RATE = 0.70
VALID_PD_TRANSFER_MODES = frozenset({
    "direct_blocking_d2d",
    "layerwise_overlap_d2d",
    "pooled_dram_ssd",
})
AUTO_PD_TRANSFER_MODES = (
    "layerwise_overlap_d2d",
    "pooled_dram_ssd",
    "direct_blocking_d2d",
)
MAX_COLOCATED_CANDIDATES = 10
MAX_PD_STAGE_CANDIDATES = 6
_POW2_WIDTHS = (1, 2, 4, 8, 16)
_PREFILL_PCP_WIDTHS = (1, 2, 4)
PREFILL_PCP_MIN_PROMPT_TOKENS = 1024
DECODE_DCP_MIN_PROMPT_TOKENS = 2048


def _pool_kv_hbm_ms(parsed_pool: CaseSpec, memory: Memory, tokens: int) -> float:
    if tokens <= 0:
        return 0.0
    parallel = parsed_pool.runtime.parallel
    width = parallel.tp * parallel.pcp
    best = 0.0
    for stage, stage_bytes in enumerate(memory.cache_bytes(tokens, padded=False)):
        dies = (
            parsed_pool.machine.dies[stage * width:(stage + 1) * width]
            or (parsed_pool.machine.dies[0],)
        )
        hbm_rate_ms = min(float(dict(die.rates)["hbm"]) for die in dies) / MS_PER_S
        best = max(best, float(stage_bytes) / max(hbm_rate_ms, PARETO_EPS))
    return best


def _extract_machine_link_tier(
    machine: Any, preferred: str,
) -> tuple[float, float, str, str] | None:
    """Return ``(bytes_per_ms, latency_us, semantics, tier_name)`` from machine."""
    fallback = "d2d_intra_node" if preferred == "d2d_inter_node" else "d2d_inter_node"
    for tier in (preferred, fallback):
        matched = [
            link for link in getattr(machine, "links", ()) or ()
            if str(link.name).startswith(tier) or str(getattr(link, "channel", "") or "") == tier
        ]
        if matched:
            cap = min(float(link.capacity_bytes_per_ms) for link in matched)
            lat = max(float(getattr(link, "latency_us", 0.0) or 0.0) for link in matched)
            sem = str(getattr(matched[0], "semantics", "per_port") or "per_port")
            if cap > 0.0:
                return cap, lat, sem, tier
        tc = dict(getattr(machine, "transfer_channels", ()) or {})
        ch_raw = tc.get(tier)
        if ch_raw is not None:
            ch = dict(ch_raw)
            for key in (
                "calibrated_uni_bw_gbps",
                "effective_uni_bw_gbps",
                "derived_uni_bw_gbps",
                "uni_peak_bw_gbps",
            ):
                val = ch.get(key)
                if val is not None and not isinstance(val, bool) and float(val) > 0.0:
                    lat = float(ch.get("alpha_us") or 0.0)
                    sem = str(ch.get("semantics") or "per_port")
                    return float(val) * BYTES_PER_MS_PER_GBPS, lat, sem, tier
    return None


def _resolve_pd_link_bandwidth(
    p_parsed: CaseSpec,
    d_parsed: CaseSpec,
    *,
    total_dies: int,
    kv_transfer_bw_gbps: float | None,
) -> dict[str, Any]:
    """Derive P->D fabric bandwidth from active SKU links unless explicitly overridden."""
    if kv_transfer_bw_gbps is not None:
        bw = check_pos_float("kv_transfer_bw_gbps", kv_transfer_bw_gbps)
        return {
            "bw_gbps": bw,
            "bw_bytes_per_ms": bw * BYTES_PER_MS_PER_GBPS,
            "latency_us": 0.0,
            "semantics": "explicit_aggregate",
            "source": "explicit",
            "same_domain": False,
        }
    p_rpn = int(getattr(p_parsed.machine, "ranks_per_node", 1) or 1)
    d_rpn = int(getattr(d_parsed.machine, "ranks_per_node", 1) or 1)
    same_domain = int(total_dies) <= min(p_rpn, d_rpn)
    preferred = "d2d_intra_node" if same_domain else "d2d_inter_node"
    p_tier = _extract_machine_link_tier(p_parsed.machine, preferred)
    d_tier = _extract_machine_link_tier(d_parsed.machine, preferred)
    tiers = [item for item in (p_tier, d_tier) if item is not None]
    if not tiers:
        raise ModelingError("missing_required_hardware_pd_link_bandwidth")
    cap_bpms = min(item[0] for item in tiers)
    lat_us = max(item[1] for item in tiers)
    semantics = tiers[0][2]
    source_tier = tiers[0][3]
    return {
        "bw_gbps": cap_bpms / BYTES_PER_MS_PER_GBPS,
        "bw_bytes_per_ms": cap_bpms,
        "latency_us": lat_us,
        "semantics": semantics,
        "source": f"sku_link:{source_tier}",
        "same_domain": same_domain,
    }


def _compute_d2d_wire_ms(handoff_plan: Mapping[str, Any], link_info: Mapping[str, Any]) -> float:
    """Compute exact P->D wire transfer time from shard-aware handoff bytes."""
    total_bytes = float(handoff_plan["total_transfer_bytes"])
    if total_bytes <= 0.0:
        return 0.0
    rate_bpms = float(link_info["bw_bytes_per_ms"])
    lat_ms = float(link_info["latency_us"]) / US_PER_MS
    if link_info["semantics"] == "explicit_aggregate":
        return total_bytes / max(rate_bpms, PARETO_EPS) + lat_ms
    src_max = max((float(v) for v in handoff_plan["source_rank_bytes"].values()), default=total_bytes)
    dst_max = max((float(v) for v in handoff_plan["destination_rank_bytes"].values()), default=total_bytes)
    return max(src_max, dst_max) / max(rate_bpms, PARETO_EPS) + lat_ms


def _compute_pooled_kv_timing(
    p_parsed: CaseSpec,
    d_parsed: CaseSpec,
    *,
    handoff_plan: Mapping[str, Any],
    p_mem: Memory,
    d_mem: Memory,
    prompt_tokens: int,
    compute_tokens: int,
    pool_dram_ratio: float,
    pool_ssd_ratio: float,
    ddr_bw_gbps: float | None,
    ssd_bw_gbps: float | None,
    link_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Compute 3-tier HBM/DRAM/SSD KV pool flush (P->Pool) and pull (Pool->D) times."""
    p_host = resolve_sku_host_dma_rates(p_parsed.machine)
    d_host = resolve_sku_host_dma_rates(d_parsed.machine)
    p_d2h_bps = p_host.get("d2h_bps") or p_host.get("h2d_bps")
    d_h2d_bps = d_host.get("h2d_bps") or d_host.get("d2h_bps")
    if not p_d2h_bps or not d_h2d_bps:
        raise ModelingError("missing_required_hardware_host_dma_rate:h2d,d2h")
    if ddr_bw_gbps is not None:
        p_ddr_bpms = d_ddr_bpms = check_pos_float("ddr_bw_gbps", ddr_bw_gbps) * BYTES_PER_MS_PER_GBPS
        resolved_ddr_gbps = float(ddr_bw_gbps)
    else:
        p_mem_bps = p_host.get("host_memory_bps")
        d_mem_bps = d_host.get("host_memory_bps")
        if not p_mem_bps or not d_mem_bps:
            raise ModelingError("missing_required_hardware_host_dma_rate:host_memory")
        p_ddr_bpms = float(p_mem_bps) / MS_PER_S
        d_ddr_bpms = float(d_mem_bps) / MS_PER_S
        resolved_ddr_gbps = min(p_ddr_bpms, d_ddr_bpms) / BYTES_PER_MS_PER_GBPS
    ssd_bpms = (
        check_pos_float("ssd_bw_gbps", ssd_bw_gbps)
        * _DEFAULT_SSD_BANDWIDTH_UTILIZATION
        * BYTES_PER_MS_PER_GBPS
        if pool_ssd_ratio > 0.0 else 0.0
    )

    flush_scale = float(compute_tokens) / float(max(1, prompt_tokens))
    unique_total = float(handoff_plan["unique_kv_bytes"]) * flush_scale
    unique_rank_max = (
        max((float(v) for v in handoff_plan["unique_source_rank_bytes"].values()), default=handoff_plan["unique_kv_bytes"])
        * flush_scale
    )
    p_d2h_bpms = float(p_d2h_bps) / MS_PER_S
    p_lat_ms = float(p_host.get("d2h_latency_us") or p_host.get("h2d_latency_us") or 0.0) / US_PER_MS
    p_dram_ms = (
        max((unique_rank_max * pool_dram_ratio) / p_d2h_bpms, (unique_total * pool_dram_ratio) / p_ddr_bpms)
        + p_lat_ms
        if unique_total > 0.0 and pool_dram_ratio > 0.0 else 0.0
    )
    p_ssd_ms = (
        max((unique_rank_max * pool_ssd_ratio) / p_d2h_bpms, (unique_total * pool_ssd_ratio) / ssd_bpms)
        + (_DEFAULT_SSD_LATENCY_US / US_PER_MS)
        if unique_total > 0.0 and pool_ssd_ratio > 0.0 else 0.0
    )
    p_flush_ms = max(p_dram_ms + p_ssd_ms, _pool_kv_hbm_ms(p_parsed, p_mem, compute_tokens)) if compute_tokens > 0 else 0.0

    pull_total = float(handoff_plan["total_transfer_bytes"])
    pull_rank_max = max((float(v) for v in handoff_plan["destination_rank_bytes"].values()), default=pull_total)
    d_h2d_bpms = float(d_h2d_bps) / MS_PER_S
    d_lat_ms = float(d_host.get("h2d_latency_us") or 0.0) / US_PER_MS
    d_dram_ms = (
        max((pull_rank_max * pool_dram_ratio) / d_h2d_bpms, (pull_total * pool_dram_ratio) / d_ddr_bpms)
        + d_lat_ms
        if pull_total > 0.0 and pool_dram_ratio > 0.0 else 0.0
    )
    if pull_total > 0.0 and pool_dram_ratio > 0.0 and link_info["source"] == "explicit":
        net_ms = (pull_total * pool_dram_ratio) / float(link_info["bw_bytes_per_ms"])
        d_dram_ms = max(d_dram_ms, net_ms)
    d_ssd_ms = (
        max((pull_rank_max * pool_ssd_ratio) / d_h2d_bpms, (pull_total * pool_ssd_ratio) / ssd_bpms)
        + (_DEFAULT_SSD_LATENCY_US / US_PER_MS)
        if pull_total > 0.0 and pool_ssd_ratio > 0.0 else 0.0
    )
    d_pull_ms = max(d_dram_ms + d_ssd_ms, _pool_kv_hbm_ms(d_parsed, d_mem, prompt_tokens)) if pull_total > 0.0 else 0.0
    return {
        "p_flush_ms": float(p_flush_ms),
        "d_pull_ms": float(d_pull_ms),
        "unique_flush_bytes": float(unique_total),
        "p_d2h_bw_gbps": float(p_d2h_bps) / 1e9,
        "d_h2d_bw_gbps": float(d_h2d_bps) / 1e9,
        "ddr_bw_gbps": float(resolved_ddr_gbps),
        "ssd_bw_gbps": float(ssd_bw_gbps) if ssd_bw_gbps is not None else None,
        "p_d2h_source": str(p_host.get("d2h_source") or p_host.get("h2d_source")),
        "d_h2d_source": str(d_host.get("h2d_source") or d_host.get("d2h_source")),
    }


def _resolve_pd_sla_and_pool_args(
    p_parsed: CaseSpec,
    d_parsed: CaseSpec,
    *,
    prompt_tokens: int | None,
    output_tokens: int | None,
    ttft_sla_ms: float | None,
    tpot_sla_ms: float | None,
    pd_transfer_mode: str,
    pool_dram_ratio: float | None,
    pool_ssd_ratio: float,
    pool_prefix_hit_ratio: float,
    ssd_bw_gbps: float | None,
    allow_unconstrained_ttft: bool = False,
) -> tuple[int, int, float, float, str, float, float, float]:
    """Enforce Domain-1 fail-fast on scenario, SLA, and external SSD inputs."""
    resolved_ttft = ttft_sla_ms if ttft_sla_ms is not None else p_parsed.slo.ttft_ms
    resolved_tpot = tpot_sla_ms if tpot_sla_ms is not None else d_parsed.slo.tpot_ms
    missing = [
        name for name, val in (
            ("prompt_tokens", prompt_tokens),
            ("output_tokens", output_tokens),
            *([] if allow_unconstrained_ttft else [("ttft_sla_ms", resolved_ttft)]),
            ("tpot_sla_ms", resolved_tpot),
        ) if val is None
    ]
    if missing:
        raise missing_scenario_error(missing)
    mode = str(pd_transfer_mode).strip().lower()
    if mode not in VALID_PD_TRANSFER_MODES:
        raise ModelingError(f"invalid_pd_transfer_mode:{pd_transfer_mode}")
    ssd_r = float(pool_ssd_ratio)
    hit_r = float(pool_prefix_hit_ratio)
    if not (0.0 <= ssd_r <= 1.0) or not (0.0 <= hit_r <= 1.0):
        raise ModelingError(f"invalid_pool_ratios:ssd={ssd_r},prefix_hit={hit_r}")
    if mode != "pooled_dram_ssd" and (ssd_r > 0.0 or hit_r > 0.0):
        raise ModelingError("pool_parameters_require_pooled_dram_ssd_mode")
    dram_r = (1.0 - ssd_r) if pool_dram_ratio is None else float(pool_dram_ratio)
    if not (0.0 <= dram_r <= 1.0) or abs((dram_r + ssd_r) - 1.0) > 1e-6:
        raise ModelingError(f"invalid_pool_dram_ssd_ratios:dram={dram_r},ssd={ssd_r}")
    if ssd_r > 0.0 and ssd_bw_gbps is None:
        raise missing_scenario_error(("ssd_bw_gbps",))
    prompt = check_pos_int("prompt_tokens", prompt_tokens, minimum=1)
    output = check_pos_int("output_tokens", output_tokens, minimum=1)
    ttft = (
        math.inf if (resolved_ttft is None and allow_unconstrained_ttft)
        else check_pos_float("ttft_sla_ms", resolved_ttft)
    )
    tpot = check_pos_float("tpot_sla_ms", resolved_tpot)
    return prompt, output, ttft, tpot, mode, dram_r, ssd_r, hit_r


def _measure_prefill_stage(
    p_parsed: CaseSpec,
    *,
    prompt: int,
    compute_tokens: int,
    hit_tokens: int,
    cache: dict[tuple[int, int, int], tuple[Memory, bool, int, float]] | None,
    cache_id: int | None,
) -> tuple[Memory, bool, int, float]:
    key = (cache_id, compute_tokens, hit_tokens) if (cache is not None and cache_id is not None) else None
    if key is not None and key in cache:
        return cache[key]
    p_mem = Memory(p_parsed, workspace_tokens=max(1, compute_tokens))
    p_kv = p_mem.cache_bytes(prompt + 1, padded=True)
    prefill_hbm_ok = bool(
        prompt <= p_parsed.runtime.max_model_len
        and (p_parsed.runtime.enable_chunked_prefill
             or compute_tokens <= p_parsed.runtime.max_num_batched_tokens)
        and all(capacity >= 0.0 and capacity >= kv
                for row in p_mem.capacity for capacity, kv in zip(row, p_kv))
    )
    p_slots = (
        min([int(p_parsed.runtime.max_num_seqs)]
            + [int(math.floor(max(0.0, float(capacity)) / float(kv)))
               for row in p_mem.capacity
               for capacity, kv in zip(row, p_kv) if kv > 0])
        if prefill_hbm_ok else 0
    )
    if compute_tokens > 0:
        _, _, p_graph = _lower_and_run_graph(
            p_parsed, phase="prefill", tokens=compute_tokens,
            context=hit_tokens, sequences=1, scope="model", run_graph=True,
            fast_span_only=True,
        )
        assert p_graph is not None
        p_step = float(p_graph["duration_ms"])
    else:
        p_step = 0.0
    result = (p_mem, prefill_hbm_ok, p_slots, p_step)
    if key is not None:
        cache[key] = result
    return result


def _measure_decode_stage(
    d_parsed: CaseSpec,
    *,
    prompt: int,
    output: int,
    eff_decode: int,
    tpot_sla_ms: float,
    exposed_kv_ms: float,
    decode_tokens: int,
    max_sequences_per_dp: int | None,
    cache: dict[tuple[int, float], dict[str, Any]] | None,
    cache_id: int | None,
) -> dict[str, Any]:
    d_dp = int(d_parsed.runtime.parallel.dp)
    handoff_tpot = exposed_kv_ms / float(decode_tokens)
    decode_budget = tpot_sla_ms - handoff_tpot
    _, d_emit_est = _resolve_speculative_tokens(
        d_parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    step_budget = decode_budget * max(1.0, float(d_emit_est))
    eff_sla_ms = (
        max(step_budget, PARETO_EPS) if step_budget > 0.0
        else tpot_sla_ms * max(1.0, float(d_emit_est))
    )
    d_sla_key = (cache_id, round(eff_sla_ms, 2)) if (cache is not None and cache_id is not None) else None
    if d_sla_key is not None and d_sla_key in cache:
        d_sla = cache[d_sla_key]
    else:
        sla_kw: dict[str, Any] = dict(
            sla_ms=eff_sla_ms, context=eff_decode, phase="decode",
            output_budget=output, scenarios=[DEFAULT_SLA_SCENARIOS[0]],
        )
        if max_sequences_per_dp is not None:
            sla_kw["max_sequences_per_dp"] = int(max_sequences_per_dp)
        d_sla = solve_sla_sizing(d_parsed, **sla_kw)
        if d_sla_key is not None:
            cache[d_sla_key] = d_sla
    d_scen = d_sla["scenarios"][0]
    emitted = float(d_sla["emitted_tokens_per_seq"])
    length_ok = bool(max(prompt, eff_decode) + output <= d_parsed.runtime.max_model_len)
    b_vram = int(d_scen["b_vram_max"]) if length_ok else 0
    b_sla = int(d_scen["b_sla"]) if decode_budget > 0.0 else 0
    b_feasible = int(d_scen["b_feasible"])
    b1_step = float(d_scen["step_ms_at_b1"])
    decode_step = (
        float(d_scen["step_ms_feasible"])
        if b_feasible >= 1 and d_scen.get("step_ms_feasible") is not None
        else b1_step
    )
    effective_tpot = decode_step / max(emitted, PARETO_EPS) + handoff_tpot
    first_itl = exposed_kv_ms + decode_step
    tpot_ok = bool(b_feasible >= 1 and effective_tpot <= tpot_sla_ms + PARETO_EPS)
    rps_d = (
        (b_feasible * d_dp * emitted * MS_PER_S)
        / (max(decode_step, PARETO_EPS) * float(decode_tokens))
        if tpot_ok else 0.0
    )
    raw_rps_d = (
        (max(1, b_feasible) * d_dp * emitted * MS_PER_S)
        / (max(decode_step, PARETO_EPS) * float(decode_tokens))
    )
    return {
        "decode_budget": decode_budget, "length_ok": length_ok,
        "b_vram": b_vram, "b_sla": b_sla, "b_feasible": b_feasible,
        "b1_step": b1_step, "decode_step": decode_step,
        "effective_tpot": effective_tpot, "first_itl": first_itl,
        "tpot_ok": tpot_ok, "rps_d": rps_d, "raw_rps_d": raw_rps_d,
        "decode_tps": rps_d * float(output),
    }


def _evaluate_pd_stages(
    p_parsed: CaseSpec,
    d_parsed: CaseSpec,
    *,
    total: int,
    prompt: int,
    output: int,
    eff_decode: int,
    ttft_sla_ms: float,
    tpot_sla_ms: float,
    mode: str,
    dram_r: float,
    ssd_r: float,
    hit_r: float,
    kv_transfer_bw_gbps: float | None,
    ddr_bw_gbps: float | None,
    ssd_bw_gbps: float | None,
    max_sequences_per_dp: int | None = None,
    _prefill_step_cache: dict[tuple[int, int, int], tuple[Memory, bool, int, float]] | None = None,
    _prefill_cache_id: int | None = None,
    _decode_sla_cache: dict[tuple[int, float], dict[str, Any]] | None = None,
    _decode_cache_id: int | None = None,
) -> dict[str, Any]:
    """Evaluate prefill, KV handoff / 3-tier pool, and decode stage rates."""
    p_dp = int(p_parsed.runtime.parallel.dp)
    hit_tokens = int(math.floor(prompt * hit_r)) if mode == "pooled_dram_ssd" else 0
    compute_tokens = max(0, prompt - hit_tokens)
    p_mem, prefill_hbm_ok, p_slots, p_step = _measure_prefill_stage(
        p_parsed, prompt=prompt, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
        cache=_prefill_step_cache, cache_id=_prefill_cache_id,
    )
    prefill_ok = bool(p_step <= ttft_sla_ms)
    decode_q_tokens, _ = _resolve_speculative_tokens(
        d_parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    d_mem = Memory(d_parsed, workspace_tokens=decode_q_tokens)
    decode_tokens = output - 1
    link_info = _resolve_pd_link_bandwidth(
        p_parsed, d_parsed, total_dies=total, kv_transfer_bw_gbps=kv_transfer_bw_gbps,
    )
    if decode_tokens <= 0:
        rps_p = (
            (p_dp * MS_PER_S) / max(p_step, PARETO_EPS)
            if (prefill_ok and prefill_hbm_ok and p_step > 0.0)
            else (float(p_dp * p_slots) * MS_PER_S if (prefill_ok and prefill_hbm_ok) else 0.0)
        )
        infeasible = (
            "prefill_hbm_oom" if not prefill_hbm_ok
            else "prefill_ttft_sla_exceeded" if not prefill_ok else None
        )
        return {
            "p_step": p_step, "prefill_hbm_ok": prefill_hbm_ok, "prefill_ok": prefill_ok,
            "compute_tokens": compute_tokens, "hit_tokens": hit_tokens,
            "kv_bytes": 0.0, "unique_kv_bytes": 0.0, "kv_ms": 0.0,
            "exposed_kv_ms": 0.0, "tail_layer_kv_ms": 0.0,
            "rps_p": rps_p, "rps_d": rps_p, "raw_rps_d": rps_p, "decode_tps": rps_p * float(output),
            "p_kv_rps_per_inst": rps_p, "d_kv_rps_per_inst": None,
            "b_sla": 0, "b_vram": int(d_parsed.runtime.max_num_seqs), "b_feasible": 0,
            "b1_step": 0.0, "decode_step": 0.0, "effective_tpot": 0.0, "first_itl": 0.0,
            "decode_budget": tpot_sla_ms, "tpot_ok": True, "infeasible": infeasible,
            "link_info": link_info, "pool_metrics": None, "handoff_plan": None,
        }

    handoff = compute_pd_handoff_plan(
        p_parsed, d_parsed, tokens=prompt, prefill_memory=p_mem, decode_memory=d_mem,
    )
    kv_bytes = float(handoff["total_transfer_bytes"])
    tail_frac = (float(handoff["tail_layer_bytes"]) / kv_bytes) if kv_bytes > 0.0 else 0.0
    pool_metrics: dict[str, Any] | None = None
    d_kv_rps_per_inst: float | None = None

    if mode == "pooled_dram_ssd":
        pool_metrics = _compute_pooled_kv_timing(
            p_parsed, d_parsed, handoff_plan=handoff, p_mem=p_mem, d_mem=d_mem,
            prompt_tokens=prompt, compute_tokens=compute_tokens,
            pool_dram_ratio=dram_r, pool_ssd_ratio=ssd_r,
            ddr_bw_gbps=ddr_bw_gbps, ssd_bw_gbps=ssd_bw_gbps, link_info=link_info,
        )
        p_flush_ms, d_pull_ms = float(pool_metrics["p_flush_ms"]), float(pool_metrics["d_pull_ms"])
        kv_ms = p_flush_ms + d_pull_ms
        tail_layer_kv_ms = p_flush_ms * tail_frac
        exposed_kv_ms = max(0.0, p_flush_ms - p_step) + d_pull_ms
        p_slot_hold_ms = max(p_step, p_flush_ms)
        per_rep = (
            min(
                MS_PER_S / max(p_step, PARETO_EPS) if p_step > 0.0 else float("inf"),
                (float(p_slots) * MS_PER_S) / max(p_slot_hold_ms, PARETO_EPS),
            ) if p_slot_hold_ms > 0.0 else float(p_slots) * MS_PER_S
        )
        p_kv_rps_per_inst = (MS_PER_S / max(p_flush_ms, 1e-9)) if p_flush_ms > 0.0 else float("inf")
        d_kv_rps_per_inst = MS_PER_S / max(d_pull_ms, 1e-9)
    else:
        wire_ms = _compute_d2d_wire_ms(handoff, link_info)
        kv_ms = max(wire_ms, _pool_kv_hbm_ms(p_parsed, p_mem, prompt), _pool_kv_hbm_ms(d_parsed, d_mem, prompt))
        tail_layer_kv_ms = kv_ms * tail_frac
        if mode == "layerwise_overlap_d2d":
            p_slot_hold_ms = max(p_step, kv_ms) + tail_layer_kv_ms
            exposed_kv_ms = max(0.0, kv_ms - p_step) + tail_layer_kv_ms
        else:
            p_slot_hold_ms = p_step + kv_ms
            exposed_kv_ms = kv_ms
        per_rep = min(
            MS_PER_S / max(p_step, PARETO_EPS),
            (float(p_slots) * MS_PER_S) / max(p_slot_hold_ms, PARETO_EPS),
        )
        p_kv_rps_per_inst = MS_PER_S / max(kv_ms, 1e-9)

    rps_p = (p_dp * per_rep) if (prefill_ok and prefill_hbm_ok) else 0.0
    dec = _measure_decode_stage(
        d_parsed, prompt=prompt, output=output, eff_decode=eff_decode,
        tpot_sla_ms=tpot_sla_ms, exposed_kv_ms=exposed_kv_ms, decode_tokens=decode_tokens,
        max_sequences_per_dp=max_sequences_per_dp,
        cache=_decode_sla_cache, cache_id=_decode_cache_id,
    )
    infeasible = (
        "prefill_hbm_oom" if not prefill_hbm_ok
        else "prefill_ttft_sla_exceeded" if not prefill_ok
        else "decode_max_model_len_exceeded" if not dec["length_ok"]
        else "decode_hbm_oom" if dec["b_vram"] < 1
        else "effective_tpot_sla_exceeded" if not dec["tpot_ok"]
        else None
    )
    return {
        "p_step": p_step, "prefill_hbm_ok": prefill_hbm_ok, "prefill_ok": prefill_ok,
        "compute_tokens": compute_tokens, "hit_tokens": hit_tokens,
        "kv_bytes": kv_bytes, "unique_kv_bytes": float(handoff["unique_kv_bytes"]),
        "kv_ms": kv_ms, "exposed_kv_ms": exposed_kv_ms, "tail_layer_kv_ms": tail_layer_kv_ms,
        "rps_p": rps_p, "p_kv_rps_per_inst": p_kv_rps_per_inst, "d_kv_rps_per_inst": d_kv_rps_per_inst,
        "infeasible": infeasible, "link_info": link_info,
        "pool_metrics": pool_metrics, "handoff_plan": handoff,
        **dec,
    }


def _mark_pd_pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    feasible_rows = [row for row in rows if row["feasible"]]
    for row in rows:
        if not row["feasible"]:
            row["pareto_optimal"] = False
            continue
        dominated = False
        for other in feasible_rows:
            if other is row:
                continue
            ge_all = (
                float(other["sustainable_tps"]) >= float(row["sustainable_tps"]) - PARETO_EPS
                and int(other["used_dies"]) <= int(row["used_dies"])
            )
            strict = (
                float(other["sustainable_tps"]) > float(row["sustainable_tps"]) + PARETO_EPS
                or int(other["used_dies"]) < int(row["used_dies"])
            )
            if ge_all and strict:
                dominated = True
                break
        row["pareto_optimal"] = not dominated
    return [row for row in rows if row.get("pareto_optimal")]


def sweep_pd_disaggregation_ratios(
    prefill_case: CaseSpec | Mapping[str, Any],
    decode_case: CaseSpec | Mapping[str, Any],
    *,
    total_dies: int,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    decode_context: int | None = None,
    ttft_sla_ms: float | None = None,
    tpot_sla_ms: float | None = None,
    kv_transfer_bw_gbps: float | None = None,
    pd_transfer_mode: str = "direct_blocking_d2d",
    pool_dram_ratio: float | None = None,
    pool_ssd_ratio: float = 0.0,
    pool_prefix_hit_ratio: float = 0.0,
    ddr_bw_gbps: float | None = None,
    ssd_bw_gbps: float | None = None,
    max_sequences_per_dp: int | None = None,
    allow_unconstrained_ttft: bool = False,
    _prefill_step_cache: dict[tuple[int, int, int], tuple[Memory, bool, int, float]] | None = None,
    _prefill_cache_id: int | None = None,
    _decode_sla_cache: dict[tuple[int, float], dict[str, Any]] | None = None,
    _decode_cache_id: int | None = None,
) -> dict[str, Any]:
    """Sweep disaggregated P/D instance counts with real handoff & 3-tier KV pooling."""
    total = check_pos_int("total_dies", total_dies, minimum=1)
    p_parsed, d_parsed = parse_case(prefill_case), parse_case(decode_case)
    prompt, output, ttft, tpot, mode, dram_r, ssd_r, hit_r = _resolve_pd_sla_and_pool_args(
        p_parsed, d_parsed,
        prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        ttft_sla_ms=ttft_sla_ms, tpot_sla_ms=tpot_sla_ms,
        pd_transfer_mode=pd_transfer_mode, pool_dram_ratio=pool_dram_ratio,
        pool_ssd_ratio=pool_ssd_ratio, pool_prefix_hit_ratio=pool_prefix_hit_ratio,
        ssd_bw_gbps=ssd_bw_gbps, allow_unconstrained_ttft=allow_unconstrained_ttft,
    )
    eff_decode = (
        prompt if decode_context is None
        else check_pos_int("decode_context", decode_context, minimum=0)
    )
    st = _evaluate_pd_stages(
        p_parsed, d_parsed, total=total, prompt=prompt, output=output,
        eff_decode=eff_decode, ttft_sla_ms=ttft, tpot_sla_ms=tpot,
        mode=mode, dram_r=dram_r, ssd_r=ssd_r, hit_r=hit_r,
        kv_transfer_bw_gbps=kv_transfer_bw_gbps,
        ddr_bw_gbps=ddr_bw_gbps, ssd_bw_gbps=ssd_bw_gbps,
        max_sequences_per_dp=max_sequences_per_dp,
        _prefill_step_cache=_prefill_step_cache,
        _prefill_cache_id=_prefill_cache_id,
        _decode_sla_cache=_decode_sla_cache,
        _decode_cache_id=_decode_cache_id,
    )
    p_par, d_par = p_parsed.runtime.parallel, d_parsed.runtime.parallel
    p_dies, d_dies = p_par.world_size, d_par.world_size
    decode_tokens = output - 1
    rows: list[dict[str, Any]] = []
    for n_p in range(1, (total // p_dies) + 1):
        max_nd = (total - n_p * p_dies) // d_dies
        for n_d in range(1, max_nd + 1):
            used = n_p * p_dies + n_d * d_dies
            r_p = n_p * float(st["rps_p"])
            if decode_tokens <= 0:
                r_kv, r_d = r_p, r_p
            else:
                r_kv_p = n_p * float(st["p_kv_rps_per_inst"])
                r_kv = (
                    min(r_kv_p, n_d * float(st["d_kv_rps_per_inst"]))
                    if st["d_kv_rps_per_inst"] is not None else r_kv_p
                )
                r_d = n_d * float(st["rps_d"])
            feasible = bool(st["infeasible"] is None and min(r_p, r_kv, r_d) > 0.0)
            r_sys = min(r_p, r_kv, r_d) if feasible else 0.0
            sustainable = r_sys * float(output)
            if decode_tokens <= 0:
                bottleneck = "prefill"
            else:
                d_cmp = r_d if st["decode_budget"] > 0.0 else n_d * float(st["raw_rps_d"])
                bottleneck = min(
                    (("prefill", r_p), ("kv_transfer", r_kv), ("decode", d_cmp)),
                    key=lambda item: item[1],
                )[0]
            rows.append({
                "n_prefill": n_p, "n_decode": n_d, "pd_ratio": f"{n_p}P:{n_d}D",
                "pd_transfer_mode": mode,
                "prefill_dies": n_p * p_dies, "decode_dies": n_d * d_dies,
                "used_dies": used, "idle_dies": total - used,
                "prefill_rps": r_p, "kv_transfer_rps": r_kv, "decode_rps": r_d,
                "sustainable_rps": r_sys, "sustainable_tps": sustainable,
                "sustainable_tps_per_die": sustainable / float(used),
                "decode_step_ms": st["decode_step"],
                "effective_tpot_ms": st["effective_tpot"],
                "first_decode_itl_ms": st["first_itl"],
                "bottleneck_stage": bottleneck, "feasible": feasible,
                "infeasible_reason": st["infeasible"], "reason": st["infeasible"],
                "metrics_boundary": "one_pipeline_stage_rate_composition",
            })
    front = _mark_pd_pareto(rows)
    best = max(front, key=lambda row: (row["sustainable_tps"], -row["used_dies"]), default=None)
    link_info = st["link_info"]
    return {
        "schema": "mindie-sim.pd-sweep/1",
        "total_dies": total,
        "prompt_tokens": prompt,
        "output_tokens": output,
        "decode_context": eff_decode,
        "ttft_sla_ms": (None if math.isinf(ttft) else ttft),
        "tpot_sla_ms": tpot,
        "pd_transfer_mode": mode,
        "kv_transfer_bw_gbps": float(link_info["bw_gbps"]),
        "kv_link_source": str(link_info["source"]),
        "kv_bytes_per_request": float(st["kv_bytes"]),
        "unique_kv_bytes_per_request": float(st["unique_kv_bytes"]),
        "kv_transfer_ms_per_request": float(st["kv_ms"]),
        "exposed_kv_handoff_ms": float(st["exposed_kv_ms"]),
        "tail_layer_kv_ms": float(st["tail_layer_kv_ms"]),
        "pool_metrics": st["pool_metrics"],
        "decode_step_ms": float(st["decode_step"]),
        "effective_tpot_ms": float(st["effective_tpot"]),
        "first_decode_itl_ms": float(st["first_itl"]),
        "prefill_instance": {
            "dies_per_instance": p_dies, "parallel": p_par.to_dict(),
            "step_ms": float(st["p_step"]), "hbm_ok": bool(st["prefill_hbm_ok"]),
            "ttft_sla_ok": bool(st["prefill_ok"]), "rps_per_instance": float(st["rps_p"]),
            "effective_prefill_tokens": int(st["compute_tokens"]),
            "prefix_hit_tokens": int(st["hit_tokens"]),
        },
        "decode_instance": {
            "dies_per_instance": d_dies, "parallel": d_par.to_dict(),
            "b_sla": int(st["b_sla"]), "b_vram_max": int(st["b_vram"]),
            "b_feasible": int(st["b_feasible"]),
            "step_ms_at_b1": float(st["b1_step"]),
            "step_ms_feasible": float(st["decode_step"]),
            "effective_tpot_ms": float(st["effective_tpot"]),
            "first_decode_itl_ms": float(st["first_itl"]),
            "tpot_sla_ok": bool(st["tpot_ok"]),
            "tps_per_instance": float(st["decode_tps"]),
            "rps_per_instance": float(st["rps_d"]),
        },
        "candidates": rows, "ratios": rows, "pareto_frontier": front,
        "best_configuration": best,
        "plan_selection": "fixed_stage_case_per_pipeline",
        "composition": "separate_prefill_and_decode_stage_programs_bound_by_min_stage_rate",
        "comparison_boundary":
            "stage_rates_are_composed_for_one_pipeline_configuration_not_stitched_per_metric",
    }


def _matches_pinned(cand: Mapping[str, int], pinned: Mapping[str, int]) -> bool:
    for axis, val in pinned.items():
        if int(cand.get(axis, 1)) != int(val):
            return False
    return True


def _seed_from_profile_topo(
    rec_topo: Mapping[str, Any] | None,
    role: str,
    *,
    max_dies: int,
    pinned: Mapping[str, int],
) -> dict[str, int] | None:
    if not isinstance(rec_topo, Mapping) or role not in rec_topo:
        return None
    raw = rec_topo[role]
    if not isinstance(raw, Mapping):
        return None
    cand = {
        "dp": int(raw.get("dp", 1)), "tp": int(raw.get("tp", 1)),
        "ep": int(raw.get("ep", 1)), "pp": int(raw.get("pp", 1)),
        "pcp": int(raw.get("pcp", 1)), "dcp": int(raw.get("dcp", raw.get("cp", 1))),
    }
    for local_key in ("q_proj_tp", "o_proj_tp", "linear_proj_tp", "dcp_linear_tp",
                      "mlp_tp", "shared_expert_tp", "lm_head_tp"):
        if raw.get(local_key):
            cand[local_key] = int(raw[local_key])
    world = cand["dp"] * cand["tp"] * cand["pp"] * cand["pcp"]
    if world <= max_dies and _matches_pinned(cand, pinned):
        return cand
    return None


def _build_colocated_candidates(
    *,
    num_devices: int,
    is_moe: bool,
    layers: int,
    rec_topo: Mapping[str, Any] | None,
    pinned: Mapping[str, int],
) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()

    def add(cand: dict[str, int]) -> None:
        dp, tp, pp, pcp = cand["dp"], cand["tp"], cand["pp"], cand["pcp"]
        ep, dcp = cand["ep"], cand["dcp"]
        world = dp * tp * pp * pcp
        if world <= 0 or world > num_devices or pp > layers:
            return
        if not is_moe and ep != 1:
            return
        if is_moe and not (ep == 1 or (dp * tp * pcp) % ep == 0):
            return
        if not _matches_pinned(cand, pinned):
            return
        key = (dp, tp, ep, pp, pcp, dcp, cand.get("o_proj_tp", 1), cand.get("lm_head_tp", 1))
        if key not in seen:
            seen.add(key)
            out.append(cand)

    seeded = _seed_from_profile_topo(rec_topo, "colocated", max_dies=num_devices, pinned=pinned)
    if seeded:
        add(seeded)
    pp_vals = (pinned["pp"],) if "pp" in pinned else (1, 2)
    pcp_vals = (pinned["pcp"],) if "pcp" in pinned else (1,)
    dcp_vals = (pinned["dcp"],) if "dcp" in pinned else (1,)
    tp_vals = (pinned["tp"],) if "tp" in pinned else tuple(w for w in _POW2_WIDTHS if w <= num_devices)
    for pp in pp_vals:
        for pcp in pcp_vals:
            for tp in tp_vals:
                unit = tp * pp * pcp
                if unit > num_devices:
                    continue
                dp_choices = (
                    (pinned["dp"],) if "dp" in pinned
                    else tuple(dict.fromkeys([num_devices // unit, 1]))
                )
                for dp in dp_choices:
                    if dp * unit > num_devices or dp < 1:
                        continue
                    for dcp in dcp_vals:
                        ep_val = pinned["ep"] if "ep" in pinned else (dp * tp * pcp if is_moe else 1)
                        add({"dp": dp, "tp": tp, "ep": ep_val, "pp": pp, "pcp": pcp, "dcp": dcp})
    return out[:MAX_COLOCATED_CANDIDATES]


def _build_pd_stage_candidates(
    *,
    role: str,
    max_dies: int,
    is_moe: bool,
    layers: int,
    prompt_tokens: int,
    rec_topo: Mapping[str, Any] | None,
    pinned: Mapping[str, int],
) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    seen: set[tuple[int, ...]] = set()

    def add(cand: dict[str, int]) -> None:
        dp, tp, pp, pcp = cand["dp"], cand["tp"], cand["pp"], cand["pcp"]
        ep, dcp = cand["ep"], cand["dcp"]
        world = dp * tp * pp * pcp
        if world <= 0 or world > max_dies or pp > layers:
            return
        if not is_moe and ep != 1:
            return
        if is_moe and not (ep == 1 or (dp * tp * pcp) % ep == 0):
            return
        if not _matches_pinned(cand, pinned):
            return
        key = (dp, tp, ep, pp, pcp, dcp, cand.get("o_proj_tp", 1), cand.get("lm_head_tp", 1))
        if key not in seen:
            seen.add(key)
            out.append(cand)

    seeded = _seed_from_profile_topo(rec_topo, role, max_dies=max_dies, pinned=pinned)
    if seeded:
        add(seeded)
    widths = tuple(w for w in _POW2_WIDTHS if w <= max_dies)
    if role == "pd_prefill":
        for tp in ((pinned["tp"],) if "tp" in pinned else widths):
            for pp in ((pinned["pp"],) if "pp" in pinned else (1, 2)):
                pcp_vals = (
                    (pinned["pcp"],) if "pcp" in pinned
                    else (_PREFILL_PCP_WIDTHS if prompt_tokens >= PREFILL_PCP_MIN_PROMPT_TOKENS else (1,))
                )
                for pcp in pcp_vals:
                    dp = int(pinned.get("dp", 1))
                    if dp * tp * pp * pcp > max_dies:
                        continue
                    ep = int(pinned.get("ep", dp * tp * pcp if is_moe else 1))
                    add({"dp": dp, "tp": tp, "ep": ep, "pp": pp, "pcp": pcp, "dcp": int(pinned.get("dcp", 1))})
    else:
        dp_widths = (pinned["dp"],) if "dp" in pinned else (widths if is_moe else (1,))
        for dp in dp_widths:
            for tp in ((pinned["tp"],) if "tp" in pinned else widths):
                pp = int(pinned.get("pp", 1))
                pcp = int(pinned.get("pcp", 1))
                if dp * tp * pp * pcp > max_dies:
                    continue
                ep = int(pinned.get("ep", dp * tp * pcp if is_moe else 1))
                dcp_choices = (
                    (pinned["dcp"],) if "dcp" in pinned
                    else ((1, tp) if tp > 1 and prompt_tokens >= DECODE_DCP_MIN_PROMPT_TOKENS else (1,))
                )
                for dcp in dcp_choices:
                    add({"dp": dp, "tp": tp, "ep": ep, "pp": pp, "pcp": pcp, "dcp": dcp})
    return out[:MAX_PD_STAGE_CANDIDATES]


def _evaluate_colocated_candidate(
    raw_case: Mapping[str, Any],
    *,
    prompt_tokens: int,
    output_tokens: int,
    ttft_sla_ms: float,
    tpot_sla_ms: float,
    mtp_tokens: int,
    max_sequences_per_dp: int | None = None,
) -> dict[str, Any]:
    parsed = parse_case(raw_case)
    par = parsed.runtime.parallel
    used_dies = par.world_size
    p_mem = Memory(parsed, workspace_tokens=prompt_tokens)
    p_kv = p_mem.cache_bytes(prompt_tokens + 1, padded=True)
    prefill_hbm_ok = bool(
        prompt_tokens <= parsed.runtime.max_model_len
        and (parsed.runtime.enable_chunked_prefill
             or prompt_tokens <= parsed.runtime.max_num_batched_tokens)
        and all(cap >= 0.0 and cap >= kv for row in p_mem.capacity for cap, kv in zip(row, p_kv))
    )
    _, _, p_graph = _lower_and_run_graph(
        parsed, phase="prefill", tokens=prompt_tokens, context=0,
        sequences=1, scope="model", run_graph=True, fast_span_only=True,
    )
    assert p_graph is not None
    p_step = float(p_graph["duration_ms"])
    ttft_ok = bool(p_step <= ttft_sla_ms)
    decode_tokens = max(0, output_tokens - 1)
    if decode_tokens == 0:
        feasible = bool(prefill_hbm_ok and ttft_ok)
        rps = (par.dp * MS_PER_S) / max(p_step, PARETO_EPS) if feasible else 0.0
        return {
            "deployment_mode": "colocated", "used_dies": used_dies, "parallel": par.to_dict(),
            "mtp_proposal_tokens": mtp_tokens, "ttft_ms": p_step, "effective_tpot_ms": 0.0,
            "b_feasible": 1 if feasible else 0, "sustainable_rps": rps,
            "sustainable_tps": rps * float(output_tokens),
            "sustainable_tps_per_die": (rps * float(output_tokens)) / float(used_dies),
            "feasible": feasible, "infeasible_reason": None if feasible else "prefill_sla_or_oom",
        }
    _, emit_est = _resolve_speculative_tokens(
        parsed, output_budget=output_tokens, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    sla_kw: dict[str, Any] = dict(
        sla_ms=tpot_sla_ms * max(1.0, float(emit_est)),
        context=prompt_tokens, phase="decode", output_budget=output_tokens,
        scenarios=[DEFAULT_SLA_SCENARIOS[0]],
    )
    if max_sequences_per_dp is not None:
        sla_kw["max_sequences_per_dp"] = int(max_sequences_per_dp)
    d_sla = solve_sla_sizing(parsed, **sla_kw)
    scen = d_sla["scenarios"][0]
    emitted = float(d_sla["emitted_tokens_per_seq"])
    b_feasible = int(scen["b_feasible"])
    decode_step = float(scen["step_ms_feasible"] if b_feasible >= 1 and scen.get("step_ms_feasible") is not None else scen["step_ms_at_b1"])
    tpot_ms = decode_step / max(emitted, PARETO_EPS)
    length_ok = bool(prompt_tokens + output_tokens <= parsed.runtime.max_model_len)
    feasible = bool(prefill_hbm_ok and ttft_ok and length_ok and b_feasible >= 1 and tpot_ms <= tpot_sla_ms + PARETO_EPS)
    req_compute_ms = p_step + (float(decode_tokens) * decode_step) / max(float(max(1, b_feasible)) * emitted, PARETO_EPS)
    rps = (float(par.dp) * MS_PER_S) / max(req_compute_ms, PARETO_EPS) if feasible else 0.0
    tps = rps * float(output_tokens)
    reason = (
        None if feasible else
        "prefill_hbm_oom" if not prefill_hbm_ok else
        "prefill_ttft_sla_exceeded" if not ttft_ok else
        "max_model_len_exceeded" if not length_ok else
        "decode_tpot_or_vram_exceeded"
    )
    return {
        "deployment_mode": "colocated", "used_dies": used_dies, "parallel": par.to_dict(),
        "mtp_proposal_tokens": mtp_tokens, "ttft_ms": p_step, "effective_tpot_ms": tpot_ms,
        "b_feasible": b_feasible, "sustainable_rps": rps, "sustainable_tps": tps,
        "sustainable_tps_per_die": tps / float(used_dies),
        "feasible": feasible, "infeasible_reason": reason,
    }


def _search_colocated_deployments(
    compose_for: Any,
    *,
    num_devices: int,
    is_moe: bool,
    layers: int,
    rec_topo: Mapping[str, Any] | None,
    pinned: Mapping[str, int],
    mtp_grid: tuple[int, ...],
    prompt: int,
    output: int,
    ttft: float,
    tpot: float,
    max_sequences_per_dp: int | None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    coloc_topos = _build_colocated_candidates(
        num_devices=num_devices, is_moe=is_moe, layers=layers,
        rec_topo=rec_topo, pinned=pinned,
    )
    rows: list[dict[str, Any]] = []
    feasible_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for mtp_n in mtp_grid:
        for topo in coloc_topos:
            raw_c = compose_for(topo, mtp_n, "colocated")
            if raw_c is None:
                continue
            try:
                row = _evaluate_colocated_candidate(
                    raw_c, prompt_tokens=prompt, output_tokens=output,
                    ttft_sla_ms=ttft, tpot_sla_ms=tpot, mtp_tokens=mtp_n,
                    max_sequences_per_dp=max_sequences_per_dp,
                )
            except (ModelingError, ValueError):
                continue
            rows.append(row)
            if row["feasible"]:
                feasible_entries.append((row, raw_c))
    return rows, feasible_entries


def _search_pd_deployments(
    compose_for: Any,
    *,
    num_devices: int,
    is_moe: bool,
    layers: int,
    rec_topo: Mapping[str, Any] | None,
    p_pin: Mapping[str, int],
    d_pin: Mapping[str, int],
    mtp_grid: tuple[int, ...],
    prompt: int,
    output: int,
    ttft: float,
    tpot: float,
    pd_transfer_mode: str | None,
    kv_transfer_bw_gbps: float | None,
    pool_dram_ratio: float | None,
    pool_ssd_ratio: float,
    pool_prefix_hit_ratio: float,
    ddr_bw_gbps: float | None,
    ssd_bw_gbps: float | None,
    max_sequences_per_dp: int | None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]]:
    p_topos = _build_pd_stage_candidates(
        role="pd_prefill", max_dies=num_devices - 1, is_moe=is_moe,
        layers=layers, prompt_tokens=prompt, rec_topo=rec_topo, pinned=p_pin,
    )
    d_topos = _build_pd_stage_candidates(
        role="pd_decode", max_dies=num_devices - 1, is_moe=is_moe,
        layers=layers, prompt_tokens=prompt, rec_topo=rec_topo, pinned=d_pin,
    )
    pd_modes = (
        (str(pd_transfer_mode).strip().lower(),)
        if pd_transfer_mode is not None and str(pd_transfer_mode).strip().lower() != "auto"
        else (("pooled_dram_ssd",) if (float(pool_ssd_ratio) > 0.0 or float(pool_prefix_hit_ratio) > 0.0)
              else AUTO_PD_TRANSFER_MODES)
    )
    prefill_cache: dict[tuple[int, int, int], tuple[Memory, bool, int, float]] = {}
    decode_cache: dict[tuple[int, float], dict[str, Any]] = {}
    p_cases = [
        (idx, parse_case(raw).runtime.parallel.world_size, raw)
        for idx, topo in enumerate(p_topos)
        for raw in (compose_for(topo, 0, "pd_prefill"),)
        if raw is not None
    ]
    rows: list[dict[str, Any]] = []
    feasible_entries: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    d_id = 0
    for mtp_n in mtp_grid:
        d_cases = []
        for topo in d_topos:
            raw_d = compose_for(topo, mtp_n, "pd_decode")
            if raw_d is not None:
                d_id += 1
                d_cases.append((d_id, parse_case(raw_d).runtime.parallel.world_size, raw_d))
        for p_idx, p_world, p_raw in p_cases:
            for d_idx, d_world, d_raw in d_cases:
                if p_world + d_world > num_devices:
                    continue
                for pd_mode in pd_modes:
                    try:
                        rep = sweep_pd_disaggregation_ratios(
                            p_raw, d_raw, total_dies=num_devices,
                            prompt_tokens=prompt, output_tokens=output,
                            ttft_sla_ms=(None if math.isinf(ttft) else ttft),
                            tpot_sla_ms=tpot,
                            kv_transfer_bw_gbps=kv_transfer_bw_gbps,
                            pd_transfer_mode=pd_mode,
                            pool_dram_ratio=(pool_dram_ratio if pd_mode == "pooled_dram_ssd" else None),
                            pool_ssd_ratio=(pool_ssd_ratio if pd_mode == "pooled_dram_ssd" else 0.0),
                            pool_prefix_hit_ratio=(pool_prefix_hit_ratio if pd_mode == "pooled_dram_ssd" else 0.0),
                            ddr_bw_gbps=ddr_bw_gbps, ssd_bw_gbps=ssd_bw_gbps,
                            max_sequences_per_dp=max_sequences_per_dp,
                            allow_unconstrained_ttft=math.isinf(ttft),
                            _prefill_step_cache=prefill_cache, _prefill_cache_id=p_idx,
                            _decode_sla_cache=decode_cache, _decode_cache_id=d_idx,
                        )
                    except (ModelingError, ValueError):
                        continue
                    best_ratio = rep.get("best_configuration")
                    row = {
                        "deployment_mode": "pd_disaggregated",
                        "pd_transfer_mode": pd_mode,
                        "mtp_proposal_tokens": mtp_n,
                        "prefill_parallel": rep["prefill_instance"]["parallel"],
                        "decode_parallel": rep["decode_instance"]["parallel"],
                        "pd_ratio": best_ratio["pd_ratio"] if best_ratio else None,
                        "used_dies": int(best_ratio["used_dies"]) if best_ratio else (p_world + d_world),
                        "b_feasible": int(rep["decode_instance"]["b_feasible"]),
                        "ttft_ms": float(rep["prefill_instance"]["step_ms"]),
                        "effective_tpot_ms": float(rep["effective_tpot_ms"]),
                        "kv_transfer_ms_per_request": float(rep["kv_transfer_ms_per_request"]),
                        "sustainable_rps": float(best_ratio["sustainable_rps"]) if best_ratio else 0.0,
                        "sustainable_tps": float(best_ratio["sustainable_tps"]) if best_ratio else 0.0,
                        "sustainable_tps_per_die": float(best_ratio["sustainable_tps_per_die"]) if best_ratio else 0.0,
                        "feasible": bool(best_ratio is not None),
                        "infeasible_reason": (
                            None if best_ratio is not None
                            else (rep["candidates"][0]["infeasible_reason"] if rep["candidates"] else "infeasible")
                        ),
                    }
                    rows.append(row)
                    if row["feasible"]:
                        feasible_entries.append((row, p_raw, d_raw))
    return rows, feasible_entries


def _validate_optimal_deployment_inputs(
    *,
    num_devices: int | None,
    prompt_tokens: int | None,
    output_tokens: int | None,
    ttft_sla_ms: float | None,
    tpot_sla_ms: float | None,
    deployment: str,
    pd_transfer_mode: str | None,
    pool_ssd_ratio: float,
    ssd_bw_gbps: float | None,
    allow_unconstrained_ttft: bool,
) -> tuple[int, int, int, float, float, str]:
    """Validate Domain-1 scenario, SLA, SSD, and deployment-mode parameters."""
    missing = [
        name for name, val in (
            ("num_devices", num_devices),
            ("prompt_tokens", prompt_tokens),
            ("output_tokens", output_tokens),
            *([] if allow_unconstrained_ttft else [("ttft_sla_ms", ttft_sla_ms)]),
            ("tpot_sla_ms", tpot_sla_ms),
        ) if val is None
    ]
    if float(pool_ssd_ratio) > 0.0 and ssd_bw_gbps is None:
        missing.append("ssd_bw_gbps")
    if missing:
        raise missing_scenario_error(missing)
    n_dev = check_pos_int("num_devices", num_devices, minimum=1)
    prompt = check_pos_int("prompt_tokens", prompt_tokens, minimum=1)
    output = check_pos_int("output_tokens", output_tokens, minimum=1)
    ttft = (
        math.inf if (ttft_sla_ms is None and allow_unconstrained_ttft)
        else check_pos_float("ttft_sla_ms", ttft_sla_ms)
    )
    tpot = check_pos_float("tpot_sla_ms", tpot_sla_ms)
    raw_dep = str(deployment or "auto").strip().lower()
    dep_norm = "pd_disaggregated" if raw_dep == "pd" else raw_dep
    if dep_norm not in {"auto", "pd_disaggregated", "colocated"}:
        raise ModelingError(f"invalid_deployment_mode:{deployment}")
    if (pd_transfer_mode is not None
            and str(pd_transfer_mode).strip().lower() not in (VALID_PD_TRANSFER_MODES | {"auto"})):
        raise ModelingError(f"invalid_pd_transfer_mode:{pd_transfer_mode}")
    return n_dev, prompt, output, ttft, tpot, dep_norm


def solve_optimal_deployment(
    model: str | Mapping[str, Any],
    hardware: str | Mapping[str, Any] = "a3",
    *,
    num_devices: int | None = None,
    prompt_tokens: int | None = None,
    output_tokens: int | None = None,
    ttft_sla_ms: float | None = None,
    tpot_sla_ms: float | None = None,
    deployment: str = "auto",
    pd_transfer_mode: str | None = None,
    kv_transfer_bw_gbps: float | None = None,
    pool_dram_ratio: float | None = None,
    pool_ssd_ratio: float = 0.0,
    pool_prefix_hit_ratio: float = 0.0,
    ddr_bw_gbps: float | None = None,
    ssd_bw_gbps: float | None = None,
    pinned_parallel: Mapping[str, int] | None = None,
    pinned_prefill_parallel: Mapping[str, int] | None = None,
    pinned_decode_parallel: Mapping[str, int] | None = None,
    mtp_proposal_tokens: int | None = None,
    mtp_acceptance_length: float | None = None,
    max_sequences_per_dp: int | None = None,
    quant: str | Mapping[str, Any] | None = None,
    layers: int | None = None,
    ranks_per_node: int | None = None,
    topology_preset: str | None = None,
    va_recommended: bool | None = None,
    raw_eager: bool = False,
    feature_track: str = "merged",
    runtime_stack: str | Mapping[str, Any] = "vllm-ascend",
    optimizations: Sequence[str] = (),
    exclude_optimizations: Sequence[str] = (),
    engram_cpu_offload: bool | None = None,
    multistream_overlap_shared_expert: bool | None = None,
    enable_shared_expert_dp: bool | None = None,
    sp: bool | None = None,
    graph_capture_mode: str | None = None,
    kv_cache_dtype: str | None = None,
    indexer_kv_dtype: str | None = None,
    allow_unconstrained_ttft: bool = False,
) -> dict[str, Any]:
    """Jointly search PD Disaggregated vs Colocated + MTP under Domain 1/2/3."""
    n_dev, prompt, output, ttft, tpot, dep_norm = _validate_optimal_deployment_inputs(
        num_devices=num_devices, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        ttft_sla_ms=ttft_sla_ms, tpot_sla_ms=tpot_sla_ms, deployment=deployment,
        pd_transfer_mode=pd_transfer_mode, pool_ssd_ratio=pool_ssd_ratio,
        ssd_bw_gbps=ssd_bw_gbps, allow_unconstrained_ttft=allow_unconstrained_ttft,
    )

    from mindie_sim.compose import compose_case
    from mindie_sim.compose_inputs import _resolve_hardware, _resolve_model_spec
    from mindie_sim.compose_runtime import va_recommended_config

    sku, _, _ = _resolve_hardware(hardware)
    model_spec = _resolve_model_spec(model)
    model_id = str(model) if isinstance(model, str) else str(model_spec.name)
    va_prof = va_recommended_config(model_id, sku)
    rec_topo = va_prof.get("recommended_topology")
    is_moe = any(plan.ffn == "moe" for plan in model_spec.layer_plans)
    eff_layers = int(layers) if layers is not None else int(model_spec.layers)
    prof_mtp = int(va_prof.get("mtp_proposal_tokens", 0) or 0)
    mtp_grid: tuple[int, ...] = (
        (0,) if (raw_eager or mtp_proposal_tokens == 0)
        else (int(mtp_proposal_tokens),) if mtp_proposal_tokens is not None
        else ((0, prof_mtp) if prof_mtp > 0 else (0,))
    )
    common_kw: dict[str, Any] = dict(
        model=model, hardware=hardware, prompt_tokens=prompt, output_tokens=output,
        ttft_slo_ms=(None if math.isinf(ttft) else ttft), tpot_slo_ms=tpot,
        quant=quant, layers=layers,
        ranks_per_node=ranks_per_node, topology_preset=topology_preset,
        va_recommended=va_recommended, raw_eager=raw_eager, feature_track=feature_track,
        runtime_stack=runtime_stack, optimizations=optimizations,
        exclude_optimizations=exclude_optimizations, engram_cpu_offload=engram_cpu_offload,
        multistream_overlap_shared_expert=multistream_overlap_shared_expert,
        enable_shared_expert_dp=enable_shared_expert_dp, sp=sp,
        graph_capture_mode=graph_capture_mode, kv_cache_dtype=kv_cache_dtype,
        indexer_kv_dtype=indexer_kv_dtype, parallel="fixed",
    )

    def compose_for(topo: Mapping[str, int], mtp_n: int, role: str = "colocated") -> dict[str, Any] | None:
        accept = (
            float(mtp_acceptance_length) if mtp_acceptance_length is not None
            else (compute_speculative_accept_step(DEFAULT_SPECULATIVE_ACCEPT_RATE, mtp_n) if mtp_n > 0 else 1.0)
        )
        try:
            return compose_case(
                **common_kw, role=role, **{k: int(v) for k, v in topo.items()},
                mtp_proposal_tokens=mtp_n, mtp_acceptance_length=accept,
            )
        except (ModelingError, ValueError):
            return None

    coloc_pin = dict(pinned_parallel or {})
    p_pin = dict(pinned_prefill_parallel if pinned_prefill_parallel is not None else coloc_pin)
    d_pin = dict(pinned_decode_parallel if pinned_decode_parallel is not None else coloc_pin)
    all_candidates: list[dict[str, Any]] = []
    coloc_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pd_entries: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    if dep_norm in {"auto", "colocated"}:
        c_rows, coloc_entries = _search_colocated_deployments(
            compose_for, num_devices=n_dev, is_moe=is_moe, layers=eff_layers,
            rec_topo=rec_topo, pinned=coloc_pin, mtp_grid=mtp_grid,
            prompt=prompt, output=output, ttft=ttft, tpot=tpot,
            max_sequences_per_dp=max_sequences_per_dp,
        )
        all_candidates.extend(c_rows)
    if dep_norm in {"auto", "pd_disaggregated"} and n_dev >= 2:
        p_rows, pd_entries = _search_pd_deployments(
            compose_for, num_devices=n_dev, is_moe=is_moe, layers=eff_layers,
            rec_topo=rec_topo, p_pin=p_pin, d_pin=d_pin, mtp_grid=mtp_grid,
            prompt=prompt, output=output, ttft=ttft, tpot=tpot,
            pd_transfer_mode=pd_transfer_mode, kv_transfer_bw_gbps=kv_transfer_bw_gbps,
            pool_dram_ratio=pool_dram_ratio, pool_ssd_ratio=pool_ssd_ratio,
            pool_prefix_hit_ratio=pool_prefix_hit_ratio, ddr_bw_gbps=ddr_bw_gbps,
            ssd_bw_gbps=ssd_bw_gbps, max_sequences_per_dp=max_sequences_per_dp,
        )
        all_candidates.extend(p_rows)

    rank_key = lambda row: (float(row["sustainable_tps"]), -int(row["used_dies"]), -float(row["effective_tpot_ms"]))
    best_coloc_entry = max(coloc_entries, key=lambda item: rank_key(item[0]), default=None)
    best_pd_entry = max(pd_entries, key=lambda item: rank_key(item[0]), default=None)
    winner_row = recommended_case = prefill_case_out = decode_case_out = None
    if best_pd_entry and (best_coloc_entry is None or rank_key(best_pd_entry[0]) >= rank_key(best_coloc_entry[0])):
        winner_row, prefill_case_out, decode_case_out = best_pd_entry[0], deepcopy(best_pd_entry[1]), deepcopy(best_pd_entry[2])
        recommended_case = deepcopy(best_pd_entry[2])
    elif best_coloc_entry:
        winner_row, recommended_case = best_coloc_entry[0], deepcopy(best_coloc_entry[1])
    elif all_candidates:
        recommended_case = compose_for({"dp": 1, "tp": 1, "ep": 1, "pp": 1, "pcp": 1, "dcp": 1}, 0)

    return {
        "schema": "mindie-sim.optimal-deployment/1",
        "model": model_id, "hardware": sku, "num_devices": n_dev,
        "prompt_tokens": prompt, "output_tokens": output,
        "ttft_sla_ms": (None if math.isinf(ttft) else ttft), "tpot_sla_ms": tpot,
        "deployment_constraint": dep_norm,
        "winner_mode": winner_row["deployment_mode"] if winner_row else None,
        "feasible": winner_row is not None,
        "best_configuration": winner_row,
        "best_pd": best_pd_entry[0] if best_pd_entry else None,
        "best_colocated": best_coloc_entry[0] if best_coloc_entry else None,
        "pareto_frontier": _mark_pd_pareto(all_candidates),
        "candidates": all_candidates,
        "recommended_case": recommended_case,
        "prefill_case": prefill_case_out,
        "decode_case": decode_case_out,
    }


__all__ = [
    "AUTO_PD_TRANSFER_MODES",
    "VALID_PD_TRANSFER_MODES",
    "solve_optimal_deployment",
    "sweep_pd_disaggregation_ratios",
]
