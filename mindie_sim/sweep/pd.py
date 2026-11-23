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

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import ctypes
import gc
import math
import re
import sys
from typing import Any

from mindie_sim.catalog.context_parallel import (
    legal_vllm_ascend_dcp_values,
    validate_vllm_ascend_dcp_codomain,
)
from mindie_sim.optimization.causal_tail import (
    compute_acceptance_tail_closure,
    resolve_barrier_predecessors,
)
from mindie_sim.optimization.dual_bound import NS_PER_SECOND, inward_floor_ns
from mindie_sim.optimization.kernel_synthesis import (
    TargetOpNode,
    load_rewrite_plan,
    solve_target_graph,
)
from mindie_sim.prediction.body_defaults import TargetGraphSet
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.disaggregated.handoff import compute_pd_handoff_plan
from mindie_sim.prediction.efficiency import (
    _lower_and_run_graph,
    _resolve_speculative_tokens,
    extract_hardware_peaks,
)
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.execution.layer_runs import (
    LayerCohortPlan,
    discover_layer_cohorts,
    execute_native_cohort_graph,
)
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.runtime.exposed_observer import (
    ObservedInterval,
    attribute_stream0_intervals,
)
from mindie_sim.prediction.sharding.local_tp import rank_zero_graph_reuse
from mindie_sim.prediction.offload.policy import (
    _DEFAULT_SSD_BANDWIDTH_UTILIZATION,
    _DEFAULT_SSD_LATENCY_US,
    missing_scenario_error,
    resolve_sku_host_dma_rates,
)
from mindie_sim.prediction.frontend.speculation import compute_speculative_accept_step
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.sweep.resize import check_pos_float, check_pos_int

MS_PER_S = 1000.0
US_PER_MS = 1000.0
US_PER_SECOND = 1_000_000.0
NS_PER_US = 1000.0
BYTES_PER_GIB = float(1024 ** 3)
BYTES_PER_MS_PER_GBPS = 1_000_000.0
PARETO_EPS = 1e-12
HALF_RATIO = 0.5
DEFAULT_ELEM_BYTES = 2.0
OVERLAP_BARRIER_EPS_US = 1e-3
DEMAND_CHANNELS = 5
LAUNCH_SHORT_US = 12.0
LAUNCH_WORK_SHARE = 0.35
RANK_TPS_DECIMALS = 6
SCREEN_TTFT_MARGIN = 1.15
SCREEN_TPOT_MARGIN = 1.25
SCREEN_RECLAIM_STRIDE = 4
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
ARCHETYPE_SCREEN_MIN_LAYERS = 12
ARCHETYPE_FOLD_MIN_LAYERS = 2
COLOCATED_STAGE2_FINALISTS = 3
PD_PREFILL_STAGE2_FINALISTS = 3
PD_DECODE_STAGE2_FINALISTS = 4
SCREEN_DECODE_BATCH_PROBE = 8
SLA_SLACK_BOUND_TOLERANCE_MS = 0.5
_OP_LAYER_PREFIX_RE = re.compile(r"^(?:draft\.)?s\d+\.l\d+\.")
_POW2_WIDTHS = (1, 2, 4, 8, 16)
_PREFILL_PCP_WIDTHS = (1, 2, 4)
PREFILL_PCP_MIN_PROMPT_TOKENS = 1024
DECODE_DCP_MIN_PROMPT_TOKENS = 2048
DEFAULT_CHUNKED_PREFILL_TOKENS = 8192
DEFAULT_CHUNKED_BATCH_SEQ_FLOOR = 64
PREFILL_DIRECT_MAX_TOKENS = 1024
PREFILL_CHUNK_PROBE_TOKENS = 512
GC_RECLAIM_MIN_WORLD_SIZE = 8


_RECLAIM_LRU_PREFIXES = (
    "mindie_sim.prediction.operators.",
    "mindie_sim.prediction.ownership",
    "mindie_sim.prediction.sharding.",
    "mindie_sim.prediction.runtime.memory_physics",
    "mindie_sim.prediction.engine_periodic.",
)


def _reclaim_sweep_memory() -> None:
    """Release transient structural memos, cyclic GC garbage, and Linux glibc heap pages."""
    from mindie_sim.prediction.execution.runner import Execution
    from mindie_sim.prediction.execution.streams import clear_stream_caches
    from mindie_sim.prediction.frontend.body_spec import _GRAPH_CACHE
    from mindie_sim.prediction.frontend.features import _NORMAL_SUMMARY_CACHE
    from mindie_sim.prediction.operators.matrix.matrix_mapping import _SELECT_CACHE
    from mindie_sim.prediction.program.invocation import _HW_INV_CACHE
    from mindie_sim.prediction.program.physical_key import _METADATA_MEMO, _SIGNATURE_MEMO
    from mindie_sim.prediction.projections import _REPLICATED_MEMO
    from mindie_sim.prediction.runtime.engine_types import _UNITS_MASK_CACHE
    from mindie_sim.prediction.structure import _PARSE_MEMO
    from mindie_sim.templates.catlass_lower import _PURE_LOWER_CACHE

    clear_stream_caches()
    for cache in (
        Execution._isolated_scripts, _SELECT_CACHE, _PARSE_MEMO, _UNITS_MASK_CACHE,
        _NORMAL_SUMMARY_CACHE, _GRAPH_CACHE, _METADATA_MEMO, _SIGNATURE_MEMO,
        _PURE_LOWER_CACHE, _HW_INV_CACHE, _REPLICATED_MEMO,
    ):
        cache.clear()
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or mod_name.endswith(".registry"):
            continue
        if not mod_name.startswith(_RECLAIM_LRU_PREFIXES):
            continue
        for obj in vars(mod).values():
            clear_fn = getattr(obj, "cache_clear", None)
            if callable(clear_fn):
                clear_fn()
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except TimeoutError:
            raise
        except (OSError, AttributeError):
            pass


def _prefill_workspace_tokens(parsed: CaseSpec, compute_tokens: int) -> int:
    """Bound prefill activation workspace to chunk size when chunked prefill is active."""
    if compute_tokens <= 0:
        return 1
    if parsed.runtime.enable_chunked_prefill and int(parsed.runtime.max_num_batched_tokens) > 0:
        return max(1, min(int(compute_tokens), int(parsed.runtime.max_num_batched_tokens)))
    return max(1, int(compute_tokens))


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
    p_phys_rpn = int(getattr(p_parsed.machine, "physical_ranks_per_node", 0) or p_rpn)
    d_phys_rpn = int(getattr(d_parsed.machine, "physical_ranks_per_node", 0) or d_rpn)
    same_domain = int(total_dies) <= min(p_phys_rpn, d_phys_rpn)
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


def _prefill_cache_key(
    cache_id: int,
    compute_tokens: int,
    hit_tokens: int,
    apply_target_envelope: bool,
    rewrite_plan: Any,
    allow_unbound_formula: bool,
) -> tuple[Any, ...]:
    """Build a mode- and rewrite-scoped cache key for prefill stage measurements."""
    return (
        cache_id, compute_tokens, hit_tokens,
        bool(apply_target_envelope), id(rewrite_plan), bool(allow_unbound_formula),
    )


def _eval_prefill_with_plan(
    parsed: CaseSpec,
    *,
    compute_tokens: int,
    hit_tokens: int,
    min_layers: int,
    rep_info: tuple[CaseSpec, dict[int, int], float] | None = None,
    eval_ctx_box: list[EvaluationContext | None] | None = None,
    reclaim_on_exit: bool = True,
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
    allow_plan_fallback: bool = False,
) -> tuple[float, bool]:
    """Price prefill with optional ``rewrite_plan``, falling back only on decode-scoped plan mismatch."""
    try:
        ms = _prefill_step_ms(
            parsed, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
            min_layers=min_layers, rep_info=rep_info, eval_ctx_box=eval_ctx_box,
            reclaim_on_exit=reclaim_on_exit,
            apply_target_envelope=apply_target_envelope,
            rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
        )
        return ms, (rewrite_plan is not None)
    except (ModelingError, ValueError) as exc:
        if allow_plan_fallback and rewrite_plan is not None and _is_plan_phase_mismatch(exc):
            ms = _prefill_step_ms(
                parsed, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
                min_layers=min_layers, rep_info=rep_info, eval_ctx_box=eval_ctx_box,
                reclaim_on_exit=reclaim_on_exit,
                apply_target_envelope=apply_target_envelope,
                rewrite_plan=None,
                allow_unbound_formula=allow_unbound_formula,
            )
            return ms, False
        raise


def _measure_prefill_stage(
    p_parsed: CaseSpec,
    *,
    prompt: int,
    compute_tokens: int,
    hit_tokens: int,
    cache: dict[tuple[Any, ...], tuple[Memory, bool, int, float]] | None,
    cache_id: int | None,
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
    allow_plan_fallback: bool = False,
) -> tuple[Memory, bool, int, float]:
    key = (
        _prefill_cache_key(
            cache_id, compute_tokens, hit_tokens,
            apply_target_envelope, rewrite_plan, allow_unbound_formula,
        )
        if (cache is not None and cache_id is not None) else None
    )
    if key is not None and key in cache:
        return cache[key]
    p_mem = Memory(p_parsed, workspace_tokens=_prefill_workspace_tokens(p_parsed, compute_tokens))
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
    if compute_tokens > 0 and prefill_hbm_ok:
        p_step, _ = _eval_prefill_with_plan(
            p_parsed, compute_tokens=compute_tokens,
            hit_tokens=hit_tokens, min_layers=ARCHETYPE_FOLD_MIN_LAYERS,
            apply_target_envelope=apply_target_envelope,
            rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
            allow_plan_fallback=allow_plan_fallback,
        )
    elif compute_tokens > 0:
        p_step = math.inf
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
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
    probe_cache: Mapping[int, float] | None = None,
    probe_sink: dict[int, float] | None = None,
    vram_info: tuple[Memory, tuple[float, ...], int] | None = None,
    rep_info: tuple[CaseSpec, dict[int, int], float] | None = None,
) -> dict[str, Any]:
    d_par = d_parsed.runtime.parallel
    dcp_ok, dcp_reason = validate_vllm_ascend_dcp_codomain(d_parsed.model, d_par)
    handoff_tpot = exposed_kv_ms / float(decode_tokens)
    decode_budget = tpot_sla_ms - handoff_tpot
    if not dcp_ok:
        return {
            "decode_budget": decode_budget, "length_ok": True,
            "b_vram": 0, "b_sla": 0, "b_feasible": 0,
            "b1_step": 0.0, "decode_step": 0.0,
            "handoff_tpot_ms": float(handoff_tpot),
            "effective_tpot": 0.0, "first_itl": exposed_kv_ms,
            "tpot_ok": False, "rps_d": 0.0, "raw_rps_d": 0.0,
            "decode_tps": 0.0, "decode_tps_per_die": 0.0,
            "avg_accepted_tokens": 1.0,
            "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
            "unused_sla_slack_ms": 0.0, "operating_regime": "SLA_LATENCY_BOUND",
            "free_hbm_headroom_gib": 0.0,
            "dcp_codomain_valid": False, "dcp_codomain_reason": dcp_reason,
        }
    d_dp = int(d_par.dp)
    _, d_emit_est = _resolve_speculative_tokens(
        d_parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    step_budget = decode_budget * max(1.0, float(d_emit_est))
    eff_sla_ms = (
        max(step_budget, PARETO_EPS) if step_budget > 0.0
        else tpot_sla_ms * max(1.0, float(d_emit_est))
    )
    d_sla_key = (cache_id, eff_sla_ms) if (cache is not None and cache_id is not None) else None
    if d_sla_key is not None and d_sla_key in cache:
        d_sla = cache[d_sla_key]
    else:
        d_sla = _solve_sla_archetype_aware(
            d_parsed, sla_ms=eff_sla_ms, context=eff_decode,
            output_budget=output, max_sequences_per_dp=max_sequences_per_dp,
            apply_target_envelope=apply_target_envelope,
            rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
            probe_cache=probe_cache,
            probe_sink=probe_sink,
            vram_info=vram_info,
            rep_info=rep_info,
        )
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
    tpot_ok = bool(dcp_ok and b_feasible >= 1 and effective_tpot <= tpot_sla_ms + PARETO_EPS)
    rps_d = (
        (b_feasible * d_dp * emitted * MS_PER_S)
        / (max(decode_step, PARETO_EPS) * float(decode_tokens))
        if tpot_ok else 0.0
    )
    raw_rps_d = (
        (max(1, b_feasible) * d_dp * emitted * MS_PER_S)
        / (max(decode_step, PARETO_EPS) * float(decode_tokens))
    )
    per_dp_dies = max(1, int(d_par.tp) * int(d_par.pcp) * int(d_par.pp))
    n_tok_per_die = (float(max(1, b_feasible)) * emitted) / float(per_dp_dies) if b_feasible >= 1 else 0.0
    us_per_tok_die = (decode_step * US_PER_MS) / n_tok_per_die if n_tok_per_die > 0.0 else 0.0
    decode_tps_per_die = (US_PER_SECOND / us_per_tok_die) if (tpot_ok and us_per_tok_die > 0.0) else 0.0
    unused_sla_slack_ms = max(0.0, float(tpot_sla_ms) - float(effective_tpot)) if tpot_ok else 0.0
    operating_regime = (
        "HBM_CAPACITY_BOUND"
        if (tpot_ok and b_vram >= 1 and b_feasible >= b_vram
            and unused_sla_slack_ms > SLA_SLACK_BOUND_TOLERANCE_MS)
        else "SLA_LATENCY_BOUND"
    )
    free_hbm_gib = float(d_scen.get("free_hbm_headroom_gib", 0.0) or 0.0)
    return {
        "decode_budget": decode_budget, "length_ok": length_ok,
        "b_vram": b_vram, "b_sla": b_sla, "b_feasible": b_feasible,
        "b1_step": b1_step, "decode_step": decode_step,
        "handoff_tpot_ms": float(handoff_tpot),
        "effective_tpot": effective_tpot, "first_itl": first_itl,
        "tpot_ok": tpot_ok, "rps_d": rps_d, "raw_rps_d": raw_rps_d,
        "decode_tps": rps_d * float(output),
        "decode_tps_per_die": decode_tps_per_die,
        "avg_accepted_tokens": emitted,
        "n_tok_per_die": n_tok_per_die,
        "us_per_tok_die": us_per_tok_die,
        "unused_sla_slack_ms": unused_sla_slack_ms,
        "operating_regime": operating_regime,
        "free_hbm_headroom_gib": free_hbm_gib,
        "dcp_codomain_valid": True,
        "dcp_codomain_reason": None,
    }


def _prefill_only_pd_stage_result(
    *, p_dp: int, p_step: float, p_slots: int, prefill_hbm_ok: bool,
    prefill_role_error: str | None, p_dcp_ok: bool, p_dcp_reason: str | None,
    ttft_sla_ms: float, tpot_sla_ms: float, compute_tokens: int, hit_tokens: int,
    output: int, d_parsed: CaseSpec, link_info: dict[str, Any],
) -> dict[str, Any]:
    prefill_ok = bool(p_step <= ttft_sla_ms)
    rps_p = (
        (p_dp * MS_PER_S) / max(p_step, PARETO_EPS)
        if (prefill_ok and prefill_hbm_ok and p_step > 0.0)
        else (float(p_dp * p_slots) * MS_PER_S if (prefill_ok and prefill_hbm_ok) else 0.0)
    )
    infeasible = (
        prefill_role_error if prefill_role_error is not None
        else "prefill_hbm_oom" if not prefill_hbm_ok
        else "prefill_ttft_sla_exceeded" if not prefill_ok else None
    )
    return {
        "p_step": p_step, "effective_ttft_ms": p_step,
        "prefill_hbm_ok": prefill_hbm_ok, "prefill_ok": prefill_ok,
        "compute_tokens": compute_tokens, "hit_tokens": hit_tokens,
        "kv_bytes": 0.0, "unique_kv_bytes": 0.0, "kv_ms": 0.0,
        "exposed_kv_ms": 0.0, "tail_layer_kv_ms": 0.0,
        "rps_p": rps_p, "rps_d": rps_p, "raw_rps_d": rps_p,
        "decode_tps": rps_p * float(output), "decode_tps_per_die": 0.0,
        "p_kv_rps_per_inst": rps_p, "d_kv_rps_per_inst": None,
        "b_sla": 0, "b_vram": int(d_parsed.runtime.max_num_seqs), "b_feasible": 0,
        "b1_step": 0.0, "decode_step": 0.0, "handoff_tpot_ms": 0.0, "effective_tpot": 0.0, "first_itl": 0.0,
        "decode_budget": tpot_sla_ms, "tpot_ok": True, "infeasible": infeasible,
        "link_info": link_info, "pool_metrics": None, "handoff_plan": None,
        "avg_accepted_tokens": 1.0, "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
        "unused_sla_slack_ms": 0.0, "operating_regime": "SLA_LATENCY_BOUND",
        "free_hbm_headroom_gib": 0.0,
        "prefill_dcp_codomain_valid": bool(p_dcp_ok), "prefill_dcp_codomain_reason": p_dcp_reason,
        "dcp_codomain_valid": True, "dcp_codomain_reason": None,
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
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
    _prefill_step_cache: dict[tuple[int, int, int], tuple[Memory, bool, int, float]] | None = None,
    _prefill_cache_id: int | None = None,
    _decode_sla_cache: dict[tuple[int, float], dict[str, Any]] | None = None,
    _decode_cache_id: int | None = None,
    _decode_probe_cache: dict[int, float] | None = None,
    _decode_vram_info: tuple[Memory, tuple[float, ...], int] | None = None,
    _decode_rep_info: tuple[CaseSpec, dict[int, int], float] | None = None,
) -> dict[str, Any]:
    """Evaluate prefill, KV handoff / 3-tier pool, and decode stage rates."""
    p_dp = int(p_parsed.runtime.parallel.dp)
    p_dcp_ok, p_dcp_reason = validate_vllm_ascend_dcp_codomain(p_parsed.model, p_parsed.runtime.parallel)
    prefill_role_error = None if p_dcp_ok else f"prefill:{p_dcp_reason}"
    hit_tokens = min(int(math.floor(prompt * hit_r)), max(0, prompt - 1)) if mode == "pooled_dram_ssd" else 0
    compute_tokens = max(0, prompt - hit_tokens)
    p_mem, prefill_hbm_ok, p_slots, p_step = _measure_prefill_stage(
        p_parsed, prompt=prompt, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
        cache=_prefill_step_cache, cache_id=_prefill_cache_id,
        apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
        allow_unbound_formula=allow_unbound_formula, allow_plan_fallback=(output > 1),
    )
    decode_q_tokens, _ = _resolve_speculative_tokens(
        d_parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    d_mem = Memory(d_parsed, workspace_tokens=decode_q_tokens)
    decode_tokens = output - 1
    link_info = _resolve_pd_link_bandwidth(
        p_parsed, d_parsed, total_dies=total, kv_transfer_bw_gbps=kv_transfer_bw_gbps,
    )
    if decode_tokens <= 0:
        return _prefill_only_pd_stage_result(
            p_dp=p_dp, p_step=p_step, p_slots=p_slots, prefill_hbm_ok=prefill_hbm_ok,
            prefill_role_error=prefill_role_error, p_dcp_ok=p_dcp_ok, p_dcp_reason=p_dcp_reason,
            ttft_sla_ms=ttft_sla_ms, tpot_sla_ms=tpot_sla_ms, compute_tokens=compute_tokens,
            hit_tokens=hit_tokens, output=output, d_parsed=d_parsed, link_info=link_info,
        )

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

    prefill_compute_ok = bool(p_step <= ttft_sla_ms)
    effective_ttft_ms = p_step + (exposed_kv_ms if mode == "direct_blocking_d2d" else 0.0)
    prefill_ok = bool(effective_ttft_ms <= ttft_sla_ms)
    rps_p = (p_dp * per_rep) if (prefill_ok and prefill_hbm_ok) else 0.0
    dec = _measure_decode_stage(
        d_parsed, prompt=prompt, output=output, eff_decode=eff_decode,
        tpot_sla_ms=tpot_sla_ms, exposed_kv_ms=exposed_kv_ms, decode_tokens=decode_tokens,
        max_sequences_per_dp=max_sequences_per_dp,
        cache=_decode_sla_cache, cache_id=_decode_cache_id,
        apply_target_envelope=apply_target_envelope,
        rewrite_plan=rewrite_plan,
        allow_unbound_formula=allow_unbound_formula,
        probe_cache=_decode_probe_cache,
        probe_sink=_decode_probe_cache,
        vram_info=_decode_vram_info,
        rep_info=_decode_rep_info,
    )
    infeasible = (
        prefill_role_error if prefill_role_error is not None
        else dec["dcp_codomain_reason"] if not dec.get("dcp_codomain_valid", True)
        else "prefill_hbm_oom" if not prefill_hbm_ok
        else "prefill_ttft_sla_exceeded" if not prefill_compute_ok
        else "decode_max_model_len_exceeded" if not dec["length_ok"]
        else "decode_hbm_oom" if dec["b_vram"] < 1
        else "effective_tpot_sla_exceeded" if not dec["tpot_ok"]
        else "prefill_ttft_sla_exceeded" if not prefill_ok
        else None
    )
    return {
        "p_step": p_step, "effective_ttft_ms": float(effective_ttft_ms),
        "prefill_hbm_ok": prefill_hbm_ok, "prefill_ok": prefill_ok,
        "compute_tokens": compute_tokens, "hit_tokens": hit_tokens,
        "kv_bytes": kv_bytes, "unique_kv_bytes": float(handoff["unique_kv_bytes"]),
        "kv_ms": kv_ms, "exposed_kv_ms": exposed_kv_ms, "tail_layer_kv_ms": tail_layer_kv_ms,
        "rps_p": rps_p, "p_kv_rps_per_inst": p_kv_rps_per_inst, "d_kv_rps_per_inst": d_kv_rps_per_inst,
        "infeasible": infeasible, "link_info": link_info,
        "pool_metrics": pool_metrics, "handoff_plan": handoff,
        "prefill_dcp_codomain_valid": bool(p_dcp_ok),
        "prefill_dcp_codomain_reason": p_dcp_reason,
        **dec,
    }


def _single_die_decode_tps(row: Mapping[str, Any]) -> float:
    """Return decode-step single-die throughput (``1e6 / us_per_tok_die``) for Iso-SLA ranking."""
    if not row.get("feasible", True):
        return 0.0
    dec_tps = float(row.get("decode_tps_per_die", 0.0) or 0.0)
    if dec_tps > 0.0:
        return dec_tps
    us_per_tok = float(row.get("us_per_tok_die", 0.0) or 0.0)
    if us_per_tok > 0.0:
        return US_PER_SECOND / us_per_tok
    if float(row.get("decode_step_ms", 0.0) or 0.0) > 0.0:
        return 0.0
    return float(row.get("e2e_tps_per_die", 0.0) or 0.0)


def _mark_pd_pareto(
    rows: list[dict[str, Any]],
    *,
    metric_fn: Any = None,
) -> list[dict[str, Any]]:
    val_fn = metric_fn if callable(metric_fn) else (lambda r: float(r["sustainable_tps"]))
    feasible_rows = [row for row in rows if row["feasible"]]
    for row in rows:
        if not row["feasible"]:
            row["pareto_optimal"] = False
            continue
        row_val = float(val_fn(row))
        dominated = False
        for other in feasible_rows:
            if other is row:
                continue
            other_val = float(val_fn(other))
            ge_all = (
                other_val >= row_val - PARETO_EPS
                and int(other["used_dies"]) <= int(row["used_dies"])
            )
            strict = (
                other_val > row_val + PARETO_EPS
                or int(other["used_dies"]) < int(row["used_dies"])
            )
            if ge_all and strict:
                dominated = True
                break
        row["pareto_optimal"] = not dominated
    return [row for row in rows if row.get("pareto_optimal")]


def _build_pd_ratio_rows(
    st: Mapping[str, Any], *, total: int, p_dies: int, d_dies: int, output: int, mode: str,
) -> list[dict[str, Any]]:
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
            e2e_per_die = sustainable / float(used)
            d_tps_per_die = float(st.get("decode_tps_per_die", 0.0) or 0.0) if feasible else 0.0
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
                "pd_transfer_mode": mode, "prefill_dies": n_p * p_dies, "decode_dies": n_d * d_dies,
                "used_dies": used, "idle_dies": total - used,
                "prefill_rps": r_p, "kv_transfer_rps": r_kv, "decode_rps": r_d,
                "sustainable_rps": r_sys, "sustainable_tps": sustainable,
                "sustainable_tps_per_die": e2e_per_die, "e2e_total_tps": sustainable,
                "e2e_tps_per_die": e2e_per_die, "decode_tps_per_die": d_tps_per_die,
                "decode_step_ms": st["decode_step"], "effective_tpot_ms": st["effective_tpot"],
                "first_decode_itl_ms": st["first_itl"], "bottleneck_stage": bottleneck,
                "feasible": feasible, "infeasible_reason": st["infeasible"], "reason": st["infeasible"],
                "metrics_boundary": "one_pipeline_stage_rate_composition",
            })
    return rows


def sweep_pd_disaggregation_ratios(
    prefill_case: CaseSpec | Mapping[str, Any], decode_case: CaseSpec | Mapping[str, Any], *,
    total_dies: int, prompt_tokens: int | None = None, output_tokens: int | None = None,
    decode_context: int | None = None, ttft_sla_ms: float | None = None,
    tpot_sla_ms: float | None = None, kv_transfer_bw_gbps: float | None = None,
    pd_transfer_mode: str = "direct_blocking_d2d", pool_dram_ratio: float | None = None,
    pool_ssd_ratio: float = 0.0, pool_prefix_hit_ratio: float = 0.0,
    ddr_bw_gbps: float | None = None, ssd_bw_gbps: float | None = None,
    max_sequences_per_dp: int | None = None, allow_unconstrained_ttft: bool = False,
    apply_target_envelope: bool = False, rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
    _prefill_step_cache: dict[tuple[int, int, int], tuple[Memory, bool, int, float]] | None = None,
    _prefill_cache_id: int | None = None,
    _decode_sla_cache: dict[tuple[int, float], dict[str, Any]] | None = None,
    _decode_cache_id: int | None = None,
    _decode_probe_cache: dict[int, float] | None = None,
    _decode_vram_info: tuple[Memory, tuple[float, ...], int] | None = None,
    _decode_rep_info: tuple[CaseSpec, dict[int, int], float] | None = None,
) -> dict[str, Any]:
    """Sweep disaggregated P/D instance counts with real handoff & 3-tier KV pooling."""
    total = check_pos_int("total_dies", total_dies, minimum=1)
    p_parsed, d_parsed = parse_case(prefill_case), parse_case(decode_case)
    prompt, output, ttft, tpot, mode, dram_r, ssd_r, hit_r = _resolve_pd_sla_and_pool_args(
        p_parsed, d_parsed, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        ttft_sla_ms=ttft_sla_ms, tpot_sla_ms=tpot_sla_ms, pd_transfer_mode=pd_transfer_mode,
        pool_dram_ratio=pool_dram_ratio, pool_ssd_ratio=pool_ssd_ratio,
        pool_prefix_hit_ratio=pool_prefix_hit_ratio, ssd_bw_gbps=ssd_bw_gbps,
        allow_unconstrained_ttft=allow_unconstrained_ttft,
    )
    eff_decode = prompt if decode_context is None else check_pos_int("decode_context", decode_context, minimum=0)
    st = _evaluate_pd_stages(
        p_parsed, d_parsed, total=total, prompt=prompt, output=output, eff_decode=eff_decode,
        ttft_sla_ms=ttft, tpot_sla_ms=tpot, mode=mode, dram_r=dram_r, ssd_r=ssd_r, hit_r=hit_r,
        kv_transfer_bw_gbps=kv_transfer_bw_gbps, ddr_bw_gbps=ddr_bw_gbps, ssd_bw_gbps=ssd_bw_gbps,
        max_sequences_per_dp=max_sequences_per_dp, apply_target_envelope=apply_target_envelope,
        rewrite_plan=rewrite_plan, allow_unbound_formula=allow_unbound_formula,
        _prefill_step_cache=_prefill_step_cache, _prefill_cache_id=_prefill_cache_id,
        _decode_sla_cache=_decode_sla_cache, _decode_cache_id=_decode_cache_id,
        _decode_probe_cache=_decode_probe_cache, _decode_vram_info=_decode_vram_info,
        _decode_rep_info=_decode_rep_info,
    )
    p_par, d_par = p_parsed.runtime.parallel, d_parsed.runtime.parallel
    p_dies, d_dies = p_par.world_size, d_par.world_size
    rows = _build_pd_ratio_rows(st, total=total, p_dies=p_dies, d_dies=d_dies, output=output, mode=mode)
    front = _mark_pd_pareto(rows)
    best = max(
        front,
        key=lambda row: (float(row["e2e_tps_per_die"]), float(row["sustainable_tps"]), -int(row["used_dies"])),
        default=None,
    )
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
        "handoff_tpot_ms": float(st.get("handoff_tpot_ms", 0.0) or 0.0),
        "tail_layer_kv_ms": float(st["tail_layer_kv_ms"]),
        "pool_metrics": st["pool_metrics"],
        "decode_step_ms": float(st["decode_step"]),
        "effective_tpot_ms": float(st["effective_tpot"]),
        "first_decode_itl_ms": float(st["first_itl"]),
        "prefill_instance": {
            "dies_per_instance": p_dies, "parallel": p_par.to_dict(),
            "step_ms": float(st["p_step"]),
            "effective_ttft_ms": float(st.get("effective_ttft_ms", st["p_step"])),
            "hbm_ok": bool(st["prefill_hbm_ok"]),
            "ttft_sla_ok": bool(st["prefill_ok"]), "rps_per_instance": float(st["rps_p"]),
            "effective_prefill_tokens": int(st["compute_tokens"]),
            "prefix_hit_tokens": int(st["hit_tokens"]),
            "dcp_codomain_valid": bool(st.get("prefill_dcp_codomain_valid", True)),
            "dcp_codomain_reason": st.get("prefill_dcp_codomain_reason"),
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
            "decode_tps_per_die": float(st.get("decode_tps_per_die", 0.0) or 0.0),
            "rps_per_instance": float(st["rps_d"]),
            "avg_accepted_tokens": float(st["avg_accepted_tokens"]),
            "n_tok_per_die": float(st["n_tok_per_die"]),
            "us_per_tok_die": float(st["us_per_tok_die"]),
            "unused_sla_slack_ms": float(st["unused_sla_slack_ms"]),
            "operating_regime": str(st["operating_regime"]),
            "free_hbm_headroom_gib": float(st["free_hbm_headroom_gib"]),
            "dcp_codomain_valid": bool(st.get("dcp_codomain_valid", True)),
            "dcp_codomain_reason": st.get("dcp_codomain_reason"),
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


def _dcp_choices_for_candidate(
    model_spec: Any,
    *,
    tp: int,
    pcp: int,
    prompt_tokens: int,
    pinned: Mapping[str, int],
) -> tuple[int, ...]:
    if "dcp" in pinned:
        cand_dcp = max(1, int(pinned["dcp"]))
        if model_spec is not None:
            ok, _ = validate_vllm_ascend_dcp_codomain(
                model_spec, ParallelSpec(tp=tp, pcp=pcp, cp=cand_dcp),
            )
            if not ok:
                return ()
        return (cand_dcp,)
    if model_spec is None:
        return (1, tp) if (tp > 1 and pcp == 1 and prompt_tokens >= DECODE_DCP_MIN_PROMPT_TOKENS) else (1,)
    legal = legal_vllm_ascend_dcp_values(model_spec, tp=tp, pcp=pcp)
    if prompt_tokens < DECODE_DCP_MIN_PROMPT_TOKENS and pcp == 1:
        return (1,)
    return legal


def _build_colocated_candidates(
    *,
    num_devices: int,
    is_moe: bool,
    layers: int,
    rec_topo: Mapping[str, Any] | None,
    pinned: Mapping[str, int],
    model_spec: Any = None,
    prompt_tokens: int = 0,
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
        if model_spec is not None:
            dcp_ok, _ = validate_vllm_ascend_dcp_codomain(
                model_spec, ParallelSpec(dp=dp, tp=tp, ep=ep, pp=pp, pcp=pcp, cp=dcp),
            )
            if not dcp_ok:
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
    tp_vals = (
        (pinned["tp"],) if "tp" in pinned
        else tuple(reversed(tuple(w for w in _POW2_WIDTHS if w <= num_devices)))
    )
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
                raw_dcp = _dcp_choices_for_candidate(
                    model_spec, tp=tp, pcp=pcp, prompt_tokens=prompt_tokens, pinned=pinned,
                )
                dcp_choices = (
                    tuple(reversed(raw_dcp))
                    if prompt_tokens >= DECODE_DCP_MIN_PROMPT_TOKENS
                    else raw_dcp
                )
                for dp in dp_choices:
                    if dp * unit > num_devices or dp < 1:
                        continue
                    for dcp in dcp_choices:
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
    model_spec: Any = None,
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
        if model_spec is not None:
            dcp_ok, _ = validate_vllm_ascend_dcp_codomain(
                model_spec, ParallelSpec(dp=dp, tp=tp, ep=ep, pp=pp, pcp=pcp, cp=dcp),
            )
            if not dcp_ok:
                return
        key = (dp, tp, ep, pp, pcp, dcp, cand.get("o_proj_tp", 1), cand.get("lm_head_tp", 1))
        if key not in seen:
            seen.add(key)
            out.append(cand)

    seeded = _seed_from_profile_topo(rec_topo, role, max_dies=max_dies, pinned=pinned)
    if seeded:
        add(seeded)
    widths = tuple(reversed(tuple(w for w in _POW2_WIDTHS if w <= max_dies)))
    if role == "pd_prefill":
        for tp in ((pinned["tp"],) if "tp" in pinned else widths):
            for pp in ((pinned["pp"],) if "pp" in pinned else (1, 2)):
                pcp_vals = (
                    (pinned["pcp"],) if "pcp" in pinned
                    else (tuple(reversed(_PREFILL_PCP_WIDTHS))
                          if prompt_tokens >= PREFILL_PCP_MIN_PROMPT_TOKENS else (1,))
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
                raw_dcp = _dcp_choices_for_candidate(
                    model_spec, tp=tp, pcp=pcp, prompt_tokens=prompt_tokens, pinned=pinned,
                )
                dcp_choices = (
                    tuple(reversed(raw_dcp))
                    if prompt_tokens >= DECODE_DCP_MIN_PROMPT_TOKENS
                    else raw_dcp
                )
                for dcp in dcp_choices:
                    add({"dp": dp, "tp": tp, "ep": ep, "pp": pp, "pcp": pcp, "dcp": dcp})
    if len(out) > MAX_PD_STAGE_CANDIDATES:
        head = out[:1] if seeded else []
        tail = sorted(
            out[len(head):],
            key=lambda c: (
                c["dp"] * c["tp"] * c["pp"] * c["pcp"],
                int(c["dcp"] > 1) if role == "pd_decode" else int(c["pcp"] > 1),
                c["tp"],
                c["dp"],
            ),
            reverse=True,
        )
        return (head + tail)[:MAX_PD_STAGE_CANDIDATES]
    return out


def _colocated_workspace_tokens(parsed: CaseSpec, prompt_tokens: int, decode_tokens: int = 1) -> int:
    prefill_ws = (
        min(int(prompt_tokens), int(parsed.runtime.max_num_batched_tokens))
        if parsed.runtime.enable_chunked_prefill and int(parsed.runtime.max_num_batched_tokens) > 0
        else int(prompt_tokens)
    )
    return max(1, prefill_ws, int(decode_tokens))


def _colocated_infeasible_row(
    par: Any,
    *,
    used_dies: int,
    mtp_tokens: int,
    reason: str | None,
    regime: str = "SLA_LATENCY_BOUND",
    dcp_ok: bool = True,
    dcp_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "deployment_mode": "colocated", "used_dies": used_dies, "parallel": par.to_dict(),
        "mtp_proposal_tokens": mtp_tokens, "ttft_ms": 0.0, "decode_step_ms": 0.0,
        "effective_tpot_ms": 0.0, "b_feasible": 0, "b_vram_max": 0,
        "sustainable_rps": 0.0, "sustainable_tps": 0.0,
        "sustainable_tps_per_die": 0.0, "decode_tps_per_die": 0.0,
        "e2e_total_tps": 0.0, "e2e_tps_per_die": 0.0,
        "avg_accepted_tokens": 1.0, "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
        "unused_sla_slack_ms": 0.0, "operating_regime": regime,
        "free_hbm_headroom_gib": 0.0,
        "dcp_codomain_valid": bool(dcp_ok), "dcp_codomain_reason": dcp_reason,
        "feasible": False, "infeasible_reason": reason,
    }


def _colocated_prefill_only_row(
    par: ParallelSpec, *, used_dies: int, mtp_tokens: int, p_step: float,
    output_tokens: int, feasible: bool,
) -> dict[str, Any]:
    rps = (par.dp * MS_PER_S) / max(p_step, PARETO_EPS) if feasible else 0.0
    e2e_tps = rps * float(output_tokens)
    e2e_per_die = e2e_tps / float(used_dies)
    return {
        "deployment_mode": "colocated", "used_dies": used_dies, "parallel": par.to_dict(),
        "mtp_proposal_tokens": mtp_tokens, "ttft_ms": p_step, "decode_step_ms": 0.0,
        "effective_tpot_ms": 0.0, "b_feasible": 1 if feasible else 0,
        "b_vram_max": 1 if feasible else 0, "sustainable_rps": rps,
        "sustainable_tps": e2e_tps, "sustainable_tps_per_die": e2e_per_die,
        "decode_tps_per_die": 0.0, "e2e_total_tps": e2e_tps, "e2e_tps_per_die": e2e_per_die,
        "avg_accepted_tokens": 1.0, "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
        "unused_sla_slack_ms": 0.0, "operating_regime": "SLA_LATENCY_BOUND",
        "free_hbm_headroom_gib": 0.0, "dcp_codomain_valid": True, "dcp_codomain_reason": None,
        "feasible": feasible, "infeasible_reason": None if feasible else "prefill_sla_or_oom",
    }


def _evaluate_colocated_candidate(
    raw_case: Mapping[str, Any],
    *,
    prompt_tokens: int,
    output_tokens: int,
    ttft_sla_ms: float,
    tpot_sla_ms: float,
    mtp_tokens: int,
    max_sequences_per_dp: int | None = None,
    probe_cache: Mapping[int, float] | None = None,
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
) -> dict[str, Any]:
    parsed = parse_case(raw_case)
    par = parsed.runtime.parallel
    used_dies = par.world_size
    dcp_ok, dcp_reason = validate_vllm_ascend_dcp_codomain(parsed.model, par)
    if not dcp_ok:
        return _colocated_infeasible_row(
            par, used_dies=used_dies, mtp_tokens=mtp_tokens,
            reason=dcp_reason, dcp_ok=False, dcp_reason=dcp_reason,
        )
    prefill_ws = _colocated_workspace_tokens(parsed, prompt_tokens)
    p_mem = Memory(parsed, workspace_tokens=prefill_ws)
    p_kv = p_mem.cache_bytes(prompt_tokens + 1, padded=True)
    prefill_hbm_ok = bool(
        prompt_tokens <= parsed.runtime.max_model_len
        and (parsed.runtime.enable_chunked_prefill
             or prompt_tokens <= parsed.runtime.max_num_batched_tokens)
        and all(cap >= 0.0 and cap >= kv for row in p_mem.capacity for cap, kv in zip(row, p_kv))
    )
    if not prefill_hbm_ok:
        return _colocated_infeasible_row(
            par, used_dies=used_dies, mtp_tokens=mtp_tokens,
            reason="prefill_hbm_oom", regime="HBM_CAPACITY_BOUND",
        )
    rep_info = discover_layer_cohorts(parsed)
    decode_tokens = max(0, output_tokens - 1)
    ctx_box: list[EvaluationContext | None] = [None]
    try:
        if decode_tokens == 0:
            p_step, _ = _eval_prefill_with_plan(
                parsed, compute_tokens=prompt_tokens, hit_tokens=0,
                min_layers=ARCHETYPE_FOLD_MIN_LAYERS, rep_info=rep_info, eval_ctx_box=ctx_box,
                apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
                allow_unbound_formula=allow_unbound_formula, allow_plan_fallback=False,
            )
            feasible = bool(dcp_ok and prefill_hbm_ok and (p_step <= ttft_sla_ms))
            return _colocated_prefill_only_row(
                par, used_dies=used_dies, mtp_tokens=mtp_tokens,
                p_step=p_step, output_tokens=output_tokens, feasible=feasible,
            )

        _, emit_est = _resolve_speculative_tokens(
            parsed, output_budget=output_tokens, tokens_per_seq=None, emitted_tokens_per_seq=None,
        )
        defer_prefill = math.isinf(ttft_sla_ms) and rewrite_plan is None
        p_step, prefill_plan_matched = (0.0, False) if defer_prefill else _eval_prefill_with_plan(
            parsed, compute_tokens=prompt_tokens, hit_tokens=0,
            min_layers=ARCHETYPE_FOLD_MIN_LAYERS, rep_info=rep_info, eval_ctx_box=ctx_box,
            apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula, allow_plan_fallback=True,
        )
        try:
            d_sla = _solve_sla_archetype_aware(
                parsed, sla_ms=tpot_sla_ms * max(1.0, float(emit_est)),
                context=prompt_tokens, output_budget=output_tokens,
                max_sequences_per_dp=max_sequences_per_dp, workspace_tokens=prefill_ws,
                rep_info=rep_info, probe_cache=probe_cache, eval_ctx_box=ctx_box,
                apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
                allow_unbound_formula=allow_unbound_formula,
            )
        except (ModelingError, ValueError) as exc:
            if not (prefill_plan_matched and _is_plan_phase_mismatch(exc)):
                raise
            d_sla = _solve_sla_archetype_aware(
                parsed, sla_ms=tpot_sla_ms * max(1.0, float(emit_est)),
                context=prompt_tokens, output_budget=output_tokens,
                max_sequences_per_dp=max_sequences_per_dp, workspace_tokens=prefill_ws,
                rep_info=rep_info, probe_cache=probe_cache, eval_ctx_box=ctx_box,
                apply_target_envelope=apply_target_envelope, rewrite_plan=None,
                allow_unbound_formula=allow_unbound_formula,
            )
        scen = d_sla["scenarios"][0]
        if defer_prefill and int(scen["b_feasible"]) >= 1:
            p_step, _ = _eval_prefill_with_plan(
                parsed, compute_tokens=prompt_tokens, hit_tokens=0,
                min_layers=ARCHETYPE_FOLD_MIN_LAYERS, rep_info=rep_info, eval_ctx_box=ctx_box,
                apply_target_envelope=apply_target_envelope, rewrite_plan=None,
                allow_unbound_formula=allow_unbound_formula, allow_plan_fallback=True,
            )
        ttft_ok = bool(p_step <= ttft_sla_ms)
    finally:
        if ctx_box[0] is not None:
            ctx_box[0] = None
            if len(parsed.machine.dies) >= GC_RECLAIM_MIN_WORLD_SIZE or int(par.dp) > 1:
                _reclaim_sweep_memory()
    emitted = float(d_sla["emitted_tokens_per_seq"])
    b_feasible = int(scen["b_feasible"])
    b_vram = int(scen.get("b_vram_max", b_feasible) or 0)
    decode_step = float(scen["step_ms_feasible"] if b_feasible >= 1 and scen.get("step_ms_feasible") is not None else scen["step_ms_at_b1"])
    tpot_ms = decode_step / max(emitted, PARETO_EPS)
    length_ok = bool(prompt_tokens + output_tokens <= parsed.runtime.max_model_len)
    feasible = bool(dcp_ok and prefill_hbm_ok and ttft_ok and length_ok and b_feasible >= 1 and tpot_ms <= tpot_sla_ms + PARETO_EPS)
    req_compute_ms = p_step + (float(decode_tokens) * decode_step) / max(float(max(1, b_feasible)) * emitted, PARETO_EPS)
    rps = (float(par.dp) * MS_PER_S) / max(req_compute_ms, PARETO_EPS) if feasible else 0.0
    e2e_total_tps = rps * float(output_tokens)
    e2e_tps_per_die = e2e_total_tps / float(used_dies)
    per_dp_dies = max(1, int(par.tp) * int(par.pcp) * int(par.pp))
    n_tok_per_die = (float(b_feasible) * emitted) / float(per_dp_dies) if b_feasible >= 1 else 0.0
    us_per_tok_die = (decode_step * US_PER_MS) / n_tok_per_die if n_tok_per_die > 0.0 else 0.0
    decode_tps_per_die = (US_PER_SECOND / us_per_tok_die) if (feasible and us_per_tok_die > 0.0) else 0.0
    decode_total_tps = decode_tps_per_die * float(used_dies)
    unused_slack_ms = max(0.0, float(tpot_sla_ms) - float(tpot_ms)) if feasible else 0.0
    operating_regime = (
        "HBM_CAPACITY_BOUND"
        if (feasible and b_vram >= 1 and b_feasible >= b_vram
            and unused_slack_ms > SLA_SLACK_BOUND_TOLERANCE_MS)
        else "SLA_LATENCY_BOUND"
    )
    reason = (
        None if feasible else
        "prefill_hbm_oom" if not prefill_hbm_ok else
        "prefill_ttft_sla_exceeded" if not ttft_ok else
        "max_model_len_exceeded" if not length_ok else
        "decode_tpot_or_vram_exceeded"
    )
    return {
        "deployment_mode": "colocated", "used_dies": used_dies, "parallel": par.to_dict(),
        "mtp_proposal_tokens": mtp_tokens, "ttft_ms": p_step,
        "decode_step_ms": decode_step, "effective_tpot_ms": tpot_ms,
        "b_feasible": b_feasible, "b_vram_max": b_vram,
        "sustainable_rps": rps,
        "sustainable_tps": decode_total_tps,
        "sustainable_tps_per_die": decode_tps_per_die,
        "decode_tps_per_die": decode_tps_per_die,
        "e2e_total_tps": e2e_total_tps,
        "e2e_tps_per_die": e2e_tps_per_die,
        "avg_accepted_tokens": emitted,
        "n_tok_per_die": n_tok_per_die,
        "us_per_tok_die": us_per_tok_die,
        "unused_sla_slack_ms": unused_slack_ms,
        "operating_regime": operating_regime,
        "free_hbm_headroom_gib": float(scen.get("free_hbm_headroom_gib", 0.0) or 0.0),
        "dcp_codomain_valid": True,
        "dcp_codomain_reason": None,
        "feasible": feasible, "infeasible_reason": reason,
    }


def _strip_layer_site_key(op_id: str) -> str:
    return _OP_LAYER_PREFIX_RE.sub("", str(op_id))


def _attribute_full_records_stream0(
    kept: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[str]]:
    """Run conserved ``ExposedObserver`` Stream 0 attribution over real Engine intervals."""
    if not kept:
        return [], []
    stream_idx: dict[str, int] = {"main": 0}
    occ_to_id = {rec["occurrence"]: str(idx) for idx, rec in enumerate(kept)}
    intervals: list[ObservedInterval] = []
    preds: dict[str, tuple[str, ...]] = {}
    min_ns: int | None = None
    max_ns = 0
    for idx, rec in enumerate(kept):
        s_ns = max(0, int(round(float(rec["start_us"]) * NS_PER_US)))
        e_ns = max(s_ns, int(round(float(rec["end_us"]) * NS_PER_US)))
        min_ns = s_ns if min_ns is None else min(min_ns, s_ns)
        max_ns = max(max_ns, e_ns)
        st = str(rec["stream"] or "main")
        s_i = stream_idx.setdefault(st, len(stream_idx))
        oid = str(idx)
        intervals.append(ObservedInterval(
            occurrence_id=oid, stream=st, stream_index=s_i,
            start_ns=s_ns, end_ns=e_ns, ordinal=idx, label=str(rec["family"] or rec["op_id"]),
        ))
        preds[oid] = tuple(occ_to_id[p] for p in rec.get("predecessors", ()) if p in occ_to_id)
    start_i = min_ns if min_ns is not None else 0
    res = attribute_stream0_intervals(intervals, start_ns=start_i, terminal_ns=max_ns, causal_preds=preds)
    exposed_us = [float(res.exposed_ns_by_op.get(str(i), 0)) / NS_PER_US for i in range(len(kept))]
    masked = [res.masked_by.get(str(i), "") for i in range(len(kept))]
    return exposed_us, masked


def _aggregate_cohort_target_nodes(
    nodes: Sequence[TargetOpNode],
    demands_ns: Sequence[tuple[int, int, int, int, int]],
) -> tuple[list[TargetOpNode], list[tuple[int, int, int, int, int]]]:
    """Fold full-depth native occurrences into structural cohort rows with exact unrounded means."""
    groups: OrderedDict[tuple[str, str], list[int]] = OrderedDict()
    for idx, node in enumerate(nodes):
        if node.archetype_id.startswith("target_arch_") or node.archetype_id == "draft_arch_0":
            key = (node.archetype_id, _strip_layer_site_key(node.op_id))
        else:
            key = (node.archetype_id, f"__exact_{idx}")
        groups.setdefault(key, []).append(idx)

    occ_remap: dict[OpOccurrence, OpOccurrence] = {}
    out_nodes: list[TargetOpNode] = []
    out_dems: list[tuple[int, int, int, int, int]] = []
    for idxs in groups.values():
        anchor = nodes[idxs[0]]
        cnt = len(idxs)
        for i in idxs:
            occ_remap[nodes[i].occurrence] = anchor.occurrence
        if cnt == 1:
            out_nodes.append(anchor)
            out_dems.append(demands_ns[idxs[0]])
            continue
        inv = 1.0 / float(cnt)
        mean_dur = sum(nodes[i].dur_us for i in idxs) * inv
        mean_exp = sum(nodes[i].exposed_dur_us for i in idxs) * inv
        mean_dem = tuple(int(round(sum(demands_ns[i][k] for i in idxs) * inv)) for k in range(DEMAND_CHANNELS))
        l_star = float(max(mean_dem)) / NS_PER_US
        eff = min(1.0, l_star / mean_dur) if (mean_dur > 0.0 and l_star > 0.0) else None
        out_dems.append(mean_dem)  # type: ignore[arg-type]
        out_nodes.append(replace(
            anchor,
            ordinal=len(out_nodes),
            repeat_count=cnt,
            end_us=anchor.start_us + mean_dur,
            dur_us=mean_dur,
            exposed_dur_us=mean_exp,
            is_overlapped=(mean_exp <= OVERLAP_BARRIER_EPS_US and mean_dur > OVERLAP_BARRIER_EPS_US),
            flops=sum(nodes[i].flops for i in idxs) * inv,
            vector_ops=sum(nodes[i].vector_ops for i in idxs) * inv,
            hbm_read_bytes=sum(nodes[i].hbm_read_bytes for i in idxs) * inv,
            hbm_write_bytes=sum(nodes[i].hbm_write_bytes for i in idxs) * inv,
            hbm_total_bytes=sum(nodes[i].hbm_total_bytes for i in idxs) * inv,
            mte2_bytes=sum(nodes[i].mte2_bytes for i in idxs) * inv,
            mte3_bytes=sum(nodes[i].mte3_bytes for i in idxs) * inv,
            fixpipe_bytes=sum(nodes[i].fixpipe_bytes for i in idxs) * inv,
            link_bytes=sum(nodes[i].link_bytes for i in idxs) * inv,
            l_star_lean_us=l_star,
            gap_us=max(0.0, mean_dur - l_star),
            gap_ratio=(mean_dur / l_star) if l_star > 0.0 else None,
            roofline_efficiency=eff,
        ))
    valid_occs = {n.occurrence for n in out_nodes}
    rewired = [
        replace(
            n,
            ordinal=i,
            predecessors=tuple(dict.fromkeys(
                mapped for p in n.predecessors
                if (mapped := occ_remap.get(p, p)) != n.occurrence and mapped in valid_occs
            )),
        )
        for i, n in enumerate(out_nodes)
    ]
    return rewired, out_dems


def _die_peaks_for_target_stage(case: CaseSpec, stage: int) -> dict[str, Any]:
    peaks = extract_hardware_peaks(case, stage=stage, replica=0)
    die = peaks["die"]
    dpath = datapath_for_die(die)
    hbm_peak = float(peaks["hbm_peak_bytes_per_s"])
    return {
        **peaks,
        "hbm_read_bytes_per_s": max(float(dpath.hbm_read_bytes_per_s), hbm_peak),
        "hbm_write_bytes_per_s": max(float(dpath.hbm_write_bytes_per_s), hbm_peak),
        "hbm_rw_alpha": float(dpath.hbm_rw_alpha),
    }


def _die_cube_peak_rate(die: Any, dtype: str, default_dtype: str = "bf16") -> float:
    for cand in (dtype, default_dtype, "bf16", "fp16"):
        if not cand:
            continue
        try:
            return float(die.peak(str(cand).strip().lower()))
        except ModelingError:
            continue
    return 0.0


def _demand_ns_floor(work: float, rate: float) -> int:
    if work <= 0.0 or rate <= 0.0:
        return 0
    return inward_floor_ns(Fraction(work) * NS_PER_SECOND / Fraction(rate))


def _target_op_work_and_demands(
    spec: Any,
    *,
    dur_us: float,
    shape: tuple[int, ...],
    dtype: str,
    family: str,
    peaks: Mapping[str, Any],
) -> tuple[tuple[int, int, int, int, int], float, float, float, float, float, float, float, float, float]:
    meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
    flops = float((spec.flops if spec is not None else 0.0) or 0.0)
    vec_ops = float((spec.vector_ops if spec is not None else 0.0) or 0.0)
    link_bytes = float((spec.link_bytes if spec is not None else 0.0) or 0.0)
    hbm_tot = float((spec.hbm_bytes if spec is not None else 0.0) or 0.0)
    mte2 = float((spec.mte2_bytes if spec is not None else 0.0) or 0.0)
    mte3 = float((spec.mte3_bytes if spec is not None else 0.0) or 0.0)
    fixpipe = float(meta.get("fixpipe_bytes", 0.0) or 0.0)
    raw_write = mte3 + fixpipe
    copy_total = mte2 + raw_write
    if copy_total > 0.0 and hbm_tot > 0.0:
        scale = hbm_tot / copy_total
        hbm_r, hbm_w = mte2 * scale, raw_write * scale
    elif "read_bytes" in meta:
        hbm_r = float(meta["read_bytes"] or 0.0)
        hbm_w = max(0.0, hbm_tot - hbm_r)
    elif copy_total > 0.0:
        hbm_r, hbm_w = mte2, raw_write
    else:
        hbm_r, hbm_w = hbm_tot, 0.0
    if flops == 0.0 and vec_ops == 0.0 and hbm_tot == 0.0 and link_bytes == 0.0:
        if dur_us > 0.0 and shape and family.lower() != "barrier":
            try:
                elem_b = float(dtype_bytes(str(dtype or "bf16").lower()))
            except ModelingError:
                elem_b = DEFAULT_ELEM_BYTES
            elems = 1
            for dim in shape:
                elems *= max(1, int(dim))
            hbm_r = float(elems) * elem_b
            hbm_w = hbm_r
            hbm_tot = hbm_r + hbm_w
            mte2, mte3 = hbm_r, hbm_w
    die = peaks["die"]
    default_dt = str(peaks.get("default_dtype") or "bf16")
    matrix_flops = (spec.matrix_flops if spec is not None else ()) or ()
    valid_pairs = [(str(dt), float(fl)) for dt, fl in matrix_flops if float(fl) > 0.0]
    if valid_pairs:
        cube_frac = Fraction(0)
        for dt, mf in valid_pairs:
            rate = _die_cube_peak_rate(die, dt, default_dt)
            if rate > 0.0:
                cube_frac += Fraction(mf) * NS_PER_SECOND / Fraction(rate)
        cube_ns = inward_floor_ns(cube_frac) if cube_frac > 0 else 0
    else:
        cube_ns = _demand_ns_floor(flops, _die_cube_peak_rate(die, dtype, default_dt))
    alpha = float(peaks.get("hbm_rw_alpha", 0.0) or 0.0)
    demands = (
        cube_ns,
        _demand_ns_floor(vec_ops, float(peaks.get("vector_peak_ops_per_s") or 0.0)),
        _demand_ns_floor(hbm_r + alpha * hbm_w, float(peaks.get("hbm_read_bytes_per_s") or 0.0)),
        _demand_ns_floor(hbm_w + alpha * hbm_r, float(peaks.get("hbm_write_bytes_per_s") or 0.0)),
        _demand_ns_floor(link_bytes, float(peaks.get("link_peak_bytes_per_s") or 0.0)),
    )
    return demands, flops, vec_ops, hbm_r, hbm_w, hbm_tot, mte2, mte3, fixpipe, link_bytes


def _prepare_target_records(
    eff_case: CaseSpec,
    timeline_rows: Sequence[Mapping[str, Any]],
    spec_by_id: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], frozenset[OpOccurrence]]:
    width = max(1, int(eff_case.runtime.parallel.tp) * int(eff_case.runtime.parallel.pcp))
    raw_recs: list[dict[str, Any]] = []
    occ_by_node: dict[str, OpOccurrence] = {}
    for row in timeline_rows:
        if not isinstance(row, Mapping) or int(row.get("replica", 0) or 0) != 0:
            continue
        oid = str(row.get("op_id") or "")
        spec = spec_by_id.get(oid)
        stage = int(spec.stage) if spec is not None else int(row.get("stage", 0) or 0)
        rank_idx = stage * width
        phys_rank = int(eff_case.machine.dies[rank_idx].rank) if 0 <= rank_idx < len(eff_case.machine.dies) else 0
        occ = OpOccurrence(replica=0, stage=stage, rank=phys_rank, op_id=oid)
        occ_by_node[f"r0:{oid}"] = occ
        if row.get("id"):
            occ_by_node[str(row["id"])] = occ
        start_us = max(0.0, float(row.get("start_ms", row.get("ready_ms", 0.0)) or 0.0) * US_PER_MS)
        end_ms = row.get("complete_ms", row.get("end_ms"))
        dur_ms = float(row.get("duration_ms", 0.0) or 0.0)
        end_us = max(start_us, float(end_ms) * US_PER_MS) if end_ms is not None else (start_us + max(0.0, dur_ms) * US_PER_MS)
        raw_recs.append({
            "occurrence": occ, "op_id": oid, "spec": spec, "row_layer": row.get("layer"),
            "family": str(spec.family if spec is not None else row.get("family", "")),
            "recipe": str(spec.recipe if spec is not None else row.get("recipe", "")),
            "stream": str((spec.stream if spec is not None else "") or row.get("stream", "") or "main"),
            "shape": tuple(spec.local_shape) if spec is not None else tuple(row.get("shape", ())),
            "dtype": str((spec.dtype if spec is not None else "") or row.get("dtype", "") or eff_case.model.compute_dtype or "bf16"),
            "raw_preds": tuple(row.get("predecessors") or ()),
            "start_us": start_us, "end_us": end_us,
        })
    for rec in raw_recs:
        rec["predecessors"] = tuple(occ_by_node[p_id] for p_id in rec.pop("raw_preds", ()) if p_id in occ_by_node)
    tail_closure = compute_acceptance_tail_closure(raw_recs)
    kept = resolve_barrier_predecessors(
        raw_recs, protected_occurrences=tail_closure, barrier_eps_us=OVERLAP_BARRIER_EPS_US,
    )
    return kept, tail_closure


_REWRITE_PLAN_ERROR_TOKENS = (
    "REWRITE_PLAN_",
    "rewrite_fusion_recipe_unbound",
    "rewrite_unresolved_op",
    "rewrite_plan_",
    "unbound_formula",
    "simulation_mode_forbids_unbound",
)


_PLAN_PHASE_MISMATCH_TOKENS = (
    "rewrite_digest_mismatch:",
    "rewrite_unresolved_op",
)


def _is_rewrite_plan_error(exc: Exception) -> bool:
    """Return True when ``exc`` signals an explicit rewrite-plan or strict-binding failure."""
    msg = str(exc)
    return any(tok in msg for tok in _REWRITE_PLAN_ERROR_TOKENS)


def _is_plan_phase_mismatch(exc: Exception) -> bool:
    """Return True when ``exc`` indicates a rewrite plan does not belong to the tested phase/graph."""
    msg = str(exc)
    return any(tok in msg for tok in _PLAN_PHASE_MISMATCH_TOKENS)


def _is_draft_op(spec: Any, row: Mapping[str, Any] | None = None) -> bool:
    """Classify speculative draft operators strictly from typed metadata (zero op-id heuristics)."""
    if spec is not None:
        meta = dict(getattr(spec, "metadata", ()) or ())
        for key in ("layer_scope", "producer_scope", "model_scope", "structure_model_scope"):
            if str(meta.get(key) or "").strip().lower() == "draft":
                return True
        if bool(int(meta.get("is_draft_model", 0) or 0)):
            return True
    if row is not None:
        for key in ("layer_scope", "producer_scope", "model_scope"):
            if str(row.get(key) or "").strip().lower() == "draft":
                return True
    return False


def _solve_target_step_ms(
    eff_case: CaseSpec,
    timeline_rows: Sequence[Mapping[str, Any]],
    spec_by_id: Mapping[str, Any],
    *,
    phase: str,
    cohort_plan: LayerCohortPlan | None = None,
    n_tok_per_die: float,
    ext_plan: Any,
    primitive_table: Any,
    allow_unbound_formula: bool,
    step_request: Any = None,
    scenario_case: CaseSpec | None = None,
) -> float:
    """Evaluate rewritten target graph on the same Engine + PrimitiveTable and return exposed ms."""
    kept, tail_closure = _prepare_target_records(eff_case, timeline_rows, spec_by_id)
    if not kept:
        return 0.0
    plan = cohort_plan if cohort_plan is not None else discover_layer_cohorts(eff_case)
    exposed_us_list, masked_list = _attribute_full_records_stream0(kept)
    peaks_by_stage: dict[int, dict[str, Any]] = {}
    stream_indices: dict[str, int] = {"main": 0}
    nodes: list[TargetOpNode] = []
    demands_ns: list[tuple[int, int, int, int, int]] = []
    for idx, rec in enumerate(kept):
        occ: OpOccurrence = rec["occurrence"]
        spec = rec["spec"]
        if occ.stage not in peaks_by_stage:
            peaks_by_stage[occ.stage] = _die_peaks_for_target_stage(eff_case, occ.stage)
        peaks = peaks_by_stage[occ.stage]
        oid = rec["op_id"]
        is_draft = _is_draft_op(spec, rec)
        lyr = spec.layer if (spec is not None and spec.layer is not None) else rec["row_layer"]
        if occ in tail_closure or oid.startswith("final."):
            arch_id = "target_tail"
        elif is_draft:
            arch_id = "draft_arch_0" if (lyr is not None and int(lyr) >= 0) else "draft_misc"
        elif lyr is not None and int(lyr) >= 0:
            c_idx = plan.cohort_by_layer.get(int(lyr), int(lyr))
            arch_id = f"target_arch_{c_idx}"
        else:
            arch_id = "target_embed" if str(rec["family"]).lower() == "embedding" else "target_tail"
        dur_us = max(0.0, float(rec["end_us"]) - float(rec["start_us"]))
        exp_us = min(dur_us, max(0.0, float(exposed_us_list[idx])))
        dem, flops, vec_ops, hbm_r, hbm_w, hbm_tot, mte2, mte3, fixpipe, link_b = _target_op_work_and_demands(
            spec, dur_us=dur_us, shape=rec["shape"], dtype=rec["dtype"], family=rec["family"], peaks=peaks,
        )
        l_star_us = float(max(dem)) / NS_PER_US
        dom = (
            "link_bound" if (link_b > 0.0 and float(dem[4]) / NS_PER_US >= l_star_us)
            else "launch_bound" if (l_star_us <= 0.0 or (dur_us <= LAUNCH_SHORT_US and l_star_us < dur_us * LAUNCH_WORK_SHARE))
            else "cube_bound" if float(dem[0]) / NS_PER_US >= l_star_us
            else "vector_bound" if float(dem[1]) / NS_PER_US >= l_star_us
            else "memory_bound"
        )
        fam_l = rec["family"].lower()
        task_type = (
            "HOST" if fam_l in {"engram_rh2d", "h2d", "offload"}
            else "HCCL" if link_b > 0.0
            else "AI_CORE" if flops > 0.0
            else "AI_VECTOR_CORE"
        )
        s_idx = stream_indices.setdefault(rec["stream"], len(stream_indices))
        eff = min(1.0, l_star_us / dur_us) if (dur_us > 0.0 and l_star_us > 0.0) else None
        demands_ns.append(dem)
        nodes.append(TargetOpNode(
            ordinal=len(nodes), occurrence=occ, op_id=oid, archetype_id=arch_id,
            repeat_count=1, family=rec["family"], recipe=rec["recipe"],
            stream=rec["stream"], stream_index=s_idx, task_type=task_type,
            shape=rec["shape"], dtype=rec["dtype"], predecessors=rec["predecessors"],
            start_us=float(rec["start_us"]), end_us=float(rec["end_us"]),
            dur_us=dur_us, exposed_dur_us=exp_us,
            is_overlapped=(exp_us <= OVERLAP_BARRIER_EPS_US and dur_us > OVERLAP_BARRIER_EPS_US),
            masked_by=masked_list[idx],
            flops=flops, vector_ops=vec_ops,
            hbm_read_bytes=hbm_r, hbm_write_bytes=hbm_w, hbm_total_bytes=hbm_tot,
            mte2_bytes=mte2, mte3_bytes=mte3, fixpipe_bytes=fixpipe, link_bytes=link_b,
            dominant_bound=dom, bound_display=dom, l_star_lean_us=l_star_us,
            gap_us=max(0.0, dur_us - l_star_us),
            gap_ratio=(dur_us / l_star_us) if l_star_us > 0.0 else None,
            roofline_efficiency=eff, absorbed_source_ops=(), semantic_atoms=(),
            logical_role=rec["family"], ascend_kernel=rec["recipe"] or rec["family"],
            kernel_backend="", kernel_status="", needs_new_kernel=False,
            new_kernel_spec="", canonical_op_key=oid, spec=spec,
        ))
    has_explicit_plan = ext_plan is not None and (
        bool(getattr(ext_plan, "edits", ()))
        or bool(getattr(ext_plan, "base_graph_sha256", ""))
        or bool(getattr(ext_plan, "framing_sha256", ""))
    )
    if not has_explicit_plan:
        nodes, demands_ns = _aggregate_cohort_target_nodes(nodes, demands_ns)
    solved = solve_target_graph(
        nodes, demands_ns, eff_case, peaks_by_stage,
        phase=phase, rewrite_plan=ext_plan,
        n_tok_per_die=max(float(n_tok_per_die), PARETO_EPS),
        primitive_table=primitive_table,
        allow_unbound_formula=allow_unbound_formula,
        step_request=step_request,
        scenario_case=scenario_case,
        compute_waterfall_and_codesign=False,
    )
    return sum(n.exposed_dur_us * max(1, int(n.repeat_count)) for n in solved.nodes) / US_PER_MS


def _archetype_weighted_step_ms(
    parsed: CaseSpec,
    *,
    phase: str,
    tokens: int,
    context: int,
    sequences: int,
    scope: str = "model",
    min_layers: int = ARCHETYPE_FOLD_MIN_LAYERS,
    rep_info: LayerCohortPlan | None = None,
    evaluation_context: EvaluationContext | None = None,
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
    n_tok_per_die: float = 1.0,
    eval_ctx_box: list[EvaluationContext | None] | None = None,
    skip_target_if_baseline_le_ms: float | None = None,
    reclaim_on_exit: bool = True,
) -> float:
    """Exact native ``0..L-1 + final`` Engine replay with unified ``PrimitiveTable`` binding."""
    del min_layers
    cohort_plan = rep_info if isinstance(rep_info, LayerCohortPlan) else discover_layer_cohorts(parsed)
    use_target = bool(apply_target_envelope or rewrite_plan is not None)
    ext_plan = load_rewrite_plan(rewrite_plan) if (use_target and rewrite_plan is not None) else None
    graphs, eff_case, _ = _lower_and_run_graph(
        parsed, phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope, run_graph=False,
    )
    need_timeline = bool(use_target)
    if evaluation_context is not None:
        eval_ctx = evaluation_context
        owns_ctx = False
    elif eval_ctx_box is not None:
        if eval_ctx_box[0] is None:
            eval_ctx_box[0] = EvaluationContext.create(eff_case)
        eval_ctx = eval_ctx_box[0]
        owns_ctx = False
    else:
        eval_ctx = EvaluationContext.create(eff_case)
        owns_ctx = True
    ext_in = {"analysis_input": 0} if scope == "body" else None
    has_kg = any("device_kernel_group" in dict(op.metadata) for op in graphs[0])
    sg = (
        TargetGraphSet({0: graphs[0]}, kernels=graphs.effective_kernels, selection=graphs.default_selection)
        if isinstance(graphs, TargetGraphSet) else {0: graphs[0]}
    ) if rank_zero_graph_reuse(eff_case.runtime.parallel, len(graphs), has_kg) else graphs
    spec_by_id = {op.op_id: op for op in graphs[0]} if need_timeline else {}
    # Unify Before and After on the EXACT SAME PrimitiveTable (zero conditional None split).
    rebound_pt = eval_ctx.bindings.primitive_table
    try:
        graph, _, _ = execute_native_cohort_graph(
            eff_case, sg, cohort_plan=cohort_plan,
            external_inputs=ext_in, isolated=False,
            evaluation=eval_ctx, fast_span_only=(not need_timeline),
            timeline_replicas=(0,) if need_timeline else None,
        )
        baseline_ms = float(graph["duration_ms"])
        timeline_rows = tuple(graph.get("timeline", ()) or ()) if need_timeline else ()
    finally:
        del graphs, sg
        if owns_ctx:
            del eval_ctx
            if reclaim_on_exit and (len(parsed.machine.dies) >= GC_RECLAIM_MIN_WORLD_SIZE or int(parsed.runtime.parallel.dp) > 1):
                _reclaim_sweep_memory()
    if not use_target:
        return baseline_ms
    if ext_plan is None and skip_target_if_baseline_le_ms is not None and baseline_ms <= skip_target_if_baseline_le_ms:
        return baseline_ms
    from mindie_sim.prediction.results import StepRequest

    return _solve_target_step_ms(
        eff_case, timeline_rows, spec_by_id,
        phase=phase, cohort_plan=cohort_plan,
        n_tok_per_die=n_tok_per_die, ext_plan=ext_plan,
        primitive_table=rebound_pt, allow_unbound_formula=allow_unbound_formula,
        step_request=StepRequest(phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope),
        scenario_case=parsed,
    )


def _free_hbm_gib_for_batch(memory: Memory, kv_by_stage: Sequence[float], batch: int) -> float:
    remaining = min(
        (float(cap) - float(batch) * float(kv)
         for row in memory.capacity for cap, kv in zip(row, kv_by_stage)),
        default=0.0,
    )
    return max(0.0, remaining) / BYTES_PER_GIB


def _fits_vram_batch(
    parsed: CaseSpec,
    *,
    cand_b: int,
    base_ws: int,
    decode_q: int,
    kv_by_stage: Sequence[float],
) -> tuple[bool, Memory]:
    ws = max(int(base_ws), max(1, int(cand_b)) * max(1, int(decode_q)))
    mem = Memory(parsed, workspace_tokens=ws)
    if cand_b <= 0:
        return all(float(cap) >= 0.0 for row in mem.capacity for cap in row), mem
    fits = all(
        float(cap) >= 0.0 and float(cap) >= float(cand_b) * float(kv)
        for row in mem.capacity
        for cap, kv in zip(row, kv_by_stage)
    )
    return fits, mem


def _solve_b_vram_max(
    parsed: CaseSpec,
    *,
    base_ws: int,
    decode_q: int,
    seq_len: int,
    cap_bound: int,
) -> tuple[Memory, tuple[float, ...], int]:
    """Binary-search maximum VRAM concurrency evaluating ``workspace_tokens(b)`` per candidate."""
    base_mem = Memory(parsed, workspace_tokens=max(1, int(base_ws)))
    d_kv = tuple(float(v) for v in base_mem.dp_cohort_cache_bytes(seq_len, padded=True))
    if not all(float(cap) >= 0.0 for row in base_mem.capacity for cap in row):
        return base_mem, d_kv, 0
    upper = max(0, int(cap_bound))
    for row in base_mem.capacity:
        for cap, kv in zip(row, d_kv):
            if kv > 0.0:
                upper = min(upper, int(math.floor(max(0.0, float(cap)) / kv)))
    if upper <= 0:
        return base_mem, d_kv, 0
    if int(upper * max(1, int(decode_q))) <= int(base_ws):
        return base_mem, d_kv, upper
    hi_fits, hi_mem = _fits_vram_batch(
        parsed, cand_b=upper, base_ws=base_ws, decode_q=decode_q, kv_by_stage=d_kv,
    )
    if hi_fits:
        return hi_mem, d_kv, upper
    lo, hi = 0, upper
    best_mem = base_mem
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        ok, mid_mem = _fits_vram_batch(
            parsed, cand_b=mid, base_ws=base_ws, decode_q=decode_q, kv_by_stage=d_kv,
        )
        if ok:
            lo, best_mem = mid, mid_mem
        else:
            hi = mid
    return best_mem, d_kv, lo


def _headroom_gib_at_batch(
    parsed: CaseSpec,
    d_mem: Memory,
    d_kv: Sequence[float],
    *,
    b_vram: int,
    batch: int,
    base_ws: int,
    decode_q: int,
) -> float:
    if batch == b_vram or batch <= 0:
        return _free_hbm_gib_for_batch(d_mem, d_kv, batch)
    ws = max(int(base_ws), max(1, int(batch)) * max(1, int(decode_q)))
    return _free_hbm_gib_for_batch(Memory(parsed, workspace_tokens=ws), d_kv, batch)


def _prefill_step_ms(
    parsed: CaseSpec,
    *,
    compute_tokens: int,
    hit_tokens: int = 0,
    min_layers: int = ARCHETYPE_FOLD_MIN_LAYERS,
    rep_info: tuple[CaseSpec, dict[int, int], float] | None = None,
    evaluation_context: EvaluationContext | None = None,
    eval_ctx_box: list[EvaluationContext | None] | None = None,
    reclaim_on_exit: bool = True,
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
) -> float:
    """Compute prefill latency via direct step or chunked-prefill midpoint quadrature."""
    if compute_tokens <= 0:
        return 0.0
    if not parsed.runtime.enable_chunked_prefill or compute_tokens <= PREFILL_DIRECT_MAX_TOKENS:
        return _archetype_weighted_step_ms(
            parsed, phase="prefill", tokens=compute_tokens,
            context=hit_tokens, sequences=1, min_layers=min_layers,
            rep_info=rep_info, evaluation_context=evaluation_context,
            eval_ctx_box=eval_ctx_box, reclaim_on_exit=reclaim_on_exit,
            apply_target_envelope=apply_target_envelope,
            rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
        )
    batched_cap = int(parsed.runtime.max_num_batched_tokens or PREFILL_CHUNK_PROBE_TOKENS)
    chunk_tok = max(1, min(int(compute_tokens), batched_cap, PREFILL_CHUNK_PROBE_TOKENS))
    mid_ctx = int(hit_tokens) + max(0, (int(compute_tokens) - chunk_tok) // 2)
    chunk_ms = _archetype_weighted_step_ms(
        parsed, phase="prefill", tokens=chunk_tok,
        context=mid_ctx, sequences=1, min_layers=min_layers,
        rep_info=rep_info, evaluation_context=evaluation_context,
        eval_ctx_box=eval_ctx_box, reclaim_on_exit=reclaim_on_exit,
        apply_target_envelope=apply_target_envelope,
        rewrite_plan=rewrite_plan,
        allow_unbound_formula=allow_unbound_formula,
    )
    return chunk_ms * (float(compute_tokens) / float(chunk_tok))


def _decode_batch_limit(parsed: CaseSpec, q_tok: int, b_vram: int) -> int:
    """Largest uniform decode batch the scheduler can issue in one step.

    ``b_vram`` is the physical HBM ceiling (kept for headroom reporting); every
    sequence also consumes ``q_tok`` query tokens from ``max_num_batched_tokens``.
    """
    budget = int(parsed.runtime.max_num_batched_tokens or 0)
    if budget <= 0:
        return int(b_vram)
    return min(int(b_vram), budget // max(1, int(q_tok)))


def _tighten_sla_bracket(
    memo: Mapping[int, float], b_vram: int, sla_ms: float,
) -> tuple[int, int, int, float | None]:
    """Tighten binary-search ``(lo, hi, best_b, best_t)`` from exact cached batch probes."""
    lo, hi = 0, b_vram + 1
    best_b = 0
    best_t: float | None = None
    for k, t_val in sorted(memo.items()):
        if k < 1 or k > b_vram:
            continue
        if t_val <= sla_ms + PARETO_EPS:
            if k > lo:
                lo, best_b, best_t = k, k, t_val
        elif k < hi:
            hi = k
    return lo, hi, best_b, best_t


def _solve_sla_archetype_aware(
    parsed: CaseSpec,
    *,
    sla_ms: float,
    context: int,
    output_budget: int,
    max_sequences_per_dp: int | None,
    workspace_tokens: int | None = None,
    rep_info: tuple[CaseSpec, dict[int, int], float] | None = None,
    evaluation_context: EvaluationContext | None = None,
    probe_cache: Mapping[int, float] | None = None,
    probe_sink: dict[int, float] | None = None,
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
    vram_info: tuple[Memory, tuple[float, ...], int] | None = None,
    eval_ctx_box: list[EvaluationContext | None] | None = None,
) -> dict[str, Any]:
    """Exact SLA concurrency solver using O(Archetypes x 1 Rank) steps."""
    q_tok, emit_est = _resolve_speculative_tokens(
        parsed, output_budget=output_budget, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    base_ws = int(workspace_tokens) if workspace_tokens is not None else max(1, int(q_tok))
    cap_bound = int(parsed.runtime.max_num_seqs)
    if max_sequences_per_dp is not None:
        cap_bound = min(cap_bound, int(max_sequences_per_dp))
    if vram_info is not None:
        d_mem, d_kv, b_vram = vram_info
    else:
        d_mem, d_kv, b_vram = _solve_b_vram_max(
            parsed, base_ws=base_ws, decode_q=max(1, int(q_tok)),
            seq_len=context + output_budget, cap_bound=cap_bound,
        )
    rep_info = rep_info or discover_layer_cohorts(parsed)
    memo: dict[int, float] = {}
    exact_keys: set[int] = set()
    for raw_k, raw_v in (probe_cache or {}).items():
        k_int = int(raw_k)
        v_flt = float(raw_v)
        if k_int >= 1:
            memo[k_int] = v_flt
            exact_keys.add(k_int)
        elif k_int <= -1 and rewrite_plan is None and v_flt <= sla_ms + PARETO_EPS:
            # Negative key encodes a non-exact baseline upper-bound hint (only valid without explicit rewrite_plan).
            memo[-k_int] = v_flt
    par = parsed.runtime.parallel
    per_dp_dies = max(1, int(par.tp) * int(par.pcp) * int(par.pp))
    owns_box = eval_ctx_box is None
    ctx_box: list[EvaluationContext | None] = [evaluation_context] if owns_box else eval_ctx_box
    use_target = bool(apply_target_envelope or rewrite_plan is not None)

    def eval_b(seq_count: int, *, require_exact: bool = True) -> float:
        key = max(1, int(seq_count))
        if key in memo and (not require_exact or key in exact_keys):
            return memo[key]
        skip_thresh = (
            (sla_ms + PARETO_EPS)
            if (use_target and rewrite_plan is None and not require_exact)
            else None
        )
        t_val = _archetype_weighted_step_ms(
            parsed, phase="decode", tokens=max(1, int(q_tok)),
            context=context, sequences=key,
            rep_info=rep_info, evaluation_context=evaluation_context,
            apply_target_envelope=apply_target_envelope,
            rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
            n_tok_per_die=(float(key) * float(emit_est)) / float(per_dp_dies),
            eval_ctx_box=ctx_box,
            skip_target_if_baseline_le_ms=skip_thresh,
        )
        memo[key] = t_val
        is_exact = (not use_target) or require_exact or (t_val > sla_ms + PARETO_EPS)
        if is_exact:
            exact_keys.add(key)
            if probe_sink is not None:
                probe_sink[key] = t_val
        return t_val

    def _result(b_sla: int, b_feas: int, t1_val: float, t_feas: float | None, headroom_b: int) -> dict[str, Any]:
        return {
            "emitted_tokens_per_seq": float(emit_est),
            "scenarios": [{
                "b_vram_max": b_vram, "b_sla": b_sla, "b_feasible": b_feas,
                "step_ms_at_b1": t1_val, "step_ms_feasible": t_feas,
                "free_hbm_headroom_gib": _headroom_gib_at_batch(
                    parsed, d_mem, d_kv, b_vram=b_vram, batch=headroom_b,
                    base_ws=base_ws, decode_q=max(1, int(q_tok)),
                ),
            }],
        }

    try:
        limit = _decode_batch_limit(parsed, q_tok, b_vram)
        if b_vram < 1:
            t1 = eval_b(1, require_exact=True)
            return _result(0 if t1 > sla_ms + PARETO_EPS else 1, 0, t1, None, 0)
        if limit < 1:
            return _result(0, 0, eval_b(1, require_exact=True), None, 0)
        lo, hi, best_b, best_t = _tighten_sla_bracket(memo, limit, sla_ms)
        if hi > limit and lo < limit:
            t_top = eval_b(limit, require_exact=True)
            if t_top <= sla_ms + PARETO_EPS:
                return _result(limit, limit, eval_b(1, require_exact=True), t_top, limit)
            hi = limit
        if lo == 0:
            t1 = eval_b(1, require_exact=False)
            if t1 > sla_ms + PARETO_EPS:
                return _result(0, 0, t1, None, 0)
            lo, best_b, best_t = 1, 1, t1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            t_mid = eval_b(mid, require_exact=False)
            if t_mid <= sla_ms + PARETO_EPS:
                lo, best_b, best_t = mid, mid, t_mid
            else:
                hi = mid
        exact_best_t = eval_b(best_b, require_exact=True)
        return _result(best_b, best_b, eval_b(1, require_exact=True), exact_best_t, best_b)
    finally:
        if owns_box and evaluation_context is None and ctx_box[0] is not None:
            ctx_box[0] = None
            if len(parsed.machine.dies) >= GC_RECLAIM_MIN_WORLD_SIZE or int(par.dp) > 1:
                _reclaim_sweep_memory()


def _screen_colocated_job(args: tuple[Any, ...]) -> tuple[int, bool, float, dict[int, float]]:
    (
        idx, raw_case, prompt_tokens, output_tokens, ttft_sla_ms, tpot_sla_ms,
        apply_target_envelope, rewrite_plan, allow_unbound_formula,
    ) = args
    try:
        parsed = parse_case(raw_case)
        par = parsed.runtime.parallel
        if not validate_vllm_ascend_dcp_codomain(parsed.model, par)[0]:
            return idx, False, -1.0, {}
        prefill_ws = _colocated_workspace_tokens(parsed, prompt_tokens)
        p_mem = Memory(parsed, workspace_tokens=prefill_ws)
        p_kv = p_mem.cache_bytes(prompt_tokens + 1, padded=True)
        prefill_ok = bool(
            prompt_tokens <= parsed.runtime.max_model_len
            and (parsed.runtime.enable_chunked_prefill or prompt_tokens <= parsed.runtime.max_num_batched_tokens)
            and all(cap >= 0.0 and cap >= kv for row in p_mem.capacity for cap, kv in zip(row, p_kv))
        )
        if not prefill_ok:
            return idx, False, -1.0, {}
        q_tok, emit_est = _resolve_speculative_tokens(
            parsed, output_budget=output_tokens, tokens_per_seq=None, emitted_tokens_per_seq=None,
        )
        _, _, b_vram = _solve_b_vram_max(
            parsed, base_ws=prefill_ws, decode_q=max(1, int(q_tok)),
            seq_len=prompt_tokens + output_tokens, cap_bound=int(parsed.runtime.max_num_seqs),
        )
        limit = _decode_batch_limit(parsed, q_tok, b_vram)
        if limit < 1:
            return idx, False, -1.0, {}
        rep_info = discover_layer_cohorts(parsed)
        prefill_plan_matched = False
        if math.isinf(ttft_sla_ms) and output_tokens > 1:
            p_ms = 0.0
        else:
            p_ms, prefill_plan_matched = _eval_prefill_with_plan(
                parsed, compute_tokens=prompt_tokens, hit_tokens=0,
                min_layers=ARCHETYPE_SCREEN_MIN_LAYERS,
                rep_info=rep_info, reclaim_on_exit=False,
                apply_target_envelope=apply_target_envelope,
                rewrite_plan=rewrite_plan,
                allow_unbound_formula=allow_unbound_formula,
                allow_plan_fallback=(output_tokens > 1),
            )
            if p_ms > ttft_sla_ms * SCREEN_TTFT_MARGIN:
                return idx, False, -p_ms, {}
        probe_b = min(limit, SCREEN_DECODE_BATCH_PROBE)
        per_dp_dies = max(1, int(par.tp) * int(par.pcp) * int(par.pp))
        screen_target = bool(apply_target_envelope or rewrite_plan is not None)
        try:
            d_ms = _archetype_weighted_step_ms(
                parsed, phase="decode", tokens=max(1, int(q_tok)),
                context=prompt_tokens, sequences=probe_b,
                rep_info=rep_info,
                apply_target_envelope=screen_target,
                rewrite_plan=rewrite_plan if screen_target else None,
                allow_unbound_formula=allow_unbound_formula,
                n_tok_per_die=(float(probe_b) * float(emit_est)) / float(per_dp_dies),
                reclaim_on_exit=False,
            )
        except (ModelingError, ValueError) as exc:
            if not (prefill_plan_matched and _is_plan_phase_mismatch(exc)):
                raise
            d_ms = _archetype_weighted_step_ms(
                parsed, phase="decode", tokens=max(1, int(q_tok)),
                context=prompt_tokens, sequences=probe_b,
                rep_info=rep_info,
                apply_target_envelope=apply_target_envelope,
                rewrite_plan=None,
                allow_unbound_formula=allow_unbound_formula,
                n_tok_per_die=(float(probe_b) * float(emit_est)) / float(per_dp_dies),
                reclaim_on_exit=False,
            )
        est_tpot = d_ms / max(float(emit_est), PARETO_EPS)
        eff_b = (
            float(probe_b)
            if est_tpot <= tpot_sla_ms * SCREEN_TPOT_MARGIN
            else max(1.0, float(probe_b) * (tpot_sla_ms / max(est_tpot, PARETO_EPS)))
        )
        score = (eff_b * float(par.dp) * float(emit_est) * MS_PER_S) / max(d_ms * float(par.world_size), PARETO_EPS)
        return idx, True, score, {int(probe_b): float(d_ms)}
    except (ModelingError, ValueError) as exc:
        if rewrite_plan is not None and _is_rewrite_plan_error(exc):
            raise
        return idx, False, -1.0, {}


def _eval_colocated_job(args: tuple[Any, ...]) -> tuple[int, dict[str, Any] | None]:
    (
        idx, raw_case, prompt, output, ttft, tpot, mtp_n, max_seq, probe_cache,
        apply_target_envelope, rewrite_plan, allow_unbound_formula,
    ) = args
    try:
        row = _evaluate_colocated_candidate(
            raw_case, prompt_tokens=prompt, output_tokens=output,
            ttft_sla_ms=ttft, tpot_sla_ms=tpot, mtp_tokens=mtp_n,
            max_sequences_per_dp=max_seq, probe_cache=probe_cache,
            apply_target_envelope=apply_target_envelope,
            rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
        )
        return idx, row
    except (ModelingError, ValueError) as exc:
        if rewrite_plan is not None and _is_rewrite_plan_error(exc):
            raise
        return idx, None


def _screen_prefill_job(
    args: tuple[Any, ...],
) -> tuple[int, bool, float, tuple[Memory, bool, int, float] | None]:
    (
        p_idx, raw_p, prompt, compute_tokens, hit_tokens, ttft_sla_ms,
        apply_target_envelope, rewrite_plan, allow_unbound_formula, allow_plan_fallback,
    ) = args
    try:
        parsed = parse_case(raw_p)
        if not validate_vllm_ascend_dcp_codomain(parsed.model, parsed.runtime.parallel)[0]:
            return p_idx, False, -1.0, None
        p_mem = Memory(parsed, workspace_tokens=_prefill_workspace_tokens(parsed, compute_tokens))
        p_kv = p_mem.cache_bytes(prompt + 1, padded=True)
        hbm_ok = bool(
            prompt <= parsed.runtime.max_model_len
            and (parsed.runtime.enable_chunked_prefill or compute_tokens <= parsed.runtime.max_num_batched_tokens)
            and all(cap >= 0.0 and cap >= kv for row in p_mem.capacity for cap, kv in zip(row, p_kv))
        )
        if not hbm_ok:
            return p_idx, False, -1.0, (p_mem, False, 0, math.inf if compute_tokens > 0 else 0.0)
        p_slots = min(
            [int(parsed.runtime.max_num_seqs)]
            + [int(math.floor(max(0.0, float(cap)) / float(kv)))
               for row in p_mem.capacity for cap, kv in zip(row, p_kv) if kv > 0]
        )
        p_ms = (
            _eval_prefill_with_plan(
                parsed, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
                min_layers=ARCHETYPE_SCREEN_MIN_LAYERS, reclaim_on_exit=False,
                apply_target_envelope=apply_target_envelope,
                rewrite_plan=rewrite_plan,
                allow_unbound_formula=allow_unbound_formula,
                allow_plan_fallback=allow_plan_fallback,
            )[0]
            if compute_tokens > 0 else 0.0
        )
        cached = (p_mem, True, p_slots, p_ms)
        par = parsed.runtime.parallel
        if p_ms > ttft_sla_ms * SCREEN_TTFT_MARGIN:
            return p_idx, False, -p_ms, cached
        rps_per_die = (float(par.dp) * MS_PER_S) / max(p_ms * float(par.world_size), PARETO_EPS)
        return p_idx, True, rps_per_die, cached
    except (ModelingError, ValueError) as exc:
        if rewrite_plan is not None and _is_rewrite_plan_error(exc):
            raise
        return p_idx, False, -1.0, None


def _screen_decode_job(args: tuple[Any, ...]) -> tuple[int, bool, float, dict[int, float]]:
    (
        d_idx, raw_d, prompt, output, tpot_sla_ms,
        apply_target_envelope, rewrite_plan, allow_unbound_formula,
    ) = args
    try:
        parsed = parse_case(raw_d)
        par = parsed.runtime.parallel
        if not validate_vllm_ascend_dcp_codomain(parsed.model, par)[0]:
            return d_idx, False, -1.0, {}
        if prompt + output > parsed.runtime.max_model_len:
            return d_idx, False, -1.0, {}
        q_tok, emit_est = _resolve_speculative_tokens(
            parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
        )
        _, _, b_vram = _solve_b_vram_max(
            parsed, base_ws=max(1, int(q_tok)), decode_q=max(1, int(q_tok)),
            seq_len=prompt + output, cap_bound=int(parsed.runtime.max_num_seqs),
        )
        limit = _decode_batch_limit(parsed, q_tok, b_vram)
        if limit < 1:
            return d_idx, False, -1.0, {}
        probe_b = min(limit, SCREEN_DECODE_BATCH_PROBE)
        per_dp_dies = max(1, int(par.tp) * int(par.pcp) * int(par.pp))
        screen_target = bool(apply_target_envelope or rewrite_plan is not None)
        d_ms = _archetype_weighted_step_ms(
            parsed, phase="decode", tokens=max(1, int(q_tok)), context=prompt, sequences=probe_b,
            apply_target_envelope=screen_target,
            rewrite_plan=rewrite_plan if screen_target else None,
            allow_unbound_formula=allow_unbound_formula,
            n_tok_per_die=(float(probe_b) * float(emit_est)) / float(per_dp_dies),
            reclaim_on_exit=False,
        )
        est_tpot = d_ms / max(float(emit_est), PARETO_EPS)
        scale = 1.0 if est_tpot <= tpot_sla_ms else (tpot_sla_ms / max(est_tpot, PARETO_EPS))
        tps_per_die = (float(probe_b) * scale * float(par.dp) * float(emit_est) * MS_PER_S) / max(d_ms * float(par.world_size), PARETO_EPS)
        return d_idx, True, tps_per_die, {int(probe_b): float(d_ms)}
    except (ModelingError, ValueError) as exc:
        if rewrite_plan is not None and _is_rewrite_plan_error(exc):
            raise
        return d_idx, False, -1.0, {}


def _eval_prefill_exact_job(args: tuple[Any, ...]) -> tuple[int, tuple[Memory, bool, int, float] | None]:
    (
        p_idx, raw_p, prompt, compute_tokens, hit_tokens,
        apply_target_envelope, rewrite_plan, allow_unbound_formula, allow_plan_fallback,
    ) = args
    try:
        parsed = parse_case(raw_p)
        res = _measure_prefill_stage(
            parsed, prompt=prompt, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
            cache=None, cache_id=None,
            apply_target_envelope=apply_target_envelope,
            rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
            allow_plan_fallback=allow_plan_fallback,
        )
        return p_idx, res
    except (ModelingError, ValueError) as exc:
        if rewrite_plan is not None and _is_rewrite_plan_error(exc):
            raise
        return p_idx, None


def _eval_decode_exact_job(
    args: tuple[Any, ...],
) -> tuple[int, float, dict[str, Any] | None, tuple[Memory, tuple[float, ...], int] | None, tuple[CaseSpec, dict[int, int], float] | None]:
    (
        d_idx, raw_d, prompt, output, tpot_sla_ms, max_seq,
        probe_cache, apply_target_envelope, rewrite_plan, allow_unbound_formula,
    ) = args
    try:
        parsed = parse_case(raw_d)
        if not validate_vllm_ascend_dcp_codomain(parsed.model, parsed.runtime.parallel)[0]:
            return d_idx, 0.0, None, None, None
        q_tok, d_emit_est = _resolve_speculative_tokens(
            parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
        )
        cap_bound = int(parsed.runtime.max_num_seqs)
        if max_seq is not None:
            cap_bound = min(cap_bound, int(max_seq))
        vram_info = _solve_b_vram_max(
            parsed, base_ws=max(1, int(q_tok)), decode_q=max(1, int(q_tok)),
            seq_len=prompt + output, cap_bound=cap_bound,
        )
        rep_info = discover_layer_cohorts(parsed)
        eff_sla_ms = tpot_sla_ms * max(1.0, float(d_emit_est))
        d_sla = _solve_sla_archetype_aware(
            parsed, sla_ms=eff_sla_ms, context=prompt,
            output_budget=output, max_sequences_per_dp=max_seq,
            probe_cache=probe_cache, probe_sink=probe_cache,
            apply_target_envelope=apply_target_envelope,
            rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
            vram_info=vram_info, rep_info=rep_info,
        )
        return d_idx, eff_sla_ms, d_sla, vram_info, rep_info
    except (ModelingError, ValueError) as exc:
        if rewrite_plan is not None and _is_rewrite_plan_error(exc):
            raise
        return d_idx, 0.0, None, None, None


def _search_colocated_deployments(
    compose_for: Any,
    *,
    num_devices: int,
    is_moe: bool,
    layers: int,
    model_spec: Any,
    rec_topo: Mapping[str, Any] | None,
    pinned: Mapping[str, int],
    mtp_grid: tuple[int, ...],
    prompt: int,
    output: int,
    ttft: float,
    tpot: float,
    max_sequences_per_dp: int | None,
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]]]:
    coloc_topos = _build_colocated_candidates(
        num_devices=num_devices, is_moe=is_moe, layers=layers,
        rec_topo=rec_topo, pinned=pinned,
        model_spec=model_spec, prompt_tokens=prompt,
    )
    composed: list[tuple[int, int, dict[str, Any]]] = []
    for mtp_n in mtp_grid:
        for topo in coloc_topos:
            raw_c = compose_for(topo, mtp_n, "colocated")
            if raw_c is not None:
                composed.append((len(composed), mtp_n, raw_c))
    if not composed:
        return [], []

    selected = composed
    probe_caches: dict[int, dict[int, float]] = {}
    if layers >= ARCHETYPE_SCREEN_MIN_LAYERS and len(composed) > COLOCATED_STAGE2_FINALISTS:
        screened = []
        for s_i, (idx, _, raw_c) in enumerate(composed):
            screened.append(_screen_colocated_job((
                idx, raw_c, prompt, output, ttft, tpot,
                apply_target_envelope, rewrite_plan, allow_unbound_formula,
            )))
            if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE and (s_i + 1) % SCREEN_RECLAIM_STRIDE == 0:
                _reclaim_sweep_memory()
        if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE:
            _reclaim_sweep_memory()
        probe_caches = {idx: pc for idx, _, _, pc in screened if pc}
        feasible_idx = sorted(
            ((score, idx) for idx, ok, score, _ in screened if ok), reverse=True,
        )
        if feasible_idx:
            keep = {idx for _, idx in feasible_idx[:COLOCATED_STAGE2_FINALISTS]}
            selected = [item for item in composed if item[0] in keep]
        else:
            keep = {idx for _, idx in sorted(((sc, idx) for idx, _, sc, _ in screened), reverse=True)[:1]}
            selected = [item for item in composed if item[0] in keep]

    by_idx = {idx: raw_c for idx, _, raw_c in selected}
    rows: list[dict[str, Any]] = []
    feasible_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for idx, mtp_n, raw_c in selected:
        _, row = _eval_colocated_job((
            idx, raw_c, prompt, output, ttft, tpot, mtp_n, max_sequences_per_dp,
            probe_caches.get(idx), apply_target_envelope, rewrite_plan, allow_unbound_formula,
        ))
        if row is None:
            continue
        rows.append(row)
        if row["feasible"]:
            feasible_entries.append((row, by_idx[idx]))
    return rows, feasible_entries


def _select_and_warm_pd_finalists(
    p_cases: list[tuple[int, int, dict[str, Any]]],
    d_by_mtp: list[tuple[int, list[tuple[int, int, dict[str, Any]]]]],
    all_d_cases: list[tuple[int, int, dict[str, Any]]],
    *,
    layers: int,
    pd_modes: tuple[str, ...],
    prompt: int,
    output: int,
    ttft: float,
    tpot: float,
    pool_prefix_hit_ratio: float,
    max_sequences_per_dp: int | None,
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
    num_devices: int = 1,
) -> tuple[
    list[tuple[int, int, dict[str, Any]]],
    list[tuple[int, list[tuple[int, int, dict[str, Any]]]]],
    dict[tuple[int, int, int], tuple[Memory, bool, int, float]],
    dict[tuple[int, float], dict[str, Any]],
    dict[int, dict[int, float]],
    dict[int, tuple[Memory, tuple[float, ...], int]],
    dict[int, tuple[CaseSpec, dict[int, int], float]],
]:
    """Run single-process Stage-1 archetype screening and Stage-2 cache warm-up for PD stages."""
    hit_tokens_pool = int(math.floor(prompt * float(pool_prefix_hit_ratio))) if "pooled_dram_ssd" in pd_modes else 0
    is_pool_only = len(pd_modes) == 1 and pd_modes[0] == "pooled_dram_ssd"
    base_compute_tok = max(0, prompt - hit_tokens_pool) if is_pool_only else prompt
    base_hit_tok = hit_tokens_pool if is_pool_only else 0

    prefill_cache: dict[tuple[Any, ...], tuple[Memory, bool, int, float]] = {}
    allow_p_fallback = output > 1
    if layers >= ARCHETYPE_SCREEN_MIN_LAYERS and len(p_cases) > PD_PREFILL_STAGE2_FINALISTS:
        p_screen = []
        for s_i, (p_idx, _, raw_p) in enumerate(p_cases):
            res_p = _screen_prefill_job((
                p_idx, raw_p, prompt, base_compute_tok, base_hit_tok, ttft,
                apply_target_envelope, rewrite_plan, allow_unbound_formula, allow_p_fallback,
            ))
            p_screen.append(res_p)
            if res_p[3] is not None:
                prefill_cache[_prefill_cache_key(
                    p_idx, base_compute_tok, base_hit_tok,
                    apply_target_envelope, rewrite_plan, allow_unbound_formula,
                )] = res_p[3]
            if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE and (s_i + 1) % SCREEN_RECLAIM_STRIDE == 0:
                _reclaim_sweep_memory()
        if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE:
            _reclaim_sweep_memory()
        p_ok = sorted(((sc, idx) for idx, ok, sc, _ in p_screen if ok), reverse=True)
        keep_p = (
            {idx for _, idx in p_ok[:PD_PREFILL_STAGE2_FINALISTS]} if p_ok
            else {idx for _, idx in sorted(((sc, idx) for idx, _, sc, _ in p_screen), reverse=True)[:1]}
        )
        p_cases = [item for item in p_cases if item[0] in keep_p]

    d_probe_caches: dict[int, dict[int, float]] = {}
    if layers >= ARCHETYPE_SCREEN_MIN_LAYERS and len(all_d_cases) > PD_DECODE_STAGE2_FINALISTS:
        d_screen = []
        for s_i, (idx, _, raw_d) in enumerate(all_d_cases):
            d_screen.append(_screen_decode_job((
                idx, raw_d, prompt, output, tpot,
                apply_target_envelope, rewrite_plan, allow_unbound_formula,
            )))
            if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE and (s_i + 1) % SCREEN_RECLAIM_STRIDE == 0:
                _reclaim_sweep_memory()
        if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE:
            _reclaim_sweep_memory()
        d_probe_caches = {idx: dict(pc) for idx, _, _, pc in d_screen if pc}
        d_ok = sorted(((sc, idx) for idx, ok, sc, _ in d_screen if ok), reverse=True)
        keep_d = (
            {idx for _, idx in d_ok[:PD_DECODE_STAGE2_FINALISTS]} if d_ok
            else {idx for _, idx in sorted(((sc, idx) for idx, _, sc, _ in d_screen), reverse=True)[:1]}
        )
        d_by_mtp = [(m, [item for item in bucket if item[0] in keep_d]) for m, bucket in d_by_mtp]
        all_d_cases = [item for item in all_d_cases if item[0] in keep_d]

    decode_cache: dict[tuple[int, float], dict[str, Any]] = {}
    d_vram_caches: dict[int, tuple[Memory, tuple[float, ...], int]] = {}
    d_rep_caches: dict[int, tuple[CaseSpec, dict[int, int], float]] = {}
    p_tok_pairs = {(prompt, 0)} if any(m != "pooled_dram_ssd" for m in pd_modes) else set()
    if "pooled_dram_ssd" in pd_modes:
        p_tok_pairs.add((max(0, prompt - hit_tokens_pool), hit_tokens_pool))
    for p_idx, _, raw_p in p_cases:
        for c_tok, h_tok in sorted(p_tok_pairs):
            p_key = _prefill_cache_key(
                p_idx, c_tok, h_tok, apply_target_envelope, rewrite_plan, allow_unbound_formula,
            )
            if p_key in prefill_cache:
                continue
            _, res = _eval_prefill_exact_job((
                p_idx, raw_p, prompt, c_tok, h_tok,
                apply_target_envelope, rewrite_plan, allow_unbound_formula, allow_p_fallback,
            ))
            if res is not None:
                prefill_cache[p_key] = res

    for idx, _, raw_d in all_d_cases:
        p_cache = d_probe_caches.setdefault(idx, {})
        _, sla_key, d_sla, vram_info, rep_info = _eval_decode_exact_job((
            idx, raw_d, prompt, output, tpot, max_sequences_per_dp,
            p_cache, apply_target_envelope, rewrite_plan, allow_unbound_formula,
        ))
        if d_sla is not None:
            decode_cache[(idx, sla_key)] = d_sla
        if vram_info is not None:
            d_vram_caches[idx] = vram_info
        if rep_info is not None:
            d_rep_caches[idx] = rep_info
    return p_cases, d_by_mtp, prefill_cache, decode_cache, d_probe_caches, d_vram_caches, d_rep_caches


def _build_pd_candidate_row(
    rep: Mapping[str, Any],
    *,
    pd_mode: str,
    mtp_n: int,
    p_world: int,
    d_world: int,
) -> dict[str, Any]:
    best_ratio = rep.get("best_configuration")
    d_inst = rep["decode_instance"]
    d_us = float(d_inst["us_per_tok_die"])
    e2e_tps = float(best_ratio["sustainable_tps"]) if best_ratio else 0.0
    e2e_per_die = float(best_ratio["e2e_tps_per_die"]) if best_ratio else 0.0
    d_tps_per_die = (US_PER_SECOND / d_us) if (best_ratio is not None and d_us > 0.0) else 0.0
    return {
        "deployment_mode": "pd_disaggregated",
        "pd_transfer_mode": pd_mode,
        "mtp_proposal_tokens": mtp_n,
        "prefill_parallel": rep["prefill_instance"]["parallel"],
        "decode_parallel": d_inst["parallel"],
        "pd_ratio": best_ratio["pd_ratio"] if best_ratio else None,
        "prefill_instances": int(best_ratio["n_prefill"]) if best_ratio else 1,
        "decode_instances": int(best_ratio["n_decode"]) if best_ratio else 1,
        "prefill_used_dies": int(best_ratio["prefill_dies"]) if best_ratio else p_world,
        "decode_used_dies": int(best_ratio["decode_dies"]) if best_ratio else d_world,
        "used_dies": int(best_ratio["used_dies"]) if best_ratio else (p_world + d_world),
        "b_feasible": int(d_inst["b_feasible"]),
        "b_vram_max": int(d_inst["b_vram_max"]),
        "ttft_ms": float(rep["prefill_instance"].get("effective_ttft_ms", rep["prefill_instance"]["step_ms"])),
        "decode_step_ms": float(d_inst["step_ms_feasible"]),
        "handoff_tpot_ms": float(rep.get("handoff_tpot_ms", 0.0) or 0.0),
        "effective_tpot_ms": float(rep["effective_tpot_ms"]),
        "avg_accepted_tokens": float(d_inst["avg_accepted_tokens"]),
        "n_tok_per_die": float(d_inst["n_tok_per_die"]),
        "us_per_tok_die": d_us,
        "unused_sla_slack_ms": float(d_inst["unused_sla_slack_ms"]),
        "operating_regime": str(d_inst["operating_regime"]),
        "free_hbm_headroom_gib": float(d_inst["free_hbm_headroom_gib"]),
        "kv_transfer_ms_per_request": float(rep["kv_transfer_ms_per_request"]),
        "sustainable_rps": float(best_ratio["sustainable_rps"]) if best_ratio else 0.0,
        "prefill_kv_rps": (
            min(float(best_ratio["prefill_rps"]), float(best_ratio["kv_transfer_rps"])) if best_ratio else 0.0
        ),
        "sustainable_tps": e2e_tps,
        "sustainable_tps_per_die": d_tps_per_die if d_tps_per_die > 0.0 else e2e_per_die,
        "decode_tps_per_die": d_tps_per_die,
        "e2e_total_tps": e2e_tps,
        "e2e_tps_per_die": e2e_per_die,
        "dcp_codomain_valid": bool(d_inst.get("dcp_codomain_valid", True)),
        "dcp_codomain_reason": d_inst.get("dcp_codomain_reason"),
        "feasible": bool(best_ratio is not None),
        "infeasible_reason": (
            None if best_ratio is not None
            else (rep["candidates"][0]["infeasible_reason"] if rep["candidates"] else "infeasible")
        ),
    }


def _search_pd_deployments(
    compose_for: Any,
    *,
    num_devices: int,
    is_moe: bool,
    layers: int,
    model_spec: Any,
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
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]]:
    p_topos = _build_pd_stage_candidates(
        role="pd_prefill", max_dies=num_devices - 1, is_moe=is_moe,
        layers=layers, prompt_tokens=prompt, rec_topo=rec_topo, pinned=p_pin,
        model_spec=model_spec,
    )
    d_topos = _build_pd_stage_candidates(
        role="pd_decode", max_dies=num_devices - 1, is_moe=is_moe,
        layers=layers, prompt_tokens=prompt, rec_topo=rec_topo, pinned=d_pin,
        model_spec=model_spec,
    )
    pd_modes = (
        (str(pd_transfer_mode).strip().lower(),)
        if pd_transfer_mode is not None and str(pd_transfer_mode).strip().lower() != "auto"
        else (("pooled_dram_ssd",) if (float(pool_ssd_ratio) > 0.0 or float(pool_prefix_hit_ratio) > 0.0)
              else AUTO_PD_TRANSFER_MODES)
    )
    p_parsed_by_id: dict[int, CaseSpec] = {}
    p_cases: list[tuple[int, int, dict[str, Any]]] = []
    for idx, topo in enumerate(p_topos):
        raw = compose_for(topo, 0, "pd_prefill")
        if raw is not None:
            parsed_p = parse_case(raw)
            p_parsed_by_id[idx] = parsed_p
            p_cases.append((idx, parsed_p.runtime.parallel.world_size, raw))
    d_parsed_by_id: dict[int, CaseSpec] = {}
    d_by_mtp: list[tuple[int, list[tuple[int, int, dict[str, Any]]]]] = []
    all_d_cases: list[tuple[int, int, dict[str, Any]]] = []
    d_id = 0
    for mtp_n in mtp_grid:
        bucket: list[tuple[int, int, dict[str, Any]]] = []
        for topo in d_topos:
            raw_d = compose_for(topo, mtp_n, "pd_decode")
            if raw_d is not None:
                d_id += 1
                parsed_d = parse_case(raw_d)
                d_parsed_by_id[d_id] = parsed_d
                item = (d_id, parsed_d.runtime.parallel.world_size, raw_d)
                bucket.append(item)
                all_d_cases.append(item)
        d_by_mtp.append((mtp_n, bucket))

    (
        p_cases, d_by_mtp, prefill_cache, decode_cache,
        d_probe_caches, d_vram_caches, d_rep_caches,
    ) = _select_and_warm_pd_finalists(
        p_cases, d_by_mtp, all_d_cases, layers=layers, pd_modes=pd_modes,
        prompt=prompt, output=output, ttft=ttft, tpot=tpot,
        pool_prefix_hit_ratio=pool_prefix_hit_ratio,
        max_sequences_per_dp=max_sequences_per_dp,
        apply_target_envelope=apply_target_envelope,
        rewrite_plan=rewrite_plan,
        allow_unbound_formula=allow_unbound_formula,
        num_devices=num_devices,
    )

    rows: list[dict[str, Any]] = []
    feasible_entries: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for mtp_n, d_cases in d_by_mtp:
        for p_idx, p_world, p_raw in p_cases:
            for d_idx, d_world, d_raw in d_cases:
                if p_world + d_world > num_devices:
                    continue
                for pd_mode in pd_modes:
                    try:
                        rep = sweep_pd_disaggregation_ratios(
                            p_parsed_by_id[p_idx], d_parsed_by_id[d_idx],
                            total_dies=num_devices,
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
                            apply_target_envelope=apply_target_envelope,
                            rewrite_plan=rewrite_plan,
                            allow_unbound_formula=allow_unbound_formula,
                            _prefill_step_cache=prefill_cache, _prefill_cache_id=p_idx,
                            _decode_sla_cache=decode_cache, _decode_cache_id=d_idx,
                            _decode_probe_cache=d_probe_caches.setdefault(d_idx, {}),
                            _decode_vram_info=d_vram_caches.get(d_idx),
                            _decode_rep_info=d_rep_caches.get(d_idx),
                        )
                    except (ModelingError, ValueError) as exc:
                        if _is_ingress_contradiction(exc) or (rewrite_plan is not None and _is_rewrite_plan_error(exc)):
                            raise
                        continue
                    row = _build_pd_candidate_row(
                        rep, pd_mode=pd_mode, mtp_n=mtp_n, p_world=p_world, d_world=d_world,
                    )
                    rows.append(row)
                    if row["feasible"]:
                        feasible_entries.append((row, p_raw, d_raw))
    return rows, feasible_entries


_ALLOWED_PINNED_AXES = frozenset({
    "dp", "tp", "ep", "pp", "pcp", "dcp", "cp",
    "q_proj_tp", "o_proj_tp", "linear_proj_tp", "dcp_linear_tp",
    "mlp_tp", "shared_expert_tp", "lm_head_tp",
})
_INGRESS_ERROR_PREFIXES = (
    "quant_unsupported_on_hardware",
    "ascend_topology_preset_",
    "hardware_topology_preset_unsupported",
    "a2_superpod_unsupported",
    "enable_fused_mc2_level3_",
    "invalid_enable_fused_mc2",
    "hccl_ccu_unsupported_on_arch",
    "unknown_hardware_sku",
    "unknown_model",
    "invalid_quant_override",
    "unknown_parallel_axis",
    "unknown_pinned_parallel_axis",
)


def _validate_pinned_axes(pinned: Mapping[str, int] | None, label: str) -> dict[str, int]:
    if not pinned:
        return {}
    unknown = sorted(str(k) for k in pinned if str(k) not in _ALLOWED_PINNED_AXES)
    if unknown:
        raise ModelingError(f"unknown_pinned_parallel_axis:{label}:{','.join(unknown)}")
    return {str(k): int(v) for k, v in pinned.items()}


def _is_ingress_contradiction(exc: Exception) -> bool:
    return isinstance(exc, ModelingError) and str(exc).startswith(_INGRESS_ERROR_PREFIXES)


def _validate_optimal_deployment_inputs(
    *,
    num_devices: int | None,
    prompt_tokens: int | None,
    output_tokens: int | None,
    ttft_sla_ms: float | None,
    tpot_sla_ms: float | None,
    deployment: str,
    pd_transfer_mode: str | None,
    pool_dram_ratio: float | None = None,
    pool_ssd_ratio: float,
    pool_prefix_hit_ratio: float = 0.0,
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
    mode_norm = str(pd_transfer_mode or "").strip().lower()
    if mode_norm in {"direct_hbm", "direct_blocking_d2d", "layerwise_overlap_d2d"}:
        has_pool_ratio = (
            (pool_dram_ratio is not None and float(pool_dram_ratio) > 0.0)
            or float(pool_ssd_ratio) > 0.0
            or float(pool_prefix_hit_ratio) > 0.0
        )
        if has_pool_ratio:
            raise ModelingError(
                f"direct_hbm_conflicts_with_kv_pool_ratios:{mode_norm}"
            )
    return n_dev, prompt, output, ttft, tpot, dep_norm


def _finalize_optimal_deployment(
    *,
    model_id: str,
    sku: str,
    n_dev: int,
    prompt: int,
    output: int,
    ttft: float,
    tpot: float,
    dep_norm: str,
    all_candidates: list[dict[str, Any]],
    coloc_entries: list[tuple[dict[str, Any], dict[str, Any]]],
    pd_entries: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    compose_for: Any,
) -> dict[str, Any]:
    def rank_key(row: Mapping[str, Any]) -> tuple[float, int, float, float]:
        return (
            float(_single_die_decode_tps(row)),
            -int(row["used_dies"]),
            float(row.get("e2e_tps_per_die", row.get("sustainable_tps_per_die", 0.0)) or 0.0),
            -float(row["effective_tpot_ms"]),
        )

    best_coloc_entry = max(coloc_entries, key=lambda item: rank_key(item[0]), default=None)
    best_pd_entry = max(pd_entries, key=lambda item: rank_key(item[0]), default=None)
    winner_row = recommended_case = prefill_case_out = decode_case_out = None
    pd_pool_summary: dict[str, Any] | None = None
    if best_pd_entry and (best_coloc_entry is None or rank_key(best_pd_entry[0]) >= rank_key(best_coloc_entry[0])):
        winner_row, prefill_case_out, decode_case_out = (
            best_pd_entry[0], deepcopy(best_pd_entry[1]), deepcopy(best_pd_entry[2]),
        )
        recommended_case = deepcopy(best_pd_entry[1] if output <= 1 else best_pd_entry[2])
        pd_pool_summary = {
            "pd_ratio": winner_row.get("pd_ratio"),
            "pd_transfer_mode": winner_row.get("pd_transfer_mode"),
            "prefill_instances": int(winner_row.get("prefill_instances", 1) or 1),
            "decode_instances": int(winner_row.get("decode_instances", 1) or 1),
            "prefill_used_dies": int(winner_row.get("prefill_used_dies", 0) or 0),
            "decode_used_dies": int(winner_row.get("decode_used_dies", 0) or 0),
            "used_dies": int(winner_row.get("used_dies", 0) or 0),
            "prefill_parallel": dict(winner_row.get("prefill_parallel") or {}),
            "decode_parallel": dict(winner_row.get("decode_parallel") or {}),
            "handoff_tpot_ms": float(winner_row.get("handoff_tpot_ms", 0.0) or 0.0),
            "prefill_kv_rps": float(winner_row.get("prefill_kv_rps", 0.0) or 0.0),
            "e2e_total_tps": float(winner_row.get("e2e_total_tps", 0.0) or 0.0),
        }
    elif best_coloc_entry:
        winner_row, recommended_case = best_coloc_entry[0], deepcopy(best_coloc_entry[1])
    else:
        del compose_for
        recommended_case = None

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
        "pareto_frontier": _mark_pd_pareto(all_candidates, metric_fn=_single_die_decode_tps),
        "candidates": all_candidates,
        "recommended_case": recommended_case,
        "prefill_case": prefill_case_out,
        "decode_case": decode_case_out,
        "pd_pool_summary": pd_pool_summary,
    }


def _make_solve_compose_fn(
    common_kw: Mapping[str, Any], *, prompt: int, max_sequences_per_dp: int | None,
    mtp_acceptance_length: float | None,
):
    from mindie_sim.compose import compose_case

    def compose_for(topo: Mapping[str, int], mtp_n: int, role: str = "colocated") -> dict[str, Any] | None:
        accept = (
            1.0 if int(mtp_n) <= 0
            else (
                float(mtp_acceptance_length) if mtp_acceptance_length is not None
                else compute_speculative_accept_step(DEFAULT_SPECULATIVE_ACCEPT_RATE, mtp_n)
            )
        )
        extra_kw: dict[str, Any] = {}
        if role in {"colocated", "pd_prefill"}:
            seq_floor = int(max_sequences_per_dp or DEFAULT_CHUNKED_BATCH_SEQ_FLOOR)
            chunk_cap = max(DEFAULT_CHUNKED_PREFILL_TOKENS, seq_floor * (1 + max(0, int(mtp_n))))
            if prompt > chunk_cap:
                extra_kw["max_num_batched_tokens"] = chunk_cap
        try:
            return compose_case(
                **common_kw, **extra_kw, role=role, **{k: int(v) for k, v in topo.items()},
                mtp_proposal_tokens=mtp_n, mtp_acceptance_length=accept,
            )
        except (ModelingError, ValueError) as exc:
            if _is_ingress_contradiction(exc):
                raise
            return None

    return compose_for


def solve_optimal_deployment(
    model: str | Mapping[str, Any], hardware: str | Mapping[str, Any] = "a3", *,
    num_devices: int | None = None, prompt_tokens: int | None = None, output_tokens: int | None = None,
    ttft_sla_ms: float | None = None, tpot_sla_ms: float | None = None, deployment: str = "auto",
    pd_transfer_mode: str | None = None, kv_transfer_bw_gbps: float | None = None,
    pool_dram_ratio: float | None = None, pool_ssd_ratio: float = 0.0, pool_prefix_hit_ratio: float = 0.0,
    ddr_bw_gbps: float | None = None, ssd_bw_gbps: float | None = None,
    pinned_parallel: Mapping[str, int] | None = None, pinned_prefill_parallel: Mapping[str, int] | None = None,
    pinned_decode_parallel: Mapping[str, int] | None = None, mtp_proposal_tokens: int | None = None,
    mtp_acceptance_length: float | None = None, max_sequences_per_dp: int | None = None,
    quant: str | Mapping[str, Any] | None = None, layers: int | None = None,
    ranks_per_node: int | None = None, topology_preset: str | None = None,
    va_recommended: bool | None = None, raw_eager: bool = False, feature_track: str = "merged",
    runtime_stack: str | Mapping[str, Any] | None = None, optimizations: Sequence[str] = (),
    exclude_optimizations: Sequence[str] = (), engram_cpu_offload: bool | None = None,
    multistream_overlap_shared_expert: bool | None = None, enable_shared_expert_dp: bool | None = None,
    sp: bool | None = None, graph_capture_mode: str | None = None,
    kv_cache_dtype: str | None = None, indexer_kv_dtype: str | None = None,
    allow_unconstrained_ttft: bool = False, apply_target_envelope: bool = False,
    rewrite_plan: Any = None, allow_unbound_formula: bool = True,
    compose_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Jointly search PD Disaggregated vs Colocated + MTP under Domain 1/2/3."""
    n_dev, prompt, output, ttft, tpot, dep_norm = _validate_optimal_deployment_inputs(
        num_devices=num_devices, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        ttft_sla_ms=ttft_sla_ms, tpot_sla_ms=tpot_sla_ms, deployment=deployment,
        pd_transfer_mode=pd_transfer_mode, pool_dram_ratio=pool_dram_ratio,
        pool_ssd_ratio=pool_ssd_ratio, pool_prefix_hit_ratio=pool_prefix_hit_ratio,
        ssd_bw_gbps=ssd_bw_gbps, allow_unconstrained_ttft=allow_unconstrained_ttft,
    )
    coloc_pin = _validate_pinned_axes(pinned_parallel, "pinned_parallel")
    p_pin = _validate_pinned_axes(
        pinned_prefill_parallel if pinned_prefill_parallel is not None else coloc_pin,
        "pinned_prefill_parallel",
    )
    d_pin = _validate_pinned_axes(
        pinned_decode_parallel if pinned_decode_parallel is not None else coloc_pin,
        "pinned_decode_parallel",
    )
    ext_plan = load_rewrite_plan(rewrite_plan) if rewrite_plan is not None else None
    use_target_env = bool(apply_target_envelope or ext_plan is not None)

    from mindie_sim.compose.inputs import _arch_key, _resolve_hardware, _resolve_model_spec
    from mindie_sim.compose.runtime import va_recommended_config

    sku, _, _ = _resolve_hardware(hardware)
    model_spec = _resolve_model_spec(model, hw_family=_arch_key(sku))
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
        quant=quant, layers=layers, ranks_per_node=ranks_per_node, topology_preset=topology_preset,
        va_recommended=va_recommended, raw_eager=raw_eager, feature_track=feature_track,
        runtime_stack=runtime_stack, optimizations=optimizations,
        exclude_optimizations=exclude_optimizations, engram_cpu_offload=engram_cpu_offload,
        multistream_overlap_shared_expert=multistream_overlap_shared_expert,
        enable_shared_expert_dp=enable_shared_expert_dp, sp=sp,
        graph_capture_mode=graph_capture_mode, kv_cache_dtype=kv_cache_dtype,
        indexer_kv_dtype=indexer_kv_dtype, parallel="fixed", homogeneous_graph_reuse="auto",
        **{k: v for k, v in (compose_overrides or {}).items() if v is not None},
    )
    compose_for = _make_solve_compose_fn(
        common_kw, prompt=prompt, max_sequences_per_dp=max_sequences_per_dp,
        mtp_acceptance_length=mtp_acceptance_length,
    )
    # Validate global hardware/quant/preset/MC2 contradictions once before candidate search.
    compose_for({"dp": 1, "tp": 1, "ep": 1, "pp": 1, "pcp": 1, "dcp": 1}, 0)

    all_candidates: list[dict[str, Any]] = []
    coloc_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pd_entries: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []

    if dep_norm in {"auto", "colocated"}:
        c_rows, coloc_entries = _search_colocated_deployments(
            compose_for, num_devices=n_dev, is_moe=is_moe, layers=eff_layers,
            model_spec=model_spec, rec_topo=rec_topo, pinned=coloc_pin, mtp_grid=mtp_grid,
            prompt=prompt, output=output, ttft=ttft, tpot=tpot,
            max_sequences_per_dp=max_sequences_per_dp, apply_target_envelope=use_target_env,
            rewrite_plan=ext_plan, allow_unbound_formula=allow_unbound_formula,
        )
        all_candidates.extend(c_rows)
    if dep_norm in {"auto", "pd_disaggregated"} and n_dev >= 2:
        p_rows, pd_entries = _search_pd_deployments(
            compose_for, num_devices=n_dev, is_moe=is_moe, layers=eff_layers,
            model_spec=model_spec, rec_topo=rec_topo, p_pin=p_pin, d_pin=d_pin, mtp_grid=mtp_grid,
            prompt=prompt, output=output, ttft=ttft, tpot=tpot, pd_transfer_mode=pd_transfer_mode,
            kv_transfer_bw_gbps=kv_transfer_bw_gbps, pool_dram_ratio=pool_dram_ratio,
            pool_ssd_ratio=pool_ssd_ratio, pool_prefix_hit_ratio=pool_prefix_hit_ratio,
            ddr_bw_gbps=ddr_bw_gbps, ssd_bw_gbps=ssd_bw_gbps,
            max_sequences_per_dp=max_sequences_per_dp, apply_target_envelope=use_target_env,
            rewrite_plan=ext_plan, allow_unbound_formula=allow_unbound_formula,
        )
        all_candidates.extend(p_rows)

    return _finalize_optimal_deployment(
        model_id=model_id, sku=sku, n_dev=n_dev, prompt=prompt, output=output,
        ttft=ttft, tpot=tpot, dep_norm=dep_norm, all_candidates=all_candidates,
        coloc_entries=coloc_entries, pd_entries=pd_entries, compose_for=compose_for,
    )


__all__ = [
    "AUTO_PD_TRANSFER_MODES",
    "VALID_PD_TRANSFER_MODES",
    "solve_optimal_deployment",
    "sweep_pd_disaggregation_ratios",
]
