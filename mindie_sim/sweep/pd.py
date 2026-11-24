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
import ctypes
import gc
import json
import math
import sys
from typing import Any

from mindie_sim.catalog.context_parallel import (
    legal_vllm_ascend_dcp_values,
    validate_vllm_ascend_dcp_codomain,
)
from mindie_sim.prediction.body_defaults import TargetGraphSet
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.disaggregated.handoff import compute_pd_handoff_plan
from mindie_sim.prediction.efficiency import (
    _lower_and_run_graph,
    _resolve_speculative_tokens,
)
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.execution.layer_runs import (
    LayerCohortPlan,
    discover_layer_cohorts,
    execute_native_cohort_graph,
)
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.sharding.local_tp import rank_zero_graph_reuse
from mindie_sim.prediction.offload.policy import (
    _DEFAULT_SSD_BANDWIDTH_UTILIZATION,
    _DEFAULT_SSD_LATENCY_US,
    missing_scenario_error,
    resolve_sku_host_dma_rates,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.scenario import (
    DEPLOYMENT_MODES,
    DeploymentMode,
    DeploymentOutcome,
    validate_deployment_mode,
)
from mindie_sim.sweep.mapreduce import (
    BoundedMapPool,
    resolve_granted_workers,
)
from mindie_sim.sweep.resize import check_pos_float, check_pos_int

MS_PER_S = 1000.0
US_PER_MS = 1000.0
US_PER_SECOND = 1_000_000.0
NS_PER_US = 1000.0
BYTES_PER_GIB = float(1024 ** 3)
BYTES_PER_MS_PER_GBPS = 1_000_000.0
PARETO_EPS = 1e-12
CLOSED_FORM_TIE_EPS = 1e-11
KV_RATE_FLOOR_MS = 1e-9
POOL_RATIO_SUM_TOL = 1e-6
HALF_RATIO = 0.5
RANK_TPS_DECIMALS = 6
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
ARCHETYPE_FOLD_MIN_LAYERS = 2
SLA_SLACK_BOUND_TOLERANCE_MS = 0.5
MIN_DISAGGREGATED_DEVICES = 2
MIN_ADMISSION_PROBE_SEQS = 16
REGIME_HBM_CAPACITY_BOUND = "HBM_CAPACITY_BOUND"
REGIME_CONFIGURED_SEQ_CAP_BOUND = "CONFIGURED_SEQ_CAP_BOUND"
REGIME_TOKEN_BUDGET_BOUND = "TOKEN_BUDGET_BOUND"
REGIME_SLA_LATENCY_BOUND = "SLA_LATENCY_BOUND"
_LOCAL_TP_AXES = (
    "q_proj_tp",
    "o_proj_tp",
    "linear_proj_tp",
    "dcp_linear_tp",
    "mlp_tp",
    "shared_expert_tp",
    "lm_head_tp",
)
_PREFILL_PCP_WIDTHS = (1, 2, 4)
PREFILL_PCP_MIN_PROMPT_TOKENS = 1024
DEFAULT_CHUNKED_PREFILL_TOKENS = 8192
DEFAULT_CHUNKED_BATCH_SEQ_FLOOR = 64
GC_RECLAIM_MIN_WORLD_SIZE = 8
STATIC_ADMIT_RECLAIM_STRIDE = 4


_RECLAIM_LRU_PREFIXES = (
    "mindie_sim.prediction.operators.",
    "mindie_sim.prediction.ownership",
    "mindie_sim.prediction.sharding.",
    "mindie_sim.prediction.runtime.memory_physics",
    "mindie_sim.prediction.engine_periodic.",
)
_WORKER_EVAL_CACHE_MAX = 2
_WORKER_EVAL_CACHE: OrderedDict[str, tuple[CaseSpec, LayerCohortPlan, EvaluationContext]] = OrderedDict()


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
        _WORKER_EVAL_CACHE, _SOLVE_COMPOSE_CACHE,
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
    del parsed_pool, memory, tokens
    raise NotImplementedError(
        "LEGACY_PURGED: deleted scalar peak-division _pool_kv_hbm_ms; price KV transport "
        "through canonical PDConnector/Engine instead of scalar peak division (S6-W5-32)"
    )


def _extract_machine_link_tier(
    machine: Any, preferred: str,
) -> tuple[float, float, str, str] | None:
    """Return ``(bytes_per_ms, latency_us, semantics, tier_name)`` strictly for ``preferred``."""
    for tier in (preferred,):
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
    """Compute exact P->D wire transfer time via canonical PDConnector/Engine."""
    del handoff_plan, link_info
    raise NotImplementedError(
        "LEGACY_PURGED: deleted scalar peak-division _compute_d2d_wire_ms; price P->D wire "
        "transfer via canonical PDConnector/Engine with G7 fleet shared resources (S6-W5-32)"
    )


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
) -> tuple[float, bool]:
    """Price prefill with optional ``rewrite_plan`` (fail-fast on mismatch; zero silent drop)."""
    ms = _prefill_step_ms(
        parsed, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
        min_layers=min_layers, rep_info=rep_info, eval_ctx_box=eval_ctx_box,
        reclaim_on_exit=reclaim_on_exit,
        apply_target_envelope=apply_target_envelope,
        rewrite_plan=rewrite_plan,
        allow_unbound_formula=allow_unbound_formula,
    )
    return ms, (rewrite_plan is not None)


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
        )
    elif compute_tokens > 0:
        p_step = math.inf
    else:
        p_step = 0.0
    result = (p_mem, prefill_hbm_ok, p_slots, p_step)
    if key is not None:
        cache[key] = result
    return result


def _compute_layerwise_overlap_ms(
    prefill_ms: float,
    transfer_ms: float,
    *,
    per_layer_bytes: Mapping[int, float] | None,
    num_layers: int,
) -> tuple[float, float]:
    """Derive ``(slot_hold_ms, exposed_kv_ms)`` from causal Engine/PDConnector layer publish."""
    del prefill_ms, transfer_ms, per_layer_bytes, num_layers
    raise NotImplementedError(
        "LEGACY_PURGED: deleted uniform-layer linear publish overlap fallback; compute "
        "layerwise publish overlap via causal Engine / PDConnector timing (S6-W5-36)"
    )


def _resolve_decode_batch_bounds(
    parsed: CaseSpec,
    *,
    q_tok: int,
    b_vram: int,
    max_sequences_per_dp: int | None,
) -> tuple[int, int, int]:
    """Return ``(search_limit, configured_max_seqs, token_batch_limit)``."""
    rt_max_seqs = max(0, int(parsed.runtime.max_num_seqs))
    cfg_seqs = (
        min(rt_max_seqs, max(0, int(max_sequences_per_dp)))
        if max_sequences_per_dp is not None
        else rt_max_seqs
    )
    budget = int(parsed.runtime.max_num_batched_tokens or 0)
    tok_limit = (budget // max(1, int(q_tok))) if budget > 0 else int(b_vram)
    limit = max(0, min(int(b_vram), int(cfg_seqs), int(tok_limit)))
    return limit, int(cfg_seqs), int(tok_limit)


def _classify_decode_regime(
    *,
    feasible: bool,
    b_feasible: int,
    b_vram: int,
    configured_max_seqs: int,
    token_batch_limit: int,
    unused_sla_slack_ms: float,
) -> tuple[str, str]:
    """Return ``(operating_regime, limiting_constraint)`` without mislabeling caps as HBM."""
    if not feasible:
        if b_vram < 1:
            return REGIME_HBM_CAPACITY_BOUND, "static_weights_exceed_hbm"
        return REGIME_SLA_LATENCY_BOUND, "tpot_sla"
    if unused_sla_slack_ms > SLA_SLACK_BOUND_TOLERANCE_MS:
        if b_vram >= 1 and b_feasible >= b_vram:
            return REGIME_HBM_CAPACITY_BOUND, "hbm_capacity"
        if configured_max_seqs < b_vram and b_feasible >= configured_max_seqs:
            return REGIME_CONFIGURED_SEQ_CAP_BOUND, "runtime_max_seqs"
        if token_batch_limit < b_vram and b_feasible >= token_batch_limit:
            return REGIME_TOKEN_BUDGET_BOUND, "token_budget"
    return REGIME_SLA_LATENCY_BOUND, "tpot_sla"


def _assemble_decode_stage_from_sla(
    d_parsed: CaseSpec,
    *,
    d_sla: Mapping[str, Any],
    length_ok: bool,
    tpot_sla_ms: float,
    exposed_kv_ms: float,
    decode_tokens: int,
    output: int,
) -> dict[str, Any]:
    """Assemble canonical decode-stage metrics from an SLA concurrency solve."""
    d_par = d_parsed.runtime.parallel
    d_dp = int(d_par.dp)
    handoff_tpot = exposed_kv_ms / float(decode_tokens)
    decode_budget = tpot_sla_ms - handoff_tpot
    d_scen = d_sla["scenarios"][0]
    emitted = float(d_sla["emitted_tokens_per_seq"])
    b_vram = int(d_scen["b_vram_max"]) if length_ok else 0
    cfg_seqs = int(d_scen.get("configured_max_num_seqs", d_parsed.runtime.max_num_seqs))
    tok_lim = int(d_scen.get("token_batch_limit", b_vram))
    b_sla = int(d_scen["b_sla"]) if decode_budget > 0.0 else 0
    b_feasible = int(d_scen["b_feasible"])
    b1_step = float(d_scen.get("step_ms_at_b1") or 0.0)
    decode_step = (
        float(d_scen["step_ms_feasible"])
        if b_feasible >= 1 and d_scen.get("step_ms_feasible") is not None
        else b1_step
    )
    effective_tpot = (decode_step / max(emitted, PARETO_EPS) + handoff_tpot) if decode_step > 0.0 else 0.0
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
        if decode_step > 0.0 else 0.0
    )
    per_dp_dies = max(1, int(d_par.tp) * int(d_par.pcp) * int(d_par.pp))
    n_tok_per_die = (float(max(1, b_feasible)) * emitted) / float(per_dp_dies) if b_feasible >= 1 else 0.0
    us_per_tok_die = (decode_step * US_PER_MS) / n_tok_per_die if n_tok_per_die > 0.0 else 0.0
    decode_tps_per_die = (US_PER_SECOND / us_per_tok_die) if (tpot_ok and us_per_tok_die > 0.0) else 0.0
    unused_sla_slack_ms = max(0.0, float(tpot_sla_ms) - float(effective_tpot)) if tpot_ok else 0.0
    operating_regime, limiting_constraint = _classify_decode_regime(
        feasible=tpot_ok,
        b_feasible=b_feasible,
        b_vram=b_vram,
        configured_max_seqs=cfg_seqs,
        token_batch_limit=tok_lim,
        unused_sla_slack_ms=unused_sla_slack_ms,
    )
    free_hbm_gib = float(d_scen.get("free_hbm_headroom_gib", 0.0) or 0.0)
    return {
        "decode_budget": decode_budget, "length_ok": length_ok,
        "b_vram": b_vram, "configured_max_num_seqs": cfg_seqs,
        "token_batch_limit": tok_lim, "b_sla": b_sla, "b_feasible": b_feasible,
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
        "limiting_constraint": limiting_constraint,
        "free_hbm_headroom_gib": free_hbm_gib,
        "dcp_codomain_valid": True,
        "dcp_codomain_reason": None,
    }


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
            "b_vram": 0, "configured_max_num_seqs": int(d_parsed.runtime.max_num_seqs),
            "token_batch_limit": 0, "b_sla": 0, "b_feasible": 0,
            "b1_step": 0.0, "decode_step": 0.0,
            "handoff_tpot_ms": float(handoff_tpot),
            "effective_tpot": 0.0, "first_itl": exposed_kv_ms,
            "tpot_ok": False, "rps_d": 0.0, "raw_rps_d": 0.0,
            "decode_tps": 0.0, "decode_tps_per_die": 0.0,
            "avg_accepted_tokens": 1.0,
            "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
            "unused_sla_slack_ms": 0.0, "operating_regime": REGIME_SLA_LATENCY_BOUND,
            "limiting_constraint": str(dcp_reason or "dcp_codomain_invalid"),
            "free_hbm_headroom_gib": 0.0,
            "dcp_codomain_valid": False, "dcp_codomain_reason": dcp_reason,
        }
    d_q_tok, d_emit_est = _resolve_speculative_tokens(
        d_parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    length_ok = bool(max(prompt, eff_decode) + output <= d_parsed.runtime.max_model_len)
    if not length_ok or (vram_info is not None and int(vram_info[2]) < 1):
        _, cfg_seqs, tok_lim = _resolve_decode_batch_bounds(
            d_parsed, q_tok=d_q_tok, b_vram=0, max_sequences_per_dp=max_sequences_per_dp,
        )
        lim_c = "decode_max_model_len_exceeded" if not length_ok else "static_weights_exceed_hbm"
        return {
            "decode_budget": decode_budget, "length_ok": length_ok,
            "b_vram": 0, "configured_max_num_seqs": cfg_seqs,
            "token_batch_limit": tok_lim, "b_sla": 0, "b_feasible": 0,
            "b1_step": 0.0, "decode_step": 0.0,
            "handoff_tpot_ms": float(handoff_tpot),
            "effective_tpot": 0.0, "first_itl": exposed_kv_ms,
            "tpot_ok": False, "rps_d": 0.0, "raw_rps_d": 0.0,
            "decode_tps": 0.0, "decode_tps_per_die": 0.0,
            "avg_accepted_tokens": float(d_emit_est),
            "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
            "unused_sla_slack_ms": 0.0, "operating_regime": REGIME_HBM_CAPACITY_BOUND,
            "limiting_constraint": lim_c, "free_hbm_headroom_gib": 0.0,
            "dcp_codomain_valid": True, "dcp_codomain_reason": None,
        }
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
            measure_b1=(cache is None),
        )
        if d_sla_key is not None:
            cache[d_sla_key] = d_sla
    return _assemble_decode_stage_from_sla(
        d_parsed, d_sla=d_sla, length_ok=length_ok,
        tpot_sla_ms=tpot_sla_ms, exposed_kv_ms=exposed_kv_ms,
        decode_tokens=decode_tokens, output=output,
    )


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


def _compute_pd_handoff_metrics(
    p_parsed: CaseSpec,
    d_parsed: CaseSpec,
    *,
    p_mem: Memory,
    d_mem: Memory,
    p_step: float,
    p_slots: int,
    prefill_hbm_ok: bool,
    p_dcp_ok: bool,
    p_dcp_reason: str | None,
    total: int,
    prompt: int,
    output: int,
    ttft_sla_ms: float,
    tpot_sla_ms: float,
    mode: str,
    dram_r: float,
    ssd_r: float,
    compute_tokens: int,
    hit_tokens: int,
    kv_transfer_bw_gbps: float | None,
    ddr_bw_gbps: float | None,
    ssd_bw_gbps: float | None,
) -> dict[str, Any]:
    """Compute exact P->D handoff/pool overlap and prefill rates without Engine calls."""
    p_dp = int(p_parsed.runtime.parallel.dp)
    prefill_role_error = None if p_dcp_ok else f"prefill:{p_dcp_reason}"
    decode_tokens = output - 1
    link_info = _resolve_pd_link_bandwidth(
        p_parsed, d_parsed, total_dies=total, kv_transfer_bw_gbps=kv_transfer_bw_gbps,
    )
    if decode_tokens <= 0:
        res = _prefill_only_pd_stage_result(
            p_dp=p_dp, p_step=p_step, p_slots=p_slots, prefill_hbm_ok=prefill_hbm_ok,
            prefill_role_error=prefill_role_error, p_dcp_ok=p_dcp_ok, p_dcp_reason=p_dcp_reason,
            ttft_sla_ms=ttft_sla_ms, tpot_sla_ms=tpot_sla_ms, compute_tokens=compute_tokens,
            hit_tokens=hit_tokens, output=output, d_parsed=d_parsed, link_info=link_info,
        )
        return {"prefill_only": True, "stage": res}

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
        p_slot_hold_ms, exposed_flush_ms = _compute_layerwise_overlap_ms(
            p_step, p_flush_ms,
            per_layer_bytes=handoff.get("per_layer_bytes"),
            num_layers=int(p_parsed.model.layers),
        )
        exposed_kv_ms = exposed_flush_ms + d_pull_ms
        per_rep = (
            min(
                MS_PER_S / max(p_step, PARETO_EPS) if p_step > 0.0 else float("inf"),
                (float(p_slots) * MS_PER_S) / max(p_slot_hold_ms, PARETO_EPS),
            ) if p_slot_hold_ms > 0.0 else float(p_slots) * MS_PER_S
        )
        p_kv_rps_per_inst = (MS_PER_S / max(p_flush_ms, KV_RATE_FLOOR_MS)) if p_flush_ms > 0.0 else float("inf")
        d_kv_rps_per_inst = MS_PER_S / max(d_pull_ms, KV_RATE_FLOOR_MS)
    else:
        wire_ms = _compute_d2d_wire_ms(handoff, link_info)
        kv_ms = max(wire_ms, _pool_kv_hbm_ms(p_parsed, p_mem, prompt), _pool_kv_hbm_ms(d_parsed, d_mem, prompt))
        tail_layer_kv_ms = kv_ms * tail_frac
        if mode == "layerwise_overlap_d2d":
            p_slot_hold_ms, exposed_kv_ms = _compute_layerwise_overlap_ms(
                p_step, kv_ms,
                per_layer_bytes=handoff.get("per_layer_bytes"),
                num_layers=int(p_parsed.model.layers),
            )
        else:
            p_slot_hold_ms = p_step + kv_ms
            exposed_kv_ms = kv_ms
        per_rep = min(
            MS_PER_S / max(p_step, PARETO_EPS),
            (float(p_slots) * MS_PER_S) / max(p_slot_hold_ms, PARETO_EPS),
        )
        p_kv_rps_per_inst = MS_PER_S / max(kv_ms, KV_RATE_FLOOR_MS)

    prefill_compute_ok = bool(p_step <= ttft_sla_ms)
    effective_ttft_ms = p_step + (exposed_kv_ms if mode == "direct_blocking_d2d" else 0.0)
    prefill_ok = bool(effective_ttft_ms <= ttft_sla_ms)
    rps_p = (p_dp * per_rep) if (prefill_ok and prefill_hbm_ok) else 0.0
    return {
        "prefill_only": False,
        "prefill_role_error": prefill_role_error,
        "p_dcp_ok": bool(p_dcp_ok),
        "p_dcp_reason": p_dcp_reason,
        "prefill_hbm_ok": bool(prefill_hbm_ok),
        "prefill_compute_ok": prefill_compute_ok,
        "prefill_ok": prefill_ok,
        "p_step": float(p_step),
        "effective_ttft_ms": float(effective_ttft_ms),
        "compute_tokens": int(compute_tokens),
        "hit_tokens": int(hit_tokens),
        "kv_bytes": kv_bytes,
        "unique_kv_bytes": float(handoff["unique_kv_bytes"]),
        "kv_ms": kv_ms,
        "exposed_kv_ms": exposed_kv_ms,
        "tail_layer_kv_ms": tail_layer_kv_ms,
        "rps_p": rps_p,
        "p_kv_rps_per_inst": p_kv_rps_per_inst,
        "d_kv_rps_per_inst": d_kv_rps_per_inst,
        "link_info": link_info,
        "pool_metrics": pool_metrics,
        "handoff_plan": handoff,
        "decode_tokens": decode_tokens,
    }


def _finalize_pd_stage_dict(handoff_info: Mapping[str, Any], dec: Mapping[str, Any]) -> dict[str, Any]:
    """Combine prefill+handoff stage metrics with decode stage metrics."""
    prefill_role_error = handoff_info["prefill_role_error"]
    infeasible = (
        prefill_role_error if prefill_role_error is not None
        else dec["dcp_codomain_reason"] if not dec.get("dcp_codomain_valid", True)
        else "prefill_hbm_oom" if not handoff_info["prefill_hbm_ok"]
        else "prefill_ttft_sla_exceeded" if not handoff_info["prefill_compute_ok"]
        else "decode_max_model_len_exceeded" if not dec["length_ok"]
        else "decode_hbm_oom" if dec["b_vram"] < 1
        else "effective_tpot_sla_exceeded" if not dec["tpot_ok"]
        else "prefill_ttft_sla_exceeded" if not handoff_info["prefill_ok"]
        else None
    )
    return {
        "p_step": handoff_info["p_step"],
        "effective_ttft_ms": handoff_info["effective_ttft_ms"],
        "prefill_hbm_ok": handoff_info["prefill_hbm_ok"],
        "prefill_ok": handoff_info["prefill_ok"],
        "compute_tokens": handoff_info["compute_tokens"],
        "hit_tokens": handoff_info["hit_tokens"],
        "kv_bytes": handoff_info["kv_bytes"],
        "unique_kv_bytes": handoff_info["unique_kv_bytes"],
        "kv_ms": handoff_info["kv_ms"],
        "exposed_kv_ms": handoff_info["exposed_kv_ms"],
        "tail_layer_kv_ms": handoff_info["tail_layer_kv_ms"],
        "rps_p": handoff_info["rps_p"],
        "p_kv_rps_per_inst": handoff_info["p_kv_rps_per_inst"],
        "d_kv_rps_per_inst": handoff_info["d_kv_rps_per_inst"],
        "infeasible": infeasible,
        "link_info": handoff_info["link_info"],
        "pool_metrics": handoff_info["pool_metrics"],
        "handoff_plan": handoff_info["handoff_plan"],
        "prefill_dcp_codomain_valid": handoff_info["p_dcp_ok"],
        "prefill_dcp_codomain_reason": handoff_info["p_dcp_reason"],
        **dec,
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
    p_dcp_ok, p_dcp_reason = validate_vllm_ascend_dcp_codomain(p_parsed.model, p_parsed.runtime.parallel)
    hit_tokens = min(int(math.floor(prompt * hit_r)), max(0, prompt - 1)) if mode == "pooled_dram_ssd" else 0
    compute_tokens = max(0, prompt - hit_tokens)
    p_mem, prefill_hbm_ok, p_slots, p_step = _measure_prefill_stage(
        p_parsed, prompt=prompt, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
        cache=_prefill_step_cache, cache_id=_prefill_cache_id,
        apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
        allow_unbound_formula=allow_unbound_formula,
    )
    decode_q_tokens, _ = _resolve_speculative_tokens(
        d_parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    d_mem = Memory(d_parsed, workspace_tokens=decode_q_tokens)
    handoff_info = _compute_pd_handoff_metrics(
        p_parsed, d_parsed, p_mem=p_mem, d_mem=d_mem, p_step=p_step, p_slots=p_slots,
        prefill_hbm_ok=prefill_hbm_ok, p_dcp_ok=p_dcp_ok, p_dcp_reason=p_dcp_reason,
        total=total, prompt=prompt, output=output, ttft_sla_ms=ttft_sla_ms,
        tpot_sla_ms=tpot_sla_ms, mode=mode, dram_r=dram_r, ssd_r=ssd_r,
        compute_tokens=compute_tokens, hit_tokens=hit_tokens,
        kv_transfer_bw_gbps=kv_transfer_bw_gbps, ddr_bw_gbps=ddr_bw_gbps, ssd_bw_gbps=ssd_bw_gbps,
    )
    if handoff_info["prefill_only"]:
        return handoff_info["stage"]
    dec = _measure_decode_stage(
        d_parsed, prompt=prompt, output=output, eff_decode=eff_decode,
        tpot_sla_ms=tpot_sla_ms, exposed_kv_ms=handoff_info["exposed_kv_ms"],
        decode_tokens=handoff_info["decode_tokens"],
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
    return _finalize_pd_stage_dict(handoff_info, dec)


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


def _build_pd_ratio_row(
    st: Mapping[str, Any],
    *,
    n_p: int,
    n_d: int,
    total: int,
    p_dies: int,
    d_dies: int,
    output: int,
    mode: str,
) -> dict[str, Any]:
    del st, n_p, n_d, total, p_dies, d_dies, output, mode
    raise NotImplementedError(
        "LEGACY_PURGED: deleted isolated instance port multiplication "
        "(n_p*p_kv_rps_per_inst / n_d*d_kv_rps_per_inst); implement G7 fleet shared "
        "resource constraint sum_j lambda_j b_{j,r} <= C_r (S6-W5-32)"
    )


def _stern_brocot_pd_candidates(
    total: int, p_dies: int, d_dies: int, rate_p: float, rate_d: float,
) -> set[tuple[int, int]]:
    """Objective-split Stern-Brocot density candidates."""
    del total, p_dies, d_dies, rate_p, rate_d
    raise NotImplementedError(
        "LEGACY_PURGED: deleted single-objective Stern-Brocot dropping non-density "
        "Pareto points; implement G7 objective-split Stern-Brocot (density) vs "
        "per-budget Pareto frontier (S6-W5-32)"
    )


def _solve_optimal_pd_ratio_closed_form(
    st: Mapping[str, Any], *, total: int, p_dies: int, d_dies: int, output: int, mode: str,
) -> dict[str, Any] | None:
    """Closed-form PD ratio solver with G7 fleet shared-resource constraint."""
    del st, total, p_dies, d_dies, output, mode
    raise NotImplementedError(
        "LEGACY_PURGED: deleted isolated port-multiplied single-objective closed-form "
        "join; implement G7 fleet shared resource constraint sum_j lambda_j b_{j,r} <= C_r "
        "+ objective-split Stern-Brocot vs per-budget Pareto (S6-W5-32)"
    )


def _build_pd_ratio_rows(
    st: Mapping[str, Any], *, total: int, p_dies: int, d_dies: int, output: int, mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n_p in range(1, (total // p_dies) + 1):
        max_nd = (total - n_p * p_dies) // d_dies
        for n_d in range(1, max_nd + 1):
            rows.append(_build_pd_ratio_row(
                st, n_p=n_p, n_d=n_d, total=total, p_dies=p_dies, d_dies=d_dies,
                output=output, mode=mode,
            ))
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
            "configured_max_num_seqs": int(st.get("configured_max_num_seqs", d_parsed.runtime.max_num_seqs)),
            "token_batch_limit": int(st.get("token_batch_limit", st["b_vram"])),
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
            "limiting_constraint": str(st.get("limiting_constraint", "")),
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


def _parallel_width_grid(max_dies: int) -> tuple[int, ...]:
    """Return power-of-two parallel widths up to ``max_dies`` in descending order."""
    limit = max(1, int(max_dies))
    widths: list[int] = []
    width = 1
    while width <= limit:
        widths.append(width)
        width *= 2
    return tuple(reversed(widths))


def _candidate_dedup_key(cand: Mapping[str, int]) -> tuple[int, ...]:
    """Deduplication key preserving every global and local projection axis."""
    return (
        int(cand["dp"]), int(cand["tp"]), int(cand["ep"]),
        int(cand["pp"]), int(cand["pcp"]), int(cand["dcp"]),
        *(int(cand.get(axis, 1)) for axis in _LOCAL_TP_AXES),
    )


def _resolve_pd_prefill_speculative_k(
    model_id: str,
    model_spec: Any,
    *,
    decode_mtp_tokens: int,
    hw_family: str,
) -> int:
    """Resolve Prefill speculative proposal count via structural linear-attention contract."""
    del model_id, model_spec, decode_mtp_tokens, hw_family
    raise NotImplementedError(
        "LEGACY_PURGED: deleted AG014 is_hybrid_recurrent_model(model_id) and hardcoded "
        "linear-attention set; wire structural catalog.cache_pools.LINEAR_ATTENTION_KINDS "
        "and SpeculativeExecutionContract (S6-W5-34)"
    )


def _sync_derived_runtime_limits(
    raw_case: Mapping[str, Any],
    *,
    max_num_seqs: int,
    mtp_n: int,
    role: str,
) -> dict[str, Any]:
    """Update sequence ceiling, token budget, and capture buckets from physical admission."""
    from mindie_sim.stacks.vllm_ascend.speculative import compute_aclgraph_capture_sizes

    updated = deepcopy(dict(raw_case))
    rt = dict(updated.get("runtime") or {})
    eff_seqs = max(1, int(max_num_seqs))
    verify_width = 1 + max(0, int(mtp_n))
    rt["max_num_seqs"] = eff_seqs
    min_tokens = eff_seqs * verify_width
    cur_batched = int(rt.get("max_num_batched_tokens") or 0)
    if role == "pd_decode":
        rt["max_num_batched_tokens"] = max(cur_batched, min_tokens) if cur_batched > 0 else min_tokens
    elif cur_batched > 0:
        rt["max_num_batched_tokens"] = max(cur_batched, min_tokens)
    comp_cfg = rt.get("compilation_config")
    if isinstance(comp_cfg, Mapping):
        comp_copy = dict(comp_cfg)
        cg_mode = str(comp_copy.get("cudagraph_mode") or "FULL_DECODE_ONLY").strip().upper()
        new_sizes = list(compute_aclgraph_capture_sizes(
            None, num_speculative_tokens=max(0, int(mtp_n)),
            max_num_seqs=eff_seqs, cudagraph_mode=cg_mode,
        ))
        comp_copy["cudagraph_capture_sizes"] = new_sizes
        rt["compilation_config"] = comp_copy
        # Let Software.from_mapping derive capture keys strictly from compilation_config.
        rt.pop("graph_capture_keys", None)
        rt.pop("piecewise_capture_keys", None)
    elif rt.get("graph_capture_keys"):
        rt["graph_capture_keys"] = list(compute_aclgraph_capture_sizes(
            None, num_speculative_tokens=max(0, int(mtp_n)),
            max_num_seqs=eff_seqs, cudagraph_mode="FULL_DECODE_ONLY",
        ))
    updated["runtime"] = rt
    return updated


_DENSE_KV_ATTENTION_KINDS = frozenset({"gqa", "mla"})
_SOLVE_COMPOSE_CACHE_MAX = 32
_PARALLEL_ADMIT_MIN_CANDS = 4
_ANCHOR_WAVE_UB_RATIO = 0.5
_SOLVE_COMPOSE_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()


def _provable_decode_per_seq_read_by_stage(
    parsed: CaseSpec,
    mem: Memory,
    *,
    prompt_tokens: int,
    pp_stages: int,
) -> list[float]:
    """Strictly provable minimum HBM read bytes per sequence per decode step by stage.

    Counts only mandatory per-step reads that every decode step must fetch from HBM:
      - 1x recurrent + conv state per linear-attention layer (``query_tokens=1``),
      - ``index_k`` + TopK KV rows for sparse-indexed dense layers,
      - unpadded full/window KV rows for non-sparse dense GQA/MLA layers.
    Consumer layers sharing another owner layer's KV, non-dense compressed/window
    caches, and offloaded KV tiers contribute ``0.0`` (strictly conservative).
    """
    out = [0.0] * pp_stages
    seq_ctx = int(prompt_tokens)
    if seq_ctx <= 0:
        return out
    if any(float(val or 0.0) > 0.0 for val in (mem.offload or {}).values()):
        return out
    from mindie_sim.catalog.cache import layer_cache_components, layer_stage_map
    from mindie_sim.catalog.cache_pools import LINEAR_ATTENTION_KINDS
    from mindie_sim.catalog.context_parallel import cp_interleave

    par = parsed.runtime.parallel
    smap = layer_stage_map(parsed.model, par, parsed.runtime)
    interleave = cp_interleave(parsed.runtime)
    pcp_div = float(max(1, int(getattr(par, "pcp", 1) or 1)))
    for lp in parsed.model.layer_plans:
        st = smap[lp.index] if lp.index < len(smap) else pp_stages - 1
        if lp.attention in LINEAR_ATTENTION_KINDS:
            comps = layer_cache_components(
                parsed.model, lp, par, seq_ctx, capacity_padded=False,
                interleave=interleave, runtime=parsed.runtime, query_tokens=1, phase="decode",
            )
            out[st] += float(comps.get("recurrent_state", 0.0)) + float(comps.get("conv_state", 0.0))
            continue
        if (lp.attention not in _DENSE_KV_ATTENTION_KINDS
                or lp.cache_kind != "dense"
                or 0 <= int(getattr(lp, "kv_owner_layer_id", -1)) < lp.index):
            continue
        topk = int(getattr(lp, "index_topk", 0) or 0)
        is_sparse = bool(
            topk > 0 or getattr(lp, "indexer", False)
            or int(getattr(lp, "topk_owner_layer_id", -1)) >= 0
        )
        if is_sparse:
            if getattr(lp, "owns_index_k", False) and getattr(lp, "indexer", False):
                idx_comps = layer_cache_components(
                    parsed.model, lp, par, seq_ctx, capacity_padded=False,
                    interleave=interleave, runtime=parsed.runtime, query_tokens=1, phase="decode",
                )
                out[st] += float(idx_comps.get("index_k", 0.0)) / pcp_div
            if topk > 0:
                tk_comps = layer_cache_components(
                    parsed.model, lp, par, min(seq_ctx, topk), capacity_padded=False,
                    interleave=interleave, runtime=parsed.runtime, query_tokens=1, phase="decode",
                )
                out[st] += (float(tk_comps.get("gqa_kv", 0.0)) + float(tk_comps.get("mla_kv", 0.0))) / pcp_div
        else:
            comps = layer_cache_components(
                parsed.model, lp, par, seq_ctx, capacity_padded=False,
                interleave=interleave, runtime=parsed.runtime, query_tokens=1, phase="decode",
            )
            out[st] += (float(comps.get("gqa_kv", 0.0)) + float(comps.get("mla_kv", 0.0))) / pcp_div
    return out


def _provable_decode_tps_upper_bound(
    parsed: CaseSpec,
    mem: Memory,
    *,
    b_limit: int,
    q_tok: int,
    emit_est: float,
    tpot_sla_ms: float,
    prompt_tokens: int = 0,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
) -> float:
    """Strictly optimistic decode throughput ceiling (tok/s/die) for sound pruning."""
    del (
        parsed, mem, b_limit, q_tok, emit_est, tpot_sla_ms,
        prompt_tokens, rewrite_plan, allow_unbound_formula,
    )
    raise NotImplementedError(
        "LEGACY_PURGED: deleted duplicate expert inclusion probability formula, "
        "EMBED_MAX_ELEM_BYTES=4.0, and private _routed_expert_bytes_by_stage import; "
        "use schema.common.inclusion_probability and public Memory/catalog.weights contracts (S6-W5-38)"
    )


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
    for local_key in _LOCAL_TP_AXES:
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
    del model_spec, tp, pcp, prompt_tokens, pinned
    raise NotImplementedError(
        "LEGACY_PURGED: deleted arbitrary DECODE_DCP_MIN_PROMPT_TOKENS pruning; derive "
        "legal DCP/TP/PP/PCP parallel grids strictly from physical hardware/model contracts (S6-W5-39)"
    )


def _build_colocated_candidates(
    *,
    num_devices: int,
    is_moe: bool,
    layers: int,
    rec_topo: Mapping[str, Any] | None,
    pinned: Mapping[str, int],
    model_spec: Any = None,
    prompt_tokens: int = 0,
    role: str = "colocated",
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
        key = _candidate_dedup_key(cand)
        if key not in seen:
            seen.add(key)
            out.append(cand)

    seeded = _seed_from_profile_topo(rec_topo, role, max_dies=num_devices, pinned=pinned)
    if seeded:
        add(seeded)
    pp_vals = (pinned["pp"],) if "pp" in pinned else (1, 2)
    pcp_vals = (pinned["pcp"],) if "pcp" in pinned else (1,)
    tp_vals = (pinned["tp"],) if "tp" in pinned else _parallel_width_grid(num_devices)
    local_pins = {k: int(pinned[k]) for k in _LOCAL_TP_AXES if k in pinned}
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
                dcp_choices = _dcp_choices_for_candidate(
                    model_spec, tp=tp, pcp=pcp, prompt_tokens=prompt_tokens, pinned=pinned,
                )
                for dp in dp_choices:
                    if dp * unit > num_devices or dp < 1:
                        continue
                    for dcp in dcp_choices:
                        ep_val = pinned["ep"] if "ep" in pinned else (dp * tp * pcp if is_moe else 1)
                        add({"dp": dp, "tp": tp, "ep": ep_val, "pp": pp, "pcp": pcp, "dcp": dcp, **local_pins})
    return out


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
        key = _candidate_dedup_key(cand)
        if key not in seen:
            seen.add(key)
            out.append(cand)

    seeded = _seed_from_profile_topo(rec_topo, role, max_dies=max_dies, pinned=pinned)
    if seeded:
        add(seeded)
    widths = _parallel_width_grid(max_dies)
    local_pins = {k: int(pinned[k]) for k in _LOCAL_TP_AXES if k in pinned}
    if role == "pd_prefill":
        for tp in ((pinned["tp"],) if "tp" in pinned else widths):
            for pp in ((pinned["pp"],) if "pp" in pinned else (1, 2)):
                pcp_vals = (
                    (pinned["pcp"],) if "pcp" in pinned
                    else (tuple(reversed(_PREFILL_PCP_WIDTHS))
                          if prompt_tokens >= PREFILL_PCP_MIN_PROMPT_TOKENS else (1,))
                )
                for pcp in pcp_vals:
                    unit = tp * pp * pcp
                    if unit > max_dies:
                        continue
                    dp_choices = (
                        (pinned["dp"],) if "dp" in pinned
                        else tuple(dict.fromkeys([max_dies // unit, 1]))
                    )
                    for dp in dp_choices:
                        if dp * unit > max_dies or dp < 1:
                            continue
                        ep = int(pinned.get("ep", dp * tp * pcp if is_moe else 1))
                        add({"dp": dp, "tp": tp, "ep": ep, "pp": pp, "pcp": pcp,
                             "dcp": int(pinned.get("dcp", 1)), **local_pins})
    else:
        dp_widths = (pinned["dp"],) if "dp" in pinned else (widths if is_moe else (1,))
        pp_vals = (pinned["pp"],) if "pp" in pinned else (1, 2)
        for pp in pp_vals:
            for dp in dp_widths:
                for tp in ((pinned["tp"],) if "tp" in pinned else widths):
                    pcp = int(pinned.get("pcp", 1))
                    if dp * tp * pp * pcp > max_dies:
                        continue
                    ep = int(pinned.get("ep", dp * tp * pcp if is_moe else 1))
                    dcp_choices = _dcp_choices_for_candidate(
                        model_spec, tp=tp, pcp=pcp, prompt_tokens=prompt_tokens, pinned=pinned,
                    )
                    for dcp in dcp_choices:
                        add({"dp": dp, "tp": tp, "ep": ep, "pp": pp, "pcp": pcp, "dcp": dcp, **local_pins})
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
    regime: str = REGIME_SLA_LATENCY_BOUND,
    dcp_ok: bool = True,
    dcp_reason: str | None = None,
    deployment_mode: str = "colocated",
    configured_max_num_seqs: int = 0,
    limiting_constraint: str = "",
) -> dict[str, Any]:
    return {
        "deployment_mode": deployment_mode, "used_dies": used_dies, "parallel": par.to_dict(),
        "mtp_proposal_tokens": mtp_tokens,
        "ttft_ms": (None if deployment_mode == "pd_decode" else 0.0),
        "decode_step_ms": (None if deployment_mode == "pd_prefill" else 0.0),
        "effective_tpot_ms": (None if deployment_mode == "pd_prefill" else 0.0),
        "b_feasible": 0, "b_vram_max": 0,
        "configured_max_num_seqs": int(configured_max_num_seqs),
        "sustainable_rps": 0.0, "sustainable_tps": 0.0,
        "sustainable_tps_per_die": 0.0, "decode_tps_per_die": 0.0,
        "e2e_total_tps": 0.0, "e2e_tps_per_die": 0.0,
        "avg_accepted_tokens": 1.0, "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
        "unused_sla_slack_ms": 0.0, "operating_regime": regime,
        "limiting_constraint": str(limiting_constraint or reason or ""),
        "free_hbm_headroom_gib": 0.0,
        "dcp_codomain_valid": bool(dcp_ok), "dcp_codomain_reason": dcp_reason,
        "feasible": False, "infeasible_reason": reason,
    }


def _colocated_prefill_only_row(
    par: ParallelSpec, *, used_dies: int, mtp_tokens: int, p_step: float,
    output_tokens: int, feasible: bool, deployment_mode: str = "colocated",
    b_vram_max: int = 1, configured_max_num_seqs: int = 1,
) -> dict[str, Any]:
    rps = (par.dp * MS_PER_S) / max(p_step, PARETO_EPS) if feasible else 0.0
    e2e_tps = rps * float(output_tokens)
    e2e_per_die = e2e_tps / float(used_dies)
    is_pd_prefill = deployment_mode == "pd_prefill"
    return {
        "deployment_mode": deployment_mode, "used_dies": used_dies, "parallel": par.to_dict(),
        "mtp_proposal_tokens": mtp_tokens, "ttft_ms": p_step,
        "decode_step_ms": (None if is_pd_prefill else 0.0),
        "effective_tpot_ms": (None if is_pd_prefill else 0.0),
        "b_feasible": 1 if feasible else 0,
        "b_vram_max": int(b_vram_max) if feasible else 0,
        "configured_max_num_seqs": int(configured_max_num_seqs),
        "sustainable_rps": rps,
        "prefill_rps_per_die": (rps / float(used_dies)) if feasible else 0.0,
        "sustainable_tps": e2e_tps, "sustainable_tps_per_die": e2e_per_die,
        "decode_tps_per_die": 0.0, "e2e_total_tps": e2e_tps, "e2e_tps_per_die": e2e_per_die,
        "avg_accepted_tokens": 1.0, "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
        "unused_sla_slack_ms": 0.0, "operating_regime": REGIME_SLA_LATENCY_BOUND,
        "limiting_constraint": ("ttft_sla" if not feasible else "ttft_sla"),
        "free_hbm_headroom_gib": 0.0, "dcp_codomain_valid": True, "dcp_codomain_reason": None,
        "feasible": feasible, "infeasible_reason": None if feasible else "prefill_sla_or_oom",
    }


def _evaluate_pd_prefill_candidate(
    raw_case: Mapping[str, Any],
    *,
    prompt_tokens: int,
    output_tokens: int,
    ttft_sla_ms: float,
    mtp_tokens: int,
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
) -> dict[str, Any]:
    """Evaluate one standalone ``pd_prefill`` pool candidate (no decode timing, ``tpot_ms=None``)."""
    parsed = parse_case(raw_case)
    par = parsed.runtime.parallel
    used_dies = par.world_size
    dcp_ok, dcp_reason = validate_vllm_ascend_dcp_codomain(parsed.model, par)
    if not dcp_ok:
        return _colocated_infeasible_row(
            par, used_dies=used_dies, mtp_tokens=mtp_tokens, reason=dcp_reason,
            dcp_ok=False, dcp_reason=dcp_reason, deployment_mode="pd_prefill",
            configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
        )
    prefill_ws = _prefill_workspace_tokens(parsed, prompt_tokens)
    p_mem = Memory(parsed, workspace_tokens=prefill_ws)
    p_kv = p_mem.cache_bytes(prompt_tokens + 1, padded=True)
    hbm_ok = bool(
        prompt_tokens <= parsed.runtime.max_model_len
        and (parsed.runtime.enable_chunked_prefill or prompt_tokens <= parsed.runtime.max_num_batched_tokens)
        and all(cap >= 0.0 and cap >= kv for row in p_mem.capacity for cap, kv in zip(row, p_kv))
    )
    if not hbm_ok:
        return _colocated_infeasible_row(
            par, used_dies=used_dies, mtp_tokens=mtp_tokens, reason="prefill_hbm_oom",
            regime=REGIME_HBM_CAPACITY_BOUND, deployment_mode="pd_prefill",
            configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
            limiting_constraint="prefill_hbm_oom",
        )
    p_slots = min(
        [int(math.floor(max(0.0, float(cap)) / float(kv)))
         for row in p_mem.capacity for cap, kv in zip(row, p_kv) if kv > 0.0]
        or [int(parsed.runtime.max_num_seqs)]
    )
    rep_info = discover_layer_cohorts(parsed)
    ctx_box: list[EvaluationContext | None] = [None]
    try:
        p_step, _ = _eval_prefill_with_plan(
            parsed, compute_tokens=prompt_tokens, hit_tokens=0,
            min_layers=ARCHETYPE_FOLD_MIN_LAYERS, rep_info=rep_info, eval_ctx_box=ctx_box,
            apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
        )
    finally:
        if ctx_box[0] is not None:
            ctx_box[0] = None
            if len(parsed.machine.dies) >= GC_RECLAIM_MIN_WORLD_SIZE or int(par.dp) > 1:
                _reclaim_sweep_memory()
    feasible = bool(hbm_ok and p_step <= ttft_sla_ms + PARETO_EPS)
    return _colocated_prefill_only_row(
        par, used_dies=used_dies, mtp_tokens=mtp_tokens, p_step=p_step,
        output_tokens=output_tokens, feasible=feasible, deployment_mode="pd_prefill",
        b_vram_max=max(1, p_slots), configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
    )


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
    deployment_mode: str = "colocated",
    vram_info: tuple[Memory, tuple[float, ...], int] | None = None,
    measure_b1: bool = True,
) -> dict[str, Any]:
    parsed = parse_case(raw_case)
    par = parsed.runtime.parallel
    used_dies = par.world_size
    is_pd_decode = deployment_mode == "pd_decode"
    dcp_ok, dcp_reason = validate_vllm_ascend_dcp_codomain(parsed.model, par)
    if not dcp_ok:
        return _colocated_infeasible_row(
            par, used_dies=used_dies, mtp_tokens=mtp_tokens, reason=dcp_reason,
            dcp_ok=False, dcp_reason=dcp_reason, deployment_mode=deployment_mode,
            configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
        )
    q_tok, emit_est = _resolve_speculative_tokens(
        parsed, output_budget=output_tokens, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    if is_pd_decode:
        base_ws = max(1, int(q_tok))
        prefill_hbm_ok = True
    else:
        base_ws = _colocated_workspace_tokens(parsed, prompt_tokens)
        p_mem = Memory(parsed, workspace_tokens=base_ws)
        p_kv = p_mem.cache_bytes(prompt_tokens + 1, padded=True)
        prefill_hbm_ok = bool(
            prompt_tokens <= parsed.runtime.max_model_len
            and (parsed.runtime.enable_chunked_prefill
                 or prompt_tokens <= parsed.runtime.max_num_batched_tokens)
            and all(cap >= 0.0 and cap >= kv for row in p_mem.capacity for cap, kv in zip(row, p_kv))
        )
        if not prefill_hbm_ok:
            return _colocated_infeasible_row(
                par, used_dies=used_dies, mtp_tokens=mtp_tokens, reason="prefill_hbm_oom",
                regime=REGIME_HBM_CAPACITY_BOUND, deployment_mode=deployment_mode,
                configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
                limiting_constraint="prefill_hbm_oom",
            )
    length_ok = bool(prompt_tokens + output_tokens <= parsed.runtime.max_model_len)
    if not length_ok:
        return _colocated_infeasible_row(
            par, used_dies=used_dies, mtp_tokens=mtp_tokens, reason="max_model_len_exceeded",
            regime=REGIME_HBM_CAPACITY_BOUND, deployment_mode=deployment_mode,
            configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
            limiting_constraint="max_model_len_exceeded",
        )
    decode_tokens = max(0, output_tokens - 1)
    if decode_tokens == 0 and not is_pd_decode:
        rep_info = discover_layer_cohorts(parsed)
        ctx_box_p: list[EvaluationContext | None] = [None]
        try:
            p_step, _ = _eval_prefill_with_plan(
                parsed, compute_tokens=prompt_tokens, hit_tokens=0,
                min_layers=ARCHETYPE_FOLD_MIN_LAYERS, rep_info=rep_info, eval_ctx_box=ctx_box_p,
                apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
                allow_unbound_formula=allow_unbound_formula,
            )
        finally:
            if ctx_box_p[0] is not None:
                ctx_box_p[0] = None
                if len(parsed.machine.dies) >= GC_RECLAIM_MIN_WORLD_SIZE or int(par.dp) > 1:
                    _reclaim_sweep_memory()
        feasible = bool(dcp_ok and prefill_hbm_ok and (p_step <= ttft_sla_ms))
        return _colocated_prefill_only_row(
            par, used_dies=used_dies, mtp_tokens=mtp_tokens,
            p_step=p_step, output_tokens=output_tokens, feasible=feasible,
        )
    resolved_vram = vram_info or _solve_b_vram_max(
        parsed, base_ws=base_ws, decode_q=max(1, int(q_tok)),
        seq_len=prompt_tokens + output_tokens, cap_bound=None,
    )
    b_vram_static = int(resolved_vram[2])
    limit_static, cfg_seqs_static, _ = _resolve_decode_batch_bounds(
        parsed, q_tok=q_tok, b_vram=b_vram_static, max_sequences_per_dp=max_sequences_per_dp,
    )
    if b_vram_static < 1 or limit_static < 1:
        lim_c = "static_weights_exceed_hbm" if b_vram_static < 1 else "configured_or_token_limit_zero"
        return _colocated_infeasible_row(
            par, used_dies=used_dies, mtp_tokens=mtp_tokens, reason="decode_hbm_oom",
            regime=REGIME_HBM_CAPACITY_BOUND, deployment_mode=deployment_mode,
            configured_max_num_seqs=cfg_seqs_static, limiting_constraint=lim_c,
        )
    rep_info = discover_layer_cohorts(parsed)
    ctx_box: list[EvaluationContext | None] = [None]
    try:
        defer_prefill = is_pd_decode or (math.isinf(ttft_sla_ms) and rewrite_plan is None)
        p_step, _ = (0.0, False) if defer_prefill else _eval_prefill_with_plan(
            parsed, compute_tokens=prompt_tokens, hit_tokens=0,
            min_layers=ARCHETYPE_FOLD_MIN_LAYERS, rep_info=rep_info, eval_ctx_box=ctx_box,
            apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
        )
        d_sla = _solve_sla_archetype_aware(
            parsed, sla_ms=tpot_sla_ms * max(1.0, float(emit_est)),
            context=prompt_tokens, output_budget=output_tokens,
            max_sequences_per_dp=max_sequences_per_dp, workspace_tokens=base_ws,
            rep_info=rep_info, probe_cache=probe_cache, eval_ctx_box=ctx_box,
            apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula, vram_info=resolved_vram,
            measure_b1=measure_b1,
        )
        scen = d_sla["scenarios"][0]
        if (not is_pd_decode) and defer_prefill and int(scen["b_feasible"]) >= 1:
            p_step, _ = _eval_prefill_with_plan(
                parsed, compute_tokens=prompt_tokens, hit_tokens=0,
                min_layers=ARCHETYPE_FOLD_MIN_LAYERS, rep_info=rep_info, eval_ctx_box=ctx_box,
                apply_target_envelope=apply_target_envelope, rewrite_plan=None,
                allow_unbound_formula=allow_unbound_formula,
            )
        ttft_ok = True if is_pd_decode else bool(p_step <= ttft_sla_ms)
    finally:
        if ctx_box[0] is not None:
            ctx_box[0] = None
            if len(parsed.machine.dies) >= GC_RECLAIM_MIN_WORLD_SIZE or int(par.dp) > 1:
                _reclaim_sweep_memory()
    return _assemble_single_pool_row(
        par=par, used_dies=used_dies, mtp_tokens=mtp_tokens, deployment_mode=deployment_mode,
        d_sla=d_sla, scen=scen, p_step=p_step, output_tokens=output_tokens,
        tpot_sla_ms=tpot_sla_ms, prefill_hbm_ok=prefill_hbm_ok, ttft_ok=ttft_ok,
        length_ok=length_ok, runtime_max_seqs=int(parsed.runtime.max_num_seqs),
    )


def _assemble_single_pool_row(
    *,
    par: ParallelSpec,
    used_dies: int,
    mtp_tokens: int,
    deployment_mode: str,
    d_sla: Mapping[str, Any],
    scen: Mapping[str, Any],
    p_step: float,
    output_tokens: int,
    tpot_sla_ms: float,
    prefill_hbm_ok: bool,
    ttft_ok: bool,
    length_ok: bool,
    runtime_max_seqs: int,
) -> dict[str, Any]:
    is_pd_decode = deployment_mode == "pd_decode"
    emitted = float(d_sla["emitted_tokens_per_seq"])
    b_feasible = int(scen["b_feasible"])
    b_vram = int(scen.get("b_vram_max", b_feasible) or 0)
    cfg_seqs = int(scen.get("configured_max_num_seqs", runtime_max_seqs))
    tok_lim = int(scen.get("token_batch_limit", b_vram))
    decode_step = float(
        scen["step_ms_feasible"]
        if b_feasible >= 1 and scen.get("step_ms_feasible") is not None
        else scen.get("step_ms_at_b1", 0.0) or 0.0
    )
    tpot_ms = (decode_step / max(emitted, PARETO_EPS)) if decode_step > 0.0 else 0.0
    feasible = bool(prefill_hbm_ok and ttft_ok and length_ok and b_feasible >= 1 and tpot_ms <= tpot_sla_ms + PARETO_EPS)
    decode_tokens = max(1, int(output_tokens) - 1)
    per_dp_dies = max(1, int(par.tp) * int(par.pcp) * int(par.pp))
    n_tok_per_die = (float(b_feasible) * emitted) / float(per_dp_dies) if b_feasible >= 1 else 0.0
    us_per_tok_die = (decode_step * US_PER_MS) / n_tok_per_die if n_tok_per_die > 0.0 else 0.0
    decode_tps_per_die = (US_PER_SECOND / us_per_tok_die) if (feasible and us_per_tok_die > 0.0) else 0.0
    decode_total_tps = decode_tps_per_die * float(used_dies)
    if is_pd_decode:
        rps = (float(b_feasible * par.dp) * emitted * MS_PER_S) / (max(decode_step, PARETO_EPS) * float(decode_tokens)) if feasible else 0.0
        e2e_total_tps = decode_total_tps
        e2e_tps_per_die = decode_tps_per_die
    else:
        req_compute_ms = p_step + (float(decode_tokens) * decode_step) / max(float(max(1, b_feasible)) * emitted, PARETO_EPS)
        rps = (float(par.dp) * MS_PER_S) / max(req_compute_ms, PARETO_EPS) if feasible else 0.0
        e2e_total_tps = rps * float(output_tokens)
        e2e_tps_per_die = e2e_total_tps / float(used_dies)
    unused_slack_ms = max(0.0, float(tpot_sla_ms) - float(tpot_ms)) if feasible else 0.0
    operating_regime, limiting_constraint = _classify_decode_regime(
        feasible=feasible, b_feasible=b_feasible, b_vram=b_vram,
        configured_max_seqs=cfg_seqs, token_batch_limit=tok_lim,
        unused_sla_slack_ms=unused_slack_ms,
    )
    reason = (
        None if feasible else
        "prefill_hbm_oom" if not prefill_hbm_ok else
        "prefill_ttft_sla_exceeded" if not ttft_ok else
        "max_model_len_exceeded" if not length_ok else
        "decode_tpot_or_vram_exceeded"
    )
    return {
        "deployment_mode": deployment_mode, "used_dies": used_dies, "parallel": par.to_dict(),
        "mtp_proposal_tokens": mtp_tokens, "ttft_ms": (None if is_pd_decode else p_step),
        "decode_step_ms": decode_step, "effective_tpot_ms": tpot_ms,
        "b_feasible": b_feasible, "b_vram_max": b_vram,
        "configured_max_num_seqs": cfg_seqs, "token_batch_limit": tok_lim,
        "sustainable_rps": rps, "sustainable_tps": decode_total_tps,
        "sustainable_tps_per_die": decode_tps_per_die, "decode_tps_per_die": decode_tps_per_die,
        "e2e_total_tps": e2e_total_tps, "e2e_tps_per_die": e2e_tps_per_die,
        "avg_accepted_tokens": emitted, "n_tok_per_die": n_tok_per_die,
        "us_per_tok_die": us_per_tok_die, "unused_sla_slack_ms": unused_slack_ms,
        "operating_regime": operating_regime, "limiting_constraint": limiting_constraint,
        "free_hbm_headroom_gib": float(scen.get("free_hbm_headroom_gib", 0.0) or 0.0),
        "dcp_codomain_valid": True, "dcp_codomain_reason": None,
        "feasible": feasible, "infeasible_reason": reason,
    }


_REWRITE_PLAN_ERROR_TOKENS = (
    "REWRITE_PLAN_",
    "rewrite_fusion_recipe_unbound",
    "rewrite_unresolved_op",
    "rewrite_plan_",
    "unbound_formula",
    "simulation_mode_forbids_unbound",
)


def _is_rewrite_plan_error(exc: Exception) -> bool:
    """Return True when ``exc`` signals an explicit rewrite-plan or strict-binding failure."""
    msg = str(exc)
    return any(tok in msg for tok in _REWRITE_PLAN_ERROR_TOKENS)


def _load_canonical_rewrite_plan(rewrite_plan: Any) -> Any:
    """Unplugged socket replacing retired ``optimization.kernel_synthesis.load_rewrite_plan``."""
    del rewrite_plan
    raise NotImplementedError(
        "LEGACY_PURGED: deleted optimization.kernel_synthesis.load_rewrite_plan; "
        "wire canonical W3a rewrite.schema.load_rewrite_plan (S6-W5-31)"
    )


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
    """Unplugged socket replacing the deleted 372-line second target-graph adapter."""
    del (
        eff_case, timeline_rows, spec_by_id, phase, cohort_plan, n_tok_per_die,
        ext_plan, primitive_table, allow_unbound_formula, step_request, scenario_case,
    )
    raise NotImplementedError(
        "LEGACY_PURGED: deleted second target-graph adapter calling retired "
        "optimization.kernel_synthesis.solve_target_graph; wire W3a canonical "
        "optimization.ladder.solve_target_step_receipt and "
        "optimization.codesign.evaluate_macro_wall_trigger (S6-W5-31)"
    )


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
    exact_flag_out: list[bool] | None = None,
) -> float:
    """Exact native ``0..L-1 + final`` Engine replay with unified ``PrimitiveTable`` binding."""
    del min_layers, skip_target_if_baseline_le_ms
    cohort_plan = rep_info if isinstance(rep_info, LayerCohortPlan) else discover_layer_cohorts(parsed)
    use_target = bool(apply_target_envelope or rewrite_plan is not None)
    ext_plan = _load_canonical_rewrite_plan(rewrite_plan) if (use_target and rewrite_plan is not None) else None
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
    timeline_rows: tuple[Mapping[str, Any], ...] = ()
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
        if exact_flag_out is not None:
            exact_flag_out.append(True)
        return baseline_ms
    from mindie_sim.prediction.results import StepRequest

    target_ms = _solve_target_step_ms(
        eff_case, timeline_rows, spec_by_id,
        phase=phase, cohort_plan=cohort_plan,
        n_tok_per_die=n_tok_per_die, ext_plan=ext_plan,
        primitive_table=rebound_pt, allow_unbound_formula=allow_unbound_formula,
        step_request=StepRequest(phase=phase, tokens=tokens, context=context, sequences=sequences, scope=scope),
        scenario_case=parsed,
    )
    if exact_flag_out is not None:
        exact_flag_out.append(True)
    return target_ms


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
    cap_bound: int | None = None,
) -> tuple[Memory, tuple[float, ...], int]:
    """Solve physical HBM concurrency via canonical DieResidentReceipt / Memory."""
    del parsed, base_ws, decode_q, seq_len, cap_bound
    raise NotImplementedError(
        "LEGACY_PURGED: deleted _fits_vram_batch_fast private Memory surgery and "
        "broken _vram_solve_cache_key (omitting async_scheduling & reading nonexistent "
        "model.hidden); delegate admission to canonical prediction.capacity DieResidentReceipt "
        "keyed by full immutable case+hardware+async fingerprint (S6-W5-33)"
    )


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
    """Compute prefill latency via exact chunked-prefill scheduler stepping."""
    del (
        parsed, compute_tokens, hit_tokens, min_layers, rep_info, evaluation_context,
        eval_ctx_box, reclaim_on_exit, apply_target_envelope, rewrite_plan,
        allow_unbound_formula,
    )
    raise NotImplementedError(
        "LEGACY_PURGED: deleted midpoint chunk extrapolation (PREFILL_DIRECT_MAX_TOKENS=1024, "
        "PREFILL_CHUNK_PROBE_TOKENS=512); wire exact scheduler.batching chunk stepping (S6-W5-36)"
    )


def _tighten_sla_bracket(
    memo: Mapping[int, float], limit: int, sla_ms: float,
) -> tuple[int, int, int, float | None]:
    """Tighten binary-search ``(lo, hi, best_b, best_t)`` from exact cached batch probes."""
    lo, hi = 0, limit + 1
    best_b = 0
    best_t: float | None = None
    for k, t_val in sorted(memo.items()):
        if k < 1 or k > limit:
            continue
        if t_val <= sla_ms + PARETO_EPS:
            if k > lo:
                lo, best_b, best_t = k, k, t_val
        elif k < hi:
            hi = k
    return lo, hi, best_b, best_t


def _build_sla_scenario_result(
    parsed: CaseSpec,
    d_mem: Memory,
    d_kv: Sequence[float],
    *,
    emit_est: float,
    b_vram: int,
    cfg_seqs: int,
    tok_lim: int,
    base_ws: int,
    decode_q: int,
    b_sla: int,
    b_feas: int,
    t1_val: float,
    t_feas: float | None,
    headroom_b: int,
) -> dict[str, Any]:
    return {
        "emitted_tokens_per_seq": float(emit_est),
        "scenarios": [{
            "b_vram_max": b_vram,
            "configured_max_num_seqs": cfg_seqs,
            "token_batch_limit": tok_lim,
            "b_sla": b_sla, "b_feasible": b_feas,
            "step_ms_at_b1": t1_val, "step_ms_feasible": t_feas,
            "free_hbm_headroom_gib": _headroom_gib_at_batch(
                parsed, d_mem, d_kv, b_vram=b_vram, batch=headroom_b,
                base_ws=base_ws, decode_q=decode_q,
            ),
        }],
    }


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
    measure_b1: bool = True,
) -> dict[str, Any]:
    """Dual-certificate exact SLA concurrency solver (G7/GA-16)."""
    del (
        parsed, sla_ms, context, output_budget, max_sequences_per_dp,
        workspace_tokens, rep_info, evaluation_context, probe_cache, probe_sink,
        apply_target_envelope, rewrite_plan, allow_unbound_formula, vram_info,
        eval_ctx_box, measure_b1,
    )
    raise NotImplementedError(
        "LEGACY_PURGED: deleted uncertified monotone binary/gallop SLA search; "
        "implement dual-certificate non-monotone exact feasibility/throughput solver (S6-W5-35)"
    )


def _rank_single_pool_row(row: Mapping[str, Any]) -> tuple[float, int, float, float]:
    """Iso-SLA ranking key for colocated, pd_decode, and pd_disaggregated rows."""
    tpot_val = row.get("effective_tpot_ms")
    return (
        float(_single_die_decode_tps(row)),
        -int(row["used_dies"]),
        float(row.get("e2e_tps_per_die", row.get("sustainable_tps_per_die", 0.0)) or 0.0),
        -(float(tpot_val) if tpot_val is not None else 0.0),
    )


def _rank_pd_prefill_row(row: Mapping[str, Any]) -> tuple[float, int, float]:
    """Iso-SLA ranking key for standalone ``pd_prefill`` rows."""
    rps_per_die = float(row.get("prefill_rps_per_die", 0.0) or 0.0)
    if rps_per_die <= 0.0 and row.get("feasible"):
        rps_per_die = float(row.get("sustainable_rps", 0.0) or 0.0) / float(max(1, int(row["used_dies"])))
    ttft_val = float(row.get("ttft_ms", 0.0) or 0.0)
    return (rps_per_die, -int(row["used_dies"]), -ttft_val)


def _static_admit_single_pool_candidate(
    raw_case: dict[str, Any],
    *,
    topo: Mapping[str, int],
    mtp_n: int,
    dep_mode: str,
    prompt: int,
    output: int,
    tpot: float,
    max_sequences_per_dp: int | None,
    rewrite_plan: Any,
    allow_unbound_formula: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Run static HBM admission and provable throughput ceiling with zero Engine calls."""
    try:
        parsed = parse_case(raw_case)
    except UnmodeledCapabilityError:
        raise
    except (ModelingError, ValueError) as exc:
        if rewrite_plan is not None and _is_rewrite_plan_error(exc):
            raise
        return None, None
    par = parsed.runtime.parallel
    used_dies = par.world_size
    dcp_ok, dcp_reason = validate_vllm_ascend_dcp_codomain(parsed.model, par)
    if not dcp_ok:
        return None, _colocated_infeasible_row(
            par, used_dies=used_dies, mtp_tokens=mtp_n, reason=dcp_reason,
            dcp_ok=False, dcp_reason=dcp_reason, deployment_mode=dep_mode,
            configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
        )
    if prompt + output > parsed.runtime.max_model_len:
        return None, _colocated_infeasible_row(
            par, used_dies=used_dies, mtp_tokens=mtp_n, reason="max_model_len_exceeded",
            regime=REGIME_HBM_CAPACITY_BOUND, deployment_mode=dep_mode,
            configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
            limiting_constraint="max_model_len_exceeded",
        )
    q_tok, emit_est = _resolve_speculative_tokens(
        parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    if dep_mode == "colocated":
        base_ws = _colocated_workspace_tokens(parsed, prompt)
        p_mem = Memory(parsed, workspace_tokens=base_ws)
        p_kv = p_mem.cache_bytes(prompt + 1, padded=True)
        prefill_ok = bool(
            (parsed.runtime.enable_chunked_prefill or prompt <= parsed.runtime.max_num_batched_tokens)
            and all(cap >= 0.0 and cap >= kv for row in p_mem.capacity for cap, kv in zip(row, p_kv))
        )
        if not prefill_ok:
            return None, _colocated_infeasible_row(
                par, used_dies=used_dies, mtp_tokens=mtp_n, reason="prefill_hbm_oom",
                regime=REGIME_HBM_CAPACITY_BOUND, deployment_mode=dep_mode,
                configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
                limiting_constraint="prefill_hbm_oom",
            )
    else:
        base_ws = max(1, int(q_tok))
    d_mem, _, b_vram = _solve_b_vram_max(
        parsed, base_ws=base_ws, decode_q=max(1, int(q_tok)),
        seq_len=prompt + output, cap_bound=None,
    )
    if b_vram < 1:
        return None, _colocated_infeasible_row(
            par, used_dies=used_dies, mtp_tokens=mtp_n, reason="decode_hbm_oom",
            regime=REGIME_HBM_CAPACITY_BOUND, deployment_mode=dep_mode,
            configured_max_num_seqs=int(max_sequences_per_dp or parsed.runtime.max_num_seqs),
            limiting_constraint="static_weights_exceed_hbm",
        )
    if max_sequences_per_dp is None:
        b_limit = b_vram
    else:
        b_limit, cfg_seqs, _ = _resolve_decode_batch_bounds(
            parsed, q_tok=q_tok, b_vram=b_vram, max_sequences_per_dp=max_sequences_per_dp,
        )
        if b_limit < 1:
            return None, _colocated_infeasible_row(
                par, used_dies=used_dies, mtp_tokens=mtp_n, reason="decode_hbm_oom",
                regime=REGIME_CONFIGURED_SEQ_CAP_BOUND, deployment_mode=dep_mode,
                configured_max_num_seqs=cfg_seqs, limiting_constraint="configured_or_token_limit_zero",
            )
    ub_tps = (
        math.inf if (dep_mode == "colocated" and output <= 1)
        else _provable_decode_tps_upper_bound(
            parsed, d_mem, b_limit=b_limit, q_tok=max(1, int(q_tok)),
            emit_est=float(emit_est), tpot_sla_ms=tpot,
            prompt_tokens=prompt,
            rewrite_plan=rewrite_plan, allow_unbound_formula=allow_unbound_formula,
        )
    )
    return {
        "topo": dict(topo), "mtp_n": int(mtp_n), "used_dies": int(used_dies),
        "pp": int(par.pp), "b_vram": int(b_vram), "b_limit": int(b_limit),
        "ub_tps": float(ub_tps), "raw_case": raw_case,
    }, None


def _rebuild_worker_error(err_info: tuple[bool, str]) -> Exception:
    is_modeling_err, msg = err_info
    return ModelingError(msg) if is_modeling_err else ValueError(msg)


def _static_admit_single_pool_worker(task: tuple[Any, ...]) -> tuple[bool, Any]:
    """Top-level pickle-safe Map worker running compose + static HBM admission."""
    (
        compose_spec, topo, mtp_n, dep_mode, prompt, output, tpot,
        max_sequences_per_dp, rewrite_plan, allow_unbound_formula, num_devices,
    ) = task
    try:
        base_kw, c_prompt, c_max_seqs, c_accept = compose_spec
        compose_fn = _make_solve_compose_fn(
            base_kw, prompt=c_prompt, max_sequences_per_dp=c_max_seqs,
            mtp_acceptance_length=c_accept,
        )
        raw_c = compose_fn(topo, mtp_n, dep_mode)
        if raw_c is None:
            return True, (False, None, None)
        desc, fail_row = _static_admit_single_pool_candidate(
            raw_c, topo=topo, mtp_n=mtp_n, dep_mode=dep_mode,
            prompt=prompt, output=output, tpot=tpot,
            max_sequences_per_dp=max_sequences_per_dp,
            rewrite_plan=rewrite_plan, allow_unbound_formula=allow_unbound_formula,
        )
        return True, (True, desc, fail_row)
    except UnmodeledCapabilityError:
        raise
    except (ModelingError, ValueError) as exc:
        return False, (isinstance(exc, ModelingError), str(exc))
    finally:
        if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE:
            _reclaim_sweep_memory()


def _eval_decode_probe_worker(task: tuple[Any, ...]) -> tuple[bool, Any]:
    """Top-level pickle-safe Map worker evaluating one ``(cand_idx, batch)`` decode probe."""
    (
        cand_idx, cand_key, eval_case, batch_b, require_exact, prompt, base_ws,
        emit_est, per_dp_dies, sla_ms, apply_target_envelope, rewrite_plan,
        allow_unbound_formula, num_devices,
    ) = task
    try:
        cached = _WORKER_EVAL_CACHE.get(cand_key)
        if cached is not None:
            _WORKER_EVAL_CACHE.move_to_end(cand_key)
            parsed, rep_info, eval_ctx = cached
        else:
            max_entries = 1 if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE else _WORKER_EVAL_CACHE_MAX
            if len(_WORKER_EVAL_CACHE) >= max_entries:
                _WORKER_EVAL_CACHE.clear()
                if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE:
                    _reclaim_sweep_memory()
            parsed = parse_case(eval_case)
            rep_info = discover_layer_cohorts(parsed)
            eval_ctx = EvaluationContext.create(parsed)
            _WORKER_EVAL_CACHE[cand_key] = (parsed, rep_info, eval_ctx)
        use_target = bool(apply_target_envelope or rewrite_plan is not None)
        skip_thresh = (
            (sla_ms + PARETO_EPS)
            if (use_target and rewrite_plan is None and not require_exact)
            else None
        )
        exact_box: list[bool] = []
        ctx_box: list[EvaluationContext | None] = [eval_ctx]
        t_val = _archetype_weighted_step_ms(
            parsed, phase="decode", tokens=base_ws, context=prompt, sequences=batch_b,
            rep_info=rep_info, evaluation_context=eval_ctx,
            apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
            n_tok_per_die=(float(batch_b) * float(emit_est)) / float(per_dp_dies),
            eval_ctx_box=ctx_box, skip_target_if_baseline_le_ms=skip_thresh,
            reclaim_on_exit=False, exact_flag_out=exact_box,
        )
        is_exact = (
            bool(exact_box and exact_box[-1])
            or (not use_target)
            or bool(require_exact)
            or (t_val > sla_ms + PARETO_EPS)
        )
        return True, (cand_idx, int(batch_b), float(t_val), bool(is_exact))
    except UnmodeledCapabilityError:
        raise
    except (ModelingError, ValueError) as exc:
        return False, (cand_idx, isinstance(exc, ModelingError), str(exc))


def _pick_parallel_decode_probes(
    st: Mapping[str, Any], budget: int,
) -> list[tuple[int, bool]]:
    """Choose deterministic batch probes for dual-certificate SLA search."""
    del st, budget
    raise NotImplementedError(
        "LEGACY_PURGED: deleted worker-count-dependent uncertified secant/binary probe "
        "picker; implement dual-certificate non-monotone SLA probing (S6-W5-35)"
    )


def _init_pd_decode_wave_states(
    wave_items: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    prompt: int,
    output: int,
    tpot: float,
    max_sequences_per_dp: int | None,
) -> list[dict[str, Any]]:
    """Initialize per-candidate SLA search states for parallel ``pd_decode`` probing."""
    states: list[dict[str, Any]] = []
    for idx, (desc, eval_case) in enumerate(wave_items):
        try:
            parsed = parse_case(eval_case)
        except UnmodeledCapabilityError:
            raise
        except (ModelingError, ValueError) as exc:
            states.append({"done": True, "ok": False, "err": (isinstance(exc, ModelingError), str(exc))})
            continue
        par = parsed.runtime.parallel
        used_dies = par.world_size
        mtp_n = int(desc["mtp_n"])
        dcp_ok, dcp_reason = validate_vllm_ascend_dcp_codomain(parsed.model, par)
        if not dcp_ok:
            row = _colocated_infeasible_row(
                par, used_dies=used_dies, mtp_tokens=mtp_n, reason=dcp_reason,
                dcp_ok=False, dcp_reason=dcp_reason, deployment_mode="pd_decode",
                configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
            )
            states.append({"done": True, "ok": True, "row": row})
            continue
        if prompt + output > parsed.runtime.max_model_len:
            row = _colocated_infeasible_row(
                par, used_dies=used_dies, mtp_tokens=mtp_n, reason="max_model_len_exceeded",
                regime=REGIME_HBM_CAPACITY_BOUND, deployment_mode="pd_decode",
                configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
                limiting_constraint="max_model_len_exceeded",
            )
            states.append({"done": True, "ok": True, "row": row})
            continue
        q_tok, emit_est = _resolve_speculative_tokens(
            parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
        )
        base_ws = max(1, int(q_tok))
        d_mem, d_kv, b_vram = _solve_b_vram_max(
            parsed, base_ws=base_ws, decode_q=base_ws, seq_len=prompt + output, cap_bound=None,
        )
        limit, cfg_seqs, tok_lim = _resolve_decode_batch_bounds(
            parsed, q_tok=q_tok, b_vram=b_vram, max_sequences_per_dp=max_sequences_per_dp,
        )
        if b_vram < 1 or limit < 1:
            lim_c = "static_weights_exceed_hbm" if b_vram < 1 else "configured_or_token_limit_zero"
            row = _colocated_infeasible_row(
                par, used_dies=used_dies, mtp_tokens=mtp_n, reason="decode_hbm_oom",
                regime=REGIME_HBM_CAPACITY_BOUND, deployment_mode="pd_decode",
                configured_max_num_seqs=cfg_seqs, limiting_constraint=lim_c,
            )
            states.append({"done": True, "ok": True, "row": row})
            continue
        states.append({
            "idx": idx, "cand_key": f"w_{idx}_{par.dp}_{par.tp}_{par.ep}_{par.pp}_{par.cp}_{par.pcp}_{mtp_n}_{limit}",
            "eval_case": eval_case, "parsed": parsed, "par": par, "used_dies": used_dies,
            "mtp_n": mtp_n, "base_ws": base_ws, "emit_est": float(emit_est),
            "per_dp_dies": max(1, int(par.tp) * int(par.pcp) * int(par.pp)),
            "sla_ms": float(tpot) * max(1.0, float(emit_est)),
            "d_mem": d_mem, "d_kv": d_kv, "b_vram": int(b_vram),
            "limit": int(limit), "cfg_seqs": int(cfg_seqs), "tok_lim": int(tok_lim),
            "lo": 0, "hi": int(limit) + 1, "best_b": 0, "best_t": None,
            "memo": {}, "exact_keys": set(), "done": False, "ok": True, "row": None, "err": None,
        })
    return states


def _finalize_pd_decode_state(
    st: dict[str, Any], *, output: int, tpot: float,
) -> tuple[bool, Any]:
    """Assemble the exact single-pool row for a converged ``pd_decode`` candidate."""
    if not st["ok"]:
        return False, st["err"]
    if st["row"] is not None:
        return True, st["row"]
    best_b = int(st["best_b"])
    best_t = st["best_t"]
    memo = st["memo"]
    t1 = float(memo.get(1) or (best_t if best_b == 1 and best_t is not None else 0.0))
    headroom_b = best_b if best_b >= 1 else 0
    scen = {
        "b_vram_max": st["b_vram"],
        "configured_max_num_seqs": st["cfg_seqs"],
        "token_batch_limit": st["tok_lim"],
        "b_sla": best_b, "b_feasible": best_b,
        "step_ms_at_b1": t1, "step_ms_feasible": best_t,
        "free_hbm_headroom_gib": _headroom_gib_at_batch(
            st["parsed"], st["d_mem"], st["d_kv"], b_vram=st["b_vram"],
            batch=headroom_b, base_ws=st["base_ws"], decode_q=st["base_ws"],
        ),
    }
    row = _assemble_single_pool_row(
        par=st["par"], used_dies=st["used_dies"], mtp_tokens=st["mtp_n"],
        deployment_mode="pd_decode",
        d_sla={"emitted_tokens_per_seq": st["emit_est"], "scenarios": [scen]},
        scen=scen, p_step=0.0, output_tokens=output, tpot_sla_ms=tpot,
        prefill_hbm_ok=True, ttft_ok=True, length_ok=True,
        runtime_max_seqs=int(st["parsed"].runtime.max_num_seqs),
    )
    return True, row


def _eval_pd_decode_wave_parallel(
    pool: BoundedMapPool,
    wave_items: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    *,
    eff_workers: int,
    prompt: int,
    output: int,
    tpot: float,
    max_sequences_per_dp: int | None,
    apply_target_envelope: bool,
    rewrite_plan: Any,
    allow_unbound_formula: bool,
    num_devices: int,
) -> list[tuple[bool, Any]]:
    """Run synchronized k-ary multi-probe SLA rounds across all granted workers."""
    states = _init_pd_decode_wave_states(
        wave_items, prompt=prompt, output=output, tpot=tpot,
        max_sequences_per_dp=max_sequences_per_dp,
    )
    while True:
        active = [st for st in states if not st["done"]]
        if not active:
            break
        base_m = max(1, eff_workers // len(active))
        extra_m = eff_workers % len(active)
        round_tasks: list[tuple[Any, ...]] = []
        for pos, st in enumerate(active):
            budget = base_m + (1 if pos < extra_m else 0)
            for batch_b, req_ex in _pick_parallel_decode_probes(st, budget):
                round_tasks.append((
                    st["idx"], st["cand_key"], st["eval_case"], batch_b, req_ex,
                    prompt, st["base_ws"], st["emit_est"], st["per_dp_dies"],
                    st["sla_ms"], apply_target_envelope, rewrite_plan,
                    allow_unbound_formula, num_devices,
                ))
        if not round_tasks:
            for st in active:
                st["done"] = True
            break
        for ok, payload in pool.map_ordered(_eval_decode_probe_worker, round_tasks):
            if not ok:
                c_idx, is_mod, msg = payload
                states[c_idx]["ok"] = False
                states[c_idx]["err"] = (is_mod, msg)
                states[c_idx]["done"] = True
                continue
            c_idx, batch_b, t_val, is_exact = payload
            st = states[c_idx]
            if not st["ok"]:
                continue
            st["memo"][batch_b] = t_val
            if is_exact:
                st["exact_keys"].add(batch_b)
        for st in active:
            if st["done"]:
                continue
            lo, hi, best_b, best_t = _tighten_sla_bracket(st["memo"], st["limit"], st["sla_ms"])
            st["lo"], st["hi"], st["best_b"], st["best_t"] = lo, hi, best_b, best_t
            if lo == 0 and 1 in st["memo"] and st["memo"][1] > st["sla_ms"] + PARETO_EPS:
                st["best_b"], st["best_t"], st["done"] = 0, None, True
            elif lo >= 1 and lo + 1 >= hi and best_b in st["exact_keys"]:
                st["done"] = True
    return [_finalize_pd_decode_state(st, output=output, tpot=tpot) for st in states]


def _eval_single_pool_worker(task: tuple[Any, ...]) -> tuple[bool, Any]:
    """Top-level pickle-safe Map worker evaluating one single-pool candidate."""
    (
        eval_case, prompt, output, ttft, tpot, mtp_n, max_sequences_per_dp,
        apply_target_envelope, rewrite_plan, allow_unbound_formula, dep_mode, num_devices,
    ) = task
    try:
        row = _evaluate_colocated_candidate(
            eval_case, prompt_tokens=prompt, output_tokens=output,
            ttft_sla_ms=ttft, tpot_sla_ms=tpot, mtp_tokens=mtp_n,
            max_sequences_per_dp=max_sequences_per_dp,
            apply_target_envelope=apply_target_envelope,
            rewrite_plan=rewrite_plan, allow_unbound_formula=allow_unbound_formula,
            deployment_mode=dep_mode, measure_b1=False,
        )
        return True, row
    except UnmodeledCapabilityError:
        raise
    except (ModelingError, ValueError) as exc:
        return False, (isinstance(exc, ModelingError), str(exc))
    finally:
        if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE:
            _reclaim_sweep_memory()


def _eval_pd_prefill_worker(task: tuple[Any, ...]) -> tuple[bool, Any]:
    """Top-level pickle-safe Map worker evaluating one standalone ``pd_prefill`` candidate."""
    (
        raw_p, prompt, output, ttft, mtp_n,
        apply_target_envelope, rewrite_plan, allow_unbound_formula, num_devices,
    ) = task
    try:
        row = _evaluate_pd_prefill_candidate(
            raw_p, prompt_tokens=prompt, output_tokens=output, ttft_sla_ms=ttft,
            mtp_tokens=mtp_n, apply_target_envelope=apply_target_envelope,
            rewrite_plan=rewrite_plan, allow_unbound_formula=allow_unbound_formula,
        )
        return True, row
    except UnmodeledCapabilityError:
        raise
    except (ModelingError, ValueError) as exc:
        return False, (isinstance(exc, ModelingError), str(exc))
    finally:
        if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE:
            _reclaim_sweep_memory()


def _collect_admitted_single_pool_candidates(
    pool: BoundedMapPool,
    *,
    compose_for: Any,
    cand_pairs: Sequence[tuple[int, Mapping[str, int]]],
    eff_workers: int,
    dep_mode: str,
    num_devices: int,
    prompt: int,
    output: int,
    tpot: float,
    max_sequences_per_dp: int | None,
    rewrite_plan: Any,
    allow_unbound_formula: bool,
    coverage: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run static HBM admission across ``cand_pairs`` using ``pool`` when beneficial."""
    rows: list[dict[str, Any]] = []
    viable: list[dict[str, Any]] = []
    compose_spec = getattr(compose_for, "_spec", None)
    use_parallel = (
        eff_workers > 1
        and len(cand_pairs) >= _PARALLEL_ADMIT_MIN_CANDS
        and compose_spec is not None
    )
    if use_parallel:
        admit_tasks = [
            (
                compose_spec, dict(topo), int(mtp_n), dep_mode, prompt, output, tpot,
                max_sequences_per_dp, rewrite_plan, allow_unbound_formula, num_devices,
            )
            for mtp_n, topo in cand_pairs
        ]
        for ok, payload in pool.map_ordered(_static_admit_single_pool_worker, admit_tasks):
            if not ok:
                exc = _rebuild_worker_error(payload)
                if _is_ingress_contradiction(exc) or (rewrite_plan is not None and _is_rewrite_plan_error(exc)):
                    raise exc
                continue
            composed, desc, fail_row = payload
            if not composed:
                continue
            coverage["total_candidates"] += 1
            if desc is not None:
                coverage["static_admitted"] += 1
                viable.append(desc)
            else:
                coverage["static_rejected"] += 1
                if fail_row is not None:
                    rows.append(fail_row)
        return rows, viable

    for mtp_n, topo in cand_pairs:
        raw_c = compose_for(topo, mtp_n, dep_mode)
        if raw_c is None:
            continue
        coverage["total_candidates"] += 1
        desc, fail_row = _static_admit_single_pool_candidate(
            raw_c, topo=topo, mtp_n=mtp_n, dep_mode=dep_mode,
            prompt=prompt, output=output, tpot=tpot,
            max_sequences_per_dp=max_sequences_per_dp,
            rewrite_plan=rewrite_plan, allow_unbound_formula=allow_unbound_formula,
        )
        if desc is not None:
            coverage["static_admitted"] += 1
            viable.append(desc)
        else:
            coverage["static_rejected"] += 1
            if fail_row is not None:
                rows.append(fail_row)
        if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE and coverage["total_candidates"] % STATIC_ADMIT_RECLAIM_STRIDE == 0:
            _reclaim_sweep_memory()
    return rows, viable


def _search_single_pool_deployments(
    compose_for: Any,
    *,
    dep_mode: str,
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
    granted_workers: int | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    """Full-grid single-pool MapReduce search for ``colocated`` or ``pd_decode``."""
    topos = _build_colocated_candidates(
        num_devices=num_devices, is_moe=is_moe, layers=layers,
        rec_topo=rec_topo, pinned=pinned,
        model_spec=model_spec, prompt_tokens=prompt, role=dep_mode,
    )
    coverage = {
        "total_candidates": 0, "static_admitted": 0, "static_rejected": 0,
        "pruned_by_upper_bound": 0, "exact_evaluated": 0,
    }
    cand_pairs = [(mtp_n, topo) for mtp_n in mtp_grid for topo in topos]
    eff_workers = resolve_granted_workers(
        granted_workers,
        task_count=(None if dep_mode == "pd_decode" else max(len(cand_pairs), 1)),
    )

    with BoundedMapPool(granted_workers=eff_workers, max_tasks=eff_workers) as pool:
        rows, viable = _collect_admitted_single_pool_candidates(
            pool, compose_for=compose_for, cand_pairs=cand_pairs, eff_workers=eff_workers,
            dep_mode=dep_mode, num_devices=num_devices, prompt=prompt, output=output, tpot=tpot,
            max_sequences_per_dp=max_sequences_per_dp, rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula, coverage=coverage,
        )
        if not viable:
            return rows, [], coverage

        viable.sort(
            key=lambda c: (int(c["used_dies"] == num_devices), -int(c["pp"]), c["ub_tps"], c["b_vram"]),
            reverse=True,
        )
        best_entry: tuple[dict[str, Any], dict[str, Any]] | None = None
        incumbent_tps = 0.0
        incumbent_dies = num_devices + 1

        def _is_prunable(desc: Mapping[str, Any]) -> bool:
            return bool(
                best_entry is not None
                and desc["used_dies"] >= incumbent_dies
                and desc["ub_tps"] < incumbent_tps - PARETO_EPS
            )

        wave_workers = min(eff_workers, len(viable))
        cursor = 0
        while cursor < len(viable):
            while cursor < len(viable) and _is_prunable(viable[cursor]):
                viable[cursor]["raw_case"] = None
                coverage["pruned_by_upper_bound"] += 1
                cursor += 1
            if cursor >= len(viable):
                break
            wave_items: list[tuple[dict[str, Any], dict[str, Any]]] = []
            wave_tasks: list[tuple[Any, ...]] = []
            is_anchor_wave = (
                best_entry is None
                and viable[cursor]["used_dies"] == num_devices
                and int(viable[cursor]["pp"]) == 1
            )
            anchor_ub_floor = viable[cursor]["ub_tps"] * _ANCHOR_WAVE_UB_RATIO if is_anchor_wave else 0.0
            while cursor < len(viable) and len(wave_items) < wave_workers:
                desc = viable[cursor]
                if is_anchor_wave and wave_items and (
                    desc["used_dies"] != num_devices
                    or int(desc["pp"]) != 1
                    or desc["ub_tps"] < anchor_ub_floor
                ):
                    break
                cursor += 1
                if _is_prunable(desc):
                    desc["raw_case"] = None
                    coverage["pruned_by_upper_bound"] += 1
                    continue
                raw_c = desc["raw_case"]
                desc["raw_case"] = None
                eval_case = (
                    _sync_derived_runtime_limits(
                        raw_c, max_num_seqs=desc["b_limit"], mtp_n=desc["mtp_n"], role=dep_mode,
                    )
                    if (max_sequences_per_dp is None and not (dep_mode == "colocated" and output <= 1))
                    else raw_c
                )
                wave_items.append((desc, eval_case))
                wave_tasks.append((
                    eval_case, prompt, output, ttft, tpot, desc["mtp_n"],
                    max_sequences_per_dp, apply_target_envelope, rewrite_plan,
                    allow_unbound_formula, dep_mode, num_devices,
                ))
            if dep_mode == "pd_decode" and eff_workers > 1 and wave_items:
                wave_results = _eval_pd_decode_wave_parallel(
                    pool, wave_items, eff_workers=eff_workers, prompt=prompt, output=output,
                    tpot=tpot, max_sequences_per_dp=max_sequences_per_dp,
                    apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
                    allow_unbound_formula=allow_unbound_formula, num_devices=num_devices,
                )
            else:
                wave_results = pool.map_ordered(_eval_single_pool_worker, wave_tasks)
            for (desc, eval_case), (ok, payload) in zip(wave_items, wave_results):
                if _is_prunable(desc):
                    coverage["pruned_by_upper_bound"] += 1
                    continue
                if not ok:
                    exc = _rebuild_worker_error(payload)
                    if rewrite_plan is not None and _is_rewrite_plan_error(exc):
                        raise exc
                    continue
                row = payload
                coverage["exact_evaluated"] += 1
                rows.append(row)
                if row["feasible"]:
                    if best_entry is None or _rank_single_pool_row(row) > _rank_single_pool_row(best_entry[0]):
                        best_entry = (row, eval_case)
                        incumbent_tps = _single_die_decode_tps(row)
                        incumbent_dies = int(row["used_dies"])
    return rows, ([best_entry] if best_entry is not None else []), coverage


def _search_pd_prefill_deployments(
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
    apply_target_envelope: bool = False,
    rewrite_plan: Any = None,
    allow_unbound_formula: bool = True,
    granted_workers: int | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any]]], dict[str, int]]:
    """Full-grid standalone ``pd_prefill`` MapReduce search over the full device budget."""
    topos = _build_pd_stage_candidates(
        role="pd_prefill", max_dies=num_devices, is_moe=is_moe,
        layers=layers, prompt_tokens=prompt, rec_topo=rec_topo,
        pinned=pinned, model_spec=model_spec,
    )
    coverage = {
        "total_candidates": 0, "static_admitted": 0, "static_rejected": 0,
        "pruned_by_upper_bound": 0, "exact_evaluated": 0,
    }
    rows: list[dict[str, Any]] = []
    viable: list[dict[str, Any]] = []
    for mtp_n in mtp_grid:
        for topo in topos:
            raw_p = compose_for(topo, mtp_n, "pd_prefill")
            if raw_p is None:
                continue
            coverage["total_candidates"] += 1
            try:
                parsed = parse_case(raw_p)
            except UnmodeledCapabilityError:
                raise
            except (ModelingError, ValueError) as exc:
                if rewrite_plan is not None and _is_rewrite_plan_error(exc):
                    raise
                coverage["static_rejected"] += 1
                continue
            par = parsed.runtime.parallel
            prefill_ws = _prefill_workspace_tokens(parsed, prompt)
            p_mem = Memory(parsed, workspace_tokens=prefill_ws)
            p_kv = p_mem.cache_bytes(prompt + 1, padded=True)
            hbm_ok = bool(
                prompt <= parsed.runtime.max_model_len
                and (parsed.runtime.enable_chunked_prefill or prompt <= parsed.runtime.max_num_batched_tokens)
                and all(cap >= 0.0 and cap >= kv for row in p_mem.capacity for cap, kv in zip(row, p_kv))
            )
            if not hbm_ok:
                coverage["static_rejected"] += 1
                rows.append(_colocated_infeasible_row(
                    par, used_dies=par.world_size, mtp_tokens=mtp_n, reason="prefill_hbm_oom",
                    regime=REGIME_HBM_CAPACITY_BOUND, deployment_mode="pd_prefill",
                    configured_max_num_seqs=int(parsed.runtime.max_num_seqs),
                    limiting_constraint="prefill_hbm_oom",
                ))
                continue
            coverage["static_admitted"] += 1
            viable.append({"mtp_n": mtp_n, "used_dies": par.world_size, "pp": int(par.pp), "raw_case": raw_p})
    if not viable:
        return rows, [], coverage
    viable.sort(key=lambda c: (int(c["used_dies"] == num_devices), int(c["pp"] == 1), c["used_dies"]), reverse=True)
    raw_cases = [desc["raw_case"] for desc in viable]
    for desc in viable:
        desc["raw_case"] = None
    tasks = [
        (
            raw_p, prompt, output, ttft, desc["mtp_n"],
            apply_target_envelope, rewrite_plan, allow_unbound_formula, num_devices,
        )
        for desc, raw_p in zip(viable, raw_cases)
    ]
    with BoundedMapPool(granted_workers=granted_workers, max_tasks=len(tasks)) as pool:
        results = pool.map_ordered(_eval_pd_prefill_worker, tasks)
    best_entry: tuple[dict[str, Any], dict[str, Any]] | None = None
    for raw_p, (ok, payload) in zip(raw_cases, results):
        if not ok:
            exc = _rebuild_worker_error(payload)
            if rewrite_plan is not None and _is_rewrite_plan_error(exc):
                raise exc
            continue
        row = payload
        coverage["exact_evaluated"] += 1
        rows.append(row)
        if row["feasible"]:
            if best_entry is None or _rank_pd_prefill_row(row) > _rank_pd_prefill_row(best_entry[0]):
                best_entry = (row, raw_p)
    return rows, ([best_entry] if best_entry is not None else []), coverage


def _pd_static_infeasible_row(
    *,
    num_devices: int,
    reason: str,
    p_par: Mapping[str, Any] | None = None,
    d_par: Mapping[str, Any] | None = None,
    mtp_n: int = 0,
    pd_mode: str = "layerwise_overlap_d2d",
) -> dict[str, Any]:
    return {
        "deployment_mode": "pd_disaggregated",
        "pd_transfer_mode": pd_mode,
        "mtp_proposal_tokens": int(mtp_n),
        "prefill_parallel": dict(p_par or {}),
        "decode_parallel": dict(d_par or {}),
        "pd_ratio": None,
        "prefill_instances": 0, "decode_instances": 0,
        "prefill_used_dies": 0, "decode_used_dies": 0,
        "used_dies": int(num_devices),
        "b_feasible": 0, "b_vram_max": 0, "configured_max_num_seqs": 0,
        "ttft_ms": 0.0, "decode_step_ms": 0.0, "handoff_tpot_ms": 0.0,
        "effective_tpot_ms": 0.0, "avg_accepted_tokens": 1.0,
        "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
        "unused_sla_slack_ms": 0.0, "operating_regime": REGIME_HBM_CAPACITY_BOUND,
        "limiting_constraint": reason, "free_hbm_headroom_gib": 0.0,
        "kv_transfer_ms_per_request": 0.0, "sustainable_rps": 0.0,
        "prefill_kv_rps": 0.0, "sustainable_tps": 0.0,
        "sustainable_tps_per_die": 0.0, "decode_tps_per_die": 0.0,
        "e2e_total_tps": 0.0, "e2e_tps_per_die": 0.0,
        "dcp_codomain_valid": True, "dcp_codomain_reason": None,
        "feasible": False, "infeasible_reason": reason,
    }


def _static_admit_pd_stages(
    compose_for: Any,
    *,
    p_topos: Sequence[Mapping[str, int]],
    d_topos: Sequence[Mapping[str, int]],
    mtp_grid: tuple[int, ...],
    p_mtp_by_d_mtp: Mapping[int, int],
    num_devices: int,
    prompt: int,
    output: int,
    tpot: float,
    compute_tokens: int,
    max_sequences_per_dp: int | None,
    rewrite_plan: Any,
    allow_unbound_formula: bool,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    list[tuple[int, list[dict[str, Any]]]],
    dict[str, int],
]:
    """Stream static HBM admission across all P and D candidates with ZERO Engine calls."""
    coverage = {
        "total_candidates": 0, "static_admitted": 0, "static_rejected": 0,
        "pruned_by_upper_bound": 0, "exact_evaluated": 0,
    }
    p_by_pmtp: dict[int, list[dict[str, Any]]] = {}
    p_id = 0
    for p_mtp in tuple(dict.fromkeys(p_mtp_by_d_mtp.values())):
        admitted_p: list[dict[str, Any]] = []
        for topo in p_topos:
            raw_p = compose_for(topo, p_mtp, "pd_prefill")
            if raw_p is None:
                continue
            coverage["total_candidates"] += 1
            try:
                parsed_p = parse_case(raw_p)
            except UnmodeledCapabilityError:
                raise
            except (ModelingError, ValueError) as exc:
                if rewrite_plan is not None and _is_rewrite_plan_error(exc):
                    raise
                coverage["static_rejected"] += 1
                continue
            par_p = parsed_p.runtime.parallel
            if not validate_vllm_ascend_dcp_codomain(parsed_p.model, par_p)[0]:
                coverage["static_rejected"] += 1
                continue
            p_mem = Memory(parsed_p, workspace_tokens=_prefill_workspace_tokens(parsed_p, compute_tokens))
            p_kv = p_mem.cache_bytes(prompt + 1, padded=True)
            hbm_ok = bool(
                prompt <= parsed_p.runtime.max_model_len
                and (parsed_p.runtime.enable_chunked_prefill or compute_tokens <= parsed_p.runtime.max_num_batched_tokens)
                and all(cap >= 0.0 and cap >= kv for row in p_mem.capacity for cap, kv in zip(row, p_kv))
            )
            if not hbm_ok:
                coverage["static_rejected"] += 1
                continue
            p_id += 1
            coverage["static_admitted"] += 1
            admitted_p.append({"id": p_id, "world": int(par_p.world_size), "p_mtp": p_mtp, "raw": raw_p})
            if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE and coverage["total_candidates"] % STATIC_ADMIT_RECLAIM_STRIDE == 0:
                _reclaim_sweep_memory()
        p_by_pmtp[p_mtp] = admitted_p

    d_by_mtp: list[tuple[int, list[dict[str, Any]]]] = []
    d_id = 0
    for mtp_n in mtp_grid:
        admitted_d: list[dict[str, Any]] = []
        for topo in d_topos:
            raw_d = compose_for(topo, mtp_n, "pd_decode")
            if raw_d is None:
                continue
            coverage["total_candidates"] += 1
            desc, _ = _static_admit_single_pool_candidate(
                raw_d, topo=topo, mtp_n=mtp_n, dep_mode="pd_decode",
                prompt=prompt, output=output, tpot=tpot,
                max_sequences_per_dp=max_sequences_per_dp,
                rewrite_plan=rewrite_plan, allow_unbound_formula=allow_unbound_formula,
            )
            if desc is None:
                coverage["static_rejected"] += 1
                continue
            d_id += 1
            coverage["static_admitted"] += 1
            eval_d = (
                _sync_derived_runtime_limits(
                    desc["raw_case"], max_num_seqs=desc["b_limit"], mtp_n=mtp_n, role="pd_decode",
                )
                if max_sequences_per_dp is None else desc["raw_case"]
            )
            admitted_d.append({
                "id": d_id, "world": int(desc["used_dies"]), "mtp_n": mtp_n,
                "b_vram": int(desc["b_vram"]), "ub_tps": float(desc["ub_tps"]), "raw": eval_d,
            })
            if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE and coverage["total_candidates"] % STATIC_ADMIT_RECLAIM_STRIDE == 0:
                _reclaim_sweep_memory()
        d_by_mtp.append((mtp_n, admitted_d))
    return p_by_pmtp, d_by_mtp, coverage


def _eval_pd_stage_worker(task: tuple[Any, ...]) -> tuple[bool, Any]:
    """Top-level pickle-safe Map worker evaluating one Prefill or Decode PD stage."""
    kind = task[0]
    num_devices = int(task[-1])
    try:
        if kind == "prefill":
            (
                _, p_idx, compute_tokens, hit_tokens, p_raw, prompt, output,
                apply_target_envelope, rewrite_plan, allow_unbound_formula, _,
            ) = task
            p_parsed = parse_case(p_raw)
            _, prefill_hbm_ok, p_slots, p_step = _measure_prefill_stage(
                p_parsed, prompt=prompt, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
                cache=None, cache_id=None, apply_target_envelope=apply_target_envelope,
                rewrite_plan=rewrite_plan, allow_unbound_formula=allow_unbound_formula,
            )
            return True, (
                "prefill",
                (p_idx, compute_tokens, hit_tokens),
                {"prefill_hbm_ok": bool(prefill_hbm_ok), "p_slots": int(p_slots), "p_step": float(p_step)},
            )
        (
            _, d_idx, d_raw, prompt, output, tpot, max_sequences_per_dp,
            apply_target_envelope, rewrite_plan, allow_unbound_formula,
            explicit_eff_sla_ms, seed_probe_cache, _,
        ) = task
        d_parsed = parse_case(d_raw)
        dcp_ok, _ = validate_vllm_ascend_dcp_codomain(d_parsed.model, d_parsed.runtime.parallel)
        d_q_tok, d_emit_est = _resolve_speculative_tokens(
            d_parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
        )
        length_ok = bool(prompt + output <= d_parsed.runtime.max_model_len)
        vram_info = _solve_b_vram_max(
            d_parsed, base_ws=max(1, int(d_q_tok)), decode_q=max(1, int(d_q_tok)),
            seq_len=prompt + output, cap_bound=None,
        )
        key = (d_idx, explicit_eff_sla_ms)
        if not dcp_ok or not length_ok or int(vram_info[2]) < 1:
            return True, ("decode", key, {"static_skip": True, "d_sla": None, "probe_cache": {}})
        eff_sla_ms = (
            float(explicit_eff_sla_ms)
            if explicit_eff_sla_ms is not None
            else (float(tpot) * max(1.0, float(d_emit_est)))
        )
        probe_sink: dict[int, float] = dict(seed_probe_cache or {})
        d_sla = _solve_sla_archetype_aware(
            d_parsed, sla_ms=eff_sla_ms, context=prompt, output_budget=output,
            max_sequences_per_dp=max_sequences_per_dp,
            apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
            allow_unbound_formula=allow_unbound_formula,
            probe_cache=probe_sink, probe_sink=probe_sink,
            vram_info=vram_info, rep_info=discover_layer_cohorts(d_parsed), measure_b1=False,
        )
        return True, (
            "decode",
            key,
            {"static_skip": False, "eff_sla_ms": eff_sla_ms, "d_sla": d_sla, "probe_cache": probe_sink},
        )
    except UnmodeledCapabilityError:
        raise
    except (ModelingError, ValueError) as exc:
        err_key = (task[1], task[2], task[3]) if kind == "prefill" else (task[1], task[10])
        return False, (kind, err_key, isinstance(exc, ModelingError), str(exc))
    finally:
        if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE:
            _reclaim_sweep_memory()


def _resolve_decode_sla_from_cache(
    d_parsed: CaseSpec,
    *,
    d_sla_zero: Mapping[str, Any],
    probe_cache: Mapping[int, float],
    vram_info: tuple[Memory, tuple[float, ...], int],
    eff_sla_ms: float,
    zero_sla_ms: float,
    d_q_tok: int,
    d_emit_est: float,
    max_sequences_per_dp: int | None,
) -> dict[str, Any] | None:
    """Resolve exact decode SLA concurrency in O(1) from Phase-1 baseline + probe cache."""
    if eff_sla_ms >= zero_sla_ms - PARETO_EPS:
        return dict(d_sla_zero)
    scen0 = d_sla_zero["scenarios"][0]
    b_zero = int(scen0["b_feasible"])
    t_zero = scen0.get("step_ms_feasible")
    if b_zero < 1:
        return dict(d_sla_zero)
    if t_zero is not None and float(t_zero) <= eff_sla_ms + PARETO_EPS:
        return dict(d_sla_zero)
    d_mem, d_kv, b_vram = vram_info
    base_ws = max(1, int(d_q_tok))
    limit, cfg_seqs, tok_lim = _resolve_decode_batch_bounds(
        d_parsed, q_tok=d_q_tok, b_vram=b_vram, max_sequences_per_dp=max_sequences_per_dp,
    )
    lo, hi, best_b, best_t = _tighten_sla_bracket(probe_cache, limit, eff_sla_ms)
    if lo == 0 and 1 in probe_cache and float(probe_cache[1]) > eff_sla_ms + PARETO_EPS:
        return {
            "emitted_tokens_per_seq": float(d_emit_est),
            "scenarios": [{
                "b_vram_max": b_vram, "configured_max_num_seqs": cfg_seqs,
                "token_batch_limit": tok_lim, "b_sla": 0, "b_feasible": 0,
                "step_ms_at_b1": float(probe_cache[1]), "step_ms_feasible": None,
                "free_hbm_headroom_gib": _headroom_gib_at_batch(
                    d_parsed, d_mem, d_kv, b_vram=b_vram, batch=0,
                    base_ws=base_ws, decode_q=base_ws,
                ),
            }],
        }
    if lo >= 1 and hi == lo + 1 and best_t is not None:
        t1 = float(probe_cache.get(1) or (best_t if best_b == 1 else 0.0))
        return {
            "emitted_tokens_per_seq": float(d_emit_est),
            "scenarios": [{
                "b_vram_max": b_vram, "configured_max_num_seqs": cfg_seqs,
                "token_batch_limit": tok_lim, "b_sla": best_b, "b_feasible": best_b,
                "step_ms_at_b1": t1, "step_ms_feasible": float(best_t),
                "free_hbm_headroom_gib": _headroom_gib_at_batch(
                    d_parsed, d_mem, d_kv, b_vram=b_vram, batch=best_b,
                    base_ws=base_ws, decode_q=base_ws,
                ),
            }],
        }
    return None


def _build_pd_candidate_row_from_stage(
    st: Mapping[str, Any],
    *,
    p_parsed: CaseSpec,
    d_parsed: CaseSpec,
    best_ratio: Mapping[str, Any] | None,
    pd_mode: str,
    mtp_n: int,
    p_world: int,
    d_world: int,
) -> dict[str, Any]:
    """Assemble one PD candidate row directly from factorized stage metrics + closed-form join."""
    d_us = float(st["us_per_tok_die"])
    e2e_tps = float(best_ratio["sustainable_tps"]) if best_ratio else 0.0
    e2e_per_die = float(best_ratio["e2e_tps_per_die"]) if best_ratio else 0.0
    d_tps_per_die = (US_PER_SECOND / d_us) if (best_ratio is not None and d_us > 0.0) else 0.0
    return {
        "deployment_mode": "pd_disaggregated",
        "pd_transfer_mode": pd_mode,
        "mtp_proposal_tokens": mtp_n,
        "prefill_parallel": p_parsed.runtime.parallel.to_dict(),
        "decode_parallel": d_parsed.runtime.parallel.to_dict(),
        "pd_ratio": best_ratio["pd_ratio"] if best_ratio else None,
        "prefill_instances": int(best_ratio["n_prefill"]) if best_ratio else 1,
        "decode_instances": int(best_ratio["n_decode"]) if best_ratio else 1,
        "prefill_used_dies": int(best_ratio["prefill_dies"]) if best_ratio else p_world,
        "decode_used_dies": int(best_ratio["decode_dies"]) if best_ratio else d_world,
        "used_dies": int(best_ratio["used_dies"]) if best_ratio else (p_world + d_world),
        "b_feasible": int(st["b_feasible"]),
        "b_vram_max": int(st["b_vram"]),
        "configured_max_num_seqs": int(st.get("configured_max_num_seqs", d_parsed.runtime.max_num_seqs)),
        "ttft_ms": float(st.get("effective_ttft_ms", st["p_step"])),
        "decode_step_ms": float(st["decode_step"]),
        "handoff_tpot_ms": float(st.get("handoff_tpot_ms", 0.0) or 0.0),
        "effective_tpot_ms": float(st["effective_tpot"]),
        "avg_accepted_tokens": float(st["avg_accepted_tokens"]),
        "n_tok_per_die": float(st["n_tok_per_die"]),
        "us_per_tok_die": d_us,
        "unused_sla_slack_ms": float(st["unused_sla_slack_ms"]),
        "operating_regime": str(st["operating_regime"]),
        "limiting_constraint": str(st.get("limiting_constraint", "")),
        "free_hbm_headroom_gib": float(st["free_hbm_headroom_gib"]),
        "kv_transfer_ms_per_request": float(st["kv_ms"]),
        "sustainable_rps": float(best_ratio["sustainable_rps"]) if best_ratio else 0.0,
        "prefill_kv_rps": (
            min(float(best_ratio["prefill_rps"]), float(best_ratio["kv_transfer_rps"])) if best_ratio else 0.0
        ),
        "sustainable_tps": e2e_tps,
        "sustainable_tps_per_die": d_tps_per_die if d_tps_per_die > 0.0 else e2e_per_die,
        "decode_tps_per_die": d_tps_per_die,
        "e2e_total_tps": e2e_tps,
        "e2e_tps_per_die": e2e_per_die,
        "dcp_codomain_valid": bool(st.get("dcp_codomain_valid", True)),
        "dcp_codomain_reason": st.get("dcp_codomain_reason"),
        "feasible": bool(best_ratio is not None),
        "infeasible_reason": None if best_ratio is not None else st["infeasible"],
    }


def _resolve_pd_mode_params(
    pd_modes: tuple[str, ...],
    *,
    prompt: int,
    pool_dram_ratio: float | None,
    pool_ssd_ratio: float,
    pool_prefix_hit_ratio: float,
    ssd_bw_gbps: float | None,
) -> dict[str, tuple[float, float, float, int, int]]:
    """Validate per-mode KV pool ratios and derive ``(dram_r, ssd_r, hit_r, hit_tok, comp_tok)``."""
    resolved: dict[str, tuple[float, float, float, int, int]] = {}
    for pd_mode in pd_modes:
        try:
            is_pool = pd_mode == "pooled_dram_ssd"
            ssd_r = float(pool_ssd_ratio) if is_pool else 0.0
            hit_r = float(pool_prefix_hit_ratio) if is_pool else 0.0
            if not (0.0 <= ssd_r <= 1.0) or not (0.0 <= hit_r <= 1.0):
                raise ModelingError(f"invalid_pool_ratios:ssd={ssd_r},prefix_hit={hit_r}")
            dram_r = (1.0 - ssd_r) if (not is_pool or pool_dram_ratio is None) else float(pool_dram_ratio)
            if not (0.0 <= dram_r <= 1.0) or abs((dram_r + ssd_r) - 1.0) > POOL_RATIO_SUM_TOL:
                raise ModelingError(f"invalid_pool_dram_ssd_ratios:dram={dram_r},ssd={ssd_r}")
            if ssd_r > 0.0 and ssd_bw_gbps is None:
                raise missing_scenario_error(("ssd_bw_gbps",))
            hit_tokens = min(int(math.floor(prompt * hit_r)), max(0, prompt - 1)) if is_pool else 0
            compute_tokens = max(0, prompt - hit_tokens)
            resolved[pd_mode] = (dram_r, ssd_r, hit_r, hit_tokens, compute_tokens)
        except UnmodeledCapabilityError:
            raise
        except (ModelingError, ValueError) as exc:
            if _is_ingress_contradiction(exc):
                raise
    return resolved


def _build_pd_factorized_handoff_records(
    pair_specs: Sequence[tuple[int, dict[str, Any], dict[str, Any], str]],
    *,
    mode_params: Mapping[str, tuple[float, float, float, int, int]],
    p_parsed: Mapping[int, CaseSpec],
    d_parsed: Mapping[int, CaseSpec],
    p_results: Mapping[tuple[int, int, int], Mapping[str, Any]],
    p_errors: Mapping[tuple[int, int, int], tuple[bool, str]],
    d_base_results: Mapping[int, Mapping[str, Any]],
    d_base_errors: Mapping[int, tuple[bool, str]],
    d_vram_caches: Mapping[int, tuple[Memory, tuple[float, ...], int]],
    d_spec_tokens: Mapping[int, tuple[int, float]],
    num_devices: int,
    prompt: int,
    output: int,
    ttft: float,
    tpot: float,
    kv_transfer_bw_gbps: float | None,
    ddr_bw_gbps: float | None,
    ssd_bw_gbps: float | None,
    max_sequences_per_dp: int | None,
    rewrite_plan: Any,
) -> tuple[list[dict[str, Any]], list[tuple[int, float]]]:
    """Phase 2: Pure-arithmetic KV handoff & O(1) decode bracket resolution across all pairs."""
    p_mem_cache: dict[tuple[int, int], Memory] = {}
    p_dcp_cache: dict[int, tuple[bool, str | None]] = {}
    d_mem_cache: dict[int, Memory] = {}
    records: list[dict[str, Any]] = []
    refine_seen: set[tuple[int, float]] = set()
    refine_keys: list[tuple[int, float]] = []

    for mtp_n, p_item, d_item, pd_mode in pair_specs:
        p_idx, d_idx = int(p_item["id"]), int(d_item["id"])
        dram_r, ssd_r, _, hit_tokens, compute_tokens = mode_params[pd_mode]
        p_key = (p_idx, compute_tokens, hit_tokens)
        if p_key in p_errors:
            exc = _rebuild_worker_error(p_errors[p_key])
            if _is_ingress_contradiction(exc) or (rewrite_plan is not None and _is_rewrite_plan_error(exc)):
                raise exc
            continue
        if output > 1 and d_idx in d_base_errors:
            exc = _rebuild_worker_error(d_base_errors[d_idx])
            if _is_ingress_contradiction(exc) or (rewrite_plan is not None and _is_rewrite_plan_error(exc)):
                raise exc
            continue
        parsed_p, parsed_d = p_parsed[p_idx], d_parsed[d_idx]
        p_res = p_results[p_key]
        if (p_idx, compute_tokens) not in p_mem_cache:
            p_mem_cache[(p_idx, compute_tokens)] = Memory(
                parsed_p, workspace_tokens=_prefill_workspace_tokens(parsed_p, compute_tokens),
            )
        if p_idx not in p_dcp_cache:
            p_dcp_cache[p_idx] = validate_vllm_ascend_dcp_codomain(parsed_p.model, parsed_p.runtime.parallel)
        if d_idx not in d_mem_cache:
            d_mem_cache[d_idx] = Memory(parsed_d, workspace_tokens=d_spec_tokens[d_idx][0])
        p_dcp_ok, p_dcp_reason = p_dcp_cache[p_idx]
        try:
            handoff_info = _compute_pd_handoff_metrics(
                parsed_p, parsed_d,
                p_mem=p_mem_cache[(p_idx, compute_tokens)], d_mem=d_mem_cache[d_idx],
                p_step=float(p_res["p_step"]), p_slots=int(p_res["p_slots"]),
                prefill_hbm_ok=bool(p_res["prefill_hbm_ok"]),
                p_dcp_ok=p_dcp_ok, p_dcp_reason=p_dcp_reason,
                total=num_devices, prompt=prompt, output=output,
                ttft_sla_ms=ttft, tpot_sla_ms=tpot, mode=pd_mode,
                dram_r=dram_r, ssd_r=ssd_r, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
                kv_transfer_bw_gbps=kv_transfer_bw_gbps, ddr_bw_gbps=ddr_bw_gbps, ssd_bw_gbps=ssd_bw_gbps,
            )
        except UnmodeledCapabilityError:
            raise
        except (ModelingError, ValueError) as exc:
            if _is_ingress_contradiction(exc) or (rewrite_plan is not None and _is_rewrite_plan_error(exc)):
                raise
            continue
        resolved_sla: dict[str, Any] | None = None
        refine_key: tuple[int, float] | None = None
        if not handoff_info["prefill_only"]:
            d_base = d_base_results[d_idx]
            if not d_base["static_skip"]:
                d_q_tok, d_emit_est = d_spec_tokens[d_idx]
                zero_sla_ms = float(d_base["eff_sla_ms"])
                decode_tokens = int(handoff_info["decode_tokens"])
                decode_budget = float(tpot) - (float(handoff_info["exposed_kv_ms"]) / float(decode_tokens))
                step_budget = decode_budget * max(1.0, float(d_emit_est))
                eff_sla_ms = max(step_budget, PARETO_EPS) if step_budget > 0.0 else zero_sla_ms
                resolved_sla = _resolve_decode_sla_from_cache(
                    parsed_d, d_sla_zero=d_base["d_sla"], probe_cache=d_base["probe_cache"],
                    vram_info=d_vram_caches[d_idx], eff_sla_ms=eff_sla_ms, zero_sla_ms=zero_sla_ms,
                    d_q_tok=d_q_tok, d_emit_est=d_emit_est, max_sequences_per_dp=max_sequences_per_dp,
                )
                if resolved_sla is None:
                    refine_key = (d_idx, eff_sla_ms)
                    if refine_key not in refine_seen:
                        refine_seen.add(refine_key)
                        refine_keys.append(refine_key)
        records.append({
            "mtp_n": mtp_n, "p_item": p_item, "d_item": d_item, "pd_mode": pd_mode,
            "handoff_info": handoff_info, "resolved_sla": resolved_sla, "refine_key": refine_key,
        })
    return records, refine_keys


def _execute_pd_stage_mapreduce(
    *,
    pair_specs: Sequence[tuple[int, dict[str, Any], dict[str, Any], str]],
    p_task_items: Mapping[tuple[int, int, int], dict[str, Any]],
    d_active: Mapping[int, dict[str, Any]],
    mode_params: Mapping[str, tuple[float, float, float, int, int]],
    num_devices: int,
    prompt: int,
    output: int,
    ttft: float,
    tpot: float,
    kv_transfer_bw_gbps: float | None,
    ddr_bw_gbps: float | None,
    ssd_bw_gbps: float | None,
    max_sequences_per_dp: int | None,
    apply_target_envelope: bool,
    rewrite_plan: Any,
    allow_unbound_formula: bool,
    granted_workers: int | None,
    coverage: dict[str, int],
) -> tuple[
    dict[int, CaseSpec],
    dict[int, CaseSpec],
    dict[int, tuple[Memory, tuple[float, ...], int]],
    list[dict[str, Any]],
    dict[tuple[int, float], dict[str, Any]],
    dict[tuple[int, float], tuple[bool, str]],
]:
    """Run Phases 1-3 of PD Stage Factorization in ONE flat bounded worker pool."""
    stage_tasks: list[tuple[Any, ...]] = [
        (
            "prefill", p_idx, comp_tok, hit_tok, p_item["raw"], prompt, output,
            apply_target_envelope, rewrite_plan, allow_unbound_formula, num_devices,
        )
        for (p_idx, comp_tok, hit_tok), p_item in p_task_items.items()
    ]
    if output > 1:
        stage_tasks.extend(
            (
                "decode", d_idx, d_item["raw"], prompt, output, tpot, max_sequences_per_dp,
                apply_target_envelope, rewrite_plan, allow_unbound_formula, None, None, num_devices,
            )
            for d_idx, d_item in d_active.items()
        )
    coverage["stage_p_evaluated"] = len(p_task_items)
    coverage["stage_d_evaluated"] = len(d_active) if output > 1 else 0
    coverage["stage_d_refinements"] = 0

    p_results: dict[tuple[int, int, int], dict[str, Any]] = {}
    p_errors: dict[tuple[int, int, int], tuple[bool, str]] = {}
    d_base_results: dict[int, dict[str, Any]] = {}
    d_base_errors: dict[int, tuple[bool, str]] = {}
    refine_results: dict[tuple[int, float], dict[str, Any]] = {}
    refine_errors: dict[tuple[int, float], tuple[bool, str]] = {}

    with BoundedMapPool(granted_workers=granted_workers, max_tasks=len(stage_tasks)) as pool:
        for ok, payload in pool.map_ordered(_eval_pd_stage_worker, stage_tasks):
            if ok:
                kind, key, data = payload
                if kind == "prefill":
                    p_results[key] = data
                else:
                    d_base_results[int(key[0])] = data
            else:
                kind, err_key, is_mod_err, err_msg = payload
                if kind == "prefill":
                    p_errors[err_key] = (is_mod_err, err_msg)
                else:
                    d_base_errors[int(err_key[0])] = (is_mod_err, err_msg)

        p_parsed = {p_idx: parse_case(p_item["raw"]) for (p_idx, _, _), p_item in p_task_items.items()}
        d_parsed: dict[int, CaseSpec] = {}
        d_vram_caches: dict[int, tuple[Memory, tuple[float, ...], int]] = {}
        d_spec_tokens: dict[int, tuple[int, float]] = {}
        for d_idx, d_item in d_active.items():
            parsed_d = parse_case(d_item["raw"])
            d_parsed[d_idx] = parsed_d
            d_q_tok, d_emit_est = _resolve_speculative_tokens(
                parsed_d, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
            )
            d_spec_tokens[d_idx] = (max(1, int(d_q_tok)), float(d_emit_est))
            d_vram_caches[d_idx] = _solve_b_vram_max(
                parsed_d, base_ws=max(1, int(d_q_tok)), decode_q=max(1, int(d_q_tok)),
                seq_len=prompt + output, cap_bound=None,
            )

        records, refine_keys = _build_pd_factorized_handoff_records(
            pair_specs, mode_params=mode_params, p_parsed=p_parsed, d_parsed=d_parsed,
            p_results=p_results, p_errors=p_errors,
            d_base_results=d_base_results, d_base_errors=d_base_errors,
            d_vram_caches=d_vram_caches, d_spec_tokens=d_spec_tokens,
            num_devices=num_devices, prompt=prompt, output=output, ttft=ttft, tpot=tpot,
            kv_transfer_bw_gbps=kv_transfer_bw_gbps, ddr_bw_gbps=ddr_bw_gbps,
            ssd_bw_gbps=ssd_bw_gbps, max_sequences_per_dp=max_sequences_per_dp,
            rewrite_plan=rewrite_plan,
        )
        if refine_keys:
            coverage["stage_d_refinements"] = len(refine_keys)
            refine_tasks = [
                (
                    "decode", d_idx, d_active[d_idx]["raw"], prompt, output, tpot,
                    max_sequences_per_dp, apply_target_envelope, rewrite_plan,
                    allow_unbound_formula, eff_sla_ms,
                    d_base_results[d_idx]["probe_cache"], num_devices,
                )
                for d_idx, eff_sla_ms in refine_keys
            ]
            for ok, payload in pool.map_ordered(_eval_pd_stage_worker, refine_tasks):
                if ok:
                    _, r_key, data = payload
                    refine_results[r_key] = data
                else:
                    _, r_key, is_mod_err, err_msg = payload
                    refine_errors[r_key] = (is_mod_err, err_msg)

    return p_parsed, d_parsed, d_vram_caches, records, refine_results, refine_errors


def _reduce_pd_factorized_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    p_parsed: Mapping[int, CaseSpec],
    d_parsed: Mapping[int, CaseSpec],
    d_vram_caches: Mapping[int, tuple[Memory, tuple[float, ...], int]],
    refine_results: Mapping[tuple[int, float], Mapping[str, Any]],
    refine_errors: Mapping[tuple[int, float], tuple[bool, str]],
    num_devices: int,
    prompt: int,
    output: int,
    tpot: float,
    max_sequences_per_dp: int | None,
    rewrite_plan: Any,
    coverage: dict[str, int],
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]]:
    """Phase 4: Deterministic closed-form Join/Reduce across all factorized PD pairs."""
    rows: list[dict[str, Any]] = []
    best_entry: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    for rec in records:
        p_item, d_item = rec["p_item"], rec["d_item"]
        p_idx, d_idx = int(p_item["id"]), int(d_item["id"])
        p_world, d_world = int(p_item["world"]), int(d_item["world"])
        parsed_p, parsed_d = p_parsed[p_idx], d_parsed[d_idx]
        handoff_info = rec["handoff_info"]
        if handoff_info["prefill_only"]:
            st = handoff_info["stage"]
        else:
            d_sla = rec["resolved_sla"]
            if d_sla is None and rec["refine_key"] is not None:
                r_key = rec["refine_key"]
                if r_key in refine_errors:
                    exc = _rebuild_worker_error(refine_errors[r_key])
                    if _is_ingress_contradiction(exc) or (rewrite_plan is not None and _is_rewrite_plan_error(exc)):
                        raise exc
                    continue
                d_sla = refine_results[r_key]["d_sla"]
            if d_sla is None:
                dec = _measure_decode_stage(
                    parsed_d, prompt=prompt, output=output, eff_decode=prompt,
                    tpot_sla_ms=tpot, exposed_kv_ms=float(handoff_info["exposed_kv_ms"]),
                    decode_tokens=int(handoff_info["decode_tokens"]),
                    max_sequences_per_dp=max_sequences_per_dp, cache=None, cache_id=None,
                    vram_info=d_vram_caches[d_idx],
                )
            else:
                dec = _assemble_decode_stage_from_sla(
                    parsed_d, d_sla=d_sla, length_ok=True, tpot_sla_ms=tpot,
                    exposed_kv_ms=float(handoff_info["exposed_kv_ms"]),
                    decode_tokens=int(handoff_info["decode_tokens"]), output=output,
                )
            st = _finalize_pd_stage_dict(handoff_info, dec)
        best_ratio = _solve_optimal_pd_ratio_closed_form(
            st, total=num_devices, p_dies=p_world, d_dies=d_world,
            output=output, mode=rec["pd_mode"],
        )
        coverage["exact_evaluated"] += 1
        row = _build_pd_candidate_row_from_stage(
            st, p_parsed=parsed_p, d_parsed=parsed_d, best_ratio=best_ratio,
            pd_mode=rec["pd_mode"], mtp_n=rec["mtp_n"], p_world=p_world, d_world=d_world,
        )
        rows.append(row)
        if row["feasible"]:
            if best_entry is None or _rank_single_pool_row(row) > _rank_single_pool_row(best_entry[0]):
                best_entry = (row, p_item["raw"], d_item["raw"])
    return rows, ([best_entry] if best_entry is not None else [])


def _search_pd_deployments(
    compose_for: Any,
    *,
    model_id: str,
    hw_family: str,
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
    granted_workers: int | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]], dict[str, int]]:
    """PD Stage Factorization: Map |P|+|D| stages concurrently, then closed-form Join."""
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
    hit_tokens_pool = int(math.floor(prompt * float(pool_prefix_hit_ratio))) if "pooled_dram_ssd" in pd_modes else 0
    is_pool_only = len(pd_modes) == 1 and pd_modes[0] == "pooled_dram_ssd"
    base_compute_tok = max(0, prompt - hit_tokens_pool) if is_pool_only else prompt
    p_mtp_by_d_mtp = {
        mtp_n: _resolve_pd_prefill_speculative_k(
            model_id, model_spec, decode_mtp_tokens=mtp_n, hw_family=hw_family,
        )
        for mtp_n in mtp_grid
    }
    p_by_pmtp, d_by_mtp, coverage = _static_admit_pd_stages(
        compose_for, p_topos=p_topos, d_topos=d_topos, mtp_grid=mtp_grid,
        p_mtp_by_d_mtp=p_mtp_by_d_mtp, num_devices=num_devices,
        prompt=prompt, output=output, tpot=tpot, compute_tokens=base_compute_tok,
        max_sequences_per_dp=max_sequences_per_dp,
        rewrite_plan=rewrite_plan, allow_unbound_formula=allow_unbound_formula,
    )
    mode_params = _resolve_pd_mode_params(
        pd_modes, prompt=prompt, pool_dram_ratio=pool_dram_ratio,
        pool_ssd_ratio=pool_ssd_ratio, pool_prefix_hit_ratio=pool_prefix_hit_ratio,
        ssd_bw_gbps=ssd_bw_gbps,
    )
    valid_modes = tuple(m for m in pd_modes if m in mode_params)
    pair_specs: list[tuple[int, dict[str, Any], dict[str, Any], str]] = []
    p_task_items: dict[tuple[int, int, int], dict[str, Any]] = {}
    d_active: dict[int, dict[str, Any]] = {}
    for mtp_n, d_list in d_by_mtp:
        p_list = p_by_pmtp.get(p_mtp_by_d_mtp[mtp_n], ())
        for p_item in p_list:
            for d_item in d_list:
                if int(p_item["world"]) + int(d_item["world"]) > num_devices:
                    continue
                d_active.setdefault(int(d_item["id"]), d_item)
                for pd_mode in valid_modes:
                    _, _, _, hit_tok, comp_tok = mode_params[pd_mode]
                    p_task_items.setdefault((int(p_item["id"]), comp_tok, hit_tok), p_item)
                    pair_specs.append((mtp_n, p_item, d_item, pd_mode))
    if not pair_specs:
        has_p = any(bool(v) for v in p_by_pmtp.values())
        has_d = any(bool(v) for _, v in d_by_mtp)
        reason = (
            "pd_prefill_and_decode_static_hbm_oom" if (not has_p and not has_d)
            else "pd_prefill_static_hbm_oom" if not has_p
            else "pd_decode_static_hbm_oom" if not has_d
            else "pd_pool_die_budget_insufficient"
        )
        return [
            _pd_static_infeasible_row(
                num_devices=num_devices, reason=reason, mtp_n=mtp_grid[0], pd_mode=pd_modes[0],
            ),
        ], [], coverage

    p_parsed, d_parsed, d_vram_caches, records, refine_results, refine_errors = _execute_pd_stage_mapreduce(
        pair_specs=pair_specs, p_task_items=p_task_items, d_active=d_active,
        mode_params=mode_params, num_devices=num_devices, prompt=prompt, output=output,
        ttft=ttft, tpot=tpot, kv_transfer_bw_gbps=kv_transfer_bw_gbps,
        ddr_bw_gbps=ddr_bw_gbps, ssd_bw_gbps=ssd_bw_gbps,
        max_sequences_per_dp=max_sequences_per_dp,
        apply_target_envelope=apply_target_envelope, rewrite_plan=rewrite_plan,
        allow_unbound_formula=allow_unbound_formula, granted_workers=granted_workers,
        coverage=coverage,
    )
    rows, best_entries = _reduce_pd_factorized_pairs(
        records, p_parsed=p_parsed, d_parsed=d_parsed, d_vram_caches=d_vram_caches,
        refine_results=refine_results, refine_errors=refine_errors,
        num_devices=num_devices, prompt=prompt, output=output, tpot=tpot,
        max_sequences_per_dp=max_sequences_per_dp, rewrite_plan=rewrite_plan, coverage=coverage,
    )
    if num_devices >= GC_RECLAIM_MIN_WORLD_SIZE:
        _reclaim_sweep_memory()
    return rows, best_entries, coverage


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
    deployment: str | None,
    pd_transfer_mode: str | None,
    pool_dram_ratio: float | None = None,
    pool_ssd_ratio: float,
    pool_prefix_hit_ratio: float = 0.0,
    ssd_bw_gbps: float | None,
    allow_unconstrained_ttft: bool,
    override_role_promoted: bool = False,
) -> tuple[int, int, int, float, float, DeploymentMode]:
    """Validate explicit deployment mode and Domain-1 scenario, SLA, and SSD parameters."""
    dep_norm = validate_deployment_mode(deployment)
    if dep_norm == "pd_decode" and ttft_sla_ms is not None:
        raise ModelingError(
            "pd_decode_ttft_unavailable:ttft_sla_ms must be None when deployment_mode='pd_decode'"
        )
    if dep_norm == "pd_prefill" and tpot_sla_ms is not None and not override_role_promoted:
        raise ModelingError(
            "pd_prefill_rejects_decode_sla:tpot_sla_ms must be None when deployment_mode='pd_prefill'"
        )
    ttft_required = (dep_norm != "pd_decode") and not allow_unconstrained_ttft
    tpot_required = dep_norm != "pd_prefill"
    missing = [
        name for name, val in (
            ("num_devices", num_devices),
            ("prompt_tokens", prompt_tokens),
            ("output_tokens", output_tokens),
            *([] if not ttft_required else [("ttft_sla_ms", ttft_sla_ms)]),
            *([] if not tpot_required else [("tpot_sla_ms", tpot_sla_ms)]),
        ) if val is None
    ]
    if float(pool_ssd_ratio) > 0.0 and ssd_bw_gbps is None:
        missing.append("ssd_bw_gbps")
    if missing:
        raise missing_scenario_error(missing)
    n_dev = check_pos_int("num_devices", num_devices, minimum=1)
    if dep_norm == "pd_disaggregated" and n_dev < MIN_DISAGGREGATED_DEVICES:
        raise ModelingError(f"pd_disaggregated_requires_at_least_two_devices:{n_dev}")
    prompt = check_pos_int("prompt_tokens", prompt_tokens, minimum=1)
    output = check_pos_int("output_tokens", output_tokens, minimum=1)
    ttft = (
        math.inf if (dep_norm == "pd_decode" or (ttft_sla_ms is None and allow_unconstrained_ttft))
        else check_pos_float("ttft_sla_ms", ttft_sla_ms)
    )
    tpot = (
        math.inf if dep_norm == "pd_prefill"
        else check_pos_float("tpot_sla_ms", tpot_sla_ms)
    )
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
            raise ModelingError(f"direct_hbm_conflicts_with_kv_pool_ratios:{mode_norm}")
    return n_dev, prompt, output, ttft, tpot, dep_norm


def _recompose_winner_case(
    case_dict: dict[str, Any] | None,
    *,
    parallel: Mapping[str, Any] | None,
    mtp_n: int,
    role: str,
    b_vram_max: int,
    max_sequences_per_dp: int | None,
    compose_for: Any,
) -> dict[str, Any] | None:
    """Re-compose the single winning case with admission-derived ``max_num_seqs``."""
    if (case_dict is None or max_sequences_per_dp is not None or b_vram_max < 1
            or not isinstance(case_dict.get("runtime"), Mapping) or not isinstance(parallel, Mapping)):
        return case_dict
    topo = {k: int(v) for k, v in parallel.items() if k in _ALLOWED_PINNED_AXES and v is not None}
    if "cp" in topo and "dcp" not in topo:
        topo["dcp"] = topo.pop("cp")
    try:
        eff_override = int(b_vram_max)
        fresh = compose_for(topo, int(mtp_n), role, max_num_seqs_override=eff_override)
    except TypeError:
        return case_dict
    return deepcopy(fresh) if isinstance(fresh, Mapping) and "runtime" in fresh else case_dict


def _build_deployment_outcome(
    *,
    dep_norm: DeploymentMode,
    winner_row: Mapping[str, Any] | None,
    all_candidates: Sequence[Mapping[str, Any]],
    recommended_case: dict[str, Any] | None,
    prefill_case: dict[str, Any] | None,
    decode_case: dict[str, Any] | None,
    ttft: float,
    tpot: float,
    search_coverage: Mapping[str, int] | None,
) -> DeploymentOutcome:
    cov_items = tuple((str(k), max(0, int(v))) for k, v in (search_coverage or {}).items())
    ttft_avail = (
        "unavailable" if dep_norm == "pd_decode"
        else ("unconstrained" if math.isinf(ttft) else "measured")
    )
    tpot_avail = (
        "unavailable" if dep_norm == "pd_prefill"
        else ("unconstrained" if math.isinf(tpot) else "measured")
    )
    if winner_row is None:
        first_reason = next(
            (str(r.get("limiting_constraint") or r.get("infeasible_reason") or "")
             for r in all_candidates if r.get("limiting_constraint") or r.get("infeasible_reason")),
            "infeasible",
        )
        max_b_vram = max((int(r.get("b_vram_max", 0) or 0) for r in all_candidates), default=0)
        cfg_seqs = max((int(r.get("configured_max_num_seqs", 0) or 0) for r in all_candidates), default=0)
        return DeploymentOutcome(
            status="infeasible", deployment_mode=dep_norm,
            actual_transfer_mode="none", limiting_constraint=first_reason,
            physical_b_vram_max=max_b_vram, configured_max_num_seqs=cfg_seqs,
            feasible_b_per_dp=0, verified_ttft_ms=None, verified_tpot_ms=None,
            ttft_availability=ttft_avail, tpot_availability=tpot_avail,
            winner_case=None, prefill_case=None, decode_case=None,
            search_coverage=cov_items,
        )
    raw_ttft = winner_row.get("ttft_ms")
    raw_tpot = winner_row.get("effective_tpot_ms")
    v_ttft = (
        None if (dep_norm == "pd_decode" or raw_ttft is None or float(raw_ttft) <= 0.0)
        else float(raw_ttft)
    )
    v_tpot = (
        None if (dep_norm == "pd_prefill" or raw_tpot is None or float(raw_tpot) <= 0.0)
        else float(raw_tpot)
    )
    return DeploymentOutcome(
        status="feasible",
        deployment_mode=dep_norm,
        actual_transfer_mode=str(winner_row.get("pd_transfer_mode") or "none"),
        limiting_constraint=str(winner_row.get("limiting_constraint") or winner_row.get("operating_regime") or ""),
        physical_b_vram_max=max(0, int(winner_row.get("b_vram_max", 0) or 0)),
        configured_max_num_seqs=max(0, int(winner_row.get("configured_max_num_seqs", 0) or 0)),
        feasible_b_per_dp=max(0, int(winner_row.get("b_feasible", 0) or 0)),
        verified_ttft_ms=v_ttft,
        verified_tpot_ms=v_tpot,
        ttft_availability=ttft_avail,
        tpot_availability=tpot_avail,
        winner_case=recommended_case,
        prefill_case=prefill_case,
        decode_case=decode_case,
        search_coverage=cov_items,
    )


def _outcome_to_dict(outcome: DeploymentOutcome) -> dict[str, Any]:
    """Serialize a validated ``DeploymentOutcome`` to JSON-safe primitives."""
    return {
        "status": str(outcome.status),
        "deployment_mode": str(outcome.deployment_mode),
        "actual_transfer_mode": str(outcome.actual_transfer_mode),
        "limiting_constraint": str(outcome.limiting_constraint),
        "physical_b_vram_max": int(outcome.physical_b_vram_max),
        "configured_max_num_seqs": int(outcome.configured_max_num_seqs),
        "feasible_b_per_dp": int(outcome.feasible_b_per_dp),
        "requested_concurrency": outcome.requested_concurrency,
        "verified_ttft_ms": outcome.verified_ttft_ms,
        "verified_tpot_ms": outcome.verified_tpot_ms,
        "ttft_availability": str(outcome.ttft_availability),
        "tpot_availability": str(outcome.tpot_availability),
        "has_winner_case": bool(outcome.winner_case is not None),
        "search_coverage": dict(outcome.search_coverage),
    }


def _finalize_optimal_deployment(
    *,
    model_id: str,
    sku: str,
    n_dev: int,
    prompt: int,
    output: int,
    ttft: float,
    tpot: float,
    dep_norm: DeploymentMode,
    all_candidates: list[dict[str, Any]],
    coloc_entries: list[tuple[dict[str, Any], dict[str, Any]]],
    pd_entries: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    compose_for: Any,
    prefill_entries: list[tuple[dict[str, Any], dict[str, Any]]] | None = None,
    max_sequences_per_dp: int | None = None,
    search_coverage: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    dep_mode = validate_deployment_mode(dep_norm)
    best_coloc_entry = max(coloc_entries, key=lambda item: _rank_single_pool_row(item[0]), default=None)
    best_pd_entry = max(pd_entries, key=lambda item: _rank_single_pool_row(item[0]), default=None)
    best_prefill_entry = max(prefill_entries or (), key=lambda item: _rank_pd_prefill_row(item[0]), default=None)
    winner_row = recommended_case = prefill_case_out = decode_case_out = None
    pd_pool_summary: dict[str, Any] | None = None
    if dep_mode == "pd_prefill" and best_prefill_entry is not None:
        winner_row, prefill_case_out = best_prefill_entry[0], deepcopy(best_prefill_entry[1])
        recommended_case = deepcopy(prefill_case_out)
    elif best_pd_entry and (best_coloc_entry is None or _rank_single_pool_row(best_pd_entry[0]) >= _rank_single_pool_row(best_coloc_entry[0])):
        winner_row, prefill_case_out, decode_case_out = (
            best_pd_entry[0], deepcopy(best_pd_entry[1]), deepcopy(best_pd_entry[2]),
        )
        decode_case_out = _recompose_winner_case(
            decode_case_out, parallel=winner_row.get("decode_parallel"),
            mtp_n=int(winner_row.get("mtp_proposal_tokens", 0) or 0), role="pd_decode",
            b_vram_max=int(winner_row.get("b_vram_max", 0) or 0),
            max_sequences_per_dp=max_sequences_per_dp, compose_for=compose_for,
        )
        recommended_case = deepcopy(prefill_case_out if output <= 1 else decode_case_out)
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
        if not (dep_mode == "colocated" and output <= 1):
            recommended_case = _recompose_winner_case(
                recommended_case, parallel=winner_row.get("parallel"),
                mtp_n=int(winner_row.get("mtp_proposal_tokens", 0) or 0), role=dep_mode,
                b_vram_max=int(winner_row.get("b_vram_max", 0) or 0),
                max_sequences_per_dp=max_sequences_per_dp, compose_for=compose_for,
            )
        if dep_mode == "pd_decode":
            decode_case_out = deepcopy(recommended_case)

    outcome = _build_deployment_outcome(
        dep_norm=dep_mode, winner_row=winner_row, all_candidates=all_candidates,
        recommended_case=recommended_case, prefill_case=prefill_case_out,
        decode_case=decode_case_out, ttft=ttft, tpot=tpot, search_coverage=search_coverage,
    )
    pareto_metric = (
        (lambda r: float(r.get("prefill_rps_per_die", 0.0) or 0.0))
        if dep_mode == "pd_prefill" else _single_die_decode_tps
    )
    return {
        "schema": "mindie-sim.optimal-deployment/1",
        "model": model_id, "hardware": sku, "num_devices": n_dev,
        "prompt_tokens": prompt, "output_tokens": output,
        "ttft_sla_ms": (None if math.isinf(ttft) else ttft),
        "tpot_sla_ms": (None if math.isinf(tpot) else tpot),
        "deployment_mode": dep_mode,
        "deployment_constraint": dep_mode,
        "winner_mode": winner_row["deployment_mode"] if winner_row else None,
        "feasible": winner_row is not None,
        "deployment_outcome": _outcome_to_dict(outcome),
        "search_coverage": dict(search_coverage or {}),
        "best_configuration": winner_row,
        "best_pd": best_pd_entry[0] if best_pd_entry else None,
        "best_colocated": best_coloc_entry[0] if (best_coloc_entry and dep_mode == "colocated") else None,
        "pareto_frontier": _mark_pd_pareto(all_candidates, metric_fn=pareto_metric),
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

    base_kw = {k: v for k, v in common_kw.items() if k != "role"}

    def compose_for(
        topo: Mapping[str, int],
        mtp_n: int,
        role: str = "colocated",
        *,
        max_num_seqs_override: int | None = None,
    ) -> dict[str, Any] | None:
        if int(mtp_n) <= 0:
            accept = 1.0
        elif mtp_acceptance_length is not None:
            accept = float(mtp_acceptance_length)
        else:
            raise NotImplementedError(
                "LEGACY_PURGED: deleted magic DEFAULT_SPECULATIVE_ACCEPT_RATE=0.70; "
                "derive acceptance strictly from SpeculativeExecutionContract / DraftPolicy (S6-W5-39)"
            )
        eff_seqs = (
            int(max_num_seqs_override) if max_num_seqs_override is not None
            else (int(max_sequences_per_dp) if max_sequences_per_dp is not None else MIN_ADMISSION_PROBE_SEQS)
        )
        verify_width = 1 + max(0, int(mtp_n))
        call_kw = dict(base_kw)
        call_kw["max_num_seqs"] = max(1, eff_seqs)
        if role == "pd_decode":
            call_kw["ttft_slo_ms"] = None
            cur_batched = int(call_kw.get("max_num_batched_tokens") or 0)
            call_kw["max_num_batched_tokens"] = max(cur_batched, call_kw["max_num_seqs"] * verify_width)
        elif role == "pd_prefill":
            call_kw["tpot_slo_ms"] = None
            seq_floor = max(call_kw["max_num_seqs"], DEFAULT_CHUNKED_BATCH_SEQ_FLOOR)
            chunk_cap = max(DEFAULT_CHUNKED_PREFILL_TOKENS, seq_floor * verify_width)
            if prompt > chunk_cap:
                call_kw["max_num_batched_tokens"] = chunk_cap
        else:
            seq_floor = max(call_kw["max_num_seqs"], DEFAULT_CHUNKED_BATCH_SEQ_FLOOR)
            chunk_cap = max(DEFAULT_CHUNKED_PREFILL_TOKENS, seq_floor * verify_width)
            if prompt > chunk_cap:
                call_kw["max_num_batched_tokens"] = chunk_cap
            elif "max_num_batched_tokens" in call_kw and call_kw["max_num_batched_tokens"] is not None:
                call_kw["max_num_batched_tokens"] = max(
                    int(call_kw["max_num_batched_tokens"]), call_kw["max_num_seqs"] * verify_width,
                )
        topo_int = {k: int(v) for k, v in topo.items()}
        cache_key: str | None = None
        if (isinstance(call_kw.get("model"), (str, Mapping))
                and isinstance(call_kw.get("hardware"), (str, Mapping))):
            try:
                cache_key = json.dumps(
                    {
                        "kw": call_kw, "role": role, "topo": topo_int,
                        "mtp_n": int(mtp_n), "accept": float(accept),
                    },
                    sort_keys=True, separators=(",", ":"), default=repr,
                )
            except (TypeError, ValueError):
                cache_key = None
        if cache_key is not None and cache_key in _SOLVE_COMPOSE_CACHE:
            _SOLVE_COMPOSE_CACHE.move_to_end(cache_key)
            return deepcopy(_SOLVE_COMPOSE_CACHE[cache_key])
        try:
            composed = compose_case(
                **call_kw, role=role, **topo_int,
                mtp_proposal_tokens=mtp_n, mtp_acceptance_length=accept,
            )
            if cache_key is not None and isinstance(composed, dict):
                _SOLVE_COMPOSE_CACHE[cache_key] = deepcopy(composed)
                while len(_SOLVE_COMPOSE_CACHE) > _SOLVE_COMPOSE_CACHE_MAX:
                    _SOLVE_COMPOSE_CACHE.popitem(last=False)
            return composed
        except UnmodeledCapabilityError:
            raise
        except (ModelingError, ValueError) as exc:
            if _is_ingress_contradiction(exc):
                raise
            return None

    compose_for._spec = (base_kw, int(prompt), max_sequences_per_dp, mtp_acceptance_length)  # type: ignore[attr-defined]
    return compose_for


def solve_optimal_deployment(
    model: str | Mapping[str, Any], hardware: str | Mapping[str, Any] = "a3", *,
    num_devices: int | None = None, prompt_tokens: int | None = None, output_tokens: int | None = None,
    ttft_sla_ms: float | None = None, tpot_sla_ms: float | None = None,
    deployment: str | None = None, deployment_mode: str | None = None,
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
    granted_workers: int | None = None,
) -> dict[str, Any]:
    """Search optimal deployment topology under explicit ``deployment_mode`` and user SLAs."""
    eff_workers = resolve_granted_workers(granted_workers)
    if deployment_mode is not None and deployment is not None:
        if str(deployment_mode).strip().lower() != str(deployment).strip().lower():
            raise ModelingError(f"conflicting_deployment_mode:{deployment_mode}!={deployment}")
    overrides_clean = {k: v for k, v in (compose_overrides or {}).items() if v is not None}
    override_role = overrides_clean.pop("role", None)
    raw_dep = deployment_mode if deployment_mode is not None else deployment
    override_role_promoted = False
    if (raw_dep is not None and str(raw_dep).strip().lower() == "colocated"
            and isinstance(override_role, str)
            and override_role.strip().lower() in {"pd_decode", "pd_prefill"}):
        raw_dep = override_role.strip().lower()
        override_role_promoted = True
    n_dev, prompt, output, ttft, tpot, dep_norm = _validate_optimal_deployment_inputs(
        num_devices=num_devices, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        ttft_sla_ms=ttft_sla_ms, tpot_sla_ms=tpot_sla_ms, deployment=raw_dep,
        pd_transfer_mode=pd_transfer_mode, pool_dram_ratio=pool_dram_ratio,
        pool_ssd_ratio=pool_ssd_ratio, pool_prefix_hit_ratio=pool_prefix_hit_ratio,
        ssd_bw_gbps=ssd_bw_gbps, allow_unconstrained_ttft=allow_unconstrained_ttft,
        override_role_promoted=override_role_promoted,
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
    ext_plan = _load_canonical_rewrite_plan(rewrite_plan) if rewrite_plan is not None else None
    use_target_env = bool(apply_target_envelope or ext_plan is not None)

    from mindie_sim.compose.inputs import _arch_key, _resolve_hardware, _resolve_model_spec
    from mindie_sim.compose.runtime import va_recommended_config

    sku, _, _ = _resolve_hardware(hardware)
    hw_family = _arch_key(sku)
    model_spec = _resolve_model_spec(model, hw_family=hw_family)
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
        ttft_slo_ms=(None if math.isinf(ttft) else ttft),
        tpot_slo_ms=(None if math.isinf(tpot) else tpot),
        quant=quant, layers=layers, ranks_per_node=ranks_per_node, topology_preset=topology_preset,
        va_recommended=va_recommended, raw_eager=raw_eager, feature_track=feature_track,
        runtime_stack=runtime_stack, optimizations=optimizations,
        exclude_optimizations=exclude_optimizations, engram_cpu_offload=engram_cpu_offload,
        multistream_overlap_shared_expert=multistream_overlap_shared_expert,
        enable_shared_expert_dp=enable_shared_expert_dp, sp=sp,
        graph_capture_mode=graph_capture_mode, kv_cache_dtype=kv_cache_dtype,
        indexer_kv_dtype=indexer_kv_dtype, parallel="fixed", homogeneous_graph_reuse="auto",
        **overrides_clean,
    )
    compose_for = _make_solve_compose_fn(
        common_kw, prompt=prompt, max_sequences_per_dp=max_sequences_per_dp,
        mtp_acceptance_length=mtp_acceptance_length,
    )
    probe_role = dep_norm if dep_norm in {"pd_decode", "pd_prefill"} else "colocated"
    compose_for({"dp": 1, "tp": 1, "ep": 1, "pp": 1, "pcp": 1, "dcp": 1}, 0, probe_role)

    all_candidates: list[dict[str, Any]] = []
    coloc_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    prefill_entries: list[tuple[dict[str, Any], dict[str, Any]]] = []
    pd_entries: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    coverage: dict[str, int] = {}

    if dep_norm in {"colocated", "pd_decode"}:
        c_rows, coloc_entries, coverage = _search_single_pool_deployments(
            compose_for, dep_mode=dep_norm, num_devices=n_dev, is_moe=is_moe, layers=eff_layers,
            model_spec=model_spec, rec_topo=rec_topo, pinned=(d_pin if dep_norm == "pd_decode" else coloc_pin),
            mtp_grid=mtp_grid, prompt=prompt, output=output, ttft=ttft, tpot=tpot,
            max_sequences_per_dp=max_sequences_per_dp, apply_target_envelope=use_target_env,
            rewrite_plan=ext_plan, allow_unbound_formula=allow_unbound_formula,
            granted_workers=eff_workers,
        )
        all_candidates.extend(c_rows)
    elif dep_norm == "pd_prefill":
        pf_rows, prefill_entries, coverage = _search_pd_prefill_deployments(
            compose_for, num_devices=n_dev, is_moe=is_moe, layers=eff_layers,
            model_spec=model_spec, rec_topo=rec_topo, pinned=p_pin, mtp_grid=mtp_grid,
            prompt=prompt, output=output, ttft=ttft, apply_target_envelope=use_target_env,
            rewrite_plan=ext_plan, allow_unbound_formula=allow_unbound_formula,
            granted_workers=eff_workers,
        )
        all_candidates.extend(pf_rows)
    elif dep_norm == "pd_disaggregated":
        p_rows, pd_entries, coverage = _search_pd_deployments(
            compose_for, model_id=model_id, hw_family=hw_family,
            num_devices=n_dev, is_moe=is_moe, layers=eff_layers,
            model_spec=model_spec, rec_topo=rec_topo, p_pin=p_pin, d_pin=d_pin, mtp_grid=mtp_grid,
            prompt=prompt, output=output, ttft=ttft, tpot=tpot, pd_transfer_mode=pd_transfer_mode,
            kv_transfer_bw_gbps=kv_transfer_bw_gbps, pool_dram_ratio=pool_dram_ratio,
            pool_ssd_ratio=pool_ssd_ratio, pool_prefix_hit_ratio=pool_prefix_hit_ratio,
            ddr_bw_gbps=ddr_bw_gbps, ssd_bw_gbps=ssd_bw_gbps,
            max_sequences_per_dp=max_sequences_per_dp, apply_target_envelope=use_target_env,
            rewrite_plan=ext_plan, allow_unbound_formula=allow_unbound_formula,
            granted_workers=eff_workers,
        )
        all_candidates.extend(p_rows)

    return _finalize_optimal_deployment(
        model_id=model_id, sku=sku, n_dev=n_dev, prompt=prompt, output=output,
        ttft=ttft, tpot=tpot, dep_norm=dep_norm, all_candidates=all_candidates,
        coloc_entries=coloc_entries, pd_entries=pd_entries, compose_for=compose_for,
        prefill_entries=prefill_entries, max_sequences_per_dp=max_sequences_per_dp,
        search_coverage=coverage,
    )


__all__ = [
    "AUTO_PD_TRANSFER_MODES",
    "VALID_PD_TRANSFER_MODES",
    "solve_optimal_deployment",
    "sweep_pd_disaggregation_ratios",
]
