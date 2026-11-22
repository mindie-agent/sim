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
from dataclasses import fields as dataclass_fields, replace
import ctypes
import gc
import math
import re
import sys
from typing import Any

try:
    from mindie_sim.catalog.context_parallel import (
        legal_vllm_ascend_dcp_values,
        validate_vllm_ascend_dcp_codomain,
    )
except ImportError:  # Standalone pre-merge compatibility before catalog DCP helpers land.
    from mindie_sim.catalog.context_parallel import is_v41_replicated_cache_model

    def _model_dcp_attention_class(model: Any) -> str:
        if is_v41_replicated_cache_model(model):
            return "replicated_cache"
        plans = tuple(getattr(model, "layer_plans", ()) or ()) or (model,)
        if any(
            str(getattr(lp, "attention", "") or "").lower() == "dsa"
            or (str(getattr(lp, "attention", "") or "").lower() == "mla"
                and bool(getattr(lp, "indexer", False)))
            for lp in plans
        ):
            return "sfa_dsa"
        if any(
            str(getattr(lp, "attention", "") or "").lower() == "mla"
            or str(getattr(lp, "cache_kind", "") or "").lower() == "window_compressed"
            for lp in plans
        ):
            return "mla"
        return "gqa"

    def legal_vllm_ascend_dcp_values(
        model: Any, *, tp: int, pcp: int = 1,
    ) -> tuple[int, ...]:
        tp_n, pcp_n = max(1, int(tp)), max(1, int(pcp))
        kind = _model_dcp_attention_class(model)
        if kind in {"gqa", "replicated_cache"}:
            return (1,)
        if pcp_n > 1:
            return tuple(dict.fromkeys((1, pcp_n, pcp_n * tp_n)))
        if kind == "sfa_dsa":
            return (1, tp_n) if tp_n > 1 else (1,)
        divisors = [1]
        width = 2
        while width <= tp_n:
            if tp_n % width == 0:
                divisors.append(width)
            width *= 2
        if tp_n not in divisors:
            divisors.append(tp_n)
        return tuple(sorted(set(divisors)))

    def validate_vllm_ascend_dcp_codomain(
        model: Any, parallel: Any,
    ) -> tuple[bool, str | None]:
        dcp = max(1, int(getattr(parallel, "cp", 1) or 1))
        tp = max(1, int(getattr(parallel, "tp", 1) or 1))
        pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
        if dcp <= 1:
            return True, None
        kind = _model_dcp_attention_class(model)
        if kind == "replicated_cache":
            return False, "vllm_ascend_replicated_cache_requires_dcp_1"
        if kind == "gqa":
            return False, "vllm_ascend_gqa_dcp_deprecated"
        legal = legal_vllm_ascend_dcp_values(model, tp=tp, pcp=pcp)
        if dcp in legal:
            return True, None
        if pcp > 1:
            return False, f"vllm_ascend_pcp_dcp_must_be_pcp_or_pcp_times_tp:dcp={dcp},pcp={pcp},tp={tp}"
        if kind == "sfa_dsa":
            return False, f"vllm_ascend_sfa_dcp_must_equal_tp:dcp={dcp},tp={tp}"
        return False, f"vllm_ascend_mla_dcp_must_divide_tp:dcp={dcp},tp={tp}"
from mindie_sim.prediction.body_defaults import TargetGraphSet
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.disaggregated.handoff import compute_pd_handoff_plan
from mindie_sim.prediction.efficiency import (
    _lower_and_run_graph,
    _resolve_speculative_tokens,
)
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.graph_analysis import graph_costs
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
from mindie_sim.schema.common import ModelingError
from mindie_sim.sweep.resize import check_pos_float, check_pos_int

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
ARCHETYPE_SCREEN_MIN_LAYERS = 12
ARCHETYPE_FOLD_MIN_LAYERS = 2
COLOCATED_STAGE2_FINALISTS = 3
PD_PREFILL_STAGE2_FINALISTS = 3
PD_DECODE_STAGE2_FINALISTS = 4
SCREEN_DECODE_BATCH_PROBE = 8
SLA_SLACK_BOUND_TOLERANCE_MS = 0.5
_LAYER_POSITION_FIELDS = frozenset({
    "index", "native_layer_id", "kv_owner_layer_id",
    "index_k_owner_layer_id", "topk_owner_layer_id",
    "candidate_source_layer_id", "index_source_layer_id",
})
_LAYER_EXCLUSION_INDEX_RE = re.compile(r"((?:layers|layer|blocks|h)[._])(\d+)([._]|$)")
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
        p_step = _prefill_step_ms(
            p_parsed, compute_tokens=compute_tokens,
            hit_tokens=hit_tokens, min_layers=ARCHETYPE_FOLD_MIN_LAYERS,
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
        d_sla = _solve_sla_archetype_aware(
            d_parsed, sla_ms=eff_sla_ms, context=eff_decode,
            output_budget=output, max_sequences_per_dp=max_sequences_per_dp,
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
    d_par = d_parsed.runtime.parallel
    per_dp_dies = max(1, int(d_par.tp) * int(d_par.pcp) * int(d_par.pp))
    n_tok_per_die = (float(max(1, b_feasible)) * emitted) / float(per_dp_dies) if b_feasible >= 1 else 0.0
    us_per_tok_die = (decode_step * US_PER_MS) / n_tok_per_die if n_tok_per_die > 0.0 else 0.0
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
        "effective_tpot": effective_tpot, "first_itl": first_itl,
        "tpot_ok": tpot_ok, "rps_d": rps_d, "raw_rps_d": raw_rps_d,
        "decode_tps": rps_d * float(output),
        "avg_accepted_tokens": emitted,
        "n_tok_per_die": n_tok_per_die,
        "us_per_tok_die": us_per_tok_die,
        "unused_sla_slack_ms": unused_sla_slack_ms,
        "operating_regime": operating_regime,
        "free_hbm_headroom_gib": free_hbm_gib,
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
    decode_q_tokens, _ = _resolve_speculative_tokens(
        d_parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    d_mem = Memory(d_parsed, workspace_tokens=decode_q_tokens)
    decode_tokens = output - 1
    link_info = _resolve_pd_link_bandwidth(
        p_parsed, d_parsed, total_dies=total, kv_transfer_bw_gbps=kv_transfer_bw_gbps,
    )
    if decode_tokens <= 0:
        prefill_ok = bool(p_step <= ttft_sla_ms)
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
            "p_step": p_step, "effective_ttft_ms": p_step,
            "prefill_hbm_ok": prefill_hbm_ok, "prefill_ok": prefill_ok,
            "compute_tokens": compute_tokens, "hit_tokens": hit_tokens,
            "kv_bytes": 0.0, "unique_kv_bytes": 0.0, "kv_ms": 0.0,
            "exposed_kv_ms": 0.0, "tail_layer_kv_ms": 0.0,
            "rps_p": rps_p, "rps_d": rps_p, "raw_rps_d": rps_p, "decode_tps": rps_p * float(output),
            "p_kv_rps_per_inst": rps_p, "d_kv_rps_per_inst": None,
            "b_sla": 0, "b_vram": int(d_parsed.runtime.max_num_seqs), "b_feasible": 0,
            "b1_step": 0.0, "decode_step": 0.0, "effective_tpot": 0.0, "first_itl": 0.0,
            "decode_budget": tpot_sla_ms, "tpot_ok": True, "infeasible": infeasible,
            "link_info": link_info, "pool_metrics": None, "handoff_plan": None,
            "avg_accepted_tokens": 1.0, "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
            "unused_sla_slack_ms": 0.0, "operating_regime": "SLA_LATENCY_BOUND",
            "free_hbm_headroom_gib": 0.0,
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

    prefill_compute_ok = bool(p_step <= ttft_sla_ms)
    effective_ttft_ms = p_step + (exposed_kv_ms if mode == "direct_blocking_d2d" else 0.0)
    prefill_ok = bool(effective_ttft_ms <= ttft_sla_ms)
    rps_p = (p_dp * per_rep) if (prefill_ok and prefill_hbm_ok) else 0.0
    dec = _measure_decode_stage(
        d_parsed, prompt=prompt, output=output, eff_decode=eff_decode,
        tpot_sla_ms=tpot_sla_ms, exposed_kv_ms=exposed_kv_ms, decode_tokens=decode_tokens,
        max_sequences_per_dp=max_sequences_per_dp,
        cache=_decode_sla_cache, cache_id=_decode_cache_id,
    )
    infeasible = (
        "prefill_hbm_oom" if not prefill_hbm_ok
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
    prefill_case: CaseSpec | Mapping[str, Any], decode_case: CaseSpec | Mapping[str, Any], *,
    total_dies: int, prompt_tokens: int | None = None, output_tokens: int | None = None,
    decode_context: int | None = None, ttft_sla_ms: float | None = None,
    tpot_sla_ms: float | None = None, kv_transfer_bw_gbps: float | None = None,
    pd_transfer_mode: str = "direct_blocking_d2d", pool_dram_ratio: float | None = None,
    pool_ssd_ratio: float = 0.0, pool_prefix_hit_ratio: float = 0.0,
    ddr_bw_gbps: float | None = None, ssd_bw_gbps: float | None = None,
    max_sequences_per_dp: int | None = None, allow_unconstrained_ttft: bool = False,
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
            "step_ms": float(st["p_step"]),
            "effective_ttft_ms": float(st.get("effective_ttft_ms", st["p_step"])),
            "hbm_ok": bool(st["prefill_hbm_ok"]),
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
            "avg_accepted_tokens": float(st["avg_accepted_tokens"]),
            "n_tok_per_die": float(st["n_tok_per_die"]),
            "us_per_tok_die": float(st["us_per_tok_die"]),
            "unused_sla_slack_ms": float(st["unused_sla_slack_ms"]),
            "operating_regime": str(st["operating_regime"]),
            "free_hbm_headroom_gib": float(st["free_hbm_headroom_gib"]),
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
        return (int(pinned["dcp"]),)
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
        if model_spec is not None and "dcp" not in pinned and dcp > 1:
            if dcp not in legal_vllm_ascend_dcp_values(model_spec, tp=tp, pcp=pcp):
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
        if model_spec is not None and "dcp" not in pinned and dcp > 1:
            if dcp not in legal_vllm_ascend_dcp_values(model_spec, tp=tp, pcp=pcp):
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
) -> dict[str, Any]:
    parsed = parse_case(raw_case)
    par = parsed.runtime.parallel
    used_dies = par.world_size
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
        return {
            "deployment_mode": "colocated", "used_dies": used_dies, "parallel": par.to_dict(),
            "mtp_proposal_tokens": mtp_tokens, "ttft_ms": 0.0, "effective_tpot_ms": 0.0,
            "b_feasible": 0, "b_vram_max": 0, "sustainable_rps": 0.0, "sustainable_tps": 0.0,
            "sustainable_tps_per_die": 0.0, "avg_accepted_tokens": 1.0,
            "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
            "unused_sla_slack_ms": 0.0, "operating_regime": "HBM_CAPACITY_BOUND",
            "free_hbm_headroom_gib": 0.0,
            "feasible": False, "infeasible_reason": "prefill_hbm_oom",
        }
    rep_info = build_archetype_representative_case(parsed, min_layers=ARCHETYPE_FOLD_MIN_LAYERS)
    p_step = _prefill_step_ms(
        parsed, compute_tokens=prompt_tokens, hit_tokens=0,
        min_layers=ARCHETYPE_FOLD_MIN_LAYERS,
        rep_info=rep_info,
    )
    ttft_ok = bool(p_step <= ttft_sla_ms)
    decode_tokens = max(0, output_tokens - 1)
    if decode_tokens == 0:
        feasible = bool(prefill_hbm_ok and ttft_ok)
        rps = (par.dp * MS_PER_S) / max(p_step, PARETO_EPS) if feasible else 0.0
        return {
            "deployment_mode": "colocated", "used_dies": used_dies, "parallel": par.to_dict(),
            "mtp_proposal_tokens": mtp_tokens, "ttft_ms": p_step, "effective_tpot_ms": 0.0,
            "b_feasible": 1 if feasible else 0, "b_vram_max": 1 if feasible else 0,
            "sustainable_rps": rps,
            "sustainable_tps": rps * float(output_tokens),
            "sustainable_tps_per_die": (rps * float(output_tokens)) / float(used_dies),
            "avg_accepted_tokens": 1.0, "n_tok_per_die": 0.0, "us_per_tok_die": 0.0,
            "unused_sla_slack_ms": 0.0, "operating_regime": "SLA_LATENCY_BOUND",
            "free_hbm_headroom_gib": 0.0,
            "feasible": feasible, "infeasible_reason": None if feasible else "prefill_sla_or_oom",
        }
    _, emit_est = _resolve_speculative_tokens(
        parsed, output_budget=output_tokens, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    d_sla = _solve_sla_archetype_aware(
        parsed, sla_ms=tpot_sla_ms * max(1.0, float(emit_est)),
        context=prompt_tokens, output_budget=output_tokens,
        max_sequences_per_dp=max_sequences_per_dp,
        workspace_tokens=prefill_ws,
        rep_info=rep_info, probe_cache=probe_cache,
    )
    scen = d_sla["scenarios"][0]
    emitted = float(d_sla["emitted_tokens_per_seq"])
    b_feasible = int(scen["b_feasible"])
    b_vram = int(scen.get("b_vram_max", b_feasible) or 0)
    decode_step = float(scen["step_ms_feasible"] if b_feasible >= 1 and scen.get("step_ms_feasible") is not None else scen["step_ms_at_b1"])
    tpot_ms = decode_step / max(emitted, PARETO_EPS)
    length_ok = bool(prompt_tokens + output_tokens <= parsed.runtime.max_model_len)
    feasible = bool(prefill_hbm_ok and ttft_ok and length_ok and b_feasible >= 1 and tpot_ms <= tpot_sla_ms + PARETO_EPS)
    req_compute_ms = p_step + (float(decode_tokens) * decode_step) / max(float(max(1, b_feasible)) * emitted, PARETO_EPS)
    rps = (float(par.dp) * MS_PER_S) / max(req_compute_ms, PARETO_EPS) if feasible else 0.0
    tps = rps * float(output_tokens)
    per_dp_dies = max(1, int(par.tp) * int(par.pcp) * int(par.pp))
    n_tok_per_die = (float(b_feasible) * emitted) / float(per_dp_dies) if b_feasible >= 1 else 0.0
    us_per_tok_die = (decode_step * US_PER_MS) / n_tok_per_die if n_tok_per_die > 0.0 else 0.0
    unused_slack_ms = max(0.0, float(tpot_sla_ms) - float(tpot_ms)) if feasible else 0.0
    operating_regime = (
        "HBM_CAPACITY_BOUND"
        if (feasible and b_vram >= 1 and b_feasible >= b_vram
            and unused_slack_ms > SLA_SLACK_BOUND_TOLERANCE_MS)
        else "SLA_LATENCY_BOUND"
    )
    dcp_ok, dcp_reason = validate_vllm_ascend_dcp_codomain(parsed.model, par)
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
        "sustainable_rps": rps, "sustainable_tps": tps,
        "sustainable_tps_per_die": tps / float(used_dies),
        "avg_accepted_tokens": emitted,
        "n_tok_per_die": n_tok_per_die,
        "us_per_tok_die": us_per_tok_die,
        "unused_sla_slack_ms": unused_slack_ms,
        "operating_regime": operating_regime,
        "free_hbm_headroom_gib": float(scen.get("free_hbm_headroom_gib", 0.0) or 0.0),
        "dcp_codomain_valid": bool(dcp_ok),
        "dcp_codomain_reason": dcp_reason,
        "feasible": feasible, "infeasible_reason": reason,
    }


def _layer_quant_exclusion_signature(quant: Any, layer_idx: int) -> tuple[Any, ...]:
    """Normalized per-layer quant exclusions & category overrides matching ``layer_idx``."""
    if quant is None:
        return ()
    target = str(int(layer_idx))
    matched_excl: list[str] = []
    for item in (getattr(quant, "exclusions", ()) or ()):
        text = str(item)
        m = _LAYER_EXCLUSION_INDEX_RE.search(text)
        if m is not None and m.group(2) == target:
            matched_excl.append(_LAYER_EXCLUSION_INDEX_RE.sub(r"\1*\3", text))
    matched_ov: list[tuple[str, Any]] = []
    raw_ov = getattr(quant, "category_overrides", ()) or ()
    ov_items = raw_ov.items() if isinstance(raw_ov, Mapping) else raw_ov
    for cat, fields in ov_items:
        cat_s = str(cat)
        m = _LAYER_EXCLUSION_INDEX_RE.search(cat_s)
        if m is not None and m.group(2) == target:
            norm_cat = _LAYER_EXCLUSION_INDEX_RE.sub(r"\1*\3", cat_s)
            norm_f = tuple(sorted(fields.items())) if isinstance(fields, Mapping) else tuple(fields)
            matched_ov.append((norm_cat, norm_f))
    if not matched_excl and not matched_ov:
        return ()
    return (tuple(sorted(matched_excl)), tuple(sorted(matched_ov)))


def _remap_layer_quant_exclusions(
    quant: Any, old_to_new: Mapping[int, int],
) -> Any:
    """Remap per-layer quant exclusions & category overrides onto deduplicated indices."""
    if quant is None:
        return quant
    exclusions = getattr(quant, "exclusions", ()) or ()
    updated_excl: list[str] = []
    excl_changed = False
    for item in exclusions:
        text = str(item)
        m = _LAYER_EXCLUSION_INDEX_RE.search(text)
        if m is None:
            updated_excl.append(text)
            continue
        excl_changed = True
        orig = int(m.group(2))
        if orig in old_to_new:
            mapped = _LAYER_EXCLUSION_INDEX_RE.sub(
                rf"\g<1>{int(old_to_new[orig])}\g<3>", text,
            )
            updated_excl.append(mapped)
    raw_ov = getattr(quant, "category_overrides", ()) or ()
    ov_items = list(raw_ov.items()) if isinstance(raw_ov, Mapping) else list(raw_ov)
    updated_ov: list[tuple[str, Any]] = []
    ov_changed = False
    for cat, fields in ov_items:
        cat_s = str(cat)
        m = _LAYER_EXCLUSION_INDEX_RE.search(cat_s)
        if m is None:
            updated_ov.append((cat_s, fields))
            continue
        ov_changed = True
        orig = int(m.group(2))
        if orig in old_to_new:
            mapped_cat = _LAYER_EXCLUSION_INDEX_RE.sub(
                rf"\g<1>{int(old_to_new[orig])}\g<3>", cat_s,
            )
            updated_ov.append((mapped_cat, fields))
    if not excl_changed and not ov_changed:
        return quant
    kwargs: dict[str, Any] = {}
    if excl_changed:
        kwargs["exclusions"] = tuple(dict.fromkeys(updated_excl))
    if ov_changed:
        kwargs["category_overrides"] = tuple(sorted(updated_ov))
    return replace(quant, **kwargs)


def _layer_archetype_key(
    lp: Any,
    stage: int,
    *,
    cand_producers: frozenset[int] = frozenset(),
    quant: Any = None,
) -> tuple[Any, ...]:
    """Automatic structural equivalence hash over every LayerPlan + quant contract field."""
    idx = int(getattr(lp, "index", 0))
    kv_own = int(getattr(lp, "kv_owner_layer_id", -1))
    idx_own = int(getattr(lp, "index_k_owner_layer_id", -1))
    topk_own = int(getattr(lp, "topk_owner_layer_id", -1))
    cand_src = int(getattr(lp, "candidate_source_layer_id", -1))
    idx_src = int(getattr(lp, "index_source_layer_id", -1))
    field_sig = tuple(
        (f.name, getattr(lp, f.name))
        for f in dataclass_fields(lp)
        if f.name not in _LAYER_POSITION_FIELDS
    )
    return (
        int(stage),
        field_sig,
        (kv_own >= 0, kv_own == idx),
        (idx_own >= 0, idx_own == idx),
        (topk_own >= 0, topk_own == idx),
        idx in cand_producers,
        cand_src >= 0,
        (idx_src >= 0, idx_src == idx),
        _layer_quant_exclusion_signature(quant, idx),
    )


def _stage_counts_for_case(parsed: CaseSpec) -> tuple[int, ...]:
    pp = max(1, int(parsed.runtime.parallel.pp))
    part = tuple(int(v) for v in (getattr(parsed.runtime, "pp_layer_partition", ()) or ()))
    n_layers = len(parsed.model.layer_plans)
    if len(part) == pp and sum(part) == n_layers:
        return part
    base, rem = divmod(n_layers, pp)
    return tuple(base + (1 if s < rem else 0) for s in range(pp))


def build_archetype_representative_case(
    parsed: CaseSpec,
    *,
    min_layers: int = ARCHETYPE_SCREEN_MIN_LAYERS,
) -> tuple[CaseSpec, dict[int, int], float]:
    """Deduplicate structurally identical layers while conserving quant, KeyPool, Engram, and Draft."""
    plans = tuple(parsed.model.layer_plans)
    if len(plans) < max(2, int(min_layers)):
        return parsed, {int(lp.index): 1 for lp in plans}, 1.0
    stage_counts = _stage_counts_for_case(parsed)
    layer_stage = [s for s, cnt in enumerate(stage_counts) for _ in range(cnt)]
    cand_producers = frozenset(
        int(getattr(lp, "candidate_source_layer_id", -1))
        for lp in plans if int(getattr(lp, "candidate_source_layer_id", -1)) >= 0
    )
    groups: dict[tuple[Any, ...], list[Any]] = {}
    for lp in plans:
        st = layer_stage[lp.index] if lp.index < len(layer_stage) else 0
        key = _layer_archetype_key(
            lp, st, cand_producers=cand_producers, quant=parsed.model.quant,
        )
        groups.setdefault(key, []).append(lp)
    if len(groups) >= len(plans) or parsed.runtime.parallel.pp > len(groups):
        return parsed, {int(lp.index): 1 for lp in plans}, 1.0

    old_to_new: dict[int, int] = {}
    layer_weights: dict[int, int] = {}
    rep_orig: dict[int, Any] = {}
    rep_stages: list[int] = []
    rep_list: list[Any] = []
    last_kv = last_idx = last_topk = last_cand = -1
    for new_idx, (grp_key, members) in enumerate(groups.items()):
        lp = members[0]
        old_idx = int(lp.index)
        old_to_new[old_idx] = new_idx
        layer_weights[new_idx] = len(members)
        rep_orig[new_idx] = lp
        rep_stages.append(int(grp_key[0]))
        kv_own = int(getattr(lp, "kv_owner_layer_id", -1))
        idx_own = int(getattr(lp, "index_k_owner_layer_id", -1))
        topk_own = int(getattr(lp, "topk_owner_layer_id", -1))
        cand_src = int(getattr(lp, "candidate_source_layer_id", -1))
        idx_src = int(getattr(lp, "index_source_layer_id", -1))
        prev_kv, prev_idx, prev_topk, prev_cand = last_kv, last_idx, last_topk, last_cand
        new_kv = (last_kv := new_idx) if kv_own == old_idx else (old_to_new.get(kv_own, prev_kv) if kv_own >= 0 else -1)
        new_idx_own = (last_idx := new_idx) if idx_own == old_idx else (old_to_new.get(idx_own, prev_idx) if idx_own >= 0 else -1)
        new_topk = (last_topk := new_idx) if topk_own == old_idx else (old_to_new.get(topk_own, prev_topk) if topk_own >= 0 else -1)
        if old_idx in cand_producers:
            last_cand = new_idx
        new_idx_src = new_idx if idx_src == old_idx else (old_to_new.get(idx_src, prev_idx) if idx_src >= 0 else -1)
        new_cand_src = old_to_new.get(cand_src, prev_cand) if cand_src >= 0 else -1
        rep_list.append(replace(
            lp, index=new_idx, kv_owner_layer_id=new_kv, index_k_owner_layer_id=new_idx_own,
            topk_owner_layer_id=new_topk, candidate_source_layer_id=new_cand_src,
            index_source_layer_id=new_idx_src,
        ))

    rep_plans = tuple(rep_list)
    aux = getattr(parsed.model, "auxiliary", None)
    rep_aux = aux
    draft_scale = 1.0
    if aux is not None and getattr(aux, "dspark", None) is not None:
        dsp = aux.dspark
        if dsp.layer_plans and len(dsp.layer_plans) > 1:
            draft_scale = float(len(dsp.layer_plans))
            d0 = replace(dsp.layer_plans[0], index=0, kv_owner_layer_id=0)
            clamped = tuple(min(int(lid), len(rep_plans) - 1) for lid in dsp.target_layer_ids)
            rep_aux = replace(aux, dspark=replace(dsp, layers=1, layer_plans=(d0,), target_layer_ids=clamped))
    rep_engram = getattr(parsed.model, "engram", None)
    if rep_engram is not None and getattr(rep_engram, "tables", ()):
        by_orig = {int(t.layer_id): t for t in rep_engram.tables}
        default_tbl = rep_engram.tables[0]
        remapped = tuple(
            replace(by_orig.get(int(orig_lp.index), default_tbl), layer_id=new_idx)
            for new_idx, orig_lp in sorted(rep_orig.items())
            if bool(getattr(orig_lp, "engram", False))
        )
        if remapped:
            rep_engram = replace(rep_engram, tables=remapped)
    rep_quant = _remap_layer_quant_exclusions(parsed.model.quant, old_to_new)
    rep_model = replace(
        parsed.model, layers=len(rep_plans), layer_plans=rep_plans,
        auxiliary=rep_aux, engram=rep_engram, quant=rep_quant,
    )
    pp = max(1, int(parsed.runtime.parallel.pp))
    rep_pp_part: tuple[int, ...] = ()
    if pp > 1:
        counts = tuple(sum(1 for st in rep_stages if st == s) for s in range(pp))
        if all(c >= 1 for c in counts) and sum(counts) == len(rep_plans):
            rep_pp_part = counts
        else:
            return parsed, {int(lp.index): 1 for lp in plans}, 1.0
    rep_case = replace(parsed, model=rep_model, runtime=replace(parsed.runtime, pp_layer_partition=rep_pp_part))
    return rep_case, layer_weights, draft_scale


def _archetype_weighted_step_ms(
    parsed: CaseSpec,
    *,
    phase: str,
    tokens: int,
    context: int,
    sequences: int,
    scope: str = "model",
    min_layers: int = ARCHETYPE_FOLD_MIN_LAYERS,
    rep_info: tuple[CaseSpec, dict[int, int], float] | None = None,
    evaluation_context: EvaluationContext | None = None,
) -> float:
    """Exact O(Unique Archetypes x 1 Representative Rank) step latency."""
    if rep_info is not None:
        rep_case, layer_weights, draft_scale = rep_info
    else:
        rep_case, layer_weights, draft_scale = build_archetype_representative_case(
            parsed, min_layers=min_layers,
        )
    reduced = (
        len(rep_case.model.layer_plans) < len(parsed.model.layer_plans)
        or draft_scale > 1.0
    )
    graphs, eff_case, _ = _lower_and_run_graph(
        rep_case, phase=phase, tokens=tokens, context=context,
        sequences=sequences, scope=scope, run_graph=False,
    )
    eval_ctx = evaluation_context or EvaluationContext.create(eff_case)
    ext_in = {"analysis_input": 0} if scope == "body" else None
    has_kg = any("device_kernel_group" in dict(op.metadata) for op in graphs[0])
    sg = (
        TargetGraphSet({0: graphs[0]}, kernels=graphs.effective_kernels, selection=graphs.default_selection)
        if isinstance(graphs, TargetGraphSet) else {0: graphs[0]}
    ) if rank_zero_graph_reuse(eff_case.runtime.parallel, len(graphs), has_kg) else graphs
    try:
        graph = graph_costs(
            eff_case, sg, external_inputs=ext_in, isolated=False,
            evaluation=eval_ctx, fast_span_only=(not reduced),
            timeline_replicas=(0,) if reduced else None,
        )
        rep_total = float(graph["duration_ms"])
        timeline_rows = tuple(graph.get("timeline", ()) or ()) if reduced else ()
    finally:
        del graphs, sg
        if evaluation_context is None:
            del eval_ctx
            if len(parsed.machine.dies) >= GC_RECLAIM_MIN_WORLD_SIZE or int(parsed.runtime.parallel.dp) > 1:
                _reclaim_sweep_memory()
    if not reduced:
        return rep_total

    # Track per-(stage, layer) completion & non-root ready times so a t=0 prefetch
    # op inside layer i cannot pull layer i's span back across earlier layers.
    stage_layer_end: dict[int, dict[int, float]] = {}
    stage_layer_dep_start: dict[int, dict[int, float]] = {}
    stage_layer_any_start: dict[int, dict[int, float]] = {}
    draft_dep_start: float | None = None
    draft_any_start: float | None = None
    draft_end: float | None = None
    for occ in timeline_rows:
        if not isinstance(occ, Mapping) or int(occ.get("replica", 0) or 0) != 0:
            continue
        st = float(occ.get("ready_ms", 0.0) or 0.0)
        en = float(occ.get("complete_ms", 0.0) or 0.0)
        if en < st:
            continue
        has_pred = bool(occ.get("predecessors"))
        op_id = str(occ.get("op_id") or "")
        is_draft = op_id.startswith("draft") or ".draft." in op_id
        if is_draft:
            draft_any_start = st if draft_any_start is None else min(draft_any_start, st)
            if has_pred:
                draft_dep_start = st if draft_dep_start is None else min(draft_dep_start, st)
            draft_end = en if draft_end is None else max(draft_end, en)
            continue
        lyr = occ.get("layer")
        if lyr is None or int(lyr) < 0:
            continue
        stage_id = int(occ.get("stage", 0) or 0)
        idx = int(lyr)
        ends = stage_layer_end.setdefault(stage_id, {})
        ends[idx] = en if idx not in ends else max(ends[idx], en)
        starts_any = stage_layer_any_start.setdefault(stage_id, {})
        starts_any[idx] = st if idx not in starts_any else min(starts_any[idx], st)
        if has_pred:
            starts_dep = stage_layer_dep_start.setdefault(stage_id, {})
            starts_dep[idx] = st if idx not in starts_dep else min(starts_dep[idx], st)

    extra_target = 0.0
    max_target_end = 0.0
    for stage_id, ends in stage_layer_end.items():
        prev_end: float | None = None
        for idx in sorted(ends):
            end_ms = ends[idx]
            max_target_end = max(max_target_end, end_ms)
            if prev_end is not None:
                layer_span = max(0.0, end_ms - prev_end)
            else:
                first_start = stage_layer_dep_start.get(stage_id, {}).get(
                    idx, stage_layer_any_start.get(stage_id, {}).get(idx, 0.0),
                )
                layer_span = max(0.0, end_ms - first_start)
            prev_end = max(prev_end, end_ms) if prev_end is not None else end_ms
            extra_target += float(layer_weights.get(idx, 1) - 1) * layer_span

    draft_start = draft_dep_start if draft_dep_start is not None else draft_any_start
    if draft_start is not None and draft_end is not None and draft_start < max_target_end <= draft_end:
        draft_start = max_target_end
    extra_draft = (
        (draft_scale - 1.0) * max(0.0, draft_end - draft_start)
        if (draft_scale > 1.0 and draft_start is not None and draft_end is not None) else 0.0
    )
    return rep_total + extra_target + extra_draft


def _free_hbm_gib_for_batch(memory: Memory, kv_by_stage: Sequence[float], batch: int) -> float:
    remaining = min(
        (float(cap) - float(batch) * float(kv)
         for row in memory.capacity for cap, kv in zip(row, kv_by_stage)),
        default=0.0,
    )
    return max(0.0, remaining) / float(1024 ** 3)


def _prefill_step_ms(
    parsed: CaseSpec,
    *,
    compute_tokens: int,
    hit_tokens: int = 0,
    min_layers: int = ARCHETYPE_FOLD_MIN_LAYERS,
    rep_info: tuple[CaseSpec, dict[int, int], float] | None = None,
    evaluation_context: EvaluationContext | None = None,
) -> float:
    """Compute prefill latency via direct step or chunked-prefill midpoint quadrature."""
    if compute_tokens <= 0:
        return 0.0
    if not parsed.runtime.enable_chunked_prefill or compute_tokens <= PREFILL_DIRECT_MAX_TOKENS:
        return _archetype_weighted_step_ms(
            parsed, phase="prefill", tokens=compute_tokens,
            context=hit_tokens, sequences=1, min_layers=min_layers,
            rep_info=rep_info, evaluation_context=evaluation_context,
        )
    batched_cap = int(parsed.runtime.max_num_batched_tokens or PREFILL_CHUNK_PROBE_TOKENS)
    chunk_tok = max(1, min(int(compute_tokens), batched_cap, PREFILL_CHUNK_PROBE_TOKENS))
    mid_ctx = int(hit_tokens) + max(0, (int(compute_tokens) - chunk_tok) // 2)
    chunk_ms = _archetype_weighted_step_ms(
        parsed, phase="prefill", tokens=chunk_tok,
        context=mid_ctx, sequences=1, min_layers=min_layers,
        rep_info=rep_info, evaluation_context=evaluation_context,
    )
    return chunk_ms * (float(compute_tokens) / float(chunk_tok))


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
) -> dict[str, Any]:
    """Exact SLA concurrency solver using O(Archetypes x 1 Rank) steps."""
    q_tok, emit_est = _resolve_speculative_tokens(
        parsed, output_budget=output_budget, tokens_per_seq=None, emitted_tokens_per_seq=None,
    )
    base_ws = int(workspace_tokens) if workspace_tokens is not None else max(1, int(q_tok))
    d_mem = Memory(parsed, workspace_tokens=base_ws)
    d_kv = d_mem.dp_cohort_cache_bytes(context + output_budget, padded=True)
    cap_bound = int(parsed.runtime.max_num_seqs)
    if max_sequences_per_dp is not None:
        cap_bound = min(cap_bound, int(max_sequences_per_dp))
    b_vram = (
        min([cap_bound] + [
            int(math.floor(max(0.0, float(cap)) / float(kv)))
            for row in d_mem.capacity for cap, kv in zip(row, d_kv) if kv > 0
        ])
        if all(cap >= 0.0 for row in d_mem.capacity for cap in row) else 0
    )
    if b_vram > 1 and int(b_vram * q_tok) > base_ws:
        # Refine workspace to max(base_ws, b_vram * decode_q) so both colocated
        # and pd_decode activation buffers are never underestimated at high batch.
        refined_mem = Memory(parsed, workspace_tokens=max(base_ws, int(b_vram * q_tok)))
        refined_b = (
            min([cap_bound] + [
                int(math.floor(max(0.0, float(cap)) / float(kv)))
                for row in refined_mem.capacity for cap, kv in zip(row, d_kv) if kv > 0
            ])
            if all(cap >= 0.0 for row in refined_mem.capacity for cap in row) else 0
        )
        d_mem, b_vram = refined_mem, min(b_vram, refined_b)

    rep_info = rep_info or build_archetype_representative_case(parsed, min_layers=ARCHETYPE_FOLD_MIN_LAYERS)
    memo: dict[int, float] = {int(k): float(v) for k, v in (probe_cache or {}).items() if int(k) >= 1}

    def eval_b(seq_count: int) -> float:
        key = max(1, int(seq_count))
        if key not in memo:
            memo[key] = _archetype_weighted_step_ms(
                parsed, phase="decode", tokens=max(1, int(q_tok)),
                context=context, sequences=key,
                rep_info=rep_info, evaluation_context=evaluation_context,
            )
        return memo[key]

    if b_vram < 1:
        t1 = eval_b(1)
        return {
            "emitted_tokens_per_seq": float(emit_est),
            "scenarios": [{
                "b_vram_max": b_vram, "b_sla": 0 if t1 > sla_ms + PARETO_EPS else 1,
                "b_feasible": 0, "step_ms_at_b1": t1, "step_ms_feasible": None,
                "free_hbm_headroom_gib": _free_hbm_gib_for_batch(d_mem, d_kv, 0),
            }],
        }
    if b_vram > 1:
        t_hi = eval_b(b_vram)
        if t_hi <= sla_ms + PARETO_EPS:
            return {
                "emitted_tokens_per_seq": float(emit_est),
                "scenarios": [{
                    "b_vram_max": b_vram, "b_sla": b_vram, "b_feasible": b_vram,
                    "step_ms_at_b1": memo.get(1, t_hi), "step_ms_feasible": t_hi,
                    "free_hbm_headroom_gib": _free_hbm_gib_for_batch(d_mem, d_kv, b_vram),
                }],
            }
    t1 = eval_b(1)
    if t1 > sla_ms + PARETO_EPS:
        return {
            "emitted_tokens_per_seq": float(emit_est),
            "scenarios": [{
                "b_vram_max": b_vram, "b_sla": 0, "b_feasible": 0,
                "step_ms_at_b1": t1, "step_ms_feasible": None,
                "free_hbm_headroom_gib": _free_hbm_gib_for_batch(d_mem, d_kv, 0),
            }],
        }
    if b_vram == 1:
        return {
            "emitted_tokens_per_seq": float(emit_est),
            "scenarios": [{
                "b_vram_max": 1, "b_sla": 1, "b_feasible": 1,
                "step_ms_at_b1": t1, "step_ms_feasible": t1,
                "free_hbm_headroom_gib": _free_hbm_gib_for_batch(d_mem, d_kv, 1),
            }],
        }
    lo, hi = 1, b_vram
    best_b, best_t = 1, t1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        t_mid = eval_b(mid)
        if t_mid <= sla_ms + PARETO_EPS:
            lo, best_b, best_t = mid, mid, t_mid
        else:
            hi = mid
    return {
        "emitted_tokens_per_seq": float(emit_est),
        "scenarios": [{
            "b_vram_max": b_vram, "b_sla": best_b, "b_feasible": best_b,
            "step_ms_at_b1": t1, "step_ms_feasible": best_t,
            "free_hbm_headroom_gib": _free_hbm_gib_for_batch(d_mem, d_kv, best_b),
        }],
    }


def _screen_colocated_job(args: tuple[Any, ...]) -> tuple[int, bool, float, dict[int, float]]:
    idx, raw_case, prompt_tokens, output_tokens, ttft_sla_ms, tpot_sla_ms = args
    try:
        parsed = parse_case(raw_case)
        par = parsed.runtime.parallel
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
        d_mem = Memory(parsed, workspace_tokens=prefill_ws)
        d_kv = d_mem.cache_bytes(prompt_tokens + output_tokens, padded=True)
        b_vram = min(
            [int(parsed.runtime.max_num_seqs)]
            + [int(math.floor(max(0.0, float(cap)) / float(kv)))
               for row in d_mem.capacity for cap, kv in zip(row, d_kv) if kv > 0]
        ) if all(cap >= 0.0 for row in d_mem.capacity for cap in row) else 0
        if b_vram < 1:
            return idx, False, -1.0, {}
        rep_info = build_archetype_representative_case(parsed, min_layers=ARCHETYPE_SCREEN_MIN_LAYERS)
        if math.isinf(ttft_sla_ms) and output_tokens > 1:
            p_ms = 0.0
        else:
            p_ms = _prefill_step_ms(
                parsed, compute_tokens=prompt_tokens, hit_tokens=0,
                min_layers=ARCHETYPE_SCREEN_MIN_LAYERS,
                rep_info=rep_info,
            )
            if p_ms > ttft_sla_ms * 1.15:
                return idx, False, -p_ms, {}
        probe_b = min(b_vram, SCREEN_DECODE_BATCH_PROBE)
        d_ms = _archetype_weighted_step_ms(
            parsed, phase="decode", tokens=max(1, int(q_tok)),
            context=prompt_tokens, sequences=probe_b,
            rep_info=rep_info,
        )
        est_tpot = d_ms / max(float(emit_est), PARETO_EPS)
        eff_b = float(probe_b) if est_tpot <= tpot_sla_ms * 1.25 else max(1.0, float(probe_b) * (tpot_sla_ms / max(est_tpot, PARETO_EPS)))
        req_ms = p_ms + (float(max(1, output_tokens - 1)) * d_ms) / max(eff_b * float(emit_est), PARETO_EPS)
        score = (float(par.dp) * MS_PER_S * float(output_tokens)) / max(req_ms, PARETO_EPS)
        return idx, True, score, {int(probe_b): float(d_ms)}
    except (ModelingError, ValueError):
        return idx, False, -1.0, {}


def _eval_colocated_job(args: tuple[Any, ...]) -> tuple[int, dict[str, Any] | None]:
    idx, raw_case, prompt, output, ttft, tpot, mtp_n, max_seq, probe_cache = args
    try:
        row = _evaluate_colocated_candidate(
            raw_case, prompt_tokens=prompt, output_tokens=output,
            ttft_sla_ms=ttft, tpot_sla_ms=tpot, mtp_tokens=mtp_n,
            max_sequences_per_dp=max_seq, probe_cache=probe_cache,
        )
        return idx, row
    except (ModelingError, ValueError):
        return idx, None


def _screen_prefill_job(args: tuple[Any, ...]) -> tuple[int, bool, float]:
    p_idx, raw_p, prompt, compute_tokens, hit_tokens, ttft_sla_ms = args
    try:
        parsed = parse_case(raw_p)
        p_mem = Memory(parsed, workspace_tokens=_prefill_workspace_tokens(parsed, compute_tokens))
        p_kv = p_mem.cache_bytes(prompt + 1, padded=True)
        hbm_ok = bool(
            prompt <= parsed.runtime.max_model_len
            and (parsed.runtime.enable_chunked_prefill or compute_tokens <= parsed.runtime.max_num_batched_tokens)
            and all(cap >= 0.0 and cap >= kv for row in p_mem.capacity for cap, kv in zip(row, p_kv))
        )
        if not hbm_ok:
            return p_idx, False, -1.0
        p_ms = (
            _prefill_step_ms(
                parsed, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
                min_layers=ARCHETYPE_SCREEN_MIN_LAYERS,
            )
            if compute_tokens > 0 else 0.0
        )
        par = parsed.runtime.parallel
        if p_ms > ttft_sla_ms * 1.15:
            return p_idx, False, -p_ms
        rps_per_die = (float(par.dp) * MS_PER_S) / max(p_ms * float(par.world_size), PARETO_EPS)
        return p_idx, True, rps_per_die
    except (ModelingError, ValueError):
        return p_idx, False, -1.0


def _screen_decode_job(args: tuple[Any, ...]) -> tuple[int, bool, float]:
    d_idx, raw_d, prompt, output, tpot_sla_ms = args
    try:
        parsed = parse_case(raw_d)
        if prompt + output > parsed.runtime.max_model_len:
            return d_idx, False, -1.0
        q_tok, emit_est = _resolve_speculative_tokens(
            parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
        )
        d_mem = Memory(parsed, workspace_tokens=max(1, int(q_tok)))
        d_kv = d_mem.cache_bytes(prompt + output, padded=True)
        b_vram = min(
            [int(parsed.runtime.max_num_seqs)]
            + [int(math.floor(max(0.0, float(cap)) / float(kv)))
               for row in d_mem.capacity for cap, kv in zip(row, d_kv) if kv > 0]
        ) if all(cap >= 0.0 for row in d_mem.capacity for cap in row) else 0
        if b_vram < 1:
            return d_idx, False, -1.0
        probe_b = min(b_vram, SCREEN_DECODE_BATCH_PROBE)
        d_ms = _archetype_weighted_step_ms(
            parsed, phase="decode", tokens=max(1, int(q_tok)), context=prompt, sequences=probe_b,
        )
        est_tpot = d_ms / max(float(emit_est), PARETO_EPS)
        par = parsed.runtime.parallel
        scale = 1.0 if est_tpot <= tpot_sla_ms else (tpot_sla_ms / max(est_tpot, PARETO_EPS))
        tps_per_die = (float(probe_b) * scale * float(par.dp) * float(emit_est) * MS_PER_S) / max(d_ms * float(par.world_size), PARETO_EPS)
        return d_idx, True, tps_per_die
    except (ModelingError, ValueError):
        return d_idx, False, -1.0


def _eval_prefill_exact_job(args: tuple[Any, ...]) -> tuple[int, tuple[Memory, bool, int, float] | None]:
    p_idx, raw_p, prompt, compute_tokens, hit_tokens = args
    try:
        parsed = parse_case(raw_p)
        res = _measure_prefill_stage(
            parsed, prompt=prompt, compute_tokens=compute_tokens, hit_tokens=hit_tokens,
            cache=None, cache_id=None,
        )
        return p_idx, res
    except (ModelingError, ValueError):
        return p_idx, None


def _eval_decode_exact_job(args: tuple[Any, ...]) -> tuple[int, float, dict[str, Any] | None]:
    d_idx, raw_d, prompt, output, tpot_sla_ms, max_seq = args
    try:
        parsed = parse_case(raw_d)
        _, d_emit_est = _resolve_speculative_tokens(
            parsed, output_budget=output, tokens_per_seq=None, emitted_tokens_per_seq=None,
        )
        eff_sla_ms = tpot_sla_ms * max(1.0, float(d_emit_est))
        d_sla = _solve_sla_archetype_aware(
            parsed, sla_ms=eff_sla_ms, context=prompt,
            output_budget=output, max_sequences_per_dp=max_seq,
        )
        return d_idx, round(eff_sla_ms, 2), d_sla
    except (ModelingError, ValueError):
        return d_idx, 0.0, None


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
        screened = [
            _screen_colocated_job((idx, raw_c, prompt, output, ttft, tpot))
            for idx, _, raw_c in composed
        ]
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
        _, row = _eval_colocated_job(
            (idx, raw_c, prompt, output, ttft, tpot, mtp_n, max_sequences_per_dp, probe_caches.get(idx))
        )
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
) -> tuple[
    list[tuple[int, int, dict[str, Any]]],
    list[tuple[int, list[tuple[int, int, dict[str, Any]]]]],
    dict[tuple[int, int, int], tuple[Memory, bool, int, float]],
    dict[tuple[int, float], dict[str, Any]],
]:
    """Run single-process Stage-1 archetype screening and Stage-2 cache warm-up for PD stages."""
    hit_tokens_pool = int(math.floor(prompt * float(pool_prefix_hit_ratio))) if "pooled_dram_ssd" in pd_modes else 0
    is_pool_only = len(pd_modes) == 1 and pd_modes[0] == "pooled_dram_ssd"
    base_compute_tok = max(0, prompt - hit_tokens_pool) if is_pool_only else prompt
    base_hit_tok = hit_tokens_pool if is_pool_only else 0

    if layers >= ARCHETYPE_SCREEN_MIN_LAYERS and len(p_cases) > PD_PREFILL_STAGE2_FINALISTS:
        p_screen = [
            _screen_prefill_job((p_idx, raw_p, prompt, base_compute_tok, base_hit_tok, ttft))
            for p_idx, _, raw_p in p_cases
        ]
        p_ok = sorted(((sc, idx) for idx, ok, sc in p_screen if ok), reverse=True)
        keep_p = (
            {idx for _, idx in p_ok[:PD_PREFILL_STAGE2_FINALISTS]} if p_ok
            else {idx for _, idx in sorted(((sc, idx) for idx, _, sc in p_screen), reverse=True)[:1]}
        )
        p_cases = [item for item in p_cases if item[0] in keep_p]

    if layers >= ARCHETYPE_SCREEN_MIN_LAYERS and len(all_d_cases) > PD_DECODE_STAGE2_FINALISTS:
        d_screen = [
            _screen_decode_job((idx, raw_d, prompt, output, tpot))
            for idx, _, raw_d in all_d_cases
        ]
        d_ok = sorted(((sc, idx) for idx, ok, sc in d_screen if ok), reverse=True)
        keep_d = (
            {idx for _, idx in d_ok[:PD_DECODE_STAGE2_FINALISTS]} if d_ok
            else {idx for _, idx in sorted(((sc, idx) for idx, _, sc in d_screen), reverse=True)[:1]}
        )
        d_by_mtp = [(m, [item for item in bucket if item[0] in keep_d]) for m, bucket in d_by_mtp]
        all_d_cases = [item for item in all_d_cases if item[0] in keep_d]

    prefill_cache: dict[tuple[int, int, int], tuple[Memory, bool, int, float]] = {}
    decode_cache: dict[tuple[int, float], dict[str, Any]] = {}
    p_tok_pairs = {(prompt, 0)} if any(m != "pooled_dram_ssd" for m in pd_modes) else set()
    if "pooled_dram_ssd" in pd_modes:
        p_tok_pairs.add((max(0, prompt - hit_tokens_pool), hit_tokens_pool))
    for p_idx, _, raw_p in p_cases:
        for c_tok, h_tok in sorted(p_tok_pairs):
            _, res = _eval_prefill_exact_job((p_idx, raw_p, prompt, c_tok, h_tok))
            if res is not None:
                prefill_cache[(p_idx, c_tok, h_tok)] = res

    for idx, _, raw_d in all_d_cases:
        _, sla_key, d_sla = _eval_decode_exact_job(
            (idx, raw_d, prompt, output, tpot, max_sequences_per_dp)
        )
        if d_sla is not None:
            decode_cache[(idx, sla_key)] = d_sla
    return p_cases, d_by_mtp, prefill_cache, decode_cache


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
    p_cases = [
        (idx, parse_case(raw).runtime.parallel.world_size, raw)
        for idx, topo in enumerate(p_topos)
        for raw in (compose_for(topo, 0, "pd_prefill"),)
        if raw is not None
    ]
    d_by_mtp: list[tuple[int, list[tuple[int, int, dict[str, Any]]]]] = []
    all_d_cases: list[tuple[int, int, dict[str, Any]]] = []
    d_id = 0
    for mtp_n in mtp_grid:
        bucket: list[tuple[int, int, dict[str, Any]]] = []
        for topo in d_topos:
            raw_d = compose_for(topo, mtp_n, "pd_decode")
            if raw_d is not None:
                d_id += 1
                item = (d_id, parse_case(raw_d).runtime.parallel.world_size, raw_d)
                bucket.append(item)
                all_d_cases.append(item)
        d_by_mtp.append((mtp_n, bucket))

    p_cases, d_by_mtp, prefill_cache, decode_cache = _select_and_warm_pd_finalists(
        p_cases, d_by_mtp, all_d_cases, layers=layers, pd_modes=pd_modes,
        prompt=prompt, output=output, ttft=ttft, tpot=tpot,
        pool_prefix_hit_ratio=pool_prefix_hit_ratio,
        max_sequences_per_dp=max_sequences_per_dp,
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
                    d_inst = rep["decode_instance"]
                    row = {
                        "deployment_mode": "pd_disaggregated",
                        "pd_transfer_mode": pd_mode,
                        "mtp_proposal_tokens": mtp_n,
                        "prefill_parallel": rep["prefill_instance"]["parallel"],
                        "decode_parallel": d_inst["parallel"],
                        "pd_ratio": best_ratio["pd_ratio"] if best_ratio else None,
                        "used_dies": int(best_ratio["used_dies"]) if best_ratio else (p_world + d_world),
                        "b_feasible": int(d_inst["b_feasible"]),
                        "b_vram_max": int(d_inst["b_vram_max"]),
                        "ttft_ms": float(rep["prefill_instance"].get("effective_ttft_ms", rep["prefill_instance"]["step_ms"])),
                        "decode_step_ms": float(d_inst["step_ms_feasible"]),
                        "effective_tpot_ms": float(rep["effective_tpot_ms"]),
                        "avg_accepted_tokens": float(d_inst["avg_accepted_tokens"]),
                        "n_tok_per_die": float(d_inst["n_tok_per_die"]),
                        "us_per_tok_die": float(d_inst["us_per_tok_die"]),
                        "unused_sla_slack_ms": float(d_inst["unused_sla_slack_ms"]),
                        "operating_regime": str(d_inst["operating_regime"]),
                        "free_hbm_headroom_gib": float(d_inst["free_hbm_headroom_gib"]),
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
    def rank_key(row: Mapping[str, Any]) -> tuple[float, int, float]:
        return (float(row["sustainable_tps"]), -int(row["used_dies"]), -float(row["effective_tpot_ms"]))

    best_coloc_entry = max(coloc_entries, key=lambda item: rank_key(item[0]), default=None)
    best_pd_entry = max(pd_entries, key=lambda item: rank_key(item[0]), default=None)
    winner_row = recommended_case = prefill_case_out = decode_case_out = None
    if best_pd_entry and (best_coloc_entry is None or rank_key(best_pd_entry[0]) >= rank_key(best_coloc_entry[0])):
        winner_row, prefill_case_out, decode_case_out = (
            best_pd_entry[0], deepcopy(best_pd_entry[1]), deepcopy(best_pd_entry[2]),
        )
        recommended_case = deepcopy(best_pd_entry[2])
    elif best_coloc_entry:
        winner_row, recommended_case = best_coloc_entry[0], deepcopy(best_coloc_entry[1])
    elif all_candidates:
        first_cand = all_candidates[0]
        raw_topo = first_cand.get("parallel") or first_cand.get("decode_parallel") or {}
        fb_topo = {
            ("dcp" if k == "cp" else k): int(raw_topo[k])
            for k in ("dp", "tp", "ep", "pp", "pcp", "cp", "dcp")
            if k in raw_topo and raw_topo[k] is not None
        }
        fb_mtp = int(first_cand.get("mtp_proposal_tokens", 0) or 0)
        fb_role = "pd_decode" if first_cand.get("deployment_mode") == "pd_disaggregated" else "colocated"
        recommended_case = (
            compose_for(fb_topo, fb_mtp, role=fb_role) if fb_topo else None
        ) or compose_for({"dp": 1, "tp": 1, "ep": 1, "pp": 1, "pcp": 1, "dcp": 1}, 0)

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
    from mindie_sim.compose.inputs import _resolve_hardware, _resolve_model_spec
    from mindie_sim.compose.runtime import va_recommended_config

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
        quant=quant, layers=layers, ranks_per_node=ranks_per_node, topology_preset=topology_preset,
        va_recommended=va_recommended, raw_eager=raw_eager, feature_track=feature_track,
        runtime_stack=runtime_stack, optimizations=optimizations,
        exclude_optimizations=exclude_optimizations, engram_cpu_offload=engram_cpu_offload,
        multistream_overlap_shared_expert=multistream_overlap_shared_expert,
        enable_shared_expert_dp=enable_shared_expert_dp, sp=sp,
        graph_capture_mode=graph_capture_mode, kv_cache_dtype=kv_cache_dtype,
        indexer_kv_dtype=indexer_kv_dtype, parallel="fixed", homogeneous_graph_reuse="auto",
    )

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
            model_spec=model_spec, rec_topo=rec_topo, pinned=coloc_pin,
            mtp_grid=mtp_grid, prompt=prompt, output=output, ttft=ttft,
            tpot=tpot, max_sequences_per_dp=max_sequences_per_dp,
        )
        all_candidates.extend(c_rows)
    if dep_norm in {"auto", "pd_disaggregated"} and n_dev >= 2:
        p_rows, pd_entries = _search_pd_deployments(
            compose_for, num_devices=n_dev, is_moe=is_moe, layers=eff_layers,
            model_spec=model_spec, rec_topo=rec_topo, p_pin=p_pin, d_pin=d_pin,
            mtp_grid=mtp_grid, prompt=prompt, output=output, ttft=ttft,
            tpot=tpot, pd_transfer_mode=pd_transfer_mode,
            kv_transfer_bw_gbps=kv_transfer_bw_gbps,
            pool_dram_ratio=pool_dram_ratio, pool_ssd_ratio=pool_ssd_ratio,
            pool_prefix_hit_ratio=pool_prefix_hit_ratio, ddr_bw_gbps=ddr_bw_gbps,
            ssd_bw_gbps=ssd_bw_gbps, max_sequences_per_dp=max_sequences_per_dp,
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
