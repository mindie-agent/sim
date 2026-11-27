"""Macro wall triggers and typed co-design opportunity collection.

[Role] Classify macro deployment walls (``HBM_CAPACITY_WALL``, ``LATENCY_WALL``,
  ``BALANCED``) and assemble typed ``CoDesignOpportunity`` records
  (``MegaFusionBound``, ``SpaceTimeTradeoffSpec``, ``CrossHwOffloadSpec``) from
  canonical ``EngineStepReceipt`` / ``ResidentMemoryReceipt`` / ``FuseRegion`` /
  ``DualTierLoweringReceipt`` / ``KernelSynthesisSpec`` inputs.
[Invariants] Host H2D bandwidth, shared root-complex ceiling, and startup
  latency honor canonical ``resolve_sku_host_dma_rates`` precedence (caller
  explicit device/channel overrides -> supplied/cached L0 ``P20_host_dma_sync``
  -> SKU transfer channels/member-scoped links; zero fabricated fallback rates),
  evaluated per physical die and shared root complex via exact event-driven
  fluid sharing with per-replica paired exposure. Space-time HBM feasibility
  compares every physical die's incremental ledger delta against that same
  die's ``ResidentMemoryReceipt`` headroom and enforces per-request role SLA
  limits without certifying unmeasured queueing or multi-step E2E bounds.
  TileLang dual-tier bounds delegate strictly to
  ``templates.fuse_region.resolve_tilelang_dual_tier_bounds`` or verified
  ``DualTierLoweringReceipt`` outcomes. Shipping readiness reflects verified
  ``FuseRegion.abi_verdict`` or explicit ``K1_shipping`` binding only.
[Timeouts & Locks] Pure bounded in-memory analysis; no I/O, locks, or device calls.
[Side-Effect Policy] Read-only over immutable inputs.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any, Literal

from mindie_sim.catalog.cache import layer_stage_map
from mindie_sim.catalog.cache_pools import LINEAR_ATTENTION_KINDS
from mindie_sim.catalog.context_parallel import (
    dcp_q_replicate_enabled,
    pcp_rank_chunk_ranges,
)
from mindie_sim.catalog.engram import _engram_shard_dims
from mindie_sim.catalog.qwen4_exp import (
    _ple_owned_hits,
    _ple_read_bytes,
    _ple_row_bytes,
    is_ple_engram,
    ple_lookup_format,
)
from mindie_sim.catalog.speculative_state import (
    resolve_speculative_proposals,
    runtime_recover_ssm,
)
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_layer_terms import layer_weight_terms
from mindie_sim.catalog.weight_terms import WeightTerm, term_bytes
from mindie_sim.catalog.weights import _expand_resident_terms, engram_table_terms
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.capacity.ledger import recover_ssm_snapshot_ring_breakdown
from mindie_sim.prediction.offload.policy import resolve_sku_host_dma_rates
from mindie_sim.prediction.operators.kda.family_kda_projection import (
    _resolve_kda_f_quant,
    _validate as _validate_kda_mixed_projection,
)
from mindie_sim.prediction.results import EngineStepReceipt, StageReceipt, StepRequest
from mindie_sim.prediction.sharding import sequence_parallel
from mindie_sim.rewrite.algebra import AlgebraicTrait
from mindie_sim.rewrite.schema import FuseRegion
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.scenario import (
    CoDesignOpportunity,
    CrossHwOffloadSpec,
    MegaFusionBound,
    SpaceTimeTradeoffSpec,
)
from mindie_sim.schema.workload import RequestView
from mindie_sim.templates.contracts import DualTierLoweringReceipt, KernelSynthesisSpec
from mindie_sim.templates.fuse_region import resolve_tilelang_dual_tier_bounds

__all__ = (
    "HBM_WALL_UTIL_PCT",
    "LATENCY_WALL_FREE_GIB",
    "SLA_NEAR_BOUND_RATIO",
    "SLA_SLACK_WALL_MS",
    "MacroWallTrigger",
    "collect_codesign_opportunities",
    "evaluate_macro_wall_trigger",
)

HBM_WALL_UTIL_PCT = 88.0
LATENCY_WALL_FREE_GIB = 8.0
SLA_SLACK_WALL_MS = 3.0
SLA_NEAR_BOUND_RATIO = 0.92
GIB_BYTES = 1024 * 1024 * 1024
MIN_FUSE_MEMBERS = 2
_BF16_ELEMENT_BYTES = int(dtype_bytes("bf16"))
_HOST_LINK_MARKERS = ("h2d", "host", "pcie")
_EXPLICIT_DMA_SOURCES = frozenset({"explicit_device", "explicit_transfer_channel"})
_MLA_PREPARED_TERM_NAMES = frozenset({
    "mlapo_prepared_uk",
    "mlapo_prepared_uk_unpadded",
    "mlapo_prepared_uv",
})
_ENGRAM_RH2D_RECIPES = frozenset({"engram_cpu_decode_pinned_staging_h2d", "ple_host_uva_lookup"})
_ENGRAM_GATHER_RECIPES = frozenset({"catalog_engram_device_gather", "ple_device_lookup"})
_ENGRAM_FAMILIES = frozenset({"engram_rh2d", "engram_gather", "ple", "engram"})
_DEFAULT_ROOT_COMPLEX_ID = "rc0"


@dataclass(frozen=True, slots=True)
class MacroWallTrigger:
    """Macro deployment wall classification for space-time trade-off guidance."""

    wall_kind: Literal["HBM_CAPACITY_WALL", "LATENCY_WALL", "BALANCED"]
    tpot_ms: float
    tpot_slo_ms: float
    sla_slack_ms: float
    hbm_headroom_gib: float
    hbm_utilization_pct: float
    tradeoff_direction: str
    recommended_actions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "wall_kind": self.wall_kind,
            "tpot_ms": round(float(self.tpot_ms), 3),
            "tpot_slo_ms": round(float(self.tpot_slo_ms), 3),
            "sla_slack_ms": round(float(self.sla_slack_ms), 3),
            "hbm_headroom_gib": round(float(self.hbm_headroom_gib), 2),
            "hbm_utilization_pct": round(float(self.hbm_utilization_pct), 1),
            "tradeoff_direction": self.tradeoff_direction,
            "recommended_actions": list(self.recommended_actions),
        }


def evaluate_macro_wall_trigger(
    *,
    tpot_ms: float,
    tpot_slo_ms: float,
    hbm_used_bytes: float,
    hbm_usable_bytes: float,
    hbm_headroom_bytes: float,
) -> MacroWallTrigger:
    """Classify the active macro deployment wall for space-time trade-off guidance."""
    slo = max(float(tpot_slo_ms), 1e-6)
    slack_ms = slo - float(tpot_ms)
    util_pct = (100.0 * float(hbm_used_bytes) / float(hbm_usable_bytes)) if hbm_usable_bytes > 0.0 else 0.0
    headroom_gib = float(hbm_headroom_bytes) / float(GIB_BYTES)

    if slack_ms >= SLA_SLACK_WALL_MS and util_pct >= HBM_WALL_UTIL_PCT:
        return MacroWallTrigger(
            wall_kind="HBM_CAPACITY_WALL",
            tpot_ms=float(tpot_ms),
            tpot_slo_ms=float(tpot_slo_ms),
            sla_slack_ms=slack_ms,
            hbm_headroom_gib=headroom_gib,
            hbm_utilization_pct=util_pct,
            tradeoff_direction="compute_for_memory_and_offload",
            recommended_actions=(
                f"HBM 利用率 {util_pct:.1f}% 且 SLA 余量 {slack_ms:.2f} ms: "
                "优先评估存储换并发（KV 分级池化/状态压缩/并行重分配）",
            ),
        )
    if float(tpot_ms) >= slo * SLA_NEAR_BOUND_RATIO and headroom_gib >= LATENCY_WALL_FREE_GIB:
        return MacroWallTrigger(
            wall_kind="LATENCY_WALL",
            tpot_ms=float(tpot_ms),
            tpot_slo_ms=float(tpot_slo_ms),
            sla_slack_ms=slack_ms,
            hbm_headroom_gib=headroom_gib,
            hbm_utilization_pct=util_pct,
            tradeoff_direction="memory_for_compute",
            recommended_actions=(
                f"TPOT {float(tpot_ms):.2f} ms 逼近 SLA {float(tpot_slo_ms):.2f} ms "
                f"且 HBM 空余 {headroom_gib:.2f} GiB: 优先评估以存换算与片上融合降延迟",
            ),
        )
    return MacroWallTrigger(
        wall_kind="BALANCED",
        tpot_ms=float(tpot_ms),
        tpot_slo_ms=float(tpot_slo_ms),
        sla_slack_ms=slack_ms,
        hbm_headroom_gib=headroom_gib,
        hbm_utilization_pct=util_pct,
        tradeoff_direction="balanced",
        recommended_actions=(
            f"延迟 ({float(tpot_ms):.2f}/{float(tpot_slo_ms):.2f} ms) 与 HBM "
            f"({util_pct:.1f}%, 余量 {headroom_gib:.2f} GiB) 处于均衡区间",
        ),
    )


def _extract_p20_h2d(primitive_table: Any) -> tuple[float | None, float | None, float | None]:
    """Read optional ``(h2d_bps, startup_ns, rc_h2d_bps)`` from supplied ``P20_host_dma_sync``."""
    h2d_bps: float | None = None
    startup_ns: float | None = None
    rc_bps: float | None = None
    cell_fn = getattr(primitive_table, "cell", None)
    if not callable(cell_fn):
        return h2d_bps, startup_ns, rc_bps
    try:
        cell = cell_fn("P20_host_dma_sync")
    except (ModelingError, KeyError):
        return h2d_bps, startup_ns, rc_bps
    try:
        bw_band = cell.band("host_h2d_bytes_per_s")
        if bw_band.p50 is not None and float(bw_band.p50) > 0.0:
            h2d_bps = float(bw_band.p50)
    except (ModelingError, KeyError):
        pass
    for metric_name in ("pinned_h2d_startup_ns", "pinned_dma_startup_ns"):
        if metric_name not in cell.metrics:
            continue
        try:
            st_band = cell.band(metric_name)
            if st_band.p50 is not None and float(st_band.p50) >= 0.0:
                startup_ns = float(st_band.p50)
                break
        except (ModelingError, KeyError):
            continue
    if "host_rc_h2d_bytes_per_s" in cell.metrics:
        try:
            rc_band = cell.band("host_rc_h2d_bytes_per_s")
            val = (
                rc_band.p50
                if (rc_band.p50 is not None and float(rc_band.p50) > 0.0)
                else rc_band.certified_upper_capacity
            )
            if val is not None and float(val) > 0.0:
                rc_bps = float(val)
        except (ModelingError, KeyError):
            pass
    return h2d_bps, startup_ns, rc_bps


def _composed_host_dma_sources(case: Case) -> Mapping[str, Any]:
    exec_pol = dict(getattr(case.runtime, "metadata", ()) or ()).get("execution_policy")
    if isinstance(exec_pol, Mapping):
        sources = exec_pol.get("host_dma_sources")
        if isinstance(sources, Mapping):
            return sources
    return {}


def _is_stamped_default_device_rate(
    target_machine: Any,
    eff_die: Any,
    role: str,
    current_bps: float,
    composed_sources: Mapping[str, Any],
) -> bool:
    """True when ``die.rates[role]`` is an unmodified rate stamped by ``compose_case``."""
    stamped_src = str(composed_sources.get(role) or "")
    if not stamped_src or stamped_src in _EXPLICIT_DMA_SOURCES:
        return False
    stripped_rates = tuple((k, v) for k, v in getattr(eff_die, "rates", ()) if str(k) != role)
    probe_die = replace(eff_die, rates=stripped_rates)
    default_rates = resolve_sku_host_dma_rates(replace(target_machine, dies=(probe_die,)))
    rate_key = "h2d_bps" if role == "h2d" else "host_memory_bps"
    default_bps = default_rates.get(rate_key)
    return (
        default_bps is not None
        and float(default_bps) > 0.0
        and math.isclose(current_bps, float(default_bps), rel_tol=1e-6, abs_tol=1e-6)
    )


def _match_member_host_link(
    case: Case,
    eff_die: Any,
) -> tuple[float | None, float | None]:
    """Resolve ``(rate_bps, startup_ns)`` from host links owned by ``eff_die``."""
    die_rank = int(getattr(eff_die, "rank", 0)) if eff_die is not None else None
    h2d_bps: float | None = None
    startup_ns: float | None = None
    for link in getattr(case.machine, "links", ()) or ():
        lname = str(getattr(link, "name", "") or "").lower()
        lchan = str(getattr(link, "channel", "") or "").lower()
        if not any(tok in lname or tok in lchan for tok in _HOST_LINK_MARKERS):
            continue
        members = tuple(getattr(link, "members", ()) or ())
        if die_rank is not None and members and die_rank not in members:
            continue
        cap_fn = getattr(link, "capacity_at", None)
        rate_bpms = (
            float(cap_fn(1))
            if callable(cap_fn)
            else float(getattr(link, "effective_peak_capacity_bytes_per_ms", 0.0) or 0.0)
        )
        if h2d_bps is None and rate_bpms > 0.0:
            h2d_bps = rate_bpms * 1000.0
        lat_us = getattr(link, "latency_us", None)
        if startup_ns is None and lat_us is not None:
            startup_ns = max(0.0, float(lat_us) * 1000.0)
    return h2d_bps, startup_ns


def _resolve_host_h2d_params(
    case: Case,
    primitive_table: Any,
    *,
    die: Any = None,
) -> tuple[float, float, float | None]:
    """Resolve ``(solo_h2d_bps, startup_ns, rc_h2d_bps)`` honoring canonical DMA precedence."""
    eff_die = die if die is not None else (case.machine.dies[0] if case.machine.dies else None)
    target_machine = (
        replace(case.machine, dies=(eff_die,), sku=(getattr(eff_die, "sku", "") or case.machine.sku))
        if eff_die is not None
        else case.machine
    )
    host_rates = resolve_sku_host_dma_rates(target_machine)
    composed_sources = _composed_host_dma_sources(case)
    p20_bps, p20_startup_ns, p20_rc_bps = _extract_p20_h2d(primitive_table)

    h2d_source = str(host_rates.get("h2d_source") or "missing")
    rate_val = host_rates.get("h2d_bps")
    h2d_is_caller_explicit = (
        h2d_source in _EXPLICIT_DMA_SOURCES
        and rate_val is not None
        and float(rate_val) > 0.0
        and not (
            h2d_source == "explicit_device"
            and eff_die is not None
            and _is_stamped_default_device_rate(
                target_machine, eff_die, "h2d", float(rate_val), composed_sources,
            )
        )
    )
    h2d_bps: float | None = None
    if h2d_is_caller_explicit:
        h2d_bps = float(rate_val)
    elif p20_bps is not None and p20_bps > 0.0:
        h2d_bps = p20_bps
    elif rate_val is not None and float(rate_val) > 0.0:
        h2d_bps = float(rate_val)

    if not h2d_is_caller_explicit and hasattr(target_machine, "host_solo_bw_gbps"):
        solo_gbps = target_machine.host_solo_bw_gbps("h2d")
        if solo_gbps is not None and float(solo_gbps) > 0.0:
            solo_bps = float(solo_gbps) * 1_000_000_000.0
            h2d_bps = solo_bps if h2d_bps is None else min(h2d_bps, solo_bps)

    startup_ns: float | None = p20_startup_ns
    if startup_ns is None and host_rates.get("h2d_latency_us") is not None:
        startup_ns = max(0.0, float(host_rates["h2d_latency_us"]) * 1000.0)

    rc_source = str(host_rates.get("host_memory_source") or "missing")
    rc_val = host_rates.get("host_memory_bps")
    rc_is_caller_explicit = (
        rc_source == "explicit_device"
        and rc_val is not None
        and float(rc_val) > 0.0
        and not (
            eff_die is not None
            and _is_stamped_default_device_rate(
                target_machine, eff_die, "host_memory", float(rc_val), composed_sources,
            )
        )
    )
    rc_bps: float | None = None
    if rc_is_caller_explicit:
        rc_bps = float(rc_val)
    elif p20_rc_bps is not None and p20_rc_bps > 0.0:
        rc_bps = p20_rc_bps
    elif rc_val is not None and float(rc_val) > 0.0:
        rc_bps = float(rc_val)
    if hasattr(target_machine, "host_root_complex_ceiling_gbps"):
        ceil_gbps = target_machine.host_root_complex_ceiling_gbps("h2d")
        if ceil_gbps is not None and float(ceil_gbps) > 0.0:
            ceil_bps = float(ceil_gbps) * 1_000_000_000.0
            rc_bps = ceil_bps if rc_bps is None else min(rc_bps, ceil_bps)

    if h2d_bps is None or startup_ns is None:
        link_bps, link_startup_ns = _match_member_host_link(case, eff_die)
        if h2d_bps is None and link_bps is not None:
            h2d_bps = link_bps
        if startup_ns is None and link_startup_ns is not None:
            startup_ns = link_startup_ns

    if h2d_bps is None or h2d_bps <= 0.0:
        raise ModelingError("codesign_host_h2d_rate_missing")
    return h2d_bps, (startup_ns if startup_ns is not None else 0.0), rc_bps


def _node_attr_str(node: Any, *names: str) -> str:
    for name in names:
        val = getattr(node, name, None)
        if val is not None and str(val).strip():
            return str(val).strip()
    occ = getattr(node, "occurrence", None)
    if occ is not None and "op_id" in names:
        return str(occ.op_id)
    return ""


def _node_meta(node: Any) -> dict[str, Any]:
    spec = getattr(node, "spec", None)
    if spec is not None and getattr(spec, "metadata", None):
        return dict(spec.metadata)
    raw = getattr(node, "metadata", None)
    return dict(raw) if raw else {}


def _node_exposed_us(node: Any) -> float:
    for attr in ("exposed_dur_us", "exposed_us"):
        val = getattr(node, attr, None)
        if val is not None:
            return max(0.0, float(val))
    ns = getattr(node, "exposed_ns", None)
    if ns is not None:
        return max(0.0, float(ns) / 1000.0)
    start = getattr(node, "start_ms", None)
    end = getattr(node, "end_ms", None)
    if start is not None and end is not None:
        return max(0.0, (float(end) - float(start)) * 1000.0)
    return 0.0


def _node_repeat(node: Any) -> int:
    return max(1, int(getattr(node, "repeat_count", 1) or 1))


def _extract_node_replica(node: Any) -> int:
    rep = getattr(node, "replica", None)
    if isinstance(rep, int) and not isinstance(rep, bool) and rep >= 0:
        return int(rep)
    occ = getattr(node, "occurrence", None)
    if occ is not None:
        occ_rep = getattr(occ, "replica", None)
        if isinstance(occ_rep, int) and not isinstance(occ_rep, bool) and occ_rep >= 0:
            return int(occ_rep)
    return 0


def _is_recover_ssm_commit_node(node: Any) -> bool:
    """True strictly for post-accept RecoverSSM commit ops (never the [B, K+1] verify scan)."""
    meta = _node_meta(node)
    stage = str(meta.get("recover_ssm_stage") or "").strip().lower()
    if stage:
        return stage == "commit"
    op_lower = _node_attr_str(node, "op_id").lower()
    rec_lower = _node_attr_str(node, "recipe", "implementation").lower()
    if "verify" in op_lower or "verify" in rec_lower or "recurrent_gated_delta_rule" in rec_lower:
        return False
    return (
        "recover_ssm_commit" in op_lower
        or "recover_ssm_commit" in rec_lower
        or "aclnnrecoverssmstate" in rec_lower
    )


def _stage_bytes_by_pp(
    case: Case,
    target_plans: Sequence[Any],
    per_layer_bytes: Callable[[Any], int],
) -> tuple[int, ...]:
    """Sum per-layer incremental HBM bytes into canonical PP stages via ``layer_stage_map``."""
    pp = max(1, int(case.runtime.parallel.pp))
    stage_map = layer_stage_map(case.model, case.runtime.parallel, case.runtime)
    stage_totals = [0] * pp
    for lp in target_plans:
        idx = int(getattr(lp, "index", 0))
        st = stage_map[idx] if 0 <= idx < len(stage_map) else pp - 1
        stage_totals[st] += int(per_layer_bytes(lp))
    return tuple(stage_totals)


def _node_engram_h2d_payload_bytes(node: Any, case: Case | None = None) -> float:
    """Resolve logical per-die H2D payload bytes for one Engram/PLE lookup or staging node."""
    if _is_engram_rh2d_node(node):
        h2d = float(getattr(node, "h2d_bytes", 0.0) or 0.0)
        if h2d <= 0.0:
            rwork = _node_meta(node).get("resource_work")
            h2d = float(rwork.get("h2d", 0.0)) if isinstance(rwork, Mapping) else 0.0
        if h2d <= 0.0:
            work_items = getattr(node, "work", ()) or ()
            h2d = sum(
                float(w.amount) for w in work_items
                if getattr(w, "unit", "") == "bytes" and "h2d" in str(getattr(w, "resource", "")).lower()
            )
        if h2d <= 0.0:
            h2d = max(
                float(getattr(node, "hbm_write_bytes", 0.0) or 0.0),
                float(getattr(node, "mte3_bytes", 0.0) or 0.0),
                float(getattr(node, "hbm_total_bytes", 0.0) or 0.0),
                float(getattr(node, "hbm_bytes", 0.0) or 0.0),
            )
        return h2d

    meta = _node_meta(node)
    h2d = float(meta.get("logical_table_read_bytes") or 0.0)
    if h2d <= 0.0 and meta.get("response_bytes_per_lookup") and meta.get("lookups"):
        h2d = float(meta["response_bytes_per_lookup"]) * float(meta["lookups"])
    if h2d <= 0.0 and case is not None and getattr(case.model, "engram", None) is not None:
        engram = case.model.engram
        shape = getattr(node, "shape", None) or getattr(getattr(node, "spec", None), "local_shape", None) or ()
        if len(shape) >= 1 and int(shape[0]) > 0:
            if is_ple_engram(engram):
                lookups = max(1, int(shape[0]) * int(engram.lookups_per_token_per_table))
                lookup_format = ple_lookup_format(case.model, case.runtime)
                hits_local = _ple_owned_hits(lookups, case.runtime.parallel.tp)
                return _ple_read_bytes(hits_local, int(engram.width), lookup_format)
            row_b = float(getattr(engram, "row_bytes", 0.0) or 0.0) or float(engram.host_response_bytes)
            return float(int(shape[0])) * row_b
    if h2d <= 0.0:
        h2d = max(
            float(getattr(node, "hbm_read_bytes", 0.0) or 0.0),
            float(getattr(node, "mte2_bytes", 0.0) or 0.0),
            float(getattr(node, "hbm_total_bytes", 0.0) or 0.0),
            float(getattr(node, "hbm_bytes", 0.0) or 0.0),
        )
    return h2d


def _engram_lookup_h2d_bytes(nodes: Sequence[Any], *, case: Case | None = None) -> float:
    """Extract host-link lookup H2D bytes only, excluding device GEMMs."""
    rh2d_nodes = [n for n in nodes if _is_engram_rh2d_node(n)]
    if rh2d_nodes:
        return sum(_node_engram_h2d_payload_bytes(n, case) * _node_repeat(n) for n in rh2d_nodes)
    gather_nodes = [
        n for n in nodes
        if _node_attr_str(n, "family").lower() == "engram_gather"
        or _node_attr_str(n, "recipe", "implementation").lower() in _ENGRAM_GATHER_RECIPES
    ]
    return sum(_node_engram_h2d_payload_bytes(n, case) * _node_repeat(n) for n in gather_nodes)


def _per_replica_sequences(case: Case, step_request: StepRequest | None) -> tuple[int, ...]:
    """Resolve conserved non-negative active sequence count per DP replica."""
    dp = max(1, int(case.runtime.parallel.dp))
    if step_request is not None:
        raw_views = tuple(getattr(step_request, "requests", ()) or ())
        if raw_views:
            active_count = sum(1 for v in raw_views if not bool(getattr(v, "padding", False)))
            return (active_count,) * dp
        raw_seqs = getattr(step_request, "sequences", 1)
        if isinstance(raw_seqs, Sequence) and not isinstance(raw_seqs, (str, bytes)):
            vals = tuple(max(0, int(x)) for x in raw_seqs if x is not None)
            if not vals:
                return (0,) * dp
            return tuple(vals[r] if r < len(vals) else vals[-1] for r in range(dp))
        scalar = max(0, int(raw_seqs)) if raw_seqs is not None else 1
        return (scalar,) * dp
    total_reqs = len(case.workload.requests)
    if total_reqs <= 0:
        return (1,) + (0,) * (dp - 1)
    base, rem = divmod(total_reqs, dp)
    return tuple(base + (1 if r < rem else 0) for r in range(dp))


def _resolve_memory_receipt(
    case: Case,
    step_request: StepRequest | None,
    *,
    memory_receipt: Any = None,
) -> Any:
    """Return authoritative per-die ``ResidentMemoryReceipt`` from receipt or ``Memory``."""
    if memory_receipt is not None and getattr(memory_receipt, "dies", None):
        return memory_receipt
    try:
        mem = Memory(case)
        if step_request is not None:
            phase = str(getattr(step_request, "phase", "decode") or "decode")
            raw_views = tuple(getattr(step_request, "requests", ()) or ())
            if raw_views:
                return mem.resident_memory_receipt(phase=phase, request_views=raw_views)
            ctx = max(0, int(getattr(step_request, "context", 0) or 0))
            q_tok = max(1, int(getattr(step_request, "tokens", 1) or 1))
            per_rep = _per_replica_sequences(case, step_request)
            batch_arg: int | tuple[int, ...] = per_rep[0] if len(set(per_rep)) == 1 else per_rep
            return mem.resident_memory_receipt(
                phase=phase,
                sequence_tokens=max(1, ctx + q_tok),
                batch_per_dp=batch_arg,
                query_tokens=q_tok,
            )
        per_rep = _per_replica_sequences(case, None)
        batch_arg = per_rep[0] if len(set(per_rep)) == 1 else per_rep
        return mem.resident_memory_receipt(
            phase="decode",
            sequence_tokens=1,
            batch_per_dp=batch_arg,
            query_tokens=1,
        )
    except (ModelingError, ValueError, ZeroDivisionError):
        return None


def _resolved_mtp_proposal_tokens(case: Case) -> int:
    """Resolve speculative proposal depth strictly from runtime/execution-policy contract."""
    canonical = resolve_speculative_proposals(case.runtime)
    if canonical > 0:
        return canonical
    meta = dict(getattr(case.runtime, "metadata", ()) or ())
    for key in ("draft_proposal_tokens", "mtp_proposal_tokens"):
        val = meta.get(key)
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return val
    exec_policy = meta.get("execution_policy")
    if isinstance(exec_policy, Mapping):
        val = exec_policy.get("mtp_proposal_tokens")
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return val
    return 0


def _view_verify_width(view: Any, default_phase: str) -> tuple[str, int]:
    """Return ``(phase, active_tokens)`` for one ``RequestView``."""
    v_phase = str(getattr(view, "phase", None) or default_phase or "decode").strip().lower()
    if bool(getattr(view, "padding", False)):
        return v_phase, max(0, int(getattr(view, "q_len", 0) or 0))
    global_q = max(0, int(getattr(view, "global_q_len", 0) or 0))
    local_q = max(1, int(getattr(view, "q_len", 1) or 1))
    prop = max(0, int(getattr(view, "draft_proposal_tokens", 0) or 0))
    if global_q > 0:
        return v_phase, global_q
    if v_phase == "decode" and prop > 0:
        return v_phase, (1 + prop if local_q <= 1 else max(local_q, 1 + prop))
    return v_phase, max(1, local_q + prop)


def _scalar_scheduled_proposal_k(case: Case, step_request: StepRequest) -> int:
    """Match ``analysis_view``: scalar steps only speculate on ``model`` decode with ``tokens == 1 + K``."""
    phase = str(getattr(step_request, "phase", "decode") or "decode").strip().lower()
    scope = str(getattr(step_request, "scope", "body") or "body").strip().lower()
    tok = int(getattr(step_request, "tokens", 1) or 1)
    if phase != "decode" or scope != "model" or tok <= 1:
        return 0
    runtime_k = _resolved_mtp_proposal_tokens(case)
    if runtime_k > 0 and tok == 1 + runtime_k:
        return runtime_k
    return 0


def _resolve_recover_ssm_k_buckets_by_replica(
    case: Case,
    step_request: StepRequest | None,
    *,
    recover_us: float,
) -> tuple[dict[int, dict[int, int]], int, int]:
    """Resolve per-replica ``{k_proposals: active_requests}`` buckets matching scheduled frontend work."""
    dp = max(1, int(case.runtime.parallel.dp))
    runtime_k = _resolved_mtp_proposal_tokens(case)
    if step_request is not None and getattr(step_request, "requests", ()):
        default_phase = str(getattr(step_request, "phase", "decode") or "decode")
        active = tuple(v for v in step_request.requests if not bool(getattr(v, "padding", False)))
        buckets: dict[int, int] = {}
        for view in active:
            v_phase, width = _view_verify_width(view, default_phase)
            if v_phase == "decode" and width > 1:
                k_val = width - 1
                buckets[k_val] = buckets.get(k_val, 0) + 1
        if not buckets and recover_us > 0.0 and active:
            buckets[max(1, runtime_k)] = len(active)
        if not buckets:
            return {}, 0, 0
        return {r: dict(buckets) for r in range(dp)}, sum(buckets.values()), max(buckets)

    if step_request is not None:
        eff_k = _scalar_scheduled_proposal_k(case, step_request)
        if eff_k <= 0 and recover_us > 0.0:
            phase = str(getattr(step_request, "phase", "decode") or "decode").strip().lower()
            tok = int(getattr(step_request, "tokens", 1) or 1)
            eff_k = max(1, tok - 1) if phase == "decode" and tok > 1 else max(1, runtime_k)
        if eff_k <= 0:
            return {}, 0, 0
        per_rep = _per_replica_sequences(case, step_request)
        by_rep = {r: ({eff_k: cnt} if cnt > 0 else {}) for r, cnt in enumerate(per_rep)}
        return by_rep, max(per_rep, default=0), eff_k

    eff_k = runtime_k if runtime_k > 0 else (1 if recover_us > 0.0 else 0)
    if eff_k <= 0:
        return {}, 0, 0
    per_rep = _per_replica_sequences(case, None)
    by_rep = {r: ({eff_k: cnt} if cnt > 0 else {}) for r, cnt in enumerate(per_rep)}
    return by_rep, max(per_rep, default=0), eff_k


def _evaluate_per_die_space_time_hbm(
    dies: Sequence[Any],
    delta_for_die: Callable[[Any], int],
    fallback_delta: int,
) -> tuple[int, int, bool]:
    """Evaluate per-die HBM fit and return ``(delta, headroom)`` from ONE binding physical die."""
    if not dies:
        return int(fallback_delta), 0, bool(int(fallback_delta) <= 0)
    entries = [(d, int(delta_for_die(d)), int(d.free_hbm_headroom_bytes)) for d in dies]
    hbm_ok = all(delta_d <= headroom_d for _, delta_d, headroom_d in entries)
    affected = [e for e in entries if e[1] != 0]
    failing = [e for e in entries if e[1] > e[2]]
    failing_affected = [e for e in affected if e[1] > e[2]]

    def _margin_key(entry: tuple[Any, int, int]) -> tuple[int, int, int, int, int, int]:
        die, delta_d, headroom_d = entry
        return (headroom_d - delta_d, -delta_d, headroom_d, int(die.replica), int(die.stage), int(die.rank))

    pool = failing_affected or failing or affected or entries
    _, rep_delta, rep_headroom = min(pool, key=_margin_key)
    return rep_delta, rep_headroom, hbm_ok


def _min_emitted_decode_tokens_per_seq(case: Case, step_request: StepRequest | None) -> float:
    """Resolve worst-case per-request committed decode tokens per step for TPOT SLA evaluation."""
    acc = float(getattr(case.runtime, "mtp_acceptance_length", 1.0) or 1.0)
    if step_request is not None and getattr(step_request, "requests", ()):
        default_phase = str(getattr(step_request, "phase", "decode") or "decode")
        decode_views = [
            v for v in step_request.requests
            if not bool(getattr(v, "padding", False))
            and str(getattr(v, "phase", None) or default_phase).strip().lower() == "decode"
        ]
        if decode_views:
            per_req: list[float] = []
            for view in decode_views:
                adv = int(getattr(view, "draft_advanced_tokens", 0) or 0)
                if adv > 0:
                    per_req.append(float(adv))
                    continue
                accepted = int(getattr(view, "draft_accepted_tokens", 0) or 0)
                if accepted > 0:
                    per_req.append(float(1 + accepted))
                    continue
                _, width = _view_verify_width(view, default_phase)
                if width > 1 and bool(getattr(view, "draft_verify", True)):
                    per_req.append(min(float(width), max(1.0, acc)))
                else:
                    per_req.append(1.0)
            return max(1.0, min(per_req))

    if step_request is not None:
        k_step = _scalar_scheduled_proposal_k(case, step_request)
        if k_step > 0:
            return min(float(1 + k_step), max(1.0, acc))
        return 1.0

    k_rt = _resolved_mtp_proposal_tokens(case)
    if k_rt > 0:
        return min(float(1 + k_rt), max(1.0, acc))
    return 1.0


def _declared_ttft_boundary(case: Case) -> str:
    meta = dict(getattr(case.runtime, "metadata", ()) or ())
    spine = meta.get("single_spine")
    if isinstance(spine, Mapping):
        scen = spine.get("scenario_contract")
        if isinstance(scen, Mapping):
            sla = scen.get("sla")
            if isinstance(sla, Mapping) and sla.get("ttft_boundary"):
                return str(sla["ttft_boundary"]).strip().lower()
    return "arrival_to_first_token"


def _is_sla_feasible(
    case: Case,
    step_request: StepRequest | None,
    committed_stage: StageReceipt | None,
    delta_step_ms: float,
) -> bool:
    """Evaluate role-specific SLA strictly from verified committed timing (zero unmeasured passes)."""
    slo = getattr(case, "slo", None)
    if slo is None:
        return True
    if step_request is not None:
        phase = str(getattr(step_request, "phase", "decode") or "decode").strip().lower()
    elif committed_stage is not None and committed_stage.step_eval is not None:
        phase = str(committed_stage.step_eval.request.phase).strip().lower()
    else:
        phase = "decode"

    # Multi-step E2E cannot be certified from an isolated step receipt (or missing timing).
    if slo.e2e_ms is not None:
        return False

    if phase == "prefill":
        if slo.ttft_ms is None:
            return True
        if committed_stage is None:
            return False
        if _declared_ttft_boundary(case) != "prefill_completion_only":
            return False
        candidate_ms = max(0.0, float(committed_stage.span_ns) / 1_000_000.0 + float(delta_step_ms))
        return candidate_ms <= float(slo.ttft_ms) + 1e-9

    if slo.tpot_ms is None and slo.itl_ms is None:
        return True
    if committed_stage is None:
        return False
    candidate_ms = max(0.0, float(committed_stage.span_ns) / 1_000_000.0 + float(delta_step_ms))
    if slo.tpot_ms is not None:
        emitted = _min_emitted_decode_tokens_per_seq(case, step_request)
        if (candidate_ms / emitted) > float(slo.tpot_ms) + 1e-9:
            return False
    if slo.itl_ms is not None and candidate_ms > float(slo.itl_ms) + 1e-9:
        return False
    return True


def _recover_ssm_physical_delta_by_die(
    case: Case,
    step_request: StepRequest | None,
    k_buckets_by_rep: Mapping[int, Mapping[int, int]],
) -> tuple[dict[tuple[int, int, int], int] | None, int]:
    """Compute exact per-die HBM delta of switching ``recover_ssm=True -> False`` via ``Memory``."""
    if not runtime_recover_ssm(case.runtime):
        return {}, 0
    alt_meta = tuple(
        (k, (False if k == "recover_ssm" else v))
        for k, v in getattr(case.runtime, "metadata", ())
    )
    alt_case = replace(case, runtime=replace(case.runtime, recover_ssm=False, metadata=alt_meta))
    cur_rec = _resolve_memory_receipt(case, step_request, memory_receipt=None)
    snap_rec = _resolve_memory_receipt(alt_case, step_request, memory_receipt=None)
    if (
        cur_rec is not None
        and snap_rec is not None
        and getattr(cur_rec, "dies", None)
        and len(cur_rec.dies) == len(snap_rec.dies)
    ):
        cur_by_key = {(int(d.replica), int(d.stage), int(d.rank)): d for d in cur_rec.dies}
        deltas: dict[tuple[int, int, int], int] = {}
        for snap_d in snap_rec.dies:
            key = (int(snap_d.replica), int(snap_d.stage), int(snap_d.rank))
            cur_d = cur_by_key.get(key)
            if cur_d is not None:
                deltas[key] = int(snap_d.simultaneous_peak_hbm_bytes) - int(cur_d.simultaneous_peak_hbm_bytes)
        if deltas:
            return deltas, max(deltas.values(), default=0)

    pp = max(1, int(case.runtime.parallel.pp))
    bd_cache: dict[tuple[int, int], tuple[int, ...]] = {}
    stage_delta_by_rep: dict[int, tuple[int, ...]] = {}
    for rep, buckets in k_buckets_by_rep.items():
        totals = [0] * pp
        for k_val, cnt in buckets.items():
            key = (cnt, k_val)
            if key not in bd_cache:
                bd_cache[key] = recover_ssm_snapshot_ring_breakdown(
                    case, cnt, k_val,
                ).stage_delta_vs_recover_ssm_bytes
            for st, val in enumerate(bd_cache[key]):
                if st < pp:
                    totals[st] += int(val)
        stage_delta_by_rep[rep] = tuple(totals)
    fallback_delta = max(
        (val for st_vals in stage_delta_by_rep.values() for val in st_vals),
        default=0,
    )
    per_stage_map = {
        (rep, st, -1): int(val)
        for rep, st_vals in stage_delta_by_rep.items()
        for st, val in enumerate(st_vals)
    }
    return per_stage_map, fallback_delta


def _mla_absorbed_layer_delta_bytes(case: Case, lp: Any) -> int | None:
    """Return incremental placed HBM bytes of absorbed ``kv_b`` UK/UV weights, or ``None`` if absent."""
    terms = layer_weight_terms(lp, case.model)
    kv_b = next((term for term in terms if term.name == "kv_b"), None)
    if kv_b is None:
        return None
    if any(term.name in _MLA_PREPARED_TERM_NAMES for term in terms):
        return 0
    p = case.runtime.parallel
    has_sparse_mla = any(
        bool(getattr(plan, "indexer", False))
        or bool(getattr(plan, "owns_topk", False))
        or int(getattr(plan, "topk_owner_layer_id", -1)) >= 0
        or int(getattr(plan, "index_k_owner_layer_id", -1)) >= 0
        for plan in case.model.layer_plans
    )
    dcp_q_rep = (
        int(p.cp) > 1
        and dcp_q_replicate_enabled(
            case.runtime,
            pcp_factor=int(p.dcp_pcp_factor),
            dp_width=int(p.dcp_dp_width),
            sparse=has_sparse_mla,
        )
    )
    base_quant = case.model.quant or QuantSpec("bf16")
    act_dtype = str(getattr(case.model, "activation_dtype", "") or "bf16").strip() or "bf16"
    pairs = _expand_resident_terms(
        kv_b,
        case.runtime.parallel,
        False,
        dcp_q_replicate=dcp_q_rep,
        model=case.model,
    )
    return int(round(sum(
        term_bytes(
            replace(term, runtime_dtype=act_dtype),
            base_quant,
            placed_parallel,
            case.machine.ranks_per_node,
        )
        for term, placed_parallel in pairs
    )))


def _kda_composed_f_layer_delta_bytes(case: Case, lp: Any) -> int | None:
    """Return incremental HBM bytes of ``F = F_b @ F_a`` iff ``kda_f_a`` & ``kda_f_b`` exist."""
    terms = {term.name: term for term in layer_weight_terms(lp, case.model)}
    if "kda_f_a" not in terms or "kda_f_b" not in terms:
        return None
    if "kda_composed_f" in terms:
        return 0
    f_a, f_b = terms["kda_f_a"], terms["kda_f_b"]
    base_quant = case.model.quant or QuantSpec("bf16")
    raw_quant = f_b.runtime_quant or f_b.quant or base_quant
    resolved_quant = _resolve_kda_f_quant(raw_quant)
    composed_term = WeightTerm(
        name="kda_composed_f",
        rows=f_b.rows,
        cols=f_a.cols,
        category="kda.f",
        shard=f_b.shard,
        count=f_b.count,
        groups=f_b.groups,
        units=f_b.units,
        unit_width=f_b.unit_width,
        source_owned=False,
        quant=resolved_quant,
        runtime_dtype=f_b.runtime_dtype,
        runtime_quant=(resolved_quant if f_b.runtime_quant is not None else None),
        placement_role=f_b.placement_role,
        placement_scope=f_b.placement_scope,
        resident_count=f_b.resident_count,
    )
    return int(round(term_bytes(composed_term, resolved_quant, case.runtime.parallel, case.machine.ranks_per_node)))


def _kda_shipping_admitted(case: Case, kda_plans: Sequence[Any]) -> bool:
    """True iff every eligible KDA layer satisfies ``vllm_ascend_kda_mixed_projection`` ABI."""
    tp = max(1, int(case.runtime.parallel.tp))
    for lp in kda_plans:
        try:
            _validate_kda_mixed_projection(lp, case.model, tp)
        except ModelingError:
            return False
    return True


def _collect_space_time_opportunities(
    case: Case,
    nodes: Sequence[Any],
    *,
    step_request: StepRequest | None,
    committed_stage: StageReceipt | None = None,
    memory_receipt: Any = None,
) -> list[CoDesignOpportunity]:
    """Build typed ``SpaceTimeTradeoffSpec`` records backed by per-die ``ResidentMemoryReceipt``."""
    out: list[CoDesignOpportunity] = []
    receipt = _resolve_memory_receipt(case, step_request, memory_receipt=memory_receipt)
    dies = tuple(getattr(receipt, "dies", ()) or ()) if receipt is not None else ()
    plans = tuple(getattr(case.model, "layer_plans", ()) or ())
    linear_plans = [lp for lp in plans if str(getattr(lp, "attention", "")).lower() in LINEAR_ATTENTION_KINDS]
    recover_us = sum(
        _node_exposed_us(n) * _node_repeat(n)
        for n in nodes
        if _is_recover_ssm_commit_node(n)
    )
    if linear_plans:
        k_buckets_by_rep, title_b, title_k = _resolve_recover_ssm_k_buckets_by_replica(
            case, step_request, recover_us=recover_us,
        )
        if k_buckets_by_rep:
            delta_map, fallback_delta = _recover_ssm_physical_delta_by_die(
                case, step_request, k_buckets_by_rep,
            )

            def _ring_die_delta(die: Any) -> int:
                if not delta_map:
                    return 0
                exact = delta_map.get((int(die.replica), int(die.stage), int(die.rank)))
                if exact is not None:
                    return exact
                return delta_map.get((int(die.replica), int(die.stage), -1), 0)

            delta_b, headroom_b, hbm_ok = _evaluate_per_die_space_time_hbm(
                dies, _ring_die_delta, fallback_delta,
            )
            delta_step_ms = -(recover_us / 1000.0)
            sla_ok = _is_sla_feasible(case, step_request, committed_stage, delta_step_ms)
            direction: Literal["space_for_time", "time_for_space"] = (
                "space_for_time" if delta_b >= 0 else "time_for_space"
            )
            out.append(CoDesignOpportunity(
                opportunity_id="space_time_recover_ssm_snapshot_ring",
                kind="space_time_tradeoff",
                track="framework",
                readiness="proposed_pr",
                title=f"RecoverSSM Snapshot Ring [B={title_b}, K+1={title_k + 1}] vs {len(linear_plans)}-Layer Recompute",
                evidence_refs=("ResidentMemoryLedger", "recover_ssm_snapshot_ring_breakdown"),
                space_time=SpaceTimeTradeoffSpec(
                    direction=direction,
                    mechanism="recover_ssm_snapshot_ring",
                    delta_hbm_bytes=delta_b,
                    free_hbm_headroom_bytes=headroom_b,
                    delta_step_ms=delta_step_ms,
                    delta_concurrency_per_dp=0,
                    delta_tps_per_die=0.0,
                    feasible_under_hbm_and_sla=bool(hbm_ok and sla_ok),
                ),
            ))

    mla_deltas_by_layer: dict[int, int] = {}
    mla_eligible_plans: list[Any] = []
    for lp in plans:
        layer_delta = _mla_absorbed_layer_delta_bytes(case, lp)
        if layer_delta is not None:
            mla_eligible_plans.append(lp)
            mla_deltas_by_layer[int(lp.index)] = layer_delta
    if mla_eligible_plans:
        mla_stage_deltas = _stage_bytes_by_pp(
            case, mla_eligible_plans, lambda lp: mla_deltas_by_layer[int(lp.index)],
        )
        delta_mla, headroom_mla, hbm_mla_ok = _evaluate_per_die_space_time_hbm(
            dies,
            lambda d: mla_stage_deltas[int(d.stage)] if 0 <= int(d.stage) < len(mla_stage_deltas) else 0,
            max(mla_stage_deltas, default=0),
        )
        sla_mla_ok = _is_sla_feasible(case, step_request, committed_stage, 0.0)
        out.append(CoDesignOpportunity(
            opportunity_id="space_time_absorbed_kv_projection",
            kind="space_time_tradeoff",
            track="framework",
            readiness="shipping",
            title=f"Absorbed MLA Projection Weights vs Factorized ({len(mla_eligible_plans)} MLA Layers)",
            evidence_refs=("ResidentMemoryLedger", "layer_weight_terms"),
            space_time=SpaceTimeTradeoffSpec(
                direction="space_for_time",
                mechanism="absorbed_kv_projection",
                delta_hbm_bytes=delta_mla,
                free_hbm_headroom_bytes=headroom_mla,
                delta_step_ms=0.0,
                delta_concurrency_per_dp=0,
                delta_tps_per_die=0.0,
                feasible_under_hbm_and_sla=bool(hbm_mla_ok and sla_mla_ok),
            ),
        ))

    kda_deltas_by_layer: dict[int, int] = {}
    kda_eligible_plans: list[Any] = []
    for lp in plans:
        layer_delta = _kda_composed_f_layer_delta_bytes(case, lp)
        if layer_delta is not None:
            kda_eligible_plans.append(lp)
            kda_deltas_by_layer[int(lp.index)] = layer_delta
    if kda_eligible_plans:
        kda_stage_deltas = _stage_bytes_by_pp(
            case,
            kda_eligible_plans,
            lambda lp: kda_deltas_by_layer[int(lp.index)],
        )
        delta_kda, headroom_kda, hbm_kda_ok = _evaluate_per_die_space_time_hbm(
            dies,
            lambda d: kda_stage_deltas[int(d.stage)] if 0 <= int(d.stage) < len(kda_stage_deltas) else 0,
            max(kda_stage_deltas, default=0),
        )
        sla_kda_ok = _is_sla_feasible(case, step_request, committed_stage, 0.0)
        kda_shipping = _kda_shipping_admitted(case, kda_eligible_plans)
        kda_track: Literal["framework", "kernel_new"] = "framework" if kda_shipping else "kernel_new"
        kda_readiness: Literal["shipping", "synthesized_target"] = (
            "shipping" if kda_shipping else "synthesized_target"
        )
        out.append(CoDesignOpportunity(
            opportunity_id="space_time_prepared_weight_materialization",
            kind="space_time_tradeoff",
            track=kda_track,
            readiness=kda_readiness,
            title=f"Prepared Materialized F = F_b @ F_a vs Factorized ({len(kda_eligible_plans)} KDA Layers)",
            evidence_refs=("ResidentMemoryLedger", "layer_weight_terms"),
            space_time=SpaceTimeTradeoffSpec(
                direction="space_for_time",
                mechanism="prepared_weight_materialization",
                delta_hbm_bytes=delta_kda,
                free_hbm_headroom_bytes=headroom_kda,
                delta_step_ms=0.0,
                delta_concurrency_per_dp=0,
                delta_tps_per_die=0.0,
                feasible_under_hbm_and_sla=bool(hbm_kda_ok and sla_kda_ok),
            ),
        ))
    return out


def _extract_node_layer(node: Any) -> int | None:
    spec = getattr(node, "spec", None)
    lyr = getattr(spec, "layer", None) if spec is not None else getattr(node, "layer", None)
    if isinstance(lyr, int) and not isinstance(lyr, bool) and lyr >= 0:
        return lyr
    op_id = _node_attr_str(node, "op_id")
    for token in op_id.split("."):
        if len(token) >= 2 and token[0] == "l" and token[1:].isdigit():
            return int(token[1:])
    return None


def _extract_node_stage(node: Any, stage_map: Sequence[int]) -> int:
    spec = getattr(node, "spec", None)
    st = getattr(spec, "stage", None) if spec is not None else getattr(node, "stage", None)
    if isinstance(st, int) and not isinstance(st, bool) and st >= 0:
        return st
    occ = getattr(node, "occurrence", None)
    if occ is not None and isinstance(getattr(occ, "stage", None), int) and occ.stage >= 0:
        return int(occ.stage)
    lyr = _extract_node_layer(node)
    if lyr is not None and 0 <= lyr < len(stage_map):
        return int(stage_map[lyr])
    return 0


def _is_engram_rh2d_node(node: Any) -> bool:
    return (
        _node_attr_str(node, "family").lower() == "engram_rh2d"
        or _node_attr_str(node, "recipe", "implementation").lower() in _ENGRAM_RH2D_RECIPES
        or float(getattr(node, "h2d_bytes", 0.0) or 0.0) > 0.0
    )


def _node_interval_ms(node: Any) -> tuple[float, float] | None:
    s = getattr(node, "start_ms", None)
    e = getattr(node, "end_ms", None)
    if isinstance(s, (int, float)) and isinstance(e, (int, float)) and float(e) > float(s):
        return float(s), float(e)
    return None


def _interval_union_ms(intervals: Sequence[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    cur_lo, cur_hi = ordered[0]
    total = 0.0
    for lo, hi in ordered[1:]:
        if lo <= cur_hi:
            if hi > cur_hi:
                cur_hi = hi
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    return total + (cur_hi - cur_lo)


def _compute_engram_hide_window_by_replica(
    nodes: Sequence[Any],
    engram_nodes: Sequence[Any],
    first_k: int,
) -> dict[int, float]:
    """Intersect eligible compute intervals with ``[prefetch_ready, consumer_start]`` per replica."""
    engram_ids = {id(n) for n in engram_nodes}
    by_rep: dict[int, list[Any]] = {}
    for n in nodes:
        by_rep.setdefault(_extract_node_replica(n), []).append(n)
    timed_hides: dict[int, float] = {}
    for rep, rep_nodes in by_rep.items():
        rep_engram = [n for n in rep_nodes if id(n) in engram_ids]
        rep_rh2d = [n for n in rep_engram if _is_engram_rh2d_node(n)]
        rep_cons = [n for n in rep_engram if not _is_engram_rh2d_node(n)] or rep_engram
        cons_starts = [
            float(s) for n in rep_cons
            if isinstance((s := getattr(n, "start_ms", None)), (int, float)) and float(s) >= 0.0
            and (_node_interval_ms(n) is not None or float(s) > 0.0)
        ]
        if not cons_starts:
            cons_starts = [
                float(e) for n in rep_rh2d
                if isinstance((e := getattr(n, "end_ms", None)), (int, float)) and float(e) > 0.0
            ]
        if not cons_starts:
            continue
        win_end = min(cons_starts)
        if rep_rh2d:
            rh2d_ready = [
                float(r) for n in rep_rh2d
                if isinstance((r := getattr(n, "ready_ms", None)), (int, float))
                and not isinstance(r, bool)
                and float(r) >= 0.0
            ]
            rh2d_starts = [
                float(s) for n in rep_rh2d
                if isinstance((s := getattr(n, "start_ms", None)), (int, float)) and float(s) >= 0.0
                and (_node_interval_ms(n) is not None or float(s) > 0.0)
            ]
            win_start = min(rh2d_ready) if rh2d_ready else (min(rh2d_starts) if rh2d_starts else 0.0)
        else:
            win_start = 0.0
        if win_end <= win_start:
            timed_hides[rep] = 0.0
            continue
        clipped: list[tuple[float, float]] = []
        for n in rep_nodes:
            if id(n) in engram_ids:
                continue
            lyr = _extract_node_layer(n)
            if lyr is not None and lyr >= first_k:
                continue
            span = _node_interval_ms(n)
            if span is None:
                continue
            lo = max(span[0], win_start)
            hi = min(span[1], win_end)
            if hi > lo:
                clipped.append((lo, hi))
        timed_hides[rep] = max(0.0, _interval_union_ms(clipped))
    return timed_hides


def _compute_engram_hide_window_ms(
    nodes: Sequence[Any],
    engram_nodes: Sequence[Any],
    first_k: int,
) -> float:
    """Intersect eligible compute intervals with ``[prefetch_ready, consumer_start]`` per occurrence."""
    timed_hides = _compute_engram_hide_window_by_replica(nodes, engram_nodes, first_k)
    if timed_hides:
        return max(0.0, min(timed_hides.values()))
    engram_ids = {id(n) for n in engram_nodes}
    hide_us = 0.0
    for node in nodes:
        if id(node) in engram_ids:
            continue
        lyr = _extract_node_layer(node)
        if lyr is not None and 0 <= lyr < first_k:
            hide_us += _node_exposed_us(node) * _node_repeat(node)
    return max(0.0, hide_us / 1000.0)


@dataclass(frozen=True, slots=True)
class _H2dTransferEvent:
    replica: int
    stage: int
    explicit_rank: int | None
    payload_bytes: float
    interval_ms: tuple[float, float] | None
    repeat_count: int = 1
    total_lookups: int | None = None
    ple_row_bytes: int = 0


def _pcp_local_prefill_tokens(views: Sequence[Any], default_phase: str, pcp: int, pcp_lane: int) -> int:
    """Resolve one PCP lane's local physical token count matching frontend PCP splitting."""
    total = 0
    for view in views:
        v_phase, width = _view_verify_width(view, default_phase)
        if (
            pcp > 1
            and v_phase == "prefill"
            and not bool(getattr(view, "padding", False))
            and not getattr(view, "query_ranges", ())
            and int(getattr(view, "global_q_len", 0) or 0) == 0
            and width > 0
        ):
            ranges = pcp_rank_chunk_ranges(width, pcp, pcp_lane)
            total += sum(hi - lo for lo, hi in ranges)
        else:
            total += width
    return total


def _append_engram_table_events(
    events: list[_H2dTransferEvent],
    *,
    rep: int,
    stage: int,
    tp: int,
    pcp: int,
    engram_cfg: Any,
    is_ple: bool,
    lookup_format: str,
    ple_row_b: int,
    lane_tokens: Sequence[int],
    placement: Any,
    edp: int = 1,
    local_heads: int | None = None,
    edp_slot_tokens: int | None = None,
) -> None:
    """Append projected per-stage or per-rank H2D transfer events for one Engram/PLE table."""
    lpt = int(engram_cfg.lookups_per_token_per_table)
    lane_lookups = tuple(max(1, int(tok) * lpt) for tok in lane_tokens)
    uneven_lanes = len(set(lane_lookups)) > 1
    if is_ple:
        needs_per_rank = placement is not None and (
            uneven_lanes or (tp > 1 and any(lk % tp != 0 for lk in lane_lookups))
        )
        if needs_per_rank:
            ranks = placement.stage_ranks(rep, stage)
            for pcp_idx, lk in enumerate(lane_lookups):
                base_h, rem_h = divmod(lk, tp)
                for tp_idx in range(tp):
                    slot = pcp_idx * tp + tp_idx
                    if slot < len(ranks):
                        hits_i = base_h + (1 if tp_idx < rem_h else 0)
                        events.append(_H2dTransferEvent(
                            rep, stage, int(ranks[slot]), float(hits_i * ple_row_b), None,
                        ))
            return
        hits_local = _ple_owned_hits(lane_lookups[0], tp)
        events.append(_H2dTransferEvent(
            rep, stage, None, _ple_read_bytes(hits_local, int(engram_cfg.width), lookup_format),
            None, total_lookups=lane_lookups[0], ple_row_bytes=ple_row_b,
        ))
        return

    # Canonical Engram lowering (catalog/engram.py:_lower_engram_table) shards
    # lookup heads uniformly across tp * edp and transfers packed rows over H2D.
    eff_edp = max(1, int(edp))
    eff_heads = int(local_heads) if local_heads is not None else max(1, lpt // max(1, tp))
    sharded_lookups = tuple(
        max(
            0,
            (max(int(tok), edp_slot_tokens) if (eff_edp > 1 and edp_slot_tokens is not None) else int(tok))
            * eff_edp * eff_heads,
        )
        for tok in lane_tokens
    )
    row_b = float(getattr(engram_cfg, "row_bytes", 0.0) or 0.0) or float(engram_cfg.host_response_bytes)
    if len(set(sharded_lookups)) > 1 and placement is not None:
        ranks = placement.stage_ranks(rep, stage)
        for pcp_idx, lk in enumerate(sharded_lookups):
            payload = float(lk) * row_b
            if payload <= 0.0:
                continue
            for tp_idx in range(tp):
                slot = pcp_idx * tp + tp_idx
                if slot < len(ranks):
                    events.append(_H2dTransferEvent(rep, stage, int(ranks[slot]), payload, None))
        return
    if sharded_lookups and sharded_lookups[0] > 0:
        events.append(_H2dTransferEvent(rep, stage, None, float(sharded_lookups[0]) * row_b, None))


def _project_step_engram_transfers(
    case: Case,
    step_request: StepRequest | None,
    engram_layers: Sequence[int],
    engram_tables: Sequence[Any],
) -> list[_H2dTransferEvent]:
    """Project per-replica Engram/PLE logical H2D transfer events matching frontend lowering."""
    engram_cfg = getattr(case.model, "engram", None)
    plans = tuple(getattr(case.model, "layer_plans", ()) or ())
    stage_map = layer_stage_map(case.model, case.runtime.parallel, case.runtime)
    dp = max(1, int(case.runtime.parallel.dp))
    tp = max(1, int(case.runtime.parallel.tp))
    pcp = max(1, int(getattr(case.runtime.parallel, "pcp", 1) or 1))
    default_phase = str(getattr(step_request, "phase", "decode") or "decode") if step_request is not None else "decode"
    views_by_rep: dict[int, tuple[RequestView, ...]] = {}
    padded_override = max(0, int(getattr(step_request, "padded_tokens", 0) or 0)) if step_request is not None else 0

    if step_request is not None and getattr(step_request, "requests", ()):
        raw_views = tuple(step_request.requests)
        for r in range(dp):
            views_by_rep[r] = raw_views
    else:
        per_rep = _per_replica_sequences(case, step_request)
        q_tok = max(1, int(getattr(step_request, "tokens", 1) or 1)) if step_request is not None else 1
        ctx = max(0, int(getattr(step_request, "context", 0) or 0)) if step_request is not None else 0
        for r, cnt in enumerate(per_rep):
            views_by_rep[r] = tuple(
                RequestView(f"codesign{r}:{idx}", q_tok, ctx, True, default_phase)
                for idx in range(cnt)
            )

    tok_by_rep_lane: dict[tuple[int, int], int] = {
        (rep, lane): max(padded_override, _pcp_local_prefill_tokens(views, default_phase, pcp, lane))
        for rep, views in views_by_rep.items()
        for lane in range(pcp)
    }
    global_tok = sum(tok_by_rep_lane[(r, 0)] for r in range(dp))
    active_layer_ids = set(engram_layers) or {int(lp.index) for lp in plans}
    active_tables = tuple(t for t in engram_tables if int(t.layer_id) in active_layer_ids)
    is_ple = engram_cfg is not None and is_ple_engram(engram_cfg)
    lookup_format = ple_lookup_format(case.model, case.runtime) if is_ple else ""
    ple_row_b = _ple_row_bytes(int(engram_cfg.width), lookup_format) if is_ple else 0
    edp, local_heads, edp_slot_tokens = 1, None, None
    if engram_cfg is not None and active_tables and not is_ple:
        meta = dict(getattr(case.runtime, "metadata", ()) or ())
        raw_slot = meta.get("engram_edp_slot_tokens")
        edp_slot_tokens = int(raw_slot) if raw_slot is not None else None
        offload_rt = (
            case.runtime if bool(getattr(case.runtime, "engram_cpu_offload", False))
            else replace(case.runtime, engram_cpu_offload=True)
        )
        node_ranks = int(getattr(case.machine, "ranks_per_node", 0) or 0) or None
        _tp_eff, edp, local_heads = _engram_shard_dims(
            engram_cfg, case.runtime.parallel, offload_rt, node_ranks=node_ranks,
        )
    try:
        placement = case.machine.validate_placement(case.runtime.parallel)
    except ModelingError:
        placement = None

    events: list[_H2dTransferEvent] = []
    for rep in range(dp):
        views = views_by_rep[rep]
        tok_lane0 = tok_by_rep_lane[(rep, 0)]
        if tok_lane0 <= 0:
            continue
        sp_active_tokens: int | None = None
        if tp > 1 and views:
            sp = sequence_parallel.select(
                case.runtime, case.model, views,
                tokens=tok_lane0, global_tokens=max(tok_lane0, global_tok),
            )
            if sp is not None and bool(getattr(sp, "stage_entry_active", False)) and int(sp.tokens_local) > 0:
                sp_active_tokens = int(sp.tokens_local)

        if engram_cfg is not None and active_tables:
            lane_tokens = tuple(
                sp_active_tokens if sp_active_tokens is not None else tok_by_rep_lane[(rep, lane)]
                for lane in range(pcp)
            )
            for tbl in active_tables:
                lyr = int(tbl.layer_id)
                st = stage_map[lyr] if 0 <= lyr < len(stage_map) else 0
                _append_engram_table_events(
                    events,
                    rep=rep,
                    stage=st,
                    tp=tp,
                    pcp=pcp,
                    engram_cfg=engram_cfg,
                    is_ple=is_ple,
                    lookup_format=lookup_format,
                    ple_row_b=ple_row_b,
                    lane_tokens=lane_tokens,
                    placement=placement,
                    edp=edp,
                    local_heads=local_heads,
                    edp_slot_tokens=edp_slot_tokens,
                )
        else:
            padded_tokens = sp_active_tokens if sp_active_tokens is not None else tok_lane0
            fallback_bytes = float(padded_tokens * max(1, int(case.model.hidden_size)) * _BF16_ELEMENT_BYTES)
            for lyr in (engram_layers or (0,)):
                st = stage_map[lyr] if 0 <= lyr < len(stage_map) else 0
                events.append(_H2dTransferEvent(rep, st, None, fallback_bytes, None))
    return events


def _extract_engram_h2d_events(
    case: Case,
    engram_nodes: Sequence[Any],
    *,
    step_request: StepRequest | None,
    engram_layers: Sequence[int],
    engram_tables: Sequence[Any],
) -> list[_H2dTransferEvent]:
    """Collect physical H2D transfer events from observed nodes or projected step geometry."""
    rh2d_nodes = [n for n in engram_nodes if _is_engram_rh2d_node(n)]
    selected = rh2d_nodes or [
        n for n in engram_nodes
        if _node_attr_str(n, "family").lower() == "engram_gather"
        or _node_attr_str(n, "recipe", "implementation").lower() in _ENGRAM_GATHER_RECIPES
    ]
    if selected:
        engram_cfg = getattr(case.model, "engram", None)
        is_ple = engram_cfg is not None and is_ple_engram(engram_cfg)
        ple_row_b = _ple_row_bytes(int(engram_cfg.width), ple_lookup_format(case.model, case.runtime)) if is_ple else 0
        stage_map = layer_stage_map(case.model, case.runtime.parallel, case.runtime)
        ranks_by_site: dict[tuple[int, int, str], set[int]] = {}
        for n in selected:
            rep = _extract_node_replica(n)
            st = _extract_node_stage(n, stage_map)
            op_id = _node_attr_str(n, "op_id")
            rk = getattr(n, "rank", None)
            if rk is None and getattr(n, "occurrence", None) is not None:
                rk = getattr(n.occurrence, "rank", None)
            if isinstance(rk, int) and not isinstance(rk, bool) and rk >= 0:
                ranks_by_site.setdefault((rep, st, op_id), set()).add(rk)
        events: list[_H2dTransferEvent] = []
        for n in selected:
            payload = _node_engram_h2d_payload_bytes(n, case)
            if payload <= 0.0:
                continue
            rep = _extract_node_replica(n)
            st = _extract_node_stage(n, stage_map)
            op_id = _node_attr_str(n, "op_id")
            rk = getattr(n, "rank", None)
            if rk is None and getattr(n, "occurrence", None) is not None:
                rk = getattr(n.occurrence, "rank", None)
            explicit_rk = (
                int(rk)
                if (
                    isinstance(rk, int)
                    and not isinstance(rk, bool)
                    and rk >= 0
                    and len(ranks_by_site.get((rep, st, op_id), ())) > 1
                )
                else None
            )
            total_lookups: int | None = None
            if is_ple and engram_cfg is not None:
                shape = getattr(n, "shape", None) or getattr(getattr(n, "spec", None), "local_shape", None) or ()
                if len(shape) >= 1 and int(shape[0]) > 0:
                    total_lookups = max(1, int(shape[0]) * int(engram_cfg.lookups_per_token_per_table))
            events.append(_H2dTransferEvent(
                replica=rep,
                stage=st,
                explicit_rank=explicit_rk,
                payload_bytes=payload,
                interval_ms=_node_interval_ms(n),
                repeat_count=_node_repeat(n),
                total_lookups=total_lookups,
                ple_row_bytes=ple_row_b,
            ))
        if events:
            return events
    return _project_step_engram_transfers(case, step_request, engram_layers, engram_tables)


def _waterfill_rc_rates(solo_by_die: Mapping[int, float], rc_bps: float | None) -> dict[int, float]:
    """Max-min fair per-die H2D rates bounded by each die's solo port and shared RC ceiling."""
    if rc_bps is None or rc_bps <= 0.0 or sum(solo_by_die.values()) <= rc_bps:
        return dict(solo_by_die)
    assigned: dict[int, float] = {}
    remaining = dict(solo_by_die)
    budget = float(rc_bps)
    while remaining:
        share = budget / float(len(remaining))
        capped = {dr: rate for dr, rate in remaining.items() if rate <= share}
        if not capped:
            for dr in remaining:
                assigned[dr] = share
            break
        for dr, rate in capped.items():
            assigned[dr] = rate
            budget -= rate
            del remaining[dr]
    return assigned


@dataclass(slots=True)
class _PhysicalDmaTask:
    release_ms: float
    payload_bytes: float
    startup_ms: float
    solo_bpms: float
    rc_key: tuple[int, str, str]
    replica: int
    seq_idx: int


def _simulate_engram_h2d_transfers(
    case: Case,
    events: Sequence[_H2dTransferEvent],
    *,
    primitive_table: Any,
) -> tuple[float, dict[int, float]]:
    """Simulate concurrent physical-die & shared-RC H2D DMA with exact release, startup, and water-filling."""
    if not events:
        _resolve_host_h2d_params(case, primitive_table)
        return 0.0, {}
    try:
        placement = case.machine.validate_placement(case.runtime.parallel)
    except ModelingError:
        placement = None
    fallback_die = case.machine.dies[0] if case.machine.dies else None
    tp = max(1, int(case.runtime.parallel.tp))
    params_cache: dict[int, tuple[float, float, float | None]] = {}

    def _die_params(die: Any) -> tuple[float, float, float | None]:
        key = int(getattr(die, "rank", 0))
        if key not in params_cache:
            params_cache[key] = _resolve_host_h2d_params(case, primitive_table, die=die)
        return params_cache[key]

    def _expand_event_dies(ev: _H2dTransferEvent) -> list[tuple[Any, float]]:
        if placement is None:
            return [(fallback_die, float(ev.payload_bytes))] if fallback_die is not None else []
        if ev.explicit_rank is not None:
            return [(placement.die_for_rank(ev.explicit_rank), float(ev.payload_bytes))]
        try:
            ranks = placement.stage_ranks(ev.replica, ev.stage)
            dies = tuple(placement.die_for_rank(rk) for rk in ranks)
        except ModelingError:
            return [(fallback_die, float(ev.payload_bytes))] if fallback_die is not None else []
        if ev.total_lookups is not None and ev.ple_row_bytes > 0 and tp > 1:
            base_h, rem_h = divmod(ev.total_lookups, tp)
            return [
                (die, float((base_h + (1 if (idx % tp) < rem_h else 0)) * ev.ple_row_bytes))
                for idx, die in enumerate(dies)
            ]
        return [(die, float(ev.payload_bytes)) for die in dies]

    queues: dict[int, list[_PhysicalDmaTask]] = {}
    rc_ceilings_bpms: dict[tuple[int, str, str], float] = {}
    rep_min_release: dict[int, float] = {}
    seq_counter = 0
    for ev in events:
        release_ms = float(ev.interval_ms[0]) if ev.interval_ms is not None else 0.0
        for die, die_bytes in _expand_event_dies(ev):
            if die is None or die_bytes <= 0.0:
                continue
            solo_bps, startup_ns, rc_bps = _die_params(die)
            dr = int(getattr(die, "rank", 0))
            rc_key = (
                int(getattr(die, "node", 0)),
                str(getattr(die, "root_complex_id", "") or _DEFAULT_ROOT_COMPLEX_ID),
                "h2d",
            )
            if rc_bps is not None and rc_bps > 0.0:
                rc_bpms = rc_bps / 1000.0
                prev_rc = rc_ceilings_bpms.get(rc_key)
                if prev_rc is None or rc_bpms < prev_rc:
                    rc_ceilings_bpms[rc_key] = rc_bpms
            for _ in range(max(1, int(ev.repeat_count))):
                queues.setdefault(dr, []).append(_PhysicalDmaTask(
                    release_ms=release_ms,
                    payload_bytes=die_bytes,
                    startup_ms=startup_ns / 1_000_000.0,
                    solo_bpms=solo_bps / 1000.0,
                    rc_key=rc_key,
                    replica=int(ev.replica),
                    seq_idx=seq_counter,
                ))
                seq_counter += 1
            prev_rel = rep_min_release.get(int(ev.replica))
            if prev_rel is None or release_ms < prev_rel:
                rep_min_release[int(ev.replica)] = release_ms

    if not queues:
        return 0.0, {}
    for dr in queues:
        queues[dr].sort(key=lambda t: (t.release_ms, t.seq_idx))

    overall_start = min(t[0].release_ms for t in queues.values())
    now = overall_start
    startup_state: dict[int, tuple[float, _PhysicalDmaTask]] = {}
    active_state: dict[int, tuple[float, _PhysicalDmaTask]] = {}
    rep_max_finish: dict[int, float] = {}
    overall_finish = overall_start

    while any(queues.values()) or startup_state or active_state:
        tol_now = 1e-12 * max(1.0, abs(now))
        for dr, q in queues.items():
            if dr not in startup_state and dr not in active_state and q and q[0].release_ms <= now + tol_now:
                task = q.pop(0)
                if task.startup_ms > 0.0:
                    startup_state[dr] = (now + task.startup_ms, task)
                else:
                    active_state[dr] = (task.payload_bytes, task)
        for dr in [k for k, (t_end, _) in startup_state.items() if t_end <= now + tol_now]:
            _, task = startup_state.pop(dr)
            active_state[dr] = (task.payload_bytes, task)

        rates_bpms: dict[int, float] = {}
        by_rc: dict[tuple[int, str, str], dict[int, float]] = {}
        for dr, (_, task) in active_state.items():
            by_rc.setdefault(task.rc_key, {})[dr] = task.solo_bpms
        for rc_key, solo_map in by_rc.items():
            rates_bpms.update(_waterfill_rc_rates(solo_map, rc_ceilings_bpms.get(rc_key)))

        finish_times = {dr: now + rem / rates_bpms[dr] for dr, (rem, _) in active_state.items()}
        candidates: list[float] = []
        for dr, q in queues.items():
            if dr not in startup_state and dr not in active_state and q:
                candidates.append(q[0].release_ms)
        candidates.extend(t_end for t_end, _ in startup_state.values())
        candidates.extend(finish_times.values())
        if not candidates:
            break
        t_next = min(candidates)
        tol_next = 1e-12 * max(1.0, abs(t_next))
        dt = max(0.0, t_next - now)
        for dr in list(active_state):
            rem, task = active_state[dr]
            rem -= rates_bpms[dr] * dt
            if finish_times[dr] <= t_next + tol_next or rem <= max(1e-9, task.payload_bytes * 1e-12):
                del active_state[dr]
                overall_finish = max(overall_finish, t_next)
                prev_fin = rep_max_finish.get(task.replica, rep_min_release[task.replica])
                rep_max_finish[task.replica] = max(prev_fin, t_next)
            else:
                active_state[dr] = (rem, task)
        now = t_next

    by_replica = {
        rep: max(0.0, rep_max_finish.get(rep, start) - start)
        for rep, start in rep_min_release.items()
    }
    return max(0.0, overall_finish - overall_start), by_replica


def _collect_cross_hw_offload_opportunities(
    case: Case,
    nodes: Sequence[Any],
    *,
    primitive_table: Any,
    step_request: StepRequest | None = None,
    memory_receipt: Any = None,
) -> list[CoDesignOpportunity]:
    """Build ``CrossHwOffloadSpec`` for Engram/PLE Host DDR async H2D prefetch vs hide window."""
    del memory_receipt
    plans = tuple(getattr(case.model, "layer_plans", ()) or ())
    engram_layers = [int(lp.index) for lp in plans if bool(getattr(lp, "engram", False))]
    engram_cfg = getattr(case.model, "engram", None)
    engram_tables = tuple(getattr(engram_cfg, "tables", ()) or ()) if engram_cfg is not None else ()
    engram_nodes = [
        n for n in nodes
        if _node_attr_str(n, "family").lower() in _ENGRAM_FAMILIES
        or "engram" in _node_attr_str(n, "op_id").lower()
        or "ple_" in _node_attr_str(n, "op_id").lower()
    ]
    if not engram_layers and not engram_tables and not engram_nodes:
        return []

    first_k = (
        min(engram_layers) if engram_layers
        else (min((int(t.layer_id) for t in engram_tables), default=1) if engram_tables else 1)
    )
    events = _extract_engram_h2d_events(
        case,
        engram_nodes,
        step_request=step_request,
        engram_layers=engram_layers,
        engram_tables=engram_tables,
    )
    overall_transfer_ms, transfer_by_rep = _simulate_engram_h2d_transfers(
        case, events, primitive_table=primitive_table,
    )
    hide_by_rep = _compute_engram_hide_window_by_replica(nodes, engram_nodes, first_k)
    if transfer_by_rep and hide_by_rep:
        worst_rep = max(
            transfer_by_rep,
            key=lambda r: (
                max(0.0, transfer_by_rep[r] - hide_by_rep.get(r, 0.0)),
                transfer_by_rep[r],
                -hide_by_rep.get(r, 0.0),
                -r,
            ),
        )
        transfer_ms = transfer_by_rep[worst_rep]
        hide_ms = hide_by_rep.get(worst_rep, 0.0)
    else:
        hide_ms = _compute_engram_hide_window_ms(nodes, engram_nodes, first_k)
        transfer_ms = overall_transfer_ms

    table_bytes = 0
    if engram_cfg is not None:
        quant = case.model.quant or QuantSpec("bf16")
        table_bytes = int(round(sum(
            term_bytes(term, quant, case.runtime.parallel, case.machine.ranks_per_node)
            for term in engram_table_terms(case.model)
        )))
    if table_bytes <= 0 and engram_cfg is not None and callable(getattr(engram_cfg, "table_storage_bytes", None)):
        try:
            table_bytes = int(round(float(engram_cfg.table_storage_bytes())))
        except (ModelingError, ValueError):
            table_bytes = 0
    if table_bytes <= 0:
        for tbl in engram_tables:
            rows = max(1, int(getattr(tbl, "rows", 0) or 1))
            width = max(1, int(getattr(tbl, "width", 0) or case.model.hidden_size))
            table_bytes += rows * width * _BF16_ELEMENT_BYTES
    if table_bytes <= 0:
        max_ev_bytes = max((ev.payload_bytes * ev.repeat_count for ev in events), default=0.0)
        table_bytes = max(
            int(round(max_ev_bytes)),
            len(engram_layers) * max(1, case.model.hidden_size) * _BF16_ELEMENT_BYTES,
        )
    already_offloaded = bool(getattr(case.runtime, "engram_cpu_offload", False))
    freed_hbm = 0 if already_offloaded else table_bytes
    net_exposed_ms = max(0.0, transfer_ms - hide_ms)
    return [CoDesignOpportunity(
        opportunity_id="cross_hw_engram_ple_host_ddr_prefetch",
        kind="cross_hw_offload_multistream",
        track="framework",
        readiness="shipping",
        title=f"Engram/PLE Host DDR Async H2D Prefetch vs NPU Layers 0..{max(0, first_k - 1)} Hide Window",
        evidence_refs=("P20_host_dma_sync",),
        cross_hw=CrossHwOffloadSpec(
            target_component="engram_ple_table",
            source_tier="host_ddr",
            transfer_engine="h2d",
            compute_hide_window_ms=hide_ms,
            transfer_ms=transfer_ms,
            net_exposed_ms=net_exposed_ms,
            freed_hbm_bytes=int(freed_hbm),
            delta_concurrency_per_dp=0,
            delta_tps_per_die=0.0,
        ),
    )]


def _dual_tier_from_receipt(
    receipt: DualTierLoweringReceipt,
) -> tuple[float | None, float | None, str | None]:
    simt_gain_ms: float | None = None
    simd_gain_ms: float | None = None
    ub_warn: str | None = None
    predecessor_ns = int(receipt.predecessor_step_ns)
    for outcome in receipt.tier_outcomes:
        if outcome.ub_receipt is not None and outcome.ub_receipt.ub_wall_warning and ub_warn is None:
            ub_warn = outcome.ub_receipt.ub_wall_warning
        if outcome.tier_kind == "tilelang_simt":
            if outcome.admitted and outcome.step_ns is not None:
                simt_gain_ms = float(predecessor_ns - outcome.step_ns) / 1_000_000.0
            else:
                simt_gain_ms = None
        elif outcome.tier_kind in {"tilelang_simd", "ascendc", "catlass"} and outcome.admitted and outcome.step_ns is not None:
            cand = float(predecessor_ns - outcome.step_ns) / 1_000_000.0
            simd_gain_ms = cand if simd_gain_ms is None else max(simd_gain_ms, cand)
    return simt_gain_ms, simd_gain_ms, ub_warn


def _classify_spec_readiness(
    spec: KernelSynthesisSpec,
    region: FuseRegion | None,
) -> tuple[Literal["framework", "kernel_new"], Literal["shipping", "proposed_pr", "synthesized_target"]]:
    if region is not None and region.abi_verdict is not None:
        status = region.abi_verdict.status
        if status == "admitted_shipping":
            return "framework", "shipping"
        if status == "admitted_proposed_pr":
            return "framework", "proposed_pr"
        return "kernel_new", "synthesized_target"
    if spec.abi_tier == "K1_shipping" and spec.binding_state == "bound_implementation":
        return "framework", "shipping"
    if spec.abi_tier == "K1_proposed_pr" and spec.binding_state == "bound_implementation":
        return "framework", "proposed_pr"
    return "kernel_new", "synthesized_target"


@dataclass(frozen=True, slots=True)
class _ReceiptOpView:
    op_id: str
    family: str
    recipe: str
    dtype: str
    exposed_dur_us: float
    layer: int | None = None
    replica: int = 0
    ready_ms: float | None = None
    start_ms: float = 0.0
    end_ms: float = 0.0
    repeat_count: int = 1
    flops: float = 0.0
    vector_ops: float = 0.0
    hbm_bytes: float = 0.0
    mte2_bytes: float = 0.0
    mte3_bytes: float = 0.0
    h2d_bytes: float = 0.0
    stage: int = 0
    rank: int | None = None
    shape: tuple[int, ...] = ()


def _committed_stage(step_receipt: EngineStepReceipt | None) -> StageReceipt | None:
    if step_receipt is None:
        return None
    for stage in reversed(step_receipt.stages):
        if stage.status == "committed":
            return stage
    return None


def _nodes_from_step_receipt(step_receipt: EngineStepReceipt) -> tuple[_ReceiptOpView, ...]:
    committed: StageReceipt | None = None
    for stage in reversed(step_receipt.stages):
        if stage.status == "committed" and stage.step_eval is not None:
            committed = stage
            break
    if committed is None or committed.step_eval is None:
        return ()
    exposed_map = dict(committed.exposed_ns_by_occ)
    out: list[_ReceiptOpView] = []
    for row in committed.step_eval.operators:
        occ = row.occurrence
        rank = -1 if occ.rank is None else occ.rank
        occ_key = f"r{occ.replica}.p{occ.stage}.d{rank}:{occ.op_id}"
        exp_ns = exposed_map.get(occ_key, exposed_map.get(occ.op_id, 0))
        flops = sum(w.amount for w in row.work if w.unit == "flop")
        vec_ops = sum(w.amount for w in row.work if w.unit == "op")
        hbm_b = sum(w.amount for w in row.work if w.resource in {"memory", "hbm"} and w.unit == "bytes")
        mte2_b = sum(w.amount for w in row.work if w.resource.startswith("mte2") and w.unit == "bytes")
        mte3_b = sum(w.amount for w in row.work if w.resource.startswith("mte3") and w.unit == "bytes")
        h2d_b = sum(w.amount for w in row.work if w.unit == "bytes" and "h2d" in w.resource.lower())
        lyr = None
        for token in str(occ.op_id).split("."):
            if len(token) >= 2 and token[0] == "l" and token[1:].isdigit():
                lyr = int(token[1:])
                break
        out.append(_ReceiptOpView(
            op_id=occ.op_id,
            family=row.family,
            recipe=row.implementation,
            dtype=row.dtype,
            exposed_dur_us=float(exp_ns) / 1000.0,
            layer=lyr,
            replica=int(occ.replica),
            ready_ms=float(row.ready_ms) if row.ready_ms is not None else None,
            start_ms=float(row.start_ms) if row.start_ms is not None else 0.0,
            end_ms=float(row.end_ms) if row.end_ms is not None else 0.0,
            flops=flops,
            vector_ops=vec_ops,
            hbm_bytes=hbm_b,
            mte2_bytes=mte2_b,
            mte3_bytes=mte3_b,
            h2d_bytes=h2d_b,
            stage=int(occ.stage),
            rank=(int(occ.rank) if occ.rank is not None else None),
            shape=tuple(int(x) for x in row.shape),
        ))
    return tuple(out)


def _memory_receipt_from_step(step_receipt: EngineStepReceipt | None) -> Any:
    if step_receipt is None:
        return None
    for stage in reversed(step_receipt.stages):
        if stage.status == "committed" and stage.memory_receipt is not None:
            return stage.memory_receipt
    return None


def collect_codesign_opportunities(
    case: Case,
    nodes: Sequence[Any] = (),
    specs: Sequence[KernelSynthesisSpec] = (),
    tiler_opps: Sequence[CoDesignOpportunity] = (),
    *,
    primitive_table: Any = None,
    step_request: StepRequest | None = None,
    step_receipt: EngineStepReceipt | None = None,
    memory_receipt: Any = None,
    fuse_regions: Sequence[FuseRegion] = (),
    dual_tier_receipts: Sequence[DualTierLoweringReceipt] = (),
) -> tuple[CoDesignOpportunity, ...]:
    """Assemble typed ``MegaFusionBound``, ``SpaceTimeTradeoffSpec`` & ``CrossHwOffloadSpec`` records."""
    committed = _committed_stage(step_receipt)
    eff_step_request = step_request
    if eff_step_request is None and committed is not None and committed.step_eval is not None:
        eff_step_request = committed.step_eval.request
    eff_nodes = tuple(nodes) if nodes else (_nodes_from_step_receipt(step_receipt) if step_receipt is not None else ())
    eff_mem = memory_receipt if memory_receipt is not None else _memory_receipt_from_step(step_receipt)
    datapath = datapath_for_die(case.machine.dies[0])
    default_cube_dtype = str(getattr(case.model, "compute_dtype", "") or "").strip()

    region_by_members: dict[tuple[str, ...], FuseRegion] = {
        tuple(ref.id for ref in reg.members): reg for reg in fuse_regions
    }
    receipt_by_region: dict[str, DualTierLoweringReceipt] = {
        rec.region_id: rec for rec in dual_tier_receipts
    }

    opps: list[CoDesignOpportunity] = []
    for spec in specs:
        if not spec.sram_verified or len(spec.constituent_op_ids) < MIN_FUSE_MEMBERS:
            continue
        exposed_us = spec.exposed_before_us if spec.exposed_before_us is not None else spec.before_us
        before_ms = max(0.0, float(exposed_us) / 1000.0)
        tiled_ms = max(0.0, float(spec.target_us) / 1000.0)
        l_star_ms = max(0.0, float(spec.l_star_us) / 1000.0)
        ub_live = max(0, int(spec.ub_peak_bytes))
        ub_cap = max(0, int(spec.ub_budget_bytes))
        has_cube = AlgebraicTrait.BILINEAR_MATMUL.value in spec.constituent_traits
        has_aiv = any(t != AlgebraicTrait.BILINEAR_MATMUL.value for t in spec.constituent_traits) or not has_cube
        matched_receipt = receipt_by_region.get(spec.spec_id)
        if matched_receipt is not None:
            simt_gain_ms, simd_gain_ms, ub_warn = _dual_tier_from_receipt(matched_receipt)
        else:
            cube_dt = default_cube_dtype if has_cube else ""
            simt_gain_ms, simd_gain_ms, ub_warn = resolve_tilelang_dual_tier_bounds(
                datapath=datapath,
                before_ms=before_ms,
                l_star_ms=l_star_ms,
                tiled_ms=tiled_ms,
                ub_live_bytes=ub_live,
                has_cube=has_cube,
                has_aiv=has_aiv,
                cube_dtype=cube_dt,
                vector_ops=0.0,
                is_vector_or_norm_region=has_aiv,
            )
        matched_region = region_by_members.get(tuple(spec.constituent_op_ids))
        track, readiness = _classify_spec_readiness(spec, matched_region)
        opps.append(CoDesignOpportunity(
            opportunity_id=f"mega_{spec.spec_id}",
            kind="mega_fusion",
            track=track,
            readiness=readiness,
            title=f"{spec.kernel_name} ({spec.suggested_backend})",
            evidence_refs=tuple(spec.constituent_op_ids),
            mega_fusion=MegaFusionBound(
                region_op_ids=tuple(spec.constituent_op_ids),
                exposed_before_ms=before_ms,
                boundary_lower_bound_ms=l_star_ms,
                tiled_engine_ms=tiled_ms,
                gain_ceiling_ms=before_ms - l_star_ms,
                gain_expected_ms=before_ms - tiled_ms,
                eliminated_gm_bytes=max(0, int(spec.elided_hbm_bytes)),
                ub_live_bytes=ub_live,
                ub_capacity_bytes=ub_cap,
                requires_unabsorbed_split_cost_check=True,
                tilelang_simt_gain_ms=simt_gain_ms,
                tilelang_simd_or_ascendc_gain_ms=simd_gain_ms,
                ub_wall_warning=ub_warn or spec.ub_wall_warning,
            ),
        ))

    opps.extend(tiler_opps)
    opps.extend(_collect_space_time_opportunities(
        case,
        eff_nodes,
        step_request=eff_step_request,
        committed_stage=committed,
        memory_receipt=eff_mem,
    ))
    opps.extend(_collect_cross_hw_offload_opportunities(
        case,
        eff_nodes,
        primitive_table=primitive_table,
        step_request=eff_step_request,
        memory_receipt=eff_mem,
    ))
    return tuple(opps)
