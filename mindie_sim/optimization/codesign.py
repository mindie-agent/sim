"""Macro wall triggers and typed co-design opportunity collection.

[Role] Classify macro deployment walls (``HBM_CAPACITY_WALL``, ``LATENCY_WALL``,
  ``BALANCED``) and assemble typed ``CoDesignOpportunity`` records
  (``MegaFusionBound``, ``SpaceTimeTradeoffSpec``, ``CrossHwOffloadSpec``) from
  canonical ``EngineStepReceipt`` / ``ResidentMemoryReceipt`` / ``FuseRegion`` /
  ``DualTierLoweringReceipt`` / ``KernelSynthesisSpec`` inputs.
[Invariants] Host H2D bandwidth and startup latency come strictly from
  ``Case.machine`` hardware channels/links or calibrated L0 ``P20_host_dma_sync``
  (zero fabricated fallback rates). TileLang dual-tier bounds delegate strictly
  to ``templates.fuse_region.resolve_tilelang_dual_tier_bounds`` or verified
  ``DualTierLoweringReceipt`` outcomes. Shipping readiness reflects verified
  ``FuseRegion.abi_verdict`` or explicit ``K1_shipping`` binding only.
[Timeouts & Locks] Pure bounded in-memory analysis; no I/O, locks, or device calls.
[Side-Effect Policy] Read-only over immutable inputs.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.capacity.ledger import recover_ssm_snapshot_ring_breakdown
from mindie_sim.prediction.offload.policy import resolve_sku_host_dma_rates
from mindie_sim.prediction.results import EngineStepReceipt, StageReceipt, StepRequest
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
_LINEAR_ATTENTION_KINDS = frozenset({"kda", "gdn"})
_ENGRAM_RH2D_RECIPES = frozenset({"engram_cpu_decode_pinned_staging_h2d", "ple_host_uva_lookup"})
_ENGRAM_GATHER_RECIPES = frozenset({"catalog_engram_device_gather", "ple_device_lookup"})
_ENGRAM_FAMILIES = frozenset({"engram_rh2d", "engram_gather", "ple", "engram"})


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


def _resolve_host_h2d_params(case: Case, primitive_table: Any) -> tuple[float, float]:
    """Read real H2D bandwidth (B/s) and startup latency (ns) from P20 or ``Case.machine``."""
    h2d_bps: float | None = None
    startup_ns: float | None = None
    cell_fn = getattr(primitive_table, "cell", None)
    if callable(cell_fn):
        try:
            cell = cell_fn("P20_host_dma_sync")
            bw_band = cell.band("host_h2d_bytes_per_s")
            if bw_band.p50 is not None and float(bw_band.p50) > 0.0:
                h2d_bps = float(bw_band.p50)
            for metric_name in ("pinned_h2d_startup_ns", "pinned_dma_startup_ns"):
                if metric_name in cell.metrics:
                    st_band = cell.band(metric_name)
                    if st_band.p50 is not None and float(st_band.p50) >= 0.0:
                        startup_ns = float(st_band.p50)
                        break
        except (ModelingError, KeyError):
            pass

    if h2d_bps is None or startup_ns is None:
        host_rates = resolve_sku_host_dma_rates(case.machine)
        if h2d_bps is None and host_rates.get("h2d_bps") is not None and float(host_rates["h2d_bps"]) > 0.0:
            h2d_bps = float(host_rates["h2d_bps"])
        if startup_ns is None and host_rates.get("h2d_latency_us") is not None:
            startup_ns = max(0.0, float(host_rates["h2d_latency_us"]) * 1000.0)

    if h2d_bps is None or startup_ns is None:
        for link in getattr(case.machine, "links", ()) or ():
            lname = str(getattr(link, "name", "") or "").lower()
            if not any(tok in lname for tok in _HOST_LINK_MARKERS):
                continue
            rate_bpms = float(getattr(link, "effective_peak_capacity_bytes_per_ms", 0.0) or 0.0)
            if h2d_bps is None and rate_bpms > 0.0:
                h2d_bps = rate_bpms * 1000.0
            lat_us = getattr(link, "latency_us", None)
            if startup_ns is None and lat_us is not None:
                startup_ns = max(0.0, float(lat_us) * 1000.0)

    if h2d_bps is None or h2d_bps <= 0.0:
        raise ModelingError("codesign_host_h2d_rate_missing")
    return h2d_bps, (startup_ns if startup_ns is not None else 0.0)


def _peak_step_int(raw: Any, default: int) -> int:
    if isinstance(raw, (tuple, list)):
        vals = [int(x) for x in raw if x is not None]
        return max(vals) if vals else int(default)
    return int(raw) if raw is not None else int(default)


def _peak_sequences_per_dp(step_request: StepRequest | None, case: Case) -> int:
    if step_request is not None:
        return max(1, _peak_step_int(getattr(step_request, "sequences", 1), 1))
    return max(1, len(case.workload.requests))


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


def _max_pp_stage_bytes(case: Case, target_plans: Sequence[Any], per_layer_bytes: Callable[[Any], int]) -> int:
    """Return the bottleneck single-stage weight byte sum across pipeline stages."""
    pp = max(1, int(case.runtime.parallel.pp))
    if pp <= 1:
        return sum(int(per_layer_bytes(lp)) for lp in target_plans)
    total_layers = max(1, int(getattr(case.model, "layers", len(case.model.layer_plans)) or 1))
    custom = tuple(getattr(case.runtime, "pp_layer_partition", ()) or ())
    if len(custom) == pp and sum(int(x) for x in custom) == total_layers:
        counts = [int(x) for x in custom]
    else:
        base, rem = divmod(total_layers, pp)
        counts = [base + (1 if idx < rem else 0) for idx in range(pp)]
    layer_to_stage: dict[int, int] = {}
    cursor = 0
    for st, cnt in enumerate(counts):
        for lyr in range(cursor, cursor + cnt):
            layer_to_stage[lyr] = st
        cursor += cnt
    stage_totals = [0] * pp
    for lp in target_plans:
        st = layer_to_stage.get(int(getattr(lp, "index", 0)), 0)
        stage_totals[st] += int(per_layer_bytes(lp))
    return max(stage_totals, default=0)


def _engram_lookup_h2d_bytes(nodes: Sequence[Any]) -> float:
    """Extract host-link lookup H2D bytes only, excluding device GEMMs."""
    rh2d_nodes = [
        n for n in nodes
        if _node_attr_str(n, "family").lower() == "engram_rh2d"
        or _node_attr_str(n, "recipe", "implementation").lower() in _ENGRAM_RH2D_RECIPES
    ]
    if rh2d_nodes:
        total = 0.0
        for n in rh2d_nodes:
            meta = _node_meta(n)
            rwork = meta.get("resource_work")
            h2d = float(rwork.get("h2d", 0.0)) if isinstance(rwork, Mapping) else 0.0
            if h2d <= 0.0:
                h2d = max(
                    float(getattr(n, "hbm_write_bytes", 0.0) or 0.0),
                    float(getattr(n, "mte3_bytes", 0.0) or 0.0),
                    float(getattr(n, "hbm_total_bytes", 0.0) or 0.0),
                    float(getattr(n, "hbm_bytes", 0.0) or 0.0),
                )
            total += h2d * _node_repeat(n)
        return total

    gather_nodes = [
        n for n in nodes
        if _node_attr_str(n, "family").lower() == "engram_gather"
        or _node_attr_str(n, "recipe", "implementation").lower() in _ENGRAM_GATHER_RECIPES
    ]
    total = 0.0
    for n in gather_nodes:
        meta = _node_meta(n)
        h2d = float(meta.get("logical_table_read_bytes") or 0.0)
        if h2d <= 0.0 and meta.get("response_bytes_per_lookup") and meta.get("lookups"):
            h2d = float(meta["response_bytes_per_lookup"]) * float(meta["lookups"])
        if h2d <= 0.0:
            h2d = max(
                float(getattr(n, "hbm_read_bytes", 0.0) or 0.0),
                float(getattr(n, "mte2_bytes", 0.0) or 0.0),
                float(getattr(n, "hbm_total_bytes", 0.0) or 0.0),
                float(getattr(n, "hbm_bytes", 0.0) or 0.0),
            )
        total += h2d * _node_repeat(n)
    return total


def _resident_free_headroom_bytes(
    case: Case,
    step_request: StepRequest | None,
    *,
    memory_receipt: Any = None,
) -> int:
    if memory_receipt is not None and getattr(memory_receipt, "dies", None):
        return min((int(d.free_hbm_headroom_bytes) for d in memory_receipt.dies), default=0)
    try:
        mem = Memory(case)
        seqs = _peak_sequences_per_dp(step_request, case)
        ctx = max(0, _peak_step_int(getattr(step_request, "context", 0), 0)) if step_request is not None else 0
        q_tok = max(1, _peak_step_int(getattr(step_request, "tokens", 1), 1)) if step_request is not None else 1
        receipt = mem.resident_memory_receipt(
            phase=str(getattr(step_request, "phase", "decode") or "decode"),
            sequence_tokens=(ctx + q_tok) if ctx > 0 else None,
            batch_per_dp=seqs,
            query_tokens=q_tok,
        )
        return min((int(d.free_hbm_headroom_bytes) for d in receipt.dies), default=0)
    except (ModelingError, ValueError, ZeroDivisionError):
        return 0


def _collect_space_time_opportunities(
    case: Case,
    nodes: Sequence[Any],
    *,
    step_request: StepRequest | None,
    memory_receipt: Any = None,
) -> list[CoDesignOpportunity]:
    """Build typed ``SpaceTimeTradeoffSpec`` records backed by ``ResidentMemoryReceipt``."""
    out: list[CoDesignOpportunity] = []
    headroom = _resident_free_headroom_bytes(case, step_request, memory_receipt=memory_receipt)
    meta = dict(getattr(case.runtime, "metadata", ()) or ())
    k_prop = int(meta.get("draft_proposal_tokens", 0) or 0)
    if k_prop <= 0 and step_request is not None and step_request.phase == "decode":
        k_prop = max(0, _peak_step_int(getattr(step_request, "tokens", 1), 1) - 1)
    seqs = _peak_sequences_per_dp(step_request, case)
    plans = tuple(getattr(case.model, "layer_plans", ()) or ())
    linear_plans = [lp for lp in plans if str(getattr(lp, "attention", "")).lower() in _LINEAR_ATTENTION_KINDS]
    recover_us = sum(
        _node_exposed_us(n) * _node_repeat(n)
        for n in nodes
        if _is_recover_ssm_commit_node(n)
    )
    if linear_plans and (k_prop > 0 or recover_us > 0.0):
        eff_k = max(1, k_prop)
        try:
            bd = recover_ssm_snapshot_ring_breakdown(case, seqs, eff_k)
            delta_b = int(bd.delta_hbm_vs_recover_ssm_bytes)
        except (ModelingError, ValueError):
            delta_b = 0
        direction = "space_for_time" if delta_b >= 0 else "time_for_space"
        feasible = bool(delta_b <= headroom) if delta_b > 0 else True
        out.append(CoDesignOpportunity(
            opportunity_id="space_time_recover_ssm_snapshot_ring",
            kind="space_time_tradeoff",
            track="framework",
            readiness="proposed_pr",
            title=f"RecoverSSM Snapshot Ring [B={seqs}, K+1={eff_k + 1}] vs {len(linear_plans)}-Layer Recompute",
            evidence_refs=("ResidentMemoryLedger", "recover_ssm_snapshot_ring_breakdown"),
            space_time=SpaceTimeTradeoffSpec(
                direction=direction,
                mechanism="recover_ssm_snapshot_ring",
                delta_hbm_bytes=delta_b,
                free_hbm_headroom_bytes=headroom,
                delta_step_ms=-(recover_us / 1000.0),
                delta_concurrency_per_dp=0,
                delta_tps_per_die=0.0,
                feasible_under_hbm_and_sla=feasible,
            ),
        ))

    tp = max(1, int(case.runtime.parallel.tp))
    mla_plans = [lp for lp in plans if int(getattr(lp, "kv_lora_rank", 0) or 0) > 0]
    if mla_plans:
        def _mla_absorbed_layer_bytes(lp: Any) -> int:
            kv_r = int(getattr(lp, "kv_lora_rank", 0) or 0)
            n_heads = max(1, int(getattr(lp, "num_heads", 1) or 1))
            v_dim = max(1, int(getattr(lp, "v_head_dim", 0) or getattr(lp, "head_dim", 1) or 1))
            qk_nope = max(1, int(getattr(lp, "qk_nope_head_dim", 0) or getattr(lp, "head_dim", 1) or 1))
            return (kv_r * n_heads * (qk_nope + v_dim) * _BF16_ELEMENT_BYTES) // tp

        prep_bytes = _max_pp_stage_bytes(case, mla_plans, _mla_absorbed_layer_bytes)
        out.append(CoDesignOpportunity(
            opportunity_id="space_time_absorbed_kv_projection",
            kind="space_time_tradeoff",
            track="framework",
            readiness="shipping",
            title=f"Absorbed MLA Projection Weights vs Factorized ({len(mla_plans)} MLA Layers)",
            evidence_refs=("ResidentMemoryLedger",),
            space_time=SpaceTimeTradeoffSpec(
                direction="space_for_time",
                mechanism="absorbed_kv_projection",
                delta_hbm_bytes=int(prep_bytes),
                free_hbm_headroom_bytes=headroom,
                delta_step_ms=0.0,
                delta_concurrency_per_dp=0,
                delta_tps_per_die=0.0,
                feasible_under_hbm_and_sla=bool(prep_bytes <= headroom),
            ),
        ))
    elif linear_plans:
        hidden = max(1, int(getattr(case.model, "hidden_size", 0) or 1))

        def _linear_prep_layer_bytes(lp: Any) -> int:
            lh = max(1, int(getattr(lp, "linear_heads", 1) or 1))
            ld = max(1, int(getattr(lp, "linear_head_dim", 1) or 1))
            return (hidden * lh * ld * _BF16_ELEMENT_BYTES) // tp

        prep_f_bytes = _max_pp_stage_bytes(case, linear_plans, _linear_prep_layer_bytes)
        out.append(CoDesignOpportunity(
            opportunity_id="space_time_prepared_weight_materialization",
            kind="space_time_tradeoff",
            track="framework",
            readiness="shipping",
            title=f"Prepared Materialized F = F_b @ F_a vs Factorized ({len(linear_plans)} Linear Layers)",
            evidence_refs=("ResidentMemoryLedger",),
            space_time=SpaceTimeTradeoffSpec(
                direction="space_for_time",
                mechanism="prepared_weight_materialization",
                delta_hbm_bytes=int(prep_f_bytes),
                free_hbm_headroom_bytes=headroom,
                delta_step_ms=0.0,
                delta_concurrency_per_dp=0,
                delta_tps_per_die=0.0,
                feasible_under_hbm_and_sla=bool(prep_f_bytes <= headroom),
            ),
        ))
    return out


def _collect_cross_hw_offload_opportunities(
    case: Case,
    nodes: Sequence[Any],
    *,
    primitive_table: Any,
) -> list[CoDesignOpportunity]:
    """Build ``CrossHwOffloadSpec`` for Engram/PLE Host DDR async H2D prefetch vs hide window."""
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
    hide_us = 0.0
    for node in nodes:
        spec = getattr(node, "spec", None)
        lyr = getattr(spec, "layer", None) if spec is not None else getattr(node, "layer", None)
        if lyr is not None and 0 <= int(lyr) < first_k:
            hide_us += _node_exposed_us(node) * _node_repeat(node)
    if hide_us <= 0.0:
        hide_us = sum(
            _node_exposed_us(n) * _node_repeat(n)
            for n in nodes if n not in engram_nodes
        ) * 0.5

    xfer_bytes = _engram_lookup_h2d_bytes(engram_nodes)
    freed_hbm = 0
    if engram_cfg is not None and callable(getattr(engram_cfg, "table_storage_bytes", None)):
        try:
            freed_hbm = int(round(float(engram_cfg.table_storage_bytes())))
        except (ModelingError, ValueError):
            freed_hbm = 0
    if freed_hbm <= 0:
        for tbl in engram_tables:
            rows = max(1, int(getattr(tbl, "rows", 0) or 1))
            width = max(1, int(getattr(tbl, "width", 0) or case.model.hidden_size))
            freed_hbm += rows * width * _BF16_ELEMENT_BYTES
    if freed_hbm <= 0:
        freed_hbm = max(int(round(xfer_bytes)), len(engram_layers) * max(1, case.model.hidden_size) * _BF16_ELEMENT_BYTES)
    if xfer_bytes <= 0.0:
        xfer_bytes = float(max(1, len(engram_layers)) * max(1, case.model.hidden_size) * _BF16_ELEMENT_BYTES)

    h2d_bps, startup_ns = _resolve_host_h2d_params(case, primitive_table)
    transfer_ms = (startup_ns / 1_000_000.0) + (xfer_bytes / h2d_bps) * 1000.0
    hide_ms = max(0.0, hide_us / 1000.0)
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
    for outcome in receipt.outcomes:
        if outcome.ub_band_diagnostic and ub_warn is None:
            ub_warn = outcome.ub_band_diagnostic
        if outcome.tier == "tilelang_simt":
            if outcome.admitted and outcome.signed_delta_ns is not None:
                simt_gain_ms = float(outcome.signed_delta_ns) / 1_000_000.0
            else:
                simt_gain_ms = 0.0
        elif outcome.tier in {"tilelang_simd", "ascendc"} and outcome.admitted and outcome.signed_delta_ns is not None:
            cand = float(outcome.signed_delta_ns) / 1_000_000.0
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
    repeat_count: int = 1
    flops: float = 0.0
    vector_ops: float = 0.0
    hbm_bytes: float = 0.0
    mte2_bytes: float = 0.0
    mte3_bytes: float = 0.0


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
        out.append(_ReceiptOpView(
            op_id=occ.op_id,
            family=row.family,
            recipe=row.implementation,
            dtype=row.dtype,
            exposed_dur_us=float(exp_ns) / 1000.0,
            flops=flops,
            vector_ops=vec_ops,
            hbm_bytes=hbm_b,
            mte2_bytes=mte2_b,
            mte3_bytes=mte3_b,
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
    eff_nodes = tuple(nodes) if nodes else (_nodes_from_step_receipt(step_receipt) if step_receipt is not None else ())
    eff_mem = memory_receipt if memory_receipt is not None else _memory_receipt_from_step(step_receipt)
    datapath = datapath_for_die(case.machine.dies[0])
    default_cube_dtype = str(getattr(case.model, "compute_dtype", "") or "").strip()

    node_by_spec_id: dict[str, Any] = {}
    for n in eff_nodes:
        syn = getattr(n, "synthesis_spec", None)
        if syn is not None and getattr(syn, "spec_id", ""):
            node_by_spec_id[str(syn.spec_id)] = n
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
        l_star_ms = min(tiled_ms, max(0.0, float(spec.l_star_us) / 1000.0))
        ub_live = max(0, int(spec.ub_peak_bytes))
        ub_cap = max(int(spec.ub_budget_bytes), ub_live)
        matched = node_by_spec_id.get(spec.spec_id)
        has_cube = (
            AlgebraicTrait.BILINEAR_MATMUL.value in spec.constituent_traits
            or (matched is not None and float(getattr(matched, "flops", 0.0) or 0.0) > 0.0)
        )
        has_aiv = any(t != AlgebraicTrait.BILINEAR_MATMUL.value for t in spec.constituent_traits) or not has_cube
        matched_receipt = receipt_by_region.get(spec.spec_id)
        if matched_receipt is not None:
            simt_gain_ms, simd_gain_ms, ub_warn = _dual_tier_from_receipt(matched_receipt)
        else:
            cube_dt = str(getattr(matched, "dtype", "") or default_cube_dtype).strip() if has_cube else ""
            simt_gain_ms, simd_gain_ms, ub_warn = resolve_tilelang_dual_tier_bounds(
                datapath=datapath,
                before_ms=before_ms,
                l_star_ms=l_star_ms,
                tiled_ms=tiled_ms,
                ub_live_bytes=ub_live,
                has_cube=has_cube,
                has_aiv=has_aiv,
                cube_dtype=cube_dt,
                vector_ops=float(getattr(matched, "vector_ops", 0.0) or 0.0) if matched is not None else 0.0,
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
    opps.extend(_collect_space_time_opportunities(case, eff_nodes, step_request=step_request, memory_receipt=eff_mem))
    opps.extend(_collect_cross_hw_offload_opportunities(case, eff_nodes, primitive_table=primitive_table))
    return tuple(opps)
