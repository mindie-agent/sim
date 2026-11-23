"""Immutable scenario, deployment-mode, SLA, stage, and co-design contracts.

[Role] Define the single-spine L0 scenario boundary: explicit ``DeploymentMode``
  (zero silent ``auto`` guessing), ``OptimizationMode`` (``simulation`` vs
  ``modeling``), 4-stage / 3-owner ``ExecutionStage`` causal chain,
  ``SLAContract``, ``ConcurrencyLimits``, ``ReadyStateAssumption``, ``PoolSpec``,
  ``DeploymentOutcome``, ``ScenarioContract``, ``CoDesignOpportunity``
  (``MegaFusionBound``, ``SpaceTimeTradeoffSpec``, ``CrossHwOffloadSpec``),
  ``validate_scenario_contract``, and domain-aware
  ``resolve_default_stack_for_domain``.
[Invariants] Zero inference-engine vendor names or imports reside in this
  module. Missing or ambiguous deployment modes raise crisp ``ModelingError``
  listing ``{colocated, pd_disaggregated, pd_prefill, pd_decode}``. Nullable
  ``ttft_ms`` / ``tpot_ms`` are never silently coerced to numeric defaults.
  ``pd_decode`` requires an explicit ``ReadyStateAssumption`` and forbids
  prompt-arrival TTFT; ``pd_prefill`` forbids decode TPOT/ITL. ``simulation``
  mode forbids ``synthesized_target`` / ``tiler_extension`` rewrites.
[Timeouts & Locks] Pure in-memory validation and hashing; no I/O or locks.
[Side-Effect Policy] Strictly frozen dataclasses; zero input mutation.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from typing import Any, Literal

from .checkpoint import CheckpointQuantMap
from .common import ModelingError
from .hardware import Hardware
from .model import ModelSpec, SloSpec, WorkloadSpec
from .operators import KERNEL_READINESS_VALUES, KernelReadiness
from .parallel import ParallelSpec
from .stack import (
    REWRITE_TRACKS,
    STACK_TRACKS,
    WORKLOAD_DOMAINS,
    WORK_UNIT_KINDS,
    PdKvTransferSpec,
    ResolvedStackOptions,
    RewriteTrack,
    RuntimeQuantPlan,
    StackCapabilitySpec,
    StackLoweringPlan,
    StackTrack,
    WorkUnitKind,
    WorkloadDomain,
)

DeploymentMode = Literal["colocated", "pd_disaggregated", "pd_prefill", "pd_decode"]
DEPLOYMENT_MODES = frozenset({"colocated", "pd_disaggregated", "pd_prefill", "pd_decode"})
DEPLOYMENT_MODE_LIST_TEXT = "{colocated, pd_disaggregated, pd_prefill, pd_decode}"

DeploymentModeOrigin = Literal[
    "cli_flag", "api_argument", "launch_script", "comparison_branch",
]
DEPLOYMENT_MODE_ORIGINS = frozenset({
    "cli_flag", "api_argument", "launch_script", "comparison_branch",
})

OptimizationMode = Literal["simulation", "modeling"]
OPTIMIZATION_MODES = frozenset({"simulation", "modeling"})

ExecutionStage = Literal["before", "config_only", "framework_rewrite", "after"]
EXECUTION_STAGES: tuple[ExecutionStage, ...] = (
    "before", "config_only", "framework_rewrite", "after",
)

StageOwner = Literal["baseline", "ops_config", "framework_python", "kernel_dev_tune"]
STAGE_OWNERS = frozenset({"baseline", "ops_config", "framework_python", "kernel_dev_tune"})

OutcomeStatus = Literal["feasible", "infeasible", "unmodeled", "budget_incomplete"]
OUTCOME_STATUSES = frozenset({"feasible", "infeasible", "unmodeled", "budget_incomplete"})

ReadyStateSource = Literal["pre_populated_hbm", "pooled_dram_restore", "pooled_ssd_restore"]
READY_STATE_SOURCES = frozenset({
    "pre_populated_hbm", "pooled_dram_restore", "pooled_ssd_restore",
})

VisionAccountingScope = Literal["text_backbone", "full_checkpoint"]
VISION_ACCOUNTING_SCOPES = frozenset({"text_backbone", "full_checkpoint"})

PoolRole = Literal["colocated", "prefill", "decode"]
POOL_ROLES = frozenset({"colocated", "prefill", "decode"})

CoDesignKind = Literal[
    "mega_fusion", "space_time_tradeoff", "cross_hw_offload_multistream", "tiler_tune",
]
CO_DESIGN_KINDS = frozenset({
    "mega_fusion", "space_time_tradeoff", "cross_hw_offload_multistream", "tiler_tune",
})

SpaceTimeDirection = Literal["space_for_time", "time_for_space"]
SpaceTimeMechanism = Literal[
    "recover_ssm_snapshot_ring",
    "prepared_weight_materialization",
    "absorbed_kv_projection",
    "activation_recompute",
    "custom",
]
SPACE_TIME_MECHANISMS = frozenset({
    "recover_ssm_snapshot_ring",
    "prepared_weight_materialization",
    "absorbed_kv_projection",
    "activation_recompute",
    "custom",
})

OffloadSourceTier = Literal["host_ddr", "ssd", "remote_hbm", "local_hbm"]
OFFLOAD_SOURCE_TIERS = frozenset({"host_ddr", "ssd", "remote_hbm", "local_hbm"})

OffloadTransferEngine = Literal["h2d", "d2h", "sdma", "hccl", "aic_aiv_dual_stream"]
OFFLOAD_TRANSFER_ENGINES = frozenset({"h2d", "d2h", "sdma", "hccl", "aic_aiv_dual_stream"})

SIMULATION_ALLOWED_READINESS = frozenset({"shipping", "proposed_pr"})
DOMAIN_WORK_UNITS: dict[WorkloadDomain, frozenset[WorkUnitKind]] = {
    "llm": frozenset({"prefill_decode_step"}),
    "vlm": frozenset({"prefill_decode_step", "vision_encode"}),
    "diffusion": frozenset({"denoise_timestep", "vae_decode", "text_encode"}),
    "custom": frozenset(WORK_UNIT_KINDS),
}
FLOAT_TOL = 1e-9
MIN_DISAGGREGATED_DEVICES = 2


def _req_str(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(code)
    return value.strip()


def _opt_pos_float(value: Any, code: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ModelingError(f"{code}:{value!r}")
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelingError(f"{code}:{value!r}") from exc
    if not math.isfinite(num) or num <= 0.0:
        raise ModelingError(f"{code}:{value!r}")
    return num


def _finite_float(value: Any, code: str, *, allow_negative: bool = False) -> float:
    if isinstance(value, bool):
        raise ModelingError(f"{code}:{value!r}")
    try:
        num = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelingError(f"{code}:{value!r}") from exc
    if not math.isfinite(num) or (not allow_negative and num < 0.0):
        raise ModelingError(f"{code}:{value!r}")
    return num


def _int_val(value: Any, code: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelingError(f"{code}:{value!r}")
    if minimum is not None and value < minimum:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


def _opt_pos_int(value: Any, code: str) -> int | None:
    if value is None:
        return None
    return _int_val(value, code, minimum=1)


def validate_deployment_mode(mode: Any) -> DeploymentMode:
    """Validate an explicit deployment mode; reject missing, ``auto``, or unknown values."""
    if not isinstance(mode, str) or not mode.strip():
        raise ModelingError(
            f"deployment_mode_required: explicit --deployment-mode must be one of "
            f"{DEPLOYMENT_MODE_LIST_TEXT}; required inputs: "
            "colocated (single pool), pd_disaggregated (disjoint P/D pools or split budget + transfer), "
            "pd_prefill (P pool, no decode TPOT/ITL), pd_decode (D pool + ReadyStateAssumption, TTFT=None)"
        )
    normalized = mode.strip().lower()
    if normalized not in DEPLOYMENT_MODES:
        raise ModelingError(
            f"deployment_mode_invalid:{mode!r}: silent guessing/auto mode is forbidden; "
            f"specify explicit --deployment-mode in {DEPLOYMENT_MODE_LIST_TEXT}"
        )
    return normalized  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class PoolSpec:
    """One physical device pool bound to a serving role."""

    pool_id: str
    role: PoolRole
    num_devices: int
    device_ids: tuple[int, ...] = ()
    parallel: ParallelSpec | None = None
    speculative_k: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "pool_id", _req_str(self.pool_id, "pool_id_required"))
        if self.role not in POOL_ROLES:
            raise ModelingError(f"pool_role_invalid:{self.role!r}")
        num = _int_val(self.num_devices, "pool_num_devices_invalid", minimum=1)
        object.__setattr__(self, "num_devices", num)
        ids = tuple(_int_val(d, "pool_device_id_invalid", minimum=0) for d in self.device_ids)
        if ids and len(set(ids)) != len(ids):
            raise ModelingError(f"pool_duplicate_device_ids:{self.pool_id}")
        if ids and len(ids) != num:
            raise ModelingError(
                f"pool_device_count_mismatch:{self.pool_id}:len={len(ids)}!=num_devices={num}"
            )
        object.__setattr__(self, "device_ids", ids)
        if self.parallel is not None and not isinstance(self.parallel, ParallelSpec):
            raise ModelingError("pool_parallel_invalid")
        object.__setattr__(self, "speculative_k", _int_val(self.speculative_k, "pool_spec_k_invalid", minimum=0))


@dataclass(frozen=True, slots=True)
class ReadyStateAssumption:
    """Explicit ready KV/state context, layout, ownership, and optional restore cost for pd_decode."""

    source: ReadyStateSource
    context_tokens: int
    layout: str
    owner_ranges: tuple[tuple[int, int, int], ...] = ()
    capacity_bytes: int = 0
    cache_abi_digests: tuple[str, ...] = ()
    restore_transfer_spec: PdKvTransferSpec | None = None

    def __post_init__(self) -> None:
        if self.source not in READY_STATE_SOURCES:
            raise ModelingError(f"ready_state_source_invalid:{self.source!r}")
        object.__setattr__(self, "context_tokens", _int_val(self.context_tokens, "ready_state_context_invalid", minimum=1))
        object.__setattr__(self, "layout", _req_str(self.layout, "ready_state_layout_required"))
        object.__setattr__(self, "capacity_bytes", _int_val(self.capacity_bytes, "ready_state_capacity_invalid", minimum=0))
        ranges: list[tuple[int, int, int]] = []
        for item in self.owner_ranges:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ModelingError("ready_state_owner_range_invalid")
            rank = _int_val(item[0], "ready_state_owner_rank_invalid", minimum=0)
            start = _int_val(item[1], "ready_state_owner_start_invalid", minimum=0)
            end = _int_val(item[2], "ready_state_owner_end_invalid", minimum=0)
            if end < start:
                raise ModelingError(f"ready_state_owner_range_order_invalid:{start}>{end}")
            ranges.append((rank, start, end))
        object.__setattr__(self, "owner_ranges", tuple(ranges))
        object.__setattr__(self, "cache_abi_digests", tuple(str(d) for d in self.cache_abi_digests))
        if self.restore_transfer_spec is not None and not isinstance(self.restore_transfer_spec, PdKvTransferSpec):
            raise ModelingError("ready_state_restore_transfer_invalid")
        if self.source in {"pooled_dram_restore", "pooled_ssd_restore"} and self.restore_transfer_spec is None:
            raise ModelingError(
                f"ready_state_pooled_restore_requires_transfer_spec:{self.source}"
            )


@dataclass(frozen=True, slots=True)
class SLAContract:
    """Nullable SLA limits and explicit measurement boundaries (zero 5000 ms coercion)."""

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    itl_ms: float | None = None
    e2e_ms: float | None = None
    ttft_boundary: Literal[
        "arrival_to_first_token", "prefill_completion_only", "unavailable",
    ] = "arrival_to_first_token"
    tpot_boundary: Literal[
        "inter_token_decode", "ready_state_decode", "unavailable",
    ] = "inter_token_decode"

    def __post_init__(self) -> None:
        object.__setattr__(self, "ttft_ms", _opt_pos_float(self.ttft_ms, "sla_ttft_invalid"))
        object.__setattr__(self, "tpot_ms", _opt_pos_float(self.tpot_ms, "sla_tpot_invalid"))
        object.__setattr__(self, "itl_ms", _opt_pos_float(self.itl_ms, "sla_itl_invalid"))
        object.__setattr__(self, "e2e_ms", _opt_pos_float(self.e2e_ms, "sla_e2e_invalid"))
        if self.ttft_boundary not in {"arrival_to_first_token", "prefill_completion_only", "unavailable"}:
            raise ModelingError(f"sla_ttft_boundary_invalid:{self.ttft_boundary!r}")
        if self.tpot_boundary not in {"inter_token_decode", "ready_state_decode", "unavailable"}:
            raise ModelingError(f"sla_tpot_boundary_invalid:{self.tpot_boundary!r}")
        if self.ttft_boundary == "unavailable" and self.ttft_ms is not None:
            raise ModelingError("sla_ttft_limit_set_when_boundary_unavailable")
        if self.tpot_boundary == "unavailable" and (self.tpot_ms is not None or self.itl_ms is not None):
            raise ModelingError("sla_decode_limit_set_when_boundary_unavailable")

    def to_slo_spec(self) -> SloSpec:
        return SloSpec(
            ttft_ms=self.ttft_ms,
            tpot_ms=self.tpot_ms,
            itl_ms=self.itl_ms,
            e2e_ms=self.e2e_ms,
        )


@dataclass(frozen=True, slots=True)
class ConcurrencyLimits:
    """Strictly separates runtime admission limit, workload demand, and solver search cap."""

    runtime_max_num_seqs: int | None = None
    runtime_max_num_seqs_origin: str = "unset"
    workload_requested_concurrency: int | None = None
    solver_search_cap: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "runtime_max_num_seqs", _opt_pos_int(self.runtime_max_num_seqs, "limits_runtime_max_seqs_invalid")
        )
        object.__setattr__(
            self,
            "workload_requested_concurrency",
            _opt_pos_int(self.workload_requested_concurrency, "limits_requested_concurrency_invalid"),
        )
        object.__setattr__(
            self, "solver_search_cap", _opt_pos_int(self.solver_search_cap, "limits_solver_cap_invalid")
        )
        if not isinstance(self.runtime_max_num_seqs_origin, str) or not self.runtime_max_num_seqs_origin.strip():
            raise ModelingError("limits_origin_invalid")


@dataclass(frozen=True, slots=True)
class ComparisonStageSpec:
    """One causal stage in the 4-stage / 3-owner Engine waterfall."""

    stage: ExecutionStage
    owner: StageOwner
    allowed_readiness: frozenset[KernelReadiness]
    allowed_rewrite_tracks: tuple[RewriteTrack, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if self.stage not in EXECUTION_STAGES:
            raise ModelingError(f"comparison_stage_invalid:{self.stage!r}")
        if self.owner not in STAGE_OWNERS:
            raise ModelingError(f"comparison_stage_owner_invalid:{self.owner!r}")
        readiness = frozenset(self.allowed_readiness)
        if not readiness or any(item not in KERNEL_READINESS_VALUES for item in readiness):
            raise ModelingError(f"comparison_stage_readiness_invalid:{sorted(readiness)!r}")
        tracks = tuple(self.allowed_rewrite_tracks)
        if any(item not in REWRITE_TRACKS for item in tracks):
            raise ModelingError(f"comparison_stage_tracks_invalid:{tracks!r}")
        object.__setattr__(self, "allowed_readiness", readiness)
        object.__setattr__(self, "allowed_rewrite_tracks", tracks)


def canonical_comparison_stages(
    optimization_mode: OptimizationMode,
    *,
    include_proposed_pr: bool = False,
) -> tuple[ComparisonStageSpec, ...]:
    """Return the canonical Engine comparison chain for ``simulation`` or ``modeling``."""
    if optimization_mode not in OPTIMIZATION_MODES:
        raise ModelingError(f"optimization_mode_invalid:{optimization_mode!r}")
    shipping_set: frozenset[KernelReadiness] = (
        frozenset({"shipping", "proposed_pr"}) if include_proposed_pr else frozenset({"shipping"})
    )
    before = ComparisonStageSpec(
        stage="before",
        owner="baseline",
        allowed_readiness=frozenset({"shipping"}),
        allowed_rewrite_tracks=(),
        description="Base shipping graph before config, framework, or kernel changes",
    )
    config_only = ComparisonStageSpec(
        stage="config_only",
        owner="ops_config",
        allowed_readiness=shipping_set,
        allowed_rewrite_tracks=("config",),
        description="Graph_cfg (Graph_C): parallelism, batching, and wired stack flags",
    )
    if optimization_mode == "simulation":
        after_sim = ComparisonStageSpec(
            stage="after",
            owner="ops_config",
            allowed_readiness=shipping_set,
            allowed_rewrite_tracks=("config",),
            description="Simulation After: shipping/proposed_pr only, zero synthesized kernels",
        )
        return (before, config_only, after_sim)
    framework_rewrite = ComparisonStageSpec(
        stage="framework_rewrite",
        owner="framework_python",
        allowed_readiness=shipping_set,
        allowed_rewrite_tracks=("config", "framework"),
        description="Graph_fwk: framework GraphIR/stream transforms over shipping kernels",
    )
    after_mod = ComparisonStageSpec(
        stage="after",
        owner="kernel_dev_tune",
        allowed_readiness=frozenset({
            "shipping", "proposed_pr", "synthesized_target", "tiler_extension",
        }),
        allowed_rewrite_tracks=("config", "framework", "kernel_new", "kernel_tune"),
        description="Graph_After: Pillars A+B+C lowered to executable KernelProgram + OwnerScheduleSpec",
    )
    return (before, config_only, framework_rewrite, after_mod)


@dataclass(frozen=True, slots=True)
class MegaFusionBound:
    """Physical boundary lower bound vs tiled Engine expectation for a mega-fusion region."""

    region_op_ids: tuple[str, ...]
    exposed_before_ms: float
    boundary_lower_bound_ms: float
    tiled_engine_ms: float
    gain_ceiling_ms: float
    gain_expected_ms: float
    eliminated_gm_bytes: int
    ub_live_bytes: int
    ub_capacity_bytes: int
    requires_unabsorbed_split_cost_check: bool = True

    def __post_init__(self) -> None:
        ops = tuple(_req_str(op, "mega_fusion_op_id_invalid") for op in self.region_op_ids)
        if not ops:
            raise ModelingError("mega_fusion_region_ops_required")
        before = _finite_float(self.exposed_before_ms, "mega_fusion_before_invalid")
        l_star = _finite_float(self.boundary_lower_bound_ms, "mega_fusion_lstar_invalid")
        tiled = _finite_float(self.tiled_engine_ms, "mega_fusion_tiled_invalid")
        ceil_gain = _finite_float(self.gain_ceiling_ms, "mega_fusion_gain_ceiling_invalid", allow_negative=True)
        exp_gain = _finite_float(self.gain_expected_ms, "mega_fusion_gain_expected_invalid", allow_negative=True)
        if tiled + FLOAT_TOL < l_star:
            raise ModelingError(
                f"mega_fusion_primal_below_boundary_bound:tiled={tiled}<l_star={l_star}"
            )
        if exp_gain > ceil_gain + FLOAT_TOL:
            raise ModelingError(
                f"mega_fusion_expected_exceeds_ceiling:expected={exp_gain}>ceiling={ceil_gain}"
            )
        if abs((before - l_star) - ceil_gain) > FLOAT_TOL:
            raise ModelingError("mega_fusion_gain_ceiling_inconsistent")
        if abs((before - tiled) - exp_gain) > FLOAT_TOL:
            raise ModelingError("mega_fusion_gain_expected_inconsistent")
        elim = _int_val(self.eliminated_gm_bytes, "mega_fusion_eliminated_gm_invalid", minimum=0)
        ub_live = _int_val(self.ub_live_bytes, "mega_fusion_ub_live_invalid", minimum=0)
        ub_cap = _int_val(self.ub_capacity_bytes, "mega_fusion_ub_cap_invalid", minimum=0)
        if ub_cap > 0 and ub_live > ub_cap:
            raise ModelingError(f"mega_fusion_ub_overflow:ub_live={ub_live}>ub_cap={ub_cap}")
        object.__setattr__(self, "region_op_ids", ops)
        object.__setattr__(self, "exposed_before_ms", before)
        object.__setattr__(self, "boundary_lower_bound_ms", l_star)
        object.__setattr__(self, "tiled_engine_ms", tiled)
        object.__setattr__(self, "gain_ceiling_ms", ceil_gain)
        object.__setattr__(self, "gain_expected_ms", exp_gain)
        object.__setattr__(self, "eliminated_gm_bytes", elim)
        object.__setattr__(self, "ub_live_bytes", ub_live)
        object.__setattr__(self, "ub_capacity_bytes", ub_cap)


@dataclass(frozen=True, slots=True)
class SpaceTimeTradeoffSpec:
    """Explicit HBM-vs-compute co-design trade-off (space-for-time or time-for-space)."""

    direction: SpaceTimeDirection
    mechanism: SpaceTimeMechanism
    delta_hbm_bytes: int
    free_hbm_headroom_bytes: int
    delta_step_ms: float
    delta_concurrency_per_dp: int
    delta_tps_per_die: float
    feasible_under_hbm_and_sla: bool

    def __post_init__(self) -> None:
        if self.direction not in {"space_for_time", "time_for_space"}:
            raise ModelingError(f"space_time_direction_invalid:{self.direction!r}")
        if self.mechanism not in SPACE_TIME_MECHANISMS:
            raise ModelingError(f"space_time_mechanism_invalid:{self.mechanism!r}")
        delta_hbm = _int_val(self.delta_hbm_bytes, "space_time_delta_hbm_invalid")
        headroom = _int_val(self.free_hbm_headroom_bytes, "space_time_headroom_invalid")
        if self.direction == "space_for_time" and delta_hbm < 0:
            raise ModelingError("space_time_space_for_time_requires_nonneg_delta_hbm")
        if self.direction == "time_for_space" and delta_hbm > 0:
            raise ModelingError("space_time_time_for_space_requires_nonpos_delta_hbm")
        if not isinstance(self.feasible_under_hbm_and_sla, bool):
            raise ModelingError("space_time_feasible_flag_invalid")
        if self.feasible_under_hbm_and_sla and delta_hbm > 0 and delta_hbm > headroom:
            raise ModelingError(
                f"space_time_hbm_exceeds_headroom:delta={delta_hbm}>headroom={headroom}"
            )
        object.__setattr__(self, "delta_hbm_bytes", delta_hbm)
        object.__setattr__(self, "free_hbm_headroom_bytes", headroom)
        object.__setattr__(self, "delta_step_ms", _finite_float(self.delta_step_ms, "space_time_step_ms_invalid", allow_negative=True))
        object.__setattr__(
            self, "delta_concurrency_per_dp", _int_val(self.delta_concurrency_per_dp, "space_time_concurrency_invalid")
        )
        object.__setattr__(
            self, "delta_tps_per_die", _finite_float(self.delta_tps_per_die, "space_time_tps_invalid", allow_negative=True)
        )


@dataclass(frozen=True, slots=True)
class CrossHwOffloadSpec:
    """Cross-tier offload / multistream hide-window contract."""

    target_component: str
    source_tier: OffloadSourceTier
    transfer_engine: OffloadTransferEngine
    compute_hide_window_ms: float
    transfer_ms: float
    net_exposed_ms: float
    freed_hbm_bytes: int
    delta_concurrency_per_dp: int
    delta_tps_per_die: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_component", _req_str(self.target_component, "cross_hw_component_required"))
        if self.source_tier not in OFFLOAD_SOURCE_TIERS:
            raise ModelingError(f"cross_hw_source_tier_invalid:{self.source_tier!r}")
        if self.transfer_engine not in OFFLOAD_TRANSFER_ENGINES:
            raise ModelingError(f"cross_hw_transfer_engine_invalid:{self.transfer_engine!r}")
        hide_ms = _finite_float(self.compute_hide_window_ms, "cross_hw_hide_window_invalid")
        xfer_ms = _finite_float(self.transfer_ms, "cross_hw_transfer_ms_invalid")
        exposed_ms = _finite_float(self.net_exposed_ms, "cross_hw_net_exposed_invalid")
        expected = max(0.0, xfer_ms - hide_ms)
        if abs(exposed_ms - expected) > FLOAT_TOL:
            raise ModelingError(
                f"cross_hw_net_exposed_inconsistent:got={exposed_ms}:expected={expected}"
            )
        object.__setattr__(self, "compute_hide_window_ms", hide_ms)
        object.__setattr__(self, "transfer_ms", xfer_ms)
        object.__setattr__(self, "net_exposed_ms", exposed_ms)
        object.__setattr__(self, "freed_hbm_bytes", _int_val(self.freed_hbm_bytes, "cross_hw_freed_hbm_invalid", minimum=0))
        object.__setattr__(
            self, "delta_concurrency_per_dp", _int_val(self.delta_concurrency_per_dp, "cross_hw_concurrency_invalid")
        )
        object.__setattr__(
            self, "delta_tps_per_die", _finite_float(self.delta_tps_per_die, "cross_hw_tps_invalid", allow_negative=True)
        )


@dataclass(frozen=True, slots=True)
class CoDesignOpportunity:
    """Typed co-design opportunity record shared across ComparisonPlan, Sheet 4, and agent_brief."""

    opportunity_id: str
    kind: CoDesignKind
    track: RewriteTrack
    readiness: KernelReadiness
    title: str
    evidence_refs: tuple[str, ...] = ()
    mega_fusion: MegaFusionBound | None = None
    space_time: SpaceTimeTradeoffSpec | None = None
    cross_hw: CrossHwOffloadSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "opportunity_id", _req_str(self.opportunity_id, "codesign_opportunity_id_required"))
        object.__setattr__(self, "title", _req_str(self.title, "codesign_title_required"))
        if self.kind not in CO_DESIGN_KINDS:
            raise ModelingError(f"codesign_kind_invalid:{self.kind!r}")
        if self.track not in REWRITE_TRACKS:
            raise ModelingError(f"codesign_track_invalid:{self.track!r}")
        if self.readiness not in KERNEL_READINESS_VALUES:
            raise ModelingError(f"codesign_readiness_invalid:{self.readiness!r}")
        object.__setattr__(self, "evidence_refs", tuple(str(r) for r in self.evidence_refs))
        self._validate_payload()

    def _validate_payload(self) -> None:
        if self.kind == "mega_fusion":
            if not isinstance(self.mega_fusion, MegaFusionBound) or self.space_time is not None or self.cross_hw is not None:
                raise ModelingError("codesign_mega_fusion_payload_mismatch")
        elif self.kind == "space_time_tradeoff":
            if not isinstance(self.space_time, SpaceTimeTradeoffSpec) or self.mega_fusion is not None or self.cross_hw is not None:
                raise ModelingError("codesign_space_time_payload_mismatch")
        elif self.kind == "cross_hw_offload_multistream":
            if not isinstance(self.cross_hw, CrossHwOffloadSpec) or self.mega_fusion is not None or self.space_time is not None:
                raise ModelingError("codesign_cross_hw_payload_mismatch")
        elif self.kind == "tiler_tune":
            if self.space_time is not None or self.cross_hw is not None:
                raise ModelingError("codesign_tiler_tune_payload_mismatch")
            if self.mega_fusion is not None and not isinstance(self.mega_fusion, MegaFusionBound):
                raise ModelingError("codesign_tiler_tune_bound_invalid")


@dataclass(frozen=True, slots=True)
class ScenarioContract:
    """Single immutable, validated scenario contract binding all inputs and plugin plans."""

    deployment_mode: DeploymentMode
    deployment_mode_origin: DeploymentModeOrigin
    workload_domain: WorkloadDomain
    work_unit_kind: WorkUnitKind
    model: ModelSpec
    checkpoint: CheckpointQuantMap
    stack_id: str
    stack_track: StackTrack
    stack_options: ResolvedStackOptions
    quant_plan: RuntimeQuantPlan
    lowering_plan: StackLoweringPlan
    hardware: Hardware
    primitive_table_digest: str
    optimization_mode: OptimizationMode = "simulation"
    execution_stage: ExecutionStage = "before"
    sla: SLAContract = field(default_factory=SLAContract)
    limits: ConcurrencyLimits = field(default_factory=ConcurrencyLimits)
    workload: WorkloadSpec | None = None
    num_devices: int | None = None
    pools: tuple[PoolSpec, ...] = ()
    allow_solver_pd_split: bool = False
    ready_state: ReadyStateAssumption | None = None
    pd_transfer: PdKvTransferSpec | None = None
    pinned_parallel: tuple[tuple[str, int], ...] = ()
    vision_accounting: VisionAccountingScope = "text_backbone"
    provenance: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        mode = validate_deployment_mode(self.deployment_mode)
        object.__setattr__(self, "deployment_mode", mode)
        object.__setattr__(self, "pools", tuple(self.pools))
        object.__setattr__(
            self,
            "pinned_parallel",
            tuple((str(k), _int_val(v, "scenario_pinned_parallel_invalid", minimum=1)) for k, v in self.pinned_parallel),
        )
        object.__setattr__(self, "provenance", tuple((str(k), str(v)) for k, v in self.provenance))
        validate_scenario_contract(self)

    def digest(self) -> str:
        payload = {
            "deployment_mode": self.deployment_mode,
            "workload_domain": self.workload_domain,
            "work_unit_kind": self.work_unit_kind,
            "optimization_mode": self.optimization_mode,
            "execution_stage": self.execution_stage,
            "model_name": self.model.name,
            "checkpoint_digest": self.checkpoint.digest(),
            "stack_id": self.stack_id,
            "stack_track": self.stack_track,
            "options_digest": self.stack_options.options_digest,
            "quant_plan_digest": self.quant_plan.plan_digest,
            "lowering_plan_digest": self.lowering_plan.plan_digest,
            "primitive_table_digest": self.primitive_table_digest,
            "num_devices": self.num_devices,
            "pinned_parallel": list(self.pinned_parallel),
            "vision_accounting": self.vision_accounting,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class DeploymentOutcome:
    """Honest solver deployment outcome separating feasible winners from infeasible/incomplete runs."""

    status: OutcomeStatus
    deployment_mode: DeploymentMode
    actual_transfer_mode: str = "none"
    limiting_constraint: str = ""
    physical_b_vram_max: int = 0
    configured_max_num_seqs: int = 0
    feasible_b_per_dp: int = 0
    requested_concurrency: int | None = None
    verified_ttft_ms: float | None = None
    verified_tpot_ms: float | None = None
    ttft_availability: Literal["measured", "unconstrained", "unavailable"] = "measured"
    tpot_availability: Literal["measured", "unconstrained", "unavailable"] = "measured"
    winner_case: Any | None = None
    prefill_case: Any | None = None
    decode_case: Any | None = None
    capacity_receipt: Any | None = None
    search_coverage: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        if self.status not in OUTCOME_STATUSES:
            raise ModelingError(f"deployment_outcome_status_invalid:{self.status!r}")
        mode = validate_deployment_mode(self.deployment_mode)
        object.__setattr__(self, "deployment_mode", mode)
        object.__setattr__(self, "verified_ttft_ms", _opt_pos_float(self.verified_ttft_ms, "outcome_ttft_invalid"))
        object.__setattr__(self, "verified_tpot_ms", _opt_pos_float(self.verified_tpot_ms, "outcome_tpot_invalid"))
        object.__setattr__(self, "physical_b_vram_max", _int_val(self.physical_b_vram_max, "outcome_b_vram_invalid", minimum=0))
        object.__setattr__(self, "configured_max_num_seqs", _int_val(self.configured_max_num_seqs, "outcome_max_seqs_invalid", minimum=0))
        object.__setattr__(self, "feasible_b_per_dp", _int_val(self.feasible_b_per_dp, "outcome_b_feasible_invalid", minimum=0))
        if self.status != "feasible" and self.winner_case is not None:
            raise ModelingError(f"deployment_outcome_winner_forbidden_for_status:{self.status}")
        if self.status == "infeasible" and self.feasible_b_per_dp != 0:
            raise ModelingError(f"deployment_outcome_infeasible_must_have_zero_batch:{self.feasible_b_per_dp}")
        if mode == "pd_decode" and self.verified_ttft_ms is not None:
            raise ModelingError("deployment_outcome_pd_decode_ttft_must_be_none")
        if mode == "pd_prefill" and self.verified_tpot_ms is not None:
            raise ModelingError("deployment_outcome_pd_prefill_tpot_must_be_none")
        object.__setattr__(
            self,
            "search_coverage",
            tuple((str(k), _int_val(v, "outcome_coverage_invalid", minimum=0)) for k, v in self.search_coverage),
        )


def _validate_pools_and_mode(contract: ScenarioContract) -> None:
    mode = contract.deployment_mode
    pools = contract.pools
    if any(not isinstance(p, PoolSpec) for p in pools):
        raise ModelingError("scenario_pools_invalid")
    if mode == "colocated":
        if contract.ready_state is not None:
            raise ModelingError("colocated_forbids_ready_state_assumption")
        if contract.pd_transfer is not None:
            raise ModelingError("colocated_forbids_pd_transfer_spec")
        if pools and (len(pools) != 1 or pools[0].role != "colocated"):
            raise ModelingError("colocated_requires_single_colocated_pool")
    elif mode == "pd_disaggregated":
        _validate_pd_disaggregated(contract)
    elif mode == "pd_prefill":
        if contract.sla.tpot_ms is not None or contract.sla.itl_ms is not None:
            raise ModelingError(
                "pd_prefill_rejects_decode_sla:tpot_ms and itl_ms are not measurable at pd_prefill boundary"
            )
        if contract.ready_state is not None:
            raise ModelingError("pd_prefill_forbids_ready_state_assumption")
        if pools and (len(pools) != 1 or pools[0].role != "prefill"):
            raise ModelingError("pd_prefill_requires_single_prefill_pool")
    elif mode == "pd_decode":
        if contract.ready_state is None:
            raise ModelingError(
                "pd_decode_requires_ready_state_assumption: specify ReadyStateAssumption(source, context_tokens, layout)"
            )
        if contract.sla.ttft_ms is not None:
            raise ModelingError(
                "pd_decode_ttft_unavailable: prompt-arrival ttft_ms is unavailable in pd_decode mode (must be None)"
            )
        if pools and (len(pools) != 1 or pools[0].role != "decode"):
            raise ModelingError("pd_decode_requires_single_decode_pool")


def _validate_pd_disaggregated(contract: ScenarioContract) -> None:
    pools = contract.pools
    transfer = contract.pd_transfer or contract.lowering_plan.scheduling.pd_kv_transfer
    if transfer is None:
        raise ModelingError(
            "pd_disaggregated_requires_transfer_spec: provide PdKvTransferSpec (blocking or layerwise_overlap)"
        )
    if pools:
        p_pools = [p for p in pools if p.role == "prefill"]
        d_pools = [p for p in pools if p.role == "decode"]
        if len(pools) != 2 or len(p_pools) != 1 or len(d_pools) != 1:
            raise ModelingError(
                "pd_disaggregated_requires_disjoint_p_and_d_pools: expected exactly one prefill and one decode pool"
            )
        p_pool, d_pool = p_pools[0], d_pools[0]
        if p_pool.device_ids and d_pool.device_ids and (set(p_pool.device_ids) & set(d_pool.device_ids)):
            raise ModelingError("pd_disaggregated_pools_must_be_physically_disjoint")
        if contract.num_devices is not None and (p_pool.num_devices + d_pool.num_devices) > contract.num_devices:
            raise ModelingError("pd_disaggregated_pools_exceed_num_devices")
    elif not (contract.allow_solver_pd_split and contract.num_devices is not None and contract.num_devices >= MIN_DISAGGREGATED_DEVICES):
        raise ModelingError(
            "pd_disaggregated_requires_pools_or_split_budget: provide two disjoint P/D PoolSpec entries "
            "or set allow_solver_pd_split=True with num_devices >= 2"
        )
    for edge in transfer.layer_edges:
        if edge.source_abi_key != edge.dest_abi_key and not (edge.requires_repack or transfer.conversion_required):
            raise ModelingError(
                f"pd_disaggregated_incompatible_role_abi:layer={edge.native_layer}:"
                f"{edge.source_abi_key}!={edge.dest_abi_key}"
            )


def _validate_domain_and_modes(contract: ScenarioContract) -> None:
    if contract.workload_domain not in WORKLOAD_DOMAINS:
        raise ModelingError(f"scenario_workload_domain_invalid:{contract.workload_domain!r}")
    if contract.work_unit_kind not in WORK_UNIT_KINDS:
        raise ModelingError(f"scenario_work_unit_invalid:{contract.work_unit_kind!r}")
    allowed_units = DOMAIN_WORK_UNITS[contract.workload_domain]
    if contract.work_unit_kind not in allowed_units:
        raise ModelingError(
            f"workload_domain_work_unit_mismatch:domain={contract.workload_domain}:"
            f"work_unit={contract.work_unit_kind}:allowed={sorted(allowed_units)}"
        )
    if contract.workload_domain == "diffusion" and contract.quant_plan.kv_cache_abi:
        raise ModelingError("diffusion_forbids_llm_kv_cache_abi")
    if contract.optimization_mode not in OPTIMIZATION_MODES:
        raise ModelingError(f"scenario_optimization_mode_invalid:{contract.optimization_mode!r}")
    if contract.execution_stage not in EXECUTION_STAGES:
        raise ModelingError(f"scenario_execution_stage_invalid:{contract.execution_stage!r}")
    plan = contract.lowering_plan
    if contract.optimization_mode == "simulation":
        if not plan.allowed_rewrite_classes.issubset(SIMULATION_ALLOWED_READINESS):
            raise ModelingError(
                f"simulation_mode_forbids_non_shipping_rewrites:{sorted(plan.allowed_rewrite_classes)}"
            )
        if any(track != "config" for track in plan.allowed_rewrite_tracks):
            raise ModelingError(
                f"simulation_mode_allows_config_track_only:{plan.allowed_rewrite_tracks!r}"
            )
    if contract.execution_stage == "framework_rewrite":
        if not plan.allowed_rewrite_classes.issubset(SIMULATION_ALLOWED_READINESS):
            raise ModelingError(
                f"framework_rewrite_stage_forbids_new_kernels:{sorted(plan.allowed_rewrite_classes)}"
            )
    if contract.execution_stage == "config_only":
        if not plan.allowed_rewrite_classes.issubset(SIMULATION_ALLOWED_READINESS):
            raise ModelingError(
                f"config_only_stage_forbids_new_kernels:{sorted(plan.allowed_rewrite_classes)}"
            )


def validate_scenario_contract(contract: ScenarioContract) -> None:
    """Enforce every cross-field invariant on ``ScenarioContract``."""
    if not isinstance(contract, ScenarioContract):
        raise ModelingError("scenario_contract_type_invalid")
    validate_deployment_mode(contract.deployment_mode)
    if contract.deployment_mode_origin not in DEPLOYMENT_MODE_ORIGINS:
        raise ModelingError(f"scenario_deployment_origin_invalid:{contract.deployment_mode_origin!r}")
    if contract.stack_track not in STACK_TRACKS:
        raise ModelingError(f"scenario_stack_track_invalid:{contract.stack_track!r}")
    if contract.vision_accounting not in VISION_ACCOUNTING_SCOPES:
        raise ModelingError(f"scenario_vision_accounting_invalid:{contract.vision_accounting!r}")
    if not isinstance(contract.stack_id, str) or not contract.stack_id.strip():
        raise ModelingError("scenario_stack_id_required")
    if contract.stack_options.stack_id != contract.stack_id:
        raise ModelingError("scenario_stack_options_id_mismatch")
    if contract.quant_plan.stack_id != contract.stack_id:
        raise ModelingError("scenario_quant_plan_stack_id_mismatch")
    if contract.lowering_plan.stack_id != contract.stack_id:
        raise ModelingError("scenario_lowering_plan_stack_id_mismatch")
    ckpt_digest = contract.checkpoint.digest()
    if contract.quant_plan.checkpoint_digest and contract.quant_plan.checkpoint_digest != ckpt_digest:
        raise ModelingError("scenario_quant_plan_checkpoint_digest_mismatch")
    if contract.num_devices is not None:
        _int_val(contract.num_devices, "scenario_num_devices_invalid", minimum=1)
    _validate_domain_and_modes(contract)
    _validate_pools_and_mode(contract)


def _check_stack_capability(
    cap: StackCapabilitySpec,
    *,
    domain: WorkloadDomain,
    work_unit: WorkUnitKind | None,
    required_phases: Sequence[str],
    hardware_backend: str | None,
    checkpoint_format: str | None,
    track: StackTrack | None,
    required_recipes: Sequence[str],
) -> str | None:
    if not cap.supports_domain(domain):
        return (
            f"stack_domain_mismatch:stack={cap.stack_id}:domain={domain}:"
            f"supported_domains={cap.supported_domains}"
        )
    if work_unit is not None and not cap.supports_work_unit(work_unit):
        return (
            f"stack_work_unit_mismatch:stack={cap.stack_id}:work_unit={work_unit}:"
            f"supported_work_units={cap.supported_work_units}"
        )
    missing_phases = [p for p in required_phases if p not in cap.supported_phases]
    if missing_phases:
        return f"stack_phase_unsupported:stack={cap.stack_id}:missing={missing_phases}"
    if hardware_backend and hardware_backend.lower() not in cap.supported_hardware_backends:
        return f"stack_hardware_unsupported:stack={cap.stack_id}:backend={hardware_backend}"
    if checkpoint_format and checkpoint_format not in cap.supported_checkpoint_formats:
        return f"stack_checkpoint_format_unsupported:stack={cap.stack_id}:format={checkpoint_format}"
    if track and track not in cap.supported_tracks:
        return f"stack_track_unsupported:stack={cap.stack_id}:track={track}"
    missing_recipes = [r for r in required_recipes if r not in cap.supported_recipe_ids]
    if missing_recipes:
        return f"stack_recipes_unsupported:stack={cap.stack_id}:missing={missing_recipes}"
    return None


def resolve_default_stack_for_domain(
    domain: WorkloadDomain,
    *,
    work_unit: WorkUnitKind | None = None,
    available_stacks: Sequence[StackCapabilitySpec] = (),
    requested_stack: str | None = None,
    domain_defaults: Mapping[WorkloadDomain, str] | None = None,
    required_phases: Sequence[str] = (),
    hardware_backend: str | None = None,
    checkpoint_format: str | None = None,
    track: StackTrack | None = None,
    required_recipes: Sequence[str] = (),
) -> str:
    """Resolve and capability-check a runtime stack ID for ``domain`` without vendor hardcoding."""
    if domain not in WORKLOAD_DOMAINS:
        raise ModelingError(f"workload_domain_invalid:{domain!r}")
    if work_unit is not None:
        if work_unit not in WORK_UNIT_KINDS:
            raise ModelingError(f"work_unit_kind_invalid:{work_unit!r}")
        if work_unit not in DOMAIN_WORK_UNITS[domain]:
            raise ModelingError(
                f"workload_domain_work_unit_mismatch:domain={domain}:work_unit={work_unit}"
            )
    by_id = {cap.stack_id: cap for cap in available_stacks}
    check_kwargs: dict[str, Any] = {
        "domain": domain,
        "work_unit": work_unit,
        "required_phases": required_phases,
        "hardware_backend": hardware_backend,
        "checkpoint_format": checkpoint_format,
        "track": track,
        "required_recipes": required_recipes,
    }
    if requested_stack is not None:
        req = _req_str(requested_stack, "requested_stack_invalid")
        if by_id:
            if req not in by_id:
                raise ModelingError(f"stack_not_registered:{req}:available={sorted(by_id)}")
            if err := _check_stack_capability(by_id[req], **check_kwargs):
                raise ModelingError(err)
        return req
    preferred = domain_defaults.get(domain) if domain_defaults else None
    if preferred is not None:
        if by_id:
            if preferred not in by_id:
                raise ModelingError(f"default_stack_not_registered:{preferred}:domain={domain}")
            if err := _check_stack_capability(by_id[preferred], **check_kwargs):
                raise ModelingError(err)
        return preferred
    for cap in available_stacks:
        if _check_stack_capability(cap, **check_kwargs) is None:
            return cap.stack_id
    raise ModelingError(
        f"no_compatible_stack_for_domain:domain={domain}:work_unit={work_unit}:"
        f"available={[cap.stack_id for cap in available_stacks]}"
    )


__all__ = [
    "CO_DESIGN_KINDS",
    "DEPLOYMENT_MODES",
    "DEPLOYMENT_MODE_LIST_TEXT",
    "DEPLOYMENT_MODE_ORIGINS",
    "DOMAIN_WORK_UNITS",
    "EXECUTION_STAGES",
    "OFFLOAD_SOURCE_TIERS",
    "OFFLOAD_TRANSFER_ENGINES",
    "OPTIMIZATION_MODES",
    "OUTCOME_STATUSES",
    "POOL_ROLES",
    "READY_STATE_SOURCES",
    "SIMULATION_ALLOWED_READINESS",
    "SPACE_TIME_MECHANISMS",
    "STAGE_OWNERS",
    "VISION_ACCOUNTING_SCOPES",
    "CoDesignKind",
    "CoDesignOpportunity",
    "ComparisonStageSpec",
    "ConcurrencyLimits",
    "CrossHwOffloadSpec",
    "DeploymentMode",
    "DeploymentModeOrigin",
    "DeploymentOutcome",
    "ExecutionStage",
    "MegaFusionBound",
    "OffloadSourceTier",
    "OffloadTransferEngine",
    "OptimizationMode",
    "OutcomeStatus",
    "PoolRole",
    "PoolSpec",
    "ReadyStateAssumption",
    "ReadyStateSource",
    "SLAContract",
    "ScenarioContract",
    "SpaceTimeDirection",
    "SpaceTimeMechanism",
    "SpaceTimeTradeoffSpec",
    "StageOwner",
    "VisionAccountingScope",
    "canonical_comparison_stages",
    "resolve_default_stack_for_domain",
    "validate_deployment_mode",
    "validate_scenario_contract",
]
