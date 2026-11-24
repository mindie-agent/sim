"""vLLM-Ascend ``RuntimeStackAdapter`` implementation.

[Role] Implement the authoritative ``RuntimeStackAdapter`` plugin for
  ``vllm-ascend``:
  - ``capabilities()`` declaring strict LLM/VLM scope (explicitly rejecting
    Diffusion/DiT with ``ModelingError("stack_domain_mismatch:...")``),
  - ``default_checkpoint_for()`` delegating to pure checkpoint registry,
  - ``resolve_options()`` separating requested vs effective mechanisms,
    ACLGraph ``CaptureBucketRule``, speculative rules, RecoverSSM, MC2/MSO
    mutex, and per-die ``StreamOverlapPolicy``,
  - ``resolve_quant_instantiation()`` delegating to 1:1 ``quant_policy``,
  - ``select_lowering_plan()`` emitting full ``StackLoweringPlan`` +
    ``StackTrackSpec`` + ``SchedulingContract`` with strict ``StackTrack``
    enforcement,
  - ``plan_iteration()`` providing live one-step continuous batching without
    ``1 + 2*K`` double-counting or pre-frontend capture padding,
  - ``launch_codec`` exposing ``VllmAscendLaunchCodec``.
[Invariants] Zero imports from ``mindie_sim.export`` or ``mindie_sim.compose``
  (AG010). Zero per-model-name executable branches (AG014). Zero silent FIA
  clamping, zero hardcoded STARS ns floors, zero blind MLA Prolog / A5-only
  AttnRes / quantized indexer overrides.
[Timeouts & Locks] Pure in-memory resolution over frozen schema contracts; no
  locks or network I/O.
[Side-Effect Policy] Stateless adapter returning fresh immutable records.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mindie_sim.schema.checkpoint import CheckpointQuantMap
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.operators import KernelReadiness
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.stack import (
    STACK_ADAPTER_PROTOCOL_VERSION,
    CaptureBucketRule,
    FrameworkRewriteKind,
    IterationDecision,
    IterationSchedulerState,
    LaunchConfigCodec,
    PdKvTransferSpec,
    PiecewiseSplitPoint,
    ResolvedStackOptions,
    RewriteTrack,
    RuntimeQuantPlan,
    SchedulerDone,
    StackCapabilitySpec,
    StackLoweringPlan,
    StackTrack,
    StarsSubmitPolicy,
    StreamLaneBudget,
    StreamOverlapPolicy,
    StreamSyncEdge,
    WaitForEvent,
)
from mindie_sim.schema.workload import IterationPlan

from .launch_codec import VllmAscendLaunchCodec
from .profiles import (
    VLLM_ASCEND_STACK_ID,
    default_checkpoint_repo,
    hardware_backend_kind,
    infer_model_workload_domain,
    load_vllm_ascend_manifest,
    normalize_hw_family,
)
from .quant_policy import instantiate_runtime_quant_plan

DEFAULT_STARS_FIFO_LANES = 4
DEFAULT_STARS_QUEUE_DEPTH = 32
DEFAULT_PD_RECEIVE_CREDITS = 4
MSO_AUX_AIC_DIVISOR = 4
MSO_AUX_AIV_DIVISOR = 4
MIN_STREAM_CORES = 1

_ALL_FRAMEWORK_REWRITES: tuple[FrameworkRewriteKind, ...] = (
    "merge_sibling_projections",
    "bind_shipping_op",
    "multistream_overlap",
    "eliminate_graph_break",
)
_MODELING_REWRITE_TRACKS: tuple[RewriteTrack, ...] = (
    "config", "framework", "kernel_new", "kernel_tune",
)
_MODELING_READINESS: frozenset[KernelReadiness] = frozenset({
    "shipping", "proposed_pr", "synthesized_target", "tiler_extension",
})
_SIMULATION_REWRITE_TRACKS: tuple[RewriteTrack, ...] = ("config",)
_PD_ROLES = frozenset({"pd_disaggregated", "pd_prefill", "pd_decode"})


def _ensure_supported_domain(model: ModelSpec) -> str:
    domain = infer_model_workload_domain(model)
    if domain not in {"llm", "vlm"}:
        raise ModelingError(
            f"stack_domain_mismatch:stack={VLLM_ASCEND_STACK_ID}:model={model.name}:"
            f"domain={domain}:supported=llm,vlm"
        )
    return domain


def _ensure_supported_hardware(hardware: Hardware) -> str:
    backend = hardware_backend_kind(hardware)
    if backend != "ascend":
        raise ModelingError(
            f"stack_hardware_unsupported:stack={VLLM_ASCEND_STACK_ID}:backend={backend}:supported=ascend"
        )
    return normalize_hw_family(hardware)


def _resolve_speculative_option(
    model: ModelSpec,
    *,
    hw: str,
    parallel: ParallelSpec,
    role_norm: str,
    requested_options: Mapping[str, Any],
    effective_kv: str,
    dispositions: list[tuple[str, str, str]],
) -> tuple[SpeculativeSpec | None, str, int, bool]:
    """Unplugged socket: silent FIA clamping, negative-K disable, and acceptance overwrite purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: enforce strict FIA K<=15 validation without clamping (#743), "
        "reject invalid/negative K, preserve caller acceptance distribution verbatim, "
        "and resolve speculative execution purely from structural SpeculativeExecutionContract"
    )


def _build_stream_overlap_policy(
    hardware: Hardware,
    model: ModelSpec,
    *,
    track: StackTrack,
    eff_mso: bool,
    has_comm: bool,
) -> StreamOverlapPolicy:
    """Build per-die stream lane budgets and event sync edges from hardware topology."""
    die = hardware.dies[0] if hardware.dies else None
    total_aic = int(die.aic_count) if die is not None else 1
    total_aiv = int(die.owner_count("vector")) if die is not None else 1
    lanes: list[StreamLaneBudget] = []
    edges: list[StreamSyncEdge] = []
    joins: list[tuple[str, tuple[str, ...]]] = []
    overlaps: list[str] = []

    has_kda = any(str(p.attention).lower() == "kda" for p in model.layer_plans)
    if eff_mso:
        aux_aic = max(MIN_STREAM_CORES, total_aic // MSO_AUX_AIC_DIVISOR)
        aux_aiv = max(MIN_STREAM_CORES, total_aiv // MSO_AUX_AIV_DIVISOR)
        main_aic = max(MIN_STREAM_CORES, total_aic - aux_aic)
        main_aiv = max(MIN_STREAM_CORES, total_aiv - aux_aiv)
        lanes.append(StreamLaneBudget(
            stream_id="main", die_id=0, aic_budget=main_aic, aiv_budget=main_aiv,
            isolation="disjoint", priority=0,
        ))
        lanes.append(StreamLaneBudget(
            stream_id="shared_expert_mso", die_id=0, aic_budget=aux_aic, aiv_budget=aux_aiv,
            isolation="disjoint", priority=1,
        ))
        edges.append(StreamSyncEdge(
            event_id="ev.mso_fork", record_stream="main", record_after="moe_router",
            wait_stream="shared_expert_mso", wait_before=("shared_expert_gate_up",),
        ))
        edges.append(StreamSyncEdge(
            event_id="ev.mso_join", record_stream="shared_expert_mso", record_after="shared_expert_down",
            wait_stream="main", wait_before=("moe_residual_add",),
        ))
        joins.append(("moe_residual_add", ("routed_moe_combine", "shared_expert_down")))
        overlaps.append("shared_expert_mso")
    else:
        lanes.append(StreamLaneBudget(
            stream_id="main", die_id=0, aic_budget=total_aic, aiv_budget=total_aiv,
            isolation="shared_cap", priority=0,
        ))

    if has_kda and track != "eager":
        lanes.append(StreamLaneBudget(
            stream_id="kda_bfg_aux", die_id=0,
            aic_budget=max(MIN_STREAM_CORES, total_aic // MSO_AUX_AIC_DIVISOR),
            aiv_budget=max(MIN_STREAM_CORES, total_aiv // MSO_AUX_AIV_DIVISOR),
            isolation="shared_cap", priority=1,
        ))
        edges.append(StreamSyncEdge(
            event_id="ev.kda_bfg_join", record_stream="kda_bfg_aux", record_after="fused_bfg_proj",
            wait_stream="main", wait_before=("kda_recurrent_step",),
        ))
        overlaps.append("kda_bfg_dual_stream")

    if has_comm:
        lanes.append(StreamLaneBudget(
            stream_id="comm", die_id=0, aic_budget=0, aiv_budget=max(MIN_STREAM_CORES, total_aiv // MSO_AUX_AIV_DIVISOR),
            isolation="shared_cap", priority=1,
        ))
        edges.append(StreamSyncEdge(
            event_id="ev.comm_done", record_stream="comm", record_after="collective_allreduce",
            wait_stream="main", wait_before=("post_comm_norm",),
        ))
        overlaps.append("comm_stream")

    return StreamOverlapPolicy(
        lanes=tuple(lanes),
        sync_edges=tuple(edges),
        joins=tuple(joins),
        service_terminal="token_commit",
        enabled_overlaps=tuple(overlaps),
    )


def _splitting_ops_for_model(model: ModelSpec) -> tuple[str, ...]:
    ops: list[str] = []
    attns = {str(p.attention).lower() for p in model.layer_plans} or {str(model.attention).lower()}
    if "mla" in attns:
        ops.append("vllm.unified_mla_attention_with_output")
    if attns - {"mla", "kda", "gdn"}:
        ops.append("vllm.unified_attention_with_output")
    if attns & {"kda", "gdn"}:
        ops.append("vllm.linear_attention")
    return tuple(ops)


def _build_piecewise_split_points(
    model: ModelSpec,
    capture_rule: CaptureBucketRule | None,
) -> tuple[PiecewiseSplitPoint, ...]:
    if capture_rule is None or capture_rule.mode not in {"piecewise", "full_and_piecewise"}:
        return ()
    splits: list[PiecewiseSplitPoint] = []
    selected_key = capture_rule.piecewise_keys[0] if capture_rule.piecewise_keys else None
    for idx, plan in enumerate(model.layer_plans):
        n_id = plan.native_layer_id if plan.native_layer_id >= 0 else plan.index
        attn = str(plan.attention or model.attention).lower()
        splits.append(PiecewiseSplitPoint(
            call_or_occurrence_id=f"target.L{n_id}.{attn}_core",
            scope="target",
            stage=0,
            segment_index=idx,
            reason=f"piecewise_splitting_op:{attn}",
            selected_key=selected_key,
            host_launch_id=f"launch.seg.{idx}",
            eager_island=True,
        ))
    return tuple(splits)


def _build_stars_submit_policies(
    capture_rule: CaptureBucketRule | None,
    stream_policy: StreamOverlapPolicy | None,
) -> tuple[StarsSubmitPolicy, ...]:
    """Unplugged socket: hardcoded STARS latency floors purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: source STARS submit cadence/first-issue timings strictly from "
        "mindie_sim/backends/profiles/ascend.json or PrimitiveTable (zero magic ns floors)"
    )


def _build_pd_kv_transfer_spec(
    quant_plan: RuntimeQuantPlan,
    *,
    hw: str,
    mode_str: str,
    transport_str: str,
) -> PdKvTransferSpec:
    """Unplugged socket: silent PD transfer mode/transport coercion purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: validate PdTransferMode and PdTransportKind strictly with "
        "ModelingError (zero silent blocking/direct_d2d coercion)"
    )


def _select_recipe_bindings(
    model: ModelSpec,
    *,
    track: StackTrack,
    hw: str,
    eff_opts: Mapping[str, Any],
) -> tuple[
    tuple[tuple[str, str, tuple[tuple[str, Any], ...]], ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
]:
    """Unplugged socket: blind MLA Prolog, A5-only AttnRes, blind QLI, and inert manifest hits purged."""
    raise NotImplementedError(
        "LEGACY_PURGED: wire structural M1 5-branch MLA Prolog predicate, cross-arch "
        "A3/A5 AttnResFwd, RuntimeQuantPlan-driven indexer dtype selection, and typed "
        "ManifestBinding.lowering_pass_id dispatch with phase/option admission"
    )


class VllmAscendStackAdapter:
    """Authoritative ``RuntimeStackAdapter`` implementation for ``vllm-ascend``."""

    def __init__(self) -> None:
        self._codec = VllmAscendLaunchCodec()

    def capabilities(self) -> StackCapabilitySpec:
        manifest = load_vllm_ascend_manifest()
        return StackCapabilitySpec(
            stack_id=VLLM_ASCEND_STACK_ID,
            manifest_revision=str(manifest["manifest_revision"]),
            supported_domains=tuple(manifest["supported_domains"]),
            supported_work_units=tuple(manifest["supported_work_units"]),
            supported_hardware_backends=tuple(manifest["supported_hardware_backends"]),
            supported_phases=tuple(manifest["supported_phases"]),
            supported_checkpoint_formats=tuple(manifest["supported_checkpoint_formats"]),
            supported_tracks=tuple(manifest["supported_tracks"]),
            supported_recipe_ids=frozenset(manifest["supported_recipe_ids"]),
            has_launch_codec=True,
            protocol_version=str(manifest.get("protocol_version") or STACK_ADAPTER_PROTOCOL_VERSION),
        )

    def default_checkpoint_for(
        self, canonical_model: str, hardware: Hardware,
    ) -> str | None:
        if hardware_backend_kind(hardware) != "ascend":
            return None
        return default_checkpoint_repo(canonical_model, normalize_hw_family(hardware))

    def resolve_options(
        self,
        model: ModelSpec,
        hardware: Hardware,
        parallel: ParallelSpec,
        *,
        role: str,
        track: StackTrack,
        requested_options: Mapping[str, Any],
        field_origins: Mapping[str, str],
    ) -> ResolvedStackOptions:
        raise NotImplementedError(
            "LEGACY_PURGED: wire structural VllmAscendStackAdapter.resolve_options "
            "(zero normalize_canonical_model/simulation_config routing, fail closed on "
            "invalid K/capture_sizes/kv_cache_dtype, and strict canonical StackTrack)"
        )

    def resolve_quant_instantiation(
        self,
        model: ModelSpec,
        checkpoint: CheckpointQuantMap,
        hardware: Hardware,
        parallel: ParallelSpec,
        options: ResolvedStackOptions,
        *,
        role: str,
        phase: str,
    ) -> RuntimeQuantPlan:
        _ensure_supported_domain(model)
        _ensure_supported_hardware(hardware)
        return instantiate_runtime_quant_plan(
            model, checkpoint, hardware, parallel, options, role=role, phase=phase,
        )

    def select_lowering_plan(
        self,
        model: ModelSpec,
        quant_plan: RuntimeQuantPlan,
        hardware: Hardware,
        parallel: ParallelSpec,
        options: ResolvedStackOptions,
    ) -> StackLoweringPlan:
        raise NotImplementedError(
            "LEGACY_PURGED: wire canonical StackTrack ('target' vs 'proposed'/'shipping'/'eager') "
            "rewrite gating, populate StackLoweringPlan.track_spec (C-1), and dispatch "
            "ManifestBinding.lowering_pass_id purely via structural predicates"
        )

    def plan_iteration(
        self,
        model: ModelSpec,
        state: IterationSchedulerState,
        options: ResolvedStackOptions,
        *,
        phase: str,
    ) -> IterationDecision:
        _ensure_supported_domain(model)
        if not state.ready_request_ids:
            if state.in_flight_completions:
                next_req, until_ms = min(state.in_flight_completions, key=lambda item: item[1])
                return WaitForEvent(
                    reason="completion", until_ms=until_ms, event_id=f"completion:{next_req}",
                )
            return SchedulerDone(completed_requests=0, reason="all_requests_completed")
        raise NotImplementedError(
            "LEGACY_PURGED: emit q_len=1 + draft_proposal_tokens=spec_k (no 1+2*K "
            "double-count in iteration.py) and delegate capture padding exclusively to "
            "prediction/frontend/iteration.py (zero pre-frontend __capture_pad__)"
        )

    @property
    def launch_codec(self) -> LaunchConfigCodec:
        return self._codec


__all__ = [
    "VllmAscendStackAdapter",
]
