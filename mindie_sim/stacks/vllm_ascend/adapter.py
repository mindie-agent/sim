"""vLLM-Ascend ``RuntimeStackAdapter`` implementation.

[Role] Implement the authoritative ``RuntimeStackAdapter`` plugin for
  ``vllm-ascend``:
  - ``capabilities()`` declaring strict LLM/VLM scope (explicitly rejecting
    Diffusion/DiT with ``ModelingError("stack_domain_mismatch:...")``),
  - ``default_checkpoint_for()`` with strict A5 MXFP vs A2/A3 INT separation,
  - ``resolve_options()`` separating requested vs effective mechanisms,
    ACLGraph ``CaptureBucketRule`` (preserving ``FULL_AND_PIECEWISE`` and
    ``n*(K+1)``), speculative rules, shipping ``RecoverSSM`` on KDA + DSpark,
    MC2/MSO mutex, and per-die ``StreamOverlapPolicy``,
  - ``resolve_quant_instantiation()`` delegating to 1:1 ``quant_policy``,
  - ``select_lowering_plan()`` emitting full ``StackLoweringPlan`` +
    ``SchedulingContract`` with strict ``simulation`` vs ``modeling`` rewrite
    track enforcement,
  - ``plan_iteration()`` providing live one-step continuous batching,
  - ``launch_codec`` exposing ``VllmAscendLaunchCodec``.
[Invariants] Zero imports from ``mindie_sim.export`` or ``mindie_sim.compose``
  (AG010). ``simulation`` mode strictly limits ``allowed_rewrite_tracks`` to
  ``("config",)`` and forbids ``synthesized_target`` / ``tiler_extension``.
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
from mindie_sim.schema.scenario import OPTIMIZATION_MODES, OptimizationMode
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.stack import (
    STACK_ADAPTER_PROTOCOL_VERSION,
    STACK_TRACKS,
    CaptureBucketRule,
    FrameworkRewriteKind,
    IterationDecision,
    IterationSchedulerState,
    LaunchConfigCodec,
    PdKvTransferSpec,
    PdLayerTransferEdge,
    PdTransferMode,
    PdTransportKind,
    PiecewiseSplitPoint,
    ResolvedStackOptions,
    RewriteTrack,
    RuntimeQuantPlan,
    SchedulerDone,
    SchedulingContract,
    StackCapabilitySpec,
    StackLoweringPlan,
    StackTrack,
    StarsSubmitPolicy,
    StreamLaneBudget,
    StreamOverlapPolicy,
    StreamSyncEdge,
    WaitForEvent,
)
from mindie_sim.schema.workload import IterationPlan, RequestView

from .launch_codec import VllmAscendLaunchCodec
from .profiles import (
    FUSED_MC2_DISABLED,
    VLLM_ASCEND_STACK_ID,
    default_checkpoint_repo,
    hardware_backend_kind,
    infer_model_workload_domain,
    load_vllm_ascend_manifest,
    normalize_canonical_model,
    normalize_hw_family,
    resolve_fused_mc2_and_mso,
    resolve_manifest_bindings,
    shipping_collective_policy,
    simulation_config,
)
from .quant_policy import DEFAULT_KV_BLOCK_SIZE, instantiate_runtime_quant_plan
from .speculative import (
    DEFAULT_MAX_NUM_SEQS,
    build_capture_bucket_rule,
    build_pd_spec_configs,
    build_post_accept_commits,
    clamp_fia_speculative_tokens,
    draft_invocation_count_for_method,
    is_hybrid_recurrent_model,
    resolve_recover_ssm_policy,
    resolve_speculative_companion,
    validate_aclgraph_capture_sizes,
)

DEFAULT_SPEC_STEP_ACCEPTANCE = 0.7
DEFAULT_STARS_FIFO_LANES = 4
DEFAULT_STARS_QUEUE_DEPTH = 32
DEFAULT_STARS_EAGER_CADENCE_NS = 1200.0
DEFAULT_STARS_EAGER_FIRST_ISSUE_NS = 4500.0
DEFAULT_STARS_GRAPH_CADENCE_NS = 150.0
DEFAULT_STARS_GRAPH_FIRST_ISSUE_NS = 1800.0
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


def _build_default_speculative_spec(draft_len: int, *, source: str = "vllm-ascend") -> SpeculativeSpec:
    k = max(0, int(draft_len))
    marginals = tuple(DEFAULT_SPEC_STEP_ACCEPTANCE ** (idx + 1) for idx in range(k))
    return SpeculativeSpec(draft_len=k, acceptance=marginals, acceptance_mode="marginal", source=source)


def _resolve_speculative_option(
    model: ModelSpec,
    *,
    canon: str,
    hw: str,
    parallel: ParallelSpec,
    role_norm: str,
    requested_options: Mapping[str, Any],
    profile_cfg: Mapping[str, Any],
    effective_kv: str,
    dispositions: list[tuple[str, str, str]],
) -> tuple[SpeculativeSpec | None, str, int, bool]:
    """Resolve effective speculative spec, method, K, and draft eager flag."""
    req_spec = requested_options.get("speculative")
    req_k_raw = requested_options.get("num_speculative_tokens")
    req_method = str(requested_options.get("spec_method") or "").strip().lower() or None
    req_enable = requested_options.get("enable_speculative")
    spec_cfg_in = requested_options.get("speculative_config")
    if isinstance(spec_cfg_in, Mapping):
        if req_k_raw is None and "num_speculative_tokens" in spec_cfg_in:
            req_k_raw = spec_cfg_in["num_speculative_tokens"]
        if req_method is None and spec_cfg_in.get("method"):
            req_method = str(spec_cfg_in["method"]).strip().lower()

    if isinstance(req_spec, SpeculativeSpec):
        raw_k = req_spec.draft_len
    elif isinstance(req_spec, Mapping):
        parsed_spec = SpeculativeSpec.from_mapping(req_spec)
        req_spec = parsed_spec
        raw_k = parsed_spec.draft_len
    elif req_k_raw is not None:
        raw_k = int(req_k_raw)
    elif req_enable is False:
        raw_k = 0
    elif req_enable is True:
        raw_k = int(profile_cfg.get("mtp_proposal_tokens") or 0) or 1
    else:
        raw_k = int(profile_cfg.get("mtp_proposal_tokens") or 0)

    if raw_k <= 0:
        return None, "", 0, False

    eff_k, clamp_warn = clamp_fia_speculative_tokens(raw_k)
    if clamp_warn:
        dispositions.append(("num_speculative_tokens", "downgraded", clamp_warn))

    resolved = resolve_speculative_companion(
        canon, target_tp=parallel.tp, target_kv_dtype=effective_kv, hw_family=hw,
        method=req_method, num_speculative_tokens=eff_k,
    )
    eff_k = resolved.num_speculative_tokens
    draft_eager = bool(resolved.companion.enforce_eager_draft)
    if role_norm in _PD_ROLES:
        p_cfg, d_cfg, pd_rule = build_pd_spec_configs(resolved, is_hybrid=is_hybrid_recurrent_model(model))
        dispositions.append(("pd_speculative_rule", "applied", pd_rule))
        if role_norm == "pd_prefill":
            eff_k = int(p_cfg.get("num_speculative_tokens", eff_k))
            draft_eager = bool(p_cfg.get("enforce_eager", True))
        else:
            eff_k = int(d_cfg.get("num_speculative_tokens", eff_k))

    if isinstance(req_spec, SpeculativeSpec) and req_spec.draft_len == eff_k:
        spec_obj = req_spec
    else:
        spec_obj = _build_default_speculative_spec(eff_k, source=f"vllm-ascend:{resolved.method}")
    return spec_obj, resolved.method, eff_k, draft_eager


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
    mode = capture_rule.mode if capture_rule is not None else "none"
    if mode in {"full", "full_decode_only"}:
        replay = "full_graph"
        cadence, first_issue = DEFAULT_STARS_GRAPH_CADENCE_NS, DEFAULT_STARS_GRAPH_FIRST_ISSUE_NS
    elif mode in {"piecewise", "full_and_piecewise"}:
        replay = "captured_segment"
        cadence, first_issue = DEFAULT_STARS_GRAPH_CADENCE_NS, DEFAULT_STARS_GRAPH_FIRST_ISSUE_NS
    else:
        replay = "eager"
        cadence, first_issue = DEFAULT_STARS_EAGER_CADENCE_NS, DEFAULT_STARS_EAGER_FIRST_ISSUE_NS

    policies: list[StarsSubmitPolicy] = [
        StarsSubmitPolicy(
            controller_scope="die", die_id=0, stream_id="main", owner_kind="aic",
            fifo_lanes=DEFAULT_STARS_FIFO_LANES, queue_depth=DEFAULT_STARS_QUEUE_DEPTH,
            issue_cadence_ns=cadence, first_issue_cost_ns=first_issue,
            predecessor_policy="stream_fifo", credit_release_on="kernel_complete",
            replay_mode=replay,  # type: ignore[arg-type]
        ),
        StarsSubmitPolicy(
            controller_scope="die", die_id=0, stream_id="main", owner_kind="aiv",
            fifo_lanes=DEFAULT_STARS_FIFO_LANES, queue_depth=DEFAULT_STARS_QUEUE_DEPTH,
            issue_cadence_ns=cadence, first_issue_cost_ns=first_issue,
            predecessor_policy="stream_fifo", credit_release_on="kernel_complete",
            replay_mode=replay,  # type: ignore[arg-type]
        ),
    ]
    if stream_policy is not None:
        for lane in stream_policy.lanes:
            if lane.stream_id == "main":
                continue
            owner = "hccl" if lane.stream_id == "comm" else "mix"
            policies.append(StarsSubmitPolicy(
                controller_scope="die", die_id=lane.die_id, stream_id=lane.stream_id,
                owner_kind=owner,  # type: ignore[arg-type]
                fifo_lanes=DEFAULT_STARS_FIFO_LANES, queue_depth=DEFAULT_STARS_QUEUE_DEPTH,
                issue_cadence_ns=cadence, first_issue_cost_ns=first_issue,
                predecessor_policy="explicit_deps", credit_release_on="kernel_complete",
                replay_mode=replay,  # type: ignore[arg-type]
            ))
    return tuple(policies)


def _build_pd_kv_transfer_spec(
    quant_plan: RuntimeQuantPlan,
    *,
    hw: str,
    mode_str: str,
    transport_str: str,
) -> PdKvTransferSpec:
    mode: PdTransferMode = "layerwise_overlap" if mode_str == "layerwise_overlap" else "blocking"
    transport: PdTransportKind = (
        transport_str if transport_str in {"direct_d2d", "pooled_dram", "pooled_ssd"} else "direct_d2d"
    )  # type: ignore[assignment]
    edges: list[PdLayerTransferEdge] = []
    for abi in (*quant_plan.kv_cache_abi, *quant_plan.index_cache_abi, *quant_plan.recurrent_state_abi):
        if abi.scope != "target" or abi.component_role == "engram_table":
            continue
        n_bytes = abi.total_plane_bytes(include_padding=True)
        edges.append(PdLayerTransferEdge(
            native_layer=abi.native_layer,
            source_ready_event=f"ev.prefill_ready.{abi.producer_key}",
            dest_ready_event=f"ev.decode_ready.{abi.producer_key}",
            source_abi_key=abi.producer_key,
            dest_abi_key=abi.producer_key,
            byte_range=(0, n_bytes),
            requires_repack=False,
        ))
    route = "ub_direct" if hw == "a5" else ("hccs_sw" if hw == "a3" else "roce")
    return PdKvTransferSpec(
        mode=mode,
        transport=transport,
        source_pool_id="prefill",
        dest_pool_id="decode",
        layer_edges=tuple(edges),
        route=route,
        receive_credits=DEFAULT_PD_RECEIVE_CREDITS,
        last_reader_event="ev.decode_kv_consume_done",
    )


def _select_recipe_bindings(
    model: ModelSpec,
    *,
    track: StackTrack,
    canon: str,
    hw: str,
    eff_opts: Mapping[str, Any],
) -> tuple[
    tuple[tuple[str, str, tuple[tuple[str, Any], ...]], ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
]:
    manifest_hits = resolve_manifest_bindings(track, canon, hw)
    attns = {str(p.attention).lower() for p in model.layer_plans} or {str(model.attention).lower()}
    has_moe = any(str(p.ffn).lower() == "moe" for p in model.layer_plans) or model.num_experts > 0
    has_idx = any(p.indexer for p in model.layer_plans)
    has_kpool = any(p.index_kpool > 1 for p in model.layer_plans)
    has_attn_res = any(p.residual_block_size > 0 for p in model.layer_plans)

    bindings: list[tuple[str, str, tuple[tuple[str, Any], ...]]] = []
    passes: list[str] = []
    origins: list[tuple[str, str]] = []
    binding_ref = {b.mechanism_id: b.ref for b in manifest_hits}

    if track != "eager":
        passes.append("AddRMSNormQuantPass")
        if hw == "a5":
            passes.append("DynamicMxQuantFusionPass")
    if "mla" in attns and bool(eff_opts.get("mlapo", True)):
        bindings.append(("mla_prolog", "vllm_ascend_mla_prolog_v3", (("arch", hw),)))
        origins.append(("mla_prolog", binding_ref.get("mla_prolog_fusion", "vllm-ascend:shipping")))
    if "kda" in attns:
        rec_id = "vllm_ascend_recurrent_kda_arch35" if hw == "a5" else "vllm_ascend_recurrent_kda_arch22"
        bindings.append(("kda_recurrent", rec_id, (("arch", hw),)))
        bindings.append(("kda_proj", "vllm_ascend_kda_mixed_projection", ()))
        origins.append(("kda_recurrent", binding_ref.get("recurrent_decode_native_kernels", "vllm-ascend:shipping")))
    if "gdn" in attns:
        bindings.append(("gdn_decode", "vllm_ascend_gdn_decode", (("arch", hw),)))
        if hw == "a5":
            bindings.append(("gdn_prefill", "gdn_prefill_arch35", ()))
        origins.append(("gdn_decode", binding_ref.get("recurrent_decode_native_kernels", "vllm-ascend:shipping")))
    if has_idx:
        bindings.append(("indexer", "cann_quant_lightning_indexer_v2_complete", (("arch", hw),)))
        origins.append(("indexer", "vllm-ascend:shipping"))
    if has_kpool:
        bindings.append(("index_kpool", "vllm_ascend_cann_keypool_caller", ()))
    if has_attn_res and hw == "a5":
        bindings.append(("attn_res", "vllm_ascend_attn_res_prefix", ()))
        origins.append(("attn_res", binding_ref.get("attn_res_prefix_fusion", "vllm-ascend:shipping")))
    if has_moe and int(eff_opts.get("enable_fused_mc2") or 0) > FUSED_MC2_DISABLED:
        mc2_lvl = int(eff_opts["enable_fused_mc2"])
        bindings.append(("moe_mc2", "cann_megamoe_fusion", (("enable_fused_mc2", mc2_lvl),)))
        origins.append(("moe_mc2", binding_ref.get("cann_megamoe_fusion", "vllm-ascend:shipping")))
    for b in manifest_hits:
        origins.append((f"manifest.{b.mechanism_id}", b.ref or b.state))

    return tuple(bindings), tuple(passes), tuple(dict.fromkeys(origins))


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
        _ensure_supported_domain(model)
        hw = _ensure_supported_hardware(hardware)
        if track not in STACK_TRACKS:
            raise ModelingError(f"stack_track_invalid:{track!r}")

        canon = normalize_canonical_model(model)
        role_norm = str(role or "colocated").strip().lower()
        profile_cfg = simulation_config(canon, hardware.sku, hw, role=role_norm)
        dispositions: list[tuple[str, str, str]] = []

        raw_opt_mode = requested_options.get("optimization_mode")
        if raw_opt_mode is None:
            opt_mode: OptimizationMode = "modeling" if track == "target" else "simulation"
        else:
            mode_str = str(raw_opt_mode).strip().lower()
            if mode_str not in OPTIMIZATION_MODES:
                raise ModelingError(f"optimization_mode_invalid:{raw_opt_mode!r}")
            opt_mode = mode_str  # type: ignore[assignment]

        tut_flags = dict(profile_cfg.get("tutorial_flags") or {})
        eff_kv = str(
            requested_options.get("kv_cache_dtype")
            or tut_flags.get("kv_cache_dtype")
            or "bf16"
        ).strip().lower()

        spec_obj, spec_method, eff_k, draft_eager = _resolve_speculative_option(
            model, canon=canon, hw=hw, parallel=parallel, role_norm=role_norm,
            requested_options=requested_options, profile_cfg=profile_cfg,
            effective_kv=eff_kv, dispositions=dispositions,
        )

        recoverssm = resolve_recover_ssm_policy(
            model, track=track, parallel=parallel, spec_method=spec_method,
            proposal_count=eff_k, opt_in_replayssm=bool(requested_options.get("use_replayssm", False)),
        )
        dispositions.append((
            "recoverssm",
            "accepted" if recoverssm.enabled else "skipped",
            recoverssm.reason,
        ))

        req_mc2 = requested_options.get("enable_fused_mc2", tut_flags.get("enable_fused_mc2"))
        req_mso = bool(requested_options.get(
            "multistream_overlap_shared_expert",
            profile_cfg.get("multistream_overlap_shared_expert", False),
        ))
        eff_mc2, eff_mso, mc2_disps = resolve_fused_mc2_and_mso(
            arch=hw, model=model, parallel=parallel,
            requested_fused_mc2=int(req_mc2) if req_mc2 is not None else None,
            requested_mso=req_mso,
        )
        dispositions.extend(mc2_disps)

        comp_req = requested_options.get("compilation_config")
        raw_cg_mode = (
            requested_options.get("cudagraph_mode")
            or (comp_req.get("cudagraph_mode") if isinstance(comp_req, Mapping) else None)
            or ("NONE" if track == "eager" else "FULL_DECODE_ONLY")
        )
        enforce_eager = bool(requested_options.get("enforce_eager", False)) or (track == "eager")
        max_seqs = int(requested_options.get("max_num_seqs") or DEFAULT_MAX_NUM_SEQS)
        explicit_cap = (
            comp_req.get("cudagraph_capture_sizes")
            if isinstance(comp_req, Mapping) else requested_options.get("cudagraph_capture_sizes")
        )
        if isinstance(explicit_cap, (list, tuple)) and eff_k > 0:
            cap_warns = validate_aclgraph_capture_sizes(
                [int(x) for x in explicit_cap], num_speculative_tokens=eff_k, cudagraph_mode=str(raw_cg_mode),
            )
            if cap_warns:
                dispositions.append(("cudagraph_capture_sizes", "downgraded", cap_warns[0]))
                explicit_cap = None

        capture_rule = build_capture_bucket_rule(
            cudagraph_mode=str(raw_cg_mode),
            num_speculative_tokens=eff_k,
            max_num_seqs=max_seqs,
            explicit_capture_sizes=explicit_cap if isinstance(explicit_cap, (list, tuple)) else None,
            enforce_eager=enforce_eager,
            enforce_eager_draft=draft_eager or bool(requested_options.get("enforce_eager_draft", False)),
            allow_mixed=bool(requested_options.get("allow_mixed_capture", False)),
            allow_ragged=bool(requested_options.get("allow_ragged_capture", False)),
        )

        has_comm = (parallel.tp * parallel.ep * parallel.pp * parallel.pcp) > 1
        stream_policy = _build_stream_overlap_policy(
            hardware, model, track=track, eff_mso=eff_mso, has_comm=has_comm,
        )
        coll_dict = shipping_collective_policy(hw, canonical=canon, role=role_norm) or {}
        pd_mode = str(
            requested_options.get("pd_kv_transfer_mode")
            or ("layerwise_overlap" if (role_norm in _PD_ROLES and track != "eager") else "blocking")
        ).strip().lower()
        pd_transport = str(requested_options.get("pd_kv_transport") or "direct_d2d").strip().lower()

        effective_mechanisms: list[tuple[str, Any]] = [
            ("optimization_mode", opt_mode),
            ("role", role_norm),
            ("include_proposed_pr", bool(requested_options.get("include_proposed_pr", track == "proposed"))),
            ("kv_cache_dtype", eff_kv),
            ("indexer_kv_dtype", str(requested_options.get("indexer_kv_dtype") or tut_flags.get("indexer_kv_dtype") or "")),
            ("kv_block_size", int(requested_options.get("kv_block_size") or profile_cfg.get("kv_block_size") or DEFAULT_KV_BLOCK_SIZE)),
            ("mlapo", bool(requested_options.get("mlapo", profile_cfg.get("mlapo", False)))),
            ("enable_fused_mc2", eff_mc2),
            ("multistream_overlap_shared_expert", eff_mso),
            ("engram_cpu_offload", bool(requested_options.get("engram_cpu_offload", profile_cfg.get("engram_cpu_offload", False)))),
            ("enable_weight_nz_layout", bool(requested_options.get("enable_weight_nz_layout", hw in {"a2", "a3"}))),
            ("spec_method", spec_method),
            ("speculative_k", eff_k),
            ("recoverssm_enabled", recoverssm.enabled),
            ("recoverssm_kind", recoverssm.kind),
            ("recoverssm_owner_count", recoverssm.owner_count),
            ("recoverssm_source_ref", recoverssm.source_ref),
            ("pd_kv_transfer_mode", pd_mode),
            ("pd_kv_transport", pd_transport),
            ("chunked_prefill", bool(requested_options.get("chunked_prefill", True))),
            ("async_scheduling", bool(requested_options.get("async_scheduling", False))),
            ("target_quant_projection", str(requested_options.get("target_quant_projection") or "")),
            ("field_origin_count", len(field_origins)),
        ]
        requested_pairs = tuple(sorted((str(k), v) for k, v in requested_options.items() if not isinstance(v, SpeculativeSpec)))
        return ResolvedStackOptions(
            stack_id=VLLM_ASCEND_STACK_ID,
            track=track,
            requested_mechanisms=requested_pairs,
            effective_mechanisms=tuple(effective_mechanisms),
            dispositions=tuple(dispositions),
            capture_rule=capture_rule,
            graph_capture_mode=capture_rule.mode,
            graph_capture_keys=capture_rule.full_keys,
            piecewise_capture_keys=capture_rule.piecewise_keys,
            splitting_ops=_splitting_ops_for_model(model),
            speculative=spec_obj,
            stream_policy=stream_policy,
            collective_policy=tuple(sorted(coll_dict.items())),
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
        _ensure_supported_domain(model)
        hw = _ensure_supported_hardware(hardware)
        canon = normalize_canonical_model(model)
        eff = {str(k): v for k, v in options.effective_mechanisms}
        opt_mode = str(eff.get("optimization_mode") or "simulation").strip().lower()
        role_norm = str(eff.get("role") or "colocated").strip().lower()

        if opt_mode == "modeling":
            allowed_tracks = _MODELING_REWRITE_TRACKS
            allowed_readiness = _MODELING_READINESS
            allowed_fw = _ALL_FRAMEWORK_REWRITES
        else:
            allowed_tracks = _SIMULATION_REWRITE_TRACKS
            include_pr = bool(eff.get("include_proposed_pr", False)) or (options.track == "proposed")
            allowed_readiness = (
                frozenset({"shipping", "proposed_pr"}) if include_pr else frozenset({"shipping"})
            )
            allowed_fw = ()

        recipe_bindings, lowering_passes, binding_origins = _select_recipe_bindings(
            model, track=options.track, canon=canon, hw=hw, eff_opts=eff,
        )

        spec_method = str(eff.get("spec_method") or "")
        spec_k = int(eff.get("speculative_k") or (options.speculative.draft_len if options.speculative else 0))
        draft_invocations = draft_invocation_count_for_method(spec_method, spec_k)

        rec_policy = resolve_recover_ssm_policy(
            model, track=options.track, parallel=parallel, spec_method=spec_method,
            proposal_count=spec_k, opt_in_replayssm=bool(eff.get("recoverssm_enabled", False)),
        )
        rec_keys = tuple(
            abi.producer_key for abi in quant_plan.recurrent_state_abi
            if abi.component_role == "recurrent_state"
        )
        post_accept_commits = build_post_accept_commits(
            rec_policy, recurrent_abi_keys=rec_keys, accepted_prefix=max(1, spec_k) if spec_k > 0 else 0,
        )

        pd_transfer: PdKvTransferSpec | None = None
        if role_norm in _PD_ROLES:
            pd_transfer = _build_pd_kv_transfer_spec(
                quant_plan,
                hw=hw,
                mode_str=str(eff.get("pd_kv_transfer_mode") or "blocking"),
                transport_str=str(eff.get("pd_kv_transport") or "direct_d2d"),
            )

        scheduling = SchedulingContract(
            capture_bucket_rule=options.capture_rule,
            piecewise_split_points=_build_piecewise_split_points(model, options.capture_rule),
            stars_submit_policy=_build_stars_submit_policies(options.capture_rule, options.stream_policy),
            stream_overlap_policy=options.stream_policy,
            post_accept_commits=post_accept_commits,
            pd_kv_transfer=pd_transfer,
            chunked_prefill_enabled=bool(eff.get("chunked_prefill", True)),
            preemption_policy="recompute",
            async_scheduling=bool(eff.get("async_scheduling", False)),
            speculative_proposal_count=spec_k,
            draft_invocation_count=draft_invocations,
        )

        tensor_layouts: list[tuple[str, str, str]] = []
        seen_roles: set[str] = set()
        for coord, abi in quant_plan.port_abis:
            key_r = f"{coord[0]}.{coord[2]}"
            if key_r not in seen_roles:
                seen_roles.add(key_r)
                tensor_layouts.append((key_r, abi.weight_layout, abi.weight_dtype))
        for kv_abi in quant_plan.kv_cache_abi[:1]:
            tensor_layouts.append((f"kv.{kv_abi.component_role}", kv_abi.planes[0].layout, kv_abi.planes[0].dtype))

        return StackLoweringPlan(
            stack_id=VLLM_ASCEND_STACK_ID,
            track=options.track,
            recipe_bindings=recipe_bindings,
            lowering_passes=lowering_passes,
            allowed_rewrite_classes=allowed_readiness,
            allowed_rewrite_tracks=allowed_tracks,
            allowed_framework_rewrites=allowed_fw,
            scheduling=scheduling,
            tensor_layouts=tuple(tensor_layouts),
            binding_origins=binding_origins,
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

        prio_map = dict(state.request_priorities)
        prog_map = {rid: (ctx, rem) for rid, ctx, rem in state.request_progress}
        prefix_map = dict(state.prefix_hits)
        ordered = sorted(
            enumerate(state.ready_request_ids),
            key=lambda pair: (prio_map.get(pair[1], 0), pair[0]),
        )

        spec_k = (
            options.speculative.draft_len
            if (options.speculative is not None and phase == "decode")
            else 0
        )
        verify_stride = spec_k + 1
        token_budget = state.max_num_batched_tokens
        if state.max_num_input_tokens is not None:
            token_budget = min(token_budget, state.max_num_input_tokens)

        views: list[RequestView] = []
        statuses: list[tuple[str, str]] = []
        used_tokens = 0
        for _, rid in ordered:
            if len(views) >= state.max_num_seqs or used_tokens >= token_budget:
                statuses.append((rid, "queued"))
                continue
            ctx, rem_prompt = prog_map.get(rid, (0, 0))
            remaining_budget = token_budget - used_tokens
            if rem_prompt > 0:
                if not state.chunked_prefill and rem_prompt > remaining_budget:
                    statuses.append((rid, "deferred_budget"))
                    continue
                q_len = min(rem_prompt, remaining_budget) if state.chunked_prefill else rem_prompt
                views.append(RequestView(
                    request_id=rid,
                    q_len=q_len,
                    context=ctx,
                    samples=(q_len >= rem_prompt),
                    phase="prefill",
                    prefix_tokens=prefix_map.get(rid, 0),
                    priority=prio_map.get(rid, 0),
                    draft_verify=False,
                    draft_prepare=False,
                    prompt_tokens=ctx + rem_prompt,
                    global_q_len=q_len,
                ))
                used_tokens += q_len
                statuses.append((rid, "running_prefill"))
            else:
                q_len = verify_stride
                if q_len > remaining_budget:
                    statuses.append((rid, "deferred_budget"))
                    continue
                views.append(RequestView(
                    request_id=rid,
                    q_len=q_len,
                    context=ctx,
                    samples=True,
                    phase="decode",
                    prefix_tokens=prefix_map.get(rid, 0),
                    priority=prio_map.get(rid, 0),
                    draft_verify=(spec_k > 0),
                    draft_prepare=(spec_k > 0),
                    draft_proposal_tokens=spec_k,
                    global_q_len=q_len,
                ))
                used_tokens += q_len
                statuses.append((rid, "running_decode"))

        if not views:
            return WaitForEvent(reason="grant_release", until_ms=state.now_ms, event_id="token_budget_exhausted")

        is_mixed = len({v.phase for v in views}) > 1
        eff_phase = "mixed" if is_mixed else views[0].phase
        if options.capture_rule is not None and used_tokens > 0:
            bucket, _ = options.capture_rule.select_bucket(
                phase=eff_phase, active_tokens=used_tokens, is_mixed=is_mixed,
            )
            if bucket is not None and bucket > used_tokens:
                views.append(RequestView(
                    request_id="__capture_pad__",
                    q_len=bucket - used_tokens,
                    context=0,
                    samples=False,
                    phase=eff_phase if eff_phase != "mixed" else "decode",
                    padding=True,
                    draft_verify=False,
                    draft_prepare=False,
                ))

        return IterationPlan(
            requests=tuple(views),
            iteration=0,
            reason=f"vllm_ascend_scheduled:{eff_phase}",
            all_request_ids=tuple(state.ready_request_ids),
            request_statuses=tuple(statuses),
        )

    @property
    def launch_codec(self) -> LaunchConfigCodec:
        return self._codec


__all__ = [
    "VllmAscendStackAdapter",
]
