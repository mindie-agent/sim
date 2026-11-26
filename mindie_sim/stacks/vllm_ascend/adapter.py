"""vLLM-Ascend ``RuntimeStackAdapter`` implementation.

[Role] Implement the authoritative ``RuntimeStackAdapter`` plugin for
  ``vllm-ascend``:
  - ``capabilities()`` declaring strict LLM/VLM scope (explicitly rejecting
    Diffusion/DiT with ``ModelingError("stack_domain_mismatch:...")``),
  - ``default_checkpoint_for()`` delegating to pure checkpoint registry,
  - ``resolve_options()`` separating requested vs effective mechanisms,
    ACLGraph ``CaptureBucketRule``, strict speculative rules, RecoverSSM,
    MC2/MSO mutex, and per-die ``StreamOverlapPolicy``,
  - ``resolve_quant_instantiation()`` delegating to 1:1 ``quant_policy``,
  - ``select_lowering_plan()`` emitting full ``StackLoweringPlan`` +
    ``StackTrackSpec`` + ``SchedulingContract`` with strict ``StackTrack``
    enforcement and ``ProbedAbiRegistry``-grounded lowering,
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

from collections.abc import Mapping, Sequence
import math
from typing import Any

from mindie_sim.backends.abi_registry import default_abi_registry
from mindie_sim.catalog.context_parallel import is_v41_replicated_cache_model
from mindie_sim.catalog.draft_spec import (
    canonical_speculative_method,
    is_supported_speculative_method,
)
from mindie_sim.prediction.operators.mla_prolog import resolve_mla_preprocess_mode
from mindie_sim.primitives.loader import (
    _table_band_p50,
    load_manifest as load_primitive_manifest,
    load_primitives,
)
from mindie_sim.schema.checkpoint import CheckpointQuantMap
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.operators import KernelReadiness
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.speculative import SpeculativeSpec
from mindie_sim.schema.stack import (
    PD_TRANSFER_MODES,
    PD_TRANSPORT_KINDS,
    STACK_ADAPTER_PROTOCOL_VERSION,
    STACK_OPTIMIZATION_MODES,
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
    StackOptimizationMode,
    StackTrack,
    StackTrackSpec,
    StarsSubmitPolicy,
    StreamLaneBudget,
    StreamOverlapPolicy,
    StreamSyncEdge,
    WaitForEvent,
)
from mindie_sim.schema.workload import IterationPlan, RequestView

from .launch_codec import VllmAscendLaunchCodec
from .profiles import (
    A2_A3_INT_ONLY_FAMILIES,
    FUSED_MC2_DISABLED,
    VLLM_ASCEND_STACK_ID,
    default_checkpoint_repo,
    hardware_backend_kind,
    infer_model_workload_domain,
    load_vllm_ascend_manifest,
    normalize_hw_family,
    resolve_fused_mc2_and_mso,
    resolve_manifest_bindings,
    shipping_collective_policy,
    simulation_config,
)
from .quant_policy import DEFAULT_KV_BLOCK_SIZE, MXFP_GROUP_SIZE, instantiate_runtime_quant_plan
from .speculative import (
    ASCEND_V2_REJECTED_SPECULATIVE_METHODS,
    DEFAULT_MAX_NUM_SEQS,
    NONCAUSAL_BLOCK_DRAFT_METHODS,
    build_capture_bucket_rule,
    build_post_accept_commits,
    draft_invocation_count_for_method,
    is_hybrid_recurrent_model,
    resolve_recover_ssm_policy,
    validate_aclgraph_capture_sizes,
    validate_fia_speculative_tokens,
)

DEFAULT_STARS_FIFO_LANES = 4
DEFAULT_STARS_QUEUE_DEPTH = 32
DEFAULT_PD_RECEIVE_CREDITS = 4
MSO_AUX_AIC_DIVISOR = 4
MSO_AUX_AIV_DIVISOR = 4
MIN_STREAM_CORES = 1
NS_PER_US = 1000.0
_QLI_NATIVE_HEAD_DIM = 128
_QLI_NATIVE_HEADS = 64
_QLI_ARCH35_HEAD_COUNTS = frozenset({32, 64})
_QLI_NATIVE_TOPK_MAX = 2048
_QLI_NATIVE_COMPRESS_MAX = 128
_KDA_NATIVE_KEY_WIDTH = 128
_KDA_NATIVE_VALUE_WIDTHS = frozenset({128, 256})
_KDA_NATIVE_MAX_VALUE_HEADS = 256
_KDA_NATIVE_STATE_DTYPES = frozenset({"bf16", "fp32"})
_GDN_ARCH35_PREFILL_HEAD_DIM = 128
_KDA_BFG_PORT_ROLES = (
    "kda.b_proj", "kda.f_a_proj", "kda.f_b_proj", "kda.f_proj", "kda.g_proj",
    "b_proj", "f_a_proj", "f_b_proj", "f_proj", "g_proj",
)
_KDA_QKV_PORT_ROLES = (
    "kda.q_proj", "kda.k_proj", "kda.v_proj", "q_proj", "k_proj", "v_proj",
)

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
_VALID_ROLES = frozenset({"colocated", "pd_disaggregated", "pd_prefill", "pd_decode"})
_VALID_KV_DTYPE_TOKENS = frozenset({
    "auto", "none", "float", "bf16", "bfloat16", "fp16", "float16",
    "int8", "c8", "kv_c8", "int8_dynamic", "int8_sym", "faquant", "fakquant",
    "fp8", "fp8_e4m3", "fp8_dynamic", "mxfp8", "mxfp8_dynamic", "w8a8_mxfp8",
    "qk_mxfp8_dynamic_v_mxfp8_per_channel", "mxfp8_kv", "mxfp8_per_channel",
})
_A5_ONLY_KV_DTYPE_TOKENS = frozenset({
    "fp8", "fp8_e4m3", "fp8_dynamic", "mxfp8", "mxfp8_dynamic", "w8a8_mxfp8",
    "qk_mxfp8_dynamic_v_mxfp8_per_channel", "mxfp8_kv", "mxfp8_per_channel",
})


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


def _validate_kv_cache_dtype(raw_kv: Any, hw: str) -> str:
    if raw_kv is None:
        return ""
    norm = str(raw_kv).strip().lower()
    if not norm:
        return ""
    if norm not in _VALID_KV_DTYPE_TOKENS:
        raise ModelingError(f"kv_cache_dtype_invalid:{raw_kv!r}")
    if hw in A2_A3_INT_ONLY_FAMILIES and norm in _A5_ONLY_KV_DTYPE_TOKENS:
        raise ModelingError(f"kv_cache_dtype_unsupported_on_hardware:{hw}:{norm}")
    return norm


def _require_strict_spec_k(raw_k: Any) -> int:
    if isinstance(raw_k, bool) or not isinstance(raw_k, int):
        raise ModelingError(f"ascend_fia_speculative_tokens_invalid:{raw_k!r}")
    if raw_k < 0:
        raise ModelingError(f"ascend_fia_speculative_tokens_negative:{raw_k}")
    if raw_k == 0:
        return 0
    return validate_fia_speculative_tokens(raw_k)


def _infer_structural_spec_method(model: ModelSpec) -> str:
    aux = model.auxiliary
    if aux is not None and aux.dspark is not None:
        return "dspark"
    if aux is not None and aux.mtp is not None:
        mtp = aux.mtp
        if (
            str(mtp.kind).lower() in {"dspark", "mtp_dspark", "dflash"}
            or bool(getattr(mtp, "noncausal_query_block", False))
            or str(getattr(mtp, "execution_mode", "") or "") == "block_dspark"
        ):
            return "dspark"
        if any(str(p.attention).lower() == "gdn" for p in model.layer_plans):
            return "qwen3_5_mtp"
        if any(str(p.attention).lower() == "mla" for p in model.layer_plans):
            return "deepseek_mtp"
        return "mtp"
    return "eagle3"


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
    """Resolve speculative spec strictly (K <= 15, zero clamping, zero 0.7 fabrication)."""
    del hw, parallel, effective_kv
    req_spec = requested_options.get("speculative")
    req_k_raw = requested_options.get("num_speculative_tokens")
    req_method = str(requested_options.get("spec_method") or "").strip().lower() or None
    req_enable = requested_options.get("enable_speculative")
    spec_cfg_in = requested_options.get("speculative_config")
    cfg_acceptance: tuple[float, ...] | None = None
    cfg_dist: tuple[float, ...] | None = None
    cfg_eager_draft: bool | None = None
    if isinstance(spec_cfg_in, Mapping):
        if req_k_raw is None and "num_speculative_tokens" in spec_cfg_in:
            req_k_raw = spec_cfg_in["num_speculative_tokens"]
        if req_method is None and spec_cfg_in.get("method"):
            req_method = str(spec_cfg_in["method"]).strip().lower()
        if spec_cfg_in.get("acceptance") is not None:
            cfg_acceptance = tuple(float(x) for x in spec_cfg_in["acceptance"])
        if spec_cfg_in.get("accepted_distribution") is not None:
            cfg_dist = tuple(float(x) for x in spec_cfg_in["accepted_distribution"])
        if "enforce_eager_draft" in spec_cfg_in:
            cfg_eager_draft = bool(spec_cfg_in["enforce_eager_draft"])

    if isinstance(req_spec, SpeculativeSpec):
        raw_k = _require_strict_spec_k(req_spec.draft_len)
    elif isinstance(req_spec, Mapping):
        req_spec = SpeculativeSpec.from_mapping(req_spec)
        raw_k = _require_strict_spec_k(req_spec.draft_len)
    elif req_k_raw is not None:
        raw_k = _require_strict_spec_k(req_k_raw)
    elif req_enable is False:
        raw_k = 0
    elif req_enable is True:
        raise ModelingError("speculative_tokens_required_when_enabled")
    else:
        raw_k = 0

    if raw_k == 0:
        return None, "", 0, False

    if req_method is not None:
        if req_method in ASCEND_V2_REJECTED_SPECULATIVE_METHODS:
            raise ModelingError(f"unsupported_speculative_method_on_ascend_v2:{req_method}")
        if not is_supported_speculative_method(req_method):
            raise ModelingError(f"unsupported_speculative_method:{req_method}")
        spec_method = canonical_speculative_method(req_method)
    else:
        spec_method = _infer_structural_spec_method(model)

    is_hybrid = is_hybrid_recurrent_model(model)
    pd_matching_k = bool(
        is_hybrid
        or spec_method in NONCAUSAL_BLOCK_DRAFT_METHODS
    )
    eff_k = raw_k
    draft_eager = bool(cfg_eager_draft) if cfg_eager_draft is not None else False
    if role_norm in _PD_ROLES:
        if pd_matching_k:
            dispositions.append(("pd_speculative_rule", "applied", f"pd_matching_k:K={eff_k}"))
        elif role_norm == "pd_prefill":
            eff_k = 1
            draft_eager = True
            dispositions.append(("pd_speculative_rule", "applied", f"pd_p1_dk:prefill_k=1:decode_k={raw_k}"))
        else:
            dispositions.append(("pd_speculative_rule", "applied", f"pd_p1_dk:decode_k={eff_k}"))

    if isinstance(req_spec, SpeculativeSpec):
        if req_spec.draft_len == eff_k:
            spec_obj = req_spec
        else:
            spec_obj = SpeculativeSpec(
                draft_len=eff_k,
                acceptance=req_spec.acceptance_marginals()[:eff_k],
                acceptance_mode="marginal",
                source=req_spec.source or f"vllm-ascend:{spec_method}",
                label=req_spec.label,
            )
    elif cfg_dist is not None and len(cfg_dist) == eff_k + 1:
        spec_obj = SpeculativeSpec(
            draft_len=eff_k,
            accepted_distribution=cfg_dist,
            source=f"vllm-ascend:{spec_method}",
        )
    elif cfg_acceptance is not None and len(cfg_acceptance) >= eff_k:
        spec_obj = SpeculativeSpec(
            draft_len=eff_k,
            acceptance=cfg_acceptance[:eff_k],
            acceptance_mode="marginal",
            source=f"vllm-ascend:{spec_method}",
        )
    else:
        spec_obj = SpeculativeSpec(
            draft_len=eff_k,
            acceptance=tuple(0.0 for _ in range(eff_k)),
            acceptance_mode="marginal",
            source=f"vllm-ascend:{spec_method}",
            label="unmeasured_acceptance_lower_bound",
        )
    return spec_obj, spec_method, eff_k, draft_eager


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
            stream_id="comm", die_id=0, aic_budget=0,
            aiv_budget=max(MIN_STREAM_CORES, total_aiv // MSO_AUX_AIV_DIVISOR),
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


def _lookup_stars_primitive_timings(hardware: Hardware) -> tuple[float | None, float | None, float | None]:
    """Read calibrated P15_rts_stars timings from PrimitiveTable for SKU or arch family."""
    hw_family = normalize_hw_family(hardware)
    sku = str(hardware.sku or "").strip()
    candidates: list[str] = []
    if sku:
        candidates.append(sku)
    manifest = load_primitive_manifest()
    tables = manifest.get("tables") if isinstance(manifest, Mapping) else {}
    if isinstance(tables, Mapping):
        for key, entry in sorted(tables.items()):
            entry_sku = str((entry or {}).get("sku") or key)
            if normalize_hw_family(entry_sku) == hw_family and key not in candidates:
                candidates.append(str(key))
    for cand in candidates:
        try:
            tbl = load_primitives(cand)
        except ModelingError:
            continue
        if not tbl.provenance.calibrated:
            continue
        issue_ns = _table_band_p50(tbl, "P15_rts_stars", "issue_interval_ns")
        aiv_first_ns = _table_band_p50(tbl, "P15_rts_stars", "aiv_first_issue_ns")
        host_ns = _table_band_p50(tbl, "P15_rts_stars", "host_submit_ns")
        if issue_ns is not None or aiv_first_ns is not None or host_ns is not None:
            return issue_ns, aiv_first_ns, host_ns
    return None, None, None


def _build_stars_submit_policies(
    hardware: Hardware,
    capture_rule: CaptureBucketRule | None,
    stream_policy: StreamOverlapPolicy | None,
) -> tuple[StarsSubmitPolicy, ...]:
    """Source STARS cadence/first-issue timings strictly from hardware profile & PrimitiveTable."""
    die = hardware.dies[0] if hardware.dies else None
    stars_spec = (
        die.dispatch_sync.stars
        if (die is not None and die.dispatch_sync is not None)
        else None
    )
    fifo_lanes = int(stars_spec.lanes) if (stars_spec is not None and int(stars_spec.lanes) > 0) else DEFAULT_STARS_FIFO_LANES
    queue_depth = int(stars_spec.queue_depth) if (stars_spec is not None and int(stars_spec.queue_depth) > 0) else DEFAULT_STARS_QUEUE_DEPTH

    launch_ns = max(0.0, float(die.launch_us) * NS_PER_US) if die is not None else 0.0
    decl_issue_ns = (
        float(stars_spec.issue_interval_us) * NS_PER_US
        if (stars_spec is not None and float(stars_spec.issue_interval_us) > 0.0)
        else None
    )
    decl_first_ns = (
        float(stars_spec.first_issue_us) * NS_PER_US
        if (stars_spec is not None and float(stars_spec.first_issue_us) > 0.0)
        else None
    )
    p15_issue_ns, p15_first_ns, p15_host_ns = _lookup_stars_primitive_timings(hardware)

    mode = capture_rule.mode if capture_rule is not None else "none"
    if mode in {"full", "full_decode_only"}:
        replay = "full_graph"
        cadence = decl_issue_ns if decl_issue_ns is not None else (p15_issue_ns if p15_issue_ns is not None else launch_ns)
        first_issue = decl_first_ns if decl_first_ns is not None else (p15_first_ns if p15_first_ns is not None else cadence)
    elif mode in {"piecewise", "full_and_piecewise"}:
        replay = "captured_segment"
        cadence = decl_issue_ns if decl_issue_ns is not None else (p15_issue_ns if p15_issue_ns is not None else launch_ns)
        first_issue = decl_first_ns if decl_first_ns is not None else (p15_first_ns if p15_first_ns is not None else cadence)
    else:
        replay = "eager"
        cadence = decl_issue_ns if decl_issue_ns is not None else (p15_issue_ns if p15_issue_ns is not None else launch_ns)
        first_issue = decl_first_ns if decl_first_ns is not None else (
            p15_host_ns if p15_host_ns is not None else (p15_first_ns if p15_first_ns is not None else launch_ns)
        )

    policies: list[StarsSubmitPolicy] = [
        StarsSubmitPolicy(
            controller_scope="die", die_id=0, stream_id="main", owner_kind="aic",
            fifo_lanes=fifo_lanes, queue_depth=queue_depth,
            issue_cadence_ns=cadence, first_issue_cost_ns=first_issue,
            predecessor_policy="stream_fifo", credit_release_on="kernel_complete",
            replay_mode=replay,  # type: ignore[arg-type]
        ),
        StarsSubmitPolicy(
            controller_scope="die", die_id=0, stream_id="main", owner_kind="aiv",
            fifo_lanes=fifo_lanes, queue_depth=queue_depth,
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
                fifo_lanes=fifo_lanes, queue_depth=queue_depth,
                issue_cadence_ns=cadence, first_issue_cost_ns=first_issue,
                predecessor_policy="explicit_deps", credit_release_on="kernel_complete",
                replay_mode=replay,  # type: ignore[arg-type]
            ))
    return tuple(policies)


def _pd_local_tokens_for_abi(abi: Any, transfer_tokens: int) -> int:
    tokens = max(1, int(transfer_tokens))
    eff_tokens = min(tokens, int(abi.window)) if abi.window else tokens
    place = abi.placement
    if place.token_ownership_axis in {"dcp_head", "cp_striped"} and len(place.rank_ids) > 1:
        cp_shard = len(place.rank_ids)
        interleave = max(1, int(abi.cp_interleave_size or abi.kv_block_size or DEFAULT_KV_BLOCK_SIZE))
        full_cycles, rem = divmod(eff_tokens, cp_shard * interleave)
        return full_cycles * interleave + min(rem, interleave)
    return eff_tokens


def _pd_transfer_bytes_for_abi(abi: Any, transfer_tokens: int) -> int:
    """Compute exact per-rank valid transfer bytes for one ``CacheStateABI`` over ``transfer_tokens``."""
    local_tokens = _pd_local_tokens_for_abi(abi, transfer_tokens)
    block = max(1, int(abi.kv_block_size or 1))
    total = 0
    for idx_pl, plane in enumerate(abi.planes):
        if plane.allocation_frequency in {"checkpoint", "layer"} or plane.plane_id == "replay_records":
            continue
        unit_bytes = plane.exact_bytes()
        if plane.allocation_frequency == "request_slot":
            if plane.plane_id == "index_tail" or abi.component_role == "compressor_state":
                cap_rows = max(1, int(plane.logical_shape[0])) if plane.logical_shape else 1
                live_rows = min(local_tokens, cap_rows)
                total += (unit_bytes * live_rows) // cap_rows
            elif abi.component_role == "recurrent_state" and abi.rollback_slots > 0:
                if idx_pl == 0 or plane.plane_id == "recurrent_state":
                    total += unit_bytes // (1 + int(abi.rollback_slots))
                elif abi.history_slots > 0:
                    total += (unit_bytes * int(abi.history_slots)) // (int(abi.history_slots) + int(abi.rollback_slots))
                else:
                    total += unit_bytes
            else:
                total += unit_bytes
        elif plane.allocation_frequency in {"token", "token_group"}:
            rows = (
                abi.compressed_append_units(0, local_tokens)
                if (abi.compression_ratio > 1 or abi.pooling_ratio > 1)
                else local_tokens
            )
            total += rows * unit_bytes
        else:
            full_pages, rem_tok = divmod(local_tokens, block)
            plane_bytes = full_pages * unit_bytes
            if rem_tok > 0:
                if "block_group" in plane.logical_axes:
                    g_ext = int(plane.group_extents[0]) if plane.group_extents else MXFP_GROUP_SIZE
                    groups = (rem_tok + max(1, g_ext) - 1) // max(1, g_ext)
                    other = math.prod(d for ax, d in zip(plane.logical_axes, plane.logical_shape) if ax != "block_group")
                    tail_b = int(round(other * groups * dtype_bytes(plane.dtype) / max(1, plane.packed_elements)))
                elif "token" in plane.logical_axes:
                    other = math.prod(d for ax, d in zip(plane.logical_axes, plane.logical_shape) if ax != "token")
                    tail_b = int(round(other * rem_tok * dtype_bytes(plane.dtype) / max(1, plane.packed_elements)))
                else:
                    tail_b = (unit_bytes * rem_tok + block - 1) // block
                plane_bytes += min(unit_bytes, tail_b)
            total += plane_bytes
    return total


def _partial_page_requires_repack(abi: Any, local_tokens: int) -> bool:
    block = max(1, int(abi.kv_block_size or 1))
    if local_tokens % block == 0:
        return False
    for plane in abi.planes:
        if plane.allocation_frequency != "page" or "token" not in plane.logical_axes:
            continue
        tok_pos = plane.logical_axes.index("token")
        if any(int(dim) > 1 for dim in plane.physical_shape[:tok_pos]):
            return True
    return False


def _parse_c8_downgraded_layers(dispositions: Sequence[tuple[str, str, str]]) -> tuple[bool, set[tuple[str, int]]]:
    all_layers = False
    layers: set[tuple[str, int]] = set()
    for name, status, detail in dispositions:
        if name != "kv_cache_c8_producer" or status != "downgraded":
            continue
        if "layer=" not in detail:
            all_layers = True
            continue
        scope = "target"
        layer_id: int | None = None
        for part in detail.split(":"):
            if part.startswith("scope="):
                scope = part.split("=", 1)[1].strip()
            elif part.startswith("layer="):
                layer_id = int(part.split("=", 1)[1].strip())
        if layer_id is not None:
            layers.add((scope, layer_id))
    return all_layers, layers


def _build_pd_kv_transfer_spec(
    quant_plan: RuntimeQuantPlan,
    *,
    hw: str,
    mode_str: str,
    transport_str: str,
    role_norm: str = "colocated",
    transfer_tokens: int = 1,
) -> PdKvTransferSpec:
    """Validate PdTransferMode and PdTransportKind strictly with zero coercion."""
    if mode_str not in PD_TRANSFER_MODES:
        raise ModelingError(f"pd_transfer_mode_invalid:{mode_str!r}")
    if transport_str not in PD_TRANSPORT_KINDS:
        raise ModelingError(f"pd_transfer_transport_invalid:{transport_str!r}")
    mode: PdTransferMode = mode_str  # type: ignore[assignment]
    transport: PdTransportKind = transport_str  # type: ignore[assignment]
    c8_all_layers, c8_layers = _parse_c8_downgraded_layers(quant_plan.dispositions)
    conv_recipe = ""
    any_conversion = False
    edges: list[PdLayerTransferEdge] = []
    for abi in (*quant_plan.kv_cache_abi, *quant_plan.index_cache_abi, *quant_plan.recurrent_state_abi):
        if abi.scope not in {"target", "draft"} or abi.component_role == "engram_table":
            continue
        local_tokens = _pd_local_tokens_for_abi(abi, transfer_tokens)
        n_bytes = _pd_transfer_bytes_for_abi(abi, transfer_tokens)
        is_c8_mismatch = (
            abi.component_role == "kv_combined"
            and (
                abi.recipe_abi_id in {"gqa_int8", "gqa_faquant"}
                or (role_norm == "pd_prefill" and (c8_all_layers or (abi.scope, abi.native_layer) in c8_layers))
            )
        )
        edge_repack = is_c8_mismatch
        edge_recipe = ""
        if is_c8_mismatch:
            src_dt = abi.planes[0].dtype if (role_norm == "pd_prefill" and abi.planes[0].dtype in {"bf16", "fp16"}) else "bf16"
            edge_recipe = f"kv_c8_{src_dt}_to_int8_quantize"
            any_conversion = True
            conv_recipe = conv_recipe or edge_recipe
        elif _partial_page_requires_repack(abi, local_tokens):
            edge_repack = True
            edge_recipe = "kv_partial_page_gather_pack"
        edges.append(PdLayerTransferEdge(
            native_layer=abi.native_layer,
            source_ready_event=f"ev.prefill_ready.{abi.producer_key}",
            dest_ready_event=f"ev.decode_ready.{abi.producer_key}",
            source_abi_key=abi.producer_key,
            dest_abi_key=abi.producer_key,
            byte_range=(0, n_bytes),
            requires_repack=edge_repack,
            repack_recipe_id=edge_recipe,
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
        conversion_required=any_conversion,
        conversion_recipe_id=conv_recipe if any_conversion else "",
        last_reader_event="ev.decode_kv_consume_done",
    )


def _mla_quant_facts(
    model: ModelSpec, quant_plan: RuntimeQuantPlan, mla_layer_id: int,
) -> tuple[str, str, bool, bool, bool]:
    port_map = {coord: abi for coord, abi in quant_plan.port_abis}
    qa_abi = port_map.get(("target", mla_layer_id, "mla.q_a_proj"))
    if qa_abi is not None:
        act_dtype = str(qa_abi.output_dtype or qa_abi.input_dtype or "bf16").lower()
        weight_quant = str(qa_abi.linear_method or "bf16").lower()
        if weight_quant == "dense":
            weight_quant = str(qa_abi.weight_dtype or act_dtype).lower()
        is_w4a8 = qa_abi.linear_method == "w4a8"
    else:
        qspec = getattr(model, "quantization", None)
        act_dtype = str(getattr(qspec, "native_format", "") or model.activation_dtype or "bf16").lower()
        weight_quant = str(getattr(qspec, "weight_format", "") or model.weight_dtype or act_dtype).lower()
        is_w4a8 = weight_quant == "int4"

    is_int8_kv = False
    has_kv_scale = False
    for kv_abi in quant_plan.kv_cache_abi:
        if kv_abi.native_layer == mla_layer_id:
            if kv_abi.planes and kv_abi.planes[0].dtype == "int8":
                is_int8_kv = True
            has_kv_scale = any(p.plane_kind == "scale" for p in kv_abi.planes)
            break
    return act_dtype, weight_quant, is_w4a8, is_int8_kv, has_kv_scale


def _index_cache_dtypes(quant_plan: RuntimeQuantPlan) -> tuple[str, str]:
    """Return ``(payload_dtype, scale_dtype)`` from resolved ``index_cache_abi``."""
    for idx_abi in quant_plan.index_cache_abi:
        payload_dt = ""
        scale_dt = ""
        for plane in idx_abi.planes:
            if plane.plane_kind == "payload" and not payload_dt:
                payload_dt = str(plane.dtype).strip().lower()
            elif plane.plane_kind == "scale" and not scale_dt:
                scale_dt = str(plane.dtype).strip().lower()
        if payload_dt:
            return payload_dt, scale_dt
    return "", ""


def _bf16_lightning_indexer_fits_ub(
    model: ModelSpec,
    hardware: Hardware,
    idx_plans: Sequence[Any],
    eff_opts: Mapping[str, Any],
) -> bool:
    """Verify BF16 LightningIndexerV2 fits UB/L1 at reachable decode row counts."""
    if str(model.activation_dtype or "").strip().lower() not in {"bf16", "bfloat16"}:
        return False
    if any(int(p.index_heads or 0) < 1 or int(p.index_dim or 0) < 1 for p in idx_plans):
        return False
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.operators.sparse_indexer import bf16_lightning as li

    worst = max(idx_plans, key=lambda p: (int(p.index_heads or 0), int(p.index_topk or 0), int(p.index_dim or 0)))
    geometry = Geometry.from_devices(
        hardware.dies,
        links=hardware.links,
        transfer_channels=hardware.transfer_channels,
    )
    recipe = li.recipes()[li.RECIPE]
    page = max(1, int(eff_opts.get("kv_block_size") or DEFAULT_KV_BLOCK_SIZE))
    topk = max(1, int(worst.index_topk or 1))
    compression = max(1, int(worst.compress_ratio or 1))
    max_rows = max(1, int(eff_opts.get("max_num_seqs") or 1))
    probe_rows = (1,) if max_rows == 1 else (1, max_rows)
    for rows in probe_rows:
        context = max(page, topk * compression + rows)
        try:
            op = li.make_op(
                query_rows=rows, context=context,
                heads=int(worst.index_heads), head_dim=int(worst.index_dim),
                topk=topk, compression=compression, page_size=page,
                kv_heads=1, query_dtype="bf16", cache_dtype="bf16",
                weight_dtype="fp32", phase="decode",
            )
            recipe.build(op, geometry)
        except ModelingError:
            return False
    return True


def _mla_prolog_v3_options(
    model: ModelSpec,
    *,
    hw: str,
    parallel: ParallelSpec,
    is_int8_kv: bool,
    mla_mode: str,
) -> tuple[tuple[str, Any], ...] | None:
    """Validate ``vllm_ascend_mla_prolog_v3`` domain and emit canonical binding options."""
    mla_layers = [p for p in model.layer_plans if str(p.attention).lower() == "mla"]
    if not mla_layers or parallel.cp < 1 or parallel.pcp < 1:
        return None
    if str(model.activation_dtype or "").strip().lower() != "bf16":
        return None
    quant = getattr(model, "quant", None)
    if quant is None:
        return (("preprocess_mode", mla_mode),)
    from mindie_sim.catalog.ops import compute_dtype_for
    from mindie_sim.catalog.quant import projection_quant
    from mindie_sim.prediction.operators.mla_prolog.family_mla_prolog import (
        BF16_CACHE_DTYPES,
        CACHE_GROUP_SIZE,
        DRAFT_SCOPES,
        MLA_PROLOG_QUERY_LORA_RANKS,
        _indexer_q_category,
        _prolog_port_formats,
    )
    from mindie_sim.prediction.operators.mla_prolog.mla_prolog_packed import (
        LAYOUT as PACKED_MLA_LAYOUT,
        SUPPORTED_QUERY_FORMATS,
        view as packed_mla_view,
    )

    packed = hw in {"a3", "a5"} and is_int8_kv and str(quant.kv_dtype).strip().lower() == "int8"
    for plan in mla_layers:
        pquant = projection_quant(quant, plan)
        w_ports, act_ports = _prolog_port_formats(pquant)
        proj_w, proj_act = w_ports[0], act_ports[0]
        layer_kv_dtype = str(pquant.kv_dtype or "").strip().lower()
        layer_is_draft = getattr(plan, "scope", "body") in DRAFT_SCOPES
        layer_packed = packed and not (layer_is_draft and layer_kv_dtype in BF16_CACHE_DTYPES)
        if (
            len(set(w_ports)) != 1
            or len(set(act_ports)) != 1
            or proj_w not in {"bf16", "int8", "fp8_e4m3"}
            or (layer_packed and layer_kv_dtype != "int8")
            or (not layer_packed and layer_kv_dtype not in BF16_CACHE_DTYPES)
            or plan.mla_variant != "latent_mla"
            or plan.cache_kind != "dense"
            or int(plan.qk_rope_head_dim or 0) <= 0
            or (plan.q_lora_rank and plan.q_lora_rank not in MLA_PROLOG_QUERY_LORA_RANKS)
        ):
            return None
        if layer_packed:
            if proj_w not in SUPPORTED_QUERY_FORMATS or proj_act != proj_w:
                return None
            try:
                packed_mla_view(plan.kv_lora_rank, plan.qk_rope_head_dim, CACHE_GROUP_SIZE)
            except ModelingError:
                return None
        if proj_act != model.activation_dtype:
            if (
                (plan.indexer or plan.owns_topk or plan.index_query_per_layer)
                and compute_dtype_for(pquant, _indexer_q_category(pquant))[1] != proj_act
            ):
                return None
    if packed:
        return (("cache_layout", PACKED_MLA_LAYOUT), ("preprocess_mode", mla_mode))
    return (("preprocess_mode", mla_mode),)


def _select_mla_prolog_binding(
    model: ModelSpec,
    *,
    hw: str,
    hw_upper: str,
    phase: str,
    parallel: ParallelSpec,
    quant_plan: RuntimeQuantPlan,
    eff_opts: Mapping[str, Any],
    registry: Any,
) -> tuple[str, str, tuple[tuple[str, Any], ...]] | None:
    """Resolve M1 MLA prolog / MLAPO recipe binding grounded in ``ProbedAbiRegistry``."""
    mla_plan = next((p for p in model.layer_plans if str(p.attention).lower() == "mla"), None)
    if mla_plan is None:
        return None
    mla_nid = mla_plan.native_layer_id if mla_plan.native_layer_id >= 0 else mla_plan.index
    act_dtype, weight_quant, is_w4a8, is_int8_kv, has_kv_scale = _mla_quant_facts(
        model, quant_plan, mla_nid,
    )
    has_faquant = any(
        w.allocation_id.startswith("faquant.") for w in quant_plan.prepared_weights
    )
    eff_seqs = max(1, int(eff_opts.get("max_num_seqs") or 1))
    eff_spec_k = max(0, int(eff_opts.get("speculative_k") or 0))
    est_tokens = eff_seqs * (1 + eff_spec_k)
    max_batched = int(eff_opts.get("max_num_batched_tokens") or 0)
    if max_batched > 0:
        est_tokens = min(est_tokens, max_batched)
    is_sparse_sfa = bool(mla_plan.indexer)
    eff_has_kv_scale = has_kv_scale if is_sparse_sfa else (is_int8_kv and has_faquant)
    req_strided = eff_opts.get("is_strided_kv_cache")
    is_strided = (
        bool(req_strided)
        if req_strided is not None
        else bool(mla_plan.cache_kind == "dense" and not (is_int8_kv and eff_has_kv_scale))
    )
    is_kv_consumer = str(eff_opts.get("role") or "").strip().lower() == "pd_decode"
    mla_mode = resolve_mla_preprocess_mode(
        arch=hw_upper,
        phase=phase,
        num_tokens=max(1, est_tokens),
        q_lora_rank=int(mla_plan.q_lora_rank or 0),
        kv_lora_rank=int(mla_plan.kv_lora_rank or 0),
        qk_rope_head_dim=int(mla_plan.qk_rope_head_dim or 0),
        has_rope=(not bool(mla_plan.mla_use_nope)) and int(mla_plan.qk_rope_head_dim or 0) > 0,
        is_sparse_sfa=is_sparse_sfa,
        is_kv_consumer=is_kv_consumer,
        dcp_size=int(parallel.cp or 1),
        pcp_size=int(parallel.pcp or 1),
        enable_mlapo=bool(eff_opts.get("mlapo", False)),
        enable_mla_prolog=True,
        qkv_a_gate_fused=bool(mla_plan.mla_qkv_a_gate_fused or mla_plan.mla_output_gate),
        is_w4a8=is_w4a8,
        is_int8_kv=is_int8_kv,
        has_kv_scale=eff_has_kv_scale,
        has_learned_rope_tail=bool(
            str(mla_plan.mla_variant or "").lower() == "latent_mqa"
            and int(mla_plan.qk_rope_head_dim or 0) > 0
        ),
        is_strided_kv_cache=is_strided,
        act_dtype=act_dtype,
        weight_quant=weight_quant,
        input_size=int(model.hidden_size),
        num_kv_heads=int(mla_plan.num_kv_heads or 1),
        num_q_heads=int(mla_plan.num_heads or model.num_heads),
        qk_nope_head_dim=int(mla_plan.qk_nope_head_dim or 0),
        registry=registry,
    )
    if mla_mode in {"prolog_v3", "prolog_v3_learned_rope_tail"}:
        v3_opts = _mla_prolog_v3_options(
            model, hw=hw, parallel=parallel, is_int8_kv=is_int8_kv, mla_mode=mla_mode,
        )
        if v3_opts is None:
            return None
        return ("mla_prolog", "vllm_ascend_mla_prolog_v3", v3_opts)
    if mla_mode == "mlapo":
        return ("mla_prolog", "vllm_ascend_mlapo_decode", (("preprocess_mode", mla_mode),))
    return None


def _kda_recurrent_domain_ok(
    kda_layers: Sequence[Any], model: ModelSpec, parallel: ParallelSpec,
) -> bool:
    """Match ``family_kda_recurrent.validate`` & ``kda_recurrent.Params`` operand domains."""
    if str(model.activation_dtype or "").strip().lower() != "bf16":
        return False
    tp = max(1, int(parallel.tp or 1))
    for plan in kda_layers:
        total_v = int(plan.linear_value_heads or plan.linear_heads or 0)
        total_k = int(plan.linear_key_heads or plan.linear_heads or 0)
        dk = int(plan.linear_key_head_dim or plan.linear_head_dim or 0)
        dv = int(plan.linear_head_dim or 0)
        local_v = math.ceil(total_v / tp) if total_v > 0 else 0
        local_k = math.ceil(total_k / tp) if total_k > 0 else 0
        state_dtype = str(getattr(plan, "recurrent_state_dtype", "") or "").strip().lower()
        if (
            dk != _KDA_NATIVE_KEY_WIDTH
            or dv not in _KDA_NATIVE_VALUE_WIDTHS
            or not (1 <= local_k <= local_v <= _KDA_NATIVE_MAX_VALUE_HEADS)
            or local_v % local_k != 0
            or state_dtype not in _KDA_NATIVE_STATE_DTYPES
        ):
            return False
    return True


def _select_kda_bindings(
    model: ModelSpec,
    *,
    hw: str,
    parallel: ParallelSpec,
    quant_plan: RuntimeQuantPlan,
) -> tuple[tuple[str, str, tuple[tuple[str, Any], ...]], ...]:
    """Resolve KDA recurrent and mixed-projection bindings strictly within declared domains."""
    kda_layers = [p for p in model.layer_plans if str(p.attention).lower() == "kda"]
    if not kda_layers:
        return ()
    from mindie_sim.catalog.paired_sfa import execution as paired_sfa_execution

    bindings: list[tuple[str, str, tuple[tuple[str, Any], ...]]] = []
    if (
        parallel.pcp >= 1
        and (
            parallel.cp == 1
            or paired_sfa_execution(model).get("recurrent_context_sharded") is False
        )
        and _kda_recurrent_domain_ok(kda_layers, model, parallel)
    ):
        rec_id = (
            "vllm_ascend_recurrent_kda_arch35"
            if hw == "a5" else "vllm_ascend_recurrent_kda_arch22"
        )
        kda_opts: tuple[tuple[str, Any], ...] = ()
        if hw == "a5":
            from mindie_sim.prediction.operators.kda.kda_source import (
                options as kda_source_options,
            )
            opt_dicts = [
                kda_source_options(p, activation_dtype=model.activation_dtype)
                for p in kda_layers
            ]
            if opt_dicts and opt_dicts[0] is not None and all(d == opt_dicts[0] for d in opt_dicts):
                kda_opts = tuple(sorted(opt_dicts[0].items()))
        bindings.append(("kda", rec_id, kda_opts))

    quant = getattr(model, "quant", None)
    if quant is not None and str(model.activation_dtype or "").strip().lower() == "bf16":
        from mindie_sim.catalog.ops import compute_dtype_for
        from mindie_sim.catalog.quant import projection_quant
        from mindie_sim.prediction.operators.kda.family_kda_projection import _resolve_kda_f_quant
        from mindie_sim.prediction.projections import _representation

        tp = max(1, int(parallel.tp or 1))
        port_map = {
            (lid, role): abi
            for (sc, lid, role), abi in quant_plan.port_abis
            if sc == "target"
        }
        mixed_ok = True
        for plan in kda_layers:
            heads = plan.linear_value_heads or plan.linear_heads
            keys = plan.linear_key_heads or plan.linear_heads
            dk = plan.linear_key_head_dim or plan.linear_head_dim
            if (
                heads != keys
                or heads % tp != 0
                or plan.gate_rank
                or dk != plan.linear_head_dim
            ):
                mixed_ok = False
                break
            pquant = _resolve_kda_f_quant(projection_quant(quant, plan))
            qkv_reps = {_representation(pquant, f"kda.{role}") for role in ("q", "k", "v")}
            bfg_reps = {_representation(pquant, f"kda.{role}") for role in ("b", "f", "g")}
            if (
                len(qkv_reps) != 1
                or compute_dtype_for(pquant, "kda.q")[1] == model.activation_dtype
                or len(bfg_reps) != 1
                or any(
                    pquant.for_category(f"kda.{role}").format_for(f"kda.{role}") != "bf16"
                    for role in ("b", "f_a", "f_b", "f", "g")
                )
            ):
                mixed_ok = False
                break
            nid = plan.native_layer_id if plan.native_layer_id >= 0 else plan.index
            kda_qkv_abis = [
                port_map[(nid, role)]
                for role in _KDA_QKV_PORT_ROLES
                if (nid, role) in port_map
            ]
            if kda_qkv_abis and any(
                abi.linear_method == "dense"
                or str(abi.input_dtype or "").strip().lower() in {"", "bf16", "fp16", "fp32"}
                for abi in kda_qkv_abis
            ):
                mixed_ok = False
                break
            kda_bfg_abis = [
                port_map[(nid, role)]
                for role in _KDA_BFG_PORT_ROLES
                if (nid, role) in port_map
            ]
            if any(
                abi.linear_method != "dense"
                or str(abi.weight_dtype or "").strip().lower() not in {"", "bf16"}
                or str(abi.input_dtype or "").strip().lower() not in {"", "bf16"}
                for abi in kda_bfg_abis
            ):
                mixed_ok = False
                break
        if mixed_ok:
            bindings.append(("kda_input_projection", "vllm_ascend_kda_mixed_projection", ()))
    return tuple(bindings)


def _select_gdn_bindings(
    model: ModelSpec,
    *,
    hw: str,
    parallel: ParallelSpec,
) -> tuple[tuple[str, str, tuple[tuple[str, Any], ...]], ...]:
    """Resolve GDN decode and A5 prefill bindings strictly within declared kernel domains."""
    gdn_layers = [p for p in model.layer_plans if str(p.attention).lower() == "gdn"]
    if not gdn_layers or parallel.pcp < 1:
        return ()
    act_dtype = str(model.activation_dtype or "").strip().lower()
    if act_dtype != "bf16":
        return ()
    from mindie_sim.prediction.operators.linear_attention.gdn_decode import (
        _resolve_local_gdn_heads,
    )

    tp = max(1, int(parallel.tp or 1))
    decode_ok = True
    prefill_ok = hw == "a5"
    for plan in gdn_layers:
        state_dtype = str(getattr(plan, "recurrent_state_dtype", "") or "").strip().lower()
        conv_dtype = str(getattr(plan, "conv_state_dtype", "") or "").strip().lower()
        if state_dtype != "fp32" or conv_dtype != act_dtype:
            decode_ok = False
            prefill_ok = False
            break
        try:
            _keys, _heads, dk, dv, _rep = _resolve_local_gdn_heads(plan, tp)
        except ModelingError:
            decode_ok = False
            prefill_ok = False
            break
        if dk != _GDN_ARCH35_PREFILL_HEAD_DIM or dv != _GDN_ARCH35_PREFILL_HEAD_DIM:
            prefill_ok = False
    bindings: list[tuple[str, str, tuple[tuple[str, Any], ...]]] = []
    if decode_ok:
        gdn_opts: tuple[tuple[str, Any], ...] = (
            (("recurrent_implementation", "arch35_fp32"),) if hw == "a5" else ()
        )
        bindings.append(("gdn_decode", "vllm_ascend_gdn_decode", gdn_opts))
    if prefill_ok:
        bindings.append(("gdn_prefill", "gdn_prefill_arch35", ()))
    return tuple(bindings)


def _select_indexer_binding(
    model: ModelSpec,
    *,
    hw: str,
    hardware: Hardware,
    parallel: ParallelSpec,
    quant_plan: RuntimeQuantPlan,
    eff_opts: Mapping[str, Any],
) -> tuple[str, str, tuple[tuple[str, Any], ...]] | None:
    """Resolve INT8/FP8/BF16 LightningIndexerV2 binding strictly within declared operator domains."""
    idx_plans = [p for p in model.layer_plans if p.indexer and int(p.index_kpool or 1) == 1]
    if not idx_plans:
        return None
    abi_idx_dtype, abi_idx_scale = _index_cache_dtypes(quant_plan)
    q_idx = getattr(model, "quant", None)
    model_idx_dtype = str(getattr(q_idx, "index_dtype", "") or "").strip().lower()
    model_idx_scale = str(getattr(q_idx, "index_scale_dtype", "") or "").strip().lower()
    idx_dtype = abi_idx_dtype or model_idx_dtype or "bf16"
    idx_scale = abi_idx_scale or model_idx_scale
    if (
        idx_dtype == "int8"
        and idx_scale == "fp16"
        and (not model_idx_dtype or model_idx_dtype == "int8")
        and (not model_idx_scale or model_idx_scale == "fp16")
        and int(parallel.cp or 1) == 1
        and all(
            not p.candidate_filter
            and int(p.index_dim or 0) == _QLI_NATIVE_HEAD_DIM
            and int(p.index_heads or 0) == _QLI_NATIVE_HEADS
            and 1 <= int(p.index_topk or 0) <= _QLI_NATIVE_TOPK_MAX
            and 1 <= int(p.compress_ratio or 0) <= _QLI_NATIVE_COMPRESS_MAX
            and (int(p.compress_ratio or 0) & (int(p.compress_ratio or 0) - 1)) == 0
            for p in idx_plans
        )
    ):
        return ("quant_lightning_indexer", "cann_quant_lightning_indexer_v2_complete", ())
    if (
        hw == "a5"
        and idx_dtype == "fp8_e4m3"
        and idx_scale == "fp32"
        and (not model_idx_dtype or model_idx_dtype == "fp8_e4m3")
        and (not model_idx_scale or model_idx_scale == "fp32")
        and int(parallel.cp or 1) == 1
        and int(parallel.pcp or 1) >= 1
        and all(
            not p.candidate_filter
            and int(p.index_dim or 0) == _QLI_NATIVE_HEAD_DIM
            and int(p.index_heads or 0) in _QLI_ARCH35_HEAD_COUNTS
            and 1 <= int(p.index_topk or 0) <= _QLI_NATIVE_TOPK_MAX
            and 0 <= int(p.compress_ratio or 0) <= _QLI_NATIVE_COMPRESS_MAX
            and (int(p.compress_ratio or 0) & (int(p.compress_ratio or 0) - 1)) == 0
            for p in idx_plans
        )
    ):
        return ("quant_lightning_indexer", "cann_quant_lightning_indexer_v2_complete", ())
    if (
        idx_dtype in {"bf16", "bfloat16"}
        and all(not p.candidate_filter for p in idx_plans)
        and _bf16_lightning_indexer_fits_ub(model, hardware, idx_plans, eff_opts)
    ):
        return ("lightning_indexer", "cann_bf16_lightning_indexer_v2_complete", ())
    return None


def _select_sparse_kv_offload_binding(
    model: ModelSpec,
    raw_config: Any,
) -> tuple[str, str, tuple[tuple[str, Any], ...]] | None:
    """Validate and bind ``vllm_ascend_sparse_kv_offload`` strictly within its declared domain."""
    if raw_config is None or raw_config is False:
        return None
    if raw_config is True:
        opts_dict: dict[str, Any] = {}
    elif isinstance(raw_config, Mapping):
        opts_dict = dict(raw_config)
    elif isinstance(raw_config, Sequence) and not isinstance(raw_config, (str, bytes)):
        try:
            opts_dict = dict(raw_config)
        except (TypeError, ValueError) as exc:
            raise ModelingError(f"sparse_kv_offload_option_invalid:{raw_config!r}") from exc
    else:
        raise ModelingError(f"sparse_kv_offload_option_invalid:{raw_config!r}")

    if "enabled" in opts_dict:
        enabled_val = opts_dict.pop("enabled")
        if not isinstance(enabled_val, bool):
            raise ModelingError(f"sparse_kv_offload_enabled_invalid:{enabled_val!r}")
        if not enabled_val:
            return None

    from mindie_sim.prediction.operators.sparse_indexer.sparse_kv_offload import (
        IMPLEMENTATION as SPARSE_KV_OFFLOAD_IMPL,
        _resolve_options as resolve_sparse_kv_offload_options,
    )
    from mindie_sim.prediction.program.implementation_registry import select as select_implementation

    select_implementation(
        {"sparse_kv_offload": {"implementation": SPARSE_KV_OFFLOAD_IMPL, "options": opts_dict}},
        "sparse_kv_offload",
    )
    resolved = resolve_sparse_kv_offload_options(opts_dict)
    mla_layers = tuple(
        p for p in model.layer_plans
        if str(p.attention).lower() == "mla"
        and str(getattr(p, "mla_variant", "") or "") == "latent_mla"
    )
    indexed_layers = tuple(p for p in mla_layers if int(p.index_topk or 0) > 0)
    layerwise = bool(resolved["layerwise_kv_offload"])
    if not mla_layers or (not indexed_layers and not layerwise):
        return None
    return ("sparse_kv_offload", SPARSE_KV_OFFLOAD_IMPL, tuple(sorted(opts_dict.items())))


_BF16_FORMATS = frozenset({"bf16", "bfloat16"})


def _is_unquantized_bf16_moe(
    model: ModelSpec,
    quant_plan: RuntimeQuantPlan | None = None,
) -> bool:
    has_moe = any(str(p.ffn).lower() == "moe" for p in model.layer_plans) or model.num_experts > 0
    if not has_moe:
        return False
    q = model.quant
    w_fmt = str(getattr(q, "expert_weight_format", "") or getattr(q, "weight_format", "bf16") or "bf16").lower()
    a_fmt = str(getattr(q, "activation_format", "bf16") or "bf16").lower()
    if w_fmt not in _BF16_FORMATS or a_fmt not in _BF16_FORMATS:
        return False
    if quant_plan is not None:
        for coord, abi in quant_plan.port_abis:
            if "experts" in str(coord[2]).lower():
                if (
                    str(abi.linear_method).lower() != "dense"
                    or str(abi.weight_dtype).lower() not in _BF16_FORMATS
                    or str(abi.input_dtype).lower() not in _BF16_FORMATS
                    or str(abi.output_dtype).lower() not in _BF16_FORMATS
                ):
                    return False
    return True


def _resolve_grouped_gemm_bf16_nd_option(
    model: ModelSpec,
    *,
    hw: str,
    track: StackTrack,
    requested_options: Mapping[str, Any],
    dispositions: list[tuple[str, str, str]],
) -> bool:
    if not requested_options.get("grouped_gemm_bf16_nd"):
        return False
    if track == "eager":
        dispositions.append(("grouped_gemm_bf16_nd", "dropped", "disabled_on_eager_track"))
        return False
    if hw not in A2_A3_INT_ONLY_FAMILIES:
        dispositions.append(("grouped_gemm_bf16_nd", "dropped", f"unsupported_hw_family:{hw}"))
        return False
    if not _is_unquantized_bf16_moe(model):
        dispositions.append(("grouped_gemm_bf16_nd", "dropped", "requires_unquantized_bf16_moe"))
        return False
    return True


def _select_grouped_nd_bf16_binding(
    model: ModelSpec,
    *,
    hw: str,
    quant_plan: RuntimeQuantPlan,
    raw_config: Any,
) -> tuple[str, str, tuple[tuple[str, Any], ...]] | None:
    """Validate and bind ``ascendc_grouped_nd_bf16`` for Arch22 unquantized BF16 MoE grouped GEMM."""
    if not raw_config or hw not in A2_A3_INT_ONLY_FAMILIES:
        return None
    if not _is_unquantized_bf16_moe(model, quant_plan):
        return None
    from mindie_sim.prediction.program.implementation_registry import select as select_implementation

    select_implementation({"grouped_gemm": "ascendc_grouped_nd_bf16"}, "grouped_gemm")
    return ("grouped_gemm", "ascendc_grouped_nd_bf16", ())


_GENERAL_TIER_OFFLOAD_KEYS = frozenset({"kvc_ddr_ratio", "kvc_ssd_ratio", "expert_ddr_ratio"})


def _resolve_sparse_kv_offload_option(
    model: ModelSpec,
    *,
    track: StackTrack,
    requested_options: Mapping[str, Any],
    dispositions: list[tuple[str, str, str]],
) -> tuple[tuple[str, Any], ...] | None:
    """Resolve optional ``sparse_kv_offload`` / ``kv_offload`` config without altering default digests."""
    raw_sparse = requested_options.get("sparse_kv_offload")
    raw_kv = requested_options.get("kv_offload")
    if raw_sparse is None and raw_kv is None:
        return None
    raw_cfg = raw_sparse if raw_sparse is not None else raw_kv
    if raw_cfg is False:
        return None
    if raw_sparse is None and isinstance(raw_cfg, Mapping) and raw_cfg:
        filtered = {k: v for k, v in raw_cfg.items() if k not in _GENERAL_TIER_OFFLOAD_KEYS}
        if not filtered:
            return None
        raw_cfg = filtered
    if isinstance(raw_cfg, Mapping) and "enabled" in raw_cfg:
        en = raw_cfg["enabled"]
        if not isinstance(en, bool):
            raise ModelingError(f"sparse_kv_offload_enabled_invalid:{en!r}")
        if not en:
            return None
    binding = _select_sparse_kv_offload_binding(model, raw_cfg)
    excluded = {
        str(x).strip()
        for x in (requested_options.get("exclude_optimizations") or ())
        if str(x).strip()
    }
    if "sparse_kv_offload" in excluded or "vllm_ascend_sparse_kv_offload" in excluded:
        dispositions.append(("sparse_kv_offload", "rejected", "excluded_by_request"))
        return None
    if track == "eager":
        dispositions.append(("sparse_kv_offload", "rejected", "eager_track_disables_sparse_kv_offload"))
        return None
    if binding is None:
        dispositions.append((
            "sparse_kv_offload",
            "rejected",
            "sparse_kv_offload_requires_indexed_mla_or_layerwise_offload",
        ))
        return None
    layerwise = bool(dict(binding[2]).get("layerwise_kv_offload", False))
    dispositions.append(("sparse_kv_offload", "accepted", "layerwise" if layerwise else "indexed_decode"))
    return binding[2]


def _select_mqsmla_binding(
    model: ModelSpec,
    *,
    hw: str,
    phase: str,
    parallel: ParallelSpec,
    manifest_hits: tuple[Any, ...],
) -> tuple[str, str, tuple[tuple[str, Any], ...]] | None:
    """Bind A5 MQSMLA DSL decode recipe only when admitted by an active manifest binding."""
    if hw != "a5" or phase not in {"decode", "mixed"} or int(parallel.cp or 1) != 1:
        return None
    if not any(b.mechanism_id == "mqsmla_dsl_544_320" for b in manifest_hits):
        return None
    from mindie_sim.catalog.cache_format import CacheFormat, cache_format
    from mindie_sim.prediction.operators.paged_mla.family_mqsmla import (
        FORMATS as MQSMLA_FORMATS,
        KV_QUANT_FP8,
        KV_QUANT_MXFP8,
        quant_mode_of as mqsmla_quant_mode_of,
    )
    from mindie_sim.prediction.operators.paged_mla.mqsmla import (
        MQSMLA_HEAD_WIDTH,
        MQSMLA_ROPE_WIDTH,
        RECIPE as MQSMLA_RECIPE,
        supported_heads,
    )
    v41_layers = tuple(
        p for p in model.layer_plans
        if str(p.attention).lower() == "mla"
        and str(getattr(p, "mla_variant", "") or "") == "latent_mqa"
        and str(getattr(p, "cache_kind", "") or "") == "window_compressed"
    )
    if not v41_layers or not (
        any(p.compressor_kind == "gated_ring" for p in v41_layers)
        or is_v41_replicated_cache_model(model)
    ):
        return None
    if str(model.activation_dtype).strip().lower() != "bf16":
        return None
    if dict(model.metadata).get("cache_representations") is not None:
        if mqsmla_quant_mode_of(model, MQSMLA_HEAD_WIDTH) not in {KV_QUANT_FP8, KV_QUANT_MXFP8}:
            if any(
                cache_format(model, region) != CacheFormat(**decl)
                for region, decl in MQSMLA_FORMATS.items()
            ):
                return None
    tp = max(1, int(parallel.tp or 1))
    if any(
        int(p.num_kv_heads or 0) != 1
        or int(p.head_dim or 0) != MQSMLA_HEAD_WIDTH
        or int(p.qk_rope_head_dim or 0) != MQSMLA_ROPE_WIDTH
        or not supported_heads(int(p.num_heads or 0) // tp)
        or (int(getattr(p, "o_groups", 0) or 0) and int(p.o_groups) % tp != 0)
        for p in v41_layers
    ):
        return None
    return ("mqsmla_attention", MQSMLA_RECIPE, ())


def _select_recipe_bindings(
    model: ModelSpec,
    *,
    track: StackTrack,
    hw: str,
    hardware: Hardware,
    parallel: ParallelSpec,
    quant_plan: RuntimeQuantPlan,
    eff_opts: Mapping[str, Any],
    phase: str,
) -> tuple[
    tuple[tuple[str, str, tuple[tuple[str, Any], ...]], ...],
    tuple[str, ...],
    tuple[tuple[str, str], ...],
    tuple[Any, ...],
]:
    """Select ABI-grounded recipe bindings & manifest lowering passes purely structurally."""
    spec_k = int(eff_opts.get("speculative_k") or 0)
    sp_enabled = bool(eff_opts.get("sp", True))
    excluded = frozenset(
        str(x).strip() for x in (eff_opts.get("exclude_optimizations") or ()) if str(x).strip()
    )
    raw_hits = resolve_manifest_bindings(
        track, model, hw, phase=phase, parallel=parallel,
        quant_plan=quant_plan, speculative_k=spec_k, sp_enabled=sp_enabled,
    )
    manifest_hits = tuple(
        b for b in raw_hits
        if b.mechanism_id not in excluded and b.lowering_pass_id not in excluded
    )
    registry = default_abi_registry()
    hw_upper = hw.upper()

    attns = {str(p.attention).lower() for p in model.layer_plans} or {str(model.attention).lower()}
    has_moe = any(str(p.ffn).lower() == "moe" for p in model.layer_plans) or model.num_experts > 0
    has_idx = any(p.indexer for p in model.layer_plans)
    has_kpool = any(int(p.index_kpool or 1) > 1 for p in model.layer_plans)
    has_attn_res = any(int(p.residual_block_size or 0) > 0 for p in model.layer_plans)

    bindings: list[tuple[str, str, tuple[tuple[str, Any], ...]]] = []
    passes: list[str] = []
    origins: list[tuple[str, str]] = []

    for b in manifest_hits:
        if b.lowering_pass_id and b.lowering_pass_id not in passes:
            passes.append(b.lowering_pass_id)

    if "mla" in attns and track != "eager":
        mla_binding = _select_mla_prolog_binding(
            model, hw=hw, hw_upper=hw_upper, phase=phase, parallel=parallel,
            quant_plan=quant_plan, eff_opts=eff_opts, registry=registry,
        )
        if mla_binding is not None:
            bindings.append(mla_binding)
            origins.append((mla_binding[0], "stack_profile"))
        mqsmla_binding = _select_mqsmla_binding(
            model, hw=hw, phase=phase, parallel=parallel, manifest_hits=manifest_hits,
        )
        if mqsmla_binding is not None:
            bindings.append(mqsmla_binding)
            origins.append((mqsmla_binding[0], "stack_profile"))

    if "kda" in attns and track != "eager":
        for kda_binding in _select_kda_bindings(
            model, hw=hw, parallel=parallel, quant_plan=quant_plan,
        ):
            bindings.append(kda_binding)
            origins.append((kda_binding[0], "stack_profile"))

    if "gdn" in attns and track != "eager":
        for gdn_binding in _select_gdn_bindings(model, hw=hw, parallel=parallel):
            bindings.append(gdn_binding)
            origins.append((gdn_binding[0], "stack_profile"))

    if has_idx and track != "eager":
        idx_binding = _select_indexer_binding(
            model, hw=hw, hardware=hardware, parallel=parallel,
            quant_plan=quant_plan, eff_opts=eff_opts,
        )
        if idx_binding is not None:
            bindings.append(idx_binding)
            origins.append((idx_binding[0], "stack_profile"))

    if has_kpool and track != "eager":
        bindings.append(("pool_key_indexer", "vllm_ascend_cann_keypool_caller", ()))
        origins.append(("pool_key_indexer", "stack_profile"))

    if has_attn_res and track != "eager" and registry.is_shipping_available("AttnResFwd", hw_upper):
        bindings.append(("attention_residual", "vllm_ascend_attn_res_prefix", ()))
        origins.append(("attention_residual", "stack_profile"))

    if has_moe and track != "eager" and int(eff_opts.get("enable_fused_mc2") or 0) > FUSED_MC2_DISABLED:
        mc2_lvl = int(eff_opts["enable_fused_mc2"])
        bindings.append(("moe_mc2", "cann_megamoe_fusion", (("enable_fused_mc2", mc2_lvl),)))
        origins.append(("moe_mc2", "stack_profile"))

    if (
        track != "eager"
        and "sparse_kv_offload" in eff_opts
        and eff_opts["sparse_kv_offload"] is not None
        and "sparse_kv_offload" not in excluded
        and "vllm_ascend_sparse_kv_offload" not in excluded
    ):
        sparse_binding = _select_sparse_kv_offload_binding(model, eff_opts["sparse_kv_offload"])
        if sparse_binding is not None:
            bindings.append(sparse_binding)
            origins.append((sparse_binding[0], "stack_profile"))

    if (
        track != "eager"
        and eff_opts.get("grouped_gemm_bf16_nd")
        and "ascendc_grouped_nd_bf16" not in excluded
    ):
        grouped_nd = _select_grouped_nd_bf16_binding(
            model, hw=hw, quant_plan=quant_plan, raw_config=eff_opts.get("grouped_gemm_bf16_nd"),
        )
        if grouped_nd is not None:
            bindings.append(grouped_nd)
            origins.append((grouped_nd[0], "stack_profile"))

    return tuple(bindings), tuple(passes), tuple(dict.fromkeys(origins)), manifest_hits


def _validate_explicit_capture_sizes(
    raw_cap: Any, *, eff_k: int, cudagraph_mode: str,
) -> tuple[int, ...] | None:
    if raw_cap is None:
        return None
    if not isinstance(raw_cap, Sequence) or isinstance(raw_cap, (str, bytes)):
        raise ModelingError(f"cudagraph_capture_sizes_invalid:{raw_cap!r}")
    parsed: list[int] = []
    for item in raw_cap:
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            raise ModelingError(f"cudagraph_capture_size_invalid:{item!r}")
        parsed.append(int(item))
    if eff_k > 0 and parsed:
        cap_warns = validate_aclgraph_capture_sizes(
            parsed, num_speculative_tokens=eff_k, cudagraph_mode=cudagraph_mode,
        )
        if cap_warns:
            raise ModelingError(cap_warns[0])
    return tuple(parsed)


def _resolve_capture_and_pd_options(
    requested_options: Mapping[str, Any],
    *,
    track: StackTrack,
    role_norm: str,
    eff_k: int,
    draft_eager: bool,
) -> tuple[Any, str, str]:
    comp_req = requested_options.get("compilation_config")
    raw_cg_mode = (
        requested_options.get("cudagraph_mode")
        or (comp_req.get("cudagraph_mode") if isinstance(comp_req, Mapping) else None)
        or ("NONE" if track == "eager" else "FULL_DECODE_ONLY")
    )
    enforce_eager = bool(requested_options.get("enforce_eager", False)) or (track == "eager")
    max_seqs = int(requested_options.get("max_num_seqs") or DEFAULT_MAX_NUM_SEQS)
    raw_cap = (
        comp_req.get("cudagraph_capture_sizes")
        if isinstance(comp_req, Mapping) and "cudagraph_capture_sizes" in comp_req
        else requested_options.get("cudagraph_capture_sizes")
    )
    explicit_cap = _validate_explicit_capture_sizes(
        raw_cap, eff_k=eff_k, cudagraph_mode=str(raw_cg_mode),
    )
    capture_rule = build_capture_bucket_rule(
        cudagraph_mode=str(raw_cg_mode),
        num_speculative_tokens=eff_k,
        max_num_seqs=max_seqs,
        explicit_capture_sizes=explicit_cap,
        enforce_eager=enforce_eager,
        enforce_eager_draft=draft_eager or bool(requested_options.get("enforce_eager_draft", False)),
        allow_mixed=bool(requested_options.get("allow_mixed_capture", False)),
        allow_ragged=bool(requested_options.get("allow_ragged_capture", False)),
    )
    pd_mode = str(
        requested_options.get("pd_kv_transfer_mode")
        or ("layerwise_overlap" if (role_norm in _PD_ROLES and track != "eager") else "blocking")
    ).strip().lower()
    if pd_mode not in PD_TRANSFER_MODES:
        raise ModelingError(f"pd_transfer_mode_invalid:{pd_mode!r}")
    pd_transport = str(requested_options.get("pd_kv_transport") or "direct_d2d").strip().lower()
    if pd_transport not in PD_TRANSPORT_KINDS:
        raise ModelingError(f"pd_transfer_transport_invalid:{pd_transport!r}")
    return capture_rule, pd_mode, pd_transport


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

        role_norm = str(role or "colocated").strip().lower()
        if role_norm not in _VALID_ROLES:
            raise ModelingError(f"stack_role_invalid:{role!r}")
        profile_cfg = simulation_config(model, hardware.sku, hw, role=role_norm, model=model)
        dispositions: list[tuple[str, str, str]] = []

        raw_opt_mode = requested_options.get("optimization_mode")
        if raw_opt_mode is None:
            opt_mode: StackOptimizationMode = "modeling" if track == "target" else "simulation"
        else:
            mode_str = str(raw_opt_mode).strip().lower()
            if mode_str not in STACK_OPTIMIZATION_MODES:
                raise ModelingError(f"optimization_mode_invalid:{raw_opt_mode!r}")
            opt_mode = mode_str  # type: ignore[assignment]

        tut_flags = dict(profile_cfg.get("tutorial_flags") or {})
        eff_kv = _validate_kv_cache_dtype(
            requested_options.get("kv_cache_dtype") or tut_flags.get("kv_cache_dtype"),
            hw,
        )

        spec_obj, spec_method, eff_k, draft_eager = _resolve_speculative_option(
            model, hw=hw, parallel=parallel, role_norm=role_norm,
            requested_options=requested_options, effective_kv=eff_kv,
            dispositions=dispositions,
        )

        sp_enabled = bool(requested_options.get("sp", profile_cfg.get("sp", True)))
        active_bindings = resolve_manifest_bindings(
            track, model, hw, parallel=parallel, speculative_k=eff_k, sp_enabled=sp_enabled,
        )

        recoverssm = resolve_recover_ssm_policy(
            model, track=track, parallel=parallel, spec_method=spec_method,
            proposal_count=eff_k, opt_in_replayssm=bool(requested_options.get("use_replayssm", False)),
            active_bindings=active_bindings,
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

        sparse_offload_opts = _resolve_sparse_kv_offload_option(
            model, track=track, requested_options=requested_options, dispositions=dispositions,
        )

        capture_rule, pd_mode, pd_transport = _resolve_capture_and_pd_options(
            requested_options, track=track, role_norm=role_norm,
            eff_k=eff_k, draft_eager=draft_eager,
        )
        has_comm = (parallel.tp * parallel.ep * parallel.pp * parallel.pcp) > 1
        stream_policy = _build_stream_overlap_policy(
            hardware, model, track=track, eff_mso=eff_mso, has_comm=has_comm,
        )
        coll_dict = shipping_collective_policy(hw, role=role_norm, model=model) or {}

        effective_mechanisms: list[tuple[str, Any]] = [
            ("optimization_mode", opt_mode),
            ("role", role_norm),
            ("phase", str(requested_options.get("phase") or "").strip().lower()),
            ("exclude_optimizations", tuple(sorted(
                str(x).strip() for x in (requested_options.get("exclude_optimizations") or ()) if str(x).strip()
            ))),
            ("include_proposed_pr", bool(requested_options.get("include_proposed_pr", track in {"proposed", "target"}))),
            ("kv_cache_dtype", eff_kv),
            ("indexer_kv_dtype", str(requested_options.get("indexer_kv_dtype") or tut_flags.get("indexer_kv_dtype") or "")),
            ("kv_block_size", int(requested_options.get("kv_block_size") or profile_cfg.get("kv_block_size") or DEFAULT_KV_BLOCK_SIZE)),
            ("max_num_seqs", int(requested_options.get("max_num_seqs") or DEFAULT_MAX_NUM_SEQS)),
            ("max_num_batched_tokens", int(requested_options.get("max_num_batched_tokens") or 0)),
            ("prompt_tokens", int(requested_options.get("prompt_tokens") or 0)),
            ("pp_layer_partition", tuple(int(x) for x in requested_options["pp_layer_partition"]) if requested_options.get("pp_layer_partition") else ()),
            ("draft_max_model_len", int(requested_options.get("draft_max_model_len") or 0)),
            ("sp", sp_enabled),
            ("mlapo", bool(requested_options.get("mlapo", profile_cfg.get("mlapo", False)))),
            ("is_strided_kv_cache", (
                bool(requested_options["is_strided_kv_cache"])
                if requested_options.get("is_strided_kv_cache") is not None else None
            )),
            ("enable_fused_mc2", eff_mc2),
            ("multistream_overlap_shared_expert", eff_mso),
            ("engram_cpu_offload", bool(requested_options.get("engram_cpu_offload", profile_cfg.get("engram_cpu_offload", False)))),
            ("enable_weight_nz_layout", bool(requested_options.get("enable_weight_nz_layout", hw in A2_A3_INT_ONLY_FAMILIES))),
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
        if sparse_offload_opts is not None:
            effective_mechanisms.append(("sparse_kv_offload", sparse_offload_opts))
        if _resolve_grouped_gemm_bf16_nd_option(
            model, hw=hw, track=track, requested_options=requested_options, dispositions=dispositions,
        ):
            effective_mechanisms.append(("grouped_gemm_bf16_nd", True))
        requested_pairs = tuple(
            sorted((str(k), v) for k, v in requested_options.items() if not isinstance(v, SpeculativeSpec))
        )
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
        if options.track not in STACK_TRACKS:
            raise ModelingError(f"stack_track_invalid:{options.track!r}")
        eff = {str(k): v for k, v in options.effective_mechanisms}
        opt_mode_raw = str(eff.get("optimization_mode") or ("modeling" if options.track == "target" else "simulation")).strip().lower()
        if opt_mode_raw not in STACK_OPTIMIZATION_MODES:
            raise ModelingError(f"optimization_mode_invalid:{opt_mode_raw!r}")
        opt_mode: StackOptimizationMode = opt_mode_raw  # type: ignore[assignment]
        role_norm = str(eff.get("role") or "colocated").strip().lower()
        explicit_phase = str(eff.get("phase") or "").strip().lower()
        if explicit_phase in {"prefill", "decode", "mixed"}:
            phase_hint = explicit_phase
        else:
            abi_phases = (
                *(a.phase for a in quant_plan.kv_cache_abi[:1]),
                *(a.phase for a in quant_plan.recurrent_state_abi[:1]),
                *(a.phase for a in quant_plan.index_cache_abi[:1]),
            )
            abi_phase = next((str(p).strip().lower() for p in abi_phases if str(p).strip()), "")
            phase_hint = (
                abi_phase if abi_phase in {"prefill", "decode", "mixed"}
                else ("prefill" if role_norm == "pd_prefill" else "decode")
            )

        # Strict canonical StackTrack gate (C-1 / S2-W2-01): synthesis unlocks ONLY on target+modeling.
        allows_target_synthesis = (options.track == "target" and opt_mode == "modeling")
        if allows_target_synthesis:
            allowed_tracks = _MODELING_REWRITE_TRACKS
            allowed_readiness = _MODELING_READINESS
            allowed_fw = _ALL_FRAMEWORK_REWRITES
        else:
            allowed_tracks = _SIMULATION_REWRITE_TRACKS
            include_pr = options.track in {"proposed", "target"}
            allowed_readiness = (
                frozenset({"shipping", "proposed_pr"}) if include_pr else frozenset({"shipping"})
            )
            allowed_fw = ()

        recipe_bindings, lowering_passes, binding_origins, manifest_hits = _select_recipe_bindings(
            model, track=options.track, hw=hw, hardware=hardware, parallel=parallel,
            quant_plan=quant_plan, eff_opts=eff, phase=phase_hint,
        )
        track_spec = StackTrackSpec(
            stack_id=VLLM_ASCEND_STACK_ID,
            track=options.track,
            optimization_mode=opt_mode,
            active_bindings=manifest_hits,
            allowed_readiness=allowed_readiness,
            allows_target_synthesis=allows_target_synthesis,
        )

        spec_method = str(eff.get("spec_method") or "")
        spec_k = int(eff.get("speculative_k") or (options.speculative.draft_len if options.speculative else 0))
        draft_invocations = draft_invocation_count_for_method(spec_method, spec_k)

        rec_policy = resolve_recover_ssm_policy(
            model, track=options.track, parallel=parallel, spec_method=spec_method,
            proposal_count=spec_k, opt_in_replayssm=bool(eff.get("recoverssm_enabled", False)),
            active_bindings=manifest_hits,
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
                role_norm=role_norm,
                transfer_tokens=max(1, int(eff.get("prompt_tokens") or 1)),
            )

        scheduling = SchedulingContract(
            capture_bucket_rule=options.capture_rule,
            piecewise_split_points=_build_piecewise_split_points(model, options.capture_rule),
            stars_submit_policy=_build_stars_submit_policies(hardware, options.capture_rule, options.stream_policy),
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
            track_spec=track_spec,
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
        input_budget = (
            min(token_budget, state.max_num_input_tokens)
            if state.max_num_input_tokens is not None
            else token_budget
        )

        views: list[RequestView] = []
        statuses: list[tuple[str, str]] = []
        used_tokens = 0
        used_input_tokens = 0
        for _, rid in ordered:
            if len(views) >= state.max_num_seqs or used_tokens >= token_budget:
                statuses.append((rid, "queued"))
                continue
            ctx, rem_prompt = prog_map.get(rid, (0, 0))
            remaining_budget = token_budget - used_tokens
            if rem_prompt > 0:
                prefill_budget = min(remaining_budget, input_budget - used_input_tokens)
                if prefill_budget <= 0 or (not state.chunked_prefill and rem_prompt > prefill_budget):
                    statuses.append((rid, "deferred_budget"))
                    continue
                q_len = min(rem_prompt, prefill_budget) if state.chunked_prefill else rem_prompt
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
                used_input_tokens += q_len
                statuses.append((rid, "running_prefill"))
            else:
                if verify_stride > remaining_budget:
                    statuses.append((rid, "deferred_budget"))
                    continue
                # Emit q_len=1 + draft_proposal_tokens=spec_k (never 1+2*K double-count).
                views.append(RequestView(
                    request_id=rid,
                    q_len=1,
                    context=ctx,
                    samples=True,
                    phase="decode",
                    prefix_tokens=prefix_map.get(rid, 0),
                    priority=prio_map.get(rid, 0),
                    draft_verify=(spec_k > 0),
                    draft_prepare=(spec_k > 0),
                    draft_proposal_tokens=spec_k,
                    global_q_len=verify_stride,
                ))
                used_tokens += verify_stride
                statuses.append((rid, "running_decode"))

        if not views:
            return WaitForEvent(
                reason="grant_release", until_ms=state.now_ms, event_id="token_budget_exhausted",
            )

        # Capture padding belongs exclusively to prediction/frontend/iteration.py.
        return IterationPlan(
            requests=tuple(views),
            iteration=0,
            reason="scheduled",
            all_request_ids=tuple(state.ready_request_ids),
            request_statuses=tuple(statuses),
        )

    @property
    def launch_codec(self) -> LaunchConfigCodec:
        return self._codec


__all__ = [
    "VllmAscendStackAdapter",
]
