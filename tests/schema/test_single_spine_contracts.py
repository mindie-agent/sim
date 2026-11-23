"""Tier-0 single-spine L0 schema contracts, injection seam, and AG010 SCC tests.

[Role] Verify immutable L0 checkpoint, cache-state ABI, runtime stack adapter,
  explicit deployment-mode/SLA/domain routing, 4-stage/3-owner waterfall,
  co-design opportunity bounds, executable birth bindings, PredictionSpine
  contract injection, and AG010 stacks-layer + Tarjan SCC cycle enforcement.
[Invariants] Pure in-memory Tier-0 verification; zero subprocesses, network,
  or host-environment mutations; all synthetic fixtures stay under ``tmp_path``.
[Timeouts & Locks] Sub-second in-memory execution; no locks.
[Side-Effect Policy] Read-only over repository assets; temp files only in ``tmp_path``.
"""
from __future__ import annotations

from dataclasses import replace
import pickle

import pytest

from _support.axes import GEN5_DT_SKU
from mindie_sim.compose import compose_case
from mindie_sim.prediction.graph import GraphPolicy
from mindie_sim.prediction.results import ReplayOptions, StepRequest
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.structure import parse_case
from mindie_sim.primitives.loader import load_primitives
from mindie_sim.schema.case import Case, Software
from mindie_sim.schema.checkpoint import (
    CacheQuantRule,
    CheckpointProvenance,
    CheckpointQuantMap,
    LayerQuantException,
    TensorPresenceDomain,
    TensorQuantRule,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import (
    KernelBinding,
    OpSpec,
    OwnerMemoryResidency,
    OwnerScheduleSpec,
    SerialPhaseSpec,
    TensorPrecisionABI,
)
from mindie_sim.schema.scenario import (
    CoDesignOpportunity,
    ConcurrencyLimits,
    CrossHwOffloadSpec,
    DeploymentOutcome,
    MegaFusionBound,
    PoolSpec,
    ReadyStateAssumption,
    SLAContract,
    ScenarioContract,
    SpaceTimeTradeoffSpec,
    canonical_comparison_stages,
    resolve_default_stack_for_domain,
    validate_deployment_mode,
)
from mindie_sim.schema.stack import (
    CachePlacementRule,
    CachePlaneABI,
    CacheStateABI,
    CaptureBucketRule,
    IterationSchedulerState,
    PdKvTransferSpec,
    PdLayerTransferEdge,
    PiecewiseSplitPoint,
    PostAcceptCommitSpec,
    ResolvedStackOptions,
    RuntimeQuantPlan,
    SchedulerDone,
    SchedulingContract,
    StackCapabilitySpec,
    StackLoweringPlan,
    StarsSubmitPolicy,
    StreamLaneBudget,
    StreamOverlapPolicy,
    StreamSyncEdge,
    WaitForEvent,
)
from mindie_sim.prediction.results import LoweringRequest
from mindie_sim.schema.workload import RequestView
import scripts.check_arch_governance as arch_gov

pytestmark = pytest.mark.tier0

VLLM_ASCEND_CAP = StackCapabilitySpec(
    stack_id="vllm-ascend",
    manifest_revision="0c99d3fc",
    supported_domains=("llm", "vlm"),
    supported_work_units=("prefill_decode_step", "vision_encode"),
    supported_hardware_backends=("ascend",),
    supported_phases=("prefill", "decode", "mixed"),
    supported_checkpoint_formats=("modelslim_description", "hf_quantization_config", "unquantized_native"),
    supported_tracks=("eager", "shipping", "proposed", "target"),
    supported_recipe_ids=frozenset({"ascend_mla_prolog_v3", "ascend_gdn_decode"}),
    has_launch_codec=True,
)
DIFFUSION_CAP = StackCapabilitySpec(
    stack_id="mindie-sd",
    manifest_revision="1.0.0",
    supported_domains=("diffusion",),
    supported_work_units=("denoise_timestep", "vae_decode", "text_encode"),
    supported_hardware_backends=("ascend",),
    supported_phases=("denoise_step", "vae_decode", "text_encode"),
    supported_checkpoint_formats=("modelslim_description", "unquantized_native"),
    supported_tracks=("eager", "shipping", "target"),
    supported_recipe_ids=frozenset({"mindie_sd_sparse_attn"}),
    has_launch_codec=False,
)


def _minimal_case() -> Case:
    raw = compose_case(
        "qwen3-32b", hardware=GEN5_DT_SKU, tp=1, layers=1,
        prompt_tokens=16, output_tokens=2, concurrency=1,
    )
    return parse_case(raw)


def _sample_checkpoint() -> CheckpointQuantMap:
    prov = CheckpointProvenance(
        checkpoint_id="org/hybrid-mxfp8-c8",
        source_format="modelslim_description",
        source_revision="a" * 40,
        source_sha256="b" * 64,
    )
    return CheckpointQuantMap(
        provenance=prov,
        native_float_dtype="bf16",
        model_quant_type="W8A8_MXFP8",
        rules=(
            TensorQuantRule(
                pattern="model.layers.*.self_attn.o_proj",
                raw_tag="FLOAT",
                presence=TensorPresenceDomain(scope="target", layer_indices=(0, 1, 2, 3)),
            ),
            TensorQuantRule(
                pattern="model.layers.*.mlp.down_proj",
                raw_tag="W8A8_MIX",
                presence=TensorPresenceDomain(scope="target"),
            ),
        ),
        layer_exceptions=(
            LayerQuantException(
                pattern="model.layers.*.self_attn.o_proj",
                scope="target",
                layer_indices=(3,),
                raw_tag="W8A8_MXFP8",
            ),
        ),
        ignored_modules=("model.layers.1.mlp.down_proj",),
        cache_rule=CacheQuantRule(
            kv_tag="QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL",
            index_tag="INT8",
            kv_scale_dtype="e8m0",
            index_scale_dtype="fp32",
            layer_scoped_kv=((3, "C8"),),
            provenance="checkpoint",
        ),
    )


def _make_scenario(
    case: Case,
    *,
    deployment_mode: str = "colocated",
    workload_domain: str = "llm",
    work_unit_kind: str = "prefill_decode_step",
    optimization_mode: str = "simulation",
    execution_stage: str = "before",
    stack_id: str = "ideal-target",
    sla: SLAContract | None = None,
    pools: tuple[PoolSpec, ...] = (),
    allow_solver_pd_split: bool = False,
    num_devices: int | None = 1,
    ready_state: ReadyStateAssumption | None = None,
    pd_transfer: PdKvTransferSpec | None = None,
    allowed_readiness: frozenset[str] = frozenset({"shipping"}),
    allowed_tracks: tuple[str, ...] = ("config",),
    kv_cache_abi: tuple[CacheStateABI, ...] = (),
    primitive_table_digest: str = "",
) -> ScenarioContract:
    ckpt = _sample_checkpoint()
    opts = ResolvedStackOptions(stack_id=stack_id, track="shipping")
    qplan = RuntimeQuantPlan(
        checkpoint_digest=ckpt.digest(),
        stack_id=stack_id,
        kv_cache_abi=kv_cache_abi,
    )
    lplan = StackLoweringPlan(
        stack_id=stack_id,
        track="shipping",
        allowed_rewrite_classes=allowed_readiness,  # type: ignore[arg-type]
        allowed_rewrite_tracks=allowed_tracks,  # type: ignore[arg-type]
        scheduling=SchedulingContract(pd_kv_transfer=pd_transfer),
    )
    return ScenarioContract(
        deployment_mode=deployment_mode,  # type: ignore[arg-type]
        deployment_mode_origin="api_argument",
        workload_domain=workload_domain,  # type: ignore[arg-type]
        work_unit_kind=work_unit_kind,  # type: ignore[arg-type]
        optimization_mode=optimization_mode,  # type: ignore[arg-type]
        execution_stage=execution_stage,  # type: ignore[arg-type]
        model=case.model,
        checkpoint=ckpt,
        stack_id=stack_id,
        stack_track="shipping",
        stack_options=opts,
        quant_plan=qplan,
        lowering_plan=lplan,
        hardware=case.machine,
        primitive_table_digest=primitive_table_digest,
        sla=sla or SLAContract(),
        limits=ConcurrencyLimits(runtime_max_num_seqs=None, workload_requested_concurrency=16),
        workload=case.workload,
        num_devices=num_devices,
        pools=pools,
        allow_solver_pd_split=allow_solver_pd_split,
        ready_state=ready_state,
        pd_transfer=pd_transfer,
    )


def test_deployment_mode_contracts_and_nullable_ttft() -> None:
    case = _minimal_case()

    # Missing or silent auto deployment mode fails with crisp self-explanatory error listing all 4 modes.
    for bad_mode in (None, "", "auto", "pd_auto"):
        with pytest.raises(ModelingError, match="colocated.*pd_disaggregated.*pd_prefill.*pd_decode"):
            validate_deployment_mode(bad_mode)

    # 1) colocated: single pool, nullable TTFT preserved (zero 5000.0 fallback).
    coloc = _make_scenario(
        case,
        deployment_mode="colocated",
        pools=(PoolSpec("main", "colocated", num_devices=1, device_ids=(0,)),),
        sla=SLAContract(ttft_ms=None, tpot_ms=25.0),
    )
    assert coloc.sla.ttft_ms is None and coloc.sla.to_slo_spec().ttft_ms is None

    # 2) pd_disaggregated: requires disjoint P/D pools or explicit split budget + transfer + ABI match.
    good_xfer = PdKvTransferSpec(
        mode="layerwise_overlap",
        layer_edges=(PdLayerTransferEdge(0, "p.l0.ready", "d.l0.ready", "mla_528b", "mla_528b", (0, 528)),),
    )
    disagg = _make_scenario(
        case,
        deployment_mode="pd_disaggregated",
        num_devices=8,
        pools=(
            PoolSpec("p", "prefill", num_devices=4, device_ids=(0, 1, 2, 3)),
            PoolSpec("d", "decode", num_devices=4, device_ids=(4, 5, 6, 7)),
        ),
        pd_transfer=good_xfer,
    )
    assert disagg.deployment_mode == "pd_disaggregated"
    # Overlapping physical device IDs rejected.
    with pytest.raises(ModelingError, match="pd_disaggregated_pools_must_be_physically_disjoint"):
        _make_scenario(
            case,
            deployment_mode="pd_disaggregated",
            num_devices=4,
            pools=(
                PoolSpec("p", "prefill", num_devices=2, device_ids=(0, 1)),
                PoolSpec("d", "decode", num_devices=2, device_ids=(1, 2)),
            ),
            pd_transfer=good_xfer,
        )
    # Missing transfer rejected; incompatible role ABI without conversion rejected.
    with pytest.raises(ModelingError, match="pd_disaggregated_requires_transfer_spec"):
        _make_scenario(case, deployment_mode="pd_disaggregated", num_devices=4, allow_solver_pd_split=True)
    bad_abi_xfer = PdKvTransferSpec(
        mode="blocking",
        layer_edges=(PdLayerTransferEdge(0, "p.ready", "d.ready", "fp8_nd", "bf16_nz"),),
    )
    with pytest.raises(ModelingError, match="pd_disaggregated_incompatible_role_abi"):
        _make_scenario(
            case, deployment_mode="pd_disaggregated", num_devices=4,
            allow_solver_pd_split=True, pd_transfer=bad_abi_xfer,
        )

    # 3) pd_prefill: rejects decode TPOT/ITL target.
    prefill = _make_scenario(
        case, deployment_mode="pd_prefill",
        pools=(PoolSpec("p", "prefill", num_devices=1),),
        sla=SLAContract(ttft_ms=800.0, tpot_ms=None, tpot_boundary="unavailable"),
    )
    assert prefill.sla.tpot_ms is None
    with pytest.raises(ModelingError, match="pd_prefill_rejects_decode_sla"):
        _make_scenario(case, deployment_mode="pd_prefill", sla=SLAContract(tpot_ms=20.0))

    # 4) pd_decode: requires ReadyStateAssumption; prompt-arrival TTFT must be None.
    with pytest.raises(ModelingError, match="pd_decode_requires_ready_state_assumption"):
        _make_scenario(case, deployment_mode="pd_decode", sla=SLAContract(ttft_ms=None, ttft_boundary="unavailable"))
    ready = ReadyStateAssumption(source="pre_populated_hbm", context_tokens=32768, layout="paged_block_128")
    decode_sc = _make_scenario(
        case, deployment_mode="pd_decode", ready_state=ready,
        sla=SLAContract(ttft_ms=None, tpot_ms=15.0, ttft_boundary="unavailable", tpot_boundary="ready_state_decode"),
    )
    assert decode_sc.sla.ttft_ms is None
    with pytest.raises(ModelingError, match="pd_decode_ttft_unavailable"):
        _make_scenario(case, deployment_mode="pd_decode", ready_state=ready, sla=SLAContract(ttft_ms=5000.0))
    with pytest.raises(ModelingError, match="ready_state_pooled_restore_requires_transfer_spec"):
        ReadyStateAssumption(source="pooled_dram_restore", context_tokens=32768, layout="paged_block_128")

    # DeploymentOutcome honesty locks.
    outcome = DeploymentOutcome(
        status="infeasible", deployment_mode="pd_decode",
        limiting_constraint="static_weights_exceed_hbm", verified_ttft_ms=None,
        ttft_availability="unavailable",
    )
    assert outcome.winner_case is None and outcome.verified_ttft_ms is None
    with pytest.raises(ModelingError, match="deployment_outcome_winner_forbidden_for_status"):
        DeploymentOutcome(status="infeasible", deployment_mode="colocated", winner_case=case)
    with pytest.raises(ModelingError, match="deployment_outcome_pd_decode_ttft_must_be_none"):
        DeploymentOutcome(status="feasible", deployment_mode="pd_decode", verified_ttft_ms=5000.0)


def test_domain_aware_stack_routing_and_diffusion_guard() -> None:
    # LLM & VLM route to compatible vllm-ascend; Diffusion rejects LLM stack!
    assert resolve_default_stack_for_domain(
        "llm", work_unit="prefill_decode_step", available_stacks=(VLLM_ASCEND_CAP,),
    ) == "vllm-ascend"
    assert resolve_default_stack_for_domain(
        "vlm", work_unit="vision_encode", available_stacks=(VLLM_ASCEND_CAP,),
    ) == "vllm-ascend"
    with pytest.raises(ModelingError, match="no_compatible_stack_for_domain"):
        resolve_default_stack_for_domain(
            "diffusion", work_unit="denoise_timestep", available_stacks=(VLLM_ASCEND_CAP,),
        )
    with pytest.raises(ModelingError, match="stack_domain_mismatch"):
        resolve_default_stack_for_domain(
            "diffusion", work_unit="denoise_timestep",
            available_stacks=(VLLM_ASCEND_CAP,), requested_stack="vllm-ascend",
        )
    assert resolve_default_stack_for_domain(
        "diffusion", work_unit="denoise_timestep", available_stacks=(VLLM_ASCEND_CAP, DIFFUSION_CAP),
    ) == "mindie-sd"
    with pytest.raises(ModelingError, match="workload_domain_work_unit_mismatch"):
        resolve_default_stack_for_domain(
            "diffusion", work_unit="prefill_decode_step", available_stacks=(DIFFUSION_CAP,),
        )


def test_heterogeneous_cache_state_abi_and_checkpoint_precedence() -> None:
    # Checkpoint exception precedence: layer_exceptions -> ignored_modules -> rules.
    ckpt = _sample_checkpoint()
    assert ckpt.resolve_raw_tag("model.layers.3.self_attn.o_proj", source_native_layer_id=3) == "W8A8_MXFP8"
    assert ckpt.resolve_raw_tag("model.layers.2.self_attn.o_proj", source_native_layer_id=2) == "FLOAT"
    assert ckpt.resolve_raw_tag("model.layers.1.mlp.down_proj", source_native_layer_id=1) == "FLOAT"
    assert ckpt.resolve_raw_tag("model.layers.2.mlp.down_proj", source_native_layer_id=2) == "W8A8_MIX"
    assert ckpt.cache_rule.kv_tag_for_layer(3) == "C8"
    assert ckpt.cache_rule.kv_tag_for_layer(0) == "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL"

    # Kimi MLA NoPE + rotary exact byte invariants (520 B / 528 B / 656 B).
    mla_fp16_scales = CacheStateABI(
        scope="target", native_layer=3, component_role="mla_latent", producer_key="target.l3.mla_nope_fp16",
        planes=(
            CachePlaneABI("latent", "payload", ("dim",), (512,), (512,), "int8", bytes_per_unit=512),
            CachePlaneABI("scale", "scale", ("groups",), (4,), (4,), "fp16", bytes_per_unit=8),
        ),
        kv_block_size=128,
    )
    mla_fp32_rotary = CacheStateABI(
        scope="target", native_layer=4, component_role="mla_latent", producer_key="target.l4.mla_rotary",
        planes=(
            CachePlaneABI("latent", "payload", ("dim",), (512,), (512,), "int8", bytes_per_unit=512),
            CachePlaneABI("scale", "scale", ("groups",), (4,), (4,), "fp32", bytes_per_unit=16),
            CachePlaneABI("rotary", "suffix", ("dim",), (64,), (64,), "bf16", bytes_per_unit=128),
        ),
        kv_block_size=128,
    )
    assert mla_fp16_scales.total_plane_bytes() == 520
    assert mla_fp32_rotary.total_plane_bytes() == 656

    # QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL: K | K-scale | V | V-scale | padding = 33,792 B before trailing align.
    mxfp_page = CacheStateABI(
        scope="target", native_layer=0, component_role="kv_combined", producer_key="target.l0.mxfp_kv",
        planes=(
            CachePlaneABI("k_payload", "payload", ("t", "h", "d"), (128, 1, 128), (128, 1, 128), "fp8_e4m3"),
            CachePlaneABI("k_scale", "scale", ("t", "h", "g"), (128, 1, 4), (128, 1, 4), "e8m0", allocation_frequency="token"),
            CachePlaneABI("v_payload", "payload", ("t", "h", "d"), (128, 1, 128), (128, 1, 128), "fp8_e4m3"),
            CachePlaneABI(
                "v_scale", "scale", ("t", "h", "g"), (128, 1, 4), (128, 1, 4), "e8m0",
                allocation_frequency="page", write_trigger="page_reassign",
            ),
            CachePlaneABI("pad", "padding", ("b",), (512,), (512,), "uint8", bytes_per_unit=512),
        ),
        kv_block_size=128,
    )
    assert mxfp_page.total_plane_bytes(include_padding=False) == 33_792

    # Qwen3.8 KPool + GDN recurrent state & RecoverSSM version advancement.
    kpool = CacheStateABI(
        scope="target", native_layer=3, component_role="index_k", producer_key="target.l3.kpool",
        planes=(CachePlaneABI("kpool", "payload", ("t", "d"), (1, 128), (1, 128), "bf16"),),
        pooling_ratio=4, tail_slots=3, draft_context_cap=64,
        placement=CachePlacementRule(pp_stage=0, rank_ids=(0,), head_range=(0, 8), state_shard_axis="tp"),
    )
    assert kpool.compressed_append_units(context_tokens=10, query_tokens=4) == 1
    assert kpool.compressed_append_units(context_tokens=12, query_tokens=3) == 0

    qplan = RuntimeQuantPlan(
        checkpoint_digest=ckpt.digest(), stack_id="ideal-target",
        kv_cache_abi=(mla_fp16_scales, mla_fp32_rotary, mxfp_page),
        index_cache_abi=(kpool,),
    )
    assert qplan.cache_abi_for("target.l3.kpool") == kpool
    with pytest.raises(ModelingError, match="must_be_typed_cache_state_abi"):
        RuntimeQuantPlan(checkpoint_digest="d", stack_id="s", kv_cache_abi=({"raw": 1},))  # type: ignore[arg-type]
    with pytest.raises(ModelingError, match="duplicate_cache_owner"):
        RuntimeQuantPlan(checkpoint_digest="d", stack_id="s", kv_cache_abi=(kpool,), index_cache_abi=(kpool,))

    commit = PostAcceptCommitSpec(
        recurrent_abi_key="target.l0.kda_state", request_id="r0", slot_id=0, accepted_prefix=3,
        base_version=4, published_version=5,
        verify_record_ready_event="ev.verify", acceptance_ready_event="ev.accept",
        base_version_ready_event="ev.base", publish_event="ev.pub",
        owner_count=69, batched_kernel_launches=1,
    )
    assert commit.owner_count == 69 and commit.published_version == 5
    with pytest.raises(ModelingError, match="post_accept_version_must_advance"):
        replace(commit, published_version=4)


def test_executable_birth_binding_and_four_stage_three_owner_waterfall() -> None:
    abi_mx = TensorPrecisionABI(
        linear_method="mx_fp8", input_dtype="fp8_e4m3", weight_dtype="fp8_e4m3",
        compute_dtype="fp8_e4m3", weight_block=(1, 32), weight_scale_dtype="e8m0",
        act_group_size=32, act_scale_dtype="e8m0",
    )
    abi_bf16 = TensorPrecisionABI(linear_method="dense", input_dtype="bf16", weight_dtype="bf16", compute_dtype="bf16")
    assert abi_mx.abi_signature != abi_bf16.abi_signature
    assert not abi_mx.is_fusion_compatible(abi_bf16)

    with pytest.raises(ModelingError, match="kernel_binding_banned_backend"):
        KernelBinding(kernel_id="kda_fused", backend="cannbot-dsl", arch_family="dav_3510")  # type: ignore[arg-type]
    with pytest.raises(ModelingError, match="owner_residency_ub_overflow"):
        OwnerMemoryResidency(ub_live_bytes=300_000, ub_capacity_bytes=262_144)

    sched = OwnerScheduleSpec(
        owner_kind="aiv", tiling_strategy="coop_row_group", parallel_axis="token_rows",
        work_tiles=42, launched_owners=42, active_owners=42, total_owners=64, max_tiles_per_owner=1,
        residency=OwnerMemoryResidency(ub_live_bytes=131_072, ub_capacity_bytes=262_144, spill_reload_passes=1),
        serial_phases=(SerialPhaseSpec("score_reduce", "vec_reduce", 4096.0, "op", dependent_chain_steps=6),),
    )
    binding = KernelBinding(
        kernel_id="aclnnAttnResFwd", backend="Custom-Ops", arch_family="dav_3510",
        readiness="synthesized_target", template_id="templates/attn_res_coop",
    )
    logical_op = OpSpec("op0", "attn_res", "decode", 0, 0, ("r0",), (14, 4096), (14, 4096), (14, 4096), (14, 4096))
    assert not logical_op.is_executable_bound
    with pytest.raises(ModelingError, match="op_missing_precision_abi"):
        logical_op.require_executable_binding()
    bound_op = replace(logical_op, precision_abi=abi_bf16, kernel_binding=binding, owner_schedule=sched)
    assert bound_op.is_executable_bound and bound_op.require_executable_binding() == (abi_bf16, binding)

    # 4-stage / 3-owner waterfall: simulation vs modeling.
    sim_stages = canonical_comparison_stages("simulation")
    assert [s.stage for s in sim_stages] == ["before", "config_only", "after"]
    assert all("synthesized_target" not in s.allowed_readiness for s in sim_stages)

    mod_stages = canonical_comparison_stages("modeling")
    assert [s.stage for s in mod_stages] == ["before", "config_only", "framework_rewrite", "after"]
    assert [s.owner for s in mod_stages] == ["baseline", "ops_config", "framework_python", "kernel_dev_tune"]
    assert "synthesized_target" not in mod_stages[2].allowed_readiness
    assert {"synthesized_target", "tiler_extension"} <= mod_stages[3].allowed_readiness

    # StackLoweringPlan enforces track <-> readiness invariants; simulation forbids synthesized_target.
    with pytest.raises(ModelingError, match="lowering_plan_kernel_new_requires_synthesized_target_readiness"):
        StackLoweringPlan(
            stack_id="ideal-target", track="shipping",
            allowed_rewrite_classes=frozenset({"shipping"}), allowed_rewrite_tracks=("config", "kernel_new"),
        )
    case = _minimal_case()
    with pytest.raises(ModelingError, match="simulation_mode_forbids_non_shipping_rewrites"):
        _make_scenario(
            case, optimization_mode="simulation", execution_stage="after",
            allowed_readiness=frozenset({"shipping", "synthesized_target"}),
            allowed_tracks=("config", "kernel_new"),
        )
    fwk_sc = _make_scenario(
        case, optimization_mode="modeling", execution_stage="framework_rewrite",
        allowed_readiness=frozenset({"shipping"}), allowed_tracks=("config", "framework"),
    )
    assert fwk_sc.execution_stage == "framework_rewrite"
    with pytest.raises(ModelingError, match="framework_rewrite_stage_forbids_new_kernels"):
        _make_scenario(
            case, optimization_mode="modeling", execution_stage="framework_rewrite",
            allowed_readiness=frozenset({"shipping", "synthesized_target"}),
            allowed_tracks=("config", "framework", "kernel_new"),
        )


def test_codesign_opportunity_bounds_and_invariants() -> None:
    # 1) MegaFusionBound: gain_expected <= gain_ceiling and tiled >= L* and UB fit.
    mf = MegaFusionBound(
        region_op_ids=("l0.rmsnorm", "l0.qkv_proj"),
        exposed_before_ms=1.20, boundary_lower_bound_ms=0.70, tiled_engine_ms=0.82,
        gain_ceiling_ms=0.50, gain_expected_ms=0.38,
        eliminated_gm_bytes=65536, ub_live_bytes=131_072, ub_capacity_bytes=262_144,
    )
    opp_mf = CoDesignOpportunity(
        opportunity_id="opp.mf.1", kind="mega_fusion", track="kernel_new",
        readiness="synthesized_target", title="Fuse RMSNorm + QKV", mega_fusion=mf,
    )
    assert opp_mf.mega_fusion is mf
    with pytest.raises(ModelingError, match="mega_fusion_primal_below_boundary_bound"):
        MegaFusionBound(
            region_op_ids=("l0.op",), exposed_before_ms=1.0, boundary_lower_bound_ms=0.8,
            tiled_engine_ms=0.6, gain_ceiling_ms=0.2, gain_expected_ms=0.4,
            eliminated_gm_bytes=0, ub_live_bytes=1024, ub_capacity_bytes=4096,
        )

    # 2) SpaceTimeTradeoffSpec: rejects feasible=True when delta_hbm exceeds headroom.
    st = SpaceTimeTradeoffSpec(
        direction="space_for_time", mechanism="recover_ssm_snapshot_ring",
        delta_hbm_bytes=16 * 1024 * 1024, free_hbm_headroom_bytes=64 * 1024 * 1024,
        delta_step_ms=-0.45, delta_concurrency_per_dp=0, delta_tps_per_die=12.5,
        feasible_under_hbm_and_sla=True,
    )
    opp_st = CoDesignOpportunity(
        opportunity_id="opp.st.1", kind="space_time_tradeoff", track="framework",
        readiness="shipping", title="RecoverSSM snapshot ring", space_time=st,
    )
    assert opp_st.space_time is st
    with pytest.raises(ModelingError, match="space_time_hbm_exceeds_headroom"):
        replace(st, delta_hbm_bytes=128 * 1024 * 1024)

    # 3) CrossHwOffloadSpec: net_exposed_ms == max(0, transfer - hide) and payload matching.
    offload = CrossHwOffloadSpec(
        target_component="engram_table", source_tier="host_ddr", transfer_engine="h2d",
        compute_hide_window_ms=1.5, transfer_ms=1.8, net_exposed_ms=0.3,
        freed_hbm_bytes=4 * 1024**3, delta_concurrency_per_dp=8, delta_tps_per_die=45.0,
    )
    opp_hw = CoDesignOpportunity(
        opportunity_id="opp.hw.1", kind="cross_hw_offload_multistream", track="framework",
        readiness="shipping", title="Overlap Engram H2D across early layers", cross_hw=offload,
    )
    assert opp_hw.cross_hw is offload
    with pytest.raises(ModelingError, match="cross_hw_net_exposed_inconsistent"):
        replace(offload, net_exposed_ms=0.0)
    with pytest.raises(ModelingError, match="codesign_mega_fusion_payload_mismatch"):
        CoDesignOpportunity(
            opportunity_id="opp.bad", kind="mega_fusion", track="kernel_new",
            readiness="synthesized_target", title="Bad payload", space_time=st,
        )


def test_capture_eligibility_preserves_full_and_piecewise() -> None:
    rt = Software.from_mapping({
        "compilation_config": {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 2, 4, 8],
        },
    })
    assert rt.captured_execution_mode == "FULL_AND_PIECEWISE"
    assert rt.supports_full_decode_capture is True
    assert rt.supports_piecewise_capture is True
    assert rt.effective_capture_scope("decode") == "full"
    assert rt.effective_capture_scope("prefill") == "piecewise"

    dec_view = (RequestView("r0", q_len=1, context=128, samples=True, phase="decode"),)
    pref_view = (RequestView("r0", q_len=4, context=0, samples=False, phase="prefill"),)
    assert GraphPolicy().select(dec_view, rt) == (1, "full")
    assert GraphPolicy().select(pref_view, rt) == (4, "piecewise")

    rule = CaptureBucketRule(
        mode="full_and_piecewise", full_keys=(2, 4, 6), piecewise_keys=(4, 8),
        verify_stride=2, enforce_eager_scopes=("draft",),
    )
    assert rule.select_bucket(phase="decode", active_tokens=4) == (4, "full")
    assert rule.select_bucket(phase="prefill", active_tokens=4) == (4, "piecewise")
    assert rule.select_bucket(phase="decode", active_tokens=4, scope="draft") == (None, "none")

    sched_contract = SchedulingContract(
        capture_bucket_rule=rule,
        piecewise_split_points=(PiecewiseSplitPoint("s0.l0.attn", segment_index=1, selected_key=4, physical_rows=4),),
        stars_submit_policy=(StarsSubmitPolicy(die_id=0, stream_id="main", owner_kind="aic", fifo_lanes=2, queue_depth=8),),
        stream_overlap_policy=StreamOverlapPolicy(
            lanes=(StreamLaneBudget("main", aic_budget=16, aiv_budget=32), StreamLaneBudget("aux", aic_budget=8, aiv_budget=16)),
            sync_edges=(StreamSyncEdge("ev.mso", "main", "l0.gate", "aux", ("l0.shared_ffn",)),),
            service_terminal="step_done",
            enabled_overlaps=("shared_expert",),
        ),
        speculative_proposal_count=4,
        draft_invocation_count=1,
    )
    assert sched_contract.speculative_proposal_count == 4 and sched_contract.draft_invocation_count == 1
    live_state = IterationSchedulerState(now_ms=12.5, ready_request_ids=("r0",), max_num_seqs=16, max_num_batched_tokens=2048)
    wait_ev = WaitForEvent(reason="arrival", until_ms=20.0)
    done_ev = SchedulerDone(completed_requests=1)
    assert live_state.now_ms == 12.5 and wait_ev.until_ms == 20.0 and done_ev.completed_requests == 1


def test_prediction_spine_injection_seam_and_primitive_table_lock() -> None:
    case = _minimal_case()
    table = load_primitives(GEN5_DT_SKU)
    before_sc = _make_scenario(
        case, optimization_mode="modeling", execution_stage="before",
        primitive_table_digest=table.envelope_sha256,
    )
    after_sc = _make_scenario(
        case, optimization_mode="modeling", execution_stage="after",
        allowed_readiness=frozenset({"shipping", "proposed_pr", "synthesized_target", "tiler_extension"}),
        allowed_tracks=("config", "framework", "kernel_new", "kernel_tune"),
        primitive_table_digest=table.envelope_sha256,
    )
    spine = PredictionSpine()
    step_req = StepRequest(phase="decode", tokens=1, context=16, sequences=1, scope="body")
    req_before = LoweringRequest(step=step_req, scenario_contract=before_sc)
    prepared_before = spine.lower(case, req_before)
    prepared_after = spine.lower(
        case, LoweringRequest(step=step_req),
        scenario_contract=after_sc,
        runtime_quant_plan=after_sc.quant_plan,
        stack_lowering_plan=after_sc.lowering_plan,
    )
    assert prepared_before.scenario_contract is before_sc
    assert prepared_after.stack_lowering_plan is after_sc.lowering_plan

    ev_before = spine.run_step(prepared_before, ReplayOptions(), primitive_table=table)
    ev_after = spine.run_step(prepared_after, ReplayOptions(), primitive_table=table)
    assert ev_before.execution_stage == "before" and ev_after.execution_stage == "after"
    assert ev_before.stack_id == "ideal-target" and ev_before.primitive_table_digest == table.envelope_sha256
    assert ev_before.primitive_table_digest == ev_after.primitive_table_digest
    assert ev_before.span_ms > 0.0 and ev_after.span_ms == pytest.approx(ev_before.span_ms)

    restored = pickle.loads(pickle.dumps(ev_after))
    assert restored.execution_stage == "after" and restored.scenario_digest == ev_after.scenario_digest

    # Swapping PrimitiveTable digest against ScenarioContract fails loudly.
    bad_sc = _make_scenario(case, primitive_table_digest="0" * 64)
    bad_step = spine.lower(case, LoweringRequest(step=step_req, scenario_contract=bad_sc))
    with pytest.raises(ModelingError, match="spine_primitive_table_digest_mismatch"):
        spine.run_step(bad_step, ReplayOptions(), primitive_table=table)


def test_ag010_stacks_layer_boundary_and_tarjan_scc_cycle_detector(tmp_path) -> None:
    proj = tmp_path / "repo"
    files = {
        "mindie_sim/schema/a.py": "from mindie_sim.schema import b\n",
        "mindie_sim/schema/b.py": "from mindie_sim.schema import a\n",
        "mindie_sim/schema/bad_stack_ref.py": "from mindie_sim.stacks import registry\n",
        "mindie_sim/catalog/cat_a.py": "from mindie_sim.primitives import prim_b\n",
        "mindie_sim/primitives/prim_b.py": "from mindie_sim.catalog import cat_a\n",
        "mindie_sim/prediction/spine.py": "from mindie_sim.stacks import registry\n",
        "mindie_sim/stacks/registry.py": "from mindie_sim.schema import a\n",
        "mindie_sim/stacks/good_plugin/adapter.py": (
            "from mindie_sim.schema import a\n"
            "from mindie_sim.catalog import cat_a\n"
            "from mindie_sim.primitives import prim_b\n"
        ),
        "mindie_sim/stacks/bad_export/codec.py": "from mindie_sim.export import bundle\n",
        "mindie_sim/stacks/bad_compose/adapter.py": "from mindie_sim.compose import runtime\n",
        "mindie_sim/stacks/cyc_a/mod.py": "from mindie_sim.stacks.cyc_b import mod\n",
        "mindie_sim/stacks/cyc_b/mod.py": "from mindie_sim.stacks.cyc_a import mod\n",
        "mindie_sim/export/bundle.py": "X = 1\n",
        "mindie_sim/compose/runtime.py": "from mindie_sim.stacks import registry\n",
    }
    for rel, code in files.items():
        target = proj / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code, encoding="utf-8")

    ag010 = [v for v in arch_gov.scan_repository(proj) if v.rule == "AG010"]
    fps = {v.fingerprint for v in ag010}
    flagged_paths = {v.path.as_posix() for v in ag010}

    assert "mindie_sim/stacks/good_plugin/adapter.py" not in flagged_paths
    assert "mindie_sim/compose/runtime.py" not in flagged_paths
    assert "AG010:upward_import:mindie_sim/schema/bad_stack_ref.py:mindie_sim.stacks" in fps
    assert "AG010:core_imports_stacks:mindie_sim/prediction/spine.py:mindie_sim.stacks" in fps
    assert "AG010:upward_import:mindie_sim/stacks/bad_export/codec.py:mindie_sim.export" in fps
    assert "AG010:upward_import:mindie_sim/stacks/bad_compose/adapter.py:mindie_sim.compose" in fps
    assert "AG010:package_scc:mindie_sim.catalog,mindie_sim.primitives" in fps
    assert "AG010:package_scc:mindie_sim.stacks.cyc_a,mindie_sim.stacks.cyc_b" in fps
    assert "AG010:module_scc:mindie_sim.schema.a,mindie_sim.schema.b" in fps
