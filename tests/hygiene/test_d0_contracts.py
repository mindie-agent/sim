"""Frozen spine contract verification (C-1..C-6, CANN + non-CANN ABI, synthetic GQA gate).

[Role] Verify all shared L0/L2 spine contracts established before parallel
  implementation branches: unified CANN + non-CANN hardware-usable ABI schema,
  explicit synthetic GQA opt-in policy, math helpers, stack track bindings,
  structural LayerPlan fields, dual-tier UB receipts, exact 0 ns Engine stage
  receipts, canonical NativeCallBinding metadata codecs, and full-package
  absence of runtime uncalibrated states.
[Invariants] Pure in-memory contract checks with zero device calls, zero
  subprocess spawns, and deterministic SHA-256 verification.
[Timeouts & Locks] Fast Tier-0 execution (< 5 s); no locks or network I/O.
[Side-Effect Policy] Read-only against packaged repo files; zero disk mutation.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from mindie_sim.catalog.draft_spec import (
    NGRAM_CPU_PROVENANCE,
    NO_DRAFT_PROVENANCE,
    REAL_CHECKPOINT_PROVENANCE,
    DSparkPreparePlan,
    DraftSpec,
    SpeculativeExecutionContract,
)
from mindie_sim.catalog.types import AuxiliarySpec
from mindie_sim.compose import compose_case
from mindie_sim.compose.runtime import resolve_case_contracts
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.native_binding import (
    NATIVE_CALL_METADATA_KEY,
    NativeCallBinding,
)
from mindie_sim.prediction.operators.paged_mla import mqsmla as mqsmla_mod
from mindie_sim.prediction.program import implementation_registry
from mindie_sim.prediction.results import (
    CANONICAL_STAGE_IDS,
    EngineStepReceipt,
    StageReceipt,
    StepEvaluation,
    StepRequest,
    SubStageReceipt,
)
from mindie_sim.prediction.scheduler.progress import DraftPolicy
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.abi import (
    ABI_CATALOG_SCHEMA,
    CANN_PACKAGE_ABI_BACKENDS,
    HARDWARE_USABLE_ABI_BACKENDS,
    NON_CANN_ABI_BACKENDS,
    AbiCatalogSnapshot,
    AbiQueryKey,
    AbiShapeConstraints,
    AbiSupportVerdict,
    OperandPortSig,
    ProbedAbiEntry,
    validate_fusion_tier_verdict,
)
from mindie_sim.schema.checkpoint import CheckpointProvenance, CheckpointQuantMap
from mindie_sim.schema.common import ModelingError, inclusion_probability, normal_max_order_z
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.hardware import Device, Hardware
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.operators import KernelBinding
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.scenario import ScenarioContract
from mindie_sim.schema.stack import (
    ManifestBinding,
    ResolvedStackOptions,
    RuntimeQuantPlan,
    StackLoweringPlan,
    StackTrackSpec,
)
from mindie_sim.schema.workload import (
    SYNTHETIC_EAGLE3_GQA_PROVENANCE,
    RequestView,
    SpeculativeWorkloadPolicy,
)
from mindie_sim.stacks.vllm_ascend.profiles import resolve_manifest_bindings
from mindie_sim.templates.contracts import (
    DualTierLoweringReceipt,
    TierLoweringOutcome,
    UbAdmissionReceipt,
)

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_REV = "405617bae227fe5b097ff163efaa60bb7f22c698"
SAMPLE_SHA = "75e8784f57fd5125ee29dd3407dab8b0cf43bf6532104fe5a89540bf7df41ddf"
SAMPLE_PROBE_SHA = "1" * 64
SAMPLE_PT_DIGEST = "4" * 64


def _src(repo: str = "vllm-project/vllm-ascend", path: str = "vllm_ascend/ops/kernel.py") -> SourceRef:
    return SourceRef(repository=repo, revision=SAMPLE_REV, relative_path=path, source_sha256=SAMPLE_SHA)


def _port(role: str = "x", dtype: str = "bf16") -> OperandPortSig:
    return OperandPortSig(
        role=role, dtype=dtype, layout="ND", transposed=False, shape=(16, 4096),
        quant_mode="dense", group_shape=(1,), scale_dtype="none", scale_shape=(),
    )


def _query(*, enabled_passes: frozenset[str] = frozenset({"norm_quant_fusion"})) -> AbiQueryKey:
    return AbiQueryKey(
        soc_arch="A5", sku_id="ascend950dt_9582", abi_id="add_rms_norm_dynamic_mx_quant",
        graph_motif="add_rmsnorm_mxquant", model_scope="target", draft_quant_mode="none",
        pr_track="shipping", operands=(_port("x"),), outputs=(_port("y", "fp8_e4m3"),),
        rank_placement="local", world_size=1, comm_group_id="local", comm_domain_width=1,
        comm_dtype="none", reduction_kind="none", activation_kind="none", activation_params=(),
        has_bias=False, has_active_mask=False, dispatch_quant_mode=-1, shared_experts_per_rank=0,
        top_k=0, moe_expert_num=0, qk_rope_head_dim=64, mla_variant="none", sfa_dcp_active=False,
        pcp_size=1, kv_cache_dtype="bf16", has_kpool_indexer=False, dcp_size=1,
        is_act_and_mul=False, enabled_stack_passes=enabled_passes,
    )


def _draft(*, is_synthetic: bool = False, provenance: str = REAL_CHECKPOINT_PROVENANCE) -> DraftSpec:
    return DraftSpec(
        kind="eagle3" if is_synthetic else "dspark", layers=1, hidden_size=4096,
        intermediate_size=14336, num_heads=32, num_kv_heads=8, head_dim=128,
        q_width=4096, kv_width=2048, block_size=5, markov_rank=64, target_layer_ids=(0, 1, 2),
        execution_mode="eagle3_serial" if is_synthetic else "block_dspark",
        is_synthetic=is_synthetic, draft_provenance=provenance,
    )


def _scenario(
    *,
    optimization_mode: str = "simulation",
    speculative_policy: SpeculativeWorkloadPolicy | None = None,
    auxiliary: AuxiliarySpec | None = None,
) -> ScenarioContract:
    hw = Hardware(
        name="Ascend950DT_9582",
        dies=(Device(
            rank=0, memory_bytes=64 * 1024**3, compute_peaks=(("bf16", 1e15),),
            rates=(("hbm", 1.6e12), ("vector", 1e12)), buffers=(), compute_units=32,
        ),),
    )
    model = ModelSpec(
        name="synthetic-structural-test", layers=1, hidden_size=4096, intermediate_size=14336,
        num_heads=32, num_kv_heads=8, head_dim=128, vocab_size=32000, auxiliary=auxiliary,
    )
    ckpt = CheckpointQuantMap(provenance=CheckpointProvenance(
        checkpoint_id="synthetic-structural-test", source_format="hf_quantization_config",
        source_revision=SAMPLE_REV, source_sha256=SAMPLE_SHA, descriptor_path="config.json",
        origin="verified_checkpoint_descriptor",
    ))
    return ScenarioContract(
        deployment_mode="colocated", deployment_mode_origin="api_argument", workload_domain="llm",
        work_unit_kind="prefill_decode_step", model=model, checkpoint=ckpt, stack_id="vllm-ascend",
        stack_track="shipping", stack_options=ResolvedStackOptions(stack_id="vllm-ascend", track="shipping"),
        quant_plan=RuntimeQuantPlan(checkpoint_digest=ckpt.digest(), stack_id="vllm-ascend"),
        lowering_plan=StackLoweringPlan(stack_id="vllm-ascend", track="shipping"), hardware=hw,
        primitive_table_digest="a" * 64, optimization_mode=optimization_mode,  # type: ignore[arg-type]
        speculative_policy=speculative_policy,
    )


def _stage(stage_id: str, track: str, span: int, idle: int, key_ch: str, plan_ch: str) -> StageReceipt:
    return StageReceipt(
        stage_id=stage_id, stack_track=track, start_tick_ns=0, end_tick_ns=span, span_ns=span,  # type: ignore[arg-type]
        l_star_ns=600_000, l_star_valid=True, execution_key_sha256=key_ch * 64,
        plan_sha256=plan_ch * 64, idle_or_wait_ns=idle, exposed_ns_by_occ=(("op0", span - idle),),
        primitive_table_digest=SAMPLE_PT_DIGEST,
        sub_stage_receipts=(SubStageReceipt("attn", "aic", span - idle),) if stage_id == CANONICAL_STAGE_IDS[0] else (),
    )


def test_abi_schema_covers_cann_and_non_cann_hardware_usable_backends() -> None:
    assert {"cann_aclnn", "hccl_mc2"} <= CANN_PACKAGE_ABI_BACKENDS
    assert {"vllm_ascend_csrc", "triton", "torch_npu", "catlass", "third_party_wheel"} <= NON_CANN_ABI_BACKENDS
    assert HARDWARE_USABLE_ABI_BACKENDS == CANN_PACKAGE_ABI_BACKENDS | NON_CANN_ABI_BACKENDS

    cann_entry = ProbedAbiEntry(
        entry_id="a5_cann_add_rmsnorm_mxquant", abi_id="add_rms_norm_dynamic_mx_quant",
        available_on_device=True, backend_kind="cann_aclnn", symbol_or_kernel="aclnnAddRmsNormDynamicMxQuant",
        device_arch="A5", stack_or_cann_version="9.3.0", role="norm_quant",
        dtype_tuple=(("bf16", "bf16", "fp8_e4m3"),), shape_constraints=AbiShapeConstraints(hidden_divisibility=32),
        source_ref=_src("cann/ops-nn", "norm/add_rms_norm_dynamic_mx_quant/tiling.cpp"),
        cann_check_params_ref="ops-nn/norm/add_rms_norm_dynamic_mx_quant/check_params.cpp:42",
        device_kernel_expansion=("AddRmsNormDynamicMxQuant",), probe_signature_sha256=SAMPLE_PROBE_SHA,
    )
    assert cann_entry.is_cann_package and not cann_entry.is_non_cann_package

    non_cann_specs = [
        ("vllm_ascend_csrc", "_C_ascend.mla_preprocess", "vllm-project/vllm-ascend", "csrc/mla_preprocess.cpp"),
        ("triton", "vllm.recoverssm_commit_triton", "vllm-project/vllm-ascend", "vllm_ascend/ops/triton/gdn/commit.py"),
        ("torch_npu", "torch_npu.npu_rms_norm", "Ascend/op-plugin", "op_plugin/ops/opapi/RmsNormKernelNpuOpApi.cpp"),
        ("catlass", "catlass_grouped_gemm_bf16", "Ascend/catlass", "examples/grouped_gemm/kernel.hpp"),
        ("third_party_wheel", "ops.fused_swiglu_step", "third-party/ascend-ops", "ops/swiglu_step.py"),
    ]
    non_cann = [
        ProbedAbiEntry(
            entry_id=f"a5_{b}", abi_id=f"abi_{b}", available_on_device=True, backend_kind=b,  # type: ignore[arg-type]
            symbol_or_kernel=sym, device_arch="A5", stack_or_cann_version="vllm-ascend@0c99d3fc",
            role="custom_kernel", dtype_tuple=(("bf16", "bf16"),), shape_constraints=AbiShapeConstraints(),
            source_ref=_src(repo, rel), vllm_ascend_ref=f"{rel}:10",
            device_kernel_expansion=(sym.split(".")[-1],), probe_signature_sha256=SAMPLE_PROBE_SHA,
        )
        for b, sym, repo, rel in non_cann_specs
    ]
    assert all(e.is_non_cann_package and not e.is_cann_package and e.cann_check_params_ref == "" for e in non_cann)

    with pytest.raises(ModelingError, match="abi_entry_cann_missing_check_params_ref"):
        replace(cann_entry, cann_check_params_ref="")
    with pytest.raises(ModelingError, match="abi_entry_source_ref_incomplete"):
        replace(non_cann[1], source_ref=SourceRef("", "", "", ""))

    snap = AbiCatalogSnapshot(
        device_arch="A5", sku_id="Ascend950DT_9582", stack_or_cann_version="9.3.0+0c99d3fc",
        entries=(cann_entry, *non_cann),
    )
    assert snap.schema_version == ABI_CATALOG_SCHEMA
    snap.verify_seal(snap.payload_sha256)
    with pytest.raises(ModelingError, match="abi_catalog_seal_mismatch"):
        snap.verify_seal("f" * 64)

    q1 = _query(enabled_passes=frozenset({"norm_quant_fusion"}))
    assert q1.canonical_digest() != _query(enabled_passes=frozenset({"norm_quant_fusion", "mc2_fusion"})).canonical_digest()

    verdict = AbiSupportVerdict(
        abi_id="abi_triton", status="admitted_shipping", available_on_device=True, backend_kind="triton",
        device_arch="A5", stack_or_cann_version="vllm-ascend@0c99d3fc", act_dtype="bf16",
        weight_dtype="bf16", quant_mode="dense", probe_manifest_sha256=snap.payload_sha256,
        engine_speedup_verified=True, query_digest=q1.canonical_digest(), matched_entry_id="a5_triton",
        symbol_or_kernel="vllm.recoverssm_commit_triton", source_ref=_src(),
        device_kernel_expansion=("recoverssm_commit_triton",),
    )
    validate_fusion_tier_verdict("K1_shipping", verdict)
    with pytest.raises(ModelingError, match="fuse_region_shipping_tier_downgrade_forbidden"):
        validate_fusion_tier_verdict("K3_dsl_pipeline", verdict)
    with pytest.raises(ModelingError, match="fuse_region_unverified_shipping_claim"):
        validate_fusion_tier_verdict("K1_shipping", replace(verdict, engine_speedup_verified=False))
    with pytest.raises(ModelingError, match="abi_verdict_query_digest_invalid"):
        replace(verdict, query_digest="?")
    with pytest.raises(ModelingError, match="abi_verdict_admitted_missing_dtype_evidence"):
        replace(verdict, act_dtype="")
    with pytest.raises(ModelingError, match="abi_verdict_admitted_missing_source_ref"):
        replace(verdict, source_ref=None)
    with pytest.raises(ModelingError, match="abi_verdict_proposed_missing_pr_ref"):
        replace(verdict, status="admitted_proposed_pr", pr_track="proposed", proposed_pr_ref="   ")

    schema_json = json.loads((REPO_ROOT / "data" / "abi" / "abi_schema_manifest.json").read_text(encoding="utf-8"))
    assert schema_json["schema"] == ABI_CATALOG_SCHEMA
    assert set(schema_json["hardware_usable_backends"]) == set(HARDWARE_USABLE_ABI_BACKENDS)


def test_synthetic_gqa_policy_locked_for_simulation_and_opt_in_for_modeling() -> None:
    assert SpeculativeWorkloadPolicy().allow_synthetic_gqa is False
    with pytest.raises(ModelingError, match="synthetic_gqa_requires_eagle3_method"):
        SpeculativeWorkloadPolicy(spec_method="mtp", num_speculative_tokens=3, allow_synthetic_gqa=True)
    with pytest.raises(ModelingError, match="synthetic_gqa_provenance_forbids_disabled_opt_in"):
        SpeculativeWorkloadPolicy(
            spec_method="eagle3", num_speculative_tokens=3, allow_synthetic_gqa=False,
            synthetic_draft_provenance=SYNTHETIC_EAGLE3_GQA_PROVENANCE,
        )

    opt_in = SpeculativeWorkloadPolicy(spec_method="eagle3", num_speculative_tokens=3, allow_synthetic_gqa=True)
    assert opt_in.synthetic_draft_provenance == SYNTHETIC_EAGLE3_GQA_PROVENANCE
    with pytest.raises(ModelingError, match="simulation_mode_forbids_synthetic_gqa_draft"):
        _scenario(optimization_mode="simulation", speculative_policy=opt_in)
    assert _scenario(optimization_mode="modeling", speculative_policy=opt_in).speculative_policy == opt_in

    real_draft = _draft(is_synthetic=False, provenance=REAL_CHECKPOINT_PROVENANCE)
    synth_draft = _draft(is_synthetic=True, provenance=SYNTHETIC_EAGLE3_GQA_PROVENANCE)
    with pytest.raises(ModelingError, match="catalog_draft_synthetic_provenance_invalid"):
        _draft(is_synthetic=True, provenance="wrong_tag")
    with pytest.raises(ModelingError, match="simulation_mode_forbids_synthetic_gqa_draft"):
        _scenario(optimization_mode="simulation", auxiliary=AuxiliarySpec(mtp=synth_draft))
    with pytest.raises(ModelingError, match="scenario_synthetic_draft_requires_policy_opt_in"):
        _scenario(optimization_mode="modeling", auxiliary=AuxiliarySpec(mtp=synth_draft))
    with pytest.raises(ModelingError, match="catalog_draft_synthetic_requires_eagle3"):
        DraftSpec(
            kind="dspark", layers=1, hidden_size=4096, intermediate_size=14336, num_heads=32,
            num_kv_heads=8, head_dim=128, q_width=4096, kv_width=2048, attention_kind="mla",
            execution_mode="eagle3_serial", is_synthetic=True, draft_provenance=SYNTHETIC_EAGLE3_GQA_PROVENANCE,
        )

    sim_contract = SpeculativeExecutionContract(
        archetype="dspark", spec_method="dspark", num_speculative_tokens=5, requires_draft_spec=True,
        draft_tp=4, neural_draft_forwards=1, requires_draft_prepare=True, optimization_mode="simulation",
        allow_synthetic_gqa=False, draft_provenance=REAL_CHECKPOINT_PROVENANCE,
    )
    with pytest.raises(ModelingError, match="speculative_contract_missing_real_draft_spec"):
        sim_contract.require_draft_for_execution(None)
    with pytest.raises(ModelingError, match="speculative_contract_rejects_synthetic_draft_without_modeling_opt_in"):
        sim_contract.require_draft_for_execution(synth_draft)
    assert sim_contract.require_draft_for_execution(real_draft) == real_draft

    dcp2 = replace(sim_contract, draft_dcp=2)
    with pytest.raises(ModelingError, match="speculative_contract_draft_dcp_exceeds_target"):
        dcp2.validate_dcp_spec_gate(real_draft, ParallelSpec(tp=1, dcp=1))

    # DSpark prepare FLOPs match GQA fused_kv_bank (kimi-k3), per_layer_qkv (minimax-m3),
    # MLA mla_prolog (kimi-k3.1), and shared_compressed_kv (V4 Flash / V4 Pro / V4.1).
    k3_prep = sim_contract.build_prepare_plan(16, draft_spec=load_model("kimi-k3").auxiliary.dspark)  # type: ignore[union-attr]
    assert isinstance(k3_prep, DSparkPreparePlan) and k3_prep.exact_gemm_flops == 10_569_646_080
    mm3_prep = sim_contract.build_prepare_plan(16, draft_spec=load_model("minimax-m3").auxiliary.mtp)  # type: ignore[union-attr]
    assert mm3_prep.exact_gemm_flops == 13_891_534_848
    k31_prep = sim_contract.build_prepare_plan(16, draft_spec=load_model("kimi-k3.1").auxiliary.dspark)  # type: ignore[union-attr]
    assert k31_prep.exact_gemm_flops == 10_643_046_400
    v41_aux = load_model("deepseek-v4.1-flash").auxiliary
    v41_prep = sim_contract.build_prepare_plan(131_072, draft_spec=(v41_aux.dspark or v41_aux.mtp))  # type: ignore[union-attr]
    assert v41_prep.exact_gemm_flops == 22_677_427_322_880

    # Schedule: empty/zero-proposal -> 0 steps, pre-expanded q_len not double-counted,
    # heterogeneous B=2 (p=3, p=1) -> step 0 verify (6), steps 1..K-1 full proposal batch B=2 -> (6, 2, 2).
    mod_contract = SpeculativeExecutionContract(
        archetype="eagle3", spec_method="eagle3", num_speculative_tokens=3, requires_draft_spec=True,
        optimization_mode="modeling", allow_synthetic_gqa=True, draft_provenance=SYNTHETIC_EAGLE3_GQA_PROVENANCE,
    )
    assert mod_contract.require_draft_for_execution(synth_draft) == synth_draft
    assert mod_contract.build_step_schedule(()).num_steps == 0
    assert mod_contract.build_step_schedule((RequestView("r0", 1, 1024, True, "decode", draft_verify=False),)).num_steps == 0
    assert mod_contract.build_step_schedule(
        (RequestView("r0", 4, 1024, True, "decode", draft_verify=True, draft_proposal_tokens=3),)
    ).body_q_len_per_step == (4, 1, 1)
    assert mod_contract.build_step_schedule((
        RequestView("r0", 4, 1024, True, "decode", draft_verify=True, draft_proposal_tokens=3),
        RequestView("r1", 2, 1024, True, "decode", draft_verify=True, draft_proposal_tokens=1),
    )).body_q_len_per_step == (6, 2, 2)
    # Continuing decode after a miss (p_i=0) with K=3 still runs step 0 with 1 query row: (1, 1, 1).
    assert mod_contract.build_step_schedule((
        RequestView("r0", 1, 1024, True, "decode", draft_verify=True, draft_proposal_tokens=0),
    )).body_q_len_per_step == (1, 1, 1)

    # Strict draft_compatible_with_spec_method: non-neural, unknown, and mismatched archetypes fail closed.
    from mindie_sim.catalog.draft_spec import draft_compatible_with_spec_method
    assert draft_compatible_with_spec_method(real_draft, "ngram") is False
    assert draft_compatible_with_spec_method(real_draft, "ngram_gpu") is False
    assert draft_compatible_with_spec_method(real_draft, "unknown_method") is False
    assert draft_compatible_with_spec_method(real_draft, "dspark") is True
    assert draft_compatible_with_spec_method(real_draft, "eagle3") is False
    assert draft_compatible_with_spec_method(synth_draft, "eagle3") is True
    with pytest.raises(ModelingError, match="dspark_prepare_requires_context_projection_mode"):
        sim_contract.build_prepare_plan(16, draft_spec=replace(real_draft, context_projection_mode="none"))

    ngram = SpeculativeExecutionContract(
        archetype="ngram", spec_method="ngram", num_speculative_tokens=4, requires_draft_spec=False,
        neural_draft_forwards=0, requires_draft_prepare=False, draft_provenance=NGRAM_CPU_PROVENANCE,
    )
    assert ngram.require_draft_for_execution(None) is None and ngram.build_step_schedule(()).num_steps == 0
    with pytest.raises(ModelingError, match="speculative_contract_forbids_neural_draft"):
        ngram.require_draft_for_execution(real_draft)
    assert SpeculativeExecutionContract(
        archetype="none", spec_method="none", num_speculative_tokens=0, requires_draft_spec=False,
        draft_provenance=NO_DRAFT_PROVENANCE,
    ).require_draft_for_execution(None) is None

    # Replay opt-out and old fabricated kimi-k2 NextN draft must fail fast; scenario agrees with stack.
    synth_case = compose_case(
        "qwen3-32b", "a3", tp=4, dp=1, layers=2, prompt_tokens=256, output_tokens=32,
        mtp_proposal_tokens=3, mtp_acceptance_length=2.2, feature_track="target",
        allow_synthetic_gqa_draft=True,
    )
    with pytest.raises(ModelingError, match="synthetic_draft_requires_explicit_allow_synthetic_gqa"):
        compose_case(
            synth_case["model"], "a3", tp=4, dp=1, layers=2, prompt_tokens=256, output_tokens=32,
            mtp_proposal_tokens=3, mtp_acceptance_length=2.2, feature_track="target",
            allow_synthetic_gqa_draft=False,
        )
    with pytest.raises(ModelingError, match="draft_structure_required_for_mtp_proposal_tokens"):
        compose_case(
            "kimi-k2", "a3", tp=8, ep=8, layers=2, prompt_tokens=1024, output_tokens=64,
            mtp_proposal_tokens=3, mtp_acceptance_length=2.2,
        )
    fabricated_k2 = load_model("kimi-k2").to_dict()
    fabricated_k2.get("catalog_facts", {}).pop("num_nextn_predict_layers", None)
    fabricated_k2["auxiliary"] = {
        "mtp": replace(
            real_draft, kind="mtp", attention_kind="mla", execution_mode="stack",
            q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
            source=(("kind", "standard_deepseek_nextn_mtp"),),
        ).to_dict(),
    }
    with pytest.raises(ModelingError, match="fabricated_mla_nextn_draft_contradicts_checkpoint_zero_nextn"):
        compose_case(
            fabricated_k2, "a3", tp=8, ep=8, prompt_tokens=128, output_tokens=16,
            mtp_proposal_tokens=3, mtp_acceptance_length=2.2,
        )

    # Full-case shipping replay rejects swapped synthetic draft; scenario/stack/scheduler agree on K=15.
    mm3_raw = compose_case("minimax-m3", "a5", tp=4, ep=4, prompt_tokens=128, output_tokens=16, mtp_proposal_tokens=5, mtp_acceptance_length=3.0)
    mm3_sc, _, mm3_lp = resolve_case_contracts(mm3_raw)
    assert mm3_sc is not None and mm3_sc.speculative_policy is not None and mm3_sc.speculative_policy.spec_method == "dspark"
    assert mm3_lp.scheduling.speculative_proposal_count == mm3_sc.speculative_policy.num_speculative_tokens == 5
    swapped_mm3 = json.loads(json.dumps(mm3_raw))
    swapped_mm3["model"]["auxiliary"]["mtp"] = synth_draft.to_dict()
    with pytest.raises(ModelingError, match="simulation_mode_forbids_synthetic_gqa_draft"):
        parse_case(swapped_mm3)

    r1_k1 = compose_case("deepseek-r1", "a3", tp=8, ep=8, layers=4, prompt_tokens=128, output_tokens=16, mtp_proposal_tokens=1, mtp_acceptance_length=1.5)
    r1_k5 = compose_case("deepseek-r1", "a3", tp=8, ep=8, layers=4, prompt_tokens=128, output_tokens=16, mtp_proposal_tokens=5, mtp_acceptance_length=3.0)
    r1_k15 = compose_case("deepseek-r1", "a3", tp=8, ep=8, layers=4, prompt_tokens=128, output_tokens=16, mtp_proposal_tokens=15, mtp_acceptance_length=4.0)
    with pytest.raises(ModelingError, match="ascend_fia_speculative_tokens_exceeds_limit:16>15"):
        compose_case("deepseek-r1", "a3", tp=8, ep=8, layers=4, prompt_tokens=128, output_tokens=16, mtp_proposal_tokens=16, mtp_acceptance_length=4.0)
    _, _, lp_k1 = resolve_case_contracts(r1_k1)
    _, _, lp_k5 = resolve_case_contracts(r1_k5)
    sc_k15, _, lp_k15 = resolve_case_contracts(r1_k15)
    parsed_k15 = parse_case(r1_k15)
    assert lp_k1.plan_digest != lp_k5.plan_digest
    assert lp_k1.scheduling.speculative_proposal_count == 1 and lp_k5.scheduling.speculative_proposal_count == 5
    assert sc_k15 is not None and sc_k15.speculative_policy is not None
    assert sc_k15.speculative_policy.num_speculative_tokens == lp_k15.scheduling.speculative_proposal_count == 15
    assert DraftPolicy.for_request(parsed_k15, parsed_k15.workload.requests[0]).proposal == 15


def test_common_math_stack_track_structural_layer_and_kernel_backends() -> None:
    assert normal_max_order_z(1) == 0.0
    assert math.isclose(normal_max_order_z(2), math.sqrt(2.0 / math.pi), rel_tol=1e-9)
    assert normal_max_order_z(16) > normal_max_order_z(8) > normal_max_order_z(2)
    assert inclusion_probability(0.0, 1.0) == 0.0 and inclusion_probability(4.0, 1.0) == 1.0
    assert math.isclose(inclusion_probability(2.0, 0.5), 0.75, rel_tol=1e-12)

    merged_binding = ManifestBinding(
        mechanism_id="mla_prolog_fusion", state="merged", ref="vllm-project/vllm-ascend@405617b",
        lowering_pass_id="MlaPrologPass", provenance=_src(), archs=("a5",),
        capability_predicates=("qk_rope_head_dim > 0",), phase_scope=("decode",),
    )
    assert merged_binding.matches_arch_and_phase("A5", "decode")
    assert not merged_binding.matches_arch_and_phase("A2", "decode")
    assert not merged_binding.matches("any-model", "a5", "decode")
    assert merged_binding.matches("any-model", "a5", "decode", capabilities={"qk_rope_head_dim": 64})
    assert not merged_binding.matches("any-model", "a5", "decode", capabilities={"qk_rope_head_dim": 0})
    lazy_or = replace(merged_binding, capability_predicates=("has_gqa or missing_key > 0",))
    assert lazy_or.matches("any-model", "a5", "decode", capabilities={"has_gqa": True})

    pr_binding = ManifestBinding(
        mechanism_id="unmerged_demo", state="unmerged_pr", ref="vllm-project/vllm-ascend#99999",
        provenance=_src(),
    )
    with pytest.raises(ModelingError, match="stack_track_spec_shipping_requires_merged_bindings_only"):
        StackTrackSpec(stack_id="vllm-ascend", track="shipping", optimization_mode="simulation", active_bindings=(pr_binding,))
    with pytest.raises(ModelingError, match="stack_track_spec_target_synthesis_requires_target_track"):
        StackTrackSpec(stack_id="vllm-ascend", track="proposed", optimization_mode="modeling", allows_target_synthesis=True)

    ship_hits = resolve_manifest_bindings("shipping", "qwen3-32b", "a3")
    assert ship_hits and all(isinstance(b, ManifestBinding) for b in ship_hits)
    assert all(bool(b.lowering_pass_id) for b in ship_hits)
    spec_a = StackTrackSpec(stack_id="vllm-ascend", track="shipping", optimization_mode="simulation", active_bindings=ship_hits[:1])
    spec_b = replace(spec_a, active_bindings=ship_hits[:2])
    assert spec_a.spec_digest != spec_b.spec_digest
    plan_a = StackLoweringPlan(stack_id="vllm-ascend", track="shipping", track_spec=spec_a)
    assert plan_a.plan_digest != replace(plan_a, track_spec=spec_b).plan_digest
    with pytest.raises(ModelingError, match="lowering_plan_readiness_exceeds_track_spec"):
        StackLoweringPlan(
            stack_id="vllm-ascend", track="shipping",
            allowed_rewrite_classes=frozenset({"shipping", "proposed_pr"}), track_spec=spec_a,
        )

    layer = LayerPlan(index=0, attention="gdn", has_sandwich_norm=True, norm_unit_offset=True, v_norm=True, gdn_layout="interleaved")
    assert layer.has_sandwich_norm and layer.norm_unit_offset and layer.v_norm and layer.gdn_layout == "interleaved"
    with pytest.raises(ModelingError, match="layer_plan_gdn_layout_unknown"):
        LayerPlan(index=0, attention="gdn", gdn_layout="unknown")

    for backend in ("CANN", "Triton", "TileLang", "AscendC", "Catlass"):
        assert KernelBinding(kernel_id=f"k_{backend}", backend=backend, arch_family="dav_3510", readiness="shipping").backend == backend  # type: ignore[arg-type]


def test_ub_admission_dual_tier_engine_receipt_and_native_binding_codec() -> None:
    ub_ok = UbAdmissionReceipt(
        region_id="reg_0", soc_arch="A5", tier_kind="tilelang_simt", algorithm="online_softmax",
        peak_ub_bytes=180_000, ub_wall_bytes=221_184, admitted=True,
        working_ub_at_peak_bytes=120_000, ring_ub_at_peak_bytes=60_000,
    )
    assert ub_ok.headroom_bytes == 41_184
    with pytest.raises(ModelingError, match="ub_admission_peak_must_equal_working_plus_ring"):
        replace(ub_ok, peak_ub_bytes=200_000)

    out_simt = TierLoweringOutcome(
        tier_kind="tilelang_simt", backend_target="tilelang", admitted=True, step_ns=950_000,
        lowering_id="simt_v1", lowering_sha256="2" * 64, ub_receipt=ub_ok,
    )
    out_simd = TierLoweringOutcome(
        tier_kind="tilelang_simd", backend_target="tilelang", admitted=True, step_ns=890_000,
        lowering_id="simd_v1", lowering_sha256="3" * 64,
    )
    dual = DualTierLoweringReceipt(region_id="reg_0", predecessor_step_ns=1_000_000, tier_outcomes=(out_simt, out_simd))
    assert dual.best_feasible_tier == "tilelang_simd" and dual.best_step_ns == 890_000

    with pytest.raises(ModelingError, match="tier_lowering_outcome_ub_tier_mismatch"):
        TierLoweringOutcome(tier_kind="tilelang_simd", backend_target="tilelang", admitted=True, step_ns=890_000, ub_receipt=ub_ok)
    with pytest.raises(ModelingError, match="dual_tier_receipt_ub_region_mismatch"):
        DualTierLoweringReceipt(region_id="reg_other", predecessor_step_ns=1_000_000, tier_outcomes=(out_simt,))
    rejected_out = TierLoweringOutcome(
        tier_kind="tilelang_simt", backend_target="tilelang", admitted=False, step_ns=None, rejection_reason="ub_spill",
    )
    with pytest.raises(ModelingError, match="dual_tier_receipt_best_forbidden_when_none_admitted"):
        DualTierLoweringReceipt(
            region_id="reg_0", predecessor_step_ns=1_000_000, tier_outcomes=(rejected_out,),
            best_lowering_id="fake_winner", best_lowering_sha256="a" * 64,
        )

    s0 = _stage(CANONICAL_STAGE_IDS[0], "shipping", 1_000_000, 100_000, "5", "6")
    s1 = _stage(CANONICAL_STAGE_IDS[1], "shipping", 920_000, 20_000, "7", "8")
    s2 = _stage(CANONICAL_STAGE_IDS[2], "proposed", 880_000, 80_000, "9", "a")
    s3 = _stage(CANONICAL_STAGE_IDS[3], "target", 800_000, 0, "b", "c")
    receipt = EngineStepReceipt(
        scenario_digest="d" * 64, primitive_table_digest=SAMPLE_PT_DIGEST,
        comparison_policy="monotonic_optimization", stages=(s0, s1, s2, s3),
    )
    assert receipt.telescoping_deltas_ns == (80_000, 40_000, 80_000)
    assert receipt.final_bound_ns == 600_000
    assert receipt.assert_stage_monotonicity() and receipt.assert_zero_ns_empty_identity()

    with pytest.raises(ModelingError, match="stage_receipt_observer_conservation_failed"):
        replace(s0, idle_or_wait_ns=50_000)
    with pytest.raises(ModelingError, match="engine_receipt_simulation_forbids_committed_stage3_target"):
        EngineStepReceipt(scenario_digest="d" * 64, primitive_table_digest=SAMPLE_PT_DIGEST, comparison_policy="simulation_only", stages=(s0, s3))
    with pytest.raises(ModelingError, match="engine_receipt_empty_identity_nonzero_delta"):
        EngineStepReceipt(
            scenario_digest="d" * 64, primitive_table_digest=SAMPLE_PT_DIGEST,
            comparison_policy="monotonic_optimization",
            stages=(s0, replace(s1, execution_key_sha256=s0.execution_key_sha256, plan_sha256=s0.plan_sha256)),
        )
    with pytest.raises(ModelingError, match="engine_receipt_endpoints_mismatch"):
        EngineStepReceipt(
            scenario_digest="d" * 64, primitive_table_digest=SAMPLE_PT_DIGEST,
            comparison_policy="monotonic_optimization", stages=(s0, s1),
            endpoints=(("invented_stage", 123),),
        )
    with pytest.raises(ModelingError, match="stage_receipt_primitive_table_digest_required"):
        replace(s0, primitive_table_digest="")
    empty_eval = StepEvaluation(
        graph_id="g0", request=StepRequest(phase="decode", tokens=1, context=128, sequences=1),
        tier="calibrated", span_ms=1.0, operators=(), pipelines=(), primitive_table_digest="",
    )
    with pytest.raises(ModelingError, match="stage_receipt_primitive_table_digest_mismatch"):
        replace(s0, step_eval=empty_eval)

    binding = NativeCallBinding(
        call_id="c" * 64, op_id="l0.recoverssm", family="recurrent", recipe="ascend_recover_ssm",
        source_boundary="kernel",
        api={"namespace": "triton", "entry": "vllm.recoverssm_commit_triton", "aclnn_symbol": None},
        source={"repository": "vllm-project/vllm-ascend", "path": "vllm_ascend/ops/triton/commit.py", "sha256": SAMPLE_SHA},
        typed={"dtype": "bf16", "layout": "ND", "shapes": {"q": [16, 128]}, "attrs": {}, "signature_hash": "e" * 64},
        kernels=({"name": "recoverssm_commit_triton", "role": "aiv", "evidence": "source_declared"},),
        member_op_ids=("l0.recoverssm",), execution_core="aiv",
    )
    decoded = NativeCallBinding.from_metadata(binding.to_metadata())
    assert decoded == binding and decoded.backend_kind == "triton"
    assert decoded.symbol == "vllm.recoverssm_commit_triton" and decoded.device_kernels == ("recoverssm_commit_triton",)
    assert replace(binding, api={"namespace": "aclnn", "aclnn_symbol": "aclnnAddRmsNorm"}).backend_kind == "cann_aclnn"
    assert replace(binding, api={"namespace": "cann_kernel", "entry": "tilelang_dsl.mqs", "aclnn_symbol": None}).backend_kind == "cann_kernel"

    with pytest.raises(ModelingError, match="native_call_binding_metadata_missing"):
        NativeCallBinding.from_metadata({"native_call": {"api": {"namespace": "aclnn"}}})
    with pytest.raises(ModelingError, match="native_call_metadata_missing_callable_or_kernel"):
        NativeCallBinding.from_metadata({NATIVE_CALL_METADATA_KEY: {
            "call_id": "1" * 64, "op_id": "op0", "family": "attention", "recipe": "r",
            "source_boundary": "kernel", "api": {}, "typed": {}, "kernels": [],
        }})
    with pytest.raises(ModelingError, match="native_call_metadata_source_sha256_invalid"):
        NativeCallBinding.from_metadata({NATIVE_CALL_METADATA_KEY: {
            **binding.to_dict(), "source": {"repository": "r", "sha256": "x"},
        }})
    with pytest.raises(ModelingError, match="native_call_metadata_evidence_unverified_established"):
        NativeCallBinding.from_metadata({NATIVE_CALL_METADATA_KEY: {
            **binding.to_dict(), "evidence": {"entry": "established"},
        }})

    assert mqsmla_mod.RECIPE == "aclnn_mixed_quant_sparse_flash_mla"
    assert not hasattr(mqsmla_mod, "LEGACY_RECIPE") and not hasattr(mqsmla_mod, "LEGACY_SOURCE")
    assert "aclnn_mixed_quant_sparse_flash_mla" in implementation_registry.IMPLEMENTATIONS
    assert "dsl_mixed_quant_sparse_flash_mla" not in implementation_registry.IMPLEMENTATIONS


def test_zero_uncalibrated_runtime_states_across_package_ast() -> None:
    for py_file in sorted((REPO_ROOT / "mindie_sim").rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=py_file.as_posix())
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                lowered = node.name.lower()
                assert "uncalibrated" not in lowered and "calibrationtier" not in lowered, (py_file, node.name)
                for stmt in node.body:
                    if isinstance(stmt, ast.Assign):
                        for t in stmt.targets:
                            if isinstance(t, ast.Name):
                                assert "uncalibrated" not in t.id.lower(), (py_file, node.name, t.id)
                        if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                            assert stmt.value.value.strip().lower() != "uncalibrated", (py_file, node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
                for target in targets:
                    if isinstance(target, ast.Name):
                        lowered = target.id.lower()
                        assert "calibrationtier" not in lowered and "calibration_tier" not in lowered, (py_file, target.id)
                val = node.value
                if isinstance(val, ast.Subscript) and isinstance(val.value, ast.Name) and val.value.id == "Literal":
                    for sub in ast.walk(val.slice):
                        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                            assert "uncalibrated" not in sub.value.lower(), (py_file, sub.value)
