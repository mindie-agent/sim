"""Tests for Stage 4C vLLM-Ascend Before operator graph & SP padding exactness."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.feature_mechanisms import (
    MechanismContext,
    _apply_basic_norm_quant,
    _apply_sp,
)
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction import body_defaults
from mindie_sim.prediction.execution.layer_runs import discover_layer_cohorts
from mindie_sim.prediction.feedforward import _absorb_redundant_mc2_unpack, moe
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.operators.attention.attention_decode import StagedGQADecode
from mindie_sim.prediction.operators.mc2.megamoe import try_lower_fused_frame
from mindie_sim.prediction.operators.mc2.moe_comm_selection import (
    ALLTOALL,
    CAPACITY_AND_WORLD_SIZE,
    DYNAMIC,
    MC2,
    MoeCommConfig,
)
from mindie_sim.prediction.operators.mla.generic_latent_attention_views import (
    index_operands_quantized,
)
from mindie_sim.prediction.operators.mla_prolog.family_mla_prolog import (
    MLA_PROLOG_QUERY_LORA_RANKS,
    _indexer_q_category,
)
from mindie_sim.prediction.operators.moe_routing.moe_gating_fusion import (
    HASH_NATIVE_KIND,
    SOFTMAX_NATIVE_KIND,
    TOPK_NATIVE_KIND,
    _native_gating_dispatch,
)
from mindie_sim.prediction.operators.vector.gated_activation import gated_activation
from mindie_sim.prediction.sharding import sequence_parallel
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.model import QuantSpec
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec

pytestmark = pytest.mark.tier0

MTP_TARGET_MEGAMOE_FLOPS = 1_207_959_552.0
MTP_DRAFT_MEGAMOE_FLOPS = 603_979_776.0
MTP_TARGET_ASSIGNMENTS = 24
MTP_DRAFT_ASSIGNMENTS = 12
MTP_BATCH_SEQUENCES = 4
MTP_VERIFY_TOKENS = 2
MTP_CONTEXT_LEN = 512
MINIMAX_TOPK_BLOCKS = 16
MINIMAX_TOPK_TOKENS = 2048
MINIMAX_TOPK_OUTPUT_BYTES = 256.0
MC2_TEST_EP = 8
MC2_TEST_HIDDEN = 1024
GEMMA4_BF16_LAYERS = {1, 9, 10, 11, 27, 28}
GEMMA4_BF16_COHORT_TOTAL = 6
GEMMA4_COHORT_COUNT = 4
STEP5_BF16_EXPERT_LAYERS = {88, 89, 90}
STEP5_BF16_COHORT_TOTAL = 3
STEP5_COHORT_COUNT = 5
STEP5_MOE_FP8_LAYER = 4
STEP5_MOE_BF16_LAYER = 88
SMALL_CONTEXT = 128
FOUR_LAYERS = 4


def _ops(model: str, *, hardware: str = "a5", phase: str = "decode",
         tokens: int = 1, context: int = 128, sequences: int = 4,
         scope: str = "body", **kwargs):
    case = parse_case(compose_case(model, hardware=hardware, **kwargs))
    return case, target_operators(case, phase=phase, tokens=tokens,
                                  context=context, sequences=sequences, scope=scope)


def test_sp_right_padding_excludes_dummy_tokens_from_moe_and_shared_work() -> None:
    case, ops = _ops(
        "deepseek-v3", hardware="a5", phase="decode", tokens=1, sequences=4, context=128,
        layers=4, tp=8, ep=8, num_devices=8, sp=True,
    )
    moe_layer = next(p for p in case.model.layer_plans if p.ffn == "moe")
    up = next(op for op in ops if op.layer == moe_layer.index and op.op_id.endswith(".expert_up"))
    shared_up = next(
        op for op in ops if op.layer == moe_layer.index and op.op_id.endswith(".shared.gate_up")
    )
    # 4 valid tokens (right-padded to 8 for tp=8 SP): routed rows & shared expert use 4, not 8.
    expected_routed_rows = 4 * moe_layer.top_k
    expected_up_flops = (
        2 * expected_routed_rows * (2 * moe_layer.moe_intermediate_size)
        * case.model.hidden_size / case.runtime.parallel.ep
    )
    assert up.flops == pytest.approx(expected_up_flops, rel=1e-2)
    assert shared_up.local_shape[0] == 4
    views = tuple(SimpleNamespace(q_len=1, padding=False) for _ in range(4))
    sp_plan = sequence_parallel.select(
        case.runtime, case.model, views, tokens=4, valid_global_tokens=4,
    )
    assert sp_plan is not None and sp_plan.valid_cohort_tokens == 4
    assert sp_plan.tokens_global == 8


def test_zero_expert_num_scales_router_width_and_real_routed_rows() -> None:
    model = load_model("deepseek-v3")
    base_plan = next(p for p in model.layer_plans if p.ffn == "moe")
    plan = replace(base_plan, num_experts=256, top_k=12)
    zero_model = replace(
        model,
        layer_plans=(plan,),
        catalog_facts=tuple(model.catalog_facts) + (("zero_expert_num", 128),),
    )
    ops, _ = moe(
        prefix="s0.l0", plan=plan, stage=0, deps="in", model=zero_model,
        parallel=ParallelSpec(tp=1, ep=1), quant=zero_model.quant, mode="decode",
        tokens=4, requests=("r0", "r1", "r2", "r3"), global_tokens=4,
    )
    router = next(op for op in ops if op.op_id.endswith(".router"))
    up = next(op for op in ops if op.op_id.endswith(".expert_up"))
    assert router.local_shape[:2] == (4, 384)
    # Expected real routed rows = 4 * 12 * (256 / 384) = 32.
    expected_flops = 2 * 32 * (2 * plan.moe_intermediate_size) * zero_model.hidden_size
    assert up.flops == pytest.approx(expected_flops)


def test_dense_flashcomm_sp_enabled_without_expert_parallel() -> None:
    case = parse_case(compose_case("qwen3-32b", hardware="a5", layers=2, tp=4, ep=1, num_devices=4))
    rt: dict = {"metadata": {}, "kernels": {}}
    ctx = MechanismContext(
        model=case.model, sku=case.machine.sku, arch="a5",
        parallel={"tp": 4, "ep": 1, "dp": 1}, caller_explicit=frozenset(),
        options={"enable_flashcomm1": True},
    )
    _apply_sp(rt, ctx)
    sp_decl = rt["kernels"].get("sequence_parallel")
    assert isinstance(sp_decl, dict)
    assert sp_decl.get("implementation") == "vllm_ascend_sequence_parallel"
    assert sp_decl.get("enable_expert_parallel") is False
    dense_rt = replace(
        case.runtime,
        kernels={**dict(case.runtime.kernels), "sequence_parallel": sp_decl},
    )
    views = tuple(SimpleNamespace(q_len=1, padding=False) for _ in range(8))
    sp_plan = sequence_parallel.select(
        dense_rt, case.model, views, tokens=8, valid_global_tokens=8,
    )
    assert sp_plan is not None and sp_plan.tp == 4 and sp_plan.valid_cohort_tokens == 8


def test_prefill_large_batch_avoids_decode_only_mc2_and_mc2_omits_redundant_unpack() -> None:
    cfg = MoeCommConfig(
        mode=DYNAMIC, policy=CAPACITY_AND_WORLD_SIZE, capacity=512,
        world_size=8, tp=8, top_k=8, experts_per_device=32, arch_family="A5",
    )
    assert cfg.select(1024, phase="prefill") != MC2
    assert cfg.select(4, phase="decode") == MC2
    cfg_wide = replace(cfg, world_size=32)
    assert cfg_wide.select(1024, phase="prefill") == ALLTOALL

    _, prefill_ops = _ops(
        "deepseek-v3", hardware="a5", phase="prefill", tokens=1024, sequences=1, context=0,
        layers=4, tp=8, ep=8, num_devices=8, sp=True,
    )
    assert not any(
        "MoeDistributeDispatch" in str(dict(op.metadata).get("native_kind", ""))
        or "MoeDistributeCombine" in str(dict(op.metadata).get("native_kind", ""))
        or op.recipe.startswith("mc2_")
        for op in prefill_ops
    )

    _, decode_ops = _ops(
        "deepseek-v3", hardware="a5", phase="decode", tokens=1, sequences=4, context=128,
        layers=4, tp=8, ep=8, num_devices=8,
    )
    moe_layers = {op.layer for op in decode_ops if op.op_id.endswith(".combine")}
    for layer_idx in moe_layers:
        layer_ops = [op for op in decode_ops if op.layer == layer_idx]
        has_mc2_combine = any(
            op.op_id.endswith(".combine")
            and ("mc2" in op.recipe or "MoeDistributeCombine" in str(dict(op.metadata).get("native_kind", "")))
            for op in layer_ops
        )
        if has_mc2_combine:
            assert not any(op.recipe == "route_unpack" for op in layer_ops)


def test_megamoe_accepts_quantized_chain_and_rejects_clamped_activation() -> None:
    def _build_chain(act_limit: float):
        pack = OpSpec(
            op_id="l0.pack", family="routing", phase="decode", stage=0, layer=0,
            requests=("r0",), semantic_shape=(4, 1024), local_shape=(4, 1024),
            padded_shape=(4, 1024), launch_shape=(4, 1024), deps=("l0.router",),
            recipe="route_pack", dtype="bf16",
        )
        disp = OpSpec(
            op_id="l0.dispatch", family="alltoallv", phase="decode", stage=0, layer=0,
            requests=("r0",), semantic_shape=(4, 1024), local_shape=(4, 1024),
            padded_shape=(4, 1024), launch_shape=(4, 1024), deps=(pack.op_id,),
            recipe="mc2_dispatch", dtype="bf16",
            metadata=(("moe_comm_mode", "mc2"), ("mc2_experts_per_rank", 16), ("mc2_top_k", 4)),
        )
        mx_meta = (
            ("weight_format", "fp8_e4m3"), ("activation_format", "fp8_e4m3"),
            ("compute_format", "fp8_e4m3"), ("activation_group_size", 32),
            ("activation_scale_format", "fp8_e8m0"),
        )
        up = OpSpec(
            op_id="l0.expert_up", family="grouped_gemm", phase="decode", stage=0, layer=0,
            requests=("r0",), semantic_shape=(16, 2048, 1024), local_shape=(16, 2048, 1024),
            padded_shape=(16, 2048, 1024), launch_shape=(16, 2048, 1024), deps=(disp.op_id,),
            recipe="grouped_gate_up", dtype="fp8_e4m3", metadata=mx_meta,
        )
        act_kind = "swiglustep" if act_limit > 0 else "silu"
        act = OpSpec(
            op_id="l0.expert_act", family="swiglu", phase="decode", stage=0, layer=0,
            requests=("r0",), semantic_shape=(16, 1024), local_shape=(16, 1024),
            padded_shape=(16, 1024), launch_shape=(16, 1024), deps=(up.op_id,),
            recipe="grouped_swiglu", dtype="bf16",
            metadata=(("activation_kind", act_kind), ("activation_limit", act_limit)),
        )
        quant = OpSpec(
            op_id="l0.expert_down.quantize", family="quantize", phase="decode", stage=0, layer=0,
            requests=("r0",), semantic_shape=(16, 1024), local_shape=(16, 1024),
            padded_shape=(16, 1024), launch_shape=(16, 1024), deps=(act.op_id,),
            recipe="dynamic_quantize", dtype="fp8_e4m3",
        )
        down = OpSpec(
            op_id="l0.expert_down", family="grouped_gemm", phase="decode", stage=0, layer=0,
            requests=("r0",), semantic_shape=(16, 1024, 1024), local_shape=(16, 1024, 1024),
            padded_shape=(16, 1024, 1024), launch_shape=(16, 1024, 1024), deps=(quant.op_id,),
            recipe="grouped_down", dtype="fp8_e4m3", metadata=mx_meta,
        )
        comb = OpSpec(
            op_id="l0.combine", family="alltoallv", phase="decode", stage=0, layer=0,
            requests=("r0",), semantic_shape=(4, 1024), local_shape=(4, 1024),
            padded_shape=(4, 1024), launch_shape=(4, 1024), deps=(down.op_id,),
            recipe="mc2_combine", dtype="bf16", metadata=(("moe_comm_mode", "mc2"),),
        )
        unpack = OpSpec(
            op_id="l0.unpack", family="routing", phase="decode", stage=0, layer=0,
            requests=("r0",), semantic_shape=(4, 1024), local_shape=(4, 1024),
            padded_shape=(4, 1024), launch_shape=(4, 1024), deps=(comb.op_id,),
            recipe="route_unpack", dtype="bf16",
        )
        ops = [pack, disp, up, act, quant, down, comb, unpack]
        return ops, {op.op_id: op for op in ops}, pack, disp, comb, unpack

    rt_shipping = SimpleNamespace(metadata={"enable_fused_mc2": 2}, kernels={})
    ops, by_id, pack, disp, comb, unpack = _build_chain(0.0)
    unclamped = try_lower_fused_frame(
        ops, by_id, {"fused_mc2_level": 2}, rt_shipping, "l0",
        4, 1024, 4, 16, 8, "bf16", pack, disp, comb, unpack, None,
    )
    assert unclamped is not None
    assert any(op.recipe == "megamoe_arch35" for op in unclamped)
    assert not any(op.op_id == "l0.expert_down.quantize" for op in unclamped)

    ops_c, by_id_c, pack_c, disp_c, comb_c, unpack_c = _build_chain(7.0)
    clamped = try_lower_fused_frame(
        ops_c, by_id_c, {"fused_mc2_level": 2}, rt_shipping, "l0",
        4, 1024, 4, 16, 8, "bf16", pack_c, disp_c, comb_c, unpack_c, None,
    )
    assert clamped is None


def test_a5_mxfp8_residual_norm_fuses_add_rmsnorm_quant_and_unit_offset_guarded() -> None:
    _, ops = _ops(
        "qwen3-32b", hardware="a5", phase="decode", tokens=1, sequences=4, context=128,
        layers=2, tp=1, ep=1, num_devices=1, quant="mxfp8",
    )
    fused_norms = [op for op in ops if op.family == "add_rmsnorm_quant"]
    assert len(fused_norms) >= 3
    by_id = {op.op_id: op for op in ops}
    for op in ops:
        if op.family == "rmsnorm_quant":
            assert not any(
                by_id.get(dep) is not None and by_id[dep].recipe == "residual_add"
                for dep in op.deps
            )

    case_27b = parse_case(compose_case("qwen3.5-27b", hardware="a5", layers=2, tp=1, ep=1, num_devices=1))
    rt: dict = {"lowering_passes": ["activation_quantization"], "metadata": {}}
    ctx = MechanismContext(
        model=case_27b.model, sku=case_27b.machine.sku, arch="a5",
        parallel={"tp": 1, "ep": 1, "dp": 1}, caller_explicit=frozenset(), options={},
    )
    _apply_basic_norm_quant(rt, ctx)
    assert "norm_quant" not in rt["lowering_passes"]


def test_gdn_high_tp_replicates_key_heads_and_interleaved_emits_fused_split() -> None:
    for tp, expected_qkvz_width in ((32, 1152), (64, 576)):
        _, ops = _ops(
            "qwen3.8-2.4t-a95b", hardware="a5", phase="decode", tokens=1, sequences=4,
            context=128, layers=2, tp=tp, ep=tp, num_devices=tp,
        )
        qkvz = next(op for op in ops if op.op_id.endswith(".linear_qkvz"))
        assert qkvz.local_shape[1] == expected_qkvz_width
        assert any(op.recipe == "gdn_key_head_replicate" for op in ops)
        assert not any(op.op_id.endswith(".qkv_concatenate") for op in ops)

    _, next80_ops = _ops(
        "qwen3-next-80b-a3b", hardware="a3", phase="decode", tokens=1, sequences=4,
        context=128, layers=2, tp=4, ep=4, num_devices=4,
    )
    assert any(op.recipe == "fused_qkvzba_split_reshape_cat" for op in next80_ops)
    assert not any(op.op_id.endswith(".qkv_concatenate") for op in next80_ops)


def test_moe_gating_dispatches_sigmoid_sqrtsoftplus_and_softmax_natives() -> None:
    sig_plan = SimpleNamespace(router_mode="sigmoid", router_noaux_tc=False)
    assert _native_gating_dispatch(sig_plan, hash_table=False, noaux_tc=False) == (TOPK_NATIVE_KIND, 1)
    sp_plan = SimpleNamespace(router_mode="sqrtsoftplus", router_noaux_tc=False)
    assert _native_gating_dispatch(sp_plan, hash_table=False, noaux_tc=False) == (HASH_NATIVE_KIND, 2)
    sm_plan = SimpleNamespace(router_mode="softmax", router_noaux_tc=False)
    assert _native_gating_dispatch(sm_plan, hash_table=False, noaux_tc=False) == (SOFTMAX_NATIVE_KIND, 0)


def test_bf16_lightning_indexer_binds_non_quant_native_kernel() -> None:
    model = load_model("deepseek-v3.2")
    bf16_quant = replace(model.quant, index_dtype="bf16", activation_format="fp8_e4m3")
    assert index_operands_quantized(bf16_quant, model) is False
    fp8_quant = replace(model.quant, index_dtype="fp8_e4m3")
    assert index_operands_quantized(fp8_quant, model) is True

    case = parse_case(compose_case(
        "deepseek-v3.2", hardware="a5", layers=4, tp=8, ep=8, num_devices=8,
    ))
    bf16_case = replace(
        case,
        model=replace(case.model, quant=replace(case.model.quant, index_dtype="bf16")),
    )
    ops = target_operators(bf16_case, phase="decode", tokens=1, sequences=4, context=512)
    idx_ops = [op for op in ops if op.recipe in {"index_scores", "quant_lightning_indexer"}]
    assert idx_ops
    for op in idx_ops:
        assert dict(op.metadata).get("native_kind") != "aclnnQuantLightningIndexerV2"


def test_deepseek_v4_head_norm_inverse_rope_collapse_and_mla_prolog_ranks() -> None:
    assert {1024, 1280} <= MLA_PROLOG_QUERY_LORA_RANKS
    q_wq = QuantSpec(name="test", exclusions=("indexer.wq",))
    assert _indexer_q_category(q_wq) == "indexer.wq"
    q_fp8 = QuantSpec(name="test", exclusions=("indexer_fp8",))
    assert _indexer_q_category(q_fp8) == "indexer_fp8"

    _, v4_ops = _ops(
        "deepseek-v4-flash-0731", hardware="a5", phase="decode", tokens=1, sequences=8,
        context=128, layers=4, tp=8, ep=8, num_devices=8, scope="model",
    )
    q_head_norms = [op for op in v4_ops if op.recipe == "q_head_norm"]
    assert q_head_norms
    assert all(dict(op.metadata).get("norm_has_weight") is False for op in q_head_norms)
    assert any(
        op.op_id.endswith(".output_inverse_rotary") and dict(op.metadata).get("inverse") is True
        for op in v4_ops
    )

    collapse_proj = next(op for op in v4_ops if op.recipe == "hc_final_collapse_proj")
    assert collapse_proj.flops == pytest.approx(2 * 8 * 4 * (4 * 4096))
    assert any(op.recipe == "hc_final_collapse_sigmoid" for op in v4_ops)
    assert any(op.recipe == "hc_final_collapse_mix" for op in v4_ops)

    _, v41_a3_ops = _ops(
        "deepseek-v4.1-flash", hardware="a3", phase="decode", tokens=1, sequences=4,
        context=128, layers=2, tp=8, ep=8, num_devices=8,
    )
    assert any(
        op.op_id.endswith(".output_inverse_rotary") and dict(op.metadata).get("inverse") is True
        for op in v41_a3_ops
    )


def test_gqa_ordinary_rope_output_gate_gemma4_and_mxfp8_c8_decode() -> None:
    _, qwen3_ops = _ops(
        "qwen3-32b", hardware="a5", phase="decode", tokens=1, sequences=4, context=128,
        layers=1, tp=1, ep=1, num_devices=1,
    )
    assert any(op.recipe == "q_norm" for op in qwen3_ops)
    assert any(op.recipe == "k_norm" for op in qwen3_ops)
    assert not any(op.recipe == "qk_norm" for op in qwen3_ops)
    assert not any(op.op_id.endswith(".qkv.value") for op in qwen3_ops)

    _, q27_ops = _ops(
        "qwen3.5-27b", hardware="a5", phase="decode", tokens=1, sequences=4, context=128,
        layers=4, tp=1, ep=1, num_devices=1,
    )
    gate_op = next(op for op in q27_ops if op.op_id.endswith(".output_gate"))
    gate_meta = dict(gate_op.metadata)
    assert gate_meta.get("input_tensor_count") == 2
    assert gate_op.mte2_bytes == pytest.approx(2 * gate_op.mte3_bytes)

    _, g26_ops = _ops(
        "gemma-4-26b-a4b", hardware="a5", phase="decode", tokens=1, sequences=1, context=128,
        layers=6, tp=1, ep=1, num_devices=1,
    )
    full_qkv = next(op for op in g26_ops if op.layer == 5 and op.recipe == "qkv_projection")
    assert full_qkv.local_shape[1] == 10240
    assert any(op.layer == 5 and op.recipe == "v_norm" for op in g26_ops)
    g26_layer_norms = [
        op for op in g26_ops
        if op.layer == 0
        and op.family in {"rmsnorm", "rmsnorm_quant", "add_rmsnorm", "add_rmsnorm_quant"}
        and op.recipe not in {"q_norm", "k_norm", "v_norm", "qk_norm"}
    ]
    assert len(g26_layer_norms) == 7

    _, g31_ops = _ops(
        "gemma-4-31b", hardware="a5", phase="decode", tokens=1, sequences=1, context=128,
        layers=6, tp=1, ep=1, num_devices=1,
    )
    full_31_qkv = next(op for op in g31_ops if op.layer == 5 and op.recipe == "qkv_projection")
    assert full_31_qkv.local_shape[1] == 20480
    assert any(op.layer == 0 and op.op_id.endswith(".post_attn_norm") for op in g31_ops)
    assert any(op.layer == 0 and op.op_id.endswith(".post_ffn_norm") for op in g31_ops)

    # Safe c8 & MXFP8 PA_NZ GQA decode defaults and direct Arch35 Cube MMA without BF16 dequant.
    for kv_tag in ("c8", "mxfp8", "qk_mxfp8_dynamic_v_mxfp8_per_channel"):
        case_kv = parse_case(compose_case("qwen3-32b", hardware="a5", layers=1, tp=1, ep=1, num_devices=1))
        case_kv = replace(case_kv, model=replace(case_kv.model, quant=replace(case_kv.model.quant, kv_dtype=kv_tag)))
        updated, _ = body_defaults.attention_defaults(case_kv, "decode", tokens=4)
        assert updated is not None
    mxfp8_op = OpSpec(
        op_id="probe.attn", family="attention", phase="decode", stage=0, layer=0,
        requests=("r0",), semantic_shape=(1, 32, 128), local_shape=(1, 32, 128),
        padded_shape=(1, 32, 128), launch_shape=(1, 32, 128), recipe="flash_attention_decode",
        dtype="bf16",
        metadata=(("gqa_decode", {
            "requests": ((1, 256),), "query_heads": 32, "kv_heads": 8, "head_dim": 128,
            "query_dtype": "bf16", "kv_dtype": "qk_mxfp8_dynamic_v_mxfp8_per_channel",
            "causal": True, "window": 0, "cp": 1,
        }),),
    )
    spec = StagedGQADecode(kv_mapping="fai_host_arch35", target_cores=32, query_tile=16).spec(mxfp8_op)
    assert spec.query_dtype == "fp8_e4m3"
    assert spec.kv_cache.needs_conversion is False
    assert spec.kv_cache.dequant_ops_per_element == 0
    assert spec.kv_cache.key_payload_row == 128 + 4


def test_minimax_index_q_tp_sharding_clamped_swiglu_keypool_pack_and_nextn_norms() -> None:
    _, mm_ops = _ops(
        "minimax-m3", hardware="a5", phase="decode", tokens=1, sequences=4, context=512,
        layers=4, tp=8, ep=8, num_devices=8,
    )
    idx_wq = next(op for op in mm_ops if op.layer == 3 and op.op_id.endswith(".index_wq"))
    assert idx_wq.local_shape[1] == 128
    assert idx_wq.flops == pytest.approx(2 * 4 * 128 * 6144)
    idx_scores = next(op for op in mm_ops if op.layer == 3 and op.op_id.endswith(".index_scores"))
    assert dict(idx_scores.metadata).get("native_kind") == "aclnnMsaIndexScore"

    step_plan = replace(load_model("step-5-preview").layer_plans[3], ffn_activation="silu", swiglu_limit=7.0)
    step_act = gated_activation(
        plan=step_plan, op_id="s0.l3.act", deps=("in",), rows=8, width=1536,
        dtype="bf16", stage=0, requests=("r0",), phase="decode", recipe="grouped_swiglu",
    )
    assert dict(step_act.metadata).get("activation_kind") == "swiglustep"
    assert step_act.recipe == "grouped_swiglustep"

    glm_plan = replace(step_plan, swiglu_limit=10.0)
    glm_act = gated_activation(
        plan=glm_plan, op_id="s0.l3.act", deps=("in",), rows=8, width=1536,
        dtype="bf16", stage=0, requests=("r0",), phase="decode", recipe="grouped_swiglu",
    )
    assert dict(glm_act.metadata).get("activation_kind") == "swiglu_clamped"
    assert glm_act.recipe == "grouped_swiglu_clamped"

    _, glm53_ops = _ops(
        "glm-5.3-flash", hardware="a5", phase="decode", tokens=1, sequences=8, context=512,
        layers=4, tp=1, ep=1, num_devices=1,
    )
    key_pool = next(op for op in glm53_ops if op.recipe == "cann_key_pool")
    assert key_pool.flops == pytest.approx(4 * 8 * 128 * 4096)
    pack_op = next(op for op in glm53_ops if op.recipe == "index_topk")
    assert pack_op.family == "elementwise"
    assert dict(pack_op.metadata).get("native_kind") == "glm5_next_cann_keypool_pack"

    _, step_ops = _ops(
        "step-5-preview", hardware="a5", phase="decode", tokens=4, sequences=1, context=128,
        layers=4, tp=8, ep=8, num_devices=8, mtp_proposal_tokens=3, scope="model",
    )
    assert any(op.op_id.endswith("enorm") and op.recipe == "draft_input_norm" for op in step_ops)
    assert any(op.op_id.endswith("hnorm") and op.recipe == "draft_input_norm" for op in step_ops)


def test_mtp_verify_moe_tokens_glm_mean_collapse_minimax_topk_mc2_unpack_and_layer_quant() -> None:
    # 1. MTP target verification counts expanded verification tokens (8) while draft stays at 4.
    _, mtp_ops = _ops(
        "deepseek-v4-flash-0731", hardware="a5", phase="decode", tokens=MTP_VERIFY_TOKENS,
        sequences=MTP_BATCH_SEQUENCES, context=MTP_CONTEXT_LEN, tp=2, ep=2, num_devices=2,
        mtp_proposal_tokens=1, scope="model",
    )
    tgt_mega = next(op for op in mtp_ops if op.recipe == "megamoe_arch35" and not op.op_id.startswith("draft"))
    drf_mega = next(op for op in mtp_ops if op.recipe == "megamoe_arch35" and op.op_id.startswith("draft"))
    assert tgt_mega.flops == pytest.approx(MTP_TARGET_MEGAMOE_FLOPS)
    assert dict(tgt_mega.metadata)["megamoe_geometry"]["routed_rows"] == pytest.approx(MTP_TARGET_ASSIGNMENTS)
    assert drf_mega.flops == pytest.approx(MTP_DRAFT_MEGAMOE_FLOPS)
    assert dict(drf_mega.metadata)["megamoe_geometry"]["routed_rows"] == pytest.approx(MTP_DRAFT_ASSIGNMENTS)

    # 2. GLM-5.3-Flash uses parameterless mean stream collapse (0 FLOPs) and omits draft collapse.
    _, glm_ops = _ops(
        "glm-5.3-flash", hardware="a5", phase="decode", tokens=MTP_VERIFY_TOKENS,
        sequences=MTP_BATCH_SEQUENCES, context=SMALL_CONTEXT, layers=FOUR_LAYERS,
        tp=1, ep=1, num_devices=1, mtp_proposal_tokens=1, scope="model",
    )
    assert not any(op.recipe == "hc_final_collapse_proj" for op in glm_ops)
    mean_ops = [op for op in glm_ops if op.recipe == "hc_final_collapse_mean" and not op.op_id.startswith("draft")]
    assert len(mean_ops) == 1 and mean_ops[0].flops == 0.0
    assert not any(op.op_id.startswith("draft") and "hc_head" in op.op_id for op in glm_ops)

    # 3. MiniMax-M3 / M3.1-Flash TopKV2 emits 16 block IDs (256 B at B=4) while keeping index_topk=2048.
    for mm_name, idx_layer in (("minimax-m3", 3), ("minimax-m3.1-flash", 0)):
        mm_case, mm_ops = _ops(
            mm_name, hardware="a5", phase="decode", tokens=1, sequences=MTP_BATCH_SEQUENCES,
            context=MTP_CONTEXT_LEN, layers=FOUR_LAYERS, tp=MC2_TEST_EP, ep=MC2_TEST_EP,
            num_devices=MC2_TEST_EP,
        )
        assert mm_case.model.layer_plans[idx_layer].index_topk == MINIMAX_TOPK_TOKENS
        topk_op = next(op for op in mm_ops if op.layer == idx_layer and op.op_id.endswith(".index_topk"))
        assert topk_op.local_shape == (MTP_BATCH_SEQUENCES, MINIMAX_TOPK_BLOCKS)
        assert topk_op.mte3_bytes == pytest.approx(MINIMAX_TOPK_OUTPUT_BYTES)

    # 4. _absorb_redundant_mc2_unpack absorbs route_unpack on MoeDistributeCombine* even when caller is set.
    comb = OpSpec(
        op_id="s0.l0.combine", family="alltoallv", phase="decode", stage=0, layer=0,
        requests=("r0",), semantic_shape=(MTP_BATCH_SEQUENCES, MC2_TEST_HIDDEN),
        local_shape=(MTP_BATCH_SEQUENCES, MC2_TEST_HIDDEN),
        padded_shape=(MTP_BATCH_SEQUENCES, MC2_TEST_HIDDEN),
        launch_shape=(MTP_BATCH_SEQUENCES, MC2_TEST_HIDDEN),
        deps=("s0.l0.expert_down",), recipe="expert_combine", dtype="bf16",
        metadata=(("native_kind", "MoeDistributeCombineV2"),),
    )
    unp = OpSpec(
        op_id="s0.l0.unpack", family="routing", phase="decode", stage=0, layer=0,
        requests=("r0",), semantic_shape=(MTP_BATCH_SEQUENCES, MC2_TEST_HIDDEN),
        local_shape=(MTP_BATCH_SEQUENCES, MC2_TEST_HIDDEN),
        padded_shape=(MTP_BATCH_SEQUENCES, MC2_TEST_HIDDEN),
        launch_shape=(MTP_BATCH_SEQUENCES, MC2_TEST_HIDDEN),
        deps=("s0.l0.combine", "s0.l0.pack", "s0.l0.router"), recipe="route_unpack", dtype="bf16",
    )
    absorbed, new_last = _absorb_redundant_mc2_unpack(
        [comb, unp], unp.op_id, prefix="s0.l0", runtime=SimpleNamespace(moe_recipe="alltoall", metadata=()),
        parallel=ParallelSpec(tp=1, ep=MC2_TEST_EP), transport=None, caller=object(),
    )
    assert new_last == comb.op_id
    assert not any(op.recipe == "route_unpack" for op in absorbed)

    # 5. Per-layer layer_quant_overrides propagate to FFN/MoE lowering & cohort discovery.
    g4_case, g4_ops = _ops(
        "gemma-4-31b", hardware="a5", phase="decode", tokens=1, sequences=1, context=SMALL_CONTEXT,
        tp=1, ep=1, num_devices=1,
    )
    g4_cohorts = discover_layer_cohorts(g4_case)
    assert len(g4_cohorts.cohorts) == GEMMA4_COHORT_COUNT
    assert sum(c.count for c in g4_cohorts.cohorts if c.representative_layer in GEMMA4_BF16_LAYERS) == GEMMA4_BF16_COHORT_TOTAL
    g4_l0 = next(op for op in g4_ops if op.layer == 0 and op.op_id.endswith(".gate_up"))
    g4_l1 = next(op for op in g4_ops if op.layer == 1 and op.op_id.endswith(".gate_up"))
    assert dict(g4_l0.metadata)["weight_format"] != "bf16"
    assert dict(g4_l1.metadata)["weight_format"] == "bf16"

    s5_case, s5_ops = _ops(
        "step-5-preview", hardware="a5", phase="decode", tokens=1, sequences=1, context=SMALL_CONTEXT,
        tp=MC2_TEST_EP, ep=MC2_TEST_EP, num_devices=MC2_TEST_EP,
    )
    s5_cohorts = discover_layer_cohorts(s5_case)
    assert len(s5_cohorts.cohorts) == STEP5_COHORT_COUNT
    assert sum(c.count for c in s5_cohorts.cohorts if c.representative_layer in STEP5_BF16_EXPERT_LAYERS) == STEP5_BF16_COHORT_TOTAL
    s5_l4 = next(op for op in s5_ops if op.layer == STEP5_MOE_FP8_LAYER and op.op_id.endswith(".expert_up"))
    s5_l88 = next(op for op in s5_ops if op.layer == STEP5_MOE_BF16_LAYER and op.op_id.endswith(".expert_up"))
    assert dict(s5_l4.metadata)["weight_format"] != "bf16"
    assert dict(s5_l88.metadata)["weight_format"] == "bf16"
