"""Stage 4B catalog geometry, curated checkpoint preservation, and cache/weight laws."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.catalog.cache import layer_cache_components, request_cache_breakdown
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.cache_index import index_row_bytes
from mindie_sim.catalog.cache_pools import (
    DS_V41_FOLDED_GROUP_TOKENS,
    DS_V41_FOLDED_ROW_BYTES,
    REGIME_ALIASED_WINDOW_COMPRESSED,
    REGIME_DUAL_PITCH_KPOOL_MAMBA,
    REGIME_SHARED_SLAB_PRIVATE_AUX,
    _packed_mla_row_bytes,
    evaluate_hybrid_pool_receipt,
    resolve_pool_allocator_contract,
)
from mindie_sim.catalog.draft_weights import draft_terms
from mindie_sim.catalog.ops import compute_dtype_for
from mindie_sim.catalog.quant import _quant_w4a8_dynamic
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_layer_terms import _layer_resolved_quant, layer_weight_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import _body_terms
from mindie_sim.compose import compose_case
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.stacks.vllm_ascend.profiles import VA_RECOMMENDED_PROFILES

pytestmark = pytest.mark.tier0

GIB = float(1024 ** 3)
MX_GROUP_SIZE = 32
BF16_SCALE_BYTES = 2.0
FP8_MLA_FP16_SCALE_ROW_BYTES = 648
FP8_MLA_FP32_SCALE_ROW_BYTES = 656
V32_INDEX_ROW_BYTES = 130.0
SINGLE_NODE_BLOCK_ROWS = 128
GEMMA4_WINDOW_ROWS = 1024
GEMMA4_PADDED_WINDOW_ROWS = 1152
STEP5_WINDOW_ROWS = 512
STEP5_PADDED_WINDOW_ROWS = 640
STEP5_ROUTER_BYTES_FP32 = 352 * 4096 * 4
GLM_FLASH_HC_FN_BYTES_FP32 = 24 * 16384 * 4
MINIMAX_TOPK_BLOCKS = 16
MINIMAX_TOPK_TOKENS = 2048
MINIMAX_INDEX_DIM = 128
MINIMAX_HIDDEN = 6144
LONGCAT_LAYERS = 28
LONGCAT_MOE_LAYERS = 14
LONGCAT_NGRAM_POST_PROJ_BYTES = 12 * 256 * 3072 * 2
W4A16_BYTES_PER_WEIGHT = 0.625
W4A16_CHECK_ROWS = 64
W4A16_CHECK_COLS = 128
W4A8_SCALE_BYTES = 4.0
EP_EIGHT = 8
STEP5_LAYER_COUNT = 92
LONGCAT_DENSE_LAYERS = 14
LONGCAT_ZERO_EXPERTS = 128
LONGCAT_ROUTER_WIDTH = 384
LONGCAT_NGRAM_EXPECTED_GIB = 58.50007
LONGCAT_KV_EXPECTED_GIB = 0.507568


def test_mxfp4_compute_dtype_follows_activation_format():
    """MXFP4 weights resolve compute dtype from the paired activation format."""
    w4a4 = QuantSpec(name="w4a4", weight_format="mxfp4", activation_format="fp4_e2m1", expert_weight_format="mxfp4")
    w4a4_mx = QuantSpec(name="w4a4_mx", weight_format="mxfp4", activation_format="mxfp4", expert_weight_format="mxfp4")
    w4a8 = QuantSpec(name="w4a8", weight_format="mxfp4", activation_format="fp8_e4m3", expert_weight_format="mxfp4")
    w4a8_mx = QuantSpec(name="w4a8_mx", weight_format="mxfp4", activation_format="mxfp8", expert_weight_format="mxfp4")
    w4a16 = QuantSpec(name="w4a16", weight_format="mxfp4", activation_format="bf16", expert_weight_format="mxfp4")
    assert compute_dtype_for(w4a4, "routed_expert") == ("mxfp4", "fp4_e2m1")
    assert compute_dtype_for(w4a4_mx, "routed_expert") == ("mxfp4", "mxfp4")
    assert compute_dtype_for(w4a8, "routed_expert") == ("fp8_e4m3", "fp8_e4m3")
    assert compute_dtype_for(w4a8_mx, "routed_expert") == ("fp8_e4m3", "mxfp8")
    assert compute_dtype_for(w4a16, "routed_expert") == ("bf16", "bf16")


def test_curated_checkpoints_preserved_when_quant_is_none():
    """Production compose preserves curated per-layer and per-category rules on A5/A3."""
    step5_a5 = parse_case(compose_case("step-5-preview", "a5", tp=4, dp=4, ep=16, num_devices=16))
    assert step5_a5.model.quant.format_for("attn.qkv") == "bf16"
    assert step5_a5.model.quant.format_for("shared_expert") == "bf16"
    assert _layer_resolved_quant(step5_a5.model, step5_a5.model.layer_plans[10]).format_for("routed_expert") == "fp8_e4m3"
    for layer_idx in (88, 89, 90):
        layer_q = _layer_resolved_quant(step5_a5.model, step5_a5.model.layer_plans[layer_idx])
        assert layer_q.format_for("routed_expert") == "bf16"

    qfn_a5 = parse_case(compose_case("qwen3.8-flash-next", "a5", tp=4, dp=1, ep=4, num_devices=4))
    assert qfn_a5.model.quant.format_for("gdn.qkvz") == "bf16"
    assert qfn_a5.model.quant.format_for("ple") == "bf16"
    assert qfn_a5.model.quant.format_for("attn.qkv") == "fp8_e4m3"
    assert qfn_a5.model.quant.format_for("routed_expert") == "mxfp4"

    g31_a5 = parse_case(compose_case("gemma-4-31b", "a5", tp=4, dp=2, ep=1, num_devices=8))
    assert g31_a5.model.quant.format_for("attn.qkv") == "fp8_e4m3"
    assert _layer_resolved_quant(g31_a5.model, g31_a5.model.layer_plans[0]).format_for("dense_mlp") == "mxfp4"
    for exc_layer in (1, 9, 10, 11, 27, 28):
        assert _layer_resolved_quant(g31_a5.model, g31_a5.model.layer_plans[exc_layer]).format_for("dense_mlp") == "bf16"

    g26_a5 = parse_case(compose_case("gemma-4-26b-a4b", "a5", tp=4, dp=2, ep=8, num_devices=8))
    assert g26_a5.model.quant.format_for("attn.qkv") == "fp8_e4m3"
    assert g26_a5.model.quant.format_for("dense_mlp") == "fp8_e4m3"
    assert g26_a5.model.quant.format_for("routed_expert") == "mxfp4"


def test_profile_quant_updates_and_quant_spec_metadata():
    """Profile arch mappings, V4 routed activation group size, and W4A8 scale_only ABI."""
    assert VA_RECOMMENDED_PROFILES["qwen3-30b-a3b"]["quant"]["a5"] == "mxfp8"
    assert VA_RECOMMENDED_PROFILES["glm-5.3"]["quant"]["a5"] == "mxfp8"
    assert VA_RECOMMENDED_PROFILES["qwen3.8-2.4t-a95b"]["quant"]["a3"] == "w4a8"
    assert VA_RECOMMENDED_PROFILES["deepseek-v4-flash-0731"]["quant"]["a3"] == "w4a8"

    q30_a5 = parse_case(compose_case("qwen3-30b-a3b", "a5", tp=4, dp=1, ep=4, num_devices=4))
    assert q30_a5.model.quant.weight_format == "fp8_e4m3"

    glm53_a5 = parse_case(compose_case("glm-5.3", "a5", tp=4, dp=8, ep=32, num_devices=32))
    assert glm53_a5.model.quant.weight_format == "fp8_e4m3"
    assert glm53_a5.model.quant.kv_dtype in {"fp8", "fp8_e4m3"}

    mm31_a5 = parse_case(compose_case("minimax-m3.1-flash", "a5", tp=4, dp=4, ep=16, num_devices=16))
    assert mm31_a5.model.quant.kv_dtype in {"fp8", "fp8_e4m3"}

    v4_flash = load_model("deepseek-v4-flash-0731")
    routed_q = v4_flash.quant.for_category("routed_expert")
    assert routed_q.activation_group_size == MX_GROUP_SIZE
    assert routed_q.activation_scale_format == "fp8_e8m0"

    w4a8 = _quant_w4a8_dynamic()
    assert w4a8.runtime_int8_weight_metadata == "scale_only"
    assert w4a8.runtime_weight_scale_format == "bf16"
    assert w4a8.runtime_weight_scale_bytes == BF16_SCALE_BYTES
    assert w4a8.runtime_activation_scale_format == "fp32"


def test_compose_lowering_activation_quant_mlapo_and_profile_mso():
    """Kimi-K3.1 activation quant, merged MLAPO native fusion, and profile MSO non-lock."""
    k31_a5 = compose_case("kimi-k3.1", "a5", tp=8, dp=1, ep=8, num_devices=8, layers=4)
    assert "activation_quantization" in (k31_a5["runtime"].get("lowering_passes") or ())

    dsv3_merged = compose_case("deepseek-v3", "a5", tp=8, dp=1, ep=8, num_devices=8, layers=4, feature_track="merged")
    lowerings = (dsv3_merged["runtime"].get("kernels") or {}).get("lowerings") or {}
    native_fusion = lowerings.get("native_fusion") or {}
    assert "mla_prolog" in (native_fusion.get("options") or {}).get("roles", ())

    qm_a5 = compose_case("qwen3.5-moe", "a5", tp=2, dp=1, ep=2, num_devices=2, layers=4, feature_track="merged")
    policy = (qm_a5["runtime"].get("metadata") or {}).get("execution_policy") or {}
    assert policy.get("mso_source") != "caller"
    mc2_level = (policy.get("moe_route") or {}).get("effective_enable_fused_mc2")
    assert mc2_level != 0


def test_fp8_mla_kpe_bf16_tail_and_v32_int8_index_k():
    """FP8 MLA retains BF16 k_pe tail + scales; DeepSeek-V3.2 index_k is INT8+FP16 (130 B/tok)."""
    glm52_a5 = parse_case(compose_case("glm-5.2", "a5", tp=8, dp=1, ep=8, num_devices=8, layers=4))
    plan = glm52_a5.model.layer_plans[0]
    assert _packed_mla_row_bytes(glm52_a5.model, plan) == FP8_MLA_FP16_SCALE_ROW_BYTES
    fmt = cache_format(glm52_a5.model, "mla_kv", plan=plan)
    assert fmt.row_bytes(plan.cache_row_elements) == FP8_MLA_FP16_SCALE_ROW_BYTES

    meta = dict(glm52_a5.model.metadata)
    meta["kv_scale_dtype"] = "fp32"
    glm52_fp32_scale = replace(glm52_a5.model, metadata=tuple(sorted(meta.items())))
    assert _packed_mla_row_bytes(glm52_fp32_scale, plan) == FP8_MLA_FP32_SCALE_ROW_BYTES

    v32 = load_model("deepseek-v3.2")
    assert (v32.quant.index_dtype, v32.quant.index_scale_dtype) == ("int8", "fp16")
    assert index_row_bytes(v32, v32.layer_plans[0].index_dim) == V32_INDEX_ROW_BYTES
    v32_a3 = parse_case(compose_case("deepseek-v3.2", "a3", tp=8, dp=1, ep=8, num_devices=8, layers=4))
    assert index_row_bytes(v32_a3.model, v32_a3.model.layer_plans[0].index_dim) == V32_INDEX_ROW_BYTES


def test_single_node_index_k_block_padding_and_swa_plus_one_page():
    """cp=1 index_k rounds up to full pages when padded; SWA allocates +1 page when seq > window."""
    parallel = ParallelSpec(tp=1, dp=1, ep=1, pp=1, cp=1)
    v32 = load_model("deepseek-v3.2")
    idx_plan = v32.layer_plans[0]
    padded = layer_cache_components(v32, idx_plan, parallel, 1, capacity_padded=True)
    unpadded = layer_cache_components(v32, idx_plan, parallel, 1, capacity_padded=False)
    assert unpadded["index_k"] == V32_INDEX_ROW_BYTES
    assert padded["index_k"] == SINGLE_NODE_BLOCK_ROWS * V32_INDEX_ROW_BYTES

    g26 = load_model("gemma-4-26b-a4b")
    swa_g26 = next(p for p in g26.layer_plans if p.window > 0)
    row_bytes_g26 = 2 * swa_g26.num_kv_heads * swa_g26.head_dim * 2
    assert layer_cache_components(g26, swa_g26, parallel, GEMMA4_WINDOW_ROWS, capacity_padded=True)["gqa_kv"] == (
        GEMMA4_WINDOW_ROWS * row_bytes_g26
    )
    assert layer_cache_components(g26, swa_g26, parallel, GEMMA4_WINDOW_ROWS + 1, capacity_padded=True)["gqa_kv"] == (
        GEMMA4_PADDED_WINDOW_ROWS * row_bytes_g26
    )
    assert layer_cache_components(g26, swa_g26, parallel, GEMMA4_WINDOW_ROWS + 1, capacity_padded=False)["gqa_kv"] == (
        GEMMA4_WINDOW_ROWS * row_bytes_g26
    )

    step5 = load_model("step-5-preview")
    swa_s5 = next(p for p in step5.layer_plans if p.window > 0)
    row_bytes_s5 = 2 * swa_s5.num_kv_heads * swa_s5.head_dim * 2
    assert layer_cache_components(step5, swa_s5, parallel, STEP5_WINDOW_ROWS + 1, capacity_padded=True)["gqa_kv"] == (
        STEP5_PADDED_WINDOW_ROWS * row_bytes_s5
    )


def test_qwen_gemma_glm_minimax_step5_deepseek_kimi_geometry_and_weights():
    """Model family geometry, router dtypes, CSA/MiniMax indexers, clones, and drafts."""
    qm = load_model("qwen3.5-moe")
    assert len(qm.layer_plans) == 64
    qm_gqa = next(p for p in qm.layer_plans if p.attention == "gqa")
    assert (qm_gqa.q_output_gate, qm_gqa.qk_norm, qm_gqa.qk_rope_head_dim, qm_gqa.shared_expert_gate) == (
        True, True, 32, True
    )
    assert qm.auxiliary.mtp.layer_plans[0].qk_rope_head_dim == 32

    q27 = load_model("qwen3.5-27b")
    q27_gqa = next(p for p in q27.layer_plans if p.attention == "gqa")
    assert (q27_gqa.qk_rope_head_dim, q27_gqa.qk_nope_head_dim) == (64, 192)
    assert q27.auxiliary.mtp.layer_plans[0].qk_rope_head_dim == 64
    q36 = load_model("qwen3.6-27b")
    assert q36.auxiliary.mtp.layer_plans[0].qk_rope_head_dim == 64

    for key in ("gemma-4-26b-a4b", "gemma-4-31b"):
        gm = load_model(key)
        full_plan = next(p for p in gm.layer_plans if p.window == 0)
        assert (full_plan.qk_rope_head_dim, full_plan.qk_nope_head_dim) == (128, 384)
        v_terms = [t for t in layer_weight_terms(full_plan, gm) if t.name == "v_proj"]
        assert len(v_terms) == 1 and not v_terms[0].source_owned and v_terms[0].runtime_owned

    glm_f = load_model("glm-5.3-flash")
    glm_mla = next(p for p in glm_f.layer_plans if p.attention == "mla")
    assert glm_mla.projection_dtype == ""
    assert (glm_mla.router_noaux_tc, glm_mla.routed_scaling_factor, glm_mla.swiglu_limit) == (True, 2.5, 10.0)
    hc_fn = next(t for t in layer_weight_terms(glm_f.layer_plans[0], glm_f) if t.name == "hc_attn_fn")
    assert term_bytes(hc_fn, glm_f.quant, ParallelSpec(tp=1)) == float(GLM_FLASH_HC_FN_BYTES_FP32)

    mm = load_model("minimax-m3")
    mm_moe = next(p for p in mm.layer_plans if p.ffn == "moe")
    assert (mm_moe.index_topk, mm_moe.candidate_topk_blocks, mm_moe.router_noaux_tc, mm_moe.routed_scaling_factor) == (
        MINIMAX_TOPK_TOKENS, MINIMAX_TOPK_BLOCKS, True, 2.0
    )
    mm_wq = next(t for t in layer_weight_terms(mm_moe, mm) if t.name == "indexer_wq")
    assert mm_wq.shard == "kv_out"
    assert term_bytes(mm_wq, mm.quant, ParallelSpec(tp=8)) == float(MINIMAX_INDEX_DIM * MINIMAX_HIDDEN * 2)

    s5 = load_model("step-5-preview")
    s5_moe = next(p for p in s5.layer_plans if p.ffn == "moe")
    assert (s5_moe.router_mode, s5_moe.router_dtype, s5_moe.router_output_dtype) == ("sigmoid", "fp32", "fp32")
    s5_router = next(t for t in layer_weight_terms(s5_moe, s5) if t.name == "router")
    assert term_bytes(s5_router, s5.quant, ParallelSpec(tp=8)) == float(STEP5_ROUTER_BYTES_FP32)
    s5_a5 = parse_case(compose_case("step-5-preview", "a5", tp=4, dp=4, ep=16, num_devices=16))
    s5_a5_moe = next(p for p in s5_a5.model.layer_plans if p.ffn == "moe")
    assert s5_a5_moe.router_dtype == "fp32"
    s5_csa = next(p for p in s5.layer_plans if p.indexer)
    csa_names = {t.name for t in layer_weight_terms(s5_csa, s5)}
    assert {"sparse_indexer_q", "sparse_indexer_k", "sparse_indexer_z", "sparse_indexer_w", "ssmax_s"} <= csa_names
    s5_dterms = draft_terms(s5.auxiliary.mtp, s5)
    assert not any(t.name == "draft_input_proj" for t in s5_dterms)
    assert all(t.shard == "out_tp" for t in s5_dterms if t.name in {"draft_output_head", "draft_eh_proj"})

    for key in ("deepseek-v4-flash-0731", "deepseek-v4-pro-0813", "deepseek-v4-10t"):
        v4_aux = load_model(key).auxiliary
        assert (v4_aux.mtp or v4_aux.dspark).layers == 1
    v41_aux = load_model("deepseek-v4.1-flash").auxiliary
    assert (v41_aux.mtp or v41_aux.dspark).layers == 3

    assert load_model("kimi-k3-dspark").layer_plans[0].qk_norm is True
    assert load_model("kimi-k3").auxiliary.dspark.qk_norm is True
    assert load_model("kimi-k2").layer_plans[0].packed_input_projection is True
    assert load_model("kimi-k3-dspark-block5").layer_plans[0].packed_input_projection is True
    assert load_model("kimi-k3.1").auxiliary.dspark.layer_plans[0].packed_input_projection is True


def test_longcat_flash_lite_ngram_zero_experts_and_dual_mla_blocks():
    """LongCat-Flash-Lite models 31.41B ngram table, zero_expert_num=128, and 28 MLA + 28 dense MLPs."""
    lc = load_model("longcat-flash-lite")
    assert len(lc.layer_plans) == LONGCAT_LAYERS
    assert sum(p.attention == "mla" for p in lc.layer_plans) == LONGCAT_LAYERS
    assert sum(p.ffn == "moe" for p in lc.layer_plans) == LONGCAT_MOE_LAYERS
    assert sum(p.ffn == "dense" for p in lc.layer_plans) == LONGCAT_DENSE_LAYERS
    assert sum(p.shared_intermediate_size == 6144 for p in lc.layer_plans) == LONGCAT_MOE_LAYERS
    assert lc.zero_expert_num == LONGCAT_ZERO_EXPERTS
    assert all(p.zero_expert_num == LONGCAT_ZERO_EXPERTS for p in lc.layer_plans if p.ffn == "moe")

    body = _body_terms(lc)
    ngram = [t for t in body if t.name == "ngram_embedding"]
    assert len(ngram) == 1
    assert term_bytes(ngram[0], lc.quant, ParallelSpec(tp=1)) / GIB == pytest.approx(
        LONGCAT_NGRAM_EXPECTED_GIB, rel=1e-4
    )
    post_proj = [t for t in body if t.name == "ngram_post_proj"]
    assert len(post_proj) == 1
    assert term_bytes(post_proj[0], lc.quant, ParallelSpec(tp=1)) == LONGCAT_NGRAM_POST_PROJ_BYTES
    assert term_bytes(post_proj[0], lc.quant, ParallelSpec(tp=EP_EIGHT)) == LONGCAT_NGRAM_POST_PROJ_BYTES
    routers = [t for t in body if t.name == "router"]
    assert len(routers) == LONGCAT_MOE_LAYERS
    assert all(t.rows == LONGCAT_ROUTER_WIDTH for t in routers)

    kv_gib = request_cache_breakdown(lc, ParallelSpec(tp=1), 4224)["total_bytes"] * 4 / GIB
    assert kv_gib == pytest.approx(LONGCAT_KV_EXPECTED_GIB, rel=1e-4)


def test_w4a16_offset_residency_glm_mean_collapse_step5_qk_norm_and_dspark_causal_mask():
    """Verify W4A16 scale+offset bytes, GLM mean collapse weights, Step-5 qk_norm, and DSpark causal mask."""
    # 1. W4A16 groupwise (wna16) charges BF16 scale + BF16 offset = 0.625 B/weight; W4A8 keeps single scale.
    w4a16_q = QuantSpec(
        name="w4a16", weight_format="int4", activation_format="bf16",
        expert_weight_format="int4", expert_block=MX_GROUP_SIZE,
        scale_format="bf16", scale_bytes=BF16_SCALE_BYTES,
    )
    expected_w4a16 = float(W4A16_CHECK_ROWS * W4A16_CHECK_COLS) * W4A16_BYTES_PER_WEIGHT
    assert w4a16_q.matrix_bytes(W4A16_CHECK_ROWS, W4A16_CHECK_COLS, "routed_expert") == pytest.approx(
        expected_w4a16
    )
    w4a8_q = _quant_w4a8_dynamic()
    expected_w4a8 = float(W4A16_CHECK_ROWS * (W4A16_CHECK_COLS // 2) + W4A16_CHECK_ROWS * W4A8_SCALE_BYTES)
    assert w4a8_q.matrix_bytes(W4A16_CHECK_ROWS, W4A16_CHECK_COLS, "routed_expert") == pytest.approx(
        expected_w4a8
    )

    # 2. GLM-5.3-Flash uses parameterless mean stream collapse (0 learned hc_head weights),
    #    whereas DeepSeek-V4 retains learned hc_head_fn/base/scale weights.
    glm_flash = load_model("glm-5.3-flash")
    assert all(lp.hc_final_collapse == "mean" for lp in glm_flash.layer_plans)
    assert glm_flash.auxiliary.mtp.mhc_final_collapse == "none"
    assert not any(t.name.startswith("hc_head_") for t in _body_terms(glm_flash))
    ds_v4 = load_model("deepseek-v4-flash-0731")
    assert all(lp.hc_final_collapse == "learned" for lp in ds_v4.layer_plans)
    assert any(t.name == "hc_head_fn" for t in _body_terms(ds_v4))

    # 3. Step-5-Preview enables qk_norm on all target attention layers.
    step5 = load_model("step-5-preview")
    assert len(step5.layer_plans) == STEP5_LAYER_COUNT
    assert all(lp.qk_norm is True for lp in step5.layer_plans)

    # 4. Kimi-K3.1 external kimi-k3-dspark-block5 inherits full_attention_causal=True -> causal mask.
    k31 = load_model("kimi-k3.1")
    assert k31.auxiliary.dspark.noncausal_query_block is False
    k3 = load_model("kimi-k3")
    assert k3.auxiliary.dspark.noncausal_query_block is True


def test_hybrid_pool_kpool_mamba_shared_slab_ple_and_aliased_window_contracts():
    """Verify exact integer hybrid block-pool contracts across KPool-Mamba, QSA+PLE, and aliased-window regimes."""
    # 1. GLM-5.3-Flash 128K spec5 TP8: exact BF16 (B_eff=640, 1,871,708,160 B) & C8 (B_eff=1,152, 1,088,446,464 B).
    for kv_dt, exp_beff, exp_bytes in (
        ("bf16", 640, 1_871_708_160),
        ("int8", 1152, 1_088_446_464),
    ):
        raw_g = compose_case(
            "glm-5.3-flash", hardware="a3", tp=8, dp=1, ep=8, num_devices=8,
            prompt_tokens=131072, output_tokens=16, kv_cache_dtype=kv_dt, mtp_proposal_tokens=5,
        )
        case_g = parse_case(raw_g)
        rec_g = evaluate_hybrid_pool_receipt(
            case_g.model, case_g.runtime.parallel, 131072,
            num_requests=1, runtime=case_g.runtime, query_tokens=6,
        )
        assert rec_g.contract.pool_regime == REGIME_DUAL_PITCH_KPOOL_MAMBA
        assert rec_g.contract.normalized_block_size == exp_beff
        assert rec_g.total_pool_bytes_by_stage == (exp_bytes,)

    # 2. Qwen3.8-Flash-Next shared slab + private PLE (286,720 B/block at spec5, +1 raw ring, DCP=1 guard).
    raw_q = compose_case(
        "qwen3.8-flash-next", hardware="a5", tp=4, dp=1, ep=4, num_devices=4,
        prompt_tokens=4096, output_tokens=16, mtp_proposal_tokens=5,
    )
    case_q = parse_case(raw_q)
    rec_q = evaluate_hybrid_pool_receipt(
        case_q.model, case_q.runtime.parallel, 4096,
        num_requests=1, runtime=case_q.runtime, query_tokens=6,
    )
    assert rec_q.contract.pool_regime == REGIME_SHARED_SLAB_PRIVATE_AUX
    assert rec_q.contract.private_bytes_per_block == 286_720
    assert rec_q.per_request_ple_page_bytes == 286_720
    assert rec_q.contract.raw_ring_blocks_per_req == 1
    assert rec_q.raw_blocks_per_request == 1
    with pytest.raises(ModelingError, match="shared_slab_requires_dcp_1_pcp_1"):
        resolve_pool_allocator_contract(
            case_q.model, ParallelSpec(tp=4, dp=1, ep=4, pp=1, cp=2), runtime=case_q.runtime,
        )

    # 3. DeepSeek-V4.1-Flash A5 folded (544 B/8 tok = +8,704 B on L20) + A3 INT8 ban + DCP=1 enforcement.
    raw_v41 = compose_case(
        "deepseek-v4.1-flash", hardware="a5", tp=8, dp=1, ep=8, num_devices=8,
        prompt_tokens=8192, output_tokens=16, kv_cache_dtype="fp8_e4m3",
    )
    case_v41 = parse_case(raw_v41)
    rec_v41 = evaluate_hybrid_pool_receipt(
        case_v41.model, case_v41.runtime.parallel, 8192,
        num_requests=1, runtime=case_v41.runtime,
    )
    assert rec_v41.contract.pool_regime == REGIME_ALIASED_WINDOW_COMPRESSED
    assert rec_v41.contract.alias_backing_pitches_by_stage == ((131_072, 131_072, 131_072, 69_632),)
    # Isolate non-compressed owner L20 without SWA aliasing to verify exact +544 B/8 tok folded KV term.
    l20 = replace(next(p for p in case_v41.model.layer_plans if p.owns_long_kv and int(p.compress_ratio or 1) == 1), index=0, window=0)
    m_l20 = replace(case_v41.model, layers=1, layer_plans=(l20,), auxiliary=None)
    rt_l20 = replace(case_v41.runtime, pp_layer_partition=())
    c_l20 = resolve_pool_allocator_contract(m_l20, case_v41.runtime.parallel, block_size=128, runtime=rt_l20)
    expected_l20 = 128 * 320 + 128 * 68 + (128 // DS_V41_FOLDED_GROUP_TOKENS) * DS_V41_FOLDED_ROW_BYTES
    assert c_l20.alias_backing_pitches_by_stage == ((expected_l20,),)

    with pytest.raises(ModelingError, match="aliased_window_compressed_requires_dcp_1"):
        resolve_pool_allocator_contract(
            case_v41.model, ParallelSpec(tp=8, dp=1, ep=8, pp=1, cp=2), runtime=case_v41.runtime,
        )
    raw_v41_a3_c8 = compose_case(
        "deepseek-v4.1-flash", hardware="a3", tp=8, dp=1, ep=8, num_devices=8,
        prompt_tokens=8192, output_tokens=16, quant="w4a8_dynamic", kv_cache_dtype="int8",
    )
    case_v41_a3_c8 = parse_case(raw_v41_a3_c8)
    with pytest.raises(ModelingError, match="aliased_window_compressed_rejects_int8_kv"):
        resolve_pool_allocator_contract(
            case_v41_a3_c8.model, case_v41_a3_c8.runtime.parallel, runtime=case_v41_a3_c8.runtime,
        )

    # 4. Prefill chunk length never expands speculative state; expired prefix never frees private SWA pages;
    #    explicit head_dim is honored even without attention_bias.
    from _support.axes import REPRESENTATIVE_SKUS
    from mindie_sim.catalog.cache_ownership import batch_cache_footprint
    from mindie_sim.model_catalog import build_model_spec
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.schema.workload import RequestView

    c_pref = parse_case(compose_case(
        "glm-5.3-flash", tp=8, quant="bf16", mtp_proposal_tokens=0,
        deployment_mode="colocated", prompt_tokens=4096, output_tokens=1,
    ))
    pf1 = batch_cache_footprint(
        c_pref, requests=(RequestView("r", 1, 4095, False, "prefill"),), replica=0, phase="prefill", padded=True,
    )
    pf4k = batch_cache_footprint(
        c_pref, requests=(RequestView("r", 4096, 0, False, "prefill"),), replica=0, phase="prefill", padded=True,
    )
    assert int(pf1.total_bytes[0]) == int(pf4k.total_bytes[0])
    m_pref = Memory(c_pref)
    rec_pf1 = m_pref.resident_memory_receipt(phase="prefill", sequence_tokens=4096, batch_per_dp=1, query_tokens=1).dies[0]
    rec_pf4k = m_pref.resident_memory_receipt(phase="prefill", sequence_tokens=4096, batch_per_dp=1, query_tokens=4096).dies[0]
    assert rec_pf1.recurrent_state_bytes == rec_pf4k.recurrent_state_bytes == 18_452_480

    c_swa = parse_case(compose_case(
        "deepseek-v4.1-flash", tp=8, pp=2, pp_layer_partition=(2, 38),
        quant="bf16", mtp_proposal_tokens=0, deployment_mode="colocated",
        prefix_cache_mode="shared", concurrency=2, prompt_tokens=4096, output_tokens=2,
    ))
    swa_0 = batch_cache_footprint(
        c_swa,
        requests=tuple(RequestView(str(i), 1, 4095, True, "decode", prefix_tokens=0, prefix_cache_key="p") for i in range(2)),
        replica=0, phase="decode", padded=True,
    )
    swa_exp = batch_cache_footprint(
        c_swa,
        requests=tuple(RequestView(str(i), 1, 4095, True, "decode", prefix_tokens=2048, prefix_cache_key="p") for i in range(2)),
        replica=0, phase="decode", padded=True,
    )
    assert int(swa_0.total_bytes[0]) == int(swa_exp.total_bytes[0]) == 1_048_576

    # Shared-prefix footprint is strictly order-invariant across heterogeneous decode views.
    c_ord = parse_case(compose_case(
        "qwen3.8-flash-next", hardware=REPRESENTATIVE_SKUS["a3"], tp=4, quant="bf16",
        mtp_proposal_tokens=5, concurrency=2, prefix_cache_mode="shared",
        prompt_tokens=4096, output_tokens=16, deployment_mode="colocated",
    ))
    views_ord = tuple(
        RequestView(str(q), q, 4096 - q, True, "decode", prefix_tokens=2560, prefix_cache_key="p", draft_proposal_tokens=q - 1)
        for q in (1, 6)
    )
    assert int(batch_cache_footprint(c_ord, requests=views_ord, replica=0, phase="decode", padded=True).total) == (
        int(batch_cache_footprint(c_ord, requests=views_ord[::-1], replica=0, phase="decode", padded=True).total)
    )

    # Explicit head_dim & qk_norm are honored identically with or without optional attention_bias=False.
    cfg_hd = dict(
        model_type="qwen3_moe", num_hidden_layers=2, hidden_size=2048,
        num_attention_heads=32, num_key_value_heads=4, head_dim=128, intermediate_size=6144,
        moe_intermediate_size=768, num_experts=128, num_experts_per_tok=8,
        vocab_size=151936, torch_dtype="bfloat16",
    )
    m_nobias = build_model_spec("typed", cfg_hd)
    m_bias = build_model_spec("typed", {**cfg_hd, "attention_bias": False})
    assert (m_nobias.layer_plans[0].head_dim, m_nobias.layer_plans[0].qk_norm, m_nobias.parameter_count()) == (
        128, True, m_bias.parameter_count(),
    )

    # Emitted PLE gate owner_schedule respects physical AIV core ceiling.
    from mindie_sim.prediction.operator_analysis import target_graphs

    c_ple_a5 = parse_case(compose_case(
        "qwen3.8-flash-next", hardware=REPRESENTATIVE_SKUS["a5"], tp=4, ep=4,
        engram_cpu_offload=False, prompt_tokens=4096, output_tokens=1,
    ))
    ple_ops = target_graphs(c_ple_a5, phase="prefill", tokens=4096, scope="model", layer_limit=4)[0]
    ple_sched = next(op.owner_schedule for op in ple_ops if op.family == "engram_apply")
    aiv_cores = c_ple_a5.machine.dies[0].units("vector")
    assert ple_sched.active_owners <= aiv_cores and ple_sched.max_tiles_per_owner == (ple_sched.work_tiles + aiv_cores - 1) // aiv_cores
