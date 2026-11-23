"""Unified resident memory ledger, per-layer cache ABIs, and weight/draft accounting.

[Role] Lock byte-exact physical invariants across all five native catalog model
  geometries, paired-SFA/prefix ownership, speculative convolution rollback,
  weight/draft precision isolation, simultaneous per-die HBM peak tracking, and
  RecoverSSM snapshot-ring HBM headroom evaluation.
[Invariants] Pure in-memory Tier-0 verification; zero device access, network
  I/O, or host filesystem mutation. Every byte oracle matches the authoritative
  producer/ledger arithmetic without tolerance drift.
[Timeouts & Locks] Deterministic arithmetic completing in a few seconds; no locks.
[Side-Effect Policy] Read-only catalog loads and in-memory Memory instances only.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from _support.builders import dense_case, linear_layer, view
from _support.memory_cases import SPARSE_OFFLOAD_SKU
from mindie_sim import compose_case
from mindie_sim.catalog.cache import (
    block_round,
    cache_component_parts,
    cache_row_parts,
    cache_row_split,
    layer_cache_components,
    request_cache_breakdown,
)
from mindie_sim.catalog.cache_format import (
    CacheFormat,
    cache_format,
    dense_c8_kv_planes,
    mxfp8_kv_packet_planes,
    mxfp8_kv_packet_section_bytes,
    resolve_layer_cache_abis,
)
from mindie_sim.catalog.cache_index import index_selection_cost
from mindie_sim.catalog.cache_ownership import (
    SHARING_PRIVATE,
    SHARING_SHARED,
    batch_cache_footprint,
    cache_component_ranges,
)
from mindie_sim.catalog.draft import draft_quant
from mindie_sim.catalog.draft_weights import draft_terms
from mindie_sim.catalog.paired_sfa import (
    minimum_block_size,
    normalize_formats,
    page_groups,
    paired_sfa_group_descriptors,
)
from mindie_sim.catalog.speculative_state import speculative_state_plan
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes, term_payload_bytes
from mindie_sim.catalog.weights import (
    _iter_model_layer_terms,
    body_head_terms,
    cache_row_bytes_by_layer,
    engram_gate_terms_by_layer,
    layer_weight_terms,
)
from mindie_sim.prediction.capacity import (
    Memory,
    MemoryPolicy,
    ResidentGrantEntry,
    estimate_recover_ssm_snapshot_ring_bytes,
    prepared_weight_bytes,
    recover_ssm_snapshot_ring_breakdown,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.dspark import DsparkSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.stack import PreparedWeightSpec

pytestmark = pytest.mark.tier0

LARGE_INT_BOUNDARY = 2**53 + 1
DEFAULT_BLOCK = 128
KIMI_ATTN_RES_BYTES = 14_336
KIMI_LATENT_MXFP8_BYTES = 26_492_928
KIMI_LATENT_MXFP8_PAYLOAD = 25_690_112
KIMI_LATENT_MXFP8_SCALES = 802_816
KIMI_PREPARED_F_TP4_BYTES = 44_040_192
QWEN_GDN_CONV_Q4_BYTES = 122_880
QWEN_GDN_CONV_Q1_BYTES = 61_440
NOPE_MLA_FP16_SCALE_ROW = 520
NOPE_MLA_FP32_SCALE_ROW = 528
ROPE64_MLA_INT8_ROW = 656
QWEN3_ASYMMETRIC_KV_ROW_TP8 = 388
MXFP8_KV_PAGE_BYTES = 33_792
KIMI_TP8_B16_K5_RING_BYTES = 4_341_104_640


def _load_full_case(model_name: str, **kwargs):
    params = dict(hardware="a5", tp=1, dp=1, ep=1)
    params.update(kwargs)
    return parse_case(compose_case(model_name, **params))


def _kimi_mixed_quant() -> QuantSpec:
    mx8 = {
        "weight_format": "fp8_e4m3",
        "activation_format": "fp8_e4m3",
        "weight_block": (1, 32),
        "scale_format": "fp8_e8m0",
        "scale_bytes": 1.0,
        "activation_group_size": 32,
        "activation_scale_format": "fp8_e8m0",
        "activation_scale_pack": 2,
    }
    bf16 = {"weight_format": "bf16", "activation_format": "bf16"}
    return QuantSpec(
        name="kimi_mixed_mxfp",
        weight_format="fp8_e4m3",
        activation_format="fp8_e4m3",
        expert_weight_format="mxfp4",
        weight_block=(1, 32),
        expert_block=32,
        scale_format="fp8_e8m0",
        scale_bytes=1.0,
        activation_group_size=32,
        activation_scale_format="fp8_e8m0",
        activation_scale_pack=2,
        kv_dtype="fp8_e4m3",
        index_dtype="fp8_e4m3",
        index_scale_dtype="fp16",
        exclusions=("norm", "router", "lm_head", "embedding", "mhc", "vision", "projector"),
        category_overrides={
            "kda.q": mx8,
            "kda.k": mx8,
            "kda.v": mx8,
            "latent_moe": mx8,
            "indexer.wq": mx8,
            "indexer.wk": bf16,
            "indexer.weights": bf16,
            "mla.o": bf16,
            "kda.o": bf16,
        },
    )


def test_integer_block_round_and_separate_interleave_from_page_block():
    assert block_round(LARGE_INT_BOUNDARY, DEFAULT_BLOCK) == 2**53 + DEFAULT_BLOCK
    assert block_round(0, DEFAULT_BLOCK) == 0
    assert block_round(129, DEFAULT_BLOCK) == 256

    case = dense_case(layers=1)
    layer = replace(case.model.layer_plans[0], num_kv_heads=1)
    parallel = ParallelSpec(tp=2, cp=2, cp_plan="dcp")
    row_bytes = 2 * 1 * layer.head_dim * 2
    # cp_interleave=64 differs from page allocation block_size=128.
    # For seq=150, rank 0 owns [0..63] + [128..149] = 86 rows -> rounds to 128,
    # rank 1 owns [64..127] = 64 rows -> rounds to 128 at page_block=128.
    unpadded_r0 = layer_cache_components(
        case.model, layer, parallel, 150,
        capacity_padded=False, interleave=64, block_size=128, rank=0,
    )["gqa_kv"]
    unpadded_r1 = layer_cache_components(
        case.model, layer, parallel, 150,
        capacity_padded=False, interleave=64, block_size=128, rank=1,
    )["gqa_kv"]
    padded_r0 = layer_cache_components(
        case.model, layer, parallel, 150,
        capacity_padded=True, interleave=64, block_size=128, rank=0,
    )["gqa_kv"]
    padded_r1 = layer_cache_components(
        case.model, layer, parallel, 150,
        capacity_padded=True, interleave=64, block_size=128, rank=1,
    )["gqa_kv"]
    assert unpadded_r0 == 86 * row_bytes
    assert unpadded_r1 == 64 * row_bytes
    assert padded_r0 == 128 * row_bytes
    assert padded_r1 == 128 * row_bytes

    # Explicit owner_spans must honor capacity_padded=True via page_block rounding.
    spans = ((0, 50), (50, 150))
    span_valid = layer_cache_components(
        case.model, layer, parallel, 150,
        capacity_padded=False, interleave=64, block_size=128, rank=0, owner_spans=spans,
    )["gqa_kv"]
    span_alloc = layer_cache_components(
        case.model, layer, parallel, 150,
        capacity_padded=True, interleave=64, block_size=128, rank=0, owner_spans=spans,
    )["gqa_kv"]
    assert span_valid == 50 * row_bytes
    assert span_alloc == 128 * row_bytes


def test_kimi_native_geometry_and_three_weight_byte_oracles():
    case = _load_full_case("kimi-k3.1", tp=4, ep=4, num_devices=4)
    model = replace(case.model, quant=_kimi_mixed_quant())
    plans = model.layer_plans
    kda_layers = [p for p in plans if p.attention == "kda"]
    mla_layers = [p for p in plans if p.attention == "mla"]
    idx_owners = [p for p in plans if p.owns_index_k]
    assert len(plans) == 93
    assert len(kda_layers) == 69
    assert len(mla_layers) == 24
    assert len(idx_owners) == 24

    # Typed per-layer CacheStateABI checks for KDA + MLA + index-K owners.
    kda_abis = resolve_layer_cache_abis(model, kda_layers[0], case.runtime.parallel, runtime=case.runtime)
    mla_abis = resolve_layer_cache_abis(model, mla_layers[0], case.runtime.parallel, runtime=case.runtime)
    assert {a.component_role for a in kda_abis} == {"recurrent_state", "conv_history"}
    assert {a.component_role for a in mla_abis} == {"mla_latent", "index_k"}

    # Oracle 1: [1, 7168] AttnRes vectors (layer attn_res_proj, mlp_res_proj, output_attn_res_proj).
    par_tp4 = ParallelSpec(tp=4, ep=4)
    kda_with_prep = replace(
        model,
        metadata=tuple({
            **dict(model.metadata),
            "kda_input_projection": {"layers": {str(kda_layers[0].index): {"dtype": "bf16"}}},
        }.items()),
    )
    kda_terms = {t.name: t for t in layer_weight_terms(kda_layers[0], kda_with_prep)}
    for res_name in ("attn_res_proj", "mlp_res_proj"):
        res_term = kda_terms[res_name]
        assert term_bytes(res_term, model.quant, par_tp4) == KIMI_ATTN_RES_BYTES
        assert term_payload_bytes(res_term, model.quant, par_tp4) == KIMI_ATTN_RES_BYTES
    head_by_name = {t.name: t for t in body_head_terms(model)}
    out_res = head_by_name["output_attn_res_proj"]
    assert term_bytes(out_res, model.quant, par_tp4) == KIMI_ATTN_RES_BYTES
    assert term_payload_bytes(out_res, model.quant, par_tp4) == KIMI_ATTN_RES_BYTES

    # Oracle 2: Replicated latent [3584, 7168] MXFP8 = 26,492,928 B (payload + E8M0/32 scales).
    moe_layer = next(p for p in plans if p.ffn == "moe" and p.expert_latent_size == 3584)
    moe_terms = {t.name: t for t in layer_weight_terms(moe_layer, model)}
    latent_down = moe_terms["latent_down"]
    assert term_bytes(latent_down, model.quant, par_tp4) == KIMI_LATENT_MXFP8_BYTES
    assert term_payload_bytes(latent_down, model.quant, par_tp4) == KIMI_LATENT_MXFP8_PAYLOAD
    assert (
        term_bytes(latent_down, model.quant, par_tp4)
        - term_payload_bytes(latent_down, model.quant, par_tp4)
    ) == KIMI_LATENT_MXFP8_SCALES

    # Oracle 3: TP4 prepared BF16 F[3072, 7168] = 44,040,192 B runtime-only (0 B source).
    prep_f = kda_terms["kda_composed_f"]
    assert term_bytes(prep_f, model.quant, par_tp4, source=True) == 0.0
    assert term_bytes(prep_f, model.quant, par_tp4, source=False) == KIMI_PREPARED_F_TP4_BYTES
    prep_spec = PreparedWeightSpec(
        allocation_id="kda_composed_f",
        source_tensors=("kda_f_a", "kda_f_b"),
        transform="kda_compose_fb_fa",
        scope="target",
        layer_indices=(kda_layers[0].index,),
        local_shape=(3072, 7168),
        dtype="bf16",
        layout="ND",
        retains_source_tensors=True,
        lifetime="persistent",
    )
    assert prepared_weight_bytes(prep_spec) == KIMI_PREPARED_F_TP4_BYTES

    # Separate indexer.wk (BF16) vs indexer.wq (MXFP8) on MLA layer.
    mla_terms = {t.name: t for t in layer_weight_terms(mla_layers[0], model)}
    wk_term = mla_terms["indexer_wk"]
    wq_term = mla_terms["indexer_wq"]
    assert wk_term.category == "indexer.wk"
    assert wq_term.category == "indexer.wq"
    assert wk_term.quant.format_for(wk_term.category) == "bf16"
    assert wq_term.quant.format_for(wq_term.category) == "fp8_e4m3"
    assert term_bytes(wk_term, model.quant, par_tp4) == wk_term.rows * wk_term.cols * 2

    # Per-layer quant overrides never collide in _iter_model_layer_terms cache.
    k0, k1 = kda_layers[0], kda_layers[1]
    two_layer = replace(
        model,
        layers=2,
        layer_plans=(replace(k0, index=0), replace(k1, index=1)),
        metadata=tuple({
            **dict(model.metadata),
            "layer_quant_overrides": {
                "1": {"category_overrides": {"kda.q": {"weight_format": "bf16", "activation_format": "bf16"}}},
            },
        }.items()),
    )
    iter_terms = dict(_iter_model_layer_terms(two_layer))
    q0 = next(t for t in iter_terms[0] if t.name == "kda_q")
    q1 = next(t for t in iter_terms[1] if t.name == "kda_q")
    assert term_bytes(q0, two_layer.quant, par_tp4) != term_bytes(q1, two_layer.quant, par_tp4)


def test_qwen_flash_geometry_conv_rollback_and_all_float_draft():
    case = _load_full_case("qwen3.8-flash-next", tp=1, ep=1)
    model = case.model
    plans = model.layer_plans
    gdn_layers = [p for p in plans if p.attention == "gdn"]
    gqa_layers = [p for p in plans if p.attention == "gqa"]
    kpool_owners = [p for p in plans if p.owns_index_k and p.index_kpool == 4]
    assert len(plans) == 48
    assert len(gdn_layers) == 36
    assert len(gqa_layers) == 12
    assert len(kpool_owners) == 12
    assert model.engram is not None

    # Conv rollback slice uses conv_width*(kernel - 1 + q - 1), matching GDN writer 122,880 B at q=4.
    par_tp1 = ParallelSpec(tp=1)
    q1_plan = speculative_state_plan(
        model, gdn_layers[0], par_tp1, query_tokens=1, recover_ssm=False, capacity_padded=True,
    )
    q4_plan = speculative_state_plan(
        model, gdn_layers[0], par_tp1, query_tokens=4, recover_ssm=False, capacity_padded=True,
    )
    assert q1_plan.conv_bytes == QWEN_GDN_CONV_Q1_BYTES
    assert q4_plan.conv_bytes == QWEN_GDN_CONV_Q4_BYTES

    # All-FLOAT 512-expert MTP clears inherited target expert formats AND target role overrides.
    quant_target = QuantSpec(
        name="qwen_flash_mixed",
        weight_format="fp8_e4m3",
        activation_format="fp8_e4m3",
        expert_weight_format="mxfp4",
        weight_block=(1, 32),
        expert_block=32,
        scale_format="fp8_e8m0",
        scale_bytes=1.0,
        activation_group_size=32,
        activation_scale_format="fp8_e8m0",
        activation_scale_pack=2,
        exclusions=("mtp", "ple", "norm", "router", "lm_head", "embedding"),
        category_overrides={
            "attn.qkv": {"weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3"},
            "dspark_routed_expert": {"activation_format": "fp8_e4m3", "activation_group_size": 32},
        },
    )
    quant_model = replace(model, quant=quant_target)
    draft = quant_model.auxiliary.mtp
    assert draft is not None and draft.num_experts == 512
    dq = draft_quant(quant_model, draft)
    assert dq.weight_format == "bf16"
    assert dq.expert_weight_format == "bf16"
    assert dq.activation_format == "bf16"
    assert dq.category_overrides == ()
    assert dq.format_for("dspark_routed_expert") == "bf16"
    assert dq.format_for("attn.qkv") == "bf16"
    assert dq.linear_method("dspark_routed_expert") == "dense"
    assert dq.linear_method("attn.qkv") == "dense"

    d_terms = draft_terms(draft, quant_model)
    expert_gate = next(t for t in d_terms if t.name == "draft_expert_gate")
    expected_expert_bytes = expert_gate.rows * expert_gate.cols * expert_gate.count * 2
    assert term_bytes(expert_gate, quant_target, par_tp1) == expected_expert_bytes
    assert term_payload_bytes(expert_gate, quant_target, par_tp1) == expected_expert_bytes

    # PLE gate/proj respects "ple" exclusion and stays BF16 with 0 scale bytes.
    ple_by_layer = engram_gate_terms_by_layer(quant_model)
    assert ple_by_layer
    for layer_terms in ple_by_layer.values():
        for term in layer_terms:
            if term.category == "ple_proj":
                assert term_bytes(term, quant_target, par_tp1) == term.rows * term.cols * 2


def test_glm_flash_nope_mla_row_formats_and_heterogeneous_isolation():
    case = _load_full_case("glm-5.3-flash", tp=1, ep=1)
    model = case.model
    plans = model.layer_plans
    kda_layers = [p for p in plans if p.attention == "kda"]
    mla_layers = [p for p in plans if p.attention == "mla"]
    assert len(plans) == 45
    assert len(kda_layers) == 34
    assert len(mla_layers) == 11
    assert all(p.qk_rope_head_dim == 0 and p.owns_index_k for p in mla_layers)

    # 520 B (512 NoPE FP8 + 4 FP16 scales) and 528 B (512 NoPE + 4 FP32 scales).
    m_fp8_fp16 = replace(model, quant=replace(model.quant, kv_dtype="fp8_e4m3"))
    fmt_520 = cache_format(m_fp8_fp16, "mla_kv", plan=mla_layers[0])
    assert fmt_520.row_parts(512) == (512, 8, 0)
    assert fmt_520.row_bytes(512) == NOPE_MLA_FP16_SCALE_ROW

    m_fp8_fp32 = replace(
        m_fp8_fp16,
        metadata=tuple({**dict(m_fp8_fp16.metadata), "kv_scale_dtype": "fp32"}.items()),
    )
    fmt_528 = cache_format(m_fp8_fp32, "mla_kv", plan=mla_layers[0])
    assert fmt_528.row_parts(512) == (512, 16, 0)
    assert fmt_528.row_bytes(512) == NOPE_MLA_FP32_SCALE_ROW

    # 656 B = 512 INT8 latent + 4 FP32 scales + 64 BF16 rotary suffix.
    rope_plan = replace(mla_layers[0], index=1, qk_rope_head_dim=64, kv_cache_elements=576)
    m_int8 = replace(model, quant=replace(model.quant, kv_dtype="int8"))
    fmt_656 = cache_format(m_int8, "mla_kv", plan=rope_plan)
    assert fmt_656.row_parts(576) == (640, 16, 0)
    assert fmt_656.row_bytes(576) == ROPE64_MLA_INT8_ROW

    # Heterogeneous model (NoPE layer 0 + RoPE layer 1) never collapses globally.
    hetero = replace(
        m_fp8_fp16,
        layers=2,
        layer_plans=(replace(mla_layers[0], index=0), rope_plan),
    )
    assert cache_format(hetero, "mla_kv", plan=hetero.layer_plans[0]).row_bytes(512) == NOPE_MLA_FP16_SCALE_ROW
    assert cache_format(hetero, "mla_kv", plan=hetero.layer_plans[1]).row_bytes(576) == 648

    # Trailing alignment padding is tracked separately from numerical scale bytes.
    padded_fmt = CacheFormat("fp8_e4m3", group_size=128, scale_dtype="fp16", alignment=64)
    assert padded_fmt.row_parts(512) == (512, 8, 56)
    assert padded_fmt.row_bytes(512) == 576
    m_aligned = replace(
        m_fp8_fp16,
        metadata=tuple({
            **dict(m_fp8_fp16.metadata),
            "cache_representations": {
                "mla_kv": {
                    "dtype": "fp8_e4m3", "group_size": 128,
                    "scale_dtype": "fp16", "alignment": 64,
                },
            },
        }.items()),
    )
    assert cache_row_parts(m_aligned, "mla_kv", 512) == (512, 8, 56)
    assert cache_row_split(m_aligned, "mla_kv", 512) == (512.0, 64.0)
    p_part, s_part, pad_part = cache_component_parts(
        m_aligned, mla_layers[0], ParallelSpec(tp=1), "mla_kv", 576.0,
    )
    assert (p_part, s_part, pad_part) == (512.0, 8.0, 56.0)


def test_deepseek_v41_index_owners_engram_and_compressed_append_delta():
    case = _load_full_case("deepseek-v4.1-flash", tp=1, ep=1)
    model = case.model
    plans = model.layer_plans
    idx_owners = [p.index for p in plans if p.owns_index_k]
    long_owners = [p.index for p in plans if p.owns_long_kv]
    assert len(plans) == 40
    assert all(p.attention == "mla" for p in plans)
    assert idx_owners == [2, 8, 14, 20]
    assert len(long_owners) == 4
    assert model.engram is not None

    # Compressed index append charges floor((ctx+q)/r) - floor(ctx/r), NOT full history.
    owner_plan = plans[2]
    ratio = max(1, owner_plan.compress_ratio, owner_plan.index_kpool)
    assert ratio > 1
    decode_no_flush = index_selection_cost(
        model, owner_plan, context_tokens=100, query_tokens=1,
    )
    assert decode_no_flush.scored_rows == (101 // ratio)
    assert decode_no_flush.cache_write_rows == ((101 // ratio) - (100 // ratio))
    assert decode_no_flush.cache_write_rows < decode_no_flush.scored_rows
    decode_flush = index_selection_cost(
        model, owner_plan, context_tokens=101, query_tokens=1,
    )
    assert decode_flush.cache_write_rows == ((102 // ratio) - (101 // ratio))

    # Qwen3.8 KPool (r=4): ctx=100, q=1 -> 0 completed rows + 1 tail row; ctx=103, q=1 -> 1 completed + 1 tail.
    qwen_case = _load_full_case("qwen3.8-flash-next", tp=1, ep=1)
    qwen_kpool_plan = next(p for p in qwen_case.model.layer_plans if p.owns_index_k and p.index_kpool == 4)
    step_mid = index_selection_cost(qwen_case.model, qwen_kpool_plan, context_tokens=100, query_tokens=1)
    step_edge = index_selection_cost(qwen_case.model, qwen_kpool_plan, context_tokens=103, query_tokens=1)
    assert step_mid.cache_write_rows == 0 and step_mid.tail_write_rows == 1
    assert step_edge.cache_write_rows == 1 and step_edge.tail_write_rows == 1


def test_qwen3_asymmetric_kv_mxfp8_packet_page_and_dense_c8_planes():
    case = _load_full_case("qwen3-32b", tp=8, ep=1, num_devices=8)
    model = case.model
    assert len(model.layer_plans) == 64
    layer0 = model.layer_plans[0]
    assert layer0.num_kv_heads == 8 and layer0.head_dim == 128

    # Asymmetric K=INT8+FP32/128, V=BF16 at TP8 -> 132 + 256 = 388 B/row (NOT 512 B).
    asym_model = replace(
        model,
        metadata=tuple({
            **dict(model.metadata),
            "cache_representations": {
                "gqa_kv_key": {"dtype": "int8", "group_size": 128, "scale_dtype": "fp32"},
                "gqa_kv_value": {"dtype": "bf16"},
            },
        }.items()),
    )
    tp8 = ParallelSpec(tp=8)
    one_tok = layer_cache_components(asym_model, layer0, tp8, 1, capacity_padded=False)
    assert one_tok["gqa_kv"] == QWEN3_ASYMMETRIC_KV_ROW_TP8
    assert cache_row_bytes_by_layer(asym_model)[0] == (0, 8 * QWEN3_ASYMMETRIC_KV_ROW_TP8)

    # QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL: K | K-scale | V | V-scale | padding = 33,792 B.
    sections = mxfp8_kv_packet_section_bytes(1, 128, 128, 128, page_alignment=1)
    assert sections == (16_384, 512, 16_384, 512, 0)
    assert sum(sections) == MXFP8_KV_PAGE_BYTES
    padded_sections = mxfp8_kv_packet_section_bytes(1, 128, 128, 128, page_alignment=4096)
    assert padded_sections[-1] == 3072
    assert sum(padded_sections) == 36_864
    planes = mxfp8_kv_packet_planes(1, 128, 128, 128, page_alignment=4096)
    assert [p.plane_id for p in planes] == [
        "k_payload", "k_scale", "v_payload", "v_scale", "trailing_padding",
    ]
    assert planes[3].allocation_frequency == "page"
    assert planes[3].write_trigger == "page_reassign"
    assert planes[4].plane_kind == "padding" and planes[4].bytes_per_unit == 3072

    mxfp_model = replace(
        model,
        metadata=tuple({
            **dict(model.metadata),
            "kv_cache_tag": "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL",
        }.items()),
    )
    page_128 = layer_cache_components(mxfp_model, layer0, tp8, 128, capacity_padded=False)
    assert page_128["gqa_kv"] == MXFP8_KV_PAGE_BYTES
    mxfp_abis = resolve_layer_cache_abis(mxfp_model, layer0, tp8, runtime=case.runtime)
    assert len(mxfp_abis) == 1 and mxfp_abis[0].total_plane_bytes() == MXFP8_KV_PAGE_BYTES

    # Dense C8 and FAQuant declare payload, checkpoint scales, INT8 offsets, and q_scale.
    c8_planes = dense_c8_kv_planes(1, 128, 128, faquant=True, num_q_heads=8)
    assert {p.plane_id for p in c8_planes} == {
        "k_payload", "v_payload", "k_scale", "v_scale", "k_offset", "v_offset", "q_scale",
    }
    assert next(p for p in c8_planes if p.plane_id == "k_scale").allocation_frequency == "checkpoint"
    assert next(p for p in c8_planes if p.plane_id == "q_scale").allocation_frequency == "layer"


def test_paired_sfa_pp1_capped_draft_no_index_error_and_ownership_invariants():
    # Reproduced PP1 + draft_max_model_len=64 case (2 groups on 1 PP stage -> ZERO IndexError).
    capped = parse_case(compose_case(
        "kimi-k3.1", hardware=SPARSE_OFFLOAD_SKU, tp=8, ep=8, dcp=8, pp=1, layers=4,
        mtp_proposal_tokens=5, max_model_len=131_072, draft_max_model_len=64,
        prompt_tokens=512, output_tokens=16,
    ))
    layer_stage = {p.index: 0 for p in capped.model.layer_plans}
    groups = page_groups(capped, layer_stage, query_tokens=6)
    descs = paired_sfa_group_descriptors(capped, layer_stage, query_tokens=6)
    assert len(groups) == 2 and len(descs) == 2
    assert descs[0].stage == 0 and descs[1].stage == 0 and descs[1].auxiliary is True
    assert descs[1].resident_tokens == 64

    reqs = (view("r0", 6, 4096, prefix=512), view("r1", 6, 4096, prefix=512))
    fp = batch_cache_footprint(capped, requests=reqs, replica=0, phase="decode", padded=True)
    assert fp.stages == 1
    assert fp.draft_total > 0.0
    assert fp.total == pytest.approx(fp.shared_once_total + fp.private_total + fp.draft_total)

    # Separately allocated replay_records survive paired_sfa batch_cache_footprint under recover_ssm.
    capped_recover = replace(
        capped, runtime=replace(capped.runtime, recover_ssm=True),
    )
    fp_rec = batch_cache_footprint(capped_recover, requests=reqs, replica=0, phase="decode", padded=True)
    assert fp_rec.state_total > 0.0

    # Extra prefill checkpoint slots do NOT widen conv rollback in minimum_block_size.
    m0 = minimum_block_size(capped.model, tp=8, proposals=5, extra=0)
    m4 = minimum_block_size(capped.model, tp=8, proposals=5, extra=4)
    assert m0 > 0 and m0 == m4

    # Global membership projection supports a KDA-only PP stage without failing.
    pp2_case = replace(
        capped,
        runtime=replace(
            capped.runtime,
            parallel=replace(capped.runtime.parallel, pp=2),
            pp_layer_partition=(2, 2),
        ),
    )
    # Layers 0, 1 are KDA-only on stage 0; layers 2, 3 include MLA on stage 1.
    pp2_groups = page_groups(pp2_case, {0: 0, 1: 0, 2: 1, 3: 1}, query_tokens=6)
    assert any(g.get("fixed_pages", 0) > 0 for g in pp2_groups)

    # Prefix cache_component_ranges never labels padded tail valid & keeps integer page_ids.
    base = dense_case(layers=1)
    ranges = cache_component_ranges(
        base.model, base.model.layer_plans[0], base.runtime.parallel,
        sequence_tokens=200, prefix_tokens=128, interleave=128, block_size=128,
        rank=None, capacity_padded=True, generation=3,
    )
    sh = next(r for r in ranges if r.sharing == SHARING_SHARED)
    pr = next(r for r in ranges if r.sharing == SHARING_PRIVATE)
    assert sh.generation == 3 and sh.page_ids == (0,)
    assert pr.page_ids == (1,)
    assert pr.valid_bytes < pr.allocated_bytes

    # Different prefix generations never coalesce into one shared pool.
    gen_case = replace(base, runtime=replace(base.runtime, prefix_cache_mode="shared"))
    r_g0 = view("a", 1, 256, prefix=128)
    r_g1 = view("b", 1, 256, prefix=128)
    r_g0_wrap = type("GenView", (), {**r_g0.__dict__, "prefix_generation": 0})()
    r_g1_wrap = type("GenView", (), {**r_g1.__dict__, "prefix_generation": 1})()
    fp_same_gen = batch_cache_footprint(gen_case, requests=(r_g0_wrap, r_g0_wrap), replica=0, phase="decode", padded=False)
    fp_diff_gen = batch_cache_footprint(gen_case, requests=(r_g0_wrap, r_g1_wrap), replica=0, phase="decode", padded=False)
    assert fp_diff_gen.shared_once_total == pytest.approx(2.0 * fp_same_gen.shared_once_total)


def test_resident_memory_ledger_peak_rank_validation_aliases_and_paired_allocation():
    base = dense_case(layers=2)
    tp2_case = replace(
        base,
        machine=replace(
            base.machine,
            dies=(base.machine.dies[0], replace(base.machine.dies[0], rank=1)),
        ),
        runtime=replace(base.runtime, parallel=ParallelSpec(tp=2, dp=1, pp=1)),
    )
    mem = Memory(tp2_case, MemoryPolicy(derive_workspace=False))

    # Invalid ranks raise ValueError instead of silently coercing to rank 0.
    for bad_rank in (-1, 2, True):
        with pytest.raises(ValueError, match="memory_rank_out_of_range"):
            mem._account_rank(bad_rank)

    # Simultaneous per-die peak NEVER sums resident max on rank 0 + staging max on rank 1.
    mem.owner_ranks[(0, "req_a")] = ((100.0, 0.0),)
    mem.staging_owner_ranks[(0, "req_a")] = ((0.0, 80.0),)
    mem._sync_usage()
    assert mem.peak[0][0] == pytest.approx(100.0)
    assert mem.peak_rank[0][0] == [100.0, 80.0]

    # ResidentMemoryReceipt tracks PreparedWeightSpec (persistent vs load_peak_only) and aliases billed ONCE.
    prep_persistent = PreparedWeightSpec(
        allocation_id="kda_composed_f",
        source_tensors=("kda_f_a", "kda_f_b"),
        transform="kda_compose_fb_fa",
        scope="target",
        layer_indices=(0,),
        local_shape=(3072, 7168),
        dtype="bf16",
        layout="ND",
        extra_metadata_bytes=0,
        lifetime="persistent",
    )
    prep_w4a8_load = PreparedWeightSpec(
        allocation_id="w4a8_repack_scratch",
        source_tensors=("expert_w",),
        transform="w4a8_post_load_scale",
        scope="target",
        layer_indices=(0,),
        local_shape=(1024, 1024),
        dtype="int4",
        layout="FRACTAL_NZ",
        extra_metadata_bytes=8192,
        lifetime="load_peak_only",
    )
    mem.register_prepared_weights((prep_persistent, prep_w4a8_load))
    base_grant = ResidentGrantEntry(
        grant_id="shared_buf",
        category="dynamic_grant",
        physical_backing_id="backing_alpha",
        runtime_bytes=65_536,
        allocated_bytes=65_536,
        valid_bytes=65_536,
    )
    alias_grant = ResidentGrantEntry(
        grant_id="alias_buf",
        category="dynamic_grant",
        physical_backing_id="backing_alias_ignored",
        alias_of="shared_buf",
        runtime_bytes=65_536,
        allocated_bytes=65_536,
        valid_bytes=65_536,
    )
    receipt = mem.resident_memory_receipt(
        scenario="tp2_alias_check",
        phase="decode",
        occurrence="step_0",
        extra_grants=(base_grant, alias_grant),
    )
    die0 = receipt.die_receipt(0, 0, 0)
    assert die0.prepared_weight_bytes == KIMI_PREPARED_F_TP4_BYTES
    assert die0.prepared_load_peak_bytes == (1024 * 1024 // 2) + 8192
    assert die0.load_peak_hbm_bytes == die0.simultaneous_peak_hbm_bytes + die0.prepared_load_peak_bytes
    # Aliased buffer billed ONCE (65,536 B, not 131,072 B).
    billed_extra = [g.billed_once_bytes for g in die0.grants if g.grant_id in {"shared_buf", "alias_buf"}]
    assert billed_extra == [65_536, 0]
    assert die0.dynamic_grant_bytes == 65_536

    # Vision accounting toggles honestly between "text_backbone" and "full_checkpoint".
    vlm_case = _load_full_case("kimi-k3.1", tp=1, ep=1, layers=4)
    vlm_mem = Memory(vlm_case, MemoryPolicy(derive_workspace=False))
    rec_text = vlm_mem.resident_memory_receipt(vision_accounting="text_backbone")
    rec_full = vlm_mem.resident_memory_receipt(vision_accounting="full_checkpoint")
    assert rec_text.dies[0].vision_weight_bytes == 0
    assert rec_full.dies[0].vision_weight_bytes > 0
    assert rec_full.dies[0].simultaneous_live_hbm_bytes > rec_text.dies[0].simultaneous_live_hbm_bytes

    # Paired SFA cache_allocation separates allocated vs valid & rank_cache_allocation diverges under CP>1.
    paired_cp = parse_case(compose_case(
        "kimi-k3.1", hardware=SPARSE_OFFLOAD_SKU, tp=8, ep=8, dcp=8, layers=4,
        mtp_proposal_tokens=5, max_model_len=16_384, prompt_tokens=256, output_tokens=16,
    ))
    p_mem = Memory(paired_cp)
    alloc_partial = p_mem.cache_allocation(100)
    assert alloc_partial["allocated"]["payload_bytes_per_stage_rank"][0] > alloc_partial["valid"]["payload_bytes_per_stage_rank"][0]
    rank_views = p_mem.rank_cache_allocation(128)
    valid_by_rank = [v["valid"]["total_bytes_per_stage_rank"][0] for v in rank_views]
    assert valid_by_rank[0] > valid_by_rank[1]

    # Paired draft bytes respect tp (not default tp=1) for GQA draft.
    draft_gqa = DraftSpec(
        kind="dspark", layers=2, hidden_size=32, intermediate_size=64,
        num_heads=8, num_kv_heads=8, head_dim=16, q_width=128, kv_width=128,
        attention_kind="gqa", block_size=5, markov_rank=16, vocab_size=128,
        dcp_sharded=False,
    )
    tp4_block = max(
        paired_cp.runtime.kv_block_size,
        minimum_block_size(paired_cp.model, tp=4, proposals=4),
    )
    tp4_paired = replace(
        paired_cp,
        model=replace(paired_cp.model, auxiliary=AuxiliarySpec(dspark=draft_gqa)),
        runtime=replace(
            paired_cp.runtime,
            kv_block_size=tp4_block,
            parallel=ParallelSpec(tp=4, dp=1, pp=1, cp=2, cp_plan="dcp"),
            dspark=DsparkSpec(4, (1.0,) * 4),
            metadata=(("draft_mode", "dspark"),),
        ),
        machine=replace(paired_cp.machine, dies=paired_cp.machine.dies[:4]),
    )
    tp4_mem = Memory(tp4_paired)
    # 8 kv_heads / tp4 = 2 local heads, kv_dtype=fp8_e4m3 (1 B) -> 2*2*16*1 = 64 B/row * 2 layers * 128 tok = 16,384 B.
    # dcp_sharded=False keeps draft_cp=1 (never divides by target cp=2 or defaults to tp=1).
    assert tp4_mem._paired_draft_bytes(128, padded=False) == 16_384


def test_resize_per_request_verify_width_and_recover_ssm_snapshot_ring_headroom():
    # Per-request verify width preserved across uniform and ragged decode resizes.
    kimi_small = _load_full_case("kimi-k3.1", tp=1, ep=1, layers=2)
    mem = Memory(kimi_small, MemoryPolicy(derive_workspace=False))
    assert mem.resize(0, "req_k0", 128, query_tokens=1) is True
    bytes_q1 = mem.owners[(0, "req_k0")][0]
    assert mem.resize(0, "req_k5", 128, proposal_tokens=5) is True
    bytes_q6 = mem.owners[(0, "req_k5")][0]
    assert bytes_q6 > bytes_q1
    assert mem.active_verify_token_width(0) == 1 + 6

    # Draft-tap workspace placed only on producing-to-consuming PP stages.
    pp2_base = _load_full_case("kimi-k3.1", tp=1, pp=2, ep=1, layers=4, mtp_proposal_tokens=3, num_devices=2)
    draft_tap = replace(pp2_base.model.auxiliary.dspark, target_layer_ids=(3,))
    pp2_tap = replace(
        pp2_base,
        model=replace(pp2_base.model, auxiliary=AuxiliarySpec(dspark=draft_tap)),
        runtime=replace(pp2_base.runtime, pp_layer_partition=(2, 2)),
    )
    tap_mem = Memory(pp2_tap, workspace_tokens=64)
    tap_stages = tap_mem._draft_tap_stage_bytes(64)
    assert tap_stages[0] == 0.0 and tap_stages[1] > 0.0

    # RecoverSSM Snapshot Ring exact bytes & SpaceTimeTradeoffSpec evaluation on full Kimi-K3.1 TP8.
    kimi_full = _load_full_case("kimi-k3.1", tp=8, dp=4, ep=32, num_devices=32)
    kimi_full = replace(kimi_full, model=replace(kimi_full.model, quant=_kimi_mixed_quant()))
    ring_bytes = estimate_recover_ssm_snapshot_ring_bytes(kimi_full, batch_per_dp=16, k_proposals=5)
    assert ring_bytes == KIMI_TP8_B16_K5_RING_BYTES
    bd = recover_ssm_snapshot_ring_breakdown(kimi_full, batch_per_dp=16, k_proposals=5)
    assert bd.linear_layers == 69
    assert bd.verify_width == 6
    assert bd.snapshot_ring_bytes_per_die == KIMI_TP8_B16_K5_RING_BYTES
    assert bd.delta_hbm_vs_recover_ssm_bytes < bd.snapshot_ring_bytes_per_die

    full_mem = Memory(kimi_full, MemoryPolicy(derive_workspace=False))
    headroom = full_mem.free_hbm_headroom_bytes(replica=0)
    assert headroom > ring_bytes
    tradeoff_ok = full_mem.evaluate_space_time_tradeoff(
        mechanism="recover_ssm_snapshot_ring",
        delta_hbm_bytes=ring_bytes,
        delta_step_ms=-0.25,
    )
    assert tradeoff_ok.feasible_under_hbm_and_sla is True
    assert tradeoff_ok.free_hbm_headroom_bytes == headroom

    tradeoff_over = full_mem.evaluate_space_time_tradeoff(
        mechanism="recover_ssm_snapshot_ring",
        delta_hbm_bytes=headroom + 1,
        delta_step_ms=-0.25,
    )
    assert tradeoff_over.feasible_under_hbm_and_sla is False


def test_projected_resident_memory_receipt_cache_matches_memory_cache_bytes_to_the_byte():
    """Projected ResidentMemoryReceipt cache pools include page-group reserve & metadata exactly."""
    paired_case = _load_full_case(
        "kimi-k3.1", tp=8, ep=8, layers=4, mtp_proposal_tokens=3, num_devices=8,
    )
    mem = Memory(paired_case, MemoryPolicy(derive_workspace=False))
    seq_tokens, batch_per_dp, q_tokens = 100, 4, 4
    expected_stage0 = int(round(batch_per_dp * mem.cache_bytes(seq_tokens, padded=True, query_tokens=q_tokens)[0]))
    receipt = mem.resident_memory_receipt(
        scenario="projected_cache_parity",
        phase="decode",
        sequence_tokens=seq_tokens,
        batch_per_dp=batch_per_dp,
        query_tokens=q_tokens,
    )
    die0 = receipt.die_receipt(0, 0, 0)
    projected_total = (
        die0.kv_cache_allocated_bytes
        + die0.index_cache_allocated_bytes
        + die0.recurrent_state_bytes
        + die0.draft_cache_bytes
    )
    assert projected_total == expected_stage0
