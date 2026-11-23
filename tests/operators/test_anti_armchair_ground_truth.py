"""Ground-truth operator physics invariants: eager chains, ABI guards, and monotonicity.

[Role] Verify that unfused PyTorch eager operator decompositions match real
  ``torch_npu`` kernel sequences, fusion ABI admission checks fail closed
  without crashing non-aligned callers, and 4-track waterfalls remain monotonic.
[Invariants] Uses ``REPRESENTATIVE_SKUS`` from ``tests._support.axes`` (zero raw
  SKU literals), bounded layer samples, and pure in-memory execution.
[Timeouts & Locks] Deterministic in-memory graph/prediction checks; no I/O.
[Side-Effect Policy] Pure assertions; zero filesystem or environment mutation.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim import compare_feature_tracks, compose_case
from mindie_sim.prediction.efficiency.analysis import parse_case
from mindie_sim.prediction.operators.mla_prolog import family_mla_prolog
from mindie_sim.prediction.operators.mx_quant_matrix import shared_w4a8
from mindie_sim.prediction.operators.recurrent import linear_attention_pipeline, recurrent
from mindie_sim.prediction.operators.vector import gated_activation, vector_pipeline
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure import export_structure
from tests._support.axes import REPRESENTATIVE_SKUS

pytestmark = pytest.mark.tier1

SAMPLE_LAYERS = 8
TP_WIDTH = 8
DECODE_TOKENS = 1
DECODE_CONTEXT = 2048
DECODE_SEQUENCES = 4
MONOTONIC_TOL_MS = 1e-5
EAGER_GATED_NORM_OP_COUNT = 11
SITU_EAGER_BASE_KERNELS = 11
SITU_EAGER_CLIPPED_KERNELS = 14
SITU_TEST_ROWS = 4
SITU_TEST_WIDTH = 128
SITU_BETA = 4.0
SITU_LINEAR_BETA = 10.0
ALIGNED_M = 4
ALIGNED_N = 128
ALIGNED_K = 256
UNALIGNED_N = 96
UNALIGNED_K = 160
SCALE_PAIR_DIVISOR = 64
SCALE_PAIR_LANES = 2
ROPE_SLOT_DIM = 64
ZERO_ROPE_DIM = 0

SITU_BASE_NATIVE_KINDS = (
    "Cast", "Slice", "RealDiv", "Tanh", "Mul",
    "Slice", "Sigmoid", "Mul", "Slice", "Mul", "Cast",
)
SITU_CLIPPED_NATIVE_KINDS = (
    "Cast", "Slice", "RealDiv", "Tanh", "Mul",
    "Slice", "Sigmoid", "Mul", "Slice",
    "RealDiv", "Tanh", "Mul", "Mul", "Cast",
)


def _assert_waterfall_monotonic(result: dict) -> None:
    wf = result["waterfall"]
    t_eager = float(wf["raw_eager"]["step_ms"])
    t_merged = float(wf["current_merged"]["step_ms"])
    t_pr = float(wf["current_unmerged_pr"]["step_ms"])
    t_ideal = float(wf["expected_ideal"]["step_ms"])
    t_roof = float(wf["hardware_roofline"]["physical_ideal_1p0x_ms"])
    assert t_eager + MONOTONIC_TOL_MS >= t_merged
    assert t_merged + MONOTONIC_TOL_MS >= t_pr
    assert t_pr + MONOTONIC_TOL_MS >= t_ideal
    assert t_ideal + MONOTONIC_TOL_MS >= t_roof


def test_indexer_layer_norm_raw_eager_retains_fp32_wrapper_casts() -> None:
    """BF16 indexer LayerNorm retains upstream FP32 wrapper casts in both raw_eager and merged."""
    sku = REPRESENTATIVE_SKUS["a3"]
    for track in ("raw_eager", "merged"):
        case = compose_case(
            "glm-5.2", hardware=sku, feature_track=track,
            tp=TP_WIDTH, num_devices=TP_WIDTH, layers=SAMPLE_LAYERS,
            indexer_kv_dtype="bf16",
        )
        rep = export_structure(
            case, phase="decode", tokens=DECODE_TOKENS,
            context=DECODE_CONTEXT, sequences=DECODE_SEQUENCES, scope="body",
        )
        recipes = {node["operator"]["recipe"] for node in rep["nodes"]}
        assert "index_norm_input_fp32" in recipes, f"{track} dropped index_norm_input_fp32"
        assert "index_norm_output_cast" in recipes, f"{track} dropped index_norm_output_cast"


def test_gated_rmsnorm_eager_differentiates_gdn_swish_from_kda_sigmoid() -> None:
    """Unfused 11-op FusedRMSNormGated uses Swish (5 ops) on GDN and Sigmoid (4 ops) on KDA."""
    sku = REPRESENTATIVE_SKUS["a5"]
    gdn_case = compose_case(
        "qwen3.5-35b-a3b", hardware=sku, feature_track="raw_eager",
        tp=TP_WIDTH, num_devices=TP_WIDTH, layers=SAMPLE_LAYERS,
    )
    kda_case = compose_case(
        "glm-5.3-flash", hardware=sku, feature_track="raw_eager",
        tp=TP_WIDTH, num_devices=TP_WIDTH, layers=SAMPLE_LAYERS,
    )
    gdn_rep = export_structure(
        gdn_case, phase="decode", tokens=DECODE_TOKENS,
        context=DECODE_CONTEXT, sequences=DECODE_SEQUENCES, scope="body",
    )
    kda_rep = export_structure(
        kda_case, phase="decode", tokens=DECODE_TOKENS,
        context=DECODE_CONTEXT, sequences=DECODE_SEQUENCES, scope="body",
    )
    gdn_groups = [sg for sg in gdn_rep["subgraphs"] if sg["role"] == "gated_rmsnorm"]
    kda_groups = [sg for sg in kda_rep["subgraphs"] if sg["role"] == "gated_rmsnorm"]
    assert gdn_groups and all(len(sg["node_ids"]) == EAGER_GATED_NORM_OP_COUNT for sg in gdn_groups)
    assert kda_groups and all(len(sg["node_ids"]) == EAGER_GATED_NORM_OP_COUNT for sg in kda_groups)

    gdn_by_id = {node["node_id"]: node["operator"] for node in gdn_rep["nodes"]}
    kda_by_id = {node["node_id"]: node["operator"] for node in kda_rep["nodes"]}
    gdn_act = [
        gdn_by_id[nid] for nid in gdn_groups[0]["node_ids"]
        if gdn_by_id[nid]["recipe"] == "linear_output_gate_silu"
    ]
    kda_act = [
        kda_by_id[nid] for nid in kda_groups[0]["node_ids"]
        if kda_by_id[nid]["recipe"] == "linear_output_gate_sigmoid"
    ]
    assert len(gdn_act) == 1
    assert gdn_act[0]["metadata"]["native_kind"] == "Swish"
    assert len(kda_act) == 1
    assert kda_act[0]["metadata"]["native_kind"] == "Sigmoid"


@pytest.mark.parametrize("gen", ("a3", "a5"))
def test_gdn_decode_raw_eager_decomposes_conv_and_gating_monotonically(gen: str) -> None:
    """GDN decode raw_eager lowers real torch conv1d + gating chains and satisfies monotonicity."""
    sku = REPRESENTATIVE_SKUS[gen]
    eager_case = compose_case(
        "qwen3.5-27b", hardware=sku, feature_track="raw_eager",
        tp=TP_WIDTH, num_devices=TP_WIDTH, layers=SAMPLE_LAYERS,
    )
    rep = export_structure(
        eager_case, phase="decode", tokens=DECODE_TOKENS,
        context=DECODE_CONTEXT, sequences=DECODE_SEQUENCES, scope="body",
    )
    recipes = {node["operator"]["recipe"] for node in rep["nodes"]}
    expected_eager_recipes = {
        "conv_state_cat", "conv_state_update", "short_convolution", "conv_silu",
        "gate_alpha_cast_fp32", "gate_alpha_add_bias", "gate_alpha_softplus",
        "gate_beta_cast_fp32", "gate_beta_sigmoid", "gate_beta_cast_out",
        "gate_alog_exp", "gate_alog_neg", "delta_gates",
    }
    assert expected_eager_recipes <= recipes

    waterfall = compare_feature_tracks(
        "qwen3.5-27b", hardware=sku, phase="decode",
        tokens=DECODE_TOKENS, context=DECODE_CONTEXT, sequences=DECODE_SEQUENCES,
        ablate_prs=False, tp=TP_WIDTH, num_devices=TP_WIDTH, layers=SAMPLE_LAYERS,
    )
    _assert_waterfall_monotonic(waterfall)


def test_situ_eager_chain_matches_fp32_kernel_inventory() -> None:
    """Unfused SituAndMul emits exact 11-kernel (base) and 14-kernel (clipped) FP32 chains."""
    case = parse_case(compose_case(
        "kimi-k3.1", hardware=REPRESENTATIVE_SKUS["a5"],
        tp=TP_WIDTH, num_devices=TP_WIDTH, layers=SAMPLE_LAYERS,
    ))
    plan = case.model.layer_plans[0]
    base_plan = replace(plan, activation_situ_beta=SITU_BETA, activation_situ_linear_beta=None)
    clipped_plan = replace(
        plan, activation_situ_beta=SITU_BETA, activation_situ_linear_beta=SITU_LINEAR_BETA,
    )

    base_ops = gated_activation.situ_eager_chain(
        plan=base_plan, op_id="s0.l0.situ", deps=("s0.l0.gate_up",),
        rows=SITU_TEST_ROWS, width=SITU_TEST_WIDTH, dtype="bf16",
        stage=0, requests=("r0",), phase="decode",
    )
    assert len(base_ops) == SITU_EAGER_BASE_KERNELS
    assert tuple(dict(op.metadata)["native_kind"] for op in base_ops) == SITU_BASE_NATIVE_KINDS
    assert base_ops[-1].op_id == "s0.l0.situ"
    assert base_ops[0].dtype == "fp32" and base_ops[-1].dtype == "bf16"

    clipped_ops = gated_activation.situ_eager_chain(
        plan=clipped_plan, op_id="s0.l0.situ", deps=("s0.l0.gate_up",),
        rows=SITU_TEST_ROWS, width=SITU_TEST_WIDTH, dtype="bf16",
        stage=0, requests=("r0",), phase="decode",
    )
    assert len(clipped_ops) == SITU_EAGER_CLIPPED_KERNELS
    assert tuple(dict(op.metadata)["native_kind"] for op in clipped_ops) == SITU_CLIPPED_NATIVE_KINDS
    assert clipped_ops[-1].op_id == "s0.l0.situ"


def _make_shared_w4a8_op(m: int, n: int, k: int) -> OpSpec:
    k_tail = (k + SCALE_PAIR_DIVISOR - 1) // SCALE_PAIR_DIVISOR
    meta = {
        **shared_w4a8.SOURCE_INTERFACE,
        "ffn_category": "shared_expert",
        "quant_category": "shared_expert",
        "matrix_groups": 1,
        "weight_layout": "FRACTAL_NZ",
        "activation_scale_shape": (m, k_tail, SCALE_PAIR_LANES),
        "weight_scale_shape": (n, k_tail, SCALE_PAIR_LANES),
    }
    return OpSpec(
        op_id="s0.l0.shared_gate_up", family="gate_up_projection", phase="decode",
        stage=0, layer=0, requests=("r0",),
        semantic_shape=(m, n, k), local_shape=(m, n, k),
        padded_shape=(m, n, k), launch_shape=(m, n, k),
        flops=0.0, vector_ops=0.0, hbm_bytes=0.0, mte2_bytes=0.0, mte3_bytes=0.0,
        deps=(), recipe="shared_gate_up", dtype="bf16",
        metadata=tuple(sorted(meta.items())),
    )


def test_shared_w4a8_alignment_guard_and_unaligned_shared_expert_fallback() -> None:
    """shared_w4a8 admits only k%64==0 & n%64==0 so unaligned shared experts fall back cleanly."""
    assert shared_w4a8.matches_source_interface(_make_shared_w4a8_op(ALIGNED_M, ALIGNED_N, ALIGNED_K))
    assert not shared_w4a8.matches_source_interface(
        _make_shared_w4a8_op(ALIGNED_M, UNALIGNED_N, ALIGNED_K)
    )
    assert not shared_w4a8.matches_source_interface(
        _make_shared_w4a8_op(ALIGNED_M, ALIGNED_N, UNALIGNED_K)
    )
    waterfall = compare_feature_tracks(
        "gemma-4-26b-a4b", hardware=REPRESENTATIVE_SKUS["a5"], phase="decode",
        tokens=DECODE_TOKENS, context=DECODE_CONTEXT, sequences=DECODE_SEQUENCES,
        ablate_prs=False, tp=TP_WIDTH, num_devices=TP_WIDTH, layers=SAMPLE_LAYERS,
    )
    _assert_waterfall_monotonic(waterfall)


def test_lowrank_hc_fused_ub_resident_executes_without_matrix_phase_crash() -> None:
    """Fused UB-resident low-rank HC preserves its matrix-capable plan on non-RegBase devices."""
    waterfall = compare_feature_tracks(
        "qwen3.8-flash-next", hardware=REPRESENTATIVE_SKUS["a3"], phase="decode",
        tokens=DECODE_TOKENS, context=DECODE_CONTEXT, sequences=DECODE_SEQUENCES,
        ablate_prs=False, tp=TP_WIDTH, num_devices=TP_WIDTH, layers=SAMPLE_LAYERS,
    )
    _assert_waterfall_monotonic(waterfall)


def test_mla_prolog_rejects_zero_rope_dim_while_admitting_nope_with_rope_slot() -> None:
    """MLA prolog rejects qk_rope_head_dim==0 while admitting do_rope=False with rope slot=64."""
    case = parse_case(compose_case(
        "kimi-k3.1", hardware=REPRESENTATIVE_SKUS["a5"],
        tp=TP_WIDTH, num_devices=TP_WIDTH, layers=SAMPLE_LAYERS,
    ))
    mla_plan = next(lp for lp in case.model.layer_plans if lp.attention == "mla")
    nope_with_slot = replace(mla_plan, mla_use_nope=True, qk_rope_head_dim=ROPE_SLOT_DIM)
    assert not nope_with_slot.mla_do_rope
    bf16_quant = replace(
        case.model.quant, name="bf16", kv_dtype="bf16",
        weight_format="bf16", activation_format="bf16", category_overrides=(),
    )
    family_mla_prolog.validate(
        plan=nope_with_slot, parallel=case.runtime.parallel, model=case.model,
        quant=bf16_quant, lowering_passes=case.runtime.lowering_passes,
    )
    zero_rope = replace(mla_plan, mla_use_nope=True, qk_rope_head_dim=ZERO_ROPE_DIM)
    with pytest.raises(ModelingError, match="mla_prolog_v3_declared_operand_domain_unsupported"):
        family_mla_prolog.validate(
            plan=zero_rope, parallel=case.runtime.parallel, model=case.model,
            quant=bf16_quant, lowering_passes=case.runtime.lowering_passes,
        )


def test_unused_pipeline_fraction_constants_are_removed() -> None:
    """Dead hardcoded share fractions are purged from vector and linear-attention pipelines."""
    for removed in ("STREAM_LOAD_SHARE", "STREAM_STORE_SHARE"):
        assert not hasattr(vector_pipeline, removed)
    for removed in ("LOAD_READ_SHARE", "LOAD_WY_SHARE", "LOAD_INVERSE_SHARE", "LOAD_OUTPUT_SHARE"):
        assert not hasattr(linear_attention_pipeline, removed)
    assert recurrent.EAGER_NORM_SILU_OPS > recurrent.EAGER_NORM_SIGMOID_OPS
