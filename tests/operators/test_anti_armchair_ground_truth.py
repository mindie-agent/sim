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


def test_probed_abi_registry_covers_cann_and_non_cann_with_joint_tuples() -> None:
    """Verify sealed A2/A3/A5 CANN + non-CANN catalogs, joint QLI tuples, and M10 guards."""
    from mindie_sim.backends.abi_registry import default_abi_registry
    from mindie_sim.schema.abi import (
        CANN_PACKAGE_ABI_BACKENDS,
        NON_CANN_ABI_BACKENDS,
        AbiQueryKey,
        OperandPortSig,
    )
    from mindie_sim.structure_implementations import operator_birth_certificate

    reg = default_abi_registry()
    assert reg.archs == {"A2", "A3", "A5"}
    for arch in ("A2", "A3", "A5"):
        snap = reg.get_snapshot(arch)
        backends = {e.backend_kind for e in snap.entries}
        assert CANN_PACKAGE_ABI_BACKENDS <= backends
        assert NON_CANN_ABI_BACKENDS <= backends

    assert reg.is_shipping_available("aclnnMoeDistributeDispatchV2", "A3")
    assert reg.is_shipping_available("aclnnMoeDistributeDispatchV3", "A5")
    assert reg.is_shipping_available("aclnnDequantSituQuant", "A3", activation="situ")
    assert not reg.is_shipping_available("aclnnDequantSituQuant", "A5", activation="situ")
    assert reg.is_shipping_available("_kda_recoverssm_verify_kernel", "A5")
    assert reg.is_shipping_available("AttnResFwd", "A3")
    assert reg.is_shipping_available("HcPost", "A5")

    def _port(role: str, dt: str, qm: str = "dense", sd: str = "none") -> OperandPortSig:
        return OperandPortSig(
            role=role, dtype=dt, layout="ND", transposed=False, shape=(16, 128),
            quant_mode=qm, group_shape=(1,), scale_dtype=sd, scale_shape=(),
        )

    def _qli(q_dt: str, w_dt: str, sd: str, qm: str = "per_token_fp8") -> AbiQueryKey:
        return AbiQueryKey(
            soc_arch="A5", sku_id="ascend950dt_9582",
            abi_id="aclnnInnerQuantLightningIndexer", graph_motif="quant_lightning_indexer",
            model_scope="target", draft_quant_mode="none", pr_track="shipping",
            operands=(
                _port("query", q_dt, qm, sd),
                _port("key", q_dt, qm, sd),
                _port("weights", w_dt),
                _port("query_dequant_scale", sd),
                _port("key_dequant_scale", sd),
            ),
            outputs=(_port("sparse_indices", "int32", "none"),),
            rank_placement="local", world_size=1, comm_group_id="local", comm_domain_width=1,
            comm_dtype="none", reduction_kind="none", activation_kind="none", activation_params=(),
            has_bias=False, has_active_mask=False, dispatch_quant_mode=-1, shared_experts_per_rank=0,
            top_k=2048, moe_expert_num=0, qk_rope_head_dim=64, mla_variant="latent_mla",
            sfa_dcp_active=False, pcp_size=1, kv_cache_dtype="bf16", has_kpool_indexer=False,
            dcp_size=1, is_act_and_mul=False, enabled_stack_passes=frozenset(),
        )

    assert reg.verify(_qli("fp8_e4m3fn", "bf16", "fp32")).status == "admitted_shipping"
    assert reg.verify(_qli("hifloat8", "bf16", "fp32")).status == "admitted_shipping"
    assert reg.verify(_qli("int8", "fp16", "fp16", qm="per_token_int8")).status == "admitted_shipping"
    # Cartesian cross-tuple leak MUST be rejected.
    bad = reg.verify(_qli("fp8_e4m3fn", "fp16", "fp32"))
    assert bad.status == "rejected_abi" and "rejected_dtype" in bad.rejection_reason

    cert_post = operator_birth_certificate(
        {"family": "mhc", "recipe": "hc_post_mix", "local_shape": (1, 16), "metadata": {}},
    )
    assert cert_post.native_kinds == ("HcPost",)
    cert_hc = operator_birth_certificate(
        {"family": "mhc", "recipe": "hc_combine", "local_shape": (1, 16), "metadata": {}},
    )
    assert cert_hc.backend_stack == "triton" and "_hc_combine_norm_kernel" in cert_hc.native_kinds
    for fam in ("elementwise", "attention"):
        unk = operator_birth_certificate(
            {"op_id": "u", "family": fam, "recipe": "no_such_recipe", "local_shape": (1, 1), "dtype": "bf16", "metadata": {}},
        )
        assert unk.readiness == "unmodeled" and unk.backend_stack == "unmodeled" and unk.native_kinds == ()

    # P1-6: strided A5 draft (width 14336) & enable_mlapo drop prolog_v3; A3 V4.1 emits AI_CPU SMLA metadata.
    from dataclasses import replace
    from types import SimpleNamespace
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.operators.mla.mla import _select_mla_lowerings
    from mindie_sim.prediction.operators.mla_prolog import resolve_mla_preprocess_mode
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.parallel import ParallelSpec

    assert resolve_mla_preprocess_mode(
        arch="A5", phase="decode", num_tokens=4, q_lora_rank=1536, kv_lora_rank=512,
        qk_rope_head_dim=64, is_draft_model=True, input_size=14336, num_kv_heads=1,
    ) == "strided_kv_adapter"
    c_ds = parse_case(compose_case("deepseek-v3.2", hardware="a5-pod", layers=4, dp=1, tp=4, ep=4, num_devices=4))
    p0 = c_ds.model.layer_plans[0]
    rt_strided = replace(c_ds.runtime, metadata=tuple(c_ds.runtime.metadata) + (("automatic_lowerings", ("mla_prolog",)), ("is_strided_kv_cache", True)))
    prolog_d, *_ = _select_mla_lowerings(
        rt_strided, replace(p0, scope="draft", num_kv_heads=1), ParallelSpec(dp=1, tp=4, ep=4),
        replace(c_ds.model, hidden_size=14336), c_ds.model.quant, (SimpleNamespace(phase="decode", q_len=4),), 4, c_ds.machine,
    )
    assert prolog_d is None
    rt_mlapo = replace(c_ds.runtime, metadata=tuple(c_ds.runtime.metadata) + (("automatic_lowerings", ("mla_prolog",)), ("enable_mlapo", True)))
    prolog_m, *_ = _select_mla_lowerings(
        rt_mlapo, p0, ParallelSpec(dp=1, tp=4, ep=4), c_ds.model, c_ds.model.quant,
        (SimpleNamespace(phase="decode", q_len=4),), 4, c_ds.machine,
    )
    assert prolog_m is None
    c_glm = compose_case("glm-5.3-flash", hardware="a5-pod", layers=4, dp=2, tp=1, ep=2, dcp=2, num_devices=2)
    c_glm["runtime"]["metadata"].pop("va_profile", None)
    c_glm["runtime"]["kernels"]["lowerings"].pop("pool_key_indexer", None)
    with pytest.raises(ModelingError, match="m10_kpool_requires_dcp_eq_1"):
        target_graphs(parse_case(c_glm), phase="decode", tokens=1, context=128, sequences=1, scope="body")
    c_v41_a3 = parse_case(compose_case("deepseek-v4.1-flash", hardware="a3-pod", layers=1, dp=1, tp=4, ep=4, num_devices=4))
    g_v41_a3 = target_graphs(c_v41_a3, phase="decode", tokens=2, context=128, sequences=1, scope="body")[0]
    smla_meta = [op for op in g_v41_a3 if op.recipe == "SparseFlashMlaMetadata"]
    assert len(smla_meta) == 1 and dict(smla_meta[0].metadata).get("task_type") == "AI_CPU"


def test_causal_subgraph_parity_negative_mutations_in_tier0() -> None:
    """Tier-0 coverage of exact flow-ID correlation, proximity ambiguity, and 4 causal diff codes."""
    from mindie_sim.observations.subgraph_parity import (
        PredictedKernelNode,
        SubgraphParityChecker,
        classify_runtime_errors,
        correlate_trace_view_events,
    )

    events = [
        {"ph": "s", "cat": "HostToDevice", "id": "f1", "ts": 10.0, "tid": "0",
         "args": {"stream_id": "0", "cann_api": "aclnnConstantPadNd"}},
        {"ph": "s", "cat": "HostToDevice", "id": "f2", "ts": 11.0, "tid": "0",
         "args": {"stream_id": "0", "cann_api": "aclnnConstantPadNd"}},
        {"ph": "s", "cat": "HostToDevice", "id": "f3", "ts": 20.0, "tid": "1",
         "args": {"stream_id": "comm0", "cann_api": "HcclReduceScatter"}},
        {"ph": "X", "cat": "kernel", "name": "MemSet", "ts": 10.5, "dur": 1.0, "tid": "0",
         "args": {"stream_id": "0", "Task Type": "AI_VECTOR_CORE", "flow_id": "f1"}},
        {"ph": "X", "cat": "kernel", "name": "PadV3", "ts": 12.0, "dur": 1.5, "tid": "0",
         "args": {"stream_id": "0", "Task Type": "AI_VECTOR_CORE", "flow_id": "f2"}},
        {"ph": "X", "cat": "kernel", "name": "hcom_reduceScatter_", "ts": 22.0, "dur": 10.0, "tid": "1",
         "args": {"stream_id": "comm0", "stream_lane": "comm", "Task Type": "COMMUNICATION",
                  "flow_id": "f3", "collective_group": "tp", "comm_ranks": [0, 1, 2, 3]}},
    ]
    dag = correlate_trace_view_events(events, rank_id=0, phase="decode", exec_mode="eager")
    assert dag.exact_causal_attribution_rate == pytest.approx(1.0)
    checker = SubgraphParityChecker()

    p_pad = PredictedKernelNode(node_id="pad", rank_id=0, stream_lane="main", task_type="AI_VECTOR_CORE", api_or_symbol="aclnnConstantPadNd")
    p_rs = PredictedKernelNode(node_id="rs", rank_id=0, stream_lane="comm", task_type="COMMUNICATION", api_or_symbol="hcom_reduceScatter_", expected_kernels=("hcom_reduceScatter_",), predecessors=("pad",), collective_group="tp", comm_ranks=(0, 1, 2, 3))
    checker.compare(signature_id="ok", hardware_sku="Ascend910_9392", device_arch="A3", config_payload={}, observed_dags=(dag,), predicted_nodes=(p_pad, p_rs)).assert_parity()

    rep_ord = checker.compare(signature_id="ord", hardware_sku="Ascend910_9392", device_arch="A3", config_payload={}, observed_dags=(dag,), predicted_nodes=(replace(p_pad, predecessors=("rs",)), p_rs))
    assert any(d.code == "DEPENDENCY_ORDER_MISMATCH" for d in rep_ord.dag_order_diffs)

    rep_grp = checker.compare(signature_id="grp", hardware_sku="Ascend910_9392", device_arch="A3", config_payload={}, observed_dags=(dag,), predicted_nodes=(p_pad, replace(p_rs, collective_group="ep")))
    assert any(d.code == "WRONG_RANK_COLLECTIVE" for d in rep_grp.rank_collective_diffs)

    rep_lane = checker.compare(signature_id="lane", hardware_sku="Ascend910_9392", device_arch="A3", config_payload={}, observed_dags=(dag,), predicted_nodes=(p_pad, replace(p_rs, stream_lane="shared_expert")))
    assert any(d.code == "WRONG_STREAM_LANE" for d in rep_lane.stream_lane_diffs)

    rep_miss = checker.compare(signature_id="miss", hardware_sku="Ascend910_9392", device_arch="A3", config_payload={}, observed_dags=(dag,), predicted_nodes=(PredictedKernelNode(node_id="hc", rank_id=0, stream_lane="main", task_type="MIX_AIC", api_or_symbol="HcPre", expected_kernels=("HcPre",)),))
    assert any(d.code == "MISSING_OR_EXTRA_LAUNCH" for d in rep_miss.launch_diffs)

    so, py = classify_runtime_errors(("libopapi.so: undefined symbol: aclnnFoo", "ModuleNotFoundError: No module named 'ops'"))
    assert "aclnnFoo" in so and "ops" in py


def test_phase3_physics_and_rot_audit_closures() -> None:
    """Verify P1-4/P1-6/P1-8/P1-9/P1-10 and S3/S4/S5 W1 P0/P1 root-cause fixes."""
    from types import SimpleNamespace
    from mindie_sim.catalog.expert_slots import expert_slot_plan
    from mindie_sim.catalog.ops import gemm_op
    from mindie_sim.catalog.quant import QuantSpec
    from mindie_sim.prediction.mechanism import Geometry
    from mindie_sim.prediction.operators import hyper_connections as hc, mc2
    from mindie_sim.prediction.operators.grouped_moe.expected_grouped import ExpectedGroupedMatmul
    from mindie_sim.prediction.operators.hyper_connections import family_attnres
    from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
    from mindie_sim.prediction.operators.mla.generic_latent_attention_views import (
        attention_view_record,
        window_cache_read_by_rank,
    )
    from mindie_sim.prediction.operators.mla_prolog import resolve_mla_preprocess_mode
    from mindie_sim.prediction.operators.moe_routing.expert_slot_work import physical_slot_work
    from mindie_sim.prediction.operators.moe_routing.moe_gating_fusion import (
        GATING_SIGMOID_WAVE_STAGES,
        _fused_gating,
        build_moe_gating_topk,
    )
    from mindie_sim.prediction.operators.paged_mla import mqsmla
    from mindie_sim.prediction.sharding import context_parallel as cp_mod
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.schema.datapath import ArchId, ComputeArchIdentity, DatapathCapabilitySpec
    from mindie_sim.schema.operators import OpSpec
    from mindie_sim.schema.parallel import ParallelSpec
    from mindie_sim.schema.workload import RequestView

    # P1-6: strided_kv_adapter strictly requires input_size not in PROLOG_V3_SUPPORTED_INPUT_SIZES.
    kw_draft = dict(arch="A5", phase="decode", num_tokens=4, q_lora_rank=1536, kv_lora_rank=512,
                    qk_rope_head_dim=64, is_draft_model=True, is_strided_kv_cache=True)
    assert resolve_mla_preprocess_mode(**kw_draft, input_size=7168) == "native_decomposed"
    assert resolve_mla_preprocess_mode(**kw_draft, input_size=14336) == "strided_kv_adapter"

    # P1-8: window_compressed sparse DCP reports exact 32/32 compressed rows per rank.
    m_wc = SimpleNamespace(activation_dtype="bf16", compute_dtype="bf16", quant=QuantSpec("bf16"), kv_cache_formats=(), metadata=())
    p_wc = SimpleNamespace(index=0, scope="target", cache_kind="window_compressed", window=128,
                           cache_row_elements=64, index_topk=32, index_kpool=1, index_kpool_always_select_tail=False,
                           noncausal_attention=False, compress_ratio=4, qk_rope_head_dim=0)
    v_wc = RequestView("r0", 1, 255, False, "decode")
    rec_wc = attention_view_record(v_wc, plan=p_wc, model=m_wc, quant=QuantSpec("bf16"), query_tile=128,
                                   eff_cp=2, block=128, interleave=128, tp=1, dsa=False, topk_present=True, heads=16, latent_dim=64)
    by_r = window_cache_read_by_rank(p_wc, v_wc, rec_wc["parts"], rec_wc["v_sweeps"], rec_wc["v_rows"], 2, 128, 64, m_wc, 128)
    assert by_r is not None and (by_r[0][3], by_r[1][3]) == (32, 32)

    # P1-9: packed_sfa_geometry pools batch variance across views (~193.36, not 210.70).
    p_sfa = SimpleNamespace(index_topk=128, kv_lora_rank=512, qk_rope_head_dim=64, cache_row_elements=576,
                            index_kpool=1, index_kpool_always_select_tail=False)
    g_sfa = cp_mod.packed_sfa_geometry(
        {}, 64, p_sfa, (RequestView("r0", 1, 4095, False, "decode"), RequestView("r1", 1, 4095, False, "decode")),
        2, cache_dtype="int8", interleave=128,
    )
    assert g_sfa is not None and 190.0 < max(
        sum(v for _, v in rank_rows) for rank_rows in g_sfa["per_query_selected_rows_by_rank"]
    ) < 200.0

    # P1-10: sub-unit ExpectedGroupedMatmul preserves exact expected_padded_m=0.16.
    op_tiny = gemm_op(
        op_id="g_tiny", deps=(), stage=0, layer=0, requests=(), m=1, n=64, k=64,
        quant=QuantSpec("bf16"), category="routed_expert", phase="decode", family="grouped_gemm", recipe="grouped_gate_up",
        metadata=(("routing_basis", "probability_moments"), ("matrix_mapping_domain", "aggregate_experts"),
                  ("matrix_weight_bank_groups", 2), ("expected_rows", 0.01), ("expected_active_experts", 0.01)),
    )
    seen_tiny: list = []

    class _CapTiny(SteadyMatmul):
        def lower(self, context, **kwargs):
            seen_tiny.append(dict(context.op.metadata))
            return super().lower(context, **kwargs)

    c_a5 = parse_case(compose_case("kimi-k3.1", "Ascend950DT_9582", layers=2, tp=1, ep=1))
    from mindie_sim.prediction.execution.context import RecipeContext
    geom_a5 = Geometry.from_devices(c_a5.machine.dies[:1])
    _rates = {"cube": 1e12, "hbm": 1e12, "mte2_aic": 1e12, "mte2_aiv": 1e12, "mte3": 1e12, "vector": 1e12, "fixpipe": 1e12, "mte1": 1e12}
    prog_tiny = ExpectedGroupedMatmul(_CapTiny()).lower(RecipeContext(
        op_tiny, {}, {k: k for k in _rates}, _rates, geom_a5.compute_units,
        resource_units=dict(geom_a5.resource_units),
        compute_peaks={"bf16": 1e15, "fp32": 1e15},
        datapath=geom_a5.datapath,
    ))
    assert seen_tiny[-1]["expected_padded_m"] == pytest.approx(0.16)
    assert seen_tiny[-1]["expected_padding_factor"] == pytest.approx(16.0)
    assert prog_tiny is not None

    # S3-001/S3-002: A5 fused_mc2_native_binding & arch22_v2_caller_local rejection on A5.
    assert mc2.fused_mc2_native_binding("bf16", fused_level=3, arch_family="A5")["shared_expert_fused"] is True
    with pytest.raises(ModelingError, match="arch35_fused_mc2_level_unsupported"):
        mc2.fused_mc2_native_binding("bf16", fused_level=1, arch_family="A5")

    # S3-031 & S3-035: cooperative AttnRes write conservation + j=0 bypass exact 128 vector ops.
    coop_m = family_attnres.point_metrics(
        {"tokens_local": 1, "hidden": 4096, "valid_blocks": 4, "mix": True, "fuse_add": False,
         "block_write_idx": -1, "save_materialized": False, "output_norm": False, "optimize_prefill": False,
         "tiling_mode": "cooperative"},
        253952, 64, arch_family="A5",
    )
    assert sum(c["write_bytes"] for c in coop_m["cores"]) == coop_m["write_bytes"] == 8192
    op_j0 = OpSpec("ar_j0", "attention_residual", "decode", 0, 0, (), (1, 64), (1, 64), (1, 64), (1, 64),
                   dtype="bf16", recipe="attn_res_fwd",
                   metadata=(("attn_res_view", {"tokens_local": 1, "hidden": 64, "valid_blocks": 0, "bank_capacity": 8,
                                                "block_token_stride": 512, "fuse_add": False, "output_norm": False,
                                                "block_write_idx": -1}),))
    assert hc.build(op_j0, geom_a5).work.resources["vector"] == pytest.approx(128.0)

    # S4-046 Option B: symmetric 2-rank top-1 preserves physical per-rank means (50, 50)
    # and reports exact cohort barrier critical_rank_weight_read_bytes() == 100.0;
    # multi-token expected total_rows == 1.0 is NOT treated as mutually exclusive.
    sp2 = expert_slot_plan(None, SimpleNamespace(num_experts=2, index=0), ParallelSpec(tp=2, ep=2))
    w2 = physical_slot_work(slot_plan=sp2, logical_inclusions=(0.5, 0.5), global_tokens=1, packed_weight_bytes=100.0)
    assert w2.weight_read_bytes(0) == pytest.approx(50.0) and w2.weight_read_bytes(1) == pytest.approx(50.0)
    assert w2.critical_rank_weight_read_bytes() == pytest.approx(100.0)
    w2_multi = physical_slot_work(slot_plan=sp2, logical_inclusions=(0.125, 0.125), global_tokens=4, packed_weight_bytes=100.0)
    assert w2_multi.weight_read_bytes(0) == pytest.approx(w2_multi.weight_read_bytes(1))
    assert w2_multi.critical_rank_weight_read_bytes() < 2.0 * w2_multi.weight_read_bytes(0)

    # S4-050: A5 non-noaux_tc sigmoid router uses GATING_SIGMOID_WAVE_STAGES (not softmax_reduce).
    sig_layer = SimpleNamespace(num_experts=64, top_k=4, router_mode="sigmoid", router_norm_topk=True,
                                router_groups=1, router_group_topk=0, routed_scaling_factor=1.0, index=0)
    sig_op = _fused_gating(prefix="r", plan=sig_layer, logits_input="lg",
                           members=(SimpleNamespace(vector_ops=256, recipe="score"),), stage=0, requests=(),
                           tokens=4, mode="decode", weight_dtype="fp32", weight_bytes=4, hash_table=False, noaux_tc=False)
    assert build_moe_gating_topk(sig_op, geom_a5).metadata["row_program"]["wave_stages"] == GATING_SIGMOID_WAVE_STAGES

    # S4-069: MQSMLA fails closed on unset preset for FP8 quant_mode.
    mqs_op = OpSpec("mqs", "attention", "decode", 0, 0, (), (1, 64, 512), (1, 64, 512), (1, 64, 512), (1, 64, 512),
                    dtype="bf16", recipe=mqsmla.RECIPE,
                    metadata=(("mqsmla_geometry", {"heads": 64, "width": 512, "metadata_support_fd": False,
                                                   "addressing": "scalar", "sequences": ((1, 128),), "window": 128,
                                                   "ratio": 4, "topk": 64, "kv_quant_mode": mqsmla.KV_QUANT_FP8}),))
    no_arch_geom = Geometry(
        compute_units=32, resource_units=dict(geom_a5.resource_units),
        buffers=dict(geom_a5.buffers), buffer_units=dict(geom_a5.buffer_units),
        compute_arch=None, datapath=DatapathCapabilitySpec(
            compute_arch=ComputeArchIdentity(ArchId(""), None), aic_count=32, aiv_count=64,
            aic_aiv_pairs=(), usable_paths=(), matrix_output_paths=frozenset(), ub_peer_edges=(), stars=None,
        ),
    )
    with pytest.raises(ModelingError, match="mqsmla_fp8_fp4_requires_arch35"):
        mqsmla.build(mqs_op, no_arch_geom)
