"""Product tests for prepared-OpSpec export precision, VLM ViT ledger parity, and kernel identities.

[Role] Verify that export bundles carry prepared execution ``OpSpec`` precision
  and ``matrix_flops`` into ``l_star_lean_us``, honor target-graph native-baseline
  rollback so ``After`` never regresses past ``Before``, maintain exact ``0.000%``
  VLM ViT residency and ``qwen4_exp`` stride-reserve parity between
  ``MemoryCalculatorLedger`` and ``DieResidentReceipt``, support ``MXFP8`` KV,
  GQA static ``C8``, and ``W4A16`` ledger formats, and emit shipping vLLM-Ascend
  kernel identities for local ``route_unpack``, non-MLA GBSA, A3 C8 paired QSFA,
  fused MRoPE, and ``group=32``-only ``DynamicMxQuant``.
[Invariants] Pure in-memory ``tier0`` tests with zero host pollution, zero raw
  SKU string literals, and exact algebraic/physical assertions.
[Timeouts & Locks] In-memory analytical & step evaluations only; no subprocesses,
  network calls, sleeps, or locks.
[Side-Effect Policy] Read-only over repository state.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.cache_format import dense_c8_kv_planes
from mindie_sim.export import (
    BACKEND_CANN,
    BACKEND_HCCL_MC2,
    BACKEND_TORCH_NPU,
    BACKEND_TRITON,
    build_export_bundle,
    compose_export_case,
    compute_memory_quant_ledger,
    evaluate_weight_storage_bytes,
)
from mindie_sim.export import bundle as bundle_mod
from mindie_sim.export.kernel_catalog import (
    resolve_ascend_kernel_identity,
    resolve_operator_display_dtype,
)
from mindie_sim.export.memory_ledger import (
    C8_CACHE_WIDTH_BYTES,
    GQA_STATIC_C8_CACHE_WIDTH_BYTES,
    MXFP8_CACHE_WIDTH_BYTES,
    _cache_dtype_override_ratio,
    _normalize_cache_dtype,
    _resolve_ledger_workspace_tokens,
)
from mindie_sim.export.models import BF16_BYTES, MXFP_GROUP_SIZE
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec

pytestmark = pytest.mark.tier0

PARITY_ABS_TOL_BYTES = 1.0
SUPERLUMINAL_EPS_US = 1e-6
BLOCKWISE_FP8_GROUP = 128
GQA_TEST_SEQ_LEN = 4096
VLM_TEST_SEQ_LEN = 4224
TEST_CONCURRENCY = 4
PREFILL_TEST_TOKENS = 256
DECODE_TEST_CTX = 512
DECODE_TEST_SEQS = 2
W4A16_TEST_ROWS = 16
W4A16_TEST_COLS = 64
W4A16_EXPECTED_TUPLE = (512, 128, 0, 640)
W4A16_EP8_EXPECTED_GIB = 78.3334
W4A16_EP8_TOL_GIB = 0.05
A3_EP_DEGREE = 16
A3_TP_DEGREE = 8
A3_DP_DEGREE = 4
GIB_DIVISOR = float(1024 ** 3)
GEMMA4_SMALL_CTX = 128


def _make_op(
    op_id: str,
    family: str,
    recipe: str,
    *,
    shape: tuple[int, ...] = (),
    dtype: str = "bf16",
    flops: float = 0.0,
    metadata: tuple[tuple[str, object], ...] = (),
) -> OpSpec:
    return OpSpec(
        op_id=op_id, family=family, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape,
        recipe=recipe, dtype=dtype, flops=flops, metadata=metadata,
    )


def test_prepared_op_spec_resolves_execution_precision_and_eliminates_superluminal_l_star():
    raw, _ = compose_export_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"],
        topology_preset="a5_pod", tp=1, ep=1, dp=1,
    )
    raw = {
        **raw,
        "model": {**raw["model"], "layers": TEST_CONCURRENCY, "layer_plans": raw["model"]["layer_plans"][:TEST_CONCURRENCY]},
    }
    case = parse_case(raw)
    bf16_peak = float(case.machine.dies[0].peak("bf16"))
    fp8_peak = float(case.machine.dies[0].peak("fp8_e4m3"))

    prefill_bundle = build_export_bundle(
        raw,
        request=StepRequest(
            phase="prefill", tokens=PREFILL_TEST_TOKENS, context=0,
            sequences=DECODE_TEST_SEQS, scope="body",
        ),
    )
    superluminal = [
        row for row in prefill_bundle.operators
        if row.dur_us + SUPERLUMINAL_EPS_US < row.l_star_lean_us
    ]
    assert superluminal == []
    by_id = {row.op_id: row for row in prefill_bundle.operators}
    for oid, expected_peak in (
        ("s0.l0.linear_qkvz", bf16_peak),
        ("s0.l3.qkv", fp8_peak),
        ("s0.l0.expert_up", fp8_peak),
    ):
        row = by_id[oid]
        assert row.cube_peak_flops_per_s == pytest.approx(expected_peak, rel=1e-6)
        assert row.dur_us + SUPERLUMINAL_EPS_US >= row.l_star_lean_us

    # Persisted prepared_ops from EvaluationContext take precedence over pre-prep catalog specs.
    raw_op = _make_op(
        "s0.l0.synthetic_gemm", "gemm", "linear",
        shape=(PREFILL_TEST_TOKENS, W4A16_TEST_COLS, W4A16_TEST_COLS),
        dtype="bf16", flops=1.0e12,
    )
    prepared_op = replace(
        raw_op,
        dtype="fp8_e4m3",
        matrix_flops=(("fp8_e4m3", 1.0e12),),
        metadata=(("execution_compute_format", "fp8_e4m3"),),
    )
    fake_entry = SimpleNamespace(replica=0, ops=(raw_op,))
    fake_eval = SimpleNamespace(prepared_ops={(0, raw_op.op_id): prepared_op})
    resolved = bundle_mod._resolve_prepared_op_specs((fake_entry,), fake_eval)
    assert resolved[(0, raw_op.op_id)].dtype == "fp8_e4m3"
    assert resolved[(0, raw_op.op_id)].matrix_flops == (("fp8_e4m3", 1.0e12),)
    assert bundle_mod._resolved_execution_dtype(resolved[(0, raw_op.op_id)]) == "fp8_e4m3"

    peaks = bundle_mod._die_peaks_for_stage(case, 0, 0)
    demands, used_peak = bundle_mod._op_lean_resource_demands_ns(
        flops=1.0e12, vector_ops=0.0, hbm_read=0.0, hbm_write=0.0,
        link_bytes=0.0, dtype="bf16", peaks=peaks,
        matrix_flops=prepared_op.matrix_flops,
    )
    assert used_peak == pytest.approx(fp8_peak, rel=1e-6)
    assert demands[0] > 0


def test_target_envelope_honors_native_baseline_rollback_and_non_regression(monkeypatch):
    raw, _ = compose_export_case(
        "qwen3-32b", REPRESENTATIVE_SKUS["a5"],
        topology_preset="a5_pod", tp=1, ep=1, dp=1,
    )
    req = StepRequest(
        phase="decode", tokens=1, context=DECODE_TEST_CTX,
        sequences=DECODE_TEST_SEQS, scope="body",
    )
    baseline = build_export_bundle(raw, request=req, apply_target_envelope=False)
    base_span = baseline.step_span_us
    base_ids = [row.op_id for row in baseline.operators]

    # Case 1: explicit reverted_to_native_baseline flag retains exact native Before rows & span.
    def _fake_reverted_solver(nodes, demands_ns, *args, **kwargs):
        inflated = [
            replace(n, dur_us=n.dur_us * 2.0, exposed_dur_us=n.exposed_dur_us * 2.0)
            for n in nodes
        ]
        return SimpleNamespace(
            nodes=tuple(inflated),
            demands_ns=tuple(demands_ns),
            archetype_spans_us={},
            synthesis_specs=(),
            semantic_audit_triggers=(),
            rewrite_plan_sha256="rollback_sha",
            verification_passed=True,
            applied_edit_receipts=(),
            stream_core_budget_summary="",
            moe_route={},
            co_design_opportunities=(),
            stage_waterfall=({"stage": "S1", "step_span_us": base_span * 2.0, "delta_from_prev_us": base_span},),
            reverted_to_native_baseline=True,
        )

    monkeypatch.setattr(bundle_mod, "solve_target_graph", _fake_reverted_solver)
    rolled_back = build_export_bundle(raw, request=req, apply_target_envelope=True)
    assert rolled_back.step_span_us == pytest.approx(base_span, rel=1e-6)
    assert [row.op_id for row in rolled_back.operators] == base_ids
    assert all(float(w.get("delta_from_prev_us", 0.0)) == 0.0 for w in rolled_back.stage_waterfall)

    # Case 2: regressed solver output without explicit flag still triggers non-regression rollback.
    def _fake_regressed_solver(nodes, demands_ns, *args, **kwargs):
        inflated = [
            replace(n, dur_us=n.dur_us * 1.5, exposed_dur_us=n.exposed_dur_us * 1.5)
            for n in nodes
        ]
        return SimpleNamespace(
            nodes=tuple(inflated),
            demands_ns=tuple(demands_ns),
            archetype_spans_us={},
            synthesis_specs=(),
            semantic_audit_triggers=(),
            rewrite_plan_sha256="regressed_sha",
            verification_passed=True,
            applied_edit_receipts=(),
            stream_core_budget_summary="",
            moe_route={},
            co_design_opportunities=(),
            stage_waterfall=(),
            reverted_to_native_baseline=False,
        )

    monkeypatch.setattr(bundle_mod, "solve_target_graph", _fake_regressed_solver)
    guarded = build_export_bundle(raw, request=req, apply_target_envelope=True)
    assert guarded.step_span_us <= base_span + SUPERLUMINAL_EPS_US
    assert [row.op_id for row in guarded.operators] == base_ids


@pytest.mark.parametrize(
    ("model_name", "parallel_kwargs"),
    [
        ("qwen3.8-flash-next", {"tp": 2, "ep": 2, "dp": 1}),
        ("kimi-k3.1", {"tp": 8, "ep": 32, "dp": 4, "dcp": 8}),
        ("deepseek-v4.1-flash", {"tp": 8, "ep": 32, "dp": 4}),
    ],
)
def test_vlm_vit_residency_parity_matches_die_resident_receipt_on_text_prompts(
    model_name, parallel_kwargs,
):
    raw, _ = compose_export_case(
        model_name, REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod", **parallel_kwargs,
    )
    case = parse_case(raw)
    ledger = compute_memory_quant_ledger(case, seq_len=VLM_TEST_SEQ_LEN, concurrency=TEST_CONCURRENCY)
    ws_tok = _resolve_ledger_workspace_tokens(
        case, phase="decode", resolved_seq=VLM_TEST_SEQ_LEN, resolved_conc=TEST_CONCURRENCY,
    )
    mem = Memory(case, workspace_tokens=ws_tok)
    receipt = mem.resident_memory_receipt(
        phase="decode", sequence_tokens=VLM_TEST_SEQ_LEN, batch_per_dp=TEST_CONCURRENCY,
    )
    die0 = receipt.die_receipt(0, 0, 0)

    assert ledger.include_vision_tower is True
    assert ledger.vision_weight_bytes_per_die > 0.0
    assert die0.vision_weight_bytes > 0
    assert abs(ledger.stage_budgets[0].total_hbm_bytes - die0.simultaneous_live_hbm_bytes) <= PARITY_ABS_TOL_BYTES

    # Explicit text_backbone / include_vision_tower=False opt-out stays in 0.000% parity too.
    ledger_no_vit = compute_memory_quant_ledger(
        case, seq_len=VLM_TEST_SEQ_LEN, concurrency=TEST_CONCURRENCY, include_vision_tower=False,
    )
    rec_no_vit = mem.resident_memory_receipt(
        phase="decode", sequence_tokens=VLM_TEST_SEQ_LEN, batch_per_dp=TEST_CONCURRENCY,
        vision_accounting="text_backbone",
    )
    die0_no_vit = rec_no_vit.die_receipt(0, 0, 0)
    assert ledger_no_vit.include_vision_tower is False
    assert die0_no_vit.vision_weight_bytes == 0
    assert abs(
        ledger_no_vit.stage_budgets[0].total_hbm_bytes - die0_no_vit.simultaneous_live_hbm_bytes
    ) <= PARITY_ABS_TOL_BYTES

    raw_tb = {
        **raw,
        "runtime": {
            **raw["runtime"],
            "metadata": {**dict(raw["runtime"].get("metadata") or {}), "vision_accounting": "text_backbone"},
        },
    }
    ledger_tb = compute_memory_quant_ledger(
        parse_case(raw_tb), seq_len=VLM_TEST_SEQ_LEN, concurrency=TEST_CONCURRENCY,
    )
    assert ledger_tb.include_vision_tower is False


def test_mxfp8_gqa_static_c8_w4a16_and_qwen4_exp_stride_reserve():
    # 1. W4A16 display & storage formula (never mislabeled as W4A8).
    assert evaluate_weight_storage_bytes(W4A16_TEST_ROWS, W4A16_TEST_COLS, "W4A16") == W4A16_EXPECTED_TUPLE
    raw_k2, _ = compose_export_case(
        "kimi-k2-thinking", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod", tp=8, ep=32, dp=4,
    )
    led_k2 = compute_memory_quant_ledger(parse_case(raw_k2), seq_len=GQA_TEST_SEQ_LEN, concurrency=TEST_CONCURRENCY)
    assert led_k2.weight_quant == "W4A16"
    assert "W4A16" in led_k2.weight_quant_display and "W4A8" not in led_k2.weight_quant_display
    assert {r.quant_format for r in led_k2.weight_rows if r.category == "routed_expert"} == {"W4A16"}

    # 2. MXFP8 KV (33/32 B/elem) & GQA static C8 (1.0 B/elem + static channel scale, no block-128 inflation).
    assert MXFP8_CACHE_WIDTH_BYTES == pytest.approx(1.0 + 1.0 / float(MXFP_GROUP_SIZE), rel=1e-12)
    assert GQA_STATIC_C8_CACHE_WIDTH_BYTES == 1.0
    raw_q3, _ = compose_export_case(
        "qwen3-32b", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        tp=4, ep=1, dp=1, kv_cache_dtype="bf16",
    )
    case_q3 = parse_case(raw_q3)
    led_bf16 = compute_memory_quant_ledger(case_q3, seq_len=GQA_TEST_SEQ_LEN, concurrency=TEST_CONCURRENCY, cache_dtype="BF16")
    led_mxfp8 = compute_memory_quant_ledger(
        case_q3, seq_len=GQA_TEST_SEQ_LEN, concurrency=TEST_CONCURRENCY,
        cache_dtype="QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL",
    )
    led_c8 = compute_memory_quant_ledger(case_q3, seq_len=GQA_TEST_SEQ_LEN, concurrency=TEST_CONCURRENCY, cache_dtype="C8")

    gqa_bf16 = next(r for r in led_bf16.cache_rows if r.component == "gqa_kv")
    gqa_mxfp8 = next(r for r in led_mxfp8.cache_rows if r.component == "gqa_kv")
    gqa_c8 = next(r for r in led_c8.cache_rows if r.component == "gqa_kv")
    static_c8 = next(r for r in led_c8.cache_rows if r.component == "c8_static_channel_scale")

    assert led_mxfp8.cache_dtype == "MXFP8"
    assert gqa_mxfp8.per_request_bytes == pytest.approx(
        gqa_bf16.per_request_bytes * (MXFP8_CACHE_WIDTH_BYTES / float(BF16_BYTES)), rel=1e-9,
    )
    assert gqa_c8.per_request_bytes == pytest.approx(
        gqa_bf16.per_request_bytes * (GQA_STATIC_C8_CACHE_WIDTH_BYTES / float(BF16_BYTES)), rel=1e-9,
    )
    gqa_lp = case_q3.model.layer_plans[0]
    local_kv = local_kv_heads(int(gqa_lp.num_kv_heads or case_q3.model.kv_heads), case_q3.runtime.parallel.tp)
    per_layer_static = sum(
        pl.bytes_per_unit
        for pl in dense_c8_kv_planes(local_kv, led_c8.block_size, int(gqa_lp.head_dim))
        if pl.allocation_frequency == "checkpoint"
    )
    expected_static = float(per_layer_static * len(case_q3.model.layer_plans))
    assert static_c8.scales_with_tokens is False
    assert static_c8.batch_total_bytes == pytest.approx(expected_static, rel=1e-9)

    # MXFP8 kv_cache_tag normalizes to MXFP8 (not FP8); MLA C8 retains block-128 scale width.
    mxfp_tagged = replace(
        case_q3.model,
        metadata=tuple(sorted({**dict(case_q3.model.metadata), "kv_cache_tag": "QK_MXFP8_DYNAMIC_V_MXFP8_PER_CHANNEL"}.items())),
    )
    assert _normalize_cache_dtype(mxfp_tagged) == "MXFP8"
    assert _cache_dtype_override_ratio(case_q3.model, "C8", component="mla_kv") == pytest.approx(
        C8_CACHE_WIDTH_BYTES / float(BF16_BYTES), rel=1e-9,
    )

    # 3. qwen4_exp_pool_breakdown stride reserve wired into page_group_reserve in capacity + export ledger.
    raw_q38, _ = compose_export_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod", tp=1, ep=1, dp=1,
    )
    case_q38 = parse_case(raw_q38)
    ws_tok = _resolve_ledger_workspace_tokens(
        case_q38, phase="decode", resolved_seq=VLM_TEST_SEQ_LEN, resolved_conc=TEST_CONCURRENCY,
    )
    mem_q38 = Memory(case_q38, workspace_tokens=ws_tok)
    stride_reserves = mem_q38._unified_pool_stride_reserve_by_stage(VLM_TEST_SEQ_LEN)
    assert stride_reserves[0] > 0.0
    projected = mem_q38._projected_stage_cache_pools(VLM_TEST_SEQ_LEN, TEST_CONCURRENCY, None)
    assert projected is not None
    assert projected[0]["page_group_reserve"] >= int(round(TEST_CONCURRENCY * stride_reserves[0]))

    led_q38 = compute_memory_quant_ledger(case_q38, seq_len=VLM_TEST_SEQ_LEN, concurrency=TEST_CONCURRENCY)
    pgr_row = next(r for r in led_q38.cache_rows if r.component == "page_group_reserve")
    assert pgr_row.per_request_bytes >= stride_reserves[0] - PARITY_ABS_TOL_BYTES
    rec_q38 = mem_q38.resident_memory_receipt(
        phase="decode", sequence_tokens=VLM_TEST_SEQ_LEN, batch_per_dp=TEST_CONCURRENCY,
    )
    assert abs(
        led_q38.stage_budgets[0].total_hbm_bytes - rec_q38.die_receipt(0, 0, 0).simultaneous_live_hbm_bytes
    ) <= PARITY_ABS_TOL_BYTES


def test_shipping_kernel_identities_route_unpack_sparse_attn_mrope_and_mx_quant():
    req = StepRequest(
        phase="decode", tokens=1, context=DECODE_TEST_CTX, sequences=DECODE_TEST_SEQS, scope="body",
    )

    # 1. qwen3.8-flash-next @ A5 (EP=1): local route_unpack -> npu_moe_token_unpermute (torch_npu);
    #    sparse GQA attention -> aclnnGenericBlockSparseAttention; group=32 MX quant -> DynamicMxQuant.
    raw_q38, _ = compose_export_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod", tp=1, ep=1, dp=1,
    )
    b_q38 = build_export_bundle(raw_q38, request=req)
    q38_ops = {o.op_id: o for o in b_q38.operators}
    assert q38_ops["s0.l0.unpack"].ascend_kernel == "npu_moe_token_unpermute"
    assert q38_ops["s0.l0.unpack"].kernel_backend == BACKEND_TORCH_NPU
    assert q38_ops["s0.l3.attention"].ascend_kernel == "aclnnGenericBlockSparseAttention"
    assert q38_ops["s0.l0.expert_up.quantize"].ascend_kernel == "DynamicMxQuant"

    # Explicit FinalizeRouting binding is preserved even at EP=1, while default EP=1 stays npu_moe_token_unpermute.
    case_q38 = parse_case(raw_q38)
    unperm_spec = _make_op(
        "s0.l0.unpack", "routing", "route_unpack",
        metadata=(("native_kind", "npu_moe_token_unpermute"),),
    )
    unperm_desc = resolve_ascend_kernel_identity(
        unperm_spec.op_id, unperm_spec, unperm_spec.family, unperm_spec.recipe,
        (), "bf16", case_q38, case_q38.model.layer_plans[0], (),
    )
    assert unperm_desc.ascend_kernel == "npu_moe_token_unpermute"
    assert unperm_desc.kernel_backend == BACKEND_TORCH_NPU

    # 2. deepseek-v3.2 @ A5: group=128 blockwise FP8 quant -> DynamicQuant (NOT DynamicMxQuant);
    #    generic alltoall expert_combine -> HcclAlltoAllV; local route_unpack -> aclnnFinalizeRoutingV2;
    #    sparse MLA stays SparseFlashMla on A5.
    raw_ds, _ = compose_export_case(
        "deepseek-v3.2", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        tp=8, ep=32, dp=4, dcp=8,
    )
    b_ds = build_export_bundle(raw_ds, request=req)
    ds_ops = {o.op_id: o for o in b_ds.operators}
    assert ds_ops["s0.l0.q_a.quantize"].ascend_kernel == "DynamicQuant"
    assert "MxQuant" not in ds_ops["s0.l0.q_a.quantize"].ascend_kernel
    assert ds_ops["s0.l3.dispatch"].ascend_kernel == "HcclAlltoAllV"
    assert ds_ops["s0.l3.combine"].ascend_kernel == "HcclAlltoAllV"
    assert ds_ops["s0.l3.combine"].kernel_backend == BACKEND_HCCL_MC2
    assert ds_ops["s0.l3.unpack"].ascend_kernel == "aclnnFinalizeRoutingV2"
    assert ds_ops["s0.l3.unpack"].kernel_backend == BACKEND_CANN
    assert ds_ops["s0.l3.latent_attention_batch"].ascend_kernel == "SparseFlashMla"

    # Direct group=32 vs group=128 discrimination for fused norm+quant families too.
    case_ds = parse_case(raw_ds)
    for fam, mx_kern in (
        ("quantize", "DynamicMxQuant"),
        ("rmsnorm_quant", "RmsNormDynamicMxQuant"),
        ("add_rmsnorm_quant", "AddRmsNormDynamicMxQuant"),
    ):
        spec_32 = _make_op(
            f"s0.l0.test_{fam}_32", fam, fam,
            metadata=(
                ("output_format", "fp8_e4m3"),
                ("activation_scale_format", "fp8_e8m0"),
                ("activation_group_size", MXFP_GROUP_SIZE),
            ),
        )
        spec_128 = _make_op(
            f"s0.l0.test_{fam}_128", fam, fam,
            metadata=(
                ("output_format", "fp8_e4m3"),
                ("activation_scale_format", "fp8_e8m0"),
                ("activation_group_size", BLOCKWISE_FP8_GROUP),
            ),
        )
        d32 = resolve_ascend_kernel_identity(spec_32.op_id, spec_32, fam, fam, (), "bf16", case_ds, None, ())
        d128 = resolve_ascend_kernel_identity(spec_128.op_id, spec_128, fam, fam, (), "bf16", case_ds, None, ())
        assert d32.ascend_kernel == mx_kern
        assert "Quant" in d128.ascend_kernel and "MxQuant" not in d128.ascend_kernel

    # 3. kimi-k3.1 @ A3 C8: preserve npu_kv_quant_sparse_flash_attention_vllm (NOT SparseFlashMla),
    #    generic alltoall -> HcclAlltoAllV, plain BF16 SiTU -> SiTU, while A5 MC2 keeps MoeDistributeCombineV3.
    raw_k3_a3, _ = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a3"], topology_preset="a3_superpod",
        tp=8, ep=32, dp=4, dcp=8, quant="w4a8", kv_cache_dtype="int8",
    )
    b_k3_a3 = build_export_bundle(raw_k3_a3, request=req)
    k3_a3_ops = {o.op_id: o for o in b_k3_a3.operators}
    assert k3_a3_ops["s0.l3.latent_attention_batch"].ascend_kernel == "npu_kv_quant_sparse_flash_attention_vllm"
    assert k3_a3_ops["s0.l3.combine"].ascend_kernel == "HcclAlltoAllV"
    assert k3_a3_ops["s0.l3.unpack"].ascend_kernel == "aclnnFinalizeRoutingV2"
    assert k3_a3_ops["s0.l1.expert_act"].ascend_kernel == "SiTU"
    assert k3_a3_ops["s0.l1.shared.situ"].ascend_kernel == "SiTU"

    raw_k3_a5, _ = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        tp=8, ep=32, dp=4, dcp=8,
    )
    b_k3_a5 = build_export_bundle(raw_k3_a5, request=req)
    k3_a5_ops = {o.op_id: o for o in b_k3_a5.operators}
    assert k3_a5_ops["s0.l1.unpack"].ascend_kernel == "MoeDistributeCombineV3"
    assert k3_a5_ops["s0.l1.unpack"].kernel_backend == BACKEND_HCCL_MC2

    # 4. qwen3.5-35b-a3b @ A3: split_qkv_rmsnorm_mrope_fused resolves BEFORE InplacePartialRotaryMul.
    raw_q35, _ = compose_export_case(
        "qwen3.5-35b-a3b", REPRESENTATIVE_SKUS["a3"], topology_preset="a3_superpod",
        tp=2, ep=2, dp=1,
    )
    b_q35 = build_export_bundle(raw_q35, request=req)
    q35_ops = {o.op_id: o for o in b_q35.operators}
    assert q35_ops["s0.l3.rope"].ascend_kernel == "split_qkv_rmsnorm_mrope_kernel"
    assert q35_ops["s0.l3.rope"].kernel_backend == BACKEND_TRITON
    assert q35_ops["s0.l3.rope"].canonical_op_key == "split_qkv_rmsnorm_mrope_fused"

    # 5. MXFP8 megamoe_arch35 surfaces compute_dtype (fp8_e4m3 / int8) instead of output bf16.
    megamoe_fp8 = _make_op(
        "s0.l1.unpack", "grouped_gemm", "megamoe_arch35", dtype="bf16",
        metadata=(("compute_dtype", "fp8_e4m3"), ("weight_format", "fp8_e4m3")),
    )
    assert resolve_operator_display_dtype(
        megamoe_fp8.op_id, megamoe_fp8, megamoe_fp8.family, megamoe_fp8.recipe,
        "bf16", case_ds, None,
    ) == "fp8_e4m3"
    megamoe_int8 = _make_op(
        "s0.l1.unpack", "grouped_gemm", "dispatch_ffn_combine", dtype="bf16",
        metadata=(("compute_format", "int8"), ("weight_format", "int8")),
    )
    assert resolve_operator_display_dtype(
        megamoe_int8.op_id, megamoe_int8, megamoe_int8.family, megamoe_int8.recipe,
        "bf16", case_ds, None,
    ) == "int8"


def test_sparse_mla_a3_situ_wo_a_w4a16_parity_and_gemma4_decode_read_conservation():
    req = StepRequest(
        phase="decode", tokens=1, context=DECODE_TEST_CTX, sequences=DECODE_TEST_SEQS, scope="body",
    )

    # 1. A3 sparse MLA never emits A5-only SparseFlashMla: C8 -> npu_kv_quant_sparse_flash_attention_vllm,
    #    BF16 -> npu_sparse_flash_attention.
    raw_g53_c8, _ = compose_export_case(
        "glm-5.3", REPRESENTATIVE_SKUS["a3"], topology_preset="a3_superpod",
        tp=1, ep=A3_EP_DEGREE, dp=A3_EP_DEGREE,
    )
    b_g53_c8 = build_export_bundle(raw_g53_c8, request=req)
    g53_c8_ops = {o.op_id: o for o in b_g53_c8.operators}
    assert g53_c8_ops["s0.l0.latent_attention_batch"].ascend_kernel == "npu_kv_quant_sparse_flash_attention_vllm"
    assert g53_c8_ops["s0.l3.dispatch"].ascend_kernel == "HcclAlltoAllV"
    assert g53_c8_ops["s0.l3.combine"].ascend_kernel == "HcclAlltoAllV"

    case_g53_bf16 = parse_case(raw_g53_c8)
    case_g53_bf16 = replace(
        case_g53_bf16,
        model=replace(case_g53_bf16.model, kv_dtype="bf16", quant=replace(case_g53_bf16.model.quant, kv_dtype="bf16")),
    )
    bf16_sparse_op = _make_op("s0.l0.latent_attention_batch", "attention", "latent_attention_batch")
    bf16_sparse_desc = resolve_ascend_kernel_identity(
        bf16_sparse_op.op_id, bf16_sparse_op, "attention", "latent_attention_batch",
        (), "bf16", case_g53_bf16, case_g53_bf16.model.layer_plans[0], (),
    )
    assert bf16_sparse_desc.ascend_kernel == "npu_sparse_flash_attention"

    # 2. SiTU vs SituMxQuant discrimination on A5 (MX output -> SituMxQuant; BF16 -> SiTU; A3 -> SiTU).
    raw_v4, _ = compose_export_case(
        "deepseek-v4-flash-0731", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod", tp=2, ep=2, dp=1,
    )
    case_v4 = parse_case(raw_v4)
    situ_bf16 = _make_op("s0.l1.expert_act", "situ", "grouped_situ", dtype="bf16", metadata=(("output_format", "bf16"),))
    situ_mx = _make_op("s0.l1.expert_act", "situ", "grouped_situ", dtype="fp8_e4m3", metadata=(("output_format", "fp8_e4m3"),))
    assert resolve_ascend_kernel_identity(
        situ_bf16.op_id, situ_bf16, "situ", "grouped_situ", (), "bf16", case_v4, None, (),
    ).ascend_kernel == "SiTU"
    assert resolve_ascend_kernel_identity(
        situ_mx.op_id, situ_mx, "situ", "grouped_situ", (), "fp8_e4m3", case_v4, None, (),
    ).ascend_kernel == "SituMxQuant"
    assert resolve_ascend_kernel_identity(
        situ_mx.op_id, situ_mx, "situ", "grouped_situ", (), "fp8_e4m3", case_g53_bf16, None, (),
    ).ascend_kernel == "SiTU"

    # 3. DeepSeek-V4 grouped_output_a (wo_a): quantized -> npu_transpose_quant_batchmatmul, BF16 -> TransposeBatchMatMul.
    b_v4 = build_export_bundle(raw_v4, request=req)
    v4_ops = {o.op_id: o for o in b_v4.operators}
    assert v4_ops["s0.l0.grouped_output_a"].ascend_kernel == "npu_transpose_quant_batchmatmul"
    wo_a_bf16 = _make_op(
        "s0.l0.grouped_output_a", "gemm", "grouped_output_a", dtype="bf16",
        metadata=(("activation_format", "bf16"), ("weight_format", "bf16")),
    )
    assert resolve_ascend_kernel_identity(
        wo_a_bf16.op_id, wo_a_bf16, "gemm", "grouped_output_a", (), "bf16", case_v4, None, (),
    ).ascend_kernel == "TransposeBatchMatMul"

    # 4. W4A16 EP=8 kimi-k2-thinking includes BF16 weight_offset (+7.38 GiB/die) with exact ledger parity.
    raw_k2_ep8, _ = compose_export_case(
        "kimi-k2-thinking", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        tp=A3_TP_DEGREE, ep=A3_TP_DEGREE, dp=1,
    )
    case_k2_ep8 = parse_case(raw_k2_ep8)
    mem_k2 = Memory(case_k2_ep8)
    led_k2 = compute_memory_quant_ledger(case_k2_ep8, seq_len=DECODE_TEST_CTX, concurrency=TEST_CONCURRENCY)
    w_gib = mem_k2.weights[0] / GIB_DIVISOR
    led_w_gib = (led_k2.stage_budgets[0].weight_payload_bytes + led_k2.stage_budgets[0].weight_scale_aux_bytes) / GIB_DIVISOR
    assert abs(w_gib - W4A16_EP8_EXPECTED_GIB) < W4A16_EP8_TOL_GIB
    assert abs(w_gib - led_w_gib) <= PARITY_ABS_TOL_BYTES / GIB_DIVISOR

    # 5. gemma-4-31b per-layer BF16 overrides conserve total decode HBM reads >= receipt weight bytes.
    raw_g4, _ = compose_export_case(
        "gemma-4-31b", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod", tp=1, ep=1, dp=1,
    )
    case_g4 = parse_case(raw_g4)
    receipt_w_gib = Memory(case_g4).weights[0] / GIB_DIVISOR
    b_g4 = build_export_bundle(
        raw_g4, request=StepRequest(phase="decode", tokens=1, context=GEMMA4_SMALL_CTX, sequences=1, scope="model"),
    )
    bundle_read_gib = sum(o.hbm_read_bytes * o.repeat_count for o in b_g4.operators) / GIB_DIVISOR
    assert bundle_read_gib >= receipt_w_gib
