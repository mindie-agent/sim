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
from mindie_sim.prediction.results import EngineStepReceipt, StageReceipt, StepRequest
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import KernelBinding, OpSpec

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
    base_span_ns = baseline.step_span_ns
    base_lstar_ns = min(baseline.step_l_star_ns, base_span_ns)
    base_ids = [row.op_id for row in baseline.operators]

    # Regression: folded display rows (repeat_count != 1) are rejected loudly by _certify_step_lower_bound.
    folded_row = replace(baseline.operators[0], repeat_count=2)
    with pytest.raises(ModelingError, match="cannot_certify_folded_display_graph"):
        bundle_mod._certify_step_lower_bound(
            [folded_row], [(100, 0, 0, 0, 0)], step_span_ns=base_span_ns,
        )

    def _make_reverted_receipt(stage3_span_ns: int, status: str, reason: str, plan_sha: str) -> EngineStepReceipt:
        s0 = StageReceipt(
            stage_id="before_merged", stack_track="shipping",
            start_tick_ns=0, end_tick_ns=base_span_ns, span_ns=base_span_ns,
            l_star_ns=base_lstar_ns, l_star_valid=True,
            execution_key_sha256="exec_s0", plan_sha256="plan_s0",
            idle_or_wait_ns=base_span_ns, status="committed",
            primitive_table_digest="pt_digest",
        )
        s1 = StageReceipt(
            stage_id="stage1_ops_config", stack_track="shipping",
            start_tick_ns=0, end_tick_ns=base_span_ns, span_ns=base_span_ns,
            l_star_ns=base_lstar_ns, l_star_valid=True,
            execution_key_sha256="", plan_sha256="",
            idle_or_wait_ns=base_span_ns, status="skipped",
            rejection_reason="same_config",
        )
        s2 = StageReceipt(
            stage_id="stage2_unmerged_pr", stack_track="proposed",
            start_tick_ns=0, end_tick_ns=base_span_ns, span_ns=base_span_ns,
            l_star_ns=base_lstar_ns, l_star_valid=True,
            execution_key_sha256="", plan_sha256="",
            idle_or_wait_ns=base_span_ns, status="skipped",
            rejection_reason="no_pr_deltas",
        )
        s3 = StageReceipt(
            stage_id="stage3_target_after", stack_track="target",
            start_tick_ns=0, end_tick_ns=stage3_span_ns, span_ns=stage3_span_ns,
            l_star_ns=base_lstar_ns, l_star_valid=(status == "committed"),
            execution_key_sha256="exec_s3" if status == "committed" else "",
            plan_sha256=plan_sha,
            idle_or_wait_ns=stage3_span_ns, status=status,
            primitive_table_digest="pt_digest" if status == "committed" else "",
            rejection_reason=reason,
        )
        return EngineStepReceipt(
            scenario_digest="scen_rollback", primitive_table_digest="pt_digest",
            comparison_policy="monotonic_optimization", stages=(s0, s1, s2, s3),
        )

    # Case 1: stage3 rejected_by_policy retains exact native Before rows, span_ns, and 0-ns stage deltas.
    def _fake_reverted_solver(*args, **kwargs):
        return _make_reverted_receipt(base_span_ns * 2, "rejected_by_policy", "regressive_target_step", "rollback_sha")

    monkeypatch.setattr(bundle_mod, "solve_target_step_receipt", _fake_reverted_solver)
    rolled_back = build_export_bundle(raw, request=req, apply_target_envelope=True)
    assert rolled_back.reverted_to_native_baseline is True
    assert rolled_back.step_span_ns == base_span_ns
    assert rolled_back.step_span_us == pytest.approx(base_span, rel=1e-6)
    assert [row.op_id for row in rolled_back.operators] == base_ids
    assert len(rolled_back.stage_waterfall) == 4
    assert all(int(w["delta_ns"]) == 0 for w in rolled_back.stage_waterfall)
    assert rolled_back.stage_waterfall[-1]["span_ns"] == base_span_ns

    # Case 2: stage3 committed with identical plan_sha256 still marks reverted_to_native_baseline=True.
    def _fake_unchanged_solver(*args, **kwargs):
        return _make_reverted_receipt(base_span_ns, "committed", "", "plan_s0")

    monkeypatch.setattr(bundle_mod, "solve_target_step_receipt", _fake_unchanged_solver)
    guarded = build_export_bundle(raw, request=req, apply_target_envelope=True)
    assert guarded.reverted_to_native_baseline is True
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
    pgr_rows = [r for r in led_q38.cache_rows if r.component in {"page_group_reserve", "page_group_reserve_seq"}]
    assert any(r.component == "page_group_reserve" for r in pgr_rows)
    assert sum(r.per_request_bytes for r in pgr_rows) >= stride_reserves[0] - PARITY_ABS_TOL_BYTES
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

    # 1. qwen3.8-flash-next @ A5 (EP=1): read-only projection from ProbedAbiRegistry & birth certificates.
    raw_q38, _ = compose_export_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod", tp=1, ep=1, dp=1,
    )
    b_q38 = build_export_bundle(raw_q38, request=req)
    q38_ops = {o.op_id: o for o in b_q38.operators}
    assert q38_ops["s0.l0.unpack"].ascend_kernel == "aclnnMoeTokenUnpermute"
    assert q38_ops["s0.l0.unpack"].kernel_backend == BACKEND_CANN
    assert q38_ops["s0.l0.unpack"].is_shipping_abi is True
    assert q38_ops["s0.l3.attention"].ascend_kernel == "aclnnBlockSparseAttention"
    assert q38_ops["s0.l0.expert_up.quantize"].ascend_kernel == "DynamicMxQuant"

    # Explicit native_kind on OpSpec is projected read-only (never heuristically rewritten).
    case_q38 = parse_case(raw_q38)
    unperm_spec = _make_op(
        "s0.l0.unpack", "routing", "route_unpack",
        metadata=(("native_kind", "npu_moe_token_unpermute"),),
    )
    unperm_desc = resolve_ascend_kernel_identity(
        unperm_spec.op_id, unperm_spec, unperm_spec.family, unperm_spec.recipe,
        (), "bf16", case_q38, case_q38.model.layer_plans[0], (),
        native_binding={"entry": "npu_moe_token_unpermute", "backend": "torch_npu"},
    )
    assert unperm_desc.ascend_kernel == "npu_moe_token_unpermute"
    assert unperm_desc.kernel_backend == BACKEND_TORCH_NPU

    # 2. deepseek-v3.2 @ A5: read-only projection emits canonical probed CANN/HCCL ABI symbols.
    raw_ds, _ = compose_export_case(
        "deepseek-v3.2", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        tp=8, ep=32, dp=4, dcp=8,
    )
    b_ds = build_export_bundle(raw_ds, request=req)
    ds_ops = {o.op_id: o for o in b_ds.operators}
    assert ds_ops["s0.l3.dispatch"].ascend_kernel == "HcclAlltoAll"
    assert ds_ops["s0.l3.combine"].ascend_kernel == "HcclAlltoAll"
    assert ds_ops["s0.l3.combine"].kernel_backend == BACKEND_HCCL_MC2
    assert ds_ops["s0.l3.unpack"].ascend_kernel == "aclnnMoeTokenUnpermute"
    assert ds_ops["s0.l3.unpack"].kernel_backend == BACKEND_CANN
    assert ds_ops["s0.l3.latent_attention_batch"].ascend_kernel == "aclnnSparseFlashAttention"

    # Direct typed KernelBinding projection preserves explicit symbol & backend read-only.
    case_ds = parse_case(raw_ds)
    for fam, kern_sym in (
        ("quantize", "DynamicMxQuant"),
        ("rmsnorm_quant", "RmsNormDynamicMxQuant"),
        ("add_rmsnorm_quant", "AddRmsNormDynamicMxQuant"),
    ):
        spec_kb = replace(
            _make_op(f"s0.l0.test_{fam}", fam, fam),
            kernel_binding=KernelBinding(kernel_id=kern_sym, backend="CANN", arch_family="dav_3510", readiness="shipping"),
        )
        d_kb = resolve_ascend_kernel_identity(spec_kb.op_id, spec_kb, fam, fam, (), "bf16", case_ds, None, ())
        assert d_kb.ascend_kernel in {kern_sym, f"aclnn{kern_sym}"}

    # 3. kimi-k3.1 @ A3 C8 vs A5: A3 projects shipping AllToAll/Unpermute + QSFA ABI; A5 projects V3 MC2 ABI.
    raw_k3_a3, _ = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a3"], topology_preset="a3_superpod",
        tp=8, ep=32, dp=4, dcp=8, quant="w4a8", kv_cache_dtype="int8",
    )
    b_k3_a3 = build_export_bundle(raw_k3_a3, request=req)
    k3_a3_ops = {o.op_id: o for o in b_k3_a3.operators}
    assert k3_a3_ops["s0.l3.latent_attention_batch"].ascend_kernel == "aclnnKvQuantSparseFlashAttentionVllm"
    assert k3_a3_ops["s0.l3.dispatch"].ascend_kernel == "HcclAlltoAll"
    assert k3_a3_ops["s0.l3.combine"].ascend_kernel == "HcclAlltoAll"
    assert k3_a3_ops["s0.l3.unpack"].ascend_kernel == "aclnnMoeTokenUnpermute"
    assert k3_a3_ops["s0.l1.expert_act"].ascend_kernel == "aclnnDequantSituQuant"
    assert k3_a3_ops["s0.l1.shared.situ"].ascend_kernel == "aclnnDequantSituQuant"

    raw_k3_a5, _ = compose_export_case(
        "kimi-k3.1", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        tp=8, ep=32, dp=4, dcp=8,
    )
    b_k3_a5 = build_export_bundle(raw_k3_a5, request=req)
    k3_a5_ops = {o.op_id: o for o in b_k3_a5.operators}
    assert k3_a5_ops["s0.l1.unpack"].ascend_kernel == "aclnnMoeDistributeCombineV3"
    assert k3_a5_ops["s0.l1.unpack"].kernel_backend == BACKEND_HCCL_MC2

    # 4. qwen3.5-35b-a3b @ A3: read-only projection preserves canonical_op_key and non-empty kernel.
    raw_q35, _ = compose_export_case(
        "qwen3.5-35b-a3b", REPRESENTATIVE_SKUS["a3"], topology_preset="a3_superpod",
        tp=2, ep=2, dp=1,
    )
    b_q35 = build_export_bundle(raw_q35, request=req)
    q35_ops = {o.op_id: o for o in b_q35.operators}
    assert q35_ops["s0.l3.rope"].ascend_kernel in {
        "split_qkv_rmsnorm_mrope_kernel", "InplacePartialRotaryMul", "aclnnInplacePartialRotaryMul",
    }
    assert "split_qkv_rmsnorm_mrope_fused" in q35_ops["s0.l3.rope"].canonical_op_key
    mrope_desc = resolve_ascend_kernel_identity(
        "s0.l3.rope", None, "rotary", "split_qkv_rmsnorm_mrope_fused", (), "bf16", parse_case(raw_q35), None, (),
        native_binding={"entry": "split_qkv_rmsnorm_mrope_kernel", "backend": "triton"},
    )
    assert mrope_desc.ascend_kernel == "split_qkv_rmsnorm_mrope_kernel"
    assert mrope_desc.kernel_backend == BACKEND_TRITON and mrope_desc.is_shipping_abi is True

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

    # 1. A3 sparse MLA projects verified probed ABI symbol (never A5-only SparseFlashMla).
    raw_g53_c8, _ = compose_export_case(
        "glm-5.3", REPRESENTATIVE_SKUS["a3"], topology_preset="a3_superpod",
        tp=1, ep=A3_EP_DEGREE, dp=A3_EP_DEGREE,
    )
    b_g53_c8 = build_export_bundle(raw_g53_c8, request=req)
    g53_c8_ops = {o.op_id: o for o in b_g53_c8.operators}
    assert g53_c8_ops["s0.l0.latent_attention_batch"].ascend_kernel == "aclnnSparseFlashAttention"
    assert g53_c8_ops["s0.l0.latent_attention_batch"].is_shipping_abi is True
    assert g53_c8_ops["s0.l3.unpack"].ascend_kernel == "aclnnMoeTokenUnpermute"
    assert g53_c8_ops["s0.l3.unpack"].is_shipping_abi is True

    # 2. DeepSeek-V4 grouped_output_a obeys CANN 9.3 BF16/FP16-only aclnnTransposeBatchMatMul ABI:
    #    BF16 -> shipping aclnnTransposeBatchMatMul (is_shipping_abi=True);
    #    block-128 FP8 -> honest non-shipping abi_precision_unsupported (is_shipping_abi=False).
    raw_v4_bf16, _ = compose_export_case(
        "deepseek-v4-flash-0731", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod",
        tp=2, ep=2, dp=1, quant="none",
    )
    b_v4_bf16 = build_export_bundle(raw_v4_bf16, request=req)
    v4_bf16_ops = {o.op_id: o for o in b_v4_bf16.operators}
    assert v4_bf16_ops["s0.l0.grouped_output_a"].ascend_kernel == "aclnnTransposeBatchMatMul"
    assert v4_bf16_ops["s0.l0.grouped_output_a"].is_shipping_abi is True

    raw_v4_fp8, _ = compose_export_case(
        "deepseek-v4-flash-0731", REPRESENTATIVE_SKUS["a5"], topology_preset="a5_pod", tp=2, ep=2, dp=1,
    )
    b_v4_fp8 = build_export_bundle(raw_v4_fp8, request=req)
    v4_fp8_ops = {o.op_id: o for o in b_v4_fp8.operators}
    assert v4_fp8_ops["s0.l0.grouped_output_a"].is_shipping_abi is False
    assert v4_fp8_ops["s0.l0.grouped_output_a"].needs_new_kernel is True
    assert "abi_precision_unsupported" in v4_fp8_ops["s0.l0.grouped_output_a"].new_kernel_spec

    # 3. W4A16 EP=8 kimi-k2-thinking includes BF16 weight_offset (+7.38 GiB/die) with exact ledger parity.
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


def test_memory_ledger_exact_integer_bijection_against_resident_receipt():
    from mindie_sim.export.ooxml import Worksheet
    from mindie_sim.export.xlsx_memory import populate_memory_sheet

    raw, _ = compose_export_case(
        "glm-5.3-flash", REPRESENTATIVE_SKUS["a3"],
        dp=1, tp=A3_TP_DEGREE, ep=A3_TP_DEGREE, quant="w4a8_dynamic", kv_cache_dtype="int8",
    )
    case = parse_case(raw)
    led = compute_memory_quant_ledger(case, seq_len=GQA_TEST_SEQ_LEN, concurrency=TEST_CONCURRENCY)
    ws_tok = _resolve_ledger_workspace_tokens(case, phase="decode", resolved_seq=GQA_TEST_SEQ_LEN, resolved_conc=TEST_CONCURRENCY)
    die = Memory(case, workspace_tokens=ws_tok).resident_memory_receipt(
        phase="decode", sequence_tokens=GQA_TEST_SEQ_LEN, batch_per_dp=TEST_CONCURRENCY,
    ).die_receipt(0, 0, 0)
    sb = led.stage_budgets[0]
    assert int(sb.weight_payload_bytes + sb.weight_scale_aux_bytes + sb.engram_ple_hbm_bytes) == (
        die.checkpoint_weight_bytes + die.prepared_weight_bytes + die.vision_weight_bytes
    )
    assert int(sb.cache_bytes) == (
        die.kv_cache_allocated_bytes + die.index_cache_allocated_bytes + die.recurrent_state_bytes + die.draft_cache_bytes
    )
    assert (int(sb.workspace_peak_bytes), int(sb.reserve_bytes), int(sb.total_hbm_bytes), int(sb.usable_hbm_budget_bytes)) == (
        die.workspace_bytes, die.reserve_bytes, die.simultaneous_live_hbm_bytes, die.usable_hbm_bytes
    )
    ws = Worksheet("Memory_Quant_Calculator")
    populate_memory_sheet(ws, SimpleNamespace(memory_ledger=led))  # type: ignore[arg-type]
    assert any(dv.sqref == "T4" and dv.formula1 == '"BF16,C8"' for dv in ws._validations)
    assert float(ws._cells[(24, 20)].value) * GIB_DIVISOR == pytest.approx(die.usable_hbm_bytes, abs=1.0)
    assert not any("uncalibrated" in str(c.value).lower() for c in ws._cells.values() if c.value is not None)
    # Caller-supplied ResidentMemoryReceipt with custom rank_token_owners & PreparedWeightSpec
    # preserves cache row conservation, peak weight stage, bottleneck replica/peak headroom, live grants,
    # PP non-peak stage formulas, Sheet 3 seq-len scaling, canonical W4A8 parity, and FAQuant C8 alias.
    from mindie_sim.catalog.quant import quant_preset
    from mindie_sim.compose import compose_case
    from mindie_sim.export.xlsx_memory import (
        _populate_cache_block, _populate_stage_budget_block, _stage_cache_formula,
    )
    from mindie_sim.prediction.capacity import MemoryPolicy, ResidentGrantEntry
    from mindie_sim.schema.stack import PreparedWeightSpec

    case_q32 = parse_case(compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a3"], dp=1, tp=2, pp=2, layers=4, quant="bf16", prompt_tokens=255, output_tokens=1))
    mem_q32 = Memory(case_q32, policy=MemoryPolicy(rank_token_owners=((0, 128), (128, 256))))
    mem_q32.register_prepared_weights((PreparedWeightSpec("q-t", ("q_proj",), "transpose", "target", (0,), (5120, 1024), "bf16", "ND"),))
    rec_q32 = mem_q32.resident_memory_receipt(sequence_tokens=256, batch_per_dp=1)
    led_q32 = compute_memory_quant_ledger(case_q32, seq_len=256, concurrency=1, memory_receipt=rec_q32)
    assert int(sum(x.batch_total_bytes for x in led_q32.cache_rows)) == int(max(x.cache_bytes for x in led_q32.stage_budgets)) == 524_288
    assert led_q32.peak_weight_stage == 0 and int(led_q32.weight_footprint_per_die_bytes) == int(led_q32.sim_memory_weight_bytes_by_stage[0])
    assert "F" in _stage_cache_formula(led_q32, 0, 100)

    case_g = parse_case(compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a3"], dp=2, tp=2, layers=2, quant="bf16", prompt_tokens=2047, output_tokens=1))
    for owner, rel in ((1, False), (0, True)):
        mg = Memory(case_g)
        assert mg.grant_batch(owner, (("scratch", mg.resident_memory_receipt(sequence_tokens=2048, batch_per_dp=1).min_free_hbm_headroom_bytes),))
        if rel:
            assert mg.release_grant(owner, "scratch")
        rg = mg.resident_memory_receipt(sequence_tokens=2048, batch_per_dp=1)
        lg = compute_memory_quant_ledger(case_g, seq_len=2048, concurrency=1, memory_receipt=rg)
        assert int(lg.stage_budgets[0].headroom_bytes) == rg.min_free_hbm_headroom_bytes

    case_g1 = parse_case(compose_case("qwen3-32b", hardware=REPRESENTATIVE_SKUS["a3"], dp=1, tp=2, layers=2, quant="bf16", prompt_tokens=2047, output_tokens=1))
    mg1 = Memory(case_g1)
    grant_fill = ResidentGrantEntry("scratch", "workspace", "scratch", allocated_bytes=mg1.resident_memory_receipt(sequence_tokens=2048, batch_per_dp=1).min_free_hbm_headroom_bytes)
    r_g1 = mg1.resident_memory_receipt(sequence_tokens=2048, batch_per_dp=1, extra_grants=(grant_fill,))
    r_g2 = mg1.resident_memory_receipt(sequence_tokens=2048, batch_per_dp=2, extra_grants=(grant_fill,))
    l_g1 = compute_memory_quant_ledger(case_g1, seq_len=2048, concurrency=1, memory_receipt=r_g1)
    l_g2 = compute_memory_quant_ledger(case_g1, seq_len=2048, concurrency=2, memory_receipt=r_g2)
    ws_g = Worksheet("S3")
    _populate_stage_budget_block(ws_g, 10, l_g2)
    used_g = sum(ws_g._cells[12, j].value for j in (3, 4, 5, 7, 8, 9))
    assert l_g1.max_hbm_concurrency_per_dp == 1
    assert int(round((ws_g._cells[12, 11].value - used_g) * GIB_DIVISOR)) == r_g2.min_free_hbm_headroom_bytes

    l_pp = compute_memory_quant_ledger(parse_case(compose_case("qwen3-next-80b-a3b", hardware=REPRESENTATIVE_SKUS["a3"], layers=4, tp=4, pp=2, pp_layer_partition=(3, 1), quant="bf16", mtp_proposal_tokens=0, prompt_tokens=256, output_tokens=1)), seq_len=256, concurrency=1)
    f_pp = _stage_cache_formula(l_pp, 1, 100)
    assert "$T$18" in f_pp and "F" not in f_pp

    for m_name in ("deepseek-v4.1-flash", "gemma-4-31b"):
        c_s = parse_case(compose_case(m_name, hardware=REPRESENTATIVE_SKUS["a3"], tp=A3_TP_DEGREE, layers=6, quant="bf16", mtp_proposal_tokens=0, prompt_tokens=8192, output_tokens=8))
        l_s = compute_memory_quant_ledger(c_s, seq_len=4096, concurrency=1)
        ws_s = Worksheet("S3")
        _populate_cache_block(ws_s, 10, l_s, ("BF16", "C8"))
        est_2x = sum(r.batch_total_bytes * (2 if "$T$21" in (ws_s._cells[12 + i, 5].formula or "") else 1) for i, r in enumerate(l_s.cache_rows))
        assert int(est_2x) == int(Memory(c_s).cache_bytes(8192)[0])

    l_w4 = compute_memory_quant_ledger(parse_case(compose_case("qwen3-30b-a3b", hardware=REPRESENTATIVE_SKUS["a3"], tp=2, ep=2, num_devices=2, layers=2, quant="bf16", prompt_tokens=128, output_tokens=1)), weight_quant="W4A8", seq_len=128, concurrency=1)
    r_exp = next(row for row in l_w4.weight_rows if row.name == "expert_gate")
    assert int(quant_preset("w4a8_dynamic").matrix_bytes(r_exp.local_rows, r_exp.local_cols, r_exp.category) * r_exp.peak_stage_repeats) == (
        evaluate_weight_storage_bytes(r_exp.local_rows, r_exp.local_cols, "W4A8", repeats=r_exp.peak_stage_repeats)[-1]
    ) == int(r_exp.per_die_bytes)
    l_faq = compute_memory_quant_ledger(case_g1, seq_len=4096, concurrency=1, cache_dtype="FAQuant")
    l_c8 = compute_memory_quant_ledger(case_g1, seq_len=4096, concurrency=1, cache_dtype="C8")
    assert (l_faq.cache_dtype, int(l_faq.stage_budgets[0].cache_bytes)) == ("C8", int(l_c8.stage_budgets[0].cache_bytes))
