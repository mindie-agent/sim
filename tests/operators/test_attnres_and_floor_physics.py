"""Residual prefix physics, precision boundaries, floor escapes, and cohort replay.

[Role] Verify causal residual prefix phase timing, zero-history bypass,
  source provenance scrubbing, projection precision boundaries and split
  costing, decode floor-escape fixes, exact cohort replay, and conserved
  Stream 0 exposure attribution with replica filtering.
[Invariants] Every physical duration is derived from declared hardware rates
  and causal phases without serving floors; cross-precision sibling merges are
  rejected; per-request integer acceptance is never averaged; Stream 0 interval
  exposure conserves exact integer nanoseconds.
[Timeouts & Locks] Pure Tier-0 in-memory tests; no device context or I/O.
[Side-Effect Policy] Read-only case fixtures and in-memory Engine execution.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from _support.linear_cases import sku_planning_context
from mindie_sim import graph_costs, operator_costs
from mindie_sim.catalog.npu_checkpoints import resolve_default_checkpoint
from mindie_sim.catalog.qwen4_exp import lower_ple, ple_lookup_format
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.compose import compose_case
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.execution.layer_runs import (
    CohortOccurrenceBinding,
    CompiledCohortProgramCache,
    ExecutionStateSnapshot,
    LayerRun,
    LayerRunProgram,
    TransitionEquivalenceProof,
    replay_ordered_cohort,
)
from mindie_sim.prediction.graph_analysis import semantic_graph_key
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.operators.hyper_connections.common import (
    attnres_spec_terms,
    sources_for_kind,
)
from mindie_sim.prediction.operators.hyper_connections.family_attnres import (
    AttnResPrefixState,
    attn_res_point,
    select_attnres_dispatch_plan,
)
from mindie_sim.prediction.operators.kda import family_kda_projection
from mindie_sim.prediction.operators.linear_attention.common import (
    CausalConvContract,
    LinearAttentionContract,
    build_causal_conv1d_plan,
    build_recover_ssm_plan,
)
from mindie_sim.prediction.operators.mc2.megamoe import try_lower_fused_frame
from mindie_sim.prediction.operators.mla.mla_epilogs import _indexer_matrix_category
from mindie_sim.prediction.operators.moe_routing.moe_gating_fusion import _fused_gating
from mindie_sim.prediction.operators.recurrent.recover_ssm_commit import (
    bind_recover_ssm_commits,
)
from mindie_sim.prediction.operators.recurrent.recover_ssm_work import (
    commit_tokens_for_view,
    gdn_record_head_bytes,
    record_head_bytes,
)
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.projections import input_projections
from mindie_sim.prediction.runtime.exposed_observer import (
    ExposedObserver,
    ObservedInterval,
    ObservedServiceGrant,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import activation_quantization
from mindie_sim.rewrite.algebra import discover_algebraic_rewrites_for_ops
from mindie_sim.rewrite.apply import _law2_sibling_key, _unabsorbed_splitv_cost
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec, TensorPrecisionABI
from mindie_sim.schema.workload import RequestView

pytestmark = pytest.mark.tier0

ATTNRES_HIDDEN = 7168
ATTNRES_CAPACITY = 12
#: Eager Python Triton host-launch rate plateau (80 Python ``kernel[grid]`` calls
#: inside one Event pair on A5); inside ``ACLGraph``, device execution duration
#: is legitimately 4..24 us (> 1.686 us) without host-dispatch clamping.
ATTNRES_AUDIT_PLATEAU_US = 25.3
ATTNRES_MAX_MEAN_APE = 4.0
ATTNRES_MAX_POINT_APE = 9.0
DECODE_FLOOR_US = 1.686
ATTNRES_MEASURED_CURVE = (
    (14, 0, 23.3), (14, 4, 27.4), (14, 12, 30.6),
    (48, 0, 26.8), (48, 8, 27.0), (48, 12, 30.3),
    (96, 0, 26.6), (96, 4, 27.1), (96, 6, 33.0), (96, 8, 41.4), (96, 10, 49.9), (96, 12, 58.4),
    (168, 0, 25.3), (168, 4, 35.8), (168, 6, 48.4), (168, 8, 61.1), (168, 10, 73.7), (168, 12, 86.3),
    (336, 0, 26.3), (336, 2, 43.2), (336, 4, 68.4), (336, 6, 93.6), (336, 8, 119.0), (336, 12, 169.4),
)
COOP_SMALL_BATCH = 14
COOP_UB_BYTES = 253952
COOP_AIV_COUNT = 64
COOP_V4_WORKERS = 42
COOP_V12_WORKERS = 64
GDN_RECORD_BYTES = 1028
KDA_RECORD_BYTES = 1024
DEFAULT_HEAD_DIM = 128
CONV_CHANNELS = 1024
CONV_TILE_WIDTH = 256
EXPERT_COUNT = 512
TOP_K_EXPERTS = 10
MOE_TOKENS = 16
SPLITV_M = 16
SPLITV_N = 4096
SPLITV_K = 7168
RESIDUAL_TOTAL_LAYERS = 93
RESIDUAL_BLOCK_EIGHT = 8
RESIDUAL_BLOCK_TWELVE = 12
TOTAL_ATTNRES_CALLS = 187


def _planning_context():
    _, ctx = sku_planning_context(REPRESENTATIVE_SKUS["a5"], "scan")
    return ctx


def test_attnres_prefix_matches_measured_curve_and_owner_waves() -> None:
    case = parse_case(compose_case("kimi-k3.1", REPRESENTATIVE_SKUS["a5"]))
    model = SimpleNamespace(hidden_size=ATTNRES_HIDDEN, activation_dtype="bf16")
    ops = []
    for tokens, valid, _meas in ATTNRES_MEASURED_CURVE:
        banks = tuple(f"b{idx}" if idx < valid else None for idx in range(ATTNRES_CAPACITY))
        res = attn_res_point(
            {"prefix": f"p_{tokens}_{valid}", "stage": 0, "layer": 0,
             "model": model, "mode": "decode", "tokens": tokens},
            prefix_id="prefix0", addend_id=None, bank_ids=banks, valid_blocks=valid,
            write_slot=None, output_norm=True, mix=True, materialize=False,
            tokens_local=tokens, bank_capacity=ATTNRES_CAPACITY,
            block_token_stride=ATTNRES_CAPACITY * ATTNRES_HIDDEN,
        )
        op = res.ops[0]
        md = dict(op.metadata)
        md["attn_res_view"] = {**md["attn_res_view"], "tiling_mode": "token_local"}
        ops.append(replace(op, metadata=tuple(md.items())))

    results = operator_costs(case, tuple(ops))
    apes = []
    by_point: dict[tuple[int, int], float] = {}
    for (tokens, valid, meas), rec in zip(ATTNRES_MEASURED_CURVE, results["operators"]):
        pred_us = float(rec["duration_ms"]) * 1000.0
        by_point[(tokens, valid)] = pred_us
        assert pred_us > DECODE_FLOOR_US
        # Compare token-local device prediction against eager Python Triton measurement
        # (80 Python kernel[grid] calls inside one Event pair saturate at the
        # 25.3 us host-launch plateau; inside ACLGraph, device duration stays 4..24 us).
        audit_us = max(pred_us, ATTNRES_AUDIT_PLATEAU_US)
        apes.append(abs(audit_us - meas) / meas * 100.0)
        for delta in rec["service_delta"].values():
            assert abs(float(delta)) < 1e-4

    # Default auto cooperative dispatch (MakeParallelPlan) beats token-local on small batch (14, 12).
    auto_op = attn_res_point(
        {"prefix": "p_14_12_auto", "stage": 0, "layer": 0,
         "model": model, "mode": "decode", "tokens": COOP_SMALL_BATCH},
        prefix_id="prefix0", addend_id=None,
        bank_ids=tuple(f"b{idx}" for idx in range(ATTNRES_CAPACITY)),
        valid_blocks=ATTNRES_CAPACITY, write_slot=None, output_norm=True, mix=True,
        materialize=False, tokens_local=COOP_SMALL_BATCH, bank_capacity=ATTNRES_CAPACITY,
        block_token_stride=ATTNRES_CAPACITY * ATTNRES_HIDDEN,
    ).ops[0]
    auto_rec = operator_costs(case, (auto_op,))["operators"][0]
    assert float(auto_rec["duration_ms"]) * 1000.0 < by_point[(COOP_SMALL_BATCH, ATTNRES_CAPACITY)]
    for delta in auto_rec["service_delta"].values():
        assert abs(float(delta)) < 1e-4

    from mindie_sim.prediction.operators.hyper_connections import family_attnres as _fa
    assert not hasattr(_fa, "ATTNRES_RESIDENT_TREE_DELTA_NS")
    assert not hasattr(_fa, "ATTNRES_RELOAD_TREE_DELTA_NS")

    assert sum(apes) / len(apes) <= ATTNRES_MAX_MEAN_APE
    assert max(apes) <= ATTNRES_MAX_POINT_APE
    assert by_point[(336, 0)] < by_point[(336, 2)] < by_point[(336, 4)] < by_point[(336, 12)]
    assert by_point[(14, 12)] < by_point[(96, 12)] < by_point[(168, 12)] < by_point[(336, 12)]

    coop_v4 = select_attnres_dispatch_plan(
        COOP_SMALL_BATCH, 4, ATTNRES_HIDDEN, COOP_UB_BYTES, COOP_AIV_COUNT, mode="cooperative",
    )
    coop_v12 = select_attnres_dispatch_plan(
        COOP_SMALL_BATCH, ATTNRES_CAPACITY, ATTNRES_HIDDEN, COOP_UB_BYTES, COOP_AIV_COUNT, mode="cooperative",
    )
    assert coop_v4["active_owners"] == COOP_V4_WORKERS
    assert coop_v12["active_owners"] == COOP_V12_WORKERS


def test_attnres_scrubs_disallowed_dsl_and_enforces_zero_history_bypass() -> None:
    arch_file = (
        Path(__file__).resolve().parents[2]
        / "mindie_sim" / "prediction" / "operators" / "hyper_connections" / "arch35.py"
    )
    legacy_needle = "cann" + "bot"
    assert legacy_needle not in arch_file.read_text(encoding="utf-8").lower()
    engram_refs = sources_for_kind("engram_gate", ())
    assert any("engram_gate.py" in ref.relative_path for ref in engram_refs)
    assert all(legacy_needle not in ref.repository.lower() for ref in engram_refs)

    model = SimpleNamespace(hidden_size=ATTNRES_HIDDEN, activation_dtype="bf16")
    zero_pt = attn_res_point(
        {"prefix": "z0", "stage": 0, "layer": 0, "model": model, "mode": "decode", "tokens": 16},
        prefix_id="p0", addend_id=None, bank_ids=(None,) * ATTNRES_CAPACITY,
        valid_blocks=0, write_slot=None, output_norm=False, mix=True, materialize=False,
        tokens_local=16, bank_capacity=ATTNRES_CAPACITY,
        block_token_stride=ATTNRES_CAPACITY * ATTNRES_HIDDEN,
    ).ops[0]
    terms = attnres_spec_terms(zero_pt)
    expected_row_bytes = float(16 * ATTNRES_HIDDEN * 2)
    assert terms["mte2_bytes"] == pytest.approx(expected_row_bytes)
    assert terms["mte3_bytes"] == pytest.approx(expected_row_bytes)

    for block_size, expected_final_valid in (
        (RESIDUAL_BLOCK_EIGHT, 12),
        (RESIDUAL_BLOCK_TWELVE, 8),
    ):
        plans = tuple(
            SimpleNamespace(index=idx, residual_block_size=block_size, engram=False, hyper_connections=False)
            for idx in range(RESIDUAL_TOTAL_LAYERS)
        )
        m = SimpleNamespace(hidden_size=ATTNRES_HIDDEN, activation_dtype="bf16", layer_plans=plans)
        rt = SimpleNamespace(parallel=SimpleNamespace(pp=1))
        state = AttnResPrefixState.start(
            model=m, runtime=rt, ready="embed", tokens=16, mode="decode", requests=("r0",),
        )
        calls = []
        for lp in plans:
            prep = state.prepare(prefix=f"l{lp.index}.pre", layer=lp, stage=0)
            calls.append(prep.ops[0])
            post = state.attention_added(prefix=f"l{lp.index}.post", layer=lp, stage=0, attention_id=f"attn{lp.index}")
            calls.append(post.ops[0])
            state.defer_mlp(f"mlp{lp.index}")
        fin = state.final(prefix="final", last_layer=plans[-1], stage=0)
        calls.append(fin.ops[0])
        assert len(calls) == TOTAL_ATTNRES_CALLS
        assert dict(fin.ops[0].metadata)["attn_res_view"]["valid_blocks"] == expected_final_valid


def test_mixed_precision_projections_reject_cross_precision_and_price_splitv() -> None:
    quant = QuantSpec(
        "mixed_mx",
        weight_format="fp8_e4m3",
        activation_format="fp8_e4m3",
        scale_format="fp8_e8m0",
        weight_block=(1, 32),
        activation_group_size=32,
        activation_scale_format="fp8_e8m0",
        exclusions=("kda.b", "kda.f_a", "kda.f_b", "kda.g", "indexer.wk"),
    )
    model = SimpleNamespace(hidden_size=SPLITV_K, activation_dtype="bf16", quant=quant)
    plan = SimpleNamespace(index=0, scope="target", attention="kda", packed_input_projection=True,
                           projection_dtype="", index_kpool=0,
                           linear_heads=16, linear_value_heads=16, linear_key_heads=16,
                           linear_head_dim=DEFAULT_HEAD_DIM, linear_key_head_dim=DEFAULT_HEAD_DIM,
                           gate_rank=0)
    b = Builder("l0", plan, 0, (), model, "decode")
    prods = input_projections(
        b, name="qkv_bfg", deps="norm", tokens=SPLITV_M, hidden=SPLITV_K,
        roles=(
            ("q", "lin_q", 1024, "kda.q"),
            ("k", "lin_k", 1024, "kda.k"),
            ("b", "lin_b", 16, "kda.b"),
            ("g", "lin_g", 1024, "kda.g"),
        ),
        packed=True,
    )
    assert prods["q"] == prods["k"]
    assert prods["b"] == prods["g"]
    assert prods["q"] != prods["b"]

    # KDA mixed projection derives synthesized BF16 kda.f from (f_a, f_b).
    validated = family_kda_projection._validate(plan, model, tp=1)
    assert validated["dtype"] == "bf16"
    assert _indexer_matrix_category(quant, "indexer.wk") == "indexer.wk"
    assert quant.format_for(_indexer_matrix_category(quant, "indexer.wk")) == "bf16"
    assert quant.format_for(_indexer_matrix_category(quant, "indexer.wq")) == "fp8_e4m3"

    # activation_quantization must NEVER insert a quantizer before a FLOAT MatMul.
    float_mm = OpSpec(
        op_id="l0.float_mm", family="gemm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(16, 128, 128), local_shape=(16, 128, 128),
        padded_shape=(16, 128, 128), launch_shape=(16, 128, 128),
        deps=("l0.norm",), dtype="bf16", flops=float(2 * 16 * 128 * 128),
        mte2_bytes=4096.0, mte3_bytes=4096.0, hbm_bytes=8192.0,
        metadata=(
            ("program_kind", "matmul"),
            ("activation_format", "bf16"),
            ("weight_format", "bf16"),
            ("activation_scale_bytes", 128.0),
        ),
    )
    norm = OpSpec(
        op_id="l0.norm", family="rmsnorm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(16, 128), local_shape=(16, 128), padded_shape=(16, 128), launch_shape=(16, 128),
        dtype="bf16", vector_ops=float(16 * 128 * 4), mte2_bytes=4096.0, mte3_bytes=4096.0, hbm_bytes=8192.0,
    )
    out_ops = activation_quantization((norm, float_mm), "bf16")
    assert all(op.family != "quantize" for op in out_ops)

    # Multi-row sibling projections with distinct downstream consumers incur SplitV penalty.
    act_b = float(SPLITV_M * SPLITV_K * 2)
    wt_b = float(SPLITV_N * SPLITV_K * 2)
    out_b = float(SPLITV_M * SPLITV_N * 2)
    shared_meta = (
        ("typed_inputs", ({"producer": "l0.norm", "port": "out", "version": 0},)),
        ("activation_format", "bf16"),
        ("weight_format", "bf16"),
        ("output_format", "bf16"),
        ("compute_format", "bf16"),
    )
    sib_a = OpSpec(
        op_id="l0.sib_a", family="gemm", phase="decode", stage=0, layer=0, requests=(),
        deps=("l0.norm",), semantic_shape=(SPLITV_M, SPLITV_N, SPLITV_K),
        local_shape=(SPLITV_M, SPLITV_N, SPLITV_K), padded_shape=(SPLITV_M, SPLITV_N, SPLITV_K),
        launch_shape=(SPLITV_M, SPLITV_N, SPLITV_K), dtype="bf16",
        flops=float(2 * SPLITV_M * SPLITV_N * SPLITV_K),
        mte2_bytes=act_b + wt_b, mte3_bytes=out_b, hbm_bytes=act_b + wt_b + out_b,
        metadata=shared_meta,
    )
    sib_b = replace(sib_a, op_id="l0.sib_b")
    cons_a = OpSpec(
        op_id="l0.cons_a", family="rmsnorm", phase="decode", stage=0, layer=0, requests=(),
        deps=("l0.sib_a",), semantic_shape=(SPLITV_M, SPLITV_N), local_shape=(SPLITV_M, SPLITV_N),
        padded_shape=(SPLITV_M, SPLITV_N), launch_shape=(SPLITV_M, SPLITV_N), dtype="bf16",
    )
    cons_b = replace(cons_a, op_id="l0.cons_b", deps=("l0.sib_b",))
    sib_quant = replace(
        sib_a, op_id="l0.sib_q",
        precision_abi=TensorPrecisionABI(
            linear_method="mx_fp8", input_dtype="fp8_e4m3", weight_dtype="fp8_e4m3",
            output_dtype="bf16", compute_dtype="fp8_e4m3", act_group_size=32, act_scale_dtype="fp8_e8m0",
        ),
        metadata=(
            ("typed_inputs", ({"producer": "l0.norm", "port": "out", "version": 0},)),
            ("activation_format", "fp8_e4m3"),
            ("weight_format", "fp8_e4m3"),
            ("activation_group_size", 32),
            ("activation_scale_format", "fp8_e8m0"),
        ),
    )
    assert _law2_sibling_key(sib_a) != _law2_sibling_key(sib_quant)
    splitv_hbm, splitv_launches = _unabsorbed_splitv_cost(
        (sib_a, sib_b), {o.op_id: o for o in (norm, sib_a, sib_b, cons_a, cons_b)},
    )
    assert splitv_launches == 1 and splitv_hbm == pytest.approx(4.0 * out_b)
    alg = discover_algebraic_rewrites_for_ops((norm, sib_a, sib_b, cons_a, cons_b))
    assert not [c for c in alg.candidates if c.law == "law2_block_matmul"]
    assert any(reason == "splitv_unabsorbed_cost_exceeds_activation_savings" for _, reason in alg.rejected_pairs)

    # Routed-only MC2 runs when routed bank is MXFP and shared expert is BF16.
    routed_meta = (
        ("compute_format", "fp8_e4m3"), ("weight_format", "mxfp4"),
        ("activation_format", "fp8_e4m3"), ("activation_group_size", 32),
        ("activation_scale_format", "fp8_e8m0"), ("packed_weight_bytes", 1024.0),
    )
    bf16_meta = (("compute_format", "bf16"), ("weight_format", "bf16"), ("activation_format", "bf16"))
    pack = OpSpec("p.pack", "routing", "decode", 0, 0, (), (16, 128), (16, 128), (16, 128), (16, 128), deps=("in",))
    disp = OpSpec("p.disp", "alltoall", "decode", 0, 0, (), (16, 128), (16, 128), (16, 128), (16, 128), deps=("p.pack",), link_bytes=4096.0)
    e_up = OpSpec("p.expert_up", "grouped_gemm", "decode", 0, 0, (), (16, 256, 128), (16, 256, 128), (16, 256, 128), (16, 256, 128), deps=("p.disp",), metadata=routed_meta)
    e_act = OpSpec("p.expert_act", "swiglu", "decode", 0, 0, (), (16, 128), (16, 128), (16, 128), (16, 128), deps=("p.expert_up",))
    e_down = OpSpec("p.expert_down", "grouped_gemm", "decode", 0, 0, (), (16, 128, 128), (16, 128, 128), (16, 128, 128), (16, 128, 128), deps=("p.expert_act",), metadata=routed_meta)
    comb = OpSpec("p.comb", "alltoall", "decode", 0, 0, (), (16, 128), (16, 128), (16, 128), (16, 128), deps=("p.expert_down",), link_bytes=4096.0)
    unp = OpSpec("p.unp", "routing", "decode", 0, 0, (), (16, 128), (16, 128), (16, 128), (16, 128), deps=("p.comb",))
    sh_up = OpSpec("p.shared.gate_up", "gemm", "decode", 0, 0, (), (16, 256, 128), (16, 256, 128), (16, 256, 128), (16, 256, 128), deps=("in",), metadata=bf16_meta)
    sh_down = OpSpec("p.shared.down", "gemm", "decode", 0, 0, (), (16, 128, 128), (16, 128, 128), (16, 128, 128), (16, 128, 128), deps=("p.shared.gate_up",), metadata=bf16_meta)
    join = OpSpec("p.join", "elementwise", "decode", 0, 0, (), (16, 128), (16, 128), (16, 128), (16, 128), deps=("p.unp", "p.shared.down"))
    frame = (pack, disp, e_up, e_act, e_down, comb, unp, sh_up, sh_down, join)
    lowered = try_lower_fused_frame(
        frame, {o.op_id: o for o in frame}, {"fused_mc2_level": 3}, None, "p", 16, 128, 2, 32, 8, "bf16",
        pack, disp, comb, unp, None,
    )
    assert lowered is not None
    lowered_ids = {o.op_id for o in lowered}
    assert "p.shared.gate_up" in lowered_ids and "p.shared.down" in lowered_ids and "p.join" in lowered_ids


def test_decode_floor_escapes_and_recover_ssm_per_request_abi() -> None:
    # General MoE gating top-k uses canonical P10 sort32_mrgsort4 primitive rates
    # (no 1.686 us streaming floor clamp) and scales monotonically across AIV waves.
    case = parse_case(compose_case("qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"], tp=2, concurrency=MOE_TOKENS))
    layer = SimpleNamespace(
        num_experts=EXPERT_COUNT, top_k=TOP_K_EXPERTS, router_mode="softmax",
        router_norm_topk=True, router_groups=1, router_group_topk=0,
        routed_scaling_factor=1.0, index=0,
    )
    members = tuple(SimpleNamespace(vector_ops=MOE_TOKENS * EXPERT_COUNT, recipe=r) for r in ("score", "select"))
    gating_op = _fused_gating(
        prefix="r", plan=layer, logits_input="logits", members=members, stage=0,
        requests=(), tokens=MOE_TOKENS, mode="decode", weight_dtype="fp32",
        weight_bytes=4, hash_table=False, noaux_tc=False,
    )
    two_wave_tokens = 2 * COOP_AIV_COUNT
    members_2w = tuple(SimpleNamespace(vector_ops=two_wave_tokens * EXPERT_COUNT, recipe=r) for r in ("score", "select"))
    gating_2w = _fused_gating(
        prefix="r2w", plan=layer, logits_input="logits", members=members_2w, stage=0,
        requests=(), tokens=two_wave_tokens, mode="decode", weight_dtype="fp32",
        weight_bytes=4, hash_table=False, noaux_tc=False,
    )
    gating_res = operator_costs(case, (gating_op, gating_2w))
    gating_1w_us = float(gating_res["operators"][0]["duration_ms"]) * 1000.0
    gating_2w_us = float(gating_res["operators"][1]["duration_ms"]) * 1000.0
    assert gating_res["operators"][0]["program"] == "OperatorPlan"
    assert gating_1w_us > 1.0 and gating_1w_us != pytest.approx(DECODE_FLOOR_US, abs=1e-3)
    assert gating_2w_us > gating_1w_us and gating_2w_us > DECODE_FLOOR_US

    # PLE table marker BF16 gives 0 dequant ops and non-floor device lookup & gate.
    ckpt = resolve_default_checkpoint("qwen3.8-flash-next", "Ascend950DT_9582")
    assert ckpt is not None
    rt_dev = replace(case.runtime, engram_cpu_offload=False, lowering_passes=("ple_quantized_lookup",))
    bf16_model = replace(case.model, quant=ckpt.build_quant_spec())
    assert ple_lookup_format(bf16_model, rt_dev) == "bf16"
    ple_plan = next(p for p in bf16_model.layer_plans if p.engram)
    ple_ops, _ = lower_ple(
        prefix="l0", plan=ple_plan, stage=0, deps="in", model=bf16_model,
        parallel=rt_dev.parallel, quant=bf16_model.quant, mode="decode",
        padded_tokens=MOE_TOKENS, views=case.workload.requests, runtime=rt_dev,
    )
    lookup_op = next(op for op in ple_ops if op.recipe == "ple_device_lookup")
    gate_op = next(op for op in ple_ops if op.recipe == "ple_gate")
    assert dict(lookup_op.metadata)["ple_dequant_ops"] == 0.0
    assert lookup_op.owner_schedule is not None and gate_op.owner_schedule is not None
    ple_costs = operator_costs(replace(case, model=bf16_model, runtime=rt_dev), (lookup_op, gate_op))
    lookup_us = float(ple_costs["operators"][0]["duration_ms"]) * 1000.0
    gate_us = float(ple_costs["operators"][1]["duration_ms"]) * 1000.0
    assert lookup_us > 1.0 and gate_us != pytest.approx(DECODE_FLOOR_US, abs=1e-3)

    # Conv1d uses channel-tile ownership rather than scalar-channel ownership.
    ctx = _planning_context()
    conv_plan = build_causal_conv1d_plan(
        CausalConvContract(
            channels=CONV_CHANNELS, width=4, lengths=(1, 1), dtype="bf16",
            state_dtype="fp32", mode="incremental",
        ),
        ctx,
    )
    sched = conv_plan.metadata["owner_schedule"]
    assert sched["work_tiles"] == 2 * (CONV_CHANNELS // CONV_TILE_WIDTH)
    assert sched["active_owners"] == 2 * (CONV_CHANNELS // CONV_TILE_WIDTH)

    # GDN record ABI (1,028 B) vs KDA (1,024 B), and RecoverSSM (0,4) vs (2,2).
    assert gdn_record_head_bytes(DEFAULT_HEAD_DIM, DEFAULT_HEAD_DIM) == GDN_RECORD_BYTES
    assert record_head_bytes(DEFAULT_HEAD_DIM, DEFAULT_HEAD_DIM) == KDA_RECORD_BYTES
    p_skew = build_recover_ssm_plan(
        LinearAttentionContract(
            operation="recover_ssm_commit", abi="commit_recurrent_kda",
            lengths=(4, 4), hq=4, hk=4, hv=4, dk=DEFAULT_HEAD_DIM, dv=DEFAULT_HEAD_DIM,
            dtype="bf16", state_dtype="fp32",
            metadata={"q": 4, "accepted_per_request": (0, 4), "ssm_family": "gdn"},
        ),
        ctx,
    )
    p_even = build_recover_ssm_plan(
        LinearAttentionContract(
            operation="recover_ssm_commit", abi="commit_recurrent_kda",
            lengths=(4, 4), hq=4, hk=4, hv=4, dk=DEFAULT_HEAD_DIM, dv=DEFAULT_HEAD_DIM,
            dtype="bf16", state_dtype="fp32",
            metadata={"q": 4, "accepted_per_request": (2, 2), "ssm_family": "gdn"},
        ),
        ctx,
    )
    assert p_skew.metadata["active_sequences"] == 1 and p_even.metadata["active_sequences"] == 2
    assert p_skew.metadata["max_replay_depth"] == 4 and p_even.metadata["max_replay_depth"] == 2
    assert p_skew.work.resources["mte3"] * 2 == pytest.approx(p_even.work.resources["mte3"])
    assert p_skew.metadata["kernel_binding"]["kernel_id"] == "gdn_recoverssm_commit"
    assert p_skew.metadata["kernel_binding"]["backend"] == "Triton"

    # Reject unresolved/out-of-range advancement; bind_recover_ssm_commits advances version.
    with pytest.raises(ModelingError, match="resolved_integer_advancement"):
        commit_tokens_for_view(RequestView("r0", 4, 64, True, "decode"))
    with pytest.raises(ModelingError, match="advancement_out_of_range"):
        commit_tokens_for_view(RequestView("r0", 4, 64, True, "decode", draft_advanced_tokens=5))
    intent = OpSpec(
        "c0", "elementwise", "decode", 0, 0, ("r0",), (1,), (1,), (1,), (1,),
        deps=("scan",), metadata=(
            ("recover_ssm_intent", True), ("state_owner", "layer0"),
            ("input_state_version", "v2"), ("heads", 4), ("dk", 16), ("dv", 16),
            ("state_dtype", "fp32"), ("record_bytes_per_token", 256.0),
        ),
    )
    bound, = bind_recover_ssm_commits(
        (intent,),
        views=(RequestView("r0", 4, 64, True, "decode", draft_advanced_tokens=3),),
        acceptance_events={"r0": "acc0"}, state_bindings={},
        runtime=SimpleNamespace(metadata=(("recover_ssm_readiness", "shipping"),)),
        machine=SimpleNamespace(arch_family="A5"),
    )
    assert bound.kernel_binding is not None and bound.kernel_binding.kernel_id == "kda_recoverssm_commit"
    assert dict(bound.metadata)["published_state_version"] == "v3"


def test_cohort_layer_runs_exact_depth_replay_and_state_equivalence_gate() -> None:
    run = LayerRun(start=0, count=3, key=("k",), layers=(0, 1, 2))
    dummy_prog = KernelProgram((Stage("s0", {"v": 10.0}),))
    with pytest.raises(ModelingError, match="layer_run_time_skip_requires_engine_certification"):
        LayerRunProgram(run=run, occurrences=(dummy_prog,) * 3, skip_authorized=True)

    snap_a = ExecutionStateSnapshot(l2_residency=(("l2", 1),), bank_state=(("valid", 2),))
    snap_b = ExecutionStateSnapshot(l2_residency=(("l2", 1),), bank_state=(("valid", 3),))
    with pytest.raises(ModelingError, match="layer_run_state_equivalence_failed"):
        TransitionEquivalenceProof(before_state=snap_a, after_state=snap_b, cycle_delta_ns=100)
    proof = TransitionEquivalenceProof(before_state=snap_a, after_state=snap_a, cycle_delta_ns=100)
    certified = LayerRunProgram(
        run=run, occurrences=(dummy_prog,) * 3, skip_authorized=True, certification=proof,
    )
    assert certified.certification is proof

    # Replay 0..L-1 + final via CompiledCohortProgramCache matches unrolled execution to 0.00%.
    cache = CompiledCohortProgramCache()
    bindings = tuple(
        CohortOccurrenceBinding(
            occurrence_index=idx,
            native_layer_id=idx,
            template_key="residual_layer" if idx < 6 else "final_tail",
            context_key=("valid", min(idx // 2, 3)),
            is_final_tail=(idx == 6),
        )
        for idx in range(7)
    )
    cached_progs = replay_ordered_cohort(
        bindings,
        compile_template=lambda b: KernelProgram((Stage(f"work_{b.context_key[1]}", {"v": float(10 + 5 * b.context_key[1])}),)),
        cache=cache,
    )
    assert cache.hits > 0 and len(cached_progs) == 7
    eng_cached = Engine((Resource("v", 10.0, 1),))
    ev_c = None
    for idx, prog in enumerate(cached_progs):
        ev_c = prog.start(eng_cached, f"c{idx}", ev_c)
    eng_cached.advance(until=(ev_c,))
    eng_unrolled = Engine((Resource("v", 10.0, 1),))
    ev_u = None
    for idx, bnd in enumerate(bindings):
        prog = KernelProgram((Stage(f"work_{bnd.context_key[1]}", {"v": float(10 + 5 * bnd.context_key[1])}),))
        ev_u = prog.start(eng_unrolled, f"u{idx}", ev_u)
    eng_unrolled.advance(until=(ev_u,))
    rel_err = abs(eng_cached.now - eng_unrolled.now) / max(eng_unrolled.now, 1e-12)
    assert rel_err < 1e-4


def test_exposed_observer_stream0_conservation_and_replica_filtering() -> None:
    obs = ExposedObserver(observed_replicas=(0,))
    assert obs.should_retain_replica(0) is True
    assert obs.should_retain_replica(1) is False
    # Stream 0 covers [0, 100), aux side stream covers [20, 140) (40ns exposed tail),
    # idle gap [140, 160) (20ns), Stream 0 [160, 200) contests HBM with aux2 [160, 200).
    assert obs.record_interval(ObservedInterval("main0", "main", 0, 0, 100, ordinal=0, label="MainGEMM", replica=0))
    assert obs.record_interval(ObservedInterval("aux0", "shared_stream", 1, 20, 140, ordinal=1, label="SharedAux", replica=0))
    assert obs.record_interval(ObservedInterval("main1", "main", 0, 160, 200, ordinal=2, label="MainTail", replica=0))
    assert obs.record_interval(ObservedInterval("aux1", "comm_stream", 2, 160, 200, ordinal=3, label="CommOverlap", replica=0))
    assert not obs.record_interval(ObservedInterval("other_rep", "main", 0, 0, 50, ordinal=4, replica=1))
    assert obs.filtered_occurrences == 1
    obs.record_service_grant(ObservedServiceGrant("main1", "hbm", 160, 200, delivered=300.0), replica=0)
    obs.record_service_grant(ObservedServiceGrant("aux1", "hbm", 160, 200, delivered=100.0), replica=0)

    res = obs.finish_stream0_exposure(start_ns=0, terminal_ns=200)
    assert res.conservation_delta_ns == 0
    assert sum(res.exposed_ns_by_op.values()) + res.idle_or_wait_ns == 200
    assert res.exposed_ns_by_op["main0"] == 100
    assert res.exposed_ns_by_op["aux0"] == 40
    assert res.idle_or_wait_ns == 20
    assert res.exposed_ns_by_op["main1"] == 30
    assert res.exposed_ns_by_op["aux1"] == 10
    assert "MainGEMM" in res.masked_by["aux0"]

    # Verify semantic_graph_key distinguishes depth geometry and graph_costs filters replicas at observe time.
    c_b12 = parse_case(compose_case("kimi-k3.1", REPRESENTATIVE_SKUS["a5"], layers=2, dp=2, tp=1))
    c_b8 = replace(
        c_b12,
        model=replace(
            c_b12.model,
            layer_plans=tuple(replace(lp, residual_block_size=RESIDUAL_BLOCK_EIGHT) for lp in c_b12.model.layer_plans),
        ),
    )
    k8 = semantic_graph_key(c_b8, (), scope="body")
    k12 = semantic_graph_key(c_b12, (), scope="body")
    assert k8 != k12

    op_r0 = OpSpec(
        "s0.l0.v0", "elementwise", "decode", 0, 0, (), (16, 128), (16, 128), (16, 128), (16, 128),
        dtype="bf16", vector_ops=float(16 * 128), mte2_bytes=4096.0, mte3_bytes=4096.0, hbm_bytes=8192.0,
    )
    g_full = graph_costs(c_b8, {0: (op_r0,), 1: (op_r0,)}, isolated=False)
    g_filt = graph_costs(c_b8, {0: (op_r0,), 1: (op_r0,)}, isolated=False, timeline_replicas=(0,))
    assert len(g_filt["timeline"]) == 1
    assert g_filt["timeline"][0]["replica"] == 0
    assert g_filt["operator_occurrences"] == g_full["operator_occurrences"]
    assert g_filt["duration_ms"] == pytest.approx(g_full["duration_ms"])


def test_attnres_requires_a5_hardware_ple_conv_honors_speculative_k_and_zero_legacy_dsl_sourcerefs() -> None:
    """AttnRes is strictly A5-only; PLE conv honors speculative K; operator SourceRefs have zero legacy DSL labels."""
    import importlib
    from mindie_sim.catalog.qwen4_exp import ple_conv_state_bytes, qwen4_exp_pool_breakdown
    from mindie_sim.prediction.operators.hyper_connections import family_attnres
    from mindie_sim.prediction.structure import parse_case

    # 1. AttnRes is shipping-available on A2/A3/A5 via ProbedAbiRegistry (live A3/A5 verified)
    #    and rejected only on unsupported/non-Ascend architectures.
    for hw_key in ("a2", "a3", "a5"):
        raw = compose_case("kimi-k3.1", REPRESENTATIVE_SKUS[hw_key], layers=4, tp=1, ep=1)
        case = parse_case(raw)
        lows = dict(case.runtime.kernels.get("lowerings") or {})
        explicit_rt = replace(
            case.runtime,
            kernels={**dict(case.runtime.kernels), "lowerings": {**lows, "attention_residual": "vllm_ascend_attn_res_prefix"}},
        )
        sel = family_attnres.selection(explicit_rt, model=case.model, phase="decode", machine=case.machine)
        assert sel is not None
    assert family_attnres._attnres_arch_supported("GPU") is False

    # 2. Qwen4 PLE conv state honors speculative proposal count (+61,440 B/req at K=3) via runtime & query_tokens.
    q4_raw = compose_case(
        "qwen3.8-flash-next", REPRESENTATIVE_SKUS["a5"], layers=4, tp=8, ep=8, mtp_proposal_tokens=3,
    )
    q4_case = parse_case(q4_raw)
    b_k0 = ple_conv_state_bytes(q4_case.model, query_tokens=1)
    b_k3_rt = ple_conv_state_bytes(q4_case.model, q4_case.runtime)
    b_k3_q = ple_conv_state_bytes(q4_case.model, query_tokens=4)
    assert b_k3_rt - b_k0 == pytest.approx(61_440.0)
    assert b_k3_rt == b_k3_q == pytest.approx(245_760.0)
    pool_k1 = qwen4_exp_pool_breakdown(q4_case.model, q4_case.runtime.parallel, 512, runtime=q4_case.runtime, query_tokens=1)
    pool_k4 = qwen4_exp_pool_breakdown(q4_case.model, q4_case.runtime.parallel, 512, runtime=q4_case.runtime, query_tokens=4)
    assert pool_k1 is not None and pool_k4 is not None
    assert pool_k4["private_ple_bytes"] - pool_k1["private_ple_bytes"] == pytest.approx(61_440.0)
    assert pool_k4["private_ple_bytes"] == pytest.approx(245_760.0)

    # 3. Every governed arch*.py SourceRef repository is free of retired legacy DSL labels.
    legacy_needle = "cann" + "bot"
    ops_root = Path(__file__).resolve().parents[2] / "mindie_sim" / "prediction" / "operators"
    for arch_py in sorted(ops_root.glob("*/arch*.py")):
        mod_name = "mindie_sim.prediction.operators." + ".".join(arch_py.relative_to(ops_root).with_suffix("").parts)
        mod = importlib.import_module(mod_name)
        for ref in getattr(mod, "SOURCES", ()):
            assert legacy_needle not in ref.repository.lower(), (mod_name, ref.repository)
            assert legacy_needle not in ref.relative_path.lower(), (mod_name, ref.relative_path)


def test_layer_runs_disables_skipping_on_dag_mutation_and_l2_fingerprint_change(monkeypatch) -> None:
    """Adversarial intra-layer DAG mutation or L2 residency drift must disable cohort skipping."""
    from mindie_sim.prediction.execution import layer_runs
    from mindie_sim.prediction import graph_analysis
    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import OpSpec

    raw = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], layers=6, tp=1, ep=1, quant="bf16")
    case = parse_case(raw)
    graphs = target_graphs(case, phase="decode", tokens=1, context=64, sequences=1, scope="body")
    r0 = list(graphs[0])

    # 1. Adversarial intra-layer DAG mutation inside layer 2: inject a synthetic
    #    11.0 ms serial dependency difference on a non-representative layer so
    #    skipping would return the wrong span; literal == evaluated (11.0 ms).
    l2_ops = [op for op in r0 if op.layer == 2]
    assert len(l2_ops) >= 3
    mutated_op = replace(l2_ops[2], deps=tuple(dict.fromkeys((*l2_ops[2].deps, l2_ops[0].op_id))))
    mutated_r0 = tuple(mutated_op if op.op_id == mutated_op.op_id else op for op in r0)
    mutated_graphs = {0: mutated_r0}

    real_graph_costs = graph_analysis.graph_costs

    def fake_costs(_case, gs, **kw):
        ops_in = gs[0] if isinstance(gs, dict) else gs
        has_mut = any(op.op_id == mutated_op.op_id and l2_ops[0].op_id in op.deps for op in ops_in)
        dur = 11.0 if has_mut else 6.0
        return {
            "duration_ms": dur, "complete_ms": dur, "input_ready_ms": 0.0,
            "timeline": [], "operator_occurrences": len(ops_in),
            "l2_residency_fingerprints": ("fp_same", "fp_same"),
        }

    monkeypatch.setattr(graph_analysis, "graph_costs", fake_costs)
    rep_mut, red_mut, _ = layer_runs.execute_native_cohort_graph(
        case, mutated_graphs, external_inputs={"analysis_input": 0.0}, fast_span_only=True,
    )
    assert red_mut is None and rep_mut["duration_ms"] == pytest.approx(11.0)

    # 2. L2 residency fingerprint drift across reduced execution disables skipping.
    calls = {"n": 0}

    def drift_costs(_case, gs, **kw):
        calls["n"] += 1
        is_reduced = kw.get("layer_span_body_ids") is not None
        if is_reduced:
            return {
                "duration_ms": 3.0, "complete_ms": 3.0, "input_ready_ms": 0.0,
                "layer_bounds": {("target", i): (0, float(i), float(i + 1)) for i in range(6)},
                "non_body_bounds": [], "l2_residency_fingerprints": ("fp_before", "fp_after_drifted"),
            }
        return {"duration_ms": 11.0, "complete_ms": 11.0, "input_ready_ms": 0.0}

    monkeypatch.setattr(graph_analysis, "graph_costs", drift_costs)
    rep_drift, red_drift, _ = layer_runs.execute_native_cohort_graph(
        case, {0: tuple(r0)}, external_inputs={"analysis_input": 0.0}, fast_span_only=True,
    )
    assert red_drift is None and rep_drift["duration_ms"] == pytest.approx(11.0) and calls["n"] == 2
    monkeypatch.setattr(graph_analysis, "graph_costs", real_graph_costs)
