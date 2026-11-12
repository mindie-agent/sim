"""Service-span chunked prefill law and hardware cube-peak sensitivity tests.

[Role]
  Verify ``chunked_prefill_mean_ttft_ms`` FCFS round-robin chunking across DP
  ranks and confirm that ``PipeSupply.from_hardware`` + ``OperatorPiecewiseCompiler``
  produce strictly monotonic GEMM durations when the declared Cube peak moves.
[Invariants]
  * ``chunked_prefill_mean_ttft_ms`` depends only on declared prompt lengths,
    ``dp``, ``max_batched_tokens``, and the single-chunk callback ``body_one``.
  * Doubling the declared Cube peak shortens a compute-bound GEMM; halving it
    lengthens that GEMM, with no validator symbols imported.
[Timeouts & Locks] Tier 0 (<5s), pure in-memory arithmetic, no I/O or locks.
[Side-Effect Policy] Stateless in-memory objects only.
"""
from __future__ import annotations

import pytest

from mindie_sim.prediction.piecewise import (
    ImplementationPolicy,
    OperatorPiecewiseCompiler,
    PipeSupply,
    WorkloadGeometry,
    chunked_prefill_mean_ttft_ms,
    lower_layer_predictions,
    service_ttft_overhead_ms,
)
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.subgraphs import BoundOperator, OperatorTemplate, StaticSignature


def test_chunked_prefill_mean_ttft_round_robin_and_chunking():
    single = chunked_prefill_mean_ttft_ms(
        [4096], dp=2, max_batched_tokens=8192, body_one=lambda q: 0.1 * q
    )
    assert single == pytest.approx(409.6)

    # Two 8192-token requests on 1 DP rank run sequentially: finishes at T and 2T, mean 1.5T.
    two_waves = chunked_prefill_mean_ttft_ms(
        [8192, 8192], dp=1, max_batched_tokens=8192, body_one=lambda q: 0.1 * q
    )
    assert two_waves == pytest.approx(819.2 * 1.5)

    # Mixed 64k + seven 8k requests across 4 DP ranks is faster than eight 64k requests.
    mixed = chunked_prefill_mean_ttft_ms(
        [65536] + [8192] * 7, dp=4, max_batched_tokens=8192, body_one=lambda q: 0.1 * q
    )
    homog = chunked_prefill_mean_ttft_ms(
        [65536] * 8, dp=4, max_batched_tokens=8192, body_one=lambda q: 0.1 * q
    )
    assert mixed < homog

    # Two-argument body_one(chunk_tokens, kv_context_tokens) tracks cumulative KV context.
    seen_chunks: list[tuple[int, int]] = []
    ctx_aware = chunked_prefill_mean_ttft_ms(
        [16384],
        dp=1,
        max_batched_tokens=8192,
        body_one=lambda q, kv: (seen_chunks.append((q, kv)) or (0.05 * q + 0.01 * kv)),
    )
    assert seen_chunks == [(8192, 8192), (8192, 16384)]
    assert ctx_aware == pytest.approx((0.05 * 8192 + 0.01 * 8192) + (0.05 * 8192 + 0.01 * 16384))
    assert service_ttft_overhead_ms(prompt_tokens=4096, tp=8, ep=32) == 0.0
    # Host time is a declared input, not a prompt/TP/EP fitted constant.
    assert service_ttft_overhead_ms(
        prompt_tokens=1024, tp=8, ep=32, host_overhead_ms=145.0,
    ) == pytest.approx(145.0)
    assert service_ttft_overhead_ms(
        prompt_tokens=4096, tp=1, ep=1, host_overhead_ms=145.0,
    ) == pytest.approx(145.0)


def test_hardware_cube_peak_sensitivity_moves_compiled_gemm():
    op = BoundOperator(
        OperatorTemplate(name="gemm", family="gemm", role="gemm"),
        StaticSignature(
            role="gemm",
            matrix=(("K", 4096), ("N", 4096)),
            quant=(
                ("act_bytes", 2.0),
                ("weight_format", "bf16"),
                ("act_format", "bf16"),
                ("weight_total", 4096 * 4096 * 2),
                ("weight_payload", 4096 * 4096 * 2),
            ),
            tiles=(("mt", 128), ("nt", 128), ("cores", 32)),
        ),
    )
    geom = WorkloadGeometry.homogeneous(1, 2048, 2048, phase="prefill")

    def _compile_duration(cube_tflops: float) -> float:
        hw = Hardware.from_mapping(
            {
                "backend": "ascend",
                "sku": "Ascend950PR_9579",
                "num_devices": 1,
                "cube_flops_per_s": cube_tflops * 1e12,
            }
        )
        supply = PipeSupply.from_hardware(hw, dtype="bf16")
        compiled = OperatorPiecewiseCompiler(
            supply, ImplementationPolicy(mt=128, nt=128, cores=32)
        ).compile_operator(op)
        return float(compiled.evaluate(geom).duration_ms)

    d_slow = _compile_duration(100.0)
    d_mid = _compile_duration(200.0)
    d_fast = _compile_duration(300.0)
    assert d_fast < d_mid < d_slow


def test_lower_layer_predictions_preserves_dense_and_lowers_ep_subgraphs():
    from types import SimpleNamespace

    # 1) Default arguments work cleanly without raising.
    assert lower_layer_predictions() == ({}, {}, 0.0, 0.0, 0.0)

    # 2) Single-row F0 call is invariant to model_label and never invents moe_grouped.
    f0_row = SimpleNamespace(
        name="gate_up", kind="F0", mechanism="gemm", total_ms=0.026597, count=1, cost=None
    )
    f0_neu = lower_layer_predictions(
        model_label="neutral", rows=[f0_row], per_layer={"gemm": 0.026597}, body_ms=0.026597
    )
    f0_glm = lower_layer_predictions(
        model_label="GLM-5", rows=[f0_row], per_layer={"gemm": 0.026597}, body_ms=0.026597
    )
    f0_k3 = lower_layer_predictions(
        model_label="Kimi-K3", tp=8, rows=[f0_row], per_layer={"gemm": 0.026597}, body_ms=0.026597
    )
    assert f0_neu == f0_glm == f0_k3
    assert f0_neu[0]["gemm"] == pytest.approx(0.026597)
    assert f0_k3[0].get("moe_grouped", 0.0) == 0.0

    # 3) Full-layer rows lower purely from structural kinds/rows and are label-invariant.
    full_rows = [
        SimpleNamespace(name="rmsnorm", kind="N0", mechanism="norm_residual", total_ms=1.22, count=61, cost=None),
        SimpleNamespace(name="mla_attention", kind="A1", mechanism="attention", total_ms=7.32, count=61, cost=None),
        SimpleNamespace(name="index_score", kind="I0", mechanism="indexer", total_ms=0.61, count=61, cost=None),
        SimpleNamespace(name="router_topk", kind="F1", mechanism="moe_router", total_ms=0.61, count=61, cost=None),
        SimpleNamespace(name="dispatch", kind="F2", mechanism="collective", total_ms=2.44, count=61, cost=None),
        SimpleNamespace(
            name="expert_gate_up",
            kind="F3",
            mechanism="moe_grouped",
            total_ms=12.20,
            count=61,
            cost=SimpleNamespace(residency_bytes=415_236_096, derived=(), work=(), times=()),
        ),
    ]
    per_layer = {
        "attention": 0.12,
        "gemm": 0.08,
        "moe_grouped": 0.20,
        "moe_router": 0.01,
        "collective": 0.04,
        "norm_residual": 0.02,
        "activation": 0.01,
        "kv_cache": 0.01,
        "indexer": 0.01,
    }
    subgraphs = {
        "attention": 0.22,
        "moe": 0.26,
        "norm_residual": 0.02,
        "hyperconnection": 0.0,
    }
    kwargs = dict(
        phase="decode",
        prompt_tokens=4096,
        concurrency=8,
        max_tokens=128,
        tp=1,
        dp=8,
        ep=8,
        graph_buckets=(1, 2, 4, 8),
        graph_bucket=1,
        layers=61,
        rows=full_rows,
        per_layer=dict(per_layer),
        subgraphs=dict(subgraphs),
        body_ms=sum(subgraphs.values()) * 61,
        decode_body_ms=sum(subgraphs.values()) * 61,
        prefill_body_ms=0.0,
    )
    pl_glm, sub_glm, body_glm, dec_glm, pf_glm = lower_layer_predictions(
        model_label="GLM-5", cell_id="glm5_decode", **kwargs
    )
    pl_neu, sub_neu, body_neu, dec_neu, pf_neu = lower_layer_predictions(
        model_label="neutral", cell_id="neutral_cell", **kwargs
    )
    assert (pl_glm, sub_glm, body_glm, dec_glm, pf_glm) == (pl_neu, sub_neu, body_neu, dec_neu, pf_neu)
    assert pl_glm["moe_grouped"] > 0.0
    assert sub_glm["moe"] > 0.0
    assert body_glm > 0.0
    assert dec_glm > 0.0
    assert pf_glm == 0.0


def test_p1_2_p1_3_p1_4_f3_routing_quant_cross_pipe_monotonicity_and_assumptions_invariance():
    from dataclasses import replace
    from mindie_sim.prediction.piecewise import RequestGeometry

    hw = Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950PR_9579", "num_devices": 8})
    hw_no_assump = replace(hw, assumptions=())
    sup_full = PipeSupply.from_hardware(hw)
    sup_no_assump = PipeSupply.from_hardware(hw_no_assump)
    assert sup_full.invocations == sup_no_assump.invocations
    assert sup_full.flops_by_dtype == sup_no_assump.flops_by_dtype

    pol = ImplementationPolicy(mt=64, nt=64, cores=8, moe_row_tile=64, split_target=10**9, max_splits=1)
    # P1-3: A0 prefill (16384 tokens, q=8, kv=1, dk=64, dv=64) with vector at 0.5x nominal
    a0_op = BoundOperator(
        OperatorTemplate(name="attention", family="attention", role="attention"),
        StaticSignature(
            role="attention",
            heads=(("q", 8), ("kv", 1), ("dk", 64), ("dv", 64)),
            quant=(("act_bytes", 2.0),),
            tiles=(("mt", 64), ("nt", 64), ("cores", 8)),
        ),
    )
    g16k = WorkloadGeometry.homogeneous(1, 16384, 16384, phase="prefill")
    r_base = dict(sup_full.rates)
    durs_a0 = []
    for hbm_mult in (0.75, 1.0, 1.5):
        sup_mod = replace(
            sup_full,
            rates=tuple({**r_base, "vector": r_base["vector"] * 0.5, "hbm": r_base["hbm"] * hbm_mult}.items()),
        )
        durs_a0.append(float(OperatorPiecewiseCompiler(sup_mod, pol).compile_operator(a0_op, kind="A0").evaluate(g16k).duration_ms))
    assert durs_a0[0] >= durs_a0[1] >= durs_a0[2]

    # P1-2: F3 prefill quant with explicit routing does not double-localize by local_batch / batch
    q_op = BoundOperator(
        OperatorTemplate(name="activation_quant", family="quantize", role="quantize"),
        StaticSignature(
            role="quantize",
            experts=(("E", 16), ("local", 2), ("k", 2)),
            quant=(("act_bytes", 2.0), ("out_format", "fp8_e4m3")),
            extras=(("elements", 256), ("ops_per_element", 2), ("read_bytes", 512), ("write_bytes", 256)),
            tiles=(("mt", 64), ("cores", 8)),
        ),
    )
    g_routed_b1 = WorkloadGeometry(
        (RequestGeometry(64, 0, phase="prefill"),),
        phase="prefill",
        local_batch=1,
        routing=((0, (32, 32)),),
    )
    g_routed_b4 = WorkloadGeometry(
        tuple(RequestGeometry(64, 0, phase="prefill") for _ in range(4)),
        phase="prefill",
        local_batch=1,
        routing=((0, (32, 32)),),
    )
    comp_q = OperatorPiecewiseCompiler(sup_full, pol).compile_operator(q_op, kind="F3")
    assert comp_q.evaluate(g_routed_b4).duration_ms == comp_q.evaluate(g_routed_b1).duration_ms

    # P1-2 (StateView routing with E == local == 2):
    from mindie_sim.prediction.subgraphs import MoeRoutingState, StateView

    q_op_e2 = BoundOperator(
        OperatorTemplate(name="activation_quant", family="quantize", role="quantize"),
        StaticSignature(
            role="quantize",
            experts=(("E", 2), ("local", 2), ("k", 2)),
            quant=(("act_bytes", 2.0), ("out_format", "fp8_e4m3")),
            extras=(("elements", 256), ("ops_per_element", 2), ("read_bytes", 512), ("write_bytes", 256)),
            tiles=(("mt", 64), ("cores", 8)),
        ),
    )
    comp_q_e2 = OperatorPiecewiseCompiler(sup_full, pol).compile_operator(q_op_e2, kind="F3")
    g_sv_b1 = WorkloadGeometry(
        tuple(RequestGeometry(2048, 0, phase="prefill") for _ in range(4)),
        phase="prefill",
        local_batch=1,
    )
    g_sv_b4 = WorkloadGeometry(
        tuple(RequestGeometry(2048, 0, phase="prefill") for _ in range(4)),
        phase="prefill",
        local_batch=4,
    )
    g_geom_b1 = WorkloadGeometry(
        tuple(RequestGeometry(2048, 0, phase="prefill") for _ in range(4)),
        phase="prefill",
        local_batch=1,
        routing=((0, (8192, 8192)),),
    )
    sv = StateView().with_routing(0, MoeRoutingState((8192, 8192), top_k=2, valid_tokens=g_sv_b1.tokens))
    a_cost = comp_q_e2.evaluate(g_sv_b1, sv)
    b_cost = comp_q_e2.evaluate(g_sv_b4, sv)
    c_cost = comp_q_e2.evaluate(g_geom_b1)
    assert a_cost == b_cost == c_cost




