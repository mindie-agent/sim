"""Piecewise pricing must be structural work plus declared hardware facts.

[Role]
  Focused acceptance tests for the production pricing boundary owned by
  ``families.py``, ``lowering.py`` and ``service_span.py``.

[Invariants]
  * Compiler work is not multiplied by token/batch/cohort/route/graph-bucket
    lookups.  ``HW_SCALE_*`` is identity unless the declared executed matrix
    dtype has a higher peak than the default cube supply.
  * A declared kernel invocation floor is inclusive and applied once; it is not
    a per-family multiple of a nominal vector/cube minimum.
  * ``lower_layer_predictions`` does not move a fabricated indexer/projection
    duration into the norm clock.  Only the compiled R1 fused residual is
    re-attributed, zero-sum.
  * L4 host overhead is a caller-declared duration.  It is independent of
    prompt/topology, and a wave term exists only for a declared chunk budget.
[Timeouts & Locks] Tier 0 in-memory arithmetic, no I/O, device or locks.
[Side-Effect Policy] Read-only construction of immutable records.
"""
from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import pytest

from mindie_sim.prediction.piecewise.compiler import (
    AnalysisDomain, ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
)
from mindie_sim.prediction.piecewise.families import (
    apply_hardware_service, compute_hardware_service_factors,
)
from mindie_sim.prediction.piecewise.expressions import as_fraction
from mindie_sim.prediction.piecewise.geometry import WorkloadGeometry
from mindie_sim.prediction.piecewise.lowering import lower_layer_predictions
from mindie_sim.prediction.piecewise.service_span import (
    service_ttft_overhead_ms, ttft_wave_scale,
)
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature,
)

_DOMAIN = AnalysisDomain(max_tokens=131072, max_batch=64, max_q=131072,
                         max_s_kv=131072)


def _operator(name: str, family: str, **groups) -> BoundOperator:
    return BoundOperator(
        OperatorTemplate(name=name, family=family, role=name),
        StaticSignature(role=name, **groups),
    )


def _compiler() -> tuple[PipeSupply, ImplementationPolicy]:
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950PR_9579", "num_devices": 8,
    })
    return PipeSupply.from_hardware(hardware), ImplementationPolicy()


def _gemm(**quant) -> BoundOperator:
    return _operator(
        "proj", "gemm",
        matrix=(("K", 4096), ("N", 4096)),
        quant=tuple(quant.items()) or (("weight_format", "bf16"),),
    )


def test_structural_work_has_no_regime_or_route_scale_lookup() -> None:
    supply, policy = _compiler()
    compiler = OperatorPiecewiseCompiler(supply, policy)
    base = compiler.compile_operator(_gemm(), kind="F0")
    scales = []
    for tokens in (6000, 8192, 8200):
        env = dict(base.segment(
            WorkloadGeometry.homogeneous(1, tokens, tokens, phase="prefill"),
        ).env)
        scales.append((float(env["HW_SCALE_CUBE"]), float(env["HW_SCALE_HBM"]),
                       float(env["HW_SCALE_VECTOR"]), float(env["HW_SCALE_MTE2"])))
    assert scales == [(1.0, 1.0, 1.0, 1.0)] * 3

    # A routed regime flag cannot change a dense GEMM's physical work scale.
    routed = compiler.compile_operator(
        _gemm(weight_format="bf16", routed_weight_format="mxfp4",
              graph_buckets=(1, 2, 4, 8, 16, 32, 64)), kind="F0",
    )
    env = dict(routed.segment(
        WorkloadGeometry.homogeneous(1, 8192, 8192, phase="prefill"),
    ).env)
    assert env["HW_SCALE_CUBE"] == 1 and env["HW_SCALE_HBM"] == 1


def test_declared_quantized_peak_is_the_only_cube_promotion() -> None:
    supply, policy = _compiler()
    peaks = dict(supply.flops_by_dtype)
    bf16, int8 = peaks["bf16"], peaks["int8"]
    compiler = OperatorPiecewiseCompiler(supply, policy)
    geometry = WorkloadGeometry.homogeneous(1, 4096, 4096, phase="prefill")

    def cube_scale(**quant) -> float:
        op = compiler.compile_operator(_gemm(**quant), kind="F0")
        return float(dict(op.segment(geometry).env)["HW_SCALE_CUBE"])

    assert cube_scale(weight_format="bf16", act_format="bf16") == 1.0
    assert cube_scale(weight_format="fp32", act_format="fp32") == pytest.approx(
        dict(supply.flops_by_dtype)["fp32"] / bf16)
    assert cube_scale(weight_format="int8", act_format="int8") == pytest.approx(int8 / bf16)
    assert cube_scale(weight_format="mxfp4", act_format="fp8_e4m3") == pytest.approx(int8 / bf16)
    # W4A16 dequantises to the activation rate, so the default peak stands.
    assert cube_scale(weight_format="mxfp4", act_format="bf16") == 1.0


def test_invocation_floor_is_inclusive_once_not_additive() -> None:
    supply = PipeSupply(
        (("cube", 1.0e6), ("vector", 1.0e6), ("hbm", 1.0e6), ("mte2", 1.0e6)),
        invocations=(("cube", 0.01),),
        resource_units=(("compute_units", 32), ("vector", 64)),
        flops_by_dtype=(("bf16", 1.0e6), ("int8", 2.0e6)),
    )
    op = SimpleNamespace(
        supply=supply, family="gemm", name="proj", kind="F0",
        operator=SimpleNamespace(theta=SimpleNamespace(find=lambda n, d=None: {
            "weight_format": "bf16", "act_format": "bf16",
        }.get(n, d))),
        _raw_stages=None,
    )
    geometry = SimpleNamespace(tokens=1)

    def duration(work: Fraction) -> Fraction:
        stages, times = [{"cube": work}], [work / Fraction(1_000_000)]
        apply_hardware_service(
            op, geometry, {}, stages, times,
            {"cube": work}, {"cube": times[0]}, {"cube": times[0]},
        )
        return times[-1]

    # Body 0.001 ms is below the 0.01 ms floor: duration is the floor only.
    assert duration(Fraction(1000)) == Fraction(1, 100)
    # 0.1 ms body exceeds the floor and pays no extra launch multiple.
    assert duration(Fraction(100000)) == Fraction(1, 10)


def test_startup_is_a_causal_maximum_not_an_extra_stack() -> None:
    supply = PipeSupply(
        (("cube", 1.0e6), ("vector", 1.0e6)),
        invocations=(("cube", 0.01), ("cube_startup_ms", 0.02)),
        resource_units=(("compute_units", 32), ("vector", 64)),
        flops_by_dtype=(("bf16", 1.0e6),),
    )
    op = SimpleNamespace(
        supply=supply, family="gemm", name="proj", kind="F0",
        operator=SimpleNamespace(theta=SimpleNamespace(find=lambda n, d=None: {
            "weight_format": "bf16", "act_format": "bf16",
        }.get(n, d))),
        _raw_stages=None,
    )
    stages, times = [{"cube": Fraction(1000)}], [Fraction(1, 1000)]
    apply_hardware_service(
        op, SimpleNamespace(tokens=1), {}, stages, times,
        {"cube": Fraction(1000)}, {"cube": times[0]}, {"cube": times[0]},
    )
    assert times[-1] == Fraction(21, 1000)


def test_elementwise_uses_one_declared_vector_minimum() -> None:
    supply = PipeSupply(
        (("cube", 1.0e6), ("vector", 1.0e6)),
        invocations=(("vector", 0.002),),
        resource_units=(("compute_units", 32), ("vector", 64)),
        flops_by_dtype=(("bf16", 1.0e6),),
    )
    op = SimpleNamespace(
        supply=supply, family="elementwise", name="silu", kind="F3",
        operator=SimpleNamespace(theta=SimpleNamespace(find=lambda n, d=None: d)),
        _raw_stages=None,
    )
    stages, times = [{"vector": Fraction(1)}], [Fraction(1, 1_000_000)]
    apply_hardware_service(
        op, SimpleNamespace(tokens=1), {}, stages, times,
        {"vector": Fraction(1)}, {"vector": times[0]}, {"vector": times[0]},
    )
    assert times[-1] == Fraction(2, 1000)


def test_collective_latency_uses_physical_rounds_only() -> None:
    supply = PipeSupply(
        (("link", 1.0e6), ("vector", 1.0e6)),
        invocations=(("link_latency_ms", 0.002),),
        resource_units=(("compute_units", 32), ("vector", 64)),
        flops_by_dtype=(),
    )

    def span(name: str, ranks: int) -> Fraction:
        theta = SimpleNamespace(
            find=lambda key, default=None: {
                "ranks": ranks, "members": ranks, "ep": ranks, "tp": ranks,
            }.get(key, default)
        )
        op = SimpleNamespace(
            supply=supply, family="collective", name=name, kind="F4",
            operator=SimpleNamespace(theta=theta), _raw_stages=None,
        )
        env: dict = {}
        _scale, _floor, _target = compute_hardware_service_factors(
            op, SimpleNamespace(tokens=1024), env, {"link": Fraction(1.0e6)},
        )
        return env["HW_DEPENDENCY_MS"]

    assert span("allreduce", 8) == Fraction(28, 1000)   # 2 * (8 - 1) rounds
    assert span("all_gather", 8) == Fraction(14, 1000)  # 8 - 1 rounds
    assert span("dispatch", 8) == Fraction(2, 1000)     # one concurrent exchange
    assert span("allreduce", 1) == 0


def test_lowering_does_not_move_i0_time_between_clocks() -> None:
    rows = (
        SimpleNamespace(name="index_proj", kind="I0", mechanism="indexer",
                        total_ms=0.5, count=60, cost=None),
        SimpleNamespace(name="mla_attention", kind="A1", mechanism="attention",
                        total_ms=4.0, count=60, cost=None),
        SimpleNamespace(name="attn_norm", kind="N0", mechanism="norm_residual",
                        total_ms=0.2, count=60, cost=None),
    )
    per_layer = {"indexer": 0.5, "attention": 4.0, "norm_residual": 0.2,
                 "gemm": 0.3}
    for phase, prompt in (("decode", 1), ("prefill", 4096)):
        adj, sub, *_ = lower_layer_predictions(
            phase=phase, prompt_tokens=prompt, layers=60, rows=rows,
            per_layer=dict(per_layer), subgraphs={},
        )
        assert adj["indexer"] == pytest.approx(0.5)
        assert adj["gemm"] == pytest.approx(0.3)
        assert adj["norm_residual"] == pytest.approx(0.2)
        assert sub["attention"] == pytest.approx(4.5)
        assert sum(adj.values()) == pytest.approx(sum(per_layer.values()))


def test_host_overhead_is_declared_not_synthesized() -> None:
    for prompt in (1, 1024, 2048, 4095):
        for tp, ep in ((1, 1), (8, 2), (2, 32)):
            assert service_ttft_overhead_ms(
                prompt_tokens=prompt, tp=tp, ep=ep, host_overhead_ms=12.5,
            ) == pytest.approx(12.5)
    assert service_ttft_overhead_ms(prompt_tokens=1024, tp=8, ep=32) == 0.0
    with pytest.raises(ValueError):
        service_ttft_overhead_ms(prompt_tokens=1024, host_overhead_ms=-1.0)


def test_wave_scale_is_mean_request_finish_not_chunk_average() -> None:
    # Single request must finish its whole body, including both 8192 chunks.
    assert ttft_wave_scale(
        phase="prefill", concurrency=1, dp=1, ep=1, layers=1,
        graph_buckets=(), prompt_tokens=16384, max_batched_tokens=8192,
    ) == 1.0
    # The body is already the local batch, so concurrent requests finish
    # together; FCFS queueing is charged by chunked_prefill_mean_ttft_ms.
    assert ttft_wave_scale(
        phase="prefill", concurrency=8, dp=4, ep=16, layers=1,
        graph_buckets=(), prompt_tokens=8192, max_batched_tokens=8192,
    ) == 1.0
    assert ttft_wave_scale(
        phase="decode", concurrency=8, dp=4, ep=16, layers=1,
        graph_buckets=(1, 2, 4, 8), prompt_tokens=8192,
    ) == 1.0


def test_fcfs_request_finish_schedule_is_per_request() -> None:
    from mindie_sim.prediction.piecewise.service_span import (
        chunked_prefill_mean_ttft_ms,
    )

    def per_chunk(_take: int, _done: int) -> float:
        return 1.0

    # C2/DP1: both 2-chunk requests queue on one rank; finishes 2 and 4.
    assert chunked_prefill_mean_ttft_ms(
        [16384, 16384], dp=1, max_batched_tokens=8192, body_one=per_chunk,
    ) == pytest.approx(3.0)
    # C2/DP2: one request per rank; each finishes its own two chunks.
    assert chunked_prefill_mean_ttft_ms(
        [16384, 16384], dp=2, max_batched_tokens=8192, body_one=per_chunk,
    ) == pytest.approx(2.0)
    # Unequal DP partition C3/DP2: rank0 has two requests, rank1 one.
    assert chunked_prefill_mean_ttft_ms(
        [16384, 16384, 16384], dp=2, max_batched_tokens=8192,
        body_one=per_chunk,
    ) == pytest.approx(8.0 / 3.0)


def test_single_request_chunk_chain_charges_both_chunks() -> None:
    from mindie_sim.prediction.piecewise.service_span import (
        chunked_prefill_mean_ttft_ms,
    )

    seen: list[tuple[int, int]] = []

    def body(take: int, done: int) -> float:
        seen.append((take, done))
        return 0.0 if done < 16384 else 1.0

    result = chunked_prefill_mean_ttft_ms(
        [16384], dp=1, max_batched_tokens=8192, body_one=body)
    assert seen == [(8192, 8192), (8192, 16384)]
    assert result == 1.0


# ---------------------------------------------------------------------------
# P0 counterfactuals on actual catalog F4 operators.
# ---------------------------------------------------------------------------

from dataclasses import replace
from functools import lru_cache

from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.subgraphs import FeatureConfig, instantiate_model
from mindie_sim.prediction.piecewise.families import extract_hardware_meta, _declared_cube_rate


@lru_cache(maxsize=4)
def _actual_f4(tp: int, dp: int, ep: int):
    model = load_model("deepseek-v4.1-flash")
    instantiated = instantiate_model(
        model, FeatureConfig.from_model(model, tp=tp, dp=dp, ep=ep))
    return next(s for s in instantiated.subgraphs
                if s.kind == "F4" and dict(s.metadata).get("role") == "shared_ffn")


def _actual_9582(devices: int = 16):
    return Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950DT_9582",
        "num_devices": devices, "ranks_per_node": 8,
    })


def _compiled_f4(tp: int = 8, dp: int = 2, ep: int = 16, devices: int = 16):
    supply = PipeSupply.from_hardware(_actual_9582(devices))
    return OperatorPiecewiseCompiler(supply, ImplementationPolicy()).compile(
        _actual_f4(tp, dp, ep)), supply


def test_declared_peak_promotes_rate_and_never_rewrites_native_work() -> None:
    compiled, supply = _compiled_f4()
    geometry = WorkloadGeometry.homogeneous(1, 4096, 4096, phase="prefill")
    gate_up = next(o for o in compiled.operators if o.name == "shared_gate_up")
    baseline = gate_up.evaluate(geometry)
    native = baseline.pipe_work("cube")
    assert native == 24_159_191_040

    doubled = dict(supply.flops_by_dtype)
    doubled["fp8_e4m3"] *= 2.0
    faster = replace(supply, flops_by_dtype=tuple(sorted(doubled.items())))
    gate_up_fast = next(
        o for o in OperatorPiecewiseCompiler(faster, ImplementationPolicy())
        .compile(_actual_f4(8, 2, 16)).operators if o.name == "shared_gate_up")
    promoted = gate_up_fast.evaluate(geometry)
    assert promoted.pipe_work("cube") == native
    # A faster declared peak cannot lengthen the same native operator.
    assert promoted.duration_ms <= baseline.duration_ms
    # When the supplied base is already the executed dtype, rate is identity.
    fp8_base = max(float(v) for k, v in doubled.items() if k == "fp8_e4m3")
    rates = tuple((name, fp8_base if name == "cube" else rate)
                  for name, rate in faster.rates)
    already_fp8 = replace(faster, rates=rates)
    assert _declared_cube_rate(
        SimpleNamespace(supply=already_fp8, family="gemm",
                        operator=gate_up_fast.operator),
        {"cube": native}) == 1


def test_activation_floor_is_nonshared_latency_not_vector_work() -> None:
    compiled, _supply = _compiled_f4()
    geometry = WorkloadGeometry.homogeneous(8, 1, 4096, phase="decode")
    activation = next(o for o in compiled.operators if o.name == "shared_activation")
    cost = activation.evaluate(geometry)
    # Native elementwise work is unchanged; no 8.8M fabricated vector ops.
    assert cost.pipe_work("vector") == 9_216
    assert cost.pipe_work("hbm") == 13_824
    assert cost.duration_ms == Fraction(1312, 1_000_000)
    assert dict(cost.derived)["service_floor_ms"] == Fraction(1312, 1_000_000)

    # Two concurrent copies complete at the floor, not twice the floor and
    # not native body work alone.
    from mindie_sim.prediction.piecewise.composition import contention_envelope
    branches = {"a": [list(stage) for stage in cost.stage_work],
                "b": [list(stage) for stage in cost.stage_work]}
    floor = dict(cost.derived)["service_floor_ms"]
    envelope = contention_envelope(
        branches, activation.supply, {"a": floor, "b": floor})
    assert envelope.makespan_ms == Fraction(1312, 1_000_000)
    assert dict(envelope.branches)["a"] == Fraction(1312, 1_000_000)


def test_intra_only_tp_allreduce_ignores_idle_external_node() -> None:
    geometry = WorkloadGeometry.homogeneous(1, 2048, 2048, phase="prefill")
    durations = []
    for devices in (8, 16):
        compiled, _supply = _compiled_f4(tp=4, dp=1, ep=4, devices=devices)
        op = next(o for o in compiled.operators if o.name == "tp_allreduce")
        cost = op.evaluate(geometry)
        assert cost.pipe_work("link_hccs") == 31_457_280
        assert cost.pipe_work("link_roce") == 0
        assert cost.pipe_work("link") == 31_457_280
        assert "link" not in dict(cost.work)
        durations.append(cost.duration_ms)
    assert durations[0] == durations[1]
    assert float(durations[0]) == pytest.approx(0.09104816326530613)


def test_extract_hardware_meta_keeps_peaks_and_routes_without_invocations() -> None:
    hardware = _actual_9582()
    bare = replace(hardware, dies=(
        replace(hardware.dies[0], kernel_invocations=()),))
    invocations, _units, peaks = extract_hardware_meta(bare, 1e-3)
    inv = dict(invocations)
    assert dict(peaks)["fp8_e4m3"] == 865_075_200_000_000.0 * 1e-3
    assert inv["link_rate_intra_bytes_per_ms"] == 392_000_000.0
    assert inv["link_rate_inter_bytes_per_ms"] == 50_000_000.0
    assert inv["link_latency_intra_ms"] < inv["link_latency_inter_ms"]


def test_execution_dtype_is_activation_not_fastest_operand() -> None:
    class _Theta:
        def __init__(self, values):
            self.values = values

        def find(self, name, default=None):
            return self.values.get(name, default)

    class _Op:
        family = "gemm"
        supply = SimpleNamespace(
            effective=lambda pipe: 432_537_600_000.0,
            flops_by_dtype=(("bf16", 432_537_600_000.0),
                            ("fp8_e4m3", 865_075_200_000.0)),
        )

    # W8A16 dequantises to the BF16 activation, even though the FP8 weight
    # peak is higher; the fastest operand is not the MMAD execution dtype.
    w8a16 = _Op()
    w8a16.operator = SimpleNamespace(theta=_Theta({
        "weight_format": "fp8_e4m3", "act_format": "bf16"}))
    assert _declared_cube_rate(w8a16, {"cube": Fraction(1)}) == 1

    mxfp8 = _Op()
    mxfp8.operator = SimpleNamespace(theta=_Theta({
        "weight_format": "mxfp4", "act_format": "fp8_e4m3"}))
    assert _declared_cube_rate(mxfp8, {"cube": Fraction(1)}) == 2

    # A declared actual execution dtype below the supplied base is legal:
    # BF16 activation on an already-FP8 supply runs at half the base rate.
    bf16 = _Op()
    bf16.supply = SimpleNamespace(
        effective=lambda pipe: 865_075_200_000.0,
        flops_by_dtype=(("bf16", 432_537_600_000.0),
                        ("fp8_e4m3", 865_075_200_000.0)),
    )
    bf16.operator = SimpleNamespace(theta=_Theta({
        "weight_format": "fp8_e4m3", "act_format": "bf16"}))
    assert _declared_cube_rate(bf16, {"cube": Fraction(1)}) == Fraction(1, 2)


def test_zero_body_declared_kernel_keeps_its_startup_and_floor() -> None:
    supply = PipeSupply(
        (("cube", 1.0e6), ("vector", 1.0e6), ("link", 1.0e6)),
        invocations=(("empty_aic", 0.000439),
                     ("empty_aic_startup_ms", 0.0001),
                     ("vector", 0.001312)),
        resource_units=(("compute_units", 32), ("vector", 64)),
        flops_by_dtype=(("bf16", 1.0e6),),
    )

    class _Theta:
        def find(self, name, default=None):
            return {"invocation_class": "empty_aic"}.get(name, default)

    op = SimpleNamespace(
        supply=supply, family="collective", name="notify", kind="F2",
        operator=SimpleNamespace(theta=_Theta()), _raw_stages=None,
    )
    env: dict = {}
    _scale, floor, _target = compute_hardware_service_factors(
        op, SimpleNamespace(tokens=1), env, {})
    assert floor == Fraction(439, 1_000_000)
    assert env["HW_STARTUP_MS"] == Fraction(1, 10_000)


def test_collective_group_uses_bound_members_not_max_axes() -> None:
    supply = PipeSupply(
        (("link", 1.0e6), ("vector", 1.0e6)),
        invocations=(("link_latency_intra_ms", 0.002),
                     ("link_latency_inter_ms", 0.005)),
        resource_units=(("compute_units", 32), ("vector", 64)),
        flops_by_dtype=(),
    )

    def dependency(name: str, values: dict[str, int]) -> Fraction:
        class _Theta:
            def find(self, key, default=None):
                return values.get(key, default)

        op = SimpleNamespace(
            supply=supply, family="collective", name=name, kind="F4",
            operator=SimpleNamespace(theta=_Theta()), _raw_stages=None,
        )
        env: dict = {}
        compute_hardware_service_factors(
            op, SimpleNamespace(tokens=1), env, {"link": Fraction(1.0e6)})
        return env["HW_DEPENDENCY_MS"]

    # Bound TP group is 4 even though unrelated EP/DP axes are larger.
    assert dependency("tp_allreduce", {
        "members": 4, "tp": 8, "ep": 16, "dp": 2, "ranks_per_node": 8,
    }) == Fraction(6) * Fraction(2, 1000)
    # Without members, a TP allreduce uses TP=8, not EP=16.
    assert dependency("tp_allreduce", {
        "tp": 8, "ep": 16, "dp": 2, "ranks_per_node": 8,
    }) == Fraction(14) * Fraction(2, 1000)
    # Cross-node TP group: intra and inter rounds use their own latencies.
    assert dependency("tp_allreduce", {
        "members": 16, "ranks_per_node": 8,
    }) == Fraction(14) * Fraction(2, 1000) + Fraction(2) * Fraction(5, 1000)


def test_serial_floors_add_and_concurrent_floors_do_not() -> None:
    from mindie_sim.prediction.piecewise.composition import contention_envelope

    supply = PipeSupply(
        (("vector", 1.0e6), ("cube", 1.0e6)),
        invocations=(("vector", 0.001312),),
        resource_units=(("compute_units", 32), ("vector", 64)),
        flops_by_dtype=(("bf16", 1.0e6),),
    )
    stages = [[("vector", Fraction(1))]]
    one = contention_envelope({"a": stages}, supply)
    assert one.makespan_ms == Fraction(1312, 1_000_000)
    # Two serial copies in one branch pay two floors.
    serial = contention_envelope(
        {"a": stages + stages}, supply,
        {"a": 2 * Fraction(1312, 1_000_000)})
    assert serial.makespan_ms == Fraction(2624, 1_000_000)
    # Nested concurrent groups: each two-copy branch has the serial floor and
    # the makespan is the slower branch, not the sum of all four.
    nested = contention_envelope(
        {"a": stages + stages, "b": stages + stages}, supply,
        {"a": 2 * Fraction(1312, 1_000_000),
         "b": 2 * Fraction(1312, 1_000_000)})
    assert nested.makespan_ms == Fraction(2624, 1_000_000)


def test_bf16_execution_under_fp8_supply_slows_elapsed_not_work() -> None:
    base, policy = _compiler()
    fp8 = dict(base.flops_by_dtype)["fp8_e4m3"]
    rates = tuple((name, fp8 if name == "cube" else rate)
                  for name, rate in base.rates)
    supply = PipeSupply(
        rates, base.structure_caps, invocations=base.invocations,
        resource_units=base.resource_units, flops_by_dtype=base.flops_by_dtype,
        undeclared_pipes=base.undeclared_pipes,
    )
    op = OperatorPiecewiseCompiler(supply, policy).compile_operator(
        _gemm(weight_format="bf16", act_format="bf16"), kind="F0")
    geometry = WorkloadGeometry.homogeneous(1, 4096, 4096, phase="prefill")
    env = dict(op.segment(geometry).env)
    assert env["HW_SCALE_CUBE"] == Fraction(1, 2)
    # Native cube work is untouched by the slower execution dtype.
    assert op.evaluate(geometry).pipe_work("cube") == 2 * 4096 * 4096 * 4096


def test_mte1_rate_does_not_change_an_hbm_bound_operator() -> None:
    compiled, supply = _compiled_f4()
    geometry = WorkloadGeometry.homogeneous(1, 4096, 4096, phase="prefill")
    op = next(o for o in compiled.operators if o.name == "shared_gate_up")
    base = op.evaluate(geometry)
    rates = tuple((name, rate * 2.0 if name == "mte1" else rate)
                  for name, rate in supply.rates)
    faster_mte1 = PipeSupply(
        rates, supply.structure_caps, invocations=supply.invocations,
        resource_units=supply.resource_units,
        flops_by_dtype=supply.flops_by_dtype,
        undeclared_pipes=supply.undeclared_pipes,
    )
    op2 = next(o for o in OperatorPiecewiseCompiler(faster_mte1, ImplementationPolicy())
               .compile(_actual_f4(8, 2, 16)).operators if o.name == "shared_gate_up")
    assert base.pipe_work("hbm") > base.pipe_work("mte1")
    assert op2.evaluate(geometry).duration_ms == base.duration_ms


def test_q2048_dispatch_step_is_one_mtu_frame_per_peer() -> None:
    supply, policy = _compiler()
    op = _operator(
        "dispatch", "transport",
        parallel=(("members", 16), ("ep", 16), ("tp", 8),
                  ("ranks_per_node", 8)),
        extras=(("wire_bytes", 8192.0), ("row_bytes", 4096.0),
                ("top_k", 8), ("routed_weight_format", "mxfp4"),
                ("kv_bf16", 1)),
    )
    compiled = OperatorPiecewiseCompiler(supply, policy).compile_operator(op)
    # Declared source: T tokens, top_k=8, EP=16, one 4096-byte routed row.
    # One source owns T*8/16 rows; each of 16 destinations gets T*128 bytes.
    # This is the declared uniform split, not an observed real-route transfer
    # count; a real routed histogram would declare its own directed cells.
    mtu, ep, top_k, row_bytes = 4096, 16, 8, 4096

    def source_send(tokens: int) -> int:
        source_rows = Fraction(tokens * top_k, ep)
        per_peer = source_rows * row_bytes / ep
        frames = -(-per_peer // mtu)
        return (ep - 1) * frames * mtu

    at_2047 = compiled.evaluate(WorkloadGeometry.homogeneous(1, 2047, 2047, phase="prefill"))
    at_2048 = compiled.evaluate(WorkloadGeometry.homogeneous(1, 2048, 2048, phase="prefill"))
    at_2049 = compiled.evaluate(WorkloadGeometry.homogeneous(1, 2049, 2049, phase="prefill"))
    # Both 2047 and 2048 are inside the final 64-frame packet, so the delta is
    # zero.  T=2049 crosses to 65 frames on each of the 15 remote peers.
    assert Fraction(at_2048.get("routed_rows")) == 2048 * top_k
    assert Fraction(at_2048.get("peer_bytes")) == 2048 * top_k * row_bytes
    assert Fraction(at_2048.get("source_routed_rows")) == Fraction(2048 * top_k, ep)
    assert Fraction(at_2048.get("per_peer_useful_bytes")) == Fraction(2048 * top_k * row_bytes, ep * ep)
    assert Fraction(at_2048.get("send_bytes")) == source_send(2048) == 3_932_160
    assert Fraction(at_2049.get("send_bytes")) == source_send(2049) == 3_993_600
    # Per-source packet work is 15 framed remote shares; the group ledger is
    # EP times that framed source ledger, never a group-total mistaken unit.
    assert Fraction(at_2048.get("send_bytes")) == (ep - 1) * 64 * mtu
    assert Fraction(at_2049.get("aggregate_wire_bytes")) == (
        ep * source_send(2049))
    assert at_2048.pipe_work("link") - at_2047.pipe_work("link") == 0
    assert at_2049.pipe_work("link") - at_2048.pipe_work("link") == 61_440
    assert Fraction(at_2048.get("send_bytes")) - Fraction(
        at_2047.get("send_bytes")) == 0
    assert Fraction(at_2049.get("send_bytes")) - Fraction(
        at_2048.get("send_bytes")) == 61_440
    assert Fraction(at_2049.get("aggregate_wire_bytes")) - Fraction(
        at_2048.get("aggregate_wire_bytes")) == 16 * 61_440
    assert at_2048.duration_ms - at_2047.duration_ms == 0
    assert at_2049.duration_ms - at_2048.duration_ms == Fraction(
        61_440, 392_000_000)


def test_ridge_coordinates_are_identity_not_fitted_ratios() -> None:
    compiled, _supply = _compiled_f4()
    geometry = WorkloadGeometry.homogeneous(1, 2048, 2048, phase="prefill")
    gate_up = next(o for o in compiled.operators if o.name == "shared_gate_up")
    env = dict(gate_up.segment(geometry).env)
    assert env["CM_RIDGE_SHIFT"] == 1
    assert env["RIDGE_EFFECTIVE_TOKENS"] == 2048
    assert env["VEC_EFFECTIVE_TOKENS"] == 2048


def test_declared_fp8_peak_below_bf16_slows_cube_only() -> None:
    compiled, supply = _compiled_f4()
    geometry = WorkloadGeometry.homogeneous(1, 4096, 4096, phase="prefill")
    op = next(o for o in compiled.operators if o.name == "shared_gate_up")
    baseline = op.evaluate(geometry)
    native = baseline.pipe_work("cube")

    bf16 = dict(supply.flops_by_dtype)["bf16"]
    slower = dict(supply.flops_by_dtype)
    slower["fp8_e4m3"] = bf16 / 4
    slow_supply = replace(supply, flops_by_dtype=tuple(sorted(slower.items())))
    slow_op = next(
        o for o in OperatorPiecewiseCompiler(slow_supply, ImplementationPolicy())
        .compile(_actual_f4(8, 2, 16)).operators if o.name == "shared_gate_up")
    slow = slow_op.evaluate(geometry)
    assert slow.pipe_work("cube") == native
    assert slow.duration_ms > baseline.duration_ms
    # Cube elapsed is native work / (base * declared peak/bf16 * declared
    # active-core/wave cap); native work is conserved.
    env = dict(slow_op.segment(geometry).env)
    effective = Fraction(bf16) * Fraction(1, 4) * as_fraction(env["HW_RATE_CUBE"])
    cube_time = dict(slow.times)["cube"]
    assert cube_time == Fraction(native) / effective


@lru_cache(maxsize=1)
def _glm_a3_case():
    import json
    from pathlib import Path
    from mindie_sim.prediction.structure import parse_case
    path = Path("examples/cases/high-concurrency-20261030/glm-kda-accepted-conditional/glm53flash-a3-b4-q6.json")
    return parse_case(json.loads(path.read_text()))


def _glm_a3_ops():
    case = _glm_a3_case()
    supply = PipeSupply.from_hardware(case.machine, dtype="int8")
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    instantiated = instantiate_model(case.model, features)
    by_kind = {"F2": [], "F4": []}
    for subgraph in instantiated.subgraphs:
        if subgraph.kind not in by_kind:
            continue
        compiled = OperatorPiecewiseCompiler(supply, ImplementationPolicy()).compile(subgraph)
        by_kind[subgraph.kind].extend(compiled.operators)
    return by_kind, supply


def test_ring_reduce_charges_per_step_vector_floor() -> None:
    ops, _supply = _glm_a3_ops()
    geometry = WorkloadGeometry.homogeneous(4, 6, 4096, phase="decode")
    op = next(o for o in ops["F4"] if o.name == "tp_allreduce")
    cost = op.evaluate(geometry)
    env = dict(op.segment(geometry).env)
    # 2*(16-1)=30 serial ring steps; each step is max(link 0.8us, vector 2.5us).
    assert env["HW_STAGE_ROUNDS"] == 30
    assert env["HW_REDUCE_ROUNDS"] == 15
    assert env["HW_GATHER_ROUNDS"] == 15
    assert env["HW_DEPENDENCY_MS"] == Fraction(75, 1000)
    assert cost.duration_ms == Fraction(94179, 1_225_000)
    assert cost.pipe_work("link_hccs") == 368_640


def test_per_peer_transport_steps_use_declared_group_and_floor() -> None:
    ops, _supply = _glm_a3_ops()
    geometry = WorkloadGeometry.homogeneous(4, 6, 4096, phase="decode")
    for name in ("dispatch", "combine"):
        op = next(o for o in ops["F2"] if o.name == name)
        cost = op.evaluate(geometry)
        env = dict(op.segment(geometry).env)
        assert env["HW_PEER_STEPS"] == 15
        # One rpn16 node: one package peer at 0.8us and 14 switch-route
        # peers at 2.5us.  The package latency is not the node route.
        assert env["HW_DEPENDENCY_MS"] == (
            Fraction(8, 10_000) + 14 * Fraction(25, 10_000))
        assert cost.duration_ms >= env["HW_DEPENDENCY_MS"]


def test_single_node_transport_ignores_unused_inter_latency() -> None:
    case = _glm_a3_case()
    supply = PipeSupply.from_hardware(case.machine, dtype="int8")
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    instantiated = instantiate_model(case.model, features)
    subgraph = next(s for s in instantiated.subgraphs
                    if s.kind == "F2" and "dispatch" in {o.name for o in s.operators})
    geometry = WorkloadGeometry.homogeneous(4, 6, 4096, phase="decode")
    base = OperatorPiecewiseCompiler(supply, ImplementationPolicy()).compile(subgraph)
    op = next(o for o in base.operators if o.name == "dispatch")
    base_cost = op.evaluate(geometry)

    inv = dict(supply.invocations)
    inv["link_latency_inter_ms"] = 80.0
    starved = PipeSupply(
        supply.rates, supply.structure_caps, invocations=tuple(sorted(inv.items())),
        resource_units=supply.resource_units, flops_by_dtype=supply.flops_by_dtype,
        undeclared_pipes=supply.undeclared_pipes,
    )
    other = OperatorPiecewiseCompiler(starved, ImplementationPolicy()).compile(subgraph)
    cost = next(o for o in other.operators if o.name == "dispatch").evaluate(geometry)
    assert cost.duration_ms == base_cost.duration_ms
    assert cost.pipe_work("link") == base_cost.pipe_work("link")


def test_collective_stage_count_is_geometry_independent_and_work_scales() -> None:
    supply = PipeSupply(
        (("link_hccs", 196_000_000.0), ("link", 196_000_000.0),
         ("vector", 1.0e12), ("hbm", 1.0e12)),
        invocations=(("link_latency_intra_ms", 0.0008),
                     ("link_latency_inter_ms", 0.0008),
                     ("vector", 0.0025)),
        resource_units=(("compute_units", 16), ("vector", 64)),
        flops_by_dtype=(),
    )
    op = _operator(
        "allreduce", "collective",
        parallel=(("members", 16), ("ranks_per_node", 16)),
        extras=(("wire_bytes", 8192.0), ("collective", "allreduce")),
    )
    compiled = OperatorPiecewiseCompiler(supply, ImplementationPolicy()).compile_operator(op)
    results = {}
    for tokens in (256, 8064):
        geometry = WorkloadGeometry.homogeneous(1, tokens, tokens, phase="prefill")
        cost = compiled.evaluate(geometry)
        env = dict(compiled.segment(geometry).env)
        results[tokens] = (cost, env)
        assert env["HW_STAGE_ROUNDS"] == 30
        assert env["HW_REDUCE_ROUNDS"] == 15
        assert env["HW_GATHER_ROUNDS"] == 15
        assert env["HW_DEPENDENCY_MS"] == Fraction(75, 1000)
    small, large = results[256][0], results[8064][0]
    assert large.pipe_work("link") > small.pipe_work("link")
    assert large.duration_ms > small.duration_ms
