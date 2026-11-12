"""Counterfactual generalization, monotonicity, and dynamic regime knot shifting.

[Role] Verify that piecewise predictions generalize smoothly and monotonically
  across model dimensions (hidden_size, intermediate_size, num_experts), core
  counts (AIC num_cores and AIV aiv_count), and all 8 hardware supply pipes
  (cube, vector, hbm, mte1, mte2, mte3, fixpipe, link), and that piecewise
  regime crossover knots shift physically when compute or bandwidth changes.
[Invariants] Pure in-memory arithmetic assertions; no empirical timing tables,
  fabricated hardware mocks, or silent fallbacks.
[Timeouts & Locks] Tier 0 (<5s), pure CPU arithmetic, no I/O or locks.
[Side-Effect Policy] Read-only catalog and in-memory model construction only.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.piecewise import (
    Const, CoreWaveGuard, ImplementationPolicy, OperatorPiecewiseCompiler,
    PipeDemand, PipeDominanceGuard, PipeSupply, RequestGeometry, Var,
    WorkloadGeometry,
)
from mindie_sim.prediction.piecewise.composition import evaluate_serial_split
from mindie_sim.prediction.piecewise.model_pipeline import _PIPELINE_DOMAIN, ModelPipeline, assemble_model
from mindie_sim.prediction.subgraphs import FeatureConfig, StateView, instantiate_model
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Device, Hardware
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.subgraphs import BoundOperator, OperatorTemplate, StaticSignature


def _compact_v41_model(
    *,
    hidden_size: int = 5120,
    num_experts: int = 192,
    top_k: int = 8,
) -> ModelSpec:
    """Build a compact 2-layer DeepSeek-V4.1-Flash structural model for sweeps."""
    base = load_model("deepseek-v4.1-flash")
    scale = hidden_size / 5120.0
    inter = max(256, int(round(base.intermediate_size * scale / 64.0)) * 64)
    moe_inter = max(128, int(round(base.moe_intermediate_size * scale / 64.0)) * 64)
    plans = []
    for new_idx, old_idx in enumerate((0, 3)):
        plan = base.layer_plans[old_idx]
        shared_inter = max(128, int(round((plan.shared_intermediate_size or moe_inter) * scale / 64.0)) * 64)
        plans.append(
            replace(
                plan,
                index=new_idx,
                kv_owner_layer_id=new_idx,
                owns_long_kv=True,
                intermediate_size=inter,
                num_experts=num_experts if plan.ffn == "moe" else 0,
                top_k=top_k if plan.ffn == "moe" else 0,
                moe_intermediate_size=moe_inter if plan.ffn == "moe" else 0,
                shared_intermediate_size=shared_inter if plan.ffn == "moe" else 0,
            )
        )
    return replace(
        base,
        hidden_size=hidden_size,
        intermediate_size=inter,
        num_experts=num_experts,
        top_k=top_k,
        moe_intermediate_size=moe_inter,
        layers=len(plans),
        body_layers=len(plans),
        layer_plans=tuple(plans),
    )


def _base_supply(num_devices: int = 8, **hw_overrides) -> PipeSupply:
    mapping = {
        "backend": "ascend",
        "sku": "Ascend950PR_9579",
        "num_devices": num_devices,
        **hw_overrides,
    }
    return PipeSupply.from_hardware(Hardware.from_mapping(mapping))


def _scale_supply(
    base: PipeSupply,
    *,
    factor: float = 1.0,
    pipes: tuple[str, ...] | None = None,
) -> PipeSupply:
    target = set(pipes) if pipes is not None else {k for k, _ in base.rates}
    rates = tuple(
        (k, float(v) * factor if k in target else float(v))
        for k, v in base.rates
    )
    flops = tuple(
        (k, float(v) * factor if "cube" in target else float(v))
        for k, v in base.flops_by_dtype
    )
    return PipeSupply(
        rates,
        base.structure_caps,
        invocations=base.invocations,
        resource_units=base.resource_units,
        flops_by_dtype=flops,
        undeclared_pipes=base.undeclared_pipes,
    )


def test_hidden_size_sweep_is_smooth_and_monotonic_without_5120_cliff():
    supply = _base_supply()
    widths = (2048, 3072, 4096, 4128, 5120, 5184, 6144, 7168, 8192)
    decode_geom = WorkloadGeometry.homogeneous(1, 1, 4096, phase="decode")
    prefill_geom = WorkloadGeometry(
        (RequestGeometry(512, 0, phase="prefill"),), phase="prefill"
    )
    mtu_geom = WorkloadGeometry(
        (RequestGeometry(64, 0, phase="prefill"),), phase="prefill"
    )
    decode_ms: list[float] = []
    prefill_ms: list[float] = []
    mtu_dispatch_ms: list[float] = []
    for h in widths:
        model = _compact_v41_model(hidden_size=h)
        features = FeatureConfig.from_model(model, tp=8, ep=8)
        pipe = assemble_model(model, supply=supply, features=features)
        decode_ms.append(float(pipe.forward(decode_geom).total_ms))
        prefill_ms.append(float(pipe.forward(prefill_geom).total_ms))
        tp1_features = FeatureConfig.from_model(model, tp=1, ep=8, dp=8)
        tp1_pipe = assemble_model(model, supply=supply, features=tp1_features)
        mtu_pred = tp1_pipe.forward(mtu_geom, fusion=False)
        dispatch_op = next(r.cost for r in mtu_pred.rows if r.cost.name == "dispatch")
        mtu_dispatch_ms.append(float(dispatch_op.duration_ms))

    for idx in range(len(widths) - 1):
        assert decode_ms[idx + 1] >= decode_ms[idx] - 1e-9, (
            f"Decode latency decreased from H={widths[idx]} to H={widths[idx + 1]}: "
            f"{decode_ms[idx]} -> {decode_ms[idx + 1]}"
        )
        assert prefill_ms[idx + 1] >= prefill_ms[idx] - 1e-9, (
            f"Prefill latency decreased from H={widths[idx]} to H={widths[idx + 1]}: "
            f"{prefill_ms[idx]} -> {prefill_ms[idx + 1]}"
        )
        assert mtu_dispatch_ms[idx + 1] >= mtu_dispatch_ms[idx] - 1e-9, (
            f"Dispatch latency decreased within MTU plateau from H={widths[idx]} to H={widths[idx + 1]}"
        )

    # Specifically verify H=5120 -> 5184 (+1.25% width) has no cliff (>15% jump or drop)
    idx_5120 = widths.index(5120)
    idx_5184 = widths.index(5184)
    rel_change = (prefill_ms[idx_5184] - prefill_ms[idx_5120]) / prefill_ms[idx_5120]
    assert 0.0 <= rel_change < 0.15


def test_num_experts_sweep_is_monotonic_across_decode_and_prefill():
    supply = _base_supply()
    experts_grid = (64, 128, 136, 192, 200, 256, 384, 512)
    decode_geom = WorkloadGeometry.homogeneous(1, 1, 4096, phase="decode")
    prefill_geom = WorkloadGeometry(
        (RequestGeometry(512, 0, phase="prefill"),), phase="prefill"
    )
    decode_ms: list[float] = []
    prefill_ms: list[float] = []
    router_hist_ms: list[float] = []
    for num_e in experts_grid:
        model = _compact_v41_model(num_experts=num_e)
        features = FeatureConfig.from_model(model, tp=8, ep=8)
        pipe = assemble_model(model, supply=supply, features=features)
        decode_ms.append(float(pipe.forward(decode_geom).total_ms))
        pre_pred = pipe.forward(prefill_geom, fusion=False)
        prefill_ms.append(float(pre_pred.total_ms))
        hist_op = next(r.cost for r in pre_pred.rows if "histogram" in r.cost.name.lower())
        router_hist_ms.append(float(hist_op.duration_ms))

    for idx in range(len(experts_grid) - 1):
        assert decode_ms[idx + 1] >= decode_ms[idx] - 1e-9, (
            f"Decode latency decreased from E={experts_grid[idx]} to E={experts_grid[idx + 1]}"
        )
        assert prefill_ms[idx + 1] >= prefill_ms[idx] - 1e-9, (
            f"Prefill latency decreased from E={experts_grid[idx]} to E={experts_grid[idx + 1]}"
        )
        assert router_hist_ms[idx + 1] >= router_hist_ms[idx] - 1e-9, (
            f"Router histogram decreased from E={experts_grid[idx]} to E={experts_grid[idx + 1]}"
        )
    assert prefill_ms[-1] > prefill_ms[0]
    # Also verify no discrete cliff at local_e = 16 -> 17 (E=128 -> E=136 on ep=8)
    idx_128 = experts_grid.index(128)
    idx_136 = experts_grid.index(136)
    assert (decode_ms[idx_136] - decode_ms[idx_128]) / decode_ms[idx_128] < 0.15


def test_core_count_and_independent_aiv_sweeps_are_monotonic():
    model = _compact_v41_model()
    decode_geom = WorkloadGeometry.homogeneous(1, 1, 4096, phase="decode")
    prefill_geom = WorkloadGeometry(
        (RequestGeometry(512, 0, phase="prefill"),), phase="prefill"
    )

    # 1. AIC core sweep (num_cores = 16 -> 24 -> 32 -> 48 -> 64)
    cores_grid = (16, 24, 32, 48, 64)
    aic_decode: list[float] = []
    aic_prefill: list[float] = []
    for cores in cores_grid:
        supply = _base_supply(num_cores=cores)
        features = FeatureConfig.from_model(model, tp=8, ep=8)
        pipe = assemble_model(model, supply=supply, features=features)
        aic_decode.append(float(pipe.forward(decode_geom).total_ms))
        aic_prefill.append(float(pipe.forward(prefill_geom).total_ms))

    for idx in range(len(cores_grid) - 1):
        assert aic_decode[idx + 1] <= aic_decode[idx] + 1e-9, (
            f"Decode slower at cores={cores_grid[idx + 1]} vs {cores_grid[idx]}: "
            f"{aic_decode[idx + 1]} > {aic_decode[idx]}"
        )
        assert aic_prefill[idx + 1] <= aic_prefill[idx] + 1e-9, (
            f"Prefill slower at cores={cores_grid[idx + 1]} vs {cores_grid[idx]}: "
            f"{aic_prefill[idx + 1]} > {aic_prefill[idx]}"
        )
    assert aic_prefill[-1] < aic_prefill[0]
    assert aic_decode[-1] < aic_decode[0]

    # 2. Independent AIV core sweep (aiv_count = 32 -> 48 -> 64 -> 96 -> 128 at fixed aic=32)
    aiv_grid = (32, 48, 64, 96, 128)
    aiv_decode: list[float] = []
    aiv_prefill: list[float] = []
    vec_op_ms: list[float] = []
    for aiv_count in aiv_grid:
        supply = _base_supply(aiv_count=aiv_count)
        assert dict(supply.resource_units)["compute_units"] == 32
        assert dict(supply.resource_units)["vector"] == aiv_count
        features = FeatureConfig.from_model(model, tp=8, ep=8)
        pipe = assemble_model(model, supply=supply, features=features)
        dec_pred = pipe.forward(decode_geom)
        pre_pred = pipe.forward(prefill_geom, fusion=False)
        aiv_decode.append(float(dec_pred.total_ms))
        aiv_prefill.append(float(pre_pred.total_ms))
        vec_sum = sum(
            float(r.cost.duration_ms)
            for r in pre_pred.rows
            if r.cost.family in {"elementwise", "quantize", "router"}
        )
        vec_op_ms.append(vec_sum)

    for idx in range(len(aiv_grid) - 1):
        assert aiv_decode[idx + 1] <= aiv_decode[idx] + 1e-9
        assert aiv_prefill[idx + 1] <= aiv_prefill[idx] + 1e-9
        assert vec_op_ms[idx + 1] <= vec_op_ms[idx] + 1e-9
    assert vec_op_ms[-1] < vec_op_ms[0]


def test_all_eight_pipes_scale_monotonically_and_preserve_link_work():
    model = _compact_v41_model()
    base = _base_supply()
    features = FeatureConfig.from_model(model, tp=8, ep=8)
    prefill_geom = WorkloadGeometry(
        (RequestGeometry(512, 0, phase="prefill"),), phase="prefill"
    )
    factors = (0.5, 0.75, 1.0, 1.5, 2.0)

    # Uniform all-pipe scaling
    uniform_ms = [
        float(
            assemble_model(
                model, supply=_scale_supply(base, factor=f), features=features
            ).forward(prefill_geom).total_ms
        )
        for f in factors
    ]
    for idx in range(len(factors) - 1):
        assert uniform_ms[idx + 1] < uniform_ms[idx]

    # Individual pipe sweeps cover every physical route.  Only pipes that are
    # an actual stage bottleneck must shorten elapsed time; a non-dominant
    # resource is correctly invariant.
    all_pipes = ("cube", "vector", "hbm", "mte1", "mte2", "mte3", "fixpipe", "link_hccs")
    moving = set()
    for pipe_name in all_pipes:
        series: list[float] = []
        link_works: list[Fraction] = []
        for f in factors:
            sup = _scale_supply(base, factor=f, pipes=(pipe_name,))
            pipe = assemble_model(model, supply=sup, features=features)
            pred = pipe.forward(prefill_geom, fusion=False)
            series.append(float(pred.total_ms))
            if pipe_name == "link_hccs":
                dispatch = next(r.cost for r in pred.rows if r.cost.name == "dispatch")
                link_works.append(dispatch.pipe_work("link"))
        for idx in range(len(factors) - 1):
            assert series[idx + 1] <= series[idx] + 1e-9, (
                f"Pipe {pipe_name} non-monotonic from {factors[idx]}x to {factors[idx + 1]}x: "
                f"{series[idx]} -> {series[idx + 1]}"
            )
        if series[-1] < series[0]:
            moving.add(pipe_name)
        else:
            # A pipe that does not shorten this workload is correctly
            # invariant; mte1 is never a stage bottleneck here.
            assert series[-1] == series[0], f"Pipe {pipe_name} changed total"
        if pipe_name == "mte1":
            assert series[-1] == series[0]
        if pipe_name == "link_hccs":
            assert len(set(link_works)) == 1 and link_works[0] > 0
    assert {"hbm", "vector"} <= moving, moving
    assert "mte1" not in moving


def test_piecewise_regime_knots_shift_with_compute_and_bandwidth():
    # (a) PipeDominanceGuard analytical crossover root shifts monotonically with supply rates
    for compute_pipe, mem_pipe in (("cube", "mte2"), ("cube", "hbm"), ("vector", "mte3")):
        roots_by_comp: list[Fraction] = []
        for comp_scale in (Fraction(1, 2), Fraction(1), Fraction(2)):
            guard = PipeDominanceGuard((
                PipeDemand(compute_pipe, Var("M") * Const(16), 100 * comp_scale),
                PipeDemand(mem_pipe, Var("M") * Const(2) + Const(4096), 50),
            ))
            roots = guard.crossover(compute_pipe, mem_pipe, "M", lower=0, upper=100_000)
            assert len(roots) == 1 and roots[0].exact
            roots_by_comp.append(roots[0].value)
        assert roots_by_comp[0] < roots_by_comp[1] < roots_by_comp[2]

        roots_by_mem: list[Fraction] = []
        for mem_scale in (Fraction(1, 2), Fraction(1), Fraction(2)):
            guard = PipeDominanceGuard((
                PipeDemand(compute_pipe, Var("M") * Const(16), 100),
                PipeDemand(mem_pipe, Var("M") * Const(2) + Const(4096), 50 * mem_scale),
            ))
            roots = guard.crossover(compute_pipe, mem_pipe, "M", lower=0, upper=100_000)
            assert len(roots) == 1 and roots[0].exact
            roots_by_mem.append(roots[0].value)
        assert roots_by_mem[0] > roots_by_mem[1] > roots_by_mem[2]

    # (b) CoreWaveGuard wave boundary shifts monotonically with cores (24 -> 32 -> 48 -> 64)
    first_second_wave_tile = [
        next(t for t in range(1, 128) if CoreWaveGuard(c).waves(t) == 2)
        for c in (24, 32, 48, 64)
    ]
    assert first_second_wave_tile == [25, 33, 49, 65]

    # (c) The piecewise ridge coordinates are structural identity, not a
    # fitted compute/memory beta.  Native work is invariant to rate sweeps.
    model = _compact_v41_model()
    base = _base_supply()
    features = FeatureConfig.from_model(model, tp=8, ep=8)
    inst = instantiate_model(model, features)
    f4_sg = inst.find("F4", layer=1, role="shared_ffn")
    geom = WorkloadGeometry((RequestGeometry(512, 0, phase="prefill"),), phase="prefill")

    def compiled(pipes: tuple[str, ...] = (), factor: float = 1.0):
        supply = _scale_supply(base, factor=factor, pipes=pipes) if pipes else base
        return OperatorPiecewiseCompiler(supply, ImplementationPolicy()).compile(f4_sg).operator("shared_gate_up")

    op_nom = compiled()
    cost_nom = op_nom.evaluate(geom)
    env_nom = dict(op_nom.segment(geom).env)
    tokens = geom.tokens
    for env in (env_nom,):
        assert _knots_tuple(env) == (Fraction(1), Fraction(tokens), Fraction(tokens))

    for pipes in (("cube", "mte1", "fixpipe"), ("hbm", "mte2"), ("vector", "mte3")):
        op = compiled(pipes, 2.0)
        env = dict(op.segment(geom).env)
        cost = op.evaluate(geom)
        assert _knots_tuple(env) == (Fraction(1), Fraction(tokens), Fraction(tokens))
        assert cost.pipe_work("cube") == cost_nom.pipe_work("cube")
        assert cost.pipe_work("hbm") == cost_nom.pipe_work("hbm")
        assert cost.duration_ms <= cost_nom.duration_ms

    # (d) Declared core/peak overrides change elapsed rate but not native work
    # or structural ridge coordinates.
    hw_64_aic = _base_supply(num_cores=64)
    hw_128_aiv = _base_supply(aiv_count=128)
    hw_both = _base_supply(
        compute_peaks={"bf16": 865.0752e12, "fp16": 865.0752e12, "int8": 1730.1504e12},
        aiv_count=128,
    )
    assert dict(hw_both.resource_units)["vector"] == 128
    assert hw_both.effective("vector") > base.effective("vector")
    for supply in (hw_64_aic, hw_128_aiv, hw_both):
        op = OperatorPiecewiseCompiler(supply, ImplementationPolicy()).compile(f4_sg).operator("shared_gate_up")
        env = dict(op.segment(geom).env)
        cost = op.evaluate(geom)
        assert _knots_tuple(env) == (Fraction(1), Fraction(tokens), Fraction(tokens))
        assert cost.pipe_work("cube") == cost_nom.pipe_work("cube")
        assert cost.pipe_work("hbm") == cost_nom.pipe_work("hbm")


def _knots_tuple(env: dict) -> tuple[Fraction, Fraction, Fraction]:
    return (env["CM_RIDGE_SHIFT"], env["RIDGE_EFFECTIVE_TOKENS"], env["VEC_EFFECTIVE_TOKENS"])

def test_long_context_indexed_prefill_exceeds_default_262k_index_rows_without_error():
    model = _compact_v41_model()
    supply = _base_supply()
    features = FeatureConfig.from_model(model, tp=8, ep=8)
    pipe = assemble_model(model, supply=supply, features=features)
    long_prefill = WorkloadGeometry(
        (RequestGeometry(2048, 0, phase="prefill"),), phase="prefill"
    )
    pred = pipe.forward(long_prefill)
    assert float(pred.total_ms) > 0.0


def test_astra_p1_p2_physics_and_provenance_audit_counterexamples():
    pol = ImplementationPolicy(mt=64, nt=64, cores=8, moe_row_tile=64, split_target=10**9, max_splits=1)

    def _hw(**kw: object) -> Hardware:
        return Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950PR_9579", "num_devices": 8, **kw})

    base_hw = _hw()
    base_sup = PipeSupply.from_hardware(base_hw)
    tiles = (("mt", 64), ("nt", 64), ("cores", 8))

    def _op(name: str, family: str, **groups) -> BoundOperator:
        return BoundOperator(OperatorTemplate(name=name, family=family, role=name), StaticSignature(role=name, **groups))

    # P1-01 & P1-03: Per-device override equivalence & private buffer/mode scaling
    s_top_aic = PipeSupply.from_hardware(_hw(compute_units=16))
    hw_dev_aic = _hw(num_devices=1, devices=[{"compute_units": 16}])
    s_dev_aic = PipeSupply.from_hardware(hw_dev_aic)
    assert s_top_aic.effective("cube") == s_dev_aic.effective("cube")
    assert dict(hw_dev_aic.dies[0].buffer_units)["l1"] == dict(base_hw.dies[0].buffer_units)["l1"] // 2
    s_top_aiv = PipeSupply.from_hardware(_hw(aiv_count=32))
    hw_dev_aiv = _hw(num_devices=1, devices=[{"aiv_count": 32}])
    s_dev_aiv = PipeSupply.from_hardware(hw_dev_aiv)
    assert s_top_aiv.effective("vector") == s_dev_aiv.effective("vector")
    assert dict(hw_dev_aiv.dies[0].buffer_units)["ub"] == dict(base_hw.dies[0].buffer_units)["ub"] // 2
    assert dict(hw_dev_aiv.dies[0].service_modes)["ub_copy"][1] == dict(base_hw.dies[0].service_modes)["ub_copy"][1] / 2

    # P1-02: Rate aliases and partial compute_peaks overrides (including simultaneous AIC=16 resize)
    s_mte = PipeSupply.from_hardware(_hw(compute_units=16, mte_bw_bytes_per_s=1e12))
    assert s_mte.effective("mte2") == 1e9
    s_rw = PipeSupply.from_hardware(_hw(compute_units=16, rates={"read": 1e12, "write": 2e12, "memory": 3.2e12}))
    assert s_rw.effective("mte2") == 1e9 and s_rw.effective("mte3") == 2e9 and s_rw.effective("hbm") == 3.2e9
    d_pk = _hw(compute_units=16, compute_peaks={"bf16": 123e12}).dies[0]
    assert dict(d_pk.compute_peaks)["bf16"] == 123e12 and dict(d_pk.compute_peaks)["fp16"] == 216.2688e12

    # P1-04 & P1-05: Monotonicity across AIC/AIV 16->17->32 and 12k-token MLA Cube 600e9->600e9+1
    a2_op = _op("attention", "attention", heads=(("q", 8), ("kv", 2), ("dk", 64), ("dv", 64)), quant=(("act_bytes", 2.0),), tiles=tiles)
    g64 = WorkloadGeometry.homogeneous(2, 32, 128, phase="prefill")
    g_attn = WorkloadGeometry.homogeneous(4, 32, 8192, phase="prefill")
    c_attn = [OperatorPiecewiseCompiler(PipeSupply.from_hardware(_hw(compute_units=c, aiv_count=64)), pol).compile_operator(a2_op).evaluate(g_attn).duration_ms for c in (16, 17, 32)]
    assert c_attn[2] <= c_attn[1] <= c_attn[0]
    ew_op = _op("rmsnorm", "elementwise", extras=(("elements", 65536), ("ops_per_element", 4)), tiles=(("mt", 64), ("cores", 8)))
    v16 = OperatorPiecewiseCompiler(PipeSupply.from_hardware(_hw(compute_units=32, aiv_count=16)), pol).compile_operator(ew_op).evaluate(g64).duration_ms
    v17 = OperatorPiecewiseCompiler(PipeSupply.from_hardware(_hw(compute_units=32, aiv_count=17)), pol).compile_operator(ew_op).evaluate(g64).duration_ms
    assert v17 <= v16
    gm_op = _op("gemm", "gemm", matrix=(("K", 256), ("N", 256)), quant=(("act_bytes", 2.0), ("weight_format", "bf16"), ("act_format", "bf16"), ("weight_total", 256 * 256 * 2), ("weight_payload", 256 * 256 * 2)), tiles=tiles)
    mla_op = _op("mla_attention", "mla", heads=(("q", 16), ("kv", 1), ("dn", 128), ("dr", 64), ("rkv", 512), ("dv", 128)), quant=(("act_bytes", 2.0),), tiles=tiles)
    r_base = dict(base_sup.rates)
    g12k = WorkloadGeometry.homogeneous(1, 12000, 12000, phase="prefill")
    t_600 = OperatorPiecewiseCompiler(replace(base_sup, rates=tuple({**r_base, "cube": 600_000_000_000.0}.items())), pol).compile_operator(mla_op).evaluate(g12k).duration_ms
    t_601 = OperatorPiecewiseCompiler(replace(base_sup, rates=tuple({**r_base, "cube": 600_000_000_001.0}.items())), pol).compile_operator(mla_op).evaluate(g12k).duration_ms
    assert t_601 <= t_600

    # P1-06: MLA prefill (14k->15k->16k), decode (121856->131072), and grouped GEMM reuse-gate continuity
    mla_comp = OperatorPiecewiseCompiler(base_sup, pol).compile_operator(mla_op)
    mla_pf = [mla_comp.evaluate(WorkloadGeometry.homogeneous(1, n, n, phase="prefill")).duration_ms for n in (14000, 15000, 16000)]
    assert mla_pf[0] < mla_pf[1] < mla_pf[2]
    mla_dc = [mla_comp.evaluate(WorkloadGeometry.homogeneous(4, 1, s, phase="decode")).duration_ms for s in (121856, 131072)]
    assert mla_dc[0] < mla_dc[1]
    gg_gate = _op("grouped", "grouped_gemm", matrix=(("K", 256), ("N", 256)), experts=(("E", 8), ("local", 8), ("k", 2)), quant=(("act_bytes", 2.0), ("weight_format", "bf16"), ("act_format", "bf16"), ("weight_total", 256 * 256 * 2), ("weight_payload", 256 * 256 * 2)), tiles=tiles)
    g96 = WorkloadGeometry.homogeneous(1, 96, 96, phase="prefill")
    gate_evals = [OperatorPiecewiseCompiler(_scale_supply(base_sup, factor=f, pipes=("hbm",)), pol).compile_operator(gg_gate).evaluate(g96) for f in (2 / 3 - 1e-12, 2 / 3, 2 / 3 + 1e-12, 0.666667)]
    assert gate_evals[0].duration_ms >= gate_evals[1].duration_ms >= gate_evals[2].duration_ms >= gate_evals[3].duration_ms
    assert abs(float(gate_evals[1].duration_ms - gate_evals[3].duration_ms)) < 1e-7
    assert abs(float(gate_evals[1].pipe_work("vector") - gate_evals[3].pipe_work("vector"))) < 10.0

    # P1-07, P1-10 & R2-P1-01: Format alias invariance & conserved HBM traffic under fixed routing (MXFP4 & FP8)
    gg_durations = []
    for fmt in ("mxfp4", "mx_fp4", "mxfp4_e2m1"):
        op_f = _op(
            "expert_gate_up", "grouped_gemm",
            matrix=(("K", 128), ("N", 128)), experts=(("E", 224), ("local", 28), ("k", 2)),
            parallel=(("tp", 8),), quant=(("weight_format", fmt), ("routed_weight_format", fmt), ("weight_per_expert", 8192), ("act_bytes", 2.0), ("out_bytes", 2.0)), tiles=tiles,
        )
        gg_durations.append(OperatorPiecewiseCompiler(base_sup, pol).compile_operator(op_f).evaluate(WorkloadGeometry.homogeneous(1, 64, 64, phase="prefill")).duration_ms)
    assert gg_durations[0] == gg_durations[1] == gg_durations[2]

    fixed_hbm = []
    for loc in (1, 8, 28):
        op_l = _op(
            "expert_gate_up", "grouped_gemm",
            matrix=(("K", 128), ("N", 128)), experts=(("E", 224), ("local", loc), ("k", 1)),
            parallel=(("tp", 8),), quant=(("weight_format", "mxfp4"), ("routed_weight_format", "mxfp4"), ("weight_per_expert", 8192), ("act_bytes", 2.0), ("out_bytes", 2.0)), tiles=tiles,
        )
        g_fix = WorkloadGeometry((RequestGeometry(64, 0, phase="prefill"),), phase="prefill", routing=((0, (64,) + (0,) * (loc - 1)),))
        ev = OperatorPiecewiseCompiler(base_sup, pol).compile_operator(op_l).evaluate(g_fix)
        fixed_hbm.append(float(ev.pipe_work("hbm")))
    assert fixed_hbm[0] >= 40_960.0 and fixed_hbm[0] == fixed_hbm[1] == fixed_hbm[2]

    fp8_evals = []
    for loc in (8, 9):
        op_fp8 = _op(
            "expert_gate_up", "grouped_gemm",
            matrix=(("K", 4096), ("N", 4096)), experts=(("E", 72), ("local", loc), ("k", 1)),
            parallel=(("tp", 1),), quant=(("weight_format", "fp8_e4m3"), ("routed_weight_format", "fp8_e4m3"), ("weight_per_expert", 4096**2), ("act_bytes", 2.0), ("out_bytes", 2.0)), tiles=tiles,
        )
        g_fp8 = WorkloadGeometry((RequestGeometry(512, 0, phase="prefill"),), phase="prefill", routing=((0, (512,) + (0,) * (loc - 1)),))
        fp8_evals.append(OperatorPiecewiseCompiler(base_sup, pol).compile_operator(op_fp8, kind="F3").evaluate(g_fp8))
    assert fp8_evals[0].duration_ms == fp8_evals[1].duration_ms
    assert float(fp8_evals[0].pipe_work("hbm")) == float(fp8_evals[1].pipe_work("hbm")) >= 25_165_824.0

    # P1-08: Provenance label invariance and neutral hardware purity (with & without resource_units)
    hw_dt = Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 8})
    hw_neu = replace(hw_dt, sku="neutral", name="neutral", hardware_key="neutral")
    g80 = WorkloadGeometry.homogeneous(1, 80, 80, phase="prefill")
    assert OperatorPiecewiseCompiler(PipeSupply.from_hardware(hw_dt), pol).compile_operator(gm_op).evaluate(g80).duration_ms == OperatorPiecewiseCompiler(PipeSupply.from_hardware(hw_neu), pol).compile_operator(gm_op).evaluate(g80).duration_ms
    pure_hw = Hardware.from_mapping({"name": "neutral", "sku": "neutral", "num_devices": 1, "device": {"memory_bytes": 64 * 1024**3, "compute_peaks": {"bf16": 400e12}, "rates": {"hbm": 1.6e12, "vector": 6.75e12}, "compute_units": 8}})
    pure_sup = PipeSupply.from_hardware(pure_hw)
    assert pure_sup.invocations == () and pure_sup.resource_units == ()
    ew256_op = _op("elementwise", "elementwise", extras=(("elements", 256), ("ops_per_element", 4)), tiles=(("mt", 64), ("cores", 8)))
    for kinv in ({}, {"cube": {"minimum_ms": 1e-6}}, {"cube": {"minimum_ms": 1e-6}, "vector": {"minimum_ms": 0}}):
        owned_hw = Hardware.from_mapping({"name": "neutral", "sku": "neutral", "num_devices": 1, "device": {"memory_bytes": 64 * 1024**3, "compute_peaks": {"bf16": 400e12}, "rates": {"hbm": 1.6e12, "vector": 6.75e12}, "compute_units": 8, "resource_units": {"vector": 8}, "kernel_invocations": kinv}})
        owned_sup = PipeSupply.from_hardware(owned_hw)
        owned_op = OperatorPiecewiseCompiler(owned_sup, pol).compile_operator(ew256_op)
        owned_env = dict(owned_op.segment(g64).env)
        owned_ev = owned_op.evaluate(g64)
        assert dict(owned_sup.resource_units)["vector"] == 8 and owned_env.get("HW_SPAN_WORK", 0) == 0
        assert owned_env["RIDGE_EFFECTIVE_TOKENS"] == 64 and owned_env["VEC_EFFECTIVE_TOKENS"] == 64 and float(owned_ev.pipe_work("hbm")) == 65_536.0

    # P1-09 & P1-10: Latency monotonicity (transport & F1 router_topk) and single-node collective monotonicity
    tr_op = _op("dispatch", "transport", parallel=(("members", 8), ("ranks_per_node", 8), ("tp", 8)), extras=(("wire_bytes", 8192.0), ("row_bytes", 8192.0), ("top_k", 2), ("latent", 512)))
    g512 = WorkloadGeometry.homogeneous(1, 512, 512, phase="prefill")
    inv_d = dict(base_sup.invocations)
    tr_durs = [OperatorPiecewiseCompiler(replace(base_sup, invocations=tuple({**inv_d, "link_latency_ms": lat}.items())), pol).compile_operator(tr_op).evaluate(g512).duration_ms for lat in (0.0018, 0.0036, 0.0072)]
    assert tr_durs[0] < tr_durs[1] < tr_durs[2]

    m_v41 = _compact_v41_model()
    f1_sg = instantiate_model(m_v41, FeatureConfig.from_model(m_v41, tp=8, ep=8)).find("F1", layer=1, role="router")
    g256 = WorkloadGeometry.homogeneous(1, 256, 256, phase="prefill")
    rt_1 = OperatorPiecewiseCompiler(replace(base_sup, invocations=tuple({**inv_d, "vector": 0.001312}.items())), pol).compile(f1_sg).operator("router_topk")
    rt_2 = OperatorPiecewiseCompiler(replace(base_sup, invocations=tuple({**inv_d, "vector": 0.002624}.items())), pol).compile(f1_sg).operator("router_topk")
    assert rt_1.evaluate(g256).duration_ms < rt_2.evaluate(g256).duration_ms
    assert dict(rt_1.segment(g256).env)["VEC_EFFECTIVE_TOKENS"] == dict(rt_2.segment(g256).env)["VEC_EFFECTIVE_TOKENS"]

    ar_durs = []
    for m_cnt in (7, 8):
        ar_op = _op("allreduce", "collective", parallel=(("members", m_cnt), ("ranks_per_node", 8)), extras=(("wire_bytes", 8192.0), ("collective", "allreduce")))
        ev_ar = OperatorPiecewiseCompiler(base_sup, pol).compile_operator(ar_op).evaluate(g512)
        ar_durs.append((ev_ar.duration_ms, float(ev_ar.pipe_work("link"))))
    assert ar_durs[0][0] < ar_durs[1][0] and ar_durs[1][1] >= 7_340_032.0

    # P2-01, P2-02, P2-03 & P2-04: Active ridge onset, balance qualification, explicit cores, domain rejection, and A1 fusion byte conservation
    mqa_op = _op("attention", "attention", heads=(("q", 8), ("kv", 1), ("dk", 64), ("dv", 64)), quant=(("act_bytes", 2.0),), tiles=tiles)
    mqa_comp = OperatorPiecewiseCompiler(base_sup, pol).compile_operator(mqa_op, kind="A0")
    assert dict(mqa_comp.program.annotations)["balance_semantics"] == "raw_pre_service_roofline_diagnostic"
    mqa_points = (6000, 8192, 8200)
    mqa_scales = [float(dict(mqa_comp.segment(WorkloadGeometry.homogeneous(1, n, n, phase="prefill")).env)["HW_SCALE_HBM"]) for n in mqa_points]
    assert mqa_scales == [1.0, 1.0, 1.0]
    # Raw attention work still grows across the nominal 8192 grid point; the
    # removed empirical HBM scale is not a physical grid correction.
    mqa_work = [float(mqa_comp.evaluate(WorkloadGeometry.homogeneous(1, n, n, phase="prefill")).pipe_work("hbm")) for n in mqa_points]
    assert mqa_work[0] < mqa_work[1] < mqa_work[2]

    for c_val in (23, 24, 25):
        pipe_c = ModelPipeline(m_v41, FeatureConfig.from_model(m_v41, cores=c_val), base_sup, ImplementationPolicy(cores=64))
        assert pipe_c.features.cores == c_val
    with pytest.raises(ModelingError):
        OperatorPiecewiseCompiler(base_sup, pol, _PIPELINE_DOMAIN).compile_operator(a2_op).evaluate(WorkloadGeometry.homogeneous(1, 100_000_000, 100_000_000, phase="prefill"))

    m_mla = ModelSpec.from_mapping({"name": "mla", "hidden_size": 64, "intermediate_size": 128, "layers": 1, "num_heads": 4, "num_kv_heads": 1, "head_dim": 16, "vocab_size": 256, "mla": {"q_lora_rank": 32, "kv_lora_rank": 16, "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8}, "layer_plans": [{"index": 0, "attention": "mla", "ffn": "dense", "num_heads": 4, "num_kv_heads": 1, "head_dim": 16, "intermediate_size": 128}]})
    a1_inst = instantiate_model(m_mla, FeatureConfig.from_model(m_mla, tp=2)).find("A1", layer=0)
    a1_comp = OperatorPiecewiseCompiler(base_sup, pol).compile(a1_inst)
    f_res = evaluate_serial_split(a1_comp, WorkloadGeometry.homogeneous(2, 12, 32, phase="prefill"), StateView(), fusion=tuple(c.name for c in a1_inst.template.fusions)).fusion_results
    assert len(f_res) > 0
    for fr in f_res:
        sem = {p: sum((dict(c.work).get(p, Fraction(0)) for c in fr.semantic), Fraction(0)) for p in ("hbm", "mte2", "mte3")}
        fus = dict(fr.fused.work)
        assert (sem["hbm"] - fus.get("hbm", Fraction(0))) == (sem["mte2"] - fus.get("mte2", Fraction(0))) + (sem["mte3"] - fus.get("mte3", Fraction(0))) > 0
