"""Multi-pipe counterfactual knots for the major operator families.

[Role] Sweep cube, vector (AIV), hbm, mte1, mte2, mte3, fixpipe and link
  rates, plus AIC and AIV core counts, and check piecewise duration is
  monotonic.  Check that PipeDominanceGuard roots and the published
  Cube/AIV/MTE2 balance ratios, CM_RIDGE_SHIFT, RIDGE_EFFECTIVE_TOKENS and
  VEC_EFFECTIVE_TOKENS move exactly when the binding pipe rate changes.
[Invariants] Assertions are closed-form supply ratios and envelope
  monotonicity.  No measured latency tables or fabricated hardware mocks.
[Timeouts & Locks] Tier 0 (<5s), pure CPU arithmetic, no I/O or locks.
[Side-Effect Policy] In-memory operator construction only.
"""
from __future__ import annotations

from fractions import Fraction

from mindie_sim.prediction.piecewise import (
    Const, ImplementationPolicy, OperatorPiecewiseCompiler, PipeDemand,
    PipeDominanceGuard, PipeSupply, Var, WorkloadGeometry,
)
from mindie_sim.prediction.piecewise.expressions import as_fraction
from mindie_sim.prediction.piecewise.physical import ARCH3510_VECTOR_MIN_MS
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.subgraphs import BoundOperator, OperatorTemplate, StaticSignature

PIPES = ("cube", "vector", "hbm", "mte1", "mte2", "mte3", "fixpipe", "link")
FACTORS = (0.5, 1.0, 2.0)
AIC_CORES = (16, 24, 32, 48, 64)
AIV_CORES = (32, 48, 64, 96, 128)
ATTN_PIPES = frozenset({"cube", "vector", "hbm", "mte1", "mte2", "mte3", "fixpipe"})
POLICY = ImplementationPolicy(mt=64, nt=64, cores=8, moe_row_tile=64,
                              split_target=10 ** 9, max_splits=1)
GEOM = WorkloadGeometry.homogeneous(2, 32, 128, phase="prefill")

_TILES = (("mt", 64), ("nt", 64), ("cores", 8))
_QUANT = (("act_bytes", 2.0), ("weight_format", "bf16"), ("act_format", "bf16"),
          ("weight_total", 256 * 256 * 2), ("weight_payload", 256 * 256 * 2))


def _bound(name: str, family: str, **groups) -> BoundOperator:
    return BoundOperator(
        OperatorTemplate(name=name, family=family, role=name),
        StaticSignature(role=name, **groups),
    )


OPERATORS = {
    "gemm": _bound("gemm", "gemm", matrix=(("K", 256), ("N", 256)),
                   quant=_QUANT, tiles=_TILES),
    "grouped_gemm": _bound(
        "grouped", "grouped_gemm", matrix=(("K", 128), ("N", 128)),
        experts=(("E", 8), ("local", 8), ("k", 2)), quant=_QUANT, tiles=_TILES),
    "attention": _bound(
        "attention", "attention",
        heads=(("q", 8), ("kv", 2), ("dk", 64), ("dv", 64)),
        quant=(("act_bytes", 2.0),), tiles=_TILES),
    "mla": _bound(
        "mla_attention", "mla",
        heads=(("q", 16), ("kv", 1), ("dn", 128), ("dr", 64),
               ("rkv", 512), ("dv", 128)),
        quant=(("act_bytes", 2.0),), tiles=_TILES),
    "indexer": _bound("index_score", "indexer",
                      heads=(("heads", 8), ("dim", 64), ("topk", 64)),
                      tiles=(("nt", 64),)),
    "compression": _bound("compress", "compression",
                          extras=(("ratio", 4), ("width", 64), ("row_bytes", 128))),
    "recurrent": _bound(
        "gated_delta", "recurrent",
        heads=(("key_heads", 4), ("value_heads", 4), ("dk", 64),
               ("dv", 64), ("chunk", 64))),
    "kv_access": _bound("kv_read", "kv_access", extras=(("row_bytes", 256),)),
    "elementwise": _bound("rmsnorm", "elementwise",
                          extras=(("elements", 256), ("ops_per_element", 4)),
                          tiles=(("mt", 64), ("cores", 8))),
    "quantize": _bound("quantize", "quantize",
                       extras=(("elements", 256), ("quantized", 1),
                               ("out_format", "int8")),
                       tiles=(("mt", 64), ("cores", 8))),
    "router": _bound("router", "router", experts=(("E", 16), ("k", 2)),
                     tiles=(("mt", 64),)),
    "collective": _bound(
        "allreduce", "collective",
        parallel=(("members", 8), ("ranks_per_node", 8)),
        extras=(("wire_bytes", 8192.0), ("collective", "allreduce"))),
    "transport": _bound(
        "dispatch", "transport",
        parallel=(("members", 8), ("ranks_per_node", 8)),
        extras=(("wire_bytes", 8192.0), ("row_bytes", 8192.0),
                ("top_k", 2), ("latent", 512))),
}


def _hardware(**overrides) -> PipeSupply:
    raw = {"backend": "ascend", "sku": "Ascend950PR_9579", "num_devices": 8,
           **overrides}
    return PipeSupply.from_hardware(Hardware.from_mapping(raw))


def _scale(base: PipeSupply, factor: float, pipes: tuple[str, ...]) -> PipeSupply:
    chosen = set(pipes)
    rates = tuple((name, float(rate) * factor if name in chosen else float(rate))
                  for name, rate in base.rates)
    return PipeSupply(
        rates, base.structure_caps, invocations=base.invocations,
        resource_units=base.resource_units, flops_by_dtype=base.flops_by_dtype,
        undeclared_pipes=base.undeclared_pipes,
    )


def _evaluate(operator, supply):
    compiled = OperatorPiecewiseCompiler(supply, POLICY).compile_operator(operator)
    return compiled.evaluate(GEOM), dict(compiled.segment(GEOM).env)


def _monotonic(series: list[Fraction], label: str) -> None:
    for index in range(len(series) - 1):
        assert series[index + 1] <= series[index], (
            f"{label} increased: {series[index]} -> {series[index + 1]}")


def _balance(supply: PipeSupply, *, kv_heads: int, qk: int, value: int,
             latent: bool) -> tuple[Fraction, Fraction, Fraction]:
    """Cube/MTE2, AIV/MTE2 and Cube/AIV crossover rows for one head shape."""
    act = 2
    kv_bytes = Fraction(kv_heads * (qk if latent else qk + value) * act)
    head_row = kv_bytes / kv_heads
    cube_row = Fraction(2 * (qk + value))
    vec_pair = 7 + Fraction(3 * value + 10, 64)
    cube = Fraction(str(supply.effective("cube")))
    vector = Fraction(str(supply.effective("vector")))
    mte2 = Fraction(str(supply.effective("mte2")))
    return (
        head_row * cube / (cube_row * mte2),
        head_row * vector / (vec_pair * mte2),
        cube_row * vector / (vec_pair * cube),
    )


def _ridge(supply: PipeSupply, tokens: int, present: frozenset[str]):
    """Analytical CM/VEC knots from the fixed 9579 reference the service layer uses."""
    peaks = dict(supply.flops_by_dtype)
    refs = {
        "cube": 865_075_200_000.0,
        "vector": 10_560_000_000.0,
        "hbm": 1_600_000_000.0,
        "mte1": 13_516_800_000.0,
        "mte2": 4_224_000_000.0,
        "mte3": 13_516_800_000.0,
        "fixpipe": 6_758_400_000.0,
    }
    undeclared = set(supply.undeclared_pipes)

    def rel(pipe: str) -> float:
        if not peaks or pipe not in present or pipe in undeclared or refs[pipe] <= 0.0:
            return 1.0
        eff = float(supply.effective(pipe))
        return max(0.25, min(4.0, eff / refs[pipe])) if eff > 0.0 else 1.0

    comp = min(rel("cube"), rel("mte1"), rel("fixpipe"))
    mem = min(rel("hbm"), rel("mte2"))
    beta = max(0.25, min(4.0, comp / max(0.25, mem)))
    vec_bw = max(0.25, min(4.0, min(
        rel("vector"), rel("hbm"), rel("mte2"), rel("mte3"),
    )))
    return (
        as_fraction(round(beta, 6)),
        as_fraction(round(tokens / beta, 6)),
        as_fraction(round(tokens / vec_bw, 6)),
    )


def _knots(env: dict) -> tuple[Fraction, Fraction, Fraction]:
    return (env["CM_RIDGE_SHIFT"], env["RIDGE_EFFECTIVE_TOKENS"],
            env["VEC_EFFECTIVE_TOKENS"])


def test_family_rate_and_core_sweeps_are_monotonic():
    base = _hardware()
    aic = [_hardware(num_cores=cores) for cores in AIC_CORES]
    aiv = [_hardware(aiv_count=cores) for cores in AIV_CORES]
    moved = {pipe: False for pipe in PIPES}
    for family, operator in OPERATORS.items():
        for pipe in PIPES:
            series = [_evaluate(operator, _scale(base, factor, (pipe,)))[0].duration_ms
                      for factor in FACTORS]
            _monotonic(series, f"{family}:{pipe}")
            if series[-1] < series[0]:
                moved[pipe] = True
        _monotonic([_evaluate(operator, supply)[0].duration_ms for supply in aic],
                   f"{family}:aic")
        _monotonic([_evaluate(operator, supply)[0].duration_ms for supply in aiv],
                   f"{family}:aiv")
    assert all(moved.values()), {pipe: live for pipe, live in moved.items() if not live}
    gemm_aic = [_evaluate(OPERATORS["gemm"], supply)[0].duration_ms for supply in aic]
    grouped_aiv = [_evaluate(OPERATORS["grouped_gemm"], supply)[0].duration_ms
                   for supply in aiv]
    assert gemm_aic[-1] < gemm_aic[0]
    assert grouped_aiv[-1] < grouped_aiv[0]


def test_guard_crossover_shifts_exactly_with_every_pipe():
    """Affine root of ``16 M / S = (2 M + 4096) / T`` moves with each supply."""

    def root(left: Fraction, right: Fraction) -> Fraction:
        return Fraction(4096) * left / (Fraction(16) * right - 2 * left)

    for pipe in PIPES:
        other = "mte2" if pipe == "hbm" else "hbm"
        roots: list[Fraction] = []
        for scale in (Fraction(1, 2), Fraction(1), Fraction(2)):
            guard = PipeDominanceGuard((
                PipeDemand(pipe, Var("M") * Const(16), 100 * scale),
                PipeDemand(other, Var("M") * Const(2) + Const(4096), 80),
            ))
            found = guard.crossover(pipe, other, "M", lower=0, upper=100_000)
            assert len(found) == 1 and found[0].exact
            expect = root(Fraction(100) * scale, Fraction(80))
            assert found[0].value == expect
            assert guard.times({"M": expect})[pipe] == guard.times({"M": expect})[other]
            assert found[0].dominance_before == other
            assert found[0].dominance_after == pipe
            roots.append(found[0].value)
        assert roots[0] < roots[1] < roots[2]


def test_balance_ratios_and_ridge_knots_track_binding_pipes():
    base = _hardware()
    tokens = GEOM.tokens
    nominal_cost, nominal_env = _evaluate(OPERATORS["attention"], base)
    nominal = _knots(nominal_env)
    assert nominal == _ridge(base, tokens, ATTN_PIPES)
    shapes = {
        "attention": dict(kv_heads=2, qk=64, value=64, latent=False),
        "mla": dict(kv_heads=1, qk=576, value=512, latent=True),
    }
    for family, shape in shapes.items():
        for pipe in PIPES:
            for factor in FACTORS:
                supply = _scale(base, factor, (pipe,))
                cost, env = _evaluate(OPERATORS[family], supply)
                balance = _balance(supply, **shape)
                assert cost.get("balance_cube_mte2") == balance[0]
                assert cost.get("balance_vec_mte2") == balance[1]
                assert cost.get("cube_vector_ratio") == balance[2]
                if family == "attention":
                    assert _knots(env) == _ridge(supply, tokens, ATTN_PIPES)

    # Cube is the binding compute pipe on bf16 9579, so both directions move
    # the ridge.  Memory pipes move it only while they are the min.
    half_cube = _knots(_evaluate(OPERATORS["attention"], _scale(base, 0.5, ("cube",)))[1])
    fast_cube = _knots(_evaluate(OPERATORS["attention"], _scale(base, 2.0, ("cube",)))[1])
    assert half_cube[0] * 2 == nominal[0] == fast_cube[0] / 2
    assert half_cube[1] == nominal[1] * 2
    assert fast_cube[1] * 2 == nominal[1]
    slow_hbm = _knots(_evaluate(OPERATORS["attention"], _scale(base, 0.5, ("hbm",)))[1])
    fast_hbm = _knots(_evaluate(OPERATORS["attention"], _scale(base, 2.0, ("hbm",)))[1])
    assert slow_hbm[0] == nominal[0] * 2 and slow_hbm[1] * 2 == nominal[1]
    assert fast_hbm[0] == nominal[0] and fast_hbm[2] == nominal[2]
    slow_mte2 = _knots(_evaluate(OPERATORS["attention"], _scale(base, 0.5, ("mte2",)))[1])
    assert slow_mte2[0] == nominal[0] * 2
    # Balance rows scale with the raw pipe ratio, not the binding min.
    assert nominal_cost.get("balance_cube_mte2") * 2 == _evaluate(
        OPERATORS["attention"], _scale(base, 2.0, ("cube",)))[0].get("balance_cube_mte2")
    assert nominal_cost.get("cube_vector_ratio") == _evaluate(
        OPERATORS["attention"], _scale(base, 2.0, ("cube",)))[0].get("cube_vector_ratio") * 2
    assert nominal_cost.get("balance_vec_mte2") * 2 == _evaluate(
        OPERATORS["attention"], _scale(base, 2.0, ("vector",)))[0].get("balance_vec_mte2")
    assert nominal_cost.get("cube_vector_ratio") * 2 == _evaluate(
        OPERATORS["attention"], _scale(base, 2.0, ("vector",)))[0].get("cube_vector_ratio")
    assert nominal_cost.get("balance_cube_mte2") == _evaluate(
        OPERATORS["attention"], _scale(base, 2.0, ("mte2",)))[0].get("balance_cube_mte2") * 2
    assert nominal_cost.get("balance_vec_mte2") == _evaluate(
        OPERATORS["attention"], _scale(base, 2.0, ("mte2",)))[0].get("balance_vec_mte2") * 2

    # MTE1 and fixpipe shift the ridge once they are the unique compute min.
    for pipe, lifted in (
        ("mte1", _scale(base, 4.0, ("cube", "fixpipe"))),
        ("fixpipe", _scale(base, 4.0, ("cube", "mte1"))),
    ):
        ridges = []
        for factor in FACTORS:
            supply = _scale(lifted, factor, (pipe,))
            env = _evaluate(OPERATORS["attention"], supply)[1]
            assert _knots(env) == _ridge(supply, tokens, ATTN_PIPES)
            ridges.append(env["CM_RIDGE_SHIFT"])
        assert ridges[0] * 2 == ridges[1] and ridges[1] * 2 == ridges[2]

    # Vector 0.5x doubles VEC tokens; 2x hands the min to the memory pipes.
    slow_vec = _knots(_evaluate(OPERATORS["attention"], _scale(base, 0.5, ("vector",)))[1])
    fast_vec = _knots(_evaluate(OPERATORS["attention"], _scale(base, 2.0, ("vector",)))[1])
    assert slow_vec[2] == nominal[2] * 2
    assert fast_vec[2] == Fraction(tokens) < nominal[2]
    slow_mte3 = _knots(_evaluate(OPERATORS["attention"], _scale(base, 0.5, ("mte3",)))[1])
    assert slow_mte3[0] == nominal[0] and slow_mte3[2] > nominal[2]
    # Link is outside the cube/memory ridge.  Its knot is the guard root
    # (above) and the transport envelope.
    for factor in (0.5, 2.0):
        env = _evaluate(OPERATORS["attention"], _scale(base, factor, ("link",)))[1]
        assert _knots(env) == nominal
    link_ms = [_evaluate(OPERATORS["transport"], _scale(base, factor, ("link",)))[0].duration_ms
               for factor in FACTORS]
    assert link_ms[0] > link_ms[1] > link_ms[2]
