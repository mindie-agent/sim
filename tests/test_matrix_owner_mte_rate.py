"""Per-owner MTE rate for one HC projection and one router GEMM.

[Role]
  The native steady matrix pins each output tile to one mte2_aic unit.
  Idle units are not lent to that tile.  These tests check that law on the
  closed-form owner copy and on one instantiated A3 kernel, then check that
  the piecewise GEMM uses the same owner and the declared compute dtype.

[Invariants]
  * The fraction is charged bytes over (slowest owner bytes times units).
    It is not a duration and not split-K tile count over policy cores.
  * A declared minimum below the copy does not change the native elapsed.
  * Halving or doubling only the mte2_aic rate moves the elapsed with it.
  * HC activations and a declared router dtype are fp32 on both paths.
    The router exclusion stays BF16 when no router dtype is declared.

[Timeouts & Locks]
  One A3 decode instantiation and two kernel launches.  No device or network.

[Side-Effect Policy]
  Reads the checked-in case.  Writes nothing.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.matrix_mapping import (
    critical_output_owner_rate_fraction, gemm_fractal_operand_bytes,
    mte_owner_unit_count, owner_tiles,
)
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.piecewise.compiler import PipeSupply
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
from mindie_sim.schema.invocation import InvocationTiming

ROOT = Path(__file__).resolve().parents[1]
A3_CASE = (ROOT / "examples/cases/high-concurrency-20261030"
           / "glm-kda-accepted-conditional/glm53flash-a3-b4-q6.json")
CLOCK = 1.65e9


def test_one_owner_share_is_one_hardware_unit():
    exact = 2 * 24 * 16384 * 4
    share = critical_output_owner_rate_fraction(
        24, 24, 16384, 128, 128, 20, act_bytes=4, weight_bytes=4,
        charged_bytes=exact)
    assert share == Fraction(1, 20)
    assert mte_owner_unit_count(()) == 0
    assert mte_owner_unit_count((("mte2_aic", 20), ("compute_units", 24))) == 20
    assert critical_output_owner_rate_fraction(
        24, 24, 16384, 128, 128, 0, act_bytes=4, weight_bytes=4) == 1


def test_router_tail_uses_the_widest_tile_not_the_average():
    extents = owner_tiles(24, 288, 128, 128, 20)
    widths = [extent_n for _count, _extent_m, extent_n, _area in extents]
    assert widths == [128, 128, 32]
    copies = [4 * 24 * 4096 + 4 * width * 4096 for width in widths]
    share = critical_output_owner_rate_fraction(
        24, 288, 4096, 128, 128, 20, act_bytes=4, weight_bytes=4)
    assert share == Fraction(sum(copies), max(copies) * 20)
    assert share != Fraction(len(copies), 20)
    charged = gemm_fractal_operand_bytes(
        24, 3, 288, 4096, 4, 4)
    fractal = critical_output_owner_rate_fraction(
        24, 288, 4096, 128, 128, 20, act_bytes=4, weight_bytes=4,
        charged_bytes=charged)
    aggregate = Fraction(10_000_000)
    assert charged / (aggregate * fractal) == max(copies) / (aggregate / 20)


def test_fractal_rows_cancel_to_the_exact_owner_copy():
    exact = 2 * 24 * 16384 * 4
    fractal = gemm_fractal_operand_bytes(24, 1, 32, 16384, 4, 4)
    assert fractal == 2 * 32 * 16384 * 4
    share = critical_output_owner_rate_fraction(
        24, 24, 16384, 128, 128, 20, act_bytes=4, weight_bytes=4,
        charged_bytes=fractal)
    aggregate = Fraction(500_000)
    assert fractal / (aggregate * share) == Fraction(exact) / (aggregate / 20)


def _definition() -> dict:
    payload = json.loads(A3_CASE.read_text(encoding="utf-8"))
    runtime = payload.setdefault("runtime", {})
    kernels = runtime.get("kernels") or {}
    for key in ("service_regimes", "invocations", "primitives"):
        kernels.pop(key, None)
    if kernels:
        runtime["kernels"] = kernels
    elif "kernels" in runtime:
        runtime.pop("kernels")
    runtime.update({"tp": 16, "dp": 1, "ep": 16, "pp": 1, "graph_mode": "none"})
    return payload


def _prepared():
    case = parse_case(_definition())
    operators = target_operators(
        case, phase="decode", tokens=6, context=4096, sequences=4)
    case = replace(case, runtime=replace(
        case.runtime, kernels=deepcopy(operators.effective_kernels)))
    hc = next(op for op in operators
              if op.recipe == "hc_projection" and ".attention.hc_project" in op.op_id)
    router = next(op for op in operators if op.recipe == "router")
    return case, hc, router


def _run(case, op):
    book = RecipeBook.from_mapping(case.runtime.kernels)
    execution = Execution(case, CostModel(case), recipes=book, trace_limit=8000)
    prepared, recipe = execution.prepare(op, 0)
    context = execution._context(prepared, 0, "none", recipe=recipe)
    records = []
    context = replace(context, record_mapping=records.append)
    engine = execution.engine
    ready = engine.submit("input_ready", release=0.0)
    complete = execution.lower_program(context, recipe).start(engine, "operator", ready)
    engine.advance(until=(complete,))
    return complete.time, prepared, records, context


def _with_minimum(case, minimum_ms):
    def edit(die):
        invocations = tuple(
            (name, replace(timing, minimum_ms=minimum_ms))
            for name, timing in die.kernel_invocations)
        if not invocations:
            invocations = (
                ("cube", InvocationTiming(startup_ms=0.0, minimum_ms=minimum_ms,
                                           provenance={"basis": "declared"})),
                ("vector", InvocationTiming(startup_ms=0.0, minimum_ms=minimum_ms,
                                             provenance={"basis": "declared"})),
            )
        return replace(die, kernel_invocations=invocations)
    dies = tuple(edit(die) for die in case.machine.dies)
    return replace(case, machine=replace(case.machine, dies=dies))


def _scale_mte2(case, factor):
    def edit(die):
        rates = tuple((name, value * factor if name == "mte2_aic" else value)
                      for name, value in die.rates)
        return replace(die, rates=rates)
    dies = tuple(edit(die) for die in case.machine.dies)
    return replace(case, machine=replace(case.machine, dies=dies))


def _short_k(op, k):
    old_m, old_n, old_k = op.local_shape[-3:]
    scale = k / old_k
    meta = dict(op.metadata)
    weight = float(meta.get("packed_weight_bytes", 0.0)) * scale
    meta["packed_weight_bytes"] = weight
    if "weight_bytes" in meta:
        meta["weight_bytes"] = weight
    act = (float(op.mte2_bytes) - float(dict(op.metadata).get("packed_weight_bytes", 0.0))
           - float(meta.get("activation_scale_bytes", 0.0) or 0.0)) * scale
    scales = float(meta.get("activation_scale_bytes", 0.0) or 0.0)
    out = float(op.mte3_bytes)
    shape = (old_m, old_n, k)
    return replace(
        op, local_shape=shape, launch_shape=shape, semantic_shape=shape,
        padded_shape=tuple((value + 15) // 16 * 16 for value in shape),
        flops=2.0 * old_m * old_n * k,
        hbm_bytes=act + weight + out + scales,
        mte2_bytes=act + weight + scales, mte3_bytes=out,
        metadata=tuple(sorted(meta.items())),
    )


def _owner_bound_ms(op, record, rate, units):
    m, n, k = op.local_shape[-3:]
    meta = dict(op.metadata)
    groups = int(meta.get("matrix_groups", 1))
    act = 4 if meta.get("activation_format") == "fp32" else 2
    weight = 4 if meta.get("weight_format") == "fp32" else 2
    tile_m = int(record["selected"]["tile_m"])
    tile_n = int(record["selected"]["tile_n"])
    copies = []
    for _count, extent_m, extent_n, _area in owner_tiles(
            m // groups, n, tile_m, tile_n, units, groups):
        copies.append(act * extent_m * k + weight * extent_n * k)
    return max(copies) / (rate / units) * 1000, copies, tile_m, tile_n


def test_native_hc_and_router_follow_one_aic_mte2_copy():
    case, hc, router = _prepared()
    die = case.machine.dies[0]
    assert case.machine.sku == "Ascend910_9362"
    units = die.units("mte2_aic")
    rate = dict(die.rates)["mte2_aic"]
    assert units == die.compute_units == 20
    assert rate == pytest.approx(32 * units * CLOCK)

    for op, shape in ((hc, (24, 24, 16384)), (router, (24, 288, 4096))):
        meta = dict(op.metadata)
        assert op.local_shape[-3:] == shape
        assert meta["compute_format"] == meta["activation_format"] == meta["weight_format"] == "fp32"
        elapsed, _prepared_op, records, _context = _run(case, op)
        bound, copies, tile_m, tile_n = _owner_bound_ms(op, records[0], rate, units)
        assert tile_m == 128 and tile_n == 128
        assert elapsed == pytest.approx(bound, rel=0.02)
        assert elapsed > 0.010
        for minimum in (0.0, 0.005, 0.010):
            changed, _op2, _records, _ctx = _run(_with_minimum(case, minimum), op)
            assert changed == pytest.approx(elapsed, rel=1e-9)
        half = _run(_scale_mte2(case, 0.5), op)[0]
        double = _run(_scale_mte2(case, 2.0), op)[0]
        assert half == pytest.approx(elapsed * 2, rel=0.02)
        assert double == pytest.approx(elapsed / 2, rel=0.02)
        assert len(copies) == (1 if op is hc else 3)
    assert max(extent_n for _c, _m, extent_n, _a in owner_tiles(24, 288, 128, 128, units)) == 128

    short = _short_k(hc, 128)
    hidden = _run(_with_minimum(case, 0.0), short)[0]
    floored = _run(_with_minimum(case, 0.005), short)[0]
    assert hidden < 0.005
    assert floored == pytest.approx(0.005, rel=1e-6)
    assert hidden != pytest.approx(floored, rel=0.1)


def _piece_operators():
    case = parse_case(_definition())
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    supply = PipeSupply.from_hardware(case.machine)
    pipeline = assemble_model(case.model, supply=supply, features=features)
    geometry = WorkloadGeometry(
        tuple(RequestGeometry(6, 4096, phase="decode") for _ in range(4)),
        phase="decode")
    found = {}
    for layer in pipeline.instantiated.layers:
        for instance in layer.subgraphs:
            for operator in instance.operators:
                key = (instance.kind, operator.template.name)
                if key in {("R0", "hc_projection"), ("F1", "router_gemm")} and key not in found:
                    found[key] = (operator, instance.kind)
        if len(found) == 2:
            break
    return case, features, supply, pipeline, geometry, found


def test_piecewise_hc_and_router_use_declared_fp32_and_one_aic():
    case, features, supply, pipeline, geometry, found = _piece_operators()
    assert case.machine.sku == "Ascend910_9362"
    assert features.cores == 24
    units = mte_owner_unit_count(supply.resource_units)
    rate = supply.rate("mte2")
    per_unit = rate / units
    assert units == 20
    # Supply rates are bytes per millisecond.  One AIC is 32 bytes per cycle.
    assert rate * 1e3 == pytest.approx(32 * units * CLOCK)

    hc, hc_kind = found[("R0", "hc_projection")]
    router, router_kind = found[("F1", "router_gemm")]
    assert float(hc.theta.find("act_bytes")) == 4
    assert hc.theta.find("act_format") == "fp32"
    assert hc.theta.find("weight_format") == "fp32"
    assert int(router.theta.find("K")) == 4096
    assert int(router.theta.find("N")) == 288
    assert router.theta.find("act_format") == router.theta.find("weight_format") == "fp32"
    assert float(router.theta.find("weight_payload")) == pytest.approx(288 * 4096 * 4)

    for operator, kind in ((hc, hc_kind), (router, router_kind)):
        compiled = pipeline.compiler.compile_operator(operator, kind=kind)
        cost = compiled.evaluate(geometry, readiness=False)
        tile_m = int(cost.get("tile_m"))
        tile_n = int(cost.get("tile_n"))
        issued = [4 * extent_m * int(operator.theta.require("K"))
                  + 4 * extent_n * int(operator.theta.require("K"))
                  for _c, extent_m, extent_n, _a in owner_tiles(
                      24, int(operator.theta.require("N")), tile_m, tile_n, units)]
        bound = max(issued) / per_unit
        assert float(cost.duration_ms) == pytest.approx(bound, rel=0.10)
        assert float(cost.duration_ms) > 0.010
        assert cost.dominant_pipe == "mte2"
        doubled = _evaluate_supply(compiled, geometry, rate_pipe=("mte2", rate * 2))
        assert float(doubled.duration_ms) == pytest.approx(float(cost.duration_ms) / 2, rel=0.05)
        raised = _evaluate_supply(compiled, geometry, cube_minimum=0.010)
        assert float(raised.duration_ms) == pytest.approx(float(cost.duration_ms), rel=1e-9)
    assert int(found[("R0", "hc_projection")][0].theta.require("N")) == 24


def _evaluate_supply(compiled, geometry, *, rate_pipe=None, cube_minimum=None):
    supply = compiled.supply
    rates = supply.rates
    invocations = supply.invocations
    if rate_pipe is not None:
        pipe, rate = rate_pipe
        rates = tuple((name, rate if name == pipe else value) for name, value in rates)
    if cube_minimum is not None:
        invocations = tuple(
            (name, cube_minimum if name == "cube" else value)
            for name, value in invocations)
    compiled.supply = PipeSupply(
        rates, supply.structure_caps, invocations, supply.resource_units,
        supply.flops_by_dtype, supply.undeclared_pipes)
    try:
        return compiled.evaluate(geometry, readiness=False)
    finally:
        compiled.supply = supply
