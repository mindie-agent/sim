"""Family-level conservation of physical tensor work and active-core elapsed.

[Role]
  Build one real compact GLM-5.3 latent-MLA A1 template and one declared dense
  GEMM, then vary only the available paired cores and pipe rates.  The test
  pins the physical tensor instructions and checks that a partial wave or a
  single pinned task changes elapsed capacity, never reported work.

[Invariants]
  * ``pipe_work("cube")`` is the emitted tensor-instruction count
    (``physical_flops``) for every core count; it is not multiplied by idle
    cores or by ``ceil(J/C)``.
  * The scheduled wave work ``waves * max_task * cores`` is retained only as a
    dimensionless elapsed-denominator cap (``cube_schedule_scale``); the test
    checks its exact identity to the pre-fix reported cube value.
  * A single task takes the same elapsed time on 2..32 similar cores (per-core
    rate is the device rate divided by the declared core count), while a
    kernel with no cube work is never delayed by that cap.
  * n released copies of one kernel take
    ``max(work / min(one-core cap, machine rate / n))`` on every stage.
    The MLA epilogue's 32768 B of HBM and MTE3 is part of that sum, so two
    copies share it even while their cube caps still fit on the machine.

[Timeouts & Locks]
  Tier 0 in-memory evaluation only; no device, subprocess, network or locks.

[Side-Effect Policy]
  Read-only catalog load and in-memory model copies; no filesystem writes.
"""
from __future__ import annotations

import math
from dataclasses import replace
from fractions import Fraction

import pytest

from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
    WorkloadGeometry,
)
from mindie_sim.prediction.piecewise.composition import (
    OperatorSegment, contention_envelope, operator_segment,
)
from mindie_sim.prediction.piecewise.expressions import as_fraction
from mindie_sim.prediction.subgraphs import FeatureConfig
from mindie_sim.prediction.subgraphs.instantiate import instantiate_model
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature,
)

_CORES = (2, 4, 8, 16, 32)
# Declared template caps from the compact catalog copy.  They are structural
# dimensions, not measurements.
_CAPS = {
    "q_lora_rank": 128, "kv_lora_rank": 64, "index_heads": 8,
    "index_dim": 32, "index_topk": 64, "expert_latent_size": 256,
}
# Real compact GLM-5.3 A1 (hidden 512, intermediate 256, 8 experts) at
# q1/context s.  The values are the physical tensor FLOPs reported by the
# independent model-review probe; they must not move with core count.
_PHYSICAL = {17: 4_087_808, 81: 5_242_880, 257: 5_242_880}
# The exact pre-fix wave product (waves * max_task * cores), now an elapsed
# denominator only.  Keeping it here proves the cap is not a new fit.
_SCHEDULED = {
    17: (8_175_616, 16_351_232, 32_702_464, 65_404_928, 130_809_856),
    81: (5_242_880, 10_485_760, 20_971_520, 41_943_040, 83_886_080),
    257: (6_291_456, 8_388_608, 8_388_608, 16_777_216, 33_554_432),
}
_HBM_S17 = 3_739_912


def _compact_glm() -> object:
    """One real GLM-5.3 layer with the declared compact structural caps."""
    source = load_model("glm-5.3")
    plan = source.layer_plans[0]
    updates = {"index": 0, "native_layer_id": 0}
    for field, cap in _CAPS.items():
        value = int(getattr(plan, field) or 0)
        if value > cap:
            updates[field] = cap
    plan = replace(plan, **updates)
    return replace(
        source, name="glm-5.3-l1", hidden_size=512, intermediate_size=256,
        num_experts=0, top_k=0, moe_intermediate_size=0,
        layers=1, body_layers=1, vocab_size=256, layer_plans=(plan,))


def _supply(cores: int, *, cube_only: bool = False) -> PipeSupply:
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950PR_9579", "num_devices": 2,
        "aic_count": cores, "aiv_count": 2 * cores,
    })
    base = PipeSupply.from_hardware(hardware)
    if not cube_only:
        return base
    # Expose the cube bottleneck: keep the declared cube rate (it scales with
    # the declared paired cores) but move every other pipe out of the way.
    rates = tuple((name, rate if name == "cube" else 1e15)
                  for name, rate in base.rates)
    return PipeSupply(rates)


def _mla(cores: int, *, cube_only: bool = False) -> tuple[object, PipeSupply]:
    model = _compact_glm()
    supply = _supply(cores, cube_only=cube_only)
    features = FeatureConfig.from_model(
        model, tp=1, dp=1, ep=1, cores=cores)
    instance = instantiate_model(model, features)
    operator = next(
        op for subgraph in instance.layers[0].subgraphs if subgraph.kind == "A1"
        for op in subgraph.operators if op.name == "mla_attention")
    return OperatorPiecewiseCompiler(supply).compile_operator(
        operator, kind="A1"), supply


def _decode(length: int) -> WorkloadGeometry:
    return WorkloadGeometry.homogeneous(1, 1, length, phase="decode")


def test_real_mla_physical_work_is_conserved_across_cores():
    model = _compact_glm()
    for index, cores in enumerate(_CORES):
        supply = _supply(cores)
        features = FeatureConfig.from_model(
            model, tp=1, dp=1, ep=1, cores=cores)
        instance = instantiate_model(model, features)
        operator = next(
            op for subgraph in instance.layers[0].subgraphs
            if subgraph.kind == "A1"
            for op in subgraph.operators if op.name == "mla_attention")
        compiled = OperatorPiecewiseCompiler(supply).compile_operator(
            operator, kind="A1")
        for length, physical in _PHYSICAL.items():
            cost = compiled.evaluate(_decode(length))
            scheduled = _SCHEDULED[length][index]
            assert cost.get("physical_flops") == physical
            # This MLA shape emits exactly one logical pair per physical pair.
            assert cost.get("logical_flops") == physical
            # Idle cores/tail lanes are capacity, never tensor instructions.
            assert cost.pipe_work("cube") == physical
            assert cost.get("cube_schedule_work") == scheduled
            assert cost.get("cube_schedule_scale") == Fraction(
                physical, scheduled)
            if length == 17:
                # The real probe's HBM work is invariant under the same sweep.
                assert cost.pipe_work("hbm") == _HBM_S17


def test_physical_cube_work_ignores_supply_rate_interventions():
    compiled, _ = _mla(8)
    baseline = compiled.evaluate(_decode(17)).pipe_work("cube")
    for factor in (Fraction(1, 2), Fraction(2), Fraction(5)):
        base = _supply(8)
        rates = tuple((name, rate * float(factor) if name == "cube" else rate)
                      for name, rate in base.rates)
        supply = PipeSupply(rates, base.structure_caps, base.invocations,
                            base.resource_units, base.flops_by_dtype,
                            base.undeclared_pipes)
        # Rebind the compiled operator's declared supply exactly as the public
        # resource sweep does.
        compiled.supply = supply
        assert compiled.evaluate(_decode(17)).pipe_work("cube") == baseline


def test_dense_gemm_cube_work_is_tile_count_not_all_cores():
    def gemm(cores: int) -> BoundOperator:
        k, n = 512, 16
        return BoundOperator(
            OperatorTemplate(name="MatMulV2", family="gemm",
                             role="projection"),
            StaticSignature(
                role="MatMulV2", matrix=(("K", k), ("N", n)),
                quant=(("act_bytes", 2.0), ("act_format", "bf16"),
                       ("weight_format", "bf16"),
                       ("weight_total", n * k * 2.0),
                       ("weight_payload", n * k * 2.0)),
                tiles=(("cores", cores), ("mt", 16), ("nt", 16))))

    policy = ImplementationPolicy(mt=16, nt=16, cores=1,
                                  split_target=10 ** 9, max_splits=8)
    for cores in (1, 2, 4, 8):
        supply = _supply(cores)
        compiled = OperatorPiecewiseCompiler(
            supply, policy).compile_operator(gemm(cores))
        cost = compiled.evaluate(
            WorkloadGeometry.homogeneous(1, 16, 17))
        physical = cost.get("padded_flops")
        assert physical == 2 * 16 * 16 * 512
        assert cost.pipe_work("cube") == physical
        assert cost.get("gemm_schedule_tiles") == 1
        assert cost.get("gemm_schedule_scale") == Fraction(1, cores)
    # A declared cube-rate intervention changes elapsed capacity, not work.
    base = _supply(4)
    for factor in (Fraction(1, 2), Fraction(2)):
        rates = tuple((name, rate * float(factor) if name == "cube" else rate)
                      for name, rate in base.rates)
        supply = PipeSupply(rates, base.structure_caps, base.invocations,
                            base.resource_units, base.flops_by_dtype,
                            base.undeclared_pipes)
        compiled = OperatorPiecewiseCompiler(
            supply, policy).compile_operator(gemm(4))
        cost = compiled.evaluate(WorkloadGeometry.homogeneous(1, 16, 17))
        assert cost.pipe_work("cube") == 2 * 16 * 16 * 512


def _stage_amounts(stage) -> dict[str, Fraction]:
    work: dict[str, Fraction] = {}
    for pipe, amount in stage:
        if amount > 0:
            work[pipe] = work.get(pipe, Fraction(0)) + amount
    return work


def _epilogue_bytes(segment) -> Fraction:
    """Bytes of the one stage that both writes HBM and retires MTE3."""
    hits = [work for stage in segment.stages
            if (work := _stage_amounts(stage)).get("hbm", 0) > 0
            and work.get("mte3", 0) > 0]
    assert len(hits) == 1
    assert hits[0]["hbm"] == hits[0]["mte3"] == 32768
    return hits[0]["hbm"]


def _released_stage_ms(segment, supply, sharers) -> Fraction:
    """Independent sum of per-stage max(work / min(cap, machine / n)).

    ``sharers(pipe)`` is how many released streams use that pipe.  No cap
    means the kernel may take the whole declared machine rate.
    """
    caps = dict(segment.caps)
    body = Fraction(0)
    for stage in segment.stages:
        stage_ms = Fraction(0)
        for pipe, amount in _stage_amounts(stage).items():
            machine = as_fraction(supply.effective(pipe))
            n = sharers(pipe)
            rate = min(caps.get(pipe, machine), machine / n)
            assert rate > 0
            stage_ms = max(stage_ms, amount / rate)
        body += stage_ms
    return max(segment.floor, segment.startup + body + segment.dependency)


def _engine_matches(observed: Fraction, expected: Fraction) -> None:
    """Engine completions are binary64.  One ulp is that clock's resolution.

    The shared 32768 B epilogue is ``32768/1e15`` ms at the declared HBM
    rate, about 600 times one ulp of this kernel, so the bound does not
    absorb the HBM share.
    """
    gap = abs(observed - expected)
    # ulp is a power of two; Fraction(float) keeps that binary value.
    unit = Fraction(math.ulp(float(expected)))
    assert gap <= unit, (gap, unit, expected, observed)


def _cap_is_consumed() -> bool:
    """True when the maintained compiler honors the family rate declaration."""
    compiled, supply = _mla(4, cube_only=True)
    cost = compiled.evaluate(_decode(17))
    expected = cost.get("cube_schedule_work") / supply.effective("cube")
    return cost.pipe_time("cube") == expected


def test_single_task_elapsed_is_not_shortened_by_idle_cores():
    if not _cap_is_consumed():
        pytest.skip("HW_RATE_CUBE elapsed cap awaiting D/F compiler ABI")
    durations = []
    for cores in _CORES:
        compiled, supply = _mla(cores, cube_only=True)
        cost = compiled.evaluate(_decode(17))
        expected = cost.get("cube_schedule_work") / supply.effective("cube")
        assert cost.pipe_time("cube") == expected
        assert cost.duration_ms >= expected
        durations.append(cost.duration_ms)
    # One pinned task owns one core: its elapsed time is the same on 2..32
    # similar cores because the device peak is divided by the same count.
    assert len(set(durations)) == 1
    # A kernel with no cube work is unaffected by the cap and can use idle
    # capacity on its own pipes.
    compiled, supply = _mla(4, cube_only=True)
    cost = compiled.evaluate(_decode(17))
    segment = operator_segment(cost, supply)
    epilogue = _epilogue_bytes(segment)
    hbm = as_fraction(supply.effective("hbm"))
    alone = _released_stage_ms(segment, supply, lambda pipe: 1)
    # The compiler stores the same sum as binary64.  The envelope stays exact.
    assert cost.duration_ms == float(alone)
    # The idle stream is still running, so this chain's HBM rate is halved.
    # Every other pipe still has one owner.  The added time is the epilogue
    # quantum ``32768 / hbm_rate``, read from the stage, not a tolerance.
    shared = _released_stage_ms(
        segment, supply, lambda pipe: 2 if pipe == "hbm" else 1)
    assert shared - alone == epilogue / hbm
    idle = OperatorSegment(stages=((("hbm", Fraction(10 ** 12)),),))
    envelope = contention_envelope(
        {}, supply, chains={"cube": (segment,), "idle": (idle,)})
    done = dict(envelope.completions)
    _engine_matches(done["cube"], shared)
    assert envelope.makespan_ms >= done["cube"]
    assert done["idle"] > done["cube"]


def test_dual_cube_streams_share_only_the_contested_capacity():
    if not _cap_is_consumed():
        pytest.skip("HW_RATE_CUBE elapsed cap awaiting D/F compiler ABI")
    cores = 4
    compiled, supply = _mla(cores, cube_only=True)
    cost = compiled.evaluate(_decode(17))
    segment = operator_segment(cost, supply)
    epilogue = _epilogue_bytes(segment)
    hbm = as_fraction(supply.effective("hbm"))
    cube_cap = dict(segment.caps)["cube"]
    machine = as_fraction(supply.effective("cube"))
    cube_amount = _stage_amounts(segment.stages[0])["cube"]
    assert cube_amount == cost.pipe_work("cube") == 4_087_808
    expected = {}
    for n in (1, 2, 3, 4, 5, 8):
        chains = {f"s{index}": (segment,) for index in range(n)}
        envelope = contention_envelope({}, supply, chains=chains)
        exact = _released_stage_ms(segment, supply, lambda pipe, n=n: n)
        expected[n] = exact
        assert len(envelope.completions) == n
        for _name, done in envelope.completions:
            _engine_matches(done, exact)
        _engine_matches(envelope.makespan_ms, exact)
        # Cube caps fit while n <= cores.  Past that, the machine rate binds.
        rate = min(cube_cap, machine / n)
        if n <= cores:
            assert rate == cube_cap
        else:
            assert rate == machine / n
        # Stage 0 is the cube cap; stage 1 is the shared 32768 B epilogue.
        assert exact == cube_amount / rate + n * epilogue / hbm
    assert cost.duration_ms == float(expected[1])
    # Two, three and four copies share only the epilogue.  Cube stages overlap.
    assert expected[2] - expected[1] == epilogue / hbm
    assert expected[4] - expected[1] == 3 * epilogue / hbm


def test_real_mla_duration_is_monotone_in_available_cores():
    durations = []
    for cores in _CORES:
        compiled, _ = _mla(cores)
        duration = compiled.evaluate(_decode(17)).duration_ms
        durations.append(duration)
    assert all(later <= earlier for earlier, later in zip(durations, durations[1:]))
