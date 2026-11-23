"""Mechanical rewrite verification and constructive search contracts.

[Role] Exercise each verification gate and the finite solver candidate spaces.
[Invariants] Evidence is physical, work is conserved, and search retains feasible seeds.
[Timeouts & Locks] Small, deterministic fixtures with bounded analytical replay.
[Side-Effect Policy] Only in-memory graph and receipt mutations.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from math import gcd, prod
from types import SimpleNamespace

import pytest

from _support.axes import REPRESENTATIVE_SKUS, SKUS
from mindie_sim.compose import compose_case
from mindie_sim.optimization import (
    EventDagSpec, EvgVisitorNode, FusedRegionSummary, ResourceChoiceProgram,
    SemanticWorkItem, TensorRangeObligation, build_dual_inputs_from_scenario,
    certify_lower_bound, generate_gcd_core_lattice, select_catlass_candidates,
    select_dsl_candidates, solve_dsl, solve_framework, streamk_crossing_partials,
)
from mindie_sim.prediction.results import OpOccurrence, OperatorSequenceRow, StepEvaluation, StepRequest
from mindie_sim.prediction.runtime.memory_physics import AddressRun
from mindie_sim.prediction.runtime.tile_solver import LiveBufferSpec, TileCandidate, TilePipelineSpec, TileShape
from mindie_sim.prediction.streams import StreamPlan, bind_stream_owners, proportional_coupled_placement
from mindie_sim.rewrite import (
    ACCEL_MAX_R, ACCEL_MAX_T, ACCEL_MAX_U, Edit, FuseRegion, KernelBoundary, OpRef, PipelineTiles,
    SelectTemplate, ShippingRegression, SoundnessInversionError, StreamCoreBudget, StreamLane,
    apply_plan, compute_overlap_pct, empty_plan, resolve_scenario, verify_plan,
)
from mindie_sim.rewrite.apply import PrefetchLease
from mindie_sim.rewrite.oracle import ScenarioRequest, prepare_scenario
from mindie_sim.schema.compute_arch import NativeComputeMode
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.templates import CatlassProblem
from mindie_sim.templates.dsl_catalog import ATTN_RES, AttnResProblem

pytestmark = pytest.mark.tier0


@pytest.fixture(scope="module")
def real_graph():
    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1)
    scenario = resolve_scenario(case, StepRequest("decode", tokens=4, context=64))
    return scenario, apply_plan(_empty(scenario), scenario)


def _empty(scenario):
    return empty_plan(scenario.base_graph_sha256, scenario.framing_sha256)


def _reason(receipt, index):
    gate = receipt.gates[index]
    assert not gate.passed
    return gate.reason


def _certificate_evidence(program, dag):
    cert, _ = certify_lower_bound(program, dag)
    return dict(dual_program=program, event_dag=dag, dual_certificate=cert)


def _with_die(scenario, die):
    machine = replace(scenario.case.machine, dies=(die,))
    return replace(scenario, case=replace(scenario.case, machine=machine), datapath=datapath_for_die(die))


@pytest.mark.parametrize(("defect", "reason"), (
    ("region", "NON_CONVEX_REGION"),
    ("credits", "CREDIT_CYCLE"),
    ("alias", "ALIAS_CONFLICT"),
    ("event", "STREAM_EVENT_UNCOVERED"),
))
def test_structure_gate_rejects_unproved_dependencies(real_graph, defect, reason):
    scenario, applied = real_graph
    assert verify_plan(scenario, applied).gates[0].passed
    qkv, norm, rope = (f"s0.l0.{suffix}" for suffix in ("qkv", "qk_norm", "rope"))
    if defect == "region":
        changed = replace(applied, absorbed_atoms=((rope, (qkv, rope)),))
    elif defect == "credits":
        edges = ((qkv, norm, 0), (norm, qkv, 0))
        changed = replace(applied, credit_edges=edges)
        assert verify_plan(scenario, replace(applied, credit_edges=(edges[0], (norm, qkv, 1)))).gates[0].passed
    elif defect == "alias":
        changed = replace(applied, in_place_aliases=((qkv, norm),))
    else:
        lanes = (StreamLane("main"), StreamLane("aux", StreamCoreBudget(0, 2)))
        changed = replace(applied, streams=(StreamPlan(0, lanes, ((norm, "aux"),)),))
    assert _reason(verify_plan(scenario, changed), 0) == reason


@pytest.mark.parametrize("pool", ("ub", "l1", "l0a", "l0b", "l0c", "l2", "hbm"))
def test_capacity_gate_checks_physical_owner_peak_at_exact_boundary(real_graph, pool):
    scenario, applied = real_graph
    die, = scenario.case.machine.dies
    capacity = die.memory_bytes if pool == "hbm" else dict(die.buffers)[pool]
    key = f"{pool}@d{die.rank}:unit0"

    def verify(peak):
        evidence = SimpleNamespace(span_ms=1, execution_evidence={"peak_allocated_per_unit": {key: peak}})
        return verify_plan(scenario, applied, step_eval=evidence)

    assert capacity > 0 and verify(capacity).gates[1].passed
    over = verify(capacity + 1)
    assert _reason(over, 1) == "MULTI_POOL_CAPACITY"
    assert key in over.gates[1].detail


def test_capacity_gate_accounts_for_simt_carveout_and_die_owned_l2(real_graph):
    scenario, base = real_graph
    die, = scenario.case.machine.dies
    assert dict(die.buffer_units)["l2"] == 1
    invalid = _with_die(scenario, replace(die, buffer_units=tuple(
        {**dict(die.buffer_units), "l2": 2}.items(),
    )))
    assert _reason(verify_plan(invalid, base), 1) == "MULTI_POOL_CAPACITY"
    assert "one_die_owner" in verify_plan(invalid, base).gates[1].detail

    target = scenario.atomic_ops[0].ops[0]
    plan = replace(_empty(scenario), edits=(Edit("simt", operation=SelectTemplate(
        OpRef("atom", target.op_id), "K3", "vector", (("uses_simt", True),),
    )),))
    simt = apply_plan(plan, scenario)
    carveout = 32 * 1024
    assert scenario.datapath.simt_ub_carveout_bytes == carveout
    capacity = dict(die.buffers)["ub"] - carveout

    def replay(peak):
        return SimpleNamespace(span_ms=1, execution_evidence={"peak_allocated_per_unit": {"ub": peak}})

    assert verify_plan(scenario, base, step_eval=replay(capacity + 1)).gates[1].passed
    assert verify_plan(scenario, simt, step_eval=replay(capacity)).gates[1].passed
    assert _reason(verify_plan(scenario, simt, step_eval=replay(capacity + 1)), 1) == "MULTI_POOL_CAPACITY"


def test_capacity_gate_aggregates_overlapping_prefetch_leases(real_graph):
    scenario, applied = real_graph
    anchor, first, *rest = scenario.atomic_ops[0].ops
    second = next(op for op in rest if op.family == "output_projection")
    capacity = scenario.datapath.usable_bytes("l1")
    leases = (PrefetchLease("first", first.op_id, "weight", anchor.op_id, "l1", capacity // 2),
              PrefetchLease("second", second.op_id, "weight", anchor.op_id, "l1", capacity // 2))
    planned = replace(applied, prefetch_leases=leases)
    assert verify_plan(scenario, planned).gates[1].passed
    too_large = replace(planned, prefetch_leases=(leases[0], replace(leases[1], max_bytes=capacity // 2 + 1)))
    receipt = verify_plan(scenario, too_large)
    assert _reason(receipt, 1) == "MULTI_POOL_CAPACITY"
    assert "l1:aic" in receipt.gates[1].detail


@pytest.fixture(scope="module")
def lane_graph(real_graph):
    original, _ = real_graph
    die, = original.case.machine.dies
    die = replace(die, compute_units=4, resource_units=tuple({**dict(die.resource_units), "vector": 6}.items()))
    scenario = _with_die(original, die)
    rank, = scenario.atomic_ops
    by_id = {op.op_id: op for op in rank.ops}
    mixed = replace(by_id["s0.l0.attention"], op_id="mixed", deps=(), metadata=())
    vector = replace(by_id["s0.l0.norm1"], op_id="vector", deps=(), metadata=())
    comm = replace(by_id["s0.l0.qkv.value"], op_id="comm", deps=(), metadata=())
    assert mixed.flops and mixed.vector_ops and vector.vector_ops and not vector.flops
    assert comm.flops == comm.vector_ops == 0
    scenario = replace(scenario, atomic_ops=(replace(rank, ops=(mixed, vector, comm)),))
    return scenario, apply_plan(_empty(scenario), scenario)


def _streams(*budgets, priorities=(0, 0, 0)):
    names = ("main", "csa", "comm")
    lanes = tuple(StreamLane(name, StreamCoreBudget(*budget), priority)
                  for name, budget, priority in zip(names, budgets, priorities, strict=True))
    return StreamPlan(0, lanes, (("vector", "csa"), ("comm", "comm")))


def _timeline(scenario, intervals):
    rows = tuple(OperatorSequenceRow(
        OpOccurrence(0, 0, 0, op.op_id), i, op.family, op.recipe, (), lane, op.local_shape,
        op.dtype, (), lo, lo, hi,
    ) for i, (op, lane, (lo, hi)) in enumerate(zip(
        scenario.atomic_ops[0].ops, ("main", "csa", "comm"), intervals, strict=True,
    )))
    return StepEvaluation("lanes", scenario.step, "should_achieve", max(row.end_ms for row in rows), rows, ())


@pytest.mark.parametrize("shared", (False, True), ids=("disjoint", "shared-cap"))
def test_core_gate_binds_exact_coupled_group_and_pure_lanes(lane_graph, shared):
    scenario, applied = lane_graph
    main = (2, 3, "shared_cap") if shared else (2, 3)
    plan = _streams(main, (0, 3), (0, 0))
    owners = bind_stream_owners(plan, scenario.datapath)
    coupled_owner = owners["main"]
    if shared:
        coupled_owner = replace(coupled_owner, pool_aic=coupled_owner.pool_aic[:2],
                                pool_aiv=coupled_owner.pool_aiv[:3])
    aic, aiv = proportional_coupled_placement(coupled_owner, scenario.datapath)
    assert (len(aic), len(aiv)) == (2, 3)
    assert owners["csa"].aic_limit == owners["comm"].aic_limit == owners["comm"].aiv_limit == 0
    assert set(aiv).isdisjoint(owners["csa"].aiv)
    step = _timeline(scenario, ((0, 10), (4, 14), (6, 16)))
    program, dag, _ = build_dual_inputs_from_scenario(scenario)
    evidence = _certificate_evidence(program, dag)
    receipt = verify_plan(scenario, replace(applied, streams=(plan,)), step_eval=step, **evidence)
    assert receipt.passed and receipt.overlap_pct == 60 and receipt.l_star_ns == evidence["dual_certificate"].l_star_lean_ns

    over = (5, 3, "shared_cap") if shared else (2, 4)
    bad = replace(applied, streams=(_streams(over, (0, 3), (0, 0)),))
    expected = "stream_shared_cap_aic_exceeded" if shared else "stream_core_budget_aiv_exceeded"
    assert _reason(verify_plan(scenario, bad, step_eval=step), 2) == expected


@pytest.mark.parametrize(("main", "vector", "reason"), (
    ((0, 3), (0, 3), "stream_zero_aic_budget"),
    ((2, 0), (0, 3), "stream_zero_aiv_budget"),
    ((1, 3), (0, 3), "stream_coupled_placement_zero"),
    ((2, 3), (2, 0), "stream_zero_aiv_budget"),
))
def test_core_gate_rejects_incompatible_owner_modes(lane_graph, main, vector, reason):
    scenario, applied = lane_graph
    planned = replace(applied, streams=(_streams(main, vector, (0, 0)),))
    assert _reason(verify_plan(scenario, planned), 2) == reason


def test_overlap_uses_busy_unions_and_rejects_empty_and_threshold_evidence(lane_graph):
    assert compute_overlap_pct(((0, 8), (4, 10)), ((5, 15),)) == 50
    assert compute_overlap_pct(((0, 8), (4, 10)), ((10, 10),)) is None
    scenario, applied = lane_graph
    planned = replace(applied, streams=(_streams((2, 3), (0, 3), (0, 0)),))
    step = _timeline(scenario, ((0, 20), (19, 39), (38, 58)))
    fake = verify_plan(scenario, planned, step_eval=step)
    assert _reason(fake, 2) == "FAKE_OVERLAP" and fake.overlap_pct == 5
    empty = _timeline(scenario, ((0, 20), (0, 20), (0, 0)))
    unproved = verify_plan(scenario, planned, step_eval=empty)
    assert _reason(unproved, 2) == "UNPROVED_OVERLAP" and unproved.overlap_pct is None


def test_capacity_gate_combines_tile_live_ranges_per_owner(lane_graph):
    scenario, _ = lane_graph
    capacity = scenario.datapath.usable_bytes("ub")

    def apply(bytes_per_tile):
        buffer = LiveBufferSpec("tile", "ub", bytes_per_tile, 0, 1)
        tile = TileCandidate("resident_1pass", TileShape(16, 16, 16), 1, (1, 1), 1, 1,
                             (buffer,), owner_axis="aiv")
        spec = TilePipelineSpec("scratch", (tile,))
        edits = tuple(Edit(name, operation=PipelineTiles(OpRef("atom", name), spec))
                      for name in ("mixed", "vector"))
        return apply_plan(replace(_empty(scenario), edits=edits), scenario)

    overlapping = _timeline(scenario, ((0, 10), (0, 10), (0, 0)))
    separated = _timeline(scenario, ((0, 10), (11, 21), (0, 0)))
    assert verify_plan(scenario, apply(capacity // 2), step_eval=overlapping).gates[1].passed
    over = apply(capacity // 2 + 32)
    assert verify_plan(scenario, over, step_eval=separated).gates[1].passed
    assert _reason(verify_plan(scenario, over, step_eval=overlapping), 1) == "MULTI_POOL_CAPACITY"


def test_core_gate_accepts_coupled_placement_with_only_shared_caps(lane_graph):
    scenario, applied = lane_graph
    plan = _streams((2, 3, "shared_cap"), (0, 3, "shared_cap"), (0, 0, "shared_cap"))
    step = _timeline(scenario, ((0, 10), (4, 14), (6, 16)))
    receipt = verify_plan(scenario, replace(applied, streams=(plan,)), step_eval=step)
    assert receipt.gates[2].passed, receipt.gates[2]


def test_native_preflight_reports_exact_builtin_coverage_and_fallback(lane_graph):
    scenario, applied = lane_graph
    healthy = verify_plan(scenario, applied)
    assert healthy.native_coverage_pct == 100 and not healthy.native_fallback_reasons

    class DerivedInt(int):
        pass

    rank, = applied.executable
    head, *rest = rank.ops
    tagged = replace(head, metadata=(("budget", DerivedInt(1)),))
    nonflat = replace(applied, executable=(replace(rank, ops=(tagged, *rest)),))
    fraction = verify_plan(scenario, nonflat)
    assert fraction.native_coverage_pct == pytest.approx(200 / 3)
    assert fraction.native_fallback_reasons == ("metadata_not_flat",)

    stream = _streams((2, 3), (0, 3), (0, 0), priorities=(0, 1, 0))
    mixed = replace(nonflat, streams=(stream,))
    evidence = {"resource_units": (ACCEL_MAX_U + 1,) * (ACCEL_MAX_R + 1),
                "peak_live_tasks": ACCEL_MAX_T + 1, "advance_steps": 8, "native_steps": 3}
    step = SimpleNamespace(span_ms=1, execution_evidence=evidence)
    receipt = verify_plan(scenario, mixed, step_eval=step)
    assert receipt.gates[1].passed
    assert receipt.native_coverage_pct == 37.5
    assert set(receipt.native_fallback_reasons) == {
        "resource_capacity_exceeded", "owner_capacity_exceeded", "task_or_link_capacity_exceeded",
        "mixed_priority", "metadata_not_flat",
    }


def _fused_summary(scenario):
    norm, matmul, *_ = scenario.atomic_ops[0].ops
    members = tuple(OpRef("atom", op.op_id) for op in (norm, matmul))
    edit = Edit("fused", operation=FuseRegion(members, (), (KernelBoundary("test", members),)))
    applied = apply_plan(replace(_empty(scenario), edits=(edit,)), scenario)
    atoms = (norm, matmul)
    semantic = (
        SemanticWorkItem("matrix", "matrix_mac", matmul.dtype, int(matmul.flops)),
        SemanticWorkItem("elementwise", "vector_alu", norm.dtype, int(norm.vector_ops), ("aiv_simd",)),
    )
    # These byte ranges describe a small, cold transaction with a partial-sector tail.
    read = TensorRangeObligation(0, "operand", 0, ((0, 17),), "hbm_cold", "owner", (AddressRun(12, 1, 15, 15),))
    write = TensorRangeObligation(0, "output", 1, ((0, 5),), "onchip_resident", "device")
    summary = FusedRegionSummary(
        "K1", "rewrite:fused", tuple(op.op_id for op in atoms), semantic,
        ((matmul.dtype, int(matmul.flops)),), (("vector", norm.vector_ops),), (read,), (write,),
        32, 64, 64, Fraction(256, 3), 0, (), 1, Fraction(0), (), None,
    )
    return applied, summary


def test_work_gate_conserves_semantic_work_on_contraction(real_graph):
    scenario, _ = real_graph
    applied, summary = _fused_summary(scenario)
    rank, = applied.executable
    fused, *rest = rank.ops
    assert (fused.flops, fused.vector_ops) == tuple(item.logical_ops for item in summary.semantic_work)
    assert verify_plan(scenario, applied, region_summaries=(summary,)).gates[3].passed
    for mutation in (replace(fused, flops=fused.flops + 1), replace(fused, vector_ops=fused.vector_ops - 1)):
        changed = replace(applied, executable=(replace(rank, ops=(mutation, *rest)),))
        assert _reason(verify_plan(scenario, changed, region_summaries=(summary,)), 3) == "WORK_CONSERVATION_MISMATCH"


def test_work_gate_rejects_forged_summary_work(real_graph):
    scenario, _ = real_graph
    applied, summary = _fused_summary(scenario)
    matrix, vector = summary.semantic_work
    forged = replace(summary, semantic_work=(replace(matrix, logical_ops=0), vector))
    receipt = verify_plan(scenario, applied, region_summaries=(forged,))
    assert not receipt.gates[3].passed


def test_empty_plan_rejects_certificate_for_unbound_cold_read():
    case = compose_case("qwen3-30b-a3b", REPRESENTATIVE_SKUS["a5"], tp=2, dp=1)
    scenario = resolve_scenario(case, StepRequest("decode", context=64))
    applied = apply_plan(_empty(scenario), scenario)
    cold_bytes = 511_212_000
    read = TensorRangeObligation(0, "external", 0, ((0, cold_bytes),), "hbm_cold", "owner")
    forged = FusedRegionSummary(
        "K2", "missing", ("missing",), (), (), (), (read,), (),
        cold_bytes, cold_bytes, cold_bytes, Fraction(cold_bytes), 0, (), 1, Fraction(0), (), None,
    )
    program, dag, _ = build_dual_inputs_from_scenario(scenario, region_summaries=(forged,))
    evidence = _certificate_evidence(program, dag)
    receipt = verify_plan(scenario, applied, wall_ns=evidence["dual_certificate"].l_star_lean_ns,
                          region_summaries=(forged,), **evidence)
    assert not receipt.passed and receipt.l_star_ns is None
    assert _reason(receipt, 3) == "LEDGER_RECONCILIATION_ERROR"
    assert receipt.gates[3].detail == "unbound_region_summary:missing"
    assert _reason(receipt, 4) == "LEAN_CERTIFICATE_INVALID"
    assert receipt.gates[4].detail == "certificate_unbound_to_scenario"


def test_fused_summary_requires_exact_region_members_and_one_summary_per_occurrence(real_graph):
    scenario, _ = real_graph
    applied, summary = _fused_summary(scenario)
    for summaries in ((replace(summary, region_id="missing"),),
                      (replace(summary, op_ids=summary.op_ids[:1]),),
                      (summary, summary)):
        receipt = verify_plan(scenario, applied, region_summaries=summaries, scope="archetype")
        assert _reason(receipt, 3) == "LEDGER_RECONCILIATION_ERROR"
        assert receipt.gates[3].detail.startswith("unbound_region_summary:")


@pytest.mark.parametrize("mutation", (
    {"logical_transfer_bytes": 31}, {"issued_copy_bytes": -1}, {"dram_sector_bytes": 96},
    {"turnaround_service_bytes": 85.5}, {"turnaround_service_bytes": Fraction(63)},
    {"realized_l2_hit_bytes": 65}, {"realized_l2_hit_bytes": 64},
))
def test_ledger_gate_rejects_inconsistent_transfer_evidence(real_graph, mutation):
    scenario, _ = real_graph
    applied, summary = _fused_summary(scenario)
    assert (summary.logical_transfer_bytes, summary.issued_copy_bytes, summary.dram_sector_bytes,
            summary.turnaround_service_bytes) == (32, 64, 64, Fraction(256, 3))
    receipt = verify_plan(scenario, applied, region_summaries=(replace(summary, **mutation),))
    assert _reason(receipt, 3) == "LEDGER_RECONCILIATION_ERROR"


@pytest.mark.parametrize("runs,logical,dram", (
    ((AddressRun(0, 4, 1, 64),), 4, 128),
    ((AddressRun(1, 4097, 1, 101), AddressRun(33, 4097, 1, 101)), 8194, 262208),
))
def test_ledger_gate_requires_strided_cold_sectors(real_graph, runs, logical, dram):
    scenario, _ = real_graph
    applied, summary = _fused_summary(scenario)
    read = replace(summary.read_obligations[0], ranges=(), address_runs=runs)
    valid = replace(summary, read_obligations=(read,), write_obligations=(),
                    logical_transfer_bytes=logical, issued_copy_bytes=dram,
                    dram_sector_bytes=dram, turnaround_service_bytes=Fraction(dram))
    assert verify_plan(scenario, applied, region_summaries=(valid,)).gates[3].passed
    for changes, detail in (
        (dict(dram_sector_bytes=dram - 1), "dram_sector_bytes"),
        (dict(logical_transfer_bytes=logical - 1), "logical_bytes"),
        (dict(issued_copy_bytes=1), "issued_copy_bytes"),
    ):
        gate = verify_plan(scenario, applied, region_summaries=(replace(valid, **changes),)).gates[3]
        assert (gate.passed, gate.reason, gate.detail) == (
            False, "LEDGER_RECONCILIATION_ERROR", f"{summary.region_id}:{detail}",
        )


def test_bound_gate_checks_certificate_and_shipping_threshold(real_graph):
    scenario, applied = real_graph
    program, dag, _ = build_dual_inputs_from_scenario(scenario)
    evidence = _certificate_evidence(program, dag)
    cert = evidence["dual_certificate"]
    lower = cert.l_star_lean_ns
    assert lower > 0
    verified = verify_plan(scenario, applied, wall_ns=lower, **evidence)
    assert verified.passed and verified.l_star_ns == lower and len(verified.gates) == 5
    with pytest.raises(SoundnessInversionError, match="SOUNDNESS_BUG_LOWER_BOUND_INVERTED"):
        verify_plan(scenario, applied, wall_ns=lower - 1, **evidence)
    for forged in (replace(cert, path_edges=(1,)), replace(cert, l_star_lean_ns=lower + 1)):
        receipt = verify_plan(scenario, applied, wall_ns=lower, **{**evidence, "dual_certificate": forged})
        assert _reason(receipt, 4) == "LEAN_CERTIFICATE_INVALID"
    receipt = verify_plan(scenario, applied, wall_ns=lower, dual_certificate=cert)
    assert _reason(receipt, 4) == "LEAN_CERTIFICATE_INVALID"
    assert _reason(verify_plan(scenario, applied, wall_ns=lower), 4) == "LEAN_CERTIFICATE_INVALID"
    with pytest.raises(SoundnessInversionError, match=f"dsl_ns={lower - 1}<l_star_ns={lower}"):
        verify_plan(scenario, applied, wall_ns=lower, framework_ns=lower, dsl_ns=lower - 1, **evidence)
    inverted = verify_plan(scenario, applied, wall_ns=lower + 1, framework_ns=lower, dsl_ns=lower + 1, **evidence)
    assert _reason(inverted, 4) == "LADDER_MONOTONICITY_VIOLATION"

    near = verify_plan(scenario, applied, wall_ns=lower, baseline_u_ns=lower, baseline_m_ns=lower + 1000, **evidence)
    assert near.shipping_regression is None
    slow = verify_plan(scenario, applied, wall_ns=lower, baseline_u_ns=lower, baseline_m_ns=lower + 1001, **evidence)
    assert slow.passed and slow.shipping_regression == ShippingRegression(
        delta_ns=1001, baseline_u_ns=lower, baseline_m_ns=lower + 1001,
    )


@pytest.mark.parametrize(("aic", "aiv"), ((4, 6), (6, 9), (4, 8), (0, 8), (4, 0), (0, 0)))
def test_gcd_lattice_enumerates_finest_proportional_and_pure_lanes(aic, aiv):
    common = gcd(aic, aiv)
    step = (aic // common, aiv // common) if common else (0, 0)
    points = generate_gcd_core_lattice(aic, aiv, include_three_stream=True)
    assert {point.isolation for point in points} == {"disjoint", "shared_cap"}
    budgets = set()
    tri = set()
    for point in points:
        assert point.gcd_step == step
        assert all(lane.budget.isolation == point.isolation for lane in point.lanes)
        by_lane = {lane.stream_id: (lane.budget.aic, lane.budget.aiv) for lane in point.lanes}
        assert by_lane["comm"] == (0, 0)
        for cube, vector in by_lane.values():
            assert 0 <= cube <= aic and 0 <= vector <= aiv
            if cube and vector:
                assert cube * aiv == vector * aic
        if point.isolation == "disjoint":
            assert all(sum(use) <= total for use, total in zip(zip(*by_lane.values()), (aic, aiv)))
        budgets.add(by_lane["main"])
        if len(by_lane) == 3:
            assert tuple(by_lane) == ("main", "csa", "comm")
            tri.add((point.isolation, by_lane["main"], by_lane["csa"]))
    assert {pair for pair in budgets if all(pair)} == {
        (k * step[0], k * step[1]) for k in range(1, common + 1) if all(step)
    }
    assert {(k, 0) for k in range(aic + 1)} | {(0, k) for k in range(aiv + 1)} <= budgets
    if min(aic, aiv):
        assert ("shared_cap", (aic, aiv), (0, aiv)) in tri
        assert ("disjoint", (aic, 0), (0, aiv)) in tri
    else:
        assert not tri


def test_fractional_mode_balance_has_exact_dual_objective():
    # Equalizing 6 + 10t and 2 + 14(1 - t) yields t=5/12 and wall=61/6.
    program = ResourceChoiceProgram(2, (6, 2), (((10, 0), (0, 14)),))
    cert, prices = certify_lower_bound(program, EventDagSpec((0,), ()))
    assert prices.lp_optimum_ns == Fraction(61, 6)
    assert cert.farkas.lower_bound_ns == cert.l_star_lean_ns == 10


@pytest.mark.parametrize("hardware", SKUS)
def test_constructive_search_keeps_best_fully_replayed_seed(hardware):
    case = compose_case("qwen3-32b", hardware, tp=1, dp=1, layers=1)
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", context=64)))
    program, dag, _ = build_dual_inputs_from_scenario(dossier.handle.scenario)
    cert, pricing = certify_lower_bound(program, dag)
    options = dict(dual_program=program, event_dag=dag, dual_certificate=cert, dual_report=pricing)
    seed = solve_framework(dossier, max_trials=0, **options)
    assert seed.tried_candidates == 0
    assert seed.best_wall_ns == min(dossier.m_wall_ns, dossier.u_wall_ns)
    assert seed.best_plan in (dossier.unmerge_plan, dossier.mainline_plan)
    framework = solve_framework(dossier, max_trials=4, **options)
    dsl_seed = solve_dsl(dossier, framework, max_trials=0, **options)
    assert (dsl_seed.best_plan, dsl_seed.best_eval) == (framework.best_plan, framework.best_eval)
    dsl = solve_dsl(dossier, framework, max_trials=4, **options)
    assert cert.l_star_lean_ns <= dsl.best_wall_ns <= framework.best_wall_ns <= seed.best_wall_ns
    for result in (seed, framework, dsl_seed, dsl):
        assert result.best_eval.verification.passed and result.best_eval.scope == "full"
        assert result.best_eval.wall_ns == result.best_wall_ns
        assert result.stage_history[-1][1] == result.best_wall_ns
        assert all(left[1] >= right[1] for left, right in zip(result.stage_history, result.stage_history[1:]))


def test_streamk_counts_crossings_against_explicit_core_assignment():
    assert streamk_crossing_partials(b_tot=8, g_k=2, c_aic=4) == 0
    assert streamk_crossing_partials(b_tot=6, g_k=2, c_aic=4) == 4
    for blocks in range(1, 12):
        for cores in range(1, 7):
            for slices in range(1, 5):
                owners = [[] for _ in range(cores)]
                count = (blocks % cores) * slices
                base, extra = divmod(count, cores)
                chunk_start = 0
                for core, segment in enumerate(owners):
                    end = chunk_start + base + (core < extra)
                    segment.extend(range(chunk_start, end))
                    chunk_start = end
                assert streamk_crossing_partials(blocks, slices, cores) == sum(
                    len({iteration // slices for iteration in segment}) for segment in owners
                )


@pytest.fixture
def matrix_op(real_graph):
    scenario, _ = real_graph
    base = next(op for op in scenario.atomic_ops[0].ops if op.family == "qkv_projection")
    shape = (64, 192, 384)
    visitors = (EvgVisitorNode("acc", "acc_load", "fp32"),
                EvgVisitorNode("store", "aux_store", "bf16", inputs=("acc",), aux_tensor_id="output"))
    return replace(base, family="gemm", dtype="bf16", local_shape=shape, flops=2 * prod(shape),
                   metadata=(("evg_tree", visitors), ("l1_tile_shape", (64, 64, 128))))


@pytest.mark.parametrize("hardware", SKUS)
def test_catlass_scheduler_choices_preserve_workspace_and_dispatch_constraints(matrix_op, hardware):
    case = compose_case("qwen3-32b", hardware, tp=1, dp=1, layers=1)
    path = resolve_scenario(case, StepRequest("decode", context=64)).datapath
    specs = select_catlass_candidates(matrix_op, path, 4, 8)
    by_id = {spec.template_id: spec for spec in specs}
    assert {"catlass.basic_pingpong", "catlass.mutex_block_mmad", "catlass.streamk",
            "catlass.multi_core_splitk", "catlass.swizzle", "catlass.evg"} <= by_id.keys()
    assert {s.splitk_factor for s in specs if s.template_id == "catlass.multi_core_splitk"} == {2, 3}
    assert {s.swizzle_direction for s in specs if s.template_id == "catlass.swizzle"} == {"Zn", "Nz"}
    assert {s.epilogue_kind for s in specs if s.template_id == "catlass.evg"} == (
        {"evg_gm", "evg_ub"} if path.has_l0c_to_ub else {"evg_gm"}
    )
    assert all((s.aic_budget, s.aiv_budget) == (4, 8) for s in specs)
    grouped = replace(matrix_op, family="grouped_gemm", metadata=(
        *matrix_op.metadata, ("group_rows", (32, 32)),
    ))
    grouped_spec, = select_catlass_candidates(grouped, path, 4, 8)
    assert grouped_spec.scheduler == "grouped_slice_m" and grouped_spec.l2_cache_hint_b == "disable"


def test_catlass_dispatch_requires_native_operand_mode(real_graph, matrix_op):
    scenario, _ = real_graph
    path = scenario.datapath
    problem = CatlassProblem(32, 32, 128, "fp8_e4m3", "fp8_e4m3", mx_mode="a8w4_mx")
    invalid = replace(matrix_op, metadata=(("catlass_problem", problem),))
    assert all(spec.template_id != "catlass.mx_matmul" for spec in select_catlass_candidates(invalid, path, 4, 8))
    valid = replace(invalid, metadata=(("catlass_problem", replace(problem, dtype_b="fp4_e2m1")),
                                       ("l1_tile_shape", (32, 32, 128)), ("mutex_stages", (1, 1, 1, 1, 1))))
    arch = path.compute_arch.declared
    native_mode = NativeComputeMode("a8w4_mx", "matrix", ("a8w4_mx",), "cube", ("fp32",))
    if not arch.supports_native("matrix", "a8w4_mx"):
        arch = replace(arch, native_modes=(*arch.native_modes, native_mode))
        path = replace(path, compute_arch=replace(path.compute_arch, declared=arch))
    specs = select_catlass_candidates(valid, path, 4, 8)
    assert any(s.template_id == "catlass.mx_matmul" and s.prologue_b == "DecompressFp4ToFp8" for s in specs)


def test_dsl_catalog_requires_real_pool_and_owner_capability(real_graph, matrix_op):
    scenario, _ = real_graph
    path = scenario.datapath
    problem = AttnResProblem(tokens=1)
    op = replace(matrix_op, family=ATTN_RES.removeprefix("dsl."), deps=(),
                 metadata=(("dsl_problem", problem),))
    (spec, members), = select_dsl_candidates((op,), path, 4, 8)
    assert members == (op.op_id,)
    assert spec.kernel_id == ATTN_RES + ".inplace" and spec.execution_mode == "vector_only"
    assert (spec.aic_budget, spec.aiv_budget) == (4, 8)
    assert 0 < spec.ub_peak_bytes <= path.usable_bytes("ub") and spec.workspace_bytes_total == 0
    assert not select_dsl_candidates((op,), path, 4, 0)
    tiny_ub = replace(path, buffers=tuple((pool, 1 if pool == "ub" else size) for pool, size in path.buffers))
    assert not select_dsl_candidates((op,), tiny_ub, 4, 8)


def test_verify_plan_blocks_nonlinear_norm_across_sum_collective_and_cross_cube_swap():
    from mindie_sim.prediction.spine import PredictionSpine
    from mindie_sim.rewrite import RewritePlan
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.schema.operators import OpSpec

    def _op(op_id, family, shape, **kw):
        return OpSpec(op_id=op_id, family=family, phase="decode", stage=0, layer=0, requests=(),
                      semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape, **kw)

    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=2, dp=1, layers=1)
    ar = _op("s0.l0.allreduce", "allreduce", (16, 4096), dtype="bf16", link_bytes=131072.0,
             metadata=(("group_axis", "tp"),))
    norm = _op("s0.l0.rmsnorm", "rmsnorm", (16, 4096), deps=("s0.l0.allreduce",), dtype="bf16",
               vector_ops=262144.0, mte2_bytes=131072.0, mte3_bytes=131072.0, hbm_bytes=262144.0)
    mm1 = _op("s0.l0.mm1", "gemm", (16, 4096, 4096), deps=("s0.l0.rmsnorm",), dtype="bf16",
              flops=float(2 * 16 * 4096 * 4096), matrix_flops=(("bf16", float(2 * 16 * 4096 * 4096)),),
              mte2_bytes=33554432.0, mte3_bytes=131072.0, hbm_bytes=33685504.0)
    mm2 = _op("s0.l0.mm2", "gemm", (16, 4096, 4096), deps=("s0.l0.mm1",), dtype="bf16",
              flops=float(2 * 16 * 4096 * 4096), matrix_flops=(("bf16", float(2 * 16 * 4096 * 4096)),),
              mte2_bytes=33554432.0, mte3_bytes=131072.0, hbm_bytes=33685504.0)

    class _GuardSpine(PredictionSpine):
        def lower(self, parsed, request, **kwargs):
            prepared = super().lower(parsed, request, **kwargs)
            ranks = tuple(replace(rank, ops=(ar, norm, mm1, mm2)) for rank in prepared.graph.executable)
            return replace(prepared, graph=replace(prepared.graph, executable=ranks, streams=()))

    scenario = resolve_scenario(case, StepRequest("decode", tokens=16, context=64), spine=_GuardSpine())
    bad_comm_refs = (OpRef("atom", "s0.l0.allreduce"), OpRef("atom", "s0.l0.rmsnorm"))
    bad_comm_plan = RewritePlan(
        base_graph_sha256=scenario.base_graph_sha256,
        framing_sha256=scenario.framing_sha256,
        edits=(Edit("bad_ar_norm", operation=FuseRegion(
            members=bad_comm_refs, preserve_ports=(),
            kernel_boundaries=(KernelBoundary("IllegalArNorm", bad_comm_refs),),
            fusion_tier="K2_epilogue",
        )),),
    )
    receipt_comm = verify_plan(scenario, apply_plan(bad_comm_plan, scenario), scope="archetype")
    assert not receipt_comm.passed
    assert any(
        g.reason == "NON_COMMUTATIVE_REWRITE" and "collective_sum_fusion" in g.detail
        for g in receipt_comm.gates if not g.passed
    )

    # Swapping causal order across Cube boundaries fails both in apply_plan and verify_plan.
    swap_refs = (OpRef("atom", "s0.l0.mm1"), OpRef("atom", "s0.l0.mm2"))
    swap_plan = RewritePlan(
        base_graph_sha256=scenario.base_graph_sha256,
        framing_sha256=scenario.framing_sha256,
        edits=(Edit("swap_cube", operation=FuseRegion(
            members=swap_refs, preserve_ports=(),
            kernel_boundaries=(
                KernelBoundary("SecondFirst", (OpRef("atom", "s0.l0.mm2"),)),
                KernelBoundary("FirstSecond", (OpRef("atom", "s0.l0.mm1"),)),
            ),
            fusion_tier="K2_epilogue",
        )),),
    )
    with pytest.raises(ModelingError, match="rewrite_kernel_boundary_dependency_order"):
        apply_plan(swap_plan, scenario)

    base_applied = apply_plan(
        RewritePlan(base_graph_sha256=scenario.base_graph_sha256, framing_sha256=scenario.framing_sha256),
        scenario,
    )
    reordered_ranks = tuple(
        replace(
            rank,
            ops=(
                ar,
                norm,
                replace(mm2, deps=("s0.l0.rmsnorm",)),
                replace(mm1, deps=("s0.l0.mm2",)),
            ),
        )
        for rank in base_applied.executable
    )
    receipt_swap = verify_plan(scenario, replace(base_applied, executable=reordered_ranks), scope="archetype")
    assert not receipt_swap.passed
    assert any(
        g.reason == "NON_COMMUTATIVE_REWRITE" and "non_commuting_reorder" in g.detail
        for g in receipt_swap.gates if not g.passed
    )


def test_algebraic_rewrite_laws_enforce_port_quant_contracts_and_activation_predicates(real_graph):
    from mindie_sim.rewrite import discover_algebraic_rewrites_for_ops
    from mindie_sim.schema.operators import OpSpec

    scenario, _ = real_graph
    path = scenario.datapath

    def _op(op_id, family, shape, **kw):
        return OpSpec(op_id=op_id, family=family, phase="decode", stage=0, layer=0, requests=(),
                      semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape, **kw)

    # Law 2: matching PortRef + (a_dtype, group_size) fuses; mismatched quant/group_size rejects.
    src = _op("s0.l0.norm", "rmsnorm", (16, 4096), dtype="bf16", vector_ops=65536.0, hbm_bytes=262144.0)
    q_proj = _op(
        "s0.l0.q_proj", "gemm", (16, 2048, 4096), deps=("s0.l0.norm",), dtype="bf16",
        flops=float(2 * 16 * 2048 * 4096), hbm_bytes=16777216.0,
        metadata=(
            ("typed_inputs", ({"producer": "s0.l0.norm", "port": "out", "version": 0},)),
            ("activation_format", "fp8_e4m3"), ("activation_group_size", 32),
        ),
    )
    k_proj = replace(q_proj, op_id="s0.l0.k_proj", local_shape=(16, 1024, 4096),
                     flops=float(2 * 16 * 1024 * 4096), hbm_bytes=8388608.0)
    v_mismatch = replace(
        q_proj, op_id="s0.l0.v_proj", local_shape=(16, 1024, 4096),
        flops=float(2 * 16 * 1024 * 4096), hbm_bytes=8388608.0,
        metadata=(
            ("typed_inputs", ({"producer": "s0.l0.norm", "port": "out", "version": 0},)),
            ("activation_format", "bf16"), ("activation_group_size", 128),
        ),
    )
    law2 = discover_algebraic_rewrites_for_ops((src, q_proj, k_proj, v_mismatch), datapath=path)
    block_cands = [c for c in law2.candidates if c.law == "law2_block_matmul"]
    assert len(block_cands) == 1
    assert block_cands[0].member_ids == ("s0.l0.q_proj", "s0.l0.k_proj")
    assert any(reason == "quant_or_sharding_contract_mismatch" for _, reason in law2.rejected_pairs)

    # Law 4 / ABI variant extension: MoE dispatch -> grouped_gemm -> situ -> quant -> grouped_gemm -> combine.
    disp = _op("s0.l0.dispatch", "dispatch", (16, 2048), dtype="bf16", link_bytes=65536.0)
    gmm1 = _op("s0.l0.gmm1", "grouped_gemm", (16, 4096, 2048), deps=("s0.l0.dispatch",), dtype="bf16",
               flops=float(2 * 16 * 4096 * 2048), hbm_bytes=16777216.0)
    situ = _op("s0.l0.act", "activation", (16, 2048), deps=("s0.l0.gmm1",), dtype="bf16",
               vector_ops=float(16 * 2048 * 6), hbm_bytes=262144.0,
               metadata=(("activation", "situ"), ("beta", 1.5)))
    quant = _op("s0.l0.act_quant", "quantize", (16, 2048), deps=("s0.l0.act",), dtype="fp8",
                vector_ops=float(16 * 2048 * 3), hbm_bytes=196608.0)
    gmm2 = _op("s0.l0.gmm2", "grouped_gemm", (16, 2048, 2048), deps=("s0.l0.act_quant",), dtype="bf16",
               flops=float(2 * 16 * 2048 * 2048), hbm_bytes=8388608.0)
    comb = _op("s0.l0.combine", "combine", (16, 2048), deps=("s0.l0.gmm2",), dtype="bf16", link_bytes=65536.0)
    moe_res = discover_algebraic_rewrites_for_ops((disp, gmm1, situ, quant, gmm2, comb), datapath=path)
    assert any(t.kind == "ABI_UNSUPPORTED_VARIANT" and t.candidate_abi == "dispatch_ffn_combine_situ_ext"
               for t in moe_res.semantic_audit_triggers)
    assert any(c.abi_id == "dispatch_ffn_combine_situ_ext" and not c.shipping_supported for c in moe_res.candidates)


def test_solve_target_graph_enforces_mode_separation_and_whitebox_l0_pricing():
    from mindie_sim.optimization import TargetOpNode, solve_target_graph
    from mindie_sim.prediction.results import OpOccurrence
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import OpSpec

    case = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1))
    shape = (16, 4096)
    tensor_bytes = float(16 * 4096 * 2)
    op1 = OpSpec(op_id="s0.l0.act", family="activation", phase="decode", stage=0, layer=0, requests=(),
                 semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
                 dtype="bf16", vector_ops=float(16 * 4096 * 6),
                 mte2_bytes=tensor_bytes, mte3_bytes=tensor_bytes, hbm_bytes=2.0 * tensor_bytes,
                 metadata=(("activation", "situ"), ("beta", 1.5)))
    op2 = OpSpec(op_id="s0.l0.quant", family="quantize", phase="decode", stage=0, layer=0, requests=(),
                 deps=("s0.l0.act",), semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
                 dtype="fp8", vector_ops=float(16 * 4096 * 3),
                 mte2_bytes=tensor_bytes, mte3_bytes=tensor_bytes * 0.5, hbm_bytes=1.5 * tensor_bytes)
    occ1 = OpOccurrence(replica=0, stage=0, rank=0, op_id="s0.l0.act")
    occ2 = OpOccurrence(replica=0, stage=0, rank=0, op_id="s0.l0.quant")
    n1 = TargetOpNode(
        0, occ1, "s0.l0.act", "arch0", 1, "activation", "situ", "main", 0, "AIV", shape, "bf16", (),
        0.0, 40.0, 40.0, 40.0, False, "", 0.0, op1.vector_ops, tensor_bytes, tensor_bytes, 2.0 * tensor_bytes,
        tensor_bytes, tensor_bytes, 0.0, 0.0, "memory_bound", "HBM", 1.0, 39.0, 40.0, 0.025,
        ("act",), ("act",), "SiTU", "situ", "CANN", "", False, "", "act", spec=op1,
    )
    n2 = TargetOpNode(
        1, occ2, "s0.l0.quant", "arch0", 1, "quantize", "dynamic_quant", "main", 0, "AIV", shape, "fp8", (occ1,),
        40.0, 80.0, 40.0, 40.0, False, "", 0.0, op2.vector_ops, tensor_bytes, tensor_bytes * 0.5, 1.5 * tensor_bytes,
        tensor_bytes, tensor_bytes * 0.5, 0.0, 0.0, "memory_bound", "HBM", 1.0, 39.0, 40.0, 0.025,
        ("quant",), ("quant",), "Quant", "quant", "CANN", "", False, "", "quant", spec=op2,
    )
    die = case.machine.dies[0]
    peaks = {0: {"hbm_peak_bytes_per_s": dict(die.rates)["hbm"]}}
    dems = ((0, 500, 500, 500, 0), (0, 500, 500, 250, 0))

    sim_res = solve_target_graph((n1, n2), dems, case, peaks, allow_unbound_formula=False)
    assert sim_res.has_unbound_formula_ops is False
    assert sim_res.binding_state == "bound_implementation"
    assert len(sim_res.nodes) == 2

    mod_res = solve_target_graph((n1, n2), dems, case, peaks, allow_unbound_formula=True)
    assert mod_res.has_unbound_formula_ops is True
    assert mod_res.binding_state == "unbound_formula_projection"
    assert len(mod_res.nodes) == 1
    fused = mod_res.nodes[0]
    assert fused.dur_us < 80.0 and fused.dur_us >= fused.l_star_lean_us
    assert mod_res.synthesis_specs[0].sram_verified is True
    assert any("beta=1.5" in eq for eq in mod_res.synthesis_specs[0].symbolic_equations)


def _mk_target_node(
    ordinal: int,
    op_spec,
    *,
    archetype_id: str = "arch0",
    repeat_count: int = 1,
    task_type: str = "AIV",
    predecessors: tuple = (),
    start_us: float = 0.0,
    dur_us: float = 20.0,
    l_star_us: float = 2.0,
    bound: str = "memory_bound",
    kernel: str = "",
):
    from mindie_sim.optimization import TargetOpNode
    from mindie_sim.prediction.results import OpOccurrence

    occ = OpOccurrence(replica=0, stage=op_spec.stage, rank=0, op_id=op_spec.op_id)
    mte2 = float(op_spec.mte2_bytes)
    mte3 = float(op_spec.mte3_bytes)
    hbm = float(op_spec.hbm_bytes)
    return TargetOpNode(
        ordinal, occ, op_spec.op_id, archetype_id, repeat_count, op_spec.family, op_spec.recipe or op_spec.family,
        op_spec.stream or "main", op_spec.stage, task_type, op_spec.local_shape, op_spec.dtype or "bf16",
        predecessors, start_us, start_us + dur_us, dur_us, dur_us, False, "",
        float(op_spec.flops), float(op_spec.vector_ops), mte2, mte3, hbm, mte2, mte3, 0.0,
        float(op_spec.link_bytes), bound, bound, l_star_us, max(0.0, dur_us - l_star_us),
        (dur_us / l_star_us) if l_star_us > 0.0 else None,
        min(1.0, l_star_us / dur_us) if dur_us > 0.0 and l_star_us > 0.0 else None,
        (op_spec.op_id,), (op_spec.op_id,), op_spec.family, kernel or op_spec.family, "CANN",
        "", False, "", op_spec.op_id, spec=op_spec,
    )


def test_strict_simulation_mode_never_calls_formula_evaluator_and_prices_bound_law2(monkeypatch):
    import mindie_sim.optimization.kernel_synthesis as ks
    from mindie_sim.optimization import solve_target_graph
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.rewrite import Edit, FuseRegion, KernelBoundary, OpRef, RewritePlan
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.schema.operators import OpSpec

    def _forbid_formula(*args, **kwargs):
        raise AssertionError("evaluate_formula_kernel must not be called in strict Simulation Mode")

    monkeypatch.setattr(ks, "evaluate_formula_kernel", _forbid_formula)

    case = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a3"], tp=1, dp=1, layers=1))
    die = case.machine.dies[0]
    peaks = {0: {"hbm_peak_bytes_per_s": dict(die.rates)["hbm"]}}

    # (A) Positive-duration AIV op keeps its bound duration and never calls formula projection.
    shape_v = (16, 4096)
    tb_v = float(16 * 4096 * 2)
    norm_op = OpSpec(
        op_id="s0.l0.norm", family="rmsnorm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=shape_v, local_shape=shape_v, padded_shape=shape_v, launch_shape=shape_v,
        dtype="bf16", vector_ops=float(16 * 4096 * 4),
        mte2_bytes=tb_v, mte3_bytes=tb_v, hbm_bytes=2.0 * tb_v,
    )
    n_norm = _mk_target_node(0, norm_op, dur_us=25.0, l_star_us=2.0)
    res_aiv = solve_target_graph((n_norm,), ((0, 2000, 1000, 1000, 0),), case, peaks, allow_unbound_formula=False)
    assert res_aiv.has_unbound_formula_ops is False
    assert res_aiv.binding_state == "bound_implementation"
    assert res_aiv.nodes[0].dur_us == pytest.approx(25.0)
    assert res_aiv.verification_passed is True

    # (B) Law-2 sibling GEMMs sharing PortRef price via bound RecipeBook with exact directional traffic.
    m_dim, n_dim, k_dim = 1, 4096, 4096
    act_b = float(m_dim * k_dim * 2)
    wt_b = float(n_dim * k_dim * 2)
    out_b = float(m_dim * n_dim * 2)
    mte2 = act_b + wt_b
    hbm = mte2 + out_b
    flops = float(2 * m_dim * n_dim * k_dim)
    g_meta = (
        ("typed_inputs", ({"producer": "s0.l0.norm", "port": "out", "version": 0},)),
        ("activation_format", "bf16"), ("weight_format", "bf16"),
        ("output_format", "bf16"), ("compute_format", "bf16"),
    )
    g1 = OpSpec(
        op_id="s0.l0.q_proj", family="gemm", recipe="gemm", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.l0.norm",), semantic_shape=(m_dim, n_dim, k_dim), local_shape=(m_dim, n_dim, k_dim),
        padded_shape=(m_dim, n_dim, k_dim), launch_shape=(m_dim, n_dim, k_dim),
        dtype="bf16", flops=flops, mte2_bytes=mte2, mte3_bytes=out_b, hbm_bytes=hbm, metadata=g_meta,
    )
    g2 = replace(g1, op_id="s0.l0.k_proj")
    n_src = _mk_target_node(0, replace(norm_op, local_shape=(m_dim, k_dim), semantic_shape=(m_dim, k_dim),
                                       padded_shape=(m_dim, k_dim), launch_shape=(m_dim, k_dim),
                                       mte2_bytes=act_b, mte3_bytes=act_b, hbm_bytes=2.0 * act_b),
                            dur_us=10.0, l_star_us=1.0)
    n_g1 = _mk_target_node(1, g1, task_type="AIC", predecessors=(n_src.occurrence,), start_us=10.0, dur_us=80.0, l_star_us=25.0)
    n_g2 = _mk_target_node(2, g2, task_type="AIC", predecessors=(n_src.occurrence,), start_us=90.0, dur_us=80.0, l_star_us=25.0)
    dems = ((0, 500, 500, 500, 0), (1000, 0, 25000, 100, 0), (1000, 0, 25000, 100, 0))
    res_l2 = solve_target_graph((n_src, n_g1, n_g2), dems, case, peaks, allow_unbound_formula=False)
    assert res_l2.has_unbound_formula_ops is False
    assert res_l2.binding_state == "bound_implementation"
    assert res_l2.verification_passed is True
    fused_l2 = next(n for n in res_l2.nodes if len(n.absorbed_source_ops) == 2)
    assert fused_l2.binding_state in {"bound_implementation", "bound_measured"}
    assert fused_l2.mte2_bytes == pytest.approx(2.0 * mte2 - act_b)
    assert fused_l2.mte3_bytes == pytest.approx(2.0 * out_b)
    assert fused_l2.hbm_total_bytes == pytest.approx(2.0 * hbm - act_b)
    assert fused_l2.synthesis_spec is not None and fused_l2.synthesis_spec.elided_hbm_bytes == int(act_b)

    # (C) Explicit caller unbound FuseRegion fails fast under allow_unbound_formula=False.
    quant_op = OpSpec(
        op_id="s0.l0.quant", family="quantize", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.l0.norm",), semantic_shape=shape_v, local_shape=shape_v, padded_shape=shape_v, launch_shape=shape_v,
        dtype="fp8", vector_ops=float(16 * 4096 * 3),
        mte2_bytes=tb_v, mte3_bytes=tb_v * 0.5, hbm_bytes=1.5 * tb_v,
    )
    n_q = _mk_target_node(1, quant_op, predecessors=(n_norm.occurrence,), start_us=25.0, dur_us=25.0, l_star_us=2.0)
    refs = (OpRef("atom", "s0.l0.norm"), OpRef("atom", "s0.l0.quant"))
    unbound_plan = RewritePlan(edits=(Edit("u_fuse", operation=FuseRegion(
        members=refs, preserve_ports=(),
        kernel_boundaries=(KernelBoundary("UnboundCustomFusion", refs),),
        fusion_tier="K3_dsl_pipeline",
    )),))
    with pytest.raises(ModelingError, match="rewrite_fusion_recipe_unbound"):
        solve_target_graph(
            (n_norm, n_q), ((0, 2000, 1000, 1000, 0), (0, 1500, 1000, 500, 0)),
            case, peaks, rewrite_plan=unbound_plan, allow_unbound_formula=False,
        )


def test_solve_target_graph_executes_all_supported_opcodes_and_rejects_unsupported():
    from mindie_sim.optimization import solve_target_graph
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.rewrite import CollectiveTransform, Edit, OpRef, PipelineTiles, RewritePlan, SelectTemplate
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.schema.operators import OpSpec

    case = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=2, dp=1, layers=1))
    die = case.machine.dies[0]
    peaks = {0: {"hbm_peak_bytes_per_s": dict(die.rates)["hbm"]}}
    route = case.machine.links[0].name

    m_dim, n_dim, k_dim = 16, 2048, 2048
    wt_b = float(n_dim * k_dim * 2)
    gemm_op = OpSpec(
        op_id="s0.l0.gemm", family="gemm", recipe="gemm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(m_dim, n_dim, k_dim), local_shape=(m_dim, n_dim, k_dim),
        padded_shape=(m_dim, n_dim, k_dim), launch_shape=(m_dim, n_dim, k_dim),
        dtype="bf16", flops=float(2 * m_dim * n_dim * k_dim),
        mte2_bytes=float(m_dim * k_dim * 2) + wt_b,
        mte3_bytes=float(m_dim * n_dim * 2),
        hbm_bytes=float(m_dim * k_dim * 2) + wt_b + float(m_dim * n_dim * 2),
        metadata=(("packed_weight_bytes", wt_b),),
    )
    comm_op = OpSpec(
        op_id="s0.l0.ar", family="allreduce", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.l0.gemm",), semantic_shape=(m_dim, n_dim), local_shape=(m_dim, n_dim),
        padded_shape=(m_dim, n_dim), launch_shape=(m_dim, n_dim),
        dtype="bf16", link_bytes=float(m_dim * n_dim * 2), link_scope="tp",
        metadata=(("group_axis", "tp"),),
    )
    n_gemm = _mk_target_node(0, gemm_op, task_type="AIC", dur_us=999.0, l_star_us=5.0)
    n_comm = _mk_target_node(1, comm_op, task_type="HCCL", predecessors=(n_gemm.occurrence,), start_us=999.0, dur_us=500.0, l_star_us=1.0)
    dems = ((5000, 0, 5000, 500, 0), (0, 0, 0, 0, 1000))

    # SelectTemplate + CollectiveTransform both execute via canonical apply_plan + verify_plan.
    plan_ok = RewritePlan(edits=(
        Edit("sel_gemm", operation=SelectTemplate(OpRef("atom", "s0.l0.gemm"), "K1", "balanced_grouped_gate_up")),
        Edit("col_os", operation=CollectiveTransform(OpRef("atom", "s0.l0.ar"), "oneshot", route)),
    ))
    solved = solve_target_graph((n_gemm, n_comm), dems, case, peaks, rewrite_plan=plan_ok, allow_unbound_formula=False)
    assert solved.verification_passed is True
    assert len(solved.applied_plan.edits) >= 2
    g_out = next(n for n in solved.nodes if n.op_id == "s0.l0.gemm")
    c_out = next(n for n in solved.nodes if n.op_id == "s0.l0.ar")
    assert g_out.dur_us < 999.0 and g_out.recipe == "balanced_grouped_gate_up"
    assert c_out.dur_us < 500.0 and "oneshot" in c_out.ascend_kernel

    # Unknown template ID MUST raise ModelingError (never silently ignored).
    plan_bad_tpl = RewritePlan(edits=(
        Edit("bad_tpl", operation=SelectTemplate(OpRef("atom", "s0.l0.gemm"), "K1", "nonexistent_recipe_xyz")),
    ))
    with pytest.raises(ModelingError, match="rewrite_template_recipe_unbound"):
        solve_target_graph((n_gemm, n_comm), dems, case, peaks, rewrite_plan=plan_bad_tpl, allow_unbound_formula=False)

    # Unknown topology route MUST raise ModelingError.
    plan_bad_route = RewritePlan(edits=(
        Edit("bad_rt", operation=CollectiveTransform(OpRef("atom", "s0.l0.ar"), "oneshot", "nonexistent_link")),
    ))
    with pytest.raises(ModelingError, match="rewrite_collective_route_unknown"):
        solve_target_graph((n_gemm, n_comm), dems, case, peaks, rewrite_plan=plan_bad_route, allow_unbound_formula=False)

    # Unsupported opcode (PipelineTiles) MUST fail explicitly instead of silently returning verification_passed=True.
    plan_unsupported = RewritePlan(edits=(
        Edit("pipe_tile", operation=PipelineTiles(OpRef("atom", "s0.l0.gemm"))),
    ))
    with pytest.raises(ModelingError, match="REWRITE_PLAN_UNSUPPORTED_OPCODE"):
        solve_target_graph((n_gemm, n_comm), dems, case, peaks, rewrite_plan=plan_unsupported, allow_unbound_formula=False)


def test_oneshot_collective_pricing_respects_switched_egress_wire_demand():
    from mindie_sim.optimization import solve_target_graph
    from mindie_sim.optimization.kernel_synthesis import _TargetPricingContext, _price_collective_on_topology
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import OpSpec

    case = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a3"], tp=8, dp=1, layers=1))
    die = case.machine.dies[0]
    peaks = {0: {"hbm_peak_bytes_per_s": dict(die.rates)["hbm"]}}
    ctx = _TargetPricingContext(case)

    # Large 8 MiB AllReduce on TP=8 switched fabric: ring wire = 1.75*M (14 MiB), oneshot egress = 7*M (56 MiB).
    msg_large = float(8 * 1024 * 1024)
    ring_link_large = msg_large * (2.0 * 7.0 / 8.0)
    ar_large = OpSpec(
        op_id="s0.l0.ar_large", family="allreduce", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(1024, 4096), local_shape=(1024, 4096), padded_shape=(1024, 4096), launch_shape=(1024, 4096),
        dtype="bf16", link_bytes=ring_link_large, link_scope="tp", metadata=(("group_axis", "tp"),),
    )
    ring_us = _price_collective_on_topology(ar_large, ctx, case, mode="ring")
    oneshot_us = _price_collective_on_topology(ar_large, ctx, case, mode="oneshot")
    assert ring_us is not None and oneshot_us is not None
    assert oneshot_us > ring_us * 2.0

    # Target solver MUST NOT falsely credit OneShot gain when OneShot is slower than ring baseline.
    n_large = _mk_target_node(0, ar_large, task_type="HCCL", dur_us=ring_us, l_star_us=ring_us * 0.8, bound="link_bound")
    res_large = solve_target_graph(
        (n_large,), ((0, 0, 0, 0, int(ring_us * 800.0)),), case, peaks, allow_unbound_formula=False,
    )
    assert res_large.nodes[0].dur_us == pytest.approx(ring_us)
    assert "One-Shot" not in res_large.nodes[0].ascend_kernel

    from mindie_sim.rewrite import CollectiveTransform, Edit, OpRef, RewritePlan
    msg_small = float(4 * 1024)
    ar_small = replace(ar_large, op_id="s0.l0.ar_small", local_shape=(1, 2048), link_bytes=msg_small * (2.0 * 7.0 / 8.0))
    ring_small_us = _price_collective_on_topology(ar_small, ctx, case, mode="ring")
    oneshot_small_us = _price_collective_on_topology(ar_small, ctx, case, mode="oneshot")
    assert ring_small_us is not None and oneshot_small_us is not None and oneshot_small_us < ring_small_us
    n_small = _mk_target_node(0, ar_small, task_type="HCCL", dur_us=ring_small_us, l_star_us=0.5, bound="link_bound")
    res_auto = solve_target_graph((n_small,), ((0, 0, 0, 0, 500),), case, peaks, allow_unbound_formula=False)
    assert res_auto.nodes[0].dur_us == pytest.approx(ring_small_us) and "One-Shot" not in res_auto.nodes[0].ascend_kernel
    case_mesh = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=8, dp=1, layers=1))
    mesh_peaks = {0: {"hbm_peak_bytes_per_s": dict(case_mesh.machine.dies[0].rates)["hbm"]}}
    ring_m = _price_collective_on_topology(ar_small, _TargetPricingContext(case_mesh), case_mesh, mode="ring")
    assert ring_m is not None
    n_m = _mk_target_node(0, ar_small, task_type="HCCL", dur_us=ring_m, l_star_us=0.5, bound="link_bound")
    plan_small = RewritePlan(edits=(Edit("os_small", operation=CollectiveTransform(OpRef("atom", "s0.l0.ar_small"), "oneshot", case_mesh.machine.links[0].name)),))
    res_small = solve_target_graph((n_m,), ((0, 0, 0, 0, 500),), case_mesh, mesh_peaks, rewrite_plan=plan_small, allow_unbound_formula=False)
    assert res_small.nodes[0].dur_us < ring_m and "oneshot" in res_small.nodes[0].ascend_kernel.lower()

    # AllGather on TP=8 splits M/(G-1) per peer so oneshot stays well-defined and finite.
    ag_op = replace(ar_small, op_id="s0.l0.ag", family="allgather", link_bytes=msg_large * (7.0 / 8.0))
    ag_os_us = _price_collective_on_topology(ag_op, ctx, case, mode="oneshot")
    assert ag_os_us is not None and ag_os_us > 0.0


def test_solve_target_graph_rejects_ub_infeasible_aiv_projection():
    from mindie_sim.optimization import solve_target_graph
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import OpSpec

    case = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1))
    die = case.machine.dies[0]
    peaks = {0: {"hbm_peak_bytes_per_s": dict(die.rates)["hbm"]}}
    # 32,768-wide BF16 row reduction exceeds 237,568 B usable A5 UB -> sram_verified=False -> no overwrite.
    shape = (4, 32768)
    tb = float(4 * 32768 * 2)
    wide_norm = OpSpec(
        op_id="s0.l0.wide_norm", family="rmsnorm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="bf16", vector_ops=float(4 * 32768 * 4),
        mte2_bytes=tb, mte3_bytes=tb, hbm_bytes=2.0 * tb,
    )
    n_wide = _mk_target_node(0, wide_norm, dur_us=95.0, l_star_us=5.0)
    res = solve_target_graph((n_wide,), ((0, 5000, 2000, 2000, 0),), case, peaks, allow_unbound_formula=True)
    assert res.nodes[0].dur_us == pytest.approx(95.0)
    assert res.has_unbound_formula_ops is False


def test_attribute_node_overlap_conserves_step_time_across_archetypes():
    from mindie_sim.optimization import attribute_node_overlap
    from mindie_sim.schema.operators import OpSpec

    op_a = OpSpec(op_id="s0.l0.compute", family="gemm", phase="decode", stage=0, layer=0, requests=(),
                  semantic_shape=(16, 1024, 1024), local_shape=(16, 1024, 1024),
                  padded_shape=(16, 1024, 1024), launch_shape=(16, 1024, 1024), dtype="bf16", flops=1e9)
    op_b = replace(op_a, op_id="s0.l1.prefetch", family="copy", layer=1, flops=0.0)
    op_c = replace(op_a, op_id="s0.l1.tail", family="rmsnorm", layer=1, flops=0.0)

    # Full containment across different archetypes: [0, 20] + [5, 15] -> total exposed == 20.0 (not 30.0).
    n0 = _mk_target_node(0, op_a, archetype_id="arch0", start_us=0.0, dur_us=20.0)
    n1 = _mk_target_node(1, op_b, archetype_id="arch1", start_us=5.0, dur_us=10.0)
    attributed, spans = attribute_node_overlap((n0, n1))
    assert attributed[0].exposed_dur_us == pytest.approx(20.0)
    assert attributed[1].exposed_dur_us == pytest.approx(0.0)
    assert spans["arch0"] == pytest.approx(20.0)
    assert spans["arch1"] == pytest.approx(0.0)
    assert sum(spans.values()) == pytest.approx(20.0)

    # Partial cross-archetype overlap (primary first): [0, 20] + [10, 25] -> exposed 20.0 + 5.0 == 25.0 == T_step.
    n2 = _mk_target_node(1, op_c, archetype_id="arch1", start_us=10.0, dur_us=15.0)
    attr2, spans2 = attribute_node_overlap((n0, n2))
    assert attr2[0].exposed_dur_us == pytest.approx(20.0)
    assert attr2[1].exposed_dur_us == pytest.approx(5.0)
    assert sum(spans2.values()) == pytest.approx(25.0)

    # Partial tail-covered overlap (tail node claims first): [0, 20] + [10, 25] -> exposed 10.0 + 15.0 == 25.0.
    n0_eq = _mk_target_node(0, replace(op_a, flops=0.0), archetype_id="arch0", start_us=0.0, dur_us=20.0)
    attr3, spans3 = attribute_node_overlap((n0_eq, n2))
    assert attr3[0].exposed_dur_us == pytest.approx(10.0)
    assert attr3[1].exposed_dur_us == pytest.approx(15.0)
    assert sum(spans3.values()) == pytest.approx(25.0)


def test_mla_prolog_matching_uses_typed_metadata_without_op_name_heuristics(real_graph):
    from mindie_sim.rewrite import discover_algebraic_rewrites_for_ops
    from mindie_sim.schema.operators import OpSpec

    scenario, _ = real_graph
    path = scenario.datapath

    def _op(op_id, family, shape, **kw):
        return OpSpec(op_id=op_id, family=family, phase="decode", stage=0, layer=0, requests=(),
                      semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape, **kw)

    # Obfuscated op_ids with ZERO name hints ("q_a"/"kv_a"/"q_b"/"absorb"/"unpack") — matched purely via typed metadata.
    x_norm = _op("s0.l0.step0", "rmsnorm", (16, 7168), dtype="bf16", vector_ops=float(16 * 7168 * 4), hbm_bytes=458752.0)
    step1 = _op("s0.l0.step1", "gemm", (16, 1536, 7168), deps=("s0.l0.step0",), dtype="bf16",
                flops=float(2 * 16 * 1536 * 7168), hbm_bytes=22020096.0,
                metadata=(("quant_category", "mla.q_a"),))
    step2 = _op("s0.l0.step2", "gemm", (16, 576, 7168), deps=("s0.l0.step0",), dtype="bf16",
                flops=float(2 * 16 * 576 * 7168), hbm_bytes=8257536.0,
                metadata=(("quant_category", "mla.kv_a"),))
    step3 = _op("s0.l0.step3", "rmsnorm", (16, 1536), deps=("s0.l0.step1",), dtype="bf16",
                vector_ops=float(16 * 1536 * 4), hbm_bytes=98304.0)
    step4 = _op("s0.l0.step4", "gemm", (16, 24576, 1536), deps=("s0.l0.step3",), dtype="bf16",
                flops=float(2 * 16 * 24576 * 1536), hbm_bytes=75497472.0,
                metadata=(("quant_category", "mla.q_b"),))
    step5 = _op("s0.l0.step5", "rotary", (16, 64), deps=("s0.l0.step4",), dtype="bf16",
                vector_ops=float(16 * 64 * 4), hbm_bytes=8192.0)
    step6 = _op("s0.l0.step6", "gemm", (16, 512, 128), deps=("s0.l0.step5",), dtype="bf16",
                flops=float(2 * 16 * 512 * 128), hbm_bytes=131072.0,
                metadata=(("role", "key_absorption"),))
    draft_step = _op("s0.l0.step7", "gemm", (16, 1536, 7168), deps=("s0.l0.step0",), dtype="bf16",
                     flops=float(2 * 16 * 1536 * 7168), hbm_bytes=22020096.0,
                     metadata=(("quant_category", "mla.q_a"), ("model_scope", "draft")))

    res = discover_algebraic_rewrites_for_ops(
        (x_norm, step1, step2, step3, step4, step5, step6, draft_step), datapath=path,
    )
    mla = [c for c in res.candidates if c.abi_id == "aclnnMlaPrologV3"]
    assert len(mla) == 1
    assert set(mla[0].member_ids) == {"s0.l0.step1", "s0.l0.step2", "s0.l0.step3", "s0.l0.step4", "s0.l0.step5", "s0.l0.step6"}
    assert "s0.l0.step7" not in mla[0].member_ids


def test_target_graph_multi_archetype_model_scope_end_to_end():
    from mindie_sim.export import build_export_bundle, compose_export_case

    raw_after, auto_req = compose_export_case(
        "qwen3-32b", "a5-pod", feature_track="expected_ideal",
        dp=1, tp=2, ep=1, prompt_tokens=512, output_tokens=32,
    )
    step_req = StepRequest(
        phase="decode", tokens=auto_req.tokens, context=auto_req.context,
        sequences=auto_req.sequences, scope="model",
    )
    bundle = build_export_bundle(raw_after, request=step_req)
    assert bundle.rewrite_verification_passed is True
    assert bundle.step_span_us > 0.0


def test_verify_plan_blocks_typed_and_named_producer_non_commutative_fusion():
    from mindie_sim.prediction.spine import PredictionSpine
    from mindie_sim.rewrite import RewritePlan
    from mindie_sim.schema.operators import OpSpec

    def _op(op_id, family, shape, **kw):
        return OpSpec(op_id=op_id, family=family, phase="decode", stage=0, layer=0, requests=(),
                      semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape, **kw)

    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=2, dp=1, layers=1)
    ar = _op("s0.l0.allreduce", "allreduce", (16, 4096), dtype="bf16", link_bytes=131072.0,
             metadata=(("group_axis", "tp"),))
    norm_typed = _op(
        "s0.l0.rmsnorm", "rmsnorm", (16, 4096), deps=(), dtype="bf16",
        vector_ops=262144.0, mte2_bytes=131072.0, mte3_bytes=131072.0, hbm_bytes=262144.0,
        metadata=(("typed_inputs", ({"producer": "s0.l0.allreduce", "port": "out", "version": 0},)),),
    )
    norm_named = _op(
        "s0.l0.rmsnorm", "rmsnorm", (16, 4096), deps=(), dtype="bf16",
        vector_ops=262144.0, mte2_bytes=131072.0, mte3_bytes=131072.0, hbm_bytes=262144.0,
        metadata=(("activation_producer", "s0.l0.allreduce"),),
    )

    for variant_norm in (norm_typed, norm_named):
        class _TypedEdgeSpine(PredictionSpine):
            def __init__(self, ops):
                super().__init__()
                self._ops = ops

            def lower(self, parsed, request, **kwargs):
                prepared = super().lower(parsed, request, **kwargs)
                ranks = tuple(replace(rank, ops=self._ops) for rank in prepared.graph.executable)
                return replace(prepared, graph=replace(prepared.graph, executable=ranks, streams=()))

        scenario = resolve_scenario(case, StepRequest("decode", tokens=16, context=64), spine=_TypedEdgeSpine((ar, variant_norm)))
        refs = (OpRef("atom", "s0.l0.allreduce"), OpRef("atom", "s0.l0.rmsnorm"))
        bad_plan = RewritePlan(
            base_graph_sha256=scenario.base_graph_sha256,
            framing_sha256=scenario.framing_sha256,
            edits=(Edit("bad_typed_ar_norm", operation=FuseRegion(
                members=refs, preserve_ports=(),
                kernel_boundaries=(KernelBoundary("IllegalTypedArNorm", refs),),
                fusion_tier="K2_epilogue",
            )),),
        )
        receipt = verify_plan(scenario, apply_plan(bad_plan, scenario), scope="archetype")
        assert not receipt.passed
        assert any(
            g.reason == "NON_COMMUTATIVE_REWRITE" and "collective_sum_fusion" in g.detail
            for g in receipt.gates if not g.passed
        )


def test_strict_mode_rejects_preexisting_unbound_nodes_multi_boundary_and_stale_digests():
    from mindie_sim.optimization import solve_target_graph
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.rewrite import Edit, FuseRegion, KernelBoundary, OpRef, RewritePlan
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.schema.operators import OpSpec

    case = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a3"], tp=1, dp=1, layers=1))
    die = case.machine.dies[0]
    peaks = {0: {"hbm_peak_bytes_per_s": dict(die.rates)["hbm"]}}

    # (A) Pre-existing unbound_formula_projection node MUST fail fast in strict Simulation Mode.
    shape_v = (16, 4096)
    tb_v = float(16 * 4096 * 2)
    norm_op = OpSpec(
        op_id="s0.l0.norm", family="rmsnorm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=shape_v, local_shape=shape_v, padded_shape=shape_v, launch_shape=shape_v,
        dtype="bf16", vector_ops=float(16 * 4096 * 4), mte2_bytes=tb_v, mte3_bytes=tb_v, hbm_bytes=2.0 * tb_v,
    )
    unbound_in = replace(_mk_target_node(0, norm_op, dur_us=10.0, l_star_us=2.0), binding_state="unbound_formula_projection")
    with pytest.raises(ModelingError, match="strict_simulation_unbound_formula_ops:s0.l0.norm"):
        solve_target_graph((unbound_in,), ((0, 1000, 1000, 1000, 0),), case, peaks, allow_unbound_formula=False)

    # (B) 2-boundary FuseRegion with unknown boundary 0 MUST be rejected even if boundary 1 is registered.
    quant_op = OpSpec(
        op_id="s0.l0.quant", family="quantize", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.l0.norm",), semantic_shape=shape_v, local_shape=shape_v, padded_shape=shape_v, launch_shape=shape_v,
        dtype="fp8", vector_ops=float(16 * 4096 * 3), mte2_bytes=tb_v, mte3_bytes=tb_v * 0.5, hbm_bytes=1.5 * tb_v,
    )
    n_norm = _mk_target_node(0, norm_op, dur_us=25.0, l_star_us=2.0)
    n_quant = _mk_target_node(1, quant_op, predecessors=(n_norm.occurrence,), start_us=25.0, dur_us=25.0, l_star_us=2.0)
    two_bound_plan = RewritePlan(edits=(Edit("two_b", operation=FuseRegion(
        members=(OpRef("atom", "s0.l0.norm"), OpRef("atom", "s0.l0.quant")),
        preserve_ports=(),
        kernel_boundaries=(
            KernelBoundary("UnknownBoundaryZero", (OpRef("atom", "s0.l0.norm"),)),
            KernelBoundary("balanced_grouped_gate_up", (OpRef("atom", "s0.l0.quant"),)),
        ),
        fusion_tier="K2_epilogue",
    )),))
    for mode_flag in (False, True):
        with pytest.raises(ModelingError, match="rewrite_fusion_recipe_unbound"):
            solve_target_graph(
                (n_norm, n_quant), ((0, 2000, 1000, 1000, 0), (0, 1500, 1000, 500, 0)),
                case, peaks, rewrite_plan=two_bound_plan, allow_unbound_formula=mode_flag,
            )

    # (C) Auto-generated Law-2 plan persists bound 'gemm' ABI + real digests and replays strictly.
    m_dim, n_dim, k_dim = 1, 4096, 4096
    act_b = float(m_dim * k_dim * 2)
    wt_b = float(n_dim * k_dim * 2)
    out_b = float(m_dim * n_dim * 2)
    mte2 = act_b + wt_b
    hbm = mte2 + out_b
    flops = float(2 * m_dim * n_dim * k_dim)
    g_meta = (
        ("typed_inputs", ({"producer": "s0.l0.norm", "port": "out", "version": 0},)),
        ("activation_format", "bf16"), ("weight_format", "bf16"),
        ("output_format", "bf16"), ("compute_format", "bf16"),
    )
    g1 = OpSpec(
        op_id="s0.l0.q_proj", family="gemm", recipe="gemm", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.l0.norm",), semantic_shape=(m_dim, n_dim, k_dim), local_shape=(m_dim, n_dim, k_dim),
        padded_shape=(m_dim, n_dim, k_dim), launch_shape=(m_dim, n_dim, k_dim),
        dtype="bf16", flops=flops, mte2_bytes=mte2, mte3_bytes=out_b, hbm_bytes=hbm, metadata=g_meta,
    )
    g2 = replace(g1, op_id="s0.l0.k_proj")
    n_src = _mk_target_node(0, replace(norm_op, local_shape=(m_dim, k_dim), semantic_shape=(m_dim, k_dim),
                                       padded_shape=(m_dim, k_dim), launch_shape=(m_dim, k_dim),
                                       mte2_bytes=act_b, mte3_bytes=act_b, hbm_bytes=2.0 * act_b),
                            dur_us=10.0, l_star_us=1.0)
    n_g1 = _mk_target_node(1, g1, task_type="AIC", predecessors=(n_src.occurrence,), start_us=10.0, dur_us=80.0, l_star_us=25.0)
    n_g2 = _mk_target_node(2, g2, task_type="AIC", predecessors=(n_src.occurrence,), start_us=90.0, dur_us=80.0, l_star_us=25.0)
    dems = ((0, 500, 500, 500, 0), (1000, 0, 25000, 100, 0), (1000, 0, 25000, 100, 0))
    auto_res = solve_target_graph((n_src, n_g1, n_g2), dems, case, peaks, allow_unbound_formula=False)
    assert auto_res.applied_plan.edits
    assert auto_res.applied_plan.base_graph_sha256 and auto_res.applied_plan.framing_sha256
    assert auto_res.applied_plan.edits[0].operation.kernel_boundaries[0].abi_id == "gemm"

    replay_res = solve_target_graph(
        (n_src, n_g1, n_g2), dems, case, peaks,
        rewrite_plan=auto_res.applied_plan, allow_unbound_formula=False,
    )
    assert replay_res.has_unbound_formula_ops is False
    assert replay_res.verification_passed is True
    assert any(len(n.absorbed_source_ops) == 2 and n.binding_state in {"bound_implementation", "bound_measured"}
               for n in replay_res.nodes)

    # (D) Stale/forged framing or base_graph digest MUST be rejected (no tautological digest copy).
    tampered_frame = replace(auto_res.applied_plan, framing_sha256="f" * 64)
    with pytest.raises(ModelingError, match="rewrite_digest_mismatch:framing_sha256"):
        solve_target_graph((n_src, n_g1, n_g2), dems, case, peaks, rewrite_plan=tampered_frame, allow_unbound_formula=False)
    tampered_base = replace(auto_res.applied_plan, base_graph_sha256="e" * 64)
    with pytest.raises(ModelingError, match="rewrite_digest_mismatch:base_graph_sha256"):
        solve_target_graph((n_src, n_g1, n_g2), dems, case, peaks, rewrite_plan=tampered_base, allow_unbound_formula=False)


def test_oneshot_collective_rejects_local_work_and_conserves_concurrent_link_contention():
    from mindie_sim.optimization import solve_target_graph
    from mindie_sim.optimization.kernel_synthesis import _TargetPricingContext, _price_collective_on_topology
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.rewrite import CollectiveTransform, Edit, OpRef, RewritePlan
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.schema.operators import OpSpec

    case = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=4, dp=1, layers=1))
    die = case.machine.dies[0]
    peaks = {0: {"hbm_peak_bytes_per_s": dict(die.rates)["hbm"]}}
    route = case.machine.links[0].name
    ctx = _TargetPricingContext(case)

    # (A) Op carrying 100 us of local compute/HBM work + 5 us transfer MUST NOT lose local work under OneShot.
    mixed_op = OpSpec(
        op_id="s0.l0.mixed_comm", family="allreduce", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(16, 4096), local_shape=(16, 4096), padded_shape=(16, 4096), launch_shape=(16, 4096),
        dtype="bf16", vector_ops=float(16 * 4096 * 8), mte2_bytes=131072.0, mte3_bytes=131072.0, hbm_bytes=262144.0,
        link_bytes=8192.0, link_scope="tp", metadata=(("group_axis", "tp"),),
    )
    n_mixed = _mk_target_node(0, mixed_op, task_type="HCCL", dur_us=105.0, l_star_us=5.0)
    res_auto_mixed = solve_target_graph((n_mixed,), ((0, 5000, 1000, 1000, 500),), case, peaks, allow_unbound_formula=False)
    assert res_auto_mixed.nodes[0].dur_us == pytest.approx(105.0)

    bad_col_plan = RewritePlan(edits=(
        Edit("mixed_os", operation=CollectiveTransform(OpRef("atom", "s0.l0.mixed_comm"), "oneshot", route)),
    ))
    with pytest.raises(ModelingError, match="oneshot_collective_requires_transport_only_work"):
        solve_target_graph(
            (n_mixed,), ((0, 5000, 1000, 1000, 500),), case, peaks,
            rewrite_plan=bad_col_plan, allow_unbound_formula=False,
        )

    # (B) Two concurrent V-byte transfers on separate streams sharing physical links MUST contend (2V/R, not V/R).
    msg_v = float(8 * 1024 * 1024)
    ring_link_v = msg_v * (2.0 * 3.0 / 4.0)
    ar_a = OpSpec(
        op_id="s0.l0.ar_a", family="allreduce", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(1024, 4096), local_shape=(1024, 4096), padded_shape=(1024, 4096), launch_shape=(1024, 4096),
        dtype="bf16", stream="comm0", link_bytes=ring_link_v, link_scope="tp", metadata=(("group_axis", "tp"),),
    )
    ar_b = replace(ar_a, op_id="s0.l0.ar_b", stream="comm1")
    single_us = _price_collective_on_topology(ar_a, ctx, case, mode="oneshot")
    assert single_us is not None and single_us > 0.0

    n_a = replace(_mk_target_node(0, ar_a, task_type="HCCL", start_us=0.0, dur_us=single_us, l_star_us=single_us * 0.5), stream="comm0", stream_index=1)
    n_b = replace(_mk_target_node(1, ar_b, task_type="HCCL", start_us=0.0, dur_us=single_us, l_star_us=single_us * 0.5), stream="comm1", stream_index=2)
    conc_plan = RewritePlan(edits=(
        Edit("os_a", operation=CollectiveTransform(OpRef("atom", "s0.l0.ar_a"), "oneshot", route)),
        Edit("os_b", operation=CollectiveTransform(OpRef("atom", "s0.l0.ar_b"), "oneshot", route)),
    ))
    res_conc = solve_target_graph(
        (n_a, n_b), ((0, 0, 0, 0, int(single_us * 500.0)), (0, 0, 0, 0, int(single_us * 500.0))),
        case, peaks, rewrite_plan=conc_plan, allow_unbound_formula=False,
    )
    total_span_us = sum(res_conc.archetype_spans_us.values())
    assert total_span_us > single_us * 1.8
    assert all(n.dur_us > single_us * 1.8 for n in res_conc.nodes)


def test_pairwise_alltoall_prices_routed_volume_above_launch_floor():
    from mindie_sim.optimization.kernel_synthesis import (
        _TargetPricingContext, _build_collective_plan, _price_collective_on_topology, extract_sku_floors,
    )
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import OpSpec

    case = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=4, dp=1, layers=1))
    route = case.machine.links[0].name
    ctx = _TargetPricingContext(case)
    floor_us = float(extract_sku_floors(case).launch_vec_us)
    a2a = OpSpec(
        op_id="s0.l0.dispatch", family="alltoall", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(256, 4096), local_shape=(256, 4096), padded_shape=(256, 4096), launch_shape=(256, 4096),
        dtype="bf16", link_bytes=float(4 * 1024 * 1024), link_scope="tp", metadata=(("group_axis", "tp"),),
    )
    # Pairwise groups keep traffic in per-rank volumes; routed messages exist only once rounds start.
    _, _, plan = _build_collective_plan(a2a, ctx, case, mode="pairwise", topology_route=route)
    assert plan is not None
    assert all(not getattr(g, "messages", ()) for g in plan.groups)
    small_us = _price_collective_on_topology(a2a, ctx, case, mode="pairwise", topology_route=route, minimum_us=floor_us)
    big_us = _price_collective_on_topology(
        replace(a2a, link_bytes=4.0 * a2a.link_bytes), ctx, case,
        mode="pairwise", topology_route=route, minimum_us=floor_us,
    )
    assert small_us is not None and small_us > floor_us
    assert big_us > small_us * 2.0
    # The automatic path (no explicit route) prices the same work instead of dropping the plan.
    auto_us = _price_collective_on_topology(a2a, ctx, case, mode="pairwise", minimum_us=floor_us)
    assert auto_us == pytest.approx(small_us)


def test_oneshot_selection_reconciles_link_ledger_with_executed_sends():
    from mindie_sim.optimization import solve_target_graph
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.rewrite import CollectiveTransform, Edit, OpRef, RewritePlan
    from mindie_sim.schema.operators import OpSpec

    link_rate = 50.0e9

    def _check(node, dem, expected_tx, useful_tx):
        assert node.link_bytes == pytest.approx(expected_tx)
        assert node.link_bytes >= useful_tx
        assert dem[4] == int(expected_tx * 1.0e9 / link_rate)
        assert node.l_star_lean_us == pytest.approx(max(dem) / 1000.0)
        assert node.gap_us == pytest.approx(max(0.0, node.dur_us - node.l_star_lean_us))

    # D6: automatic path preserves baseline ring without After-only pure-comm OneShot promotion;
    # explicit CollectiveTransform on TP8 reconciles one-shot sends (M to 7 peers).
    case8 = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=8, dp=1, layers=1))
    peaks8 = {0: {"hbm_peak_bytes_per_s": dict(case8.machine.dies[0].rates)["hbm"], "link_peak_bytes_per_s": link_rate}}
    msg = float(4 * 1024)
    ar = OpSpec(
        op_id="s0.l0.ar", family="allreduce", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(1, 2048), local_shape=(1, 2048), padded_shape=(1, 2048), launch_shape=(1, 2048),
        dtype="bf16", link_bytes=msg * (2.0 * 7.0 / 8.0), link_scope="tp", metadata=(("group_axis", "tp"),),
    )
    n_ar = _mk_target_node(0, ar, task_type="HCCL", dur_us=200.0, l_star_us=0.5, bound="link_bound")
    res_auto = solve_target_graph((n_ar,), ((0, 0, 0, 0, 500),), case8, peaks8, allow_unbound_formula=False)
    assert "One-Shot" not in res_auto.nodes[0].ascend_kernel
    plan8 = RewritePlan(edits=(Edit("os8", operation=CollectiveTransform(OpRef("atom", "s0.l0.ar"), "oneshot", case8.machine.links[0].name)),))
    res = solve_target_graph((n_ar,), ((0, 0, 0, 0, 500),), case8, peaks8, rewrite_plan=plan8, allow_unbound_formula=False)
    assert "oneshot" in res.nodes[0].ascend_kernel.lower()
    _check(res.nodes[0], res.demands_ns[0], res.nodes[0].link_bytes, 7.0 * msg)

    # Explicit CollectiveTransform: TP4 on framed one-credit link (1,024 B -> 3 x 400 B = 1,200 B wire/peer).
    raw4 = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=4, dp=1, layers=1)
    raw4["machine"]["links"][0].update(transaction_bytes=352, alignment_bytes=64, protocol_overhead_bytes=48)
    case4 = parse_case(raw4)
    peaks4 = {0: {"hbm_peak_bytes_per_s": dict(case4.machine.dies[0].rates)["hbm"], "link_peak_bytes_per_s": link_rate}}
    msg4 = float(1024)
    ar4 = replace(ar, op_id="s0.l0.ar4", link_bytes=msg4 * (2.0 * 3.0 / 4.0))
    n_ar4 = _mk_target_node(0, ar4, task_type="HCCL", dur_us=200.0, l_star_us=0.5, bound="link_bound")
    plan = RewritePlan(edits=(
        Edit("os4", operation=CollectiveTransform(OpRef("atom", "s0.l0.ar4"), "oneshot", case4.machine.links[0].name)),
    ))
    res4 = solve_target_graph(
        (n_ar4,), ((0, 0, 0, 0, 500),), case4, peaks4, rewrite_plan=plan, allow_unbound_formula=False,
    )
    assert res4.nodes[0].link_bytes == pytest.approx(3.0 * 1200.0)
    assert res4.nodes[0].link_bytes > 3.0 * msg4
    _check(res4.nodes[0], res4.demands_ns[0], 3.0 * 1200.0, 3.0 * msg4)


def test_solve_target_graph_accepts_public_prepare_scenario_sealed_plan_and_rejects_tampered():
    from mindie_sim.api import optimize_ladder
    from mindie_sim.export import build_export_bundle
    from mindie_sim.prediction.results import StepRequest
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.rewrite import ScenarioRequest, prepare_scenario
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.solve import solve_deployment

    raw = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], dp=1, tp=2, ep=1, pp=1, layers=2, prompt_tokens=1024, output_tokens=64, concurrency=1)
    case, req = parse_case(raw), StepRequest(phase="decode", tokens=1, context=1024, sequences=1)
    dossier = prepare_scenario(ScenarioRequest(case=case, step=req))
    assert build_export_bundle(raw, request=req, apply_target_envelope=True, rewrite_plan=dossier.unmerge_plan, allow_unbound_formula=False).step_span_us > 0.0
    assert build_export_bundle(raw, request=req, apply_target_envelope=True, rewrite_plan=dossier.mainline_plan, allow_unbound_formula=False).step_span_us > 0.0
    with pytest.raises(ModelingError, match="rewrite_digest_mismatch:framing_sha256"):
        build_export_bundle(raw, request=req, apply_target_envelope=True, rewrite_plan=replace(dossier.unmerge_plan, framing_sha256="a" * 64), allow_unbound_formula=False)
    assert optimize_ladder(case, req, max_trials=1, allow_unbound_formula=True) is not None
    from mindie_sim.sweep.pd import solve_optimal_deployment
    sol0 = solve_optimal_deployment("qwen3-32b", REPRESENTATIVE_SKUS["a5"], num_devices=1, prompt_tokens=128, output_tokens=1, ttft_sla_ms=5000.0, tpot_sla_ms=100.0, layers=1, feature_track="unmerged_pr", apply_target_envelope=False)
    dos_p = prepare_scenario(ScenarioRequest(case=parse_case(sol0["recommended_case"]), step=StepRequest(phase="prefill", tokens=128, context=0, sequences=1)))
    sol_p = solve_deployment("qwen3-32b", REPRESENTATIVE_SKUS["a5"], num_devices=1, prompt_tokens=128, output_tokens=1, ttft_slo_ms=5000.0, tpot_slo_ms=100.0, layers=1, mode="simulation", compare_track="unmerged_pr", rewrite_plan=dos_p.unmerge_plan)
    assert sol_p["feasible"] is True and sol_p["recommended_deployment"]["best_configuration"]["ttft_ms"] > 0.0


def test_formula_sram_sizes_norm_row_by_reduction_width_not_downstream_vocab():
    from mindie_sim.optimization.kernel_synthesis import FormulaKernelSpec, FormulaTensorSpec, _solve_formula_sram
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import OpSpec

    # Final RMSNorm (hidden 5,120) feeding a TP2 LM head with 75,968 local logits.
    case = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a3"], tp=2, dp=1, layers=1))
    datapath = datapath_for_die(case.machine.dies[0])
    hidden, vocab = 5120, 75968
    norm = OpSpec(
        op_id="s0.final_norm", family="rmsnorm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(1, hidden), local_shape=(1, hidden), padded_shape=(1, hidden), launch_shape=(1, hidden),
        dtype="bf16", vector_ops=float(hidden * 4), mte2_bytes=float(hidden * 2), mte3_bytes=float(hidden * 2),
        hbm_bytes=float(hidden * 4),
    )
    head = OpSpec(
        op_id="s0.lm_head", family="gemm", recipe="gemm", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.final_norm",), semantic_shape=(1, vocab, hidden), local_shape=(1, vocab, hidden),
        padded_shape=(1, vocab, hidden), launch_shape=(1, vocab, hidden), dtype="bf16",
        flops=float(2 * vocab * hidden), mte2_bytes=float(vocab * hidden * 2), mte3_bytes=float(vocab * 2),
        hbm_bytes=float(vocab * hidden * 2 + vocab * 2),
    )
    spec = replace(
        FormulaKernelSpec.from_ops((norm, head)),
        outputs=(FormulaTensorSpec("logits", (1, vocab), "bf16", role="output"),),
    )
    _, algo, tile, ub_pk, ub_budget, *_rest, ok = _solve_formula_sram(spec, datapath)
    assert "N=5120," in tile
    assert ok is True and 0 < ub_pk <= ub_budget
    assert algo in {"resident_1pass", "online_reduce"}
