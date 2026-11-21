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
    DagEdge, EventDagSpec, EvgVisitorNode, FusedRegionSummary, ResourceChoiceProgram,
    SemanticWorkItem, TensorRangeObligation, build_dual_inputs_from_scenario,
    certify_lower_bound, generate_gcd_core_lattice, select_catlass_candidates,
    select_dsl_candidates, solve_dsl, solve_framework, streamk_crossing_partials,
)
from mindie_sim.prediction.results import OpOccurrence, OperatorSequenceRow, StepEvaluation, StepRequest
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
    carveout = 16 * 1024
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
    receipt = verify_plan(scenario, replace(applied, streams=(plan,)), step_eval=step)
    assert receipt.passed and receipt.overlap_pct == 60

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
    read = TensorRangeObligation(0, "operand", 0, ((0, 17), (12, 27)), "hbm_cold", "owner")
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


def test_bound_gate_checks_certificate_and_shipping_threshold(real_graph):
    scenario, applied = real_graph
    program = ResourceChoiceProgram(2, (23, 19), (((4, 0), (0, 2)),))
    dag = EventDagSpec((10, 20), (DagEdge(0, 1, 5),))
    cert, _ = certify_lower_bound(program, dag)
    assert cert.l_star_lean_ns == 35
    evidence = dict(dual_program=program, event_dag=dag, dual_certificate=cert)
    verified = verify_plan(scenario, applied, wall_ns=35, **evidence)
    assert verified.passed and verified.l_star_ns == 35 and len(verified.gates) == 5
    with pytest.raises(SoundnessInversionError, match="SOUNDNESS_BUG_LOWER_BOUND_INVERTED"):
        verify_plan(scenario, applied, wall_ns=34, **evidence)
    for forged in (replace(cert, path_edges=(1,)), replace(cert, l_star_lean_ns=36)):
        receipt = verify_plan(scenario, applied, wall_ns=100, **{**evidence, "dual_certificate": forged})
        assert _reason(receipt, 4) == "LEAN_CERTIFICATE_INVALID"
    receipt = verify_plan(scenario, applied, wall_ns=100, dual_certificate=cert)
    assert _reason(receipt, 4) == "LEAN_CERTIFICATE_INVALID"
    inverted = verify_plan(scenario, applied, wall_ns=100, framework_ns=80, dsl_ns=81, **evidence)
    assert _reason(inverted, 4) == "LADDER_MONOTONICITY_VIOLATION"

    near = verify_plan(scenario, applied, wall_ns=35, baseline_u_ns=2000, baseline_m_ns=3000, **evidence)
    assert near.shipping_regression is None
    slow = verify_plan(scenario, applied, wall_ns=35, baseline_u_ns=2000, baseline_m_ns=3001, **evidence)
    assert slow.passed and slow.shipping_regression == ShippingRegression(
        delta_ns=1001, baseline_u_ns=2000, baseline_m_ns=3001,
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
                EvgVisitorNode("store", "aux_store", "bf16", inputs=("acc",)))
    return replace(base, family="grouped_gemm", dtype="bf16", local_shape=shape, flops=2 * prod(shape),
                   metadata=(("group_rows", (32, 32)), ("evg_tree", visitors)))


def _select(op, path, **facts):
    return select_catlass_candidates(replace(op, metadata=tuple({**dict(op.metadata), **facts}.items())), path, 4, 8)


@pytest.mark.parametrize("hardware", SKUS)
def test_catlass_scheduler_choices_preserve_workspace_and_dispatch_constraints(matrix_op, hardware):
    case = compose_case("qwen3-32b", hardware, tp=1, dp=1, layers=1)
    scenario = resolve_scenario(case, StepRequest("decode", context=64))
    path = scenario.datapath
    specs = _select(matrix_op, path)
    evg = "evg_ub" if path.has_l0c_to_ub else "evg_gm"
    by_template = {spec.template_id: spec for spec in specs}
    expected = ("ping_pong", "mutex", "streamk", "multi_core_splitk", "swizzle", "grouped_slice_m", evg)
    assert {f"catlass.{kind}" for kind in expected} == by_template.keys()
    assert by_template[f"catlass.{evg}"].workspace_bytes == (0 if path.has_l0c_to_ub else 64 * 192 * 4)
    assert by_template["catlass.grouped_slice_m"].l2_cache_hint_b == "disable"
    # Three output blocks split into three K slices occupy runs of 3, 2, 2, 2 iterations.
    assert by_template["catlass.streamk"].workspace_bytes == (1 + 1 + 2 + 1) * 64 * 64 * 4
    assert {s.splitk_factor for s in specs if s.scheduler == "multi_core_splitk"} == {2, 3}
    assert {s.swizzle_direction for s in specs if s.template_id == "catlass.swizzle"} == {"Zn", "Nz"}
    for stages, bias, flag, accepted in (
        ((5, 5, 5, 5, 6), True, False, True),
        ((5, 5, 5, 5, 7), True, False, False),
        ((2, 2, 2, 2, 2), False, True, False),
        ((2, 2, 2, 2, 1), False, True, True),
    ):
        options = _select(matrix_op, path, mutex_stages=stages, has_bias=bias, enable_unit_flag=flag)
        assert any(s.template_id == "catlass.mutex" for s in options) is accepted
    one_chunk = replace(matrix_op, local_shape=(64, 192, 128))
    assert all(s.scheduler != "multi_core_splitk" for s in _select(one_chunk, path))


@pytest.mark.parametrize(("dtype", "lhs", "rhs", "factor"), (
    ("mxfp8", "mxfp8", "mxfp8", 1),
    ("mxfp4", "mxfp4", "mxfp4", 1),
    ("a8w4_mx", "mxfp8", "mxfp4", 16),
))
def test_catlass_scaled_formats_require_declared_native_mode(real_graph, matrix_op, dtype, lhs, rhs, factor):
    scenario, _ = real_graph
    path = scenario.datapath
    facts = dict(native_matrix_mode=dtype, dtype_a=lhs, dtype_b=rhs, swiglu_mx_quant=True)
    assert not _select(matrix_op, path, **facts)
    native_mode = NativeComputeMode(dtype, "matrix", (dtype,), "cube", ("fp32",))
    arch = path.compute_arch.declared
    declared = replace(arch, native_modes=(*arch.native_modes, native_mode))
    capable = replace(path, compute_arch=replace(path.compute_arch, declared=declared))
    specs = _select(matrix_op, capable, **facts)
    selected = next(s for s in specs if s.template_id == f"catlass.{dtype}")
    assert (selected.dtype_a, selected.dtype_b, selected.mx_scale_group, selected.l1_scale_factor_k) == (
        lhs, rhs, 32, factor,
    )
    assert selected.prologue_b == ("DecompressFp4ToFp8" if dtype == "a8w4_mx" else None)
    epilogue = next(s for s in specs if s.template_id == "catlass.grouped_swiglu_mx_quant")
    assert epilogue.l2_cache_hint_b == "disable" and epilogue.scheduler == "grouped_slice_m"


@pytest.mark.parametrize(("backend", "limit"), (("catlass", 1024), ("tla_dsl", 2048)))
def test_catlass_simt_admits_exact_thread_caps(real_graph, matrix_op, backend, limit):
    scenario, _ = real_graph
    facts = dict(simt_backend=backend, simt_buffers=(LiveBufferSpec("scratch", "ub", 1024, 0, 1),),
                 simt_workspace_bytes=2048)
    admitted = _select(matrix_op, scenario.datapath, simt_thread_block_dim=(limit // 32, 32, 1), **facts)
    simt, = tuple(s for s in admitted if s.simt_thread_block_dim is not None)
    assert prod(simt.simt_thread_block_dim) == limit and simt.aic_budget == 0 and simt.workspace_bytes == 2048
    too_wide = _select(matrix_op, scenario.datapath, simt_thread_block_dim=(limit + 1, 1, 1), **facts)
    assert all(s.simt_thread_block_dim is None for s in too_wide)


@pytest.mark.parametrize(("kernel", "mode"), (
    ("flash_kda", "mix_cube_vector"), ("mega_recurrent_kda", "mix_cube_vector"),
    ("commit_recurrent_kda", "vector_only"), ("mla_prolog_v4", "mix_cube_vector"),
    ("block_attn_res_update_rms_norm", "vector_only"), ("quant_sparse_lightning_indexer", "mix_cube_vector"),
    ("mixed_quant_sparse_flash_mla", "mix_cube_vector"),
))
def test_dsl_catalog_requires_real_pool_and_owner_capability(real_graph, matrix_op, kernel, mode):
    scenario, _ = real_graph
    path = scenario.datapath
    scratch = LiveBufferSpec("tile", "ub", 1024, 0, 1, slots=2)
    tile = TileCandidate("resident_1pass", TileShape(16, 16, 16), 2, (4, 8), 8, 1,
                         (scratch,), owner_axis="aiv")
    pipeline = TilePipelineSpec(kernel, (tile,))
    op = replace(matrix_op, family=kernel, deps=(), metadata=(("dsl_tile_spec", pipeline), ("dsl_workspace_bytes", 4096)))
    (spec, members), = select_dsl_candidates((op,), path, 4, 8)
    assert members == (op.op_id,)
    assert spec.kernel_id == f"dsl.{kernel}" and spec.execution_mode == mode
    assert spec.aic_budget == (0 if mode == "vector_only" else 4) and spec.aiv_budget == 8
    assert spec.ring_depths == (("tile", "ub", 2),)
    assert (spec.ub_peak_bytes, spec.workspace_bytes_total) == (2048, 4096)
    assert not select_dsl_candidates((op,), path, 4, 0)
    if mode == "mix_cube_vector":
        assert not select_dsl_candidates((op,), path, 0, 8)
    too_large = replace(tile, buffers=(replace(scratch, bytes_per_owner=dict(path.buffers)["ub"]),))
    invalid = replace(op, metadata=(("dsl_tile_spec", replace(pipeline, candidates=(too_large,))),
                                    ("dsl_workspace_bytes", 4096)))
    assert not select_dsl_candidates((invalid,), path, 4, 8)
