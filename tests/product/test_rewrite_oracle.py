"""Contract tests for archetype screening and full occurrence verification.

[Role] Exercise interning, context reuse, bounded batches, and final replay.
[Invariants] Each layer is counted once and full replay has its own receipt.
[Timeouts & Locks] Monotonic deadlines and bounded cold and warm candidate batches.
[Side-Effect Policy] Analytical replay and a temporary SQLite store.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from fractions import Fraction
import sqlite3
from time import monotonic

import pytest

from _support.axes import REPRESENTATIVE_SKUS
from _support.linear_cases import tile_edge
from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.compose import compose_case
from mindie_sim.optimization import (
    EventDagSpec, ResourceChoiceProgram, build_dual_inputs_from_scenario, certify_lower_bound,
    compute_compulsory_hbm_cut, inward_floor_ns, solve_dsl, solve_framework,
)
from mindie_sim.optimization import contracts
from mindie_sim.optimization.dsl_solver import _catlass_proposals, _dsl_proposals
from mindie_sim.prediction.efficiency.cost import CostLedger, CostModel
from mindie_sim.prediction.efficiency.analysis import scale_case_hardware
from mindie_sim.prediction.evaluation import EvaluationContext, ExecutionBindings
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.program.primitives import PrimitiveStore
from mindie_sim.prediction.program import Stage
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.spine import PredictionSpine
from mindie_sim.prediction.structure import parse_case
from mindie_sim.rewrite import (
    Edit, EvalBudget, OpRef, OracleCache, PortRef, QuantizeContract, ScenarioRequest, StepArchetype, WeightPrefetch,
    decode_rewrite_plan, encode_rewrite_plan, evaluate_rewrites, finalize_plan, intern_step_archetypes, plan_sha256, prepare_scenario,
    verify_plan,
)
from mindie_sim.rewrite import oracle
from mindie_sim.schema.common import ModelingError
from mindie_sim.templates import CatlassProblem, FusedRegionSummary, SemanticWorkItem, builtin_template_catalog
from mindie_sim.templates.dsl_catalog import ATTN_RES, AttnResProblem
from mindie_sim.templates.dsl_lower import ATTN_RES_VL

pytestmark = pytest.mark.tier0

PREFETCH_BATCH_BUDGET_MS = 50
PREFETCH_COPY_STEP_BYTES = 4096
NS_PER_SECOND = 10**9


@pytest.fixture(scope="module")
def request_inputs():
    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=6)
    return ScenarioRequest(case, StepRequest("decode", context=64))


@pytest.fixture(scope="module")
def dossier(request_inputs):
    return prepare_scenario(request_inputs)


@pytest.fixture(scope="module")
def candidates(dossier):
    return _candidate_plans(dossier, {0, 1, 3, 5})


def _candidate_plans(dossier, layers):
    scenario = dossier.handle.scenario
    quant = scenario.case.model.quant
    targets = sorted((op for op in scenario.atomic_ops[0].ops
                      if op.layer in layers and op.family in {"qkv_projection", "output_projection"}),
                     key=lambda op: (op.family, op.layer))
    return tuple(replace(dossier.unmerge_plan, edits=(Edit(
        f"quantize-{i}",
        operation=QuantizeContract((op.op_id,), quant.weight_format, quant.activation_format,
                                   quant.kv_dtype, quant.activation_group_size, quant.activation_scale_format),
    ),)) for i, op in enumerate(targets))


def test_archetypes_keep_boundaries_and_exact_layer_multiplicity(dossier):
    rank, = dossier.handle.scenario.atomic_ops
    actual = intern_step_archetypes(rank.ops)
    assert actual == dossier.archetypes == intern_step_archetypes(tuple(rank.ops))
    assert 2 <= len(actual) <= 5
    assert Counter(layer for archetype in actual for layer in archetype.layer_indices) == Counter(range(6))
    assert {a.layer_indices for a in actual} == {(0,), (1, 2, 3, 4), (5,)}
    assert len({a.structural_key for a in actual}) == len(actual)
    for archetype in actual:
        assert isinstance(archetype, StepArchetype)
        assert archetype.multiplicity == len(archetype.layer_indices)
        assert archetype.representative_layer == min(archetype.layer_indices)
        assert archetype.ops == tuple(op for op in rank.ops if op.layer == archetype.representative_layer)
    assert sum(len(a.ops) * a.multiplicity for a in actual) == len(rank.ops)
    assert dossier.archetype_coverage_ratio == 1

    # A distinct interior shape must split its bucket without changing any multiplicity.
    changed = next(op for op in rank.ops if op.layer == 2)
    variant = tuple(replace(op, local_shape=(1,)) if op is changed else op for op in rank.ops)
    shapes = intern_step_archetypes(variant)
    assert {a.layer_indices for a in shapes} == {(0,), (1, 3, 4), (2,), (5,)}
    assert sum(a.multiplicity for a in shapes) == 6


def test_prepare_reuses_one_context_and_evicts_digest_and_request_alias(request_inputs, monkeypatch):
    contexts, bindings, models = [], [], []
    create_context, create_bindings = EvaluationContext.create, ExecutionBindings.create
    init_model = CostModel.__init__

    def bind(cls, *args, **kwargs):
        result = create_bindings(*args, **kwargs)
        bindings.append(result)
        return result

    def create(cls, *args, **kwargs):
        result = create_context(*args, **kwargs)
        contexts.append(result)
        return result

    def init(self, *args, **kwargs):
        init_model(self, *args, **kwargs)
        models.append(self)

    monkeypatch.setattr(EvaluationContext, "create", classmethod(create))
    monkeypatch.setattr(ExecutionBindings, "create", classmethod(bind))
    monkeypatch.setattr(CostModel, "__init__", init)

    cache = OracleCache(max_entries=2)
    first = prepare_scenario(request_inputs, cache=cache)
    context = first.handle.evaluation_context
    assert contexts == [context]
    assert len(bindings) == len(models) == 1
    assert prepare_scenario(request_inputs, cache=cache).handle is first.handle
    assert cache.get(first.scenario_digest) is first.handle
    assert len(contexts) == len(bindings) == len(models) == 1

    def different_context(length):
        return replace(request_inputs, step=replace(request_inputs.step, context=length))

    second_request = different_context(96)
    second = prepare_scenario(second_request, cache=cache)
    assert cache.get(first.scenario_digest) is first.handle  # refresh its LRU position
    third = prepare_scenario(different_context(128), cache=cache)
    assert len({first.scenario_digest, second.scenario_digest, third.scenario_digest}) == 3
    assert cache.get(second.scenario_digest) is None
    assert cache.get(third.scenario_digest) is third.handle
    rebuilt = prepare_scenario(second_request, cache=cache)
    assert rebuilt.scenario_digest == second.scenario_digest
    assert rebuilt.handle is not second.handle
    assert len(contexts) == len(bindings) == len(models) == 4


@pytest.mark.parametrize("size", (4, 8))
def test_warm_batch_screens_fresh_prefetch_work_within_budget(size):
    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=64)
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", context=64)))
    archetypes = sorted(dossier.archetypes, key=lambda a: a.representative_layer)
    assert [a.multiplicity for a in archetypes] == [1, 62, 1]

    def prefetch(archetype, amount):
        target = next(op for op in archetype.ops if op.family == "qkv_projection")
        anchor, = target.deps
        edit = Edit("prefetch", operation=WeightPrefetch(
            PortRef(OpRef("atom", target.op_id), "weight", 0),
            PortRef(OpRef("atom", anchor), "output", 0), max_bytes=amount,
        ))
        return replace(dossier.unmerge_plan, edits=(edit,))

    warmup = tuple(prefetch(a, PREFETCH_COPY_STEP_BYTES) for a in archetypes)
    warmed = evaluate_rewrites(dossier.handle, (dossier.unmerge_plan, *warmup))
    base, *warm_copies = warmed.results
    assert base.accepted and all(item.accepted for item in warm_copies)
    amounts = tuple((i + 2) * PREFETCH_COPY_STEP_BYTES for i in range(size))
    plans = tuple(prefetch(archetypes[i % len(archetypes)], amount) for i, amount in enumerate(amounts))
    assert {plan_sha256(plan) for plan in plans}.isdisjoint(item.plan_sha256 for item in warmed.results)

    result = evaluate_rewrites(dossier.handle, plans)
    print(f"{size} fresh prefetch plans took {result.elapsed_ms:.3f} ms")
    assert result.elapsed_ms < PREFETCH_BATCH_BUDGET_MS
    assert result.scenario_digest == dossier.scenario_digest and result.scope == "archetype"
    assert len({item.plan_sha256 for item in result.results}) == size
    die, = dossier.handle.scenario.case.machine.dies
    hbm_bytes_per_second = dict(die.rates)["hbm"]
    for candidate, plan, amount in zip(result.results, plans, amounts, strict=True):
        assert candidate.plan_sha256 == plan_sha256(plan)
        assert candidate.accepted and candidate.coverage_ratio == 1
        dma_ns = round(amount * NS_PER_SECOND / hbm_bytes_per_second)
        assert dma_ns > 0
        assert candidate.estimated_wall_ns == base.estimated_wall_ns + dma_ns
        assert candidate.archetype_wall_ns == base.archetype_wall_ns + dma_ns
        assert candidate.verification.scope == "archetype" and candidate.verification.passed
        assert candidate.verification.wall_ns == candidate.estimated_wall_ns
        assert len(candidate.verification.gates) == 4

    replay = finalize_plan(dossier.handle, plans[-1])
    assert replay.verification.passed
    copy, = (row for row in replay.step_eval.operators if row.family == "prefetch")
    assert copy.start_ms < copy.end_ms
    assert sum(item.amount for item in copy.work if item.resource == "hbm") == amounts[-1]


@pytest.mark.parametrize("limit", (0, 1, 3, 8))
def test_candidate_budget_preserves_order_and_caps_consumption(dossier, candidates, limit):
    requested = (*candidates, candidates[0])
    consumed = []

    def plans():
        for candidate in requested:
            consumed.append(candidate)
            yield candidate

    result = evaluate_rewrites(dossier.handle, plans(), budget=EvalBudget(max_candidates=limit))
    assert consumed == list(requested[:limit])
    assert [item.plan_sha256 for item in result.results] == [plan_sha256(p) for p in consumed]
    assert all(item.accepted for item in result.results)
    with pytest.raises(ModelingError, match="rewrite_candidate_budget_invalid"):
        EvalBudget(max_candidates=-1)


def test_oracle_deadline_and_cooperative_cancellation(dossier, candidates, request_inputs):
    expired = EvalBudget(deadline_mono=monotonic() - 1)
    with pytest.raises(TimeoutError, match="rewrite_oracle_deadline"):
        prepare_scenario(request_inputs, budget=expired)
    with pytest.raises(TimeoutError, match="rewrite_oracle_deadline"):
        finalize_plan(dossier.handle, candidates[0], budget=expired)
    expired_batch = evaluate_rewrites(dossier.handle, candidates[:2], budget=expired)
    for item in expired_batch.results:
        assert (item.status, item.reason, item.verification) == ("cancelled", "rewrite_oracle_deadline", None)
    stop = iter((False, True))
    cancelled = evaluate_rewrites(dossier.handle, candidates[:1],
                                  budget=EvalBudget(cancelled=lambda: next(stop, True)))
    item, = cancelled.results
    assert (item.status, item.reason, item.verification) == ("cancelled", "rewrite_oracle_cancelled", None)
    assert item.estimated_wall_ns is None
    assert evaluate_rewrites(dossier.handle, candidates[:1]).results[0].accepted


def test_finalize_replays_every_occurrence_and_matches_spine(dossier, candidates):
    handle = dossier.handle
    scenario = handle.scenario
    program, dag, _ = build_dual_inputs_from_scenario(scenario)
    certificate, _ = certify_lower_bound(program, dag)
    options = dict(dual_program=program, event_dag=dag, dual_certificate=certificate)
    plan = candidates[0]
    assert evaluate_rewrites(handle, (plan,)).results[0].accepted
    result = finalize_plan(handle, plan, **options)
    reference = PredictionSpine().run_step(scenario.atomic_step)
    assert result.scenario_digest == dossier.scenario_digest
    assert result.scope == result.verification.scope == "full"
    assert result.plan_sha256 == result.verification.plan_sha256 == plan_sha256(plan)
    assert result.verification.scenario_sha256 == scenario.scenario_sha256
    assert result.applied.edit_receipts[0].status == "applied"
    assert result.step_eval.span_ms == pytest.approx(reference.span_ms, rel=1e-12)
    assert result.wall_ns == result.verification.wall_ns == round(reference.span_ms * 1_000_000)
    # A precision annotation leaves the executable timing and occurrence graph unchanged.
    expected = {row.occurrence: row for row in reference.operators}
    assert {row.occurrence for row in result.step_eval.operators} == expected.keys()
    for row in result.step_eval.operators:
        baseline = expected[row.occurrence]
        assert row.work == baseline.work
        assert (row.start_ms, row.end_ms) == pytest.approx((baseline.start_ms, baseline.end_ms))
    assert len(result.step_eval.operators) == sum(len(rank.ops) for rank in scenario.atomic_ops)
    assert 0 < result.verification.l_star_ns == certificate.l_star_lean_ns <= result.wall_ns
    assert len(result.verification.gates) == 5 and result.verification.passed
    assert result.verification == verify_plan(
        scenario, result.applied, step_eval=result.step_eval,
        baseline_u_ns=handle.full_u_ns, baseline_m_ns=handle.full_m_ns, **options,
    )


def test_finalize_binds_certificate_to_prepared_scenario():
    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=2, dp=1, layers=1)
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", context=64)))
    handle, plan = dossier.handle, dossier.unmerge_plan
    program, dag, _ = build_dual_inputs_from_scenario(handle.scenario)
    certificate, _ = certify_lower_bound(program, dag)
    accepted = finalize_plan(handle, plan, dual_program=program, event_dag=dag, dual_certificate=certificate)
    assert accepted.verification.passed
    assert accepted.verification.l_star_ns == certificate.l_star_lean_ns

    invented_ns = accepted.wall_ns + 1
    other_program, other_dag = ResourceChoiceProgram(1, (invented_ns,), ()), EventDagSpec((invented_ns,), ())
    other_certificate, _ = certify_lower_bound(other_program, other_dag)
    assert other_certificate.l_star_lean_ns > accepted.wall_ns
    rejected = finalize_plan(handle, plan, dual_program=other_program, event_dag=other_dag,
                             dual_certificate=other_certificate)
    direct = verify_plan(handle.scenario, accepted.applied, wall_ns=accepted.wall_ns,
                         dual_program=other_program, event_dag=other_dag, dual_certificate=other_certificate)
    for receipt in (rejected.verification, direct):
        *ordinary, bound = receipt.gates
        assert all(gate.passed for gate in ordinary)
        assert receipt.wall_ns == accepted.wall_ns
        assert (bound.passed, bound.reason, bound.detail) == (
            False, "LEAN_CERTIFICATE_INVALID", "certificate_unbound_to_scenario",
        )


def test_reopening_primitive_store_does_not_reuse_closed_context(tmp_path, request_inputs):
    path = tmp_path / "primitives.sqlite"
    with PrimitiveStore(path, writable=True):
        pass
    cache = OracleCache()
    with PrimitiveStore(path) as initial:
        first = prepare_scenario(request_inputs, cache=cache, primitives=initial)
        first_result = finalize_plan(first.handle, first.unmerge_plan)
        assert first_result.verification.passed
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        initial.connection.execute("SELECT 1")

    with PrimitiveStore(path) as reopened:
        second = prepare_scenario(request_inputs, cache=cache, primitives=reopened)
        assert second.scenario_digest == first.scenario_digest
        assert second.handle is not first.handle
        assert second.handle.evaluation_context.cost.primitives is reopened
        result = finalize_plan(second.handle, second.unmerge_plan)
        assert result.verification.passed and result.wall_ns == first_result.wall_ns


class _DeclaredGraphSpine(PredictionSpine):
    def __init__(self, *ops):
        super().__init__()
        self.ops = ops

    def lower(self, case, request, **kwargs):
        prepared = super().lower(case, request, **kwargs)
        rank, = prepared.graph.executable
        graph = replace(prepared.graph, executable=(replace(rank, ops=self.ops),), streams=())
        return replace(prepared, graph=graph)


@pytest.mark.parametrize("geometry", ((1, 1, 1), (32, 32, 160)), ids=("scalar", "multi-panel"))
def test_catalog_matrix_replay_binds_lowered_obligations(geometry):
    m, n, k = geometry
    op = gemm_op(op_id="gemm", deps=(), requests=(), stage=0, layer=0,
                 m=m, n=n, k=k, quant=QuantSpec("bf16"), category="dense", phase="decode",
                 metadata=(("l1_tile_shape", (32, 32, 64)), ("l0_tile_shape", (32, 32, 32))))
    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1)
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode")), spine=_DeclaredGraphSpine(op))
    scenario = dossier.handle.scenario
    path = scenario.datapath
    (operation,), spec = next(_catlass_proposals((op,), path, path.aic_count, path.aiv_count))
    assert operation.template_id == "catlass.basic_pingpong" and operation.template_id not in dossier.handle.recipe_book.recipes
    plan = replace(dossier.unmerge_plan, edits=(Edit("matrix", operation=operation),))
    screened, = evaluate_rewrites(dossier.handle, (plan,)).results
    full = finalize_plan(dossier.handle, plan)
    assert screened.accepted, tuple(g.detail for g in screened.verification.gates) if screened.verification else screened.reason
    assert full.verification.passed, full.verification
    summary, = full.region_summaries
    assert type(summary) is contracts.FusedRegionSummary is FusedRegionSummary
    assert summary.spec == spec and summary.op_ids == (op.op_id,)
    lowered = builtin_template_catalog().lower("K2", spec, CatlassProblem(m, n, k), path)
    if m == 1:
        assert (summary.logical_transfer_bytes, summary.dram_sector_bytes) == (6, 96)
    else:
        assert lowered.stages_count == 106
    assert summary.logical_transfer_bytes == 2 * (m * k + k * n + m * n)
    cut = compute_compulsory_hbm_cut((summary,))[0]
    assert cut.read_bytes + cut.write_bytes == summary.dram_sector_bytes
    program, dag, _ = build_dual_inputs_from_scenario(scenario, region_summaries=(summary,))
    certificate, _ = certify_lower_bound(program, dag)
    receipt = verify_plan(scenario, full.applied, wall_ns=full.wall_ns, region_summaries=(summary,),
                          dual_program=program, event_dag=dag, dual_certificate=certificate)
    assert receipt.passed and receipt.l_star_ns == full.verification.l_star_ns


def test_constructive_search_replays_catalog_vector_kernel(monkeypatch):
    problem = AttnResProblem(tokens=1)
    width = problem.hidden_size
    op = vector_op(op_id="residual", deps=(), stage=0, layer=0, requests=(), elements=width,
                   ops_per_element=18, vector_ops_override=18 * width + 3 + 2 * ATTN_RES_VL,
                   dtype="bf16", family="elementwise", phase="decode", recipe="elementwise",
                   metadata=(("dsl_kernel", ATTN_RES), ("dsl_problem", problem)),
                   shape=(1, width), read_bytes_override=16 * width + 8, write_bytes_override=6 * width)
    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1)
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode")), spine=_DeclaredGraphSpine(op))
    path = dossier.handle.scenario.datapath
    ((operation,), spec), = _dsl_proposals((op,), path, path.aic_count, path.aiv_count)
    assert operation.template_id == ATTN_RES + ".inplace" and operation.template_id not in dossier.handle.recipe_book.recipes
    screened = []

    def record(handle, plans, **kwargs):
        result = evaluate_rewrites(handle, plans, **kwargs)
        screened.extend((plan, candidate) for plan, candidate in zip(plans, result.results, strict=True))
        return result

    monkeypatch.setattr(oracle, "evaluate_rewrites", record)
    framework = solve_framework(dossier, max_trials=0)
    solve_dsl(dossier, framework, max_trials=3)
    (plan, candidate), = [(plan, result) for plan, result in screened
                          if any(edit.operation == operation for edit in plan.edits)]
    assert candidate.accepted, tuple(g.detail for g in candidate.verification.gates) if candidate.verification else candidate.reason
    full = finalize_plan(dossier.handle, plan)
    assert full.verification.passed, full.verification
    summary, = full.region_summaries
    assert summary.spec == spec
    assert sum(item.logical_ops for item in summary.semantic_work) == op.vector_ops


def _catalog_case():
    return compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1)


def _replay_plans(case, ops, proposals, *, primitive_table=None):
    dossier = prepare_scenario(
        ScenarioRequest(case, StepRequest("decode"), primitive_table=primitive_table),
        spine=_DeclaredGraphSpine(*ops),
        primitive_table=primitive_table,
    )
    handle = dossier.handle
    path = handle.scenario.datapath
    for (operation,), spec in proposals(ops, path, path.aic_count, path.aiv_count):
        yield handle, replace(dossier.unmerge_plan, edits=(Edit("catalog", operation=operation),)), spec


def _replay_checked(handle, plan):
    screened, = evaluate_rewrites(handle, (plan,)).results
    assert screened.accepted, screened.verification or screened.reason
    full = finalize_plan(handle, plan)
    assert full.verification.passed, full.verification
    return full


@pytest.mark.parametrize("allocation_key", ("allocation_id", "weight_allocation_id"))
def test_cold_weight_and_catalog_operand_share_one_allocation(allocation_key):
    width = 64
    op = gemm_op(op_id="gemm", deps=(), requests=(), stage=0, layer=0, m=1, n=width, k=width,
                 quant=QuantSpec("bf16"), category="dense", phase="decode",
                 metadata=((allocation_key, "W"), ("weight_initial_residency", "hbm_cold")))
    case = scale_case_hardware(_catalog_case(), hbm_bw_scale=1e-5)
    handle, plan, _ = next(_replay_plans(case, (op,), _catlass_proposals))
    full = _replay_checked(handle, plan)
    summary, = full.region_summaries
    assert "W" in {item.allocation_id for item in summary.read_obligations}
    program, dag, names = build_dual_inputs_from_scenario(
        handle.scenario, applied=full.applied, region_summaries=full.region_summaries,
    )
    read_ns = next(work for name, work in zip(names, program.global_work) if name.startswith("hbm_bus_r@"))
    path = handle.scenario.datapath
    assert path.hbm_rw_alpha == 1
    read_bytes, write_bytes = 2 * (width * width + width), 2 * width
    read_service_ns = Fraction((read_bytes + write_bytes) * NS_PER_SECOND, int(path.hbm_read_bytes_per_s))
    assert read_ns == inward_floor_ns(read_service_ns)
    bound, _ = certify_lower_bound(program, dag)
    assert bound.l_star_lean_ns == full.verification.l_star_ns <= full.wall_ns


@pytest.mark.parametrize("alpha", (0, 0.5))
def test_catalog_replay_preserves_directional_hbm_service(alpha):
    tile_width = 32
    problem = CatlassProblem(tile_width + 1, tile_width + 1, tile_width)
    tile = (tile_width,) * 3
    op = gemm_op(op_id="gemm", deps=(), requests=(), stage=0, layer=0, m=problem.m, n=problem.n, k=problem.k,
                 quant=QuantSpec("bf16"), category="dense", phase="decode",
                 metadata=(("l1_tile_shape", tile), ("l0_tile_shape", tile)))
    case = parse_case(_catalog_case())
    die, = case.machine.dies
    rate = 1024 * 1000
    die = replace(die, hbm_read_bytes_per_s=rate, hbm_write_bytes_per_s=rate, hbm_rw_alpha=alpha,
                  rates=tuple(dict(die.rates, hbm=rate).items()))
    case = replace(case, machine=replace(case.machine, dies=(die,)))
    handle, plan, spec = next(_replay_plans(case, (op,), _catlass_proposals, primitive_table=False))
    full = _replay_checked(handle, plan)
    context = handle.evaluation_context
    execution = Execution(case, context.cost, recipes=handle.recipe_book, bindings=context.bindings,
                          ledger=CostLedger(), isolated_replica=0, evaluation=context)
    prepared, base_recipe = execution.prepare(op, 0)
    bound_context = execution._context(prepared, 0, "none", recipe=base_recipe)
    direct = builtin_template_catalog().lower("K2", spec, problem, bound_context.datapath, context=bound_context)
    engine = execution.engine
    done = direct.program.start(engine, "direct", engine.submit("ready"))
    engine.advance(until=(done,))
    assert full.step_eval.span_ms == pytest.approx(done.time, abs=1e-12)
    if alpha == 0:
        # The critical tile follows the contiguous A read and strided B read.
        # Its B load and C store each pay 256 B per strided row.
        a_bytes = 2 * problem.m * problem.k
        strided_tile_bytes = 256 * tile_width
        assert full.step_eval.span_ms == pytest.approx((a_bytes + 2 * strided_tile_bytes) / (rate / 1000), abs=2e-5)
    bus = direct.program.hbm_bus
    evidence = full.step_eval.execution_evidence
    assert evidence["serviced"][bound_context.resource_names["hbm"]] == 0
    for row, counter in ((bus.row_r, bus.ctr_r), (bus.row_w, bus.ctr_w)):
        assert engine.delivered[counter] > 0
        assert evidence["serviced"][row] == pytest.approx(engine.serviced[row])
        assert evidence["resource_busy_ms"][row] == pytest.approx(engine.busy_ms[row])


def _pointwise(name, *, deps=(), elements=1, **overrides):
    return vector_op(op_id=name, deps=deps, requests=(), stage=0, layer=0, elements=elements,
                     ops_per_element=1, dtype="bf16", family="elementwise", phase="decode",
                     recipe="elementwise", **overrides)


@pytest.mark.parametrize("separate_declaring_ops", (False, True))
def test_attention_residual_fusion_replays_each_declaration(separate_declaring_ops):
    problem = AttnResProblem(tokens=1)
    width = problem.hidden_size
    head = _pointwise("head", elements=width)
    declaration = dict(dsl_kernel=ATTN_RES, dsl_problem=problem, dsl_members=("head", "tail"))
    tail_metadata = {**declaration, "dsl_preserve_ports": ()}
    if separate_declaring_ops:
        head = replace(head, metadata=(*head.metadata, *declaration.items()))
    else:
        tail_metadata["dsl_variants"] = ({}, {"dsl_kernel": ATTN_RES + ".inplace"})
    tail = _pointwise("tail", deps=("head",), elements=width, metadata=tuple(tail_metadata.items()),
                      vector_ops_override=17 * width + 3 + 2 * ATTN_RES_VL)

    def proposals(ops, path, aic, aiv):
        return _dsl_proposals(ops, path, aic // 2, aiv // 2)

    first, second = _replay_plans(_catalog_case(), (head, tail), proposals)
    plans = (first[1], second[1])
    assert len({plan_sha256(plan) for plan in plans}) == len(plans)
    for handle, plan, spec in (first, second):
        decoded = decode_rewrite_plan(encode_rewrite_plan(plan))
        assert decoded == plan
        full = _replay_checked(handle, decoded)
        summary, = full.region_summaries
        assert summary.spec == spec and summary.op_ids == ("head", "tail")


def test_declared_fusion_retains_external_predecessor_in_replay():
    src = _pointwise("src")
    make = _pointwise("make", deps=("src",))
    take = _pointwise("take", deps=("make",))
    edge = replace(tile_edge("ub->ub", ("aiv", "aiv")), representation=make.dtype)
    members = make, take
    declaration = dict(
        numerically_equivalent=True, region_id="pointwise_region", op_ids=("make", "take"),
        edge_routes=(edge,), backend="dsl_vec", compiler_options=dict(
            edge_bytes={edge.view_id: int(make.mte3_bytes)},
            op_stages={op.op_id: Stage(op.op_id, {"vector": op.vector_ops}) for op in members},
            semantic_work=tuple(SemanticWorkItem(op.op_id, "vector_alu", op.dtype, int(op.vector_ops), ("aiv_simd",))
                                for op in members),
        ),
    )
    take = replace(take, metadata=(*take.metadata, ("fuse_regions", (declaration,))))
    (handle, plan, _), = _replay_plans(_catalog_case(), (src, make, take), _dsl_proposals)
    full = _replay_checked(handle, plan)
    _, fused = full.applied.executable[0].ops
    assert fused.deps == (src.op_id,)
    before, after = full.step_eval.operators
    assert after.predecessors == (before.occurrence,) and before.end_ms <= after.start_ms


def test_indexed_catlass_simt_screens_and_finalizes_on_vector_rate():
    width = 32
    problem = CatlassProblem(1, width, width, "fp16", "fp16", "fp16", indices=(1,), physical_m=2)
    op = gemm_op(op_id="gemm", deps=(), requests=(), stage=0, layer=0, m=problem.m, n=problem.n, k=problem.k,
                 quant=QuantSpec("fp16"), category="dense", phase="decode",
                 metadata=(("catlass_problem", problem),))
    handle, plan, spec = next(trial for trial in _replay_plans(_catalog_case(), (op,), _catlass_proposals)
                              if trial[2].dispatch_policy == "MatmulGatherScatterSimt")
    full = _replay_checked(handle, plan)
    summary, = full.region_summaries
    assert summary.spec == spec and not summary.issued_cube_flops_by_dtype
    assert dict(summary.vector_ops_by_pipe)["aiv_simt"] == op.flops
    service = full.step_eval.execution_evidence["serviced"]
    assert sum(value for role, value in service.items() if role.startswith("vector@")) == op.flops


def test_k3_isolated_cache_key_distinguishes_same_edit_id_declarations():
    base = parse_case(_catalog_case())
    die, = base.machine.dies
    rate = 1024 * 1000
    die = replace(die, hbm_read_bytes_per_s=rate, hbm_write_bytes_per_s=rate, hbm_rw_alpha=0.0,
                  rates=tuple(dict(die.rates, hbm=rate).items()))
    case = replace(base, machine=replace(base.machine, dies=(die,)))
    make, take = _pointwise("make"), _pointwise("take", deps=("make",))
    decls = tuple(dict(
        numerically_equivalent=True, region_id="same_id", op_ids=("make", "take"), edge_routes=(edg,), backend="dsl_vec",
        compiler_options=dict(edge_bytes={edg.view_id: 8192}, l2_reservation_bytes=int(dict(die.buffers)["l2"]),
                              op_stages={op.op_id: Stage(op.op_id, {"vector": op.vector_ops}) for op in (make, take)},
                              semantic_work=tuple(SemanticWorkItem(op.op_id, "vector_alu", op.dtype, int(op.vector_ops), ("aiv_simd",))
                                                  for op in (make, take))),
    ) for edg in (replace(tile_edge("ub->ub", ("aiv", "aiv")), representation=make.dtype),
                  replace(tile_edge("gm_spill", ("aiv", "aiv")), representation=make.dtype)))
    take_d = replace(take, metadata=(*take.metadata, ("fuse_regions", decls)))
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode")), spine=_DeclaredGraphSpine(make, take_d))
    dp = dossier.handle.scenario.datapath
    props = list(_dsl_proposals((make, take_d), dp, dp.aic_count, dp.aiv_count))
    p_ub = replace(dossier.unmerge_plan, edits=(Edit("f", operation=props[0][0][0]),))
    p_gm = replace(dossier.unmerge_plan, edits=(Edit("f", operation=props[1][0][0]),))
    f_ub, f_gm = _replay_checked(dossier.handle, p_ub), _replay_checked(dossier.handle, p_gm)
    assert f_ub.region_summaries[0].dram_sector_bytes == 0
    assert f_gm.region_summaries[0].dram_sector_bytes == 16384 and f_gm.verification.l_star_ns is not None


def test_simulation_mode_fails_fast_and_modeling_mode_evaluates_formula_kernel():
    from mindie_sim.rewrite import FuseRegion, KernelBoundary, OpRef, evaluate_rewrites
    from mindie_sim.schema.operators import OpSpec

    rows, hidden = 128, 4096
    shape = (rows, hidden)
    tensor_bytes = float(rows * hidden * 2)
    norm_op = OpSpec(
        op_id="norm", family="rmsnorm", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="bf16", vector_ops=float(rows * hidden * 4),
        mte2_bytes=tensor_bytes, mte3_bytes=tensor_bytes, hbm_bytes=2.0 * tensor_bytes,
    )
    quant_op = OpSpec(
        op_id="quant", family="quantize", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        deps=("norm",), dtype="fp8",
        vector_ops=float(rows * hidden * 3),
        mte2_bytes=tensor_bytes, mte3_bytes=tensor_bytes * 0.5, hbm_bytes=1.5 * tensor_bytes,
    )
    dossier = prepare_scenario(
        ScenarioRequest(_catalog_case(), StepRequest("decode")),
        spine=_DeclaredGraphSpine(norm_op, quant_op),
    )
    refs = (OpRef("atom", "norm"), OpRef("atom", "quant"))
    fuse_edit = Edit("custom_nq", operation=FuseRegion(
        members=refs, preserve_ports=(),
        kernel_boundaries=(KernelBoundary("CustomFormulaNormQuant", refs),),
        fusion_tier="K3_dsl_pipeline",
    ))
    plan = replace(dossier.unmerge_plan, edits=(fuse_edit,))

    # Simulation Mode (default allow_unbound_formula=False) MUST fail fast on unbound fusion.
    with pytest.raises(ModelingError, match="rewrite_fusion_recipe_unbound:rewrite:custom_nq:CustomFormulaNormQuant"):
        finalize_plan(dossier.handle, plan)
    sim_batch = evaluate_rewrites(dossier.handle, (plan,))
    assert sim_batch.results[0].status == "invalid"
    assert "rewrite_fusion_recipe_unbound" in sim_batch.results[0].reason

    # Modeling Mode (allow_unbound_formula=True) uses white-box FusedPipelineRecipe on Engine.
    model_batch = evaluate_rewrites(dossier.handle, (plan,), allow_unbound_formula=True)
    cand = model_batch.results[0]
    assert cand.status == "ok" and cand.verification is not None and cand.verification.passed
    assert cand.has_unbound_formula_ops is True
    assert cand.unbound_formula_ops == ("rewrite:custom_nq",)
    assert ("rewrite:custom_nq", "unbound_formula_projection") in cand.binding_states

    baseline = finalize_plan(dossier.handle, dossier.unmerge_plan)
    fused_full = finalize_plan(dossier.handle, plan, allow_unbound_formula=True)
    assert fused_full.verification.passed
    assert fused_full.verification.l_star_ns is not None
    assert fused_full.verification.l_star_ns <= fused_full.wall_ns
    assert fused_full.wall_ns < baseline.wall_ns
    assert fused_full.has_unbound_formula_ops is True
    assert fused_full.unbound_formula_ops == ("rewrite:custom_nq",)
    assert ("rewrite:custom_nq", "unbound_formula_projection") in fused_full.binding_states


def test_oneshot_collective_transform_replays_and_passes_all_verification_gates():
    from mindie_sim.rewrite import CollectiveTransform, OpRef

    case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=2, dp=1, layers=1)
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode", tokens=4, context=64)))
    collective = next(op for op in dossier.handle.scenario.atomic_ops[0].ops if op.link_bytes > 0)
    route = dossier.handle.scenario.case.machine.links[0].name
    edit = Edit("oneshot_comm", operation=CollectiveTransform(OpRef("atom", collective.op_id), "oneshot", route))
    plan = replace(dossier.unmerge_plan, edits=(edit,))
    full = finalize_plan(dossier.handle, plan)
    assert full.verification.passed
    assert full.verification.l_star_ns is not None
    assert full.verification.l_star_ns <= full.wall_ns
    assert full.has_unbound_formula_ops is False
    row = next(item for item in full.step_eval.operators if item.occurrence.op_id == collective.op_id)
    assert row.implementation == "OneShotCollective"


def test_formula_kernel_sdk_prices_whitebox_l0_pipes_and_fp32_sram():
    from mindie_sim.optimization import (
        FormulaKernelSpec,
        FormulaTensorSpec,
        evaluate_formula_kernel,
        simulate_formula_kernel,
    )

    case = parse_case(_catalog_case())
    vec_spec = FormulaKernelSpec(
        name="custom_rmsnorm_quant",
        equations=("t1 = RMSNorm(x; gamma)", "y, s = DynamicQuant(t1)"),
        inputs=(FormulaTensorSpec("x", (64, 4096), "bf16", "input"),),
        outputs=(FormulaTensorSpec("y", (64, 4096), "fp8", "output"),),
        vector_ops_fp32=float(64 * 4096 * 7),
        row_reduction=True,
    )
    res_vec = evaluate_formula_kernel(vec_spec, case)
    assert res_vec == simulate_formula_kernel(vec_spec, case)
    assert res_vec.sram_verified is True
    assert 0 < res_vec.ub_peak_bytes <= res_vec.ub_budget_bytes
    assert res_vec.projected_us >= res_vec.l_star_us > 0.0
    assert res_vec.binding_state == "unbound_formula_projection"
    assert res_vec.has_unbound_formula_ops is True
    assert res_vec.abi_signature_id.startswith("formula:custom_rmsnorm_quant:")

    cube_spec = FormulaKernelSpec(
        name="custom_gmm_situ",
        equations=("t1 = x @ W^T", "y = SiTU(t1_gate, t1_up; beta=1.0)"),
        inputs=(FormulaTensorSpec("x", (32, 2048), "bf16", "input"),),
        weights=(FormulaTensorSpec("W", (4096, 2048), "bf16", "weight"),),
        outputs=(FormulaTensorSpec("y", (32, 2048), "bf16", "output"),),
        matrix_flops=(("bf16", float(2 * 32 * 4096 * 2048)),),
        vector_ops_fp32=float(32 * 2048 * 6),
        activation="situ",
        activation_params=(("beta", 1.0),),
    )
    res_cube = evaluate_formula_kernel(cube_spec, case)
    assert res_cube.sram_verified is True
    assert res_cube.projected_us >= res_cube.l_star_us > 0.0
    assert res_cube.l1_peak_bytes > 0 and res_cube.l0_peak_bytes > 0


def test_formula_kernel_and_oracle_reject_ub_infeasible_row_reduction():
    from mindie_sim.optimization import FormulaKernelSpec, FormulaTensorSpec, evaluate_formula_kernel
    from mindie_sim.rewrite import FuseRegion, KernelBoundary, OpRef, evaluate_rewrites
    from mindie_sim.schema.operators import OpSpec

    # 32,768-wide BF16 row reduction requires 32768 * 4 * 3 = 393,216 B FP32 UB (> 237,568 B usable on A5).
    rows, hidden = 4, 32768
    shape = (rows, hidden)
    tensor_bytes = float(rows * hidden * 2)
    a5_case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1)
    parsed_a5 = parse_case(a5_case)

    wide_spec = FormulaKernelSpec(
        name="wide_row_norm_quant",
        equations=("t1 = RMSNorm(x; gamma)", "y, s = DynamicQuant(t1)"),
        inputs=(FormulaTensorSpec("x", shape, "bf16", "input"),),
        outputs=(FormulaTensorSpec("y", shape, "fp8", "output"),),
        vector_ops_fp32=float(rows * hidden * 7),
        row_reduction=True,
    )
    eval_res = evaluate_formula_kernel(wide_spec, parsed_a5)
    assert eval_res.sram_verified is False
    assert eval_res.tile_algorithm == "ub_infeasible"
    assert eval_res.ub_peak_bytes > eval_res.ub_budget_bytes

    norm_op = OpSpec(
        op_id="wide_norm", family="rmsnorm", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="bf16", vector_ops=float(rows * hidden * 4),
        mte2_bytes=tensor_bytes, mte3_bytes=tensor_bytes, hbm_bytes=2.0 * tensor_bytes,
    )
    quant_op = OpSpec(
        op_id="wide_quant", family="quantize", phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        deps=("wide_norm",), dtype="fp8", vector_ops=float(rows * hidden * 3),
        mte2_bytes=tensor_bytes, mte3_bytes=tensor_bytes * 0.5, hbm_bytes=1.5 * tensor_bytes,
    )
    dossier = prepare_scenario(
        ScenarioRequest(a5_case, StepRequest("decode", tokens=rows, context=64)),
        spine=_DeclaredGraphSpine(norm_op, quant_op),
    )
    refs = (OpRef("atom", "wide_norm"), OpRef("atom", "wide_quant"))
    plan = replace(dossier.unmerge_plan, edits=(Edit("fuse_wide", operation=FuseRegion(
        members=refs, preserve_ports=(),
        kernel_boundaries=(KernelBoundary("WideUnboundNormQuant", refs),),
        fusion_tier="K3_dsl_pipeline",
    )),))
    batch = evaluate_rewrites(dossier.handle, (plan,), allow_unbound_formula=True)
    assert batch.results[0].status == "invalid"
    assert "MULTI_POOL_CAPACITY" in batch.results[0].reason
    with pytest.raises(ModelingError, match="MULTI_POOL_CAPACITY"):
        finalize_plan(dossier.handle, plan, allow_unbound_formula=True)


def test_formula_dag_equations_and_pipe_stage_ordering():
    from mindie_sim.optimization import FormulaKernelSpec, FusedPipelineRecipe
    from mindie_sim.prediction.program.binding import RecipeContext
    from mindie_sim.schema.operators import OpSpec

    m_dim, n_dim, k_dim = 16, 2048, 4096
    act_b = float(m_dim * k_dim * 2)
    wt_b = float(n_dim * k_dim * 2)
    out_b = float(m_dim * n_dim * 2)
    gemm_flops = float(2 * m_dim * n_dim * k_dim)

    # 1) Sibling projections sharing one external activation input MUST produce parallel equations.
    q_op = OpSpec(
        op_id="s0.l0.q_proj", family="gemm", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.l0.norm",), semantic_shape=(m_dim, n_dim, k_dim), local_shape=(m_dim, n_dim, k_dim),
        padded_shape=(m_dim, n_dim, k_dim), launch_shape=(m_dim, n_dim, k_dim),
        dtype="bf16", flops=gemm_flops, mte2_bytes=act_b + wt_b, mte3_bytes=out_b, hbm_bytes=act_b + wt_b + out_b,
        metadata=(("typed_inputs", ({"producer": "s0.l0.norm", "port": "out", "version": 0},)),),
    )
    k_op = replace(q_op, op_id="s0.l0.k_proj")
    sib_spec = FormulaKernelSpec.from_ops((q_op, k_op), name="parallel_qk")
    assert len(sib_spec.inputs) == 1 and sib_spec.inputs[0].name == "x"
    assert [o.name for o in sib_spec.outputs] == ["y0", "y1"]
    assert any(eq.startswith("y0 = x @ W_") for eq in sib_spec.equations)
    assert any(eq.startswith("y1 = x @ W_") for eq in sib_spec.equations)
    assert not any("t1" in eq for eq in sib_spec.equations)

    def _recipe_ctx(op):
        res_names = {"hbm": "hbm", "mte2": "mte2", "mte3": "mte3", "cube": "cube", "vector": "vector", "link": "link"}
        rates = {"hbm": 1.6e9, "mte2": 1.6e9, "mte3": 1.6e9, "cube": 1.0, "vector": 1.0e10, "link": 2.0e8}
        work = {"hbm": op.hbm_bytes, "mte2": op.mte2_bytes, "mte3": op.mte3_bytes}
        if op.flops > 0:
            work["cube"] = 0.05
        if op.vector_ops > 0:
            work["vector"] = op.vector_ops
        if op.link_bytes > 0:
            work["link"] = op.link_bytes
        return RecipeContext(
            op=op, work=work, resource_names=res_names, rates=rates, core_count=24,
        )

    # 2) RMSNorm -> GEMM orders AIV vector pipe strictly BEFORE Cube MMA pipe.
    norm_op = OpSpec(
        op_id="s0.l0.norm", family="rmsnorm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(m_dim, k_dim), local_shape=(m_dim, k_dim),
        padded_shape=(m_dim, k_dim), launch_shape=(m_dim, k_dim),
        dtype="bf16", vector_ops=float(m_dim * k_dim * 4),
        mte2_bytes=act_b, mte3_bytes=act_b, hbm_bytes=2.0 * act_b,
    )
    ng_spec = FormulaKernelSpec.from_ops((norm_op, q_op), name="norm_then_gemm")
    vec_st = next(s for s in ng_spec.pipe_stages if s.pipe_kind == "vector")
    cube_st = next(s for s in ng_spec.pipe_stages if s.pipe_kind == "cube")
    assert vec_st.name in cube_st.after
    assert cube_st.name not in vec_st.after
    def _tile_stages(prog):
        for st in prog.stages:
            if getattr(st, "program", None) is not None and hasattr(st.program, "tile"):
                return list(st.program.tile(0))
        return list(prog.stages)

    ng_op = replace(q_op, op_id="fused_ng", vector_ops=norm_op.vector_ops)
    ng_prog = FusedPipelineRecipe(
        pipe_stages=ng_spec.pipe_stages,
    ).lower(_recipe_ctx(ng_op))
    ng_after = {st.name: st.after for st in _tile_stages(ng_prog)}
    assert "cube_mma" in ng_after and any("aiv" in pred for pred in ng_after["cube_mma"])

    # 3) AllGather -> GEMM orders Link pipe strictly BEFORE Cube MMA pipe.
    ag_op = OpSpec(
        op_id="s0.l0.allgather", family="allgather", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(m_dim, k_dim), local_shape=(m_dim, k_dim),
        padded_shape=(m_dim, k_dim), launch_shape=(m_dim, k_dim),
        dtype="bf16", link_bytes=act_b, link_scope="tp",
        mte2_bytes=act_b, mte3_bytes=act_b, hbm_bytes=2.0 * act_b,
    )
    q_after_ag = replace(q_op, deps=("s0.l0.allgather",), metadata=())
    ag_spec = FormulaKernelSpec.from_ops((ag_op, q_after_ag), name="ag_then_gemm")
    link_st = next(s for s in ag_spec.pipe_stages if s.pipe_kind == "link")
    cube_st2 = next(s for s in ag_spec.pipe_stages if s.pipe_kind == "cube")
    assert link_st.name in cube_st2.after
    assert cube_st2.name not in link_st.after
    ag_fused_op = replace(q_after_ag, op_id="fused_ag_mm", link_bytes=act_b, link_scope="tp")
    ag_prog = FusedPipelineRecipe(
        pipe_stages=ag_spec.pipe_stages,
    ).lower(_recipe_ctx(ag_fused_op))
    ag_after = {st.name: st.after for st in _tile_stages(ag_prog)}
    assert "cube_mma" in ag_after and "link_pipe" in ag_after["cube_mma"]


def test_oracle_cache_separates_disabled_primitive_table_and_formula_mode_digests():
    from mindie_sim.rewrite import FuseRegion, KernelBoundary, OpRef
    from mindie_sim.rewrite._oracle_prefetch import primitive_table_digest_token

    assert primitive_table_digest_token(False) != primitive_table_digest_token(None)

    cache = OracleCache(max_entries=8)
    raw_case = compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, layers=1)
    step = StepRequest("decode", tokens=1, context=64)

    d_auto = prepare_scenario(ScenarioRequest(raw_case, step), cache=cache, primitive_table=None)
    d_disabled = prepare_scenario(ScenarioRequest(raw_case, step), cache=cache, primitive_table=False)
    assert d_auto.scenario_digest != d_disabled.scenario_digest
    assert d_auto.handle is not d_disabled.handle
    assert cache.get(d_auto.scenario_digest) is d_auto.handle
    assert cache.get(d_disabled.scenario_digest) is d_disabled.handle

    # Strict vs formula-enabled handles MUST NOT share a scenario_digest or overwrite one another.
    d_strict = prepare_scenario(
        ScenarioRequest(raw_case, step), cache=cache, primitive_table=False, allow_unbound_formula=False,
    )
    d_formula = prepare_scenario(
        ScenarioRequest(raw_case, step), cache=cache, primitive_table=False, allow_unbound_formula=True,
    )
    assert d_strict.scenario_digest != d_formula.scenario_digest
    strict_handle = cache.get(d_strict.scenario_digest)
    formula_handle = cache.get(d_formula.scenario_digest)
    assert strict_handle is not None and strict_handle.allow_unbound_formula is False
    assert formula_handle is not None and formula_handle.allow_unbound_formula is True

    ops = strict_handle.scenario.atomic_ops[0].ops
    pair = next(
        (ops[i].op_id, ops[i + 1].op_id)
        for i in range(len(ops) - 1)
        if ops[i].op_id in ops[i + 1].deps and ops[i].stage == ops[i + 1].stage
    )
    refs = (OpRef("atom", pair[0]), OpRef("atom", pair[1]))
    plan = replace(
        d_strict.unmerge_plan,
        edits=(Edit("u_mode", operation=FuseRegion(
            members=refs, preserve_ports=(),
            kernel_boundaries=(KernelBoundary("UnboundCacheModeCheck", refs),),
            fusion_tier="K3_dsl_pipeline",
        )),),
    )
    with pytest.raises(ModelingError, match="rewrite_fusion_recipe_unbound"):
        finalize_plan(strict_handle, plan)


def test_formula_sram_sizes_internal_atomic_ops_and_rejects_unclipped_l0c_overflow():
    from mindie_sim.optimization import (
        FormulaKernelSpec,
        TargetOpNode,
        evaluate_formula_kernel,
        synthesize_kernel_spec,
    )
    from mindie_sim.prediction.results import OpOccurrence
    from mindie_sim.rewrite.algebra import classify_op_traits
    from mindie_sim.schema.datapath import datapath_for_die
    from mindie_sim.schema.operators import OpSpec

    case = parse_case(compose_case("qwen3-32b", REPRESENTATIVE_SKUS["a3"], tp=1, dp=1, layers=1))
    die = case.machine.dies[0]

    # Counterexample A: 4 -> 4096 -> RMSNorm -> 4 with UB=24 KiB (8 KiB usable after 16 KiB SIMT carveout).
    tight_ub_buffers = tuple(("ub", 24 * 1024) if k == "ub" else (k, v) for k, v in die.buffers)
    tight_ub_die = replace(die, buffers=tight_ub_buffers)
    op_up = OpSpec(
        op_id="expand", family="gemm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(1, 4096, 4), local_shape=(1, 4096, 4), padded_shape=(1, 4096, 4), launch_shape=(1, 4096, 4),
        dtype="bf16", flops=float(2 * 1 * 4096 * 4), mte2_bytes=32776.0, mte3_bytes=8192.0, hbm_bytes=40968.0,
    )
    op_norm = OpSpec(
        op_id="inner_norm", family="rmsnorm", phase="decode", stage=0, layer=0, requests=(),
        deps=("expand",), semantic_shape=(1, 4096), local_shape=(1, 4096), padded_shape=(1, 4096), launch_shape=(1, 4096),
        dtype="bf16", vector_ops=float(4096 * 4), mte2_bytes=8192.0, mte3_bytes=8192.0, hbm_bytes=16384.0,
    )
    op_dn = OpSpec(
        op_id="shrink", family="gemm", phase="decode", stage=0, layer=0, requests=(),
        deps=("inner_norm",), semantic_shape=(1, 4, 4096), local_shape=(1, 4, 4096),
        padded_shape=(1, 4, 4096), launch_shape=(1, 4, 4096),
        dtype="bf16", flops=float(2 * 1 * 4 * 4096), mte2_bytes=40960.0, mte3_bytes=8.0, hbm_bytes=40968.0,
    )
    narrow_io_spec = FormulaKernelSpec.from_ops((op_up, op_norm, op_dn), name="narrow_io_wide_norm")
    assert narrow_io_spec.inputs[0].shape == (1, 4)
    assert narrow_io_spec.outputs[0].shape == (1, 4)
    res_ub = evaluate_formula_kernel(narrow_io_spec, tight_ub_die)
    assert res_ub.sram_verified is False
    assert res_ub.ub_budget_bytes == 8 * 1024
    assert res_ub.ub_peak_bytes >= 16 * 1024

    # Counterexample B: 64 KiB L0C accumulator demand MUST NOT be clipped to a 32 KiB L0C budget.
    tight_l0_buffers = tuple(("l0c", 32 * 1024) if k == "l0c" else (k, v) for k, v in die.buffers)
    tight_l0_die = replace(die, buffers=tight_l0_buffers)
    tight_l0_dp = datapath_for_die(tight_l0_die)
    cube_only = FormulaKernelSpec.from_ops((op_up,), name="l0c_overflow_cube")
    res_l0 = evaluate_formula_kernel(cube_only, tight_l0_die)
    assert res_l0.sram_verified is False
    assert res_l0.l0_budget_bytes == 32 * 1024
    assert res_l0.l0_peak_bytes == 64 * 1024

    occ = OpOccurrence(replica=0, stage=0, rank=0, op_id=op_up.op_id)
    node_up = TargetOpNode(
        0, occ, op_up.op_id, "arch0", 1, op_up.family, "gemm", "main", 0, "AIC", op_up.local_shape, "bf16", (),
        0.0, 10.0, 10.0, 10.0, False, "", op_up.flops, 0.0, op_up.mte2_bytes, op_up.mte3_bytes, op_up.hbm_bytes,
        op_up.mte2_bytes, op_up.mte3_bytes, 0.0, 0.0, "cube_bound", "Cube", 2.0, 8.0, 5.0, 0.2,
        ("expand",), ("expand",), "GEMM", "gemm", "CANN", "", False, "", "expand", spec=op_up,
    )
    kspec = synthesize_kernel_spec(
        spec_id="l0_check", kernel_name="l0_check", abi_tier="K2_epilogue", suggested_backend="CANN",
        status_tag="test", opt_pillar="pillar1_fusion", members=(node_up,),
        member_traits=(classify_op_traits(op_up),), datapath=tight_l0_dp,
        elided_hbm_bytes=0, l_star_us=2.0, before_us=10.0, target_us=8.0,
    )
    assert kspec.sram_verified is False
    assert kspec.l0_peak_bytes == 64 * 1024 and kspec.l0_budget_bytes == 32 * 1024


def test_formula_dag_preserves_duplicate_operands_order_and_independent_cube_stages():
    from mindie_sim.optimization import FormulaKernelSpec, FusedPipelineRecipe
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.program.binding import RecipeContext
    from mindie_sim.schema.operators import OpSpec

    # 1) Add(x, x) keeps duplicate operands without inventing an unbound symbol 'r'.
    add_dup = OpSpec(
        op_id="s0.l0.add_self", family="add", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.l0.ext",), semantic_shape=(16, 128), local_shape=(16, 128),
        padded_shape=(16, 128), launch_shape=(16, 128), dtype="bf16",
        vector_ops=2048.0, mte2_bytes=4096.0, mte3_bytes=4096.0, hbm_bytes=8192.0,
        metadata=(("typed_inputs", (
            {"producer": "s0.l0.ext", "port": "out", "version": 0},
            {"producer": "s0.l0.ext", "port": "out", "version": 0},
        )),),
    )
    dup_spec = FormulaKernelSpec.from_ops((add_dup,), name="add_self")
    assert dup_spec.equations == ("y = x + x [UB FP32]",)

    # 2) Operand ordering preserves external-before-internal operand order.
    inner_act = OpSpec(
        op_id="s0.l0.inner", family="silu", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.l0.ext",), semantic_shape=(16, 128), local_shape=(16, 128),
        padded_shape=(16, 128), launch_shape=(16, 128), dtype="bf16",
        vector_ops=2048.0, mte2_bytes=4096.0, mte3_bytes=4096.0, hbm_bytes=8192.0,
    )
    add_ord = OpSpec(
        op_id="s0.l0.add_ext_first", family="add", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.l0.ext", "s0.l0.inner"), semantic_shape=(16, 128), local_shape=(16, 128),
        padded_shape=(16, 128), launch_shape=(16, 128), dtype="bf16",
        vector_ops=2048.0, mte2_bytes=8192.0, mte3_bytes=4096.0, hbm_bytes=12288.0,
        metadata=(("typed_inputs", (
            {"producer": "s0.l0.ext", "port": "out", "version": 0},
            {"producer": "s0.l0.inner", "port": "out", "version": 0},
        )),),
    )
    ord_spec = FormulaKernelSpec.from_ops((inner_act, add_ord), name="ext_before_internal")
    assert ord_spec.equations[-1] == "y = x + t1 [UB FP32]"

    # 3) Independent Cube A (1 us) -> Vec V (10 us) alongside Cube B (9 us) schedules in 11 us (not 20 us).
    cube_a = OpSpec(
        op_id="s0.l0.cube_a", family="gemm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(1, 16, 16), local_shape=(1, 16, 16), padded_shape=(1, 16, 16), launch_shape=(1, 16, 16),
        dtype="bf16", flops=1.0e6, mte2_bytes=0.0, mte3_bytes=0.0, hbm_bytes=0.0,
    )
    cube_b = OpSpec(
        op_id="s0.l0.cube_b", family="gemm", phase="decode", stage=0, layer=0, requests=(),
        semantic_shape=(1, 144, 16), local_shape=(1, 144, 16), padded_shape=(1, 144, 16), launch_shape=(1, 144, 16),
        dtype="bf16", flops=9.0e6, mte2_bytes=0.0, mte3_bytes=0.0, hbm_bytes=0.0,
    )
    vec_v = OpSpec(
        op_id="s0.l0.vec_v", family="silu", phase="decode", stage=0, layer=0, requests=(),
        deps=("s0.l0.cube_a",), semantic_shape=(1, 16), local_shape=(1, 16),
        padded_shape=(1, 16), launch_shape=(1, 16), dtype="bf16",
        vector_ops=10.0e6, mte2_bytes=0.0, mte3_bytes=0.0, hbm_bytes=0.0,
    )
    dag_spec = FormulaKernelSpec.from_ops((cube_a, cube_b, vec_v), name="indep_cube_vec")
    cube_stages = [s for s in dag_spec.pipe_stages if s.pipe_kind == "cube"]
    vec_stage = next(s for s in dag_spec.pipe_stages if s.pipe_kind == "vector")
    assert len(cube_stages) == 2
    assert len(vec_stage.after) == 1
    dep_cube = next(s for s in cube_stages if s.name == vec_stage.after[0])
    indep_cube = next(s for s in cube_stages if s.name != vec_stage.after[0])
    assert dep_cube.share == pytest.approx(0.1)
    assert indep_cube.share == pytest.approx(0.9)

    fused_op = replace(cube_a, op_id="fused_abv", flops=10.0e6, vector_ops=10.0e6)
    ctx = RecipeContext(
        op=fused_op,
        work={"cube": 0.010, "vector": 0.010},
        resource_names={"cube": "cube", "vector": "vector"},
        rates={"cube": 1.0, "vector": 1.0},
        core_count=1,
    )
    prog = FusedPipelineRecipe(slots=1, tile_count=1, pipe_stages=dag_spec.pipe_stages).lower(ctx)
    eng = Engine([Resource("cube", 1.0, "ms", "fifo", 1), Resource("vector", 1.0, "ms", "fifo", 1)], backend="python")
    ready = eng.submit("ready")
    done = prog.start(eng, "abv", ready)
    eng.advance(until=(done,))
    assert done.time * 1000.0 == pytest.approx(11.0)
