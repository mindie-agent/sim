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
from mindie_sim.prediction.cost import CostLedger, CostModel
from mindie_sim.prediction.efficiency.analysis import scale_case_hardware
from mindie_sim.prediction.evaluation import EvaluationContext, ExecutionBindings
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.primitives import PrimitiveStore
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


def _replay_plans(case, ops, proposals):
    dossier = prepare_scenario(ScenarioRequest(case, StepRequest("decode")), spine=_DeclaredGraphSpine(*ops))
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
    read_bytes = 2 * (width * width + width)
    rate = handle.scenario.datapath.hbm_read_bytes_per_s
    assert read_ns == inward_floor_ns(Fraction(read_bytes * NS_PER_SECOND, int(rate)))
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
    handle, plan, spec = next(_replay_plans(case, (op,), _catlass_proposals))
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
        assert full.step_eval.span_ms == pytest.approx(6.1875, abs=2e-5)
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
