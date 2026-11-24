"""Five-level ladder, unified single-engine region search, and 4-stage Engine receipts.

[Role] Orchestrate framework + unified algebraic/CATLASS/DSL region searches on
  one immutable ``PrimitiveTable``, reconcile exact Stream-0 wall attribution
  via ``exposed_observer.attribute_stream0_intervals``, certify primal-dual
  lower bounds, and produce canonical 4-stage ``EngineStepReceipt`` records.
[Invariants] Exact integer telescopes (0 ns error), disjoint Stream-0 wall-time
  ownership, bidirectional Equation 7 verification on all accepted ``FuseRegion``
  edits, and monotonic stage progression under ``monotonic_optimization``.
[Timeouts & Locks] Search is bounded by oracle budgets and caller trial limits;
  no module locks or external I/O.
[Side-Effect Policy] Read-only evaluation over immutable scenarios and plans.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.results import (
    CaseInput,
    ComparisonPolicy,
    EngineStepReceipt,
    OpOccurrence,
    ReplayOptions,
    StageCommitStatus,
    StageId,
    StageReceipt,
    StepEvaluation,
    StepRequest,
)
from mindie_sim.prediction.runtime.exposed_observer import (
    ObservedInterval,
    attribute_stream0_intervals,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.stack import StackTrack

from .contracts import CapacityTable, CatlassKernelSpec, DslKernelSpec, FuseRegionSpec
from .dual_bound import (
    NS_PER_MS,
    DualSubgradientReport,
    OptimalityCertificate,
    _atomic_ops,
    _machine,
    _unique_ops,
    build_dual_inputs_from_scenario,
    certify_lower_bound,
)
from .framework_solver import (
    DEFAULT_MAX_TRIALS,
    CandidateSearchStats,
    FrameworkSolveResult,
    _TrialSearch,
    _accepted,
    _field,
    _interleave,
    _resolved,
    _scenario_lowering_plan,
    _search_bases,
    _seed_plans,
    _step,
    _stream_candidates,
    _stream_plan,
    _verification,
    _wall,
    _with_operations,
    solve_framework,
)

if TYPE_CHECKING:
    from mindie_sim.rewrite.algebra import SemanticAuditTrigger
    from mindie_sim.rewrite.oracle import FullPlanEvaluation, ScenarioDossier
    from mindie_sim.rewrite.schema import RewritePlan
    from mindie_sim.rewrite.verify import ShippingRegression, VerificationReceipt

FIVE_LEVEL_REPORT_SCHEMA = "mindie-sim.five-level-report/1"
SHIPPING_REGRESSION_TOLERANCE_NS = 1000
DEFAULT_LADDER_TRIALS = 16
STEP_WAIT_REGION = "step:wait"
SHIPPING_OVERLAY = "shipping_overlay"
UNIFIED_REGION_STAGES = ("R1_algebraic_and_dsl", "R2_catlass", "R3_stream_retune")
_MAIN_STREAM_INDEX = 0
_AUX_STREAM_INDEX = 1
_COMM_STREAM_INDEX = 2

__all__ = (
    "FIVE_LEVEL_REPORT_SCHEMA",
    "UNIFIED_REGION_STAGES",
    "FiveLevelLadderMetrics",
    "RegionWallAttribution",
    "OrderedTelescopeStep",
    "LooMarginalImpact",
    "FiveLevelWaterfall",
    "FiveLevelReport",
    "UnifiedRegionSolveResult",
    "partition_step_wall_ns",
    "solve_unified_regions",
    "solve_five_level_ladder",
    "solve_target_step_receipt",
)


@dataclass(frozen=True, slots=True)
class FiveLevelLadderMetrics:
    l_star_lean_ns: int
    l_star_hw_ns: int | None
    a_star_dsl_ns: int
    a_star_framework_ns: int
    m_mainline_ns: int
    u_unmerge_ns: int
    delta_shipping_ns: int
    delta_framework_ns: int
    delta_operator_dsl_ns: int
    delta_optimality_gap_ns: int
    certified_gap_pct: float | None


@dataclass(frozen=True, slots=True)
class RegionWallAttribution:
    region_id: str
    op_ids: tuple[str, ...]
    u_wall_ns: int
    m_wall_ns: int
    fw_wall_ns: int
    dsl_wall_ns: int


@dataclass(frozen=True, slots=True)
class OrderedTelescopeStep:
    stage_id: str
    before_ns: int
    after_ns: int
    delta_ns: int
    plan_sha256: str


@dataclass(frozen=True, slots=True)
class LooMarginalImpact:
    edit_id: str
    opcode: str
    marginal_delta_ns: int


@dataclass(frozen=True, slots=True)
class FiveLevelWaterfall:
    ordered_telescope: tuple[OrderedTelescopeStep, ...]
    loo_impacts: tuple[LooMarginalImpact, ...]
    by_region: tuple[RegionWallAttribution, ...]
    reconciliation_ns: tuple[int, int, int, int]
    shipping_regression: ShippingRegression | None


@dataclass(frozen=True, slots=True)
class UnifiedRegionSolveResult:
    """Single-engine unified region search result (algebraic + catalog + CATLASS + stream)."""

    best_plan: RewritePlan
    best_eval: FullPlanEvaluation
    best_wall_ns: int
    catlass_specs: tuple[CatlassKernelSpec, ...]
    dsl_specs: tuple[DslKernelSpec, ...]
    fuse_specs: tuple[FuseRegionSpec, ...]
    stage_history: tuple[tuple[str, int, str], ...]
    tried_candidates: int
    search_stats: CandidateSearchStats = CandidateSearchStats()
    audit_triggers: tuple[SemanticAuditTrigger, ...] = ()


@dataclass(frozen=True, slots=True)
class FiveLevelReport:
    schema: Literal["mindie-sim.five-level-report/1"]
    scenario_digest: str
    ladder: FiveLevelLadderMetrics
    dual_certificate: OptimalityCertificate
    dual_subgradient: DualSubgradientReport
    waterfall: FiveLevelWaterfall
    framework_plan: RewritePlan
    dsl_plan: RewritePlan
    catlass_specs: tuple[CatlassKernelSpec, ...]
    dsl_specs: tuple[DslKernelSpec, ...]
    fuse_specs: tuple[FuseRegionSpec, ...]
    unmerge_eval: StepEvaluation
    mainline_eval: StepEvaluation
    framework_eval: StepEvaluation
    dsl_eval: StepEvaluation
    verification: VerificationReceipt
    search_stats: CandidateSearchStats = CandidateSearchStats()
    primitive_coverage_gaps: tuple[str, ...] = ()


def _occurrence_key(occurrence: OpOccurrence) -> tuple[int, int, int, str]:
    rank = -1 if occurrence.rank is None else occurrence.rank
    return occurrence.replica, occurrence.stage, rank, occurrence.op_id


def _occurrence_label(occurrence: OpOccurrence) -> str:
    replica, stage, rank, op_id = _occurrence_key(occurrence)
    return f"r{replica}.p{stage}.d{rank}:{op_id}"


def _stream_index_for_name(stream_name: str) -> int:
    low = str(stream_name or "main").strip().lower()
    if low == "main":
        return _MAIN_STREAM_INDEX
    if "comm" in low or "hccl" in low:
        return _COMM_STREAM_INDEX
    return _AUX_STREAM_INDEX


def _partition(eval_result: StepEvaluation) -> tuple[tuple[OpOccurrence | None, int], ...]:
    """Partition ``eval_result`` wall time via conserved ``attribute_stream0_intervals``."""
    wall = round(eval_result.span_ms * NS_PER_MS)
    if wall < 0:
        raise ValueError("step span must be nonnegative")
    if wall == 0:
        return ()

    occ_by_id: dict[str, OpOccurrence] = {}
    id_by_occ: dict[OpOccurrence, str] = {}
    intervals: list[ObservedInterval] = []
    rows_indexed: list[tuple[str, Any]] = []
    for idx, row in enumerate(eval_result.operators):
        if row.start_ms is None or row.end_ms is None:
            continue
        start_ns = min(wall, max(0, round(row.start_ms * NS_PER_MS)))
        end_ns = min(wall, max(start_ns, round(row.end_ms * NS_PER_MS)))
        key = str(idx)
        occ_by_id[key] = row.occurrence
        id_by_occ.setdefault(row.occurrence, key)
        rows_indexed.append((key, row))
        stream_name = str(row.stream or "main")
        intervals.append(ObservedInterval(
            occurrence_id=key,
            stream=stream_name,
            stream_index=_stream_index_for_name(stream_name),
            start_ns=start_ns,
            end_ns=end_ns,
            ordinal=int(row.ordinal),
            label=row.occurrence.op_id,
            replica=int(row.occurrence.replica),
        ))

    causal_preds: dict[str, tuple[str, ...]] = {}
    for key, row in rows_indexed:
        preds = tuple(id_by_occ[pred] for pred in row.predecessors if pred in id_by_occ)
        if preds:
            causal_preds[key] = preds

    exposure = attribute_stream0_intervals(
        intervals,
        start_ns=0,
        terminal_ns=wall,
        causal_preds=causal_preds or None,
    )
    out: list[tuple[OpOccurrence | None, int]] = []
    for key, duration in exposure.exposed_ns_by_op.items():
        if duration > 0:
            out.append((occ_by_id[key], int(duration)))
    if exposure.idle_or_wait_ns > 0:
        out.append((None, int(exposure.idle_or_wait_ns)))
    return tuple(out)


def partition_step_wall_ns(eval_result: StepEvaluation) -> tuple[tuple[str, int], ...]:
    """Attribute each interval once along the conserved Stream 0 exposure timeline."""
    wall_by_op: dict[str, int] = defaultdict(int)
    for owner, duration in _partition(eval_result):
        wall_by_op[STEP_WAIT_REGION if owner is None else owner.op_id] += duration
    return tuple(wall_by_op.items())


def _by_region(evaluations):
    parents = {}

    def root(key):
        parents.setdefault(key, key)
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(a, b):
        a, b = root(a), root(b)
        if a != b:
            parents[max(a, b)] = min(a, b)

    for evaluation in evaluations:
        for row in evaluation.operators:
            root(_occurrence_key(row.occurrence))
        for pipeline in evaluation.pipelines:
            owner = _occurrence_key(pipeline.owner)
            atoms = (*pipeline.member_op_ids, *pipeline.absorbed_source_op_ids)
            for op_id in atoms:
                join(owner, (*owner[:-1], op_id))

    members = defaultdict(set)
    for key in parents:
        members[root(key)].add(key[-1])

    partitions = []
    for evaluation in evaluations:
        walls = defaultdict(int)
        for owner, duration in _partition(evaluation):
            region = None if owner is None else root(_occurrence_key(owner))
            walls[region] += duration
        partitions.append(walls)
    regions = [
        RegionWallAttribution(
            f"r{replica}.p{stage}.d{rank}:{op}", tuple(sorted(names)),
            *(p[replica, stage, rank, op] for p in partitions),
        )
        for (replica, stage, rank, op), names in sorted(members.items())
    ]
    if any(None in p for p in partitions):
        regions.append(RegionWallAttribution(STEP_WAIT_REGION, (), *(p[None] for p in partitions)))
    return tuple(regions)


def _ordered_telescope(mainline_ns, framework, dsl):
    before = mainline_ns
    ordered = []
    for stage, after, digest in (*framework.stage_history, *dsl.stage_history):
        ordered.append(OrderedTelescopeStep(stage, before, after, before - after, digest))
        before = after
    assert before == dsl.best_wall_ns
    assert sum(step.delta_ns for step in ordered) == mainline_ns - dsl.best_wall_ns
    return tuple(ordered)


def _shipping_regression(verification, u, m):
    from mindie_sim.rewrite.verify import ShippingRegression

    delta = m - u
    if delta <= SHIPPING_REGRESSION_TOLERANCE_NS:
        return None
    regression = getattr(verification, "shipping_regression", None)
    if regression is not None and regression.loo_culprits:
        return regression
    culprits = ((SHIPPING_OVERLAY, -delta),)
    if regression is not None:
        return replace(regression, loo_culprits=culprits)
    return ShippingRegression(
        delta_ns=delta, loo_culprits=culprits, baseline_u_ns=u, baseline_m_ns=m,
    )


def _shipping_overlay_loo(search, dossier, plan, wall, u, m):
    unmerge, mainline = _seed_plans(dossier)
    if (
        m == u
        or plan.base_graph_sha256 != mainline.base_graph_sha256
        or plan.base_graph_sha256 == unmerge.base_graph_sha256
    ):
        return ()
    rebased = replace(
        plan,
        base_graph_sha256=unmerge.base_graph_sha256,
        framing_sha256=unmerge.framing_sha256,
    )
    if not rebased.edits:
        without = u
    else:
        result = search.try_finalize(rebased)
        if result is None or not _accepted(result):
            return ()
        without = _wall(result)
    return (LooMarginalImpact(SHIPPING_OVERLAY, "ShippingOverlay", without - wall),)


def _algebraic_proposals_for_scenario(
    scenario,
    ops,
    path,
    aic: int,
    aiv: int,
    *,
    search: _TrialSearch,
    allow_target_synthesis: bool,
):
    from mindie_sim.rewrite.algebra import (
        DEFAULT_SHIPPING_STACK_PASSES,
        discover_algebraic_rewrites_for_ops,
    )
    from mindie_sim.rewrite.schema import FuseRegion
    from mindie_sim.templates.contracts import FormulaKernelSpec
    from mindie_sim.templates.fuse_region import solve_formula_sram

    lowering_plan = _scenario_lowering_plan(scenario)
    track_spec = lowering_plan.track_spec if lowering_plan is not None else None
    manifest_bindings = track_spec.active_bindings if track_spec is not None else ()
    eff_track: StackTrack = (
        track_spec.track if track_spec is not None
        else (lowering_plan.track if lowering_plan is not None else ("target" if allow_target_synthesis else "shipping"))
    )
    if eff_track == "eager":
        return (), ()
    can_synth = allow_target_synthesis and (track_spec is None or track_spec.allows_target_synthesis)
    enabled_passes = (
        frozenset(lowering_plan.lowering_passes)
        if (lowering_plan is not None and lowering_plan.lowering_passes)
        else DEFAULT_SHIPPING_STACK_PASSES
    )
    machine = _machine(scenario)
    arch_family = str(getattr(machine, "arch_family", "") or "")
    sku_id = str(getattr(machine, "sku", "") or getattr(machine, "name", "") or "")
    case = getattr(scenario, "case", None)
    parallel = getattr(getattr(case, "runtime", None), "parallel", None)
    tp_size = int(parallel.tp) if parallel is not None else None
    ep_size = int(parallel.ep) if parallel is not None else None

    discovered = discover_algebraic_rewrites_for_ops(
        ops,
        datapath=path,
        include_abi_extensions=can_synth,
        arch_family=arch_family,
        sku_id=sku_id,
        tp_size=tp_size,
        ep_size=ep_size,
        pr_track=eff_track,
        track_spec=track_spec,
        manifest_bindings=manifest_bindings,
        enabled_stack_passes=enabled_passes,
    )
    ops_by_id = {op.op_id: op for op in ops}
    proposals: list[tuple[Any, ...]] = []
    accepted_ops: list[FuseRegion] = []
    for edit, cand in zip(discovered.plan.edits, discovered.candidates):
        region = edit.operation
        if not isinstance(region, FuseRegion):
            continue
        if region.fusion_tier in {"K2_epilogue", "K3_dsl_pipeline"}:
            if not can_synth:
                continue
            member_ops = tuple(ops_by_id[mid] for mid in cand.member_ids if mid in ops_by_id)
            if member_ops:
                try:
                    formula = FormulaKernelSpec.from_ops(member_ops, name=cand.abi_id)
                    *_, sram_ok = solve_formula_sram(formula, path, cube_budget=aic, vec_budget=aiv)
                    if not sram_ok:
                        continue
                except ModelingError:
                    continue
        if cand.abi_verdict is not None and cand.abi_verdict.status in {"admitted_shipping", "admitted_proposed_pr"}:
            search.register_verdict(cand.abi_id, cand.member_ids, cand.abi_verdict)
        proposals.append((region,))
        accepted_ops.append(region)
    if len(accepted_ops) > 1:
        proposals.insert(0, tuple(accepted_ops))
    return tuple(proposals), discovered.semantic_audit_triggers


def solve_unified_regions(
    dossier: ScenarioDossier,
    fw_result: FrameworkSolveResult,
    *,
    dual_report: DualSubgradientReport | None = None,
    dual_program=None,
    event_dag=None,
    dual_certificate=None,
    max_trials: int = DEFAULT_MAX_TRIALS,
    allow_target_synthesis: bool = True,
) -> UnifiedRegionSolveResult:
    """Unified single-engine region search via ``rewrite.{algebra,catalog_replay,oracle}``."""
    del dual_report
    from mindie_sim.rewrite.catalog_replay import _catlass_proposals, _dsl_proposals, _matrix_problem
    from mindie_sim.rewrite.schema import OpRef

    search = _TrialSearch(
        dossier,
        max_trials,
        dual_program=dual_program,
        event_dag=event_dag,
        dual_certificate=dual_certificate,
    )
    search.seed(fw_result.best_plan, fw_result.best_eval)
    all_triggers: list[Any] = []

    def candidates(base, scenario, index: int, stage: str):
        ops = _unique_ops(op for _, op in _atomic_ops(scenario))
        paths = tuple(datapath_for_die(device) for device in getattr(_machine(scenario), "dies", ()))
        if index == len(UNIFIED_REGION_STAGES) - 1:
            _, evaluation = search.frontier.get(base.base_graph_sha256, (base, None))
            applied = getattr(evaluation, "applied_plan", getattr(evaluation, "applied", None))
            placed_ops = _unique_ops(applied.executable) if applied is not None else ops
            atomic_ids = {op.op_id for op in ops}
            handles = ()
            if applied is not None:
                handles = tuple(
                    (op_id, OpRef("result_of", edit_id))
                    for edit_id, output in applied.handles
                    for op_id in output
                    if len(output) == 1 and op_id not in atomic_ids
                )
            stream_iter = _stream_candidates(scenario, placed_ops, handles)
            first = next(stream_iter, None)
            search.note_eligible(1 if first is not None else 0)
            if first is not None:
                yield _stream_plan(base, stage, first)
                for operation in stream_iter:
                    yield _stream_plan(base, stage, operation)
            return
        if not paths:
            return
        path = paths[0]
        aic, aiv = min(p.aic_count for p in paths), min(p.aiv_count for p in paths)
        if index == 0:
            dsl_props = [operations for operations, _ in _dsl_proposals(ops, path, aic, aiv)]
            alg_props, triggers = _algebraic_proposals_for_scenario(
                scenario, ops, path, aic, aiv,
                search=search, allow_target_synthesis=allow_target_synthesis,
            )
            all_triggers.extend(triggers)
            combined = (*dsl_props, *alg_props)
            search.note_eligible(len(combined))
            for operations in combined:
                yield _with_operations(base, stage, operations)
            return

        bottlenecks = dict(getattr(dossier, "bottleneck_ops", ()))
        sites = sorted(ops, key=lambda op: -bottlenecks.get(op.op_id, 0))
        eligible_k2 = [op for op in sites if _matrix_problem(op, dict(op.metadata)) is not None]
        search.note_eligible(len(eligible_k2))
        for operations, _ in _catlass_proposals(eligible_k2, path, aic, aiv):
            yield _with_operations(base, stage, operations)

    for index, stage in enumerate(UNIFIED_REGION_STAGES):
        limit = search.remaining // (len(UNIFIED_REGION_STAGES) - index)
        bases = tuple(_search_bases(dossier, search))
        search.stage(
            stage,
            _interleave(candidates(base, scenario, index, stage) for base, scenario in bases),
            limit,
        )

    specs = tuple(dict.fromkeys(summary.spec for summary in getattr(search.best_eval, "region_summaries", ())))
    return UnifiedRegionSolveResult(
        best_plan=search.best_plan,
        best_eval=search.best_eval,
        best_wall_ns=_wall(search.best_eval),
        catlass_specs=tuple(s for s in specs if isinstance(s, CatlassKernelSpec)),
        dsl_specs=tuple(s for s in specs if isinstance(s, DslKernelSpec)),
        fuse_specs=tuple(s for s in specs if isinstance(s, FuseRegionSpec)),
        stage_history=tuple(search.history),
        tried_candidates=search.tried,
        search_stats=search.stats(),
        audit_triggers=tuple(all_triggers),
    )


def solve_five_level_ladder(
    case,
    request: StepRequest,
    *,
    spine=None,
    replay: ReplayOptions = ReplayOptions(),
    primitives=None,
    primitive_table: CapacityTable | None = None,
    stack_id: str = "cann93_va_main",
    max_trials: int = DEFAULT_LADDER_TRIALS,
    allow_unbound_formula: bool = False,
) -> FiveLevelReport:
    from mindie_sim.rewrite.oracle import EvalBudget, ScenarioRequest, _baseline, prepare_scenario
    from mindie_sim.rewrite.verify import SoundnessInversionError

    scenario_request = ScenarioRequest(case, request, stack_id, replay, primitive_table=primitive_table)
    dossier = prepare_scenario(
        scenario_request,
        budget=EvalBudget(deadline_mono=replay.deadline_mono),
        spine=spine,
        primitives=primitives,
        primitive_table=primitive_table,
        allow_unbound_formula=bool(allow_unbound_formula),
    )
    scenario = _resolved(dossier)
    program, dag, names = build_dual_inputs_from_scenario(scenario)
    _, pricing = certify_lower_bound(program, dag)
    prices_by_index = dict(zip(pricing.resource_names, names))
    pricing = replace(
        pricing,
        resource_names=names,
        tight_resources=tuple(prices_by_index[key] for key in pricing.tight_resources),
    )
    framework = solve_framework(dossier, dual_report=pricing, max_trials=max_trials)
    dsl = solve_unified_regions(
        dossier,
        framework,
        dual_report=pricing,
        max_trials=max_trials,
        allow_target_synthesis=True,
    )

    resolved = _baseline(dossier.handle, dsl.best_plan).scenario
    binding = dict(applied=dsl.best_eval.applied, region_summaries=dsl.best_eval.region_summaries)
    program, dag, names = build_dual_inputs_from_scenario(resolved, **binding)
    certificate, subgradient = certify_lower_bound(program, dag)
    names_by_index = dict(zip(subgradient.resource_names, names))
    subgradient = replace(
        subgradient,
        resource_names=names,
        tight_resources=tuple(names_by_index[name] for name in subgradient.tight_resources),
    )
    lower_bound = certificate.l_star_lean_ns
    hw_bound = None
    if primitive_table not in (None, False):
        hw_prog, hw_dag, _ = build_dual_inputs_from_scenario(
            resolved, primitive_table=primitive_table, rate_mode="certified_upper_capacity", **binding,
        )
        hw_cert, _ = certify_lower_bound(hw_prog, hw_dag)
        hw_bound = hw_cert.l_star_lean_ns

    u, m = framework.seed_u_ns, framework.seed_m_ns
    a_fw, a_dsl = framework.best_wall_ns, dsl.best_wall_ns
    if not 0 <= lower_bound <= a_dsl <= a_fw <= min(m, u):
        raise SoundnessInversionError(
            f"five-level chain violated: lower={lower_bound}, dsl={a_dsl}, "
            f"framework={a_fw}, mainline={m}, unmerge={u}"
        )
    deltas = (u - m, m - a_fw, a_fw - a_dsl, a_dsl - lower_bound)
    assert sum(deltas) == u - lower_bound
    metrics = FiveLevelLadderMetrics(
        lower_bound,
        hw_bound,
        a_dsl,
        a_fw,
        m,
        u,
        *deltas,
        (100 * deltas[-1] / lower_bound) if lower_bound else None,
    )

    evaluations = (
        _step(_field(dossier, "unmerge_eval", "unmerge")),
        _step(_field(dossier, "mainline_eval", "mainline")),
        _step(framework.best_eval),
        _step(dsl.best_eval),
    )
    regions = _by_region(evaluations)
    reconciliation = tuple(
        sum(getattr(region, name) for region in regions) - wall
        for name, wall in zip(("u_wall_ns", "m_wall_ns", "fw_wall_ns", "dsl_wall_ns"), (u, m, a_fw, a_dsl))
    )
    if any(abs(gap) > SHIPPING_REGRESSION_TOLERANCE_NS for gap in reconciliation):
        raise ValueError(f"region wall attribution does not reconcile: {reconciliation}")

    verification = _verification(dsl.best_eval)
    plan = dsl.best_plan
    search = _TrialSearch(dossier, len(plan.edits))
    search.seed(plan, dsl.best_eval)
    regression = _shipping_regression(verification, u, m)
    if regression is not None and verification is not None:
        verification = replace(verification, shipping_regression=regression)
    impacts = framework.loo_impacts if plan == framework.best_plan else ()
    if len(impacts) < len(plan.edits):
        impacts = search.loo(plan, a_dsl, search.remaining)
    opcodes = {edit.edit_id: type(edit.operation).__name__ for edit in plan.edits}
    loo = tuple(LooMarginalImpact(edit_id, opcodes[edit_id], impact) for edit_id, impact in impacts)
    loo += _shipping_overlay_loo(search, dossier, plan, a_dsl, u, m)
    waterfall = FiveLevelWaterfall(
        _ordered_telescope(m, framework, dsl),
        loo,
        regions,
        reconciliation,
        regression,
    )
    digest = getattr(dossier, "scenario_digest", None)
    if digest is None:
        digest = _field(_field(dossier, "handle", "scenario"), "scenario_digest", "digest")
    combined_stats = framework.search_stats.merge(dsl.search_stats)
    eval_ctx = getattr(getattr(dossier, "handle", None), "evaluation_context", None)
    bindings = getattr(eval_ctx, "bindings", None)
    coverage_gaps = tuple(bindings.primitive_coverage_gaps) if bindings is not None else ()
    return FiveLevelReport(
        FIVE_LEVEL_REPORT_SCHEMA,
        digest,
        metrics,
        certificate,
        subgradient,
        waterfall,
        framework.best_plan,
        dsl.best_plan,
        dsl.catlass_specs,
        dsl.dsl_specs,
        dsl.fuse_specs,
        *evaluations,
        verification,
        search_stats=combined_stats,
        primitive_coverage_gaps=coverage_gaps,
    )


def _resolve_primitive_table_digest(handle: Any, case: Case, explicit_table: Any) -> str:
    from mindie_sim.rewrite._oracle_prefetch import primitive_table_digest_token

    eval_ctx = getattr(handle, "evaluation_context", None)
    table = getattr(eval_ctx, "primitive_table", None) or explicit_table
    for attr in ("envelope_sha256", "payload_sha256", "digest"):
        val = getattr(table, attr, "")
        if isinstance(val, str) and val.strip():
            return val.strip()
    token = primitive_table_digest_token(table)
    sku = str(getattr(case.machine, "sku", "") or getattr(case.machine, "name", "") or "default")
    payload = json.dumps({"sku": sku, "token": token}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _try_resident_memory_receipt(case: Case, request: StepRequest) -> Any:
    try:
        mem = Memory(case)
        seqs = request.sequences
        batch = max(seqs) if isinstance(seqs, (tuple, list)) and seqs else int(seqs or 1)
        ctx = max(0, int(request.context or 0))
        q_tok = max(1, int(request.tokens or 1))
        return mem.resident_memory_receipt(
            phase=request.phase,
            sequence_tokens=(ctx + q_tok) if ctx > 0 else None,
            batch_per_dp=max(1, batch),
            query_tokens=q_tok,
        )
    except (ModelingError, ValueError, ZeroDivisionError):
        return None


def _certify_stage_l_star(
    resolved,
    *,
    applied=None,
    region_summaries=(),
    primitive_table=None,
) -> int:
    binding: dict[str, Any] = {}
    if applied is not None:
        binding["applied"] = applied
        binding["region_summaries"] = region_summaries
    prog, dag, _ = build_dual_inputs_from_scenario(
        resolved,
        primitive_table=primitive_table,
        rate_mode="engine_ceiling",
        **binding,
    )
    cert, _ = certify_lower_bound(prog, dag)
    return int(cert.l_star_lean_ns)


def _build_stage_receipt(
    *,
    stage_id: StageId,
    stack_track: StackTrack,
    step_eval: StepEvaluation,
    scenario_digest: str,
    primitive_table_digest: str,
    plan_digest: str,
    l_star_ns: int,
    memory_receipt: Any,
    status: StageCommitStatus = "committed",
    rejection_reason: str = "",
) -> StageReceipt:
    span_ns = max(0, round(step_eval.span_ms * NS_PER_MS))
    bound_eval = replace(step_eval, primitive_table_digest=primitive_table_digest)
    by_occ: dict[str, int] = defaultdict(int)
    idle_ns = 0
    for owner, duration in _partition(bound_eval):
        if owner is None:
            idle_ns += duration
        else:
            by_occ[_occurrence_label(owner)] += duration
    exec_key = hashlib.sha256(
        json.dumps(
            {
                "scenario_digest": scenario_digest,
                "graph_id": bound_eval.graph_id,
                "plan_sha256": plan_digest,
                "primitive_table_digest": primitive_table_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return StageReceipt(
        stage_id=stage_id,
        stack_track=stack_track,
        start_tick_ns=0,
        end_tick_ns=span_ns,
        span_ns=span_ns,
        l_star_ns=int(l_star_ns),
        l_star_valid=(status == "committed"),
        execution_key_sha256=exec_key,
        plan_sha256=plan_digest,
        idle_or_wait_ns=idle_ns,
        exposed_ns_by_occ=tuple(by_occ.items()),
        committed_case_digest=scenario_digest,
        status=status,
        primitive_table_digest=primitive_table_digest,
        step_eval=bound_eval,
        memory_receipt=memory_receipt,
        rejection_reason=rejection_reason,
    )


def _clone_identity_stage(
    prev: StageReceipt,
    *,
    stage_id: StageId,
    stack_track: StackTrack,
    status: StageCommitStatus = "committed",
    rejection_reason: str = "",
) -> StageReceipt:
    return StageReceipt(
        stage_id=stage_id,
        stack_track=stack_track,
        start_tick_ns=prev.start_tick_ns,
        end_tick_ns=prev.end_tick_ns,
        span_ns=prev.span_ns,
        l_star_ns=prev.l_star_ns,
        l_star_valid=(prev.l_star_valid if status == "committed" else False),
        execution_key_sha256=prev.execution_key_sha256,
        plan_sha256=prev.plan_sha256,
        idle_or_wait_ns=prev.idle_or_wait_ns,
        exposed_ns_by_occ=prev.exposed_ns_by_occ,
        committed_case_digest=prev.committed_case_digest,
        status=status,
        primitive_table_digest=prev.primitive_table_digest,
        step_eval=prev.step_eval,
        memory_receipt=prev.memory_receipt,
        sub_stage_receipts=prev.sub_stage_receipts,
        rejection_reason=rejection_reason,
    )


def _evaluate_stage2_framework(
    dossier,
    stage1: StageReceipt,
    *,
    base_resolved,
    base_track: StackTrack,
    stack_track: StackTrack,
    comparison_policy: ComparisonPolicy,
    primitive_table: CapacityTable | None,
    pt_digest: str,
    mem_receipt: Any,
    max_trials: int,
) -> tuple[StageReceipt, FrameworkSolveResult | None, DualSubgradientReport | None]:
    from mindie_sim.rewrite.oracle import _baseline
    from mindie_sim.rewrite.schema import plan_sha256

    if comparison_policy == "simulation_only" or stack_track not in {"proposed", "target"}:
        return _clone_identity_stage(stage1, stage_id="stage2_unmerged_pr", stack_track=base_track), None, None

    prog0, dag0, names0 = build_dual_inputs_from_scenario(base_resolved)
    _, pricing = certify_lower_bound(prog0, dag0)
    by_idx = dict(zip(pricing.resource_names, names0))
    pricing = replace(
        pricing,
        resource_names=names0,
        tight_resources=tuple(by_idx[k] for k in pricing.tight_resources),
    )
    framework_result = solve_framework(dossier, dual_report=pricing, max_trials=max_trials)
    fw_eval = _step(framework_result.best_eval)
    fw_span_ns = max(0, round(fw_eval.span_ms * NS_PER_MS))
    if comparison_policy == "monotonic_optimization" and fw_span_ns > stage1.span_ns:
        stage2 = _clone_identity_stage(stage1, stage_id="stage2_unmerged_pr", stack_track="proposed")
        return stage2, framework_result, pricing
    fw_resolved = _baseline(dossier.handle, framework_result.best_plan).scenario
    l_star_2 = _certify_stage_l_star(
        fw_resolved,
        applied=framework_result.best_eval.applied,
        region_summaries=framework_result.best_eval.region_summaries,
        primitive_table=primitive_table,
    )
    stage2 = _build_stage_receipt(
        stage_id="stage2_unmerged_pr",
        stack_track="proposed",
        step_eval=fw_eval,
        scenario_digest=dossier.scenario_digest,
        primitive_table_digest=pt_digest,
        plan_digest=plan_sha256(framework_result.best_plan),
        l_star_ns=l_star_2,
        memory_receipt=mem_receipt,
    )
    return stage2, framework_result, pricing


def _evaluate_stage3_target(
    dossier,
    stage2: StageReceipt,
    *,
    framework_result: FrameworkSolveResult | None,
    pricing: DualSubgradientReport | None,
    ext_plan: RewritePlan | None,
    stack_track: StackTrack,
    comparison_policy: ComparisonPolicy,
    budget,
    primitive_table: CapacityTable | None,
    pt_digest: str,
    mem_receipt: Any,
    max_trials: int,
    allow_unbound_formula: bool,
) -> StageReceipt:
    from mindie_sim.rewrite.oracle import _baseline, finalize_plan
    from mindie_sim.rewrite.schema import plan_sha256

    if comparison_policy == "simulation_only":
        return _clone_identity_stage(
            stage2,
            stage_id="stage3_target_after",
            stack_track=stage2.stack_track,
            status="skipped",
            rejection_reason="simulation_only_forbids_target_synthesis",
        )
    if ext_plan is not None:
        ext_eval = finalize_plan(
            dossier.handle,
            ext_plan,
            budget=budget,
            primitive_table=primitive_table,
            allow_unbound_formula=bool(allow_unbound_formula),
        )
        if not _accepted(ext_eval):
            return _clone_identity_stage(
                stage2,
                stage_id="stage3_target_after",
                stack_track="target",
                status="rejected_by_policy",
                rejection_reason="external_rewrite_plan_verification_failed",
            )
        ext_step = _step(ext_eval)
        ext_span_ns = max(0, round(ext_step.span_ms * NS_PER_MS))
        if comparison_policy == "monotonic_optimization" and ext_span_ns > stage2.span_ns:
            return _clone_identity_stage(stage2, stage_id="stage3_target_after", stack_track="target")
        ext_resolved = _baseline(dossier.handle, ext_plan).scenario
        l_star_3 = _certify_stage_l_star(
            ext_resolved,
            applied=ext_eval.applied,
            region_summaries=ext_eval.region_summaries,
            primitive_table=primitive_table,
        )
        return _build_stage_receipt(
            stage_id="stage3_target_after",
            stack_track="target",
            step_eval=ext_step,
            scenario_digest=dossier.scenario_digest,
            primitive_table_digest=pt_digest,
            plan_digest=plan_sha256(ext_plan),
            l_star_ns=l_star_3,
            memory_receipt=mem_receipt,
        )
    if stack_track != "target":
        return _clone_identity_stage(
            stage2,
            stage_id="stage3_target_after",
            stack_track=stage2.stack_track,
            status="skipped",
            rejection_reason=f"non_target_track:{stack_track}",
        )
    if framework_result is None:
        raise ModelingError("target_stage_requires_framework_seed")
    unified = solve_unified_regions(
        dossier,
        framework_result,
        dual_report=pricing,
        max_trials=max_trials,
        allow_target_synthesis=True,
    )
    u_eval = _step(unified.best_eval)
    u_span_ns = max(0, round(u_eval.span_ms * NS_PER_MS))
    if comparison_policy == "monotonic_optimization" and u_span_ns > stage2.span_ns:
        return _clone_identity_stage(stage2, stage_id="stage3_target_after", stack_track="target")
    u_resolved = _baseline(dossier.handle, unified.best_plan).scenario
    l_star_3 = _certify_stage_l_star(
        u_resolved,
        applied=unified.best_eval.applied,
        region_summaries=unified.best_eval.region_summaries,
        primitive_table=primitive_table,
    )
    return _build_stage_receipt(
        stage_id="stage3_target_after",
        stack_track="target",
        step_eval=u_eval,
        scenario_digest=dossier.scenario_digest,
        primitive_table_digest=pt_digest,
        plan_digest=plan_sha256(unified.best_plan),
        l_star_ns=l_star_3,
        memory_receipt=mem_receipt,
    )


def solve_target_step_receipt(
    case: CaseInput,
    request: StepRequest,
    *,
    spine=None,
    replay: ReplayOptions = ReplayOptions(),
    primitives=None,
    primitive_table: CapacityTable | None = None,
    stack_id: str = "cann93_va_main",
    stack_track: StackTrack = "target",
    comparison_policy: ComparisonPolicy = "monotonic_optimization",
    rewrite_plan: RewritePlan | Mapping[str, Any] | str | Path | None = None,
    max_trials: int = DEFAULT_LADDER_TRIALS,
    allow_unbound_formula: bool = False,
) -> EngineStepReceipt:
    """Produce the canonical 4-stage ``EngineStepReceipt`` on one shared ``PrimitiveTable``."""
    from mindie_sim.rewrite.oracle import EvalBudget, ScenarioRequest, _baseline, prepare_scenario
    from mindie_sim.rewrite.schema import load_rewrite_plan, plan_sha256

    parsed_case = Case.from_mapping(case)
    ext_plan = load_rewrite_plan(rewrite_plan) if rewrite_plan is not None else None
    budget = EvalBudget(deadline_mono=replay.deadline_mono)
    dossier = prepare_scenario(
        ScenarioRequest(parsed_case, request, stack_id, replay, primitive_table=primitive_table),
        budget=budget,
        spine=spine,
        primitives=primitives,
        primitive_table=primitive_table,
        allow_unbound_formula=bool(allow_unbound_formula),
    )
    pt_digest = _resolve_primitive_table_digest(dossier.handle, parsed_case, primitive_table)
    mem_receipt = _try_resident_memory_receipt(parsed_case, request)
    unmerge_plan, mainline_plan = _seed_plans(dossier)

    base_track: StackTrack = "eager" if stack_track == "eager" else "shipping"
    base_plan = unmerge_plan if stack_track == "eager" else mainline_plan
    base_eval = _step(
        _field(dossier, "unmerge_eval", "unmerge")
        if stack_track == "eager"
        else _field(dossier, "mainline_eval", "mainline")
    )
    base_resolved = _baseline(dossier.handle, base_plan).scenario
    l_star_0 = _certify_stage_l_star(base_resolved, primitive_table=primitive_table)
    stage0 = _build_stage_receipt(
        stage_id="before_merged",
        stack_track=base_track,
        step_eval=base_eval,
        scenario_digest=dossier.scenario_digest,
        primitive_table_digest=pt_digest,
        plan_digest=plan_sha256(base_plan),
        l_star_ns=l_star_0,
        memory_receipt=mem_receipt,
    )
    stage1 = _clone_identity_stage(stage0, stage_id="stage1_ops_config", stack_track=base_track)
    stage2, framework_result, pricing = _evaluate_stage2_framework(
        dossier,
        stage1,
        base_resolved=base_resolved,
        base_track=base_track,
        stack_track=stack_track,
        comparison_policy=comparison_policy,
        primitive_table=primitive_table,
        pt_digest=pt_digest,
        mem_receipt=mem_receipt,
        max_trials=max_trials,
    )
    stage3 = _evaluate_stage3_target(
        dossier,
        stage2,
        framework_result=framework_result,
        pricing=pricing,
        ext_plan=ext_plan,
        stack_track=stack_track,
        comparison_policy=comparison_policy,
        budget=budget,
        primitive_table=primitive_table,
        pt_digest=pt_digest,
        mem_receipt=mem_receipt,
        max_trials=max_trials,
        allow_unbound_formula=allow_unbound_formula,
    )
    last_committed = stage3 if stage3.status == "committed" else stage2
    return EngineStepReceipt(
        scenario_digest=dossier.scenario_digest,
        primitive_table_digest=pt_digest,
        comparison_policy=comparison_policy,
        stages=(stage0, stage1, stage2, stage3),
        final_bound_ns=last_committed.l_star_ns,
    )
