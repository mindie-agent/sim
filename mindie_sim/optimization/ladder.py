"""Five-level ladder, checked lower bound and replay attribution.

[Role] Orchestrate both searches and reconcile their full-step timing evidence.
[Invariants] Exact integer telescopes and disjoint wall-time ownership.
[Timeouts & Locks] Search is bounded by oracle budgets and caller trial limits.
[Side-Effect Policy] Read-only evaluation. All mutation stays in local bookkeeping.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal

from mindie_sim.prediction.results import ReplayOptions, StepEvaluation, StepRequest

from .contracts import CapacityTable, CatlassKernelSpec, DslKernelSpec, FuseRegionSpec
from .dual_bound import (
    NS_PER_MS, DualSubgradientReport, OptimalityCertificate,
    build_dual_inputs_from_scenario, certify_lower_bound,
)
from .framework_solver import (
    CandidateSearchStats, _TrialSearch, _accepted, _field, _resolved, _seed_plans, _step,
    _verification, _wall, solve_framework,
)

if TYPE_CHECKING:
    from mindie_sim.rewrite.schema import RewritePlan
    from mindie_sim.rewrite.verify import ShippingRegression, VerificationReceipt

FIVE_LEVEL_REPORT_SCHEMA = "mindie-sim.five-level-report/1"
SHIPPING_REGRESSION_TOLERANCE_NS = 1000
DEFAULT_LADDER_TRIALS = 16
STEP_WAIT_REGION = "step:wait"
SHIPPING_OVERLAY = "shipping_overlay"

__all__ = (
    "FIVE_LEVEL_REPORT_SCHEMA", "FiveLevelLadderMetrics", "RegionWallAttribution",
    "OrderedTelescopeStep", "LooMarginalImpact", "FiveLevelWaterfall", "FiveLevelReport",
    "partition_step_wall_ns", "solve_five_level_ladder",
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


def _occurrence_key(occurrence):
    rank = -1 if occurrence.rank is None else occurrence.rank
    return occurrence.replica, occurrence.stage, rank, occurrence.op_id


def _partition(eval_result):
    del eval_result
    raise NotImplementedError(
        "LEGACY_PURGED: consume canonical exposed/idle intervals from prediction.runtime.exposed_observer"
    )


def partition_step_wall_ns(eval_result: StepEvaluation) -> tuple[tuple[str, int], ...]:
    """Attribute each interval once along the recorded completion chain."""
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
    return ShippingRegression(delta_ns=delta, loo_culprits=culprits,
                              baseline_u_ns=u, baseline_m_ns=m)


def _shipping_overlay_loo(search, dossier, plan, wall, u, m):
    unmerge, mainline = _seed_plans(dossier)
    if (m == u or plan.base_graph_sha256 != mainline.base_graph_sha256
            or plan.base_graph_sha256 == unmerge.base_graph_sha256):
        return ()
    rebased = replace(plan, base_graph_sha256=unmerge.base_graph_sha256,
                      framing_sha256=unmerge.framing_sha256)
    if not rebased.edits:
        without = u
    else:
        result = search.try_finalize(rebased)
        if result is None or not _accepted(result):
            return ()
        without = _wall(result)
    return (LooMarginalImpact(SHIPPING_OVERLAY, "ShippingOverlay", without - wall),)


def solve_five_level_ladder(
    case, request: StepRequest, *, spine=None, replay: ReplayOptions = ReplayOptions(),
    primitives=None, primitive_table: CapacityTable | None = None,
    stack_id: str = "cann93_va_main", max_trials: int = DEFAULT_LADDER_TRIALS,
    allow_unbound_formula: bool = False,
) -> FiveLevelReport:
    from mindie_sim.rewrite.oracle import EvalBudget, ScenarioRequest, _baseline, prepare_scenario
    from mindie_sim.rewrite.verify import SoundnessInversionError

    scenario_request = ScenarioRequest(case, request, stack_id, replay, primitive_table=primitive_table)
    dossier = prepare_scenario(scenario_request, budget=EvalBudget(deadline_mono=replay.deadline_mono),
                               spine=spine, primitives=primitives, primitive_table=primitive_table,
                               allow_unbound_formula=bool(allow_unbound_formula))
    scenario = _resolved(dossier)
    program, dag, names = build_dual_inputs_from_scenario(scenario)
    _, pricing = certify_lower_bound(program, dag)
    prices_by_index = dict(zip(pricing.resource_names, names))
    pricing = replace(pricing, resource_names=names,
                      tight_resources=tuple(prices_by_index[key] for key in pricing.tight_resources))
    framework = solve_framework(dossier, dual_report=pricing, max_trials=max_trials)
    raise NotImplementedError(
        "LEGACY_PURGED: wire unified rewrite.algebra + templates.fuse_region + Engine replay"
    )
    resolved = _baseline(dossier.handle, dsl.best_plan).scenario
    binding = dict(applied=dsl.best_eval.applied, region_summaries=dsl.best_eval.region_summaries)
    program, dag, names = build_dual_inputs_from_scenario(resolved, **binding)
    certificate, subgradient = certify_lower_bound(program, dag)
    names_by_index = dict(zip(subgradient.resource_names, names))
    subgradient = replace(subgradient, resource_names=names,
                          tight_resources=tuple(names_by_index[name] for name in subgradient.tight_resources))
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
    metrics = FiveLevelLadderMetrics(lower_bound, hw_bound, a_dsl, a_fw, m, u, *deltas,
                                     (100 * deltas[-1] / lower_bound) if lower_bound else None)

    evaluations = (_step(_field(dossier, "unmerge_eval", "unmerge")),
                   _step(_field(dossier, "mainline_eval", "mainline")),
                   _step(framework.best_eval), _step(dsl.best_eval))
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
        regions, reconciliation, regression,
    )
    digest = getattr(dossier, "scenario_digest", None)
    if digest is None:
        digest = _field(_field(dossier, "handle", "scenario"), "scenario_digest", "digest")
    combined_stats = framework.search_stats.merge(dsl.search_stats)
    coverage_gaps = tuple(getattr(
        getattr(getattr(dossier, "handle", None), "evaluation_context", None),
        "bindings", None,
    ).primitive_coverage_gaps) if getattr(
        getattr(getattr(dossier, "handle", None), "evaluation_context", None),
        "bindings", None,
    ) is not None else ()
    return FiveLevelReport(
        FIVE_LEVEL_REPORT_SCHEMA, digest, metrics, certificate, subgradient, waterfall,
        framework.best_plan, dsl.best_plan, dsl.catlass_specs, dsl.dsl_specs, dsl.fuse_specs,
        *evaluations, verification, search_stats=combined_stats, primitive_coverage_gaps=coverage_gaps,
    )
