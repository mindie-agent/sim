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

from .contracts import CatlassKernelSpec, DslKernelSpec, FuseRegionSpec, PrimitiveCapacityTable
from .dsl_solver import solve_dsl
from .dual_bound import (
    NS_PER_MS, DualSubgradientReport, OptimalityCertificate,
    build_dual_inputs_from_scenario, certify_lower_bound,
)
from .framework_solver import (
    _TrialSearch, _accepted, _field, _resolved, _seed_plans, _step, _verification, _wall,
    solve_framework,
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


def _occurrence_key(occurrence):
    rank = -1 if occurrence.rank is None else occurrence.rank
    return occurrence.replica, occurrence.stage, rank, occurrence.op_id


def _partition(eval_result):
    wall = round(eval_result.span_ms * NS_PER_MS)
    if wall < 0:
        raise ValueError("step span must be nonnegative")
    endpoints = {}

    def quantize(value):
        if value not in endpoints:
            endpoints[value] = min(wall, max(0, round(value * NS_PER_MS)))
        return endpoints[value]

    rows = {}
    for row in eval_result.operators:
        if row.start_ms is None or row.end_ms is None:
            continue
        start, end = quantize(row.start_ms), quantize(row.end_ms)
        if end >= start:
            rows[row.occurrence] = start, end, row

    def completion(record):
        _, end, row = record
        return -end, row.ordinal, _occurrence_key(row.occurrence)

    service = getattr(eval_result, "service_intervals", {})
    current = min(rows.values(), key=completion, default=None)
    cursor = wall
    visited = set()
    intervals = []
    while current is not None:
        start, end, row = current
        if row.occurrence in visited:
            raise ValueError("cycle in completion evidence")
        visited.add(row.occurrence)
        completion_ns = min(end, cursor)
        if cursor > completion_ns:
            intervals.append((None, cursor - completion_ns))
        if completion_ns > start:
            owner = row.occurrence
            spans = service.get((owner.replica, owner.op_id))
            if spans:
                consumed = start
                elementary = []
                for lo, hi in sorted((max(start, quantize(lo)), min(completion_ns, quantize(hi)))
                                     for lo, hi in spans):
                    lo = max(lo, consumed)
                    if hi <= lo:
                        continue
                    if lo > consumed:
                        elementary.append((None, lo - consumed))
                    elementary.append((owner, hi - lo))
                    consumed = hi
                if completion_ns > consumed:
                    elementary.append((None, completion_ns - consumed))
                intervals.extend(reversed(elementary))
            else:
                intervals.append((owner, completion_ns - start))
        cursor = min(start, completion_ns)
        ready = cursor if row.ready_ms is None else min(cursor, quantize(row.ready_ms))
        current = min((rows[pred] for pred in row.predecessors
                       if pred in rows and rows[pred][1] <= ready),
                      key=completion, default=None)
    if cursor:
        intervals.append((None, cursor))
    return tuple(reversed(intervals))


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
    primitives=None, primitive_table: PrimitiveCapacityTable | None = None,
    stack_id: str = "cann93_va_main", max_trials: int = DEFAULT_LADDER_TRIALS,
) -> FiveLevelReport:
    from mindie_sim.rewrite.oracle import EvalBudget, ScenarioRequest, prepare_scenario
    from mindie_sim.rewrite.verify import SoundnessInversionError

    scenario_request = ScenarioRequest(case, request, stack_id, replay)
    dossier = prepare_scenario(scenario_request, budget=EvalBudget(deadline_mono=replay.deadline_mono),
                               spine=spine, primitives=primitives)
    scenario = _resolved(dossier)
    summaries = getattr(dossier, "region_summaries", ())
    program, dag, names = build_dual_inputs_from_scenario(scenario, region_summaries=summaries)
    certificate, subgradient = certify_lower_bound(program, dag)
    names_by_index = dict(zip(subgradient.resource_names, names))
    subgradient = replace(subgradient, resource_names=names,
                          tight_resources=tuple(names_by_index[name] for name in subgradient.tight_resources))
    lower_bound = certificate.l_star_lean_ns
    hw_bound = None
    if primitive_table is not None:
        hw_prog, hw_dag, _ = build_dual_inputs_from_scenario(
            scenario, region_summaries=summaries, primitive_table=primitive_table,
            rate_mode="certified_upper_capacity",
        )
        hw_cert, _ = certify_lower_bound(hw_prog, hw_dag)
        hw_bound = hw_cert.l_star_lean_ns

    options = dict(dual_report=subgradient, dual_program=program, event_dag=dag,
                   dual_certificate=certificate, max_trials=max_trials)
    framework = solve_framework(dossier, **options)
    dsl = solve_dsl(dossier, framework, **options)
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
    search = _TrialSearch(dossier, len(plan.edits), dual_program=program, event_dag=dag,
                          dual_certificate=certificate)
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
    return FiveLevelReport(
        FIVE_LEVEL_REPORT_SCHEMA, digest, metrics, certificate, subgradient, waterfall,
        framework.best_plan, dsl.best_plan, dsl.catlass_specs, dsl.dsl_specs, dsl.fuse_specs,
        *evaluations, verification,
    )
