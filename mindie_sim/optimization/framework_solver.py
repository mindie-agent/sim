"""Constructive framework optimization over immutable rewrite plans.

[Role] Propose local, collective, stream and prefetch interventions for replay.
[Invariants] Only a strictly faster full-occurrence, verified replay wins.
[Timeouts & Locks] Trial counts bound search. The oracle owns replay deadlines.
[Side-Effect Policy] Read-only scenario and plan handling through the rewrite oracle.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from itertools import islice, zip_longest
from math import gcd
from typing import TYPE_CHECKING, Literal

from mindie_sim.prediction.streams import (
    MAIN_STREAM, StreamCoreBudget, StreamEvent, StreamLane, StreamPlan,
    bind_stream_owners, lower_stream_edges, proportional_coupled_placement,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die

from .dual_bound import NS_PER_MS, DualSubgradientReport, _atomic_ops, _machine, _op_contract_key

if TYPE_CHECKING:
    from mindie_sim.rewrite.oracle import FullPlanEvaluation, ScenarioDossier
    from mindie_sim.rewrite.schema import RewritePlan

DEFAULT_MAX_TRIALS = 24
ORACLE_BATCH_SIZE = 8
REQUIRED_GATE_COUNT = 5
FRAMEWORK_STAGES = (
    "F1_local_k1", "F2_collective_dbo", "F3_stream_core", "F4_prefetch",
)
__all__ = (
    "CandidateSearchStats", "CoreLatticePoint", "FrameworkSolveResult",
    "generate_gcd_core_lattice", "solve_framework",
)


@dataclass(frozen=True, slots=True)
class CandidateSearchStats:
    """Explicit disclosure of constructive search funnel counts and rejections."""

    eligible_sites: int = 0
    generated_candidates: int = 0
    rejected_candidates: int = 0
    replayed_candidates: int = 0
    rejection_reasons: tuple[tuple[str, int], ...] = ()

    def merge(self, other: CandidateSearchStats) -> CandidateSearchStats:
        reasons = Counter(dict(self.rejection_reasons))
        reasons.update(dict(other.rejection_reasons))
        return CandidateSearchStats(
            eligible_sites=self.eligible_sites + other.eligible_sites,
            generated_candidates=self.generated_candidates + other.generated_candidates,
            rejected_candidates=self.rejected_candidates + other.rejected_candidates,
            replayed_candidates=self.replayed_candidates + other.replayed_candidates,
            rejection_reasons=tuple(sorted(reasons.items())),
        )


@dataclass(frozen=True, slots=True)
class CoreLatticePoint:
    lanes: tuple[StreamLane, ...]
    isolation: Literal["disjoint", "shared_cap"]
    gcd_step: tuple[int, int]

@dataclass(frozen=True, slots=True)
class FrameworkSolveResult:
    best_plan: RewritePlan
    best_eval: FullPlanEvaluation
    best_wall_ns: int
    seed_u_ns: int
    seed_m_ns: int
    stage_history: tuple[tuple[str, int, str], ...]
    loo_impacts: tuple[tuple[str, int], ...]
    tried_candidates: int
    search_stats: CandidateSearchStats = CandidateSearchStats()

def _wave_points(counts, total, step=1):
    if not total or not step:
        return ()
    widths = tuple(range(step, total + 1, step))
    if not counts:
        return widths
    result = []
    previous = None
    for width in widths:
        waves = tuple((count + width - 1) // width for count in counts)
        if waves != previous or width == total:
            result.append(width)
        previous = waves
    return tuple(result)

def generate_gcd_core_lattice(
    aic_total: int, aiv_total: int, tile_counts: Sequence[int] = (), *,
    include_three_stream: bool = False,
) -> tuple[CoreLatticePoint, ...]:
    if min(aic_total, aiv_total, *tile_counts, 0) < 0:
        raise ValueError("core and tile counts must be nonnegative")
    common = gcd(aic_total, aiv_total)
    step = (aic_total // common, aiv_total // common) if common else (0, 0)
    aic_widths = _wave_points(tile_counts, aic_total)
    aiv_widths = _wave_points(tile_counts, aiv_total)
    main_budgets = [(aic_total, aiv_total), (0, 0)]
    main_budgets.extend((aic, 0) for aic in reversed(aic_widths))
    main_budgets.extend((0, aiv) for aiv in reversed(aiv_widths))
    if all(step):
        main_budgets.extend((aic, aic // step[0] * step[1])
                            for aic in reversed(_wave_points(tile_counts, aic_total, step[0])))
    points = []
    for isolation in ("disjoint", "shared_cap"):
        def emit(budgets):
            lanes = tuple(StreamLane(name, StreamCoreBudget(aic, aiv, isolation))
                          for name, (aic, aiv) in budgets)
            points.append(CoreLatticePoint(lanes, isolation, step))

        for aic, aiv in dict.fromkeys(main_budgets):
            main = (MAIN_STREAM, (aic, aiv))
            emit((main, ("comm", (0, 0))))
            if not include_three_stream or not aic:
                continue
            max_aic = aic_total - aic if isolation == "disjoint" else aic_total
            max_aiv = aiv_total - aiv if isolation == "disjoint" else aiv_total
            groups = min(max_aic // step[0], max_aiv // step[1]) if all(step) else 0
            aux = [(0, width) for width in reversed(aiv_widths) if width <= max_aiv]
            if max_aiv:
                aux.insert(0, (0, max_aiv))
            aux.extend((k * step[0], k * step[1]) for k in range(groups, 0, -1))
            for sparse in dict.fromkeys(aux):
                emit((main, ("csa", sparse), ("comm", (0, 0))))
    return tuple(points)

def _field(record, *names):
    for name in names:
        value = getattr(record, name, None)
        if value is not None:
            return value
    raise AttributeError(f"{type(record).__name__} must supply {names[0]}")

def _resolved(dossier):
    value = _field(dossier, "resolved", "resolved_scenario", "scenario", "handle")
    return getattr(value, "resolved", getattr(value, "scenario", value))

def _step(evaluation):
    if hasattr(evaluation, "span_ms") and hasattr(evaluation, "operators"):
        return evaluation
    return _field(evaluation, "evaluation", "step_eval", "step")

def _wall(evaluation):
    wall = getattr(evaluation, "wall_ns", None)
    if wall is None:
        wall = getattr(evaluation, "estimated_wall_ns", None)
    return wall if wall is not None else round(_step(evaluation).span_ms * NS_PER_MS)

def _verification(evaluation):
    return _field(evaluation, "verification", "receipt")

def _accepted(evaluation):
    receipt = _verification(evaluation)
    gates = _field(receipt, "gates", "gate_receipts")
    if hasattr(gates, "values"):
        gates = gates.values()
    gates = tuple(gates)
    return (len(gates) == REQUIRED_GATE_COUNT
            and all(_field(gate, "passed", "ok") for gate in gates)
            and getattr(receipt, "scope", "full") == "full")


def _operation_dependencies(operation):
    from mindie_sim.rewrite.schema import KernelBoundary, OpRef, PortRef

    def visit(value):
        if isinstance(value, OpRef):
            return (value.id,) if value.kind == "result_of" else ()
        if isinstance(value, tuple):
            return tuple(dict.fromkeys(dep for item in value for dep in visit(item)))
        if is_dataclass(value) and (value is operation or isinstance(value, (PortRef, KernelBoundary))):
            return visit(tuple(getattr(value, field.name) for field in fields(value)))
        return ()

    return visit(operation)

def _without_edit(plan, edit_id):
    dependencies = {edit.edit_id: set(edit.after + _operation_dependencies(edit.operation))
                    for edit in plan.edits}
    remove = {edit_id}
    while True:
        dependents = {edit_id for edit_id, after in dependencies.items() if remove & after}
        if dependents <= remove:
            return replace(plan, edits=tuple(edit for edit in plan.edits if edit.edit_id not in remove))
        remove.update(dependents)

def _with_operations(plan, stage, operations):
    from mindie_sim.rewrite.schema import Edit

    edits = list(plan.edits)
    existing = {edit.edit_id for edit in edits}
    for index, operation in enumerate(operations):
        stem = f"{stage}_{index}"
        edit_id = stem
        suffix = 0
        while edit_id in existing:
            suffix += 1
            edit_id = f"{stem}_{suffix}"
        edits.append(Edit(edit_id=edit_id, after=_operation_dependencies(operation),
                          operation=operation))
        existing.add(edit_id)
    return replace(plan, edits=tuple(edits))

class _TrialSearch:
    def __init__(self, dossier, max_trials, *, dual_program=None, event_dag=None, dual_certificate=None):
        from mindie_sim.rewrite.oracle import EvalBudget

        self.handle = _field(dossier, "handle", "scenario")
        self.budget = getattr(dossier, "budget", None) or EvalBudget()
        self.options = dict(dual_program=dual_program, event_dag=event_dag,
                            dual_certificate=dual_certificate)
        self.remaining = max(0, max_trials)
        self.tried = 0
        self.eligible_sites = 0
        self.generated = 0
        self.rejected = 0
        self.replayed = 0
        self.rejection_reasons: Counter[str] = Counter()
        self.history = []
        self.seen = set()
        self.frontier = {}

    def stats(self) -> CandidateSearchStats:
        return CandidateSearchStats(
            eligible_sites=self.eligible_sites,
            generated_candidates=self.generated,
            rejected_candidates=self.rejected,
            replayed_candidates=self.replayed,
            rejection_reasons=tuple(sorted(self.rejection_reasons.items())),
        )

    def note_eligible(self, count: int) -> None:
        self.eligible_sites += max(0, int(count))

    def finalize(self, plan):
        from mindie_sim.rewrite.oracle import finalize_plan
        if getattr(self, "best_plan", None) == plan:
            return self.best_eval
        handle_base = getattr(getattr(self.handle, "scenario", None), "base_graph_sha256", None)
        options = self.options if (handle_base is None or plan.base_graph_sha256 == handle_base) else {}
        return finalize_plan(self.handle, plan, budget=self.budget, **options)

    def seed(self, plan, evaluation):
        from mindie_sim.rewrite.schema import plan_sha256

        self.best_plan, self.best_eval = plan, evaluation
        self.seen.add(plan_sha256(plan))
        self.frontier[plan.base_graph_sha256] = (plan, evaluation)

    def try_finalize(self, plan):
        from mindie_sim.rewrite.verify import SoundnessInversionError

        try:
            return self.finalize(plan)
        except SoundnessInversionError:
            raise
        except ModelingError as exc:
            self.rejection_reasons[str(exc).partition(":")[0] or "modeling_error"] += 1
            return None

    def compare(self, plans, limit):
        from mindie_sim.rewrite.oracle import evaluate_rewrites
        from mindie_sim.rewrite.schema import plan_sha256

        def unique():
            for plan in plans:
                digest = plan_sha256(plan)
                if digest not in self.seen:
                    self.seen.add(digest)
                    self.generated += 1
                    yield plan

        pending = unique()
        while limit > 0 and self.remaining > 0:
            batch = tuple(islice(pending, min(limit, self.remaining, self.budget.max_candidates, ORACLE_BATCH_SIZE)))
            if not batch:
                return
            limit -= len(batch)
            self.remaining -= len(batch)
            self.tried += len(batch)
            screened = evaluate_rewrites(self.handle, batch, budget=self.budget)
            evaluations = tuple(_field(screened, "candidates", "evaluations", "results"))
            if len(evaluations) != len(batch):
                raise ValueError("oracle did not return one result per candidate")
            ranked = []
            for plan, estimate in zip(batch, evaluations):
                if getattr(estimate, "accepted", True) is False:
                    self.rejected += 1
                    reason = str(getattr(estimate, "reason", "") or getattr(estimate, "status", "rejected"))
                    self.rejection_reasons[reason.partition(":")[0] or "rejected"] += 1
                    continue
                try:
                    estimated_wall = _wall(estimate)
                except AttributeError:
                    self.rejected += 1
                    self.rejection_reasons["missing_wall_estimate"] += 1
                    continue
                ranked.append((estimated_wall, plan_sha256(plan), plan))
            for _, _, plan in sorted(ranked):
                self.replayed += 1
                result = self.try_finalize(plan)
                if result is not None and _accepted(result):
                    key = plan.base_graph_sha256
                    previous = self.frontier.get(key)
                    if previous is None or _wall(result) < _wall(previous[1]):
                        self.frontier[key] = (plan, result)
                    if _wall(result) < _wall(self.best_eval):
                        self.seed(plan, result)
                else:
                    if result is not None:
                        self.rejection_reasons["finalize_gate_failed"] += 1
                    else:
                        self.rejection_reasons["finalize_error"] += 1

    def stage(self, stage_id, plans, limit):
        from mindie_sim.rewrite.schema import plan_sha256

        self.compare(plans, limit)
        self.history.append((stage_id, _wall(self.best_eval), plan_sha256(self.best_plan)))

    def loo(self, plan, wall_ns, limit, *, consider=False):
        impacts = []
        for edit in islice(plan.edits, min(limit, self.remaining)):
            reduced = _without_edit(plan, edit.edit_id)
            self.remaining -= 1
            self.tried += 1
            result = self.try_finalize(reduced)
            if result is not None and _accepted(result):
                impacts.append((edit.edit_id, _wall(result) - wall_ns))
                if consider and _wall(result) < _wall(self.best_eval):
                    self.seed(reduced, result)
        return tuple(impacts)

def _seed_plans(dossier):
    from mindie_sim.rewrite.schema import empty_plan

    mainline = _field(dossier, "mainline_plan")
    unmerge = getattr(dossier, "unmerge_plan", None)
    if unmerge is None:
        unmerge = empty_plan(base_graph_sha256=mainline.base_graph_sha256,
                             framing_sha256=mainline.framing_sha256)
    return unmerge, mainline


def _search_bases(dossier, search):
    from mindie_sim.rewrite.oracle import ScenarioHandle, _baseline

    seen = []
    for seed in (search.best_plan, *_seed_plans(dossier)):
        plan, _ = search.frontier.get(seed.base_graph_sha256, (seed, None))
        if plan in seen:
            continue
        seen.append(plan)
        scenario = (_baseline(search.handle, plan).scenario
                    if isinstance(search.handle, ScenarioHandle) else _resolved(dossier))
        yield plan, scenario


def _interleave(proposals):
    return (plan for wave in zip_longest(*proposals) for plan in wave if plan is not None)

def _local_operations(scenario, ops, mainline):
    from mindie_sim.rewrite.schema import HoistOrDeduplicate, OpRef

    groups = defaultdict(list)
    for op in ops:
        meta = dict(op.metadata)
        targets = (OpRef("atom", op.op_id),)
        key = _op_contract_key(op, meta)
        if key is not None:
            groups[key].extend(targets)
        if meta.get("load_time"):
            yield HoistOrDeduplicate(targets, "load_time")
        if meta.get("device_group_list_elide_d2h"):
            yield HoistOrDeduplicate(targets, "device_group_list_elide_d2h")
    for targets in groups.values():
        if len(targets) > 1:
            yield HoistOrDeduplicate(tuple(targets), "intra_step")
    yield from getattr(scenario, "shipping_local_fusions", ())
    yield from (edit.operation for edit in mainline.edits
                if getattr(edit.operation, "fusion_tier", None) == "K1_shipping"
                and not (edit.after or _operation_dependencies(edit.operation)))

def _collective_operations(ops, dual_report):
    from mindie_sim.rewrite.schema import CollectiveTransform, OpRef, PipelineTiles

    links = () if dual_report is None else tuple(
        name.removeprefix("link:").split("@", 1)[0]
        for name, weight in zip(dual_report.resource_names, dual_report.weights)
        if weight > 0 and name.startswith("link:")
    )
    by_priority = []
    dbo = []
    for op in ops:
        meta = dict(op.metadata)
        ref = OpRef("atom", op.op_id)
        for mode, route in meta.get("collective_transforms", ()):
            by_priority.append((route not in links, op.op_id, mode, route,
                                CollectiveTransform(ref, mode, route)))
        tile = meta.get("dbo_tile_spec", meta.get("tile_pipeline_spec"))
        if op.family in {"grouped_gemm", "grouped_moe", "mc2"}:
            dbo.append(PipelineTiles(ref, tile, cohort_split="dbo_2cohort"))
    ranked = sorted(by_priority, key=lambda entry: entry[:-1])
    yield from (entry[-1] for entry in ranked if not entry[0])
    yield from dbo[:1]
    yield from (entry[-1] for entry in ranked if entry[0])
    yield from dbo[1:]

def _independent(left, right, by_id):
    if left.stage != right.stage:
        return False
    for source, target in ((left, right), (right, left)):
        pending, visited = list(target.deps), set()
        while pending:
            op_id = pending.pop()
            if op_id == source.op_id:
                return False
            if op_id in by_id and op_id not in visited:
                visited.add(op_id)
                pending.extend(by_id[op_id].deps)
    return True

def _stream_candidates(scenario, ops, op_refs=()):
    from mindie_sim.rewrite.schema import OpRef, StreamSchedule

    machine = _machine(scenario)
    devices = getattr(machine, "dies", ())
    if not devices:
        return
    paths = tuple((device.rank, datapath_for_die(device)) for device in devices)
    aic = min(path.aic_count for _, path in paths)
    aiv = min(path.aiv_count for _, path in paths)
    handles = dict(op_refs)
    by_id = {op.op_id: op for op in ops}
    kinds, sparse = {}, set()
    for op in ops:
        meta = dict(op.metadata)
        cube = bool(op.flops or op.matrix_flops or meta.get("requires_aic"))
        vec = bool(op.vector_ops or meta.get("requires_aiv"))
        kinds[op.op_id] = cube, vec
        if meta.get("sparse") or "sparse" in op.family or "indexer" in op.family:
            sparse.add(op.op_id)
    comm = [op for op in ops if op.link_bytes and not any(kinds[op.op_id])]
    cube_ops = [op for op in ops if kinds[op.op_id][0] and not op.link_bytes]
    side = [op for op in ops if not op.link_bytes and (
        op.op_id in sparse or kinds[op.op_id] == (False, True) or
        dict(op.metadata).get("independent_stream")
    )]
    layouts = []
    for transfer in comm:
        for compute in cube_ops:
            if not _independent(compute, transfer, by_id):
                continue
            for auxiliary in side:
                if auxiliary.op_id != compute.op_id and all(
                    _independent(auxiliary, other, by_id) for other in (compute, transfer)
                ):
                    layouts.append({transfer.op_id: "comm", auxiliary.op_id: "csa"})
                    break
            layouts.append({transfer.op_id: "comm"})
            break
        if layouts:
            break
    if not layouts:
        for auxiliary in side:
            if any(compute.op_id != auxiliary.op_id and _independent(compute, auxiliary, by_id)
                   for compute in cube_ops):
                layouts.append({auxiliary.op_id: "csa"})
                break
    counts = tuple(sorted({int(value) for op in ops
                          if (value := dict(op.metadata).get("tile_count", 0)) > 0}))

    def prepare_assignment(assignment):
        lane_ids = set(assignment.values()) | {MAIN_STREAM}
        grouped = defaultdict(list)
        for op in ops:
            target = assignment.get(op.op_id, MAIN_STREAM)
            for dep in op.deps:
                source = assignment.get(dep, MAIN_STREAM)
                if dep in by_id and by_id[dep].stage == op.stage and source != target:
                    grouped[dep, source, target].append(op.op_id)
        events = tuple(StreamEvent(f"event_{i}", source, dep, target, tuple(dict.fromkeys(wait)))
                       for i, ((dep, source, target), wait) in enumerate(grouped.items()))
        raw = tuple(assignment.items())
        refs = tuple((handles.get(op_id, OpRef("atom", op_id)), lane) for op_id, lane in raw)
        lane_needs: dict[str, tuple[bool, bool, bool]] = {}
        for op_id, (needs_cube, needs_vec) in kinds.items():
            lane = assignment.get(op_id, MAIN_STREAM)
            prev_c, prev_v, prev_cv = lane_needs.get(lane, (False, False, False))
            lane_needs[lane] = (
                prev_c or needs_cube,
                prev_v or needs_vec,
                prev_cv or (needs_cube and needs_vec),
            )
        return lane_ids, events, raw, refs, lane_needs

    def schedule(prepared, point, edges_verified):
        lane_ids, events, raw, refs, lane_needs = prepared
        lanes = tuple(lane for lane in point.lanes if lane.stream_id in lane_ids)
        if {lane.stream_id for lane in lanes} != lane_ids:
            return None, edges_verified
        try:
            plan = None
            for rank, path in paths:
                plan = StreamPlan(rank, lanes, raw, events)
                owners = bind_stream_owners(plan, path)
                for lane, (needs_cube, needs_vec, needs_both) in lane_needs.items():
                    bound = owners[lane]
                    if (needs_cube and not bound.aic_limit) or (needs_vec and not bound.aiv_limit):
                        return None, edges_verified
                    if needs_both and bound.isolation == "disjoint":
                        proportional_coupled_placement(bound, path, stream_id=lane)
            if not edges_verified and plan is not None:
                lower_stream_edges(ops, plan)
                edges_verified = True
        except ModelingError:
            return None, edges_verified
        return StreamSchedule(lanes, refs, events), edges_verified

    groups = []
    for assignment in layouts:
        prepared = prepare_assignment(assignment)
        with_aux = "csa" in assignment.values()
        points = generate_gcd_core_lattice(aic, aiv, counts, include_three_stream=with_aux)
        for isolation in ("shared_cap", "disjoint"):
            groups.append((prepared, iter(tuple(p for p in points if p.isolation == isolation)), [False]))
    while groups:
        pending = []
        for prepared, points, verified_flag in groups:
            for point in points:
                candidate, verified = schedule(prepared, point, verified_flag[0])
                verified_flag[0] = verified
                if candidate is not None:
                    yield candidate
                    pending.append((prepared, points, verified_flag))
                    break
        groups = pending

def _stream_plan(plan, stage, operation):
    from mindie_sim.rewrite.schema import StreamSchedule

    for edit in reversed(plan.edits):
        if isinstance(edit.operation, StreamSchedule):
            return replace(plan, edits=tuple(
                replace(item, operation=operation,
                        after=tuple(dict.fromkeys(item.after + _operation_dependencies(operation))))
                if item.edit_id == edit.edit_id else item
                for item in plan.edits
            ))
    return _with_operations(plan, stage, (operation,))

def _prefetch_operations(scenario, ops):
    from mindie_sim.rewrite.schema import WeightPrefetch

    paths = (datapath_for_die(device) for device in getattr(_machine(scenario), "dies", ()))
    capacity = min((dict(path.buffers).get("l1", 0) for path in paths), default=0)
    for op in ops:
        for port, anchor, byte_count in dict(op.metadata).get("prefetch_windows", ()):
            if 0 < byte_count <= capacity:
                yield WeightPrefetch(port, anchor, max_bytes=byte_count)

def solve_framework(
    dossier: ScenarioDossier, *, dual_report: DualSubgradientReport | None = None,
    dual_program=None, event_dag=None, dual_certificate=None, max_trials: int = DEFAULT_MAX_TRIALS,
) -> FrameworkSolveResult:
    from mindie_sim.rewrite.verify import SoundnessInversionError

    search = _TrialSearch(dossier, max_trials, dual_program=dual_program,
                          event_dag=event_dag, dual_certificate=dual_certificate)
    unmerge, mainline = _seed_plans(dossier)
    u_ns = _wall(_field(dossier, "unmerge_eval", "unmerge"))
    m_ns = _wall(_field(dossier, "mainline_eval", "mainline"))
    seed = mainline if m_ns <= u_ns else unmerge
    full = search.try_finalize(seed)
    if full is None or not _accepted(full) or _wall(full) > min(m_ns, u_ns):
        alternate = unmerge if seed == mainline else mainline
        full = search.try_finalize(alternate)
        seed = alternate
    if full is None or not _accepted(full) or _wall(full) > min(m_ns, u_ns):
        raise SoundnessInversionError("no fully verified plan meets the fastest baseline")
    search.seed(seed, full)
    alternate = unmerge if seed == mainline else mainline
    if alternate != seed:
        other = search.try_finalize(alternate)
        if other is not None and _accepted(other):
            if _wall(other) < _wall(full):
                search.seed(alternate, other)
            else:
                search.frontier[alternate.base_graph_sha256] = (alternate, other)

    loo_quota = search.remaining // (len(FRAMEWORK_STAGES) + 1)
    quota = (search.remaining - loo_quota + len(FRAMEWORK_STAGES) - 1) // len(FRAMEWORK_STAGES)
    before = search.remaining
    search.loo(mainline, m_ns, quota // 2, consider=True)
    def proposals(base, scenario, stage):
        ops = _atomic_ops(scenario)
        if stage == "F3_stream_core":
            stream_iter = _stream_candidates(scenario, ops)
            first = next(stream_iter, None)
            search.note_eligible(1 if first is not None else 0)
            if first is not None:
                yield _stream_plan(base, stage, first)
                for op in stream_iter:
                    yield _stream_plan(base, stage, op)
            return
        operations = tuple(_local_operations(scenario, ops, mainline) if stage == "F1_local_k1"
                           else _collective_operations(ops, dual_report) if stage == "F2_collective_dbo"
                           else _prefetch_operations(scenario, ops))
        search.note_eligible(len(operations))
        yield from (_with_operations(base, stage, (op,)) for op in operations)

    for index, stage in enumerate(FRAMEWORK_STAGES):
        plans = _interleave(proposals(base, scenario, stage) for base, scenario in _search_bases(dossier, search))
        limit = (search.remaining - loo_quota) // (len(FRAMEWORK_STAGES) - index)
        if index == 0:
            limit = max(0, quota - (before - search.remaining))
        search.stage(stage, plans, limit)
    impacts = search.loo(search.best_plan, _wall(search.best_eval), search.remaining)
    return FrameworkSolveResult(search.best_plan, search.best_eval, _wall(search.best_eval),
                                u_ns, m_ns, tuple(search.history), impacts, search.tried,
                                search_stats=search.stats())
