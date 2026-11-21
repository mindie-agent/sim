"""Read-only bounded oracle for rewrite screening and full replay.

[Role] Intern representative layers and reuse one prepared evaluation context.
[Invariants] Screening estimates use measured representatives and exact counts.
  Only full replay returns a full-occurrence verification receipt.
[Timeouts & Locks] Poll budgets between trials and at Engine deadline checks.
  The cache lock protects only LRU bookkeeping, never lowering or replay.
[Side-Effect Policy] Uses the analytical Engine and never starts device work.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import islice
from threading import Lock
from time import monotonic
from typing import Any, Literal

from mindie_sim.prediction.collectives.ring_pairwise import PairwiseCollective, RingCollective
from mindie_sim.prediction.evaluation import ACTIVE_DEADLINE_MONO, EvaluationContext
from mindie_sim.prediction.fusion_bridge import (
    CanonicalPlan, _canonical_hardware, add_prefetch, apply_prefetch_rewrite, settle_prefetch_rewrite,
)
from mindie_sim.prediction.fusion_pool import register_recipe
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.results import ReplayOptions, StepEvaluation, StepRequest
from mindie_sim.prediction.runtime.tile_solver import TilePipelineSpec
from mindie_sim.prediction.weight_prefetch import PrefetchDecision
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

from .apply import AppliedPlan, DEFAULT_STACK_ID, ResolvedScenario, _solve_tiles, apply_plan, resolve_scenario
from .schema import FuseRegion, RewritePlan, SelectTemplate, _digest_default, _json_digest, empty_plan, plan_sha256
from .verify import NS_PER_MS, VerificationReceipt, _combine_replays, _evaluation, verify_plan

DEFAULT_MAX_CANDIDATES = 8
PREFETCH_PRIORITY = 10
DEFAULT_MAX_ARCHETYPES = 5
DEFAULT_CACHE_ENTRIES = 16
MAX_RETAINED_OPERATORS = 16000
MAX_OVERLAP_TRACE_EVENTS = 8192
MAX_BOTTLENECK_OPS = 16
MAX_LAYER_CACHE_ENTRIES = 256

@dataclass(frozen=True, slots=True)
class ScenarioRequest:
    case: Case | Mapping[str, Any]
    step: StepRequest
    stack_id: str = DEFAULT_STACK_ID
    replay: ReplayOptions = ReplayOptions()
@dataclass(frozen=True, slots=True)
class EvalBudget:
    max_candidates: int = DEFAULT_MAX_CANDIDATES
    deadline_mono: float | None = None
    cancelled: Callable[[], bool] | None = None

    def __post_init__(self):
        if type(self.max_candidates) is not int or self.max_candidates < 0:
            raise ModelingError("rewrite_candidate_budget_invalid")
@dataclass(frozen=True, slots=True)
class StepArchetype:
    archetype_id: str
    representative_layer: int
    layer_indices: tuple[int, ...]
    multiplicity: int
    structural_key: str
    ops: tuple[OpSpec, ...]
@dataclass(frozen=True, slots=True)
class ScenarioHandle:
    scenario_digest: str
    scenario: ResolvedScenario
    archetypes: tuple[StepArchetype, ...]
    evaluation_context: EvaluationContext
    recipe_book: RecipeBook
    u_eval: StepEvaluation
    m_eval: StepEvaluation
    archetype_u_ns: int
    full_u_ns: int
    full_m_ns: int
    _representative_ns: tuple[int, ...] = field(default=(), repr=False)
    _replay: ReplayOptions = field(default_factory=ReplayOptions, repr=False)
    _mainline_book: RecipeBook | None = field(default=None, repr=False)
    _mainline_digest: str = field(default="", repr=False)
    _layer_cache: OrderedDict[str, tuple[int, StepEvaluation | None]] = field(
        default_factory=OrderedDict, repr=False, compare=False,
    )
@dataclass(frozen=True, slots=True)
class _OptimizationScenario:
    scenario: ResolvedScenario
    atomic_ops: tuple[OpSpec, ...]

    def __getattr__(self, name):
        return getattr(self.scenario, name)
@dataclass(frozen=True, slots=True)
class ScenarioDossier:
    scenario_digest: str
    handle: ScenarioHandle
    archetypes: tuple[StepArchetype, ...]
    u_wall_ns: int
    m_wall_ns: int
    archetype_coverage_ratio: float
    bottleneck_ops: tuple[tuple[str, float], ...]

    @property
    def resolved(self):
        # Constructive proposals name one representative per shared op identity.
        scenario = self.handle.scenario
        ops = {op.op_id: op for rank in reversed(scenario.atomic_ops) for op in rank.ops}
        return _OptimizationScenario(scenario, tuple(ops.values()))

    unmerge_eval = property(lambda self: self.handle.u_eval)
    mainline_eval = property(lambda self: self.handle.m_eval)
    @property
    def unmerge_plan(self):
        scenario = self.handle.scenario
        return empty_plan(scenario.base_graph_sha256, scenario.framing_sha256)

    @property
    def mainline_plan(self):
        return empty_plan(self.handle._mainline_digest or self.unmerge_plan.base_graph_sha256,
                          self.unmerge_plan.framing_sha256)
@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    plan_sha256: str
    status: Literal["ok", "invalid", "gate_failed", "cancelled"]
    scope: Literal["archetype", "full"]
    estimated_wall_ns: int | None
    archetype_wall_ns: int | None
    coverage_ratio: float
    verification: VerificationReceipt | None
    reason: str = ""

    @property
    def accepted(self):
        return self.status == "ok"
@dataclass(frozen=True, slots=True)
class BatchEvaluation:
    scenario_digest: str
    scope: Literal["archetype"]
    results: tuple[CandidateEvaluation, ...]
    elapsed_ms: float
@dataclass(frozen=True, slots=True)
class FullPlanEvaluation:
    scenario_digest: str
    plan_sha256: str
    scope: Literal["full"]
    step_eval: StepEvaluation
    wall_ns: int
    verification: VerificationReceipt
    applied: AppliedPlan
    elapsed_ms: float
class OracleCache:
    """Bounded LRU of complete scenario handles, including a graph-size limit."""

    def __init__(self, max_entries: int = DEFAULT_CACHE_ENTRIES):
        if type(max_entries) is not int or max_entries < 1:
            raise ModelingError("rewrite_cache_capacity_invalid")
        self.max_entries = max_entries
        self._entries = OrderedDict()
        self._requests = {}
        self._operators = 0
        self._lock = Lock()

    def get(self, scenario_digest: str) -> ScenarioHandle | None:
        with self._lock:
            digest = self._requests.get(scenario_digest, scenario_digest)
            entry = self._entries.get(digest)
            if entry is None:
                return None
            self._entries.move_to_end(digest)
            return entry[0]

    def put(self, handle: ScenarioHandle, *, request_key: str = "") -> None:
        operators = len(handle.u_eval.operators) + len(handle.m_eval.operators)
        if operators > MAX_RETAINED_OPERATORS:
            return
        key = handle.scenario_digest
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._operators -= previous[1]
                self._requests.pop(previous[2], None)
            self._entries[key] = (handle, operators, request_key)
            self._operators += operators
            if request_key:
                self._requests[request_key] = key
            while len(self._entries) > self.max_entries or self._operators > MAX_RETAINED_OPERATORS:
                _, (_, count, alias) = self._entries.popitem(last=False)
                self._operators -= count
                self._requests.pop(alias, None)

def intern_step_archetypes(ops: tuple[OpSpec, ...],
                           max_archetypes: int = DEFAULT_MAX_ARCHETYPES) -> tuple[StepArchetype, ...]:
    """Intern homogeneous interior layers and keep distinct stage boundaries."""
    if type(max_archetypes) is not int or max_archetypes < 1:
        raise ModelingError("rewrite_archetype_budget_invalid")
    by_layer = _layers(ops)
    stage_layers = defaultdict(list)
    for layer, members in by_layer.items():
        if layer >= 0:
            for stage in {op.stage for op in members}:
                stage_layers[stage].append(layer)
    bounds = {stage: (min(layers), max(layers)) for stage, layers in stage_layers.items()}
    buckets = {}
    for layer, members in sorted(by_layer.items()):
        key = (
            tuple((op.family, op.dtype, op.local_shape, op.stage, op.phase, op.stream)
                  for op in members),
            tuple((stage, layer == bounds[stage][0], layer == bounds[stage][1])
                  for stage in sorted({op.stage for op in members} & bounds.keys())),
        )
        buckets.setdefault(key, []).append(layer)
    groups = list(buckets.items())
    if len(groups) > max_archetypes:
        groups.sort(key=lambda item: (not any(first or last for _, first, last in item[0][1]),
                                     -len(item[1]), item[1][0]))
        groups = sorted(groups[:max_archetypes], key=lambda item: item[1][0])
    result = []
    for index, (key, layers) in enumerate(groups):
        rep = layers[0]
        result.append(StepArchetype(f"archetype:{index}", rep, tuple(layers), len(layers),
                                    _json_digest(key), tuple(by_layer[rep])))
    return tuple(result)
def _layers(ops):
    layers = defaultdict(list)
    for op in ops:
        layers[op.layer].append(op)
    return layers
def _deadline(budget, replay):
    return min((item for item in (replay.deadline_mono, budget.deadline_mono)
                if item is not None), default=None)
def _check_budget(budget, deadline):
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError("rewrite_oracle_deadline")
    if budget.cancelled is not None and budget.cancelled():
        raise TimeoutError("rewrite_oracle_cancelled")
def _report(case, graphs, context, book, replay, *, deadline=None, fast=False, trace_limit=0, streams=(), cut=False):
    external = {dep: 0.0 for ops in graphs.values()
                for dep in _external_dependencies(ops) if cut or dep == "analysis_input"}
    token = ACTIVE_DEADLINE_MONO.set(deadline)
    try:
        return graph_costs(
            case, graphs, recipes=book, external_inputs=external, evaluation=context,
            isolated=replay.isolated and not cut, include_submission=replay.include_submission,
            trace_limit=max(trace_limit, replay.event_limit) if not fast else 0,
            fast_span_only=fast, stream_plans=streams or None,
        )
    finally:
        ACTIVE_DEADLINE_MONO.reset(token)
def _cut_layer_key(graph, book, streams, need_eval):
    ranks = []
    for rank, ops in sorted(graph.items()):
        ordinals = {op.op_id: index for index, op in enumerate(ops)}
        op_sigs = []
        for op in ops:
            recipe = book.recipes.get(op.recipe) if op.recipe else None
            recipe_sig = (type(recipe).__name__, repr(recipe)) if recipe is not None else None
            op_sigs.append((
                op.op_id if need_eval else "", op.family, op.recipe, recipe_sig, op.dtype,
                op.local_shape, op.stage, op.phase, op.stream, op.flops, op.vector_ops,
                op.mte2_bytes, op.mte3_bytes, op.hbm_bytes, op.link_bytes, op.link_scope,
                op.matrix_flops, tuple(sorted(op.metadata)),
                tuple(ordinals.get(dep, -1) for dep in op.deps),
            ))
        ranks.append((rank, tuple(op_sigs)))
    stream_sigs = tuple(
        (plan.rank, plan.lanes, plan.budgets, plan.assignments, plan.events) for plan in streams
    )
    return _json_digest((tuple(ranks), stream_sigs, bool(need_eval)), default=_digest_default)


def _cache_layer_put(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > MAX_LAYER_CACHE_ENTRIES:
        cache.popitem(last=False)


def _external_dependencies(ops):
    ids = {op.op_id for op in ops}
    return {dep for op in ops for dep in op.deps if dep not in ids}

def _dossier(handle):
    cost = defaultdict(float)
    for row in handle.u_eval.operators:
        if row.start_ms is not None and row.end_ms is not None:
            cost[row.occurrence.op_id] += row.end_ms - row.start_ms
    bottlenecks = sorted(cost.items(), key=lambda item: (-item[1], item[0]))[:MAX_BOTTLENECK_OPS]
    return ScenarioDossier(handle.scenario_digest, handle, handle.archetypes,
                           handle.full_u_ns, handle.full_m_ns, _coverage(handle), tuple(bottlenecks))
def _coverage(handle):
    covered = {layer for item in handle.archetypes for layer in item.layer_indices}
    all_layers = {op.layer for entry in handle.scenario.atomic_ops for op in entry.ops}
    return len(covered) / len(all_layers) if all_layers else 1.0
def prepare_scenario(request: ScenarioRequest, *, budget: EvalBudget = EvalBudget(),
                     cache=None, spine=None, primitives=None) -> ScenarioDossier:
    deadline = _deadline(budget, request.replay)
    _check_budget(budget, deadline)
    case = Case.from_mapping(request.case)
    replay_frame = asdict(replace(request.replay, deadline_mono=None))
    if cache is not None and spine is None:
        request_key = _json_digest((case.to_dict(), asdict(request.step), request.stack_id,
                                    replay_frame, primitives), default=_digest_default)
        if (hit := cache.get(request_key)) is not None:
            return _dossier(hit)
    scenario = resolve_scenario(case, request.step, spine=spine, stack_id=request.stack_id,
                                 primitives=primitives)
    atomic, mainline = scenario.atomic_step, scenario.mainline_step or scenario.atomic_step
    book = RecipeBook.from_mapping(atomic.case.runtime.kernels)
    context = EvaluationContext.create(atomic.case, book, primitives)
    book_m = (book if mainline.case.runtime.kernels == atomic.case.runtime.kernels
              else RecipeBook.from_mapping(mainline.case.runtime.kernels))

    def evaluate(prepared, recipes, tier):
        _check_budget(budget, deadline)
        graphs = {entry.replica: entry.ops for entry in prepared.graph.executable}
        report = _report(prepared.case, graphs, context, recipes, request.replay, deadline=deadline,
                         streams=prepared.graph.streams)
        return _evaluation(scenario, graphs, report, prepared.graph_id, tier, prepared)

    u_eval = evaluate(atomic, book, "unmerge")
    m_eval = evaluate(mainline, book_m, "mainline")
    archetypes = intern_step_archetypes(scenario.atomic_ops[0].ops if scenario.atomic_ops else ())
    per_rank = {entry.replica: _layers(entry.ops) for entry in scenario.atomic_ops}
    spans = []
    layer_cache: OrderedDict[str, tuple[int, StepEvaluation | None]] = OrderedDict()
    for item in archetypes:
        _check_budget(budget, deadline)
        graphs = {rep: tuple(layers.get(item.representative_layer, ())) for rep, layers in per_rank.items()}
        report = _report(atomic.case, graphs, context, book, request.replay, deadline=deadline, fast=True, cut=True)
        span_ns = round(report["duration_ms"] * NS_PER_MS)
        spans.append(span_ns)
        _cache_layer_put(layer_cache, _cut_layer_key(graphs, book, (), False), (span_ns, None))
    digest = _json_digest((scenario.scenario_sha256, replay_frame))
    mainline_digest = _json_digest(("mainline", mainline.graph.executable, mainline.case.runtime.to_dict()),
                                   default=_digest_default)
    handle = ScenarioHandle(
        digest, scenario, archetypes, context, book, u_eval, m_eval,
        sum(span * item.multiplicity for span, item in zip(spans, archetypes)),
        round(u_eval.span_ms * NS_PER_MS), round(m_eval.span_ms * NS_PER_MS),
        tuple(spans), replace(request.replay, deadline_mono=None), book_m, mainline_digest, layer_cache,
    )
    if cache is not None and spine is None:
        cache.put(handle, request_key=request_key)
    return _dossier(handle)
def _baseline(handle, plan):
    scenario = handle.scenario
    if (plan.base_graph_sha256 != scenario.base_graph_sha256
            and plan.base_graph_sha256 == handle._mainline_digest and scenario.mainline_step):
        step = scenario.mainline_step
        scenario = replace(scenario, base_graph_sha256=handle._mainline_digest,
                           atomic_step=step, atomic_ops=step.graph.executable,
                           stream_plans=step.graph.streams)
        return scenario, handle._mainline_book or handle.recipe_book, handle.m_eval
    return scenario, handle.recipe_book, handle.u_eval
def _bind_prefetch(case, replica, ops, book, leases):
    if not leases:
        return ops, book
    hardware = _canonical_hardware(case, None, {})
    plan = CanonicalPlan(case, ops, None, hardware, (), book)
    for lease in leases:
        target = next((op for op in plan.ops if op.op_id == lease.target_op), None)
        if target is None:
            continue
        if not getattr(plan.book.select(target), "traffic_rewrite_capable", False):
            raise ModelingError(f"rewrite_prefetch_recipe_unbound:{target.op_id}")
        amount = lease.max_bytes
        weight = int(dict(target.metadata).get("weight_bytes", amount))
        decision = PrefetchDecision(
            target.op_id, (lease.port_name,), weight, amount, amount, weight - amount,
            amount, 0, hardware.hbm_bytes_per_ms, 0, True, "", (),
            priority=PREFETCH_PRIORITY,
        )
        added = add_prefetch(plan, decision, hardware)
        if added is None:
            raise ModelingError(f"rewrite_prefetch_duplicate_target:{target.op_id}")
        plan, prefetch_id = added
        settled = settle_prefetch_rewrite(plan, decision, prefetch_id, amount, hardware)
        if settled is None:
            raise ModelingError(f"rewrite_prefetch_settlement_failed:{target.op_id}")
        plan = apply_prefetch_rewrite(plan, settled, prefetch_id)
        plan = replace(plan, ops=tuple(replace(op, deps=(lease.window_anchor_op,))
                                       if op.op_id == prefetch_id else op for op in plan.ops))
    generated = {op.recipe for op in plan.ops if op.family in {"prefetch", "release"}}
    book = plan.book
    for name in generated:
        book = register_recipe(book, f"{name}:r{replica}", plan.book.recipes[name])
    return tuple(replace(op, recipe=f"{op.recipe}:r{replica}") if op.recipe in generated
                 else op for op in plan.ops), book

def _bind_replay(scenario, applied, book):
    """Resolve only variants actually implemented by the installed recipe book."""
    if applied.elided_d2h_ops:
        raise ModelingError("rewrite_device_group_list_recipe_unbound")
    if applied.prefetch_leases and applied.streams:
        raise ModelingError("rewrite_prefetch_stream_events_unbound")
    templates = {binding.target_op: binding for binding in applied.template_bindings}
    tiles = {binding.target_op: binding.tile_spec for binding in applied.tile_bindings}
    streams = {stream.rank: stream for stream in applied.streams}
    registered, graphs = {}, {}
    collective = {binding.target_op: binding for binding in applied.collective_rewrites}
    for entry in applied.executable:
        bound = []
        for op in entry.ops:
            if (binding := collective.get(op.op_id)) is not None:
                recipes = {"ring": RingCollective, "pairwise": PairwiseCollective}
                links = scenario.case.machine.links
                if len(links) != 1 or links[0].name != binding.topology_route or binding.mode not in recipes:
                    raise ModelingError(f"rewrite_collective_route_recipe_unbound:{op.op_id}")
                identifier = f"rewrite_collective:{op.op_id}"
                registered[identifier] = recipes[binding.mode]()
                op = replace(op, recipe=identifier)
            if op.op_id in templates and templates[op.op_id].template_id in book.recipes:
                op = replace(op, recipe=templates[op.op_id].template_id)
            if op.family == "rewrite_fused" and (len(next(
                    edit.operation.kernel_boundaries for edit in applied.plan.edits
                    if op.op_id == f"rewrite:{edit.edit_id}")) != 1 or op.recipe not in book.recipes):
                raise ModelingError(f"rewrite_fusion_recipe_unbound:{op.op_id}:{op.recipe}")
            tile = tiles.get(op.op_id)
            if isinstance(tile, TilePipelineSpec):
                recipe = book.select(op)
                shape = _solve_tiles(scenario, entry.replica, op, tile, streams.get(entry.replica))
                if all(hasattr(recipe, name) for name in ("tile_m", "tile_n", "tile_k", "slots")):
                    m, n, k = shape.tile_shape.m, shape.tile_shape.n, shape.tile_shape.k
                    identifier = f"rewrite_tile:{entry.replica}:{op.op_id}"
                    registered[identifier] = replace(recipe, tile_m=m, tile_n=n, tile_k=k, slots=shape.depth)
                    op = replace(op, recipe=identifier)
                else:
                    raise ModelingError(f"rewrite_tile_recipe_unbound:{op.op_id}")
            elif op.op_id in templates and templates[op.op_id].template_id not in book.recipes:
                raise ModelingError(f"rewrite_template_recipe_unbound:{op.op_id}")
            bound.append(op)
        graphs[entry.replica] = tuple(bound)
    for identifier, recipe in registered.items():
        book = register_recipe(book, identifier, recipe)
    for replica, ops in graphs.items():
        graphs[replica], book = _bind_prefetch(scenario.atomic_step.case, replica, ops, book,
                                              applied.prefetch_leases)
    return graphs, book
def _cut_plans(graphs, streams):
    plans = []
    for plan in streams:
        ids = {op.op_id for op in graphs[plan.rank]}
        events = []
        for event in plan.events:
            waits = tuple(target for target in event.wait_before if target in ids)
            if event.record_after in ids and waits:
                events.append(replace(event, wait_before=waits))
        plans.append(replace(plan, assignments=tuple(pair for pair in plan.assignments if pair[0] in ids),
                             events=tuple(events)))
    return tuple(plans)
def _estimate(handle, scenario, applied, book, baseline, budget, deadline):
    baseline_ns = round(baseline.span_ms * NS_PER_MS)
    if all(receipt.status == "noop" for receipt in applied.edit_receipts):
        return baseline_ns, handle.archetype_u_ns, baseline
    if baseline is handle.m_eval and baseline is not handle.u_eval:
        raise ModelingError("rewrite_archetypes_require_atomic_base")
    graphs, book = _bind_replay(scenario, applied, book)
    layers = {replica: _layers(ops) for replica, ops in graphs.items()}
    original = {entry.replica: _layers(entry.ops) for entry in scenario.atomic_ops}
    changed = sorted({layer for rank, by_layer in layers.items()
                      for layer in by_layer.keys() | original[rank].keys()
                      if by_layer.get(layer, ()) != original[rank].get(layer, ())}
                     | {op.layer for plan in applied.streams if plan not in scenario.stream_plans
                        for op in graphs[plan.rank]})
    reference = {layer: span for item, span in zip(handle.archetypes, handle._representative_ns)
                 for layer in item.layer_indices}
    archetype_ns = handle.archetype_u_ns
    rows = []
    overlap = any(len(plan.lanes) > 1 for plan in applied.streams)
    rebind = any(isinstance(edit.operation, (FuseRegion, SelectTemplate)) for edit in applied.plan.edits)
    need_eval = overlap or rebind
    for layer in changed:
        _check_budget(budget, deadline)
        if layer not in reference:
            raise ModelingError("rewrite_edited_layer_outside_archetype_coverage")
        graph = {rank: tuple(by_layer.get(layer, ())) for rank, by_layer in layers.items()}
        cut_streams = _cut_plans(graph, applied.streams)
        key = _cut_layer_key(graph, book, cut_streams, need_eval)
        cached = handle._layer_cache.get(key)
        if cached is not None:
            handle._layer_cache.move_to_end(key)
            span_ns, row_eval = cached
        else:
            report = _report(scenario.atomic_step.case, graph, handle.evaluation_context,
                             book, handle._replay, deadline=deadline, fast=not need_eval,
                             trace_limit=MAX_OVERLAP_TRACE_EVENTS if overlap else 0,
                             streams=cut_streams, cut=True)
            span_ns = round(report["duration_ms"] * NS_PER_MS)
            row_eval = (_evaluation(scenario, graph, report, applied.plan_sha256, "should_achieve")
                        if need_eval else None)
            _cache_layer_put(handle._layer_cache, key, (span_ns, row_eval))
        archetype_ns += span_ns - reference[layer]
        if row_eval is not None:
            rows.append(row_eval)
    evidence = _combine_replays(rows) if need_eval else baseline
    return max(0, baseline_ns + archetype_ns - handle.archetype_u_ns), archetype_ns, evidence

def evaluate_rewrites(
    scenario: ScenarioHandle, candidates: Sequence[RewritePlan], *, budget: EvalBudget = EvalBudget(),
) -> BatchEvaluation:
    started = monotonic()
    deadline = _deadline(budget, scenario._replay)
    coverage = _coverage(scenario)
    results = []
    seen = {}
    for plan in islice(candidates, budget.max_candidates):
        digest = ""
        try:
            digest = plan_sha256(plan)
            _check_budget(budget, deadline)
            if digest in seen:
                results.append(seen[digest])
                continue
            resolved, book, baseline = _baseline(scenario, plan)
            applied = apply_plan(plan, resolved)
            wall, archetype_wall, evidence = _estimate(scenario, resolved, applied, book,
                                                       baseline, budget, deadline)
            verification = verify_plan(resolved, applied, step_eval=evidence, wall_ns=wall, scope="archetype")
            _check_budget(budget, deadline)
            failure = next((gate.reason for gate in verification.gates if not gate.passed), "")
            result = CandidateEvaluation(digest, "gate_failed" if failure else "ok", "archetype",
                                         wall, archetype_wall, coverage, verification, failure)
            seen[digest] = result
        except TimeoutError as error:
            result = CandidateEvaluation(digest, "cancelled", "archetype", None, None, coverage, None, str(error))
        except (ValueError, TypeError) as error:
            result = CandidateEvaluation(digest, "invalid", "archetype", None, None, coverage, None, str(error))
        results.append(result)
    return BatchEvaluation(scenario.scenario_digest, "archetype", tuple(results), (monotonic() - started) * 1000)

def finalize_plan(
    scenario: ScenarioHandle, plan: RewritePlan, *, budget: EvalBudget = EvalBudget(),
    dual_certificate=None, dual_program=None, event_dag=None,
) -> FullPlanEvaluation:
    started = monotonic()
    deadline = _deadline(budget, scenario._replay)
    _check_budget(budget, deadline)
    resolved, book, baseline = _baseline(scenario, plan)
    applied = apply_plan(plan, resolved)
    graphs, book = _bind_replay(resolved, applied, book)
    overlap = any(len(stream.lanes) > 1 for stream in applied.streams)
    report = _report(resolved.atomic_step.case, graphs, scenario.evaluation_context, book, scenario._replay,
                     deadline=deadline, streams=applied.streams,
                     trace_limit=MAX_OVERLAP_TRACE_EVENTS if overlap else 0)
    step = _evaluation(resolved, graphs, report, applied.plan_sha256, baseline.tier if not plan.edits
                       else "should_achieve", resolved.atomic_step if not plan.edits else None)
    wall = round(step.span_ms * NS_PER_MS)
    verification = verify_plan(resolved, applied, step_eval=step, wall_ns=wall,
                               dual_certificate=dual_certificate, dual_program=dual_program,
                               event_dag=event_dag, baseline_u_ns=scenario.full_u_ns,
                               baseline_m_ns=scenario.full_m_ns)
    return FullPlanEvaluation(scenario.scenario_digest, applied.plan_sha256, "full", step, wall,
                              verification, applied, (monotonic() - started) * 1000)

__all__ = [
    "ScenarioRequest", "EvalBudget", "StepArchetype", "ScenarioHandle", "ScenarioDossier",
    "CandidateEvaluation", "BatchEvaluation", "FullPlanEvaluation", "OracleCache",
    "intern_step_archetypes", "prepare_scenario", "evaluate_rewrites", "finalize_plan",
]
