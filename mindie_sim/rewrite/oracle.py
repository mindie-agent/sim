"""Read-only bounded oracle for rewrite screening and full replay.

[Role] Intern representative layers and reuse one prepared evaluation context.
[Invariants] Screening uses measured representatives and exact counts.
  Prefetch-only screening prices DMA service without overlap credit.
  Only full replay returns a full-occurrence verification receipt.
[Timeouts & Locks] Poll budgets between trials and at Engine deadline checks.
  The cache lock protects prepared data and LRU bookkeeping.
[Side-Effect Policy] Uses the analytical Engine and never starts device work.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from itertools import islice
from math import ceil, fsum, isclose
from threading import Lock
from time import monotonic
from typing import Any, Literal

from mindie_sim.catalog.ops import comm_op
from mindie_sim.prediction.collectives.ring_pairwise import PairwiseCollective, RingCollective
from mindie_sim.prediction.evaluation import ACTIVE_DEADLINE_MONO, EvaluationContext
from mindie_sim.prediction.execution.split_barrier import DEFAULT_SPLIT_DTYPE_BYTES
from mindie_sim.prediction.execution.runner import EAGER_MOE_DISPATCH_RECIPES
from mindie_sim.prediction.operators.grouped_moe.expected_grouped import ExpectedGroupedMatmul
from mindie_sim.prediction.fusion.bridge import _canonical_hardware
from mindie_sim.prediction.fusion.pool import register_recipe
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.operators.moe_transport.moe_callers import RoutingPolicy
from mindie_sim.prediction.program.primitives import PrimitiveStore
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.prediction.results import ReplayOptions, StepEvaluation, StepRequest
from mindie_sim.prediction.runtime.tile_solver import TilePipelineSpec
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import RankOps

from ._oracle_prefetch import bind_prefetch_ops, bind_prefetch_streams, primitive_table_digest_token
from .apply import AppliedPlan, DEFAULT_STACK_ID, ResolvedScenario, _solve_tiles, apply_plan, resolve_scenario
from .schema import (
    FuseRegion, PipelineTiles, QuantizeContract, RewritePlan, SelectTemplate, WeightPrefetch, _digest_default,
    _json_digest, _references, empty_plan, plan_sha256,
)
from .verify import NS_PER_MS, VerificationReceipt, _COHORT_PREFIX, _combine_replays, _evaluation, _work_totals, verify_plan

DEFAULT_MAX_CANDIDATES = 8
DEFAULT_MAX_ARCHETYPES = 5
DEFAULT_CACHE_ENTRIES = 16
MAX_RETAINED_OPERATORS = 16000
MAX_OVERLAP_TRACE_EVENTS = 8192
MAX_BOTTLENECK_OPS = 16
MAX_SCREEN_CACHE_ENTRIES = 256
GROUPED_MATRIX_RANK = 3
@dataclass(frozen=True, slots=True)
class ScenarioRequest:
    case: Case | Mapping[str, Any]
    step: StepRequest
    stack_id: str = DEFAULT_STACK_ID
    replay: ReplayOptions = ReplayOptions()
    primitive_table: Any = None
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
class _Seed:
    scenario: ResolvedScenario
    book: RecipeBook
    evaluation: StepEvaluation
    archetypes: tuple[StepArchetype, ...]
    layers: dict
    sites: dict
    work: dict
    layer_work: dict
    reference: dict
    total_ns: int

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
    _unmerge: _Seed = field(repr=False)
    _mainline: _Seed = field(repr=False)
    _replay: ReplayOptions = field(repr=False)
    _layer_cache: OrderedDict = field(default_factory=OrderedDict, repr=False, compare=False)
    _candidates: OrderedDict = field(default_factory=OrderedDict, repr=False, compare=False)
    _certificate: dict = field(default_factory=dict, repr=False, compare=False)
    _hardware_cache: list = field(default_factory=list, repr=False, compare=False)
    _cache_lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    @property
    def _prefetch_hardware(self):
        with self._cache_lock:
            if not self._hardware_cache:
                self._hardware_cache.append(_canonical_hardware(self.scenario.atomic_step.case, None, {}))
            return self._hardware_cache[0]

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
        scenario = self.handle._mainline.scenario
        return empty_plan(scenario.base_graph_sha256, scenario.framing_sha256)
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
    region_summaries: tuple = ()
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
        if isinstance(handle.evaluation_context.cost.primitives, PrimitiveStore):
            return
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
    from mindie_sim.optimization.dsl_solver import _replay_ops

    external = {dep: 0.0 for ops in graphs.values()
                for dep in _external_dependencies(ops) if cut or dep == "analysis_input"}
    token = ACTIVE_DEADLINE_MONO.set(deadline)
    try:
        return graph_costs(
            case, {rank: _replay_ops(ops) for rank, ops in graphs.items()},
            recipes=book, external_inputs=external, evaluation=context,
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
                op.matrix_flops, op.layout, op.source_boundary, op.returns,
                tuple(sorted(op.metadata, key=lambda pair: pair[0])),
                tuple(ordinals.get(dep, -1) for dep in op.deps),
            ))
        ranks.append((rank, tuple(op_sigs)))
    return _json_digest((tuple(ranks), streams, bool(need_eval)), default=_digest_default)

def _memo_put(cache, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > MAX_SCREEN_CACHE_ENTRIES:
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
    seed = handle._unmerge
    return len(seed.reference) / len(seed.layers) if seed.layers else 1.0
def prepare_scenario(request: ScenarioRequest, *, budget: EvalBudget = EvalBudget(),
                     cache=None, spine=None, primitives=None, primitive_table=None) -> ScenarioDossier:
    deadline = _deadline(budget, request.replay)
    _check_budget(budget, deadline)
    case = Case.from_mapping(request.case)
    prim_table = primitive_table if primitive_table is not None else request.primitive_table
    if isinstance(primitives, PrimitiveStore):
        cache = None
    replay_frame = asdict(replace(request.replay, deadline_mono=None))
    if cache is not None and spine is None:
        request_key = _json_digest((case.to_dict(), asdict(request.step), request.stack_id,
                                    replay_frame, primitives, primitive_table_digest_token(prim_table)),
                                   default=_digest_default)
        if (hit := cache.get(request_key)) is not None:
            return _dossier(hit)
    scenario = resolve_scenario(case, request.step, spine=spine, stack_id=request.stack_id,
                                 primitives=primitives, primitive_table=prim_table)
    atomic, mainline = scenario.atomic_step, scenario.mainline_step or scenario.atomic_step
    book = RecipeBook.from_mapping(atomic.case.runtime.kernels)
    context = EvaluationContext.create(atomic.case, book, primitives, primitive_table=prim_table)
    book_m = (book if mainline.case.runtime.kernels == atomic.case.runtime.kernels
              else RecipeBook.from_mapping(mainline.case.runtime.kernels))

    def prepare_seed(resolved, recipes, tier):
        prepared = resolved.atomic_step
        _check_budget(budget, deadline)
        graphs = {entry.replica: entry.ops for entry in prepared.graph.executable}
        report = _report(prepared.case, graphs, context, recipes, request.replay, deadline=deadline,
                         streams=prepared.graph.streams,
                         trace_limit=MAX_OVERLAP_TRACE_EVENTS if prepared.graph.streams else 0)
        evaluation = _evaluation(resolved, graphs, report, prepared.graph_id, tier, prepared)
        archetypes = intern_step_archetypes(next(iter(graphs.values()), ()))
        layers, sites, reference = {}, defaultdict(set), {}
        for replica, ops in graphs.items():
            for layer, members in _layers(ops).items():
                layers.setdefault(layer, dict.fromkeys(graphs, ()))[replica] = tuple(members)
            for op in ops:
                meta = dict(op.metadata)
                for key in (op.op_id, op.family, meta.get("model_scope"), meta.get("quant_scope")):
                    if key is not None:
                        sites[key].add(op.layer)
        for item in archetypes:
            _check_budget(budget, deadline)
            cut = layers[item.representative_layer]
            streams = _cut_plans(cut, resolved.stream_plans)
            report = _report(prepared.case, cut, context, recipes, request.replay, deadline=deadline,
                             fast=True, cut=True, streams=streams)
            reference.update(dict.fromkeys(item.layer_indices, round(report["duration_ms"] * NS_PER_MS)))
        return _Seed(resolved, recipes, evaluation, archetypes, layers, dict(sites),
                     {rank: _work_totals(ops) for rank, ops in graphs.items()},
                     {layer: {rank: _work_totals(ops) for rank, ops in cut.items()} for layer, cut in layers.items()},
                     reference, sum(reference.values()))

    digest = _json_digest((scenario.scenario_sha256, replay_frame))
    mainline_digest = _json_digest(("mainline", mainline.graph.executable, mainline.case.runtime.to_dict()),
                                   default=_digest_default)
    u = prepare_seed(scenario, book, "unmerge")
    resolved_m = replace(scenario, base_graph_sha256=mainline_digest,
                         atomic_step=mainline, atomic_ops=mainline.graph.executable,
                         stream_plans=mainline.graph.streams)
    m = prepare_seed(resolved_m, book_m, "mainline")
    handle = ScenarioHandle(digest, scenario, u.archetypes, context, book, u.evaluation, m.evaluation,
                            u.total_ns, round(u.evaluation.span_ms * NS_PER_MS),
                            round(m.evaluation.span_ms * NS_PER_MS), u, m,
                            replace(request.replay, deadline_mono=None))
    if cache is not None and spine is None:
        cache.put(handle, request_key=request_key)
    return _dossier(handle)
def _baseline(handle, plan):
    return (handle._mainline if plan.base_graph_sha256 == handle._mainline.scenario.base_graph_sha256
            else handle._unmerge)

@dataclass(frozen=True, slots=True)
class _DboGrouped:
    recipe: ExpectedGroupedMatmul
    cohorts: tuple[tuple[float, float], ...]

    def lower(self, context):
        op = context.op
        meta = dict(op.metadata)
        _, n, k = op.local_shape
        work = Work(dict(op.matrix_flops or ((op.dtype, op.flops),)),
                    {k: v for k, v in context.native_work.items() if k != "cube"})
        original_weight = meta.get("weight_bytes", 0)
        routed_weight = meta.get("routed_weight_read_bytes", original_weight)
        shared_weight = original_weight - routed_weight
        stages = []
        for cohort, (expected_rows, active) in enumerate(self.cohorts):
            share = expected_rows / meta["expected_rows"]
            name = f"{_COHORT_PREFIX}_{cohort}"
            ratio = active / meta["expected_active_experts"]
            cold_shared = shared_weight if cohort == 0 else 0
            weight = routed_weight * ratio + cold_shared
            scaled = work.scaled(share)
            traffic = {role: (work.resources.get(role, 0) - original_weight) * share + weight
                       for role in ("hbm", "mte2")}
            phase = context.for_phase(Work(scaled.matrix, {**scaled.resources, **traffic}), kernel=name)
            values = dict(phase.op.metadata, expected_rows=expected_rows, expected_active_experts=active,
                          weight_bytes=weight, routed_weight_read_bytes=routed_weight * ratio)
            for key in ("packed_weight_bytes", "weight_payload_bytes"):
                if key in meta and original_weight:
                    values[key] = meta[key] * weight / original_weight
            if "activation_scale_bytes" in meta:
                values["activation_scale_bytes"] = meta["activation_scale_bytes"] * share
            child = replace(phase.op, local_shape=(ceil(expected_rows), n, k), metadata=tuple(values.items()))
            program = self.recipe.lower(replace(phase, op=child, record_mapping=None))
            stages.append(Stage(name, program=program))
        return KernelProgram(tuple(stages), minimum_ms=context.minimum_ms)

def _cohort_moments(case, op, meta):
    model, parallel = case.model, case.runtime.parallel
    layer = next((layer for layer in model.layer_plans if layer.index == op.layer), None)
    if layer is None or not layer.num_experts or meta.get("physical_slot_rows"):
        raise ModelingError(f"rewrite_dbo_routing_unbound:{op.op_id}")
    policy = RoutingPolicy()
    ids = meta.get("representative_expert_ids")

    def moments(tokens):
        if ids:
            rows, active = policy.expert_moments(model, layer, tokens)
            return fsum(rows[i] for i in ids), fsum(active[i] for i in ids)
        return policy.local(model, layer, parallel, tokens, owner=meta.get("ep_rank"))

    per_token, _ = moments(1)
    tokens = round(meta["expected_rows"] / per_token) if per_token else 0
    if tokens < 2 or not all(isclose(a, b) for a, b in zip(
        moments(tokens), (meta["expected_rows"], meta["expected_active_experts"])
    )):
        raise ModelingError(f"rewrite_dbo_routing_unbound:{op.op_id}")
    return moments(tokens // 2), moments(tokens - tokens // 2)

def _device_counts(ops, targets, book, case, include_submission):
    targets = set(targets)
    producers = {}
    supported = {"grouped_gemm", "grouped_moe", "mc2_protocol", "alltoall", "alltoallv"}
    updated = []
    known = {op.op_id for op in ops}
    for op in ops:
        domain = op.stage, op.layer
        meta = dict(op.metadata)
        if (op.op_id in targets or meta.get("device_group_list") is True
                or any(producers.get(dep) == domain for dep in op.deps)):
            producers[op.op_id] = domain
        if op.op_id in producers and (
            op.family in supported or op.recipe in EAGER_MOE_DISPATCH_RECIPES
        ):
            meta.update(device_group_list=True, requires_host_split_sizes=False, eager_d2h_sync=False)
            if op.recipe in EAGER_MOE_DISPATCH_RECIPES:
                # The existing dispatch recipe consumes device counts. Its eager
                # entry-point ID otherwise inserts a host split-size rendezvous.
                identifier = f"rewrite_device_counts:{op.recipe}"
                book = register_recipe(book, identifier, book.select(op))
                op = replace(op, recipe=identifier)
                ep = int(meta.get("moe_ep_size") or case.runtime.parallel.ep)
                if (include_submission and op.phase == "prefill" and ep > 1
                        and meta.get("graph_replay_scope", "none") == "none"
                        and case.machine.host_solo_bw_gbps("d2h")):
                    # Keep the device rendezvous that the implicit host barrier
                    # previously scheduled before it performed the D2H copy.
                    gather = op.op_id.rsplit(".", 1)[0] + ".moe_histogram_gather"
                    if gather not in known:
                        size = int(meta.get("split_dtype_bytes") or DEFAULT_SPLIT_DTYPE_BYTES)
                        updated.append(comm_op(
                            op_id=gather, deps=op.deps, stage=op.stage, layer=op.layer,
                            requests=op.requests, family="allgather", phase=op.phase, recipe="ring_allgather",
                            link_bytes=(ep - 1) * ep * size, shape=(ep, ep), metadata=(("group_axis", "ep"),),
                        ))
                    op = replace(op, deps=tuple(dict.fromkeys((*op.deps, gather))))
            op = replace(op, metadata=tuple(meta.items()))
        updated.append(op)
    return tuple(updated), book

def _bind_replay(scenario, applied, book, handle):
    """Bind the selected executable recipes and collect catalog audit summaries."""
    from mindie_sim.optimization.dsl_solver import catalog_recipe_id, replay_catalog_binding
    templates = {binding.target_op: binding for binding in applied.template_bindings}
    atomic = {entry.replica: entry for entry in scenario.atomic_ops}
    summaries = []
    tiles = {binding.target_op: binding for binding in applied.tile_bindings}
    streams = {stream.rank: stream for stream in applied.streams}
    registered, graphs = {}, {}
    topology = handle.evaluation_context.bindings.topology
    collective = {binding.target_op: binding for binding in applied.collective_rewrites}
    for entry in applied.executable:
        bound = []
        ops, book = (_device_counts(entry.ops, applied.elided_d2h_ops, book,
                                    scenario.atomic_step.case, handle._replay.include_submission)
                     if applied.elided_d2h_ops else (entry.ops, book))
        for op in ops:
            template = templates.get(op.op_id)
            catalog = replay_catalog_binding(scenario, atomic[entry.replica], op, template,
                                             applied, streams.get(entry.replica))
            if catalog:
                identifier = catalog_recipe_id(entry.replica, op, template, catalog[1])
                registered[identifier], summary = catalog
                summaries.append(summary)
                op = replace(op, recipe=identifier)
            elif template:
                if template.template_id not in book.recipes:
                    raise ModelingError(f"rewrite_template_recipe_unbound:{op.op_id}:{template.template_id}")
                op = replace(op, recipe=template.template_id)
            if (binding := collective.get(op.op_id)) is not None:
                recipes = {"ring": RingCollective, "pairwise": PairwiseCollective}
                meta = dict(op.metadata)
                groups = topology.groups(meta.get("group_axis", "tp"), entry.replica,
                                         meta.get("source_stage", op.stage), metadata=meta)
                pairs = ((a, b) for group in groups for a, b in (
                    zip(group, group[1:] + group[:1]) if binding.mode == "ring"
                    else ((a, b) for a in group for b in group)) if a != b)
                if (binding.mode not in recipes or binding.topology_route not in topology.links
                        or any(not any(name == binding.topology_route for name, _, _ in topology.route(a, b))
                               for a, b in pairs)):
                    raise ModelingError(f"rewrite_collective_route_recipe_unbound:{op.op_id}")
                identifier = f"rewrite_collective:{op.op_id}:{binding.mode}:{binding.topology_route}"
                registered[identifier] = recipes[binding.mode]()
                op = replace(op, recipe=identifier)
            if op.family == "rewrite_fused" and (len(next(
                    edit.operation.kernel_boundaries for edit in applied.plan.edits
                    if op.op_id == f"rewrite:{edit.edit_id}")) != 1 or op.recipe not in book.recipes and not catalog):
                raise ModelingError(f"rewrite_fusion_recipe_unbound:{op.op_id}:{op.recipe}")
            binding = tiles.get(op.op_id)
            tile = binding.tile_spec if binding else None
            if binding is None:
                bound.append(op)
                continue
            recipe = registered.get(op.recipe) or book.select(op)
            if isinstance(tile, TilePipelineSpec):
                shape, _ = _solve_tiles(scenario, entry.replica, op, tile, streams.get(entry.replica))
                matrix = recipe.matrix if isinstance(recipe, ExpectedGroupedMatmul) else recipe
                if not all(hasattr(matrix, name) for name in ("tile_m", "tile_n", "tile_k", "slots")):
                    raise ModelingError(f"rewrite_tile_recipe_unbound:{op.op_id}")
                m, n, k = shape.tile_shape.m, shape.tile_shape.n, shape.tile_shape.k
                matrix = replace(matrix, tile_m=m, tile_n=n, tile_k=k, slots=shape.depth)
                recipe = replace(recipe, matrix=matrix) if isinstance(recipe, ExpectedGroupedMatmul) else matrix
            if binding.cohort_split == "dbo_2cohort":
                meta = dict(op.metadata)
                if (not isinstance(recipe, ExpectedGroupedMatmul) or len(op.local_shape) != GROUPED_MATRIX_RANK
                        or op.local_shape[0] < 2 or meta.get("expert_rows") or meta.get("rows_by_expert")):
                    raise ModelingError(f"rewrite_dbo_recipe_unbound:{op.op_id}")
                recipe = _DboGrouped(recipe, _cohort_moments(scenario.atomic_step.case, op, meta))
            identifier = f"rewrite_tile:{entry.replica}:{op.op_id}"
            # Register the base matrix; selection will apply its grouped adapter once.
            registered[identifier] = recipe.matrix if isinstance(recipe, ExpectedGroupedMatmul) else recipe
            bound.append(replace(op, recipe=identifier))
        graphs[entry.replica] = tuple(bound)
    for identifier, recipe in registered.items():
        book = register_recipe(book, identifier, recipe)
    if applied.prefetch_leases:
        for replica, ops in graphs.items():
            graphs[replica], book = bind_prefetch_ops(scenario.atomic_step.case, replica, ops, book,
                                                      applied.prefetch_leases, handle._prefetch_hardware)
    bound_streams = bind_prefetch_streams(graphs, applied.streams, applied.prefetch_leases)
    return graphs, book, tuple(summaries), bound_streams
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
def _screen_scenario(seed, plan):
    scenario = seed.scenario
    if scenario.stream_plans or not plan.edits:
        return scenario, seed.work
    layers = set()
    for edit in plan.edits:
        operation = edit.operation
        if isinstance(operation, QuantizeContract):
            for site in set(operation.scope) - set(operation.exclusions):
                layers.update(seed.sites.get(site, ()))
        elif isinstance(operation, (SelectTemplate, PipelineTiles, WeightPrefetch)):
            for ref in _references(operation):
                if ref.kind != "atom":
                    return scenario, seed.work
                layers.update(seed.sites.get(ref.id, ()))
        else:
            return scenario, seed.work
    ranks = tuple(RankOps(entry.replica, tuple(
        op for layer in sorted(layers) for op in seed.layers[layer][entry.replica]
    )) for entry in scenario.atomic_ops)
    work = seed.layer_work[next(iter(layers))] if len(layers) == 1 else None
    return replace(scenario, atomic_ops=ranks), work

def _replay_cut(handle, seed, graph, book, streams, need_eval, trace, deadline):
    key = (seed.scenario.base_graph_sha256, _cut_layer_key(graph, book, streams, need_eval), trace)
    with handle._cache_lock:
        cached = handle._layer_cache.get(key)
        if cached is not None:
            handle._layer_cache.move_to_end(key)
            return cached
    scenario = seed.scenario
    report = _report(scenario.atomic_step.case, graph, handle.evaluation_context, book, handle._replay,
                     deadline=deadline, fast=not need_eval, trace_limit=trace, streams=streams, cut=True)
    span = round(report["duration_ms"] * NS_PER_MS)
    evaluation = _evaluation(scenario, graph, report, key[1], "should_achieve") if need_eval else None
    with handle._cache_lock:
        _memo_put(handle._layer_cache, key, (span, evaluation))
    return span, evaluation

def _estimate(handle, seed, scenario, applied, budget, deadline):
    baseline = seed.evaluation
    baseline_ns = round(baseline.span_ms * NS_PER_MS)
    # Quantization declarations are compatibility assertions against the
    # prepared representation. Their annotations leave the physical bound intact.
    if (all(receipt.status == "noop" for receipt in applied.edit_receipts)
            or all(isinstance(edit.operation, QuantizeContract) for edit in applied.plan.edits)):
        return baseline_ns, seed.total_ns, baseline, baseline, None, ()
    graphs, book, summaries, bound_streams = _bind_replay(scenario, applied, seed.book, handle)
    layers = {replica: _layers(ops) for replica, ops in graphs.items()}
    domain = {op.layer for rank in scenario.atomic_ops for op in rank.ops}
    changed_streams = {plan.rank for plan in bound_streams if plan not in scenario.stream_plans}
    changed = sorted(layer for layer in domain if changed_streams or any(
        tuple(by_layer.get(layer, ())) != seed.layers[layer][rank] for rank, by_layer in layers.items()
    ))
    if not seed.reference.keys() >= set(changed):
        raise ModelingError("rewrite_edited_layer_outside_archetype_coverage")
    if (applied.prefetch_leases and not bound_streams and not handle._replay.include_submission
            and all(isinstance(edit.operation, WeightPrefetch) for edit in applied.plan.edits)):
        # Screen the extra DMA service with the copies serialized and no overlap credit.
        copied = fsum(op.hbm_bytes for ops in graphs.values() for op in ops if op.family == "prefetch")
        extra_ns = round(copied / handle._prefetch_hardware.hbm_bytes_per_ms * NS_PER_MS)
        return baseline_ns + extra_ns, seed.total_ns + extra_ns, baseline, baseline, None, summaries
    archetype_ns = seed.total_ns
    rows, comparisons = [], []
    overlap = any(len(plan.lanes) > 1 for plan in bound_streams) or any(
        tile.cohort_split == "dbo_2cohort" for tile in applied.tile_bindings)
    rebind = any(isinstance(edit.operation, (FuseRegion, SelectTemplate)) for edit in applied.plan.edits)
    need_eval = overlap or rebind
    for layer in changed:
        _check_budget(budget, deadline)
        graph = {rank: tuple(by_layer.get(layer, ())) for rank, by_layer in layers.items()}
        span, row = _replay_cut(handle, seed, graph, book, _cut_plans(graph, bound_streams),
                                need_eval, MAX_OVERLAP_TRACE_EVENTS if overlap else 0, deadline)
        reference = seed.reference[layer]
        if rebind:
            original = seed.layers[layer]
            reference, comparison = _replay_cut(handle, seed, original, seed.book,
                _cut_plans(original, scenario.stream_plans), True, 0, deadline)
            comparisons.append(comparison)
        archetype_ns += span - reference
        if row is not None:
            rows.append(row)
    return (max(0, baseline_ns + archetype_ns - seed.total_ns), archetype_ns,
            _combine_replays(rows) or baseline, _combine_replays(comparisons) or baseline,
            (graphs, book, seed.book, handle.evaluation_context, deadline), summaries)

def evaluate_rewrites(
    scenario: ScenarioHandle, candidates: Sequence[RewritePlan], *, budget: EvalBudget = EvalBudget(),
) -> BatchEvaluation:
    started = monotonic()
    deadline = _deadline(budget, scenario._replay)
    coverage = _coverage(scenario)
    results = []
    for plan in islice(candidates, budget.max_candidates):
        digest = ""
        try:
            digest = plan_sha256(plan)
            _check_budget(budget, deadline)
            with scenario._cache_lock:
                hit = scenario._candidates.get(digest)
                if hit is not None:
                    scenario._candidates.move_to_end(digest)
            if hit is not None:
                _check_budget(budget, deadline)
                results.append(hit)
                continue
            seed = _baseline(scenario, plan)
            resolved, work = _screen_scenario(seed, plan)
            applied = apply_plan(plan, resolved)
            wall, archetype_wall, evidence, comparison, service_replay, summaries = _estimate(
                scenario, seed, resolved, applied, budget, deadline)
            verification = verify_plan(resolved, applied, step_eval=evidence, wall_ns=wall,
                scope="archetype", baseline_eval=comparison, baseline_work=work, service_replay=service_replay,
                region_summaries=summaries)
            _check_budget(budget, deadline)
            failure = next((gate.reason for gate in verification.gates if not gate.passed), "")
            coverage = len(seed.reference) / len(seed.layers) if seed.layers else 1.0
            result = CandidateEvaluation(digest, "gate_failed" if failure else "ok", "archetype",
                                         wall, archetype_wall, coverage, verification, failure)
        except TimeoutError as error:
            result = CandidateEvaluation(digest, "cancelled", "archetype", None, None, coverage, None, str(error))
        except (ValueError, TypeError) as error:
            result = CandidateEvaluation(digest, "invalid", "archetype", None, None, coverage, None, str(error))
        if digest and result.status != "cancelled":
            with scenario._cache_lock:
                _memo_put(scenario._candidates, digest, result)
        results.append(result)
    return BatchEvaluation(scenario.scenario_digest, "archetype", tuple(results), (monotonic() - started) * 1000)
def finalize_plan(
    scenario: ScenarioHandle, plan: RewritePlan, *, budget: EvalBudget = EvalBudget(),
    dual_certificate=None, dual_program=None, event_dag=None, primitive_table=None, rate_mode="engine_ceiling",
) -> FullPlanEvaluation:
    started = monotonic()
    deadline = _deadline(budget, scenario._replay)
    _check_budget(budget, deadline)
    seed = _baseline(scenario, plan)
    resolved, baseline = seed.scenario, seed.evaluation
    applied = apply_plan(plan, resolved)
    graphs, book, summaries, bound_streams = _bind_replay(resolved, applied, seed.book, scenario)
    overlap = any(len(stream.lanes) > 1 for stream in bound_streams) or any(
        tile.cohort_split == "dbo_2cohort" for tile in applied.tile_bindings)
    report = _report(resolved.atomic_step.case, graphs, scenario.evaluation_context, book, scenario._replay,
                     deadline=deadline, streams=bound_streams,
                     trace_limit=MAX_OVERLAP_TRACE_EVENTS if overlap else 0)
    step = _evaluation(resolved, graphs, report, applied.plan_sha256, baseline.tier if not plan.edits
                       else "should_achieve", resolved.atomic_step if not plan.edits else None)
    wall = round(step.span_ms * NS_PER_MS)
    if dual_certificate is None and dual_program is None and event_dag is None:
        from mindie_sim.optimization.dual_bound import build_dual_inputs_from_scenario, certify_lower_bound
        cacheable = not summaries and primitive_table is None and rate_mode == "engine_ceiling"
        cached = scenario._certificate.get(resolved.base_graph_sha256) if cacheable else None
        if cached is None:
            dual_program, event_dag, _ = build_dual_inputs_from_scenario(
                resolved, applied=applied, region_summaries=summaries, primitive_table=primitive_table, rate_mode=rate_mode)
            dual_certificate, _ = certify_lower_bound(dual_program, event_dag)
            if cacheable:
                with scenario._cache_lock:
                    scenario._certificate[resolved.base_graph_sha256] = dual_program, event_dag, dual_certificate
        else:
            dual_program, event_dag, dual_certificate = cached
    _check_budget(budget, deadline)
    verification = verify_plan(resolved, applied, step_eval=step, wall_ns=wall,
                               dual_certificate=dual_certificate, dual_program=dual_program,
                               event_dag=event_dag, baseline_u_ns=scenario.full_u_ns,
                               baseline_m_ns=scenario.full_m_ns, baseline_eval=baseline,
                               baseline_work=seed.work, region_summaries=summaries, primitive_table=primitive_table, rate_mode=rate_mode,
                               service_replay=(graphs, book, seed.book, scenario.evaluation_context, deadline))
    return FullPlanEvaluation(scenario.scenario_digest, applied.plan_sha256, "full", step, wall,
                              verification, applied, (monotonic() - started) * 1000, summaries)

__all__ = [
    "ScenarioRequest", "EvalBudget", "StepArchetype", "ScenarioHandle", "ScenarioDossier",
    "CandidateEvaluation", "BatchEvaluation", "FullPlanEvaluation", "OracleCache",
    "intern_step_archetypes", "prepare_scenario", "evaluate_rewrites", "finalize_plan",
]
