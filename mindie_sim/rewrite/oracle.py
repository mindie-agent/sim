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

from collections import Counter, OrderedDict, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import hashlib
from itertools import islice
from math import ceil, fsum, isclose
import re
from threading import Lock
from time import monotonic
from typing import Any, Literal

from mindie_sim.catalog.ops import comm_op
from mindie_sim.prediction.collectives import OneShotCollective
from mindie_sim.prediction.collectives.ring_pairwise import PairwiseCollective, RingCollective
from mindie_sim.prediction.evaluation import ACTIVE_DEADLINE_MONO, EvaluationContext
from mindie_sim.prediction.execution.layer_runs import (
    LayerCohortPlan,
    LayerCohortSpec,
    _body_layer_key,
    _is_diff_op_safe_to_price_isolated,
    _is_target_body_op,
    discover_layer_cohorts,
    execute_native_cohort_graph,
)
from mindie_sim.prediction.execution.operator_replay import operator_costs
from mindie_sim.prediction.execution.split_barrier import DEFAULT_SPLIT_DTYPE_BYTES
from mindie_sim.prediction.execution.runner import EAGER_MOE_DISPATCH_RECIPES
from mindie_sim.prediction.operators.grouped_moe.expected_grouped import ExpectedGroupedMatmul
from mindie_sim.prediction.fusion.bridge import HardwareProfile
from mindie_sim.prediction.fusion.pool import register_recipe
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.operator_analysis import physical_id
from mindie_sim.prediction.operators.moe_transport.moe_callers import RoutingPolicy
from mindie_sim.prediction.program.physical_key import physical_key
from mindie_sim.prediction.program.primitives import PrimitiveStore
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.prediction.results import (
    FusedPipelineDescriptor, OpOccurrence, RankOps, ReplayOptions, StepEvaluation, StepRequest,
)
from mindie_sim.prediction.runtime.dp_cohort_dedup import _ops_symmetric
from mindie_sim.prediction.runtime.tile_solver import TilePipelineSpec
from mindie_sim.prediction.streams import MAIN_STREAM, bind_stream_owners
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.templates.contracts import FormulaKernelSpec
from mindie_sim.templates.fuse_region import FusedPipelineRecipe, solve_formula_sram

from ._oracle_prefetch import bind_prefetch_ops, bind_prefetch_streams, primitive_table_digest_token
from .apply import (
    AppliedPlan, DEFAULT_STACK_ID, ResolvedScenario, _die, _materialize_control_deps,
    _path, _solve_tiles, apply_plan, resolve_scenario,
)
from .catalog_replay import (
    _CatalogReplayProgram,
    _PIPE_ROLE_MAP,
    _canon_digest_value,
    _replay_ops,
    catalog_recipe_id,
)
from .dual_cert import build_dual_inputs_from_scenario, solve_dual_lower_bound
from .schema import (
    FuseRegion, PipelineTiles, QuantizeContract, RewritePlan, SelectTemplate, WeightPrefetch, _digest_default,
    _json_digest, _references, empty_plan, plan_sha256,
)
from .verify import (
    NS_PER_MS,
    VerificationReceipt,
    _COHORT_PREFIX,
    _combine_replays,
    _evaluation,
    _memoized_catalog_lowering,
    _norm_catalog_datapath,
    _verify_region_summary_bindings,
    _work_totals,
    verify_plan,
)

DEFAULT_MAX_CANDIDATES = 8
DEFAULT_MAX_ARCHETYPES = 5
DEFAULT_CACHE_ENTRIES = 16
MAX_RETAINED_OPERATORS = 16000
MAX_OVERLAP_TRACE_EVENTS = 8192
MAX_BOTTLENECK_OPS = 16
MAX_SCREEN_CACHE_ENTRIES = 256
GROUPED_MATRIX_RANK = 3
_RECIPE_DIGEST_LEN = 16
_DIFF_OP_OVERLAP_EPS_MS = 1e-9
_CTX_SITE_RE = re.compile(r"@r(\d+):s(\d+)(?:$|:)")
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
    _finalized: OrderedDict = field(default_factory=OrderedDict, repr=False, compare=False)
    _certificate: dict = field(default_factory=dict, repr=False, compare=False)
    _catalog_memo: dict = field(default_factory=dict, repr=False, compare=False)
    _hardware_cache: dict = field(default_factory=dict, repr=False, compare=False)
    _cache_lock: Lock = field(default_factory=Lock, repr=False, compare=False)
    allow_unbound_formula: bool = False

    def _prefetch_hardware_for(self, replica: int, stage: int) -> HardwareProfile:
        key = (int(replica), int(stage))
        with self._cache_lock:
            hit = self._hardware_cache.get(key)
            if hit is None:
                die = _die(self.scenario, int(replica), int(stage))
                hit = HardwareProfile.from_die(die)
                self._hardware_cache[key] = hit
            return hit

    @property
    def _prefetch_hardware(self) -> HardwareProfile:
        return self._prefetch_hardware_for(0, 0)

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
    has_unbound_formula_ops: bool = False
    unbound_formula_ops: tuple[str, ...] = ()
    binding_states: tuple[tuple[str, str], ...] = ()

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
    has_unbound_formula_ops: bool = False
    unbound_formula_ops: tuple[str, ...] = ()
    binding_states: tuple[tuple[str, str], ...] = ()
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

_ARCHETYPE_PHYSICAL_META_KEYS = (
    "resource_work", "weight_bytes", "packed_weight_bytes",
    "weight_format", "activation_format", "output_format", "compute_format",
    "uses_simt", "tile_uses_simt", "template_param_uses_simt",
    "read_pipe", "group_axis", "quant_category", "model_scope",
    "load_time", "immutable", "pure", "fixpipe_bytes", "mte1_bytes",
    "weight_initial_residency", "initial_residency",
)
_SIMT_BACKEND_TARGETS = frozenset({"tilelang_simt", "tla_simt"})


def _archetype_op_signature(op: OpSpec, local_idx: Mapping[str, int], op_layer: Mapping[str, int]):
    meta = dict(op.metadata) if op.metadata else {}
    norm_deps = tuple(
        ("local", local_idx[dep]) if dep in local_idx
        else ("layer_delta", op.layer - op_layer[dep]) if dep in op_layer
        else ("ext",)
        for dep in op.deps
    )
    phys_meta = tuple(
        (key, _canon_digest_value(meta[key]))
        for key in _ARCHETYPE_PHYSICAL_META_KEYS
        if key in meta
    )
    p_abi = getattr(op, "precision_abi", None)
    return (
        op.family, op.recipe, op.dtype, op.layout,
        op.local_shape, op.padded_shape, op.launch_shape,
        op.stage, op.phase, op.stream,
        float(op.flops), tuple((str(dt), float(fl)) for dt, fl in op.matrix_flops),
        float(op.vector_ops), float(op.hbm_bytes),
        float(op.mte2_bytes), float(op.mte3_bytes),
        float(op.link_bytes), op.link_scope,
        p_abi.abi_signature if p_abi is not None else "",
        norm_deps, phys_meta,
    )


def intern_step_archetypes(ops: tuple[OpSpec, ...],
                           max_archetypes: int = DEFAULT_MAX_ARCHETYPES) -> tuple[StepArchetype, ...]:
    """Intern homogeneous interior layers with equivalent physical work, traffic, and topology."""
    if type(max_archetypes) is not int or max_archetypes < 1:
        raise ModelingError("rewrite_archetype_budget_invalid")
    by_layer = _layers(ops)
    op_layer = {op.op_id: op.layer for op in ops}
    stage_layers = defaultdict(list)
    for layer, members in by_layer.items():
        if layer >= 0:
            for stage in {op.stage for op in members}:
                stage_layers[stage].append(layer)
    bounds = {stage: (min(layers), max(layers)) for stage, layers in stage_layers.items()}
    buckets = {}
    for layer, members in sorted(by_layer.items()):
        local_idx = {op.op_id: idx for idx, op in enumerate(members)}
        key = (
            tuple(_archetype_op_signature(op, local_idx, op_layer) for op in members),
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
def _stream_touched_layers(ops: Sequence[OpSpec], stream_plan: Any) -> set[int]:
    """Return target body layer indices touched by lane assignments, stream events, or non-main streams."""
    if stream_plan is None:
        return set()
    touched_ids: set[str] = {oid for oid, _lane in getattr(stream_plan, "assignments", ())}
    for event in getattr(stream_plan, "events", ()):
        touched_ids.add(event.record_after)
        touched_ids.update(event.wait_before)
    return {
        int(op.layer) for op in ops
        if _is_target_body_op(op)
        and (op.op_id in touched_ids or (op.stream or MAIN_STREAM) != MAIN_STREAM)
    }


def refine_layer_cohort_plan(
    case: Any, ops: Sequence[OpSpec], *, stream_plan: Any = None,
) -> LayerCohortPlan:
    """Sub-split structural layer cohorts by actual per-layer lowered op signatures."""
    base = discover_layer_cohorts(case)
    by_layer: dict[int, list[OpSpec]] = defaultdict(list)
    for op in ops:
        if _is_target_body_op(op):
            by_layer[int(op.layer)].append(op)
    touched_stream_layers = _stream_touched_layers(ops, stream_plan)
    plans_by_idx = {
        int(getattr(lp, "index", idx)): lp
        for idx, lp in enumerate(getattr(getattr(case, "model", None), "layer_plans", ()) or ())
    }
    cohorts: list[LayerCohortSpec] = []
    cohort_by_layer: dict[int, int] = {}
    for cohort in base.cohorts:
        subgroups: OrderedDict[tuple[Any, ...], list[int]] = OrderedDict()
        for layer_idx in cohort.layer_indices:
            layer_ops = by_layer.get(int(layer_idx), ())
            sig: tuple[Any, ...] = tuple(
                (
                    op.family,
                    op.recipe,
                    op.dtype,
                    None if _is_diff_op_safe_to_price_isolated(op) else op.local_shape,
                    op.stage,
                    op.stream,
                    None if _is_diff_op_safe_to_price_isolated(op) else physical_key(op),
                )
                for op in layer_ops
            )
            if int(layer_idx) in touched_stream_layers:
                sig = (*sig, ("stream_singleton", int(layer_idx)))
            subgroups.setdefault(sig, []).append(int(layer_idx))
        for sub_sig, indices in subgroups.items():
            c_idx = len(cohorts)
            rep_idx = indices[0]
            cohorts.append(LayerCohortSpec(
                cohort_index=c_idx,
                stage=cohort.stage,
                representative_layer=rep_idx,
                layer_indices=tuple(indices),
                representative_plan=plans_by_idx.get(rep_idx, cohort.representative_plan),
                signature=(*cohort.signature, sub_sig),
            ))
            for idx in indices:
                cohort_by_layer[idx] = c_idx
    return LayerCohortPlan(
        cohorts=tuple(cohorts),
        cohort_by_layer=cohort_by_layer,
        stage_counts=base.stage_counts,
        total_target_layers=base.total_target_layers,
        total_draft_layers=base.total_draft_layers,
    )


def _norm_layer_label_value(val: Any, lyr: int, op_id: str) -> Any:
    if isinstance(val, str):
        out = val.replace(op_id, "__SELF_OP__") if op_id and op_id in val else val
        out = out.replace(f"layer[{lyr}]", "layer[__L__]")
        out = out.replace(f".l{lyr}.", ".l__L__.")
        out = out.replace(f"layer{lyr}:", "layer__L__:")
        if out == f"layer{lyr}":
            out = "layer__L__"
        return out
    if isinstance(val, (tuple, list)):
        return tuple(_norm_layer_label_value(x, lyr, op_id) for x in val)
    if isinstance(val, Mapping):
        return tuple(
            sorted(
                (
                    str(k),
                    (-1 if k == "layer" and v == lyr and type(v) is int else _norm_layer_label_value(v, lyr, op_id)),
                )
                for k, v in val.items()
            )
        )
    return val


def _layer_normalized_diff_sig(op: OpSpec, book: Any) -> tuple[Any, ...]:
    """Structural op signature normalizing only exact layer-index labels in metadata."""
    lyr = int(op.layer)
    meta = tuple(
        sorted(
            (
                str(k),
                (-1 if k == "layer" and v == lyr and type(v) is int else _norm_layer_label_value(v, lyr, op.op_id)),
            )
            for k, v in op.metadata
        )
    )
    rec_map = getattr(book, "recipes", None) or {}
    rec_obj = rec_map.get(op.recipe) if op.recipe else None
    rec_tag = (type(rec_obj).__qualname__, repr(rec_obj)) if rec_obj is not None else ()
    return (
        int(op.stage), op.family, op.recipe, rec_tag, op.phase, op.dtype, op.layout,
        op.local_shape, op.padded_shape, op.launch_shape,
        op.flops, op.matrix_flops, op.vector_ops,
        op.hbm_bytes, op.mte2_bytes, op.mte3_bytes, op.link_bytes, op.link_scope,
        op.source_boundary, meta,
    )


def prewarm_cohort_diff_ops(
    case: Any,
    ops0: Sequence[OpSpec],
    cohort_plan: LayerCohortPlan,
    context: EvaluationContext,
    book: Any,
) -> None:
    """Price each unique layer-label-normalized depth-varying op once across both caches."""
    if context is None or not ops0:
        return
    dur_cache = getattr(context, "_cohort_pkey_dur_cache", None)
    if dur_cache is None:
        dur_cache = {}
        try:
            setattr(context, "_cohort_pkey_dur_cache", dur_cache)
        except (AttributeError, TypeError):
            return
    serv_cache = getattr(context, "_cohort_diff_serviced_cache", None)
    if serv_cache is None:
        serv_cache = {}
        try:
            setattr(context, "_cohort_diff_serviced_cache", serv_cache)
        except (AttributeError, TypeError):
            serv_cache = None

    by_layer: dict[int, list[OpSpec]] = defaultdict(list)
    for op in ops0:
        if _is_target_body_op(op):
            by_layer[int(op.layer)].append(op)
    diff_entries: dict[tuple[Any, ...], OpSpec] = {}
    for cohort in cohort_plan.cohorts:
        if len(cohort.layer_indices) < 2:
            continue
        rep_ops = by_layer.get(int(cohort.representative_layer), ())
        if not rep_ops:
            continue
        rep_pkeys = tuple((int(op.stage), physical_key(op)) for op in rep_ops)
        for lyr in cohort.layer_indices:
            if int(lyr) == int(cohort.representative_layer):
                continue
            l_ops = by_layer.get(int(lyr), ())
            if len(l_ops) != len(rep_ops):
                continue
            for op_r, pkey_r, op_l in zip(rep_ops, rep_pkeys, l_ops):
                pkey_l = (int(op_l.stage), physical_key(op_l))
                if (
                    pkey_l != pkey_r
                    and _is_diff_op_safe_to_price_isolated(op_l)
                    and _is_diff_op_safe_to_price_isolated(op_r)
                ):
                    diff_entries.setdefault(pkey_r, op_r)
                    diff_entries.setdefault(pkey_l, op_l)
    if not diff_entries:
        return

    rec_map = getattr(book, "recipes", None) or {}
    ctx_rec_map = getattr(getattr(context, "recipes", None), "recipes", None) or {}

    def _serv_key(item: OpSpec) -> tuple[Any, ...]:
        rec_obj = rec_map.get(item.recipe) if item.recipe else None
        return (int(item.stage), physical_id(item), item.recipe, type(rec_obj).__qualname__ if rec_obj is not None else "")

    by_norm: dict[tuple[Any, ...], list[tuple[tuple[Any, ...], OpSpec]]] = defaultdict(list)
    for pkey, op in diff_entries.items():
        if pkey in dur_cache and (serv_cache is None or _serv_key(op) in serv_cache):
            continue
        if op.recipe in rec_map and op.recipe in ctx_rec_map and rec_map[op.recipe] != ctx_rec_map[op.recipe]:
            continue
        by_norm[_layer_normalized_diff_sig(op, book)].append((pkey, op))
    if not by_norm:
        return
    reps: list[OpSpec] = []
    pid_to_norms: dict[tuple[int, str], list[tuple[Any, ...]]] = defaultdict(list)
    for nsig, group in by_norm.items():
        rep_op = group[0][1]
        reps.append(rep_op)
        pid_to_norms[(int(rep_op.stage), physical_id(rep_op))].append(nsig)
    oc = operator_costs(case, tuple(reps), recipes=book, evaluation=context, replica=0)
    for row in oc.get("operators", ()):
        r_stage = int(dict(row.get("operator") or {}).get("stage", 0) or 0)
        r_pid = str(row["physical_id"])
        dur_val = float(row["duration_ms"])
        serv_val = dict(row.get("serviced") or {})
        for nsig in pid_to_norms.get((r_stage, r_pid), ()):
            for pkey, item in by_norm[nsig]:
                dur_cache[pkey] = dur_val
                if serv_cache is not None:
                    serv_cache[_serv_key(item)] = serv_val


def _parse_span_task(task_name: str, known_ids: set[str]) -> tuple[int, str] | None:
    rep_tok, _, tail = str(task_name).partition(":")
    if not rep_tok.startswith("r") or not rep_tok[1:].isdigit():
        return None
    _, _, tail = tail.partition(":")
    keys = tail.split(":")
    for count in range(len(keys), 0, -1):
        oid = ":".join(keys[:count])
        if oid in known_ids:
            return int(rep_tok[1:]), oid
    return None


def _compute_kept_op_shifts(
    r0_rows: Sequence[Mapping[str, Any]],
    ops0: Sequence[OpSpec],
    op_rep_map: Mapping[str, str],
    context: EvaluationContext | None,
) -> dict[str, float]:
    """Reconstruct exact cumulative timeline shift applied to each kept op by cohort expansion."""
    ops0_by_id = {op.op_id: op for op in ops0}
    row_by_op = {str(row.get("op_id", "")): row for row in r0_rows}
    rep_keys: set[tuple[str, int]] = set()
    skipped_pairs: dict[tuple[str, int], list[tuple[OpSpec, OpSpec]]] = defaultdict(list)
    skipped_rep_key: dict[tuple[str, int], tuple[str, int]] = {}
    for oid, rep_oid in op_rep_map.items():
        op_l = ops0_by_id.get(oid)
        op_r = ops0_by_id.get(rep_oid)
        if op_l is None or op_r is None:
            continue
        l_key = _body_layer_key(op_l)
        r_key = _body_layer_key(op_r)
        if l_key is not None and r_key is not None:
            rep_keys.add(r_key)
            skipped_rep_key[l_key] = r_key
            skipped_pairs[l_key].append((op_l, op_r))
    rep_rows_by_layer: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in r0_rows:
        op0 = ops0_by_id.get(str(row.get("op_id", "")))
        bkey = _body_layer_key(op0) if op0 is not None else None
        if bkey in rep_keys:
            rep_rows_by_layer[bkey].append(row)
    rep_span_by_layer: dict[tuple[str, int], float] = {
        r_key: max(float(r["complete_ms"]) for r in rows) - min(float(r["ready_ms"]) for r in rows)
        for r_key, rows in rep_rows_by_layer.items()
        if rows
    }
    dur_cache = getattr(context, "_cohort_pkey_dur_cache", None) or {}
    skipped_span_by_layer: dict[tuple[str, int], float] = {}
    for l_key, pairs in skipped_pairs.items():
        span_l = rep_span_by_layer.get(skipped_rep_key[l_key], 0.0)
        for op_l, op_r in pairs:
            pkey_l = (int(op_l.stage), physical_key(op_l))
            pkey_r = (int(op_r.stage), physical_key(op_r))
            if pkey_l == pkey_r:
                continue
            if pkey_l in dur_cache and pkey_r in dur_cache:
                span_l += float(dur_cache[pkey_l]) - float(dur_cache[pkey_r])
            elif op_l.op_id in row_by_op and op_r.op_id in row_by_op:
                span_l += float(row_by_op[op_l.op_id]["elapsed_ms"]) - float(row_by_op[op_r.op_id]["elapsed_ms"])
        skipped_span_by_layer[l_key] = span_l
    cum_shift = 0.0
    seen_skipped: set[tuple[str, int]] = set()
    kept_shift: dict[str, float] = {}
    for row in r0_rows:
        oid = str(row.get("op_id", ""))
        if oid in op_rep_map:
            op_l = ops0_by_id.get(oid)
            l_key = _body_layer_key(op_l) if op_l is not None else None
            if l_key is not None and l_key not in seen_skipped:
                seen_skipped.add(l_key)
                cum_shift += skipped_span_by_layer.get(l_key, 0.0)
        else:
            kept_shift[oid] = cum_shift
    return kept_shift


def _expand_cohort_service_spans(
    report: dict[str, Any],
    replay_graphs: Mapping[int, Sequence[OpSpec]],
    op_rep_map: Mapping[str, str],
    context: EvaluationContext | None = None,
) -> dict[str, Any]:
    """Shift kept-op spans to expanded coordinates and expand service spans across all replicas."""
    if not op_rep_map or 0 not in replay_graphs:
        return report
    events = report.get("events") or ()
    evicted = dict(report.get("evicted_service_spans") or {})
    if not events and not evicted:
        return report
    timeline = report.get("timeline") or ()
    r0_rows = [row for row in timeline if int(row.get("replica", 0)) == 0]
    kept_shift = _compute_kept_op_shifts(r0_rows, replay_graphs[0], op_rep_map, context)
    timing_by_op: dict[tuple[int, str], tuple[float, float, float]] = {
        (int(row.get("replica", 0)), str(row.get("op_id", ""))): (
            float(row.get("ready_ms", 0.0) or 0.0),
            float(row.get("complete_ms", 0.0) or 0.0),
            float(row.get("elapsed_ms", 0.0) or 0.0),
        )
        for row in timeline
    }
    stage_by_op: dict[str, int] = {op.op_id: int(op.stage) for op in replay_graphs[0]}
    known_ids = set(stage_by_op)
    rep_spans: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    changed = False
    for ev_name, spans in list(evicted.items()):
        parsed = _parse_span_task(ev_name, known_ids)
        if parsed is None:
            continue
        rep, oid = parsed
        sh = kept_shift.get(oid, 0.0)
        shifted = [(float(s) + sh, float(e) + sh) for s, e in spans]
        if sh:
            evicted[ev_name] = shifted
            changed = True
        rep_spans[(rep, oid)].extend(shifted)
    new_events: list[Any] = []
    events_changed = False
    for ev in events:
        parsed = _parse_span_task(str(ev.get("task", "")), known_ids)
        if parsed is not None and "start_ms" in ev and "end_ms" in ev:
            rep, oid = parsed
            sh = kept_shift.get(oid, 0.0)
            s_ms = float(ev["start_ms"]) + sh
            e_ms = float(ev["end_ms"]) + sh
            if sh:
                new_events.append({**ev, "start_ms": s_ms, "end_ms": e_ms})
                events_changed = True
            else:
                new_events.append(ev)
            if ev.get("serviced"):
                rep_spans[(rep, oid)].append((s_ms, e_ms))
        else:
            new_events.append(ev)
    replicas = tuple(sorted(replay_graphs))
    for oid, rep_oid in op_rep_map.items():
        if oid == rep_oid:
            continue
        stage = stage_by_op.get(oid, 0)
        for rep in replicas:
            src_spans = rep_spans.get((rep, rep_oid)) or rep_spans.get((0, rep_oid))
            dst_t = timing_by_op.get((rep, oid))
            rep_t = timing_by_op.get((rep, rep_oid))
            if not src_spans or dst_t is None or rep_t is None:
                continue
            ready_oid, complete_oid, elapsed_oid = dst_t
            ready_rep, _complete_rep, elapsed_rep = rep_t
            if not isclose(elapsed_oid, elapsed_rep, rel_tol=1e-9, abs_tol=1e-9):
                dst_spans = [(ready_oid, max(ready_oid, complete_oid))]
            else:
                delta = ready_oid - ready_rep
                dst_spans = [(s + delta, e + delta) for s, e in src_spans]
            evicted[f"r{rep}:s{stage}:{oid}"] = dst_spans
            changed = True
    if not changed and not events_changed:
        return report
    out = {**report, "evicted_service_spans": evicted}
    if events_changed:
        out["events"] = new_events
    return out


def _diff_op_isolated_in_timeline(op_id: str, timeline: Sequence[Mapping[str, Any]]) -> bool:
    """Return True iff ``op_id`` on replica 0 does not overlap any other op in ``timeline``."""
    target = next(
        (r for r in timeline if int(r.get("replica", 0)) == 0 and str(r.get("op_id", "")) == op_id),
        None,
    )
    if target is None:
        return False
    t_lo = float(target.get("ready_ms", 0.0) or 0.0)
    t_hi = float(target.get("complete_ms", t_lo) or t_lo)
    for row in timeline:
        if int(row.get("replica", 0)) != 0 or str(row.get("op_id", "")) == op_id:
            continue
        r_lo = float(row.get("ready_ms", 0.0) or 0.0)
        r_hi = float(row.get("complete_ms", r_lo) or r_lo)
        if r_lo + _DIFF_OP_OVERLAP_EPS_MS < t_hi and r_hi > t_lo + _DIFF_OP_OVERLAP_EPS_MS:
            return False
    return True


def _accumulate_diff_cohort_metrics(
    case: Any,
    diff_pairs: Sequence[tuple[OpSpec, OpSpec]],
    rep_timelines: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    replicas: Sequence[int],
    context: EvaluationContext,
    book: RecipeBook,
    extra_serviced: Counter[str],
    extra_busy: Counter[str],
) -> bool:
    """Accumulate exact serviced & busy deltas for strictly isolated depth-varying diff ops."""
    for op_r, op_l in diff_pairs:
        r_key = _body_layer_key(op_r)
        if (
            r_key is None
            or not _is_diff_op_safe_to_price_isolated(op_r)
            or not _is_diff_op_safe_to_price_isolated(op_l)
            or float(op_r.link_bytes or 0.0) > 0.0
            or float(op_l.link_bytes or 0.0) > 0.0
            or not _diff_op_isolated_in_timeline(op_r.op_id, rep_timelines.get(r_key, ()))
        ):
            return False
    metrics_cache = getattr(context, "_cohort_diff_metrics_cache", None)
    if metrics_cache is None:
        metrics_cache = {}
        try:
            setattr(context, "_cohort_diff_metrics_cache", metrics_cache)
        except (AttributeError, TypeError):
            metrics_cache = None

    def _diff_cache_key(item: OpSpec) -> tuple[Any, ...]:
        rec_obj = book.recipes.get(item.recipe) if item.recipe else None
        rec_tag = type(rec_obj).__qualname__ if rec_obj is not None else ""
        return (int(item.stage), physical_id(item), item.recipe, rec_tag)

    diff_metrics: dict[tuple[int, str], tuple[dict[str, float], dict[str, float]]] = {}
    for op_r, op_l in diff_pairs:
        for item in (op_r, op_l):
            skey = (int(item.stage), physical_id(item))
            if skey in diff_metrics:
                continue
            ckey = _diff_cache_key(item)
            if metrics_cache is not None and ckey in metrics_cache:
                diff_metrics[skey] = metrics_cache[ckey]
                continue
            ext = {dep: 0.0 for dep in _external_dependencies((item,))}
            rep_one = graph_costs(
                case, {0: (item,)}, recipes=book, external_inputs=ext,
                evaluation=context, isolated=False, include_submission=False,
                trace_limit=0, fast_span_only=False, timeline_replicas=(0,),
            )
            ex_one = rep_one.get("execution") or {}
            entry = (dict(ex_one.get("serviced") or {}), dict(ex_one.get("resource_busy_ms") or {}))
            diff_metrics[skey] = entry
            if metrics_cache is not None:
                metrics_cache[ckey] = entry

    def _add_mirrored(target: Counter[str], src_map: Mapping[str, float], sign: float) -> None:
        for res_name, amt in src_map.items():
            if not amt or "scheduler@" in res_name:
                continue
            delta = sign * float(amt)
            target[res_name] += delta
            if "@r0:" in res_name:
                for peer in peers:
                    target[res_name.replace("@r0:", f"@r{peer}:", 1)] += delta

    peers = replicas[1:]
    for op_r, op_l in diff_pairs:
        key_r = (int(op_r.stage), physical_id(op_r))
        key_l = (int(op_l.stage), physical_id(op_l))
        serv_l, busy_l = diff_metrics.get(key_l, ({}, {}))
        serv_r, busy_r = diff_metrics.get(key_r, ({}, {}))
        _add_mirrored(extra_serviced, serv_l, 1.0)
        _add_mirrored(extra_busy, busy_l, 1.0)
        _add_mirrored(extra_serviced, serv_r, -1.0)
        _add_mirrored(extra_busy, busy_r, -1.0)
    return True


def _expand_cohort_serviced(
    case: Any,
    replay_graphs: Mapping[int, Sequence[OpSpec]],
    report: dict[str, Any],
    op_rep_map: Mapping[str, str],
    context: EvaluationContext,
    book: RecipeBook,
    replay: ReplayOptions,
) -> dict[str, Any] | None:
    """Reconstruct exact full-graph executed clocks, busy intervals, and service counters."""
    if not op_rep_map or 0 not in replay_graphs:
        return report
    exec_info = report.get("execution")
    if not isinstance(exec_info, Mapping) or not isinstance(exec_info.get("serviced"), Mapping):
        return report
    ops0 = replay_graphs[0]
    ops_by_id = {op.op_id: op for op in ops0}
    ops_by_rep_body: dict[int, dict[tuple[str, int], list[OpSpec]]] = {}
    for rep, r_ops in replay_graphs.items():
        by_body: dict[tuple[str, int], list[OpSpec]] = defaultdict(list)
        for op in r_ops:
            bkey = _body_layer_key(op)
            if bkey is not None:
                by_body[bkey].append(op)
        ops_by_rep_body[rep] = by_body
    skipped_layers_by_rep: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    diff_pairs: list[tuple[OpSpec, OpSpec]] = []
    for skipped_id, rep_id in op_rep_map.items():
        op_l = ops_by_id.get(skipped_id)
        op_r = ops_by_id.get(rep_id)
        if op_l is None or op_r is None:
            continue
        l_key = _body_layer_key(op_l)
        r_key = _body_layer_key(op_r)
        if l_key is not None and r_key is not None:
            skipped_layers_by_rep[r_key].add(l_key)
        if (int(op_l.stage), physical_key(op_l)) != (int(op_r.stage), physical_key(op_r)):
            diff_pairs.append((op_r, op_l))
    extra: Counter[str] = Counter()
    extra_busy: Counter[str] = Counter()
    rep_timelines: dict[tuple[str, int], tuple[Mapping[str, Any], ...]] = {}
    replicas = tuple(sorted(replay_graphs))
    for rep_key, skipped_keys in skipped_layers_by_rep.items():
        if not skipped_keys:
            continue
        rep_cut = {r: tuple(ops_by_rep_body[r].get(rep_key, ())) for r in replicas}
        if not rep_cut.get(0):
            continue
        cut_ext = {dep: 0.0 for r_ops in rep_cut.values() for dep in _external_dependencies(r_ops)}
        rep_report = graph_costs(
            case, rep_cut, recipes=book, external_inputs=cut_ext,
            evaluation=context, isolated=False,
            include_submission=replay.include_submission,
            trace_limit=0, fast_span_only=False,
            timeline_replicas=(0,),
        )
        if len(rep_cut) > 1:
            rep_report = _mirror_dedup_directional_evidence(case, context, rep_cut, rep_report)
        rep_timelines[rep_key] = tuple(rep_report.get("timeline") or ())
        mult = len(skipped_keys)
        rep_exec = rep_report.get("execution") or {}
        for res_name, amount in (rep_exec.get("serviced") or {}).items():
            if amount and "scheduler@" not in res_name:
                extra[res_name] += float(amount) * mult
        for res_name, busy_amt in (rep_exec.get("resource_busy_ms") or {}).items():
            if busy_amt and "scheduler@" not in res_name:
                extra_busy[res_name] += float(busy_amt) * mult
    if diff_pairs and not _accumulate_diff_cohort_metrics(
        case, diff_pairs, rep_timelines, replicas, context, book, extra, extra_busy,
    ):
        return None
    serviced = dict(exec_info["serviced"])
    for res_name, delta in extra.items():
        if delta:
            serviced[res_name] = max(0.0, float(serviced.get(res_name, 0.0)) + delta)
    busy_ms = dict(exec_info.get("resource_busy_ms") or {})
    for res_name, delta in extra_busy.items():
        if delta:
            busy_ms[res_name] = max(0.0, float(busy_ms.get(res_name, 0.0)) + delta)
    full_time_ms = float(report.get("complete_ms", report.get("duration_ms", exec_info.get("time_ms", 0.0))))
    new_exec = {
        **exec_info,
        "time_ms": full_time_ms,
        "serviced": serviced,
        "resource_busy_ms": busy_ms,
    }
    return {**report, "execution": new_exec}


def _replicas_hardware_symmetric(case: Any, context: EvaluationContext, replicas: Sequence[int]) -> bool:
    if len(replicas) < 2:
        return True
    topology = context.bindings.topology
    dies_by_rank = {int(d.rank): d for d in case.machine.dies}
    pp = max(1, int(case.runtime.parallel.pp))

    def _hw_sig(rep: int) -> tuple[Any, ...]:
        return tuple(
            (
                len(ranks := tuple(topology.ranks(rep, stage))),
                tuple(getattr(dies_by_rank.get(int(rk)), "resource_signature", None) for rk in ranks),
            )
            for stage in range(pp)
        )

    sig0 = _hw_sig(replicas[0])
    return all(_hw_sig(rep) == sig0 for rep in replicas[1:])


def _graphs_symmetric_for_dedup(replay_graphs: Mapping[int, Sequence[OpSpec]]) -> tuple[int, tuple[int, ...]] | None:
    if len(replay_graphs) < 2:
        return None
    replicas = tuple(sorted(replay_graphs))
    r0 = replicas[0]
    ops0 = replay_graphs[r0]
    self0 = f"analysis{r0}:"
    for rep in replicas[1:]:
        ops_r = replay_graphs[rep]
        if len(ops_r) != len(ops0):
            return None
        self_r = f"analysis{rep}:"
        if any(not _ops_symmetric(op0, op_r, self0, {}, self_r, {}) for op0, op_r in zip(ops0, ops_r)):
            return None
    return r0, replicas[1:]


def _expand_symmetric_peer_timeline(
    report: dict[str, Any],
    replay_graphs: Mapping[int, Sequence[OpSpec]],
    r0: int,
    peers: Sequence[int],
) -> dict[str, Any]:
    """Mirror canonical r0 expanded cohort timeline rows onto symmetric DP peers."""
    base_timeline = list(report.get("timeline") or ())
    if not base_timeline or not peers:
        return report
    r0_rows = [row for row in base_timeline if int(row.get("replica", -1)) == r0]
    if not r0_rows:
        return report
    existing_reps = {int(row.get("replica", -1)) for row in base_timeline}
    r0_prefix = f"r{r0}:"
    added: list[dict[str, Any]] = []
    for rep in peers:
        if rep in existing_reps:
            continue
        rep_prefix = f"r{rep}:"
        for row in r0_rows:
            clone = dict(row)
            oid = str(row.get("op_id", ""))
            clone["id"] = f"{rep_prefix}{oid}"
            clone["replica"] = rep
            clone["predecessors"] = [
                (rep_prefix + p[len(r0_prefix):]) if isinstance(p, str) and p.startswith(r0_prefix) else p
                for p in (row.get("predecessors") or ())
            ]
            added.append(clone)
    if not added:
        return report
    expanded = base_timeline + added
    return {
        **report,
        "timeline": expanded,
        "operator_occurrences": sum(1 for r in expanded if str(r.get("family", "")) != "barrier"),
    }


def _mirror_dedup_directional_evidence(
    case: Any,
    context: EvaluationContext,
    replay_graphs: Mapping[int, Sequence[OpSpec]],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Mirror canonical dynamic directional HBM & per-unit peaks onto deduplicated DP peers."""
    sym = _graphs_symmetric_for_dedup(replay_graphs)
    if sym is None:
        return report
    r0, peers = sym
    if not _replicas_hardware_symmetric(case, context, (r0, *peers)):
        return report
    exec_info = report.get("execution")
    if not isinstance(exec_info, Mapping):
        return report
    src_tag = f"@r{r0}:"
    serviced = dict(exec_info.get("serviced") or {})
    busy = dict(exec_info.get("resource_busy_ms") or {})
    peaks = dict(exec_info.get("peak_allocated") or {})
    unit_peaks = dict(exec_info.get("peak_allocated_per_unit") or {})
    changed = False
    dir_suffixes = (":r", ":w", ":ctr_r", ":ctr_w")
    skipped_dir_peers = tuple(
        rep for rep in peers
        if not any(
            f"@r{rep}:" in k and v and k.endswith(dir_suffixes)
            for k, v in serviced.items()
        )
    )
    if skipped_dir_peers:
        for res_name, amt in list(serviced.items()):
            if not amt or src_tag not in res_name or not res_name.endswith(dir_suffixes):
                continue
            b_val = busy.get(res_name)
            for rep in skipped_dir_peers:
                dst_name = res_name.replace(src_tag, f"@r{rep}:", 1)
                if not serviced.get(dst_name):
                    serviced[dst_name] = amt
                    if b_val is not None:
                        busy[dst_name] = b_val
                    changed = True
    r0_unit_peaks = {k: v for k, v in unit_peaks.items() if src_tag in k and v}
    if r0_unit_peaks:
        for rep in peers:
            rep_tag = f"@r{rep}:"
            if any(rep_tag in k and v for k, v in unit_peaks.items()):
                continue
            for k, v in r0_unit_peaks.items():
                unit_peaks[k.replace(src_tag, rep_tag, 1)] = v
                changed = True
            for k, v in list(peaks.items()):
                if src_tag in k and v:
                    dst = k.replace(src_tag, rep_tag, 1)
                    if not peaks.get(dst):
                        peaks[dst] = v
                        changed = True
    if not changed:
        return report
    new_exec = {
        **exec_info,
        "serviced": serviced,
        "resource_busy_ms": busy,
        "peak_allocated": peaks,
        "peak_allocated_per_unit": unit_peaks,
    }
    return {**report, "execution": new_exec}


def _stream_plan_full_main_budget(
    context: EvaluationContext,
    stream_plan: Any,
    ops: Sequence[OpSpec],
) -> bool:
    """True when ``MAIN_STREAM`` owns an uncapped ``shared_cap`` core budget on every stage."""
    if stream_plan is None:
        return True
    stages = {int(op.stage) for op in ops} or {0}
    replica = int(stream_plan.rank)
    for stage in stages:
        path = context.bindings.datapath_for_stage(replica, stage)
        owners = bind_stream_owners(stream_plan, path).get(MAIN_STREAM)
        if (
            owners is None
            or owners.isolation != "shared_cap"
            or owners.aic_limit < path.aic_count
            or owners.aiv_limit < path.aiv_count
        ):
            return False
    return True


def _resolve_cohort_streams(
    replay_graphs: Mapping[int, Sequence[OpSpec]],
    context: EvaluationContext,
    streams: Sequence[Any],
    book: Any,
    eff_trace: int,
) -> tuple[bool, Any]:
    """Return ``(eligible, stream0)`` for exact cohort reduction with optional symmetric streams."""
    if eff_trace > 0 and any(
        isinstance(rec, _DboGrouped) for rec in (getattr(book, "recipes", None) or {}).values()
    ):
        return False, None
    if not streams:
        return eff_trace == 0, None
    by_rank = {int(plan.rank): plan for plan in streams}
    if len(by_rank) != len(streams) or set(by_rank) != set(replay_graphs) or 0 not in by_rank:
        return False, None
    stream0 = by_rank[0]
    if len(replay_graphs) > 1:
        sig0 = (stream0.lanes, stream0.assignments, stream0.events)
        if any((plan.lanes, plan.assignments, plan.events) != sig0 for plan in streams):
            return False, None
    if not _stream_plan_full_main_budget(context, stream0, replay_graphs[0]):
        return False, None
    touched_ids: set[str] = {oid for oid, _lane in getattr(stream0, "assignments", ())}
    for event in getattr(stream0, "events", ()):
        touched_ids.add(event.record_after)
        touched_ids.update(event.wait_before)
    if any(
        (bkey := _body_layer_key(op)) is not None
        and bkey[0] != "target"
        and (op.op_id in touched_ids or (op.stream or MAIN_STREAM) != MAIN_STREAM)
        for op in replay_graphs[0]
    ):
        return False, None
    return True, stream0


def _report(case, graphs, context, book, replay, *, deadline=None, fast=False, trace_limit=0, streams=(), cut=False):
    materialized = {rank: _materialize_control_deps(ops) for rank, ops in graphs.items()}
    external = {dep: 0.0 for ops in materialized.values()
                for dep in _external_dependencies(ops) if cut or dep == "analysis_input"}
    replay_graphs = {rank: _replay_ops(ops) for rank, ops in materialized.items()}
    eff_trace = max(trace_limit, replay.event_limit) if not fast else 0
    token = ACTIVE_DEADLINE_MONO.set(deadline)
    try:
        can_cohort, stream0 = (
            _resolve_cohort_streams(replay_graphs, context, streams, book, eff_trace)
            if (not cut and not replay.isolated and 0 in replay_graphs)
            else (False, None)
        )
        sym = _graphs_symmetric_for_dedup(replay_graphs) if (can_cohort and len(replay_graphs) > 1) else None
        if can_cohort and (
            len(replay_graphs) == 1
            or (sym is not None and _replicas_hardware_symmetric(case, context, tuple(sorted(replay_graphs))))
        ):
            refined = refine_layer_cohort_plan(case, replay_graphs[0], stream_plan=stream0)
            prewarm_cohort_diff_ops(case, replay_graphs[0], refined, context, book)
            report, reduced_map, op_rep_map = execute_native_cohort_graph(
                case, replay_graphs, cohort_plan=refined, recipes=book, external_inputs=external,
                include_submission=replay.include_submission, isolated=False,
                trace_limit=eff_trace, compact=False, evaluation=context,
                fast_span_only=fast, stream_plans=(streams or None),
                timeline_replicas=(0,) if len(replay_graphs) == 1 else None,
            )
            if len(replay_graphs) > 1 and not fast:
                report = _mirror_dedup_directional_evidence(case, context, replay_graphs, report)
            if reduced_map is not None and op_rep_map:
                if sym is not None:
                    report = _expand_symmetric_peer_timeline(report, replay_graphs, sym[0], sym[1])
                if eff_trace > 0:
                    report = _expand_cohort_service_spans(report, replay_graphs, op_rep_map, context=context)
                if not fast:
                    # Full cohort replay in ``layer_runs._rebuild_full_cohort_counters``
                    # already scales ``execution["serviced"]`` and ``resource_busy_ms``
                    # to exact full depth; only fast-span reports need expansion here.
                    return report
                expanded = _expand_cohort_serviced(
                    case, replay_graphs, report, op_rep_map, context, book, replay,
                )
                if expanded is not None:
                    return expanded
            else:
                return report
        report = graph_costs(
            case, replay_graphs,
            recipes=book, external_inputs=external, evaluation=context,
            isolated=replay.isolated and not cut, include_submission=replay.include_submission,
            trace_limit=eff_trace,
            fast_span_only=fast, stream_plans=streams or None,
        )
        if len(replay_graphs) > 1 and not fast and eff_trace == 0 and not streams:
            report = _mirror_dedup_directional_evidence(case, context, replay_graphs, report)
        return report
    finally:
        ACTIVE_DEADLINE_MONO.reset(token)


def _normalize_cut_streams(streams, ordinals_by_rank):
    norm = []
    for plan in streams:
        ords = ordinals_by_rank.get(plan.rank, {})
        assignments = tuple((ords.get(oid, oid), lane) for oid, lane in plan.assignments)
        events = tuple(
            (
                ords.get(ev.record_after, ev.record_after),
                tuple(ords.get(target, target) for target in ev.wait_before),
                getattr(ev, "event_id", ""),
            )
            for ev in plan.events
        )
        norm.append((plan.rank, plan.lanes, assignments, events))
    return tuple(norm)


def _cut_layer_key(graph, book, streams, need_eval):
    ranks = []
    ordinals_by_rank = {}
    for rank, ops in sorted(graph.items()):
        ordinals = {op.op_id: index for index, op in enumerate(ops)}
        ordinals_by_rank[rank] = ordinals
        op_sigs = []
        for op in ops:
            recipe = book.recipes.get(op.recipe) if op.recipe else None
            recipe_sig = (type(recipe).__name__, repr(recipe)) if recipe is not None else None
            if need_eval:
                op_sigs.append((
                    op.op_id, op.family, op.recipe, recipe_sig, op.dtype,
                    op.local_shape, op.stage, op.phase, op.stream, op.flops, op.vector_ops,
                    op.mte2_bytes, op.mte3_bytes, op.hbm_bytes, op.link_bytes, op.link_scope,
                    op.matrix_flops, op.layout, op.source_boundary, op.returns,
                    tuple(sorted(op.metadata, key=lambda pair: pair[0])),
                    tuple(ordinals.get(dep, -1) for dep in op.deps),
                ))
            else:
                op_sigs.append((
                    physical_key(op), op.stage, op.stream, op.recipe, recipe_sig,
                    op.phase, op.returns,
                    tuple(ordinals.get(dep, -1) for dep in op.deps),
                ))
        ranks.append((rank, tuple(op_sigs)))
    norm_streams = streams if need_eval else _normalize_cut_streams(streams, ordinals_by_rank)
    return _json_digest((tuple(ranks), norm_streams, bool(need_eval)), default=_digest_default)

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
def prepare_scenario(
    request: ScenarioRequest,
    *,
    budget: EvalBudget = EvalBudget(),
    cache=None,
    spine=None,
    primitives=None,
    primitive_table=None,
    allow_unbound_formula: bool = False,
) -> ScenarioDossier:
    deadline = _deadline(budget, request.replay)
    _check_budget(budget, deadline)
    from mindie_sim.prediction.structure import parse_case
    case = parse_case(request.case)
    prim_table = primitive_table if primitive_table is not None else request.primitive_table
    if isinstance(primitives, PrimitiveStore):
        cache = None
    replay_frame = asdict(replace(request.replay, deadline_mono=None))
    if cache is not None and spine is None:
        request_key = _json_digest((case.to_dict(), asdict(request.step), request.stack_id,
                                    replay_frame, primitives, primitive_table_digest_token(prim_table),
                                    bool(allow_unbound_formula)),
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

    seed_cut_cache: dict[str, int] = {}
    seed_full_cache: dict[str, dict[str, Any]] = {}

    def prepare_seed(resolved, recipes, tier):
        prepared = resolved.atomic_step
        _check_budget(budget, deadline)
        graphs = {entry.replica: entry.ops for entry in prepared.graph.executable}
        full_key = _cut_layer_key(graphs, recipes, prepared.graph.streams, False)
        report = seed_full_cache.get(full_key)
        if report is None:
            report = _report(prepared.case, graphs, context, recipes, request.replay, deadline=deadline,
                             streams=prepared.graph.streams,
                             trace_limit=MAX_OVERLAP_TRACE_EVENTS if prepared.graph.streams else 0)
            seed_full_cache[full_key] = report
        seed_plan_sha = plan_sha256(empty_plan(resolved.base_graph_sha256, resolved.framing_sha256))
        evaluation = _evaluation(
            resolved, graphs, report, prepared.graph_id, tier, prepared,
            scenario_sha256=resolved.scenario_sha256,
            plan_sha256=seed_plan_sha,
        )
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
        covered_layers = {layer for item in archetypes for layer in item.layer_indices}
        for layer in sorted(covered_layers):
            _check_budget(budget, deadline)
            cut = layers[layer]
            streams = _cut_plans(cut, resolved.stream_plans)
            cut_key = _cut_layer_key(cut, recipes, streams, False)
            span_ns = seed_cut_cache.get(cut_key)
            if span_ns is None:
                cut_report = _report(prepared.case, cut, context, recipes, request.replay, deadline=deadline,
                                     fast=True, cut=True, streams=streams)
                span_ns = round(cut_report["duration_ms"] * NS_PER_MS)
                seed_cut_cache[cut_key] = span_ns
            reference[layer] = span_ns
        return _Seed(resolved, recipes, evaluation, archetypes, layers, dict(sites),
                     {rank: _work_totals(ops) for rank, ops in graphs.items()},
                     {layer: {rank: _work_totals(ops) for rank, ops in cut.items()} for layer, cut in layers.items()},
                     reference, sum(reference.values()))

    digest = _json_digest((scenario.scenario_sha256, replay_frame, bool(allow_unbound_formula)))
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
                            replace(request.replay, deadline_mono=None),
                            allow_unbound_formula=bool(allow_unbound_formula))
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
    eplb_damped = meta.get("routing_eplb_active")
    zero_experts = int(meta.get("zero_experts", 0) or 0)
    target = (meta["expected_rows"], meta["expected_active_experts"])

    candidates = []
    if ids:
        candidates.append(
            lambda t: tuple(fsum(vals[i] for i in ids) for vals in policy.expert_moments(model, layer, t))
        )
    else:
        owner = meta.get("ep_rank")
        if owner is not None:
            candidates.append(
                lambda t, _o=owner: policy.local(
                    model, layer, parallel, t, owner=_o, runtime=case.runtime,
                    eplb_damped=eplb_damped, zero_experts=zero_experts,
                )
            )
        candidates.append(
            lambda t: policy.local(
                model, layer, parallel, t, owner=None, runtime=case.runtime,
                eplb_damped=eplb_damped, zero_experts=zero_experts,
            )
        )

    raw_valid = meta.get("routing_valid_tokens")
    valid_tokens = int(raw_valid) if isinstance(raw_valid, int) and not isinstance(raw_valid, bool) and raw_valid >= 2 else 0
    for moments in candidates:
        per_token, _ = moments(1)
        token_guesses = []
        if valid_tokens >= 2:
            token_guesses.append(valid_tokens)
        inferred = round(meta["expected_rows"] / per_token) if per_token else 0
        if inferred >= 2 and inferred not in token_guesses:
            token_guesses.append(inferred)
        for tokens in token_guesses:
            if all(isclose(a, b) for a, b in zip(moments(tokens), target)):
                return moments(tokens // 2), moments(tokens - tokens // 2)
    raise ModelingError(f"rewrite_dbo_routing_unbound:{op.op_id}")

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

def _bind_collective_op(op, replica, binding, topology, registered):
    recipes = {"ring": RingCollective, "pairwise": PairwiseCollective, "oneshot": OneShotCollective}
    meta = dict(op.metadata)
    groups = topology.groups(meta.get("group_axis", "tp"), replica,
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
    return replace(op, recipe=identifier)


def _op_or_absorbed_uses_simt(op: OpSpec, absorbed_ops: Sequence[OpSpec]) -> bool:
    for item in (op, *absorbed_ops):
        meta = dict(item.metadata) if item.metadata else {}
        if meta.get("uses_simt") or meta.get("tile_uses_simt") or meta.get("template_param_uses_simt"):
            return True
        backend = str(meta.get("backend_target") or meta.get("backend") or "").strip().lower()
        if backend in _SIMT_BACKEND_TARGETS:
            return True
    return False


def _prepare_fused_owner_pipe_work(op: OpSpec, *, uses_simt: bool) -> OpSpec:
    """Apply split ``mte2_aic``/``mte2_aiv``, ``fixpipe``, & SIMT flags on Engine replay copy."""
    meta = dict(op.metadata) if op.metadata else {}
    aic_b = float(meta.get("fused_mte2_aic_bytes", 0.0) or 0.0)
    aiv_b = float(meta.get("fused_mte2_aiv_bytes", 0.0) or 0.0)
    fixpipe_b = float(meta.get("fixpipe_bytes", 0.0) or 0.0)
    changed = False
    new_mte2 = op.mte2_bytes
    res_work = dict(meta.get("resource_work") or {})
    if aic_b > 0.0 and aiv_b > 0.0:
        res_work["mte2_aic"] = float(res_work.get("mte2_aic", 0.0) or 0.0) + aic_b
        res_work["mte2_aiv"] = float(res_work.get("mte2_aiv", 0.0) or 0.0) + aiv_b
        new_mte2 = 0.0
        changed = True
    if fixpipe_b > 0.0 and float(res_work.get("fixpipe", 0.0) or 0.0) <= 0.0:
        res_work["fixpipe"] = fixpipe_b
        changed = True
    if changed:
        meta["resource_work"] = res_work
    if uses_simt and not meta.get("tile_uses_simt"):
        meta["tile_uses_simt"] = True
        changed = True
    if not changed:
        return op
    return replace(op, mte2_bytes=new_mte2, metadata=tuple(meta.items()))


def _bind_formula_fused_recipe(scenario, atomic_entry, op, applied):
    """Build a white-box L0-pipe ``FusedPipelineRecipe`` from absorbed atomic ops."""
    by_id = {atom.op_id: atom for atom in atomic_entry.ops}
    absorbed_ids = dict(applied.absorbed_atoms).get(op.op_id, ())
    absorbed_ops = tuple(by_id[oid] for oid in absorbed_ids if oid in by_id) or (op,)
    formula = FormulaKernelSpec.from_ops(absorbed_ops, name=op.recipe or op.op_id)
    uses_simt = formula.uses_simt or _op_or_absorbed_uses_simt(op, absorbed_ops)
    if uses_simt and not formula.uses_simt:
        formula = replace(formula, uses_simt=True)
    datapath = _path(scenario, atomic_entry.replica, op.stage)
    if datapath is None:
        raise ModelingError(f"MULTI_POOL_CAPACITY:{op.op_id}:ub_datapath_missing")
    _, algo, _, ub_pk, ub_cap, l1_pk, l1_cap, l0_pk, l0_cap, solved_depth, sram_ok = (
        solve_formula_sram(formula, datapath)
    )
    if not sram_ok:
        raise ModelingError(
            f"MULTI_POOL_CAPACITY:{op.op_id}:{algo}:ub={ub_pk}>{ub_cap}"
            f":l1={l1_pk}/{l1_cap}:l0={l0_pk}/{l0_cap}"
        )
    depth = max(1, int(solved_depth))
    return FusedPipelineRecipe(
        slots=depth,
        row_reduction=formula.row_reduction,
        abi_signature_id=formula.resolved_abi_signature(),
        binding_state="unbound_formula_projection",
        pipe_stages=formula.pipe_stages,
    ), uses_simt


_SHIPPING_ABI_RECIPE_NAMES: dict[str, tuple[str, ...]] = {
    "aclnnAddRmsNormDynamicQuant": ("add_rmsnorm_quant",),
    "aclnnAddRmsNormDynamicMxQuant": ("add_rmsnorm_quant",),
    "aclnnSwiGluQuant": ("swiglu_quantize",),
    "aclnnDequantSituQuant": ("situ_quantize",),
    "aclnnAllGatherMatmulV2": ("mc2_all_gather_matmul",),
    "aclnnMatmulReduceScatterV2": ("mc2_matmul_reduce_scatter",),
    "aclnnMlaPrologV4WeightNz": ("mla_prolog_v4_arch35",),
    "aclnnMlaPrologV3WeightNz": ("mla_prolog_v4_arch35",),
    "aclnnMegaMoe": ("megamoe_arch35",),
}


def _recipe_content_digest(value: Any) -> str:
    payload = _canon_digest_value(value)
    return hashlib.sha256(repr(payload).encode("utf-8")).hexdigest()[:_RECIPE_DIGEST_LEN]


def _context_replica_stage(context: Any) -> tuple[int, int] | None:
    """Extract executing ``(replica, stage)`` from bound physical resource/pool names."""
    for mapping in (
        getattr(context, "resource_names", None),
        getattr(context, "pool_names", None),
    ):
        if not isinstance(mapping, Mapping):
            continue
        for name in mapping.values():
            if isinstance(name, str):
                match = _CTX_SITE_RE.search(name)
                if match is not None:
                    return int(match.group(1)), int(match.group(2))
    return None


@dataclass(frozen=True, slots=True)
class _CatalogSiteBinding:
    die: int
    datapath: Any
    op_id: str
    program: Any
    lower: Callable[..., Any]


def _make_dispatched_catalog_recipe(
    work: Work, sites: dict[tuple[int, int], _CatalogSiteBinding],
) -> PlannedRecipe:
    """Build one DP-dedup-safe PlannedRecipe dispatching each executing site to its own die."""
    def build(_op, _geometry):
        def program(context):
            key = _context_replica_stage(context)
            site = sites.get(key) if key is not None else None
            if site is None:
                if len(sites) == 1:
                    site = next(iter(sites.values()))
                else:
                    raise ModelingError(f"rewrite_catalog_site_unbound:{key}")
            dp = context.datapath or site.datapath
            if hasattr(site.program, "with_context"):
                bound = site.program.with_context(dp, context)
            else:
                bound = site.lower(dp, context=context, die=site.die, region_id=site.op_id).program
            return _CatalogReplayProgram(bound) if getattr(bound, "hbm_bus", None) is not None else bound

        return OperatorPlan(work, program)

    return PlannedRecipe(build)


def _dp_catalog_recipe_id(
    op: OpSpec, template: Any, summary: Any, work: Work, path: Any, limits: tuple[int, int],
) -> str:
    """Content-address catalog recipes independent of replica/die so symmetric DP dedup holds."""
    norm_summary = replace(
        summary,
        semantic_work=tuple(replace(item, die=0, replica=0) for item in summary.semantic_work),
        read_obligations=tuple(replace(item, die=0) for item in summary.read_obligations),
        write_obligations=tuple(replace(item, die=0) for item in summary.write_obligations),
    )
    norm_template = replace(template, replica=0) if hasattr(template, "replica") else template
    digest = _recipe_content_digest((norm_summary, work, norm_template, _norm_catalog_datapath(path), limits))
    return f"rewrite_template:{op.op_id}:{digest}"


def _catalog_program_work(result: Any) -> Work:
    resources: Counter[str] = Counter()
    if hasattr(result.program, "resource_work"):
        for role, amount in result.program.resource_work:
            if role != "cube":
                resources[_PIPE_ROLE_MAP.get(role, role)] += amount
    else:
        for stage in result.program.stages:
            for role, amount in stage.work.items():
                if role != "cube":
                    resources[_PIPE_ROLE_MAP.get(role, role)] += amount
    return Work(dict(result.summary.issued_cube_flops_by_dtype), resources)


def _bind_catalog_site(
    scenario: ResolvedScenario,
    atomic_entry: Any,
    op: OpSpec,
    template: Any,
    applied: AppliedPlan,
    stream: Any,
    catalog_memo: dict[Any, Any],
    catalog_sites: dict[str, dict[tuple[int, int], _CatalogSiteBinding]],
    registered: dict[str, Any],
) -> tuple[str, Any] | None:
    """Bind one K2/K3 catalog op with cross-layer/replica memo and exact per-die dispatch."""
    resolved = _memoized_catalog_lowering(
        scenario, atomic_entry, op, template, applied, stream, catalog_memo, need_program=True,
    )
    if resolved is None:
        return None
    kind = resolved[0]
    if kind == "explicit":
        _, recipe, summary = resolved[0], resolved[1], resolved[2]
        identifier = catalog_recipe_id(atomic_entry.replica, op, template, summary)
        registered[identifier] = recipe
        return identifier, summary
    _, site_prog, summary, path, limits, die, site_lower, result = resolved
    work = _catalog_program_work(result)
    identifier = _dp_catalog_recipe_id(op, template, summary, work, path, limits)
    sites = catalog_sites.get(identifier)
    if sites is None:
        sites = {}
        catalog_sites[identifier] = sites
        registered[identifier] = _make_dispatched_catalog_recipe(work, sites)
    sites[(atomic_entry.replica, op.stage)] = _CatalogSiteBinding(
        die=die, datapath=path, op_id=op.op_id, program=site_prog, lower=site_lower,
    )
    return identifier, summary


def _resolve_fused_binding(scenario, atomic_entry, op, applied, book, catalog, handle, registered, *, allow_unbound_formula):
    """Enforce 3-tier resolution: UserProfile > Extension/RecipeBook > FormulaKernelSpec."""
    edit_op = next(
        edit.operation for edit in applied.plan.edits
        if op.op_id == f"rewrite:{edit.edit_id}"
    )
    boundaries = edit_op.kernel_boundaries
    has_measured_store = isinstance(handle.evaluation_context.cost.primitives, PrimitiveStore)
    if len(boundaries) == 1 and (catalog or op.recipe in book.recipes):
        state = "bound_measured" if (has_measured_store and op.recipe in book.recipes) else "bound_implementation"
        return op, state, False
    if (
        len(boundaries) == 1
        and op.recipe in {"gemm", "aclnnMatmul", "aclnnQuantMatmulV4"}
        and dict(op.metadata).get("law2_block_gemm") is True
        and op.flops > 0
    ):
        selected = book.select(replace(op, family="gemm", recipe="gemm"))
        if not isinstance(selected, FusedPipelineRecipe):
            sig = _recipe_content_digest((type(selected).__qualname__, repr(selected)))
            identifier = f"rewrite_law2_gemm:{sig}"
            registered[identifier] = selected
            state = "bound_measured" if has_measured_store else "bound_implementation"
            return replace(op, recipe=identifier), state, False
    if (
        len(boundaries) == 1
        and edit_op.fusion_tier in {"K1_shipping", "K1_proposed_pr"}
        and op.recipe in _SHIPPING_ABI_RECIPE_NAMES
    ):
        for candidate_key in _SHIPPING_ABI_RECIPE_NAMES[op.recipe]:
            selected = book.recipes.get(candidate_key) or book.families.get(candidate_key)
            if selected is not None and not isinstance(selected, FusedPipelineRecipe):
                identifier = f"rewrite_k1_abi:{candidate_key}"
                registered[identifier] = selected
                state = "bound_measured" if has_measured_store else "bound_implementation"
                return replace(op, recipe=identifier), state, False
    if len(boundaries) != 1 or not allow_unbound_formula:
        raise ModelingError(f"rewrite_fusion_recipe_unbound:{op.op_id}:{op.recipe}")
    formula_recipe, uses_simt = _bind_formula_fused_recipe(scenario, atomic_entry, op, applied)
    sig = _recipe_content_digest(formula_recipe)
    identifier = f"rewrite_formula:{op.recipe}:{sig}"
    registered[identifier] = formula_recipe
    bound_op = _prepare_fused_owner_pipe_work(replace(op, recipe=identifier), uses_simt=uses_simt)
    return bound_op, "unbound_formula_projection", True


def _bind_replay(scenario, applied, book, handle, *, allow_unbound_formula: bool = False):
    """Bind the selected executable recipes and collect catalog audit summaries."""
    templates = {
        (getattr(binding, "replica", 0), binding.target_op): binding
        for binding in applied.template_bindings
    }
    atomic = {entry.replica: entry for entry in scenario.atomic_ops}
    summaries, unbound_ops, binding_states = [], [], []
    tiles = {
        (getattr(binding, "replica", 0), binding.target_op): binding
        for binding in applied.tile_bindings
    }
    streams = {stream.rank: stream for stream in applied.streams}
    registered, graphs = {}, {}
    catalog_memo = handle._catalog_memo
    catalog_sites: dict[str, dict[tuple[int, int], _CatalogSiteBinding]] = {}
    topology = handle.evaluation_context.bindings.topology
    collective = {
        (getattr(binding, "replica", 0), binding.target_op): binding
        for binding in applied.collective_rewrites
    }
    elided_sites = getattr(applied, "elided_d2h_sites", ())
    for entry in applied.executable:
        bound = []
        rank_elided = (
            tuple(oid for rep, oid in elided_sites if rep == entry.replica)
            if elided_sites else applied.elided_d2h_ops
        )
        ops, book = (_device_counts(entry.ops, rank_elided, book,
                                    scenario.atomic_step.case, handle._replay.include_submission)
                     if rank_elided else (entry.ops, book))
        for op in ops:
            site = (entry.replica, op.op_id)
            template = templates.get(site)
            catalog = _bind_catalog_site(
                scenario, atomic[entry.replica], op, template, applied,
                streams.get(entry.replica), catalog_memo, catalog_sites, registered,
            )
            if catalog:
                identifier, summary = catalog
                summaries.append(summary)
                op = replace(op, recipe=identifier)
            elif template:
                is_law2_gemm_tpl = (
                    template.template_id == "gemm"
                    and op.family == "rewrite_fused"
                    and dict(op.metadata).get("law2_block_gemm") is True
                )
                if template.template_id not in book.recipes and not is_law2_gemm_tpl:
                    raise ModelingError(f"rewrite_template_recipe_unbound:{op.op_id}:{template.template_id}")
                op = replace(op, recipe=template.template_id)
            if (col_binding := collective.get(site)) is not None:
                op = _bind_collective_op(op, entry.replica, col_binding, topology, registered)
            if op.family == "rewrite_fused":
                op, state, is_unbound = _resolve_fused_binding(
                    scenario, atomic[entry.replica], op, applied, book, catalog, handle, registered,
                    allow_unbound_formula=allow_unbound_formula,
                )
                binding_states.append((op.op_id, state))
                if is_unbound:
                    unbound_ops.append(op.op_id)
            binding = tiles.get(site)
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
            # Register the base matrix; selection will apply its grouped adapter once.
            reg_recipe = recipe.matrix if isinstance(recipe, ExpectedGroupedMatmul) else recipe
            if isinstance(reg_recipe, PlannedRecipe) and op.recipe in registered:
                identifier = op.recipe
            else:
                sig = _recipe_content_digest((type(reg_recipe).__qualname__, repr(reg_recipe)))
                identifier = f"rewrite_tile:{sig}"
                registered[identifier] = reg_recipe
            bound.append(replace(op, recipe=identifier))
        graphs[entry.replica] = tuple(bound)
    for identifier, recipe in registered.items():
        book = register_recipe(book, identifier, recipe)
    if applied.prefetch_leases:
        for replica, ops in graphs.items():
            graphs[replica], book = bind_prefetch_ops(
                scenario.atomic_step.case, replica, ops, book, applied.prefetch_leases,
                lambda stage, r=replica: handle._prefetch_hardware_for(r, stage),
            )
    bound_streams = bind_prefetch_streams(graphs, applied.streams, applied.prefetch_leases)
    return (
        graphs, book, tuple(summaries), bound_streams,
        tuple(dict.fromkeys(unbound_ops)), tuple(dict.fromkeys(binding_states)),
    )
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

def _remap_cut_evaluation(row: Any, src_graph: Mapping[int, Sequence[OpSpec]], dst_graph: Mapping[int, Sequence[OpSpec]]) -> Any:
    """Remap a cut-layer evaluation onto a structurally identical peer layer."""
    id_map: dict[tuple[int, str], str] = {}
    for rank, src_ops in src_graph.items():
        dst_ops = dst_graph.get(rank, ())
        for src_op, dst_op in zip(src_ops, dst_ops):
            id_map[(rank, src_op.op_id)] = dst_op.op_id
    new_ops = tuple(
        replace(
            item,
            occurrence=replace(
                item.occurrence,
                op_id=id_map.get((item.occurrence.replica, item.occurrence.op_id), item.occurrence.op_id),
            ),
            predecessors=tuple(
                replace(pred, op_id=id_map.get((pred.replica, pred.op_id), pred.op_id))
                for pred in item.predecessors
            ),
        )
        for item in row.operators
    )
    new_service = {
        (rep, id_map.get((rep, oid), oid)): spans
        for (rep, oid), spans in row.service_intervals.items()
    }
    new_cohorts = {
        (rep, id_map.get((rep, oid), oid)): groups
        for (rep, oid), groups in row.cohort_replicas.items()
    }
    return replace(row, operators=new_ops, service_intervals=new_service, cohort_replicas=new_cohorts)


def _stream_main_budget_full(scenario: ResolvedScenario, plan: Any, replica: int, stages: set[int]) -> bool:
    if plan is None:
        return True
    for stage in stages:
        path = _path(scenario, replica, stage)
        owners = bind_stream_owners(plan, path).get(MAIN_STREAM)
        if (
            owners is None
            or owners.isolation != "shared_cap"
            or owners.aic_limit < path.aic_count
            or owners.aiv_limit < path.aiv_count
        ):
            return False
    return True


def _stream_changed_layers(
    scenario: ResolvedScenario,
    graphs: Mapping[int, Sequence[OpSpec]],
    bound_streams: Sequence[Any],
    domain: set[int],
) -> set[int]:
    """Return layers actually affected by StreamPlan differences vs ``scenario.stream_plans``."""
    orig_by_rank = {plan.rank: plan for plan in scenario.stream_plans}
    new_by_rank = {plan.rank: plan for plan in bound_streams}
    affected: set[int] = set()
    for rank in orig_by_rank.keys() | new_by_rank.keys():
        orig_plan = orig_by_rank.get(rank)
        new_plan = new_by_rank.get(rank)
        if orig_plan == new_plan:
            continue
        ops = graphs.get(rank, ())
        stages = {int(op.stage) for op in ops}
        orig_main = next((lane for lane in orig_plan.lanes if lane.stream_id == MAIN_STREAM), None) if orig_plan else None
        new_main = next((lane for lane in new_plan.lanes if lane.stream_id == MAIN_STREAM), None) if new_plan else None
        main_unchanged = (
            (orig_plan is not None and new_plan is not None and orig_main == new_main)
            or (
                _stream_main_budget_full(scenario, orig_plan, rank, stages)
                and _stream_main_budget_full(scenario, new_plan, rank, stages)
            )
        )
        if not main_unchanged:
            return set(domain)
        touched_ids: set[str] = set()
        for plan in (orig_plan, new_plan):
            if plan is None:
                continue
            for oid, lane in plan.assignments:
                if lane != MAIN_STREAM:
                    touched_ids.add(oid)
            for event in plan.events:
                touched_ids.add(event.record_after)
                touched_ids.update(event.wait_before)
        if touched_ids:
            affected.update(int(op.layer) for op in ops if op.op_id in touched_ids)
    return affected


def _effective_cut_streams(
    scenario: ResolvedScenario, graph: Mapping[int, Sequence[OpSpec]], cut_streams: tuple[Any, ...],
) -> tuple[Any, ...]:
    """Drop no-op cut StreamPlans (zero non-main assignments/events, full main budget)."""
    if not cut_streams:
        return ()
    for plan in cut_streams:
        ops = graph.get(plan.rank, ())
        stages = {int(op.stage) for op in ops}
        if (
            plan.events
            or any(lane != MAIN_STREAM for _, lane in plan.assignments)
            or not _stream_main_budget_full(scenario, plan, plan.rank, stages)
        ):
            return cut_streams
    return ()


def _estimate(handle, seed, scenario, applied, budget, deadline, *, allow_unbound_formula: bool = False):
    baseline = seed.evaluation
    baseline_ns = round(baseline.span_ms * NS_PER_MS)
    # Quantization declarations are compatibility assertions against the
    # prepared representation. Their annotations leave the physical bound intact.
    if (all(receipt.status == "noop" for receipt in applied.edit_receipts)
            or all(isinstance(edit.operation, QuantizeContract) for edit in applied.plan.edits)):
        return baseline_ns, seed.total_ns, baseline, baseline, None, (), (), ()
    graphs, book, summaries, bound_streams, unbound_ops, binding_states = _bind_replay(
        scenario, applied, seed.book, handle, allow_unbound_formula=allow_unbound_formula,
    )
    layers = {replica: _layers(ops) for replica, ops in graphs.items()}
    domain = {op.layer for rank in scenario.atomic_ops for op in rank.ops}
    stream_layers = _stream_changed_layers(scenario, graphs, bound_streams, domain)
    changed = sorted(layer for layer in domain if layer in stream_layers or any(
        tuple(by_layer.get(layer, ())) != seed.layers[layer][rank] for rank, by_layer in layers.items()
    ))
    if not seed.reference.keys() >= set(changed):
        raise ModelingError("rewrite_edited_layer_outside_archetype_coverage")
    if (applied.prefetch_leases and not bound_streams and not handle._replay.include_submission
            and all(isinstance(edit.operation, WeightPrefetch) for edit in applied.plan.edits)):
        # Screen the extra DMA service per replica using each die's own HBM bandwidth.
        extra_ms = max(
            (fsum(
                op.hbm_bytes / handle._prefetch_hardware_for(rep, op.stage).hbm_bytes_per_ms
                for op in ops if op.family == "prefetch"
            ) for rep, ops in graphs.items()),
            default=0.0,
        )
        extra_ns = round(extra_ms * NS_PER_MS)
        return baseline_ns + extra_ns, seed.total_ns + extra_ns, baseline, baseline, None, summaries, unbound_ops, binding_states
    archetype_ns = seed.total_ns
    rows, comparisons = [], []
    overlap = any(len(plan.lanes) > 1 for plan in bound_streams) or any(
        tile.cohort_split == "dbo_2cohort" for tile in applied.tile_bindings)
    rebind = any(isinstance(edit.operation, (FuseRegion, SelectTemplate)) for edit in applied.plan.edits)
    need_eval = overlap or rebind
    catalog_layers = {
        op.layer
        for summary in summaries
        for rank_entry in applied.executable
        for op in rank_entry.ops
        if op.op_id == summary.region_id
    }
    grouped_cuts: OrderedDict[tuple[Any, ...], list[tuple[int, dict[int, tuple[Any, ...]], tuple[Any, ...]]]] = OrderedDict()
    for layer in changed:
        graph = {rank: tuple(by_layer.get(layer, ())) for rank, by_layer in layers.items()}
        cut_streams = _effective_cut_streams(scenario, graph, _cut_plans(graph, bound_streams))
        if layer in catalog_layers:
            gkey: tuple[Any, ...] = ("unique", layer)
        else:
            after_sig = _cut_layer_key(graph, book, cut_streams, False)
            if rebind:
                orig_graph = seed.layers[layer]
                before_sig = _cut_layer_key(
                    orig_graph, seed.book, _cut_plans(orig_graph, scenario.stream_plans), False,
                )
            else:
                before_sig = ""
            gkey = (after_sig, before_sig, int(seed.reference[layer]))
        grouped_cuts.setdefault(gkey, []).append((layer, graph, cut_streams))
    dbo_targets = {
        (getattr(tile, "replica", 0), tile.target_op)
        for tile in applied.tile_bindings if tile.cohort_split == "dbo_2cohort"
    }
    for members in grouped_cuts.values():
        _check_budget(budget, deadline)
        layer, graph, cut_streams = members[0]
        mult = len(members)
        layer_overlap = any(len(plan.lanes) > 1 for plan in cut_streams) or (
            bool(dbo_targets) and any(
                (rep, op.op_id) in dbo_targets for rep, ops in graph.items() for op in ops
            )
        )
        span, row = _replay_cut(handle, seed, graph, book, cut_streams,
                                need_eval, MAX_OVERLAP_TRACE_EVENTS if layer_overlap else 0, deadline)
        reference = seed.reference[layer]
        comparison = None
        if rebind:
            original = seed.layers[layer]
            reference, comparison = _replay_cut(handle, seed, original, seed.book,
                _cut_plans(original, scenario.stream_plans), True, 0, deadline)
            comparisons.append(comparison)
        archetype_ns += (span - reference) * mult
        if row is not None:
            rows.append(row)
            if overlap and mult > 1:
                for other_layer, other_graph, _ in members[1:]:
                    rows.append(_remap_cut_evaluation(row, graph, other_graph))
                    if comparison is not None:
                        comparisons.append(_remap_cut_evaluation(comparison, seed.layers[layer], seed.layers[other_layer]))
    return (max(0, baseline_ns + archetype_ns - seed.total_ns), archetype_ns,
            _combine_replays(rows) or baseline, _combine_replays(comparisons) or baseline,
            (graphs, book, seed.book, handle.evaluation_context, deadline), summaries,
            unbound_ops, binding_states)

def evaluate_rewrites(
    scenario: ScenarioHandle,
    candidates: Sequence[RewritePlan],
    *,
    budget: EvalBudget = EvalBudget(),
    allow_unbound_formula: bool | None = None,
) -> BatchEvaluation:
    started = monotonic()
    deadline = _deadline(budget, scenario._replay)
    coverage = _coverage(scenario)
    use_formula = scenario.allow_unbound_formula if allow_unbound_formula is None else bool(allow_unbound_formula)
    results = []
    for plan in islice(candidates, budget.max_candidates):
        digest = ""
        cache_key = None
        try:
            digest = plan_sha256(plan)
            cache_key = (digest, use_formula)
            _check_budget(budget, deadline)
            with scenario._cache_lock:
                hit = scenario._candidates.get(cache_key)
                if hit is not None:
                    scenario._candidates.move_to_end(cache_key)
            if hit is not None:
                _check_budget(budget, deadline)
                results.append(hit)
                continue
            seed = _baseline(scenario, plan)
            resolved, work = _screen_scenario(seed, plan)
            applied = apply_plan(plan, resolved)
            wall, archetype_wall, evidence, comparison, service_replay, summaries, unbound_ops, b_states = _estimate(
                scenario, seed, resolved, applied, budget, deadline, allow_unbound_formula=use_formula,
            )
            verification = verify_plan(resolved, applied, step_eval=evidence, wall_ns=wall,
                scope="archetype", baseline_eval=comparison, baseline_work=work, service_replay=service_replay,
                region_summaries=summaries, catalog_memo=scenario._catalog_memo)
            _check_budget(budget, deadline)
            failure = next((gate.reason for gate in verification.gates if not gate.passed), "")
            coverage = len(seed.reference) / len(seed.layers) if seed.layers else 1.0
            result = CandidateEvaluation(
                digest, "gate_failed" if failure else "ok", "archetype",
                wall, archetype_wall, coverage, verification, failure,
                has_unbound_formula_ops=bool(unbound_ops),
                unbound_formula_ops=unbound_ops,
                binding_states=b_states,
            )
        except TimeoutError as error:
            result = CandidateEvaluation(digest, "cancelled", "archetype", None, None, coverage, None, str(error))
        except (ValueError, TypeError) as error:
            result = CandidateEvaluation(digest, "invalid", "archetype", None, None, coverage, None, str(error))
        if cache_key is not None and result.status != "cancelled":
            with scenario._cache_lock:
                _memo_put(scenario._candidates, cache_key, result)
        results.append(result)
    return BatchEvaluation(scenario.scenario_digest, "archetype", tuple(results), (monotonic() - started) * 1000)

def _pipeline_implementation_for_op(op: OpSpec, edit: Any) -> str:
    """Carry verified physical boundary symbol (or explicit unbound formula tag) onto pipeline receipt."""
    recipe = str(op.recipe or "").strip()
    if recipe.startswith("rewrite_formula:"):
        return recipe
    operation = getattr(edit, "operation", None) if edit is not None else None
    if isinstance(operation, FuseRegion) and len(operation.kernel_boundaries) == 1:
        abi_id = str(operation.kernel_boundaries[0].abi_id or "").strip()
        if abi_id:
            return abi_id
    kb = getattr(op, "kernel_binding", None)
    if kb is not None and getattr(kb, "kernel_id", ""):
        return str(kb.kernel_id)
    return recipe or op.family


def finalize_plan(
    scenario: ScenarioHandle,
    plan: RewritePlan,
    *,
    budget: EvalBudget = EvalBudget(),
    dual_certificate=None,
    dual_program=None,
    event_dag=None,
    primitive_table=None,
    rate_mode="engine_ceiling",
    allow_unbound_formula: bool | None = None,
) -> FullPlanEvaluation:
    started = monotonic()
    deadline = _deadline(budget, scenario._replay)
    _check_budget(budget, deadline)
    use_formula = scenario.allow_unbound_formula if allow_unbound_formula is None else bool(allow_unbound_formula)
    can_memo = dual_certificate is None and dual_program is None and event_dag is None
    fin_key = (
        (plan_sha256(plan), use_formula, str(rate_mode), primitive_table_digest_token(primitive_table))
        if can_memo else None
    )
    if fin_key is not None:
        with scenario._cache_lock:
            cached_eval = scenario._finalized.get(fin_key)
            if cached_eval is not None:
                scenario._finalized.move_to_end(fin_key)
        if cached_eval is not None:
            _check_budget(budget, deadline)
            return cached_eval
    seed = _baseline(scenario, plan)
    resolved, baseline = seed.scenario, seed.evaluation
    applied = apply_plan(plan, resolved)
    graphs, book, summaries, bound_streams, unbound_ops, b_states = _bind_replay(
        resolved, applied, seed.book, scenario, allow_unbound_formula=use_formula,
    )
    overlap = any(len(stream.lanes) > 1 for stream in bound_streams) or any(
        tile.cohort_split == "dbo_2cohort" for tile in applied.tile_bindings)
    can_reuse_baseline = (
        not plan.edits
        and not summaries
        and not applied.absorbed_atoms
        and not applied.tile_bindings
        and not applied.prefetch_leases
        and bound_streams == resolved.stream_plans
        and (not bound_streams or overlap)
    )
    if can_reuse_baseline:
        step = replace(baseline, graph_id=applied.plan_sha256, plan_sha256=applied.plan_sha256)
    else:
        report = _report(resolved.atomic_step.case, graphs, scenario.evaluation_context, book, scenario._replay,
                         deadline=deadline, streams=bound_streams,
                         trace_limit=MAX_OVERLAP_TRACE_EVENTS if overlap else 0)
        absorbed_map = dict(applied.absorbed_atoms)
        edit_by_op_id = {f"rewrite:{edit.edit_id}": edit for edit in applied.plan.edits}
        fused_pipelines = tuple(
            FusedPipelineDescriptor(
                owner=OpOccurrence(replica, op.stage, _die(resolved, replica, op.stage).rank, op.op_id),
                family=op.family,
                implementation=_pipeline_implementation_for_op(op, edit_by_op_id.get(op.op_id)),
                member_op_ids=(op.op_id,),
                absorbed_source_op_ids=tuple(absorbed_map[op.op_id]),
                phases=(),
                sources=(
                    (op.kernel_binding.source_ref,)
                    if getattr(op, "kernel_binding", None) is not None and op.kernel_binding.source_ref is not None
                    else ()
                ),
            )
            for replica, ops in graphs.items()
            for op in ops
            if absorbed_map.get(op.op_id)
        )
        step = _evaluation(
            resolved, graphs, report, applied.plan_sha256,
            baseline.tier if not plan.edits else "should_achieve",
            resolved.atomic_step if not plan.edits else None,
            scenario_sha256=resolved.scenario_sha256,
            plan_sha256=applied.plan_sha256,
            extra_pipelines=fused_pipelines,
        )
    wall = round(step.span_ms * NS_PER_MS)
    if dual_certificate is None and dual_program is None and event_dag is None:
        cacheable = (
            not applied.plan.edits
            and not summaries
            and primitive_table is None
            and rate_mode == "engine_ceiling"
        )
        cached = scenario._certificate.get(resolved.base_graph_sha256) if cacheable else None
        if cached is None:
            if summaries:
                _verify_region_summary_bindings(
                    resolved, applied, summaries, require_catalog=True, catalog_memo=scenario._catalog_memo,
                )
            dual_program, event_dag, _ = build_dual_inputs_from_scenario(
                resolved,
                applied=applied,
                region_summaries=summaries,
                primitive_table=primitive_table,
                rate_mode=rate_mode,
                verify_summaries=not summaries,
            )
            dual_certificate, _ = solve_dual_lower_bound(dual_program, event_dag)
            if cacheable:
                with scenario._cache_lock:
                    scenario._certificate[resolved.base_graph_sha256] = (
                        dual_program, event_dag, dual_certificate,
                    )
        else:
            dual_program, event_dag, dual_certificate = cached
    _check_budget(budget, deadline)
    verification = verify_plan(resolved, applied, step_eval=step, wall_ns=wall,
                               dual_certificate=dual_certificate, dual_program=dual_program,
                               event_dag=event_dag, baseline_u_ns=scenario.full_u_ns,
                               baseline_m_ns=scenario.full_m_ns, baseline_eval=baseline,
                               baseline_work=seed.work, region_summaries=summaries, primitive_table=primitive_table, rate_mode=rate_mode,
                               service_replay=(graphs, book, seed.book, scenario.evaluation_context, deadline),
                               catalog_memo=scenario._catalog_memo)
    result = FullPlanEvaluation(
        scenario.scenario_digest, applied.plan_sha256, "full", step, wall,
        verification, applied, (monotonic() - started) * 1000, summaries,
        has_unbound_formula_ops=bool(unbound_ops),
        unbound_formula_ops=unbound_ops,
        binding_states=b_states,
    )
    if fin_key is not None:
        with scenario._cache_lock:
            _memo_put(scenario._finalized, fin_key, result)
    return result

__all__ = [
    "ScenarioRequest", "EvalBudget", "StepArchetype", "ScenarioHandle", "ScenarioDossier",
    "CandidateEvaluation", "BatchEvaluation", "FullPlanEvaluation", "OracleCache",
    "intern_step_archetypes", "prepare_scenario", "evaluate_rewrites", "finalize_plan",
]
