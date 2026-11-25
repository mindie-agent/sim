"""Transactional interpretation of declarative graph edits.

[Role] Bind a rewrite plan to its frozen scenario and transform rank-local DAGs.
[Invariants] Edits run in stable dependency order. Failed edits publish no graph.
  Rewrite annotations are flat and graph handles are kept outside operator metadata.
[Timeouts & Locks] Per-call scratch state, with no shared mutation or locks.
[Side-Effect Policy] Read-only lowering and immutable replacement records.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from graphlib import CycleError, TopologicalSorter
from heapq import heappop, heappush
from math import inf, isfinite, prod
from types import SimpleNamespace
from typing import Any, Literal

from mindie_sim.prediction.body_defaults import consume_target_declaration
from mindie_sim.prediction.fusion.control_rewrite import (
    apply_incoming_controls, rewritten_control_metadata,
)
from mindie_sim.prediction.fusion.ports import (
    CONTROL_METADATA_KEYS, INPUT_PORTS_KEY, control_dependency_names, control_value_names,
    declared_control_order, full_operand_ports, input_port_entries, metadata_pairs,
    replace_input_ports,
)
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.runtime import tile_solver
from mindie_sim.prediction.runtime.tile_solver import TilePipelineSpec
from mindie_sim.prediction.spine import PredictionSpine, PreparedStep
from mindie_sim.prediction.streams import MAIN_STREAM, StreamLane, StreamPlan, bind_stream_owners, lower_stream_edges
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import DatapathCapabilitySpec, datapath_for_die, ub_peer_edges_for_rank
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.results import DEFAULT_STACK_NAME, LoweringRequest, RankOps

from ._oracle_prefetch import die_placements_for_case, primitive_table_digest_token
from .schema import (
    CollectiveTransform, FuseRegion, FusionTier, HoistOrDeduplicate, OpRef,
    PipelineTiles, PortRef, QuantizeContract, RewritePlan, SelectTemplate, StreamSchedule,
    TemplateTier, VariantRef, WeightPrefetch, _digest_default, _flat, _json_digest, _ordered_edits, _references,
    plan_sha256,
)

DEFAULT_STACK_ID = "cann93_va_main"
_FP32_UPCAST_BYTES = 4
_POINTWISE_TILE_ELEMS = 128
_BLOCK_QUANT_TILE_ELEMS = 256
_L1_TILE_K_ELEMS = 64
_L0C_TILE_ELEMS = 128
_GEMM_SHAPE_RANK = 3
_ROW_REDUCTION_FAMILIES = frozenset({
    "rmsnorm", "layernorm", "add_rmsnorm", "add_rmsnorm_quant", "rmsnorm_quant",
    "softmax", "quantize", "attention_residual", "mhc", "l2norm",
})
_DBO_FAMILIES = frozenset({
    "grouped_moe", "grouped_gemm", "moe_gemm", "routed_moe",
    "mc2", "mc2_protocol", "moe", "alltoall", "dispatch", "combine",
})
_COLLECTIVE_FAMILIES = _DBO_FAMILIES | {
    "collective", "allreduce", "all_reduce", "allgather", "all_gather",
    "reduce_scatter", "alltoallv", "p2p", "send", "recv",
}
_HOST_SYNC_FAMILIES = frozenset({"eager_split_barrier", "d2h_sync", "host_d2h_sync"})
_NON_DATA_PRODUCER_FAMILIES = _HOST_SYNC_FAMILIES | frozenset({
    "barrier", "event", "sync", "control", "acceptance",
})
_NON_ACTIVATION_ROLES = frozenset({"weight", "control", "barrier", "event"})
_GROUPED_MATMUL_FAMILIES = frozenset({"grouped_gemm", "grouped_matmul", "grouped_moe"})

@dataclass(frozen=True, slots=True)
class ResolvedScenario:
    case: Case
    step: StepRequest
    stack_id: str
    base_graph_sha256: str
    framing_sha256: str
    scenario_sha256: str
    atomic_step: PreparedStep
    mainline_step: PreparedStep | None
    atomic_ops: tuple[RankOps, ...]
    datapath: DatapathCapabilitySpec
    stream_plans: tuple[StreamPlan, ...] = ()
    die_placements: tuple[tuple[int, tuple[int, int]], ...] = ()
    bilinear_contractions: tuple[Any, ...] = ()
    primitive_table: Any = None
@dataclass(frozen=True, slots=True)
class EditReceipt:
    edit_id: str
    opcode: str
    status: Literal["applied", "noop", "rejected"]
    affected_ops: tuple[str, ...]
    detail: str = ""
@dataclass(frozen=True, slots=True)
class PrefetchLease:
    edit_id: str
    target_op: str
    port_name: str
    window_anchor_op: str
    pool: Literal["l1"]
    max_bytes: int
    replica: int = 0
@dataclass(frozen=True, slots=True)
class TemplateBinding:
    edit_id: str
    target_op: str
    tier: TemplateTier
    template_id: str
    params: tuple[tuple[str, str | int | bool], ...]
@dataclass(frozen=True, slots=True)
class TileBinding:
    edit_id: str
    target_op: str
    cohort_split: Literal["none", "dbo_2cohort"]
    tile_spec: TilePipelineSpec | VariantRef | None
@dataclass(frozen=True, slots=True)
class CollectiveRewriteRecord:
    edit_id: str
    target_op: str
    mode: str
    topology_route: str
@dataclass(frozen=True, slots=True)
class AppliedPlan:
    plan: RewritePlan
    plan_sha256: str
    scenario_sha256: str
    executable: tuple[RankOps, ...]
    streams: tuple[StreamPlan, ...]
    handles: tuple[tuple[str, tuple[str, ...]], ...]
    edit_receipts: tuple[EditReceipt, ...]
    absorbed_atoms: tuple[tuple[str, tuple[str, ...]], ...]
    prefetch_leases: tuple[PrefetchLease, ...]
    template_bindings: tuple[TemplateBinding, ...]
    tile_bindings: tuple[TileBinding, ...]
    hoisted_load_time: tuple[str, ...]
    elided_d2h_ops: tuple[str, ...]
    quant_contracts: tuple[QuantizeContract, ...]
    collective_rewrites: tuple[CollectiveRewriteRecord, ...]
    region_tiers: tuple[tuple[str, FusionTier], ...] = ()
    in_place_aliases: tuple[tuple[str, str], ...] = ()
    credit_edges: tuple[tuple[str, str, int], ...] = ()
def _lower_prepared_step_for_rewrite(
    spine: PredictionSpine,
    case: Case,
    request: LoweringRequest,
) -> PreparedStep:
    """Lower a ``PreparedStep`` for rewrite analysis without building unused structure payloads."""
    if type(spine) is not PredictionSpine:
        return spine.lower(case, request)
    base = spine.stacks.resolve_base(case, request)
    rewritten = spine.rewrite.apply(base)
    effective_case = (
        consume_target_declaration(base.case, rewritten.target_graphs)
        if rewritten.target_graphs is not None else base.case
    )
    active_replicas = frozenset(entry.replica for entry in rewritten.executable)
    bound_owners = spine._bind_stream_plans(effective_case, rewritten.streams, active_replicas)
    book = RecipeBook.from_mapping(effective_case.runtime.kernels, family_registry=spine.families)
    resolved_ops = spine._resolve_governed_operators(effective_case, rewritten, book, bound_owners)
    graph_id = _json_digest(
        (effective_case.model.name, request.tier, request.stack, rewritten.executable, rewritten.streams),
        default=_digest_default,
    )
    return PreparedStep(
        graph=rewritten,
        case=effective_case,
        operators=resolved_ops,
        graph_id=graph_id,
        structure_payload=None,
    )


def resolve_scenario(case, request: StepRequest, *, spine=None,
                     stack_id: str = DEFAULT_STACK_ID, primitives=None,
                     primitive_table=None) -> ResolvedScenario:
    """Lower atomic and shipping graphs against a shared declared scenario."""
    case = Case.from_mapping(case)
    spine = PredictionSpine() if spine is None else spine
    stack = DEFAULT_STACK_NAME if stack_id == DEFAULT_STACK_ID else stack_id
    atomic = _lower_prepared_step_for_rewrite(
        spine, case, LoweringRequest(step=request, tier="unmerge", apply_body_defaults=False, stack=stack),
    )
    mainline = _lower_prepared_step_for_rewrite(
        spine, case, LoweringRequest(step=request, tier="mainline", apply_body_defaults=True, stack=stack),
    )
    graph = atomic.graph
    base = _json_digest(graph.executable, default=_digest_default)
    framing = _json_digest(
        (case.to_dict(), atomic.case.runtime.to_dict(), asdict(request), stack_id,
         graph.base.sources, graph.streams, primitives),
        default=_digest_default,
    )
    digest = _json_digest((base, framing, mainline.graph_id, primitive_table_digest_token(primitive_table)),
                          default=_digest_default)
    datapath = _die_path(atomic.case.machine, atomic.case.machine.dies[0])
    return ResolvedScenario(
        case, request, stack_id, base, framing, digest, atomic, mainline,
        graph.executable, datapath, graph.streams, die_placements_for_case(atomic.case), (), primitive_table,
    )
def _annotate(op: OpSpec, **values) -> OpSpec:
    if any(key == "occurrence" or not _flat(value) for key, value in values.items()):
        raise ModelingError(f"rewrite_metadata_not_flat:{op.op_id}")
    # The input op can carry source-owned lowering facts such as typed ports.
    return replace(op, metadata=tuple({**dict(op.metadata), **values}.items()))
def _die(scenario, replica, stage):
    parallel = scenario.case.runtime.parallel
    return scenario.case.machine.dies[(replica * parallel.pp + stage) * parallel.tp * parallel.pcp]
def _die_path(machine, die):
    return datapath_for_die(die, ub_peer_edges=ub_peer_edges_for_rank(
        die.rank, machine.links, machine.transfer_channels))
def _path(scenario, replica, stage):
    return _die_path(scenario.case.machine, _die(scenario, replica, stage))
def _solve_tiles(scenario, replica, op, spec, stream=None):
    die = _die(scenario, replica, op.stage)
    path = _die_path(scenario.case.machine, die)
    pool_owners = {**dict.fromkeys(tile_solver.AIC_POOLS, {"aic"}),
                   **dict.fromkeys(tile_solver.AIV_POOLS, {"aiv"}),
                   "hbm": {"die"}, "l2": {"aic", "aiv", "die"}}
    owners = (None if stream is None else bind_stream_owners(stream, path)[
        dict(stream.assignments).get(op.op_id, MAIN_STREAM)])
    hw = SimpleNamespace(datapath=path, buffers=die.buffers, buffer_units=die.buffer_units,
                         stream_owners=owners)
    carveout = max(spec.simt_ub_carveout_bytes, path.simt_ub_carveout_bytes)
    rejected = []
    for candidate in sorted(spec.candidates, key=lambda item: tile_solver.ALGORITHM_LADDER_RANK[item.algorithm]):
        invalid = next((buf.pool for buf in candidate.buffers if buf.owner_kind not in pool_owners.get(buf.pool, ())),
                       None)
        if invalid is not None:
            rejected.append(f"{invalid}:owner")
            continue
        granule = spec.pool_granules.get("hbm", tile_solver.DEFAULT_POOL_GRANULE_BYTES)
        peak = tile_solver._pool_peak_bytes(candidate.buffers, "die", "hbm", granule)
        if peak > die.memory_bytes:
            rejected.append(f"hbm:bytes:{peak}>{die.memory_bytes}")
            continue
        try:
            solved = tile_solver.solve(replace(spec, candidates=(candidate,), simt_ub_carveout_bytes=carveout),
                                       op, hw)
        except tile_solver.TileInfeasibleError as error:
            rejected.append(str(error))
            continue
        hbm = (tile_solver.OwnerPoolPeak("die", "hbm", peak, die.memory_bytes),) if peak else ()
        return replace(solved, pool_peaks=(*solved.pool_peaks, *hbm)), candidate
    raise ModelingError(f"MULTI_POOL_CAPACITY:{op.op_id}:{','.join(rejected) or 'no_candidate'}")
def _typed_inputs(op):
    return tuple((key, port) for key, fields in op.metadata if key in {"typed_inputs", "input_ports"}
                 for port in fields if isinstance(port, Mapping))
def _port_key(port):
    return (port.get("port") or port.get("output") or port.get("name") or "output",
            port.get("version", port.get("state_version")) or 0)
def _positional_name(dep, port):
    name = port.removeprefix(dep + ":")
    return "output" if name == "out" else name
def _named_inputs(op):
    return {key: dep for key, dep in op.metadata if key.endswith("_producer") and isinstance(dep, str)}


_GRAPH_INDEX_CACHE_MAX = 16
_PORT_PAIR_LEN = 2
_MIDPOINT_FACTOR = 0.5
_GRAPH_INDEX_CACHE: OrderedDict[tuple[int, int], tuple[Any, ...]] = OrderedDict()


def _op_candidate_refs(op: OpSpec) -> set[str]:
    """Single-pass extraction of all dependency, port, control, and named producer IDs."""
    names = {str(dep) for dep in op.deps}
    named: dict[str, str] = {}
    for key, value in metadata_pairs(op.metadata):
        if key == INPUT_PORTS_KEY and isinstance(value, (tuple, list)):
            for item in value:
                if isinstance(item, (tuple, list)) and len(item) == _PORT_PAIR_LEN:
                    dep, port = str(item[0]), str(item[1])
                    if dep and port:
                        names.add(dep)
        if key in CONTROL_METADATA_KEYS:
            for name in control_value_names(value):
                if name:
                    names.add(name)
        if key in {"typed_inputs", "input_ports"}:
            for port in value:
                if isinstance(port, Mapping):
                    prod = port.get("producer")
                    if prod is not None:
                        names.add(prod)
        if isinstance(key, str) and key.endswith("_producer") and isinstance(value, str):
            named[key] = value
    if named:
        names.update(named.values())
    return names


def _compute_topo_pos(deps_map: Mapping[str, Any]) -> dict[str, int] | None:
    try:
        return {node: idx for idx, node in enumerate(TopologicalSorter(deps_map).static_order())}
    except CycleError:
        return None


def _build_graph_index(
    ops: Sequence[OpSpec], stream=None,
) -> tuple[dict[str, frozenset[str]], dict[str, tuple[str, ...]], dict[str, int] | None]:
    known = {op.op_id for op in ops}
    deps: dict[str, set[str]] = {
        op.op_id: {name for name in _op_candidate_refs(op) if name in known and name != op.op_id}
        for op in ops
    }
    if stream is not None:
        for event in stream.events:
            for target in event.wait_before:
                if target in deps and event.record_after in deps:
                    deps[target].add(event.record_after)
    users_lists: dict[str, list[str]] = {op.op_id: [] for op in ops}
    for key, predecessors in deps.items():
        for dep in predecessors:
            users_lists[dep].append(key)
    deps_frozen = {key: frozenset(predecessors) for key, predecessors in deps.items()}
    users_tuple = {key: tuple(children) for key, children in users_lists.items()}
    return deps_frozen, users_tuple, _compute_topo_pos(deps_frozen)


def _graph_index(
    ops: Sequence[OpSpec], stream=None,
) -> tuple[dict[str, frozenset[str]], dict[str, tuple[str, ...]], dict[str, int] | None]:
    if not isinstance(ops, tuple):
        return _build_graph_index(ops, stream)
    cache_key = (id(ops), id(stream))
    cached = _GRAPH_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] is ops and cached[1] is stream:
        _GRAPH_INDEX_CACHE.move_to_end(cache_key)
        return cached[2], cached[3], cached[4]
    deps_frozen, users_tuple, topo_pos = _build_graph_index(ops, stream)
    _GRAPH_INDEX_CACHE[cache_key] = (ops, stream, deps_frozen, users_tuple, topo_pos)
    while len(_GRAPH_INDEX_CACHE) > _GRAPH_INDEX_CACHE_MAX:
        _GRAPH_INDEX_CACHE.popitem(last=False)
    return deps_frozen, users_tuple, topo_pos


def _dependencies(ops, stream=None):
    deps_frozen, _, _ = _graph_index(ops, stream)
    return {key: set(predecessors) for key, predecessors in deps_frozen.items()}


def _successors(ops, stream=None):
    _, users_tuple, _ = _graph_index(ops, stream)
    return {key: list(children) for key, children in users_tuple.items()}


def _convex_from_index(
    users: Mapping[str, Any],
    members: set[str],
    topo_pos: Mapping[str, float | int] | None = None,
) -> bool:
    if not members:
        return True
    max_pos = max(topo_pos[m] for m in members) if topo_pos is not None else None
    pending: list[str] = []
    for member in members:
        for user in users[member]:
            if user not in members and (max_pos is None or topo_pos[user] < max_pos):
                pending.append(user)
    visited: set[str] = set()
    while pending:
        node = pending.pop()
        if node in members:
            return False
        if node not in visited:
            visited.add(node)
            for nxt in users[node]:
                if nxt in members:
                    return False
                if max_pos is None or topo_pos[nxt] < max_pos:
                    pending.append(nxt)
    return True


def _convex(ops, members: set[str], stream=None) -> bool:
    _, users, topo_pos = _graph_index(ops, stream)
    return _convex_from_index(users, members, topo_pos)


def _topological(ops, *, dependencies=None):
    positions = {key: i for i, key in enumerate(ops)}
    if dependencies is None:
        dependencies, _, _ = _graph_index(tuple(ops.values()))
    if any(key in op.deps for key, op in ops.items()):
        raise ModelingError("rewrite_execution_dag_cycle")
    if all(positions[dep] < positions[key] for key, deps in dependencies.items() for dep in deps):
        return tuple(ops.values())
    sorter = TopologicalSorter(dependencies)
    try:
        sorter.prepare()
    except CycleError as error:
        raise ModelingError("rewrite_execution_dag_cycle") from error
    pending, ordered = [], []
    while sorter.is_active():
        for key in sorter.get_ready():
            heappush(pending, (positions[key], key))
        _, key = heappop(pending)
        ordered.append(ops[key])
        sorter.done(key)
    return tuple(ordered)
def _check_stream(ops, plan):
    try:
        return lower_stream_edges(ops, plan)
    except ModelingError as error:
        message = str(error)
        for prefix, code in (("stream_cross_dependency_missing_event", "STREAM_EVENT_UNCOVERED"),
                             ("stream_event_cycle", "STREAM_DAG_CYCLE")):
            if message.startswith(prefix):
                raise ModelingError(f"{code}:{message}") from error
        raise
def _ports(op):
    meta = dict(op.metadata)
    result = {(name, 0) for name in ("op", "output", *op.returns)}
    if any(key in meta for key in ("weight_bytes", "local_weight_bytes", "weight_read_bytes")):
        result.add(("weight", 0))
    if isinstance(meta.get("primary_output"), str):
        result.add((meta["primary_output"], 0))
    for key in ("typed_inputs", "typed_outputs", "output_ports", "input_ports", "ports"):
        for port in meta.get(key, ()):
            if isinstance(port, Mapping):
                name = port.get("name") or port.get("output") or port.get("port") or port.get("consumer_field")
                if isinstance(name, str):
                    version = port.get("version", port.get("state_version"))
                    result.add((name, 0 if version is None else int(version)))
    return result
def _tile_fits_resident_pool(
    source: OpSpec, target: OpSpec, pool: str, path: DatapathCapabilitySpec,
) -> bool:
    """Verify per-tile SRAM fit via TileSolver (Ascend FP32 4B/elem UB upcast)."""
    shape = tuple(source.local_shape or target.local_shape or ())
    inner = max(1, int(shape[-1])) if shape else _POINTWISE_TILE_ELEMS
    sfam, tfam = str(source.family or "").lower(), str(target.family or "").lower()
    srec, trec = str(source.recipe or "").lower(), str(target.recipe or "").lower()
    is_row = (
        sfam in _ROW_REDUCTION_FAMILIES or tfam in _ROW_REDUCTION_FAMILIES
        or any(tok in srec or tok in trec for tok in ("norm", "softmax", "quant"))
    )
    smeta = dict(source.metadata)
    tmeta = dict(target.metadata)
    group_sz = int(tmeta.get("quant_group_size") or tmeta.get("group_size") or 0)
    is_block_quant = tfam == "quantize" and group_sz > 0 and group_sz < inner and sfam not in _ROW_REDUCTION_FAMILIES
    if is_block_quant:
        tile_elems = min(inner, max(group_sz, _BLOCK_QUANT_TILE_ELEMS))
    elif is_row:
        tile_elems = inner
    else:
        tile_elems = min(inner, _POINTWISE_TILE_ELEMS)

    src_dtype = str(smeta.get("output_format") or smeta.get("activation_format") or source.dtype or "").strip().lower()
    dst_dtype = str(tmeta.get("output_format") or target.dtype or "").strip().lower()
    if not src_dtype or not dst_dtype:
        raise ModelingError(f"rewrite_tile_missing_dtype:{source.op_id}:{target.op_id}")
    carveout = int(path.simt_ub_carveout_bytes)
    candidates: list[tile_solver.TileCandidate] = []
    if pool == "ub":
        in_b = tile_elems * max(1, int(round(float(dtype_bytes(src_dtype)))))
        out_b = tile_elems * max(1, int(round(float(dtype_bytes(dst_dtype)))))
        fp32_b = tile_elems * _FP32_UPCAST_BYTES
        for depth in (2, 1):
            buffers = (
                tile_solver.LiveBufferSpec("tile_in", "ub", in_b, 0, 1, owner_kind="aiv", slots=depth),
                tile_solver.LiveBufferSpec("tile_fp32_a", "ub", fp32_b, 0, 2, owner_kind="aiv", slots=1),
                tile_solver.LiveBufferSpec("tile_fp32_b", "ub", fp32_b, 1, 2, owner_kind="aiv", slots=1),
                tile_solver.LiveBufferSpec("tile_out", "ub", out_b, 1, 2, owner_kind="aiv", slots=depth),
            )
            candidates.append(tile_solver.TileCandidate(
                "resident_1pass", tile_solver.TileShape(1, tile_elems, 1),
                depth, (1, 1), 1, 1, buffers, uses_simt=True, owner_axis="aiv",
            ))
    elif pool == "l1":
        tw = min(inner, _POINTWISE_TILE_ELEMS)
        buf_b = tw * _L1_TILE_K_ELEMS * max(1, int(round(float(dtype_bytes(src_dtype)))))
        buffers = (
            tile_solver.LiveBufferSpec("tile_a", "l1", buf_b, 0, 1, owner_kind="aic", slots=2),
            tile_solver.LiveBufferSpec("tile_b", "l1", buf_b, 0, 1, owner_kind="aic", slots=2),
        )
        candidates.append(tile_solver.TileCandidate(
            "resident_1pass", tile_solver.TileShape(tw, tw, _L1_TILE_K_ELEMS),
            2, (1, 1), 1, 1, buffers, owner_axis="aic",
        ))
    else:
        buf_b = _L0C_TILE_ELEMS * _L0C_TILE_ELEMS * _FP32_UPCAST_BYTES
        buffers = (
            tile_solver.LiveBufferSpec("tile_acc", "l0c", buf_b, 0, 1, owner_kind="aic", slots=1),
        )
        candidates.append(tile_solver.TileCandidate(
            "resident_1pass", tile_solver.TileShape(_L0C_TILE_ELEMS, _L0C_TILE_ELEMS, _L1_TILE_K_ELEMS),
            1, (1, 1), 1, 1, buffers, owner_axis="aic",
        ))
    hw = SimpleNamespace(datapath=path, buffers=dict(path.buffers), buffer_units={"l2": 1}, stream_owners=None)
    for cand in candidates:
        spec = TilePipelineSpec(str(source.family or "fused"), (cand,), simt_ub_carveout_bytes=carveout)
        try:
            tile_solver.solve(spec, source, hw)
            return True
        except (tile_solver.TileInfeasibleError, ModelingError):
            continue
    return False


def _is_non_data_producer(prod_id: str, all_ops: Mapping[str, OpSpec] | None) -> bool:
    if prod_id.startswith("control:"):
        return True
    if all_ops is not None and prod_id in all_ops:
        fam = str(all_ops[prod_id].family or "").strip().lower()
        return fam in _NON_DATA_PRODUCER_FAMILIES
    return False


def _producer_matches_gemm_activation(producer_op: OpSpec, consumer_op: OpSpec) -> bool:
    if not producer_op.local_shape:
        return False
    if len(consumer_op.local_shape) != _GEMM_SHAPE_RANK:
        return True
    m_dim, _, k_dim = (int(d) for d in consumer_op.local_shape)
    if m_dim <= 0 or k_dim <= 0:
        return False
    if producer_op.flops > 0 and len(producer_op.local_shape) == _GEMM_SHAPE_RANK:
        out_shape = (int(producer_op.local_shape[0]), int(producer_op.local_shape[1]))
    else:
        out_shape = tuple(int(d) for d in producer_op.local_shape)
    if not out_shape or any(d <= 0 for d in out_shape):
        return False
    return out_shape[-1] == k_dim or prod(out_shape) == m_dim * k_dim


def _activation_port_signature(op: OpSpec, all_ops: Mapping[str, OpSpec] | None = None) -> tuple[tuple[str, str, int], ...] | None:
    """Return proven non-barrier activation ``PortRef`` tuple ``((producer, port, version), ...)`` or ``None``."""
    controls = control_dependency_names(op, all_ops)
    typed = _typed_inputs(op)
    if typed:
        ports: list[tuple[str, str, int]] = []
        for _, entry in typed:
            if str(entry.get("role") or "").strip().lower() in _NON_ACTIVATION_ROLES:
                continue
            producer = entry.get("producer")
            if not isinstance(producer, str) or not producer:
                continue
            if producer in controls or _is_non_data_producer(producer, all_ops):
                continue
            name, ver = _port_key(entry)
            norm = "output" if name == "out" else str(name)
            ports.append((producer, norm, int(ver)))
        return tuple(ports) if ports else None
    explicit = input_port_entries(op.metadata)
    if explicit:
        ports = [
            (dep, _positional_name(dep, port), 0)
            for dep, port in explicit
            if dep and _positional_name(dep, port) != "weight" and not _is_non_data_producer(dep, all_ops)
        ]
        return tuple(ports) if ports else None
    named = {k: v for k, v in _named_inputs(op).items() if "weight" not in k.lower()}
    if named:
        ports = [
            (dep, key, 0)
            for key, dep in sorted(named.items())
            if dep and dep not in controls and not _is_non_data_producer(dep, all_ops)
        ]
        return tuple(ports) if ports else None
    data_deps = tuple(
        d for d in op.deps
        if d and d not in controls and not _is_non_data_producer(d, all_ops)
    )
    if len(data_deps) == 1 and all_ops is not None:
        dep = data_deps[0]
        producer_op = all_ops.get(dep)
        if producer_op is None or len(set(producer_op.returns)) > 1:
            return None
        if not _producer_matches_gemm_activation(producer_op, op):
            return None
        return ((dep, "output", 0),)
    return None


def _activation_quant_key(op: OpSpec) -> tuple[str, int | None, str]:
    """Return canonical ``(a_dtype, group_size, scale_format)`` activation contract."""
    meta = dict(op.metadata)
    a_dt = str(
        meta.get("activation_format") or meta.get("input_format") or meta.get("dtype_a") or op.dtype or ""
    ).strip().lower()
    if not a_dt:
        raise ModelingError(f"rewrite_missing_activation_dtype:{op.op_id}")
    raw_gs = meta.get("activation_group_size", meta.get("quant_group_size"))
    group_size = int(raw_gs) if isinstance(raw_gs, int) and not isinstance(raw_gs, bool) and raw_gs > 0 else None
    scale_fmt = str(meta.get("activation_scale_format") or meta.get("quant_scale_dtype") or "").strip().lower()
    return a_dt, group_size, scale_fmt


def _sharding_key(op: OpSpec) -> tuple[str, str]:
    meta = dict(op.metadata) if op.metadata else {}
    return (
        str(meta.get("sharding") or meta.get("shard_rule") or meta.get("parallel_axis") or ""),
        str(meta.get("group_axis") or meta.get("output_collective_group") or ""),
    )


_FLOAT_PRECISION_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})
_QUANT_PRECISION_DTYPES = frozenset({
    "int8", "int4", "w8a8", "w4a8", "fp8", "fp8_e4m3", "fp8_e5m2",
    "mxfp8", "mxfp4", "fp4", "fp4_e2m1",
})


def _weight_dtype_key(op: OpSpec) -> str:
    meta = dict(op.metadata) if op.metadata else {}
    abi = getattr(op, "precision_abi", None)
    return str(
        meta.get("weight_format")
        or meta.get("weight_dtype")
        or (abi.weight_dtype if abi is not None else "")
        or op.dtype
        or ""
    ).strip().lower()


def _precision_abi_tuple(op: OpSpec) -> tuple[Any, ...]:
    """Full TensorPrecisionABI equality tuple for homogeneous sibling GEMM packing."""
    meta = dict(op.metadata) if op.metadata else {}
    abi = getattr(op, "precision_abi", None)
    a_dt, group_size, scale_fmt = _activation_quant_key(op)
    w_dt = _weight_dtype_key(op)
    out_dt = str(meta.get("output_format") or (abi.output_dtype if abi else "") or op.dtype or "").strip().lower()
    comp_dt = str(meta.get("compute_format") or (abi.compute_dtype if abi else "") or op.dtype or "").strip().lower()
    if not out_dt or not comp_dt:
        raise ModelingError(f"rewrite_missing_precision_dtype:{op.op_id}")
    method = str(meta.get("linear_method") or (abi.linear_method if abi else "") or "").strip().lower()
    w_layout = str(meta.get("weight_layout") or meta.get("weight_representation") or (abi.weight_layout if abi else "") or "").strip().lower()
    w_block = tuple(meta.get("weight_block") or ()) if isinstance(meta.get("weight_block"), (tuple, list)) else ()
    w_scale_fmt = str(meta.get("weight_scale_format") or (abi.weight_scale_dtype if abi else "") or "").strip().lower()
    act_scale_pack = str(meta.get("activation_scale_pack") or (abi.act_scale_pack if abi else "") or "").strip().lower()
    abi_sig = str(meta.get("precision_abi_signature") or (abi.abi_signature if abi else "") or "")
    return (
        method,
        w_dt,
        w_layout,
        w_block,
        w_scale_fmt,
        a_dt,
        group_size,
        scale_fmt,
        act_scale_pack,
        out_dt,
        comp_dt,
        abi_sig,
    )


def _unabsorbed_splitv_cost(
    siblings: Sequence[OpSpec],
    all_ops: Mapping[str, OpSpec] | None,
) -> tuple[float, int]:
    """Price physical unabsorbed SplitV / DataCopyPad HBM R+W and launch count.

    Single-row outputs (``M <= 1``) slice contiguously with zero copy.
    Multi-row outputs (``M > 1``) require physical contiguous slice
    materialization (``SplitV`` / ``DataCopyPad``) unless EVERY downstream
    consumer explicitly declares ``accepts_strided_view=True`` or
    ``packed_view_compatible=True``. Unknown output dtypes fail fast.
    """
    if not all_ops or len(siblings) < 2:
        return 0.0, 0
    sib_ids = {op.op_id for op in siblings}
    m_dim = int(siblings[0].local_shape[0]) if len(siblings[0].local_shape) >= 2 else 0
    if m_dim <= 1:
        return 0.0, 0
    consumers_by_sib: dict[str, set[str]] = {op.op_id: set() for op in siblings}
    for cand in all_ops.values():
        if cand.op_id in sib_ids:
            continue
        matched = [dep for dep in cand.deps if dep in consumers_by_sib]
        if not matched:
            continue
        meta = dict(cand.metadata) if cand.metadata else {}
        if bool(meta.get("accepts_strided_view")) or bool(meta.get("packed_view_compatible")):
            continue
        for dep in matched:
            consumers_by_sib[dep].add(cand.op_id)
    total_slice_bytes = 0.0
    unabsorbed_slices = 0
    for op in siblings:
        if not consumers_by_sib[op.op_id]:
            continue
        n_dim = int(op.local_shape[1]) if len(op.local_shape) >= 2 else 0
        meta = dict(op.metadata) if op.metadata else {}
        out_dt = str(meta.get("output_format") or op.dtype or "").strip()
        if not out_dt:
            raise ModelingError(f"rewrite_splitv_dtype_missing:{op.op_id}")
        elem = float(dtype_bytes(out_dt))
        total_slice_bytes += float(m_dim * n_dim) * elem
        unabsorbed_slices += 1
    if total_slice_bytes <= 0.0 or unabsorbed_slices == 0:
        return 0.0, 0
    # One physical SplitV launch reads the packed output and writes each contiguous slice.
    return 2.0 * total_slice_bytes, 1


def has_illegal_cross_precision_merge(ops: Sequence[OpSpec]) -> bool:
    """Return True if matrix ops mix FLOAT and quantized precisions without explicit cast."""
    matrix_ops = [op for op in ops if float(op.flops or 0.0) > 0.0 or bool(op.matrix_flops)]
    if len(matrix_ops) < 2:
        return False
    classes: set[str] = set()
    abis = []
    for op in matrix_ops:
        meta = dict(op.metadata) if op.metadata else {}
        abi = getattr(op, "precision_abi", None)
        if abi is not None:
            has_cast = bool(abi.explicit_cast_on_entry or abi.explicit_cast_on_exit)
            dts = {
                str(abi.input_dtype).strip().lower(),
                str(abi.weight_dtype).strip().lower(),
                str(abi.compute_dtype).strip().lower(),
            }
            abis.append(abi)
        else:
            has_cast = bool(meta.get("explicit_cast_on_entry") or meta.get("explicit_cast_on_exit"))
            dts = {
                str(meta.get(k) or "").strip().lower()
                for k in ("activation_format", "weight_format", "compute_format")
                if meta.get(k)
            }
            if not dts:
                raw_dt = str(op.dtype or "").strip().lower()
                if not raw_dt:
                    raise ModelingError(f"rewrite_missing_precision_dtype:{op.op_id}")
                dts.add(raw_dt)
        if not has_cast:
            if dts & _QUANT_PRECISION_DTYPES:
                classes.add("quant")
            elif dts & _FLOAT_PRECISION_DTYPES:
                classes.add("float")
    if "float" in classes and "quant" in classes:
        return True
    if len(abis) >= 2:
        base = abis[0]
        if any(not base.is_fusion_compatible(other) for other in abis[1:]):
            return True
    return False


def _has_cross_precision_sibling_gemm(
    ordered: Sequence[OpSpec],
    kernel_for: Mapping[str, int],
    dependencies: Mapping[str, set[str]],
    keys: set[str],
) -> bool:
    """Detect forbidden FLOAT-vs-quantized or incompatible TensorPrecisionABI sibling GEMM merge."""
    if len(ordered) < 2 or len(set(kernel_for.values())) != 1:
        return False
    if not all(op.flops > 0 and not op.link_bytes and len(op.local_shape) == _GEMM_SHAPE_RANK for op in ordered):
        return False
    if any(dependencies[op.op_id] & keys for op in ordered):
        return False
    # Check explicit TensorPrecisionABI incompatibility first.
    abis = [getattr(op, "precision_abi", None) for op in ordered]
    if all(abi is not None for abi in abis):
        if any(not abis[0].is_fusion_compatible(other) for other in abis[1:]):
            return True
    # Check FLOAT vs quantized weight or activation boundary.
    w_classes = set()
    a_classes = set()
    for op in ordered:
        w_dt = _weight_dtype_key(op)
        a_dt, _, _ = _activation_quant_key(op)
        if w_dt in _FLOAT_PRECISION_DTYPES:
            w_classes.add("float")
        elif w_dt in _QUANT_PRECISION_DTYPES:
            w_classes.add("quant")
        if a_dt in _FLOAT_PRECISION_DTYPES:
            a_classes.add("float")
        elif a_dt in _QUANT_PRECISION_DTYPES:
            a_classes.add("quant")
    return len(w_classes) > 1 or len(a_classes) > 1


def _law2_sibling_key(
    op: OpSpec, all_ops: Mapping[str, OpSpec] | None = None,
) -> tuple[Any, ...] | None:
    """Single canonical Law-2 shared-activation predicate over PortRef, (M, K), full ABI, and sharding."""
    fam = str(op.family or "").strip().lower()
    rec = str(op.recipe or "").strip().lower()
    if fam in _GROUPED_MATMUL_FAMILIES or "grouped" in rec:
        return None
    if not (op.flops > 0 and not op.link_bytes and len(op.local_shape) == _GEMM_SHAPE_RANK):
        return None
    act_ports = _activation_port_signature(op, all_ops=all_ops)
    if not act_ports:
        return None
    a_dt, group_size, scale_fmt = _activation_quant_key(op)
    model_scope = str(dict(op.metadata).get("model_scope") or "target").strip().lower()
    return (
        act_ports,
        int(op.local_shape[0]),
        int(op.local_shape[2]),
        a_dt,
        group_size,
        scale_fmt,
        _weight_dtype_key(op),
        _sharding_key(op),
        _precision_abi_tuple(op),
        model_scope,
    )


def _sibling_activation_dedup_bytes(members, kernel_for, all_ops=None) -> float:
    """Compute duplicate input activation HBM read bytes eliminated by Law-2 sibling GEMM fusion."""
    member_ids = {op.op_id for op in members}
    member_deps = _dependencies(tuple(members))
    groups: dict[tuple, list[OpSpec]] = {}
    for op in members:
        if member_deps.get(op.op_id, set()) & member_ids:
            continue
        sib_key = _law2_sibling_key(op, all_ops=all_ops)
        if sib_key is None or any(prod_id in member_ids for prod_id, _, _ in sib_key[0]):
            continue
        groups.setdefault((kernel_for[op.op_id], sib_key), []).append(op)
    saved = 0.0
    for (_k_idx, sib_key), siblings in groups.items():
        m_dim, k_dim, a_dt = sib_key[1], sib_key[2], sib_key[3]
        if len(siblings) < 2 or m_dim <= 0 or k_dim <= 0:
            continue
        elem_b = float(dtype_bytes(a_dt))
        act_bytes = float(m_dim * k_dim) * elem_b
        for sib in siblings[1:]:
            smeta = dict(sib.metadata)
            raw_wt = smeta.get("packed_weight_bytes", smeta.get("weight_bytes"))
            wt_b = (
                float(raw_wt)
                if isinstance(raw_wt, (int, float)) and not isinstance(raw_wt, bool) and raw_wt > 0
                else 0.0
            )
            avail = min(float(sib.mte2_bytes), max(0.0, float(sib.hbm_bytes) - float(sib.mte3_bytes)))
            if 0.0 < wt_b <= float(sib.mte2_bytes):
                avail = min(avail, float(sib.mte2_bytes) - wt_b)
            saved += min(act_bytes, max(0.0, avail))
    return saved


def _law2_matrix_metadata(
    ordered: Sequence[OpSpec],
    fused_shape: tuple[int, ...],
    fused_mte2: float,
) -> list[tuple[str, Any]]:
    """Preserve matrix precision/weight/epilogue metadata across Law-2 sibling GEMM fusion."""
    member_metas = [dict(op.metadata) for op in ordered]
    out: list[tuple[str, Any]] = []
    for mkey in (
        "weight_format", "weight_dtype", "weight_layout", "weight_representation",
        "activation_format", "input_format", "output_format", "compute_format",
        "execution_compute_format", "activation_group_size",
        "activation_scale_format", "weight_scale_format",
        "dequant", "dequant_before_compute", "matrix_groups",
    ):
        vals = [m.get(mkey) for m in member_metas if m.get(mkey) is not None]
        if len(vals) == len(member_metas) and len(set(vals)) == 1:
            out.append((mkey, vals[0]))
    packed_vals = [m.get("packed_weight_bytes") for m in member_metas if isinstance(m.get("packed_weight_bytes"), (int, float))]
    weight_vals = [m.get("weight_bytes") for m in member_metas if isinstance(m.get("weight_bytes"), (int, float))]
    if len(packed_vals) == len(member_metas):
        packed_b = float(sum(packed_vals))
    elif len(weight_vals) == len(member_metas):
        packed_b = float(sum(weight_vals))
    elif len(fused_shape) == _GEMM_SHAPE_RANK:
        m_dim, n_dim, k_dim = int(fused_shape[0]), int(fused_shape[1]), int(fused_shape[2])
        a_dt, _, _ = _activation_quant_key(ordered[0])
        w_dt = str(member_metas[0].get("weight_format") or member_metas[0].get("weight_dtype") or ordered[0].dtype or "").strip().lower()
        if not w_dt:
            raise ModelingError(f"rewrite_missing_weight_dtype:{ordered[0].op_id}")
        act_b = float(m_dim * k_dim) * float(dtype_bytes(a_dt))
        dense_w = float(n_dim * k_dim) * float(dtype_bytes(w_dt))
        packed_b = min(dense_w, max(0.0, float(fused_mte2) - act_b))
    else:
        packed_b = 0.0
    if packed_b > 0.0:
        out.append(("packed_weight_bytes", packed_b))
        if len(weight_vals) == len(member_metas):
            out.append(("weight_bytes", float(sum(weight_vals))))
    for skey in ("column_scale_bytes",):
        svals = [m.get(skey) for m in member_metas if isinstance(m.get(skey), (int, float))]
        if len(svals) == len(member_metas):
            out.append((skey, float(sum(svals))))
    rscales = [m.get("row_scale_bytes") for m in member_metas if isinstance(m.get("row_scale_bytes"), (int, float))]
    if len(rscales) == len(member_metas):
        out.append(("row_scale_bytes", float(max(rscales))))
    epilogues = [m.get("matrix_epilogue") for m in member_metas if isinstance(m.get("matrix_epilogue"), Mapping)]
    if len(epilogues) == len(member_metas) and len({e.get("kind") for e in epilogues}) == 1:
        out.append(("matrix_epilogue", {
            "kind": epilogues[0].get("kind"),
            "row_scale_bytes": float(max(float(e.get("row_scale_bytes", 0.0) or 0.0) for e in epilogues)),
            "column_scale_bytes": float(sum(float(e.get("column_scale_bytes", 0.0) or 0.0) for e in epilogues)),
            "operations_per_output": float(epilogues[0].get("operations_per_output", 0.0) or 0.0),
        }))
    return out


def _resident_savings(members, kernel_for, preserved, external_reads, path, all_ops=None):
    """Elide on-chip intermediate HBM round-trips verified by datapath + TileSolver."""
    saved_read = saved_write = 0.0
    for source in members:
        if source.op_id in preserved or source.op_id in external_reads:
            continue
        sm = dict(source.metadata)
        raw_writes = sm.get("write_bytes")
        explicit_w = type(raw_writes) in {int, float}
        writes = float(raw_writes) if explicit_w else min(float(source.hbm_bytes), float(source.mte3_bytes))
        if not 0.0 < writes <= min(float(source.hbm_bytes), float(source.mte3_bytes)):
            continue
        consumers = [
            op for op in members
            if source.op_id in op.deps or any(
                isinstance(port, Mapping) and port.get("producer") == source.op_id
                for port in dict(op.metadata).get("typed_inputs", ())
            )
        ]
        if len(consumers) != 1:
            continue
        target = consumers[0]
        if kernel_for[source.op_id] != kernel_for[target.op_id]:
            continue
        tm = dict(target.metadata)
        s_owner, t_owner = sm.get("owner_id"), tm.get("owner_id")
        owners_conflict = s_owner is not None and t_owner is not None and s_owner != t_owner
        if source.flops and not target.flops:
            route = bool(path.has_l0c_to_ub) and _tile_fits_resident_pool(source, target, "ub", path)
        elif not source.flops and target.flops:
            route = bool(path.has_ub_to_l1) and _tile_fits_resident_pool(source, target, "l1", path)
        elif not source.flops and not target.flops:
            route = (not owners_conflict) and _tile_fits_resident_pool(source, target, "ub", path)
        else:
            route = (s_owner is not None and s_owner == t_owner
                     and _tile_fits_resident_pool(source, target, "l0c", path))
        raw_reads = tm.get("resident_input_bytes")
        if raw_reads is None:
            matching = [p.get("read_bytes") for p in tm.get("typed_inputs", ())
                        if isinstance(p, Mapping) and p.get("producer") == source.op_id]
            raw_reads = matching[0] if len(matching) == 1 else None
        explicit_r = type(raw_reads) in {int, float}
        if explicit_w and explicit_r:
            reads = float(raw_reads)
            if reads != writes:
                continue
            handoff = writes
        elif explicit_r:
            handoff = min(writes, float(raw_reads))
        else:
            handoff = min(writes, float(target.hbm_bytes), float(target.mte2_bytes))
        if route and 0.0 < handoff <= min(float(target.hbm_bytes), float(target.mte2_bytes)):
            saved_read += handoff
            saved_write += handoff
    saved_read += _sibling_activation_dedup_bytes(members, kernel_for, all_ops=all_ops)
    max_m2 = sum(float(op.mte2_bytes) for op in members)
    max_m3 = sum(float(op.mte3_bytes) for op in members)
    max_hbm = sum(float(op.hbm_bytes) for op in members)
    saved_read = min(saved_read, max_m2)
    saved_write = min(saved_write, max_m3)
    if saved_read + saved_write > max_hbm:
        excess = saved_read + saved_write - max_hbm
        trim_r = min(saved_read, excess)
        saved_read -= trim_r
        saved_write = max(0.0, saved_write - (excess - trim_r))
    return saved_read, saved_write

class _RankDraft:
    def __init__(self, entry, stream):
        self.replica = entry.replica
        self.ops = {op.op_id: op for op in entry.ops}
        if len(self.ops) != len(entry.ops):
            raise ModelingError(f"rewrite_duplicate_atom:{entry.replica}")
        self.handles: dict[str, tuple[str, ...]] = {}
        self.stream = stream
        self.absorbed: dict[str, tuple[str, ...]] = {}
        self.tiers: dict[str, FusionTier] = {}
        self.leases: list[PrefetchLease] = []
        self.templates: list[TemplateBinding] = []
        self.tiles: list[TileBinding] = []
        self.collectives: list[CollectiveRewriteRecord] = []
        self.hoisted: list[str] = []
        self.elided: list[str] = []
        self._cand_refs: dict[str, set[str]] = {}
        self._base_deps: dict[str, set[str]] = {}
        self._base_users: dict[str, set[str]] = {k: set() for k in self.ops}
        self._consumers: dict[str, set[str]] = {}
        for oid, op in self.ops.items():
            refs = _op_candidate_refs(op)
            self._cand_refs[oid] = refs
            deps = {r for r in refs if r in self.ops and r != oid}
            self._base_deps[oid] = deps
            for dep in deps:
                self._base_users[dep].add(oid)
            for ref in refs:
                self._consumers.setdefault(ref, set()).add(oid)
        self._topo_pos: dict[str, float] | None = None

    def _reindex_op_inputs(self, oid: str) -> None:
        for ref in self._cand_refs.get(oid, ()):
            cons = self._consumers.get(ref)
            if cons is not None:
                cons.discard(oid)
                if not cons:
                    del self._consumers[ref]
        for dep in self._base_deps.get(oid, ()):
            users = self._base_users.get(dep)
            if users is not None:
                users.discard(oid)
        op = self.ops[oid]
        refs = _op_candidate_refs(op)
        self._cand_refs[oid] = refs
        deps = {r for r in refs if r in self.ops and r != oid}
        self._base_deps[oid] = deps
        self._base_users.setdefault(oid, set())
        for dep in deps:
            self._base_users.setdefault(dep, set()).add(oid)
        for ref in refs:
            self._consumers.setdefault(ref, set()).add(oid)

    def _remove_op_index(self, oid: str) -> None:
        for ref in self._cand_refs.pop(oid, ()):
            cons = self._consumers.get(ref)
            if cons is not None:
                cons.discard(oid)
                if not cons:
                    del self._consumers[ref]
        for dep in self._base_deps.pop(oid, ()):
            users = self._base_users.get(dep)
            if users is not None:
                users.discard(oid)
        for user in self._base_users.pop(oid, ()):
            user_deps = self._base_deps.get(user)
            if user_deps is not None:
                user_deps.discard(oid)

    def _consumer_ids(self, keys: set[str]) -> set[str]:
        found: set[str] = set()
        for key in keys:
            cons = self._consumers.get(key)
            if cons:
                found.update(cons)
        found.difference_update(keys)
        return found

    def _effective_deps_for(self, keys: set[str]) -> dict[str, set[str]]:
        deps = {k: set(self._base_deps[k]) for k in keys}
        if self.stream is not None:
            for event in self.stream.events:
                if event.record_after in self.ops:
                    for target in event.wait_before:
                        if target in deps:
                            deps[target].add(event.record_after)
        return deps

    def _is_convex(self, keys: set[str]) -> bool:
        if self.stream is None or not self.stream.events:
            if self._topo_pos is None:
                raw = _compute_topo_pos(self._base_deps)
                self._topo_pos = None if raw is None else {k: float(v) for k, v in raw.items()}
            return _convex_from_index(self._base_users, keys, self._topo_pos)
        deps = {k: set(v) for k, v in self._base_deps.items()}
        users = {k: set(v) for k, v in self._base_users.items()}
        for event in self.stream.events:
            if event.record_after in deps:
                for target in event.wait_before:
                    if target in deps:
                        deps[target].add(event.record_after)
                        users[event.record_after].add(target)
        return _convex_from_index(users, keys, _compute_topo_pos(deps))

    def _update_topo_after_replace(self, removed: set[str], replacement: str | None) -> None:
        if self._topo_pos is None:
            return
        if self.stream is not None and self.stream.events:
            self._topo_pos = None
            return
        for key in removed:
            self._topo_pos.pop(key, None)
        if replacement is None or replacement not in self.ops:
            return
        preds = self._base_deps.get(replacement, ())
        succs = self._base_users.get(replacement, ())
        if replacement in self._topo_pos:
            pos = self._topo_pos[replacement]
            if any(self._topo_pos.get(p, inf) >= pos for p in preds) or any(
                self._topo_pos.get(s, -inf) <= pos for s in succs
            ):
                self._topo_pos = None
            return
        p_max = max((self._topo_pos[p] for p in preds), default=None)
        s_min = min((self._topo_pos[s] for s in succs), default=None)
        if p_max is None and s_min is None:
            self._topo_pos[replacement] = 0.0
        elif p_max is None:
            self._topo_pos[replacement] = s_min - 1.0
        elif s_min is None:
            self._topo_pos[replacement] = p_max + 1.0
        elif p_max < s_min:
            mid = p_max + (s_min - p_max) * _MIDPOINT_FACTOR
            if p_max < mid < s_min:
                self._topo_pos[replacement] = mid
            else:
                self._topo_pos = None
        else:
            self._topo_pos = None

    def resolve(self, ref: OpRef) -> OpSpec:
        ids = (ref.id,) if ref.kind == "atom" else self.handles.get(ref.id, ())
        if len(ids) != 1 or ids[0] not in self.ops:
            raise ModelingError(f"rewrite_unresolved_op:{ref.kind}:{ref.id}")
        op = self.ops[ids[0]]
        if ref.output != "op" and not any(name == ref.output for name, _ in _ports(op)):
            raise ModelingError(f"rewrite_unknown_output:{op.op_id}:{ref.output}")
        return op

    def port(self, ref: PortRef) -> OpSpec:
        op = self.resolve(ref.op)
        if (ref.name, ref.version) not in _ports(op):
            raise ModelingError(f"rewrite_unknown_port:{op.op_id}:{ref.name}:{ref.version}")
        return op

    def replace_nodes(self, removed, replacement, outputs=None):
        removed = removed - {replacement}
        redirects = dict.fromkeys(removed, replacement)
        affected = [key for key in self._consumer_ids(removed) if key in self.ops]
        for key in affected:
            op = self.ops[key]
            ports = input_port_entries(op.metadata)
            typed = {port.get("producer") for _, port in _typed_inputs(op)}
            controls = declared_control_order(op.metadata)
            named = {k: v for k, v in _named_inputs(op).items() if v in self.ops or v in removed}
            inputs = (*op.deps, *controls, *(dep for dep, _ in ports), *sorted(typed - {None}), *named.values())
            if removed.intersection(inputs):
                metadata = rewritten_control_metadata(op.metadata, redirects) if replacement else tuple(
                    (k, {dep: item for dep, item in v.items() if dep not in removed}
                     if isinstance(v, Mapping) else tuple(dep for dep in control_value_names(v) if dep not in removed))
                    if k in CONTROL_METADATA_KEYS else (k, v) for k, v in op.metadata
                )
                def endpoint(dep):
                    return redirects.get(dep, dep)
                metadata = tuple((k, endpoint(v)) if k in named else (k, v)
                                 for k, v in metadata if k not in named or endpoint(v))
                def position(dep, port):
                    name = (outputs or {}).get((dep, _positional_name(dep, port), 0),
                                               port.removeprefix(dep + ":"))
                    if name == "output" and port == f"{dep}:out":
                        name = "out"
                    return endpoint(dep), f"{replacement}:{name}" if dep in removed else port

                def typed_port(port):
                    producer = port["producer"]
                    name, version = _port_key(port)
                    renamed = (outputs or {}).get((producer, name, version), name)
                    return {**port, "producer": endpoint(producer), **{
                        k: renamed for k in ("port", "output", "name") if port.get(k) == name}}

                if outputs is not None:
                    ports = full_operand_ports(op, self.ops)
                metadata = replace_input_ports(metadata, tuple(
                    position(dep, port) for dep, port in ports if endpoint(dep)))
                metadata = tuple((k, tuple(
                    typed_port(port)
                    if isinstance(port, Mapping) and port.get("producer") in removed else port
                    for port in v if not isinstance(port, Mapping)
                    or port.get("producer") not in removed or replacement is not None
                ) if k in {"typed_inputs", "input_ports"} else v) for k, v in metadata)
                deps = tuple(dict.fromkeys(dep for src in inputs
                    if (dep := endpoint(src)) and dep != key))
                self.ops[key] = replace(op, deps=deps, metadata=metadata)
        for key in removed:
            del self.ops[key]
            self._remove_op_index(key)
        if replacement is not None and replacement in self.ops:
            self._reindex_op_inputs(replacement)
        for key in affected:
            self._reindex_op_inputs(key)
        if self.stream is not None:
            assign = dict(self.stream.assignments)
            if replacement is not None:
                lanes = {assign.get(key, MAIN_STREAM) for key in removed}
                if replacement in assign:
                    lanes.add(assign[replacement])
                if len(lanes) > 1:
                    raise ModelingError("rewrite_cross_stream_contraction")
                if lanes:
                    assign[replacement] = lanes.pop()
            def endpoint(key):
                return replacement if key in removed else key

            events = []
            for event in self.stream.events:
                record = endpoint(event.record_after)
                waits = tuple(dict.fromkeys(
                    key for item in event.wait_before if (key := endpoint(item)) and key != record
                ))
                if record and waits:
                    events.append(replace(event, record_after=record, wait_before=waits))
            self.stream = replace(self.stream, assignments=tuple(
                (key, value) for key, value in assign.items() if key not in removed
            ), events=tuple(events))
        self._update_topo_after_replace(removed, replacement)

    def _assemble_fused_op(
        self, *, fused_id, spec, ordered, keys, kernel_for, dependencies,
        boundary, typed, named, cut, live, declared, read, write, consumer_ops=None,
    ):
        tail = ordered[-1]
        matrix = Counter()
        for op in ordered:
            for dtype, amount in (op.matrix_flops or (((op.dtype, op.flops),) if op.flops else ())):
                matrix[dtype] += amount
        link_scope = next((item.link_scope for item in ordered if item.link_bytes), tail.link_scope)
        boundary_abis = tuple(b.abi_id for b in spec.kernel_boundaries)
        abi = boundary_abis[0] if len(boundary_abis) == 1 else "+".join(boundary_abis)
        reserved = {name for _, name, _ in live} | set(tail.returns) | {"op", "out"}
        outputs, used = {}, set()
        for dep, name, version in sorted(live, key=lambda item: (item[0] != tail.op_id, item)):
            port = name
            while (port, version) in used or (port != name and port in reserved):
                port = f"{port}__{dep}"
            outputs[dep, name, version] = port
            used.add((port, version))
        source_ports = {(op.op_id, *_port_key(port)): port for op in ordered
                        for key, ports in op.metadata if key == "typed_outputs" for port in ports
                        if isinstance(port, Mapping)}
        typed_keys = (declared | source_ports.keys()) & live | {key for key in live if key[-1]}
        typed_outputs = tuple({**source_ports.get(key, {}), "producer": fused_id,
                               "port": outputs[key], "output": outputs[key], "name": outputs[key], "version": key[-1]}
                              for key in sorted(typed_keys))
        fused_shape = tail.local_shape
        law2_keys = [_law2_sibling_key(op, all_ops=self.ops) for op in ordered]
        is_law2_sibling_gemm = (
            len(ordered) >= 2
            and len(set(kernel_for.values())) == 1
            and all(not (dependencies[op.op_id] & keys) for op in ordered)
            and all(k is not None for k in law2_keys)
            and len(set(law2_keys)) == 1
            and all(not any(prod_id in keys for prod_id, _, _ in k[0]) for k in law2_keys if k is not None)
        )
        splitv_hbm = 0.0
        splitv_launches = 0
        if is_law2_sibling_gemm:
            fused_shape = (
                int(ordered[0].local_shape[0]),
                sum(int(op.local_shape[1]) for op in ordered),
                int(ordered[0].local_shape[2]),
            )
            splitv_hbm, splitv_launches = _unabsorbed_splitv_cost(
                ordered, self.ops if consumer_ops is None else consumer_ops,
            )
        splitv_half = 0.5 * splitv_hbm
        fused_mte2 = sum(op.mte2_bytes for op in ordered) - read + splitv_half
        fused_mte3 = sum(op.mte3_bytes for op in ordered) - write + splitv_half
        fused_hbm = sum(op.hbm_bytes for op in ordered) - read - write + splitv_hbm
        extra_meta: list[tuple[str, Any]] = [
            ("fusion_tier", spec.fusion_tier),
            ("kernel_abi", abi),
            ("kernel_boundary_count", len(spec.kernel_boundaries) + splitv_launches),
            ("gm_round_trip_eliminated_bytes", read + write),
            ("fusion_route", "resident" if read or write else "gm_spill"),
        ]
        if spec.backend_target is not None:
            extra_meta.append(("backend_target", spec.backend_target))
        if spec.abi_query_digest:
            extra_meta.append(("abi_query_digest", spec.abi_query_digest))
        if spec.lowering_id:
            extra_meta.append(("lowering_id", spec.lowering_id))
        if spec.lowering_sha256:
            extra_meta.append(("lowering_sha256", spec.lowering_sha256))
        if spec.ub_peak_bytes > 0:
            extra_meta.append(("ub_peak_bytes", spec.ub_peak_bytes))
        if spec.ub_wall_bytes > 0:
            extra_meta.append(("ub_wall_bytes", spec.ub_wall_bytes))
        if spec.abi_verdict is not None:
            extra_meta.append(("abi_verdict_status", spec.abi_verdict.status))
            extra_meta.append(("abi_verdict_entry_id", spec.abi_verdict.matched_entry_id))
            extra_meta.append(("abi_verdict_manifest_sha256", spec.abi_verdict.probe_manifest_sha256))
        if splitv_hbm > 0.0:
            extra_meta.append(("splitv_unabsorbed_hbm_bytes", splitv_hbm))
            extra_meta.append(("splitv_launch_count", splitv_launches))
        if is_law2_sibling_gemm:
            extra_meta.append(("law2_block_gemm", True))
            extra_meta.extend(_law2_matrix_metadata(ordered, fused_shape, fused_mte2))
        for op in ordered:
            axis = dict(op.metadata).get("group_axis")
            if isinstance(axis, str) and axis:
                extra_meta.append(("group_axis", axis))
                break
        fused = replace(
            tail, op_id=fused_id, family="rewrite_fused", recipe=abi, source_boundary="",
            semantic_shape=fused_shape if is_law2_sibling_gemm else tail.semantic_shape,
            local_shape=fused_shape,
            padded_shape=fused_shape if is_law2_sibling_gemm else tail.padded_shape,
            launch_shape=fused_shape if is_law2_sibling_gemm else tail.launch_shape,
            returns=tuple(dict.fromkeys((*tail.returns, *(ref.name for ref in spec.preserve_ports),
                                        *outputs.values()))),
            flops=sum(op.flops for op in ordered), matrix_flops=tuple(sorted(matrix.items())),
            vector_ops=sum(op.vector_ops for op in ordered),
            link_bytes=sum(op.link_bytes for op in ordered), link_scope=link_scope,
            hbm_bytes=fused_hbm, mte2_bytes=fused_mte2, mte3_bytes=fused_mte3,
            metadata=tuple(extra_meta),
        )
        fused = apply_incoming_controls(fused, ordered, keys, boundary, self.ops)
        explicit = {port for op in ordered for port in input_port_entries(op.metadata)}
        metadata = replace_input_ports(fused.metadata, tuple(port for port in boundary if port in explicit))
        metadata += tuple(named.items())
        metadata += tuple((key, tuple(port for name, port in typed if name == key))
                          for key in dict.fromkeys(name for name, _ in typed))
        if typed_outputs:
            metadata += (("typed_outputs", typed_outputs),)
        fused = replace(fused, deps=tuple(dict.fromkeys((*fused.deps, *sorted(cut)))), metadata=metadata)
        return fused, outputs

    def fuse(self, edit, scenario):
        spec = edit.operation
        members = tuple(self.resolve(ref) for ref in spec.members)
        keys = {op.op_id for op in members}
        if not members or len(members) != len(keys) or not spec.kernel_boundaries:
            raise ModelingError("rewrite_fusion_members_invalid")
        kernel_for = {}
        for index, boundary in enumerate(spec.kernel_boundaries):
            if not boundary.abi_id or not boundary.members:
                raise ModelingError("rewrite_empty_kernel_boundary")
            for ref in boundary.members:
                op_id = self.resolve(ref).op_id
                if op_id in kernel_for:
                    raise ModelingError("rewrite_duplicate_kernel_member")
                kernel_for[op_id] = index
        if keys != kernel_for.keys():
            raise ModelingError("rewrite_kernel_boundary_membership")
        assignments = dict(self.stream.assignments) if self.stream is not None else {}
        if len({(op.stage, op.phase, assignments.get(op.op_id, op.stream or MAIN_STREAM))
                for op in members}) != 1:
            raise ModelingError("rewrite_fusion_cross_execution_domain")
        if not self._is_convex(keys):
            raise ModelingError(f"NON_CONVEX_REGION:{edit.edit_id}")
        ordered = tuple(op for op in _topological(self.ops, dependencies=self._base_deps) if op.op_id in keys)
        dependencies = self._effective_deps_for(keys)
        if any(kernel_for[dep] > kernel_for[op.op_id]
               for op in ordered for dep in dependencies[op.op_id] if dep in keys):
            raise ModelingError("rewrite_kernel_boundary_dependency_order")
        boundary = tuple(dict.fromkeys(port for op in ordered
                         for port in (*full_operand_ports(op, self.ops), *input_port_entries(op.metadata))
                         if port[0] not in keys))
        typed = tuple((key, port) for op in ordered for key, port in _typed_inputs(op)
                      if port.get("producer") not in keys)
        named = {key: dep for op in ordered for key, dep in _named_inputs(op).items() if dep not in keys}
        cut = {dep for op in ordered for dep in (*op.deps, *dependencies[op.op_id]) if dep not in keys}
        users = tuple(self.ops[cid] for cid in self._consumer_ids(keys) if cid in self.ops)
        declared = {(port["producer"], *_port_key(port)) for op in users
                    for _, port in _typed_inputs(op) if port.get("producer") in keys}
        consumed = {(dep, _positional_name(dep, port), 0) for op in users
                    for dep, port in full_operand_ports(op, self.ops) if dep in keys}
        retained = {(self.port(ref).op_id, ref.name, ref.version) for ref in spec.preserve_ports}
        readers = {dep for dep, _, _ in consumed | declared} | {
            dep for op in users for dep in _named_inputs(op).values() if dep in keys}
        preserved = {dep for dep, _, _ in retained}
        if preserved - (keys | cut):
            raise ModelingError("rewrite_fusion_port_outside_cut")
        if (readers - {ordered[-1].op_id}) - preserved:
            raise ModelingError("rewrite_fusion_live_out_unpreserved")
        if any(op.link_bytes for op in ordered) and len({op.link_scope for op in ordered if op.link_bytes}) > 1:
            raise ModelingError("rewrite_fusion_mixed_link_scopes")
        if _has_cross_precision_sibling_gemm(ordered, kernel_for, dependencies, keys):
            raise ModelingError("rewrite_fusion_precision_abi_mismatch")
        fused_id = f"rewrite:{edit.edit_id}"
        if fused_id in self.ops:
            raise ModelingError(f"rewrite_duplicate_fused_id:{fused_id}")
        path = _path(scenario, self.replica, ordered[-1].stage)
        read, write = _resident_savings(ordered, kernel_for, preserved, readers, path, all_ops=self.ops)
        live = consumed | declared | {key for key in retained if key[0] in keys}
        fused, outputs = self._assemble_fused_op(
            fused_id=fused_id, spec=spec, ordered=ordered, keys=keys,
            kernel_for=kernel_for, dependencies=dependencies, boundary=boundary,
            typed=typed, named=named, cut=cut, live=live, declared=declared,
            read=read, write=write, consumer_ops={op.op_id: op for op in users},
        )
        insertion = next(key for key in self.ops if key in keys)
        self.ops = {key: item for previous, op in self.ops.items() for key, item in (
            ((fused_id, fused), (previous, op)) if previous == insertion else ((previous, op),))}
        self._reindex_op_inputs(fused_id)
        self.replace_nodes(keys, fused_id, outputs)
        self.absorbed[fused_id] = tuple(atom for op in ordered
                                        for atom in self.absorbed.pop(op.op_id, (op.op_id,)))
        self.tiers[fused_id] = spec.fusion_tier
        for key in keys:
            self.tiers.pop(key, None)
        return tuple(op.op_id for op in ordered), (fused_id,)

    def hoist(self, spec):
        members = tuple(self.resolve(ref) for ref in spec.targets)
        if not members or len({op.op_id for op in members}) != len(members):
            raise ModelingError("rewrite_hoist_targets_invalid")
        ids = tuple(op.op_id for op in members)
        if spec.scope == "device_group_list_elide_d2h":
            for op in members:
                meta = dict(op.metadata)
                if not (meta.get("device_group_list_elide_d2h")
                        or op.family in {"split", "routing", "router"} | _HOST_SYNC_FAMILIES):
                    raise ModelingError(f"rewrite_d2h_ineligible:{op.op_id}")
                if op.family in _HOST_SYNC_FAMILIES:
                    if (len(op.deps) != 1 or op.deps[0] not in self.ops or op.flops
                            or op.matrix_flops or op.vector_ops or op.link_bytes):
                        raise ModelingError(f"rewrite_d2h_dependency_unbound:{op.op_id}")
                    source, = op.deps
                    self.ops[source] = _annotate(self.ops[source], device_group_list=True)
                    self.replace_nodes({op.op_id}, source)
                    continue
                self.ops[op.op_id] = _annotate(
                    op, elide_d2h=True, device_group_list=True, eager_d2h_sync=False,
                    requires_host_split_sizes=False)
            self.elided.extend(ids)
            return ids, tuple(key for key in ids if key in self.ops)
        if any(op.link_bytes or dict(op.metadata).get("state_effect")
               or op.op_id in self.absorbed for op in members):
            raise ModelingError("rewrite_hoist_effectful")
        if spec.scope == "load_time":
            initialized = set(self.hoisted)
            for op in _topological(self.ops, dependencies=self._base_deps):
                if _immutable(op) and set(op.deps) <= initialized:
                    initialized.add(op.op_id)
            if set(ids) - initialized:
                raise ModelingError("rewrite_load_time_dynamic_input")
            self.replace_nodes(set(ids), None)
            self.hoisted.extend(ids)
            return ids, ()
        def signature(op):
            metadata = tuple(sorted(op.metadata, key=lambda item: item[0]))
            return replace(op, op_id="", metadata=()), _json_digest(metadata, default=_digest_default)
        expected = signature(members[0])
        if any(dict(op.metadata).get("pure") is not True or signature(op) != expected
               for op in members):
            raise ModelingError("rewrite_cse_signature_mismatch")
        canonical = next(key for key in self.ops if key in ids)
        deleted = tuple(key for key in ids if key != canonical)
        self.replace_nodes(set(deleted), canonical)
        return deleted, (canonical,)
def _immutable(op):
    meta = dict(op.metadata)
    return bool(meta.get("constant") or meta.get("immutable") or meta.get("load_time")
                or op.family == "constant") and not (op.link_bytes or meta.get("state_effect"))
def _prefetch(draft, edit, scenario):
    spec = edit.operation
    target, anchor = draft.port(spec.target_port), draft.port(spec.window_anchor)
    if target.stage != anchor.stage or target.op_id == anchor.op_id:
        raise ModelingError("rewrite_prefetch_window_invalid")
    ordered = {op.op_id: i for i, op in enumerate(_topological(draft.ops, dependencies=draft._base_deps))}
    if ordered[anchor.op_id] >= ordered[target.op_id]:
        raise ModelingError("rewrite_prefetch_anchor_not_ready")
    meta = dict(target.metadata)
    weight = meta.get("weight_bytes", meta.get("local_weight_bytes", meta.get("weight_read_bytes", 0)))
    if (type(weight) not in {int, float} or not isfinite(weight) or weight <= 0
            or meta.get("weight_is_dynamic") or spec.target_port.version != 0):
        raise ModelingError("rewrite_prefetch_static_weight_required")
    if spec.max_bytes is not None and (type(spec.max_bytes) is not int or spec.max_bytes < 0):
        raise ModelingError("rewrite_prefetch_max_bytes_invalid")
    if any(lease.target_op == target.op_id for lease in draft.leases):
        raise ModelingError("rewrite_prefetch_duplicate_target")
    cap = _path(scenario, draft.replica, target.stage).usable_bytes(spec.pool)
    limit = cap if spec.max_bytes is None else min(cap, spec.max_bytes)
    amount = min(int(weight), int(target.hbm_bytes), limit)
    if amount:
        draft.leases.append(PrefetchLease(
            edit.edit_id, target.op_id, spec.target_port.name,
            anchor.op_id, spec.pool, amount, replica=draft.replica,
        ))
        # The replay moves these bytes only when a physical DMA/lease is bound.
        draft.ops[target.op_id] = _annotate(target, weight_prefetch_pool=spec.pool,
                                             weight_prefetch_bytes=amount,
                                             weight_prefetch_foreground_bytes=target.hbm_bytes - amount)
    return ((target.op_id,) if amount else ()), (target.op_id,)
def _quantize(draft, spec, scenario):
    if spec.group_size is not None and spec.group_size <= 0:
        raise ModelingError("rewrite_quant_group_invalid")
    for dtype in (spec.w_dtype, spec.a_dtype, spec.kv_dtype, spec.scale_dtype):
        if dtype is not None:
            dtype_bytes(dtype)
    contract = scenario.case.model.quant
    if contract is not None:
        declared = (contract.weight_format, contract.activation_format, contract.kv_dtype)
        if declared != (spec.w_dtype, spec.a_dtype, spec.kv_dtype) or (
            spec.group_size is not None and spec.group_size != contract.activation_group_size
        ) or (spec.scale_dtype is not None and spec.scale_dtype != contract.activation_scale_format):
            raise ModelingError("rewrite_quant_precision_mismatch")
    active = set(spec.scope) - set(spec.exclusions)
    found = []
    for op in draft.ops.values():
        meta = dict(op.metadata)
        scopes = {op.op_id, op.family, meta.get("model_scope"), meta.get("quant_scope")}
        if not (scopes & active) or scopes.intersection(spec.exclusions):
            continue
        for keys, dtype in ((("weight_format", "weight_dtype"), spec.w_dtype),
                            (("activation_format",), spec.a_dtype), (("kv_dtype",), spec.kv_dtype)):
            if any(meta.get(key) and meta[key] != dtype for key in keys):
                raise ModelingError(f"rewrite_quant_precision_mismatch:{op.op_id}")
        if contract is None and op.flops and op.dtype not in {spec.w_dtype, spec.a_dtype}:
            raise ModelingError(f"rewrite_quant_precision_mismatch:{op.op_id}")
        draft.ops[op.op_id] = _annotate(op, quant_w_dtype=spec.w_dtype, quant_a_dtype=spec.a_dtype,
                                       quant_kv_dtype=spec.kv_dtype, quant_group_size=spec.group_size,
                                       quant_scale_dtype=spec.scale_dtype)
        found.append(op.op_id)
    return tuple(found), tuple(found)
def _schedule(draft, spec, scenario, all_ids):
    def present(ref):
        return ref.id in (draft.ops if ref.kind == "atom" else draft.handles)

    assignments = tuple((draft.resolve(ref).op_id, lane)
                        for ref, lane in spec.assignments if present(ref))
    events = []
    for event in spec.events:
        endpoints = {event.record_after, *event.wait_before}
        if endpoints <= draft.ops.keys():
            events.append(event)
        elif endpoints & draft.ops.keys() or endpoints - all_ids:
            raise ModelingError(f"rewrite_stream_event_endpoint_missing:{event.event_id}")
    plan = StreamPlan(draft.replica, spec.lanes, assignments, tuple(events))
    for stage in {op.stage for op in draft.ops.values()}:
        bind_stream_owners(plan, _path(scenario, draft.replica, stage))
    _check_stream(_topological(draft.ops, dependencies=draft._base_deps), plan)
    draft.stream = plan
    draft._topo_pos = None
    return tuple(key for key, _ in assignments), ()
def _target_edit(draft, edit, scenario):
    spec = edit.operation
    target = draft.resolve(spec.target)
    key = target.op_id
    meta = dict(target.metadata)
    if isinstance(spec, SelectTemplate):
        params = dict(spec.params)
        if (not spec.template_id or len(params) != len(spec.params)
                or any(type(k) is not str or type(v) not in {str, int, bool}
                       for k, v in spec.params)):
            raise ModelingError("rewrite_template_params_invalid")
        hints = {f"template_param_{k}": v for k, v in params.items()}
        draft.ops[key] = _annotate(target, template_tier=spec.tier, template_id=spec.template_id, **hints)
        draft.templates.append(TemplateBinding(edit.edit_id, key, spec.tier, spec.template_id, spec.params))
    elif isinstance(spec, PipelineTiles):
        tile = spec.tile_spec
        if isinstance(tile, VariantRef):
            variants = dict(meta.get("tile_variants", ()))
            if not isinstance(variants.get(tile.variant_id), TilePipelineSpec):
                raise ModelingError(f"rewrite_tile_variant_unknown:{tile.variant_id}")
            tile = variants[tile.variant_id]
        if tile is None and spec.cohort_split == "none":
            return (), (key,)
        if spec.cohort_split == "dbo_2cohort":
            if target.family not in _DBO_FAMILIES:
                raise ModelingError(f"rewrite_dbo_family_unsupported:{target.family}")
            rows = meta.get("routed_rows", target.local_shape[0] if target.local_shape else 0)
            if "matrix_group_rows" in meta:
                rows = sum(meta["matrix_group_rows"])
            if rows // 2 <= 0:
                raise ModelingError(f"rewrite_dbo_empty_cohort:{key}")
        hints = {"cohort_split": spec.cohort_split}
        if tile is not None:
            solution, candidate = _solve_tiles(scenario, draft.replica, target, tile, draft.stream)
            hints.update(tile_algorithm=solution.algorithm, tile_uses_simt=candidate.uses_simt)
        draft.ops[key] = _annotate(target, **hints)
        draft.tiles.append(TileBinding(edit.edit_id, key, spec.cohort_split, tile))
    elif isinstance(spec, CollectiveTransform):
        if not (target.link_bytes > 0 or target.family in _COLLECTIVE_FAMILIES):
            raise ModelingError(f"rewrite_collective_target_invalid:{key}")
        if not spec.mode or not any(link.name == spec.topology_route for link in scenario.case.machine.links):
            raise ModelingError(f"rewrite_collective_route_unknown:{spec.topology_route}")
        draft.ops[key] = _annotate(target, collective_mode=spec.mode, topology_route=spec.topology_route)
        draft.collectives.append(CollectiveRewriteRecord(edit.edit_id, key, spec.mode, spec.topology_route))
    return (key,), (key,)
def _execute(draft, edit, scenario, all_ids):
    spec = edit.operation
    if isinstance(spec, FuseRegion):
        return draft.fuse(edit, scenario)
    if isinstance(spec, HoistOrDeduplicate):
        return draft.hoist(spec)
    if isinstance(spec, WeightPrefetch):
        return _prefetch(draft, edit, scenario)
    if isinstance(spec, StreamSchedule):
        return _schedule(draft, spec, scenario, all_ids)
    if isinstance(spec, QuantizeContract):
        return _quantize(draft, spec, scenario)
    return _target_edit(draft, edit, scenario)

def apply_plan(plan: RewritePlan, scenario: ResolvedScenario) -> AppliedPlan:
    """Apply the plan transactionally to independent rank-local scratch graphs."""
    for field in ("base_graph_sha256", "framing_sha256"):
        if getattr(plan, field) != getattr(scenario, field):
            raise ModelingError(f"rewrite_digest_mismatch:{field}")
    streams = {stream.rank: stream for stream in scenario.stream_plans}
    drafts = [_RankDraft(entry, streams.get(entry.replica)) for entry in scenario.atomic_ops]
    receipts, handles, quantized = [], [], []
    for edit in _ordered_edits(plan.edits, handles=True):
        refs = tuple(_references(edit.operation))
        if any(ref.kind == "atom" and not any(ref.id in d.ops for d in drafts) for ref in refs):
            raise ModelingError(f"rewrite_unresolved_atom:{edit.edit_id}")
        all_ids = (
            {key for draft in drafts for key in draft.ops}
            if isinstance(edit.operation, StreamSchedule)
            else None
        )
        affected, results = {}, {}
        eligible = 0
        for draft in drafts:
            if refs and not isinstance(edit.operation, StreamSchedule) and not all(
                ref.id in (draft.ops if ref.kind == "atom" else draft.handles) for ref in refs
            ):
                continue
            eligible += 1
            touched, output = _execute(draft, edit, scenario, all_ids)
            draft.handles[edit.edit_id] = output
            affected.update(dict.fromkeys(touched))
            results.update(dict.fromkeys(output))
        if not eligible:
            raise ModelingError(f"rewrite_no_matching_rank:{edit.edit_id}")
        if isinstance(edit.operation, QuantizeContract) and affected:
            quantized.append(edit.operation)
        schedule = isinstance(edit.operation, StreamSchedule)
        changes_streams = schedule and (edit.operation.events or edit.operation.lanes != (StreamLane(MAIN_STREAM),)
                                        or streams or any(op.stream not in {"", MAIN_STREAM}
                                            for rank in scenario.atomic_ops for op in rank.ops))
        status = "applied" if affected or changes_streams else "noop"
        receipts.append(EditReceipt(edit.edit_id, type(edit.operation).__name__, status, tuple(affected)))
        handles.append((edit.edit_id, tuple(results)))

    executable = []
    for draft in drafts:
        ordered = _topological(draft.ops, dependencies=draft._base_deps) if plan.edits else tuple(draft.ops.values())
        if draft.stream is not None:
            ordered = _check_stream(ordered, draft.stream)
        executable.append(RankOps(draft.replica, ordered))

    def collect(attribute):
        seen_vals = []
        for draft in drafts:
            seen_vals.extend(v for v in getattr(draft, attribute) if v not in seen_vals)
        return tuple(seen_vals)

    return AppliedPlan(
        plan, plan_sha256(plan), scenario.scenario_sha256, tuple(executable),
        tuple(draft.stream for draft in drafts if draft.stream is not None),
        tuple(handles), tuple(receipts),
        tuple(dict.fromkeys(item for draft in drafts for item in draft.absorbed.items())),
        collect("leases"), collect("templates"), collect("tiles"), collect("hoisted"),
        collect("elided"), tuple(quantized), collect("collectives"),
        tuple(dict.fromkeys(item for draft in drafts for item in draft.tiers.items())),
    )

apply = apply_plan

__all__ = [
    "ResolvedScenario", "EditReceipt", "PrefetchLease", "TemplateBinding", "TileBinding",
    "CollectiveRewriteRecord", "AppliedPlan", "resolve_scenario", "apply_plan", "apply",
]
