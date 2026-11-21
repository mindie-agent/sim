"""Transactional interpretation of declarative graph edits.

[Role] Bind a rewrite plan to its frozen scenario and transform rank-local DAGs.
[Invariants] Edits run in stable dependency order. Failed edits publish no graph.
  Rewrite annotations are flat and graph handles are kept outside operator metadata.
[Timeouts & Locks] Per-call scratch state, with no shared mutation or locks.
[Side-Effect Policy] Read-only lowering and immutable replacement records.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from graphlib import CycleError, TopologicalSorter
from heapq import heappop, heappush
from math import isfinite
from types import SimpleNamespace
from typing import Literal

from mindie_sim.prediction.fusion_control_rewrite import ordering_prerequisites
from mindie_sim.prediction.results import StepRequest
from mindie_sim.prediction.runtime import tile_solver
from mindie_sim.prediction.runtime.tile_solver import TilePipelineSpec
from mindie_sim.prediction.spine import PredictionSpine, PreparedStep
from mindie_sim.prediction.streams import MAIN_STREAM, StreamLane, StreamPlan, bind_stream_owners, lower_stream_edges
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import DatapathCapabilitySpec, datapath_for_die, ub_peer_edges_for_rank
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import DEFAULT_STACK_NAME, LoweringRequest, RankOps

from .schema import (
    CollectiveTransform, Edit, FuseRegion, FusionTier, HoistOrDeduplicate, OpRef,
    PipelineTiles, PortRef, QuantizeContract, RewritePlan, SelectTemplate, StreamSchedule,
    TemplateTier, VariantRef, WeightPrefetch, _digest_default, _flat, _json_digest, _ordered_edits, _references,
    plan_sha256,
)

DEFAULT_STACK_ID = "cann93_va_main"
SIMT_UB_CARVEOUT_BYTES = 16384
_DBO_FAMILIES = frozenset({
    "grouped_moe", "grouped_gemm", "moe_gemm", "routed_moe",
    "mc2", "mc2_protocol", "moe", "alltoall", "dispatch", "combine",
})
_COLLECTIVE_FAMILIES = _DBO_FAMILIES | {
    "collective", "allreduce", "all_reduce", "allgather", "all_gather",
    "reduce_scatter", "alltoallv", "p2p", "send", "recv",
}

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
def resolve_scenario(case, request: StepRequest, *, spine=None,
                     stack_id: str = DEFAULT_STACK_ID, primitives=None) -> ResolvedScenario:
    """Lower atomic and shipping graphs against a shared declared scenario."""
    case = Case.from_mapping(case)
    spine = PredictionSpine() if spine is None else spine
    stack = DEFAULT_STACK_NAME if stack_id == DEFAULT_STACK_ID else stack_id
    atomic = spine.lower(case, LoweringRequest(step=request, tier="unmerge",
                                               apply_body_defaults=False, stack=stack))
    mainline = spine.lower(case, LoweringRequest(step=request, tier="mainline",
                                                 apply_body_defaults=True, stack=stack))
    graph = atomic.graph
    base = _json_digest(graph.executable, default=_digest_default)
    framing = _json_digest(
        (case.to_dict(), atomic.case.runtime.to_dict(), asdict(request), stack_id,
         graph.base.sources, graph.streams, primitives),
        default=_digest_default,
    )
    digest = _json_digest((base, framing, mainline.graph_id))
    datapath = _die_path(atomic.case.machine, atomic.case.machine.dies[0])
    return ResolvedScenario(case, request, stack_id, base, framing, digest, atomic,
                            mainline, graph.executable, datapath, graph.streams)
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
    owners = (None if stream is None else bind_stream_owners(stream, path)[
        dict(stream.assignments).get(op.op_id, MAIN_STREAM)])
    hw = SimpleNamespace(datapath=path, buffers=die.buffers, buffer_units=die.buffer_units,
                         stream_owners=owners)
    if any(candidate.uses_simt for candidate in spec.candidates):
        spec = replace(spec, simt_ub_carveout_bytes=max(
            spec.simt_ub_carveout_bytes, path.simt_ub_carveout_bytes, SIMT_UB_CARVEOUT_BYTES))
    return tile_solver.solve(spec, op, hw)
def _successors(ops):
    users = {op.op_id: [] for op in ops}
    for key, predecessors in ordering_prerequisites(tuple(ops)).items():
        for dep in predecessors:
            users[dep].append(key)
    return users
def _convex(ops, members: set[str]) -> bool:
    users = _successors(ops)
    pending = [user for member in members for user in users[member] if user not in members]
    visited = set()
    while pending:
        node = pending.pop()
        if node in members:
            return False
        if node not in visited:
            visited.add(node)
            pending.extend(users[node])
    return True
def _topological(ops):
    positions = {key: i for i, key in enumerate(ops)}
    dependencies = ordering_prerequisites(tuple(ops.values()))
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
                name = port.get("name") or port.get("output") or port.get("consumer_field")
                if isinstance(name, str):
                    version = port.get("version", port.get("state_version"))
                    result.add((name, 0 if version is None else int(version)))
    return result
def _resident_savings(members, kernel_for, preserved, external_reads, path):
    """Charge GM unless both operand ledgers and a resident handoff establish savings."""
    saved_read = saved_write = 0.0
    for source in members:
        if source.op_id in preserved or source.op_id in external_reads:
            continue
        sm = dict(source.metadata)
        writes = sm.get("write_bytes")
        if type(writes) not in {int, float} or not 0 < writes <= min(source.hbm_bytes, source.mte3_bytes):
            continue
        consumers = [op for op in members if source.op_id in op.deps]
        if len(consumers) != 1:
            continue
        target = consumers[0]
        if kernel_for[source.op_id] != kernel_for[target.op_id]:
            continue
        tm = dict(target.metadata)
        capacity = dict(path.buffers)
        if source.flops and not target.flops:
            route = path.has_l0c_to_ub and writes <= capacity.get("ub", 0)
        elif not source.flops and target.flops:
            route = path.has_ub_to_l1 and writes <= capacity.get("l1", 0)
        else:
            pool = "l0c" if source.flops else "ub"
            route = (sm.get("owner_id") is not None
                     and sm.get("owner_id") == tm.get("owner_id")
                     and writes <= capacity.get(pool, 0))
        reads = tm.get("resident_input_bytes")
        if reads is None:
            matching = [p.get("read_bytes") for p in tm.get("typed_inputs", ())
                        if isinstance(p, Mapping) and p.get("producer") == source.op_id]
            reads = matching[0] if len(matching) == 1 else None
        if route and reads == writes and writes <= min(target.hbm_bytes, target.mte2_bytes):
            saved_read += writes
            saved_write += writes
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

    def replace_nodes(self, removed, replacement):
        removed = removed - {replacement}
        for key in removed:
            del self.ops[key]
        for key, op in self.ops.items():
            if removed.intersection(op.deps):
                deps = (replacement if dep in removed else dep for dep in op.deps)
                self.ops[key] = replace(op, deps=tuple(dict.fromkeys(dep for dep in deps if dep and dep != key)))
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
        if not _convex(tuple(self.ops.values()), keys):
            raise ModelingError(f"NON_CONVEX_REGION:{edit.edit_id}")
        ordered = tuple(op for op in _topological(self.ops) if op.op_id in keys)
        if any(kernel_for[dep] > kernel_for[op.op_id]
               for op in ordered for dep in op.deps if dep in keys):
            raise ModelingError("rewrite_kernel_boundary_dependency_order")
        cut = {dep for op in ordered for dep in op.deps if dep not in keys}
        readers = {dep for op in self.ops.values() if op.op_id not in keys
                   for dep in op.deps if dep in keys}
        preserved = {self.port(ref).op_id for ref in spec.preserve_ports}
        if preserved - (keys | cut):
            raise ModelingError("rewrite_fusion_port_outside_cut")
        if (readers - {ordered[-1].op_id}) - preserved:
            raise ModelingError("rewrite_fusion_live_out_unpreserved")
        if any(op.link_bytes for op in ordered) and len({op.link_scope for op in ordered if op.link_bytes}) > 1:
            raise ModelingError("rewrite_fusion_mixed_link_scopes")
        tail = ordered[-1]
        fused_id = f"rewrite:{edit.edit_id}"
        if fused_id in self.ops:
            raise ModelingError(f"rewrite_duplicate_fused_id:{fused_id}")
        path = _path(scenario, self.replica, tail.stage)
        read, write = _resident_savings(ordered, kernel_for, preserved, readers, path)
        matrix = Counter()
        for op in ordered:
            for dtype, amount in (op.matrix_flops or (((op.dtype, op.flops),) if op.flops else ())):
                matrix[dtype] += amount
        link_scope = next((item.link_scope for item in ordered if item.link_bytes), tail.link_scope)
        abi = spec.kernel_boundaries[-1].abi_id
        fused = replace(
            tail, op_id=fused_id, family="rewrite_fused", recipe=abi, source_boundary="",
            deps=tuple(dict.fromkeys(dep for op in ordered for dep in op.deps if dep in cut)),
            returns=tuple(dict.fromkeys((*tail.returns, *(ref.name for ref in spec.preserve_ports)))),
            flops=sum(op.flops for op in ordered), matrix_flops=tuple(sorted(matrix.items())),
            vector_ops=sum(op.vector_ops for op in ordered),
            link_bytes=sum(op.link_bytes for op in ordered), link_scope=link_scope,
            hbm_bytes=sum(op.hbm_bytes for op in ordered) - read - write,
            mte2_bytes=sum(op.mte2_bytes for op in ordered) - read,
            mte3_bytes=sum(op.mte3_bytes for op in ordered) - write,
            metadata=(("fusion_tier", spec.fusion_tier), ("kernel_abi", abi),
                      ("gm_round_trip_eliminated_bytes", read + write),
                      ("fusion_route", "resident" if read or write else "gm_spill")),
        )
        insertion = next(key for key in self.ops if key in keys)
        self.ops = {key: item for previous, op in self.ops.items() for key, item in (
            ((fused_id, fused), (previous, op)) if previous == insertion else ((previous, op),))}
        self.replace_nodes(keys, fused_id)
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
                        or op.family in {"split", "routing", "router", "eager_split_barrier"}):
                    raise ModelingError(f"rewrite_d2h_ineligible:{op.op_id}")
                self.ops[op.op_id] = _annotate(
                    op, elide_d2h=True, device_group_list=True, eager_d2h_sync=False,
                    requires_host_split_sizes=False)
            self.elided.extend(ids)
            return ids, ids
        if any(op.link_bytes or dict(op.metadata).get("state_effect")
               or op.op_id in self.absorbed for op in members):
            raise ModelingError("rewrite_hoist_effectful")
        if spec.scope == "load_time":
            initialized = set(self.hoisted)
            for op in _topological(self.ops):
                if _immutable(op) and set(op.deps) <= initialized:
                    initialized.add(op.op_id)
            if set(ids) - initialized:
                raise ModelingError("rewrite_load_time_dynamic_input")
            self.replace_nodes(set(ids), None)
            self.hoisted.extend(ids)
            return ids, ()
        def signature(op):
            meta = dict(op.metadata)
            return (op.family, op.dtype, op.local_shape, op.deps, op.stage, op.layout,
                    op.requests, op.returns, op.recipe, op.flops, op.vector_ops,
                    meta.get("input_port_versions"), meta.get("semantic_contract", meta.get("contract")))
        if any(dict(op.metadata).get("pure") is not True or signature(op) != signature(members[0])
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
    ordered = {op.op_id: i for i, op in enumerate(_topological(draft.ops))}
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
    limit = min(cap, spec.max_bytes or cap)
    amount = min(int(weight), int(target.hbm_bytes), limit)
    if amount:
        draft.leases.append(PrefetchLease(edit.edit_id, target.op_id, spec.target_port.name,
                                          anchor.op_id, spec.pool, amount))
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
    _check_stream(_topological(draft.ops), plan)
    draft.stream = plan
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
            solution = _solve_tiles(scenario, draft.replica, target, tile, draft.stream)
            hints.update(tile_algorithm=solution.algorithm, tile_uses_simt=any(
                choice.uses_simt and choice.algorithm == solution.algorithm
                and choice.tile_shape == solution.tile_shape for choice in tile.candidates))
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
        all_ids = {key for draft in drafts for key in draft.ops}
        if any(ref.kind == "atom" and ref.id not in all_ids for ref in refs):
            raise ModelingError(f"rewrite_unresolved_atom:{edit.edit_id}")
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
        ordered = _topological(draft.ops) if plan.edits else tuple(draft.ops.values())
        if draft.stream is not None:
            ordered = _check_stream(ordered, draft.stream)
        executable.append(RankOps(draft.replica, ordered))

    def collect(attribute):
        result = []
        for draft in drafts:
            for value in getattr(draft, attribute):
                if value not in result:
                    result.append(value)
        return tuple(result)

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
