"""Canonical Case/ModelSpec -> selected OpSpec bridge for the optimizer.

[Role] Project the selected ``Case``/``ModelSpec`` operator DAG into the typed
  graph used for candidate discovery, apply fusion/stream rewrites to that
  ``OpSpec`` tuple, and score every candidate through the common
  ``graph_costs(..., include_submission=True, isolated=False)`` path and its
  real ``Execution``/``Engine``.
[Invariants] A fused node preserves every declared physical service field
  except the proven eliminated HBM access. The projection carries the original
  op id, family, stream, stage, phase, real state ports/versions and declared
  weight provenance; a source-kernel planner stays opaque and non-fusable.
  The complete global positional data-operand sequence survives every
  contraction and every generated stream event independently of the
  deduplicated scheduling ``deps``; a record/wait pair is a control edge and
  never replaces a data producer in the tensor expression.
  Evaluation lowers projected tensor/state and external readiness only on
  transient copies; equal stream names never substitute for completion.
  Trials own a private ``RecipeBook`` overlay; recipe and device-group ids are
  occurrence-scoped. ``graph_costs`` remains the only timing authority.
[Timeouts & Locks] Finite CASE graph; no I/O, clocks, threads or locks.
[Side-Effect Policy] Returns new immutable OpSpec tuples and private books.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

from mindie_sim.prediction.body_defaults import consume_target_declaration
from mindie_sim.prediction.execution.engine_plan import HardwareProfile
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec

from mindie_sim.prediction.execution.capture_barriers import normalized_capture_metadata
from mindie_sim.prediction.fusion.control_rewrite import (
    apply_incoming_controls, region_boundary_ports, rewrite_consumers,
)
from mindie_sim.prediction.fusion.feasibility import CubeResources, ReductionMerge
from mindie_sim.prediction.fusion.ports import (
    INPUT_PORTS_KEY, event_control_rewrite, full_operand_ports,
    replace_input_ports, replace_scalar_metadata, restore_input_ports, retained_interface,
    retained_ports, rewritten_dependencies,
)
from mindie_sim.prediction.fusion.recipe import candidate_with_hbm
from mindie_sim.prediction.fusion.graph import TypedGraph
from mindie_sim.prediction.fusion.pool import (
    PoolHolder, PrefetchPoolProgram, ReleasePoolProgram, register_recipe,
    units_for_pool,
)
from mindie_sim.prediction.fusion.projection import execution_prerequisites, project_graph
from mindie_sim.prediction.execution.weight_prefetch import PrefetchDecision

@dataclass(frozen=True)
class CanonicalPlan:
    case: Any
    ops: tuple[OpSpec, ...]
    graph: TypedGraph
    hardware: HardwareProfile
    external_inputs: tuple[str, ...]
    book: RecipeBook
    model_name: str = ""
    provenance: str = "selected_opspec_projection"
    external_ready_ms: tuple[tuple[str, float], ...] = ()
    opaque_ops: tuple[str, ...] = ()
    external_data: tuple[str, ...] = ()
    pool_holders: tuple[tuple[str, Any], ...] = ()

    def evaluate(self, ops: tuple[OpSpec, ...] | None = None,
                 *, book: RecipeBook | None = None,
                 capture_mode: str | None = None,
                 audit_allocations: bool = False) -> dict:
        chosen = tuple(self.ops if ops is None else ops)
        # Reproject both entry points: even dataclass replacement or mutable
        # caller metadata must not leave evaluation using a stale stored graph.
        # Validate crossings before transient lowering can mask missing order.
        projected = project_graph(self.case, chosen, opaque_ops=self.opaque_ops,
                                  external_data=self.external_data)
        projected.validate()
        execution_ops = execution_prerequisites(chosen, projected)
        external = {name: 0.0 for name in
                    (*self.external_inputs, *projected.external_inputs)}
        external.update({name: float(value) for name, value in self.external_ready_ms
                         if name in external})
        return graph_costs(self.case, execution_ops, recipes=book or self.book,
                           external_inputs=external, include_submission=True,
                           isolated=False,
                           audit_allocations=audit_allocations,
                           replay_scope="full" if capture_mode == "full" else None)

    def rewrite(self, ops: tuple[OpSpec, ...], name: str = "") -> "CanonicalPlan":
        projection = project_graph(self.case, ops, opaque_ops=self.opaque_ops,
                                   external_data=self.external_data)
        return replace(self, ops=ops, graph=projection, model_name=name or self.model_name)


def _canonical_hardware(case, hardware: HardwareProfile | None,
                        options: Mapping[str, Any]) -> HardwareProfile:
    if hardware is not None:
        return hardware
    overrides = dict(options.get("hardware", {}))
    if "prefetch_capacity_bytes" not in overrides:
        die = case.machine.dies[0]
        buffers = dict(die.buffers)
        overrides["prefetch_capacity_bytes"] = int(
            buffers.get("l1", 0) * die.pool_units("l1"))
    return HardwareProfile.from_case(case, **overrides)


def _opaque_ids(ops: tuple[OpSpec, ...], book: RecipeBook) -> tuple[str, ...]:
    found = set()
    for op in ops:
        if op.source_boundary:
            found.add(op.op_id)
            continue
        try:
            recipe = book.select(op)
        except TimeoutError:
            raise
        except Exception:
            continue
        if isinstance(recipe, PlannedRecipe) and not getattr(
            recipe, "traffic_rewrite_capable", False
        ):
            found.add(op.op_id)
    return tuple(sorted(found))


def canonical_plan(case, ops, *, hardware: HardwareProfile | None = None,
                   options: Mapping[str, Any] | None = None) -> CanonicalPlan:
    """Parse/select the canonical Case and build its projection + recipe book."""
    options = dict(options or {})
    target = ops
    raw_ops = tuple(ops)
    consumed = consume_target_declaration(parse_case(case), target)
    book = RecipeBook.from_mapping(getattr(consumed.runtime, "kernels", {}) or {})
    opaque = _opaque_ids(raw_ops, book)
    external_data = tuple(str(item) for item in options.get(
        "external_inputs", ("analysis_input",)))
    projection = project_graph(consumed, raw_ops, opaque_ops=opaque,
                               external_data=external_data)
    external = projection.external_inputs
    ready = tuple(sorted((str(name), float(value)) for name, value
                         in dict(options.get("external_ready_ms", {})).items()))
    return CanonicalPlan(consumed, raw_ops, projection,
                         _canonical_hardware(consumed, hardware, options),
                         external, book,
                         str(getattr(getattr(consumed, "model", None), "name", "")),
                         "selected_opspec_projection", ready, opaque,
                         external_data, ())


def add_recipe(book: RecipeBook, recipe_id: str, recipe: PlannedRecipe) -> RecipeBook:
    """Register one optimizer recipe in a private overlay; never mutate ``book``."""
    return register_recipe(book, recipe_id, recipe)


def fused_op_spec(candidate, *, phase: str | None = None,
                  declared_hbm: int | None = None,
                  member_specs: tuple[OpSpec, ...] = (),
                  boundary_ports: tuple[tuple[str, str], ...] = ()) -> OpSpec:
    """Rewrite one candidate region into a real OpSpec carrying full service."""
    analysis = candidate.analysis
    member_ops = tuple(candidate.member_ops)
    saved = int(analysis.traffic.saved_bytes)
    if declared_hbm is None:
        hbm_after = int(analysis.traffic.candidate_bytes)
    else:
        # Conserve every declared byte except the proven eliminated accesses.
        hbm_after = max(0, int(declared_hbm) - saved)
    first = analysis.ops[member_ops[0]]
    fused_id = "fused:" + "+".join(member_ops)
    retained_value, retained_spec = retained_interface(analysis, member_specs)
    port_records, value_port, _primary_value = retained_ports(
        analysis, member_specs, fused_id)
    matrix: dict[str, float] = {}
    vector = mte2 = mte3 = link = cube = 0.0
    for op_id in member_ops:
        op = analysis.ops[op_id]
        declared = op.param("matrix_flops")
        if isinstance(declared, Mapping) and declared:
            for dtype, amount in declared.items():
                matrix[str(dtype)] = matrix.get(str(dtype), 0.0) + float(amount)
        vector += dict(op.work).get("vector", 0.0)
        cube += dict(op.work).get("cube", 0.0)
        mte2 += dict(op.work).get("mte2", 0.0)
        mte3 += dict(op.work).get("mte3", 0.0)
        link += dict(op.work).get("link", 0.0)
    declared_matrix = sum(matrix.values())
    if cube > declared_matrix:
        dtype = first.dtype or "bf16"
        matrix[dtype] = matrix.get(dtype, 0.0) + (cube - declared_matrix)
    flops = sum(matrix.values())
    allocations = tuple(analysis.worksheet.allocations) + tuple(
        analysis.worksheet.cube_allocations)
    buffers = [{"id": a.name, "pool": a.pool, "bytes": a.total_bytes,
                "units": tuple(a.units), "lifetime": a.lifetime}
               for a in allocations]
    extra: dict[str, float] = {}
    group_contract: dict[str, object] = {}
    member_hbm = {spec.op_id: float(spec.hbm_bytes) for spec in member_specs}
    for op_id in member_ops:
        source = analysis.ops[op_id]
        for resource, amount in source.work:
            if resource == "hbm":
                # ``metadata.resource_work`` is additive to the OpSpec field;
                # only the declared storage field is reduced by proven savings.
                delta = float(amount) - member_hbm.get(op_id, 0.0)
                if delta > 1e-9:
                    extra["hbm"] = extra.get("hbm", 0.0) + delta
            elif resource not in {"vector", "cube", "mte2", "mte3", "link"}:
                extra[resource] = extra.get(resource, 0.0) + float(amount)
        for key in ("group_sizes", "group_list", "expert_rows"):
            if source.param(key) is not None:
                group_contract[key] = source.param(key)
    # Phase-only work (priced Cube/Vector handoff, cross-tile merge) is part of
    # the executed service even though no member OpSpec field declares it.
    phase_totals: dict[str, float] = {}
    for _phase, phase_demand in analysis.phase_work:
        for resource, amount in phase_demand.items():
            phase_totals[resource] = phase_totals.get(resource, 0.0) + float(amount)
    extra_vector = phase_totals.get("vector", 0.0) - vector
    if extra_vector > 1e-9:
        vector += extra_vector
    for resource, amount in phase_totals.items():
        if resource in {"hbm", "vector", "cube", "mte2", "mte3", "link"}:
            continue
        delta = amount - extra.get(resource, 0.0)
        if delta > 1e-9:
            extra[resource] = extra.get(resource, 0.0) + delta
    retained_records = tuple(
        (analysis.values[value_id].value_id, analysis.values[value_id].dtype,
         tuple(analysis.values[value_id].shape), analysis.values[value_id].layout)
        for value_id in getattr(analysis, "retained_outputs", ())
        if value_id in analysis.values)
    metadata = (
        ("device_kernel_group", {"id": candidate.recipe_id,
                                 "kernel": candidate.pattern.fused_op_name,
                                 "entry": True, "exit": True,
                                 "buffers": tuple(buffers)}),
        ("optimizer_members", member_ops),
        ("numerical_policy", candidate.numerical_policy),
        ("fusion_role", candidate.recipe_id),
        ("fusion_traffic", tuple((v.value_id, v.saved_bytes)
                                 for v in analysis.traffic.values)),
        ("declared_hbm_before", int(declared_hbm or 0)),
        ("proven_eliminated_hbm", saved),
        # The full surviving interface, including secondary declared ports.
        ("retained_outputs", retained_records),
        *(
            (("uses_simt", True),)
            if any(
                analysis.ops[m].param("uses_simt") or analysis.ops[m].param("tile_uses_simt")
                for m in member_ops
            )
            else ()
        ),
        ("retained_ports", tuple(record.metadata_record()
                                 for record in port_records)),
        ("retained_port_map", tuple(sorted(value_port.items()))),
    )
    capture_meta = normalized_capture_metadata(member_specs)
    if capture_meta:
        metadata = (*metadata, *tuple(sorted(capture_meta.items())))
    if extra:
        metadata = (*metadata, ("resource_work", dict(sorted(extra.items()))))
    if group_contract:
        metadata = (*metadata, ("group_contract", dict(sorted(group_contract.items()))))
    if boundary_ports:
        # Compose the surviving boundary operands, not only the producer ids:
        # a prior contraction may expose two distinct ports of one producer.
        metadata = (*metadata, (INPUT_PORTS_KEY, tuple(boundary_ports)))
    leader = member_specs[0] if member_specs else None
    if retained_value is not None:
        # A retained output is an external interface: its dtype/shape/layout is
        # authoritative even under relaxed numerics.  Never copy the first
        # member's dtype into the surviving output.
        observed = tuple(int(dim) for dim in retained_value.shape) or (1,)
        if retained_spec is not None \
                and tuple(int(dim) for dim in retained_spec.local_shape) == observed:
            local_shape = observed
            padded_shape = tuple(int(dim) for dim in
                                 (retained_spec.padded_shape or observed)) or observed
            launch_shape = tuple(int(dim) for dim in
                                 (retained_spec.launch_shape or observed)) or observed
            semantic_shape = tuple(int(dim) for dim in
                                   (retained_spec.semantic_shape or observed)) or observed
        else:
            local_shape = padded_shape = launch_shape = semantic_shape = observed
        dtype = retained_value.dtype or (leader.dtype if leader is not None
                                         else first.dtype)
        layout = retained_value.layout or (
            leader.layout if leader is not None else "contiguous")
    else:
        local_shape = tuple(int(dim) for dim in (
            leader.local_shape if leader is not None else (1,))) or (1,)
        padded_shape = tuple(int(dim) for dim in (
            leader.padded_shape if leader is not None else local_shape)) or local_shape
        launch_shape = tuple(int(dim) for dim in (
            leader.launch_shape if leader is not None else local_shape)) or local_shape
        semantic_shape = (tuple(int(dim) for dim in leader.semantic_shape)
                          if leader is not None and leader.semantic_shape
                          else first.param("semantic_shape", (1,)))
        dtype = (leader.dtype if leader is not None else first.dtype)
        layout = (leader.layout if leader is not None else first.layout)
    return OpSpec(
        op_id=fused_id, family="native_fusion",
        phase=phase or first.phase, stage=first.stage, layer=first.stage,
        requests=(), semantic_shape=semantic_shape,
        local_shape=local_shape, padded_shape=padded_shape,
        launch_shape=launch_shape,
        flops=flops, hbm_bytes=float(hbm_after),
        mte2_bytes=mte2, mte3_bytes=mte3, vector_ops=vector, link_bytes=link,
        matrix_flops=tuple(sorted(matrix.items())),
        dtype=dtype, layout=layout,
        recipe=candidate.recipe_id,
        coverage="analytical_prototype",
        metadata=tuple(sorted(metadata, key=lambda item: item[0])),
    )


@dataclass(frozen=True)
class Rewrite:
    ops: tuple[OpSpec, ...]
    fused: OpSpec
    recipe: PlannedRecipe
    candidate_id: str
    members: tuple[str, ...]


def rewrite_fusion(plan: CanonicalPlan, candidate) -> Rewrite:
    """Contract the region in the selected OpSpec tuple and register its recipe."""
    members = set(candidate.member_ops)
    present = {op.op_id for op in plan.ops}
    if not members <= present:
        raise ValueError("fusion_member_no_longer_present")
    member_specs = tuple(op for op in plan.ops if op.op_id in members)
    # Materialize every operator's complete positional data-operand tuple and
    # its control classifier before touching deps.  A later deps deduplication
    # must never substitute for the ordered data-operand expression, and a
    # caller event must stay a scheduling edge rather than becoming a tensor.
    by_name = {op.op_id: op for op in plan.ops}
    operand_layout = {op.op_id: full_operand_ports(op, by_name)
                      for op in plan.ops}
    declared = int(sum(float(op.hbm_bytes) for op in member_specs))
    # Additive metadata work is executed service, never part of the proven
    # elided intermediate tensor bytes.  Include it in the conserved total so
    # the recipe rebalance cannot delete it.
    additive = int(sum(float(dict(op.metadata).get("resource_work", {}).get("hbm", 0.0))
                       for op in member_specs))
    conserved = declared + additive
    if conserved:
        candidate = candidate_with_hbm(
            candidate, max(0, conserved - int(candidate.analysis.traffic.saved_bytes)))
    boundary = region_boundary_ports(plan.ops, members, operand_layout)
    fused = fused_op_spec(
        candidate, declared_hbm=declared, member_specs=member_specs,
        boundary_ports=boundary)
    _records, value_port, _primary = retained_ports(
        candidate.analysis, member_specs, fused.op_id)
    # Incoming controls become real scheduling edges and keep their control
    # classification on the fused node; a name that is also a genuine boundary
    # data producer stays a data operand and is never declared control-only.
    fused = apply_incoming_controls(
        fused, member_specs, members, boundary, by_name)
    rewritten, anchor = rewrite_consumers(
        plan.ops, members, operand_layout, fused, value_port, by_name)
    rewritten = list(rewritten)
    rewritten.insert(anchor, fused)
    return Rewrite(tuple(rewritten), fused, candidate.recipe, candidate.candidate_id,
                   candidate.member_ops)


def apply_rewrite(plan: CanonicalPlan, rewrite: Rewrite) -> CanonicalPlan:
    book = add_recipe(plan.book, rewrite.fused.recipe, rewrite.recipe)
    return replace(plan, ops=rewrite.ops, book=book,
                   graph=project_graph(plan.case, rewrite.ops,
                                       opaque_ops=plan.opaque_ops,
                                       external_data=plan.external_data))


def evaluate_rewrite(plan: CanonicalPlan, rewrite: Rewrite) -> tuple[CanonicalPlan, dict]:
    updated = apply_rewrite(plan, rewrite)
    return updated, updated.evaluate()


@dataclass(frozen=True)
class PrefetchRewrite:
    ops: tuple[OpSpec, ...]
    prefetch: OpSpec
    consumer: OpSpec
    recipe: PlannedRecipe
    target_bytes: int
    completed_bytes: int


def _prefetch_recipe(decision: PrefetchDecision, holder: PoolHolder,
                     amount: int) -> PlannedRecipe:
    work = Work(resources={"hbm": float(amount)})

    def program(context):
        names = context.resource_names
        return PrefetchPoolProgram(
            holder, float(amount), units_for_pool(context, "l1"),
            caps={names["hbm"]: float(decision.spare_bytes_per_ms)},
            priority=decision.priority, work_resource=names["hbm"],
            pool_name=dict(context.pool_names).get("l1", "l1"))

    def build(op, geometry):
        del op, geometry
        return OperatorPlan(work, program, metadata={"optimizer_prefetch": True})

    return PlannedRecipe(build)


def _release_recipe(holder: PoolHolder) -> PlannedRecipe:
    def program(context):
        # The release must resolve the same per-(replica, stage) physical pool
        # namespace the copy acquired; a bare holder lookup would free a peer
        # replica's grant in a shared DP/PP Engine.
        return ReleasePoolProgram(
            holder, str(dict(context.pool_names).get("l1", "l1")))

    def build(op, geometry):
        del op, geometry
        return OperatorPlan(Work(), program, metadata={"optimizer_pool_release": True})

    return PlannedRecipe(build)


def add_prefetch(plan: CanonicalPlan, decision: PrefetchDecision,
                 hardware: HardwareProfile) -> tuple[CanonicalPlan, str] | None:
    """Add a low-priority prefetch and its consumer-completion pool release."""
    consumer = next((op for op in plan.ops if op.op_id == decision.op_id), None)
    if consumer is None:
        return None
    prefetch_id = f"prefetch_W_{decision.op_id}"
    release_id = f"release_W_{decision.op_id}"
    existing = {op.op_id for op in plan.ops}
    if prefetch_id in existing or release_id in existing:
        return None
    holder = PoolHolder(prefetch_id)
    prefetch = OpSpec(
        op_id=prefetch_id, family="prefetch", phase=consumer.phase,
        stage=consumer.stage, layer=consumer.layer, requests=(),
        semantic_shape=(1,), local_shape=(1,), padded_shape=(1,), launch_shape=(1,),
        hbm_bytes=float(decision.target_bytes), matrix_flops=(),
        dtype=consumer.dtype, layout=consumer.layout, recipe=prefetch_id,
        coverage="analytical_prototype", stream="prefetch",
        metadata=(("prefetch_target", decision.op_id),
                  ("prefetch_priority", decision.priority),
                  ("prefetch_caps", {"hbm": float(decision.spare_bytes_per_ms)}),
                  ("prefetch_pool", "l1"),
                  ("prefetch_units", int(hardware.prefetch_units)),
                  ("prefetch_bytes", int(decision.target_bytes)),
                  ("pointer_stable", True)),
    )
    release = OpSpec(
        op_id=release_id, family="release", phase=consumer.phase,
        stage=consumer.stage, layer=consumer.layer, requests=(),
        semantic_shape=(1,), local_shape=(1,), padded_shape=(1,), launch_shape=(1,),
        # Hold the L1 grant until both the consumer kernel and the producer
        # copy have retired; the consumer alone can outlive a delayed copy.
        deps=(consumer.op_id, prefetch_id), recipe=release_id, coverage="pool_release",
        dtype=consumer.dtype, layout=consumer.layout, stream=consumer.stream or "main",
        metadata=(("prefetch_release_for", prefetch_id),
                  ("prefetch_target", decision.op_id),
                  ("prefetch_priority", decision.priority),
                  ("prefetch_release_bytes", int(decision.target_bytes)),
                  ("prefetch_release_units", int(hardware.prefetch_units)),
                  ("prefetch_pool", "l1"),
                  ("pointer_stable", True)),
    )
    book = add_recipe(plan.book, prefetch_id,
                      _prefetch_recipe(decision, holder, decision.target_bytes))
    book = add_recipe(book, release_id, _release_recipe(holder))
    index = next(i for i, item in enumerate(plan.ops) if item.op_id == consumer.op_id)
    before = plan.ops[:index]
    after = plan.ops[index + 1:]
    ops = (*before, prefetch, consumer, release, *after)
    updated = replace(plan, ops=ops, book=book,
                      graph=project_graph(plan.case, ops,
                                          opaque_ops=plan.opaque_ops,
                                          external_data=plan.external_data),
                      pool_holders=(*plan.pool_holders, (prefetch_id, holder)))
    return updated, prefetch_id


def settle_prefetch_rewrite(plan: CanonicalPlan, decision: PrefetchDecision,
                            prefetch_id: str, completed_bytes: int,
                            hardware: HardwareProfile) -> PrefetchRewrite | None:
    done = max(0, min(int(completed_bytes), int(decision.target_bytes)))
    if done <= 0:
        return None
    prefetch = next((op for op in plan.ops if op.op_id == prefetch_id), None)
    consumer = next((op for op in plan.ops if op.op_id == decision.op_id), None)
    release_id = f"release_W_{decision.op_id}"
    if prefetch is None or consumer is None:
        return None
    if done > int(consumer.hbm_bytes) + 1:
        raise ValueError("prefetch_exceeds_consumer_hbm")
    holder = dict(plan.pool_holders).get(prefetch_id)
    if holder is None:
        raise ValueError("prefetch_pool_holder_missing")
    resource_work = dict(dict(consumer.metadata).get("resource_work", {}))
    resource_work["mte1"] = float(resource_work.get("mte1", 0.0) + done)
    metadata = replace_scalar_metadata(
        consumer.metadata, resource_work=resource_work, prefetch_hit_bytes=float(done))
    # The prefetch id is a scheduling dependency, but appending it to a
    # deduplicated ``deps`` tuple must not regroup or delete the consumer's
    # positional data operands.  Materialize the complete sequence first.
    by_name = {op.op_id: op for op in plan.ops}
    positions = full_operand_ports(consumer, by_name)
    entries = (*positions, (prefetch_id, f"{prefetch_id}:out"))
    scheduled = (*rewritten_dependencies(consumer, positions, {}, by_name),
                 prefetch_id)
    consumer = replace(
        consumer, hbm_bytes=float(consumer.hbm_bytes - done),
        metadata=replace_input_ports(metadata, entries),
        deps=tuple(dict.fromkeys(scheduled)))
    ops = []
    for op in plan.ops:
        if op.op_id == consumer.op_id:
            ops.append(consumer)
        elif op.op_id == prefetch_id:
            ops.append(replace(op, hbm_bytes=float(done),
                               metadata=replace_scalar_metadata(
                                   op.metadata, prefetch_bytes=int(done))))
        elif op.op_id == release_id:
            ops.append(replace(op, metadata=replace_scalar_metadata(
                op.metadata, prefetch_release_bytes=int(done))))
        else:
            ops.append(op)
    return PrefetchRewrite(tuple(ops), replace(prefetch, hbm_bytes=float(done)),
                           consumer, _prefetch_recipe(decision, holder, done),
                           decision.target_bytes, done)


def apply_prefetch_rewrite(plan: CanonicalPlan,
                           rewrite: PrefetchRewrite, prefetch_id: str) -> CanonicalPlan:
    book = add_recipe(plan.book, prefetch_id, rewrite.recipe)
    return replace(plan, ops=rewrite.ops, book=book,
                   graph=project_graph(plan.case, rewrite.ops,
                                       opaque_ops=plan.opaque_ops,
                                       external_data=plan.external_data))


def _event_op(consumer: OpSpec, name: str, deps: tuple[str, ...], cost: float,
              kind: str) -> OpSpec:
    return OpSpec(
        op_id=name, family="event", phase=consumer.phase, stage=consumer.stage,
        layer=consumer.layer, requests=(), semantic_shape=(1,),
        local_shape=(1,), padded_shape=(1,), launch_shape=(1,),
        deps=deps, recipe="", coverage="event_cost", stream="event",
        dtype=consumer.dtype, layout=consumer.layout,
        metadata=(("event_kind", kind), ("optimizer_generated_event", True),
                  ("resource_work", {"host": float(cost)})))


def _strip_events(ops: tuple[OpSpec, ...]) -> tuple[OpSpec, ...]:
    """Remove only optimizer-generated crossings, preserving caller events.

    A caller-authored ``family="event"`` operator and every dependency on it
    are semantic graph nodes.  Rewiring is restricted to a generated event's
    own record/wait chain, which maps back to its original producer.
    """
    generated = {
        op.op_id: op for op in ops
        if op.family == "event"
        and bool(dict(op.metadata).get("optimizer_generated_event"))}
    clean = tuple(op for op in ops if op.op_id not in generated)
    mapping: dict[str, tuple[str, ...]] = {}
    # Explicit post-order stack: event chains can be as deep as the caller's
    # graph and must never exhaust Python recursion.
    for start in generated:
        if start in mapping:
            continue
        stack = [(start, False)]
        while stack:
            event_id, expanded = stack.pop()
            if event_id in mapping:
                continue
            unresolved = [dep for dep in generated[event_id].deps
                          if dep in generated and dep not in mapping]
            if unresolved and not expanded:
                stack.append((event_id, True))
                stack.extend((dep, False) for dep in reversed(unresolved))
                continue
            found: list[str] = []
            for dep in generated[event_id].deps:
                if dep in generated:
                    found.extend(mapping.get(dep, ()))
                else:
                    found.append(dep)
            mapping[event_id] = tuple(dict.fromkeys(found))
    restored = []
    for op in clean:
        direct = {dep for dep in op.deps if dep not in generated}
        deps: list[str] = []
        for dep in op.deps:
            if dep not in generated:
                deps.append(dep)
                continue
            for origin in mapping[dep]:
                # A restored producer that is already a direct dependency (for
                # example the fused data producer plus the wait that wrapped it)
                # must not be duplicated: the scheduling tuple would then imply
                # a second tensor operand the explicit port list never declared.
                if origin not in direct and origin not in deps:
                    deps.append(origin)
        deps = tuple(deps)
        metadata = restore_input_ports(op.metadata, mapping) if deps != op.deps \
            else op.metadata
        restored.append(replace(op, deps=deps, metadata=metadata)
                        if deps != op.deps or metadata != op.metadata else op)
    return tuple(restored)


def insert_stream_events(ops: tuple[OpSpec, ...],
                         hardware: HardwareProfile) -> tuple[OpSpec, ...]:
    """Insert record/wait work once per cross-stream edge.

    Existing event nodes are stripped and the clean graph's edges are recomputed,
    so repeating the rewrite is idempotent rather than wrapping old events.  A
    wait event is a *control* dependency: before any producer is replaced in
    ``deps``, the consumer's complete positional data-operand sequence is
    materialized so projection still reads the original producer ports.  The
    sequence is preserved verbatim (including a repeated producer) while the
    scheduling ``deps`` are independently deduplicated.
    """
    clean = _strip_events(ops)
    if hardware.record_ms <= 0 and hardware.wait_ms <= 0:
        return clean
    by_id = {op.op_id: op for op in clean}
    result: list[OpSpec] = []
    for op in clean:
        replacements: dict[str, str] = {}
        emitted: set[str] = set()
        for dep in op.deps:
            parent = by_id.get(dep)
            if parent is None or (parent.stream or "main") == (op.stream or "main"):
                continue
            tag = f"{dep}->{op.op_id}"
            record = f"event:record:{tag}"
            wait = f"event:wait:{tag}"
            if tag not in emitted:
                emitted.add(tag)
                result.append(_event_op(op, record, (dep,),
                                        hardware.record_ms, "record"))
                result.append(_event_op(op, wait, (record,),
                                        hardware.wait_ms, "wait"))
            replacements[dep] = wait
        deps, metadata = event_control_rewrite(op, replacements, by_id) \
            if replacements else (op.deps, op.metadata)
        result.append(replace(op, deps=deps, metadata=metadata)
                      if deps != op.deps or metadata != op.metadata else op)
    return tuple(result)


def canonical_stream_move(plan: CanonicalPlan, node_ids: tuple[str, ...],
                          hardware: HardwareProfile, stream: str = "aux") -> tuple[OpSpec, ...]:
    if hardware.max_streams < 2:
        raise ValueError("stream_capability_insufficient")
    chosen = set(node_ids)
    for op in plan.ops:
        if op.op_id not in chosen:
            continue
        if (op.stream or "main") != "main":
            raise ValueError("stream_move_on_reserved_stream")
        node = plan.graph.ops.get(op.op_id)
        if node is None or node.collective or node.state_effect or node.capture_boundary \
                or node.resource == "host" \
                or op.family in {"prefetch", "native_fusion", "event", "release"}:
            raise ValueError(f"stream_move_preserves_barrier:{op.op_id}")
    moved = tuple(replace(op, stream=stream) if op.op_id in chosen else op
                  for op in plan.ops)
    return insert_stream_events(moved, hardware)


def timeline_by_op(result: Mapping[str, Any]) -> dict[str, dict]:
    return {str(row["op_id"]): row for row in result.get("timeline", ())}


def prefetch_completed_before(result: Mapping[str, Any], prefetch_id: str,
                              consumer_id: str) -> tuple[int, float, float]:
    rows = timeline_by_op(result)
    prefetch = rows.get(prefetch_id)
    consumer = rows.get(consumer_id)
    if prefetch is None or consumer is None:
        return 0, 0.0, 0.0
    complete = float(prefetch.get("complete_ms", float("inf")))
    ready = float(consumer.get("ready_ms", float("inf")))
    return (1 if complete <= ready + 1e-12 else 0), complete, ready


def build_cube_resources(hardware: HardwareProfile) -> CubeResources | None:
    return hardware.cube_capacities


def build_reduction_merge(*, bytes_: int, ops: float, tiles: int = 2) -> ReductionMerge:
    return ReductionMerge(bytes_, ops, tiles)
