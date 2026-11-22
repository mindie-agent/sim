"""Single internal lowering, causal replay, and projection spine for ``mindie_sim``.

[Role] Unify stack resolution, graph rewrite, governed operator-family plan
  dispatch, ``runtime.Engine`` replay, certified primal-dual 5-level ladder
  optimization (zero ``0.80x`` heuristic scaling), canonical ``graph_id``
  structure projection, SLA sizing, and topology/P-D sweeps behind one
  ``PredictionSpine`` class.
[Invariants] One fixed-step replay creates exactly one ``ExecutionBindings``
  and one ``CostModel`` via ``EvaluationContext.create``. Governed families
  execute directly through ``RecipeBook`` + ``FamilyDispatcher`` with zero
  adapter shims. ``graph_id`` matches ``mindie_sim.structure.export_op_structure``.
[Timeouts & Locks] Bounded per-step virtual replay honoring optional
  ``ReplayOptions.deadline_mono``; no locks or device contexts.
[Side-Effect Policy] Pure in-memory evaluation; never mutates caller cases.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import (
    LoweringRequest,
    SoftwareStackRegistry,
    default_stack_registry,
)
from .body_defaults import consume_target_declaration
from .evaluation import ACTIVE_DEADLINE_MONO, EvaluationContext
from .graph_analysis import graph_costs
from .mechanism import Geometry
from .operators.contracts import (
    OperatorPlanningContext,
    ResolvedOperator,
)
from .operators.registry import (
    FamilyDispatcher,
    FamilyRegistry,
    builtin_family_registry,
)
from .program import RecipeBook
from .results import (
    FusedPipelineDescriptor,
    JsonObject,
    OpOccurrence,
    OperatorSequenceRow,
    ReplayOptions,
    ServiceEvaluation,
    SlaCandidate,
    SlaEvaluation,
    SlaRequest,
    StepEvaluation,
    StepRequest,
    StructureExport,
    StructureRequest,
    SweepRequest,
    ThreeTierEvaluation,
    TopologyPoint,
    TopologySweepResult,
    WaveBreakdown,
    WorkItem,
    aligned_table_for,
    coerce_stream,
    freeze_json,
)
from .rewrite import ReplayRecipeBook, RewritePipeline, RewrittenStep
from .streams import MAIN_STREAM, StreamOwners, StreamPlan, bind_stream_owners

MS_PER_SECOND = 1000.0
NS_PER_MS = 1_000_000.0
DEFAULT_PD_TOTAL_DIES = 16
DEFAULT_OPTIMIZE_TRIALS = 16


@dataclass(frozen=True, slots=True)
class PreparedStep:
    """Lowered step carrying executable rank graphs, governed plans, and ``graph_id``."""

    graph: RewrittenStep
    case: Case
    operators: tuple[ResolvedOperator, ...]
    graph_id: str
    structure_payload: JsonObject | None = None


def _work_items(op: OpSpec) -> tuple[WorkItem, ...]:
    items: list[WorkItem] = []
    for dtype, flops in op.matrix_flops:
        if flops:
            items.append(WorkItem("cube", "flop", float(flops), str(dtype)))
    if not items and op.flops:
        items.append(WorkItem("cube", "flop", float(op.flops), op.dtype or None))
    for role, unit, amount in (
        ("hbm", "bytes", op.hbm_bytes),
        ("mte2", "bytes", op.mte2_bytes),
        ("mte3", "bytes", op.mte3_bytes),
        ("vector", "op", op.vector_ops),
        ("link", "bytes", op.link_bytes),
    ):
        if amount:
            items.append(WorkItem(role, unit, float(amount), op.dtype or None))
    meta = dict(op.metadata) if op.metadata else {}
    for role, amount in sorted(dict(meta.get("resource_work", {})).items()):
        if amount:
            items.append(WorkItem(str(role), "bytes", float(amount), op.dtype or None))
    return tuple(items)


def _parse_occurrence(node_id: str, fallback_replica: int, stage: int) -> OpOccurrence:
    if node_id.startswith("r") and ":" in node_id:
        prefix, op_id = node_id.split(":", 1)
        if prefix[1:].isdigit():
            return OpOccurrence(int(prefix[1:]), stage, None, op_id)
    return OpOccurrence(fallback_replica, stage, None, node_id)


def _wave_breakdown_from_report(report: Mapping[str, Any]) -> WaveBreakdown | None:
    raw = report.get("wave_breakdown")
    if not isinstance(raw, Mapping):
        return None
    return WaveBreakdown(
        waves=int(raw["waves"]),
        active_owners=int(raw["active_owners"]),
        total_owners=int(raw["total_owners"]),
        tail_owners=int(raw["tail_owners"]),
        cube_ms=float(raw["cube_ms"]),
        vector_ms=float(raw["vector_ms"]),
        memory_ms=float(raw["memory_ms"]),
        communication_ms=float(raw["communication_ms"]),
        control_ms=float(raw["control_ms"]),
        overlap_ratio=float(raw["overlap_ratio"]),
    )


class PredictionSpine:
    """Single internal orchestration spine for lowering, replay, and projection."""

    def __init__(
        self,
        stacks: SoftwareStackRegistry | None = None,
        rewrite: RewritePipeline | None = None,
        families: FamilyRegistry | None = None,
    ) -> None:
        self.stacks = default_stack_registry() if stacks is None else stacks
        self.families = builtin_family_registry() if families is None else families
        self.dispatcher = FamilyDispatcher(self.families)
        self.rewrite = (
            RewritePipeline(self.families.rewrite_rules())
            if rewrite is None else rewrite
        )

    def _stage_geometry(self, case: Case, replica: int, stage: int) -> Geometry:
        p = case.runtime.parallel
        width = p.tp * p.pcp
        start = (int(replica) * p.pp + int(stage)) * width
        dies = case.machine.dies[start : start + width]
        if not dies:
            dies = case.machine.dies[:1]
        return Geometry.from_devices(
            dies,
            links=case.machine.links,
            transfer_channels=case.machine.transfer_channels,
        )

    def _bind_stream_plans(
        self,
        case: Case,
        streams: tuple[StreamPlan, ...],
        active_replicas: frozenset[int],
    ) -> dict[tuple[int, int, str], StreamOwners]:
        if not streams:
            return {}
        p = case.runtime.parallel
        bound: dict[tuple[int, int, str], StreamOwners] = {}
        for plan in streams:
            rep = int(plan.rank)
            if rep not in active_replicas:
                raise ModelingError(f"stream_plan_unmatched_rank:{plan.rank}")
            for stage in range(p.pp):
                geom = self._stage_geometry(case, rep, stage)
                owners_map = bind_stream_owners(plan, geom.datapath)  # type: ignore[arg-type]
                for stream_id, owners in owners_map.items():
                    bound[(rep, stage, stream_id)] = owners
        return bound

    def _admission_memory_view(self, case: Case) -> Any:
        from .runtime.l2_residency import AdmissionMemoryView
        die_l2 = {int(die.rank): int(dict(die.buffers).get("l2", 0)) for die in case.machine.dies}
        if 0 not in die_l2 and case.machine.dies:
            die_l2[0] = int(dict(case.machine.dies[0].buffers).get("l2", 0))
        memory_access = {int(die.rank): die.memory_access for die in case.machine.dies}
        if case.machine.dies:
            memory_access.setdefault(0, case.machine.dies[0].memory_access)
        return AdmissionMemoryView(die_l2, memory_access_by_die=memory_access)

    def _resolve_governed_operators(
        self,
        case: Case,
        rewritten: RewrittenStep,
        book: RecipeBook,
        stream_owners: Mapping[tuple[int, int, str], StreamOwners] = (),  # type: ignore[assignment]
    ) -> tuple[ResolvedOperator, ...]:
        geometries: dict[tuple[int, int], Geometry] = {}
        resolved: list[ResolvedOperator] = []
        p = case.runtime.parallel
        width = p.tp * p.pcp
        owners_lookup = dict(stream_owners) if stream_owners else {}
        admission_view = self._admission_memory_view(case)
        for rank_ops in rewritten.executable:
            for op in rank_ops.ops:
                spec = self.families.match(op)
                if spec is None:
                    continue
                key = (rank_ops.replica, int(op.stage))
                geom = geometries.get(key)
                if geom is None:
                    geom = self._stage_geometry(case, rank_ops.replica, int(op.stage))
                    geometries[key] = geom
                rank_idx = (rank_ops.replica * p.pp + int(op.stage)) * width
                physical_rank = (
                    int(case.machine.dies[rank_idx].rank)
                    if rank_idx < len(case.machine.dies) else None
                )
                meta = dict(op.metadata) if op.metadata else {}
                req_abi = meta.get("requested_abi")
                owners = owners_lookup.get(
                    (rank_ops.replica, int(op.stage), op.stream or MAIN_STREAM)
                )
                ctx = OperatorPlanningContext(
                    occurrence=OpOccurrence(
                        replica=rank_ops.replica,
                        stage=int(op.stage),
                        rank=physical_rank,
                        op_id=op.op_id,
                    ),
                    geometry=geom,
                    datapath=geom.datapath,  # type: ignore[arg-type]
                    recipe_book=book,
                    requested_abi=str(req_abi) if req_abi else None,
                    stream_owners=owners,
                    admission_memory_view=admission_view,
                    **rewritten.sequence_context(rank_ops.replica, op),
                )
                resolved.append(self.dispatcher.resolve_plan(op, ctx))
        return tuple(resolved)

    def _compute_structure_payload(
        self,
        case: Case,
        rewritten: RewrittenStep,
        *,
        replicas: tuple[int, ...] | None = None,
        expand_calls: bool = True,
        native_evidence: bool = False,
        source: JsonObject | None = None,
    ) -> tuple[str, JsonObject]:
        from mindie_sim.structure import (
            export_op_structure,
            structure_inputs_from_graphs,
        )

        req = rewritten.base.request
        if (
            req.step is not None
            and rewritten.target_graphs is not None
            and not self.rewrite.rules
        ):
            phys_map = {entry.replica: entry.ops for entry in rewritten.executable}
            graphs_in, options = structure_inputs_from_graphs(
                rewritten.base.case,
                rewritten.target_graphs,
                phase=req.step.phase,
                tokens=req.step.tokens,
                context=req.step.context,
                sequences=req.step.sequences,
                scope=req.step.scope,
                source=dict(source) if source is not None else None,
                expand_calls=expand_calls,
                apply_body_defaults=req.apply_body_defaults,
                replicas=list(replicas) if replicas is not None else None,
                draft_prepare=req.step.draft_prepare,
                physical_graphs=phys_map,
            )
            report = export_op_structure(
                graphs_in, native_evidence=native_evidence, **options
            )
            return str(report["graph_id"]), freeze_json(report)  # type: ignore[return-value]

        graphs = {entry.replica: entry.ops for entry in rewritten.executable}
        if replicas is not None:
            graphs = {r: graphs[r] for r in replicas if r in graphs}
        phases = {ops[0].phase for ops in graphs.values() if ops and ops[0].phase}
        inferred_phase = req.step.phase if req.step is not None else (
            phases.pop() if len(phases) == 1 else "mixed" if phases else "decode"
        )
        exec_meta: dict[str, Any] = {
            "phase": inferred_phase,
            "scope": req.step.scope if req.step is not None else "declared",
        }
        draft_prepare = req.step.draft_prepare if req.step is not None else any(
            view.phase == "prefill" and not view.padding and view.draft_prepare
            for replica in graphs for view in rewritten.request_views.get(replica, ())
        )
        if draft_prepare:
            exec_meta["draft_prepare"] = True
        if rewritten.streams:
            exec_meta["streams"] = {
                str(plan.rank): list(plan.lane_ids()) for plan in rewritten.streams
            }
        report = export_op_structure(
            graphs,
            model=case.model.to_dict(),
            execution=exec_meta,
            kernels=case.runtime.kernels,
            expand_calls=expand_calls,
            native_evidence=native_evidence,
            source=dict(source) if source is not None else None,
        )
        return str(report["graph_id"]), freeze_json(report)  # type: ignore[return-value]

    def lower(
        self,
        case: Case,
        request: LoweringRequest,
        /,
        *,
        streams: tuple[StreamPlan, ...] = (),
    ) -> PreparedStep:
        """Lower ``case`` + ``request`` into a prepared step with canonical ``graph_id``."""
        base = self.stacks.resolve_base(case, request)
        rewritten = self.rewrite.apply(base, streams=streams)
        effective_case = (
            consume_target_declaration(base.case, rewritten.target_graphs)
            if rewritten.target_graphs is not None else base.case
        )
        active_replicas = frozenset(entry.replica for entry in rewritten.executable)
        bound_owners = self._bind_stream_plans(
            effective_case, rewritten.streams, active_replicas
        )
        book = RecipeBook.from_mapping(
            effective_case.runtime.kernels, family_registry=self.families
        )
        resolved_ops = self._resolve_governed_operators(
            effective_case, rewritten, book, bound_owners
        )
        graph_id, payload = self._compute_structure_payload(effective_case, rewritten)
        return PreparedStep(
            graph=rewritten,
            case=effective_case,
            operators=resolved_ops,
            graph_id=graph_id,
            structure_payload=payload,
        )

    def run_step(
        self,
        step: PreparedStep,
        replay: ReplayOptions = ReplayOptions(),
        /,
        *,
        primitives: Any = None,
        primitive_table: Any = None,
    ) -> StepEvaluation:
        """Execute ``step`` once on ``runtime.Engine`` using a single ``EvaluationContext``."""
        case = step.case
        book = ReplayRecipeBook.from_mapping(
            case.runtime.kernels, family_registry=self.families
        )
        book.graph = step.graph
        evaluation = EvaluationContext.create(
            case, book, primitives, primitive_table=primitive_table
        )
        graphs: Mapping[int, tuple[OpSpec, ...]] = {
            entry.replica: entry.ops for entry in step.graph.executable
        }
        req = step.graph.base.request
        step_req = req.step or StepRequest(phase="decode")
        external_inputs = {"analysis_input": 0.0} if any(
            "analysis_input" in op.deps
            for ops in graphs.values() for op in ops
        ) else None

        token = ACTIVE_DEADLINE_MONO.set(replay.deadline_mono)
        try:
            report = graph_costs(
                case,
                graphs,
                recipes=book,
                primitives=primitives,
                external_inputs=external_inputs,
                include_submission=replay.include_submission,
                isolated=replay.isolated,
                trace_limit=replay.event_limit,
                evaluation=evaluation,
                stream_plans=step.graph.streams or None,
                request_views=step.graph.request_views,
                padded_tokens=step_req.padded_tokens,
                sequence_layout=step_req.layout,
            )
        finally:
            ACTIVE_DEADLINE_MONO.reset(token)

        op_index: dict[tuple[int, str], OpSpec] = {
            (entry.replica, op.op_id): op
            for entry in step.graph.executable
            for op in entry.ops
        }
        resolved_by_occ = {
            (res.occurrence.replica, res.occurrence.op_id): res
            for res in step.operators
        }
        rows: list[OperatorSequenceRow] = []
        p = case.runtime.parallel
        width = p.tp * p.pcp
        timeline = report.get("timeline") or report.get("operators") or ()
        occ_by_node: dict[str, OpOccurrence] = {}
        row_occs: list[OpOccurrence] = []
        for row in timeline:
            replica = int(row.get("replica", 0))
            op_id = str(row.get("op_id", ""))
            spec = op_index.get((replica, op_id))
            stage = int(spec.stage) if spec is not None else int(row.get("stage", 0))
            res = resolved_by_occ.get((replica, op_id))
            rank_idx = (replica * p.pp + stage) * width
            physical_rank = (
                res.occurrence.rank
                if res is not None
                else (int(case.machine.dies[rank_idx].rank)
                      if 0 <= rank_idx < len(case.machine.dies) else None)
            )
            occ = OpOccurrence(replica=replica, stage=stage, rank=physical_rank, op_id=op_id)
            row_occs.append(occ)
            occ_by_node[f"r{replica}:{op_id}"] = occ
            if row.get("id"):
                occ_by_node[str(row["id"])] = occ
        for ordinal, (row, occ) in enumerate(zip(timeline, row_occs)):
            replica = occ.replica
            op_id = occ.op_id
            stage = occ.stage
            spec = op_index.get((replica, op_id))
            res = resolved_by_occ.get((replica, op_id))
            preds = tuple(
                occ_by_node.get(str(dep))
                or _parse_occurrence(str(dep), replica, stage)
                for dep in row.get("predecessors", ())
            )
            impl = (
                res.implementation
                if res is not None else str(row.get("program") or row.get("recipe") or "")
            )
            effective_spec = res.op if res is not None else spec
            ready = row.get("ready_ms")
            complete = row.get("complete_ms")
            iso = row.get("isolated_ms")
            stream_val = spec.stream if spec is not None else row.get("stream")
            rows.append(OperatorSequenceRow(
                occurrence=occ,
                ordinal=ordinal,
                family=str(row.get("family", "")),
                implementation=impl,
                predecessors=preds,
                stream=coerce_stream(stream_val),
                shape=tuple(int(x) for x in (effective_spec.local_shape if effective_spec is not None else ())),
                dtype=str(effective_spec.dtype if effective_spec is not None else ""),
                work=_work_items(effective_spec) if effective_spec is not None else (),
                ready_ms=float(ready) if ready is not None else None,
                start_ms=float(ready) if ready is not None else None,
                end_ms=float(complete) if complete is not None else None,
                isolated_ms=float(iso) if iso is not None else None,
            ))

        pipelines = tuple(
            res.descriptor for res in step.operators if res.descriptor is not None
        )
        accel = report.get("accel_status") or {}
        wave = _wave_breakdown_from_report(report)
        return StepEvaluation(
            graph_id=step.graph_id,
            request=step_req,
            tier=req.tier,
            span_ms=float(report["duration_ms"]),
            operators=tuple(rows),
            pipelines=pipelines,
            wave_breakdown=wave,
            accel_backend=str(
                report.get("engine_backend")
                or accel.get("accel_backend")
                or accel.get("backend")
                or ""
            ),
            native_steps=int(accel.get("native_steps", 0) or 0),
        )

    def optimize_ladder(
        self,
        case: Case,
        request: StepRequest,
        /,
        *,
        replay: ReplayOptions = ReplayOptions(),
        primitives: Any = None,
        primitive_table: Any = None,
        stack_id: str = "cann93_va_main",
        max_trials: int = DEFAULT_OPTIMIZE_TRIALS,
    ) -> Any:
        """Run the certified 5-level primal-dual optimization ladder on ``runtime.Engine``."""
        from mindie_sim.optimization.ladder import solve_five_level_ladder

        return solve_five_level_ladder(
            case,
            request,
            spine=self,
            replay=replay,
            primitives=primitives,
            primitive_table=primitive_table,
            stack_id=stack_id,
            max_trials=max_trials,
        )

    def evaluate_three_tiers(
        self,
        case: Case,
        request: StepRequest,
        /,
        *,
        replay: ReplayOptions = ReplayOptions(),
        primitives: Any = None,
        primitive_table: Any = None,
    ) -> ThreeTierEvaluation:
        """Evaluate ``unmerge``, ``mainline``, and constructive ``should_achieve`` via primal-dual ladder."""
        report = self.optimize_ladder(
            case,
            request,
            replay=replay,
            primitives=primitives,
            primitive_table=primitive_table,
        )
        unmerge_eval = report.unmerge_eval
        mainline_eval = report.mainline_eval
        should_eval = report.dsl_eval
        table = aligned_table_for(
            mainline_eval,
            tier_spans=(
                ("unmerge", unmerge_eval.span_ms),
                ("mainline", mainline_eval.span_ms),
                ("should_achieve", should_eval.span_ms),
            ),
        )
        return ThreeTierEvaluation(
            unmerge=unmerge_eval,
            mainline=mainline_eval,
            should_achieve=should_eval,
            certified_lower_bound_ms=float(report.ladder.l_star_lean_ns) / NS_PER_MS,
            aligned_table=table,
        )

    def run_service(
        self,
        case: Case,
        replay: ReplayOptions = ReplayOptions(),
        /,
        *,
        primitives: Any = None,
    ) -> ServiceEvaluation:
        """Evaluate a full serving workload on the shared simulation driver."""
        from .api import predict

        token = ACTIVE_DEADLINE_MONO.set(replay.deadline_mono)
        try:
            report = predict(
                case,
                primitives=primitives,
                event_limit=replay.event_limit,
            )
        finally:
            ACTIVE_DEADLINE_MONO.reset(token)
        metrics = report.get("metrics") or {}
        completed = int(metrics.get("served_requests", 0) or 0)
        emitted = int(metrics.get("output_tokens", 0) or 0)
        span_ms = float(metrics.get("makespan_ms", 0.0) or 0.0)
        return ServiceEvaluation(
            summary=freeze_json(report),  # type: ignore[arg-type]
            completed_requests=completed,
            emitted_tokens=emitted,
            span_ms=span_ms,
        )

    def solve_sla(self, case: Case, request: SlaRequest, /) -> SlaEvaluation:
        """Solve SLA-constrained concurrency sizing through the canonical solver."""
        from .efficiency import solve_sla_sizing

        report = solve_sla_sizing(
            case,
            sla_ms=request.latency_ms,
            context=request.step.context,
            tokens_per_seq=request.step.tokens,
            max_sequences_per_dp=request.max_sequences_per_dp,
            phase=request.step.phase,
            scope=request.step.scope,
            output_budget=request.output_budget,
        )
        candidates: list[SlaCandidate] = []
        hw_label = str(case.machine.sku or case.machine.name)
        for scenario in report.get("scenarios", ()):
            label = str(scenario.get("scenario_id") or scenario.get("id") or hw_label)
            seqs = int(scenario.get("b_feasible", 0) or 0)
            raw_span = scenario.get("step_ms_feasible")
            if raw_span is None:
                raw_span = scenario.get("step_ms_at_b1", 0.0)
            span = float(raw_span or 0.0)
            tps = float(scenario.get("tps_feasible", 0.0) or 0.0)
            fits = bool(scenario.get("capacity_feasible", False))
            candidates.append(SlaCandidate(
                hardware=label,
                sequences_per_dp=seqs,
                span_ms=span,
                throughput_tokens_s=tps,
                fits_memory=fits,
            ))
        feasible = [
            c for c, scen in zip(candidates, report.get("scenarios", ()))
            if bool(scen.get("feasible", False)) and c.sequences_per_dp > 0
        ]
        best = max(feasible, key=lambda c: c.throughput_tokens_s) if feasible else None
        return SlaEvaluation(best=best, candidates=tuple(candidates))

    def sweep_topology(
        self,
        case: Case,
        request: SweepRequest,
        /,
        *,
        decode_case: Case | None = None,
    ) -> TopologySweepResult:
        """Sweep topology or P/D disaggregation candidates and return Pareto frontier."""
        from mindie_sim.sweep import sweep_pd_disaggregation_ratios, sweep_topologies_iter
        from mindie_sim.sweep.engine import compute_pareto_frontier

        if request.mode == "pd":
            total_dies = (
                int(request.device_counts[0])
                if request.device_counts else max(len(case.machine.dies), DEFAULT_PD_TOTAL_DIES)
            )
            report = sweep_pd_disaggregation_ratios(
                case,
                decode_case or case,
                total_dies=total_dies,
                prompt_tokens=max(1, request.sla.step.context or request.sla.step.tokens),
                output_tokens=max(1, request.sla.output_budget),
                tpot_sla_ms=request.sla.latency_ms,
            )
            all_rows = report.get("candidates", ())
            front_rows = report.get("pareto_frontier", ())
            frontier: list[TopologyPoint] = []
            for row in front_rows:
                raw_span = row.get("effective_tpot_ms")
                if raw_span is None:
                    raw_span = row.get("decode_step_ms", 0.0)
                frontier.append(TopologyPoint(
                    device_count=int(row.get("used_dies", total_dies)),
                    parallel=(("prefill_dies", int(row.get("prefill_dies", 0))),),
                    decode_parallel=(("decode_dies", int(row.get("decode_dies", 0))),),
                    span_ms=float(raw_span or 0.0),
                    throughput_tokens_s=float(row.get("sustainable_tps", 0.0) or 0.0),
                    fits_memory=True,
                ))
            return TopologySweepResult(evaluated_count=len(all_rows), frontier=tuple(frontier))

        grid = list(request.device_counts) if request.device_counts else None
        evaluated_rows = list(sweep_topologies_iter(
            case,
            num_devices_grid=grid,
            sla_ms=request.sla.latency_ms,
            context=request.sla.step.context,
            tokens_per_seq=request.sla.step.tokens,
            output_budget=request.sla.output_budget,
            max_sequences_per_dp=request.sla.max_sequences_per_dp,
            phase=request.sla.step.phase,
            scope=request.sla.step.scope,
        ))
        compute_pareto_frontier(evaluated_rows)
        points: list[TopologyPoint] = []
        for row in evaluated_rows:
            if not row.get("pareto_optimal"):
                continue
            par = row.get("parallel") or {}
            raw_span = row.get("step_ms_feasible")
            if raw_span is None:
                raw_span = row.get("step_ms_at_b1", 0.0)
            points.append(TopologyPoint(
                device_count=int(row.get("num_devices", len(case.machine.dies))),
                parallel=tuple(sorted((str(k), int(v)) for k, v in par.items())),
                decode_parallel=None,
                span_ms=float(raw_span or 0.0),
                throughput_tokens_s=float(row.get("tps_feasible", 0.0) or 0.0),
                fits_memory=True,
            ))
        return TopologySweepResult(evaluated_count=len(evaluated_rows), frontier=tuple(points))

    def project_structure(
        self,
        step: PreparedStep,
        request: StructureRequest,
        /,
    ) -> StructureExport:
        """Project a canonical ``StructureExport`` sealed by ``graph_id``."""
        graph_id, payload = self._compute_structure_payload(
            step.case,
            step.graph,
            replicas=request.replicas,
            expand_calls=request.expand_calls,
            native_evidence=request.native_evidence,
            source=request.source,
        )
        return StructureExport(graph_id=graph_id, payload=payload)


@lru_cache(maxsize=1)
def default_spine() -> PredictionSpine:
    """Return the shared default ``PredictionSpine`` instance."""
    return PredictionSpine()


__all__ = [
    "PredictionSpine",
    "PreparedStep",
    "default_spine",
]
