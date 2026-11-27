"""Graph rewrite pipeline and indexer rewrite rules.

[Role] Own RewritePipeline orchestration, sharding/fusion pass registration,
  and sparse-indexer graph rewrite rules.
[Invariants] Preserves SSA value lineage and pass priority determinism.
[Timeouts & Locks] Pure in-memory graph rewriting; no locks.
[Side-Effect Policy] Pure functional transforms."""
from __future__ import annotations

from dataclasses import replace

from .implementations import rewire_replacements
from .rewrite_mhc_rules import merge_meta, users_map


NO_WORK = 0
LOCAL_REDUCTION_SHARDS = 1
NO_PORTS: dict = {}
COMPLETE_LOCAL_REDUCTION = "complete_local"
SCORE_RECIPES = frozenset({"index_scores", "batch_prefix_indexer"})


def online_indexer_topk(ops, source_dtype):
    """Opt-in algorithm; not an assertion about a CANN binary's workspace."""
    del source_dtype
    by_id, users = {item.op_id: item for item in ops}, users_map(ops)
    replacements, aliases, removed = {}, {}, set()
    for topk in ops:
        if (topk.recipe != "index_topk" or topk.plan_required
                or len(topk.deps) != LOCAL_REDUCTION_SHARDS):
            continue
        score = by_id.get(topk.deps[NO_WORK])
        if (score is None or score.plan_required
                or score.recipe not in SCORE_RECIPES
                or users[score.op_id] != [topk.op_id]
                or score.stage != topk.stage or score.stream != topk.stream):
            continue
        metadata = dict(score.metadata)
        if (metadata.get("index_score_shards") != LOCAL_REDUCTION_SHARDS
                or metadata.get("index_context_partitions") != LOCAL_REDUCTION_SHARDS
                or metadata.get("index_candidate_filter")
                or metadata.get("index_head_reduction") != COMPLETE_LOCAL_REDUCTION
                or metadata.get("graph_segment")
                != dict(topk.metadata).get("graph_segment")):
            continue
        score_bytes = float(score.mte3_bytes)
        if score_bytes != float(topk.mte2_bytes):
            continue
        hbm = score.hbm_bytes + topk.hbm_bytes - score_bytes - topk.mte2_bytes
        if hbm < NO_WORK:
            continue
        matrix = {}
        for item in (score, topk):
            work = dict(item.matrix_flops) or ({item.dtype: item.flops}
                                               if item.flops else {})
            for dtype, amount in work.items():
                matrix[dtype] = matrix.get(dtype, NO_WORK) + amount
        fused = replace(
            topk, deps=score.deps, family="indexer", recipe="index_online_topk",
            dtype=score.dtype, flops=score.flops + topk.flops,
            matrix_flops=tuple(sorted(matrix.items())),
            vector_ops=score.vector_ops + topk.vector_ops,
            hbm_bytes=hbm, mte2_bytes=score.mte2_bytes,
            mte3_bytes=topk.mte3_bytes,
            metadata=merge_meta(score, topk, extra=(
                ("score_materialization", "analytical_online_ub"),
                ("native_binary_claimed", False),
                ("output_format", "int32"),
                ("removed_score_roundtrip_bytes", score_bytes + topk.mte2_bytes),
            )))
        replacements[topk.op_id] = fused
        aliases[score.op_id] = topk.op_id
        removed.add(score.op_id)
    if not replacements:
        return list(ops)
    return rewire_replacements(
        ops, replacements=replacements,
        aliases={**{key: key for key in replacements}, **aliases},
        removed=removed, ports=NO_PORTS)


from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView
from .mechanism import OperatorPlan
from .program import RecipeBook
from .results import RankOps, ResolvedBase
from .streams import StreamPlan, lower_stream_edges


class RewriteRule(Protocol):
    """Single deterministic operator-graph transformation pass."""

    @property
    def name(self) -> str: ...

    def apply(
        self,
        ops: tuple[OpSpec, ...],
        base: ResolvedBase,
        /,
    ) -> tuple[OpSpec, ...]: ...


@dataclass(frozen=True, slots=True)
class RewrittenStep:
    """Logical and executable per-rank operator sequences after rewrite passes."""

    base: ResolvedBase
    logical: tuple[RankOps, ...]
    executable: tuple[RankOps, ...]
    streams: tuple[StreamPlan, ...]
    target_graphs: Any = None
    request_views: Mapping[int, tuple[RequestView, ...]] = field(default_factory=dict)
    padded_tokens: Mapping[int, int] = field(default_factory=dict)

    def sequence_context(self, replica: int, op: OpSpec) -> dict[str, Any]:
        """Read the replica batch and scope constituent operators to their rows."""
        meta = dict(op.metadata)
        step = self.base.request.step
        if "request_views" in meta:
            views = tuple(
                v if isinstance(v, RequestView) else RequestView(**dict(v))
                for v in meta["request_views"]
            )
        else:
            views = self.request_views.get(replica, ())
            if op.requests:
                ids = frozenset(op.requests)
                if any(v.request_id in ids for v in views if not v.padding):
                    views = tuple(v for v in views if v.padding or v.request_id in ids)
        return {
            "requests": views,
            "padded_tokens": int(meta.get("padded_tokens", self.padded_tokens.get(replica, 0))),
            "layout": str(meta.get("sequence_layout") or (step.layout if step else "") or op.layout),
        }


class ReplayRecipeBook(RecipeBook):
    """Bind sequence geometry by occurrence during graph replay and isolated pricing."""

    graph: RewrittenStep

    def prepare(self, op, geometry):
        # Deferred engine tasks may lower after another replica was submitted.
        # Bind by the occurrence's replica at the existing recipe boundary.
        if geometry.replica not in self.graph.request_views and "request_views" not in dict(op.metadata):
            return super().prepare(op, geometry)
        sequence = self.graph.sequence_context(geometry.replica, op)
        prepared, recipe = super().prepare(op, replace(geometry, **sequence))
        if isinstance(recipe, OperatorPlan):
            def bind(context):
                return recipe.bind_context(replace(context, **sequence))

            return prepared, replace(recipe, context_binder=bind)
        return prepared, recipe


class RewritePipeline:
    """Ordered sequence of ``RewriteRule`` passes plus N-stream edge lowering."""

    __slots__ = ("rules",)

    def __init__(self, rules: tuple[RewriteRule, ...] = (), /) -> None:
        self.rules = tuple(rules)

    def _run_rules(
        self,
        ops: tuple[OpSpec, ...],
        base: ResolvedBase,
        plan: StreamPlan | None,
    ) -> tuple[OpSpec, ...]:
        current = tuple(ops)
        for rule in self.rules:
            current = tuple(rule.apply(current, base))
        if plan is not None:
            current = lower_stream_edges(current, plan)
        return current

    def apply(
        self,
        base: ResolvedBase,
        /,
        *,
        streams: tuple[StreamPlan, ...] = (),
    ) -> RewrittenStep:
        """Emit logical and executable graphs for ``base`` in one pass."""
        request = base.request
        stream_by_rank = {int(plan.rank): plan for plan in streams}
        if len(stream_by_rank) != len(streams):
            raise ModelingError("stream_duplicate_rank_plan")

        if request.step is not None:
            from .body_defaults import consume_target_declaration
            from mindie_sim.prediction.operators.hyper_connections.native_fusion import lower_native_fusions
            from .operator_analysis import target_graphs

            step = request.step
            graphs = target_graphs(
                base.case,
                phase=step.phase,
                tokens=step.tokens,
                context=step.context,
                sequences=step.sequences,
                scope=step.scope,
                media_tokens_per_sequence=step.media_tokens_per_sequence,
                resource_fusion=False,
                draft_prepare=step.draft_prepare,
                apply_body_defaults=request.apply_body_defaults,
                requests=step.requests,
                padded_tokens=step.padded_tokens,
                layout=step.layout,
            )
            effective_case = consume_target_declaration(base.case, graphs)
            logical = tuple(
                RankOps(replica=int(replica), ops=tuple(ops))
                for replica, ops in sorted(graphs.items())
            )
            unmatched = set(stream_by_rank) - {entry.replica for entry in logical}
            if unmatched:
                raise ModelingError(
                    "stream_plan_unmatched_rank:"
                    + ",".join(str(r) for r in sorted(unmatched))
                )
            executable = tuple(
                RankOps(
                    replica=entry.replica,
                    ops=self._run_rules(
                        tuple(lower_native_fusions(entry.ops, effective_case.runtime)),
                        base,
                        stream_by_rank.get(entry.replica),
                    ),
                )
                for entry in logical
            )
            return RewrittenStep(
                base=base,
                logical=logical,
                executable=executable,
                streams=tuple(streams),
                target_graphs=graphs,
                request_views=graphs.request_views,
                padded_tokens={
                    replica: step.padded_tokens or max(
                        (dict(op.metadata).get("graph_physical_query_rows", 0) for op in ops),
                        default=0,
                    )
                    for replica, ops in graphs.items()
                },
            )

        if request.declared_ops:
            logical = tuple(request.declared_ops)
            unmatched = set(stream_by_rank) - {entry.replica for entry in logical}
            if unmatched:
                raise ModelingError(
                    "stream_plan_unmatched_rank:"
                    + ",".join(str(r) for r in sorted(unmatched))
                )
            executable = tuple(
                RankOps(
                    replica=entry.replica,
                    ops=self._run_rules(entry.ops, base, stream_by_rank.get(entry.replica)),
                )
                for entry in logical
            )
            return RewrittenStep(
                base=base,
                logical=logical,
                executable=executable,
                streams=tuple(streams),
            )

        return self._apply_iterations(base, request, stream_by_rank, streams)

    def _apply_iterations(
        self,
        base: ResolvedBase,
        request: Any,
        stream_by_rank: Mapping[int, StreamPlan],
        streams: tuple[StreamPlan, ...],
    ) -> RewrittenStep:
        from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
        from mindie_sim.schema.workload import IterationPlan, selected_mode
        from .execution.cohort_traffic import bind_cohort_traffic
        from .execution.dispatch import cohorts as topology_cohorts
        from .execution.rank_ledger import ep_replica_group
        from .frontend import Frontend
        from .frontend.body import _verify_query_len
        from .operator_analysis import (
            _ep_routing_geometry_by_replica,
            graph_padded_idle,
            synchronize_moe_allgather_cohort,
        )
        from .sharding.local_tp import lower_local_projection_tp_cohort

        frontend = Frontend(
            attention_policy=AttentionPolicy(**base.case.runtime.attention_policy)
        )
        iterations = tuple((int(replica), iteration) for replica, iteration in request.iterations)
        known_replicas = {replica for replica, _ in iterations}
        if len(known_replicas) != len(iterations):
            raise ModelingError("iteration_duplicate_replica")
        unmatched = set(stream_by_rank) - known_replicas
        if unmatched:
            raise ModelingError(
                "stream_plan_unmatched_rank:"
                + ",".join(str(r) for r in sorted(unmatched))
            )
        by_replica = dict(iterations)
        topo_groups = [
            tuple(int(r) for r in group)
            for group in topology_cohorts(base.case.runtime.parallel, base.case.machine)
        ]
        assigned: set[int] = set()
        iteration_groups: list[tuple[int, ...]] = []
        for group in topo_groups:
            present = tuple(r for r in group if r in by_replica)
            if present:
                iteration_groups.append(present)
                assigned.update(present)
        for replica, _ in iterations:
            if replica not in assigned:
                iteration_groups.append((replica,))
                assigned.add(replica)

        has_cohort_realization = any(
            isinstance(item, Mapping) and bool(item.get("cohort", False))
            for item in (base.case.runtime.routing.get("realizations") or ())
        )
        request_views: dict[int, tuple[RequestView, ...]] = {}
        padded_tokens: dict[int, int] = {}
        raw_graphs: dict[int, tuple[OpSpec, ...]] = {}
        for group in iteration_groups:
            if len(group) == 1:
                replica = group[0]
                work = frontend(
                    by_replica[replica], base.case.model, base.case.runtime, base.case.machine,
                    replica=replica,
                )
                request_views[replica] = work.plan.requests
                padded_tokens[replica] = work.padded_tokens
                raw_graphs[replica] = tuple(work.ops)
                continue
            logical_views = {r: tuple(by_replica[r].requests) for r in group}
            routing_count_views = {
                r: tuple(
                    replace(v, q_len=_verify_query_len(v), draft_proposal_tokens=0)
                    for v in logical_views[r]
                )
                for r in group
            }
            active_views = tuple(
                v for r in group for v in logical_views[r] if not getattr(v, "padding", False)
            )
            cohort_phase = (
                selected_mode(IterationPlan(active_views))
                if active_views else "idle"
            )
            counts = {r: len(logical_views[r]) for r in group}
            pad_idle = graph_padded_idle(base.case, cohort_phase, counts)
            default_context = next((v.context for v in active_views), 0)
            synced_counts, allgather_padded = synchronize_moe_allgather_cohort(
                base.case, cohort_phase, routing_count_views,
                pad_idle=pad_idle, default_context=default_context, prefix="iteration",
            )
            active_rows = {
                r: sum(v.q_len for v in routing_count_views[r])
                for r in group
            }
            global_views = tuple(v for r in group for v in logical_views[r])
            ep_routing_by_r = _ep_routing_geometry_by_replica(
                base.case, group, routing_count_views, active_rows, allgather_padded,
            )
            cohort_graphs: dict[int, tuple[OpSpec, ...]] = {}
            for replica in group:
                extra_padding = synced_counts[replica][len(logical_views[replica]):]
                plan_in = (
                    replace(by_replica[replica], requests=logical_views[replica] + extra_padding)
                    if extra_padding else by_replica[replica]
                )
                cohort_tokens, source_rows, routing_tokens = ep_routing_by_r[replica]
                cohort_kwargs: dict[str, Any] = {
                    "global_tokens": cohort_tokens,
                    "global_views": global_views,
                    "source_rows_by_replica": source_rows,
                    "routing_global_tokens": routing_tokens,
                    "cohort_views_by_replica": logical_views,
                }
                if has_cohort_realization:
                    ep_reps = ep_replica_group(
                        base.case.runtime.parallel, base.case.machine, group, replica,
                    )
                    cohort_kwargs["routing_views"] = tuple(
                        v for r in ep_reps for v in logical_views[r]
                    )
                work = frontend(
                    plan_in, base.case.model, base.case.runtime, base.case.machine,
                    replica=replica, **cohort_kwargs,
                )
                request_views[replica] = work.plan.requests
                padded_tokens[replica] = work.padded_tokens
                cohort_graphs[replica] = tuple(work.ops)
            cohort_graphs = bind_cohort_traffic(base.case, cohort_graphs)
            cohort_graphs = lower_local_projection_tp_cohort(
                cohort_graphs,
                model=base.case.model,
                runtime=base.case.runtime,
                machine=base.case.machine,
            )
            raw_graphs.update(cohort_graphs)
        logical_rows: list[RankOps] = []
        executable_rows: list[RankOps] = []
        for replica, _ in iterations:
            ops = tuple(raw_graphs[replica])
            logical_rows.append(RankOps(replica=replica, ops=ops))
            executable_rows.append(RankOps(
                replica=replica,
                ops=self._run_rules(ops, base, stream_by_rank.get(replica)),
            ))
        return RewrittenStep(
            base=base,
            logical=tuple(logical_rows),
            executable=tuple(executable_rows),
            streams=tuple(streams),
            request_views=request_views,
            padded_tokens=padded_tokens,
        )


__all__ = [
    "RewritePipeline",
    "RewriteRule",
    "RewrittenStep",
]
