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

        from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
        from .frontend import Frontend

        frontend = Frontend(
            attention_policy=AttentionPolicy(**base.case.runtime.attention_policy)
        )
        logical_rows: list[RankOps] = []
        executable_rows: list[RankOps] = []
        request_views: dict[int, tuple[RequestView, ...]] = {}
        padded_tokens: dict[int, int] = {}
        known_replicas = {int(replica) for replica, _ in request.iterations}
        unmatched = set(stream_by_rank) - known_replicas
        if unmatched:
            raise ModelingError(
                "stream_plan_unmatched_rank:"
                + ",".join(str(r) for r in sorted(unmatched))
            )
        for replica, iteration in request.iterations:
            replica = int(replica)
            work = frontend(
                iteration, base.case.model, base.case.runtime, base.case.machine,
                replica=replica,
            )
            request_views[replica] = work.plan.requests
            padded_tokens[replica] = work.padded_tokens
            ops = tuple(work.ops)
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
