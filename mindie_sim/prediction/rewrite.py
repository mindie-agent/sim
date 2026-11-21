"""Single rewrite pipeline producing logical and executable rank operator graphs.

[Role] Accept a ``ResolvedBase`` (fixed step, scheduled iterations, or explicit
  ``declared_ops``), run ordered ``RewriteRule`` passes and optional general
  N-stream ``StreamPlan`` edge lowering once, and return an immutable
  ``RewrittenStep``.
[Invariants] Fixed-step lowering invokes graph generation exactly once per
  rewrite. Stream plans are applied per rank through ``lower_stream_edges`` so
  transitive cross-stream dependencies and acyclicity are verified before
  execution.
[Timeouts & Locks] Finite single-pass graph lowering; no I/O or locks.
[Side-Effect Policy] Returns new immutable ``RewrittenStep`` records.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.stack_registry import RankOps, ResolvedBase
from .mechanism import OperatorPlan
from .program import RecipeBook
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
