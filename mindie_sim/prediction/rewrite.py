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

from dataclasses import dataclass
from typing import Any, Protocol

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import RankOps, ResolvedBase
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
            from .native_fusion import lower_native_fusions
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
            )
            effective_case = consume_target_declaration(base.case, graphs)
            logical = tuple(
                RankOps(replica=int(replica), ops=tuple(ops))
                for replica, ops in sorted(graphs.items())
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
            )

        if request.declared_ops:
            logical = tuple(request.declared_ops)
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

        from .attention import AttentionPolicy
        from .frontend import Frontend

        frontend = Frontend(
            attention_policy=AttentionPolicy(**base.case.runtime.attention_policy)
        )
        logical_rows: list[RankOps] = []
        executable_rows: list[RankOps] = []
        for replica, iteration in request.iterations:
            work = frontend(
                iteration, base.case.model, base.case.runtime, base.case.machine,
                replica=int(replica),
            )
            ops = tuple(work.ops)
            logical_rows.append(RankOps(replica=int(replica), ops=ops))
            executable_rows.append(RankOps(
                replica=int(replica),
                ops=self._run_rules(ops, base, stream_by_rank.get(int(replica))),
            ))
        return RewrittenStep(
            base=base,
            logical=tuple(logical_rows),
            executable=tuple(executable_rows),
            streams=tuple(streams),
        )


__all__ = [
    "RewritePipeline",
    "RewriteRule",
    "RewrittenStep",
]
