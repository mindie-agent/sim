"""Low-level work and iteration records.

[Role]
  Define immutable shape lineage and work records used by workload lowering.

[Invariants]
  * Every op keeps semantic, local, padded and launch shapes distinct.
  * GEMM work is 2*M*N*K; vector work is elements*ops_per_element.
  * No output token is added by a work constructor.

[Timeouts & Locks]
  Pure arithmetic.  No I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from .workload import IterationPlan


@dataclass(frozen=True)
class OpSpec:
    op_id: str
    family: str
    phase: str
    stage: int
    layer: int
    requests: tuple[str, ...]
    semantic_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    padded_shape: tuple[int, ...]
    launch_shape: tuple[int, ...]
    flops: float = 0.0
    hbm_bytes: float = 0.0
    mte2_bytes: float = 0.0
    mte3_bytes: float = 0.0
    vector_ops: float = 0.0
    link_bytes: float = 0.0
    link_scope: str = "d2d"
    deps: tuple[str, ...] = ()
    recipe: str = ""
    coverage: str = "decomposed"
    metadata: tuple[tuple[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    dtype: str = ""
    layout: str = "contiguous"
    #: Declared return quantities beyond the primary output (e.g. the
    #: ``softmax_max``/``softmax_sum`` partials an LSE merge consumes).  An op
    #: that omits this declares no mergeable partial; a merge consumer must
    #: name the quantities it reads or it is not a valid composition.
    returns: tuple[str, ...] = ()
    #: Declared device stream for this op (``main``/``aux``), when the lowering
    #: can state it from the runtime (V4/V4.1 attention path).  Empty means the
    #: default main stream is used, including its ordering edges.  Streams
    #: are ordering relations, never capacity-1 resources.
    stream: str = ""
    #: Fused matrix work uses separate precision rates on one physical resource.
    #: Empty retains the single-dtype work declaration.
    matrix_flops: tuple[tuple[str, float], ...] = ()
    #: A source call expands into kernels before pricing and host submission.
    #: Both source boundaries require an exact planner; empty is semantic work.
    source_boundary: Literal["", "kernel", "call"] = ""

    @property
    def plan_required(self) -> bool:
        return bool(self.source_boundary)

    def to_dict(self, *, include_work: bool = True) -> dict[str, Any]:
        payload = {
            "op_id": self.op_id, "family": self.family, "phase": self.phase,
            "stage": self.stage, "layer": self.layer,
            "requests": list(self.requests),
            "semantic_shape": list(self.semantic_shape),
            "local_shape": list(self.local_shape),
            "padded_shape": list(self.padded_shape),
            "launch_shape": list(self.launch_shape),
            "recipe": self.recipe, "deps": list(self.deps),
            "dtype": self.dtype, "layout": self.layout,
            "coverage": self.coverage, "notes": list(self.notes),
            "metadata": dict(self.metadata),
        }
        if self.returns:
            payload["returns"] = list(self.returns)
        if self.plan_required:
            payload["source_boundary"] = self.source_boundary
        if self.stream:
            payload["stream"] = self.stream
        if include_work:
            payload.update({
                "flops": self.flops, "hbm_bytes": self.hbm_bytes,
                "mte2_bytes": self.mte2_bytes, "mte3_bytes": self.mte3_bytes,
                "vector_ops": self.vector_ops, "link_bytes": self.link_bytes,
                "link_scope": self.link_scope,
            })
            if self.matrix_flops:
                payload["matrix_flops"] = dict(self.matrix_flops)
        return payload


@dataclass(frozen=True)
class RoutingStats:
    routing_tokens: float
    local_experts: int
    slowest_rank_tokens: float
    local_padded_tokens: float
    capacity_per_expert: float
    dropped_tokens: float
    distribution: str
    local_routing_tokens: float = 0.0
    global_scale: float = 1.0
    local_expert_counts: tuple[int, ...] = ()
    #: Logical (unpadded) rows per local expert.  ``local_expert_counts``
    #: retains the declared aligned implementation view;
    #: the served grouped shape uses this field so the op's M is the real
    #: routed row count, not the graph-bucket or per-expert padding cap.
    local_expert_logical_counts: tuple[int, ...] = ()
    #: Expected number of this rank's local experts that receive at least one
    #: routed token in the iteration.  A decode step with one token and top-k=6
    #: does not read all 96 local expert weight matrices; a grouped GEMM that
    #: skips the zero-token experts streams only the active ones.  Derived from
    #: the declared routing distribution and the EP group's routed-token volume,
    #: never from a measured latency.
    active_local_experts: float = 0.0
    #: Binomial inclusion moments: expected active experts and expected
    #: aligned rows.  These are statistical work identities, not integer shapes.
    expected_active_experts: float = 0.0
    expected_padded_rows: float | None = None
    expected_padded_rows_status: str = "unknown"
    #: Global expert ids of the rank whose counts are reported, in local order.
    local_expert_ids: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_tokens": self.routing_tokens,
            "local_routing_tokens": self.local_routing_tokens,
            "global_scale": self.global_scale,
            "local_expert_counts": list(self.local_expert_counts),
            **({"local_expert_logical_counts": list(self.local_expert_logical_counts)}
               if self.local_expert_logical_counts else {}),
            **({"active_local_experts": self.active_local_experts}
               if self.active_local_experts > 0.0 else {}),
            "local_experts": self.local_experts,
            **({"local_expert_ids": list(self.local_expert_ids)}
               if self.local_expert_ids else {}),
            "slowest_rank_tokens": self.slowest_rank_tokens,
            "local_padded_tokens": self.local_padded_tokens,
            "capacity_per_expert": self.capacity_per_expert,
            "dropped_tokens": self.dropped_tokens,
            "distribution": self.distribution,
        }


@dataclass(frozen=True)
class IterationWork:
    ops: tuple[OpSpec, ...]
    plan: IterationPlan
    stage_count: int
    valid_tokens: int
    padded_tokens: int
    graph_bucket: int | None
    mode: str
    recipe: str
    routing: RoutingStats | None = None
    recipe_guards: tuple[dict[str, Any], ...] = ()
    notes: tuple[str, ...] = ()
    #: Replay scope for this iteration: ``none``, ``full`` or ``piecewise``.
    #: PIECEWISE captures the non-attention subgraph and leaves attention/split
    #: ops eager; it is never a whole-iteration FULL claim.
    replay_scope: str = ""
    #: Declared splitting_ops for PIECEWISE; ``None`` uses the target default
    #: attention boundary.
    splitting_ops: tuple[str, ...] | None = None
    #: Rank-local token count when the execution plan splits the batch (PCP).
    #: ``valid_tokens`` stays the client-visible global count so throughput and
    #: padding accounting are not rewritten by a rank-local view.
    local_tokens: int = 0
    #: Declared plan/work record (strategy, degrees, pairs, local rows).  Kept
    #: separate from the prediction plan so an execution trace can be audited
    #: without re-deriving the split.
    cp_work: tuple[tuple[str, Any], ...] = ()
    #: Comm buffers/collectives this iteration's plan requires.
    cp_comm: tuple[tuple[str, Any], ...] = ()
    implementation_selections: tuple[dict[str, Any], ...] = ()

    def to_dict(self, *, include_ops: bool = True) -> dict[str, Any]:
        payload = {
            "iteration": self.plan.iteration,
            "mode": self.mode,
            "recipe": self.recipe,
            "valid_tokens": self.valid_tokens,
            "padded_tokens": self.padded_tokens,
            "padding_tokens": self.padded_tokens - self.valid_tokens,
            "graph_bucket": self.graph_bucket,
            "replay_scope": self.replay_scope,
            "stage_count": self.stage_count,
            "routing": self.routing.to_dict() if self.routing is not None else None,
            "recipe_guards": [dict(item) for item in self.recipe_guards],
            "notes": list(self.notes),
            "requests": [view.to_dict() for view in self.plan.requests],
            "implementation_selections": list(self.implementation_selections),
        }
        if self.local_tokens:
            payload["local_tokens"] = int(self.local_tokens)
        if self.cp_work:
            payload["cp_work"] = dict(self.cp_work)
        if self.cp_comm:
            payload["cp_comm"] = [dict(self.cp_comm)]
        if include_ops:
            payload["ops"] = [op.to_dict() for op in self.ops]
        return payload


__all__ = ["OpSpec", "RoutingStats", "IterationWork"]
