"""Low-level op records and GEMM/vector work constructors.

[Role]
  Define the immutable OpSpec shape lineage record and the small constructors
  used by workload lowering and normative scheduling.

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
from typing import Any, Sequence

from .a3_types import align_up, dtype_bytes


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
        if include_work:
            payload.update({
                "flops": self.flops, "hbm_bytes": self.hbm_bytes,
                "mte2_bytes": self.mte2_bytes, "mte3_bytes": self.mte3_bytes,
                "vector_ops": self.vector_ops, "link_bytes": self.link_bytes,
                "link_scope": self.link_scope,
            })
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "routing_tokens": self.routing_tokens,
            "local_routing_tokens": self.local_routing_tokens,
            "global_scale": self.global_scale,
            "local_expert_counts": list(self.local_expert_counts),
            "local_experts": self.local_experts,
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

    def to_dict(self, *, include_ops: bool = True) -> dict[str, Any]:
        payload = {
            "iteration": self.plan.iteration,
            "mode": self.mode,
            "recipe": self.recipe,
            "valid_tokens": self.valid_tokens,
            "padded_tokens": self.padded_tokens,
            "padding_tokens": self.padded_tokens - self.valid_tokens,
            "graph_bucket": self.graph_bucket,
            "stage_count": self.stage_count,
            "routing": self.routing.to_dict() if self.routing is not None else None,
            "recipe_guards": [dict(item) for item in self.recipe_guards],
            "notes": list(self.notes),
            "requests": [view.to_dict() for view in self.plan.requests],
        }
        if include_ops:
            payload["ops"] = [op.to_dict() for op in self.ops]
        return payload


def _shape(tokens: int, parts: Sequence[int]) -> tuple[int, ...]:
    return (int(tokens), *(int(part) for part in parts))


def _op(**kwargs: Any) -> OpSpec:
    return OpSpec(**kwargs)


def _gemm_op(
    *, op_id: str, deps: tuple[str, ...], stage: int, layer: int, requests: tuple[str, ...],
    m: int, n: int, k: int, dtype: str, family: str, phase: str, recipe: str,
    semantic_m: int | None = None, semantic_n: int | None = None, semantic_k: int | None = None,
    notes: tuple[str, ...] = (), metadata: tuple[tuple[str, Any], ...] = (),
) -> OpSpec:
    bytes_per = dtype_bytes(dtype)
    m, n, k = max(1, m), max(1, n), max(1, k)
    flops = 2.0 * m * n * k
    in_bytes = (m * k + k * n) * bytes_per
    out_bytes = m * n * bytes_per
    return _op(
        op_id=op_id, family=family, phase=phase, stage=stage, layer=layer,
        requests=requests,
        semantic_shape=(semantic_m or m, semantic_n or n, semantic_k or k),
        local_shape=(m, max(1, n // 1), k),
        padded_shape=(m, n, k), launch_shape=(m, n, k),
        flops=flops, hbm_bytes=in_bytes + out_bytes,
        mte2_bytes=in_bytes, mte3_bytes=out_bytes,
        deps=deps, recipe=recipe,
        notes=notes + ("gemm_formula=2*M*N*K",),
        metadata=metadata, dtype=dtype,
    )


def _vec_op(
    *, op_id: str, deps: tuple[str, ...], stage: int, layer: int, requests: tuple[str, ...],
    elements: int, ops_per_element: float, dtype: str, family: str, phase: str,
    recipe: str, read_write: float = 2.0, shape: tuple[int, ...] | None = None,
) -> OpSpec:
    bytes_per = dtype_bytes(dtype)
    elements = max(1, int(elements))
    shape = shape or (elements,)
    return _op(
        op_id=op_id, family=family, phase=phase, stage=stage, layer=layer,
        requests=requests, semantic_shape=shape, local_shape=shape,
        padded_shape=(align_up(elements, 256),), launch_shape=(align_up(elements, 256),),
        vector_ops=float(elements) * ops_per_element,
        hbm_bytes=float(elements) * read_write * bytes_per,
        mte2_bytes=float(elements) * bytes_per,
        mte3_bytes=float(elements) * bytes_per,
        deps=deps, recipe=recipe,
        notes=("vector_work=elements*ops_per_element",), dtype=dtype,
    )



__all__ = ["OpSpec", "RoutingStats", "IterationWork"]
