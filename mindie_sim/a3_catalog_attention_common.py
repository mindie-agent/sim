"""Common attention geometry/dependency records for catalog lowering.

[Role]
  Exact causal/compressed pair oracles, collective helper, legal TP widths and
  the shared AttentionResult record.

[Invariants]
  * Pair counts are exact integer sums, not fractional averages at boundaries.
  * MLA q_b/kv_b use local heads once; grouped output keeps its group axis.
  * Collective payloads are logical activation bytes.

[Timeouts & Locks]
  Pure arithmetic; no I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

from .a3_catalog_cache import local_kv_heads
from .a3_catalog_ops import comm_op
from .a3_model_types import LayerPlan, ModelSpec
from .a3_ops import OpSpec
from .a3_types import ParallelSpec, dtype_bytes


@dataclass
class AttentionResult:
    ops: list[OpSpec] = field(default_factory=list)
    last: str = ""
    output_ids: tuple[str, ...] = ()
    cache_store_id: str | None = None
    topk_id: str | None = None
    selected_rows: int = 0
    index_read_bytes: float = 0.0
    cache_write_rows: int = 0
    attention_outputs: tuple[str, ...] = ()
    index_store_id: str | None = None
    candidate_id: str | None = None


def _triangular(q: int, window: int) -> float:
    n = min(max(0, q), max(0, window))
    return n * (n + 1) / 2.0


def _causal_pairs(q: int, context: int, window: int) -> float:
    """Exact sum_{i=1..q} min(context+i, window) (window<=0 means no window)."""
    q = max(1, int(q))
    context = max(0, int(context))
    if window <= 0:
        return q * context + q * (q + 1) / 2.0
    window = max(1, int(window))
    if context >= window:
        return float(q * window)
    full = min(q, window - context)
    partial = full * context + full * (full + 1) / 2.0
    return float(partial + (q - full) * window)


def _floor_sum(n: int, m: int, a: int, b: int) -> int:
    """sum_{i=0..n-1} floor((a*i+b)/m), non-negative arguments."""
    n, m, a, b = int(n), int(m), int(a), int(b)
    result = 0
    while True:
        if a >= m:
            result += (n - 1) * n * (a // m) // 2
            a %= m
        if b >= m:
            result += n * (b // m)
            b %= m
        y_max = a * n + b
        if y_max < m:
            return result
        n = y_max // m
        b = y_max % m
        m, a = a, m


def _long_selected_pairs(q: int, context: int, ratio: int, topk: int) -> float:
    """Exact sum_{i=1..q} min((context+i)//ratio, topk); topk<=0 means no cap."""
    q = max(1, int(q))
    context = max(0, int(context))
    ratio = int(ratio)
    if ratio <= 0:
        return 0.0
    topk = int(topk)
    if topk <= 0:
        return float(_floor_sum(q, ratio, 1, context + 1))
    threshold = topk * ratio - context
    unsaturated = min(q, max(0, threshold - 1))
    floor_sum = _floor_sum(unsaturated, ratio, 1, context + 1)
    return float(floor_sum + (q - unsaturated) * topk)


def _requests(views: Sequence[Any]) -> tuple[str, ...]:
    return tuple(view.request_id for view in views)


def _allreduce(
    *, prefix: str, deps: tuple[str, ...], stage: int, layer: int,
    requests: tuple[str, ...], model: ModelSpec, parallel: ParallelSpec,
    mode: str, tokens: int, suffix: str = "attn",
) -> OpSpec:
    h = model.hidden_size
    return comm_op(
        op_id=f"{prefix}.tp_{suffix}", deps=deps, stage=stage, layer=layer,
        requests=requests, family="allreduce", phase=mode,
        recipe="catalog_tp_allreduce",
        link_bytes=2.0 * (parallel.tp - 1) / parallel.tp * tokens * h
        * dtype_bytes(model.activation_dtype),
        shape=(tokens, h),
        metadata=(("collective", "allreduce"), ("payload_semantics", "logical_activation")),
    )


def _group_plan(plan: LayerPlan, parallel: ParallelSpec) -> tuple[bool, int]:
    groups = int(plan.o_groups)
    tp = max(1, parallel.tp)
    if groups <= 0 or plan.num_heads <= 0 or groups > plan.num_heads:
        return False, 1
    group_heads = plan.num_heads // groups
    local_heads = max(1, plan.num_heads // tp)
    if groups % tp == 0 and group_heads * (groups // tp) == local_heads:
        return True, groups // tp
    return False, groups


def _qkv_widths(plan: LayerPlan, parallel: ParallelSpec) -> tuple[int, int, int, int]:
    local_heads = max(1, plan.num_heads // parallel.tp)
    local_kv = local_kv_heads(plan.num_kv_heads, parallel.tp)
    q_width = (2 if plan.q_output_gate else 1) * local_heads * plan.head_dim
    kv_width = local_kv * plan.head_dim
    return local_heads, local_kv, q_width, kv_width
