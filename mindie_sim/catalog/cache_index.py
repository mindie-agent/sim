"""Compressed index-K ownership, capacity and selection read costs.

[Role]
  Compute whole-block CP interleave ownership, worst-rank capacity/store row
  bounds, physical index-K row sizing and TopK selection read/write costs.

[Invariants]
  * A DSA-CP index-K cache is owned in whole interleave blocks: rank ``r``
    holds every block ``b`` with ``b % cp == r``, so per-rank row ownership
    sums strictly to ``rows`` (conservation) and capacity is
    ``ceil(ceil(rows / block) / cp) * block``.
  * Explicit ``cache_representations['index_k']`` formats route through the
    shared ``CacheFormat``; absent representations retain plain element+scale
    arithmetic.

[Timeouts & Locks]
  Pure bounded integer/float arithmetic; no I/O or locks.

[Side-Effect Policy]
  Returns immutable integers, tuples and ``IndexReadCost`` records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mindie_sim.catalog.cache_format import _layer_quant, _lookup_region_decl, cache_format
from mindie_sim.catalog.cache_pools import exact_element_bytes
from mindie_sim.schema.common import ModelingError, dtype_bytes

#: V4/V4.1 lightning-indexer CP interleave.  The kernel interleaves the
#: compressed index-K cache in 128-token groups where one group is a whole
#: ownership block for one CP rank.
INDEX_INTERLEAVE_TOKENS = 128
DEFAULT_CANDIDATE_BLOCKS = 2048
DEFAULT_CANDIDATE_BLOCK_SIZE = 8
GQA_POOL_TAIL_ELEM_BYTES = 2
MLA_POOL_TAIL_ELEM_BYTES = 4


def completed_rows(sequence_tokens: int, ratio: int) -> int:
    if ratio <= 0:
        return 0
    return max(0, int(sequence_tokens) // int(ratio))


def new_completed_rows(context: int, query: int, ratio: int) -> int:
    if ratio <= 0:
        return 0
    return completed_rows(context + query, ratio) - completed_rows(context, ratio)


def window_rows(sequence_tokens: int, window: int) -> int:
    if window <= 0:
        return 0
    return max(0, min(int(sequence_tokens), int(window)))


def candidate_valid_rows(
    source_rows: int, *, blocks: int = DEFAULT_CANDIDATE_BLOCKS,
    block_size: int = DEFAULT_CANDIDATE_BLOCK_SIZE, first_block: int = 0,
) -> int:
    """Rows represented by selected candidate blocks (partial final block kept)."""
    source_rows = max(0, int(source_rows))
    blocks = max(0, int(blocks))
    block_size = max(1, int(block_size))
    first = max(0, int(first_block))
    start = first * block_size
    end = min(source_rows, (first + blocks) * block_size)
    return max(0, end - start)


def index_k_block_rows(compress_ratio: int) -> int:
    """Compressed index-K rows in one CP interleave (ownership) block."""
    ratio = int(compress_ratio)
    if ratio <= 0:
        raise ModelingError("catalog_index_k_ratio_invalid")
    return max(1, int(INDEX_INTERLEAVE_TOKENS) // ratio)


def index_k_ownership_rows(
    rows: int, cp: int, compress_ratio: int,
) -> tuple[int, ...]:
    """Per-rank owned index-K rows under whole-block interleave."""
    total = max(0, int(rows))
    cp = max(1, int(cp))
    block = index_k_block_rows(compress_ratio)
    if cp == 1:
        return (total,)
    blocks = (-(-total // block)) if total else 0
    owned = [0] * cp
    for index in range(blocks):
        first = index * block
        size = min(block, total - first)
        if size > 0:
            owned[index % cp] += int(size)
    return tuple(owned)


def index_k_owned_block_rows(
    rows: int, cp: int, compress_ratio: int,
) -> tuple[int, ...]:
    """Per-rank index-K rows *allocated* under whole-block ownership."""
    total = max(0, int(rows))
    cp = max(1, int(cp))
    block = index_k_block_rows(compress_ratio)
    if cp == 1:
        return (total,)
    blocks = (-(-total // block)) if total else 0
    owned = [0] * cp
    for index in range(blocks):
        owned[index % cp] += block
    return tuple(owned)


def index_k_capacity_rows(rows: int, cp: int, compress_ratio: int) -> int:
    """Per-rank index-K capacity = max over ranks of allocated block rows."""
    total = max(0, int(rows))
    cp = max(1, int(cp))
    if cp == 1:
        return total
    block = index_k_block_rows(compress_ratio)
    blocks = (-(-total // block)) if total else 0
    return int(-(-blocks // cp) * block)


def index_k_store_rows(
    new_rows: int, cp: int, compress_ratio: int,
) -> int:
    """Per-rank index-K *valid rows* a single iteration may have to store."""
    owned = index_k_ownership_rows(new_rows, cp, compress_ratio)
    return int(max(owned) if owned else 0)


def _quant(model: Any) -> Any:
    return getattr(model, "quant", None)


def index_element_bytes(model: Any) -> float:
    quant = _quant(model)
    dtype = getattr(quant, "index_dtype", "") if quant is not None else ""
    return dtype_bytes(str(dtype or "bf16"))


def index_scale_bytes(model: Any) -> float:
    quant = _quant(model)
    scale = getattr(quant, "index_scale_dtype", "") if quant is not None else ""
    if not scale or str(scale) == "none":
        return 0.0
    return dtype_bytes(str(scale))


def _resolve_layer_plan(
    model: Any,
    layer_id: int,
    *,
    scope: str | None = None,
    fallback: Any = None,
) -> Any:
    """Resolve a layer plan by ``index`` or ``native_layer_id`` (safe on sliced models)."""
    target = int(layer_id)
    if target < 0:
        return fallback
    layer_plans = getattr(model, "layer_plans", ()) or ()
    for attr in ("index", "native_layer_id"):
        same_scope = next(
            (cand for cand in layer_plans
             if int(getattr(cand, attr, -1)) == target
             and (scope is None or getattr(cand, "scope", None) == scope)),
            None,
        )
        if same_scope is not None:
            return same_scope
    for attr in ("index", "native_layer_id"):
        any_scope = next(
            (cand for cand in layer_plans if int(getattr(cand, attr, -1)) == target),
            None,
        )
        if any_scope is not None:
            return any_scope
    if 0 <= target < len(layer_plans):
        return layer_plans[target]
    return fallback


def _resolve_index_k_owner_plan(model: Any, plan: Any) -> Any:
    """Resolve the owning layer plan for ``index_k`` via ``index_k_owner_layer_id``."""
    if plan is None:
        return None
    cur = plan
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        owner_id = int(getattr(cur, "index_k_owner_layer_id", -1))
        if owner_id < 0:
            return cur
        nxt = _resolve_layer_plan(
            model, owner_id, scope=getattr(cur, "scope", None), fallback=cur,
        )
        if nxt is cur or nxt is None:
            return cur
        cur = nxt
    return cur


def index_row_parts(
    model: Any,
    index_dim: int,
    *,
    plan: Any = None,
    phase: str | None = None,
) -> tuple[int, int, int]:
    """Exact ``(payload_bytes, scale_bytes, padding_bytes)`` per index-K row."""
    owner_plan = _resolve_index_k_owner_plan(model, plan) if plan is not None else None
    metadata = dict(getattr(model, "metadata", ()) or ())
    if _lookup_region_decl(metadata, "index_k", plan=owner_plan, phase=phase) is not None:
        return cache_format(model, "index_k", plan=owner_plan, phase=phase).row_parts(int(index_dim))
    quant = _layer_quant(model, owner_plan)[0] if owner_plan is not None else _quant(model)
    idx_dtype = str((getattr(quant, "index_dtype", "") if quant is not None else "") or "bf16").strip().lower()
    idx_scale = str((getattr(quant, "index_scale_dtype", "") if quant is not None else "") or "none").strip().lower()
    payload = exact_element_bytes(int(index_dim), idx_dtype)
    scale = exact_element_bytes(1, idx_scale) if (idx_scale and idx_scale != "none") else 0
    return int(payload), int(scale), 0


def index_row_bytes(
    model: Any,
    index_dim: int,
    *,
    plan: Any = None,
    phase: str | None = None,
) -> float:
    """Physical index row bytes from an explicit per-layer or global representation."""
    payload, scale, padding = index_row_parts(model, index_dim, plan=plan, phase=phase)
    return float(payload + scale + padding)


@dataclass(frozen=True)
class IndexReadCost:
    scored_rows: int
    selected_rows: int
    candidate_rows: int
    index_row_bytes: float
    index_read_bytes: float
    cache_write_rows: int
    cache_write_bytes: float
    tail_write_rows: int = 0
    tail_write_bytes: float = 0.0


def index_selection_cost(
    model: Any,
    plan: Any,
    sequence_tokens: int | None = None,
    *,
    context_tokens: int | None = None,
    query_tokens: int | None = None,
    phase: str | None = None,
) -> IndexReadCost:
    """Compute index scoring reads and incremental compressed append/tail writes.

    When ``context_tokens`` or ``query_tokens`` is given, compressed index-K
    appends charge ``floor((context + q) / r) - floor(context / r)`` completed
    rows plus uncompressed pooling/compression tail updates separately instead
    of billing the full prefix history as writes.
    """
    if not (plan.indexer or plan.owns_topk or plan.index_query_per_layer):
        return IndexReadCost(0, 0, 0, 0.0, 0.0, 0, 0.0, 0, 0.0)
    producer = _resolve_index_k_owner_plan(model, plan)
    ratio = max(1, int(producer.compress_ratio or 1), int(producer.index_kpool or 1))
    if context_tokens is not None or query_tokens is not None:
        q = max(0, int(query_tokens if query_tokens is not None else 0))
        if context_tokens is not None:
            ctx = max(0, int(context_tokens))
            seq = ctx + q if sequence_tokens is None else max(0, int(sequence_tokens))
        else:
            seq = max(0, int(sequence_tokens or 0))
            ctx = max(0, seq - q)
    else:
        seq = max(0, int(sequence_tokens or 0))
        ctx = 0
        q = seq
    rows = completed_rows(seq, ratio)
    candidates = 0
    if plan.candidate_filter:
        source = _resolve_layer_plan(
            model, plan.candidate_source_layer_id,
            scope=getattr(plan, "scope", None), fallback=producer,
        )
        candidates = candidate_valid_rows(
            completed_rows(seq, max(1, source.compress_ratio)),
            blocks=plan.candidate_topk_blocks, block_size=plan.candidate_block_size)
    scored = min(rows, candidates) if plan.candidate_filter else rows
    eff_kpool = max(1, int(producer.index_kpool or 1), int(plan.index_kpool or 1))
    eff_topk = int(plan.index_topk or producer.index_topk or 0)
    limit = eff_topk // eff_kpool
    selected = min(scored, limit) if eff_topk else scored
    eff_index_dim = int(producer.index_dim or plan.index_dim or 0)
    width = index_row_bytes(model, eff_index_dim, plan=producer, phase=phase)
    writes = new_completed_rows(ctx, q, ratio) if plan.owns_index_k else 0
    tail_rows = 0
    tail_bytes = 0.0
    if plan.owns_index_k and ratio > 1 and q > 0:
        if int(plan.index_kpool or 0) > 1:
            tail_rows = q
            tail_row_bytes = (
                float(plan.index_dim * GQA_POOL_TAIL_ELEM_BYTES)
                if plan.attention == "gqa"
                else float(2 * plan.index_dim * MLA_POOL_TAIL_ELEM_BYTES)
            )
            tail_bytes = float(tail_rows) * tail_row_bytes
        else:
            tail_rows = min(q, (ctx + q) % ratio)
            tail_bytes = float(tail_rows) * width
    return IndexReadCost(
        scored, selected, candidates, width, scored * width,
        writes, writes * width, tail_rows, tail_bytes,
    )


__all__ = [
    "INDEX_INTERLEAVE_TOKENS",
    "IndexReadCost",
    "candidate_valid_rows",
    "completed_rows",
    "index_element_bytes",
    "index_k_block_rows",
    "index_k_capacity_rows",
    "index_k_owned_block_rows",
    "index_k_ownership_rows",
    "index_k_store_rows",
    "index_row_bytes",
    "index_row_parts",
    "index_scale_bytes",
    "index_selection_cost",
    "new_completed_rows",
    "window_rows",
]
