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

from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.schema.common import ModelingError, dtype_bytes

#: V4/V4.1 lightning-indexer CP interleave.  The kernel interleaves the
#: compressed index-K cache in 128-token groups where one group is a whole
#: ownership block for one CP rank.
INDEX_INTERLEAVE_TOKENS = 128
DEFAULT_CANDIDATE_BLOCKS = 2048
DEFAULT_CANDIDATE_BLOCK_SIZE = 8


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


def index_row_bytes(model: Any, index_dim: int) -> float:
    """Physical index row bytes from an explicit source representation."""
    raw = dict(getattr(model, "metadata", ())).get("cache_representations", {})
    if "index_k" in raw:
        return float(cache_format(model, "index_k").row_bytes(index_dim))
    return float(index_dim) * index_element_bytes(model) + index_scale_bytes(model)


@dataclass(frozen=True)
class IndexReadCost:
    scored_rows: int
    selected_rows: int
    candidate_rows: int
    index_row_bytes: float
    index_read_bytes: float
    cache_write_rows: int
    cache_write_bytes: float


def index_selection_cost(model: Any, plan: Any, sequence_tokens: int) -> IndexReadCost:
    if not (plan.indexer or plan.owns_topk or plan.index_query_per_layer):
        return IndexReadCost(0, 0, 0, 0.0, 0.0, 0, 0.0)
    owner = plan.index_k_owner_layer_id
    producer = model.layer_plans[owner] if owner >= 0 else plan
    rows = completed_rows(sequence_tokens, max(1, producer.compress_ratio, producer.index_kpool))
    candidates = 0
    if plan.candidate_filter:
        source = model.layer_plans[plan.candidate_source_layer_id]
        candidates = candidate_valid_rows(
            completed_rows(sequence_tokens, max(1, source.compress_ratio)),
            blocks=plan.candidate_topk_blocks, block_size=plan.candidate_block_size)
    scored = min(rows, candidates) if plan.candidate_filter else rows
    limit = plan.index_topk // plan.index_kpool
    selected = min(scored, limit) if plan.index_topk else scored
    width = index_row_bytes(model, plan.index_dim)
    writes = rows if plan.owns_index_k else 0
    return IndexReadCost(scored, selected, candidates, width, scored * width, writes, writes * width)


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
    "index_scale_bytes",
    "index_selection_cost",
    "new_completed_rows",
    "window_rows",
]
