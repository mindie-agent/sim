"""Causal pooled-index geometry, separate from the token KV cache.

[Role] Count compressed index keys and expanded token selections exactly.
[Invariants] A complete pool contributes one index key and pool_size KV rows.
  The unfinished causal tail is appended after selected complete pools. KV
  rows remain uncompressed; pool selection never caps the index scan at top-k.
[Timeouts & Locks] Constant-space integer arithmetic; no I/O or locks.
[Side-Effect Policy] Pure structural accounting, without measured latencies.
"""
from dataclasses import dataclass


def completed_prefix(tokens, pool_size):
    """Sum floor(n / pool_size) for n in [1, tokens]."""
    groups, tail = divmod(tokens, pool_size)
    return pool_size * groups * (groups - 1) // 2 + groups * (tail + 1)


def selected_tokens(position, topk, pool_size=1, always_select_tail=False):
    """Valid token indices for one query after seeing ``position`` tokens."""
    pools, tail = divmod(position, pool_size)
    return min(pools, topk // pool_size) * pool_size + (tail if always_select_tail else 0)


@dataclass(frozen=True)
class PoolSelection:
    scanned_pools: int
    selected_pools: int
    selected_tokens: int
    causal_tail_tokens: int
    completed_pools: int
    historical_pool_rows: int
    retained_tail_rows: int


def pool_selection(query, context, topk, pool_size, *, always_select_tail):
    """Aggregate independent causal queries, including a pool-crossing call."""
    if min(query, context) < 0 or pool_size <= 0 or topk <= 0 or topk % pool_size:
        raise ValueError("index_pool_requires_nonnegative_query_context_and_integral_topk_pools")
    scanned = (completed_prefix(context + query, pool_size)
               - completed_prefix(context, pool_size))
    pool_topk = topk // pool_size
    unsaturated = min(query, max(0, topk - context - 1))
    selected = (completed_prefix(context + unsaturated, pool_size)
                - completed_prefix(context, pool_size)
                + (query - unsaturated) * pool_topk)
    tails = (query * context + query * (query + 1) // 2 - pool_size * scanned
             if always_select_tail else 0)
    completed = (context + query) // pool_size - context // pool_size
    return PoolSelection(
        scanned, selected, selected * pool_size + tails, tails, completed,
        context % pool_size if completed else 0, min(query, pool_size),
    )
