"""Logical reads for query-specific indexed attention.

[Role] Separate indexed KV gathers from dense window tile reuse.
[Invariants] A query owns its selection; another query cannot reuse those rows
  merely because it shares a query tile. Shared KV heads do not multiply reads.
  Logical reads are not a measured off-chip traffic or cache-hit estimate.
[Timeouts & Locks] Constant-space integer arithmetic; no waits or locks.
[Side-Effect Policy] Pure workload accounting; no observation reads or fitting.
"""
from __future__ import annotations

from dataclasses import dataclass

ZERO = 0
ONE = 1
TWO = 2
DEFAULT_QUERY_TILE = 128
DEFAULT_INDEX_BYTES = 4


@dataclass(frozen=True)
class SparseAttentionTraffic:
    gather_rows: int
    window_sweep_rows: int
    index_read_bytes: int

    @property
    def total_cache_rows(self):
        return self.gather_rows + self.window_sweep_rows

    def metadata(self):
        return {
            "indexed_gather_rows": self.gather_rows,
            "dense_window_sweep_rows": self.window_sweep_rows,
            "selection_index_read_bytes": self.index_read_bytes,
            "index_read_scope": "valid_selected_slots; sentinel_scans_not_modeled",
            "query_reuse": "independent_selection_per_query",
            "head_reuse": "shared_latent_kv_within_query",
            "traffic_scope": "logical_cache_read_demand; no_cache_hit_model",
            "unmodeled_source_mechanisms": (
                "gm_scratch_materialization_and_rereads",
                "physical_copy_ownership_and_transactions",
                "scalar_index_and_page_lookup_dependencies",
                "physical_matrix_padding_and_source_pipeline",
            ),
        }


def selected_query_owners(query, compute_units):
    """Source TND sparse-attention tasks: one AIC owner per query row.

    The arch22 V_TEMPLATE assigns one query head-group (``gS1Idx``) to an AIC
    task.  A single-token decode therefore activates one AIC and its two AIV
    subblocks, not the whole chip.  This is only the finite task count; work
    remains in :func:`sparse_attention_traffic`.
    """
    values = (query, compute_units)
    if (any(not isinstance(value, int) or isinstance(value, bool) for value in values)
            or query < 0 or compute_units <= 0):
        raise ValueError("sparse_attention_requires_integer_owner_geometry")
    return min(query, compute_units)


def _completed_prefix(tokens, ratio):
    """Sum floor(position / ratio) for positions 1 through tokens."""
    groups, tail = divmod(tokens, ratio)
    return ratio * groups * (groups - 1) // 2 + groups * (tail + 1)


def _window_sweeps(query, context, window, query_tile, *, noncausal=False, kv_len=None):
    if window == ZERO or query == ZERO:
        return ZERO
    if noncausal:
        horizon = max(ZERO, int(kv_len) if kv_len is not None else int(context) + int(query))
        return ((query + query_tile - ONE) // query_tile) * min(horizon, window)
    full, tail = divmod(query, query_tile)
    unsaturated = min(full, max(ZERO, (window - context - ONE) // query_tile))
    rows = (unsaturated * context
            + query_tile * unsaturated * (unsaturated + ONE) // TWO
            + (full - unsaturated) * window)
    return rows + (min(context + query, window) if tail else ZERO)


def sparse_attention_traffic(query, context, topk, *, compression_ratio=ONE,
                             window=ZERO, query_tile=DEFAULT_QUERY_TILE,
                             index_bytes=DEFAULT_INDEX_BYTES,
                             index_pool_size=ONE, index_pool_tail=False,
                             noncausal=False, kv_len=None):
    """Count causal/noncausal valid selections and a separate dense-window read path."""
    values = (query, context, topk, compression_ratio, window, query_tile, index_bytes)
    if (any(not isinstance(value, int) or isinstance(value, bool) for value in values)
            or query < ZERO or context < ZERO or window < ZERO
            or min(topk, compression_ratio, query_tile, index_bytes) <= ZERO):
        raise ValueError("sparse_attention_requires_integer_causal_geometry")
    if index_pool_size > ONE:
        from mindie_sim.catalog.index_pool import pool_selection
        if compression_ratio != ONE or window:
            raise ValueError("index_pool_requires_uncompressed_token_kv_without_dense_window")
        rows = pool_selection(query, context, topk, index_pool_size,
                              always_select_tail=index_pool_tail,
                              noncausal=noncausal, kv_len=kv_len).selected_tokens
    elif noncausal:
        horizon = max(ZERO, int(kv_len) if kv_len is not None else int(context) + int(query))
        rows = query * min(horizon // compression_ratio, topk)
    else:
        unsaturated = min(query, max(ZERO, topk * compression_ratio - ONE - context))
        rows = (_completed_prefix(context + unsaturated, compression_ratio)
                - _completed_prefix(context, compression_ratio)
                + (query - unsaturated) * topk)
    return SparseAttentionTraffic(
        gather_rows=rows,
        window_sweep_rows=_window_sweeps(query, context, window, query_tile,
                                         noncausal=noncausal, kv_len=kv_len),
        index_read_bytes=rows * index_bytes,
    )
