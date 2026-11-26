"""Tile and cache access planning for BF16 LightningIndexerV2.

[Role] Derive causal panels, cache transfers and TopK exchange storage.
[Invariants] Q tiles share finite L1 with K panels. Per-owner lists and score
  slots retain their actual capacity and lifetimes through exchange.
[Timeouts & Locks] Work scales with active tiles and panels; no I/O or locks.
[Side-Effect Policy] Uses local cache state and returns an access ledger.
"""
import math
from typing import Literal

from mindie_sim.catalog.attention_common import _floor_sum
from mindie_sim.prediction.runtime.l2_residency import LineSegmentLruCache
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/lightning_indexer_v2/op_host/lightning_indexer_v2_tiling.cpp",
        source_sha256="b8988f5bdc1ccf8900c2a7853c4edf012f8cb99f6e9b5af40af56167bcd9aade",
    ),
)

# Native MIX_AIC M tile shared with the quantized sibling. It bounds how many
# query rows reuse one resident key tile before the next key stream starts.
QUERY_TILE_ROWS = 16
SCORE_ELEMENT_BYTES = int(dtype_bytes("fp32"))
INDEX_ELEMENT_BYTES = int(dtype_bytes("int32"))
LOCAL_LIST_ELEMENT_BYTES = SCORE_ELEMENT_BYTES + INDEX_ELEMENT_BYTES
CACHE_LINE_BYTES = 128
CACHE_TRANSFER_BYTES = 32
_WEIGHTED_REDUCTION_OPS_PER_HEAD = 3


def _ceil_div(value, divisor):
    return -(-int(value) // int(divisor))


def _weighted_vector_ops(pairs, heads, kv_heads):
    """One weight multiply, ReLU and accumulate per head, then one sum."""
    return pairs * (_WEIGHTED_REDUCTION_OPS_PER_HEAD * heads - kv_heads)


def _sentinel_elements(spans, compression, topk, kv_heads):
    """Missing TopK slots initialized per query row and KV head."""
    total = 0
    for q, c in spans:
        short = max(0, min(q, topk * compression - c - 1))
        total += short * topk - _floor_sum(short, compression, 1, c + 1)
    return total * kv_heads


def _panel_pairs(q_start, q_end, context, compression, key_start, key_end):
    """Causal pairs of query rows [q_start, q_end) against key rows in a panel."""
    total = 0
    for row in range(q_start, q_end):
        visible = (context + row + 1) // compression
        stop = min(key_end, visible)
        if stop > key_start:
            total += stop - key_start
    return total


def _panel_selection_ops(tile, panel, compression, topk, kv_heads):
    """Selection comparisons for one causal panel (Top1 is a max reduce)."""
    if topk != 1:
        return panel["pairs"] * kv_heads * math.log2(topk)
    total = 0
    for row in range(tile["q_start"], tile["q_end"]):
        visible = (tile["context"] + row + 1) // compression
        count = max(0, min(panel["key_end"], visible) - panel["key_start"])
        total += max(0, count - 1)
    return total * kv_heads


def _tile_sentinel(tile, compression, topk, kv_heads):
    q_rows = tile["q_end"] - tile["q_start"]
    return _sentinel_elements(
        ((q_rows, tile["context"] + tile["q_start"]),),
        compression, topk, kv_heads)


def _key_access_plan(spans, *, compression, key_row_bytes, q_row_bytes,
                     q_tile_rows, topk, kv_heads, aiv_units, ub_capacity,
                     l1_capacity, l2_capacity, owner, identity_prefix,
                     gm_capacity=None, exchange_grant=None,
                     score_row_limit=None, span_keys=None):
    """[Cohesive-Pipeline] Resolve one joint plan for Q/K, scores, lists, exchange and output.

    The plan owns Q tile residency, K panel residency and the causal panel
    order.  It produces logical, GM->L1, L1->L0 and HBM bytes; the executable
    program consumes the same panel boundaries.  L1 capacity is shared by the
    live Q tile and K panels; a full-K retain is selected only when both fit.

    The UB solve includes the exchange receive slot *before* the score panel
    is fixed.  When the receive slot fits inside the score slot it aliases that
    slot after the last panel update; otherwise a dedicated slot is reserved
    and the read is chunked.  GM exchange uses one per-tile grant reused across
    tiles: a tile's write waits for the previous tile's read, and the grant is
    released only after the last read (at ``index_release``).  A caller-held
    grant is bound once instead of allocating a second one.
    """
    line = CACHE_LINE_BYTES
    transfer = CACHE_TRANSFER_BYTES
    l1_capacity = max(0, int(l1_capacity or 0))
    l2_capacity = None if l2_capacity is None else max(0, int(l2_capacity))
    gm_capacity = None if gm_capacity is None else max(0, int(gm_capacity))
    max_live_q = max((q for q, _ in spans), default=1)
    q_tile_rows = min(max(1, int(q_tile_rows)), max(1, max_live_q))
    if l1_capacity > 0:
        # Reserve at least half of L1 for K; a smaller Q tile is an
        # analytical sub-tile when the device cannot hold a full M=16 tile.
        q_tile_rows = min(q_tile_rows,
                          max(1, l1_capacity // max(1, 2 * q_row_bytes)))
    if ub_capacity:
        max_visible = max((((c + q) // compression) for q, c in spans), default=0)
        list_rows = max(1, min(topk, max_visible))
        per_q_ub = (list_rows * kv_heads * LOCAL_LIST_ELEMENT_BYTES
                    + kv_heads * SCORE_ELEMENT_BYTES)
        receive_reserve = (
            topk * kv_heads * LOCAL_LIST_ELEMENT_BYTES
            if aiv_units > 1 and max_visible > topk else 0
        )
        ub_for_tiles = (
            int(ub_capacity) - receive_reserve
            if int(ub_capacity) > receive_reserve else int(ub_capacity)
        )
        if q_tile_rows * per_q_ub > ub_for_tiles:
            q_tile_rows = max(1, ub_for_tiles // max(1, per_q_ub))
    q_allocation = q_tile_rows * q_row_bytes if l1_capacity > 0 else 0
    k_capacity = max(0, l1_capacity - q_allocation)
    resolved_keys = (
        tuple(span_keys)
        if span_keys is not None and len(span_keys) == len(spans)
        else tuple(range(len(spans)))
    )
    visible_by_key: dict[object, int] = {}
    for key, (q, c) in zip(resolved_keys, spans):
        span_bytes = ((c + q) // compression) * key_row_bytes
        if span_bytes > visible_by_key.get(key, 0):
            visible_by_key[key] = span_bytes
    total_visible = sum(visible_by_key.values())
    retained = l1_capacity > 0 and total_visible <= k_capacity
    k_allocation = (total_visible if retained
                    else min(total_visible, k_capacity))
    panel_rows = (max(1, k_allocation // key_row_bytes)
                  if k_allocation >= key_row_bytes else 0)
    list_bytes, panel_rows = _score_panel_layout(
        spans, compression, topk, q_tile_rows, kv_heads, ub_capacity, score_row_limit,
        panel_rows)
    l2_retained = (l2_capacity is not None and l2_capacity > 0
                   and total_visible <= l2_capacity)
    l1_state = (LineSegmentLruCache(k_allocation, line)
                if retained and k_allocation > 0 else None)
    l2_state = (LineSegmentLruCache(l2_capacity, line)
                if l2_capacity is not None and l2_capacity > 0 else None)
    (tiles, score_bytes, key_gm, key_hbm, key_logical, q_bytes_total,
     pairs_total, panels_total) = _key_tiles(
        spans, identity_prefix, q_tile_rows, q_row_bytes, compression, panel_rows,
        key_row_bytes, retained, l1_state, l2_state, l2_retained, line,
        aiv_units, topk, kv_heads, resolved_keys)
    exchange_total = sum(tile["exchange_bytes"] for tile in tiles)
    exchange_per_tile = max((tile["exchange_bytes"] for tile in tiles), default=0)
    receive_required = (topk * kv_heads * LOCAL_LIST_ELEMENT_BYTES if exchange_total else 0)
    receive_bytes = 0
    receive_alias = "none"
    receive_chunks = 0
    if receive_required:
        if receive_required <= score_bytes:
            # The score slot is dead after the last panel's local update, so
            # the exchange receive slot can reuse it without another buffer.
            receive_alias = "score_slot"
            receive_chunks = 1
        else:
            spare = (max(0, int(ub_capacity) - list_bytes - score_bytes)
                     if ub_capacity else receive_required)
            receive_bytes = min(receive_required, spare)
            if receive_bytes <= 0:
                # The first score panel used the whole residual UB.  Re-solve
                # with a panel small enough to reserve the real receive slot.
                if score_row_limit is None and ub_capacity:
                    spare_rows = ((int(ub_capacity) - list_bytes
                                   - receive_required)
                                  // max(1, q_tile_rows * kv_heads * SCORE_ELEMENT_BYTES))
                    if spare_rows >= 1:
                        return _key_access_plan(
                            spans, compression=compression,
                            key_row_bytes=key_row_bytes,
                            q_row_bytes=q_row_bytes, q_tile_rows=q_tile_rows,
                            topk=topk, kv_heads=kv_heads, aiv_units=aiv_units,
                            ub_capacity=ub_capacity, l1_capacity=l1_capacity,
                            l2_capacity=l2_capacity, owner=owner,
                            identity_prefix=identity_prefix,
                            gm_capacity=gm_capacity,
                            exchange_grant=exchange_grant,
                            score_row_limit=spare_rows,
                            span_keys=span_keys)
                raise ModelingError(
                    "bf16_lightning_indexer_exchange_receive_has_no_ub")
            receive_alias = "dedicated"
            receive_chunks = _ceil_div(receive_required, receive_bytes)
    workspace = (list_bytes + score_bytes
                 + (receive_bytes if receive_alias == "dedicated" else 0))
    if ub_capacity and workspace > ub_capacity:
        raise ModelingError("bf16_lightning_indexer_ub_workspace_exceeds_capacity")
    grant_name, grant_bytes = _exchange_grant(exchange_grant, exchange_per_tile, gm_capacity)
    return {
        "owner": int(owner),
        "aiv_units": int(aiv_units),
        "line_bytes": line, "transfer_bytes": transfer,
        "q_tile_rows": q_tile_rows, "q_row_bytes": q_row_bytes,
        "q_allocation_bytes": q_allocation,
        "k_allocation_bytes": k_allocation,
        "l1_allocation_bytes": q_allocation + k_allocation,
        "l2_allocation_bytes": (min(total_visible, l2_capacity)
                                if l2_capacity is not None and l2_capacity > 0
                                else 0),
        "l1_capacity_bytes": l1_capacity,
        "l2_capacity_bytes": l2_capacity,
        "l1_policy": ("retain_prefix" if retained else
                      "stream_panels" if l1_capacity > 0 else "no_l1"),
        "l2_policy": ("retain_prefix" if l2_retained else
                      "stream_panels" if l2_capacity is not None and l2_capacity > 0
                      else "no_l2"),
        "retained": retained, "l2_retained": l2_retained,
        "panel_rows": panel_rows,
        "score_bytes": score_bytes,
        "list_bytes": list_bytes,
        "exchange_receive_bytes": int(receive_required),
        "exchange_receive_additional_bytes": int(receive_bytes),
        "exchange_receive_alias": receive_alias,
        "exchange_receive_chunks": int(receive_chunks),
        "workspace_bytes": workspace,
        "exchange_total_bytes": int(exchange_total),
        "exchange_per_tile_bytes": int(exchange_per_tile),
        "exchange_gm_bytes": int(exchange_per_tile if grant_name is None else 0),
        "exchange_gm_capacity_bytes": gm_capacity,
        "exchange_gm_fits": (gm_capacity is None
                             or gm_capacity >= exchange_per_tile),
        "exchange_grant": grant_name,
        "exchange_grant_bytes": grant_bytes,
        "exchange_grant_reused": grant_name is not None,
        "exchange_hbm_bytes": 2 * int(exchange_total),
        "full_scores_bytes": pairs_total * SCORE_ELEMENT_BYTES,
        "full_lists_bytes": (sum(q for q, _ in spans) * topk * kv_heads
                             * aiv_units * LOCAL_LIST_ELEMENT_BYTES),
        "total_visible_bytes": total_visible,
        "key_logical_bytes": key_logical,
        "key_gm_bytes": key_gm,
        "key_hbm_bytes": key_hbm,
        "q_bytes": q_bytes_total,
        "pairs": pairs_total,
        "panels": panels_total,
        "tiles": tiles,
    }


def _score_panel_layout(
    spans, compression, topk, q_tile_rows, kv_heads, ub_capacity, score_row_limit, panel_rows,
):
    # Per-AIV destination buffers: one 16-row score panel and one local-list
    # slot.  They are reused across panels/tiles; the score slot is also the
    # exchange staging slot after its panel is consumed.
    max_visible = max((((c + q) // compression) for q, c in spans), default=0)
    list_rows = max(1, min(topk, max_visible))
    list_bytes = q_tile_rows * list_rows * kv_heads * LOCAL_LIST_ELEMENT_BYTES
    max_score_rows = 0
    if ub_capacity and list_bytes < ub_capacity:
        max_score_rows = max(
            1, (ub_capacity - list_bytes) // max(1, q_tile_rows * kv_heads * SCORE_ELEMENT_BYTES))
        if score_row_limit is not None:
            max_score_rows = min(max_score_rows, max(1, int(score_row_limit)))
        if panel_rows:
            panel_rows = min(panel_rows, max_score_rows)
        else:
            panel_rows = max_score_rows
    elif score_row_limit is not None:
        panel_rows = max(1, min(panel_rows or int(score_row_limit),
                                int(score_row_limit)))
    return list_bytes, panel_rows


def _key_tiles(
    spans, identity_prefix, q_tile_rows, q_row_bytes, compression, panel_rows, key_row_bytes,
    retained, l1_state, l2_state, l2_retained, line, aiv_units, topk, kv_heads,
    span_keys=None,
):
    tiles = []
    score_bytes = 0
    key_gm = key_hbm = key_logical = q_bytes_total = pairs_total = 0
    panels_total = 0
    keys = (
        tuple(span_keys)
        if span_keys is not None and len(span_keys) == len(spans)
        else tuple(range(len(spans)))
    )
    del identity_prefix, l2_retained
    for request, (q_len, context) in enumerate(spans):
        span_key = keys[request]
        for q_start in range(0, q_len, q_tile_rows):
            q_end = min(q_len, q_start + q_tile_rows)
            q_bytes = (q_end - q_start) * q_row_bytes
            visible = (context + q_end) // compression
            if not visible:
                ranges = ()
            elif panel_rows:
                ranges = tuple((start, min(visible, start + panel_rows))
                               for start in range(0, visible, panel_rows))
            else:
                ranges = ((0, visible),)
            panels = []
            for key_start, key_end in ranges:
                rows = key_end - key_start
                logical = rows * key_row_bytes
                pairs = _panel_pairs(q_start, q_end, context, compression,
                                     key_start, key_end)
                start_line = (key_start * key_row_bytes) // line
                end_line = _ceil_div(key_end * key_row_bytes, line)
                if retained and l1_state is not None:
                    gm_miss_spans = l1_state.access_lines(span_key, start_line, end_line)
                    gm = sum(e - s for s, e in gm_miss_spans) * line
                else:
                    gm_miss_spans = ((start_line, end_line),) if end_line > start_line else ()
                    gm = logical
                if l2_state is not None:
                    hbm_miss_lines = 0
                    for m_start, m_end in gm_miss_spans:
                        for h_start, h_end in l2_state.access_lines(span_key, m_start, m_end):
                            hbm_miss_lines += h_end - h_start
                    hbm = min(gm, hbm_miss_lines * line)
                else:
                    hbm = gm
                panels.append({
                    "key_start": key_start, "key_end": key_end,
                    "rows": rows, "logical_bytes": logical,
                    "gm_bytes": gm, "hbm_bytes": hbm, "pairs": pairs,
                })
                score_bytes = max(score_bytes, pairs * kv_heads * SCORE_ELEMENT_BYTES)
                key_gm += gm
                key_hbm += hbm
                key_logical += logical
                pairs_total += pairs
                panels_total += 1
            q_bytes_total += q_bytes
            lists = (min(aiv_units, max(1, (visible + topk - 1) // topk))
                     if visible else 0)
            entries = min(visible, lists * topk) if lists else 0
            exchange_tile = ((q_end - q_start) * entries * LOCAL_LIST_ELEMENT_BYTES * kv_heads
                             if lists > 1 else 0)
            merge_ops = ((q_end - q_start) * entries
                         * max(1.0, math.log2(lists)) * kv_heads
                         if lists > 1 else 0.0)
            panel_merge_ops = ((q_end - q_start) * kv_heads
                               * max(0, len(panels) - 1)
                               if topk == 1 else 0)
            tiles.append({
                "request": request, "q_start": q_start, "q_end": q_end,
                "context": context, "q_bytes": q_bytes,
                "visible_rows": visible, "panels": panels,
                "lists": lists, "entries": entries,
                "exchange_bytes": exchange_tile, "merge_ops": merge_ops,
                "panel_merge_ops": panel_merge_ops,
                "output_bytes": ((q_end - q_start) * kv_heads * topk * INDEX_ELEMENT_BYTES),
            })
    return tiles, score_bytes, key_gm, key_hbm, key_logical, q_bytes_total, pairs_total, panels_total


def _exchange_grant(exchange_grant, exchange_per_tile, gm_capacity):
    grant_name = None
    grant_bytes = None
    if exchange_grant is not None:
        if isinstance(exchange_grant, dict):
            grant_name = str(exchange_grant.get("name")
                             or exchange_grant.get("grant") or "")
            grant_bytes = exchange_grant.get("bytes")
        else:
            grant_name = str(exchange_grant)
        if not grant_name:
            raise ModelingError("bf16_lightning_indexer_exchange_grant_invalid")
        if grant_bytes is not None:
            grant_bytes = int(grant_bytes)
            if grant_bytes < exchange_per_tile:
                raise ModelingError(
                    "bf16_lightning_indexer_exchange_grant_too_small")
    if gm_capacity is not None and gm_capacity < exchange_per_tile:
        raise ModelingError("bf16_lightning_indexer_exchange_gm_capacity")
    return grant_name, grant_bytes
