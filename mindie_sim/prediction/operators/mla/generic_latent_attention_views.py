"""Latent-attention request views and cache traffic accounting.

[Role] Derive logical pairs, tensor views, cache reads and index selection work
  from request spans, cache layout and context-parallel ownership.
[Invariants] Shared prefixes deduplicate external reads while preserving
  per-tile refills. Packed storage and index arithmetic have distinct dtypes.
[Timeouts & Locks] Finite scans of request spans, query tiles and ranks.
[Side-Effect Policy] Index selection updates the supplied operator builder.
  Other helpers return derived records. No I/O, locks or device work.
"""
from __future__ import annotations

from dataclasses import replace
import math

from mindie_sim.catalog.attention_common import (
    _causal_pairs, _long_selected_pairs, active_dcp_degree, min_positive,
)
from mindie_sim.catalog.cache import (
    completed_rows, index_row_bytes, packed_region_row_bytes,
)
from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.catalog.context_parallel import (
    dsa_cp_token_ranges, owned_positions, split_causal_pairs, split_pcp_pairs,
)
from mindie_sim.schema.common import ModelingError

from mindie_sim.prediction.operators.sparse_indexer.batch_prefix_indexer import RECIPE_NAME as BATCH_INDEXER_RECIPE
from .mla_structure import declare_indexer

INDEX_QUERY_TILE = 128
INDEX_NO_ROWS = 0
INDEX_IDENTITY_COMPRESSION = 1
DEFAULT_BLOCK_SIZE = 128
CANDIDATE_BYTES = 8
FP32_BYTES = 4
INT32_BYTES = 4
MIN_LOG_TOPK = 2
EMPTY_INDEX_ROWS = 0
MIN_INDEX_FACTOR = 1
# A packed index row stores two 4-bit elements per byte. Its score reduction
# consumes a dequantized FP4 payload, so the declared compute identity is the
# dequantized BF16 path (the same convention ``compute_dtype_for`` applies to
# MXFP4 weights). No FP4 cube peak is invented, and the row is no longer
# priced as the INT8 indexer declared by ``quant.index_dtype``.
PACKED_INDEX_ARITHMETIC_DTYPE = "bf16"
#: Unquantized floating formats; alone they never select the fused quantized kernel.
UNQUANTIZED_INDEX_DTYPES = frozenset({"bf16", "fp16", "fp32"})
BF16_OPERAND_FORMAT = "bf16"


BF16_INDEXER_NATIVE_KIND = "npu_lightning_indexer"


def index_operands_quantized(quant, model=None):
    """True ONLY when index_q / index_k operands are actually quantized.

    An unquantized BF16/FP16 index_k / index_q pair must dispatch to
    ``npu_lightning_indexer`` (not ``aclnnQuantLightningIndexerV2``), even
    when the surrounding model weights or main KV cache are quantized.
    """
    if model is not None:
        raw = dict(getattr(model, "metadata", ())).get("cache_representations", {})
        if "index_k" in raw:
            fmt = str((raw["index_k"] or {}).get("dtype") or "").strip().lower()
            if fmt:
                return fmt not in UNQUANTIZED_INDEX_DTYPES
    index_dtype = str(getattr(quant, "index_dtype", "") or "").strip().lower()
    return bool(index_dtype) and index_dtype not in UNQUANTIZED_INDEX_DTYPES


def stack_compressor_kind(model, plan):
    """Compressor variant of the window-compressed cache stack this layer uses.

    V4.1's leading sliding-window-only layers declare no compressor of their
    own, yet they belong to the same cache/attention family as the compressing
    layers, so the stack variant is resolved once per model.
    """
    if plan.compress_ratio and plan.compressor_kind:
        return plan.compressor_kind
    kinds = {other.compressor_kind for other in model.layer_plans
             if other.attention == "mla" and other.cache_kind == "window_compressed"
             and other.compress_ratio and other.compressor_kind}
    return next(iter(kinds)) if len(kinds) == 1 else ""


def index_cache_format(model):
    """Declared index-cache representation, or the declared index precision.

    The index key region has its own representation. When the model does not
    declare one, the fallback is the declared *indexer* dtype, never the
    attention KV dtype: those are separate cache regions of the same case.
    """
    raw = dict(getattr(model, "metadata", ())).get("cache_representations", {})
    if "index_k" in raw:
        return cache_format(model, "index_k")
    declared = str(getattr(getattr(model, "quant", None), "index_dtype", "") or "int8")
    return CacheFormat(declared)


def index_cache_is_packed(model):
    """True when one index-cache storage byte holds more than one element."""
    return index_cache_format(model).packed_elements > 1


def index_arithmetic_dtype(model, quant):
    """Logical indexer arithmetic dtype, distinct from packed row storage."""
    if index_cache_is_packed(model):
        return PACKED_INDEX_ARITHMETIC_DTYPE
    return quant.index_dtype


def index_cache_dtype(model):
    """Resolved index-cache storage dtype of the declared index region."""
    return index_cache_format(model).dtype


def visible_parts(view):
    """(query, context) spans whose global starts set the visible KV rows."""
    ranges = getattr(view, "query_ranges", ()) or ()
    if not ranges:
        return ((view.q_len, view.context),)
    return tuple((hi - lo, view.context + lo) for lo, hi in ranges if hi > lo)


def selected_pairs(parts, ratio, topk, *, noncausal=False, kv_len=None):
    return sum(
        _long_selected_pairs(query, context, ratio, topk,
                             noncausal=noncausal, kv_len=kv_len)
        for query, context in parts
    )


def _index_owner_rows(tokens, compression, cp, interleave):
    ratio = max(INDEX_IDENTITY_COMPRESSION, int(compression))
    rows = completed_rows(tokens, ratio)
    block = max(INDEX_IDENTITY_COMPRESSION, int(interleave) // ratio)
    return max(owned_positions(INDEX_NO_ROWS, rows - INDEX_IDENTITY_COMPRESSION,
                               cp, block, rank)
               for rank in range(cp))


def index_key_window(parts, *, query_tile=INDEX_QUERY_TILE, noncausal=False,
                     kv_len=None, cp=INDEX_IDENTITY_COMPRESSION,
                     interleave=INDEX_QUERY_TILE,
                     compression=INDEX_IDENTITY_COMPRESSION):
    """Union rows and per-query-tile refills in compressed, rank-owned rows."""
    tile = max(INDEX_IDENTITY_COMPRESSION, int(query_tile))
    union_rows = refill_rows = INDEX_NO_ROWS
    for query, context in parts:
        query = int(query)
        for start in range(INDEX_NO_ROWS, query, tile):
            end = min(query, start + tile)
            horizon = int(context) + (query if noncausal else end)
            if kv_len is not None:
                horizon = int(kv_len) if noncausal else min(horizon, int(kv_len))
            owner_rows = _index_owner_rows(horizon, compression, int(cp), interleave)
            union_rows = max(union_rows, owner_rows)
            refill_rows += owner_rows
    return union_rows, refill_rows


def unique_index_key_rows(views, *, query_tile=INDEX_QUERY_TILE, noncausal=False,
                          cp=INDEX_IDENTITY_COMPRESSION,
                          interleave=INDEX_QUERY_TILE,
                          compression=INDEX_IDENTITY_COMPRESSION, share_prefix=True):
    """Deduplicate HBM prefixes only; private suffixes and tile refills add."""
    groups: dict[object, int] = {}
    private_rows = refills = INDEX_NO_ROWS
    for view in views:
        union_rows, view_refills = index_key_window(
            visible_parts(view), query_tile=query_tile, noncausal=noncausal,
            kv_len=view.kv_len, cp=cp, interleave=interleave,
            compression=compression)
        refills += view_refills
        prefix = min(int(getattr(view, "prefix_tokens", INDEX_NO_ROWS)
                         or INDEX_NO_ROWS),
                     int(view.context)) if share_prefix else INDEX_NO_ROWS
        prefix = max(INDEX_NO_ROWS, prefix) // interleave * interleave
        shared = min(union_rows,
                     _index_owner_rows(prefix, compression, cp, interleave))
        key = str(getattr(view, "prefix_key", "")
                  or getattr(view, "prefix_cache_key", ""))
        groups[key] = max(groups.get(key, INDEX_NO_ROWS), shared)
        private_rows += union_rows - shared
    return private_rows + sum(groups.values()), refills


def semantic_index_selection(b, *, deps, views, plan, model, quant, parallel,
                            shard, prefix, padded_tokens, runtime=None,
                            score_rows=None, l2_bytes=None, machine=None):
    """Declare the indexer score/TopK pair with local sharded row accounting."""
    valid = sum(view.q_len for view in views)
    cand_window = (
        plan.candidate_topk_blocks * plan.candidate_block_size
        if plan.candidate_filter else 0
    )
    ratio = max(1, plan.compress_ratio)
    block_size = max(
        1, int(getattr(runtime, "kv_block_size", DEFAULT_BLOCK_SIZE) or DEFAULT_BLOCK_SIZE))
    pairs_by_view: list[int] = []
    for view in views:
        parts = visible_parts(view)
        v_rows = int(selected_pairs(
            parts, ratio, cand_window,
            noncausal=bool(plan.noncausal_attention), kv_len=view.kv_len))
        pairs_by_view.append(v_rows)
    # The TopK publisher's selected rows are the rows its consumers attend,
    # not the full per-query score plane that feeds the reducer.
    selected_by_view = [
        sparse_over_parts(
            visible_parts(view), plan, INDEX_QUERY_TILE,
            dense=plan.cache_kind == "dense", kv_len=view.kv_len,
            topk=plan.index_topk).gather_rows
        for view in views]
    local = score_rows is not None
    rows = max(float(EMPTY_INDEX_ROWS), float(score_rows)) if local \
        else float(sum(pairs_by_view))
    divisor = 1 if local else max(1, int(shard))
    row_b = index_row_bytes(model, plan.index_dim)
    # External HBM is the physical union of the tile key sets (deduplicated
    # only across a shared prefix); MTE2 keeps each tile's fetch service.
    cp = int(parallel.cp) if local else INDEX_IDENTITY_COMPRESSION
    cfg_fam = runtime.kernels.get("families", {}) if runtime else {}
    cfg_rec = runtime.kernels.get("recipes", {}) if runtime else {}
    pinned = "indexer" in cfg_fam or "index_scores" in cfg_rec
    shared_reads = (runtime is not None and runtime.prefix_cache_mode == "shared"
                    and runtime.prefix_read_mode != "per_sequence" and not pinned)
    read_args = dict(noncausal=bool(plan.noncausal_attention), cp=cp,
                     interleave=block_size, compression=ratio)
    private_rows, refill_rows = unique_index_key_rows(
        views, share_prefix=False, **read_args)
    union_rows, _ = unique_index_key_rows(
        views, share_prefix=shared_reads, **read_args)
    hbm_rows = float(union_rows)
    if l2_bytes is not None and float(l2_bytes) > 0.0:
        working = hbm_rows * row_b
        if working > float(l2_bytes):
            spill = max(0.0, 1 - float(l2_bytes) / working)
            hbm_rows += max(0.0, float(refill_rows) - hbm_rows) * spill
    native_fused = False
    native_bf16_indexer = False
    if runtime is not None:
        # Imported lazily: the epilog module imports this builder for the
        # shared index representation, so a module-level edge would cycle.
        from mindie_sim.prediction.implementations import select
        from .mla_epilogs import native_fusion_active
        lowerings = dict(getattr(runtime, "kernels", {}).get("lowerings", {}) or {})
        native_sel = select(lowerings, "native_fusion")
        claimed_role = (native_sel is not None
                        and "lightning_indexer" in (native_sel.options.get("roles") or ()))
        native_eligible = (native_fusion_active(runtime, machine)
                           and not claimed_role
                           and select(lowerings, "lightning_indexer") is None
                           and select(lowerings, "quant_lightning_indexer") is None)
        quantized_index = index_operands_quantized(quant, model)
        native_fused = native_eligible
    # Query and head-weight operands are real reads in both lowerings.
    by_id = {op.op_id: op for op in b.ops}
    operand_bytes = sum(float(by_id[dep].mte3_bytes)
                        for dep in deps[:2] if dep in by_id)
    if native_fused:
        # One CANN kernel consumes Q, head weights and the candidate cache rows
        # and writes only the token indices; dispatch QuantLightningIndexerV2
        # only when index Q/K operands are actually quantized, else BF16
        # npu_lightning_indexer.
        op_name = "index_quant_lightning" if quantized_index else "index_lightning"
        op_recipe = "quant_lightning_indexer" if quantized_index else "index_scores"
        native_kind = (
            "aclnnQuantLightningIndexerV2" if quantized_index else BF16_INDEXER_NATIVE_KIND
        )
        fused = b.work(
            op_name, deps, (valid, plan.index_heads, plan.index_dim),
            flops=2 * rows * plan.index_heads * plan.index_dim / divisor,
            vector=(rows * plan.index_heads * 2 / divisor
                    + rows * math.log2(max(MIN_LOG_TOPK, plan.index_topk)) / divisor),
            read=operand_bytes + refill_rows * row_b / divisor,
            hbm_read=operand_bytes + hbm_rows * row_b / divisor,
            write=valid * plan.index_topk * INT32_BYTES,
            family="indexer", dtype=index_arithmetic_dtype(model, quant),
            recipe=op_recipe,
            metadata=(("native_kind", native_kind),
                      ("fused_members", ("index_scores", "index_topk")),
                      ("intermediate_score_bytes", 0),
                      ("selection_output_dtype", "int32"),
                      ("selected_token_rows", float(sum(selected_by_view))),
                      ("selected_request_ids",
                       tuple(view.request_id for view in views)),
                      ("selected_tokens_by_request", tuple(selected_by_view)),
                      ("index_score_shards", max(MIN_INDEX_FACTOR, int(shard))),
                      ("index_context_partitions", cp),
                      ("index_candidate_filter", bool(plan.candidate_filter)),
                      ("index_head_reduction", "complete_local")))
        if shared_reads and union_rows < private_rows:
            fused_op = b.ops[-1]
            b.ops[-1] = replace(fused_op, metadata=fused_op.metadata + (
                ("prefix_read_reuse_applied", True),
                ("prefix_read_mode", "tile"),
                ("prefix_read_logical_bytes", private_rows * row_b / divisor),
                ("prefix_read_hbm_bytes", hbm_rows * row_b / divisor),
                ("prefix_read_l2_core_bytes", refill_rows * row_b / divisor),
            ))
        declare_indexer(b.ops, prefix=prefix, plan=plan, parallel=parallel,
                        model=model, quant=quant, views=views,
                        padded_tokens=padded_tokens, scores=fused, topk=fused)
        return fused
    scores = b.work("index_scores", deps, (valid, plan.index_heads, plan.index_dim),
                    flops=2 * rows * plan.index_heads * plan.index_dim / divisor,
                    vector=rows * plan.index_heads * 2 / divisor,
                    read=operand_bytes + float(refill_rows) * row_b / divisor,
                    hbm_read=operand_bytes + hbm_rows * row_b / divisor,
                    write=rows * FP32_BYTES / divisor, family="indexer",
                    dtype=index_arithmetic_dtype(model, quant),
                    metadata=(("index_score_shards", max(MIN_INDEX_FACTOR, int(shard))),
                              ("index_context_partitions", cp),
                              ("index_candidate_filter", bool(plan.candidate_filter)),
                              ("index_head_reduction", "complete_local")))
    if shared_reads and union_rows < private_rows:
        # The HBM union is already charged. Do not apply a second prefix plan
        # based on per-token score arithmetic to these per-tile key reads.
        op = b.ops[-1]
        b.ops[-1] = replace(op, recipe=BATCH_INDEXER_RECIPE, metadata=op.metadata + (
            ("component", "index_k"), ("prefix_read_reuse_applied", True),
            ("prefix_read_mode", "tile"),
            ("prefix_read_logical_bytes", private_rows * row_b / divisor),
            ("prefix_read_hbm_bytes", hbm_rows * row_b / divisor),
            ("prefix_read_l2_core_bytes", refill_rows * row_b / divisor),
        ))
    topk = b.work("index_topk", (scores,), (valid, plan.index_topk),
                 vector=rows * math.log2(max(MIN_LOG_TOPK, plan.index_topk)) / divisor,
                 read=rows * FP32_BYTES / divisor,
                 write=valid * plan.index_topk * INT32_BYTES,
                 family="topk", dtype="fp32",
                 metadata=(("selection_output_dtype", "int32"),
                           ("selected_token_rows", float(sum(selected_by_view))),
                           ("selected_request_ids",
                            tuple(view.request_id for view in views)),
                           ("selected_tokens_by_request", tuple(selected_by_view))))
    declare_indexer(b.ops, prefix=prefix, plan=plan, parallel=parallel, model=model,
                    quant=quant, views=views, padded_tokens=padded_tokens,
                    scores=scores, topk=topk)
    return topk


def owned_completed_rows(view, ratio):
    """Compression groups completed inside the rank's owned query intervals."""
    context = max(0, int(getattr(view, "context", 0) or 0))
    ranges = tuple(getattr(view, "query_ranges", ()) or ())
    if not ranges:
        return (completed_rows(context + int(view.q_len), ratio)
                - completed_rows(context, ratio))
    return sum(
        completed_rows(context + end, ratio) - completed_rows(context + start, ratio)
        for start, end in ranges if int(end) > int(start))


def view_causal_pairs(view, query, context, window):
    ranges = getattr(view, "query_ranges", ()) if view.phase == "prefill" else ()
    if not ranges:
        return _causal_pairs(query, context, window)
    counted = sum(
        _causal_pairs(end - start, context + start, window)
        for start, end in ranges if end > start
    )
    catalog = sum(
        split_pcp_pairs(end - start, context + start, 1,
                        window=max(0, int(window))).total
        for start, end in ranges if end > start
    )
    if counted != catalog:
        raise ModelingError(f"pcp_query_range_pairs_diverged:{counted}!={catalog}")
    return counted


def window_tile_rows(plan, context, query, start, query_tile, kv_len):
    """Distinct KV rows one causal tile reads from the first query's window.

    A tile can contain several queries; their sliding windows overlap, so the
    span runs from the first query's window start through the last query's end
    rather than charging only ``min(horizon, window)`` rows.
    """
    hi = int(kv_len) if plan.noncausal_attention else context + min(
        query, start + query_tile)
    if plan.window <= 0:
        return hi
    if plan.noncausal_attention:
        return min(hi, plan.window)
    low = max(0, context + start + 1 - plan.window)
    return max(0, hi - low)


def kv_sweeps(parts, plan, view, query_tile, *, dense, topk):
    """Tile sweeps. Each span's horizon is ``context_before_span + tile_end``."""
    sweeps = 0
    long_cap = plan.index_topk if topk else view.kv_len
    dense_cap = min_positive(plan.window, plan.index_topk if topk else 0)
    for query, context in parts:
        for start in range(0, query, query_tile):
            horizon = (view.kv_len if plan.noncausal_attention
                       else context + min(query, start + query_tile))
            if dense:
                if dense_cap <= 0:
                    sweeps += horizon
                elif plan.window > 0 and dense_cap == plan.window:
                    sweeps += window_tile_rows(
                        plan, context, query, start, query_tile, view.kv_len)
                else:
                    sweeps += min(horizon, dense_cap)
            else:
                window_rows = (window_tile_rows(
                    plan, context, query, start, query_tile, view.kv_len)
                    if plan.window else 0)
                sweeps += window_rows + min(
                    completed_rows(horizon, plan.compress_ratio), long_cap)
    return sweeps


def sparse_over_parts(parts, plan, query_tile, *, dense, kv_len=None, topk=None):
    from mindie_sim.prediction.operators.sparse_indexer.sparse_attention import SparseAttentionTraffic, sparse_attention_traffic
    cap = plan.index_topk if topk is None else int(topk)
    gather = 0
    window = 0
    reads = 0
    for query, context in parts:
        traffic = sparse_attention_traffic(
            query, context, cap,
            compression_ratio=1 if dense else plan.compress_ratio,
            window=0 if dense else plan.window,
            query_tile=query_tile, index_pool_size=plan.index_kpool,
            index_pool_tail=plan.index_kpool_always_select_tail,
            noncausal=bool(plan.noncausal_attention), kv_len=kv_len)
        gather += traffic.gather_rows
        window += traffic.window_sweep_rows
        reads += traffic.index_read_bytes
    return SparseAttentionTraffic(gather, window, reads)


def _dense_window_tile_spans(parts, view, plan, query_tile):
    """Yield ``(lo, hi)`` visible KV intervals for each dense query tile."""
    for q_part, c_part in parts:
        for start in range(0, q_part, query_tile):
            hi = (view.kv_len if plan.noncausal_attention
                  else c_part + min(q_part, start + query_tile))
            lo = (
                (max(0, hi - plan.window) if plan.noncausal_attention
                 else max(0, c_part + start + 1 - plan.window))
                if plan.window > 0 else 0
            )
            yield lo, hi


def _window_unique_rows(win_spans, view, plan, eff_cp, interleave):
    """Unique rank-owned KV rows bounded by dense sliding window when active."""
    if plan.window > 0 and win_spans:
        u_lo = min(lo for lo, _ in win_spans)
        u_hi = max(hi for _, hi in win_spans)
        if eff_cp > 1:
            return [owned_positions(u_lo, u_hi - 1, eff_cp, interleave, rank)
                    for rank in range(eff_cp)]
        return [max(0, u_hi - u_lo)]
    if eff_cp > 1:
        return [owned_positions(0, view.kv_len - 1, eff_cp, interleave, rank)
                for rank in range(eff_cp)]
    return [math.ceil(view.kv_len / eff_cp)] * eff_cp


def attention_view_record(view, *, plan, model, quant, query_tile, eff_cp, block,
                          interleave, tp, dsa, topk_present, heads, latent_dim,
                          shared_selection=None):
    """Per-view logical pairs, owned sweeps and batch row vectors."""
    context, query = view.context, view.q_len
    parts = dsa_cp_prefill_parts(view, tp) if dsa else visible_parts(view)
    eff_q = (sum(q_part for q_part, _ in parts)
             if dsa and view.phase == "prefill" else query)
    selected_rows = (shared_selection.rows_for(view.request_id)
                     if shared_selection is not None else None)
    dense_cap = min_positive(plan.window, plan.index_topk if topk_present else 0)
    if plan.cache_kind == "dense":
        if topk_present:
            pairs = selected_pairs(
                parts, 1, dense_cap,
                noncausal=bool(plan.noncausal_attention), kv_len=view.kv_len)
        elif plan.noncausal_attention:
            vis_kv = (min(view.kv_len, plan.window)
                      if plan.window > 0 else view.kv_len)
            pairs = float(eff_q * vis_kv)
        elif dsa and view.phase == "prefill":
            pairs = sum(_causal_pairs(q_part, c_part, plan.window)
                        for q_part, c_part in parts)
        else:
            pairs = view_causal_pairs(view, query, context, plan.window)
        sweeps = kv_sweeps(parts, plan, view, query_tile, dense=True,
                           topk=bool(topk_present))
    else:
        if plan.noncausal_attention:
            long_cap = plan.index_topk if topk_present else view.kv_len
            pairs = float(eff_q * (
                (min(view.kv_len, plan.window) if plan.window else 0)
                + min(completed_rows(view.kv_len, plan.compress_ratio), long_cap)
            ))
        else:
            win_p = (
                sum(_causal_pairs(q_part, c_part, plan.window)
                    for q_part, c_part in parts)
                if dsa and view.phase == "prefill"
                else view_causal_pairs(view, query, context, plan.window)
            ) if plan.window else 0.0
            pairs = win_p + selected_pairs(
                parts, plan.compress_ratio, plan.index_topk if topk_present else 0,
                noncausal=bool(plan.noncausal_attention), kv_len=view.kv_len)
        sweeps = kv_sweeps(parts, plan, view, query_tile, dense=False,
                           topk=bool(topk_present))
    selection_reads, selection_metadata = 0, ()
    if topk_present:
        traffic = sparse_over_parts(
            parts, plan, query_tile, dense=plan.cache_kind == "dense",
            kv_len=view.kv_len,
            topk=dense_cap if plan.cache_kind == "dense" else plan.index_topk)
        if selected_rows is not None:
            # A shared MTP selection belongs to its producer: every later
            # proposal step reads the same INT32 rows and attends the same
            # selected tokens, regardless of its wider verification batch.
            gathered = traffic.gather_rows
            fixed = max(0, int(selected_rows))
            traffic = replace(
                traffic, gather_rows=fixed,
                index_read_bytes=int(fixed * INT32_BYTES))
            if plan.index_kpool <= 1:
                pairs = float(pairs) + float(fixed - gathered)
        if plan.index_kpool > 1:
            pairs = float(traffic.total_cache_rows)
        sweeps = traffic.total_cache_rows
        selection_reads = traffic.index_read_bytes
        selection_metadata = tuple(traffic.metadata().items()) + (
            ("program_kind", "indexed_attention"),
            ("indexed_attention_geometry", {
                "query": eff_q, "context": context, "topk": plan.index_topk,
                **({"sequences": parts} if parts != ((eff_q, context),) else {}),
                "compression_ratio": (1 if plan.cache_kind == "dense"
                                      else plan.compress_ratio),
                "window": 0 if plan.cache_kind == "dense" else plan.window,
                "heads": heads, "latent": latent_dim,
                "rope": plan.qk_rope_head_dim, "cp": eff_cp,
                "query_dtype": model.activation_dtype,
                "cache_dtype": quant.kv_dtype,
                **({"index_pool_size": plan.index_kpool,
                    "index_pool_tail": plan.index_kpool_always_select_tail}
                   if plan.index_kpool > 1 else {}),
            }),
        )
    shared = 0
    raw_pref = max(0, min(
        int(getattr(view, "prefix_tokens", 0) or 0), int(context)))
    aligned_pref = (raw_pref // block) * block
    dense_unindexed = plan.cache_kind == "dense" and not topk_present
    if eff_cp > 1 and dense_unindexed:
        if plan.noncausal_attention:
            v_pairs = list(split_causal_pairs(
                eff_q, view.kv_len - eff_q, eff_cp, interleave,
                window=plan.window, noncausal=True).per_rank)
        else:
            v_pairs = [
                sum(split_causal_pairs(
                    q_part, c_part, eff_cp, interleave,
                    window=plan.window, noncausal=False).per_rank[rank]
                    for q_part, c_part in parts)
                for rank in range(eff_cp)
            ]
        v_sweeps = [0] * eff_cp
        v_shared = [0] * eff_cp
        win_spans: list[tuple[int, int]] = []
        for lo, hi in _dense_window_tile_spans(parts, view, plan, query_tile):
            if hi > lo:
                win_spans.append((lo, hi))
            sh_hi = min(hi, aligned_pref)
            for rank in range(eff_cp):
                v_sweeps[rank] += owned_positions(lo, hi - 1, eff_cp, interleave, rank)
                if sh_hi > lo:
                    v_shared[rank] += owned_positions(lo, sh_hi - 1, eff_cp, interleave, rank)
        v_rows = _window_unique_rows(win_spans, view, plan, eff_cp, interleave)
    else:
        v_pairs = [float(pairs) / eff_cp] * eff_cp
        v_sweeps = [math.ceil(sweeps / eff_cp)] * eff_cp
        shared_base = 0
        win_spans_single: list[tuple[int, int]] = []
        if dense_unindexed:
            for lo, hi in _dense_window_tile_spans(parts, view, plan, query_tile):
                if hi > lo:
                    win_spans_single.append((lo, hi))
                sh_hi = min(hi, aligned_pref)
                if sh_hi > lo:
                    shared_base += sh_hi - lo
        shared = math.ceil(shared_base / eff_cp) if eff_cp else shared_base
        v_shared = [shared] * eff_cp
        v_rows = (
            _window_unique_rows(win_spans_single, view, plan, eff_cp, interleave)
            if dense_unindexed
            else [math.ceil(view.kv_len / eff_cp)] * eff_cp
        )
    return {
        "view": view, "parts": parts, "query": query, "eff_q": eff_q,
        "pairs": float(pairs), "sweeps": sweeps, "v_pairs": v_pairs,
        "v_sweeps": v_sweeps, "v_shared": v_shared, "v_rows": v_rows,
        "selection_reads": selection_reads,
        "selection_metadata": selection_metadata,
    }


def window_cache_read(plan, view, parts, sweeps, eff_cp, query_tile, elements, model):
    """Local window/compressed cache bytes for one view's worst rank."""
    if plan.cache_kind != "window_compressed":
        return None
    original_rows = sum(
        window_tile_rows(plan, c_part, q_part, start, query_tile, view.kv_len)
        for q_part, c_part in parts
        for start in range(0, q_part, query_tile)
    ) if plan.window else 0
    full = (
        original_rows * packed_region_row_bytes(model, "window", elements)
        + (sweeps - original_rows) * packed_region_row_bytes(
            model, "compressed_kv", elements))
    return full / eff_cp


def dsa_cp_prefill_parts(view, tp):
    """Worst-rank prefill query span under DSA-CP query-token SP."""
    if view.phase != "prefill" or tp <= 1 or view.q_len <= 0:
        return visible_parts(view)
    best_parts = ()
    best_cost = -1
    for rank in range(tp):
        lo, hi, _, _ = dsa_cp_token_ranges(view.q_len, tp, rank)
        if hi <= lo:
            continue
        span_q, span_c = hi - lo, view.context + lo
        cost = span_q * span_c + span_q * (span_q + 1) // 2
        if cost > best_cost:
            best_cost, best_parts = cost, ((span_q, span_c),)
    return best_parts


def attention_tensor_view(plan, view, heads, model, quant, parallel, *, eff_cp=None,
                          machine=None):
    """Declare active latent tensors separately from paged cache allocation."""
    resolved_cp = int(parallel.cp) if eff_cp is None else int(eff_cp)
    dense_paged = (
        plan.mla_variant == "latent_mla" and plan.cache_kind == "dense"
        and not plan.index_topk
        and model.activation_dtype == "bf16" and quant.kv_dtype == "bf16"
    )
    if parallel.pcp != 1 or (resolved_cp != 1 and not dense_paged):
        return ()
    common = {
        "query_dtype": model.activation_dtype,
        "cache_dtype": quant.kv_dtype,
        "output_dtype": model.activation_dtype,
    }
    if plan.mla_variant == "latent_mla" and plan.index_topk > 0:
        query = (view.q_len, heads, plan.kv_lora_rank)
        return (("attention_tensor_view", {
            **common, "kind": "indexed_latent",
            "query_shape": query, "output_shape": query,
            "query_rope_shape": (view.q_len, heads, plan.qk_rope_head_dim),
            "cache_head_shape": (1, plan.kv_lora_rank),
            "rope_cache_head_shape": (1, plan.qk_rope_head_dim),
        }),)
    if (plan.mla_variant == "latent_mla" and plan.cache_kind == "dense"
            and model.activation_dtype == "bf16" and quant.kv_dtype == "bf16"):
        query = (view.q_len, heads, plan.kv_lora_rank + plan.qk_rope_head_dim)
        return (("attention_tensor_view", {
            **common, "kind": "dense_paged_latent",
            "query_shape": query,
            "cache_head_shape": (1, plan.kv_lora_rank + plan.qk_rope_head_dim),
            "output_shape": (heads, view.q_len, plan.kv_lora_rank),
            "page_size": int(getattr(parallel, "kv_block_size", 0) or 0),
            "source_scope": "generic_dense_latent_online_attention",
        }),)
    if (plan.mla_variant == "latent_mqa" and plan.cache_kind == "window_compressed"
            and plan.head_dim == plan.qk_nope_head_dim + plan.qk_rope_head_dim):
        query = (view.q_len, heads, plan.head_dim)
        regions = dict(model.metadata).get("cache_representations", {})
        # The compressor variant and the hardware generation, not a model name,
        # select the native sparse-MLA operator family.
        family = {"compressor_kind": stack_compressor_kind(model, plan),
                  "arch_family": str(getattr(machine, "arch_family", "") or "")}
        if regions:
            return (("attention_tensor_view", {
                "kind": "mixed_quant_shared_kv", "query_shape": query, "output_shape": query,
                "query_dtype": model.activation_dtype, "output_dtype": model.activation_dtype,
                "cache_representations": regions,
                "original_cache_row_bytes": packed_region_row_bytes(
                    model, "window", plan.head_dim),
                "compressed_cache_row_bytes": packed_region_row_bytes(
                    model, "compressed_kv", plan.head_dim),
                "cache_head_shape": (1, plan.head_dim),
                "cache_dtype": cache_format(model, "window").dtype,
                "compressed_cache": plan.compress_ratio > 0,
                **family,
            }),)
        return (("attention_tensor_view", {
            **common, "kind": "shared_kv",
            "query_shape": query, "output_shape": query,
            "cache_head_shape": (1, plan.head_dim),
            "compressed_cache": plan.compress_ratio > 0,
            **family,
        }),)
    return ()


def owned_store_rows(rows_by_view, views, parallel, block, *, contexts=None):
    """Worst-rank rows written by DCP-owned stores for the supplied view rows."""
    degrees = [active_dcp_degree(parallel, view.phase) for view in views]
    starts = [int(view.context) for view in views] if contexts is None else list(contexts)
    max_degree = max(degrees, default=1)
    block = max(1, int(block))
    best = 0
    for rank in range(max_degree):
        total = 0
        for start, rows, degree in zip(starts, rows_by_view, degrees):
            rows = max(0, int(rows))
            if not rows:
                continue
            if degree <= 1:
                total += rows
            elif rank < degree:
                total += owned_positions(
                    int(start), int(start) + rows - 1, degree, block, rank)
        best = max(best, total)
    return best


def sharded_index_visible_rows(views, parallel, block, ratio):
    """Worst-rank visible index rows under a sharded indexer."""
    degrees = [active_dcp_degree(parallel, view.phase) for view in views]
    max_degree = max(degrees, default=1)
    block = max(1, int(block))
    ratio = max(1, int(ratio))
    per_rank = [0] * max_degree
    for view, degree in zip(views, degrees):
        for offset in range(int(view.q_len)):
            horizon = completed_rows(view.context + offset + 1, ratio)
            if horizon <= 0:
                continue
            if degree <= 1:
                per_rank = [value + horizon for value in per_rank]
            else:
                for rank in range(degree):
                    per_rank[rank] += owned_positions(
                        0, horizon - 1, degree, block, rank)
    return max(per_rank, default=0.0)
