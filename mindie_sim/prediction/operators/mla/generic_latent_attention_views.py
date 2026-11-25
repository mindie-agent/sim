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

from collections.abc import Mapping
from dataclasses import replace
import math
from typing import Literal

from mindie_sim.catalog.attention_common import (
    _causal_pairs, _long_selected_pairs, active_dcp_degree, min_positive,
)
from mindie_sim.catalog.cache import (
    completed_rows, index_row_bytes, packed_region_row_bytes,
)
from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.catalog.context_parallel import (
    SparseDcpSequence,
    dsa_cp_token_ranges,
    owned_positions,
    round_sparse_dcp_counts,
    round_sparse_dcp_matrix,
    sparse_dcp_capacities,
    sparse_dcp_sequence_for_total,
    split_causal_pairs,
    split_pcp_pairs,
    split_sparse_dcp_batch,
    split_sparse_dcp_batch_per_sequence,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef

from mindie_sim.prediction.operators.sparse_indexer.batch_prefix_indexer import RECIPE_NAME as BATCH_INDEXER_RECIPE
from .mla_structure import declare_indexer

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "vllm-project/vllm-ascend",
        "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        "vllm_ascend/attention/mla_v1.py",
        "f7f672e001651a748d2e88ef0f36797ef78db63321ac657f1c7a944c0f561c22",
    ),
    SourceRef(
        "vllm-project/vllm-ascend",
        "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        "vllm_ascend/models/deepseek_v41/indexer.py",
        "45ec4290a41c25497d3cadfdb40c7466b5a40ae467a76200d82d4fa34b15fa02",
    ),
)

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


def _has_declared_index_k(model, *, plan=None, phase=None) -> bool:
    meta = dict(getattr(model, "metadata", ()) or ())
    keys = (f"index_k:{phase}", "index_k") if phase else ("index_k",)
    if plan is not None:
        per_layer = meta.get("layer_cache_representations")
        if isinstance(per_layer, Mapping):
            layer_map = per_layer.get(str(plan.index))
            if layer_map is None:
                layer_map = per_layer.get(plan.index)
            if isinstance(layer_map, Mapping) and any(layer_map.get(k) is not None for k in keys):
                return True
    global_regions = meta.get("cache_representations")
    return isinstance(global_regions, Mapping) and any(global_regions.get(k) is not None for k in keys)


def index_cache_format(model, *, plan=None, phase=None):
    """Declared index-cache representation, or the declared index precision.

    The index key region has its own representation. When the model does not
    declare one, the fallback is the declared *indexer* dtype, never the
    attention KV dtype: those are separate cache regions of the same case.
    """
    if _has_declared_index_k(model, plan=plan, phase=phase):
        return cache_format(model, "index_k", plan=plan, phase=phase)
    declared = str(getattr(getattr(model, "quant", None), "index_dtype", "") or "int8")
    return CacheFormat(declared)


def index_cache_is_packed(model, *, plan=None, phase=None):
    """True when one index-cache storage byte holds more than one element."""
    return index_cache_format(model, plan=plan, phase=phase).packed_elements > 1


def index_arithmetic_dtype(model, quant, *, plan=None, phase=None):
    """Logical indexer arithmetic dtype, distinct from packed row storage."""
    if index_cache_is_packed(model, plan=plan, phase=phase):
        return PACKED_INDEX_ARITHMETIC_DTYPE
    return quant.index_dtype


def index_cache_dtype(model, *, plan=None, phase=None):
    """Resolved index-cache storage dtype of the declared index region."""
    return index_cache_format(model, plan=plan, phase=phase).dtype


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


def _owned_completed_rows_for_rank(tokens, compression, cp, interleave, rank):
    """Exact store-on-completion ownership for compressed rows (handles ``I % R != 0``)."""
    ratio = max(INDEX_IDENTITY_COMPRESSION, int(compression))
    degree = max(INDEX_IDENTITY_COMPRESSION, int(cp))
    stride = max(INDEX_IDENTITY_COMPRESSION, int(interleave))
    rows = completed_rows(tokens, ratio)
    if rows <= 0:
        return INDEX_NO_ROWS
    if degree <= 1:
        return rows
    if ratio == 1:
        return owned_positions(INDEX_NO_ROWS, rows - 1, degree, stride, rank)
    if stride % ratio == 0:
        block = stride // ratio
        return owned_positions(INDEX_NO_ROWS, rows - 1, degree, block, rank)
    # General store-on-completion owner(j) = (((j + 1) * R - 1) // I) % D.
    # Use periodic cycle LCM(I * D, R) // R to remain O(1) for huge contexts.
    cycle_raw = math.lcm(stride * degree, ratio)
    cycle_rows = cycle_raw // ratio
    full_cycles, rem_rows = divmod(rows, cycle_rows)
    rem_owned = sum(
        1 for j in range(rem_rows)
        if (((j + 1) * ratio - 1) // stride) % degree == rank
    )
    if full_cycles == 0:
        return rem_owned
    cycle_owned = sum(
        1 for j in range(cycle_rows)
        if (((j + 1) * ratio - 1) // stride) % degree == rank
    )
    return full_cycles * cycle_owned + rem_owned


def _index_owner_rows(tokens, compression, cp, interleave):
    degree = max(INDEX_IDENTITY_COMPRESSION, int(cp))
    return max(
        _owned_completed_rows_for_rank(tokens, compression, degree, interleave, rank)
        for rank in range(degree)
    )


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
                          compression=INDEX_IDENTITY_COMPRESSION, share_prefix=True,
                          prefix_block_size: int | None = None):
    """Deduplicate HBM prefixes only; private suffixes and tile refills add.

    Aligns raw prefix once using ``prefix_block_size`` (falling back to
    ``interleave``) before mapping to store-on-completion compressed rows.
    """
    align_block = max(
        INDEX_IDENTITY_COMPRESSION,
        int(prefix_block_size) if prefix_block_size is not None else int(interleave),
    )
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
        prefix = (max(INDEX_NO_ROWS, prefix) // align_block) * align_block
        shared = min(union_rows,
                     _index_owner_rows(prefix, compression, cp, interleave))
        key = str(getattr(view, "prefix_key", "")
                  or getattr(view, "prefix_cache_key", ""))
        groups[key] = max(groups.get(key, INDEX_NO_ROWS), shared)
        private_rows += union_rows - shared
    return private_rows + sum(groups.values()), refills


def _prefix_reuse_meta(private_rows, hbm_rows, refill_rows, row_b, divisor, *, include_component=False):
    head = (("component", "index_k"),) if include_component else ()
    return head + (
        ("prefix_read_reuse_applied", True),
        ("prefix_read_mode", "tile"),
        ("prefix_read_logical_bytes", private_rows * row_b / divisor),
        ("prefix_read_hbm_bytes", hbm_rows * row_b / divisor),
        ("prefix_read_l2_core_bytes", refill_rows * row_b / divisor),
    )


def _resolve_native_indexer_mode(runtime, machine, quant, model) -> tuple[bool, bool]:
    if runtime is None:
        return False, False
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
    return native_eligible, index_operands_quantized(quant, model)


def semantic_index_selection(b, *, deps, views, plan, model, quant, parallel,
                            shard, prefix, padded_tokens, runtime=None,
                            score_rows=None, l2_bytes=None, machine=None,
                            cp_interleave_size=None):
    """Declare the indexer score/TopK pair with local sharded row accounting."""
    valid = sum(view.q_len for view in views)
    cand_window = (
        plan.candidate_topk_blocks * plan.candidate_block_size
        if plan.candidate_filter else 0
    )
    ratio = max(1, plan.compress_ratio)
    block_size = max(
        1, int(getattr(runtime, "kv_block_size", DEFAULT_BLOCK_SIZE) or DEFAULT_BLOCK_SIZE))
    eff_interleave = max(
        1,
        int(cp_interleave_size)
        if cp_interleave_size is not None
        else int(getattr(runtime, "cp_interleave_size", DEFAULT_BLOCK_SIZE) or DEFAULT_BLOCK_SIZE),
    )
    pairs_by_view = [
        int(selected_pairs(
            visible_parts(view), ratio, cand_window,
            noncausal=bool(plan.noncausal_attention), kv_len=view.kv_len))
        for view in views
    ]
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
                     interleave=eff_interleave, compression=ratio,
                     prefix_block_size=block_size)
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
    native_fused, quantized_index = _resolve_native_indexer_mode(runtime, machine, quant, model)
    sel_phase = b.common.get("phase")
    packed_index = index_cache_is_packed(model, plan=plan, phase=sel_phase)
    # Query and head-weight operands are real reads in both lowerings.
    by_id = {op.op_id: op for op in b.ops}
    operand_bytes = sum(float(by_id[dep].mte3_bytes)
                        for dep in deps[:2] if dep in by_id)
    weight_op = by_id.get(deps[1]) if len(deps) > 1 else None
    weight_dtype = (
        str(dict(weight_op.metadata).get("output_format") or weight_op.dtype or "").strip().lower()
        if weight_op is not None else ""
    )
    if native_fused and quantized_index and quant.index_dtype == "int8" and not packed_index:
        # quant_lightning_indexer_tiling.cpp:468-491 requires fp16 weights for INT8 Q/K.
        weight_dtype = "fp16"
    weight_meta = (("weight_dtype", weight_dtype),) if weight_dtype else ()
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
            family="indexer", dtype=index_arithmetic_dtype(model, quant, plan=plan, phase=sel_phase),
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
                      ("index_head_reduction", "complete_local")) + weight_meta)
        if shared_reads and union_rows < private_rows:
            fused_op = b.ops[-1]
            b.ops[-1] = replace(fused_op, metadata=fused_op.metadata + _prefix_reuse_meta(
                private_rows, hbm_rows, refill_rows, row_b, divisor))
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
        b.ops[-1] = replace(op, recipe=BATCH_INDEXER_RECIPE, metadata=op.metadata + _prefix_reuse_meta(
            private_rows, hbm_rows, refill_rows, row_b, divisor, include_component=True))
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


def _view_noncausal_window_q(view, eff_q: int) -> int:
    """Global block query count governing a noncausal SWA window under PCP splitting."""
    return max(int(eff_q), int(getattr(view, "global_q_len", 0) or 0))


def window_tile_rows(plan, context, query, start, query_tile, kv_len, *, global_q=None):
    """Distinct KV rows one tile reads from the first query's window.

    For causal attention, a tile can contain several queries whose sliding
    windows overlap, so the span runs from the first query's window start
    through the last query's end. For noncausal sliding-window attention
    (upstream ``dsa_v41.py:1523,1684``: ``min(seq_lens, query_lens + window)``),
    every query attends to the trailing context window plus the whole current
    query/draft block (preserving global block width under PCP query splits).
    """
    hi = int(kv_len) if plan.noncausal_attention else context + min(
        query, start + query_tile)
    if plan.window <= 0:
        return hi
    if plan.noncausal_attention:
        span_q = query if global_q is None else max(int(query), int(global_q))
        return min(hi, plan.window + span_q)
    low = max(0, context + start + 1 - plan.window)
    return max(0, hi - low)


def kv_sweeps(parts, plan, view, query_tile, *, dense, topk):
    """Tile sweeps. Each span's horizon is ``context_before_span + tile_end``."""
    sweeps = 0
    long_cap = plan.index_topk if topk else view.kv_len
    dense_cap = min_positive(plan.window, plan.index_topk if topk else 0)
    view_global_q = int(getattr(view, "global_q_len", 0) or 0) or None
    for query, context in parts:
        for start in range(0, query, query_tile):
            horizon = (view.kv_len if plan.noncausal_attention
                       else context + min(query, start + query_tile))
            if dense:
                if dense_cap <= 0:
                    sweeps += horizon
                elif plan.window > 0 and dense_cap == plan.window:
                    sweeps += window_tile_rows(
                        plan, context, query, start, query_tile, view.kv_len,
                        global_q=view_global_q)
                else:
                    sweeps += min(horizon, dense_cap)
            else:
                window_rows = (window_tile_rows(
                    plan, context, query, start, query_tile, view.kv_len,
                    global_q=view_global_q)
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
    view_global_q = int(getattr(view, "global_q_len", 0) or 0)
    for q_part, c_part in parts:
        span_q = max(q_part, view_global_q)
        for start in range(0, q_part, query_tile):
            hi = (view.kv_len if plan.noncausal_attention
                  else c_part + min(q_part, start + query_tile))
            lo = (
                (max(0, hi - (plan.window + span_q)) if plan.noncausal_attention
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


def _noncausal_window_rank_pairs(
    eff_q: int, kv_len: int, window: int, eff_cp: int, interleave: int,
    *, window_q: int | None = None,
) -> list[float]:
    """Exact rank-owned noncausal SWA pairs conserving ``eff_q * min(kv_len, window + window_q)``."""
    if eff_q <= 0 or kv_len <= 0:
        return [0.0] * eff_cp
    span_q = int(eff_q) if window_q is None else max(int(eff_q), int(window_q))
    lo = max(0, kv_len - (window + span_q)) if window > 0 else 0
    hi = kv_len - 1
    if hi < lo:
        return [0.0] * eff_cp
    return [
        float(eff_q * owned_positions(lo, hi, eff_cp, interleave, rank))
        for rank in range(eff_cp)
    ]


def _sparse_window_partition(parts, view, plan, query_tile, eff_cp, interleave):
    """Exact per-rank dense-window pairs and sweeps for window_compressed sparse layers."""
    if plan.cache_kind == "dense" or plan.window <= 0:
        return [0.0] * eff_cp, [0] * eff_cp
    if plan.noncausal_attention:
        eff_q_parts = sum(q_part for q_part, _ in parts)
        win_pairs = _noncausal_window_rank_pairs(
            eff_q_parts, view.kv_len, plan.window, eff_cp, interleave,
            window_q=_view_noncausal_window_q(view, eff_q_parts),
        )
    else:
        win_pairs = [
            sum(split_causal_pairs(
                q_part, c_part, eff_cp, interleave,
                window=plan.window, noncausal=False).per_rank[rank]
                for q_part, c_part in parts)
            for rank in range(eff_cp)
        ]
    win_sweeps = [0] * eff_cp
    for lo, hi in _dense_window_tile_spans(parts, view, plan, query_tile):
        if hi > lo:
            for rank in range(eff_cp):
                win_sweeps[rank] += owned_positions(lo, hi - 1, eff_cp, interleave, rank)
    return win_pairs, win_sweeps


def _partition_sparse_view(
    view, plan, parts, eff_q, pairs, sweeps, eff_cp, query_tile, interleave, runtime,
):
    """Per-rank sparse Top-K DCP partition for one view plus pooling descriptor."""
    ratio = 1 if plan.cache_kind == "dense" else max(1, int(plan.compress_ratio or 1))
    always_tail = (
        int(view.kv_len) % int(plan.index_kpool)
        if int(plan.index_kpool) > 1 and bool(plan.index_kpool_always_select_tail)
        else 0
    )
    win_pairs, win_sweeps = _sparse_window_partition(
        parts, view, plan, query_tile, eff_cp, interleave)
    sparse_pairs_total = max(0.0, float(pairs) - sum(win_pairs))
    sparse_sweeps_total = max(0, int(sweeps) - sum(win_sweeps))
    pairs_seq = sparse_dcp_sequence_for_total(
        view.kv_len, sparse_pairs_total, eff_q,
        compress_ratio=ratio, always_select_tail=always_tail)
    sweeps_seq = sparse_dcp_sequence_for_total(
        view.kv_len, sparse_sweeps_total, eff_q,
        compress_ratio=ratio, always_select_tail=always_tail)
    p_split = split_sparse_dcp_batch((pairs_seq,), eff_cp, interleave, runtime=runtime, unit="pairs")
    s_split = split_sparse_dcp_batch((sweeps_seq,), eff_cp, interleave, runtime=runtime, unit="sweeps")
    s_caps = sparse_dcp_capacities(sweeps_seq, eff_cp, interleave)
    s_ints = round_sparse_dcp_counts(s_split.per_rank, sparse_sweeps_total, s_caps)
    v_pairs = [win_pairs[r] + p_split.per_rank[r] for r in range(eff_cp)]
    v_sweeps = [win_sweeps[r] + s_ints[r] for r in range(eff_cp)]
    v_shared = [0] * eff_cp
    if plan.cache_kind == "window_compressed":
        win_spans_wc = [
            (lo, hi) for lo, hi in _dense_window_tile_spans(parts, view, plan, query_tile)
            if hi > lo
        ] if plan.window else []
        win_rows_wc = (
            _window_unique_rows(win_spans_wc, view, plan, eff_cp, interleave)
            if win_spans_wc else [0] * eff_cp
        )
        comp_rows_wc = [
            _owned_completed_rows_for_rank(view.kv_len, ratio, eff_cp, interleave, rank)
            for rank in range(eff_cp)
        ]
        v_rows = [win_rows_wc[r] + comp_rows_wc[r] for r in range(eff_cp)]
    else:
        v_rows = [owned_positions(0, view.kv_len - 1, eff_cp, interleave, rank) for rank in range(eff_cp)]
    spec = {
        "pairs_seq": pairs_seq,
        "sweeps_seq": sweeps_seq,
        "sweeps_total": sparse_sweeps_total,
        "win_pairs": tuple(win_pairs),
        "win_sweeps": tuple(win_sweeps),
    }
    return v_pairs, v_sweeps, v_shared, v_rows, spec


def pool_sparse_view_records(
    view_records: list[dict], eff_cp: int, interleave: int, runtime=None,
) -> None:
    """Apply batch-level variance pooling across sparse DCP view records in place."""
    if eff_cp <= 1 or not view_records:
        return
    sparse_indices = [
        idx for idx, rec in enumerate(view_records) if rec.get("sparse_dcp_spec") is not None
    ]
    if not sparse_indices:
        return
    specs = [view_records[idx]["sparse_dcp_spec"] for idx in sparse_indices]
    pairs_seqs: list[SparseDcpSequence] = [spec["pairs_seq"] for spec in specs]
    sweeps_seqs: list[SparseDcpSequence] = [spec["sweeps_seq"] for spec in specs]
    _, pairs_per_seq = split_sparse_dcp_batch_per_sequence(
        pairs_seqs, eff_cp, interleave, runtime=runtime, unit="pairs")
    _, sweeps_per_seq = split_sparse_dcp_batch_per_sequence(
        sweeps_seqs, eff_cp, interleave, runtime=runtime, unit="sweeps")
    seq_totals = [int(spec["sweeps_total"]) for spec in specs]
    seq_caps = [sparse_dcp_capacities(seq, eff_cp, interleave) for seq in sweeps_seqs]
    sweeps_matrix = round_sparse_dcp_matrix(sweeps_per_seq, seq_totals, seq_caps)
    for pos, rec_idx in enumerate(sparse_indices):
        spec = specs[pos]
        win_p = spec["win_pairs"]
        win_s = spec["win_sweeps"]
        view_records[rec_idx]["v_pairs"] = [win_p[r] + pairs_per_seq[pos][r] for r in range(eff_cp)]
        view_records[rec_idx]["v_sweeps"] = [win_s[r] + sweeps_matrix[pos][r] for r in range(eff_cp)]


def _partition_unindexed_window_compressed_view(
    view, plan, parts, eff_q, pairs, eff_cp, query_tile, interleave,
) -> tuple[list[float], list[int], list[int], list[int]]:
    """Exact rank-owned window + store-on-completion compressed rows for unindexed CP>1."""
    win_pairs, win_sweeps = _sparse_window_partition(
        parts, view, plan, query_tile, eff_cp, interleave)
    ratio = max(1, int(plan.compress_ratio or 1))
    win_spans_wc = [
        (lo, hi) for lo, hi in _dense_window_tile_spans(parts, view, plan, query_tile)
        if hi > lo
    ] if plan.window else []
    win_rows_wc = (
        _window_unique_rows(win_spans_wc, view, plan, eff_cp, interleave)
        if win_spans_wc else [0] * eff_cp
    )
    comp_sweeps = [0] * eff_cp
    for q_part, c_part in parts:
        for start in range(0, q_part, query_tile):
            horizon = (view.kv_len if plan.noncausal_attention
                       else c_part + min(q_part, start + query_tile))
            for rank in range(eff_cp):
                comp_sweeps[rank] += _owned_completed_rows_for_rank(
                    horizon, ratio, eff_cp, interleave, rank,
                )
    comp_rows = [
        _owned_completed_rows_for_rank(view.kv_len, ratio, eff_cp, interleave, rank)
        for rank in range(eff_cp)
    ]
    comp_pairs_total = max(0.0, float(pairs) - sum(win_pairs))
    comp_rows_sum = sum(comp_rows)
    if plan.noncausal_attention:
        comp_pairs = [float(eff_q * comp_rows[r]) for r in range(eff_cp)]
    elif comp_rows_sum > 0:
        comp_pairs = [
            comp_pairs_total * (float(comp_rows[r]) / float(comp_rows_sum))
            for r in range(eff_cp)
        ]
    else:
        comp_pairs = [0.0] * eff_cp
    v_pairs = [win_pairs[r] + comp_pairs[r] for r in range(eff_cp)]
    v_sweeps = [win_sweeps[r] + comp_sweeps[r] for r in range(eff_cp)]
    v_rows = [win_rows_wc[r] + comp_rows[r] for r in range(eff_cp)]
    return v_pairs, v_sweeps, [0] * eff_cp, v_rows


def _single_rank_view_rows(
    view, plan, parts, dense_unindexed, win_spans_single, eff_cp, query_tile, interleave,
):
    """Unique row counts for the single-CP or fallback branch."""
    if dense_unindexed:
        return _window_unique_rows(win_spans_single, view, plan, eff_cp, interleave)
    if plan.cache_kind == "window_compressed":
        win_spans_wc1 = [
            (lo, hi) for lo, hi in _dense_window_tile_spans(parts, view, plan, query_tile)
            if hi > lo
        ] if plan.window else []
        win_u1 = (
            _window_unique_rows(win_spans_wc1, view, plan, eff_cp, interleave)
            if win_spans_wc1 else [0] * eff_cp
        )
        comp_u1 = completed_rows(view.kv_len, max(1, int(plan.compress_ratio or 1)))
        return [win_u1[r] + comp_u1 for r in range(eff_cp)]
    return [math.ceil(view.kv_len / eff_cp)] * eff_cp


def attention_view_record(view, *, plan, model, quant, query_tile, eff_cp, block,
                          interleave, tp, dsa, topk_present, heads, latent_dim,
                          shared_selection=None, runtime=None):
    """Per-view logical pairs, owned sweeps and batch row vectors."""
    context, query = view.context, view.q_len
    parts = dsa_cp_prefill_parts(view, tp) if dsa else visible_parts(view)
    eff_q = (sum(q_part for q_part, _ in parts)
             if dsa and view.phase == "prefill" else query)
    selected_rows = (shared_selection.rows_for(view.request_id)
                     if shared_selection is not None else None)
    window_q = _view_noncausal_window_q(view, eff_q)
    dense_cap = min_positive(plan.window, plan.index_topk if topk_present else 0)
    if plan.cache_kind == "dense":
        if topk_present:
            pairs = selected_pairs(
                parts, 1, dense_cap,
                noncausal=bool(plan.noncausal_attention), kv_len=view.kv_len)
        elif plan.noncausal_attention:
            vis_kv = (min(view.kv_len, plan.window + window_q)
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
                (min(view.kv_len, plan.window + window_q) if plan.window else 0)
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
    sparse_dcp_spec = None
    dense_unindexed = plan.cache_kind == "dense" and not topk_present
    if eff_cp > 1 and dense_unindexed:
        if plan.noncausal_attention:
            v_pairs = _noncausal_window_rank_pairs(
                eff_q, view.kv_len, plan.window, eff_cp, interleave,
                window_q=window_q,
            )
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
    elif eff_cp > 1 and topk_present:
        v_pairs, v_sweeps, v_shared, v_rows, sparse_dcp_spec = _partition_sparse_view(
            view, plan, parts, eff_q, pairs, sweeps, eff_cp, query_tile, interleave, runtime)
    elif eff_cp > 1 and plan.cache_kind == "window_compressed":
        v_pairs, v_sweeps, v_shared, v_rows = _partition_unindexed_window_compressed_view(
            view, plan, parts, eff_q, pairs, eff_cp, query_tile, interleave)
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
        v_rows = _single_rank_view_rows(
            view, plan, parts, dense_unindexed, win_spans_single, eff_cp, query_tile, interleave)
    return {
        "view": view, "parts": parts, "query": query, "eff_q": eff_q,
        "pairs": float(pairs), "sweeps": sweeps, "v_pairs": v_pairs,
        "v_sweeps": v_sweeps, "v_shared": v_shared, "v_rows": v_rows,
        "selection_reads": selection_reads,
        "selection_metadata": selection_metadata,
        **({"sparse_dcp_spec": sparse_dcp_spec} if sparse_dcp_spec is not None else {}),
    }


def window_cache_read_by_rank(
    plan, view, parts, v_sweeps, v_rows, eff_cp, query_tile, elements, model, interleave,
) -> list[tuple[int, int, int, int, float, float]] | None:
    """Per-rank ``(win_sweeps, comp_sweeps, win_unique, comp_unique, win_row_b, comp_row_b)``."""
    if plan.cache_kind != "window_compressed":
        return None
    degree = max(1, int(eff_cp))
    stride = max(1, int(interleave))
    phase = getattr(view, "phase", None)
    win_row_b = float(packed_region_row_bytes(model, "window", elements, plan=plan, phase=phase))
    comp_row_b = float(packed_region_row_bytes(model, "compressed_kv", elements, plan=plan, phase=phase))
    win_sweeps_by_rank = [0] * degree
    win_spans: list[tuple[int, int]] = []
    if plan.window:
        for lo, hi in _dense_window_tile_spans(parts, view, plan, query_tile):
            if hi > lo:
                win_spans.append((lo, hi))
                for rank in range(degree):
                    win_sweeps_by_rank[rank] += owned_positions(
                        lo, hi - 1, degree, stride, rank,
                    )
    win_unique_by_rank = (
        _window_unique_rows(win_spans, view, plan, degree, stride)
        if win_spans else [0] * degree
    )
    out: list[tuple[int, int, int, int, float, float]] = []
    for rank in range(degree):
        w_sw = int(win_sweeps_by_rank[rank])
        tot_sw = int(v_sweeps[rank]) if rank < len(v_sweeps) else w_sw
        c_sw = max(0, tot_sw - w_sw)
        w_uniq = int(win_unique_by_rank[rank])
        tot_uniq = int(v_rows[rank]) if rank < len(v_rows) else w_uniq
        c_uniq = max(0, tot_uniq - w_uniq)
        out.append((w_sw, c_sw, w_uniq, c_uniq, win_row_b, comp_row_b))
    return out


def window_cache_read(plan, view, parts, sweeps, eff_cp, query_tile, elements, model):
    """Local window/compressed cache bytes for one view's worst rank."""
    if plan.cache_kind != "window_compressed":
        return None
    original_rows = sum(
        window_tile_rows(plan, c_part, q_part, start, query_tile, view.kv_len)
        for q_part, c_part in parts
        for start in range(0, q_part, query_tile)
    ) if plan.window else 0
    phase = getattr(view, "phase", None)
    full = (
        original_rows * packed_region_row_bytes(model, "window", elements, plan=plan, phase=phase)
        + (sweeps - original_rows) * packed_region_row_bytes(
            model, "compressed_kv", elements, plan=plan, phase=phase))
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


def sharded_index_visible_rows(views, parallel, block, ratio, *, sharded_dcp=None):
    """Worst-rank visible index rows under a sharded indexer."""
    if sharded_dcp is not None:
        degrees = [max(1, int(sharded_dcp)) for _ in views]
    else:
        degrees = [active_dcp_degree(parallel, view.phase) for view in views]
    max_degree = max(degrees, default=1)
    block = max(1, int(block))
    ratio = max(1, int(ratio))
    per_rank = [0] * max_degree
    for view, degree in zip(views, degrees):
        q_len = max(0, int(view.q_len))
        if q_len <= 0:
            continue
        if ratio == 1:
            if degree <= 1:
                rows = int(_causal_pairs(q_len, int(view.context), 0))
                per_rank = [value + rows for value in per_rank]
            else:
                split = split_causal_pairs(q_len, int(view.context), degree, block)
                for rank in range(degree):
                    per_rank[rank] += int(split.per_rank[rank])
            continue
        for offset in range(q_len):
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
