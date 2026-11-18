"""Finite generic dense latent-MLA online-attention schedule.

[Role] Price one independent dense BF16 latent decode request, or the actual
  same-DP contiguous C8 group, without claiming a recovered vendor tiler.
[Invariants] One query owner per request; K is streamed in finite tiles; Q is
  retained on its owner; online max/sum and FP32 [H,Dv] accumulators update
  serially; one final store per request. Batch aggregation is one invocation
  with independent private owners, not two synthetic launch floors.
[Timeouts & Locks] O(requests + key tiles), no I/O or locks.
[Side-Effect Policy] Returns immutable plans; no engine mutation.
"""
from __future__ import annotations

from dataclasses import replace
import math

from mindie_sim.catalog.attention_common import (
    _causal_pairs, _long_selected_pairs, active_dcp_degree, min_positive,
)
from mindie_sim.catalog.cache import completed_rows, index_row_bytes
from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.catalog.context_parallel import (
    dsa_cp_token_ranges, owned_positions, split_causal_pairs, split_pcp_pairs,
)
from mindie_sim.schema.common import ModelingError, dtype_bytes
from .batch_prefix_indexer import RECIPE_NAME as BATCH_INDEXER_RECIPE
from .mechanism import OperatorPlan, Work
from .mla_structure import declare_indexer
from .ownership import read_role, unit_group
from .program import KernelProgram, ParallelProgram, Stage
from .engine import Allocation

KEY_TILE = 128
INDEX_QUERY_TILE = 128
INDEX_NO_ROWS = 0
INDEX_IDENTITY_COMPRESSION = 1
DEFAULT_BLOCK_SIZE = 128
CANDIDATE_BYTES = 8
FP32_BYTES = 4
INT32_BYTES = 4
MIN_LOG_TOPK = 2
MIN_BATCH_MEMBERS = 2
EMPTY_INDEX_ROWS = 0
MIN_INDEX_FACTOR = 1
# A packed index row stores two 4-bit elements per byte. Its score reduction
# consumes a dequantized FP4 payload, so the declared compute identity is the
# dequantized BF16 path (the same convention ``compute_dtype_for`` applies to
# MXFP4 weights). No FP4 cube peak is invented, and the row is no longer
# priced as the INT8 indexer declared by ``quant.index_dtype``.
PACKED_INDEX_ARITHMETIC_DTYPE = "bf16"


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


def _schedule(op):
    raw = dict(op.metadata).get("latent_attention_schedule")
    if not isinstance(raw, dict):
        raise ModelingError("generic_latent_schedule_metadata_required")
    heads = int(raw["heads"])
    qk_dim = int(raw.get("qk_dim", 576))
    latent_dim = int(raw.get("latent_dim", 512))
    dv = int(raw.get("dv", latent_dim))
    kv_row = int(raw["kv_row_bytes"])
    if min(heads, qk_dim, latent_dim, dv, kv_row) <= 0:
        raise ModelingError("generic_latent_schedule_geometry_invalid")
    sequences = raw.get("sequences")
    if sequences is None:
        sequences = ((int(raw.get("query_rows", 1)), int(raw.get("keys", 0))),)
    sequences = tuple((int(q), int(keys)) for q, keys in sequences)
    if not sequences or any(q < 0 or keys < 0 for q, keys in sequences):
        raise ModelingError("generic_latent_schedule_sequence_invalid")
    if any(keys == 0 or q == 0 for q, keys in sequences):
        return None
    return heads, qk_dim, latent_dim, dv, kv_row, sequences


def _alloc(context, core, cores, tag, heads, q_bytes, kv_tile, latent_dim, dv, qk_dim):
    ham = ((heads + 15) // 16) * 16
    pv_keys = max(1, min(KEY_TILE, 65536 // max(1, latent_dim * 2)))
    pv_l0 = pv_keys * latent_dim * 2
    specs = (
        ("q_l1", "l1", q_bytes),
        ("kv_l1", "l1", kv_tile),
        ("acc_ub", "ub", heads * dv * 4 + heads * KEY_TILE * 4),
        ("qk_l0a", "l0a", ham * min(KEY_TILE, max(1, qk_dim)) * 2),
        ("pv_l0b", "l0b", pv_l0),
        ("out_l0c", "l0c", ham * dv * 4),
    )
    items = []
    for label, pool, size in specs:
        if pool not in context.pool_names or size <= 0:
            continue
        p_units = max(1, int(context.pool_units.get(pool, cores)))
        items.append(Allocation(f"{label}_{tag}", context.pool_names[pool], size,
                                (core % p_units,)))
    return tuple(items)


def _item_stages(context, core, cores, tag, heads, qk_dim, latent_dim, dv,
                 q, keys, kv_row, q_bytes, out_bytes, *, after=()):
    read = read_role(context.resource_names)
    place = context.owned(core, cores, (read, "vector", "mte3", "cube"))
    kv_tile = min(KEY_TILE, keys) * kv_row
    alloc = _alloc(context, core, cores, tag, heads, q_bytes, kv_tile, latent_dim, dv, qk_dim)
    ham = ((heads + 15) // 16) * 16

    def tile(index):
        start = index * KEY_TILE
        count = min(KEY_TILE, keys - start)
        load = count * kv_row
        qk_i = 2 * count * ham * qk_dim
        pv_i = 2 * count * ham * latent_dim
        return (
            Stage(f"key_load{tag}_{index}", context.bind(Work(
                resources={"hbm": load, read: load})), placement=place),
            Stage(f"qk{tag}_{index}", context.bind(Work(
                matrix={"bf16": qk_i},
                resources={"vector": 4 * count * heads})),
                  after=(f"key_load{tag}_{index}",), placement=place),
            Stage(f"online{tag}_{index}", context.bind(Work(
                resources={"vector": 8 * count * heads + 2 * heads})),
                  after=(f"qk{tag}_{index}",), placement=place),
            Stage(f"pv{tag}_{index}", context.bind(Work(
                matrix={"bf16": pv_i},
                resources={"vector": 2 * count * heads})),
                  after=(f"online{tag}_{index}",), placement=place),
            Stage(f"update{tag}_{index}", context.bind(Work(
                resources={"vector": 4 * heads * dv})),
                  after=(f"pv{tag}_{index}",), placement=place),
        )

    stages = [Stage(f"alloc{tag}", acquire=alloc, after=after, placement=place),
              Stage(f"q_load{tag}", context.bind(Work(
                  resources={"hbm": q_bytes, read: q_bytes})),
                    after=(f"alloc{tag}",), placement=place)]
    previous = f"q_load{tag}"
    for index in range((keys + KEY_TILE - 1) // KEY_TILE):
        for stage in tile(index):
            # The rolling KV slot may not be overwritten before the preceding
            # update consumed it; each tile's four stages stay causal.
            stages.append(replace(stage, after=(previous,)))
            previous = stage.name
    fin = f"finalize{tag}"
    stages.append(Stage(fin, context.bind(Work(
        resources={"vector": 2 * heads * dv,
                   "hbm": out_bytes, "mte3": out_bytes})),
        after=(previous,), placement=place, free=alloc))
    return stages, fin


def _max_heads_per_owner(heads, q, keys, qk_dim, latent_dim, dv, kv_row, geometry):
    buffers = dict(getattr(geometry, "buffers", {}) or {})

    def cap(pool):
        if pool not in buffers:
            return None
        return int(buffers[pool])

    limit = heads
    ub_cap = cap("ub")
    if ub_cap is not None:
        limit = min(limit, max(1, ub_cap // max(1, (dv + KEY_TILE) * 4)))
    l0c_cap = cap("l0c")
    if l0c_cap is not None:
        max_ham = max(16, (l0c_cap // max(1, dv * 4) // 16) * 16)
        if l0c_cap < 16 * dv * 4:
            max_ham = max(1, l0c_cap // max(1, dv * 4))
        limit = min(limit, max_ham)
    l0a_cap = cap("l0a")
    if l0a_cap is not None:
        qk_panel = min(KEY_TILE, max(1, qk_dim)) * 2
        max_ham = max(16, (l0a_cap // max(1, qk_panel) // 16) * 16)
        if l0a_cap < 16 * qk_panel:
            max_ham = max(1, l0a_cap // max(1, qk_panel))
        limit = min(limit, max_ham)
    l1_cap = cap("l1")
    kv_tile = min(KEY_TILE, keys) * kv_row
    if l1_cap is not None and l1_cap > kv_tile:
        limit = min(limit, max(1, (l1_cap - kv_tile) // max(1, q * qk_dim * 2)))
    if limit >= heads:
        return heads
    return max(16, (limit // 16) * 16) if limit >= 16 else max(1, limit)


def build(op, geometry):
    parsed = _schedule(op)
    if parsed is None:
        return OperatorPlan(Work(), lambda context: KernelProgram(()),
                            metadata={"generic_latent_attention": {"idle": 1}})
    heads, qk_dim, latent_dim, dv, kv_row, sequences = parsed
    declared_roles = set(getattr(geometry, "resource_units", ()) or ()) or {"mte2"}
    read = read_role(declared_roles)
    total = Work()
    logical = issued = 0
    total_q = total_kv = total_out = 0
    owner_items = []
    for q, keys in sequences:
        head_step = _max_heads_per_owner(heads, q, keys, qk_dim, latent_dim, dv, kv_row, geometry)
        for h_start in range(0, heads, head_step):
            h_cur = min(head_step, heads - h_start)
            owner_items.append((q, keys, h_cur))
            q_bytes = q * h_cur * qk_dim * 2
            out_bytes = q * h_cur * latent_dim * 2
            total_q += q_bytes
            total_kv += keys * kv_row
            total_out += out_bytes
            logical += 2 * keys * h_cur * (qk_dim + latent_dim)
            ham = ((h_cur + 15) // 16) * 16
            issued += 2 * keys * ham * (qk_dim + latent_dim)
            total += Work(resources={"hbm": q_bytes, read: q_bytes})
            total += Work(resources={"hbm": keys * kv_row, read: keys * kv_row})
            total += Work(matrix={"bf16": 2 * keys * ham * (qk_dim + latent_dim)})
            owners = (keys + KEY_TILE - 1) // KEY_TILE
            vector = (14 * keys * h_cur + 2 * h_cur * owners
                      + 4 * h_cur * dv * owners + 2 * h_cur * dv)
            total += Work(resources={"vector": vector})
            total += Work(resources={"hbm": out_bytes, "mte3": out_bytes})

    def program(context):
        cores = max(1, int(context.core_count))
        per_core = []
        for core in range(min(cores, len(owner_items))):
            chain = []
            prev = ()
            for tag in range(core, len(owner_items), cores):
                q, keys, h_cur = owner_items[tag]
                stages, fin = _item_stages(
                    context, core, cores, tag, h_cur, qk_dim, latent_dim, dv,
                    q, keys, kv_row, q * h_cur * qk_dim * 2, q * h_cur * latent_dim * 2,
                    after=prev)
                chain.extend(stages)
                prev = (fin,)
            per_core.append(KernelProgram(tuple(chain)))
        return KernelProgram((Stage("owners", program=ParallelProgram(tuple(per_core))),))

    return OperatorPlan(
        total, program,
        metadata={"generic_latent_attention": {
            "heads": heads, "batch": len(sequences), "sequences": sequences,
            "key_tile": KEY_TILE, "qk_dim": qk_dim, "latent_dim": latent_dim,
            "dv": dv, "kv_row_bytes": kv_row,
            "q_bytes": total_q, "kv_unique_bytes": total_kv,
            "output_bytes": total_out, "logical_flops": logical,
            "issued_flops": issued,
            "source_scope": "generic_one_query_owner_online_qk_softmax_pv",
        }},
    )


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
                            score_rows=None, l2_bytes=None):
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
    if runtime is not None:
        # Imported lazily: the epilog module imports this builder for the
        # shared index representation, so a module-level edge would cycle.
        from .implementations import select
        from .mla_cann_epilogs import native_fusion_active
        lowerings = dict(getattr(runtime, "kernels", {}).get("lowerings", {}) or {})
        native_sel = select(lowerings, "native_fusion")
        claimed_role = (native_sel is not None
                        and "lightning_indexer" in (native_sel.options.get("roles") or ()))
        native_fused = (native_fusion_active(runtime)
                        and not claimed_role
                        and select(lowerings, "lightning_indexer") is None
                        and select(lowerings, "quant_lightning_indexer") is None)
    # Query and head-weight operands are real reads in both lowerings.
    by_id = {op.op_id: op for op in b.ops}
    operand_bytes = sum(float(by_id[dep].mte3_bytes)
                        for dep in deps[:2] if dep in by_id)
    if native_fused:
        # One CANN 9.2.1 kernel consumes Q, head weights and the candidate
        # cache rows and writes only the token indices. The FP32 score plane
        # is never materialized, so its write+read disappears.
        fused = b.work(
            "index_quant_lightning", deps, (valid, plan.index_heads, plan.index_dim),
            flops=2 * rows * plan.index_heads * plan.index_dim / divisor,
            vector=(rows * plan.index_heads * 2 / divisor
                    + rows * math.log2(max(MIN_LOG_TOPK, plan.index_topk)) / divisor),
            read=operand_bytes + refill_rows * row_b / divisor,
            hbm_read=operand_bytes + hbm_rows * row_b / divisor,
            write=valid * plan.index_topk * INT32_BYTES,
            family="indexer", dtype=index_arithmetic_dtype(model, quant),
            recipe="quant_lightning_indexer",
            metadata=(("native_kind", "aclnnQuantLightningIndexerV2"),
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
    from .sparse_attention import SparseAttentionTraffic, sparse_attention_traffic
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
    if eff_cp > 1 and plan.cache_kind == "dense" and not topk_present:
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
        raw_pref = max(0, min(
            int(getattr(view, "prefix_tokens", 0) or 0), int(context)))
        aligned_pref = (raw_pref // block) * block
        for q_part, c_part in parts:
            for start in range(0, q_part, query_tile):
                hi = (view.kv_len if plan.noncausal_attention
                      else c_part + min(q_part, start + query_tile))
                lo = (
                    (max(0, hi - plan.window) if plan.noncausal_attention
                     else max(0, c_part + start + 1 - plan.window))
                    if plan.window > 0 else 0
                )
                sh_hi = min(hi, aligned_pref)
                for rank in range(eff_cp):
                    v_sweeps[rank] += owned_positions(
                        lo, hi - 1, eff_cp, interleave, rank)
                    if sh_hi > lo:
                        v_shared[rank] += owned_positions(
                            lo, sh_hi - 1, eff_cp, interleave, rank)
        v_rows = [owned_positions(0, view.kv_len - 1, eff_cp, interleave, rank)
                  for rank in range(eff_cp)]
    else:
        v_pairs = [float(pairs) / eff_cp] * eff_cp
        v_sweeps = [math.ceil(sweeps / eff_cp)] * eff_cp
        raw_pref = max(0, min(
            int(getattr(view, "prefix_tokens", 0) or 0), int(context)))
        aligned_pref = (raw_pref // block) * block
        shared_base = 0
        if plan.cache_kind == "dense" and not topk_present:
            for q_part, c_part in parts:
                for start in range(0, q_part, query_tile):
                    hi = (view.kv_len if plan.noncausal_attention
                          else c_part + min(q_part, start + query_tile))
                    lo = (
                        (max(0, hi - plan.window) if plan.noncausal_attention
                         else max(0, c_part + start + 1 - plan.window))
                        if plan.window > 0 else 0
                    )
                    sh_hi = min(hi, aligned_pref)
                    if sh_hi > lo:
                        shared_base += sh_hi - lo
        shared = math.ceil(shared_base / eff_cp) if eff_cp else shared_base
        v_shared = [shared] * eff_cp
        v_rows = [math.ceil(view.kv_len / eff_cp)] * eff_cp
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
        original_rows * cache_format(model, "window").row_bytes(elements)
        + (sweeps - original_rows) * cache_format(
            model, "compressed_kv").row_bytes(elements))
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
                "original_cache_row_bytes": cache_format(model, "window").row_bytes(plan.head_dim),
                "compressed_cache_row_bytes": cache_format(
                    model, "compressed_kv").row_bytes(plan.head_dim),
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


def batch_ops(ops, outputs, *, runtime, views, prefix):
    """Aggregate same-DP dense decode requests into one structural invocation."""
    if runtime is None or len(outputs) < MIN_BATCH_MEMBERS or len(outputs) != len(views):
        return outputs
    if runtime.parallel.pcp != 1 and any(v.phase == "prefill" for v in views):
        return outputs
    policy = getattr(runtime, "attention_policy", None)
    if policy is not None:
        fused = (policy.get("fused_batch") if hasattr(policy, "get")
                 else getattr(policy, "fused_batch", None))
        if fused is False:
            return outputs
    if "attention" in runtime.kernels.get("families", {}):
        return outputs
    by_id = {op.op_id: op for op in ops}
    decode_pairs = [
        (v, by_id[out_id])
        for v, out_id in zip(views, outputs)
        if v.q_len == 1 and v.phase == "decode" and out_id in by_id
        and dict(by_id[out_id].metadata).get("program_kind") == "dense_latent_attention_generic"
        and not by_id[out_id].plan_required
        and by_id[out_id].recipe not in runtime.kernels.get("recipes", {})
    ]
    if len(decode_pairs) < 2:
        return outputs
    decode_views = [v for v, _ in decode_pairs]
    members = [op for _, op in decode_pairs]
    ids = {op.op_id for op in members}
    schedules = [dict(op.metadata).get("latent_attention_schedule") for op in members]
    if any(item is None for item in schedules):
        return outputs
    common = [{key: value for key, value in item.items()
               if key not in {"query_rows", "keys", "sequences", "q_bytes",
                              "output_bytes", "kv_unique_bytes", "owners", "batch"}}
              for item in schedules]
    if any(item != common[0] for item in common[1:]):
        return outputs
    sequences = tuple((int(v.q_len), int(item["keys"]))
                      for v, item in zip(decode_views, schedules))
    meta = dict(members[0].metadata)
    schedule = dict(meta["latent_attention_schedule"])
    activation_bytes = dtype_bytes(members[0].dtype or "bf16")
    qk_width = int(schedule["qk_dim"])
    latent_width = int(schedule["latent_dim"])
    schedule.update({"sequences": sequences, "batch": len(sequences),
                     "query_rows": sum(q for q, _ in sequences),
                     "keys": max(keys for _, keys in sequences),
                     "owners": len(sequences),
                     "kv_unique_bytes": sum(keys * schedule["kv_row_bytes"]
                                            for _, keys in sequences),
                     "q_bytes": sum(q * schedule["heads"] * qk_width * activation_bytes
                                    for q, _ in sequences),
                     "output_bytes": sum(q * schedule["heads"] * latent_width * activation_bytes
                                         for q, _ in sequences)})
    meta["latent_attention_schedule"] = schedule
    view = dict(meta.get("attention_tensor_view", {}))
    if view:
        view["query_shape"] = (sum(q for q, _ in sequences), schedule["heads"],
                               qk_width)
        view["output_shape"] = (schedule["heads"], sum(q for q, _ in sequences),
                                latent_width)
        view["batch_count"] = len(sequences)
        meta["attention_tensor_view"] = view
    meta["latent_batch_boundary"] = "generic_single_invocation_private_query_owners"
    meta["latent_sequence_requests"] = tuple(v.request_id for v in decode_views)
    meta["latent_batch_members"] = tuple(op.recipe for op in members)
    shape = (len(sequences), schedule["heads"], qk_width)
    batched = replace(
        members[0], op_id=prefix + ".latent_attention_batch",
        recipe="latent_attention0",
        requests=tuple(dict.fromkeys(req for op in members for req in op.requests)),
        deps=tuple(dict.fromkeys(dep for op in members for dep in op.deps)),
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        flops=sum(op.flops for op in members), vector_ops=sum(op.vector_ops for op in members),
        hbm_bytes=sum(op.hbm_bytes for op in members),
        mte2_bytes=sum(op.mte2_bytes for op in members),
        mte3_bytes=sum(op.mte3_bytes for op in members), metadata=tuple(meta.items()))
    first = next(i for i, op in enumerate(ops) if op.op_id in ids)
    ops[:] = [op for op in ops if op.op_id not in ids]
    ops.insert(first, batched)
    new_outputs = []
    inserted = False
    for out_id in outputs:
        if out_id in ids:
            if not inserted:
                new_outputs.append(batched.op_id)
                inserted = True
        else:
            new_outputs.append(out_id)
    return new_outputs


def batch_general_mla_ops(ops, outputs, *, views, prefix, runtime, policy):
    """Combine compatible analytic MLA work in a non-eager invocation."""
    runtime_meta = dict(getattr(runtime, "metadata", ()) or ())
    raw_eager = bool(runtime_meta.get("raw_eager")
                     or runtime_meta.get("execution_policy", {}).get("raw_eager"))
    if len(outputs) < MIN_BATCH_MEMBERS or raw_eager:
        return outputs
    cfg_fam = runtime.kernels.get("families", {}) if runtime else {}
    cfg_rec = runtime.kernels.get("recipes", {}) if runtime else {}
    if "attention" in cfg_fam:
        return outputs
    by_id = {op.op_id: op for op in ops}
    members = [by_id[out_id] for out_id in outputs if out_id in by_id]
    if (len(members) != len(outputs)
            or any(op.plan_required or op.source_boundary or op.recipe in cfg_rec
                   for op in members)):
        return outputs
    # Dense schedules use batch_ops; sparse analytic work preserves every
    # query and its geometry below, including explicit indexed programs.
    first_phase = members[0].phase
    if any(op.phase != first_phase for op in members) or any(
            dict(op.metadata).get("program_kind") == "dense_latent_attention_generic"
            for op in members):
        return outputs
    ids = {op.op_id for op in members}
    reqs = tuple(dict.fromkeys(req for op in members for req in op.requests))
    deps = tuple(dict.fromkeys(dep for op in members for dep in op.deps))
    first_meta = dict(members[0].metadata)
    geometries = [dict(op.metadata).get("indexed_attention_geometry") for op in members]
    if all(geometries):
        geometry = dict(geometries[0])
        sequences = tuple(pair for item in geometries for pair in
                          item.get("sequences", ((item.get("query"), item.get("context")),)))
        geometry.pop("query", None)
        geometry.pop("context", None)
        geometry["sequences"] = sequences
        first_meta["indexed_attention_geometry"] = geometry
    for key in ("indexed_gather_rows", "dense_window_sweep_rows",
                "selection_index_read_bytes"):
        first_meta[key] = sum(dict(op.metadata).get(key, 0) for op in members)
    tensor = dict(first_meta.get("attention_tensor_view", {}))
    for key in ("query_shape", "query_rope_shape", "output_shape"):
        if key in tensor:
            axis = 1 if key == "output_shape" and tensor["kind"] == "dense_paged_latent" else 0
            shape = list(tensor[key])
            shape[axis] = sum(dict(op.metadata)["attention_tensor_view"][key][axis]
                              for op in members)
            tensor[key] = tuple(shape)
    if tensor:
        tensor["batch_count"] = len(members)
        first_meta["attention_tensor_view"] = tensor
    traffic = dict(first_meta.get("attention_traffic_plan", {}))
    if traffic:
        for key in traffic:
            if key != "residency_basis":
                traffic[key] = sum(dict(op.metadata)["attention_traffic_plan"][key]
                                   for op in members)
        first_meta["attention_traffic_plan"] = traffic
    first_meta["request_shapes"] = tuple((v.q_len, v.context) for v in views)
    first_meta["logical_pairs"] = sum(
        float(dict(op.metadata).get("logical_pairs", 0.0)) for op in members)
    first_meta["kv_sweeps"] = sum(
        int(dict(op.metadata).get("kv_sweeps", 0)) for op in members)
    first_shape = members[0].semantic_shape
    total_q = sum(int(op.semantic_shape[0]) for op in members)
    max_kv = max(int(op.semantic_shape[1]) for op in members)
    sem_shape = (total_q, max_kv, *first_shape[2:])
    loc_kv = sum(int(op.local_shape[1]) for op in members)
    loc_shape = (total_q, loc_kv, *members[0].local_shape[2:])
    batched = replace(
        members[0], op_id=prefix + ".latent_attention_batch",
        recipe="latent_attention0", requests=reqs, deps=deps,
        semantic_shape=sem_shape, local_shape=loc_shape,
        padded_shape=sem_shape, launch_shape=sem_shape,
        flops=sum(op.flops for op in members),
        vector_ops=sum(op.vector_ops for op in members),
        hbm_bytes=sum(op.hbm_bytes for op in members),
        mte2_bytes=sum(op.mte2_bytes for op in members),
        mte3_bytes=sum(op.mte3_bytes for op in members),
        metadata=tuple(first_meta.items()),
    )
    first_idx = next(i for i, op in enumerate(ops) if op.op_id in ids)
    ops[:] = [op for op in ops if op.op_id not in ids]
    ops.insert(first_idx, batched)
    return [batched.op_id]
