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
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.context_parallel import (
    dsa_cp_token_ranges, owned_positions, split_causal_pairs, split_pcp_pairs,
)
from mindie_sim.schema.common import ModelingError
from .batch_prefix_indexer import apply_indexer_prefix_reads
from .mechanism import OperatorPlan, Work
from .mla_structure import declare_indexer
from .ownership import read_role, unit_group
from .program import KernelProgram, ParallelProgram, Stage
from .engine import Allocation

KEY_TILE = 128
DEFAULT_BLOCK_SIZE = 128
CANDIDATE_BYTES = 8
FP32_BYTES = 4
MIN_LOG_TOPK = 2
ZERO = 0
ONE = 1
TWO = 2
ZERO_FLOAT = 0.0
MIN_BATCH_MEMBERS = 2


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


def semantic_index_selection(b, *, deps, views, plan, model, quant, parallel,
                            shard, prefix, padded_tokens, runtime=None,
                            score_rows=None):
    """Declare the indexer score/TopK pair with local sharded row accounting."""
    valid = sum(view.q_len for view in views)
    cand_window = (
        plan.candidate_topk_blocks * plan.candidate_block_size
        if plan.candidate_filter else ZERO
    )
    ratio = max(ONE, plan.compress_ratio)
    block_size = max(
        ONE, int(getattr(runtime, "kv_block_size", DEFAULT_BLOCK_SIZE) or DEFAULT_BLOCK_SIZE))
    rows_by_view: list[int] = []
    shared_by_view: list[int] = []
    for view in views:
        parts = visible_parts(view)
        v_rows = int(selected_pairs(
            parts, ratio, cand_window,
            noncausal=bool(plan.noncausal_attention), kv_len=view.kv_len))
        raw_pref = max(ZERO, min(int(getattr(view, "prefix_tokens", ZERO) or ZERO),
                                 int(view.context)))
        aligned_pref = ((raw_pref // block_size) * block_size) // ratio
        q_cnt = sum(q_part for q_part, _ in parts)
        rows_by_view.append(v_rows)
        shared_by_view.append(min(v_rows, aligned_pref * max(ONE, q_cnt)))
    local = score_rows is not None
    rows = max(ZERO_FLOAT, float(score_rows)) if local else float(sum(rows_by_view))
    divisor = ONE if local else max(ONE, int(shard))
    row_b = index_row_bytes(model, plan.index_dim)
    scores = b.work("index_scores", deps, (valid, plan.index_heads, plan.index_dim),
                    flops=TWO * rows * plan.index_heads * plan.index_dim / divisor,
                    vector=rows * plan.index_heads * TWO / divisor,
                    read=rows * row_b / divisor,
                    write=rows * FP32_BYTES / divisor, family="indexer",
                    dtype=quant.index_dtype)
    if not local and shard == ONE and b.ops:
        cfg_fam = runtime.kernels.get("families", {}) if runtime else {}
        cfg_rec = runtime.kernels.get("recipes", {}) if runtime else {}
        pinned = "indexer" in cfg_fam or "index_scores" in cfg_rec
        b.ops[-ONE] = apply_indexer_prefix_reads(
            b.ops[-ONE], views=views, rows_by_view=rows_by_view,
            shared_rows_by_view=shared_by_view, row_bytes=float(row_b),
            layer_index=plan.index, runtime=runtime,
            kernel_reuse="native_per_sequence" if pinned else "tile",
            allow_recipe_rebind=not pinned)
    topk = b.work("index_topk", (scores,), (valid, plan.index_topk),
                 vector=rows * math.log2(max(MIN_LOG_TOPK, plan.index_topk)) / divisor,
                 read=rows * FP32_BYTES / divisor,
                 write=valid * plan.index_topk * CANDIDATE_BYTES,
                 family="topk", dtype="fp32",
                 metadata=(("selection_output_dtype", "int32"),))
    declare_indexer(b.ops, prefix=prefix, plan=plan, parallel=parallel, model=model,
                    quant=quant, views=views, padded_tokens=padded_tokens,
                    scores=scores, topk=topk)
    return topk


def view_causal_pairs(view, query, context, window):
    ranges = getattr(view, "query_ranges", ()) if view.phase == "prefill" else ()
    if not ranges:
        return _causal_pairs(query, context, window)
    counted = sum(
        _causal_pairs(end - start, context + start, window)
        for start, end in ranges if end > start
    )
    catalog = sum(
        split_pcp_pairs(end - start, context + start, ONE,
                        window=max(ZERO, int(window))).total
        for start, end in ranges if end > start
    )
    if counted != catalog:
        raise ModelingError(f"pcp_query_range_pairs_diverged:{counted}!={catalog}")
    return counted


def kv_sweeps(parts, plan, view, query_tile, *, dense, topk):
    """Tile sweeps. Each span's horizon is ``context_before_span + tile_end``."""
    sweeps = ZERO
    long_cap = plan.index_topk if topk else view.kv_len
    dense_cap = min_positive(plan.window, plan.index_topk if topk else ZERO)
    for query, context in parts:
        for start in range(ZERO, query, query_tile):
            horizon = (view.kv_len if plan.noncausal_attention
                       else context + min(query, start + query_tile))
            if dense:
                sweeps += min(horizon, dense_cap) if dense_cap > ZERO else horizon
            else:
                window_rows = min(horizon, plan.window) if plan.window else ZERO
                sweeps += window_rows + min(
                    completed_rows(horizon, plan.compress_ratio), long_cap)
    return sweeps


def sparse_over_parts(parts, plan, query_tile, *, dense, kv_len=None, topk=None):
    from .sparse_attention import SparseAttentionTraffic, sparse_attention_traffic
    cap = plan.index_topk if topk is None else int(topk)
    gather = ZERO
    window = ZERO
    reads = ZERO
    for query, context in parts:
        traffic = sparse_attention_traffic(
            query, context, cap,
            compression_ratio=ONE if dense else plan.compress_ratio,
            window=ZERO if dense else plan.window,
            query_tile=query_tile, index_pool_size=plan.index_kpool,
            index_pool_tail=plan.index_kpool_always_select_tail,
            noncausal=bool(plan.noncausal_attention), kv_len=kv_len)
        gather += traffic.gather_rows
        window += traffic.window_sweep_rows
        reads += traffic.index_read_bytes
    return SparseAttentionTraffic(gather, window, reads)


def attention_view_record(view, *, plan, model, quant, query_tile, eff_cp, block,
                          interleave, tp, dsa, topk_present, heads, latent_dim):
    """Per-view logical pairs, owned sweeps and batch row vectors."""
    context, query = view.context, view.q_len
    parts = dsa_cp_prefill_parts(view, tp) if dsa else visible_parts(view)
    eff_q = (sum(q_part for q_part, _ in parts)
             if dsa and view.phase == "prefill" else query)
    dense_cap = min_positive(plan.window, plan.index_topk if topk_present else ZERO)
    if plan.cache_kind == "dense":
        if topk_present:
            pairs = selected_pairs(
                parts, ONE, dense_cap,
                noncausal=bool(plan.noncausal_attention), kv_len=view.kv_len)
        elif plan.noncausal_attention:
            vis_kv = (min(view.kv_len, plan.window)
                      if plan.window > ZERO else view.kv_len)
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
                (min(view.kv_len, plan.window) if plan.window else ZERO)
                + min(completed_rows(view.kv_len, plan.compress_ratio), long_cap)
            ))
        else:
            win_p = (
                sum(_causal_pairs(q_part, c_part, plan.window)
                    for q_part, c_part in parts)
                if dsa and view.phase == "prefill"
                else view_causal_pairs(view, query, context, plan.window)
            ) if plan.window else ZERO_FLOAT
            pairs = win_p + selected_pairs(
                parts, plan.compress_ratio, plan.index_topk if topk_present else ZERO,
                noncausal=bool(plan.noncausal_attention), kv_len=view.kv_len)
        sweeps = kv_sweeps(parts, plan, view, query_tile, dense=False,
                           topk=bool(topk_present))
    selection_reads, selection_metadata = ZERO, ()
    if topk_present:
        traffic = sparse_over_parts(
            parts, plan, query_tile, dense=plan.cache_kind == "dense",
            kv_len=view.kv_len,
            topk=dense_cap if plan.cache_kind == "dense" else plan.index_topk)
        if plan.index_kpool > ONE:
            pairs = float(traffic.total_cache_rows)
        sweeps = traffic.total_cache_rows
        selection_reads = traffic.index_read_bytes
        selection_metadata = tuple(traffic.metadata().items()) + (
            ("program_kind", "indexed_attention"),
            ("indexed_attention_geometry", {
                "query": eff_q, "context": context, "topk": plan.index_topk,
                **({"sequences": parts} if parts != ((eff_q, context),) else {}),
                "compression_ratio": (ONE if plan.cache_kind == "dense"
                                      else plan.compress_ratio),
                "window": ZERO if plan.cache_kind == "dense" else plan.window,
                "heads": heads, "latent": latent_dim,
                "rope": plan.qk_rope_head_dim, "cp": eff_cp,
                "query_dtype": model.activation_dtype,
                "cache_dtype": quant.kv_dtype,
                **({"index_pool_size": plan.index_kpool,
                    "index_pool_tail": plan.index_kpool_always_select_tail}
                   if plan.index_kpool > ONE else {}),
            }),
        )
    shared = ZERO
    if eff_cp > ONE and plan.cache_kind == "dense" and not topk_present:
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
        v_sweeps = [ZERO] * eff_cp
        v_shared = [ZERO] * eff_cp
        raw_pref = max(ZERO, min(
            int(getattr(view, "prefix_tokens", ZERO) or ZERO), int(context)))
        aligned_pref = (raw_pref // block) * block
        for q_part, c_part in parts:
            for start in range(ZERO, q_part, query_tile):
                hi = (view.kv_len if plan.noncausal_attention
                      else c_part + min(q_part, start + query_tile))
                lo = (
                    (max(ZERO, hi - plan.window) if plan.noncausal_attention
                     else max(ZERO, c_part + start + ONE - plan.window))
                    if plan.window > ZERO else ZERO
                )
                sh_hi = min(hi, aligned_pref)
                for rank in range(eff_cp):
                    v_sweeps[rank] += owned_positions(
                        lo, hi - ONE, eff_cp, interleave, rank)
                    if sh_hi > lo:
                        v_shared[rank] += owned_positions(
                            lo, sh_hi - ONE, eff_cp, interleave, rank)
        v_rows = [owned_positions(ZERO, view.kv_len - ONE, eff_cp, interleave, rank)
                  for rank in range(eff_cp)]
    else:
        v_pairs = [float(pairs) / eff_cp] * eff_cp
        v_sweeps = [math.ceil(sweeps / eff_cp)] * eff_cp
        raw_pref = max(ZERO, min(
            int(getattr(view, "prefix_tokens", ZERO) or ZERO), int(context)))
        aligned_pref = (raw_pref // block) * block
        shared_base = ZERO
        if plan.cache_kind == "dense" and not topk_present:
            for q_part, c_part in parts:
                for start in range(ZERO, q_part, query_tile):
                    hi = (view.kv_len if plan.noncausal_attention
                          else c_part + min(q_part, start + query_tile))
                    lo = (
                        (max(ZERO, hi - plan.window) if plan.noncausal_attention
                         else max(ZERO, c_part + start + ONE - plan.window))
                        if plan.window > ZERO else ZERO
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
        min(view.kv_len if plan.noncausal_attention
            else c_part + min(q_part, start + query_tile), plan.window)
        for q_part, c_part in parts
        for start in range(ZERO, q_part, query_tile)
    ) if plan.window else ZERO
    full = (
        original_rows * cache_format(model, "window").row_bytes(elements)
        + (sweeps - original_rows) * cache_format(
            model, "compressed_kv").row_bytes(elements))
    return full / eff_cp


def dsa_cp_prefill_parts(view, tp):
    """Worst-rank prefill query span under DSA-CP query-token SP."""
    if view.phase != "prefill" or tp <= ONE or view.q_len <= ZERO:
        return visible_parts(view)
    best_parts = ()
    best_cost = -ONE
    for rank in range(tp):
        lo, hi, _, _ = dsa_cp_token_ranges(view.q_len, tp, rank)
        if hi <= lo:
            continue
        span_q, span_c = hi - lo, view.context + lo
        cost = span_q * span_c + span_q * (span_q + ONE) // TWO
        if cost > best_cost:
            best_cost, best_parts = cost, ((span_q, span_c),)
    return best_parts


def attention_tensor_view(plan, view, heads, model, quant, parallel, *, eff_cp=None):
    """Declare active latent tensors separately from paged cache allocation."""
    resolved_cp = int(parallel.cp) if eff_cp is None else int(eff_cp)
    dense_paged = (
        plan.mla_variant == "latent_mla" and plan.cache_kind == "dense"
        and not plan.index_topk
        and model.activation_dtype == "bf16" and quant.kv_dtype == "bf16"
    )
    if parallel.pcp != ONE or (resolved_cp != ONE and not dense_paged):
        return ()
    common = {
        "query_dtype": model.activation_dtype,
        "cache_dtype": quant.kv_dtype,
        "output_dtype": model.activation_dtype,
    }
    if plan.mla_variant == "latent_mla" and plan.index_topk > ZERO:
        query = (view.q_len, heads, plan.kv_lora_rank)
        return (("attention_tensor_view", {
            **common, "kind": "indexed_latent",
            "query_shape": query, "output_shape": query,
            "query_rope_shape": (view.q_len, heads, plan.qk_rope_head_dim),
            "cache_head_shape": (ONE, plan.kv_lora_rank),
            "rope_cache_head_shape": (ONE, plan.qk_rope_head_dim),
        }),)
    if (plan.mla_variant == "latent_mla" and plan.cache_kind == "dense"
            and model.activation_dtype == "bf16" and quant.kv_dtype == "bf16"):
        query = (view.q_len, heads, plan.kv_lora_rank + plan.qk_rope_head_dim)
        return (("attention_tensor_view", {
            **common, "kind": "dense_paged_latent",
            "query_shape": query,
            "cache_head_shape": (ONE, plan.kv_lora_rank + plan.qk_rope_head_dim),
            "output_shape": (heads, view.q_len, plan.kv_lora_rank),
            "page_size": int(getattr(parallel, "kv_block_size", ZERO) or ZERO),
            "source_scope": "generic_dense_latent_online_attention",
        }),)
    if (plan.mla_variant == "latent_mqa" and plan.cache_kind == "window_compressed"
            and plan.head_dim == plan.qk_nope_head_dim + plan.qk_rope_head_dim):
        query = (view.q_len, heads, plan.head_dim)
        regions = dict(model.metadata).get("cache_representations", {})
        if regions:
            return (("attention_tensor_view", {
                "kind": "mixed_quant_shared_kv", "query_shape": query, "output_shape": query,
                "query_dtype": model.activation_dtype, "output_dtype": model.activation_dtype,
                "cache_representations": regions,
                "original_cache_row_bytes": cache_format(model, "window").row_bytes(plan.head_dim),
                "compressed_cache_row_bytes": cache_format(
                    model, "compressed_kv").row_bytes(plan.head_dim),
            }),)
        return (("attention_tensor_view", {
            **common, "kind": "shared_kv",
            "query_shape": query, "output_shape": query,
            "cache_head_shape": (ONE, plan.head_dim),
            "compressed_cache": plan.compress_ratio > ZERO,
        }),)
    return ()


def owned_store_rows(rows_by_view, views, parallel, block, *, contexts=None):
    """Worst-rank rows written by DCP-owned stores for the supplied view rows."""
    degrees = [active_dcp_degree(parallel, view.phase) for view in views]
    starts = [int(view.context) for view in views] if contexts is None else list(contexts)
    max_degree = max(degrees, default=ONE)
    block = max(ONE, int(block))
    best = ZERO
    for rank in range(max_degree):
        total = ZERO
        for start, rows, degree in zip(starts, rows_by_view, degrees):
            rows = max(ZERO, int(rows))
            if not rows:
                continue
            if degree <= ONE:
                total += rows
            elif rank < degree:
                total += owned_positions(
                    int(start), int(start) + rows - ONE, degree, block, rank)
        best = max(best, total)
    return best


def sharded_index_visible_rows(views, parallel, block, ratio):
    """Worst-rank visible index rows under a sharded indexer."""
    degrees = [active_dcp_degree(parallel, view.phase) for view in views]
    max_degree = max(degrees, default=ONE)
    block = max(ONE, int(block))
    ratio = max(ONE, int(ratio))
    per_rank = [ZERO] * max_degree
    for view, degree in zip(views, degrees):
        for offset in range(int(view.q_len)):
            horizon = completed_rows(view.context + offset + ONE, ratio)
            if horizon <= ZERO:
                continue
            if degree <= ONE:
                per_rank = [value + horizon for value in per_rank]
            else:
                for rank in range(degree):
                    per_rank[rank] += owned_positions(
                        ZERO, horizon - ONE, degree, block, rank)
    return max(per_rank, default=ZERO_FLOAT)


def batch_ops(ops, outputs, *, runtime, views, prefix):
    """Aggregate same-DP dense decode requests into one structural invocation."""
    if runtime is None or len(outputs) < MIN_BATCH_MEMBERS or len(outputs) != len(views):
        return outputs
    if runtime.parallel.pcp != ONE and any(v.phase == "prefill" for v in views):
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
    schedule.update({"sequences": sequences, "batch": len(sequences),
                     "query_rows": sum(q for q, _ in sequences),
                     "keys": max(keys for _, keys in sequences),
                     "owners": len(sequences),
                     "kv_unique_bytes": sum(keys * schedule["kv_row_bytes"]
                                            for _, keys in sequences),
                     "q_bytes": sum(q * schedule["heads"] * schedule.get("qk_dim", 576) * 2
                                    for q, _ in sequences),
                     "output_bytes": sum(q * schedule["heads"] * schedule.get("latent_dim", 512) * 2
                                         for q, _ in sequences)})
    meta["latent_attention_schedule"] = schedule
    view = dict(meta.get("attention_tensor_view", {}))
    if view:
        view["query_shape"] = (sum(q for q, _ in sequences), schedule["heads"],
                               schedule.get("qk_dim", 576))
        view["output_shape"] = (schedule["heads"], sum(q for q, _ in sequences),
                                schedule.get("latent_dim", 512))
        view["batch_count"] = len(sequences)
        meta["attention_tensor_view"] = view
    meta["latent_batch_boundary"] = "generic_single_invocation_private_query_owners"
    meta["latent_sequence_requests"] = tuple(v.request_id for v in decode_views)
    meta["latent_batch_members"] = tuple(op.recipe for op in members)
    shape = (len(sequences), schedule["heads"], schedule.get("qk_dim", 576))
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
    """Combine remaining compatible analytic MLA attention ops when fused_batch is enabled."""
    if len(outputs) < MIN_BATCH_MEMBERS or (policy is not None and not policy.fused_batch):
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
    ids = {op.op_id for op in members}
    reqs = tuple(dict.fromkeys(req for op in members for req in op.requests))
    deps = tuple(dict.fromkeys(dep for op in members for dep in op.deps))
    first_meta = dict(members[ZERO].metadata)
    first_meta["request_shapes"] = tuple((v.q_len, v.context) for v in views)
    first_meta["logical_pairs"] = sum(
        float(dict(op.metadata).get("logical_pairs", ZERO_FLOAT)) for op in members)
    first_meta["kv_sweeps"] = sum(
        int(dict(op.metadata).get("kv_sweeps", ZERO)) for op in members)
    first_shape = members[ZERO].semantic_shape
    total_q = sum(int(op.semantic_shape[ZERO]) for op in members)
    max_kv = max(int(op.semantic_shape[ONE]) for op in members)
    sem_shape = (total_q, max_kv, *first_shape[TWO:])
    loc_kv = sum(int(op.local_shape[ONE]) for op in members)
    loc_shape = (total_q, loc_kv, *members[ZERO].local_shape[TWO:])
    batched = replace(
        members[ZERO], op_id=prefix + ".latent_attention_batch",
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
