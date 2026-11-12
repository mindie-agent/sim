"""Attention, latent-MLA, recurrent, vision and draft family models.

[Role]
  Build guarded resource demands for GQA/MHA/MQA FlashAttention prefill and
  paged decode, latent MLA with Q/KV LoRA absorption/expansion, shared-KV
  window/compressed attention, and the recurrent/auxiliary canonical families.
  Indexer/KPool physics is imported from ``families_recurrent_indexer``.

[Invariants]
  * Query-tile steps are ``ceil(q*g / M_t)``: one KV read serves a group when
    ``q*g <= M_t``; beyond that every extra step re-reads KV.  KV tile steps
    are ``ceil(s_kv / N_t)`` and paged MTE2 traffic charges whole pages and a
    minimum burst.  Split-K adds exactly ``2*(P-1)*B*H_q*q*(4*d_v+8)`` bytes
    for partial output/LSE traffic plus vector merge.  The core-wave tail
    ``ceil(J/C)`` cannot borrow idle cores.
  * Cube fractal alignment is 16; UB vector rows are padded to 32 bytes.  True
    HBM widths stay unpadded so bytes are conserved.
  * MLA absorbs ``d_n -> r_kv`` with per-head weights and expands
    ``r_kv -> d_v``; the latent attention width is ``r_kv + d_r``.  Ragged
    pairs, real request lengths and state traffic are never replaced by a
    mean or a fitted lookup.
  * Draft/verify work is never discounted by acceptance; recurrent state
    traffic and committed slots are explicit.

[Timeouts & Locks]
  Pure bounded expression construction and O(requests + query tiles) sums; no
  I/O, waits or locks.

[Side-Effect Policy]
  Returns immutable ``FamilyTemplate`` records.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from typing import Any

from mindie_sim.schema.common import ModelingError
from .compiler import (
    CompiledOperator, FamilyTemplate, SegmentAxis, _demand, _frac, _out_name,
)
from .expressions import Add, CeilDiv, Const, Div, Max, Min, Mul, Sub, Var
from .guards import SplitKGuard, StepGuard
from .families_recurrent_indexer import (
    BURST_BYTES, DESCRIPTOR_BYTES, PAGE_ROWS, align, compile_compression,
    compile_indexer, compile_recurrent, gather_bytes, index_scanned_rows,
    k1_new_rows, range_burst_bytes, selected_token_rows,
)


def _ceil_div(value: int, divisor: int) -> int:
    return -(-int(value) // int(divisor))


# ---------------------------------------------------------------------------
# Attention geometry
# ---------------------------------------------------------------------------


def _static_attention(ctx: CompiledOperator, operator, *, mla: bool) -> dict[str, Any]:
    theta = operator.theta
    heads = max(1, int(theta.find("q", 1) or 1))
    kv_heads = max(1, int(theta.find("kv", 1) or 1))
    # GQA groups are ceil(H_q / H_kv): a ragged final group is padded rather
    # than dropped.  Under TP this is the local query/KV head ratio, including
    # KV-head replication when tp exceeds the logical KV head count.
    group = heads if mla else max(1, _ceil_div(heads, kv_heads))
    # A1 records dn/dr/rkv and omits dk.  Never price that missing head as 1.
    head_map = theta.group("heads")
    dk = int(head_map.get("dk") or (
        int(head_map.get("dn", 128) or 128) + int(head_map.get("dr", 64) or 0)))
    dk = max(1, dk)
    dv = max(1, int(theta.find("dv", 1) or 1))
    dn = max(1, int(theta.find("dn", dk) or dk))
    dr = max(0, int(theta.find("dr", 0) or 0))
    rkv = max(0, int(theta.find("rkv", 0) or 0))
    if mla and rkv <= 0:
        rkv = dk
    qk_width = rkv + dr if mla else dk
    v_width = rkv if mla else dv
    q_width = dn + dr if mla else dk
    act = _frac(theta.find("act_bytes", 2) or 2)
    # Ordinary attention stores independent K and V rows per KV head; the
    # latent MLA cache stores one (r_kv + d_r)-wide row per latent head.
    default_kv = (Fraction(kv_heads) * qk_width if mla
                  else Fraction(kv_heads) * (qk_width + v_width)) * act
    kv_bytes = _frac(theta.find("kv_bytes", default_kv) or default_kv)
    mt = max(1, int(theta.require("mt")))
    nt = max(1, int(theta.require("nt")))
    cores = max(1, int(theta.require("cores")))
    page = max(1, int(theta.find("page", 0) or PAGE_ROWS))
    ub_multiple = max(1, _ceil_div(32, max(1, int(act))))
    branch = str(theta.find("branch", "") or "")
    ratio = max(1, int(theta.find("compress_ratio", 0) or 1))
    topk = max(0, int(theta.find("index_topk", 0) or 0))
    kpool = max(1, int(theta.find("kpool", 1) or 1))
    always_tail = bool(int(theta.find("kpool_always_select_tail", 0) or 0))
    cube_qk = align(qk_width, 16)
    cube_v = align(v_width, 16)
    dn_cube, rkv_cube, dv_cube = align(dn, 16), align(rkv, 16), align(dv, 16)
    ub_v = align(v_width, ub_multiple)
    R_cube = Fraction(str(ctx.supply.effective("cube")))
    R_mte2 = Fraction(str(ctx.supply.effective("mte2")))
    R_vec = Fraction(str(ctx.supply.effective("vector")))
    head_row = kv_bytes / kv_heads
    cube_row = 2 * (cube_qk + cube_v)
    vec_pair = 7 + Fraction(3 * ub_v + 10, nt)
    balance_cube_mte2 = head_row * R_cube / (cube_row * R_mte2)
    balance_vec_mte2 = head_row * R_vec / (vec_pair * R_mte2)
    cube_vector_ratio = cube_row * R_vec / (vec_pair * R_cube)
    return {
        "heads": heads, "kv_heads": kv_heads, "group": group,
        "dk": dk, "dv": dv, "dn": dn, "dr": dr, "rkv": rkv,
        "qk_width": qk_width, "v_width": v_width, "q_width": q_width,
        "act": act, "kv_bytes": kv_bytes, "head_kv_row": kv_bytes / kv_heads,
        "mt": mt, "nt": nt, "cores": cores, "page": page,
        "ub_multiple": ub_multiple, "ub_v": ub_v,
        "cube_qk": cube_qk, "cube_v": cube_v,
        "dn_cube": dn_cube, "rkv_cube": rkv_cube, "dv_cube": dv_cube,
        "branch": branch, "ratio": ratio, "topk": topk, "kpool": kpool,
        "always_tail": always_tail,
        "balance_cube_mte2": balance_cube_mte2,
        "balance_vec_mte2": balance_vec_mte2,
        "cube_vector_ratio": cube_vector_ratio,
    }


def _attention_metrics(ctx: CompiledOperator, operator, geometry,
                       *, mla: bool = False) -> dict[str, Any]:
    s = _static_attention(ctx, operator, mla=mla)
    heads, kv_heads, group = s["heads"], s["kv_heads"], s["group"]
    mt, nt, page = s["mt"], s["nt"], s["page"]
    window = max(0, int(operator.theta.find("window", 0) or 0))
    causal = bool(operator.theta.find("causal", 1))
    compressed = (not mla) and s["branch"] == "compressed"
    pairs = query_rows = max_rows = max_kv = tasks = kv_tiles = kv_rows = 0
    burst = Fraction(0)
    page_txn = physical = cube = fixpipe = selected = compressed_new = 0
    from collections import Counter
    vector = 0
    q_bytes = Fraction(0)
    logical = 0
    task_cube_max = Fraction(0)
    absorbed = int(operator.theta.find("absorb", 1) or 0) != 0
    for request, req_mult in Counter(geometry.requests).items():
        query, context = request.q_len, request.context
        if not query:
            continue
        ranges = request.query_ranges
        if ranges and sum(max(0, end - start) for start, end in ranges) > 1_048_576:
            raise ModelingError("workload_domain_exceeded:max_tokens")
        # Physical tile bounds follow the selected positions in token order;
        # caller range order must never shrink the causal KV horizon.
        positions = (tuple(sorted(position for start, end in ranges
                                  for position in range(start, end)))
                     if ranges else None)
        selected_q_len = len(positions) if positions is not None else query
        if not selected_q_len:
            continue
        max_kv = max(max_kv, request.kv_len)
        # Restricted query ranges select real physical rows: tile counts, Q
        # loads and output rows follow the selected length, never the padded
        # request length.
        query_rows += req_mult * selected_q_len * heads
        q_bytes += req_mult * selected_q_len * heads * s["q_width"] * s["act"]
        rows_total = selected_q_len * group
        max_rows = max(max_rows, rows_total)
        n_qt = max(1, _ceil_div(rows_total, mt))
        if n_qt > 262_144:
            raise ModelingError("workload_domain_exceeded:max_query_tiles")
        if compressed:
            compressed_new += req_mult * k1_new_rows(request.kv_len, selected_q_len,
                                                     s["ratio"])
            sel_positions = sel_tiles = 0
            if positions is None and s["topk"] > 0 and s["kpool"] <= 1:
                sat = max(0, min(query, s["topk"] * max(1, s["ratio"]) - context - 1))
                sat_tiles = min(n_qt, _ceil_div((sat + 1) * group, mt) + 1)
                tile_picks = [0] * sat_tiles
                for index in range(sat):
                    picked = selected_token_rows(context + index + 1, s["ratio"],
                                                 s["topk"], s["kpool"], s["always_tail"])
                    sel_positions += picked
                    if picked:
                        sel_tiles += _ceil_div(picked, nt)
                    for ti in range((index * group) // mt, min(sat_tiles, ((index + 1) * group - 1) // mt + 1)):
                        tile_picks[ti] += picked
                rem = query - sat
                if rem > 0:
                    sel_positions += rem * s["topk"]
                    sel_tiles += rem * _ceil_div(s["topk"], nt)
                    for index in range(sat, min(query, (sat_tiles * mt + group - 1) // group)):
                        for ti in range((index * group) // mt, min(sat_tiles, ((index + 1) * group - 1) // mt + 1)):
                            tile_picks[ti] += s["topk"]
                    if n_qt > sat_tiles:
                        full_t = min(n_qt - 1, rows_total // mt - 1)
                        if full_t >= sat_tiles:
                            toks = (full_t * mt + mt - 1) // group - (full_t * mt) // group + 1
                            tile_picks.append(toks * s["topk"])
                        last_toks = (rows_total - 1) // group - ((n_qt - 1) * mt) // group + 1
                        tile_picks.append(last_toks * s["topk"])
            else:
                tile_picks = [0] * n_qt
                token_iter = positions if positions is not None else range(query)
                for index, token in enumerate(token_iter):
                    picked = selected_token_rows(context + token + 1, s["ratio"],
                                                 s["topk"], s["kpool"],
                                                 s["always_tail"])
                    sel_positions += picked
                    if picked:
                        sel_tiles += _ceil_div(picked, nt)
                    first_tile = (index * group) // mt
                    last_tile = ((index + 1) * group - 1) // mt
                    for tile_index in range(first_tile,
                                            min(n_qt, last_tile + 1)):
                        tile_picks[tile_index] += picked
            max_kv = max(max_kv, min(request.kv_len, max(1, sel_positions)))
            selected += req_mult * kv_heads * sel_positions
            pairs_c = heads * sel_positions
            pairs += req_mult * pairs_c
            physical += req_mult * pairs_c
            logical += req_mult * 2 * pairs_c * (s["dk"] + s["dv"])
            cube += req_mult * kv_heads * 2 * group * sel_positions * (
                s["cube_qk"] + s["cube_v"])
            vector += req_mult * (7 * pairs_c + selected_q_len * heads * (
                3 * s["ub_v"] + 10))
            fixpipe += req_mult * kv_heads * group * sel_tiles * s["cube_v"] * 4
            tasks += req_mult * n_qt * kv_heads
            kv_tiles += req_mult * kv_heads * sel_tiles
            kv_rows += req_mult * kv_heads * sel_positions
            burst += req_mult * kv_heads * gather_bytes(sel_positions, s["head_kv_row"])
            page_txn += req_mult * kv_heads * sel_positions
            if tile_picks:
                task_cube_max = max(
                    task_cube_max,
                    2 * group * max(tile_picks) * (s["cube_qk"] + s["cube_v"]))
            continue
        eff_win = (min(window, s["topk"]) if window else s["topk"]) if s["topk"] > 0 else window
        unabs = mla and not absorbed and selected_q_len > 1 and s["topk"] == 0
        cqk = align(s["dk"], 16) if unabs else s["cube_qk"]
        cv = s["dv_cube"] if unabs else s["cube_v"]
        req_pairs = request.pairs(window=eff_win, causal=causal)
        pairs += req_mult * req_pairs
        if mla:
            logical += req_mult * 2 * req_pairs * heads * (
                s["qk_width"] + s["v_width"])
        else:
            logical += req_mult * 2 * req_pairs * heads * (
                s["dk"] + s["dv"])
        req_cube = req_tasks = 0
        full_tiles = rows_total // mt if positions is None else 0
        sat_tile = (min(full_tiles, _ceil_div(max(0, eff_win - context - 1) * group, mt))
                    if (positions is None and causal and eff_win > 0) else full_tiles)
        t_period = math.lcm(mt, group * math.lcm(max(1, page), max(1, nt))) // mt if positions is None else 1
        rem_tiles = full_tiles - sat_tile
        full_p, rem_p = divmod(rem_tiles, t_period) if rem_tiles > 0 else (0, 0)
        tile_iter = itertools.chain(
            ((t, 1) for t in range(sat_tile)),
            ((sat_tile + j, full_p + (1 if j < rem_p else 0))
             for j in range(min(rem_tiles, t_period))),
            ((t, 1) for t in range(full_tiles, n_qt)),
        )
        for tile, w in tile_iter:
            if not w:
                continue
            row_start = tile * mt
            row_end = min(rows_total, row_start + mt)
            rows = row_end - row_start
            first_index = row_start // group
            last_index = (row_end - 1) // group
            if positions is None:
                first_token, last_token = first_index, last_index
            else:
                first_token, last_token = (positions[first_index],
                                           positions[last_index])
            if causal:
                hi = context + last_token + 1
                lo = max(0, context + first_token + 1 - eff_win) if eff_win else 0
            else:
                hi = request.kv_len
                lo = max(0, hi - eff_win) if eff_win else 0
            n_rows = max(1, hi - lo)
            n_kt = max(1, _ceil_div(hi, nt) - lo // nt)
            paged, pages = range_burst_bytes(lo, hi, s["head_kv_row"], page=page)
            burst += req_mult * w * kv_heads * paged
            page_txn += req_mult * w * kv_heads * pages
            kv_rows += req_mult * w * kv_heads * n_rows
            kv_tiles += req_mult * w * kv_heads * n_kt
            tasks += req_mult * w * kv_heads
            req_tasks += w * kv_heads
            physical += req_mult * w * kv_heads * rows * n_rows
            tile_c = 2 * rows * n_rows * (cqk + cv)
            cube += req_mult * w * kv_heads * tile_c
            req_cube += w * kv_heads * tile_c
            if n_qt <= s["cores"]:
                task_cube_max = max(task_cube_max, tile_c)
            vector += req_mult * w * kv_heads * (
                7 * rows * n_rows + rows * (3 * s["ub_v"] + 10) * n_kt)
            fixpipe += req_mult * w * kv_heads * rows * cv * 4 * n_kt
        if n_qt > s["cores"] and req_tasks:
            task_cube_max = max(task_cube_max, Fraction(req_cube, req_tasks))
    vector = Fraction(vector)
    use_abs = mla and absorbed
    absorb = 2 * query_rows * s["dn_cube"] * s["rkv_cube"] if use_abs else 0
    expand = 2 * query_rows * s["rkv_cube"] * s["dv_cube"] if use_abs else 0
    weights = (heads * (s["dn_cube"] * s["rkv_cube"]
                        + s["rkv_cube"] * s["dv_cube"]) * s["act"]) if use_abs else 0
    cube_total = cube + absorb + expand
    vector_total = vector + (query_rows * (s["dn_cube"] + s["rkv_cube"]
                                           + s["dv_cube"]) if mla else 0)
    logical += absorb + expand
    # Absorb/expand are uniform per query row; spreading the extra work over
    # the declared tasks keeps a balanced wave equal to the prior mean model
    # while an unequal task carries its own worst-case share.
    if tasks:
        task_cube_max += Fraction(absorb + expand) / tasks
    out_bytes = Fraction(query_rows * s["dv"]) * s["act"]
    kv_logical = Fraction(kv_rows) * s["head_kv_row"]
    return {
        "pairs": pairs, "PAIRS": pairs, "PAIRS_PHYSICAL": physical,
        "query_rows": query_rows, "QUERY_ROWS": query_rows,
        "QUERY_ROWS_MAX": max(1, max_rows), "query_rows_max": max(1, max_rows),
        "MAX_KV_LEN": max(1, max_kv), "max_kv_len": max(1, max_kv),
        "kv_sweeps": kv_rows, "kv_positions": kv_rows, "KV_ROWS": kv_rows,
        "tiles": max(1, kv_tiles), "KV_TILES": max(1, kv_tiles),
        "query_tiles": max(1, tasks), "QUERY_TASKS": max(1, tasks),
        "kv_work": max(1, max_rows * max_kv),
        "KV_BURST_BYTES": burst, "KV_LOGICAL_BYTES": kv_logical,
        "PAGE_TXNS": page_txn, "Q_BYTES": q_bytes,
        "CUBE_FLOPS": cube_total, "CUBE_TASK_MAX": task_cube_max,
        "VECTOR_OPS": vector_total,
        "FIXPIPE_BYTES": Fraction(fixpipe), "OUT_BYTES": out_bytes,
        "ABSORB_FLOPS": absorb, "EXPAND_FLOPS": expand,
        "WEIGHT_BYTES": Fraction(weights), "LOGICAL_FLOPS": logical,
        "SELECTED_ROWS": selected, "COMPRESSED_NEW_ROWS": compressed_new,
        "heads": heads, "kv_heads": kv_heads, "group": group,
        "dk": s["dk"], "dv": s["dv"], "kv_bytes": s["kv_bytes"],
        "flops": logical,
    }


def _split_expressions(ctx: CompiledOperator, static: dict[str, Any],
                       metrics: dict[str, Any]):
    cores, dv, mt = static["cores"], static["dv"], static["mt"]
    splits = Var("SPLITS")
    rows = Var("QUERY_ROWS")
    tasks = Var("QUERY_TASKS")
    total_tasks = Mul((splits, tasks))
    waves = CeilDiv(total_tasks, Const(cores))
    # A wave cannot finish before its longest pinned task.  Split-K divides
    # each declared task, so the per-task maximum also scales by the split.
    per_task = Div(Max((Var("CUBE_FLOPS"),
                        Mul((Var("CUBE_TASK_MAX"), tasks)))), total_tasks)
    cube = Mul((waves, per_task, Const(cores)))
    partial_row = Const(4 * dv + 8)
    partial_write = Mul((Sub(splits, Const(1)), rows, partial_row))
    merge_read = partial_write
    out = Var("OUT_BYTES")
    q_and_weight = Add((Var("Q_BYTES"), Var("WEIGHT_BYTES")))
    dma = Add((Var("KV_BURST_BYTES"), Mul((q_and_weight, splits))))
    main = (
        _demand(ctx, "cube", cube),
        _demand(ctx, "mte2", dma),
        _demand(ctx, "mte1", dma),
        _demand(ctx, "mte3", partial_write),
        _demand(ctx, "hbm", Add((Var("KV_BURST_BYTES"),
                                 Mul((q_and_weight, splits)), partial_write))),
        _demand(ctx, "fixpipe", Var("FIXPIPE_BYTES")),
        _demand(ctx, "vector", Var("VECTOR_OPS")),
    )
    merge_vector = Add((Mul((Sub(splits, Const(1)), rows, Const(3 * dv + 9))),
                        Mul((rows, Const(dv + 1)))))
    merge = (
        _demand(ctx, "mte2", merge_read),
        _demand(ctx, "mte3", out),
        _demand(ctx, "hbm", Add((merge_read, out))),
        _demand(ctx, "vector", merge_vector),
    )
    tail = Sub(total_tasks, Mul((Sub(waves, Const(1)), Const(cores))))
    derived = (
        ("causal_pairs", Var("PAIRS")),
        ("query_rows", rows),
        ("kv_positions", Var("KV_ROWS")),
        ("KV_ROWS", Var("KV_ROWS")),
        ("KV_LOGICAL_BYTES", Var("KV_LOGICAL_BYTES")),
        ("logical_flops", Var("LOGICAL_FLOPS")),
        ("physical_pairs", Var("PAIRS_PHYSICAL")),
        ("physical_flops", Var("CUBE_FLOPS")),
        ("tiles", Mul((Var("KV_TILES"), splits))),
        ("tile_waves", CeilDiv(Mul((Var("KV_TILES"), splits)), Const(cores))),
        ("waves", waves), ("kv_splits", splits),
        ("cube_task_max", Var("CUBE_TASK_MAX")),
        ("cube_per_task", per_task),
        ("tail_cores", tail),
        ("active_cores", Min((total_tasks, Const(cores)))),
        ("query_tile_steps", tasks),
        ("query_tile_step", CeilDiv(Var("QUERY_ROWS_MAX"), Const(mt))),
        ("kv_tile_steps", Var("KV_TILES")),
        ("page_transactions", Var("PAGE_TXNS")),
        ("kv_burst_bytes", Var("KV_BURST_BYTES")),
        ("q_bytes", Var("Q_BYTES")), ("query_bytes", Var("Q_BYTES")),
        ("Q_BYTES", Var("Q_BYTES")),
        ("output_bytes", out), ("OUT_BYTES", out),
        ("kv_burst_amplification", Div(Var("KV_BURST_BYTES"),
                                       Max((Var("KV_LOGICAL_BYTES"), Const(1))))),
        ("split_partial_bytes", Mul((Const(2), partial_write))),
        ("split_merge_vector", merge_vector),
        ("kv_reads_reuse_factor", CeilDiv(rows, Const(mt))),
        # Raw pre-service single-tile roofline balance diagnostics; actual
        # segment axes are QUERY_TILES/KV_TILES/SPLITS and runtime service
        # transitions use RIDGE_EFFECTIVE_TOKENS and VEC_EFFECTIVE_TOKENS.
        ("cube_mte2_balance_rows", Const(static["balance_cube_mte2"])),
        ("cube_mte2_wave_rows",
         Const(static["balance_cube_mte2"] / static["cores"])),
        ("vector_mte2_balance_rows", Const(static["balance_vec_mte2"])),
        ("balance_cube_mte2", Const(static["balance_cube_mte2"])),
        ("balance_vec_mte2", Const(static["balance_vec_mte2"])),
        ("cube_vector_ratio", Const(static["cube_vector_ratio"])),
        ("raw_roofline_balance_cube_mte2", Const(static["balance_cube_mte2"])),
        ("raw_roofline_balance_vec_mte2", Const(static["balance_vec_mte2"])),
        ("raw_roofline_cube_vector_ratio", Const(static["cube_vector_ratio"])),
        ("group_heads", Const(static["group"])),
        ("query_heads", Const(static["heads"])),
        ("kv_heads", Const(static["kv_heads"])),
        ("compressed_rows", Var("COMPRESSED_NEW_ROWS")),
        ("selected_rows", Var("SELECTED_ROWS")),
        ("dk_cube", Const(static["cube_qk"])),
        ("dv_cube", Const(static["cube_v"])),
        ("dv_ub", Const(static["ub_v"])),
        ("kv_row_bytes", Const(static["kv_bytes"])),
        ("q_row_width", Const(static["q_width"])),
    )
    return main, merge, derived, out


def _attention_axes(static: dict[str, Any], ctx: CompiledOperator) -> tuple[SegmentAxis, ...]:
    policy = ctx.policy.split_policy()
    return (
        SegmentAxis("QUERY_TILES", "query_rows_max",
                    StepGuard("query_tiles", "query_rows_max", static["mt"]),
                    "max_query_tiles"),
        SegmentAxis("KV_TILES", "max_kv_len",
                    StepGuard("kv_tiles", "max_kv_len", static["nt"]),
                    "max_kv_tiles"),
        SegmentAxis("SPLITS", "kv_work",
                    SplitKGuard(policy, coordinate="kv_work"), "max_work"),
    )


def _compile_attention(ctx: CompiledOperator, operator) -> FamilyTemplate:
    static = _static_attention(ctx, operator, mla=False)
    axes = _attention_axes(static, ctx)

    def binder(geometry, state):
        return _attention_metrics(ctx, operator, geometry, mla=False)

    main, merge, derived, out = _split_expressions(ctx, static, {})
    return FamilyTemplate(
        axes=axes, stages=(main, merge),
        materialization=(("attention_output", out),),
        internal=(), residency=Const(0), derived=derived,
        regime_exprs=(), binder=binder,
        annotations=(("family", "attention"),
                     ("branch", static["branch"] or "dense"),
                     ("group", str(static["group"])),
                     ("dk", str(static["dk"])), ("dv", str(static["dv"])),
                     ("dk_cube", str(static["cube_qk"])),
                     ("dv_cube", str(static["cube_v"])),
                     ("window", str(int(operator.theta.find("window", 0) or 0))),
                     ("causal", str(int(operator.theta.find("causal", 1)))),
                     ("page", str(static["page"])),
                     ("balance_semantics", "raw_pre_service_roofline_diagnostic"),
                     ("timing_selector_coordinates", "RIDGE_EFFECTIVE_TOKENS,VEC_EFFECTIVE_TOKENS")),
    )


def _compile_mla(ctx: CompiledOperator, operator) -> FamilyTemplate:
    static = _static_attention(ctx, operator, mla=True)
    axes = _attention_axes(static, ctx)

    def binder(geometry, state):
        return _attention_metrics(ctx, operator, geometry, mla=True)

    main, merge, derived, out = _split_expressions(ctx, static, {})
    groups = max(0, int(operator.theta.find("o_groups", 0) or 0))
    o_rank = max(0, int(operator.theta.find("o_lora_rank", 0) or 0))
    group_width = (static["heads"] * static["dv"] // groups) if groups else 0
    o_a = 2 * groups * o_rank * group_width
    o_b = 2 * group_width * groups * o_rank
    derived = derived + (
        ("absorb_flops", Var("ABSORB_FLOPS")),
        ("v_expand_flops", Var("EXPAND_FLOPS")),
        ("latent_width", Const(static["rkv"] + static["dr"])),
        ("grouped_o_a_flops", Mul((Var("QUERY_ROWS"), Const(o_a)))),
        ("grouped_o_b_flops", Mul((Var("QUERY_ROWS"), Const(o_b)))),
        ("absorb_weight_bytes", Const(
            static["heads"] * static["dn_cube"] * static["rkv_cube"] * static["act"])),
        ("expand_weight_bytes", Const(
            static["heads"] * static["rkv_cube"] * static["dv_cube"] * static["act"])),
    )
    return FamilyTemplate(
        axes=axes, stages=(main, merge),
        materialization=(("mla_attention_output", out),),
        internal=(), residency=Const(0), derived=derived,
        regime_exprs=(), binder=binder,
        annotations=(("family", "mla"), ("rkv", str(static["rkv"])),
                     ("dn", str(static["dn"])), ("dr", str(static["dr"])),
                     ("dv", str(static["dv"])),
                     ("latent_width", str(static["rkv"] + static["dr"])),
                     ("absorb", "1"),
                     ("balance_semantics", "raw_pre_service_roofline_diagnostic"),
                     ("timing_selector_coordinates", "RIDGE_EFFECTIVE_TOKENS,VEC_EFFECTIVE_TOKENS")),
    )


# ---------------------------------------------------------------------------
# Auxiliary families (vision, draft) and the recurrent dispatch
# ---------------------------------------------------------------------------


def _compile_recurrent(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    return compile_recurrent(ctx, operator)


def _compile_vision(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    width = int(theta.require("width"))
    layers = int(theta.find("layers", 0) or 0)
    heads = int(theta.find("heads", 0) or 0)
    patch = int(theta.find("patch_width", width) or width)
    mt = int(theta.require("mt"))
    axes = (SegmentAxis("MEDIA_TILES", "media_tokens",
                        StepGuard("media_tiles", "media_tokens", mt,
                                  lower_bound=0), "max_media_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        tokens = int(geometry.media_tokens)
        flops = (2 * tokens * patch * width * max(1, layers)
                 + 2 * tokens * width * width)
        return {"media": tokens, "vision_flops": flops}

    media = Var("MEDIA")
    flops = Add((Mul((Const(2), media, Const(patch), Const(width), Const(max(1, layers)))),
                 Mul((Const(2), media, Const(width), Const(width)))))
    traffic = Mul((media, Const(width * 2)))
    main = (
        _demand(ctx, "cube", flops),
        _demand(ctx, "mte2", traffic),
        _demand(ctx, "mte3", traffic),
        _demand(ctx, "hbm", Add((traffic, traffic))),
    )
    return FamilyTemplate(
        axes=axes, stages=(main,), materialization=(("media_hidden", traffic),),
        internal=(), residency=Const(0),
        derived=(("flops", flops), ("media_tokens", media)),
        regime_exprs=(), binder=binder,
        annotations=(("family", "vision"),),
    )


def _compile_draft(ctx: CompiledOperator, operator: BoundOperator) -> FamilyTemplate:
    theta = operator.theta
    mt = int(theta.require("mt"))
    extra = dict(theta.extras)
    draft_len = int(extra.get("draft_len", 1) or 1)
    verify_tokens = int(extra.get("verify_tokens", draft_len + 1) or 1)
    vocab = int(theta.find("N", 1) or 1)
    hidden = int(theta.find("K", 1) or 1)
    weight_total = _frac(theta.find("weight_total", 0.0) or 0.0)
    axes = (SegmentAxis("TOKEN_TILES", "tokens", StepGuard("draft_tiles", "tokens", mt),
                        "max_tokens"),)

    def binder(geometry: WorkloadGeometry, state: StateView) -> dict[str, Any]:
        return {"draft_len": draft_len, "verify_tokens": verify_tokens,
                "draft_vocab": vocab, "draft_hidden": hidden,
                "ACCEPT": state.acceptance.accepted,
                "COMMIT": state.acceptance.progress,
                "accepted": state.acceptance.accepted,
                "commit_progress": state.acceptance.progress}

    tokens = Var("T")
    if operator.name == "draft_heads":
        flops = Mul((Const(2), tokens, Const(draft_len * hidden * hidden)))
        traffic = Mul((tokens, Const(hidden * 2)))
        main = (
            _demand(ctx, "cube", flops),
            _demand(ctx, "mte2", traffic),
            _demand(ctx, "mte3", traffic),
            _demand(ctx, "hbm", Add((traffic, traffic))),
        )
        derived = (("draft_flops", flops), ("draft_len", Const(draft_len)))
    elif operator.name == "target_verify":
        flops = Mul((Const(2), tokens, Const(verify_tokens * hidden * vocab)))
        traffic = Mul((Const(1), Const(weight_total)))
        logits = Mul((tokens, Const(verify_tokens * vocab * 2)))
        main = (
            _demand(ctx, "cube", flops),
            _demand(ctx, "mte2", traffic),
            _demand(ctx, "mte3", logits),
            _demand(ctx, "hbm", Add((traffic, logits))),
        )
        derived = (("verify_flops", flops), ("verify_tokens", Const(verify_tokens)),
                   ("acceptance_discount", Const(0)))
    else:
        traffic = Mul((tokens, Const(hidden * 2)))
        main = (
            _demand(ctx, "mte2", traffic),
            _demand(ctx, "mte3", traffic),
            _demand(ctx, "hbm", Add((traffic, traffic))),
        )
        derived = (("commit_bytes", traffic), ("commit_progress", Var("COMMIT")),
                   ("acceptance_discount", Const(0)))
    return FamilyTemplate(
        axes=axes, stages=(main,), materialization=((_out_name(ctx), traffic),),
        internal=(), residency=Const(weight_total), derived=derived,
        regime_exprs=(("ACCEPT", Var("ACCEPT")), ("COMMIT", Var("COMMIT"))),
        binder=binder, annotations=(("family", "draft"),),
    )


__all__ = [
    "_compile_attention", "_compile_draft", "_compile_mla",
    "_compile_recurrent", "_compile_vision",
    "compile_compression", "compile_indexer", "index_scanned_rows",
    "k1_new_rows", "selected_token_rows",
]
