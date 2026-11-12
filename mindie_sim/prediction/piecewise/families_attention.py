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
    minimum burst.  Incremental q=1 page attention also moves one int32 block
    index per distinct ``(batch, block)`` on HBM.  Repeated ``GetValue`` calls
    stay a lookup count: the scalar rate is not declared.  Split-K adds exactly
    ``2*(P-1)*B*H_q*q*(4*d_v+8)`` bytes for partial output/LSE traffic plus
    vector merge.  The core-wave tail ``ceil(J/C)`` cannot borrow idle cores:
    it caps elapsed cube throughput through ``HW_RATE_CUBE = physical/scheduled``
    and never turns idle cores into FLOPs. Launch width does not scale HBM.
    An A5 q=1 pfa-merge route prices one KV split. Its owner pipeline is the
    excess over that unsplit body. Generic split partials stay off that route.
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

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .compiler import (
    CompiledOperator, FamilyTemplate, SegmentAxis, _demand, _frac, _out_name,
)
from .expressions import Add, CeilDiv, Const, Div, Max, Mul, Sub, Var
from .guards import SplitKGuard, StepGuard
from .families_attention_ifa import (
    attention_notes, attention_table_mode, bind_fia_launches,
    fia_program_duration, schedule_attention,
)
from .families_recurrent_indexer import (
    BURST_BYTES, DESCRIPTOR_BYTES, PAGE_ROWS, align, compile_compression,
    compile_indexer, compile_recurrent, gather_bytes, index_scanned_rows,
    k1_new_rows, range_burst_bytes, selected_token_rows,
)


def _ceil_div(value: int, divisor: int) -> int:
    return -(-int(value) // int(divisor))


# GetBlockIdx returns int32.  L1 K-split reissues that load once per 128-wide
# K slice when the head is wider than 256.  V stays one walk.
_BLOCK_INDEX_BYTES = 4
_L1_K_SPLIT = 128
_L1_K_SPLIT_MIN = 256



# ---------------------------------------------------------------------------
# Attention geometry
# ---------------------------------------------------------------------------


def _static_attention(ctx: CompiledOperator, operator, *, mla: bool) -> dict[str, Any]:
    theta = operator.theta
    heads = max(1, int(theta.find("q", 1) or 1))
    kv_heads = max(1, int(theta.find("kv", 1) or 1))
    # GQA group is ceil(H_q / H_kv), including a ragged tail and TP replication.
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
    # Ordinary KV stores independent K and V. Latent MLA stores one (r_kv + d_r) row.
    default_kv = (Fraction(kv_heads) * qk_width if mla
                  else Fraction(kv_heads) * (qk_width + v_width)) * act
    kv_bytes = _frac(theta.find("kv_bytes", default_kv) or default_kv)
    mt = max(1, int(theta.require("mt")))
    nt = max(1, int(theta.require("nt")))
    cores = max(1, int(theta.require("cores")))
    page = max(1, int(theta.find("page", 0) or PAGE_ROWS))
    ub_multiple = max(1, _ceil_div(32, max(1, int(act))))
    branch = str(theta.find("branch", "") or "")
    declared_ratio = int(theta.find("compress_ratio", 0) or 0)
    if branch == "compressed" and declared_ratio <= 0:
        # A compressed launch with no declared compression is a fabricated
        # branch, not a window launch.  The builder omits it; refuse a
        # malformed template instead of silently coercing 0 -> 1.
        raise ModelingError(
            "a2_compressed_branch_requires_positive_ratio:"
            f"{ctx.name}:layer={ctx.layer}")
    ratio = max(1, declared_ratio)
    topk = max(0, int(theta.find("index_topk", 0) or 0))
    kpool = max(1, int(theta.find("kpool", 1) or 1))
    always_tail = bool(int(theta.find("kpool_always_select_tail", 0) or 0))
    kv_dtype = str(theta.find("kv_dtype", "bf16") or "bf16").lower()
    cp = max(1, int(theta.find("cp", 1) or 1))
    cp_plan = str(theta.find("cp_plan", "none") or "none")
    cp_interleave = max(1, int(theta.find("cp_interleave", 128) or 128))
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
        "always_tail": always_tail, "kv_dtype": kv_dtype,
        "cp": cp, "cp_plan": cp_plan, "cp_interleave": cp_interleave,
        "balance_cube_mte2": balance_cube_mte2,
        "balance_vec_mte2": balance_vec_mte2,
        "cube_vector_ratio": cube_vector_ratio,
        "dequant_ops": (5 if kv_dtype == "int8" else 4),
        "packed_sparse": bool(mla and kv_dtype in {"int8", "fp8", "fp8_e4m3"}
                              and topk > 0 and heads == 64 and rkv == 512
                              and dv == 256),
    }


def _packed_indexed_hbm(keys_total: int, heads: int, latent: int) -> Fraction:
    """Source HBM bytes of the packed-int8 indexed attention phase chain.

    Uses the retained arch22 phase ledger (gather/publish/QK/softmax/PV/
    nupdate/partial).  This is source work, not a per-tile latency floor.
    """
    from mindie_sim.prediction.indexed_quant_attention import phases
    total = 0
    tile = 512
    for start in range(0, int(keys_total), tile):
        keys = min(tile, int(keys_total) - start)
        for _name, work in phases(keys, int(heads), int(latent), 0,
                                  first=start == 0,
                                  last=start + keys == int(keys_total),
                                  scratch_policy="offchip", operand_modes=True):
            total += work.resources.get("hbm", 0)
    return Fraction(total)


def _sparse_rows_for_request(request, topk, ratio, kpool=1, always_tail=False):
    """Sum causal selected rows over this request's real positions only."""
    total = 0
    for offset in range(int(request.q_len)):
        total += selected_token_rows(int(request.context) + offset + 1,
                                     ratio, topk, kpool, always_tail)
    return total


def _dcp_sparse_local_rows(available: int, picked: int, cp: int, block: int) -> int:
    """Worst-rank selected rows for one causal compressed-row range.

    The retained SFA DCP source keeps a replicated-view top-k index only when
    that KV block belongs to the local rank (``_remap_sparse_indices``:
    ``floor(index / interleave) % dcp == rank``).  The retained case does not
    carry the actual top-k index ids, so this prices the declared
    uniform-within-selected-range basis: the ``picked`` rows are spread over
    the exact ``available`` causal compressed rows, and block ownership uses
    the same compressed-KV block as ``catalog/cache.py`` (``interleave //
    ratio``).  It returns the active/worst rank's ceil count; no per-token
    arrays are stored.
    """
    available, picked = int(available), int(picked)
    if available <= 0 or picked <= 0:
        return 0
    # The local SFA reads compressed KV rows, so a token-level kpool/tail count
    # cannot select more rows than the completed compressed range holds.
    picked = min(picked, available)
    from mindie_sim.catalog.context_parallel import owned_positions
    owned = max(owned_positions(0, available - 1, cp, block, rank)
                for rank in range(cp))
    return (picked * owned + available - 1) // available


def _attention_metrics(ctx: CompiledOperator, operator, geometry,
                       *, mla: bool = False, paged_table: str = "auto") -> dict[str, Any]:
    s = _static_attention(ctx, operator, mla=mla)
    heads, kv_heads, group = s["heads"], s["kv_heads"], s["group"]
    mt, nt, page = s["mt"], s["nt"], s["page"]
    window = max(0, int(operator.theta.find("window", 0) or 0))
    causal = bool(operator.theta.find("causal", 1))
    compressed = (not mla) and s["branch"] == "compressed"
    packed_kv = s["kv_dtype"] in {"int8", "fp8", "fp8_e4m3"}
    # The quantized matmul/weight activation is not the FIA boundary dtype.
    # Ordinary non-MLA attention carries BF16 Q/output even when the model
    # executes quantized weights.  The gathered lane width is the physical
    # leading row count: C1 has one valid request padded to two graph rows,
    # C8 has two real rows.  Valid KV work stays per real request.
    lanes_static = geometry.token_lanes()
    # Only the selected A5 non-MLA paged d256 route gets the physical-row/BF16
    # boundary correction.  A2, dk128 and raw/other templates keep their prior
    # priced launches.  This is the QG9 source gate, not a default.
    declared_io_dtype = operator.theta.find("kv_dtype", None)
    ordinary_io = (
        (not mla) and (not compressed)
        and str(getattr(ctx.supply, "arch_family", "") or "") == "A5"
        and int(operator.theta.find("page_attention", 0) or 0) == 1
        and declared_io_dtype is not None
        and str(declared_io_dtype).lower() in {"bf16", "fp16"}
        and int(s["dk"]) != 128 and int(s["dv"]) > 128 and int(s["group"]) < 64)
    io_rows = (max(1, int(lanes_static["gathered_tokens"]),
                   int(lanes_static["valid_tokens"]))
               if ordinary_io else 0)
    io_act = (Fraction(dtype_bytes(str(declared_io_dtype)))
              if ordinary_io else s["act"])
    valid_query_tokens = int(lanes_static["valid_tokens"]) if ordinary_io else 0
    pairs = query_rows = max_rows = max_kv = tasks = kv_tiles = kv_rows = 0
    burst = Fraction(0)
    packed_hbm = Fraction(0)
    page_txn = physical = cube = fixpipe = selected = compressed_new = 0
    from collections import Counter
    vector = 0
    q_bytes = Fraction(0)
    logical = 0
    task_cube_max = Fraction(0)
    absorbed = int(operator.theta.find("absorb", 1) or 0) != 0
    # Distinct (batch, block) int32 slots are the HBM lower bound. Head and
    # K/V reissue stay an unpriced lookup count.
    block_entries = 0
    block_lookups = 0
    span_by_request: dict = {}
    k_copies = (1 if s["dk"] <= _L1_K_SPLIT_MIN
                else _ceil_div(s["dk"], _L1_K_SPLIT))
    charge_table, route_q1 = attention_table_mode(
        ctx, operator, geometry, s, mla=mla, compressed=compressed,
        paged_table=paged_table)
    for request, req_mult in Counter(geometry.requests).items():
        query, context = request.q_len, request.context
        span_lo = span_hi = None
        walk_pages = 0
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
        q_bytes += req_mult * selected_q_len * heads * s["q_width"] * (io_act if ordinary_io else s["act"])
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
            # DCP shards the SFA KV cache. The replicated indexer/TopK stays
            # global, then the retained source remaps each top-k index to the
            # local rank when its KV block is owned (`_remap_sparse_indices`:
            # floor(index / interleave) % dcp == rank). Actual top-k ids are
            # unavailable in the retained case, so price the per-query selected
            # range on the exact local block ownership; DSA-CP is a query-token
            # plan and never takes this KV partition.
            cp = max(1, int(s["cp"]))
            cp_active = cp > 1 and str(s["cp_plan"]) == "dcp"
            if cp_active:
                block = max(1, int(s["cp_interleave"]) // max(1, int(s["ratio"])))
                local_sel_positions = local_sel_tiles = 0
                local_tile_picks = [0] * n_qt
                token_iter = positions if positions is not None else range(query)
                for index, token in enumerate(token_iter):
                    position = context + token + 1
                    picked = selected_token_rows(
                        position, s["ratio"], s["topk"], s["kpool"],
                        s["always_tail"])
                    local_picked = _dcp_sparse_local_rows(
                        position // max(1, int(s["ratio"])), picked, cp, block)
                    local_sel_positions += local_picked
                    if local_picked:
                        local_sel_tiles += _ceil_div(local_picked, nt)
                    first_tile = (index * group) // mt
                    last_tile = ((index + 1) * group - 1) // mt
                    for tile_index in range(first_tile,
                                            min(n_qt, last_tile + 1)):
                        local_tile_picks[tile_index] += local_picked
            else:
                local_sel_positions = sel_positions
                local_sel_tiles = sel_tiles
                local_tile_picks = tile_picks
            max_kv = max(max_kv, min(request.kv_len, max(1, local_sel_positions)))
            if packed_kv and s["topk"] > 0:
                packed_hbm += req_mult * _packed_indexed_hbm(
                    local_sel_positions, heads, s["rkv"])
            selected += req_mult * kv_heads * local_sel_positions
            pairs_c = heads * local_sel_positions
            pairs += req_mult * pairs_c
            physical += req_mult * pairs_c
            logical += req_mult * 2 * pairs_c * (s["dk"] + s["dv"])
            cube += req_mult * kv_heads * 2 * group * local_sel_positions * (
                s["cube_qk"] + s["cube_v"])
            vector += req_mult * (7 * pairs_c + selected_q_len * heads * (
                3 * s["ub_v"] + 10))
            fixpipe += req_mult * kv_heads * group * local_sel_tiles * s["cube_v"] * 4
            tasks += req_mult * n_qt * kv_heads
            kv_tiles += req_mult * kv_heads * local_sel_tiles
            kv_rows += req_mult * kv_heads * local_sel_positions
            burst += req_mult * kv_heads * gather_bytes(local_sel_positions, s["head_kv_row"])
            page_txn += req_mult * kv_heads * local_sel_positions
            if local_tile_picks:
                task_cube_max = max(
                    task_cube_max,
                    2 * group * max(local_tile_picks) * (s["cube_qk"] + s["cube_v"]))
            continue
        eff_win = (min(window, s["topk"]) if window else s["topk"]) if (
            mla and s["topk"] > 0) else window
        unabs = mla and not absorbed and selected_q_len > 1 and s["topk"] == 0
        cqk = align(s["dk"], 16) if unabs else s["cube_qk"]
        cv = s["dv_cube"] if unabs else s["cube_v"]
        global_pairs = request.pairs(window=eff_win, causal=causal)
        cp_active = (s["cp"] > 1 and s["cp_plan"] in {"dcp", "dsa_cp"}
                     and not request.query_ranges)
        if cp_active:
            from mindie_sim.catalog.context_parallel import split_causal_pairs
            local_split = split_causal_pairs(
                request.q_len, request.context, s["cp"], s["cp_interleave"],
                window=eff_win, noncausal=not causal)
            local_pairs = int(local_split.max_rank)
            share = (Fraction(local_pairs, global_pairs)
                     if global_pairs else Fraction(0))
        else:
            local_pairs, share = global_pairs, Fraction(1)
        req_pairs = local_pairs
        kv_mult = req_mult * share
        if mla and packed_kv and s["topk"] > 0:
            packed_hbm += req_mult * _packed_indexed_hbm(
                req_pairs, heads, s["rkv"])
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
            burst += kv_mult * w * kv_heads * paged
            page_txn += kv_mult * w * kv_heads * pages
            if charge_table:
                if span_lo is None:
                    span_lo, span_hi = lo, hi
                else:
                    span_lo = min(span_lo, lo)
                    span_hi = max(span_hi, hi)
                walk_pages += w * pages
            kv_rows += kv_mult * w * kv_heads * n_rows
            kv_tiles += kv_mult * w * kv_heads * n_kt
            tasks += kv_mult * w * kv_heads
            req_tasks += w * kv_heads
            physical += kv_mult * w * kv_heads * rows * n_rows
            tile_c = 2 * rows * n_rows * (cqk + cv)
            cube += kv_mult * w * kv_heads * tile_c
            req_cube += w * kv_heads * tile_c
            if n_qt <= s["cores"]:
                task_cube_max = max(task_cube_max, tile_c)
            vector += kv_mult * w * kv_heads * (
                7 * rows * n_rows + rows * (3 * s["ub_v"] + 10) * n_kt)
            fixpipe += kv_mult * w * kv_heads * rows * cv * 4 * n_kt
        if n_qt > s["cores"] and req_tasks:
            task_cube_max = max(task_cube_max, Fraction(req_cube, req_tasks))
        if charge_table and span_lo is not None and span_hi > span_lo:
            _, span_pages = range_burst_bytes(
                span_lo, span_hi, s["head_kv_row"], page=page)
            block_entries += kv_mult * span_pages
            block_lookups += (kv_mult * kv_heads * walk_pages
                              * (k_copies + 1))
            if route_q1:
                span_by_request[request] = span_hi - span_lo
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
    # Spread absorb/expand across declared tasks so a balanced wave keeps the mean.
    if tasks:
        task_cube_max += Fraction(absorb + expand) / tasks
    # Output rows follow the physical leading rows; KV/QK/PV work above still
    # follows the valid selected rows.  No second cache/history is invented.
    physical_query_rows = io_rows * heads if ordinary_io else query_rows
    if ordinary_io:
        q_bytes = Fraction(physical_query_rows * s["q_width"]) * io_act
    out_bytes = (Fraction(physical_query_rows * s["dv"]) * io_act
                 if ordinary_io else Fraction(query_rows * s["dv"]) * s["act"])
    kv_logical = Fraction(kv_rows) * s["head_kv_row"]
    mte1_bytes = Fraction(0)
    if s.get("packed_sparse"):
        # R64 SFA source: Q=cat(512,64) BF16, packed KV row656, BF16 QK/PV.
        # Sparse owner item G64: 32 AIC owners walk contiguous 32 S1 queries;
        # the critical owner is the final contiguous prefix, not mean triangular
        # work.  This is a source upper bound when only declared topk is known.
        q_tokens = sum(int(req.q_len) for req in geometry.requests)
        core_count = max(1, int(s["cores"]))
        per_core = max(1, -(-q_tokens // core_count))
        rows = sum(_sparse_rows_for_request(req, s["topk"], s["ratio"],
                                            s["kpool"], s["always_tail"])
                   for req in geometry.requests)
        per_query = [selected_token_rows(
            int(req.context) + offset + 1, s["ratio"], s["topk"],
            s["kpool"], s["always_tail"])
            for req in geometry.requests for offset in range(int(req.q_len))]
        aligned = [((value + 15) // 16) * 16 for value in per_query]
        max_core_pairs = max(
            (sum(aligned[start:start + per_core])
             for start in range(0, len(aligned), per_core)), default=0)
        q_bytes = Fraction(query_rows * 576 * 2)
        out_bytes = Fraction(query_rows * 512 * 2)
        packed = Fraction(rows * 656)
        packed_hbm = packed + q_bytes
        cube_total = Fraction(2 * rows * heads * (576 + 512))
        task_cube_max = Fraction(2 * max_core_pairs * heads * (576 + 512))
        tile_counts = [((value + 127) // 128) for value in per_query]
        first_tiles = len(per_query)
        mid_tiles = sum(max(0, count - 2) for count in tile_counts)
        last_tiles = first_tiles
        single_tiles = sum(1 for count in tile_counts if count == 1)
        multi_last = max(0, last_tiles - single_tiles)
        v2_vector = (mid_tiles * 2 * s["heads"] * 512
                     + multi_last * 3 * s["heads"] * 512
                     + single_tiles * s["heads"] * 512)
        vector_total = Fraction(rows * 512 * s.get("dequant_ops", 4) + v2_vector)
        mte1_bytes = Fraction(0)
        for value in per_query:
            remaining = value
            while remaining > 0:
                rows = min(128, remaining)
                aligned = ((rows + 15) // 16) * 16
                mte1_bytes += 64 * 576 * 2 + aligned * 576 * 2
                mte1_bytes += 64 * aligned * 2 + 512 * aligned * 2
                remaining -= rows
        fixpipe = (sum(4 * 64 * (((value + 7) // 8) * 8) for value in per_query)
                   + sum(4 * 64 * 512 * ((value + 127) // 128)
                         for value in per_query))
        kv_rows = rows
        l1_bytes = Fraction(rows * 576 * 2)
        burst = packed
        selected = rows
        kv_logical = l1_bytes
        absorb = 0
        expand = 0
        weights = 0
        logical = Fraction(2 * rows * heads * (576 + 512))
    scheduled = schedule_attention(ctx, operator, geometry, s, mla, {
        "query_rows": query_rows, "tasks": tasks, "cube_total": cube_total,
        "task_cube_max": task_cube_max, "max_rows": max_rows, "max_kv": max_kv,
        "span_by_request": span_by_request, "route_q1": route_q1,
        "window": window, "q_bytes": q_bytes, "weights": Fraction(weights),
        "burst": burst, "block_bytes": block_entries * _BLOCK_INDEX_BYTES,
        "out_bytes": out_bytes, "vector_total": vector_total,
        "fixpipe": fixpipe, "packed_hbm": packed_hbm,
        "io_rows": io_rows, "io_act": io_act,
        "physical_query_rows": physical_query_rows,
        "valid_query_tokens": valid_query_tokens,
    })
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
        "PACKED_HBM": packed_hbm,
        "PAGE_TXNS": page_txn, "Q_BYTES": q_bytes,
        "CUBE_FLOPS": cube_total, "CUBE_TASK_MAX": task_cube_max,
        "VECTOR_OPS": vector_total,
        "MTE1_BYTES": mte1_bytes,
        "FIXPIPE_BYTES": Fraction(fixpipe), "OUT_BYTES": out_bytes,
        "ABSORB_FLOPS": absorb, "EXPAND_FLOPS": expand,
        "WEIGHT_BYTES": Fraction(weights), "LOGICAL_FLOPS": logical,
        "SELECTED_ROWS": selected, "COMPRESSED_NEW_ROWS": compressed_new,
        "BLOCK_TABLE_ENTRIES": block_entries,
        "BLOCK_TABLE_BYTES": block_entries * _BLOCK_INDEX_BYTES,
        "BLOCK_TABLE_LOOKUPS": block_lookups,
        "heads": heads, "kv_heads": kv_heads, "group": group,
        "dk": s["dk"], "dv": s["dv"], "kv_bytes": s["kv_bytes"],
        "flops": logical,
        **scheduled,
    }


def _split_expressions(ctx: CompiledOperator, static: dict[str, Any],
                       metrics: dict[str, Any]):
    cores, mt = static["cores"], static["mt"]
    splits = Var("EFFECTIVE_SPLITS")
    rows = Var("QUERY_ROWS")
    tasks = Var("QUERY_TASKS")
    total_tasks = Mul((splits, tasks))
    waves = CeilDiv(total_tasks, Const(cores))
    # A wave's longest task bounds elapsed cube time. Reported cube work stays physical.
    per_task = Div(Max((Var("CUBE_FLOPS"),
                        Mul((Var("CUBE_TASK_MAX"), tasks)))), total_tasks)
    cube = Var("CUBE_FLOPS")
    partial_write = Var("PARTIAL_BYTES")
    out = Var("OUT_BYTES")
    dma = Var("DMA_BYTES")
    packed_main = (static.get("kv_dtype") in {"int8", "fp8", "fp8_e4m3"}
                   and static.get("topk", 0) > 0)
    main = (
        _demand(ctx, "cube", cube),
        _demand(ctx, "mte2", dma),
        _demand(ctx, "mte1", (Var("MTE1_BYTES") if packed_main else dma)),
        _demand(ctx, "mte3", partial_write),
        _demand(ctx, "hbm", (Var("PACKED_HBM")
                             if packed_main
                             else Add((dma, partial_write, Var("BLOCK_TABLE_BYTES"))))),
        _demand(ctx, "fixpipe", Var("FIXPIPE_BYTES")),
        _demand(ctx, "vector", Var("VECTOR_OPS")),
    )
    merge_vector = Var("MERGE_VECTOR_OPS")
    merge = (
        _demand(ctx, "mte2", partial_write),
        _demand(ctx, "mte3", out),
        _demand(ctx, "hbm", Add((partial_write, out))),
        _demand(ctx, "vector", merge_vector),
    )
    tail = Sub(total_tasks, Mul((Sub(waves, Const(1)), Const(cores))))
    derived = (
        ("causal_pairs", Var("PAIRS")),
        ("query_rows", rows),
        ("kv_positions", Var("KV_ROWS")),
        ("logical_flops", Var("LOGICAL_FLOPS")),
        ("physical_pairs", Var("PAIRS_PHYSICAL")),
        ("physical_flops", Var("CUBE_FLOPS")),
        ("tiles", Mul((Var("KV_TILES"), splits))),
        ("tile_waves", CeilDiv(Mul((Var("KV_TILES"), splits)), Const(cores))),
        ("waves", waves), ("kv_splits", splits),
        ("cube_task_max", Var("CUBE_TASK_MAX")),
        ("cube_per_task", per_task),
        ("cube_schedule_work", Var("CUBE_SCHEDULE_WORK")),
        ("cube_schedule_scale", Var("HW_RATE_CUBE")),
        ("tail_cores", tail),
        ("active_cores", Var("ACTIVE_CORES")),
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
        ("block_table_entries", Var("BLOCK_TABLE_ENTRIES")),
        ("block_table_bytes", Var("BLOCK_TABLE_BYTES")),
        ("block_table_lookups", Var("BLOCK_TABLE_LOOKUPS")),
        ("host_used_aic", Var("HOST_USED_AIC")),
        ("host_s2_tiles", Var("HOST_S2_TILES")),
        ("intra_owner_pipeline_ms", Var("INTRA_OWNER_PIPELINE_MS")),
        ("intra_owner_exposed_ms", Var("INTRA_OWNER_EXPOSED_MS")),
        ("d256_host_selected", Var("D256_HOST_SELECTED")),
        ("fia_launch_count", Var("FIA_LAUNCH_COUNT")),
        ("fia_serial_ms", Var("FIA_SERIAL_MS")),
        ("lane_scheduled", Var("LANE_SCHEDULED")),
        ("lane_gathered", Var("LANE_GATHERED")),
        ("lane_valid", Var("LANE_VALID")),
        ("fia_query_tokens", Var("FIA_QUERY_TOKENS")),
        ("fia_physical_rows", Var("FIA_PHYSICAL_ROWS")),
        ("fia_valid_query_tokens", Var("FIA_VALID_QUERY_TOKENS")),
        ("fia_io_dtype_bytes", Var("FIA_IO_DTYPE_BYTES")),
        ("hbm_bytes", Var("HBM_BYTES")),
        ("launch0_hbm_bytes", Var("LAUNCH0_HBM_BYTES")),
        ("launch0_split_partial_bytes", Var("LAUNCH0_SPLIT_PARTIAL_BYTES")),
        ("launch0_output_bytes", Var("LAUNCH0_OUTPUT_BYTES")),
        ("launch0_owners", Var("LAUNCH0_OWNERS")),
        ("launch0_active_cores", Var("LAUNCH0_ACTIVE_CORES")),
        ("launch0_kv_splits", Var("LAUNCH0_KV_SPLITS")),
        ("launch0_block_bytes", Var("LAUNCH0_BLOCK_BYTES")),
        ("launch0_selected", Var("LAUNCH0_SELECTED")),
        ("launch0_pipeline_ms", Var("LAUNCH0_PIPELINE_MS")),
        ("launch0_exposed_ms", Var("LAUNCH0_EXPOSED_MS")),
        ("launch0_s2_tiles", Var("LAUNCH0_S2_TILES")),
        ("launch0_duration_ms", Var("LAUNCH0_DURATION_MS")),
        ("launch1_hbm_bytes", Var("LAUNCH1_HBM_BYTES")),
        ("launch1_split_partial_bytes", Var("LAUNCH1_SPLIT_PARTIAL_BYTES")),
        ("launch1_output_bytes", Var("LAUNCH1_OUTPUT_BYTES")),
        ("launch1_owners", Var("LAUNCH1_OWNERS")),
        ("launch1_active_cores", Var("LAUNCH1_ACTIVE_CORES")),
        ("launch1_kv_splits", Var("LAUNCH1_KV_SPLITS")),
        ("launch1_block_bytes", Var("LAUNCH1_BLOCK_BYTES")),
        ("launch1_selected", Var("LAUNCH1_SELECTED")),
        ("launch1_pipeline_ms", Var("LAUNCH1_PIPELINE_MS")),
        ("launch1_exposed_ms", Var("LAUNCH1_EXPOSED_MS")),
        ("launch1_s2_tiles", Var("LAUNCH1_S2_TILES")),
        ("launch1_duration_ms", Var("LAUNCH1_DURATION_MS")),
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
        return bind_fia_launches(ctx, operator, geometry, state, _attention_metrics)

    main, merge, derived, out = _split_expressions(ctx, static, {})
    return FamilyTemplate(
        axes=axes, stages=(main, merge),
        materialization=(("attention_output", out),),
        internal=(), residency=Const(0), derived=derived,
        regime_exprs=(), binder=binder,
        annotations=attention_notes(ctx, operator, static),
        program_duration=fia_program_duration(ctx),
    )


def _finite_double_buffer(stages, *, load_free=0, compute_free=0):
    """Two alternating L0 slots, each load then own MMA; next load overlaps MMA."""
    load_free = Fraction(load_free)
    compute_free = Fraction(compute_free)
    slot_free = [Fraction(0), Fraction(0)]
    load_busy = Fraction(0)
    compute_busy = Fraction(0)
    last = Fraction(0)
    for index, (load, compute) in enumerate(stages):
        slot = index % 2
        start = max(load_free, slot_free[slot])
        end = start + load
        cstart = max(end, compute_free, slot_free[slot])
        cend = cstart + compute
        load_free = end
        compute_free = cend
        slot_free[slot] = cend
        load_busy += load
        compute_busy += compute
        last = max(last, cend, end)
    return last, load_busy, compute_busy


def _sparse_mte1_rate(ctx, geometry, q_tokens, static) -> tuple[Fraction | None, dict]:
    """Resolve the SAT5 scoped sparse MTE1 service through the shared selector."""
    if (q_tokens < 1024 or static.get("heads") != 64
            or static.get("rkv") != 512 or static.get("dv") != 256
            or static.get("kv_dtype") not in {"fp8", "fp8_e4m3"}):
        return None, {"basis": "analytic_underfilled_or_wrong_geometry"}
    active = tuple(geometry.requests)
    if len(active) != 1:
        return None, {"basis": "analytic_multi_request_not_saturated_single"}
    req = active[0]
    if req.phase != "prefill" or int(req.context) != 0 or int(req.q_len) != q_tokens:
        return None, {"basis": "analytic_context_or_nonprefill"}
    if int(static.get("cores", 0)) != 32:
        return None, {"basis": "analytic_non32_owner"}
    meta = dict(getattr(ctx, "feature_metadata", ()) or ())
    supply = meta.get("profiler_measured_supply") or {}
    records = supply.get("resource_services") or ()
    if not isinstance(records, (list, tuple)) or not records:
        return None, {"basis": "analytic_no_resource_services"}
    hardware = str(meta.get("hardware_key") or meta.get("hardware_sku")
                   or meta.get("device") or "")
    # Query identity comes only from the deployed case feature record.  Never
    # borrow the producer's source_identity; that would make an R35 query match
    # an R64 record that happened to be attached to the same supply mapping.
    query_image = str(meta.get("image_id") or meta.get("oci_image_digest")
                      or meta.get("deployment_image_id") or "")
    query_scope = str(meta.get("mx_quant_source_revision")
                      or meta.get("source_scope") or "")
    carried = dict(meta.get("source_identity") or {})
    if carried.get("image_digest") and query_image and str(
            carried["image_digest"]).lower().split(":")[-1] != query_image.lower().split(":")[-1]:
        carried = {}
    if query_image and query_scope.lower() == "r64":
        source_identity = {"source_scope": "r64", "image_digest": query_image}
    elif carried and str(carried.get("source_scope") or "").lower() == "r64":
        source_identity = dict(carried)
    else:
        source_identity = {}
    if not hardware or not source_identity:
        return None, {"basis": "analytic_missing_query_source_identity"}
    from mindie_sim.observations.resource_services import select_resource_service
    selected = select_resource_service(
        records, hardware=hardware, source_identity=source_identity,
        dtype="bf16", q_len=int(req.q_len), context=int(req.context),
        kv_len=int(req.kv_len), causal=True, owners=32,
        phase="prefill", op_state="dynamic",
        mix_block_num=meta.get("mix_block_num"))
    if not selected:
        return None, {"basis": "analytic_no_matching_record"}
    rate = Fraction(str(selected["rate_bytes_per_s"])) / 1000
    return rate, {"basis": "measured_resource_service", "selected": selected}


def _compile_mla(ctx: CompiledOperator, operator) -> FamilyTemplate:
    static = _static_attention(ctx, operator, mla=True)
    axes = _attention_axes(static, ctx)

    def binder(geometry, state):
        return _attention_metrics(ctx, operator, geometry, mla=True)

    def sparse_schedule(geometry, state):
        if not static.get("packed_sparse"):
            return None
        q_tokens = sum(int(req.q_len) for req in geometry.requests)
        cores = max(1, int(static["cores"]))
        per_core = max(1, -(-q_tokens // cores))
        per_query = [selected_token_rows(
            int(req.context) + offset + 1, static["ratio"], static["topk"],
            static["kpool"], static["always_tail"])
            for req in geometry.requests for offset in range(int(req.q_len))]
        aligned = [((value + 15) // 16) * 16 for value in per_query]
        chunks = [sum(aligned[start:start + per_core])
                  for start in range(0, len(aligned), per_core)]
        critical = max(chunks, default=0)
        raw_critical = max((sum(per_query[start:start + per_core])
                             for start in range(0, len(per_query), per_core)),
                            default=0)
        owners = max(1, len(chunks))
        hbm = Fraction(str(ctx.supply.effective("hbm"))) / cores
        vec = Fraction(str(ctx.supply.effective("vector"))) / cores
        mte1 = Fraction(str(ctx.supply.effective("mte1"))) / cores
        measured_rate, measured_meta = _sparse_mte1_rate(
            ctx, geometry, sum(int(req.q_len) for req in geometry.requests), static)
        if measured_rate is not None:
            mte1 = measured_rate
        mte3 = Fraction(str(ctx.supply.effective("mte3"))) / cores
        mte2 = Fraction(str(ctx.supply.effective("mte2"))) / cores
        ub_mode = dict(getattr(ctx.supply, "service_modes", ()) or ())
        ub_spec = ub_mode.get("ub_copy")
        ub_rate = (Fraction(str(ub_spec[1])) / 1000 / cores
                   if ub_spec is not None else Fraction(0))
        peaks = dict(getattr(ctx.supply, "compute_peaks", ()) or ())
        bf16 = Fraction(str(peaks.get("bf16", 0.0) or 0.0)) / cores
        # Task iterator over the critical owner's real Q bounds.  Each task is
        # one <=128 sparse S2 tile; the last tile is 16-aligned exactly.  Q is
        # loaded once per query and output once per query on its last tile.
        owner_index = max(range(0, len(chunks)), key=lambda k: chunks[k]) if chunks else 0
        start_index = owner_index * per_core
        end_index = min(len(per_query), start_index + per_core)
        task_times = []
        for index in range(start_index, end_index):
            remaining = per_query[index]
            first = True
            while remaining > 0:
                rows = min(128, remaining)
                aligned = ((rows + 15) // 16) * 16
                qbytes = (static["heads"] * 576 * 2) if first else 0
                t_hbm = Fraction(rows * 656 + qbytes) / hbm
                t_mte2 = Fraction(rows * 656 + qbytes) / mte2
                v0_vec = Fraction(rows * 512 * static.get("dequant_ops", 4)) / vec
                t_ubv0 = (Fraction(rows * (512 + 64) * 2) / ub_rate
                          if ub_rate else Fraction(0))
                # Same-tile V0 readiness: GM->UB/dequant and UB->L1 producer
                # must be complete before this tile's QK. Streaming overlap is
                # max of concurrent services, not a global pipe sum.
                v0 = max(t_hbm, t_mte2, v0_vec, t_ubv0)
                qk_windows = ((128, 128), (128, 128), (128, 128),
                              (128, 128), (64, 64))
                qk_stages = tuple(
                    (Fraction(64 * kw * 2 + aligned * kw * 2) / mte1,
                     Fraction(aligned * 2 * static["heads"] * kw) / bf16)
                    for kw, _ in qk_windows)
                qk, qk_load_busy, qk_cube_busy = _finite_double_buffer(qk_stages)
                qk_mte1 = qk_load_busy
                qk_cube = qk_cube_busy
                softmax_ops = Fraction(aligned * static["heads"] * 8) / vec
                p_ub = (Fraction(static["heads"] * 128 * 2) / ub_rate
                        if ub_rate else Fraction(0))
                v1 = softmax_ops + p_ub
                p_load = Fraction(64 * aligned * 2) / mte1
                pv_stages = tuple(
                    (Fraction(128 * aligned * 2) / mte1,
                     Fraction(aligned * 2 * static["heads"] * 128) / bf16)
                    for _ in range(4))
                pv, pv_load_busy, pv_cube_busy = _finite_double_buffer(
                    pv_stages, load_free=p_load, compute_free=p_load)
                pv_mte1 = p_load + pv_load_busy
                pv_cube = pv_cube_busy
                last_tile = remaining <= 128
                first_tile = first
                ub_mode = dict(getattr(ctx.supply, "service_modes", ()) or ())
                ub_spec = ub_mode.get("ub_copy")
                if ub_spec is not None:
                    # service_modes retain the declared units/s convention;
                    # pipeline rates are units/ms.
                    ub_rate = Fraction(str(ub_spec[1])) / 1000 / cores
                    v2_first = (Fraction(static["heads"] * 512 * 4) / ub_rate
                                if first_tile else Fraction(0))
                else:
                    # UB-local copy is exposed, but no wall service is asserted.
                    v2_first = Fraction(0)
                v2_mid = (Fraction(2 * static["heads"] * 512) / vec
                          if not first_tile and not last_tile else Fraction(0))
                if last_tile and first_tile:
                    # single S2 tile: first FP32 copy then cast/store; no
                    # previous-partial normalization/accumulation.
                    v2_last = max(Fraction(static["heads"] * 512) / vec,
                                  Fraction(static["heads"] * 512 * 2) / mte3)
                elif last_tile:
                    v2_last = max(Fraction(3 * static["heads"] * 512) / vec,
                                  Fraction(static["heads"] * 512 * 2) / mte3)
                else:
                    v2_last = Fraction(0)
                fix_qk = (Fraction(4 * 64 * (((rows + 7) // 8) * 8))
                          / (Fraction(str(ctx.supply.effective("fixpipe"))) / cores))
                fix_pv = (Fraction(4 * 64 * 512)
                          / (Fraction(str(ctx.supply.effective("fixpipe"))) / cores))
                t_vec_v0 = v0_vec
                t_mte1 = qk_mte1 + pv_mte1
                t_vec_v1 = softmax_ops
                if last_tile and first_tile:
                    t_v2_vec = Fraction(static["heads"] * 512) / vec
                    t_mte3 = Fraction(static["heads"] * 512 * 2) / mte3
                elif last_tile:
                    t_v2_vec = Fraction(3 * static["heads"] * 512) / vec
                    t_mte3 = Fraction(static["heads"] * 512 * 2) / mte3
                else:
                    t_v2_vec = Fraction(0)
                    t_mte3 = Fraction(0)
                task_times.append((v0, qk, v1, pv, v2_first, v2_mid,
                                   v2_last, fix_qk, fix_pv, t_hbm, t_mte2,
                                   t_vec_v0, t_ubv0, t_mte1, t_vec_v1,
                                   t_v2_vec, t_mte3, qk_cube, pv_cube, p_ub))
                remaining -= rows
                first = False
        # Event loop with monotone cursors; O(number of tiles), no repeated
        # 0..n scans or per-stage arrays beyond the tile list.
        n = len(task_times)
        v0_end = [Fraction(0)] * n
        qk_end = [Fraction(0)] * n
        v1_end = [Fraction(0)] * n
        pv_end = [Fraction(0)] * n
        v2_end = [Fraction(0)] * n
        fixqk_end = [Fraction(0)] * n
        fixpv_end = [Fraction(0)] * n
        v0_done = [False] * n
        qk_done = [False] * n
        v1_done = [False] * n
        pv_done = [False] * n
        v2_done = [False] * n
        fixqk_done = [False] * n
        fixpv_done = [False] * n
        nv0 = nqk = nv1 = npv = nv2 = nfq = nfp = 0
        aic_free = aiv_free = fix_free = Fraction(0)
        qk_count = pv_count = v0_count = v1_count = v2_count = 0
        aic_total = aiv_total = fix_total = Fraction(0)
        v0_total = qk_total = v1_total = pv_total = v2_total = Fraction(0)
        v2_first_total = v2_mid_total = v2_last_total = Fraction(0)
        ub_total = Fraction(0)
        while v2_count < n:
            candidates = []
            if nqk < n and v0_done[nqk]:
                candidates.append(("qk", nqk, max(aic_free, v0_end[nqk])))
            if npv < n and v1_done[npv]:
                candidates.append(("pv", npv, max(aic_free, v1_end[npv])))
            if (nv0 < n and v0_count - v1_count < 2
                    and (nv0 < 3 or v2_done[nv0 - 3])):
                release = v2_end[nv0 - 3] if nv0 >= 3 else Fraction(0)
                candidates.append(("v0", nv0, max(aiv_free, release)))
            if nv1 < n and fixqk_done[nv1]:
                candidates.append(("v1", nv1, max(aiv_free, fixqk_end[nv1])))
            if nv2 < n and fixpv_done[nv2]:
                candidates.append(("v2", nv2, max(aiv_free, fixpv_end[nv2])))
            if nfq < n and qk_done[nfq]:
                candidates.append(("fq", nfq, max(fix_free, qk_end[nfq])))
            if nfp < n and pv_done[nfp]:
                candidates.append(("fp", nfp, max(fix_free, pv_end[nfp])))
            if not candidates:
                break
            if qk_count - pv_count >= 2:
                pv_ready = [c for c in candidates if c[0] == "pv"]
                if pv_ready:
                    candidates = pv_ready
            kind, index, start = min(candidates, key=lambda c: c[2])
            if kind == "qk":
                qk_end[index] = start + task_times[index][1]
                qk_done[index] = True
                nqk += 1
                qk_count += 1
                aic_free = qk_end[index]
                qk_total += task_times[index][17]
            elif kind == "pv":
                pv_end[index] = start + task_times[index][3]
                pv_done[index] = True
                npv += 1
                pv_count += 1
                aic_free = pv_end[index]
                pv_total += task_times[index][18]
            elif kind == "fq":
                fixqk_end[index] = start + task_times[index][7]
                fixqk_done[index] = True
                nfq += 1
                fix_free = fixqk_end[index]
                fix_total += task_times[index][7]
            elif kind == "fp":
                fixpv_end[index] = start + task_times[index][8]
                fixpv_done[index] = True
                nfp += 1
                fix_free = fixpv_end[index]
                fix_total += task_times[index][8]
            elif kind == "v0":
                v0_end[index] = start + task_times[index][0]
                v0_done[index] = True
                nv0 += 1
                v0_count += 1
                aiv_free = v0_end[index]
                v0_total += task_times[index][0]
            elif kind == "v1":
                v1_end[index] = start + task_times[index][2]
                v1_done[index] = True
                nv1 += 1
                v1_count += 1
                aiv_free = v1_end[index]
                v1_total += task_times[index][2]
            else:
                v2 = (task_times[index][4] + task_times[index][5]
                      + task_times[index][6])
                v2_end[index] = start + v2
                v2_done[index] = True
                nv2 += 1
                v2_count += 1
                aiv_free = v2_end[index]
                v2_total += v2
        aic_total = qk_total + pv_total
        aiv_total = v0_total + v1_total + v2_total
        hbm_total = v0_total
        mte1_total = v1_total
        mte3_total = v2_total
        duration = max(aic_free, aiv_free)
        hbm_busy = sum((task[9] for task in task_times), Fraction(0))
        mte2_busy = sum((task[10] for task in task_times), Fraction(0))
        vec_v0_busy = sum((task[11] for task in task_times), Fraction(0))
        ub_v0_busy = sum((task[12] for task in task_times), Fraction(0))
        mte1_busy = sum((task[13] for task in task_times), Fraction(0))
        vec_v1_busy = sum((task[14] for task in task_times), Fraction(0))
        vec_v2_busy = sum((task[15] for task in task_times), Fraction(0))
        mte3_busy = sum((task[16] for task in task_times), Fraction(0))
        vector_busy = vec_v0_busy + vec_v1_busy + vec_v2_busy
        ub_busy = (ub_v0_busy + sum((task[4] for task in task_times), Fraction(0))
                   + sum((task[19] for task in task_times), Fraction(0)))
        busy = {"cube": aic_total, "vector": vector_busy,
                "hbm": hbm_busy, "mte1": mte1_busy, "mte2": mte2_busy,
                "mte3": mte3_busy, "fixpipe": fix_total,
                "ub_copy": ub_busy}
        peak = max(busy.values())
        dominant = next(name for name, value in busy.items() if value == peak)
        tied = tuple(name for name, value in busy.items() if value == peak)
        return duration, dominant, tied, busy

    def sparse_stage_stats(geometry, state):
        if not static.get("packed_sparse"):
            return ()
        q_tokens = sum(int(req.q_len) for req in geometry.requests)
        cores = max(1, int(static["cores"]))
        per_core = max(1, -(-q_tokens // cores))
        per_query = [selected_token_rows(
            int(req.context) + offset + 1, static["ratio"], static["topk"],
            static["kpool"], static["always_tail"])
            for req in geometry.requests for offset in range(int(req.q_len))]
        aligned = [((value + 15) // 16) * 16 for value in per_query]
        chunks = [sum(aligned[start:start + per_core])
                  for start in range(0, len(aligned), per_core)]
        if not chunks:
            return ()
        owner_index = max(range(len(chunks)), key=lambda k: chunks[k])
        start_index = owner_index * per_core
        end_index = min(len(per_query), start_index + per_core)
        hbm = Fraction(str(ctx.supply.effective("hbm"))) / cores
        vec = Fraction(str(ctx.supply.effective("vector"))) / cores
        mte1 = Fraction(str(ctx.supply.effective("mte1"))) / cores
        measured_rate, measured_meta = _sparse_mte1_rate(
            ctx, geometry, sum(int(req.q_len) for req in geometry.requests), static)
        if measured_rate is not None:
            mte1 = measured_rate
        mte3 = Fraction(str(ctx.supply.effective("mte3"))) / cores
        mte2 = Fraction(str(ctx.supply.effective("mte2"))) / cores
        peaks = dict(getattr(ctx.supply, "compute_peaks", ()) or ())
        bf16 = Fraction(str(peaks.get("bf16", 0.0) or 0.0)) / cores
        ub_mode = dict(getattr(ctx.supply, "service_modes", ()) or ())
        ub_spec = ub_mode.get("ub_copy")
        ub_rate = (Fraction(str(ub_spec[1])) / 1000 / cores
                   if ub_spec is not None else Fraction(0))
        v0_total = qk_total = v1_total = pv_total = v2_total = Fraction(0)
        v2_first_total = v2_mid_total = v2_last_total = Fraction(0)
        ub_total = Fraction(0)
        for index in range(start_index, end_index):
            remaining = per_query[index]
            first = True
            while remaining > 0:
                rows = min(128, remaining)
                aligned = ((rows + 15) // 16) * 16
                qbytes = (static["heads"] * 576 * 2) if first else 0
                t_ub = (Fraction(rows * (512 + 64) * 2) / ub_rate
                        if ub_rate else Fraction(0))
                v0_total += max(Fraction(rows * 656 + qbytes) / hbm,
                                Fraction(rows * 656 + qbytes) / mte2,
                                Fraction(rows * 512 * static.get("dequant_ops", 4)) / vec,
                                t_ub)
                qk_stages = tuple(
                    (Fraction(64 * kw * 2 + aligned * kw * 2) / mte1,
                     Fraction(aligned * 2 * static["heads"] * kw) / bf16)
                    for kw in (128, 128, 128, 128, 64))
                qk_total += _finite_double_buffer(qk_stages)[0]
                p_ub = (Fraction(static["heads"] * 128 * 2) / ub_rate
                        if ub_rate else Fraction(0))
                v1_total += Fraction(aligned * static["heads"] * 8) / vec + p_ub
                p_load = Fraction(64 * aligned * 2) / mte1
                pv_stages = tuple(
                    (Fraction(128 * aligned * 2) / mte1,
                     Fraction(aligned * 2 * static["heads"] * 128) / bf16)
                    for _ in range(4))
                pv_total += _finite_double_buffer(
                    pv_stages, load_free=p_load, compute_free=p_load)[0]
                if first and ub_rate:
                    v2_first_total += Fraction(static["heads"] * 512 * 4) / ub_rate
                if not first and remaining > 128:
                    v2_mid_total += Fraction(2 * static["heads"] * 512) / vec
                if remaining <= 128:
                    last_ops = (Fraction(static["heads"] * 512) / vec
                                if first else Fraction(3 * static["heads"] * 512) / vec)
                    v2_last_total += max(last_ops,
                                         Fraction(static["heads"] * 512 * 2) / mte3)
                remaining -= rows
                first = False
        return (("v0_gather_dequant_l1", v0_total),
                ("qk_bf16", qk_total), ("v1_softmax", v1_total),
                ("pv_bf16", pv_total), ("v2_first_copy", v2_first_total),
                ("v2_mid_merge", v2_mid_total), ("v2_last_update", v2_last_total))

    def sparse_minimum(geometry, state):
        if not static.get("packed_sparse"):
            return Fraction(0)
        q_tokens = sum(int(req.q_len) for req in geometry.requests)
        cores = max(1, int(static["cores"]))
        per_core = max(1, -(-q_tokens // cores))
        per_query = [selected_token_rows(
            int(req.context) + offset + 1, static["ratio"], static["topk"],
            static["kpool"], static["always_tail"])
            for req in geometry.requests for offset in range(int(req.q_len))]
        aligned = [((value + 15) // 16) * 16 for value in per_query]
        critical_rows = max(
            (sum(aligned[start:start + per_core])
             for start in range(0, len(aligned), per_core)), default=0)
        critical_flops = 2 * critical_rows * static["heads"] * (576 + 512)
        peaks = dict(getattr(ctx.supply, "compute_peaks", ()) or ())
        bf16 = float(peaks.get("bf16", 0.0) or 0.0)
        if bf16 <= 0:
            return Fraction(0)
        return Fraction(critical_flops) / (Fraction(str(bf16)) / cores)

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
        program_duration=sparse_schedule,
        program_stages=sparse_stage_stats,
        minimum_duration=sparse_minimum,
        annotations=(("family", "mla"), ("rkv", str(static["rkv"])),
                     ("dn", str(static["dn"])), ("dr", str(static["dr"])),
                     ("dv", str(static["dv"])),
                     ("latent_width", str(static["rkv"] + static["dr"])),
                     ("absorb", "1"),
                     ("block_table_index", "absent"),
                     ("block_table_scalar_rate", "unavailable"),
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
