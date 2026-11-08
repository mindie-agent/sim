"""Recurrent, lightning-indexer and KPool piecewise family physics.

[Role]
  Model CausalConv1d, GDN/KDA chunked recurrence, quantized LightningIndexer
  selection and K1 compression/pooling as multi-stage resource demands.  The
  same helpers are imported by ``families_attention`` for the sparse compressed
  attention branch and the hidden-state recurrent operators.

[Invariants]
  * State traffic is the physical FP32 ``H_v * d_k * d_v * 4`` matrix read once
    per request and committed once per accepted slot; it is never a per-token
    weight lookup.
  * K1 row production is the telescoping difference
    ``floor(s_i / r) - floor((s_i - q_i) / r)``; independent per-chunk ceil
    counts double-count partial groups and are rejected.
  * Chunked prefill executes both the intra-chunk ``C x C`` matrices and the
    inter-chunk state products; a smaller chunk adds intra work without
    removing the state edge.
  * Decode indexer gathers charge a minimum MTE2 burst per strided row.  Prefill
    and draft-prefill (q_len > 1) load each query-tile wave's causal index prefix
    once from the contiguous or paged index cache.  The Cube -> Fixpipe -> Vector
    chain stays serial: scores are complete before the top-k stage starts.
  * All geometry comes from declared shapes, tiles and state.  No measured
    latency, vendor-name branch or fitted table is consulted.

[Timeouts & Locks]
  O(requests) integer arithmetic per evaluation; no I/O, waits, device work or locks.

[Side-Effect Policy]
  Returns immutable ``FamilyTemplate`` records only.  Importing this module
  performs no registration or global mutation.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

from mindie_sim.catalog.index_pool import completed_prefix, selected_tokens
from mindie_sim.schema.common import ModelingError
from .compiler import (
    CompiledOperator, FamilyTemplate, SegmentAxis, _demand, _frac, _out_name,
)
from .expressions import Add, Const, Div, Max, Mul, Sub, Var, as_fraction
from .guards import StepGuard


# Ascend MTE2 moves data in finite bursts; a paged or gathered access below
# this granularity still consumes one transaction.  These are structural
# transfer constants, not measured latencies.
BURST_BYTES = 512
DESCRIPTOR_BYTES = 8
PAGE_ROWS = 128
HEAD_REDUCE_M = 16
DEFAULT_ELEMENT_BYTES = Fraction(2)


def align(value: int, multiple: int) -> int:
    """Round a logical extent up to one Cube/UB fractal or byte block."""
    value, multiple = int(value), int(multiple)
    if value <= 0:
        return 0
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _ceil_div(value: int, divisor: int) -> int:
    return -(-int(value) // int(divisor))


def gather_bytes(rows: int, row_bytes: Fraction, *,
                 burst: int = BURST_BYTES,
                 descriptor: int = DESCRIPTOR_BYTES) -> Fraction:
    """MTE2 bytes for ``rows`` independent strided gather transactions."""
    if rows <= 0:
        return Fraction(0)
    return Fraction(rows) * (max(Fraction(row_bytes), Fraction(burst))
                             + Fraction(descriptor))


def range_burst_bytes(lo: int, hi: int, row_bytes: Fraction, *,
                      page: int = PAGE_ROWS, burst: int = BURST_BYTES,
                      descriptor: int = DESCRIPTOR_BYTES) -> tuple[Fraction, int]:
    """Page-aligned MTE2 bytes and page-table entries for a contiguous range.

    Each touched page is one transaction; a partially filled page still moves
    at least one burst.  The returned byte count includes descriptors, so HBM
    and MTE2 stay byte-conservative.
    """
    lo, hi, page = int(lo), int(hi), max(1, int(page))
    if hi <= lo:
        return Fraction(0), 0
    r = row_bytes if isinstance(row_bytes, Fraction) else Fraction(row_bytes)
    num, den = r.numerator, r.denominator
    b_num, d_num = burst * den, descriptor * den
    first, last = lo // page, (hi + page - 1) // page
    touched = last - first
    if touched == 1:
        tot = max((hi - lo) * num, b_num) + d_num
        return (Fraction(tot) if den == 1 else Fraction(tot, den)), 1
    tot = max(((first + 1) * page - lo) * num, b_num) + max((hi - (last - 1) * page) * num, b_num)
    if touched > 2:
        tot += (touched - 2) * max(page * num, b_num)
    tot += touched * d_num
    return (Fraction(tot) if den == 1 else Fraction(tot, den)), touched


def k1_new_rows(total: int, query: int, ratio: int) -> int:
    """Exact cross-block compression rows produced by one request step.

    ``total = context + query`` is the post-step length.  Summing this over
    steps telescopes to ``floor(total_final / ratio)``; summing independent
    ``ceil(query / ratio)`` values does not.
    """
    total, query, ratio = int(total), int(query), int(ratio)
    if ratio < 1:
        raise ModelingError(f"compression_ratio_invalid:{ratio}")
    if query < 0 or total < query:
        raise ModelingError(f"compression_geometry_invalid:{query}:{total}")
    return total // ratio - (total - query) // ratio


def selected_token_rows(position: int, ratio: int, topk: int, kpool: int = 1,
                        always_select_tail: bool = False) -> int:
    """Index rows gathered for one causal query at ``position`` tokens."""
    position, ratio = int(position), max(1, int(ratio))
    kpool, topk = max(1, int(kpool)), max(0, int(topk))
    if kpool > 1 and topk and topk % kpool == 0:
        return selected_tokens(position, topk, kpool, always_select_tail)
    available = position // ratio if ratio > 1 else position
    return min(available, topk) if topk else available


def index_scanned_rows(query: int, context: int, kpool: int = 1) -> int:
    """Causal index-key count summed over a contiguous query block."""
    query, context, kpool = int(query), int(context), max(1, int(kpool))
    return (completed_prefix(context + query, kpool)
            - completed_prefix(context, kpool))


def state_matrix_bytes(value_heads: int, dk: int, dv: int) -> int:
    """One FP32 recurrent state matrix for one request."""
    return max(0, int(value_heads) * int(dk) * int(dv) * 4)


def _element_count(row_bytes: Fraction, element_bytes: Fraction) -> int:
    if row_bytes <= 0 or element_bytes <= 0:
        return 1
    return max(1, int(math.ceil(float(row_bytes / element_bytes))))


def _conv_static(theta) -> dict[str, Any]:
    row_bytes = _frac(theta.find("row_bytes", 2) or 2)
    declared = theta.find("act_bytes", None)
    qkv_width = int(theta.find("qkv_width", 0) or 0)
    if declared is not None:
        act_bytes = as_fraction(declared)
    elif qkv_width > 0 and row_bytes > 0:
        act_bytes = row_bytes / qkv_width
    else:
        act_bytes = DEFAULT_ELEMENT_BYTES
    if act_bytes <= 0:
        act_bytes = DEFAULT_ELEMENT_BYTES
    elements = _element_count(row_bytes, act_bytes)
    state_rows = max(1, int(theta.find("rows", 1) or 1))
    kernel = max(1, int(theta.find("kernel", 4) or 4))
    history = max(0, int(theta.find("history_rows", kernel - 1) or 0))
    return {
        "row_bytes": row_bytes, "elements": elements,
        "act_bytes": act_bytes,
        "state_rows": state_rows, "kernel": kernel, "history_rows": history,
        "state_bytes": Fraction(state_rows) * row_bytes,
    }


def _conv_metrics(theta, geometry) -> dict[str, Any]:
    static = _conv_static(theta)
    tokens = max(0, geometry.tokens)
    batch = geometry.batch
    state = static["state_bytes"]
    moved = Fraction(tokens) * static["row_bytes"]
    vector = (Fraction(tokens) * static["elements"] * (static["kernel"] + 2)
              + Fraction(batch) * static["state_rows"] * static["elements"] * 2)
    return {
        "T": max(1, tokens), "B": batch,
        "tokens": max(1, tokens), "batch": batch,
        "conv_kernel": static["kernel"],
        "conv_history_rows": static["history_rows"],
        "conv_state_rows": static["state_rows"],
        "conv_state_bytes": state,
        "conv_state_read": state * batch, "conv_state_write": state * batch,
        "conv_input_bytes": moved, "conv_output_bytes": moved,
        "conv_elements": static["elements"],
        "conv_vector_ops": vector, "CONV_VECTOR": vector,
    }


def _delta_static(theta) -> dict[str, Any]:
    key_heads = max(1, int(theta.find("key_heads", 1) or 1))
    value_heads = max(1, int(theta.find("value_heads", key_heads) or key_heads))
    dk = max(1, int(theta.find("dk", 1) or 1))
    dv = max(1, int(theta.find("dv", 1) or 1))
    chunk = max(1, int(theta.find("chunk", 64) or 64))
    declared = theta.find("act_bytes", None)
    state_row = Fraction(theta.find("state_row_bytes", 0) or 0)
    if declared is not None:
        act_bytes = as_fraction(declared)
    elif state_row > 0:
        recurrent = Fraction(value_heads * dk * dv * 4)
        history_width = 2 * key_heads * dk + value_heads * dv
        residual = state_row - recurrent
        act_bytes = (residual / history_width
                     if residual > 0 and history_width > 0
                     else DEFAULT_ELEMENT_BYTES)
    else:
        act_bytes = DEFAULT_ELEMENT_BYTES
    if act_bytes <= 0:
        act_bytes = DEFAULT_ELEMENT_BYTES
    gate_rank = max(0, int(theta.find("gate_rank", 0) or 0))
    kda = bool(int(theta.find("accepted_state", 0) or 0)) or gate_rank > 0
    gate_elements = value_heads * (gate_rank if kda and gate_rank else dv)
    input_elements = 2 * key_heads * dk + value_heads * dv + gate_elements
    return {
        "key_heads": key_heads, "value_heads": value_heads,
        "dk": dk, "dv": dv, "chunk": chunk, "kda": kda,
        "act_bytes": act_bytes, "gate_elements": gate_elements,
        "input_elements": input_elements,
        "state_matrix_bytes": state_matrix_bytes(value_heads, dk, dv),
    }


def _delta_metrics(theta, geometry, state) -> dict[str, Any]:
    static = _delta_static(theta)
    chunk = static["chunk"]
    accepted = int(state.acceptance.accepted)
    batch = geometry.batch
    commit = max(accepted, batch) if static["kda"] else batch
    chunks = intra = state_ops = vector_state = 0
    intra_unit = (2 * static["dk"] * static["key_heads"]
                  + 2 * static["dv"] * static["value_heads"])
    state_unit = 4 * static["dk"] * static["dv"] * static["value_heads"]
    short_unit = 7 * static["dk"] * static["dv"] * static["value_heads"]
    for request in geometry.requests:
        length = int(request.q_len)
        if length <= 0:
            continue
        full, rem = divmod(length, chunk)
        chunks += full + (1 if rem else 0)
        intra += intra_unit * (full * chunk * chunk + rem * rem)
        phase = str(getattr(request, "phase", "") or "")
        max_q = length
        if max_q <= 16 or "decode" in phase:
            vector_state += short_unit * length
        else:
            state_ops += state_unit * length
    matrix = static["state_matrix_bytes"]
    read, write = batch * matrix, commit * matrix
    input_bytes = (Fraction(geometry.tokens) * static["input_elements"]
                   * static["act_bytes"])
    output_bytes = (Fraction(geometry.tokens) * static["value_heads"]
                    * static["dv"] * static["act_bytes"])
    vector = (Fraction(geometry.tokens) * static["value_heads"] * (
        static["dk"] + static["dv"]) * 3
        + chunks * static["value_heads"] * static["dv"] * 2 + vector_state)
    logical_flops = intra + state_unit * geometry.tokens
    cube_flops = intra + state_ops
    return {
        "T": max(1, geometry.tokens), "B": batch,
        "tokens": max(1, geometry.tokens), "batch": batch,
        "key_heads": static["key_heads"], "value_heads": static["value_heads"],
        "dk": static["dk"], "dv": static["dv"], "chunk": chunk, "chunks": max(1, chunks),
        "CHUNKS": max(1, chunks),
        "state_matrix_bytes": matrix,
        "delta_state_read": read, "delta_state_write": write,
        "delta_input_bytes": input_bytes, "DELTA_INPUT_BYTES": input_bytes,
        "delta_output_bytes": output_bytes, "DELTA_OUTPUT_BYTES": output_bytes,
        "act_bytes": static["act_bytes"], "gate_elements": static["gate_elements"],
        "delta_flops": logical_flops, "DELTA_FLOPS": logical_flops, "DELTA_CUBE_FLOPS": cube_flops,
        "delta_vector_ops": vector, "DELTA_VECTOR": vector,
        "ACCEPT": accepted, "accepted": accepted, "COMMIT": commit, "commit_rows": commit,
    }


def _compile_sinkhorn(ctx: CompiledOperator, operator) -> FamilyTemplate:
    """R0 Sinkhorn and HC mix. Physical vector work, never supply rates."""
    spec = operator.theta
    s = spec if isinstance(spec, dict) else type("_S", (), {"get": lambda self, key, default=None: (
        default if spec.find(key, None) is None else spec.find(key, None))})()
    iters = max(1, int(s.get("iters", 20)))
    mult = max(1, int(s.get("mult", 4)))
    hidden = 1
    for key in ("H", "hidden", "hidden_size", "width"):
        raw = s.get(key, None)
        if raw:
            hidden = max(1, int(raw))
            break
    norm = mult * (mult - 1) + mult + mult * mult
    sink_per = 5 * mult * mult + (2 * iters - 1) * norm
    pre_per, post_per = hidden * (2 * mult - 1), mult * hidden * (2 * mult + 1)
    per = {"hc_pre_mix": pre_per, "hc_post_mix": post_per}.get(operator.name, sink_per)

    def binder(geometry, state):
        tokens = max(0, int(geometry.tokens))
        ops = Fraction(tokens * per)
        return {"T": max(1, tokens), "tokens": max(1, tokens), "SINK_OPS": ops,
                "sinkhorn_ops": Fraction(tokens * sink_per),
                "hc_pre_mix": Fraction(tokens * pre_per),
                "hc_post_mix": Fraction(tokens * post_per)}

    ops = Var("SINK_OPS")
    return FamilyTemplate(
        axes=(SegmentAxis("TOKEN_TILES", "tokens",
                          StepGuard("sinkhorn_tiles", "tokens", 1), "max_tokens"),),
        stages=((_demand(ctx, "vector", ops),),),
        materialization=((_out_name(ctx), ops),), internal=(), residency=Const(0),
        derived=(("sinkhorn_ops", ops),), regime_exprs=(), binder=binder,
        annotations=(("family", operator.name), ("iters", str(iters)),
                     ("mult", str(mult))),
    )


def compile_recurrent(ctx: CompiledOperator, operator) -> FamilyTemplate:
    if operator.name in {"sinkhorn", "hc_pre_mix", "hc_post_mix"}:
        return _compile_sinkhorn(ctx, operator)
    theta = operator.theta
    if operator.name == "conv":
        static = _conv_static(theta)
        axes = (SegmentAxis("TOKEN_TILES", "tokens",
                            StepGuard("conv_tiles", "tokens", static["kernel"]),
                            "max_tokens"),)

        def binder(geometry, state):
            return _conv_metrics(theta, geometry)

        state = Const(static["state_bytes"])
        moved = Mul((Var("T"), Const(static["row_bytes"])))
        vector = Var("CONV_VECTOR")
        main = (
            _demand(ctx, "vector", vector),
            _demand(ctx, "mte2", Add((moved, Mul((Var("B"), state))))),
            _demand(ctx, "mte3", Add((moved, Mul((Var("B"), state))))),
            _demand(ctx, "hbm", Add((Mul((Const(2), moved)),
                                     Mul((Const(2), Var("B"), state))))),
        )
        return FamilyTemplate(
            axes=axes, stages=(main,), materialization=((_out_name(ctx), moved),),
            internal=(), residency=Const(0),
            derived=(("conv_kernel", Const(static["kernel"])),
                     ("conv_history_rows", Const(static["history_rows"])),
                     ("conv_state_bytes", state), ("conv_ops", vector)),
            regime_exprs=(), binder=binder,
            annotations=(("family", "recurrent_conv"), ("kernel", str(static["kernel"]))),
        )

    static = _delta_static(theta)
    axes = (SegmentAxis("CHUNK_TILES", "tokens",
                        StepGuard("recurrent_tiles", "tokens", static["chunk"]),
                        "max_tokens"),)

    def binder(geometry, state):
        return _delta_metrics(theta, geometry, state)

    matrix = Const(static["state_matrix_bytes"])
    read, write = Mul((Var("B"), matrix)), Mul((Var("COMMIT"), matrix))
    inputs, outputs = Var("DELTA_INPUT_BYTES"), Var("DELTA_OUTPUT_BYTES")
    main = (
        _demand(ctx, "cube", Var("DELTA_CUBE_FLOPS")),
        _demand(ctx, "vector", Var("DELTA_VECTOR")),
        _demand(ctx, "mte2", Add((read, inputs))),
        _demand(ctx, "mte1", read),
        _demand(ctx, "mte3", Add((write, outputs))),
        _demand(ctx, "hbm", Add((read, write, inputs, outputs))),
    )
    return FamilyTemplate(
        axes=axes, stages=(main,), materialization=((_out_name(ctx), outputs),),
        internal=(), residency=Const(0),
        derived=(("flops", Var("DELTA_FLOPS")), ("state_bytes", read),
                 ("state_matrix_bytes", matrix), ("input_bytes", inputs),
                 ("output_bytes", outputs), ("delta_input_bytes", inputs),
                 ("delta_output_bytes", outputs),
                 ("act_bytes", Const(static["act_bytes"])),
                 ("gate_elements", Const(static["gate_elements"])),
                 ("accepted_slots", Var("ACCEPT")), ("commit_rows", Var("COMMIT")),
                 ("chunk_size", Const(static["chunk"])), ("chunks", Var("CHUNKS")),
                 ("key_heads", Const(static["key_heads"])),
                 ("value_heads", Const(static["value_heads"])),
                 ("dk", Const(static["dk"])), ("dv", Const(static["dv"]))),
        regime_exprs=(("ACCEPT", Var("ACCEPT")), ("COMMIT", Var("COMMIT")),
                      ("CHUNKS", Var("CHUNKS"))),
        binder=binder,
        annotations=(("family", "recurrent_delta"), ("chunk", str(static["chunk"])),
                     ("accepted_state", str(int(static["kda"])))),
    )


def _floor_sum(count, modulus, step, base):
    """sum_{i=0..count-1} floor((step*i + base) / modulus)."""
    if count <= 0:
        return 0
    if base < 0:
        shift = (-base + modulus - 1) // modulus
        return _floor_sum(count, modulus, step, base + shift * modulus) - shift * count
    total = 0
    if step >= modulus:
        total += (count - 1) * count * (step // modulus) // 2
        step %= modulus
    if base >= modulus:
        total += count * (base // modulus)
        base %= modulus
    y_max = (step * count + base) // modulus
    if y_max == 0:
        return total
    x_max = y_max * modulus - base
    total += (count - (x_max + step - 1) // step) * y_max
    return total + _floor_sum(y_max, step, modulus, (step - x_max % step) % step)


def _ap_floor(count, step, base, modulus):
    """sum_{i=1..count} floor((base + i*step) / modulus)."""
    return 0 if count <= 0 else _floor_sum(count, modulus, step, step + base)


def _mod_below(count, step, base, modulus, limit):
    """Count i=1..count with (base + i*step) mod modulus < limit."""
    if count <= 0 or limit <= 0:
        return 0
    if limit >= modulus:
        return count
    return (_ap_floor(count, step, base, modulus)
            - _ap_floor(count, step, base - limit, modulus))


def _capped_floor_sum(limit, kpool, topk):
    """sum_{position=1..limit} min(position // kpool, topk); a 0 floor adds 0."""
    if limit <= 0 or topk <= 0:
        return 0
    groups, tail = divmod(limit, kpool)
    if groups <= 0:
        return 0
    if groups <= topk:
        return kpool * (groups - 1) * groups // 2 + (tail + 1) * groups
    return kpool * (topk - 1) * topk // 2 + topk * (limit - topk * kpool + 1)


def _wave_prefix_traffic(base, step, count, kpool, row_bytes):
    """Bytes and pages for wave ends i=1..count, height (base+i*step)//kpool.

    Matches ``range_burst_bytes(0, height)``: a short page pays one burst, and
    a short tail inside a larger row pays ``burst - tail*row``.
    """
    if count <= 0:
        return Fraction(0), 0
    page, burst, desc = PAGE_ROWS, BURST_BYTES, DESCRIPTOR_BYTES
    modulus = kpool * page
    pages = (_ap_floor(count, step, base, modulus) + count
             - _mod_below(count, step, base, modulus, kpool))
    if row_bytes <= 0 or Fraction(page) * row_bytes < burst:
        return Fraction(pages) * (burst + desc), pages
    heights = _ap_floor(count, step, base, kpool)
    partial = _ceil_div(burst * row_bytes.denominator, row_bytes.numerator)
    penalty = Fraction(0)
    if partial > 1:
        hits = remainder_sum = 0
        for residue in range(1, partial):
            higher = _mod_below(count, step, base, modulus, (residue + 1) * kpool)
            lower = _mod_below(count, step, base, modulus, residue * kpool)
            taken = higher - lower
            hits += taken
            remainder_sum += taken * residue
        penalty = Fraction(hits) * burst - Fraction(remainder_sum) * row_bytes
    return Fraction(heights) * row_bytes + Fraction(pages) * desc + penalty, pages


def _paged_index_traffic(context, query, kpool, wave, row_bytes):
    """Prefill index-key traffic: one paged prefix load per query tile."""
    full, remainder = divmod(int(query), max(1, int(wave)))
    total, pages = _wave_prefix_traffic(context, max(1, int(wave)), full, kpool, row_bytes)
    if remainder:
        extra, extra_pages = range_burst_bytes(0, (context + query) // kpool, row_bytes)
        total += extra
        pages += extra_pages
    return total, pages


def _index_kpool(ctx: CompiledOperator, theta) -> int:
    """Selector scores the producer's pooled row domain, never a default 1."""
    raw = theta.find("kpool", None)
    if raw is None:
        raw = theta.find("compress_ratio", None)
    if raw is None:
        for sibling in getattr(ctx, "siblings", ()):
            if sibling.name in {"index_score", "index_filter"}:
                raw = sibling.theta.find("kpool", None)
                if raw is not None:
                    break
    return max(1, int(raw if raw is not None else 1))


def _index_metrics(theta, geometry=None, *, kpool: int | None = None,
                   wave: int = 1) -> dict[str, Any]:
    heads = max(1, int(theta.find("heads", 1) or 1))
    dim = max(1, int(theta.find("dim", 1) or 1))
    topk = max(1, int(theta.find("topk", 1) or 1))
    if kpool is None:
        raw = theta.find("kpool", None)
        if raw is None:
            raw = theta.find("compress_ratio", None)
        kpool = int(raw) if raw is not None else 1
    kpool = max(1, kpool)
    wave = max(1, int(wave or 1))
    index_bytes = _frac(theta.find("index_bytes", 1) or 1)
    scale_bytes = _frac(theta.find("scale_bytes", 2) or 2)
    row_bytes = Fraction(dim) * index_bytes
    queries = scanned = selected = 0
    gather = Fraction(0)
    transactions = 0
    paged_queries = burst_requests = 0
    for request in (geometry.requests if geometry is not None else ()):
        if not request.q_len:
            continue
        queries += request.q_len
        rows = index_scanned_rows(request.q_len, request.context, kpool)
        scanned += rows
        selected += (_capped_floor_sum(request.context + request.q_len, kpool, topk)
                     - _capped_floor_sum(request.context, kpool, topk))
        # Prefill q_len>1 reloads each query tile's prefix from paged cache.
        # Decode-phase steps, including q_len>1, keep one burst per scanned row.
        paged = (request.phase in {"prefill", "draft_prefill"}
                 and request.q_len > 1)
        if paged:
            key_bytes, pages = _paged_index_traffic(
                request.context, request.q_len, kpool, wave, row_bytes)
            gather += key_bytes
            transactions += pages
            paged_queries += request.q_len
        else:
            gather += gather_bytes(rows, row_bytes)
            transactions += rows
            burst_requests += 1
    if paged_queries and burst_requests:
        query_row = Fraction((paged_queries + 1) * heads) * row_bytes
    elif paged_queries:
        query_row = Fraction(paged_queries * heads) * row_bytes
    else:
        query_row = Fraction(heads) * row_bytes
    qk = 2 * scanned * heads * dim
    reduce = 2 * scanned * HEAD_REDUCE_M
    fixpipe = 2 * scanned * heads + 4 * scanned
    comparisons = scanned * math.log2(max(2, topk))
    selection = queries * topk * 4 if scanned else 0
    vector = Fraction(scanned * heads * 3 + scanned * 2) + comparisons
    query_bytes = (Fraction(queries * heads * dim) * index_bytes
                   if scanned else Fraction(0))
    return {
        "QUERIES": max(1, queries), "queries": max(1, queries),
        "SCANNED": scanned, "index_rows": scanned,
        "SELECTED": selected, "selected_tokens": selected,
        "index_heads": heads, "index_dim": dim,
        "index_topk": topk, "index_kpool": kpool,
        "index_row_bytes": row_bytes, "scale_bytes": scale_bytes,
        "GATHER": gather, "gather_bytes": gather,
        "gather_transactions": transactions,
        "QUERY_ROW": query_row,
        "QK_INT8": Fraction(qk), "HEAD_REDUCE": Fraction(reduce),
        "qk_int8_flops": qk, "head_reduce_flops": reduce,
        "FIXPIPE": Fraction(fixpipe), "fixpipe_bytes": Fraction(fixpipe),
        "QUERY_BYTES": query_bytes,
        "INDEX_VECTOR": vector, "index_vector_ops": vector,
        "TOPK_OPS": Fraction(comparisons), "topk_comparisons": Fraction(comparisons),
        "SELECTION_BYTES": Fraction(selection),
        "selection_output_bytes": Fraction(selection),
        "index_work": scanned * heads * dim,
    }


def compile_indexer(ctx: CompiledOperator, operator) -> FamilyTemplate:
    theta = operator.theta
    kpool = _index_kpool(ctx, theta)
    metrics = _index_metrics(theta, kpool=kpool)
    nt = max(1, int(theta.find("nt", ctx.policy.nt) or ctx.policy.nt))
    axes = (SegmentAxis("INDEX_TILES", "index_rows",
                        StepGuard("index_tiles", "index_rows", nt, lower_bound=0),
                        "max_index_rows"),)
    if operator.name in {"index_score", "index_filter"}:
        def binder(geometry, state):
            observed = _index_metrics(theta, geometry, kpool=kpool, wave=nt)
            rows = int(observed["SCANNED"])
            tiles = _ceil_div(rows, nt) if rows else 0
            fill, steady = (1, rows - 1) if tiles > 1 else (rows, 0)
            heads, dim = observed["index_heads"], observed["index_dim"]
            # Decode: gather_bytes(1, row).  Prefill: paged-wave bytes / pairs.
            gather_row = (observed["GATHER"] / rows) if rows else 0
            return dict(observed,
                        FILL_ROWS=fill, STEADY=steady,
                        GATHER_ROW=gather_row,
                        GATHER_TX=observed["gather_transactions"],
                        QUERY_ROW=observed["QUERY_ROW"],
                        QK_ROW=Fraction(2 * heads * dim),
                        FIX_ROW=Fraction(2 * heads + 4),
                        REDUCE_ROW=Fraction(2 * HEAD_REDUCE_M),
                        SCORE_ROW=Fraction(4))

        fill, steady = Var("FILL_ROWS"), Var("STEADY")
        gather_row, query_row = Var("GATHER_ROW"), Var("QUERY_ROW")
        qk_row = Var("QK_ROW")
        reduce_row, score_row = Var("REDUCE_ROW"), Var("SCORE_ROW")
        fix_row = Var("FIX_ROW")
        load_row = Add((Mul((fill, gather_row)), query_row))
        score_load = (_demand(ctx, "mte2", load_row), _demand(ctx, "mte1", load_row), _demand(ctx, "hbm", load_row))
        score_cube = (_demand(ctx, "cube", Mul((fill, qk_row))),)
        score_fix = (_demand(ctx, "fixpipe", Mul((fill, fix_row))),)
        score_reduce = (_demand(ctx, "vector", Mul((fill, reduce_row))),)
        score_store = (_demand(ctx, "mte3", Mul((fill, score_row))), _demand(ctx, "hbm", Mul((fill, score_row))))
        std_store = Mul((steady, score_row))
        std_hbm = Add((Mul((steady, gather_row)), std_store))
        std_vec = Mul((steady, reduce_row))
        steady_stage = (
            _demand(ctx, "mte2", Mul((steady, gather_row))),
            _demand(ctx, "mte1", Mul((steady, gather_row))),
            _demand(ctx, "hbm", std_hbm),
            _demand(ctx, "cube", Mul((steady, qk_row))),
            _demand(ctx, "fixpipe", Mul((steady, fix_row))),
            _demand(ctx, "vector", std_vec),
            _demand(ctx, "mte3", std_store),
        )
        scores = Mul((Var("SCANNED"), Const(4)))
        gather, fixpipe = Var("GATHER"), Var("FIXPIPE")
        logical = Mul((Var("SCANNED"), Const(metrics["index_row_bytes"])))
        amplification = Div(gather, Max((logical, Const(1))))
        return FamilyTemplate(
            axes=axes,
            stages=(score_load, score_cube, score_fix, score_reduce,
                    score_store, steady_stage),
            materialization=(("index_scores", scores),),
            internal=(), residency=Const(0),
            derived=(("index_rows", Var("SCANNED")),
                     ("index_flops", Add((Var("QK_INT8"), Var("HEAD_REDUCE")))),
                     ("logical_flops", Var("QK_INT8")), ("qk_int8_flops", Var("QK_INT8")),
                     ("head_reduce_flops", Var("HEAD_REDUCE")), ("fixpipe_bytes", fixpipe),
                     ("score_output_bytes", scores), ("gather_transactions", Var("GATHER_TX")),
                     ("gather_amplification", amplification)),
            regime_exprs=(), binder=binder,
            annotations=(("family", "indexer_score"), ("heads", str(metrics["index_heads"])),
                         ("dim", str(metrics["index_dim"])), ("topk", str(metrics["index_topk"])),
                         ("kpool", str(metrics["index_kpool"]))),
        )

    select_ops = max(1, int(theta.find("select_ops_per_row", 4) or 4))

    def binder_select(geometry, state):
        observed = _index_metrics(theta, geometry, kpool=kpool, wave=nt)
        rows = int(observed["SCANNED"])
        queries = max(1, int(observed["QUERIES"]))
        selected = int(observed["SELECTED"])
        return {"INDEX_ROWS": rows, "QUERIES": queries,
                "SEL_OPS": Fraction(rows * select_ops + selected),
                "SEL_READ": Fraction(rows * 4),
                "SEL_WRITE": Fraction(selected * 4),
                "SELECTED": selected,
                "index_rows": rows,
                "index_work": rows * select_ops}

    read, write = Var("SEL_READ"), Var("SEL_WRITE")
    main = (
        _demand(ctx, "vector", Var("SEL_OPS")),
        _demand(ctx, "mte2", read),
        _demand(ctx, "mte3", write),
        _demand(ctx, "hbm", Add((read, write))),
    )
    return FamilyTemplate(
        axes=axes, stages=(main,),
        materialization=(("index_selection", write),),
        internal=(), residency=Const(0),
        derived=(("index_rows", Var("INDEX_ROWS")),
                 ("selection_ops", Var("SEL_OPS")),
                 ("selected_tokens", Var("SELECTED"))),
        regime_exprs=(), binder=binder_select,
        annotations=(("family", "indexer_select"),
                     ("topk", str(metrics["index_topk"]))),
    )


# K1 compression, pooling and tail retention


def _compression_metrics(theta, geometry=None) -> dict[str, Any]:
    ratio = max(1, int(theta.find("ratio", 1) or 1))
    width = max(1, int(theta.find("width", 1) or 1))
    row_bytes = _frac(theta.find("row_bytes", 2) or 2)
    state_rows = max(0, int(theta.find("state_rows", 0) or 0))
    always_tail = int(theta.find("always_tail", 0) or 0)
    new_rows = source_rows = tail_rows = remainder = 0
    for request in (geometry.requests if geometry is not None else ()):
        produced = k1_new_rows(request.kv_len, request.q_len, ratio)
        new_rows += produced
        source_rows += produced * ratio
        remainder = max(remainder, request.kv_len % ratio)
        if request.q_len and request.kv_len % ratio:
            tail_rows += 1
    return {
        "ROWS": new_rows, "new_rows": new_rows,
        "pool_rows": new_rows, "source_rows": source_rows,
        "tail_rows": tail_rows, "TAIL_ROWS": tail_rows,
        "remainder": remainder, "REMAINDER": remainder,
        "ratio": ratio, "width": width, "row_bytes": row_bytes,
        "state_rows": state_rows,
        "POOL_OUT": new_rows + always_tail,
        "pool_out_rows": new_rows + always_tail,
        "compression_state_bytes": Fraction(state_rows) * row_bytes,
    }


def compile_compression(ctx: CompiledOperator, operator) -> FamilyTemplate:
    theta = operator.theta
    metrics = _compression_metrics(theta)
    ratio, row_bytes = metrics["ratio"], metrics["row_bytes"]
    axes = (SegmentAxis("ROWS", "new_rows",
                        StepGuard("compress_rows", "new_rows", 1, lower_bound=0),
                        "max_tokens"),)

    def binder(geometry, state):
        return _compression_metrics(theta, geometry)

    if operator.name == "compress":
        new_rows = Var("ROWS")
        source_bytes = Mul((new_rows, Const(ratio * row_bytes)))
        state_read = Const(metrics["compression_state_bytes"])
        writes = Mul((Var("POOL_OUT"), Const(row_bytes)))
        vector = Mul((new_rows, Const(metrics["width"] * (4 * ratio + 4))))
        main = (
            _demand(ctx, "vector", vector),
            _demand(ctx, "mte2", Add((source_bytes, state_read))),
            _demand(ctx, "mte3", writes),
            _demand(ctx, "hbm", Add((source_bytes, state_read, writes))),
        )
        return FamilyTemplate(
            axes=axes, stages=(main,),
            materialization=(("compressed_kv", writes),),
            internal=(), residency=Const(0),
            derived=(("new_rows", new_rows), ("remainder", Var("REMAINDER")),
                     ("compressed_rows", new_rows),
                     ("source_rows", Mul((new_rows, Const(ratio)))),
                     ("compression_state_bytes", state_read)),
            regime_exprs=(("REMAINDER", Var("REMAINDER")),),
            binder=binder,
            state_effects=(("compress", f"ratio={ratio}"),),
            annotations=(("family", "compression"), ("ratio", str(ratio))),
        )
    if operator.name == "kpool":
        pools, out_rows = Var("ROWS"), Var("POOL_OUT")
        reads = Mul((pools, Const(ratio * row_bytes)))
        writes = Mul((out_rows, Const(row_bytes)))
        vector = Mul((pools, Const(ratio * metrics["width"] * 2)))
        main = (
            _demand(ctx, "vector", vector),
            _demand(ctx, "mte2", reads),
            _demand(ctx, "mte3", writes),
            _demand(ctx, "hbm", Add((reads, writes))),
        )
        return FamilyTemplate(
            axes=axes, stages=(main,),
            materialization=(("pooled_index", writes),),
            internal=(), residency=Const(0),
            derived=(("new_rows", pools), ("pool_rows", pools),
                     ("pool_out_rows", out_rows),
                     ("selected_pools", pools),
                     ("remainder", Var("REMAINDER"))),
            regime_exprs=(("REMAINDER", Var("REMAINDER")),),
            binder=binder,
            annotations=(("family", "kpool"), ("ratio", str(ratio)),
                         ("always_tail", str(int(theta.find("always_tail", 0) or 0)))),
        )
    tail_rows = Var("TAIL_ROWS")
    writes = Mul((tail_rows, Const(row_bytes)))
    main = (
        _demand(ctx, "mte3", writes),
        _demand(ctx, "hbm", writes),
        _demand(ctx, "vector", Mul((tail_rows, Const(metrics["width"])))),
    )
    return FamilyTemplate(
        axes=axes, stages=(main,),
        materialization=(("tail_kv", writes),),
        internal=(), residency=Const(0),
        derived=(("tail_rows", tail_rows),
                 ("remainder", Var("REMAINDER"))),
        regime_exprs=(("REMAINDER", Var("REMAINDER")),),
        binder=binder,
        annotations=(("family", "compression_tail"), ("ratio", str(ratio))),
    )


__all__ = [
    "BURST_BYTES", "DESCRIPTOR_BYTES", "HEAD_REDUCE_M", "PAGE_ROWS",
    "align", "gather_bytes", "range_burst_bytes", "k1_new_rows",
    "selected_token_rows", "index_scanned_rows", "state_matrix_bytes",
    "compile_recurrent", "_compile_sinkhorn", "compile_indexer", "compile_compression",
]
