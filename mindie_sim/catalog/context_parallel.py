"""Context ownership and collective geometry.

[Role] Partition DCP key positions, PCP zigzag query ranges and DSA-CP
  query-token TP chunks.
[Invariants] DCP reuses TP ranks; PCP adds a query-sharding axis; DSA-CP
  shards query tokens over the TP group with a full/replicated KV cache.
  Pair counts conserve the causal/windowed triangle and block ownership is
  explicit.
[Timeouts & Locks] Bounded arithmetic, no device or observation access.
[Side-Effect Policy] No external mutation.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
import heapq
import math
from typing import Any

from mindie_sim.schema.common import ModelingError, normal_max_order_z
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.model import ModelSpec

#: DCP strategy (kept as the historical string because plan consumers already
#: report it).  It shards KV and merges (out, LSE); DSA indexers are replicated.
CP_STRATEGY = "kv_context_shard_lse_merge"
DCP_STRATEGY = CP_STRATEGY
#: PCP strategy: local query chunks, full K/V gathered, final outputs (no merge).
PCP_STRATEGY = "pcp_query_split_kv_gather"
#: DSA-CP strategy: local query tokens over full/replicated KV; output tokens
#: are all-gathered after the layer.  It is not the DCP KV-shard/LSE plan.
DSA_CP_STRATEGY = "dsa_cp_query_token_sp_full_kv_all_gather"
CP_PLAN_DCP = "dcp"
CP_PLAN_PCP = "pcp"
CP_PLAN_PCP_DCP = "pcp_dcp"
CP_PLAN_DSA_CP = "dsa_cp"
CP_PLAN_NONE = "none"
NO_WINDOW = 0
CP_PLAN_KINDS = (CP_PLAN_DCP, CP_PLAN_PCP, CP_PLAN_PCP_DCP, CP_PLAN_DSA_CP)
#: Dual-phase strategy: PCP owns the prefill query shard and K/V gather, DCP
#: owns the decode KV context shard and the ``npu_attention_update`` LSE merge.
PCP_DCP_STRATEGY = "pcp_prefill_query_split_dcp_decode_kv_shard_lse_merge"
CP_INDEXER_POLICY = "replicated"
DSA_CP_INDEXER_POLICY = "replicated_cache_local_query_topk"
PCP_INDEXER_POLICY = "pcp_local_query_full_index_k_gather"
#: Attention kinds that own a PCP query-shard schedule. Projection-only layers
#: may sit beside them; a model still needs one real query-shard builder.  The
#: sparse/window-compressed families (DSA plus SWA/CSA/CSA2/HCA) own the same
#: schedule on mrv2 models: the local zigzag query chunk scores the gathered
#: full index-K prefix and the compressed/window KV rows are gathered across
#: the PCP group.
_CP_ATTENTION_TYPES = {
    "gqa", "mla", "dsa", "kda", "gdn", "projection_only",
    "window_compressed", "swa", "csa", "csa2", "hca",
}
#: Families that execute a real query-shard builder (a model needs one).
_CP_QUERY_SHARD_FAMILIES = {
    "gqa", "mla", "dsa", "kda", "gdn", "window_compressed",
    "swa", "csa", "csa2", "hca",
}
#: Families whose KV context DCP can shard and merge.
_CP_CONTEXT_FAMILIES = {
    "gqa", "mla", "dsa", "window_compressed", "swa", "csa", "csa2", "hca",
}
CP_UNMODELED_SCHEDULES = ("compact_kv_gather",)
DEFAULT_CP_INTERLEAVE = 128
_FP32_BYTES = 4.0
#: First-principles DSA/SFA sparse Top-K DCP rank-skew physical constants.
SINK_MAX_TOKENS = 128
SINK_FRACTION_DIVISOR = 16
SINK_MIN_TOKENS = 4
TAIL_MAX_TOKENS = 512
TAIL_FRACTION_DIVISOR = 4
SEMANTIC_COHERENCE_TOKENS = 64
_SPARSE_EPS = 1e-12
_BALANCE_ITER_FACTOR = 4


def strategy_for_plan(plan: str) -> str:
    """The declared execution strategy for one plan name."""
    return {
        CP_PLAN_DCP: DCP_STRATEGY,
        CP_PLAN_PCP: PCP_STRATEGY,
        CP_PLAN_PCP_DCP: PCP_DCP_STRATEGY,
        CP_PLAN_DSA_CP: DSA_CP_STRATEGY,
        CP_PLAN_NONE: CP_PLAN_NONE,
    }.get(str(plan), str(plan))


@dataclass(frozen=True)
class CpSplit:
    """Exact per-rank partition of one attention work quantity."""

    cp: int
    total: float
    per_rank: tuple[float, ...]
    max_rank: float
    unit: str = "pairs"

    @property
    def imbalance(self) -> float:
        """max_rank / balanced_share; 1.0 means a perfectly even split."""
        balanced = self.total / self.cp if self.cp else 0.0
        return self.max_rank / balanced if balanced > 0 else 1.0

    def metadata(self, prefix: str = "cp") -> tuple[tuple[str, Any], ...]:
        return (
            (f"{prefix}", self.cp),
            (f"{prefix}_strategy", CP_STRATEGY),
            (f"{prefix}_{self.unit}_total", float(self.total)),
            (f"{prefix}_{self.unit}_max_rank", float(self.max_rank)),
            (f"{prefix}_{self.unit}_per_rank", tuple(float(v) for v in self.per_rank)),
            (f"{prefix}_imbalance", float(self.imbalance)),
        )


def _block_stats(
    lo: int, hi: int, cp: int, interleave: int, rank: int,
) -> tuple[int, int]:
    """(count, sum of owned positions) for positions in ``[lo, hi]``.

    Closed form over block indices: owned blocks are ``rank + m*cp``; only the
    last block of a prefix can be partial.  This is the primitive every split
    below uses, so no caller loops over positions.
    """
    if hi < lo or hi < 0:
        return (0, 0)
    cp = max(1, int(cp))
    return (
        _owned_prefix(hi, cp, interleave, rank)[0]
        - _owned_prefix(lo - 1, cp, interleave, rank)[0],
        _owned_prefix(hi, cp, interleave, rank)[1]
        - _owned_prefix(lo - 1, cp, interleave, rank)[1],
    )


def _owned_prefix(t: int, cp: int, block: int, rank: int) -> tuple[int, int]:
    """(count, sum) of positions ``<= t`` owned by ``rank``."""
    if t < 0:
        return (0, 0)
    t, cp, block, rank = int(t), int(cp), int(block), int(rank)
    last_block = t // block
    if last_block < rank:
        return (0, 0)
    complete = (last_block - rank) // cp
    count = complete * block
    # Full owned blocks m=0..complete-1: B^2*(r*m + cp*m*(m-1)/2) + m*B*(B-1)/2.
    total = (
        block * block * (rank * complete + cp * complete * (complete - 1) // 2)
        + complete * block * (block - 1) // 2
    )
    start = (rank + complete * cp) * block
    tail = min(block, t - start + 1)
    if tail > 0:
        count += tail
        total += tail * (start + start + tail - 1) // 2
    return (count, total)


def _affine_sum(
    lo: int, hi: int, cp: int, interleave: int, rank: int, slope: int, offset: int,
) -> int:
    """sum over owned positions k in [lo, hi] of ``slope*k + offset``."""
    if hi < lo:
        return 0
    count, position_sum = _block_stats(lo, hi, cp, interleave, rank)
    return int(slope) * position_sum + int(offset) * count


def owned_positions(lo: int, hi: int, cp: int, interleave: int, rank: int) -> int:
    """Number of positions in ``[lo, hi]`` owned by ``rank``."""
    return _block_stats(int(lo), int(hi), int(cp), int(interleave), int(rank))[0]


def split_causal_pairs(
    q_len: int, context: int, cp: int, interleave: int,
    *, window: int = NO_WINDOW, noncausal: bool = False, padding: bool = False,
) -> CpSplit:
    """Exact causal/windowed pair partition across ``cp`` context ranks.

    Row ``i`` (0-based) sits at absolute position ``context + i`` and sees
    ``[max(0, t-window+1), t]`` (``window <= 0`` means the whole prefix).  A key
    contributes one pair per visible row, so summing that row count over the
    owned key positions is the exact per-rank pair count -- including the
    triangular tail that a naive ``pairs/cp`` hides.
    """
    cp = max(1, int(cp))
    q_len = max(0, int(q_len))
    context = max(0, int(context))
    interleave = max(1, int(interleave))
    if cp == 1:
        total = _pair_total(q_len, context, window, noncausal, padding)
        return CpSplit(1, total, (total,), total)
    if padding:
        rows = context + 1
        per = tuple(float(q_len * owned_positions(0, context, cp, interleave, r))
                    for r in range(cp))
        total = float(q_len * rows)
    elif noncausal:
        rows = context + q_len
        win = int(window)
        lo = max(0, context - win) if win > 0 else 0
        visible = min(rows, win + q_len) if win > 0 else rows
        per = tuple(float(q_len * owned_positions(lo, rows - 1, cp, interleave, r))
                    for r in range(cp))
        total = float(q_len * visible)
    elif int(window) <= 0:
        per = tuple(float(_causal_rank_pairs(q_len, context, cp, interleave, r))
                    for r in range(cp))
        total = float(q_len * context + q_len * (q_len + 1) // 2)
    else:
        per = tuple(float(_window_rank_pairs(
            q_len, context, cp, interleave, r, int(window),
        )) for r in range(cp))
        total = _causal_pairs(q_len, context, int(window))
    return CpSplit(
        cp=cp, total=total, per_rank=per, max_rank=max(per, default=0.0),
    )


def _pair_total(
    q_len: int, context: int, window: int, noncausal: bool, padding: bool,
) -> float:
    if padding:
        return float(q_len * (context + 1))
    if noncausal:
        rows = context + q_len
        win = int(window)
        visible = min(rows, win + q_len) if win > 0 else rows
        return float(q_len * visible)
    if window <= 0:
        return float(q_len * context + q_len * (q_len + 1) // 2)
    return _causal_pairs(q_len, context, window)


def _causal_pairs(q_len: int, context: int, window: int) -> float:
    """Exact ``sum_{i=1..q} min(context+i, window)`` (window <= 0 = no window)."""
    if window <= 0:
        return float(q_len * context + q_len * (q_len + 1) / 2.0)
    if context >= window:
        return float(q_len * window)
    full = min(q_len, window - context)
    return float(
        full * context + full * (full + 1) / 2.0 + (q_len - full) * window
    )


def _causal_rank_pairs(
    q_len: int, context: int, cp: int, interleave: int, rank: int,
) -> int:
    # Keys below the context are visible to every row; the tail keys of the
    # current chunk are visible to fewer rows each.
    total = q_len * owned_positions(0, context - 1, cp, interleave, rank)
    if q_len:
        # sum_{k owned in [context, context+q-1]} (context + q - k)
        total += _affine_sum(
            context, context + q_len - 1, cp, interleave, rank,
            -1, context + q_len,
        )
    return total


def _window_rank_pairs(
    q_len: int, context: int, cp: int, interleave: int, rank: int, window: int,
) -> int:
    """Keys contributing to a sliding causal window, partition-exact."""
    total = 0
    # Keys before the first query position: rows 0..(k+window-1-context).
    r1_lo = max(0, context - window + 1)
    r1_hi = context - 1
    if r1_hi >= r1_lo:
        r1a_hi = min(r1_hi, context + q_len - window - 1)
        if r1a_hi >= r1_lo:
            # count(k) = k + window - context
            total += _affine_sum(r1_lo, r1a_hi, cp, interleave, rank, 1, window - context)
        r1b_lo = max(r1_lo, context + q_len - window)
        if r1_hi >= r1b_lo:
            total += _affine_sum(r1b_lo, r1_hi, cp, interleave, rank, 0, q_len)
    # Keys inside the current chunk: full window until the final window-1 keys.
    r2_lo = context
    r2_hi = context + q_len - 1
    if r2_hi >= r2_lo:
        r2a_hi = min(r2_hi, context + q_len - window)
        if r2a_hi >= r2_lo:
            total += _affine_sum(r2_lo, r2a_hi, cp, interleave, rank, 0, window)
        r2b_lo = max(r2_lo, context + q_len - window + 1)
        if r2_hi >= r2b_lo:
            total += _affine_sum(r2b_lo, r2_hi, cp, interleave, rank, -1, context + q_len)
    return total


def split_rows(rows: int, cp: int, unit: str = "rows") -> CpSplit:
    """Split a sparse selected-row count across ranks (uniform ownership)."""
    rows = max(0, int(rows))
    cp = max(1, int(cp))
    base, extra = divmod(rows, cp)
    per = tuple(float(base + (1 if rank < extra else 0)) for rank in range(cp))
    total = float(rows)
    if cp == 1:
        return CpSplit(1, total, (total,), total, unit)
    return CpSplit(cp, total, per, max(per, default=0.0), unit)


# ---------------------------------------------------------------------------
# Sparse Top-K DCP straggler rank skew (Sink + Local Tail + Semantic Variance)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SparseDcpSequence:
    """One sequence's sparse KV selection geometry for DCP rank partitioning."""

    kv_len: int
    selected_rows: float
    q_weight: float = 1.0
    compress_ratio: int = 1
    always_select_tail: int = 0


def sparse_dcp_sequence_for_total(
    kv_len: int,
    total_work: float,
    q_len: int,
    *,
    compress_ratio: int = 1,
    always_select_tail: int = 0,
) -> SparseDcpSequence:
    """Map a per-view total sparse work quantity into a ``SparseDcpSequence``."""
    ratio = max(1, int(compress_ratio))
    raw_kv = max(0, int(kv_len))
    eff_rows = (raw_kv + ratio - 1) // ratio if raw_kv > 0 else 0
    total = max(0.0, float(total_work))
    q_w = float(max(1, int(q_len)))
    tail = max(0, int(always_select_tail))
    if eff_rows <= 0 or total <= 0.0:
        return SparseDcpSequence(raw_kv, 0.0, q_w, ratio, tail)
    per_q = total / q_w
    if per_q <= float(eff_rows):
        return SparseDcpSequence(raw_kv, per_q, q_w, ratio, tail)
    return SparseDcpSequence(raw_kv, float(eff_rows), total / float(eff_rows), ratio, tail)


def dcp_sparse_imbalance_override(runtime: Any) -> float | None:
    """Resolve optional explicit sparse DCP imbalance multiplier (>= 1.0)."""
    raw = runtime_cp_flag(runtime, "dcp_sparse_imbalance", "sfa_dcp_imbalance")
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ModelingError(f"dcp_sparse_imbalance_invalid:{raw!r}")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ModelingError(f"dcp_sparse_imbalance_invalid:{raw!r}") from exc
    if not math.isfinite(value) or value < 1.0:
        raise ModelingError(f"dcp_sparse_imbalance_invalid:{raw!r}")
    return value


def _sparse_sequence_moments(
    seq: SparseDcpSequence, cp: int, interleave: int,
) -> tuple[list[float], list[float], list[float], list[float], list[float], float, list[float]]:
    """Per-rank weighted (cap, pinned, sem_mean, sem_var, residual_cap, total, hard_pinned)."""
    ratio = max(1, int(seq.compress_ratio))
    raw_kv = max(0, int(seq.kv_len))
    domain = (raw_kv + ratio - 1) // ratio if raw_kv > 0 else 0
    block = max(1, int(interleave) // ratio)
    weight = max(0.0, float(seq.q_weight))
    sel = min(float(domain), max(0.0, float(seq.selected_rows)))
    zeros = [0.0] * cp
    if domain <= 0 or sel <= 0.0 or weight <= 0.0:
        return (list(zeros), list(zeros), list(zeros), list(zeros), list(zeros), 0.0, list(zeros))
    raw_caps = [owned_positions(0, domain - 1, cp, block, r) for r in range(cp)]
    caps = [weight * float(c) for c in raw_caps]
    if sel >= float(domain) - 1e-12:
        return (caps, list(caps), list(zeros), list(zeros), list(zeros), weight * float(domain), list(caps))
    k_int = int(math.floor(sel))
    forced_tail = min(domain, k_int, max(0, int(seq.always_select_tail)))
    sink_budget = max(0, min(domain, k_int) - forced_tail)
    sink_cap = SINK_MAX_TOKENS // ratio
    sink_floor = (
        min(SINK_MIN_TOKENS, sink_budget)
        if ratio == 1
        else min(SINK_MIN_TOKENS // ratio, sink_budget)
    )
    sink_rows = min(
        sink_budget, max(sink_floor, min(sink_cap, k_int // SINK_FRACTION_DIVISOR)),
    )
    tail_cap = TAIL_MAX_TOKENS // ratio
    tail_target = max(forced_tail, min(tail_cap, k_int // TAIL_FRACTION_DIVISOR))
    tail_rows = min(domain - sink_rows, max(0, k_int - sink_rows), tail_target)
    a_sink = [
        owned_positions(0, sink_rows - 1, cp, block, r) if sink_rows > 0 else 0
        for r in range(cp)
    ]
    a_tail = [
        owned_positions(domain - tail_rows, domain - 1, cp, block, r) if tail_rows > 0 else 0
        for r in range(cp)
    ]
    a_forced_tail = [
        owned_positions(domain - forced_tail, domain - 1, cp, block, r) if forced_tail > 0 else 0
        for r in range(cp)
    ]
    res_raw = [raw_caps[r] - a_sink[r] - a_tail[r] for r in range(cp)]
    res_total = sum(res_raw)
    k_sem = max(0.0, sel - float(sink_rows) - float(tail_rows))
    pinned = [weight * float(a_sink[r] + a_tail[r]) for r in range(cp)]
    hard_pinned = [weight * float(a_forced_tail[r]) for r in range(cp)]
    residual_caps = [weight * float(res_raw[r]) for r in range(cp)]
    if k_sem <= 0.0 or res_total <= 0:
        return (caps, pinned, list(zeros), list(zeros), residual_caps, weight * sel, hard_pinned)
    g_eff = min(float(block), float(max(1, SEMANTIC_COHERENCE_TOKENS // ratio)), k_sem)
    fpc = max(0.0, 1.0 - k_sem / float(res_total))
    w_sq = weight * weight
    sem_means: list[float] = []
    variances: list[float] = []
    for r in range(cp):
        p_r = float(res_raw[r]) / float(res_total)
        sem_means.append(weight * k_sem * p_r)
        variances.append(w_sq * k_sem * g_eff * p_r * (1.0 - p_r) * fpc)
    return (caps, pinned, sem_means, variances, residual_caps, weight * sel, hard_pinned)


def _waterfill_with_floors(
    remaining: float,
    floors: Sequence[float],
    caps: Sequence[float],
    *,
    exclude: int = -1,
) -> list[float]:
    """Waterfill ``remaining`` movable rows to minimize peak ``floor[r] + add[r] <= cap[r]``."""
    cp = len(caps)
    alloc = [0.0 if r == exclude else float(floors[r]) for r in range(cp)]
    rem = max(0.0, float(remaining))
    movable = [r for r in range(cp) if r != exclude and alloc[r] < float(caps[r]) - 1e-12]
    while rem > 1e-12 and movable:
        min_lvl = min(alloc[r] for r in movable)
        lowest = [r for r in movable if alloc[r] <= min_lvl + 1e-12]
        upper_cands = [float(caps[r]) for r in lowest] + [
            alloc[r] for r in movable if alloc[r] > min_lvl + 1e-12
        ]
        next_lvl = min(upper_cands)
        step = (next_lvl - min_lvl) * float(len(lowest))
        if rem <= step + 1e-12:
            share = rem / float(len(lowest))
            for r in lowest:
                alloc[r] = min(float(caps[r]), alloc[r] + share)
            rem = 0.0
            break
        for r in lowest:
            alloc[r] = next_lvl
        rem = max(0.0, rem - step)
        movable = [r for r in movable if alloc[r] < float(caps[r]) - 1e-12]
    return alloc


def _waterfill_sequence_on_batch(
    alloc_i: list[float],
    batch_load: list[float],
    caps_i: Sequence[float],
    remaining: float,
    active_ranks: tuple[int, ...],
) -> None:
    """Waterfill one sequence's movable rows against cumulative batch load."""
    rem = max(0.0, float(remaining))
    movable = [r for r in active_ranks if alloc_i[r] < float(caps_i[r]) - _SPARSE_EPS]
    while rem > _SPARSE_EPS and movable:
        min_lvl = min(batch_load[r] for r in movable)
        lowest = [r for r in movable if batch_load[r] <= min_lvl + _SPARSE_EPS]
        upper_cands = [
            batch_load[r] + (float(caps_i[r]) - alloc_i[r]) for r in lowest
        ] + [batch_load[r] for r in movable if batch_load[r] > min_lvl + _SPARSE_EPS]
        next_lvl = min(upper_cands)
        step = (next_lvl - min_lvl) * float(len(lowest))
        add = (rem / float(len(lowest))) if rem <= step + _SPARSE_EPS else (next_lvl - min_lvl)
        if add <= _SPARSE_EPS:
            break
        for r in lowest:
            actual = min(float(caps_i[r]) - alloc_i[r], add)
            alloc_i[r] += actual
            batch_load[r] += actual
            rem = max(0.0, rem - actual)
        movable = [r for r in active_ranks if alloc_i[r] < float(caps_i[r]) - _SPARSE_EPS]


def _find_rebalance_path(
    allocs: Sequence[Sequence[float]],
    hard_floors: Sequence[Sequence[float]],
    caps: Sequence[Sequence[float]],
    batch_load: Sequence[float],
    active_ranks: tuple[int, ...],
    src: int,
    target_max: float,
) -> tuple[int, dict[int, tuple[int, int, float]]] | None:
    """Find a residual augmenting path from peak rank ``src`` to a lower-load rank."""
    parent: dict[int, tuple[int, int, float]] = {}
    visited = {src}
    queue = [src]
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for seq_idx, alloc_i in enumerate(allocs):
            give = alloc_i[u] - float(hard_floors[seq_idx][u])
            if give <= _SPARSE_EPS:
                continue
            caps_i = caps[seq_idx]
            for v in active_ranks:
                if v in visited or float(caps_i[v]) - alloc_i[v] <= _SPARSE_EPS:
                    continue
                edge_cap = min(give, float(caps_i[v]) - alloc_i[v])
                visited.add(v)
                parent[v] = (u, seq_idx, edge_cap)
                if batch_load[v] < target_max - _SPARSE_EPS:
                    return v, parent
                queue.append(v)
    return None


def _minimax_balance_batch(
    allocs: list[list[float]],
    rems: Sequence[float],
    hard_floors: Sequence[Sequence[float]],
    caps: Sequence[Sequence[float]],
    active_ranks: tuple[int, ...],
    cp: int,
) -> list[float]:
    """Jointly waterfill and minimax-balance batch allocations across ``active_ranks``."""
    batch_load = [sum(row[r] for row in allocs) for r in range(cp)]
    if not active_ranks:
        return batch_load
    order = sorted(
        range(len(allocs)),
        key=lambda i: (
            sum(max(0.0, float(caps[i][r]) - allocs[i][r]) for r in active_ranks) - float(rems[i]),
            i,
        ),
    )
    for i in order:
        if rems[i] > _SPARSE_EPS:
            _waterfill_sequence_on_batch(allocs[i], batch_load, caps[i], rems[i], active_ranks)
    max_iters = len(active_ranks) * max(1, len(allocs)) * _BALANCE_ITER_FACTOR
    for _ in range(max_iters):
        hi = max(active_ranks, key=lambda r: batch_load[r])
        lo_val = min(batch_load[r] for r in active_ranks)
        if batch_load[hi] - lo_val <= _SPARSE_EPS:
            break
        found = _find_rebalance_path(
            allocs, hard_floors, caps, batch_load, active_ranks, hi, batch_load[hi],
        )
        if found is None:
            break
        dst, parent = found
        path: list[tuple[int, int, int, float]] = []
        cur = dst
        while cur != hi:
            u, seq_idx, edge_cap = parent[cur]
            path.append((u, cur, seq_idx, edge_cap))
            cur = u
        bottleneck = min(item[3] for item in path)
        delta = min(bottleneck, (batch_load[hi] - batch_load[dst]) / 2.0)
        if delta <= _SPARSE_EPS:
            break
        for u, v, seq_idx, _ in path:
            allocs[seq_idx][u] -= delta
            allocs[seq_idx][v] += delta
        batch_load[hi] -= delta
        batch_load[dst] += delta
    return batch_load


def _apply_sparse_imbalance_override(
    moments: Sequence[tuple[list[float], list[float], list[float], list[float], list[float], float, list[float]]],
    cp: int,
    alpha: float,
    total: float,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Place ``min(hi[r*], alpha * total / cp)`` on highest-capacity rank ``r*`` respecting hard pins."""
    caps_list = [m[0] for m in moments]
    hard_list = [m[6] for m in moments]
    if alpha <= 1.0 + _SPARSE_EPS:
        allocs = [list(hard_i) for hard_i in hard_list]
        rems = [max(0.0, m[5] - sum(m[6])) for m in moments]
        batch_load = _minimax_balance_batch(
            allocs, rems, hard_list, caps_list, tuple(range(cp)), cp,
        )
        return tuple(batch_load), tuple(tuple(row) for row in allocs)
    eff_caps = [
        sum(
            m[6][r] + min(max(0.0, m[0][r] - m[6][r]), max(0.0, m[5] - sum(m[6])))
            for m in moments
        )
        for r in range(cp)
    ]
    raw_caps = [sum(m[0][r] for m in moments) for r in range(cp)]
    r_star = max(range(cp), key=lambda r: (eff_caps[r], raw_caps[r]))
    bounds: list[tuple[float, float]] = []
    for caps_i, _, _, _, _, tot_i, hard_i in moments:
        rem_tot_i = max(0.0, tot_i - sum(hard_i))
        res_i = [max(0.0, caps_i[r] - hard_i[r]) for r in range(cp)]
        other_res = sum(res_i[r] for r in range(cp) if r != r_star)
        lo_i = hard_i[r_star] + max(0.0, rem_tot_i - other_res)
        hi_i = hard_i[r_star] + min(res_i[r_star], rem_tot_i)
        bounds.append((lo_i, hi_i))
    sum_lo = sum(lo for lo, _ in bounds)
    sum_hi = sum(hi for _, hi in bounds)
    k_star_batch = min(sum_hi, max(sum_lo, alpha * (total / float(cp))))
    span = sum_hi - sum_lo
    frac = (k_star_batch - sum_lo) / span if span > _SPARSE_EPS else 0.0
    allocs = []
    rems = []
    for (_, _, _, _, _, tot_i, hard_i), (lo_i, hi_i) in zip(moments, bounds):
        k_star_i = lo_i + (hi_i - lo_i) * frac
        row = list(hard_i)
        row[r_star] = k_star_i
        other_hard = sum(hard_i[r] for r in range(cp) if r != r_star)
        allocs.append(row)
        rems.append(max(0.0, tot_i - k_star_i - other_hard))
    active_ranks = tuple(r for r in range(cp) if r != r_star)
    batch_load = _minimax_balance_batch(allocs, rems, hard_list, caps_list, active_ranks, cp)
    return tuple(batch_load), tuple(tuple(row) for row in allocs)


def _apply_sparse_default_skew(
    moments: Sequence[tuple[list[float], list[float], list[float], list[float], list[float], float, list[float]]],
    cp: int,
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Pool deterministic means and semantic variances across batch sequences."""
    batch_mean = [sum(m[1][r] + m[2][r] for m in moments) for r in range(cp)]
    batch_var = [sum(m[3][r] for m in moments) for r in range(cp)]
    batch_res = [sum(m[4][r] for m in moments) for r in range(cp)]
    batch_caps = [sum(m[0][r] for m in moments) for r in range(cp)]
    active_ranks = sum(1 for r in range(cp) if batch_res[r] > 1e-12)
    z_factor = normal_max_order_z(active_ranks)
    r_star = max(
        range(cp),
        key=lambda r: (batch_mean[r] + z_factor * math.sqrt(max(0.0, batch_var[r])), batch_caps[r]),
    )
    seq_shifts: list[tuple[float, float]] = []
    for caps_i, pinned_i, sem_i, _, _, _, _ in moments:
        mu_star = pinned_i[r_star] + sem_i[r_star]
        sem_other = sum(sem_i[r] for r in range(cp) if r != r_star)
        room = max(0.0, caps_i[r_star] - mu_star)
        seq_shifts.append((min(room, sem_other), sem_other))
    max_shift_total = sum(shift for shift, _ in seq_shifts)
    delta = min(max_shift_total, z_factor * math.sqrt(max(0.0, batch_var[r_star])))
    per_seq: list[tuple[float, ...]] = []
    for (_, pinned_i, sem_i, _, _, _, _), (max_shift_i, sem_other_i) in zip(moments, seq_shifts):
        alloc_i = [pinned_i[r] + sem_i[r] for r in range(cp)]
        if delta > 1e-12 and max_shift_total > 1e-12 and max_shift_i > 1e-12:
            d_i = delta * (max_shift_i / max_shift_total)
            alloc_i[r_star] += d_i
            if sem_other_i > 1e-12:
                scale = d_i / sem_other_i
                for r in range(cp):
                    if r != r_star:
                        alloc_i[r] = max(0.0, alloc_i[r] - sem_i[r] * scale)
        per_seq.append(tuple(alloc_i))
    batch_per_rank = tuple(sum(row[r] for row in per_seq) for r in range(cp))
    return batch_per_rank, tuple(per_seq)


def split_sparse_dcp_batch_per_sequence(
    sequences: Sequence[SparseDcpSequence],
    cp: int,
    interleave: int,
    *,
    imbalance_override: float | None = None,
    runtime: Any = None,
    unit: str = "pairs",
) -> tuple[CpSplit, tuple[tuple[float, ...], ...]]:
    """Batch-pooled sparse Top-K DCP rank partition with per-sequence breakdown."""
    degree = max(1, int(cp))
    block = max(1, int(interleave))
    seq_list = tuple(sequences)
    override = (
        dcp_sparse_imbalance_override(runtime)
        if imbalance_override is None
        else float(imbalance_override)
    )
    if override is not None and (not math.isfinite(override) or override < 1.0):
        raise ModelingError(f"dcp_sparse_imbalance_invalid:{override!r}")
    if degree <= 1:
        per_seq = tuple(
            (max(0.0, float(s.q_weight)) * min(
                float((max(0, int(s.kv_len)) + max(1, int(s.compress_ratio)) - 1)
                      // max(1, int(s.compress_ratio)))
                if int(s.kv_len) > 0 else 0.0,
                max(0.0, float(s.selected_rows)),
            ),)
            for s in seq_list
        )
        total = float(sum(row[0] for row in per_seq))
        return CpSplit(1, total, (total,), total, unit), per_seq
    if not seq_list:
        zeros = (0.0,) * degree
        return CpSplit(degree, 0.0, zeros, 0.0, unit), ()
    moments = tuple(_sparse_sequence_moments(seq, degree, block) for seq in seq_list)
    total = float(sum(m[5] for m in moments))
    if total <= 0.0:
        zeros = (0.0,) * degree
        return CpSplit(degree, 0.0, zeros, 0.0, unit), tuple(zeros for _ in seq_list)
    if override is not None:
        batch_per_rank, per_seq = _apply_sparse_imbalance_override(moments, degree, override, total)
    else:
        batch_per_rank, per_seq = _apply_sparse_default_skew(moments, degree)
    return (
        CpSplit(degree, total, batch_per_rank, max(batch_per_rank, default=0.0), unit),
        per_seq,
    )


def split_sparse_dcp_batch(
    sequences: Sequence[SparseDcpSequence],
    cp: int,
    interleave: int,
    *,
    imbalance_override: float | None = None,
    runtime: Any = None,
    unit: str = "pairs",
) -> CpSplit:
    """Batch-pooled sparse Top-K DCP rank partition across ``sequences``."""
    return split_sparse_dcp_batch_per_sequence(
        sequences, cp, interleave,
        imbalance_override=imbalance_override, runtime=runtime, unit=unit,
    )[0]


def split_sparse_dcp_ranks(
    kv_len: int,
    selected_rows: float,
    cp: int,
    interleave: int,
    *,
    compress_ratio: int = 1,
    always_select_tail: int = 0,
    q_weight: float = 1.0,
    imbalance_override: float | None = None,
    runtime: Any = None,
    unit: str = "pairs",
) -> CpSplit:
    """First-principles sparse Top-K DCP rank skew for one sequence."""
    seq = SparseDcpSequence(
        kv_len=int(kv_len),
        selected_rows=float(selected_rows),
        q_weight=float(q_weight),
        compress_ratio=max(1, int(compress_ratio)),
        always_select_tail=max(0, int(always_select_tail)),
    )
    return split_sparse_dcp_batch(
        (seq,), cp, interleave,
        imbalance_override=imbalance_override, runtime=runtime, unit=unit,
    )


def sparse_dcp_capacities(
    seq: SparseDcpSequence, cp: int, interleave: int,
) -> list[int]:
    """Per-rank integer row capacities ``ceil(q_weight * C_{i, r})`` for ``seq``."""
    degree = max(1, int(cp))
    ratio = max(1, int(seq.compress_ratio))
    raw_kv = max(0, int(seq.kv_len))
    domain = (raw_kv + ratio - 1) // ratio if raw_kv > 0 else 0
    block = max(1, int(interleave) // ratio)
    weight = max(0.0, float(seq.q_weight))
    if domain <= 0 or weight <= 0.0:
        return [0] * degree
    return [
        int(math.ceil(weight * float(owned_positions(0, domain - 1, degree, block, r))))
        for r in range(degree)
    ]


def round_sparse_dcp_counts(
    per_rank: Sequence[float],
    total: int,
    capacities: Sequence[int] | None = None,
) -> list[int]:
    """Largest-remainder integer rounding conserving ``total`` within ``capacities``."""
    cp = len(per_rank)
    target = max(0, int(total))
    if cp == 0:
        return []
    caps = (
        [max(0, int(c)) for c in capacities]
        if capacities is not None
        else [target] * cp
    )
    base = [min(caps[r], max(0, int(math.floor(per_rank[r])))) for r in range(cp)]
    rem = min(target, sum(caps)) - sum(base)
    if rem > 0:
        order = sorted(
            (r for r in range(cp) if base[r] < caps[r]),
            key=lambda r: (per_rank[r] - base[r], per_rank[r], -r),
            reverse=True,
        )
        for r in order:
            if rem <= 0:
                break
            add = min(rem, caps[r] - base[r])
            if add > 0:
                base[r] += 1
                rem -= 1
        while rem > 0:
            progressed = False
            for r in order:
                if rem <= 0:
                    break
                if base[r] < caps[r]:
                    base[r] += 1
                    rem -= 1
                    progressed = True
            if not progressed:
                break
    return base


def _normalize_feasible_row_fractions(
    raw_row: Sequence[float], target: int, caps: Sequence[int],
) -> list[Fraction]:
    """Construct exact rational row entries in ``[0, caps[r]]`` summing to ``target``."""
    width = len(caps)
    if target <= 0:
        return [Fraction(0)] * width
    vals: list[Fraction] = []
    for r in range(width):
        raw = float(raw_row[r])
        if not math.isfinite(raw):
            raise ModelingError(f"dcp_sparse_nonfinite_entry:{raw!r}")
        vals.append(min(Fraction(caps[r]), max(Fraction(0), Fraction(raw))))
    total = sum(vals, Fraction(0))
    if total == target:
        return vals
    if total < target:
        deficit = Fraction(target) - total
        headroom = [Fraction(caps[r]) - vals[r] for r in range(width)]
        cap_rem = sum(headroom, Fraction(0))
        if cap_rem <= 0:
            return vals
        return [vals[r] + deficit * headroom[r] / cap_rem for r in range(width)]
    scale = Fraction(target) / total
    return [v * scale for v in vals]


def _add_flow_edge(
    adj: list[list[list[int]]], u: int, v: int, cap: int, cost: int,
) -> int:
    """Append forward/reverse residual edge pair and return forward edge index."""
    fwd_idx = len(adj[u])
    rev_idx = len(adj[v])
    adj[u].append([v, rev_idx, int(cap), int(cost)])
    adj[v].append([u, fwd_idx, 0, -int(cost)])
    return fwd_idx


def _solve_dag_min_cost_flow(
    n_nodes: int, src: int, snk: int, adj: list[list[list[int]]], target_flow: int,
) -> int:
    """Exact integer successive-shortest-path min-cost flow on a DAG-ordered network."""
    if target_flow <= 0:
        return 0
    pot: list[int | None] = [None] * n_nodes
    pot[src] = 0
    for u in range(n_nodes):
        pu = pot[u]
        if pu is None:
            continue
        for v, _, cap, cost in adj[u]:
            if cap > 0 and (pot[v] is None or pu + cost < pot[v]):
                pot[v] = pu + cost
    pushed = 0
    while pushed < target_flow:
        dist: list[int | None] = [None] * n_nodes
        parent_u = [-1] * n_nodes
        parent_e = [-1] * n_nodes
        dist[src] = 0
        heap: list[tuple[int, int]] = [(0, src)]
        while heap:
            du, u = heapq.heappop(heap)
            if dist[u] is None or du != dist[u]:
                continue
            pu = pot[u]
            assert pu is not None
            for ei, (v, _, cap, cost) in enumerate(adj[u]):
                if cap <= 0:
                    continue
                pv = pot[v]
                if pv is None:
                    continue
                nd = du + cost + pu - pv
                if dist[v] is None or nd < dist[v]:
                    dist[v] = nd
                    parent_u[v] = u
                    parent_e[v] = ei
                    heapq.heappush(heap, (nd, v))
        if dist[snk] is None:
            break
        for v in range(n_nodes):
            dv = dist[v]
            if dv is not None and pot[v] is not None:
                pot[v] += dv
        flow = target_flow - pushed
        cur = snk
        while cur != src:
            u = parent_u[cur]
            ei = parent_e[cur]
            flow = min(flow, adj[u][ei][2])
            cur = u
        cur = snk
        while cur != src:
            u = parent_u[cur]
            ei = parent_e[cur]
            rev = adj[u][ei][1]
            adj[u][ei][2] -= flow
            adj[cur][rev][2] += flow
            cur = u
        pushed += flow
    return pushed


def _round_sparse_dcp_prescribed_columns(
    exact_rows: list[list[Fraction]],
    row_targets: list[int],
    caps: list[list[int]],
    column_targets: Sequence[int],
) -> list[list[int]]:
    """Solve exact bounded Hoffman rounding for caller-prescribed integer column sums."""
    rows = len(exact_rows)
    cols = len(caps[0])
    if len(column_targets) != cols:
        raise ModelingError("dcp_column_targets_infeasible:width_mismatch")
    col_t = [int(t) for t in column_targets]
    total_flow = sum(row_targets)
    if any(t < 0 for t in col_t) or sum(col_t) != total_flow:
        raise ModelingError("dcp_column_targets_infeasible:total_mismatch")
    floors = [[int(math.floor(exact_rows[i][r])) for r in range(cols)] for i in range(rows)]
    ceils = [
        [min(caps[i][r], int(math.ceil(exact_rows[i][r]))) for r in range(cols)]
        for i in range(rows)
    ]
    for r in range(cols):
        lo_sum = sum(floors[i][r] for i in range(rows))
        hi_sum = sum(ceils[i][r] for i in range(rows))
        if col_t[r] < lo_sum or col_t[r] > hi_sum:
            raise ModelingError(
                f"dcp_column_targets_out_of_rounding_bounds:col={r},target={col_t[r]},range=[{lo_sum},{hi_sum}]"
            )
    row_supplies = [row_targets[i] - sum(floors[i]) for i in range(rows)]
    col_demands = [col_t[r] - sum(floors[i][r] for i in range(rows)) for r in range(cols)]
    delta_flow = sum(row_supplies)
    if delta_flow == 0:
        return floors
    src = 0
    col_off = 1 + rows
    snk = col_off + cols
    adj: list[list[list[int]]] = [[] for _ in range(snk + 1)]
    tie_span = (rows + cols + 2) * (rows * cols + cols + 2)
    common_denom = math.lcm(*(f.denominator for row in exact_rows for f in row), 1)
    cost_scale = common_denom * tie_span * 2
    for i, supply in enumerate(row_supplies):
        if supply > 0:
            _add_flow_edge(adj, src, 1 + i, supply, 0)
    cell_edge: list[list[int]] = [[-1] * cols for _ in range(rows)]
    for i in range(rows):
        for r in range(cols):
            if floors[i][r] < ceils[i][r]:
                frac = exact_rows[i][r] - floors[i][r]
                cost = -int(frac * cost_scale) + (i * cols + r)
                cell_edge[i][r] = _add_flow_edge(adj, 1 + i, col_off + r, 1, cost)
    for r, demand in enumerate(col_demands):
        if demand > 0:
            _add_flow_edge(adj, col_off + r, snk, demand, 0)
    pushed = _solve_dag_min_cost_flow(snk + 1, src, snk, adj, delta_flow)
    if pushed != delta_flow:
        raise ModelingError("dcp_column_targets_infeasible")
    result = [list(row) for row in floors]
    for i in range(rows):
        for r in range(cols):
            ei = cell_edge[i][r]
            if ei >= 0 and adj[1 + i][ei][2] == 0:
                result[i][r] += 1
    return result


def round_sparse_dcp_matrix(
    per_seq: Sequence[Sequence[float]],
    seq_totals: Sequence[int],
    seq_caps: Sequence[Sequence[int]],
    *,
    column_targets: Sequence[int] | None = None,
) -> list[list[int]]:
    """Round per-sequence rank counts via bounded Hoffman min-cost integral circulation.

    Conserves exact row totals ``sum_r m_{i,r} == K_i``, integer capacities
    ``0 <= floor(x_{i,r}) <= m_{i,r} <= ceil(x_{i,r}) <= C_{i,r}``, and
    bounded column totals ``floor(T_r) <= sum_i m_{i,r} <= ceil(T_r)``.
    """
    rows = len(per_seq)
    if rows == 0:
        return []
    cols = len(per_seq[0])
    if cols == 0:
        return [[] for _ in range(rows)]
    caps = [[max(0, int(c)) for c in seq_caps[i]] for i in range(rows)]
    row_targets: list[int] = []
    for i in range(rows):
        raw_t = int(seq_totals[i])
        cap_sum = sum(caps[i])
        if raw_t < 0 or raw_t > cap_sum:
            raise ModelingError(
                f"dcp_row_target_exceeds_capacity:row={i},target={raw_t},cap={cap_sum}"
            )
        row_targets.append(raw_t)
    exact_rows = [
        _normalize_feasible_row_fractions(per_seq[i], row_targets[i], caps[i])
        for i in range(rows)
    ]
    if column_targets is not None:
        return _round_sparse_dcp_prescribed_columns(exact_rows, row_targets, caps, column_targets)

    floors = [[int(math.floor(exact_rows[i][r])) for r in range(cols)] for i in range(rows)]
    col_fracs = [sum((exact_rows[i][r] for i in range(rows)), Fraction(0)) for r in range(cols)]
    col_floors = [int(math.floor(col_fracs[r])) for r in range(cols)]
    total_k = sum(row_targets)
    row_supplies = [row_targets[i] - sum(floors[i]) for i in range(rows)]
    delta_flow = sum(row_supplies)
    if delta_flow == 0:
        return floors

    src = 0
    col_off = 1 + rows
    t_node = col_off + cols
    snk = t_node + 1
    adj: list[list[list[int]]] = [[] for _ in range(snk + 1)]
    tie_span = (rows + cols + 2) * (rows * cols + cols + 2)
    common_denom = math.lcm(
        *(f.denominator for row in exact_rows for f in row),
        *(cf.denominator for cf in col_fracs),
        1,
    )
    cost_scale = common_denom * tie_span * 2

    for i, supply in enumerate(row_supplies):
        if supply > 0:
            _add_flow_edge(adj, src, 1 + i, supply, 0)
    cell_edge: list[list[int]] = [[-1] * cols for _ in range(rows)]
    for i in range(rows):
        for r in range(cols):
            frac = exact_rows[i][r] - floors[i][r]
            if frac > 0 and floors[i][r] < caps[i][r]:
                cost = -int(frac * cost_scale) + (i * cols + r)
                cell_edge[i][r] = _add_flow_edge(adj, 1 + i, col_off + r, 1, cost)
    for r in range(cols):
        floor_sum_r = sum(floors[i][r] for i in range(rows))
        mandatory = col_floors[r] - floor_sum_r
        if mandatory > 0:
            _add_flow_edge(adj, col_off + r, snk, mandatory, 0)
        col_rem = col_fracs[r] - col_floors[r]
        if col_rem > 0:
            cost = -int(col_rem * cost_scale) + r
            _add_flow_edge(adj, col_off + r, t_node, 1, cost)
    t_demand = total_k - sum(col_floors)
    if t_demand > 0:
        _add_flow_edge(adj, t_node, snk, t_demand, 0)

    pushed = _solve_dag_min_cost_flow(snk + 1, src, snk, adj, delta_flow)
    if pushed != delta_flow:
        raise ModelingError(f"dcp_hoffman_circulation_infeasible:pushed={pushed},expected={delta_flow}")
    result = [list(row) for row in floors]
    for i in range(rows):
        for r in range(cols):
            ei = cell_edge[i][r]
            if ei >= 0 and adj[1 + i][ei][2] == 0:
                result[i][r] += 1
    return result


# ---------------------------------------------------------------------------
# PCP: prefill query split + full-KV gather, no partial-output merge
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PcpSplit:
    """Exact per-rank partition of prefill queries and causal pairs."""

    pcp: int
    q_len: int
    context: int
    total: float
    per_rank: tuple[float, ...]
    max_rank: float
    per_rank_tokens: tuple[int, ...]
    max_tokens: int
    #: ``(rank, chunk_index, start, length, pairs)`` for every non-empty chunk.
    chunks: tuple[tuple[int, int, int, int, float], ...] = ()
    strategy: str = PCP_STRATEGY

    @property
    def imbalance(self) -> float:
        balanced = self.total / self.pcp if self.pcp else 0.0
        return self.max_rank / balanced if balanced > 0 else 1.0

    def chunk_ranges(self, rank: int) -> tuple[tuple[int, int], ...]:
        return tuple(
            (start, start + length) for owner, _, start, length, _ in self.chunks
            if owner == int(rank) and length > 0
        )

    def metadata(self, prefix: str = "pcp") -> tuple[tuple[str, Any], ...]:
        return (
            (prefix, self.pcp),
            (f"{prefix}_strategy", self.strategy),
            (f"{prefix}_split", "dual_chunk_swap"),
            (f"{prefix}_chunks", int(2 * self.pcp)),
            (f"{prefix}_pairs_total", float(self.total)),
            (f"{prefix}_pairs_max_rank", float(self.max_rank)),
            (f"{prefix}_pairs_per_rank", tuple(float(v) for v in self.per_rank)),
            (f"{prefix}_tokens_per_rank", tuple(int(v) for v in self.per_rank_tokens)),
            (f"{prefix}_max_tokens", int(self.max_tokens)),
            (f"{prefix}_imbalance", float(self.imbalance)),
        )


def _pcp_chunk_bounds(q_len: int, pcp: int) -> tuple[int, ...]:
    """Contiguous bounds for ``2*pcp`` DualChunkSwap chunks matching upstream ``PCPManager``.

    Upstream sizes each mirrored chunk to ``ceil(q_len / (2*pcp))`` and clips
    trailing chunks to ``q_len``.  Real token sums stay exact (no invented tokens).
    """
    q_len = max(0, int(q_len))
    pcp = max(1, int(pcp))
    chunks = 2 * pcp
    chunk_size = (q_len + chunks - 1) // chunks if q_len > 0 else 0
    return tuple(min(index * chunk_size, q_len) for index in range(chunks + 1))


def pcp_rank_chunk_ranges(q_len: int, pcp: int, rank: int) -> tuple[tuple[int, int], ...]:
    """The contiguous ``(start, end)`` query ranges owned by one PCP rank."""
    bounds = _pcp_chunk_bounds(q_len, pcp)
    chunks = 2 * max(1, int(pcp))
    rank = int(rank)
    ranges: list[tuple[int, int]] = []
    for index in (rank, chunks - 1 - rank):
        start, end = bounds[index], bounds[index + 1]
        if end > start:
            ranges.append((start, end))
    return tuple(ranges)


def split_pcp_pairs(
    q_len: int, context: int, pcp: int, *, window: int = 0,
) -> PcpSplit:
    """Exact causal-pair partition for DualChunkSwap prefill chunks.

    Every query is owned by exactly one rank, so per-rank pairs sum to the CP1
    causal pair count.  The max rank is the latency-relevant work; no balance
    assumption is subtracted.
    """
    pcp = max(1, int(pcp))
    q_len = max(0, int(q_len))
    context = max(0, int(context))
    window = max(0, int(window))
    bounds = _pcp_chunk_bounds(q_len, pcp)
    chunks = 2 * pcp
    per_rank = [0.0] * pcp
    per_tokens = [0] * pcp
    chunk_rows: list[tuple[int, int, int, int, float]] = []
    for index in range(chunks):
        start, end = bounds[index], bounds[index + 1]
        if end <= start:
            continue
        rank = index if index < pcp else chunks - 1 - index
        length = end - start
        pairs = _causal_pairs(length, context + start, window)
        per_rank[rank] += pairs
        per_tokens[rank] += length
        chunk_rows.append((rank, index, start, length, pairs))
    total = float(sum(per_rank))
    return PcpSplit(
        pcp=pcp, q_len=q_len, context=context, total=total,
        per_rank=tuple(per_rank), max_rank=max(per_rank, default=0.0),
        per_rank_tokens=tuple(per_tokens), max_tokens=max(per_tokens, default=0),
        chunks=tuple(chunk_rows),
    )


# ---------------------------------------------------------------------------
# DSA-CP: query-token sequence parallel over the TP group
#
# Retained source: vllm_ascend/attention/context_parallel/dsa_cp.py
# (sha256 dceec1a878d0357f4cdd6647ff868d319d5e24eaa32eb55c27ea754b7d71b797).
# `_local_token_range` splits the flattened token stream into contiguous TP
# chunks padded to the TP group; `_build_local_token_metadata` intersects each
# request with the rank's chunk and truncates `local_seq_lens` to the local
# chunk end; `_indexer_select_topk` scores the local query tokens over the full
# index-K prefix; `_gather_cp_output` all-gathers the local output tokens.  The
# KV/index cache stays full/replicated.  This is not the DCP KV-shard/LSE plan.
# ---------------------------------------------------------------------------
def dsa_cp_token_ranges(tokens: int, tp: int, rank: int) -> tuple[int, int, int, int]:
    """The source's ``_local_token_range``: start, end, per-rank, padded total."""
    tokens = max(0, int(tokens))
    tp = max(1, int(tp))
    rank = int(rank)
    if not 0 <= rank < tp:
        raise ModelingError(f"dsa_cp_rank_invalid:{rank}:{tp}")
    padded = ((tokens + tp - 1) // tp) * tp
    per_rank = padded // tp
    start = rank * per_rank
    return start, start + per_rank, per_rank, padded


def dcp_indexer_sharded(model: Any, runtime: Any = None) -> bool:
    """True when sparse LightningIndexer cache is sharded across DCP ranks.

    By default ``vllm-ascend`` replicates the LightningIndexer cache across DCP
    ranks and remaps global Top-K indices via Triton ``sparse_index_remap``.
    Sharding is opt-in via model execution policy or runtime
    ``enable_sfa_dcp_sharded_indexer`` / ``dcp_indexer_sharded``.
    """
    if runtime is not None:
        meta = dict(getattr(runtime, "metadata", ()) or ())
        extra = dict(meta.get("additional_config") or ()) if isinstance(meta.get("additional_config"), dict) else {}
        for source in (meta, extra):
            for key in ("enable_sfa_dcp_sharded_indexer", "dcp_indexer_sharded"):
                if key in source and source[key] is not None:
                    return bool(source[key])
    from .paired_sfa import execution
    return bool(execution(model).get("dcp_indexer_sharded"))


def cp_interleave(runtime: Any) -> int:
    """Declared KV block interleave; defaults to the KV block size."""
    raw = dict(getattr(runtime, "metadata", ()) or ()).get(
        "cp_kv_cache_interleave_size"
    )
    if raw is not None:
        try:
            value = int(raw)
        except (TypeError, ValueError) as exc:
            raise ModelingError(f"cp_interleave_invalid:{raw}") from exc
        if value < 1:
            raise ModelingError(f"cp_interleave_invalid:{raw}")
        return value
    block = int(getattr(runtime, "kv_block_size", 0) or 0)
    return block if block > 0 else DEFAULT_CP_INTERLEAVE


def cp_requested(parallel: ParallelSpec) -> bool:
    return int(getattr(parallel, "cp", 1) or 1) > 1


def cp_plan_kind(parallel: ParallelSpec) -> str:
    """The declared plan, resolved defensively for objects built before this field."""
    plan = str(getattr(parallel, "cp_plan", CP_PLAN_NONE) or CP_PLAN_NONE)
    if plan in CP_PLAN_KINDS or plan == CP_PLAN_NONE:
        return plan
    return CP_PLAN_DCP if cp_requested(parallel) else CP_PLAN_NONE


def dcp_axis_factors(parallel: ParallelSpec) -> tuple[int, int, int]:
    """Factorize ``parallel.cp`` into ``(pcp_factor, tp_gather, dcp_dp_width)``."""
    from mindie_sim.schema.parallel import factor_dcp_axes

    return factor_dcp_axes(
        int(getattr(parallel, "cp", 1) or 1),
        pcp=int(getattr(parallel, "pcp", 1) or 1),
        tp=int(getattr(parallel, "tp", 1) or 1),
        dp=int(getattr(parallel, "dp", 1) or 1),
    )


def is_v41_replicated_cache_model(model: ModelSpec) -> bool:
    """True for latent-MQA C2 window-compressed backends requiring replicated KV caches.

    Matches upstream ``DeepseekV41CacheLayer.supports_dcp = False`` and
    ``AscendDSAV41PCPMetadataBuilder`` requiring ``DCP=1`` at backend
    construction regardless of how many leading layers are sliced.
    """
    plans = tuple(getattr(model, "layer_plans", ()) or ()) or (model,)
    if any(
        str(getattr(plan, "mla_variant", "") or "") == "latent_mqa"
        and str(getattr(plan, "cache_kind", "") or "") == "window_compressed"
        and (int(getattr(plan, "compress_ratio", 0) or 0) == 2
             or str(getattr(plan, "compressor_kind", "") or "") == "gated_ring")
        for plan in plans
    ):
        return True
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    ratios = tuple(
        int(r)
        for key in ("compress_ratios", "full_model_compress_ratios")
        for r in (facts.get(key) or ())
        if r is not None
    )
    attn_types = dict(facts.get("attention_types") or ())
    full_attn_types = dict(facts.get("full_model_attention_types") or ())
    has_latent_mqa = any(str(getattr(plan, "mla_variant", "") or "") == "latent_mqa" for plan in plans)
    return bool(
        has_latent_mqa
        and (
            2 in ratios
            or int(attn_types.get("csa2", 0) or 0) > 0
            or int(full_attn_types.get("csa2", 0) or 0) > 0
        )
    )


def layer_dcp_factors(plan: Any, parallel: ParallelSpec) -> tuple[int, int, int]:
    """Effective ``(pcp_factor, layer_tp_gather, dp_width)`` KV-sharding factors for one layer.

    MLA/DSA/CSA/CSA2/SWA/HCA layers hold 1 compressed latent KV head regardless
    of any HuggingFace ``num_kv_heads`` metadata field.  GQA layers shard across
    ``tp_gather`` only up to their intra-TP KV replication factor
    ``max(1, tp // num_kv_heads)``; hybrid layers already head-sharded on TP
    (such as sliding layers with ``num_kv_heads >= tp``) keep
    ``layer_tp_gather = 1`` rather than double-dividing KV cache or FLOPs.
    """
    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp <= 1:
        return (1, 1, 1)
    attn = str(getattr(plan, "attention", "") or "").strip().lower()
    if attn not in _CP_CONTEXT_FAMILIES:
        return (1, 1, 1)
    pcp_factor, tp_gather, dp_width = dcp_axis_factors(parallel)
    if attn != "gqa":
        return (pcp_factor, tp_gather, dp_width)
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    kv_heads = max(1, int(getattr(plan, "num_kv_heads", 1) or 1))
    kv_rep = (tp // kv_heads) if (tp >= kv_heads and tp % kv_heads == 0) else 1
    if tp_gather <= 1:
        layer_tp_gather = 1
    elif kv_rep > 1 and tp_gather <= kv_rep and kv_rep % tp_gather == 0:
        layer_tp_gather = tp_gather
    elif kv_rep == 1:
        layer_tp_gather = 1
    else:
        raise ModelingError(
            f"cp_exceeds_gqa_decode_domain:cp={cp},tp={tp},kv_heads={kv_heads}"
        )
    return (pcp_factor, layer_tp_gather, dp_width)


def layer_dcp_degree(plan: Any, parallel: ParallelSpec) -> int:
    """Effective KV-cache and decode attention DCP shard count for one layer."""
    pcp_factor, layer_tp_gather, dp_width = layer_dcp_factors(plan, parallel)
    return int(pcp_factor * layer_tp_gather * dp_width)


def kv_cache_replication_info(
    model: ModelSpec, parallel: ParallelSpec,
) -> dict[str, Any]:
    """Diagnostic KV-cache replication metadata (informational only, never a ban)."""
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    cp = max(1, int(getattr(parallel, "cp", 1) or 1))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    plans = tuple(getattr(model, "layer_plans", ()) or ()) or (model,)
    max_rep = 1
    for plan in plans:
        attn = str(getattr(plan, "attention", "") or "").strip().lower()
        if attn not in _CP_CONTEXT_FAMILIES:
            continue
        if attn == "gqa":
            kv_heads = max(1, int(getattr(plan, "num_kv_heads", 1) or 1))
            raw_tp_rep = (tp // kv_heads) if (tp >= kv_heads and tp % kv_heads == 0) else 1
        else:
            raw_tp_rep = tp
        _, layer_tp_gather, _ = layer_dcp_factors(plan, parallel)
        residual = max(1, raw_tp_rep // max(1, layer_tp_gather))
        if residual > max_rep:
            max_rep = residual
    if tp == 1 and cp == 1 and pcp == 1:
        sharding = "pure_dp_ep"
    elif pcp > 1 and cp > 1:
        sharding = "pcp_dcp"
    elif pcp > 1:
        sharding = "pcp"
    elif cp > 1:
        sharding = "dcp"
    elif max_rep > 1:
        sharding = "tp_kv_replicated"
    else:
        sharding = "tp_kv_sharded"
    return {
        "kv_cache_replicated_tp": bool(tp > 1 and max_rep > 1),
        "kv_cache_replication_factor": int(max_rep),
        "canonical_kv_sharding": sharding,
    }


def validate_cp_geometry(parallel: ParallelSpec) -> None:
    """Algebraic-only CP guards across PCP, TP, and DP axes."""
    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp <= 1:
        return
    dp = max(1, int(getattr(parallel, "dp", 1) or 1))
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    max_domain = dp * tp * pcp
    if cp > max_domain:
        raise ModelingError(
            f"cp_exceeds_parallel_domain:cp={cp}>dp*tp*pcp={max_domain}"
        )
    pcp_factor, tp_gather, dp_width = dcp_axis_factors(parallel)
    rem = cp // pcp_factor
    if (pcp % pcp_factor != 0 or cp % pcp_factor != 0
            or tp % tp_gather != 0 or rem % tp_gather != 0
            or dp % dp_width != 0 or pcp_factor * tp_gather * dp_width != cp):
        raise ModelingError(
            f"dcp_factorization_invalid:dcp={cp},pcp={pcp},tp={tp},dp={dp}"
        )


def validate_pcp_geometry(parallel: ParallelSpec) -> None:
    """PCP adds a rank axis and owns a prefill-only schedule."""
    if int(getattr(parallel, "pcp", 1) or 1) <= 1:
        return
    if int(getattr(parallel, "cp", 1) or 1) > 1:
        validate_cp_geometry(parallel)


def validate_cp_plan(model: ModelSpec, parallel: ParallelSpec) -> None:
    """Validate the declared plan against topology and model."""
    plan = cp_plan_kind(parallel)
    if plan in {CP_PLAN_PCP, CP_PLAN_PCP_DCP}:
        validate_pcp_geometry(parallel)
        verify_pcp_model(model, parallel)
        if int(getattr(parallel, "cp", 1) or 1) > 1:
            verify_cp_model(model, parallel)
    elif plan == CP_PLAN_DSA_CP:
        # DSA-CP is query-token SP over the TP group, not DCP over `cp`.
        if int(getattr(parallel, "cp", 1) or 1) != 1:
            raise ModelingError("dsa_cp_query_token_sp_excludes_dcp")
        if int(getattr(parallel, "pcp", 1) or 1) > 1:
            raise ModelingError("dsa_cp_query_token_sp_excludes_pcp")
        verify_dsa_cp_model(model, parallel)
    elif plan == CP_PLAN_DCP:
        validate_cp_geometry(parallel)
        verify_cp_model(model, parallel)
    elif plan != CP_PLAN_NONE:
        raise ModelingError(f"cp_plan_unmodeled:{plan}")


def has_kpool_indexer(model: ModelSpec) -> bool:
    """True when any layer declares a pooled-key indexer (``index_kpool > 1``)."""
    plans = tuple(getattr(model, "layer_plans", ()) or ()) or (model,)
    return any(int(getattr(lp, "index_kpool", 1) or 1) > 1 for lp in plans)


def verify_cp_model(model: ModelSpec, parallel: ParallelSpec) -> None:
    """Refuse a model/strategy CP cannot express.  Called by ``validate``."""
    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp <= 1:
        return
    if is_v41_replicated_cache_model(model):
        raise ModelingError("v41_replicated_cache_requires_dcp_1")
    kinds = _attention_families(model)
    if not kinds & _CP_CONTEXT_FAMILIES:
        raise ModelingError(f"cp_requires_context_attention:{'+'.join(sorted(kinds))}")
    plans = tuple(getattr(model, "layer_plans", ()) or ()) or (model,)
    kv_degrees = [
        layer_dcp_degree(plan, parallel)
        for plan in plans
        if str(getattr(plan, "attention", "") or "").strip().lower()
        in _CP_CONTEXT_FAMILIES
    ]
    max_layer_dcp = max(kv_degrees, default=1)
    if max_layer_dcp < cp:
        raise ModelingError(
            f"cp_exceeds_gqa_decode_domain:cp={cp},max_layer_dcp={max_layer_dcp}"
        )


def _cp_attention_family(plan: Any) -> str:
    """The CP schedule family a declared layer plan executes.

    Window/compressed layers are named by their compression schedule, a
    sparse latent layer by ``dsa``, and every other layer by its declared
    attention mechanism.  Only declared plan fields participate.
    """
    if str(getattr(plan, "cache_kind", "") or "") == "window_compressed":
        ratio = int(getattr(plan, "compress_ratio", 0) or 0)
        if ratio <= 0:
            return "swa"
        declared = str(getattr(plan, "architecture", "") or "").strip().lower()
        if declared in {"swa", "csa", "csa2", "hca"}:
            return declared
        return "hca" if ratio >= 128 else "csa"
    if str(getattr(plan, "attention", "") or "") == "mla":
        sparse = (bool(getattr(plan, "indexer", False))
                  or bool(getattr(plan, "owns_topk", False))
                  or int(getattr(plan, "topk_owner_layer_id", -1) or -1) >= 0
                  or int(getattr(plan, "index_k_owner_layer_id", -1) or -1) >= 0)
        return "dsa" if sparse else "mla"
    return str(getattr(plan, "attention", "") or "unmodeled")


def _attention_families(model: ModelSpec) -> set[str]:
    plans = getattr(model, "layer_plans", ()) or ()
    if plans:
        return {_cp_attention_family(plan) for plan in plans}
    return {_cp_attention_family(model)}


def verify_pcp_model(model: ModelSpec, parallel: ParallelSpec) -> None:
    """PCP requires a declared prefill query-shard attention.

    GQA, MLA, KDA, and GDN own that schedule. Projection-only layers may sit
    beside them. A model whose kinds miss that set, or that declares some
    other attention, has no query-shard builder.
    """
    kinds = _attention_families(model)
    if not kinds <= _CP_ATTENTION_TYPES or not (kinds & _CP_QUERY_SHARD_FAMILIES):
        raise ModelingError(
            f"pcp_requires_declared_query_shard_builder:{'+'.join(sorted(kinds))}"
        )


def verify_dsa_cp_model(model: ModelSpec, parallel: ParallelSpec) -> None:
    """DSA-CP is query-token SP: TP group, sparse indexer, declared top-k."""
    tp = int(getattr(parallel, "tp", 1) or 1)
    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp != 1:
        raise ModelingError("dsa_cp_query_token_sp_excludes_dcp")
    if tp <= 1:
        raise ModelingError("dsa_cp_query_token_sp_requires_tp")
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    if not plans:
        raise ModelingError("dsa_cp_requires_catalog_sparse_model")
    topk = max((plan.index_topk for plan in plans), default=0)
    indexers = [plan for plan in plans if bool(getattr(plan, "indexer", False))]
    if not indexers:
        raise ModelingError("dsa_cp_requires_sparse_indexer")
    if topk <= 0:
        raise ModelingError("dsa_cp_requires_declared_topk")


def merge_payload_bytes(
    q_tokens: int, local_heads: int, v_dim: int, cp: int,
) -> tuple[float, float]:
    """(full partial payload, all-gather link volume per rank).

    The partial carries ``out`` and ``LSE`` in FP32; each rank receives the
    other ranks' shards, i.e. ``(cp-1)/cp`` of the full partial payload.
    """
    cp = max(1, int(cp))
    full = float(max(1, int(q_tokens)) * max(1, int(local_heads))
                 * max(1, int(v_dim) + 1) * _FP32_BYTES)
    return full, full * (cp - 1) / cp


def merge_update_elements(
    q_tokens: int, local_heads: int, v_dim: int, cp: int,
) -> float:
    """Elements touched by the LSE merge (``npu_attention_update``)."""
    cp = max(1, int(cp))
    return float(max(1, int(q_tokens)) * max(1, int(local_heads))
                 * max(1, int(v_dim)) * cp)


def runtime_cp_flag(runtime: Any, *keys: str) -> Any:
    """Resolve a boolean/scalar CP flag from ``runtime.metadata`` or ``additional_config``."""
    if runtime is None:
        return None
    meta = dict(getattr(runtime, "metadata", ()) or ())
    extra = meta.get("additional_config")
    extra_map = dict(extra) if isinstance(extra, dict) else {}
    for source in (meta, extra_map):
        for key in keys:
            if key in source and source[key] is not None:
                return source[key]
    return None


def dcp_q_replicate_enabled(
    runtime: Any, *, pcp_factor: int, dp_width: int, sparse: bool,
) -> bool:
    """Opt-in Q-projection replication on dense MLA with single-axis TP DCP."""
    if pcp_factor != 1 or dp_width != 1 or sparse:
        return False
    return bool(runtime_cp_flag(runtime, "dcp_q_replicate", "enable_dcp_q_replicate"))


def _has_non_negative_id(obj: Any, attr: str) -> bool:
    raw = getattr(obj, attr, None)
    return raw is not None and not isinstance(raw, bool) and int(raw) >= 0


def _model_dcp_attention_class(model: ModelSpec) -> str:
    """Classify a model's KV-reading attention family for vLLM-Ascend DCP co-domain rules."""
    if is_v41_replicated_cache_model(model):
        return "replicated_cache"
    if has_kpool_indexer(model):
        return "kpool_indexer"
    plans = tuple(getattr(model, "layer_plans", ()) or ()) or (model,)
    has_sfa_dsa = any(
        str(getattr(lp, "attention", "") or "").lower() == "dsa"
        or (
            str(getattr(lp, "attention", "") or "").lower() == "mla"
            and (
                bool(getattr(lp, "indexer", False))
                or bool(getattr(lp, "owns_topk", False))
                or _has_non_negative_id(lp, "topk_owner_layer_id")
                or _has_non_negative_id(lp, "index_k_owner_layer_id")
            )
        )
        for lp in plans
    )
    if has_sfa_dsa:
        return "sfa_dsa"
    has_mla = any(
        str(getattr(lp, "attention", "") or "").lower() == "mla"
        or str(getattr(lp, "cache_kind", "") or "").lower() == "window_compressed"
        for lp in plans
    )
    if has_mla:
        return "mla"
    has_gqa = any(
        str(getattr(lp, "attention", "") or "").lower() in {"gqa", "mha", "mqa"}
        for lp in plans
    )
    return "gqa" if has_gqa else "non_kv"


def legal_vllm_ascend_dcp_values(
    model: ModelSpec, *, tp: int, pcp: int = 1,
) -> tuple[int, ...]:
    """Structural ``vllm-ascend`` legal DCP degrees reusing TP (and optional PCP) ranks.

    Enforces M10 structural legality:
    - ``dcp == 1`` is always legal (including PCP-only ``pcp > 1, dcp == 1``).
    - Replicated cache (DS-V4.1 latent-MQA C2), KPool/QSA (``index_kpool > 1``),
      and non-KV models: ``dcp == 1`` only.
    - SFA / DSA sparse MLA: ``dcp in {1, pcp, pcp * tp}`` (``{1, tp}`` at ``pcp == 1``).
    - Standard MLA: all positive divisors of ``tp * pcp``.
    - GQA: ``(1, pcp)`` when ``pcp > 1`` else ``(1,)``.
    """
    tp_n = max(1, int(tp))
    pcp_n = max(1, int(pcp))
    kind = _model_dcp_attention_class(model)
    if kind in {"replicated_cache", "kpool_indexer", "non_kv"}:
        return (1,)
    if kind == "gqa":
        return (1, pcp_n) if pcp_n > 1 else (1,)
    if kind == "sfa_dsa":
        if pcp_n > 1:
            return tuple(dict.fromkeys((1, pcp_n, pcp_n * tp_n)))
        return (1, tp_n) if tp_n > 1 else (1,)
    domain = tp_n * pcp_n
    return tuple(d for d in range(1, domain + 1) if domain % d == 0)


def validate_vllm_ascend_dcp_codomain(
    model: ModelSpec, parallel: ParallelSpec, *, runtime: Any = None,
) -> tuple[bool, str | None]:
    """Check whether ``parallel`` obeys structural DCP <-> TP co-domain rules."""
    dcp = max(1, int(getattr(parallel, "cp", 1) or 1))
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    kind = _model_dcp_attention_class(model)
    if dcp <= 1:
        return True, None
    if kind == "replicated_cache":
        return False, "vllm_ascend_replicated_cache_requires_dcp_1"
    if kind == "kpool_indexer":
        return False, "kpool_indexer_requires_dcp_1"
    if kind == "non_kv":
        return False, "vllm_ascend_non_kv_model_requires_dcp_1"
    if kind == "gqa":
        if pcp > 1 and dcp != pcp:
            return False, f"vllm_ascend_gqa_pcp_dcp_must_match:pcp={pcp},dcp={dcp}"
        if pcp == 1 and tp % dcp != 0:
            return False, f"vllm_ascend_gqa_dcp_must_divide_tp:dcp={dcp},tp={tp}"
        return True, "vllm_ascend_gqa_dcp_deprecated"
    legal = legal_vllm_ascend_dcp_values(model, tp=tp, pcp=pcp)
    if dcp not in legal:
        if kind == "sfa_dsa":
            if pcp > 1:
                return (
                    False,
                    f"vllm_ascend_sfa_pcp_dcp_must_be_pcp_or_pcp_times_tp:"
                    f"dcp={dcp},pcp={pcp},tp={tp}",
                )
            return False, f"vllm_ascend_sfa_dcp_must_equal_tp:dcp={dcp},tp={tp}"
        if pcp > 1:
            return (
                False,
                f"vllm_ascend_mla_pcp_dcp_must_divide_tp_times_pcp:"
                f"dcp={dcp},pcp={pcp},tp={tp}",
            )
        return False, f"vllm_ascend_mla_dcp_must_divide_tp:dcp={dcp},tp={tp}"
    if kind == "sfa_dsa" and runtime is not None and not dcp_indexer_sharded(model, runtime):
        interleave = cp_interleave(runtime)
        if interleave % DEFAULT_CP_INTERLEAVE != 0:
            return (
                False,
                f"vllm_ascend_sfa_replicated_dcp_interleave_not_multiple_of_128:"
                f"interleave={interleave}",
            )
    return True, None
