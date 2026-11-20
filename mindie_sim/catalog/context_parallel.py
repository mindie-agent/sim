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

from dataclasses import dataclass
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.subgraphs import KV_READING_ATTENTION_KINDS

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
        lo = max(0, rows - win) if win > 0 else 0
        visible = max(0, rows - lo)
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
        visible = min(rows, win) if win > 0 else rows
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


def split_pairs(pairs: float, cp: int, unit: str = "pairs") -> CpSplit:
    """Split a balanced pair count (sparse selected rows) across ranks."""
    pairs = max(0.0, float(pairs))
    cp = max(1, int(cp))
    share = pairs / cp
    per = tuple(share for _ in range(cp))
    return CpSplit(cp, pairs, per, share, unit)


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
    """Equal-as-possible contiguous bounds for ``2*pcp`` DualChunkSwap chunks.

    The upstream chunker pads every chunk to ``ceil(q/2pcp)``; the simulator
    prices real tokens only, so it redistributes the remainder at the tail.
    Sums stay exact (no invented tokens) and the chunk sizes differ by <= 1.
    """
    q_len = max(0, int(q_len))
    pcp = max(1, int(pcp))
    chunks = 2 * pcp
    return tuple((index * q_len) // chunks for index in range(chunks + 1))


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


def pcp_query_tokens(q_len: int, pcp: int) -> tuple[int, ...]:
    """Tokens owned by each rank (max rank is the declared local batch)."""
    return tuple(
        sum(end - start for start, end in pcp_rank_chunk_ranges(q_len, pcp, rank))
        for rank in range(max(1, int(pcp)))
    )


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


def pcp_worst_rank(split: PcpSplit) -> int:
    """Rank with the largest causal work (lowest index on a tie)."""
    if not split.per_rank:
        return 0
    return max(range(len(split.per_rank)), key=lambda rank: split.per_rank[rank])


def pcp_kv_gather_bytes(
    prefill_tokens: int, kv_row_bytes: float, pcp: int,
) -> tuple[float, float]:
    """(gathered full K/V payload, all-gather link volume per rank).

    PCP gathers the *current prefill chunk's* K/V across the PCP group; the
    older context is already replicated on every rank.  The full payload is the
    gathered plane; each rank receives the other ranks' ``(pcp-1)/pcp``.
    """
    pcp = max(1, int(pcp))
    full = float(max(0, int(prefill_tokens)) * float(kv_row_bytes))
    return full, full * (pcp - 1) / pcp


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


def split_index_rows(rows: int, cp: int) -> CpSplit:
    """Max-rank shard of a token/index row count (block-interleaved owner)."""
    return split_rows(rows, cp, unit="rows")


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


def pcp_requested(parallel: ParallelSpec) -> bool:
    return int(getattr(parallel, "pcp", 1) or 1) > 1


def cp_plan_kind(parallel: ParallelSpec) -> str:
    """The declared plan, resolved defensively for objects built before this field."""
    plan = str(getattr(parallel, "cp_plan", CP_PLAN_NONE) or CP_PLAN_NONE)
    if plan in CP_PLAN_KINDS or plan == CP_PLAN_NONE:
        return plan
    return CP_PLAN_DCP if cp_requested(parallel) else CP_PLAN_NONE


def validate_cp_geometry(parallel: ParallelSpec) -> None:
    """Structural CP guards across TP (when PCP=1) or TPxPCP (when PCP>1)."""
    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp <= 1:
        return
    tp = int(parallel.tp)
    pcp = int(getattr(parallel, "pcp", 1) or 1)
    if pcp <= 1:
        if tp <= 1:
            raise ModelingError("cp_requires_tp_group")
        if cp > tp:
            raise ModelingError(f"cp_exceeds_tp:{cp}>{tp}")
        if tp % cp != 0:
            raise ModelingError(f"tp_not_divisible_by_cp:{tp}%{cp}")
        return
    if cp not in {pcp, tp * pcp}:
        raise ModelingError(f"dcp_pcp_domain_invalid:dcp={cp},pcp={pcp},tp={tp}")


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


def verify_cp_model(model: ModelSpec, parallel: ParallelSpec) -> None:
    """Refuse a model/strategy CP cannot express.  Called by ``validate``."""
    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp <= 1:
        return
    pcp = int(getattr(parallel, "pcp", 1) or 1)
    kinds = _attention_families(model)
    if not kinds & _CP_CONTEXT_FAMILIES:
        raise ModelingError(f"cp_requires_context_attention:{'+'.join(sorted(kinds))}")
    for plan in getattr(model, "layer_plans", ()) or ():
        if plan.attention == "gqa":
            _verify_gqa_kv_domain(int(plan.num_kv_heads), int(parallel.tp), cp, pcp=pcp)
    if not getattr(model, "layer_plans", ()) and model.attention == "gqa":
        _verify_gqa_kv_domain(int(model.num_kv_heads), int(parallel.tp), cp, pcp=pcp)


def _attention_kinds(model: ModelSpec) -> set[str]:
    plans = getattr(model, "layer_plans", ()) or ()
    if plans:
        return {str(plan.attention) for plan in plans}
    return {str(model.attention)}


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


def _verify_gqa_kv_domain(
    num_kv_heads: int, tp: int, cp: int, *, pcp: int = 1,
) -> None:
    if pcp > 1 and cp > 1 and cp != pcp:
        raise ModelingError(f"gqa_pcp_dcp_requires_equal_degrees:pcp={pcp},dcp={cp}")
    kv_heads = max(1, num_kv_heads)
    if pcp <= 1:
        domain = tp // kv_heads
    elif tp >= kv_heads:
        domain = (tp // kv_heads) * pcp
    else:
        domain = pcp if kv_heads % max(1, tp) == 0 else 0
    if domain < cp or cp <= 0 or domain % cp != 0:
        raise ModelingError(f"cp_exceeds_gqa_decode_domain:cp={cp},tp//kv_heads={domain}")


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


def cp_layer_scope(
    model: ModelSpec, plan: str = CP_PLAN_DCP,
) -> dict[str, Any]:
    """Which layers CP shards and which stay replicated, declared explicitly."""
    kinds = _attention_families(model)
    if plan == CP_PLAN_DSA_CP:
        return {
            "context_addressed": [],
            "replicated": sorted(kinds),
            "strategy": strategy_for_plan(plan),
            "phase_scope": "prefill",
            "query_token_sharded": True,
            "kv_cache": "replicated_full",
            "indexer_cache": "replicated_full",
            "indexer_policy": DSA_CP_INDEXER_POLICY,
            "indexer_cache_sharded": False,
            "local_indexer": "local_query_topk_over_full_kv_prefix",
            "output_merge": "token_all_gather",
        }
    scope = {
        "context_addressed": sorted(kinds & _CP_CONTEXT_FAMILIES),
        "replicated": sorted(kinds - _CP_CONTEXT_FAMILIES),
        "strategy": strategy_for_plan(plan),
        "indexer_policy": (PCP_INDEXER_POLICY if plan == CP_PLAN_PCP
                           else CP_INDEXER_POLICY),
        "indexer_cache_sharded": False,
    }
    from .paired_sfa import execution
    if plan in {CP_PLAN_DCP, CP_PLAN_PCP_DCP} and execution(model).get("dcp_indexer_sharded"):
        scope.update(indexer_cache_sharded=True,
                     indexer_policy="token_sharded_candidate_topk",
                     output_merge="head_alltoall_lse")
    if plan == CP_PLAN_PCP:
        scope.update({
            "prefill_cp_plan": "pcp",
            "decode_cp_plan": "none",
            "phase_scope": "prefill",
            "pcp_kv_gather": "full_index_k_and_active_kv_rows",
            "indexer_policy": PCP_INDEXER_POLICY,
        })
    elif plan == CP_PLAN_PCP_DCP:
        scope.update({
            "prefill_cp_plan": "pcp",
            "decode_cp_plan": "dcp",
            "phase_scope": "prefill+decode",
            "pcp_kv_gather": "full_index_k_and_active_kv_rows",
            "decode_merge": "npu_attention_update",
        })
    return scope


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


def merge_metadata(
    cp: int, q_tokens: int, local_heads: int, v_dim: int,
) -> tuple[tuple[str, Any], ...]:
    full, link = merge_payload_bytes(q_tokens, local_heads, v_dim, cp)
    return (
        ("cp", cp),
        ("cp_strategy", CP_STRATEGY),
        ("partial_reduction", "logsumexp"),
        ("partial_payload_bytes", full),
        ("merge_link_bytes", link),
        ("merge_update", "npu_attention_update"),
        ("merge_dtype", "fp32"),
    )


def split_metadata(split: CpSplit, *, local_rows: int = 0) -> tuple[tuple[str, Any], ...]:
    extra: tuple[tuple[str, Any], ...] = ()
    if local_rows:
        extra = (("cp_local_rows", int(local_rows)),)
    return split.metadata() + extra
