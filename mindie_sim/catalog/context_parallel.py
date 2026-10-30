"""Context ownership and collective geometry.

[Role] Partition DCP key positions and PCP zigzag query ranges.
[Invariants] DCP reuses TP ranks; PCP adds a query-sharding axis. Pair counts
  conserve the causal/windowed triangle and block ownership is explicit.
[Timeouts & Locks] Bounded arithmetic, no device or observation access.
[Side-Effect Policy] No external mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.model import ModelSpec

#: DCP strategy (kept as the historical string because plan consumers already
#: report it).  It shards KV and merges (out, LSE); DSA indexers are replicated.
CP_STRATEGY = "kv_context_shard_lse_merge"
DCP_STRATEGY = CP_STRATEGY
#: PCP strategy: local query chunks, full K/V gathered, final outputs (no merge).
PCP_STRATEGY = "pcp_query_split_kv_gather"
#: DSA-CP strategy: sharded local indexer TopK + exact global merge + LSE merge.
DSA_CP_STRATEGY = "dsa_cp_sharded_topk_lse_merge"
CP_PLAN_DCP = "dcp"
CP_PLAN_PCP = "pcp"
CP_PLAN_DSA_CP = "dsa_cp"
CP_PLAN_NONE = "none"
CP_PLAN_KINDS = (CP_PLAN_DCP, CP_PLAN_PCP, CP_PLAN_DSA_CP)
CP_INDEXER_POLICY = "replicated"
DSA_CP_INDEXER_POLICY = "sharded_local_topk"
CP_UNMODELED_SCHEDULES = ("compact_kv_gather", "query_token_shard")
DEFAULT_CP_INTERLEAVE = 128
_FP32_BYTES = 4.0
#: Bytes per gathered indexer candidate: FP32 score + int32 row id.  The merge
#: exchanges only selected candidates, never the full index-K plane.
_INDEX_CANDIDATE_BYTES = 8.0


def strategy_for_plan(plan: str) -> str:
    """The declared execution strategy for one plan name."""
    return {
        CP_PLAN_DCP: DCP_STRATEGY,
        CP_PLAN_PCP: PCP_STRATEGY,
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
    *, window: int = 0, noncausal: bool = False, padding: bool = False,
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
        per = tuple(float(q_len * owned_positions(0, rows - 1, cp, interleave, r))
                    for r in range(cp))
        total = float(q_len * rows)
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
        return float(q_len * (context + q_len))
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
# DSA-CP: sharded local indexer TopK + exact global candidate merge
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DsaCpTopKPlan:
    """Declared candidate geometry for one query token's sparse selection."""

    cp: int
    scored_rows: int
    local_scored_rows: int
    topk: int
    global_selected_rows: int
    local_candidates: int
    merged_candidates: int
    merge_elements: float
    candidate_payload_bytes: float
    merge_link_bytes: float
    #: The union of per-rank local top-k provably contains the global top-k:
    #: a global top-k can take at most ``k`` rows from any single shard.
    exact: bool = True

    def metadata(self, prefix: str = "dsa_cp") -> tuple[tuple[str, Any], ...]:
        return (
            (f"{prefix}_topk_exact", self.exact),
            (f"{prefix}_scored_rows", int(self.scored_rows)),
            (f"{prefix}_local_scored_rows", int(self.local_scored_rows)),
            (f"{prefix}_topk", int(self.topk)),
            (f"{prefix}_selected_rows_global", int(self.global_selected_rows)),
            (f"{prefix}_local_candidates", int(self.local_candidates)),
            (f"{prefix}_merged_candidates", int(self.merged_candidates)),
            (f"{prefix}_candidate_payload_bytes", float(self.candidate_payload_bytes)),
            (f"{prefix}_candidate_link_bytes", float(self.merge_link_bytes)),
            (f"{prefix}_selection", "local_topk_then_global_merge"),
        )


def plan_dsa_cp_topk(
    scored_rows: int, topk: int, cp: int, *, tokens: int = 1,
) -> DsaCpTopKPlan:
    """Plan the sharded indexer TopK for ``tokens`` query rows.

    Each rank scores only its KV shard and keeps ``min(local_rows, topk)``
    candidates.  The union is all-gathered and merged to the global top-k.  This
    is exact: a global top-k has at most ``k`` rows from any one shard, so no
    global winner can be missing from a shard's local top-k.
    """
    cp = max(1, int(cp))
    scored = max(0, int(scored_rows))
    topk = max(0, int(topk))
    local_scored = int(math.ceil(scored / cp)) if cp > 1 else scored
    global_selected = min(scored, topk) if topk else scored
    local_k = min(local_scored, topk) if topk else local_scored
    merged = cp * local_k
    token_rows = max(1, int(tokens))
    full = float(merged * token_rows * _INDEX_CANDIDATE_BYTES)
    link = full * (cp - 1) / cp if cp > 1 else 0.0
    return DsaCpTopKPlan(
        cp=cp, scored_rows=scored, local_scored_rows=local_scored, topk=topk,
        global_selected_rows=global_selected, local_candidates=local_k,
        merged_candidates=merged,
        merge_elements=float(merged * token_rows),
        candidate_payload_bytes=full, merge_link_bytes=link,
        exact=merged >= global_selected,
    )


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
    """Structural CP guards: DCP subdivides the TP group, it never exceeds it."""
    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp <= 1:
        return
    tp = int(parallel.tp)
    if tp <= 1:
        raise ModelingError("cp_requires_tp_group")
    if cp > tp:
        raise ModelingError(f"cp_exceeds_tp:{cp}>{tp}")
    if tp % cp != 0:
        raise ModelingError(f"tp_not_divisible_by_cp:{tp}%{cp}")


def validate_pcp_geometry(parallel: ParallelSpec) -> None:
    """PCP adds a rank axis and owns a prefill-only schedule.

    DCP reuses TP, PCP multiplies the world, so having both at once would need a
    2-D KV layout this model does not declare.  Refuse it instead of quietly
    dropping one axis.  The same applies to PP (stage chain) and DP (replicas):
    PP and DP compose through placement; recurrent PCP needs a separate state-transfer schedule.
    """
    pcp = int(getattr(parallel, "pcp", 1) or 1)
    if pcp <= 1:
        return
    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp > 1:
        raise ModelingError(f"cp_pcp_combination_unmodeled:cp={cp},pcp={pcp}")
    if int(parallel.ep) > 1:
        # EP's group would span the PCP axis in the deployed layout; this
        # simulator's EP domain is dp*tp only, so the combination is refused
        # instead of priced with the wrong group.
        raise ModelingError(f"pcp_with_ep_unmodeled:ep={parallel.ep},pcp={pcp}")


def validate_cp_plan(model: ModelSpec, parallel: ParallelSpec) -> None:
    """Validate the declared plan against topology and model."""
    plan = cp_plan_kind(parallel)
    if plan == CP_PLAN_PCP:
        validate_pcp_geometry(parallel)
        verify_pcp_model(model, parallel)
    elif plan == CP_PLAN_DSA_CP:
        if int(parallel.pcp) > 1:
            raise ModelingError("dsa_cp_with_pcp_unmodeled")
        validate_cp_geometry(parallel)
        verify_cp_model(model, parallel)
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
    kinds = _attention_kinds(model)
    if not kinds & {"gqa", "mla"}:
        raise ModelingError(f"cp_requires_context_attention:{'+'.join(sorted(kinds))}")
    for plan in getattr(model, "layer_plans", ()) or ():
        if plan.attention == "gqa":
            _verify_gqa_kv_domain(int(plan.num_kv_heads), int(parallel.tp), cp)
    if not getattr(model, "layer_plans", ()) and model.attention == "gqa":
        _verify_gqa_kv_domain(int(model.num_kv_heads), int(parallel.tp), cp)


def _attention_kinds(model: ModelSpec) -> set[str]:
    plans = getattr(model, "layer_plans", ()) or ()
    if plans:
        return {str(plan.attention) for plan in plans}
    return {str(model.attention)}


def _verify_gqa_kv_domain(num_kv_heads: int, tp: int, cp: int) -> None:
    # DCP for GQA requires (tp // num_kv_heads) >= cp and an integral divisor;
    # a rank below the KV-head count replicates heads and cannot own a context
    # shard (vLLM: ``tp // num_kv_heads % dcp == 0``).
    domain = tp // max(1, num_kv_heads)
    if domain < cp or cp <= 0 or domain % cp != 0:
        raise ModelingError(
            f"cp_exceeds_gqa_decode_domain:cp={cp},tp//kv_heads={domain}"
        )


def verify_pcp_model(model: ModelSpec, parallel: ParallelSpec) -> None:
    """PCP requires a prefill-capable context attention and no recurrent state.

    Splitting a prefill sequence across ranks is safe for context-addressed
    attention because each query's visible prefix is gathered.  It is not safe
    for a KDA/GDN scan: the recurrent state depends on the exact token order,
    so a rank holding a middle chunk cannot reconstruct the state without
    another schedule.  Catalog layer-plan models are refused explicitly until
    that lowering is declared.
    """
    kinds = _attention_kinds(model)
    recurrent = sorted(kinds & {"kda", "gdn"})
    if recurrent:
        raise ModelingError(f"pcp_recurrent_state_unmodeled:{'+'.join(recurrent)}")
    if kinds != {"gqa"}:
        raise ModelingError(
            f"pcp_requires_declared_query_shard_builder:{'+'.join(sorted(kinds))}"
        )


def verify_dsa_cp_model(model: ModelSpec, parallel: ParallelSpec) -> None:
    """DSA-CP is only defined for a sparse indexer with a declared global top-k."""
    cp = int(getattr(parallel, "cp", 1) or 1)
    if cp <= 1:
        raise ModelingError("dsa_cp_requires_cp")
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    if not plans:
        raise ModelingError("dsa_cp_requires_catalog_sparse_model")
    facts = dict(getattr(model, "catalog_facts", ()) or ())
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
    kinds = _attention_kinds(model)
    dsa_cp = plan == CP_PLAN_DSA_CP
    return {
        "context_addressed": sorted(kinds & {"gqa", "mla"}),
        "replicated": sorted(kinds - {"gqa", "mla"}),
        "strategy": strategy_for_plan(plan),
        "indexer_policy": (
            DSA_CP_INDEXER_POLICY if dsa_cp else CP_INDEXER_POLICY
        ),
        "indexer_cache_sharded": bool(dsa_cp),
    }


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
