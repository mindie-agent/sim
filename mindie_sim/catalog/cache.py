"""Compressed-cache, window, candidate and state geometry for catalog models.

[Role]
  Compute exact completed compressed rows, local-window rows, sparse candidate
  validity, TopK selection reads, C2 parity across chunks, owner-shared long
  KV/index storage and per-rank per-request cache/state bytes.

[Invariants]
  * Completed rows are ``floor(committed_tokens / ratio)``; an odd residual
    remains in state across chunk boundaries instead of being rounded up.
  * Candidate block IDs expand to real valid rows; the final partial block is
    kept and block 0 is not silently treated as an extra candidate block.
  * Unique producer storage is counted once; every consumer still pays its own
    logical read.  Sharing is not a network collective by itself.
  * A DSA-CP index-K cache is owned in whole interleave blocks: a rank holds
    every block ``b`` with ``b % cp == rank``, so its row count is
    ``ceil(ceil(rows / block) / cp) * block``, never the uniform
    ``ceil(rows / cp)``.  The uniform reading is kept as
    ``declared_uniform_ceil`` for comparison and is never the charge.
  * SWA windows saturate at the configured window while still receiving new
    rows.  A fixed C2 ring keeps capacity but updates contents.
  * KDA/GDN recurrent matrices are heads*Dk*Dv FP32 endpoints; convolution
    history is a separate bounded BF16 ring.  Neither grows per past token.
  * KV-head replication uses at least one local head.

[Timeouts & Locks]
  Pure integer/float arithmetic bounded by layer count.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from mindie_sim.catalog.context_parallel import cp_plan_kind
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.model import WorkloadSpec
from mindie_sim.catalog.cache_format import cache_format


def completed_rows(sequence_tokens: int, ratio: int) -> int:
    if ratio <= 0:
        return 0
    return max(0, int(sequence_tokens) // int(ratio))


def new_completed_rows(context: int, query: int, ratio: int) -> int:
    if ratio <= 0:
        return 0
    return completed_rows(context + query, ratio) - completed_rows(context, ratio)


def window_rows(sequence_tokens: int, window: int) -> int:
    if window <= 0:
        return 0
    return max(0, min(int(sequence_tokens), int(window)))


def candidate_valid_rows(
    source_rows: int, *, blocks: int = 2048, block_size: int = 8,
    first_block: int = 0,
) -> int:
    """Rows represented by selected candidate blocks (partial final block kept).

    ``first_block=0`` is the conservative production set (block 0 is not
    presumed excluded).  ``first_block=1, blocks=2048`` reproduces the
    conditional 16377-row oracle at 16385 source rows; it is not a universal
    candidate assumption.
    """
    source_rows = max(0, int(source_rows))
    blocks = max(0, int(blocks))
    block_size = max(1, int(block_size))
    first = max(0, int(first_block))
    start = first * block_size
    end = min(source_rows, (first + blocks) * block_size)
    return max(0, end - start)


def block_round(rows: int, block_size: int) -> int:
    if block_size <= 1:
        return int(rows)
    return int(math.ceil(int(rows) / int(block_size)) * int(block_size))


#: V4/V4.1 lightning-indexer CP interleave.  The arch32 kernel interleaves the
#: compressed index-K cache in 128-token groups and one group is a whole
#: ownership block for one CP rank; with the primary compress ratio 4 a group
#: is 32 compressed rows (A's kernel reading: INTERLEAVE=128, CMP=4).
INDEX_INTERLEAVE_TOKENS = 128


def index_k_block_rows(compress_ratio: int) -> int:
    """Compressed index-K rows in one CP interleave (ownership) block."""
    ratio = int(compress_ratio)
    if ratio <= 0:
        raise ModelingError("catalog_index_k_ratio_invalid")
    return max(1, int(INDEX_INTERLEAVE_TOKENS) // ratio)


def index_k_ownership_rows(
    rows: int, cp: int, compress_ratio: int,
) -> tuple[int, ...]:
    """Per-rank owned index-K rows under whole-block interleave.

    Rank ``r`` owns blocks ``b`` with ``b % cp == r``; the final block may be
    partial, so ranks can hold ``0`` or a whole block more than the uniform
    slice.  The tuple sums to ``rows`` (conservation), which is what makes it
    an ownership partition rather than a second total.
    """
    total = max(0, int(rows))
    cp = max(1, int(cp))
    block = index_k_block_rows(compress_ratio)
    if cp == 1:
        return (total,)
    blocks = (-(-total // block)) if total else 0
    owned = [0] * cp
    for index in range(blocks):
        first = index * block
        size = min(block, total - first)
        if size > 0:
            owned[index % cp] += int(size)
    return tuple(owned)


def index_k_owned_block_rows(
    rows: int, cp: int, compress_ratio: int,
) -> tuple[int, ...]:
    """Per-rank index-K rows *allocated* under whole-block ownership.

    The engine allocates a whole interleave block to the rank that owns it, so
    a rank holding one valid row in the final partial block still owns the
    block.  This is the same convention ``_context_shard_blocks`` uses for the
    block-interleaved KV, and it is what the capacity charge must use.
    """
    total = max(0, int(rows))
    cp = max(1, int(cp))
    block = index_k_block_rows(compress_ratio)
    if cp == 1:
        # No CP shard means no block ownership split; the single rank keeps
        # the plan's existing (unrounded) index-K row count.
        return (total,)
    blocks = (-(-total // block)) if total else 0
    owned = [0] * cp
    for index in range(blocks):
        owned[index % cp] += block
    return tuple(owned)


def index_k_capacity_rows(rows: int, cp: int, compress_ratio: int) -> int:
    """Per-rank index-K capacity = max over ranks of allocated block rows.

    Closed form ``ceil(ceil(rows / block) / cp) * block``: the rank that is
    handed the extra whole block.  Every rank's capacity row is charged this
    bound because the per-token KV price is shared by all ranks.
    """
    total = max(0, int(rows))
    cp = max(1, int(cp))
    if cp == 1:
        return total
    block = index_k_block_rows(compress_ratio)
    blocks = (-(-total // block)) if total else 0
    return int(-(-blocks // cp) * block)


def index_k_store_rows(
    new_rows: int, cp: int, compress_ratio: int,
) -> int:
    """Per-rank index-K *valid rows* a single iteration may have to store.

    The store side uses the same whole-block ownership as the capacity but
    charges valid rows, not the allocated block: the rank that owns the final
    partial block writes only its valid rows.  Bound = max over ranks of
    ``index_k_ownership_rows``, which is at most one block more than the
    uniform ``ceil(new_rows / cp)``.
    """
    owned = index_k_ownership_rows(new_rows, cp, compress_ratio)
    return int(max(owned) if owned else 0)


def local_kv_heads(num_kv_heads: int, tp: int) -> int:
    return max(1, int(math.ceil(int(num_kv_heads) / max(1, int(tp)))))


def _quant(model: Any) -> Any:
    return getattr(model, "quant", None)


def kv_element_bytes(model: Any) -> float:
    """Declared KV row width in bytes; the quant plan owns the swap axis."""
    quant = _quant(model)
    dtype = getattr(quant, "kv_dtype", "") if quant is not None else ""
    return dtype_bytes(str(dtype or getattr(model, "kv_dtype", "bf16")))


def kv_bytes_for(model: Any, elements: float) -> float:
    return float(elements) * kv_element_bytes(model)


def index_element_bytes(model: Any) -> float:
    quant = _quant(model)
    dtype = getattr(quant, "index_dtype", "") if quant is not None else ""
    return dtype_bytes(str(dtype or "bf16"))


def index_scale_bytes(model: Any) -> float:
    quant = _quant(model)
    scale = getattr(quant, "index_scale_dtype", "") if quant is not None else ""
    if not scale or str(scale) == "none":
        return 0.0
    return dtype_bytes(str(scale))


def index_row_bytes(model: Any, index_dim: int) -> float:
    """Index data plus one declared scale per row (the tree's cache layout)."""
    return float(index_dim) * index_element_bytes(model) + index_scale_bytes(model)


def _facts(model: Any) -> dict[str, Any]:
    return dict(getattr(model, "catalog_facts", ()) or ())


@dataclass(frozen=True)
class IndexReadCost:
    scored_rows: int
    selected_rows: int
    candidate_rows: int
    index_row_bytes: float
    index_read_bytes: float
    cache_write_rows: int
    cache_write_bytes: float


def index_selection_cost(model, plan, sequence_tokens):
    if not (plan.indexer or plan.owns_topk or plan.index_query_per_layer):
        return IndexReadCost(0, 0, 0, 0., 0., 0, 0.)
    owner = plan.index_k_owner_layer_id
    producer = model.layer_plans[owner] if owner >= 0 else plan
    rows = completed_rows(sequence_tokens, max(1, producer.compress_ratio, producer.index_kpool))
    candidates = 0
    if plan.candidate_filter:
        source = model.layer_plans[plan.candidate_source_layer_id]
        candidates = candidate_valid_rows(
            completed_rows(sequence_tokens, max(1, source.compress_ratio)),
            blocks=plan.candidate_topk_blocks, block_size=plan.candidate_block_size)
    scored = min(rows, candidates) if plan.candidate_filter else rows
    limit = plan.index_topk // plan.index_kpool
    selected = min(scored, limit) if plan.index_topk else scored
    width = index_row_bytes(model, plan.index_dim)
    writes = rows if plan.owns_index_k else 0
    return IndexReadCost(scored, selected, candidates, width, scored * width, writes, writes * width)


def attention_rows(model, plan, sequence_tokens):
    seq = max(0, int(sequence_tokens))
    if plan.attention in {"kda", "gdn"} or plan.cache_kind == "recurrent_unmodeled":
        return 0, 0, 0
    if plan.cache_kind == "dense":
        return min(seq, plan.window) if plan.window else seq, 0, 0
    window = window_rows(seq, plan.window)
    rows = completed_rows(seq, plan.compress_ratio)
    if plan.index_topk and (plan.indexer or plan.topk_owner_layer_id >= 0):
        rows = min(rows, plan.index_topk)
    return 0, rows, window


def _local_state_sharded(plan: Any, tp: int) -> tuple[float, float]:
    tp = max(1, int(tp))
    recurrent = float(plan.recurrent_state_elements) / tp
    conv = float(plan.conv_state_elements) / tp
    return recurrent, conv


def _context_shard_rows(rows: int, cp: int) -> int:
    """Block-interleaved context rows owned by one rank (ceil of blocks)."""
    if cp <= 1:
        return int(rows)
    return int(math.ceil(int(rows) / int(cp)))


def _context_shard_blocks(seq: int, block: int, cp: int) -> int:
    if cp <= 1:
        return block_round(seq, block)
    blocks = int(math.ceil(max(0, int(seq)) / max(1, int(block))))
    return int(math.ceil(blocks / int(cp)) * int(block))


def layer_cache_components(model, plan, parallel, sequence_tokens, *, capacity_padded=True,
                           interleave=128):
    seq = max(0, int(sequence_tokens))
    tp, cp = parallel.tp, parallel.cp
    result = {}
    def local_rows(rows, block=interleave):
        return (_context_shard_blocks(rows, block, cp) if capacity_padded
                else rows / cp)
    if plan.attention in {"kda", "gdn"}:
        result["recurrent_state"] = plan.recurrent_state_elements / tp * dtype_bytes(plan.recurrent_state_dtype)
        result["conv_state"] = plan.conv_state_elements / tp * dtype_bytes(plan.conv_state_dtype)
        return result
    if plan.attention == "unmodeled" or plan.cache_kind == "none":
        return result
    if plan.attention in {"gqa", "projection_only"}:
        if 0 <= plan.kv_owner_layer_id < plan.index:
            return result
        rows = min(seq, plan.window) if plan.window else seq
        result["gqa_kv"] = kv_bytes_for(model, local_rows(rows) * 2 * local_kv_heads(plan.num_kv_heads, tp) * plan.head_dim)
        return result
    if plan.attention != "mla":
        raise ModelingError(f"cache_attention_unknown:{plan.attention}")
    if plan.cache_kind == "dense":
        rows = min(seq, plan.window) if plan.window else seq
        result["mla_kv"] = local_rows(rows) * cache_format(model, "mla_kv").row_bytes(plan.cache_row_elements)
    else:
        if plan.window:
            result["window"] = local_rows(min(seq, plan.window)) * cache_format(
                model, "window").row_bytes(plan.cache_row_elements)
        if plan.owns_long_kv:
            result["compressed_kv"] = local_rows(
                completed_rows(seq, plan.compress_ratio), max(1, interleave // max(1, plan.compress_ratio))) \
                * cache_format(model, "compressed_kv").row_bytes(plan.cache_row_elements)
            if plan.compressor_state_rows and seq:
                rows = plan.compressor_state_rows if capacity_padded else min(seq, plan.compressor_state_rows)
                result["compressor_state"] = rows * 2 * plan.compressor_width * plan.head_dim * 4
    if plan.owns_index_k:
        index_ratio = max(1, plan.compress_ratio, plan.index_kpool)
        rows = completed_rows(seq, index_ratio)
        if parallel.cp_plan == "dsa_cp":
            rows = index_k_capacity_rows(rows, cp, index_ratio) if capacity_padded else rows / cp
        result["index_k"] = rows * index_row_bytes(model, plan.index_dim)
        if plan.index_kpool > 1 and seq:
            # Raw normalized keys and gates remain FP32 in a bounded ring;
            # the pooled BF16 plane is independent of uncompressed token KV.
            capacity = dict(model.metadata).get("native_keypool_caller", {}).get(
                "tail_capacity", plan.index_kpool)
            rows = capacity if capacity_padded else min(seq, capacity)
            result["index_pool_tail"] = rows * 2 * plan.index_dim * 4
        if plan.index_input == "hidden" and plan.compress_ratio and seq:
            result["index_compressor_state"] = min(seq, plan.compressor_state_rows) * 2 * plan.compressor_width * plan.index_dim * 4
    return result


def stage_cache_components(
    model: Any, parallel: ParallelSpec, sequence_tokens: int,
    *, capacity_padded: bool = True,
) -> list[dict[str, float]]:
    pp = max(1, parallel.pp)
    stages = [dict() for _ in range(pp)]
    per = max(1, model.layers // pp)
    for plan in model.layer_plans:
        stage = min(pp - 1, plan.index // per) if pp > 1 else 0
        for key, value in layer_cache_components(
            model, plan, parallel, sequence_tokens,
            capacity_padded=capacity_padded,
        ).items():
            stages[stage][key] = stages[stage].get(key, 0.0) + value
    return stages


def request_cache_breakdown(
    model: Any, parallel: ParallelSpec, sequence_tokens: int,
) -> dict[str, Any]:
    stages = stage_cache_components(model, parallel, sequence_tokens)
    totals: dict[str, float] = {}
    for stage in stages:
        for key, value in stage.items():
            totals[key] = totals.get(key, 0.0) + value
    stage_bytes = [float(sum(stage.values())) for stage in stages]
    return {
        "sequence_tokens": int(sequence_tokens),
        "by_component": totals,
        "stage_bytes": stage_bytes,
        "total_bytes": float(sum(stage_bytes)),
        "max_stage_bytes": float(max(stage_bytes) if stage_bytes else 0.0),
        "largest_stage": int(max(range(len(stage_bytes)), key=lambda i: stage_bytes[i])) if stage_bytes else 0,
    }


def state_bytes_per_request_per_rank(model: Any, parallel: ParallelSpec) -> float:
    total = 0.0
    for plan in model.layer_plans:
        if plan.attention in {"kda", "gdn"}:
            recurrent, conv = _local_state_sharded(plan, parallel.tp)
            total += recurrent * 4.0 + conv * 2.0
    return float(total)


def cache_bytes_per_token_estimate(
    model: Any, parallel: ParallelSpec, sequence_tokens: int,
) -> tuple[float, float]:
    """(max-stage cache bytes per token, total cache bytes at sequence)."""
    seq = max(1, int(sequence_tokens))
    stages = stage_cache_components(model, parallel, seq)
    stage_bytes = [float(sum(stage.values())) for stage in stages]
    total = float(sum(stage_bytes))
    return (max(stage_bytes) / seq if stage_bytes else 0.0), total


def peak_live_tokens(workload: WorkloadSpec) -> int:
    """Largest single-request committed token span, never lifetime sum."""
    if not workload.requests:
        return 0
    return max(int(request.prompt_tokens) + int(request.output_tokens) for request in workload.requests)


def live_request_cache_peak(
    model: Any, parallel: ParallelSpec, workload: WorkloadSpec,
    *, block_size: int = 128, media_tokens: int = 0,
) -> dict[str, Any]:
    per_request = []
    for index, request in enumerate(workload.requests):
        seq = int(request.prompt_tokens) + int(request.output_tokens) + (
            int(media_tokens) if index == 0 else 0
        )
        breakdown = request_cache_breakdown(model, parallel, seq)
        per_request.append({
            "request_id": request.request_id,
            "sequence_tokens": seq,
            "cache_bytes": breakdown["total_bytes"],
            "max_stage_bytes": breakdown["max_stage_bytes"],
            "block_rounded_tokens": block_round(seq, block_size),
        })
    return {
        "per_request": per_request,
        "peak_bytes": max((item["cache_bytes"] for item in per_request), default=0.0),
        "peak_max_stage_bytes": max((item["max_stage_bytes"] for item in per_request), default=0.0),
        "basis": "single live request peak; not a lifetime sum over sequential requests",
    }


__all__ = [
    "IndexReadCost", "completed_rows", "new_completed_rows", "window_rows",
    "candidate_valid_rows", "block_round", "index_element_bytes",
    "index_row_bytes", "index_scale_bytes", "kv_bytes_for", "kv_element_bytes",
    "local_kv_heads",
    "index_selection_cost", "attention_rows", "layer_cache_components",
    "stage_cache_components", "request_cache_breakdown",
    "state_bytes_per_request_per_rank", "cache_bytes_per_token_estimate",
    "peak_live_tokens", "live_request_cache_peak",
]
