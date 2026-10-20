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

from .a3_types import A3InputError, ParallelSpec, WorkloadSpec, dtype_bytes


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


def local_kv_heads(num_kv_heads: int, tp: int) -> int:
    return max(1, int(math.ceil(int(num_kv_heads) / max(1, int(tp)))))


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


def index_selection_cost(
    model: Any, plan: Any, sequence_tokens: int,
) -> IndexReadCost:
    facts = _facts(model)
    index_dim = int(facts.get("index_head_dim", 0) or 128)
    topk = int(facts.get("index_topk", 0) or 0)
    native = str(facts.get("profile", ""))
    candidate_source = int(plan.candidate_source_layer_id)
    candidate_blocks = int(facts.get("candidate_topk_blocks", 0) or 2048)
    candidate_block = int(facts.get("candidate_block_size", 0) or 8)
    if plan.attention != "mla":
        return IndexReadCost(0, 0, 0, 0.0, 0.0, 0, 0.0)
    if native == "glm":
        # Full producers own a token-level BF16 index plane; shared layers
        # conservatively execute their own query/selection against it.
        executes = bool(plan.indexer) or plan.indexer_type == "shared"
        if not executes:
            return IndexReadCost(0, 0, 0, 0.0, 0.0, 0, 0.0)
        rows = completed_rows(sequence_tokens, 1)
        selected = min(rows, topk) if topk else rows
        row_bytes = float(index_dim * 2)
        write_rows = rows if plan.indexer else 0
        return IndexReadCost(
            scored_rows=rows, selected_rows=selected, candidate_rows=0,
            index_row_bytes=row_bytes, index_read_bytes=rows * row_bytes,
            cache_write_rows=write_rows,
            cache_write_bytes=write_rows * row_bytes,
        )
    if not (plan.indexer or plan.owns_topk):
        return IndexReadCost(0, 0, 0, 0.0, 0.0, 0, 0.0)
    owner = int(plan.index_k_owner_layer_id)
    if owner < 0 or owner >= len(model.layer_plans):
        raise A3InputError(f"catalog_index_owner_missing:layer{plan.index}")
    owner_ratio = int(model.layer_plans[owner].compress_ratio)
    if owner_ratio <= 0:
        raise A3InputError(f"catalog_index_owner_ratio_invalid:layer{plan.index}")
    rows = completed_rows(sequence_tokens, owner_ratio)
    candidate_rows = 0
    if candidate_source >= 0 and plan.index > candidate_source:
        candidate_rows = candidate_valid_rows(
            sequence_tokens if int(model.layer_plans[candidate_source].compress_ratio) == 1
            else completed_rows(sequence_tokens, int(model.layer_plans[candidate_source].compress_ratio)),
            blocks=candidate_blocks, block_size=candidate_block,
        )
        scored = candidate_rows
    else:
        scored = rows
    selected = min(scored, topk) if topk else scored
    row_bytes = float(index_dim + 2 if native in {"deepseek_v4", "deepseek_v41"} else index_dim * 2)
    # Index-K writes happen only for declared index owners.
    write_rows = rows if plan.owns_index_k else 0
    write_row_bytes = float(index_dim + 2) if native in {"deepseek_v4", "deepseek_v41"} else float(index_dim * 2)
    return IndexReadCost(
        scored_rows=scored, selected_rows=selected,
        candidate_rows=candidate_rows, index_row_bytes=row_bytes,
        index_read_bytes=scored * row_bytes,
        cache_write_rows=write_rows,
        cache_write_bytes=write_rows * write_row_bytes,
    )


def attention_rows(
    model: Any, plan: Any, sequence_tokens: int,
) -> tuple[int, int, int]:
    """Return (dense_cache_rows, selected_long_rows, swa_rows) per layer."""
    seq = max(0, int(sequence_tokens))
    if plan.attention == "gqa":
        return seq, 0, 0
    if plan.attention in {"kda", "gdn"}:
        return 0, 0, 0
    if plan.attention != "mla":
        raise A3InputError(f"catalog_cache_attention_unsupported:{plan.attention}")
    native = str(_facts(model).get("profile", ""))
    if native in {"kimi", "glm"}:
        return seq, 0, 0
    swa = window_rows(seq, plan.window)
    if plan.compress_ratio <= 0:
        return 0, 0, swa
    facts = _facts(model)
    topk = int(facts.get("index_topk", 0) or 0)
    if native == "deepseek_v4":
        rows = completed_rows(seq, plan.compress_ratio)
        if plan.compress_ratio == 4:
            return 0, (min(rows, topk) if topk else rows), swa
        return 0, rows, swa
    if native == "deepseek_v41":
        cost = index_selection_cost(model, plan, seq)
        if plan.indexer_type in {"full", "shared"} and plan.owns_topk:
            selected = cost.selected_rows
        elif plan.topk_owner_layer_id >= 0:
            owner_cost = index_selection_cost(
                model, model.layer_plans[plan.topk_owner_layer_id], seq,
            )
            selected = owner_cost.selected_rows
        else:
            selected = min(completed_rows(seq, plan.compress_ratio), topk or seq)
        return 0, selected, swa
    raise A3InputError(f"catalog_cache_profile_unsupported:{native}")


def _local_state_sharded(plan: Any, tp: int) -> tuple[float, float]:
    tp = max(1, int(tp))
    recurrent = float(plan.recurrent_state_elements) / tp
    conv = float(plan.conv_state_elements) / tp
    return recurrent, conv


def layer_cache_components(
    model: Any, plan: Any, parallel: ParallelSpec, sequence_tokens: int,
) -> dict[str, float]:
    """Per-rank storage for one layer at the given committed sequence."""
    seq = max(0, int(sequence_tokens))
    block = 64
    tp = max(1, parallel.tp)
    components: dict[str, float] = {}
    facts = _facts(model)
    native = str(facts.get("profile", ""))
    if plan.attention == "gqa":
        local = local_kv_heads(plan.num_kv_heads, tp)
        rows = block_round(seq, block)
        components["gqa_kv"] = float(rows * 2 * local * plan.head_dim * 2)
        return components
    if plan.attention in {"kda", "gdn"}:
        recurrent, conv = _local_state_sharded(plan, tp)
        components["recurrent_state"] = recurrent * 4.0
        components["conv_state"] = conv * 2.0
        return components
    if plan.attention != "mla":
        raise A3InputError(f"catalog_cache_attention_unsupported:{plan.attention}")
    if native in {"kimi", "glm"}:
        rows = block_round(seq, block)
        components["mla_kv"] = float(rows * plan.cache_row_elements * 2.0)
        if native == "glm" and plan.indexer:
            # Producer planes only; shared consumers read the owner's plane.
            components["index_k"] = float(seq * int(facts.get("index_head_dim", 128)) * 2.0)
        return components
    if native not in {"deepseek_v4", "deepseek_v41"}:
        raise A3InputError(f"catalog_cache_profile_unsupported:{native}")
    swa_rows = window_rows(seq, plan.window)
    if plan.window:
        components["swa"] = float(swa_rows * plan.cache_row_elements * 2.0)
    if plan.owns_long_kv and plan.compress_ratio > 0:
        rows = completed_rows(seq, plan.compress_ratio)
        components["long_kv"] = float(rows * plan.cache_row_elements * 2.0)
    if plan.owns_index_k and plan.compress_ratio > 0:
        rows = completed_rows(seq, plan.compress_ratio)
        components["index_k"] = float(rows * (int(facts.get("index_head_dim", 128)) + 2))
    if native == "deepseek_v4":
        if plan.compress_ratio == 4:
            components["c4_compressor_state"] = float(min(seq, 8) * 4 * plan.head_dim * 4)
            components["c4_index_state"] = float(min(seq, 8) * 4 * int(facts.get("index_head_dim", 128)) * 4)
        elif plan.compress_ratio == 128:
            components["c128_compressor_state"] = float(min(seq, 128) * 2 * plan.head_dim * 4)
    if native == "deepseek_v41" and plan.owns_long_kv and plan.compress_ratio == 2 and seq > 0:
        # Three C2 owners each keep a live FP32 KV+score ring of capacity 32:
        # 32 rows * 2 * head_dim * 4 bytes.  Valid rows are separate below.
        components["c2_ring"] = float(32 * 2 * plan.head_dim * 4)
    return components


def stage_cache_components(
    model: Any, parallel: ParallelSpec, sequence_tokens: int,
) -> list[dict[str, float]]:
    pp = max(1, parallel.pp)
    stages = [dict() for _ in range(pp)]
    per = max(1, model.layers // pp)
    for plan in model.layer_plans:
        stage = min(pp - 1, plan.index // per) if pp > 1 else 0
        for key, value in layer_cache_components(model, plan, parallel, sequence_tokens).items():
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
    "candidate_valid_rows", "block_round", "local_kv_heads",
    "index_selection_cost", "attention_rows", "layer_cache_components",
    "stage_cache_components", "request_cache_breakdown",
    "state_bytes_per_request_per_rank", "cache_bytes_per_token_estimate",
    "peak_live_tokens", "live_request_cache_peak",
]
