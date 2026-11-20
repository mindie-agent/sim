"""Compressed-cache, window, candidate and state geometry for catalog models.

[Role]
  Compute exact completed compressed rows, local-window rows, sparse candidate
  validity, TopK selection reads, C2 parity across chunks, owner-shared long
  KV/index storage, independent pool page pitches and per-rank per-request
  cache/state bytes.

[Invariants]
  * Completed rows are ``floor(committed_tokens / ratio)``; an odd residual
    remains in state across chunk boundaries instead of being rounded up.
  * Candidate block IDs expand to real valid rows; the final partial block is
    kept and block 0 is not silently treated as an extra candidate block.
  * Unique producer storage is counted once; every consumer still pays its own
    logical read.  Sharing is not a network collective by itself.
  * A DSA-CP index-K cache is owned in whole interleave blocks via
    ``cache_index``; per-rank rows sum strictly to total valid rows.
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
from typing import Any

from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.cache_index import (
    INDEX_INTERLEAVE_TOKENS,
    IndexReadCost,
    candidate_valid_rows,
    completed_rows,
    index_element_bytes,
    index_k_block_rows,
    index_k_capacity_rows,
    index_k_owned_block_rows,
    index_k_ownership_rows,
    index_k_store_rows,
    index_row_bytes,
    index_scale_bytes,
    index_selection_cost,
    new_completed_rows,
    window_rows,
)
from mindie_sim.catalog.cache_pools import (
    LINEAR_ATTENTION_KINDS as _LINEAR_ATTENTION_KINDS,
    _ceil_div,
    _draft_is_dcp_sharded,
    _draft_row_bytes_int,
    _packed_mla_row_bytes,
    _packed_mla_row_parts,
    exact_element_bytes,
    gbsa_cache_layout,
    request_pool_page_pitches,
    separate_pool_page_pitches,
    target_mla_page_bytes,
)
from mindie_sim.catalog.context_parallel import cp_plan_kind
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.model import WorkloadSpec
from mindie_sim.schema.parallel import ParallelSpec

from .mqsmla_layout import packed_row_parts

#: Reserved per-row scale/alignment bytes for INT8 FA3 C8 GQA/index rows.
C8_SCALE_RESERVED_BYTES = 4
FP32_ELEMENT_BYTES = 4
BF16_ELEMENT_BYTES = 2
DEFAULT_CACHE_BLOCK_SIZE = 128


def block_round(rows: int, block_size: int) -> int:
    if block_size <= 1:
        return int(rows)
    return int(math.ceil(int(rows) / int(block_size)) * int(block_size))


def local_kv_heads(num_kv_heads: int, tp: int) -> int:
    from mindie_sim.schema.parallel import local_kv_heads as _heads
    return _heads(num_kv_heads, tp)


def _quant(model: Any) -> Any:
    return getattr(model, "quant", None)


def kv_element_bytes(model: Any) -> float:
    """Declared KV row width in bytes; the quant plan owns the swap axis."""
    quant = _quant(model)
    dtype = getattr(quant, "kv_dtype", "") if quant is not None else ""
    return dtype_bytes(str(dtype or getattr(model, "kv_dtype", "bf16")))


def kv_bytes_for(model: Any, elements: float) -> float:
    return float(elements) * kv_element_bytes(model)


def _mqsmla_mode_row_split(model: Any, region: str, width: int) -> tuple[float, float] | None:
    parts = packed_row_parts(model, region, width)
    if parts is not None:
        data, scales, padding = parts
        return float(data), float(scales + padding)
    return None


def cache_row_split(model: Any, region: str, width: int) -> tuple[float, float]:
    """Return materialized data and allocated scale-plane bytes (including pad).

    These two allocation columns exhaust the physical row. Typed operator
    views obtain the numerical scales and alignment bytes from packed_row_parts.
    """
    packed = _mqsmla_mode_row_split(model, region, width)
    if packed is not None:
        return packed
    fmt = cache_format(model, region)
    quantized = int(width) - int(fmt.suffix_elements)
    if quantized < 0:
        raise ModelingError("cache_suffix_exceeds_row_width")
    total = int(fmt.row_bytes(int(width)))
    scale = (
        exact_element_bytes(_ceil_div(quantized, fmt.group_size), fmt.scale_dtype)
        if fmt.group_size else 0
    )
    payload = total - scale
    if payload < 0:
        raise ModelingError("cache_row_split_negative_payload")
    return float(payload), float(scale)


def packed_region_row_bytes(model: Any, region: str, width: int) -> int:
    """Physical row width for ``region``, honoring arch35 MQSMLA Mode-1/2."""
    parts = packed_row_parts(model, region, width)
    if parts is not None:
        data, scales, padding = parts
        return int(data) + int(scales) + int(padding)
    return int(cache_format(model, region).row_bytes(int(width)))


def cache_component_split(
    model: Any, plan: Any, parallel: ParallelSpec, component: str, total: float,
) -> tuple[float, float]:
    """Decompose one ``layer_cache_components`` value without rounding it again."""
    total = float(total)
    if total <= 0.0:
        return 0.0, 0.0
    if component in {"mla_kv", "window", "compressed_kv"}:
        packed_parts = _packed_mla_row_parts(model, plan) if component == "mla_kv" else None
        if packed_parts is not None:
            payload, scale = float(packed_parts[0]), float(packed_parts[1])
        else:
            payload, scale = cache_row_split(model, component, plan.cache_row_elements)
    elif component == "gqa_kv":
        width = 2 * local_kv_heads(plan.num_kv_heads, parallel.tp) * plan.head_dim
        payload, scale = cache_row_split(model, "gqa_kv", width)
    elif component == "index_k":
        raw = dict(getattr(model, "metadata", ())).get("cache_representations", {})
        if "index_k" in raw:
            payload, scale = cache_row_split(model, "index_k", plan.index_dim)
        else:
            payload = float(plan.index_dim) * index_element_bytes(model)
            scale = index_scale_bytes(model)
    else:
        return total, 0.0
    row_total = payload + scale
    if row_total <= 0.0:
        return total, 0.0
    return total * (payload / row_total), total * (scale / row_total)


def layer_cache_decomposition(
    model: Any, plan: Any, parallel: ParallelSpec, sequence_tokens: int, *,
    capacity_padded: bool = True, interleave: int = INDEX_INTERLEAVE_TOKENS,
) -> dict[str, dict[str, float]]:
    """Per-component payload/scale decomposition for one layer's cache."""
    components = layer_cache_components(
        model, plan, parallel, sequence_tokens,
        capacity_padded=capacity_padded, interleave=interleave,
    )
    result: dict[str, dict[str, float]] = {}
    for name, value in components.items():
        payload, scale = cache_component_split(model, plan, parallel, name, value)
        result[name] = {
            "total": float(value),
            "payload": float(payload),
            "scale": float(scale),
        }
    return result


def _facts(model: Any) -> dict[str, Any]:
    return dict(getattr(model, "catalog_facts", ()) or ())


def attention_rows(model: Any, plan: Any, sequence_tokens: int) -> tuple[int, int, int]:
    del model
    seq = max(0, int(sequence_tokens))
    if plan.attention in _LINEAR_ATTENTION_KINDS or plan.cache_kind == "recurrent_unmodeled":
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


def owned_block_rows(rows: int, block: int, cp: int, rank: int) -> int:
    """Exact token rows this physical rank owns under block interleave."""
    rows = max(0, int(rows))
    block = max(1, int(block))
    cp = max(1, int(cp))
    rank = int(rank)
    if not 0 <= rank < cp:
        raise ModelingError(f"cache_owner_rank_out_of_range:{rank}")
    if cp == 1:
        return block_round(rows, block)
    blocks = int(math.ceil(rows / block)) if rows else 0
    if rank >= blocks:
        return 0
    count = (blocks - 1 - rank) // cp + 1
    last = rank + (count - 1) * cp
    owned = count * block
    tail = rows - last * block
    if tail < block:
        owned -= block - tail
    return int(owned)


def owned_token_rows(rows: int, owner_spans: Any, rank: int, *,
                     start: int = 0) -> int | None:
    """Rows owned by ``rank`` from an explicit half-open token partition."""
    if owner_spans is None:
        return None
    spans = tuple((int(lo), int(hi)) for lo, hi in owner_spans)
    rank = int(rank)
    if not 0 <= rank < len(spans):
        raise ModelingError(f"cache_owner_rank_out_of_range:{rank}")
    rows = max(0, int(rows))
    begin = max(0, int(start))
    end = begin + rows
    covered = sum(max(0, min(end, hi) - max(begin, lo)) for lo, hi in spans)
    lo, hi = spans[rank]
    owned = int(max(0, min(end, hi) - max(begin, lo)))
    if rows > covered and rank == len(spans) - 1:
        owned += rows - covered
    return owned


def _runtime_recover_ssm(runtime: Any) -> bool:
    if runtime is None:
        return False
    if bool(getattr(runtime, "recover_ssm", False)):
        return True
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    return metadata.get("recover_ssm") is True


def _active_state_components(model: Any, plan: Any, parallel: ParallelSpec, *,
                             runtime: Any, sequence_tokens: int,
                             query_tokens: int | None, phase: str,
                             capacity_padded: bool) -> dict[str, float]:
    del sequence_tokens
    from mindie_sim.catalog.speculative_state import (
        resolve_speculative_query_tokens,
        speculative_state_plan,
    )
    q = (int(query_tokens) if query_tokens is not None
         else resolve_speculative_query_tokens(runtime, phase=phase))
    state = speculative_state_plan(
        model, plan, parallel, query_tokens=q,
        recover_ssm=_runtime_recover_ssm(runtime),
        capacity_padded=capacity_padded,
        state_contract=getattr(plan, "state_contract", None),
        runtime=runtime,
    )
    result = {"recurrent_state": state.recurrent_bytes,
              "conv_state": state.conv_bytes}
    if state.replay_bytes > 0:
        result["replay_records"] = state.replay_bytes
    if state.transient_bytes > 0:
        result["state_transient"] = state.transient_bytes
    return result


def layer_cache_components(model: Any, plan: Any, parallel: ParallelSpec,
                           sequence_tokens: int, *, capacity_padded: bool = True,
                           interleave: int = INDEX_INTERLEAVE_TOKENS,
                           rank: int | None = None, owner_spans: Any = None,
                           runtime: Any = None, query_tokens: int | None = None,
                           phase: str = "decode") -> dict[str, float]:
    seq = max(0, int(sequence_tokens))
    int8_fa3 = gbsa_cache_layout(runtime) == "int8_fa3"
    tp, cp = parallel.tp, parallel.cp
    result: dict[str, float] = {}

    def local_rows(rows: int, block: int = interleave, *, start: int = 0, ratio: int = 1) -> float:
        if rank is not None:
            spans = (tuple((lo // ratio, hi // ratio) for lo, hi in owner_spans)
                     if owner_spans is not None and ratio > 1 else owner_spans)
            owned = owned_token_rows(rows, spans, rank, start=start)
            if owned is not None:
                return owned
            if capacity_padded:
                return owned_block_rows(rows, block, cp, rank)
            return rows / cp
        return (_context_shard_blocks(rows, block, cp) if capacity_padded
                else rows / cp)

    if plan.attention in _LINEAR_ATTENTION_KINDS:
        if runtime is not None or query_tokens is not None:
            result = _active_state_components(
                model, plan, parallel, runtime=runtime,
                sequence_tokens=seq, query_tokens=query_tokens, phase=phase,
                capacity_padded=capacity_padded)
        else:
            result["recurrent_state"] = float(exact_element_bytes(
                _ceil_div(int(plan.recurrent_state_elements), tp),
                str(plan.recurrent_state_dtype),
            ))
            result["conv_state"] = float(exact_element_bytes(
                _ceil_div(int(plan.conv_state_elements), tp),
                str(plan.conv_state_dtype),
            ))
        if plan.engram and seq > 0:
            from mindie_sim.catalog.qwen4_exp import is_qwen4_exp_ple, ple_conv_state_bytes
            if is_qwen4_exp_ple(getattr(model, "engram", None)):
                result["ple_conv_state"] = ple_conv_state_bytes(model, runtime)
        return result
    if plan.attention == "unmodeled" or plan.cache_kind == "none":
        return result
    if plan.attention in {"gqa", "projection_only"}:
        if not 0 <= plan.kv_owner_layer_id < plan.index:
            rows = min(seq, plan.window) if plan.window else seq
            start = max(0, seq - rows) if plan.window else 0
            heads = local_kv_heads(plan.num_kv_heads, tp)
            if int8_fa3:
                row_bytes = 2 * heads * (plan.head_dim + C8_SCALE_RESERVED_BYTES)
            else:
                width = 2 * heads * plan.head_dim
                row_bytes = cache_format(model, "gqa_kv").row_bytes(width)
            result["gqa_kv"] = local_rows(rows, start=start) * row_bytes
    elif plan.attention != "mla":
        raise ModelingError(f"cache_attention_unknown:{plan.attention}")
    elif plan.cache_kind == "dense":
        rows = min(seq, plan.window) if plan.window else seq
        start = max(0, seq - rows) if plan.window else 0
        result["mla_kv"] = local_rows(rows, start=start) * _packed_mla_row_bytes(model, plan)
    else:
        if plan.window:
            rows = min(seq, plan.window)
            result["window"] = local_rows(
                rows, start=max(0, seq - rows)) * packed_region_row_bytes(
                model, "window", plan.cache_row_elements)
        if plan.owns_long_kv:
            ratio = max(1, plan.compress_ratio)
            compressed = local_rows(completed_rows(seq, plan.compress_ratio),
                                    max(1, interleave // ratio), ratio=ratio)
            result["compressed_kv"] = compressed * packed_region_row_bytes(
                model, "compressed_kv", plan.cache_row_elements)
            if plan.compressor_state_rows and seq:
                rows = plan.compressor_state_rows if capacity_padded else min(seq, plan.compressor_state_rows)
                result["compressor_state"] = rows * 2 * plan.compressor_width * plan.head_dim * FP32_ELEMENT_BYTES
    if plan.engram and seq > 0:
        from mindie_sim.catalog.qwen4_exp import is_qwen4_exp_ple, ple_conv_state_bytes
        if is_qwen4_exp_ple(getattr(model, "engram", None)):
            result["ple_conv_state"] = ple_conv_state_bytes(model, runtime)
    if plan.owns_index_k:
        index_ratio = max(1, plan.compress_ratio, plan.index_kpool)
        rows = completed_rows(seq, index_ratio)
        from .paired_sfa import execution
        if execution(model).get("dcp_indexer_sharded") and cp > 1:
            rows = local_rows(rows, max(1, interleave // index_ratio), ratio=index_ratio)
        elif parallel.cp_plan == "dsa_cp":
            if rank is not None:
                cp_rank = (int(rank) % max(1, int(cp))
                           if owner_spans is not None else int(rank))
                if not 0 <= cp_rank < cp:
                    raise ModelingError(f"cache_owner_rank_out_of_range:{rank}")
                rows = (
                    index_k_owned_block_rows(rows, cp, index_ratio)[cp_rank]
                    if capacity_padded else
                    index_k_ownership_rows(rows, cp, index_ratio)[cp_rank]
                )
            else:
                rows = index_k_capacity_rows(rows, cp, index_ratio) if capacity_padded else rows / cp
        if int8_fa3:
            result["index_k"] = rows * (plan.index_dim + C8_SCALE_RESERVED_BYTES)
        else:
            result["index_k"] = rows * index_row_bytes(model, plan.index_dim)
        if plan.index_kpool > 1 and seq:
            if plan.attention == "gqa":
                ring_rows = 2 * plan.index_kpool if capacity_padded else min(seq, 2 * plan.index_kpool)
                result["index_pool_tail"] = ring_rows * plan.index_dim * BF16_ELEMENT_BYTES
            else:
                capacity = dict(model.metadata).get("native_keypool_caller", {}).get(
                    "tail_capacity", plan.index_kpool)
                rows = capacity if capacity_padded else min(seq, capacity)
                result["index_pool_tail"] = rows * 2 * plan.index_dim * FP32_ELEMENT_BYTES
        if plan.index_input == "hidden" and plan.compress_ratio and seq:
            result["index_compressor_state"] = (
                min(seq, plan.compressor_state_rows)
                * 2 * plan.compressor_width * plan.index_dim * FP32_ELEMENT_BYTES
            )
    return result


def layer_stage_map(model: Any, parallel: ParallelSpec, runtime: Any = None) -> tuple[int, ...]:
    """Resolved physical PP stage for every declared layer."""
    pp = max(1, int(parallel.pp))
    layers = max(1, int(getattr(model, "layers", 0) or len(
        getattr(model, "layer_plans", ()))))
    partition = tuple(
        int(value) for value in (getattr(runtime, "pp_layer_partition", ()) or ()))
    if partition:
        if len(partition) != pp or sum(partition) != layers:
            raise ModelingError(
                f"pp_partition_layers:{sum(partition)}!={layers}")
        return tuple(
            stage for stage, count in enumerate(partition) for _ in range(count))
    if pp == 1:
        return (0,) * layers
    return tuple(min(pp - 1, index * pp // layers)
                 for index in range(layers))


def uses_inherited_draft_cache(model: Any, draft_kind: str | None) -> bool:
    """True when MTP draft KV/index/pool storage follows declared LayerPlans."""
    if draft_kind != "mtp" or getattr(model, "auxiliary", None) is None:
        return False
    if dict(getattr(model, "metadata", ()) or ()).get("cache_allocator") == "paired_sfa":
        return False
    return any(
        layer.window > 0 or layer.compress_ratio > 0 or layer.owns_index_k
        for layer in getattr(model, "layer_plans", ())
    )


def stage_cache_components(
    model: Any, parallel: ParallelSpec, sequence_tokens: int,
    *, capacity_padded: bool = True, draft_kind: str | None = None,
    interleave: int = INDEX_INTERLEAVE_TOKENS, runtime: Any = None,
    query_tokens: int | None = None, phase: str = "decode",
) -> list[dict[str, float]]:
    pp = max(1, parallel.pp)
    stages: list[dict[str, float]] = [{} for _ in range(pp)]
    stage_map = layer_stage_map(model, parallel, runtime)
    for plan in model.layer_plans:
        stage = stage_map[plan.index] if plan.index < len(stage_map) else pp - 1
        for key, value in layer_cache_components(
            model, plan, parallel, sequence_tokens,
            capacity_padded=capacity_padded, interleave=interleave,
            runtime=runtime, query_tokens=query_tokens, phase=phase,
        ).items():
            stages[stage][key] = stages[stage].get(key, 0.0) + value
    if draft_kind and getattr(model, "auxiliary", None) is not None:
        if uses_inherited_draft_cache(model, draft_kind):
            from mindie_sim.catalog.cache_ownership import draft_cache_components
            draft_values = draft_cache_components(
                model, parallel, draft_kind=draft_kind,
                sequence_tokens=sequence_tokens, capacity_padded=capacity_padded,
                runtime=runtime, query_tokens=query_tokens, phase=phase,
            )
            stages[-1]["draft_kv"] = stages[-1].get("draft_kv", 0.0) + float(
                draft_values["draft_kv"])
            if draft_values.get("draft_state", 0.0) > 0.0:
                stages[-1]["draft_state"] = stages[-1].get(
                    "draft_state", 0.0) + float(draft_values["draft_state"])
        else:
            from mindie_sim.catalog.draft import optional_draft_spec
            draft = optional_draft_spec(model, draft_kind)
            row_bytes = _draft_row_bytes_int(model, draft, tp=parallel.tp)
            seq = max(0, int(sequence_tokens))
            block = max(1, int(interleave))
            draft_sharded = _draft_is_dcp_sharded(
                model, parallel, draft_kind,
                dcp=int(getattr(parallel, "cp", 1) or 1),
                draft_page=block * row_bytes,
            )
            draft_cp = int(parallel.cp) if draft_sharded else 1
            seq_rows = (
                _context_shard_blocks(seq, block, draft_cp) if capacity_padded
                else seq / draft_cp
            )
            stages[-1]["draft_kv"] = stages[-1].get("draft_kv", 0.0) + float(
                seq_rows * row_bytes * draft.layers
            )
    return stages


def request_cache_breakdown(
    model: Any, parallel: ParallelSpec, sequence_tokens: int,
    *, draft_kind: str | None = None, interleave: int = INDEX_INTERLEAVE_TOKENS,
    runtime: Any = None, query_tokens: int | None = None, phase: str = "decode",
) -> dict[str, Any]:
    block_size = int(
        interleave if interleave != INDEX_INTERLEAVE_TOKENS
        else (getattr(runtime, "kv_block_size", 0) or interleave or 1)
    )
    stages = stage_cache_components(
        model, parallel, sequence_tokens, draft_kind=draft_kind, interleave=block_size,
        runtime=runtime, query_tokens=query_tokens, phase=phase,
    )
    totals: dict[str, float] = {}
    for stage in stages:
        for key, value in stage.items():
            totals[key] = totals.get(key, 0.0) + value
    stage_bytes = [float(sum(stage.values())) for stage in stages]
    from mindie_sim.catalog.qwen4_exp import qwen4_exp_pool_breakdown
    qwen4_pool = qwen4_exp_pool_breakdown(model, parallel, sequence_tokens, runtime=runtime)
    out: dict[str, Any] = {
        "sequence_tokens": int(sequence_tokens),
        "by_component": totals,
        "stage_bytes": stage_bytes,
        "total_bytes": float(sum(stage_bytes)),
        "max_stage_bytes": float(max(stage_bytes) if stage_bytes else 0),
        "largest_stage": int(max(range(len(stage_bytes)), key=lambda i: stage_bytes[i])) if stage_bytes else 0,
    }
    if qwen4_pool is not None:
        out["qwen4_exp_pool"] = qwen4_pool
    pool_page_pitches = request_pool_page_pitches(
        model, parallel, block_size=block_size, draft_kind=draft_kind,
        dcp=int(getattr(parallel, "cp", 1) or 1), runtime=runtime,
    )
    if pool_page_pitches is not None:
        out["pool_page_pitches"] = pool_page_pitches
    return out


def state_bytes_per_request_per_rank(model: Any, parallel: ParallelSpec,
                                     runtime: Any = None) -> float:
    total = 0
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    axis = str(dict(getattr(runtime, "metadata", ()) or ()).get(
        "kda_parallel_axis", ""))
    for plan in model.layer_plans:
        if plan.attention not in _LINEAR_ATTENTION_KINDS:
            continue
        shards = max(1, int(parallel.tp)) * (
            pcp if (axis == "tp_x_pcp" and plan.attention == "kda") else 1
        )
        total += exact_element_bytes(
            _ceil_div(int(plan.recurrent_state_elements), shards),
            str(plan.recurrent_state_dtype),
        )
        total += exact_element_bytes(
            _ceil_div(int(plan.conv_state_elements), shards),
            str(plan.conv_state_dtype),
        )
    return float(total)


def cache_bytes_per_token_estimate(
    model: Any, parallel: ParallelSpec, sequence_tokens: int,
    *, draft_kind: str | None = None, interleave: int = INDEX_INTERLEAVE_TOKENS,
) -> tuple[float, float]:
    """(max-stage cache bytes per token, total cache bytes at sequence)."""
    seq = max(1, int(sequence_tokens))
    stages = stage_cache_components(
        model, parallel, seq, draft_kind=draft_kind, interleave=interleave,
    )
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
    *, block_size: int = DEFAULT_CACHE_BLOCK_SIZE, media_tokens: int = 0,
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
    "INDEX_INTERLEAVE_TOKENS",
    "IndexReadCost", "completed_rows", "new_completed_rows", "window_rows",
    "candidate_valid_rows", "block_round", "index_element_bytes",
    "index_k_block_rows", "index_k_capacity_rows", "index_k_owned_block_rows",
    "index_k_ownership_rows", "index_k_store_rows",
    "index_row_bytes", "index_scale_bytes", "kv_bytes_for", "kv_element_bytes",
    "local_kv_heads", "cp_plan_kind",
    "index_selection_cost", "attention_rows", "cache_row_split",
    "packed_region_row_bytes",
    "cache_component_split", "layer_cache_components",
    "layer_cache_decomposition", "layer_stage_map", "stage_cache_components",
    "request_cache_breakdown",
    "gbsa_cache_layout", "target_mla_page_bytes", "separate_pool_page_pitches",
    "request_pool_page_pitches",
    "state_bytes_per_request_per_rank", "cache_bytes_per_token_estimate",
    "peak_live_tokens", "live_request_cache_peak",
]
