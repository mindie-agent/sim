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

from typing import Any

from mindie_sim.catalog.cache_format import cache_format, gqa_kv_formats
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
    index_row_parts,
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
from mindie_sim.catalog.context_parallel import (
    cp_interleave, cp_plan_kind, dcp_indexer_sharded, layer_dcp_degree,
)
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
    """Exact integer ceiling multiple of ``block_size`` (immune to float53 rounding)."""
    count = int(rows)
    block = int(block_size)
    if count <= 0:
        return 0
    if block <= 1:
        return count
    return ((count + block - 1) // block) * block


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


def cache_row_parts(
    model: Any,
    region: str,
    width: int,
    *,
    plan: Any = None,
    phase: str | None = None,
) -> tuple[int, int, int]:
    """Exact integer ``(payload_bytes, scale_bytes, padding_bytes)`` per row."""
    parts = packed_row_parts(model, region, width, plan=plan, phase=phase)
    if parts is not None:
        data, scales, padding = parts
        return int(data), int(scales), int(padding)
    return cache_format(model, region, plan=plan, phase=phase).row_parts(int(width))


def cache_row_split(
    model: Any,
    region: str,
    width: int,
    *,
    plan: Any = None,
    phase: str | None = None,
) -> tuple[float, float]:
    """Return materialized data and allocated scale-plane bytes (including pad).

    Callers that need trailing alignment padding separated from numerical scales
    use ``cache_row_parts`` directly.
    """
    data, scales, padding = cache_row_parts(
        model, region, width, plan=plan, phase=phase,
    )
    return float(data), float(scales + padding)


def packed_region_row_bytes(
    model: Any,
    region: str,
    width: int,
    *,
    plan: Any = None,
    phase: str | None = None,
) -> int:
    """Physical row width for ``region``, honoring per-layer ABI and MQSMLA modes."""
    data, scales, padding = cache_row_parts(
        model, region, width, plan=plan, phase=phase,
    )
    return int(data + scales + padding)


def cache_component_parts(
    model: Any,
    plan: Any,
    parallel: ParallelSpec,
    component: str,
    total: float,
    *,
    phase: str | None = None,
) -> tuple[float, float, float]:
    """Decompose one ``layer_cache_components`` total into ``(payload, scale, padding)``."""
    total = float(total)
    if total <= 0.0:
        return 0.0, 0.0, 0.0
    if component in {"mla_kv", "window", "compressed_kv"}:
        packed_parts = (
            _packed_mla_row_parts(model, plan, phase=phase)
            if component == "mla_kv" else None
        )
        if packed_parts is not None:
            payload, scale, padding = float(packed_parts[0]), float(packed_parts[1]), 0.0
        else:
            p_b, s_b, pad_b = cache_row_parts(
                model, component, plan.cache_row_elements, plan=plan, phase=phase,
            )
            payload, scale, padding = float(p_b), float(s_b), float(pad_b)
    elif component == "gqa_kv":
        heads = local_kv_heads(plan.num_kv_heads, parallel.tp)
        key_fmt, val_fmt, is_shared = gqa_kv_formats(model, plan=plan, phase=phase)
        if is_shared:
            p_b, s_b, pad_b = cache_row_parts(
                model, "gqa_kv", 2 * heads * plan.head_dim, plan=plan, phase=phase,
            )
            payload, scale, padding = float(p_b), float(s_b), float(pad_b)
        else:
            side_width = heads * plan.head_dim
            kp, ks, kpad = key_fmt.row_parts(side_width)
            vp, vs, vpad = val_fmt.row_parts(side_width)
            payload, scale, padding = float(kp + vp), float(ks + vs), float(kpad + vpad)
    elif component == "index_k":
        p_b, s_b, pad_b = index_row_parts(model, plan.index_dim, plan=plan, phase=phase)
        payload, scale, padding = float(p_b), float(s_b), float(pad_b)
    else:
        return total, 0.0, 0.0
    row_total = payload + scale + padding
    if row_total <= 0.0:
        return total, 0.0, 0.0
    return (
        total * (payload / row_total),
        total * (scale / row_total),
        total * (padding / row_total),
    )


def cache_component_split(
    model: Any, plan: Any, parallel: ParallelSpec, component: str, total: float,
    *, phase: str | None = None,
) -> tuple[float, float]:
    """Decompose one ``layer_cache_components`` value into ``(payload, scale+pad)``."""
    payload, scale, padding = cache_component_parts(
        model, plan, parallel, component, total, phase=phase,
    )
    return payload, scale + padding


def layer_cache_decomposition(
    model: Any, plan: Any, parallel: ParallelSpec, sequence_tokens: int, *,
    capacity_padded: bool = True, interleave: int = INDEX_INTERLEAVE_TOKENS,
    phase: str = "decode",
) -> dict[str, dict[str, float]]:
    """Per-component payload/scale/padding decomposition for one layer's cache."""
    components = layer_cache_components(
        model, plan, parallel, sequence_tokens,
        capacity_padded=capacity_padded, interleave=interleave, phase=phase,
    )
    result: dict[str, dict[str, float]] = {}
    for name, value in components.items():
        payload, scale, padding = cache_component_parts(
            model, plan, parallel, name, value, phase=phase,
        )
        result[name] = {
            "total": float(value),
            "payload": float(payload),
            "scale": float(scale),
            "padding": float(padding),
        }
    return result


def _facts(model: Any) -> dict[str, Any]:
    return dict(getattr(model, "catalog_facts", ()) or ())


def _context_shard_blocks(seq: int, block: int, cp: int) -> int:
    rows = max(0, int(seq))
    blk = max(1, int(block))
    if cp <= 1:
        return block_round(rows, blk)
    blocks = _ceil_div(rows, blk)
    return _ceil_div(blocks, int(cp)) * blk


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


def _local_cache_rows(
    plan: Any,
    parallel: ParallelSpec,
    rows: int,
    *,
    cp: int,
    cp_stride: int,
    alloc_block: int,
    capacity_padded: bool,
    rank: int | None,
    owner_spans: Any,
    start: int = 0,
    ratio: int = 1,
) -> float:
    """Compute rank-local or aggregate rows keeping CP stride and page block distinct."""
    stride = max(1, int(cp_stride))
    page_blk = max(1, int(alloc_block))
    if rank is not None:
        spans = (
            tuple((lo // ratio, hi // ratio) for lo, hi in owner_spans)
            if owner_spans is not None and ratio > 1 else owner_spans
        )
        owned = owned_token_rows(rows, spans, rank, start=start)
        if owned is not None:
            return float(block_round(owned, page_blk) if capacity_padded else owned)
        if cp <= 1:
            return float(block_round(rows, page_blk) if capacity_padded else rows)
        if rows <= 0:
            return 0.0
        from mindie_sim.catalog.context_parallel import layer_dcp_factors, owned_positions
        pcp_f, tp_g, dp_w = layer_dcp_factors(plan, parallel)
        tp_dim = max(1, int(parallel.tp))
        pcp_dim = max(1, int(parallel.pcp))
        pp_dim = max(1, int(parallel.pp))
        dp_dim = max(1, int(parallel.dp))
        r_int = int(rank)
        tp_i = r_int % tp_dim
        pcp_i = (r_int // tp_dim) % pcp_dim
        dp_i = (r_int // (tp_dim * pcp_dim * pp_dim)) % dp_dim
        if dp_w > 1:
            cp_rank = ((dp_i % dp_w) * (pcp_f * tp_g) + (pcp_i % pcp_f) * tp_g + (tp_i % tp_g)) % cp
        else:
            cp_rank = ((tp_i % tp_g) * pcp_f + (pcp_i % pcp_f)) % cp
        exact = owned_positions(start, start + rows - 1, cp, stride, cp_rank)
        return float(block_round(exact, page_blk) if capacity_padded else exact)
    if capacity_padded:
        if cp <= 1:
            return float(block_round(rows, page_blk))
        if rows <= 0:
            return 0.0
        if stride == page_blk and start == 0:
            return float(_context_shard_blocks(rows, page_blk, cp))
        from mindie_sim.catalog.context_parallel import owned_positions
        worst = max(owned_positions(start, start + rows - 1, cp, stride, r) for r in range(cp))
        return float(block_round(worst, page_blk))
    return float(rows) if cp <= 1 else rows / cp


def _window_cache_rows(seq: int, window: int | None, *, padded: bool, block: int) -> int:
    """Sliding-window row count, adding +1 active write page when padded and seq > window."""
    if not window or window <= 0:
        return seq
    if padded and seq > window:
        blk = max(1, int(block))
        return (_ceil_div(window, blk) + 1) * blk
    return min(seq, window)


def _gqa_layer_row_bytes(
    model: Any, plan: Any, tp: int, *, int8_fa3: bool, phase: str,
) -> int:
    heads = local_kv_heads(plan.num_kv_heads, tp)
    if int8_fa3:
        return 2 * heads * (int(plan.head_dim) + C8_SCALE_RESERVED_BYTES)
    key_fmt, val_fmt, is_shared = gqa_kv_formats(model, plan=plan, phase=phase)
    if is_shared:
        return int(key_fmt.row_bytes(2 * heads * int(plan.head_dim)))
    side_width = heads * int(plan.head_dim)
    return int(key_fmt.row_bytes(side_width) + val_fmt.row_bytes(side_width))


def layer_cache_components(model: Any, plan: Any, parallel: ParallelSpec,
                           sequence_tokens: int, *, capacity_padded: bool = True,
                           interleave: int = INDEX_INTERLEAVE_TOKENS,
                           block_size: int | None = None,
                           rank: int | None = None, owner_spans: Any = None,
                           runtime: Any = None, query_tokens: int | None = None,
                           phase: str = "decode") -> dict[str, float]:
    seq = max(0, int(sequence_tokens))
    int8_fa3 = gbsa_cache_layout(runtime) == "int8_fa3"
    tp = parallel.tp
    cp = layer_dcp_degree(plan, parallel)
    page_block = int(
        block_size if block_size is not None
        else (getattr(runtime, "kv_block_size", 0) or interleave)
    )
    result: dict[str, float] = {}

    def local_rows(rows: int, stride: int = interleave, *, start: int = 0, ratio: int = 1) -> float:
        alloc_blk = max(1, page_block // max(1, ratio))
        return _local_cache_rows(
            plan, parallel, rows, cp=cp, cp_stride=stride, alloc_block=alloc_blk,
            capacity_padded=capacity_padded, rank=rank, owner_spans=owner_spans,
            start=start, ratio=ratio,
        )

    if plan.attention in _LINEAR_ATTENTION_KINDS:
        if runtime is not None or query_tokens is not None:
            result = _active_state_components(
                model, plan, parallel, runtime=runtime,
                sequence_tokens=seq, query_tokens=query_tokens, phase=phase,
                capacity_padded=capacity_padded)
        else:
            pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
            shards = tp * (pcp if plan.attention == "kda" else 1)
            result["recurrent_state"] = float(exact_element_bytes(
                _ceil_div(int(plan.recurrent_state_elements), shards),
                str(plan.recurrent_state_dtype),
            ))
            result["conv_state"] = float(exact_element_bytes(
                _ceil_div(int(plan.conv_state_elements), shards),
                str(plan.conv_state_dtype),
            ))
        if plan.engram and seq > 0:
            from mindie_sim.catalog.qwen4_exp import is_qwen4_exp_ple, ple_conv_state_bytes
            if is_qwen4_exp_ple(getattr(model, "engram", None)):
                result["ple_conv_state"] = ple_conv_state_bytes(
                    model, runtime, query_tokens=query_tokens,
                )
        return result
    if plan.attention == "unmodeled" or plan.cache_kind == "none":
        return result
    if plan.attention in {"gqa", "projection_only"}:
        if not 0 <= plan.kv_owner_layer_id < plan.index:
            rows = _window_cache_rows(seq, plan.window, padded=capacity_padded, block=page_block)
            start = max(0, seq - min(seq, plan.window)) if plan.window else 0
            row_bytes = _gqa_layer_row_bytes(
                model, plan, tp, int8_fa3=int8_fa3, phase=phase,
            )
            result["gqa_kv"] = local_rows(rows, start=start) * row_bytes
    elif plan.attention != "mla":
        raise ModelingError(f"cache_attention_unknown:{plan.attention}")
    elif plan.cache_kind == "dense":
        rows = _window_cache_rows(seq, plan.window, padded=capacity_padded, block=page_block)
        start = max(0, seq - min(seq, plan.window)) if plan.window else 0
        result["mla_kv"] = local_rows(rows, start=start) * _packed_mla_row_bytes(
            model, plan, phase=phase,
        )
    else:
        if plan.window:
            result["window"] = local_rows(
                min(seq, plan.window), start=max(0, seq - min(seq, plan.window))) * packed_region_row_bytes(
                model, "window", plan.cache_row_elements, plan=plan, phase=phase)
        if plan.owns_long_kv:
            ratio = max(1, plan.compress_ratio)
            compressed = local_rows(completed_rows(seq, plan.compress_ratio),
                                    max(1, interleave // ratio), ratio=ratio)
            result["compressed_kv"] = compressed * packed_region_row_bytes(
                model, "compressed_kv", plan.cache_row_elements, plan=plan, phase=phase)
            if plan.compressor_state_rows and seq:
                rows = plan.compressor_state_rows if capacity_padded else min(seq, plan.compressor_state_rows)
                result["compressor_state"] = rows * 2 * plan.compressor_width * plan.head_dim * FP32_ELEMENT_BYTES
    if plan.engram and seq > 0:
        from mindie_sim.catalog.qwen4_exp import is_qwen4_exp_ple, ple_conv_state_bytes
        if is_qwen4_exp_ple(getattr(model, "engram", None)):
            result["ple_conv_state"] = ple_conv_state_bytes(
                model, runtime, query_tokens=query_tokens,
            )
    if plan.owns_index_k:
        index_ratio = max(1, plan.compress_ratio, plan.index_kpool)
        rows = completed_rows(seq, index_ratio)
        index_block = max(1, page_block // index_ratio)
        if dcp_indexer_sharded(model, runtime) and cp > 1:
            rows = local_rows(rows, max(1, interleave // index_ratio), ratio=index_ratio)
        elif parallel.cp_plan == "dsa_cp" and cp > 1:
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
        elif cp <= 1:
            rows = local_rows(rows, max(1, interleave // index_ratio), ratio=index_ratio)
        elif capacity_padded:
            rows = float(block_round(rows, index_block))
        if int8_fa3:
            result["index_k"] = rows * (plan.index_dim + C8_SCALE_RESERVED_BYTES)
        else:
            result["index_k"] = rows * index_row_bytes(
                model, plan.index_dim, plan=plan, phase=phase,
            )
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
            idx_state_rows = (
                plan.compressor_state_rows if capacity_padded
                else min(seq, plan.compressor_state_rows)
            )
            result["index_compressor_state"] = (
                idx_state_rows
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
    interleave: int = INDEX_INTERLEAVE_TOKENS, block_size: int | None = None,
    runtime: Any = None, query_tokens: int | None = None, phase: str = "decode",
) -> list[dict[str, float]]:
    pp = max(1, parallel.pp)
    stages: list[dict[str, float]] = [{} for _ in range(pp)]
    stage_map = layer_stage_map(model, parallel, runtime)
    for plan in model.layer_plans:
        stage = stage_map[plan.index] if plan.index < len(stage_map) else pp - 1
        for key, value in layer_cache_components(
            model, plan, parallel, sequence_tokens,
            capacity_padded=capacity_padded, interleave=interleave,
            block_size=block_size, runtime=runtime,
            query_tokens=query_tokens, phase=phase,
        ).items():
            stages[stage][key] = stages[stage].get(key, 0.0) + value
    if draft_kind and getattr(model, "auxiliary", None) is not None:
        from mindie_sim.catalog.draft import optional_draft_spec
        from mindie_sim.catalog.draft_spec import effective_draft_context_cap
        draft = optional_draft_spec(model, draft_kind)
        draft_seq = max(0, int(sequence_tokens))
        draft_cap = effective_draft_context_cap(draft, runtime)
        if draft_cap > 0:
            draft_seq = min(draft_seq, draft_cap)
        if uses_inherited_draft_cache(model, draft_kind):
            from mindie_sim.catalog.cache_ownership import draft_cache_components
            draft_values = draft_cache_components(
                model, parallel, draft_kind=draft_kind,
                sequence_tokens=draft_seq, capacity_padded=capacity_padded,
                runtime=runtime, query_tokens=query_tokens, phase=phase,
            )
            stages[-1]["draft_kv"] = stages[-1].get("draft_kv", 0.0) + float(
                draft_values["draft_kv"])
            if draft_values.get("draft_state", 0.0) > 0.0:
                stages[-1]["draft_state"] = stages[-1].get(
                    "draft_state", 0.0) + float(draft_values["draft_state"])
        else:
            row_bytes = _draft_row_bytes_int(model, draft, tp=parallel.tp)
            block = max(1, int(block_size if block_size is not None else (
                getattr(runtime, "kv_block_size", 0) or interleave
            )))
            draft_sharded = _draft_is_dcp_sharded(
                model, parallel, draft_kind,
                dcp=int(getattr(parallel, "cp", 1) or 1),
                draft_page=block * row_bytes,
            )
            draft_cp = int(parallel.cp) if draft_sharded else 1
            seq_rows = (
                _context_shard_blocks(draft_seq, block, draft_cp) if capacity_padded
                else draft_seq / draft_cp
            )
            stages[-1]["draft_kv"] = stages[-1].get("draft_kv", 0.0) + float(
                seq_rows * row_bytes * draft.layers
            )
    return stages


def request_cache_breakdown(
    model: Any, parallel: ParallelSpec, sequence_tokens: int,
    *, draft_kind: str | None = None, interleave: int | None = None,
    runtime: Any = None, query_tokens: int | None = None, phase: str = "decode",
) -> dict[str, Any]:
    block_size = int(
        interleave if interleave is not None
        else (getattr(runtime, "kv_block_size", 0) or INDEX_INTERLEAVE_TOKENS)
    )
    cp_stride = int(
        interleave if interleave is not None
        else (cp_interleave(runtime) if runtime is not None else block_size)
    )
    stages = stage_cache_components(
        model, parallel, sequence_tokens, draft_kind=draft_kind,
        interleave=cp_stride, block_size=block_size,
        runtime=runtime, query_tokens=query_tokens, phase=phase,
    )
    totals: dict[str, float] = {}
    for stage in stages:
        for key, value in stage.items():
            totals[key] = totals.get(key, 0.0) + value
    stage_bytes = [float(sum(stage.values())) for stage in stages]
    from mindie_sim.catalog.qwen4_exp import qwen4_exp_pool_breakdown
    qwen4_pool = qwen4_exp_pool_breakdown(
        model, parallel, sequence_tokens, runtime=runtime, query_tokens=query_tokens,
    )
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


__all__ = [
    "INDEX_INTERLEAVE_TOKENS",
    "IndexReadCost", "completed_rows", "new_completed_rows", "window_rows",
    "candidate_valid_rows", "block_round", "index_element_bytes",
    "index_k_block_rows", "index_k_capacity_rows", "index_k_owned_block_rows",
    "index_k_ownership_rows", "index_k_store_rows",
    "index_row_bytes", "index_row_parts", "index_scale_bytes", "kv_element_bytes",
    "local_kv_heads", "cp_plan_kind",
    "index_selection_cost", "cache_row_parts", "cache_row_split",
    "packed_region_row_bytes",
    "cache_component_parts", "cache_component_split", "layer_cache_components",
    "layer_cache_decomposition", "layer_stage_map", "stage_cache_components",
    "request_cache_breakdown",
    "gbsa_cache_layout", "target_mla_page_bytes", "separate_pool_page_pitches",
    "request_pool_page_pitches",
    "cache_bytes_per_token_estimate",
]
