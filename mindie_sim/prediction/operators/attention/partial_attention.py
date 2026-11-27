"""Known projection and cache work surrounding an unsupported attention core.

[Role] Preserve executable known components and explicit causal omissions.
[Invariants] An omitted core is a zero-work marker, never a GQA substitute.
  Shared KV consumes the real producer and transfers across PP only when needed.
[Timeouts & Locks] Bounded layer/request arithmetic; no I/O or locks.
[Side-Effect Policy] Returns new operators without changing model state.
"""
from mindie_sim.catalog.attention_common import AttentionResult, _allreduce
from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.cache_format import gqa_kv_formats
from mindie_sim.catalog.context_parallel import DEFAULT_CP_INTERLEAVE, layer_dcp_degree, owned_positions
from mindie_sim.catalog.ops import barrier_op, cache_store_op, comm_op, gemm_op
from mindie_sim.schema.common import ModelingError, dtype_bytes


_MX_SCALE_GROUP_SIZE = 32
_MX_V_SCALE_TOKEN_GROUP = 32
_MX_SCALE_DTYPES = frozenset({"fp8_e8m0", "e8m0"})
_FP8_E4M3_DTYPES = frozenset({"fp8_e4m3", "mxfp8"})


def _is_mxfp8_fmt(fmt) -> bool:
    if fmt is None:
        return False
    return (
        str(getattr(fmt, "dtype", "") or "").strip().lower() in _FP8_E4M3_DTYPES
        and int(getattr(fmt, "group_size", 0) or 0) == _MX_SCALE_GROUP_SIZE
        and str(getattr(fmt, "scale_dtype", "") or "").strip().lower() in _MX_SCALE_DTYPES
        and int(getattr(fmt, "suffix_elements", 0) or 0) == 0
        and int(getattr(fmt, "packed_elements", 1) or 1) == 1
    )


def _gqa_kv_row_bytes(model, plan, kv_heads: int, v_dim: int, quant, mode: str) -> float:
    """Resolve physical per-token GQA KV cache row bytes matching the cache producer."""
    if model is not None:
        key_fmt, value_fmt, shared_format = gqa_kv_formats(model, plan=plan, phase=mode)
        if shared_format:
            return float(key_fmt.row_bytes(kv_heads * (plan.head_dim + v_dim)))
        return float(
            key_fmt.row_bytes(kv_heads * plan.head_dim)
            + value_fmt.row_bytes(kv_heads * v_dim)
        )
    return float(kv_heads * (plan.head_dim + v_dim) * dtype_bytes(quant.kv_dtype))


def _gqa_kv_transfer_bytes(
    model, plan, kv_heads: int, v_dim: int, quant, mode: str, tokens: int,
) -> float:
    """Physical bytes for ``tokens`` of GQA KV cache, preserving MXFP8 V token-group scales."""
    if tokens <= 0:
        return 0.0
    if model is not None:
        key_fmt, value_fmt, _shared = gqa_kv_formats(model, plan=plan, phase=mode)
        if _is_mxfp8_fmt(key_fmt) and _is_mxfp8_fmt(value_fmt):
            k_scale_groups = -(-plan.head_dim // _MX_SCALE_GROUP_SIZE)
            v_scale_groups = -(-tokens // _MX_V_SCALE_TOKEN_GROUP)
            return float(
                tokens * kv_heads * (plan.head_dim + v_dim + k_scale_groups)
                + v_scale_groups * kv_heads * v_dim
            )
    return float(tokens) * _gqa_kv_row_bytes(model, plan, kv_heads, v_dim, quant, mode)


def _max_dcp_rank_tokens(
    ranges, dcp: int, parallel, *, rank_candidates=None, interleave: int | None = None,
) -> int:
    """Rank-exact (when ``rank_candidates`` is given) or heaviest-rank owned KV row count."""
    if dcp <= 1:
        return sum(hi - lo + 1 for lo, hi in ranges if hi >= lo)
    eff_interleave = max(
        1,
        int(interleave)
        if interleave is not None
        else int(getattr(parallel, "cp_interleave", DEFAULT_CP_INTERLEAVE) or DEFAULT_CP_INTERLEAVE),
    )
    ranks = (
        tuple(int(r) for r in rank_candidates if 0 <= int(r) < dcp)
        if rank_candidates is not None else tuple(range(dcp))
    )
    return max(
        (sum(owned_positions(lo, hi, dcp, eff_interleave, rank) for lo, hi in ranges if hi >= lo)
         for rank in ranks),
        default=0,
    )


def shared_kv_input(
    *, prefix, plan, stage, views, cache_ids, stage_of, parallel, quant, mode, model=None,
    rank_candidates=None, cp_interleave_size: int | None = None,
):
    owner = plan.kv_owner_layer_id
    if owner not in cache_ids:
        raise ModelingError(f"shared_kv_producer_missing:{plan.index}:{owner}")
    ready = cache_ids[owner]
    source_stage = stage_of(owner)
    if source_stage == stage:
        return [], ready
    v_dim = plan.v_head_dim or plan.head_dim
    kv_heads = local_kv_heads(plan.num_kv_heads, parallel.tp)
    dcp = max(1, layer_dcp_degree(plan, parallel))
    kv_tokens = sum(view.kv_len for view in views)
    rank_tokens = _max_dcp_rank_tokens(
        tuple((0, view.kv_len - 1) for view in views if view.kv_len > 0),
        dcp, parallel, rank_candidates=rank_candidates, interleave=cp_interleave_size,
    )
    owner_plans = getattr(model, "layer_plans", ()) if model is not None else ()
    owner_plan = owner_plans[owner] if 0 <= owner < len(owner_plans) else plan
    volume = _gqa_kv_transfer_bytes(model, owner_plan, kv_heads, v_dim, quant, mode, rank_tokens)
    op = comm_op(
        op_id=prefix + ".shared_kv_transfer", deps=(ready,), stage=stage, layer=plan.index,
        requests=tuple(view.request_id for view in views), family="cache_transfer",
        phase=mode, recipe="shared_kv_transfer", link_bytes=volume,
        shape=(kv_tokens, kv_heads, plan.head_dim + v_dim),
        metadata=(("group_axis", "pp"), ("source_stage", source_stage),
                  ("destination_stage", stage), ("v_head_dim", v_dim),
                  ("layer_dcp_degree", dcp)),
    )
    return [op], op.op_id


def unmodeled(*, prefix, deps, stage, plan, views, mode, **_):
    marker = barrier_op(
        op_id=prefix + ".unmodeled_attention", deps=(deps,), stage=stage,
        layer=plan.index, requests=tuple(v.request_id for v in views), phase=mode,
        recipe="unmodeled_attention", shape=(),
        metadata=(("unmodeled_component", plan.architecture),),
    )
    return AttentionResult(ops=[marker], last=marker.op_id)


def projection_only(*, prefix, deps, stage, plan, views, mode, model, parallel,
                    quant, padded_tokens, cache_ids, stage_of, **_):
    requests = tuple(v.request_id for v in views)
    heads = plan.num_heads // parallel.tp
    kv = local_kv_heads(plan.num_kv_heads, parallel.tp)
    v_dim = plan.v_head_dim or plan.head_dim
    dcp = max(1, layer_dcp_degree(plan, parallel))
    shared = 0 <= plan.kv_owner_layer_id < plan.index
    q = plan.q_projection_width(heads, plan.head_dim)
    projected = gemm_op(
        op_id=prefix + ".known_qkv", deps=(deps,), stage=stage, layer=plan.index,
        requests=requests, m=padded_tokens, n=q + (0 if shared else kv * (plan.head_dim + v_dim)),
        k=model.hidden_size, quant=quant, category="attn.qkv", phase=mode,
        recipe="qkv_projection", family="qkv_projection",
        metadata=(("partial_model_component", True),),
    )
    ops, inputs, cache = [projected], [projected.op_id], None
    if plan.cache_kind == "dense":
        if shared:
            extra, cache = shared_kv_input(
                prefix=prefix, plan=plan, stage=stage, views=views, cache_ids=cache_ids,
                stage_of=stage_of, parallel=parallel, quant=quant, mode=mode, model=model)
            ops.extend(extra)
        else:
            store_rows = _max_dcp_rank_tokens(
                tuple((v.context, v.context + v.logical_q_len - 1) for v in views if v.logical_q_len > 0),
                dcp, parallel,
            )
            store = cache_store_op(
                op_id=prefix + ".known_cache", deps=(projected.op_id,), stage=stage,
                layer=plan.index, requests=requests, phase=mode, recipe="kv_store",
                rows=store_rows,
                row_bytes=_gqa_kv_row_bytes(model, plan, kv, v_dim, quant, mode),
                dtype=quant.kv_dtype, kind="gqa",
            )
            ops.append(store)
            cache = store.op_id
        inputs.append(cache)
    marker = barrier_op(
        op_id=prefix + ".unmodeled_attention_core", deps=tuple(inputs), stage=stage,
        layer=plan.index, requests=requests, phase=mode,
        recipe="unmodeled_attention_core", shape=(),
        metadata=(("unmodeled_component", plan.architecture),),
    )
    ops.append(marker)
    output = gemm_op(
        op_id=prefix + ".known_attention_output", deps=(marker.op_id,),
        stage=stage, layer=plan.index, requests=requests, m=padded_tokens,
        n=model.hidden_size, k=heads * v_dim, quant=quant,
        category="attn.o", phase=mode, recipe="attention_output", family="output_projection",
        metadata=(("partial_model_component", True),),
    )
    ops.append(output)
    last = output.op_id
    if parallel.tp > 1:
        reduction = _allreduce(
            prefix=prefix, deps=(last,), stage=stage, layer=plan.index,
            requests=requests, model=model, parallel=parallel, mode=mode, tokens=padded_tokens)
        ops.append(reduction)
        last = reduction.op_id
    return AttentionResult(ops=ops, last=last, cache_store_id=cache)
