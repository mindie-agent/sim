"""Known projection and cache work surrounding an unsupported attention core.

[Role] Preserve executable known components and explicit causal omissions.
[Invariants] An omitted core is a zero-work marker, never a GQA substitute.
  Shared KV consumes the real producer and transfers across PP only when needed.
[Timeouts & Locks] Bounded layer/request arithmetic; no I/O or locks.
[Side-Effect Policy] Returns new operators without changing model state.
"""
from mindie_sim.catalog.attention_common import AttentionResult, _allreduce
from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.ops import barrier_op, cache_store_op, comm_op, gemm_op
from mindie_sim.schema.common import ModelingError, dtype_bytes


def shared_kv_input(*, prefix, plan, stage, views, cache_ids, stage_of, parallel, quant, mode):
    owner = plan.kv_owner_layer_id
    if owner not in cache_ids:
        raise ModelingError(f"shared_kv_producer_missing:{plan.index}:{owner}")
    ready = cache_ids[owner]
    source_stage = stage_of(owner)
    if source_stage == stage:
        return [], ready
    volume = (sum(view.kv_len for view in views) * 2 *
              local_kv_heads(plan.num_kv_heads, parallel.tp) * plan.head_dim *
              dtype_bytes(quant.kv_dtype) / parallel.cp)
    op = comm_op(
        op_id=prefix + ".shared_kv_transfer", deps=(ready,), stage=stage, layer=plan.index,
        requests=tuple(view.request_id for view in views), family="cache_transfer",
        phase=mode, recipe="shared_kv_transfer", link_bytes=volume,
        shape=(sum(view.kv_len for view in views), plan.num_kv_heads, plan.head_dim),
        metadata=(("group_axis", "pp"), ("source_stage", source_stage),
                  ("destination_stage", stage)),
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
    shared = 0 <= plan.kv_owner_layer_id < plan.index
    q = heads * plan.head_dim * (2 if plan.q_output_gate else 1)
    projected = gemm_op(
        op_id=prefix + ".known_qkv", deps=(deps,), stage=stage, layer=plan.index,
        requests=requests, m=padded_tokens, n=q + (0 if shared else 2 * kv * plan.head_dim),
        k=model.hidden_size, quant=quant, category="attn.qkv", phase=mode,
        recipe="qkv_projection", family="qkv_projection",
        metadata=(("partial_model_component", True),),
    )
    ops, inputs, cache = [projected], [projected.op_id], None
    if plan.cache_kind == "dense":
        if shared:
            extra, cache = shared_kv_input(
                prefix=prefix, plan=plan, stage=stage, views=views, cache_ids=cache_ids,
                stage_of=stage_of, parallel=parallel, quant=quant, mode=mode)
            ops.extend(extra)
        else:
            store = cache_store_op(
                op_id=prefix + ".known_cache", deps=(projected.op_id,), stage=stage,
                layer=plan.index, requests=requests, phase=mode, recipe="kv_store",
                rows=sum(v.q_len for v in views) / parallel.cp,
                row_bytes=2 * kv * plan.head_dim * dtype_bytes(quant.kv_dtype),
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
        n=model.hidden_size, k=heads * plan.head_dim, quant=quant,
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
