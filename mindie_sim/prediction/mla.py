"""Latent, compressed and indexed attention from explicit layer geometry.

[Role] Compose projections, cache producers, sparse selection and attention.
[Invariants] Cache ownership and selection reuse are dependencies. Compression
  counts complete groups per request. Logical pairs and tiled memory reads are
  different quantities. No model/vendor identity selects an execution path.
[Timeouts & Locks] O(requests + query tiles) arithmetic, no observation reads.
[Side-Effect Policy] Constructs operators only.
"""
import math

from mindie_sim.catalog.attention_common import (
    AttentionResult, _causal_pairs, _long_selected_pairs, _group_plan, dcp_partial_merge_ops,
)
from mindie_sim.catalog.cache import completed_rows, index_selection_cost, index_row_bytes
from mindie_sim.catalog.context_parallel import split_causal_pairs
from mindie_sim.schema.common import dtype_bytes, ModelingError
from .implementations import select
from .lowering import Builder
from .mla_structure import declare_indexer
from .compression import compressor
from .indexer_representation import index_representation
from .mla_native import batched_matrix_metadata, index_layer_norm
from .projections import mla_input_projections
from . import mla_head_padding
from .cache_representation import store as representation_store
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.ops import compute_dtype_for


def _attention_tensor_view(plan, view, heads, model, quant, parallel):
    """Declare active latent tensors separately from paged cache allocation."""
    if parallel.cp != 1 or parallel.pcp != 1:
        return ()
    common = {
        "query_dtype": model.activation_dtype,
        "cache_dtype": quant.kv_dtype,
        "output_dtype": model.activation_dtype,
    }
    if plan.mla_variant == "latent_mla" and plan.index_topk > 0:
        query = (view.q_len, heads, plan.kv_lora_rank)
        return (("attention_tensor_view", {
            **common, "kind": "indexed_latent",
            "query_shape": query, "output_shape": query,
            "query_rope_shape": (view.q_len, heads, plan.qk_rope_head_dim),
            "cache_head_shape": (1, plan.kv_lora_rank),
            "rope_cache_head_shape": (1, plan.qk_rope_head_dim),
        }),)
    if (plan.mla_variant == "latent_mqa" and plan.cache_kind == "window_compressed"
            and plan.head_dim == plan.qk_nope_head_dim + plan.qk_rope_head_dim):
        query = (view.q_len, heads, plan.head_dim)
        regions = dict(model.metadata).get("cache_representations", {})
        if regions:
            return (("attention_tensor_view", {
                "kind": "mixed_quant_shared_kv", "query_shape": query, "output_shape": query,
                "query_dtype": model.activation_dtype, "output_dtype": model.activation_dtype,
                "cache_representations": regions,
                "original_cache_row_bytes": cache_format(model, "window").row_bytes(plan.head_dim),
                "compressed_cache_row_bytes": cache_format(
                    model, "compressed_kv").row_bytes(plan.head_dim),
            }),)
        return (("attention_tensor_view", {
            **common, "kind": "shared_kv",
            "query_shape": query, "output_shape": query,
            "cache_head_shape": (1, plan.head_dim),
            "compressed_cache": plan.compress_ratio > 0,
        }),)
    return ()


def mla(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
        padded_tokens, cache_ids, topk_ids, index_cache_ids, candidate_ids,
        stage_of, cp_interleave_size=128, runtime=None, policy=None):
    b = Builder(prefix, plan, stage, views, model, mode)
    prolog = (select(runtime.kernels.get("lowerings", {}), "mla_prolog")
              if runtime else None)
    if prolog:
        prolog.module.validate(plan=plan, parallel=parallel, model=model, quant=quant,
                               lowering_passes=runtime.lowering_passes, **prolog.options)
    partial_rotary = (select(runtime.kernels.get("lowerings", {}), "v41_rotary")
                      if runtime else None)
    if partial_rotary:
        partial_rotary.module.validate(plan, fused_prolog=bool(prolog))
    selected = (select(runtime.kernels.get("lowerings", {}), "mla_decode")
                if runtime and any(view.phase == "decode" for view in views) else None)
    paged = (select(runtime.kernels.get("lowerings", {}), "paged_mla_attention")
             if runtime and plan.mla_variant == "latent_mla" and plan.cache_kind == "dense"
             and not plan.index_topk and any(view.phase == "decode" for view in views) else None)
    if paged and selected:
        raise ModelingError("paged_mla_and_mla_decode_bind_the_same_core")
    selected = paged or selected
    mixed_quant = (select(runtime.kernels.get("lowerings", {}), "mqsmla_attention")
                   if runtime and plan.mla_variant == "latent_mqa"
                   and plan.cache_kind == "window_compressed"
                   and any(view.phase == "decode" for view in views) else None)
    if mixed_quant:
        mixed_quant.module.validate(views=views, plan=plan, parallel=parallel,
                                    model=model, padded_tokens=padded_tokens,
                                    **mixed_quant.options)
    if selected:
        selected.module.validate(
            views=views, plan=plan, parallel=parallel, model=model, quant=quant,
            padded_tokens=padded_tokens, **selected.options)
    q, h, heads = padded_tokens, model.hidden_size, plan.num_heads // parallel.tp
    head_contract = (mla_head_padding.contract(model, plan, views)
                     if runtime and select(runtime.kernels.get("lowerings", {}), "mla_head_padding")
                     else None)
    logical_heads = heads
    if head_contract:
        heads = head_contract["physical_heads"]
    cp = parallel.cp
    width = plan.qk_nope_head_dim + plan.qk_rope_head_dim
    latent = plan.mla_variant != "latent_mqa"
    query_tile = policy.query_tile if policy else 128
    kv_width = plan.kv_lora_rank + (plan.qk_rope_head_dim if latent else 0)
    if plan.q_lora_rank:
        projected = mla_input_projections(
            b, plan=plan, tokens=q, hidden=h, kv_width=kv_width, deps=deps,
            packed=prolog is None)
        q_a, kv_a = projected["q_a"], projected["kv_a"]
        q_norm = b.vector("q_norm", (q_a,), q * plan.q_lora_rank, 5, family="rmsnorm")
        q_b = q_ready = b.matrix("q_b", (q_norm,), q, heads * width, plan.q_lora_rank,
                                category="mla.q_b")
    else:
        # q_lora_rank=0 is the checkpoint's direct q projection: one real
        # hidden->heads*(nope+rope) matrix, not zero-sized decomposition.
        if prolog:
            raise ModelingError("mla_prolog_v3_direct_q_unsupported")
        q_a = q_norm = q_b = q_ready = b.matrix(
            "q_proj", (deps,), q, heads * width, h, category="mla.q_proj")
        kv_a = b.matrix("kv_a", (deps,), q, kv_width, h, category="mla.kv_a")
    kv = b.vector("kv_norm", (kv_a,), q * plan.kv_lora_rank, 5, family="rmsnorm")
    if partial_rotary:
        q_ready = partial_rotary.module.declare(
            b, name="q_rotary", deps=(q_ready,), tokens=q, heads=heads,
            head_dim=width, rotary_width=plan.qk_rope_head_dim)
        rotary = partial_rotary.module.declare(
            b, name="kv_rotary", deps=(kv,), tokens=q, heads=1,
            head_dim=plan.kv_lora_rank, rotary_width=plan.qk_rope_head_dim)
    else:
        rotary = b.vector("rotary", (kv, q_ready), q * (heads + 1) * plan.qk_rope_head_dim, 6)
    valid = sum(v.q_len for v in views)
    packed_store = (prolog.module.cache_store(
        b, deps=(kv, rotary), rows=math.ceil(valid / cp), plan=plan, **prolog.options)
        if prolog else None)
    store = packed_store or (representation_store(b, "kv_store", (rotary,), math.ceil(valid / cp),
                                  plan.cache_row_elements, "window")
             if plan.cache_kind == "window_compressed" else
             b.store("kv_store", (rotary,), math.ceil(valid / cp),
                     plan.cache_row_elements * dtype_bytes(quant.kv_dtype), quant.kv_dtype))
    if latent:
        q_ready = b.matrix("q_absorb", (q_ready,), q * heads, plan.kv_lora_rank,
                           plan.qk_nope_head_dim, weight_count=heads,
                           category="mla_absorb",
                           metadata=batched_matrix_metadata(
                               tokens=q, heads=heads, input_width=plan.qk_nope_head_dim,
                               output_width=plan.kv_lora_rank, role="key_absorption"))
    if prolog:
        prolog.module.declare(
            b.ops, prefix=prefix, plan=plan, model=model, parallel=parallel, quant=quant,
            tokens=q, members=(q_a, q_norm, q_b, kv_a, kv, rotary, store, q_ready),
            q_a=q_a, q_b=q_b, kv_a=kv_a, q_absorb=q_ready, **prolog.options)
    long_store = store if plan.cache_kind == "dense" else None
    compressed_latent = None
    index_store, topk, candidate = None, None, None
    pool_weights = None

    def owner(mapping, layer, name, volume):
        if layer not in mapping:
            raise ModelingError(f"producer_missing:{prefix}:{name}:{layer}")
        ready = mapping[layer]
        producer_stage = stage_of(layer)
        if producer_stage != stage:
            ready = b.communicate(name + "_transfer", (ready,), volume, "cache_transfer", "pp",
                                  source_stage=producer_stage, destination_stage=stage)
        return ready

    if plan.cache_kind == "window_compressed" and plan.compress_ratio:
        if plan.owns_long_kv:
            new_rows = sum(completed_rows(v.context + v.q_len, plan.compress_ratio)
                           - completed_rows(v.context, plan.compress_ratio) for v in views)
            compressed_latent, rotated = compressor(
                b, name="compress", source=deps, tokens=valid, padded_tokens=q,
                valid_rows=new_rows, width=plan.head_dim, plan=plan, model=model,
                return_latent=True)
            long_store = representation_store(
                b, "compressed_store", (rotated,), math.ceil(new_rows / cp),
                plan.cache_row_elements, "compressed_kv")
        else:
            rows = sum(completed_rows(v.kv_len, plan.compress_ratio) for v in views)
            long_store = owner(cache_ids, plan.kv_owner_layer_id, "compressed_cache",
                               rows * cache_format(model, "compressed_kv").row_bytes(
                                   plan.cache_row_elements) / cp)

    if plan.index_kpool > 1 and (parallel.cp != 1 or parallel.pcp != 1):
        raise ModelingError("pooled_indexer_context_partition_not_declared")
    if plan.owns_index_k and plan.index_kpool > 1:
        from .index_pool import producer
        index_store, pool_weights = producer(
            b, source=deps, views=views, plan=plan, model=model, quant=quant,
            padded_tokens=q)
    elif plan.owns_index_k:
        index_source = long_store if plan.index_input == "compressed" else deps
        if (plan.index_input == "compressed"
                and cache_format(model, "compressed_kv").group_size):
            if compressed_latent is None:
                raise ModelingError("packed_compressed_indexer_requires_pre_rope_latent_producer")
            # Quantized, post-RoPE cache bytes cannot masquerade as the
            # compressor's pre-RoPE BF16 latent for the index projection.
            index_source = compressed_latent
        index_rows = sum(completed_rows(v.kv_len, max(1, plan.compress_ratio))
                         - completed_rows(v.context, max(1, plan.compress_ratio)) for v in views)
        input_rows = index_rows if plan.index_input == "compressed" else valid
        input_width = plan.head_dim if plan.index_input == "compressed" else h
        if plan.compress_ratio > 1 and plan.index_input == "hidden":
            normalized = compressor(
                b, name="index_compress", source=deps, tokens=valid, padded_tokens=q,
                valid_rows=index_rows, width=plan.index_dim, plan=plan, model=model)
        else:
            projected = b.matrix("index_k", (index_source,), input_rows, plan.index_dim, input_width,
                                 category="indexer_fp8")
            # The ordinary latent indexer uses mean/variance LayerNorm. The
            # compressor above has RMS normalization and its own parameters.
            normalized = index_layer_norm(
                b, source=projected, rows=index_rows, width=plan.index_dim)
            normalized = b.vector(
                "index_rotary", (normalized,), index_rows * plan.qk_rope_head_dim, 6,
                family="rotary", shape=(index_rows, plan.index_dim),
                read_bytes_override=index_rows * (
                    plan.index_dim * dtype_bytes(model.activation_dtype) + plan.qk_rope_head_dim * 8),
                write_bytes_override=index_rows * plan.index_dim * dtype_bytes(model.activation_dtype))
        normalized = index_representation(
            b, "index_key", normalized, index_rows, plan.index_dim,
            quant=quant, model=model, rotate_basis=plan.compress_ratio > 1)
        index_store = b.store("index_store", (normalized,), index_rows,
                              index_row_bytes(model, plan.index_dim), quant.index_dtype)
    if plan.indexer or plan.owns_topk or plan.index_query_per_layer:
        if index_store is None:
            index_store = owner(index_cache_ids, plan.index_k_owner_layer_id, "index_cache",
                                sum(index_selection_cost(model, plan, v.kv_len).index_read_bytes for v in views))
        q_index = b.matrix("index_q", (q_norm,), q, plan.index_heads * plan.index_dim,
                           plan.q_lora_rank, category="indexer_fp8")
        if plan.index_kpool == 1:
            q_index = b.vector(
                "index_query_rotary", (q_index,), valid * plan.index_heads * plan.qk_rope_head_dim, 6,
                family="rotary", shape=(valid, plan.index_heads, plan.index_dim))
        q_index = index_representation(
            b, "index_query", q_index, (q if plan.index_kpool > 1 else valid) * plan.index_heads, plan.index_dim,
            quant=quant, model=model, rotate_basis=plan.compress_ratio > 1)
        weights_input = deps
        weights_dtype = compute_dtype_for(b.quant, "indexer.weights")[1]
        # A declared FP32 projection consumes an FP32 input, not a BF16
        # allocation relabelled by the matrix's compute identity.
        if (pool_weights is None and weights_dtype in {"fp32", "fp16", "bf16"}
                and weights_dtype != model.activation_dtype):
            weights_input = b.vector(
                "index_weights_input_cast", (deps,), q * h, 1,
                family="cast", dtype=weights_dtype, shape=(q, h),
                read_bytes_override=q * h * dtype_bytes(model.activation_dtype),
                write_bytes_override=q * h * dtype_bytes(weights_dtype),
                metadata=(("input_format", model.activation_dtype),
                          ("output_format", weights_dtype)))
        # The KPool producer's fused [K, head weights] projection exposes a
        # strided column view. Its consumer cast reads only the head slice.
        weights = pool_weights or b.matrix(
            "index_weights", (weights_input,), q, plan.index_heads, h,
            category="indexer.weights")
        if quant.index_dtype == "int8":
            weights = b.vector(
                "index_weights_fp16", (weights,), q * plan.index_heads, 1,
                family="cast", dtype="fp16", shape=(q, plan.index_heads),
                read_bytes_override=q * plan.index_heads * dtype_bytes(weights_dtype),
                write_bytes_override=q * plan.index_heads * 2,
                metadata=(("source_semantics", "quant_indexer_weights_float16"),))
        elif plan.index_kpool > 1:
            if weights_dtype != quant.index_dtype:
                weights = b.vector(
                    "index_weights_cast", (weights,), q * plan.index_heads, 1,
                    family="cast", dtype=quant.index_dtype, shape=(q, plan.index_heads),
                    read_bytes_override=q * plan.index_heads * dtype_bytes(weights_dtype),
                    write_bytes_override=q * plan.index_heads * dtype_bytes(quant.index_dtype),
                    metadata=(("input_format", weights_dtype), ("output_format", quant.index_dtype))
                    + ((("input_column_slice", (plan.index_dim, plan.index_dim + plan.index_heads)),)
                       if pool_weights else ()))
            weights = b.vector("index_weights_scale", (weights,), q * plan.index_heads, 1,
                               dtype=quant.index_dtype, shape=(q, plan.index_heads))
        selection_deps = [q_index, weights, index_store]
        if plan.candidate_filter:
            selection_deps.append(owner(candidate_ids, plan.candidate_source_layer_id, "candidate",
                                        valid * plan.candidate_topk_blocks * 4))
        shard = cp if parallel.cp_plan == "dsa_cp" else 1
        if plan.index_kpool > 1:
            from .index_pool import selection
            binding = (select(runtime.kernels.get("lowerings", {}), "pool_key_indexer")
                       if runtime else None)
            topk = selection(b, query=q_index, weights=weights, cache=index_store,
                             views=views, plan=plan, quant=quant, padded_tokens=q,
                             **(binding.options if binding else {}))
        else:
            rows = sum(_long_selected_pairs(v.q_len, v.context, max(1, plan.compress_ratio),
                       plan.candidate_topk_blocks * plan.candidate_block_size if plan.candidate_filter else 0)
                       for v in views)
            scores = b.work("index_scores", selection_deps, (valid, plan.index_heads, plan.index_dim),
                            flops=2 * rows * plan.index_heads * plan.index_dim / shard,
                            vector=rows * plan.index_heads * 2 / shard,
                            read=rows * index_row_bytes(model, plan.index_dim) / shard,
                            write=rows * 4 / shard, family="indexer", dtype=quant.index_dtype)
            topk = b.work("index_topk", (scores,), (valid, plan.index_topk),
                         vector=rows * math.log2(max(2, plan.index_topk)) / shard,
                         read=rows * 4 / shard, write=valid * plan.index_topk * 4,
                         family="topk", dtype="fp32",
                         metadata=(("selection_output_dtype", "int32"),))
            declare_indexer(
                b.ops, prefix=prefix, plan=plan, parallel=parallel, model=model, quant=quant,
                tokens=valid, scores=scores, topk=topk)
        if shard > 1:
            gathered = b.communicate("candidate_gather", (topk,), valid * plan.index_topk * 8 * (cp - 1),
                                     "allgather", "cp")
            topk = b.vector("candidate_merge", (gathered,), valid * plan.index_topk * cp,
                           math.log2(max(2, plan.index_topk)), dtype="fp32", family="topk")
        if plan.owns_candidate_blocks:
            candidate = b.vector("candidate_blocks", (topk,), valid * plan.index_topk, 2, dtype="int32")
    elif plan.topk_owner_layer_id >= 0:
        topk = owner(topk_ids, plan.topk_owner_layer_id, "topk", valid * plan.index_topk * 4)

    if head_contract:
        q_ready, rotary = mla_head_padding.query_inputs(
            b, declaration=head_contract, tokens=q, query=q_ready, rotary=rotary, plan=plan)
    outputs = []
    metadata_ready = None
    metadata_source = (select(runtime.kernels.get("lowerings", {}), "sparse_mla_metadata")
                       if runtime else None)
    if metadata_source:
        metadata_ready = metadata_source.module.lower(
            b, views=views, plan=plan, parallel=parallel, quant=quant, deps=(topk or deps,),
            padded_tokens=q, stage_of=stage_of, **metadata_source.options)
    latent_dim = plan.kv_lora_rank if latent else plan.head_dim
    qk_dim = latent_dim + plan.qk_rope_head_dim if latent else width
    if selected:
        outputs.append(selected.module.lower_attention(
            b, views=views, plan=plan, heads=heads, runtime=runtime,
            query=q_ready, rotary=rotary, store=store,
            **({"head_contract": head_contract} if paged else {}), **selected.options))
    for index, view in enumerate(() if selected else views):
        context, query = view.context, view.q_len
        if plan.cache_kind == "dense":
            pairs = query * (context + query) if plan.noncausal_attention else _causal_pairs(query, context, plan.window)
            if topk:
                pairs = _long_selected_pairs(query, context, 1, plan.index_topk)
            sweeps = sum(min(context + min(query, start + query_tile), plan.index_topk or plan.window or view.kv_len)
                         for start in range(0, query, query_tile))
        else:
            pairs = _causal_pairs(query, context, plan.window) if plan.window else 0
            pairs += _long_selected_pairs(query, context, plan.compress_ratio, plan.index_topk if topk else 0)
            sweeps = sum((min(context + min(query, start + query_tile), plan.window) if plan.window else 0)
                         + min(completed_rows(context + min(query, start + query_tile), plan.compress_ratio),
                               plan.index_topk if topk else view.kv_len)
                         for start in range(0, query, query_tile))
        selection_reads, selection_metadata = 0, ()
        if topk:
            from .sparse_attention import sparse_attention_traffic
            traffic = sparse_attention_traffic(
                query, context, plan.index_topk,
                compression_ratio=1 if plan.cache_kind == "dense" else plan.compress_ratio,
                window=0 if plan.cache_kind == "dense" else plan.window,
                query_tile=query_tile, index_pool_size=plan.index_kpool,
                index_pool_tail=plan.index_kpool_always_select_tail)
            if plan.index_kpool > 1:
                pairs = traffic.total_cache_rows
            sweeps = traffic.total_cache_rows
            selection_reads = traffic.index_read_bytes
            selection_metadata = tuple(traffic.metadata().items()) + (
                ("program_kind", "indexed_attention"),
                ("indexed_attention_geometry", {
                    "query": query, "context": context, "topk": plan.index_topk,
                    "compression_ratio": 1 if plan.cache_kind == "dense" else plan.compress_ratio,
                    "window": 0 if plan.cache_kind == "dense" else plan.window,
                    "heads": heads, "latent": latent_dim,
                    "rope": plan.qk_rope_head_dim, "cp": cp,
                    "query_dtype": model.activation_dtype, "cache_dtype": quant.kv_dtype,
                    **({"index_pool_size": plan.index_kpool,
                        "index_pool_tail": plan.index_kpool_always_select_tail}
                       if plan.index_kpool > 1 else {}),
                }),
            )
        local_pairs = pairs / cp
        if cp > 1 and plan.cache_kind == "dense" and not topk:
            local_pairs = split_causal_pairs(query, context, cp, cp_interleave_size,
                                             window=plan.window,
                                             noncausal=plan.noncausal_attention).max_rank
        transfer = query * heads * (qk_dim + latent_dim) * dtype_bytes(model.activation_dtype)
        cache_read = sweeps * plan.cache_row_elements * dtype_bytes(quant.kv_dtype)
        if plan.cache_kind == "window_compressed":
            original_rows = sum(min(context + min(query, start + query_tile), plan.window)
                                for start in range(0, query, query_tile)) if plan.window else 0
            cache_read = (original_rows * cache_format(model, "window").row_bytes(plan.cache_row_elements)
                          + (sweeps - original_rows) * cache_format(
                              model, "compressed_kv").row_bytes(plan.cache_row_elements))
        last = b.work(
            f"latent_attention{index}", tuple(dict.fromkeys(
                x for x in (q_ready, rotary if latent else None, store, long_store, topk,
                            metadata_ready) if x)),
            (query, view.kv_len, heads, qk_dim, latent_dim),
            flops=2 * local_pairs * heads * (qk_dim + latent_dim),
            vector=5 * local_pairs * heads,
            read=cache_read / cp + selection_reads + transfer / 2,
            write=transfer / 2, family="attention",
            metadata=(("query_tile", query_tile), ("logical_pairs", pairs), ("kv_sweeps", sweeps))
                     + selection_metadata
                     + _attention_tensor_view(plan, view, heads, model, quant, parallel)
                     + ((("mla_head_contract", dict(head_contract)),) if head_contract else ()),
        )
        if (runtime and view.phase == "prefill" and topk
                and plan.cache_kind == "window_compressed"
                and plan.mla_variant == "latent_mqa"):
            sharedkv = select(runtime.kernels.get("lowerings", {}), "sharedkv_attention")
            if sharedkv:
                b.ops[-1] = sharedkv.module.lower_op(
                    b.ops[-1], selection=sharedkv, view=view, plan=plan, views=views,
                    heads=heads, parallel=parallel, model=model, quant=quant)
        if runtime and topk and plan.cache_kind == "dense":
            indexed = select(runtime.kernels.get("lowerings", {}), "indexed_attention")
            if indexed:
                b.ops[-1] = indexed.module.lower_op(
                    b.ops[-1], selection=indexed, view=view, views=views,
                    plan=plan, parallel=parallel)
        if cp > 1:
            merged, last = dcp_partial_merge_ops(
                prefix=prefix + f".request{index}", view=view, cp=cp, local_heads=heads,
                v_dim=latent_dim, stage=stage, layer=plan.index, mode=mode, deps=(last,),
                total_pairs=pairs, local_pairs=local_pairs, total_rows=view.kv_len,
                local_rows=math.ceil(view.kv_len / cp), interleave=cp_interleave_size,
            )
            b.ops.extend(merged)
        outputs.append(last)
    if mixed_quant:
        outputs = mixed_quant.module.lower_attention(
            b.ops, outputs, selection=mixed_quant, views=views, plan=plan,
            topk_present=bool(topk), prefix=prefix)
    if not selected and plan.cache_kind == "dense" and cp == 1 and parallel.pcp == 1:
        from .indexed_attention import batch_analytic_ops
        outputs = batch_analytic_ops(b.ops, outputs, runtime=runtime, views=views, prefix=prefix)
    heads = logical_heads
    if latent:
        outputs = [b.matrix("v_expand", outputs, q * heads, plan.v_head_dim, plan.kv_lora_rank,
                            weight_count=heads,
                            category="mla_absorb",
                            metadata=batched_matrix_metadata(
                                tokens=q, heads=heads, input_width=plan.kv_lora_rank,
                                output_width=plan.v_head_dim, role="value_expansion")
                                + mla_head_padding.output_metadata(head_contract))]
    output_width = heads * (plan.v_head_dim if latent else plan.head_dim)
    if plan.mla_output_gate and latent:
        gate = b.matrix("output_gate", (deps,), q, output_width, h, category="mla.g")
        outputs = [b.vector("gated_output", (*outputs, gate), q * output_width, 5)]
    if partial_rotary:
        outputs = [partial_rotary.module.declare(
            b, name="output_inverse_rotary", deps=outputs, tokens=q, heads=heads,
            head_dim=plan.head_dim, rotary_width=plan.qk_rope_head_dim, inverse=True)]
    group_parallel, groups = _group_plan(plan, parallel)
    if plan.o_groups:
        if not group_parallel and parallel.tp > 1:
            outputs = [b.communicate("output_gather", outputs, q * output_width * 2 * (parallel.tp - 1),
                                     "allgather")]
        group_width = plan.num_heads * (plan.v_head_dim if latent else plan.head_dim) // plan.o_groups
        first = b.matrix("grouped_output_a", outputs, q * groups, plan.o_lora_rank, group_width,
                         weight_count=groups, category="mla.o_a",
                         metadata=batched_matrix_metadata(
                             tokens=q, heads=groups, input_width=group_width,
                             output_width=plan.o_lora_rank, role="grouped_output"),
                         weight_bytes_override=b.quant.matrix_bytes(groups * plan.o_lora_rank, group_width, "mla.o_a"))
        last = b.matrix("grouped_output_b", (first,), q, h, groups * plan.o_lora_rank,
                        category="mla.o_b")
    else:
        last = b.matrix("output", outputs, q, h, output_width, category="mla.o")
    if parallel.tp > 1 and (not plan.o_groups or group_parallel):
        last = b.communicate("output_reduce", (last,), 2 * (parallel.tp - 1) / parallel.tp * q * h * 2,
                             "allreduce")
    return AttentionResult(ops=b.ops, last=last, cache_store_id=long_store or store,
                           index_store_id=index_store, topk_id=topk, candidate_id=candidate)
