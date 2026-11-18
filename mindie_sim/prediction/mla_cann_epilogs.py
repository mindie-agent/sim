"""CANN cache epilogs and the index producer/selection lowering.

[Role] Own the fused cache epilogs and the index-key path of latent attention.
[Invariants] A fused epilog replaces one producer write plus one cache read; it
  never removes a value that another consumer still reads.  The native indexer
  consumes checkpoint BF16 head weights directly, with no Cast wrapper.  A
  cross-step TopK reuse keeps every per-step KPool producer and skips only the
  selection, preserving arithmetic per produced cache row.
[Timeouts & Locks] O(views + index pools) arithmetic; no I/O or locks.
[Side-Effect Policy] Rewrites only the caller's builder operator list.
"""
from dataclasses import replace
import math

from mindie_sim.catalog.cache import completed_rows, index_row_bytes, index_selection_cost
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.ops import compute_dtype_for
from mindie_sim.schema.common import ModelingError, dtype_bytes

from .generic_latent_attention import (
    index_cache_dtype, index_cache_is_packed, owned_store_rows,
    sharded_index_visible_rows, semantic_index_selection,
)
from .implementations import select
from .indexer_representation import index_representation
from .mla_native import index_layer_norm

INDEXER_QUANT_CACHE = "aclnnIndexerQuantCache"
INDEXER_TOP_K_KIND = "aclnnQuantLightningIndexerV2"
KV_COMPRESS_EPILOG = "aclnnKvCompressEpilog"
KV_SCATTER_EPILOG = "aclnnScatterPaKvCache"
BLOCK_QUANT_KIND = "aclnnDynamicBlockQuant"
BLOCK_QUANT_TILE = 128
INDEXER_QUANT_CACHE_RECIPE = "indexer_quant_cache"
FP32_BYTES = int(dtype_bytes("fp32"))
FP16_BYTES = int(dtype_bytes("fp16"))
FP32_WEIGHT_DTYPE = "fp32"
INT32_BYTES = int(dtype_bytes("int32"))
ROPE_OPS = 6
CANDIDATE_BYTES = 8
MIN_LOG_TOPK = 2
CANDIDATE_BLOCK_OPS = 2
DEFAULT_BLOCK_SIZE = 128
ASCEND_ARCH_FAMILIES = frozenset({"A2", "A3", "A5"})
NATIVE_KEYPOOL_IMPLEMENTATION = "vllm_ascend_cann_keypool_caller"


def _policy(runtime):
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    policy = metadata.get("execution_policy")
    policy = dict(policy) if isinstance(policy, dict) else {}
    raw_eager = bool(policy.get("raw_eager", metadata.get("raw_eager", False)))
    automatic = bool(policy.get("automatic_optional_fusions", True))
    return metadata, policy, raw_eager, automatic


def native_fusion_active(runtime, machine=None):
    """True on a declared Ascend family unless eager/generic was requested."""
    if runtime is None:
        return False
    metadata, _, raw_eager, automatic = _policy(runtime)
    if raw_eager or not automatic:
        return False
    family = str(getattr(machine, "arch_family", "")
                 or metadata.get("arch_family") or "")
    return family.upper() in ASCEND_ARCH_FAMILIES


def raw_eager_requested(runtime):
    """True only for an explicit PyTorch-eager decomposition request."""
    return _policy(runtime)[2]


def native_indexer_active(runtime):
    """True when a CANN indexer implementation owns this graph."""
    if runtime is None:
        return False
    lowerings = dict(getattr(runtime, "kernels", {}).get("lowerings", {}) or {})
    return (select(lowerings, "pool_key_indexer") is not None
            or select(lowerings, "quant_lightning_indexer") is not None)


def kv_cache_epilog(op, fmt, region, *, native):
    """Name the grouped KV quantize+scatter epilog without re-deducting bytes.

    ``cache_representation.store`` already writes the grouped quantized row in
    one read and one write, so only the native epilog identity is attached.
    """
    if not native or not fmt.group_size:
        return op
    kind = KV_COMPRESS_EPILOG if fmt.dtype != op.dtype else KV_SCATTER_EPILOG
    width = int(op.local_shape[1]) if len(op.local_shape) > 1 else 0
    return replace(op, metadata=op.metadata + (
        ("native_kind", kind),
        ("cache_epilog", {
            "kind": kind, "region": region,
            "row_bytes": fmt.row_bytes(width) if width else 0,
            "quantized_payload_dtype": fmt.dtype,
            "scale_dtype": fmt.scale_dtype,
            "group_size": fmt.group_size,
            "single_read_single_write": True,
        }),
        ("native_quantize_and_scatter", True),
    ))


def block_quant_epilog(op, fmt, region, *, native, width):
    """Name the NoPE C8 block-quantized cache write without changing bytes."""
    if not native or fmt.dtype != "fp8_e4m3" or fmt.group_size != BLOCK_QUANT_TILE:
        return op
    row_bytes = fmt.row_bytes(width)
    return replace(op, metadata=op.metadata + (
        ("native_kind", BLOCK_QUANT_KIND),
        ("cache_epilog", {
            "kind": BLOCK_QUANT_KIND, "scatter": KV_SCATTER_EPILOG,
            "region": region, "tile": BLOCK_QUANT_TILE,
            "row_bytes": row_bytes,
            "row_components": {"fp8_payload": width, "scale": row_bytes - width},
            "group_size": fmt.group_size, "scale_dtype": fmt.scale_dtype,
        }),
        ("cache_row_bytes", row_bytes),
    ))


def index_row_quantization(builder, store_id, quantize_id, *, width, native):
    """Fuse index-key quantization and cache scatter into one epilog.

    The quantization producer already writes exactly one cache row per row
    (payload plus group scales).  Keeping it external makes the synthetic
    ``cache_store`` read that row back and write it again.  The fused epilog
    keeps the BF16 input read and the cache-row write, removes the intermediate
    pair, and preserves the total vector work.
    """
    if not native or store_id is None or quantize_id is None:
        return store_id
    by_id = {op.op_id: op for op in builder.ops}
    store = by_id.get(store_id)
    quantize = by_id.get(quantize_id)
    if store is None or quantize is None or store.family != "cache_store":
        return store_id
    if quantize.family != "quantize" or quantize_id not in store.deps:
        return store_id
    inputs = tuple(quantize.deps)
    extra = tuple(dep for dep in store.deps if dep != quantize_id)
    metadata = dict(quantize.metadata)
    fused = replace(
        store, deps=tuple(dict.fromkeys((*inputs, *extra))),
        mte2_bytes=quantize.mte2_bytes, mte3_bytes=store.mte3_bytes,
        hbm_bytes=quantize.mte2_bytes + store.mte3_bytes,
        vector_ops=quantize.vector_ops, dtype=quantize.dtype,
        recipe=INDEXER_QUANT_CACHE_RECIPE,
        metadata=store.metadata + (
            ("native_kind", INDEXER_QUANT_CACHE),
            ("fused_members", (quantize.op_id, store.op_id)),
            ("payload_dtype", metadata.get("output_format")),
            ("cache_epilog", {
                "kind": INDEXER_QUANT_CACHE,
                "payload_dtype": metadata.get("output_format"),
                "scale_dtype": metadata.get("scale_format"),
                "scale_shape": metadata.get("scale_shape"),
                "row_elements": width,
                "single_read_single_write": True,
            }),
            ("index_topk_kind", INDEXER_TOP_K_KIND),
        ))
    builder.ops = [op for op in builder.ops
                   if op.op_id not in {quantize_id, store_id}]
    return builder.add(fused)


def index_head_weights(builder, *, plan, model, quant, deps, pool_weights, tokens,
                       native):
    """Project indexer head weights and never wrap a native consumer in casts."""
    # The per-layer resolved projection format is the operand contract; the
    # model-level preset may still be W8A8 while indexer head weights stay BF16.
    weights_dtype = compute_dtype_for(builder.quant, "indexer.weights")[1]
    weights_input = deps
    if (not native and pool_weights is None and weights_dtype in {"fp32", "fp16", "bf16"}
            and weights_dtype != model.activation_dtype):
        weights_input = builder.vector(
            "index_weights_input_cast", (deps,), tokens * model.hidden_size, 1,
            family="cast", dtype=weights_dtype, shape=(tokens, model.hidden_size),
            read_bytes_override=tokens * model.hidden_size * dtype_bytes(model.activation_dtype),
            write_bytes_override=tokens * model.hidden_size * dtype_bytes(weights_dtype),
            metadata=(("input_format", model.activation_dtype),
                      ("output_format", weights_dtype)))
    # The KPool producer's fused [K, head weights] projection exposes a
    # strided column view. Its legacy consumer cast reads only the head slice.
    weights = pool_weights or builder.matrix(
        "index_weights", (weights_input,), tokens, plan.index_heads, model.hidden_size,
        category="indexer.weights")
    if native:
        # ``aclnnQuantLightningIndexerV2`` / ``aclnnPoolKeyIndexer`` consume the
        # checkpoint head-weight projection directly.  Any external cast would
        # be a launch the production graph does not execute.
        return weights
    if index_cache_is_packed(model):
        if weights_dtype != FP32_WEIGHT_DTYPE:
            return builder.vector(
                "index_weights_fp32", (weights,), tokens * plan.index_heads, 1,
                family="cast", dtype=FP32_WEIGHT_DTYPE,
                shape=(tokens, plan.index_heads),
                read_bytes_override=tokens * plan.index_heads * dtype_bytes(weights_dtype),
                write_bytes_override=tokens * plan.index_heads * FP32_BYTES,
                metadata=(("source_semantics", "packed_indexer_weights_float32"),))
        return weights
    if quant.index_dtype == "int8":
        return builder.vector(
            "index_weights_fp16", (weights,), tokens * plan.index_heads, 1,
            family="cast", dtype="fp16", shape=(tokens, plan.index_heads),
            read_bytes_override=tokens * plan.index_heads * dtype_bytes(weights_dtype),
            write_bytes_override=tokens * plan.index_heads * FP16_BYTES,
            metadata=(("source_semantics", "quant_indexer_weights_float16"),))
    if plan.index_kpool > 1:
        if weights_dtype != quant.index_dtype:
            weights = builder.vector(
                "index_weights_cast", (weights,), tokens * plan.index_heads, 1,
                family="cast", dtype=quant.index_dtype,
                shape=(tokens, plan.index_heads),
                read_bytes_override=tokens * plan.index_heads * dtype_bytes(weights_dtype),
                write_bytes_override=tokens * plan.index_heads * dtype_bytes(quant.index_dtype),
                metadata=(("input_format", weights_dtype),
                          ("output_format", quant.index_dtype))
                + ((("input_column_slice",
                     (plan.index_dim, plan.index_dim + plan.index_heads)),)
                   if pool_weights else ()))
        return builder.vector(
            "index_weights_scale", (weights,), tokens * plan.index_heads, 1,
            dtype=quant.index_dtype, shape=(tokens, plan.index_heads))
    return weights


def _store_rows(rows_by_view, views, parallel, block, *, contexts=None):
    return max(1, owned_store_rows(
        rows_by_view, views, parallel, block, contexts=contexts))


def lower_index_path(builder, *, plan, model, quant, parallel, views, deps, runtime,
                     padded_tokens, valid, cp, own_block, long_store,
                     compressed_latent, cache_ids, index_cache_ids, candidate_ids,
                     topk_ids, q_norm, stage, stage_of, index_rope, pool_binding,
                     pcp_prefill, dsa_prefill, hierarchy=None, reuse_selection=False,
                     shared_topk=None, fused_compressor=False):
    """Produce the index cache and (optionally) its TopK selection."""
    b = builder
    q = padded_tokens
    native_indexer = native_indexer_active(runtime)
    ratio = max(1, plan.compress_ratio)
    sharded_index = bool(dict(model.metadata).get("attention_execution", {}).get(
        "dcp_indexer_sharded")) and cp > 1
    idx_block = max(1, own_block // ratio)
    index_store, topk, candidate, pool_weights = None, None, None, None

    def owner(mapping, layer, name, volume):
        if layer not in mapping:
            raise ModelingError(f"producer_missing:{b.prefix}:{name}:{layer}")
        ready = mapping[layer]
        producer_stage = stage_of(layer)
        if producer_stage != stage:
            ready = b.communicate(name + "_transfer", (ready,), volume,
                                  "cache_transfer", "pp",
                                  source_stage=producer_stage, destination_stage=stage)
        return ready

    if plan.owns_index_k and plan.index_kpool > 1:
        from . import index_pool
        pool_module = pool_binding.module if pool_binding else index_pool
        producer_options = (pool_binding.options if pool_binding and
                            pool_binding.name == "vllm_ascend_cann_keypool_caller" else {})
        index_store, pool_weights = pool_module.producer(
            b, source=deps, views=views, plan=plan, model=model, quant=quant,
            padded_tokens=q, **producer_options)
    elif plan.owns_index_k:
        index_source = long_store if plan.index_input == "compressed" else deps
        if (plan.index_input == "compressed"
                and cache_format(model, "compressed_kv").group_size):
            if compressed_latent is None:
                raise ModelingError("packed_compressed_indexer_requires_pre_rope_latent_producer")
            index_source = compressed_latent
        index_rows_by_view = [
            completed_rows(view.context + view.q_len, ratio)
            - completed_rows(view.context, ratio)
            for view in views
        ]
        index_rows = sum(index_rows_by_view)
        input_rows = index_rows if plan.index_input == "compressed" else valid
        input_width = plan.head_dim if plan.index_input == "compressed" else model.hidden_size
        if plan.compress_ratio > 1 and plan.index_input == "hidden":
            from .compression import compressor
            normalized = compressor(
                b, name="index_compress", source=deps, tokens=valid, padded_tokens=q,
                valid_rows=index_rows, width=plan.index_dim, plan=plan, model=model,
                fused=fused_compressor)
        else:
            projected = b.matrix("index_k", (index_source,), input_rows, plan.index_dim,
                                 input_width, category="indexer_fp8")
            normalized = index_layer_norm(
                b, source=projected, rows=index_rows, width=plan.index_dim,
                native=native_indexer)
            if index_rope:
                normalized = b.vector(
                    "index_rotary", (normalized,), index_rows * index_rope, ROPE_OPS,
                    family="rotary", shape=(index_rows, plan.index_dim),
                    read_bytes_override=index_rows * (
                        plan.index_dim * dtype_bytes(model.activation_dtype)
                        + index_rope * CANDIDATE_BYTES),
                    write_bytes_override=index_rows * plan.index_dim
                    * dtype_bytes(model.activation_dtype))
        normalized = index_representation(
            b, "index_key", normalized, index_rows, plan.index_dim,
            quant=quant, model=model, rotate_basis=plan.compress_ratio > 1)
        last = b.ops[-1]
        quantize_id = (last.op_id if last.family == "quantize"
                       and last.op_id == b.prefix + ".index_key_quantize" else None)
        index_starts = [completed_rows(view.context, ratio) for view in views]
        index_store_rows = (
            _store_rows(index_rows_by_view, views, parallel, idx_block,
                        contexts=index_starts)
            if sharded_index and index_rows else index_rows
        )
        index_store = b.store("index_store", (normalized,), index_store_rows,
                              index_row_bytes(model, plan.index_dim), quant.index_dtype)
        index_store = index_row_quantization(
            b, index_store, quantize_id, width=plan.index_dim, native=native_indexer)
    if pcp_prefill and plan.indexer and plan.owns_index_k and index_store is not None:
        # One stored index vector is index_dim wide, not index_heads copies of it.
        if plan.index_kpool > 1:
            rows = sum((view.context + (view.global_q_len or view.q_len)) // plan.index_kpool
                       - view.context // plan.index_kpool
                       for view in views if view.phase == "prefill")
        else:
            rows = sum(completed_rows(
                view.context + (view.global_q_len or view.q_len), ratio)
                - completed_rows(view.context, ratio)
                for view in views if view.phase == "prefill")
        index_store = b.communicate(
            "pcp_index_gather", (index_store,),
            rows * index_row_bytes(model, plan.index_dim)
            * (parallel.pcp - 1) / parallel.pcp,
            "allgather", "pcp")
    if reuse_selection:
        if shared_topk is None or plan.index not in shared_topk:
            raise ModelingError("mtp_shared_topk_selection_missing")
        return index_store, shared_topk[plan.index], None, pool_weights
    if plan.indexer or plan.owns_topk or plan.index_query_per_layer:
        if index_store is None:
            if plan.scope == "draft" and plan.index_k_owner_layer_id < 0:
                index_store = deps
            else:
                index_store = owner(
                    index_cache_ids, plan.index_k_owner_layer_id, "index_cache",
                    sum(index_selection_cost(model, plan, view.kv_len).index_read_bytes
                        for view in views))
        q_index = b.matrix("index_q", (q_norm,), q,
                           plan.index_heads * plan.index_dim,
                           plan.q_lora_rank, category="indexer_fp8")
        if plan.index_kpool == 1 and index_rope:
            q_index = b.vector(
                "index_query_rotary", (q_index,),
                valid * plan.index_heads * index_rope, ROPE_OPS,
                family="rotary", shape=(valid, plan.index_heads, plan.index_dim))
        q_index = index_representation(
            b, "index_query", q_index,
            (q if plan.index_kpool > 1 else valid) * plan.index_heads, plan.index_dim,
            quant=quant, model=model, rotate_basis=plan.compress_ratio > 1)
        weights = index_head_weights(
            b, plan=plan, model=model, quant=quant, deps=deps,
            pool_weights=pool_weights, tokens=q, native=native_indexer)
        selection_deps = [q_index, weights, index_store]
        if plan.candidate_filter:
            if plan.scope == "draft" and plan.candidate_source_layer_id < 0:
                selection_deps.append(deps)
            else:
                selection_deps.append(owner(
                    candidate_ids, plan.candidate_source_layer_id, "candidate",
                    valid * plan.candidate_topk_blocks * INT32_BYTES))
        shard = (parallel.tp if dsa_prefill
                 else cp if sharded_index else 1)
        score_rows = (
            sharded_index_visible_rows(views, parallel, idx_block, ratio)
            if sharded_index else None
        )
        if plan.index_kpool > 1:
            from . import index_pool
            if pool_binding and pool_binding.name == NATIVE_KEYPOOL_IMPLEMENTATION:
                topk = pool_binding.module.selection(
                    b, query=q_index, weights=weights, cache=index_store,
                    views=views, plan=plan, quant=quant, padded_tokens=q,
                    l2_bytes=getattr(b, "l2_bytes", None), runtime=runtime,
                    hierarchy=hierarchy, **pool_binding.options)
            else:
                topk = index_pool.selection(
                    b, query=q_index, weights=weights, cache=index_store,
                    views=views, plan=plan, quant=quant, padded_tokens=q,
                    l2_bytes=getattr(b, "l2_bytes", None), runtime=runtime,
                    **(pool_binding.options if pool_binding else {}))
        else:
            lowerings = (runtime.kernels.get("lowerings", {}) if runtime else {})
            quant_indexer = (select(lowerings, "quant_lightning_indexer")
                             if runtime else None)
            bf16_indexer = select(lowerings, "lightning_indexer") \
                if runtime and not quant_indexer else None
            if quant_indexer:
                topk = quant_indexer.module.declare(
                    b, deps=selection_deps, views=views, plan=plan,
                    quant=quant, parallel=parallel, **quant_indexer.options)
                # This explicit arch22 source owns its per-query key/scale
                # ledger. Shared capacity does not establish batch read reuse.
            elif (bf16_indexer is not None and len(views) == 1
                  and parallel.cp == 1 and not plan.candidate_filter):
                from . import family_lightning_indexer as li
                if bf16_indexer.name != li.IMPLEMENTATION:
                    raise ModelingError("mla_lightning_indexer_selected_not_bf16")
                view = views[0]
                topk = b.add(li.make_op(
                    query_rows=view.q_len, context=view.context,
                    op_id=b.prefix + ".index_lightning",
                    deps=tuple(selection_deps), stage=stage, layer=plan.index,
                    requests=(view.request_id,), phase=b.common["phase"],
                    heads=plan.index_heads, head_dim=plan.index_dim,
                    topk=plan.index_topk, compression=ratio,
                    page_size=int(getattr(runtime, "kv_block_size", 0)
                                  or DEFAULT_BLOCK_SIZE),
                    kv_heads=1, query_dtype=model.activation_dtype,
                    cache_dtype=index_cache_dtype(model), weight_dtype="fp32",
                    query_lengths=(view.q_len,), context_lengths=(view.context,),
                    binding_revision=str(getattr(runtime, "vllm_version", "unknown")),
                    runtime_version=str(getattr(runtime, "vllm_version", "unknown")),
                ))
            else:
                topk = semantic_index_selection(
                    b, deps=selection_deps, views=views, plan=plan, model=model,
                    quant=quant, parallel=parallel, shard=shard, prefix=b.prefix,
                    padded_tokens=q, runtime=runtime, score_rows=score_rows,
                    l2_bytes=getattr(b, "l2_bytes", None))
        if shard > 1 and not dsa_prefill:
            topk = b.work(
                "candidate_pack", (topk,), (valid, plan.index_topk),
                read=valid * plan.index_topk * CANDIDATE_BYTES,
                write=valid * plan.index_topk * CANDIDATE_BYTES,
                family="elementwise")
            gathered = b.communicate(
                "candidate_gather", (topk,),
                valid * plan.index_topk * CANDIDATE_BYTES * (cp - 1),
                "allgather", "cp")
            topk = b.vector("candidate_merge", (gathered,),
                            valid * plan.index_topk * cp,
                            math.log2(max(MIN_LOG_TOPK, plan.index_topk)),
                            dtype="fp32", family="topk")
        if plan.owns_candidate_blocks:
            candidate = b.vector("candidate_blocks", (topk,),
                                 valid * plan.index_topk, CANDIDATE_BLOCK_OPS,
                                 dtype="int32")
    elif plan.topk_owner_layer_id >= 0:
        topk = owner(topk_ids, plan.topk_owner_layer_id, "topk",
                     valid * plan.index_topk * INT32_BYTES)
    return index_store, topk, candidate, pool_weights
