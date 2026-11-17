"""Latent, compressed and indexed attention from explicit layer geometry.

[Role] Compose projections, cache producers, sparse selection and attention.
[Invariants] Cache ownership and selection reuse are dependencies. Compression
  counts complete groups per request. Logical pairs and tiled memory reads are
  different quantities. No model/vendor identity selects an execution path.
[Timeouts & Locks] O(requests + query tiles) arithmetic, no observation reads.
[Side-Effect Policy] Constructs operators only.
"""
import math
from dataclasses import replace

from mindie_sim.catalog.attention_common import (
    AttentionResult, _group_plan, active_dcp_degree, dcp_partial_merge_ops,
)
from mindie_sim.catalog.cache import completed_rows, index_row_bytes, index_selection_cost
from mindie_sim.schema.common import dtype_bytes, ModelingError
from .batch_prefix_attention import (
    annotate_sparse_prefix_unknown, apply_attention_prefix_reads,
)
from .implementations import select
from .lowering import Builder
from .compression import compressor
from .indexer_representation import index_representation
from .mla_native import batched_matrix_metadata, index_layer_norm
from .memory_traffic import (AttentionAccessSpec, AttentionSchedule,
                             default_hierarchy, plan_attention_traffic)
from .projections import mla_input_projections
from . import mla_head_padding
from .cache_representation import store as representation_store
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.ops import compute_dtype_for
from .generic_latent_attention import (
    attention_tensor_view, attention_view_record, dsa_cp_prefill_parts,
    kv_sweeps, owned_store_rows, selected_pairs, semantic_index_selection,
    sharded_index_visible_rows, sparse_over_parts, view_causal_pairs,
    visible_parts, window_cache_read,
)

# Historical private names remain importable for source-geometry probes.
_semantic_index_selection = semantic_index_selection
_sparse_over_parts = sparse_over_parts
_kv_sweeps = kv_sweeps
_selected_pairs = selected_pairs
_visible_parts = visible_parts
_view_causal_pairs = view_causal_pairs
_dsa_cp_prefill_parts = dsa_cp_prefill_parts
_attention_tensor_view = attention_tensor_view

ZERO = 0
ONE = 1
TWO = 2
FOUR = 4
FIVE = 5
ZERO_FLOAT = 0.0
CANDIDATE_BYTES = 8
MIN_LOG_TOPK = 2
DEFAULT_INTERLEAVE = 128
DEFAULT_BLOCK_SIZE = 128
DEFAULT_QUERY_TILE = 128
MAX_GENERIC_OWNERS = 32
ROPE_OPS = 6


def _store_rows(rows_by_view, views, parallel, block, *, contexts=None):
    return max(ONE, owned_store_rows(
        rows_by_view, views, parallel, block, contexts=contexts))


def _memory_hierarchy(machine):
    """Declared device hierarchy for traffic planning, or the frontend default."""
    if machine is not None and getattr(machine, "dies", ()):
        hierarchy = getattr(machine.dies[ZERO].physical_mechanisms,
                            "memory_hierarchy", None)
        if hierarchy is not None and hierarchy.l2_capacity_bytes > ZERO:
            return hierarchy
    return default_hierarchy()


def mla(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
        padded_tokens, cache_ids, topk_ids, index_cache_ids, candidate_ids,
        stage_of, cp_interleave_size=DEFAULT_INTERLEAVE, runtime=None, policy=None,
        machine=None):
    b = Builder(prefix, plan, stage, views, model, mode)
    hierarchy = _memory_hierarchy(machine)
    b.l2_bytes = hierarchy.effective_l2_bytes
    from . import latent_dcp
    from mindie_sim.catalog.paired_sfa import execution
    gathered_dcp = latent_dcp.enabled(model, parallel, views, runtime)
    cp = max((active_dcp_degree(parallel, view.phase) for view in views), default=ONE)
    execution_flags = execution(model)
    paired = dict(model.metadata).get("cache_allocator") == "paired_sfa"
    own_block = (int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE)
                 if paired else int(cp_interleave_size))
    index_rope = int(execution_flags.get("index_rope_dim", plan.qk_rope_head_dim))
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
    dsa_prefill = (
        parallel.cp_plan == "dsa_cp" and parallel.tp > ONE
        and any(view.phase == "prefill" for view in views)
    )
    width = plan.qk_nope_head_dim + plan.qk_rope_head_dim
    latent = plan.mla_variant != "latent_mqa"
    query_tile = policy.query_tile if policy else DEFAULT_QUERY_TILE
    kv_width = plan.kv_lora_rank + (plan.qk_rope_head_dim if latent else 0)
    if plan.q_lora_rank:
        projected = mla_input_projections(
            b, plan=plan, tokens=q, hidden=h, kv_width=kv_width, deps=deps,
            packed=prolog is None)
        q_a, kv_a = projected["q_a"], projected["kv_a"]
        q_norm = b.vector("q_norm", (q_a,), q * plan.q_lora_rank, FIVE, family="rmsnorm")
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
    kv = b.vector("kv_norm", (kv_a,), q * plan.kv_lora_rank, FIVE, family="rmsnorm")
    if partial_rotary:
        q_ready = partial_rotary.module.declare(
            b, name="q_rotary", deps=(q_ready,), tokens=q, heads=heads,
            head_dim=width, rotary_width=plan.qk_rope_head_dim)
        rotary = partial_rotary.module.declare(
            b, name="kv_rotary", deps=(kv,), tokens=q, heads=ONE,
            head_dim=plan.kv_lora_rank, rotary_width=plan.qk_rope_head_dim)
    else:
        rotary = b.vector("rotary", (kv, q_ready),
                          q * (heads + ONE) * plan.qk_rope_head_dim, ROPE_OPS)
    valid = sum(view.q_len for view in views)
    pcp_prefill = parallel.pcp > ONE and any(view.phase == "prefill" for view in views)
    store_rows = _store_rows([view.q_len for view in views], views, parallel, own_block)
    packed_store = (prolog.module.cache_store(
        b, deps=(kv, rotary), rows=store_rows, plan=plan, **prolog.options)
        if prolog else None)
    store = packed_store or (representation_store(
        b, "kv_store", (rotary,), store_rows, plan.cache_row_elements,
        "window" if plan.cache_kind == "window_compressed" else "mla_kv"))
    # This lowering writes the local KV row. Shared long-KV consumers still do.
    produced, owns_kv = store, store is not None
    if pcp_prefill and owns_kv:
        rows = sum(view.global_q_len or view.q_len for view in views if view.phase == "prefill")
        if packed_store is not None:
            from .mla_prolog_packed import view as packed_kv_view
            kv_row = packed_kv_view(plan.kv_lora_rank, plan.qk_rope_head_dim)["row_bytes"]
        elif plan.cache_kind == "window_compressed":
            kv_row = cache_format(model, "window").row_bytes(plan.cache_row_elements)
        else:
            kv_row = plan.cache_row_elements * dtype_bytes(quant.kv_dtype)
        store = b.communicate(
            "pcp_kv_gather", (produced,),
            rows * kv_row * (parallel.pcp - ONE) / parallel.pcp,
            "allgather", "pcp")
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
            tokens=q, members=(q_a, q_norm, q_b, kv_a, kv, rotary, produced, q_ready),
            q_a=q_a, q_b=q_b, kv_a=kv_a, q_absorb=q_ready, **prolog.options)
    if gathered_dcp:
        q_ready, rotary = latent_dcp.query_gather(
            b, q_ready, rotary, q, heads, plan.kv_lora_rank + plan.qk_rope_head_dim, cp)
        heads *= cp
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

    ratio = max(ONE, plan.compress_ratio)
    if plan.cache_kind == "window_compressed" and plan.compress_ratio:
        compressed_rows = [
            completed_rows(view.context + view.q_len, ratio)
            - completed_rows(view.context, ratio) for view in views
        ]
        if plan.owns_long_kv:
            new_rows = sum(compressed_rows)
            compressed_latent, rotated = compressor(
                b, name="compress", source=deps, tokens=valid, padded_tokens=q,
                valid_rows=new_rows, width=plan.head_dim, plan=plan, model=model,
                return_latent=True)
            compressed_starts = [completed_rows(view.context, ratio) for view in views]
            long_store = representation_store(
                b, "compressed_store", (rotated,),
                _store_rows(compressed_rows, views, parallel, own_block,
                            contexts=compressed_starts) if new_rows else ZERO,
                plan.cache_row_elements, "compressed_kv")
            if pcp_prefill:
                global_comp_rows = sum(
                    completed_rows(view.context + (view.global_q_len or view.q_len), ratio)
                    - completed_rows(view.context, ratio)
                    for view in views if view.phase == "prefill")
                if global_comp_rows > ZERO:
                    comp_row_bytes = cache_format(
                        model, "compressed_kv").row_bytes(plan.cache_row_elements)
                    long_store = b.communicate(
                        "pcp_compressed_kv_gather", (long_store,),
                        global_comp_rows * comp_row_bytes
                        * (parallel.pcp - ONE) / parallel.pcp,
                        "allgather", "pcp")
        else:
            rows = sum(completed_rows(view.kv_len, ratio) for view in views)
            long_store = owner(cache_ids, plan.kv_owner_layer_id, "compressed_cache",
                               rows * cache_format(model, "compressed_kv").row_bytes(
                                   plan.cache_row_elements) / max(ONE, cp))

    if plan.index_kpool > ONE and parallel.cp != ONE:
        raise ModelingError("pooled_indexer_context_partition_not_declared")
    pool_binding = (select(runtime.kernels.get("lowerings", {}), "pool_key_indexer")
                    if runtime and plan.index_kpool > ONE else None)
    index_rows_by_view: list[int] = []
    ratio = max(ONE, plan.compress_ratio)
    sharded_index = bool(execution_flags.get("dcp_indexer_sharded")) and cp > ONE
    idx_block = max(ONE, own_block // ratio)
    if plan.owns_index_k and plan.index_kpool > ONE:
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
            # Quantized, post-RoPE cache bytes cannot masquerade as the
            # compressor's pre-RoPE BF16 latent for the index projection.
            index_source = compressed_latent
        index_rows_by_view = [
            completed_rows(view.context + view.q_len, ratio)
            - completed_rows(view.context, ratio)
            for view in views
        ]
        index_rows = sum(index_rows_by_view)
        input_rows = index_rows if plan.index_input == "compressed" else valid
        input_width = plan.head_dim if plan.index_input == "compressed" else h
        if plan.compress_ratio > ONE and plan.index_input == "hidden":
            normalized = compressor(
                b, name="index_compress", source=deps, tokens=valid, padded_tokens=q,
                valid_rows=index_rows, width=plan.index_dim, plan=plan, model=model)
        else:
            projected = b.matrix("index_k", (index_source,), input_rows, plan.index_dim,
                                 input_width, category="indexer_fp8")
            # The ordinary latent indexer uses mean/variance LayerNorm. The
            # compressor above has RMS normalization and its own parameters.
            normalized = index_layer_norm(
                b, source=projected, rows=index_rows, width=plan.index_dim)
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
            quant=quant, model=model, rotate_basis=plan.compress_ratio > ONE)
        index_starts = [completed_rows(view.context, ratio) for view in views]
        index_store_rows = (
            _store_rows(index_rows_by_view, views, parallel, idx_block,
                        contexts=index_starts)
            if sharded_index and index_rows else index_rows
        )
        index_store = b.store("index_store", (normalized,), index_store_rows,
                              index_row_bytes(model, plan.index_dim), quant.index_dtype)
    if pcp_prefill and plan.indexer and plan.owns_index_k and index_store is not None:
        # One stored index vector is index_dim wide, not index_heads copies of it.
        if plan.index_kpool > ONE:
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
            * (parallel.pcp - ONE) / parallel.pcp,
            "allgather", "pcp")
    if plan.indexer or plan.owns_topk or plan.index_query_per_layer:
        if index_store is None:
            if plan.scope == "draft" and plan.index_k_owner_layer_id < ZERO:
                index_store = deps
            else:
                index_store = owner(
                    index_cache_ids, plan.index_k_owner_layer_id, "index_cache",
                    sum(index_selection_cost(model, plan, view.kv_len).index_read_bytes
                        for view in views))
        q_index = b.matrix("index_q", (q_norm,), q, plan.index_heads * plan.index_dim,
                           plan.q_lora_rank, category="indexer_fp8")
        if plan.index_kpool == ONE and index_rope:
            q_index = b.vector(
                "index_query_rotary", (q_index,),
                valid * plan.index_heads * index_rope, ROPE_OPS,
                family="rotary", shape=(valid, plan.index_heads, plan.index_dim))
        q_index = index_representation(
            b, "index_query", q_index,
            (q if plan.index_kpool > ONE else valid) * plan.index_heads, plan.index_dim,
            quant=quant, model=model, rotate_basis=plan.compress_ratio > ONE)
        weights_input = deps
        weights_dtype = compute_dtype_for(b.quant, "indexer.weights")[ONE]
        # A declared FP32 projection consumes an FP32 input, not a BF16
        # allocation relabelled by the matrix's compute identity.
        if (pool_weights is None and weights_dtype in {"fp32", "fp16", "bf16"}
                and weights_dtype != model.activation_dtype):
            weights_input = b.vector(
                "index_weights_input_cast", (deps,), q * h, ONE,
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
                "index_weights_fp16", (weights,), q * plan.index_heads, ONE,
                family="cast", dtype="fp16", shape=(q, plan.index_heads),
                read_bytes_override=q * plan.index_heads * dtype_bytes(weights_dtype),
                write_bytes_override=q * plan.index_heads * TWO,
                metadata=(("source_semantics", "quant_indexer_weights_float16"),))
        elif plan.index_kpool > ONE:
            if weights_dtype != quant.index_dtype:
                weights = b.vector(
                    "index_weights_cast", (weights,), q * plan.index_heads, ONE,
                    family="cast", dtype=quant.index_dtype, shape=(q, plan.index_heads),
                    read_bytes_override=q * plan.index_heads * dtype_bytes(weights_dtype),
                    write_bytes_override=q * plan.index_heads * dtype_bytes(quant.index_dtype),
                    metadata=(("input_format", weights_dtype),
                              ("output_format", quant.index_dtype))
                    + ((("input_column_slice",
                         (plan.index_dim, plan.index_dim + plan.index_heads)),)
                       if pool_weights else ()))
            weights = b.vector("index_weights_scale", (weights,), q * plan.index_heads, ONE,
                               dtype=quant.index_dtype, shape=(q, plan.index_heads))
        selection_deps = [q_index, weights, index_store]
        if plan.candidate_filter:
            if plan.scope == "draft" and plan.candidate_source_layer_id < ZERO:
                selection_deps.append(deps)
            else:
                selection_deps.append(owner(
                    candidate_ids, plan.candidate_source_layer_id, "candidate",
                    valid * plan.candidate_topk_blocks * FOUR))
        shard = (parallel.tp if dsa_prefill
                 else cp if sharded_index else ONE)
        score_rows = (
            sharded_index_visible_rows(views, parallel, idx_block, ratio)
            if sharded_index else None
        )
        if plan.index_kpool > ONE:
            from . import index_pool
            pool_module = pool_binding.module if pool_binding else index_pool
            topk = pool_module.selection(
                b, query=q_index, weights=weights, cache=index_store,
                views=views, plan=plan, quant=quant, padded_tokens=q,
                l2_bytes=getattr(b, "l2_bytes", None),
                **(pool_binding.options if pool_binding else {}))
        else:
            quant_indexer = (select(runtime.kernels.get("lowerings", {}),
                                    "quant_lightning_indexer") if runtime else None)
            bf16_indexer = (select(runtime.kernels.get("lowerings", {}), "lightning_indexer")
                            if runtime and not quant_indexer else None)
            if quant_indexer:
                topk = quant_indexer.module.declare(
                    b, deps=selection_deps, views=views, plan=plan,
                    quant=quant, parallel=parallel, **quant_indexer.options)
                if (runtime is not None
                        and runtime.prefix_cache_mode == "shared"
                        and b.ops and b.ops[-ONE].op_id == topk):
                    from .batch_prefix_indexer import apply_indexer_prefix_reads
                    ratio = max(ONE, int(plan.compress_ratio or ONE))
                    row_b = float(index_row_bytes(b.model, plan.index_dim))
                    logical = tuple(
                        max(ZERO, int(view.q_len) * min(
                            max(ZERO, int(view.kv_len) // ratio),
                            int(plan.index_topk))) for view in views)
                    shared = tuple(
                        min(rows, max(ZERO, int(view.context) // ratio))
                        for rows, view in zip(logical, views))
                    b.ops[-ONE] = apply_indexer_prefix_reads(
                        b.ops[-ONE], views=views, rows_by_view=logical,
                        shared_rows_by_view=shared, row_bytes=row_b,
                        layer_index=plan.index, runtime=runtime,
                        kernel_reuse="batch_tile", allow_recipe_rebind=False)
            elif (bf16_indexer is not None and len(views) == ONE
                  and parallel.cp == ONE and not plan.candidate_filter):
                from . import family_lightning_indexer as li
                if bf16_indexer.name != li.IMPLEMENTATION:
                    raise ModelingError("mla_lightning_indexer_selected_not_bf16")
                view = views[ZERO]
                li_op = li.make_op(
                    query_rows=view.q_len, context=view.context,
                    op_id=prefix + ".index_lightning",
                    deps=tuple(selection_deps), stage=stage, layer=plan.index,
                    requests=(view.request_id,), phase=mode,
                    heads=plan.index_heads, head_dim=plan.index_dim,
                    topk=plan.index_topk, compression=ratio,
                    page_size=int(getattr(runtime, "kv_block_size", ZERO)
                                  or DEFAULT_BLOCK_SIZE),
                    kv_heads=ONE, query_dtype=model.activation_dtype,
                    cache_dtype=quant.kv_dtype, weight_dtype="fp32",
                    query_lengths=(view.q_len,), context_lengths=(view.context,),
                    binding_revision=str(getattr(runtime, "vllm_version", "unknown")),
                    runtime_version=str(getattr(runtime, "vllm_version", "unknown")),
                )
                topk = b.add(li_op)
            else:
                topk = semantic_index_selection(
                    b, deps=selection_deps, views=views, plan=plan, model=model,
                    quant=quant, parallel=parallel, shard=shard, prefix=prefix,
                    padded_tokens=q, runtime=runtime, score_rows=score_rows,
                    l2_bytes=getattr(b, "l2_bytes", None))
        if shard > ONE and not dsa_prefill:
            topk = b.work(
                "candidate_pack", (topk,), (valid, plan.index_topk),
                read=valid * plan.index_topk * CANDIDATE_BYTES,
                write=valid * plan.index_topk * CANDIDATE_BYTES,
                family="elementwise")
            gathered = b.communicate(
                "candidate_gather", (topk,),
                valid * plan.index_topk * CANDIDATE_BYTES * (cp - ONE),
                "allgather", "cp")
            topk = b.vector("candidate_merge", (gathered,),
                            valid * plan.index_topk * cp,
                            math.log2(max(MIN_LOG_TOPK, plan.index_topk)),
                            dtype="fp32", family="topk")
        if plan.owns_candidate_blocks:
            candidate = b.vector("candidate_blocks", (topk,),
                                 valid * plan.index_topk, TWO, dtype="int32")
    elif plan.topk_owner_layer_id >= ZERO:
        topk = owner(topk_ids, plan.topk_owner_layer_id, "topk",
                     valid * plan.index_topk * FOUR)

    if head_contract:
        q_ready, rotary = mla_head_padding.query_inputs(
            b, declaration=head_contract, tokens=q, query=q_ready, rotary=rotary, plan=plan)
    sparse_offload = (select(runtime.kernels.get("lowerings", {}), "sparse_kv_offload")
                      if runtime else None)
    offload_budget = dict(model.metadata).get("sparse_kv_offload") if sparse_offload else None
    offload_info = None
    if offload_budget and plan.index in offload_budget.get("offload_layers", ()):
        if (any(view.phase == "prefill" for view in views)
                and offload_budget["options"].get("layerwise_kv_offload")):
            sparse_offload.module.lower_prefill_offload(
                b, plan=plan, views=views, store_id=store, budget=offload_budget)
        elif any(view.phase == "decode" for view in views):
            store, topk, offload_info = sparse_offload.module.lower_decode_offload(
                b, plan=plan, views=views, store_id=store, topk_id=topk, budget=offload_budget)
            if plan.cache_kind == "dense":
                long_store = store
    outputs = []
    metadata_ready = None
    metadata_source = (select(runtime.kernels.get("lowerings", {}), "sparse_mla_metadata")
                       if runtime else None)
    if metadata_source:
        metadata_ready = metadata_source.module.lower(
            b, views=views, plan=plan, parallel=parallel, quant=quant,
            deps=(topk or deps,), padded_tokens=q, stage_of=stage_of,
            **metadata_source.options)
    latent_dim = plan.kv_lora_rank if latent else plan.head_dim
    qk_dim = latent_dim + plan.qk_rope_head_dim if latent else width
    if selected:
        sel_out = selected.module.lower_attention(
            b, views=views, plan=plan, heads=heads, runtime=runtime,
            query=q_ready, rotary=rotary, store=store,
            **({"head_contract": head_contract, "padded_tokens": q} if paged else {}),
            **selected.options)
        if cp > ONE and not gathered_dcp:
            tot_rows = sum(view.kv_len for view in views)
            merged, sel_out = dcp_partial_merge_ops(
                prefix=prefix, views=views, cp=cp, local_heads=heads,
                v_dim=latent_dim, stage=stage, layer=plan.index, mode=mode,
                deps=(sel_out,), total_pairs=ZERO_FLOAT, local_pairs=ZERO_FLOAT,
                total_rows=tot_rows, local_rows=math.ceil(tot_rows / cp),
                interleave=cp_interleave_size)
            b.ops.extend(merged)
        outputs.append(sel_out)
    block_size = max(ONE, int(getattr(runtime, "kv_block_size", DEFAULT_BLOCK_SIZE)
                              or DEFAULT_BLOCK_SIZE))
    phase_groups: list[tuple[int, list[tuple[int, object]]]] = []
    grp_map: dict[int, list[tuple[int, object]]] = {}
    for index, view in enumerate(() if selected else views):
        eff_cp = active_dcp_degree(parallel, view.phase)
        if eff_cp not in grp_map:
            bucket: list[tuple[int, object]] = []
            grp_map[eff_cp] = bucket
            phase_groups.append((eff_cp, bucket))
        grp_map[eff_cp].append((index, view))
    multi_group = len(phase_groups) > ONE
    for grp_idx, (eff_cp, indexed_views) in enumerate(phase_groups):
        grp_views = tuple(view for _, view in indexed_views)
        grp_prefix = f"{prefix}.phase{grp_idx}" if multi_group else prefix
        grp_dsa = dsa_prefill and any(view.phase == "prefill" for view in grp_views)
        rank_pairs = [ZERO_FLOAT] * eff_cp
        rank_sweeps = [ZERO] * eff_cp
        rank_rows = [ZERO] * eff_cp
        view_records: list[dict] = []
        for orig_idx, view in indexed_views:
            rec = attention_view_record(
                view, plan=plan, model=model, quant=quant, query_tile=query_tile,
                eff_cp=eff_cp, block=block_size, interleave=cp_interleave_size,
                tp=parallel.tp, dsa=grp_dsa, topk_present=bool(topk),
                heads=heads, latent_dim=latent_dim)
            for rank in range(eff_cp):
                rank_pairs[rank] += rec["v_pairs"][rank]
                rank_sweeps[rank] += rec["v_sweeps"][rank]
                rank_rows[rank] += rec["v_rows"][rank]
            view_records.append({"orig_idx": orig_idx, **rec})
        worst_pairs_r = max(range(eff_cp),
                            key=lambda rank: (rank_pairs[rank], rank_sweeps[rank]))
        worst_sweeps_r = max(range(eff_cp),
                             key=lambda rank: (rank_sweeps[rank], rank_pairs[rank]))
        grp_outputs: list[str] = []
        sweeps_by_view: list[int] = []
        shared_by_view: list[int] = []
        total_pairs_sum = ZERO_FLOAT
        total_rows_sum = ZERO
        kv_hbm_total = ZERO_FLOAT
        kv_mte2_total = ZERO_FLOAT
        kv_components: dict[str, tuple[float, float]] = {}
        kv_row_b = plan.cache_row_elements * dtype_bytes(quant.kv_dtype)
        for rec in view_records:
            orig_idx = rec["orig_idx"]
            view = rec["view"]
            query, eff_q = rec["query"], rec["eff_q"]
            parts = rec["parts"]
            pairs, sweeps = rec["pairs"], rec["sweeps"]
            local_pairs = rec["v_pairs"][worst_pairs_r]
            local_sweeps = rec["v_sweeps"][worst_sweeps_r]
            local_rows = rec["v_rows"][worst_pairs_r]
            sweeps_by_view.append(local_sweeps)
            shared_by_view.append(min(local_sweeps, rec["v_shared"][worst_sweeps_r]))
            total_pairs_sum += pairs
            total_rows_sum += view.kv_len
            attn_heads = (plan.num_heads
                          if grp_dsa and view.phase == "prefill" else heads)
            q_read = eff_q * attn_heads * qk_dim * dtype_bytes(model.activation_dtype)
            out_write = eff_q * attn_heads * latent_dim * dtype_bytes(model.activation_dtype)
            local_cache_read = window_cache_read(
                plan, view, parts, sweeps, eff_cp, query_tile,
                plan.cache_row_elements, model)
            traffic = plan_attention_traffic(
                AttentionAccessSpec(
                    q_lengths=(int(eff_q),),
                    context_lengths=(int(max(ZERO, view.kv_len - eff_q)),),
                    unique_rows=int(max(ZERO, local_rows)),
                    refill_rows=int(max(ZERO, local_sweeps)),
                    selected_rows=int(max(ZERO, local_sweeps)) if topk else ZERO,
                    row_bytes=float(kv_row_b),
                    phase=mode,
                    topk=int(plan.index_topk) if topk else ZERO,
                    representation=("absorbed_mla" if latent else "gqa"),
                    cache_identity=str(getattr(view, "prefix_key", "") or ""),
                ),
                hierarchy,
                AttentionSchedule(query_tile=query_tile),
            )
            if local_cache_read is None:
                local_cache_read = traffic.hbm_read_bytes
            elif local_sweeps > ZERO:
                traffic = plan_attention_traffic(
                    AttentionAccessSpec(
                        q_lengths=(int(eff_q),),
                        context_lengths=(int(max(ZERO, view.kv_len - eff_q)),),
                        unique_rows=int(max(ZERO, local_rows)),
                        refill_rows=int(max(ZERO, local_sweeps)),
                        selected_rows=(int(max(ZERO, local_sweeps))
                                       if topk else ZERO),
                        row_bytes=float(local_cache_read) / float(local_sweeps),
                        phase=mode,
                        topk=int(plan.index_topk) if topk else ZERO,
                        cache_identity=str(getattr(view, "prefix_key", "") or ""),
                    ),
                    hierarchy,
                    AttentionSchedule(query_tile=query_tile),
                )
                local_cache_read = traffic.hbm_read_bytes
            vis_keys = (
                min(local_rows, plan.window) if (plan.window > ZERO and eff_cp == ONE)
                else local_sweeps if (plan.cache_kind == "dense" and not topk
                                      and eff_q == ONE)
                else local_rows
            )
            kv_hbm_component = float(local_cache_read)
            kv_mte2_component = float(traffic.private_refill_bytes)
            kv_hbm_total += kv_hbm_component
            kv_mte2_total += kv_mte2_component
            is_generic_decode = (
                latent and view.phase == "decode" and eff_q == ONE
                and plan.cache_kind == "dense" and not topk and not plan.indexer
                and model.activation_dtype == "bf16" and quant.kv_dtype == "bf16"
            )
            last = b.work(
                f"latent_attention{orig_idx}", tuple(dict.fromkeys(
                    item for item in (q_ready, rotary if latent else None, store,
                                      long_store, topk, metadata_ready) if item)),
                (eff_q, view.kv_len, attn_heads, qk_dim, latent_dim),
                flops=TWO * local_pairs * attn_heads * (qk_dim + latent_dim),
                vector=FIVE * local_pairs * attn_heads,
                read=kv_mte2_component + rec["selection_reads"] + q_read,
                hbm_read=kv_hbm_component + rec["selection_reads"] + q_read,
                write=out_write, family="attention",
                metadata=(("query_tile", query_tile), ("logical_pairs", pairs),
                          ("kv_sweeps", sweeps),
                          ("attention_traffic_plan", traffic.to_dict()))
                         + rec["selection_metadata"]
                         + attention_tensor_view(plan, view, attn_heads, model, quant,
                                                 parallel, eff_cp=eff_cp)
                         + ((("program_kind", "dense_latent_attention_generic"),)
                            if is_generic_decode else ())
                         + ((("latent_attention_schedule", {
                             "heads": attn_heads, "query_rows": eff_q,
                             "keys": max(ZERO, vis_keys),
                             "qk_dim": qk_dim, "latent_dim": latent_dim,
                             "dv": int(latent_dim),
                             "q_bytes": q_read,
                             "kv_row_bytes": kv_row_b,
                             "kv_unique_bytes": kv_row_b * max(ZERO, vis_keys),
                             "output_bytes": out_write,
                             "owners": max(ONE, min(
                                 MAX_GENERIC_OWNERS,
                                 math.ceil(max(ONE, vis_keys) / DEFAULT_QUERY_TILE))),
                             "tile_keys": DEFAULT_QUERY_TILE,
                             "page_size": int(getattr(runtime, "kv_block_size", ZERO)
                                              or DEFAULT_BLOCK_SIZE),
                             "source_scope": "generic_online_attention",
                         }),) if is_generic_decode else ())
                         + ((("mla_head_contract", dict(head_contract)),)
                            if head_contract else ())
                         + ((("sparse_kv_offload", offload_info),) if offload_info else ()),
            )
            kv_components[b.ops[-ONE].op_id] = (
                kv_hbm_component, kv_mte2_component)
            b.ops[-ONE] = replace(
                b.ops[-ONE],
                local_shape=(eff_q, local_rows, attn_heads, qk_dim, latent_dim),
            )
            if offload_info and offload_info.get("use_fused_overlap"):
                b.ops[-ONE] = replace(
                    b.ops[-ONE],
                    recipe="fused_sparse_attention_overlap",
                    source_boundary="kernel",
                    metadata=b.ops[-ONE].metadata + (("sparse_offload_sfa", offload_info),),
                )
            if (runtime and view.phase == "prefill" and topk
                    and plan.cache_kind == "window_compressed"
                    and plan.mla_variant == "latent_mqa"):
                sharedkv = select(runtime.kernels.get("lowerings", {}), "sharedkv_attention")
                if sharedkv:
                    b.ops[-ONE] = sharedkv.module.lower_op(
                        b.ops[-ONE], selection=sharedkv, view=view, plan=plan,
                        views=grp_views, heads=attn_heads, parallel=parallel,
                        model=model, quant=quant)
            if runtime and topk and plan.cache_kind == "dense":
                indexed = select(runtime.kernels.get("lowerings", {}), "indexed_attention")
                if indexed:
                    b.ops[-ONE] = indexed.module.lower_op(
                        b.ops[-ONE], selection=indexed, view=view, views=grp_views,
                        plan=plan, parallel=parallel)
            grp_outputs.append(last)
        if mixed_quant:
            grp_outputs = mixed_quant.module.lower_attention(
                b.ops, grp_outputs, selection=mixed_quant, views=grp_views, plan=plan,
                topk_present=bool(topk), prefix=grp_prefix)
        elif not selected:
            from .generic_latent_attention import batch_general_mla_ops, batch_ops
            from .indexed_attention import batch_analytic_ops
            if plan.cache_kind == "dense" and not pcp_prefill:
                grp_outputs = batch_ops(
                    b.ops, grp_outputs, runtime=runtime, views=grp_views,
                    prefix=grp_prefix)
                if eff_cp == ONE:
                    grp_outputs = batch_analytic_ops(
                        b.ops, grp_outputs, runtime=runtime, views=grp_views,
                        prefix=grp_prefix)
            if not gathered_dcp:
                grp_outputs = batch_general_mla_ops(
                    b.ops, grp_outputs, views=grp_views, prefix=grp_prefix,
                    runtime=runtime, policy=policy)
            cfg_fam = runtime.kernels.get("families", {}) if runtime else {}
            cfg_rec = runtime.kernels.get("recipes", {}) if runtime else {}
            pinned_attn = "attention" in cfg_fam or any(
                item in cfg_rec for item in
                (f"latent_attention{ZERO}", "latent_attention_batch"))
            out_set = set(grp_outputs)
            for idx_op, op_item in enumerate(b.ops):
                if op_item.op_id not in out_set or op_item.source_boundary:
                    continue
                if topk:
                    b.ops[idx_op] = annotate_sparse_prefix_unknown(op_item, runtime)
                elif len(grp_outputs) == ONE:
                    b.ops[idx_op] = apply_attention_prefix_reads(
                        op_item, views=grp_views, sweeps_by_view=sweeps_by_view,
                        shared_sweeps_by_view=shared_by_view, row_bytes=float(kv_row_b),
                        layer_index=plan.index, runtime=runtime,
                        kernel_reuse=("batch_tile"
                                      if runtime is not None
                                      and runtime.prefix_cache_mode == "shared"
                                      else ("native_per_sequence" if pinned_attn else "tile")),
                        allow_recipe_rebind=not pinned_attn,
                        kv_hbm_bytes=kv_hbm_total,
                        kv_mte2_bytes=kv_mte2_total)
                else:
                    record = view_records[grp_outputs.index(op_item.op_id)] \
                        if op_item.op_id in grp_outputs else None
                    if record is not None:
                        component = kv_components.get(op_item.op_id,
                                                      (ZERO_FLOAT, ZERO_FLOAT))
                        b.ops[idx_op] = apply_attention_prefix_reads(
                            op_item, views=(record["view"],),
                            sweeps_by_view=(sweeps_by_view[grp_outputs.index(op_item.op_id)],),
                            shared_sweeps_by_view=(shared_by_view[grp_outputs.index(op_item.op_id)],),
                            row_bytes=float(kv_row_b), layer_index=plan.index,
                            runtime=runtime, kernel_reuse="native_per_sequence",
                            allow_recipe_rebind=False,
                            kv_hbm_bytes=component[ZERO],
                            kv_mte2_bytes=component[ONE])
        if eff_cp > ONE and grp_outputs:
            out_set = set(grp_outputs)
            for idx_op, op_item in enumerate(b.ops):
                if op_item.op_id not in out_set:
                    continue
                b.ops[idx_op] = replace(
                    op_item,
                    metadata=op_item.metadata + (
                        ("cp_per_rank_pairs", tuple(rank_pairs)),
                        ("cp_per_rank_sweeps", tuple(rank_sweeps)),
                        ("cp_bottleneck_pairs_rank", worst_pairs_r),
                        ("cp_bottleneck_sweeps_rank", worst_sweeps_r),
                    ))
        if eff_cp > ONE and grp_outputs and not gathered_dcp:
            merged, merged_last = dcp_partial_merge_ops(
                prefix=grp_prefix, views=grp_views, cp=eff_cp, local_heads=heads,
                v_dim=latent_dim, stage=stage, layer=plan.index, mode=mode,
                deps=tuple(grp_outputs), total_pairs=total_pairs_sum,
                local_pairs=rank_pairs[worst_pairs_r], total_rows=total_rows_sum,
                local_rows=rank_rows[worst_pairs_r], interleave=cp_interleave_size,
            )
            b.ops.extend(merged)
            grp_outputs = [merged_last]
        outputs.extend(grp_outputs)
    if gathered_dcp and not selected:
        outputs = latent_dcp.attention_batch(
            b, outputs, views, plan=plan, heads=heads,
            local_heads=logical_heads, cp=cp, rows=q)
    heads = logical_heads
    if latent:
        outputs = [b.matrix("v_expand", outputs, q * heads, plan.v_head_dim,
                            plan.kv_lora_rank, weight_count=heads,
                            category="mla_absorb",
                            metadata=batched_matrix_metadata(
                                tokens=q, heads=heads, input_width=plan.kv_lora_rank,
                                output_width=plan.v_head_dim, role="value_expansion")
                            + mla_head_padding.output_metadata(head_contract))]
    output_width = heads * (plan.v_head_dim if latent else plan.head_dim)
    if plan.mla_output_gate and latent:
        gate = b.matrix("output_gate", (deps,), q, output_width, h, category="mla.g")
        outputs = [b.vector("gated_output", (*outputs, gate), q * output_width, FIVE)]
    if partial_rotary:
        outputs = [partial_rotary.module.declare(
            b, name="output_inverse_rotary", deps=outputs, tokens=q, heads=heads,
            head_dim=plan.head_dim, rotary_width=plan.qk_rope_head_dim, inverse=True)]
    group_parallel, groups = _group_plan(plan, parallel)
    if plan.o_groups:
        if not group_parallel and parallel.tp > ONE:
            outputs = [b.communicate(
                "output_gather", outputs, q * output_width * TWO * (parallel.tp - ONE),
                "allgather")]
        group_width = (plan.num_heads
                       * (plan.v_head_dim if latent else plan.head_dim) // plan.o_groups)
        first = b.matrix("grouped_output_a", outputs, q * groups, plan.o_lora_rank,
                         group_width, weight_count=groups, category="mla.o_a",
                         metadata=batched_matrix_metadata(
                             tokens=q, heads=groups, input_width=group_width,
                             output_width=plan.o_lora_rank, role="grouped_output"),
                         weight_bytes_override=b.quant.matrix_bytes(
                             groups * plan.o_lora_rank, group_width, "mla.o_a"))
        last = b.matrix("grouped_output_b", (first,), q, h, groups * plan.o_lora_rank,
                        category="mla.o_b")
    else:
        last = b.matrix("output", outputs, q, h, output_width, category="mla.o")
    if parallel.tp > ONE and (not plan.o_groups or group_parallel):
        if dsa_prefill:
            last = b.communicate(
                "dsa_cp_output_all_gather", (last,),
                (parallel.tp - ONE) / parallel.tp * q * h
                * dtype_bytes(model.activation_dtype), "allgather", "tp")
        else:
            last = b.communicate(
                "output_reduce", (last,),
                TWO * (parallel.tp - ONE) / parallel.tp * q * h
                * dtype_bytes(model.activation_dtype), "allreduce")
    return AttentionResult(ops=b.ops, last=last, cache_store_id=long_store or store,
                           index_store_id=index_store, topk_id=topk,
                           candidate_id=candidate)
