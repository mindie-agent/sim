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
from functools import partial

from mindie_sim.catalog.attention_common import (
    AttentionResult, _group_plan,
)
from mindie_sim.catalog.cache import completed_rows, packed_region_row_bytes
from mindie_sim.catalog.context_parallel import (
    dcp_indexer_sharded, is_v41_replicated_cache_model, layer_dcp_degree,
    layer_dcp_factors, owned_positions,
)
from mindie_sim.schema.common import dtype_bytes, ModelingError
from mindie_sim.prediction.operators.attention.batch_prefix_attention import (
    annotate_sparse_prefix_unknown, apply_attention_prefix_reads,
    apply_sparse_prefix_reads,
)
from mindie_sim.prediction.operators.context_parallel.graph import (
    CP_DECODE_COMM_STREAM,
    active_layer_dcp,
    dcp_q_replicate_enabled,
    emit_dcp_output_exchange_combine,
    emit_dcp_query_gather,
    emit_mtp_current_attention,
    emit_pcp_hidden_gather_restore,
    emit_pcp_o_proj_weight_allgather,
    emit_pcp_restore_unpermute,
    emit_sfa_index_remap,
    emit_sfa_prefill_kv_gather,
    emit_v41_pcp_c2_ring_gather,
    emit_v41_pcp_c2_ring_write,
    paired_sfa_attention_batch,
    pcp_o_proj_weight_sharding_enabled,
    pcp_uniform_peer_rows,
    resolve_cp_rank_candidates,
    resolve_dcp_cohort_decode,
    should_split_mtp_dcp,
)
from mindie_sim.prediction.implementations import select
from mindie_sim.prediction.implementation_selection import is_automatic_lowering
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.operators.mla_prolog.compression import compressor
from .mla_epilogs import (
    SharedSelectionRef, block_quant_epilog, kv_cache_epilog, native_cache_store,
    native_fusion_active, lower_index_path,
)
from .mla_native import batched_matrix_metadata
from mindie_sim.prediction.memory_traffic import (AttentionAccessSpec, AttentionSchedule,
                             default_hierarchy, plan_attention_traffic)
from mindie_sim.prediction.projections import mla_input_projections
from mindie_sim.prediction.operators.mla import mla_head_padding
from mindie_sim.prediction.cache_representation import store as representation_store
from mindie_sim.catalog.cache_format import cache_format
from .generic_latent_attention import (
    attention_tensor_view, attention_view_record, dsa_cp_prefill_parts,
    kv_sweeps, owned_completed_rows, owned_store_rows, selected_pairs,
    semantic_index_selection, sparse_over_parts, view_causal_pairs,
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

DEFAULT_INTERLEAVE = 128
DEFAULT_BLOCK_SIZE = 128
DEFAULT_QUERY_TILE = 128
MAX_GENERIC_OWNERS = 32
ROPE_OPS = 6
RMSNORM_OPS_PER_ELEM = 5
SOFTMAX_VECTOR_OPS_PER_PAIR = 5
GATE_APPLY_OPS_PER_ELEMENT = 5


def _store_rows(rows_by_view, views, parallel, block, *, contexts=None):
    return max(1, owned_store_rows(
        rows_by_view, views, parallel, block, contexts=contexts))


def _memory_hierarchy(machine):
    """Declared device hierarchy for traffic planning, or the frontend default."""
    if machine is not None and getattr(machine, "dies", ()):
        hierarchy = getattr(machine.dies[0].physical_mechanisms,
                            "memory_hierarchy", None)
        if hierarchy is not None and hierarchy.l2_capacity_bytes > 0:
            return hierarchy
    return default_hierarchy()


def _aic_owners(machine):
    """Declared AIC owner count of the representative shard, or ``None``."""
    dies = tuple(getattr(machine, "dies", ()) or ())
    return min(die.compute_units for die in dies) if dies else None


def _mla_idle_dcp_owner(
    *, prefix, plan, stage, deps, model, parallel, quant, mode, runtime, policy,
    machine, cp_interleave_size, cohort_dec_views, peer_dec_rows, local_dp_pos,
    pcp_factor, tp_gather, dp_width, physical_rank=None, explicit_physical_rank=False,
) -> AttentionResult:
    """Idle KV-owner peer in a cross-DP DCP cohort: gather Q, run attention, return partial O/LSE."""
    b = Builder(prefix, plan, stage, cohort_dec_views, model, mode)
    hierarchy = _memory_hierarchy(machine)
    b.l2_bytes = hierarchy.effective_l2_bytes
    heads = plan.num_heads // parallel.tp
    width = plan.qk_nope_head_dim + plan.qk_rope_head_dim
    latent = plan.mla_variant != "latent_mqa"
    latent_dim = plan.kv_lora_rank if latent else plan.head_dim
    qk_dim = latent_dim + plan.qk_rope_head_dim if latent else width
    gather_w = (plan.kv_lora_rank + plan.qk_rope_head_dim) if latent else width
    query_tile = policy.query_tile if policy else DEFAULT_QUERY_TILE
    is_sparse = bool(plan.indexer or plan.owns_topk
                     or plan.topk_owner_layer_id >= 0
                     or plan.index_k_owner_layer_id >= 0)
    q_dcp, dcp_heads, _ = emit_dcp_query_gather(
        b.ops, prefix=prefix, stage=stage, layer=plan.index,
        requests=b.common["requests"], mode=mode, deps=(deps,),
        rows=0, local_heads=heads, head_width=gather_w,
        act_dtype=model.activation_dtype, tp_gather=tp_gather,
        dp_width=dp_width, pack_inputs=False, q_replicate=False,
        peer_rows=peer_dec_rows, local_dp_pos=local_dp_pos,
        pcp_factor=pcp_factor,
    )
    outputs: list[str] = []
    request_op = partial(
        _append_mla_request, b=b, runtime=runtime, parallel=parallel, quant=quant,
        local_heads=heads, dcp_heads=dcp_heads, dp_width=dp_width,
        qk_dim=qk_dim, latent_dim=latent_dim, latent=latent,
        query_tile=query_tile, hierarchy=hierarchy, topk=None,
        q_local=q_dcp, rotary_local=q_dcp, q_dcp=q_dcp, rotary_dcp=q_dcp,
        store=None, long_store=None, metadata_ready=None,
        machine=machine, head_contract=None, offload_info=None)
    _append_mla_attention(
        b, outputs, (), runtime, None, parallel, False, model, plan, quant,
        query_tile, cp_interleave_size, None, heads, dcp_heads, latent_dim, qk_dim,
        False, request_op, None, False, False,
        is_sparse, pcp_factor, tp_gather, dp_width, policy, heads, 0,
        hierarchy, stage, mode, q_dcp, None,
        cohort_decode_views=cohort_dec_views,
        peer_decode_rows=peer_dec_rows,
        local_dp_pos=local_dp_pos,
        idle_dcp_owner=True,
        physical_rank=physical_rank,
        explicit_dp_cohort=True,
        explicit_physical_rank=explicit_physical_rank)
    return AttentionResult(ops=b.ops, last=outputs[-1], cache_store_id=None,
                           index_store_id=None, topk_id=None, candidate_id=None)


def mla(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
        padded_tokens, cache_ids, topk_ids, index_cache_ids, candidate_ids,
        stage_of, cp_interleave_size=DEFAULT_INTERLEAVE, runtime=None, policy=None,
        machine=None, reuse_index_selection=False, shared_topk=None,
        dcp_cohort_views_by_member=None, dcp_local_pos=0,
        physical_rank=None, explicit_physical_rank=False):
    pcp_factor, tp_gather, raw_dp_width = layer_dcp_factors(plan, parallel)
    dp_width = raw_dp_width if int(getattr(parallel, "dp", 1) or 1) >= raw_dp_width else 1
    cohort_dec_views, peer_dec_rows, local_dp_pos = resolve_dcp_cohort_decode(
        views, dp_width=dp_width,
        cohort_views_by_member=dcp_cohort_views_by_member,
        local_dp_pos=dcp_local_pos)
    if (padded_tokens == 0 and not views and layer_dcp_degree(plan, parallel) > 1
            and dp_width > 1 and cohort_dec_views):
        return _mla_idle_dcp_owner(
            prefix=prefix, plan=plan, stage=stage, deps=deps, model=model,
            parallel=parallel, quant=quant, mode=mode, runtime=runtime,
            policy=policy, machine=machine, cp_interleave_size=cp_interleave_size,
            cohort_dec_views=cohort_dec_views, peer_dec_rows=peer_dec_rows,
            local_dp_pos=local_dp_pos, pcp_factor=pcp_factor,
            tp_gather=tp_gather, dp_width=dp_width,
            physical_rank=physical_rank,
            explicit_physical_rank=explicit_physical_rank)
    b = Builder(prefix, plan, stage, views, model, mode)
    hierarchy = _memory_hierarchy(machine)
    b.l2_bytes = hierarchy.effective_l2_bytes
    from mindie_sim.catalog.paired_sfa import execution
    layer_dcp = layer_dcp_degree(plan, parallel)
    is_sparse = bool(plan.indexer or plan.owns_topk
                     or plan.topk_owner_layer_id >= 0
                     or plan.index_k_owner_layer_id >= 0)
    sharded_indexer = dcp_indexer_sharded(model, runtime) and layer_dcp > 1
    has_prefill = any(view.phase == "prefill" for view in views)
    sfa_replicated_prefill_path = bool(
        layer_dcp > 1 and is_sparse and not sharded_indexer and has_prefill
    )
    cp = (1 if sfa_replicated_prefill_path else
          max((active_layer_dcp(plan, parallel, view.phase, sharded_indexer=sharded_indexer)
               for view in views), default=1))
    if not sfa_replicated_prefill_path and dp_width > 1 and cohort_dec_views:
        cp = max(cp, layer_dcp)
    decode_dcp = (not sfa_replicated_prefill_path and cp > 1
                  and (any(view.phase != "prefill" for view in views)
                       or sharded_indexer
                       or (dp_width > 1 and bool(cohort_dec_views))))
    execution_flags = execution(model)
    paired = dict(model.metadata).get("cache_allocator") == "paired_sfa"
    paired_sfa_mode = paired and not any(view.phase != "decode" for view in views)
    q_replicate = dcp_q_replicate_enabled(
        runtime, pcp_factor=pcp_factor, dp_width=dp_width, sparse=is_sparse)
    own_block = (int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE)
                 if paired else int(cp_interleave_size))
    index_rope = (int(execution_flags.get("index_rope_dim", plan.qk_rope_head_dim))
                  if plan.mla_do_rope else 0)
    prolog, partial_rotary, paged, selected, mixed_quant = _select_mla_lowerings(
        runtime, plan, parallel, model, quant, views, padded_tokens, machine)
    q, h, heads = padded_tokens, model.hidden_size, plan.num_heads // parallel.tp
    head_contract = (mla_head_padding.contract(model, plan, views)
                     if runtime and select(runtime.kernels.get("lowerings", {}), "mla_head_padding")
                     else None)
    logical_heads = heads
    if head_contract:
        heads = head_contract["physical_heads"]
    dsa_prefill = (
        parallel.cp_plan == "dsa_cp" and parallel.tp > 1
        and any(view.phase == "prefill" for view in views)
    )
    width = plan.qk_nope_head_dim + plan.qk_rope_head_dim
    latent = plan.mla_variant != "latent_mqa"
    query_tile = policy.query_tile if policy else DEFAULT_QUERY_TILE
    kv_width = plan.kv_lora_rank + (plan.qk_rope_head_dim if latent else 0)
    do_rope = bool(plan.mla_do_rope)
    compressor_binding = (select(runtime.kernels.get("lowerings", {}), "compressor")
                          if runtime else None)
    native_fusion = bool(compressor_binding) and native_fusion_active(runtime, machine)
    native_cache_epilogs = native_fusion_active(runtime, machine)
    is_v41 = is_v41_replicated_cache_model(model)
    (q_local, rotary_local, q_dcp, rotary_dcp, q_norm, valid, pcp_prefill,
     store, dcp_heads, cache_hidden, cache_views, cache_q, v4_hidden_gathered,
     kpool_gathered) = _project_mla_inputs(
        b, q, h, heads, width, kv_width, deps, prolog, do_rope, partial_rotary, views,
        parallel, own_block, native_cache_epilogs, latent, quant,
        decode_dcp=decode_dcp, pcp_factor=pcp_factor, tp_gather=tp_gather,
        dp_width=dp_width, q_replicate=q_replicate, is_sparse=is_sparse,
        peer_rows=(peer_dec_rows if (dp_width > 1 and decode_dcp) else None),
        local_dp_pos=local_dp_pos, runtime=runtime, is_v41=is_v41)
    long_store, compressed_latent = _produce_long_cache(
        b, store, stage_of, cache_views, cache_hidden,
        sum(v.q_len for v in cache_views), cache_q,
        native_fusion, parallel, own_block,
        native_cache_epilogs, pcp_prefill and not v4_hidden_gathered, cache_ids, cp,
        runtime=runtime, is_v41=is_v41)
    if sfa_replicated_prefill_path:
        pf_kv_rows = sum(view.kv_len for view in views)
        if pf_kv_rows > 0:
            kv_fmt = cache_format(model, "mla_kv")
            store = emit_sfa_prefill_kv_gather(
                b.ops, prefix=prefix, stage=stage, layer=plan.index,
                requests=b.common["requests"], mode=mode, store_dep=store,
                prefill_context_rows=pf_kv_rows,
                kv_row_bytes=kv_fmt.row_bytes(plan.cache_row_elements),
                layer_dcp=layer_dcp, kv_dtype=kv_fmt.dtype,
                recipe="sfa_dcp_prefill_kv_gather")
            if plan.cache_kind == "dense":
                long_store = store
    elif not is_sparse and layer_dcp > 1 and has_prefill:
        vblock = max(1, int(layer_dcp) * int(cp_interleave_size))
        pf_ctx_padded = sum(
            math.ceil(int(view.context) / vblock) * int(cp_interleave_size) * int(layer_dcp)
            for view in views if view.phase == "prefill" and view.context > 0
        )
        if pf_ctx_padded > 0:
            kv_fmt = cache_format(model, "mla_kv")
            store = emit_sfa_prefill_kv_gather(
                b.ops, prefix=prefix, stage=stage, layer=plan.index,
                requests=b.common["requests"], mode=mode, store_dep=store,
                prefill_context_rows=pf_ctx_padded,
                kv_row_bytes=kv_fmt.row_bytes(plan.cache_row_elements),
                layer_dcp=layer_dcp, kv_dtype=kv_fmt.dtype,
                recipe="dcp_prefill_kv_gather")
            if plan.cache_kind == "dense":
                long_store = store

    pool_binding = (select(runtime.kernels.get("lowerings", {}), "pool_key_indexer")
                    if runtime and plan.index_kpool > 1 else None)
    v41_comp_gathered = bool(
        is_v41 and pcp_prefill and plan.owns_long_kv
        and plan.cache_kind == "window_compressed" and plan.compress_ratio > 0
        and plan.index_input == "compressed"
    )
    if kpool_gathered is not None:
        idx_hidden, idx_views, idx_q = kpool_gathered
    elif v4_hidden_gathered:
        idx_hidden, idx_views, idx_q = cache_hidden, cache_views, cache_q
    elif v41_comp_gathered:
        idx_views = tuple(
            replace(v, q_len=int(v.global_q_len or v.q_len),
                    query_ranges=((0, int(v.global_q_len or v.q_len)),))
            if v.phase == "prefill" else v for v in views
        )
        idx_hidden, idx_q = None, max(q * parallel.pcp, sum(v.q_len for v in idx_views))
    else:
        idx_hidden, idx_views, idx_q = None, None, None
    index_store, topk, candidate, pool_weights = lower_index_path(
        b, plan=plan, model=model, quant=quant, parallel=parallel, views=views,
        deps=deps, runtime=runtime, padded_tokens=q, valid=valid, cp=cp,
        own_block=own_block, long_store=long_store,
        compressed_latent=compressed_latent, cache_ids=cache_ids,
        index_cache_ids=index_cache_ids, candidate_ids=candidate_ids,
        topk_ids=topk_ids, q_norm=q_norm, stage=stage, stage_of=stage_of,
        index_rope=index_rope, pool_binding=pool_binding,
        pcp_prefill=pcp_prefill, dsa_prefill=dsa_prefill, hierarchy=hierarchy,
        reuse_selection=reuse_index_selection, shared_topk=shared_topk,
        fused_compressor=native_fusion,
        cache_hidden_dep=idx_hidden,
        cache_views=idx_views,
        cache_padded_tokens=idx_q,
        skip_pcp_index_gather=(v4_hidden_gathered or kpool_gathered is not None
                               or v41_comp_gathered))
    attn_topk = topk
    if decode_dcp and topk and not sharded_indexer and plan.index_topk > 0:
        attn_topk = emit_sfa_index_remap(
            b.ops, prefix=prefix, stage=stage, layer=plan.index,
            requests=b.common["requests"], mode=mode, topk_dep=topk,
            rows=max(1, valid), topk=plan.index_topk,
            layer_dcp=cp, interleave=cp_interleave_size)
    if head_contract:
        q_local, rotary_local = mla_head_padding.query_inputs(
            b, declaration=head_contract, tokens=q, query=q_local,
            rotary=rotary_local, plan=plan)
        if not decode_dcp:
            q_dcp, rotary_dcp = q_local, rotary_local
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
            store, attn_topk, offload_info = sparse_offload.module.lower_decode_offload(
                b, plan=plan, views=views, store_id=store, topk_id=attn_topk, budget=offload_budget)
            if plan.cache_kind == "dense":
                long_store = store
    outputs = []
    metadata_ready = None
    metadata_source = (select(runtime.kernels.get("lowerings", {}), "sparse_mla_metadata")
                       if runtime else None)
    if metadata_source:
        metadata_ready = metadata_source.module.lower(
            b, views=views, plan=plan, parallel=parallel, quant=quant,
            deps=(attn_topk or deps,), padded_tokens=q, stage_of=stage_of,
            **metadata_source.options)
    latent_dim = plan.kv_lora_rank if latent else plan.head_dim
    qk_dim = latent_dim + plan.qk_rope_head_dim if latent else width
    if selected:
        sel_heads = dcp_heads if decode_dcp else heads
        sel_views = (cohort_dec_views if (decode_dcp and dp_width > 1 and cohort_dec_views)
                     else views)
        sel_out = selected.module.lower_attention(
            b, views=sel_views, plan=plan, heads=sel_heads, runtime=runtime,
            query=q_dcp if decode_dcp else q_local,
            rotary=rotary_dcp if decode_dcp else rotary_local, store=store,
            **({"head_contract": head_contract,
                "padded_tokens": sum(v.q_len for v in sel_views) if dp_width > 1 else q}
               if paged else {}),
            **selected.options)
        if decode_dcp and not paired_sfa_mode:
            sel_out = emit_dcp_output_exchange_combine(
                b.ops, prefix=prefix, stage=stage, layer=plan.index,
                requests=b.common["requests"], mode=mode, deps=(sel_out,),
                rows=max(1, q), local_heads=logical_heads, v_dim=latent_dim,
                act_dtype=model.activation_dtype, pcp_factor=pcp_factor,
                tp_gather=tp_gather, dp_width=dp_width,
                peer_rows=(peer_dec_rows if dp_width > 1 else None),
                local_dp_pos=local_dp_pos)
        outputs.append(sel_out)
    request_op = partial(
        _append_mla_request, b=b, runtime=runtime, parallel=parallel, quant=quant,
        local_heads=heads, dcp_heads=dcp_heads, dp_width=dp_width,
        qk_dim=qk_dim, latent_dim=latent_dim, latent=latent,
        query_tile=query_tile, hierarchy=hierarchy, topk=attn_topk,
        q_local=q_local, rotary_local=rotary_local, q_dcp=q_dcp, rotary_dcp=rotary_dcp,
        store=store, long_store=long_store, metadata_ready=metadata_ready,
        machine=machine, head_contract=head_contract, offload_info=offload_info)
    _append_mla_attention(
        b, outputs, views, runtime, selected, parallel, dsa_prefill, model, plan, quant,
        query_tile, cp_interleave_size, attn_topk, heads, dcp_heads, latent_dim, qk_dim,
        reuse_index_selection, request_op, mixed_quant, pcp_prefill, paired_sfa_mode,
        is_sparse, pcp_factor, tp_gather, dp_width, policy, logical_heads, q,
        hierarchy, stage, mode, q_local, store,
        cohort_decode_views=(cohort_dec_views if dp_width > 1 else None),
        peer_decode_rows=(peer_dec_rows if dp_width > 1 else None),
        local_dp_pos=local_dp_pos, sharded_indexer=sharded_indexer,
        physical_rank=physical_rank,
        explicit_dp_cohort=(dcp_cohort_views_by_member is not None),
        explicit_physical_rank=bool(explicit_physical_rank),
        sfa_replicated_prefill_path=sfa_replicated_prefill_path)
    last = _project_mla_output(
        b, logical_heads, latent, outputs, q, plan, head_contract, deps, h, partial_rotary,
        parallel, model, dsa_prefill, views=views, runtime=runtime)
    return AttentionResult(ops=b.ops, last=last, cache_store_id=long_store or store,
                           index_store_id=index_store, topk_id=topk,
                           candidate_id=candidate)


def _select_mla_lowerings(runtime, plan, parallel, model, quant, views, padded_tokens, machine):
    prolog = (select(runtime.kernels.get("lowerings", {}), "mla_prolog")
              if runtime else None)
    if prolog:
        prolog.module.validate(plan=plan, parallel=parallel, model=model, quant=quant,
                               lowering_passes=runtime.lowering_passes, **prolog.options)
    partial_rotary = (select(runtime.kernels.get("lowerings", {}), "v41_rotary")
                      if runtime else None)
    if partial_rotary and plan.mla_do_rope:
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
                   and any(view.phase in {"decode", "draft"} for view in views) else None)
    if mixed_quant and not sum(int(view.q_len) for view in views):
        # ``dsa_v41`` slices to zero actual tokens and skips the attention call
        # on an idle DP replica; no native binding is claimed without rows.
        mixed_quant = None
    if mixed_quant:
        reason = mixed_quant.module.unsupported_reason(
            views=views, plan=plan, parallel=parallel, model=model,
            padded_tokens=padded_tokens, cores=_aic_owners(machine),
            topk_present=bool(plan.index_topk),
            implementation=mixed_quant.name, **mixed_quant.options)
        if reason:
            # A default-installed policy must never turn a supported graph into
            # a planner failure; an explicitly requested binding still must.
            if not is_automatic_lowering(runtime.metadata, "mqsmla_attention"):
                raise ModelingError(reason)
            mixed_quant = None
    if selected:
        selected.module.validate(
            views=views, plan=plan, parallel=parallel, model=model, quant=quant,
            padded_tokens=padded_tokens, **selected.options)
    return prolog, partial_rotary, paged, selected, mixed_quant


def _produce_mla_cache(
    b, views, parallel, own_block, rotary, kv, kv_a, do_rope, prolog, native_cache_epilogs,
    pcp_prefill, *, is_v41=False, padded_tokens=0,
):
    plan, model = b.plan, b.model
    cache_deps = ((rotary,) if do_rope else (kv, kv_a))
    cache_region = "window" if plan.cache_kind == "window_compressed" else "mla_kv"
    v41_pre_store_gather = bool(
        is_v41 and pcp_prefill and plan.cache_kind == "window_compressed" and int(parallel.pcp) > 1
    )
    if v41_pre_store_gather:
        per_peer, global_rows = pcp_uniform_peer_rows(
            views, parallel.pcp, local_rows=max(1, int(padded_tokens)))
        proj_w = int(plan.kv_lora_rank or plan.cache_row_elements)
        proj_dtype = model.activation_dtype
        proj_row_b = float(proj_w) * dtype_bytes(proj_dtype)
        gathered = b.communicate(
            "pcp_kv_gather", cache_deps,
            float(per_peer) * proj_row_b * float(parallel.pcp - 1),
            "allgather", "pcp", dtype=proj_dtype,
            shape=(per_peer * int(parallel.pcp), proj_w),
            source="vllm_ascend.attention.context_parallel.dsa_v41_cp")
        restored = emit_pcp_restore_unpermute(
            b.ops, prefix=b.prefix, name="pcp_kv_restore",
            stage=b.common["stage"], layer=plan.index,
            requests=b.common["requests"], mode=b.common["phase"],
            dep=gathered, global_rows=global_rows,
            row_bytes=proj_row_b, dtype=proj_dtype)
        cache_deps = (restored,)
        store_rows = max(1, global_rows)
    else:
        store_rows = _store_rows([view.q_len for view in views], views, parallel, own_block)
    packed_store = (prolog.module.cache_store(
        b, deps=cache_deps, rows=store_rows, plan=plan, **prolog.options)
        if prolog else None)
    store = packed_store or (representation_store(
        b, "kv_store", cache_deps, store_rows, plan.cache_row_elements,
        cache_region))
    if store is not None and packed_store is None:
        cache_fmt = cache_format(model, cache_region)
        for position, item in enumerate(b.ops):
            if item.op_id != store:
                continue
            written = kv_cache_epilog(item, cache_fmt, cache_region,
                                      native=native_cache_epilogs)
            if plan.qk_rope_head_dim == 0:
                written = block_quant_epilog(
                    written, cache_fmt, cache_region, native=native_cache_epilogs,
                    width=plan.cache_row_elements)
            b.ops[position] = written
            store = written.op_id
            break
    if store is not None and native_cache_epilogs:
        for position, item in enumerate(b.ops):
            if item.op_id != store:
                continue
            b.ops[position] = native_cache_store(item, native=True)
            store = b.ops[position].op_id
            break
    produced, owns_kv = store, store is not None
    if pcp_prefill and owns_kv and not v41_pre_store_gather:
        rows = sum(view.global_q_len or view.q_len for view in views if view.phase == "prefill")
        if packed_store is not None:
            from mindie_sim.prediction.operators.mla_prolog.mla_prolog_packed import view as packed_kv_view
            kv_row = packed_kv_view(plan.kv_lora_rank, plan.qk_rope_head_dim)["row_bytes"]
            kv_dtype = "int8"
        else:
            kv_row = packed_region_row_bytes(
                model, cache_region, plan.cache_row_elements)
            kv_dtype = cache_format(model, cache_region).dtype
        store = b.communicate(
            "pcp_kv_gather", (produced,),
            rows * kv_row * (parallel.pcp - 1) / parallel.pcp,
            "allgather", "pcp", dtype=kv_dtype)
        if plan.cache_kind == "window_compressed":
            store = emit_pcp_restore_unpermute(
                b.ops, prefix=b.prefix, name="pcp_kv_restore",
                stage=b.common["stage"], layer=plan.index,
                requests=b.common["requests"], mode=b.common["phase"],
                dep=store, global_rows=rows, row_bytes=kv_row, dtype=kv_dtype)
    return store, produced


def _project_mla_inputs(
    b, q, h, heads, width, kv_width, deps, prolog, do_rope, partial_rotary, views, parallel,
    own_block, native_cache_epilogs, latent, quant, *, decode_dcp, pcp_factor=1, tp_gather,
    dp_width, q_replicate, is_sparse, peer_rows=None, local_dp_pos=0, runtime=None,
    is_v41=False,
):
    plan, model = b.plan, b.model
    q_replicate_active = bool(q_replicate and tp_gather > 1 and layer_dcp_degree(plan, parallel) > 1)
    proj_heads = (heads * tp_gather) if q_replicate_active else heads
    post_q_heads = heads if (q_replicate_active and not decode_dcp) else proj_heads
    norm_dtype = model.activation_dtype
    norm_bytes = dtype_bytes(norm_dtype)
    pcp_prefill = parallel.pcp > 1 and any(view.phase == "prefill" for view in views)
    kpool_gathered = None
    if pcp_prefill and plan.owns_index_k and plan.index_kpool > 1 and int(parallel.pcp) > 1:
        kpool_views = tuple(
            replace(v, q_len=int(v.global_q_len or v.q_len),
                    query_ranges=((0, int(v.global_q_len or v.q_len)),))
            if v.phase == "prefill" else v
            for v in views
        )
        kpool_per_peer, kpool_q = pcp_uniform_peer_rows(views, parallel.pcp, local_rows=q)
        kpool_source = emit_pcp_hidden_gather_restore(
            b.ops, prefix=b.prefix,
            gather_name="pcp_keypool_hidden_gather",
            restore_name="pcp_keypool_hidden_restore",
            stage=b.common["stage"], layer=plan.index,
            requests=b.common["requests"], mode=b.common["phase"],
            dep=deps, local_rows=q, global_rows=kpool_q,
            hidden=h, pcp=int(parallel.pcp),
            act_dtype=norm_dtype, runtime=runtime,
            source="vllm_ascend.attention.indexer_kpool",
            stream=CP_DECODE_COMM_STREAM,
            per_peer_rows=kpool_per_peer,
        )
        kpool_gathered = (kpool_source, kpool_views, kpool_q)
    v4_hidden_gathered = (
        pcp_prefill and plan.cache_kind == "window_compressed"
        and plan.compress_ratio != 2 and not is_v41
    )
    if v4_hidden_gathered:
        cache_views = tuple(
            replace(v, q_len=int(v.global_q_len or v.q_len),
                    query_ranges=((0, int(v.global_q_len or v.q_len)),))
            if v.phase == "prefill" else v
            for v in views
        )
        v4_per_peer, cache_q = pcp_uniform_peer_rows(views, parallel.pcp, local_rows=q)
        cache_hidden = emit_pcp_hidden_gather_restore(
            b.ops, prefix=b.prefix, gather_name="pcp_hidden_gather",
            restore_name="pcp_hidden_restore", stage=b.common["stage"],
            layer=plan.index, requests=b.common["requests"],
            mode=b.common["phase"], dep=deps, local_rows=q,
            global_rows=cache_q, hidden=h, pcp=parallel.pcp,
            act_dtype=model.activation_dtype, runtime=runtime,
            source="vllm_ascend.attention.context_parallel.dsa_cp",
            per_peer_rows=v4_per_peer,
        )
    else:
        cache_views, cache_q, cache_hidden = views, q, deps
    if plan.q_lora_rank:
        if v4_hidden_gathered:
            q_a = b.matrix("q_a", (deps,), q, plan.q_lora_rank, h, category="mla.q_a")
            kv_a = b.matrix("kv_a", (cache_hidden,), cache_q, kv_width, h, category="mla.kv_a")
        else:
            projected = mla_input_projections(
                b, plan=plan, tokens=q, hidden=h, kv_width=kv_width, deps=deps,
                packed=prolog is None)
            q_a, kv_a = projected["q_a"], projected["kv_a"]
        q_norm = b.vector(
            "q_norm", (q_a,), q * plan.q_lora_rank, RMSNORM_OPS_PER_ELEM,
            family="rmsnorm", shape=(q, plan.q_lora_rank),
            read_bytes_override=q * plan.q_lora_rank * norm_bytes
                                + plan.q_lora_rank * norm_bytes,
            write_bytes_override=q * plan.q_lora_rank * norm_bytes,
            metadata=(("reduction_width", plan.q_lora_rank),
                      ("norm_has_weight", True),
                      ("norm_weight_dtype", norm_dtype),
                      ("norm_gamma_runtime_bytes", plan.q_lora_rank * norm_bytes),
                      ("norm_eps", model.norm_eps)))
        q_b = q_ready = b.matrix("q_b", (q_norm,), q, proj_heads * width, plan.q_lora_rank,
                                category="mla.q_b")
    else:
        if prolog:
            raise ModelingError("mla_prolog_v3_direct_q_unsupported")
        q_a = q_norm = q_b = q_ready = b.matrix(
            "q_proj", (deps,), q, proj_heads * width, h, category="mla.q_proj")
        kv_a = b.matrix("kv_a", (cache_hidden,), cache_q, kv_width, h, category="mla.kv_a")
    kv = b.vector(
        "kv_norm", (kv_a,), cache_q * plan.kv_lora_rank, RMSNORM_OPS_PER_ELEM,
        family="rmsnorm", shape=(cache_q, plan.kv_lora_rank),
        read_bytes_override=cache_q * plan.kv_lora_rank * norm_bytes
                            + plan.kv_lora_rank * norm_bytes,
        write_bytes_override=cache_q * plan.kv_lora_rank * norm_bytes,
        metadata=(("reduction_width", plan.kv_lora_rank),
                  ("norm_has_weight", True),
                  ("norm_weight_dtype", norm_dtype),
                  ("norm_gamma_runtime_bytes", plan.kv_lora_rank * norm_bytes),
                  ("norm_eps", model.norm_eps)))
    if not do_rope:
        rotary = kv
    elif partial_rotary:
        q_ready = partial_rotary.module.declare(
            b, name="q_rotary", deps=(q_ready,), tokens=q, heads=post_q_heads,
            head_dim=width, rotary_width=plan.qk_rope_head_dim)
        rotary = partial_rotary.module.declare(
            b, name="kv_rotary", deps=(kv,), tokens=cache_q, heads=1,
            head_dim=plan.kv_lora_rank, rotary_width=plan.qk_rope_head_dim)
    else:
        rotary = b.vector("rotary", (kv, q_ready),
                          (q * post_q_heads + cache_q) * plan.qk_rope_head_dim, ROPE_OPS)
    valid = sum(view.q_len for view in views)
    store, produced = _produce_mla_cache(
        b, cache_views, parallel, own_block, rotary, kv, kv_a, do_rope, prolog,
        native_cache_epilogs, pcp_prefill and not v4_hidden_gathered,
        is_v41=is_v41, padded_tokens=q)
    if latent:
        q_ready = b.matrix("q_absorb", (q_ready,), q * post_q_heads, plan.kv_lora_rank,
                           plan.qk_nope_head_dim, weight_count=post_q_heads,
                           category="mla_absorb",
                           metadata=batched_matrix_metadata(
                               tokens=q, heads=post_q_heads, input_width=plan.qk_nope_head_dim,
                               output_width=plan.kv_lora_rank, role="key_absorption"))
    if prolog:
        prolog.module.declare(
            b.ops, prefix=b.prefix, plan=plan, model=model, parallel=parallel, quant=quant,
            tokens=q, members=tuple(dict.fromkeys(
                (q_a, q_norm, q_b, kv_a, kv, rotary, produced, q_ready))),
            q_a=q_a, q_b=q_b, kv_a=kv_a, q_absorb=q_ready, **prolog.options)
    elif plan.q_lora_rank:
        from mindie_sim.prediction.operators.mla_prolog import family_mla_prolog
        rt_meta = dict(getattr(runtime, "metadata", ()) or ())
        if rt_meta.get("raw_eager") or (rt_meta.get("execution_policy") or {}).get("raw_eager"):
            reason = "raw_eager_unfused_mla_prolog"
        elif plan.mla_variant != "latent_mla":
            reason = f"mla_variant_{plan.mla_variant}_unfused"
        elif plan.cache_kind != "dense":
            reason = f"cache_kind_{plan.cache_kind}_unfused"
        elif plan.q_lora_rank not in family_mla_prolog.MLA_PROLOG_QUERY_LORA_RANKS:
            reason = f"query_lora_rank_{plan.q_lora_rank}_outside_v3_v4_domain"
        else:
            reason = "mla_prolog_lowering_not_selected"
        family_mla_prolog.declare_semantic(
            b.ops, prefix=b.prefix, plan=plan, model=model, parallel=parallel, quant=quant,
            tokens=q,
            members=(q_a, q_norm, q_b, kv_a, kv, rotary, produced, q_ready),
            q_a=q_a, q_b=q_b, kv_a=kv_a,
            q_absorb=q_ready if latent else None,
            anchor=q_ready, reason=reason)
    q_local, rotary_local = q_ready, rotary
    q_dcp, rotary_dcp, dcp_heads = q_local, rotary_local, (proj_heads if decode_dcp else heads)
    if decode_dcp and (tp_gather > 1 or dp_width > 1):
        gather_w = (plan.kv_lora_rank + plan.qk_rope_head_dim) if latent else width
        has_pf = any(v.phase == "prefill" for v in views)
        dcp_q_rows = (
            sum(int(v.q_len) for v in views if v.phase != "prefill")
            if (has_pf and not is_sparse) else q
        )
        gathered, dcp_heads, _ = emit_dcp_query_gather(
            b.ops, prefix=b.prefix, stage=b.common["stage"], layer=plan.index,
            requests=b.common["requests"], mode=b.common["phase"],
            deps=(q_local, rotary_local) if latent else (q_local,),
            rows=max(1, dcp_q_rows), local_heads=heads, head_width=gather_w,
            act_dtype=model.activation_dtype, tp_gather=tp_gather,
            dp_width=dp_width, pack_inputs=latent, q_replicate=q_replicate,
            stream=CP_DECODE_COMM_STREAM if is_sparse else "",
            peer_rows=peer_rows, local_dp_pos=local_dp_pos,
            pcp_factor=pcp_factor,
        )
        if not q_replicate:
            q_dcp, rotary_dcp = gathered, gathered
    return (q_local, rotary_local, q_dcp, rotary_dcp, q_norm, valid, pcp_prefill,
            store, dcp_heads, cache_hidden, cache_views, cache_q, v4_hidden_gathered,
            kpool_gathered)


def _produce_v41_pcp_long_cache(
    b, views, deps, valid, q, parallel, native_cache_epilogs, runtime,
):
    """Exact ``dsa_v41_cp.py:662-743`` pre-pool C2 ring gather -> pool -> latent gather -> store."""
    from mindie_sim.prediction.graph import GraphPolicy
    from mindie_sim.prediction.operators.mla_prolog.compression import (
        COMPRESSION_RECIPE,
        FP32_BYTES,
        NORM_VECTOR_OPS_PER_ELEMENT,
        POOL_VECTOR_OPS_PER_ELEMENT,
        ROTARY_VECTOR_OPS_PER_ELEMENT,
    )
    plan, model = b.plan, b.model
    ratio = max(1, plan.compress_ratio)
    width = plan.head_dim
    expansion, hidden = plan.compressor_width, model.hidden_size
    act_dtype = model.activation_dtype
    act_bytes = dtype_bytes(act_dtype)
    local_comp_rows = max(1, sum(owned_completed_rows(view, ratio) for view in views))
    global_comp_rows = sum(
        completed_rows(view.context + (view.global_q_len or view.q_len), ratio)
        - completed_rows(view.context, ratio)
        for view in views if view.phase == "prefill"
    )
    global_comp_rows = max(global_comp_rows, local_comp_rows)
    common_peer_rows, global_token_rows = pcp_uniform_peer_rows(
        views, parallel.pcp, local_rows=valid,
    )
    if ratio == 2:
        fp32_kwargs = {
            "category": "compressor_fp32_runtime",
            "activation_dtype": "fp32",
            "output_dtype": "fp32",
            "metadata": (("compute_format", "fp32"),),
        }
        projection = b.matrix(
            "compress_values", (deps,), q, expansion * width, hidden, **fp32_kwargs)
        gates = b.matrix(
            "compress_gates", (deps,), q, expansion * width, hidden, **fp32_kwargs)
        actual_scope = (
            GraphPolicy().select(views, runtime)[1] if runtime is not None else "none"
        )
        full_capture = actual_scope == "full"
        ring_gather_id = emit_v41_pcp_c2_ring_gather(
            b.ops, prefix=b.prefix, stage=b.common["stage"],
            layer=plan.index, requests=b.common["requests"],
            mode=b.common["phase"], deps=(projection, gates), views=views,
            pcp=parallel.pcp, compressor_width=expansion,
            head_dim=width, min_send_rows=(common_peer_rows if full_capture else 0))
        expected_gather_id = f"{b.prefix}.pcp_c2_ring_tail_gather"
        has_gather = ring_gather_id == expected_gather_id
        pool_deps = (projection, gates, ring_gather_id) if has_gather else (projection, gates)
        pooled = b.vector(
            "compress_pool", pool_deps, valid * expansion * width,
            POOL_VECTOR_OPS_PER_ELEMENT,
            read_bytes_override=(valid + min(valid, plan.compressor_state_rows))
                                * expansion * width * FP32_BYTES * 2
                                + ratio * expansion * width * FP32_BYTES,
            write_bytes_override=max(1, valid) * width * act_bytes,
            metadata=(("state_dtype", "fp32"), ("ape_dtype", "fp32"),
                      ("valid_completed_rows", local_comp_rows)),
        )
        ring_write_dep = (
            emit_v41_pcp_c2_ring_write(
                b.ops, prefix=b.prefix, stage=b.common["stage"],
                layer=plan.index, requests=b.common["requests"],
                mode=b.common["phase"], deps=(ring_gather_id, pooled),
                views=views, compressor_width=expansion, head_dim=width,
                num_ring_rows=(global_token_rows if full_capture else 0))
            if has_gather else None
        )
    else:
        projection = b.matrix(
            "compress_values", (deps,), q, expansion * width, hidden,
            category=COMPRESSION_RECIPE)
        ring_write_dep = None
        pooled = projection
    local_latent = b.vector(
        "compress_norm", (pooled,), local_comp_rows * width,
        NORM_VECTOR_OPS_PER_ELEMENT, family="rmsnorm",
        shape=(local_comp_rows, width),
        read_bytes_override=local_comp_rows * width * act_bytes
                            + width * dtype_bytes(plan.compressor_norm_dtype),
        write_bytes_override=local_comp_rows * width * act_bytes,
        metadata=(("reduction_width", width), ("norm_has_weight", True),
                  ("norm_weight_dtype", plan.compressor_norm_dtype),
                  ("norm_eps", model.norm_eps)),
    )
    # Upstream ``dsa_v41_cp.py:656-660,742`` pads token-aligned latent rows to
    # ``local_num_tokens_after_padding`` across PCP ranks, AllGathers, then
    # restores ``hidden_restore_idx`` (``global_token_rows``).
    per_peer_latent = max(1, common_peer_rows)
    global_latent_rows = max(1, global_token_rows)
    latent_row_b = float(width) * act_bytes
    gathered = b.communicate(
        "pcp_compressed_kv_gather", (local_latent,),
        float(per_peer_latent) * latent_row_b * float(parallel.pcp - 1),
        "allgather", "pcp", dtype=act_dtype,
        shape=(per_peer_latent * int(parallel.pcp), width),
        source="vllm_ascend.attention.context_parallel.dsa_v41_cp")
    restored_latent = emit_pcp_restore_unpermute(
        b.ops, prefix=b.prefix, name="pcp_compressed_kv_restore",
        stage=b.common["stage"], layer=plan.index,
        requests=b.common["requests"], mode=b.common["phase"],
        dep=gathered, global_rows=global_latent_rows,
        row_bytes=latent_row_b, dtype=act_dtype)
    rotated = restored_latent
    if plan.mla_do_rope:
        rotated = b.vector(
            "compress_rotary", (restored_latent,),
            global_comp_rows * plan.qk_rope_head_dim,
            ROTARY_VECTOR_OPS_PER_ELEMENT, family="rotary",
            shape=(global_comp_rows, width),
            read_bytes_override=global_comp_rows * (
                width * act_bytes + plan.qk_rope_head_dim * FP32_BYTES * 2),
            write_bytes_override=global_comp_rows * width * act_bytes,
        )
    store_deps = (rotated, ring_write_dep) if ring_write_dep else (rotated,)
    long_store = representation_store(
        b, "compressed_store", store_deps, global_comp_rows,
        plan.cache_row_elements, "compressed_kv")
    if long_store is not None:
        compressed_fmt = cache_format(model, "compressed_kv")
        for position, item in enumerate(b.ops):
            if item.op_id == long_store:
                written = kv_cache_epilog(
                    item, compressed_fmt, "compressed_kv",
                    native=native_cache_epilogs)
                b.ops[position] = native_cache_store(
                    written, native=native_cache_epilogs)
                long_store = b.ops[position].op_id
                break
    return long_store, restored_latent


def _produce_long_cache(
    b, store, stage_of, views, deps, valid, q, native_fusion, parallel, own_block,
    native_cache_epilogs, pcp_prefill, cache_ids, cp, *, runtime=None, is_v41=False,
):
    plan, model = b.plan, b.model
    long_store = store if plan.cache_kind == "dense" else None
    compressed_latent = None

    def owner(mapping, layer, name, volume):
        if layer not in mapping:
            raise ModelingError(f"producer_missing:{b.prefix}:{name}:{layer}")
        ready = mapping[layer]
        producer_stage = stage_of(layer)
        if producer_stage != b.common["stage"]:
            ready = b.communicate(name + "_transfer", (ready,), volume, "cache_transfer", "pp",
                                  source_stage=producer_stage, destination_stage=b.common["stage"])
        return ready

    ratio = max(1, plan.compress_ratio)
    if plan.cache_kind == "window_compressed" and plan.compress_ratio:
        compressed_rows = [owned_completed_rows(view, ratio) for view in views]
        if plan.owns_long_kv:
            if is_v41 and pcp_prefill and int(parallel.pcp) > 1:
                return _produce_v41_pcp_long_cache(
                    b, views, deps, valid, q, parallel, native_cache_epilogs, runtime)
            new_rows = sum(compressed_rows)
            compressed_latent, rotated = compressor(
                b, name="compress", source=deps, tokens=valid, padded_tokens=q,
                valid_rows=new_rows, width=plan.head_dim, plan=plan, model=model,
                return_latent=True, fused=native_fusion)
            compressed_starts = [completed_rows(view.context, ratio) for view in views]
            long_store = representation_store(
                b, "compressed_store", (rotated,),
                _store_rows(compressed_rows, views, parallel, own_block,
                            contexts=compressed_starts) if new_rows else 0,
                plan.cache_row_elements, "compressed_kv")
            if long_store is not None:
                compressed_fmt = cache_format(model, "compressed_kv")
                for position, item in enumerate(b.ops):
                    if item.op_id == long_store:
                        written = kv_cache_epilog(
                            item, compressed_fmt, "compressed_kv",
                            native=native_cache_epilogs)
                        b.ops[position] = native_cache_store(
                            written, native=native_cache_epilogs)
                        long_store = b.ops[position].op_id
                        break
            if pcp_prefill:
                global_comp_rows = sum(
                    completed_rows(view.context + (view.global_q_len or view.q_len), ratio)
                    - completed_rows(view.context, ratio)
                    for view in views if view.phase == "prefill")
                if global_comp_rows > 0:
                    comp_fmt = cache_format(model, "compressed_kv")
                    comp_row_bytes = packed_region_row_bytes(
                        model, "compressed_kv", plan.cache_row_elements)
                    long_store = b.communicate(
                        "pcp_compressed_kv_gather", (long_store,),
                        global_comp_rows * comp_row_bytes
                        * (parallel.pcp - 1) / parallel.pcp,
                        "allgather", "pcp", dtype=comp_fmt.dtype)
                    long_store = emit_pcp_restore_unpermute(
                        b.ops, prefix=b.prefix, name="pcp_compressed_kv_restore",
                        stage=b.common["stage"], layer=plan.index,
                        requests=b.common["requests"], mode=b.common["phase"],
                        dep=long_store, global_rows=global_comp_rows,
                        row_bytes=comp_row_bytes, dtype=comp_fmt.dtype)
        else:
            rows = sum(completed_rows(view.kv_len, ratio) for view in views)
            long_store = owner(cache_ids, plan.kv_owner_layer_id, "compressed_cache",
                               rows * packed_region_row_bytes(
                                   model, "compressed_kv", plan.cache_row_elements) / max(1, cp))
    return long_store, compressed_latent


def _append_mla_request(
    rec, grp_views, grp_dsa, eff_cp, worst_pairs_r, worst_sweeps_r, kv_row_b, *, b, runtime,
    parallel, quant, local_heads, dcp_heads, dp_width, qk_dim, latent_dim, latent,
    query_tile, hierarchy, topk, q_local, rotary_local, q_dcp, rotary_dcp,
    store, long_store, metadata_ready, machine, head_contract, offload_info,
):
    plan, model = b.plan, b.model
    mode = b.common["phase"]
    orig_idx = rec["orig_idx"]
    view = rec["view"]
    eff_q = rec["eff_q"]
    parts = rec["parts"]
    pairs, sweeps = rec["pairs"], rec["sweeps"]
    local_pairs = rec["v_pairs"][worst_pairs_r]
    local_sweeps = rec["v_sweeps"][worst_sweeps_r]
    local_rows = rec["v_rows"][worst_pairs_r]
    attn_heads = (plan.num_heads if grp_dsa and view.phase == "prefill"
                  else (dcp_heads if eff_cp > 1 else local_heads))
    q_ready = q_dcp if eff_cp > 1 else q_local
    rotary = rotary_dcp if eff_cp > 1 else rotary_local
    q_eff = eff_q
    q_read = q_eff * attn_heads * qk_dim * dtype_bytes(model.activation_dtype)
    out_write = q_eff * attn_heads * latent_dim * dtype_bytes(model.activation_dtype)
    local_cache_read = window_cache_read(
        plan, view, parts, sweeps, eff_cp, query_tile,
        plan.cache_row_elements, model)
    traffic = plan_attention_traffic(
        AttentionAccessSpec(
            q_lengths=(int(q_eff),),
            context_lengths=(int(max(0, view.kv_len - eff_q)),),
            unique_rows=int(max(0, local_rows)),
            refill_rows=int(max(0, local_sweeps)),
            selected_rows=int(max(0, local_sweeps)) if topk else 0,
            row_bytes=float(kv_row_b),
            phase=mode,
            topk=int(plan.index_topk) if topk else 0,
            representation=("absorbed_mla" if latent else "gqa"),
            cache_identity=str(getattr(view, "prefix_key", "") or ""),
        ),
        hierarchy,
        AttentionSchedule(query_tile=query_tile),
    )
    if local_cache_read is None:
        local_cache_read = traffic.hbm_read_bytes
    elif local_sweeps > 0:
        traffic = plan_attention_traffic(
            AttentionAccessSpec(
                q_lengths=(int(q_eff),),
                context_lengths=(int(max(0, view.kv_len - eff_q)),),
                unique_rows=int(max(0, local_rows)),
                refill_rows=int(max(0, local_sweeps)),
                selected_rows=(int(max(0, local_sweeps))
                               if topk else 0),
                row_bytes=float(local_cache_read) / float(local_sweeps),
                phase=mode,
                topk=int(plan.index_topk) if topk else 0,
                cache_identity=str(getattr(view, "prefix_key", "") or ""),
            ),
            hierarchy,
            AttentionSchedule(query_tile=query_tile),
        )
        local_cache_read = traffic.hbm_read_bytes
    vis_keys = (
        min(local_rows, plan.window) if (plan.window > 0 and eff_cp == 1)
        else local_sweeps if (plan.cache_kind == "dense" and not topk
                              and eff_q == 1)
        else local_rows
    )
    kv_hbm_component = float(local_cache_read)
    kv_mte2_component = float(traffic.private_refill_bytes)
    is_generic_decode = (
        latent and view.phase == "decode" and eff_q == 1
        and plan.cache_kind == "dense" and not topk and not plan.indexer
        and model.activation_dtype == "bf16" and quant.kv_dtype == "bf16"
    )
    last = b.work(
        f"latent_attention{orig_idx}", tuple(dict.fromkeys(
            item for item in (q_ready, rotary if latent else None, store,
                              long_store, topk, metadata_ready) if item)),
        (q_eff, view.kv_len, attn_heads, qk_dim, latent_dim),
        flops=2 * local_pairs * attn_heads * (qk_dim + latent_dim),
        vector=SOFTMAX_VECTOR_OPS_PER_PAIR * local_pairs * attn_heads,
        read=kv_mte2_component + rec["selection_reads"] + q_read,
        hbm_read=kv_hbm_component + rec["selection_reads"] + q_read,
        write=out_write, family="attention",
        metadata=(("query_tile", query_tile), ("logical_pairs", pairs),
                  ("kv_sweeps", sweeps),
                  ("attention_traffic_plan", traffic.to_dict()))
                 + rec["selection_metadata"]
                 + attention_tensor_view(plan, view, attn_heads, model, quant,
                                         parallel, eff_cp=eff_cp,
                                         machine=machine)
                 + ((("program_kind", "dense_latent_attention_generic"),)
                    if is_generic_decode else ())
                 + ((("latent_attention_schedule", {
                     "heads": attn_heads, "query_rows": q_eff,
                     "keys": max(0, vis_keys),
                     "qk_dim": qk_dim, "latent_dim": latent_dim,
                     "dv": int(latent_dim),
                     "q_bytes": q_read,
                     "kv_row_bytes": kv_row_b,
                     "kv_unique_bytes": kv_row_b * max(0, vis_keys),
                     "output_bytes": out_write,
                     "owners": max(1, min(
                         MAX_GENERIC_OWNERS,
                         math.ceil(max(1, vis_keys) / DEFAULT_QUERY_TILE))),
                     "tile_keys": DEFAULT_QUERY_TILE,
                     "page_size": int(getattr(runtime, "kv_block_size", 0)
                                      or DEFAULT_BLOCK_SIZE),
                     "source_scope": "generic_online_attention",
                 }),) if is_generic_decode else ())
                 + ((("mla_head_contract", dict(head_contract)),)
                    if head_contract else ())
                 + ((("sparse_kv_offload", offload_info),) if offload_info else ()),
    )
    b.ops[-1] = replace(
        b.ops[-1],
        local_shape=(q_eff, local_rows, attn_heads, qk_dim, latent_dim),
    )
    if offload_info and offload_info.get("use_fused_overlap"):
        b.ops[-1] = replace(
            b.ops[-1],
            recipe="fused_sparse_attention_overlap",
            source_boundary="kernel",
            metadata=b.ops[-1].metadata + (("sparse_offload_sfa", offload_info),),
        )
    if (runtime and view.phase == "prefill" and topk
            and plan.cache_kind == "window_compressed"
            and plan.mla_variant == "latent_mqa"):
        sharedkv = select(runtime.kernels.get("lowerings", {}), "sharedkv_attention")
        if sharedkv:
            b.ops[-1] = sharedkv.module.lower_op(
                b.ops[-1], selection=sharedkv, view=view, plan=plan,
                views=grp_views, heads=attn_heads, parallel=parallel,
                model=model, quant=quant)
    if runtime and topk and plan.cache_kind == "dense":
        indexed = select(runtime.kernels.get("lowerings", {}), "indexed_attention")
        if indexed:
            b.ops[-1] = indexed.module.lower_op(
                b.ops[-1], selection=indexed, view=view, views=grp_views,
                plan=plan, parallel=parallel)
    return last, kv_hbm_component, kv_mte2_component


def _batch_mla_attention(
    b, grp_outputs, grp_views, grp_prefix, plan, parallel, runtime, mixed_quant, topk, selected,
    pcp_prefill, eff_cp, paired_sfa_mode, pcp_factor, tp_gather, dp_width, policy,
    heads, logical_heads, q, kv_row_b, kv_hbm_total,
    kv_mte2_total, hierarchy, sweeps_by_view, shared_by_view, view_records, kv_components,
    *, peer_rows=None, local_dp_pos=0,
):
    if mixed_quant:
        grp_outputs = mixed_quant.module.lower_attention(
            b.ops, grp_outputs, selection=mixed_quant, views=grp_views, plan=plan,
            topk_present=bool(topk), prefix=grp_prefix, parallel=parallel,
            runtime=runtime)
    elif not selected:
        from .generic_latent_attention import batch_general_mla_ops, batch_ops
        from mindie_sim.prediction.operators.paged_mla.indexed_attention import batch_analytic_ops
        if plan.cache_kind == "dense" and not pcp_prefill:
            grp_outputs = batch_ops(
                b.ops, grp_outputs, runtime=runtime, views=grp_views,
                prefix=grp_prefix)
            if eff_cp == 1:
                grp_outputs = batch_analytic_ops(
                    b.ops, grp_outputs, runtime=runtime, views=grp_views,
                    prefix=grp_prefix)
        if not paired_sfa_mode:
            grp_outputs = batch_general_mla_ops(
                b.ops, grp_outputs, views=grp_views, prefix=grp_prefix,
                runtime=runtime, policy=policy)
        from mindie_sim.prediction.implementation_selection import is_caller_pinned_binding
        pinned_attn = (
            is_caller_pinned_binding(runtime, "families", "attention")
            or any(
                is_caller_pinned_binding(runtime, "recipes", item)
                for item in ("latent_attention0", "latent_attention_batch")
            )
        )
        if paired_sfa_mode:
            grp_outputs = paired_sfa_attention_batch(
                b, grp_outputs, grp_views, plan=plan, heads=heads,
                local_heads=logical_heads, cp=eff_cp, rows=q, runtime=runtime,
                pcp_factor=pcp_factor, tp_gather=tp_gather, dp_width=dp_width,
                peer_rows=peer_rows, local_dp_pos=local_dp_pos)
        out_set = ({b.prefix + ".latent_attention_batch"} if paired_sfa_mode
                   else set(grp_outputs))
        for idx_op, op_item in enumerate(b.ops):
            if op_item.op_id not in out_set or op_item.source_boundary:
                continue
            if topk:
                b.ops[idx_op] = (
                    apply_sparse_prefix_reads(
                        op_item, views=grp_views, row_bytes=float(kv_row_b),
                        runtime=runtime, hierarchy=hierarchy,
                        kv_hbm_bytes=kv_hbm_total,
                        kv_mte2_bytes=kv_mte2_total)
                    if len(grp_outputs) == 1 and eff_cp == 1
                    and plan.cache_kind == "dense"
                    else annotate_sparse_prefix_unknown(op_item, runtime))
            elif len(grp_outputs) == 1:
                read_mode = str(getattr(runtime, "prefix_read_mode", "auto") or "auto")
                reuse_kind = (
                    ("native_per_sequence" if read_mode == "batch_shared" else "paged_l2")
                    if pinned_attn
                    else (
                        "batch_tile"
                        if runtime is not None and runtime.prefix_cache_mode == "shared"
                        else "tile"
                    )
                )
                b.ops[idx_op] = apply_attention_prefix_reads(
                    op_item, views=grp_views, sweeps_by_view=sweeps_by_view,
                    shared_sweeps_by_view=shared_by_view, row_bytes=float(kv_row_b),
                    layer_index=plan.index, runtime=runtime,
                    kernel_reuse=reuse_kind,
                    allow_recipe_rebind=not pinned_attn,
                    kv_hbm_bytes=kv_hbm_total,
                    kv_mte2_bytes=kv_mte2_total,
                    geometry=hierarchy)
            else:
                record = view_records[grp_outputs.index(op_item.op_id)] \
                    if op_item.op_id in grp_outputs else None
                if record is not None:
                    component = kv_components.get(op_item.op_id,
                                                  (0.0, 0.0))
                    b.ops[idx_op] = apply_attention_prefix_reads(
                        op_item, views=(record["view"],),
                        sweeps_by_view=(sweeps_by_view[grp_outputs.index(op_item.op_id)],),
                        shared_sweeps_by_view=(shared_by_view[grp_outputs.index(op_item.op_id)],),
                        row_bytes=float(kv_row_b), layer_index=plan.index,
                        runtime=runtime, kernel_reuse="native_per_sequence",
                        allow_recipe_rebind=False,
                        kv_hbm_bytes=component[0],
                        kv_mte2_bytes=component[1])
    return grp_outputs


def _history_only_pairs(view, eff_cp, interleave, window):
    """Exact history-only KV pairs per DCP rank for MTP dual-stream split."""
    if view.q_len <= 1 or view.context <= 0:
        return None
    hi = view.context - 1
    lo = max(0, view.context - window + 1) if window > 0 else 0
    return [
        float(view.q_len * owned_positions(lo, hi, eff_cp, interleave, rank))
        for rank in range(eff_cp)
    ]


def _append_mla_attention(
    b, outputs, views, runtime, selected, parallel, dsa_prefill, model, plan, quant, query_tile,
    cp_interleave_size, topk, local_heads, dcp_heads, latent_dim, qk_dim, reuse_index_selection,
    request_op, mixed_quant, pcp_prefill, paired_sfa_mode, is_sparse, pcp_factor, tp_gather,
    dp_width, policy, logical_heads, q, hierarchy, stage, mode, q_local, store,
    *, cohort_decode_views=None, peer_decode_rows=None, local_dp_pos=0,
    idle_dcp_owner=False, sharded_indexer=False, physical_rank=None,
    explicit_dp_cohort=False, explicit_physical_rank=False,
    sfa_replicated_prefill_path=False,
):
    block_size = max(1, int(getattr(runtime, "kv_block_size", DEFAULT_BLOCK_SIZE)
                              or DEFAULT_BLOCK_SIZE))
    layer_dcp = layer_dcp_degree(plan, parallel)
    has_dp_decode = (not sfa_replicated_prefill_path and dp_width > 1
                     and layer_dcp > 1 and bool(cohort_decode_views))
    phase_groups: list[tuple[int, list[tuple[int, object]]]] = []
    grp_map: dict[int, list[tuple[int, object]]] = {}
    for index, view in enumerate(() if selected else views):
        if has_dp_decode and view.phase != "prefill":
            continue
        eff_cp = (1 if sfa_replicated_prefill_path else
                  active_layer_dcp(plan, parallel, view.phase, sharded_indexer=sharded_indexer))
        if eff_cp not in grp_map:
            bucket: list[tuple[int, object]] = []
            grp_map[eff_cp] = bucket
            phase_groups.append((eff_cp, bucket))
        grp_map[eff_cp].append((index, view))
    if not selected and has_dp_decode:
        phase_groups.append((layer_dcp, list(enumerate(cohort_decode_views))))
    multi_group = len(phase_groups) > 1
    for grp_idx, (eff_cp, indexed_views) in enumerate(phase_groups):
        grp_views = tuple(view for _, view in indexed_views)
        grp_prefix = f"{b.prefix}.phase{grp_idx}" if multi_group else b.prefix
        grp_dsa = dsa_prefill and any(view.phase == "prefill" for view in grp_views)
        mtp_split = should_split_mtp_dcp(plan, grp_views, layer_dcp=eff_cp, sparse=is_sparse)
        grp_heads = dcp_heads if eff_cp > 1 else local_heads
        rank_pairs = [0.0] * eff_cp
        rank_sweeps = [0] * eff_cp
        rank_rows = [0] * eff_cp
        view_records: list[dict] = []
        for orig_idx, view in indexed_views:
            rec = attention_view_record(
                view, plan=plan, model=model, quant=quant, query_tile=query_tile,
                eff_cp=eff_cp, block=block_size, interleave=cp_interleave_size,
                tp=parallel.tp, dsa=grp_dsa,
                topk_present=bool(topk) or (idle_dcp_owner and is_sparse),
                heads=grp_heads, latent_dim=latent_dim,
                shared_selection=(topk if (reuse_index_selection
                                           and isinstance(topk, SharedSelectionRef))
                                  else None))
            if mtp_split:
                hist = _history_only_pairs(view, eff_cp, cp_interleave_size, int(plan.window or 0))
                if hist is not None:
                    rec = {**rec, "v_pairs": hist, "pairs": float(sum(hist))}
            for rank in range(eff_cp):
                rank_pairs[rank] += rec["v_pairs"][rank]
                rank_sweeps[rank] += rec["v_sweeps"][rank]
                rank_rows[rank] += rec["v_rows"][rank]
            view_records.append({"orig_idx": orig_idx, **rec})
        worst_pairs_r = max(range(eff_cp),
                            key=lambda rank: (rank_pairs[rank], rank_sweeps[rank]))
        worst_sweeps_r = max(range(eff_cp),
                             key=lambda rank: (rank_sweeps[rank], rank_pairs[rank]))
        cand_ranks = resolve_cp_rank_candidates(
            parallel, eff_cp=eff_cp, pcp_factor=pcp_factor, tp_gather=tp_gather,
            dp_width=dp_width, local_dp_pos=local_dp_pos, physical_rank=physical_rank,
            explicit_dp_cohort=explicit_dp_cohort,
            explicit_physical_rank=explicit_physical_rank)
        sel_pairs_r = max(cand_ranks, key=lambda rank: (rank_pairs[rank], rank_sweeps[rank]))
        sel_sweeps_r = max(cand_ranks, key=lambda rank: (rank_sweeps[rank], rank_pairs[rank]))
        grp_outputs: list[str] = []
        sweeps_by_view: list[int] = []
        shared_by_view: list[int] = []
        kv_hbm_total = 0.0
        kv_mte2_total = 0.0
        kv_components: dict[str, tuple[float, float]] = {}
        kv_row_b = cache_format(model, "mla_kv").row_bytes(plan.cache_row_elements)
        for rec in view_records:
            local_sweeps = rec["v_sweeps"][sel_sweeps_r]
            sweeps_by_view.append(local_sweeps)
            shared_by_view.append(min(local_sweeps, rec["v_shared"][sel_sweeps_r]))
            last, kv_hbm_component, kv_mte2_component = request_op(
                rec, grp_views, grp_dsa, eff_cp, sel_pairs_r, sel_sweeps_r, kv_row_b)
            kv_hbm_total += kv_hbm_component
            kv_mte2_total += kv_mte2_component
            kv_components[last] = kv_hbm_component, kv_mte2_component
            grp_outputs.append(last)
        grp_outputs = _batch_mla_attention(
            b, grp_outputs, grp_views, grp_prefix, plan, parallel, runtime, mixed_quant, topk,
            selected, pcp_prefill, eff_cp, paired_sfa_mode, pcp_factor, tp_gather, dp_width,
            policy, grp_heads, logical_heads, q,
            kv_row_b, kv_hbm_total, kv_mte2_total, hierarchy, sweeps_by_view, shared_by_view,
            view_records, kv_components,
            peer_rows=(peer_decode_rows if (dp_width > 1 and eff_cp > 1) else None),
            local_dp_pos=local_dp_pos)
        if eff_cp > 1 and grp_outputs:
            out_set = ({b.prefix + ".latent_attention_batch"} if paired_sfa_mode
                       else set(grp_outputs))
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
                        ("dcp_mtp_split", "history" if mtp_split else "unified"),
                    ))
        if eff_cp > 1 and grp_outputs and not paired_sfa_mode:
            current_dep = None
            if mtp_split:
                current_dep = emit_mtp_current_attention(
                    b.ops, prefix=grp_prefix, stage=stage, layer=plan.index,
                    requests=b.common["requests"], mode=mode,
                    deps=tuple(dict.fromkeys((*grp_outputs, q_local, store))),
                    views=grp_views, local_heads=logical_heads, qk_dim=qk_dim,
                    v_dim=latent_dim, kv_row_bytes=kv_row_b,
                    act_dtype=model.activation_dtype)
            grp_rows = max(1, sum(v.q_len for v in grp_views) if multi_group else max(0, q))
            merged_last = emit_dcp_output_exchange_combine(
                b.ops, prefix=grp_prefix, stage=stage, layer=plan.index,
                requests=b.common["requests"], mode=mode,
                deps=tuple(grp_outputs), rows=grp_rows, local_heads=logical_heads,
                v_dim=latent_dim, act_dtype=model.activation_dtype,
                pcp_factor=pcp_factor, tp_gather=tp_gather, dp_width=dp_width,
                local_current_dep=current_dep,
                peer_rows=(peer_decode_rows if dp_width > 1 else None),
                local_dp_pos=local_dp_pos,
                meta_extra=(("cp_interleave", int(cp_interleave_size)),))
            grp_outputs = [merged_last]
        outputs.extend(grp_outputs)


def _project_mla_output(
    b, logical_heads, latent, outputs, q, plan, head_contract, deps, h, partial_rotary,
    parallel, model, dsa_prefill, *, views=(), runtime=None,
):
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
        outputs = [b.vector("gated_output", (*outputs, gate), q * output_width,
                         GATE_APPLY_OPS_PER_ELEMENT)]
    if partial_rotary and plan.mla_do_rope:
        outputs = [partial_rotary.module.declare(
            b, name="output_inverse_rotary", deps=outputs, tokens=q, heads=heads,
            head_dim=plan.head_dim, rotary_width=plan.qk_rope_head_dim, inverse=True)]
    pcp_shard_o = (parallel.pcp > 1 and pcp_o_proj_weight_sharding_enabled(runtime))
    pcp_is_prefill = any(getattr(v, "phase", "") == "prefill" for v in views)
    o_in_width = output_width
    o_rows = q
    group_parallel, groups = _group_plan(plan, parallel)
    if plan.o_groups:
        if not group_parallel and parallel.tp > 1:
            outputs = [b.communicate(
                "output_gather", outputs,
                q * output_width * dtype_bytes(model.activation_dtype)
                * (parallel.tp - 1),
                "allgather")]
        group_width = (plan.num_heads
                       * (plan.v_head_dim if latent else plan.head_dim) // plan.o_groups)
        eff_groups = groups
        if pcp_shard_o:
            if pcp_is_prefill:
                w_bytes = float(
                    b.quant.matrix_bytes(groups * plan.o_lora_rank, group_width, "mla.o_a")
                    + b.quant.matrix_bytes(h, groups * plan.o_lora_rank, "mla.o_b")
                )
                wg = emit_pcp_o_proj_weight_allgather(
                    b.ops, prefix=b.prefix, stage=b.common["stage"], layer=plan.index,
                    requests=b.common["requests"], mode=b.common["phase"],
                    deps=(deps,), weight_bytes=w_bytes, pcp=parallel.pcp,
                    weight_dtype=model.activation_dtype)
                outputs = [*outputs, wg]
            else:
                if groups % parallel.pcp != 0:
                    raise ModelingError(f"pcp_o_proj_groups_not_divisible:{groups}%{parallel.pcp}")
                eff_groups = groups // parallel.pcp
                if parallel.pcp_shard_decode_requests:
                    o_rows = q * parallel.pcp
                    full_in_w = groups * group_width
                    outputs = [b.communicate(
                        "pcp_o_proj_token_allgather", outputs,
                        q * full_in_w * dtype_bytes(model.activation_dtype) * (parallel.pcp - 1),
                        "allgather", "pcp")]
        first = b.matrix("grouped_output_a", outputs, o_rows * eff_groups, plan.o_lora_rank,
                         group_width, weight_count=eff_groups, category="mla.o_a",
                         metadata=batched_matrix_metadata(
                             tokens=o_rows, heads=eff_groups, input_width=group_width,
                             output_width=plan.o_lora_rank, role="grouped_output"),
                         weight_bytes_override=b.quant.matrix_bytes(
                             eff_groups * plan.o_lora_rank, group_width, "mla.o_a"))
        last = b.matrix("grouped_output_b", (first,), o_rows, h, eff_groups * plan.o_lora_rank,
                        category="mla.o_b")
        if pcp_shard_o and not pcp_is_prefill:
            if parallel.pcp_shard_decode_requests:
                last = b.communicate(
                    "pcp_o_proj_reduce_scatter", (last,),
                    (parallel.pcp - 1) * q * h * dtype_bytes(model.activation_dtype),
                    "reduce_scatter", "pcp")
            else:
                last = b.communicate(
                    "pcp_o_proj_allreduce", (last,),
                    2 * (parallel.pcp - 1) / parallel.pcp * q * h
                    * dtype_bytes(model.activation_dtype), "allreduce", "pcp",
                    shape=(q, h))
    else:
        if pcp_shard_o:
            if pcp_is_prefill:
                w_bytes = float(b.quant.matrix_bytes(h, output_width, "mla.o"))
                wg = emit_pcp_o_proj_weight_allgather(
                    b.ops, prefix=b.prefix, stage=b.common["stage"], layer=plan.index,
                    requests=b.common["requests"], mode=b.common["phase"],
                    deps=(deps,), weight_bytes=w_bytes, pcp=parallel.pcp,
                    weight_dtype=model.activation_dtype)
                outputs = [*outputs, wg]
            else:
                o_in_width = max(1, math.ceil(output_width / parallel.pcp))
                if parallel.pcp_shard_decode_requests:
                    o_rows = q * parallel.pcp
                    outputs = [b.communicate(
                        "pcp_o_proj_token_allgather", outputs,
                        q * output_width * dtype_bytes(model.activation_dtype) * (parallel.pcp - 1),
                        "allgather", "pcp")]
        last = b.matrix("output", outputs, o_rows, h, o_in_width, category="mla.o")
        if pcp_shard_o and not pcp_is_prefill:
            if parallel.pcp_shard_decode_requests:
                last = b.communicate(
                    "pcp_o_proj_reduce_scatter", (last,),
                    (parallel.pcp - 1) * q * h * dtype_bytes(model.activation_dtype),
                    "reduce_scatter", "pcp")
            else:
                last = b.communicate(
                    "pcp_o_proj_allreduce", (last,),
                    2 * (parallel.pcp - 1) / parallel.pcp * q * h
                    * dtype_bytes(model.activation_dtype), "allreduce", "pcp",
                    shape=(q, h))
    if parallel.tp > 1 and (not plan.o_groups or group_parallel):
        if dsa_prefill:
            last = b.communicate(
                "dsa_cp_output_all_gather", (last,),
                (parallel.tp - 1) / parallel.tp * q * h
                * dtype_bytes(model.activation_dtype), "allgather", "tp")
        else:
            last = b.communicate(
                "output_reduce", (last,),
                2 * (parallel.tp - 1) / parallel.tp * q * h
                * dtype_bytes(model.activation_dtype), "allreduce",
                shape=(q, h))
    return last
