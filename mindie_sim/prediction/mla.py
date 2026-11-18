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
from mindie_sim.catalog.cache import completed_rows
from mindie_sim.schema.common import dtype_bytes, ModelingError
from .batch_prefix_attention import (
    annotate_sparse_prefix_unknown, apply_attention_prefix_reads,
    apply_sparse_prefix_reads,
)
from .implementations import select
from .implementation_selection import is_automatic_lowering
from .lowering import Builder
from .compression import compressor
from .mla_cann_epilogs import (
    SharedSelectionRef, block_quant_epilog, kv_cache_epilog, native_cache_store,
    native_fusion_active, lower_index_path,
)
from .mla_native import batched_matrix_metadata
from .memory_traffic import (AttentionAccessSpec, AttentionSchedule,
                             default_hierarchy, plan_attention_traffic)
from .projections import mla_input_projections
from . import mla_head_padding
from .cache_representation import store as representation_store
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


def mla(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
        padded_tokens, cache_ids, topk_ids, index_cache_ids, candidate_ids,
        stage_of, cp_interleave_size=DEFAULT_INTERLEAVE, runtime=None, policy=None,
        machine=None, reuse_index_selection=False, shared_topk=None):
    b = Builder(prefix, plan, stage, views, model, mode)
    hierarchy = _memory_hierarchy(machine)
    b.l2_bytes = hierarchy.effective_l2_bytes
    from . import latent_dcp
    from mindie_sim.catalog.paired_sfa import execution
    gathered_dcp = latent_dcp.enabled(model, parallel, views, runtime)
    cp = max((active_dcp_degree(parallel, view.phase) for view in views), default=1)
    execution_flags = execution(model)
    paired = dict(model.metadata).get("cache_allocator") == "paired_sfa"
    own_block = (int(getattr(runtime, "kv_block_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE)
                 if paired else int(cp_interleave_size))
    index_rope = (int(execution_flags.get("index_rope_dim", plan.qk_rope_head_dim))
                  if plan.mla_do_rope else 0)
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
    # The single ``aclnnCompressor`` is a selected native implementation.  A
    # hardware family alone does not rewrite the default decomposition; the
    # assembly step binds ``lowerings.compressor`` and ``raw_eager`` vetoes it.
    compressor_binding = (select(runtime.kernels.get("lowerings", {}), "compressor")
                          if runtime else None)
    native_fusion = bool(compressor_binding) and native_fusion_active(runtime, machine)
    native_cache_epilogs = native_fusion_active(runtime, machine)
    norm_dtype = model.activation_dtype
    norm_bytes = dtype_bytes(norm_dtype)
    if plan.q_lora_rank:
        projected = mla_input_projections(
            b, plan=plan, tokens=q, hidden=h, kv_width=kv_width, deps=deps,
            packed=prolog is None)
        q_a, kv_a = projected["q_a"], projected["kv_a"]
        # A two-dimensional norm payload lets the activation-quantization plan
        # fuse ``q_norm + q_b.quantize`` into one ``rmsnorm_quantize`` kernel.
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
    kv = b.vector(
        "kv_norm", (kv_a,), q * plan.kv_lora_rank, RMSNORM_OPS_PER_ELEM,
        family="rmsnorm", shape=(q, plan.kv_lora_rank),
        read_bytes_override=q * plan.kv_lora_rank * norm_bytes
                            + plan.kv_lora_rank * norm_bytes,
        write_bytes_override=q * plan.kv_lora_rank * norm_bytes,
        metadata=(("reduction_width", plan.kv_lora_rank),
                  ("norm_has_weight", True),
                  ("norm_weight_dtype", norm_dtype),
                  ("norm_gamma_runtime_bytes", plan.kv_lora_rank * norm_bytes),
                  ("norm_eps", model.norm_eps)))
    if not do_rope:
        # The declared NoPE tail is a learned positional payload already carried
        # by the normalized latent; no rotary operator executes.
        rotary = kv
    elif partial_rotary:
        q_ready = partial_rotary.module.declare(
            b, name="q_rotary", deps=(q_ready,), tokens=q, heads=heads,
            head_dim=width, rotary_width=plan.qk_rope_head_dim)
        rotary = partial_rotary.module.declare(
            b, name="kv_rotary", deps=(kv,), tokens=q, heads=1,
            head_dim=plan.kv_lora_rank, rotary_width=plan.qk_rope_head_dim)
    else:
        rotary = b.vector("rotary", (kv, q_ready),
                          q * (heads + 1) * plan.qk_rope_head_dim, ROPE_OPS)
    valid = sum(view.q_len for view in views)
    pcp_prefill = parallel.pcp > 1 and any(view.phase == "prefill" for view in views)
    store_rows = _store_rows([view.q_len for view in views], views, parallel, own_block)
    # A NoPE row carries the normalized latent plus the learned positional
    # tail projected by ``kv_a``; a rotary row carries the rotated latent only
    # (which already depends on ``kv``).
    cache_deps = ((rotary,) if do_rope else (kv, kv_a))
    packed_store = (prolog.module.cache_store(
        b, deps=cache_deps, rows=store_rows, plan=plan, **prolog.options)
        if prolog else None)
    cache_region = "window" if plan.cache_kind == "window_compressed" else "mla_kv"
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
    # The packed prolog writer and a plain latent row bypass the grouped
    # epilog wrapper above; on a CANN graph they are still native stores.
    if store is not None and native_cache_epilogs:
        for position, item in enumerate(b.ops):
            if item.op_id != store:
                continue
            b.ops[position] = native_cache_store(item, native=True)
            store = b.ops[position].op_id
            break
    # This lowering writes the local KV row. Shared long-KV consumers still do.
    produced, owns_kv = store, store is not None
    if pcp_prefill and owns_kv:
        rows = sum(view.global_q_len or view.q_len for view in views if view.phase == "prefill")
        if packed_store is not None:
            from .mla_prolog_packed import view as packed_kv_view
            kv_row = packed_kv_view(plan.kv_lora_rank, plan.qk_rope_head_dim)["row_bytes"]
        else:
            # The gather moves stored rows, including their declared scale or
            # packed suffix, not a bare activation-dtype payload.
            kv_row = cache_format(model, cache_region).row_bytes(
                plan.cache_row_elements)
        store = b.communicate(
            "pcp_kv_gather", (produced,),
            rows * kv_row * (parallel.pcp - 1) / parallel.pcp,
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
            tokens=q, members=tuple(dict.fromkeys(
                (q_a, q_norm, q_b, kv_a, kv, rotary, produced, q_ready))),
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

    ratio = max(1, plan.compress_ratio)
    if plan.cache_kind == "window_compressed" and plan.compress_ratio:
        compressed_rows = [owned_completed_rows(view, ratio) for view in views]
        if plan.owns_long_kv:
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
                    comp_row_bytes = cache_format(
                        model, "compressed_kv").row_bytes(plan.cache_row_elements)
                    long_store = b.communicate(
                        "pcp_compressed_kv_gather", (long_store,),
                        global_comp_rows * comp_row_bytes
                        * (parallel.pcp - 1) / parallel.pcp,
                        "allgather", "pcp")
        else:
            rows = sum(completed_rows(view.kv_len, ratio) for view in views)
            long_store = owner(cache_ids, plan.kv_owner_layer_id, "compressed_cache",
                               rows * cache_format(model, "compressed_kv").row_bytes(
                                   plan.cache_row_elements) / max(1, cp))

    pool_binding = (select(runtime.kernels.get("lowerings", {}), "pool_key_indexer")
                    if runtime and plan.index_kpool > 1 else None)
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
        fused_compressor=native_fusion)
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
        if cp > 1 and not gathered_dcp:
            tot_rows = sum(view.kv_len for view in views)
            merged, sel_out = dcp_partial_merge_ops(
                prefix=prefix, views=views, cp=cp, local_heads=heads,
                v_dim=latent_dim, stage=stage, layer=plan.index, mode=mode,
                deps=(sel_out,), total_pairs=0.0, local_pairs=0.0,
                total_rows=tot_rows, local_rows=math.ceil(tot_rows / cp),
                interleave=cp_interleave_size)
            b.ops.extend(merged)
        outputs.append(sel_out)
    block_size = max(1, int(getattr(runtime, "kv_block_size", DEFAULT_BLOCK_SIZE)
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
    multi_group = len(phase_groups) > 1
    for grp_idx, (eff_cp, indexed_views) in enumerate(phase_groups):
        grp_views = tuple(view for _, view in indexed_views)
        grp_prefix = f"{prefix}.phase{grp_idx}" if multi_group else prefix
        grp_dsa = dsa_prefill and any(view.phase == "prefill" for view in grp_views)
        rank_pairs = [0.0] * eff_cp
        rank_sweeps = [0] * eff_cp
        rank_rows = [0] * eff_cp
        view_records: list[dict] = []
        for orig_idx, view in indexed_views:
            rec = attention_view_record(
                view, plan=plan, model=model, quant=quant, query_tile=query_tile,
                eff_cp=eff_cp, block=block_size, interleave=cp_interleave_size,
                tp=parallel.tp, dsa=grp_dsa, topk_present=bool(topk),
                heads=heads, latent_dim=latent_dim,
                shared_selection=(topk if (reuse_index_selection
                                           and isinstance(topk, SharedSelectionRef))
                                  else None))
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
        total_pairs_sum = 0.0
        total_rows_sum = 0
        kv_hbm_total = 0.0
        kv_mte2_total = 0.0
        kv_components: dict[str, tuple[float, float]] = {}
        kv_row_b = cache_format(model, "mla_kv").row_bytes(plan.cache_row_elements)
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
                        q_lengths=(int(eff_q),),
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
            kv_hbm_total += kv_hbm_component
            kv_mte2_total += kv_mte2_component
            is_generic_decode = (
                latent and view.phase == "decode" and eff_q == 1
                and plan.cache_kind == "dense" and not topk and not plan.indexer
                and model.activation_dtype == "bf16" and quant.kv_dtype == "bf16"
            )
            last = b.work(
                f"latent_attention{orig_idx}", tuple(dict.fromkeys(
                    item for item in (q_ready, rotary if latent else None, store,
                                      long_store, topk, metadata_ready) if item)),
                (eff_q, view.kv_len, attn_heads, qk_dim, latent_dim),
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
                             "heads": attn_heads, "query_rows": eff_q,
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
            kv_components[b.ops[-1].op_id] = (
                kv_hbm_component, kv_mte2_component)
            b.ops[-1] = replace(
                b.ops[-1],
                local_shape=(eff_q, local_rows, attn_heads, qk_dim, latent_dim),
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
            grp_outputs.append(last)
        if mixed_quant:
            grp_outputs = mixed_quant.module.lower_attention(
                b.ops, grp_outputs, selection=mixed_quant, views=grp_views, plan=plan,
                topk_present=bool(topk), prefix=grp_prefix, parallel=parallel)
        elif not selected:
            from .generic_latent_attention import batch_general_mla_ops, batch_ops
            from .indexed_attention import batch_analytic_ops
            if plan.cache_kind == "dense" and not pcp_prefill:
                grp_outputs = batch_ops(
                    b.ops, grp_outputs, runtime=runtime, views=grp_views,
                    prefix=grp_prefix)
                if eff_cp == 1:
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
                ("latent_attention0", "latent_attention_batch"))
            if gathered_dcp:
                grp_outputs = latent_dcp.attention_batch(
                    b, grp_outputs, grp_views, plan=plan, heads=heads,
                    local_heads=logical_heads, cp=eff_cp, rows=q)
            out_set = ({prefix + ".latent_attention_batch"} if gathered_dcp
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
        if eff_cp > 1 and grp_outputs:
            out_set = ({prefix + ".latent_attention_batch"} if gathered_dcp
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
                    ))
        if eff_cp > 1 and grp_outputs and not gathered_dcp:
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
                * dtype_bytes(model.activation_dtype), "allreduce")
    return AttentionResult(ops=b.ops, last=last, cache_store_id=long_store or store,
                           index_store_id=index_store, topk_id=topk,
                           candidate_id=candidate)
