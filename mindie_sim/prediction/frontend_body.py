"""Lower structural model layers and output heads.

[Role] Build attention, feedforward, residual, stage transfer, and logits work.
[Invariants] Structural builders own operator scope and causal dependencies.
[Timeouts & Locks] Traverses active layers only. No locks or device work.
[Side-Effect Policy] Builds operator records and updates caller-owned TopK maps.
"""
from __future__ import annotations

from . import frontend as _frontend


_SAMPLING_OPS_PER_LOGIT = 5
_HIDDEN_SELECTION_INDEX_BYTES = 8


def _scope(ops, block, role="body"):
    """The composing builder owns scope; operator names do not identify it."""
    return [_frontend.replace(op, metadata=tuple({
        **dict(op.metadata), "structure_block": block, "structure_role": role,
    }.items())) for op in ops]


def _strip_recover_intents(ops):
    """Remove deferred commit intents and reconnect their consumers.

    A body/model assembly prices only forward work; the post-acceptance state
    publication belongs to the service assembly, so a leftover deferred intent
    must never make a pure body depend on a sampling event.
    """
    intents = {op.op_id: op for op in ops if _frontend.is_recover_ssm_intent(op)}
    if not intents:
        return tuple(ops)

    def resolve(dep, seen):
        if dep not in intents or dep in seen:
            return (dep,)
        parents = tuple(
            parent for parent in intents[dep].deps
            if parent != dep)
        return tuple(
            item for parent in parents
            for item in resolve(parent, seen | {dep}))

    result = []
    for op in ops:
        if op.op_id in intents:
            continue
        deps = tuple(dict.fromkeys(
            parent for dep in op.deps for parent in resolve(dep, frozenset())))
        result.append(_frontend.replace(op, deps=deps) if deps != op.deps else op)
    return tuple(result)


class _FrontendBody:
    def body(self, model, runtime, views, *, prefix="", ready="", tokens=None,
             mode="prefill", global_tokens=None, stages=None, replica=0,
             features=None, source_rows_by_replica=None,
             routing_global_tokens=None, routing_views=None,
             cohort_views_by_replica=None,
             physical_rank=None, query_position=None,
             include_recover_commits: bool = False, machine=None,
             layer_limit: int | None = None,
             topk_scope=None, reuse_index_selection: bool = False, rank_ledger=None):
        _frontend._reject_unmodeled_dbo(runtime, "frontend.body")
        p = runtime.parallel
        if layer_limit is not None:
            limit = max(1, int(layer_limit))
            model = _frontend.replace(model, layer_plans=tuple(model.layer_plans[:limit]))
        features = self.features if features is None else features
        # An explicit mechanism exclusion owns every selector it can install,
        # including the structural body default resolved before this call.
        runtime = _frontend.strip_excluded_native_kda(runtime)
        if (rank_ledger is None and cohort_views_by_replica is not None and views
                and (runtime.parallel.rank_table is not None
                     or runtime.parallel.pcp > 1 or runtime.kernels.get("force_eplb_source"))):
            from .rank_ledger import rank_ledger_for
            parent = rank_ledger_for(self, model, runtime, views, tokens,
                                     cohort_views_by_replica, replica, machine,
                                     physical_rank, routing_views)
            if parent is not None:
                rank_ledger = parent
                global_tokens = routing_global_tokens = rank_ledger.physical
                source_rows_by_replica = (rank_ledger.valid if runtime.kernels.get(
                    "force_eplb_source") else rank_ledger.allocated)
        requests = tuple(view.request_id for view in views)
        tokens = sum(view.q_len for view in views) if tokens is None else tokens
        stages = stages or _frontend.stage_map(model, runtime)
        ops, previous = [], ready
        sp = _frontend.sequence_parallel.select(runtime, model, views, tokens=tokens,
                                      global_tokens=global_tokens, rank_ledger=rank_ledger)
        if sp:
            if runtime.moe_recipe != "alltoall":
                raise ValueError("sequence_parallel_requires_unchunked_alltoall")
            tokens, global_tokens = sp.tokens_local, sp.cohort_tokens
            if tokens:
                shard = _frontend.sequence_parallel.input_shard(
                    plan=sp, op_id=prefix + "sp_shard", deps=(previous,),
                    stage=stages[0] if stages else 0, layer=-1, requests=requests, phase=mode,
                    width=model.hidden_size, dtype=model.activation_dtype)
                ops.append(shard)
                previous = shard.op_id
        caches, indices, topks, candidates = {}, {}, {}, {}
        if topk_scope is not None:
            # Reuse a TopK selection produced by an earlier proposal step.
            topks = topk_scope
        residual_sources = []
        attnres = (_frontend.family_attnres.selection(
            runtime, model=model, phase=mode)
            if (tokens and mode not in {"draft", "draft_prefill"}) else None)
        attn_state = None
        if attnres is not None:
            if len(set(stages)) != 1 and runtime.parallel.pp == 1:
                raise _frontend.ModelingError("attn_res_source_caller_requires_pp1")
            attn_state = _frontend.family_attnres.AttnResPrefixState.start(
                model=model, runtime=runtime, ready=previous, tokens=tokens,
                mode=mode, requests=requests,
                optimize_prefill=attnres["optimize_prefill"])
        for index, layer in enumerate(model.layer_plans):
            stage = stages[index]
            name = f"{prefix}s{stage}.l{index}"
            common = dict(prefix=name, plan=layer, stage=stage, requests=requests,
                          tokens=tokens, model=model, mode=mode,
                          residual_sources=tuple(residual_sources),
                          features=features,
                          source_rows_by_replica=source_rows_by_replica)
            residual = self.residuals["lowrank_hyper" if getattr(layer, "hc_lowrank", 0) > 0 else
                                      "hyper" if layer.hyper_connections else
                                      "attention" if layer.residual_block_size else "plain"]
            if tokens:
                if attn_state is not None:
                    prepared = attn_state.prepare(
                        prefix=name + ".attention", layer=layer, stage=stage)
                    ops.extend(_frontend._scope(prepared.ops, "attention", "residual"))
                    hidden = prepared.hidden_id
                else:
                    if layer.engram and _frontend.is_qwen4_exp_ple(model.engram):
                        extra, previous = _frontend.lower_ple(
                            prefix=name, plan=layer, stage=stage, deps=previous,
                            model=model, parallel=p, quant=model.quant, mode=mode,
                            padded_tokens=tokens, views=views, runtime=runtime)
                        ops.extend(extra)
                    pre, hidden, saved = residual.before(
                        {**common, "prefix": name + ".attention", "structure_block": "attention"}, previous)
                    ops.extend(_frontend._scope(pre, "attention", "residual"))
                    if getattr(layer, "hc_lowrank", 0) <= 0:
                        normalization = _frontend.norm(
                            name=name + ".norm1", deps=(hidden,), stage=stage, layer=index,
                            requests=requests, tokens=tokens, width=model.hidden_size, model=model, mode=mode,
                        )
                        ops.extend(_frontend._scope((normalization,), "attention"))
                        hidden = normalization.op_id
                attention_ready = hidden
                if sp:
                    gathered = _frontend.sequence_parallel.allgather(
                        plan=sp, op_id=name + ".attention.sp_gather",
                        deps=(attention_ready,), stage=stage, layer=index,
                        requests=requests, phase=mode, width=model.hidden_size,
                        dtype=model.activation_dtype)
                    ops.extend(_frontend._scope((gathered,), "attention"))
                    attention_ready = gathered.op_id
                    if sp.physical_global != sp.attention_tokens:
                        sliced = _frontend.sequence_parallel.attention_input(
                            plan=sp, op_id=name + ".attention.sp_input_view",
                            deps=(attention_ready,), stage=stage, layer=index,
                            requests=requests, phase=mode, width=model.hidden_size,
                            dtype=model.activation_dtype)
                        ops.extend(_frontend._scope((sliced,), "attention"))
                        attention_ready = sliced.op_id
                selection_kwargs = (
                    {"reuse_index_selection": True, "shared_topk": topks}
                    if reuse_index_selection and layer.attention == "mla" else {})
                attention = self.attention[layer.attention](
                    prefix=name, plan=layer, views=views, stage=stage, deps=attention_ready,
                    model=model, parallel=p, quant=model.quant, mode=mode,
                    padded_tokens=sp.attention_tokens if sp else tokens,
                    cache_ids=caches, topk_ids=topks, index_cache_ids=indices,
                    candidate_ids=candidates, stage_of=lambda layer: stages[layer],
                    cp_interleave_size=runtime.kv_block_size, runtime=runtime,
                    policy=self.attention_policy, machine=machine,
                    **selection_kwargs,
                )
                if sp:
                    attention_ops, attention_last = _frontend.sequence_parallel.close_output(
                        attention.ops, attention.last, plan=sp, width=model.hidden_size,
                        dtype=model.activation_dtype, input_rows=sp.attention_tokens)
                    attention = _frontend.replace(attention, ops=attention_ops, last=attention_last)
                ops.extend(_frontend._scope(attention.ops, "attention"))
                for mapping, result in ((caches, attention.cache_store_id), (indices, attention.index_store_id),
                                        (topks, attention.topk_id), (candidates, attention.candidate_id)):
                    if result:
                        mapping[layer.index] = result
                if attention.index_store_id:
                    caches["index", layer.index] = attention.index_store_id
                if attn_state is not None:
                    post = attn_state.attention_added(
                        prefix=name + ".ffn", layer=layer, stage=stage,
                        attention_id=attention.last)
                    ops.extend(_frontend._scope(post.ops, "ffn", "residual"))
                else:
                    post, previous = residual.after({**common, "prefix": name + ".attention",
                                                    "structure_block": "attention"},
                                                    attention.last, saved)
                    ops.extend(_frontend._scope(post, "attention", "residual"))
                if layer.engram and not _frontend.is_qwen4_exp_ple(model.engram):
                    if attn_state is not None:
                        raise _frontend.ModelingError("attn_res_selected_with_engram_unsupported")
                    extra, previous = _frontend.lower_engram(
                        prefix=name, plan=layer, stage=stage, deps=previous,
                        model=model, parallel=p, quant=model.quant, mode=mode,
                        padded_tokens=tokens, views=views, runtime=runtime,
                    )
                    ops.extend(extra)
                if attn_state is not None:
                    hidden = post.hidden_id
                else:
                    pre, hidden, saved = residual.before(
                        {**common, "prefix": name + ".ffn", "structure_block": "ffn"}, previous)
                    ops.extend(_frontend._scope(pre, "ffn", "residual"))
                    if getattr(layer, "hc_lowrank", 0) <= 0:
                        normalization = _frontend.norm(
                            name=name + ".norm2", deps=(hidden,), stage=stage, layer=index,
                            requests=requests, tokens=tokens, width=model.hidden_size, model=model, mode=mode,
                        )
                        ops.extend(_frontend._scope((normalization,), "ffn"))
                        hidden = normalization.op_id
            else:
                hidden = previous
            if tokens or layer.ffn == "moe":
                routing_tokens = (routing_global_tokens
                                  if routing_global_tokens is not None
                                  else global_tokens)
                # explicit token ledgers match genuine serving request views or
                # an optional target-graph correlation passed by the caller.
                realization_views = routing_views if routing_views is not None else views
                realization = (_frontend.select_realization(
                    runtime, layer, phase=mode, tokens=tokens,
                    valid_tokens=(rank_ledger.valid[rank_ledger.owner] if rank_ledger and sp else
                                  sp.valid_per_tp[0] if sp else sum(view.logical_q_len for view in views)),
                    global_tokens=routing_tokens,
                    token_shards=(p.tp if sp and (rank_ledger is not None or routing_tokens is None
                                                  or routing_tokens >= tokens * p.tp) else 1),
                    valid_source_tokens=sp.valid_global if sp else None, replica=replica,
                    views=realization_views, source_geometry=rank_ledger,
                    physical_rank=(rank_ledger.owner if rank_ledger is not None else physical_rank),
                    query_position=query_position)
                    if layer.ffn == "moe" else None)
                ffn_tokens = tokens
                if sp and layer.ffn == "dense":
                    gathered = _frontend.sequence_parallel.allgather(
                        plan=sp, op_id=name + ".ffn.sp_gather", deps=(hidden,),
                        stage=stage, layer=index, requests=requests, phase=mode,
                        width=model.hidden_size, dtype=model.activation_dtype)
                    ops.extend(_frontend._scope((gathered,), "ffn"))
                    hidden, ffn_tokens = gathered.op_id, sp.tokens_global
                ffn_ops, previous = self.ffn[layer.ffn](
                    **{**common, "tokens": ffn_tokens},
                    deps=hidden, parallel=p, global_tokens=global_tokens,
                    routing=self.routing, routing_realization=realization,
                    moe_recipe=runtime.moe_recipe, mc2_chunks=runtime.mc2_chunks,
                    overlap_shared=bool(runtime.multistream_overlap_shared_expert),
                    runtime=runtime, sequence_plan=sp,
                    ep_owner=(rank_ledger.owner if rank_ledger is not None else
                              int(physical_rank) % p.ep if physical_rank is not None else None),
                    price_owner=(rank_ledger.owner if rank_ledger is not None else None),
                )
                if sp and layer.ffn == "dense":
                    ffn_ops, previous = _frontend.sequence_parallel.close_output(
                        ffn_ops, previous, plan=sp, width=model.hidden_size,
                        dtype=model.activation_dtype)
                ops.extend(_frontend._scope(ffn_ops, "ffn"))
            if tokens:
                if attn_state is not None:
                    attn_state.defer_mlp(previous)
                else:
                    post, previous = residual.after(
                        {**common, "prefix": name + ".ffn", "structure_block": "ffn"}, previous, saved)
                    ops.extend(_frontend._scope(post, "ffn", "residual"))
            if attn_state is None and layer.residual_block_size and (index + 1) % layer.residual_block_size == 0:
                residual_sources.append(previous)
            if index + 1 < len(stages) and stages[index + 1] != stage:
                transfer = _frontend.comm_op(
                    op_id=name + ".pp", deps=(previous,), stage=stage, layer=index,
                    requests=requests, family="pp_p2p", phase=mode, recipe="pipeline_send",
                    link_bytes=tokens * model.hidden_size * _frontend.dtype_bytes(model.activation_dtype)
                               * max(1, layer.hc_mult), shape=(tokens, model.hidden_size),
                    metadata=(("group_axis", "pp"), ("destination_stage", stages[index + 1])),
                )
                ops.append(transfer)
                previous = transfer.op_id
                if attn_state is not None:
                    if attn_state.pending_mlp_id is not None:
                        attn_state.pending_mlp_id = previous
                        ptx = _frontend.comm_op(
                            op_id=f"{name}.prefix_pp", deps=(attn_state.prefix_id,),
                            stage=stage, layer=index, requests=requests, family="pp_p2p",
                            phase=mode, recipe="pipeline_residual_send",
                            link_bytes=tokens * model.hidden_size * _frontend.dtype_bytes(model.activation_dtype),
                            shape=(tokens, model.hidden_size),
                            metadata=(("group_axis", "pp"), ("destination_stage", stages[index + 1])))
                        ops.append(ptx)
                        attn_state.prefix_id = ptx.op_id
                    else:
                        attn_state.prefix_id = previous
                    moved_banks = list(attn_state.bank_ids)
                    for slot_idx, bank_id in enumerate(moved_banks):
                        if bank_id is not None:
                            btx = _frontend.comm_op(
                                op_id=f"{name}.residual_pp{slot_idx}", deps=(bank_id,),
                                stage=stage, layer=index, requests=requests, family="pp_p2p",
                                phase=mode, recipe="pipeline_residual_send",
                                link_bytes=tokens * model.hidden_size * _frontend.dtype_bytes(model.activation_dtype),
                                shape=(tokens, model.hidden_size),
                                metadata=(("group_axis", "pp"), ("destination_stage", stages[index + 1])))
                            ops.append(btx)
                            moved_banks[slot_idx] = btx.op_id
                    attn_state.bank_ids = tuple(moved_banks)
                moved = []
                for source_index, source in enumerate(residual_sources):
                    transfer = _frontend.comm_op(
                        op_id=f"{name}.residual_pp{source_index}", deps=(source,),
                        stage=stage, layer=index, requests=requests, family="pp_p2p",
                        phase=mode, recipe="pipeline_residual_send",
                        link_bytes=tokens * model.hidden_size * _frontend.dtype_bytes(model.activation_dtype),
                        shape=(tokens, model.hidden_size),
                        metadata=(("group_axis", "pp"), ("destination_stage", stages[index + 1])))
                    ops.append(transfer)
                    moved.append(transfer.op_id)
                residual_sources = moved
        if attn_state is not None:
            last = model.layer_plans[-1]
            final = attn_state.final(prefix=prefix + "final", last_layer=last,
                                     stage=stages[last.index])
            ops.extend(_frontend._scope(final.ops, "ffn", "residual"))
            previous = final.hidden_id
        if sp and tokens:
            gathered = _frontend.sequence_parallel.allgather(
                plan=sp, op_id=prefix + "sp_output_gather", deps=(previous,),
                stage=stages[-1] if stages else 0, layer=-1, requests=requests, phase=mode,
                width=model.hidden_size, dtype=model.activation_dtype)
            ops.append(gathered)
            previous = gathered.op_id
            if sp.physical_global > sp.input_rows:
                trimmed = _frontend.sequence_parallel.output_trim(
                    plan=sp, op_id=prefix + "sp_output_trim", deps=(previous,),
                    stage=stages[-1] if stages else 0, layer=-1, requests=requests, phase=mode,
                    width=model.hidden_size, dtype=model.activation_dtype)
                ops.append(trimmed)
                previous = trimmed.op_id
        from mindie_sim.prediction.operators.sparse_indexer.sparse_mla_metadata import before_forward
        ops = before_forward(ops, ready)
        if not include_recover_commits:
            ops = _frontend._strip_recover_intents(ops)
        return ops, previous

    @staticmethod
    def final_mixer(model, ready, *, prefix, mode, stage, tokens, requests):
        """Emit the Qwen4Exp stream collapse, or the ordinary final norm."""
        if any(getattr(plan, "hc_lowrank", 0) > 0 for plan in model.layer_plans):
            return _frontend.lowrank_hyper_final_mixer(
                prefix=prefix + "final", deps=ready, stage=stage,
                requests=requests, tokens=tokens, model=model, mode=mode)
        final = _frontend.norm(name=prefix + "final_norm", deps=(ready,), stage=stage,
                     layer=_frontend.FINAL_LAYER, requests=requests, tokens=tokens,
                     width=model.hidden_size, model=model, mode=mode)
        return [final], final.op_id

    def logits(self, model, runtime, views, ready, *, prefix="", mode="decode", stage=None, sample=True,
               input_tokens=None, normalized=False):
        p = runtime.parallel
        stage = p.pp - 1 if stage is None else stage
        sampled = tuple(view for view in views if view.samples)
        rows = sum(view.q_len if view.phase == "decode" else 1 for view in sampled)
        input_tokens = sum(view.q_len for view in views) if input_tokens is None else input_tokens
        requests = tuple(view.request_id for view in sampled)
        ops, last = [], ready
        if not normalized and runtime.final_norm_scope == "all" and input_tokens:
            finals, last = self.final_mixer(model, last, prefix=prefix, mode=mode,
                stage=stage, tokens=input_tokens, requests=tuple(view.request_id for view in views))
            ops.extend(finals)
        if not rows:
            return ops, last
        if rows != input_tokens:
            select = self.select_hidden(model, views, last, prefix=prefix, mode=mode, stage=stage)
            ops.append(select)
            last = select.op_id
        if not normalized and runtime.final_norm_scope == "selected":
            finals, last = self.final_mixer(model, last, prefix=prefix, mode=mode,
                stage=stage, tokens=rows, requests=requests)
            ops.extend(finals)
        logits = _frontend.gemm_op(
            op_id=prefix + "logits", deps=(last,), stage=stage, layer=-1, requests=requests,
            m=rows, n=_frontend.math.ceil(model.vocab_size / p.tp), k=model.hidden_size, quant=model.quant,
            category="lm_head", phase=mode, family="logits", recipe="vocab_logits",
        )
        ops.append(logits)
        last = logits.op_id
        if p.tp > 1:
            gather = _frontend.comm_op(
                op_id=prefix + "logits_gather", deps=(last,), stage=stage, layer=-1, requests=requests,
                family="allgather", phase=mode, recipe="ring_allgather",
                link_bytes=rows * model.vocab_size * 2 * (p.tp - 1) / p.tp,
                shape=(rows, model.vocab_size),
                metadata=(("group_axis", "tp"), ("steps", p.tp - 1)),
            )
            ops.append(gather)
            last = gather.op_id
        if not sample:
            return ops, last
        sample = _frontend.vector_op(
            op_id=prefix + "sample", deps=(last,), stage=stage, layer=-1, requests=requests,
            elements=rows * model.vocab_size, ops_per_element=_SAMPLING_OPS_PER_LOGIT, dtype="fp32",
            family="sampling", phase=mode, recipe="sample_logits", shape=(rows, model.vocab_size),
        )
        ops.append(sample)
        return ops, sample.op_id

    @staticmethod
    def select_hidden(model, views, ready, *, prefix="", mode="decode", stage=0):
        sampled = tuple(view for view in views if view.samples)
        rows = sum(view.q_len if view.phase == "decode" else 1 for view in sampled)
        payload = rows * model.hidden_size * _frontend.dtype_bytes(model.activation_dtype)
        return _frontend.vector_op(
            op_id=prefix + "select_hidden", deps=(ready,), stage=stage, layer=-1,
            requests=tuple(view.request_id for view in sampled), elements=rows * model.hidden_size,
            ops_per_element=0, dtype=model.activation_dtype, family="tensor_select", phase=mode,
            recipe="select_hidden", shape=(rows, model.hidden_size),
            read_bytes_override=payload + rows * _HIDDEN_SELECTION_INDEX_BYTES, write_bytes_override=payload,
            metadata=(("selection", "all_verify_rows_or_last_prefill_row"),))


def _representative_attention(
    frontend, residual, *, layer, rep_idx, local_views, local_requests, local_tokens,
    mode, model, runtime, parallel, features, source_rows_by_replica, machine, ready,
    attention_state,
):
    """Build the attention portion of a representative layer, including residuals."""
    caches, indices_map, topks, candidates = attention_state
    lyr = layer
    p = parallel
    cur_prev = ready
    name = f"s0.l{rep_idx}"
    common = dict(
        prefix=name, plan=lyr, stage=0, requests=local_requests,
        tokens=local_tokens, model=model, mode=mode,
        residual_sources=(), features=features,
        source_rows_by_replica=source_rows_by_replica,
    )
    pre_r, hidden, saved = residual.before(
        {**common, "prefix": name + ".attention", "structure_block": "attention"},
        cur_prev,
    )
    norm1 = _frontend.norm(
        name=name + ".norm1", deps=(hidden,), stage=0, layer=rep_idx,
        requests=local_requests, tokens=local_tokens,
        width=model.hidden_size, model=model, mode=mode,
    )
    attn = frontend.attention[lyr.attention](
        prefix=name, plan=lyr, views=local_views, stage=0,
        deps=norm1.op_id, model=model, parallel=p, quant=model.quant,
        mode=mode, padded_tokens=local_tokens,
        cache_ids=caches, topk_ids=topks, index_cache_ids=indices_map,
        candidate_ids=candidates, stage_of=lambda _l: 0,
        cp_interleave_size=runtime.kv_block_size, runtime=runtime,
        policy=frontend.attention_policy, machine=machine,
    )
    post_a, cur_prev = residual.after(
        {**common, "prefix": name + ".attention", "structure_block": "attention"},
        attn.last, saved,
    )
    attn_sub_raw = [
        *_frontend._scope(pre_r, "attention", "residual"),
        *_frontend._scope((norm1,), "attention"),
        *_frontend._scope(attn.ops, "attention"),
        *_frontend._scope(post_a, "attention", "residual"),
    ]
    return attn_sub_raw, cur_prev, common


def _build_representative_body(
    frontend, ops, *, active_rep_indices, layers, arch_for_layer, body_ready,
    runtime, local_views, local_tokens, model, mode, features, realizations,
    source_rows_by_replica, body_global_tokens, physical_rank, machine, parallel,
):
    """Append one raw layer per active archetype and record its boundaries."""
    cur_prev = body_ready
    arch_input_id, arch_output_id, raw_attn_by_arch = {}, {}, {}
    local_requests = tuple(v.request_id for v in local_views)
    residual = frontend.residuals["plain"]
    attention_state = ({}, {}, {}, {})
    p = parallel
    for rep_idx in active_rep_indices:
        lyr = layers[rep_idx]
        akey = arch_for_layer[rep_idx]
        in_id = cur_prev
        arch_input_id[akey] = in_id
        attn_sub_raw, cur_prev, common = _representative_attention(
            frontend, residual, layer=lyr, rep_idx=rep_idx, local_views=local_views,
            local_requests=local_requests, local_tokens=local_tokens, mode=mode, model=model,
            runtime=runtime, parallel=p, features=features,
            source_rows_by_replica=source_rows_by_replica, machine=machine,
            ready=in_id, attention_state=attention_state,
        )
        name = common["prefix"]
        if lyr.attention not in {"gdn", "kda", "projection_only", "unmodeled"}:
            raw_attn_by_arch[akey] = (in_id, attn_sub_raw, cur_prev)
        ops.extend(attn_sub_raw)
        pre_f, hidden, saved = residual.before(
            {**common, "prefix": name + ".ffn", "structure_block": "ffn"},
            cur_prev,
        )
        ops.extend(_frontend._scope(pre_f, "ffn", "residual"))
        norm2 = _frontend.norm(
            name=name + ".norm2", deps=(hidden,), stage=0, layer=rep_idx,
            requests=local_requests, tokens=local_tokens,
            width=model.hidden_size, model=model, mode=mode,
        )
        ops.extend(_frontend._scope((norm2,), "ffn"))
        ffn_ops, cur_prev = frontend.ffn[lyr.ffn](
            **common, deps=norm2.op_id, parallel=p,
            global_tokens=body_global_tokens, routing=frontend.routing,
            routing_realization=realizations[rep_idx],
            moe_recipe=runtime.moe_recipe, mc2_chunks=runtime.mc2_chunks,
            overlap_shared=bool(runtime.multistream_overlap_shared_expert),
            runtime=runtime, sequence_plan=None,
            ep_owner=(int(physical_rank) % int(p.ep)
                      if physical_rank is not None and int(p.ep) > 0
                      else None),
        )
        ops.extend(_frontend._scope(ffn_ops, "ffn"))
        post_f, cur_prev = residual.after(
            {**common, "prefix": name + ".ffn", "structure_block": "ffn"},
            cur_prev, saved,
        )
        ops.extend(_frontend._scope(post_f, "ffn", "residual"))
        arch_output_id[akey] = cur_prev

    return cur_prev, arch_input_id, arch_output_id, raw_attn_by_arch
