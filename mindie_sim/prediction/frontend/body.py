"""Lower structural model layers and output heads.

[Role] Build attention, feedforward, residual, stage transfer, and logits work.
[Invariants] Structural builders own operator scope and causal dependencies.
[Timeouts & Locks] Traverses active layers only. No locks or device work.
[Side-Effect Policy] Builds operator records and updates caller-owned TopK maps.
"""
from __future__ import annotations

from mindie_sim.catalog.context_parallel import cp_interleave
from mindie_sim.catalog.weight_terms import vocab_parallel_rows
from mindie_sim.prediction import frontend as _frontend
from mindie_sim.prediction.sharding.routing import _balanced_ep_owner
from mindie_sim.schema.common import align_up
from mindie_sim.schema.model import QuantSpec


_SAMPLING_OPS_PER_LOGIT = 5
_HIDDEN_SELECTION_INDEX_BYTES = 8
_HC_COLLAPSE_SIGMOID_OPS = 4
_HC_COLLAPSE_FP32_BYTES = 4
_SANDWICH_MIN_NORMS_PER_LAYER = 4


def _has_sandwich_norm(layer, model) -> bool:
    """Return True iff the layer declares sandwich (4-norm) post-attention/post-FFN norms."""
    if bool(getattr(layer, "has_sandwich_norm", False)):
        return True
    facts = dict(getattr(model, "catalog_facts", ()) or ())
    norm_cfg = facts.get("normalization")
    return (
        isinstance(norm_cfg, dict)
        and int(norm_cfg.get("norms_per_layer", 0) or 0) >= _SANDWICH_MIN_NORMS_PER_LAYER
    )


def _mhc_mean_collapse_ops(model, ready, *, prefix, mode, stage, tokens, requests, hc_mult):
    """Parameterless ``x.mean(dim=1)`` contraction across ``hc_mult`` streams (0 matrix FLOPs)."""
    h = int(model.hidden_size)
    act_bytes = _frontend.dtype_bytes(model.activation_dtype)
    mean = _frontend.vector_op(
        op_id=prefix + "hc_head_mean", deps=(ready,), stage=stage,
        layer=_frontend.FINAL_LAYER, requests=requests,
        elements=tokens * h, ops_per_element=hc_mult,
        dtype=model.activation_dtype, family="mhc", phase=mode,
        recipe="hc_final_collapse_mean", shape=(tokens, h),
        read_bytes_override=tokens * hc_mult * h * act_bytes,
        write_bytes_override=tokens * h * act_bytes,
        metadata=(("hc_mult", hc_mult), ("collapse_mode", "mean")),
    )
    return [mean], mean.op_id


def _mhc_stream_collapse_ops(model, ready, *, prefix, mode, stage, tokens, requests, hc_mult):
    """Collapse ``hc_mult`` hyper-connection streams into one hidden state before final norm."""
    h = int(model.hidden_size)
    act_bytes = _frontend.dtype_bytes(model.activation_dtype)
    fp32_quant = QuantSpec(name="hc_head_fp32", weight_format="fp32", activation_format="fp32")
    proj = _frontend.gemm_op(
        op_id=prefix + "hc_head_proj", deps=(ready,), stage=stage,
        layer=_frontend.FINAL_LAYER, requests=requests,
        m=tokens, n=hc_mult, k=hc_mult * h, quant=fp32_quant,
        category="mhc_final_collapse", phase=mode, family="gemm",
        recipe="hc_final_collapse_proj",
        metadata=(("hc_mult", hc_mult), ("collapse_input_width", hc_mult * h)),
    )
    sig = _frontend.vector_op(
        op_id=prefix + "hc_head_sigmoid", deps=(proj.op_id,), stage=stage,
        layer=_frontend.FINAL_LAYER, requests=requests,
        elements=tokens * hc_mult, ops_per_element=_HC_COLLAPSE_SIGMOID_OPS,
        dtype="fp32", family="mhc", phase=mode, recipe="hc_final_collapse_sigmoid",
        shape=(tokens, hc_mult),
    )
    mix = _frontend.vector_op(
        op_id=prefix + "hc_head_mix", deps=(ready, sig.op_id), stage=stage,
        layer=_frontend.FINAL_LAYER, requests=requests,
        elements=tokens * h, ops_per_element=2 * hc_mult,
        dtype=model.activation_dtype, family="mhc", phase=mode,
        recipe="hc_final_collapse_mix", shape=(tokens, h),
        read_bytes_override=(tokens * hc_mult * h * act_bytes
                             + tokens * hc_mult * _HC_COLLAPSE_FP32_BYTES),
        write_bytes_override=tokens * h * act_bytes,
        metadata=(("hc_mult", hc_mult),),
    )
    return [proj, sig, mix], mix.op_id


def _emit_sandwich_post_ffn_norms(
    ops, previous, saved, *, name, layer, stage, index, requests, tokens, model, mode,
):
    """Emit sandwich-norm post-FFN (and MoE parallel branch) RMSNorms on the causal DAG."""
    if layer.ffn == "moe":
        pre_exp = _frontend.norm(
            name=name + ".pre_experts_norm", deps=(saved,), stage=stage, layer=index,
            requests=requests, tokens=tokens, width=model.hidden_size, model=model, mode=mode,
        )
        router_norm = _frontend.norm(
            name=name + ".router_norm", deps=(saved,), stage=stage, layer=index,
            requests=requests, tokens=tokens, width=model.hidden_size, model=model, mode=mode,
        )
        post_exp = _frontend.norm(
            name=name + ".post_experts_norm",
            deps=(previous, pre_exp.op_id, router_norm.op_id),
            stage=stage, layer=index, requests=requests, tokens=tokens,
            width=model.hidden_size, model=model, mode=mode,
        )
        post_ffn = _frontend.norm(
            name=name + ".post_ffn_norm", deps=(post_exp.op_id,), stage=stage, layer=index,
            requests=requests, tokens=tokens, width=model.hidden_size, model=model, mode=mode,
        )
        ops.extend(_frontend._scope((pre_exp, router_norm, post_exp, post_ffn), "ffn"))
        return post_ffn.op_id
    post_ffn = _frontend.norm(
        name=name + ".post_ffn_norm", deps=(previous,), stage=stage, layer=index,
        requests=requests, tokens=tokens, width=model.hidden_size, model=model, mode=mode,
    )
    ops.extend(_frontend._scope((post_ffn,), "ffn"))
    return post_ffn.op_id


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


def _verify_query_len(view) -> int:
    """Expand decode query length by proposal count ONCE without double-counting."""
    q_len = int(getattr(view, "q_len", 0) or 0)
    if getattr(view, "padding", False):
        return q_len
    prop = int(getattr(view, "draft_proposal_tokens", 0) or 0)
    if getattr(view, "phase", "") == "decode" and prop > 0:
        if q_len <= 1:
            return 1 + prop
        return max(q_len, 1 + prop)
    return q_len + prop


def _expand_target_cohort_views(cohort_views_by_replica, *, include_spec_tokens: bool = True):
    """Expand speculative verification rows once across every DP cohort member."""
    if cohort_views_by_replica is None:
        return None
    if not include_spec_tokens:
        return {member: tuple(grp) for member, grp in cohort_views_by_replica.items()}
    return {
        member: tuple(
            _frontend.replace(view, q_len=_verify_query_len(view))
            for view in grp
        )
        for member, grp in cohort_views_by_replica.items()
    }


def _dcp_cohort_context(
    parallel, replica, views, cohort_views_by_replica, *, include_spec_tokens: bool = True,
):
    """Resolve cross-DP DCP cohort member views and local position for one replica."""
    if not bool(getattr(parallel, "cp_spans_dp", False)):
        return None, 0, False
    dp_w = max(1, int(getattr(parallel, "dcp_dp_width", 1) or 1))
    first = (int(replica) // dp_w) * dp_w
    members = tuple(range(first, min(int(parallel.dp), first + dp_w)))
    if cohort_views_by_replica is not None:
        expanded = _expand_target_cohort_views(
            cohort_views_by_replica, include_spec_tokens=include_spec_tokens,
        )
        by_member = tuple(tuple(expanded.get(m, ())) for m in members)
        pos = members.index(int(replica)) if int(replica) in members else 0
    else:
        by_member = tuple(tuple(views) for _ in members)
        pos = 0
    has_decode = any(v.phase != "prefill" for grp in by_member for v in grp)
    return by_member, pos, has_decode


def _resolve_hc_final_collapse(model) -> tuple[int, str]:
    """Return ``(hc_mult, collapse_mode)`` across active hyper-connection layers."""
    hc_plans = [
        plan for plan in model.layer_plans
        if getattr(plan, "hyper_connections", False) and int(getattr(plan, "hc_mult", 1) or 1) > 1
    ]
    hc_mult = max((int(getattr(plan, "hc_mult", 1) or 1) for plan in hc_plans), default=1)
    collapse_modes = {
        str(getattr(plan, "hc_final_collapse", "learned") or "learned").strip().lower()
        for plan in hc_plans
    }
    collapse_mode = (
        "none" if "none" in collapse_modes
        else "mean" if "mean" in collapse_modes
        else "learned"
    )
    return hc_mult, collapse_mode


def _select_body_sp(runtime, model, views, *, tokens, global_tokens, rank_ledger, cohort_valid, mode):
    """Resolve sequence-parallel plan directly through canonical W1 selector."""
    return _frontend.sequence_parallel.select(
        runtime, model, views, tokens=tokens,
        global_tokens=global_tokens, rank_ledger=rank_ledger,
        valid_global_tokens=cohort_valid,
        allow_draft_scope=(mode == "draft"),
    )


def _close_body_sp(
    ops, previous, *, sp, model, stages, prefix, requests, mode,
    hc_collapse_before_sp_gather: bool,
):
    """Emit optional pre-gather mHC collapse, SP output AllGather, and trim."""
    stage_end = stages[-1] if stages else 0
    hc_mult, collapse_mode = _resolve_hc_final_collapse(model)
    has_lowrank = any(getattr(plan, "hc_lowrank", 0) > 0 for plan in model.layer_plans)
    gather_width = model.hidden_size
    if hc_mult > 1:
        if hc_collapse_before_sp_gather:
            if has_lowrank:
                c_ops, previous = _frontend.lowrank_hyper_final_mixer(
                    prefix=prefix + "final", deps=previous, stage=stage_end,
                    requests=requests, tokens=sp.tokens_local, model=model, mode=mode,
                )
                ops.extend(c_ops)
            elif collapse_mode == "mean":
                c_ops, previous = _mhc_mean_collapse_ops(
                    model, previous, prefix=prefix, mode=mode, stage=stage_end,
                    tokens=sp.tokens_local, requests=requests, hc_mult=hc_mult,
                )
                ops.extend(c_ops)
            elif collapse_mode == "learned":
                c_ops, previous = _mhc_stream_collapse_ops(
                    model, previous, prefix=prefix, mode=mode, stage=stage_end,
                    tokens=sp.tokens_local, requests=requests, hc_mult=hc_mult,
                )
                ops.extend(c_ops)
            else:
                gather_width = hc_mult * model.hidden_size
        else:
            gather_width = hc_mult * model.hidden_size
    gathered = _frontend.sequence_parallel.allgather(
        plan=sp, op_id=prefix + "sp_output_gather", deps=(previous,),
        stage=stage_end, layer=-1, requests=requests, phase=mode,
        width=gather_width, dtype=model.activation_dtype)
    ops.append(gathered)
    previous = gathered.op_id
    if sp.physical_global > sp.input_rows:
        trimmed = _frontend.sequence_parallel.output_trim(
            plan=sp, op_id=prefix + "sp_output_trim", deps=(previous,),
            stage=stage_end, layer=-1, requests=requests, phase=mode,
            width=gather_width, dtype=model.activation_dtype)
        ops.append(trimmed)
        previous = trimmed.op_id
    return previous


def _has_nonuniform_expert_routing(model, layer, parallel) -> bool:
    """True when declared expert routing assigns unequal probability across EP ranks."""
    ep = int(getattr(parallel, "ep", 1) or 1)
    if ep <= 1 or layer.ffn != "moe" or int(layer.num_experts) <= 0:
        return False
    routing = tuple(getattr(model, "expert_routing", ()) or ())
    return len(routing) > 1 and len(set(routing)) > 1


class _FrontendBody:
    def _lower_body_attention(
        self, ops, previous, *, common, name, layer, stage, index, model, runtime,
        views, mode, tokens, requests, p, sp, attn_state, residual,
        caches, indices, topks, candidates, stages, reuse_index_selection,
        dcp_cohort_views, dcp_local_pos, has_dcp_decode, rank_ledger, physical_rank,
        explicit_physical_rank, machine, replica=None, gbsa_metadata_cache=None,
    ):
        saved = None
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
            sp_attn = bool(sp and sp.applies_to_attention)
            if sp_attn:
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
            selection_kwargs = {}
            if reuse_index_selection and layer.attention == "mla":
                selection_kwargs["reuse_index_selection"] = True
                selection_kwargs["shared_topk"] = topks
            if dcp_cohort_views is not None and layer.attention in {"gqa", "mla"}:
                selection_kwargs["dcp_cohort_views_by_member"] = dcp_cohort_views
                selection_kwargs["dcp_local_pos"] = dcp_local_pos
            if layer.attention in {"gqa", "mla", "gdn", "kda"}:
                selection_kwargs["physical_rank"] = (
                    physical_rank if physical_rank is not None
                    else (rank_ledger.owner if rank_ledger is not None else None)
                )
                selection_kwargs["explicit_physical_rank"] = bool(explicit_physical_rank)
            if layer.attention == "gqa":
                selection_kwargs["gbsa_metadata_cache"] = gbsa_metadata_cache
            attention = self.attention[layer.attention](
                prefix=name, plan=layer, views=views, stage=stage, deps=attention_ready,
                model=model, parallel=p, quant=model.quant, mode=mode,
                padded_tokens=sp.attention_tokens if sp_attn else tokens,
                cache_ids=caches, topk_ids=topks, index_cache_ids=indices,
                candidate_ids=candidates, stage_of=lambda layer: stages[layer],
                cp_interleave_size=cp_interleave(runtime), runtime=runtime,
                policy=self.attention_policy, machine=machine,
                **selection_kwargs,
            )
            if sp_attn:
                attention_ops, attention_last = _frontend.sequence_parallel.close_output(
                    attention.ops, attention.last, plan=sp, width=model.hidden_size,
                    dtype=model.activation_dtype, input_rows=sp.attention_tokens,
                    machine=machine, replica=replica, physical_rank=physical_rank)
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
                attn_out = attention.last
                if _has_sandwich_norm(layer, model):
                    post_attn = _frontend.norm(
                        name=name + ".post_attn_norm", deps=(attn_out,), stage=stage,
                        layer=index, requests=requests, tokens=tokens,
                        width=model.hidden_size, model=model, mode=mode,
                    )
                    ops.extend(_frontend._scope((post_attn,), "attention"))
                    attn_out = post_attn.op_id
                post, previous = residual.after({**common, "prefix": name + ".attention",
                                                "structure_block": "attention"},
                                                attn_out, saved)
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
        elif has_dcp_decode and layer.attention in {"gqa", "mla"}:
            attention = self.attention[layer.attention](
                prefix=name, plan=layer, views=(), stage=stage, deps=previous,
                model=model, parallel=p, quant=model.quant, mode=mode,
                padded_tokens=0, cache_ids=caches, topk_ids=topks,
                index_cache_ids=indices, candidate_ids=candidates,
                stage_of=lambda layer: stages[layer],
                cp_interleave_size=cp_interleave(runtime), runtime=runtime,
                policy=self.attention_policy, machine=machine,
                dcp_cohort_views_by_member=dcp_cohort_views,
                dcp_local_pos=dcp_local_pos,
                physical_rank=(
                    physical_rank if physical_rank is not None
                    else (rank_ledger.owner if rank_ledger is not None else None)
                ),
                explicit_physical_rank=bool(explicit_physical_rank),
            )
            ops.extend(_frontend._scope(attention.ops, "attention"))
            previous = attention.last
            hidden = previous
        else:
            hidden = previous
        return previous, hidden, saved

    def _lower_body_ffn(
        self, ops, previous, hidden, saved, *, common, name, layer, stage, index,
        model, runtime, views, mode, tokens, requests, p, sp, attn_state, residual,
        routing_global_tokens, global_tokens, routing_views, rank_ledger,
        physical_rank, replica, query_position,
        explicit_physical_rank=False, multi_replica_ep=False, machine=None,
    ):
        if tokens or layer.ffn == "moe":
            routing_tokens = (routing_global_tokens
                              if routing_global_tokens is not None
                              else global_tokens)
            # explicit token ledgers match genuine serving request views or
            # an optional target-graph correlation passed by the caller.
            realization_views = routing_views if routing_views is not None else views
            sp_entry = bool(sp and sp.stage_entry_active)
            sp_dense = bool(sp and sp.applies_to_dense_ffn)
            sp_moe = bool(sp and sp.applies_to_moe)
            realization = (_frontend.select_realization(
                runtime, layer, phase=mode, tokens=tokens,
                valid_tokens=(rank_ledger.valid[rank_ledger.owner] if rank_ledger and sp_entry else
                              sp.valid_per_tp[0] if sp_entry else sum(view.logical_q_len for view in views)),
                global_tokens=routing_tokens,
                token_shards=(p.tp if sp_entry and (rank_ledger is not None or routing_tokens is None
                                                    or routing_tokens >= tokens * p.tp) else 1),
                valid_source_tokens=sp.valid_global if sp_entry else None, replica=replica,
                views=realization_views, source_geometry=rank_ledger,
                physical_rank=(rank_ledger.owner if rank_ledger is not None else physical_rank),
                query_position=query_position,
                explicit_physical_rank=bool(explicit_physical_rank), machine=machine)
                if layer.ffn == "moe" else None)
            ffn_tokens = tokens
            if sp_dense and layer.ffn == "dense":
                gathered = _frontend.sequence_parallel.allgather(
                    plan=sp, op_id=name + ".ffn.sp_gather", deps=(hidden,),
                    stage=stage, layer=index, requests=requests, phase=mode,
                    width=model.hidden_size, dtype=model.activation_dtype)
                ops.extend(_frontend._scope((gathered,), "ffn"))
                hidden, ffn_tokens = gathered.op_id, sp.tokens_global
            resolved_ep_owner = (
                rank_ledger.owner if rank_ledger is not None else
                _balanced_ep_owner(p, int(physical_rank), replica=replica, machine=machine)
                if physical_rank is not None else None
            )
            per_owner_ep_pricing = bool(
                resolved_ep_owner is not None
                and (
                    explicit_physical_rank
                    or (multi_replica_ep and _has_nonuniform_expert_routing(model, layer, p))
                )
            )
            ffn_ops, previous = self.ffn[layer.ffn](
                **{**common, "tokens": ffn_tokens},
                deps=hidden, parallel=p, global_tokens=global_tokens,
                routing=self.routing, routing_realization=realization,
                moe_recipe=runtime.moe_recipe, mc2_chunks=runtime.mc2_chunks,
                overlap_shared=bool(runtime.multistream_overlap_shared_expert),
                runtime=runtime, sequence_plan=(sp if (sp_moe if layer.ffn == "moe" else sp_dense) else None),
                ep_owner=resolved_ep_owner,
                price_owner=(rank_ledger.owner if rank_ledger is not None else
                             resolved_ep_owner if per_owner_ep_pricing else None),
            )
            if sp_dense and layer.ffn == "dense":
                ffn_ops, previous = _frontend.sequence_parallel.close_output(
                    ffn_ops, previous, plan=sp, width=model.hidden_size,
                    dtype=model.activation_dtype,
                    machine=machine, replica=replica, physical_rank=physical_rank)
            ops.extend(_frontend._scope(ffn_ops, "ffn"))
        if tokens:
            if attn_state is not None:
                attn_state.defer_mlp(previous)
            else:
                if _has_sandwich_norm(layer, model):
                    previous = _emit_sandwich_post_ffn_norms(
                        ops, previous, saved, name=name, layer=layer, stage=stage,
                        index=index, requests=requests, tokens=tokens, model=model, mode=mode,
                    )
                post, previous = residual.after(
                    {**common, "prefix": name + ".ffn", "structure_block": "ffn"}, previous, saved)
                ops.extend(_frontend._scope(post, "ffn", "residual"))
        return previous

    @staticmethod
    def _transfer_body_stage(
        ops, previous, residual_sources, *, name, layer, stage, index, stages,
        requests, mode, tokens, model, attn_state, sp=None, prefix="",
    ):
        next_stage = stages[index + 1]
        hc_mult = (
            max(1, int(getattr(layer, "hc_mult", 1) or 1))
            if getattr(layer, "hyper_connections", False) else 1
        )
        act_bytes = _frontend.dtype_bytes(model.activation_dtype)

        def _transfer_tensor(
            dep_id, *, pp_op_id, recipe, channel_mult, gather_prefix, shard_op_id,
        ):
            cur = dep_id
            width = model.hidden_size * channel_mult
            if sp is not None and tokens > 0:
                gathered = _frontend.sequence_parallel.allgather(
                    plan=sp, op_id=f"{gather_prefix}sp_output_gather", deps=(cur,),
                    stage=stage, layer=index, requests=requests, phase=mode,
                    width=model.hidden_size, dtype=model.activation_dtype,
                    channel_multiplier=channel_mult,
                )
                ops.append(gathered)
                cur = gathered.op_id
                if sp.physical_global > sp.input_rows:
                    trimmed = _frontend.sequence_parallel.output_trim(
                        plan=sp, op_id=f"{gather_prefix}sp_output_trim", deps=(cur,),
                        stage=stage, layer=index, requests=requests, phase=mode,
                        width=width, dtype=model.activation_dtype,
                    )
                    ops.append(trimmed)
                    cur = trimmed.op_id
                pp_tokens = sp.input_rows
            else:
                pp_tokens = tokens
            transfer = _frontend.comm_op(
                op_id=pp_op_id, deps=(cur,), stage=stage, layer=index,
                requests=requests, family="pp_p2p", phase=mode, recipe=recipe,
                link_bytes=pp_tokens * width * act_bytes,
                shape=(pp_tokens, width),
                metadata=(("group_axis", "pp"), ("destination_stage", next_stage)),
            )
            ops.append(transfer)
            cur = transfer.op_id
            if sp is not None and tokens > 0:
                shard = _frontend.sequence_parallel.input_shard(
                    plan=sp, op_id=shard_op_id, deps=(cur,),
                    stage=next_stage, layer=-1, requests=requests, phase=mode,
                    width=model.hidden_size, dtype=model.activation_dtype,
                    channel_multiplier=channel_mult,
                )
                ops.append(shard)
                cur = shard.op_id
            return cur

        orig_previous = previous
        previous = _transfer_tensor(
            previous, pp_op_id=f"{name}.pp", recipe="pipeline_send",
            channel_mult=hc_mult, gather_prefix=f"{name}.",
            shard_op_id=f"{prefix}s{next_stage}.sp_shard",
        )
        transferred = {orig_previous: previous} if hc_mult == 1 else {}
        if attn_state is not None:
            if attn_state.pending_mlp_id is not None:
                attn_state.pending_mlp_id = previous
                attn_state.prefix_id = _transfer_tensor(
                    attn_state.prefix_id, pp_op_id=f"{name}.prefix_pp",
                    recipe="pipeline_residual_send", channel_mult=1,
                    gather_prefix=f"{name}.prefix_",
                    shard_op_id=f"{prefix}s{next_stage}.prefix_sp_shard",
                )
            else:
                attn_state.prefix_id = previous
            moved_banks = list(attn_state.bank_ids)
            for slot_idx, bank_id in enumerate(moved_banks):
                if bank_id is not None:
                    moved_banks[slot_idx] = _transfer_tensor(
                        bank_id, pp_op_id=f"{name}.residual_pp{slot_idx}",
                        recipe="pipeline_residual_send", channel_mult=1,
                        gather_prefix=f"{name}.residual{slot_idx}_",
                        shard_op_id=f"{prefix}s{next_stage}.residual{slot_idx}_sp_shard",
                    )
            attn_state.bank_ids = tuple(moved_banks)
        moved = []
        for source_index, source in enumerate(residual_sources):
            if source in transferred:
                moved.append(transferred[source])
                continue
            moved_id = _transfer_tensor(
                source, pp_op_id=f"{name}.residual_pp{source_index}",
                recipe="pipeline_residual_send", channel_mult=1,
                gather_prefix=f"{name}.residual{source_index}_",
                shard_op_id=f"{prefix}s{next_stage}.residual{source_index}_sp_shard",
            )
            transferred[source] = moved_id
            moved.append(moved_id)
        return previous, moved

    def body(self, model, runtime, views, *, prefix="", ready="", tokens=None,
             mode="prefill", global_tokens=None, stages=None, replica=0,
             features=None, source_rows_by_replica=None,
             routing_global_tokens=None, routing_views=None,
             cohort_views_by_replica=None,
             physical_rank=None, explicit_physical_rank: bool = False,
             query_position=None,
             include_recover_commits: bool = False, machine=None,
             layer_limit: int | None = None,
             topk_scope=None, reuse_index_selection: bool = False, rank_ledger=None,
             hc_collapse_before_sp_gather: bool = False):
        _frontend._reject_unmodeled_dbo(runtime, "frontend.body")
        p = runtime.parallel
        if layer_limit is not None:
            limit = max(1, int(layer_limit))
            model = _frontend.replace(model, layer_plans=tuple(model.layer_plans[:limit]))
        features = self.features if features is None else features
        # An explicit mechanism exclusion owns every selector it can install,
        # including the structural body default resolved before this call.
        runtime = _frontend.strip_excluded_native_kda(runtime)
        include_spec_tokens = mode not in {"draft", "draft_prefill"}
        target_cohort_views = _expand_target_cohort_views(
            cohort_views_by_replica, include_spec_tokens=include_spec_tokens,
        )
        if (rank_ledger is None and target_cohort_views is not None and views
                and (runtime.parallel.rank_table is not None
                     or runtime.parallel.pcp > 1 or runtime.kernels.get("force_eplb_source"))):
            from mindie_sim.prediction.execution.rank_ledger import rank_ledger_for
            ledger_peers = {
                m: tuple(_frontend.replace(v, draft_proposal_tokens=0) for v in grp)
                for m, grp in target_cohort_views.items()
            }
            parent = rank_ledger_for(self, model, runtime, views, tokens,
                                     ledger_peers, replica, machine,
                                     physical_rank, routing_views)
            if parent is not None:
                rank_ledger = parent
                global_tokens = routing_global_tokens = rank_ledger.physical
                source_rows_by_replica = (rank_ledger.valid if runtime.kernels.get(
                    "force_eplb_source") else rank_ledger.allocated)
        requests = tuple(view.request_id for view in views)
        tokens = sum(view.q_len for view in views) if tokens is None else tokens
        if (tokens == 0 and p.pcp > 1
                and any(v.phase == "prefill" and int(getattr(v, "global_q_len", 0) or 0) > 0
                        for v in views)):
            from mindie_sim.prediction.sharding.context_parallel import pcp_uniform_peer_rows
            tokens, _ = pcp_uniform_peer_rows(views, p.pcp, local_rows=0)
        stages = stages or _frontend.stage_map(model, runtime)
        ops, previous = [], ready
        cohort_valid = (
            sum(
                int(getattr(view, "global_q_len", 0) or 0) or int(view.q_len)
                for grp in target_cohort_views.values()
                for view in grp if not getattr(view, "padding", False)
            )
            if target_cohort_views is not None else None
        )
        sp = _select_body_sp(
            runtime, model, views, tokens=tokens,
            global_tokens=global_tokens, rank_ledger=rank_ledger,
            cohort_valid=cohort_valid, mode=mode,
        )
        sp_entry = bool(sp and sp.stage_entry_active)
        if sp_entry:
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
        gbsa_metadata_cache: dict = {}
        if topk_scope is not None:
            # Reuse a TopK selection produced by an earlier proposal step.
            topks = topk_scope
        residual_sources = []
        attnres = (_frontend.family_attnres.selection(
            runtime, model=model, phase=mode, machine=machine)
            if (tokens and mode not in {"draft", "draft_prefill"}) else None)
        attn_state = None
        if attnres is not None:
            if len(set(stages)) != 1 and runtime.parallel.pp == 1:
                raise _frontend.ModelingError("attn_res_source_caller_requires_pp1")
            attn_state = _frontend.family_attnres.AttnResPrefixState.start(
                model=model, runtime=runtime, ready=previous, tokens=tokens,
                mode=mode, requests=requests,
                optimize_prefill=attnres["optimize_prefill"])
        dcp_cohort_views, dcp_local_pos, has_dcp_decode = _dcp_cohort_context(
            p, replica, views, target_cohort_views, include_spec_tokens=include_spec_tokens)
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
            previous, hidden, saved = self._lower_body_attention(
                ops, previous, common=common, name=name, layer=layer, stage=stage,
                index=index, model=model, runtime=runtime, views=views, mode=mode,
                tokens=tokens, requests=requests, p=p, sp=sp, attn_state=attn_state,
                residual=residual, caches=caches, indices=indices, topks=topks,
                candidates=candidates, stages=stages, reuse_index_selection=reuse_index_selection,
                dcp_cohort_views=dcp_cohort_views, dcp_local_pos=dcp_local_pos,
                has_dcp_decode=has_dcp_decode, rank_ledger=rank_ledger,
                physical_rank=physical_rank, explicit_physical_rank=explicit_physical_rank,
                machine=machine, replica=replica, gbsa_metadata_cache=gbsa_metadata_cache,
            )
            previous = self._lower_body_ffn(
                ops, previous, hidden, saved, common=common, name=name, layer=layer,
                stage=stage, index=index, model=model, runtime=runtime, views=views,
                mode=mode, tokens=tokens, requests=requests, p=p, sp=sp,
                attn_state=attn_state, residual=residual, routing_global_tokens=routing_global_tokens,
                global_tokens=global_tokens, routing_views=routing_views, rank_ledger=rank_ledger,
                physical_rank=physical_rank, replica=replica, query_position=query_position,
                explicit_physical_rank=explicit_physical_rank,
                multi_replica_ep=bool(target_cohort_views and len(target_cohort_views) > 1 and p.ep_spans_dp),
                machine=machine,
            )
            if attn_state is None and layer.residual_block_size and (index + 1) % layer.residual_block_size == 0:
                residual_sources.append(previous)
            if index + 1 < len(stages) and stages[index + 1] != stage:
                previous, residual_sources = self._transfer_body_stage(
                    ops, previous, residual_sources, name=name, layer=layer,
                    stage=stage, index=index, stages=stages, requests=requests,
                    mode=mode, tokens=tokens, model=model, attn_state=attn_state,
                    sp=sp if sp_entry else None, prefix=prefix,
                )
        if attn_state is not None:
            last = model.layer_plans[-1]
            final = attn_state.final(prefix=prefix + "final", last_layer=last,
                                     stage=stages[last.index])
            ops.extend(_frontend._scope(final.ops, "ffn", "residual"))
            previous = final.hidden_id
        if sp_entry and tokens:
            previous = _close_body_sp(
                ops, previous, sp=sp, model=model, stages=stages,
                prefix=prefix, requests=requests, mode=mode,
                hc_collapse_before_sp_gather=hc_collapse_before_sp_gather,
            )
        from mindie_sim.prediction.operators.sparse_indexer.sparse_mla_metadata import before_forward
        ops = before_forward(ops, ready)
        if not include_recover_commits:
            ops = _frontend._strip_recover_intents(ops)
        return ops, previous

    @staticmethod
    def final_mixer(
        model, ready, *, prefix, mode, stage, tokens, requests,
        already_hc_collapsed: bool = False,
    ):
        """Emit low-rank or full-stream mHC collapse before final norm, or plain final norm."""
        if any(getattr(plan, "hc_lowrank", 0) > 0 for plan in model.layer_plans):
            if already_hc_collapsed:
                return [], ready
            return _frontend.lowrank_hyper_final_mixer(
                prefix=prefix + "final", deps=ready, stage=stage,
                requests=requests, tokens=tokens, model=model, mode=mode)
        hc_mult, collapse_mode = _resolve_hc_final_collapse(model)
        prefix_ops = []
        if hc_mult > 1 and tokens > 0 and not already_hc_collapsed:
            if collapse_mode == "mean":
                prefix_ops, ready = _mhc_mean_collapse_ops(
                    model, ready, prefix=prefix, mode=mode, stage=stage,
                    tokens=tokens, requests=requests, hc_mult=hc_mult,
                )
            elif collapse_mode == "learned":
                prefix_ops, ready = _mhc_stream_collapse_ops(
                    model, ready, prefix=prefix, mode=mode, stage=stage,
                    tokens=tokens, requests=requests, hc_mult=hc_mult,
                )
        final = _frontend.norm(name=prefix + "final_norm", deps=(ready,), stage=stage,
                     layer=_frontend.FINAL_LAYER, requests=requests, tokens=tokens,
                     width=model.hidden_size, model=model, mode=mode)
        return [*prefix_ops, final], final.op_id

    def logits(self, model, runtime, views, ready, *, prefix="", mode="decode", stage=None, sample=True,
               input_tokens=None, normalized=False, selected_logit_rows: int | None = None,
               vocab_padding_size: int = 0, lmhead_pad_rows: int = 0,
               hc_collapsed: bool = False):
        p = runtime.parallel
        stage = p.pp - 1 if stage is None else stage
        sampled = tuple(view for view in views if view.samples)
        rows = (
            int(selected_logit_rows)
            if selected_logit_rows is not None
            else sum(view.q_len if view.phase == "decode" else 1 for view in sampled)
        )
        input_tokens = sum(view.q_len for view in views) if input_tokens is None else input_tokens
        requests = tuple(view.request_id for view in sampled)
        ops, last = [], ready
        if not normalized and runtime.final_norm_scope == "all" and input_tokens:
            finals, last = self.final_mixer(
                model, last, prefix=prefix, mode=mode,
                stage=stage, tokens=input_tokens, requests=tuple(view.request_id for view in views),
                already_hc_collapsed=hc_collapsed,
            )
            ops.extend(finals)
        if not rows:
            return ops, last
        if rows != input_tokens:
            hc_mult, collapse_mode = _resolve_hc_final_collapse(model)
            has_lowrank = any(getattr(plan, "hc_lowrank", 0) > 0 for plan in model.layer_plans)
            select_width = (
                hc_mult * model.hidden_size
                if (
                    not normalized
                    and runtime.final_norm_scope == "selected"
                    and not hc_collapsed
                    and hc_mult > 1
                    and (has_lowrank or collapse_mode != "none")
                )
                else model.hidden_size
            )
            select = self.select_hidden(
                model, views, last, prefix=prefix, mode=mode, stage=stage,
                selected_rows=rows, width=select_width,
            )
            ops.append(select)
            last = select.op_id
        if not normalized and runtime.final_norm_scope == "selected":
            finals, last = self.final_mixer(
                model, last, prefix=prefix, mode=mode,
                stage=stage, tokens=rows, requests=requests,
                already_hc_collapsed=hc_collapsed,
            )
            ops.extend(finals)
        head_rows = max(rows, int(lmhead_pad_rows or 0))
        if vocab_padding_size > 0:
            alignment = _frontend.math.lcm(int(vocab_padding_size), p.tp)
            cols_per_rank = int(align_up(model.vocab_size, alignment)) // p.tp
            eff_vocab = p.tp * cols_per_rank
        else:
            cols_per_rank = vocab_parallel_rows(model.vocab_size, p.tp)
            eff_vocab = p.tp * cols_per_rank
        logits = _frontend.gemm_op(
            op_id=prefix + "logits", deps=(last,), stage=stage, layer=-1, requests=requests,
            m=head_rows, n=cols_per_rank, k=model.hidden_size, quant=model.quant,
            category="lm_head", phase=mode, family="logits", recipe="vocab_logits",
            metadata=(
                (("vocab_padded_size", eff_vocab), ("vocab_cols_per_rank", cols_per_rank))
                if (vocab_padding_size > 0 or eff_vocab != model.vocab_size) else ()
            ) + (
                (("vocab_padding_size", int(vocab_padding_size)),)
                if vocab_padding_size > 0 else ()
            ) + ((("lmhead_padded_rows", head_rows), ("lmhead_valid_rows", rows)) if head_rows > rows else ()),
        )
        ops.append(logits)
        last = logits.op_id
        if p.tp > 1:
            gather = _frontend.comm_op(
                op_id=prefix + "logits_gather", deps=(last,), stage=stage, layer=-1, requests=requests,
                family="allgather", phase=mode, recipe="ring_allgather",
                link_bytes=head_rows * eff_vocab * _frontend.dtype_bytes(model.activation_dtype) * (p.tp - 1) / p.tp,
                shape=(head_rows, eff_vocab),
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
    def select_hidden(
        model, views, ready, *, prefix="", mode="decode", stage=0,
        selected_rows: int | None = None, width: int | None = None,
    ):
        sampled = tuple(view for view in views if view.samples)
        rows = (
            int(selected_rows)
            if selected_rows is not None
            else sum(view.q_len if view.phase == "decode" else 1 for view in sampled)
        )
        row_width = model.hidden_size if width is None else int(width)
        payload = rows * row_width * _frontend.dtype_bytes(model.activation_dtype)
        return _frontend.vector_op(
            op_id=prefix + "select_hidden", deps=(ready,), stage=stage, layer=-1,
            requests=tuple(view.request_id for view in sampled), elements=rows * row_width,
            ops_per_element=0, dtype=model.activation_dtype, family="tensor_select", phase=mode,
            recipe="select_hidden", shape=(rows, row_width),
            read_bytes_override=payload + rows * _HIDDEN_SELECTION_INDEX_BYTES, write_bytes_override=payload,
            metadata=(("selection", "all_verify_rows_or_last_prefill_row"),))


def _representative_attention(
    frontend, residual, *, layer, rep_idx, local_views, local_requests, local_tokens,
    mode, model, runtime, parallel, features, source_rows_by_replica, machine, ready,
    attention_state, physical_rank=None, gbsa_metadata_cache=None,
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
    attn_kwargs: dict = {}
    if physical_rank is not None:
        attn_kwargs["physical_rank"] = int(physical_rank)
        attn_kwargs["explicit_physical_rank"] = False
    if lyr.attention == "gqa":
        attn_kwargs["gbsa_metadata_cache"] = gbsa_metadata_cache
    attn = frontend.attention[lyr.attention](
        prefix=name, plan=lyr, views=local_views, stage=0,
        deps=norm1.op_id, model=model, parallel=p, quant=model.quant,
        mode=mode, padded_tokens=local_tokens,
        cache_ids=caches, topk_ids=topks, index_cache_ids=indices_map,
        candidate_ids=candidates, stage_of=lambda _l: 0,
        cp_interleave_size=cp_interleave(runtime), runtime=runtime,
        policy=frontend.attention_policy, machine=machine,
        **attn_kwargs,
    )
    attn_out = attn.last
    extra_norms: list = []
    if _has_sandwich_norm(lyr, model):
        post_attn = _frontend.norm(
            name=name + ".post_attn_norm", deps=(attn_out,), stage=0,
            layer=rep_idx, requests=local_requests, tokens=local_tokens,
            width=model.hidden_size, model=model, mode=mode,
        )
        extra_norms.append(post_attn)
        attn_out = post_attn.op_id
    post_a, cur_prev = residual.after(
        {**common, "prefix": name + ".attention", "structure_block": "attention"},
        attn_out, saved,
    )
    attn_sub_raw = [
        *_frontend._scope(pre_r, "attention", "residual"),
        *_frontend._scope((norm1,), "attention"),
        *_frontend._scope(attn.ops, "attention"),
        *_frontend._scope(tuple(extra_norms), "attention"),
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
    gbsa_metadata_cache: dict = {}
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
            ready=in_id, attention_state=attention_state, physical_rank=physical_rank,
            gbsa_metadata_cache=gbsa_metadata_cache,
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
        if _has_sandwich_norm(lyr, model):
            cur_prev = _emit_sandwich_post_ffn_norms(
                ops, cur_prev, saved, name=name, layer=lyr, stage=0,
                index=rep_idx, requests=local_requests, tokens=local_tokens,
                model=model, mode=mode,
            )
        post_f, cur_prev = residual.after(
            {**common, "prefix": name + ".ffn", "structure_block": "ffn"},
            cur_prev, saved,
        )
        ops.extend(_frontend._scope(post_f, "ffn", "residual"))
        arch_output_id[akey] = cur_prev

    return cur_prev, arch_input_id, arch_output_id, raw_attn_by_arch
