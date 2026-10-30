"""Model structure to operator graph, with pluggable semantic builders.

[Role] Compose model blocks, media, speculative work and software graph capture.
[Invariants] There is one target body per verify pass. Dependencies express
  actual producers and consumers. A model name and a calibration row cannot
  choose a code path; semantic kinds dispatch through explicit registries.
[Timeouts & Locks] O(active layer graph), no iteration history, device or I/O.
[Side-Effect Policy] Creates immutable operator graphs.
"""
from __future__ import annotations

from dataclasses import replace
import math

from mindie_sim.catalog.engram import lower_engram
from mindie_sim.catalog.media import lower_media
from mindie_sim.catalog.ops import barrier_op, comm_op, gemm_op, vector_op
from mindie_sim.catalog.context_parallel import pcp_rank_chunk_ranges, split_pcp_pairs
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.workload import selected_mode
from .attention import ATTENTION_BUILDERS, AttentionPolicy
from .blocks import RESIDUAL_BUILDERS, norm
from .feedforward import FFN_BUILDERS, RoutingPolicy
from .graph import GraphPolicy
from .routing_realizations import select_realization
from .structure import normalize_model, stage_map
from .transforms import lower_implementations
from . import family_attnres, sequence_parallel


def _scope(ops, block, role="body"):
    """The composing builder owns scope; operator names do not identify it."""
    return [replace(op, metadata=tuple({
        **dict(op.metadata), "structure_block": block, "structure_role": role,
    }.items())) for op in ops]


class Frontend:
    def __init__(self, *, attention=None, ffn=None, residuals=None, graph=None,
                 attention_policy=None, routing=None):
        self.attention = {**ATTENTION_BUILDERS, **(attention or {})}
        self.ffn = {**FFN_BUILDERS, **(ffn or {})}
        self.residuals = {**RESIDUAL_BUILDERS, **(residuals or {})}
        self.graph = graph or GraphPolicy()
        self.attention_policy = attention_policy or AttentionPolicy()
        self.routing = routing or RoutingPolicy()

    def body(self, model, runtime, views, *, prefix="", ready="", tokens=None,
             mode="prefill", global_tokens=None, stages=None, replica=0):
        p = runtime.parallel
        requests = tuple(view.request_id for view in views)
        tokens = sum(view.q_len for view in views) if tokens is None else tokens
        stages = stages or stage_map(model, runtime)
        ops, previous = [], ready
        sp = sequence_parallel.select(runtime, model, views, tokens=tokens,
                                      global_tokens=global_tokens)
        if sp:
            if runtime.moe_recipe != "alltoall":
                raise ValueError("sequence_parallel_requires_unchunked_alltoall")
            tokens, global_tokens = sp.tokens_local, sp.cohort_tokens
            if tokens:
                shard = sequence_parallel.input_shard(
                    plan=sp, op_id=prefix + "sp_shard", deps=(previous,),
                    stage=0, layer=-1, requests=requests, phase=mode,
                    width=model.hidden_size, dtype=model.activation_dtype)
                ops.append(shard)
                previous = shard.op_id
        caches, indices, topks, candidates = {}, {}, {}, {}
        residual_sources = []
        attnres = family_attnres.selection(runtime) if tokens else None
        attn_state = None
        if attnres is not None:
            if len(set(stages)) != 1:
                raise ModelingError("attn_res_source_caller_requires_pp1")
            attn_state = family_attnres.AttnResPrefixState.start(
                model=model, runtime=runtime, ready=previous, tokens=tokens,
                mode=mode, requests=requests,
                optimize_prefill=attnres["optimize_prefill"])
        for index, layer in enumerate(model.layer_plans):
            stage = stages[index]
            name = f"{prefix}s{stage}.l{index}"
            common = dict(prefix=name, plan=layer, stage=stage, requests=requests,
                          tokens=tokens, model=model, mode=mode, residual_sources=tuple(residual_sources))
            residual = self.residuals["hyper" if layer.hyper_connections else
                                      "attention" if layer.residual_block_size else "plain"]
            if tokens:
                if attn_state is not None:
                    prepared = attn_state.prepare(
                        prefix=name + ".attention", layer=layer, stage=stage)
                    ops.extend(_scope(prepared.ops, "attention", "residual"))
                    hidden = prepared.hidden_id
                else:
                    pre, hidden, saved = residual.before(
                        {**common, "prefix": name + ".attention", "structure_block": "attention"}, previous)
                    ops.extend(_scope(pre, "attention", "residual"))
                    normalization = norm(
                        name=name + ".norm1", deps=(hidden,), stage=stage, layer=index,
                        requests=requests, tokens=tokens, width=model.hidden_size, model=model, mode=mode,
                    )
                    ops.extend(_scope((normalization,), "attention"))
                    hidden = normalization.op_id
                attention_ready = hidden
                if sp:
                    gathered = sequence_parallel.allgather(
                        plan=sp, op_id=name + ".attention.sp_gather",
                        deps=(attention_ready,), stage=stage, layer=index,
                        requests=requests, phase=mode, width=model.hidden_size,
                        dtype=model.activation_dtype)
                    ops.extend(_scope((gathered,), "attention"))
                    attention_ready = gathered.op_id
                    if sp.physical_global != sp.attention_tokens:
                        sliced = sequence_parallel.attention_input(
                            plan=sp, op_id=name + ".attention.sp_input_view",
                            deps=(attention_ready,), stage=stage, layer=index,
                            requests=requests, phase=mode, width=model.hidden_size,
                            dtype=model.activation_dtype)
                        ops.extend(_scope((sliced,), "attention"))
                        attention_ready = sliced.op_id
                attention = self.attention[layer.attention](
                    prefix=name, plan=layer, views=views, stage=stage, deps=attention_ready,
                    model=model, parallel=p, quant=model.quant, mode=mode,
                    padded_tokens=sp.attention_tokens if sp else tokens,
                    cache_ids=caches, topk_ids=topks, index_cache_ids=indices,
                    candidate_ids=candidates, stage_of=lambda layer: stages[layer],
                    cp_interleave_size=runtime.kv_block_size, runtime=runtime,
                    policy=self.attention_policy,
                )
                if sp:
                    attention_ops, attention_last = sequence_parallel.close_output(
                        attention.ops, attention.last, plan=sp, width=model.hidden_size,
                        dtype=model.activation_dtype, input_rows=sp.attention_tokens)
                    attention = replace(attention, ops=attention_ops, last=attention_last)
                ops.extend(_scope(attention.ops, "attention"))
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
                    ops.extend(_scope(post.ops, "ffn", "residual"))
                else:
                    post, previous = residual.after({**common, "prefix": name + ".attention",
                                                    "structure_block": "attention"},
                                                    attention.last, saved)
                    ops.extend(_scope(post, "attention", "residual"))
                if layer.engram:
                    if attn_state is not None:
                        raise ModelingError("attn_res_selected_with_engram_unsupported")
                    extra, previous = lower_engram(
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
                    ops.extend(_scope(pre, "ffn", "residual"))
                    normalization = norm(
                        name=name + ".norm2", deps=(hidden,), stage=stage, layer=index,
                        requests=requests, tokens=tokens, width=model.hidden_size, model=model, mode=mode,
                    )
                    ops.extend(_scope((normalization,), "ffn"))
                    hidden = normalization.op_id
            else:
                hidden = previous
            if tokens or layer.ffn == "moe":
                realization = (select_realization(
                    runtime, layer, phase=mode, tokens=tokens,
                    valid_tokens=sp.valid_per_tp[0] if sp else sum(view.q_len for view in views),
                    global_tokens=global_tokens, token_shards=p.tp if sp else 1,
                    valid_source_tokens=sp.valid_global if sp else None, replica=replica)
                    if layer.ffn == "moe" else None)
                ffn_tokens = tokens
                if sp and layer.ffn == "dense":
                    gathered = sequence_parallel.allgather(
                        plan=sp, op_id=name + ".ffn.sp_gather", deps=(hidden,),
                        stage=stage, layer=index, requests=requests, phase=mode,
                        width=model.hidden_size, dtype=model.activation_dtype)
                    ops.extend(_scope((gathered,), "ffn"))
                    hidden, ffn_tokens = gathered.op_id, sp.tokens_global
                ffn_ops, previous = self.ffn[layer.ffn](
                    **{**common, "tokens": ffn_tokens},
                    deps=hidden, parallel=p, global_tokens=global_tokens,
                    routing=self.routing, routing_realization=realization,
                    moe_recipe=runtime.moe_recipe, mc2_chunks=runtime.mc2_chunks,
                    overlap_shared=bool(runtime.multistream_overlap_shared_expert),
                    runtime=runtime, sequence_plan=sp,
                )
                if sp and layer.ffn == "dense":
                    ffn_ops, previous = sequence_parallel.close_output(
                        ffn_ops, previous, plan=sp, width=model.hidden_size,
                        dtype=model.activation_dtype)
                ops.extend(_scope(ffn_ops, "ffn"))
            if tokens:
                if attn_state is not None:
                    attn_state.defer_mlp(previous)
                else:
                    post, previous = residual.after(
                        {**common, "prefix": name + ".ffn", "structure_block": "ffn"}, previous, saved)
                    ops.extend(_scope(post, "ffn", "residual"))
            if attn_state is None and layer.residual_block_size and (index + 1) % layer.residual_block_size == 0:
                residual_sources.append(previous)
            if index + 1 < len(stages) and stages[index + 1] != stage:
                transfer = comm_op(
                    op_id=name + ".pp", deps=(previous,), stage=stage, layer=index,
                    requests=requests, family="pp_p2p", phase=mode, recipe="pipeline_send",
                    link_bytes=tokens * model.hidden_size * dtype_bytes(model.activation_dtype)
                               * max(1, layer.hc_mult), shape=(tokens, model.hidden_size),
                    metadata=(("group_axis", "pp"), ("destination_stage", stages[index + 1])),
                )
                ops.append(transfer)
                previous = transfer.op_id
                moved = []
                for source_index, source in enumerate(residual_sources):
                    transfer = comm_op(
                        op_id=f"{name}.residual_pp{source_index}", deps=(source,),
                        stage=stage, layer=index, requests=requests, family="pp_p2p",
                        phase=mode, recipe="pipeline_residual_send",
                        link_bytes=tokens * model.hidden_size * dtype_bytes(model.activation_dtype),
                        shape=(tokens, model.hidden_size),
                        metadata=(("group_axis", "pp"), ("destination_stage", stages[index + 1])))
                    ops.append(transfer)
                    moved.append(transfer.op_id)
                residual_sources = moved
        if attn_state is not None:
            last = model.layer_plans[-1]
            final = attn_state.final(prefix=prefix + "final", last_layer=last,
                                     stage=stages[last.index])
            ops.extend(_scope(final.ops, "ffn", "residual"))
            previous = final.hidden_id
        if sp and tokens:
            gathered = sequence_parallel.allgather(
                plan=sp, op_id=prefix + "sp_output_gather", deps=(previous,),
                stage=0, layer=-1, requests=requests, phase=mode,
                width=model.hidden_size, dtype=model.activation_dtype)
            ops.append(gathered)
            previous = gathered.op_id
            if sp.physical_global > sp.input_rows:
                trimmed = sequence_parallel.output_trim(
                    plan=sp, op_id=prefix + "sp_output_trim", deps=(previous,),
                    stage=0, layer=-1, requests=requests, phase=mode,
                    width=model.hidden_size, dtype=model.activation_dtype)
                ops.append(trimmed)
                previous = trimmed.op_id
        from .sparse_mla_metadata import before_forward
        return before_forward(ops, ready), previous

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
            final = norm(name=prefix + "final_norm", deps=(last,), stage=stage, layer=-1,
                         requests=tuple(view.request_id for view in views), tokens=input_tokens,
                         width=model.hidden_size, model=model, mode=mode)
            ops.append(final)
            last = final.op_id
        if not rows:
            return ops, last
        if rows != input_tokens:
            select = self.select_hidden(model, views, last, prefix=prefix, mode=mode, stage=stage)
            ops.append(select)
            last = select.op_id
        if not normalized and runtime.final_norm_scope == "selected":
            final = norm(name=prefix + "final_norm", deps=(last,), stage=stage, layer=-1,
                         requests=requests, tokens=rows, width=model.hidden_size, model=model, mode=mode)
            ops.append(final)
            last = final.op_id
        logits = gemm_op(
            op_id=prefix + "logits", deps=(last,), stage=stage, layer=-1, requests=requests,
            m=rows, n=math.ceil(model.vocab_size / p.tp), k=model.hidden_size, quant=model.quant,
            category="lm_head", phase=mode, family="logits", recipe="vocab_logits",
        )
        ops.append(logits)
        last = logits.op_id
        if p.tp > 1:
            gather = comm_op(
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
        sample = vector_op(
            op_id=prefix + "sample", deps=(last,), stage=stage, layer=-1, requests=requests,
            elements=rows * model.vocab_size, ops_per_element=5, dtype="fp32",
            family="sampling", phase=mode, recipe="sample_logits", shape=(rows, model.vocab_size),
        )
        ops.append(sample)
        return ops, sample.op_id

    @staticmethod
    def select_hidden(model, views, ready, *, prefix="", mode="decode", stage=0):
        sampled = tuple(view for view in views if view.samples)
        rows = sum(view.q_len if view.phase == "decode" else 1 for view in sampled)
        payload = rows * model.hidden_size * dtype_bytes(model.activation_dtype)
        return vector_op(
            op_id=prefix + "select_hidden", deps=(ready,), stage=stage, layer=-1,
            requests=tuple(view.request_id for view in sampled), elements=rows * model.hidden_size,
            ops_per_element=0, dtype=model.activation_dtype, family="tensor_select", phase=mode,
            recipe="select_hidden", shape=(rows, model.hidden_size),
            read_bytes_override=payload + rows * 8, write_bytes_override=payload,
            metadata=(("selection", "all_verify_rows_or_last_prefill_row"),))

    @staticmethod
    def partition_views(views, runtime, padded):
        degree = runtime.parallel.pcp
        if degree == 1 or not any(v.phase == "prefill" for v in views):
            return views, padded, False
        costs = [0.0] * degree
        for view in views:
            if view.phase == "prefill":
                split = split_pcp_pairs(view.q_len, view.context, degree)
                costs = [old + current for old, current in zip(costs, split.per_rank)]
        rank = max(range(degree), key=costs.__getitem__)
        ranges = [pcp_rank_chunk_ranges(view.q_len, degree, rank) for view in views]
        local = tuple(replace(view, q_len=sum(end - start for start, end in parts),
                              global_q_len=view.q_len, query_ranges=parts)
                      if view.phase == "prefill" else view for view, parts in zip(views, ranges))
        return local, sum(view.q_len for view in local), True

    def __call__(self, plan, model, runtime, machine, *, global_tokens=None, global_views=None,
                 resource_fusion=True, replica=0):
        model = normalize_model(model)
        # An empty local query set can still receive expert rows. Its execution
        # phase belongs to the active cohort, not a fictitious "idle" kernel.
        mode = selected_mode(plan if plan.requests else replace(
            plan, requests=tuple(global_views or ())))
        views = tuple(replace(view, q_len=view.q_len + view.draft_proposal_tokens)
                      for view in plan.requests)
        valid = sum(view.q_len for view in views)
        bucket, scope = self.graph.select(views, runtime) if views else (None, "none")
        padded = max(valid, bucket or 0)
        requests = tuple(view.request_id for view in views)
        ops = []
        entry = barrier_op(
            op_id="iteration_input", deps=(), stage=0, layer=-1, requests=requests, phase=mode,
            recipe="input_ready", shape=(valid, model.hidden_size),
        )
        ops.append(entry)
        previous = entry.op_id
        if any(view.draft_proposal_tokens for view in (global_views or plan.requests)):
            from .speculation import draft_program
            draft_ops, previous = draft_program(self, model, runtime, plan.requests, previous,
                                                global_views=global_views, replica=replica)
            ops.extend(draft_ops)
        media_counts = {str(request): int(count) for request, count in
                        _media_counts(runtime).items()}
        for index, view in enumerate(views):
            if view.media_encoded or view.phase != "prefill" or not media_counts.get(view.request_id):
                continue
            media = lower_media(
                model=model, runtime=runtime, parallel=runtime.parallel, quant=model.quant,
                mode=mode, views=(view,), media_tokens_override=media_counts[view.request_id],
            )
            rename = {op.op_id: f"media{index}:{op.op_id}" for op in media.ops}
            ops.extend(replace(op, op_id=rename[op.op_id],
                               deps=tuple(rename[dep] for dep in op.deps) or (previous,))
                       for op in media.ops)
            previous = rename[media.last]
        local_views, local_tokens, pcp_prefill = self.partition_views(views, runtime, padded)
        if valid:
            embedding = vector_op(
                op_id="embedding", deps=(previous,), stage=0, layer=-1, requests=requests,
                elements=local_tokens * model.hidden_size, ops_per_element=1,
                dtype=model.activation_dtype, family="embedding", phase=mode, recipe="embedding",
            )
            ops.append(embedding)
            previous = embedding.op_id
        body, previous = self.body(model, runtime, local_views, ready=previous, tokens=local_tokens,
                                   mode=mode, global_tokens=global_tokens, replica=replica)
        ops.extend(body)
        if pcp_prefill:
            if runtime.final_norm_scope == "all":
                final = norm(name="final_norm", deps=(previous,), stage=runtime.parallel.pp - 1,
                             layer=-1, requests=requests, tokens=local_tokens, width=model.hidden_size,
                             model=model, mode=mode)
                ops.append(final)
                previous = final.op_id
            select = self.select_hidden(model, views, previous, mode=mode, stage=runtime.parallel.pp - 1)
            ops.append(select)
            previous = select.op_id
            rows = sum(v.q_len if v.phase == "decode" else 1 for v in views if v.samples)
            gather = comm_op(
                op_id="pcp_hidden_gather", deps=(previous,), stage=runtime.parallel.pp - 1,
                layer=-1, requests=requests, family="allgather", phase=mode,
                recipe="pcp_hidden_gather", link_bytes=rows * model.hidden_size * dtype_bytes(model.activation_dtype)
                * (runtime.parallel.pcp - 1) / runtime.parallel.pcp,
                shape=(rows, model.hidden_size), metadata=(("group_axis", "pcp"),))
            ops.append(gather)
            previous = gather.op_id
        heads, previous = self.logits(
            model, runtime, views, previous, mode=mode,
            input_tokens=rows if pcp_prefill else padded,
            normalized=pcp_prefill and runtime.final_norm_scope == "all")
        ops.extend(heads)
        proposal_count = sum(view.draft_proposal_tokens for view in plan.requests)
        if proposal_count:
            accept = vector_op(
                op_id="accept", deps=(previous,), stage=runtime.parallel.pp - 1, layer=-1,
                requests=requests, elements=proposal_count, ops_per_element=6,
                dtype="fp32", family="acceptance", phase=mode, recipe="speculative_acceptance",
            )
            ops.append(accept)
            previous = accept.op_id
        committed = barrier_op(
            op_id="token_commit", deps=(previous,), stage=runtime.parallel.pp - 1, layer=-1,
            requests=tuple(view.request_id for view in views if view.samples),
            phase=mode, recipe="token_commit", shape=(len(requests),),
        )
        ops.append(replace(committed, family="token_commit"))
        preparation_views = tuple(view for view in plan.requests if view.phase == "prefill"
                                  and view.draft_prepare)
        global_preparation = tuple(view for view in (global_views or plan.requests)
                                   if view.phase == "prefill" and view.draft_prepare)
        if dict(runtime.metadata).get("draft_mode") and (
                preparation_views or global_views and global_preparation):
            from .speculation import draft_program
            preparation, _ = draft_program(
                self, model, runtime, preparation_views, committed.op_id,
                global_views=global_preparation if global_views else None, prefill=True,
                replica=replica)
            ops.extend(preparation)
        ops = self.graph.apply(lower_implementations(
            ops, runtime, model.activation_dtype, resource_fusion=False,
            local_tokens=local_tokens), scope, runtime)
        if resource_fusion:
            from .native_fusion import lower_native_fusions
            ops = lower_native_fusions(ops, runtime, local_tokens=local_tokens)
        return IterationWork(
            ops=ops,
            plan=plan, stage_count=runtime.parallel.pp,
            valid_tokens=valid, padded_tokens=padded, graph_bucket=bucket, mode=mode,
            recipe="structural", replay_scope=scope,
        )


def _media_counts(runtime):
    import json
    raw = dict(runtime.metadata).get("media_tokens_by_request", "{}")
    return json.loads(raw) if isinstance(raw, str) else raw
