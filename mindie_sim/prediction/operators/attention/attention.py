"""Attention operators selected by semantic kind and implementation policy.

[Role] Lower GQA directly and adapt structural MLA/DSA and recurrent programs.
[Invariants] Real request lengths determine pairs and cache traffic. Query
  padding changes projections, not valid tokens. Kernel batching and KV reuse
  are software parameters independent of availability of measurements.
[Timeouts & Locks] Bounded by requests and layers; pure arithmetic.
[Side-Effect Policy] Returns new OpSpec records; no hardware work or I/O.
"""
from __future__ import annotations

from dataclasses import replace
from functools import partial
import math

from mindie_sim.catalog.attention_common import (
    AttentionResult, _allreduce, _causal_pairs, _long_selected_pairs,
    min_positive,
)
from mindie_sim.prediction.operators.mla.mla import mla
from mindie_sim.prediction.operators.recurrent.recurrent import linear
from mindie_sim.prediction.sharding.context_parallel import (
    CP_CHUNKED_PREFILL_STREAM,
    CP_DECODE_COMM_STREAM,
    active_layer_dcp,
    emit_dcp_output_exchange_combine,
    emit_dcp_query_gather,
    emit_mtp_current_attention,
    emit_pcp_o_proj_weight_allgather,
    emit_pcp_restore_unpermute,
    pcp_o_proj_weight_sharding_enabled,
    pcp_uniform_peer_rows,
    resolve_cp_rank_candidates,
    resolve_dcp_cohort_decode,
    should_split_mtp_dcp,
)
from .partial_attention import projection_only, unmodeled, shared_kv_input
from mindie_sim.prediction.memory_traffic import (AttentionAccessSpec, AttentionSchedule,
                             plan_attention_traffic)
from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.cache_format import CacheFormat
from mindie_sim.catalog.ops import gemm_op, vector_op, comm_op, barrier_op
from mindie_sim.catalog.context_parallel import (
    layer_dcp_degree, layer_dcp_factors, owned_positions, split_causal_pairs,
)
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.common import dtype_bytes
from .batch_prefix_attention import (
    annotate_sparse_prefix_unknown, apply_attention_prefix_reads,
)
from .gqa_contracts import (
    AttentionPolicy,
    _CacheProducer,
    _QKV_NORM_ROPE_SOURCE,
    _attention_tensor_view,
    _cache_tensor_view,
    _contiguous_tensor_view,
    _decode_contract,
    _declare_qkv_norm_rope,
    _gqa_formats,
    _memory_hierarchy,
    _prefill_contract,
    _qkv_output_copy,
    _qkv_views,
    _store_cache,
)
from mindie_sim.prediction.operators.sparse_indexer.gqa_indexer import lower_gqa_indexer
from mindie_sim.prediction.implementation_selection import is_caller_pinned_binding
from mindie_sim.prediction.prefix_reads import _merged_interval_rows, owned_window_intervals
from mindie_sim.prediction.operators.sparse_indexer.sparse_attention import sparse_attention_traffic


OUTPUT_GATE_OPS_PER_ELEMENT = 5
RMS_NORM_OPS_PER_ELEMENT = 5

#: One rotate-half pair contributes a sine/cosine vector expression per
#: rotated element: rotate, multiply, add, plus the sin/cos table lookup.
ROTARY_OPS_PER_ELEMENT = 6
GATE_SIGMOID_OPS_PER_ELEMENT = 4
GATE_APPLY_OPS_PER_ELEMENT = 1
FLOPS_PER_FMA = 2
SOFTMAX_VECTOR_OPS_PER_PAIR = 5
DEFAULT_CP_INTERLEAVE = 128
INT32_BYTES = 4


def _lower_gqa_dcp_query(
    ops, *, query_producer, views, parallel, prefix, stage, plan, requests, mode,
    model, heads, width, padded_tokens, layer_dcp, cacheless, pcp_factor, tp_gather,
    dp_width, cohort_dec_views, peer_dec_rows, local_dp_pos, idle_dcp_owner,
):
    has_decode_dcp = layer_dcp > 1 and (
        any(v.phase != "prefill" for v in views) or (dp_width > 1 and bool(cohort_dec_views))
    )
    has_chunked_pf_dcp = (
        layer_dcp > 1 and not cacheless
        and any(v.phase == "prefill" and v.context > 0 for v in views)
    )
    dcp_query_producer = query_producer
    dcp_heads = heads
    chunked_pf_only = has_chunked_pf_dcp and not has_decode_dcp
    if has_chunked_pf_dcp and parallel.pcp > 1:
        # Upstream ``attention_cp.py:695`` slices out decode rows before the PCP
        # history-query AllGather so only prefill tokens are gathered/restored.
        pf_only_views = tuple(v for v in views if v.phase == "prefill")
        pcp_q_local, pf_global_q = pcp_uniform_peer_rows(
            pf_only_views, parallel.pcp, local_rows=0)
        q_row_bytes = float(heads * width) * dtype_bytes(model.activation_dtype)
        pcp_q_ag = comm_op(
            op_id=f"{prefix}.pcp_query_allgather", deps=(query_producer,),
            stage=stage, layer=plan.index, requests=requests, family="allgather",
            phase=mode, recipe="pcp_query_allgather",
            link_bytes=float(pcp_q_local) * q_row_bytes * float(parallel.pcp - 1),
            shape=(pcp_q_local * parallel.pcp, heads, width),
            metadata=(("group_axis", "pcp"), ("stream", CP_CHUNKED_PREFILL_STREAM)),
        )
        ops.append(replace(pcp_q_ag, dtype=model.activation_dtype, stream=CP_CHUNKED_PREFILL_STREAM))
        dcp_query_producer = emit_pcp_restore_unpermute(
            ops, prefix=prefix, name="pcp_query_restore", stage=stage, layer=plan.index,
            requests=requests, mode=mode, dep=pcp_q_ag.op_id,
            global_rows=pf_global_q, row_bytes=q_row_bytes,
            dtype=model.activation_dtype, stream=CP_CHUNKED_PREFILL_STREAM,
        )
    if (has_decode_dcp or has_chunked_pf_dcp) and (tp_gather > 1 or dp_width > 1):
        dcp_query_producer, dcp_heads, _ = emit_dcp_query_gather(
            ops, prefix=prefix, stage=stage, layer=plan.index,
            requests=requests, mode=mode, deps=(dcp_query_producer,),
            rows=(0 if idle_dcp_owner else max(1, padded_tokens)),
            local_heads=heads, head_width=width,
            act_dtype=model.activation_dtype, tp_gather=tp_gather,
            dp_width=dp_width, pack_inputs=False,
            stream=CP_CHUNKED_PREFILL_STREAM if chunked_pf_only else "",
            peer_rows=(peer_dec_rows if (dp_width > 1 and has_decode_dcp) else None),
            local_dp_pos=local_dp_pos, pcp_factor=pcp_factor,
        )
    return dcp_query_producer, dcp_heads


def gqa(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
        padded_tokens, policy=AttentionPolicy(), cp_interleave_size=DEFAULT_CP_INTERLEAVE,
        runtime=None, cache_ids=None, stage_of=None, topk_ids=None,
        index_cache_ids=None, candidate_ids=None, machine=None, hierarchy=None,
        dcp_cohort_views_by_member=None, dcp_local_pos=0,
        physical_rank=None, explicit_physical_rank=False, **_) -> AttentionResult:
    from mindie_sim.prediction.implementations import add_input_ports, select

    cacheless = plan.cache_kind == "none"
    if cacheless:
        views = tuple(replace(v, context=0, prefix_tokens=0,
                              prefix_cache_key="") for v in views)
        quant = replace(quant, kv_dtype=model.activation_dtype)
    selected = select(runtime.kernels.get("lowerings", {}), "gqa_attention") if runtime else None
    if selected:
        selected.module.validate(views=views, plan=plan, parallel=parallel, policy=policy,
                                 quant=quant, model=model, **selected.options)
    heads = plan.num_heads // parallel.tp
    kv_heads = local_kv_heads(plan.num_kv_heads, parallel.tp)
    width = plan.head_dim
    key_fmt, value_fmt, cache_regions, shared_format = _gqa_formats(model)
    if cacheless:
        key_fmt = value_fmt = CacheFormat(model.activation_dtype)
        shared_format = True
    shared = not cacheless and 0 <= plan.kv_owner_layer_id < plan.index
    # ``k_eq_v`` shares the *projection weight*; it never shares the post-norm /
    # post-RoPE cache tensor.  K and V are distinct planes for store, read and
    # PCP gather, so capacity and execution bytes conserve.
    projection_planes = 1 if plan.k_eq_v else 2
    cache_planes = 2
    projected_kv_heads = 0 if shared else kv_heads
    configured_recipes = runtime.kernels.get("recipes", {}) if runtime else {}
    if cacheless:
        attention_recipe = ("flash_attention_prefill"
                            if "flash_attention_prefill" in configured_recipes
                            else "flash_attention")
    elif mode in {"decode", "draft"} and "flash_attention_decode" in configured_recipes:
        attention_recipe = "flash_attention_decode"
    elif (mode in {"prefill", "draft_prefill"}
          and "flash_attention_prefill" in configured_recipes):
        attention_recipe = "flash_attention_prefill"
    else:
        attention_recipe = "flash_attention"
    pcp_factor, tp_gather, raw_dp_width = layer_dcp_factors(plan, parallel)
    dp_width = raw_dp_width if int(getattr(parallel, "dp", 1) or 1) >= raw_dp_width else 1
    layer_dcp = layer_dcp_degree(plan, parallel)
    cohort_dec_views, peer_dec_rows, local_dp_pos = resolve_dcp_cohort_decode(
        views, dp_width=dp_width,
        cohort_views_by_member=dcp_cohort_views_by_member,
        local_dp_pos=dcp_local_pos)
    idle_dcp_owner = (
        padded_tokens == 0 and not views and layer_dcp > 1
        and dp_width > 1 and bool(cohort_dec_views)
    )
    requests = tuple(view.request_id for view in (cohort_dec_views if idle_dcp_owner else views))
    hierarchy = _memory_hierarchy(machine, hierarchy=hierarchy)
    gate_separate = plan.q_gate_separate
    if idle_dcp_owner:
        ops, store_ids, topk_id, index_store_id, candidate_id = [], (), None, None, None
        eff_topk, sel_topk = 0, 0
        block_size = int(getattr(runtime, "kv_block_size", DEFAULT_CP_INTERLEAVE)
                         or DEFAULT_CP_INTERLEAVE)
        query_producer = deps
    else:
        (ops, qkv, rope, separate_gate, split_views, split_copies,
         normalized) = _gqa_projections(
            prefix, plan, stage, deps, model, quant, mode, padded_tokens, requests,
            heads, projected_kv_heads, projection_planes, width, shared, gate_separate)
        valid, pcp_prefill, store_ids = _publish_gqa_cache(
            ops, rope, split_copies, shared, views, parallel, prefix, stage, plan, requests, mode,
            cache_planes, kv_heads, width, quant, cp_interleave_size, cacheless, cache_ids,
            stage_of, shared_format, model, key_fmt, value_fmt)
        eff_topk, sel_topk, block_size, index_store_id, topk_id, candidate_id = lower_gqa_indexer(
            ops,
            prefix=prefix,
            plan=plan,
            views=views,
            stage=stage,
            deps=deps,
            model=model,
            parallel=parallel,
            quant=quant,
            mode=mode,
            padded_tokens=padded_tokens,
            valid=valid,
            pcp_prefill=pcp_prefill,
            runtime=runtime,
            topk_ids=topk_ids,
            index_cache_ids=index_cache_ids,
            machine=machine,
        )
        query_producer = rope.op_id
    eff_window = min_positive(plan.window, eff_topk)
    dense_win = int(plan.window or 0)
    dcp_query_producer, dcp_heads = _lower_gqa_dcp_query(
        ops, query_producer=query_producer, views=views, parallel=parallel,
        prefix=prefix, stage=stage, plan=plan, requests=requests, mode=mode,
        model=model, heads=heads, width=width, padded_tokens=padded_tokens,
        layer_dcp=layer_dcp, cacheless=cacheless, pcp_factor=pcp_factor,
        tp_gather=tp_gather, dp_width=dp_width, cohort_dec_views=cohort_dec_views,
        peer_dec_rows=peer_dec_rows, local_dp_pos=local_dp_pos,
        idle_dcp_owner=idle_dcp_owner)
    local_attn_deps = ((*store_ids, topk_id, query_producer) if topk_id
                       else (*store_ids, query_producer))
    dcp_attn_deps = ((*store_ids, topk_id, dcp_query_producer) if topk_id
                     else (*store_ids, dcp_query_producer))
    streams = kv_heads if policy.reuse_kv_across_query_heads else heads
    kv_row_bytes = (
        key_fmt.row_bytes(cache_planes * streams * width)
        if shared_format
        else key_fmt.row_bytes(streams * width) + value_fmt.row_bytes(streams * width)
    )
    user_pinned_attn = (
        is_caller_pinned_binding(runtime, "families", "attention")
        or is_caller_pinned_binding(runtime, "recipes", attention_recipe)
        or is_caller_pinned_binding(runtime, "recipes", "flash_attention")
    )
    request_op = partial(
        _gqa_request_op, kv_heads=kv_heads, local_heads=heads, dcp_heads=dcp_heads,
        dp_width=dp_width, width=width,
        kv_row_bytes=kv_row_bytes, plan=plan, mode=mode, hierarchy=hierarchy,
        policy=policy, model=model, sel_topk=sel_topk, topk_id=topk_id,
        eff_topk=eff_topk, prefix=prefix, stage=stage,
        local_attn_deps=local_attn_deps, dcp_attn_deps=dcp_attn_deps,
        attention_recipe=attention_recipe, streams=streams, quant=quant,
        parallel=parallel, eff_window=eff_window, runtime=runtime, cacheless=cacheless,
        selected=selected, query_producer=query_producer,
        dcp_query_producer=dcp_query_producer, add_input_ports=add_input_ports)
    attention = _append_gqa_attention(
        ops, views, parallel, prefix, eff_topk, eff_window, plan, policy, dense_win,
        cp_interleave_size, block_size, kv_heads, heads, dcp_heads, width, kv_row_bytes,
        mode, hierarchy, model, stage, streams, quant, runtime, cacheless, selected,
        user_pinned_attn, request_op, pcp_factor=pcp_factor, tp_gather=tp_gather,
        dp_width=dp_width, local_attn_deps=local_attn_deps,
        cohort_decode_views=(cohort_dec_views if dp_width > 1 else None),
        peer_decode_rows=(peer_dec_rows if dp_width > 1 else None),
        local_dp_pos=local_dp_pos, physical_rank=physical_rank,
        explicit_dp_cohort=(dcp_cohort_views_by_member is not None),
        explicit_physical_rank=bool(explicit_physical_rank))
    if idle_dcp_owner:
        return AttentionResult(ops=ops, last=attention[-1],
                               cache_store_id=None, index_store_id=None,
                               topk_id=None, candidate_id=None)
    last = _gqa_output(
        ops, attention, prefix, separate_gate, stage, plan, requests, padded_tokens, heads,
        width, model, mode, split_copies, qkv, quant, parallel, shared, normalized, rope,
        split_views, gate_separate, views=views, runtime=runtime, deps=deps)
    return AttentionResult(ops=ops, last=last,
                           cache_store_id=None if cacheless else store_ids[0],
                           index_store_id=index_store_id, topk_id=topk_id,
                           candidate_id=candidate_id)


def _gqa_projections(
    prefix, plan, stage, deps, model, quant, mode, padded_tokens, requests, heads,
    projected_kv_heads, projection_planes, width, shared, gate_separate,
):
    q_width = plan.q_projection_width(heads, width)
    kv_width = projected_kv_heads * width
    total_n = q_width + projection_planes * kv_width
    packed_slices: list[dict[str, object]] = [
        {"name": "q", "shape": (padded_tokens, q_width),
         "strides": (total_n, 1), "offset_elements": 0,
         "units": heads, "unit_width": q_width // max(1, heads), "shard": "out_tp"},
    ]
    if kv_width > 0:
        packed_slices.append({
            "name": "k", "shape": (padded_tokens, kv_width),
            "strides": (total_n, 1), "offset_elements": q_width,
            "units": projected_kv_heads, "unit_width": width, "shard": "kv_out",
        })
        if projection_planes > 1:
            packed_slices.append({
                "name": "v", "shape": (padded_tokens, kv_width),
                "strides": (total_n, 1), "offset_elements": q_width + kv_width,
                "units": projected_kv_heads, "unit_width": width, "shard": "kv_out",
            })
    qkv = gemm_op(
        op_id=prefix + ".qkv", deps=(deps,), stage=stage, layer=plan.index,
        requests=requests, m=padded_tokens, n=total_n,
        k=model.hidden_size, quant=quant, category="attn.qkv", phase=mode,
        family="qkv_projection", recipe="qkv_projection",
        metadata=(("packed_projection_slices", tuple(packed_slices)),),
    )
    ops = [qkv]
    ready = qkv.op_id
    separate_gate = None
    if gate_separate:
        gate_proj = gemm_op(
            op_id=prefix + ".g_proj", deps=(deps,), stage=stage, layer=plan.index,
            requests=requests, m=padded_tokens, n=heads, k=model.hidden_size,
            quant=quant, category="attn.gate", phase=mode,
            family="gate_projection", recipe="attention_q_gate_projection",
        )
        gate_sigmoid = vector_op(
            op_id=prefix + ".g_sigmoid", deps=(gate_proj.op_id,), stage=stage,
            layer=plan.index, requests=requests, elements=padded_tokens * heads,
            ops_per_element=GATE_SIGMOID_OPS_PER_ELEMENT, dtype="fp32",
            family="sigmoid", phase=mode,
            recipe="attention_q_gate_sigmoid",
            shape=(padded_tokens, heads),
        )
        ops.extend((gate_proj, gate_sigmoid))
        separate_gate = gate_sigmoid.op_id
    split_views, split_copies = {}, []
    normalized = None
    if plan.qk_norm:
        split_views = _qkv_views(
            padded_tokens, heads, projected_kv_heads, width,
            plan.q_output_gate and not gate_separate, plan.k_eq_v)
        normalized = vector_op(
            op_id=prefix + ".qk_norm", deps=(ready,), stage=stage, layer=plan.index,
            requests=requests, elements=padded_tokens * (heads + projected_kv_heads) * width,
            ops_per_element=RMS_NORM_OPS_PER_ELEMENT, dtype=model.activation_dtype, family="rmsnorm",
            phase=mode, recipe="qk_norm", shape=(padded_tokens, heads + projected_kv_heads, width),
            metadata=(("input_views", {
                name: split_views[name].to_dict()
                for name in (("query",) if shared else ("query", "key"))}),))
        ops.append(normalized)
        ready = normalized.op_id
        split_copies = [
            _qkv_output_copy(qkv, split_views[name], name, model.activation_dtype)
            for name in ("value", "gate")
            if name in split_views and (name != "value" or not shared)
        ]
        ops.extend(split_copies)
    # A declared partial-RoPE width rotates only that prefix of each head; the
    # remaining tail is an unchanged view, never a shrunk projection.
    rotary_width = (plan.qk_rope_head_dim
                    if plan.qk_rope_head_dim or plan.qk_nope_head_dim else width)
    rope = vector_op(
        op_id=prefix + ".rope", deps=(ready,), stage=stage, layer=plan.index,
        requests=requests,
        elements=padded_tokens * (heads + projected_kv_heads) * rotary_width,
        ops_per_element=ROTARY_OPS_PER_ELEMENT, dtype=model.activation_dtype,
        family="rotary", phase=mode,
        recipe="rotary", shape=(padded_tokens, heads + projected_kv_heads, rotary_width),
        metadata=(("rotary_dim", rotary_width), ("head_dim", width),
                  ("tail_policy", "unchanged_view")),
    )
    return ops, qkv, rope, separate_gate, split_views, split_copies, normalized


def _publish_gqa_cache(
    ops, rope, split_copies, shared, views, parallel, prefix, stage, plan, requests, mode,
    cache_planes, kv_heads, width, quant, cp_interleave_size, cacheless, cache_ids, stage_of,
    shared_format, model, key_fmt, value_fmt,
):
    valid = sum(view.q_len for view in views)
    store_ready = rope.op_id
    value_ready = (split_copies[VALUE_COPY].op_id
                   if split_copies and not shared else None)
    pcp_prefill = not shared and parallel.pcp > 1 and any(v.phase == "prefill" for v in views)
    if pcp_prefill:
        gather = comm_op(
            op_id=prefix + ".kv_gather",
            deps=(rope.op_id, value_ready) if value_ready else (rope.op_id,),
            stage=stage, layer=plan.index,
            requests=requests, family="allgather", phase=mode, recipe="pcp_kv_gather",
            link_bytes=cache_planes * kv_heads * width * dtype_bytes(quant.kv_dtype)
                       * sum(v.global_q_len or v.q_len for v in views if v.phase == "prefill")
                       * (parallel.pcp - 1) / parallel.pcp,
            shape=(valid, kv_heads, width), metadata=(("group_axis", "pcp"),))
        ops.append(gather)
        store_ready = gather.op_id
    # The gather consumes rotary output, so rotary must precede cache writes
    # in the emitted graph order; it is also a dependency of the gather.
    ops.insert(ops.index(gather) if pcp_prefill else len(ops), rope)
    max_dcp = max((active_layer_dcp(plan, parallel, v.phase) for v in views), default=1)
    store_rows = max(
        sum(
            (v.global_q_len or v.q_len)
            if active_layer_dcp(plan, parallel, v.phase) <= 1 else
            owned_positions(v.context,
                            v.context + (v.global_q_len or v.q_len) - 1,
                            active_layer_dcp(plan, parallel, v.phase),
                            cp_interleave_size, rank)
            for v in views)
        for rank in range(max_dcp))
    store_deps = ((store_ready, value_ready)
                  if value_ready and not pcp_prefill else (store_ready,))
    if cacheless:
        store_ids = store_deps
    elif shared:
        extra, owner_ready = shared_kv_input(
            prefix=prefix, plan=plan, stage=stage, views=views,
            cache_ids=cache_ids or {}, stage_of=stage_of,
            parallel=parallel, quant=quant, mode=mode)
        ops.extend(extra)
        readiness = barrier_op(
            op_id=prefix + ".shared_cache_ready", deps=(rope.op_id, owner_ready),
            stage=stage, layer=plan.index, requests=requests, phase=mode,
            recipe="shared_cache_dependency", shape=())
        ops.append(readiness)
        store_ids = (readiness.op_id,)
    elif shared_format:
        store_ids = (_store_cache(
            ops, prefix=prefix, deps=store_deps, rows=store_rows,
            width=cache_planes * kv_heads * width, region="gqa_kv",
            stage=stage, layer=plan.index, requests=requests, phase=mode,
            model=model, views=views, kv_heads=kv_heads, head_width=width,
            parallel=parallel, quant=quant, key_fmt=key_fmt, value_fmt=value_fmt,
            shared_format=True),)
    else:
        key_store = _store_cache(
            ops, prefix=prefix, deps=store_deps, rows=store_rows, width=kv_heads * width,
            region="gqa_kv_key", name="cache_key", stage=stage, layer=plan.index, requests=requests,
            phase=mode, model=model, views=views, kv_heads=kv_heads, head_width=width,
            parallel=parallel, quant=quant, key_fmt=key_fmt, value_fmt=value_fmt, shared_format=False)
        value_store = _store_cache(
            ops, prefix=prefix, deps=store_deps, rows=store_rows, width=kv_heads * width,
            region="gqa_kv_value", name="cache_value", stage=stage, layer=plan.index, requests=requests,
            phase=mode, model=model, views=views, kv_heads=kv_heads, head_width=width,
            parallel=parallel, quant=quant, key_fmt=key_fmt, value_fmt=value_fmt, shared_format=False)
        store_ids = (key_store, value_store)
    return valid, pcp_prefill, store_ids


def _gqa_rank_work(
    indexed_views, eff_cp, eff_topk, eff_window, plan, policy, dense_win, cp_interleave_size,
    block_size,
):
    rank_pairs = [0.0] * eff_cp
    rank_sweeps = [0] * eff_cp
    rank_rows = [0] * eff_cp
    per_view_pairs: list[tuple[float, list[float]]] = []
    per_view_sweeps: list[list[int]] = []
    per_view_shared: list[list[int]] = []
    per_view_rows: list[list[int]] = []
    per_view_spans: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    for _, view in indexed_views:
        q, context = view.q_len, view.context
        ranges = view.query_ranges or ((0, q),)
        selected_q = sum(end - start for start, end in ranges if end > start)
        win_spans: list[tuple[int, int]] = []
        sh_spans: list[tuple[int, int]] = []
        if eff_topk > 0:
            pairs = sum(
                _long_selected_pairs(
                    end - start, context + start, 1, eff_window,
                    noncausal=bool(plan.noncausal_attention), kv_len=view.kv_len)
                for start, end in ranges if end > start
            )
            gathers = sum(
                sparse_attention_traffic(
                    end - start, context + start, eff_window,
                    query_tile=policy.query_tile,
                    noncausal=bool(plan.noncausal_attention),
                    kv_len=view.kv_len).gather_rows
                for start, end in ranges if end > start
            )
            v_pairs = [float(pairs) / eff_cp] * eff_cp
            v_sweeps = [math.ceil(gathers / eff_cp)] * eff_cp
            v_shared = [0] * eff_cp
        else:
            visible_kv = min(view.kv_len, dense_win) if dense_win > 0 else view.kv_len
            pairs = (
                float(selected_q * visible_kv) if plan.noncausal_attention else
                sum(_causal_pairs(end - start, context + start, dense_win)
                    for start, end in ranges if end > start)
            )
            if eff_cp == 1:
                v_pairs = [float(pairs)]
            elif plan.noncausal_attention:
                v_pairs = list(split_causal_pairs(
                    selected_q, view.kv_len - selected_q, eff_cp, cp_interleave_size,
                    window=dense_win, noncausal=True).per_rank)
            else:
                v_pairs = [
                    sum(split_causal_pairs(
                        end - start, context + start, eff_cp, cp_interleave_size,
                        window=dense_win, noncausal=False).per_rank[r]
                        for start, end in ranges if end > start)
                    for r in range(eff_cp)
                ]
            v_sweeps = [0] * eff_cp
            v_shared = [0] * eff_cp
            raw_prefix = max(0, min(int(getattr(view, "prefix_tokens", 0) or 0), int(context)))
            aligned_prefix = (raw_prefix // block_size) * block_size
            for begin, end in ranges:
                for start in range(begin, end, policy.query_tile):
                    if plan.noncausal_attention:
                        hi = view.kv_len
                        lo = max(0, hi - dense_win) if dense_win > 0 else 0
                    else:
                        hi = context + min(end, start + policy.query_tile)
                        lo = max(0, context + start + 1 - dense_win) if dense_win > 0 else 0
                    shared_hi = min(hi, aligned_prefix)
                    if hi > lo:
                        win_spans.append((lo, hi))
                    if shared_hi > lo:
                        sh_spans.append((lo, shared_hi))
                    for rank in range(eff_cp):
                        v_sweeps[rank] += owned_positions(
                            lo, hi - 1, eff_cp, cp_interleave_size, rank)
                        if shared_hi > lo:
                            v_shared[rank] += owned_positions(
                                lo, shared_hi - 1, eff_cp, cp_interleave_size, rank)
        v_rows = (
            [owned_positions(0, view.kv_len - 1, eff_cp, cp_interleave_size, r)
             for r in range(eff_cp)]
            if eff_cp > 1 else [view.kv_len]
        )
        for r in range(eff_cp):
            rank_pairs[r] += v_pairs[r]
            rank_sweeps[r] += v_sweeps[r]
            rank_rows[r] += v_rows[r]
        per_view_pairs.append((float(pairs), v_pairs))
        per_view_sweeps.append(v_sweeps)
        per_view_shared.append(v_shared)
        per_view_rows.append(v_rows)
        per_view_spans.append((win_spans, sh_spans))
    worst_r = max(range(eff_cp), key=lambda r: (rank_pairs[r], rank_sweeps[r]))
    worst_sweeps_r = max(
        range(eff_cp), key=lambda r: (rank_sweeps[r], rank_pairs[r]))
    return (rank_pairs, rank_sweeps, rank_rows, per_view_pairs, per_view_sweeps,
            per_view_shared, per_view_rows, per_view_spans, worst_r, worst_sweeps_r)


def _gqa_request_op(
    orig_idx, view, pairs, v_pairs, v_sweeps, local_pairs, view_sweeps, local_rows, eff_cp, *,
    kv_heads, local_heads, dcp_heads, dp_width, width, kv_row_bytes, plan, mode, hierarchy,
    policy, model, sel_topk, topk_id, eff_topk, prefix, stage,
    local_attn_deps, dcp_attn_deps, attention_recipe, streams, quant,
    parallel, eff_window, runtime, cacheless, selected, query_producer,
    dcp_query_producer, add_input_ports, chunked_pf_pcp=False,
):
    heads = dcp_heads if eff_cp > 1 else local_heads
    q_eff = (int(getattr(view, "global_q_len", 0) or view.q_len)
             if chunked_pf_pcp else int(view.q_len))
    eff_local_pairs = float(local_pairs)
    q_prod = dcp_query_producer if eff_cp > 1 else query_producer
    attention_deps = dcp_attn_deps if eff_cp > 1 else local_attn_deps
    traffic = plan_attention_traffic(
        AttentionAccessSpec(
            q_lengths=(q_eff,),
            context_lengths=(int(max(0, view.kv_len - view.q_len)),),
            kv_heads=int(kv_heads), q_heads=int(heads),
            key_dim=int(width), value_dim=int(width),
            row_bytes=float(kv_row_bytes),
            unique_rows=int(max(0, local_rows)),
            refill_rows=int(max(0, view_sweeps)),
            selected_rows=(int(max(0, view_sweeps))
                           if int(getattr(plan, "index_topk", 0) or 0) > 0
                           else 0),
            phase=mode,
            topk=int(getattr(plan, "index_topk", 0) or 0),
            representation="gqa",
            cache_identity=str(getattr(view, "prefix_key", "") or ""),
        ),
        hierarchy,
        AttentionSchedule(query_tile=policy.query_tile),
    )
    kv_bytes = float(traffic.hbm_read_bytes)
    kv_refill_bytes = float(traffic.private_refill_bytes)
    qk_width = width
    v_width = int(getattr(plan, "v_head_dim", 0) or width)
    element_bytes = dtype_bytes(model.activation_dtype)
    query_bytes = q_eff * heads * qk_width * element_bytes
    output_bytes = q_eff * heads * v_width * element_bytes
    selection_bytes = (float(q_eff * sel_topk * INT32_BYTES)
                       if topk_id and eff_topk > 0 else 0.0)
    shape = (q_eff, view.kv_len, heads, qk_width)
    op = OpSpec(
        op_id=f"{prefix}.attention.{orig_idx}", family="attention", phase=mode,
        stage=stage, layer=plan.index, requests=(view.request_id,),
        semantic_shape=shape, local_shape=(q_eff, local_rows, heads, qk_width),
        padded_shape=shape, launch_shape=shape,
        flops=FLOPS_PER_FMA * eff_local_pairs * heads * (qk_width + v_width),
        vector_ops=SOFTMAX_VECTOR_OPS_PER_PAIR * eff_local_pairs * heads,
        hbm_bytes=kv_bytes + query_bytes + output_bytes + selection_bytes,
        mte2_bytes=(kv_refill_bytes + query_bytes + selection_bytes),
        mte3_bytes=output_bytes,
        deps=attention_deps, recipe=attention_recipe, dtype=model.compute_dtype,
        metadata=(("query_tile", policy.query_tile), ("causal_pairs", pairs),
                  ("local_pairs", eff_local_pairs), ("kv_streams", streams),
                  ("kv_sweeps", view_sweeps),
                  ("attention_traffic_plan", traffic.to_dict()),
                  ("cp_per_rank_pairs", tuple(v_pairs)),
                  ("cp_per_rank_sweeps", tuple(v_sweeps)))
                 + _decode_contract(
                     (view,), heads, kv_heads, width, model, quant, parallel, plan, mode,
                     window=eff_window, cp=eff_cp)
                 + _prefill_contract(
                     (view,), heads, kv_heads, width, model, quant, parallel, plan, mode,
                     window=eff_window, cp=eff_cp)
                 + _attention_tensor_view(
                     (view,), heads, kv_heads, width, model, quant, parallel,
                     policy, runtime, causal=not plan.noncausal_attention,
                     cacheless=cacheless),
    )
    if selected:
        op = selected.module.lower_op(
            op, view=view, plan=plan, key_heads=kv_heads, model=model,
            runtime=runtime, **selected.options)
    if q_prod == query_producer:
        op = add_input_ports(op, ({"op": query_producer, "output": "query"},))
    return op, kv_bytes, kv_refill_bytes


def _build_gqa_requests(
    indexed_views, per_view_pairs, per_view_sweeps, per_view_rows, per_view_spans,
    per_view_shared, worst_r, worst_sweeps_r, eff_cp, cp_interleave_size, request_op,
    *, chunked_pf_pcp: bool = False,
):
    batches: list[OpSpec] = []
    sweeps_by_view: list[int] = []
    shared_sweeps_by_view: list[int] = []
    unique_win_by_view: list[int] = []
    shared_iv_by_view: list[tuple[tuple[int, int], ...]] = []
    total_pairs_sum = 0.0
    total_rows_sum = 0
    kv_hbm_total = 0.0
    kv_mte2_total = 0.0
    kv_components: list[tuple[float, float]] = []
    for v_pos, (orig_idx, view) in enumerate(indexed_views):
        pairs, v_pairs = per_view_pairs[v_pos]
        local_pairs = v_pairs[worst_r]
        view_sweeps = per_view_sweeps[v_pos][worst_sweeps_r]
        local_rows = per_view_rows[v_pos][worst_r]
        win_spans, sh_spans = per_view_spans[v_pos]
        owned_win = owned_window_intervals(
            win_spans, eff_cp, cp_interleave_size, worst_sweeps_r)
        owned_sh = owned_window_intervals(
            sh_spans, eff_cp, cp_interleave_size, worst_sweeps_r)
        sweeps_by_view.append(view_sweeps)
        shared_sweeps_by_view.append(min(view_sweeps, per_view_shared[v_pos][worst_sweeps_r]))
        unique_win_by_view.append(_merged_interval_rows(owned_win))
        shared_iv_by_view.append(owned_sh)
        total_pairs_sum += pairs
        total_rows_sum += view.kv_len
        op, kv_bytes, kv_refill_bytes = request_op(
            orig_idx, view, pairs, v_pairs, per_view_sweeps[v_pos], local_pairs,
            view_sweeps, local_rows, eff_cp, chunked_pf_pcp=chunked_pf_pcp)
        kv_hbm_total += kv_bytes
        kv_mte2_total += kv_refill_bytes
        kv_components.append((float(kv_bytes), float(kv_refill_bytes)))
        batches.append(op)
    return (batches, sweeps_by_view, shared_sweeps_by_view, unique_win_by_view,
            shared_iv_by_view, total_pairs_sum, total_rows_sum, kv_hbm_total,
            kv_mte2_total, kv_components)


def _gqa_history_pairs(
    indexed_views, base_per_view_pairs, eff_cp, interleave, window,
    *, chunked_pf: bool = False, use_global_q: bool = False,
):
    """Disjoint history-only pairs per rank for MTP/chunked-prefill DCP split."""
    rank_pairs = [0.0] * eff_cp
    per_view = []
    min_q = 1 if chunked_pf else 2
    for v_pos, (_, view) in enumerate(indexed_views):
        q_hist = (int(getattr(view, "global_q_len", 0) or view.q_len)
                  if use_global_q else int(view.q_len))
        if q_hist >= min_q and view.context > 0:
            hi = int(view.context) - 1
            if window <= 0:
                vp = [float(q_hist * owned_positions(0, hi, eff_cp, interleave, r))
                      for r in range(eff_cp)]
            else:
                vp = [0.0] * eff_cp
                max_i = min(q_hist, max(0, int(window) - 1))
                full_i = max(0, min(max_i, int(window) - int(view.context)))
                if full_i > 0:
                    for r in range(eff_cp):
                        vp[r] += float(full_i * owned_positions(0, hi, eff_cp, interleave, r))
                for i in range(full_i, max_i):
                    lo = int(view.context) + i - int(window) + 1
                    for r in range(eff_cp):
                        vp[r] += float(owned_positions(lo, hi, eff_cp, interleave, r))
            total_p = float(sum(vp))
        else:
            total_p, vp = base_per_view_pairs[v_pos]
        for r in range(eff_cp):
            rank_pairs[r] += vp[r]
        per_view.append((total_p, vp))
    return rank_pairs, per_view


def _group_gqa_phase_views(views, plan, parallel, *, dp_width, cacheless, cohort_decode_views):
    layer_dcp = layer_dcp_degree(plan, parallel)
    phase_groups: list[tuple[int, bool, list[tuple[int, object]]]] = []
    group_map: dict[tuple[int, bool], list[tuple[int, object]]] = {}
    has_dp_decode = dp_width > 1 and layer_dcp > 1 and bool(cohort_decode_views)
    for index, view in enumerate(views):
        if has_dp_decode and view.phase != "prefill":
            continue
        chunked_pf = (
            view.phase == "prefill" and view.context > 0
            and layer_dcp > 1 and not cacheless
        )
        eff_cp = layer_dcp if chunked_pf else active_layer_dcp(plan, parallel, view.phase)
        gkey = (eff_cp, chunked_pf)
        if gkey not in group_map:
            bucket: list[tuple[int, object]] = []
            group_map[gkey] = bucket
            phase_groups.append((eff_cp, chunked_pf, bucket))
        group_map[gkey].append((index, view))
    if has_dp_decode:
        phase_groups.append((layer_dcp, False, list(enumerate(cohort_decode_views))))
    return phase_groups


def _lower_gqa_dcp_combine(
    ops, *, grp_prefix, stage, plan, grp_reqs, mode, grp_attn, grp_views, grp_valid,
    grp_hist_valid, local_heads, width, v_width, kv_row_bytes, model, pcp_factor,
    tp_gather, dp_width, local_attn_deps, split_curr, chunked_pf, chunked_pf_pcp,
    dense_win, peer_decode_rows, local_dp_pos, cp_interleave_size,
):
    current_dep = None
    if split_curr and (not chunked_pf or grp_valid > 0):
        current_deps = (local_attn_deps if chunked_pf
                        else tuple(dict.fromkeys((*grp_attn, *local_attn_deps))))
        current_dep = emit_mtp_current_attention(
            ops, prefix=grp_prefix, stage=stage, layer=plan.index,
            requests=grp_reqs, mode=mode, deps=current_deps,
            views=grp_views, local_heads=local_heads, qk_dim=width,
            v_dim=v_width, kv_row_bytes=kv_row_bytes,
            act_dtype=model.activation_dtype,
            stream=("" if chunked_pf else CP_DECODE_COMM_STREAM),
            allow_q1=chunked_pf, window=dense_win)
    return emit_dcp_output_exchange_combine(
        ops, prefix=grp_prefix, stage=stage, layer=plan.index,
        requests=grp_reqs, mode=mode, deps=tuple(grp_attn),
        rows=max(1, grp_hist_valid), local_heads=local_heads, v_dim=v_width,
        act_dtype=model.activation_dtype, pcp_factor=pcp_factor,
        tp_gather=tp_gather, dp_width=dp_width,
        local_current_dep=current_dep,
        peer_rows=(peer_decode_rows if (dp_width > 1 and not chunked_pf) else None),
        local_dp_pos=local_dp_pos,
        stream=(CP_CHUNKED_PREFILL_STREAM if chunked_pf else ""),
        merge_rows=(grp_valid if chunked_pf_pcp else None),
        meta_extra=(("cp_interleave", int(cp_interleave_size)),
                    ("gqa_chunked_prefill_dcp", bool(chunked_pf))))


def _append_gqa_attention(
    ops, views, parallel, prefix, eff_topk, eff_window, plan, policy, dense_win,
    cp_interleave_size, block_size, kv_heads, local_heads, dcp_heads, width, kv_row_bytes,
    mode, hierarchy, model, stage, streams, quant, runtime, cacheless, selected,
    user_pinned_attn, request_op, *, pcp_factor, tp_gather, dp_width, local_attn_deps,
    cohort_decode_views=None, peer_decode_rows=None, local_dp_pos=0,
    physical_rank=None, explicit_dp_cohort=False, explicit_physical_rank=False,
):
    phase_groups = _group_gqa_phase_views(
        views, plan, parallel, dp_width=dp_width, cacheless=cacheless,
        cohort_decode_views=cohort_decode_views)
    attention: list[str] = []
    multi_group = len(phase_groups) > 1
    v_width = int(getattr(plan, "v_head_dim", 0) or width)
    for grp_idx, (eff_cp, chunked_pf, indexed_views) in enumerate(phase_groups):
        grp_views = tuple(v for _, v in indexed_views)
        grp_reqs = tuple(v.request_id for v in grp_views)
        grp_valid = sum(v.q_len for v in grp_views)
        grp_prefix = f"{prefix}.phase{grp_idx}" if multi_group else prefix
        heads = dcp_heads if eff_cp > 1 else local_heads
        split_curr = (
            chunked_pf
            or should_split_mtp_dcp(plan, grp_views, layer_dcp=eff_cp, sparse=bool(eff_topk > 0))
        )
        chunked_pf_pcp = bool(chunked_pf and parallel.pcp > 1)
        grp_hist_valid = (
            sum(int(getattr(v, "global_q_len", 0) or v.q_len) for v in grp_views)
            if chunked_pf_pcp else grp_valid
        )
        (rank_pairs, rank_sweeps, rank_rows, per_view_pairs, per_view_sweeps,
         per_view_shared, per_view_rows, per_view_spans, worst_r, worst_sweeps_r) = _gqa_rank_work(
            indexed_views, eff_cp, eff_topk, eff_window, plan, policy, dense_win,
            cp_interleave_size, block_size)
        if split_curr and eff_cp > 1:
            rank_pairs, per_view_pairs = _gqa_history_pairs(
                indexed_views, per_view_pairs, eff_cp, cp_interleave_size, dense_win,
                chunked_pf=chunked_pf, use_global_q=chunked_pf_pcp)
            worst_r = max(range(eff_cp), key=lambda r: (rank_pairs[r], rank_sweeps[r]))
        cand_ranks = resolve_cp_rank_candidates(
            parallel, eff_cp=eff_cp, pcp_factor=pcp_factor, tp_gather=tp_gather,
            dp_width=dp_width, local_dp_pos=local_dp_pos, physical_rank=physical_rank,
            explicit_dp_cohort=explicit_dp_cohort,
            explicit_physical_rank=explicit_physical_rank)
        sel_r = max(cand_ranks, key=lambda r: (rank_pairs[r], rank_sweeps[r]))
        sel_sweeps_r = max(cand_ranks, key=lambda r: (rank_sweeps[r], rank_pairs[r]))
        (batches, sweeps_by_view, shared_sweeps_by_view, unique_win_by_view,
         shared_iv_by_view, total_pairs_sum, total_rows_sum, kv_hbm_total,
         kv_mte2_total, kv_components) = _build_gqa_requests(
            indexed_views, per_view_pairs, per_view_sweeps, per_view_rows, per_view_spans,
            per_view_shared, sel_r, sel_sweeps_r, eff_cp, cp_interleave_size,
            request_op, chunked_pf_pcp=chunked_pf_pcp)
        grp_attn: list[str] = []
        if policy.fused_batch and not selected and batches:
            fields = ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes")
            batched = replace(
                batches[0], op_id=grp_prefix + ".attention", requests=grp_reqs,
                semantic_shape=(grp_hist_valid, max(v.kv_len for v in grp_views), heads, width),
                local_shape=(grp_hist_valid, rank_rows[sel_r], heads, width),
                metadata=(("request_shapes", tuple((v.q_len, v.context) for v in grp_views)),
                          ("query_tile", policy.query_tile),
                          ("kv_streams", streams),
                          ("kv_sweeps", sum(sweeps_by_view)),
                          ("cp_per_rank_pairs", tuple(rank_pairs)),
                          ("cp_per_rank_sweeps", tuple(rank_sweeps)),
                          ("cp_bottleneck_pairs_rank", worst_r),
                          ("cp_bottleneck_sweeps_rank", worst_sweeps_r),
                          ("dcp_mtp_split", "history" if split_curr else "unified"))
                         + _decode_contract(
                             grp_views, heads, kv_heads, width, model, quant,
                             parallel, plan, mode, window=eff_window, cp=eff_cp)
                         + _prefill_contract(
                             grp_views, heads, kv_heads, width, model, quant,
                             parallel, plan, mode, window=eff_window, cp=eff_cp)
                         + _attention_tensor_view(
                             grp_views, heads, kv_heads, width, model, quant, parallel,
                             policy, runtime, causal=not plan.noncausal_attention,
                             cacheless=cacheless),
                **{key: sum(getattr(op, key) for op in batches) for key in fields})
            if eff_topk > 0:
                batched = annotate_sparse_prefix_unknown(batched, runtime)
            elif not cacheless:
                read_mode = str(getattr(runtime, "prefix_read_mode", "auto") or "auto")
                reuse_kind = (
                    ("native_per_sequence" if read_mode == "batch_shared" else "paged_l2")
                    if user_pinned_attn
                    else "tile"
                )
                batched = apply_attention_prefix_reads(
                    batched, views=grp_views, sweeps_by_view=sweeps_by_view,
                    shared_sweeps_by_view=shared_sweeps_by_view, row_bytes=float(kv_row_bytes),
                    layer_index=plan.index, runtime=runtime,
                    kernel_reuse=reuse_kind,
                    allow_recipe_rebind=not user_pinned_attn,
                    kv_hbm_bytes=kv_hbm_total,
                    kv_mte2_bytes=kv_mte2_total,
                    unique_rows_by_view=unique_win_by_view,
                    shared_intervals_by_view=shared_iv_by_view,
                    geometry=hierarchy,
                )
            ops.append(batched)
            grp_attn = [batched.op_id]
        elif batches:
            if eff_topk > 0:
                batches = [annotate_sparse_prefix_unknown(item, runtime) for item in batches]
            elif not cacheless:
                for pos, item in enumerate(batches):
                    view = indexed_views[pos][1]
                    kv_hbm_b, kv_mte2_b = kv_components[pos]
                    batches[pos] = apply_attention_prefix_reads(
                        item, views=(view,), sweeps_by_view=(sweeps_by_view[pos],),
                        shared_sweeps_by_view=(shared_sweeps_by_view[pos],),
                        row_bytes=float(kv_row_bytes),
                        layer_index=plan.index, runtime=runtime,
                        kernel_reuse="native_per_sequence", allow_recipe_rebind=False,
                        kv_hbm_bytes=kv_hbm_b,
                        kv_mte2_bytes=kv_mte2_b,
                        unique_rows_by_view=(unique_win_by_view[pos],),
                        shared_intervals_by_view=(shared_iv_by_view[pos],),
                    )
            ops.extend(batches)
            grp_attn = [item.op_id for item in batches]
        if eff_cp > 1 and grp_attn:
            grp_attn = [_lower_gqa_dcp_combine(
                ops, grp_prefix=grp_prefix, stage=stage, plan=plan, grp_reqs=grp_reqs,
                mode=mode, grp_attn=grp_attn, grp_views=grp_views, grp_valid=grp_valid,
                grp_hist_valid=grp_hist_valid, local_heads=local_heads, width=width,
                v_width=v_width, kv_row_bytes=kv_row_bytes, model=model,
                pcp_factor=pcp_factor, tp_gather=tp_gather, dp_width=dp_width,
                local_attn_deps=local_attn_deps, split_curr=split_curr,
                chunked_pf=chunked_pf, chunked_pf_pcp=chunked_pf_pcp,
                dense_win=dense_win, peer_decode_rows=peer_decode_rows,
                local_dp_pos=local_dp_pos, cp_interleave_size=cp_interleave_size)]
        attention.extend(grp_attn)
    return attention


def _gqa_output(
    ops, attention, prefix, separate_gate, stage, plan, requests, padded_tokens, heads, width,
    model, mode, split_copies, qkv, quant, parallel, shared, normalized, rope, split_views,
    gate_separate, *, views=(), runtime=None, deps="",
):
    if gate_separate:
        gate = vector_op(
            op_id=prefix + ".output_gate", deps=tuple(attention) + (separate_gate,),
            stage=stage, layer=plan.index, requests=requests,
            elements=padded_tokens * heads * width,
            ops_per_element=GATE_APPLY_OPS_PER_ELEMENT,
            dtype=model.activation_dtype, family="elementwise", phase=mode,
            recipe="attention_gate",
        )
        ops.append(gate)
        attention = [gate.op_id]
    elif plan.q_output_gate:
        gate_ready = split_copies[-1].op_id if split_copies else qkv.op_id
        gate = vector_op(
            op_id=prefix + ".output_gate", deps=tuple(attention) + (gate_ready,),
            stage=stage, layer=plan.index, requests=requests,
            elements=padded_tokens * heads * width, ops_per_element=OUTPUT_GATE_OPS_PER_ELEMENT,
            dtype=model.activation_dtype, family="elementwise", phase=mode, recipe="attention_gate",
        )
        ops.append(gate)
        attention = [gate.op_id]
    pcp_shard_o = (parallel.pcp > 1 and pcp_o_proj_weight_sharding_enabled(runtime))
    pcp_is_prefill = any(getattr(v, "phase", "") == "prefill" for v in views)
    o_full_width = heads * width
    o_in_width = o_full_width
    o_rows = padded_tokens
    if pcp_shard_o:
        if pcp_is_prefill:
            w_bytes = float(quant.matrix_bytes(model.hidden_size, o_full_width, "attn.o"))
            wg = emit_pcp_o_proj_weight_allgather(
                ops, prefix=prefix, stage=stage, layer=plan.index,
                requests=requests, mode=mode, deps=(deps,) if deps else tuple(attention),
                weight_bytes=w_bytes, pcp=parallel.pcp,
                weight_dtype=model.activation_dtype)
            attention = [*attention, wg]
        else:
            o_in_width = max(1, math.ceil(o_full_width / parallel.pcp))
            if parallel.pcp_shard_decode_requests:
                o_rows = padded_tokens * parallel.pcp
                ag = comm_op(
                    op_id=prefix + ".pcp_o_proj_token_allgather",
                    deps=tuple(attention), stage=stage, layer=plan.index,
                    requests=requests, family="allgather", phase=mode,
                    recipe="pcp_o_proj_token_allgather",
                    link_bytes=padded_tokens * o_full_width * dtype_bytes(model.activation_dtype)
                    * (parallel.pcp - 1),
                    shape=(o_rows, o_full_width),
                    metadata=(("group_axis", "pcp"),),
                )
                ops.append(replace(ag, dtype=model.activation_dtype))
                attention = [ag.op_id]
    output = gemm_op(
        op_id=prefix + ".output", deps=tuple(attention), stage=stage, layer=plan.index,
        requests=requests, m=o_rows, n=model.hidden_size, k=o_in_width,
        quant=quant, category="attn.o", phase=mode, family="output_projection", recipe="attention_output",
    )
    ops.append(output)
    last = output.op_id
    if pcp_shard_o and not pcp_is_prefill:
        if parallel.pcp_shard_decode_requests:
            rs = comm_op(
                op_id=prefix + ".pcp_o_proj_reduce_scatter", deps=(last,),
                stage=stage, layer=plan.index, requests=requests,
                family="reduce_scatter", phase=mode, recipe="pcp_o_proj_reduce_scatter",
                link_bytes=(parallel.pcp - 1) * padded_tokens * model.hidden_size
                * dtype_bytes(model.activation_dtype),
                shape=(padded_tokens, model.hidden_size),
                metadata=(("group_axis", "pcp"),),
            )
            ops.append(replace(rs, dtype=model.activation_dtype))
            last = rs.op_id
        else:
            ar = comm_op(
                op_id=prefix + ".pcp_o_proj_allreduce", deps=(last,),
                stage=stage, layer=plan.index, requests=requests,
                family="allreduce", phase=mode, recipe="pcp_o_proj_allreduce",
                link_bytes=2 * (parallel.pcp - 1) / parallel.pcp * padded_tokens
                * model.hidden_size * dtype_bytes(model.activation_dtype),
                shape=(padded_tokens, model.hidden_size),
                metadata=(("group_axis", "pcp"),),
            )
            ops.append(replace(ar, dtype=model.activation_dtype))
            last = ar.op_id
    if parallel.tp > 1:
        reduction = _allreduce(prefix=prefix, deps=(last,), stage=stage, layer=plan.index,
                               requests=requests, model=model, parallel=parallel,
                               mode=mode, tokens=padded_tokens)
        ops.append(reduction)
        last = reduction.op_id
    if plan.qk_norm and not shared and not plan.k_eq_v:
        _declare_qkv_norm_rope(
            ops, qkv, normalized, rope, split_copies, split_views, model.activation_dtype)
    return last


#: Index of the value copy inside the emitted split-copy list.
VALUE_COPY = 0

ATTENTION_BUILDERS = {
    "gqa": gqa, "mla": mla, "gdn": linear, "kda": linear,
    "mamba": linear, "gla": linear, "deltanet": linear,
    "lightning": linear, "retnet": linear,
    "projection_only": projection_only, "unmodeled": unmodeled,
}
