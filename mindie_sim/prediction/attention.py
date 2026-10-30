"""Attention operators selected by semantic kind and implementation policy.

[Role] Lower GQA directly and adapt structural MLA/DSA and recurrent programs.
[Invariants] Real request lengths determine pairs and cache traffic. Query
  padding changes projections, not valid tokens. Kernel batching and KV reuse
  are software parameters independent of availability of measurements.
[Timeouts & Locks] Bounded by requests and layers; pure arithmetic.
[Side-Effect Policy] Returns new OpSpec records; no hardware work or I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from mindie_sim.catalog.attention_common import AttentionResult, _allreduce, _causal_pairs, dcp_partial_merge_ops
from .mla import mla
from .recurrent import linear
from .partial_attention import projection_only, unmodeled, shared_kv_input
from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.ops import gemm_op, vector_op, cache_store_op, comm_op, barrier_op
from mindie_sim.catalog.context_parallel import split_causal_pairs, owned_positions
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.tensors import TensorLayout
from .structure_groups import declare_group


_QKV_NORM_ROPE_SOURCE = {
    "repository": "vllm-ascend",
    "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "path": "vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py",
    "sha256": "ba3c7e740c3a2a9c8850733d3be77fefac79bc3ed02a45b4be076b58d6f7632d",
    "symbol": "triton_split_qkv_rmsnorm_mrope",
    "lines": "29-57,83-110,262-278,282-366",
    "source_state": "clean_at_revision",
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


def _qkv_views(tokens, heads, kv_heads, width, has_gate):
    """Q and gate alternate by head; all slices retain the projected row pitch."""
    query, kv = heads * width, kv_heads * width
    gate = query if has_gate else 0
    pitch = query + gate + 2 * kv
    storage = (tokens, pitch)
    views = {
        "query": TensorLayout(
            (tokens, heads, width), (pitch, width * (2 if has_gate else 1), 1),
            storage_shape=storage),
        "key": TensorLayout(
            (tokens, kv_heads, width), (pitch, width, 1),
            offset=query + gate, storage_shape=storage),
        "value": TensorLayout(
            (tokens, kv_heads, width), (pitch, width, 1),
            offset=query + gate + kv, storage_shape=storage),
    }
    if has_gate:
        views["gate"] = TensorLayout(
            (tokens, heads, width), (pitch, 2 * width, 1),
            offset=width, storage_shape=storage)
    return views


def _qkv_output_copy(qkv, layout, output, dtype):
    """The native wrapper allocates V/gate outputs and writes the selected values."""
    return vector_op(
        op_id=qkv.op_id + "." + output, deps=(qkv.op_id,),
        stage=qkv.stage, layer=qkv.layer, requests=qkv.requests,
        elements=layout.elements, ops_per_element=0, dtype=dtype,
        family="tensor_copy", phase=qkv.phase, recipe="qkv_split_materialize",
        shape=layout.shape,
        metadata=(("source_layout", layout.to_dict()),
                  ("target_layout", TensorLayout.contiguous(layout.shape).to_dict()),
                  ("source_output", output),
                  ("materialization_reason", "split_qkv_native_output"),
                  ("native_source", _QKV_NORM_ROPE_SOURCE)),
    )


def _declare_qkv_norm_rope(ops, qkv, normalized, rope, copies, views, dtype):
    """The projected tensor is an input; its GEMM and later attention stay outside."""
    members = (normalized.op_id, rope.op_id, *(op.op_id for op in copies))
    declare_group(
        ops, key=qkv.op_id + ".norm_rope", role="qkv_norm_rope",
        family="qkv_preprocess", members=members,
        entries=(normalized.op_id, *(op.op_id for op in copies)), anchor=rope.op_id,
        view={"input_shape": views["query"].storage_shape, "dtype": dtype,
              "input_views": {name: layout.to_dict() for name, layout in views.items()},
              "output_shapes": {name: (layout.shape[0], math.prod(layout.shape[1:]))
                                for name, layout in views.items()},
              "required_output_slots": 4},
        implementation={
            "native_kind": "split_qkv_rmsnorm_mrope_kernel",
            "selection": None,
            "conditions": {"qk_norm": True, "gate_present": "gate" in views,
                           "boundary_input": "projected_qkv",
                           "boundary_views": {
                               name: layout.to_dict() for name, layout in views.items()}},
        },
        source=_QKV_NORM_ROPE_SOURCE,
        coverage={
            "kind": "complete_subgraph",
            "required_output_slots": 4,
            "limitations": [
                "retained_capture_exposes_only_one_of_q_k_v_gate_output_descriptors",
                "mrope_sections_partial_rotary_width_epsilon_bias_and_weight_shift_not_declared",
                "semantic_work_is_not_a_selected_native_dispatch_or_measured_duration",
            ],
        },
    )


@dataclass(frozen=True)
class AttentionPolicy:
    query_tile: int = 128
    reuse_kv_across_query_heads: bool = True
    fused_batch: bool = True
    recurrent_chunk: int = 64
    kv_storage: str = "auto"

    def __post_init__(self):
        if self.kv_storage not in {"auto", "paged"}:
            raise ModelingError("attention_kv_storage_requires_auto_or_paged")


def _attention_tensor_view(views, heads, kv_heads, width, model, quant,
                           parallel, policy, runtime):
    """A paged ABI requires an explicit storage declaration, not a query length."""
    if policy.kv_storage != "paged":
        return _contiguous_tensor_view(
            views, heads, kv_heads, width, model, quant, parallel)
    if runtime is None or parallel.cp != 1 or parallel.pcp != 1:
        return ()
    query = (sum(view.q_len for view in views), heads, width)
    tail = (runtime.kv_block_size, kv_heads * width)
    return (("attention_tensor_view", {
        "kind": "paged_gqa", "logical_layout": "TND", "kv_storage": "paged",
        "query_shape": query, "output_shape": query,
        "key_tail_shape": tail, "value_tail_shape": tail,
        "query_dtype": model.activation_dtype, "output_dtype": model.activation_dtype,
        "key_dtype": quant.kv_dtype, "value_dtype": quant.kv_dtype,
        "batch_count": len(views), "kv_block_size": runtime.kv_block_size,
        "local_kv_heads": kv_heads, "head_dim": width,
    }),)


def _contiguous_tensor_view(views, heads, kv_heads, width, model, quant, parallel):
    """Fresh prefill has explicit TND operands, before any paged-cache reuse."""
    if parallel.cp != 1 or parallel.pcp != 1 or any(
        view.phase != "prefill" or view.context for view in views
    ):
        return ()
    query = (sum(view.q_len for view in views), heads, width)
    kv = (sum(view.kv_len for view in views), kv_heads, width)
    return (("attention_tensor_view", {
        "query_shape": query, "key_shape": kv, "value_shape": kv,
        "output_shape": query, "query_dtype": model.activation_dtype,
        "key_dtype": quant.kv_dtype, "value_dtype": quant.kv_dtype,
        "output_dtype": model.activation_dtype,
        "logical_layout": "TND", "kv_storage": "contiguous",
    }),)


def _cache_tensor_view(views, kv_heads, width, model, quant, parallel):
    """Declare newly written tokens without inventing cache allocation or slots."""
    if (parallel.cp != 1 or parallel.pcp != 1
            or model.activation_dtype != quant.kv_dtype):
        return ()
    tokens = sum(view.q_len for view in views)
    return (("cache_tensor_view", {
        "key_shape": (tokens, kv_heads, width),
        "value_shape": (tokens, kv_heads, width),
        "key_dtype": model.activation_dtype,
        "value_dtype": model.activation_dtype,
        "slot_shape": (tokens,),
    }),)


def _decode_contract(views, heads, kv_heads, width, model, quant, parallel, plan, mode):
    if mode not in {"decode", "draft"}:
        return ()
    return (("gqa_decode", {
        "requests": tuple((view.q_len, view.context) for view in views),
        "query_heads": heads, "kv_heads": kv_heads, "head_dim": width,
        "query_dtype": model.activation_dtype, "kv_dtype": quant.kv_dtype,
        "causal": not plan.noncausal_attention, "window": plan.window or 0,
        "cp": parallel.cp,
    }),)


def _prefill_contract(views, heads, kv_heads, width, model, quant, parallel, plan, mode):
    """Declare per-request prefill geometry; the explicit recipe validates it."""
    if mode not in {"prefill", "draft_prefill"}:
        return ()
    return (("gqa_prefill", {
        "requests": tuple((view.q_len, view.context) for view in views),
        "query_heads": heads, "kv_heads": kv_heads, "head_dim": width,
        "query_dtype": model.activation_dtype, "kv_dtype": quant.kv_dtype,
        "causal": not plan.noncausal_attention, "window": plan.window or 0,
        "cp": parallel.cp,
    }),)


def gqa(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
        padded_tokens, policy=AttentionPolicy(), cp_interleave_size=128,
        runtime=None, cache_ids=None, stage_of=None, **_) -> AttentionResult:
    from .implementations import select

    selected = select(runtime.kernels.get("lowerings", {}), "gqa_attention") if runtime else None
    if selected:
        selected.module.validate(views=views, plan=plan, parallel=parallel, policy=policy,
                                 quant=quant, model=model, **selected.options)
    heads = plan.num_heads // parallel.tp
    kv_heads = local_kv_heads(plan.num_kv_heads, parallel.tp)
    width = plan.head_dim
    shared = 0 <= plan.kv_owner_layer_id < plan.index
    projected_kv_heads = 0 if shared else kv_heads
    configured_recipes = runtime.kernels.get("recipes", {}) if runtime else {}
    if mode in {"decode", "draft"} and "flash_attention_decode" in configured_recipes:
        attention_recipe = "flash_attention_decode"
    elif (mode in {"prefill", "draft_prefill"}
          and "flash_attention_prefill" in configured_recipes):
        attention_recipe = "flash_attention_prefill"
    else:
        attention_recipe = "flash_attention"
    requests = tuple(view.request_id for view in views)
    q_width = heads * width * (2 if plan.q_output_gate else 1)
    qkv = gemm_op(
        op_id=prefix + ".qkv", deps=(deps,), stage=stage, layer=plan.index,
        requests=requests, m=padded_tokens, n=q_width + 2 * projected_kv_heads * width,
        k=model.hidden_size, quant=quant, category="attn.qkv", phase=mode,
        family="qkv_projection", recipe="qkv_projection",
    )
    ops = [qkv]
    ready = qkv.op_id
    split_views, split_copies = {}, []
    if plan.qk_norm:
        split_views = _qkv_views(
            padded_tokens, heads, projected_kv_heads, width, plan.q_output_gate)
        normalized = vector_op(
            op_id=prefix + ".qk_norm", deps=(ready,), stage=stage, layer=plan.index,
            requests=requests, elements=padded_tokens * (heads + projected_kv_heads) * width,
            ops_per_element=5, dtype=model.activation_dtype, family="rmsnorm",
            phase=mode, recipe="qk_norm", shape=(padded_tokens, heads + projected_kv_heads, width),
            metadata=(("input_views", {
                name: split_views[name].to_dict()
                for name in (("query",) if shared else ("query", "key"))}),))
        ops.append(normalized)
        ready = normalized.op_id
        split_copies = [
            _qkv_output_copy(qkv, split_views[name], name, model.activation_dtype)
            for name in ("value", "gate") if name in split_views and (name != "value" or not shared)
        ]
        ops.extend(split_copies)
    rope = vector_op(
        op_id=prefix + ".rope", deps=(ready,), stage=stage, layer=plan.index,
        requests=requests, elements=padded_tokens * (heads + projected_kv_heads) * width,
        ops_per_element=6, dtype=model.activation_dtype, family="rotary", phase=mode,
        recipe="rotary", shape=(padded_tokens, heads + projected_kv_heads, width),
    )
    valid = sum(view.q_len for view in views)
    store_ready = rope.op_id
    value_ready = split_copies[0].op_id if split_copies and not shared else None
    pcp_prefill = not shared and parallel.pcp > 1 and any(v.phase == "prefill" for v in views)
    if pcp_prefill:
        gather = comm_op(
            op_id=prefix + ".kv_gather",
            deps=(rope.op_id, value_ready) if value_ready else (rope.op_id,),
            stage=stage, layer=plan.index,
            requests=requests, family="allgather", phase=mode, recipe="pcp_kv_gather",
            link_bytes=2 * kv_heads * width * dtype_bytes(quant.kv_dtype)
                       * sum(v.global_q_len or v.q_len for v in views if v.phase == "prefill")
                       * (parallel.pcp - 1) / parallel.pcp,
            shape=(valid, kv_heads, width), metadata=(("group_axis", "pcp"),))
        ops.append(gather)
        store_ready = gather.op_id
    store = cache_store_op(
        op_id=prefix + ".cache",
        deps=((store_ready, value_ready) if value_ready and not pcp_prefill
              else (store_ready,)),
        stage=stage, layer=plan.index,
        requests=requests, phase=mode, recipe="kv_store",
        rows=max(sum(owned_positions(v.context, v.context + (v.global_q_len or v.q_len) - 1,
                                     parallel.cp, cp_interleave_size, rank) for v in views)
                 for rank in range(parallel.cp)),
        row_bytes=2 * kv_heads * width * dtype_bytes(quant.kv_dtype),
        dtype=quant.kv_dtype, kind="gqa",
        metadata=_cache_tensor_view(views, kv_heads, width, model, quant, parallel),
    )
    if shared:
        extra, owner_ready = shared_kv_input(
            prefix=prefix, plan=plan, stage=stage, views=views,
            cache_ids=cache_ids or {}, stage_of=stage_of,
            parallel=parallel, quant=quant, mode=mode)
        ops.extend(extra)
        store = barrier_op(
            op_id=prefix + ".shared_cache_ready", deps=(rope.op_id, owner_ready),
            stage=stage, layer=plan.index, requests=requests, phase=mode,
            recipe="shared_cache_dependency", shape=())
    # The gather consumes rotary output, so rotary must precede it in DAG order.
    ops.insert(ops.index(gather) if pcp_prefill else len(ops), rope)
    ops.append(store)
    attention, batches = [], []
    for index, view in enumerate(views):
        q, context = view.q_len, view.context
        ranges = view.query_ranges or ((0, q),)
        pairs = (q * view.kv_len if plan.noncausal_attention else
                 sum(_causal_pairs(end - start, context + start, plan.window) for start, end in ranges if end > start))
        local_pairs, local_rows = pairs, context + q
        if parallel.cp > 1:
            split = split_causal_pairs(q, context, parallel.cp, cp_interleave_size,
                                       window=plan.window, noncausal=plan.noncausal_attention)
            local_pairs = split.max_rank
            local_rows = max(owned_positions(0, context + q - 1, parallel.cp, cp_interleave_size, rank)
                             for rank in range(parallel.cp))
        # One K/V sweep per query tile, cropped to each tile's causal horizon.
        # The final tile is real work, not a full padded sequence.
        sweeps = [0] * parallel.cp
        for begin, end in ranges:
            for start in range(begin, end, policy.query_tile):
                hi = view.kv_len if plan.noncausal_attention else context + min(end, start + policy.query_tile)
                lo = max(0, context + start + 1 - plan.window) if plan.window else 0
                for rank in range(parallel.cp):
                    sweeps[rank] += owned_positions(lo, hi - 1, parallel.cp, cp_interleave_size, rank)
        streams = kv_heads if policy.reuse_kv_across_query_heads else heads
        kv_bytes = max(sweeps) * 2 * streams * width * dtype_bytes(quant.kv_dtype)
        input_bytes = q * heads * width * dtype_bytes(model.activation_dtype)
        shape = (q, context + q, heads, width)
        op = OpSpec(
            op_id=f"{prefix}.attention.{index}", family="attention", phase=mode,
            stage=stage, layer=plan.index, requests=(view.request_id,),
            semantic_shape=shape, local_shape=(q, local_rows, heads, width),
            padded_shape=shape, launch_shape=shape, flops=4 * local_pairs * heads * width,
            vector_ops=5 * local_pairs * heads, hbm_bytes=kv_bytes + 2 * input_bytes,
            mte2_bytes=kv_bytes + input_bytes, mte3_bytes=input_bytes,
            deps=(store.op_id,), recipe=attention_recipe, dtype=model.compute_dtype,
            metadata=(("query_tile", policy.query_tile), ("causal_pairs", pairs),
                      ("local_pairs", local_pairs), ("kv_streams", streams))
                     + _decode_contract(
                         (view,), heads, kv_heads, width, model, quant, parallel, plan, mode)
                     + _prefill_contract(
                         (view,), heads, kv_heads, width, model, quant, parallel, plan, mode)
                     + _attention_tensor_view(
                         (view,), heads, kv_heads, width, model, quant, parallel,
                         policy, runtime),
        )
        if selected:
            op = selected.module.lower_op(
                op, view=view, plan=plan, key_heads=kv_heads, model=model,
                runtime=runtime, **selected.options)
        batches.append(op)
        last = op.op_id
        if parallel.cp > 1:
            ops.append(op)
            merged, last = dcp_partial_merge_ops(
                prefix=f"{prefix}.request{index}", view=view, cp=parallel.cp,
                local_heads=heads, v_dim=width, stage=stage, layer=plan.index, mode=mode,
                deps=(last,), total_pairs=pairs, local_pairs=local_pairs,
                total_rows=context + q, local_rows=local_rows, interleave=cp_interleave_size,
            )
            ops.extend(merged)
        attention.append(last)
    if parallel.cp == 1 and policy.fused_batch and not selected:
        from dataclasses import replace
        fields = ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes")
        batched = replace(batches[0], op_id=prefix + ".attention", requests=requests,
                          semantic_shape=(valid, max(v.kv_len for v in views), heads, width),
                          local_shape=(valid, sum(v.kv_len for v in views), heads, width),
                          metadata=(("request_shapes", tuple((v.q_len, v.context) for v in views)),
                                    ("query_tile", policy.query_tile),
                                    ("kv_streams", kv_heads if policy.reuse_kv_across_query_heads else heads))
                                   + _decode_contract(
                                       views, heads, kv_heads, width, model, quant,
                                       parallel, plan, mode)
                                   + _prefill_contract(
                                       views, heads, kv_heads, width, model, quant,
                                       parallel, plan, mode)
                                   + _attention_tensor_view(
                                       views, heads, kv_heads, width, model, quant, parallel,
                                       policy, runtime),
                          **{key: sum(getattr(op, key) for op in batches) for key in fields})
        ops.append(batched)
        attention = [batched.op_id]
    elif parallel.cp == 1:
        ops.extend(batches)
    if plan.q_output_gate:
        gate_ready = split_copies[-1].op_id if split_copies else qkv.op_id
        gate = vector_op(
            op_id=prefix + ".output_gate", deps=tuple(attention) + (gate_ready,),
            stage=stage, layer=plan.index, requests=requests,
            elements=padded_tokens * heads * width, ops_per_element=5,
            dtype=model.activation_dtype, family="elementwise", phase=mode, recipe="attention_gate",
        )
        ops.append(gate)
        attention = [gate.op_id]
    output = gemm_op(
        op_id=prefix + ".output", deps=tuple(attention), stage=stage, layer=plan.index,
        requests=requests, m=padded_tokens, n=model.hidden_size, k=heads * width,
        quant=quant, category="attn.o", phase=mode, family="output_projection", recipe="attention_output",
    )
    ops.append(output)
    last = output.op_id
    if parallel.tp > 1:
        reduction = _allreduce(prefix=prefix, deps=(last,), stage=stage, layer=plan.index,
                               requests=requests, model=model, parallel=parallel,
                               mode=mode, tokens=padded_tokens)
        ops.append(reduction)
        last = reduction.op_id
    if plan.qk_norm and not shared:
        _declare_qkv_norm_rope(
            ops, qkv, normalized, rope, split_copies, split_views, model.activation_dtype)
    return AttentionResult(ops=ops, last=last, cache_store_id=store.op_id)


ATTENTION_BUILDERS = {"gqa": gqa, "mla": mla, "gdn": linear, "kda": linear,
                      "projection_only": projection_only, "unmodeled": unmodeled}
