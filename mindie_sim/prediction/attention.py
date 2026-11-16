"""Attention operators selected by semantic kind and implementation policy.

[Role] Lower GQA directly and adapt structural MLA/DSA and recurrent programs.
[Invariants] Real request lengths determine pairs and cache traffic. Query
  padding changes projections, not valid tokens. Kernel batching and KV reuse
  are software parameters independent of availability of measurements.
[Timeouts & Locks] Bounded by requests and layers; pure arithmetic.
[Side-Effect Policy] Returns new OpSpec records; no hardware work or I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

from mindie_sim.catalog.attention_common import (
    AttentionResult, _allreduce, _causal_pairs, _long_selected_pairs,
    active_dcp_degree, dcp_partial_merge_ops, min_positive,
)
from .mla import mla
from .recurrent import linear
from .partial_attention import projection_only, unmodeled, shared_kv_input
from .cache_representation import store as representation_store
from .attention_kv_access import paged_fia_plan
from mindie_sim.catalog.cache import (
    completed_rows, index_row_bytes, index_selection_cost, local_kv_heads,
)
from mindie_sim.catalog.cache_format import cache_format
from mindie_sim.catalog.ops import gemm_op, vector_op, cache_store_op, comm_op, barrier_op
from mindie_sim.catalog.context_parallel import split_causal_pairs, owned_positions
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.tensors import TensorLayout
from .batch_prefix_attention import (
    annotate_sparse_prefix_unknown, apply_attention_prefix_reads,
)
from .batch_prefix_indexer import apply_indexer_prefix_reads
from .sparse_attention import sparse_attention_traffic
from .structure_groups import declare_group

ZERO = 0
ONE = 1
TWO = 2
THREE = 3
FOUR = 4
FIVE = 5
ZERO_FLOAT = 0.0
READ_WRITE_ONE = 1.0
NEG_ONE = -1
DEFAULT_BLOCK_SIZE = 128
DEFAULT_CP_INTERLEAVE = 128
DEFAULT_IDX_DIM = 64
FP32_BYTES = 4


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


def _qkv_views(tokens, heads, kv_heads, width, has_gate, k_eq_v=False):
    """Q and gate alternate by head; all slices retain the projected row pitch."""
    query, kv = heads * width, kv_heads * width
    gate = query if has_gate else 0
    pitch = query + gate + (1 if k_eq_v else 2) * kv
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
            offset=query + gate if k_eq_v else query + gate + kv, storage_shape=storage),
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


def _gqa_formats(model):
    """Resolve shared or independent K/V cache formats once."""
    regions = dict(getattr(model, "metadata", ())).get("cache_representations", {})
    if "gqa_kv_key" in regions or "gqa_kv_value" in regions:
        return (cache_format(model, "gqa_kv_key"),
                cache_format(model, "gqa_kv_value"), regions, False)
    return cache_format(model, "gqa_kv"), cache_format(model, "gqa_kv"), regions, True


def _attention_tensor_view(views, heads, kv_heads, width, model, quant,
                           parallel, policy, runtime, causal=True):
    """A paged ABI requires an explicit storage declaration, not a query length."""
    if policy.kv_storage != "paged":
        return _contiguous_tensor_view(
            views, heads, kv_heads, width, model, quant, parallel)
    if runtime is None or parallel.cp != 1 or parallel.pcp != 1:
        return ()
    query = (sum(view.q_len for view in views), heads, width)
    tail = (runtime.kv_block_size, kv_heads * width)
    key_fmt, value_fmt, regions, shared = _gqa_formats(model)
    native = None
    page_size = int(getattr(runtime, "kv_block_size", 0) or 0)
    if page_size > 0 and all(view.q_len > 0 for view in views):
        native = paged_fia_plan(
            requests=tuple((view.q_len, view.context + view.q_len)
                           for view in views),
            query_lengths=tuple(view.q_len for view in views),
            kv_lengths=tuple(view.context + view.q_len for view in views),
            page_size=page_size,
            query_heads=heads, kv_heads=kv_heads, head_dim=width,
            query_dtype=model.activation_dtype,
            output_dtype=model.activation_dtype,
            key_format=key_fmt, value_format=value_fmt,
            phase=getattr(views[0], "phase", "decode"),
            mask="causal" if causal else "none",
            return_lse=False, output_prepare=None,
        )
    return (("attention_tensor_view", {
        "kind": "paged_gqa", "logical_layout": "TND", "kv_storage": "paged",
        "query_shape": query, "output_shape": query,
        "key_tail_shape": tail, "value_tail_shape": tail,
        "query_dtype": model.activation_dtype, "output_dtype": model.activation_dtype,
        "key_dtype": key_fmt.dtype, "value_dtype": value_fmt.dtype,
        "batch_count": len(views), "kv_block_size": runtime.kv_block_size,
        "local_kv_heads": kv_heads, "head_dim": width,
        "cache_row_width": 2 * kv_heads * width,
        "cache_head_width": width,
        "cache_source_residency": str(
            dict(getattr(model, "metadata", ())).get(
                "cache_source_residency", "unresolved")),
        **({"cache_representations": regions} if regions else {}),
        **({"native_paged_fia": native} if native is not None else {}),
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


def _cache_tensor_view(views, kv_heads, width, key_fmt, value_fmt, shared,
                       model, parallel):
    """Declare newly written tokens without inventing cache allocation or slots."""
    if parallel.cp != 1 or parallel.pcp != 1:
        return ()
    tokens = sum(view.q_len for view in views)
    total = (key_fmt.row_bytes(2 * kv_heads * width) if shared
             else key_fmt.row_bytes(kv_heads * width)
             + value_fmt.row_bytes(kv_heads * width))
    return (("cache_tensor_view", {
        "key_shape": (tokens, kv_heads, width),
        "value_shape": (tokens, kv_heads, width),
        "key_dtype": key_fmt.dtype,
        "value_dtype": value_fmt.dtype,
        "slot_shape": (tokens,),
        "cache_row_bytes": total,
        "cache_group_size": key_fmt.group_size if shared else 0,
        "cache_scale_dtype": key_fmt.scale_dtype if shared else "mixed",
    }),)


def _decode_contract(views, heads, kv_heads, width, model, quant, parallel, plan, mode,
                     *, window=None, cp=None):
    if mode not in {"decode", "draft"}:
        return ()
    requests = tuple((view.q_len, view.context) for view in views)
    queries = {query for query, _ in requests}
    contexts = {context for _, context in requests}
    eff_win = int(plan.window or ZERO) if window is None else int(window)
    eff_cp = int(parallel.cp) if cp is None else int(cp)
    return (("gqa_decode", {
        "requests": requests,
        **({"uniform_query_tokens": next(iter(queries))} if len(queries) == ONE else {}),
        **({"uniform_context_tokens": next(iter(contexts))} if len(contexts) == ONE else {}),
        "query_heads": heads, "kv_heads": kv_heads, "head_dim": width,
        "query_dtype": model.activation_dtype, "kv_dtype": quant.kv_dtype,
        "causal": not plan.noncausal_attention, "window": eff_win,
        "cp": eff_cp,
    }),)


def _prefill_contract(views, heads, kv_heads, width, model, quant, parallel, plan, mode,
                      *, window=None, cp=None):
    """Declare per-request prefill geometry; the explicit recipe validates it."""
    if mode not in {"prefill", "draft_prefill"}:
        return ()
    eff_win = int(plan.window or ZERO) if window is None else int(window)
    eff_cp = int(parallel.cp) if cp is None else int(cp)
    return (("gqa_prefill", {
        "requests": tuple((view.q_len, view.context) for view in views),
        "query_heads": heads, "kv_heads": kv_heads, "head_dim": width,
        "query_dtype": model.activation_dtype, "kv_dtype": quant.kv_dtype,
        "causal": not plan.noncausal_attention, "window": eff_win,
        "cp": eff_cp,
    }),)


class _CacheProducer:
    """Minimal existing-builder view for T07 cache representation producers."""

    def __init__(self, ops, *, prefix, stage, layer, requests, phase, model):
        self.ops = ops
        self.prefix = prefix
        self.model = model
        self.common = {"stage": stage, "layer": layer,
                       "requests": requests, "phase": phase}

    def add(self, op):
        self.ops.append(op)


def _store_cache(ops, *, prefix, deps, rows, width, region, stage, layer,
                 requests, phase, model, views, kv_heads, head_width, parallel,
                 quant, key_fmt, value_fmt, shared_format, name="cache"):
    """Use T07 store: BF16 source and packed output are separate ledgers."""
    producer = _CacheProducer(
        ops, prefix=prefix, stage=stage, layer=layer,
        requests=requests, phase=phase, model=model)
    store_id = representation_store(
        producer, name, deps, rows, width, region)
    view = _cache_tensor_view(
        views, kv_heads, head_width, key_fmt, value_fmt, shared_format,
        model, parallel)
    if ops:
        op = ops[-1]
        metadata = dict(op.metadata)
        metadata["cache_kind"] = "gqa"
        if view:
            for key, value in view:
                metadata[key] = value
        ops[-1] = replace(op, metadata=tuple(sorted(metadata.items())))
    return store_id


def gqa(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
        padded_tokens, policy=AttentionPolicy(), cp_interleave_size=DEFAULT_CP_INTERLEAVE,
        runtime=None, cache_ids=None, stage_of=None, topk_ids=None,
        index_cache_ids=None, candidate_ids=None, **_) -> AttentionResult:
    from .implementations import select

    selected = select(runtime.kernels.get("lowerings", {}), "gqa_attention") if runtime else None
    if selected:
        selected.module.validate(views=views, plan=plan, parallel=parallel, policy=policy,
                                 quant=quant, model=model, **selected.options)
    heads = plan.num_heads // parallel.tp
    kv_heads = local_kv_heads(plan.num_kv_heads, parallel.tp)
    width = plan.head_dim
    key_fmt, value_fmt, cache_regions, shared_format = _gqa_formats(model)
    shared = 0 <= plan.kv_owner_layer_id < plan.index
    kv_planes = 1 if plan.k_eq_v else 2
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
        requests=requests, m=padded_tokens, n=q_width + kv_planes * projected_kv_heads * width,
        k=model.hidden_size, quant=quant, category="attn.qkv", phase=mode,
        family="qkv_projection", recipe="qkv_projection",
    )
    ops = [qkv]
    ready = qkv.op_id
    split_views, split_copies = {}, []
    if plan.qk_norm:
        split_views = _qkv_views(
            padded_tokens, heads, projected_kv_heads, width, plan.q_output_gate, plan.k_eq_v)
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
            for name in ("value", "gate")
            if name in split_views and (name != "value" or (not shared and not plan.k_eq_v))
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
    value_ready = (split_copies[0].op_id if split_copies and not shared and not plan.k_eq_v
                   else None)
    pcp_prefill = not shared and parallel.pcp > 1 and any(v.phase == "prefill" for v in views)
    if pcp_prefill:
        gather = comm_op(
            op_id=prefix + ".kv_gather",
            deps=(rope.op_id, value_ready) if value_ready else (rope.op_id,),
            stage=stage, layer=plan.index,
            requests=requests, family="allgather", phase=mode, recipe="pcp_kv_gather",
            link_bytes=kv_planes * kv_heads * width * dtype_bytes(quant.kv_dtype)
                       * sum(v.global_q_len or v.q_len for v in views if v.phase == "prefill")
                       * (parallel.pcp - 1) / parallel.pcp,
            shape=(valid, kv_heads, width), metadata=(("group_axis", "pcp"),))
        ops.append(gather)
        store_ready = gather.op_id
    # The gather consumes rotary output, so rotary must precede cache writes
    # in the emitted graph order; it is also a dependency of the gather.
    ops.insert(ops.index(gather) if pcp_prefill else len(ops), rope)
    max_dcp = max((active_dcp_degree(parallel, v.phase) for v in views), default=ONE)
    store_rows = max(
        sum(
            (v.global_q_len or v.q_len)
            if active_dcp_degree(parallel, v.phase) <= ONE else
            owned_positions(v.context,
                            v.context + (v.global_q_len or v.q_len) - ONE,
                            max_dcp, cp_interleave_size, rank)
            for v in views)
        for rank in range(max_dcp))
    store_deps = ((store_ready, value_ready)
                  if value_ready and not pcp_prefill else (store_ready,))
    if shared:
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
            width=2 * kv_heads * width, region="gqa_kv",
            stage=stage, layer=plan.index, requests=requests, phase=mode,
            model=model, views=views, kv_heads=kv_heads, head_width=width,
            parallel=parallel, quant=quant, key_fmt=key_fmt,
            value_fmt=value_fmt, shared_format=True),)
    else:
        key_store = _store_cache(
            ops, prefix=prefix, deps=store_deps, rows=store_rows,
            width=kv_heads * width, region="gqa_kv_key", name="cache_key",
            stage=stage, layer=plan.index, requests=requests, phase=mode,
            model=model, views=views, kv_heads=kv_heads, head_width=width,
            parallel=parallel, quant=quant, key_fmt=key_fmt,
            value_fmt=value_fmt, shared_format=False)
        value_store = _store_cache(
            ops, prefix=prefix, deps=store_deps, rows=store_rows,
            width=kv_heads * width, region="gqa_kv_value", name="cache_value",
            stage=stage, layer=plan.index, requests=requests, phase=mode,
            model=model, views=views, kv_heads=kv_heads, head_width=width,
            parallel=parallel, quant=quant, key_fmt=key_fmt,
            value_fmt=value_fmt, shared_format=False)
        store_ids = (key_store, value_store)
    plans = model.layer_plans
    idx_raw, sel_raw = int(plan.index_k_owner_layer_id), int(plan.topk_owner_layer_id)
    if plan.indexer:
        idx_owner = sel_owner = plan.index
    else:
        idx_owner = idx_raw if 0 <= idx_raw < plan.index else -1
        sel_owner = (sel_raw if 0 <= sel_raw < min(plan.index, len(plans))
                     and (plans[sel_raw].indexer or plans[sel_raw].owns_topk) else idx_owner)
    has_indexer = bool(plan.indexer or plan.owns_topk or idx_owner >= 0 or sel_owner >= 0)
    eff_topk = int(plan.index_topk) if has_indexer else 0
    index_store_id = candidate_id = None
    topk_id = (topk_ids or {}).get(sel_owner) if sel_owner >= 0 else None
    block_size = max(
        ONE, int(getattr(runtime, "kv_block_size", DEFAULT_BLOCK_SIZE) or DEFAULT_BLOCK_SIZE))
    configured_families = runtime.kernels.get("families", {}) if runtime else {}
    user_pinned_idx = (
        "indexer" in configured_families or "index_scores" in configured_recipes)
    if eff_topk > ZERO and (plan.indexer or plan.owns_index_k or plan.owns_topk):
        cost = index_selection_cost(
            model, plan, max((v.kv_len for v in views), default=ZERO))
        row_bytes = cost.index_row_bytes or index_row_bytes(model, plan.index_dim)
        idx_heads = max(ONE, int(plan.index_heads or ONE))
        idx_dim = max(ONE, int(plan.index_dim or DEFAULT_IDX_DIM))
        idx_ratio = max(ONE, int(plan.compress_ratio or ONE), int(plan.index_kpool or ONE))
        cand_cap = (plan.candidate_topk_blocks * plan.candidate_block_size
                    if plan.candidate_filter else ZERO)
        new_idx_rows = (
            sum(completed_rows(v.context + e_p, idx_ratio)
                - completed_rows(v.context + s_p, idx_ratio)
                for v in views
                for s_p, e_p in (v.query_ranges or ((ZERO, v.q_len),))
                if e_p > s_p)
            if idx_ratio > ONE else valid
        )
        rows_by_view: list[int] = []
        shared_idx_by_view: list[int] = []
        for v in views:
            v_ranges = v.query_ranges or ((ZERO, v.q_len),)
            v_scored = int(sum(
                _long_selected_pairs(
                    e_p - s_p, v.context + s_p, idx_ratio, cand_cap,
                    noncausal=bool(plan.noncausal_attention), kv_len=v.kv_len)
                for s_p, e_p in v_ranges if e_p > s_p
            ))
            raw_pref = max(ZERO, min(int(getattr(v, "prefix_tokens", ZERO) or ZERO), int(v.context)))
            aligned_pref = ((raw_pref // block_size) * block_size) // idx_ratio
            q_cnt = sum(e_p - s_p for s_p, e_p in v_ranges if e_p > s_p)
            rows_by_view.append(v_scored)
            shared_idx_by_view.append(min(v_scored, aligned_pref * max(ONE, q_cnt)))
        scored_rows = sum(rows_by_view)
        idx_wq = gemm_op(
            op_id=prefix + ".index_wq", deps=(deps,), stage=stage, layer=plan.index,
            requests=requests, m=padded_tokens, n=idx_heads * idx_dim, k=model.hidden_size,
            quant=quant, category="indexer", phase=mode, recipe="index_wq",
            family="index_projection")
        idx_wk = gemm_op(
            op_id=prefix + ".index_wk", deps=(deps,), stage=stage, layer=plan.index,
            requests=requests, m=padded_tokens, n=idx_dim, k=model.hidden_size,
            quant=quant, category="indexer", phase=mode, recipe="index_wk",
            family="index_projection")
        idx_store = cache_store_op(
            op_id=prefix + ".index_store", deps=(idx_wk.op_id,), stage=stage,
            layer=plan.index, requests=requests, phase=mode, recipe="index_store",
            rows=new_idx_rows, row_bytes=max(ONE, int(round(row_bytes))),
            dtype=quant.index_dtype, kind="index")
        q_idx_bytes = valid * idx_heads * idx_dim * dtype_bytes(quant.index_dtype)
        shape = (scored_rows, idx_heads, idx_dim)
        idx_scores = OpSpec(
            op_id=prefix + ".index_scores", family="indexer", phase=mode, stage=stage,
            layer=plan.index, requests=requests, semantic_shape=shape, local_shape=shape,
            padded_shape=shape, launch_shape=shape,
            flops=float(TWO * scored_rows * idx_heads * idx_dim),
            vector_ops=float(THREE * scored_rows * idx_heads),
            hbm_bytes=float(scored_rows * (row_bytes + FP32_BYTES) + q_idx_bytes),
            mte2_bytes=float(scored_rows * row_bytes + q_idx_bytes),
            mte3_bytes=float(scored_rows * FP32_BYTES),
            deps=(idx_wq.op_id, idx_store.op_id), recipe="index_scores",
            dtype=quant.index_dtype)
        idx_scores = apply_indexer_prefix_reads(
            idx_scores, views=views, rows_by_view=rows_by_view,
            shared_rows_by_view=shared_idx_by_view, row_bytes=float(row_bytes),
            layer_index=plan.index, runtime=runtime,
            kernel_reuse="native_per_sequence" if user_pinned_idx else "tile",
            allow_recipe_rebind=not user_pinned_idx)
        idx_topk = vector_op(
            op_id=prefix + ".index_topk", deps=(idx_scores.op_id,), stage=stage,
            layer=plan.index, requests=requests, elements=scored_rows,
            ops_per_element=FOUR, dtype="fp32", family="topk", phase=mode,
            recipe="index_topk", shape=(valid, eff_topk), read_write=READ_WRITE_ONE,
            metadata=(("selection_output_dtype", "int32"),))
        ops.extend((idx_wq, idx_wk, idx_store, idx_scores, idx_topk))
        index_store_id, topk_id = idx_store.op_id, idx_topk.op_id
    elif idx_owner >= ZERO:
        index_store_id = (index_cache_ids or {}).get(idx_owner)
    eff_window = min_positive(plan.window, eff_topk)
    dense_win = int(plan.window or ZERO)
    attention_deps = (*store_ids, topk_id) if topk_id else store_ids
    streams = kv_heads if policy.reuse_kv_across_query_heads else heads
    kv_row_bytes = (
        key_fmt.row_bytes(TWO * streams * width) if shared_format
        else key_fmt.row_bytes(streams * width) + value_fmt.row_bytes(streams * width)
    )
    user_pinned_attn = (
        "attention" in configured_families
        or attention_recipe in configured_recipes
        or "flash_attention" in configured_recipes)
    phase_groups: list[tuple[int, list[tuple[int, object]]]] = []
    group_map: dict[int, list[tuple[int, object]]] = {}
    for index, view in enumerate(views):
        eff_cp = active_dcp_degree(parallel, view.phase)
        if eff_cp not in group_map:
            bucket: list[tuple[int, object]] = []
            group_map[eff_cp] = bucket
            phase_groups.append((eff_cp, bucket))
        group_map[eff_cp].append((index, view))
    attention: list[str] = []
    multi_group = len(phase_groups) > ONE
    for grp_idx, (eff_cp, indexed_views) in enumerate(phase_groups):
        grp_views = tuple(v for _, v in indexed_views)
        grp_reqs = tuple(v.request_id for v in grp_views)
        grp_valid = sum(v.q_len for v in grp_views)
        grp_prefix = f"{prefix}.phase{grp_idx}" if multi_group else prefix
        rank_pairs = [ZERO_FLOAT] * eff_cp
        rank_sweeps = [ZERO] * eff_cp
        rank_rows = [ZERO] * eff_cp
        per_view_pairs: list[tuple[float, list[float]]] = []
        per_view_sweeps: list[list[int]] = []
        per_view_shared: list[list[int]] = []
        per_view_rows: list[list[int]] = []
        for _, view in indexed_views:
            q, context = view.q_len, view.context
            ranges = view.query_ranges or ((ZERO, q),)
            selected_q = sum(end - start for start, end in ranges if end > start)
            if eff_topk > ZERO:
                pairs = sum(
                    _long_selected_pairs(
                        end - start, context + start, ONE, eff_window,
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
                v_shared = [ZERO] * eff_cp
            else:
                visible_kv = min(view.kv_len, dense_win) if dense_win > ZERO else view.kv_len
                pairs = (
                    float(selected_q * visible_kv) if plan.noncausal_attention else
                    sum(_causal_pairs(end - start, context + start, dense_win)
                        for start, end in ranges if end > start)
                )
                if eff_cp == ONE:
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
                v_sweeps = [ZERO] * eff_cp
                v_shared = [ZERO] * eff_cp
                raw_prefix = max(ZERO, min(int(getattr(view, "prefix_tokens", ZERO) or ZERO), int(context)))
                aligned_prefix = (raw_prefix // block_size) * block_size
                for begin, end in ranges:
                    for start in range(begin, end, policy.query_tile):
                        if plan.noncausal_attention:
                            hi = view.kv_len
                            lo = max(ZERO, hi - dense_win) if dense_win > ZERO else ZERO
                        else:
                            hi = context + min(end, start + policy.query_tile)
                            lo = max(ZERO, context + start + ONE - dense_win) if dense_win > ZERO else ZERO
                        shared_hi = min(hi, aligned_prefix)
                        for rank in range(eff_cp):
                            v_sweeps[rank] += owned_positions(
                                lo, hi - ONE, eff_cp, cp_interleave_size, rank)
                            if shared_hi > lo:
                                v_shared[rank] += owned_positions(
                                    lo, shared_hi - ONE, eff_cp, cp_interleave_size, rank)
            v_rows = (
                [owned_positions(ZERO, view.kv_len - ONE, eff_cp, cp_interleave_size, r)
                 for r in range(eff_cp)]
                if eff_cp > ONE else [view.kv_len]
            )
            for r in range(eff_cp):
                rank_pairs[r] += v_pairs[r]
                rank_sweeps[r] += v_sweeps[r]
                rank_rows[r] += v_rows[r]
            per_view_pairs.append((float(pairs), v_pairs))
            per_view_sweeps.append(v_sweeps)
            per_view_shared.append(v_shared)
            per_view_rows.append(v_rows)
        worst_r = max(range(eff_cp), key=lambda r: (rank_pairs[r], rank_sweeps[r]))
        batches: list[OpSpec] = []
        sweeps_by_view: list[int] = []
        shared_sweeps_by_view: list[int] = []
        total_pairs_sum = ZERO_FLOAT
        total_rows_sum = ZERO
        for v_pos, (orig_idx, view) in enumerate(indexed_views):
            pairs, v_pairs = per_view_pairs[v_pos]
            local_pairs = v_pairs[worst_r]
            view_sweeps = per_view_sweeps[v_pos][worst_r]
            local_rows = per_view_rows[v_pos][worst_r]
            sweeps_by_view.append(view_sweeps)
            shared_sweeps_by_view.append(min(view_sweeps, per_view_shared[v_pos][worst_r]))
            total_pairs_sum += pairs
            total_rows_sum += view.kv_len
            kv_bytes = view_sweeps * kv_row_bytes
            input_bytes = view.q_len * heads * width * dtype_bytes(model.activation_dtype)
            shape = (view.q_len, view.kv_len, heads, width)
            op = OpSpec(
                op_id=f"{prefix}.attention.{orig_idx}", family="attention", phase=mode,
                stage=stage, layer=plan.index, requests=(view.request_id,),
                semantic_shape=shape, local_shape=(view.q_len, local_rows, heads, width),
                padded_shape=shape, launch_shape=shape, flops=FOUR * local_pairs * heads * width,
                vector_ops=FIVE * local_pairs * heads, hbm_bytes=kv_bytes + TWO * input_bytes,
                mte2_bytes=kv_bytes + input_bytes, mte3_bytes=input_bytes,
                deps=attention_deps, recipe=attention_recipe, dtype=model.compute_dtype,
                metadata=(("query_tile", policy.query_tile), ("causal_pairs", pairs),
                          ("local_pairs", local_pairs), ("kv_streams", streams))
                         + _decode_contract(
                             (view,), heads, kv_heads, width, model, quant, parallel, plan, mode,
                             window=eff_window, cp=eff_cp)
                         + _prefill_contract(
                             (view,), heads, kv_heads, width, model, quant, parallel, plan, mode,
                             window=eff_window, cp=eff_cp)
                         + _attention_tensor_view(
                             (view,), heads, kv_heads, width, model, quant, parallel,
                             policy, runtime, causal=not plan.noncausal_attention),
            )
            if selected:
                op = selected.module.lower_op(
                    op, view=view, plan=plan, key_heads=kv_heads, model=model,
                    runtime=runtime, **selected.options)
            batches.append(op)
        grp_attn: list[str] = []
        if policy.fused_batch and not selected and batches:
            fields = ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes")
            batched = replace(
                batches[ZERO], op_id=grp_prefix + ".attention", requests=grp_reqs,
                semantic_shape=(grp_valid, max(v.kv_len for v in grp_views), heads, width),
                local_shape=(grp_valid, rank_rows[worst_r], heads, width),
                metadata=(("request_shapes", tuple((v.q_len, v.context) for v in grp_views)),
                          ("query_tile", policy.query_tile),
                          ("kv_streams", streams))
                         + _decode_contract(
                             grp_views, heads, kv_heads, width, model, quant,
                             parallel, plan, mode, window=eff_window, cp=eff_cp)
                         + _prefill_contract(
                             grp_views, heads, kv_heads, width, model, quant,
                             parallel, plan, mode, window=eff_window, cp=eff_cp)
                         + _attention_tensor_view(
                             grp_views, heads, kv_heads, width, model, quant, parallel,
                             policy, runtime, causal=not plan.noncausal_attention),
                **{key: sum(getattr(op, key) for op in batches) for key in fields})
            if eff_topk > ZERO:
                batched = annotate_sparse_prefix_unknown(batched, runtime)
            else:
                batched = apply_attention_prefix_reads(
                    batched, views=grp_views, sweeps_by_view=sweeps_by_view,
                    shared_sweeps_by_view=shared_sweeps_by_view, row_bytes=float(kv_row_bytes),
                    layer_index=plan.index, runtime=runtime,
                    kernel_reuse="native_per_sequence" if user_pinned_attn else "tile",
                    allow_recipe_rebind=not user_pinned_attn,
                )
            ops.append(batched)
            grp_attn = [batched.op_id]
        elif batches:
            if eff_topk > ZERO:
                batches = [annotate_sparse_prefix_unknown(item, runtime) for item in batches]
            else:
                batches[ZERO] = apply_attention_prefix_reads(
                    batches[ZERO], views=grp_views, sweeps_by_view=sweeps_by_view,
                    shared_sweeps_by_view=shared_sweeps_by_view, row_bytes=float(kv_row_bytes),
                    layer_index=plan.index, runtime=runtime,
                    kernel_reuse="native_per_sequence", allow_recipe_rebind=False,
                )
            ops.extend(batches)
            grp_attn = [item.op_id for item in batches]
        if eff_cp > ONE and grp_attn:
            merged, merged_last = dcp_partial_merge_ops(
                prefix=grp_prefix, views=grp_views, cp=eff_cp,
                local_heads=heads, v_dim=width, stage=stage, layer=plan.index, mode=mode,
                deps=tuple(grp_attn), total_pairs=total_pairs_sum, local_pairs=rank_pairs[worst_r],
                total_rows=total_rows_sum, local_rows=rank_rows[worst_r],
                interleave=cp_interleave_size,
            )
            ops.extend(merged)
            grp_attn = [merged_last]
        attention.extend(grp_attn)
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
    if plan.qk_norm and not shared and not plan.k_eq_v:
        _declare_qkv_norm_rope(
            ops, qkv, normalized, rope, split_copies, split_views, model.activation_dtype)
    return AttentionResult(ops=ops, last=last, cache_store_id=store_ids[0],
                           index_store_id=index_store_id, topk_id=topk_id,
                           candidate_id=candidate_id)


ATTENTION_BUILDERS = {
    "gqa": gqa, "mla": mla, "gdn": linear, "kda": linear,
    "mamba": linear, "gla": linear, "deltanet": linear,
    "lightning": linear, "retnet": linear,
    "projection_only": projection_only, "unmodeled": unmodeled,
}
