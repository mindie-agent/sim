"""Latent, compressed and indexed attention from explicit layer geometry.

[Role] Compose projections, cache producers, sparse selection and attention.
[Invariants] Cache ownership and selection reuse are dependencies. Compression
  counts complete groups per request. Logical pairs and tiled memory reads are
  different quantities. No model/vendor identity selects an execution path.
[Timeouts & Locks] O(requests + query tiles) arithmetic, no observation reads.
[Side-Effect Policy] Constructs operators only.
"""
from collections.abc import Mapping
import math
from dataclasses import replace
from functools import partial
from typing import Literal

from mindie_sim.catalog.attention_common import (
    AttentionResult, _group_plan, store_dcp_degree,
)
from mindie_sim.catalog.cache import completed_rows, packed_region_row_bytes
from mindie_sim.catalog.context_parallel import (
    dcp_indexer_sharded, is_v41_replicated_cache_model, layer_dcp_degree,
    layer_dcp_factors, owned_positions,
)
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.schema.common import dtype_bytes, ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.operators.attention.batch_prefix_attention import (
    annotate_sparse_prefix_unknown, apply_attention_prefix_reads,
    apply_sparse_prefix_reads,
)
from mindie_sim.prediction.sharding.context_parallel import (
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
from mindie_sim.prediction.implementation_selection import (
    is_automatic_lowering, is_caller_pinned_binding,
)
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.operators.mla_prolog.compression import compressor
from .mla_epilogs import (
    SharedSelectionRef, block_quant_epilog, emit_ai_cpu_metadata_op,
    kv_cache_epilog, native_cache_store, native_fusion_active, lower_index_path,
    validate_m10_mla_legality,
)
from .mla_native import batched_matrix_metadata
from mindie_sim.prediction.memory_traffic import (
    AttentionAccessSpec, AttentionSchedule, TrafficPlan, plan_attention_traffic,
    require_machine_memory_hierarchy,
)
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
from .generic_latent_attention_views import (
    pool_sparse_view_records, window_cache_read_by_rank,
)

# Shared private alias retained strictly for cross-wave attention callers.
_attention_tensor_view = attention_tensor_view

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "vllm-project/vllm-ascend",
        "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        "vllm_ascend/attention/mla_v1.py",
        "f7f672e001651a748d2e88ef0f36797ef78db63321ac657f1c7a944c0f561c22",
    ),
    SourceRef(
        "vllm-project/vllm-ascend",
        "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        "vllm_ascend/attention/sfa_v1.py",
        "058aeb723a4ef3cdb0e07a5fc0180ec53f8227f62d98710144ecffb9cc66678f",
    ),
)

DEFAULT_INTERLEAVE = 128
DEFAULT_BLOCK_SIZE = 128
DEFAULT_QUERY_TILE = 128
MAX_GENERIC_OWNERS = 32
ROPE_OPS = 6
RMSNORM_OPS_PER_ELEM = 5
SOFTMAX_VECTOR_OPS_PER_PAIR = 5
GATE_APPLY_OPS_PER_ELEMENT = 5
FP32_BYTES = int(dtype_bytes("fp32"))


def _store_rows(
    rows_by_view, views, parallel, block, *, contexts=None,
    rank_candidates=None, storage_degrees=None, ratio: int = 1,
):
    return owned_store_rows(
        rows_by_view, views, parallel, block, contexts=contexts,
        rank_candidates=rank_candidates, storage_degrees=storage_degrees, ratio=ratio,
    )


def _is_fresh_dense_prefill_view(view) -> bool:
    return view.phase == "prefill" and int(view.context) == 0


def _unabsorbed_mla_eligible(
    plan, *, is_sparse: bool, dsa_prefill: bool = False,
    selected=None, head_contract=None, prolog=None,
) -> bool:
    return bool(
        prolog is None
        and plan.mla_variant != "latent_mqa"
        and plan.cache_kind == "dense"
        and not is_sparse
        and not plan.index_topk
        and not dsa_prefill
        and selected is None
        and head_contract is None
    )


def _is_unabsorbed_mla_prefill(
    plan, views, *, is_sparse: bool, dsa_prefill: bool = False,
    selected=None, head_contract=None, prolog=None,
) -> bool:
    """True when every local view is a fresh dense non-sparse latent_mla prefill."""
    return bool(
        _unabsorbed_mla_eligible(
            plan, is_sparse=is_sparse, dsa_prefill=dsa_prefill,
            selected=selected, head_contract=head_contract, prolog=prolog,
        )
        and bool(views)
        and all(_is_fresh_dense_prefill_view(view) for view in views)
    )


def _has_unabsorbed_mla_prefill(
    plan, views, *, is_sparse: bool, dsa_prefill: bool = False,
    selected=None, head_contract=None, prolog=None,
) -> bool:
    """True when at least one local view runs unabsorbed fresh dense MLA prefill."""
    return bool(
        _unabsorbed_mla_eligible(
            plan, is_sparse=is_sparse, dsa_prefill=dsa_prefill,
            selected=selected, head_contract=head_contract, prolog=prolog,
        )
        and any(_is_fresh_dense_prefill_view(view) for view in views)
    )


def _enforce_pre_rope_latent_order(b, compressed_latent: str | None) -> None:
    """Order in-place ``compress_rotary`` strictly after pre-RoPE ``index_k`` (dsa_v41.py:625)."""
    if not compressed_latent:
        return
    rotary_id = f"{b.prefix}.compress_rotary"
    index_k_id = f"{b.prefix}.index_k"
    rotary_idx = next(
        (i for i, op in enumerate(b.ops)
         if op.op_id == rotary_id and compressed_latent in op.deps),
        None,
    )
    index_k_idx = next(
        (i for i, op in enumerate(b.ops)
         if op.op_id == index_k_id and compressed_latent in op.deps),
        None,
    )
    if rotary_idx is None or index_k_idx is None or rotary_idx > index_k_idx:
        return
    # Collect transitive post-rotary closure (compress_rotary -> compressed_store -> pcp_kv_gather).
    moved_ids = {rotary_id}
    for op in b.ops[rotary_idx + 1:]:
        if any(dep in moved_ids for dep in op.deps):
            moved_ids.add(op.op_id)
    if index_k_id in moved_ids:
        return
    moved = []
    kept = []
    for op in b.ops:
        if op.op_id in moved_ids:
            if op.op_id == rotary_id and index_k_id not in op.deps:
                op = replace(op, deps=tuple(dict.fromkeys((*op.deps, index_k_id))))
            moved.append(op)
        else:
            kept.append(op)
    insert_at = next(i for i, op in enumerate(kept) if op.op_id == index_k_id) + 1
    b.ops = kept[:insert_at] + moved + kept[insert_at:]


def _memory_hierarchy(machine, *, hierarchy=None):
    """Declared device hierarchy for traffic planning; fails closed when absent."""
    return require_machine_memory_hierarchy(machine, hierarchy=hierarchy)


def _aic_owners(machine):
    """Declared AIC owner count of the representative shard, or ``None``."""
    dies = tuple(getattr(machine, "dies", ()) or ())
    return min(die.compute_units for die in dies) if dies else None


def _mla_idle_dcp_owner(
    *, prefix, plan, stage, deps, model, parallel, quant, mode, runtime, policy,
    machine, cp_interleave_size, cohort_dec_views, peer_dec_rows, local_dp_pos,
    pcp_factor, tp_gather, dp_width, physical_rank=None, explicit_physical_rank=False,
    hierarchy=None,
) -> AttentionResult:
    """Idle KV-owner peer in a cross-DP DCP cohort: gather Q, run attention, return partial O/LSE."""
    b = Builder(prefix, plan, stage, cohort_dec_views, model, mode)
    hierarchy = _memory_hierarchy(machine, hierarchy=hierarchy)
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


def _apply_mla_dcp_prefill_kv_gather(
    b, *, prefix, stage, plan, model, views, mode, store, long_store,
    sfa_replicated_prefill_path, is_sparse, layer_dcp, has_prefill, cp_interleave_size,
    dcp_cohort_views_by_member=None,
):
    member_lists = tuple(dcp_cohort_views_by_member) if dcp_cohort_views_by_member else (views,)
    if sfa_replicated_prefill_path:
        pf_kv_rows = max(
            (sum(int(view.kv_len) for view in (member_views or ())) for member_views in member_lists),
            default=0,
        )
        if pf_kv_rows > 0:
            kv_fmt = cache_format(model, "mla_kv", plan=plan, phase=mode)
            store = emit_sfa_prefill_kv_gather(
                b.ops, prefix=prefix, stage=stage, layer=plan.index,
                requests=b.common["requests"], mode=mode, store_dep=store,
                prefill_context_rows=pf_kv_rows,
                kv_row_bytes=packed_region_row_bytes(
                    model, "mla_kv", plan.cache_row_elements, plan=plan, phase=mode),
                layer_dcp=layer_dcp, kv_dtype=kv_fmt.dtype,
                recipe="sfa_dcp_prefill_kv_gather")
            if plan.cache_kind == "dense":
                long_store = store
    elif not is_sparse and layer_dcp > 1 and has_prefill:
        vblock = max(1, int(layer_dcp) * int(cp_interleave_size))
        pf_ctx_padded = max(
            (
                sum(
                    math.ceil(int(view.context) / vblock) * int(cp_interleave_size) * int(layer_dcp)
                    for view in (member_views or ()) if view.phase == "prefill" and view.context > 0
                )
                for member_views in member_lists
            ),
            default=0,
        )
        if pf_ctx_padded > 0:
            kv_fmt = cache_format(model, "mla_kv", plan=plan, phase=mode)
            store = emit_sfa_prefill_kv_gather(
                b.ops, prefix=prefix, stage=stage, layer=plan.index,
                requests=b.common["requests"], mode=mode, store_dep=store,
                prefill_context_rows=pf_ctx_padded,
                kv_row_bytes=packed_region_row_bytes(
                    model, "mla_kv", plan.cache_row_elements, plan=plan, phase=mode),
                layer_dcp=layer_dcp, kv_dtype=kv_fmt.dtype,
                recipe="dcp_prefill_kv_gather")
            if plan.cache_kind == "dense":
                long_store = store
    return store, long_store


def _resolve_mla_index_inputs(
    *, runtime, plan, is_v41, pcp_prefill, kpool_gathered, v4_hidden_gathered,
    cache_hidden, cache_views, cache_q, views, parallel, q,
):
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
    return pool_binding, v41_comp_gathered, idx_hidden, idx_views, idx_q


def _infer_mla_arch(runtime, machine) -> str:
    if machine is not None:
        m_fam = str(getattr(machine, "arch_family", "") or "").strip().upper()
        if m_fam in {"A2", "A3", "A5"}:
            return m_fam
        dies = getattr(machine, "dies", ()) or ()
        if dies:
            ca = getattr(dies[0], "compute_arch", None)
            arch_id = str(getattr(ca, "arch_id", "") or getattr(ca, "preset", "") or "").strip().lower()
            if "3510" in arch_id or arch_id == "a5":
                return "A5"
            if "910b" in arch_id or arch_id == "a2":
                return "A2"
            if "2201" in arch_id or arch_id == "a3":
                sku_low = str(getattr(machine, "sku", "") or "").lower()
                return "A2" if "910b" in sku_low else "A3"
    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    for key in ("arch_family", "device_arch"):
        arch = str(rt_meta.get(key, "") or "").strip().upper()
        if arch in {"A2", "A3", "A5"}:
            return arch
    return ""


def _apply_mla_sparse_offload_and_selected(
    b, *, prefix, stage, mode, plan, model, runtime, parallel, quant, views,
    deps, store, long_store, attn_topk, q, stage_of, heads, dcp_heads, logical_heads,
    latent_dim, decode_dcp, paired_sfa_mode, pcp_factor, tp_gather, dp_width,
    cohort_dec_views, peer_dec_rows, local_dp_pos, q_local, rotary_local,
    q_dcp, rotary_dcp, head_contract, selected, paged, mixed_quant=None, machine=None,
    physical_rank=None,
):
    sparse_offload = (select(runtime.kernels.get("lowerings", {}), "sparse_kv_offload")
                      if runtime else None)
    offload_budget = dict(model.metadata).get("sparse_kv_offload") if sparse_offload else None
    offload_info = None
    if offload_budget and plan.index in offload_budget.get("offload_layers", ()):
        prefill_views = tuple(view for view in views if view.phase == "prefill")
        decode_views = tuple(view for view in views if view.phase == "decode")
        if prefill_views and offload_budget["options"].get("layerwise_kv_offload"):
            sparse_offload.module.lower_prefill_offload(
                b, plan=plan, views=prefill_views, store_id=store, budget=offload_budget)
        if decode_views:
            store, attn_topk, offload_info = sparse_offload.module.lower_decode_offload(
                b, plan=plan, views=decode_views, store_id=store, topk_id=attn_topk,
                budget=offload_budget, parallel=parallel, physical_rank=physical_rank,
                fuse_copy_sfa_attention=True)
            if plan.cache_kind == "dense":
                long_store = store
    outputs: list[str] = []
    metadata_ready = None
    metadata_source = (select(runtime.kernels.get("lowerings", {}), "sparse_mla_metadata")
                       if runtime else None)
    if metadata_source:
        metadata_ready = metadata_source.module.lower(
            b, views=views, plan=plan, parallel=parallel, quant=quant,
            deps=(attn_topk or deps,), padded_tokens=q, stage_of=stage_of,
            **metadata_source.options)
    elif (
        (
            (plan.mla_variant == "latent_mqa" and plan.cache_kind == "window_compressed")
            or int(getattr(plan, "index_kpool", 0) or 0) > 0
        )
        and plan.index_topk > 0
        and any(int(v.q_len) > 0 for v in views)
    ):
        metadata_ready = emit_ai_cpu_metadata_op(
            b, kind="SparseFlashMlaMetadata", deps=(attn_topk or deps,),
            rows=max(1, q), arch=_infer_mla_arch(runtime, machine),
        )
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
    return store, long_store, attn_topk, offload_info, metadata_ready, outputs


def _resolve_mla_dcp_flags(
    *, plan, model, parallel, views, runtime, cp_interleave_size,
    pcp_factor, dp_width, cohort_dec_views, dcp_cohort_views_by_member=None,
):
    from mindie_sim.catalog.paired_sfa import execution
    layer_dcp = layer_dcp_degree(plan, parallel)
    is_sparse = bool(plan.indexer or plan.owns_topk
                     or plan.topk_owner_layer_id >= 0
                     or plan.index_k_owner_layer_id >= 0)
    sharded_indexer = dcp_indexer_sharded(model, runtime) and layer_dcp > 1
    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    has_local_prefill = any(view.phase == "prefill" for view in views)
    # Inspect the UNFILTERED cohort views (resolve_dcp_cohort_decode strips prefill views).
    has_cohort_prefill = any(
        view.phase == "prefill"
        for member_views in (dcp_cohort_views_by_member or ())
        for view in (member_views or ())
    ) or any(view.phase == "prefill" for view in (cohort_dec_views or ()))
    # Upstream sfa_cp.py:629: cohort-wide prefill presence forces decode-only peers
    # onto the same replicated-indexer KV-gather collective path.
    has_prefill = bool(
        has_local_prefill
        or has_cohort_prefill
        or rt_meta.get("pcp_has_global_prefill", False)
        or rt_meta.get("dcp_has_global_prefill", False)
    )
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
    return (layer_dcp, is_sparse, sharded_indexer, has_prefill,
            sfa_replicated_prefill_path, cp, decode_dcp, paired_sfa_mode,
            q_replicate, own_block, index_rope)


def _prepare_mla_views(plan, model, runtime, mode, parallel, views,
                       dcp_cohort_views_by_member, dcp_local_pos):
    from mindie_sim.prediction.operators.attention.attention import _cap_draft_views, _draft_kv_cap
    draft_cap = _draft_kv_cap(plan, model, runtime, mode)
    if draft_cap > 0:
        views = _cap_draft_views(views, draft_cap)
    pcp_factor, tp_gather, raw_dp_width = layer_dcp_factors(plan, parallel)
    dp_width = raw_dp_width if int(getattr(parallel, "dp", 1) or 1) >= raw_dp_width else 1
    cohort_dec_views, peer_dec_rows, local_dp_pos = resolve_dcp_cohort_decode(
        views, dp_width=dp_width,
        cohort_views_by_member=dcp_cohort_views_by_member,
        local_dp_pos=dcp_local_pos)
    if draft_cap > 0 and cohort_dec_views:
        cohort_dec_views = _cap_draft_views(cohort_dec_views, draft_cap)
    return views, pcp_factor, tp_gather, dp_width, cohort_dec_views, peer_dec_rows, local_dp_pos


def _compressor_native_fusion_active(runtime, machine) -> bool:
    """Gate fused compressor strictly on sealed ProbedAbiRegistry (zero fabricated aclnnCompressor)."""
    if runtime is None or not native_fusion_active(runtime, machine):
        return False
    if select(runtime.kernels.get("lowerings", {}), "compressor") is None:
        return False
    arch = _infer_mla_arch(runtime, machine)
    if not arch:
        return False
    from mindie_sim.backends.abi_registry import default_abi_registry
    return default_abi_registry().is_shipping_available("aclnnCompressor", arch)


def _resolve_mla_geometry(
    *, plan, model, parallel, views, runtime, policy, machine,
    is_sparse, layer_dcp, pcp_factor, tp_gather, dp_width, local_dp_pos,
    physical_rank, dcp_cohort_views_by_member, explicit_physical_rank,
    selected, prolog,
):
    heads = plan.num_heads // parallel.tp
    head_contract = (mla_head_padding.contract(model, plan, views)
                     if runtime and select(runtime.kernels.get("lowerings", {}), "mla_head_padding")
                     else None)
    logical_heads = heads
    heads = head_contract["physical_heads"] if head_contract else heads
    dsa_prefill = (parallel.cp_plan == "dsa_cp" and parallel.tp > 1
                   and any(view.phase == "prefill" for view in views))
    width = plan.qk_nope_head_dim + plan.qk_rope_head_dim
    latent = plan.mla_variant != "latent_mqa"
    query_tile = policy.query_tile if policy else DEFAULT_QUERY_TILE
    kv_width = plan.kv_lora_rank + (plan.qk_rope_head_dim if latent else 0)
    has_unabsorbed_prefill = _has_unabsorbed_mla_prefill(
        plan, views, is_sparse=is_sparse, dsa_prefill=dsa_prefill,
        selected=selected, head_contract=head_contract, prolog=prolog,
    )
    all_unabsorbed_prefill = _is_unabsorbed_mla_prefill(
        plan, views, is_sparse=is_sparse, dsa_prefill=dsa_prefill,
        selected=selected, head_contract=head_contract, prolog=prolog,
    )
    store_dcp_ranks = (
        resolve_cp_rank_candidates(
            parallel, eff_cp=layer_dcp, pcp_factor=pcp_factor, tp_gather=tp_gather,
            dp_width=dp_width, local_dp_pos=local_dp_pos, physical_rank=physical_rank,
            explicit_dp_cohort=(dcp_cohort_views_by_member is not None),
            explicit_physical_rank=bool(explicit_physical_rank),
        )
        if layer_dcp > 1 else (0,)
    )
    latent_dim = plan.kv_lora_rank if latent else plan.head_dim
    qk_dim = (latent_dim + plan.qk_rope_head_dim) if latent else width
    return (
        heads, logical_heads, head_contract, dsa_prefill, width, latent,
        query_tile, kv_width, has_unabsorbed_prefill, all_unabsorbed_prefill,
        store_dcp_ranks, latent_dim, qk_dim,
    )


def _finalize_mla_index_and_head_inputs(
    b, *, prefix, stage, mode, plan, q, valid, cp, decode_dcp, sharded_indexer,
    cp_interleave_size, compressed_latent, topk, head_contract,
    q_local, rotary_local, q_dcp, rotary_dcp,
):
    _enforce_pre_rope_latent_order(b, compressed_latent)
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
    return attn_topk, q_local, rotary_local, q_dcp, rotary_dcp


def mla(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
        padded_tokens, cache_ids, topk_ids, index_cache_ids, candidate_ids,
        stage_of, cp_interleave_size=DEFAULT_INTERLEAVE, runtime=None, policy=None,
        machine=None, hierarchy=None, reuse_index_selection=False, shared_topk=None,
        dcp_cohort_views_by_member=None, dcp_local_pos=0,
        physical_rank=None, explicit_physical_rank=False):
    (views, pcp_factor, tp_gather, dp_width,
     cohort_dec_views, peer_dec_rows, local_dp_pos) = _prepare_mla_views(
        plan, model, runtime, mode, parallel, views,
        dcp_cohort_views_by_member, dcp_local_pos)
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
            explicit_physical_rank=explicit_physical_rank, hierarchy=hierarchy)
    b = Builder(prefix, plan, stage, views, model, mode)
    hierarchy = _memory_hierarchy(machine, hierarchy=hierarchy)
    b.l2_bytes = hierarchy.effective_l2_bytes
    (layer_dcp, is_sparse, sharded_indexer, has_prefill,
     sfa_replicated_prefill_path, cp, decode_dcp, paired_sfa_mode,
     q_replicate, own_block, index_rope) = _resolve_mla_dcp_flags(
        plan=plan, model=model, parallel=parallel, views=views, runtime=runtime,
        cp_interleave_size=cp_interleave_size, pcp_factor=pcp_factor,
        dp_width=dp_width, cohort_dec_views=cohort_dec_views,
        dcp_cohort_views_by_member=dcp_cohort_views_by_member)
    eff_dcp_only = max(1, tp_gather * dp_width)
    prolog, partial_rotary, paged, selected, mixed_quant = _select_mla_lowerings(
        runtime, plan, parallel, model, quant, views, padded_tokens, machine,
        eff_dcp=eff_dcp_only, is_sparse=is_sparse,
        cp_interleave_size=cp_interleave_size, sharded_indexer=sharded_indexer)
    q, h = padded_tokens, model.hidden_size
    (heads, logical_heads, head_contract, dsa_prefill, width, latent,
     query_tile, kv_width, has_unabsorbed_prefill, all_unabsorbed_prefill,
     store_dcp_ranks, latent_dim, qk_dim) = _resolve_mla_geometry(
        plan=plan, model=model, parallel=parallel, views=views, runtime=runtime,
        policy=policy, machine=machine, is_sparse=is_sparse, layer_dcp=layer_dcp,
        pcp_factor=pcp_factor, tp_gather=tp_gather, dp_width=dp_width,
        local_dp_pos=local_dp_pos, physical_rank=physical_rank,
        dcp_cohort_views_by_member=dcp_cohort_views_by_member,
        explicit_physical_rank=explicit_physical_rank,
        selected=selected, prolog=prolog)
    do_rope = bool(plan.mla_do_rope)
    native_fusion = _compressor_native_fusion_active(runtime, machine)
    native_cache_epilogs = native_fusion_active(runtime, machine)
    is_v41 = is_v41_replicated_cache_model(model)
    (q_local, q_unabsorbed, rotary_local, q_dcp, rotary_dcp, q_norm, valid, pcp_prefill,
     store, dcp_heads, cache_hidden, cache_views, cache_q, v4_hidden_gathered,
     kpool_gathered, kv_b_dep) = _project_mla_inputs(
        b, q, h, heads, width, kv_width, deps, prolog, do_rope, partial_rotary, views,
        parallel, own_block, native_cache_epilogs, latent, quant,
        decode_dcp=decode_dcp, pcp_factor=pcp_factor, tp_gather=tp_gather,
        dp_width=dp_width, q_replicate=q_replicate, is_sparse=is_sparse,
        peer_rows=(peer_dec_rows if (dp_width > 1 and decode_dcp) else None),
        local_dp_pos=local_dp_pos, runtime=runtime, is_v41=is_v41, machine=machine,
        layer_dcp=layer_dcp, store_rank_candidates=store_dcp_ranks,
        has_unabsorbed_prefill=has_unabsorbed_prefill,
        all_unabsorbed_prefill=all_unabsorbed_prefill)
    long_store, compressed_latent = _produce_long_cache(
        b, store, stage_of, cache_views, cache_hidden,
        sum(v.q_len for v in cache_views), cache_q,
        native_fusion, parallel, own_block,
        native_cache_epilogs, pcp_prefill and not v4_hidden_gathered, cache_ids, cp,
        runtime=runtime, is_v41=is_v41, layer_dcp=layer_dcp,
        store_rank_candidates=store_dcp_ranks)
    store, long_store = _apply_mla_dcp_prefill_kv_gather(
        b, prefix=prefix, stage=stage, plan=plan, model=model, views=views, mode=mode,
        store=store, long_store=long_store,
        sfa_replicated_prefill_path=sfa_replicated_prefill_path,
        is_sparse=is_sparse, layer_dcp=layer_dcp, has_prefill=has_prefill,
        cp_interleave_size=cp_interleave_size,
        dcp_cohort_views_by_member=dcp_cohort_views_by_member)
    pool_binding, v41_comp_gathered, idx_hidden, idx_views, idx_q = _resolve_mla_index_inputs(
        runtime=runtime, plan=plan, is_v41=is_v41, pcp_prefill=pcp_prefill,
        kpool_gathered=kpool_gathered, v4_hidden_gathered=v4_hidden_gathered,
        cache_hidden=cache_hidden, cache_views=cache_views, cache_q=cache_q,
        views=views, parallel=parallel, q=q)
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
                               or v41_comp_gathered),
        machine=machine, store_rank_candidates=store_dcp_ranks)
    attn_topk, q_local, rotary_local, q_dcp, rotary_dcp = _finalize_mla_index_and_head_inputs(
        b, prefix=prefix, stage=stage, mode=mode, plan=plan, q=q, valid=valid,
        cp=cp, decode_dcp=decode_dcp, sharded_indexer=sharded_indexer,
        cp_interleave_size=cp_interleave_size, compressed_latent=compressed_latent,
        topk=topk, head_contract=head_contract, q_local=q_local,
        rotary_local=rotary_local, q_dcp=q_dcp, rotary_dcp=rotary_dcp)
    store, long_store, attn_topk, offload_info, metadata_ready, outputs = (
        _apply_mla_sparse_offload_and_selected(
            b, prefix=prefix, stage=stage, mode=mode, plan=plan, model=model,
            runtime=runtime, parallel=parallel, quant=quant, views=views,
            deps=deps, store=store, long_store=long_store, attn_topk=attn_topk,
            q=q, stage_of=stage_of, heads=heads, dcp_heads=dcp_heads,
            logical_heads=logical_heads, latent_dim=latent_dim,
            decode_dcp=decode_dcp, paired_sfa_mode=paired_sfa_mode,
            pcp_factor=pcp_factor, tp_gather=tp_gather, dp_width=dp_width,
            cohort_dec_views=cohort_dec_views, peer_dec_rows=peer_dec_rows,
            local_dp_pos=local_dp_pos, q_local=q_local, rotary_local=rotary_local,
            q_dcp=q_dcp, rotary_dcp=rotary_dcp, head_contract=head_contract,
            selected=selected, paged=paged, mixed_quant=mixed_quant, machine=machine,
            physical_rank=physical_rank)
    )
    eff_paired_sfa_mode = False if (offload_info and offload_info.get("fused_sfa_recipe")) else paired_sfa_mode
    request_op = partial(
        _append_mla_request, b=b, runtime=runtime, parallel=parallel, quant=quant,
        local_heads=heads, dcp_heads=dcp_heads, dp_width=dp_width,
        qk_dim=qk_dim, latent_dim=latent_dim, unabsorbed_qk_dim=width,
        unabsorbed_v_dim=plan.v_head_dim, latent=latent, query_tile=query_tile,
        hierarchy=hierarchy, topk=attn_topk, q_local=q_local, q_unabsorbed=q_unabsorbed,
        rotary_local=rotary_local, q_dcp=q_dcp, rotary_dcp=rotary_dcp,
        store=store, long_store=long_store, metadata_ready=metadata_ready,
        machine=machine, head_contract=head_contract, offload_info=offload_info,
        kv_b_dep=kv_b_dep)
    unabsorbed_outputs = _append_mla_attention(
        b, outputs, views, runtime, selected, parallel, dsa_prefill, model, plan, quant,
        query_tile, cp_interleave_size, attn_topk, heads, dcp_heads, latent_dim, qk_dim,
        reuse_index_selection, request_op, mixed_quant, pcp_prefill, eff_paired_sfa_mode,
        is_sparse, pcp_factor, tp_gather, dp_width, policy, logical_heads, q,
        hierarchy, stage, mode, q_local, store,
        cohort_decode_views=(cohort_dec_views if dp_width > 1 else None),
        peer_decode_rows=(peer_dec_rows if dp_width > 1 else None),
        local_dp_pos=local_dp_pos, sharded_indexer=sharded_indexer,
        physical_rank=physical_rank,
        explicit_dp_cohort=(dcp_cohort_views_by_member is not None),
        explicit_physical_rank=bool(explicit_physical_rank),
        sfa_replicated_prefill_path=sfa_replicated_prefill_path,
        has_unabsorbed_prefill=has_unabsorbed_prefill,
        unabsorbed_qk_dim=width, unabsorbed_v_dim=plan.v_head_dim)
    last = _project_mla_output(
        b, logical_heads, latent, outputs, q, plan, head_contract, deps, h, partial_rotary,
        parallel, model, dsa_prefill, views=views, runtime=runtime,
        has_unabsorbed_prefill=has_unabsorbed_prefill,
        unabsorbed_outputs=unabsorbed_outputs)
    return AttentionResult(ops=b.ops, last=last, cache_store_id=long_store or store,
                           index_store_id=index_store, topk_id=topk,
                           candidate_id=candidate)


def _resolve_automatic_mla_preprocess_mode(
    *,
    runtime,
    plan,
    parallel,
    model,
    quant,
    views,
    padded_tokens,
    machine,
    eff_dcp: int = 1,
    is_sparse: bool = False,
) -> str:
    """Resolve the 5-branch M1 preprocess mode from runtime/plan/model geometry."""
    from mindie_sim.prediction.operators.mla_prolog import resolve_mla_preprocess_mode

    arch = _infer_mla_arch(runtime, machine)
    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    phase_str = "prefill" if any(v.phase == "prefill" for v in views) else "decode"
    q_attn = quant.for_category("attention") if hasattr(quant, "for_category") else quant
    w_dt = str(
        getattr(q_attn, "weight_format", "")
        or getattr(q_attn, "weight_dtype", "")
        or getattr(quant, "weight_format", "")
        or getattr(quant, "weight_dtype", "")
        or ""
    ).strip().lower()
    act_dt = str(
        getattr(q_attn, "activation_format", "")
        or getattr(model, "activation_dtype", "")
        or ""
    ).strip().lower()
    if not w_dt:
        w_dt = act_dt
    if not act_dt or not w_dt:
        raise ModelingError("mla_preprocess_dtype_missing")
    raw_input_size = (
        rt_meta.get("fused_qkv_a_input_size")
        or getattr(plan, "input_hidden_size", None)
        or getattr(plan, "fused_qkv_a_input_size", None)
        or getattr(model, "hidden_size", None)
    )
    input_size = int(raw_input_size) if raw_input_size else None
    num_kv_heads = int(getattr(plan, "num_kv_heads", 1) or 1)
    num_q_heads = int(getattr(plan, "num_heads", 0) or getattr(model, "num_heads", 0) or 0)
    qk_nope_dim = int(getattr(plan, "qk_nope_head_dim", 0) or 0)
    return resolve_mla_preprocess_mode(
        arch=arch,
        phase=phase_str,
        num_tokens=int(padded_tokens),
        q_lora_rank=int(plan.q_lora_rank or 0),
        kv_lora_rank=int(plan.kv_lora_rank or 0),
        qk_rope_head_dim=int(plan.qk_rope_head_dim or 0),
        has_rope=bool(plan.mla_do_rope),
        is_sparse_sfa=bool(is_sparse),
        dcp_size=int(eff_dcp),
        pcp_size=int(getattr(parallel, "pcp", 1) or 1),
        enable_mlapo=bool(rt_meta.get("enable_mlapo", False)),
        enable_mla_prolog=True,
        qkv_a_gate_fused=bool(getattr(plan, "qkv_a_gate_fused", False)),
        is_w4a8=("w4a8" in w_dt or "int4" in w_dt or "w4a16" in w_dt),
        is_int8_kv=(str(getattr(quant, "kv_dtype", "") or "").lower() == "int8"),
        has_kv_scale=bool(rt_meta.get("has_kv_scale", True)),
        is_strided_kv_cache=bool(rt_meta.get("is_strided_kv_cache", False)),
        is_draft_model=(str(getattr(plan, "scope", "target")) == "draft"),
        act_dtype=act_dt,
        weight_quant=w_dt,
        input_size=input_size,
        num_kv_heads=num_kv_heads,
        num_q_heads=num_q_heads,
        qk_nope_head_dim=qk_nope_dim,
    )


_VALID_MLA_STACK_TRACKS = frozenset({"eager", "shipping", "proposed", "target"})


def _select_mla_lowerings(
    runtime, plan, parallel, model, quant, views, padded_tokens, machine,
    *, eff_dcp: int = 1, is_sparse: bool = False,
    cp_interleave_size: int = DEFAULT_INTERLEAVE, sharded_indexer: bool = False,
):
    arch = _infer_mla_arch(runtime, machine)
    lowerings = runtime.kernels.get("lowerings", {}) if runtime else {}
    rt_meta_sel = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    exec_pol = (
        rt_meta_sel.get("execution_policy")
        if isinstance(rt_meta_sel.get("execution_policy"), Mapping) else {}
    )
    single_spine = (
        rt_meta_sel.get("single_spine")
        if isinstance(rt_meta_sel.get("single_spine"), Mapping) else {}
    )
    stack_track = str(
        rt_meta_sel.get("stack_track") or single_spine.get("stack_track") or ""
    ).strip().lower()
    if stack_track and stack_track not in _VALID_MLA_STACK_TRACKS:
        raise ModelingError(f"mla_stack_track_invalid:{stack_track}")
    explicit_non_shipping = bool(
        stack_track in {"eager", "target"}
        or rt_meta_sel.get("va_recommended") is False
        or exec_pol.get("va_recommended") is False
    )
    shipping_kpool = bool(
        select(lowerings, "pool_key_indexer") is not None
        or rt_meta_sel.get("va_profile")
        or (not explicit_non_shipping and stack_track in {"shipping", "proposed"})
    )
    validate_m10_mla_legality(
        plan=plan,
        eff_dcp=eff_dcp,
        arch=arch,
        quant_kv_dtype=quant.kv_dtype,
        cp_interleave_size=cp_interleave_size,
        sharded_indexer=sharded_indexer,
        enforce_shipping_kpool=shipping_kpool,
    )
    prolog = (select(lowerings, "mla_prolog") if runtime else None)
    auto_prolog = not is_caller_pinned_binding(runtime, "lowerings", "mla_prolog")
    if (prolog is not None and auto_prolog and plan.q_lora_rank
            and _is_unabsorbed_mla_prefill(plan, views, is_sparse=is_sparse)
            and not bool(rt_meta_sel.get("is_strided_kv_cache", False))):
        prolog = None
    elif prolog is not None and is_automatic_lowering(getattr(runtime, "metadata", ()), "mla_prolog"):
        mode_resolved = _resolve_automatic_mla_preprocess_mode(
            runtime=runtime, plan=plan, parallel=parallel, model=model, quant=quant,
            views=views, padded_tokens=padded_tokens, machine=machine,
            eff_dcp=eff_dcp, is_sparse=is_sparse,
        )
        # Only ``prolog_v3`` / ``prolog_v3_learned_rope_tail`` use ``vllm_ascend_mla_prolog_v3``.
        # ``mlapo``, ``strided_kv_adapter``, and ``native_decomposed`` MUST NOT retain prolog_v3.
        if mode_resolved not in {"prolog_v3", "prolog_v3_learned_rope_tail"}:
            prolog = None
    if prolog:
        if not plan.q_lora_rank:
            raise ModelingError("mla_prolog_v3_direct_q_unsupported")
        prolog.module.validate(plan=plan, parallel=parallel, model=model, quant=quant,
                               lowering_passes=runtime.lowering_passes, arch=arch, **prolog.options)
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


_GROUP_QUANT_OPS_PER_ELEMENT = 4


def _reconcile_kv_store_format(op, *, model, cache_fmt, cache_region, rows, width, row_bytes):
    """Align kv_store byte accounting with the resolved layer/phase CacheFormat."""
    meta = dict(op.metadata)
    stored = rows * row_bytes
    current_row = meta.get("cache_row_bytes") or dict(meta.get("cache_representation") or {}).get("row_bytes")
    if int(op.mte3_bytes) == stored and current_row == row_bytes:
        return op
    source = rows * width * dtype_bytes(model.activation_dtype)
    if cache_fmt.group_size:
        groups = (width + cache_fmt.group_size - 1) // cache_fmt.group_size
        shape = (rows, width)
        mte2 = float(source)
        vec_ops = float(rows * (_GROUP_QUANT_OPS_PER_ELEMENT * width + groups))
        dtype = model.activation_dtype
    else:
        shape = (max(1, rows), max(1, row_bytes))
        mte2 = float(stored)
        vec_ops = float(stored)
        dtype = cache_fmt.dtype
    meta["cache_row_bytes"] = row_bytes
    meta["cache_format"] = {
        "region": cache_region, "dtype": cache_fmt.dtype,
        "group_size": cache_fmt.group_size, "scale_dtype": cache_fmt.scale_dtype,
        "suffix_elements": cache_fmt.suffix_elements,
        "suffix_dtype": cache_fmt.suffix_dtype,
        "packed_elements": cache_fmt.packed_elements,
    }
    if isinstance(meta.get("cache_representation"), dict):
        meta["cache_representation"] = {
            **meta["cache_representation"],
            "dtype": cache_fmt.dtype,
            "group_size": cache_fmt.group_size,
            "scale_dtype": cache_fmt.scale_dtype,
            "row_bytes": row_bytes,
        }
    return replace(
        op,
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype=dtype, vector_ops=vec_ops, mte2_bytes=mte2, mte3_bytes=float(stored),
        hbm_bytes=mte2 + float(stored), metadata=tuple(meta.items()),
    )


def _produce_mla_cache(
    b, views, parallel, own_block, rotary, kv, kv_a, do_rope, prolog, native_cache_epilogs,
    pcp_prefill, *, is_v41=False, padded_tokens=0, layer_dcp=1, store_rank_candidates=None,
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
        storage_degrees = [
            min(store_dcp_degree(parallel, view.phase), max(1, int(layer_dcp)))
            for view in views
        ]
        store_rows = _store_rows(
            [view.q_len for view in views], views, parallel, own_block,
            rank_candidates=store_rank_candidates, storage_degrees=storage_degrees,
        )
    packed_store = (prolog.module.cache_store(
        b, deps=cache_deps, rows=store_rows, plan=plan, **prolog.options)
        if prolog else None)
    store = packed_store or (representation_store(
        b, "kv_store", cache_deps, store_rows, plan.cache_row_elements,
        cache_region))
    if store is not None and packed_store is None:
        store_phase = b.common["phase"]
        cache_fmt = cache_format(model, cache_region, plan=plan, phase=store_phase)
        expected_row_b = packed_region_row_bytes(
            model, cache_region, plan.cache_row_elements, plan=plan, phase=store_phase)
        for position, item in enumerate(b.ops):
            if item.op_id != store:
                continue
            item = _reconcile_kv_store_format(
                item, model=model, cache_fmt=cache_fmt, cache_region=cache_region,
                rows=store_rows, width=plan.cache_row_elements, row_bytes=expected_row_b,
            )
            written = kv_cache_epilog(item, cache_fmt, cache_region,
                                      native=native_cache_epilogs, row_bytes=expected_row_b)
            if plan.qk_rope_head_dim == 0:
                written = block_quant_epilog(
                    written, cache_fmt, cache_region, native=native_cache_epilogs,
                    width=plan.cache_row_elements, row_bytes=expected_row_b)
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
                model, cache_region, plan.cache_row_elements, plan=plan, phase=b.common["phase"])
            kv_dtype = cache_format(model, cache_region, plan=plan, phase=b.common["phase"]).dtype
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


def _prepare_mla_pcp_hidden_gathers(
    b, *, plan, model, views, parallel, q, h, deps, norm_dtype, runtime, is_v41, pcp_prefill,
):
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
    return kpool_gathered, v4_hidden_gathered, cache_views, cache_q, cache_hidden


def _declare_mla_prolog_or_semantic(
    b, *, prolog, plan, model, parallel, quant, runtime, q,
    q_a, q_norm, q_b, kv_a, kv, rotary, produced, q_ready, latent,
    preprocess_mode=None, unabsorbed_prefill=False,
):
    if prolog:
        prolog.module.declare(
            b.ops, prefix=b.prefix, plan=plan, model=model, parallel=parallel, quant=quant,
            tokens=q, members=tuple(dict.fromkeys(
                (q_a, q_norm, q_b, kv_a, kv, rotary, produced, q_ready))),
            q_a=q_a, q_b=q_b, kv_a=kv_a, q_absorb=q_ready, **prolog.options)
        return
    if not plan.q_lora_rank:
        return
    from mindie_sim.prediction.operators.mla_prolog import family_mla_prolog
    rt_meta = dict(getattr(runtime, "metadata", ()) or ())
    single_spine = (
        rt_meta.get("single_spine")
        if isinstance(rt_meta.get("single_spine"), Mapping) else {}
    )
    track = str(rt_meta.get("stack_track") or single_spine.get("stack_track") or "").strip().lower()
    if unabsorbed_prefill:
        reason = "unabsorbed_mla_prefill"
    elif track == "eager":
        reason = "eager_unfused_mla_prolog"
    elif preprocess_mode == "mlapo":
        reason = "mlapo_selected"
    elif preprocess_mode == "strided_kv_adapter":
        reason = "strided_kv_adapter_selected"
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
        q_absorb=q_ready if (latent and not unabsorbed_prefill) else None,
        anchor=q_ready, reason=reason)


def _emit_unabsorbed_kv_b(
    b, *, kv_src: str, rows: int, heads: int, plan, model,
) -> str:
    """Emit dense BF16 ``kv_b_proj`` and contiguous K_nope/V split copies (mla_v1.py:1301,3219)."""
    act_dtype = model.activation_dtype
    out_width = heads * (plan.qk_nope_head_dim + plan.v_head_dim)
    dense_quant = QuantSpec(
        name=act_dtype, weight_format=act_dtype,
        activation_format=act_dtype, native_format=act_dtype,
    )
    kv_b_id = b.add(gemm_op(
        op_id=b.prefix + ".kv_b", deps=(kv_src,), m=rows, n=out_width,
        k=plan.kv_lora_rank, quant=dense_quant, category="mla_absorb",
        recipe="kv_b", activation_dtype=act_dtype, output_dtype=act_dtype,
        metadata=(("native_kind", "kv_b_proj"), ("unabsorbed_mla_prefill", True)),
        **b.common,
    ))
    copy_bytes = float(rows * out_width * dtype_bytes(act_dtype))
    return b.work(
        "kv_b_split_copy", (kv_b_id,), (rows, out_width),
        vector=0.0, read=copy_bytes, write=copy_bytes,
        family="copy", dtype=act_dtype, recipe="kv_b_split_copy",
        metadata=(("native_kind", "kv_b_split_contiguous"), ("unabsorbed_mla_prefill", True)),
    )


def _project_mla_inputs(
    b, q, h, heads, width, kv_width, deps, prolog, do_rope, partial_rotary, views, parallel,
    own_block, native_cache_epilogs, latent, quant, *, decode_dcp, pcp_factor=1, tp_gather,
    dp_width, q_replicate, is_sparse, peer_rows=None, local_dp_pos=0, runtime=None,
    is_v41=False, machine=None, layer_dcp=1, store_rank_candidates=None,
    has_unabsorbed_prefill=False, all_unabsorbed_prefill=False,
):
    plan, model = b.plan, b.model
    q_replicate_active = bool(q_replicate and tp_gather > 1 and layer_dcp_degree(plan, parallel) > 1)
    proj_heads = (heads * tp_gather) if q_replicate_active else heads
    post_q_heads = heads if (q_replicate_active and not decode_dcp) else proj_heads
    norm_dtype = model.activation_dtype
    norm_bytes = dtype_bytes(norm_dtype)
    pcp_prefill = parallel.pcp > 1 and any(view.phase == "prefill" for view in views)
    kpool_gathered, v4_hidden_gathered, cache_views, cache_q, cache_hidden = (
        _prepare_mla_pcp_hidden_gathers(
            b, plan=plan, model=model, views=views, parallel=parallel, q=q, h=h,
            deps=deps, norm_dtype=norm_dtype, runtime=runtime, is_v41=is_v41,
            pcp_prefill=pcp_prefill)
    )
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
        if (plan.mla_variant == "latent_mqa"
                and plan.cache_kind == "window_compressed" and not is_v41):
            # V4 DSA (Flash-0731 / Pro-0813 / 10t) applies a weightless RMSNorm
            # over each 512-dim projected query head between q_b and RoPE.
            q_ready = b.vector(
                "q_head_norm", (q_ready,), q * proj_heads * width, RMSNORM_OPS_PER_ELEM,
                family="rmsnorm", shape=(q, proj_heads, width),
                read_bytes_override=q * proj_heads * width * norm_bytes,
                write_bytes_override=q * proj_heads * width * norm_bytes,
                metadata=(("reduction_width", width),
                          ("norm_has_weight", False),
                          ("norm_eps", model.norm_eps)))
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
        is_v41=is_v41, padded_tokens=q, layer_dcp=layer_dcp,
        store_rank_candidates=store_rank_candidates)
    q_unabsorbed = q_ready
    unabsorbed_rows = (
        sum(int(v.q_len) for v in views if _is_fresh_dense_prefill_view(v))
        if has_unabsorbed_prefill else 0
    )
    has_local_absorbed = any(not _is_fresh_dense_prefill_view(v) for v in views)
    absorbed_rows = (
        max(0, q - unabsorbed_rows)
        if (has_unabsorbed_prefill and has_local_absorbed)
        else (0 if has_unabsorbed_prefill else q)
    )
    if latent and (absorbed_rows > 0 or prolog is not None):
        abs_m = q if prolog is not None else absorbed_rows
        q_ready = b.matrix("q_absorb", (q_ready,), abs_m * post_q_heads, plan.kv_lora_rank,
                           plan.qk_nope_head_dim, weight_count=post_q_heads,
                           category="mla_absorb",
                           metadata=batched_matrix_metadata(
                               tokens=abs_m, heads=post_q_heads, input_width=plan.qk_nope_head_dim,
                               output_width=plan.kv_lora_rank, role="key_absorption"))
    kv_b_dep = (
        _emit_unabsorbed_kv_b(
            b, kv_src=kv, rows=unabsorbed_rows, heads=post_q_heads, plan=plan, model=model,
        )
        if unabsorbed_rows > 0 else None
    )
    preprocess_mode = (
        _resolve_automatic_mla_preprocess_mode(
            runtime=runtime, plan=plan, parallel=parallel, model=model, quant=quant,
            views=views, padded_tokens=q, machine=machine,
            eff_dcp=max(1, tp_gather * dp_width), is_sparse=is_sparse,
        )
        if (prolog is None and runtime is not None and plan.q_lora_rank) else None
    )
    _declare_mla_prolog_or_semantic(
        b, prolog=prolog, plan=plan, model=model, parallel=parallel, quant=quant,
        runtime=runtime, q=q, q_a=q_a, q_norm=q_norm, q_b=q_b, kv_a=kv_a, kv=kv,
        rotary=rotary, produced=produced, q_ready=q_ready, latent=latent,
        preprocess_mode=preprocess_mode, unabsorbed_prefill=all_unabsorbed_prefill)
    q_local = q_ready
    rotary_local = rotary
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
    return (q_local, q_unabsorbed, rotary_local, q_dcp, rotary_dcp, q_norm, valid, pcp_prefill,
            store, dcp_heads, cache_hidden, cache_views, cache_q, v4_hidden_gathered,
            kpool_gathered, kv_b_dep)


def _emit_v41_pcp_c2_pool_stage(
    b, *, views, deps, valid, parallel, runtime, plan, width, expansion, hidden,
    act_dtype, act_bytes, local_comp_rows, common_peer_rows, global_token_rows,
):
    """Emit FP32 C2 projection, PCP ring tail gather, ``_pool_kernel``, and ring write."""
    from mindie_sim.prediction.graph import GraphPolicy
    from mindie_sim.prediction.operators.mla_prolog.compression import (
        CAST_VECTOR_OPS_PER_ELEMENT,
        COMPRESSION_FP32_RECIPE,
        FP32_BYTES,
        KV_SCORE_PAIR_STREAMS,
        POOL_VECTOR_OPS_PER_ELEMENT,
        ZERO_FILL_VECTOR_OPS_PER_ELEMENT,
    )
    ratio = max(1, plan.compress_ratio)
    proj_tokens = valid
    input_cast = b.vector(
        "compress_input_cast", (deps,), proj_tokens * hidden,
        CAST_VECTOR_OPS_PER_ELEMENT,
        family="cast", dtype="fp32", shape=(proj_tokens, hidden),
        read_bytes_override=proj_tokens * hidden * act_bytes,
        write_bytes_override=proj_tokens * hidden * FP32_BYTES,
        metadata=(("input_dtype", act_dtype), ("output_dtype", "fp32")),
    )
    fp32_kwargs = {
        "category": COMPRESSION_FP32_RECIPE,
        "activation_dtype": "fp32",
        "output_dtype": "fp32",
        "metadata": (("compute_format", "fp32"),),
    }
    projection = b.matrix(
        "compress_values", (input_cast,), proj_tokens, expansion * width, hidden, **fp32_kwargs)
    gates = b.matrix(
        "compress_gates", (input_cast,), proj_tokens, expansion * width, hidden, **fp32_kwargs)
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
    pool_read_bytes = (
        local_comp_rows * ratio * KV_SCORE_PAIR_STREAMS * expansion * width * FP32_BYTES
        if local_comp_rows > 0 else 0
    )
    pool_write_bytes = (valid + local_comp_rows) * width * act_bytes
    pool_vec_ops = (
        valid * width * ZERO_FILL_VECTOR_OPS_PER_ELEMENT
        + local_comp_rows * ratio * expansion * width * POOL_VECTOR_OPS_PER_ELEMENT
    )
    pooled = b.vector(
        "compress_pool", pool_deps, pool_vec_ops, 1,
        read_bytes_override=pool_read_bytes,
        write_bytes_override=pool_write_bytes,
        metadata=(("state_dtype", "fp32"),
                  ("valid_completed_rows", local_comp_rows),
                  ("token_aligned_rows", valid),
                  ("native_kinds", ("ZerosLike", "_pool_kernel")),
                  ("kernel", "_pool_kernel")),
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
    return pooled, ring_write_dep


def _produce_v41_pcp_long_cache(
    b, views, deps, valid, q, parallel, native_cache_epilogs, runtime,
):
    """Exact ``dsa_v41_cp.py:662-743`` pre-pool C2 ring gather -> pool -> latent gather -> store."""
    from mindie_sim.prediction.operators.mla_prolog.compression import (
        COMPRESSION_RECIPE,
        FP32_BYTES,
        NORM_VECTOR_OPS_PER_ELEMENT,
        ROPE_TRIG_TABLES,
        ROTARY_VECTOR_OPS_PER_ELEMENT,
    )
    plan, model = b.plan, b.model
    ratio = max(1, plan.compress_ratio)
    width = plan.head_dim
    expansion, hidden = plan.compressor_width, model.hidden_size
    act_dtype = model.activation_dtype
    act_bytes = dtype_bytes(act_dtype)
    local_comp_rows = sum(owned_completed_rows(view, ratio) for view in views)
    global_comp_rows = sum(
        completed_rows(
            view.context + (view.global_q_len or view.q_len if view.phase == "prefill" else view.q_len),
            ratio,
        ) - completed_rows(view.context, ratio)
        for view in views
    )
    global_comp_rows = max(global_comp_rows, local_comp_rows)
    common_peer_rows, global_token_rows = pcp_uniform_peer_rows(
        views, parallel.pcp, local_rows=valid,
    )
    if ratio == 2:
        pooled, ring_write_dep = _emit_v41_pcp_c2_pool_stage(
            b, views=views, deps=deps, valid=valid, parallel=parallel,
            runtime=runtime, plan=plan, width=width, expansion=expansion,
            hidden=hidden, act_dtype=act_dtype, act_bytes=act_bytes,
            local_comp_rows=local_comp_rows, common_peer_rows=common_peer_rows,
            global_token_rows=global_token_rows,
        )
    else:
        pooled = b.matrix(
            "compress_values", (deps,), q, expansion * width, hidden,
            category=COMPRESSION_RECIPE)
        ring_write_dep = None
    norm_rows = valid if ratio == 2 else local_comp_rows
    local_latent = (
        b.vector(
            "compress_norm", (pooled,), norm_rows * width,
            NORM_VECTOR_OPS_PER_ELEMENT, family="rmsnorm",
            shape=(norm_rows, width),
            read_bytes_override=norm_rows * width * act_bytes
                                + width * dtype_bytes(plan.compressor_norm_dtype),
            write_bytes_override=norm_rows * width * act_bytes,
            metadata=(("reduction_width", width), ("norm_has_weight", True),
                      ("norm_weight_dtype", plan.compressor_norm_dtype),
                      ("norm_eps", model.norm_eps)),
        )
        if norm_rows > 0 else pooled
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
    if plan.mla_do_rope and (ratio == 2 or global_comp_rows > 0):
        rope_dim = plan.qk_rope_head_dim
        rotary_rows = global_latent_rows if ratio == 2 else global_comp_rows
        rotated = b.vector(
            "compress_rotary", (restored_latent,),
            rotary_rows * rope_dim,
            ROTARY_VECTOR_OPS_PER_ELEMENT, family="rotary",
            shape=(rotary_rows, width),
            read_bytes_override=rotary_rows * (
                rope_dim * act_bytes + rope_dim * FP32_BYTES * ROPE_TRIG_TABLES),
            write_bytes_override=rotary_rows * rope_dim * act_bytes,
        )
    if global_comp_rows <= 0:
        return None, restored_latent
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
    layer_dcp=1, store_rank_candidates=None,
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
                return_latent=True, fused=native_fusion,
                tokens_by_view=[v.q_len for v in views])
            storage_degrees = [
                min(store_dcp_degree(parallel, view.phase), max(1, int(layer_dcp)))
                for view in views
            ]
            long_store = representation_store(
                b, "compressed_store", (rotated,),
                _store_rows(
                    [view.q_len for view in views], views, parallel, own_block,
                    rank_candidates=store_rank_candidates,
                    storage_degrees=storage_degrees, ratio=ratio,
                ) if new_rows else 0,
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


def _plan_rank_attention_traffic(
    *, plan, model, view, parts, rec, eff_cp, worst_pairs_r, worst_sweeps_r,
    kv_row_b, query_tile, hierarchy, mode, topk, latent, interleave,
):
    """Evaluate exact selected-rank L2-aware HBM traffic with separate region byte widths (M1b)."""
    q_eff = rec["eff_q"]
    local_sweeps = rec["v_sweeps"][worst_sweeps_r]
    local_rows = rec["v_rows"][worst_pairs_r]
    by_rank = window_cache_read_by_rank(
        plan, view, parts, rec["v_sweeps"], rec["v_rows"], eff_cp,
        query_tile, plan.cache_row_elements, model, interleave,
    )
    if by_rank is not None:
        rank_idx = max(0, min(int(worst_sweeps_r), len(by_rank) - 1))
        w_sw, c_sw, w_u, c_u, w_rb, c_rb = by_rank[rank_idx]
        working_bytes = float(w_u) * float(w_rb) + float(c_u) * float(c_rb)
        logical_bytes = float(w_sw) * float(w_rb) + float(c_sw) * float(c_rb)
        effective_l2 = float(getattr(hierarchy, "effective_l2_bytes", 0.0) or 0.0)
        if working_bytes <= 0.0 and logical_bytes <= 0.0:
            hbm_kv = 0.0
            basis = "empty_rank_partition"
        elif topk:
            hbm_kv = logical_bytes
            basis = "unknown_dynamic_topk_per_query_gather"
        elif working_bytes <= effective_l2:
            hbm_kv = min(logical_bytes, working_bytes)
            basis = "dense_unique_working_set_resident"
        else:
            spill = max(0.0, 1.0 - min(working_bytes, effective_l2) / working_bytes)
            hbm_kv = min(logical_bytes, working_bytes + max(0.0, logical_bytes - working_bytes) * spill)
            basis = "capacity_spill_window"
        traffic = TrafficPlan(
            logical_read_bytes=logical_bytes,
            hbm_read_bytes=hbm_kv,
            hbm_write_bytes=0.0,
            l2_hit_bytes=max(0.0, logical_bytes - hbm_kv),
            private_refill_bytes=logical_bytes,
            index_bytes=0.0,
            scale_bytes=0.0,
            working_set_bytes=working_bytes,
            residency_basis=basis,
        )
        return traffic, traffic.hbm_read_bytes
    traffic = plan_attention_traffic(
        AttentionAccessSpec(
            q_lengths=(int(q_eff),),
            context_lengths=(int(max(0, view.kv_len - q_eff)),),
            unique_rows=int(max(0, local_rows)),
            refill_rows=int(max(0, local_sweeps)),
            selected_rows=int(max(0, local_sweeps)) if topk else 0,
            unique_rows_known=True,
            refill_rows_known=True,
            row_bytes=float(kv_row_b),
            phase=mode,
            topk=int(plan.index_topk) if topk else 0,
            representation=("absorbed_mla" if latent else "gqa"),
            cache_identity=str(getattr(view, "prefix_key", "") or ""),
        ),
        hierarchy,
        AttentionSchedule(query_tile=query_tile),
    )
    return traffic, traffic.hbm_read_bytes


def _append_mla_request(
    rec, grp_views, grp_dsa, eff_cp, worst_pairs_r, worst_sweeps_r, kv_row_b, *, b, runtime,
    parallel, quant, local_heads, dcp_heads, dp_width, qk_dim, latent_dim, latent,
    query_tile, hierarchy, topk, q_local, rotary_local, q_dcp, rotary_dcp,
    store, long_store, metadata_ready, machine, head_contract, offload_info,
    unabsorbed_qk_dim=None, unabsorbed_v_dim=None, q_unabsorbed=None,
    kv_b_dep=None, view_unabsorbed=False,
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
    eff_qk_dim = (unabsorbed_qk_dim if (view_unabsorbed and unabsorbed_qk_dim is not None) else qk_dim)
    eff_v_dim = (unabsorbed_v_dim if (view_unabsorbed and unabsorbed_v_dim is not None) else latent_dim)
    q_ready = (
        q_unabsorbed if (view_unabsorbed and q_unabsorbed is not None)
        else (q_dcp if eff_cp > 1 else q_local)
    )
    rotary = rotary_local if view_unabsorbed else (rotary_dcp if eff_cp > 1 else rotary_local)
    eff_kv_b = kv_b_dep if view_unabsorbed else None
    q_eff = eff_q
    q_read = q_eff * attn_heads * eff_qk_dim * dtype_bytes(model.activation_dtype)
    lse_write = (q_eff * attn_heads * FP32_BYTES) if view_unabsorbed else 0
    out_write = q_eff * attn_heads * eff_v_dim * dtype_bytes(model.activation_dtype) + lse_write
    interleave = int(getattr(runtime, "cp_interleave_size", DEFAULT_INTERLEAVE) or DEFAULT_INTERLEAVE)
    traffic, local_cache_read = _plan_rank_attention_traffic(
        plan=plan, model=model, view=view, parts=parts, rec=rec, eff_cp=eff_cp,
        worst_pairs_r=worst_pairs_r, worst_sweeps_r=worst_sweeps_r,
        kv_row_b=kv_row_b, query_tile=query_tile, hierarchy=hierarchy,
        mode=mode, topk=topk, latent=(latent and not view_unabsorbed),
        interleave=interleave,
    )
    vis_keys = (
        local_rows if plan.window > 0
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
    tensor_view_meta = (
        (("mla_attention_mode", "unabsorbed_prefill"), ("lse_write_bytes", float(lse_write)))
        if view_unabsorbed
        else attention_tensor_view(plan, view, attn_heads, model, quant,
                                   parallel, eff_cp=eff_cp, machine=machine)
    )
    last = b.work(
        f"latent_attention{orig_idx}", tuple(dict.fromkeys(
            item for item in (q_ready, rotary if latent else None, eff_kv_b, store,
                              long_store, topk, metadata_ready) if item)),
        (q_eff, view.kv_len, attn_heads, eff_qk_dim, eff_v_dim),
        flops=2 * local_pairs * attn_heads * (eff_qk_dim + eff_v_dim),
        vector=SOFTMAX_VECTOR_OPS_PER_PAIR * local_pairs * attn_heads,
        read=kv_mte2_component + rec["selection_reads"] + q_read,
        hbm_read=kv_hbm_component + rec["selection_reads"] + q_read,
        write=out_write, family="attention",
        metadata=(("query_tile", query_tile), ("logical_pairs", pairs),
                  ("kv_sweeps", sweeps),
                  ("attention_traffic_plan", traffic.to_dict()))
                 + rec["selection_metadata"]
                 + tensor_view_meta
                 + ((("program_kind", "dense_latent_attention_generic"),)
                    if is_generic_decode else ())
                 + ((("latent_attention_schedule", {
                     "heads": attn_heads, "query_rows": q_eff,
                     "keys": max(0, vis_keys),
                     "qk_dim": eff_qk_dim, "latent_dim": eff_v_dim,
                     "dv": int(eff_v_dim),
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
        local_shape=(q_eff, local_rows, attn_heads, eff_qk_dim, eff_v_dim),
    )
    fused_sfa = (
        offload_info.get("fused_sfa_recipe")
        if (offload_info and view.phase == "decode") else None
    )
    if fused_sfa:
        req_scope = (view.request_id,) if view.request_id else b.ops[-1].requests
        b.ops[-1] = replace(
            b.ops[-1],
            requests=req_scope,
            recipe=fused_sfa,
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
    if runtime and topk and plan.cache_kind == "dense" and not fused_sfa:
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
    *, peer_rows=None, local_dp_pos=0, cp_interleave_size=None,
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
                peer_rows=peer_rows, local_dp_pos=local_dp_pos,
                interleave=cp_interleave_size)
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


def _attach_mla_dcp_metadata(
    b, grp_outputs, paired_sfa_mode, rank_pairs, rank_sweeps,
    worst_pairs_r, worst_sweeps_r, mtp_split, sparse_phase,
):
    """Attach per-rank DCP balance and sparse straggler skew metadata."""
    out_set = ({b.prefix + ".latent_attention_batch"} if paired_sfa_mode
               else set(grp_outputs))
    eff_cp = len(rank_pairs)
    balanced = sum(rank_pairs) / float(eff_cp) if eff_cp > 0 else 0.0
    cp_sparse_imb = (
        float(rank_pairs[worst_pairs_r]) / balanced
        if (sparse_phase and balanced > 0.0)
        else 1.0
    )
    sparse_meta = (("cp_sparse_imbalance", float(cp_sparse_imb)),) if sparse_phase else ()
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
            ) + sparse_meta)


def _mla_view_kv_row_bytes(model, plan, view_phase: str) -> int:
    return packed_region_row_bytes(
        model, "mla_kv", plan.cache_row_elements, plan=plan, phase=view_phase)


def _group_mla_attention_views(
    *, views, selected, model, plan, parallel, local_heads, dcp_heads,
    unabsorbed_qk_dim, unabsorbed_v_dim, layer_dcp, has_dp_decode, cohort_decode_views,
    sfa_replicated_prefill_path, sharded_indexer, has_unabsorbed_prefill,
):
    phase_groups: list[tuple[int, int, bool, list[tuple[int, object]]]] = []
    grp_map: dict[tuple[int, int, bool], list[tuple[int, object]]] = {}
    for index, view in enumerate(() if selected else views):
        if has_dp_decode and view.phase != "prefill":
            continue
        eff_cp = (1 if sfa_replicated_prefill_path else
                  active_layer_dcp(plan, parallel, view.phase, sharded_indexer=sharded_indexer))
        view_unabsorbed = bool(has_unabsorbed_prefill and _is_fresh_dense_prefill_view(view))
        view_kv_row_b = (
            (dcp_heads if eff_cp > 1 else local_heads) * (unabsorbed_qk_dim + unabsorbed_v_dim)
            * int(dtype_bytes(model.activation_dtype))
            if view_unabsorbed
            else _mla_view_kv_row_bytes(model, plan, view.phase)
        )
        grp_key = (eff_cp, view_kv_row_b, view_unabsorbed)
        if grp_key not in grp_map:
            bucket: list[tuple[int, object]] = []
            grp_map[grp_key] = bucket
            phase_groups.append((eff_cp, grp_key[1], view_unabsorbed, bucket))
        grp_map[grp_key].append((index, view))
    if not selected and has_dp_decode:
        next_view_idx = max(
            (idx for _, _, _, bucket in phase_groups for idx, _ in bucket),
            default=-1,
        ) + 1
        decode_row_b = _mla_view_kv_row_bytes(model, plan, "decode")
        phase_groups.append(
            (layer_dcp, decode_row_b, False, list(enumerate(cohort_decode_views, start=next_view_idx)))
        )
    return phase_groups


def _append_mla_attention(
    b, outputs, views, runtime, selected, parallel, dsa_prefill, model, plan, quant, query_tile,
    cp_interleave_size, topk, local_heads, dcp_heads, latent_dim, qk_dim, reuse_index_selection,
    request_op, mixed_quant, pcp_prefill, paired_sfa_mode, is_sparse, pcp_factor, tp_gather,
    dp_width, policy, logical_heads, q, hierarchy, stage, mode, q_local, store,
    *, cohort_decode_views=None, peer_decode_rows=None, local_dp_pos=0,
    idle_dcp_owner=False, sharded_indexer=False, physical_rank=None,
    explicit_dp_cohort=False, explicit_physical_rank=False,
    sfa_replicated_prefill_path=False, has_unabsorbed_prefill=False,
    unabsorbed_qk_dim=None, unabsorbed_v_dim=None,
) -> set[str]:
    block_size = max(1, int(getattr(runtime, "kv_block_size", DEFAULT_BLOCK_SIZE)
                              or DEFAULT_BLOCK_SIZE))
    layer_dcp = layer_dcp_degree(plan, parallel)
    has_dp_decode = (not sfa_replicated_prefill_path and dp_width > 1
                     and layer_dcp > 1 and bool(cohort_decode_views))
    eff_unabs_qk = unabsorbed_qk_dim if unabsorbed_qk_dim is not None else qk_dim
    eff_unabs_v = unabsorbed_v_dim if unabsorbed_v_dim is not None else latent_dim
    phase_groups = _group_mla_attention_views(
        views=views, selected=selected, model=model, plan=plan, parallel=parallel,
        local_heads=local_heads, dcp_heads=dcp_heads,
        unabsorbed_qk_dim=eff_unabs_qk, unabsorbed_v_dim=eff_unabs_v,
        layer_dcp=layer_dcp, has_dp_decode=has_dp_decode,
        cohort_decode_views=cohort_decode_views,
        sfa_replicated_prefill_path=sfa_replicated_prefill_path,
        sharded_indexer=sharded_indexer,
        has_unabsorbed_prefill=has_unabsorbed_prefill,
    )
    multi_group = len(phase_groups) > 1
    unabsorbed_outputs: set[str] = set()
    for grp_idx, (eff_cp, kv_row_b, grp_unabsorbed, indexed_views) in enumerate(phase_groups):
        grp_views = tuple(view for _, view in indexed_views)
        grp_prefix = f"{b.prefix}.phase{grp_idx}" if multi_group else b.prefix
        grp_dsa = dsa_prefill and any(view.phase == "prefill" for view in grp_views)
        mtp_split = should_split_mtp_dcp(plan, grp_views, layer_dcp=eff_cp, sparse=is_sparse)
        grp_heads = dcp_heads if eff_cp > 1 else local_heads
        grp_v_dim = eff_unabs_v if grp_unabsorbed else latent_dim
        grp_qk_dim = eff_unabs_qk if grp_unabsorbed else qk_dim
        rank_pairs = [0.0] * eff_cp
        rank_sweeps = [0] * eff_cp
        rank_rows = [0] * eff_cp
        view_records: list[dict] = []
        sparse_phase = bool(topk) or (idle_dcp_owner and is_sparse)
        for orig_idx, view in indexed_views:
            rec = attention_view_record(
                view, plan=plan, model=model, quant=quant, query_tile=query_tile,
                eff_cp=eff_cp, block=block_size, interleave=cp_interleave_size,
                tp=parallel.tp, dsa=grp_dsa,
                topk_present=sparse_phase,
                heads=grp_heads, latent_dim=grp_v_dim,
                shared_selection=(topk if (reuse_index_selection
                                           and isinstance(topk, SharedSelectionRef))
                                  else None),
                runtime=runtime)
            if mtp_split:
                hist = _history_only_pairs(view, eff_cp, cp_interleave_size, int(plan.window or 0))
                if hist is not None:
                    rec = {**rec, "v_pairs": hist, "pairs": float(sum(hist))}
            view_records.append({"orig_idx": orig_idx, **rec})
        if eff_cp > 1 and sparse_phase:
            pool_sparse_view_records(
                view_records, eff_cp, cp_interleave_size, runtime=runtime)
        rank_region_bytes = [0.0] * eff_cp
        for rec in view_records:
            for rank in range(eff_cp):
                rank_pairs[rank] += rec["v_pairs"][rank]
                rank_sweeps[rank] += rec["v_sweeps"][rank]
                rank_rows[rank] += rec["v_rows"][rank]
            if plan.cache_kind == "window_compressed":
                by_r = window_cache_read_by_rank(
                    plan, rec["view"], rec["parts"], rec["v_sweeps"], rec["v_rows"],
                    eff_cp, query_tile, plan.cache_row_elements, model, cp_interleave_size,
                )
                if by_r is not None:
                    for rank, (w_sw, c_sw, w_u, c_u, w_rb, c_rb) in enumerate(by_r):
                        rank_region_bytes[rank] += (
                            float(w_sw) * float(w_rb) + float(c_sw) * float(c_rb)
                            + float(w_u) * float(w_rb) + float(c_u) * float(c_rb)
                        )
        worst_pairs_r = max(range(eff_cp),
                            key=lambda rank: (rank_pairs[rank], rank_sweeps[rank]))
        worst_sweeps_r = max(range(eff_cp),
                             key=lambda rank: (rank_region_bytes[rank], rank_sweeps[rank], rank_pairs[rank]))
        cand_ranks = resolve_cp_rank_candidates(
            parallel, eff_cp=eff_cp, pcp_factor=pcp_factor, tp_gather=tp_gather,
            dp_width=dp_width, local_dp_pos=local_dp_pos, physical_rank=physical_rank,
            explicit_dp_cohort=explicit_dp_cohort,
            explicit_physical_rank=explicit_physical_rank)
        sel_pairs_r = max(cand_ranks, key=lambda rank: (rank_pairs[rank], rank_sweeps[rank]))
        sel_sweeps_r = max(
            cand_ranks,
            key=lambda rank: (rank_region_bytes[rank], rank_sweeps[rank], rank_pairs[rank]),
        )
        grp_outputs: list[str] = []
        sweeps_by_view: list[int] = []
        shared_by_view: list[int] = []
        kv_hbm_total = 0.0
        kv_mte2_total = 0.0
        kv_components: dict[str, tuple[float, float]] = {}
        for rec in view_records:
            local_sweeps = rec["v_sweeps"][sel_sweeps_r]
            sweeps_by_view.append(local_sweeps)
            shared_by_view.append(min(local_sweeps, rec["v_shared"][sel_sweeps_r]))
            last, kv_hbm_component, kv_mte2_component = request_op(
                rec, grp_views, grp_dsa, eff_cp, sel_pairs_r, sel_sweeps_r, kv_row_b,
                view_unabsorbed=grp_unabsorbed)
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
            local_dp_pos=local_dp_pos, cp_interleave_size=cp_interleave_size)
        if eff_cp > 1 and grp_outputs:
            _attach_mla_dcp_metadata(
                b, grp_outputs, paired_sfa_mode, rank_pairs, rank_sweeps,
                worst_pairs_r, worst_sweeps_r, mtp_split, sparse_phase)
        if eff_cp > 1 and grp_outputs and not paired_sfa_mode:
            current_dep = None
            if mtp_split:
                current_dep = emit_mtp_current_attention(
                    b.ops, prefix=grp_prefix, stage=stage, layer=plan.index,
                    requests=b.common["requests"], mode=mode,
                    deps=tuple(dict.fromkeys((*grp_outputs, q_local, store))),
                    views=grp_views, local_heads=logical_heads, qk_dim=grp_qk_dim,
                    v_dim=grp_v_dim, kv_row_bytes=kv_row_b,
                    act_dtype=model.activation_dtype)
            grp_rows = max(1, sum(v.q_len for v in grp_views) if multi_group else max(0, q))
            merged_last = emit_dcp_output_exchange_combine(
                b.ops, prefix=grp_prefix, stage=stage, layer=plan.index,
                requests=b.common["requests"], mode=mode,
                deps=tuple(grp_outputs), rows=grp_rows, local_heads=logical_heads,
                v_dim=grp_v_dim, act_dtype=model.activation_dtype,
                pcp_factor=pcp_factor, tp_gather=tp_gather, dp_width=dp_width,
                local_current_dep=current_dep,
                peer_rows=(peer_decode_rows if dp_width > 1 else None),
                local_dp_pos=local_dp_pos,
                meta_extra=(("cp_interleave", int(cp_interleave_size)),))
            grp_outputs = [merged_last]
        if grp_unabsorbed:
            unabsorbed_outputs.update(grp_outputs)
        outputs.extend(grp_outputs)
    return unabsorbed_outputs


def _project_mla_output(
    b, logical_heads, latent, outputs, q, plan, head_contract, deps, h, partial_rotary,
    parallel, model, dsa_prefill, *, views=(), runtime=None,
    has_unabsorbed_prefill=False, unabsorbed_outputs=(),
):
    heads = logical_heads
    unabsorbed_set = set(unabsorbed_outputs)
    unabsorbed_list = [op_id for op_id in outputs if op_id in unabsorbed_set]
    absorbed_list = [op_id for op_id in outputs if op_id not in unabsorbed_set]
    has_local_absorbed = any(not _is_fresh_dense_prefill_view(v) for v in views)
    unabsorbed_rows = (
        sum(int(v.q_len) for v in views if _is_fresh_dense_prefill_view(v))
        if has_unabsorbed_prefill else 0
    )
    v_expand_rows = (
        max(0, q - unabsorbed_rows)
        if (has_unabsorbed_prefill and has_local_absorbed)
        else (0 if has_unabsorbed_prefill else q)
    )
    if latent and absorbed_list and v_expand_rows > 0:
        expanded = b.matrix("v_expand", absorbed_list, v_expand_rows * heads, plan.v_head_dim,
                            plan.kv_lora_rank, weight_count=heads,
                            category="mla_absorb",
                            metadata=batched_matrix_metadata(
                                tokens=v_expand_rows, heads=heads, input_width=plan.kv_lora_rank,
                                output_width=plan.v_head_dim, role="value_expansion")
                            + mla_head_padding.output_metadata(head_contract))
        outputs = [*unabsorbed_list, expanded]
    output_width = heads * (plan.v_head_dim if latent else plan.head_dim)
    if plan.mla_output_gate and latent:
        gate = b.matrix("output_gate", (deps,), q, output_width, h, category="mla.g")
        outputs = [b.vector("gated_output", (*outputs, gate), q * output_width,
                         GATE_APPLY_OPS_PER_ELEMENT)]
    if partial_rotary and plan.mla_do_rope:
        outputs = [partial_rotary.module.declare(
            b, name="output_inverse_rotary", deps=outputs, tokens=q, heads=heads,
            head_dim=plan.head_dim, rotary_width=plan.qk_rope_head_dim, inverse=True)]
    elif (plan.mla_variant == "latent_mqa"
            and plan.cache_kind == "window_compressed"
            and plan.mla_do_rope and plan.qk_rope_head_dim > 0):
        rope_elems = q * heads * plan.qk_rope_head_dim
        rope_bytes = rope_elems * dtype_bytes(model.activation_dtype)
        outputs = [b.work(
            "output_inverse_rotary", tuple(outputs), (q, heads, plan.qk_rope_head_dim),
            vector=rope_elems * ROPE_OPS, read=rope_bytes, write=rope_bytes,
            family="vector", dtype=model.activation_dtype, recipe="output_inverse_rotary",
            metadata=(("rotary_dim", plan.qk_rope_head_dim),
                      ("head_dim", plan.head_dim), ("inverse", True),
                      ("native_kind", "InplacePartialRotaryMul"),
                      ("rotary_role", "output_inverse_rotary")))]
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
        if not group_parallel and parallel.tp > 1:
            tail = b.ops[-1]
            b.ops[-1] = replace(
                tail,
                metadata=tail.metadata + (("tp_output_placement", "replicated"),),
            )
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
