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
from typing import Literal

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
    layer_dcp_degree,
    layer_dcp_factors,
    owned_positions,
    round_sparse_dcp_matrix,
    sparse_dcp_capacities,
    sparse_dcp_sequence_for_total,
    split_causal_pairs,
    split_sparse_dcp_batch_per_sequence,
)
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
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
    gqa_kv_memory_accesses,
    _declare_ordinary_qk_norm_rope,
    _declare_qkv_norm_rope,
    _gqa_formats,
    _memory_hierarchy,
    _prefill_contract,
    _qkv_output_copy,
    _qkv_views,
    _store_cache,
    uses_mrope_qkv_kernel,
)
from mindie_sim.prediction.operators.sparse_indexer.gqa_indexer import (
    is_block_msa_indexer,
    is_qsa_pooled_indexer,
    is_region_csa_indexer,
    lower_gqa_indexer,
)
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
GBSA_NATIVE_KIND = "aclnnGenericBlockSparseAttention"
GBSA_META_NATIVE_KIND = "GenericBlockSparseAttentionMetadata"
FIA_PREFILL_NATIVE_KIND = "FusedInferAttentionScore"
QSA_FIA_MAX_LOGICAL_KV_LENGTH = 24 * 1024
GBSA_METADATA_REUSE_ARCHS = frozenset({"A3", "A5", "DAV_3510", "NPU_ARCH_3510"})


def _qsa_view_uses_fia(view) -> bool:
    """Upstream ``qwen4_exp/attention_dispatch.py:27-46`` request-level FIA predicate."""
    if getattr(view, "padding", False) or str(getattr(view, "phase", "")) != "prefill":
        return False
    # Classify the scheduler's whole chunk (invariant across PCP shards).
    q_len = max(0, int(getattr(view, "global_q_len", 0) or getattr(view, "q_len", 0) or 0))
    context = max(0, int(getattr(view, "context", 0) or 0))
    kv_len = max(q_len, context + q_len, int(getattr(view, "kv_len", 0) or 0))
    fits = kv_len <= QSA_FIA_MAX_LOGICAL_KV_LENGTH
    crosses = (
        kv_len > QSA_FIA_MAX_LOGICAL_KV_LENGTH
        and (kv_len - q_len) < QSA_FIA_MAX_LOGICAL_KV_LENGTH
    )
    return fits or crosses


def _supports_gbsa_metadata_reuse(runtime, machine) -> bool:
    """Upstream ``hardware_profile.py:370,394,487``: A3 and A5 support reuse; A2 does not."""
    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    for candidate in (
        getattr(machine, "arch_family", None),
        rt_meta.get("arch_family"),
        getattr(runtime, "arch_family", None),
    ):
        text = str(candidate or "").strip().upper()
        if not text:
            continue
        if "A2" in text:
            return False
        if text in GBSA_METADATA_REUSE_ARCHS or any(tok in text for tok in ("A3", "A5", "3510", "950")):
            return True
        return False
    return False


def _sparse_gqa_native_kind(plan, *, mode: str, eff_topk: int, topk_id: str | None, view=None) -> str:
    if eff_topk <= 0 or not topk_id:
        return ""
    if is_block_msa_indexer(plan) or is_region_csa_indexer(plan):
        return GBSA_NATIVE_KIND
    if is_qsa_pooled_indexer(plan):
        if view is not None:
            return FIA_PREFILL_NATIVE_KIND if _qsa_view_uses_fia(view) else GBSA_NATIVE_KIND
        return FIA_PREFILL_NATIVE_KIND if mode == "prefill" else GBSA_NATIVE_KIND
    return ""


GBSA_METADATA_TOTAL_WORDS = 1024
GBSA_METADATA_WORD_BYTES = 4
GBSA_METADATA_OUTPUT_BYTES = GBSA_METADATA_TOTAL_WORDS * GBSA_METADATA_WORD_BYTES
INT64_BYTES = 8


def _emit_gbsa_metadata_if_needed(
    ops, *, prefix: str, stage: int, plan, requests, mode: str,
    views, topk_id: str | None, eff_topk: int, sel_topk: int, selected,
    local_heads: int, kv_heads: int, head_dim: int, padded_tokens: int,
    quant, runtime=None, machine=None, gbsa_metadata_cache: dict | None = None,
) -> str | None:
    """Emit AI_CPU ``GenericBlockSparseAttentionMetadata`` (fixed 1024 int32 words = 4,096 B).

    Mirrors CANN ``generic_block_sparse_attention_metadata_tiling.h:21``,
    upstream ``vllm_ascend/models/minimax_m3/msa_m3.py:838-927`` stage+hardware-gated
    decode reuse, and ``qwen4_exp/attention_dispatch.py`` request-level QSA dispatch.
    """
    if selected or eff_topk <= 0 or not topk_id:
        return None
    if is_qsa_pooled_indexer(plan):
        non_pad_all = tuple(v for v in views if not getattr(v, "padding", False))
        gbsa_non_pad = tuple(v for v in non_pad_all if not _qsa_view_uses_fia(v))
        if not gbsa_non_pad:
            return None
        meta_views = views if len(gbsa_non_pad) == len(non_pad_all) else gbsa_non_pad
        eff_padded = padded_tokens if len(meta_views) == len(views) else sum(int(v.physical_q_len) for v in meta_views)
    elif is_block_msa_indexer(plan) or is_region_csa_indexer(plan):
        meta_views = views
        eff_padded = padded_tokens
    else:
        return None
    non_pad = tuple(v for v in meta_views if not getattr(v, "padding", False))
    q_lens = {int(v.q_len) for v in non_pad}
    uniform_decode = (
        mode == "decode"
        and bool(non_pad)
        and len(q_lens) <= 1
        and all(v.phase == "decode" for v in non_pad)
    )
    batch = max(1, len(meta_views))
    phys_q = max(sum(int(v.physical_q_len) for v in meta_views), int(eff_padded))
    max_q = max((int(v.q_len) for v in non_pad), default=1)
    raw_kv_dtype = getattr(quant, "kv_dtype", "") if quant is not None else ""
    if not raw_kv_dtype:
        raise ModelingError("gbsa_metadata_requires_explicit_kv_dtype")
    kv_dtype = str(raw_kv_dtype)
    can_reuse = (
        gbsa_metadata_cache is not None
        and uniform_decode
        and is_block_msa_indexer(plan)
        and _supports_gbsa_metadata_reuse(runtime, machine)
    )
    reuse_key = (
        (int(stage), local_heads, kv_heads, head_dim, sel_topk, max_q, phys_q, kv_dtype)
        if can_reuse
        else None
    )
    if reuse_key is not None and reuse_key in gbsa_metadata_cache:
        return gbsa_metadata_cache[reuse_key]
    meta_reqs = tuple(v.request_id for v in meta_views) if meta_views else requests
    # Upstream msa_m3_npu.py:193-258 inputs: select_num_idx[H_kv, M_phys] (int32),
    # cu_seqlens_q[B+1] (int64), seqused_kv[B] (int32). Output: fixed 1024 int32 words.
    input_read_bytes = (
        kv_heads * phys_q * GBSA_METADATA_WORD_BYTES
        + (batch + 1) * INT64_BYTES
        + batch * GBSA_METADATA_WORD_BYTES
    )
    output_write_bytes = GBSA_METADATA_OUTPUT_BYTES
    call_spec = {
        "batch": batch,
        "rows": phys_q,
        "valid_rows": sum(int(v.q_len) for v in non_pad),
        "heads": local_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "topk": sel_topk,
        "max_query": max_q,
        "max_context": max((int(v.kv_len) for v in non_pad), default=0),
        "input_read_bytes": input_read_bytes,
        "output_write_bytes": output_write_bytes,
        "output_shape": (GBSA_METADATA_TOTAL_WORDS,),
        "output_dtype": "int32",
        "query_layout": "TND",
        "cache_layout": "PA_BBND",
        "decode_reusable": reuse_key is not None,
    }
    meta_op = OpSpec(
        op_id=f"{prefix}.gbsa_metadata",
        family="attention_metadata",
        phase=mode,
        stage=stage,
        layer=plan.index,
        requests=meta_reqs,
        semantic_shape=(GBSA_METADATA_TOTAL_WORDS,),
        local_shape=(GBSA_METADATA_TOTAL_WORDS,),
        padded_shape=(GBSA_METADATA_TOTAL_WORDS,),
        launch_shape=(GBSA_METADATA_TOTAL_WORDS,),
        flops=0.0,
        vector_ops=0.0,
        hbm_bytes=float(input_read_bytes + output_write_bytes),
        mte2_bytes=float(input_read_bytes),
        mte3_bytes=float(output_write_bytes),
        deps=(topk_id,) if topk_id else (),
        recipe="generic_block_sparse_attention_metadata",
        dtype="int32",
        source_boundary="kernel",
        metadata=(
            ("kernel", GBSA_META_NATIVE_KIND),
            ("native_kind", GBSA_META_NATIVE_KIND),
            ("execution_core", "aicpu"),
            ("invocation_class", "aicpu"),
            ("execution_domain", "AI_CPU"),
            ("metadata_call", call_spec),
        ),
    )
    ops.append(meta_op)
    ready_id = meta_op.op_id
    if reuse_key is not None:
        publish = vector_op(
            op_id=f"{prefix}.gbsa_metadata_publish",
            deps=(meta_op.op_id,),
            stage=stage,
            layer=plan.index,
            requests=requests,
            elements=GBSA_METADATA_TOTAL_WORDS,
            ops_per_element=0,
            dtype="int32",
            family="copy",
            phase=mode,
            recipe="copy",
            shape=(GBSA_METADATA_TOTAL_WORDS,),
            read_bytes_override=float(output_write_bytes),
            write_bytes_override=float(output_write_bytes),
            metadata=(("metadata_buffer_publication", True), ("decode_reused", True)),
        )
        ops.append(publish)
        ready_id = publish.op_id
        gbsa_metadata_cache[reuse_key] = ready_id
    return ready_id
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


DRAFT_SCOPES = frozenset({"draft", "mtp", "dspark", "standalone_draft"})


def _draft_kv_cap(plan, model, runtime, mode: str) -> int:
    """Explicit draft KV length cap (never reads target max_position_embeddings)."""
    if mode not in {"draft", "draft_prefill"} and getattr(plan, "scope", "") not in DRAFT_SCOPES:
        return 0
    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    rt_extra = dict(rt_meta.get("additional_config") or ()) if isinstance(rt_meta.get("additional_config"), dict) else {}
    model_meta = dict(getattr(model, "metadata", ()) or ()) if model is not None else {}
    candidates = [
        int(getattr(plan, attr, 0) or 0)
        for attr in ("max_model_len", "draft_max_model_len", "max_position_embeddings")
    ]
    for source in (rt_meta, rt_extra, model_meta):
        raw = source.get("draft_max_model_len")
        if raw is not None:
            candidates.append(int(raw or 0))
    positive = [val for val in candidates if val > 0]
    return min(positive) if positive else 0


def _cap_draft_views(views, cap: int):
    """Bound draft view contexts so draft attention never attends past its KV cap."""
    if cap <= 0 or not views:
        return views
    capped = []
    for view in views:
        if view.kv_len <= cap:
            capped.append(view)
            continue
        max_ctx = max(0, cap - int(view.q_len))
        new_ctx = min(int(view.context), max_ctx)
        new_pref = min(int(getattr(view, "prefix_tokens", 0) or 0), new_ctx)
        capped.append(replace(view, context=new_ctx, prefix_tokens=new_pref))
    return tuple(capped)


def _resolve_gqa_recipe(cacheless: bool, mode: str, configured_recipes) -> str:
    if cacheless:
        return "flash_attention_prefill" if "flash_attention_prefill" in configured_recipes else "flash_attention"
    if mode in {"decode", "draft"} and "flash_attention_decode" in configured_recipes:
        return "flash_attention_decode"
    if mode in {"prefill", "draft_prefill"} and "flash_attention_prefill" in configured_recipes:
        return "flash_attention_prefill"
    return "flash_attention"


TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/attention/attention_v1.py",
        source_sha256="1194f995eccb90309140d2e19e6dcc2291548cd58d8523c24a5ee1c79836360a",
    ),
)
_SPARSE_FLOP_TOLERANCE: float = 1e-6


def _selected_supports_sparse_selection(selected) -> bool:
    """True only when a custom gqa_attention lowering explicitly consumes sparse TopK."""
    module = getattr(selected, "module", None)
    if bool(getattr(module, "SUPPORTS_SPARSE_SELECTION", False)):
        return True
    caps = getattr(module, "CAPABILITIES", ()) or ()
    return "block_sparse_attention" in caps


def gqa(*, prefix, plan, views, stage, deps, model, parallel, quant, mode,
        padded_tokens, policy=AttentionPolicy(), cp_interleave_size=DEFAULT_CP_INTERLEAVE,
        runtime=None, cache_ids=None, stage_of=None, topk_ids=None,
        index_cache_ids=None, candidate_ids=None, machine=None, hierarchy=None,
        dcp_cohort_views_by_member=None, dcp_local_pos=0,
        physical_rank=None, explicit_physical_rank=False,
        gbsa_metadata_cache=None, **_) -> AttentionResult:
    from mindie_sim.prediction.implementations import add_input_ports, select

    cacheless = plan.cache_kind == "none"
    if cacheless:
        views = tuple(replace(v, context=0, prefix_tokens=0,
                              prefix_cache_key="") for v in views)
        quant = replace(quant, kv_dtype=model.activation_dtype)
    draft_cap = _draft_kv_cap(plan, model, runtime, mode)
    if draft_cap > 0 and not cacheless:
        views = _cap_draft_views(views, draft_cap)
    selected = select(runtime.kernels.get("lowerings", {}), "gqa_attention") if runtime else None
    if selected:
        if int(getattr(plan, "index_topk", 0) or 0) > 0 and not _selected_supports_sparse_selection(selected):
            raise ModelingError(
                f"gqa_attention_lowering_ignores_sparse_selection:{getattr(selected, 'name', selected.module)}"
            )
        selected.module.validate(views=views, plan=plan, parallel=parallel, policy=policy,
                                 quant=quant, model=model, **selected.options)
    heads = plan.num_heads // parallel.tp
    kv_heads = local_kv_heads(plan.num_kv_heads, parallel.tp)
    width = plan.head_dim
    v_width = int(getattr(plan, "v_head_dim", 0) or width)
    key_fmt, value_fmt, cache_regions, shared_format = _gqa_formats(model, plan=plan, phase=mode)
    if cacheless:
        key_fmt = value_fmt = CacheFormat(model.activation_dtype)
        shared_format = True
    shared = not cacheless and 0 <= plan.kv_owner_layer_id < plan.index
    # Shipping vLLM/vLLM-Ascend clones checkpoint K into the runtime V slot for
    # ``k_eq_v`` layers (e.g. Gemma-4 full attention), so QKV projects both K
    # and V planes (``(heads + 2 * kv_heads) * width``) into distinct cache planes.
    projection_planes = 2
    cache_planes = 2
    projected_kv_heads = 0 if shared else kv_heads
    configured_recipes = runtime.kernels.get("recipes", {}) if runtime else {}
    attention_recipe = _resolve_gqa_recipe(cacheless, mode, configured_recipes)
    pcp_factor, tp_gather, raw_dp_width = layer_dcp_factors(plan, parallel)
    dp_width = raw_dp_width if int(getattr(parallel, "dp", 1) or 1) >= raw_dp_width else 1
    layer_dcp = layer_dcp_degree(plan, parallel)
    cohort_dec_views, peer_dec_rows, local_dp_pos = resolve_dcp_cohort_decode(
        views, dp_width=dp_width,
        cohort_views_by_member=dcp_cohort_views_by_member,
        local_dp_pos=dcp_local_pos)
    if draft_cap > 0 and not cacheless and cohort_dec_views:
        cohort_dec_views = _cap_draft_views(cohort_dec_views, draft_cap)
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
            ops, prefix=prefix, plan=plan, views=views, stage=stage, deps=deps,
            model=model, parallel=parallel, quant=quant, mode=mode, padded_tokens=padded_tokens,
            valid=valid, pcp_prefill=pcp_prefill, runtime=runtime, topk_ids=topk_ids,
            index_cache_ids=index_cache_ids, machine=machine,
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
    gbsa_meta_id = _emit_gbsa_metadata_if_needed(
        ops, prefix=prefix, stage=stage, plan=plan, requests=requests, mode=mode,
        views=views, topk_id=topk_id, eff_topk=eff_topk, sel_topk=sel_topk,
        selected=selected, local_heads=heads, kv_heads=kv_heads, head_dim=width,
        padded_tokens=padded_tokens, quant=quant,
        runtime=runtime, machine=machine,
        gbsa_metadata_cache=gbsa_metadata_cache,
    )
    sparse_deps = tuple(dict.fromkeys(dep for dep in (topk_id, gbsa_meta_id) if dep))
    fia_sparse_deps = (topk_id,) if topk_id else ()
    local_attn_deps = (*store_ids, *sparse_deps, query_producer)
    dcp_attn_deps = (*store_ids, *sparse_deps, dcp_query_producer)
    local_fia_attn_deps = (*store_ids, *fia_sparse_deps, query_producer)
    dcp_fia_attn_deps = (*store_ids, *fia_sparse_deps, dcp_query_producer)
    streams = kv_heads if policy.reuse_kv_across_query_heads else heads
    kv_row_bytes = (
        key_fmt.row_bytes(streams * (width + v_width))
        if shared_format
        else key_fmt.row_bytes(streams * width) + value_fmt.row_bytes(streams * v_width)
    )
    user_pinned_attn = (
        is_caller_pinned_binding(runtime, "families", "attention")
        or is_caller_pinned_binding(runtime, "recipes", attention_recipe)
        or is_caller_pinned_binding(runtime, "recipes", "flash_attention")
    )
    qsa_dispatch = bool(is_qsa_pooled_indexer(plan) and eff_topk > 0 and topk_id and not selected)
    request_op = partial(
        _gqa_request_op, kv_heads=kv_heads, local_heads=heads, dcp_heads=dcp_heads,
        dp_width=dp_width, width=width,
        kv_row_bytes=kv_row_bytes, plan=plan, mode=mode, hierarchy=hierarchy,
        policy=policy, model=model, sel_topk=sel_topk, topk_id=topk_id,
        eff_topk=eff_topk, prefix=prefix, stage=stage,
        local_attn_deps=local_attn_deps, dcp_attn_deps=dcp_attn_deps,
        local_fia_attn_deps=local_fia_attn_deps, dcp_fia_attn_deps=dcp_fia_attn_deps,
        attention_recipe=attention_recipe, streams=streams, quant=quant,
        parallel=parallel, eff_window=eff_window, dense_win=dense_win,
        runtime=runtime, cacheless=cacheless,
        selected=selected, query_producer=query_producer,
        dcp_query_producer=dcp_query_producer, add_input_ports=add_input_ports,
        key_fmt=key_fmt, value_fmt=value_fmt, shared_format=shared_format,
        cp_interleave_size=cp_interleave_size, block_size=block_size)
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
        explicit_physical_rank=bool(explicit_physical_rank),
        qsa_dispatch=qsa_dispatch,
        key_fmt=key_fmt, value_fmt=value_fmt, shared_format=shared_format)
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


def _emit_ordinary_qk_norm(
    prefix, plan, stage, ready, model, mode, padded_tokens, requests,
    heads, projected_kv_heads, width, shared, split_views,
):
    """Emit separate Q-norm and K-norm for ordinary text RoPE (zero-copy V)."""
    act_bytes = dtype_bytes(model.activation_dtype)
    q_norm = vector_op(
        op_id=prefix + ".q_norm", deps=(ready,), stage=stage, layer=plan.index,
        requests=requests, elements=padded_tokens * heads * width,
        ops_per_element=RMS_NORM_OPS_PER_ELEMENT, dtype=model.activation_dtype,
        family="rmsnorm", phase=mode, recipe="q_norm",
        shape=(padded_tokens, heads, width),
        read_bytes_override=padded_tokens * heads * width * act_bytes + width * act_bytes,
        write_bytes_override=padded_tokens * heads * width * act_bytes,
        metadata=(("reduction_width", width), ("norm_has_weight", True),
                  ("input_views", {"query": split_views["query"].to_dict()})),
    )
    emitted = [q_norm]
    if not shared and projected_kv_heads > 0:
        k_norm = vector_op(
            op_id=prefix + ".k_norm", deps=(ready,), stage=stage, layer=plan.index,
            requests=requests, elements=padded_tokens * projected_kv_heads * width,
            ops_per_element=RMS_NORM_OPS_PER_ELEMENT, dtype=model.activation_dtype,
            family="rmsnorm", phase=mode, recipe="k_norm",
            shape=(padded_tokens, projected_kv_heads, width),
            read_bytes_override=(padded_tokens * projected_kv_heads * width * act_bytes
                                 + width * act_bytes),
            write_bytes_override=padded_tokens * projected_kv_heads * width * act_bytes,
            metadata=(("reduction_width", width), ("norm_has_weight", True),
                      ("input_views", {"key": split_views["key"].to_dict()})),
        )
        emitted.append(k_norm)
    return emitted, tuple(op.op_id for op in emitted)


def _gqa_projections(
    prefix, plan, stage, deps, model, quant, mode, padded_tokens, requests, heads,
    projected_kv_heads, projection_planes, width, shared, gate_separate,
):
    v_width = int(getattr(plan, "v_head_dim", 0) or width)
    q_width = plan.q_projection_width(heads, width)
    kv_width = projected_kv_heads * width
    v_proj_width = projected_kv_heads * v_width
    kv_total_width = (kv_width + v_proj_width) if projection_planes == 2 else (projection_planes * kv_width)
    total_n = q_width + kv_total_width
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
                "name": "v", "shape": (padded_tokens, v_proj_width),
                "strides": (total_n, 1), "offset_elements": q_width + kv_width,
                "units": projected_kv_heads, "unit_width": v_width, "shard": "kv_out",
            })
    qkv_meta = [("packed_projection_slices", tuple(packed_slices))]
    if plan.k_eq_v:
        qkv_meta.append(("k_eq_v_runtime_cloned_v", True))
    qkv = gemm_op(
        op_id=prefix + ".qkv", deps=(deps,), stage=stage, layer=plan.index,
        requests=requests, m=padded_tokens, n=total_n,
        k=model.hidden_size, quant=quant, category="attn.qkv", phase=mode,
        family="qkv_projection", recipe="qkv_projection",
        metadata=tuple(qkv_meta),
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
    mrope_active = uses_mrope_qkv_kernel(model, plan)
    rope_deps = (ready,)
    if plan.qk_norm:
        split_views = _qkv_views(
            padded_tokens, heads, projected_kv_heads, width,
            plan.q_output_gate and not gate_separate, False)
        if mrope_active:
            normalized = vector_op(
                op_id=prefix + ".qk_norm", deps=(ready,), stage=stage, layer=plan.index,
                requests=requests, elements=padded_tokens * (heads + projected_kv_heads) * width,
                ops_per_element=RMS_NORM_OPS_PER_ELEMENT, dtype=model.activation_dtype, family="rmsnorm",
                phase=mode, recipe="qk_norm", shape=(padded_tokens, heads + projected_kv_heads, width),
                metadata=(("input_views", {
                    name: split_views[name].to_dict()
                    for name in (("query",) if shared else ("query", "key"))}),))
            ops.append(normalized)
            rope_deps = (normalized.op_id,)
            split_copies = [
                _qkv_output_copy(qkv, split_views[name], name, model.activation_dtype)
                for name in ("value", "gate")
                if name in split_views and (name != "value" or not shared)
            ]
            ops.extend(split_copies)
        else:
            # Ordinary text RoPE invokes Q-norm and K-norm separately and keeps
            # V as a zero-copy view off QKV (no standalone V copy or MRoPE fusion).
            norm_ops, rope_deps = _emit_ordinary_qk_norm(
                prefix, plan, stage, ready, model, mode, padded_tokens, requests,
                heads, projected_kv_heads, width, shared, split_views,
            )
            ops.extend(norm_ops)
    norm_cfg = dict(getattr(model, "catalog_facts", ()) or {}).get("normalization")
    has_v_norm = (
        bool(getattr(plan, "v_norm", False))
        or (isinstance(norm_cfg, dict) and bool(norm_cfg.get("v_norm", False)))
    )
    if has_v_norm and not shared and projected_kv_heads > 0:
        act_bytes = dtype_bytes(model.activation_dtype)
        v_norm_elems = padded_tokens * projected_kv_heads * v_width
        v_norm = vector_op(
            op_id=prefix + ".v_norm", deps=(qkv.op_id,), stage=stage, layer=plan.index,
            requests=requests, elements=v_norm_elems,
            ops_per_element=RMS_NORM_OPS_PER_ELEMENT, dtype=model.activation_dtype,
            family="rmsnorm", phase=mode, recipe="v_norm",
            shape=(padded_tokens, projected_kv_heads, v_width),
            read_bytes_override=v_norm_elems * act_bytes,
            write_bytes_override=v_norm_elems * act_bytes,
            metadata=(("reduction_width", v_width), ("norm_has_weight", False)),
        )
        ops.append(v_norm)
        split_copies = [v_norm, *split_copies]
    # A declared partial-RoPE width rotates only that prefix of each head; the
    # remaining tail is an unchanged view, never a shrunk projection.
    rotary_width = (plan.qk_rope_head_dim
                    if plan.qk_rope_head_dim or plan.qk_nope_head_dim else width)
    rope = vector_op(
        op_id=prefix + ".rope", deps=rope_deps, stage=stage, layer=plan.index,
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
    v_width = int(getattr(plan, "v_head_dim", 0) or width)
    kv_plane_width = (width + v_width) if cache_planes == 2 else (cache_planes * width)
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
            link_bytes=kv_plane_width * kv_heads * dtype_bytes(quant.kv_dtype)
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
            width=kv_plane_width * kv_heads, region="gqa_kv",
            stage=stage, layer=plan.index, requests=requests, phase=mode,
            model=model, views=views, kv_heads=kv_heads, head_width=width,
            parallel=parallel, quant=quant, key_fmt=key_fmt, value_fmt=value_fmt,
            shared_format=True, v_head_width=v_width, plan=plan),)
    else:
        key_store = _store_cache(
            ops, prefix=prefix, deps=store_deps, rows=store_rows, width=kv_heads * width,
            region="gqa_kv_key", name="cache_key", stage=stage, layer=plan.index, requests=requests,
            phase=mode, model=model, views=views, kv_heads=kv_heads, head_width=width,
            parallel=parallel, quant=quant, key_fmt=key_fmt, value_fmt=value_fmt,
            shared_format=False, plan=plan)
        value_store = _store_cache(
            ops, prefix=prefix, deps=store_deps, rows=store_rows, width=kv_heads * v_width,
            region="gqa_kv_value", name="cache_value", stage=stage, layer=plan.index, requests=requests,
            phase=mode, model=model, views=views, kv_heads=kv_heads, head_width=v_width,
            parallel=parallel, quant=quant, key_fmt=key_fmt, value_fmt=value_fmt,
            shared_format=False, plan=plan)
        store_ids = (key_store, value_store)
    return valid, pcp_prefill, store_ids


def _gqa_sparse_dcp_split(
    indexed_views, eff_cp, eff_window, plan, policy, cp_interleave_size, runtime,
):
    """Batch-pooled sparse Top-K DCP rank partition for GQA views."""
    pool_size = max(1, int(getattr(plan, "index_kpool", 1) or 1))
    pool_tail = bool(getattr(plan, "index_kpool_always_select_tail", False))
    pairs_totals: list[float] = []
    gathers_totals: list[int] = []
    pairs_seqs = []
    sweeps_seqs = []
    for _, view in indexed_views:
        q, context = view.q_len, view.context
        ranges = view.query_ranges or ((0, q),)
        selected_q = sum(end - start for start, end in ranges if end > start)
        pairs = float(sum(
            _long_selected_pairs(
                end - start, context + start, 1, eff_window,
                noncausal=bool(plan.noncausal_attention), kv_len=view.kv_len)
            for start, end in ranges if end > start
        ))
        gathers = int(sum(
            sparse_attention_traffic(
                end - start, context + start, eff_window,
                query_tile=policy.query_tile,
                noncausal=bool(plan.noncausal_attention),
                kv_len=view.kv_len).gather_rows
            for start, end in ranges if end > start
        ))
        always_tail = int(view.kv_len) % pool_size if pool_size > 1 and pool_tail else 0
        pairs_totals.append(pairs)
        gathers_totals.append(gathers)
        pairs_seqs.append(sparse_dcp_sequence_for_total(
            view.kv_len, pairs, selected_q, compress_ratio=1,
            always_select_tail=always_tail))
        sweeps_seqs.append(sparse_dcp_sequence_for_total(
            view.kv_len, gathers, selected_q, compress_ratio=1,
            always_select_tail=always_tail))
    if eff_cp <= 1:
        return (
            pairs_totals,
            [[p] for p in pairs_totals],
            [[g] for g in gathers_totals],
        )
    _, per_seq_pairs = split_sparse_dcp_batch_per_sequence(
        pairs_seqs, eff_cp, cp_interleave_size, runtime=runtime, unit="pairs")
    _, per_seq_sweeps = split_sparse_dcp_batch_per_sequence(
        sweeps_seqs, eff_cp, cp_interleave_size, runtime=runtime, unit="sweeps")
    seq_caps = [sparse_dcp_capacities(seq, eff_cp, cp_interleave_size) for seq in sweeps_seqs]
    sweeps_matrix = round_sparse_dcp_matrix(per_seq_sweeps, gathers_totals, seq_caps)
    return (
        pairs_totals,
        [list(row) for row in per_seq_pairs],
        sweeps_matrix,
    )


def _gqa_rank_work(
    indexed_views, eff_cp, eff_topk, eff_window, plan, policy, dense_win, cp_interleave_size,
    block_size, runtime=None,
):
    rank_pairs = [0.0] * eff_cp
    rank_sweeps = [0] * eff_cp
    rank_rows = [0] * eff_cp
    per_view_pairs: list[tuple[float, list[float]]] = []
    per_view_sweeps: list[list[int]] = []
    per_view_shared: list[list[int]] = []
    per_view_rows: list[list[int]] = []
    per_view_spans: list[tuple[list[tuple[int, int]], list[tuple[int, int]]]] = []
    sparse_totals = sparse_v_pairs = sparse_v_sweeps = None
    if eff_topk > 0:
        sparse_totals, sparse_v_pairs, sparse_v_sweeps = _gqa_sparse_dcp_split(
            indexed_views, eff_cp, eff_window, plan, policy, cp_interleave_size, runtime)
    for pos, (_, view) in enumerate(indexed_views):
        q, context = view.q_len, view.context
        ranges = view.query_ranges or ((0, q),)
        selected_q = sum(end - start for start, end in ranges if end > start)
        win_spans: list[tuple[int, int]] = []
        sh_spans: list[tuple[int, int]] = []
        if eff_topk > 0:
            pairs = sparse_totals[pos]
            v_pairs = sparse_v_pairs[pos]
            v_sweeps = sparse_v_sweeps[pos]
            v_shared = [0] * eff_cp
        else:
            visible_kv = (
                min(view.kv_len, dense_win + selected_q) if (plan.noncausal_attention and dense_win > 0)
                else min(view.kv_len, dense_win) if dense_win > 0
                else view.kv_len
            )
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
                        lo = max(0, context - dense_win) if dense_win > 0 else 0
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
        if dense_win > 0 and win_spans and not (eff_topk > 0 and plan.index_topk > 0):
            u_lo = min(lo for lo, _ in win_spans)
            u_hi = max(hi for _, hi in win_spans)
            v_rows = (
                [owned_positions(u_lo, u_hi - 1, eff_cp, cp_interleave_size, r)
                 for r in range(eff_cp)]
                if eff_cp > 1 else [max(0, u_hi - u_lo)]
            )
        else:
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
    parallel, eff_window, dense_win, runtime, cacheless, selected, query_producer,
    dcp_query_producer, add_input_ports, key_fmt, value_fmt, shared_format,
    cp_interleave_size, block_size, chunked_pf_pcp=False, cp_rank=None,
    physical_rank=None, selected_rows=None,
    local_fia_attn_deps=None, dcp_fia_attn_deps=None,
):
    heads = dcp_heads if eff_cp > 1 else local_heads
    q_eff = (int(getattr(view, "global_q_len", 0) or view.q_len)
             if chunked_pf_pcp else int(view.q_len))
    eff_local_pairs = float(local_pairs)
    q_prod = dcp_query_producer if eff_cp > 1 else query_producer
    sparse_nk = _sparse_gqa_native_kind(plan, view=view, mode=mode, eff_topk=eff_topk, topk_id=topk_id)
    if sparse_nk == FIA_PREFILL_NATIVE_KIND and local_fia_attn_deps is not None:
        attention_deps = dcp_fia_attn_deps if eff_cp > 1 else local_fia_attn_deps
    else:
        attention_deps = dcp_attn_deps if eff_cp > 1 else local_attn_deps
    qk_width = width
    v_width = int(getattr(plan, "v_head_dim", 0) or width)
    traffic = plan_attention_traffic(
        AttentionAccessSpec(
            q_lengths=(q_eff,),
            context_lengths=(int(max(0, view.kv_len - view.q_len)),),
            kv_heads=int(kv_heads), q_heads=int(heads),
            key_dim=int(qk_width), value_dim=int(v_width),
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
    element_bytes = dtype_bytes(model.activation_dtype)
    query_bytes = q_eff * heads * qk_width * element_bytes
    output_bytes = q_eff * heads * v_width * element_bytes
    idx_q_heads = (
        local_kv_heads(max(1, int(plan.index_heads or 1)), max(1, int(parallel.tp or 1)))
        if is_block_msa_indexer(plan)
        else 1
    )
    selection_bytes = (
        float(q_eff * sel_topk * idx_q_heads * INT32_BYTES)
        if topk_id and eff_topk > 0 else 0.0
    )
    sparse_nk_meta = (("native_kind", sparse_nk),) if (sparse_nk and not selected) else ()
    op_id = f"{prefix}.attention.{orig_idx}"
    shape = (q_eff, view.kv_len, heads, qk_width)
    op = OpSpec(
        op_id=op_id, family="attention", phase=mode,
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
                  ("selection_read_bytes", selection_bytes),
                  ("attention_traffic_plan", traffic.to_dict()),
                  ("cp_per_rank_pairs", tuple(v_pairs)),
                  ("cp_per_rank_sweeps", tuple(v_sweeps)))
                 + sparse_nk_meta
                 + _decode_contract(
                     (view,), heads, kv_heads, width, model, quant, parallel, plan, mode,
                     window=eff_window, cp=eff_cp)
                 + _prefill_contract(
                     (view,), heads, kv_heads, width, model, quant, parallel, plan, mode,
                     window=eff_window, cp=eff_cp)
                 + _attention_tensor_view(
                     (view,), heads, kv_heads, width, model, quant, parallel,
                     policy, runtime, causal=not plan.noncausal_attention,
                     cacheless=cacheless, plan=plan, phase=mode)
                 + gqa_kv_memory_accesses(
                     (view,), op_id=op_id, stage=stage, layer=plan.index,
                     kv_heads=kv_heads, width=qk_width, v_width=v_width,
                     key_fmt=key_fmt, value_fmt=value_fmt, shared_format=shared_format,
                     mode=mode, window=dense_win, cp=eff_cp,
                     cp_interleave=cp_interleave_size, cp_rank=cp_rank,
                     physical_rank=physical_rank,
                     block_size=block_size,
                     prefix_cache_mode=str(getattr(runtime, "prefix_cache_mode", "private") or "private"),
                     cacheless=cacheless, noncausal=bool(plan.noncausal_attention),
                     allow_shared_prefix=False,
                     use_global_q=chunked_pf_pcp,
                     selected_topk=eff_topk,
                     selected_rows_by_view=((int(selected_rows),) if selected_rows is not None else None),
                 ),
    )
    if selected:
        sparse_layer = eff_topk > 0 or int(getattr(plan, "index_topk", 0) or 0) > 0
        if sparse_layer and not _selected_supports_sparse_selection(selected):
            raise ModelingError(
                f"gqa_attention_lowering_ignores_sparse_selection:{getattr(selected, 'name', selected.module)}"
            )
        sparse_flops = float(op.flops)
        op = selected.module.lower_op(
            op, view=view, plan=plan, key_heads=kv_heads, model=model,
            runtime=runtime, **selected.options)
        if sparse_layer and float(op.flops) > sparse_flops * (1.0 + _SPARSE_FLOP_TOLERANCE):
            raise ModelingError(
                f"gqa_attention_lowering_violated_sparse_topk_flops:{getattr(selected, 'name', selected.module)}"
            )
    if q_prod == query_producer:
        op = add_input_ports(op, ({"op": query_producer, "output": "query"},))
    return op, kv_bytes, kv_refill_bytes


def _build_gqa_requests(
    indexed_views, per_view_pairs, per_view_sweeps, per_view_rows, per_view_spans,
    per_view_shared, worst_r, worst_sweeps_r, eff_cp, cp_interleave_size, request_op,
    *, chunked_pf_pcp: bool = False, physical_rank=None,
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
            view_sweeps, local_rows, eff_cp, chunked_pf_pcp=chunked_pf_pcp,
            cp_rank=worst_r, physical_rank=physical_rank,
            selected_rows=per_view_sweeps[v_pos][worst_r])
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


def _group_gqa_phase_views(
    views, plan, parallel, *, dp_width, cacheless, cohort_decode_views,
    qsa_dispatch: bool = False,
):
    layer_dcp = layer_dcp_degree(plan, parallel)
    phase_groups: list[tuple[int, bool, list[tuple[int, object]]]] = []
    group_map: dict[tuple[int, bool, str], list[tuple[int, object]]] = {}
    has_dp_decode = dp_width > 1 and layer_dcp > 1 and bool(cohort_decode_views)
    for index, view in enumerate(views):
        if has_dp_decode and view.phase != "prefill":
            continue
        chunked_pf = (
            view.phase == "prefill" and view.context > 0
            and layer_dcp > 1 and not cacheless
        )
        eff_cp = layer_dcp if chunked_pf else active_layer_dcp(plan, parallel, view.phase)
        qsa_branch = (
            ("fia" if _qsa_view_uses_fia(view) else "gbsa")
            if qsa_dispatch else ""
        )
        gkey = (eff_cp, chunked_pf, qsa_branch)
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
    mixed_split_rows = (
        sum(int(v.q_len) for v in grp_views if int(v.q_len) > 1)
        if (split_curr and not chunked_pf and any(int(v.q_len) == 1 for v in grp_views))
        else None
    )
    eff_merge_rows = (
        grp_valid if chunked_pf_pcp
        else mixed_split_rows if (mixed_split_rows is not None and mixed_split_rows > 0)
        else None
    )
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
        merge_rows=eff_merge_rows,
        meta_extra=(("cp_interleave", int(cp_interleave_size)),
                    ("gqa_chunked_prefill_dcp", bool(chunked_pf))))


def _subtract_current_chunk_history_traffic(
    indexed_views, per_view_rows, per_view_sweeps, per_view_spans,
    rank_rows, rank_sweeps, *, eff_cp: int, cp_interleave_size: int, chunked_pf: bool,
) -> None:
    """Clip history-only unique/sweep KV spans strictly to ``[0, context)`` (D2 §2.1.1)."""
    for v_pos, (_, view) in enumerate(indexed_views):
        if not ((chunked_pf or int(view.q_len) > 1) and int(view.kv_len) > int(view.context)):
            continue
        ctx = int(view.context)
        win_spans, sh_spans = per_view_spans[v_pos]
        clipped_win = [(lo, min(hi, ctx)) for lo, hi in win_spans if min(hi, ctx) > lo]
        clipped_sh = [(lo, min(hi, ctx)) for lo, hi in sh_spans if min(hi, ctx) > lo]
        per_view_spans[v_pos] = (clipped_win, clipped_sh)
        u_lo = min((lo for lo, _ in clipped_win), default=0)
        u_hi = max((hi for _, hi in clipped_win), default=0)
        for r in range(eff_cp):
            old_rows = per_view_rows[v_pos][r]
            new_rows = (
                owned_positions(u_lo, u_hi - 1, eff_cp, cp_interleave_size, r)
                if u_hi > u_lo else 0
            )
            new_rows = min(old_rows, new_rows)
            per_view_rows[v_pos][r] = new_rows
            rank_rows[r] = max(0, rank_rows[r] - (old_rows - new_rows))
            old_sw = per_view_sweeps[v_pos][r]
            new_sw = sum(
                owned_positions(lo, hi - 1, eff_cp, cp_interleave_size, r)
                for lo, hi in clipped_win
            )
            new_sw = min(old_sw, new_sw)
            per_view_sweeps[v_pos][r] = new_sw
            rank_sweeps[r] = max(0, rank_sweeps[r] - (old_sw - new_sw))


def _finalize_gqa_group_batches(
    ops, batches, *, grp_prefix, grp_reqs, grp_views, grp_hist_valid, indexed_views,
    sweeps_by_view, shared_sweeps_by_view, unique_win_by_view, shared_iv_by_view,
    kv_hbm_total, kv_mte2_total, kv_components, rank_pairs, rank_sweeps, rank_rows,
    sel_r, worst_r, worst_sweeps_r, split_curr, sparse_imb_meta,
    heads, kv_heads, width, v_width, streams, kv_row_bytes, eff_topk, eff_window,
    dense_win, eff_cp, cp_interleave_size, block_size,
    key_fmt, value_fmt, shared_format,
    policy, model, quant, parallel, plan, mode, hierarchy, runtime, cacheless,
    selected, user_pinned_attn, chunked_pf_pcp=False, physical_rank=None,
    selected_rows_by_view=None,
) -> list[str]:
    """Fuse or annotate per-group GQA attention batch ops with prefix-read accounting."""
    if policy.fused_batch and not selected and batches:
        fields = ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes")
        batched_nk = dict(batches[0].metadata).get("native_kind")
        batched_nk_meta = (("native_kind", batched_nk),) if batched_nk else ()
        batched_sel_bytes = sum(float(dict(op.metadata).get("selection_read_bytes", 0.0)) for op in batches)
        batched_op_id = grp_prefix + ".attention"
        batched = replace(
            batches[0], op_id=batched_op_id, requests=grp_reqs,
            semantic_shape=(grp_hist_valid, max(v.kv_len for v in grp_views), heads, width),
            local_shape=(grp_hist_valid, rank_rows[sel_r], heads, width),
            metadata=(("request_shapes", tuple((v.q_len, v.context) for v in grp_views)),
                      ("query_tile", policy.query_tile),
                      ("kv_streams", streams),
                      ("kv_sweeps", sum(sweeps_by_view)),
                      ("selection_read_bytes", batched_sel_bytes),
                      ("cp_per_rank_pairs", tuple(rank_pairs)),
                      ("cp_per_rank_sweeps", tuple(rank_sweeps)),
                      ("cp_bottleneck_pairs_rank", worst_r),
                      ("cp_bottleneck_sweeps_rank", worst_sweeps_r),
                      ("dcp_mtp_split", "history" if split_curr else "unified"))
                     + batched_nk_meta
                     + sparse_imb_meta
                     + _decode_contract(
                         grp_views, heads, kv_heads, width, model, quant,
                         parallel, plan, mode, window=eff_window, cp=eff_cp)
                     + _prefill_contract(
                         grp_views, heads, kv_heads, width, model, quant,
                         parallel, plan, mode, window=eff_window, cp=eff_cp)
                     + _attention_tensor_view(
                         grp_views, heads, kv_heads, width, model, quant, parallel,
                         policy, runtime, causal=not plan.noncausal_attention,
                         cacheless=cacheless, plan=plan, phase=mode)
                     + gqa_kv_memory_accesses(
                         grp_views, op_id=batched_op_id, stage=batches[0].stage,
                         layer=plan.index, kv_heads=kv_heads, width=width, v_width=v_width,
                         key_fmt=key_fmt, value_fmt=value_fmt, shared_format=shared_format,
                         mode=mode, window=dense_win, cp=eff_cp,
                         cp_interleave=cp_interleave_size, cp_rank=sel_r,
                         physical_rank=physical_rank,
                         block_size=block_size,
                         prefix_cache_mode=str(getattr(runtime, "prefix_cache_mode", "private") or "private"),
                         cacheless=cacheless, noncausal=bool(plan.noncausal_attention),
                         allow_shared_prefix=(eff_topk <= 0),
                         use_global_q=chunked_pf_pcp,
                         selected_topk=eff_topk,
                         selected_rows_by_view=selected_rows_by_view,
                     ),
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
                kernel_reuse=reuse_kind, allow_recipe_rebind=not user_pinned_attn,
                kv_hbm_bytes=kv_hbm_total, kv_mte2_bytes=kv_mte2_total,
                unique_rows_by_view=unique_win_by_view,
                shared_intervals_by_view=shared_iv_by_view, geometry=hierarchy,
            )
        ops.append(batched)
        return [batched.op_id]
    if not batches:
        return []
    if eff_topk > 0:
        batches = [
            annotate_sparse_prefix_unknown(
                replace(item, metadata=item.metadata + sparse_imb_meta)
                if sparse_imb_meta else item,
                runtime,
            )
            for item in batches
        ]
    elif not cacheless:
        for pos, item in enumerate(batches):
            view = indexed_views[pos][1]
            kv_hbm_b, kv_mte2_b = kv_components[pos]
            batches[pos] = apply_attention_prefix_reads(
                item, views=(view,), sweeps_by_view=(sweeps_by_view[pos],),
                shared_sweeps_by_view=(shared_sweeps_by_view[pos],),
                row_bytes=float(kv_row_bytes), layer_index=plan.index, runtime=runtime,
                kernel_reuse="native_per_sequence", allow_recipe_rebind=False,
                kv_hbm_bytes=kv_hbm_b, kv_mte2_bytes=kv_mte2_b,
                unique_rows_by_view=(unique_win_by_view[pos],),
                shared_intervals_by_view=(shared_iv_by_view[pos],),
            )
    ops.extend(batches)
    return [item.op_id for item in batches]


def _append_gqa_attention(
    ops, views, parallel, prefix, eff_topk, eff_window, plan, policy, dense_win,
    cp_interleave_size, block_size, kv_heads, local_heads, dcp_heads, width, kv_row_bytes,
    mode, hierarchy, model, stage, streams, quant, runtime, cacheless, selected,
    user_pinned_attn, request_op, *, pcp_factor, tp_gather, dp_width, local_attn_deps,
    cohort_decode_views=None, peer_decode_rows=None, local_dp_pos=0,
    physical_rank=None, explicit_dp_cohort=False, explicit_physical_rank=False,
    qsa_dispatch: bool = False, key_fmt=None, value_fmt=None, shared_format: bool = True,
):
    phase_groups = _group_gqa_phase_views(
        views, plan, parallel, dp_width=dp_width, cacheless=cacheless,
        cohort_decode_views=cohort_decode_views, qsa_dispatch=qsa_dispatch)
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
            (chunked_pf and eff_topk <= 0 and not bool(plan.noncausal_attention))
            or should_split_mtp_dcp(
                plan, grp_views, layer_dcp=eff_cp, sparse=bool(eff_topk > 0),
            )
        )
        chunked_pf_pcp = bool(chunked_pf and parallel.pcp > 1)
        grp_hist_valid = (
            sum(int(getattr(v, "global_q_len", 0) or v.q_len) for v in grp_views)
            if chunked_pf_pcp else grp_valid
        )
        (rank_pairs, rank_sweeps, rank_rows, per_view_pairs, per_view_sweeps,
         per_view_shared, per_view_rows, per_view_spans, worst_r, worst_sweeps_r) = _gqa_rank_work(
            indexed_views, eff_cp, eff_topk, eff_window, plan, policy, dense_win,
            cp_interleave_size, block_size, runtime=runtime)
        if split_curr and eff_cp > 1:
            rank_pairs, per_view_pairs = _gqa_history_pairs(
                indexed_views, per_view_pairs, eff_cp, cp_interleave_size, dense_win,
                chunked_pf=chunked_pf, use_global_q=chunked_pf_pcp)
            _subtract_current_chunk_history_traffic(
                indexed_views, per_view_rows, per_view_sweeps, per_view_spans,
                rank_rows, rank_sweeps, eff_cp=eff_cp,
                cp_interleave_size=cp_interleave_size, chunked_pf=chunked_pf,
            )
            worst_r = max(range(eff_cp), key=lambda r: (rank_pairs[r], rank_sweeps[r]))
            worst_sweeps_r = max(range(eff_cp), key=lambda r: (rank_sweeps[r], rank_pairs[r]))
        balanced_pairs = sum(rank_pairs) / float(eff_cp) if eff_cp > 0 else 0.0
        sparse_imb_meta = (
            (("cp_sparse_imbalance",
              float(rank_pairs[worst_r]) / balanced_pairs if balanced_pairs > 0.0 else 1.0),)
            if eff_cp > 1 and eff_topk > 0
            else ()
        )
        cand_ranks = resolve_cp_rank_candidates(
            parallel, eff_cp=eff_cp, pcp_factor=pcp_factor, tp_gather=tp_gather,
            dp_width=dp_width, local_dp_pos=local_dp_pos, physical_rank=physical_rank,
            explicit_dp_cohort=explicit_dp_cohort,
            explicit_physical_rank=explicit_physical_rank)
        sel_r = max(cand_ranks, key=lambda r: (rank_pairs[r], rank_sweeps[r]))
        sel_sweeps_r = max(cand_ranks, key=lambda r: (rank_sweeps[r], rank_pairs[r]))
        eff_phys_rank = (
            int(physical_rank)
            if (explicit_physical_rank and physical_rank is not None)
            else None
        )
        sel_sparse_rows = (
            tuple(int(per_view_sweeps[pos][sel_r]) for pos in range(len(indexed_views)))
            if eff_topk > 0
            else None
        )
        (batches, sweeps_by_view, shared_sweeps_by_view, unique_win_by_view,
         shared_iv_by_view, _total_pairs, _total_rows, kv_hbm_total,
         kv_mte2_total, kv_components) = _build_gqa_requests(
            indexed_views, per_view_pairs, per_view_sweeps, per_view_rows, per_view_spans,
            per_view_shared, sel_r, sel_sweeps_r, eff_cp, cp_interleave_size,
            request_op, chunked_pf_pcp=chunked_pf_pcp, physical_rank=eff_phys_rank)
        grp_attn = _finalize_gqa_group_batches(
            ops, batches, grp_prefix=grp_prefix, grp_reqs=grp_reqs, grp_views=grp_views,
            grp_hist_valid=grp_hist_valid, indexed_views=indexed_views,
            sweeps_by_view=sweeps_by_view, shared_sweeps_by_view=shared_sweeps_by_view,
            unique_win_by_view=unique_win_by_view, shared_iv_by_view=shared_iv_by_view,
            kv_hbm_total=kv_hbm_total, kv_mte2_total=kv_mte2_total, kv_components=kv_components,
            rank_pairs=rank_pairs, rank_sweeps=rank_sweeps, rank_rows=rank_rows,
            sel_r=sel_r, worst_r=worst_r, worst_sweeps_r=worst_sweeps_r,
            split_curr=split_curr, sparse_imb_meta=sparse_imb_meta,
            heads=heads, kv_heads=kv_heads, width=width, v_width=v_width, streams=streams,
            kv_row_bytes=kv_row_bytes, eff_topk=eff_topk, eff_window=eff_window,
            dense_win=dense_win, eff_cp=eff_cp, cp_interleave_size=cp_interleave_size,
            block_size=block_size, key_fmt=key_fmt, value_fmt=value_fmt,
            shared_format=shared_format,
            policy=policy, model=model, quant=quant, parallel=parallel, plan=plan, mode=mode,
            hierarchy=hierarchy, runtime=runtime, cacheless=cacheless, selected=selected,
            user_pinned_attn=user_pinned_attn, chunked_pf_pcp=chunked_pf_pcp,
            physical_rank=eff_phys_rank, selected_rows_by_view=sel_sparse_rows,
        )
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
    v_width = int(getattr(plan, "v_head_dim", 0) or width)
    act_bytes = dtype_bytes(model.activation_dtype)
    gate_elems = padded_tokens * heads * v_width
    if gate_separate:
        gate = vector_op(
            op_id=prefix + ".output_gate", deps=tuple(attention) + (separate_gate,),
            stage=stage, layer=plan.index, requests=requests,
            elements=gate_elems,
            ops_per_element=GATE_APPLY_OPS_PER_ELEMENT,
            dtype=model.activation_dtype, family="elementwise", phase=mode,
            recipe="attention_gate",
            read_bytes_override=gate_elems * act_bytes + padded_tokens * heads * dtype_bytes("fp32"),
            write_bytes_override=gate_elems * act_bytes,
            metadata=(("input_tensor_count", 2),),
        )
        ops.append(gate)
        attention = [gate.op_id]
    elif plan.q_output_gate:
        gate_ready = split_copies[-1].op_id if split_copies else qkv.op_id
        gate = vector_op(
            op_id=prefix + ".output_gate", deps=tuple(attention) + (gate_ready,),
            stage=stage, layer=plan.index, requests=requests,
            elements=gate_elems, ops_per_element=OUTPUT_GATE_OPS_PER_ELEMENT,
            dtype=model.activation_dtype, family="elementwise", phase=mode, recipe="attention_gate",
            read_bytes_override=2 * gate_elems * act_bytes,
            write_bytes_override=gate_elems * act_bytes,
            metadata=(("input_tensor_count", 2),),
        )
        ops.append(gate)
        attention = [gate.op_id]
    pcp_shard_o = (parallel.pcp > 1 and pcp_o_proj_weight_sharding_enabled(runtime))
    pcp_is_prefill = any(getattr(v, "phase", "") == "prefill" for v in views)
    o_full_width = heads * v_width
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
    if (plan.qk_norm and normalized is not None and not shared
            and not plan.k_eq_v and uses_mrope_qkv_kernel(model, plan)):
        _declare_qkv_norm_rope(
            ops, qkv, normalized, rope, split_copies, split_views, model.activation_dtype)
    elif plan.qk_norm and normalized is None and not shared and not plan.k_eq_v and split_views:
        norm_ids = tuple(
            op.op_id for op in ops if op.op_id in (prefix + ".q_norm", prefix + ".k_norm")
        )
        if norm_ids:
            _declare_ordinary_qk_norm_rope(
                ops, qkv, norm_ids, rope, split_views, model.activation_dtype)
    return last


#: Index of the value copy inside the emitted split-copy list.
VALUE_COPY = 0

ATTENTION_BUILDERS = {
    "gqa": gqa, "mla": mla, "gdn": linear, "kda": linear,
    "mamba": linear, "gla": linear, "deltanet": linear,
    "lightning": linear, "retnet": linear,
    "projection_only": projection_only, "unmodeled": unmodeled,
}
