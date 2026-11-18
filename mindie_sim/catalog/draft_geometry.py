"""Pure draft LayerPlan inheritance geometry.

[Role]
  One pure owner for draft sparse/SWA/compressed/indexer inheritance so draft
  execution and capacity instantiate the same plans instead of maintaining a
  second geometry.

[Invariants]
  * Explicit per-layer compression/SWA choices win; otherwise the declared
    target reference layer supplies ownership, window, indexer and mHC fields.
  * Shared index-key ownership and per-layer query behaviour are preserved.
  * No model-name, measurement or hardware branch participates.

[Timeouts & Locks]
  Bounded by the draft layer count; no I/O or locks.

[Side-Effect Policy]
  Pure immutable LayerPlan construction.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.catalog.draft import optional_draft_spec
from mindie_sim.schema.model import LayerPlan

FIRST = 0
LAST = -1
ZERO = 0
ONE = 1


def _reference_attention_plan(model, draft, index: int) -> LayerPlan | None:
    from .paired_sfa import independent_draft
    if independent_draft(draft):
        return None
    if model.auxiliary is not None and model.auxiliary.standalone_draft:
        return None
    matching = [p for p in model.layer_plans if p.attention == draft.attention_kind]
    if not matching:
        return None
    ref = None
    if index < len(draft.target_layer_ids):
        target_id = int(draft.target_layer_ids[index])
        if ZERO <= target_id < len(model.layer_plans):
            candidate = model.layer_plans[target_id]
            if candidate.attention == draft.attention_kind:
                ref = candidate
    if ref is None:
        owners = [
            p for p in matching
            if (not p.indexer and p.topk_owner_layer_id < ZERO and p.kv_owner_layer_id in (LAST, p.index))
            or (p.owns_index_k and p.owns_topk)
        ]
        ref = owners[LAST] if owners else matching[LAST]
    if not ref.indexer and ZERO <= ref.topk_owner_layer_id < len(model.layer_plans):
        donor = model.layer_plans[ref.topk_owner_layer_id]
        if donor.attention == draft.attention_kind and donor.indexer:
            ref = replace(
                ref, indexer=True, indexer_type=donor.indexer_type,
                owns_index_k=True, owns_topk=True,
                index_query_per_layer=donor.index_query_per_layer,
                index_heads=donor.index_heads, index_dim=donor.index_dim,
                index_topk=donor.index_topk, index_kpool=donor.index_kpool,
                index_kpool_compress=donor.index_kpool_compress,
                index_kpool_always_select_tail=donor.index_kpool_always_select_tail,
                index_input=donor.index_input,
            )
    if not ref.owns_long_kv and ZERO <= ref.kv_owner_layer_id < len(model.layer_plans):
        kv_donor = model.layer_plans[ref.kv_owner_layer_id]
        if kv_donor.attention == draft.attention_kind and (
            kv_donor.owns_long_kv or kv_donor.compress_ratio > ZERO
        ):
            ref = replace(
                ref,
                owns_long_kv=True,
                compress_ratio=ref.compress_ratio if ref.compress_ratio > ZERO else kv_donor.compress_ratio,
                compressor_kind=kv_donor.compressor_kind if ref.compressor_kind == "none" else ref.compressor_kind,
                compressor_width=kv_donor.compressor_width if ref.compressor_kind == "none" else ref.compressor_width,
                compressor_state_rows=(
                    kv_donor.compressor_state_rows
                    if ref.compressor_kind == "none" else ref.compressor_state_rows
                ),
            )
    return ref


def _draft_layer_plan(model, draft, index: int, plan_fields: dict) -> LayerPlan:
    ref = _reference_attention_plan(model, draft, index)
    explicit_ratio = (
        int(draft.draft_compress_ratios[index])
        if index < len(draft.draft_compress_ratios) else None
    )
    if explicit_ratio is not None and explicit_ratio > ZERO and (
        ref is None or ref.compress_ratio != explicit_ratio
    ):
        same_ratio = [
            p for p in model.layer_plans
            if p.attention == draft.attention_kind and p.compress_ratio == explicit_ratio and p.owns_long_kv
        ]
        if same_ratio:
            ref = same_ratio[LAST]
        elif ref is not None:
            ref = replace(ref, cache_kind="window_compressed", compress_ratio=explicit_ratio, owns_long_kv=True)

    mhc = int(draft.mhc_mult or (ref.hc_mult if ref and ref.hyper_connections else ZERO))
    mhc_lr = int(draft.mhc_lowrank or (ref.hc_lowrank if ref and ref.hyper_connections else ZERO))
    mhc_iters = int(draft.mhc_sinkhorn_iters or (ref.hc_sinkhorn_iters if ref and ref.hyper_connections else ZERO))
    kv_elems = int(draft.kv_cache_elements or (ref.kv_cache_elements if ref else ZERO))
    qk_norm = bool(draft.qk_norm or (ref.qk_norm if ref else False))
    q_gate = bool(draft.q_output_gate or (ref.q_output_gate if ref else False))
    mla_gate = bool(ref.mla_output_gate) if ref else False

    swa_only = (
        explicit_ratio == ZERO
        or (
            explicit_ratio is None
            and ref is not None
            and not ref.owns_long_kv
            and ref.compress_ratio == ZERO
            and ref.kv_owner_layer_id < ZERO
            and ref.window > ZERO
        )
    )
    if swa_only:
        window = int(draft.sliding_window or (ref.window if ref else ZERO))
        cache_kind = ref.cache_kind if (ref and ref.cache_kind in {"dense", "window_compressed"}) else "window_compressed"
        return LayerPlan(
            index=index, attention=draft.attention_kind, architecture="swa",
            ffn="moe" if draft.num_experts else "dense",
            moe_intermediate_size=draft.expert_intermediate_size, scope="draft",
            noncausal_attention=draft.noncausal_query_block,
            mla_variant=draft.mla_variant, mla_output_gate=mla_gate,
            kv_cache_elements=kv_elems, qk_norm=qk_norm, q_output_gate=q_gate,
            cache_kind=cache_kind, window=window, kv_owner_layer_id=LAST, owns_long_kv=False,
            compress_ratio=ZERO, compressor_kind="none", compressor_width=ONE, compressor_state_rows=ZERO,
            indexer=False, indexer_type="none", owns_index_k=False, index_k_owner_layer_id=LAST,
            owns_topk=False, topk_owner_layer_id=LAST, index_query_per_layer=False,
            hyper_connections=bool(mhc), hc_mult=mhc, hc_lowrank=mhc_lr, hc_sinkhorn_iters=mhc_iters,
            **plan_fields,
        )

    if ref is None:
        standalone_win = int(draft.sliding_window or getattr(draft, "window", ZERO) or ZERO)
        standalone_comp = int(
            explicit_ratio if explicit_ratio is not None
            else (getattr(draft, "compress_ratio", ZERO) or ZERO)
        )
        standalone_arch = (
            "swa" if standalone_win > ZERO and standalone_comp == ZERO else "full"
        )
        standalone_ck = "window_compressed" if standalone_comp > ZERO else "dense"
        return LayerPlan(
            index=index, attention=draft.attention_kind, architecture=standalone_arch,
            ffn="moe" if draft.num_experts else "dense",
            moe_intermediate_size=draft.expert_intermediate_size, scope="draft",
            noncausal_attention=draft.noncausal_query_block,
            mla_variant=draft.mla_variant, kv_cache_elements=kv_elems,
            qk_norm=qk_norm, q_output_gate=q_gate,
            cache_kind=standalone_ck, window=standalone_win,
            compress_ratio=standalone_comp,
            kv_owner_layer_id=index, owns_long_kv=True,
            hyper_connections=bool(mhc), hc_mult=mhc, hc_lowrank=mhc_lr, hc_sinkhorn_iters=mhc_iters,
            **plan_fields,
        )

    has_indexer = bool((ref.indexer or ref.owns_topk) and ref.index_topk > ZERO)
    share_index_k = bool(has_indexer and draft.index_share_for_mtp_iteration and draft.attention_kind == "mla")
    indexer_type = (
        "shared" if share_index_k
        else (ref.indexer_type if ref.indexer_type in {"full", "c4"} else "full")
    ) if has_indexer else "none"
    owns_long = True if ref.cache_kind == "dense" else bool(
        ref.owns_long_kv or ref.compress_ratio > ZERO or ref.kv_owner_layer_id >= ZERO
    )
    window = int(draft.sliding_window or ref.window)
    return LayerPlan(
        index=index, attention=draft.attention_kind, architecture=ref.architecture,
        ffn="moe" if draft.num_experts else "dense",
        moe_intermediate_size=draft.expert_intermediate_size, scope="draft",
        noncausal_attention=draft.noncausal_query_block,
        mla_variant=draft.mla_variant, mla_output_gate=mla_gate,
        kv_cache_elements=kv_elems, qk_norm=qk_norm, q_output_gate=q_gate,
        cache_kind=ref.cache_kind, window=window,
        kv_owner_layer_id=index if owns_long else LAST, owns_long_kv=owns_long,
        compress_ratio=ref.compress_ratio, compressor_kind=ref.compressor_kind,
        compressor_width=ref.compressor_width, compressor_state_rows=ref.compressor_state_rows,
        indexer=has_indexer, indexer_type=indexer_type,
        owns_index_k=bool(has_indexer and not share_index_k),
        index_k_owner_layer_id=(index if (has_indexer and not share_index_k) else LAST),
        owns_topk=has_indexer, topk_owner_layer_id=(index if has_indexer else LAST),
        index_query_per_layer=bool(ref.index_query_per_layer) if has_indexer else False,
        index_heads=ref.index_heads if has_indexer else ZERO,
        index_dim=ref.index_dim if has_indexer else ZERO,
        index_topk=ref.index_topk if has_indexer else ZERO,
        index_kpool=ref.index_kpool if has_indexer else ONE,
        index_kpool_compress=bool(ref.index_kpool_compress) if has_indexer else False,
        index_kpool_always_select_tail=bool(ref.index_kpool_always_select_tail) if has_indexer else False,
        index_input=ref.index_input,
        candidate_filter=bool(ref.candidate_filter) if has_indexer else False,
        candidate_block_size=ref.candidate_block_size if has_indexer else ZERO,
        candidate_topk_blocks=ref.candidate_topk_blocks if has_indexer else ZERO,
        candidate_source_layer_id=LAST,
        hyper_connections=bool(mhc), hc_mult=mhc, hc_lowrank=mhc_lr, hc_sinkhorn_iters=mhc_iters,
        **plan_fields,
    )


#: Draft geometry fields a native auxiliary declaration carries per block.
_DRAFT_PLAN_FIELDS = (
    "num_heads", "num_kv_heads", "head_dim", "q_lora_rank", "kv_lora_rank",
    "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim", "o_lora_rank",
    "o_groups", "num_experts", "top_k", "num_shared_experts", "intermediate_size",
)


_INHERITED_ROUTER_FIELDS = (
    "router_mode", "router_dtype", "router_norm_topk", "router_groups",
    "router_group_topk", "router_noaux_tc", "routed_scaling_factor",
    "ffn_activation", "swiglu_limit", "swiglu_shared_limit",
    "swiglu_alpha", "swiglu_beta",
)


def _inherited_router_fields(model, draft) -> dict:
    """Draft MoE layers execute the target body's declared router.

    A draft block is not a second router architecture: it reuses the body's
    routing mode, correction bias and FFN activation scalars.  The correction
    bias parameter is counted per draft MoE layer by ``draft_weights``.
    """
    if int(getattr(draft, "num_experts", ZERO) or ZERO) <= ZERO:
        return {}
    donor = next((plan for plan in model.layer_plans if plan.ffn == "moe"), None)
    if donor is None:
        return {}
    fields = {name: getattr(donor, name) for name in _INHERITED_ROUTER_FIELDS}
    fields["router_input_size"] = int(getattr(draft, "hidden_size", ZERO)
                                      or donor.router_input_size)
    return fields


def native_draft_layer_plans(model, draft) -> tuple:
    """Native explicit auxiliary plans for one declared draft spec.

    Adapters call this once at model-load time so the declared ``layer_plans``
    are the single geometry authority; execution and capacity never re-derive a
    donor at use time.
    """
    plan_fields = {key: getattr(draft, key) for key in _DRAFT_PLAN_FIELDS}
    plan_fields.update(_inherited_router_fields(model, draft))
    return tuple(_draft_layer_plan(model, draft, index, plan_fields)
                 for index in range(int(draft.layers)))


def draft_layer_plans(model, kind="mtp"):
    """Declared native auxiliary plans; no implicit donor synthesis."""
    from mindie_sim.schema.common import ModelingError
    draft = optional_draft_spec(model, kind)
    plans = tuple(draft.layer_plans)
    if not plans or len(plans) != int(draft.layers):
        raise ModelingError("draft_explicit_layer_plans_required")
    return plans

