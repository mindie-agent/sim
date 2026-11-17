"""Paired latent/indexer/draft cache contracts.

[Role] Decode native sparse plans and quote shared pool pages for serving.
[Invariants] A pool ID reserves every region, including IDs used for states.
  Geometry selects mechanisms; source identities never select performance.
[Timeouts & Locks] Pure O(layers + pipeline stages) arithmetic; no locks.
[Side-Effect Policy] Returns fresh records; no weights, devices or I/O.
"""
from dataclasses import replace
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .cache_format import cache_format

ZERO = 0
ONE = 1
TWO = 2
QUANT_GROUP = 128
SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "c3d1d1fc1dbaee5c38c0601d23be99f14129fa46",
    "path": "vllm_ascend/models/kimi_k3/paired_sfa_cache.py",
}


def sparse_plans(config, plans):
    dsa = config.get("dsa_config")
    if not dsa:
        return plans
    layers = {int(value) - ONE for value in dsa.get("dsa_layers", ())}
    full = {plan.index for plan in plans if plan.attention == "mla"}
    if not layers or not layers <= full:
        raise ModelingError("kimi_dsa_layers_must_be_one_based_full_attention_layers")
    dimensions = {"index_heads": int(dsa["index_num_heads"]),
                  "index_dim": int(dsa["index_head_dim"]),
                  "index_topk": int(dsa["index_topk"])}
    if min(dimensions.values()) <= ZERO:
        raise ModelingError("kimi_dsa_dimensions_must_be_positive")
    if int(dsa.get("index_topk_freq", ONE)) != ONE:
        raise ModelingError("kimi_dsa_topk_reuse_schedule_not_declared")
    return [replace(plan, architecture="mla_dsa", indexer=True,
                    indexer_type="full", owns_index_k=True, owns_topk=True,
                    index_k_owner_layer_id=plan.index, topk_owner_layer_id=plan.index,
                    index_query_per_layer=True, **dimensions)
            if plan.index in layers else plan for plan in plans]


def execution(model):
    return dict(model.metadata).get("attention_execution", {})


def normalize_formats(model):
    if dict(model.metadata).get("cache_allocator") != "paired_sfa":
        return model
    metadata = dict(model.metadata)
    regions = dict(metadata.get("cache_representations", {}))
    latent = next((plan for plan in model.layer_plans if plan.attention == "mla"), None)
    sparse = next((plan for plan in model.layer_plans if plan.indexer), None)
    if latent is None:
        # A truncated slice (for example Kimi-K3.1 ``layers=2``) may contain
        # only KDA layers: the paired MLA/indexer regions do not exist, so the
        # ordinary per-component allocator owns this model.
        metadata["declared_cache_allocator"] = metadata["cache_allocator"]
        metadata["cache_allocator"] = "component"
        metadata["cache_allocator_selection"] = (
            "selected_layers_have_no_paired_mla_regions")
        return replace(model, metadata=tuple(metadata.items()))
    if dtype_bytes(model.quant.kv_dtype) == ONE:
        regions.setdefault("mla_kv", {
            "dtype": model.quant.kv_dtype, "group_size": QUANT_GROUP,
            "scale_dtype": "fp32", "suffix_elements": latent.qk_rope_head_dim,
            "suffix_dtype": "bf16"})
    if sparse is not None and dtype_bytes(model.quant.index_dtype) == ONE:
        regions.setdefault("index_k", {
            "dtype": model.quant.index_dtype, "group_size": sparse.index_dim,
            "scale_dtype": model.quant.index_scale_dtype
            if model.quant.index_scale_dtype not in {"", "none"} else "fp16"})
    metadata["cache_representations"] = regions
    return replace(model, metadata=tuple(metadata.items()))


def independent_draft(draft):
    return dict(draft.source).get("kind") == "external_kimi_k3_dspark"


def draft_row(model, draft):
    from .draft import draft_quant
    from .cache import local_kv_heads
    width = (draft.kv_cache_elements or draft.kv_lora_rank + draft.qk_rope_head_dim
             if draft.attention_kind == "mla" else
             TWO * local_kv_heads(draft.num_kv_heads, ONE) * draft.head_dim)
    return width * dtype_bytes(draft_quant(model, draft).kv_dtype)


def page_groups(case, layer_stage, query_tokens):
    """Use ordinary CachePages, with history IDs plus bounded state IDs."""
    model, runtime = case.model, case.runtime
    if dict(model.metadata).get("cache_allocator") != "paired_sfa":
        return ()
    if runtime.recover_ssm:
        raise ModelingError("paired_sfa_recover_ssm_layout_not_declared")
    from .draft import optional_draft_spec
    from .cache import index_row_bytes
    p, block = runtime.parallel, runtime.kv_block_size
    metadata = dict(runtime.metadata)
    mode = metadata.get("mamba_cache_mode", "align")
    if mode not in {"align", "none"}:
        raise ModelingError("paired_sfa_state_cache_mode_not_supported")
    fallback = max(ZERO, query_tokens - ONE)
    draft_len = getattr(runtime.dspark, "draft_len", None)
    if draft_len is not None and "draft_proposal_tokens" not in metadata:
        fallback = max(ZERO, int(draft_len) - ONE)
    proposals = int(metadata.get("num_speculative_state_blocks",
                                metadata.get("draft_proposal_tokens", fallback)))
    extra = int(metadata.get("num_prefill_checkpoint_blocks", ZERO))
    if min(proposals, extra) < ZERO:
        raise ModelingError("paired_sfa_negative_state_slots")
    slots = (TWO + proposals + extra if mode == "align" else ONE + proposals)
    draft_kind = metadata.get("draft_mode") or ("dspark" if runtime.dspark else "")
    draft = optional_draft_spec(model, draft_kind) if draft_kind else None
    result = []
    for stage in range(p.pp):
        plans = [plan for plan in model.layer_plans if layer_stage[plan.index] == stage]
        main = [plan for plan in plans if plan.attention == "mla"]
        recurrent = [plan for plan in plans if plan.attention == "kda"]
        if not main or any(plan.attention not in {"mla", "kda"} for plan in plans):
            raise ModelingError("paired_sfa_requires_mla_and_optional_kda_per_stage")
        rows = {cache_format(model, "mla_kv").row_bytes(plan.cache_row_elements) for plan in main}
        if len(rows) != ONE:
            raise ModelingError("paired_sfa_requires_uniform_main_pages")
        main_page = block * rows.pop()
        state_groups = math.ceil(len(recurrent) / len(main))
        state_width = math.ceil(len(recurrent) / state_groups) if state_groups else ZERO
        main_slots = max(len(main), state_width)
        # One recurrent checkpoint must fit in one main slot. Speculative
        # checkpoints consume distinct IDs rather than widening this page.
        for plan in recurrent:
            state_page = (plan.recurrent_state_elements * dtype_bytes(plan.recurrent_state_dtype)
                          + plan.conv_width * (plan.conv_kernel - ONE + proposals)
                          * dtype_bytes(plan.conv_state_dtype)) / p.tp
            if state_page > main_page:
                raise ModelingError("paired_sfa_block_too_small_for_recurrent_page")
        index = sum(index_row_bytes(model, plan.index_dim)
                    for plan in main if plan.owns_index_k)
        draft_bytes = (block * draft_row(model, draft) * draft.layers
                       if draft is not None and stage == p.pp - ONE
                       and (proposals > ZERO or bool(runtime.dspark)) else ZERO)
        result.append({
            "layers": [plan.index for plan in plans],
            "tokens_per_page": block * p.cp,
            "bytes_per_group_page": int(main_slots * main_page + block * index + draft_bytes),
            "recycling_pages": state_groups * slots,
        })
    return tuple(result)


BLOCK_ALIGNMENT = 16


def minimum_block_size(model, *, tp: int = 1, proposals: int = 0, extra: int = 0) -> int:
    """Minimum ``kv_block_size`` (aligned to 16) so every KDA state fits one main page."""
    if dict(model.metadata).get("cache_allocator") != "paired_sfa":
        return 0
    if not any(plan.indexer for plan in model.layer_plans):
        # A truncated plan set without the sparse owner cannot declare the
        # paired regions; no recurrent-page lower bound is claimed.
        return 0
    normalized = normalize_formats(model)
    main = [plan for plan in normalized.layer_plans if plan.attention == "mla"]
    recurrent = [plan for plan in normalized.layer_plans if plan.attention == "kda"]
    if not main or not recurrent:
        return 0
    row_bytes = min(
        cache_format(normalized, "mla_kv").row_bytes(plan.cache_row_elements)
        for plan in main
    )
    if row_bytes <= 0:
        return 0
    tp_degree = max(1, int(tp))
    spec_slots = max(0, int(proposals)) + max(0, int(extra))
    required = 0
    for plan in recurrent:
        state_page = (
            plan.recurrent_state_elements * dtype_bytes(plan.recurrent_state_dtype)
            + plan.conv_width * (plan.conv_kernel - 1 + spec_slots)
            * dtype_bytes(plan.conv_state_dtype)
        ) / tp_degree
        tokens = math.ceil(state_page / row_bytes)
        if tokens > required:
            required = tokens
    return ((required + BLOCK_ALIGNMENT - 1) // BLOCK_ALIGNMENT) * BLOCK_ALIGNMENT
