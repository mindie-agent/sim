"""Paired latent/indexer/draft cache contracts.

[Role] Decode native sparse plans and quote shared pool pages for serving.
[Invariants] A pool ID reserves every region, including IDs used for states.
  Geometry selects mechanisms; source identities never select performance.
[Timeouts & Locks] Pure O(layers + pipeline stages) arithmetic; no locks.
[Side-Effect Policy] Returns fresh records; no weights, devices or I/O.
"""
from dataclasses import dataclass, replace

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .cache_format import cache_format
from .cache_pools import _ceil_div, _draft_is_dcp_sharded, _draft_row_bytes_int, exact_element_bytes
from .draft_spec import effective_draft_context_cap

QUANT_GROUP = 128
NOPE_FP8_DTYPES = frozenset({"fp8", "fp8_e4m3"})
SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "c3d1d1fc1dbaee5c38c0601d23be99f14129fa46",
    "path": "vllm_ascend/models/kimi_k3/paired_sfa_cache.py",
}


@dataclass(frozen=True)
class PairedSfaGroupDescriptor:
    """Explicit group/stage/block-table/overlay binding for paired SFA pools."""

    group_id: int
    stage: int
    block_table_id: str
    overlay_region: str
    auxiliary: bool
    layers: tuple[int, ...]
    tokens_per_page: int
    bytes_per_group_page: int
    resident_tokens: int | None
    recycling_pages: int
    fixed_pages: int
    history_slot_count: int
    recycling_slot_ids: tuple[int, ...]
    draft_cp: int = 1


def sparse_plans(config, plans):
    dsa = config.get("dsa_config")
    if not dsa:
        return plans
    layers = {int(value) - 1 for value in dsa.get("dsa_layers", ())}
    full = {plan.index for plan in plans if plan.attention == "mla"}
    if not layers or not layers <= full:
        raise ModelingError("kimi_dsa_layers_must_be_one_based_full_attention_layers")
    dimensions = {"index_heads": int(dsa["index_num_heads"]),
                  "index_dim": int(dsa["index_head_dim"]),
                  "index_topk": int(dsa["index_topk"])}
    if min(dimensions.values()) <= 0:
        raise ModelingError("kimi_dsa_dimensions_must_be_positive")
    if int(dsa.get("index_topk_freq", 1)) != 1:
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
    per_layer = {
        str(k): dict(v)
        for k, v in dict(metadata.get("layer_cache_representations", {})).items()
    }
    mla_plans = [plan for plan in model.layer_plans if plan.attention == "mla"]
    sparse_plans = [
        plan for plan in model.layer_plans if plan.indexer or plan.owns_index_k
    ]
    if not mla_plans:
        # A truncated slice (for example Kimi-K3.1 ``layers=2``) may contain
        # only KDA layers: the paired MLA/indexer regions do not exist, so the
        # ordinary per-component allocator owns this model.
        metadata["declared_cache_allocator"] = metadata["cache_allocator"]
        metadata["cache_allocator"] = "component"
        metadata["cache_allocator_selection"] = (
            "selected_layers_have_no_paired_mla_regions")
        return replace(model, metadata=tuple(metadata.items()))
    kv_dtype = str(model.quant.kv_dtype).strip().lower()
    scale_override = str(
        metadata.get("kv_scale_dtype")
        or getattr(model.quant, "kv_scale_dtype", "")
        or ""
    ).strip().lower()
    mla_decls: list[dict[str, object]] = []
    if dtype_bytes(kv_dtype) == 1:
        for plan in mla_plans:
            rope = int(getattr(plan, "qk_rope_head_dim", 0) or 0)
            default_scale = (
                "fp16" if (rope == 0 and kv_dtype in NOPE_FP8_DTYPES) else "fp32"
            )
            decl = {
                "dtype": kv_dtype,
                "group_size": QUANT_GROUP,
                "scale_dtype": scale_override or default_scale,
                "suffix_elements": rope,
                "suffix_dtype": "bf16",
            }
            layer_entry = per_layer.setdefault(str(plan.index), {})
            layer_entry.setdefault("mla_kv", decl)
            mla_decls.append(dict(layer_entry["mla_kv"]))
        if mla_decls and all(d == mla_decls[0] for d in mla_decls):
            regions.setdefault("mla_kv", mla_decls[0])
    idx_dtype = str(model.quant.index_dtype).strip().lower()
    idx_decls: list[dict[str, object]] = []
    if sparse_plans and dtype_bytes(idx_dtype) == 1:
        idx_scale = (
            model.quant.index_scale_dtype
            if model.quant.index_scale_dtype not in {"", "none"} else "fp16"
        )
        for plan in sparse_plans:
            decl = {
                "dtype": idx_dtype,
                "group_size": int(plan.index_dim),
                "scale_dtype": idx_scale,
            }
            layer_entry = per_layer.setdefault(str(plan.index), {})
            layer_entry.setdefault("index_k", decl)
            idx_decls.append(dict(layer_entry["index_k"]))
        if idx_decls and all(d == idx_decls[0] for d in idx_decls):
            regions.setdefault("index_k", idx_decls[0])
    metadata["cache_representations"] = regions
    if per_layer:
        metadata["layer_cache_representations"] = per_layer
    return replace(model, metadata=tuple(metadata.items()))


def independent_draft(draft):
    return bool(getattr(draft, "independent_context", False))


def draft_row(model, draft, *, tp=1):
    return _draft_row_bytes_int(model, draft, tp=max(1, int(tp)))


def _kda_state_shards(parallel, metadata) -> int:
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    pcp = max(1, int(getattr(parallel, "pcp", 1) or 1))
    axis = str(metadata.get("kda_parallel_axis", "")).strip().lower()
    return tp * (1 if axis == "tp" else pcp)


def _kda_state_page_bytes(plan, shards: int, proposals: int) -> int:
    rec_elems = _ceil_div(int(plan.recurrent_state_elements), shards)
    rec_bytes = exact_element_bytes(rec_elems, str(plan.recurrent_state_dtype))
    conv_width = int(getattr(plan, "conv_width", 0) or 0)
    kernel = max(1, int(getattr(plan, "conv_kernel", 1) or 1))
    if conv_width > 0:
        conv_elems = _ceil_div(conv_width * (kernel - 1 + proposals), shards)
    else:
        base_conv = _ceil_div(int(plan.conv_state_elements), shards)
        slice_w = _ceil_div(base_conv, max(1, kernel - 1)) if base_conv else 0
        conv_elems = base_conv + proposals * slice_w
    conv_bytes = exact_element_bytes(conv_elems, str(plan.conv_state_dtype))
    return rec_bytes + conv_bytes


def _project_paired_sfa_stage(
    model,
    runtime,
    p,
    *,
    stage: int,
    plans: list,
    main_page: int,
    block: int,
    slots: int,
    proposals: int,
    query_tokens: int,
    draft,
    draft_kind: str,
    draft_cap: int,
    start_group_id: int,
):
    from .cache import index_row_bytes, layer_cache_components

    main = [plan for plan in plans if plan.attention == "mla"]
    recurrent = [plan for plan in plans if plan.attention == "kda"]
    draft_active = (
        draft is not None and stage == max(1, int(p.pp)) - 1
        and (proposals > 0 or bool(runtime.dspark))
    )
    draft_bytes = (
        block * draft_row(model, draft, tp=p.tp) * draft.layers
        if draft_active else 0
    )
    draft_sharded = (
        _draft_is_dcp_sharded(
            model, p, draft_kind, dcp=int(p.cp), draft_page=int(draft_bytes),
        )
        if draft_active else False
    )
    draft_cp = int(p.cp) if draft_sharded else 1
    capped_draft = draft_active and draft_bytes > 0 and 0 < draft_cap < int(runtime.max_model_len)
    split_draft = draft_active and draft_bytes > 0 and (capped_draft or draft_cp != int(p.cp))
    bundled_draft = 0 if split_draft else draft_bytes

    if runtime.recover_ssm:
        membership: dict[str, object] = {"members": [
            {"layer": plan.index, "component": component}
            for plan in plans
            for component in layer_cache_components(
                model, plan, p, runtime.max_model_len, runtime=runtime,
                query_tokens=query_tokens, capacity_padded=False)
            if component not in {"replay_records", "state_transient"}
        ]}
    else:
        membership = {"layers": [plan.index for plan in plans]}

    group_dicts: list[dict[str, object]] = []
    descriptors: list[PairedSfaGroupDescriptor] = []
    group_id = start_group_id
    if main:
        state_groups = _ceil_div(len(recurrent), len(main)) if recurrent else 0
        state_width = _ceil_div(len(recurrent), state_groups) if state_groups else 0
        main_slots = max(len(main), state_width)
        index = sum(
            index_row_bytes(model, plan.index_dim, plan=plan)
            for plan in main if plan.owns_index_k
        )
        rec_pages = state_groups * slots
        page_bytes = int(main_slots * main_page + block * index + bundled_draft)
        group_dicts.append({
            **membership,
            "tokens_per_page": block * int(p.cp),
            "bytes_per_group_page": page_bytes,
            "recycling_pages": rec_pages,
        })
        descriptors.append(PairedSfaGroupDescriptor(
            group_id=group_id,
            stage=stage,
            block_table_id=f"stage{stage}:main_paired",
            overlay_region="mla_kda_overlay" if recurrent else "mla_main",
            auxiliary=False,
            layers=tuple(plan.index for plan in plans),
            tokens_per_page=block * int(p.cp),
            bytes_per_group_page=page_bytes,
            resident_tokens=None,
            recycling_pages=rec_pages,
            fixed_pages=0,
            history_slot_count=main_slots,
            recycling_slot_ids=tuple(range(rec_pages)),
            draft_cp=draft_cp if bundled_draft else 1,
        ))
        group_id += 1
    else:
        rec_pages = slots
        page_bytes = int(len(recurrent) * main_page)
        group_dicts.append({
            **membership,
            "tokens_per_page": block * int(p.cp),
            "bytes_per_group_page": page_bytes,
            "fixed_pages": rec_pages,
            "recycling_pages": 0,
        })
        descriptors.append(PairedSfaGroupDescriptor(
            group_id=group_id,
            stage=stage,
            block_table_id=f"stage{stage}:kda_state_only",
            overlay_region="kda_state_overlay",
            auxiliary=False,
            layers=tuple(plan.index for plan in plans),
            tokens_per_page=block * int(p.cp),
            bytes_per_group_page=page_bytes,
            resident_tokens=None,
            recycling_pages=0,
            fixed_pages=rec_pages,
            history_slot_count=0,
            recycling_slot_ids=tuple(range(rec_pages)),
            draft_cp=1,
        ))
        group_id += 1
        if draft_active and draft_bytes > 0 and not split_draft:
            split_draft = True

    if split_draft:
        aux_entry: dict[str, object] = {
            "layers": [plans[-1].index],
            "auxiliary": True,
            "tokens_per_page": block * draft_cp,
            "bytes_per_group_page": int(draft_bytes),
            "recycling_pages": 0,
        }
        if capped_draft:
            aux_entry["resident_tokens"] = int(draft_cap)
        group_dicts.append(aux_entry)
        descriptors.append(PairedSfaGroupDescriptor(
            group_id=group_id,
            stage=stage,
            block_table_id=f"stage{stage}:draft_aux",
            overlay_region="draft_dedicated",
            auxiliary=True,
            layers=(plans[-1].index,),
            tokens_per_page=block * draft_cp,
            bytes_per_group_page=int(draft_bytes),
            resident_tokens=int(draft_cap) if capped_draft else None,
            recycling_pages=0,
            fixed_pages=0,
            history_slot_count=int(draft.layers),
            recycling_slot_ids=(),
            draft_cp=draft_cp,
        ))
        group_id += 1
    return group_dicts, descriptors, group_id


def _paired_sfa_stage_entries(case, layer_stage, query_tokens):
    """Build global paired-SFA membership once and project onto PP stages."""
    model, runtime = case.model, case.runtime
    if dict(model.metadata).get("cache_allocator") != "paired_sfa":
        return (), ()
    from .draft import optional_draft_spec

    p, block = runtime.parallel, int(runtime.kv_block_size)
    metadata = dict(runtime.metadata)
    mode = metadata.get("mamba_cache_mode", "align")
    if mode not in {"align", "none"}:
        raise ModelingError("paired_sfa_state_cache_mode_not_supported")
    fallback = max(0, int(query_tokens) - 1)
    draft_len = getattr(runtime.dspark, "draft_len", None)
    if draft_len is not None and "draft_proposal_tokens" not in metadata:
        fallback = max(0, int(draft_len))
    proposals = int(metadata.get(
        "num_speculative_state_blocks",
        metadata.get("draft_proposal_tokens", fallback),
    ))
    extra = int(metadata.get("num_prefill_checkpoint_blocks", 0))
    if min(proposals, extra) < 0:
        raise ModelingError("paired_sfa_negative_state_slots")
    spec_slots = 0 if runtime.recover_ssm else proposals
    slots = (2 + spec_slots + extra if mode == "align" else 1 + spec_slots)

    all_plans = tuple(model.layer_plans)
    if any(plan.attention not in {"mla", "kda"} for plan in all_plans):
        raise ModelingError("paired_sfa_requires_mla_and_optional_kda_per_stage")
    all_main = [plan for plan in all_plans if plan.attention == "mla"]
    all_recurrent = [plan for plan in all_plans if plan.attention == "kda"]
    if not all_main:
        raise ModelingError("paired_sfa_requires_mla_and_optional_kda_per_stage")
    mla_indices = {plan.index for plan in all_main}
    for plan in all_plans:
        if plan.owns_index_k and plan.index not in mla_indices:
            raise ModelingError("paired_sfa_index_owner_must_be_mla")
        if getattr(plan, "index_k_owner_layer_id", -1) >= 0 and plan.index_k_owner_layer_id not in mla_indices:
            raise ModelingError("paired_sfa_index_owner_must_be_mla")

    rows = {
        cache_format(model, "mla_kv", plan=plan).row_bytes(plan.cache_row_elements)
        for plan in all_main
    }
    if len(rows) != 1:
        raise ModelingError("paired_sfa_requires_uniform_main_pages")
    main_page = block * rows.pop()
    kda_shards = _kda_state_shards(p, metadata)
    for plan in all_recurrent:
        if _kda_state_page_bytes(plan, kda_shards, proposals) > main_page:
            raise ModelingError("paired_sfa_block_too_small_for_recurrent_page")

    draft_kind = metadata.get("draft_mode") or ("dspark" if runtime.dspark else "")
    draft = optional_draft_spec(model, draft_kind) if draft_kind else None
    draft_cap = effective_draft_context_cap(draft, runtime) if draft is not None else 0

    group_dicts: list[dict[str, object]] = []
    descriptors: list[PairedSfaGroupDescriptor] = []
    group_id = 0
    for stage in range(max(1, int(p.pp))):
        plans = [plan for plan in all_plans if layer_stage[plan.index] == stage]
        if not plans:
            continue
        stage_groups, stage_descs, group_id = _project_paired_sfa_stage(
            model, runtime, p, stage=stage, plans=plans, main_page=main_page,
            block=block, slots=slots, proposals=proposals,
            query_tokens=query_tokens, draft=draft, draft_kind=draft_kind,
            draft_cap=draft_cap, start_group_id=group_id,
        )
        group_dicts.extend(stage_groups)
        descriptors.extend(stage_descs)
    return tuple(group_dicts), tuple(descriptors)


def page_groups(case, layer_stage, query_tokens):
    """Use ordinary CachePages, with history IDs plus bounded state IDs."""
    groups, _ = _paired_sfa_stage_entries(case, layer_stage, query_tokens)
    return groups


def paired_sfa_group_descriptors(case, layer_stage, query_tokens) -> tuple[PairedSfaGroupDescriptor, ...]:
    """Explicit group, stage, block-table, and overlay descriptors for paired SFA."""
    _, descriptors = _paired_sfa_stage_entries(case, layer_stage, query_tokens)
    return descriptors


BLOCK_ALIGNMENT = 16


def minimum_block_size(
    model,
    *,
    tp: int = 1,
    proposals: int = 0,
    extra: int = 0,
    pcp: int = 1,
    kda_parallel_axis: str = "",
) -> int:
    """Minimum ``kv_block_size`` (aligned to 16) so every KDA state fits one main page.

    Extra checkpoint slots (``extra``) occupy separate recycling page IDs rather
    than widening the convolution rollback buffer; only ``proposals`` widens the
    active verify rollback window.
    """
    del extra
    if dict(model.metadata).get("cache_allocator") != "paired_sfa":
        return 0
    if not any(plan.indexer for plan in model.layer_plans):
        return 0
    normalized = normalize_formats(model)
    main = [plan for plan in normalized.layer_plans if plan.attention == "mla"]
    recurrent = [plan for plan in normalized.layer_plans if plan.attention == "kda"]
    if not main or not recurrent:
        return 0
    row_bytes = min(
        cache_format(normalized, "mla_kv", plan=plan).row_bytes(plan.cache_row_elements)
        for plan in main
    )
    if row_bytes <= 0:
        return 0
    axis = str(kda_parallel_axis or "").strip().lower()
    shards = max(1, int(tp)) * (1 if axis == "tp" else max(1, int(pcp)))
    prop = max(0, int(proposals))
    required = 0
    for plan in recurrent:
        state_page = _kda_state_page_bytes(plan, shards, prop)
        tokens = _ceil_div(state_page, row_bytes)
        if tokens > required:
            required = tokens
    return ((required + BLOCK_ALIGNMENT - 1) // BLOCK_ALIGNMENT) * BLOCK_ALIGNMENT
