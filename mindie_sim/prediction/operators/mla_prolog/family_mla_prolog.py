"""Selected Ascend MLA prolog representation over the common semantic DAG.

[Role] Declare source-required operand precision and an explicit fusion group.
[Invariants] Source selection is explicit. Unquantized UK/UV parameters remain
  unquantized; group export does not pretend its operators have fused timing.
[Timeouts & Locks] Bounded layer metadata; no device contexts, I/O or locks.
[Side-Effect Policy] Returns declarations and updates caller-owned OpSpec records.
"""
from dataclasses import replace

from mindie_sim.catalog.ops import compute_dtype_for
from mindie_sim.schema.common import ModelingError

from mindie_sim.prediction.structure_groups import declare_group


IMPLEMENTATION = "vllm_ascend_mla_prolog_v3"
ABSORBED_CATEGORY = "mla_absorb"
CACHE_GROUP_SIZE = 128
BF16_CACHE_DTYPES = frozenset({"bf16", "fp8_e4m3"})
#: Declared V3/V4 prolog Q-LoRA ranks (V4-Flash 1024, V4.1 1280, V3/R1 1536, GLM 2048).
MLA_PROLOG_QUERY_LORA_RANKS = frozenset({1024, 1280, 1536, 2048})
MXFP8_FORMAT = "fp8_e4m3"
MXFP8_QUERY_QUANT_MODE = 3
E8M0_GROUP_SIZE = 32
BF16_QUERY_QUANT_MODE = 0
SOURCE = {
    "repository": "vllm-ascend",
    "revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "path": "vllm_ascend/attention/sfa_v1.py",
    "sha256": "c4ea3f33ba6c71d9a9163dc397c12d92dba67aa26c292ab1ce52605b3c91ad42",
    "symbols": [
        "AscendSFAImpl.process_weights_after_loading",
        "_sfa_preprocess_prolog_v3",
    ],
    "lines": "596-629,960-1069",
    "boundary": "independent_source_declaration; not_capture_binary_identity",
}


def recipes():
    # The declaration retains the existing operator recipes and costs.
    return {}


def prepare_model(model, runtime):
    from mindie_sim.catalog.quant import projection_quant
    from mindie_sim.prediction.implementation_selection import prepared_selection
    options = prepared_selection(runtime.kernels.get("lowerings", {}), "mla_prolog").options
    for plan in model.layer_plans:
        if plan.attention == "mla":
            validate(plan=plan, parallel=runtime.parallel, model=model,
                     quant=projection_quant(model.quant, plan),
                     lowering_passes=runtime.lowering_passes, **options)
    if options.get("cache_layout"):
        from mindie_sim.catalog.cache_format import with_packed_mla_cache
        rope = max((int(plan.qk_rope_head_dim) for plan in model.layer_plans
                    if plan.attention == "mla"), default=0)
        model = with_packed_mla_cache(model, rope)
    # The source asserts an unquantized kv_b checkpoint. The same category
    # governs UK/UV execution and weight capacity, including source bytes.
    quant = replace(model.quant, exclusions=tuple(dict.fromkeys(
        (*model.quant.exclusions, ABSORBED_CATEGORY))))
    return replace(model, quant=quant)


DRAFT_SCOPES = frozenset({"draft", "mtp", "dspark", "standalone_draft"})


def _effective_cache_layout(plan, quant, cache_layout):
    """Unpacked draft MLA layers ignore target-forwarded packed C8 cache layout."""
    from .mla_prolog_packed import LAYOUT
    if (cache_layout == LAYOUT
            and getattr(plan, "scope", "body") in DRAFT_SCOPES
            and quant.kv_dtype in BF16_CACHE_DTYPES):
        return None
    return cache_layout


def _indexer_q_category(quant) -> str:
    overrides = dict(getattr(quant, "category_overrides", ()) or ())
    exclusions = set(getattr(quant, "exclusions", ()) or ())
    if "indexer.wq" in overrides or "indexer.wq" in exclusions:
        return "indexer.wq"
    if "indexer_fp8" in overrides or "indexer_fp8" in exclusions:
        return "indexer_fp8"
    return "indexer.wq"


def _prolog_port_formats(quant) -> tuple[tuple[str, str, str], tuple[str, str, str]]:
    qa_w = quant.format_for("mla.q_a")
    qb_w = quant.format_for("mla.q_b")
    kva_w = quant.format_for("mla.kv_a")
    qa_act = compute_dtype_for(quant, "mla.q_a")[1]
    qb_act = compute_dtype_for(quant, "mla.q_b")[1]
    kva_act = compute_dtype_for(quant, "mla.kv_a")[1]
    return (qa_w, qb_w, kva_w), (qa_act, qb_act, kva_act)


def validate(*, plan, parallel, model, quant, lowering_passes,
             cache_layout=None, cache_group_size=CACHE_GROUP_SIZE):
    from .mla_prolog_packed import (
        SUPPORTED_QUERY_FORMATS, LAYOUT, view,
    )
    cache_layout = _effective_cache_layout(plan, quant, cache_layout)
    packed = cache_layout == LAYOUT
    if cache_layout not in (None, LAYOUT) or (packed and quant.kv_dtype != "int8"):
        raise ModelingError("mla_prolog_v3_cache_layout_dtype_mismatch")
    w_ports, act_ports = _prolog_port_formats(quant)
    if len(set(w_ports)) != 1 or len(set(act_ports)) != 1:
        raise ModelingError("mla_prolog_v3_projection_port_precision_mismatch")
    proj_w, proj_act = w_ports[0], act_ports[0]
    if packed:
        # DCP keeps one rank-owned row slice; it does not change the C8 row ABI.
        view(plan.kv_lora_rank, plan.qk_rope_head_dim, cache_group_size)
        if proj_w not in SUPPORTED_QUERY_FORMATS or proj_act != proj_w:
            raise ModelingError("mla_prolog_packed_requires_declared_projection_operands")
    if (plan.mla_variant != "latent_mla" or plan.cache_kind != "dense"
            or plan.qk_rope_head_dim <= 0
            or parallel.cp < 1 or parallel.pcp < 1
            or model.activation_dtype != "bf16"
            or (quant.kv_dtype not in BF16_CACHE_DTYPES and not packed)
            or proj_w not in {"bf16", "int8", "fp8_e4m3"}):
        raise ModelingError("mla_prolog_v3_declared_operand_domain_unsupported")
    # The V3/V4 prolog owns one fused Q-A reduction; its source kernel is
    # admitted only for the two declared LoRA ranks.
    if (plan.q_lora_rank
            and plan.q_lora_rank not in MLA_PROLOG_QUERY_LORA_RANKS):
        # A direct-Q plan has no Q-LoRA prolog at all; its own builder raises
        # the direct-Q boundary instead of this rank guard.
        raise ModelingError("mla_prolog_v3_query_lora_rank_unsupported")
    if proj_act != model.activation_dtype:
        if "activation_quantization" not in lowering_passes:
            raise ModelingError("mla_prolog_v3_quantized_operands_require_activation_quantization")
        idx_q_cat = _indexer_q_category(quant)
        if ((plan.indexer or plan.owns_topk or plan.index_query_per_layer)
                and compute_dtype_for(quant, idx_q_cat)[1] != proj_act):
            # The selected source exposes the same quantized normalized query
            # and scales to q_b and index_q. A BF16 indexer needs a different
            # externally visible producer and therefore another source domain.
            raise ModelingError("mla_prolog_v3_query_norm_indexer_activation_mismatch")


def _native_kind(packed, quant):
    from .mla_prolog_packed import NATIVE_KIND_V3, NATIVE_KIND_V4, MXFP8_QUERY_FORMAT
    del packed
    w_ports, _ = _prolog_port_formats(quant)
    if all(fmt == MXFP8_QUERY_FORMAT for fmt in w_ports):
        return NATIVE_KIND_V4
    return NATIVE_KIND_V3


def declare(ops, *, prefix, plan, model, parallel, quant, tokens, members,
            q_a, q_b, kv_a, q_absorb, cache_layout=None, cache_group_size=CACHE_GROUP_SIZE):
    from .mla_prolog_packed import (
        LAYOUT, MXFP8_QUERY_FORMAT, SOURCE as PACKED_SOURCE, SOURCE_V4,
        view as packed_view,
    )
    cache_layout = _effective_cache_layout(plan, quant, cache_layout)
    packed = cache_layout == LAYOUT
    cache = packed_view(plan.kv_lora_rank, plan.qk_rope_head_dim, cache_group_size) if packed else None
    by_id = {op.op_id: op for op in ops}
    qa, qb, kva, absorb = (dict(by_id[key].metadata) for key in (q_a, q_b, kv_a, q_absorb))
    do_rope = bool(plan.mla_do_rope)
    native_kind = _native_kind(packed, quant)
    mxfp8_query = native_kind.endswith("V4WeightNz")
    query_norm_output = bool(
        plan.indexer or plan.owns_topk or plan.index_query_per_layer)
    # V4 quantizes the normalized query with group-32 E8M0 scales; V3 keeps
    # one FP32 scale per row (or no scale at all without an index consumer).
    query_norm_scale_dtype = "fp8_e8m0" if mxfp8_query else "fp32"
    query_norm_scale_shape = (
        () if not query_norm_output
        else (tokens, plan.q_lora_rank // E8M0_GROUP_SIZE) if mxfp8_query
        else (tokens, 1))
    source = SOURCE
    if mxfp8_query:
        source = SOURCE_V4
    elif packed:
        source = PACKED_SOURCE
    declare_group(
        ops, key=prefix + ".mla_prolog", role="mla_prolog", family="mla",
        members=members, entries=(q_a, kv_a), anchor=q_absorb,
        view={
            "tokens": tokens, "hidden": model.hidden_size,
            "heads": plan.num_heads // parallel.tp,
            "query_rank": plan.q_lora_rank, "latent_rank": plan.kv_lora_rank,
            "nope_width": plan.qk_nope_head_dim, "rope_width": plan.qk_rope_head_dim,
            "token_dtype": qa["activation_format"],
            "projection_weight_dtypes": (
                qa["weight_format"], qb["weight_format"], kva["weight_format"]),
            "absorbed_weight_dtype": absorb["weight_format"],
            "norm_dtype": model.activation_dtype, "cache_dtype": quant.kv_dtype,
            **({"packed_cache": cache,
                "token_scale_dtype": "fp32", "token_scale_shape": (tokens, 1),
                "projection_scale_dtype": "fp32",
                "projection_scale_shapes": ((1, plan.q_lora_rank),
                    (1, plan.num_heads // parallel.tp * (plan.qk_nope_head_dim + plan.qk_rope_head_dim)),
                    (1, plan.kv_lora_rank + plan.qk_rope_head_dim)),
                "query_norm_scale_dtype": query_norm_scale_dtype,
                "query_norm_scale_shape": query_norm_scale_shape,
                "query_nope_scale_shape": (), "rope_cache_output_shape": (),
                } if packed else {}),
            "query_norm_dtype": qb["activation_format"],
            "query_norm_output": query_norm_output,
            "do_rope": do_rope,
            "parallel_cp": int(parallel.cp),
            "query_quant_mode": (MXFP8_QUERY_QUANT_MODE
                                 if qb["activation_format"] == MXFP8_QUERY_FORMAT
                                 else BF16_QUERY_QUANT_MODE),
        },
        implementation={
            "native_kind": native_kind,
            "native_kinds": (
                (native_kind, "MlaPrologV4WeightNz", "MlaPrologV4")
                if native_kind.endswith("V4WeightNz")
                else (native_kind, "MlaPrologV3")
            ),
            "selection": IMPLEMENTATION,
            "conditions": {
                "cache_mode": "PA_BSND", "quantized_kv_cache": packed,
                **({"packed_cache": cache} if packed else {}),
                **({"mxfp8_query": True, "e8m0_group": E8M0_GROUP_SIZE}
                   if native_kind.endswith("V4WeightNz") else {}),
                "do_rope": do_rope,
                "query_rope_member": do_rope,
                "data_parallel_context": int(parallel.cp),
                "checkpoint_absorbed_weight_dtype": "bf16",
                "query_norm_output": bool(plan.indexer or plan.owns_topk or plan.index_query_per_layer),
                "timing_boundary": "expanded_semantic_operators; no_fused_timing_claim",
            },
        },
        source=source,
        fusion_status="fused_framework_enable",
        fusion_reason=("mla_prolog_v4_selected_lowering"
                       if mxfp8_query else "mla_prolog_v3_selected_lowering"),
    )


def declare_semantic(ops, *, prefix, plan, model, parallel, quant, tokens, members,
                     q_a, q_b, kv_a, q_absorb=None, anchor=None, reason="mla_prolog_unfused"):
    """Declare semantic ``mla_prolog`` subgraph even when V3/V4 fusion is not selected."""
    by_id = {op.op_id: op for op in ops}
    filtered = tuple(dict.fromkeys(op_id for op_id in members if op_id in by_id))
    if not filtered:
        return
    entries = tuple(dict.fromkeys(op_id for op_id in (q_a, kv_a) if op_id in by_id))
    if not entries:
        entries = (filtered[0],)
    target_anchor = anchor if anchor in by_id else filtered[-1]
    qa = dict(by_id[q_a].metadata) if q_a in by_id else {}
    qb = dict(by_id[q_b].metadata) if q_b in by_id else {}
    kva = dict(by_id[kv_a].metadata) if kv_a in by_id else {}
    absorb = dict(by_id[q_absorb].metadata) if q_absorb and q_absorb in by_id else {}
    native_kind = _native_kind(False, quant)
    query_norm_output = bool(
        plan.indexer or plan.owns_topk or plan.index_query_per_layer)
    default_fmt = quant.format_for("mla")
    declare_group(
        ops, key=prefix + ".mla_prolog_semantic", role="mla_prolog", family="mla",
        members=filtered, entries=entries, anchor=target_anchor,
        view={
            "tokens": tokens, "hidden": model.hidden_size,
            "heads": max(1, plan.num_heads // max(1, parallel.tp)),
            "query_rank": plan.q_lora_rank, "latent_rank": plan.kv_lora_rank,
            "nope_width": plan.qk_nope_head_dim, "rope_width": plan.qk_rope_head_dim,
            "token_dtype": qa.get("activation_format", model.activation_dtype),
            "projection_weight_dtypes": (
                qa.get("weight_format", default_fmt),
                qb.get("weight_format", default_fmt),
                kva.get("weight_format", default_fmt),
            ),
            "absorbed_weight_dtype": absorb.get("weight_format"),
            "norm_dtype": model.activation_dtype, "cache_dtype": quant.kv_dtype,
            "query_norm_dtype": qb.get("activation_format", model.activation_dtype),
            "query_norm_output": query_norm_output,
            "do_rope": bool(plan.mla_do_rope),
            "parallel_cp": int(parallel.cp),
            "query_quant_mode": (MXFP8_QUERY_QUANT_MODE
                                 if default_fmt == MXFP8_FORMAT
                                 else BF16_QUERY_QUANT_MODE),
        },
        implementation={
            "native_kind": native_kind,
            "native_kinds": (
                (native_kind, "MlaPrologV4WeightNz", "MlaPrologV4")
                if native_kind.endswith("V4WeightNz")
                else (native_kind, "MlaPrologV3")
            ),
            "selection": None,
            "conditions": {
                "cache_mode": plan.cache_kind,
                "mla_variant": plan.mla_variant,
                "do_rope": bool(plan.mla_do_rope),
                "data_parallel_context": int(parallel.cp),
                "query_norm_output": query_norm_output,
                "timing_boundary": "unfused_semantic_mla_prolog_decomposition",
            },
        },
        source={"kind": "semantic_builder",
                "module": "mindie_sim.prediction.operators.mla.mla"},
        fusion_status="unfused_decomposed",
        fusion_reason=reason,
        native_fusion_eligible=False,
    )


def cache_store(builder, *, deps, rows, plan, cache_layout=None, cache_group_size=CACHE_GROUP_SIZE):
    from .mla_prolog_packed import LAYOUT, store
    cache_layout = _effective_cache_layout(plan, builder.quant, cache_layout)
    if cache_layout == LAYOUT:
        return store(builder, "kv_store", deps, rows, plan,
                     query_format=quant_format(builder.quant, "mla"))
    return None


def quant_format(quant, category):
    from mindie_sim.catalog.ops import compute_dtype_for
    return compute_dtype_for(quant, category)[1]
