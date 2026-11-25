"""Selected Ascend MLA prolog representation over the common semantic DAG.

[Role] Declare source-required operand precision and an explicit fusion group.
[Invariants] Source selection is explicit. Unquantized UK/UV parameters remain
  unquantized; group export does not pretend its operators have fused timing.
[Timeouts & Locks] Bounded layer metadata; no device contexts, I/O or locks.
[Side-Effect Policy] Returns declarations and updates caller-owned OpSpec records.
"""
from dataclasses import replace
from typing import Literal

from mindie_sim.catalog.ops import compute_dtype_for
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef

from mindie_sim.prediction.structure_groups import declare_group


TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCE_REF = SourceRef(
    repository="vllm-project/vllm-ascend",
    revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
    relative_path="vllm_ascend/attention/sfa_v1.py",
    source_sha256="058aeb723a4ef3cdb0e07a5fc0180ec53f8227f62d98710144ecffb9cc66678f",
)
IMPLEMENTATION = "vllm_ascend_mla_prolog_v3"
ABSORBED_CATEGORY = "mla_absorb"
CACHE_GROUP_SIZE = 128
BF16_CACHE_DTYPES = frozenset({"bf16", "fp8_e4m3"})
#: CANN 9.3 V3/V4 prolog shape invariants (mla_prolog_tiling_check.cpp:248,259,318,340).
MLA_PROLOG_QUERY_LORA_RANKS = frozenset({1536, 2048})
MLA_PROLOG_KV_LORA_RANK = 512
MLA_PROLOG_NOPE_HEAD_DIMS = frozenset({128, 192})
MLA_PROLOG_ROPE_HEAD_DIM = 64
ARCH_A2_A3 = frozenset({"A2", "A3"})
SUPPORTED_PROLOG_ARCHS = frozenset({"A2", "A3", "A5"})
MXFP8_FORMAT = "fp8_e4m3"
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


def _resolve_arch(*, runtime=None, model=None, arch=None) -> str:
    if arch:
        return str(arch).strip().upper()
    for carrier in (runtime, model):
        meta = dict(getattr(carrier, "metadata", ()) or ()) if carrier is not None else {}
        raw = meta.get("arch_family")
        if raw:
            return str(raw).strip().upper()
    return ""


def prepare_model(model, runtime):
    from mindie_sim.catalog.quant import projection_quant
    from mindie_sim.prediction.implementation_selection import prepared_selection
    options = prepared_selection(runtime.kernels.get("lowerings", {}), "mla_prolog").options
    arch = _resolve_arch(runtime=runtime, model=model)
    for plan in model.layer_plans:
        if plan.attention == "mla":
            validate(plan=plan, parallel=runtime.parallel, model=model,
                     quant=projection_quant(model.quant, plan),
                     lowering_passes=runtime.lowering_passes, arch=arch, **options)
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


def _resolved_projection_format(quant) -> tuple[str, str]:
    w_ports, act_ports = _prolog_port_formats(quant)
    if len(set(w_ports)) != 1 or len(set(act_ports)) != 1:
        raise ModelingError("mla_prolog_v3_projection_port_precision_mismatch")
    return w_ports[0], act_ports[0]


def validate(*, plan, parallel, model, quant, lowering_passes,
             cache_layout=None, cache_group_size=CACHE_GROUP_SIZE, arch=None):
    from .mla_prolog_packed import (
        SUPPORTED_QUERY_FORMATS, LAYOUT, view,
    )
    cache_layout = _effective_cache_layout(plan, quant, cache_layout)
    packed = cache_layout == LAYOUT
    if cache_layout not in (None, LAYOUT):
        raise ModelingError("mla_prolog_v3_cache_layout_dtype_mismatch")
    proj_w, proj_act = _resolved_projection_format(quant)
    arch_norm = _resolve_arch(model=model, arch=arch)
    if arch_norm:
        if arch_norm not in SUPPORTED_PROLOG_ARCHS:
            raise ModelingError(f"mla_prolog_v3_arch_unsupported:{arch_norm}")
        if arch_norm in ARCH_A2_A3:
            if proj_w == MXFP8_FORMAT or proj_act == MXFP8_FORMAT:
                raise ModelingError("mla_prolog_v4_mxfp8_requires_a5")
            if str(quant.kv_dtype or "").strip().lower().startswith("fp8"):
                raise ModelingError("mla_prolog_v3_fp8_kv_requires_a5")
    allowed_packed_kv = {"int8", MXFP8_FORMAT} if proj_w == MXFP8_FORMAT else {"int8"}
    if packed and quant.kv_dtype not in allowed_packed_kv:
        raise ModelingError("mla_prolog_v3_cache_layout_dtype_mismatch")
    if packed:
        # DCP keeps one rank-owned row slice; it does not change the C8 row ABI.
        view(plan.kv_lora_rank, plan.qk_rope_head_dim, cache_group_size, query_format=proj_act)
        if proj_w not in SUPPORTED_QUERY_FORMATS or proj_act != proj_w:
            raise ModelingError("mla_prolog_packed_requires_declared_projection_operands")
    if (plan.mla_variant != "latent_mla" or plan.cache_kind != "dense"
            or plan.kv_lora_rank != MLA_PROLOG_KV_LORA_RANK
            or plan.qk_nope_head_dim not in MLA_PROLOG_NOPE_HEAD_DIMS
            or plan.qk_rope_head_dim != MLA_PROLOG_ROPE_HEAD_DIM
            or parallel.cp < 1 or parallel.pcp < 1
            or model.activation_dtype != "bf16"
            or (quant.kv_dtype not in BF16_CACHE_DTYPES and not packed)
            or proj_w not in {"bf16", "int8", "fp8_e4m3"}):
        raise ModelingError("mla_prolog_v3_declared_operand_domain_unsupported")
    # The V3/V4 prolog owns one fused Q-A reduction; its source kernel is
    # admitted only for CANN's declared LoRA ranks (1536, 2048).
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
        LAYOUT, SOURCE as PACKED_SOURCE, SOURCE_V4,
        packed_operand_scales, view as packed_view,
    )
    cache_layout = _effective_cache_layout(plan, quant, cache_layout)
    packed = cache_layout == LAYOUT
    by_id = {op.op_id: op for op in ops}
    qa, qb, kva, absorb = (dict(by_id[key].metadata) for key in (q_a, q_b, kv_a, q_absorb))
    proj_act = _resolved_projection_format(quant)[1]
    local_heads = plan.num_heads // parallel.tp
    cache = (
        packed_view(plan.kv_lora_rank, plan.qk_rope_head_dim, cache_group_size, query_format=proj_act)
        if packed else None
    )
    packed_scales: dict[str, object] = {}
    if packed:
        tok_s_dt, tok_s_sh, proj_s_dt, proj_s_sh = packed_operand_scales(
            tokens=tokens, hidden=model.hidden_size, local_heads=local_heads,
            q_lora_rank=plan.q_lora_rank, kv_lora_rank=plan.kv_lora_rank,
            qk_nope_head_dim=plan.qk_nope_head_dim, qk_rope_head_dim=plan.qk_rope_head_dim,
            query_format=proj_act,
        )
        packed_scales = {
            "packed_cache": cache,
            "token_scale_dtype": tok_s_dt, "token_scale_shape": tok_s_sh,
            "projection_scale_dtype": proj_s_dt, "projection_scale_shapes": proj_s_sh,
        }
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
            "heads": local_heads,
            "query_rank": plan.q_lora_rank, "latent_rank": plan.kv_lora_rank,
            "nope_width": plan.qk_nope_head_dim, "rope_width": plan.qk_rope_head_dim,
            "token_dtype": qa["activation_format"],
            "projection_weight_dtypes": (
                qa["weight_format"], qb["weight_format"], kva["weight_format"]),
            "absorbed_weight_dtype": absorb["weight_format"],
            "norm_dtype": model.activation_dtype, "cache_dtype": quant.kv_dtype,
            **({**packed_scales,
                "query_norm_scale_dtype": query_norm_scale_dtype,
                "query_norm_scale_shape": query_norm_scale_shape,
                "query_nope_scale_shape": (), "rope_cache_output_shape": (),
                } if packed else {}),
            "query_norm_dtype": qb["activation_format"],
            "query_norm_output": query_norm_output,
            "do_rope": do_rope,
            "parallel_cp": int(parallel.cp),
            "query_quant_mode": BF16_QUERY_QUANT_MODE,
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
            "query_quant_mode": BF16_QUERY_QUANT_MODE,
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
        _, proj_act = _resolved_projection_format(builder.quant)
        return store(builder, "kv_store", deps, rows, plan,
                     query_format=proj_act)
    return None


def quant_format(quant, category):
    from mindie_sim.catalog.ops import compute_dtype_for
    return compute_dtype_for(quant, category)[1]
