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

from .structure_groups import declare_group


IMPLEMENTATION = "vllm_ascend_mla_prolog_v3"
ABSORBED_CATEGORY = "mla_absorb"
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
    from .implementations import select
    options = select(runtime.kernels.get("lowerings", {}), "mla_prolog").options
    for plan in model.layer_plans:
        if plan.attention == "mla":
            validate(plan=plan, parallel=runtime.parallel, model=model, quant=model.quant,
                     lowering_passes=runtime.lowering_passes, **options)
    if options.get("cache_layout"):
        from mindie_sim.catalog.cache_format import with_packed_mla_cache
        model = with_packed_mla_cache(model, 64)
    # The source asserts an unquantized kv_b checkpoint. The same category
    # governs UK/UV execution and weight capacity, including source bytes.
    quant = replace(model.quant, exclusions=tuple(dict.fromkeys(
        (*model.quant.exclusions, ABSORBED_CATEGORY))))
    return replace(model, quant=quant)


def validate(*, plan, parallel, model, quant, lowering_passes,
             cache_layout=None, cache_group_size=128):
    from .mla_prolog_packed import LAYOUT, view
    packed = cache_layout == LAYOUT
    if cache_layout not in (None, LAYOUT) or (packed and quant.kv_dtype != "int8"):
        raise ModelingError("mla_prolog_v3_cache_layout_dtype_mismatch")
    if packed:
        view(plan.kv_lora_rank, plan.qk_rope_head_dim, cache_group_size)
        if compute_dtype_for(quant, "mla")[1] != "int8" or quant.format_for("mla") != "int8":
            raise ModelingError("mla_prolog_packed_requires_int8_projection_operands")
    if (plan.mla_variant != "latent_mla" or plan.cache_kind != "dense"
            or parallel.cp != 1 or parallel.pcp != 1
            or model.activation_dtype != "bf16" or (quant.kv_dtype != "bf16" and not packed)
            or quant.format_for("mla") not in {"bf16", "int8", "fp8_e4m3"}):
        raise ModelingError("mla_prolog_v3_declared_operand_domain_unsupported")
    projection_activation = compute_dtype_for(quant, "mla")[1]
    if projection_activation != model.activation_dtype:
        if "activation_quantization" not in lowering_passes:
            raise ModelingError("mla_prolog_v3_quantized_operands_require_activation_quantization")
        if ((plan.indexer or plan.owns_topk or plan.index_query_per_layer)
                and compute_dtype_for(quant, "indexer_fp8")[1] != projection_activation):
            # The selected source exposes the same quantized normalized query
            # and scales to q_b and index_q. A BF16 indexer needs a different
            # externally visible producer and therefore another source domain.
            raise ModelingError("mla_prolog_v3_query_norm_indexer_activation_mismatch")


def declare(ops, *, prefix, plan, model, parallel, quant, tokens, members,
            q_a, q_b, kv_a, q_absorb, cache_layout=None, cache_group_size=128):
    from .mla_prolog_packed import LAYOUT, SOURCE as PACKED_SOURCE, view as packed_view
    packed = cache_layout == LAYOUT
    cache = packed_view(plan.kv_lora_rank, plan.qk_rope_head_dim, cache_group_size) if packed else None
    by_id = {op.op_id: op for op in ops}
    qa, qb, kva, absorb = (dict(by_id[key].metadata) for key in (q_a, q_b, kv_a, q_absorb))
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
                "query_norm_scale_dtype": "fp32",
                "query_norm_scale_shape": ((tokens, 1) if
                    (plan.indexer or plan.owns_topk or plan.index_query_per_layer) else ()),
                "query_nope_scale_shape": (), "rope_cache_output_shape": (),
                } if packed else {}),
            "query_norm_dtype": qb["activation_format"],
            "query_norm_output": bool(plan.indexer or plan.owns_topk or plan.index_query_per_layer),
        },
        implementation={
            "native_kind": "MlaPrologV3", "selection": IMPLEMENTATION,
            "conditions": {
                "cache_mode": "PA_BSND", "quantized_kv_cache": packed,
                **({"packed_cache": cache} if packed else {}),
                "checkpoint_absorbed_weight_dtype": "bf16",
                "query_norm_output": bool(plan.indexer or plan.owns_topk or plan.index_query_per_layer),
                "timing_boundary": "expanded_semantic_operators; no_fused_timing_claim",
            },
        },
        source=PACKED_SOURCE if packed else SOURCE,
    )


def cache_store(builder, *, deps, rows, plan, cache_layout=None, cache_group_size=128):
    from .mla_prolog_packed import LAYOUT, store
    if cache_layout == LAYOUT:
        return store(builder, "kv_store", deps, rows, plan)
    return None
