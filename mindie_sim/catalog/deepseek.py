"""DeepSeek V4/V4.1 and standard V2/V2-Lite/V3/R1 adapters.

[Role]
  Normalize body compression schedules, V4.1 source-owner arrays, grouped
  output, mHC, hash routing and optional MTP/DSpark specs.  The standard
  V2/V3 family adapter reads latent-MLA geometry and the native dense/MoE
  schedule without V4 compression, indexer, mHC or grouped-output fields.

[Invariants]
  * compress_ratios are sliced to the body count; draft zeros never become body.
  * V4.1 owners come from declared source arrays, not modulo arithmetic.
  * Standard V2/V3 MoE follows ``n_routed_experts is not None`` AND
    ``layer_idx >= first_k_dense_replace`` AND
    ``layer_idx % moe_layer_freq == 0``; dense layers use the dense width.
  * Standard latent MLA has one KV head (the latent cache row); q_lora_rank
    null is a real direct-Q projection and stays 0 in the plan.
  * Body and optional draft/verify plans are separate.

[Timeouts & Locks]
  Pure mapping reads.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from mindie_sim.catalog.config import _int, _require
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, merge_source
from mindie_sim.schema.model import LayerPlan
from mindie_sim.schema.common import ModelingError


def _deepseek_dims(c: Mapping[str, Any], *, v41: bool) -> dict[str, int]:
    h = int(_require(c, "hidden_size"))
    heads = int(_require(c, "num_attention_heads"))
    head_dim = int(_require(c, "head_dim"))
    rope = _int(c, "qk_rope_head_dim", 0)
    q_rank = _int(c, "q_lora_rank", 0)
    if q_rank <= 0:
        raise ModelingError("model_config_q_lora_rank_missing")
    has_kv_rank = "kv_lora_rank" in c and c.get("kv_lora_rank") is not None
    kv_rank = _int(c, "kv_lora_rank", head_dim) if has_kv_rank else head_dim
    qk_nope = _int(c, "qk_nope_head_dim", head_dim - rope)
    if qk_nope + rope != head_dim:
        raise ModelingError(f"model_config_mla_head_dim_mismatch:{qk_nope}+{rope}!={head_dim}")
    v_dim = _int(c, "v_head_dim", head_dim) if has_kv_rank else head_dim
    o_rank = _int(c, "o_lora_rank", 0)
    o_groups = _int(c, "o_groups", 0)
    if o_rank <= 0 or o_groups <= 0:
        raise ModelingError("model_config_grouped_output_missing")
    return {
        "h": h, "heads": heads, "head_dim": head_dim, "rope": rope,
        "q_rank": q_rank, "kv_rank": kv_rank, "qk_nope": qk_nope,
        "v_dim": v_dim, "o_rank": o_rank, "o_groups": o_groups,
        "kv_cache_elements": (kv_rank + rope) if has_kv_rank else head_dim,
        "variant": "latent_mla" if has_kv_rank else "latent_mqa",
    }


#: ``moe_router_dtype == "float32"`` is a checkpoint declaration of an FP32
#: gate matrix.  ``AscendGateLinear`` Tier 4 keeps BF16 weights and accumulates
#: the Cube GEMM in FP32, so the declaration costs no input cast and no FP32
#: weight copy.  A V4/V4.1 checkpoint that does not declare it keeps the
#: retained per-family operand precision.
_FP32_ROUTER_DECLARATIONS = frozenset({"float32", "fp32"})


def _router_operands(c: Mapping[str, Any], *, v41: bool) -> dict[str, Any]:
    """Declared router operand precision for one DeepSeek family config."""
    if str(c.get("moe_router_dtype", "")).strip().lower() in _FP32_ROUTER_DECLARATIONS:
        return {"router_dtype": "", "router_output_dtype": "fp32",
                "router_precast_weight_copy": False}
    if v41:
        return {"router_dtype": "", "router_precast_weight_copy": False}
    return {"router_dtype": "fp32", "router_precast_weight_copy": True}


def _deepseek_plans(c: Mapping[str, Any], *, v41: bool) -> list[LayerPlan]:
    d = _deepseek_dims(c, v41=v41)
    layers = int(_require(c, "num_hidden_layers"))
    ratios = [int(v) for v in c.get("compress_ratios", ())]
    if len(ratios) < layers:
        raise ModelingError(f"model_config_compress_ratios_short:{len(ratios)}<{layers}")
    ratios = ratios[:layers]
    experts = _int(c, "n_routed_experts")
    top_k = _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    shared = _int(c, "n_shared_experts")
    window = _int(c, "sliding_window", 0)
    hc_mult = _int(c, "hc_mult")
    hc_iters = _int(c, "hc_sinkhorn_iters")
    hash_layers = _int(c, "num_hash_layers", 0)
    kv_sources = tuple(int(v) for v in c.get("kv_source_layer_ids", ()) or ())
    index_sources = tuple(int(v) for v in c.get("index_source_layer_ids", ()) or ())
    candidate = _int(c, "candidate_source_layer_id", -1)
    if v41:
        if not kv_sources or not index_sources:
            raise ModelingError("model_config_v41_source_layers_missing")
        if candidate < 0:
            raise ModelingError("model_config_v41_candidate_source_missing")
        for source in (kv_sources, index_sources, (candidate,)):
            if any(v < 0 or v >= layers for v in source):
                raise ModelingError("model_config_v41_source_out_of_range")
        if tuple(sorted(set(kv_sources))) != kv_sources or tuple(sorted(set(index_sources))) != index_sources:
            raise ModelingError("model_config_v41_sources_not_sorted_unique")
        if not set(kv_sources).issubset(index_sources):
            raise ModelingError("model_config_v41_kv_sources_not_index_sources")
    plans: list[LayerPlan] = []
    for index, ratio in enumerate(ratios):
        if not v41 and ratio not in {0, 4, 128}:
            raise ModelingError(f"model_config_v4_ratio_unsupported:{index}:{ratio}")
        if v41 and ratio not in {0, 1, 2}:
            raise ModelingError(f"model_config_v41_ratio_unsupported:{index}:{ratio}")
        if v41:
            kv_source = max((v for v in kv_sources if v <= index), default=-1) if ratio else -1
            index_source = max((v for v in index_sources if v <= index), default=-1) if ratio else -1
            topk_source = index_source
            owns_kv = index in kv_sources
            owns_index_k = index in kv_sources
            owns_topk = index in index_sources
            if ratio and (kv_source < 0 or index_source < 0):
                raise ModelingError(f"model_config_v41_source_missing:{index}")
            if ratio and not owns_kv and ratios[kv_source] != ratio:
                raise ModelingError(f"model_config_v41_owner_ratio_mismatch:{index}:{kv_source}")
            if not ratio and (owns_kv or owns_index_k or owns_topk):
                raise ModelingError(f"model_config_v41_local_owner_invalid:{index}")
            candidate_filter = bool(ratio and index_source > candidate)
            candidate_source = candidate if candidate_filter else -1
            indexer = owns_topk
            indexer_type = (
                "full" if owns_kv and owns_topk else ("shared" if indexer else "none")
            )
            # Paper-aligned schedule names: ratio 0 is the SWA window branch,
            # ratios 1/2 are the cross-layer shared Compressed Sparse
            # Attention v2 (candidate-block filtering) schedule.
            architecture = {0: "swa", 1: "csa2", 2: "csa2"}[ratio]
            index_status = "exact_config_source_ids"
        else:
            kv_source = index if ratio > 0 else -1
            index_source = index if ratio == 4 else -1
            topk_source = index if ratio == 4 else -1
            owns_kv = ratio > 0
            owns_index_k = ratio == 4
            owns_topk = ratio == 4
            candidate_filter = False
            candidate_source = -1
            indexer = ratio == 4
            indexer_type = "c4" if ratio == 4 else "none"
            # Paper-aligned schedule names: 4 is single-stage Compressed
            # Sparse Attention, 128 is Hierarchically Compressed Attention.
            architecture = {0: "swa", 4: "csa", 128: "hca"}[ratio]
            index_status = "exact_native_compression_schedule"
        plans.append(LayerPlan(
            index=index, attention="mla", ffn="moe",
            architecture=architecture, compress_ratio=ratio, indexer=indexer,
            indexer_type=indexer_type, candidate_filter=candidate_filter,
            window=window, num_heads=d["heads"], num_kv_heads=1,
            head_dim=d["head_dim"], q_lora_rank=d["q_rank"],
            kv_lora_rank=d["kv_rank"], qk_nope_head_dim=d["qk_nope"],
            qk_rope_head_dim=d["rope"], v_head_dim=d["v_dim"],
            o_lora_rank=d["o_rank"], o_groups=d["o_groups"],
            mla_variant=d["variant"], mla_output_gate=True,
            kv_cache_elements=d["kv_cache_elements"],
            kv_owner_layer_id=kv_source, owns_long_kv=owns_kv,
            index_k_owner_layer_id=kv_source if ratio else -1,
            owns_index_k=owns_index_k,
            topk_owner_layer_id=topk_source, owns_topk=owns_topk,
            owns_candidate_blocks=bool(v41 and index == candidate),
            candidate_source_layer_id=candidate_source,
            index_source_layer_id=index_source,
            index_source_status=index_status,
            intermediate_size=0, num_experts=experts, top_k=top_k,
            moe_intermediate_size=moe_i, num_shared_experts=shared,
            router_input_size=d["h"], router_mode=(
                "hash" if not v41 and index < hash_layers else str(c.get("scoring_func", "softmax"))),
            router_norm_topk=bool(c.get("norm_topk_prob", False)),
            **_router_operands(c, v41=v41),
            expert_latent_size=0,
            hyper_connections=True, hc_mult=hc_mult, hc_sinkhorn_iters=hc_iters,
            native_layer_id=index, scope="body",
        ))
    return plans


DSPARK_KIND = "dspark"
DSPARK_INPUT_MODE = "projected_concat"
DSPARK_SOURCE_KIND = "native_dspark_declared_separately"
DSPARK_BLOCK_THRESHOLD = 1
DSPARK_PP_CAPABILITY = (1, 2, 4, 8, 16)
DSV41_DSPARK_SOURCE: dict[str, str] = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/models/deepseek_v41/dspark.py",
    "source_sha256": "2439810997a297b404d814fcf934a367f3daa3a937f594aa808c9204b3165379",
}
DSV4_DSPARK_SOURCE: dict[str, str] = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/models/deepseek_v4/model.py",
    "source_sha256": "4327e0360c38b63898f107cdb70569ff2aa2bd10099bf324183ca57a3da31f0d",
}


def _draft_compat_kwargs(**candidates: Any) -> dict[str, Any]:
    declared = DraftSpec.__dataclass_fields__
    return {key: value for key, value in candidates.items() if key in declared}


def _aux_deepseek(c: Mapping[str, Any], *, v41: bool, d: Mapping[str, Any]) -> AuxiliarySpec:
    h = int(c["hidden_size"])
    experts = _int(c, "n_routed_experts")
    top_k = _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    shared = _int(c, "n_shared_experts")
    targets = tuple(int(v) for v in c.get("dspark_target_layer_ids", ()) or ())
    block = _int(c, "dspark_block_size", 1)
    markov = _int(c, "dspark_markov_rank", 0)
    nextn = _int(c, "num_nextn_predict_layers", 0)
    layers = len(targets) if targets else nextn
    if layers <= 0:
        return AuxiliarySpec()
    body_layers = int(_require(c, "num_hidden_layers"))
    ratios = [int(v) for v in (c.get("compress_ratios") or ())]
    draft_ratios = (
        tuple(ratios[body_layers : body_layers + layers])
        if len(ratios) >= body_layers + layers else ()
    )
    # The native V4.1 MTP module is the DSpark module: it uses
    # dspark_n_routed_experts, not an additional ordinary E384 branch.
    draft_experts = _int(c, "dspark_n_routed_experts", experts) if v41 else experts
    draft_topk = _int(c, "dspark_num_experts_per_tok", top_k) if v41 else top_k
    # The V4/V4.1 auxiliary module is the native DSpark block: it consumes the
    # concatenated target features through one ``main_proj[H, len(targets)*H]``
    # projection and one ``main_norm[H]``, not two H x H projections.  It is a
    # declared DSpark mode selected explicitly through ``draft_kind`` and stays
    # distinct from an ordinary NextN module.
    dspark_mode = bool(block > DSPARK_BLOCK_THRESHOLD or markov)
    dspark_source = DSV41_DSPARK_SOURCE if v41 else DSV4_DSPARK_SOURCE
    draft = DraftSpec(
        kind="mtp_dspark", layers=layers, hidden_size=h,
        intermediate_size=moe_i, num_heads=d["heads"], num_kv_heads=1,
        head_dim=d["head_dim"], q_width=d["heads"] * d["head_dim"],
        kv_width=d["head_dim"], attention_kind="mla",
        q_lora_rank=d["q_rank"], kv_lora_rank=d["kv_rank"],
        qk_nope_head_dim=d["qk_nope"], qk_rope_head_dim=d["rope"],
        v_head_dim=d["v_dim"], o_lora_rank=d["o_rank"], o_groups=d["o_groups"],
        num_experts=draft_experts, top_k=draft_topk,
        expert_intermediate_size=moe_i,
        num_shared_experts=shared if draft_experts else 0,
        target_layer_ids=targets or tuple(range(max(0, layers))),
        draft_compress_ratios=draft_ratios,
        sliding_window=_int(c, "sliding_window", 0),
        block_size=max(1, block), markov_rank=markov,
        confidence_enabled=bool(markov),
        confidence_executes=False,
        confidence_input_width=h + markov if markov else 0,
        target_shared_embedding=True, target_shared_lm_head=True,
        mla_variant=d["variant"], kv_cache_elements=d["kv_cache_elements"],
        input_mode="dual_normalized" if _int(c, "hc_mult") else "concat",
        base_logits=True, markov_replicated=True,
        noncausal_query_block=True, pp_capability=DSPARK_PP_CAPABILITY,
        vocab_size=_int(c, "vocab_size"),
        mhc_mult=_int(c, "hc_mult"), mhc_sinkhorn_iters=_int(c, "hc_sinkhorn_iters"),
        input_projection_category="dspark_input",
        source=merge_source(
            dspark_source,
            {
                "kind": "native_mtp_dspark_selected",
                "independent_context": "false",
                "context_projection_mode": "shared_compressed_kv",
            },
        ),
        notes=("native se-tools MTP uses dspark expert geometry; one selected mode",),
        **_draft_compat_kwargs(
            independent_context=False,
            context_projection_mode="shared_compressed_kv",
        ),
    )
    if dspark_mode:
        # The declared auxiliary for a multi-token block IS the native DSpark
        # module: one ``main_proj`` over the concatenated targets, not two H x H
        # NextN projections.  It is selected explicitly through ``draft_kind``.
        draft = replace(
            draft, kind=DSPARK_KIND, input_mode=DSPARK_INPUT_MODE,
            source=merge_source(
                dspark_source,
                {
                    "kind": DSPARK_SOURCE_KIND,
                    "independent_context": "false",
                    "context_projection_mode": "shared_compressed_kv",
                },
            ))
    return AuxiliarySpec(
        mtp=draft, dspark=None,
        notes=("mtp_and_dspark_names_do_not_prove_two_modes",),
    )


def _standard_deepseek_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    """Body plans for the native DeepSeek V2/V2-Lite/V3/R1 checkpoints."""
    layers = int(_require(c, "num_hidden_layers"))
    h = int(_require(c, "hidden_size"))
    heads = int(_require(c, "num_attention_heads"))
    nope = int(_require(c, "qk_nope_head_dim"))
    rope = int(_require(c, "qk_rope_head_dim"))
    v_dim = int(_require(c, "v_head_dim"))
    kv_rank = int(_require(c, "kv_lora_rank"))
    q_raw = c.get("q_lora_rank")
    q_rank = 0 if q_raw is None else int(q_raw)
    if min(nope, rope, v_dim, kv_rank) <= 0 or q_rank < 0:
        raise ModelingError("model_config_deepseek_standard_mla_dims_invalid")
    qk_width = nope + rope
    dense_i = int(_require(c, "intermediate_size"))
    first_dense = int(_int(c, "first_k_dense_replace", 0))
    freq = int(_int(c, "moe_layer_freq", 1))
    if freq <= 0:
        raise ModelingError("model_config_moe_layer_freq_invalid")
    experts_raw = c.get("n_routed_experts")
    experts = _int(c, "n_routed_experts") if experts_raw is not None else 0
    top_k = _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    shared = _int(c, "n_shared_experts")
    scoring = str(c.get("scoring_func", "softmax"))
    if scoring not in {"softmax", "sigmoid"}:
        raise ModelingError(f"model_config_deepseek_scoring_func_unsupported:{scoring}")
    idx_topk = _int(c, "index_topk")
    idx_heads = _int(c, "index_n_heads")
    idx_dim = _int(c, "index_head_dim")
    has_indexer = bool(idx_topk > 0 and idx_heads > 0 and idx_dim > 0 and q_rank > 0)
    plans: list[LayerPlan] = []
    for index in range(layers):
        moe = experts_raw is not None and index >= first_dense and index % freq == 0
        if moe and (experts <= 0 or top_k <= 0 or moe_i <= 0):
            raise ModelingError(f"model_config_deepseek_moe_dims_invalid:{index}")
        plans.append(LayerPlan(
            index=index, attention="mla", architecture="mla_standard",
            ffn="moe" if moe else "dense",
            num_heads=heads, num_kv_heads=1, head_dim=qk_width,
            q_lora_rank=q_rank, kv_lora_rank=kv_rank,
            qk_nope_head_dim=nope, qk_rope_head_dim=rope, v_head_dim=v_dim,
            mla_variant="latent_mla", kv_cache_elements=kv_rank + rope,
            kv_owner_layer_id=index, owns_long_kv=True,
            indexer=has_indexer, indexer_type="full" if has_indexer else "none",
            index_topk=idx_topk if has_indexer else 0,
            index_heads=idx_heads if has_indexer else 0,
            index_dim=idx_dim if has_indexer else 0,
            owns_index_k=has_indexer, owns_topk=has_indexer,
            index_k_owner_layer_id=index if has_indexer else -1,
            topk_owner_layer_id=index if has_indexer else -1,
            index_query_per_layer=has_indexer,
            intermediate_size=0 if moe else dense_i,
            num_experts=experts if moe else 0,
            top_k=top_k if moe else 0,
            moe_intermediate_size=moe_i if moe else 0,
            num_shared_experts=shared if moe else 0,
            shared_intermediate_size=shared * moe_i if moe else 0,
            router_input_size=h, router_mode=scoring, router_dtype="fp32",
            router_norm_topk=bool(c.get("norm_topk_prob", False)),
            router_groups=_int(c, "n_group"),
            router_group_topk=_int(c, "topk_group"),
            router_noaux_tc=str(c.get("topk_method", "")) == "noaux_tc",
            routed_scaling_factor=_routed_scaling_factor(c),
            native_layer_id=index, scope="body",
        ))
    return plans


def _routed_scaling_factor(c: Mapping[str, Any]) -> float:
    """Declared post-selection weight scale (1.0 keeps raw scores)."""
    factor = c.get("routed_scaling_factor")
    if factor is None:
        return 1.0
    return float(factor)


def _aux_standard_deepseek(c: Mapping[str, Any]) -> AuxiliarySpec | None:
    """Native next-token modules are separate; no reliable body aux yet."""
    if _int(c, "num_nextn_predict_layers") > 0:
        return None
    return AuxiliarySpec()


def _standard_deepseek_gaps(c: Mapping[str, Any]) -> tuple[str, ...]:
    gaps: list[str] = []
    if bool(c.get("attention_bias", False)):
        gaps.append("attention_bias_not_in_layer_plan")
    # ``noaux_tc`` and its routed scaling factor are declared on every MoE
    # LayerPlan (and its draft layers); no separate coverage gap remains.
    if _int(c, "num_nextn_predict_layers") > 0:
        gaps.append("optional_mtp_not_modeled")
    return tuple(gaps)
