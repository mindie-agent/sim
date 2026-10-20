"""DeepSeek V4/V4.1 layer-plan and draft adapters.

[Role]
  Normalize body compression schedules, V4.1 source-owner arrays, grouped
  output, mHC, hash routing and optional MTP/DSpark specs.

[Invariants]
  * compress_ratios are sliced to the body count; draft zeros never become body.
  * V4.1 owners come from declared source arrays, not modulo arithmetic.
  * Body and optional draft/verify plans are separate.

[Timeouts & Locks]
  Pure mapping reads.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping

from .a3_catalog_config import _int, _require
from .a3_catalog_types import AuxiliarySpec, DraftSpec, merge_source
from .a3_model_types import LayerPlan
from .a3_types import A3InputError


def _deepseek_dims(c: Mapping[str, Any], *, v41: bool) -> dict[str, int]:
    h = int(_require(c, "hidden_size"))
    heads = int(_require(c, "num_attention_heads"))
    head_dim = int(_require(c, "head_dim"))
    rope = _int(c, "qk_rope_head_dim", 0)
    q_rank = _int(c, "q_lora_rank", 0)
    if q_rank <= 0:
        raise A3InputError("model_config_q_lora_rank_missing")
    has_kv_rank = "kv_lora_rank" in c and c.get("kv_lora_rank") is not None
    kv_rank = _int(c, "kv_lora_rank", head_dim) if has_kv_rank else head_dim
    qk_nope = _int(c, "qk_nope_head_dim", head_dim - rope)
    if qk_nope + rope != head_dim:
        raise A3InputError(f"model_config_mla_head_dim_mismatch:{qk_nope}+{rope}!={head_dim}")
    v_dim = _int(c, "v_head_dim", head_dim) if has_kv_rank else head_dim
    o_rank = _int(c, "o_lora_rank", 0)
    o_groups = _int(c, "o_groups", 0)
    if o_rank <= 0 or o_groups <= 0:
        raise A3InputError("model_config_grouped_output_missing")
    return {
        "h": h, "heads": heads, "head_dim": head_dim, "rope": rope,
        "q_rank": q_rank, "kv_rank": kv_rank, "qk_nope": qk_nope,
        "v_dim": v_dim, "o_rank": o_rank, "o_groups": o_groups,
        "kv_cache_elements": (kv_rank + rope) if has_kv_rank else head_dim,
        "variant": "latent_mla" if has_kv_rank else "latent_mqa",
    }


def _deepseek_plans(c: Mapping[str, Any], *, v41: bool) -> list[LayerPlan]:
    d = _deepseek_dims(c, v41=v41)
    layers = int(_require(c, "num_hidden_layers"))
    ratios = [int(v) for v in c.get("compress_ratios", ())]
    if len(ratios) < layers:
        raise A3InputError(f"model_config_compress_ratios_short:{len(ratios)}<{layers}")
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
            raise A3InputError("model_config_v41_source_layers_missing")
        if candidate < 0:
            raise A3InputError("model_config_v41_candidate_source_missing")
        for source in (kv_sources, index_sources, (candidate,)):
            if any(v < 0 or v >= layers for v in source):
                raise A3InputError("model_config_v41_source_out_of_range")
        if tuple(sorted(set(kv_sources))) != kv_sources or tuple(sorted(set(index_sources))) != index_sources:
            raise A3InputError("model_config_v41_sources_not_sorted_unique")
        if not set(kv_sources).issubset(index_sources):
            raise A3InputError("model_config_v41_kv_sources_not_index_sources")
    plans: list[LayerPlan] = []
    for index, ratio in enumerate(ratios):
        if not v41 and ratio not in {0, 4, 128}:
            raise A3InputError(f"model_config_v4_ratio_unsupported:{index}:{ratio}")
        if v41 and ratio not in {0, 1, 2}:
            raise A3InputError(f"model_config_v41_ratio_unsupported:{index}:{ratio}")
        if v41:
            kv_source = max((v for v in kv_sources if v <= index), default=-1) if ratio else -1
            index_source = max((v for v in index_sources if v <= index), default=-1) if ratio else -1
            topk_source = index_source
            owns_kv = index in kv_sources
            owns_index_k = index in kv_sources
            owns_topk = index in index_sources
            if ratio and (kv_source < 0 or index_source < 0):
                raise A3InputError(f"model_config_v41_source_missing:{index}")
            if ratio and not owns_kv and ratios[kv_source] != ratio:
                raise A3InputError(f"model_config_v41_owner_ratio_mismatch:{index}:{kv_source}")
            if not ratio and (owns_kv or owns_index_k or owns_topk):
                raise A3InputError(f"model_config_v41_local_owner_invalid:{index}")
            candidate_filter = bool(ratio and index_source > candidate)
            candidate_source = candidate if candidate_filter else -1
            indexer = owns_topk
            indexer_type = (
                "full" if owns_kv and owns_topk else ("shared" if indexer else "none")
            )
            architecture = {0: "swa_only", 1: "dsa_shared_compressed_1", 2: "dsa_shared_compressed_2"}[ratio]
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
            architecture = {0: "swa_only", 4: "dsa_compressed_4", 128: "dsa_compressed_128"}[ratio]
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
            expert_latent_size=0,
            hyper_connections=True, hc_mult=hc_mult, hc_sinkhorn_iters=hc_iters,
            native_layer_id=index, scope="body",
        ))
    return plans


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
    # The native V4.1 MTP module is the DSpark module: it uses
    # dspark_n_routed_experts, not an additional ordinary E384 branch.
    draft_experts = _int(c, "dspark_n_routed_experts", experts) if v41 else experts
    draft_topk = _int(c, "dspark_num_experts_per_tok", top_k) if v41 else top_k
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
        block_size=max(1, block), markov_rank=markov,
        confidence_enabled=bool(markov),
        confidence_executes=False,
        confidence_input_width=h + markov if markov else 0,
        target_shared_embedding=True, target_shared_lm_head=True,
        base_logits=True, markov_replicated=True,
        noncausal_query_block=True, pp_capability=(1,),
        vocab_size=_int(c, "vocab_size"),
        mhc_mult=_int(c, "hc_mult"), mhc_sinkhorn_iters=_int(c, "hc_sinkhorn_iters"),
        input_projection_category="dspark_input",
        source=merge_source({"kind": "native_mtp_dspark_selected"}),
        notes=("native se-tools MTP uses dspark expert geometry; one selected mode",),
    )
    return AuxiliarySpec(
        mtp=draft, dspark=None,
        notes=("mtp_and_dspark_names_do_not_prove_two_modes",),
    )

