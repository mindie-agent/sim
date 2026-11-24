"""Qwen3.8 hybrid and standard Qwen2/2.5/3 dense/MoE adapters.

[Role]
  Normalize hybrid layer types, unequal GDN key/value heads, packed qkvz/ba
  widths, dense/MoE FFN, the standalone DSpark draft model and the ordinary
  Qwen2/Qwen2.5/Qwen3 dense/MoE bodies.

[Invariants]
  * GDN key and value head counts/dims are independent.
  * Gated Q projection doubles the Q width without changing K/V width.
  * DSpark is recognized by its architecture, not generic model_type=qwen3.
  * Standard Qwen head expansion, QK-norm, bias gaps, MoE schedule and window
    placement come from the native config/source, not from the hybrid defaults.

[Timeouts & Locks]
  Pure mapping reads.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.catalog.config import _int, _require
from mindie_sim.catalog.draft_spec import parse_draft_max_position_embeddings
from mindie_sim.catalog.hf_config import declared_partial_rotary, rotary_dimensions
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, merge_source
from mindie_sim.schema.model import LayerPlan
from mindie_sim.schema.common import ModelingError, dtype_bytes

QWEN_PP_CAPABILITY = (1, 2, 4, 8, 16)
DEFAULT_MLA_DSPARK_BLOCK = 5


def _qwen_normalization(c: Mapping[str, Any]) -> dict[str, Any]:
    """Unplugged W5a socket — AG014 model_type normalization dispatch purged."""
    del c
    raise NotImplementedError(
        "LEGACY_PURGED: replace AG014 model_type check in _qwen_normalization "
        "with structural config predicates (e.g. apply_layernorm_1p / linear_num_key_heads)"
    )


def _qwen_output_gate(c: Mapping[str, Any]) -> bool:
    if c.get("attn_output_gate") is not None:
        return bool(c["attn_output_gate"])
    gate_type = str(c.get("output_gate_type") or "").strip().lower()
    if gate_type:
        return gate_type not in {"none", "false", "0"}
    raise NotImplementedError(
        "LEGACY_PURGED: replace AG014 model_type fallback in _qwen_output_gate "
        "with structural config predicate"
    )


def _qwen_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    h = int(_require(c, "hidden_size"))
    layers = int(_require(c, "num_hidden_layers"))
    heads = int(_require(c, "num_attention_heads"))
    kv_heads = int(_require(c, "num_key_value_heads"))
    head_dim = int(_require(c, "head_dim"))
    interval = _int(c, "full_attention_interval")
    layer_types = [str(v) for v in (c.get("layer_types") or (
        ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(layers)]
        if interval > 0 else ()
    ))]
    if len(layer_types) != layers:
        raise ModelingError(f"model_config_qwen_layer_types_mismatch:{len(layer_types)}!={layers}")
    dense_i = _int(c, "intermediate_size")
    experts = _int(c, "num_experts")
    top_k = _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    shared_i = _int(c, "shared_expert_intermediate_size")
    key_heads = int(_require(c, "linear_num_key_heads"))
    value_heads = int(_require(c, "linear_num_value_heads"))
    key_dim = int(_require(c, "linear_key_head_dim"))
    value_dim = int(_require(c, "linear_value_head_dim"))
    conv_k = int(_require(c, "linear_conv_kernel_dim"))
    conv_width = 2 * key_heads * key_dim + value_heads * value_dim
    gate = _qwen_output_gate(c)
    hc_count = _int(c, "hc_count")
    hc_lowrank = _int(c, "hc_lowrank")
    indexer_heads = _int(c, "indexer_n_heads")
    indexer_dim = _int(c, "indexer_head_dim")
    indexer_budget = _int(c, "indexer_budget")
    indexer_ratio = _int(c, "indexer_compress_ratio")
    indexer = indexer_heads > 0 and indexer_dim > 0 and indexer_budget > 0
    ssm_raw = str(c.get("mamba_ssm_dtype") or "fp32").lower()
    ssm_dtype = {
        "float32": "fp32", "fp32": "fp32",
        "bfloat16": "bf16", "bf16": "bf16",
        "float16": "fp16", "fp16": "fp16",
    }.get(ssm_raw, ssm_raw)
    moe = experts > 0
    # Qwen3-Next and Qwen3.5 MoE share SharedFusedMoE's softmax router
    # and normalize the selected expert weights, including omitted HF defaults.
    router_mode = str(c.get("scoring_func", "softmax"))
    router_norm_topk = bool(c.get("norm_topk_prob", True))
    partial = declared_partial_rotary(c)
    rope_dim, nope_dim = rotary_dimensions(head_dim, partial)
    plans: list[LayerPlan] = []
    for index, kind in enumerate(layer_types):
        del index, kind
        raise NotImplementedError(
            "LEGACY_PURGED: wire structural GDN LayerPlan.gdn_layout='interleaved' "
            "and AG014-free sparse attention kind dispatch in _qwen_plans"
        )
    return plans


def _aux_qwen(c: Mapping[str, Any]) -> AuxiliarySpec:
    nextn = _int(c, "mtp_num_hidden_layers") or _int(c, "num_nextn_predict_layers")
    if nextn <= 0:
        return AuxiliarySpec()
    h = int(c["hidden_size"])
    heads = int(c["num_attention_heads"])
    kv_heads = int(c["num_key_value_heads"])
    head_dim = int(c["head_dim"])
    gate = _qwen_output_gate(c)
    partial = declared_partial_rotary(c)
    rope_dim, nope_dim = rotary_dimensions(head_dim, partial)
    experts = _int(c, "num_experts")
    moe_i = _int(c, "moe_intermediate_size") if experts else _int(c, "intermediate_size")
    mtp = DraftSpec(
        kind="mtp", layers=nextn, hidden_size=h, intermediate_size=moe_i,
        num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
        qk_rope_head_dim=rope_dim, qk_nope_head_dim=nope_dim,
        q_width=(2 if gate else 1) * heads * head_dim,
        kv_width=kv_heads * head_dim, attention_kind="gqa",
        kv_cache_elements=2 * kv_heads * head_dim,
        input_mode="dual_normalized" if _int(c, "hc_lowrank") else "concat",
        weight_category="mtp", qk_norm=True, q_output_gate=gate,
        num_experts=experts, top_k=_int(c, "num_experts_per_tok") if experts else 0,
        expert_intermediate_size=moe_i,
        num_shared_experts=1 if experts else 0,
        block_size=1, markov_rank=0, confidence_enabled=False,
        target_shared_embedding=True, target_shared_lm_head=True,
        base_logits=True, pp_capability=QWEN_PP_CAPABILITY,
        vocab_size=int(c["vocab_size"]),
        mhc_mult=_int(c, "hc_count"),
        mhc_lowrank=_int(c, "hc_lowrank"),
        max_position_embeddings=parse_draft_max_position_embeddings(c),
        source=merge_source({"kind": "qwen_mtp"}),
        notes=("Qwen body excludes the optional MTP layer",),
    )
    return AuxiliarySpec(mtp=mtp, notes=("qwen_mtp_is_optional",))


def _dspark_mla_rank(c: Mapping[str, Any]) -> int:
    return int(c.get("kv_lora_rank") or 0)


def _dspark_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    layers = int(_require(c, "num_hidden_layers"))
    heads = int(_require(c, "num_attention_heads"))
    dense_i = int(_require(c, "intermediate_size"))
    kv_lora = _dspark_mla_rank(c)
    if kv_lora > 0:
        q_lora = int(_require(c, "q_lora_rank"))
        nope = int(_require(c, "qk_nope_head_dim"))
        rope = int(_require(c, "qk_rope_head_dim"))
        v_dim = int(_require(c, "v_head_dim"))
        head_dim = nope + rope
        kv_heads = int(c.get("num_key_value_heads", 1) or 1)
        causal = bool((c.get("dflash_config") or {}).get(
            "causal", c.get("full_attention_causal", False)))
        return [
            LayerPlan(
                index=index, attention="mla", architecture="mla", ffn="dense",
                num_heads=heads, num_kv_heads=kv_heads,
                q_lora_rank=q_lora, kv_lora_rank=kv_lora,
                qk_nope_head_dim=nope, qk_rope_head_dim=rope, v_head_dim=v_dim,
                head_dim=head_dim, mla_variant="latent_mla",
                mla_output_gate=bool(c.get("mla_use_output_gate", False)),
                packed_input_projection=True,
                kv_cache_elements=kv_lora + rope, intermediate_size=dense_i,
                noncausal_attention=not causal,
                kv_owner_layer_id=index, native_layer_id=index, scope="body",
            )
            for index in range(layers)
        ]
    _ = int(_require(c, "hidden_size"))
    kv_heads = int(_require(c, "num_key_value_heads"))
    head_dim = int(_require(c, "head_dim"))
    return [
        LayerPlan(
            index=index, attention="gqa", architecture="gqa", ffn="dense",
            num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
            intermediate_size=dense_i,
            qk_norm=True,
            noncausal_attention=True,
            kv_cache_elements=2 * kv_heads * head_dim,
            kv_owner_layer_id=index,
            native_layer_id=index, scope="body",
        )
        for index in range(layers)
    ]


def _aux_dspark(c: Mapping[str, Any]) -> AuxiliarySpec:
    h = int(c["hidden_size"])
    markov = _int(c, "markov_rank", 256)
    draft_max_len = parse_draft_max_position_embeddings(c, is_standalone_draft=True)
    kv_lora = _dspark_mla_rank(c)
    if kv_lora > 0:
        nope = int(_require(c, "qk_nope_head_dim"))
        rope = int(_require(c, "qk_rope_head_dim"))
        v_dim = int(_require(c, "v_head_dim"))
        heads = int(_require(c, "num_attention_heads"))
        q_lora = int(_require(c, "q_lora_rank"))
        head_dim = nope + rope
        targets = tuple(int(v) for v in (
            c.get("target_layer_ids")
            or (c.get("dflash_config") or {}).get("target_layer_ids")
            or (2, 23, 47, 71, 89)
        ))
        causal = bool((c.get("dflash_config") or {}).get(
            "causal", c.get("full_attention_causal", False)))
        spec = DraftSpec(
            kind="standalone_draft", layers=int(c["num_hidden_layers"]),
            hidden_size=h, intermediate_size=int(c["intermediate_size"]),
            num_heads=heads, num_kv_heads=1, head_dim=head_dim,
            q_lora_rank=q_lora, kv_lora_rank=kv_lora,
            qk_nope_head_dim=nope, qk_rope_head_dim=rope, v_head_dim=v_dim,
            kv_cache_elements=kv_lora + rope,
            q_width=heads * head_dim, kv_width=kv_lora + rope,
            attention_kind="mla", mla_variant="latent_mla",
            target_layer_ids=targets, block_size=_int(c, "block_size", DEFAULT_MLA_DSPARK_BLOCK),
            markov_rank=markov,
            confidence_enabled=bool(c.get("enable_confidence_head", False)),
            confidence_executes=False, confidence_input_width=h + markov,
            target_shared_embedding=True, target_shared_lm_head=True,
            base_logits=True, markov_replicated=True,
            noncausal_query_block=not causal, pp_capability=QWEN_PP_CAPABILITY,
            vocab_size=int(c["vocab_size"]),
            max_position_embeddings=draft_max_len,
            source=merge_source({
                "kind": "standalone_dspark_draft",
                "independent_context": "true",
                "context_projection_mode": "mla_prolog",
            }),
            notes=("DSpark MLA draft model is its own body; target features are model inputs",),
            independent_context=True,
            context_projection_mode="mla_prolog",
        )
        return AuxiliarySpec(dspark=spec, standalone_draft=True)
    targets = tuple(int(v) for v in (c.get("dflash_config", {}) or {}).get("target_layer_ids", ()) or ())
    block = _int(c, "block_size", 7)
    kv_heads = int(c["num_key_value_heads"])
    head_dim = int(c["head_dim"])
    spec = DraftSpec(
        kind="standalone_draft", layers=int(c["num_hidden_layers"]),
        hidden_size=h, intermediate_size=int(c["intermediate_size"]),
        num_heads=int(c["num_attention_heads"]),
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        kv_cache_elements=2 * kv_heads * head_dim,
        q_width=int(c["num_attention_heads"]) * head_dim,
        kv_width=kv_heads * head_dim,
        attention_kind="gqa", target_layer_ids=targets, block_size=block,
        markov_rank=markov,
        confidence_enabled=bool(c.get("enable_confidence_head", False)),
        confidence_executes=False,
        confidence_input_width=h + markov,
        target_shared_embedding=True, target_shared_lm_head=True,
        base_logits=True, markov_replicated=True,
        noncausal_query_block=True, pp_capability=QWEN_PP_CAPABILITY,
        vocab_size=int(c["vocab_size"]),
        max_position_embeddings=draft_max_len,
        source=merge_source({
            "kind": "standalone_dspark_draft",
            "independent_context": "true",
            "context_projection_mode": "fused_kv_bank",
            "dcp_sharded": "false",
        }),
        notes=("DSpark draft model is its own body; target features are model inputs",),
        independent_context=True,
        context_projection_mode="fused_kv_bank",
        dcp_sharded=False,
    )
    return AuxiliarySpec(dspark=spec, standalone_draft=True)


# --- standard dense and MoE bodies (unplugged pending AG014-free predicates) ---


def _standard_qwen_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    """Unplugged W5a socket — AG014 model_type dispatch purged."""
    del c
    raise NotImplementedError(
        "LEGACY_PURGED: replace AG014 standard model_type checks in _standard_qwen_plans "
        "with structural HF config predicates"
    )


def _aux_standard_qwen(c: Mapping[str, Any]) -> AuxiliarySpec:
    """A layer count alone does not specify a standard-Qwen draft architecture."""
    del c
    return AuxiliarySpec()


def _standard_qwen_gaps(c: Mapping[str, Any]) -> tuple[str, ...]:
    """Unplugged W5a socket — AG014 model_type gap dispatch purged."""
    del c
    raise NotImplementedError(
        "LEGACY_PURGED: replace AG014 standard model_type checks in _standard_qwen_gaps "
        "with structural HF config predicates"
    )
