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


def _has_linear_attention_config(c: Mapping[str, Any]) -> bool:
    """True when the HF config declares GDN / linear-attention head geometry."""
    return any(
        key in c
        for key in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_conv_kernel_dim",
        )
    )


def _qwen_gdn_layout(c: Mapping[str, Any]) -> str:
    """Resolve structural GDN projection layout (interleaved QKVZ/BA vs standard)."""
    explicit = str(c.get("gdn_layout") or "").strip().lower()
    if explicit:
        return explicit
    if c.get("linear_expand_v") is not None or bool(c.get("apply_layernorm_1p")):
        return "interleaved"
    return "standard"


def _qwen_normalization(c: Mapping[str, Any]) -> dict[str, Any]:
    """Structural unit-offset RMSNorm declaration derived strictly from config keys."""
    uses_unit_offset = bool(c.get("apply_layernorm_1p")) or (
        _has_linear_attention_config(c)
        and _int(c, "hc_lowrank") <= 0
        and not c.get("ple_layer_ids")
    )
    if not uses_unit_offset:
        return {}
    return {
        "norm_weight_offset": 1.0,
        "norm_multiply_before_output_cast": True,
        "source": {
            "repository": "vllm",
            "revision": "752a3a504485790a2e8491cacbb35c137339ad34",
            "model_path": "vllm/model_executor/models/qwen3_5.py",
            "model_sha256": "5f47ae4f4a08d0a78dd681d58b290f3298744c73a82f1349f3e2853469ef73e6",
            "model_lines": "39,165-170,248",
            "normalization_path": "vllm/model_executor/layers/layernorm.py",
            "normalization_sha256": "696615d858292fea7a2942480803cd4c7d2d1e88d818bfe5cd9d695ac464f37c",
            "normalization_lines": "129-157",
            "boundary": "declared_model_semantics; not_capture_binary_identity",
        },
    }


def _qwen_output_gate(c: Mapping[str, Any]) -> bool:
    if c.get("attn_output_gate") is not None:
        return bool(c["attn_output_gate"])
    gate_type = str(c.get("output_gate_type") or "").strip().lower()
    if gate_type:
        return gate_type not in {"none", "false", "0"}
    return _has_linear_attention_config(c)


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
    norm_unit_offset = bool(_qwen_normalization(c).get("norm_weight_offset", 0.0) == 1.0)
    has_sandwich_norm = bool(c.get("has_sandwich_norm", False))
    v_norm = bool(c.get("v_norm", False))
    gdn_layout = _qwen_gdn_layout(c)
    plans: list[LayerPlan] = []
    for index, kind in enumerate(layer_types):
        if kind == "linear_attention":
            plans.append(LayerPlan(
                index=index, attention="gdn", architecture="gdn",
                ffn="moe" if moe else "dense",
                num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
                intermediate_size=dense_i if not moe else 0,
                num_experts=experts if moe else 0, top_k=top_k if moe else 0,
                moe_intermediate_size=moe_i if moe else 0,
                num_shared_experts=1 if moe else 0,
                shared_intermediate_size=shared_i if moe else 0,
                shared_expert_gate=bool(moe),
                router_input_size=h, router_mode=router_mode,
                router_norm_topk=router_norm_topk,
                linear_heads=value_heads, linear_head_dim=value_dim,
                linear_key_heads=key_heads, linear_key_head_dim=key_dim,
                linear_value_heads=value_heads,
                conv_kernel=conv_k, conv_width=conv_width,
                recurrent_state_elements=value_heads * key_dim * value_dim,
                recurrent_state_dtype=ssm_dtype,
                conv_state_elements=conv_width * (conv_k - 1),
                conv_state_dtype="bf16",
                state_bytes=float(value_heads * key_dim * value_dim * dtype_bytes(ssm_dtype)
                                  + conv_width * (conv_k - 1) * dtype_bytes("bf16")),
                q_output_gate=gate,
                hyper_connections=hc_count > 0, hc_mult=hc_count, hc_lowrank=hc_lowrank,
                has_sandwich_norm=has_sandwich_norm,
                norm_unit_offset=norm_unit_offset,
                v_norm=v_norm,
                gdn_layout=gdn_layout,
                kv_owner_layer_id=index,
                native_layer_id=index, scope="body",
            ))
        elif kind in {"full_attention", "indexed_attention"} or kind.endswith("_sparse_attention"):
            plans.append(LayerPlan(
                index=index, attention="gqa", architecture="gated_gqa" if gate else "gqa",
                ffn="moe" if moe else "dense",
                num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
                intermediate_size=dense_i if not moe else 0,
                num_experts=experts if moe else 0, top_k=top_k if moe else 0,
                moe_intermediate_size=moe_i if moe else 0,
                num_shared_experts=1 if moe else 0,
                shared_intermediate_size=shared_i if moe else 0,
                shared_expert_gate=bool(moe),
                router_input_size=h, router_mode=router_mode,
                router_norm_topk=router_norm_topk,
                qk_norm=True,
                qk_rope_head_dim=rope_dim, qk_nope_head_dim=nope_dim,
                q_output_gate=gate,
                hyper_connections=hc_count > 0, hc_mult=hc_count, hc_lowrank=hc_lowrank,
                has_sandwich_norm=has_sandwich_norm,
                norm_unit_offset=norm_unit_offset,
                v_norm=v_norm,
                indexer=indexer, indexer_type="full" if indexer else "none",
                owns_index_k=indexer, index_k_owner_layer_id=index if indexer else -1,
                owns_topk=indexer, topk_owner_layer_id=index if indexer else -1,
                index_heads=indexer_heads if indexer else 0,
                index_dim=indexer_dim if indexer else 0,
                index_topk=indexer_budget if indexer else 0,
                index_kpool=max(1, indexer_ratio) if indexer else 1,
                index_kpool_compress=indexer and indexer_ratio > 1,
                kv_cache_elements=2 * kv_heads * head_dim,
                kv_owner_layer_id=index,
                native_layer_id=index, scope="body",
            ))
        else:
            raise ModelingError(f"model_config_qwen_layer_type_unsupported:{index}:{kind}")
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


# --- standard dense and MoE GQA bodies (pure structural predicates) --------


def _has_modern_standard_gqa_schema(c: Mapping[str, Any]) -> bool:
    """True for configs declaring modern explicit attention_bias / QK-norm schema."""
    return ("attention_bias" in c) or ("use_qk_norm" in c) or ("qk_norm" in c)


def _standard_qwen_head_dim(
    c: Mapping[str, Any], *, legacy_shared_moe: bool, hidden: int, heads: int,
) -> int:
    """Resolve explicit head_dim or derive hidden/heads for legacy shared-expert MoE."""
    if heads <= 0:
        raise ModelingError(f"model_config_qwen_heads_invalid:{heads}")
    raw = c.get("head_dim")
    if legacy_shared_moe or raw is None:
        if hidden % heads:
            raise ModelingError(
                f"model_config_qwen_head_dim_not_integral:{hidden}%{heads}")
        return hidden // heads
    head_dim = int(raw)
    if head_dim <= 0:
        raise ModelingError(f"model_config_qwen_head_dim_invalid:{head_dim}")
    return head_dim


def _standard_qwen_mlp_only(c: Mapping[str, Any]) -> tuple[set[int], tuple[str, ...]]:
    raw = c.get("mlp_only_layers")
    if raw is None:
        return set(), ()
    try:
        return {int(value) for value in raw}, ()
    except (TypeError, ValueError):
        return set(), ("mlp_only_layers_invalid",)


def _standard_qwen_ffn_config(
    c: Mapping[str, Any], *, legacy_shared_moe: bool,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Decode only declared standard GQA/MoE FFN fields via structural keys."""
    dense_i = _int(c, "intermediate_size")
    experts = _int(c, "num_experts")
    top_k = _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    step = _int(c, "decoder_sparse_step", 1)
    if experts > 0 and step <= 0:
        raise ModelingError(f"model_config_qwen_decoder_sparse_step_invalid:{step}")
    mlp_only, issues = _standard_qwen_mlp_only(c)
    shared_count = 0
    shared_i = 0
    shared_gate = False
    if experts > 0 and legacy_shared_moe:
        shared_i = _int(c, "shared_expert_intermediate_size")
        shared_count = 1 if shared_i > 0 else 0
        shared_gate = shared_i > 0
    elif experts > 0:
        shared_count = _int(c, "num_shared_experts", _int(c, "n_shared_experts"))
        shared_i = _int(c, "shared_expert_intermediate_size")
        if shared_count > 0 and shared_i <= 0:
            shared_i = shared_count * moe_i
        elif shared_count <= 0 and shared_i > 0:
            shared_count = 1
        if shared_count <= 0:
            shared_i = 0
        shared_gate = bool(c.get("shared_expert_gate", False)) if shared_i > 0 else False
    return {
        "dense_i": dense_i,
        "experts": experts,
        "top_k": top_k,
        "moe_i": moe_i,
        "step": step,
        "mlp_only": mlp_only,
        "shared_count": shared_count,
        "shared_i": shared_i,
        "shared_gate": shared_gate,
    }, issues


def _standard_qwen_layer_moe(index: int, ffn: Mapping[str, Any]) -> bool:
    experts = int(ffn["experts"])
    if experts <= 0:
        return False
    return index not in ffn["mlp_only"] and (index + 1) % int(ffn["step"]) == 0


def _standard_qwen_window(
    c: Mapping[str, Any], layers: int,
) -> tuple[list[str], list[int], tuple[str, ...]]:
    issues: list[str] = []
    raw_types = c.get("layer_types")
    if raw_types is not None:
        values = list(raw_types) if isinstance(raw_types, (list, tuple)) else []
        if not isinstance(raw_types, (list, tuple)):
            issues.append("layer_types_invalid")
        if len(values) != layers:
            issues.append(f"layer_types_length_mismatch:{len(values)}!={layers}")
        kinds, windows = [], []
        for index in range(layers):
            kind = str(values[index]) if index < len(values) else "unspecified_attention"
            window = 0
            if kind == "sliding_attention":
                declared = c.get("sliding_window")
                if declared is None:
                    issues.append(f"sliding_layer_type_without_declared_window:{index}")
                else:
                    window = int(declared)
                    if window <= 0:
                        issues.append(f"sliding_window_invalid:{window}")
                        window = 0
            elif kind != "full_attention":
                issues.append(f"unsupported_attention_layer:{index}:{kind}")
            kinds.append(kind)
            windows.append(window)
        return kinds, windows, tuple(dict.fromkeys(issues))

    kinds = ["full_attention"] * layers
    windows = [0] * layers
    if bool(c.get("use_sliding_window", False)):
        declared = c.get("sliding_window")
        if declared is None:
            issues.append("sliding_window_enabled_without_window")
            kinds = ["unresolved_window_schedule"] * layers
        else:
            window = int(declared)
            if window <= 0:
                issues.append(f"sliding_window_invalid:{window}")
                kinds = ["unresolved_window_schedule"] * layers
            elif c.get("max_window_layers") is None:
                issues.append("sliding_window_max_window_layers_missing")
                kinds = ["unresolved_window_schedule"] * layers
            else:
                max_layers = int(c.get("max_window_layers"))
                for index in range(layers):
                    if index >= max_layers:
                        windows[index] = window
    return kinds, windows, tuple(dict.fromkeys(issues))


def _standard_qwen_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    """Plans for standard dense and MoE GQA bodies using structural HF predicates."""
    hidden = int(_require(c, "hidden_size"))
    layers = int(_require(c, "num_hidden_layers"))
    heads = _int(c, "num_attention_heads")
    kv_heads = _int(c, "num_key_value_heads", heads)
    modern = _has_modern_standard_gqa_schema(c)
    legacy_shared_moe = _int(c, "num_experts") > 0 and not modern
    head_dim = _standard_qwen_head_dim(
        c, legacy_shared_moe=legacy_shared_moe, hidden=hidden, heads=heads,
    ) if heads else 0
    ffn, _ = _standard_qwen_ffn_config(c, legacy_shared_moe=legacy_shared_moe)
    kinds, windows, _ = _standard_qwen_window(c, layers)
    if "use_qk_norm" in c:
        qk_norm = bool(c["use_qk_norm"])
    elif "qk_norm" in c:
        qk_norm = bool(c["qk_norm"])
    else:
        qk_norm = modern
    activation = str(c.get("hidden_act", "silu"))
    partial = declared_partial_rotary(c)
    rope_dim, nope_dim = rotary_dimensions(head_dim, partial)
    has_sandwich_norm = bool(c.get("has_sandwich_norm", False))
    norm_unit_offset = bool(c.get("apply_layernorm_1p", False))
    v_norm = bool(c.get("v_norm", False))
    plans: list[LayerPlan] = []
    for index in range(layers):
        moe = _standard_qwen_layer_moe(index, ffn)
        if moe:
            experts = int(ffn["experts"])
            top_k = int(ffn["top_k"])
            moe_i = int(ffn["moe_i"])
            if top_k <= 0 or top_k > experts or moe_i <= 0:
                raise ModelingError(f"model_config_qwen_moe_dims_invalid:{index}")
            ffn_fields: dict[str, Any] = {
                "ffn": "moe", "intermediate_size": 0,
                "num_experts": experts, "top_k": top_k,
                "moe_intermediate_size": moe_i,
                "num_shared_experts": int(ffn["shared_count"]),
                "shared_intermediate_size": int(ffn["shared_i"]),
                "shared_expert_gate": bool(ffn["shared_gate"]),
                "router_input_size": hidden, "router_mode": "softmax",
                "router_dtype": "",
                "router_norm_topk": bool(c.get("norm_topk_prob", False)),
            }
        else:
            dense_i = int(ffn["dense_i"])
            if dense_i <= 0:
                raise ModelingError(f"model_config_qwen_dense_intermediate_missing:{index}")
            ffn_fields = {
                "ffn": "dense", "intermediate_size": dense_i,
                "num_experts": 0, "top_k": 0, "moe_intermediate_size": 0,
                "num_shared_experts": 0, "shared_intermediate_size": 0,
                "shared_expert_gate": False, "router_input_size": 0,
                "router_mode": "softmax", "router_norm_topk": False,
            }
        known_core = (
            kinds[index] == "full_attention"
            or (kinds[index] == "sliding_attention" and windows[index] > 0)
        )
        if not known_core:
            projections_known = heads > 0 and kv_heads > 0 and head_dim > 0
            plans.append(LayerPlan(
                index=index,
                attention="projection_only" if projections_known else "unmodeled",
                architecture=kinds[index],
                num_heads=heads if projections_known else 0,
                num_kv_heads=kv_heads if projections_known else 0,
                head_dim=head_dim if projections_known else 0,
                qk_norm=False, window=0, cache_kind="none",
                qk_rope_head_dim=rope_dim, qk_nope_head_dim=nope_dim,
                has_sandwich_norm=has_sandwich_norm,
                norm_unit_offset=norm_unit_offset,
                v_norm=v_norm,
                ffn_activation=activation,
                native_layer_id=index, scope="body", **ffn_fields,
            ))
            continue
        if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
            raise ModelingError(
                f"model_config_qwen_gqa_group_not_integral:{heads}%{kv_heads}")
        plans.append(LayerPlan(
            index=index, attention="gqa", architecture="gqa",
            num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
            qk_rope_head_dim=rope_dim, qk_nope_head_dim=nope_dim,
            qk_norm=qk_norm, window=windows[index],
            has_sandwich_norm=has_sandwich_norm,
            norm_unit_offset=norm_unit_offset,
            v_norm=v_norm,
            kv_cache_elements=2 * kv_heads * head_dim,
            kv_owner_layer_id=index, ffn_activation=activation,
            native_layer_id=index, scope="body", **ffn_fields,
        ))
    return plans


def _aux_standard_qwen(c: Mapping[str, Any]) -> AuxiliarySpec:
    """A layer count alone does not specify a standard-Qwen draft architecture."""
    del c
    return AuxiliarySpec()


def _standard_qwen_gaps(c: Mapping[str, Any]) -> tuple[str, ...]:
    """Visible unmodeled standard GQA fields derived strictly from structural config keys."""
    hidden = int(_require(c, "hidden_size"))
    heads = _int(c, "num_attention_heads")
    layers = int(_require(c, "num_hidden_layers"))
    modern = _has_modern_standard_gqa_schema(c)
    legacy_shared_moe = _int(c, "num_experts") > 0 and not modern
    issues = list(_standard_qwen_window(c, layers)[2])
    issues.extend(_standard_qwen_ffn_config(c, legacy_shared_moe=legacy_shared_moe)[1])
    if not modern:
        if bool(c.get("qkv_bias", True)):
            issues.append("qkv_bias_unmodeled")
        if legacy_shared_moe and c.get("head_dim") is not None:
            try:
                declared = int(c["head_dim"])
            except (TypeError, ValueError):
                issues.append("head_dim_invalid")
            else:
                derived = hidden // heads if heads and hidden % heads == 0 else 0
                if declared != derived:
                    issues.append(
                        f"head_dim_ignored_by_legacy_shared_moe_source:{declared}!={derived}")
    elif bool(c.get("attention_bias", False)):
        issues.append("attention_bias_unmodeled")
    activation = str(c.get("hidden_act", "silu"))
    if activation != "silu":
        issues.append(f"hidden_act_unsupported:{activation}")
    if bool(c.get("attn_output_gate", False)):
        issues.append("q_output_gate_unsupported_for_standard_gqa")
    if _int(c, "num_nextn_predict_layers") > 0 or _int(c, "mtp_num_hidden_layers") > 0:
        issues.append("optional_mtp_schema_not_modeled")
    return tuple(dict.fromkeys(issues))
