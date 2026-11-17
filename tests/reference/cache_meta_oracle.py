"""Independent raw-config geometry and KV cache byte oracle.

[Role]
  Derive the declared per-layer attention/FFN/MoE schedule directly from the
  raw model JSON (no simulator plan, term or cache helper) and compute
  per-request KV/state bytes with explicit payload (unpadded) and allocated
  (capacity-block) accounting.

[Invariants]
  * Every schedule choice is a constructor field read (layer_types,
    compress_ratios, linear_attn_config, sparse_attention_config, ...).
  * Cache rows are ``seq`` (or ``min(seq, window)``), ``seq // ratio`` for a
    completed compressed row, and block-rounded only in allocated mode.
  * ``cp`` block-interleaves rows; ``tp`` replicates at least one KV head.
  * Recurrent layers own O(1) state and no token KV plane.

[Timeouts & Locks]
  Pure integer arithmetic bounded by layer count and (for PLE) a small prime
  walk; no I/O.

[Side-Effect Policy]
  Read-only pure functions; no device, network or global mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

INDEX_INTERLEAVE = 128
PLE_BF16_BYTES = 2
PLE_DEFAULT_CONV_KERNEL = 4
PLE_DEFAULT_DIVISOR = 128
POOL_TAIL_BYTES, POOL_TAIL_SCALE = 2, 4
ZERO = 0
ONE = 1
TWO = 2
FOUR = 4
KV_CACHE_PLANES = 2
REFERENCE_PINS = (
    "vllm/b65a4b4d8a25b64115135baea63373573385a8de",
    "transformers/ea9eb5de74c7d9ff92d2ea2e50c0840b1453ffb2",
    "step/364b3b03fed2477aa0953fe855a2c4c8c48ee3cc",
)


def _i(c: Mapping[str, Any], key: str, default: int = ZERO) -> int:
    value = c.get(key, default)
    return int(default) if value is None else int(value)


def flatten_config(config: Mapping[str, Any]):
    text = config.get("text_config")
    vision = config.get("vision_config")
    merged = {**config, **text} if isinstance(text, Mapping) else dict(config)
    return merged, vision if isinstance(vision, Mapping) else None


def config_profile(config: Mapping[str, Any], c: Mapping[str, Any]) -> str:
    archs = " ".join(str(v) for v in config.get("architectures", ()) or ())
    mt = str(c.get("model_type", config.get("model_type", ""))).lower()
    if "DSparkDraftModel" in archs or "K3DSpark" in archs or mt == "k3_dspark":
        return "kimi_dspark"
    if mt in {"gemma4", "gemma4_text"} or "Gemma4" in archs:
        return "gemma4"
    if mt == "minimax_m3_vl" or "MiniMaxM3" in archs or c.get("family") == "minimax_m3":
        return "minimax_m3"
    if mt in {"step5", "step5_vl"} or "Step5" in archs:
        return "step5"
    if "num_layers" in c and "num_hidden_layers" not in c:
        return "generic"
    if "deepseek_v41" in mt or "DeepseekV41" in archs:
        return "deepseek_v41"
    if "deepseek_v4" in mt or "DeepseekV4" in archs:
        return "deepseek_v4"
    if "glm_moe_dsa" in mt or "glm5_next" in mt or "Glm5Next" in archs:
        return "glm"
    if "kimi" in mt or "Kimi" in archs:
        if c.get("linear_attn_config"):
            return "kimi"
        if mt in {"kimi_k2", "kimi_k25", "kimi_vl", "deepseek_v3"}:
            return "standard_kimi"
        return "generic"
    if "deepseek" in mt or "Deepseek" in archs:
        if mt in {"deepseek_v2", "deepseek_v3", "deepseek_v32"} and c.get("kv_lora_rank"):
            return "standard_deepseek"
        return "generic"
    if "glm" in mt or "Glm" in archs or "ChatGLM" in archs:
        return "standard_glm" if mt in {"glm4", "glm4_moe", "chatglm"} else "generic"
    if "qwen" in mt or "Qwen" in archs:
        hybrid = ("linear_attention" in (c.get("layer_types") or ())
                  or _i(c, "full_attention_interval") > 0)
        if hybrid and c.get("linear_num_key_heads"):
            return "qwen"
        return "standard_qwen" if mt in {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"} else "generic"
    return "generic"


@dataclass(frozen=True)
class RefLayer:
    index: int
    attention: str = "gqa"; ffn: str = "dense"; arch: str = "gqa"
    heads: int = 0; kv_heads: int = 0; head_dim: int = 0; window: int = 0
    compress_ratio: int = 0; q_lora: int = 0; kv_lora: int = 0
    nope: int = 0; rope: int = 0; v_dim: int = 0
    o_groups: int = 0; o_lora: int = 0; mla_gate: bool = False; variant: str = "none"
    dense_i: int = 0; experts: int = 0; moe_i: int = 0; shared: int = 0
    shared_i: int = 0; shared_gate: bool = False; latent: int = 0
    qk_norm: bool = False; qgate: bool = False; gate_separate: bool = False
    k_eq_v: bool = False; mhc: bool = False; hc_mult: int = 0; hc_lowrank: int = 0
    residual: int = 0; indexer: bool = False; owns_index_k: bool = False
    owns_topk: bool = False; index_shared: bool = False; index_query: bool = False
    index_input: str = "hidden"; index_heads: int = 0; index_dim: int = 0
    index_topk: int = 0; index_kpool: int = 1; compressor_width: int = 1
    compressor_rows: int = 0; owns_kv: bool = True
    lin_heads: int = 0; lin_dim: int = 0; lin_key_heads: int = 0
    lin_key_dim: int = 0; lin_value_heads: int = 0
    conv_k: int = 0; conv_width: int = 0; rec_state: int = 0; conv_state: int = 0
    ssm_dtype: str = "fp32"; gate_rank: int = 0; gemma4: bool = False
    engram: bool = False; cache_row: int = 0; router_copy: bool = False
    index_cp_shard: bool = False


# --------------------------------------------------------------------------
# Family schedules (constructor shape declarations).
# --------------------------------------------------------------------------
def _deepseek_standard(c: Mapping[str, Any]) -> list[RefLayer]:
    layers = _i(c, "num_hidden_layers")
    heads = _i(c, "num_attention_heads")
    nope, rope = _i(c, "qk_nope_head_dim"), _i(c, "qk_rope_head_dim")
    v_dim, kv_rank = _i(c, "v_head_dim"), _i(c, "kv_lora_rank")
    q_rank = _i(c, "q_lora_rank")
    dense_i, first = _i(c, "intermediate_size"), _i(c, "first_k_dense_replace")
    freq = _i(c, "moe_layer_freq", 1)
    has_experts = c.get("n_routed_experts") is not None
    experts = _i(c, "n_routed_experts")
    moe_i = _i(c, "moe_intermediate_size")
    shared = _i(c, "n_shared_experts")
    idx = (_i(c, "index_topk"), _i(c, "index_n_heads"), _i(c, "index_head_dim"))
    has_indexer = min(idx) > 0 and q_rank > 0
    out = []
    for index in range(layers):
        moe = has_experts and index >= first and index % freq == 0
        out.append(RefLayer(
            index=index, attention="mla", arch="mla_standard",
            ffn="moe" if moe else "dense", heads=heads, kv_heads=1,
            head_dim=nope + rope, q_lora=q_rank, kv_lora=kv_rank, nope=nope,
            rope=rope, v_dim=v_dim, variant="latent_mla",
            dense_i=0 if moe else dense_i, experts=experts if moe else 0,
            moe_i=moe_i if moe else 0, shared=shared if moe else 0,
            shared_i=shared * moe_i if moe else 0,
            indexer=has_indexer, owns_index_k=has_indexer, owns_topk=has_indexer,
            index_query=has_indexer, index_heads=idx[1] if has_indexer else 0,
            index_dim=idx[2] if has_indexer else 0,
            index_topk=idx[0] if has_indexer else 0,
            cache_row=kv_rank + rope,
        ))
    return out


def _deepseek_v4(c: Mapping[str, Any], v41: bool) -> list[RefLayer]:
    layers = _i(c, "num_hidden_layers")
    head_dim, rope = _i(c, "head_dim"), _i(c, "qk_rope_head_dim")
    q_rank = _i(c, "q_lora_rank")
    has_kv = "kv_lora_rank" in c and c.get("kv_lora_rank") is not None
    kv_rank = _i(c, "kv_lora_rank") if has_kv else head_dim
    nope = _i(c, "qk_nope_head_dim", head_dim - rope)
    v_dim = _i(c, "v_head_dim", head_dim) if has_kv else head_dim
    o_lora, o_groups = _i(c, "o_lora_rank"), _i(c, "o_groups")
    experts, moe_i = _i(c, "n_routed_experts"), _i(c, "moe_intermediate_size")
    shared = _i(c, "n_shared_experts")
    window, hc = _i(c, "sliding_window"), _i(c, "hc_mult")
    idx = (_i(c, "index_topk"), _i(c, "index_n_heads"), _i(c, "index_head_dim"))
    ratios = [int(v) for v in c.get("compress_ratios", ())][:layers]
    kv_sources = {int(v) for v in c.get("kv_source_layer_ids", ()) or ()}
    index_sources = {int(v) for v in c.get("index_source_layer_ids", ()) or ()}
    out = []
    for index, ratio in enumerate(ratios):
        if v41:
            owns_kv, owns_index = index in kv_sources, index in kv_sources
            owns_topk = index in index_sources
            indexer, shared_index = owns_topk, owns_topk and index not in kv_sources
            arch = {0: "swa", 1: "csa2", 2: "csa2"}[ratio]
            state_rows = 32 if ratio == 2 else ratio
            comp_width, index_input = 1, "compressed"
        else:
            owns_kv = ratio > 0
            owns_index = owns_topk = indexer = ratio == 4
            shared_index = False
            arch = {0: "swa", 4: "csa", 128: "hca"}[ratio]
            state_rows = ratio * (TWO if ratio == 4 else ONE)
            comp_width, index_input = (TWO if ratio == 4 else ONE), "hidden"
        out.append(RefLayer(
            index=index, attention="mla", arch=arch, ffn="moe",
            heads=_i(c, "num_attention_heads"), kv_heads=1, head_dim=head_dim,
            window=window, compress_ratio=ratio, q_lora=q_rank, kv_lora=kv_rank,
            nope=nope, rope=rope, v_dim=v_dim, o_groups=o_groups, o_lora=o_lora,
            variant="latent_mla" if has_kv else "latent_mqa", experts=experts,
            moe_i=moe_i, shared=shared, shared_i=shared * moe_i, mhc=True,
            hc_mult=hc, indexer=indexer, owns_index_k=owns_index,
            owns_topk=owns_topk, index_shared=shared_index, index_query=indexer,
            index_input=index_input, index_heads=idx[1], index_dim=idx[2],
            index_topk=idx[0], compressor_width=comp_width,
            compressor_rows=state_rows, owns_kv=owns_kv and ratio > 0,
            router_copy=not v41,
            cache_row=kv_rank + rope if has_kv else head_dim,
        ))
    return out


def _kimi(c: Mapping[str, Any]) -> list[RefLayer]:
    layers = _i(c, "num_hidden_layers")
    h = _i(c, "hidden_size")
    experts = _i(c, "num_experts")
    moe_i = _i(c, "moe_intermediate_size")
    shared, dense_i = _i(c, "num_shared_experts"), _i(c, "intermediate_size")
    dense_first = _i(c, "first_k_dense_replace")
    latent = _i(c, "routed_expert_hidden_size", h)
    latent = 0 if latent == h else latent
    linear = c.get("linear_attn_config") or {}
    full_one = {int(v) for v in linear.get("full_attn_layers", ())}
    kda_one = {int(v) for v in linear.get("kda_layers", ())}
    full = {v - 1 for v in full_one} if full_one else set(range(layers)) - {v - 1 for v in kda_one}
    kda = set(range(layers)) - full
    heads, kv_heads = _i(c, "num_attention_heads"), _i(c, "num_key_value_heads")
    q_rank, kv_rank = _i(c, "q_lora_rank"), _i(c, "kv_lora_rank")
    nope, rope = _i(c, "qk_nope_head_dim"), _i(c, "qk_rope_head_dim")
    v_dim = _i(c, "v_head_dim", 128)
    kda_heads, kda_dim = _i(linear, "num_heads"), _i(linear, "head_dim")
    conv_k = _i(linear, "short_conv_kernel_size", FOUR)
    conv_width = 3 * kda_heads * kda_dim
    residual = _i(c, "attn_res_block_size")
    dsa = c.get("dsa_config") or {}
    dsa_layers = {int(v) - 1 for v in dsa.get("dsa_layers", ()) or ()}
    dsa_dims = (_i(dsa, "index_topk"), _i(dsa, "index_num_heads"), _i(dsa, "index_head_dim"))
    out = []
    for index in range(layers):
        ffn = "dense" if index < dense_first else "moe"
        ffn_fields = dict(
            ffn=ffn, dense_i=dense_i if ffn == "dense" else 0,
            experts=experts if ffn == "moe" else 0, moe_i=moe_i if ffn == "moe" else 0,
            shared=shared if ffn == "moe" else 0,
            shared_i=shared * moe_i if ffn == "moe" else 0,
            latent=latent if ffn == "moe" else 0,
        )
        if index in kda:
            out.append(RefLayer(
                index=index, attention="kda", arch="kda", lin_heads=kda_heads,
                lin_dim=kda_dim, lin_key_heads=kda_heads, lin_key_dim=kda_dim,
                lin_value_heads=kda_heads, conv_k=conv_k, conv_width=conv_width,
                rec_state=kda_heads * kda_dim * kda_dim,
                conv_state=conv_width * (conv_k - 1), residual=residual,
                **ffn_fields,
            ))
            continue
        indexer = index in dsa_layers
        out.append(RefLayer(
            index=index, attention="mla", arch="mla_dsa" if indexer else "mla",
            heads=heads, kv_heads=kv_heads, head_dim=nope + rope, q_lora=q_rank,
            kv_lora=kv_rank, nope=nope, rope=rope, v_dim=v_dim, variant="latent_mla",
            mla_gate=bool(c.get("mla_use_output_gate", True)),
            indexer=indexer, owns_index_k=indexer, owns_topk=indexer,
            index_query=indexer, index_heads=dsa_dims[1] if indexer else 0,
            index_dim=dsa_dims[2] if indexer else 0,
            index_topk=dsa_dims[0] if indexer else 0, residual=residual,
            index_cp_shard=indexer and bool(dsa), cache_row=kv_rank + rope,
            **ffn_fields,
        ))
    return out


def _glm(c: Mapping[str, Any]) -> list[RefLayer]:
    layers = _i(c, "num_hidden_layers")
    heads, kv_heads = _i(c, "num_attention_heads"), _i(c, "num_key_value_heads")
    q_rank, kv_rank = _i(c, "q_lora_rank"), _i(c, "kv_lora_rank")
    qk_head = _i(c, "qk_head_dim")
    nope, rope = _i(c, "qk_nope_head_dim"), _i(c, "qk_rope_head_dim")
    v_dim = _i(c, "v_head_dim")
    dense_i, experts = _i(c, "intermediate_size"), _i(c, "n_routed_experts")
    moe_i, shared = _i(c, "moe_intermediate_size"), _i(c, "n_shared_experts")
    kpool = max(1, _i(c, "index_kpool", 1))
    idx_heads, idx_dim = _i(c, "index_n_heads"), _i(c, "index_head_dim")
    idx_topk = _i(c, "index_topk")
    mlp_types = [str(v) for v in (c.get("mlp_layer_types") or ())]
    layer_types = [str(v) for v in (c.get("layer_types") or ())]
    out = []
    if layer_types or c.get("linear_attn_config"):
        linear = c.get("linear_attn_config") or {}
        kda_heads, kda_dim = _i(linear, "num_heads"), _i(linear, "head_dim")
        conv_k = _i(linear, "short_conv_kernel_size", FOUR)
        conv_width = 3 * kda_heads * kda_dim
        hyper, hc = bool(c.get("mhc")), _i(c, "hc_mult")
        for index, kind in enumerate(layer_types):
            ffn = "dense" if index < len(mlp_types) and mlp_types[index] == "dense" else "moe"
            ffn_fields = dict(
                ffn=ffn, dense_i=dense_i if ffn == "dense" else 0,
                experts=experts if ffn == "moe" else 0, moe_i=moe_i if ffn == "moe" else 0,
                shared=shared if ffn == "moe" else 0,
                shared_i=shared * moe_i if ffn == "moe" else 0,
            )
            if kind in {"linear_attention", "kda"}:
                out.append(RefLayer(
                    index=index, attention="kda", arch="kda", lin_heads=kda_heads,
                    lin_dim=kda_dim, lin_key_heads=kda_heads, lin_key_dim=kda_dim,
                    lin_value_heads=kda_heads, conv_k=conv_k, conv_width=conv_width,
                    rec_state=kda_heads * kda_dim * kda_dim,
                    conv_state=conv_width * (conv_k - 1), gate_rank=kda_dim,
                    mhc=hyper, hc_mult=hc, **ffn_fields,
                ))
            else:
                out.append(RefLayer(
                    index=index, attention="mla", arch="mla_dsa", heads=heads,
                    kv_heads=kv_heads, head_dim=qk_head, q_lora=q_rank, kv_lora=kv_rank,
                    nope=nope, rope=rope, v_dim=v_dim, variant="latent_mla",
                    indexer=True, owns_index_k=True, owns_topk=True, index_query=True,
                    index_heads=idx_heads, index_dim=idx_dim, index_topk=idx_topk,
                    index_kpool=kpool, mhc=hyper, hc_mult=hc, cache_row=kv_rank + rope,
                    **ffn_fields,
                ))
        return out
    index_types = [str(v) for v in (c.get("indexer_types") or ())]
    first_dense = _i(c, "first_k_dense_replace")
    for index in range(layers):
        ffn = "dense" if (mlp_types and index < len(mlp_types) and mlp_types[index] == "dense") or (
            not mlp_types and index < first_dense) else "moe"
        ffn_fields = dict(
            ffn=ffn, dense_i=dense_i if ffn == "dense" else 0,
            experts=experts if ffn == "moe" else 0, moe_i=moe_i if ffn == "moe" else 0,
            shared=shared if ffn == "moe" else 0,
            shared_i=shared * moe_i if ffn == "moe" else 0,
        )
        indexer_type = (index_types[index] if index < len(index_types)
                        else ("full" if not index_types else "shared"))
        indexer = indexer_type == "full"
        out.append(RefLayer(
            index=index, attention="mla", arch="mla_dsa", heads=heads,
            kv_heads=kv_heads, head_dim=qk_head, q_lora=q_rank, kv_lora=kv_rank,
            nope=nope, rope=rope, v_dim=v_dim, variant="latent_mla",
            indexer=indexer, owns_index_k=indexer, owns_topk=indexer,
            index_shared=not indexer, index_query=indexer,
            index_heads=idx_heads, index_dim=idx_dim, index_topk=idx_topk,
            index_kpool=kpool, cache_row=kv_rank + rope, **ffn_fields,
        ))
    return out


def _qwen_gate(c: Mapping[str, Any]) -> bool:
    if c.get("attn_output_gate") is not None:
        return bool(c.get("attn_output_gate"))
    gate = str(c.get("output_gate_type") or "").strip().lower()
    if gate:
        return gate not in {"none", "false", "0"}
    return str(c.get("model_type") or "").strip().lower() == "qwen3_next"


def _qwen(c: Mapping[str, Any]) -> list[RefLayer]:
    layers = _i(c, "num_hidden_layers")
    heads, kv_heads = _i(c, "num_attention_heads"), _i(c, "num_key_value_heads")
    head_dim = _i(c, "head_dim")
    interval = _i(c, "full_attention_interval")
    kinds = [str(v) for v in (c.get("layer_types") or (
        ["full_attention" if (i + 1) % interval == 0 else "linear_attention"
         for i in range(layers)] if interval > 0 else ()))]
    dense_i, experts = _i(c, "intermediate_size"), _i(c, "num_experts")
    moe_i = _i(c, "moe_intermediate_size")
    shared_i = _i(c, "shared_expert_intermediate_size")
    key_heads = _i(c, "linear_num_key_heads")
    value_heads = _i(c, "linear_num_value_heads")
    key_dim = _i(c, "linear_key_head_dim")
    value_dim = _i(c, "linear_value_head_dim")
    conv_k = _i(c, "linear_conv_kernel_dim")
    conv_width = TWO * key_heads * key_dim + value_heads * value_dim
    gate = _qwen_gate(c)
    hc, hc_lr = _i(c, "hc_count"), _i(c, "hc_lowrank")
    idx_heads, idx_dim = _i(c, "indexer_n_heads"), _i(c, "indexer_head_dim")
    idx_budget = _i(c, "indexer_budget")
    idx_ratio = max(1, _i(c, "indexer_compress_ratio", 1))
    indexer = idx_heads > 0 and idx_dim > 0 and idx_budget > 0
    ssm = str(c.get("mamba_ssm_dtype") or "fp32").lower()
    ssm = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}.get(ssm, ssm)
    moe = experts > 0
    out = []
    for index, kind in enumerate(kinds):
        ffn_fields = dict(
            ffn="moe" if moe else "dense", dense_i=dense_i if not moe else 0,
            experts=experts if moe else 0, moe_i=moe_i if moe else 0,
            shared=1 if moe else 0, shared_i=shared_i if moe else 0,
            shared_gate=bool(moe),
        )
        if kind == "linear_attention":
            out.append(RefLayer(
                index=index, attention="gdn", arch="gdn", heads=heads,
                kv_heads=kv_heads, head_dim=head_dim, lin_heads=value_heads,
                lin_dim=value_dim, lin_key_heads=key_heads, lin_key_dim=key_dim,
                lin_value_heads=value_heads, conv_k=conv_k, conv_width=conv_width,
                rec_state=value_heads * key_dim * value_dim,
                conv_state=conv_width * (conv_k - 1), ssm_dtype=ssm,
                qgate=gate, mhc=hc > 0, hc_mult=hc, hc_lowrank=hc_lr,
                **ffn_fields,
            ))
            continue
        out.append(RefLayer(
            index=index, attention="gqa", arch="gated_gqa" if gate else "gqa",
            heads=heads, kv_heads=kv_heads, head_dim=head_dim, qk_norm=True,
            qgate=gate, mhc=hc > 0, hc_mult=hc, hc_lowrank=hc_lr,
            indexer=indexer, owns_index_k=indexer, owns_topk=indexer,
            index_query=indexer, index_heads=idx_heads if indexer else 0,
            index_dim=idx_dim if indexer else 0,
            index_topk=idx_budget if indexer else 0,
            index_kpool=idx_ratio if indexer else 1, cache_row=KV_CACHE_PLANES * kv_heads * head_dim,
            **ffn_fields,
        ))
    return out


def _standard_qwen(c: Mapping[str, Any]) -> list[RefLayer]:
    mt = str(c.get("model_type", ""))
    hidden, layers = _i(c, "hidden_size"), _i(c, "num_hidden_layers")
    heads = _i(c, "num_attention_heads")
    kv_heads = _i(c, "num_key_value_heads", heads)
    raw = c.get("head_dim")
    head_dim = hidden // heads if mt == "qwen2_moe" or raw is None else int(raw)
    dense_i, experts = _i(c, "intermediate_size"), _i(c, "num_experts")
    moe_i = _i(c, "moe_intermediate_size")
    step = _i(c, "decoder_sparse_step", 1)
    mlp_only = {int(v) for v in (c.get("mlp_only_layers") or ())}
    shared, shared_i, shared_gate = 0, 0, False
    if experts > 0 and mt == "qwen2_moe":
        shared_i = _i(c, "shared_expert_intermediate_size")
        shared, shared_gate = (1 if shared_i > 0 else 0), shared_i > 0
    elif experts > 0 and mt == "qwen3_moe":
        shared = _i(c, "num_shared_experts", _i(c, "n_shared_experts"))
        shared_i = _i(c, "shared_expert_intermediate_size")
        if shared > 0 and shared_i <= 0:
            shared_i = shared * moe_i
        elif shared <= 0 and shared_i > 0:
            shared = 1
        if shared <= 0:
            shared_i = 0
        shared_gate = bool(c.get("shared_expert_gate", False)) if shared_i > 0 else False
    qk_norm = mt in {"qwen3", "qwen3_moe"}
    kinds = [str(v) for v in (c.get("layer_types") or ())]
    windows = [0] * layers
    sliding = _i(c, "sliding_window")
    if kinds:
        for index in range(min(layers, len(kinds))):
            if kinds[index] == "sliding_attention" and sliding > 0:
                windows[index] = sliding
    elif bool(c.get("use_sliding_window", False)) and sliding > 0 and c.get("max_window_layers") is not None:
        for index in range(int(c["max_window_layers"]), layers):
            windows[index] = sliding
    out = []
    for index in range(layers):
        moe = experts > 0 and index not in mlp_only and (index + 1) % max(1, step) == 0
        out.append(RefLayer(
            index=index, attention="gqa", arch="gqa", heads=heads, kv_heads=kv_heads,
            head_dim=head_dim, window=windows[index], qk_norm=qk_norm,
            ffn="moe" if moe else "dense", dense_i=dense_i if not moe else 0,
            experts=experts if moe else 0, moe_i=moe_i if moe else 0,
            shared=shared if moe else 0, shared_i=shared_i if moe else 0,
            shared_gate=shared_gate if moe else False,
            cache_row=KV_CACHE_PLANES * kv_heads * head_dim,
        ))
    return out


def _gemma4(c: Mapping[str, Any]) -> list[RefLayer]:
    layers, hidden = _i(c, "num_hidden_layers"), _i(c, "hidden_size")
    heads = _i(c, "num_attention_heads")
    head_dim = _i(c, "head_dim") or hidden // heads
    global_hd = _i(c, "global_head_dim") or 512
    kv_heads = _i(c, "num_key_value_heads") or heads
    global_kv = _i(c, "num_global_key_value_heads") or kv_heads
    inter = (_i(c, "intermediate_size") or _i(c, "shared_expert_intermediate_size")
             or _i(c, "moe_intermediate_size") or 1)
    kinds = [str(v) for v in (c.get("layer_types") or ())]
    k_eq_v = bool(c.get("attention_k_eq_v", False))
    window = _i(c, "sliding_window") or 1024
    declared = (_i(c, "num_experts") or _i(c, "n_routed_experts")
                or _i(c, "num_routed_experts") or _i(c, "num_local_experts"))
    moe_flag = c.get("enable_moe_block")
    moe = bool(moe_flag) if moe_flag is not None else bool(declared)
    experts = declared if moe else 0
    top_k = (_i(c, "top_k_experts") or _i(c, "num_experts_per_tok")) if moe else 0
    moe_i = _i(c, "moe_intermediate_size") if moe else 0
    shared = (_i(c, "n_shared_experts") or _i(c, "num_shared_experts") or 1) if moe else 0
    shared_i = (_i(c, "shared_expert_intermediate_size") or inter) if moe else 0
    out = []
    for index, kind in enumerate(kinds):
        full = kind == "full_attention"
        layer_kv = global_kv if full else kv_heads
        layer_hd = global_hd if full else head_dim
        out.append(RefLayer(
            index=index, attention="gqa", arch="gemma4", ffn="moe" if moe else "dense",
            window=0 if full else window, heads=heads, kv_heads=layer_kv,
            head_dim=layer_hd, dense_i=0 if moe else inter, experts=experts,
            moe_i=moe_i, shared=shared, shared_i=shared_i, qk_norm=True,
            k_eq_v=bool(full and k_eq_v), gemma4=True,
            cache_row=KV_CACHE_PLANES * layer_kv * layer_hd,
        ))
    return out


def _minimax(c: Mapping[str, Any]) -> list[RefLayer]:
    layers, hidden = _i(c, "num_hidden_layers"), _i(c, "hidden_size")
    heads, kv_heads = _i(c, "num_attention_heads"), _i(c, "num_key_value_heads")
    head_dim = _i(c, "head_dim")
    experts = _i(c, "num_local_experts") or _i(c, "num_experts")
    dense_i = _i(c, "dense_intermediate_size") or _i(c, "intermediate_size")
    moe_i = _i(c, "moe_intermediate_size") or _i(c, "intermediate_size")
    shared = _i(c, "n_shared_experts") or _i(c, "num_shared_experts")
    shared_i = _i(c, "shared_intermediate_size") or moe_i
    raw_freq = c.get("moe_layer_freq")
    if isinstance(raw_freq, (list, tuple)):
        freq = [int(v) for v in raw_freq][:layers] + [0] * (layers - len(raw_freq[:layers]))
    elif raw_freq is None:
        freq = [1] * layers
    else:
        freq = [1 if int(raw_freq) > 0 and index % int(raw_freq) == 0 else 0
                for index in range(layers)]
    sparse = dict(c.get("sparse_attention_config") or {})
    on = bool(sparse.get("use_sparse_attention", False))
    sparse_freq = [int(v) for v in (sparse.get("sparse_attention_freq") or ())]
    start = _i(sparse, "sparse_layers_start", layers)
    block = _i(sparse, "sparse_block_size")
    topk_blocks = _i(sparse, "sparse_topk_blocks")
    local_blocks = _i(sparse, "sparse_local_block")
    idx_heads = _i(sparse, "sparse_num_index_heads")
    idx_dim = _i(sparse, "sparse_index_dim")
    qk_norm = bool(c.get("use_qk_norm", True))
    gate = bool(c.get("attention_output_gate", False))
    out = []
    for index in range(layers):
        active = on and ((index < len(sparse_freq) and sparse_freq[index]) or index >= start)
        moe = bool(freq[index])
        out.append(RefLayer(
            index=index, attention="gqa", arch="gqa", ffn="moe" if moe else "dense",
            heads=heads, kv_heads=kv_heads, head_dim=head_dim,
            dense_i=0 if moe else dense_i, experts=experts if moe else 0,
            moe_i=moe_i if moe else 0, shared=shared if moe else 0,
            shared_i=shared_i if moe else 0,
            qk_norm=qk_norm, qgate=gate, indexer=active,
            owns_index_k=active, owns_topk=active,
            index_heads=idx_heads if active else 0, index_dim=idx_dim if active else 0,
            index_topk=(topk_blocks + local_blocks) * block if active else 0,
            cache_row=KV_CACHE_PLANES * kv_heads * head_dim,
        ))
    return out


def _step5(c: Mapping[str, Any]) -> list[RefLayer]:
    layers = _i(c, "num_hidden_layers")
    kinds = [str(v) for v in (c.get("layer_types") or ())][:layers]
    hidden, heads = _i(c, "hidden_size"), _i(c, "num_attention_heads")
    kv_heads = _i(c, "num_attention_groups") or _i(c, "num_key_value_heads") or heads
    head_dim = _i(c, "head_dim")
    window = _i(c, "sliding_window") or 512
    experts = (_i(c, "moe_num_experts") or _i(c, "n_routed_experts")
               or _i(c, "num_routed_experts") or _i(c, "num_experts"))
    top_k = _i(c, "moe_top_k") or _i(c, "num_experts_per_tok")
    moe_i = _i(c, "moe_intermediate_size")
    shared_i = _i(c, "share_expert_dim") or _i(c, "shared_expert_intermediate_size") or moe_i
    raw_moe = c.get("moe_layer_list")
    if isinstance(raw_moe, (list, tuple)):
        moe_set = {int(v) for v in raw_moe}
    else:
        enum = c.get("moe_layers_enum")
        moe_set = {int(v) for v in str(enum).split(",") if v.strip()} if enum is not None else set()
    if bool(c.get("use_moe", False)) and not moe_set:
        raise ValueError("step5_moe_schedule_missing")
    if not moe_set and c.get("use_moe") is None and experts > 0:
        moe_set = set(range(layers))
    intermediate = (_i(c, "intermediate_size") if _i(c, "intermediate_size")
                    or len(moe_set) < layers else shared_i or moe_i)
    sparse = dict(c.get("sparse_config") or {})
    on = bool(sparse.get("enabled", False))
    apply_types = ({str(v) for v in (sparse.get("apply_to_layer_types") or ("full_attention",))}
                   if on else set())
    idx_heads = _i(sparse, "sparse_indexer_num_heads")
    idx_dim = _i(sparse, "proxy_dim")
    topk = _i(sparse, "topk")
    block = max(1, _i(sparse, "region_block_size", 1))
    gate = bool(c.get("use_head_wise_attn_gate", False))
    out = []
    for index, kind in enumerate(kinds):
        full, moe = kind == "full_attention", index in moe_set
        sparse_layer = bool(on and kind in apply_types)
        out.append(RefLayer(
            index=index, attention="gqa", arch="gqa", ffn="moe" if moe else "dense",
            window=0 if full else window, heads=heads, kv_heads=kv_heads,
            head_dim=head_dim, dense_i=0 if moe else intermediate,
            experts=experts if moe else 0, moe_i=moe_i if moe else 0,
            shared=1 if moe else 0, shared_i=shared_i if moe else 0,
            qgate=gate, gate_separate=gate, indexer=sparse_layer,
            owns_index_k=sparse_layer, owns_topk=sparse_layer,
            index_heads=idx_heads if sparse_layer else 0,
            index_dim=idx_dim if sparse_layer else 0,
            index_topk=topk * block if sparse_layer else 0,
            index_kpool=block if sparse_layer else 1,
            cache_row=KV_CACHE_PLANES * kv_heads * head_dim,
        ))
    return out


def _dspark(c: Mapping[str, Any]) -> tuple[list[RefLayer], dict[str, Any]]:
    layers = _i(c, "num_hidden_layers")
    heads, dense_i = _i(c, "num_attention_heads"), _i(c, "intermediate_size")
    kv_lora = _i(c, "kv_lora_rank")
    if kv_lora > 0:
        q_lora = _i(c, "q_lora_rank")
        nope, rope = _i(c, "qk_nope_head_dim"), _i(c, "qk_rope_head_dim")
        v_dim = _i(c, "v_head_dim", 128) or 128
        head_dim = nope + rope
        kv_heads = _i(c, "num_key_value_heads", 1) or 1
        rows = [RefLayer(
            index=index, attention="mla", arch="mla", ffn="dense", heads=heads,
            kv_heads=kv_heads, head_dim=head_dim, q_lora=q_lora, kv_lora=kv_lora,
            nope=nope, rope=rope, v_dim=v_dim, variant="latent_mla",
            mla_gate=bool(c.get("mla_use_output_gate", False)),
            dense_i=dense_i, cache_row=kv_lora + rope,
        ) for index in range(layers)]
        kind = "mla"
    else:
        head_dim = _i(c, "head_dim")
        kv_heads = _i(c, "num_key_value_heads")
        rows = [RefLayer(
            index=index, attention="gqa", arch="gqa", ffn="dense", heads=heads,
            kv_heads=kv_heads, head_dim=head_dim, dense_i=dense_i,
            cache_row=KV_CACHE_PLANES * kv_heads * head_dim,
        ) for index in range(layers)]
        kind = "gqa"
    dflash = c.get("dflash_config") or {}
    targets = tuple(int(v) for v in (
        c.get("target_layer_ids") or dflash.get("target_layer_ids") or ()))
    markov = _i(c, "markov_rank", 256)
    return rows, {"attention": kind, "input_width": len(targets) * _i(c, "hidden_size"),
                  "markov_rank": markov, "target_layer_ids": list(targets)}


def build_layers(config: Mapping[str, Any]):
    c, _ = flatten_config(config)
    builder = config_profile(config, c)
    if builder in {"deepseek_v4", "deepseek_v41"}:
        layers = _deepseek_v4(c, builder == "deepseek_v41")
    elif builder in {"standard_deepseek", "standard_kimi"}:
        layers = _deepseek_standard(c)
    elif builder == "kimi":
        layers = _kimi(c)
    elif builder == "glm":
        layers = _glm(c)
    elif builder == "qwen":
        layers = _qwen(c)
    elif builder == "standard_qwen":
        layers = _standard_qwen(c)
    elif builder == "gemma4":
        layers = _gemma4(c)
    elif builder == "minimax_m3":
        layers = _minimax(c)
    elif builder == "step5":
        layers = _step5(c)
    elif builder == "kimi_dspark":
        layers, draft = _dspark(c)
        return {"builder": builder, "layers": layers, "draft": draft, "standalone": True}
    else:
        raise ValueError(f"reference_builder_unsupported:{builder}")
    return {"builder": builder, "layers": layers, "draft": None, "standalone": False}


# --------------------------------------------------------------------------
# KV cache byte oracle (payload = unpadded rows, allocated = capacity blocks).
# --------------------------------------------------------------------------
def _dtype_bytes(name: str) -> int:
    return {"int8": 1, "uint8": 1, "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
            "fp16": 2, "bf16": 2, "bfloat16": 2, "float16": 2,
            "fp32": 4, "float32": 4}.get(str(name), 2)


def _block_round(rows: int, block: int) -> int:
    if block <= 1:
        return int(rows)
    return ((int(rows) + block - 1) // block) * block


def _local_rows(rows: int, block: int, cp: int, padded: bool):
    if cp <= 1:
        return _block_round(rows, block) if padded else int(rows)
    if padded:
        blocks = (int(rows) + max(1, block) - 1) // max(1, block)
        return ((blocks + cp - 1) // cp) * block
    return int(rows) / cp


def _local_kv_heads(num_kv_heads: int, tp: int) -> int:
    return max(1, (int(num_kv_heads) + max(1, tp) - 1) // max(1, tp))


def _layer_cache_bytes(layer: RefLayer, seq: int, *, quant: Mapping[str, Any],
                       padded: bool, tp: int, cp: int) -> int:
    kv_bytes = _dtype_bytes(str(quant.get("kv_dtype", "bf16")))
    index_bytes = _dtype_bytes(str(quant.get("index_dtype", "bf16")))
    scale_bytes = ZERO if str(quant.get("index_scale_dtype", "none")) in {"", "none"} \
        else _dtype_bytes(str(quant["index_scale_dtype"]))
    total = ZERO
    if layer.attention in {"gdn", "kda", "mamba"}:
        total += layer.rec_state // max(1, tp) * _dtype_bytes(layer.ssm_dtype)
        total += layer.conv_state // max(1, tp) * _dtype_bytes("bf16")
        return total
    if layer.attention in {"gqa", "projection_only"}:
        rows = min(seq, layer.window) if layer.window else seq
        width = KV_CACHE_PLANES * _local_kv_heads(layer.kv_heads, tp) * layer.head_dim
        total += _local_rows(rows, INDEX_INTERLEAVE, cp, padded) * width * kv_bytes
        if layer.owns_index_k and layer.index_dim:
            total += _index_entries(layer, seq, padded, tp, cp, index_bytes, scale_bytes)
        return total
    if layer.attention != "mla":
        return total
    width = layer.cache_row
    if not layer.compress_ratio:
        rows = min(seq, layer.window) if layer.window else seq
        total += _local_rows(rows, INDEX_INTERLEAVE, cp, padded) * width * kv_bytes
    elif layer.window:
        total += _local_rows(min(seq, layer.window), INDEX_INTERLEAVE, cp, padded) * width * kv_bytes
    if layer.compress_ratio and layer.owns_kv:
        block = max(ONE, INDEX_INTERLEAVE // max(ONE, layer.compress_ratio))
        rows = seq // max(1, layer.compress_ratio)
        total += _local_rows(rows, block, cp, padded) * width * kv_bytes
        if layer.compressor_rows and seq:
            state_rows = layer.compressor_rows if padded else min(seq, layer.compressor_rows)
            total += state_rows * TWO * layer.compressor_width * layer.head_dim * FOUR
    if layer.owns_index_k and layer.index_dim:
        total += _index_entries(layer, seq, padded, tp, cp, index_bytes, scale_bytes)
    return total


def _index_entries(layer: RefLayer, seq: int, padded: bool, tp: int, cp: int,
                   index_bytes: int, scale_bytes: int) -> int:
    ratio = max(ONE, layer.compress_ratio, layer.index_kpool)
    rows = seq // ratio
    if layer.index_cp_shard and cp > ONE:
        rows = _local_rows(rows, max(ONE, INDEX_INTERLEAVE // ratio), cp, padded)
    total = rows * (layer.index_dim * index_bytes + scale_bytes)
    if layer.index_kpool > ONE and seq:
        if layer.attention == "gqa":
            ring = TWO * layer.index_kpool if padded else min(seq, TWO * layer.index_kpool)
            total += ring * layer.index_dim * POOL_TAIL_BYTES
        else:
            capacity = layer.index_kpool
            ring = capacity if padded else min(seq, capacity)
            total += ring * TWO * layer.index_dim * POOL_TAIL_SCALE
    if layer.index_input == "hidden" and layer.compress_ratio and seq:
        total += min(seq, layer.compressor_rows) * TWO * layer.compressor_width * layer.index_dim * FOUR
    return total


def cache_totals(geometry: Mapping[str, Any], seq: int, *, padded: bool,
                 tp: int = 1, cp: int = 1) -> int:
    seq = max(0, int(seq))
    quant = geometry["quant"]
    total = ZERO
    for layer in geometry["layers"]:
        total += _layer_cache_bytes(layer, seq, quant=quant, padded=padded, tp=tp, cp=cp)
        if layer.engram and seq and geometry["engram"] is not None \
                and geometry["engram"].get("kind") == "ple":
            e = geometry["engram"]
            conv_len = (_i(e, "conv_k", PLE_DEFAULT_CONV_KERNEL) - ONE) * _i(e, "max_ngram")
            total += conv_len * _i(e, "hc_mult") * _i(e, "hidden") * PLE_BF16_BYTES
    return total

def cache_layout_declaration(geometry: Mapping[str, Any]) -> dict[str, Any]:
    layers = geometry["layers"]
    compress = [layer.compress_ratio for layer in layers]
    return {
        "layers": len(layers),
        "attention_types": _counts(layer.attention for layer in layers),
        "ffn_types": _counts(layer.ffn for layer in layers),
        "window": max((layer.window for layer in layers), default=0),
        "compress_ratios": compress,
        "index_k_layers": sum(1 for layer in layers if layer.owns_index_k),
        "compressor_state_rows": [layer.compressor_rows for layer in layers],
        "kv_dtype": geometry["quant"].get("kv_dtype", "bf16"),
        "index_dtype": geometry["quant"].get("index_dtype", "bf16"),
        "index_scale_dtype": geometry["quant"].get("index_scale_dtype", "none"),
        "recurrent_layers": sum(1 for layer in layers if layer.attention in {"gdn", "kda"}),
        "engram_tables": len((geometry["engram"] or {}).get("tables", ())),
    }


def _counts(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


__all__ = [
    "REFERENCE_PINS", "RefLayer", "build_layers", "cache_layout_declaration",
    "cache_totals", "config_profile", "flatten_config",
]
