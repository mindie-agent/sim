"""KV cache byte-total and sparse/draft family builders for cache_meta_oracle.

[Role] Independent per-layer KV/index/recurrent/PLE byte-total formulas and
  remaining family layer builders (MiniMax, Step-5, DSpark) consumed by
  ``cache_meta_oracle`` so every ``tests/**/*.py`` stays ``<= 600`` lines.
[Invariants] Pure arithmetic from raw HF config mappings; never imports
  ``mindie_sim.catalog`` builders under test.
[Timeouts & Locks] Pure in-memory arithmetic; no I/O or locks.
[Side-Effect Policy] Side-effect-free; returns fresh records and dictionaries.
"""
from __future__ import annotations

from typing import Any, Mapping

from .cache_meta_layers import (
    INDEX_INTERLEAVE,
    KV_CACHE_PLANES,
    PLE_BF16_BYTES,
    PLE_DEFAULT_CONV_KERNEL,
    POOL_TAIL_BYTES,
    POOL_TAIL_SCALE,
    RefLayer,
    _deepseek_standard,
    _deepseek_v4,
    _gemma4,
    _glm,
    _i,
    _kimi,
    _qwen,
    _standard_qwen,
    config_profile,
    flatten_config,
)

def _minimax(c: Mapping[str, Any]) -> list[RefLayer]:
    layers = _i(c, "num_hidden_layers")
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
            index_topk=topk_blocks * block if active else 0,
            router_bias=bool(moe and c.get("use_routing_bias", False)),
            cache_row=KV_CACHE_PLANES * kv_heads * head_dim,
        ))
    return out


def _step5(c: Mapping[str, Any]) -> list[RefLayer]:
    layers = _i(c, "num_hidden_layers")
    kinds = [str(v) for v in (c.get("layer_types") or ())][:layers]
    heads = _i(c, "num_attention_heads")
    kv_heads = _i(c, "num_attention_groups") or _i(c, "num_key_value_heads") or heads
    head_dim = _i(c, "head_dim")
    window = _i(c, "sliding_window") or 512
    experts = (_i(c, "moe_num_experts") or _i(c, "n_routed_experts")
               or _i(c, "num_routed_experts") or _i(c, "num_experts"))
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
            router_bias=bool(moe and c.get("use_moe_router_bias", False)),
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
            kv_heads=kv_heads, head_dim=head_dim, dense_i=dense_i, qk_norm=True,
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


def _window_rows(seq: int, window: int, *, padded: bool, block: int) -> int:
    if window <= 0:
        return seq
    if padded and seq > window:
        blk = max(1, int(block))
        return ((int(window) + blk - 1) // blk + 1) * blk
    return min(seq, window)


def _layer_cache_bytes(layer: RefLayer, seq: int, *, quant: Mapping[str, Any],
                       padded: bool, tp: int, cp: int) -> int:
    kv_bytes = _dtype_bytes(str(quant.get("kv_dtype", "bf16")))
    index_bytes = _dtype_bytes(str(quant.get("index_dtype", "bf16")))
    scale_bytes = 0 if str(quant.get("index_scale_dtype", "none")) in {"", "none"} \
        else _dtype_bytes(str(quant["index_scale_dtype"]))
    total = 0
    if layer.attention in {"gdn", "kda", "mamba"}:
        total += layer.rec_state // max(1, tp) * _dtype_bytes(layer.ssm_dtype)
        total += layer.conv_state // max(1, tp) * _dtype_bytes("bf16")
        return total
    if layer.attention in {"gqa", "projection_only"}:
        rows = _window_rows(seq, layer.window, padded=padded, block=INDEX_INTERLEAVE)
        width = KV_CACHE_PLANES * _local_kv_heads(layer.kv_heads, tp) * layer.head_dim
        total += _local_rows(rows, INDEX_INTERLEAVE, cp, padded) * width * kv_bytes
        if layer.owns_index_k and layer.index_dim:
            total += _index_entries(layer, seq, padded, tp, cp, index_bytes, scale_bytes)
        return total
    if layer.attention != "mla":
        return total
    width = layer.cache_row
    if not layer.compress_ratio:
        rows = (
            min(seq, layer.window) if layer.arch == "swa"
            else _window_rows(seq, layer.window, padded=padded, block=INDEX_INTERLEAVE)
        )
        total += _local_rows(rows, INDEX_INTERLEAVE, cp, padded) * width * kv_bytes
    elif layer.window:
        total += _local_rows(min(seq, layer.window), INDEX_INTERLEAVE, cp, padded) * width * kv_bytes
    if layer.compress_ratio and layer.owns_kv:
        block = max(1, INDEX_INTERLEAVE // max(1, layer.compress_ratio))
        rows = seq // max(1, layer.compress_ratio)
        total += _local_rows(rows, block, cp, padded) * width * kv_bytes
        if layer.compressor_rows and seq:
            state_rows = layer.compressor_rows if padded else min(seq, layer.compressor_rows)
            total += state_rows * 2 * layer.compressor_width * layer.head_dim * 4
    if layer.owns_index_k and layer.index_dim:
        total += _index_entries(layer, seq, padded, tp, cp, index_bytes, scale_bytes)
    return total


def _index_entries(layer: RefLayer, seq: int, padded: bool, tp: int, cp: int,
                   index_bytes: int, scale_bytes: int) -> int:
    ratio = max(1, layer.compress_ratio, layer.index_kpool)
    rows = seq // ratio
    block = max(1, INDEX_INTERLEAVE // ratio)
    if layer.index_cp_shard and cp > 1:
        rows = _local_rows(rows, block, cp, padded)
    elif cp <= 1:
        rows = _local_rows(rows, block, cp, padded)
    elif padded:
        rows = _block_round(rows, block)
    total = rows * (layer.index_dim * index_bytes + scale_bytes)
    if layer.index_kpool > 1 and seq:
        if layer.attention == "gqa":
            ring = 2 * layer.index_kpool if padded else min(seq, 2 * layer.index_kpool)
            total += ring * layer.index_dim * POOL_TAIL_BYTES
        else:
            capacity = layer.index_kpool
            ring = capacity if padded else min(seq, capacity)
            total += ring * 2 * layer.index_dim * POOL_TAIL_SCALE
    if layer.index_input == "hidden" and layer.compress_ratio and seq:
        total += min(seq, layer.compressor_rows) * 2 * layer.compressor_width * layer.index_dim * 4
    return total


def cache_totals(geometry: Mapping[str, Any], seq: int, *, padded: bool,
                 tp: int = 1, cp: int = 1) -> int:
    seq = max(0, int(seq))
    quant = geometry["quant"]
    total = 0
    for layer in geometry["layers"]:
        total += _layer_cache_bytes(layer, seq, quant=quant, padded=padded, tp=tp, cp=cp)
        if layer.engram and seq and geometry["engram"] is not None \
                and geometry["engram"].get("kind") == "ple":
            e = geometry["engram"]
            conv_len = (_i(e, "conv_k", PLE_DEFAULT_CONV_KERNEL) - 1) * _i(e, "max_ngram")
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
