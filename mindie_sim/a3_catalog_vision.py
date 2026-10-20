"""Vision and Engram specification adapters.

[Role]
  Normalize packed vision QKV widths, merge/aligner geometry, MLP gating and
  Engram table/lookup dimensions.

[Invariants]
  * Vision qkv_width is packed QKV rows; head_dim is query_width/heads.
  * Kimi explicit per-Q width is converted to packed rows once.
  * Engram columns are (max_ngram-1)*heads per table.

[Timeouts & Locks]
  Pure mapping reads.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping

from .a3_catalog_config import _int
from .a3_catalog_types import EngramSpec, EngramTableSpec, VisionSpec, merge_source
from .a3_types import A3InputError


def _vision_v41(config: Mapping[str, Any], text: Mapping[str, Any]) -> VisionSpec:
    v = config.get("vision_config")
    if not isinstance(v, Mapping):
        raise A3InputError("model_config_vision_missing")
    hidden = int(v["hidden_size"])
    patch = int(v["patch_size"])
    downsample = int(v.get("downsample_ratio", 1) or 1)
    return VisionSpec(
        kind="deepseek_v41",
        layers=int(v["num_hidden_layers"]), hidden_size=hidden,
        intermediate_size=int(v["intermediate_size"]),
        num_heads=int(v["num_attention_heads"]), qkv_width=3 * hidden,
        gated_mlp=True, activation="silu", norm_type="layernorm", bias=True,
        patch_size=patch, merge_area=downsample * downsample,
        downsample_ratio=downsample,
        out_hidden_size=int(text["hidden_size"]),
        aligner_hidden_size=int(text["hidden_size"]),
        image_token_id=int(config.get("image_token_id", 0) or 0),
        source=merge_source({"vision_config": "deepseek_v41_vision"}),
    )


def _vision_kimi(config: Mapping[str, Any], text: Mapping[str, Any]) -> VisionSpec:
    v = config.get("vision_config")
    if not isinstance(v, Mapping):
        raise A3InputError("model_config_vision_missing")
    merge = v.get("merge_kernel_size") or [2, 2]
    merge_area = int(merge[0]) * int(merge[1])
    hidden = int(v["vt_hidden_size"])
    out_h = int(v.get("text_hidden_size", text["hidden_size"]))
    return VisionSpec(
        kind="kimi_k3", layers=int(v["vt_num_hidden_layers"]),
        hidden_size=hidden, intermediate_size=int(v["vt_intermediate_size"]),
        num_heads=int(v["vt_num_attention_heads"]),
        qkv_width=3 * int(v["qkv_hidden_size"]), patch_size=int(v["patch_size"]),
        gated_mlp=False, activation="gelu", norm_type="rmsnorm", bias=False,
        merge_area=merge_area, out_hidden_size=out_h,
        aligner_hidden_size=int(v["vt_intermediate_size"]),
        pos_embed_tokens=int(v.get("init_pos_emb_height", 0)) * int(v.get("init_pos_emb_width", 0)),
        image_token_id=int(config.get("media_placeholder_token_id", 0) or 0),
        source=merge_source({"vision_config": "kimi_k3_vision"}),
    )


def _vision_qwen(config: Mapping[str, Any], text: Mapping[str, Any]) -> VisionSpec:
    v = config.get("vision_config")
    if not isinstance(v, Mapping):
        raise A3InputError("model_config_vision_missing")
    hidden = int(v["hidden_size"])
    merge = int(v.get("spatial_merge_size", 2) or 2)
    out_h = int(v.get("out_hidden_size", text["hidden_size"]))
    return VisionSpec(
        kind="qwen3_5", layers=int(v["depth"]), hidden_size=hidden,
        intermediate_size=int(v["intermediate_size"]),
        num_heads=int(v["num_heads"]), qkv_width=3 * hidden,
        gated_mlp=False, activation="gelu", norm_type="layernorm", bias=False,
        patch_size=int(v["patch_size"]),
        temporal_patch=int(v.get("temporal_patch_size", 1) or 1),
        merge_area=merge * merge, out_hidden_size=out_h,
        aligner_hidden_size=0,
        pos_embed_tokens=int(v.get("num_position_embeddings", 0) or 0),
        deepstack=bool(v.get("deepstack_visual_indexes")),
        image_token_id=int(config.get("image_token_id", 0) or 0),
        video_token_id=int(config.get("video_token_id", 0) or 0),
        source=merge_source({"vision_config": "qwen3_5_vision"}),
    )


def _engram(c: Mapping[str, Any]) -> EngramSpec | None:
    layer_ids = [int(v) for v in c.get("engram_layer_ids", ()) or ()]
    rows = [int(v) for v in c.get("engram_num_embeddings", ()) or ()]
    if not layer_ids:
        return None
    if len(layer_ids) != len(rows):
        raise A3InputError("model_config_engram_table_mismatch")
    width = _int(c, "engram_head_dim", 0)
    max_ngram = _int(c, "engram_max_ngram_size", 0)
    heads = _int(c, "engram_n_heads", 0)
    if width <= 0 or max_ngram < 2 or heads <= 0:
        raise A3InputError("model_config_engram_geometry_missing")
    tables = tuple(
        EngramTableSpec(layer_id=layer, table_id=index, rows=row, width=width)
        for index, (layer, row) in enumerate(zip(layer_ids, rows))
    )
    return EngramSpec(
        tables=tables, max_ngram=max_ngram, heads=heads, width=width,
        pad_token_id=_int(c, "engram_pad_token_id", 0),
        source=merge_source({"engram_layers": str(len(tables))}),
    )
