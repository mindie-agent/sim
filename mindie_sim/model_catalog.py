"""Catalog loader for the nine se-tools models, measured A5 served checkpoints
and extended architecture presets.

[Role]
  Load the checked-in actual configs and expose independent identities for all
  nine se-tools models plus extended architecture presets (LongCat-Flash-Lite,
  MiniMax-M3, Bailing-V3, Qwen3.5-MoE, HunyuanVideo-DiT).  Validate hybrid
  recurrent (GDN/KDA), cross-layer KV owner sharing, Latent MoE projection
  width, and DiT 3D visual token expansion invariants.

[Invariants]
  * Constants come from config JSON or explicit caller inputs; no model-name
    latency constants and no source-SHA admission gate.
  * ``model.layers`` equals the body-layer count; optional draft/MTP never
    becomes a body layer and never runs on a plain forward.
  * Recurrent/conv state (GDN/KDA/Mamba) is an O(1)-in-seq_len per-sequence
    endpoint; cross-layer KV allocation counts only ``kv_owner_layers``.
  * Missing or invalid dimensions raise ``ModelingError``; zero placeholders are
    not returned for weights.

[Timeouts & Locks]
  Reads local JSON files only; no network, NPU, subprocess or locks.

[Side-Effect Policy]
  No filesystem writes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.coverage import model_coverage
from mindie_sim.catalog.specs import build_model_spec as _build_model_spec
from mindie_sim.catalog.weights import (
    body_parameter_count, body_weight_bytes, sharded_weight_bytes as _sharded,
    stage_weight_bytes, stage_weight_terms, weight_breakdown,
)

CONFIG_DIR = Path(__file__).with_name("model_configs")
CATALOG: dict[str, str] = {
    "deepseek-v4-10t": "deepseek-v4-10t.json",
    "deepseek-v4-flash-0731": "deepseek-v4-flash-0731.json",
    "deepseek-v4-pro-0813": "deepseek-v4-pro-0813.json",
    "deepseek-v4.1-flash": "deepseek-v4.1-flash.json",
    "glm-5.3": "glm-5.3.json",
    # The measured A5 GLM-5.3-Flash-BF16 served checkpoint is a distinct
    # 45-layer hybrid (11 DSA + 34 KDA) release, not a rewrite of the 78-layer
    # GLM-5.3 catalog entry.  It gets its own config so neither model is
    # misrepresented by the other's geometry.
    "glm-5.3-flash": "glm-5.3-flash.json",
    "kimi-k3": "kimi-k3.json",
    "kimi-k3-dspark": "kimi-k3-dspark.json",
    "qwen3.8-2.4t-a95b": "qwen3.8-2.4t-a95b.json",
    "qwen3.8-27b": "qwen3.8-27b.json",
    # Qwen3.6-27B-W8A8 publishes the same body geometry as qwen3.8-27b but is
    # a distinct measured checkpoint/identity; keep a separate config so the
    # true128k row8 case cannot silently resolve to the Qwen3.8 model.
    "qwen3.6-27b": "qwen3.6-27b.json",
    "qwen3.5-35b-a3b": "qwen3.5-35b-a3b.json",
}

MODEL_ALIASES = {
    "deepseek-v4-flash": "deepseek-v4-flash-0731",
    "deepseek-v4-pro": "deepseek-v4-pro-0813",
    "deepseek-v4.1": "deepseek-v4.1-flash",
    "qwen3.8": "qwen3.8-27b",
    "DeepSeek-V4-10T": "deepseek-v4-10t",
    "DeepSeek-V4-Flash-0731": "deepseek-v4-flash-0731",
    "DeepSeek-V4-Pro-0813": "deepseek-v4-pro-0813",
    "DeepSeek-V4.1-Flash": "deepseek-v4.1-flash",
    "GLM-5.3": "glm-5.3",
    "GLM-5.3-Flash": "glm-5.3-flash",
    "glm53-flash": "glm-5.3-flash",
    # The measured A3 GLM-5.3-W4A8-INT checkpoint publishes the same 78-layer
    # / hidden 6144 / 256-expert top-8 / MLA 512-2048 / index_topk 2048 body
    # as the `glm-5.3` catalog entry, so the A3 case binds that config by
    # identity instead of forking a byte-identical model file.  Only the
    # quantization preset differs: the checkpoint is W4A8_DYNAMIC routed
    # experts + W8A8_DYNAMIC attention/shared/dense, declared by the case.
    "GLM-5.3-W4A8": "glm-5.3",
    "GLM-5.3-W4A8-INT": "glm-5.3",
    "glm-5.3-w4a8": "glm-5.3",
    "Kimi-K3": "kimi-k3",
    "Kimi-K3-DSpark": "kimi-k3-dspark",
    "Qwen3.8-2.4T-A95B": "qwen3.8-2.4t-a95b",
    "Qwen3.8-27B": "qwen3.8-27b",
    # Qwen3.6-27B-W8A8 has the same body dimensions but a distinct measured
    # identity; the row8 case must not resolve to qwen3.8-27b.
    "Qwen3.6-27B": "qwen3.6-27b",
    "qwen3.6-27b": "qwen3.6-27b",
    "Qwen3.5-35B-A3B": "qwen3.5-35b-a3b",
    "qwen3.5-35b-a3b": "qwen3.5-35b-a3b",
    "qwen3_5_moe_a3b": "qwen3.5-35b-a3b",
    # The measured Qwen3.6-35B-A3B-W8A8 checkpoint publishes the same
    # 40-layer / 30-GDN + 10-GQA / 256-expert-top-8 / hidden-2048 MoE config
    # as the Qwen3.5-35B-A3B entry (byte-identical geometry; only the weights
    # and the MTP recipe differ).  The A3 case binds that config by identity
    # instead of forking a byte-identical model file; the 5120-hidden body
    # belongs to the dense 27B model, not this MoE checkpoint.
    "Qwen3.6-35B-A3B": "qwen3.5-35b-a3b",
    "qwen3.6-35b-a3b": "qwen3.5-35b-a3b",
}

ARCHITECTURE_PRESET_CATALOG: dict[str, str] = {
    "longcat-flash-lite": "longcat-flash-lite.json",
    "minimax-m3": "minimax-m3.json",
    "bailing-v3": "bailing-v3.json",
    "qwen3.5-moe": "qwen3.5-moe.json",
    "hunyuan-video-dit": "hunyuan-video-dit.json",
}

ARCHITECTURE_PRESET_ALIASES = {
    "LongCat-Flash-Lite": "longcat-flash-lite",
    "longcat_flash_lite": "longcat-flash-lite",
    "MiniMax-M3": "minimax-m3",
    "MiniMax-M3-VL": "minimax-m3",
    "minimax_m3": "minimax-m3",
    "Bailing-V3": "bailing-v3",
    "Bailing-Moe-V3": "bailing-v3",
    "bailing_v3": "bailing-v3",
    "Qwen3.5-MoE": "qwen3.5-moe",
    "Qwen3.5-MoE-A35B": "qwen3.5-moe",
    "qwen3_5_moe": "qwen3.5-moe",
    "HunyuanVideo": "hunyuan-video-dit",
    "HunyuanVideo-DiT": "hunyuan-video-dit",
    "hunyuan_video": "hunyuan-video-dit",
}

_RECURRENT_KINDS = {"gdn", "kda", "mamba"}
_TOKEN_KV_KINDS = {"gqa", "mla", "gqa_sparse_indexer"}
_DIT_ACTIVE_CACHE_METHODS = {"teacache", "fbcache", "taylorseer"}
_DIT_ACTIVE_SPARSE_METHODS = {"topk", "svg"}


def _catalog_key(key: str) -> str:
    """Normalize declared catalog labels while preserving local path spelling."""
    aliases = {**MODEL_ALIASES, **ARCHITECTURE_PRESET_ALIASES}
    if key in aliases:
        return aliases[key]
    normalized = str(key).strip().lower()
    if normalized in aliases:
        return aliases[normalized]
    if normalized in CATALOG or normalized in ARCHITECTURE_PRESET_CATALOG:
        return normalized
    return key


@dataclass(frozen=True)
class ModelArchitectureProfile:
    """Validated architecture profile for hybrid LLM/MoE/DiT capacity & fusion planning."""

    key: str
    name: str
    model_type: str
    family: str
    num_layers: int
    hidden_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    attention_type: str
    layer_attention_schedule: tuple[str, ...]
    recurrent_layer_indices: tuple[int, ...]
    token_kv_layer_indices: tuple[int, ...]
    physical_kv_owner_layers: tuple[int, ...]
    consumer_shared_kv_layers: tuple[int, ...]
    kv_consumer_to_owner: tuple[tuple[int, int], ...]
    ffn_type: str
    ffn_hidden_size: int
    moe_intermediate_size: int
    num_routed_experts: int
    zero_expert_num: int
    num_shared_experts: int
    top_k: int
    first_k_dense_layers: int
    latent_moe_hidden_size: int | None
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_head_dim: int
    recurrent_variant: str
    recurrent_num_k_heads: int
    recurrent_num_v_heads: int
    recurrent_head_k_dim: int
    recurrent_head_v_dim: int
    conv_kernel_size: int
    visual_tokens: int
    text_tokens: int
    total_dit_tokens: int
    dit_cache_method: str
    sparse_attention_method: str
    ulysses_degree: int
    ring_degree: int
    ulysses_anything: bool
    weight_dtype: str
    kv_dtype: str
    activation_dtype: str
    dense_weight_bytes: float
    routed_expert_weight_bytes: float
    single_expert_weight_bytes: float

    @property
    def moe_dispatch_token_width(self) -> int:
        return int(self.latent_moe_hidden_size or self.hidden_size)

    @property
    def moe_dispatch_compression_ratio(self) -> float:
        return float(self.moe_dispatch_token_width) / float(max(1, self.hidden_size))

    def validate_tp(self, tp_size: int) -> None:
        if tp_size <= 0:
            raise ModelingError(f"tp_size_invalid:{tp_size}")
        if self.recurrent_layer_indices and self.recurrent_variant == "gdn" and tp_size > 1:
            if self.recurrent_num_k_heads % tp_size != 0:
                raise ModelingError(
                    f"gdn_tp_inexact_k_heads:{self.recurrent_num_k_heads}%{tp_size}"
                )
            if self.recurrent_num_v_heads % tp_size != 0:
                raise ModelingError(
                    f"gdn_tp_inexact_v_heads:{self.recurrent_num_v_heads}%{tp_size}"
                )

    def recurrent_state_bytes_per_seq(self, tp_size: int = 1, pp_size: int = 1) -> float:
        """Per-sequence recurrent matrix bytes on one rank (O(1) in seq_len)."""
        self.validate_tp(tp_size)
        if not self.recurrent_layer_indices:
            return 0.0
        tp = max(1, int(tp_size))
        pp = max(1, int(pp_size))
        local_v_heads = max(1, math.ceil(self.recurrent_num_v_heads / tp))
        per_layer = float(local_v_heads * self.recurrent_head_k_dim * self.recurrent_head_v_dim * 4.0)
        layers_on_stage = math.ceil(len(self.recurrent_layer_indices) / pp)
        return float(layers_on_stage * per_layer)

    def conv_state_bytes_per_seq(self, tp_size: int = 1, pp_size: int = 1) -> float:
        """Per-sequence causal conv history bytes on one rank (O(1) in seq_len)."""
        self.validate_tp(tp_size)
        if not self.recurrent_layer_indices or self.conv_kernel_size <= 1:
            return 0.0
        tp = max(1, int(tp_size))
        pp = max(1, int(pp_size))
        if self.recurrent_variant == "gdn":
            channels = (
                2 * self.recurrent_num_k_heads * self.recurrent_head_k_dim
                + self.recurrent_num_v_heads * self.recurrent_head_v_dim
            )
        else:
            channels = 3 * self.recurrent_num_v_heads * self.recurrent_head_v_dim
        local_channels = math.ceil(channels / tp)
        per_layer = float(local_channels * (self.conv_kernel_size - 1) * 2.0)
        layers_on_stage = math.ceil(len(self.recurrent_layer_indices) / pp)
        return float(layers_on_stage * per_layer)

    def total_state_bytes_per_seq(self, tp_size: int = 1, pp_size: int = 1) -> float:
        """Total recurrent + conv endpoint state per active sequence on one rank."""
        return self.recurrent_state_bytes_per_seq(tp_size, pp_size) + self.conv_state_bytes_per_seq(tp_size, pp_size)

    def kv_bytes_per_owner_layer_per_token(self, tp_size: int = 1, dcp_size: int = 1) -> float:
        """Physical KV (+ optional indexer K) bytes per token for a single owner layer."""
        tp = max(1, int(tp_size))
        dcp = max(1, int(dcp_size))
        kv_b = dtype_bytes(self.kv_dtype)
        # Check the first owner layer's attention kind
        owner_kind = (
            self.layer_attention_schedule[self.physical_kv_owner_layers[0]]
            if self.physical_kv_owner_layers
            else self.attention_type
        )
        if owner_kind == "mla" or self.kv_lora_rank > 0:
            # MLA caches compressed latent kv_lora_rank + qk_rope_head_dim
            kv_row_bytes = float((self.kv_lora_rank + self.qk_rope_head_dim) * kv_b)
        else:
            local_kv = max(1, math.ceil(self.num_key_value_heads / tp))
            kv_row_bytes = float(2 * local_kv * self.head_dim * kv_b)
        # Context-addressed KV is DCP-sharded; the DSA indexer plane is
        # replicated on every CP rank (global TopK), so it is never divided.
        index_row_bytes = (
            float(self.index_head_dim * kv_b) if self.index_head_dim > 0 else 0.0
        )
        return kv_row_bytes / dcp + index_row_bytes

    def kv_bytes_per_token_per_rank(
        self, tp_size: int = 1, dcp_size: int = 1, pp_size: int = 1,
    ) -> float:
        """Physical KV bytes per token on one rank, counting ONLY owner layers."""
        if not self.physical_kv_owner_layers:
            return 0.0
        pp = max(1, int(pp_size))
        owners_on_stage = math.ceil(len(self.physical_kv_owner_layers) / pp)
        return float(owners_on_stage * self.kv_bytes_per_owner_layer_per_token(tp_size, dcp_size))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "model_type": self.model_type,
            "family": self.family,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "attention_type": self.attention_type,
            "recurrent_layers": len(self.recurrent_layer_indices),
            "token_kv_layers": len(self.token_kv_layer_indices),
            "physical_kv_owner_layers": list(self.physical_kv_owner_layers),
            "consumer_shared_kv_layers": list(self.consumer_shared_kv_layers),
            "latent_moe_hidden_size": self.latent_moe_hidden_size,
            "moe_dispatch_token_width": self.moe_dispatch_token_width,
            "moe_dispatch_compression_ratio": self.moe_dispatch_compression_ratio,
            "visual_tokens": self.visual_tokens,
            "total_dit_tokens": self.total_dit_tokens,
            "state_bytes_per_seq_tp1": self.total_state_bytes_per_seq(1, 1),
            "kv_bytes_per_token_tp1": self.kv_bytes_per_token_per_rank(1, 1, 1),
        }


def build_architecture_profile(config: Mapping[str, Any], *, key: str | None = None) -> ModelArchitectureProfile:
    if not isinstance(config, Mapping):
        raise ModelingError("architecture_profile_config_invalid")
    model_key = str(key or config.get("model_key") or config.get("name") or "custom").strip()
    name = str(config.get("name") or model_key)
    model_type = str(config.get("model_type") or config.get("family") or "custom")
    family = str(config.get("family") or model_type)
    num_layers = int(config.get("num_layers") or 0)
    hidden_size = int(config.get("hidden_size") or 0)
    vocab_size = int(config.get("vocab_size") or 32000)
    num_heads = int(config.get("num_attention_heads") or 0)
    num_kv_heads = int(config.get("num_key_value_heads") or num_heads)
    head_dim = int(config.get("head_dim") or (hidden_size // max(1, num_heads) if num_heads else 0))
    if num_layers <= 0 or hidden_size <= 0 or num_heads <= 0 or head_dim <= 0:
        raise ModelingError(f"architecture_profile_dimensions_invalid:{model_key}")

    attn_type = str(config.get("attention_type") or "gqa").strip().lower()
    raw_schedule = config.get("layer_attention_schedule")
    if isinstance(raw_schedule, (list, tuple)) and raw_schedule:
        if len(raw_schedule) != num_layers:
            raise ModelingError(
                f"layer_attention_schedule_length_mismatch:{len(raw_schedule)}!={num_layers}"
            )
        schedule = tuple(str(item).strip().lower() for item in raw_schedule)
    else:
        default_kind = "mla" if "mla" in attn_type else ("gqa" if "gqa" in attn_type else attn_type)
        schedule = tuple(default_kind for _ in range(num_layers))

    recurrent_indices = tuple(i for i, kind in enumerate(schedule) if kind in _RECURRENT_KINDS)
    token_kv_indices = tuple(i for i, kind in enumerate(schedule) if kind in _TOKEN_KV_KINDS)

    lin_cfg = config.get("linear_attention_config")
    lin_map = dict(lin_cfg) if isinstance(lin_cfg, Mapping) else {}
    rec_variant = str(
        lin_map.get("variant") or (schedule[recurrent_indices[0]] if recurrent_indices else "")
    ).strip().lower()
    rec_k_heads = int(lin_map.get("num_k_heads") or num_kv_heads)
    rec_v_heads = int(lin_map.get("num_v_heads") or num_heads)
    rec_k_dim = int(lin_map.get("head_k_dim") or head_dim)
    rec_v_dim = int(lin_map.get("head_v_dim") or head_dim)
    conv_kernel = int(lin_map.get("conv_kernel_size") or (4 if recurrent_indices else 0))
    if recurrent_indices and rec_variant == "gdn" and rec_k_dim != rec_v_dim:
        raise ModelingError(f"gdn_head_k_v_dim_mismatch:{rec_k_dim}!={rec_v_dim}")

    raw_owners = config.get("kv_owner_layers")
    if raw_owners is None:
        owners = list(token_kv_indices)
    else:
        if not isinstance(raw_owners, (list, tuple)):
            raise ModelingError("kv_owner_layers_invalid")
        owners = [int(idx) for idx in raw_owners]

    owner_set: set[int] = set()
    ordered_owners: list[int] = []
    token_kv_set = set(token_kv_indices)
    for idx in owners:
        if idx < 0 or idx >= num_layers:
            raise ModelingError(f"kv_owner_layer_out_of_range:{idx}")
        if idx in recurrent_indices or idx not in token_kv_set:
            raise ModelingError(f"kv_owner_layer_is_recurrent_or_non_kv:{idx}")
        if idx not in owner_set:
            owner_set.add(idx)
            ordered_owners.append(idx)

    raw_consumer_map = config.get("kv_consumer_to_owner")
    explicit_consumer_map: dict[int, int] = {}
    if isinstance(raw_consumer_map, Mapping):
        for k_str, v_idx in raw_consumer_map.items():
            explicit_consumer_map[int(k_str)] = int(v_idx)

    consumers: list[int] = []
    consumer_pairs: list[tuple[int, int]] = []
    for idx in token_kv_indices:
        if idx in owner_set:
            continue
        consumers.append(idx)
        if idx in explicit_consumer_map:
            target_owner = explicit_consumer_map[idx]
            if target_owner not in owner_set or target_owner >= idx:
                raise ModelingError(f"kv_consumer_owner_invalid:{idx}->{target_owner}")
        else:
            earlier_owners = [o for o in ordered_owners if o < idx]
            if not earlier_owners:
                raise ModelingError(f"kv_consumer_missing_earlier_owner:{idx}")
            target_owner = earlier_owners[-1]
        consumer_pairs.append((idx, target_owner))

    latent_moe_raw = config.get("latent_moe_hidden_size")
    latent_moe: int | None = None
    if latent_moe_raw is not None:
        latent_moe = int(latent_moe_raw)
        if latent_moe <= 0 or latent_moe > hidden_size:
            raise ModelingError(f"latent_moe_hidden_size_invalid:{latent_moe}")

    dit_raw = config.get("dit_config")
    dit_map = dict(dit_raw) if isinstance(dit_raw, Mapping) else {}
    visual_tokens = 0
    text_tokens = 0
    total_dit_tokens = 0
    dit_cache_method = str(dit_map.get("dit_cache_method") or "NoCache")
    sparse_method = str(dit_map.get("sparse_attention_method") or "no_sparse")
    ulysses_deg = int(dit_map.get("ulysses_degree") or 1)
    ring_deg = int(dit_map.get("ring_degree") or 1)
    ulysses_any = bool(dit_map.get("ulysses_anything", False))
    if dit_map:
        vid_h = int(dit_map.get("video_height") or 0)
        vid_w = int(dit_map.get("video_width") or 0)
        vid_f = int(dit_map.get("video_frames") or 0)
        vae_t = int(dit_map.get("vae_temporal_stride") or 4)
        vae_s = int(dit_map.get("vae_spatial_stride") or 8)
        patch = list(dit_map.get("patch_size") or (1, 2, 2))
        if len(patch) != 3 or any(int(p) <= 0 for p in patch) or vid_h <= 0 or vid_w <= 0 or vid_f <= 0:
            raise ModelingError("dit_video_dimensions_invalid")
        pt, ph, pw = int(patch[0]), int(patch[1]), int(patch[2])
        latent_t = ((vid_f - 1) // vae_t + 1) if vid_f > 1 else 1
        grid_t = latent_t // pt
        grid_h = vid_h // (vae_s * ph)
        grid_w = vid_w // (vae_s * pw)
        if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
            raise ModelingError("dit_patch_grid_invalid")
        visual_tokens = grid_t * grid_h * grid_w
        text_tokens = max(0, int(dit_map.get("text_sequence_len") or 0))
        total_dit_tokens = visual_tokens + text_tokens
        if (
            dit_cache_method.strip().lower() in _DIT_ACTIVE_CACHE_METHODS
            and sparse_method.strip().lower() in _DIT_ACTIVE_SPARSE_METHODS
        ):
            raise ModelingError("dit_cache_and_sparse_attention_mutually_exclusive")
        if ulysses_deg <= 0 or ring_deg <= 0:
            raise ModelingError("dit_sp_degree_invalid")
        if ulysses_any and ring_deg > 1:
            raise ModelingError("uaa_does_not_support_hybrid_ring_sp")
        sp_world = ulysses_deg * ring_deg
        if not ulysses_any:
            if num_heads % ulysses_deg != 0:
                raise ModelingError(f"dit_sp_head_divisibility_invalid:{num_heads}%{ulysses_deg}")
            if (grid_h % sp_world != 0) and (grid_w % sp_world != 0):
                raise ModelingError(f"dit_sp_spatial_divisibility_invalid:{grid_h}x{grid_w}%{sp_world}")

    ffn_type = str(config.get("ffn_type") or "dense").strip().lower()
    ffn_hidden = int(config.get("ffn_hidden_size") or (4 * hidden_size))
    moe_inter = int(config.get("moe_intermediate_size") or 0)
    num_routed = int(config.get("num_routed_experts") or 0)
    zero_experts = int(config.get("zero_expert_num") or 0)
    num_shared = int(config.get("num_shared_experts") or 0)
    top_k = int(config.get("top_k") or 0)
    first_k_dense = int(config.get("first_k_dense_layers") or 0)
    q_lora = int(config.get("q_lora_rank") or 0)
    kv_lora = int(config.get("kv_lora_rank") or 0)
    qk_nope = int(config.get("qk_nope_head_dim") or 0)
    qk_rope = int(config.get("qk_rope_head_dim") or 0)
    v_dim = int(config.get("v_head_dim") or head_dim)
    sparse_cfg = config.get("sparse_attention_config")
    index_dim = int(
        config.get("index_head_dim")
        or (sparse_cfg.get("index_head_dim", 0) if isinstance(sparse_cfg, Mapping) else 0)
    )
    weight_dtype = str(config.get("weight_dtype") or "bf16")
    kv_dtype = str(config.get("kv_dtype") or "bf16")
    act_dtype = str(config.get("activation_dtype") or "bf16")
    wb = dtype_bytes(weight_dtype)

    moe_layers = max(0, num_layers - first_k_dense) if ffn_type == "moe" and num_routed > 0 else 0
    dense_layers = num_layers - moe_layers
    # Attention + embedding/LM-head + dense FFN + shared experts + optional Latent MoE projections
    embed_params = 2 * vocab_size * hidden_size
    attn_params_per_layer = 4 * hidden_size * (num_heads * head_dim)
    dense_ffn_params = dense_layers * 3 * hidden_size * ffn_hidden
    shared_ffn_params = moe_layers * num_shared * 3 * hidden_size * max(moe_inter, ffn_hidden // 4)
    latent_proj_params = (
        moe_layers * 2 * hidden_size * latent_moe if latent_moe is not None else 0
    )
    dense_weight_bytes = float(
        (embed_params + num_layers * attn_params_per_layer + dense_ffn_params + shared_ffn_params + latent_proj_params)
        * wb
    )
    expert_in_dim = latent_moe if latent_moe is not None else hidden_size
    single_expert_params_all_layers = moe_layers * 3 * expert_in_dim * moe_inter
    single_expert_weight_bytes = float(single_expert_params_all_layers * wb)
    routed_expert_weight_bytes = float(num_routed * single_expert_weight_bytes)

    return ModelArchitectureProfile(
        key=model_key,
        name=name,
        model_type=model_type,
        family=family,
        num_layers=num_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        attention_type=attn_type,
        layer_attention_schedule=schedule,
        recurrent_layer_indices=recurrent_indices,
        token_kv_layer_indices=token_kv_indices,
        physical_kv_owner_layers=tuple(ordered_owners),
        consumer_shared_kv_layers=tuple(consumers),
        kv_consumer_to_owner=tuple(consumer_pairs),
        ffn_type=ffn_type,
        ffn_hidden_size=ffn_hidden,
        moe_intermediate_size=moe_inter,
        num_routed_experts=num_routed,
        zero_expert_num=zero_experts,
        num_shared_experts=num_shared,
        top_k=top_k,
        first_k_dense_layers=first_k_dense,
        latent_moe_hidden_size=latent_moe,
        q_lora_rank=q_lora,
        kv_lora_rank=kv_lora,
        qk_nope_head_dim=qk_nope,
        qk_rope_head_dim=qk_rope,
        v_head_dim=v_dim,
        index_head_dim=index_dim,
        recurrent_variant=rec_variant,
        recurrent_num_k_heads=rec_k_heads,
        recurrent_num_v_heads=rec_v_heads,
        recurrent_head_k_dim=rec_k_dim,
        recurrent_head_v_dim=rec_v_dim,
        conv_kernel_size=conv_kernel,
        visual_tokens=visual_tokens,
        text_tokens=text_tokens,
        total_dit_tokens=total_dit_tokens,
        dit_cache_method=dit_cache_method,
        sparse_attention_method=sparse_method,
        ulysses_degree=ulysses_deg,
        ring_degree=ring_deg,
        ulysses_anything=ulysses_any,
        weight_dtype=weight_dtype,
        kv_dtype=kv_dtype,
        activation_dtype=act_dtype,
        dense_weight_bytes=dense_weight_bytes,
        routed_expert_weight_bytes=routed_expert_weight_bytes,
        single_expert_weight_bytes=single_expert_weight_bytes,
    )


def load_architecture_profile(
    key: str, *, config_root: str | Path | None = None,
) -> ModelArchitectureProfile:
    resolved = _catalog_key(key)
    root = Path(config_root) if config_root is not None else CONFIG_DIR
    if resolved in ARCHITECTURE_PRESET_CATALOG:
        path = root / ARCHITECTURE_PRESET_CATALOG[resolved]
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ModelingError(f"architecture_preset_unreadable:{resolved}:{type(exc).__name__}") from exc
        return build_architecture_profile(config, key=resolved)
    if resolved in CATALOG:
        model = load_model(resolved, config_root=config_root)
        schedule = [plan.attention for plan in model.layer_plans]
        owners = [
            plan.index for plan in model.layer_plans
            if plan.attention in _TOKEN_KV_KINDS and (plan.owns_long_kv or plan.attention == "gqa" or plan.compress_ratio == 0)
        ]
        return build_architecture_profile(
            {
                "model_key": resolved,
                "name": model.name,
                "model_type": dict(model.catalog_facts).get("profile", resolved),
                "family": dict(model.catalog_facts).get("profile", resolved),
                "num_layers": model.layers,
                "hidden_size": model.hidden_size,
                "vocab_size": model.vocab_size,
                "num_attention_heads": model.num_heads,
                "num_key_value_heads": model.num_kv_heads,
                "head_dim": model.head_dim,
                "attention_type": model.attention,
                "layer_attention_schedule": schedule,
                "kv_owner_layers": owners,
                "ffn_type": model.ffn,
                "ffn_hidden_size": model.intermediate_size,
                "moe_intermediate_size": model.moe_intermediate_size,
                "num_routed_experts": model.num_experts,
                "top_k": model.top_k,
                "kv_lora_rank": model.mla.kv_lora_rank if model.mla else 0,
                "qk_rope_head_dim": model.mla.qk_rope_head_dim if model.mla else 0,
            },
            key=resolved,
        )
    raise ModelingError(f"architecture_preset_unknown:{key}")


def list_architecture_profiles() -> list[dict[str, Any]]:
    return [load_architecture_profile(key).to_dict() for key in ARCHITECTURE_PRESET_CATALOG]


def build_model_spec(key: str, config: dict[str, Any]) -> ModelSpec:
    return _build_model_spec(key, config)


def derive_expert_pruned(base_key: str, *, num_experts: int, top_k: int,
                         moe_intermediate_size: int) -> ModelSpec:
    """Immutable per-layer expert-prune of a catalog model.

    Attention family, compress_ratio, indexer and KV geometry are preserved layer by
    layer; only the routed-expert shape changes, so a pruned checkpoint keeps the
    original 43-layer swa/csa/hca split instead of collapsing to a uniform plan.
    """
    from dataclasses import replace

    base = load_model(base_key)
    plans = tuple(
        replace(plan, num_experts=int(num_experts), top_k=int(top_k),
                moe_intermediate_size=int(moe_intermediate_size))
        for plan in base.layer_plans
    )
    derived = replace(base, layer_plans=plans, num_experts=int(num_experts),
                      top_k=int(top_k), moe_intermediate_size=int(moe_intermediate_size))
    return derived


def load_model(key: str, *, config_root: str | Path | None = None) -> ModelSpec:
    key = str(key)
    key = _catalog_key(key)
    root = Path(config_root) if config_root is not None else CONFIG_DIR
    entries = {**CATALOG, **ARCHITECTURE_PRESET_CATALOG}
    path = root / entries[key] if key in entries else Path(key).expanduser()
    if path.is_dir():
        path = path / "config.json"
    if key not in entries and not path.is_file():
        raise ModelingError(f"model_catalog_unknown:{key}")
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ModelingError(f"model_config_size_exceeds_metadata_limit:{key}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ModelingError(f"model_config_unreadable:{key}:{type(exc).__name__}") from exc
    return _build_model_spec(key, config)


def sharded_weight_bytes(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_draft: bool = False, include_engram: bool = True,
) -> float:
    """Largest actual PP-stage per-rank weight footprint."""
    parallel.validate(model)
    from mindie_sim.prediction.structure import normalize_model
    model = normalize_model(model)
    return _sharded(
        model, parallel, node_ranks, include_draft=include_draft,
        include_engram=include_engram,
    )


def model_weight_detail(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1,
) -> dict[str, Any]:
    stages = stage_weight_bytes(model, parallel, node_ranks)
    stages_draft = stage_weight_bytes(
        model, parallel, node_ranks, include_draft=True,
    )
    return {
        "stage_bytes": stages,
        "stage_bytes_with_optional_draft": stages_draft,
        "largest_stage_bytes": max(stages) if stages else 0.0,
        "largest_stage_with_draft_bytes": max(stages_draft) if stages_draft else 0.0,
        "largest_stage": max(range(len(stages)), key=lambda i: stages[i]) if stages else 0,
        "category_bytes": weight_breakdown(model, parallel, node_ranks),
        "logical_body_parameters": body_parameter_count(model),
        "logical_body_weight_bytes": body_weight_bytes(model),
        "node_ranks": node_ranks,
    }


def list_models() -> list[dict[str, Any]]:
    rows = []
    for key in {**CATALOG, **ARCHITECTURE_PRESET_CATALOG}:
        model = load_model(key)
        facts = dict(model.catalog_facts)
        rows.append({
            "key": key, "name": model.name, "layers": model.layers,
            "body_layers": model.body_layers,
            "hidden_size": model.hidden_size, "heads": model.num_heads,
            "experts": model.num_experts, "top_k": model.top_k,
            "parameter_count": model.parameter_count(),
            "resident_weight_bytes": body_weight_bytes(model),
            "attention_types": facts.get("attention_types", {}),
            "ffn_types": facts.get("ffn_types", {}),
            "has_optional_draft": model.has_optional_draft,
            "has_vision": model.vision is not None,
            "engram_tables": facts.get("engram_tables", 0),
            "facts": facts,
            **model_coverage(model),
        })
    return rows


__all__ = [
    "CATALOG", "MODEL_ALIASES", "ARCHITECTURE_PRESET_CATALOG",
    "ARCHITECTURE_PRESET_ALIASES", "CONFIG_DIR", "LayerPlan",
    "ModelArchitectureProfile", "build_architecture_profile",
    "load_architecture_profile", "list_architecture_profiles",
    "load_model", "list_models", "sharded_weight_bytes", "model_weight_detail",
    "stage_weight_terms", "body_weight_bytes", "body_parameter_count",
]
