"""Typed quantization, draft, vision and Engram specifications for catalog models.

[Role]
  Hold the reusable, table-driven semantics needed by the nine-model catalog:
  packed weight/scale byte formulas, optional draft/MTP/verification geometry,
  vision-tower/media expansion geometry and Engram lookup/tables.  Model-family
  adapters in ``a3_catalog_specs`` only populate these records.

[Invariants]
  * Weight-file bytes, loaded/runtime residency bytes, activation bytes and
    collective payload bytes are separate quantities.
  * FP4 packing is rows*ceil(cols/2) plus E8M0 groups rows*ceil(cols/32);
    scales never disappear into a generic compression ratio.
  * FP8 block scales use explicit block shape and scale dtype; GLM's FP32
    scales differ from V4/V41 E8M0 scales.
  * Unquantized exclusions are category-based and explicit; router, norms,
    LM head, embeddings, vision and selected draft paths stay in their native
    format.
  * A missing vision/draft/Engram declaration raises on use instead of
    silently producing zero work.
  * Source revisions are provenance only and are not admission keys.

[Timeouts & Locks]
  Pure immutable records and arithmetic.  No I/O, locks, sleeps or devices.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .a3_types import A3InputError

_FP32_CATEGORIES = {
    "param_fp32", "mhc", "ape", "conv_weight_f32", "a_log", "dt_bias",
    "compressor_fp32_runtime",
}
_BF16_CATEGORIES = {
    "norm", "router", "router_bias", "lm_head", "embedding", "vision",
    "projector", "vision_pos", "image_token", "engram_gate", "dspark_input",
    "compressor_bf16_runtime", "target_alias_embedding", "target_alias_lm_head",
}
_RUNTIME_OVERRIDE = {
    "engram_gate": "bf16",
    "dspark_input": "bf16",
    "engram_table": "fp8_e4m3",
}
_EXPERT_CATEGORIES = {"routed_expert", "dspark_routed_expert"}
_SHARED_CATEGORIES = {"shared_expert"}
_FP8_CATEGORIES = {
    "attn", "mla", "gdn", "kda", "dense_mlp", "compressor", "indexer",
    "indexer_fp8", "shared_expert",
}


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


@dataclass(frozen=True)
class QuantSpec:
    """Explicit representation plan; never a single global activation dtype."""

    name: str
    weight_format: str = "bf16"
    activation_format: str = "bf16"
    expert_weight_format: str = "bf16"
    weight_block: tuple[int, int] = (0, 0)
    expert_block: int = 32
    scale_format: str = "fp32"
    scale_bytes: float = 4.0
    kv_dtype: str = "bf16"
    index_dtype: str = "int8"
    index_scale_dtype: str = "fp16"
    exclusions: tuple[str, ...] = ()
    engram_table_mode: str = "device_bf16"
    source: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()

    def format_for(self, category: str, *, source: bool = False) -> str:
        cat = str(category)
        # Native source FP32 categories are never downgraded by a vector/gate
        # exclusion.  Exclusions describe "do not quantize", not "cast to BF16".
        if cat in _FP32_CATEGORIES:
            return "fp32"
        if cat in {"conv_weight_bf16", "linear_param_bf16"}:
            return "bf16"
        if cat in self.exclusions:
            return "bf16"
        if cat in _BF16_CATEGORIES:
            return "bf16"
        if cat == "conv_weight_f32":
            return "fp32"
        if cat in _EXPERT_CATEGORIES:
            return self.expert_weight_format
        if cat in _SHARED_CATEGORIES:
            return "bf16" if self.weight_format == "bf16" else self.weight_format
        if cat in _FP8_CATEGORIES:
            return self.weight_format
        if cat == "engram_table":
            return self.engram_table_format
        if cat == "norm_f32":
            return "fp32"
        return self.weight_format if self.weight_format != "bf16" else "bf16"

    def source_format_for(self, category: str) -> str:
        """Checkpoint representation before an explicit runtime conversion."""
        cat = str(category)
        if cat in {"compressor_fp32_runtime", "compressor_bf16_runtime"}:
            return "bf16"
        if cat == "engram_table":
            return "fp8_e4m3" if self.weight_format != "bf16" else "bf16"
        if cat == "engram_gate":
            return self.weight_format if self.weight_format != "bf16" else "bf16"
        if cat == "dspark_input":
            return self.weight_format if self.weight_format != "bf16" else "bf16"
        if cat == "engram_table":
            return self.weight_format if self.weight_format != "bf16" else "bf16"
        return self.format_for(cat, source=True)

    def matrix_bytes(self, rows: int, cols: int, category: str, *, source: bool = False) -> float:
        rows, cols = int(rows), int(cols)
        if rows <= 0 or cols <= 0:
            raise A3InputError(f"catalog_weight_shape_invalid:{category}:{rows}x{cols}")
        fmt = self.source_format_for(category) if source else self.format_for(category)
        if str(category) == "engram_table":
            # Engram tables have explicit host/device storage modes.  Scales
            # are one per row per32 columns, not a 2-D block grid.
            mode = "host_fp8_e8m0" if source else self.engram_table_mode
            if mode == "device_bf16":
                return float(rows * cols * 2)
            if mode == "device_int8_fp32":
                return float(rows * cols + rows * _ceil_div(cols, 32) * 4)
            if mode in {"host_fp8_e8m0", "source_fp8_e8m0"}:
                return float(rows * cols + rows * _ceil_div(cols, 32))
            raise A3InputError(f"catalog_engram_table_mode_unsupported:{mode}")
        if fmt in {"bf16", "bfloat16", "fp16", "float16"}:
            return float(rows * cols * 2)
        if fmt in {"fp32", "float32"}:
            return float(rows * cols * 4)
        if fmt in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
            data = float(rows * cols)
            block_rows, block_cols = self.weight_block
            if block_rows <= 0 or block_cols <= 0:
                raise A3InputError("catalog_fp8_block_missing")
            scale_rows = _ceil_div(rows, block_rows)
            scale_cols = _ceil_div(cols, block_cols)
            return data + scale_rows * scale_cols * float(self.scale_bytes)
        if fmt in {"mxfp4", "fp4", "fp4_e2m1"}:
            group = max(1, int(self.expert_block))
            data = float(rows * _ceil_div(cols, 2))
            scales = float(rows * _ceil_div(cols, group))
            return data + scales
        if fmt in {"int8", "w8a8"}:
            return float(rows * cols)
        raise A3InputError(f"catalog_weight_format_unsupported:{fmt}")

    @property
    def engram_table_format(self) -> str:
        mode = self.engram_table_mode
        if mode == "device_bf16":
            return "bf16"
        if mode == "device_int8_fp32":
            return "int8"
        return "fp8_e4m3"

    def logical_parameters(self, rows: int, cols: int) -> int:
        return int(rows) * int(cols)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "weight_format": self.weight_format,
            "activation_format": self.activation_format,
            "expert_weight_format": self.expert_weight_format,
            "weight_block": list(self.weight_block),
            "expert_block": self.expert_block,
            "scale_format": self.scale_format, "scale_bytes": self.scale_bytes,
            "kv_dtype": self.kv_dtype, "index_dtype": self.index_dtype,
            "index_scale_dtype": self.index_scale_dtype,
            "exclusions": list(self.exclusions),
            "source": dict(self.source), "notes": list(self.notes),
        }


@dataclass(frozen=True)
class VisionSpec:
    kind: str
    layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    qkv_width: int
    patch_size: int
    in_channels: int = 3
    temporal_patch: int = 1
    merge_area: int = 1
    downsample_ratio: int = 1
    out_hidden_size: int = 0
    aligner_hidden_size: int = 0
    pos_embed_tokens: int = 0
    deepstack: bool = False
    gated_mlp: bool = False
    activation: str = "gelu"
    norm_type: str = "layernorm"
    norm_eps: float = 1e-6
    bias: bool = True
    image_token_id: int = 0
    video_token_id: int = 0
    source: tuple[tuple[str, str], ...] = ()

    @property
    def patch_input_width(self) -> int:
        return int(self.in_channels * self.patch_size ** 2 * self.temporal_patch)

    @property
    def aligner_input_width(self) -> int:
        return int(self.hidden_size * self.merge_area)

    @property
    def query_width(self) -> int:
        """Single-Q projection width; packed QKV rows are three times this."""
        return self.qkv_width // 3

    @property
    def head_dim(self) -> int:
        if self.num_heads <= 0 or self.qkv_width <= 0:
            return 0
        return self.query_width // self.num_heads

    def merged_tokens(self, patch_tokens: int) -> int:
        return max(1, int(math.ceil(int(patch_tokens) / max(1, self.merge_area))))

    def patch_tokens_for_media(self, media_tokens: int) -> int:
        """Media tokens are post-merge text positions; return pre-merge patches."""
        return max(1, int(media_tokens) * max(1, self.merge_area))

    def validate(self) -> None:
        for name, value in (
            ("layers", self.layers), ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_heads", self.num_heads), ("qkv_width", self.qkv_width),
            ("patch_size", self.patch_size), ("merge_area", self.merge_area),
            ("out_hidden_size", self.out_hidden_size),
        ):
            if int(value) <= 0:
                raise A3InputError(f"catalog_vision_field_invalid:{name}")
        if self.qkv_width % 3 != 0 or self.query_width % self.num_heads != 0:
            raise A3InputError(
                f"catalog_vision_qkv_not_divisible:{self.qkv_width}%3/"
                f"{self.query_width}%{self.num_heads}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "layers": self.layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_heads": self.num_heads, "head_dim": self.head_dim,
            "query_width": self.query_width,
            "qkv_width": self.qkv_width, "qkv_packed_rows": self.qkv_width,
            "patch_size": self.patch_size,
            "in_channels": self.in_channels, "temporal_patch": self.temporal_patch,
            "merge_area": self.merge_area, "downsample_ratio": self.downsample_ratio,
            "out_hidden_size": self.out_hidden_size,
            "aligner_hidden_size": self.aligner_hidden_size,
            "aligner_input_width": self.aligner_input_width,
            "patch_input_width": self.patch_input_width,
            "pos_embed_tokens": self.pos_embed_tokens,
            "deepstack": self.deepstack, "gated_mlp": self.gated_mlp,
            "activation": self.activation, "norm_type": self.norm_type,
            "norm_eps": self.norm_eps, "bias": self.bias,
            "image_token_id": self.image_token_id,
            "video_token_id": self.video_token_id,
            "source": dict(self.source),
        }


@dataclass(frozen=True)
class DraftSpec:
    """Optional MTP/DSpark proposal, verification and Markov/confidence state."""

    kind: str
    layers: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    q_width: int
    kv_width: int
    attention_kind: str = "gqa"
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    o_lora_rank: int = 0
    o_groups: int = 0
    num_experts: int = 0
    top_k: int = 0
    expert_intermediate_size: int = 0
    num_shared_experts: int = 0
    target_layer_ids: tuple[int, ...] = ()
    block_size: int = 1
    markov_rank: int = 0
    confidence_enabled: bool = False
    confidence_executes: bool = False
    confidence_input_width: int = 0
    target_shared_embedding: bool = False
    target_shared_lm_head: bool = False
    base_logits: bool = False
    markov_replicated: bool = True
    noncausal_query_block: bool = False
    pp_capability: tuple[int, ...] = (1,)
    vocab_size: int = 0
    mhc_mult: int = 0
    mhc_sinkhorn_iters: int = 0
    activation_dtype: str = "bf16"
    input_projection_category: str = "dspark_input"
    source: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def input_width(self) -> int:
        if self.kind in {"dspark", "mtp_dspark"} and self.target_layer_ids:
            return int(len(self.target_layer_ids) * self.hidden_size)
        if self.kind == "standalone_draft" and self.target_layer_ids:
            return int(len(self.target_layer_ids) * self.hidden_size)
        return int(2 * self.hidden_size)

    @property
    def proposal_tokens(self) -> int:
        return int(self.block_size if self.kind in {"dspark", "mtp_dspark", "standalone_draft"} else 1)

    @property
    def verify_tokens(self) -> int:
        return int(self.block_size if self.kind in {"dspark", "mtp_dspark", "standalone_draft"} else 1)

    def validate(self) -> None:
        for name, value in (
            ("layers", self.layers), ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("num_heads", self.num_heads), ("num_kv_heads", self.num_kv_heads),
            ("head_dim", self.head_dim),
        ):
            if int(value) <= 0:
                raise A3InputError(f"catalog_draft_field_invalid:{name}")
        if self.q_width <= 0 or self.kv_width <= 0:
            raise A3InputError("catalog_draft_projection_width_invalid")
        if self.num_experts and self.top_k <= 0:
            raise A3InputError("catalog_draft_topk_missing")
        if self.kind in {"dspark", "mtp_dspark", "standalone_draft"} and self.block_size <= 0:
            raise A3InputError("catalog_dspark_block_missing")
        if self.kind in {"dspark", "mtp_dspark", "standalone_draft"} and self.markov_rank <= 0:
            raise A3InputError("catalog_dspark_markov_missing")
        if self.confidence_enabled:
            expected = self.hidden_size + self.markov_rank
            if self.confidence_input_width not in {0, expected}:
                raise A3InputError(
                    f"catalog_dspark_confidence_width_invalid:{self.confidence_input_width}!={expected}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "layers": self.layers,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "attention_kind": self.attention_kind,
            "num_heads": self.num_heads, "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim, "q_width": self.q_width,
            "kv_width": self.kv_width, "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "v_head_dim": self.v_head_dim, "o_lora_rank": self.o_lora_rank,
            "o_groups": self.o_groups, "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_intermediate_size": self.expert_intermediate_size,
            "num_shared_experts": self.num_shared_experts,
            "target_layer_ids": list(self.target_layer_ids),
            "block_size": self.block_size, "markov_rank": self.markov_rank,
            "confidence_enabled": self.confidence_enabled,
            "confidence_executes": self.confidence_executes,
            "confidence_input_width": self.confidence_input_width,
            "target_shared_embedding": self.target_shared_embedding,
            "target_shared_lm_head": self.target_shared_lm_head,
            "base_logits": self.base_logits,
            "markov_replicated": self.markov_replicated,
            "noncausal_query_block": self.noncausal_query_block,
            "pp_capability": list(self.pp_capability),
            "input_width": self.input_width,
            "proposal_tokens": self.proposal_tokens,
            "vocab_size": self.vocab_size, "activation_dtype": self.activation_dtype,
            "source": dict(self.source), "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EngramTableSpec:
    layer_id: int
    table_id: int
    rows: int
    width: int
    source_dtype: str = "fp8_e4m3"
    scale_dtype: str = "fp8_e8m0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id, "table_id": self.table_id,
            "rows": self.rows, "width": self.width,
            "source_dtype": self.source_dtype, "scale_dtype": self.scale_dtype,
        }


@dataclass(frozen=True)
class EngramSpec:
    tables: tuple[EngramTableSpec, ...]
    max_ngram: int
    heads: int
    width: int
    pad_token_id: int = 0
    no_dedup: bool = True
    storage_mode: str = "device_bf16"
    host_response_bytes: int = 512
    id_bytes: int = 8
    source: tuple[tuple[str, str], ...] = ()

    @property
    def columns_per_table(self) -> int:
        return max(0, (self.max_ngram - 1) * self.heads)

    @property
    def lookups_per_token_per_table(self) -> int:
        return self.columns_per_table

    @property
    def lookups_per_token(self) -> int:
        return self.columns_per_table * len(self.tables)

    def validate(self) -> None:
        if not self.tables:
            raise A3InputError("catalog_engram_tables_missing")
        if self.max_ngram < 2 or self.heads <= 0 or self.width <= 0:
            raise A3InputError("catalog_engram_geometry_invalid")
        for table in self.tables:
            if table.rows <= 0 or table.width != self.width:
                raise A3InputError(f"catalog_engram_table_invalid:{table.layer_id}")

    @property
    def row_bytes(self) -> float:
        width = int(self.width)
        if self.storage_mode == "device_bf16":
            return float(width * 2)
        if self.storage_mode == "device_int8_fp32":
            return float(width + _ceil_div(width, 32) * 4)
        if self.storage_mode == "host_fp8_e8m0":
            return float(width + _ceil_div(width, 32))
        raise A3InputError(f"catalog_engram_storage_mode_unsupported:{self.storage_mode}")

    def table_storage_bytes(self, *, source: bool = False, node_rows: int | None = None) -> float:
        total = 0.0
        for table in self.tables:
            rows = table.rows
            if node_rows is not None and node_rows > 1:
                rows = int(math.ceil(rows / node_rows))
            if source:
                total += rows * (table.width + _ceil_div(table.width, 32))
            elif self.storage_mode == "device_bf16":
                total += rows * table.width * 2
            elif self.storage_mode == "device_int8_fp32":
                total += rows * (table.width + _ceil_div(table.width, 32) * 4)
            else:
                total += rows * (table.width + _ceil_div(table.width, 32))
        return float(total)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tables": [table.to_dict() for table in self.tables],
            "max_ngram": self.max_ngram, "heads": self.heads, "width": self.width,
            "columns_per_table": self.columns_per_table,
            "lookups_per_token_per_table": self.lookups_per_token_per_table,
            "lookups_per_token": self.lookups_per_token,
            "pad_token_id": self.pad_token_id, "no_dedup": self.no_dedup,
            "storage_mode": self.storage_mode, "row_bytes": self.row_bytes,
            "host_response_bytes": self.host_response_bytes,
            "id_bytes": self.id_bytes,
            "source": dict(self.source),
        }


@dataclass(frozen=True)
class AuxiliarySpec:
    """Model-level optional branches kept outside the body layer schedule."""

    mtp: DraftSpec | None = None
    dspark: DraftSpec | None = None
    external_draft_key: str = ""
    standalone_draft: bool = False
    include_vision_weights: bool = True
    notes: tuple[str, ...] = ()

    def enabled_draft(self, mode: str) -> DraftSpec | None:
        if mode in {"dspark", "verify", "draft_verify"}:
            return self.dspark or self.mtp
        if mode in {"mtp", "draft"}:
            return self.mtp or self.dspark
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mtp": self.mtp.to_dict() if self.mtp else None,
            "dspark": self.dspark.to_dict() if self.dspark else None,
            "external_draft_key": self.external_draft_key,
            "standalone_draft": self.standalone_draft,
            "include_vision_weights": self.include_vision_weights,
            "notes": list(self.notes),
        }


def merge_source(
    base: Mapping[str, Any], extra: Mapping[str, Any] | None = None,
) -> tuple[tuple[str, str], ...]:
    items = {str(k): str(v) for k, v in dict(base).items() if v is not None}
    if extra:
        items.update({str(k): str(v) for k, v in dict(extra).items() if v is not None})
    return tuple(sorted(items.items()))


__all__ = [
    "QuantSpec", "VisionSpec", "DraftSpec", "EngramTableSpec", "EngramSpec",
    "AuxiliarySpec", "merge_source",
]
