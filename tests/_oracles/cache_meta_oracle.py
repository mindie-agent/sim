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
from __future__ import annotations

from .cache_meta_layers import (  # noqa: F401
    INDEX_INTERLEAVE,
    KV_CACHE_PLANES,
    PLE_BF16_BYTES,
    PLE_DEFAULT_CONV_KERNEL,
    PLE_DEFAULT_DIVISOR,
    POOL_TAIL_BYTES,
    POOL_TAIL_SCALE,
    REFERENCE_PINS,
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
from .cache_meta_totals import (
    build_layers,
    cache_layout_declaration,
    cache_totals,
)

__all__ = [
    "REFERENCE_PINS",
    "RefLayer",
    "build_layers",
    "cache_layout_declaration",
    "cache_totals",
    "config_profile",
    "flatten_config",
]
