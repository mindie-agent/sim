"""Generic shared-KV batch attention tile traversal.

[Role]
  A declared batch-key-tile attention recipe that reuses physical KV ranges
  across participating queries while preserving each request's QK/PV FLOPs,
  causal masks, running-softmax state, sparse selections and spill work.

[Invariants]
  * Only complete immutable physical KV ranges are shared; speculative/private
    rows remain request-specific.
  * Sparse TopK selections are shared only over their actual selected union;
    without selected-row identity the conservative read bound is retained.
  * Scores, outputs, masks and arithmetic are unchanged; only key/value fill and
    core transfer identity differ.

[Timeouts & Locks]
  Pure program construction; no I/O or locks.

[Side-Effect Policy]
  Returns immutable program records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mindie_sim.schema.common import ModelingError


ZERO = 0
ONE = 1
RECIPE_NAME = "batch_prefix_attention"


@dataclass(frozen=True)
class BatchPrefixAttention:
    """Shared key/value tile traversal with per-query softmax state."""

    algorithm = RECIPE_NAME

    def lower(self, context):
        from .program import KernelProgram, Stage

        metadata = dict(getattr(context.op, "metadata", ()) or {})
        if metadata.get("prefix_read_reuse_applied") is not True:
            raise ModelingError("batch_prefix_attention_requires_applied_read_plan")
        if not metadata.get("selected_row_identity") and metadata.get(
                "sparse_selection_unknown"):
            metadata["read_reuse_applied"] = False
            metadata["read_reuse_reason"] = "sparse_selected_rows_unknown"
        query_tiles = max(ONE, int(metadata.get("query_tiles", ONE) or ONE))
        key_tiles = max(ONE, int(metadata.get("key_tiles", ONE) or ONE))
        stages = (
            Stage("query_state", {name: value for name, value in context.work.items()
                                  if name == "vector"}),
            Stage("key_tile_fill", {name: value for name, value in context.work.items()
                                    if name in {"hbm", "mte2", "mte2_aiv"}},
                  after=("query_state",)),
            Stage("qk_pv", {name: value for name, value in context.work.items()
                            if name == "cube"}, after=("key_tile_fill",)),
        )
        return KernelProgram(stages, minimum_ms=context.minimum_ms)


RECIPE = BatchPrefixAttention()


def select(op: Any, runtime: Any) -> str | None:
    """Return the registered recipe name for an eligible shared-KV scan."""
    metadata = dict(getattr(op, "metadata", ()) or {})
    if getattr(runtime, "prefix_cache_mode", "private") != "shared":
        return None
    if getattr(runtime, "prefix_read_mode", "auto") == "per_sequence":
        return None
    if metadata.get("prefix_read_reuse_applied") is not True:
        return None
    if metadata.get("read_reuse_applied") is False:
        return None
    return RECIPE_NAME
