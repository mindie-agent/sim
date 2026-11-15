"""Generic batch-prefix indexer recipe.

[Role]
  A declared analytical key-tile traversal for shared prefix index reads.  It
  reuses the existing Engine program ABI and keeps score/query/TopK arithmetic
  unchanged; only the physical key-tile fill/read identity is shared.

[Invariants]
  * Selected only with a declared batch reuse mode and a shared prefix pool.
  * A source-native indexer is never relabeled; it keeps its own identity.
  * Work, caps, placement and release boundaries remain the existing ones.

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
RECIPE_NAME = "batch_prefix_indexer"
SUPPORTED_AXES = frozenset({"index_k", "key", "kv"})


@dataclass(frozen=True)
class BatchPrefixIndexer:
    """Key-tile outer traversal feeding participating query tiles."""

    algorithm = RECIPE_NAME

    def lower(self, context):
        from .program import KernelProgram, Stage

        metadata = dict(getattr(context.op, "metadata", ()) or {})
        if metadata.get("prefix_read_reuse_applied") is not True:
            raise ModelingError("batch_prefix_indexer_requires_applied_read_plan")
        component = str(metadata.get("component", "index_k"))
        if component not in SUPPORTED_AXES:
            raise ModelingError(f"batch_prefix_indexer_axis_unsupported:{component}")
        if context.collective_plan is None and "link" in context.native_work:
            # A transport-only op is not an indexer scan.
            raise ModelingError("batch_prefix_indexer_requires_index_work")
        return KernelProgram((
            Stage("key_tiles", context.work),
        ), minimum_ms=context.minimum_ms)


RECIPE = BatchPrefixIndexer()


def select(op: Any, runtime: Any) -> str | None:
    """Return the registered recipe name for an eligible shared index scan."""
    metadata = dict(getattr(op, "metadata", ()) or {})
    if getattr(runtime, "prefix_cache_mode", "private") != "shared":
        return None
    if getattr(runtime, "prefix_read_mode", "auto") == "per_sequence":
        return None
    if metadata.get("prefix_read_reuse_applied") is not True:
        return None
    return RECIPE_NAME
