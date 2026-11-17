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

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from mindie_sim.prediction.prefix_reads import (
    CacheAccess, apply_prefix_read_plan, plan_prefix_reads,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.runtime_features import canonical_prefix_cache_key


RECIPE_NAME = "batch_prefix_indexer"
SUPPORTED_AXES = frozenset({"index_k", "key", "kv"})


@dataclass(frozen=True)
class BatchPrefixIndexer:
    """Key-tile outer traversal feeding participating query tiles."""

    algorithm = RECIPE_NAME

    def lower(self, context):
        from .sparse_indexer_pipeline import TiledSparseIndexerRecipe

        metadata = dict(getattr(context.op, "metadata", ()) or ())
        if metadata.get("prefix_read_reuse_applied") is not True:
            raise ModelingError("batch_prefix_indexer_requires_applied_read_plan")
        component = str(metadata.get("component", "index_k"))
        if component not in SUPPORTED_AXES:
            raise ModelingError(f"batch_prefix_indexer_axis_unsupported:{component}")
        if context.collective_plan is None and "link" in context.native_work:
            raise ModelingError("batch_prefix_indexer_requires_index_work")
        return TiledSparseIndexerRecipe().lower(context)



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


def apply_indexer_prefix_reads(
    op: Any, *, views: Sequence[Any], rows_by_view: Sequence[int],
    shared_rows_by_view: Sequence[int], row_bytes: float, layer_index: int,
    runtime: Any, kernel_reuse: str = "tile", allow_recipe_rebind: bool = True,
    hbm_component: float | None = None,
    mte2_component: float | None = None,
) -> Any:
    """Replace the precharged index-K components with their physical union."""
    if runtime is None or getattr(runtime, "prefix_cache_mode", "private") != "shared":
        return op
    accesses = []
    for view, rows, shared in zip(views, rows_by_view, shared_rows_by_view):
        if int(rows) <= 0:
            continue
        ranges = getattr(view, "query_ranges", None)
        q_tokens = (
            max(1, sum(int(end) - int(start) for start, end in ranges
                       if int(end) > int(start)))
            if ranges else max(1, int(getattr(view, "q_len", 1) or 1))
        )
        raw_context = getattr(view, "context", None)
        context_tokens = (
            max(0, int(raw_context))
            if raw_context is not None
            else max(0, int(rows) - q_tokens)
        )
        physical_prefix = min(int(shared), int(rows))
        if getattr(view, "prefix_tokens", None) is not None:
            physical_prefix = min(physical_prefix, max(0, int(view.prefix_tokens)))
        accesses.append(CacheAccess(
            request_id=str(view.request_id),
            context_tokens=context_tokens,
            q_len=q_tokens,
            prefix_tokens=max(0, physical_prefix),
            prefix_key=canonical_prefix_cache_key(
                "shared", getattr(view, "prefix_cache_key", "")),
            row_bytes=float(row_bytes),
            layer=int(layer_index),
            component="index_k",
            logical_rows=int(rows),
        ))
    meta = dict(getattr(op, "metadata", ()) or ())
    meta.setdefault("component", "index_k")
    tagged = replace(
        op, metadata=tuple(sorted(meta.items(), key=lambda pair: pair[0])))
    read_mode = str(getattr(runtime, "prefix_read_mode", "auto") or "auto")
    plan = plan_prefix_reads(accesses, mode=read_mode, kernel_reuse=kernel_reuse)
    updated = apply_prefix_read_plan(
        tagged, plan, hbm_component=hbm_component,
        mte2_component=mte2_component)
    if allow_recipe_rebind and select(updated, runtime) == RECIPE_NAME:
        updated = replace(updated, recipe=RECIPE_NAME)
    return updated
