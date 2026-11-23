"""Deterministic zero-dependency pytest collection sharding helpers.

[Role] Partition collected pytest items into pairwise-disjoint shards whose
  union equals 100% of collected items without third-party plugins.
[Invariants] ``num_shards >= 1`` and ``0 <= shard_index < num_shards``;
  sharding sorts items deterministically by ``nodeid`` before strided slicing.
[Timeouts & Locks] Pure O(N log N) in-memory partition; no I/O or locks.
[Side-Effect Policy] Pure functions; never mutates caller sequences.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def partition_shard_items(
    items: Sequence[T], shard_index: int, num_shards: int,
) -> tuple[list[T], list[T]]:
    """Return ``(selected, deselected)`` for ``shard_index`` of ``num_shards``."""
    if isinstance(num_shards, bool) or isinstance(shard_index, bool):
        raise ValueError("shard_index and num_shards must be integers")
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if not 0 <= shard_index < num_shards:
        raise ValueError(f"shard_index must satisfy 0 <= {shard_index} < {num_shards}")
    if num_shards == 1:
        return list(items), []
    ordered = sorted(items, key=lambda item: getattr(item, "nodeid", str(item)))
    selected = ordered[shard_index::num_shards]
    selected_ids = {id(item) for item in selected}
    deselected = [item for item in ordered if id(item) not in selected_ids]
    return selected, deselected
