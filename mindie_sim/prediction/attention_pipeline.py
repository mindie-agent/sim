"""Tiled FlashAttention and paged-attention physical recipe.

[Role] Partition Q load, KV refill, QK, online softmax, PV and final store into
  finite phases over a bounded query/head tile pipeline.
[Invariants] QK and PV issue the declared Cube work; softmax and normalization
  issue Vector work; KV refills remain private/L2 service and HBM misses are
  charged once by the traffic planner. No aggregate envelope stage is emitted.
[Timeouts & Locks] Bounded tile count and phase count; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass

from .physical_recipe_base import (COMPUTE_CATEGORIES, IO_HBM_CATEGORY, Phase,
                                   build_plan, io_weight, tiled_phased_program)
from .program import KernelProgram

ZERO_FLOAT = 0.0
ONE = 1
ONE_FLOAT = 1.0
Q_LOAD_SHARE = 0.25
KV_LOAD_SHARE = 0.75
QK_SHARE = 0.55
PV_SHARE = 0.45
SOFTMAX_SHARE = 0.35
NORMALIZE_SHARE = 0.65
PIPELINE_SLOTS = 2
ATTENTION_TILE_CAP = 64


@dataclass(frozen=True)
class TiledFlashAndPagedAttentionRecipe:
    """Class A: attention Q/KV microtiles with coupled score/PV work."""

    tile_cap: int = ATTENTION_TILE_CAP
    slots: int = PIPELINE_SLOTS
    traffic_rewrite_capable: bool = True

    def build(self, op, geometry):
        return build_plan(self, op, geometry)

    def lower(self, context):
        read_share = io_weight(context, "hbm")
        write_share = io_weight(context, "hbm_write")
        phases = (
            Phase("q_load", {IO_HBM_CATEGORY: Q_LOAD_SHARE * read_share,
                             "mte2": Q_LOAD_SHARE}),
            Phase("kv_load", {IO_HBM_CATEGORY: KV_LOAD_SHARE * read_share,
                              "mte2": KV_LOAD_SHARE}),
            Phase("qk", {"cube": QK_SHARE}, after=("q_load", "kv_load")),
            Phase("online_softmax", {"vector": SOFTMAX_SHARE}, after=("qk",)),
            Phase("pv", {"cube": PV_SHARE}, after=("online_softmax",)),
            Phase("normalize_store", {
                "vector": NORMALIZE_SHARE,
                IO_HBM_CATEGORY: write_share,
                "mte3": ONE_FLOAT,
            }, after=("pv",)),
        )
        return tiled_phased_program(
            context, phases, slots=self.slots, cap=self.tile_cap)

    def lower_single(self, context) -> KernelProgram:
        """Diagnostic single-tile lowering retaining the same phase DAG."""
        return tiled_phased_program(context, (
            Phase("q_load", {IO_HBM_CATEGORY: ONE_FLOAT, "mte2": ONE_FLOAT}),
            Phase("kv_load", {}),
            Phase("qk", {"cube": QK_SHARE}, after=("q_load", "kv_load")),
            Phase("online_softmax", {"vector": SOFTMAX_SHARE}, after=("qk",)),
            Phase("pv", {"cube": PV_SHARE}, after=("online_softmax",)),
            Phase("normalize_store", {"vector": NORMALIZE_SHARE,
                                      "mte3": ONE_FLOAT}, after=("pv",)),
        ), slots=self.slots, cap=max(ONE, context.core_count))


TiledAttentionRecipe = TiledFlashAndPagedAttentionRecipe
