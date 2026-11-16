"""Tiled sparse indexer, score and TopK physical recipe.

[Role] Execute key/query load, per-head or weighted-query score, activation and
  UB TopK update as finite phases. Final indices are the only materialized
  output of the fused path.
[Invariants] Score arithmetic remains per query; shared dense prefix key reads
  amortize only through the declared batch traversal. A fused score+TopK path
  writes zero score bytes to GM, while an explicitly unfused path retains its
  materialization declaration.
[Timeouts & Locks] Bounded tile and heap work; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass

from .physical_recipe_base import (IO_HBM_CATEGORY, Phase, build_plan,
                                   io_weight, tiled_phased_program)

ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
QUERY_LOAD_SHARE = 0.30
KEY_LOAD_SHARE = 0.70
SCORE_CUBE_SHARE = 1.0
SCORE_VECTOR_SHARE = 0.20
TOPK_VECTOR_SHARE = 0.80
PIPELINE_SLOTS = 2
INDEXER_TILE_CAP = 64


@dataclass(frozen=True)
class TiledSparseIndexerRecipe:
    """Class B: sparse score, activation and UB TopK selection."""

    tile_cap: int = INDEXER_TILE_CAP
    slots: int = PIPELINE_SLOTS
    traffic_rewrite_capable: bool = True

    def build(self, op, geometry):
        return build_plan(self, op, geometry)

    def lower(self, context):
        read_share = io_weight(context, "hbm")
        write_share = io_weight(context, "hbm_write")
        phases = (
            Phase("query_load", {IO_HBM_CATEGORY: QUERY_LOAD_SHARE * read_share,
                                 "mte2": QUERY_LOAD_SHARE}),
            Phase("key_load", {IO_HBM_CATEGORY: KEY_LOAD_SHARE * read_share,
                               "mte2": KEY_LOAD_SHARE}),
            Phase("score", {"cube": SCORE_CUBE_SHARE,
                            "vector": SCORE_VECTOR_SHARE},
                  after=("query_load", "key_load")),
            Phase("topk_update", {"vector": TOPK_VECTOR_SHARE},
                  after=("score",)),
            Phase("store", {IO_HBM_CATEGORY: write_share, "mte3": ONE_FLOAT},
                  after=("topk_update",)),
        )
        return tiled_phased_program(
            context, phases, slots=self.slots, cap=self.tile_cap)


TiledIndexerRecipe = TiledSparseIndexerRecipe
