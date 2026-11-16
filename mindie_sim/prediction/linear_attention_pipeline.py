"""Chunked linear-attention physical recipe.

[Role] Execute intra-chunk products, WY transform, FP32 block-inverse merge,
  16-row diagonal updates, recurrent state transition and output projection.
[Invariants] Chunk products and WY issue the declared Cube work. The triangular
  inverse retains dependent AIV row updates and FP32 Cube block products. State
  transition remains ordered per sequence/head; independent chunks overlap only
  through declared slots.
[Timeouts & Locks] Bounded chunk/tile count; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass

from .physical_recipe_base import (IO_HBM_CATEGORY, Phase, build_plan,
                                   io_weight, tiled_phased_program)

ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
PRODUCTS_SHARE = 0.45
WY_SHARE = 0.25
INVERSE_SHARE = 0.20
OUTPUT_SHARE = 0.10
LOAD_READ_SHARE = 0.25
LOAD_WY_SHARE = 0.35
LOAD_INVERSE_SHARE = 0.15
LOAD_OUTPUT_SHARE = 0.25
DIAGONAL_VECTOR_SHARE = 0.55
MERGE_VECTOR_SHARE = 0.20
OUTPUT_VECTOR_SHARE = 0.25
PIPELINE_SLOTS = 2
CHUNK_TILE_CAP = 64


@dataclass(frozen=True)
class ChunkedLinearAttentionRecipe:
    """Class C: chunk products, triangular inverse and recurrent output."""

    tile_cap: int = CHUNK_TILE_CAP
    slots: int = PIPELINE_SLOTS
    traffic_rewrite_capable: bool = True

    def build(self, op, geometry):
        return build_plan(self, op, geometry)

    def lower(self, context):
        read_share = io_weight(context, "hbm")
        write_share = io_weight(context, "hbm_write")
        phases = (
            Phase("chunk_load", {
                IO_HBM_CATEGORY: read_share,
                "mte2": ONE_FLOAT,
            }),
            Phase("chunk_products", {
                "cube": PRODUCTS_SHARE,
                "vector": DIAGONAL_VECTOR_SHARE,
            }, after=("chunk_load",)),
            Phase("wy_transform", {"cube": WY_SHARE,
                                   "vector": MERGE_VECTOR_SHARE},
                  after=("chunk_products",)),
            Phase("block_inverse_merge", {
                "cube": INVERSE_SHARE,
                "vector": MERGE_VECTOR_SHARE,
            }, after=("wy_transform",)),
            Phase("recurrent_state", {
                "vector": DIAGONAL_VECTOR_SHARE,
            }, after=("block_inverse_merge",)),
            Phase("output_projection", {
                "cube": OUTPUT_SHARE,
                "vector": OUTPUT_VECTOR_SHARE,
                IO_HBM_CATEGORY: write_share,
                "mte3": ONE_FLOAT,
            }, after=("recurrent_state",)),
        )
        return tiled_phased_program(
            context, phases, slots=self.slots, cap=self.tile_cap)


ChunkedLinearAttention = ChunkedLinearAttentionRecipe
