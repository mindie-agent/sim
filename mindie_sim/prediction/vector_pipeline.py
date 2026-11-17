"""AIV tiled vector, reduction, quantization, gather and scatter recipe.

[Role] Execute streaming, norm/reduction, dynamic quantization, gather, atomic
  scatter, recurrence and write-only families as finite AIV tile phases.
[Invariants] Every tile owns a declared fraction of the operator's physical
  work. Reduction and quantization retain their multi-pass dependencies; an
  atomic scatter uses the declared RMW service instead of a free write.
[Timeouts & Locks] Bounded AIV tile count; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass

from .physical_recipe_base import (IO_HBM_CATEGORY, Phase, build_plan,
                                   io_weight, tiled_phased_program)

HALF_SHARE = 0.5
STREAM_LOAD_SHARE = 0.30
STREAM_STORE_SHARE = 0.70
PIPELINE_SLOTS = 2
AIV_TILE_CAP = 64
NORM_FAMILIES = frozenset({"rmsnorm", "layernorm", "mhc"})
REDUCTION_FAMILIES = frozenset({"attention_residual", "softmax"})
QUANT_FAMILIES = frozenset({"quantize"})
SCATTER_RECIPES = frozenset({"route_unpack"})
WRITE_ONLY_RECIPES = frozenset({"core_output_zero"})
RECURRENCE_FAMILIES = frozenset({"conv1d", "gdn_gating"})


@dataclass(frozen=True)
class AIVTiledVectorAndGatherRecipe:
    """Class D: one finite AIV mechanism with declared submode phases."""

    tile_cap: int = AIV_TILE_CAP
    slots: int = PIPELINE_SLOTS
    traffic_rewrite_capable: bool = True

    def build(self, op, geometry):
        return build_plan(self, op, geometry)

    @staticmethod
    def submode_from_family(family: str, recipe: str) -> str:
        if family in QUANT_FAMILIES:
            return "quantization"
        if family in NORM_FAMILIES or family in REDUCTION_FAMILIES:
            return "reduction"
        if recipe in SCATTER_RECIPES:
            return "atomic_scatter"
        if recipe in WRITE_ONLY_RECIPES:
            return "write_only"
        if family in RECURRENCE_FAMILIES:
            return "recurrence"
        if family in {"embedding", "routing", "engram_gather", "engram_query", "tensor_select"}:
            return "gather"
        return "streaming"

    @staticmethod
    def submode(op) -> str:
        family = str(getattr(op, "family", ""))
        recipe = str(getattr(op, "recipe", ""))
        metadata = dict(getattr(op, "metadata", ()) or ())
        if family in QUANT_FAMILIES:
            return "quantization"
        if family in NORM_FAMILIES or family in REDUCTION_FAMILIES:
            return "reduction"
        if recipe in SCATTER_RECIPES and int(metadata.get("top_k", 1) or 1) > 1:
            return "atomic_scatter"
        if recipe in WRITE_ONLY_RECIPES:
            return "write_only"
        if family in RECURRENCE_FAMILIES:
            return "recurrence"
        if family in {"embedding", "routing", "engram_gather", "engram_query", "tensor_select"}:
            return "gather"
        return "streaming"

    def lower(self, context):
        submode = self.submode(context.op)
        read_share = io_weight(context, "hbm")
        write_share = io_weight(context, "hbm_write")
        if submode == "write_only":
            phases = (
                Phase("tile_fill", {"vector": 1.0,
                                    IO_HBM_CATEGORY: 1.0,
                                    "mte3": 1.0}),
            )
        elif submode == "quantization":
            phases = (
                Phase("row_load", {IO_HBM_CATEGORY: read_share,
                                   "mte2": 1.0}),
                Phase("absmax_reduce", {"vector": HALF_SHARE},
                      after=("row_load",)),
                Phase("scale_cast_store", {
                    "vector": HALF_SHARE,
                    IO_HBM_CATEGORY: write_share,
                    "mte3": 1.0,
                }, after=("absmax_reduce",)),
            )
        elif submode == "reduction":
            phases = (
                Phase("row_load", {IO_HBM_CATEGORY: read_share,
                                   "mte2": 1.0}),
                Phase("statistics", {"vector": HALF_SHARE},
                      after=("row_load",)),
                Phase("normalize_store", {
                    "vector": HALF_SHARE,
                    IO_HBM_CATEGORY: write_share,
                    "mte3": 1.0,
                }, after=("statistics",)),
            )
        elif submode == "atomic_scatter":
            phases = (
                Phase("row_load", {IO_HBM_CATEGORY: read_share,
                                   "mte2": 1.0}),
                Phase("atomic_rmw", {"vector": HALF_SHARE},
                      after=("row_load",)),
                Phase("combine_store", {
                    "vector": HALF_SHARE,
                    IO_HBM_CATEGORY: write_share,
                    "mte3": 1.0,
                }, after=("atomic_rmw",)),
            )
        elif submode == "recurrence":
            phases = (
                Phase("history_load", {IO_HBM_CATEGORY: read_share,
                                       "mte2": 1.0}),
                Phase("window_update", {"vector": HALF_SHARE},
                      after=("history_load",)),
                Phase("history_store", {
                    "vector": HALF_SHARE,
                    IO_HBM_CATEGORY: write_share,
                    "mte3": 1.0,
                }, after=("window_update",)),
            )
        elif submode == "gather":
            phases = (
                Phase("index_load", {IO_HBM_CATEGORY: read_share,
                                     "mte2": 1.0}),
                Phase("gather_rows", {"vector": 1.0},
                      after=("index_load",)),
                Phase("gather_store", {
                    IO_HBM_CATEGORY: write_share,
                    "mte3": 1.0,
                }, after=("gather_rows",)),
            )
        else:
            phases = (
                Phase("tile_load", {IO_HBM_CATEGORY: read_share,
                                    "mte2": 1.0}),
                Phase("tile_compute", {"vector": 1.0,
                                       "cube": 1.0},
                      after=("tile_load",)),
                Phase("tile_store", {
                    IO_HBM_CATEGORY: write_share,
                    "mte3": 1.0,
                }, after=("tile_compute",)),
            )
        if any(float(value) > 0.0 for key, value in context.work.items()
               if "link" in str(key).split(":")):
            # A declared state/transfer payload still needs its physical link
            # service even when the value transform runs on AIV.
            phases = phases + (Phase("link_service", {"link": 1.0}),)
        for pipe in ("mte1", "fixpipe"):
            if any(float(value) > 0.0 and str(key).startswith(pipe)
                   for key, value in context.work.items()):
                # A fused/vector op that declares local operand staging keeps
                # that physical pipe service in its own phase.
                phases = phases + (Phase(f"{pipe}_service", {pipe: 1.0}),)
        return tiled_phased_program(
            context, phases, slots=self.slots, cap=self.tile_cap)


AIVStreamingPipeline = AIVTiledVectorAndGatherRecipe
AIVDynamicQuantRecipe = AIVTiledVectorAndGatherRecipe
AIVAtomicScatterCombineRecipe = AIVTiledVectorAndGatherRecipe
