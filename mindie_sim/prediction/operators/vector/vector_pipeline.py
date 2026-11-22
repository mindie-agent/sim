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

from dataclasses import dataclass, replace

from mindie_sim.prediction.physical_recipe_base import (IO_HBM_CATEGORY, Phase, build_plan,
                                   io_weight, tiled_phased_program)

MS_PER_SECOND = 1000
HALF_SHARE = 0.5
STREAM_LOAD_SHARE = 0.30
STREAM_STORE_SHARE = 0.70
PIPELINE_SLOTS = 2
# AscendC double-buffered VECIN(2)+VECOUT(2) allows MTE2, VEC and MTE3 to run
# concurrently across 3 in-flight tiles in 3-phase vector pipelines.
THREE_STAGE_PIPELINE_SLOTS = 3
AIV_TILE_CAP = 64
HBM_RW_ASYMMETRY_RATIO = 1.5
# CANN ops-math/math/mul/op_kernel/mul_apt.cpp (BroadcastSch: Brcb/Dup + op + mask).
BROADCAST_VECTOR_OP_FACTOR = 3.0
# CANN ops-math/math/reduce_sum/op_kernel/reduce_sum_apt.cpp (ReduceOpTilingDataNonContiguous).
OUTER_REDUCE_VECTOR_OP_FACTOR = 2.0
NON_STREAMING_VECTOR_SUBMODES = frozenset({"broadcast_elementwise", "outer_axis_reduction"})
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

    def bind_context(self, context):
        """Use declared sustainable DRAM, die-wide L2 and CANN submode rates."""
        submode = self.submode(context.op)
        op = context.op
        hbm_bytes = float(getattr(op, "hbm_bytes", 0.0) or 0.0)
        read_bytes = float(getattr(op, "mte2_bytes", 0.0) or 0.0)
        write_bytes = float(getattr(op, "mte3_bytes", 0.0) or 0.0)
        hbm_read = max(0.0, hbm_bytes - write_bytes)
        hbm_write = min(hbm_bytes, write_bytes)
        if submode == "write_only" or hbm_read <= 0.0 or hbm_write > HBM_RW_ASYMMETRY_RATIO * hbm_read:
            dram_rate = context.service_rate("hbm_write_dram") or context.service_rate("mte2_aiv_dram")
        elif hbm_write <= 0.0:
            dram_rate = context.service_rate("mte2_aiv_dram")
        elif read_bytes > HBM_RW_ASYMMETRY_RATIO * write_bytes:
            dram_rate = context.service_rate("hbm_rw21_dram") or context.service_rate("mte2_aiv_dram")
        else:
            dram_rate = (
                context.service_rate("hbm_rw11_dram")
                or context.service_rate("hbm_rw21_dram")
                or context.service_rate("mte2_aiv_dram")
            )
        updated_modes = dict(context.service_modes)
        changed = False
        if dram_rate is not None and "hbm" in context.resource_names:
            # Only service units change. Native HBM bytes and on-chip MTE2
            # refills remain intact; reads and writes contend on the same bus.
            ceiling = min(dram_rate,
                          context.rates[context.resource_names["hbm"]] * MS_PER_SECOND)
            updated_modes["hbm"] = ("hbm", ceiling)
            changed = True
        if submode in NON_STREAMING_VECTOR_SUBMODES and "vector" in context.resource_names:
            vec_rate = context.rates[context.resource_names["vector"]] * MS_PER_SECOND
            op_factor = (
                BROADCAST_VECTOR_OP_FACTOR
                if submode == "broadcast_elementwise"
                else OUTER_REDUCE_VECTOR_OP_FACTOR
            )
            updated_modes["vector"] = ("vector", vec_rate / op_factor)
            changed = True
        l2_hit_bytes = max(0.0, read_bytes + write_bytes - hbm_bytes)
        if (
            l2_hit_bytes > 0.0
            and read_bytes > 0.0
            and submode in NON_STREAMING_VECTOR_SUBMODES
        ):
            l2_mode = "l2_die_bw" if submode == "outer_axis_reduction" else "l2_contiguous_bw"
            l2_bw = context.service_rate(l2_mode) or context.service_rate("l2_die_bw")
            mte2_role = (
                "mte2_aiv"
                if "mte2_aiv" in context.resource_names
                else ("mte2" if "mte2" in context.resource_names else None)
            )
            if l2_bw is not None and l2_bw > 0.0 and mte2_role is not None:
                # Shared die-wide L2 bandwidth bounds combined L2 read+write traffic.
                eff_mte2 = l2_bw * (read_bytes / (read_bytes + write_bytes))
                mte2_peak = context.rates[context.resource_names[mte2_role]] * MS_PER_SECOND
                updated_modes["mte2"] = (mte2_role, min(eff_mte2, mte2_peak))
                changed = True
        return replace(context, service_modes=updated_modes) if changed else context

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
        declared_submode = str(metadata.get("vector_submode") or "")
        if declared_submode in NON_STREAMING_VECTOR_SUBMODES:
            return declared_submode
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
        read_share = io_weight(context, "hbm_read")
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
        slots = (
            max(self.slots, THREE_STAGE_PIPELINE_SLOTS)
            if submode in NON_STREAMING_VECTOR_SUBMODES
            else self.slots
        )
        return tiled_phased_program(
            context, phases, slots=slots, cap=self.tile_cap)


AIVStreamingPipeline = AIVTiledVectorAndGatherRecipe
AIVDynamicQuantRecipe = AIVTiledVectorAndGatherRecipe
AIVAtomicScatterCombineRecipe = AIVTiledVectorAndGatherRecipe
