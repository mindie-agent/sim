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
from math import prod

from mindie_sim.prediction.physical_recipe_base import (
    IO_HBM_CATEGORY,
    Phase,
    _phase_amounts,
    _phase_caps,
    _tile_count,
    build_plan,
    io_weight,
    phased_program,
    tiled_phased_program,
)
from mindie_sim.prediction.program.pipeline import PipelineProgram
from mindie_sim.prediction.program.stages import KernelProgram, Stage
from mindie_sim.schema.common import ModelingError, dtype_bytes

MS_PER_SECOND = 1000
HALF_SHARE = 0.5
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
PRECOMPUTED_SCALE_QUANT_RECIPES = frozenset({"quant_allreduce_quantize"})
SCATTER_RECIPES = frozenset({"route_unpack"})
WRITE_ONLY_RECIPES = frozenset({"core_output_zero"})
RECURRENCE_FAMILIES = frozenset({"conv1d", "gdn_gating"})


def _quantization_group_geometry(op) -> tuple[int, int, int, int, int] | None:
    """Return ``(rows, width, group_size, groups_per_row, total_groups)`` for dynamic quantization."""
    recipe = str(getattr(op, "recipe", "") or "")
    metadata = dict(getattr(op, "metadata", ()) or ())
    if (recipe in PRECOMPUTED_SCALE_QUANT_RECIPES
            or bool(metadata.get("precomputed_scale", False))
            or str(metadata.get("activation_scheme", "") or "").strip().lower() == "static"):
        return None
    shape = tuple(getattr(op, "local_shape", ()) or getattr(op, "semantic_shape", ()) or ())
    if not shape:
        return None
    dims = tuple(int(dim) for dim in shape)
    if any(dim <= 0 for dim in dims):
        return None
    width = dims[-1]
    rows = 1
    for dim in dims[:-1]:
        rows *= dim
    raw_group = metadata.get("activation_group_size")
    if raw_group is None:
        raw_group = metadata.get("group_size")
    if raw_group is None:
        group_size = width
    else:
        group_size = int(raw_group)
        if group_size <= 0:
            return rows, width, rows * width, 1, 1
    if rows > 1 and group_size >= rows * width:
        return rows, width, rows * width, 1, 1
    eff_group = min(group_size, width)
    groups_per_row = max(1, (width + eff_group - 1) // eff_group)
    return rows, width, eff_group, groups_per_row, rows * groups_per_row


def _quantization_scale_write_bytes(op, total_elems: int, total_mte3: float) -> float:
    """Infer scale-tensor write bytes inside ``mte3`` when declared."""
    if total_mte3 <= 0.0:
        return 0.0
    metadata = dict(getattr(op, "metadata", ()) or ())
    scale_fmt = str(metadata.get("scale_format") or metadata.get("activation_scale_format") or "").strip().lower()
    scale_shape = metadata.get("scale_shape")
    if scale_fmt and scale_fmt != "none" and isinstance(scale_shape, (tuple, list)) and scale_shape:
        try:
            s_bytes = float(prod(int(dim) for dim in scale_shape) * dtype_bytes(scale_fmt))
        except (ModelingError, TypeError, ValueError):
            s_bytes = 0.0
        if 0.0 < s_bytes < total_mte3:
            return s_bytes
    out_fmt = str(metadata.get("output_format") or "").strip().lower()
    if out_fmt:
        try:
            payload = float(total_elems) * float(dtype_bytes(out_fmt))
        except (ModelingError, TypeError, ValueError):
            payload = 0.0
        if 0.0 < payload < total_mte3:
            return total_mte3 - payload
    return 0.0


def _group_aligned_pipeline(context, phases, *, slots: int, cap: int, geom: tuple[int, int, int, int, int]):
    """Run reduction-bearing quantization phases on complete group boundaries."""
    rows, width, group_size, groups_per_row, group_count = geom
    count = min(_tile_count(context, cap=cap), group_count)
    if count <= 1:
        return phased_program(context, phases)
    uniform_groups = (width % group_size == 0) if group_size < width else True
    if group_count % count == 0 and uniform_groups:
        return tiled_phased_program(context, phases, slots=slots, cap=count)
    amounts = _phase_amounts(context, phases)
    eff_slots = max(1, int(slots))
    pipelined = eff_slots > 1
    head_caps = _phase_caps(context, amounts, pipelined=pipelined, tile_index=0, tile_count=count)
    mid_caps = _phase_caps(context, amounts, pipelined=pipelined, tile_index=1, tile_count=count)
    tail_caps = _phase_caps(context, amounts, pipelined=pipelined, tile_index=count - 1, tile_count=count)
    total_elems = rows * width
    store_mte3 = next(
        (float(amount.get("mte3", 0.0)) for phase, amount in zip(phases, amounts) if phase.name == "scale_cast_store"),
        0.0,
    )
    scale_write_bytes = _quantization_scale_write_bytes(context.op, total_elems, store_mte3)
    base, rem = divmod(group_count, count)

    def _prefix_elems(group_idx: int) -> int:
        if group_count == 1:
            return total_elems if group_idx >= 1 else 0
        row_idx, rem_group = divmod(group_idx, groups_per_row)
        return row_idx * width + min(rem_group * group_size, width)

    def tile(index: int):
        tile_caps = head_caps if index == 0 else (tail_caps if index == count - 1 else mid_caps)
        g0 = index * base + min(index, rem)
        g1 = (index + 1) * base + min(index + 1, rem)
        elem_share = float(_prefix_elems(g1) - _prefix_elems(g0)) / float(total_elems)
        group_share = float(g1 - g0) / float(group_count)
        store_io_share = (
            ((store_mte3 - scale_write_bytes) * elem_share + scale_write_bytes * group_share) / store_mte3
            if (0.0 < scale_write_bytes < store_mte3)
            else elem_share
        )
        stages = []
        for phase, amount, caps in zip(phases, amounts, tile_caps):
            work = {}
            for name, value in amount.items():
                if value <= 0.0:
                    continue
                if phase.name == "scale_cast_store" and name != context.resource_names.get("vector", "vector"):
                    work[name] = value * store_io_share
                else:
                    work[name] = value * elem_share
            stages.append(Stage(phase.name, work, after=phase.after, caps=caps, keep_empty=True))
        return stages

    pipeline = PipelineProgram(count, eff_slots, tile, minimum_ms=context.minimum_ms)
    return KernelProgram(
        (Stage("setup", latency=context.setup_ms), Stage("tiles", after=("setup",), program=pipeline)),
        minimum_ms=context.minimum_ms,
    )


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
        elif hbm_read > HBM_RW_ASYMMETRY_RATIO * hbm_write:
            dram_rate = context.service_rate("hbm_rw21_dram") or context.service_rate("mte2_aiv_dram")
        else:
            dram_rate = (
                context.service_rate("hbm_rw11_dram")
                or context.service_rate("hbm_rw21_dram")
                or context.service_rate("mte2_aiv_dram")
            )
        updated_caps = dict(context.service_caps)
        changed = False
        if dram_rate is not None and "hbm" in context.resource_names:
            # Preserve declared physical bytes in Work; enforce sustainable
            # DRAM ceilings via Stage.caps so concurrent streams never inflate
            # shared-bus demand.
            hbm_name = context.resource_names["hbm"]
            ceiling_ms = min(dram_rate / MS_PER_SECOND, context.rates[hbm_name])
            if ceiling_ms > 0.0:
                prev = updated_caps.get(hbm_name)
                updated_caps[hbm_name] = min(prev, ceiling_ms) if prev is not None else ceiling_ms
                changed = True
        if submode in NON_STREAMING_VECTOR_SUBMODES and "vector" in context.resource_names:
            vec_name = context.resource_names["vector"]
            op_factor = (
                BROADCAST_VECTOR_OP_FACTOR
                if submode == "broadcast_elementwise"
                else OUTER_REDUCE_VECTOR_OP_FACTOR
            )
            vec_cap_ms = context.rates[vec_name] / op_factor
            if vec_cap_ms > 0.0:
                prev = updated_caps.get(vec_name)
                updated_caps[vec_name] = min(prev, vec_cap_ms) if prev is not None else vec_cap_ms
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
                mte2_name = context.resource_names[mte2_role]
                mte2_cap_ms = min(eff_mte2 / MS_PER_SECOND, context.rates[mte2_name])
                if mte2_cap_ms > 0.0:
                    prev = updated_caps.get(mte2_name)
                    updated_caps[mte2_name] = min(prev, mte2_cap_ms) if prev is not None else mte2_cap_ms
                    changed = True
        return replace(context, service_caps=updated_caps) if changed else context

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
            if (submode in NON_STREAMING_VECTOR_SUBMODES
                or (context.service_caps and len(phases) >= THREE_STAGE_PIPELINE_SLOTS))
            else self.slots
        )
        if submode == "quantization":
            geom = _quantization_group_geometry(context.op)
            if geom is not None:
                return _group_aligned_pipeline(
                    context, phases, slots=slots, cap=self.tile_cap, geom=geom,
                )
        return tiled_phased_program(
            context, phases, slots=slots, cap=self.tile_cap)


AIVStreamingPipeline = AIVTiledVectorAndGatherRecipe
AIVDynamicQuantRecipe = AIVTiledVectorAndGatherRecipe
AIVAtomicScatterCombineRecipe = AIVTiledVectorAndGatherRecipe
