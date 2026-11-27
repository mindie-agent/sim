"""Per-owner matrix resource phases with explicit timing approximation.

[Role] Execute finite mapping work without expanding every K/micro tile.
[Invariants] Preserve mapped physical work, ownership and final output work.
  Serial owner phases omit K/micro overlap; they are not a detailed schedule
  or a strict duration bound under competing resource service. Mapping screens
  tile footprints; aggregate phase work never becomes a fictitious live buffer.
[Timeouts & Locks] O(owners) tasks after bounded mapping search; no I/O or locks.
[Side-Effect Policy] Submit phases on the caller's Engine and record the mapping.
"""
from dataclasses import dataclass
import math
from typing import Literal

from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from .matrix import l2_hbm_read_scale, plain_store_role
from .matrix_epilogue import MatrixEpilogue
from .matrix_mapping import MappedMatmul, matrix_geometry, owner_tiles, splitk_executable
from mindie_sim.prediction.ownership import read_role
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_OPS_TRANSFORMER_REV = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "cann/ops-transformer", _OPS_TRANSFORMER_REV,
        "mc2/3rd/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp",
        "472db5c1bb12fd01289005f636dfa7b86cb0d49c9ce8ee77313fe8c430d618a0",
    ),
)


@dataclass(frozen=True)
class PhaseMatmul(MappedMatmul):
    """Use the selected mapping's exact demands in serial per-owner phases."""

    def lower(self, context):
        recipe, record = self.select(context)
        op, names = context.op, context.resource_names
        geometry = matrix_geometry(context, recipe)
        if getattr(recipe, "k_splits", 1) > 1 and splitk_executable(context, recipe):
            from .matrix_splitk import splitk_program
            program = splitk_program(
                context, recipe.k_splits, recipe=recipe, geometry=geometry, record=record,
            )
            if context.record_mapping is not None:
                context.record_mapping(record)
            return program
        record.update({
            "execution_abstraction": "serial_per_owner_resource_phases",
            "overlap": "between_owners_only; within_owner_K_micro_overlap_omitted",
            "buffer_model": "selected_tile_capacity_screening_only; no_phase_residency",
            "geometry": geometry.to_dict(),
        })
        if context.record_mapping is not None:
            context.record_mapping(record)
        total_m, n, k = geometry.local_shape
        if not total_m:
            return KernelProgram((Stage("setup", latency=context.setup_ms),),
                                 minimum_ms=context.minimum_ms)
        meta, cores = dict(op.metadata), context.core_count
        groups, m = geometry.groups, geometry.m_per_group
        epilogue = MatrixEpilogue.from_op(op)
        inputs, weight = epilogue.operands(op)
        operand_read = read_role(names)
        scale_read = read_role(names, aiv=0)
        store_role = plain_store_role(names, epilogue)
        width = dtype_bytes(meta.get("compute_format", op.dtype or "bf16"))
        hbm_read_scale = l2_hbm_read_scale(
            op, inputs=inputs, weight=weight, groups=groups,
            group_rows=geometry.group_rows, nn=geometry.nn, tile_m=recipe.tile_m,
            reuse_inputs=recipe.reuse_inputs, reuse_weights=recipe.reuse_weights,
            row_scale_bytes=epilogue.row_scale_bytes,
            column_scale_bytes=epilogue.column_scale_bytes,
        )
        owners = geometry.owners
        left = owner_tiles(m, math.ceil(n / recipe.compute_n), recipe.tile_m,
                           recipe.tile_n // recipe.compute_n, cores, groups)
        right = owner_tiles(math.ceil(m / recipe.compute_m), n,
                            recipe.tile_m // recipe.compute_m, recipe.tile_n, cores, groups)
        excluded = {
            names["hbm"], names.get("mte1"), names.get("mte2"),
            names.get("mte3"), names.get(operand_read), names.get(scale_read),
        }
        if store_role == "fixpipe":
            excluded.add(names.get("fixpipe"))
        compute = {name: value for name, value in context.work.items()
                   if name not in excluded}
        if epilogue.operations_per_output:
            compute.pop(names["vector"], None)

        def transfer(role, amount, *, hbm=None):
            demand = {names[role]: amount} if role in names else {}
            if hbm is not None:
                demand[names["hbm"]] = hbm
            return demand

        programs = []
        for core, (_, a, b, area) in enumerate(owners):
            fraction = area / (total_m * n)
            a_copy, b_copy = inputs * a / total_m, weight * b / (n * groups)
            reads = (
                (inputs * fraction if recipe.reuse_inputs else a_copy)
                + (weight * fraction if recipe.reuse_weights else b_copy)
            ) * hbm_read_scale
            inner = (left[core][3] * max(inputs / total_m, k * width)
                     + right[core][3] * max(weight / (n * groups), k * width))
            stages = [
                Stage("load", transfer(operand_read, a_copy + b_copy, hbm=reads),
                      placement=context.owned(core, cores, (operand_read,))),
                Stage("operand", transfer("mte1", inner), after=("load",),
                      placement=context.owned(core, cores, ("mte1",))),
                Stage("matrix", {name: value * fraction for name, value in compute.items()},
                      after=("operand",),
                      placement=context.owned(core, cores, ("cube", "vector"))),
            ]
            predecessor = "matrix"
            if epilogue.operations_per_output:
                row = epilogue.row_scale_bytes * a / total_m
                column = epilogue.column_scale_bytes * b / (n * groups)
                scale_hbm = ((epilogue.row_scale_bytes * fraction
                              if recipe.reuse_inputs else row)
                             + (epilogue.column_scale_bytes * fraction
                                if recipe.reuse_weights else column))
                stages.extend((
                    Stage("scales", transfer(scale_read, row + column, hbm=scale_hbm * hbm_read_scale),
                          after=("matrix",),
                          placement=context.owned(core, cores, (scale_read,))),
                    Stage("epilogue", {names["vector"]:
                          epilogue.operations_per_output * area}, after=("scales",),
                          placement=context.owned(core, cores, ("vector",))),
                ))
                predecessor = "epilogue"
            writes = op.mte3_bytes * fraction
            stages.append(Stage("store", transfer(store_role, writes, hbm=writes),
                                after=(predecessor,),
                                placement=context.owned(core, cores, (store_role,))))
            programs.append(KernelProgram(tuple(stages)))
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("owners", after=("setup",), program=ParallelProgram(tuple(programs))),
        ), minimum_ms=context.minimum_ms)
