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

from mindie_sim.schema.common import dtype_bytes
from .matrix_epilogue import MatrixEpilogue
from .matrix_mapping import MappedMatmul, owner_tiles
from .ownership import read_role
from .program import KernelProgram, ParallelProgram, Stage


@dataclass(frozen=True)
class PhaseMatmul(MappedMatmul):
    """Use the selected mapping's exact demands in serial per-owner phases."""

    def lower(self, context):
        recipe, record = self.select(context)
        record.update({
            "execution_abstraction": "serial_per_owner_resource_phases",
            "overlap": "between_owners_only; within_owner_K_micro_overlap_omitted",
            "buffer_model": "selected_tile_capacity_screening_only; no_phase_residency",
        })
        if context.record_mapping is not None:
            context.record_mapping(record)
        op, names = context.op, context.resource_names
        total_m, n, k = op.local_shape[-3:]
        if not total_m:
            return KernelProgram((Stage("setup", latency=context.setup_ms),),
                                 minimum_ms=context.minimum_ms)
        meta, cores = dict(op.metadata), context.core_count
        groups = int(meta.get("matrix_groups", 1))
        m = total_m // groups
        epilogue = MatrixEpilogue.from_op(op)
        inputs, weight = epilogue.operands(op)
        operand_read = read_role(names)
        scale_read = read_role(names, aiv=0)
        width = dtype_bytes(meta.get("compute_format", op.dtype or "bf16"))
        owners = owner_tiles(m, n, recipe.tile_m, recipe.tile_n, cores, groups)
        left = owner_tiles(m, math.ceil(n / recipe.compute_n), recipe.tile_m,
                           recipe.tile_n // recipe.compute_n, cores, groups)
        right = owner_tiles(math.ceil(m / recipe.compute_m), n,
                            recipe.tile_m // recipe.compute_m, recipe.tile_n, cores, groups)
        compute = {name: value for name, value in context.work.items()
                   if name not in {names["hbm"], names.get("mte1"), names.get("mte2"),
                                   names.get("mte3"), names.get(operand_read),
                                   names.get(scale_read)}}
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
            reads = ((inputs * fraction if recipe.reuse_inputs else a_copy)
                     + (weight * fraction if recipe.reuse_weights else b_copy))
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
                    Stage("scales", transfer(scale_read, row + column, hbm=scale_hbm),
                          after=("matrix",),
                          placement=context.owned(core, cores, (scale_read,))),
                    Stage("epilogue", {names["vector"]:
                          epilogue.operations_per_output * area}, after=("scales",),
                          placement=context.owned(core, cores, ("vector",))),
                ))
                predecessor = "epilogue"
            writes = op.mte3_bytes * fraction
            stages.append(Stage("store", transfer("mte3", writes, hbm=writes),
                                after=(predecessor,),
                                placement=context.owned(core, cores, ("mte3",))))
            programs.append(KernelProgram(tuple(stages)))
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("owners", after=("setup",), program=ParallelProgram(tuple(programs))),
        ), minimum_ms=context.minimum_ms)
