"""Partitioned accumulator publication and vector output on existing resources.

[Role] Compose a declared matrix-to-vector handoff, scale reads and final stores.
[Invariants] Padded publication bytes are distinct from useful output bytes.
  Empty vector partitions consume no service. Each active partition owns its
  buffer and copy/vector pipes; a sibling cannot lend its idle throughput.
[Timeouts & Locks] O(partitions * output_slots) live state, no I/O or locks.
[Side-Effect Policy] Builds programs for the caller's ordinary causal Engine.
"""
from dataclasses import dataclass
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .mechanism import Work
from .ownership import read_role
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage


def align(value, granule):
    return math.ceil(value / granule) * granule


@dataclass(frozen=True)
class PartitionedMatrixOutput:
    """A declared physical path, independent of any matrix dispatch selector."""

    transfer_role: str
    partitions: int
    publication_row_alignment: int
    column_alignment: int
    accumulator_width: int = 4
    output_chunks: int = 4
    output_slots: int = 2
    buffer_alignment: int = 32
    pool: str = "ub"

    def allocations(self, context, core, buffer_shape, epilogue):
        """Keep declared per-owner buffers resident across the whole invocation."""
        if min(self.partitions, self.publication_row_alignment, self.column_alignment,
               self.accumulator_width, self.output_chunks, self.output_slots,
               self.buffer_alignment) < 1:
            raise ModelingError("matrix_output_geometry_must_be_positive")
        if self.pool not in context.pool_names:
            raise ModelingError(f"matrix_output_pool_required:{self.pool}")
        total_rows, n, _ = context.op.local_shape[-3:]
        groups = int(dict(context.op.metadata).get("matrix_groups", 1))
        owners = context.core_count * self.partitions
        if context.resource_units.get("vector", context.core_count) != owners:
            raise ModelingError("matrix_output_vector_partition_topology_mismatch")
        if context.pool_units.get(self.pool, context.core_count) != owners:
            raise ModelingError("matrix_output_buffer_partition_topology_mismatch")
        role = read_role(context.resource_names, aiv=0)
        for required in (role, "mte3", context.physical_role(self.transfer_role)):
            if required not in context.resource_names:
                raise ModelingError(f"matrix_output_resource_required:{required}")
        width = dtype_bytes(dict(context.op.metadata).get("output_format", "bf16"))
        resident_rows = math.ceil(buffer_shape[0] / self.partitions)
        resident_columns = align(buffer_shape[1], self.column_alignment)
        row_scale_width = epilogue.row_scale_bytes / total_rows
        column_scale_width = epilogue.column_scale_bytes / (groups * n)
        resident = (resident_rows * resident_columns * self.accumulator_width
                    + buffer_shape[1] * column_scale_width
                    + align(resident_rows * row_scale_width, self.buffer_alignment)
                    + self.output_slots * math.ceil(resident_rows / self.output_chunks)
                    * buffer_shape[1] * width)
        return tuple(context.allocate(
            f"partition{partition}", self.pool, resident,
            (core * self.partitions + partition,))
            for partition in range(self.partitions))

    def lower(self, context, *, core, rows, columns, buffer_shape, epilogue,
              reuse_inputs, reuse_weights):
        total_rows, n, _ = context.op.local_shape[-3:]
        groups = int(dict(context.op.metadata).get("matrix_groups", 1))
        owners = context.core_count * self.partitions
        role = read_role(context.resource_names, aiv=0)
        width = dtype_bytes(dict(context.op.metadata).get("output_format", "bf16"))
        per_partition = math.ceil(rows / self.partitions)
        row_scale_width = epilogue.row_scale_bytes / total_rows
        column_scale_width = epilogue.column_scale_bytes / (groups * n)
        programs = []
        for partition in range(self.partitions):
            owner = core * self.partitions + partition
            part_rows = max(0, min(per_partition, rows - partition * per_partition))
            if not part_rows:
                continue
            placement = context.owned(owner, owners, ("vector", role, "mte3"))
            row_copy = part_rows * row_scale_width
            column_copy = columns * column_scale_width
            fraction = part_rows * columns / (total_rows * n)
            scale_hbm = (
                epilogue.row_scale_bytes * fraction if reuse_inputs else row_copy
            ) + (
                epilogue.column_scale_bytes * fraction if reuse_weights else column_copy
            )
            chunk_count = min(self.output_chunks, part_rows)
            chunk_rows = math.ceil(part_rows / chunk_count)
            chunk_count = math.ceil(part_rows / chunk_rows)

            def chunk(index, *, part_rows=part_rows, chunk_rows=chunk_rows,
                      placement=placement):
                height = min(chunk_rows, part_rows - index * chunk_rows)
                output = height * columns * width
                return (
                    Stage("vector", context.bind(Work(resources={
                        "vector": epilogue.operations_per_output * height * columns})),
                          placement=placement),
                    Stage("store", context.bind(Work(resources={
                        "hbm": output, "mte3": output})), after=("vector",),
                          placement=placement),
                )

            programs.append(KernelProgram((
                Stage("scales", context.bind(Work(resources={
                    role: row_copy + column_copy, "hbm": scale_hbm})),
                      placement=placement),
                Stage("output", after=("scales",),
                      program=PipelineProgram(chunk_count, self.output_slots, chunk)),
            )))
        published = (align(rows, self.publication_row_alignment)
                     * columns * self.accumulator_width)
        return KernelProgram((
            Stage("publish", context.bind(Work(resources={self.transfer_role: published})),
                  placement=context.owned(core, context.core_count, (self.transfer_role,))),
            Stage("partitions", after=("publish",), program=ParallelProgram(tuple(programs))),
        ))
