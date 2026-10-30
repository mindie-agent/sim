"""Tiled pointwise programs with independently declared memory paths.

[Role] Lower row/column loops, masked work and scalar or DMA output service.
[Invariants] A scalar store is not DMA bandwidth. Payload bytes and transfer
  operations are conserved separately. Each unit owns consecutive row tiles.
  Masked arithmetic still executes; masked memory accesses do not.
[Timeouts & Locks] Lazy row/column steps; O(unit count) temporary geometry.
[Side-Effect Policy] Submits only to the caller's virtual engine.
"""
from dataclasses import dataclass
from collections import Counter
import math

from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .transactions import QueuedStore


@dataclass(frozen=True)
class TiledPointwise:
    input_bytes_per_element: float
    outputs: tuple[dict, ...]
    parameter_bytes_per_col: float = 0
    tile_rows: int = 64
    tile_cols: int = 8
    unit_role: str = "vector"

    def lower(self, context):
        rows, cols = context.op.local_shape
        names, rates = context.resource_names, context.rates
        units = context.resource_units.get(self.unit_role, context.core_count)
        row_iterations = math.ceil(rows / (units * self.tile_rows))
        col_iterations = math.ceil(cols / self.tile_cols)
        vector_per_element = context.op.vector_ops / (rows * cols) if rows * cols else 0

        def memory_work(amount, resource, bytes_per_work=1):
            work = {names["hbm"]: amount}
            if resource in names:
                work[names[resource]] = amount / bytes_per_work
            elif resource not in {"mte2", "mte3"}:
                raise ValueError(f"physical_resource_rate_missing:{resource}")
            return work

        def cap(resource, participating):
            if resource not in names:
                return {}
            physical_units = context.resource_units.get(resource, units)
            return {names[resource]: rates[names[resource]] *
                    min(1, participating / physical_units)}

        # Group only units with the same complete row ownership and address
        # phase. Independent groups keep independent instruction streams.
        ownership = Counter()
        for unit in range(units):
            start_row = unit * row_iterations * self.tile_rows
            owned = max(0, min(row_iterations * self.tile_rows, rows - start_row))
            phases = tuple(
                (output.get("offset_bytes", 0) +
                 start_row * output.get("row_stride_bytes", cols * output["bytes_per_element"]))
                % output["queue"]["granule_bytes"] if "queue" in output else 0
                for output in self.outputs)
            ownership[owned, phases] += 1

        def tile(index, owned, phases, members):
            row_index, col_index = divmod(index, col_iterations)
            valid_cols = min(self.tile_cols, cols - col_index * self.tile_cols)
            valid_rows = max(0, min(self.tile_rows, owned - row_index * self.tile_rows))
            elements = members * valid_rows * valid_cols
            reads = (elements * self.input_bytes_per_element +
                     members * valid_cols * self.parameter_bytes_per_col)
            stages = [
                Stage("load", memory_work(reads, "mte2"), caps=cap("mte2", members)),
                Stage("compute", {names["vector"]: members * self.tile_rows * self.tile_cols *
                                  vector_per_element}, after=("load",), caps=cap("vector", members)),
            ]
            for index, output in enumerate(self.outputs):
                amount = elements * output["bytes_per_element"]
                resource = output.get("resource", "mte3")
                if "queue" in output:
                    queue = output["queue"]
                    element_bytes = output["bytes_per_element"]
                    row_stride = output.get("row_stride_bytes", cols * element_bytes)
                    offset = (phases[index] + row_index * self.tile_rows * row_stride +
                              col_index * self.tile_cols * element_bytes) % queue["granule_bytes"]
                    physical_units = context.resource_units.get(resource, units)
                    program = QueuedStore(
                        valid_rows, valid_cols, row_stride, element_bytes, offset, members,
                        names[resource], rates[names[resource]] / physical_units, names["hbm"],
                        **queue,
                    )
                    stages.append(Stage(f"store{index}", after=(stages[-1].name,), program=program))
                    continue
                stages.append(Stage(
                    f"store{index}", memory_work(amount, resource, output.get("bytes_per_work", 1)),
                    after=(stages[-1].name,), caps=cap(resource, members),
                ))
            return stages

        programs = tuple(PipelineProgram(
            row_iterations * col_iterations, 1,
            lambda index, owned=owned, phases=phases, members=members:
            tile(index, owned, phases, members),
        ) for (owned, phases), members in ownership.items())
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("units", after=("setup",), program=ParallelProgram(programs)),
        ), context.minimum_ms)
