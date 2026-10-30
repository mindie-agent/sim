"""Finite matrix tiles, operand traffic and on-chip residency.

[Role] Implement an explicit M/N wave and K-panel matrix program.
[Invariants] Tails conserve arithmetic. Ideal shared-cache reuse changes HBM
  traffic, never per-tile private-buffer copies or FLOPs. A panel buffer remains
  live until consumed; accumulators span K panels. Tiles own units and buffers.
[Timeouts & Locks] Lazy O(compute_units * slots) tile construction, no I/O.
[Side-Effect Policy] Submits through the shared program engine only.
"""
from dataclasses import dataclass
import math

from mindie_sim.schema.common import dtype_bytes
from .engine import Allocation
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .refill import RefillPipeline
from .matrix_epilogue import MatrixEpilogue
from .ownership import read_role


def operand_transfer(context, left, right, *, directional=False):
    """A/B instructions share one MTE1 pipe; a source may select its modes."""
    if directional and all(role in context.service_modes
                           for role in ("mte1_l0a", "mte1_l0b")):
        from .mechanism import Work
        return context.bind(Work(resources={"mte1_l0a": left, "mte1_l0b": right}))
    name = context.resource_names.get("mte1")
    return {name: left + right} if name is not None else {}


@dataclass(frozen=True)
class TiledMatmul:
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 128
    slots: int = 2
    reuse_inputs: bool = True
    reuse_weights: bool = True
    operand_pool: str = "l1"
    accumulator_pool: str = "l0c"
    distributed_transfers: bool = True

    def lower(self, context):
        if min(self.tile_m, self.tile_n, self.tile_k, self.slots) < 1:
            raise ValueError("matrix_tile_dimensions_and_slots_must_be_positive")
        op, names = context.op, context.resource_names
        total_m, n, k = op.local_shape[-3:]
        metadata = dict(op.metadata)
        groups = int(metadata.get("matrix_groups", 1))
        if groups < 1 or total_m % groups:
            raise ValueError("matrix_groups_require_equal_integer_rows")
        m = total_m // groups
        nm, nn, nk = (math.ceil(m / self.tile_m), math.ceil(n / self.tile_n),
                      math.ceil(k / self.tile_k))
        tiles = groups * nm * nn
        cores = context.core_count
        epilogue = MatrixEpilogue.from_op(op)
        input_bytes, weight = epilogue.operands(op)
        operand_read = read_role(names)
        scale_read = read_role(names, aiv=0)
        act_width = dtype_bytes(metadata.get("activation_format", op.dtype or "bf16"))
        output_width = dtype_bytes(metadata.get("output_format", "bf16"))
        operand_pool = context.pool_names.get(self.operand_pool)
        accumulator_pool = context.pool_names.get(self.accumulator_pool)

        def panel(core, index):
            wave, kk = divmod(index, nk)
            tile = wave * cores + core
            within = tile % (nm * nn)
            row, column = divmod(within, nn)
            a = min(self.tile_m, m - row * self.tile_m)
            b = min(self.tile_n, n - column * self.tile_n)
            kd = min(self.tile_k, k - kk * self.tile_k)
            fraction = a * b / (total_m * n)
            k_fraction = kd / k
            # The flat tile index is group-major. A tile reads its own group's
            # B only; A is private per N tile. Ideal shared-cache reuse
            # suppresses only HBM reads, not cache-to-owner copy service.
            a_copy = input_bytes * a / total_m * k_fraction
            b_copy = weight / groups * b / n * k_fraction
            a_read = input_bytes * fraction * k_fraction if self.reuse_inputs else a_copy
            b_read = weight * fraction * k_fraction if self.reuse_weights else b_copy
            reads = a_read + b_read
            writes = op.mte3_bytes * fraction if kk == nk - 1 else 0
            load = {names["hbm"]: reads}
            store = {names["hbm"]: writes}
            for role, volume, target in ((operand_read, a_copy + b_copy, load),
                                         ("mte3", writes, store)):
                if role in names:
                    target[names[role]] = volume
            compute = {key: value * fraction * k_fraction
                       for key, value in context.work.items()
                       if key not in {names["hbm"], names.get("mte2"), names.get("mte3"),
                                      names.get(operand_read), names.get(scale_read)}}
            if epilogue.operations_per_output:
                compute.pop(names["vector"], None)
            # Explicit equal groups have physical A/B panel domains. A one-group
            # declaration retains the prior buffer approximation, including
            # aggregate expert moments whose individual shapes are unspecified.
            resident = a_copy + b_copy if groups > 1 else (a + b) * kd * act_width
            # ``wave`` is a per-owner ordinal; accumulators span K panels.
            operand = (() if operand_pool is None else (
                Allocation(f"panel{index}", operand_pool,
                           resident, (core,)),))
            accumulator = (() if accumulator_pool is None else (
                Allocation(f"accumulator{wave}", accumulator_pool,
                           a * b * max(4, output_width), (core,)),))
            stages = [
                Stage("load", load, acquire=operand,
                      placement=context.owned(core, cores, (operand_read,))
                      if self.distributed_transfers else {}),
                Stage("compute", compute, after=("load",),
                      placement=context.owned(core, cores, ("cube", "vector")),
                      acquire=accumulator if kk == 0 else (), free=operand),
            ]
            predecessor = "compute"
            if kk == nk - 1 and epilogue.operations_per_output:
                scale_copy, scale_hbm = epilogue.scales(
                    a, b, total_m, n, groups, reuse_inputs=self.reuse_inputs,
                    reuse_weights=self.reuse_weights)
                read = scale_read
                scale_work = {names["hbm"]: scale_hbm}
                if read in names:
                    scale_work[names[read]] = scale_work.get(names[read], 0) + scale_copy
                stages.extend((
                    Stage("scales", scale_work, after=("compute",),
                          placement=context.owned(core, cores, (read,))),
                    Stage("epilogue", {names["vector"]: epilogue.operations_per_output * a * b},
                          after=("scales",),
                          placement=context.owned(core, cores, ("vector",))),
                ))
                predecessor = "epilogue"
            stages.append(Stage("store", store, after=(predecessor,),
                      placement=context.owned(core, cores, ("mte3",))
                      if self.distributed_transfers else {},
                      free=accumulator if kk == nk - 1 else ()))
            return tuple(stages)

        programs = tuple(PipelineProgram(
            math.ceil((tiles - core) / cores) * nk, self.slots,
            lambda index, core=core: panel(core, index))
            for core in range(min(cores, tiles)))
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("cores", after=("setup",), program=ParallelProgram(programs)),
        ), minimum_ms=context.minimum_ms)


@dataclass(frozen=True)
class HierarchicalMatmul:
    """A private refill tile containing smaller matrix-consumption tiles.

    The L1 panel remains resident until every inner tile consumes it. Each
    inner tile still transfers operands to its private matrix operand buffers.
    No cache hit removes service at a different memory boundary.
    """

    tile_m: int = 256
    tile_n: int = 256
    tile_k: int = 128
    compute_m: int = 128
    compute_n: int = 128
    slots: int = 2
    compute_slots: int = 2
    reuse_inputs: bool = True
    reuse_weights: bool = True

    def lower(self, context, *, owner_tiles=None, output=None, serial_tiles=False,
              directional_mte1=False):
        if min(self.tile_m, self.tile_n, self.tile_k, self.compute_m,
               self.compute_n, self.slots, self.compute_slots) < 1:
            raise ValueError("matrix_tile_dimensions_and_slots_must_be_positive")
        if self.tile_m % self.compute_m or self.tile_n % self.compute_n:
            raise ValueError("matrix_refill_tiles_must_be_multiples_of_compute_tiles")
        op, names = context.op, context.resource_names
        total_m, n, k = op.local_shape[-3:]
        metadata = dict(op.metadata)
        groups = int(metadata.get("matrix_groups", 1))
        if groups < 1 or total_m % groups:
            raise ValueError("matrix_groups_require_equal_integer_rows")
        m = total_m // groups
        if not total_m:
            return KernelProgram((Stage("setup", latency=context.setup_ms),),
                                 minimum_ms=context.minimum_ms)
        nm, nn, nk = (math.ceil(m / self.tile_m), math.ceil(n / self.tile_n),
                      math.ceil(k / self.tile_k))
        cores = context.core_count
        epilogue = MatrixEpilogue.from_op(op)
        inputs, weight = epilogue.operands(op)
        operand_read = read_role(names)
        scale_read = read_role(names, aiv=0)
        compute_width = dtype_bytes(metadata.get("compute_format", op.dtype or "bf16"))
        output_width = dtype_bytes(metadata.get("output_format", "bf16"))
        work = {name: value for name, value in context.work.items()
                if name not in {names["hbm"], names.get("mte1"),
                                names.get("mte2"), names.get("mte3"),
                                names.get(operand_read), names.get(scale_read)}}
        if epilogue.operations_per_output:
            work.pop(names["vector"], None)
        if owner_tiles is None:
            counts = tuple(math.ceil((groups * nm * nn - core) / cores)
                           for core in range(min(cores, groups * nm * nn)))
            tile_sources = None
        else:
            # A source adapter supplies finite owner-local rectangles. K-panel
            # execution and all resource/buffer semantics remain shared here.
            counts = tuple(count for count, _ in owner_tiles)
            tile_sources = [tuple(tiles) for _, tiles in owner_tiles]

        def allocation(name, pool, size, core):
            return (() if pool not in context.pool_names else
                    (Allocation(name, context.pool_names[pool], size, (core,)),))

        def transfer(role, volume, *, hbm=None):
            result = {names[role]: volume} if role in names else {}
            if hbm is not None:
                result[names["hbm"]] = hbm
            return result

        def panel(core, index):
            wave, kk = divmod(index, nk)
            if tile_sources is None:
                tile = wave * cores + core
                a = min(self.tile_m, m - ((tile // nn) % nm) * self.tile_m)
                b = min(self.tile_n, n - (tile % nn) * self.tile_n)
            else:
                a, b = tile_sources[core][wave]
            kd = min(self.tile_k, k - kk * self.tile_k)
            fraction, k_fraction = a * b / (total_m * n), kd / k
            a_copy = inputs * a / total_m * k_fraction
            b_copy = weight * b / (n * groups) * k_fraction
            reads = ((inputs * fraction * k_fraction if self.reuse_inputs else a_copy)
                     + (weight * fraction * k_fraction if self.reuse_weights else b_copy))
            writes = op.mte3_bytes * fraction if kk == nk - 1 else 0
            operand = allocation(f"panel{index}", "l1", a_copy + b_copy, core)
            accumulator = allocation(f"accumulator{wave}", "l0c",
                                     a * b * max(4, output_width), core)
            inner_n = math.ceil(b / self.compute_n)

            def steps():
                count = math.ceil(a / self.compute_m) * inner_n
                for part in range(count):
                    ai = min(self.compute_m, a - part // inner_n * self.compute_m)
                    bi = min(self.compute_n, b - part % inner_n * self.compute_n)
                    left, right = a_copy * ai / a, b_copy * bi / b
                    # Keep packed payload/scales and the widened representation
                    # bound. Conversion latency is not inferred from these bytes.
                    left = max(left, ai * kd * compute_width)
                    right = max(right, bi * kd * compute_width)
                    local = (
                        *allocation(f"a{index}:{part}", "l0a", left, core),
                        *allocation(f"b{index}:{part}", "l0b", right, core),
                    )
                    demand = {name: value * ai * bi / (total_m * n) * k_fraction
                              for name, value in work.items()}
                    stages = [
                        Stage("operand", operand_transfer(context, left, right,
                              directional=directional_mte1), acquire=local,
                              placement=context.owned(core, cores, ("mte1",))),
                        Stage("matrix", demand, after=("operand",), free=local,
                              acquire=accumulator if kk == 0 and part == 0 else (),
                              placement=context.owned(core, cores, ("cube", "vector"))),
                    ]
                    if part == count - 1 and kk == nk - 1:
                        if output is not None:
                            stages.append(Stage(
                                "output", after=("matrix",), free=accumulator,
                                program=output.lower(
                                    context, core=core, rows=a, columns=b,
                                    buffer_shape=(self.tile_m, self.tile_n), epilogue=epilogue,
                                    reuse_inputs=self.reuse_inputs,
                                    reuse_weights=self.reuse_weights)))
                        elif epilogue.operations_per_output:
                            scale_copy, scale_hbm = epilogue.scales(
                                a, b, total_m, n, groups,
                                reuse_inputs=self.reuse_inputs, reuse_weights=self.reuse_weights)
                            stages.extend((
                                Stage("scales", transfer(scale_read, scale_copy, hbm=scale_hbm),
                                      after=("matrix",),
                                      placement=context.owned(core, cores, (scale_read,))),
                                Stage("epilogue", {names["vector"]:
                                      epilogue.operations_per_output * a * b},
                                      after=("scales",),
                                      placement=context.owned(core, cores, ("vector",))),
                            ))
                        if output is None:
                            stages.append(Stage("store", transfer("mte3", writes, hbm=writes),
                                                after=("epilogue",) if epilogue.operations_per_output
                                                else ("matrix",), free=accumulator,
                                                placement=context.owned(core, cores, ("mte3",))))
                    yield tuple(stages), part == count - 1

            return (Stage("load", transfer(operand_read, a_copy + b_copy, hbm=reads),
                          acquire=operand, placement=context.owned(core, cores, (operand_read,))),
                    steps(), operand)

        programs = []
        for core, count in enumerate(counts):
            if not count:
                continue
            program = RefillPipeline(
                count * nk, self.slots, self.compute_slots,
                lambda index, core=core: panel(core, index), nk if serial_tiles else None)
            if output is not None:
                buffers = output.allocations(
                    context, core, (self.tile_m, self.tile_n), epilogue)
                program = KernelProgram((
                    Stage("reserve_output", acquire=buffers),
                    Stage("body", after=("reserve_output",), program=program),
                    Stage("release_output", after=("body",), free=buffers),
                ))
            programs.append(program)
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("cores", after=("setup",), program=ParallelProgram(tuple(programs))),
        ), minimum_ms=context.minimum_ms)
