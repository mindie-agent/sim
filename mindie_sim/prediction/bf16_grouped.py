"""Explicit CANN nonquant grouped matrix mapping on the common engine.

[Role] Retain the source's fixed multi-bank tiling and private AIC copies.
[Invariants] Integer group rows, not activation capacity, determine work. Each
  bank has its own tiles; cross-bank ownership rolls without reset. GM output
  is a BF16 Fixpipe write. No timing, model name or observation enters planning.
[Timeouts & Locks] O(groups + output tiles); compact K loops use bounded state.
[Side-Effect Policy] Pure plans submitted to the existing Engine only.
"""
from dataclasses import dataclass
from math import ceil

from mindie_sim.schema.common import ModelingError
from .matrix_cycle import matrix_cycle
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .refill import RefillPipeline

SOURCE = {
    "repository": "https://gitcode.com/cann/ops-transformer",
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "host": "gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.cpp",
    "kernel": "gmm/grouped_matmul/op_kernel/arch35/non_quant/grouped_matmul_kernel.h",
    "dependency_repository": "https://gitcode.com/cann/ops-tensor",
    "dependency_revision": "5074e676c509af531df1f83554b6e9394cb45859",
    "dependency_paths": ["include/blaze/gemm/block/block_scheduler_grouped_matmul.h",
                         "include/blaze/gemm/block/block_mmad_matmul_basic.h",
                         "include/blaze/gemm/kernel/kernel_grouped_matmul.h"],
    "file_sha256": {
        'gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.cpp':
            'ff47cab5a7fe02b90ee488efcebcc561d1870fcd73ed3600c3db650b624e3940',
        'gmm/grouped_matmul/op_kernel/arch35/non_quant/grouped_matmul_kernel.h':
            '75cb656bebf4d441edf04c8ab0813a0b8260eddbe949413fde07154f8be6d298',
        'block_scheduler_grouped_matmul.h':
            '23e6cca3c273e4d261f88a6ec7c0764279edccfc3cce346e4cedfde01784c48b',
        'block_mmad_matmul_basic.h':
            'ff3a9a10cad8b1587bc687a22da6fabe8cf188e0282425e59df76811e3c0a1b6',
        'kernel_grouped_matmul.h':
            'e9906b5892817354bada66c29a4ee22803a3c72781b8207d2032ebd34bfdc175',
    },
    "capture_binary_identity": "not_established",
}


@dataclass(frozen=True)
class Mapping:
    m: int
    n: int
    k: int
    l1_k: int

    @property
    def buffers(self):
        return {"l1": 2 * (self.m + self.n) * self.l1_k * 2,
                "l0a": 2 * self.m * self.k * 2,
                "l0b": 2 * self.n * self.k * 2,
                "l0c": self.m * self.n * 4}


def mapping(geometry):
    """CalBaseMMTiling/CalL1Tiling multi-group SPLIT_M branch, no bias."""
    pools = geometry.buffers
    n = 256
    k = pools.get("l0b", 0) // 2 // (n * 2) // 16 * 16
    if not k:
        raise ModelingError("bf16_grouped_requires_source_l0_capacity")
    m = min(256, pools.get("l0a", 0) // 2 // (k * 2),
            pools.get("l0c", 0) // (n * 4))
    a_size = pools.get("l1", 0) - 256 * 1024
    step_a = a_size // 2 // max(1, m) // k // 2
    step_b = (256 * 1024) // 2 // n // k // 2
    if min(m, step_a, step_b) <= 0:
        raise ModelingError("bf16_grouped_requires_source_l1_capacity")
    # The outer wrapper uses the smaller of the two harmonized K steps.
    result = Mapping(m, n, k, k * min(step_a, step_b))
    if any(size > pools.get(pool, 0) for pool, size in result.buffers.items()):
        raise ModelingError("bf16_grouped_source_buffers_exceed_declared_capacity")
    return result


def owner_tiles(rows, n, tile, cores):
    owners = [[] for _ in range(cores)]
    cursor = 0
    for bank, m in enumerate(rows):
        # Source ASWT uses M-window=4 and reverses N on alternate windows.
        nm, nn = ceil(m / tile.m), ceil(n / tile.n)
        if not nm:
            continue
        main = min(4, nm)
        last_window = nm // main - 1
        for window in range(last_window + 1):
            start = window * main
            end = nm if window == last_window else start + main
            for ni0 in range(nn):
                ni = nn - 1 - ni0 if window % 2 else ni0
                for mi in range(start, end):
                    owners[cursor % cores].append((bank, min(tile.m, m - mi * tile.m),
                                                  min(tile.n, n - ni * tile.n)))
                    cursor += 1
    return tuple(tuple(value) for value in owners)


def tile_work(m, n, k):
    reads = 2 * (m + n) * k
    return Work({"bf16": 2 * m * n * k}, {
        "hbm": reads + 2 * m * n, "mte2_aic": reads,
        "mte1": reads, "fixpipe": 2 * m * n})


def build(op, geometry, *, compact=True):
    md = dict(op.metadata)
    rows = tuple(md.get("matrix_group_rows", ()))
    total, n, k = op.local_shape[-3:]
    if (len(rows) < 2 or any(type(v) is not int or v < 0 for v in rows)
            or sum(rows) != total or min(n, k) <= 0):
        raise ModelingError("bf16_grouped_requires_multibank_integer_row_contract")
    if any(md.get(key, op.dtype) != "bf16" for key in
           ("input_format", "weight_format", "compute_format", "output_format")):
        raise ModelingError("bf16_grouped_requires_bf16_operands_output_no_quant")
    if md.get("matrix_epilogue"):
        raise ModelingError("bf16_grouped_no_quant_epilogue_supported")
    cores = geometry.compute_units
    for role in ("mte1", "mte2_aic", "fixpipe"):
        if geometry.resource_units.get(role) != cores:
            raise ModelingError("bf16_grouped_requires_private_aic:" + role)
    tile = mapping(geometry)
    for pool in tile.buffers:
        if geometry.buffer_units.get(pool) != cores:
            raise ModelingError("bf16_grouped_requires_aic_buffer_ownership:" + pool)
    owners = owner_tiles(rows, n, tile, cores)
    # Every launched AIC reads every group-list element, including zero banks.
    scalar = Work(resources={"hbm": len(rows) * 8})
    total_work = scalar.scaled(cores)
    for tiles in owners:
        for _, m, width in tiles:
            total_work += tile_work(m, width, k)

    def program(context):
        def one(owner, tiles):
            def stage(name, work, **kwargs):
                return Stage(name, context.bind(work), placement=context.owned(
                    owner, cores, set(work.resources) - {"hbm"}
                    | ({"cube"} if work.matrix else set())), **kwargs)

            def duration(s):
                return max((amount / (context.rates[key] *
                    (len(s.placement[key]) / cores if key in s.placement else 1))
                            for key, amount in s.work.items()), default=0)

            def output(index):
                _, m, width = tiles[index]
                count, tail = divmod(k, tile.l1_k)

                def parts(kd):
                    load = stage("load", Work(resources={"hbm": 2 * (m + width) * kd,
                                                       "mte2_aic": 2 * (m + width) * kd}))
                    steps = []
                    for kk in range(0, kd, tile.k):
                        step = min(tile.k, kd - kk)
                        operand = stage("operand", Work(resources={"mte1": 2 * (m + width) * step}))
                        cube = stage("matrix", Work({"bf16": 2 * m * width * step}), after=("operand",))
                        steps.append((operand, cube))
                    return load, tuple(steps)

                write = stage("store", Work(resources={"hbm": 2 * m * width,
                                                       "fixpipe": 2 * m * width}))
                if not compact:
                    def panel(index):
                        load, steps = parts(tail if tail and index == count else tile.l1_k)
                        return load, iter((s, i + 1 == len(steps)) for i, s in enumerate(steps)), ()
                    stages = (Stage("K", program=RefillPipeline(count + bool(tail), 2, 2, panel)),
                              Stage("store", write.work, after=("K",), placement=write.placement))
                else:
                    full = parts(tile.l1_k) if count else None
                    end = parts(tail) if tail else None
                    first, last = (full or end)[0], (end or full)[1][-1][1]
                    cycle = matrix_cycle(
                        duration(full[0]) if full else 0,
                        tuple((duration(a), duration(b)) for a, b in (full or end)[1]), count, 2, 2,
                        tail_load_ms=duration(end[0]) if end else None,
                        tail_micro_steps=tuple((duration(a), duration(b)) for a, b in end[1]) if end else None)
                    work, placement = {}, {}
                    for repeat, piece in ((count, full), (int(bool(tail)), end)):
                        if piece is None:
                            continue
                        for s in (piece[0], *(s for pair in piece[1] for s in pair)):
                            for key, value in s.work.items():
                                work[key] = work.get(key, 0) + repeat * value
                            placement.update(s.placement)
                    for s in (first, last):
                        for key, value in s.work.items():
                            work[key] -= value
                    work = {key: max(0, value) for key, value in work.items() if value > 0}
                    span = max(0, cycle.span_ms - duration(first) - duration(last))
                    stages = (
                        first,
                        Stage("steady", work, after=("load",), placement=placement, coupled=True,
                              caps={key: value / span for key, value in work.items()} if span else {}),
                        Stage("matrix", last.work, after=("steady",), placement=last.placement),
                        Stage("store", write.work, after=("matrix",), placement=write.placement))
                return (Stage("tile", program=KernelProgram(stages)),)

            allocation = tuple(context.allocate(f"owner{owner}:{pool}", pool, size, (owner,))
                               for pool, size in tile.buffers.items()) if tiles else ()
            return KernelProgram((
                stage("metadata", scalar, acquire=allocation),
                Stage("outputs", after=("metadata",), program=PipelineProgram(len(tiles), 1, output)),
                Stage("release", after=("outputs",), free=allocation)))
        return ParallelProgram(tuple(one(owner, tiles) for owner, tiles in enumerate(owners)))

    return OperatorPlan(total_work, program, {
        "execution_class": "cube", "source_program": SOURCE,
        "source_mapping": {"base_m": tile.m, "base_n": tile.n,
                           "base_k": tile.k, "l1_k": tile.l1_k,
                           "output_tiles_by_owner": tuple(len(v) for v in owners),
                           "group_rows": rows, "buffers_per_owner": tile.buffers},
        "execution_abstraction": "finite_source_K_pipeline_compact" if compact else "finite_source_K_pipeline_detailed",
        "approximation": "source_output_tiles_serial;compact_external_contention_fluid",
        "unmodeled_source_features": ["scalar_descriptor_and_flag_instruction_latency",
                                      "ND_to_NZ_transaction_efficiency", "physical_small_M_mmad_issue_cycles",
                                      "cross_output_tile_prefetch", "capture_binary_tiling_identity"],
    })


def recipe(*, compact=True):
    return PlannedRecipe(lambda op, geometry: build(op, geometry, compact=compact))
