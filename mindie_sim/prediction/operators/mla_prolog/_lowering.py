"""Finite programs and transfer accounting shared by MLA prolog strategies.

[Role] Bind physical phase demands, placement and workspace lifetimes to Engine stages.
[Invariants] The pipeline descriptor and executable stages sum the same phase work.
[Timeouts & Locks] Bounded by modeled tile shapes and owners. No locks.
[Side-Effect Policy] Constructs value records and programs without I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef

from ...mechanism import Work
from ...ownership import read_role
from ...program import KernelProgram, ParallelProgram, Program, RecipeContext, Stage
from ...program.pipeline import PipelinePhase as ChunkPhase
from ...program.pipeline import PipelineTemplate, PipelineTemplateProgram
from ...results import FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.memory_physics import AddressRun, AddressRunPlan, MemoryAccessPattern
from ...runtime.ring_buffer import RingBuffer, lower_generation
from ...streams import StreamOwners


TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCE_REF = SourceRef(
    repository="cann/ops-transformer",
    revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
    relative_path="attention/mla_prolog/op_host/mla_prolog_tiling.cpp",
    source_sha256="1770937bd4a58222c5606ce03bec3c574e23bd563ea341c03a1710f153962909",
)
FRACTAL_MN = 16
FRACTAL_K_BF16 = 16
FRACTAL_K_BYTE = 32
BF16_BYTES = 2
FP32_BYTES = 4
MX_GROUP_SIZE = 32


def ceil_div(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _align(value: int, multiple: int) -> int:
    return ceil_div(value, multiple) * multiple


def fractal_issues(m: int, n: int, k: int, k0: int) -> tuple[int, int]:
    """Return the issue count and corresponding issued FLOPs."""
    if min(m, n, k) < 0 or k0 not in {FRACTAL_K_BF16, FRACTAL_K_BYTE}:
        raise ModelingError("mla_prolog_invalid_fractal_geometry")
    issues = ceil_div(m, FRACTAL_MN) * ceil_div(n, FRACTAL_MN) * ceil_div(k, k0)
    return issues, 2 * issues * FRACTAL_MN * FRACTAL_MN * k0


def _plane(rows: int, width: int, *, stride: int | None = None, offset: int = 0) -> AddressRun:
    return AddressRun(offset, rows, width, width if stride is None else stride)


def _token_chunks(spans, start: int, rows: int, tile: int):
    """Keep the host's logical step batch while forming physical X runs."""
    logical = 0
    runs = []
    for q, _, physical, _, _ in spans:
        if q:
            if runs and runs[-1][2] + logical - runs[-1][0] == physical:
                begin, _, physical = runs.pop()
            else:
                begin = logical
            runs.append((begin, logical + q, physical))
            logical += q
    stop = start + rows
    cursor = 0
    while start < stop:
        size = min(stop - start, tile)
        while runs[cursor][1] <= start:
            cursor += 1
        begin, end, physical = runs[cursor]
        if end >= start + size:
            repeats = max(1, (min(end, stop) - start) // tile)
            addresses = ((physical + start - begin, size),)
        else:
            repeats = 1
            addresses = []
            for index in range(cursor, len(runs)):
                lo, hi, base = runs[index]
                if lo >= start + size:
                    break
                first = max(start, lo)
                addresses.append((base + first - lo, min(start + size, hi) - first))
            addresses = tuple(addresses)
        yield size, repeats, addresses
        start += size * repeats


def _parts(size: int, count: int) -> tuple[int, ...]:
    base, large = divmod(size, count)
    return tuple(base + (owner < large) for owner in range(count))


def _dual_rows(rows: int) -> tuple[int, int]:
    first = min(rows, _align(rows, FRACTAL_MN) // 2)
    return first, rows - first


def _activation(width: int, dtype: str) -> int:
    if dtype == "bf16":
        return width * BF16_BYTES
    if dtype == "int8":
        return width + FP32_BYTES
    return width + ceil_div(width, MX_GROUP_SIZE)


def _weight(n: int, k: int, dtype: str, k0: int) -> tuple[AddressRun, ...]:
    size = BF16_BYTES if dtype == "bf16" else 1
    payload = _plane(_align(n, FRACTAL_MN), _align(k, k0) * size)
    if dtype == "bf16":
        return (payload,)
    scales = FP32_BYTES if dtype == "int8" else ceil_div(k, MX_GROUP_SIZE)
    return payload, _plane(n, scales)


@dataclass(frozen=True, slots=True)
class _Demand:
    owner: int
    axis: str
    work: Work
    audit: tuple[tuple[str, int | str], ...] = ()


def _transfer_operands(
    owner: int, axis: str, pipe: str, tensor: str,
    allocations: tuple[tuple[str, tuple[AddressRun, ...]], ...],
    *, write: bool = False, gm: bool = True, l2: bool = False, alignment: int | None = None,
    logical_bytes: int | None = None,
) -> _Demand:
    """Account for distinct backing allocations without aliasing their offset-0 spans."""
    aligned = 0
    sectors = 0
    inferred_logical = 0
    direction = "write" if write else "read"
    for backing, runs in allocations:
        if not runs:
            continue
        addresses = AddressRunPlan(runs) if alignment is None else AddressRunPlan(runs, alignment_bytes=alignment)
        run_logical = sum(run.rows * run.row_bytes for run in runs)
        pattern = MemoryAccessPattern(
            f"{tensor}:{backing}", 0, 0, owner, direction, pipe, run_logical, addresses,
        )
        inferred_logical += pattern.logical_bytes
        aligned += addresses.aligned_copy_bytes()
        if gm:
            sectors += addresses.dram_sector_and_service()[0]
    total_logical = inferred_logical if logical_bytes is None else logical_bytes
    work = {pipe: aligned}
    if gm:
        work["hbm"] = sectors
        if l2:
            work["l2"] = sectors
    return _Demand(owner, axis, Work(resources=work), (
        ("tensor", tensor), ("logical_bytes", total_logical),
        ("aligned_bytes", aligned), ("sector_bytes", sectors),
    ))


def _transfer(
    owner: int, axis: str, pipe: str, tensor: str, *runs: AddressRun,
    write: bool = False, gm: bool = True, l2: bool = False, alignment: int | None = None,
    logical_bytes: int | None = None,
) -> _Demand:
    return _transfer_operands(
        owner, axis, pipe, tensor, (("data", runs),),
        write=write, gm=gm, l2=l2, alignment=alignment, logical_bytes=logical_bytes,
    )


@dataclass(frozen=True, slots=True)
class _Phase:
    name: str
    after: tuple[str, ...]
    demands: tuple[_Demand, ...]


@dataclass(frozen=True, slots=True)
class _ResidentTiling:
    token_tile: int
    token_tiles: int
    k_splits: int
    column_tile: int
    panel: int
    a_groups: int
    b_groups: int
    gm_alignment: int


@dataclass(frozen=True, slots=True)
class _GenericTiling:
    token_tile: int
    column_tile: int
    k_splits: int
    reduce_tile: int
    separate_projections: bool
    qb_token_tile: int | None = None


def _generic_projection(name, rows, width, hidden, owners, tiling, start_owner, *, l2, alignment):
    """Aggregate the round-robin (token tile, column tile, K split) job queue by owner."""
    m, n, k = tiling.token_tile, tiling.column_tile, hidden // tiling.k_splits
    padded = _align(rows, m)
    jobs_per_tile = width // n * tiling.k_splits
    jobs = padded // m * jobs_per_tile
    dtype = "fp8_e4m3"
    # Model the uniform NZ panels and their MX scale rows once per projection job.
    activation = (_plane(m, k, stride=hidden), _plane(m, k // MX_GROUP_SIZE, stride=hidden // MX_GROUP_SIZE))
    weight = _weight(n, k, dtype, FRACTAL_K_BYTE)
    partial = _plane(m, n * FP32_BYTES, stride=width * FP32_BYTES)
    issues, issued = fractal_issues(m, n, k, FRACTAL_K_BYTE)
    reads, operands, cubes, writes = [], [], [], []
    for index, owner in enumerate(owners):
        first = (index - start_owner) % len(owners)
        count = ceil_div(jobs - first, len(owners))
        if not count:
            continue
        full = max(0, ceil_div(jobs - jobs_per_tile - first, len(owners)))
        live = count * m - (count - full) * (padded - rows)
        load = _transfer_operands(
            owner, "aic", "mte2_aic", name, (
                ("act", (activation[0],)),
                ("act_scale", (activation[1],)),
                ("weight", (weight[0],)),
                ("weight_scale", weight[1:]),
            ), l2=l2,
        )
        pipe_bytes = load.work.resources["mte2_aic"]
        reads.append(_repeat_demand(load, count))
        operands.append(_Demand(owner, "aic", Work(resources={"mte1": pipe_bytes * count})))
        cubes.append(_Demand(owner, "aic", Work(matrix={dtype: issued * count}), (
            ("logical_flops", 2 * live * n * k), ("fractal_issues", issues * count),
        )))
        store = _transfer(owner, "aic", "fixpipe", name, partial, write=True, l2=l2, alignment=alignment)
        stores = _repeat_demand(store, count)
        writes.append(replace(stores, audit=(
            ("tensor", name), ("logical_bytes", live * n * FP32_BYTES),
            ("aligned_bytes", stores.work.resources["fixpipe"]), ("sector_bytes", stores.work.resources["hbm"]),
        )))
    return reads, operands, cubes, writes


def _repeat_demand(demand: _Demand, count: int) -> _Demand:
    counters = {"logical_bytes", "aligned_bytes", "sector_bytes", "logical_flops", "fractal_issues", "issued_flops"}
    audit = tuple((key, value * count if key in counters else value) for key, value in demand.audit)
    return replace(demand, work=demand.work.scaled(count), audit=audit)


@dataclass(frozen=True, slots=True)
class _Chunk:
    count: int
    phases: tuple[_Phase, ...]
    workspace_bytes: int
    qb_ub_slots: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class _PhysicalOwnerProgram:
    """Submit the disjoint lane's already-bound stages with their physical cores."""

    pipeline: ParallelProgram

    def start(self, engine, name, ready, *, owner=""):
        # The generic stream binder maps lane-relative placements and allocations.
        # This pipeline already selected and validated physical pairs during planning.
        return self.pipeline.start(engine, name, ready, owner=owner)


@dataclass(frozen=True, slots=True)
class _PrologProgramBuilder:
    groups: tuple[tuple[_Chunk, ...], ...]
    aic_count: int
    aiv_count: int
    stream_owners: StreamOwners | None = None

    def __call__(self, context: RecipeContext) -> Program:
        programs = tuple(
            PipelineTemplateProgram(PipelineTemplate(
                owner, 1, tuple(ChunkPhase("step_batch", self.stages(chunk, context), chunk.count)
                                for chunk in chunks),
            ))
            for owner, chunks in enumerate(self.groups)
        )
        if not programs:
            return KernelProgram(())
        pipeline = ParallelProgram(programs)
        if self.stream_owners is not None and self.stream_owners.isolation == "disjoint":
            return _PhysicalOwnerProgram(pipeline)
        return pipeline

    def stages(self, chunk: _Chunk, context: RecipeContext) -> tuple[Stage, ...]:
        stages = []
        roots: tuple[str, ...] = ()
        if chunk.workspace_bytes:
            pool = context.pool_names.get("hbm", "hbm")
            ring = RingBuffer(pool=pool, allocation="mla_workspace", depth=1,
                              slot_bytes=chunk.workspace_bytes if "hbm" in context.pool_names else 0)
            events = lower_generation(ring, 0, "reserve", ("complete",))
            # The finite template repeats this slot. Its next reservation waits for the previous last use.
            credit, reserve, publish, consumer, *drain = ring.to_stages()
            stages.extend((credit, replace(reserve, previous=(events.last_use,)), publish))
            roots = (events.publish,)

        aic_binding, aiv_binding = {}, {}
        stream = self.stream_owners
        if stream is not None and stream.isolation == "shared_cap" and context.stream_owners is not None:
            # The shared-cap binder also attaches quota resources and consumes pool indices.
            aic_binding = {owner: lane for lane, owner in enumerate(stream.pool_aic)}
            aiv_binding = {owner: lane for lane, owner in enumerate(stream.pool_aiv)}
        qb_buffers = tuple(context.allocate(f"wqb_acc:{owner}", "ub", size, (aiv_binding.get(owner, owner),))
                           for owner, size in chunk.qb_ub_slots if "ub" in context.pool_names)
        for phase in chunk.phases:
            deps = phase.after or roots
            if qb_buffers and phase.name == "mm3_wqb_fixpipe_ub":
                stages.append(Stage("reserve_wqb_ub", after=deps, acquire=qb_buffers))
                deps = ("reserve_wqb_ub",)
            owners = []
            for index, demand in enumerate(phase.demands):
                name = f"{phase.name}:{demand.owner}:{index}"
                work = demand.work
                resources = dict(work.resources)
                read = "mte2_aic" if demand.axis == "aic" else "mte2_aiv"
                if read in resources and read not in context.resource_names:
                    resources[read_role(context.resource_names, None if demand.axis == "aic" else 0)] = resources.pop(read)
                owned_roles = tuple(role for role in resources if role not in {"hbm", "l2"})
                if work.matrix:
                    owned_roles += ("cube",)
                binding = aic_binding if demand.axis == "aic" else aiv_binding
                stages.append(Stage(
                    name, context.bind(Work(work.matrix, resources)), after=deps,
                    placement=context.owned(binding.get(demand.owner, demand.owner),
                                            self.aic_count if demand.axis == "aic" else self.aiv_count, owned_roles),
                    metadata=dict(demand.audit),
                ))
                owners.append(name)
            stages.append(Stage(phase.name, after=tuple(owners) or deps))
        if qb_buffers:
            stages.append(Stage("release_wqb_ub", after=("query_out_vector",), free=qb_buffers))

        if chunk.workspace_bytes:
            upstream = {name for phase in chunk.phases for name in phase.after}
            sinks = tuple(phase.name for phase in chunk.phases if phase.name not in upstream)
            stages.extend((replace(consumer, after=sinks), *drain))
        return tuple(stages)


def _summarize(program: _PrologProgramBuilder, family: str, implementation: str,
               sources: tuple[SourceRef, ...]) -> tuple[Work, FusedPipelineSpec]:
    phases: dict[str, tuple[tuple[str, ...], Work]] = {}
    for chunks in program.groups:
        for chunk in chunks:
            for phase in chunk.phases:
                deps, work = phases.get(phase.name, (phase.after, Work()))
                for demand in phase.demands:
                    work += demand.work.scaled(chunk.count)
                phases[phase.name] = deps, work

    total = Work()
    descriptor = []
    for name, (after, work) in phases.items():
        total += work
        items = tuple(WorkItem("cube", "flop", amount, dtype) for dtype, amount in sorted(work.matrix.items()))
        items += tuple(WorkItem(role, "op" if role == "vector" else "bytes", amount)
                       for role, amount in sorted(work.resources.items()) if amount)
        descriptor.append(PipelinePhase(name, after, items))
    return total, FusedPipelineSpec(family, implementation, tuple(descriptor), sources)
