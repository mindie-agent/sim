"""Finite programs and transfer accounting shared by MLA prolog strategies.

[Role] Bind physical phase demands, placement and workspace lifetimes to Engine stages.
[Invariants] The pipeline descriptor and executable stages sum the same phase work.
[Timeouts & Locks] Bounded by modeled tile shapes and owners. No locks.
[Side-Effect Policy] Constructs value records and programs without I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

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


@dataclass(frozen=True, slots=True)
class _Demand:
    owner: int
    axis: str
    work: Work
    audit: tuple[tuple[str, int | str], ...] = ()


def _transfer(
    owner: int, axis: str, pipe: str, tensor: str, *runs: AddressRun,
    write: bool = False, gm: bool = True, l2: bool = False, alignment: int | None = None,
    logical_bytes: int | None = None,
) -> _Demand:
    addresses = AddressRunPlan(runs) if alignment is None else AddressRunPlan(runs, alignment_bytes=alignment)
    pattern = MemoryAccessPattern(
        tensor, 0, 0, owner, "write" if write else "read", pipe,
        sum(run.rows * run.row_bytes for run in runs) if logical_bytes is None else logical_bytes,
        addresses,
    )
    aligned = addresses.aligned_copy_bytes()
    sectors = addresses.dram_sector_and_service()[0] if gm else 0
    work = {pipe: aligned}
    if gm:
        work["hbm"] = sectors
        if l2:
            work["l2"] = sectors
    return _Demand(owner, axis, Work(resources=work), (
        ("tensor", pattern.tensor_id), ("logical_bytes", pattern.logical_bytes),
        ("aligned_bytes", aligned), ("sector_bytes", sectors),
    ))


@dataclass(frozen=True, slots=True)
class _Phase:
    name: str
    after: tuple[str, ...]
    demands: tuple[_Demand, ...]


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
