"""Lower hyper-connections phases onto the shared resource engine.

[Role] Build the prepared work ledger and submit owner-local phases.
[Invariants] History reads rebind at Engine admission on the physical die's
  shared L2 view. AIV owners issue at most one AttnRes token at a time.
[Timeouts & Locks] Virtual execution with finite token chains. No locks.
[Side-Effect Policy] Planning is pure. Admission only updates Engine state.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from typing import Literal

from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from ...engine import Engine, Event
from ...mechanism import OperatorPlan, Work
from ...ownership import unit_group
from ...program import KernelProgram, RecipeContext, Stage
from ...results import FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.memory_physics import (
    DEFAULT_L2_GRANULE_BYTES,
    DEFAULT_SECTOR_BYTES,
    AddressRun,
    AddressRunPlan,
    AdmissionMemoryView,
    HbmBusSpec,
    MemoryAccessPattern,
    VisibilityContract,
)
from ..contracts import OperatorPlanningContext

FAMILY = "hyper_connections"


def gap(code: str, detail: str) -> UnmodeledCapabilityError:
    return UnmodeledCapabilityError(
        capability=FAMILY, entry=f"{FAMILY}.resolve_operator_plan", code=code, message=detail,
    )


@dataclass(frozen=True, slots=True)
class Phase:
    name: str
    after: tuple[str, ...]
    owner_kind: Literal["aic", "aiv"]
    owners: tuple[int, ...]
    work: Work = field(default_factory=Work)
    access: MemoryAccessPattern | None = None
    final_history_read: bool = False
    metadata: tuple[tuple[str, int | str], ...] = ()

    def reference_work(self, l2_capacity: int) -> Work:
        """Prepare standalone work assuming one active token per used AIV."""
        access = self.access
        if access is None:
            return self.work
        aligned = access.runs.aligned_copy_bytes()
        l2 = aligned if self.final_history_read and access.reuse_distance_bytes <= l2_capacity else 0
        dram = 0 if l2 else access.runs.dram_sector_and_service(())[1]
        return self.work + Work(resources={access.pipe: aligned, "hbm": dram, "l2": l2})


def _copy_rates(ctx: RecipeContext, access: MemoryAccessPattern) -> tuple[str, int, float, float]:
    pipe = access.pipe if access.pipe in ctx.resource_names else "mte2"
    row = ctx.resource_names[pipe]
    units = ctx.resource_units.get(pipe, ctx.resource_units.get(row, 1))
    nominal = ctx.rates[row] / units
    hit, miss = (ctx.service_rate(f"{pipe}_{location}") for location in ("l2", "dram"))
    return pipe, units, (nominal if hit is None else hit / (1000 * units)), (
        nominal if miss is None else miss / (1000 * units)
    )


@dataclass(frozen=True, slots=True)
class TokenProgram:
    """Execute one token per owner, admitting each history access as it starts."""

    stages: tuple[Stage, ...]
    phases: tuple[Phase, ...]
    token_ranges: tuple[tuple[int, int], ...]
    context: RecipeContext
    l2_capacity: int
    l2_units: int

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        rows = self.context.resource_names
        hbm_row = rows["hbm"]
        # The scoped L2 resource denotes one physical die in this Engine.
        die_row = rows.get("l2", hbm_row)
        views = vars(engine).setdefault("_l2_admission_views", {})
        if die_row not in views:
            views[die_row] = AdmissionMemoryView(
                {0: self.l2_capacity}, l2_buffer_units_by_die={0: self.l2_units},
            )
        memory = views[die_row]
        # The standard HBM row takes byte demand and already owns the bus rate.
        bus = HbmBusSpec(0, 1, 1, 0, row_r=hbm_row, row_w=hbm_row, ctr_r=hbm_row, ctr_w=hbm_row)
        complete = engine.future(f"{name}:complete")
        invocation = f"{name}:{complete.sequence}"

        def issue(chain: Iterator[tuple[int, int]], predecessor: Event) -> None:
            nonlocal remaining
            bounds = next(chain, None)
            if bounds is None:
                remaining -= 1
                if not remaining:
                    engine.resolve(complete)
                return
            begin, end = bounds
            events: dict[str, Event] = {}
            first = self.stages[begin]
            if first.after:
                events[first.after[0]] = predecessor
            for index in range(begin, end):
                stage, phase = self.stages[index], self.phases[index]
                deps = tuple(events[dep] for dep in stage.after) or (predecessor,)
                task = f"{name}:{stage.name}"
                access = phase.access
                if access is None or access.reuse_key is None:
                    event = stage.submit(engine, task, deps, lambda items: items, owner)
                else:
                    pipe, _, hit, miss = _copy_rates(self.context, access)
                    copy_owner, = stage.placement[rows[pipe]]
                    key = replace(access.reuse_key, reuse_stream=invocation)
                    access = replace(access, owner=copy_owner, reuse_key=key)
                    bound = self.context.bind_access(
                        access, memory, hbm_bus=bus, copy_hit_bytes_per_ms=hit,
                        copy_miss_bytes_per_ms=miss,
                    )
                    event = engine.submit_bound_access(
                        task, bound, after=deps, owner=owner, memory_state=memory, pattern=access,
                    )
                    if phase.final_history_read:
                        engine.on_complete(event, lambda _, key=key: memory.retire_workspace(key))
                events[stage.name] = event
            engine.on_complete(event, lambda done: issue(chain, done))

        by_owner: dict[int, list[tuple[int, int]]] = {}
        for bounds in self.token_ranges:
            phase = self.phases[bounds[0]]
            by_owner.setdefault(phase.owners[0], []).append(bounds)
        remaining = len(by_owner)
        for ranges in by_owner.values():
            issue(iter(ranges), ready)
        return complete


@dataclass(frozen=True, slots=True)
class PhaseProgram:
    """Picklable preparation and lowering for one invocation's phase graph."""

    phases: tuple[Phase, ...]
    aic_count: int
    aiv_count: int
    l2_capacity: int
    l2_units: int
    token_ranges: tuple[tuple[int, int], ...]

    @property
    def reference_l2_capacity(self) -> int:
        granule = DEFAULT_L2_GRANULE_BYTES
        owners = len({self.phases[begin].owners[0] for begin, _ in self.token_ranges}) or 1
        return granule * (self.l2_capacity // (granule * owners))

    def __call__(self, ctx: RecipeContext) -> KernelProgram | TokenProgram:
        stages = []
        l2_capacity = self.reference_l2_capacity
        for phase in self.phases:
            demand = ctx.bind(phase.work)
            roles = list(phase.work.resources) + (["cube"] if phase.work.matrix else [])
            metadata = dict(phase.metadata)
            access = phase.access
            if access is not None:
                pipe, units, hit, miss = _copy_rates(ctx, access)
                row = ctx.resource_names[pipe]
                work = phase.reference_work(l2_capacity).resources
                l2 = work["l2"]
                aligned = work[access.pipe]
                demand[row] = (l2 / hit + (aligned - l2) / miss) * ctx.rates[row] / units
                for role in ("hbm", "l2"):
                    if work[role] and role in ctx.resource_names:
                        demand[ctx.resource_names[role]] = work[role]
                roles.append(pipe)
                metadata.update(logical_bytes=access.logical_bytes, l2_hit_bytes=l2)
            domain = self.aic_count if phase.owner_kind == "aic" else self.aiv_count
            placement = {}
            for role in roles:
                if role in {"hbm", "l2"}:
                    continue
                resource = ctx.resource_names[ctx.physical_role(role)]
                if resource in demand:
                    units = ctx.resource_units.get(ctx.physical_role(role), domain)
                    placement[resource] = tuple(sorted({
                        unit for owner in phase.owners for unit in unit_group(owner, domain, units)
                    }))
            stages.append(Stage(phase.name, demand, after=phase.after, placement=placement,
                                coupled=access is not None, metadata=metadata))
        if any(phase.final_history_read for phase in self.phases):
            return TokenProgram(tuple(stages), self.phases, self.token_ranges, ctx, self.l2_capacity, self.l2_units)
        return KernelProgram(tuple(stages))


class PhaseGraph:
    """Assemble an invocation DAG with explicit AIC and AIV ownership."""

    def __init__(self, ctx: OperatorPlanningContext) -> None:
        self.ctx = ctx
        self.phases: list[Phase] = []
        self.token_ranges: list[tuple[int, int]] = []
        self.aic = self._owners("aic")
        self.aiv = self._owners("aiv")
        spec = ctx.datapath.memory_access
        self.sector = spec.sector_bytes if spec is not None else DEFAULT_SECTOR_BYTES

    def _owners(self, kind: Literal["aic", "aiv"]) -> tuple[int, ...]:
        available = getattr(self.ctx, f"effective_{kind}_cores")
        owners = self.ctx.stream_owners
        if owners is not None and owners.isolation == "disjoint":
            return tuple(getattr(owners, kind))[:available]
        return tuple(range(available))

    def end_token(self) -> None:
        begin = self.token_ranges[-1][1] if self.token_ranges else 0
        self.token_ranges.append((begin, len(self.phases)))

    def add(self, name, after=(), *, work=None, owners=None, aic=False,
            metadata=(), access=None, final_history_read=False) -> str:
        domain = self.aic if aic else self.aiv
        owners = domain if owners is None else owners
        if not owners:
            kind = "aic" if aic else "aiv"
            raise gap(f"hyper_connections_requires_{kind}_owners", name)
        self.phases.append(Phase(name, after, "aic" if aic else "aiv", owners,
                                 work or Work(), access, final_history_read, metadata))
        return name

    def transfer(self, name, size, after=(), *, aic=False, write=False, owners=None,
                 run=None, reuse=None, distance=None, final_history_read=False) -> str:
        if owners is None:
            owners = self.aic if aic else self.aiv
        pipe = ("fixpipe" if aic else "mte3") if write else ("mte2_aic" if aic else "mte2_aiv")
        runs = AddressRunPlan((run or AddressRun(0, 1, size, size),), sector_bytes=self.sector)
        access = MemoryAccessPattern(
            tensor_id=reuse.backing_allocation if reuse else name, version=0,
            die=0, owner=owners[0] if owners else 0, direction="write" if write else "read",
            pipe=pipe, logical_bytes=size, runs=runs, reuse_key=reuse,
            reuse_distance_bytes=distance, visibility=VisibilityContract(live_out=write),
        )
        return self.add(name, after, owners=owners, aic=aic, access=access, final_history_read=final_history_read)

    def vector(self, name, amount, after, *, owners=None, metadata=()) -> str:
        return self.add(name, after, work=Work(resources={"vector": amount}),
                        owners=owners, metadata=metadata)

    def finish(self, implementation, dtype, sources, metadata) -> OperatorPlan:
        buffers = dict(self.ctx.datapath.buffers)
        l2 = buffers.get("l2", self.ctx.geometry.buffers.get("l2", 0))
        units = self.ctx.geometry.buffer_units.get("l2", 1)
        if l2 and units != 1:
            raise ModelingError("shared_l2_buffer_requires_one_die_owner")
        program = PhaseProgram(tuple(self.phases), self.ctx.datapath.aic_count,
                               self.ctx.datapath.aiv_count, l2, units, tuple(self.token_ranges))
        total = Work()
        phases = []
        reference_l2 = program.reference_l2_capacity
        for phase in program.phases:
            work = phase.reference_work(reference_l2)
            total += work
            items = [WorkItem("cube", "flop", flops, precision) for precision, flops in work.matrix.items()]
            items.extend(WorkItem(role, "op" if role == "vector" else "bytes", value, dtype)
                         for role, value in work.resources.items() if value)
            phases.append(PipelinePhase(phase.name, phase.after, tuple(items)))
        spec = FusedPipelineSpec(FAMILY, implementation, tuple(phases), sources)
        return OperatorPlan(total, program, metadata, pipeline_spec=spec)
