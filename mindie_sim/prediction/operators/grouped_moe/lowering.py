"""Lower grouped MoE routing, storage boundaries and physical stage placement.

[Role] Bind physical work and route each cohort through the common Engine.
[Invariants] UB grants and releases are causal, zero-work stage boundaries.
[Timeouts & Locks] Finite virtual events. No locks, waits or I/O.
[Side-Effect Policy] Only program execution submits events to the caller's Engine.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING, Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from ...mechanism import Work
from ...ownership import unit_group
from ...program import KernelProgram, Program, RecipeContext, Stage, scope_allocations
from ...runtime.ring_buffer import RingBuffer

if TYPE_CHECKING:
    from .common import MoECohortSplit

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-ascend",
        revision="7a7ed073c481d5e3a2fc2194c5b879efb2d373c0",
        relative_path="vllm_ascend/ops/fused_moe/fused_moe.py",
        source_sha256="0647bf1676132121f738cf0d7c569502b5c67a745558d6107e0a39e9bbd9d6c4",
    ),
    SourceRef(
        repository="vllm-ascend",
        revision="7a7ed073c481d5e3a2fc2194c5b879efb2d373c0",
        relative_path="vllm_ascend/ops/fused_moe/prepare_finalize.py",
        source_sha256="f42c57c2835494e886bf6deef054362ac1be9faaea7b1220a257210c7e996611",
    ),
)

ROUTING_INDEX_BYTES = 4
ROUTING_WEIGHT_BYTES = 4
GROUP_COUNT_BYTES = 8
PER_TOKEN_SCALE_BYTES = 4
MX_SCALE_GROUP_SIZE = 32
MX_SCALE_PAIR_BYTES = 2
MX_SCALE_PAIR_WIDTH = MX_SCALE_GROUP_SIZE * MX_SCALE_PAIR_BYTES
MX_DTYPES = frozenset({"mxfp8", "mxfp4"})
QUANT_OPS_PER_ELEMENT = 4
AIC_ROLES = frozenset({"cube", "mte1", "mte2_aic", "fixpipe"})
AIV_ROLES = frozenset({"mte2_aiv", "mte3", "vector"})
_ROUTING_SEQUENCE_KEYS = frozenset({
    "row_ids", "restore_map", "expert_rows", "wave_expert_rows",
    "cohort_row_ids", "cohort_compact_indices", "cohort_restore_map",
})


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def element_bytes(elements: int, dtype: str) -> int:
    if dtype in {"mxfp4", "fp4", "fp4_e2m1", "int4", "w4a8"}:
        return ceil_div(elements, 2)
    return elements * (1 if dtype in {"mxfp8", "hif8"} else int(dtype_bytes(dtype)))


def mx_scale_bytes(width: int, *, padded: bool = True) -> int:
    # CANN stores E8M0 scales as [rows, ceil(width / 64), 2].
    if padded:
        return ceil_div(width, MX_SCALE_PAIR_WIDTH) * MX_SCALE_PAIR_BYTES
    return ceil_div(width, MX_SCALE_GROUP_SIZE)


def tensor_bytes(
    rows: int, width: int, dtype: str, *, scaled: bool = False,
    mx_scaled: bool = False, mx_padded: bool = True,
) -> int:
    if mx_scaled or dtype in MX_DTYPES:
        scale = mx_scale_bytes(width, padded=mx_padded)
    elif scaled:
        scale = PER_TOKEN_SCALE_BYTES
    else:
        scale = 0
    return rows * (element_bytes(width, dtype) + scale)


@dataclass(frozen=True, slots=True)
class StageDemand:
    name: str
    after: tuple[str, ...] = ()
    work: Work = field(default_factory=Work)
    kind: str = ""
    owner: int = 0
    metadata: tuple[tuple[str, object], ...] = ()
    ub_acquire: tuple[str, int] | None = None
    ub_release: tuple[str, int] | None = None


def _slot(context: RecipeContext, owner: int, spec: tuple[str, int] | None):
    if spec is None:
        return ()
    label, size = spec
    units = context.pool_units.get("ub", context.effective_aiv_cores)
    owners = context.datapath.aiv_count if context.datapath else context.resource_units.get("vector", units)
    mapped = unit_group(owner, owners, units)
    return tuple(
        _ring_slot(context.pool_names["ub"], label, size, index) for index in mapped
    )


def _ring_slot(pool, label, size, owner):
    ring = RingBuffer(owner=owner, pool=pool, allocation=f"{label}:{owner}", slot_bytes=size, depth=1)
    ring.initial_credit(0)
    return ring.to_stages(release_slots_at_end=False)[0].acquire[0]


@dataclass(frozen=True, slots=True)
class GroupedMoeProgram:
    demands: tuple[StageDemand, ...]
    aic_count: int
    aiv_count: int

    def __call__(self, context: RecipeContext) -> Program:
        stages = []
        encoded_rows: dict[int, str] = {}
        for demand in self.demands:
            work = context.bind(demand.work)
            placement = {}
            if demand.kind:
                roles = AIC_ROLES if demand.kind == "aic" else AIV_ROLES
                owners = self.aic_count if demand.kind == "aic" else self.aiv_count
                placement = {
                    name: units for name, units in context.owned(demand.owner, owners, roles).items()
                    if name in work
                }
            metadata = dict(demand.metadata)
            # Routing builders use the tuples. Executable stages share one string per sequence.
            for key, rows in metadata.items():
                if type(rows) is tuple and key in _ROUTING_SEQUENCE_KEYS:
                    identity = id(rows)
                    if identity not in encoded_rows:
                        encoded_rows[identity] = ",".join(map(str, rows))
                    metadata[key] = encoded_rows[identity]
            stages.append(Stage(
                demand.name, work, after=demand.after, placement=placement, coupled=bool(work),
                metadata=metadata,
                acquire=_slot(context, demand.owner, demand.ub_acquire),
                free=_slot(context, demand.owner, demand.ub_release),
            ))
        return (BoundedMoEProgram(tuple(stages)) if any(s.acquire or s.free for s in stages)
                else KernelProgram(tuple(stages)))


@dataclass(frozen=True, slots=True)
class BoundedMoEProgram:
    """Keep future UB grants and releases out of the live numeric frontier."""

    stages: tuple[Stage, ...]

    def bind_stream_owners(self, owners, datapath, **kwargs):
        from mindie_sim.prediction.streams import bind_stage_to_stream
        bound = tuple(
            bind_stage_to_stream(stage, owners, datapath, **kwargs)
            for stage in self.stages
        )
        if all(new is old for new, old in zip(bound, self.stages)):
            return self
        return replace(self, stages=bound)

    def start(self, engine, name, ready, *, owner=""):
        complete = engine.future(name + ":complete")
        successors = defaultdict(list)
        pending = {}
        for stage in self.stages:
            parents = stage.after or (None,)
            pending[stage.name] = len(parents)
            for parent in parents:
                successors[parent].append(stage)
        sinks = sum(stage.name not in successors for stage in self.stages)

        def publish(key, _event):
            nonlocal sinks
            children = successors.pop(key, ())
            if key is not None and not children:
                sinks -= 1
                if not sinks:
                    engine.resolve(complete)
                return
            for stage in children:
                pending[stage.name] -= 1
                if pending[stage.name] == 0:
                    del pending[stage.name]
                    scoped_acq = scope_allocations(name, stage.acquire)
                    scoped_free = scope_allocations(name, stage.free)
                    extra = {}
                    if stage.cohort_widened:
                        extra["cohort_widened"] = stage.cohort_widened
                    if stage.cohort_members:
                        extra["cohort_members"] = stage.cohort_members
                    if stage.program is not None:
                        entry = engine.submit(
                            f"{name}:{stage.name}", stage.work, latency=stage.latency,
                            caps=stage.caps, acquire=scoped_acq, metadata=stage.metadata,
                            placement=stage.placement, coupled=stage.coupled,
                            priority=stage.priority, owner=owner, **extra,
                        )
                        nested_done = stage.program.start(
                            engine, f"{name}:{stage.name}:program", entry, owner=owner,
                        )
                        engine.submit(
                            f"{name}:{stage.name}:done", after=(nested_done,),
                            free=scoped_free, owner=owner,
                            native_hook=True, on_complete=partial(publish, stage.name),
                        )
                        continue
                    event = engine.submit(
                        f"{name}:{stage.name}", stage.work, latency=stage.latency,
                        caps=stage.caps, acquire=scoped_acq, free=scoped_free,
                        metadata=stage.metadata, placement=stage.placement,
                        coupled=stage.coupled, priority=stage.priority, owner=owner,
                        native_hook=True, on_complete=partial(publish, stage.name),
                        **extra,
                    )
                    if (stage.acquire or stage.free) and not stage.work and stage.latency <= 0.0:
                        _settle_storage_boundary(engine, event)
                    elif stage.work and event.sequence not in engine._native_seqs:
                        # A numeric row the kernel cannot mirror belongs to
                        # the reference frontier before it receives service.
                        accel = engine._accel
                        if accel is not None and accel.enabled:
                            accel.disable(engine, "mixed_frontier")

        if ready.done:
            publish(None, ready)
        else:
            engine.submit(name + ":entry", after=(ready,), owner=owner,
                          native_hook=True, on_complete=partial(publish, None))
        return complete


def _settle_storage_boundary(engine, event):
    """Apply a ready zero-work grant in the native completion publication.

    The Engine's ordinary admission path retains contended grants, including
    when an earlier Python-ready task is queued. Uncontended boundaries use
    the same ledger and tracing as normal starts/completions. Settling just
    this event keeps allocation rows out of the next C11 numeric epoch.
    """
    seq = event.sequence
    task = engine._tasks[seq]
    if engine._frontier_next_release() <= engine.now or any(
        (priority, key) < (task.priority, seq) and key in engine._tasks
        for priority, key in engine._ready_heap
    ):
        return
    if engine._can_allocate(task):
        engine._start_ready(seq)
        engine._complete(task)


def validate_transport_work(stages, expected):
    """Check the directed transport ledger at the boundary that declared it."""
    for direction, amount in zip(("dispatch", "combine"), expected):
        if amount is not None and amount != sum(
            stage.work.resources.get("link", 0) for stage in stages
            if dict(stage.metadata).get("transport_direction") == direction
        ):
            raise ModelingError("grouped_moe_transport_link_work_not_conserved")


TransportBuilder = Callable[[str, tuple[str, ...], int, tuple[tuple[str, object], ...]], Iterable[StageDemand]]


@dataclass(frozen=True, slots=True)
class DualBatchMoEPipeline:
    split: MoECohortSplit

    def stages(self, contract, ctx, emit, *, transport_builder=None):
        self.split.validate_experts(contract.group_rows)
        stages, combines = [], []
        preceding_dispatch, start = (), 0
        splits = (
            (self.split.cohort0_row_ids, self.split.cohort0_expert_rows),
            (self.split.cohort1_row_ids, self.split.cohort1_expert_rows),
        )
        active_count = sum(bool(ids or any(rows)) for ids, rows in splits)
        transfers = contract.cohort_transport_bytes
        if not transfers and not transport_builder and active_count > 1 and any(contract.transport_bytes):
            raise ModelingError("grouped_moe_dbo_requires_exact_cohort_link_bytes")
        for cohort, (ids, rows) in enumerate(splits):
            if transfers:
                amounts = transfers[cohort]
            elif transport_builder:
                amounts = (None, None)
            else:
                amounts = contract.transport_bytes if ids or any(rows) else (0, 0)
            if not (ids or any(rows) or any(amounts) or transport_builder):
                continue
            prefix = f"c{cohort}_"
            end = start + len(ids)
            metadata = (("cohort_id", cohort), ("slot", cohort), ("row_ids", ids),
                        ("row_start", start), ("row_end", end), ("restore_map", ids),
                        ("expert_rows", rows))
            start = end
            sub = replace(contract, m=sum(rows), group_rows=rows, tokens=len(ids), transport_bytes=amounts)
            admit, release = prefix + "ub_admit", prefix + "ub_release"
            branch = emit(sub, prefix, (admit,), metadata, transport_builder)
            if not (ids or any(rows) or any(any(stage.work.resources.values()) for stage in branch)):
                continue
            capacity = ctx.datapath.usable_bytes("ub")
            slot = (prefix + "routing", min(capacity // 2, ROUTING_INDEX_BYTES * self.split.top_k))
            if not slot[1]:
                raise ModelingError("grouped_moe_dbo_ub_slot_empty")
            stages.extend((
                StageDemand(admit, preceding_dispatch, metadata=metadata, ub_acquire=slot),
                *branch,
                StageDemand(release, (prefix + "combine",), metadata=metadata, ub_release=slot),
            ))
            preceding_dispatch = (prefix + "dispatch",)
            combines.append(release)
        if transport_builder:
            validate_transport_work(stages, contract.transport_bytes)
        # Combine writes to original row positions, so restore is a completion fence.
        stages.append(StageDemand("join_restore", tuple(combines)))
        return stages


def _routing_local(name, after, contract, metadata, *, dispatch, publish_output, read_counts):
    payload_dtype = contract.dtype_a if dispatch else contract.output_dtype
    payload = tensor_bytes(contract.m, contract.k, payload_dtype, scaled=payload_dtype == "int8")
    index_bytes = ROUTING_INDEX_BYTES if dispatch else ROUTING_WEIGHT_BYTES
    count = contract.tokens if dispatch or publish_output else 0
    token_bytes = count * element_bytes(contract.k, contract.output_dtype)
    read, write = (token_bytes, payload) if dispatch else (
        payload + token_bytes * bool(contract.shared_intermediate), token_bytes
    )
    read += count * contract.top_k * index_bytes
    if read_counts and (count or contract.m):
        read += len(contract.group_rows) * GROUP_COUNT_BYTES
    vector = count * contract.k * (
        QUANT_OPS_PER_ELEMENT if dispatch and payload_dtype not in {"bf16", "fp16"} else
        (1 if dispatch else 2 * contract.top_k + bool(contract.shared_intermediate))
    )
    counts = (("source_tokens", count),) if dispatch else (("output_tokens", count),)
    return StageDemand(name + ":local", after, Work(resources={
        "hbm": read + write, "l2": read + write, "mte2_aiv": read, "vector": vector, "mte3": write,
    }), metadata=(*metadata, *counts))


def routing_stages(name, after, contract, metadata, builder: TransportBuilder | None, *,
                   publish_output=True, read_counts=True):
    """Bind directed peer stages; the endpoint owns token pack and publication."""
    dispatch = name.endswith("dispatch")
    local = _routing_local(name, after, contract, metadata, dispatch=dispatch,
                           publish_output=publish_output, read_counts=read_counts)
    expected = contract.transport_bytes[0 if dispatch else 1]
    amount = expected or 0
    remote = name + ":transport"
    deps = (local.name,) if dispatch else after
    tag = (*metadata, ("transport_direction", "dispatch" if dispatch else "combine"))
    link = [replace(stage, metadata=tuple({**dict(stage.metadata), **dict(tag)}.items()))
            for stage in builder(remote, deps, amount, tag)] if builder else (
        [StageDemand(remote, deps, Work(resources={"link": amount}), metadata=tag)] if amount else []
    )
    if expected is not None and sum(stage.work.resources.get("link", 0) for stage in link) != expected:
        raise ModelingError("grouped_moe_transport_link_work_not_conserved")
    if not dispatch and link:
        local = replace(local, after=tuple(stage.name for stage in link))
    stages = [local, *link] if dispatch else [*link, local]
    return [*stages, StageDemand(name, (local.name, *(stage.name for stage in link)), metadata=metadata)]
