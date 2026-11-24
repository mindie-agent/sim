"""Replay template memory accesses against a live admission view.

[Role] Bind template transfers, track reuse lifetimes and record deliveries.
[Invariants] An L2 fill becomes readable at its transfer completion.
[Timeouts & Locks] Engine events sequence dependencies without waits or locks.
[Side-Effect Policy] Execution updates only the supplied engine and admission view.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction

from mindie_sim.prediction.program.stages import KernelProgram, Stage, scope_allocations
from mindie_sim.prediction.runtime.l2_residency import _dram_service_segments
from mindie_sim.prediction.runtime.memory_physics import (
    AdmissionMemoryView, BoundAccess, HbmBusSpec, MemoryAccessPattern, ReuseKey,
    SectorCoverage,
)
from mindie_sim.schema.common import ModelingError

from .contracts import (
    FuseRegionSpec, FusedRegionSummary, TensorRangeObligation, _entry_with_runs, _logical_obligation,
    _obligation_runs, _partition_runs, _split_entry_residency, _union_obligations,
)


class _ReuseLifetimes:
    """Pin each allocation to its producer key through its final access."""

    def __init__(self, state, patterns):
        self.state = state
        self.pending = Counter(p.reuse_key for p in patterns if p.reuse_key is not None)
        self.keys: dict[ReuseKey, tuple[ReuseKey, bool]] = {}

    def resolve(self, pattern):
        key = pattern.reuse_key
        if key is None:
            return pattern
        if key not in self.keys:
            resolved = self.state.resolve_reuse_key(key)
            self.keys[key] = resolved, self.state.retain_template_key(resolved)
        return replace(pattern, reuse_key=self.keys[key][0])

    def complete(self, key):
        if key is not None:
            self.pending[key] -= 1
            if not self.pending[key]:
                resolved, owned = self.keys.pop(key)
                if owned:
                    self.state.release_template_key(resolved)

    def close(self):
        for resolved, owned in self.keys.values():
            if owned:
                self.state.release_template_key(resolved)
        self.keys.clear()


@dataclass(frozen=True)
class _BoundTemplateProgram(KernelProgram):
    """Declare directional admission rows before submitting the bound DAG."""

    hbm_bus: HbmBusSpec | None = None
    live_state: AdmissionMemoryView | None = None
    l2_reservation: tuple[int, int, int] | None = None
    summary: FusedRegionSummary | None = None

    def start(self, engine, name, ready, *, owner=""):
        bus = self.hbm_bus
        if bus is not None:
            present = (bus.row_r in engine.resources, bus.row_w in engine.resources)
            if not any(present):
                engine.add_hbm_bus(bus)
            elif all(present):
                engine.bind_hbm_bus(bus)
            else:
                raise ModelingError("template_directional_hbm_rows_incomplete")
        if self.live_state is None:
            return super().start(engine, name, ready, owner=owner)
        delivered = Counter()
        lifetimes = _ReuseLifetimes(self.live_state, (
            stage.program.pattern for stage in self.stages if isinstance(stage.program, _LiveAccess)
        ))
        obligations = {"read": [], "write": []}
        explicit = self.summary.spec if isinstance(self.summary.spec, FuseRegionSpec) else None
        stages = []
        for stage in self.stages:
            if isinstance(stage.program, _LiveAccess):
                stage = replace(stage, program=replace(stage.program, delivered=delivered, lifetimes=lifetimes,
                                                       obligations=obligations, explicit=explicit is not None))
            elif (engine._accel is not None and (stage.acquire or stage.free)
                    and not (stage.program or stage.work or stage.latency or stage.fragment or stage.deliveries)):
                scoped = replace(stage, acquire=scope_allocations(name, stage.acquire),
                                 free=scope_allocations(name, stage.free))
                stage = Stage(stage.name, after=stage.after, program=_LiveAllocation(f"{name}:{stage.name}", scoped))
            stages.append(stage)
        state, key = self.live_state, name + ":l2_reservation"
        reservation = self.l2_reservation
        if reservation is not None:
            die, reserved, capacity = reservation

            def grant(_event):
                if state.reserved_bytes(die) + reserved > capacity:
                    raise ModelingError("fuse_region_live_l2_reservation_overflow")
                state.grant_reservation(die, key, reserved)

            ready = engine.submit(key, after=(ready,), owner=owner, on_complete=grant, native_hook=True)
        invocation = KernelProgram(tuple(stages))
        done = invocation.start(engine, name, ready, owner=owner)
        # Scoped observers are unique to this invocation and cannot reuse a compiled plan.
        engine._compiled_programs.pop(id(invocation), None)

        def finalize(_event):
            lifetimes.close()
            if reservation is not None:
                state.release_reservation(die, key)
            reads, writes = obligations["read"], obligations["write"]
            if explicit is not None:
                reads = _observed_obligations(explicit.live_in_obligations, reads)
                writes = _observed_obligations(explicit.live_out_obligations, writes)
            self.summary._record_delivery(delivered, _union_obligations(reads), _union_obligations(writes))

        return engine.submit(name + ":delivery", after=(done,), owner=owner,
                             on_complete=finalize, native_hook=True)


@dataclass(frozen=True, slots=True)
class _LiveAllocation:
    name: str
    stage: Stage

    def start(self, engine, name, ready, *, owner=""):
        return engine._submit_allocation_boundary(self.name, self.stage, (ready,), owner=owner)


@dataclass(frozen=True, slots=True)
class _LiveAccess:
    bound: BoundAccess
    state: AdmissionMemoryView
    pattern: MemoryAccessPattern
    obligation: TensorRangeObligation | None
    delivered: Counter | None = None
    lifetimes: _ReuseLifetimes | None = None
    obligations: dict | None = None
    explicit: bool = False

    def start(self, engine, name, ready, *, owner=""):
        del engine, name, ready, owner
        raise NotImplementedError(
            "LEGACY_PURGED: separate realized_hbm_delivered_bytes from dram_bus_service_bytes in fuse_region_live.py"
        )


def _observed_obligations(declared, observed):
    """Use observed residency for the covered parts of explicit region obligations."""
    by_tensor = defaultdict(list)
    for entry in observed:
        by_tensor[entry.die, entry.allocation_id, entry.tensor_version].append(entry)
    for entry in declared:
        matching = by_tensor[entry.die, entry.allocation_id, entry.tensor_version]
        if not matching or entry.initial_residency == "onchip_resident":
            yield entry
            continue
        tagged = [(run, None) for run in _obligation_runs(entry)]
        tagged.extend((run, item.initial_residency) for item in matching for run in _obligation_runs(item))
        split = defaultdict(list)
        for residency, run in _partition_runs(tagged, lambda labels: (
            "hbm_cold" if "hbm_cold" in labels else "l2_warm" if "l2_warm" in labels else entry.initial_residency
        ) if None in labels else None):
            split[residency].append(run)
        for residency, runs in split.items():
            yield _entry_with_runs(entry, runs, initial_residency=residency)


def _rational_service(plan, resident: SectorCoverage, rules):
    service = Fraction()
    for sectors, width, stride in _dram_service_segments(plan, resident):
        service += sectors
        if width:
            penalty = sum((Fraction(str(rule.penalty_bytes)) for rule in rules
                           if rule.matches(width, stride)), Fraction())
            service += sectors * penalty / width
    return service
