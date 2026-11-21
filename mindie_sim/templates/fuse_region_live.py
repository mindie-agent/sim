"""Replay template memory accesses against a live admission view.

[Role] Bind template transfers, track reuse lifetimes and record deliveries.
[Invariants] An L2 fill becomes readable at its transfer completion.
[Timeouts & Locks] Engine events sequence dependencies without waits or locks.
[Side-Effect Policy] Execution updates only the supplied engine and admission view.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from fractions import Fraction

from mindie_sim.prediction.program.stages import KernelProgram
from mindie_sim.prediction.runtime.l2_residency import _dram_service_segments
from mindie_sim.prediction.runtime.memory_physics import (
    AdmissionMemoryView, BoundAccess, HbmBusSpec, MemoryAccessPattern, ReuseKey,
    SectorCoverage, bind_memory_access,
)
from mindie_sim.schema.common import ModelingError

from .contracts import FusedRegionSummary, _logical_obligation, _obligation_runs, _split_entry_residency


class _ReuseLifetimes:
    """Pin each allocation to its producer key through its final access."""

    def __init__(self, state, patterns):
        self.state = state
        self.pending = Counter(p.reuse_key for p in patterns if p.reuse_key is not None)
        self.keys: dict[ReuseKey, tuple[ReuseKey, bool]] = {}

    def resolve(self, pattern, *, retain=True):
        key = pattern.reuse_key
        if key is None:
            return pattern
        if key not in self.keys:
            resolved = self.state.resolve_reuse_key(key)
            if not retain:
                return replace(pattern, reuse_key=resolved)
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
            elif not all(present):
                raise ModelingError("template_directional_hbm_rows_incomplete")
        if self.live_state is None:
            return super().start(engine, name, ready, owner=owner)
        delivered = Counter()
        lifetimes = _ReuseLifetimes(self.live_state, (
            stage.program.pattern for stage in self.stages if isinstance(stage.program, _LiveAccess)
        ))
        stages = tuple(replace(stage, program=replace(stage.program, delivered=delivered, lifetimes=lifetimes))
                       if isinstance(stage.program, _LiveAccess) else stage for stage in self.stages)
        state, key = self.live_state, name + ":l2_reservation"
        reservation = self.l2_reservation
        if reservation is not None:
            die, reserved, capacity = reservation

            def grant(_event):
                if state.reserved_bytes(die) + reserved > capacity:
                    raise ModelingError("fuse_region_live_l2_reservation_overflow")
                state.grant_reservation(die, key, reserved)

            ready = engine.submit(key, after=(ready,), owner=owner, on_start=grant)
        invocation = KernelProgram(stages)
        done = invocation.start(engine, name, ready, owner=owner)
        # Scoped observers are unique to this invocation and cannot reuse a compiled plan.
        engine._compiled_programs.pop(id(invocation), None)

        def finalize(_event):
            lifetimes.close()
            if reservation is not None:
                state.release_reservation(die, key)
            self.summary._record_delivery(delivered)

        engine.on_complete(done, finalize)
        return done


@dataclass(frozen=True, slots=True)
class _LiveAccess:
    bound: BoundAccess
    state: AdmissionMemoryView
    pattern: MemoryAccessPattern
    delivered: Counter | None = None
    lifetimes: _ReuseLifetimes | None = None

    def start(self, engine, name, ready, *, owner=""):
        done = engine.future(name + ":delivered")

        def admit(event):
            bus, hit, miss, l2, copy_row, l2_row, rules, nominal = self.bound.bind_config
            pattern = self.lifetimes.resolve(self.pattern, retain=False)
            plan = pattern.runs
            aligned = plan.aligned_copy_bytes()
            resident = self.state.resident_coverage_for(pattern, aligned)
            waits = self.state.inflight_fill_dependencies(engine, pattern, resident)
            if waits:
                barrier = engine.boundary(name + ":await_l2_fill", after=(event, *waits))
                engine.on_complete(barrier, admit)
                return
            pattern = self.lifetimes.resolve(self.pattern)
            bound = bind_memory_access(
                pattern, self.state, hbm_bus=bus, copy_hit_bytes_per_ms=hit, copy_miss_bytes_per_ms=miss,
                l2_bytes_per_ms=l2, pipe_row_id=copy_row, l2_row_id=l2_row, burst_rules=rules,
                nominal_pipe_bytes_per_ms=nominal, commit=False,
            )
            ledger = bound.ledger
            if not ledger.l2_hit_bytes:
                resident = SectorCoverage()
            fill = self.state.start_l2_fill(engine, name, pattern, aligned, resident) if ledger.dram_sector_bytes else None
            access = engine.submit_bound_access(name, bound, after=(event,), owner=owner)

            def record(dirty):
                if self.delivered is not None:
                    self.delivered.update({
                        "logical_transfer_bytes": ledger.logical_tensor_bytes,
                        "issued_copy_bytes": ledger.aligned_transfer_bytes,
                        "dram_sector_bytes": ledger.dram_sector_bytes + dirty,
                        "turnaround_service_bytes": dirty + (
                            _rational_service(plan, resident, rules)
                            if ledger.dram_sector_bytes else 0
                        ),
                        "realized_l2_hit_bytes": ledger.l2_hit_bytes,
                        "realized_l2_miss_bytes": ledger.dram_miss_copy_bytes,
                        "realized_hbm_bytes": ledger.dram_bus_transaction_bytes + dirty,
                    })
                self.lifetimes.complete(self.pattern.reuse_key)
                engine.resolve(done)

            def complete(transfer):
                self._settle_residency(pattern, ledger, resident)
                dirty = sum(item.bytes for item in self.state.drain_writebacks(bus.die))
                if fill is not None:
                    self.state.finish_l2_fill(engine, pattern, fill)
                if dirty:
                    work = {term.row_id: term.ms_per_progress for term in bus.terms_for("write", dirty)
                            if term.ms_per_progress}
                    writeback = engine.submit(name + ":dirty_writeback", work, after=(transfer,), owner=owner,
                                              coupled=len(work) > 1, deliveries=((bus.ctr_w, dirty),))
                    engine.on_complete(writeback, lambda _: record(dirty))
                else:
                    record(0)

            engine.on_complete(access, complete)

        engine.on_complete(ready, admit)
        return done

    def _settle_residency(self, pattern, ledger, resident):
        aligned = ledger.aligned_transfer_bytes
        if pattern.direction == "read" and ledger.l2_hit_bytes:
            # Only bytes fetched from HBM may repopulate sectors evicted while
            # this access was in flight.
            if not ledger.dram_miss_copy_bytes:
                if self.state.preview_access(pattern, aligned)[1]:
                    return
            else:
                entry = _logical_obligation(pattern)
                cold = tuple(run for item in _split_entry_residency(entry, resident)
                             if item.initial_residency == "hbm_cold" for run in _obligation_runs(item))
                pattern = replace(pattern, runs=replace(pattern.runs, runs=cold))
        self.state.admit_access(pattern, aligned)


def _rational_service(plan, resident: SectorCoverage, rules):
    service = Fraction()
    for sectors, width, stride in _dram_service_segments(plan, resident):
        service += sectors
        if width:
            penalty = sum((Fraction(str(rule.penalty_bytes)) for rule in rules
                           if rule.matches(width, stride)), Fraction())
            service += sectors * penalty / width
    return service
