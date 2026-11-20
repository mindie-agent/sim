"""Dynamic N-stream L2 working-set residency, dirty writeback, and 3-ledger binding.

[Role] Track per-die shared L2 residency across arbitrary N >= 1 active streams
  and full ``ReuseKey`` identities using canonical sector intervals, settle
  dirty private evictions with future readers as explicit ``DirtyWriteback``
  records, and bind ``MemoryAccessPattern`` into 3-ledger ``BoundAccess``
  without bind-time mutation (``commit=False`` default).
[Invariants] Shared L2 enforces ``l2_buffer_units == 1`` per die. Every
  classified access conserves ``l2_hit_bytes + dram_miss_copy_bytes ==
  aligned_transfer_bytes``. Evicted entries never resurrect when pressure drops.
  Untouched sector ranges never false-hit on resident sibling offsets.
[Timeouts & Locks] Pure in-memory interval arithmetic; no I/O, threads or locks.
[Side-Effect Policy] Read-only ``preview_access`` never mutates state; mutations
  occur strictly on explicit admission/release/reservation transitions.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError
from .memory_physics import (
    BoundAccess,
    BurstTurnRule,
    ByteLedger,
    ConstraintTerm,
    CounterTerm,
    DEFAULT_L2_GRANULE_BYTES,
    HbmBusSpec,
    MemoryAccessPattern,
    ReuseKey,
    _intersect_byte_intervals,
    _measure_byte_intervals,
    _merge_byte_intervals,
)


@dataclass(frozen=True, slots=True)
class DirtyWriteback:
    """Settled DRAM writeback emitted when a live dirty private L2 line evicts."""

    key: ReuseKey
    die: int
    bytes: int
    ranges: tuple[tuple[int, int], ...]


@dataclass(slots=True)
class _ResidentL2Entry:
    key: ReuseKey
    working_set_bytes: int
    version: int
    dirty: bool
    private_workspace: bool
    evicted: bool = False
    resident_ranges: tuple[tuple[int, int], ...] = ()
    dirty_ranges: tuple[tuple[int, int], ...] = ()
    has_future_reader: bool = True


class AdmissionMemoryView:
    """Dynamic per-die shared L2 working-set knee tracker across N >= 1 active streams."""

    def __init__(
        self,
        l2_capacity_by_die: Mapping[int, int],
        *,
        l2_buffer_units_by_die: Mapping[int, int] | None = None,
        granule_bytes: int = DEFAULT_L2_GRANULE_BYTES,
    ) -> None:
        if type(granule_bytes) is not int or granule_bytes < 1:
            raise ModelingError(f"l2_granule_invalid:{granule_bytes!r}")
        self.granule_bytes = granule_bytes
        self._l2_capacity: dict[int, int] = {}
        self._reserved: dict[int, dict[str, int]] = {}
        self._active_streams: dict[int, set[str]] = {}
        self._active_keys: dict[int, set[ReuseKey]] = {}
        self._resident: dict[ReuseKey, _ResidentL2Entry] = {}
        self._pending_writebacks: dict[int, list[DirtyWriteback]] = {}
        units_map = dict(l2_buffer_units_by_die or {})
        for die, cap in l2_capacity_by_die.items():
            if type(die) is not int or die < 0 or type(cap) is not int or cap < 0:
                raise ModelingError(f"l2_capacity_invalid:{die!r}:{cap!r}")
            if units_map.get(die, 1) != 1:
                raise ModelingError("shared_l2_buffer_requires_one_die_owner")
            self._l2_capacity[die] = cap
            self._reserved[die] = {}
            self._active_streams[die] = set()
            self._active_keys[die] = set()
            self._pending_writebacks[die] = []

    def active_stream_count(self, die: int) -> int:
        keys = self._active_keys.get(die, ())
        streams = self._active_streams.get(die, ())
        covered = {k.reuse_stream for k in keys}
        return len(keys) + sum(1 for s in streams if s not in covered)

    def reserved_bytes(self, die: int) -> int:
        return sum(self._reserved.get(die, {}).values())

    def effective_l2_bytes(self, die: int) -> int:
        """Compute ``g * floor(max(0, C_L2 - C_reserved) / (g * max(1, n_active)))``."""
        cap = self._l2_capacity.get(die)
        if cap is None:
            raise ModelingError(f"l2_die_undeclared:{die}")
        avail = max(0, cap - self.reserved_bytes(die))
        n_active = max(1, self.active_stream_count(die))
        granule = self.granule_bytes
        return granule * (avail // (granule * n_active))

    def _settle_eviction(self, entry: _ResidentL2Entry) -> None:
        if entry.dirty and entry.has_future_reader:
            wb_bytes = _measure_byte_intervals(entry.dirty_ranges) or entry.working_set_bytes
            self._pending_writebacks[entry.key.die].append(DirtyWriteback(
                key=entry.key, die=entry.key.die,
                bytes=wb_bytes, ranges=entry.dirty_ranges,
            ))
            entry.dirty = False
            entry.dirty_ranges = ()
        entry.evicted = True

    def _evict_over_capacity(self, die: int) -> None:
        eff = self.effective_l2_bytes(die)
        for entry in self._resident.values():
            if entry.key.die == die and not entry.evicted and entry.working_set_bytes > eff:
                self._settle_eviction(entry)

    def pending_writebacks(self, die: int) -> tuple[DirtyWriteback, ...]:
        return tuple(self._pending_writebacks.get(die, ()))

    def drain_writebacks(self, die: int) -> tuple[DirtyWriteback, ...]:
        items = tuple(self._pending_writebacks.get(die, ()))
        if die in self._pending_writebacks:
            self._pending_writebacks[die].clear()
        return items

    def enter_stream(self, die: int, stream_id: str) -> int:
        """Activate a reuse stream on ``die`` and immediately evict over-knee entries."""
        if die not in self._l2_capacity:
            raise ModelingError(f"l2_die_undeclared:{die}")
        self._active_streams[die].add(stream_id)
        self._evict_over_capacity(die)
        return self.effective_l2_bytes(die)

    def exit_stream(self, die: int, stream_id: str) -> int:
        """Deactivate a reuse stream; never resurrects previously evicted entries."""
        if die not in self._l2_capacity:
            raise ModelingError(f"l2_die_undeclared:{die}")
        self._active_streams[die].discard(stream_id)
        self._active_keys[die] = {
            k for k in self._active_keys[die] if k.reuse_stream != stream_id
        }
        return self.effective_l2_bytes(die)

    def release_key(self, key: ReuseKey) -> int:
        """Release one active ``ReuseKey`` lifetime without resurrecting evicted lines."""
        if key.die not in self._l2_capacity:
            raise ModelingError(f"l2_die_undeclared:{key.die}")
        self._active_keys[key.die].discard(key)
        return self.effective_l2_bytes(key.die)

    def grant_reservation(self, die: int, alloc_id: str, size_bytes: int) -> int:
        """Reserve explicit L2 capacity on ``die`` and immediately evict over-knee entries."""
        if die not in self._l2_capacity:
            raise ModelingError(f"l2_die_undeclared:{die}")
        if type(size_bytes) is not int or size_bytes < 0:
            raise ModelingError(f"l2_reservation_invalid:{alloc_id}:{size_bytes!r}")
        self._reserved[die][alloc_id] = size_bytes
        self._evict_over_capacity(die)
        return self.effective_l2_bytes(die)

    def release_reservation(self, die: int, alloc_id: str) -> int:
        """Release explicit L2 reservation without resurrecting evicted lines."""
        if die not in self._l2_capacity:
            raise ModelingError(f"l2_die_undeclared:{die}")
        self._reserved[die].pop(alloc_id, None)
        return self.effective_l2_bytes(die)

    def retire_workspace(self, key: ReuseKey) -> None:
        """Terminate dead private workspace residency without DRAM writeback."""
        if key.die in self._active_keys:
            self._active_keys[key.die].discard(key)
        entry = self._resident.pop(key, None)
        if entry is not None:
            entry.has_future_reader = False
            entry.dirty = False

    def is_resident(self, key: ReuseKey) -> bool:
        entry = self._resident.get(key)
        return entry is not None and not entry.evicted

    @staticmethod
    def _range_hit_miss(
        req_ivs: tuple[tuple[int, int], ...],
        res_ivs: tuple[tuple[int, int], ...],
        aligned_bytes: int,
    ) -> tuple[int, int]:
        if not req_ivs or not res_ivs:
            return 0, aligned_bytes
        req_b = _measure_byte_intervals(req_ivs)
        if req_b <= 0:
            return 0, aligned_bytes
        ov_b = _measure_byte_intervals(_intersect_byte_intervals(req_ivs, res_ivs))
        if ov_b <= 0:
            return 0, aligned_bytes
        if ov_b >= req_b:
            return aligned_bytes, 0
        hit = min(aligned_bytes, int(round(aligned_bytes * (float(ov_b) / float(req_b)))))
        return hit, aligned_bytes - hit

    def preview_access(
        self, pattern: MemoryAccessPattern, aligned_bytes: int,
    ) -> tuple[int, int]:
        """Read-only ``(l2_hit_bytes, dram_miss_copy_bytes)`` with zero state mutation."""
        if aligned_bytes <= 0:
            return 0, 0
        key = pattern.reuse_key
        if key is None:
            return 0, aligned_bytes
        if key.die != pattern.die:
            raise ModelingError(f"reuse_key_die_mismatch:{key.die}!={pattern.die}")
        if key.die not in self._l2_capacity:
            raise ModelingError(f"l2_die_undeclared:{key.die}")
        keys = set(self._active_keys.get(key.die, ()))
        keys.add(key)
        streams = self._active_streams.get(key.die, ())
        covered = {k.reuse_stream for k in keys}
        n_eff = max(1, len(keys) + sum(1 for s in streams if s not in covered))
        avail = max(0, self._l2_capacity[key.die] - self.reserved_bytes(key.die))
        eff = self.granule_bytes * (avail // (self.granule_bytes * n_eff))
        working_set = (
            pattern.reuse_distance_bytes
            if pattern.reuse_distance_bytes is not None else aligned_bytes
        )
        existing = self._resident.get(key)
        req_ivs = pattern.runs.sector_intervals()
        if pattern.direction == "read":
            if (
                existing is not None and not existing.evicted
                and existing.version == pattern.version
                and existing.working_set_bytes <= eff and working_set <= eff
            ):
                return self._range_hit_miss(req_ivs, existing.resident_ranges, aligned_bytes)
            return 0, aligned_bytes
        if pattern.visibility.private_workspace and not pattern.visibility.live_out and working_set <= eff:
            return aligned_bytes, 0
        return 0, aligned_bytes

    def admit_access(
        self,
        pattern: MemoryAccessPattern,
        aligned_bytes: int,
    ) -> tuple[int, int]:
        """Classify ``(l2_hit_bytes, dram_miss_copy_bytes)`` and update L2 state."""
        if aligned_bytes <= 0:
            return 0, 0
        key = pattern.reuse_key
        if key is None:
            return 0, aligned_bytes
        if key.die != pattern.die:
            raise ModelingError(f"reuse_key_die_mismatch:{key.die}!={pattern.die}")
        if key.die not in self._l2_capacity:
            raise ModelingError(f"l2_die_undeclared:{key.die}")
        self._active_keys[key.die].add(key)
        self._evict_over_capacity(key.die)
        eff = self.effective_l2_bytes(key.die)
        working_set = (
            pattern.reuse_distance_bytes
            if pattern.reuse_distance_bytes is not None else aligned_bytes
        )
        existing = self._resident.get(key)
        req_ivs = pattern.runs.sector_intervals()
        if pattern.direction == "read":
            if (
                existing is not None and not existing.evicted
                and existing.version == pattern.version and working_set <= eff
            ):
                hit_b, miss_b = self._range_hit_miss(req_ivs, existing.resident_ranges, aligned_bytes)
                if miss_b > 0:
                    existing.resident_ranges = _merge_byte_intervals((*existing.resident_ranges, *req_ivs))
                    existing.working_set_bytes = max(existing.working_set_bytes, working_set)
                return hit_b, miss_b
            if working_set <= eff:
                self._resident[key] = _ResidentL2Entry(
                    key=key, working_set_bytes=working_set, version=pattern.version,
                    dirty=False, private_workspace=pattern.visibility.private_workspace,
                    evicted=False, resident_ranges=req_ivs,
                )
            elif existing is not None:
                self._settle_eviction(existing)
            return 0, aligned_bytes
        is_priv = pattern.visibility.private_workspace and not pattern.visibility.live_out
        if is_priv and working_set <= eff:
            prev_res = existing.resident_ranges if (existing and not existing.evicted and existing.version == pattern.version) else ()
            prev_dirty = existing.dirty_ranges if (existing and not existing.evicted and existing.version == pattern.version) else ()
            self._resident[key] = _ResidentL2Entry(
                key=key, working_set_bytes=working_set, version=pattern.version,
                dirty=True, private_workspace=True, evicted=False,
                resident_ranges=_merge_byte_intervals((*prev_res, *req_ivs)),
                dirty_ranges=_merge_byte_intervals((*prev_dirty, *req_ivs)),
                has_future_reader=True,
            )
            return aligned_bytes, 0
        if working_set <= eff:
            self._resident[key] = _ResidentL2Entry(
                key=key, working_set_bytes=working_set, version=pattern.version,
                dirty=False, private_workspace=False, evicted=False,
                resident_ranges=req_ivs,
            )
        elif existing is not None:
            self._settle_eviction(existing)
        return 0, aligned_bytes


def bind_memory_access(
    pattern: MemoryAccessPattern,
    state: AdmissionMemoryView | None,
    *,
    hbm_bus: HbmBusSpec,
    copy_hit_bytes_per_ms: float,
    copy_miss_bytes_per_ms: float,
    l2_bytes_per_ms: float,
    pipe_row_id: str,
    l2_row_id: str,
    burst_rules: Sequence[BurstTurnRule] = (),
    commit: bool = False,
) -> BoundAccess:
    """Bind one ``MemoryAccessPattern`` into exact 3-ledger audit and row terms."""
    if hbm_bus.die != pattern.die:
        raise ModelingError(f"hbm_bus_die_mismatch:{hbm_bus.die}!={pattern.die}")
    if copy_hit_bytes_per_ms <= 0.0 or copy_miss_bytes_per_ms <= 0.0 or l2_bytes_per_ms <= 0.0:
        raise ModelingError("bind_access_pipe_rate_nonpositive")

    aligned_bytes = pattern.runs.aligned_copy_bytes()
    if aligned_bytes == 0:
        ledger = ByteLedger(
            logical_tensor_bytes=pattern.logical_bytes, aligned_transfer_bytes=0,
            dram_bus_transaction_bytes=0.0, l2_hit_bytes=0,
            dram_miss_copy_bytes=0, dram_sector_bytes=0,
        )
        after = (pattern.visibility.producer_event,) if pattern.visibility.producer_event else ()
        return BoundAccess(
            ledger=ledger, constraints=(), deliveries=(), after=after,
            owner=pattern.owner, pipe_row_id=pipe_row_id, l2_row_id=l2_row_id,
        )

    if state is not None:
        hit_bytes, miss_bytes = (
            state.admit_access(pattern, aligned_bytes)
            if commit else state.preview_access(pattern, aligned_bytes)
        )
    else:
        hit_bytes, miss_bytes = 0, aligned_bytes

    if miss_bytes > 0:
        sector_bytes, dram_service_bytes = pattern.runs.dram_sector_and_service(burst_rules)
        if hit_bytes > 0 and aligned_bytes > 0:
            miss_ratio = float(miss_bytes) / float(aligned_bytes)
            sector_bytes = int(round(sector_bytes * miss_ratio))
            dram_service_bytes *= miss_ratio
    else:
        sector_bytes = 0
        dram_service_bytes = 0.0

    ledger = ByteLedger(
        logical_tensor_bytes=pattern.logical_bytes,
        aligned_transfer_bytes=aligned_bytes,
        dram_bus_transaction_bytes=dram_service_bytes,
        l2_hit_bytes=hit_bytes,
        dram_miss_copy_bytes=miss_bytes,
        dram_sector_bytes=sector_bytes,
    )

    constraints: list[ConstraintTerm] = []
    pipe_ms = (
        float(hit_bytes) / copy_hit_bytes_per_ms
        + float(miss_bytes) / copy_miss_bytes_per_ms
    )
    row_rates: dict[str, float] = {}
    if pipe_ms > 0.0:
        constraints.append(ConstraintTerm(pipe_row_id, pipe_ms))
        row_rates[pipe_row_id] = (
            copy_miss_bytes_per_ms if miss_bytes >= hit_bytes else copy_hit_bytes_per_ms
        )
    if hit_bytes > 0:
        constraints.append(ConstraintTerm(l2_row_id, float(hit_bytes) / l2_bytes_per_ms))
        row_rates[l2_row_id] = l2_bytes_per_ms
    deliveries: list[CounterTerm] = [CounterTerm(pipe_row_id, float(aligned_bytes))]
    if hit_bytes > 0:
        deliveries.append(CounterTerm(l2_row_id, float(hit_bytes)))
    if dram_service_bytes > 0.0:
        term_r, term_w = hbm_bus.terms_for(pattern.direction, dram_service_bytes)
        if term_r.ms_per_progress > 0.0:
            constraints.append(term_r)
        if term_w.ms_per_progress > 0.0:
            constraints.append(term_w)
        ctr_id = hbm_bus.ctr_r if pattern.direction == "read" else hbm_bus.ctr_w
        deliveries.append(CounterTerm(ctr_id, dram_service_bytes))

    after = (pattern.visibility.producer_event,) if pattern.visibility.producer_event else ()
    return BoundAccess(
        ledger=ledger, constraints=tuple(constraints), deliveries=tuple(deliveries),
        after=after, owner=pattern.owner, pipe_row_id=pipe_row_id,
        l2_row_id=l2_row_id, row_rates=row_rates,
    )


__all__ = [
    "AdmissionMemoryView",
    "DirtyWriteback",
    "bind_memory_access",
]
