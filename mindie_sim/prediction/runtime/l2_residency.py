"""Dynamic N-stream L2 working-set residency, dirty writeback, and 3-ledger binding.

[Role] Track per-die shared L2 residency across arbitrary N >= 1 active streams
  and full ``ReuseKey`` identities using exact sector coverage, settle
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

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError
from .memory_physics import (
    AddressRun,
    AddressRunPlan,
    BoundAccess,
    BurstTurnRule,
    ByteLedger,
    CACHE_POLICY_LAST_USE,
    ConstraintTerm,
    CounterTerm,
    DEFAULT_L2_GRANULE_BYTES,
    HbmBusSpec,
    MemoryAccessPattern,
    ReuseKey,
    SectorCoverage,
    _L2_CACHE_DISABLE,
    _intersect_byte_intervals,
    _measure_byte_intervals,
    _merge_byte_intervals,
    _subtract_byte_intervals,
    burst_turnaround_efficiency,
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
    resident_coverage: SectorCoverage = SectorCoverage()
    dirty_ranges: tuple[tuple[int, int], ...] = ()
    dirty_coverage: SectorCoverage = SectorCoverage()
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
        self._template_key_users: dict[ReuseKey, int] = {}
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
        streams = set(self._active_streams.get(die, ()))
        streams.update(k.reuse_stream for k in self._active_keys.get(die, ()))
        return len(streams)

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
            wb_bytes = (
                entry.dirty_coverage.measure_bytes()
                or _measure_byte_intervals(entry.dirty_ranges)
                or entry.working_set_bytes
            )
            self._pending_writebacks[entry.key.die].append(DirtyWriteback(
                key=entry.key, die=entry.key.die,
                bytes=wb_bytes, ranges=entry.dirty_ranges,
            ))
            entry.dirty = False
            entry.dirty_ranges = ()
            entry.dirty_coverage = SectorCoverage()
        entry.evicted = True

    def _touch_resident(self, entry: _ResidentL2Entry) -> None:
        self._resident.pop(entry.key, None)
        self._resident[entry.key] = entry

    def _eviction_victims(
        self, die: int, eff: int, avail: int, *, protect_key: ReuseKey | None = None,
    ) -> list[_ResidentL2Entry]:
        """Pure read-only LRU victim list matching exact over-capacity eviction rules."""
        victims: list[_ResidentL2Entry] = []
        evicted: set[ReuseKey] = set()
        for entry in self._resident.values():
            if entry.key.die == die and not entry.evicted and entry.working_set_bytes > eff:
                victims.append(entry)
                evicted.add(entry.key)
        by_stream: dict[str, list[_ResidentL2Entry]] = {}
        for entry in self._resident.values():
            if entry.key.die == die and not entry.evicted and entry.key not in evicted:
                by_stream.setdefault(entry.key.reuse_stream, []).append(entry)
        for entries in by_stream.values():
            total = sum(e.working_set_bytes for e in entries)
            for victim in sorted(entries, key=lambda e: e.key == protect_key):
                if total <= eff:
                    break
                victims.append(victim)
                evicted.add(victim.key)
                total -= victim.working_set_bytes
        live = [
            e for e in self._resident.values()
            if e.key.die == die and not e.evicted and e.key not in evicted
        ]
        total_die = sum(e.working_set_bytes for e in live)
        for victim in sorted(live, key=lambda e: e.key == protect_key):
            if total_die <= avail:
                break
            victims.append(victim)
            evicted.add(victim.key)
            total_die -= victim.working_set_bytes
        return victims

    def _evict_over_capacity(self, die: int, *, protect_key: ReuseKey | None = None) -> None:
        eff = self.effective_l2_bytes(die)
        avail = max(0, self._l2_capacity[die] - self.reserved_bytes(die))
        for victim in self._eviction_victims(die, eff, avail, protect_key=protect_key):
            self._settle_eviction(victim)

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

    def resolve_reuse_key(self, fallback: ReuseKey) -> ReuseKey:
        """Find the latest resident producer, then an active key for this allocation version."""
        identity = fallback.die, fallback.backing_allocation, fallback.tensor_version

        def matches(key):
            return (key.die, key.backing_allocation, key.tensor_version) == identity

        for key in reversed(self._resident):
            if matches(key) and self.is_resident(key):
                return key
        active = self._active_keys.get(fallback.die, ())
        return min((key for key in active if matches(key)), key=lambda key: key.reuse_stream, default=fallback)

    def retain_template_key(self, key: ReuseKey) -> bool:
        """Share template-owned key lifetimes and leave caller-owned keys with the caller."""
        users = self._template_key_users.get(key, 0)
        if not users and key in self._active_keys.get(key.die, ()):
            return False
        self._template_key_users[key] = users + 1
        return True

    def release_template_key(self, key: ReuseKey) -> None:
        users = self._template_key_users[key]
        if users == 1:
            del self._template_key_users[key]
            self.release_key(key)
        else:
            self._template_key_users[key] = users - 1

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
    def _coverage_hit_miss(
        req_cov: SectorCoverage,
        res_cov: SectorCoverage,
        aligned_bytes: int,
    ) -> tuple[int, int]:
        if req_cov.empty or res_cov.empty:
            return 0, aligned_bytes
        req_b = req_cov.measure_bytes()
        if req_b <= 0:
            return 0, aligned_bytes
        ov_b = req_cov.overlap_bytes(res_cov)
        if ov_b <= 0:
            return 0, aligned_bytes
        if ov_b >= req_b:
            return aligned_bytes, 0
        hit = min(aligned_bytes, int(round(aligned_bytes * (float(ov_b) / float(req_b)))))
        return hit, aligned_bytes - hit

    def _preview_effective_l2(self, key: ReuseKey) -> int:
        streams = set(self._active_streams.get(key.die, ()))
        streams.update(k.reuse_stream for k in self._active_keys.get(key.die, ()))
        streams.add(key.reuse_stream)
        n_eff = max(1, len(streams))
        avail = max(0, self._l2_capacity[key.die] - self.reserved_bytes(key.die))
        return self.granule_bytes * (avail // (self.granule_bytes * n_eff))

    def _classify_access(
        self,
        pattern: MemoryAccessPattern,
        aligned_bytes: int,
        eff: int,
        req_cov: SectorCoverage,
        working_set: int,
        *,
        evicted_keys: set[ReuseKey] | frozenset[ReuseKey] = frozenset(),
    ) -> tuple[int, int]:
        """Shared read-only hit/miss decision for both preview and commit."""
        existing = self._resident.get(pattern.reuse_key)
        is_last_use = pattern.cache_policy == CACHE_POLICY_LAST_USE
        if pattern.direction == "read":
            if (
                existing is not None and not existing.evicted
                and pattern.reuse_key not in evicted_keys
                and existing.version == pattern.version
                and existing.working_set_bytes <= eff and working_set <= eff
            ):
                return self._coverage_hit_miss(req_cov, existing.resident_coverage, aligned_bytes)
            return 0, aligned_bytes
        if (
            pattern.visibility.private_workspace
            and not pattern.visibility.live_out
            and working_set <= eff
            and not is_last_use
        ):
            return aligned_bytes, 0
        return 0, aligned_bytes

    def preview_access(
        self, pattern: MemoryAccessPattern, aligned_bytes: int,
    ) -> tuple[int, int]:
        """Read-only ``(l2_hit_bytes, dram_miss_copy_bytes)`` with zero state mutation."""
        if aligned_bytes <= 0:
            return 0, 0
        if pattern.cache_policy == _L2_CACHE_DISABLE:
            return 0, aligned_bytes
        key = pattern.reuse_key
        if key is None:
            return 0, aligned_bytes
        if key.die != pattern.die:
            raise ModelingError(f"reuse_key_die_mismatch:{key.die}!={pattern.die}")
        if key.die not in self._l2_capacity:
            raise ModelingError(f"l2_die_undeclared:{key.die}")
        eff = self._preview_effective_l2(key)
        avail = max(0, self._l2_capacity[key.die] - self.reserved_bytes(key.die))
        evicted_keys = {e.key for e in self._eviction_victims(key.die, eff, avail)}
        req_cov = pattern.runs.sector_coverage()
        working_set = max(
            pattern.reuse_distance_bytes
            if pattern.reuse_distance_bytes is not None else aligned_bytes,
            req_cov.measure_bytes(),
        )
        return self._classify_access(
            pattern, aligned_bytes, eff, req_cov, working_set, evicted_keys=evicted_keys,
        )

    def resident_coverage_for(
        self, pattern: MemoryAccessPattern, aligned_bytes: int,
    ) -> SectorCoverage:
        """Return currently resident L2 ``SectorCoverage`` applicable to ``pattern``."""
        if aligned_bytes <= 0 or pattern.cache_policy == _L2_CACHE_DISABLE:
            return SectorCoverage()
        req_cov = pattern.runs.sector_coverage()
        if req_cov.empty:
            return SectorCoverage()
        key = pattern.reuse_key
        if key is None or key.die != pattern.die or key.die not in self._l2_capacity:
            return SectorCoverage()
        existing = self._resident.get(key)
        if (
            pattern.direction == "read" and existing is not None
            and not existing.evicted and existing.version == pattern.version
        ):
            eff = self._preview_effective_l2(key)
            avail = max(0, self._l2_capacity[key.die] - self.reserved_bytes(key.die))
            evicted_keys = {e.key for e in self._eviction_victims(key.die, eff, avail)}
            ws = max(
                pattern.reuse_distance_bytes if pattern.reuse_distance_bytes is not None else aligned_bytes,
                req_cov.measure_bytes(),
            )
            if key not in evicted_keys and existing.working_set_bytes <= eff and ws <= eff:
                return existing.resident_coverage
        return SectorCoverage()

    def resident_intervals_for(
        self, pattern: MemoryAccessPattern, aligned_bytes: int,
    ) -> tuple[tuple[int, int], ...]:
        """Return currently resident L2 sector intervals applicable to ``pattern``."""
        return self.resident_coverage_for(pattern, aligned_bytes).intervals

    def _fill_identity(self, pattern):
        key = pattern.reuse_key
        if key is not None and pattern.direction == "read" and pattern.cache_policy != _L2_CACHE_DISABLE:
            return self, key.die, key.backing_allocation, key.tensor_version

    def inflight_fill_dependencies(self, engine, pattern, resident: SectorCoverage):
        """Return same-allocation fills that cover sectors this reader still misses."""
        identity = self._fill_identity(pattern)
        pending = getattr(engine, "_inflight_l2_fills", {}).get(identity)
        if not pending:
            return ()
        requested = pattern.runs.sector_coverage()
        waits = []
        for done, (filling, previously_resident) in pending.items():
            overlap = requested.overlap_bytes(filling)
            if not overlap:
                continue
            available = resident.union(previously_resident)
            # Inclusion-exclusion gives the intersection of both requests with
            # the sectors excluded from the reader's or the fill's misses.
            excluded = (requested.overlap_bytes(available) + filling.overlap_bytes(available)
                        - requested.union(filling).overlap_bytes(available))
            if overlap > excluded:
                waits.append(done)
        return tuple(waits)

    def start_l2_fill(self, engine, name, pattern, aligned_bytes, resident: SectorCoverage):
        identity = self._fill_identity(pattern)
        if identity is None or pattern.cache_policy == CACHE_POLICY_LAST_USE:
            return None
        coverage = pattern.runs.sector_coverage()
        working_set = pattern.reuse_distance_bytes if pattern.reuse_distance_bytes is not None else aligned_bytes
        if max(working_set, coverage.measure_bytes()) > self._preview_effective_l2(pattern.reuse_key):
            return None
        done = engine.future(name + ":l2_fill")
        # The engine owns events. Analytical snapshots of this view contain
        # committed residency only and never copy a running engine.
        if not hasattr(engine, "_inflight_l2_fills"):
            engine._inflight_l2_fills = {}
        engine._inflight_l2_fills.setdefault(identity, {})[done] = (coverage, resident)
        return done

    def finish_l2_fill(self, engine, pattern, done):
        identity = self._fill_identity(pattern)
        fills = engine._inflight_l2_fills
        del fills[identity][done]
        if not fills[identity]:
            del fills[identity]
        engine.resolve(done)

    def miss_intervals(
        self, pattern: MemoryAccessPattern, aligned_bytes: int,
    ) -> tuple[tuple[int, int], ...]:
        """Return exact missed sector byte intervals for ``pattern`` against current L2."""
        req_ivs = pattern.runs.sector_intervals()
        res_ivs = self.resident_intervals_for(pattern, aligned_bytes)
        return _subtract_byte_intervals(req_ivs, res_ivs) if res_ivs else req_ivs

    def admit_access(
        self,
        pattern: MemoryAccessPattern,
        aligned_bytes: int,
    ) -> tuple[int, int]:
        """Classify ``(l2_hit_bytes, dram_miss_copy_bytes)`` and update L2 state."""
        if aligned_bytes <= 0:
            return 0, 0
        if pattern.cache_policy == _L2_CACHE_DISABLE:
            return 0, aligned_bytes
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
        req_cov = pattern.runs.sector_coverage()
        req_ivs = req_cov.intervals
        req_span = req_cov.measure_bytes()
        working_set = max(
            pattern.reuse_distance_bytes if pattern.reuse_distance_bytes is not None else aligned_bytes,
            req_span,
        )
        hit_b, miss_b = self._classify_access(pattern, aligned_bytes, eff, req_cov, working_set)
        existing = self._resident.get(key)
        is_last_use = pattern.cache_policy == CACHE_POLICY_LAST_USE
        if pattern.direction == "read":
            if (
                existing is not None and not existing.evicted
                and existing.version == pattern.version and working_set <= eff
            ):
                if is_last_use:
                    if not existing.dirty:
                        existing.evicted = True
                    return hit_b, miss_b
                if miss_b > 0:
                    merged_cov = existing.resident_coverage.union(req_cov)
                    merged_b = merged_cov.measure_bytes()
                    if max(existing.working_set_bytes, working_set, merged_b) <= eff:
                        existing.resident_coverage = merged_cov
                        existing.resident_ranges = merged_cov.intervals
                        existing.working_set_bytes = max(
                            existing.working_set_bytes, working_set, merged_b,
                        )
                        self._touch_resident(existing)
                    else:
                        self._settle_eviction(existing)
                        self._touch_resident(_ResidentL2Entry(
                            key=key, working_set_bytes=working_set, version=pattern.version,
                            dirty=False, private_workspace=pattern.visibility.private_workspace,
                            evicted=False, resident_ranges=req_ivs, resident_coverage=req_cov,
                        ))
                else:
                    self._touch_resident(existing)
                self._evict_over_capacity(key.die, protect_key=key)
                return hit_b, miss_b
            if not is_last_use and working_set <= eff:
                if existing is not None and not existing.evicted:
                    self._settle_eviction(existing)
                self._touch_resident(_ResidentL2Entry(
                    key=key, working_set_bytes=working_set, version=pattern.version,
                    dirty=False, private_workspace=pattern.visibility.private_workspace,
                    evicted=False, resident_ranges=req_ivs, resident_coverage=req_cov,
                ))
                self._evict_over_capacity(key.die, protect_key=key)
            elif existing is not None:
                self._settle_eviction(existing)
            return hit_b, miss_b
        if hit_b > 0:
            same = existing is not None and not existing.evicted and existing.version == pattern.version
            prev_cov = existing.resident_coverage if same else SectorCoverage()
            prev_dirty = existing.dirty_ranges if same else ()
            prev_dirty_cov = existing.dirty_coverage if same else SectorCoverage()
            merged_cov = prev_cov.union(req_cov)
            merged_b = merged_cov.measure_bytes()
            if same and max(existing.working_set_bytes, working_set, merged_b) <= eff:
                res_cov = merged_cov
                res_ivs = merged_cov.intervals
                dirty_ivs = _merge_byte_intervals((*prev_dirty, *req_ivs))
                dirty_cov = prev_dirty_cov.union(req_cov)
                ws_new = max(existing.working_set_bytes, working_set, merged_b)
            else:
                if existing is not None and not existing.evicted:
                    self._settle_eviction(existing)
                res_cov, res_ivs, dirty_ivs, dirty_cov, ws_new = (
                    req_cov, req_ivs, req_ivs, req_cov, working_set,
                )
            self._touch_resident(_ResidentL2Entry(
                key=key, working_set_bytes=ws_new, version=pattern.version,
                dirty=True, private_workspace=True, evicted=False,
                resident_ranges=res_ivs, resident_coverage=res_cov,
                dirty_ranges=dirty_ivs, dirty_coverage=dirty_cov, has_future_reader=True,
            ))
            self._evict_over_capacity(key.die, protect_key=key)
            return hit_b, miss_b
        if not is_last_use and working_set <= eff:
            if existing is not None and not existing.evicted:
                self._settle_eviction(existing)
            self._touch_resident(_ResidentL2Entry(
                key=key, working_set_bytes=working_set, version=pattern.version,
                dirty=False, private_workspace=False, evicted=False,
                resident_ranges=req_ivs, resident_coverage=req_cov,
            ))
            self._evict_over_capacity(key.die, protect_key=key)
        elif existing is not None:
            self._settle_eviction(existing)
        return hit_b, miss_b


def _dram_service_segments(
    plan: AddressRunPlan, covered: SectorCoverage,
) -> Iterator[tuple[int, int, int]]:
    """Yield missed bytes and burst geometry; width zero denotes scale metadata."""
    partial = not covered.empty
    sec = plan.sector_bytes
    seen_aliases: set[str] = set()
    missed_sectors = False
    for run in plan.runs:
        if run.rows == 0 or run.row_bytes == 0:
            continue
        if run.proven_alias_group is not None:
            if run.proven_alias_group in seen_aliases:
                continue
            seen_aliases.add(run.proven_alias_group)
        if not partial and len(plan.runs) == 1:
            missed_sectors = True
            yield run.sector_union_bytes(sec), run.row_bytes, run.stride_bytes
            continue
        run_cov = AddressRunPlan((run,), sector_bytes=sec).sector_coverage()
        if not partial or run_cov.periodic:
            missed = (run_cov.measure_bytes() - run_cov.overlap_bytes(covered)
                      if run_cov.periodic or covered.periodic else
                      _measure_byte_intervals(_subtract_byte_intervals(run_cov.intervals, covered.intervals)))
            if missed > 0:
                missed_sectors = True
                yield missed, run.row_bytes, run.stride_bytes
        else:
            intervals = (run_cov.intervals if covered.periodic else
                         _subtract_byte_intervals(run_cov.intervals, covered.intervals))
            for lo, hi in intervals:
                span = hi - lo
                if covered.periodic:
                    span -= SectorCoverage(intervals=((lo, hi),)).overlap_bytes(covered)
                if span > 0:
                    missed_sectors = True
                    stride = run.stride_bytes
                    yield span, span, stride if stride == 0 or stride > run.row_bytes else span
        covered = covered.union(run_cov)
    if missed_sectors and plan.scale_bytes:
        yield ((plan.scale_bytes + sec - 1) // sec) * sec, 0, 0


def _missed_dram_sector_and_service(
    plan: AddressRunPlan, resident_cov: SectorCoverage, burst_rules: Sequence[BurstTurnRule],
) -> tuple[int, float]:
    sector_total, service_total = 0, 0.0
    for span, width, stride in _dram_service_segments(plan, resident_cov):
        sector_total += span
        service_total += float(span) / burst_turnaround_efficiency(width, stride, burst_rules)
    return sector_total, service_total


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
    nominal_pipe_bytes_per_ms: float | None = None,
) -> BoundAccess:
    """Bind one ``MemoryAccessPattern`` into exact 3-ledger audit and row terms."""
    if hbm_bus.die != pattern.die:
        raise ModelingError(f"hbm_bus_die_mismatch:{hbm_bus.die}!={pattern.die}")
    if copy_hit_bytes_per_ms <= 0.0 or copy_miss_bytes_per_ms <= 0.0 or l2_bytes_per_ms <= 0.0:
        raise ModelingError("bind_access_pipe_rate_nonpositive")
    nom_pipe = (
        float(nominal_pipe_bytes_per_ms)
        if nominal_pipe_bytes_per_ms is not None else float(copy_hit_bytes_per_ms)
    )
    if nom_pipe <= 0.0:
        raise ModelingError("bind_access_pipe_rate_nonpositive")
    bind_cfg = (
        hbm_bus, float(copy_hit_bytes_per_ms), float(copy_miss_bytes_per_ms),
        float(l2_bytes_per_ms), pipe_row_id, l2_row_id, tuple(burst_rules), nom_pipe,
    )

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
            bind_config=bind_cfg,
        )

    res_cov = state.resident_coverage_for(pattern, aligned_bytes) if state is not None else SectorCoverage()
    if state is not None:
        hit_bytes, miss_bytes = (
            state.admit_access(pattern, aligned_bytes)
            if commit else state.preview_access(pattern, aligned_bytes)
        )
    else:
        hit_bytes, miss_bytes = 0, aligned_bytes

    if miss_bytes > 0:
        if hit_bytes > 0 and not res_cov.empty:
            sector_bytes, dram_service_bytes = _missed_dram_sector_and_service(
                pattern.runs, res_cov, burst_rules,
            )
        else:
            sector_bytes, dram_service_bytes = pattern.runs.dram_sector_and_service(burst_rules)
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
        row_rates[pipe_row_id] = nom_pipe
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
        l2_row_id=l2_row_id, row_rates=row_rates, bind_config=bind_cfg,
    )


__all__ = [
    "AdmissionMemoryView",
    "DirtyWriteback",
    "bind_memory_access",
]
