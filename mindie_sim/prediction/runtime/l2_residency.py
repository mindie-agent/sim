"""Shared per-die L2 residency and directional memory-access binding.

[Role] Track exact sector coverage, dirty evictions and bound byte ledgers.
[Invariants] One L2 owner per die. Hit and miss bytes sum to aligned copy bytes.
  Evicted coverage stays cold until an explicit fill.
[Timeouts & Locks] In-memory interval arithmetic with no I/O, threads or locks.
[Side-Effect Policy] Only explicit admission, release and reservation mutate L2.
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import math

from mindie_sim.schema.common import ModelingError
from .memory_physics import (
    AddressRun, AddressRunPlan,
    BoundAccess, BurstTurnRule, ByteLedger,
    CACHE_POLICY_DISABLE, CACHE_POLICY_LAST_USE, CACHE_POLICY_NORMAL, CACHE_POLICY_PERSISTENT,
    CachePolicy, VALID_CACHE_POLICIES,
    ConstraintTerm, CounterTerm,
    DEFAULT_L2_GRANULE_BYTES,
    HbmBusSpec, MemoryAccessPattern,
    ReuseKey, SectorCoverage,
    _L2_CACHE_DISABLE,
    _measure_byte_intervals, _subtract_byte_intervals,
    burst_turnaround_efficiency,
)
from .periodic_runs import (
    _coalesce_runs,
    _coverage_overlap,
    _coverage_runs,
    _merged_bytes,
    _partition_runs,
    _range_run,
    _residency_runs,
    _run_end,
)

_MEBIBYTE = 1024 * 1024
_CACHE_EVICTION_ORDER = {CACHE_POLICY_NORMAL: 0, CACHE_POLICY_LAST_USE: 1, CACHE_POLICY_PERSISTENT: 2}


_LEGACY_CACHE_POLICIES = frozenset({CACHE_POLICY_NORMAL, CACHE_POLICY_DISABLE})


def cache_policy_for_arch(arch_id: str, policy: CachePolicy) -> CachePolicy:
    """Resolve a semantic CANN cache hint for the declared ISA."""
    policies = {
        "2201": _LEGACY_CACHE_POLICIES, "3510": VALID_CACHE_POLICIES,
    }
    arch = str(arch_id).lower().removeprefix("npu_arch_").removeprefix("dav_")
    if policy not in policies.get(arch, (CACHE_POLICY_NORMAL,)):
        raise ModelingError(f"cache_policy_unsupported:{arch_id}:{policy}")
    return policy


def _p05_knee(ws_sweep: Mapping[int | str, float], capacity: int) -> tuple[tuple[float, float], ...]:
    """Recover a hit fraction from warm/cold read service times on the spill limb."""
    rates = sorted(
        (int(size[:-len("MiB")]) * _MEBIBYTE if isinstance(size, str) and size.endswith("MiB")
         else int(size), float(rate))
        for size, rate in ws_sweep.items()
    )
    if not rates or not capacity:
        return ()
    if any(size <= 0 or not math.isfinite(rate) or rate <= 0 for size, rate in rates):
        raise ModelingError("l2_ws_sweep_invalid")
    peak = max(range(len(rates)), key=lambda index: rates[index][1])
    warm = rates[peak][1]
    cold = min(rate for _, rate in rates[peak:])
    if not 0 < cold < warm:
        return ()
    points = []
    fraction = 1.0
    for size, rate in rates[peak:]:
        fraction = min(fraction, (1 / cold - 1 / rate) / (1 / cold - 1 / warm))
        points.append((size / capacity, fraction))
    return tuple(points)


def _access_plan(plan: AddressRunPlan, spec) -> AddressRunPlan:
    if spec is None:
        return plan
    return replace(
        plan, sector_bytes=None if plan._default_sector else plan.sector_bytes,
        l2_line_bytes=plan.l2_line_bytes or getattr(spec, "l2_line_bytes", None),
        l2_sector_bytes=plan.l2_sector_bytes or getattr(spec, "l2_sector_bytes", None),
        hbm_burst_granule_bytes=plan.hbm_burst_granule_bytes or getattr(spec, "hbm_burst_granule_bytes", 0),
        hbm_scatter_row_open_bytes=plan.hbm_scatter_row_open_bytes or getattr(spec, "hbm_scatter_row_open_bytes", 0),
    )


@dataclass(frozen=True, slots=True)
class DirtyWriteback:
    """Settled DRAM writeback emitted when a live dirty private L2 line evicts."""

    key: ReuseKey
    die: int
    bytes: int
    ranges: tuple[tuple[int, int], ...]
    coverage: SectorCoverage = SectorCoverage()


@dataclass(slots=True)
class _ResidentL2Entry:
    key: ReuseKey
    working_set_bytes: int
    version: int
    dirty: bool
    private_workspace: bool
    evicted: bool = False
    resident_coverage: SectorCoverage = SectorCoverage()
    resident_bytes: int = 0
    dirty_ranges: tuple[tuple[int, int], ...] = ()
    dirty_coverage: SectorCoverage = SectorCoverage()
    has_future_reader: bool = True
    cache_policy: str = CACHE_POLICY_NORMAL
    line_footprint_bytes: int | None = None
    resident_intervals_mut: list[tuple[int, int]] | None = None

    def coverage_snapshot(self) -> SectorCoverage:
        """Return an immutable ``SectorCoverage`` snapshot for external inspection."""
        if self.resident_intervals_mut is None:
            return self.resident_coverage
        return SectorCoverage(
            intervals=tuple(self.resident_intervals_mut),
            _canonical=True,
            _bytes=self.resident_bytes,
        )


def _entry_overlap_bytes(req_cov: SectorCoverage, entry: _ResidentL2Entry) -> int:
    """Compute exact byte overlap against ``entry`` without copying its mutable interval list."""
    if req_cov.empty or entry.resident_bytes <= 0:
        return 0
    if entry.resident_intervals_mut is not None and not req_cov.periodic:
        from ._periodic_stream import _overlap_canonical_intervals
        return _overlap_canonical_intervals(req_cov.intervals, entry.resident_intervals_mut)
    return _coverage_overlap(req_cov, entry.coverage_snapshot())


class AdmissionMemoryView:
    """Track shared L2 with strict LRU or an optional smooth spill curve.

    Curve points are (working set / effective capacity, hit fraction).
    ``ws_sweep`` accepts P05 working-set sizes and sustained bytes per second.
    """

    def __init__(
        self,
        l2_capacity_by_die: Mapping[int, int],
        *,
        l2_buffer_units_by_die: Mapping[int, int] | None = None,
        granule_bytes: int | None = None,
        memory_access=None,
        smooth_knee: bool = False,
        knee_ratio: float = 2,
        knee_curve: Sequence[tuple[float, float]] = (),
        spill_curve: Sequence[tuple[float, float]] = (),
        ws_sweep: Mapping[int | str, float] | None = None,
        arch_id: str | None = None,
    ) -> None:
        if granule_bytes is None:
            granule_bytes = getattr(memory_access, "l2_sector_bytes", DEFAULT_L2_GRANULE_BYTES)
        if type(granule_bytes) is not int or granule_bytes < 1:
            raise ModelingError(f"l2_granule_invalid:{granule_bytes!r}")
        self.granule_bytes = granule_bytes
        self.memory_access = memory_access
        self.arch_id = arch_id
        if not math.isfinite(knee_ratio) or knee_ratio <= 1:
            raise ModelingError(f"l2_knee_ratio_invalid:{knee_ratio}")
        if knee_curve and spill_curve:
            raise ModelingError("l2_knee_curve_declared_twice")
        curve = tuple(knee_curve or spill_curve)
        if not curve and ws_sweep:
            curve = _p05_knee(ws_sweep, max(l2_capacity_by_die.values(), default=0))
        if (any(not math.isfinite(x) or x <= 0 or not 0 <= y <= 1 for x, y in curve)
                or any(x0 >= x1 or y0 < y1 for (x0, y0), (x1, y1) in zip(curve, curve[1:]))):
            raise ModelingError("l2_knee_curve_invalid")
        self._knee_curve = curve or (((1.0, 1.0), (knee_ratio, 0.0)) if smooth_knee else ())
        self._l2_capacity: dict[int, int] = {}
        self._reserved: dict[int, dict[str, int]] = {}
        self._active_streams: dict[int, set[str]] = {}
        self._active_keys: dict[int, set[ReuseKey]] = {}
        self._template_key_users: dict[ReuseKey, int] = {}
        self._resident: dict[ReuseKey, _ResidentL2Entry] = {}
        self._pending_writebacks: dict[int, list[DirtyWriteback]] = {}
        self._last_prior_coverage: SectorCoverage = SectorCoverage()
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
        """Return per-stream capacity in strict mode and the shared budget in smooth mode."""
        cap = self._l2_capacity.get(die)
        if cap is None:
            raise ModelingError(f"l2_die_undeclared:{die}")
        avail = max(0, cap - self.reserved_bytes(die))
        n_active = 1 if self._knee_curve else max(1, self.active_stream_count(die))
        granule = self.granule_bytes
        return granule * (avail // (granule * n_active))

    def _hit_fraction(self, working_set: int, capacity: int) -> float:
        if not self._knee_curve or capacity <= 0:
            return float(working_set <= capacity)
        pressure = working_set / capacity
        x0, y0 = self._knee_curve[0]
        if pressure <= x0:
            return y0
        for x1, y1 in self._knee_curve[1:]:
            if pressure <= x1:
                return y0 + (y1 - y0) * (pressure - x0) / (x1 - x0)
            x0, y0 = x1, y1
        return y0

    def _settle_eviction(self, entry: _ResidentL2Entry) -> None:
        if entry.dirty and entry.has_future_reader:
            wb_bytes = (
                _merged_bytes(entry.dirty_coverage)
                or _measure_byte_intervals(entry.dirty_ranges)
                or entry.working_set_bytes
            )
            cov = (
                entry.dirty_coverage if not entry.dirty_coverage.empty
                else SectorCoverage(intervals=entry.dirty_ranges or ((0, wb_bytes),))
            )
            self._pending_writebacks[entry.key.die].append(DirtyWriteback(
                key=entry.key, die=entry.key.die,
                bytes=wb_bytes, ranges=entry.dirty_ranges, coverage=cov,
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
        oversized = False
        total_die = 0
        for entry in self._resident.values():
            if entry.key.die != die or entry.evicted:
                continue
            if (entry.working_set_bytes > eff
                    and (entry.dirty or not self._knee_curve
                         or not self._hit_fraction(entry.working_set_bytes, eff))):
                oversized = True
                break
            total_die += (
                min(entry.working_set_bytes, avail)
                if self._knee_curve and not entry.dirty else entry.working_set_bytes
            )
        if not oversized and total_die <= (avail if self._knee_curve else min(avail, eff)):
            return []

        victims: list[_ResidentL2Entry] = []
        evicted: set[ReuseKey] = set()

        def priority(entry):
            return _CACHE_EVICTION_ORDER[entry.cache_policy], entry.key == protect_key

        def charge(entry):
            return (min(entry.working_set_bytes, avail)
                    if self._knee_curve and not entry.dirty else entry.working_set_bytes)

        for entry in self._resident.values():
            if (entry.key.die == die and not entry.evicted and entry.working_set_bytes > eff
                    and (entry.dirty or not self._knee_curve
                         or not self._hit_fraction(entry.working_set_bytes, eff))):
                victims.append(entry)
                evicted.add(entry.key)
        if not self._knee_curve:
            by_stream: dict[str, list[_ResidentL2Entry]] = {}
            for entry in self._resident.values():
                if entry.key.die == die and not entry.evicted and entry.key not in evicted:
                    by_stream.setdefault(entry.key.reuse_stream, []).append(entry)
            for entries in by_stream.values():
                total = sum(e.working_set_bytes for e in entries)
                for victim in sorted(entries, key=priority):
                    if total <= eff:
                        break
                    victims.append(victim)
                    evicted.add(victim.key)
                    total -= victim.working_set_bytes
        live = [
            e for e in self._resident.values()
            if e.key.die == die and not e.evicted and e.key not in evicted
        ]
        total_die = sum(map(charge, live))
        for victim in sorted(live, key=priority):
            if total_die <= avail:
                break
            victims.append(victim)
            evicted.add(victim.key)
            total_die -= charge(victim)
        return victims

    def _evict_over_capacity(self, die: int, *, protect_key: ReuseKey | None = None) -> None:
        eff = self.effective_l2_bytes(die)
        avail = max(0, self._l2_capacity[die] - self.reserved_bytes(die))
        for victim in self._eviction_victims(die, eff, avail, protect_key=protect_key):
            self._settle_eviction(victim)

    def _activate_key(self, key: ReuseKey) -> None:
        self._active_keys[key.die].add(key)
        self._evict_over_capacity(key.die)

    def pending_writebacks(self, die: int) -> tuple[DirtyWriteback, ...]:
        return tuple(self._pending_writebacks.get(die, ()))

    def drain_writebacks(self, die: int) -> tuple[DirtyWriteback, ...]:
        items = tuple(self._pending_writebacks.get(die, ()))
        if die in self._pending_writebacks:
            self._pending_writebacks[die].clear()
        return items

    def pop_matching_writebacks(
        self, die: int, allocation: str, version: int, coverage: SectorCoverage,
    ) -> tuple[DirtyWriteback, ...]:
        """Pop pending writebacks on ``die`` overlapping ``(allocation, version, coverage)``."""
        queue = self._pending_writebacks.get(die)
        if not queue:
            return ()
        matched: list[DirtyWriteback] = []
        kept: list[DirtyWriteback] = []
        for item in queue:
            item_cov = item.coverage if not item.coverage.empty else SectorCoverage(intervals=item.ranges)
            if (item.key.backing_allocation == allocation
                    and item.key.tensor_version == version
                    and _coverage_overlap(coverage, item_cov) > 0):
                matched.append(item)
            else:
                kept.append(item)
        if matched:
            self._pending_writebacks[die] = kept
        return tuple(matched)

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
        req_b = _merged_bytes(req_cov)
        if req_b <= 0:
            return 0, aligned_bytes
        ov_b = _coverage_overlap(req_cov, res_cov)
        if ov_b <= 0:
            return 0, aligned_bytes
        if ov_b >= req_b:
            return aligned_bytes, 0
        hit = min(aligned_bytes, int(round(aligned_bytes * (float(ov_b) / float(req_b)))))
        return hit, aligned_bytes - hit

    def _preview_effective_l2(self, key: ReuseKey) -> int:
        if self._knee_curve:
            return self.effective_l2_bytes(key.die)
        streams = set(self._active_streams.get(key.die, ()))
        streams.update(k.reuse_stream for k in self._active_keys.get(key.die, ()))
        streams.add(key.reuse_stream)
        n_eff = max(1, len(streams))
        avail = max(0, self._l2_capacity[key.die] - self.reserved_bytes(key.die))
        return self.granule_bytes * (avail // (self.granule_bytes * n_eff))

    def _geometry(self, pattern: MemoryAccessPattern, aligned_bytes: int) -> tuple[SectorCoverage, int]:
        plan = _access_plan(pattern.runs, self.memory_access)
        sector = plan.l2_sector_bytes or plan.sector_bytes
        sectors = plan if sector == plan.sector_bytes else replace(plan, sector_bytes=sector)
        coverage = sectors.sector_coverage()
        line = plan.l2_line_bytes
        if line and line != sector:
            span = (
                plan.runs[0].sector_union_bytes(line) if len(plan.runs) == 1
                else _merged_bytes(replace(plan, sector_bytes=line).sector_coverage())
            )
        else:
            span = _merged_bytes(coverage)
        reuse_distance = pattern.reuse_distance_bytes
        return coverage, max(aligned_bytes if reuse_distance is None else reuse_distance, span)

    @staticmethod
    def _footprint(coverage: SectorCoverage, line_bytes: int | None) -> int:
        if line_bytes is None:
            return _merged_bytes(coverage)
        if not coverage.periodic:
            from ._periodic_stream import _aligned_intervals_bytes
            return _aligned_intervals_bytes(coverage.intervals, line_bytes)
        lines = AddressRunPlan(tuple(_coverage_runs(coverage)), sector_bytes=line_bytes)
        return _merged_bytes(lines.sector_coverage())

    def _policy(self, pattern: MemoryAccessPattern) -> str:
        return (pattern.cache_policy if self.arch_id is None
                else cache_policy_for_arch(self.arch_id, pattern.cache_policy))

    def _key(self, pattern: MemoryAccessPattern, aligned_bytes: int) -> ReuseKey | None:
        if aligned_bytes <= 0 or self._policy(pattern) == _L2_CACHE_DISABLE:
            return None
        key = pattern.reuse_key
        if key is not None:
            if key.die != pattern.die:
                raise ModelingError(f"reuse_key_die_mismatch:{key.die}!={pattern.die}")
            if key.die not in self._l2_capacity:
                raise ModelingError(f"l2_die_undeclared:{key.die}")
        return key

    def _existing(self, pattern, key, eff, *, preview):
        entry = self._resident.get(key)
        if entry is None or entry.evicted or entry.version != pattern.version:
            return None
        if preview:
            avail = max(0, self._l2_capacity[key.die] - self.reserved_bytes(key.die))
            if any(entry is victim for victim in self._eviction_victims(key.die, eff, avail)):
                return None
        return entry

    def _classify_access_with_overlap(
        self, pattern, aligned_bytes, eff, coverage, working_set, entry,
    ) -> tuple[int, int, int | None]:
        """Return ``(hit, miss, overlap_bytes_or_None)`` using one overlap scan."""
        if pattern.direction == "read":
            if entry is not None:
                fraction = self._hit_fraction(max(working_set, entry.working_set_bytes), eff)
                if fraction:
                    req_b = _merged_bytes(coverage)
                    ov_b = 0 if req_b <= 0 else _entry_overlap_bytes(coverage, entry)
                    if ov_b <= 0:
                        return 0, aligned_bytes, 0
                    hit = (aligned_bytes if ov_b >= req_b
                           else min(aligned_bytes, int(round(aligned_bytes * (float(ov_b) / float(req_b))))))
                    if fraction != 1.0:
                        hit = round(hit * fraction)
                    return hit, aligned_bytes - hit, ov_b
        elif (pattern.visibility.private_workspace and not pattern.visibility.live_out
              and working_set <= eff
              and not (self.arch_id is None and pattern.cache_policy == CACHE_POLICY_LAST_USE)):
            return aligned_bytes, 0, None
        return 0, aligned_bytes, None

    def _classify_access(self, pattern, aligned_bytes, eff, coverage, working_set, entry):
        """Share the hit/miss decision between preview and commit."""
        hit, miss, _ = self._classify_access_with_overlap(
            pattern, aligned_bytes, eff, coverage, working_set, entry,
        )
        return hit, miss

    def preview_access(self, pattern: MemoryAccessPattern, aligned_bytes: int) -> tuple[int, int]:
        """Read-only hit and miss copy bytes."""
        key = self._key(pattern, aligned_bytes)
        if key is None:
            return 0, max(0, aligned_bytes)
        eff = self._preview_effective_l2(key)
        coverage, working_set = self._geometry(pattern, aligned_bytes)
        entry = self._existing(pattern, key, eff, preview=True)
        return self._classify_access(pattern, aligned_bytes, eff, coverage, working_set, entry)

    def resident_coverage_for(self, pattern: MemoryAccessPattern, aligned_bytes: int) -> SectorCoverage:
        """Return eligible address coverage before applying any statistical spill."""
        key = self._key(pattern, aligned_bytes)
        if key is not None and pattern.direction == "read":
            eff = self._preview_effective_l2(key)
            existing = self._existing(pattern, key, eff, preview=True)
            if existing is not None:
                coverage, working_set = self._geometry(pattern, aligned_bytes)
                if not coverage.empty and self._hit_fraction(max(working_set, existing.working_set_bytes), eff):
                    return existing.coverage_snapshot()
        return SectorCoverage()

    def resident_intervals_for(self, pattern: MemoryAccessPattern, aligned_bytes: int) -> tuple[tuple[int, int], ...]:
        """Return currently resident L2 sector intervals applicable to ``pattern``."""
        return self.resident_coverage_for(pattern, aligned_bytes).intervals

    def miss_intervals(
        self, pattern: MemoryAccessPattern, aligned_bytes: int,
    ) -> tuple[tuple[int, int], ...]:
        """Return exact missed sector byte intervals for ``pattern`` against current L2."""
        req_ivs = pattern.runs.sector_intervals()
        res_ivs = self.resident_intervals_for(pattern, aligned_bytes)
        return _subtract_byte_intervals(req_ivs, res_ivs) if res_ivs else req_ivs

    def admit_access(self, pattern: MemoryAccessPattern, aligned_bytes: int) -> tuple[int, int]:
        """Classify copy bytes and update exact address coverage and LRU state."""
        hit, miss, _ = self._admit_access_with_prior(pattern, aligned_bytes)
        return hit, miss

    def _merge_entry_inplace(
        self, existing: _ResidentL2Entry, pattern: MemoryAccessPattern,
        coverage: SectorCoverage, working_set: int, eff: int,
        read: bool, dirty: bool, dirty_cov: SectorCoverage,
        resident_span: int, overlap: int,
    ) -> bool:
        """Attempt exact merge into ``existing``; returns True if updated in-place."""
        merged_span = existing.resident_bytes + resident_span - overlap
        spec = self.memory_access
        line = pattern.runs.l2_line_bytes or getattr(spec, "l2_line_bytes", None)
        sector = pattern.runs.l2_sector_bytes or getattr(spec, "l2_sector_bytes", None)
        if line == (sector or pattern.runs.sector_bytes):
            line = None
        can_list_merge = (
            not coverage.periodic
            and (existing.resident_intervals_mut is not None or not existing.resident_coverage.periodic)
        )
        if can_list_merge:
            from ._periodic_stream import _plan_canonical_list_merge
            ivs = existing.resident_intervals_mut
            if ivs is None:
                ivs = list(existing.resident_coverage.intervals)
                existing.resident_intervals_mut = ivs
            lo_idx, hi_idx, merged_mid, new_fp = _plan_canonical_list_merge(
                ivs, coverage.intervals, existing.line_footprint_bytes, line,
            )
            footprint = merged_span if new_fp is None else new_fp
            merged_ws = max(existing.working_set_bytes, working_set, footprint)
            if not (self._hit_fraction(merged_ws, eff) if read and not existing.dirty else merged_ws <= eff):
                return False
            ivs[lo_idx:hi_idx] = merged_mid
            existing.working_set_bytes = merged_ws
            existing.resident_bytes = merged_span
            existing.line_footprint_bytes = new_fp
            existing.cache_policy = pattern.cache_policy
            if dirty:
                existing.dirty = True
                existing.dirty_coverage = existing.dirty_coverage.union(dirty_cov)
                existing.dirty_ranges = existing.dirty_coverage.intervals
            self._touch_resident(existing)
            return True
        base_cov = existing.coverage_snapshot()
        merged = base_cov.union(coverage) if line else None
        footprint = self._footprint(merged, line) if merged is not None else merged_span
        merged_ws = max(existing.working_set_bytes, working_set, footprint)
        if not (self._hit_fraction(merged_ws, eff) if read and not existing.dirty else merged_ws <= eff):
            return False
        existing.resident_intervals_mut = None
        existing.resident_coverage = base_cov.union(coverage) if merged is None else merged
        existing.working_set_bytes = merged_ws
        existing.resident_bytes = merged_span
        existing.line_footprint_bytes = footprint if line else None
        existing.cache_policy = pattern.cache_policy
        if dirty:
            existing.dirty = True
            existing.dirty_coverage = existing.dirty_coverage.union(dirty_cov)
            existing.dirty_ranges = existing.dirty_coverage.intervals
        self._touch_resident(existing)
        return True

    def _admit_access_with_prior(
        self, pattern: MemoryAccessPattern, aligned_bytes: int,
    ) -> tuple[int, int, SectorCoverage]:
        """Single-pass admission returning ``(hit, miss, prior_resident_coverage)``."""
        key = self._key(pattern, aligned_bytes)
        if key is None:
            return 0, max(0, aligned_bytes), SectorCoverage()
        self._activate_key(key)
        eff = self.effective_l2_bytes(key.die)
        coverage, working_set = self._geometry(pattern, aligned_bytes)
        existing = self._existing(pattern, key, eff, preview=False)
        hit, miss, ov_cached = self._classify_access_with_overlap(
            pattern, aligned_bytes, eff, coverage, working_set, existing,
        )
        prior_out = (
            existing.coverage_snapshot()
            if (existing is not None and pattern.direction == "read" and hit > 0 and miss > 0)
            else SectorCoverage()
        )
        self._last_prior_coverage = prior_out
        read = pattern.direction == "read"
        legacy_last = self.arch_id is None and pattern.cache_policy == CACHE_POLICY_LAST_USE
        if read and legacy_last and existing is not None and working_set <= eff:
            if not existing.dirty:
                existing.evicted = True
            return hit, miss, prior_out

        previous = self._resident.get(key)
        if legacy_last or not (self._hit_fraction(working_set, eff) if read else working_set <= eff):
            if previous is not None and not previous.evicted:
                self._settle_eviction(previous)
            return hit, miss, prior_out

        dirty = not read and hit > 0
        merge = existing is not None and (read or dirty)
        dirty_cov = coverage if dirty else SectorCoverage()
        resident_span = _merged_bytes(coverage)
        if merge:
            if read and not miss:
                existing.cache_policy = pattern.cache_policy
                self._touch_resident(existing)
                return hit, miss, prior_out
            overlap = _entry_overlap_bytes(coverage, existing) if ov_cached is None else ov_cached
            if self._merge_entry_inplace(
                existing, pattern, coverage, working_set, eff,
                read, dirty, dirty_cov, resident_span, overlap,
            ):
                self._evict_over_capacity(key.die, protect_key=key)
                return hit, miss, prior_out
        if previous is not None and not previous.evicted:
            self._settle_eviction(previous)
        self._touch_resident(_ResidentL2Entry(
            key=key, working_set_bytes=working_set, version=pattern.version,
            dirty=dirty, private_workspace=pattern.visibility.private_workspace,
            resident_coverage=coverage, resident_bytes=resident_span,
            dirty_ranges=dirty_cov.intervals, dirty_coverage=dirty_cov, cache_policy=pattern.cache_policy,
            line_footprint_bytes=None,
        ))
        self._evict_over_capacity(key.die, protect_key=key)
        return hit, miss, prior_out


def _burst_service_bytes(plan: AddressRunPlan, sectors: int, run: AddressRun, rules: Sequence[BurstTurnRule],
                        *, width: int | None = None, stride: int | None = None) -> float:
    if rules:
        return sectors / burst_turnaround_efficiency(
            run.row_bytes if width is None else width,
            run.stride_bytes if stride is None else stride, rules,
        )
    granule = 0
    if run.layout in {"scatter", "scattered", "gather", "indexed"}:
        granule = plan.hbm_scatter_row_open_bytes
    if run.stride_bytes > run.row_bytes:
        granule = max(granule, plan.hbm_burst_granule_bytes)
    if granule <= run.row_bytes:
        return float(sectors)
    # Sector fills may already pay for part or all of the service granule.
    wire = run.sector_union_bytes(plan.sector_bytes)
    rows = run.rows if run.stride_bytes else 1
    return sectors * max(1.0, rows * granule / wire)


def _dram_service_segments(
    plan: AddressRunPlan, covered: SectorCoverage,
) -> Iterator[tuple[int, int, int]]:
    """Yield missed bytes and burst geometry; width zero denotes scale metadata."""
    for span, width, stride, _ in _dram_segments(plan, covered):
        yield span, width, stride


def _dram_segments(
    plan: AddressRunPlan, covered: SectorCoverage,
) -> Iterator[tuple[int, int, int, AddressRun | None]]:
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
            yield run.sector_union_bytes(sec), run.row_bytes, run.stride_bytes, run
            continue
        run_cov = AddressRunPlan((run,), sector_bytes=sec).sector_coverage()
        if not partial or run_cov.periodic:
            missed = (_merged_bytes(run_cov) - _coverage_overlap(run_cov, covered)
                      if run_cov.periodic or covered.periodic else
                      _measure_byte_intervals(_subtract_byte_intervals(run_cov.intervals, covered.intervals)))
            if missed > 0:
                missed_sectors = True
                yield missed, run.row_bytes, run.stride_bytes, run
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
                    yield span, span, stride if stride == 0 or stride > run.row_bytes else span, run
        if len(plan.runs) > 1:
            covered = covered.union(run_cov)
    if missed_sectors and plan.scale_bytes:
        yield ((plan.scale_bytes + sec - 1) // sec) * sec, 0, 0, None


def _missed_dram_sector_and_service(
    plan: AddressRunPlan, resident_cov: SectorCoverage, burst_rules: Sequence[BurstTurnRule],
) -> tuple[int, float]:
    sector_total, service_total = 0, 0.0
    for span, width, stride, run in _dram_segments(plan, resident_cov):
        sector_total += span
        service_total += span if run is None else _burst_service_bytes(
            plan, span, run, burst_rules, width=width, stride=stride,
        )
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

    plan = _access_plan(pattern.runs, state.memory_access if isinstance(state, AdmissionMemoryView) else None)
    plan = _access_plan(plan, hbm_bus.memory_access)
    if plan is not pattern.runs:
        pattern = replace(pattern, runs=plan)
    aligned_bytes = plan.aligned_copy_bytes()
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

    res_cov = SectorCoverage()
    if state is None:
        hit_bytes, miss_bytes = 0, aligned_bytes
    elif commit:
        if hasattr(state, "_admit_access_with_prior"):
            hit_bytes, miss_bytes, res_cov = state._admit_access_with_prior(pattern, aligned_bytes)
        else:
            if hasattr(state, "resident_coverage_for"):
                res_cov = state.resident_coverage_for(pattern, aligned_bytes)
            hit_bytes, miss_bytes = state.admit_access(pattern, aligned_bytes)
    else:
        hit_bytes, miss_bytes = state.preview_access(pattern, aligned_bytes)
        if hit_bytes and miss_bytes:
            res_cov = state.resident_coverage_for(pattern, aligned_bytes)

    if miss_bytes > 0:
        if hit_bytes > 0 and not res_cov.empty:
            sector_bytes, dram_service_bytes = _missed_dram_sector_and_service(
                plan, res_cov, burst_rules,
            )
            if isinstance(state, AdmissionMemoryView) and state._knee_curve:
                coverage, _ = state._geometry(pattern, aligned_bytes)
                spatial_hits, _ = state._coverage_hit_miss(coverage, res_cov, aligned_bytes)
                if spatial_hits > hit_bytes:
                    spill = (spatial_hits - hit_bytes) / spatial_hits
                    cold_sectors, cold_service = plan.dram_sector_and_service(burst_rules)
                    sector_bytes += round(max(0, cold_sectors - sector_bytes) * spill)
                    dram_service_bytes += max(0.0, cold_service - dram_service_bytes) * spill
        else:
            sector_bytes, dram_service_bytes = plan.dram_sector_and_service(burst_rules)
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
        if sector_bytes:
            deliveries.append(CounterTerm(ctr_id, float(sector_bytes)))

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
