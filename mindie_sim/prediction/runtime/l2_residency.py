"""Shared per-die L2 residency and directional memory-access binding.

[Role] Track exact sector coverage, dirty evictions and bound byte ledgers.
[Invariants] One L2 owner per die. Hit and miss bytes sum to aligned copy bytes.
  Evicted coverage stays cold until an explicit fill.
[Timeouts & Locks] In-memory interval arithmetic with no I/O, threads or locks.
[Side-Effect Policy] Only explicit admission, release and reservation mutate L2.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
import math
from typing import Any, NewType, cast

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.device_input import MemoryAccessSpec
from .memory_physics import (
    AddressRun, AddressRunPlan,
    BoundAccess, BurstTurnRule, ByteLedger,
    CACHE_POLICY_DISABLE, CACHE_POLICY_LAST_USE, CACHE_POLICY_NORMAL, CACHE_POLICY_PERSISTENT,
    CachePolicy, VALID_CACHE_POLICIES,
    ConstraintTerm, CounterTerm,
    DEFAULT_L2_GRANULE_BYTES,
    HbmBusSpec, MemoryAccessPattern, PeriodicSectorRun,
    ReuseKey, SectorCoverage,
    _L2_CACHE_DISABLE, _MAX_EXPLICIT_SECTOR_INTERVALS,
    _align_up, _intersect_byte_intervals, _measure_byte_intervals,
    _merge_byte_intervals, _subtract_byte_intervals,
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
_RECT_LEDGER_CACHE_LIMIT = 256
_CACHE_EVICTION_ORDER = {CACHE_POLICY_NORMAL: 0, CACHE_POLICY_LAST_USE: 1, CACHE_POLICY_PERSISTENT: 2}
_NORMAL_CACHE_POLICY: CachePolicy = "normal"

RawDieL2Bytes = NewType("RawDieL2Bytes", int)
UsableL2Bytes = NewType("UsableL2Bytes", int)


_LEGACY_CACHE_POLICIES = frozenset({CACHE_POLICY_NORMAL, CACHE_POLICY_DISABLE})


def _plan_sector_bytes(plan: AddressRunPlan) -> int:
    sec = plan.sector_bytes
    if sec is None:
        raise ModelingError("address_run_plan_sector_bytes_unset")
    return sec


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
    fitting = [rate for size, rate in rates if size <= capacity]
    spill = [(size, rate) for size, rate in rates if size > capacity]
    if not fitting or not spill:
        return ()
    warm, cold = max(fitting), min(rate for _, rate in spill)
    if not 0 < cold < warm:
        return ()
    points = [(1.0, 1.0)]
    fraction = 1.0
    for size, rate in spill:
        fraction = min(fraction, (1 / cold - 1 / rate) / (1 / cold - 1 / warm))
        points.append((size / capacity, fraction))
    return tuple(points)


def primitive_working_set_sweep(table: Any) -> Mapping[int | str, float]:
    """Read the working-set sweep from the declared primitive table."""
    cell = getattr(table, "cell", None)
    if cell is None:
        return {}
    sweep: Mapping[int | str, float] = cell("P05_l2_residency_reuse").parameters.get("ws_sweep_bytes_per_s", {})
    return sweep


def _access_plan(plan: AddressRunPlan, spec: MemoryAccessSpec | None) -> AddressRunPlan:
    if spec is None or (
        not plan._default_sector and plan.l2_line_bytes and plan.l2_sector_bytes
        and plan.hbm_burst_granule_bytes
        and (plan.hbm_scatter_row_open_bytes or not getattr(spec, "hbm_scatter_row_open_bytes", 0))
    ):
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
    cache_policy: CachePolicy = _NORMAL_CACHE_POLICY
    line_footprint_bytes: int | None = None
    resident_intervals_mut: list[tuple[int, int]] | None = None
    resident_scale_intervals: tuple[tuple[int, int], ...] = ()
    resident_bands: Any = None

    def coverage_snapshot(self) -> SectorCoverage:
        """Return an immutable ``SectorCoverage`` snapshot for external inspection."""
        if self.resident_bands is not None:
            return SectorCoverage(
                intervals=tuple(self.resident_bands.to_intervals()),
                scale_intervals=self.resident_scale_intervals,
                _canonical=True,
                _bytes=self.resident_bytes,
            )
        if self.resident_intervals_mut is None:
            return self.resident_coverage
        return SectorCoverage(
            intervals=tuple(self.resident_intervals_mut),
            scale_intervals=self.resident_scale_intervals,
            _canonical=True,
            _bytes=self.resident_bytes,
        )


def _materialize_entry_bands(entry: _ResidentL2Entry) -> None:
    if entry.resident_bands is not None:
        entry.resident_intervals_mut = entry.resident_bands.to_intervals()
        entry.resident_bands = None


def _payload_only(cov: SectorCoverage) -> SectorCoverage:
    if not cov.scale_intervals:
        return cov
    return SectorCoverage(cov.intervals, cov.periodic, _canonical=cov._canonical)


def _cov_payload_bytes(cov: SectorCoverage) -> int:
    if not cov.intervals and not cov.periodic:
        return 0
    return _merged_bytes(_payload_only(cov))


def _cov_bytes(cov: SectorCoverage) -> int:
    return _cov_payload_bytes(cov) + _measure_byte_intervals(cov.scale_intervals)


def _cov_overlap(left: SectorCoverage, right: SectorCoverage) -> int:
    if left.empty or right.empty:
        return 0
    from ._periodic_stream import _overlap_canonical_intervals
    scale_ov = (
        _overlap_canonical_intervals(left.scale_intervals, right.scale_intervals)
        if left.scale_intervals and right.scale_intervals else 0
    )
    if (not left.intervals and not left.periodic) or (not right.intervals and not right.periodic):
        return scale_ov
    return _coverage_overlap(_payload_only(left), _payload_only(right)) + scale_ov


def _entry_payload_overlap_bytes(req_cov: SectorCoverage, entry: _ResidentL2Entry) -> int:
    """Compute exact payload byte overlap without copying the mutable interval list."""
    scale_b = _measure_byte_intervals(entry.resident_scale_intervals)
    if (not req_cov.intervals and not req_cov.periodic) or entry.resident_bytes <= scale_b:
        return 0
    if entry.resident_bands is not None and not req_cov.periodic:
        return sum(entry.resident_bands.overlap_interval(lo, hi) for lo, hi in req_cov.intervals)
    if entry.resident_intervals_mut is not None and not req_cov.periodic:
        from ._periodic_stream import _overlap_canonical_intervals
        return _overlap_canonical_intervals(
            req_cov.intervals, cast(tuple[tuple[int, int], ...], entry.resident_intervals_mut),
        )
    return _coverage_overlap(_payload_only(req_cov), _payload_only(entry.coverage_snapshot()))


def _entry_interval_overlap(lo: int, hi: int, entry: _ResidentL2Entry) -> int:
    if hi <= lo:
        return 0
    if entry.resident_bands is not None:
        return entry.resident_bands.overlap_interval(lo, hi)
    if entry.resident_intervals_mut is not None:
        from ._periodic_stream import _overlap_canonical_intervals
        return _overlap_canonical_intervals(
            ((lo, hi),), cast(tuple[tuple[int, int], ...], entry.resident_intervals_mut),
        )
    if not entry.resident_coverage.periodic:
        from ._periodic_stream import _overlap_canonical_intervals
        return _overlap_canonical_intervals(((lo, hi),), entry.resident_coverage.intervals)
    return _coverage_overlap(
        SectorCoverage(intervals=((lo, hi),), _canonical=True),
        _payload_only(entry.resident_coverage),
    )


def _entry_overlap_bytes(req_cov: SectorCoverage, entry: _ResidentL2Entry) -> int:
    """Compute exact byte overlap against ``entry`` across payload and scale planes."""
    if req_cov.empty or entry.resident_bytes <= 0:
        return 0
    from ._periodic_stream import _overlap_canonical_intervals
    scale_ov = (
        _overlap_canonical_intervals(req_cov.scale_intervals, entry.resident_scale_intervals)
        if req_cov.scale_intervals and entry.resident_scale_intervals else 0
    )
    return _entry_payload_overlap_bytes(req_cov, entry) + scale_ov


def _materialize_runs(runs: Iterable[AddressRun]) -> tuple[tuple[int, int], ...]:
    ivs: list[tuple[int, int]] = []
    for run in runs:
        if run.rows <= 0 or run.row_bytes <= 0:
            continue
        if run.rows == 1 or run.stride_bytes <= run.row_bytes:
            ivs.append((run.base_byte, _run_end(run)))
        else:
            base, stride, width = run.base_byte, run.stride_bytes, run.row_bytes
            ivs.extend((addr, addr + width) for addr in range(base, base + run.rows * stride, stride))
    return _merge_byte_intervals(ivs)


def _single_span_hit_bytes(
    base: int, logical_len: int, aligned_len: int, sec: int, align: int,
    interval_overlap_fn: Callable[[int, int], int],
) -> int:
    s_lo = (base // sec) * sec
    s_hi = _align_up(base + logical_len, sec)
    sec_ov = interval_overlap_fn(s_lo, s_hi)
    if sec_ov <= 0:
        return 0
    if sec_ov >= s_hi - s_lo:
        return aligned_len
    if base % align == 0:
        return min(aligned_len, interval_overlap_fn(base, base + aligned_len))
    log_ov = interval_overlap_fn(base, base + logical_len)
    return min(aligned_len, int(round(aligned_len * (float(log_ov) / float(logical_len)))))


def _run_hit_bytes(
    run: AddressRun, rows_eff: int, run_aligned: int, sec: int, align: int,
    overlap_fn: Callable[[SectorCoverage], int],
    interval_overlap_fn: Callable[[int, int], int],
) -> int:
    run_cov = AddressRunPlan((run,), sector_bytes=sec).sector_coverage()
    run_req = _cov_payload_bytes(run_cov)
    if run_req <= 0:
        return 0
    run_ov = overlap_fn(run_cov)
    if run_ov <= 0:
        return 0
    if run_ov >= run_req:
        return run_aligned
    if run.stride_bytes == run.row_bytes:
        return _single_span_hit_bytes(
            run.base_byte, rows_eff * run.row_bytes, run_aligned, sec, align, interval_overlap_fn,
        )
    row_align = _align_up(run.row_bytes, align)
    if run.stride_bytes == 0:
        return rows_eff * _single_span_hit_bytes(
            run.base_byte, run.row_bytes, row_align, sec, align, interval_overlap_fn,
        )
    if rows_eff <= _MAX_EXPLICIT_SECTOR_INTERVALS:
        hit_sum = 0
        for step in range(rows_eff):
            addr = run.base_byte + step * run.stride_bytes
            hit_sum += _single_span_hit_bytes(
                addr, run.row_bytes, row_align, sec, align, interval_overlap_fn,
            )
        return hit_sum
    return min(run_aligned, int(round(run_aligned * (float(run_ov) / float(run_req)))))


def _plan_hit_bytes(
    plan: AddressRunPlan,
    overlap_fn: Callable[[SectorCoverage], int],
    interval_overlap_fn: Callable[[int, int], int],
    resident_scale_intervals: tuple[tuple[int, int], ...],
    aligned_bytes: int,
) -> int:
    """Weighted aligned copy hit bytes across per-row/run payload spans and disjoint scale sectors."""
    seen_aliases: set[str] = set()
    plan_aligned = 0
    plan_hit = 0
    sec = _plan_sector_bytes(plan)
    align = plan.alignment_bytes
    for run in plan.runs:
        if run.rows == 0 or run.row_bytes == 0:
            continue
        if run.proven_alias_group is not None:
            if run.proven_alias_group in seen_aliases:
                continue
            seen_aliases.add(run.proven_alias_group)
            rows_eff = 1 if run.stride_bytes == 0 else run.rows
        else:
            rows_eff = run.rows
        run_aligned = (
            _align_up(rows_eff * run.row_bytes, align)
            if run.stride_bytes == run.row_bytes
            else rows_eff * _align_up(run.row_bytes, align)
        )
        plan_aligned += run_aligned
        plan_hit += _run_hit_bytes(
            run, rows_eff, run_aligned, sec, align, overlap_fn, interval_overlap_fn,
        )
    scale_span = plan.scale_span() if plan_aligned > 0 else None
    if scale_span is not None:
        scale_lo, scale_hi = scale_span
        scale_aligned = _align_up(plan.scale_bytes, align)
        plan_aligned += scale_aligned
        if resident_scale_intervals:
            from ._periodic_stream import _overlap_canonical_intervals
            plan_hit += _single_span_hit_bytes(
                scale_lo, scale_hi - scale_lo, scale_aligned, sec, align,
                lambda lo, hi: _overlap_canonical_intervals(((lo, hi),), resident_scale_intervals),
            )
    if plan_aligned <= 0:
        return 0
    if aligned_bytes == plan_aligned:
        return plan_hit
    return min(aligned_bytes, int(round(aligned_bytes * (float(plan_hit) / float(plan_aligned)))))


class AdmissionMemoryView:
    """Track shared L2 with strict LRU or an optional smooth spill curve.

    Curve points are (working set / effective capacity, hit fraction).
    ``ws_sweep`` accepts P05 working-set sizes and sustained bytes per second.
    Per-die access specs and sweeps override the scalar defaults.
    """

    def __init__(
        self,
        l2_capacity_by_die: Mapping[int, int],
        *,
        l2_buffer_units_by_die: Mapping[int, int] | None = None,
        granule_bytes: int | None = None,
        memory_access: MemoryAccessSpec | None = None,
        memory_access_by_die: Mapping[int, MemoryAccessSpec | None] | None = None,
        smooth_knee: bool = False,
        knee_ratio: float = 2,
        knee_curve: Sequence[tuple[float, float]] = (),
        spill_curve: Sequence[tuple[float, float]] = (),
        ws_sweep: Mapping[int | str, float] | None = None,
        ws_sweep_by_die: Mapping[int, Mapping[int | str, float]] | None = None,
        arch_id: str | None = None,
    ) -> None:
        self._memory_access_default = memory_access
        self._memory_access_by_die = dict(memory_access_by_die or {})
        self.arch_id = arch_id
        if not math.isfinite(knee_ratio) or knee_ratio <= 1:
            raise ModelingError(f"l2_knee_ratio_invalid:{knee_ratio}")
        if knee_curve and spill_curve:
            raise ModelingError("l2_knee_curve_declared_twice")
        curve = tuple(knee_curve or spill_curve)
        if (any(not math.isfinite(x) or x <= 0 or not 0 <= y <= 1 for x, y in curve)
                or any(x0 >= x1 or y0 < y1 for (x0, y0), (x1, y1) in zip(curve, curve[1:]))):
            raise ModelingError("l2_knee_curve_invalid")
        fallback_curve = ((1.0, 1.0), (knee_ratio, 0.0)) if smooth_knee else ()
        self._knee_curves_by_die: dict[int, tuple[tuple[float, float], ...]] = {}
        self._granule_by_die: dict[int, int] = {}
        self._l2_capacity: dict[int, int] = {}
        self._reserved: dict[int, dict[str, int]] = {}
        self._active_streams: dict[int, set[str]] = {}
        self._active_keys: dict[int, set[ReuseKey]] = {}
        self._active_stream_counts: dict[int, int] = {}
        self._template_key_users: dict[ReuseKey, int] = {}
        self._resident: dict[ReuseKey, _ResidentL2Entry] = {}
        self._pending_writebacks: dict[int, list[DirtyWriteback]] = {}
        self._last_prior_coverage: SectorCoverage = SectorCoverage()
        self._rect_ledger_cache: dict[tuple[int, int, float, int, int, int], ByteLedger] = {}
        units_map = dict(l2_buffer_units_by_die or {})
        sweeps = ws_sweep_by_die or {}
        default_sweep = ws_sweep or {}
        for die, cap in l2_capacity_by_die.items():
            if type(die) is not int or die < 0 or type(cap) is not int or cap < 0:
                raise ModelingError(f"l2_capacity_invalid:{die!r}:{cap!r}")
            if units_map.get(die, 1) != 1:
                raise ModelingError("shared_l2_buffer_requires_one_die_owner")
            access = self.memory_access_for(die)
            self._granule_by_die[die] = (
                getattr(access, "l2_sector_bytes", DEFAULT_L2_GRANULE_BYTES)
                if granule_bytes is None else granule_bytes
            )
            self._knee_curves_by_die[die] = (
                curve or _p05_knee(sweeps.get(die, default_sweep), cap) or fallback_curve
            )
            self._l2_capacity[die] = cap
            self._reserved[die] = {}
            self._active_streams[die] = set()
            self._active_keys[die] = set()
            self._pending_writebacks[die] = []
        self._default_die = 0 if 0 in self._l2_capacity else next(iter(self._l2_capacity), 0)
        first_access = self.memory_access_for(self._default_die)
        self.memory_access = (
            first_access if all(self.memory_access_for(die) == first_access for die in self._l2_capacity)
            else memory_access
        )
        self.granule_bytes = (
            getattr(self.memory_access, "l2_sector_bytes", DEFAULT_L2_GRANULE_BYTES)
            if granule_bytes is None else granule_bytes
        )
        if type(self.granule_bytes) is not int or self.granule_bytes < 1:
            raise ModelingError(f"l2_granule_invalid:{self.granule_bytes!r}")
        self._knee_curve = next(filter(None, self._knee_curves_by_die.values()), ())

    def has_die(self, die: int) -> bool:
        """Return True when ``die`` is declared in this admission view."""
        return die in self._l2_capacity

    def memory_access_for(self, die: int) -> MemoryAccessSpec | None:
        return self._memory_access_by_die.get(die, self._memory_access_default)

    def active_stream_count(self, die: int) -> int:
        cached = self._active_stream_counts.get(die)
        if cached is not None:
            return cached
        active_s = self._active_streams.get(die)
        active_k = self._active_keys.get(die)
        if not active_s:
            if not active_k:
                count = 0
            elif len(active_k) == 1:
                count = 1
            else:
                count = len({k.reuse_stream for k in active_k})
        elif not active_k:
            count = len(active_s)
        else:
            streams = set(active_s)
            streams.update(k.reuse_stream for k in active_k)
            count = len(streams)
        self._active_stream_counts[die] = count
        return count

    def residency_fingerprint(self, die: int | None = None) -> str:
        """Fingerprint the admission state, including the order of LRU entries."""
        digest = sha256()

        def record(*fields: object) -> None:
            digest.update(repr(fields).encode())
            digest.update(b"\n")

        def coverage(
            intervals: Sequence[tuple[int, int]],
            periodic: Sequence[PeriodicSectorRun],
            scale_intervals: Sequence[tuple[int, int]] = (),
        ) -> None:
            for interval in intervals:
                record("interval", interval)
            for run in periodic:
                record("periodic", run)
            for interval in scale_intervals:
                record("scale", interval)

        record("arch", self.arch_id)
        for key, users in sorted(self._template_key_users.items(), key=lambda pair: repr(pair[0])):
            if die is None or die == key.die:
                record("template", key, users)
        for rank, capacity in sorted(self._l2_capacity.items()):
            if die is not None and die != rank:
                continue
            record("die", rank, capacity, self._granule_by_die[rank],
                   self.memory_access_for(rank), self._knee_curves_by_die[rank],
                   sorted(self._reserved[rank].items()),
                   sorted(self._active_streams[rank]),
                   sorted(map(repr, self._active_keys[rank])))
            for writeback in self._pending_writebacks[rank]:
                record("writeback", writeback.key, writeback.bytes)
                coverage(writeback.coverage.intervals or writeback.ranges,
                         writeback.coverage.periodic, writeback.coverage.scale_intervals)
        for key, entry in self._resident.items():
            if die is not None and die != key.die:
                continue
            record("resident", key, entry.version, entry.cache_policy, entry.working_set_bytes, entry.resident_bytes,
                   entry.evicted, entry.dirty, entry.has_future_reader, entry.private_workspace,
                   entry.line_footprint_bytes)
            if entry.resident_bands is not None:
                coverage(entry.resident_bands.to_intervals(), (), entry.resident_scale_intervals)
            else:
                intervals = entry.resident_intervals_mut
                coverage(entry.resident_coverage.intervals if intervals is None else intervals,
                         entry.resident_coverage.periodic if intervals is None else (),
                         entry.resident_scale_intervals)
            record("dirty")
            coverage(entry.dirty_coverage.intervals or entry.dirty_ranges,
                     entry.dirty_coverage.periodic, entry.dirty_coverage.scale_intervals)
        return digest.hexdigest()

    def reserved_bytes(self, die: int) -> int:
        reserved = self._reserved.get(die)
        return sum(reserved.values()) if reserved else 0

    def available_l2_bytes(self, die: int) -> int:
        """Physical capacity minus live reservations, before stream sharing."""
        cap = self._l2_capacity.get(die)
        if cap is None:
            raise ModelingError(f"l2_die_undeclared:{die}")
        reserved = self._reserved.get(die)
        return cap if not reserved else max(0, cap - sum(reserved.values()))

    def effective_l2_bytes(self, die: int) -> int:
        """Return per-stream capacity in strict mode and the shared budget in smooth mode."""
        avail = self.available_l2_bytes(die)
        n_active = 1 if self._knee_curves_by_die[die] else max(1, self.active_stream_count(die))
        granule = self._granule_by_die[die]
        return granule * (avail // (granule * n_active))

    def _hit_fraction(self, working_set: int, capacity: int, *, die: int | None = None) -> float:
        if capacity <= 0:
            return 0.0
        curve = self._knee_curves_by_die.get(self._default_die if die is None else die, ())
        if not curve:
            return float(working_set <= capacity)
        pressure = working_set / capacity
        x0, y0 = curve[0]
        if pressure <= x0:
            return y0
        for x1, y1 in curve[1:]:
            if pressure <= x1:
                return y0 + (y1 - y0) * (pressure - x0) / (x1 - x0)
            x0, y0 = x1, y1
        return y0

    def _settle_eviction(self, entry: _ResidentL2Entry) -> None:
        if entry.dirty and entry.has_future_reader:
            wb_bytes = (
                _cov_bytes(entry.dirty_coverage)
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
        smooth = bool(self._knee_curves_by_die[die])
        oversized = False
        total_die = 0
        for entry in self._resident.values():
            if entry.key.die != die or entry.evicted:
                continue
            if (entry.working_set_bytes > eff
                    and (entry.dirty or not smooth
                         or not self._hit_fraction(entry.working_set_bytes, eff, die=die))):
                oversized = True
                break
            total_die += (
                min(entry.working_set_bytes, avail)
                if smooth and not entry.dirty else entry.working_set_bytes
            )
        if not oversized and total_die <= (avail if smooth else min(avail, eff)):
            return []

        victims: list[_ResidentL2Entry] = []
        evicted: set[ReuseKey] = set()

        def priority(entry: _ResidentL2Entry) -> tuple[int, bool]:
            return _CACHE_EVICTION_ORDER[entry.cache_policy], entry.key == protect_key

        def charge(entry: _ResidentL2Entry) -> int:
            return (min(entry.working_set_bytes, avail)
                    if smooth and not entry.dirty else entry.working_set_bytes)

        for entry in self._resident.values():
            if (entry.key.die == die and not entry.evicted and entry.working_set_bytes > eff
                    and (entry.dirty or not smooth
                         or not self._hit_fraction(entry.working_set_bytes, eff, die=die))):
                victims.append(entry)
                evicted.add(entry.key)
        if not smooth:
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

    def _evict_over_capacity_known(
        self, die: int, eff: int, avail: int, *, protect_key: ReuseKey | None = None,
    ) -> None:
        for victim in self._eviction_victims(die, eff, avail, protect_key=protect_key):
            self._settle_eviction(victim)

    def _evict_over_capacity(self, die: int, *, protect_key: ReuseKey | None = None) -> None:
        avail = self.available_l2_bytes(die)
        n_active = 1 if self._knee_curves_by_die[die] else max(1, self.active_stream_count(die))
        granule = self._granule_by_die[die]
        eff = granule * (avail // (granule * n_active))
        self._evict_over_capacity_known(die, eff, avail, protect_key=protect_key)

    def _cached_rect_ledger(
        self, logical: int, aligned: int, service: float, hit: int, miss: int, sectors: int,
    ) -> ByteLedger:
        cache = self._rect_ledger_cache
        lkey = (logical, aligned, service, hit, miss, sectors)
        ledger = cache.get(lkey)
        if ledger is None:
            ledger = ByteLedger(
                logical_tensor_bytes=logical, aligned_transfer_bytes=aligned,
                dram_bus_transaction_bytes=service, l2_hit_bytes=hit,
                dram_miss_copy_bytes=miss, dram_sector_bytes=sectors,
            )
            if len(cache) < _RECT_LEDGER_CACHE_LIMIT:
                cache[lkey] = ledger
        return ledger

    def _activate_key(self, key: ReuseKey) -> None:
        keys = self._active_keys[key.die]
        if key not in keys:
            keys.add(key)
            self._active_stream_counts.pop(key.die, None)
            self._evict_over_capacity(key.die)

    def pending_writebacks(self, die: int) -> tuple[DirtyWriteback, ...]:
        return tuple(self._pending_writebacks.get(die, ()))

    def drain_writebacks(self, die: int) -> tuple[DirtyWriteback, ...]:
        queue = self._pending_writebacks.get(die)
        if not queue:
            return ()
        items = tuple(queue)
        queue.clear()
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
                    and _cov_overlap(coverage, item_cov) > 0):
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
        self._active_stream_counts.pop(die, None)
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
        self._active_stream_counts.pop(die, None)
        return self.effective_l2_bytes(die)

    def release_key(self, key: ReuseKey) -> int:
        """Release one active ``ReuseKey`` lifetime without resurrecting evicted lines."""
        if key.die not in self._l2_capacity:
            raise ModelingError(f"l2_die_undeclared:{key.die}")
        self._active_keys[key.die].discard(key)
        self._active_stream_counts.pop(key.die, None)
        return self.effective_l2_bytes(key.die)

    def resolve_reuse_key(self, fallback: ReuseKey) -> ReuseKey:
        """Find the latest resident producer, then an active key for this allocation version."""
        identity = fallback.die, fallback.backing_allocation, fallback.tensor_version

        def matches(key: ReuseKey) -> bool:
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
            self._active_stream_counts.pop(key.die, None)
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
        req_b = _cov_bytes(req_cov)
        if req_b <= 0:
            return 0, aligned_bytes
        ov_b = _cov_overlap(req_cov, res_cov)
        if ov_b <= 0:
            return 0, aligned_bytes
        if ov_b >= req_b:
            return aligned_bytes, 0
        hit = min(aligned_bytes, int(round(aligned_bytes * (float(ov_b) / float(req_b)))))
        return hit, aligned_bytes - hit

    def _preview_effective_l2(self, key: ReuseKey) -> int:
        if self._knee_curves_by_die[key.die]:
            return self.effective_l2_bytes(key.die)
        streams = set(self._active_streams.get(key.die, ()))
        streams.update(k.reuse_stream for k in self._active_keys.get(key.die, ()))
        streams.add(key.reuse_stream)
        n_eff = max(1, len(streams))
        avail = max(0, self._l2_capacity[key.die] - self.reserved_bytes(key.die))
        granule = self._granule_by_die[key.die]
        return granule * (avail // (granule * n_eff))

    def access_geometry(
        self, pattern: MemoryAccessPattern, aligned_bytes: int | None = None,
    ) -> tuple[SectorCoverage, int]:
        """Return die-resolved sector coverage and L2 working-set footprint."""
        if aligned_bytes is None:
            aligned_bytes = pattern.runs.aligned_copy_bytes()
        return self._geometry(pattern, aligned_bytes)

    def _geometry(self, pattern: MemoryAccessPattern, aligned_bytes: int) -> tuple[SectorCoverage, int]:
        plan = _access_plan(pattern.runs, self.memory_access_for(pattern.die))
        # Track residency at DRAM sector granularity so L2 never credits hits
        # for bytes that were never billed to dram_sector_bytes, while charging
        # capacity footprint at the SKU L2 line/sector allocation granule.
        coverage = plan.sector_coverage()
        alloc_granule = plan.l2_line_bytes or plan.l2_sector_bytes
        if alloc_granule and alloc_granule != plan.sector_bytes:
            payload_span = (
                plan.runs[0].sector_union_bytes(alloc_granule) if len(plan.runs) == 1
                else _cov_payload_bytes(replace(plan, sector_bytes=alloc_granule, scale_bytes=0).sector_coverage())
            )
            scale_alloc = _measure_byte_intervals(plan.scale_sector_intervals(alloc_granule))
            span = payload_span + scale_alloc
        else:
            span = _cov_bytes(coverage)
        reuse_distance = pattern.reuse_distance_bytes
        base_ws = span if not coverage.empty else aligned_bytes
        return coverage, (base_ws if reuse_distance is None else max(reuse_distance, span))

    @staticmethod
    def _footprint(coverage: SectorCoverage, line_bytes: int | None) -> int:
        if line_bytes is None:
            return _cov_bytes(coverage)
        from ._periodic_stream import _aligned_intervals_bytes
        scale_fp = (
            _aligned_intervals_bytes(coverage.scale_intervals, line_bytes)
            if coverage.scale_intervals else 0
        )
        if not coverage.periodic:
            return _aligned_intervals_bytes(coverage.intervals, line_bytes) + scale_fp
        lines = AddressRunPlan(tuple(_coverage_runs(coverage)), sector_bytes=line_bytes)
        return _cov_payload_bytes(lines.sector_coverage()) + scale_fp

    def _policy(self, pattern: MemoryAccessPattern) -> CachePolicy:
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

    def _existing(
        self, pattern: MemoryAccessPattern, key: ReuseKey, eff: int, *, preview: bool,
    ) -> _ResidentL2Entry | None:
        entry = self._resident.get(key)
        if entry is None or entry.evicted or entry.version != pattern.version:
            return None
        if preview and key not in self._active_keys[key.die]:
            avail = max(0, self._l2_capacity[key.die] - self.reserved_bytes(key.die))
            if any(entry is victim for victim in self._eviction_victims(key.die, eff, avail)):
                return None
        return entry

    def _classify_access_with_overlap(
        self,
        pattern: MemoryAccessPattern,
        aligned_bytes: int,
        eff: int,
        coverage: SectorCoverage,
        working_set: int,
        entry: _ResidentL2Entry | None,
    ) -> tuple[int, int, int | None]:
        """Return ``(hit, miss, overlap_bytes_or_None)`` using one overlap scan."""
        if eff <= 0:
            return 0, aligned_bytes, None
        if pattern.direction == "read":
            if entry is None:
                return 0, aligned_bytes, None
            live_entry = entry
            fraction = self._hit_fraction(max(working_set, live_entry.working_set_bytes), eff, die=pattern.die)
            if not fraction:
                return 0, aligned_bytes, None
            plan = _access_plan(pattern.runs, self.memory_access_for(pattern.die))
            ov_cached: int | None = None
            if len(plan.runs) == 1 and not coverage.scale_intervals:
                req_b = _cov_payload_bytes(coverage)
                ov_b = 0 if req_b <= 0 else _entry_payload_overlap_bytes(coverage, live_entry)
                if ov_b <= 0:
                    return 0, aligned_bytes, 0
                if ov_b >= req_b:
                    hit = aligned_bytes if fraction == 1.0 else round(aligned_bytes * fraction)
                    return hit, aligned_bytes - hit, ov_b
                ov_cached = ov_b
            hit = _plan_hit_bytes(
                plan,
                lambda rc: _entry_payload_overlap_bytes(rc, live_entry),
                lambda lo, hi: _entry_interval_overlap(lo, hi, live_entry),
                live_entry.resident_scale_intervals,
                aligned_bytes,
            )
            if fraction != 1.0:
                hit = round(hit * fraction)
            hit = max(0, min(aligned_bytes, hit))
            return hit, aligned_bytes - hit, ov_cached
        if (pattern.visibility.private_workspace and not pattern.visibility.live_out
                and working_set <= eff
                and not (self.arch_id is None and pattern.cache_policy == CACHE_POLICY_LAST_USE)):
            return aligned_bytes, 0, None
        return 0, aligned_bytes, None

    def _classify_access(
        self,
        pattern: MemoryAccessPattern,
        aligned_bytes: int,
        eff: int,
        coverage: SectorCoverage,
        working_set: int,
        entry: _ResidentL2Entry | None,
    ) -> tuple[int, int]:
        """Share the hit/miss decision between preview and commit."""
        hit, miss, _ = self._classify_access_with_overlap(
            pattern, aligned_bytes, eff, coverage, working_set, entry,
        )
        return hit, miss

    def preview_access(
        self, pattern: MemoryAccessPattern, aligned_bytes: int | None = None,
    ) -> tuple[int, int]:
        """Pure read-only addressed hit and miss copy bytes."""
        if aligned_bytes is None:
            aligned_bytes = pattern.runs.aligned_copy_bytes()
        key = self._key(pattern, aligned_bytes)
        if key is None:
            return 0, max(0, aligned_bytes)
        eff = self._preview_effective_l2(key)
        if eff <= 0:
            return 0, aligned_bytes
        coverage, working_set = self._geometry(pattern, aligned_bytes)
        entry = self._existing(pattern, key, eff, preview=True)
        return self._classify_access(pattern, aligned_bytes, eff, coverage, working_set, entry)

    def resident_coverage_for(
        self, pattern: MemoryAccessPattern, aligned_bytes: int | None = None,
    ) -> SectorCoverage:
        """Return eligible address coverage before applying any statistical spill."""
        if aligned_bytes is None:
            aligned_bytes = pattern.runs.aligned_copy_bytes()
        key = self._key(pattern, aligned_bytes)
        if key is not None and pattern.direction == "read":
            eff = self._preview_effective_l2(key)
            if eff <= 0:
                return SectorCoverage()
            existing = self._existing(pattern, key, eff, preview=True)
            if existing is not None:
                coverage, working_set = self._geometry(pattern, aligned_bytes)
                if not coverage.empty and self._hit_fraction(
                    max(working_set, existing.working_set_bytes), eff, die=key.die,
                ):
                    return existing.coverage_snapshot()
        return SectorCoverage()

    def resident_intervals_for(
        self, pattern: MemoryAccessPattern, aligned_bytes: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Return currently resident L2 sector intervals applicable to ``pattern``."""
        if aligned_bytes is None:
            aligned_bytes = pattern.runs.aligned_copy_bytes()
        res_cov = self.resident_coverage_for(pattern, aligned_bytes)
        if not res_cov.intervals and not res_cov.periodic:
            return ()
        plan = _access_plan(pattern.runs, self.memory_access_for(pattern.die))
        req_cov = plan.sector_coverage()
        if not req_cov.intervals and not req_cov.periodic:
            return ()
        if not req_cov.periodic and not res_cov.periodic:
            return _intersect_byte_intervals(req_cov.intervals, res_cov.intervals)
        warm = (
            run for is_warm, run in _residency_runs(
                tuple(_coverage_runs(req_cov)), tuple(_coverage_runs(res_cov)),
            ) if is_warm
        )
        return _materialize_runs(warm)

    def miss_intervals(
        self, pattern: MemoryAccessPattern, aligned_bytes: int | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Return exact missed sector byte intervals for ``pattern`` against current L2."""
        if aligned_bytes is None:
            aligned_bytes = pattern.runs.aligned_copy_bytes()
        plan = _access_plan(pattern.runs, self.memory_access_for(pattern.die))
        req_cov = plan.sector_coverage()
        if not req_cov.intervals and not req_cov.periodic:
            return ()
        res_cov = self.resident_coverage_for(pattern, aligned_bytes)
        if not req_cov.periodic and not res_cov.periodic:
            return _subtract_byte_intervals(req_cov.intervals, res_cov.intervals) if res_cov.intervals else req_cov.intervals
        if not res_cov.intervals and not res_cov.periodic:
            return _materialize_runs(_coverage_runs(req_cov))
        cold = (
            run for is_warm, run in _residency_runs(
                tuple(_coverage_runs(req_cov)), tuple(_coverage_runs(res_cov)),
            ) if not is_warm
        )
        return _materialize_runs(cold)

    def admit_access(
        self, pattern: MemoryAccessPattern, aligned_bytes: int | None = None,
    ) -> tuple[int, int]:
        """Classify copy bytes and update exact address coverage and LRU state."""
        if aligned_bytes is None:
            aligned_bytes = pattern.runs.aligned_copy_bytes()
        hit, miss, _ = self._admit_access_with_prior(pattern, aligned_bytes)
        return hit, miss

    def _merge_entry_inplace(
        self, existing: _ResidentL2Entry, pattern: MemoryAccessPattern,
        coverage: SectorCoverage, working_set: int, eff: int,
        read: bool, dirty: bool, dirty_cov: SectorCoverage,
        resident_span: int, overlap: int,
    ) -> bool:
        """Attempt exact merge into ``existing``; returns True if updated in-place."""
        if eff <= 0:
            return False
        _materialize_entry_bands(existing)
        from ._periodic_stream import _aligned_intervals_bytes, _merge_canonical_pairs
        merged_scale_ivs = _merge_canonical_pairs(existing.resident_scale_intervals, coverage.scale_intervals)
        merged_scale_bytes = _measure_byte_intervals(merged_scale_ivs)
        merged_span = existing.resident_bytes + resident_span - overlap
        spec = self.memory_access_for(pattern.die)
        line = (
            pattern.runs.l2_line_bytes or getattr(spec, "l2_line_bytes", None)
            or pattern.runs.l2_sector_bytes or getattr(spec, "l2_sector_bytes", None)
        )
        if line == pattern.runs.sector_bytes:
            line = None
        scale_fp = (
            _aligned_intervals_bytes(merged_scale_ivs, line)
            if (line and merged_scale_ivs) else merged_scale_bytes
        )
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
            payload_fp = (merged_span - merged_scale_bytes) if new_fp is None else new_fp
            footprint = payload_fp + scale_fp
            merged_ws = max(existing.working_set_bytes, working_set, footprint)
            if not (self._hit_fraction(merged_ws, eff, die=pattern.die)
                    if read and not existing.dirty else merged_ws <= eff):
                return False
            ivs[lo_idx:hi_idx] = merged_mid
            existing.working_set_bytes = merged_ws
            existing.resident_bytes = merged_span
            existing.resident_scale_intervals = merged_scale_ivs
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
        if not (self._hit_fraction(merged_ws, eff, die=pattern.die)
                if read and not existing.dirty else merged_ws <= eff):
            return False
        existing.resident_intervals_mut = None
        existing.resident_coverage = base_cov.union(coverage) if merged is None else merged
        existing.working_set_bytes = merged_ws
        existing.resident_bytes = merged_span
        existing.resident_scale_intervals = merged_scale_ivs
        existing.line_footprint_bytes = (footprint - scale_fp) if line else None
        existing.cache_policy = pattern.cache_policy
        if dirty:
            existing.dirty = True
            existing.dirty_coverage = existing.dirty_coverage.union(dirty_cov)
            existing.dirty_ranges = existing.dirty_coverage.intervals
        self._touch_resident(existing)
        return True

    def _try_merge_strided_band(
        self, existing: _ResidentL2Entry, die: int, eff: int, avail: int, key: ReuseKey,
        stride: int, line: int | None, r_lo: int, r_hi: int, c_lo: int, c_hi: int,
        working_set: int, merged_span: int, policy: CachePolicy,
    ) -> bool:
        """Attempt exact in-place strided-band merge; return True if merged into ``existing``."""
        eb = existing.resident_bands
        assert eb is not None
        if existing.dirty:
            if merged_span > eff:
                return False
        elif not self._hit_fraction(merged_span, eff, die=die):
            return False
        line_aligned = bool(line and stride % line == 0 and c_lo % line == 0 and c_hi % line == 0)
        if not line or (eb.line_aligned_to == line and line_aligned):
            payload_fp = merged_span
            merged_ws = max(existing.working_set_bytes, working_set, payload_fp)
            can_merge = (
                bool(self._hit_fraction(merged_ws, eff, die=die))
                if not existing.dirty else merged_ws <= eff
            )
            if not can_merge:
                return False
            eb.merge_inplace(r_lo, r_hi, c_lo, c_hi)
            if not line_aligned:
                eb.line_aligned_to = None
        elif stride % line == 0:
            lo_idx, hi_idx, mid, payload_fp = eb.plan_merge_slice(r_lo, r_hi, c_lo, c_hi, line)
            merged_ws = max(existing.working_set_bytes, working_set, payload_fp)
            can_merge = (
                bool(self._hit_fraction(merged_ws, eff, die=die))
                if not existing.dirty else merged_ws <= eff
            )
            if not can_merge:
                return False
            eb.bands[lo_idx:hi_idx] = mid
            eb.line_aligned_to = None
            eb.aligned_cache = (line, payload_fp)
        else:
            from ._periodic_stream import _strided_bands_aligned_bytes

            new_bands = eb.planned_merge(r_lo, r_hi, c_lo, c_hi)
            payload_fp = _strided_bands_aligned_bytes(new_bands, stride, line)
            merged_ws = max(existing.working_set_bytes, working_set, payload_fp)
            can_merge = (
                bool(self._hit_fraction(merged_ws, eff, die=die))
                if not existing.dirty else merged_ws <= eff
            )
            if not can_merge:
                return False
            eb.bands = new_bands
            eb.line_aligned_to = None
            eb.aligned_cache = None
        existing.working_set_bytes = merged_ws
        existing.resident_bytes = merged_span
        existing.line_footprint_bytes = payload_fp if line else None
        existing.cache_policy = policy
        self._touch_resident(existing)
        self._evict_over_capacity_known(die, eff, avail, protect_key=key)
        return True

    def _admit_strided_band_read(
        self, pattern: MemoryAccessPattern, aligned_bytes: int, key: ReuseKey, eff: int,
    ) -> tuple[int, int, SectorCoverage] | None:
        """O(bands) exact read admission for single-run strided 2D row bands."""
        spec = self.memory_access_for(pattern.die)
        plan = _access_plan(pattern.runs, spec)
        if len(plan.runs) != 1 or plan.scale_bytes:
            return None
        sec = _plan_sector_bytes(plan)
        line = plan.l2_line_bytes or plan.l2_sector_bytes
        if line == sec:
            line = None
        from ._periodic_stream import _StridedBandSet, _band_aligned_bytes, _strided_band_from_run
        band = _strided_band_from_run(plan.runs[0], sec, line, _MAX_EXPLICIT_SECTOR_INTERVALS)
        if band is None:
            return None
        stride, _, r_lo, r_hi, c_lo, c_hi = band
        n_rows = r_hi - r_lo
        existing = self._existing(pattern, key, eff, preview=False)
        if existing is None:
            if n_rows <= 1:
                return None
        else:
            eb = existing.resident_bands
            if eb is None or eb.stride != stride or eb.sector_bytes != sec or existing.resident_scale_intervals:
                return None
        resident_span = n_rows * (c_hi - c_lo)
        line_aligned = bool(line and stride % line == 0 and c_lo % line == 0 and c_hi % line == 0)
        span = resident_span if (not line or line_aligned) else _band_aligned_bytes(r_lo, r_hi, stride, ((c_lo, c_hi),), line)
        rd = pattern.reuse_distance_bytes
        working_set = span if rd is None else (rd if rd > span else span)
        hit, miss, ov_b = 0, aligned_bytes, 0
        prior_out = SectorCoverage()
        if eff > 0 and existing is not None:
            ws_eff = working_set if working_set >= existing.working_set_bytes else existing.working_set_bytes
            fraction = self._hit_fraction(ws_eff, eff, die=pattern.die)
            if fraction:
                ov_b = existing.resident_bands.overlap_rect(r_lo, r_hi, c_lo, c_hi)
                if ov_b >= resident_span:
                    hit = aligned_bytes if fraction == 1.0 else round(aligned_bytes * fraction)
                    miss = aligned_bytes - hit
                elif ov_b > 0:
                    hit = _plan_hit_bytes(
                        plan, lambda _rc: ov_b, existing.resident_bands.overlap_interval, (), aligned_bytes,
                    )
                    if fraction != 1.0:
                        hit = round(hit * fraction)
                    hit = max(0, min(aligned_bytes, hit))
                    miss = aligned_bytes - hit
                if hit > 0 and miss > 0:
                    prior_out = existing.coverage_snapshot()
        self._last_prior_coverage = prior_out
        legacy_last = self.arch_id is None and pattern.cache_policy == CACHE_POLICY_LAST_USE
        if legacy_last and existing is not None and eff > 0 and working_set <= eff:
            if not existing.dirty:
                existing.evicted = True
            return hit, miss, prior_out
        previous = self._resident.get(key)
        if eff <= 0 or legacy_last or not self._hit_fraction(working_set, eff, die=key.die):
            if previous is not None and not previous.evicted:
                self._settle_eviction(previous)
            return hit, miss, prior_out
        avail = self.available_l2_bytes(key.die)
        if existing is not None:
            if not miss:
                existing.cache_policy = pattern.cache_policy
                self._touch_resident(existing)
                return hit, miss, prior_out
            if self._try_merge_strided_band(
                existing, key.die, eff, avail, key, stride, line,
                r_lo, r_hi, c_lo, c_hi, working_set,
                existing.resident_bytes + resident_span - ov_b, pattern.cache_policy,
            ):
                return hit, miss, prior_out
        if previous is not None and not previous.evicted:
            self._settle_eviction(previous)
        if n_rows > 1:
            fresh = _ResidentL2Entry(
                key=key, working_set_bytes=working_set, version=pattern.version,
                dirty=False, private_workspace=pattern.visibility.private_workspace,
                resident_coverage=SectorCoverage(), resident_bytes=resident_span,
                cache_policy=pattern.cache_policy, line_footprint_bytes=None,
                resident_bands=_StridedBandSet(
                    stride, sec, [(r_lo, r_hi, ((c_lo, c_hi),))],
                    line if line_aligned else None,
                    (line, span) if (line and stride % line == 0) else None,
                ),
            )
        else:
            iv = (r_lo * stride + c_lo, r_lo * stride + c_hi)
            fresh = _ResidentL2Entry(
                key=key, working_set_bytes=working_set, version=pattern.version,
                dirty=False, private_workspace=pattern.visibility.private_workspace,
                resident_coverage=SectorCoverage(intervals=(iv,), _canonical=True, _bytes=resident_span),
                resident_bytes=resident_span, cache_policy=pattern.cache_policy,
                line_footprint_bytes=None,
            )
        self._touch_resident(fresh)
        self._evict_over_capacity_known(key.die, eff, avail, protect_key=key)
        return hit, miss, prior_out

    def admit_rect_read_ledger(
        self,
        die: int,
        key: ReuseKey | None,
        version: int,
        base_byte: int,
        rows: int,
        row_bytes: int,
        row_stride: int,
        *,
        alignment_bytes: int,
        sector_bytes: int,
        l2_line_bytes: int | None,
        l2_sector_bytes: int | None,
        burst_granule: int,
        burst_rules: Sequence[BurstTurnRule] = (),
    ) -> ByteLedger | None:
        """Zero-allocation exact ``ByteLedger`` admission for single-run RowMajor reads."""
        if rows <= 0 or row_bytes <= 0 or row_stride < row_bytes or rows > _MAX_EXPLICIT_SECTOR_INTERVALS:
            return None
        logical_bytes = rows * row_bytes
        if row_stride == row_bytes:
            aligned_bytes = _align_up(logical_bytes, alignment_bytes)
        else:
            aligned_bytes = rows * _align_up(row_bytes, alignment_bytes)
        if aligned_bytes == 0:
            return self._cached_rect_ledger(logical_bytes, 0, 0.0, 0, 0, 0)
        sec = sector_bytes
        line = l2_line_bytes or l2_sector_bytes
        if line == sec:
            line = None
        if line is not None and line < sec:
            return None
        is_contiguous = rows == 1 or row_stride == row_bytes
        line_aligned = False
        if is_contiguous:
            s_lo = (base_byte // sec) * sec
            s_hi = ((base_byte + logical_bytes + sec - 1) // sec) * sec
            resident_span = s_hi - s_lo
            span = ((((s_hi + line - 1) // line) - (s_lo // line)) * line) if line else resident_span
            stride = r_lo = r_hi = c_lo = c_hi = 0
        else:
            from ._periodic_stream import _band_aligned_bytes, _strided_band_from_ints

            band = _strided_band_from_ints(
                base_byte, rows, row_bytes, row_stride, sec, line, _MAX_EXPLICIT_SECTOR_INTERVALS,
            )
            if band is None:
                return None
            stride, _, r_lo, r_hi, c_lo, c_hi = band
            resident_span = rows * (c_hi - c_lo)
            line_aligned = bool(line and stride % line == 0 and c_lo % line == 0 and c_hi % line == 0)
            span = resident_span if (not line or line_aligned) else _band_aligned_bytes(
                r_lo, r_hi, stride, ((c_lo, c_hi),), line,
            )

        if key is None:
            service = _rect_dram_service_bytes(
                resident_span, rows, row_bytes, row_stride, burst_granule, burst_rules,
            )
            return self._cached_rect_ledger(logical_bytes, aligned_bytes, service, 0, aligned_bytes, resident_span)
        if key.die != die:
            raise ModelingError(f"reuse_key_die_mismatch:{key.die}!={die}")
        if die not in self._l2_capacity:
            raise ModelingError(f"l2_die_undeclared:{die}")
        if self.arch_id is not None and cache_policy_for_arch(self.arch_id, _NORMAL_CACHE_POLICY) == _L2_CACHE_DISABLE:
            return None
        self._activate_key(key)
        avail = self.available_l2_bytes(die)
        n_active = 1 if self._knee_curves_by_die[die] else max(1, self.active_stream_count(die))
        granule = self._granule_by_die[die]
        eff = granule * (avail // (granule * n_active))
        existing = self._resident.get(key)
        if existing is not None and (existing.evicted or existing.version != version):
            existing = None

        if is_contiguous:
            if (
                eff <= 0 or existing is None or existing.resident_bands is not None
                or existing.resident_scale_intervals
            ):
                return None
            ivs: Sequence[tuple[int, int]] | None = existing.resident_intervals_mut
            if ivs is None:
                if existing.resident_coverage.periodic:
                    return None
                ivs = existing.resident_coverage.intervals
            if len(ivs) != 1 or ivs[0][0] > s_lo or s_hi > ivs[0][1]:
                return None
            ws_eff = span if span >= existing.working_set_bytes else existing.working_set_bytes
            if self._hit_fraction(ws_eff, eff, die=die) != 1.0:
                return None
            self._last_prior_coverage = SectorCoverage()
            existing.cache_policy = _NORMAL_CACHE_POLICY
            self._touch_resident(existing)
            return self._cached_rect_ledger(logical_bytes, aligned_bytes, 0.0, aligned_bytes, 0, 0)

        if existing is not None:
            eb = existing.resident_bands
            if eb is None or eb.stride != stride or eb.sector_bytes != sec or existing.resident_scale_intervals:
                return None
            if eff > 0:
                ws_eff = span if span >= existing.working_set_bytes else existing.working_set_bytes
                fraction = self._hit_fraction(ws_eff, eff, die=die)
                if fraction:
                    ov_b = eb.overlap_rect(r_lo, r_hi, c_lo, c_hi)
                    if ov_b >= resident_span and fraction == 1.0:
                        self._last_prior_coverage = SectorCoverage()
                        existing.cache_policy = _NORMAL_CACHE_POLICY
                        self._touch_resident(existing)
                        return self._cached_rect_ledger(logical_bytes, aligned_bytes, 0.0, aligned_bytes, 0, 0)
                    if ov_b > 0:
                        return None
        self._last_prior_coverage = SectorCoverage()
        previous = self._resident.get(key)
        service = _rect_dram_service_bytes(
            resident_span, rows, row_bytes, row_stride, burst_granule, burst_rules,
        )
        miss_ledger = self._cached_rect_ledger(logical_bytes, aligned_bytes, service, 0, aligned_bytes, resident_span)
        if eff <= 0 or not self._hit_fraction(span, eff, die=die):
            if previous is not None and not previous.evicted:
                self._settle_eviction(previous)
            return miss_ledger
        if existing is not None and self._try_merge_strided_band(
            existing, die, eff, avail, key, stride, line,
            r_lo, r_hi, c_lo, c_hi, span, existing.resident_bytes + resident_span, _NORMAL_CACHE_POLICY,
        ):
            return miss_ledger
        if previous is not None and not previous.evicted:
            self._settle_eviction(previous)
        from ._periodic_stream import _StridedBandSet

        fresh = _ResidentL2Entry(
            key=key, working_set_bytes=span, version=version,
            dirty=False, private_workspace=False,
            resident_coverage=SectorCoverage(), resident_bytes=resident_span,
            cache_policy=_NORMAL_CACHE_POLICY, line_footprint_bytes=None,
            resident_bands=_StridedBandSet(
                stride, sec, [(r_lo, r_hi, ((c_lo, c_hi),))],
                line if line_aligned else None,
                (line, span) if (line and stride % line == 0) else None,
            ),
        )
        self._touch_resident(fresh)
        self._evict_over_capacity_known(die, eff, avail, protect_key=key)
        return miss_ledger

    def _admit_access_with_prior(
        self, pattern: MemoryAccessPattern, aligned_bytes: int,
    ) -> tuple[int, int, SectorCoverage]:
        """Single-pass admission returning ``(hit, miss, prior_resident_coverage)``."""
        key = self._key(pattern, aligned_bytes)
        if key is None:
            return 0, max(0, aligned_bytes), SectorCoverage()
        self._activate_key(key)
        eff = self.effective_l2_bytes(key.die)
        if pattern.direction == "read":
            fast = self._admit_strided_band_read(pattern, aligned_bytes, key, eff)
            if fast is not None:
                return fast
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
        if read and legacy_last and existing is not None and eff > 0 and working_set <= eff:
            if not existing.dirty:
                existing.evicted = True
            return hit, miss, prior_out

        previous = self._resident.get(key)
        if eff <= 0 or legacy_last or not (self._hit_fraction(working_set, eff, die=key.die) if read else working_set <= eff):
            if previous is not None and not previous.evicted:
                self._settle_eviction(previous)
            return hit, miss, prior_out

        dirty = not read and hit > 0
        dirty_cov = coverage if dirty else SectorCoverage()
        resident_span = _cov_bytes(coverage)
        if existing is not None and (read or dirty):
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
            line_footprint_bytes=None, resident_scale_intervals=coverage.scale_intervals,
        ))
        self._evict_over_capacity(key.die, protect_key=key)
        return hit, miss, prior_out


def _rect_dram_service_bytes(
    sectors: int, rows: int, row_bytes: int, row_stride: int,
    burst_granule: int, rules: Sequence[BurstTurnRule],
) -> float:
    granule = float(burst_granule) if row_stride > row_bytes else 0.0
    if rules:
        eff = burst_turnaround_efficiency(row_bytes, row_stride, rules)
        if granule <= 0.0:
            return sectors / eff
        if eff < 1.0:
            granule = max(granule, row_bytes / eff)
    if granule <= row_bytes:
        return float(sectors)
    eff_rows = rows if row_stride else 1
    return sectors * max(1.0, eff_rows * granule / sectors)


def _burst_service_bytes(plan: AddressRunPlan, sectors: int, run: AddressRun, rules: Sequence[BurstTurnRule],
                        *, width: int | None = None, stride: int | None = None) -> float:
    granule = 0.0
    if run.layout in {"scatter", "scattered", "gather", "indexed"}:
        granule = float(plan.hbm_scatter_row_open_bytes)
    if run.stride_bytes > run.row_bytes:
        granule = max(granule, float(plan.hbm_burst_granule_bytes))
    if rules:
        w = run.row_bytes if width is None else width
        s = run.stride_bytes if stride is None else stride
        eff = burst_turnaround_efficiency(w, s, rules)
        if granule <= 0.0:
            return sectors / eff
        if eff < 1.0:
            granule = max(granule, w / eff)
    if granule <= run.row_bytes:
        return float(sectors)
    # Sector fills may already pay for part or all of the service granule.
    wire = run.sector_union_bytes(_plan_sector_bytes(plan))
    rows = run.rows if run.stride_bytes else 1
    return sectors * max(1.0, rows * granule / wire)


def _canonical_dram_runs(
    plan: AddressRunPlan, burst_rules: Sequence[BurstTurnRule] = (),
) -> tuple[AddressRun, ...]:
    """Return deduplicated active runs in canonical burst-cost order independent of input order."""
    nonempty = [run for run in plan.runs if run.rows > 0 and run.row_bytes > 0]
    if len(nonempty) <= 1:
        return tuple(nonempty)
    # Deduplicate proven alias groups deterministically before cost ordering.
    nonempty.sort(
        key=lambda r: (
            r.base_byte, r.rows, r.row_bytes, r.stride_bytes, r.layout,
            r.proven_alias_group or "",
        )
    )
    deduped: list[AddressRun] = []
    seen_aliases: set[str] = set()
    for run in nonempty:
        if run.proven_alias_group is not None:
            if run.proven_alias_group in seen_aliases:
                continue
            seen_aliases.add(run.proven_alias_group)
        deduped.append(run)
    if len(deduped) <= 1:
        return tuple(deduped)

    sec = _plan_sector_bytes(plan)

    def _order_key(run: AddressRun) -> tuple[float, int, int, int, int, int, str]:
        wire = run.sector_union_bytes(sec)
        svc = _burst_service_bytes(plan, wire, run, burst_rules) if wire > 0 else 0.0
        ratio = (svc / float(wire)) if wire > 0 else 1.0
        return (ratio, -wire, run.base_byte, run.rows, run.row_bytes, run.stride_bytes, run.layout)

    deduped.sort(key=_order_key)
    return tuple(deduped)


def _dram_segments(
    plan: AddressRunPlan, covered: SectorCoverage,
    *, burst_rules: Sequence[BurstTurnRule] = (),
) -> Iterator[tuple[int, int, int, AddressRun | None]]:
    covered_payload = _payload_only(covered)
    partial = not covered_payload.empty
    sec = _plan_sector_bytes(plan)
    ordered = _canonical_dram_runs(plan, burst_rules)
    for run in ordered:
        if not partial and len(ordered) == 1:
            yield run.sector_union_bytes(sec), run.row_bytes, run.stride_bytes, run
            continue
        run_cov = AddressRunPlan((run,), sector_bytes=sec).sector_coverage()
        if not partial or run_cov.periodic:
            missed = (_cov_payload_bytes(run_cov) - _cov_overlap(run_cov, covered_payload)
                      if run_cov.periodic or covered_payload.periodic else
                      _measure_byte_intervals(_subtract_byte_intervals(run_cov.intervals, covered_payload.intervals)))
            if missed > 0:
                yield missed, run.row_bytes, run.stride_bytes, run
        else:
            intervals = (run_cov.intervals if covered_payload.periodic else
                         _subtract_byte_intervals(run_cov.intervals, covered_payload.intervals))
            for lo, hi in intervals:
                span = hi - lo
                if covered_payload.periodic:
                    span -= SectorCoverage(intervals=((lo, hi),)).overlap_bytes(covered_payload)
                if span > 0:
                    stride = run.stride_bytes
                    yield span, span, stride if stride == 0 or stride > run.row_bytes else span, run
        if len(ordered) > 1:
            covered_payload = covered_payload.union(run_cov)
    if ordered and plan.scale_bytes:
        req_scale = plan.scale_sector_intervals(sec)
        missed_scale = _measure_byte_intervals(
            _subtract_byte_intervals(req_scale, covered.scale_intervals)
            if covered.scale_intervals else req_scale
        )
        if missed_scale > 0:
            yield missed_scale, 0, 0, None


def _missed_dram_sector_and_service(
    plan: AddressRunPlan, resident_cov: SectorCoverage, burst_rules: Sequence[BurstTurnRule],
) -> tuple[int, float]:
    sector_total, service_total = 0, 0.0
    for span, width, stride, run in _dram_segments(plan, resident_cov, burst_rules=burst_rules):
        sector_total += span
        service_total += span if run is None else _burst_service_bytes(
            plan, span, run, burst_rules, width=width, stride=stride,
        )
    return sector_total, service_total


def admit_memory_ledger(
    pattern: MemoryAccessPattern,
    state: AdmissionMemoryView | None,
    *,
    bus_memory_access: MemoryAccessSpec | None = None,
    burst_rules: Sequence[BurstTurnRule] = (),
    commit: bool = False,
) -> tuple[ByteLedger, MemoryAccessPattern, SectorCoverage]:
    """Resolve and admit one ``MemoryAccessPattern`` into an exact ``ByteLedger``."""
    access = state.memory_access_for(pattern.die) if isinstance(state, AdmissionMemoryView) else None
    plan = _access_plan(pattern.runs, access)
    if plan is not pattern.runs:
        pattern = replace(pattern, runs=plan)
    plan = _access_plan(plan, bus_memory_access)
    aligned_bytes = plan.aligned_copy_bytes()
    if aligned_bytes == 0:
        return ByteLedger(
            logical_tensor_bytes=pattern.logical_bytes, aligned_transfer_bytes=0,
            dram_bus_transaction_bytes=0.0, l2_hit_bytes=0,
            dram_miss_copy_bytes=0, dram_sector_bytes=0,
        ), pattern, SectorCoverage()

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
            if isinstance(state, AdmissionMemoryView) and state._knee_curves_by_die.get(pattern.die):
                res_payload = _payload_only(res_cov)
                spatial_hits = _plan_hit_bytes(
                    plan,
                    lambda rc: _cov_overlap(rc, res_payload),
                    lambda lo, hi: _cov_overlap(SectorCoverage(intervals=((lo, hi),), _canonical=True), res_payload),
                    res_cov.scale_intervals,
                    aligned_bytes,
                )
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
    return ledger, pattern, res_cov


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

    ledger, pattern, _ = admit_memory_ledger(
        pattern, state, bus_memory_access=hbm_bus.memory_access,
        burst_rules=burst_rules, commit=commit,
    )
    aligned_bytes = ledger.aligned_transfer_bytes
    if aligned_bytes == 0:
        after = (pattern.visibility.producer_event,) if pattern.visibility.producer_event else ()
        return BoundAccess(
            ledger=ledger, constraints=(), deliveries=(), after=after,
            owner=pattern.owner, pipe_row_id=pipe_row_id, l2_row_id=l2_row_id,
            bind_config=bind_cfg,
        )
    hit_bytes = ledger.l2_hit_bytes
    miss_bytes = ledger.dram_miss_copy_bytes
    sector_bytes = ledger.dram_sector_bytes
    dram_service_bytes = ledger.dram_bus_transaction_bytes

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


def resolve_usable_l2_bytes(
    admission_memory_view: AdmissionMemoryView | None,
    die: int,
    *,
    fallback_raw_l2: RawDieL2Bytes = RawDieL2Bytes(0),
) -> UsableL2Bytes:
    """Resolve live usable L2 bytes without ever falling back to raw capacity when effective L2 is 0."""
    raw_cap = int(fallback_raw_l2)
    if admission_memory_view is not None:
        effective = int(admission_memory_view.effective_l2_bytes(die))
        if raw_cap > 0:
            return UsableL2Bytes(max(0, min(raw_cap, effective)))
        return UsableL2Bytes(max(0, effective))
    return UsableL2Bytes(max(0, raw_cap))


def resolve_engine_admission_view(
    engine: Any,
    *,
    die: int,
    raw_l2: RawDieL2Bytes,
    l2_units: int = 1,
) -> AdmissionMemoryView:
    """Return ``engine.admission_memory_view`` when attached and covering ``die``; construct a fallback only when None."""
    if type(l2_units) is not int or l2_units != 1:
        raise ModelingError("shared_l2_buffer_requires_one_die_owner")
    attached = getattr(engine, "admission_memory_view", None)
    if attached is not None:
        if not isinstance(attached, AdmissionMemoryView):
            raise ModelingError(f"engine_admission_memory_view_invalid:{type(attached).__name__}")
        if not attached.has_die(die):
            raise ModelingError(f"l2_die_undeclared:{die}")
        return attached
    return AdmissionMemoryView(
        {die: max(0, int(raw_l2))},
        l2_buffer_units_by_die={die: l2_units},
    )


def _extract_memory_hierarchy(source: Any) -> Any:
    """Resolve a ``MemoryHierarchySpec`` (or mapping) from a context, geometry, device, or spec."""
    cur: Any = source
    if cur is None:
        return None
    if isinstance(cur, Mapping):
        if "write_allocate" in cur:
            return cur
        if "geometry" in cur:
            cur = cur["geometry"]
    else:
        if hasattr(cur, "write_allocate"):
            return cur
        geometry = getattr(cur, "geometry", None)
        if geometry is not None:
            cur = geometry
    if cur is None:
        return None
    if isinstance(cur, Mapping):
        if "write_allocate" in cur:
            return cur
        if "physical_mechanisms" in cur:
            cur = cur["physical_mechanisms"]
    else:
        mechanisms = getattr(cur, "physical_mechanisms", None)
        if mechanisms is not None:
            cur = mechanisms
    if cur is None:
        return None
    if isinstance(cur, Mapping):
        if "write_allocate" in cur:
            return cur
        return cur.get("memory_hierarchy")
    if hasattr(cur, "write_allocate"):
        return cur
    return getattr(cur, "memory_hierarchy", None)


def l2_write_allocate_enabled(source: Any) -> bool:
    """Read declared ``MemoryHierarchySpec.write_allocate`` without assuming ``True``."""
    hierarchy = _extract_memory_hierarchy(source)
    if hierarchy is None:
        return False
    if isinstance(hierarchy, Mapping):
        return bool(hierarchy.get("write_allocate", False))
    return bool(getattr(hierarchy, "write_allocate", False))


@dataclass(slots=True)
class _LruLineSpan:
    """Mutable LRU queue node representing contiguous resident lines ``[start, end)``."""

    stream_key: object
    start: int
    end: int


class LineSegmentLruCache:
    """Exact sequential line-eviction LRU cache over disjoint ``[start_line, end_line)`` spans."""

    __slots__ = ("capacity_lines", "line_bytes", "_segments", "_total_lines")

    def __init__(self, capacity_bytes: int, line_bytes: int) -> None:
        self.line_bytes: int = max(1, int(line_bytes))
        self.capacity_lines: int = max(0, int(capacity_bytes)) // self.line_bytes
        self._segments: list[tuple[object, int, int]] = []
        self._total_lines: int = 0

    def access_lines(
        self, stream_key: object, start_line: int, end_line: int,
    ) -> tuple[tuple[int, int], ...]:
        """Access ``[start_line, end_line)`` sequentially, returning missed spans and updating LRU."""
        if end_line <= start_line:
            return ()
        if self.capacity_lines <= 0:
            return ((start_line, end_line),)

        lru: list[_LruLineSpan] = []
        req_spans: list[tuple[int, int, _LruLineSpan]] = []
        for seg_key, seg_start, seg_end in self._segments:
            if seg_key != stream_key or seg_end <= start_line or seg_start >= end_line:
                lru.append(_LruLineSpan(seg_key, seg_start, seg_end))
                continue
            ov_start = max(seg_start, start_line)
            ov_end = min(seg_end, end_line)
            if seg_start < ov_start:
                lru.append(_LruLineSpan(seg_key, seg_start, ov_start))
            node = _LruLineSpan(seg_key, ov_start, ov_end)
            lru.append(node)
            req_spans.append((ov_start, ov_end, node))
            if ov_end < seg_end:
                lru.append(_LruLineSpan(seg_key, ov_end, seg_end))

        req_spans.sort(key=lambda item: item[0])
        free = max(0, self.capacity_lines - self._total_lines)
        head = 0
        n_lru = len(lru)

        def evict(count: int) -> None:
            nonlocal free, head
            if free:
                take = min(free, count)
                free -= take
                count -= take
            while count > 0 and head < n_lru:
                cur = lru[head]
                seg_len = cur.end - cur.start
                if seg_len <= 0:
                    head += 1
                    continue
                if seg_len <= count:
                    cur.start = cur.end
                    head += 1
                    count -= seg_len
                else:
                    cur.start += count
                    count = 0

        misses: list[tuple[int, int]] = []

        def add_miss(lo: int, hi: int) -> None:
            if misses and misses[-1][1] == lo:
                misses[-1] = (misses[-1][0], hi)
            else:
                misses.append((lo, hi))

        cursor = start_line
        for u0, v0, node in req_spans:
            if u0 > cursor:
                add_miss(cursor, u0)
                evict(u0 - cursor)
                cursor = u0
            if node.start == u0 and node.start < node.end:
                node.start = node.end
            else:
                add_miss(u0, v0)
                evict(v0 - u0)
            cursor = v0
        if cursor < end_line:
            add_miss(cursor, end_line)
            evict(end_line - cursor)

        req_lines = end_line - start_line
        if req_lines >= self.capacity_lines:
            keep_start = end_line - self.capacity_lines
            self._segments = [(stream_key, keep_start, end_line)]
            self._total_lines = self.capacity_lines
            return tuple(misses)

        merged: list[tuple[object, int, int]] = []
        total = 0
        for idx in range(head, n_lru):
            cur = lru[idx]
            if cur.start >= cur.end:
                continue
            if merged and merged[-1][0] == cur.stream_key and merged[-1][2] == cur.start:
                merged[-1] = (cur.stream_key, merged[-1][1], cur.end)
            else:
                merged.append((cur.stream_key, cur.start, cur.end))
            total += cur.end - cur.start
        if merged and merged[-1][0] == stream_key and merged[-1][2] == start_line:
            merged[-1] = (stream_key, merged[-1][1], end_line)
        else:
            merged.append((stream_key, start_line, end_line))
        total += req_lines
        self._segments = merged
        self._total_lines = total
        return tuple(misses)


__all__ = [
    "AdmissionMemoryView",
    "DirtyWriteback",
    "LineSegmentLruCache",
    "RawDieL2Bytes",
    "UsableL2Bytes",
    "bind_memory_access",
    "l2_write_allocate_enabled",
    "resolve_engine_admission_view",
    "resolve_usable_l2_bytes",
]
