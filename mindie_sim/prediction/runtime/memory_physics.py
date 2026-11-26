"""Three-ledger memory physics, dynamic L2 working-set knee, 2-row HBM bus and vector sub-pipes.

[Role] Model physical memory transactions across logical tensor bytes (L),
  aligned owner-pipe copy bytes (A = A_hit + A_miss), and directional DRAM bus
  service bytes (D) with 32 B sector unions, burst turnaround penalties, dynamic
  N-stream L2 working-set knees, 2-row directional HBM controller constraints,
  vector sub-pipe issue/execution constraints, and the canonical prepared-row
  epoch solver mirrored in C11.
[Invariants] Every ByteLedger conserves ``l2_hit_bytes + dram_miss_copy_bytes
  == aligned_transfer_bytes``. Shared L2 strictly requires ``l2_buffer_units == 1``
  per die. Solo read or solo write incurs zero RW turnaround penalty; concurrent
  read+write obeys ``x + alpha * y <= 1`` and ``alpha * x + y <= 1``. Evicted L2
  entries are never resurrected when active stream pressure drops.
[Timeouts & Locks] O(min(rows, sector_bytes)) exact integer sector unions and
  bounded progressive-filling loops; no I/O, threads or locks.
[Side-Effect Policy] Pure value records except explicit state transitions on
  caller-owned ``AdmissionMemoryView`` instances.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
import math
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError

DEFAULT_SECTOR_BYTES = 32
DEFAULT_ALIGN_BYTES = 32
DEFAULT_L2_GRANULE_BYTES = 128
BINARY64_EXACT_INT_LIMIT = 1 << 53
_MAX_EXPLICIT_SECTOR_INTERVALS = 4096
_SECTOR_INTERVAL_CACHE_LIMIT = 4096
_BISECT_OVERLAP_RATIO = 16
HBM_DIR_READ = 0
HBM_DIR_WRITE = 1
VALID_COPY_PIPES = frozenset({"mte2_aic", "mte2_aiv", "mte3", "fixpipe"})
VALID_VISIBILITY_SCOPES = frozenset({"owner", "die", "device", "host"})
CachePolicy = Literal["normal", "last_use", "disable", "persistent"]
_L2_CACHE_DISABLE: CachePolicy = "disable"
CACHE_POLICY_NORMAL: CachePolicy = "normal"
CACHE_POLICY_LAST_USE: CachePolicy = "last_use"
CACHE_POLICY_DISABLE: CachePolicy = _L2_CACHE_DISABLE
CACHE_POLICY_PERSISTENT: CachePolicy = "persistent"
VALID_CACHE_POLICIES: frozenset[CachePolicy] = frozenset({
    CACHE_POLICY_NORMAL, CACHE_POLICY_LAST_USE, CACHE_POLICY_DISABLE, CACHE_POLICY_PERSISTENT,
})


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _align_up(value: int, alignment: int) -> int:
    if value <= 0 or alignment <= 1:
        return max(0, value)
    return ((value + alignment - 1) // alignment) * alignment


def _merge_byte_intervals(
    intervals: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    cleaned = [(lo, hi) for lo, hi in intervals if hi > lo]
    if not cleaned:
        return ()
    cleaned.sort()
    merged: list[tuple[int, int]] = [cleaned[0]]
    for lo, hi in cleaned[1:]:
        p_lo, p_hi = merged[-1]
        if lo <= p_hi:
            if hi > p_hi:
                merged[-1] = (p_lo, hi)
        else:
            merged.append((lo, hi))
    return tuple(merged)


def _intersect_byte_intervals(
    left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    if not left or not right:
        return ()
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        lo = max(left[i][0], right[j][0])
        hi = min(left[i][1], right[j][1])
        if hi > lo:
            out.append((lo, hi))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return tuple(out)


def _measure_byte_intervals(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(hi - lo for lo, hi in intervals)


def _subtract_byte_intervals(
    base: Sequence[tuple[int, int]], cut: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    if not base or not cut:
        return tuple(base)
    if base[-1][1] <= cut[0][0] or cut[-1][1] <= base[0][0]:
        return tuple(base)
    out: list[tuple[int, int]] = []
    nc = len(cut)
    use_bisect = len(base) * _BISECT_OVERLAP_RATIO < nc
    j = 0
    for lo, hi in base:
        cur = lo
        if use_bisect:
            j = bisect_left(cut, (cur, 0))
            if j > 0 and cut[j - 1][1] > cur:
                j -= 1
        else:
            while j < nc and cut[j][1] <= cur:
                j += 1
        k = j
        while k < nc and cut[k][0] < hi:
            if cut[k][0] > cur:
                out.append((cur, min(hi, cut[k][0])))
            cur = max(cur, cut[k][1])
            if cur >= hi:
                break
            k += 1
        if cur < hi:
            out.append((cur, hi))
    return tuple(out)


@dataclass(frozen=True, slots=True)
class ReuseKey:
    """Unique L2 reuse stream identity on a single physical die."""

    die: int
    backing_allocation: str
    tensor_version: int
    reuse_stream: str = "main"

    def __post_init__(self) -> None:
        if type(self.die) is not int or self.die < 0:
            raise ModelingError(f"reuse_key_die_invalid:{self.die!r}")
        if not isinstance(self.backing_allocation, str) or not self.backing_allocation:
            raise ModelingError("reuse_key_backing_required")
        if type(self.tensor_version) is not int or self.tensor_version < 0:
            raise ModelingError(f"reuse_key_version_invalid:{self.tensor_version!r}")
        if not isinstance(self.reuse_stream, str) or not self.reuse_stream:
            raise ModelingError("reuse_key_stream_required")


@dataclass(frozen=True, slots=True)
class VisibilityContract:
    """Producer-consumer visibility and writeback boundary for one memory access."""

    producer_event: str | None = None
    scope: Literal["owner", "die", "device", "host"] = "die"
    live_out: bool = False
    private_workspace: bool = False
    proven_alias: bool = False

    def __post_init__(self) -> None:
        if self.scope not in VALID_VISIBILITY_SCOPES:
            raise ModelingError(f"visibility_scope_invalid:{self.scope!r}")


@dataclass(frozen=True, slots=True)
class BurstTurnRule:
    """Mutually exclusive row/stride burst turnaround penalty domain."""

    min_row_bytes: int
    max_row_bytes: int | None
    min_stride_bytes: int
    max_stride_bytes: int | None
    penalty_bytes: float
    evidence_id: str = ""

    def __post_init__(self) -> None:
        if self.min_row_bytes < 0 or self.min_stride_bytes < 0:
            raise ModelingError("burst_turn_rule_domain_negative")
        if self.max_row_bytes is not None and self.max_row_bytes <= self.min_row_bytes:
            raise ModelingError("burst_turn_rule_row_bounds_invalid")
        if self.max_stride_bytes is not None and self.max_stride_bytes <= self.min_stride_bytes:
            raise ModelingError("burst_turn_rule_stride_bounds_invalid")
        if not math.isfinite(self.penalty_bytes) or self.penalty_bytes < 0.0:
            raise ModelingError(f"burst_turn_penalty_invalid:{self.penalty_bytes!r}")

    def matches(self, row_bytes: int, stride_bytes: int) -> bool:
        if row_bytes < self.min_row_bytes:
            return False
        if self.max_row_bytes is not None and row_bytes >= self.max_row_bytes:
            return False
        if stride_bytes < self.min_stride_bytes:
            return False
        if self.max_stride_bytes is not None and stride_bytes >= self.max_stride_bytes:
            return False
        return True


def burst_turnaround_efficiency(
    row_bytes: int,
    stride_bytes: int,
    rules: Sequence[BurstTurnRule] = (),
) -> float:
    """Return ``eta(r, s) = r / (r + P_turn(s))`` in ``(0, 1]``."""
    if row_bytes <= 0:
        return 1.0
    penalty = 0.0
    matched = 0
    for rule in rules:
        if rule.matches(row_bytes, stride_bytes):
            penalty = float(rule.penalty_bytes)
            matched += 1
    if matched > 1:
        raise ModelingError(f"burst_turn_rules_overlapping:{row_bytes}:{stride_bytes}")
    if penalty <= 0.0:
        return 1.0
    return float(row_bytes) / (float(row_bytes) + penalty)


@dataclass(frozen=True, slots=True)
class AddressRun:
    """One strided physical address sequence belonging to a transaction group."""

    base_byte: int
    rows: int
    row_bytes: int
    stride_bytes: int
    layout: str = "contiguous"
    proven_alias_group: str | None = None

    def __post_init__(self) -> None:
        if self.base_byte < 0 or self.rows < 0 or self.row_bytes < 0 or self.stride_bytes < 0:
            raise ModelingError("address_run_negative_geometry")

    @classmethod
    def from_gm_gap(cls, base_byte: int, rows: int, row_bytes: int, gap_bytes: int, *,
                    layout: str = "contiguous", proven_alias_group: str | None = None) -> AddressRun:
        """Convert a signed DataCopyPad GM gap into a byte stride."""
        return cls(base_byte, rows, row_bytes, row_bytes + gap_bytes, layout, proven_alias_group)

    def sector_union_bytes(self, sector_bytes: int = DEFAULT_SECTOR_BYTES) -> int:
        """Exact 32 B sector union bytes in O(min(rows, sector_bytes)) integer arithmetic."""
        if self.rows == 0 or self.row_bytes == 0:
            return 0
        if sector_bytes < 1:
            raise ModelingError(f"sector_bytes_invalid:{sector_bytes}")
        base = self.base_byte
        width = self.row_bytes
        stride = self.stride_bytes
        s0 = base // sector_bytes
        e0 = _ceil_div(base + width, sector_bytes)
        first_sectors = e0 - s0
        if self.rows == 1 or stride == 0:
            return first_sectors * sector_bytes
        if stride == width:
            total_end = _ceil_div(base + self.rows * width, sector_bytes)
            return (total_end - s0) * sector_bytes
        period = sector_bytes // math.gcd(sector_bytes, stride)
        rem = self.rows - 1
        step_limit = min(rem, period)
        deltas: list[int] = []
        prev_end = e0
        for step in range(1, step_limit + 1):
            addr = base + step * stride
            sk = addr // sector_bytes
            ek = _ceil_div(addr + width, sector_bytes)
            start_new = sk if sk > prev_end else prev_end
            added = ek - start_new if ek > start_new else 0
            deltas.append(added)
            if ek > prev_end:
                prev_end = ek
        full_periods, tail = divmod(rem, period)
        total_sectors = (
            first_sectors
            + full_periods * sum(deltas)
            + sum(deltas[:tail])
        )
        return total_sectors * sector_bytes


def _run_is_sector_contiguous(run: AddressRun, sec: int) -> bool:
    """True iff every adjacent row pair touches or overlaps at sector granularity."""
    if run.rows <= 1 or run.stride_bytes == 0 or run.stride_bytes <= run.row_bytes:
        return True
    period = sec // math.gcd(sec, run.stride_bytes)
    limit = min(run.rows - 1, period)
    for step in range(limit):
        cur_end = _ceil_div(run.base_byte + step * run.stride_bytes + run.row_bytes, sec)
        nxt_start = (run.base_byte + (step + 1) * run.stride_bytes) // sec
        if nxt_start > cur_end:
            return False
    return True


@dataclass(frozen=True, slots=True)
class PeriodicSectorRun:
    """Exact compact representation of a gapped periodic strided sector set."""

    base_byte: int
    rows: int
    row_bytes: int
    stride_bytes: int
    sector_bytes: int = DEFAULT_SECTOR_BYTES

    def total_bytes(self) -> int:
        return AddressRun(
            self.base_byte, self.rows, self.row_bytes, self.stride_bytes,
        ).sector_union_bytes(self.sector_bytes)

    def _row_bounds(self, step: int) -> tuple[int, int]:
        addr = self.base_byte + step * self.stride_bytes
        return (addr // self.sector_bytes) * self.sector_bytes, _align_up(
            addr + self.row_bytes, self.sector_bytes,
        )

    def _period(self) -> int:
        return self.stride_bytes if self.stride_bytes % self.sector_bytes == 0 else math.lcm(self.stride_bytes, self.sector_bytes)

    def _bounds(self) -> tuple[int, int]:
        if self.rows <= 0 or self.row_bytes <= 0:
            return 0, 0
        return self._row_bounds(0)[0], self._row_bounds(self.rows - 1)[1]

    def canonical(self) -> PeriodicSectorRun:
        """Canonicalize to sector-aligned origin/span when stride is sector-aligned."""
        if self.stride_bytes > 0 and self.stride_bytes % self.sector_bytes == 0 and self.rows > 0:
            s0, e0 = self._row_bounds(0)
            return PeriodicSectorRun(s0, self.rows, e0 - s0, self.stride_bytes, self.sector_bytes)
        return self

    def _window_intervals(self, lo: int, hi: int) -> tuple[tuple[int, int], ...]:
        if hi <= lo or self.rows <= 0 or self.row_bytes <= 0 or self.stride_bytes <= 0:
            return ()
        sec, stride = self.sector_bytes, self.stride_bytes
        hi_limit = ((hi + sec - 1) // sec) * sec - 1
        if hi_limit < self.base_byte:
            return ()
        k_hi = min(self.rows - 1, (hi_limit - self.base_byte) // stride)
        lo_sec = (lo // sec) * sec
        num_lo = lo_sec - self.row_bytes + 1 - self.base_byte
        k_lo = 0 if num_lo <= 0 else (num_lo + stride - 1) // stride
        if k_lo > k_hi:
            return ()
        ivs = []
        for step in range(k_lo, k_hi + 1):
            s_k, e_k = self._row_bounds(step)
            c_lo, c_hi = max(lo, s_k), min(hi, e_k)
            if c_hi > c_lo:
                ivs.append((c_lo, c_hi))
        return _merge_byte_intervals(ivs)

    def overlap_with_interval(self, lo: int, hi: int) -> int:
        """Exact byte overlap between this periodic sector set and ``[lo, hi)``."""
        r_lo, r_hi = self._bounds()
        c_lo, c_hi = max(lo, r_lo), min(hi, r_hi)
        if c_hi <= c_lo or self.stride_bytes <= 0:
            return 0
        period = self._period()
        if (c_hi - c_lo) // self.stride_bytes <= _MAX_EXPLICIT_SECTOR_INTERVALS:
            return _measure_byte_intervals(self._window_intervals(c_lo, c_hi))
        in_lo, in_hi = min(c_hi, max(c_lo, r_lo + period)), max(min(c_hi, max(c_lo, r_lo + period)), min(c_hi, r_hi - period))
        total = _measure_byte_intervals(self._window_intervals(c_lo, in_lo))
        if in_hi > in_lo:
            full, rem = divmod(in_hi - in_lo, period)
            total += (full * _measure_byte_intervals(self._window_intervals(in_lo, in_lo + period)) if full else 0)
            total += (_measure_byte_intervals(self._window_intervals(in_lo + full * period, in_hi)) if rem else 0)
        return total + (_measure_byte_intervals(self._window_intervals(in_hi, c_hi)) if c_hi > in_hi else 0)

    def overlap_with_periodic(self, other: PeriodicSectorRun) -> int:
        """Exact byte overlap between two periodic sector runs."""
        return _measure_periodic_combo((self,), (other,), intersect=True)


def _measure_periodic_combo(
    left: tuple[PeriodicSectorRun, ...], right: tuple[PeriodicSectorRun, ...], *,
    intersect: bool, extra_left_ivs: tuple[tuple[int, int], ...] = (),
    extra_right_ivs: tuple[tuple[int, int], ...] = (),
) -> int:
    """Exact byte measure of union (``intersect=False``) or intersection of runs + intervals."""
    from .periodic_runs import _measure_periodic_combo as _measure_impl
    return _measure_impl(
        left, right, intersect=intersect,
        extra_left_ivs=extra_left_ivs, extra_right_ivs=extra_right_ivs,
    )


@dataclass(frozen=True, slots=True)
class SectorCoverage:
    """Exact sector coverage combining explicit intervals, periodic runs, and disjoint scale sectors."""

    intervals: tuple[tuple[int, int], ...] = ()
    periodic: tuple[PeriodicSectorRun, ...] = ()
    scale_intervals: tuple[tuple[int, int], ...] = ()
    _canonical: bool = field(default=False, repr=False, compare=False)
    _bytes: int | None = field(default=None, repr=False, compare=False)

    @property
    def empty(self) -> bool:
        return not self.intervals and not self.periodic and not self.scale_intervals

    def measure_bytes(self) -> int:
        if self._bytes is not None:
            return self._bytes
        if not self.periodic:
            payload = (
                _measure_byte_intervals(self.intervals) if self._canonical
                else _measure_byte_intervals(_merge_byte_intervals(self.intervals))
            )
        else:
            payload = _measure_periodic_combo(self.periodic, (), intersect=False, extra_left_ivs=self.intervals)
        scale = (
            _measure_byte_intervals(self.scale_intervals) if self._canonical
            else _measure_byte_intervals(_merge_byte_intervals(self.scale_intervals))
        )
        val = payload + scale
        object.__setattr__(self, "_bytes", val)
        return val

    def overlap_bytes(self, other: SectorCoverage) -> int:
        if self.empty or other.empty:
            return 0
        from ._periodic_stream import _overlap_canonical_intervals
        scale_ov = 0
        if self.scale_intervals and other.scale_intervals:
            ls = self.scale_intervals if self._canonical else _merge_byte_intervals(self.scale_intervals)
            rs = other.scale_intervals if other._canonical else _merge_byte_intervals(other.scale_intervals)
            scale_ov = _overlap_canonical_intervals(ls, rs)
        if not self.periodic and not other.periodic:
            if not self.intervals or not other.intervals:
                return scale_ov
            left = self.intervals if self._canonical else _merge_byte_intervals(self.intervals)
            right = other.intervals if other._canonical else _merge_byte_intervals(other.intervals)
            return _overlap_canonical_intervals(left, right) + scale_ov
        return _measure_periodic_combo(
            self.periodic, other.periodic, intersect=True,
            extra_left_ivs=self.intervals, extra_right_ivs=other.intervals,
        ) + scale_ov

    def union(self, other: SectorCoverage) -> SectorCoverage:
        if self.empty:
            return other
        if other.empty:
            return self
        if self._canonical and other._canonical:
            from ._periodic_stream import _merge_canonical_pairs
            ivs = _merge_canonical_pairs(self.intervals, other.intervals)
            scale_ivs = _merge_canonical_pairs(self.scale_intervals, other.scale_intervals)
        else:
            ivs = _merge_byte_intervals((*self.intervals, *other.intervals))
            scale_ivs = _merge_byte_intervals((*self.scale_intervals, *other.scale_intervals))
        per = (
            self.periodic if not other.periodic
            else other.periodic if not self.periodic
            else tuple(dict.fromkeys(r.canonical() for r in (*self.periodic, *other.periodic)))
        )
        return SectorCoverage(
            intervals=ivs, periodic=per,
            scale_intervals=scale_ivs,
            _canonical=True,
        )


@dataclass(frozen=True, slots=True)
class AddressRunPlan:
    """Explicit physical runs, layout alignment, and scale metadata for one access."""

    runs: tuple[AddressRun, ...] = ()
    alignment_bytes: int = DEFAULT_ALIGN_BYTES
    scale_bytes: int = 0
    sector_bytes: int | None = None
    expanded_queries: bool = True
    l2_line_bytes: int | None = None
    l2_sector_bytes: int | None = None
    hbm_burst_granule_bytes: int = 0
    hbm_scatter_row_open_bytes: int = 0
    scale_base_byte: int | None = None
    _default_sector: bool = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_default_sector", self.sector_bytes is None)
        resolved_sector = (
            (self.l2_sector_bytes or DEFAULT_SECTOR_BYTES)
            if self.sector_bytes is None else self.sector_bytes
        )
        if self.sector_bytes is None:
            object.__setattr__(self, "sector_bytes", resolved_sector)
        if (self.alignment_bytes < 1 or resolved_sector < 1 or self.scale_bytes < 0
                or any(g is not None and g < 1 for g in (self.l2_line_bytes, self.l2_sector_bytes))
                or min(self.hbm_burst_granule_bytes, self.hbm_scatter_row_open_bytes) < 0
                or (self.scale_base_byte is not None and self.scale_base_byte < 0)):
            raise ModelingError("address_run_plan_invalid_parameters")

    def _resolved_sector_bytes(self) -> int:
        sec = self.sector_bytes
        if sec is None:
            raise ModelingError("address_run_plan_sector_bytes_unset")
        return sec

    def scale_span(self) -> tuple[int, int] | None:
        """Logical half-open byte interval on the disjoint scale plane, or ``None``."""
        if self.scale_bytes <= 0:
            return None
        active = [run for run in self.runs if run.rows > 0 and run.row_bytes > 0]
        if not active:
            return None
        if self.scale_base_byte is not None:
            scale_lo = self.scale_base_byte
        else:
            min_base = min(run.base_byte for run in active)
            if min_base == 0:
                scale_lo = 0
            else:
                max_end = max(
                    run.base_byte + (run.rows - 1) * run.stride_bytes + run.row_bytes
                    for run in active
                )
                payload_span = max_end - min_base
                scale_lo = (min_base * self.scale_bytes) // payload_span if payload_span > 0 else 0
        return scale_lo, scale_lo + self.scale_bytes

    def scale_sector_intervals(self, sector_bytes: int | None = None) -> tuple[tuple[int, int], ...]:
        """Sector-rounded interval on the disjoint scale plane."""
        span = self.scale_span()
        if span is None:
            return ()
        sec = self._resolved_sector_bytes() if sector_bytes is None else sector_bytes
        lo, hi = span
        return (((lo // sec) * sec, _align_up(hi, sec)),)

    def aligned_copy_bytes(self) -> int:
        """Exact owner-pipe layout transfer bytes (A_f), deduplicating proven aliases."""
        seen_aliases: set[str] = set()
        total = 0
        for run in self.runs:
            if run.rows == 0 or run.row_bytes == 0:
                continue
            if run.proven_alias_group is not None:
                if run.proven_alias_group in seen_aliases:
                    continue
                seen_aliases.add(run.proven_alias_group)
                rows_eff = 1 if run.stride_bytes == 0 else run.rows
            else:
                rows_eff = run.rows
            total += (_align_up(rows_eff * run.row_bytes, self.alignment_bytes) if run.stride_bytes == run.row_bytes
                      else rows_eff * _align_up(run.row_bytes, self.alignment_bytes))
        return total + (_align_up(self.scale_bytes, self.alignment_bytes) if total > 0 and self.scale_bytes else 0)

    def dram_sector_and_service(self, burst_rules: Sequence[BurstTurnRule] = ()) -> tuple[int, float]:
        """Exact integer 32 B sector bytes and burst-adjusted DRAM service bytes (D_f)."""
        ordered = _canonical_dram_runs(self, burst_rules)
        covered = SectorCoverage()
        sector_total, service_total = 0, 0.0
        single_run = len(ordered) == 1
        sec = self._resolved_sector_bytes()
        for run in ordered:
            if single_run:
                sec_bytes = run.sector_union_bytes(sec)
            else:
                run_cov = AddressRunPlan((run,), alignment_bytes=self.alignment_bytes, sector_bytes=sec).sector_coverage()
                sec_bytes = max(0, run_cov.measure_bytes() - run_cov.overlap_bytes(covered))
                covered = covered.union(run_cov)
            if sec_bytes > 0:
                sector_total += sec_bytes
                service_total += _burst_service_bytes(self, sec_bytes, run, burst_rules)
        if sector_total > 0 and self.scale_bytes > 0:
            scale_sec = _measure_byte_intervals(self.scale_sector_intervals(sec))
            sector_total += scale_sec
            service_total += float(scale_sec)
        return sector_total, service_total

    def sector_intervals(self) -> tuple[tuple[int, int], ...]:
        """Return canonical disjoint ``[lo, hi)`` sector-aligned byte intervals."""
        return _cached_sector_intervals(self.runs, self._resolved_sector_bytes())

    def periodic_sector_runs(self) -> tuple[PeriodicSectorRun, ...]:
        """Return exact compact periodic descriptors for gapped runs with ``rows > 4096``."""
        sec = self._resolved_sector_bytes()
        return tuple(dict.fromkeys(
            PeriodicSectorRun(run.base_byte, run.rows, run.row_bytes, run.stride_bytes, sec).canonical()
            for run in self.runs
            if run.rows > _MAX_EXPLICIT_SECTOR_INTERVALS and run.row_bytes > 0 and not _run_is_sector_contiguous(run, sec)
        ))

    def sector_coverage(self) -> SectorCoverage:
        """Exact sector coverage retaining periodic strided sets and disjoint scale sectors."""
        sec = self._resolved_sector_bytes()
        scale_ivs = self.scale_sector_intervals(sec)
        scale_sec = _measure_byte_intervals(scale_ivs)
        known_bytes = (
            self.runs[0].sector_union_bytes(sec) + scale_sec
            if len(self.runs) == 1 else None
        )
        return SectorCoverage(
            intervals=self.sector_intervals(),
            periodic=self.periodic_sector_runs(),
            scale_intervals=scale_ivs,
            _canonical=True,
            _bytes=known_bytes,
        )


@lru_cache(maxsize=_SECTOR_INTERVAL_CACHE_LIMIT)
def _cached_sector_intervals(
    runs: tuple[AddressRun, ...], sec: int,
) -> tuple[tuple[int, int], ...]:
    if len(runs) == 1:
        run = runs[0]
        if run.rows == 0 or run.row_bytes == 0:
            return ()
        s0 = (run.base_byte // sec) * sec
        if _run_is_sector_contiguous(run, sec):
            end = (
                run.base_byte + run.row_bytes
                if (run.rows == 1 or run.stride_bytes == 0)
                else run.base_byte + (run.rows - 1) * run.stride_bytes + run.row_bytes
            )
            return ((s0, _align_up(end, sec)),)
        if run.rows <= _MAX_EXPLICIT_SECTOR_INTERVALS:
            base, stride, width, rows = run.base_byte, run.stride_bytes, run.row_bytes, run.rows
            # Strictly positive stride larger than aligned row width+sector guarantees
            # sorted, pairwise-disjoint sector intervals with no merge pass needed.
            if stride >= ((width + sec - 1) // sec + 1) * sec:
                if base % sec == 0 and stride % sec == 0:
                    w_aligned = ((width + sec - 1) // sec) * sec
                    return tuple((addr, addr + w_aligned) for addr in range(base, base + rows * stride, stride))
                return tuple(
                    (((addr := base + step * stride) // sec) * sec, _align_up(addr + width, sec))
                    for step in range(rows)
                )
    ivs: list[tuple[int, int]] = []
    for run in runs:
        if run.rows == 0 or run.row_bytes == 0:
            continue
        s0 = (run.base_byte // sec) * sec
        if _run_is_sector_contiguous(run, sec):
            end = run.base_byte + run.row_bytes if (run.rows == 1 or run.stride_bytes == 0) else run.base_byte + (run.rows - 1) * run.stride_bytes + run.row_bytes
            ivs.append((s0, _align_up(end, sec)))
        elif run.rows <= _MAX_EXPLICIT_SECTOR_INTERVALS:
            for step in range(run.rows):
                addr = run.base_byte + step * run.stride_bytes
                ivs.append(((addr // sec) * sec, _align_up(addr + run.row_bytes, sec)))
    return _merge_byte_intervals(ivs)


@dataclass(frozen=True, slots=True)
class MemoryAccessPattern:
    """Declarative physical memory access passed to ``RecipeContext.bind_access``."""

    tensor_id: str
    version: int
    die: int
    owner: int
    direction: Literal["read", "write"]
    pipe: Literal["mte2_aic", "mte2_aiv", "mte3", "fixpipe"]
    logical_bytes: int
    runs: AddressRunPlan
    reuse_key: ReuseKey | None = None
    reuse_distance_bytes: int | None = None
    visibility: VisibilityContract = field(default_factory=VisibilityContract)
    cache_policy: CachePolicy = CACHE_POLICY_NORMAL

    def __post_init__(self) -> None:
        if not self.tensor_id:
            raise ModelingError("memory_access_tensor_id_required")
        if type(self.version) is not int or self.version < 0:
            raise ModelingError(f"memory_access_version_invalid:{self.version!r}")
        if type(self.die) is not int or self.die < 0:
            raise ModelingError(f"memory_access_die_invalid:{self.die!r}")
        if type(self.owner) is not int or self.owner < 0:
            raise ModelingError(f"memory_access_owner_invalid:{self.owner!r}")
        if self.direction not in {"read", "write"}:
            raise ModelingError(f"memory_access_direction_invalid:{self.direction!r}")
        if self.pipe not in VALID_COPY_PIPES:
            raise ModelingError(f"memory_access_pipe_invalid:{self.pipe!r}")
        if type(self.logical_bytes) is not int or self.logical_bytes < 0:
            raise ModelingError(f"memory_access_logical_bytes_invalid:{self.logical_bytes!r}")
        if self.reuse_distance_bytes is not None and (
            type(self.reuse_distance_bytes) is not int or self.reuse_distance_bytes < 0
        ):
            raise ModelingError(f"memory_access_reuse_distance_invalid:{self.reuse_distance_bytes!r}")
        if self.cache_policy not in VALID_CACHE_POLICIES:
            raise ModelingError(f"memory_access_cache_policy_invalid:{self.cache_policy!r}")


@dataclass(frozen=True, slots=True)
class ByteLedger:
    """Three-ledger byte conservation audit record for one bound access."""

    logical_tensor_bytes: int
    aligned_transfer_bytes: int
    dram_bus_transaction_bytes: float
    l2_hit_bytes: int
    dram_miss_copy_bytes: int
    dram_sector_bytes: int

    @property
    def dram_service_bytes(self) -> float:
        return self.dram_bus_transaction_bytes

    def __post_init__(self) -> None:
        for field_name in (
            "logical_tensor_bytes", "aligned_transfer_bytes",
            "l2_hit_bytes", "dram_miss_copy_bytes", "dram_sector_bytes",
        ):
            val = getattr(self, field_name)
            if type(val) is not int or val < 0:
                raise ModelingError(f"byte_ledger_{field_name}_invalid:{val!r}")
        if self.l2_hit_bytes + self.dram_miss_copy_bytes != self.aligned_transfer_bytes:
            raise ModelingError("byte_ledger_copy_conservation_violated")
        if not math.isfinite(self.dram_bus_transaction_bytes) or self.dram_bus_transaction_bytes < 0.0:
            raise ModelingError("byte_ledger_dram_service_invalid")
        if self.dram_miss_copy_bytes == 0 and (
            self.dram_sector_bytes != 0 or self.dram_bus_transaction_bytes != 0.0
        ):
            raise ModelingError("byte_ledger_zero_miss_requires_zero_dram")


@dataclass(frozen=True, slots=True)
class ConstraintTerm:
    """One normalized row constraint ``ms_per_progress`` for a single flow."""

    row_id: str
    ms_per_progress: float

    def __post_init__(self) -> None:
        if not self.row_id:
            raise ModelingError("constraint_term_row_required")
        if not math.isfinite(self.ms_per_progress) or self.ms_per_progress < 0.0:
            raise ModelingError(f"constraint_term_invalid:{self.row_id}:{self.ms_per_progress!r}")


@dataclass(frozen=True, slots=True)
class CounterTerm:
    """One unique directional delivery counter entry per unit progress."""

    counter_id: str
    work_per_progress: float

    def __post_init__(self) -> None:
        if not self.counter_id:
            raise ModelingError("counter_term_id_required")
        if not math.isfinite(self.work_per_progress) or self.work_per_progress < 0.0:
            raise ModelingError(f"counter_term_invalid:{self.counter_id}:{self.work_per_progress!r}")


@dataclass(frozen=True, slots=True)
class BoundAccess:
    """Lowered physical constraints, directional deliveries, and 3-ledger audit."""

    ledger: ByteLedger
    constraints: tuple[ConstraintTerm, ...]
    deliveries: tuple[CounterTerm, ...]
    after: tuple[str, ...] = ()
    owner: int = 0
    pipe_row_id: str = ""
    l2_row_id: str = ""
    row_rates: Mapping[str, float] = field(default_factory=dict)
    bind_config: tuple[Any, ...] | None = None

    def stage_placement(self) -> dict[str, tuple[int, ...]]:
        """Pin the copy pipe to ``owner`` when it carries positive work."""
        if not self.pipe_row_id:
            return {}
        if any(t.row_id == self.pipe_row_id and t.ms_per_progress > 0.0 for t in self.constraints):
            return {self.pipe_row_id: (self.owner,)}
        return {}

    def stage_work(
        self, *, resource_rates: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Return Engine-native work without double-dividing byte-rate pipes."""
        work: dict[str, float] = {}
        for term in self.constraints:
            if term.ms_per_progress <= 0.0:
                continue
            rate = (
                resource_rates[term.row_id]
                if resource_rates is not None and term.row_id in resource_rates
                else self.row_rates.get(term.row_id)
            )
            amt = term.ms_per_progress * float(rate) if rate is not None else term.ms_per_progress
            work[term.row_id] = work.get(term.row_id, 0.0) + amt
        return work

    def to_stage(
        self,
        name: str,
        *,
        after: Sequence[str] = (),
        resource_rates: Mapping[str, float] | None = None,
    ) -> Any:
        """Build a coupled owner-pinned ``Stage`` for this bound access."""
        from ..program.stages import Stage
        deps = tuple(dict.fromkeys((*self.after, *after)))
        work = self.stage_work(resource_rates=resource_rates)
        delivs = tuple((c.counter_id, c.work_per_progress) for c in self.deliveries if c.work_per_progress > 0.0)
        return Stage(
            name, work=work, after=deps,
            placement=self.stage_placement(), coupled=bool(work),
            deliveries=delivs,
        )


@dataclass(frozen=True, slots=True)
class HbmBusSpec:
    """Per-die directional 2-row HBM controller specification."""

    die: int
    read_bytes_per_ms: float
    write_bytes_per_ms: float
    alpha_rw: float
    row_r: str = ""
    row_w: str = ""
    ctr_r: str = ""
    ctr_w: str = ""
    memory_access: Any = field(default=None, compare=False, repr=False, kw_only=True)

    def __post_init__(self) -> None:
        if type(self.die) is not int or self.die < 0:
            raise ModelingError(f"hbm_bus_die_invalid:{self.die!r}")
        if not math.isfinite(self.read_bytes_per_ms) or self.read_bytes_per_ms <= 0.0:
            raise ModelingError(f"hbm_bus_read_rate_invalid:{self.read_bytes_per_ms!r}")
        if not math.isfinite(self.write_bytes_per_ms) or self.write_bytes_per_ms <= 0.0:
            raise ModelingError(f"hbm_bus_write_rate_invalid:{self.write_bytes_per_ms!r}")
        if not math.isfinite(self.alpha_rw) or not (0.0 <= self.alpha_rw <= 1.0):
            raise ModelingError(f"hbm_bus_alpha_invalid:{self.alpha_rw!r}")
        if not self.row_r:
            object.__setattr__(self, "row_r", f"hbm_bus_r@d{self.die}")
        if not self.row_w:
            object.__setattr__(self, "row_w", f"hbm_bus_w@d{self.die}")
        if not self.ctr_r:
            object.__setattr__(self, "ctr_r", f"hbm_read@d{self.die}")
        if not self.ctr_w:
            object.__setattr__(self, "ctr_w", f"hbm_write@d{self.die}")

    def terms_for(
        self, direction: Literal["read", "write"], dram_service_bytes: float
    ) -> tuple[ConstraintTerm, ConstraintTerm]:
        """Emit the directional 2-row constraint pair ``(row_r, row_w)``."""
        if dram_service_bytes <= 0.0:
            return (
                ConstraintTerm(self.row_r, 0.0),
                ConstraintTerm(self.row_w, 0.0),
            )
        if direction == "read":
            t_ms = dram_service_bytes / self.read_bytes_per_ms
            return (
                ConstraintTerm(self.row_r, t_ms),
                ConstraintTerm(self.row_w, self.alpha_rw * t_ms),
            )
        t_ms = dram_service_bytes / self.write_bytes_per_ms
        return (
            ConstraintTerm(self.row_r, self.alpha_rw * t_ms),
            ConstraintTerm(self.row_w, t_ms),
        )


from .l2_residency import (
    AdmissionMemoryView, DirtyWriteback, _burst_service_bytes, _canonical_dram_runs,
    bind_memory_access, cache_policy_for_arch,
)


__all__ = [
    "AdmissionMemoryView", "AddressRun", "AddressRunPlan", "BINARY64_EXACT_INT_LIMIT",
    "BoundAccess", "BurstTurnRule", "ByteLedger", "CachePolicy", "ConstraintTerm",
    "CounterTerm", "DirtyWriteback", "HbmBusSpec", "MemoryAccessPattern",
    "PeriodicSectorRun", "ReuseKey", "SectorCoverage", "VisibilityContract",
    "CACHE_POLICY_DISABLE", "CACHE_POLICY_LAST_USE", "CACHE_POLICY_NORMAL",
    "CACHE_POLICY_PERSISTENT", "VALID_CACHE_POLICIES", "_L2_CACHE_DISABLE",
    "DEFAULT_ALIGN_BYTES", "DEFAULT_L2_GRANULE_BYTES", "DEFAULT_SECTOR_BYTES",
    "HBM_DIR_READ", "HBM_DIR_WRITE", "bind_memory_access", "burst_turnaround_efficiency",
    "cache_policy_for_arch",
]
