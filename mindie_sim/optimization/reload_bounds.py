"""Whole-step compulsory HBM sectors and direct-MAC reload certificates.

[Role] Bound cut traffic and expose direction choices for excess service.
[Invariants] Union compulsory sectors by version. Share reload bounds by maximum.
[Timeouts & Locks] Logarithmic inversion and bounded row expansion, with no locks.
[Side-Effect Policy] Pure integer and rational calculations.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import isqrt, lcm

from .contracts import BilinearContractionSpec, FusedRegionSummary, TensorRangeObligation
from .dual_bound import EpochAccount, verify_epoch_trace

SECTOR_BYTES = 32
BITS_PER_BYTE = 8
MAX_EXPLICIT_STRIDED_ROWS = 4096
MAX_EXPLICIT_EPOCH_ACCOUNTS = 65536
BILINEAR_PRODUCT_FACTOR = 4
RANK_ONE_KINDS = frozenset({
    "gdn", "kda", "linear_attention", "mamba", "gla", "deltanet", "lightning", "retnet",
})

__all__ = (
    "SECTOR_BYTES", "BITS_PER_BYTE", "MAX_EXPLICIT_STRIDED_ROWS", "CompulsoryHbmCut",
    "PebbleReloadResult", "SurplusCouplingSite", "compute_compulsory_hbm_cut",
    "direct_mac_pebble_reload_bound", "build_surplus_coupling_sites",
)


@dataclass(frozen=True, slots=True)
class CompulsoryHbmCut:
    die: int
    read_bytes: int
    write_bytes: int
    read_keys_count: int
    write_keys_count: int


@dataclass(frozen=True, slots=True)
class PebbleReloadResult:
    contraction_id: str
    die: int
    eligible: bool
    reason: str
    total_pairs: int
    min_io_bits: int
    min_io_bytes: int
    epochs: int
    epoch_accounts: tuple[EpochAccount, ...]


@dataclass(frozen=True, slots=True)
class SurplusCouplingSite:
    die: int
    surplus_bytes: int
    read_only: bool
    alpha_rw: Fraction
    modes: tuple[tuple[str, tuple[tuple[str, Fraction], ...]], ...]


def _union_count(spans: Sequence[tuple[int, int]]) -> int:
    end = count = 0
    for lo, hi in sorted(spans):
        count += max(0, hi - max(end, lo))
        end = max(end, hi)
    return count


@dataclass(slots=True)
class _SectorSet:
    intervals: set[tuple[int, int]] = field(default_factory=set)
    periodic: list[tuple[int, int, int, int]] = field(default_factory=list)
    periodic_bytes: int = 0

    def add(self, obligation: TensorRangeObligation, sector: int) -> None:
        def add_range(lo: int, hi: int) -> None:
            if lo < 0 or hi < lo:
                raise ValueError("invalid tensor byte range")
            if hi > lo:
                self.intervals.add((lo // sector, (hi + sector - 1) // sector))

        for lo, hi in obligation.ranges:
            add_range(lo, hi)
        for run in obligation.address_runs:
            if run.rows == 0 or run.row_bytes == 0:
                continue
            if run.stride_bytes == 0:
                run = replace(run, rows=1)
            row0_hi = (run.base_byte + run.row_bytes + sector - 1) // sector
            row1_lo = (run.base_byte + run.stride_bytes) // sector
            if run.rows == 1 or run.stride_bytes <= run.row_bytes or (
                run.stride_bytes % sector == 0 and row0_hi >= row1_lo
            ):
                end_byte = run.base_byte + (run.rows - 1) * run.stride_bytes + run.row_bytes
                add_range(run.base_byte, end_byte)
            elif run.rows > MAX_EXPLICIT_STRIDED_ROWS:
                if run.stride_bytes % sector == 0:
                    s0 = run.base_byte // sector
                    self.periodic.append((s0, row0_hi - s0, run.stride_bytes // sector, run.rows))
                self.periodic_bytes = max(self.periodic_bytes, run.sector_union_bytes(sector))
            else:
                for row in range(run.rows):
                    lo = run.base_byte + row * run.stride_bytes
                    add_range(lo, lo + run.row_bytes)

    def measure(self, sector: int) -> int:
        if self.periodic:
            period = lcm(*(step for _, _, step, _ in self.periodic))
            if period <= MAX_EXPLICIT_STRIDED_ROWS:
                by_res: dict[int, list[tuple[int, int]]] = defaultdict(list)
                for lo, hi in self.intervals:
                    for res in range(period):
                        q_lo = (lo - res + period - 1) // period
                        q_hi = (hi - 1 - res) // period + 1
                        if q_lo < q_hi:
                            by_res[res].append((q_lo, q_hi))
                for s0, width_sec, step, rows in self.periodic:
                    mult = period // step
                    for j in range(width_sec):
                        for m in range(min(mult, rows)):
                            start_sec = s0 + j + m * step
                            count = (rows - 1 - m) // mult + 1
                            q0, res = divmod(start_sec, period)
                            by_res[res].append((q0, q0 + count))
                total = sum(_union_count(spans) for spans in by_res.values())
                return max(total * sector, self.periodic_bytes)
        return max(_union_count(tuple(self.intervals)) * sector, self.periodic_bytes)


def compute_compulsory_hbm_cut(
    summaries: Sequence[FusedRegionSummary], *, sector_bytes: int = SECTOR_BYTES,
) -> dict[int, CompulsoryHbmCut]:
    if sector_bytes < 1:
        raise ValueError("sector size must be positive")
    reads: dict[tuple[int, str, int], _SectorSet] = defaultdict(_SectorSet)
    writes: dict[tuple[int, str, int], _SectorSet] = defaultdict(_SectorSet)
    for summary in summaries:
        for item in summary.read_obligations:
            if item.initial_residency == "hbm_cold" and item.tensor_version == 0:
                reads[item.die, item.allocation_id, item.tensor_version].add(item, sector_bytes)
        for item in summary.write_obligations:
            if item.required_final_visibility in {"device", "host"}:
                writes[item.die, item.allocation_id, item.tensor_version].add(item, sector_bytes)

    totals: dict[int, CompulsoryHbmCut] = {}
    for (die, _, _), sectors in reads.items():
        cut = totals.get(die, CompulsoryHbmCut(die, 0, 0, 0, 0))
        totals[die] = replace(cut, read_bytes=cut.read_bytes + sectors.measure(sector_bytes),
                              read_keys_count=cut.read_keys_count + 1)
    for (die, _, _), sectors in writes.items():
        cut = totals.get(die, CompulsoryHbmCut(die, 0, 0, 0, 0))
        totals[die] = replace(cut, write_bytes=cut.write_bytes + sectors.measure(sector_bytes),
                              write_keys_count=cut.write_keys_count + 1)
    return totals


def _full_epoch(capacity: int, resident: int, denom: int, bits_acc: int) -> EpochAccount:
    # Between survivor-floor jumps the completed-pair term increases. At jump
    # boundaries the survivor term is integral and convexity applies before flooring.
    budget = 2 * capacity
    first_jump = budget % bits_acc
    last_jump = budget - ((capacity + bits_acc - 1) // bits_acc) * bits_acc
    loads = (0, capacity, first_jump, last_jump)

    def account(load: int) -> EpochAccount:
        complete = (resident + load) ** 2 // denom
        split = (budget - load) // bits_acc
        return EpochAccount(complete, split, complete + split)

    return max((account(load) for load in loads if 0 <= load <= capacity),
               key=lambda epoch: epoch.epoch_pair_cap)


def direct_mac_pebble_reload_bound(spec: BilinearContractionSpec) -> PebbleReloadResult:
    pairs = spec.m * spec.n
    if not (spec.distinct_operands and spec.address_injective_across_k):
        return PebbleReloadResult(spec.contraction_id, spec.die, False,
                                  "operand_address_multiplicity_unproved", max(0, pairs), 0, 0, 0, ())
    reasons = (
        (spec.crosses_cut, "boundary_crossing_unproved"),
        (spec.pair_injective and spec.kind.lower() not in RANK_ONE_KINDS, "pair_injectivity_unproved"),
        (spec.direct_mac_no_recomp, "direct_mac_without_recomputation_unproved"),
        (spec.m >= 0 and spec.n >= 0 and min(spec.k_red, spec.bits_a, spec.bits_b,
         spec.bits_acc, spec.pool_capacity_bytes) > 0, "invalid_contraction_geometry"),
    )
    for eligible, reason in reasons:
        if not eligible:
            return PebbleReloadResult(spec.contraction_id, spec.die, False, reason,
                                      max(0, pairs), 0, 0, 0, ())

    capacity = BITS_PER_BYTE * spec.pool_capacity_bytes
    denom = BILINEAR_PRODUCT_FACTOR * spec.k_red**2 * spec.bits_a * spec.bits_b
    entry = 0 if spec.cold_entry else capacity
    target_square = pairs * denom
    root = isqrt(target_square)
    threshold = root + (root * root < target_square)
    io = max(0, threshold - entry)
    epochs = 1
    if io <= capacity:
        complete = (entry + io) ** 2 // denom
        trace = (EpochAccount(complete, 0, complete),)
    else:
        first = _full_epoch(capacity, entry, denom, spec.bits_acc)
        interior = _full_epoch(capacity, capacity, denom, spec.bits_acc)

        def envelope(io_bits: int) -> int:
            full, offset = divmod(io_bits - 1, capacity)
            final = (capacity + offset + 1) ** 2 // denom
            return first.epoch_pair_cap + (full - 1) * interior.epoch_pair_cap + final

        lo, hi = capacity, io
        if interior.epoch_pair_cap == 0 and envelope(hi) < pairs:
            # No finite I/O budget can realize this geometry under the hypotheses.
            return PebbleReloadResult(spec.contraction_id, spec.die, False,
                                      "epoch_capacity_cannot_cover_pairs", pairs, 0, 0, 0, ())
        while envelope(hi) < pairs:
            hi *= 2
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if envelope(mid) >= pairs:
                hi = mid
            else:
                lo = mid
        io = hi
        full, offset = divmod(io - 1, capacity)
        complete = (capacity + offset + 1) ** 2 // denom
        epochs = full + 1
        repeats = full - 1
        if epochs > MAX_EXPLICIT_EPOCH_ACCOUNTS:
            # Lean checks additive accounts. Fold a repeated interior run into
            # its exact sum so witness size stays bounded for extreme geometry.
            middle = (EpochAccount(interior.complete_pairs * repeats,
                                   interior.split_out_pairs * repeats,
                                   interior.epoch_pair_cap * repeats),)
        else:
            middle = (interior,) * repeats
        trace = (first, *middle, EpochAccount(complete, 0, complete))
    assert verify_epoch_trace(trace, pairs)
    return PebbleReloadResult(spec.contraction_id, spec.die, True, "eligible",
                              pairs, io, io // BITS_PER_BYTE, epochs, trace)


def build_surplus_coupling_sites(
    cuts: Mapping[int, CompulsoryHbmCut], reload_results: Sequence[PebbleReloadResult], *,
    alpha_rw: Fraction = Fraction(1, 1), write_free_by_die: Mapping[int, bool] = (),
) -> tuple[SurplusCouplingSite, ...]:
    if not 0 <= alpha_rw <= 1:
        raise ValueError("read/write cross-charge must be in [0, 1]")
    upper_reads = dict(write_free_by_die)
    bounds: dict[int, int] = defaultdict(int)
    for result in reload_results:
        if result.eligible:
            bounds[result.die] = max(bounds[result.die], result.min_io_bytes)

    sites = []
    for die, bound in sorted(bounds.items()):
        cut = cuts.get(die, CompulsoryHbmCut(die, 0, 0, 0, 0))
        surplus = max(0, bound - cut.read_bytes - cut.write_bytes)
        if surplus == 0:
            continue
        read = f"hbm_bus_r@d{die}"
        write = f"hbm_bus_w@d{die}"
        size = Fraction(surplus)
        modes = (("mode_surplus_read", (
            (read, size), (write, alpha_rw * size), (f"mte2_total@d{die}", size),
        )),)
        read_only = upper_reads.get(die, False)
        if not read_only:
            modes += (("mode_surplus_write", (
                (read, alpha_rw * size), (write, size), (f"write_total@d{die}", size),
            )),)
        sites.append(SurplusCouplingSite(die, surplus, read_only, alpha_rw, modes))
    return tuple(sites)
