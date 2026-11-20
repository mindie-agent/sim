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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError

DEFAULT_SECTOR_BYTES = 32
DEFAULT_ALIGN_BYTES = 32
DEFAULT_L2_GRANULE_BYTES = 128
BINARY64_EXACT_INT_LIMIT = 1 << 53
_MAX_EXPLICIT_SECTOR_INTERVALS = 4096
HBM_DIR_READ = 0
HBM_DIR_WRITE = 1
VALID_COPY_PIPES = frozenset({"mte2_aic", "mte2_aiv", "mte3", "fixpipe"})
VALID_VISIBILITY_SCOPES = frozenset({"owner", "die", "device", "host"})


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
        if self.rows == 1:
            return first_sectors * sector_bytes
        if stride == 0:
            if self.proven_alias_group is not None:
                return first_sectors * sector_bytes
            return self.rows * first_sectors * sector_bytes
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


@dataclass(frozen=True, slots=True)
class AddressRunPlan:
    """Explicit physical runs, layout alignment, and scale metadata for one access."""

    runs: tuple[AddressRun, ...] = ()
    alignment_bytes: int = DEFAULT_ALIGN_BYTES
    scale_bytes: int = 0
    sector_bytes: int = DEFAULT_SECTOR_BYTES
    expanded_queries: bool = True

    def __post_init__(self) -> None:
        if self.alignment_bytes < 1 or self.sector_bytes < 1 or self.scale_bytes < 0:
            raise ModelingError("address_run_plan_invalid_parameters")

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
            if run.stride_bytes == run.row_bytes:
                total += _align_up(rows_eff * run.row_bytes, self.alignment_bytes)
            else:
                total += rows_eff * _align_up(run.row_bytes, self.alignment_bytes)
        if total > 0:
            total += _align_up(self.scale_bytes, self.alignment_bytes) if self.scale_bytes else 0
        return total

    def dram_sector_and_service(
        self,
        burst_rules: Sequence[BurstTurnRule] = (),
    ) -> tuple[int, float]:
        """Exact integer 32 B sector bytes and burst-adjusted DRAM service bytes (D_f)."""
        seen_aliases: set[str] = set()
        sector_total = 0
        service_total = 0.0
        for run in self.runs:
            if run.rows == 0 or run.row_bytes == 0:
                continue
            if run.proven_alias_group is not None:
                if run.proven_alias_group in seen_aliases:
                    continue
                seen_aliases.add(run.proven_alias_group)
            sec_bytes = run.sector_union_bytes(self.sector_bytes)
            eta = burst_turnaround_efficiency(run.row_bytes, run.stride_bytes, burst_rules)
            sector_total += sec_bytes
            service_total += float(sec_bytes) / eta
        if sector_total > 0 and self.scale_bytes > 0:
            scale_sec = _align_up(self.scale_bytes, self.sector_bytes)
            sector_total += scale_sec
            service_total += float(scale_sec)
        return sector_total, service_total

    def sector_intervals(self) -> tuple[tuple[int, int], ...]:
        """Return canonical disjoint ``[lo, hi)`` sector-aligned byte intervals."""
        sec = self.sector_bytes
        ivs: list[tuple[int, int]] = []
        for run in self.runs:
            if run.rows == 0 or run.row_bytes == 0:
                continue
            s0 = (run.base_byte // sec) * sec
            if run.rows == 1 or run.stride_bytes == 0:
                ivs.append((s0, _align_up(run.base_byte + run.row_bytes, sec)))
            elif run.stride_bytes <= run.row_bytes + sec:
                end = run.base_byte + (run.rows - 1) * run.stride_bytes + run.row_bytes
                ivs.append((s0, _align_up(end, sec)))
            elif run.rows <= _MAX_EXPLICIT_SECTOR_INTERVALS:
                for step in range(run.rows):
                    addr = run.base_byte + step * run.stride_bytes
                    ivs.append(((addr // sec) * sec, _align_up(addr + run.row_bytes, sec)))
            else:
                end = run.base_byte + (run.rows - 1) * run.stride_bytes + run.row_bytes
                ivs.append((s0, _align_up(end, sec)))
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


@dataclass(frozen=True, slots=True)
class ByteLedger:
    """Three-ledger byte conservation audit record for one bound access."""

    logical_tensor_bytes: int
    aligned_transfer_bytes: int
    dram_bus_transaction_bytes: float
    l2_hit_bytes: int
    dram_miss_copy_bytes: int
    dram_sector_bytes: int

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


from .l2_residency import AdmissionMemoryView, DirtyWriteback, bind_memory_access


__all__ = [
    "AdmissionMemoryView",
    "AddressRun",
    "AddressRunPlan",
    "BINARY64_EXACT_INT_LIMIT",
    "BoundAccess",
    "BurstTurnRule",
    "ByteLedger",
    "ConstraintTerm",
    "CounterTerm",
    "DEFAULT_ALIGN_BYTES",
    "DEFAULT_L2_GRANULE_BYTES",
    "DEFAULT_SECTOR_BYTES",
    "DirtyWriteback",
    "HBM_DIR_READ",
    "HBM_DIR_WRITE",
    "HbmBusSpec",
    "MemoryAccessPattern",
    "ReuseKey",
    "VisibilityContract",
    "bind_memory_access",
    "burst_turnaround_efficiency",
]
