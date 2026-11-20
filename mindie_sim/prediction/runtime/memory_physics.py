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

    def stage_work(self) -> dict[str, float]:
        """Return merged physical resource demand mapping suitable for ``Stage``."""
        work: dict[str, float] = {}
        for term in self.constraints:
            if term.ms_per_progress > 0.0:
                work[term.row_id] = work.get(term.row_id, 0.0) + term.ms_per_progress
        return work


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


@dataclass(slots=True)
class _ResidentL2Entry:
    key: ReuseKey
    working_set_bytes: int
    version: int
    dirty: bool
    private_workspace: bool
    evicted: bool = False


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
        self._resident: dict[ReuseKey, _ResidentL2Entry] = {}
        units_map = dict(l2_buffer_units_by_die or {})
        for die, cap in l2_capacity_by_die.items():
            if type(die) is not int or die < 0 or type(cap) is not int or cap < 0:
                raise ModelingError(f"l2_capacity_invalid:{die!r}:{cap!r}")
            if units_map.get(die, 1) != 1:
                raise ModelingError("shared_l2_buffer_requires_one_die_owner")
            self._l2_capacity[die] = cap
            self._reserved[die] = {}
            self._active_streams[die] = set()

    def active_stream_count(self, die: int) -> int:
        return len(self._active_streams.get(die, ()))

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

    def _evict_over_capacity(self, die: int) -> None:
        eff = self.effective_l2_bytes(die)
        for entry in self._resident.values():
            if entry.key.die == die and not entry.evicted and entry.working_set_bytes > eff:
                entry.evicted = True

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
        return self.effective_l2_bytes(die)

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
        self._resident.pop(key, None)

    def is_resident(self, key: ReuseKey) -> bool:
        entry = self._resident.get(key)
        return entry is not None and not entry.evicted

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
        self.enter_stream(key.die, key.reuse_stream)
        eff = self.effective_l2_bytes(key.die)
        working_set = (
            pattern.reuse_distance_bytes
            if pattern.reuse_distance_bytes is not None else aligned_bytes
        )
        existing = self._resident.get(key)
        if pattern.direction == "read":
            if (
                existing is not None
                and not existing.evicted
                and existing.version == pattern.version
                and working_set <= eff
            ):
                return aligned_bytes, 0
            if working_set <= eff:
                self._resident[key] = _ResidentL2Entry(
                    key=key,
                    working_set_bytes=working_set,
                    version=pattern.version,
                    dirty=False,
                    private_workspace=pattern.visibility.private_workspace,
                    evicted=False,
                )
            elif existing is not None:
                existing.evicted = True
            return 0, aligned_bytes
        # Write path: private workspace fitting in C_eff avoids DRAM writeback;
        # live-out dirty writes settle to DRAM.
        if pattern.visibility.private_workspace and not pattern.visibility.live_out and working_set <= eff:
            self._resident[key] = _ResidentL2Entry(
                key=key,
                working_set_bytes=working_set,
                version=pattern.version,
                dirty=True,
                private_workspace=True,
                evicted=False,
            )
            return aligned_bytes, 0
        if working_set <= eff:
            self._resident[key] = _ResidentL2Entry(
                key=key,
                working_set_bytes=working_set,
                version=pattern.version,
                dirty=False,
                private_workspace=False,
                evicted=False,
            )
        elif existing is not None:
            existing.evicted = True
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
) -> BoundAccess:
    """Bind one ``MemoryAccessPattern`` into exact 3-ledger audit and normalized row terms."""
    if hbm_bus.die != pattern.die:
        raise ModelingError(f"hbm_bus_die_mismatch:{hbm_bus.die}!={pattern.die}")
    if copy_hit_bytes_per_ms <= 0.0 or copy_miss_bytes_per_ms <= 0.0 or l2_bytes_per_ms <= 0.0:
        raise ModelingError("bind_access_pipe_rate_nonpositive")

    aligned_bytes = pattern.runs.aligned_copy_bytes()
    if aligned_bytes == 0:
        ledger = ByteLedger(
            logical_tensor_bytes=pattern.logical_bytes,
            aligned_transfer_bytes=0,
            dram_bus_transaction_bytes=0.0,
            l2_hit_bytes=0,
            dram_miss_copy_bytes=0,
            dram_sector_bytes=0,
        )
        after = (pattern.visibility.producer_event,) if pattern.visibility.producer_event else ()
        return BoundAccess(ledger=ledger, constraints=(), deliveries=(), after=after)

    if state is not None:
        hit_bytes, miss_bytes = state.admit_access(pattern, aligned_bytes)
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
    if pipe_ms > 0.0:
        constraints.append(ConstraintTerm(pipe_row_id, pipe_ms))
    if hit_bytes > 0:
        constraints.append(ConstraintTerm(l2_row_id, float(hit_bytes) / l2_bytes_per_ms))
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
        ledger=ledger,
        constraints=tuple(constraints),
        deliveries=tuple(deliveries),
        after=after,
    )


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
    "HBM_DIR_READ",
    "HBM_DIR_WRITE",
    "HbmBusSpec",
    "MemoryAccessPattern",
    "ReuseKey",
    "VisibilityContract",
    "bind_memory_access",
    "burst_turnaround_efficiency",
]
