"""Cache path geometry, policy and access accounting.

[Role] Declare transaction geometry and describe or journal resolved physical work.
[Invariants] Payload, resident line ranges and next-level work are tracked separately.
[Timeouts & Locks] Records are bounded by the current access's normalized ranges.
[Side-Effect Policy] Updates the caller's per-access journal when recording progress.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from mindie_sim.schema.common import ModelingError

from ..cache_path.geometry import byte_spans_of_line_spans

# granular (npu_arch_3510.md:28); no cache hit/miss policy is claimed by it.
A5_LINE_BYTES = 128
# A5 GM copy / UB transaction granule already used by the existing binder.
A5_TRANSFER_BYTES = 32


def _align_up(value: int, base: int) -> int:
    return (value + base - 1) // base * base


@dataclass(frozen=True)
class CacheGeometry:
    """One declarative cache level between GM requests and next-level lines.

    ``capacity_bytes`` is the competition scope shared by every identity.
    ``line_bytes`` is the next-level transaction width. ``transfer_bytes`` is
    the GM/on-chip copy granule. ``staging_lines`` bounds in-flight occupancy;
    an access never creates one object per byte or per payload line.
    """

    capacity_bytes: int = 0
    line_bytes: int = A5_LINE_BYTES
    transfer_bytes: int = A5_TRANSFER_BYTES
    staging_lines: int = 1

    def __post_init__(self):
        for name in ("capacity_bytes", "line_bytes", "transfer_bytes", "staging_lines"):
            value = getattr(self, name)
            if type(value) is not int:
                raise ModelingError(f"cache_geometry_requires_integer:{name}")
        if self.capacity_bytes < 0:
            raise ModelingError("cache_geometry_capacity_must_be_nonnegative")
        if self.line_bytes < 1 or self.transfer_bytes < 1 or self.staging_lines < 1:
            raise ModelingError("cache_geometry_granules_must_be_positive")

    @property
    def capacity_lines(self) -> int:
        return self.capacity_bytes // self.line_bytes


@dataclass(frozen=True)
class CachePolicy:
    """Access behavior declared by the selected implementation path."""

    bypass: bool = False
    # A cold path has no reusable identity. It never looks up, fills or
    # evicts; state remains untouched even when a long-lived state is passed.
    cold: bool = False
    write_policy: str = "write_through"
    ready_ms: float = 0.0

    def __post_init__(self):
        if type(self.bypass) is not bool or type(self.cold) is not bool:
            raise ModelingError("cache_policy_bypass_and_cold_must_be_boolean")
        if self.write_policy not in {"write_through", "write_back"}:
            raise ModelingError("cache_policy_unknown_write_policy")
        if not isinstance(self.ready_ms, (int, float)) or isinstance(self.ready_ms, bool):
            raise ModelingError("cache_policy_ready_must_be_numeric")
        if not (self.ready_ms >= 0.0):
            raise ModelingError("cache_policy_ready_must_be_nonnegative")


@dataclass(frozen=True)
class CacheAccessResult:
    """Physical quantities resolved for one access against current state."""

    identity: str
    mode: str
    logical_bytes: int
    gm_bytes: int
    onchip_bytes: int
    staging_bytes: int
    hbm_read_bytes: int
    hbm_write_bytes: int
    hbm_read_lines: int
    hbm_write_lines: int
    cache_read_bytes: int
    fill_bytes: int
    evicted_bytes: int
    writeback_bytes: int
    # New bytes written through now because they were not retained. Dirty
    # writebacks remain separate; hbm_write_bytes == direct + writeback.
    direct_write_bytes: int
    hit_lines: int
    miss_lines: int
    resident_bytes: int
    bypassed: bool
    cold_path: bool
    ready_ms: float
    line_bytes: int = A5_LINE_BYTES

    @property
    def hbm_bytes(self) -> int:
        return self.hbm_read_bytes + self.hbm_write_bytes

    @property
    def transactions(self) -> int:
        return self.hbm_read_lines + self.hbm_write_lines

    def to_dict(self) -> dict:
        return {
            "identity": self.identity, "mode": self.mode,
            "logical_bytes": self.logical_bytes, "gm_bytes": self.gm_bytes,
            "onchip_bytes": self.onchip_bytes, "staging_bytes": self.staging_bytes,
            "hbm_read_bytes": self.hbm_read_bytes,
            "hbm_write_bytes": self.hbm_write_bytes,
            "hbm_bytes": self.hbm_bytes,
            "hbm_read_lines": self.hbm_read_lines,
            "hbm_write_lines": self.hbm_write_lines,
            "cache_read_bytes": self.cache_read_bytes,
            "fill_bytes": self.fill_bytes, "evicted_bytes": self.evicted_bytes,
            "writeback_bytes": self.writeback_bytes,
            "direct_write_bytes": self.direct_write_bytes,
            "hit_lines": self.hit_lines, "miss_lines": self.miss_lines,
            "resident_bytes": self.resident_bytes, "bypassed": self.bypassed,
            "cold_path": self.cold_path,
            "ready_ms": self.ready_ms,
            "transactions": self.transactions,
        }


def _byte_spans_of_lines(ranges, line):
    return byte_spans_of_line_spans(ranges, line)


@dataclass
class CacheAccessJournal:
    """One access's physical responsibility and its real terminal history.

    Prepared capacity, actual issued stages, publication and retirement live in
    this single record.  A failed issuance therefore cannot silently free a
    reservation or staging slot: the journal still says which work was already
    submitted and whether its physical effect was published.
    """

    identity: str
    holder: int | None = None
    target_lines: int = 0
    reserved_bytes: int = 0
    dirty_lines: int = 0
    dirty_bytes: int = 0
    drain_bytes: int = 0
    direct_bytes: int = 0
    staged_bytes: int = 0
    issued: list = field(default_factory=list)
    published: bool = False
    drained: bool = False
    retired: bool = False
    failure: str = ""

    def note_issue(self, label: str, resource: str | None, amount: float) -> None:
        self.issued.append((label, resource, float(amount)))

    def note_publish(self) -> None:
        self.published = True

    def note_drain(self) -> None:
        self.drained = True

    def note_retire(self) -> None:
        self.retired = True

    def snapshot(self) -> dict:
        return {
            "identity": self.identity, "holder": self.holder,
            "target_lines": self.target_lines,
            "reserved_bytes": self.reserved_bytes,
            "dirty_lines": self.dirty_lines, "dirty_bytes": self.dirty_bytes,
            "drain_bytes": self.drain_bytes,
            "direct_bytes": self.direct_bytes,
            "staged_bytes": self.staged_bytes,
            "issued": [list(item) for item in self.issued],
            "published": self.published, "drained": self.drained,
            "retired": self.retired, "failure": self.failure,
        }
