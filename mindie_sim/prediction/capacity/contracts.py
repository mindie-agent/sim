"""Memory policy and physical backing records.

[Role] Define the memory policy and the immutable facts used by the ledger.
[Invariants] Physical identity includes rank, stage, storage, tier, key and generation.
  Token ownership and backing ranges are expressed as half-open spans.
[Timeouts & Locks] Validation is local and bounded by supplied ranges and rank counts.
[Side-Effect Policy] No I/O or mutation of caller-owned records.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ..runtime.contracts import LocalBoundaryToken


@dataclass(frozen=True)
class _LedgerBoundary(LocalBoundaryToken):
    """A real completed boundary token for legacy name-based grant/reader calls."""


@dataclass(frozen=True)
class _HoldSnapshot:
    """One consumer's registered backing range, version and binding facts."""

    stages: tuple[float, ...]
    host_dram: float
    version: int
    staging: tuple[float, ...] = ()
    binding: tuple[tuple[str, object], ...] = ()
    #: The same backing split by physical rank.  Empty keeps the legacy
    #: stage-level call shape; a placement-aware caller fills one row per rank.
    rank_stages: tuple[tuple[float, ...], ...] = ()
    rank_staging: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class _PhysicalRef:
    """One holder's reference to an explicit physical allocation.

    Backing identity is ``(replica, rank, stage, storage, tier, key,
    generation)``.  Two holders with the same identity share one physical
    allocation and their half-open ranges are unioned.  A different live
    generation, or a different key on the same rank/stage, is a second
    allocation and is charged separately.
    """

    rank: int
    stage: int
    storage: str
    tier: str
    physical_key: str
    generation: object
    ranges: tuple[tuple[float, float], ...]
    host_ranges: tuple[tuple[float, float], ...] = ()


def _finite_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_invalid")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}_invalid")
    return number


def _hashable_generation(value):
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _half_open_ranges(value, size: float = 0.0) -> tuple[tuple[float, float], ...]:
    """Normalise explicit half-open byte ranges.

    Accepts ``None`` (meaning ``[0, size)``), a single number, a flat even
    sequence such as ``(0, 4096)``, a flat odd sequence of legacy extents, or a
    sequence of ``(start, end)`` pairs.  Zero-width intervals are dropped.
    """
    size = max(0.0, _finite_number(size, "range_size"))
    if value is None:
        return ((0.0, size),) if size > 0.0 else ()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (value,)
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError("range_sequence_invalid") from exc
    if not items:
        return ((0.0, size),) if size > 0.0 else ()
    if all(isinstance(item, (int, float)) and not isinstance(item, bool)
           for item in items):
        numbers = [_finite_number(item, "range_bound") for item in items]
        if any(number < 0.0 for number in numbers):
            raise ValueError("range_bound_negative")
        if len(numbers) == 1:
            return ((0.0, numbers[0]),) if numbers[0] > 0.0 else ()
        if len(numbers) % 2:
            return tuple((0.0, number) for number in numbers if number > 0.0)
        pairs = [(numbers[index], numbers[index + 1])
                 for index in range(0, len(numbers), 2)]
    else:
        pairs = []
        for item in items:
            try:
                lo, hi = item
            except (TypeError, ValueError) as exc:
                raise ValueError("range_pair_invalid") from exc
            pairs.append((_finite_number(lo, "range_bound"),
                          _finite_number(hi, "range_bound")))
    result = []
    for lo, hi in pairs:
        if lo < 0.0:
            raise ValueError("range_start_negative")
        if hi < lo:
            raise ValueError("range_end_before_start")
        if hi > lo:
            result.append((lo, hi))
    return tuple(result)


def _merge_ranges(ranges) -> tuple[tuple[float, float], ...]:
    ordered = sorted((float(lo), float(hi)) for lo, hi in ranges if hi > lo)
    merged: list[tuple[float, float]] = []
    for lo, hi in ordered:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def _union_bytes(ranges) -> float:
    return float(sum(hi - lo for lo, hi in _merge_ranges(ranges)))


def _normalise_owner_spans(raw, width: int) -> tuple[tuple[int, int], ...] | None:
    """Normalise the T05 token-owner partition for ``width`` physical ranks.

    ``raw`` may be per-rank row counts (implicitly cumulative token spans) or
    explicit half-open ``(start, end)`` token intervals.  ``None`` keeps the
    legacy stage-level behaviour; an explicit partition is never replaced by
    a stage average.
    """
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        raw = raw.get("token_spans") or raw.get("spans") or raw.get("rows")
    if raw is None:
        return None
    items = tuple(raw)
    if len(items) != int(width):
        raise ValueError("memory_token_owner_partition_width_mismatch")
    if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
        spans = []
        start = 0
        for count in items:
            if count < 0:
                raise ValueError("memory_token_owner_partition_negative")
            spans.append((start, start + int(count)))
            start += int(count)
        return tuple(spans)
    spans = []
    for item in items:
        try:
            lo, hi = item
        except (TypeError, ValueError) as exc:
            raise ValueError("memory_token_owner_partition_invalid") from exc
        lo, hi = int(lo), int(hi)
        if lo < 0 or hi < lo:
            raise ValueError("memory_token_owner_partition_invalid")
        spans.append((lo, hi))
    return tuple(spans)


@dataclass(frozen=True)
class MemoryPolicy:
    workspace_bytes: float = 0
    graph_pool_bytes: float = 0
    communication_buffer_bytes: float = 0
    runtime_bytes: float = 0
    derive_workspace: bool = True
    cache_page_groups: tuple[dict, ...] = ()
    async_delayed_release_steps: int | None = None
    async_in_flight_lookahead_steps: int | None = None
    async_workspace_buffer_multiplier: float | None = None
    expert_ddr_ratio: float | None = None
    kvc_ddr_ratio: float | None = None
    kvc_ssd_ratio: float | None = None
    #: Number of implementation workspaces that are simultaneously live
    #: (for example independent streams).  The base workspace is multiplied by
    #: this count; async/expert staging reserves remain their own components.
    concurrent_workspace_copies: int = 1
    #: Explicit transient representation bytes (for example a selected
    #: dequantisation tile).  Zero means "not selected"; it is never inferred
    #: from a dtype or a row size.
    temporary_dequant_bytes: float = 0.0
    #: T05 placement fact: token rows owned by each physical rank, either as
    #: per-rank counts or explicit half-open token spans.  ``None`` keeps the
    #: legacy stage-level split; an explicit uneven partition is never
    #: replaced by a stage average or a group minimum.
    rank_token_owners: tuple | None = None

    @property
    def reserved(self) -> float:
        return sum((self.workspace_bytes, self.graph_pool_bytes,
                    self.communication_buffer_bytes, self.runtime_bytes))
