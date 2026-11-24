"""Transactional physical capacity, range and reader ledger.

[Role] The only implementation of byte-range union accounting, grants,
  reader leases and deferred releases.  Mutable payload lives exclusively in
  ``RuntimeState.capacity_domains``; this module holds no live allocation.
[Invariants] A grant batch commits every component or none.  Physical usage
  is the cardinality of the union of live half-open integer ranges per
  distinct BackingKey (pool slices never add bytes atop a parent reservation).
  Generations are validated on every reader/release.  Rejected transactions
  do not change grants, readers, capacities, current, peaks or the epoch.
[Timeouts & Locks] No I/O, locks or clocks.  Ordered interval lists are
  O(L + q); this bound is exposed through ``usage``/``coverage``.
[Side-Effect Policy] Mutates only the supplied domain's ledger records.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from .contracts import (
    AllocationBinding,
    AllocationKey,
    BackingKey,
    CapacityDomainState,
    EventKey,
    GrantResult,
    PoolKey,
    ReaderKey,
    ReleaseResult,
)

__all__ = [
    "BackingRecord",
    "Ledger",
    "LedgerError",
    "ReaderLease",
    "normalise_ranges",
    "ranges_bytes",
    "union_ranges",
]


class LedgerError(ValueError):
    """Explicit ledger protocol violation with a stable reason token."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def normalise_ranges(
    ranges: Iterable[object] | None,
    *,
    extent: float = 0.0,
) -> tuple[tuple[int, int], ...]:
    """Canonical sorted disjoint half-open integer byte intervals.

    ``None``/empty means ``[0, floor(extent))``.  Adjacent and overlapping
    intervals are merged; zero-width intervals are dropped.  All bounds must
    be finite, non-negative and integral byte positions.
    """
    if extent < 0 or not math.isfinite(float(extent)):
        raise LedgerError("range_extent_invalid")
    if ranges is None:
        limit = int(math.floor(float(extent)))
        return ((0, limit),) if limit > 0 else ()
    items = tuple(ranges)
    if not items:
        limit = int(math.floor(float(extent)))
        return ((0, limit),) if limit > 0 else ()
    parsed: list[tuple[int, int]] = []
    for item in items:
        try:
            lo, hi = item
        except (TypeError, ValueError):
            raise LedgerError("range_pair_invalid") from None
        for bound in (lo, hi):
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise LedgerError("range_bound_invalid")
            if not math.isfinite(float(bound)) or float(bound) < 0:
                raise LedgerError("range_bound_invalid")
            if float(bound) != math.floor(float(bound)):
                raise LedgerError("range_bound_not_integral")
        start, end = int(lo), int(hi)
        if end < start:
            raise LedgerError("range_reversed")
        if end > start:
            parsed.append((start, end))
    if not parsed:
        return ()
    parsed.sort()
    merged: list[tuple[int, int]] = []
    for start, end in parsed:
        if merged and start <= merged[-1][1]:
            previous = merged[-1]
            if end > previous[1]:
                merged[-1] = (previous[0], end)
        else:
            merged.append((start, end))
    return tuple(merged)


def ranges_bytes(ranges: Iterable[tuple[int, int]]) -> int:
    return sum(int(hi) - int(lo) for lo, hi in ranges)


def union_ranges(
    ranges: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    return normalise_ranges(ranges)


def _event_key(state, event, *, generation: int = 0) -> EventKey:
    if event is None:
        raise LedgerError("event_required")
    key = getattr(event, "key", None)
    if isinstance(key, EventKey):
        return key
    sequence = getattr(event, "sequence", None)
    if isinstance(sequence, bool) or not isinstance(sequence, int):
        raise LedgerError("event_sequence_invalid")
    event_uid = getattr(event, "runtime_uid", None)
    if not event_uid:
        raise NotImplementedError(
            "LEGACY_PURGED: wire event-origin runtime_uid on Event/RuntimeState "
            "in prediction/runtime/ledger.py:_event_key (S6-SG-03)"
        )
    return EventKey(
        runtime_uid=str(event_uid), sequence=sequence,
        generation=int(getattr(event, "generation", generation) or generation),
    )


def _event_done(event) -> bool:
    if event is None:
        return False
    if bool(getattr(event, "done", False)):
        return True
    return getattr(event, "time", None) is not None


def _finite2(value, reason: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LedgerError(reason)
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise LedgerError(reason)
    return number


@dataclass(slots=True)
class BackingRecord:
    """One physical backing: canonical union plus its live holders.

    ``segments`` is an ordered disjoint list of ``(lo, hi)``; ``holders`` maps
    an AllocationKey to its normalised ranges.  Bytes are counted once per
    backing regardless of how many holders reference the same range.
    """

    key: BackingKey
    pool: PoolKey
    holders: dict[AllocationKey, tuple[tuple[int, int], ...]] = field(
        default_factory=dict)
    segments: tuple[tuple[int, int], ...] = ()
    parent: AllocationKey | None = None

    @property
    def used_bytes(self) -> int:
        return ranges_bytes(self.segments)

    def coverage(self) -> int:
        return self.used_bytes

    def contains(self, ranges: tuple[tuple[int, int], ...]) -> bool:
        return all(
            any(start >= lo and end <= hi for lo, hi in self.segments)
            for start, end in ranges
        )

    def rebuild(self) -> int:
        parts = [part for value in self.holders.values() for part in value]
        before = self.used_bytes
        self.segments = union_ranges(parts)
        return self.used_bytes - before


@dataclass(slots=True)
class ReaderLease:
    key: ReaderKey
    allocation: AllocationKey
    backing: BackingKey | None
    ranges: tuple[tuple[int, int], ...]
    active: bool = True


@dataclass(slots=True)
class _Grant:
    key: AllocationKey
    pool: PoolKey
    amount: float
    backing: BackingKey | None
    ranges: tuple[tuple[int, int], ...]
    holder: str
    access: str
    storage: str
    acquired_generation: int
    acquire: bool
    reservation_parent: AllocationKey | None
    acquired_at: EventKey
    last_use: EventKey | None = None
    last_use_done: bool = False
    readers: set[ReaderKey] = field(default_factory=set)
    children: set[AllocationKey] = field(default_factory=set)
    children_pool_bytes: float = 0.0
    released: bool = False
    deferred: bool = False



class Ledger:
    """Canonical ledger algorithms operating on ``RuntimeState``.

    Public methods are intentionally narrow: ``try_acquire``, ``retain_reader``
    and ``release`` are the frozen T04 signatures.  ``retire_reader`` and
    ``settle`` complete the reader/deferred lifecycle without another root.
    """

    # ------------------------------------------------------------- domains
    @staticmethod
    def domains(state) -> Mapping[str, CapacityDomainState]:
        domains = getattr(state, "capacity_domains", None)
        if domains is None:
            raise LedgerError("runtime_state_requires_capacity_domains")
        return domains

    @classmethod
    def ensure_domain(
        cls,
        state,
        domain_id: str,
        *,
        capacities: Mapping[PoolKey, float] | None = None,
        adopt: bool = True,
    ) -> CapacityDomainState:
        """Create/return the one domain owned by this runtime or a detached one.

        A domain already owned by another runtime is rejected: a live ledger may
        never be bound to two runtimes.
        """
        if not isinstance(domain_id, str) or not domain_id:
            raise LedgerError("domain_id_invalid")
        domains = cls.domains(state)
        runtime_uid = str(getattr(state, "uid", "") or "")
        domain = domains.get(domain_id)
        if domain is None:
            domain = CapacityDomainState(domain_id=domain_id)
            domains[domain_id] = domain
        owner = domain.owner_runtime_uid
        if adopt and owner is not None and runtime_uid and owner != runtime_uid:
            raise LedgerError("memory_runtime_conflict")
        if adopt and runtime_uid:
            domain.owner_runtime_uid = runtime_uid
        for pool, capacity in dict(capacities or {}).items():
            if not isinstance(pool, PoolKey):
                raise LedgerError("pool_key_required")
            value = _finite2(capacity, "pool_capacity_invalid")
            existing = domain.capacities.get(pool)
            if existing is not None and float(existing) != value:
                raise LedgerError("pool_capacity_conflict")
            domain.capacities[pool] = value
        return domain

    @classmethod
    def adopt(cls, state, domain: CapacityDomainState) -> CapacityDomainState:
        """Adopt a detached Memory domain once, without copying its payload."""
        if not isinstance(domain, CapacityDomainState):
            raise LedgerError("capacity_domain_required")
        domains = cls.domains(state)
        runtime_uid = str(getattr(state, "uid", "") or "")
        owner = domain.owner_runtime_uid
        if owner is not None and runtime_uid and owner != runtime_uid:
            raise LedgerError("memory_runtime_conflict")
        existing = domains.get(domain.domain_id)
        if existing is not None and existing is not domain:
            raise LedgerError("memory_domain_conflict")
        domains[domain.domain_id] = domain
        if runtime_uid:
            domain.owner_runtime_uid = runtime_uid
        return domain

    @staticmethod
    def _domain_for_pool(state, pool: PoolKey) -> CapacityDomainState:
        domains = Ledger.domains(state)
        exact = domains.get(pool.physical_domain)
        if exact is not None:
            return exact
        raise LedgerError("capacity_domain_missing")

    @staticmethod
    def _find_grant(state, allocation: AllocationKey):
        for domain in Ledger.domains(state).values():
            grant = domain.grants.get(allocation)
            if grant is not None:
                return domain, grant
        return None, None

    # -------------------------------------------------------------- grants
    @classmethod
    def try_acquire(
        cls,
        state,
        batch: Sequence[AllocationBinding],
        *,
        holder,
        at,
    ) -> GrantResult:
        """Atomically acquire a complete batch or change nothing.

        Preparation computes every backing-union delta and pool envelope on
        local overlays.  Commit installs holders, usage and peaks and advances
        the domain epoch exactly once.
        """
        items = tuple(batch)
        if not items:
            return GrantResult(granted=True, allocations=())
        allocation_key = _event_key(state, at)
        holder_id = getattr(holder, "runtime_uid", None)
        holder_id = str(holder) if holder_id is None else str(holder)
        keys: set[AllocationKey] = set()
        domains: list[CapacityDomainState] = []
        prepared: list[tuple[CapacityDomainState, _Grant]] = []
        backings: dict[BackingKey, BackingRecord] = {}
        deltas: dict[tuple[str, PoolKey], float] = {}
        blocked: list[PoolKey] = []

        for item in items:
            if not isinstance(item, AllocationBinding):
                raise LedgerError("allocation_binding_required")
            key = item.key
            if not isinstance(key, AllocationKey):
                raise LedgerError("allocation_key_required")
            if key in keys:
                raise LedgerError("allocation_duplicate")
            keys.add(key)
            if not isinstance(item.pool, PoolKey):
                raise LedgerError("pool_key_required")
            domain = cls._domain_for_pool(state, item.pool)
            if domain not in domains:
                domains.append(domain)
            if key in domain.grants:
                raise LedgerError("allocation_already_granted")
            amount = _finite2(item.amount, "allocation_amount_invalid")
            ranges = normalise_ranges(item.byte_ranges, extent=amount)
            if item.backing is not None and not isinstance(
                    item.backing, BackingKey):
                raise LedgerError("backing_key_required")
            grant = _Grant(
                key=key,
                pool=item.pool,
                amount=amount,
                backing=item.backing,
                ranges=ranges,
                holder=str(item.owner or holder_id),
                access=str(item.access),
                storage=str(item.storage),
                acquired_generation=int(key.generation),
                acquire=bool(item.acquire),
                reservation_parent=item.reservation_parent,
                acquired_at=allocation_key,
                last_use=(_event_key(state, item.last_use)
                          if item.last_use is not None else None),
            )
            if item.backing is not None:
                if item.backing.generation != key.generation:
                    raise LedgerError("backing_generation_mismatch")
                record = backings.get(item.backing)
                if record is None:
                    live = domain.backings.get(item.backing)
                    if live is None:
                        record = BackingRecord(key=item.backing, pool=item.pool,
                                               parent=item.reservation_parent)
                    else:
                        if live.pool != item.pool:
                            raise LedgerError("backing_pool_mismatch")
                        record = BackingRecord(
                            key=live.key, pool=live.pool,
                            holders=dict(live.holders),
                            segments=live.segments, parent=live.parent)
                    backings[item.backing] = record
                elif record.pool != item.pool:
                    raise LedgerError("backing_pool_mismatch")
                if record.parent is not None and item.reservation_parent != record.parent:
                    raise LedgerError("backing_parent_mismatch")
                if item.reservation_parent is not None:
                    parent = domain.grants.get(item.reservation_parent)
                    if parent is None or parent.backing != item.backing:
                        raise LedgerError("reservation_parent_invalid")
                    if not ranges:
                        raise LedgerError("reservation_slice_empty")
                    record.holders[key] = ranges
                    new_bytes = record.rebuild()
                    if new_bytes:
                        # A slice inside its parent must never add pool bytes.
                        raise LedgerError("reservation_not_backed")
                    grant.ranges = ranges
                elif grant.acquire:
                    if not ranges and amount > 0:
                        raise LedgerError("backing_ranges_required")
                    record.holders[key] = ranges
                    delta = record.rebuild()
                    slot = (domain.domain_id, item.pool)
                    deltas[slot] = deltas.get(slot, 0.0) + float(delta)
                else:
                    # Externally supplied reference: binding alone never
                    # reacquires bytes or creates a second physical charge.
                    record = None
            else:
                if item.reservation_parent is not None:
                    raise LedgerError("reservation_parent_requires_backing")
                delta = amount if item.acquire else 0.0
                slot = (domain.domain_id, item.pool)
                deltas[slot] = deltas.get(slot, 0.0) + float(delta)
            prepared.append((domain, grant))

        # Validate complete envelope on transaction-local deltas.
        for (domain_id, pool), delta in deltas.items():
            domain = next(item for item in domains if item.domain_id == domain_id)
            capacity = domain.capacities.get(pool)
            current = float(domain.current.get(pool, 0.0))
            if capacity is None:
                capacity = math.inf
            if current + delta > float(capacity) + max(
                    1e-9, abs(float(capacity)) * 1e-12):
                if pool not in blocked:
                    blocked.append(pool)
        if blocked:
            return GrantResult(
                granted=False,
                blocked_pools=tuple(blocked),
                epoch=min(domain.epoch for domain in domains),
                reason="capacity_exceeded",
            )

        # Commit all components.
        for backing, record in backings.items():
            if not record.holders:
                continue
            domain = cls._domain_for_pool(state, record.pool)
            domain.backings[backing] = record
        for domain, grant in prepared:
            domain.grants[grant.key] = grant
            if grant.reservation_parent is not None:
                parent = domain.grants.get(grant.reservation_parent)
                if parent is not None:
                    parent.children.add(grant.key)
        by_domain: dict[str, CapacityDomainState] = {
            domain.domain_id: domain for domain in domains}
        for (domain_id, pool), delta in deltas.items():
            domain = by_domain[domain_id]
            value = float(domain.current.get(pool, 0.0)) + delta
            domain.current[pool] = value
            peak = float(domain.peaks.get(pool, 0.0))
            if value > peak:
                domain.peaks[pool] = value
        for domain in domains:
            domain.epoch += 1
        return GrantResult(
            granted=True,
            allocations=tuple(item.key for item in items),
            epoch=min(domain.epoch for domain in domains),
            reason="granted",
        )

    # ------------------------------------------------------------- readers
    @classmethod
    def retain_reader(
        cls,
        state,
        allocation: AllocationKey,
        *,
        reader,
        ranges=(),
        expected_generation: int | None = None,
    ) -> ReaderKey:
        """Register a real reader lease before the consumer can start."""
        domain, grant = cls._find_grant(state, allocation)
        if grant is None or grant.released:
            raise LedgerError("allocation_not_live")
        if (expected_generation is not None
                and int(expected_generation) != int(allocation.generation)):
            raise LedgerError("generation_mismatch")
        event_key = _event_key(state, reader)
        extent = grant.amount if grant.backing is None else 0.0
        requested = normalise_ranges(ranges, extent=extent)
        if requested:
            if grant.ranges:
                if not all(any(start >= lo and end <= hi
                               for lo, hi in grant.ranges)
                           for start, end in requested):
                    raise LedgerError("reader_ranges_outside_allocation")
            elif sum(end - start for start, end in requested) > grant.amount:
                raise LedgerError("reader_ranges_outside_allocation")
        else:
            requested = grant.ranges
        ordinal = len(grant.readers)
        key = ReaderKey(allocation=allocation, event=event_key, ordinal=ordinal)
        while key in domain.readers:
            ordinal += 1
            key = ReaderKey(allocation=allocation, event=event_key,
                            ordinal=ordinal)
        domain.readers[key] = ReaderLease(
            key=key, allocation=allocation, backing=grant.backing,
            ranges=requested)
        grant.readers.add(key)
        domain.epoch += 1
        return key

    @classmethod
    def retire_reader(cls, state, reader: ReaderKey) -> ReleaseResult:
        """End a real reader lease and complete a matching deferred release."""
        domain, grant = cls._find_grant(state, reader.allocation)
        if grant is None:
            raise LedgerError("allocation_not_live")
        lease = domain.readers.get(reader)
        if lease is None or not lease.active:
            raise LedgerError("reader_not_live")
        lease.active = False
        del domain.readers[reader]
        grant.readers.discard(reader)
        result = ReleaseResult(released=False, epoch=domain.epoch)
        if (grant.deferred and not grant.readers and not grant.children
                and (grant.last_use is None or grant.last_use_done)):
            result = cls._complete_release(state, domain, grant)
        return result

    @classmethod
    def settle(cls, state, completed: Iterable[event] = ()) -> tuple[ReleaseResult, ...]:
        """Complete deferred releases whose last-use events are really done."""
        done = set()
        for event in completed:
            done.add(_event_key(state, event))
        results: list[ReleaseResult] = []
        for domain in cls.domains(state).values():
            for grant in tuple(domain.grants.values()):
                if not grant.deferred or grant.readers or grant.children:
                    continue
                if grant.last_use is not None and grant.last_use not in done:
                    continue
                if grant.last_use is not None:
                    grant.last_use_done = True
                results.append(cls._complete_release(state, domain, grant))
        return tuple(results)

    # -------------------------------------------------------------- release
    @classmethod
    def release(
        cls,
        state,
        allocation: AllocationKey,
        *,
        after,
        expected_generation: int | None = None,
    ) -> ReleaseResult:
        """Release one allocation exactly once; stale generations are rejected."""
        if not isinstance(allocation, AllocationKey):
            raise LedgerError("allocation_key_required")
        domain, grant = cls._find_grant(state, allocation)
        if grant is None:
            return ReleaseResult(released=False, epoch=0)
        if (expected_generation is not None
                and int(expected_generation) != int(allocation.generation)):
            raise LedgerError("generation_mismatch")
        if grant.released:
            return ReleaseResult(released=False, epoch=domain.epoch)
        if after is not None:
            grant.last_use = _event_key(state, after)
            grant.last_use_done = _event_done(after)
        if (grant.children or grant.readers
                or (grant.last_use is not None and not grant.last_use_done)):
            grant.deferred = True
            return ReleaseResult(released=False, deferred=True,
                                 epoch=domain.epoch)
        return cls._complete_release(state, domain, grant)

    @classmethod
    def _complete_release(cls, state, domain, grant) -> ReleaseResult:
        if grant.released:
            return ReleaseResult(released=False, epoch=domain.epoch)
        freed: list[tuple[PoolKey, float]] = []
        if grant.reservation_parent is not None:
            parent = domain.grants.get(grant.reservation_parent)
            if parent is not None:
                parent.children.discard(grant.key)
            if grant.backing is not None:
                record = domain.backings.get(grant.backing)
                if record is not None:
                    record.holders.pop(grant.key, None)
                    record.rebuild()
                    if not record.holders:
                        domain.backings.pop(grant.backing, None)
            grant.released = True
            domain.grants.pop(grant.key, None)
            domain.epoch += 1
            return ReleaseResult(released=True, epoch=domain.epoch)
        if grant.backing is not None:
            record = domain.backings.get(grant.backing)
            if record is not None:
                record.holders.pop(grant.key, None)
                freed_bytes = -record.rebuild()
                if not record.holders:
                    domain.backings.pop(grant.backing, None)
                current = float(domain.current.get(grant.pool, 0.0))
                current = max(0.0, current - max(0.0, freed_bytes))
                domain.current[grant.pool] = current
                if freed_bytes > 0:
                    freed.append((grant.pool, float(freed_bytes)))
            grant.released = True
            domain.grants.pop(grant.key, None)
            domain.epoch += 1
            return ReleaseResult(released=True, epoch=domain.epoch,
                                 freed_by_pool=tuple(freed))
        if grant.acquire:
            current = float(domain.current.get(grant.pool, 0.0))
            current = max(0.0, current - grant.amount)
            domain.current[grant.pool] = current
            if grant.amount:
                freed.append((grant.pool, grant.amount))
        grant.released = True
        domain.grants.pop(grant.key, None)
        domain.epoch += 1
        return ReleaseResult(released=True, epoch=domain.epoch,
                             freed_by_pool=tuple(freed))

    # ---------------------------------------------------------------- views
    @staticmethod
    def usage(state) -> Mapping[str, Mapping[PoolKey, float]]:
        """Immutable public numeric snapshot; never exposes mutable maps."""
        return MappingProxyType({
            domain_id: MappingProxyType(dict(domain.current))
            for domain_id, domain in Ledger.domains(state).items()
        })

    @staticmethod
    def peaks(state) -> Mapping[str, Mapping[PoolKey, float]]:
        return MappingProxyType({
            domain_id: MappingProxyType(dict(domain.peaks))
            for domain_id, domain in Ledger.domains(state).items()
        })

    @staticmethod
    def live(state, domain_id: str | None = None) -> Mapping[AllocationKey, str]:
        domains = Ledger.domains(state)
        selected = ([domains[domain_id]] if domain_id in domains
                    else list(domains.values()))
        return MappingProxyType({
            key: grant.holder
            for domain in selected
            for key, grant in domain.grants.items()
            if not grant.released
        })
