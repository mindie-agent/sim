"""Cache residency and transaction lifetimes.

[Role] Track resident ranges, access publication, rollback and writeback.
[Invariants] Reservations own their claims. Dirty lines clear after writeback completes.
[Timeouts & Locks] Reservations and resident ranges are bounded by caller-owned cache
  state.
[Side-Effect Policy] Updates cache state and submits writeback work to the caller's
  Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from mindie_sim.schema.common import ModelingError

from ..cache_path.geometry import (
    CacheAccessGeometry,
    drop_last_spans,
    intersect_spans,
    range_count,
    subtract_spans,
)
from ..engine import Engine, Event
from .access import _prepare_access
from .contracts import (
    CacheAccessJournal,
    CacheAccessResult,
    CacheGeometry,
    CachePolicy,
    _byte_spans_of_lines,
)


@dataclass
class _Resident:
    lines: int
    # Counts remain the compatibility view. Ranges are actual resident/dirty
    # lines; byte ranges distinguish a resident line from a valid byte span.
    dirty_lines: int = 0
    ranges: tuple = ()
    dirty_ranges: tuple = ()
    valid_bytes: tuple = ()
    dirty_bytes_ranges: tuple = ()


class CacheState:
    """Mutable per-level residency, recency and dirty-prefix state.

    Access order is the call order. Each identity denotes one reusable data
    object; repeated accesses to it reuse the retained line prefix. Different
    identities compete for one capacity by least-recent use. A caller resolving
    speculative work should pass ``clone()`` and commit only the selected path.
    """

    def __init__(self, geometry: CacheGeometry):
        if not isinstance(geometry, CacheGeometry):
            raise ModelingError("cache_state_requires_geometry")
        self.geometry = geometry
        self._resident: dict[str, _Resident] = {}
        self._resident_bytes = 0
        # Monotonic last-use ticket per resident identity. Dict insertion order
        # is only a display order; eviction and rollback use this stable rank so
        # overlapping reservations cannot reorder the LRU list.
        self._recency: dict[str, int] = {}
        self._clock = 0
        # Lines reserved by in-flight accesses but not yet published (fill or
        # dirty write). They are not hittable and cannot be evicted.
        self._reserved_bytes = 0
        # identity -> set of unique reservation holders. A holder releases only
        # the identities it actually claimed; a zero-byte reservation claims
        # nothing and can never release another access's responsibility.
        self._inflight: dict[str, set[int]] = {}
        self._holder_sequence = 0
        self._version = 0
        # A pending writeback owns the dirty set. No access/invalidate/reset
        # may change it until the executable writeback completes.
        self._pending_writeback = None
        # One runtime root may own this mutable cache domain. A detached state
        # stays usable standalone; adopting it registers the same object in
        # RuntimeState.cache_domains and never copies residency.
        self.runtime_uid: str | None = None
        self.cache_domain_id: str | None = None
        self.counters = {
            "accesses": 0, "reads": 0, "writes": 0, "hits": 0, "misses": 0,
            "fills_bytes": 0, "evictions": 0, "writeback_bytes": 0,
            "direct_write_bytes": 0, "logical_bytes": 0, "gm_bytes": 0,
            "onchip_bytes": 0, "hbm_read_bytes": 0, "hbm_write_bytes": 0,
        }

    @property
    def version(self) -> int:
        return self._version

    def bind_runtime(self, state, *, domain_id: str | None = None) -> str | None:
        """Adopt this residency state into the one RuntimeState cache domain."""
        uid = getattr(state, "uid", None)
        if not isinstance(uid, str) or not uid:
            raise ModelingError("cache_state_requires_runtime_uid")
        if self.runtime_uid is not None and self.runtime_uid != uid:
            raise ModelingError("cache_runtime_conflict")
        domains = getattr(state, "cache_domains", None)
        if domains is None:
            raise ModelingError("cache_state_runtime_missing_cache_domains")
        key = domain_id or self.cache_domain_id or f"cache:{id(self):x}"
        existing = domains.get(key)
        if existing is not None and existing is not self:
            raise ModelingError(f"cache_domain_conflict:{key}")
        domains[key] = self
        self.runtime_uid = uid
        self.cache_domain_id = key
        return key

    def clone(self) -> "CacheState":
        if self._pending_writeback is not None:
            raise ModelingError("cache_state_writeback_in_flight")
        self._require_no_inflight()
        if self.dirty_bytes:
            # Two independently flushed copies would materialize the same
            # dirty bytes twice. Reject rather than invent alias tracking.
            raise ModelingError("cache_clone_dirty_requires_flush")
        result = CacheState(self.geometry)
        result._resident = {
            key: _Resident(value.lines, value.dirty_lines, value.ranges,
                           value.dirty_ranges, value.valid_bytes,
                           value.dirty_bytes_ranges)
            for key, value in self._resident.items()}
        result._resident_bytes = self._resident_bytes
        result._recency = dict(self._recency)
        result._clock = self._clock
        result._version = self._version
        result.counters = dict(self.counters)
        return result

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    @property
    def resident_identities(self) -> int:
        return len(self._resident)

    @property
    def dirty_bytes(self) -> int:
        return sum(range_count(value.dirty_bytes_ranges)
                   for value in self._resident.values())

    @property
    def inflight_identities(self) -> int:
        return len(self._inflight)

    def resident(self, identity: str) -> tuple[int, int]:
        """Return retained line count and dirty line count."""
        value = self._resident.get(identity)
        return (0, 0) if value is None else (value.lines, value.dirty_lines)

    def resident_ranges(self, identity: str):
        """Return actual retained and dirty line spans."""
        value = self._resident.get(identity)
        return ((), ()) if value is None else (value.ranges, value.dirty_ranges)

    def _require_no_inflight(self) -> None:
        if self._inflight:
            raise ModelingError("cache_access_in_flight")

    def _new_holder(self) -> int:
        self._holder_sequence += 1
        return self._holder_sequence

    def _next_recency(self) -> int:
        self._clock += 1
        return self._clock

    def _touch(self, identity: str) -> None:
        self._recency[identity] = self._next_recency()
        self._reorder_resident()

    def _reorder_resident(self) -> None:
        if len(self._resident) > 1:
            self._resident = dict(sorted(
                self._resident.items(),
                key=lambda item: (self._recency.get(item[0], 0),
                                  item[0])))

    def _drop_recency(self, identity: str) -> None:
        self._recency.pop(identity, None)

    def _claim(self, holder: int, identities) -> None:
        for identity in identities:
            self._inflight.setdefault(identity, set()).add(holder)

    def _release(self, holder: int) -> None:
        for identity in tuple(self._inflight):
            holders = self._inflight[identity]
            holders.discard(holder)
            if not holders:
                del self._inflight[identity]

    def _identity_held(self, identity: str) -> bool:
        return identity in self._inflight

    def reset(self) -> None:
        """Forget all residency explicitly, discarding dirty data."""
        if self._pending_writeback is not None:
            raise ModelingError("cache_state_writeback_in_flight")
        self._require_no_inflight()
        self._resident.clear()
        self._resident_bytes = 0
        self._recency.clear()
        self._version += 1

    def flush(self) -> "CacheWriteback":
        """Return the executable writeback of the current dirty prefix.

        The dirty set is not cleared and counters do not change here. A
        ``CacheWriteback`` owns the dirty set until its Engine task completes;
        while it is pending, accesses/invalidate/reset/clone are rejected.
        """
        if self._pending_writeback is not None:
            raise ModelingError("cache_state_writeback_in_flight")
        self._require_no_inflight()
        lines = sum(value.dirty_lines for value in self._resident.values())
        if not lines:
            return CacheWriteback(self, 0)
        plan = CacheWriteback(self, lines)
        self._pending_writeback = plan
        return plan

    def _finish_writeback(self, plan: "CacheWriteback") -> None:
        if self._pending_writeback is not plan:
            raise ModelingError("cache_writeback_completion_without_pending_plan")
        written = plan.bytes
        for value in self._resident.values():
            value.dirty_lines = 0
            value.dirty_ranges = ()
            value.dirty_bytes_ranges = ()
        self.counters["writeback_bytes"] += written
        self.counters["hbm_write_bytes"] += written
        self._pending_writeback = None
        self._version += 1

    def invalidate(self, identity: str | None = None) -> int:
        """Drop clean residency; reject dirty data until it is flushed.

        Dirty invalidation would silently lose an external effect, so it is
        rejected rather than returned as a counter. Flush the dirty prefix
        first, then invalidate the clean resident object.
        """
        if self._pending_writeback is not None:
            raise ModelingError("cache_state_writeback_in_flight")
        self._require_no_inflight()
        line = self.geometry.line_bytes
        keys = (list(self._resident) if identity is None else
                ([identity] if identity in self._resident else []))
        if any(self._resident[key].dirty_lines for key in keys):
            raise ModelingError("cache_invalidate_dirty_requires_flush")
        removed = 0
        for key in keys:
            value = self._resident.pop(key)
            self._drop_recency(key)
            self._resident_bytes -= value.lines * line
            removed += value.lines
        if keys:
            self._version += 1
        return removed

    def access(self, identity: str, logical_bytes: int, *, write: bool = False,
               policy: CachePolicy | None = None) -> CacheAccessResult:
        """Resolve and commit one contiguous access through the range resolver.

        This is the synchronous adapter for the same byte/line projection used
        by ``prepare_access`` and the executable path; it is not a second
        install mechanism.
        """
        policy = policy or CachePolicy()
        if not isinstance(identity, str) or not identity:
            raise ModelingError("cache_access_identity_must_be_nonempty_string")
        if type(logical_bytes) is not int or logical_bytes < 0:
            raise ModelingError("cache_access_bytes_must_be_nonnegative_integer")
        geometry = (CacheAccessGeometry(rows=1, row_bytes=logical_bytes)
                    if logical_bytes else CacheAccessGeometry(rows=0,
                                                              row_bytes=0))
        reservation = self.prepare_access(identity, geometry, write=write,
                                          policy=policy)
        # The synchronous adapter is one atomic access: its result already
        # accounts the victim/preflush HBM write, so the slot handoff is part
        # of this access rather than a separately executable drain stage.
        reservation.mark_drained()
        reservation.publish()
        reservation.finish()
        return reservation.result

    def _evict_ranges(self, needed_lines, keep):
        """Build a read-only eviction plan; no resident value is changed here."""
        line = self.geometry.line_bytes
        plans = []
        remaining = needed_lines
        ordered = sorted(self._resident.items(),
                         key=lambda item: (self._recency.get(item[0], 0),
                                           item[0]))
        for key, value in ordered:
            if key == keep or self._identity_held(key):
                continue
            if remaining <= 0:
                break
            drop = min(value.lines, remaining)
            if not drop:
                continue
            remaining -= drop
            snapshot = replace(value)
            recency = self._recency.get(key, 0)
            remaining_ranges, dropped = drop_last_spans(value.ranges, drop)
            dirty_dropped = range_count(intersect_spans(
                dropped, value.dirty_ranges))
            dropped_bytes = _byte_spans_of_lines(dropped, line)
            updated = _Resident(
                value.lines - drop, value.dirty_lines - dirty_dropped,
                remaining_ranges,
                subtract_spans(value.dirty_ranges, dropped),
                subtract_spans(value.valid_bytes, dropped_bytes),
                subtract_spans(value.dirty_bytes_ranges, dropped_bytes))
            plans.append((key, recency, snapshot,
                          None if updated.lines == 0 else updated, drop,
                          dirty_dropped))
        if remaining > 0:
            if self._reserved_bytes:
                raise ModelingError("cache_capacity_reserved_exhausted")
            if any(self._identity_held(key) for key in self._resident):
                raise ModelingError("cache_capacity_inflight_held")
            raise ModelingError("cache_state_capacity_accounting_inconsistent")
        evicted = sum(item[4] for item in plans)
        written = sum(item[5] * line for item in plans)
        return evicted, written, tuple(plans)

    def _apply_evictions(self, plans) -> None:
        line = self.geometry.line_bytes
        for key, _recency, _snapshot, updated, drop, _dirty in plans:
            if updated is None:
                self._resident.pop(key, None)
                self._drop_recency(key)
            else:
                self._resident[key] = updated
            self._resident_bytes -= drop * line

    @staticmethod
    def _restore_resident_snapshots(state, restorations) -> None:
        """Restore snapshots with their stable last-use rank, not dict index."""
        line = state.geometry.line_bytes
        for recency, key, snapshot in sorted(
                restorations, key=lambda item: (item[0], str(item[1]))):
            current = state._resident.pop(key, None)
            if current is not None:
                state._resident_bytes -= current.lines * line
            state._resident[key] = snapshot
            state._recency[key] = recency
            state._resident_bytes += snapshot.lines * line
        state._reorder_resident()

    @staticmethod
    def _restore_victims(state, victims):
        CacheState._restore_resident_snapshots(
            state, ((recency, key, snapshot)
                    for key, recency, snapshot, _updated, _drop, _dirty
                    in victims))

    def prepare_access(self, identity: str, geometry: "CacheAccessGeometry", *,
                       write: bool = False,
                       policy: CachePolicy | None = None) -> "CacheAccessReservation":
        """Reserve an access against current state for event-time execution.

        Existing lines of the same identity move to the reserved set. The
        actual requested line spans (offset/stride holes included) determine
        hit, miss, fill, dirty and RMW work. Victims and their dirty
        responsibility are held until the reservation terminal event; cancel
        restores them if no physical work was issued.
        """
        return _prepare_access(self, identity, geometry, write=write, policy=policy)

    def _account(self, result: CacheAccessResult) -> None:
        counters = self.counters
        counters["accesses"] += 1
        counters["reads" if result.mode == "read" else "writes"] += 1
        if result.hit_lines or result.miss_lines:
            counters["hits" if result.hit_lines else "misses"] += 1
        counters["fills_bytes"] += result.fill_bytes
        counters["evictions"] += result.evicted_bytes
        counters["writeback_bytes"] += result.writeback_bytes
        counters["direct_write_bytes"] += result.direct_write_bytes
        counters["logical_bytes"] += result.logical_bytes
        counters["gm_bytes"] += result.gm_bytes
        counters["onchip_bytes"] += result.onchip_bytes
        counters["hbm_read_bytes"] += result.hbm_read_bytes
        counters["hbm_write_bytes"] += result.hbm_write_bytes

    def snapshot(self) -> dict:
        """Bounded export: totals and identity count, never one row per line."""
        return {
            "geometry": {
                "capacity_bytes": self.geometry.capacity_bytes,
                "line_bytes": self.geometry.line_bytes,
                "transfer_bytes": self.geometry.transfer_bytes,
                "staging_lines": self.geometry.staging_lines,
            },
            "resident_bytes": self._resident_bytes,
            "reserved_bytes": self._reserved_bytes,
            "resident_identities": len(self._resident),
            "dirty_bytes": self.dirty_bytes,
            "inflight_identities": len(self._inflight),
            "writeback_in_flight": self._pending_writeback is not None,
            "version": self._version,
            "counters": dict(self.counters),
        }


@dataclass(eq=False)
class CacheAccessReservation:
    """Lines/victims held by one in-flight access; publish installs the target."""

    state: "CacheState"
    identity: str
    target_lines: int
    dirty_lines: int
    target_ranges: tuple
    dirty_ranges: tuple
    target_valid_bytes: tuple
    target_dirty_bytes: tuple
    existing_ranges: tuple
    existing_dirty: tuple
    existing_valid: tuple
    existing_dirty_bytes: tuple
    victims: tuple
    result: "CacheAccessResult"
    bypassed: bool
    holder: int | None = None
    claimed: tuple = ()
    published: bool = False
    finished: bool = False
    # Stable last-use ticket and lines moved from resident to the reversible
    # reservation (clean bypass write).
    existing_recency: int = -1
    existing_reserved_lines: int = 0
    # Dirty victim/preflush bytes that must reach HBM before this target may
    # occupy the same physical slot. The sync adapter and an executable drain
    # stage both close this explicitly.
    drain_bytes: int = 0
    drain_completed: bool = True
    preflushed_ranges: tuple = ()
    journal: CacheAccessJournal | None = None

    def __post_init__(self):
        if self.journal is None:
            line = self.state.geometry.line_bytes
            self.journal = CacheAccessJournal(
                identity=self.identity, holder=self.holder,
                target_lines=self.target_lines,
                reserved_bytes=(self.target_lines * line
                                + self.existing_reserved_lines * line),
                dirty_lines=self.dirty_lines,
                dirty_bytes=range_count(self.target_dirty_bytes),
                drain_bytes=self.drain_bytes,
                direct_bytes=self.result.direct_write_bytes,
                staged_bytes=self.result.staging_bytes)

    @property
    def held_identities(self):
        """Identities claimed by this reservation (zero-byte holds none)."""
        return self.claimed

    def note_issue(self, label: str, resource: str | None,
                   amount: float) -> None:
        self.journal.note_issue(label, resource, amount)

    def mark_drained(self) -> None:
        """Mark the victim/preflush physical handoff as really completed."""
        if not self.drain_completed:
            self.drain_completed = True
            self.journal.note_drain()

    def fail(self, error: Exception) -> None:
        self.journal.failure = f"{type(error).__name__}:{error}"

    def publish(self) -> None:
        if self.finished:
            raise ModelingError("cache_reservation_already_finished")
        if self.published:
            raise ModelingError("cache_reservation_already_published")
        if not self.drain_completed:
            raise ModelingError("cache_reservation_drain_incomplete")
        line = self.state.geometry.line_bytes
        if self.bypassed:
            if self.existing_reserved_lines:
                self.state._reserved_bytes -= (
                    self.existing_reserved_lines * line)
                self.state._version += 1
            self.published = True
            self.journal.reserved_bytes = 0
            self.journal.note_publish()
            return
        if not self.target_lines:
            self.published = True
            self.journal.reserved_bytes = 0
            self.journal.note_publish()
            return
        self.state._reserved_bytes -= self.target_lines * line
        self.state._resident_bytes += self.target_lines * line
        self.state._resident[self.identity] = _Resident(
            self.target_lines, self.dirty_lines, self.target_ranges,
            self.dirty_ranges, self.target_valid_bytes, self.target_dirty_bytes)
        self.state._touch(self.identity)
        self.state._version += 1
        self.published = True
        self.journal.reserved_bytes = 0
        self.journal.note_publish()

    def finish(self) -> None:
        if self.finished:
            raise ModelingError("cache_reservation_already_finished")
        if not self.published:
            raise ModelingError("cache_reservation_finished_before_publish")
        self.state._account(self.result)
        if self.holder is not None:
            self.state._release(self.holder)
        self.state._version += 1
        self.finished = True
        self.journal.note_retire()

    def cancel(self) -> None:
        if self.published or self.finished:
            raise ModelingError("cache_reservation_already_published")
        line = self.state.geometry.line_bytes
        if self.target_lines:
            self.state._reserved_bytes -= self.target_lines * line
        if self.existing_reserved_lines:
            self.state._reserved_bytes -= self.existing_reserved_lines * line
        restorations = []
        if self.existing_ranges:
            # Only the dirty lines that actually drained during preflush are
            # durable in HBM; untouched dirty lines and target-overlapped dirty
            # lines whose direct write never ran remain dirty on rollback.
            if self.drain_completed and self.preflushed_ranges:
                dirty_ranges = subtract_spans(
                    self.existing_dirty, self.preflushed_ranges)
                preflushed_bytes = _byte_spans_of_lines(
                    self.preflushed_ranges, line)
                dirty_bytes = subtract_spans(
                    self.existing_dirty_bytes, preflushed_bytes)
            else:
                dirty_ranges = self.existing_dirty
                dirty_bytes = self.existing_dirty_bytes
            restorations.append((
                self.existing_recency, self.identity,
                _Resident(range_count(self.existing_ranges),
                          range_count(dirty_ranges),
                          self.existing_ranges, dirty_ranges,
                          self.existing_valid, dirty_bytes)))
        if not (self.drain_completed and self.drain_bytes):
            restorations.extend(
                (recency, key, snapshot)
                for key, recency, snapshot, _updated, _drop, _dirty
                in self.victims)
        CacheState._restore_resident_snapshots(self.state, restorations)
        if self.holder is not None:
            self.state._release(self.holder)
        self.state._version += 1
        self.finished = True
        self.journal.reserved_bytes = 0
        self.journal.note_retire()


@dataclass(eq=False)
class CacheWriteback:
    """Executable writeback of one CacheState's dirty prefix.

    Created by ``CacheState.flush()``. It owns the dirty set while pending;
    ``start`` submits real HBM work and clears the dirty prefix/counters only
    on task completion. A plan that is never started leaves the state blocked,
    which is an explicit incomplete external effect rather than a fake clear.
    """

    state: CacheState
    lines: int
    started: bool = False

    @property
    def bytes(self) -> int:
        return self.lines * self.state.geometry.line_bytes

    def start(self, engine: Engine, name: str, ready: Event, *,
              hbm_resource: str = "hbm", owner: str = "") -> Event:
        if not isinstance(engine, Engine):
            raise ModelingError("cache_writeback_requires_engine")
        if self.started:
            raise ModelingError("cache_writeback_already_started")
        if not self.bytes:
            self.started = True
            return engine.submit(name + ":empty", after=(ready,), owner=owner)
        if self.state._pending_writeback is not self:
            raise ModelingError("cache_writeback_is_not_pending")
        if hbm_resource not in engine.resources:
            raise ModelingError(f"cache_writeback_hbm_resource_missing:{hbm_resource}")
        self.started = True
        try:
            event = engine.submit(
                name, {hbm_resource: float(self.bytes)}, after=(ready,),
                on_complete=lambda _: self.state._finish_writeback(self),
                owner=owner,
            )
        except Exception:
            self.started = False
            raise
        return event

    def cancel(self) -> None:
        """Release the pending state without writing or clearing dirty data."""
        if self.started:
            raise ModelingError("cache_writeback_already_started")
        if self.state._pending_writeback is self:
            self.state._pending_writeback = None
