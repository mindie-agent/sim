"""Reserve cache access work against current residency.

[Role] Resolve hits, fills, writes, bypass and dirty-victim work for one access.
[Invariants] Reserved lines stay unavailable until the transaction publishes or cancels.
[Timeouts & Locks] Scans resident identities and requested ranges in memory.
[Side-Effect Policy] Reserves cache lines and returns a reversible transaction.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.schema.common import ModelingError

from ..cache_path.geometry import (
    CacheAccessGeometry,
    full_line_spans,
    intersect_spans,
    lines_of_byte_spans,
    range_count,
    subtract_spans,
    take_first_spans,
    union_spans,
)
from .contracts import CacheAccessResult, CachePolicy, _byte_spans_of_lines


def _prepare_access(
    self, identity: str, geometry: CacheAccessGeometry, *,
    write: bool = False, policy: CachePolicy | None = None,
):
    """Reserve an access against current state for event-time execution.

    Existing lines of the same identity move to the reserved set. The
    actual requested line spans (offset/stride holes included) determine
    hit, miss, fill, dirty and RMW work. Victims and their dirty
    responsibility are held until the reservation terminal event; cancel
    restores them if no physical work was issued.
    """
    from .residency import CacheAccessReservation

    policy = policy or CachePolicy()
    if not isinstance(identity, str) or not identity:
        raise ModelingError("cache_access_identity_must_be_nonempty_string")
    if not isinstance(geometry, CacheAccessGeometry):
        raise ModelingError("cache_access_requires_typed_geometry")
    if self._pending_writeback is not None:
        raise ModelingError("cache_state_writeback_in_flight")
    if self._identity_held(identity):
        raise ModelingError("cache_identity_access_in_flight")
    line = self.geometry.line_bytes
    summary = geometry.summary(
        line, write=write, transfer_bytes=self.geometry.transfer_bytes)
    line_count = summary.line_count
    request_ranges = summary.line_segments
    transfer_bytes = summary.gm_bytes if summary.logical_bytes else 0
    staging = (min(transfer_bytes, line * self.geometry.staging_lines)
               if summary.logical_bytes else 0)
    capacity = self.geometry.capacity_lines
    bypass = policy.bypass or policy.cold or capacity == 0
    existing = self._resident.get(identity)
    if line_count == 0:
        result = CacheAccessResult(
            identity=identity, mode="write" if write else "read",
            logical_bytes=0, gm_bytes=0, onchip_bytes=0, staging_bytes=0,
            hbm_read_bytes=0, hbm_write_bytes=0, hbm_read_lines=0,
            hbm_write_lines=0, cache_read_bytes=0, fill_bytes=0,
            evicted_bytes=0, writeback_bytes=0, direct_write_bytes=0,
            hit_lines=0, miss_lines=0, resident_bytes=self._resident_bytes,
            bypassed=bypass, cold_path=policy.cold,
            ready_ms=float(policy.ready_ms),
        )
        return CacheAccessReservation(
            self, identity, 0, 0, target_ranges=(), dirty_ranges=(),
            target_valid_bytes=(), target_dirty_bytes=(),
            existing_ranges=(), existing_dirty=(), existing_valid=(),
            existing_dirty_bytes=(), victims=(),
            result=result, bypassed=True, holder=None, claimed=())
    if bypass:
        return _reserve_bypass(
            self, identity, existing, summary, policy, write,
            line, line_count, transfer_bytes, staging)
    return _reserve_resident(
        self, identity, geometry, existing, summary, policy, write,
        line, line_count, request_ranges, capacity, staging)


def _reserve_bypass(
    self, identity, existing, summary, policy, write,
    line, line_count, transfer_bytes, staging,
):
    from .residency import CacheAccessReservation

    bypass_existing = None
    existing_recency = -1
    existing_reserved_lines = 0
    if existing is not None:
        if existing.dirty_lines:
            raise ModelingError("cache_bypass_conflicts_with_dirty_identity")
        if write:
            # Keep the clean old row as a rollback reservation until
            # this access is irreversibly published or cancelled.
            bypass_existing = replace(existing)
            existing_recency = self._recency.get(identity, 0)
            existing_reserved_lines = existing.lines
    if bypass_existing is not None:
        self._resident.pop(identity)
        self._drop_recency(identity)
        self._resident_bytes -= existing_reserved_lines * line
        self._reserved_bytes += existing_reserved_lines * line
    hbm_read_lines = hbm_write_lines = direct_lines = 0
    if write:
        hbm_write_lines = line_count
        direct_lines = line_count
    else:
        hbm_read_lines = line_count
    hbm_read_lines += summary.rmw_read_lines
    result = CacheAccessResult(
        identity=identity, mode="write" if write else "read",
        logical_bytes=summary.logical_bytes,
        gm_bytes=transfer_bytes, onchip_bytes=transfer_bytes,
        staging_bytes=staging,
        hbm_read_bytes=hbm_read_lines * line,
        hbm_write_bytes=hbm_write_lines * line,
        hbm_read_lines=hbm_read_lines,
        hbm_write_lines=hbm_write_lines,
        cache_read_bytes=0, fill_bytes=0, evicted_bytes=0,
        writeback_bytes=0, direct_write_bytes=direct_lines * line,
        hit_lines=0, miss_lines=line_count,
        resident_bytes=self._resident_bytes + self._reserved_bytes,
        bypassed=True, cold_path=policy.cold,
        ready_ms=float(policy.ready_ms),
    )
    holder = self._new_holder()
    held = (identity,)
    self._claim(holder, held)
    return CacheAccessReservation(
        self, identity, 0, 0, target_ranges=(), dirty_ranges=(),
        target_valid_bytes=(), target_dirty_bytes=(),
        existing_ranges=(() if bypass_existing is None
                         else bypass_existing.ranges),
        existing_dirty=(() if bypass_existing is None
                        else bypass_existing.dirty_ranges),
        existing_valid=(() if bypass_existing is None
                        else bypass_existing.valid_bytes),
        existing_dirty_bytes=(() if bypass_existing is None
                              else bypass_existing.dirty_bytes_ranges),
        victims=(),
        result=result, bypassed=True, holder=holder, claimed=held,
        existing_recency=existing_recency,
        existing_reserved_lines=existing_reserved_lines)


def _reserve_resident(
    self, identity, geometry, existing, summary, policy, write,
    line, line_count, request_ranges, capacity, staging,
):
    from .residency import CacheAccessReservation

    resident_ranges = () if existing is None else existing.ranges
    resident_valid = () if existing is None else existing.valid_bytes
    dirty_lines = () if existing is None else existing.dirty_ranges
    dirty_bytes = () if existing is None else existing.dirty_bytes_ranges
    existing_snapshot = None if existing is None else replace(existing)
    existing_recency = (-1 if existing is None
                        else self._recency.get(identity, 0))
    request_bytes = summary.byte_segments
    request_lines = request_ranges
    resident_total = (self._resident_bytes + self._reserved_bytes) // line
    existing_lines = range_count(resident_ranges)
    free = capacity - resident_total
    max_new = max(0, capacity - existing_lines)
    candidates = subtract_spans(request_lines, resident_ranges)
    need = max(0, min(range_count(candidates), max_new) - max(0, free))
    evicted_lines = victim_writeback = 0
    victims = ()
    if need:
        evicted_lines, victim_writeback, victims = self._evict_ranges(
            need, identity)
    free_after = max(
        0, capacity
        - ((self._resident_bytes - evicted_lines * line
            + self._reserved_bytes) // line))
    retain = take_first_spans(
        candidates, min(range_count(candidates), free_after))
    target_ranges = union_spans(resident_ranges, retain)
    target_lines = range_count(target_ranges)
    target_byte_ranges = _byte_spans_of_lines(target_ranges, line)
    hit_bytes = intersect_spans(request_bytes, resident_valid)
    miss_bytes = subtract_spans(request_bytes, resident_valid)
    hit_lines_ranges = lines_of_byte_spans(hit_bytes, line)
    miss_lines_ranges = lines_of_byte_spans(miss_bytes, line)
    rmw_lines = ()
    if write and geometry.partial_line == "read_modify_write":
        covered = union_spans(request_bytes, resident_valid)
        rmw_lines = subtract_spans(
            request_lines, full_line_spans(covered, line))
    valid_bytes_new = resident_valid
    dirty_bytes_new = dirty_bytes
    hbm_read_lines = hbm_write_lines = direct_lines = 0
    writeback_bytes = victim_writeback
    if write and policy.write_policy == "write_through":
        preflush_lines = subtract_spans(dirty_lines, request_lines)
        preflush = range_count(preflush_lines)
        hbm_write_lines = line_count + preflush
        direct_lines = line_count
        writeback_bytes += preflush * line
        valid_bytes_new = union_spans(
            resident_valid,
            intersect_spans(request_bytes, target_byte_ranges),
            _byte_spans_of_lines(
                intersect_spans(rmw_lines, target_ranges), line))
        dirty_bytes_new = ()
    elif write:
        retained_bytes = intersect_spans(request_bytes,
                                         target_byte_ranges)
        direct_ranges = subtract_spans(request_lines, target_ranges)
        direct_lines = range_count(direct_ranges)
        hbm_write_lines = direct_lines
        valid_bytes_new = union_spans(
            resident_valid, retained_bytes,
            _byte_spans_of_lines(
                intersect_spans(rmw_lines, target_ranges), line))
        dirty_bytes_new = union_spans(
            subtract_spans(dirty_bytes, request_bytes), retained_bytes)
    else:
        hbm_read_lines = range_count(miss_lines_ranges)
        fetched_resident = intersect_spans(miss_lines_ranges,
                                           resident_ranges)
        valid_bytes_new = union_spans(
            resident_valid,
            _byte_spans_of_lines(
                union_spans(retain, fetched_resident), line))
    hbm_read_lines += range_count(rmw_lines)
    if write and policy.write_policy == "write_through":
        hbm_write_lines += victim_writeback // line
    elif write:
        hbm_write_lines += victim_writeback // line
    else:
        hbm_write_lines = victim_writeback // line
    direct_write_bytes = direct_lines * line
    if hbm_write_lines * line != direct_write_bytes + writeback_bytes:
        raise ModelingError("cache_write_conservation_failed")
    drain_bytes = writeback_bytes
    dirty_lines_new = lines_of_byte_spans(dirty_bytes_new, line)
    # All arithmetic above was read-only. Apply the complete plan at once.
    reserved_added = False
    try:
        self._apply_evictions(victims)
        if existing is not None:
            self._resident.pop(identity)
            self._drop_recency(identity)
            self._resident_bytes -= existing.lines * line
        self._reserved_bytes += target_lines * line
        reserved_added = True
        if (self._resident_bytes + self._reserved_bytes
                > self.geometry.capacity_bytes):
            raise ModelingError(
                "cache_state_capacity_accounting_inconsistent")
    except Exception:
        if reserved_added:
            self._reserved_bytes -= target_lines * line
        restorations = [(recency, key, snapshot)
                        for key, recency, snapshot, _updated, _drop, _dirty
                        in victims]
        if existing_snapshot is not None:
            restorations.append(
                (existing_recency, identity, existing_snapshot))
        self._restore_resident_snapshots(self, restorations)
        raise
    holder = self._new_holder()
    held = (identity,) + tuple(item[0] for item in victims)
    self._claim(holder, held)
    result = CacheAccessResult(
        identity=identity, mode="write" if write else "read",
        logical_bytes=summary.logical_bytes,
        gm_bytes=summary.gm_bytes, onchip_bytes=summary.gm_bytes,
        staging_bytes=staging,
        hbm_read_bytes=hbm_read_lines * line,
        hbm_write_bytes=hbm_write_lines * line,
        hbm_read_lines=hbm_read_lines, hbm_write_lines=hbm_write_lines,
        cache_read_bytes=(0 if write else range_count(hit_bytes)),
        fill_bytes=range_count(retain) * line,
        evicted_bytes=evicted_lines * line,
        writeback_bytes=writeback_bytes,
        direct_write_bytes=direct_write_bytes,
        hit_lines=range_count(hit_lines_ranges),
        miss_lines=range_count(miss_lines_ranges),
        resident_bytes=self._resident_bytes + self._reserved_bytes,
        bypassed=False, cold_path=False, ready_ms=float(policy.ready_ms),
    )
    return CacheAccessReservation(
        self, identity, target_lines, range_count(dirty_lines_new),
        target_ranges, dirty_lines_new, valid_bytes_new, dirty_bytes_new,
        resident_ranges, dirty_lines, resident_valid, dirty_bytes,
        victims, result, False, holder, held,
        existing_recency=existing_recency,
        drain_bytes=drain_bytes, drain_completed=not drain_bytes)
