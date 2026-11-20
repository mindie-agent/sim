"""Buffer admission, memory-grant lifecycle, task start/completion and settle.

[Role] Admit ready tasks against per-unit pool capacities and attached memory
  ledgers, publish task starts/completions and drain zero-time settle rounds.
[Invariants] Grants release only at the declared free boundary (or after any
  live ``last_use`` event); settle rounds are bounded by ``zero_progress_limit``.
[Timeouts & Locks] Bounded virtual settle loop; no threads, sleeps, locks or I/O.
[Side-Effect Policy] Mutates Engine allocations, grants, traces and frontier heaps.
"""
from __future__ import annotations

import heapq

from . import frontier
from .contracts import OccurrenceKey
from .engine_types import (
    Allocation, Event, ExecutionError, _Task, _binding_for, _memory_binding,
    _units_to_mask,
)


class _EngineLifecycleMixin:
    def _can_allocate(self, task: _Task) -> bool:
        if not task.acquire and not task.allocation_demands:
            # No live buffer, no pool demand and no grant: admission is exact.
            return True
        allocs = self.allocations
        if allocs:
            for item in task.acquire:
                if item.name in allocs:
                    raise ExecutionError(f"allocation_already_live:{item.name}")
        ledger = task.memory_ledger
        pools = task.memory_pools or self.memory_pools
        prep = task.prepared
        if (prep is not None
                and (ledger is None or prep.acquire_pools.isdisjoint(pools))):
            p_masks = prep.pool_unit_masks
            if p_masks is not None:
                pool_masks = self._pool_unit_masks
                disjoint = True
                for p_name, p_m in p_masks:
                    if pool_masks.get(p_name, 0) & p_m:
                        disjoint = False
                        break
                if disjoint:
                    return prep.fits_empty_pools
            else:
                unit_alloc_fast = self._sync_unit_allocated()
                if (not unit_alloc_fast
                        or prep.demand_keys.isdisjoint(unit_alloc_fast)):
                    return prep.fits_empty_pools
        replica = task.memory_replica
        stage = task.memory_stage
        unit_alloc = self._sync_unit_allocated()
        engine_pools = self.pools
        # Static amounts belong to the prepared stage; only current occupancy
        # is reevaluated as other streams acquire and release the same units.
        for key, amount in task.allocation_demands:
            # The T08 Memory ledger owns the attached HBM/host pools; do not
            # count the same bytes in both Engine and Memory.
            if ledger is not None and key[0] in pools:
                continue
            if unit_alloc.get(key, 0) + amount > engine_pools[key[0]]:
                return False
        if (ledger is not None and not task.memory_granted
                and any(item.pool in pools for item in task.acquire)):
            metadata = task.metadata
            defaults = _memory_binding(metadata)
            bindings = task.memory_bindings or {}
            grants = []
            for item in task.acquire:
                if item.pool not in pools:
                    continue
                spec = {"name": item.name, "bytes": float(item.size)}
                spec.update(defaults)
                override = _binding_for(item.name, bindings)
                if override:
                    spec.update(override)
                spec["stage"] = int(spec.get("stage", stage))
                # A binding may declare the range/bytes in its own physical
                # unit. Engine keeps the declared unit for T08 range checks and
                # never silently rescales a caller's bytes.
                spec.setdefault("replica", replica)
                grants.append(spec)
            if grants:
                # T08's native Memory accepts per-item mappings. A routing
                # ledger (PD endpoint dispatch) may still accept only legacy
                # (name, bytes) pairs plus uniform keywords; use that path
                # only when every item really has the same binding.
                if getattr(ledger, "grant_records", None) is not None:
                    granted = ledger.grant_batch(replica, grants, stage=stage)
                else:
                    uniform = {}
                    for field in ("owner", "storage", "ranges", "generation",
                                  "event", "host_bytes"):
                        values = [spec.get(field) for spec in grants]
                        if any(value != values[0] for value in values[1:]):
                            raise ExecutionError(
                                "memory_ledger_requires_per_item_binding:"
                                + str(field))
                        uniform[field] = values[0]
                    items = [(spec["name"], spec["bytes"]) for spec in grants]
                    granted = ledger.grant_batch(
                        replica, items, stage=stage, **uniform)
                if not granted:
                    return False
                task.memory_granted = True
                task.memory_grant_names = tuple(
                    spec["name"] for spec in grants)
                for spec in grants:
                    name = spec["name"]
                    record = {
                        "replica": replica, "ledger": ledger,
                        "binding": dict(spec),
                        "stage": spec["stage"],
                    }
                    self._memory_grants[name] = record
                self._alloc_state_version += 1
        return True

    def _release_memory_grant(self, name: str) -> None:
        name = str(name)
        record = self._memory_grants.get(name)
        if record is None:
            self._memory_pending.discard(name)
            return
        last_use = record["binding"].get("last_use")
        if isinstance(last_use, Event) and not last_use.done:
            # The logical free happened, but the real last reader/copy is
            # still running. Keep the physical grant and release exactly at
            # that event instead of reusing the block for another request.
            if name not in self._memory_pending:
                self._memory_pending.add(name)
                self.submit(
                    f"memory_last_use:{name}", after=(last_use,),
                    on_complete=lambda _event: self._release_memory_grant(name))
            return
        self._memory_pending.discard(name)
        record = self._memory_grants.pop(name)
        self._alloc_state_version += 1
        ledger = record["ledger"]
        binding = record["binding"]
        release = getattr(ledger, "release_grant", None)
        if callable(release):
            released = release(
                record["replica"], name,
                generation=binding.get("generation"),
                event=binding.get("event"))
        else:
            # Routing ledger used by PD endpoints: it dispatches a real
            # endpoint release by name and has no per-generation overload.
            released = ledger.release_grants(record["replica"], [name])
        if not released:
            # Re-insert so the real holder is not silently forgotten, then
            # surface the mismatch instead of freeing a younger generation.
            self._memory_grants[name] = record
            self._alloc_state_version += 1
            raise ExecutionError(f"memory_release_binding_mismatch:{name}")

    def _complete(self, task: _Task, moment: float | None = None,
                  native: bool = False) -> None:
        key = task.event.sequence
        now = self.now if moment is None else moment
        task.event.time = now
        if self._periodic_runs:
            if key not in self._periodic_tasks:
                self._external_completions += 1
            self._periodic_tasks.pop(key, None)
        elif self._periodic_tasks:
            self._periodic_tasks.pop(key, None)
        # A grant belongs to the live allocation and its real last use, not to
        # the task that happened to acquire it. Release exactly at the free
        # boundary; an acquire-only task keeps its grant until a later free.
        if task.free:
            allocs = self.allocations
            allocated = self.allocated
            mem_grants = self._memory_grants
            mem_pending = self._memory_pending
            has_mem = bool(mem_grants or mem_pending)
            alloc_dq = (self._alloc_trace_dq
                        if self._alloc_trace_maxlen else None)
            tname = task.event.name
            popped_pools = []
            for item in task.free:
                iname = item if item.__class__ is str else item.name
                if has_mem and (iname in mem_grants or iname in mem_pending):
                    self._release_memory_grant(iname)
                live = allocs.pop(iname, None)
                if live is None:
                    raise ExecutionError(f"release_without_live_allocation:{iname}")
                allocated[live.pool] -= live.size * len(live.units)
                popped_pools.append(live)
                if alloc_dq is not None:
                    alloc_dq.append((now, 0, live, tname))
            wave_rec = self._wave_recorder
            if wave_rec is not None:
                for live in popped_pools:
                    for unit in live.units:
                        wave_rec.on_allocation(
                            live.pool, int(unit), 0, 0, int(round(live.size)),
                            now, "release",
                        )
            n_free = len(task.free)
            self.allocation_event_count += n_free
            self._alloc_state_version += n_free
            pool_masks = self._pool_unit_masks
            if not allocs:
                for live in popped_pools:
                    lpool = live.pool
                    allocated[lpool] = 0
                    pool_masks[lpool] = 0
                if self._unit_allocated:
                    self._unit_allocated.clear()
                self._unit_alloc_dirty = False
            elif self._unit_alloc_dirty:
                for live in popped_pools:
                    pool_masks[live.pool] = 0
                for rem_a in allocs.values():
                    if rem_a.size > 0:
                        rpool = rem_a.pool
                        pool_masks[rpool] = (
                            pool_masks.get(rpool, 0) | _units_to_mask(rem_a.units)
                        )
                for live in popped_pools:
                    lpool = live.pool
                    if not pool_masks.get(lpool, 0):
                        allocated[lpool] = 0
            elif (len(allocs) <= 4
                  and sum(len(a.units) for a in allocs.values()) <= 8):
                for live in popped_pools:
                    pool_masks[live.pool] = 0
                unit_alloc = self._unit_allocated
                unit_alloc.clear()
                for rem_a in allocs.values():
                    rpool = rem_a.pool
                    rsize = rem_a.size
                    if rsize > 0:
                        pool_masks[rpool] = (
                            pool_masks.get(rpool, 0) | _units_to_mask(rem_a.units)
                        )
                    for u in rem_a.units:
                        pair = (rpool, u)
                        unit_alloc[pair] = unit_alloc.get(pair, 0) + rsize
                for live in popped_pools:
                    lpool = live.pool
                    if not pool_masks.get(lpool, 0):
                        allocated[lpool] = 0
            else:
                unit_alloc = self._unit_allocated
                for live in popped_pools:
                    lpool = live.pool
                    lsize = live.size
                    clear_mask = 0
                    for unit in live.units:
                        pair = (lpool, unit)
                        new_val = unit_alloc[pair] - lsize
                        if new_val:
                            unit_alloc[pair] = new_val
                        else:
                            del unit_alloc[pair]
                            clear_mask |= 1 << unit
                    if clear_mask:
                        new_m = pool_masks.get(lpool, 0) & ~clear_mask
                        pool_masks[lpool] = new_m
                        if not new_m:
                            allocated[lpool] = 0
        if task.metadata:
            occurrence = task.metadata.get("occurrence")
            if isinstance(occurrence, OccurrenceKey):
                record = self.occurrence_metrics.setdefault(
                    str(occurrence.launch),
                    {"occurrence": occurrence, "submitted": 0, "completed": 0,
                     "service": {}, "started": 0})
                record["completed"] += 1
                for resource, amount in task.service.items():
                    record["service"][resource] = (
                        record["service"].get(resource, 0.0) + amount)
        if key in self._active:
            self._active.discard(key)
            if task.latency > 0.0 and self._latency_active:
                self._latency_active.discard(key)
            if task.coupled:
                self._coupled_active -= 1
            if task.placement:
                self._placed_active -= 1
        if self._completable_seen:
            self._completable_seen.discard(key)
        if self._fifo_live and self.task_uses_fifo(task):
            self._fifo_live -= 1
        if self._inline_blocked and self.task_blocks_native(task):
            self._inline_blocked -= 1
        del self._tasks[key]
        self.completed_count += 1
        t_ml = self._trace_maxlen
        if t_ml:
            tdq = self._trace_dq
            if t_ml > 0 and len(tdq) == t_ml:
                self.trace_dropped += 1
            tdq.append((
                task.event.name, task.owner, task.start, now,
                task.demand, task.service, task.metadata,
                task.placement, task.coupled,
            ))
        if native:
            self._native_slots.pop(key, None)
            self._native_seqs.discard(key)
            if self._native_active > 0:
                self._native_active -= 1
        waiters = task.event._waiters
        if native or self._native_seqs:
            if waiters:
                self._wake(task.event, native=native)
        elif waiters:
            tasks = self._tasks
            ready_seen = self._ready_seen
            ready_heap = self._ready_heap
            releases = self._releases
            if len(waiters) == 1:
                successor = waiters.pop()
                dependent = tasks.get(successor)
                if dependent is not None:
                    w = dependent.waiting - 1
                    dependent.waiting = w
                    if not w:
                        if dependent.release <= now:
                            if successor not in ready_seen:
                                ready_seen.add(successor)
                                heapq.heappush(ready_heap, (dependent.priority, successor))
                        else:
                            heapq.heappush(releases, (dependent.release, successor))
            else:
                for successor in waiters:
                    dependent = tasks.get(successor)
                    if dependent is None:
                        continue
                    dependent.waiting -= 1
                    if not dependent.waiting:
                        if dependent.release <= now:
                            if successor not in ready_seen:
                                ready_seen.add(successor)
                                heapq.heappush(ready_heap, (dependent.priority, successor))
                        else:
                            heapq.heappush(releases, (dependent.release, successor))
                waiters.clear()
        if task.callback is not None:
            task.callback(task.event)

    def _note_allocation(self, action: str, item: Allocation, task: str) -> None:
        self.allocation_event_count += 1
        # Every acquire/release of a live buffer invalidates the cached
        # allocation/grant frontier exactly once.
        self._alloc_state_version += 1
        if self._alloc_trace_maxlen:
            self._alloc_trace_dq.append((
                self.now, 1 if action == "acquire" else 0, item, task,
            ))

    def _start_ready(self, key: int) -> None:
        task = self._tasks[key]
        now = self.now
        task.start = task.event.started = now
        if task.metadata:
            occurrence = task.metadata.get("occurrence")
            if isinstance(occurrence, OccurrenceKey):
                record = self.occurrence_metrics.setdefault(
                    str(occurrence.launch),
                    {"occurrence": occurrence, "submitted": 0, "completed": 0,
                     "service": {}, "started": 0})
                record["started"] += 1
        obs1 = task.start_observer
        if obs1 is not None:
            task.start_observer = None
            if not obs1._waiters and obs1._callbacks is None:
                del self._promises[obs1.sequence]
                obs1.started = obs1.time = now
            else:
                obs1.started = now
                self.resolve(obs1)
        if self._start_waiters:
            for observer in self._start_waiters.pop(key, ()):
                observer.started = now
                self.resolve(observer)
        if task.start_callback is not None:
            # The callback runs before grants so a visibility/contract failure
            # cannot leak a buffer into the caller's engine.
            task.start_callback(task.event)
        if task.acquire:
            prep = task.prepared
            sid = prep.static_id if prep is not None else 0
            pool_masks = self._pool_unit_masks
            p_masks = prep.pool_unit_masks if prep is not None else None
            if sid != 0 and p_masks is not None:
                empty_pools = True
                for p_name, p_m in p_masks:
                    if pool_masks.get(p_name, 0) & p_m:
                        empty_pools = False
                        break
            elif sid != 0:
                unit_alloc_chk = self._sync_unit_allocated()
                empty_pools = (
                    not unit_alloc_chk
                    or prep.demand_keys.isdisjoint(unit_alloc_chk)
                )
            else:
                empty_pools = False
            alloc_dq = (self._alloc_trace_dq
                        if self._alloc_trace_maxlen else None)
            tname = task.event.name
            n_acq = len(task.acquire)
            if empty_pools and p_masks is not None and sid in self._seen_floor_peaks:
                allocs = self.allocations
                allocated = self.allocated
                for item in task.acquire:
                    allocs[item.name] = item
                    allocated[item.pool] += item.size * len(item.units)
                    if alloc_dq is not None:
                        alloc_dq.append((now, 1, item, tname))
                for p_name, p_m in p_masks:
                    pool_masks[p_name] = pool_masks.get(p_name, 0) | p_m
                self._unit_alloc_dirty = True
                self.allocation_event_count += n_acq
                self._alloc_state_version += n_acq
            else:
                allocs = self.allocations
                allocated = self.allocated
                unit_alloc = self._sync_unit_allocated()
                peak_unit = self.peak_unit_allocated
                peak_pool = self.peak_allocated
                for item in task.acquire:
                    allocs[item.name] = item
                    ipool = item.pool
                    isize = item.size
                    allocated[ipool] += isize * len(item.units)
                    if isize > 0:
                        pool_masks[ipool] = (
                            pool_masks.get(ipool, 0) | _units_to_mask(item.units)
                        )
                    p_max = peak_pool[ipool]
                    for unit in item.units:
                        pair = (ipool, unit)
                        val = unit_alloc.get(pair, 0) + isize
                        unit_alloc[pair] = val
                        prev_pu = peak_unit.get(pair)
                        if prev_pu is None or val > prev_pu:
                            peak_unit[pair] = val
                        if val > p_max:
                            p_max = val
                    peak_pool[ipool] = p_max
                    if alloc_dq is not None:
                        alloc_dq.append((now, 1, item, tname))
                self.allocation_event_count += n_acq
                self._alloc_state_version += n_acq
                if empty_pools:
                    self._seen_floor_peaks.add(sid)
            wave_rec = self._wave_recorder
            if wave_rec is not None:
                for item in task.acquire:
                    for unit in item.units:
                        wave_rec.on_allocation(
                            item.pool, int(unit), 0, 0, int(round(item.size)),
                            now, "grant",
                        )
        wave_rec = self._wave_recorder
        if wave_rec is not None:
            owners = tuple(
                sorted({u for units in (task.placement or {}).values() for u in units})
            ) or (0,)
            wave_rec.on_admit(key, now, owners=owners)
        if self._accel is not None and key in self._native_seqs:
            self._accel.activate(self, key)
        if task.demand or task.latency > 0.0:
            self._active.add(key)
            if task.latency > 0.0 and (
                    self._accel is None or key not in self._native_seqs):
                self._latency_active.add(key)
            if task.coupled:
                self._coupled_active += 1
            if task.placement:
                self._placed_active += 1
            if not task.remaining and now >= now + task.latency:
                if key not in self._completable_seen:
                    self._completable_seen.add(key)
                    heapq.heappush(self._completable, key)
        else:
            if key not in self._completable_seen:
                self._completable_seen.add(key)
                heapq.heappush(self._completable, key)

    def _settle(self) -> None:
        self._settle_rounds = 0
        accel = self._accel
        if accel is not None and accel.pending:
            accel.admit(self)
        if self._releases and self._releases[0][0] <= self.now:
            for key in frontier.pop_due_releases(self):
                self._push_ready(key)
        latency_active = self._latency_active
        if latency_active:
            # Only a task that still owes a declared latency can be waiting for
            # the clock here; every zero-latency task is pushed when its work
            # empties, so no full active-set scan is needed.
            now = self.now
            tasks_map = self._tasks
            if len(latency_active) == 1:
                lkey = next(iter(latency_active))
                ltask = tasks_map[lkey]
                if not ltask.remaining and now >= ltask.start + ltask.latency:
                    if lkey not in self._completable_seen:
                        self._completable_seen.add(lkey)
                        heapq.heappush(self._completable, lkey)
            else:
                for key in latency_active:
                    task = tasks_map[key]
                    if not task.remaining and now >= task.start + task.latency:
                        if key not in self._completable_seen:
                            self._completable_seen.add(key)
                            heapq.heappush(self._completable, key)
        ready_heap = self._ready_heap
        ready_seen = self._ready_seen
        tasks = self._tasks
        completable = self._completable
        completable_seen = self._completable_seen
        _start_ready = self._start_ready
        _complete = self._complete
        _heappop = heapq.heappop
        while True:
            deferred: list[int] | None = None
            while ready_heap:
                if len(ready_heap) == 1:
                    _, key = ready_heap.pop()
                else:
                    _, key = _heappop(ready_heap)
                task = tasks.get(key)
                ready_seen.discard(key)
                if task is None or task.start is not None:
                    continue
                if ((task.acquire or task.allocation_demands)
                        and not self._can_allocate(task)):
                    if deferred is None:
                        deferred = [key]
                    else:
                        deferred.append(key)
                    continue
                _start_ready(key)
            if deferred is not None:
                for key in deferred:
                    self._push_ready(key)
            if not completable:
                return
            # Drain the snapshot before starting tasks woken by these callbacks.
            # Lower sequence numbers still complete first, matching sorted order.
            if len(completable) == 1:
                key = completable.pop()
                completable_seen.discard(key)
                task = tasks.get(key)
                if (task is not None and not task.remaining
                        and self.now >= task.start + task.latency):
                    _complete(task)
            else:
                for key in frontier.take_completable(self):
                    task = tasks.get(key)
                    if task is None or task.remaining or self.now < task.start + task.latency:
                        continue
                    _complete(task)
            # A completion callback may submit another zero-work successor at
            # the same timestamp. Count rounds, not just time steps, so such a
            # chain cannot spin forever without advancing the clock.
            self._settle_rounds += 1
            if self._settle_rounds > self.zero_progress_limit:
                raise ExecutionError("execution_zero_progress")

    def _split_asymmetric_cohorts(self) -> bool:
        """Split active widened cohorts when a peer breaks member unit symmetry."""
        if self._placed_active == 0 or len(self._active) < 2:
            return False
        to_split: list[int] = []
        tasks = self._tasks
        active = self._active
        for key in active:
            task = tasks[key]
            if task.cohort_size <= 1 or not task.cohort_members or not task.remaining:
                continue
            members_dict = dict(task.cohort_members)
            broken = False
            for peer_key in active:
                if peer_key == key:
                    continue
                peer = tasks[peer_key]
                if not peer.placement or not peer.remaining:
                    continue
                for rname, members in members_dict.items():
                    if rname not in task.remaining or peer.remaining.get(rname, 0.0) <= 0.0:
                        continue
                    peer_units = peer.placement.get(rname)
                    if peer_units is None:
                        continue
                    peer_set = set(peer_units)
                    if len({len(peer_set.intersection(m)) for m in members}) > 1:
                        broken = True
                        break
                if broken:
                    break
            if broken:
                to_split.append(key)
        if not to_split:
            return False
        from .engine_types import _EMPTY_FROZENSET
        for key in to_split:
            task = tasks[key]
            cohort = task.cohort_size
            members_dict = dict(task.cohort_members)
            active.discard(key)
            if task.latency > 0.0:
                self._latency_active.discard(key)
            if task.coupled:
                self._coupled_active -= 1
            if task.placement:
                self._placed_active -= 1
            remaining_members = [cohort]
            orig_task = task

            def _member_done(member_ev, m_task_ref, orig=orig_task, rem_box=remaining_members):
                for rn, amt in m_task_ref.service.items():
                    orig.service[rn] = orig.service.get(rn, 0.0) + amt
                rem_box[0] -= 1
                if rem_box[0] == 0:
                    orig.remaining.clear()
                    self._complete(orig, moment=member_ev.time)

            for idx in range(cohort):
                m_rem = {r: amt / cohort for r, amt in task.remaining.items()}
                m_caps = {r: cap / cohort for r, cap in task.caps.items()}
                m_place = {
                    r: (members_dict[r][idx] if r in members_dict else units)
                    for r, units in task.placement.items()
                }
                self._sequence += 1
                m_ev = Event(self._sequence, f"{task.event.name}#member{idx}", started=task.start)
                m_prep = self._prepare(
                    m_ev.name, m_rem, latency=task.latency, caps=m_caps,
                    placement=m_place, coupled=task.coupled, priority=task.priority,
                )
                m_task = _Task(
                    m_ev, dict(m_rem), m_prep.demand, 0, task.release,
                    task.latency, m_prep.caps, (), (), task.owner, task.metadata,
                    m_prep.placement, (), task.priority, task.coupled, None, None,
                    task.start, {}, False, None, 0, 0, _EMPTY_FROZENSET, None, (),
                    1, m_prep, None, False, (), (),
                )
                m_task.callback = lambda ev, mt=m_task, fn=_member_done: fn(ev, mt)
                tasks[m_ev.sequence] = m_task
                if self.task_uses_fifo(m_task):
                    self._fifo_live += 1
                if self.task_blocks_native(m_task):
                    self._inline_blocked += 1
                active.add(m_ev.sequence)
                if m_task.latency > 0.0:
                    self._latency_active.add(m_ev.sequence)
                if m_task.coupled:
                    self._coupled_active += 1
                if m_task.placement:
                    self._placed_active += 1
        return True
