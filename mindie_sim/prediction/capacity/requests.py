"""Request cache admission, resizing and asynchronous lifetimes.

[Role] Manage the logical request demand and its live reader holds.
[Invariants] Retired requests keep physical backing until the corresponding
  consumers release it. Admission includes every retained range.
[Timeouts & Locks] Bounded by active requests, ranks and reader holds. No locks or I/O.
[Side-Effect Policy] Updates request ownership and queues for delayed release.
"""
from __future__ import annotations

from .contracts import _HoldSnapshot, _half_open_ranges


class _MemoryRequestsMixin:
    def _replicate_ranks(self, values) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(values[stage]) for _ in range(self.width))
            for stage in range(self.case.runtime.parallel.pp)
        )

    def _physical_union(
        self, key: tuple[int, str], demand: tuple[float, ...], host_demand: float,
        staging_demand: tuple[float, ...],
        demand_ranks=None, staging_ranks=None,
    ):
        """Physical backing/staging union across ranks, host and stages."""
        p = self.case.runtime.parallel
        if demand_ranks is None:
            demand_ranks = self._replicate_ranks(demand)
        if staging_ranks is None:
            staging_ranks = self._replicate_ranks(staging_demand)
        holds = self.async_consumer_holds.get(key)
        physical_ranks = [
            [float(demand_ranks[stage][rank]) for rank in range(self.width)]
            for stage in range(p.pp)
        ]
        physical_staging_ranks = [
            [float(staging_ranks[stage][rank]) for rank in range(self.width)]
            for stage in range(p.pp)
        ]
        hosts = [float(host_demand)]
        if holds:
            for snapshot in holds.values():
                rank_stages = snapshot.rank_stages or self._replicate_ranks(
                    snapshot.stages)
                rank_staging = snapshot.rank_staging or self._replicate_ranks(
                    snapshot.staging)
                for stage in range(p.pp):
                    for rank in range(self.width):
                        physical_ranks[stage][rank] = max(
                            physical_ranks[stage][rank],
                            float(rank_stages[stage][rank]))
                        physical_staging_ranks[stage][rank] = max(
                            physical_staging_ranks[stage][rank],
                            float(rank_staging[stage][rank]))
                hosts.append(float(snapshot.host_dram))
        physical = tuple(max(row) for row in physical_ranks)
        host_physical = max(hosts)
        physical_staging = tuple(max(row) for row in physical_staging_ranks)
        return (physical, host_physical, physical_staging,
                tuple(tuple(row) for row in physical_ranks),
                tuple(tuple(row) for row in physical_staging_ranks))

    def _set_physical(
        self, replica: int, key: tuple[int, str], physical: tuple[float, ...],
        host_physical: float, staging_physical: tuple[float, ...] | None = None,
        *, physical_ranks=None, staging_ranks=None,
    ) -> None:
        changed = self.owners.get(key) != tuple(physical)
        self.owners[key] = tuple(physical)
        self.owner_ranks[key] = (
            tuple(tuple(row) for row in physical_ranks)
            if physical_ranks is not None else self._replicate_ranks(physical))
        host_before = self.host_dram_owners.get(key, 0.0)
        self.host_dram_owners[key] = float(host_physical)
        changed = changed or float(host_before) != float(host_physical)
        if staging_physical is not None:
            before = self.staging_owners.get(key)
            self.staging_owners[key] = tuple(staging_physical)
            self.staging_owner_ranks[key] = (
                tuple(tuple(row) for row in staging_ranks)
                if staging_ranks is not None else
                self._replicate_ranks(staging_physical))
            changed = changed or before != tuple(staging_physical)
        self._sync_usage()

    def _resident_admission(
        self, replica: int, request: str, tokens: int,
    ) -> dict:
        key = replica, request
        resident_ranks, staging_ranks = self._kv_residency_ranks(tokens)
        demand = tuple(max(row) for row in resident_ranks)
        demand_staging = tuple(max(row) for row in staging_ranks)
        host_demand = self.host_dram_cache_bytes(tokens) if self.sparse_offload else 0.0
        (physical, host_physical, physical_staging,
         physical_ranks, physical_staging_ranks) = self._physical_union(
            key, demand, host_demand, demand_staging,
            demand_ranks=resident_ranks, staging_ranks=staging_ranks)
        p = self.case.runtime.parallel
        request_usage = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        staging_usage = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        for stage in range(p.pp):
            for rank in range(self.width):
                request_usage[replica][stage][rank] = float(
                    physical_ranks[stage][rank])
                staging_usage[replica][stage][rank] = float(
                    physical_staging_ranks[stage][rank])
        host_request = [0.0] * p.dp
        host_request[replica] = float(host_physical)
        hbm, host = self._state_usage(
            exclude_request=key, request_usage=request_usage,
            staging_usage=staging_usage, host_request=host_request)
        return {
            "demand": demand, "physical": physical,
            "physical_ranks": physical_ranks,
            "demand_ranks": tuple(tuple(row) for row in resident_ranks),
            "demand_staging": demand_staging,
            "demand_staging_ranks": tuple(tuple(row) for row in staging_ranks),
            "physical_staging": physical_staging,
            "physical_staging_ranks": physical_staging_ranks,
            "host_demand": float(host_demand),
            "host_physical": float(host_physical),
            "candidate_hbm": hbm, "candidate_host": host,
            "ok": self._usage_budget_ok(hbm, host),
        }

    def admissible_tokens(
        self, replica: int, request: str, tokens: int, *, granularity: int = 1,
    ) -> int:
        """Largest requested KV span that fits without changing the ledger.

        This is a capacity feedback query, not a dispatch decision.  The
        Scheduler/Engine owner decides whether to use a smaller chunk and must
        record the actual chosen span.  A fixed infeasible plan is never
        rewritten to a different dtype/algorithm.  HBM and host DRAM use the
        same candidate calculation as ``resize``, including every physical
        rank's own budget.
        """
        if not 0 <= replica < self.case.runtime.parallel.dp:
            raise ValueError("memory_admission_replica_out_of_range")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("memory_admission_tokens_requires_nonnegative_integer")
        if isinstance(granularity, bool) or not isinstance(granularity, int) or granularity < 1:
            raise ValueError("memory_admission_granularity_requires_positive_integer")
        lo, hi, best = 0, int(tokens), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if mid % granularity:
                mid -= mid % granularity
            if mid < lo:
                break
            candidate = self._resident_admission(replica, request, mid)
            if candidate["ok"]:
                best = max(best, mid)
                lo = mid + granularity
            else:
                hi = mid - granularity
        return int(best)

    def admission_decision(
        self, replica: int, request: str, tokens: int,
    ) -> dict:
        """Pure request-admission decision with the same refusal vocabulary.

        ``admissible_tokens`` answers "how much fits"; this answer separates a
        shortfall that another live holder's real release can repair from a
        demand that can never fit in this replica.  It consumes the identical
        ``_resident_admission`` candidate used by ``resize``.
        """
        if not 0 <= replica < self.case.runtime.parallel.dp:
            raise ValueError("memory_admission_replica_out_of_range")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
            raise ValueError("memory_admission_tokens_requires_nonnegative_integer")
        key = (replica, request)
        candidate = self._resident_admission(replica, request, tokens)
        base_hbm, base_host = self._state_usage(exclude_request=key)
        decision = self._grant_decision_payload(
            replica, candidate["candidate_hbm"], candidate["candidate_host"],
            base_hbm, base_host, [], exclude_request=key)
        decision.update({
            "request": request, "tokens": int(tokens),
            "physical_ranks": candidate["physical_ranks"],
        })
        return decision

    def resize(self, replica: int, request: str, tokens: int) -> bool:
        key = replica, request
        candidate = self._resident_admission(replica, request, tokens)
        if not candidate["ok"]:
            return False
        changed = (
            self.owners.get(key) != candidate["physical"]
            or self.owner_ranks.get(key) != candidate["physical_ranks"]
            or self.logical_owners.get(key) != candidate["demand"]
            or self.host_dram_owners.get(key) != candidate["host_physical"]
            or self.staging_owners.get(key) != candidate["physical_staging"]
        )
        self._set_physical(
            replica, key, candidate["physical"], candidate["host_physical"],
            candidate["physical_staging"],
            physical_ranks=candidate["physical_ranks"],
            staging_ranks=candidate["physical_staging_ranks"])
        self.logical_owners[key] = candidate["demand"]
        self.logical_owner_ranks[key] = candidate["demand_ranks"]
        self.logical_host_owners[key] = candidate["host_demand"]
        self.logical_staging_owners[key] = candidate["demand_staging"]
        self.logical_staging_rank[key] = candidate["demand_staging_ranks"]
        self.request_versions[key] = self.request_versions.get(key, 0) + 1
        if changed:
            self.budget_version += 1
        return True

    def _free_request(self, replica: int, request: str) -> None:
        """Remove the request's physical cache once it is safe to do so."""
        key = (replica, request)
        before = self._replica_occupied(replica)
        self.async_consumer_holds.pop(key, None)
        self.logical_owners.pop(key, None)
        self.logical_owner_ranks.pop(key, None)
        self.logical_host_owners.pop(key, None)
        self.logical_staging_owners.pop(key, None)
        self.logical_staging_rank.pop(key, None)
        self.staging_owners.pop(key, None)
        self.staging_owner_ranks.pop(key, None)
        self.request_versions.pop(key, None)
        self.retired_requests.discard(key)
        self.delayed_owners.pop(key, None)
        sizes = self.owners.pop(key, (0.0,) * len(self.weights))
        self.owner_ranks.pop(key, None)
        self.host_dram_owners.pop(key, None)
        self._sync_usage()
        released = max(0.0, before - self._replica_occupied(replica))
        if any(size > 0.0 for size in sizes):
            self.budget_version += 1
            self.release_events.append({
                "kind": "request", "replica": replica, "request": request,
                "stages": sizes, "released_bytes": released,
            })

    def release(self, replica: int, request: str) -> None:
        """Retire a request without clearing unfinished async consumers.

        A live request's current consumer completing does not retire the
        request; a retired request's final consumer is the actual free point.
        Retirement immediately recomputes the physical union of the remaining
        readers, so an unreferenced old tail is not retained.
        """
        key = (replica, request)
        if key in self.async_consumer_holds and self.async_consumer_holds[key]:
            self.retired_requests.add(key)
            self.logical_owners[key] = (0.0,) * len(self.weights)
            self.logical_owner_ranks[key] = tuple(
                (0.0,) * self.width for _ in range(self.case.runtime.parallel.pp))
            self.logical_host_owners[key] = 0.0
            self.logical_staging_owners[key] = (0.0,) * len(self.weights)
            self.logical_staging_rank[key] = tuple(
                (0.0,) * self.width for _ in range(self.case.runtime.parallel.pp))
            self.delayed_owners.pop(key, None)
            zeros = (0.0,) * len(self.weights)
            (physical, host_physical, physical_staging,
             physical_ranks, staging_ranks) = self._physical_union(
                key, zeros, 0.0, zeros)
            self._set_physical(
                replica, key, physical, host_physical, physical_staging,
                physical_ranks=physical_ranks,
                staging_ranks=staging_ranks)
            self.budget_version += 1
            return
        self._free_request(replica, request)

    def force_release(self, replica: int, request: str) -> None:
        """Cancel/rollback escape hatch: drop holds and free immediately."""
        key = (replica, request)
        self.async_consumer_holds.pop(key, None)
        self.retired_requests.discard(key)
        self._free_request(replica, request)

    def _hold_binding(self, storage, ranges, generation, event) -> tuple:
        return tuple(sorted({
            "storage": storage, "generation": generation, "event": event,
            "ranges": tuple(ranges) if ranges is not None else (),
        }.items(), key=lambda item: item[0]))

    def _retain_typed_readers(self, replica: int, grant: str, consumer: str,
                              event):
        """Register the canonical reader lease for one named grant reader.

        Returns an empty tuple when no runtime ledger is adopted, or the real
        ReaderKey tuple.  Returns ``None`` if a lease cannot be registered so
        the caller leaves its own holder projection unchanged.
        """
        if self.ledger_domain is None:
            return ()
        keys = self._typed_grants.get((replica, grant), ())
        if not keys:
            return ()
        from ..runtime.ledger import Ledger as _Ledger, LedgerError
        reader = event if event is not None else self._ledger_boundary()
        retained = []
        try:
            for allocation in keys:
                retained.append(_Ledger.retain_reader(
                    self.runtime_state, allocation, reader=reader))
        except LedgerError:
            for reader_key in retained:
                try:
                    _Ledger.retire_reader(self.runtime_state, reader_key)
                except LedgerError:
                    pass
            return None
        return tuple(retained)

    def hold_for_async_consumer(
        self, replica: int, request: str, consumer: str, *,
        storage: str | None = None, ranges=None, generation: int | None = None,
        event=None, grant: str | None = None, rank: int | None = None,
    ) -> bool:
        """Register one async consumer that still reads a real holder.

        With ``grant`` the reader references that grant's physical backing, so
        releasing the grant handle does not free bytes before this reader's
        last-use event.  Without it the legacy request-cache hold is unchanged.
        Repeating the same handle/binding is idempotent; a different binding
        under the same consumer name is refused and leaves the first reader
        untouched.
        """
        if not consumer:
            raise ValueError("memory_async_consumer_name_required")
        if grant is not None:
            record = self.grant_records.get((replica, str(grant)))
            if record is None:
                return False
            if rank is not None:
                record_rank = record.get("rank")
                if record_rank is None or int(rank) != int(record_rank):
                    return False
            if generation is not None and record.get("generation") is not None \
                    and int(generation) != int(record.get("generation")):
                return False
            ref_ranges = (
                _half_open_ranges(ranges, record["bytes"])
                if ranges is not None else record.get("_ranges"))
            ref = self._make_ref(
                replica, rank=record.get("rank"), stage=record["stage"],
                storage=record.get("storage"), tier=record.get("tier", "hbm"),
                physical_key=record.get("physical_key"),
                generation=record.get("generation"),
                size=record["bytes"], ranges=ref_ranges,
                host_bytes=record.get("host_bytes", 0.0))
            readers = self.grant_readers.get((replica, str(grant)))
            existing = readers.get(consumer) if readers else None
            if existing is not None:
                return (
                    existing.get("ref") == ref
                    and existing.get("generation") == generation
                    and existing.get("event") == event
                    and existing.get("storage") == storage
                    and tuple(existing.get("ranges") or ())
                    == tuple(ref_ranges or ()))
            holder = (replica, "grant_reader", f"{grant}::{consumer}")
            hbm, host = self._state_usage(extra_holders=[(holder, (ref,))])
            if not self._usage_budget_ok(hbm, host):
                return False
            typed_readers = self._retain_typed_readers(
                replica, str(grant), consumer, event)
            if typed_readers is None:
                return False
            self._set_ref(holder, (ref,))
            self.grant_readers.setdefault(
                (replica, str(grant)), {})[consumer] = {
                "ref": ref, "generation": generation, "event": event,
                "storage": storage, "ranges": tuple(ref_ranges or ()),
                "rank": ref.rank, "readers": typed_readers,
            }
            self._sync_usage()
            self.budget_version += 1
            return True
        key = (replica, request)
        if key not in self.owners:
            return False
        stages = self.logical_owners.get(
            key, self.owners.get(key, (0.0,) * len(self.weights)))
        rank_stages = self.logical_owner_ranks.get(key)
        if rank_stages is None:
            rank_stages = self._replicate_ranks(stages)
        host = self.logical_host_owners.get(
            key, self.host_dram_owners.get(key, 0.0))
        staging = self.logical_staging_owners.get(
            key, (0.0,) * len(self.weights))
        rank_staging = self.logical_staging_rank.get(key)
        if rank_staging is None:
            rank_staging = self._replicate_ranks(staging)
        binding = self._hold_binding(storage, ranges, generation, event)
        snapshot = _HoldSnapshot(
            tuple(stages), float(host), self.request_versions.get(key, 0),
            tuple(staging), binding,
            tuple(tuple(row) for row in rank_stages),
            tuple(tuple(row) for row in rank_staging),
        )
        holds = self.async_consumer_holds.get(key)
        existing = holds.get(consumer) if holds else None
        if existing is not None:
            return existing == snapshot
        self.async_consumer_holds.setdefault(key, {})[consumer] = snapshot
        self.budget_version += 1
        return True

    def release_async_consumer(
        self, replica: int, request: str, consumer: str, *,
        generation: int | None = None, event=None, rank: int | None = None,
    ) -> bool:
        """Complete one consumer; only the real last reference frees bytes."""
        key = (replica, request)
        holds = self.async_consumer_holds.get(key)
        if holds and consumer in holds:
            if rank is not None:
                # A request-cache hold has no single physical rank; a
                # rank-qualified completion belongs to an explicit reader.
                return False
            snapshot = holds[consumer]
            binding = dict(snapshot.binding)
            bound_generation = binding.get("generation")
            if (generation is not None and bound_generation is not None
                    and int(generation) != int(bound_generation)):
                return False
            bound_event = binding.get("event")
            if event is not None and event != bound_event:
                return False
            before = self._replica_occupied(replica)
            del holds[consumer]
            if not holds:
                del self.async_consumer_holds[key]
            if key in self.retired_requests and not holds:
                self._free_request(replica, request)
            elif key in self.owners:
                demand = self.logical_owners.get(key, (0.0,) * len(self.weights))
                demand_ranks = self.logical_owner_ranks.get(key)
                host_demand = self.logical_host_owners.get(key, 0.0)
                staging_demand = self.logical_staging_owners.get(
                    key, (0.0,) * len(self.weights))
                staging_rank = self.logical_staging_rank.get(key)
                (physical, host_physical, physical_staging,
                 physical_ranks, staging_ranks) = self._physical_union(
                    key, demand, host_demand, staging_demand,
                    demand_ranks=demand_ranks, staging_ranks=staging_rank)
                self._set_physical(
                    replica, key, physical, host_physical, physical_staging,
                    physical_ranks=physical_ranks,
                    staging_ranks=staging_ranks)
                self.budget_version += 1
            self.release_events.append({
                "kind": "async_consumer", "replica": replica,
                "request": request, "consumer": consumer,
                "stages": snapshot.stages, "host_dram": snapshot.host_dram,
                "staging": snapshot.staging, "version": snapshot.version,
                "binding": dict(snapshot.binding),
                "released_bytes": max(
                    0.0, before - self._replica_occupied(replica)),
            })
            return True
        readers = self.grant_readers.get((replica, str(request)))
        if readers and consumer in readers:
            snapshot = readers[consumer]
            if rank is not None and int(snapshot["ref"].rank) != int(rank):
                return False
            bound_generation = snapshot.get("generation")
            if (generation is not None and bound_generation is not None
                    and int(generation) != int(bound_generation)):
                return False
            bound_event = snapshot.get("event")
            if event is not None and event != bound_event:
                return False
            if self.ledger_domain is not None:
                from ..runtime.ledger import Ledger as _Ledger, LedgerError
                for reader_key in snapshot.get("readers", ()):
                    try:
                        _Ledger.retire_reader(self.runtime_state, reader_key)
                    except LedgerError:
                        pass
            before = self._replica_occupied(replica)
            self._set_ref((replica, "grant_reader", f"{request}::{consumer}"), ())
            del readers[consumer]
            if not readers:
                del self.grant_readers[(replica, str(request))]
            self._sync_usage()
            self.release_events.append({
                "kind": "grant_reader", "replica": replica,
                "grant": request, "consumer": consumer,
                "binding": snapshot,
                "released_bytes": max(
                    0.0, before - self._replica_occupied(replica)),
            })
            self.budget_version += 1
            return True
        return False

    def release_grant_reader(
        self, replica: int, grant: str, consumer: str, *,
        generation: int | None = None, event=None, rank: int | None = None,
    ) -> bool:
        return self.release_async_consumer(
            replica, grant, consumer, generation=generation, event=event,
            rank=rank)

    def is_retired(self, replica: int, request: str) -> bool:
        return (replica, request) in self.retired_requests

    def pending_async_consumers(self, replica: int, request: str) -> tuple[str, ...]:
        return tuple(sorted(self.async_consumer_holds.get((replica, request), ())))

    def defer_release(self, replica: int, request: str, steps: int | None = None) -> None:
        """Retain KV ownership for finished requests across async scheduling lookahead steps."""
        delay = self.async_delayed_release_steps if steps is None else max(0, int(steps))
        if delay <= 0 or (replica, request) not in self.owners:
            self.release(replica, request)
            return
        self.delayed_owners[replica, request] = (delay, self.owners[replica, request])
        pinned = [0.0] * len(self.weights)
        for (rep, _), (_, sizes) in self.delayed_owners.items():
            if rep == replica:
                pinned = [cur + sz for cur, sz in zip(pinned, sizes)]
        self.peak_delayed_release[replica] = [
            max(old, new) for old, new in zip(self.peak_delayed_release[replica], pinned)
        ]

    def has_delayed_releases(self, replica: int) -> bool:
        return any(rep == replica for rep, _ in self.delayed_owners)

    def step_delayed_releases(self, replica: int) -> None:
        for (rep, req), (rem, sizes) in list(self.delayed_owners.items()):
            if rep != replica:
                continue
            if rem <= 1:
                self.release(replica, req)
            else:
                self.delayed_owners[rep, req] = (rem - 1, sizes)

    def flush_delayed_releases(self, replica: int | None = None) -> None:
        for rep, req in list(self.delayed_owners):
            if replica is None or rep == replica:
                self.release(rep, req)
