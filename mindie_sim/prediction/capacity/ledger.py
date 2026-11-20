"""Runtime capacity bindings and physical allocation accounting.

[Role] Bind typed pools and charge live physical backing ranges.
[Invariants] Each physical generation is charged for the union of live ranges.
  Budget checks and peak counters use the same committed accounting.
[Timeouts & Locks] Bounded by physical ranks and live holders. No locks or I/O.
[Side-Effect Policy] Mutates this prediction’s capacity domain and usage counters.
"""
from __future__ import annotations

import math

from mindie_sim.catalog.cache_pages import Allocation

from ..runtime.contracts import (
    AllocationBinding,
    AllocationKey,
    BackingKey,
    CapacityDomainState,
    OccurrenceKey,
    PoolKey,
)
from ..runtime.ledger import Ledger
from .contracts import (
    _LedgerBoundary,
    _PhysicalRef,
    _half_open_ranges,
    _hashable_generation,
    _union_bytes,
)


class _MemoryLedgerMixin:
    # ------------------------------------------------------------------
    # Canonical RuntimeState capacity domain
    # ------------------------------------------------------------------
    def ledger_pool(self, replica: int, stage: int, rank: int, *,
                    tier: str = "hbm", domain: str | None = None) -> PoolKey:
        """Physical pool identity for one request allocation rank/stage."""
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp:
            raise ValueError("memory_grant_index_out_of_range")
        if not 0 <= stage < p.pp:
            raise ValueError("memory_stage_out_of_range")
        if not 0 <= rank < self.width:
            raise ValueError("memory_rank_out_of_range")
        domain_id = domain or (self.ledger_domain.domain_id
                               if self.ledger_domain is not None else "memory")
        return PoolKey(physical_domain=domain_id, rank=int(rank), tier=tier,
                       name=f"stage{int(stage)}", unit=int(replica))

    def _ledger_generation(self, value) -> int:
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return int(value)
        key = _hashable_generation(value)
        if key not in self._ledger_generations:
            self._ledger_generations[key] = len(self._ledger_generations)
        return self._ledger_generations[key]

    def _ledger_occurrence(self, replica: int, stage: int, rank: int):
        self._ledger_serial += 1
        return OccurrenceKey(
            runtime_uid=str(getattr(self.runtime_state, "uid", "memory")),
            launch=self._ledger_serial, ordinal=self._ledger_serial,
            replica=int(replica), rank=int(rank), stage=int(stage))

    def _ledger_boundary(self) -> _LedgerBoundary:
        self._ledger_serial += 1
        return _LedgerBoundary(self._ledger_serial)

    def _typed_spec_bindings(self, replica: int, spec: dict):
        """Translate one legacy grant spec onto canonical typed identities.

        The declared ``rank`` is a storage-holder identity, not a new physical
        die: a producer may name a KV owner rank that the placed envelope does
        not enumerate (for example a TP1 case whose binding still names the
        source rank).  Byte accounting already normalises that legacy value
        through ``_account_rank``; the canonical pool/backing identity must use
        the same accounting rank so a grant cannot fail with
        ``memory_rank_out_of_range`` while the legacy ledger admits it.  The
        raw declared rank stays in ``grant_records``/``grant_bindings``.
        """
        if self.ledger_domain is None:
            return ()
        stage = int(spec.get("stage", 0))
        rank = spec.get("rank")
        rank = 0 if rank is None else int(rank)
        account_rank = self._account_rank(rank)
        amount = float(spec.get("bytes", 0.0))
        generation = self._ledger_generation(spec.get("generation"))
        occurrence = self._ledger_occurrence(replica, stage, rank)
        backing = BackingKey(
            physical_domain=self.ledger_domain.domain_id, rank=account_rank,
            tier=str(spec.get("tier", "hbm")),
            storage_id=str(spec.get("physical_key", "")),
            generation=generation)
        ranges = tuple(
            (int(math.floor(lo)), int(math.ceil(hi)))
            for lo, hi in (spec.get("_ranges") or ()))
        if not ranges and amount > 0:
            ranges = ((0, int(math.floor(amount))),)
        key = AllocationKey(
            occurrence=occurrence, scope_kind="invocation",
            scope_ordinal=int(spec.get("stage", 0)), symbol=self._ledger_serial,
            generation=generation)
        result = [AllocationBinding(
            key=key, pool=self.ledger_pool(
                replica, stage, account_rank, tier=str(spec.get("tier", "hbm")),
                domain=self.ledger_domain.domain_id),
            amount=amount, backing=backing, byte_ranges=ranges,
            owner=str(spec.get("owner", "")),
            storage=str(spec.get("storage", "")), access="readwrite")]
        host = float(spec.get("host_bytes", 0.0))
        if host > 0:
            self._ledger_serial += 1
            host_key = AllocationKey(
                occurrence=self._ledger_occurrence(replica, stage, -1),
                scope_kind="external", scope_ordinal=self._ledger_serial,
                symbol=self._ledger_serial, generation=generation)
            result.append(AllocationBinding(
                key=host_key,
                pool=PoolKey(
                    physical_domain=self.ledger_domain.domain_id, rank=-1,
                    tier="host", name="host", unit=int(replica)),
                amount=host,
                backing=BackingKey(
                    physical_domain=self.ledger_domain.domain_id, rank=-1,
                    tier="host",
                    storage_id=str(spec.get("physical_key", "")) + "#host",
                    generation=generation),
                byte_ranges=((0, int(math.floor(host))),),
                owner=str(spec.get("owner", "")),
                storage=str(spec.get("storage", "")), access="readwrite"))
        return tuple(result)

    def bind_runtime(self, state, *, domain_id: str = "memory") -> CapacityDomainState:
        """Adopt a detached ledger domain into the one runtime root.

        Capacities are the same per-rank envelopes used by ``grant_batch``;
        an unbounded host tier is left absent (no configured ceiling) rather
        than encoded as a fake finite number.  Binding a second, different
        runtime to the same live domain raises ``memory_runtime_conflict``.
        """
        if self.ledger_domain is not None and self.runtime_state is not state:
            raise ValueError("memory_runtime_conflict")
        p = self.case.runtime.parallel
        capacities: dict[PoolKey, float] = {}
        for replica in range(p.dp):
            for stage in range(p.pp):
                for rank, cap in enumerate(self.rank_capacity[replica][stage]):
                    if math.isfinite(float(cap)):
                        capacities[self.ledger_pool(
                            replica, stage, rank, domain=domain_id)] = float(cap)
            host = float(self.host_dram_capacity[replica])
            if math.isfinite(host):
                capacities[PoolKey(
                    physical_domain=domain_id, rank=-1, tier="host",
                    name="host", unit=replica)] = host
        domain = Ledger.ensure_domain(
            state, domain_id, capacities=capacities, adopt=False)
        state.claim_capacity_domain(domain)
        self.runtime_state = state
        self.ledger_domain = domain
        return domain

    def typed_binding(
        self, replica: int, name: str, *, tier: str = "hbm",
        rank: int | None = None, stage: int = 0, scope_kind: str = "invocation",
        scope_ordinal: int = 0, generation: int = 0,
        backing_key: str | None = None, ranges=None, amount: float | None = None,
        acquire: bool = True, parent: AllocationKey | None = None,
        access: str = "readwrite", storage: str = "", owner: str = "",
        occurrence: OccurrenceKey | None = None,
    ) -> AllocationBinding:
        """Typed binding for the canonical ledger; names remain provenance."""
        if self.ledger_domain is None:
            raise ValueError("memory_ledger_not_bound")
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp:
            raise ValueError("memory_grant_index_out_of_range")
        record = self.grant_records.get((replica, str(name)), {})
        if rank is None:
            rank = record.get("rank")
            if rank is None:
                rank = 0
        if stage == 0:
            stage = int(record.get("stage", stage))
        if not 0 <= stage < p.pp:
            raise ValueError("memory_stage_out_of_range")
        value = float(record.get("bytes", 0.0) if amount is None else amount)
        domain_id = self.ledger_domain.domain_id
        pool = self.ledger_pool(replica, stage, rank, tier=tier, domain=domain_id)
        occurrence = occurrence or OccurrenceKey(
            runtime_uid=str(getattr(self.runtime_state, "uid", "memory")),
            launch=0, ordinal=scope_ordinal, replica=replica, rank=rank,
            stage=stage)
        key = AllocationKey(
            occurrence=occurrence, scope_kind=scope_kind,
            scope_ordinal=scope_ordinal, symbol=int(scope_ordinal),
            generation=int(generation))
        backing = None
        if backing_key is not None:
            backing = BackingKey(
                physical_domain=domain_id, rank=int(rank), tier=tier,
                storage_id=str(backing_key), generation=int(generation))
        raw = ranges if ranges is not None else record.get("ranges")
        byte_ranges = _half_open_ranges(raw, value)
        return AllocationBinding(
            key=key, pool=pool, amount=value, backing=backing,
            byte_ranges=byte_ranges, reservation_parent=parent,
            owner=str(owner or record.get("owner", "")),
            storage=str(storage or record.get("storage", "")),
            access=access, acquire=acquire)

    def grant_typed(self, bindings, *, holder, at):
        """Atomic typed grant against the adopted RuntimeState domain."""
        if self.ledger_domain is None:
            raise ValueError("memory_ledger_not_bound")
        return Ledger.try_acquire(self.runtime_state, tuple(bindings),
                                  holder=holder, at=at)

    def retain_reader_typed(self, allocation: AllocationKey, *, reader,
                            ranges=(), expected_generation=None):
        if self.ledger_domain is None:
            raise ValueError("memory_ledger_not_bound")
        return Ledger.retain_reader(
            self.runtime_state, allocation, reader=reader, ranges=ranges,
            expected_generation=expected_generation)

    def release_typed(self, allocation: AllocationKey, *, after,
                      expected_generation=None):
        if self.ledger_domain is None:
            raise ValueError("memory_ledger_not_bound")
        return Ledger.release(
            self.runtime_state, allocation, after=after,
            expected_generation=expected_generation)

    def ledger_usage(self) -> dict[PoolKey, float]:
        if self.ledger_domain is None:
            return {}
        return dict(self.ledger_domain.current)

    # ------------------------------------------------------------------
    # Unified physical reference ledger
    # ------------------------------------------------------------------
    def _account_rank(self, rank) -> int:
        """Rank used for byte accounting; legacy out-of-range values clamp to 0."""
        value = 0 if rank is None else int(rank)
        return value if 0 <= value < self.width else 0

    def _make_ref(
        self, replica: int, *, rank, stage, storage, tier, physical_key,
        generation, size, ranges=None, host_bytes=0.0, host_ranges=None,
    ) -> _PhysicalRef:
        if not 0 <= int(stage) < self.case.runtime.parallel.pp:
            raise ValueError("memory_physical_stage_out_of_range")
        tier = str(tier or "hbm").lower()
        ranges = _half_open_ranges(ranges, size)
        host = _half_open_ranges(host_ranges, host_bytes) if host_ranges is not None else (
            ((0.0, float(host_bytes)),) if float(host_bytes) > 0.0 else ())
        if tier == "host":
            host = ranges or host
            ranges = ()
        return _PhysicalRef(
            rank=self._account_rank(rank), stage=int(stage),
            storage=str(storage or ""), tier=tier,
            physical_key=str(physical_key or ""),
            generation=_hashable_generation(generation),
            ranges=tuple(ranges), host_ranges=tuple(host),
        )

    def _set_ref(self, holder: tuple[int, str, str],
                 refs: tuple[_PhysicalRef, ...]) -> None:
        if refs:
            self._refs[holder] = tuple(refs)
        else:
            self._refs.pop(holder, None)

    def _selected_refs(
        self, replica: int, stage: int, items: dict[str, Allocation],
    ) -> dict[str, _PhysicalRef]:
        """One physical reference per selected holder, aliasing explicit roots."""
        by_name = dict(items)
        refs: dict[str, _PhysicalRef] = {}
        for item in items.values():
            source = item
            if item.pool:
                # A suballocation is inside its declared pool reservation.
                # Both are holders on the same backing; the reservation's
                # bytes already cover the suballocation's range.
                pool = by_name.get(item.pool)
                if pool is not None:
                    source = pool
                key = str(getattr(source, "physical_key", None) or item.pool)
            elif item.alias_of:
                seen = set()
                current = item
                while current.alias_of:
                    if current.name in seen:
                        break
                    seen.add(current.name)
                    target = by_name.get(current.alias_of)
                    if target is None:
                        break
                    current = target
                source = current
                key = str(getattr(source, "physical_key", None)
                          or getattr(item, "physical_key", None)
                          or item.alias_of)
            else:
                key = str(getattr(item, "physical_key", None) or item.name)
            ranges = getattr(item, "ranges", None)
            if ranges is None:
                ranges = getattr(source, "ranges", None)
            refs[item.name] = self._make_ref(
                replica, rank=getattr(source, "rank", None),
                stage=stage,
                storage=getattr(source, "storage", None),
                tier=getattr(source, "tier", "hbm"),
                physical_key=key,
                generation=getattr(source, "generation", None),
                size=item.bytes, ranges=ranges,
            )
        return refs

    def _grant_ref(self, spec: dict) -> _PhysicalRef:
        return self._make_ref(
            spec["replica"], rank=spec.get("rank"), stage=spec["stage"],
            storage=spec.get("storage"), tier=spec.get("tier", "hbm"),
            physical_key=spec.get("physical_key") or spec["name"],
            generation=spec.get("generation"), size=spec["bytes"],
            ranges=spec.get("_ranges"),
            host_bytes=spec.get("host_bytes", 0.0),
        )

    def _shared_ref(self, replica: int, physical: str, record: dict):
        refs = []
        for stage, amount in enumerate(record["stages"]):
            if amount <= 0.0:
                continue
            refs.append(self._make_ref(
                replica, rank=record.get("rank"), stage=stage,
                storage=record.get("storage"), tier=record.get("tier", "hbm"),
                physical_key=physical, generation=None,
                size=float(amount),
            ))
        return tuple(refs)

    def _request_usage_ranks(self) -> list[list[list[float]]]:
        result = [
            [[0.0] * self.width for _ in range(self.case.runtime.parallel.pp)]
            for _ in range(self.case.runtime.parallel.dp)
        ]
        for (replica, _request), rows in self.owner_ranks.items():
            for stage, row in enumerate(rows):
                for rank, value in enumerate(row):
                    result[replica][stage][rank] += float(value)
        return result

    def _staging_usage_ranks(self) -> list[list[list[float]]]:
        result = [
            [[0.0] * self.width for _ in range(self.case.runtime.parallel.pp)]
            for _ in range(self.case.runtime.parallel.dp)
        ]
        for (replica, _request), rows in self.staging_owner_ranks.items():
            for stage, row in enumerate(rows):
                for rank, value in enumerate(row):
                    result[replica][stage][rank] += float(value)
        return result

    def _state_usage(
        self, *, extra_holders=(), drop_holders=(), exclude_request=None,
        request_usage=None, staging_usage=None, host_request=None,
    ):
        """Candidate HBM/host usage without mutating any ledger state."""
        p = self.case.runtime.parallel
        hbm = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        host = [0.0] * p.dp
        holders = dict(self._refs)
        for holder in drop_holders:
            holders.pop(holder, None)
        for holder, refs in extra_holders:
            holders[holder] = tuple(refs)
        backing_ranges: dict[tuple, list] = {}
        host_ranges: dict[tuple, list] = {}
        for holder, refs in holders.items():
            replica = holder[0]
            for ref in refs:
                if ref.tier == "host":
                    if ref.host_ranges:
                        host_ranges.setdefault((
                            replica, ref.rank, ref.stage, ref.storage,
                            ref.physical_key, ref.generation), []).extend(
                                ref.host_ranges)
                    continue
                if ref.tier not in {"hbm", ""}:
                    # On-chip/L2/SSD pools are owned by their real allocator;
                    # this ledger only budgets attached HBM and host DRAM.
                    continue
                backing_ranges.setdefault((
                    replica, ref.rank, ref.stage, ref.storage, ref.tier,
                    ref.physical_key, ref.generation), []).extend(ref.ranges)
                if ref.host_ranges:
                    host_ranges.setdefault((
                        replica, ref.rank, ref.stage, ref.storage,
                        ref.physical_key, ref.generation), []).extend(
                            ref.host_ranges)
        for (replica, rank, stage, _storage, _tier, _key, _generation), ranges in \
                backing_ranges.items():
            hbm[replica][stage][rank] += _union_bytes(ranges)
        for (replica, _request), rows in self.owner_ranks.items():
            if (replica, _request) == exclude_request:
                continue
            for stage, row in enumerate(rows):
                for rank, value in enumerate(row):
                    hbm[replica][stage][rank] += float(value)
        for (replica, _request), rows in self.staging_owner_ranks.items():
            if (replica, _request) == exclude_request:
                continue
            for stage, row in enumerate(rows):
                for rank, value in enumerate(row):
                    hbm[replica][stage][rank] += float(value)
        if request_usage is not None:
            for replica, stages in enumerate(request_usage):
                for stage, row in enumerate(stages):
                    for rank, value in enumerate(row):
                        hbm[replica][stage][rank] += float(value)
        if staging_usage is not None:
            for replica, stages in enumerate(staging_usage):
                for stage, row in enumerate(stages):
                    for rank, value in enumerate(row):
                        hbm[replica][stage][rank] += float(value)
        for (replica, request), value in self.host_dram_owners.items():
            if (replica, request) == exclude_request:
                continue
            host[replica] += float(value)
        if host_request is not None:
            for replica, value in enumerate(host_request):
                host[replica] += float(value)
        for (replica, _rank, _stage, _storage, _key, _generation), ranges in \
                host_ranges.items():
            host[replica] += _union_bytes(ranges)
        return hbm, host

    def _usage_budget_ok(self, hbm, host) -> bool:
        for replica in range(self.case.runtime.parallel.dp):
            for stage, row in enumerate(hbm[replica]):
                capacities = self.rank_capacity[replica][stage]
                for rank, used in enumerate(row):
                    cap = capacities[rank]
                    if used > cap + max(1e-9, abs(cap) * 1e-12):
                        return False
            cap = self.host_dram_capacity[replica]
            if host[replica] > cap + max(1e-9, abs(cap) * 1e-12):
                return False
        return True

    def _ref_category_usage(self):
        p = self.case.runtime.parallel
        selected = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        shared = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        granted = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        readers = [
            [[0.0] * self.width for _ in range(p.pp)] for _ in range(p.dp)
        ]
        backings: dict[tuple, dict] = {}
        for holder, refs in self._refs.items():
            kind = holder[1]
            if kind == "grant_reader":
                category = "reader"
            elif kind == "selected":
                category = "selected"
            elif kind == "shared":
                category = "shared"
            elif kind == "grant":
                category = "grant"
            else:
                category = kind
            for ref in refs:
                if ref.tier not in {"hbm", ""} or not ref.ranges:
                    continue
                key = (holder[0], ref.rank, ref.stage, ref.storage,
                       ref.tier, ref.physical_key, ref.generation)
                entry = backings.setdefault(key, {"ranges": [], "kinds": set()})
                entry["ranges"].extend(ref.ranges)
                entry["kinds"].add(category)
        for (replica, rank, stage, _storage, _tier, _key, _generation), entry in \
                backings.items():
            value = _union_bytes(entry["ranges"])
            kinds = entry["kinds"]
            if "selected" in kinds:
                selected[replica][stage][rank] += value
            elif "shared" in kinds:
                shared[replica][stage][rank] += value
            elif "grant" in kinds:
                granted[replica][stage][rank] += value
            elif "reader" in kinds:
                readers[replica][stage][rank] += value
        return selected, shared, granted, readers

    def _sync_usage(self) -> None:
        """Recompute committed counters and peaks from the current holders."""
        p = self.case.runtime.parallel
        hbm, host = self._state_usage()
        request = self._request_usage_ranks()
        staging = self._staging_usage_ranks()
        selected, shared, granted, readers = self._ref_category_usage()
        self.selected_rank = selected
        self.shared_rank = shared
        self.granted_rank = granted
        self.reader_rank = readers
        self.live_rank = request
        self.staging_rank = staging
        for replica in range(p.dp):
            self.live[replica] = [max(row) for row in request[replica]]
            self.staging_live[replica] = float(sum(
                max(row) for row in staging[replica]))
            self.host_dram_live[replica] = float(host[replica])
            self.host_dram_peak[replica] = max(
                self.host_dram_peak[replica], self.host_dram_live[replica])
            for stage in range(p.pp):
                total = max(hbm[replica][stage])
                resident = max(request[replica][stage])
                stage_staging = max(staging[replica][stage])
                self.peak_combined[replica][stage] = max(
                    self.peak_combined[replica][stage], total)
                self.peak[replica][stage] = max(
                    self.peak[replica][stage], resident + stage_staging)
                self.peak_kv_resident[replica][stage] = max(
                    self.peak_kv_resident[replica][stage], resident)
                self.peak_kv_staging[replica][stage] = max(
                    self.peak_kv_staging[replica][stage], stage_staging)
                self.selected_reserved[replica][stage] = max(
                    selected[replica][stage])
                self.shared_reserved[replica][stage] = max(
                    shared[replica][stage])
                self.granted[replica][stage] = max(granted[replica][stage])
        self._rebuild_grant_backings()

    def _rebuild_grant_backings(self) -> None:
        backings: dict[tuple, dict] = {}
        for (replica, name), record in self.grant_records.items():
            binding = self.grant_bindings.get((replica, name), {})
            key = (replica, binding.get("storage"), binding.get("tier"),
                   binding.get("physical_key"), binding.get("rank"))
            row = backings.setdefault(key, {
                "stages": [0.0] * self.case.runtime.parallel.pp,
                "host": 0.0, "names": [],
            })
            for stage, value in enumerate(self.grants.get((replica, name),
                                                          ())):
                row["stages"][stage] = max(row["stages"][stage], float(value))
            row["host"] = max(row["host"], float(
                self.grant_hosts.get((replica, name), 0.0)))
            row["names"].append(name)
        self.grant_backings = backings
        self.granted_host = [0.0] * self.case.runtime.parallel.dp
        for key, row in backings.items():
            self.granted_host[key[0]] += float(row["host"])

    def occupied_bytes(self, replica: int, stage: int, *, rank: int | None = None) -> float:
        """Committed HBM bytes (excluding fixed weights/reserve/workspace)."""
        hbm, _host = self._state_usage()
        if rank is None:
            return float(max(hbm[replica][stage]))
        return float(hbm[replica][stage][self._account_rank(rank)])

    def host_occupied_bytes(self, replica: int) -> float:
        _hbm, host = self._state_usage()
        return float(host[replica])

    def physical_usage(self, replica: int) -> dict:
        """Explain committed per-rank usage from the same candidate ledger."""
        hbm, host = self._state_usage()
        return {
            "replica": replica,
            "hbm_bytes_per_stage_rank": hbm[replica],
            "host_bytes": host[replica],
            "rank_capacity_bytes_per_stage_rank": self.rank_capacity[replica],
            "budget_version": self.budget_version,
        }

    def _replica_occupied(self, replica: int) -> float:
        hbm, _host = self._state_usage()
        return float(sum(max(row) for row in hbm[replica]))

    def _update_combined_peak(self, replica: int, stage: int) -> None:
        """Compatibility wrapper: peaks are committed-state only."""
        self._sync_usage()
