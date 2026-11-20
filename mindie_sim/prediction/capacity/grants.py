"""Candidate admission and lifetime tracking for memory grants.

[Role] Preflight, commit and release named allocation grants.
[Invariants] Failed requests leave the ledger intact. A batch is committed only
  when its entire physical allocation union fits the budget.
[Timeouts & Locks] Bounded by the candidate batch and live grants. No locks or I/O.
[Side-Effect Policy] Updates grant holders and release events on the prediction ledger.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from ..runtime.contracts import AllocationKey
from .contracts import _half_open_ranges


class _MemoryGrantsMixin:
    def _grant_item(self, item, defaults: dict) -> dict | None:
        if isinstance(item, Mapping):
            raw = dict(item)
            name = raw.pop("name", None)
            amount = raw.pop("bytes", raw.pop("size", None))
        else:
            try:
                name, amount = item
            except (TypeError, ValueError):
                return None
            raw = {}
        name = str(name or "")
        if not name or isinstance(amount, bool) or not isinstance(amount, (int, float)):
            return None
        value = float(amount)
        if not math.isfinite(value) or value < 0.0:
            return None
        raw_ranges = raw.get("ranges", defaults["ranges"])
        try:
            ref_ranges = _half_open_ranges(raw_ranges, value)
        except ValueError:
            return None
        host = raw.get("host_bytes", defaults["host_bytes"])
        if isinstance(host, bool) or not isinstance(host, (int, float)):
            return None
        host = float(host)
        if not math.isfinite(host) or host < 0.0:
            return None
        rank = raw.get("rank", defaults["rank"])
        if rank is not None:
            if isinstance(rank, bool) or not isinstance(rank, int):
                return None
            if rank < 0:
                return None
        stage = int(raw.get("stage", defaults["stage"]))
        if not 0 <= stage < self.case.runtime.parallel.pp:
            return None
        result = {
            "name": name, "bytes": value,
            "owner": raw.get("owner", defaults["owner"]),
            "storage": raw.get("storage", defaults["storage"]),
            "tier": str(raw.get("tier", defaults["tier"]) or "hbm"),
            "rank": rank,
            "stage": stage,
            "generation": raw.get("generation", defaults["generation"]),
            "ranges": raw_ranges,
            "event": raw.get("event", defaults["event"]),
            "host_bytes": host,
            "physical_key": str(raw.get("physical_key", name) or name),
            "_ranges": ref_ranges,
        }
        return result

    def _grant_candidate(
        self, replica: int, grants, *, stage: int = 0,
        owner: str | None = None, storage: str | None = None,
        ranges=None, generation: int | None = None, event=None,
        host_bytes: float = 0.0,
    ) -> tuple[list[dict], dict]:
        """Resolve one batch into a pure decision plus the parsed holders."""
        p = self.case.runtime.parallel
        if not 0 <= replica < p.dp:
            raise ValueError("memory_grant_index_out_of_range")
        defaults = {
            "owner": owner, "storage": storage,
            "tier": "hbm", "rank": None, "stage": int(stage),
            "generation": generation,
            "ranges": tuple(ranges) if ranges is not None else (),
            "event": event, "host_bytes": host_bytes,
        }
        parsed = []
        names = set()
        for item in grants:
            spec = self._grant_item(item, defaults)
            if (spec is None or (replica, spec["name"]) in self.grants
                    or spec["name"] in names):
                return [], {
                    "ok": False, "reason": "invalid_or_duplicate",
                    "replica": replica,
                    "budget_version": self.budget_version,
                    "requested": (), "wakeable": False, "permanent": False,
                    "shortfall_bytes_per_stage_rank": [],
                    "releasable_bytes_per_stage_rank": [],
                    "host_shortfall_bytes": 0.0,
                    "host_releasable_bytes": 0.0,
                    "wake_events": (),
                }
            names.add(spec["name"])
            spec["replica"] = replica
            parsed.append(spec)
        if not parsed:
            return [], {
                "ok": True, "reason": "empty", "replica": replica,
                "budget_version": self.budget_version,
                "requested": (), "wakeable": False, "permanent": False,
                "shortfall_bytes_per_stage_rank": [
                    [0.0] * self.width for _ in range(p.pp)],
                "releasable_bytes_per_stage_rank": [
                    [0.0] * self.width for _ in range(p.pp)],
                "host_shortfall_bytes": 0.0,
                "host_releasable_bytes": 0.0,
                "wake_events": (),
            }
        new_holders = [
            ((replica, "grant", spec["name"]), (self._grant_ref(spec),))
            for spec in parsed
        ]
        hbm, host = self._state_usage(extra_holders=new_holders)
        current_hbm, current_host = self._state_usage()
        return parsed, self._grant_decision_payload(
            replica, hbm, host, current_hbm, current_host, parsed)

    def _grant_decision_payload(
        self, replica: int, hbm, host, current_hbm, current_host,
        parsed: list[dict], *, exclude_request: tuple[int, str] | None = None,
    ) -> dict:
        """Explain an atomic grant decision without mutating the ledger."""
        p = self.case.runtime.parallel
        ok = self._usage_budget_ok(hbm, host)
        shortfall = [[0.0] * self.width for _ in range(p.pp)]
        releasable = [[0.0] * self.width for _ in range(p.pp)]
        recoverable = True
        for stage in range(p.pp):
            capacities = self.rank_capacity[replica][stage]
            for rank in range(self.width):
                cap = capacities[rank]
                used = float(hbm[replica][stage][rank])
                base = max(0.0, float(current_hbm[replica][stage][rank]))
                if used > cap + max(1e-9, abs(cap) * 1e-12):
                    shortfall[stage][rank] = used - cap
                    releasable[stage][rank] = base
                    if shortfall[stage][rank] > base + max(
                            1e-9, abs(cap) * 1e-12):
                        recoverable = False
        host_cap = self.host_dram_capacity[replica]
        host_used = float(host[replica])
        host_base = max(0.0, float(current_host[replica]))
        host_shortfall = 0.0
        host_releasable = 0.0
        if host_used > host_cap + max(1e-9, abs(host_cap) * 1e-12):
            host_shortfall = host_used - host_cap
            host_releasable = host_base
            if host_shortfall > host_base + max(1e-9, abs(host_cap) * 1e-12):
                recoverable = False
        wakeable = (not ok) and recoverable
        permanent = (not ok) and not recoverable
        wake_events = []
        if not ok:
            for (rep, name), binding in sorted(self.grant_bindings.items()):
                if rep != replica or binding.get("event") is None:
                    continue
                wake_events.append({
                    "kind": "grant", "name": name,
                    "stage": binding.get("stage"),
                    "rank": binding.get("rank"),
                    "event": binding["event"],
                })
            for (rep, grant), readers in sorted(self.grant_readers.items()):
                if rep != replica:
                    continue
                for consumer, snapshot in sorted(readers.items()):
                    if snapshot.get("event") is None:
                        continue
                    wake_events.append({
                        "kind": "grant_reader", "grant": grant,
                        "consumer": consumer,
                        "event": snapshot["event"],
                    })
            for (rep, request), holds in sorted(
                    self.async_consumer_holds.items()):
                if rep != replica or (rep, request) == exclude_request:
                    continue
                for consumer, snapshot in sorted(holds.items()):
                    binding = dict(snapshot.binding)
                    if binding.get("event") is None:
                        continue
                    wake_events.append({
                        "kind": "async_consumer", "request": request,
                        "consumer": consumer, "event": binding["event"],
                    })
        return {
            "ok": ok,
            "reason": ("admitted" if ok else
                       "waitable_release" if wakeable else
                       "permanent_budget"),
            "replica": replica,
            "budget_version": self.budget_version,
            "requested": tuple(spec["name"] for spec in parsed),
            "wakeable": wakeable,
            "permanent": permanent,
            "shortfall_bytes_per_stage_rank": shortfall,
            "releasable_bytes_per_stage_rank": releasable,
            "host_shortfall_bytes": host_shortfall,
            "host_releasable_bytes": host_releasable,
            "wake_events": tuple(wake_events),
        }

    def grant_decision(
        self, replica: int, grants, *, stage: int = 0,
        owner: str | None = None, storage: str | None = None,
        ranges=None, generation: int | None = None, event=None,
        host_bytes: float = 0.0,
    ) -> dict:
        """Pure admission decision for ``grant_batch``.

        The result separates a refusal that a real holder release can repair
        from a permanent demand/available deficit.  Known holder Events are
        returned so a scheduler can wait on the actual completion instead of a
        timer.  This method never changes refs, peaks, version or callbacks.
        """
        _parsed, decision = self._grant_candidate(
            replica, grants, stage=stage, owner=owner, storage=storage,
            ranges=ranges, generation=generation, event=event,
            host_bytes=host_bytes)
        return decision

    def grant_batch(
        self, replica: int, grants, *, stage: int = 0,
        owner: str | None = None, storage: str | None = None,
        ranges=None, generation: int | None = None, event=None,
        host_bytes: float = 0.0,
    ) -> bool:
        """Atomically grant per-item physical references in one budget.

        Each item may be the legacy ``(name, bytes)`` tuple or a mapping with
        ``name/bytes/owner/storage/tier/rank/stage/generation/ranges/event/
        host_bytes/physical_key``.  Uniform keyword defaults preserve old
        calls.  All holders in the call are resolved into one candidate:
        same-backing ranges are unioned, distinct live generations or keys are
        charged separately, and every physical rank/stage plus host is checked
        before one commit.  Failure changes no ledger, peak, version or
        callback.  ``grant_decision`` exposes the same candidate decision.
        """
        p = self.case.runtime.parallel
        parsed, decision = self._grant_candidate(
            replica, grants, stage=stage, owner=owner, storage=storage,
            ranges=ranges, generation=generation, event=event,
            host_bytes=host_bytes)
        if not decision["ok"]:
            return False
        typed: dict[tuple[int, str], tuple[AllocationKey, ...]] = {}
        if self.ledger_domain is not None:
            from ..runtime.ledger import Ledger as _Ledger
            plan = []
            for spec in parsed:
                bindings = self._typed_spec_bindings(replica, spec)
                if bindings:
                    plan.append((spec["name"], bindings))
            flat = tuple(
                binding for _name, bindings in plan for binding in bindings)
            if flat:
                grant = _Ledger.try_acquire(
                    self.runtime_state, flat,
                    holder=self._ledger_occurrence(
                        replica, int(parsed[0]["stage"]),
                        int(parsed[0].get("rank") or 0)),
                    at=self._ledger_boundary())
                if not grant.granted:
                    return False
                granted = set(grant.allocations)
                for name, bindings in plan:
                    typed[(replica, name)] = tuple(
                        binding.key for binding in bindings
                        if binding.key in granted)
        for spec in parsed:
            spec_ref = self._grant_ref(spec)
            self._set_ref((replica, "grant", spec["name"]), (spec_ref,))
            self.grants[(replica, spec["name"])] = tuple(
                spec["bytes"] if index == spec["stage"] else 0.0
                for index in range(p.pp))
            self.grant_hosts[(replica, spec["name"])] = spec["host_bytes"]
            self.grant_bindings[(replica, spec["name"])] = {
                "owner": spec["owner"], "storage": spec["storage"],
                "tier": spec["tier"], "rank": spec["rank"],
                "stage": spec["stage"], "ranges": spec["ranges"],
                "generation": spec["generation"], "event": spec["event"],
                "host_bytes": spec["host_bytes"],
                "physical_key": spec["physical_key"],
            }
            self.grant_records[(replica, spec["name"])] = dict(spec)
        if parsed:
            self._sync_usage()
            self.budget_version += 1
        self._typed_grants.update(typed)
        return True

    def _remove_grant(
        self, replica: int, name: str, binding: dict, *,
        sync: bool = True,
    ) -> float:
        typed = self._typed_grants.pop((replica, str(name)), ())
        if typed and self.ledger_domain is not None:
            from ..runtime.ledger import Ledger as _Ledger
            for allocation in typed:
                _Ledger.release(self.runtime_state, allocation,
                                after=self._ledger_boundary())
        before = self._replica_occupied(replica)
        before_host = self.host_occupied_bytes(replica)
        self._set_ref((replica, "grant", name), ())
        self.grant_records.pop((replica, name), None)
        self.grants.pop((replica, name), None)
        self.grant_bindings.pop((replica, name), None)
        self.grant_hosts.pop((replica, name), None)
        after = self._replica_occupied(replica)
        after_host = self.host_occupied_bytes(replica)
        released = max(0.0, before - after)
        released_total = released + max(0.0, before_host - after_host)
        if sync:
            self._sync_usage()
            self.budget_version += 1
        else:
            self._rebuild_grant_backings()
        self.release_events.append({
            "kind": "grant", "replica": replica, "name": name,
            **binding, "released_bytes": released_total,
        })
        return released

    def release_grant(
        self, replica: int, name: str, *, generation: int | None = None,
        event=None, rank: int | None = None,
    ) -> bool:
        key = (replica, str(name))
        record = self.grant_records.get(key)
        if record is None:
            return False
        if generation is not None:
            record_generation = record.get("generation")
            if (record_generation is None
                    or int(generation) != int(record_generation)):
                return False
        if rank is not None:
            record_rank = record.get("rank")
            if record_rank is None or int(rank) != int(record_rank):
                return False
        if (event is not None and event != record.get("event")):
            return False
        self._remove_grant(replica, str(name), self.grant_bindings.get(key, {}))
        return True

    def release_grants(self, replica: int, names) -> float:
        """Release named grants; returns the summed physical released bytes."""
        released = 0.0
        changed = False
        for name in names:
            key = (replica, str(name))
            if key not in self.grant_records:
                continue
            released += self._remove_grant(
                replica, str(name), self.grant_bindings.get(key, {}),
                sync=False)
            changed = True
        if changed:
            self._sync_usage()
            self.budget_version += 1
        return released

    def outstanding_holders(self) -> dict:
        """Known live grants/selected/shared/async holders for T14 wakeup."""
        grants = []
        for (replica, name), record in sorted(self.grant_records.items()):
            grants.append({"replica": replica, "name": name,
                           **self.grant_bindings.get((replica, name), {})})
        return {
            "budget_version": self.budget_version,
            "grants": grants,
            "selected": [
                {"replica": replica, "stage": stage,
                 "bytes": self.selected_reserved[replica][stage]}
                for replica in range(self.case.runtime.parallel.dp)
                for stage in range(self.case.runtime.parallel.pp)
                if self.selected_reserved[replica][stage]
            ],
            "shared": [
                {"replica": replica, "owner": owner, **record}
                for (replica, owner), record in sorted(self.shared_pages.items())
            ],
            "async_consumers": [
                {"replica": key[0], "request": key[1], "consumer": consumer,
                 **snapshot.__dict__}
                for key, holds in sorted(self.async_consumer_holds.items())
                for consumer, snapshot in sorted(holds.items())
            ],
            "grant_readers": [
                {"replica": key[0], "grant": key[1], "consumer": consumer,
                 "storage": snapshot.get("storage"),
                 "generation": snapshot.get("generation"),
                 "event": self._binding_report(
                     {"event": snapshot.get("event")}).get("event"),
                 "ranges": snapshot.get("ranges")}
                for key, readers in sorted(self.grant_readers.items())
                for consumer, snapshot in sorted(readers.items())
            ],
        }

    def drain_release_events(self) -> tuple[dict, ...]:
        """Only real byte releases can wake another waiter."""
        events = tuple(
            event for event in self.release_events
            if float(event.get("released_bytes", 0.0)) > 0.0)
        self.release_events.clear()
        return events

    def live_grants(self, replica: int, stage: int) -> float:
        return float(self.granted[replica][stage])
