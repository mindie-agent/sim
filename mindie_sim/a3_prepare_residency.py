"""Business-layer residency ledger for prepared outputs.

[Role]
  Account prepared-output capacity separately from transient preparation
  service.  The accepted resource engine keeps task-scoped reservations; this
  ledger holds *static* model-resident output and *request* output from
  admission until the last consumer/request release, so a short preparation
  task cannot make an overcommitted resident set look feasible.

[Invariants]
  * Workspace lifetime is the preparation task interval.
  * Model-resident output is held for the run unless the caller marks it
    already accounted (no double charge of existing model storage).
  * Request output is held until the caller releases its request/iteration.
  * Missing capacity evidence is an explicit gap, never a wildcard.
  * No engine rewrite and no fabricated service work.

[Timeouts & Locks]
  Pure in-memory accounting; no I/O, device or locks.

[Side-Effect Policy]
  Adds ledger metadata to an engine only when explicitly called by preparation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class ResidencyEntry:
    entry_id: str
    pool: str
    owner: str
    lifetime: str
    bytes: int
    admitted_ms: float
    request_id: str | None = None
    iteration: int = -1
    status: str = "admitted"
    released_ms: float | None = None
    accounted_elsewhere: bool = False
    task_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "pool": self.pool,
            "owner": self.owner,
            "lifetime": self.lifetime,
            "bytes": self.bytes,
            "admitted_ms": self.admitted_ms,
            "request_id": self.request_id,
            "iteration": self.iteration,
            "status": self.status,
            "released_ms": self.released_ms,
            "accounted_elsewhere": self.accounted_elsewhere,
            "task_id": self.task_id,
        }


@dataclass
class PrepareResidencyLedger:
    """Track occupied prepared-output bytes per physical capacity pool."""

    pool_capacity: dict[str, float] = field(default_factory=dict)
    entries: list[ResidencyEntry] = field(default_factory=list)
    _counter: int = 0

    def capacity(self, pool: str) -> float | None:
        value = self.pool_capacity.get(pool)
        return None if value is None else float(value)

    def _active(self, pool: str) -> list[ResidencyEntry]:
        return [
            entry for entry in self.entries
            if entry.pool == pool and entry.status == "admitted"
            and not entry.accounted_elsewhere
        ]

    def occupied(self, pool: str) -> int:
        return sum(int(entry.bytes) for entry in self._active(pool))

    def admit(
        self,
        *,
        pool: str,
        capacity_bytes: int,
        lifetime: str,
        owner: str,
        admitted_ms: float,
        request_id: str | None = None,
        iteration: int = -1,
        accounted_elsewhere: bool = False,
    ) -> ResidencyEntry:
        self._counter += 1
        entry = ResidencyEntry(
            entry_id=f"residency:{self._counter}",
            pool=str(pool), owner=str(owner),
            lifetime=str(lifetime or "workspace"),
            bytes=max(0, int(capacity_bytes)),
            admitted_ms=float(admitted_ms),
            request_id=request_id, iteration=int(iteration),
            accounted_elsewhere=bool(accounted_elsewhere),
        )
        if lifetime == "workspace":
            # Engine task reservations already enforce the short preparation
            # interval.  A workspace output must not leak into the persistent
            # static/request ledger after that task completes.
            entry.accounted_elsewhere = True
            entry.status = "workspace_task_scoped"
            self.entries.append(entry)
            return entry
        if accounted_elsewhere:
            entry.status = "already_accounted"
            self.entries.append(entry)
            return entry
        cap = self.capacity(pool)
        if cap is None or cap <= 0:
            entry.status = "missing_capacity"
            self.entries.append(entry)
            return entry
        if self.occupied(pool) + entry.bytes > cap + 1e-9:
            entry.status = "overcommit"
            self.entries.append(entry)
            return entry
        self.entries.append(entry)
        return entry

    def release(self, entry_id: str, now_ms: float) -> bool:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                if entry.status != "admitted":
                    return False
                entry.status = "released"
                entry.released_ms = float(now_ms)
                return True
        return False

    def release_request(self, request_id: str, now_ms: float) -> int:
        released = 0
        for entry in self.entries:
            if entry.status == "admitted" and entry.request_id == request_id:
                entry.status = "released"
                entry.released_ms = float(now_ms)
                released += 1
        return released

    def release_iteration(
        self, request_id: str, iteration: int, now_ms: float,
    ) -> int:
        released = 0
        for entry in self.entries:
            if (entry.status == "admitted"
                    and entry.request_id == request_id
                    and entry.iteration == int(iteration)):
                entry.status = "released"
                entry.released_ms = float(now_ms)
                released += 1
        return released

    def report(self) -> dict[str, Any]:
        pools = {}
        for pool, cap in sorted(self.pool_capacity.items()):
            active = self._active(pool)
            pools[pool] = {
                "capacity_bytes": float(cap),
                "occupied_bytes": sum(int(entry.bytes) for entry in active),
                "admitted_entries": len(active),
                "static_entries": sum(
                    1 for entry in active if entry.lifetime in {"model_resident", "static"}
                ),
                "request_entries": sum(
                    1 for entry in active if entry.request_id is not None
                ),
                "overcommitted": self.occupied(pool) > float(cap) + 1e-9,
            }
        return {
            "pools": pools,
            "entries": [entry.to_dict() for entry in self.entries],
            "overcommit_entries": [
                entry.to_dict() for entry in self.entries
                if entry.status == "overcommit"
            ],
            "missing_capacity_entries": [
                entry.to_dict() for entry in self.entries
                if entry.status == "missing_capacity"
            ],
            "basis": "business-layer residency; engine reservations remain task-scoped",
        }


def ledger_for_engine(engine: Any) -> PrepareResidencyLedger:
    ledger = getattr(engine, "prepare_residency", None)
    if not isinstance(ledger, PrepareResidencyLedger):
        ledger = PrepareResidencyLedger()
        setattr(engine, "prepare_residency", ledger)
    return ledger


__all__ = [
    "PrepareResidencyLedger", "ResidencyEntry", "ledger_for_engine",
]
