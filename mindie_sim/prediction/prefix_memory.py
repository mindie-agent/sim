"""Prefix-pool reference and admission adapter for the Memory ledger.

[Role]
  Reference-count immutable shared prefix pages once per DP replica and expose
  the private/COW remainder to the existing Memory ledger.  This module owns no
  second allocator or Engine.

[Invariants]
  * A prefix key owns one physical byte reservation per replica; retaining an
    existing key increments references without adding bytes.
  * Equal keys must declare equal page bytes; unequal declarations fail.
  * Release frees bytes only when the last active/retiring/in-flight reference
    disappears.
  * Different prefix keys and different DP replicas never share.

[Timeouts & Locks]
  Single-threaded bounded counters; no I/O or locks.

[Side-Effect Policy]
  Mutates only this pool's reference table.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError


ZERO = 0
ONE = 1
ONE_FLOAT = 1.0
REL_TOL = 1e-12


@dataclass(frozen=True)
class PrefixReservation:
    key: str
    replica: int
    bytes: float
    references: int


class PrefixPool:
    """Reference-counted shared prefix pages for one DP replica."""

    def __init__(self, *, replica: int = ZERO) -> None:
        self.replica = int(replica)
        self._bytes: dict[str, float] = {}
        self._references: Counter = Counter()

    def retain(self, key: str, bytes_: float) -> bool:
        """Retain one reference; return True when a new page was reserved."""
        name = str(key)
        value = float(bytes_)
        if not name or value < ZERO:
            raise ModelingError("prefix_pool_reservation_invalid")
        existing = self._bytes.get(name)
        if existing is not None and abs(existing - value) > max(ONE_FLOAT, abs(existing) * REL_TOL):
            raise ModelingError(f"prefix_pool_size_conflict:{name}")
        self._references[name] += ONE
        if existing is None:
            self._bytes[name] = value
            return True
        return False

    def release(self, key: str) -> float:
        """Release one reference and return freed bytes (zero while referenced)."""
        name = str(key)
        if self._references[name] <= ZERO:
            raise ModelingError(f"prefix_pool_release_without_reference:{name}")
        self._references[name] -= ONE
        if self._references[name] > ZERO:
            return ZERO
        self._references.pop(name, None)
        return float(self._bytes.pop(name, ZERO))

    def references(self, key: str) -> int:
        return int(self._references[str(key)])

    @property
    def shared_bytes(self) -> float:
        return float(sum(self._bytes.values()))

    def reservations(self) -> tuple[PrefixReservation, ...]:
        return tuple(
            PrefixReservation(name, self.replica, value,
                              int(self._references[name]))
            for name, value in sorted(self._bytes.items()))

    def to_dict(self) -> dict:
        return {
            "replica": self.replica,
            "shared_bytes": self.shared_bytes,
            "reservations": [
                {"key": item.key, "bytes": item.bytes,
                 "references": item.references}
                for item in self.reservations()],
        }
