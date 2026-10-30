"""Message-level useful and physical wire accounting.

[Role] Keep collective message boundaries when expanding physical transactions.
[Invariants] Granules and framing belong to a declared link, never a model name.
  Rounding happens per message before aggregation; each hop reframes its useful
  payload independently. No transaction declaration retains continuous bytes.
[Timeouts & Locks] Pure bounded arithmetic; no packet or iteration history.
[Side-Effect Policy] Immutable records; no I/O or engine mutation.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class MessageFlow:
    source: int
    destination: int
    payload_bytes: float
    messages: int = 1

    @property
    def useful_bytes(self):
        return self.payload_bytes * self.messages

    def on_link(self, link):
        """Return (wire bytes, transactions) for this link's framing contract."""
        if not self.payload_bytes:
            return 0.0, 0
        if link.transaction_bytes is None:
            return self.useful_bytes, None
        aligned = math.ceil(self.payload_bytes / link.alignment_bytes) * link.alignment_bytes
        transactions = math.ceil(aligned / link.transaction_bytes) * self.messages
        wire = transactions * (link.transaction_bytes + link.protocol_overhead_bytes)
        return wire, transactions


@dataclass(frozen=True)
class RoutedHop:
    resource: str
    useful_bytes: float
    wire_bytes: float
    transactions: int | None
    transaction_frame: tuple[int, int] | None
    latency_ms: float
    additional_resources: tuple[str, ...] = ()

    @property
    def demand(self):
        """One transmission simultaneously occupies all path capacities."""
        return {name: self.wire_bytes for name in (self.resource, *self.additional_resources)}


@dataclass(frozen=True)
class RoutedMessage:
    flow: MessageFlow
    hops: tuple[RoutedHop, ...]


@dataclass(frozen=True)
class CommunicationWork:
    """One replica's sending ranks; routed totals count every physical hop."""

    useful_by_rank: Mapping[int, float]
    useful_by_resource: Mapping[str, float]
    wire_by_resource: Mapping[str, float]
    transactions_by_resource: Mapping[str, int | None]
    transaction_frames: Mapping[str, tuple[int, int]]
    messages: int
    setup_ms: float
    routed_useful_bytes: float = 0
    physical_wire_bytes: float = 0

    def to_dict(self):
        return {
            "boundary": "sending_ranks_in_replica",
            "useful_bytes": sum(self.useful_by_rank.values()),
            "routed_useful_bytes": self.routed_useful_bytes,
            "wire_bytes": self.physical_wire_bytes,
            "capacity_byte_service": sum(self.wire_by_resource.values()),
            "resource_boundary": "simultaneous_capacity_demands_not_additional_transmissions",
            "messages": self.messages,
            "setup_ms": self.setup_ms,
            "useful_bytes_by_rank": {str(rank): value for rank, value in self.useful_by_rank.items()},
            "resources": {
                name: {
                    "useful_bytes": self.useful_by_resource[name],
                    "wire_bytes": wire,
                    "transactions": self.transactions_by_resource[name],
                    **({"transaction_payload_bytes": self.transaction_frames[name][0],
                        "transaction_wire_bytes": self.transaction_frames[name][1]}
                       if name in self.transaction_frames else {}),
                }
                for name, wire in self.wire_by_resource.items()
            },
        }
