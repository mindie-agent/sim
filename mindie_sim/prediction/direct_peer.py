"""Finite direct-peer transport without receive-forward round barriers.

[Role] Schedule explicitly selected independent remote-window peer messages on
  the existing topology and Engine, including asymmetric cross-DP cohorts.
[Invariants] Each directed cell is routed and framed once. A sender owns a
  declared finite number of outstanding slots; a slot is reused only after
  remote arrival. No peer waits for an unrelated prior peer's receive. Rank
  completion includes every required incoming and outgoing message. Local
  quantization, packing and expert computation remain separate caller work.
[Timeouts & Locks] O(ranks * outstanding slots) in-flight state; virtual events.
[Side-Effect Policy] Engine tasks only; no I/O or hardware execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from mindie_sim.schema.common import ModelingError
from .collectives import PairwiseCollective, PairwiseGroup, _ready_by_rank
from .communication import MessageFlow


@dataclass(frozen=True)
class DirectPeerGroup(PairwiseGroup):
    max_inflight_per_rank: int = 1

    def start_participants(self, engine, name, ready_by_rank, *, owner=""):
        initial = _ready_by_rank(self.ranks, ready_by_rank)
        count = len(self.ranks)
        if count <= 1:
            return initial
        volumes = dict(self.volumes)
        peer = self.peer_traffic
        # Count obligations without keeping a second dense traffic matrix.
        expected_receive = [0] * count
        for src in range(count):
            for dst in range(count):
                if src != dst and self._payload(src, dst, volumes, peer):
                    expected_receive[dst] += 1
        complete = {r: engine.future(f"{name}:rank{r}:complete") for r in self.ranks}
        issued = [0] * count
        outstanding = [0] * count
        received = [0] * count

        def retire(index):
            if (issued[index] == count - 1 and not outstanding[index]
                    and received[index] == expected_receive[index]
                    and initial[self.ranks[index]].done):
                event = complete[self.ranks[index]]
                if not event.done:
                    engine.resolve(event)

        def issue(index):
            while (issued[index] < count - 1
                   and outstanding[index] < self.max_inflight_per_rank):
                offset = issued[index] + 1
                issued[index] += 1
                destination_index = (index + offset) % count
                payload = self._payload(index, destination_index, volumes, peer)
                if not payload:
                    continue
                source, destination = self.ranks[index], self.ranks[destination_index]
                outstanding[index] += 1
                prefix = f"{name}:rank{source}:peer{destination}"
                previous = engine.submit(
                    prefix + ":ready", after=(initial[source], initial[destination]),
                    owner=owner,
                    metadata={"source_rank": source, "destination_rank": destination,
                              "transport_schedule": "direct_peer"})
                for hop_index, hop in enumerate(self.topology.route_message(
                        MessageFlow(source, destination, payload)).hops):
                    service = engine.submit(
                        f"{prefix}:hop{hop_index}:service",
                        hop.demand, after=(previous,), owner=owner, coupled=True)
                    previous = engine.submit(
                        f"{prefix}:hop{hop_index}:arrival", after=(service,),
                        latency=hop.latency_ms, owner=owner)

                def arrive(_, src=index, dst=destination_index):
                    outstanding[src] -= 1
                    received[dst] += 1
                    retire(dst)
                    issue(src)

                engine.submit(prefix + ":received", after=(previous,),
                              owner=owner, on_complete=arrive)
            retire(index)

        for index, rank in enumerate(self.ranks):
            engine.submit(f"{name}:rank{rank}:entry", after=(initial[rank],),
                          owner=owner, on_complete=lambda _, i=index: issue(i))
        return complete

    def _payload(self, source, destination, volumes, peer):
        return (peer.payload(source, destination) if peer is not None
                else volumes[self.ranks[source]] / (len(self.ranks) - 1))


@dataclass(frozen=True)
class DirectPeerCollective(PairwiseCollective):
    """Declared coalesced peer window; not an inferred vendor token scheduler."""

    max_inflight_per_rank: int = field(kw_only=True)
    algorithm: ClassVar[str] = "direct_peer"

    def __post_init__(self):
        if (not isinstance(self.max_inflight_per_rank, int)
                or isinstance(self.max_inflight_per_rank, bool)
                or self.max_inflight_per_rank < 1):
            raise ModelingError("direct_peer_window_requires_positive_integer")

    @property
    def plan_options(self):
        return {"max_inflight_per_rank": self.max_inflight_per_rank}

    def accounting_description(self, accounting):
        result = accounting.to_dict()
        serial = result.pop("setup_ms")
        return {**result, "setup_ms": 0.0,
                "unused_serial_round_estimate_ms": serial,
                "latency_model": "per_hop_arrival_in_finite_direct_peer_window",
                **self.plan_options,
                "coverage": "transport_only; caller_local_work_separate"}

