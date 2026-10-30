"""Bounded ring and pairwise collective transport on the common engine.

[Role] Execute declared allreduce, reduce-scatter, allgather and alltoall
  transport rounds, including complete cross-DP cohorts.
[Invariants] A rank's next send follows its own prior first-hop service and its
  own prior receive. Peer sends are not serialized by a collective-wide barrier.
  Every hop is routed and framed once by Topology.route_message, then served,
  propagated and forwarded causally. Empty traffic consumes no wire or latency.
[Timeouts & Locks] Virtual events; O(ranks) counters plus actual in-flight
  messages. Pairwise rounds are generated lazily rather than retained.
[Side-Effect Policy] Submits tasks to the caller's Engine; no runtime dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import ClassVar

from mindie_sim.schema.common import ModelingError
from .communication import MessageFlow, RoutedMessage


def plan_options(recipe, op):
    """A source template may derive its finite peer window from group geometry."""
    value = getattr(recipe, "plan_options", {})
    return value(op) if callable(value) else value


def _ready_by_rank(ranks, ready_by_rank):
    """Require exact event coverage before a joint schedule submits any work."""
    required, supplied = set(ranks), set(ready_by_rank)
    if required != supplied:
        missing = ",".join(map(str, sorted(required - supplied)))
        unexpected = ",".join(map(str, sorted(supplied - required)))
        raise ModelingError(
            f"collective_ready_coverage_mismatch:missing={missing}:"
            f"unexpected={unexpected}")
    return {rank: ready_by_rank[rank] for rank in ranks}


def _joint_complete(engine, name, events, ready, owner):
    return engine.submit(name + ":complete", after=tuple(events) or (ready,),
                         owner=owner)


@dataclass(frozen=True)
class RingGroup:
    ranks: tuple[int, ...]
    rounds: int
    messages: tuple[RoutedMessage, ...]

    def start(self, engine, name, ready, *, owner=""):
        events = self.start_participants(
            engine, name, {rank: ready for rank in self.ranks}, owner=owner)
        return _joint_complete(engine, name, events.values(), ready, owner)

    def start_participants(self, engine, name, ready_by_rank, *, owner=""):
        """Per-rank receive-forward completion without a collective-wide gate."""
        initial = _ready_by_rank(self.ranks, ready_by_rank)
        if not self.ranks:
            return {}
        if self.rounds <= 0 or not self.messages or not any(
                message.flow.useful_bytes for message in self.messages):
            return initial
        count = len(self.ranks)
        if len(self.messages) != count or len(set(self.ranks)) != count:
            raise ModelingError("ring_collective_message_rank_count_invalid")
        complete = {rank: engine.future(f"{name}:rank{rank}:complete")
                    for rank in self.ranks}
        issued = [-1] * count
        sent = [-1] * count
        received = [-1] * count
        sends = [initial[rank] for rank in self.ranks]
        arrivals = [initial[rank] for rank in self.ranks]

        def retire(index):
            event = complete[self.ranks[index]]
            predecessor = (index - 1) % count
            # received[i] is source i's message arriving at i+1. Rank index's
            # own last receive is therefore received[predecessor].
            if (sent[index] >= self.rounds - 1
                    and received[predecessor] >= self.rounds - 1):
                if not event.done:
                    engine.resolve(event)

        def issue(index):
            next_round = issued[index] + 1
            predecessor = (index - 1) % count
            if (next_round >= self.rounds or sent[index] < next_round - 1
                    or received[predecessor] < next_round - 1):
                return
            issued[index] = next_round
            message = self.messages[index]
            prefix = f"{name}:rank{self.ranks[index]}:round{next_round}"
            # The first send waits source and destination entry. Later sends
            # retain predecessor receive-forward ordering without a new gate.
            dependencies = ((initial[self.ranks[index]],
                             initial[message.flow.destination])
                            if next_round == 0
                            else (sends[index], arrivals[predecessor]))
            previous = engine.submit(
                prefix + ":ready", after=dependencies,
                owner=owner, metadata={"source_rank": message.flow.source,
                                       "destination_rank": message.flow.destination,
                                       "round": next_round},
            )

            def transmitted(event):
                sent[index] = next_round
                sends[index] = event
                retire(index)
                issue(index)

            for hop_index, hop in enumerate(message.hops):
                service = engine.submit(
                    f"{prefix}:hop{hop_index}:service", hop.demand,
                    after=(previous,), owner=owner, coupled=True,
                    on_complete=transmitted if hop_index == 0 else None,
                )
                previous = engine.submit(
                    f"{prefix}:hop{hop_index}:arrival", after=(service,),
                    latency=hop.latency_ms, owner=owner,
                )

            def arrived(event):
                received[index] = next_round
                arrivals[index] = event
                receiver = (index + 1) % count
                retire(receiver)
                issue(receiver)

            engine.submit(prefix + ":received", after=(previous,), on_complete=arrived,
                          owner=owner)

        for index in range(count):
            issue(index)
        return complete


@dataclass(frozen=True)
class RingPlan:
    family: str
    groups: tuple[RingGroup, ...]

    @property
    def ranks(self):
        return tuple(sorted({rank for group in self.groups for rank in group.ranks}))

    def start(self, engine, name, ready, *, owner=""):
        events = self.start_participants(
            engine, name, {rank: ready for rank in self.ranks}, owner=owner)
        return _joint_complete(engine, name, events.values(), ready, owner)

    def start_participants(self, engine, name, ready_by_rank, *, owner=""):
        _ready_by_rank(self.ranks, ready_by_rank)
        events = {}
        for index, group in enumerate(self.groups):
            part = group.start_participants(
                engine, f"{name}:group{index}",
                {rank: ready_by_rank[rank] for rank in group.ranks}, owner=owner)
            overlap = set(events) & set(part)
            if overlap:
                raise ModelingError(
                    "ring_collective_rank_accounted_twice:"
                    + ",".join(map(str, sorted(overlap))))
            events.update(part)
        if set(events) != set(self.ranks):
            raise ModelingError("ring_collective_group_completion_missing")
        return events

    def validate_path_model(self, path_model):
        if path_model not in {"direct", "store_forward"}:
            raise ModelingError(
                f"ring_collective_path_model_unsupported:{path_model}")
        if path_model == "direct" and any(
                len(message.hops) != 1
                for group in self.groups for message in group.messages):
            raise ModelingError(
                "ring_collective_multihop_requires_store_forward_declaration")


@dataclass(frozen=True)
class PairwiseGroup:
    """Causal alltoall rounds with uniform totals or explicit directed cells."""

    ranks: tuple[int, ...]
    volumes: tuple[tuple[int, float], ...]
    rounds: int
    topology: object = field(repr=False, compare=False)
    peer_traffic: object | None = field(default=None, repr=False, compare=False)

    def start(self, engine, name, ready, *, owner=""):
        events = self.start_participants(
            engine, name, {rank: ready for rank in self.ranks}, owner=owner)
        return _joint_complete(engine, name, events.values(), ready, owner)

    def start_participants(self, engine, name, ready_by_rank, *, owner=""):
        initial = _ready_by_rank(self.ranks, ready_by_rank)
        if not self.ranks:
            return {}
        if self.rounds != max(len(self.ranks) - 1, 0):
            raise ModelingError(
                "pairwise_collective_round_count_conflicts_with_schedule")
        volumes = dict(self.volumes)
        if set(volumes) != set(self.ranks):
            raise ModelingError("pairwise_collective_volume_rank_count_invalid")
        if self.peer_traffic is not None:
            if self.peer_traffic.size != len(self.ranks):
                raise ModelingError(
                    "pairwise_collective_peer_group_size_mismatch")
            for index, rank in enumerate(self.ranks):
                total = self.peer_traffic.row_total(index)
                if not math.isclose(
                        total, volumes[rank], abs_tol=1e-9, rel_tol=1e-12):
                    raise ModelingError(
                        f"pairwise_collective_peer_row_total_mismatch:{rank}")
        if not self.rounds or not any(volumes.values()):
            return initial
        count, rounds = len(self.ranks), self.rounds
        complete = {rank: engine.future(f"{name}:rank{rank}:complete")
                    for rank in self.ranks}
        sent_count = [0] * count
        expected_receive = [0] * count
        send_busy = [False] * count
        next_round = [0] * count
        last_send = [initial[rank] for rank in self.ranks]
        received = [initial[rank] for rank in self.ranks]
        pending = [{} for _ in range(count)]

        def retire(index):
            if sent_count[index] == rounds and expected_receive[index] == rounds:
                event = complete[self.ranks[index]]
                if not event.done:
                    engine.resolve(event)

        def deliver(index, round_index, event):
            """Advance a receiver by contiguous rounds; buffer only actual early arrivals."""
            if round_index < expected_receive[index]:
                raise ModelingError("pairwise_collective_receive_round_regressed")
            if round_index > expected_receive[index]:
                if round_index in pending[index]:
                    raise ModelingError("pairwise_collective_receive_round_duplicate")
                pending[index][round_index] = event
                return
            while True:
                received[index] = event
                expected_receive[index] += 1
                retire(index)
                issue(index)
                following = expected_receive[index]
                event = pending[index].pop(following, None)
                if event is None:
                    return

        def issue(index):
            round_index = next_round[index]
            if round_index >= rounds or send_busy[index]:
                return
            if (round_index and (sent_count[index] < round_index
                                 or expected_receive[index] < round_index)):
                return
            send_busy[index] = True
            next_round[index] = round_index + 1
            source = self.ranks[index]
            destination_index = (index + round_index + 1) % count
            destination = self.ranks[destination_index]
            payload = (self.peer_traffic.payload(index, destination_index)
                       if self.peer_traffic is not None
                       else volumes[source] / rounds)
            dependencies = [initial[source]]
            # A zero-byte logical send has no destination buffer obligation.
            if payload:
                dependencies.append(initial[destination])
            if round_index:
                dependencies.extend((last_send[index], received[index]))
            prefix = f"{name}:rank{source}:round{round_index}"

            def dispatched(event, payload=payload):
                if not payload:
                    last_send[index] = event
                    sent_count[index] += 1
                    send_busy[index] = False
                    retire(index)
                    issue(index)
                    engine.submit(
                        prefix + ":received",
                        after=(event, initial[destination]),
                        on_complete=lambda arrival: deliver(
                            destination_index, round_index, arrival),
                        owner=owner)
                    return
                message = self.topology.route_message(MessageFlow(
                    source, destination, payload))
                previous = event

                def sent(event):
                    last_send[index] = event
                    sent_count[index] += 1
                    send_busy[index] = False
                    retire(index)
                    issue(index)

                for hop_index, hop in enumerate(message.hops):
                    service = engine.submit(
                        f"{prefix}:hop{hop_index}:service",
                        hop.demand, after=(previous,),
                        owner=owner, coupled=True, on_complete=sent if hop_index == 0 else None,
                    )
                    previous = engine.submit(
                        f"{prefix}:hop{hop_index}:arrival", after=(service,),
                        latency=hop.latency_ms, owner=owner,
                    )

                engine.submit(
                    prefix + ":received",
                    after=(previous, initial[destination]),
                    on_complete=lambda arrival: deliver(
                        destination_index, round_index, arrival),
                    owner=owner)

            engine.submit(
                prefix + ":ready", after=tuple(dependencies), owner=owner,
                metadata={"source_rank": source, "destination_rank": destination,
                          "round": round_index},
                on_complete=dispatched,
            )

        for index in range(count):
            issue(index)
        return complete


@dataclass(frozen=True)
class PairwisePlan:
    family: str
    groups: tuple[PairwiseGroup, ...]

    @property
    def ranks(self):
        return tuple(sorted({rank for group in self.groups for rank in group.ranks}))

    def start(self, engine, name, ready, *, owner=""):
        events = self.start_participants(
            engine, name, {rank: ready for rank in self.ranks}, owner=owner)
        return _joint_complete(engine, name, events.values(), ready, owner)

    def start_participants(self, engine, name, ready_by_rank, *, owner=""):
        _ready_by_rank(self.ranks, ready_by_rank)
        events = {}
        for index, group in enumerate(self.groups):
            part = group.start_participants(
                engine, f"{name}:group{index}",
                {rank: ready_by_rank[rank] for rank in group.ranks}, owner=owner)
            overlap = set(events) & set(part)
            if overlap:
                raise ModelingError(
                    "pairwise_collective_rank_accounted_twice:"
                    + ",".join(map(str, sorted(overlap))))
            events.update(part)
        if set(events) != set(self.ranks):
            raise ModelingError("pairwise_collective_group_completion_missing")
        return events

    def validate_path_model(self, path_model):
        if path_model not in {"direct", "store_forward"}:
            raise ModelingError(
                f"pairwise_collective_path_model_unsupported:{path_model}")
        if path_model == "direct":
            for group in self.groups:
                volumes = dict(group.volumes)
                if group.peer_traffic is not None:
                    if group.peer_traffic.size != len(group.ranks):
                        raise ModelingError(
                            "pairwise_collective_peer_group_size_mismatch")
                    for source_index, source in enumerate(group.ranks):
                        for destination_index, destination in enumerate(group.ranks):
                            if source_index == destination_index:
                                continue
                            payload = group.peer_traffic.payload(
                                source_index, destination_index)
                            if (payload and len(group.topology.route(
                                    source, destination)) != 1):
                                raise ModelingError(
                                    "pairwise_collective_multihop_requires_"
                                    "store_forward_declaration")
                    continue
                if not any(volumes.values()):
                    continue
                for source in group.ranks:
                    if not volumes[source]:
                        continue
                    for destination in group.ranks:
                        if (source != destination
                                and len(group.topology.route(source, destination)) != 1):
                            raise ModelingError(
                                "pairwise_collective_multihop_requires_"
                                "store_forward_declaration")


@dataclass(frozen=True)
class RingCollective:
    """Transport-only uniform chunks, with an explicit physical path model."""

    path_model: str = "direct"
    algorithm: ClassVar[str] = "ring"

    def lower(self, context):
        from .program import KernelProgram, Stage

        if any(amount for role, amount in context.native_work.items() if role != "link"):
            raise ModelingError("ring_collective_requires_transport_only_work")
        if self.path_model not in {"direct", "store_forward"}:
            raise ModelingError(f"ring_collective_path_model_unsupported:{self.path_model}")
        if not context.native_work.get("link"):
            return KernelProgram((), minimum_ms=context.minimum_ms)
        if context.collective_plan is None:
            raise ModelingError("ring_collective_requires_bound_topology")
        plan = context.collective_plan()
        # The aggregate setup_ms contains all ring propagation delays. Every
        # actual message now carries its own latency; adding it again is wrong.
        plan.validate_path_model(self.path_model)
        return KernelProgram((Stage("ring", program=plan),), minimum_ms=context.minimum_ms)


@dataclass(frozen=True)
class PairwiseCollective:
    """Transport-only balanced alltoall with an explicit physical path model."""

    path_model: str = "direct"
    algorithm: ClassVar[str] = "pairwise"

    def lower(self, context):
        from .program import KernelProgram, Stage

        if any(amount for role, amount in context.native_work.items() if role != "link"):
            raise ModelingError("pairwise_collective_requires_transport_only_work")
        if self.path_model not in {"direct", "store_forward"}:
            raise ModelingError(
                f"pairwise_collective_path_model_unsupported:{self.path_model}")
        if context.collective_plan is None:
            if not context.native_work.get("link"):
                return KernelProgram((), minimum_ms=context.minimum_ms)
            raise ModelingError("pairwise_collective_requires_bound_topology")
        # A locally idle source may still receive from its bound cohort.
        plan = context.collective_plan()
        # setup_ms is an aggregate legacy estimate; per-hop events are exact.
        plan.validate_path_model(self.path_model)
        return KernelProgram((Stage("pairwise", program=plan),),
                             minimum_ms=context.minimum_ms)
