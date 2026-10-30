"""Route communication through actual shared physical links.

[Role] Map collective groups and point-to-point payloads to link service.
[Invariants] A route must connect both endpoints. Shared fabric capacity is
  shared by every flow; full-duplex ports charge source Tx and destination Rx.
  Per-pair links own independent directed capacities. Logical message bytes
  and transaction-rounded physical wire bytes remain distinct. Joint plans
  cover complete physical groups, account each rank once and never merge
  unrelated groups.
[Timeouts & Locks] Finite rank graph, bounded route cache; no network I/O.
[Side-Effect Policy] Only caches paths within this simulation.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import math

from mindie_sim.schema.common import ModelingError
from .communication import CommunicationWork, MessageFlow, RoutedHop, RoutedMessage
from .engine import Resource
from .peer_traffic import PeerTraffic


_PORT_SEMANTICS = {"per_port", "fully_connected", "mesh"}


def _declared_peer_traffic(metadata, size):
    if "peer_traffic" not in metadata:
        return None
    return PeerTraffic.from_mapping(metadata["peer_traffic"], size)


def _peer_totals_match(actual, declared):
    return math.isclose(actual, declared, abs_tol=1e-9, rel_tol=1e-12)


def _peer_declared_volume(peer, metadata, index):
    """A representative source invocation may declare the whole-group mean.

    Integer source shards can differ across TP ranks. The explicit matrix still
    owns every physical send; the scalar operator ledger is then its per-rank
    mean, rather than a false claim that all source rows have equal volume.
    """
    basis = metadata.get("peer_volume_basis", "per_rank")
    if basis == "per_rank":
        return peer.row_total(index)
    if basis == "group_mean":
        return math.fsum(peer.row_total(i) for i in range(peer.size)) / peer.size
    raise ModelingError(f"peer_traffic_volume_basis_unknown:{basis}")


class Topology:
    def __init__(self, machine, parallel, namespace=""):
        self.machine, self.parallel = machine, parallel
        self.namespace = namespace
        self.links = {link.name: link for link in machine.links}
        self.routes = {}

    def resources(self):
        result = []
        for link in self.links.values():
            if link.semantics == "per_pair":
                result.extend(Resource(
                    f"{self.namespace}link:{link.name}:pair{source}>{destination}",
                    link.capacity_bytes_per_ms, "bytes",
                ) for source in link.members for destination in link.members if source != destination)
            elif link.semantics in _PORT_SEMANTICS:
                result.extend(Resource(f"{self.namespace}link:{link.name}:{direction}:port{rank}",
                                       link.capacity_bytes_per_ms, "bytes")
                              for rank in link.members for direction in ("tx", "rx"))
            else:
                result.append(Resource(f"{self.namespace}link:{link.name}", link.capacity_bytes_per_ms, "bytes"))
            if link.switch_capacity_bytes_per_ms is not None:
                result.append(Resource(f"{self.namespace}link:{link.name}:switch_plane",
                                       link.switch_capacity_bytes_per_ms, "bytes"))
        return result

    def route(self, source, destination):
        key = source, destination
        if key in self.routes:
            return self.routes[key]
        queue, seen = deque([(source, ())]), {source}
        while queue:
            rank, path = queue.popleft()
            if rank == destination:
                if len(self.routes) < 8192:
                    self.routes[key] = path
                return path
            for link in sorted(self.links.values(), key=lambda item: (len(item.members), item.name)):
                if rank not in link.members:
                    continue
                for neighbor in link.members:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append((neighbor, path + ((link.name, rank, neighbor),)))
        raise ValueError(f"physical_route_missing:{source}->{destination}")

    def ranks(self, replica, stage):
        p = self.parallel
        width = p.tp * p.pcp
        start = (replica * p.pp + stage) * width
        return tuple(device.rank for device in self.machine.dies[start:start + width])

    def groups(self, axis, replica, stage):
        """Rank axes are independent; expert groups never absorb PCP copies."""
        p = self.parallel
        sources = self.ranks(replica, stage)
        if axis == "cp":
            return tuple(sources[i:i + p.cp] for i in range(0, len(sources), p.cp))
        if axis == "etp":
            if p.ep_spans_dp:
                return tuple((rank,) for rank in sources)
            return tuple(sources[base + i:base + p.tp:p.ep]
                         for base in range(0, len(sources), p.tp) for i in range(p.ep))
        if axis == "pcp":
            return tuple(sources[i::p.tp] for i in range(p.tp))
        if axis == "ep":
            if not p.ep_spans_dp:
                return tuple(sources[i:i + p.ep] for i in range(0, len(sources), p.ep))
            count = p.ep // p.tp
            first = replica // count * count
            return tuple(
                tuple(rank for r in range(first, first + count)
                      for rank in self.ranks(r, stage)[copy * p.tp:(copy + 1) * p.tp])
                for copy in range(p.pcp)
            )
        if axis == "tp":
            return tuple(sources[i:i + p.tp] for i in range(0, len(sources), p.tp))
        raise ModelingError(f"communication_group_axis_unknown:{axis}")

    def messages(self, op, replica, volume):
        """Yield compact repeated messages and the declared latency round count."""
        p, metadata = self.parallel, dict(op.metadata)
        family = metadata.get("collective", op.family)
        axis = metadata.get("group_axis", "tp")
        source_stage = int(metadata.get("source_stage", op.stage))
        sources = self.ranks(replica, source_stage)
        if "peer_traffic" in metadata:
            if family not in {"alltoall", "alltoallv"}:
                raise ModelingError(
                    "peer_traffic_requires_peer_collective")
            groups = tuple(self.groups(axis, replica, source_stage))
            if not groups:
                return
            sizes = {len(group) for group in groups}
            if len(sizes) > 1:
                raise ModelingError("peer_traffic_group_size_inconsistent")
            peer = _declared_peer_traffic(metadata, len(groups[0]))
            for group in groups:
                if peer.size != len(group):
                    raise ModelingError(
                        f"peer_traffic_group_size_mismatch:"
                        f"{peer.size}!={len(group)}")
                rounds = metadata.get("steps", len(group) - 1)
                if (not isinstance(rounds, int) or isinstance(rounds, bool)
                        or rounds != len(group) - 1):
                    raise ModelingError(
                        f"communication_round_count_invalid:{rounds}")
                for index, source in enumerate(group):
                    if source not in sources:
                        continue
                    actual = _peer_declared_volume(peer, metadata, index)
                    if not _peer_totals_match(actual, volume):
                        raise ModelingError(
                            f"peer_traffic_source_volume_mismatch:{source}:"
                            f"{actual}!={volume}")
                    for destination_index, destination in enumerate(group):
                        payload = peer.payload(index, destination_index)
                        if payload:
                            yield MessageFlow(
                                source, destination, payload), rounds
            return
        if family == "alltoallv":
            raise ModelingError(
                "pairwise_collective_alltoallv_requires_explicit_peer_matrix")
        if not volume:
            return
        if family == "pp_p2p" or axis == "pp":
            stage = int(metadata.get("destination_stage", min(op.stage + 1, p.pp - 1)))
            for source, destination in zip(sources, self.ranks(replica, stage)):
                if source != destination:
                    yield MessageFlow(source, destination, volume), 1
            return
        for group in self.groups(axis, replica, source_stage):
            if len(group) <= 1:
                continue
            rounds = metadata.get("steps", (2 if family == "allreduce" else 1) * (len(group) - 1))
            if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
                raise ModelingError(f"communication_round_count_invalid:{rounds}")
            for index, source in enumerate(group):
                if source not in sources:
                    continue
                if family in {"alltoall", "alltoallv"}:
                    for destination in group:
                        if source != destination:
                            yield MessageFlow(source, destination, volume / (len(group) - 1)), rounds
                else:
                    # Frontends already give total per-rank ring traffic.
                    # Split before framing, so each dependent round keeps its tail.
                    yield MessageFlow(
                        source, group[(index + 1) % len(group)], volume / rounds, rounds,
                    ), rounds

    def resource(self, link, source, destination):
        name = f"{self.namespace}link:{link.name}"
        if link.semantics == "per_pair":
            return f"{name}:pair{source}>{destination}"
        if link.semantics in _PORT_SEMANTICS:
            return f"{name}:tx:port{source}"
        return name

    def capacity_resources(self, link, source, destination):
        names = [self.resource(link, source, destination)]
        if link.semantics in _PORT_SEMANTICS:
            names.append(f"{self.namespace}link:{link.name}:rx:port{destination}")
        if link.switch_capacity_bytes_per_ms is not None:
            names.append(f"{self.namespace}link:{link.name}:switch_plane")
        return tuple(names)

    def route_message(self, flow):
        """Frame each physical hop once, shared by accounting and execution."""
        hops = []
        for name, source, destination in self.route(flow.source, flow.destination):
            link = self.links[name]
            wire, transactions = flow.on_link(link)
            frame = ((link.transaction_bytes, link.transaction_bytes + link.protocol_overhead_bytes)
                     if link.transaction_bytes is not None else None)
            hops.append(RoutedHop(
                self.resource(link, source, destination), flow.useful_bytes, wire,
                transactions, frame, link.propagation_us / 1000,
                self.capacity_resources(link, source, destination)[1:],
            ))
        return RoutedMessage(flow, tuple(hops))

    def _joint_groups(self, op, participants):
        """Resolve complete physical groups and one volume per source rank."""
        if not isinstance(participants, Mapping):
            raise ModelingError("joint_collective_participants_must_be_mapping")
        metadata = dict(op.metadata)
        axis = metadata.get("group_axis", "tp")
        stage = int(metadata.get("source_stage", op.stage))
        if not 0 <= stage < self.parallel.pp:
            raise ModelingError(f"joint_collective_source_stage_invalid:{stage}")
        values = {}
        for replica, volume in participants.items():
            if (not isinstance(replica, int) or isinstance(replica, bool)
                    or not 0 <= replica < self.parallel.dp):
                raise ModelingError(f"joint_collective_replica_invalid:{replica}")
            try:
                value = float(volume)
            except (TypeError, ValueError):
                raise ModelingError(f"joint_collective_volume_invalid:{volume}") from None
            if not math.isfinite(value) or value < 0:
                raise ModelingError(f"joint_collective_volume_invalid:{volume}")
            values[replica] = value

        volume_by_rank = {}
        for replica in sorted(values):
            for rank in self.ranks(replica, stage):
                if rank in volume_by_rank:
                    raise ModelingError(
                        f"joint_collective_rank_accounted_twice:{rank}")
                volume_by_rank[rank] = values[replica]

        groups, seen, owner = [], {}, {}
        for replica in sorted(values):
            for ranks in self.groups(axis, replica, stage):
                if ranks in seen:
                    continue
                for rank in ranks:
                    previous = owner.get(rank)
                    if previous is not None and previous != ranks:
                        raise ModelingError(
                            f"joint_collective_group_membership_conflict:{rank}")
                seen[ranks] = len(groups)
                groups.append(ranks)
                for rank in ranks:
                    owner[rank] = ranks

        for ranks in groups:
            if len(set(ranks)) != len(ranks):
                raise ModelingError("joint_collective_group_membership_duplicate")
            missing = sorted(set(ranks) - set(volume_by_rank))
            if missing:
                raise ModelingError(
                    "joint_collective_group_membership_missing:"
                    + ",".join(map(str, missing)))
        unmapped = sorted(set(volume_by_rank) - set(owner))
        if unmapped:
            raise ModelingError(
                "joint_collective_group_membership_unmapped:"
                + ",".join(map(str, unmapped)))
        return metadata.get("collective", op.family), tuple(groups), volume_by_rank, metadata

    def joint_plan(self, op, participants, *, algorithm="ring", max_inflight_per_rank=1):
        """Bind complete cross-DP cohorts without inferring a runtime algorithm."""
        from .collectives import PairwiseGroup, PairwisePlan, RingGroup, RingPlan

        if algorithm not in {"ring", "pairwise", "direct_peer"}:
            raise ModelingError(f"joint_collective_algorithm_unsupported:{algorithm}")
        family, unique_groups, volume_by_rank, metadata = self._joint_groups(
            op, participants)
        if algorithm == "ring":
            if "peer_traffic" in metadata:
                raise ModelingError(
                    "peer_traffic_ring_collective_unsupported")
            phases = {"allreduce": 2, "reduce_scatter": 1, "allgather": 1}
            if family not in phases:
                raise ModelingError(f"ring_collective_family_unsupported:{family}")
            groups = []
            for ranks in unique_groups:
                volumes = {rank: volume_by_rank[rank] for rank in ranks}
                if len(set(volumes.values())) > 1:
                    raise ModelingError(
                        "ring_collective_requires_equal_per_rank_volume")
                rounds = max(phases[family] * (len(ranks) - 1), 0)
                if metadata.get("steps", rounds) != rounds:
                    raise ModelingError(
                        "ring_collective_round_count_conflicts_with_algorithm")
                if len(ranks) <= 1 or not next(iter(volumes.values()), 0):
                    messages = ()
                else:
                    messages = tuple(self.route_message(MessageFlow(
                        rank, ranks[(index + 1) % len(ranks)],
                        volumes[rank] / rounds,
                    )) for index, rank in enumerate(ranks))
                groups.append(RingGroup(ranks, rounds, messages))
            return RingPlan(family, tuple(groups))

        if "peer_traffic" not in metadata:
            if family == "alltoallv":
                raise ModelingError(
                    "pairwise_collective_alltoallv_requires_explicit_peer_matrix")
            if family != "alltoall":
                raise ModelingError(f"pairwise_collective_family_unsupported:{family}")
            peer = None
        else:
            if family not in {"alltoall", "alltoallv"}:
                raise ModelingError(
                    "peer_traffic_requires_peer_collective")
            sizes = {len(ranks) for ranks in unique_groups}
            if len(sizes) > 1:
                raise ModelingError("peer_traffic_group_size_inconsistent")
            peer = (_declared_peer_traffic(metadata, next(iter(sizes)))
                    if sizes else None)
        groups = []
        for ranks in unique_groups:
            rounds = max(len(ranks) - 1, 0)
            if metadata.get("steps", rounds) != rounds:
                raise ModelingError(
                    "pairwise_collective_round_count_conflicts_with_schedule")
            if peer is not None:
                if peer.size != len(ranks):
                    raise ModelingError(
                        f"peer_traffic_group_size_mismatch:"
                        f"{peer.size}!={len(ranks)}")
                totals = []
                for index, rank in enumerate(ranks):
                    total = peer.row_total(index)
                    declared = volume_by_rank[rank]
                    if not _peer_totals_match(
                            _peer_declared_volume(peer, metadata, index), declared):
                        raise ModelingError(
                            f"peer_traffic_source_volume_mismatch:{rank}:"
                            f"{total}!={declared}")
                    totals.append((rank, total))
            else:
                totals = [(rank, volume_by_rank[rank]) for rank in ranks]
            if algorithm == "direct_peer":
                from .direct_peer import DirectPeerGroup
                groups.append(DirectPeerGroup(
                    ranks, tuple(totals), rounds, self, peer_traffic=peer,
                    max_inflight_per_rank=max_inflight_per_rank))
            else:
                groups.append(PairwiseGroup(
                    ranks, tuple(totals), rounds, self, peer_traffic=peer))
        return PairwisePlan(family, tuple(groups))

    def ring_plan(self, op, replica, *, volume=None):
        """Bind a uniform-chunk local ring; no vendor algorithm is inferred."""
        from .collectives import RingPlan

        volume = op.link_bytes if volume is None else volume
        try:
            plan = self.joint_plan(op, {replica: volume}, algorithm="ring")
        except ModelingError as error:
            if str(error).startswith("joint_collective_group_membership"):
                raise ModelingError(
                    "ring_collective_requires_complete_local_group") from None
            raise
        # Preserve the local declaration's historical empty/singleton behavior.
        return RingPlan(plan.family, tuple(
            group for group in plan.groups
            if len(group.ranks) > 1 and group.rounds and group.messages))

    def pairwise_plan(self, op, replica, *, volume=None):
        """Bind a balanced local pairwise declaration; peers stay uniform."""
        volume = op.link_bytes if volume is None else volume
        return self.joint_plan(op, {replica: volume}, algorithm="pairwise")

    def accounting(self, op, replica, *, volume=None):
        volume = op.link_bytes if volume is None else volume
        useful_by_rank, useful, wire, transactions = {}, {}, {}, {}
        transaction_frames = {}
        latency, count = 0.0, 0
        routed_useful, physical_wire = 0.0, 0.0
        for flow, rounds in self.messages(op, replica, volume):
            useful_by_rank[flow.source] = useful_by_rank.get(flow.source, 0) + flow.useful_bytes
            count += flow.messages
            path_latency = 0
            for hop in self.route_message(flow).hops:
                routed_useful += hop.useful_bytes
                physical_wire += hop.wire_bytes
                for resource in hop.demand:
                    useful[resource] = useful.get(resource, 0) + hop.useful_bytes
                    wire[resource] = wire.get(resource, 0) + hop.wire_bytes
                    transactions[resource] = (None if hop.transactions is None
                                              else transactions.get(resource, 0) + hop.transactions)
                    if hop.transaction_frame is not None:
                        transaction_frames[resource] = hop.transaction_frame
                path_latency += hop.latency_ms
            latency = max(latency, path_latency * rounds)
        return CommunicationWork(
            useful_by_rank, useful, wire, transactions, transaction_frames, count, latency,
            routed_useful, physical_wire,
        )

    def demand(self, op, replica, *, volume=None):
        work = self.accounting(op, replica, volume=volume)
        return dict(work.wire_by_resource), work.setup_ms
