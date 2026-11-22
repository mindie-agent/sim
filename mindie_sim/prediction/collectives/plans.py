"""Collective plan binding, joint group validation and wire-byte accounting.

[Role]
  Bind ring, pairwise, direct-peer, direct-mesh and startup-wave collective
  plans over a ``Topology`` and account physical wire bytes and hop startup
  once.

[Invariants]
  * ``transport_plan`` freezes iterable/generator ``flows`` once at entry so
    multi-pass capacity, buffer-ID, and path construction never exhaust input
    generators.
  * Joint plans cover complete physical groups, account each rank once, and
    never merge unrelated groups.
  * A direct mesh requires one declared ``per_pair`` hop between every
    single-node peer pair.

[Timeouts & Locks]
  Bounded participant groups and bounded ``_transport_bindings`` cache; no
  threads, sleeps, locks or I/O.

[Side-Effect Policy]
  Populates bounded per-Topology transport option caches only.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

from mindie_sim.schema.common import ModelingError
from ..communication import CommunicationWork, MessageFlow
from ..collectives.peer_traffic import PeerTraffic
from .ring_pairwise import (
    OneShotGroup,
    OneShotPlan,
    PairwiseGroup,
    PairwisePlan,
    RingGroup,
    RingPlan,
)
from .transport import (
    DEFAULT_MAX_PHYSICAL_CHUNKS,
    ChunkSequence,
    FlowPath,
    TransportOptions,
    TransportPlan,
    chunk_count,
    chunk_size_at,
    chunk_sizes,
)

MAX_TRANSPORT_BINDINGS = 8192
DIRECT_MESH_FAMILIES = frozenset({"allreduce", "reduce_scatter", "allgather"})
DIRECT_MESH_PHASES = {
    "allreduce": ("reduce_scatter", "all_gather"),
    "reduce_scatter": ("reduce_scatter",),
    "allgather": ("all_gather",),
}
DIRECT_MESH_MAX_PEERS = 8
ONESHOT_FAMILIES = frozenset({
    "allreduce", "reduce_scatter", "allgather", "alltoall", "alltoallv",
})
_COLLECTIVE_FAMILY_ALIASES = {
    "all_reduce": "allreduce",
    "all_gather": "allgather",
    "all_to_all": "alltoall",
    "all_to_all_v": "alltoallv",
}


def _declared_peer_traffic(metadata, size):
    if "peer_traffic" not in metadata:
        return None
    return PeerTraffic.from_mapping(metadata["peer_traffic"], size)


def _peer_totals_match(actual, declared):
    return math.isclose(actual, declared, abs_tol=1e-9, rel_tol=1e-12)


def _peer_declared_volume(peer, metadata, index):
    """A representative source invocation may declare the whole-group mean."""
    basis = metadata.get("peer_volume_basis", "per_rank")
    if basis == "per_rank":
        return peer.row_total(index)
    if basis == "group_mean":
        return math.fsum(peer.row_total(i) for i in range(peer.size)) / peer.size
    raise ModelingError(f"peer_traffic_volume_basis_unknown:{basis}")


class _TopologyPlansMixin:
    def transport_plan(self, flows, *, chunk_bytes=None, receive_capacity_bytes=None,
                       hop_startup_us=0.0,
                       max_physical_chunks=DEFAULT_MAX_PHYSICAL_CHUNKS,
                       transport=None, receive_buffer_id=None):
        """Plan directed flows with actual routes, chunks and receive credits."""
        # Freeze once at entry so generator/iterator inputs are never exhausted
        # across capacity, buffer-ID, and path passes.
        frozen_flows = tuple(flows)
        if transport is not None:
            options = TransportOptions.from_mapping(transport)
            if (chunk_bytes is not None or receive_capacity_bytes is not None
                    or receive_buffer_id is not None
                    or float(hop_startup_us or 0.0)
                    or max_physical_chunks != DEFAULT_MAX_PHYSICAL_CHUNKS):
                raise ModelingError("transport_options_conflict")
            chunk_bytes = options.chunk_bytes
            receive_capacity_bytes = (
                dict(options.receive_capacity_bytes)
                if options.receive_capacity_bytes else None)
            if options.receive_buffer_ids:
                receive_buffer_id = dict(options.receive_buffer_ids)
            hop_startup_us = options.hop_startup_us
            max_physical_chunks = options.max_physical_chunks
        if max_physical_chunks is not None and (
                isinstance(max_physical_chunks, bool)
                or not isinstance(max_physical_chunks, int)
                or max_physical_chunks < 1):
            raise ModelingError(
                f"transport_max_physical_chunks_invalid:{max_physical_chunks}")
        if chunk_bytes is not None:
            try:
                chunk = float(chunk_bytes)
            except (TypeError, ValueError):
                raise ModelingError(
                    f"transport_chunk_bytes_invalid:{chunk_bytes}") from None
            if not math.isfinite(chunk) or chunk <= 0:
                raise ModelingError(f"transport_chunk_bytes_invalid:{chunk_bytes}")
        else:
            chunk = None
        try:
            startup = float(hop_startup_us or 0.0)
        except (TypeError, ValueError):
            raise ModelingError(
                f"transport_hop_startup_invalid:{hop_startup_us}") from None
        if not math.isfinite(startup) or startup < 0:
            raise ModelingError(f"transport_hop_startup_invalid:{hop_startup_us}")

        for flow in frozen_flows:
            if not isinstance(flow, MessageFlow):
                raise ModelingError("transport_flow_must_be_message_flow")

        capacity = []
        if receive_capacity_bytes is not None:
            if isinstance(receive_capacity_bytes, Mapping):
                items = receive_capacity_bytes.items()
            else:
                items = ((rank, receive_capacity_bytes)
                         for rank in sorted({flow.destination for flow in frozen_flows}))
            for rank, value in items:
                amount = float(value)
                if not math.isfinite(amount) or amount < 0:
                    raise ModelingError(
                        f"transport_receive_capacity_invalid:{rank}:{value}")
                capacity.append((int(rank), amount))

        buffer_ids = []
        if receive_buffer_id is not None:
            if isinstance(receive_buffer_id, Mapping):
                items = receive_buffer_id.items()
            else:
                items = ((rank, receive_buffer_id)
                         for rank in sorted({flow.destination for flow in frozen_flows}))
            for rank, value in items:
                text = str(value)
                if not text:
                    raise ModelingError(
                        f"transport_receive_buffer_id_invalid:{rank}")
                buffer_ids.append((int(rank), text))
        known = {rank for rank, _ in capacity}
        for rank in known:
            if not any(item[0] == rank for item in buffer_ids):
                buffer_ids.append(
                    (rank, f"{self.namespace}rank:{rank}"))
        buffer_ids = sorted(buffer_ids)

        paths = []
        for flow in frozen_flows:
            count = chunk_count(flow.payload_bytes, chunk)
            if max_physical_chunks is not None and count > max_physical_chunks:
                raise ModelingError(
                    "transport_chunk_expansion_unbounded:"
                    f"{count}>{max_physical_chunks}")
            cap = next((value for rank, value in capacity
                        if rank == flow.destination), None)
            if cap is not None and cap <= 0:
                raise ModelingError(
                    f"transport_receive_capacity_zero:{flow.destination}")
            largest = (chunk_size_at(flow.payload_bytes, chunk, 0)
                       if count else 0.0)
            if cap is not None and largest > cap + 1e-9:
                raise ModelingError(
                    "transport_chunk_exceeds_receive_capacity:"
                    f"{largest}>{cap}")
            chunks = ChunkSequence(
                self, flow, chunk, max_physical_chunks)
            paths.append(FlowPath(flow, chunks))

        unknowns = []
        if chunk is None:
            unknowns.append(("transfer_chunk_bytes",
                             "not_declared; one logical message per hop"))
        else:
            physical = max((len(path.chunks) for path in paths), default=0)
            unknowns.append((
                "transfer_physical_chunks",
                f"{physical}_declared_chunks_lazy"))
        if not capacity:
            unknowns.append(("receive_capacity_bytes",
                             "not_declared; no finite receive credit"))
        if not startup:
            unknowns.append(("protocol_startup_us",
                             "not_declared; no startup gate submitted"))
        return TransportPlan(
            self, tuple(paths), chunk, max_physical_chunks,
            tuple(sorted(capacity)), tuple(buffer_ids), startup,
            tuple(unknowns))

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
            for ranks in self.groups(axis, replica, stage, metadata=metadata):
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

    def joint_plan(self, op, participants, *, algorithm="ring", max_inflight_per_rank=1,
                   chunk_bytes=None, hop_startup_us=0.0,
                   receive_capacity_bytes_per_peer=None,
                   receive_buffer_id=None,
                   max_physical_chunks=DEFAULT_MAX_PHYSICAL_CHUNKS, transport=None):
        """Bind complete cross-DP cohorts without inferring a runtime algorithm."""
        if transport is not None:
            options = TransportOptions.from_mapping(transport)
            if (chunk_bytes is not None or receive_capacity_bytes_per_peer is not None
                    or receive_buffer_id is not None
                    or max_inflight_per_rank != 1
                    or float(hop_startup_us or 0.0)
                    or max_physical_chunks != DEFAULT_MAX_PHYSICAL_CHUNKS):
                raise ModelingError("transport_options_conflict")
            chunk_bytes = options.chunk_bytes
            hop_startup_us = options.hop_startup_us
            receive_capacity_bytes_per_peer = (
                options.receive_capacity_bytes_per_peer)
            receive_buffer_id = options.receive_buffer_id
            max_inflight_per_rank = options.max_inflight_per_rank
            max_physical_chunks = options.max_physical_chunks
        resolved = TransportOptions(
            chunk_bytes=chunk_bytes, hop_startup_us=hop_startup_us,
            receive_capacity_bytes_per_peer=receive_capacity_bytes_per_peer,
            max_physical_chunks=max_physical_chunks)
        if len(self._transport_bindings) >= MAX_TRANSPORT_BINDINGS:
            self._transport_bindings.clear()
        self._transport_bindings[id(op)] = resolved
        self._transport_bindings[getattr(op, "op_id", id(op))] = resolved
        if algorithm not in {"ring", "pairwise", "oneshot", "direct_peer", "direct_mesh",
                             "startup_waves"}:
            raise ModelingError(f"joint_collective_algorithm_unsupported:{algorithm}")
        if (algorithm not in {"direct_peer", "startup_waves"}
                and (chunk_bytes is not None or hop_startup_us
                     or receive_capacity_bytes_per_peer is not None)):
            raise ModelingError(
                f"{algorithm}_collective_does_not_accept_chunked_peer_options")
        family, unique_groups, volume_by_rank, metadata = self._joint_groups(
            op, participants)
        alltoall = str(family).strip().lower() in {"alltoall", "alltoallv"}
        startup = None
        if alltoall and unique_groups:
            startup = self.startup_descriptor(
                op, len(next(iter(unique_groups))))
        if algorithm == "direct_mesh":
            return self._joint_direct_mesh_plan(family, unique_groups, volume_by_rank, metadata)
        if algorithm == "oneshot":
            return self._joint_oneshot_plan(family, unique_groups, volume_by_rank, metadata)
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
            if alltoall:
                startup = self.startup_descriptor(op, len(ranks))
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
                from mindie_sim.prediction.operators.collectives.direct_peer import DirectPeerGroup
                groups.append(DirectPeerGroup(
                    ranks, tuple(totals), rounds, self, peer_traffic=peer,
                    max_inflight_per_rank=max_inflight_per_rank,
                    chunk_bytes=chunk_bytes,
                    hop_startup_us=hop_startup_us,
                    receive_capacity_bytes_per_peer=(
                        receive_capacity_bytes_per_peer),
                    receive_buffer_id=receive_buffer_id,
                    max_physical_chunks=max_physical_chunks))
            elif algorithm == "startup_waves":
                from ..collectives.waves import WaveGroup
                wave_rounds = (startup.startup_rounds
                               if startup is not None else rounds)
                groups.append(WaveGroup(
                    ranks, tuple(totals), wave_rounds, self, peer_traffic=peer,
                    max_inflight_per_rank=max_inflight_per_rank))
            else:
                groups.append(PairwiseGroup(
                    ranks, tuple(totals), rounds, self, peer_traffic=peer))
        if algorithm == "startup_waves":
            from ..collectives.waves import WavePlan
            return WavePlan(
                family, tuple(groups),
                startup_rounds=(startup.startup_rounds
                                if startup is not None else None))
        return PairwisePlan(
            family, tuple(groups),
            startup_rounds=(startup.startup_rounds if startup is not None else None),
            startup_application=("physical_program_owns_arrivals"
                                 if alltoall else "startup_model"))

    def _joint_oneshot_plan(self, family, unique_groups, volume_by_rank, metadata):
        """Bind a single-round all-peer one-shot plan from hardware link + launch physics.

        For ``allgather``, ``reduce_scatter``, and uniform ``alltoall``, each rank
        splits its declared TX volume equally across all ``G - 1`` remote peers in
        one concurrent round (utilizing every parallel pair on ``per_pair`` meshes
        and full egress port bandwidth on switched fabrics while conserving exact
        per-rank TX bytes).  For ``allreduce``, each rank broadcasts its full
        logical tensor ``M_api`` directly to every remote peer so small/medium
        collectives save ``(2G - 3)`` hop latencies while large payloads naturally
        yield to Ring on total link service.  Explicit ``peer_traffic`` matrices
        (``alltoall`` / ``alltoallv``) emit each directed peer payload verbatim.
        """
        from .band_plan import api_bytes_from_link_volume

        norm = _COLLECTIVE_FAMILY_ALIASES.get(str(family).strip().lower(), str(family).strip().lower())
        if norm not in ONESHOT_FAMILIES:
            raise ModelingError(f"oneshot_collective_family_unsupported:{family}")
        peer = None
        if "peer_traffic" in metadata:
            if norm not in {"alltoall", "alltoallv"}:
                raise ModelingError("peer_traffic_requires_peer_collective")
            sizes = {len(ranks) for ranks in unique_groups}
            if len(sizes) > 1:
                raise ModelingError("peer_traffic_group_size_inconsistent")
            peer = _declared_peer_traffic(metadata, next(iter(sizes))) if sizes else None
        elif norm == "alltoallv":
            raise ModelingError(
                "oneshot_collective_alltoallv_requires_explicit_peer_matrix")
        startup_ms = self._oneshot_startup_ms()
        groups = []
        for ranks in unique_groups:
            count = len(ranks)
            volumes = {rank: float(volume_by_rank[rank]) for rank in ranks}
            if count <= 1 or not any(volumes.values()):
                groups.append(OneShotGroup(ranks, (), startup_ms=0.0))
                continue
            routed = []
            if peer is not None:
                if peer.size != count:
                    raise ModelingError(
                        f"peer_traffic_group_size_mismatch:{peer.size}!={count}")
                for src_idx, src in enumerate(ranks):
                    declared = volumes[src]
                    if not _peer_totals_match(
                            _peer_declared_volume(peer, metadata, src_idx), declared):
                        raise ModelingError(
                            f"peer_traffic_source_volume_mismatch:{src}:"
                            f"{peer.row_total(src_idx)}!={declared}")
                    for dst_idx, dst in enumerate(ranks):
                        if src_idx == dst_idx:
                            continue
                        payload = float(peer.payload(src_idx, dst_idx))
                        if payload > 0.0:
                            routed.append(self.route_message(MessageFlow(src, dst, payload)))
            elif norm == "allreduce":
                for src in ranks:
                    vol = volumes[src]
                    if vol <= 0.0:
                        continue
                    per_peer = float(api_bytes_from_link_volume(norm, count, vol, metadata))
                    if per_peer <= 0.0:
                        continue
                    for dst in ranks:
                        if dst != src:
                            routed.append(self.route_message(MessageFlow(src, dst, per_peer)))
            else:
                peers = float(count - 1)
                for src in ranks:
                    vol = volumes[src]
                    if vol <= 0.0:
                        continue
                    per_peer = vol / peers
                    for dst in ranks:
                        if dst != src:
                            routed.append(self.route_message(MessageFlow(src, dst, per_peer)))
            groups.append(OneShotGroup(ranks, tuple(routed), startup_ms=startup_ms))
        return OneShotPlan(norm, tuple(groups))

    def _joint_direct_mesh_plan(self, family, unique_groups, volume_by_rank, metadata):
        from mindie_sim.prediction.operators.collectives.direct_peer import DirectMeshGroup, DirectMeshPhase, DirectMeshPlan

        family = {"all_reduce": "allreduce", "all_gather": "allgather"}.get(family, family)
        if family not in DIRECT_MESH_FAMILIES:
            raise ModelingError(
                f"direct_mesh_collective_family_unsupported:{family}")
        names = DIRECT_MESH_PHASES[family]
        groups = []
        for ranks in unique_groups:
            if len(ranks) <= 1:
                continue
            if len(ranks) > DIRECT_MESH_MAX_PEERS:
                raise ModelingError(
                    f"direct_mesh_collective_exceeds_declared_peers:{len(ranks)}")
            volumes = {rank: volume_by_rank[rank] for rank in ranks}
            if len(set(volumes.values())) > 1:
                raise ModelingError(
                    "direct_mesh_collective_requires_equal_per_rank_volume")
            if len({self._rank_node[rank] for rank in ranks}) != 1:
                raise ModelingError(
                    "direct_mesh_collective_requires_single_node_group")
            rounds = len(names) * (len(ranks) - 1)
            if metadata.get("steps", rounds) != rounds:
                raise ModelingError(
                    "direct_mesh_collective_round_count_conflicts_with_declaration")
            links = []
            for source in ranks:
                for destination in ranks:
                    if source == destination:
                        continue
                    path = self.route(source, destination)
                    if len(path) != 1:
                        raise ModelingError(
                            "direct_mesh_collective_requires_one_hop_peer")
                    link = self.links[path[0][0]]
                    if link.semantics != "per_pair":
                        raise ModelingError(
                            "direct_mesh_collective_requires_per_pair_links")
                    links.append(link)
            volume = next(iter(volumes.values()), 0.0)
            payload = volume / rounds if rounds else 0.0
            startup_ms = self.mesh_phase_startup_ms(links)
            phase_startups = [startup_ms] * len(names)
            from .band_plan import api_bytes_from_link_volume
            from .protocol import (
                MS_PER_S as _MS_PER_S,
                _candidate_bands,
                _infer_group_scope,
                select_band,
            )
            if _candidate_bands(ranks, self, _infer_group_scope(ranks, self)):
                api_b = api_bytes_from_link_volume(family, len(ranks), volume, metadata)
                band = select_band(family, ranks, self, api_b)
                band_ms = float(band.startup_us) / _MS_PER_S
                if band.startup_scope == "collective":
                    phase_startups = [band_ms] + [0.0] * max(0, len(names) - 1)
                else:
                    phase_startups = [band_ms] * len(names)
            groups.append(DirectMeshGroup(
                ranks, tuple((rank, volumes[rank]) for rank in ranks),
                tuple(DirectMeshPhase(
                    ranks, payload, self, phase=name, startup_ms=p_ms)
                    for name, p_ms in zip(names, phase_startups)), self))
        return DirectMeshPlan(family, tuple(groups))

    def ring_plan(self, op, replica, *, volume=None):
        """Bind a uniform-chunk local ring; no vendor algorithm is inferred."""
        volume = op.link_bytes if volume is None else volume
        try:
            plan = self.joint_plan(op, {replica: volume}, algorithm="ring")
        except ModelingError as error:
            if str(error).startswith("joint_collective_group_membership"):
                raise ModelingError(
                    "ring_collective_requires_complete_local_group") from None
            raise
        return RingPlan(plan.family, tuple(
            group for group in plan.groups
            if len(group.ranks) > 1 and group.rounds and group.messages))

    def pairwise_plan(self, op, replica, *, volume=None):
        """Bind a balanced local pairwise declaration; peers stay uniform."""
        volume = op.link_bytes if volume is None else volume
        return self.joint_plan(op, {replica: volume}, algorithm="pairwise")

    def oneshot_plan(self, op, replica, *, volume=None):
        """Bind a single-round concurrent local one-shot collective."""
        volume = op.link_bytes if volume is None else volume
        try:
            plan = self.joint_plan(op, {replica: volume}, algorithm="oneshot")
        except ModelingError as error:
            if str(error).startswith("joint_collective_group_membership"):
                raise ModelingError(
                    "oneshot_collective_requires_complete_local_group") from None
            raise
        return OneShotPlan(plan.family, tuple(
            group for group in plan.groups
            if len(group.ranks) > 1 and group.messages))

    def _accounting_flows(self, flow, chunk_bytes,
                          max_physical_chunks=DEFAULT_MAX_PHYSICAL_CHUNKS):
        """Physical framing units for the public byte ledger."""
        if chunk_bytes is None:
            return (flow,)
        sizes = chunk_sizes(
            flow.payload_bytes, chunk_bytes, max_physical_chunks)
        return tuple(MessageFlow(flow.source, flow.destination, size,
                                 flow.messages) for size in sizes)

    def accounting(self, op, replica, *, volume=None, transport=None,
                   recipe=None):
        """Account the one selected physical plan."""
        mesh = self.bound_mesh_plan(op, replica)
        if mesh is not None:
            return self._work_with_capacity(mesh.accounting())
        if transport is None and recipe is not None:
            from mindie_sim.prediction.operators.collectives.direct_peer import resolve_transport_options
            transport = resolve_transport_options(recipe, op)
        if transport is None:
            transport = self._transport_bindings.get(id(op))
        if transport is None:
            transport = self._transport_bindings.get(getattr(op, "op_id", None))
        if transport is not None:
            options = TransportOptions.from_mapping(transport)
            chunk = options.chunk_bytes
            max_physical_chunks = options.max_physical_chunks
        else:
            metadata = dict(op.metadata)
            chunk = metadata.get("transport_chunk_bytes")
            max_physical_chunks = metadata.get(
                "transport_max_physical_chunks", DEFAULT_MAX_PHYSICAL_CHUNKS)
        volume = op.link_bytes if volume is None else volume
        if chunk is not None:
            try:
                chunk = float(chunk)
            except (TypeError, ValueError):
                raise ModelingError(
                    f"transport_chunk_bytes_invalid:{chunk}") from None
            if not math.isfinite(chunk) or chunk <= 0:
                raise ModelingError(
                    f"transport_chunk_bytes_invalid:{chunk}")
        useful_by_rank, useful, wire, transactions = {}, {}, {}, {}
        transaction_frames = {}
        latency, max_path_latency, count = 0.0, 0.0, 0
        routed_useful, physical_wire = 0.0, 0.0
        for flow, rounds in self.messages(op, replica, volume):
            for physical in self._accounting_flows(
                    flow, chunk, max_physical_chunks):
                useful_by_rank[physical.source] = (
                    useful_by_rank.get(physical.source, 0)
                    + physical.useful_bytes)
                count += physical.messages
                path_latency = 0
                for hop in self.route_message(physical).hops:
                    routed_useful += hop.useful_bytes
                    physical_wire += hop.wire_bytes
                    for resource in hop.demand:
                        useful[resource] = useful.get(resource, 0) + hop.useful_bytes
                        wire[resource] = wire.get(resource, 0) + hop.wire_bytes
                        transactions[resource] = (
                            None if hop.transactions is None
                            else transactions.get(resource, 0) + hop.transactions)
                        if hop.transaction_frame is not None:
                            transaction_frames[resource] = hop.transaction_frame
                    path_latency += hop.latency_ms
                max_path_latency = max(max_path_latency, path_latency)
                latency = max(latency, path_latency * rounds)
        return self._work_with_capacity(CommunicationWork(
            useful_by_rank, useful, wire, transactions, transaction_frames, count, latency,
            routed_useful, physical_wire, max_path_latency,
        ))

    def band_collective_plan(
        self, op, participants, *, algorithm="ring", transport=None,
        expansion_mode=None, stack="default",
    ):
        """Build one unified ``BandCollectivePlan`` when protocol bands apply.

        Returns ``None`` when no traversed links/channels declare
        ``protocol_bands``, when explicit non-band ``transport`` options or
        non-band algorithms are requested, or when group size ``<= 1``.
        """
        if algorithm not in {"ring", "pairwise"}:
            return None
        if transport is not None:
            opts = (
                TransportOptions.from_mapping(transport)
                if isinstance(transport, Mapping)
                else transport
            )
            if (opts.chunk_bytes is not None
                    or float(opts.hop_startup_us or 0.0) > 0.0
                    or opts.receive_capacity_bytes_per_peer is not None
                    or opts.receive_buffer_id is not None):
                return None
        from .band_plan import build_band_collective_plan
        from .protocol import SUPPORTED_COLLECTIVES, _candidate_bands, _infer_group_scope

        raw_meta = dict(op.metadata)
        raw_family = str(raw_meta.get("collective", op.family)).strip().lower()
        aliases = {"all_reduce": "allreduce", "all_gather": "allgather",
                   "all_to_all": "alltoall", "all_to_all_v": "alltoallv"}
        norm_family = aliases.get(raw_family, raw_family)
        if norm_family not in SUPPORTED_COLLECTIVES:
            return None
        raw_axis = raw_meta.get("group_axis", op.link_scope)
        if raw_axis == "pp":
            return None

        try:
            family, unique_groups, volume_by_rank, metadata = self._joint_groups(
                op, participants
            )
        except ModelingError as exc:
            if (not str(exc).startswith("joint_collective_group_membership")
                    or len(participants) != 1 or transport is None
                    or "peer_traffic" in raw_meta):
                raise
            replica = next(iter(participants))
            vol = float(participants[replica])
            source_stage = int(raw_meta.get("source_stage", op.stage))
            axis = raw_meta.get("group_axis", "tp")
            groups = tuple(self.groups(axis, replica, source_stage, metadata=raw_meta))
            if not groups or all(len(ranks) <= 1 for ranks in groups):
                return None
            family = raw_family
            unique_groups = tuple(dict.fromkeys(groups))
            volume_by_rank = {rank: vol for ranks in unique_groups for rank in ranks}
            metadata = raw_meta
        norm_family = aliases.get(str(family).strip().lower(), str(family).strip().lower())
        if norm_family not in SUPPORTED_COLLECTIVES:
            return None
        if not unique_groups or all(len(ranks) <= 1 for ranks in unique_groups):
            return None
        if any(
            not _candidate_bands(ranks, self, _infer_group_scope(ranks, self))
            for ranks in unique_groups
            if len(ranks) > 1
        ):
            return None
        if algorithm == "ring":
            if "peer_traffic" in metadata:
                return None
            phases = {"allreduce": 2, "reduce_scatter": 1, "allgather": 1}
            if norm_family in phases:
                for ranks in unique_groups:
                    rounds = max(phases[norm_family] * (len(ranks) - 1), 0)
                    if metadata.get("steps", rounds) != rounds:
                        return None
            elif norm_family != "alltoall":
                return None
        peer = None
        if "peer_traffic" in metadata:
            if norm_family not in {"alltoall", "alltoallv"}:
                return None
            sizes = {len(ranks) for ranks in unique_groups}
            if len(sizes) != 1:
                return None
            peer = _declared_peer_traffic(metadata, next(iter(sizes)))
            for ranks in unique_groups:
                for index, rank in enumerate(ranks):
                    declared = volume_by_rank[rank]
                    if not _peer_totals_match(
                            _peer_declared_volume(peer, metadata, index), declared):
                        return None
        elif norm_family == "alltoallv":
            return None
        exp = expansion_mode or metadata.get("expansion_mode")
        stack_id = str(metadata.get("stack_id") or stack or "default")
        try:
            return build_band_collective_plan(
                self, norm_family, unique_groups, volume_by_rank, metadata,
                peer=peer, stack=stack_id, expansion_mode=exp,
            )
        except ModelingError as exc:
            if str(exc).startswith("collective_band_missing"):
                return None
            raise

    def demand(self, op, replica, *, volume=None, transport=None, recipe=None):
        work = self.accounting(
            op, replica, volume=volume, transport=transport, recipe=recipe)
        return work.wire_by_resource, work.setup_ms
