"""Launch-local MC2 payload and readiness state.

[Role]
  Hold the immutable EP peer view, routed payload policy and endpoint-memory
  bookkeeping used by MC2 phase launches.

[Invariants]
  * A peer class is derived from the declared rank/node topology, never from a
    representative program's label.
  * Peer weights over a nonempty peer set sum to one; an empty set is legal.
  * Destination HBM movement is apportioned across every owning phase.
  * No timing, capacity or rate is inferred from a measured duration.

[Timeouts & Locks]
  Pure in-memory state; no I/O, device work or locks.

[Side-Effect Policy]
  Returns immutable views; the cached representative state is never mutated.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.program import RecipeContext
from .mc2_program import (
    MIN_MC2_AIV_UNITS, Mc2LaunchView, _as_fraction, _peer_class_for,
    _resolve_aiv_units, _subtract_endpoint_hbm,
)
from .mc2_fullmesh import (
    EXPERT_STATUS_RECORD_BYTES, directed_cell_payload, directed_transport_cells,
)

TARGET_BACKEND = "ascend"
_OPS_REVISION = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_dispatch_v2/op_kernel/arch35/moe_distribute_dispatch_v2_a5_full_mesh.h",
        source_sha256="3194ed49e10aab74976053840a5e771051be946d0945664fd5a3ca34a32bdb6b",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=_OPS_REVISION,
        relative_path="mc2/moe_distribute_combine_v2/op_kernel/arch35/moe_distribute_combine_v2_a5_mte.h",
        source_sha256="4e2a3f571df2e0e3718c8e61145638483be15445383ed77c3873381874ffb5a9",
    ),
)


class Mc2ProgramState:
    """Launch-local peer/payload policy shared with the phase scheduler."""

    def __init__(self, phases, plans, link_by_phase, latency, rates, units,
                 names, service_modes, peer_count=1, rank=-1,
                 peer_weights=None, node_ranks=0, peer_classes=None,
                 latencies=None, resource_names=None, payload_frames=None,
                 payload_packet_bytes=0, payload_basis="unknown",
                 payload_local_frames=0, route_matrix=None,
                 route_reverse=False, exact_payload=False,
                 route_matrix_bytes=None, uniform_payload_bytes=0,
                 experts=0, transport_basis=None, rank_nodes=None):
        self.phases = tuple(phases)
        self.plans = dict(plans)
        self.link_by_phase = dict(link_by_phase)
        self.rates = dict(rates)
        self.units = dict(units)
        self.names = dict(names)
        self.service_modes = dict(service_modes)
        self.peer_count = max(1, int(peer_count))
        peers = max(0, self.peer_count - 1)
        classes = tuple(sorted(
            (int(peer), ("hccs" if str(value).endswith("hccs") or
                         str(value).lower() in {"local", "intra", "same_node"}
                         else "roce"))
            for peer, value in dict(peer_classes or {}).items()))
        if peers:
            if peer_weights is None:
                weights = tuple(Fraction(1, peers) for _ in range(peers))
            else:
                weights = tuple(_as_fraction(item) for item in peer_weights)
                if (len(weights) != peers
                        or sum(weights, Fraction(0)) != 1):
                    raise ModelingError("mc2_peer_weights_invalid")
        else:
            weights = ()
        if rank >= 0 and rank < self.peer_count:
            labels = tuple(item for item in range(self.peer_count)
                           if item != int(rank))
        else:
            labels = tuple(range(peers))
        if latencies is None:
            intra = inter = _as_fraction(latency)
        else:
            intra, inter = (_as_fraction(item) for item in latencies)
        self.view = Mc2LaunchView(
            rank=int(rank), peer_labels=labels, peer_classes=classes,
            peer_weights=weights, node_ranks=max(0, int(node_ranks)),
            intra_latency_ms=intra, inter_latency_ms=inter,
            resource_names=tuple(sorted(
                (str(name), str(value))
                for name, value in dict(resource_names or {}).items())),
            payload_frames=tuple(payload_frames or ()),
            payload_packet_bytes=int(payload_packet_bytes or 0),
            payload_basis=str(payload_basis),
            payload_local_frames=int(payload_local_frames))
        self.aiv = _resolve_aiv_units(self.units)
        #: Exact source force-EPLB assignment cells, indexed [source][expert
        #: destination].  They are whole routed rows, never a statistical
        #: expectation promoted to observed destinations.
        self.route_matrix = (
            tuple(tuple(int(value) for value in row) for row in route_matrix)
            if route_matrix else None)
        self.route_reverse = bool(route_reverse)
        self.exact_payload = bool(exact_payload)
        #: Expected directed payload bytes from a declared statistical routing
        #: policy.  These are policy expectations, not integers observed at a
        #: peer, so they never populate payload_frames.
        self.route_matrix_bytes = (
            tuple(tuple(float(value) for value in row)
                  for row in route_matrix_bytes)
            if route_matrix_bytes else None)
        #: Declared useful payload bytes one source sends across all peers for
        #: a uniform peer policy.  Each edge gets an equal share.
        self.uniform_payload_bytes = float(uniform_payload_bytes or 0)
        self.experts = max(0, int(experts))
        #: Whole-call policy basis.  A rank-local frame set may be empty even
        #: when other EP sources are active; that does not make the operation a
        #: known-zero source.
        self.transport_basis = str(transport_basis or payload_basis)
        self.rank_nodes: dict[int, int] = {
            int(r): int(n) for r, n in dict(rank_nodes or ()).items()
        }
        #: Mapped phase service work, excluding peer link bytes.  A launch-local
        #: rank adds its own exact directed link work below.
        self._phase_work = {}
        self.work = ()

    def frames_for(self, rank):
        """Whole routed frames this launch rank sends to each remote peer."""
        matrix = self.route_matrix
        if not matrix:
            return ()
        rank = int(rank)
        if not 0 <= rank < len(matrix):
            return ()
        frames = []
        for peer in range(len(matrix)):
            if peer == rank:
                continue
            count = (matrix[peer][rank] if self.route_reverse
                     else matrix[rank][peer])
            if count > 0:
                frames.append((peer, int(count)))
        return tuple(sorted(frames))

    def local_frames_for(self, rank):
        matrix = self.route_matrix
        if not matrix:
            return 0
        rank = int(rank)
        if not 0 <= rank < len(matrix):
            return 0
        return int(matrix[rank][rank])

    def payload_basis_for(self, rank):
        if not self.route_matrix:
            return self.view.payload_basis
        return ("actual_zero" if not self.frames_for(rank)
                else self.view.payload_basis)

    @property
    def has_payload_policy(self):
        return bool(self.route_matrix is not None
                    or self.route_matrix_bytes is not None
                    or self.uniform_payload_bytes > 0)

    def payload_bytes(self, source, destination):
        """Useful payload bytes for one directed source->destination cell."""
        useful, _frames = directed_cell_payload(
            source,
            destination,
            peer_count=self.peer_count,
            packet_bytes=self.view.payload_packet_bytes,
            route_matrix=self.route_matrix,
            route_reverse=self.route_reverse,
            route_matrix_bytes=self.route_matrix_bytes,
            uniform_payload_bytes=self.uniform_payload_bytes,
        )
        return useful

    def frame_count(self, source, destination):
        if not self.route_matrix or source == destination:
            return 0
        return int(self.route_matrix[destination][source]
                   if self.route_reverse else self.route_matrix[source][destination])

    def count_bytes(self, source_index, group_size):
        ep = max(1, int(group_size))
        if ep <= 1:
            return 0.0
        return float(self.experts * EXPERT_STATUS_RECORD_BYTES * (ep - 1)) / ep

    def transport_cells(self, transport, count=False):
        """Directed cells this launch owns under its actual EP membership."""
        return directed_transport_cells(
            group=tuple(transport.get("group") or ()),
            sources=tuple(transport.get("sources") or ()),
            experts=self.experts,
            packet_bytes=self.view.payload_packet_bytes,
            route_matrix=self.route_matrix,
            route_reverse=self.route_reverse,
            route_matrix_bytes=self.route_matrix_bytes,
            uniform_payload_bytes=self.uniform_payload_bytes,
            count=count,
        )

    def transport_account(self, transport):
        """Physical coupled work for every directed cell this launch owns."""
        from mindie_sim.prediction.communication import MessageFlow
        route = transport.get("route")
        payload = {"work": {}, "edges": 0, "frames": 0,
                   "tx_bytes": 0.0, "rx_bytes": 0.0, "useful": 0.0,
                   "useful_bytes": 0.0, "sources": tuple(transport.get("sources") or ()),
                   "group": tuple(transport.get("group") or ())}
        for _si, _di, source, destination, useful, frames in self.transport_cells(
                transport):
            if not callable(route) or useful <= 0:
                continue
            message = route(MessageFlow(source, destination, float(useful)))
            for hop in message.hops:
                for name, amount in hop.demand.items():
                    if not amount:
                        continue
                    payload["work"][str(name)] = (
                        payload["work"].get(str(name), 0.0) + float(amount))
                    lowered = str(name).lower()
                    if lowered.startswith("ub_inj:") or ":ub_inj:" in lowered:
                        continue
                    if ":tx:" in lowered:
                        payload["tx_bytes"] += float(amount)
                    elif ":rx:" in lowered:
                        payload["rx_bytes"] += float(amount)
                    else:
                        # A directed per-pair resource is both source TX and
                        # destination RX of the one transfer.
                        payload["tx_bytes"] += float(amount)
                        payload["rx_bytes"] += float(amount)
            payload["edges"] += 1
            payload["frames"] += int(frames)
        payload["useful"] = sum(
            float(useful) for _si, _di, _source, _destination, useful, _frames
            in self.transport_cells(transport) if useful > 0)
        payload["useful_bytes"] = payload["useful"]
        count = {"work": {}, "edges": 0, "frames": 0,
                 "tx_bytes": 0.0, "rx_bytes": 0.0, "useful": 0.0,
                 "useful_bytes": 0.0, "sources": payload["sources"], "group": payload["group"]}
        for _si, _di, source, destination, useful, _frames in self.transport_cells(
                transport, count=True):
            if not callable(route) or useful <= 0:
                continue
            message = route(MessageFlow(source, destination, float(useful)))
            for hop in message.hops:
                for name, amount in hop.demand.items():
                    if not amount:
                        continue
                    count["work"][str(name)] = (
                        count["work"].get(str(name), 0.0) + float(amount))
                    lowered = str(name).lower()
                    if lowered.startswith("ub_inj:") or ":ub_inj:" in lowered:
                        continue
                    if ":tx:" in lowered:
                        count["tx_bytes"] += float(amount)
                    elif ":rx:" in lowered:
                        count["rx_bytes"] += float(amount)
                    else:
                        count["tx_bytes"] += float(amount)
                        count["rx_bytes"] += float(amount)
            count["edges"] += 1
            count["useful"] += float(useful)
            count["useful_bytes"] += float(useful)
        return payload, count

    def _edge_class(self, source_rank: int, index: int, peer: int) -> str:
        """Classify directed ``(source_rank, peer)`` edge from physical node placement."""
        src, dst = int(source_rank), int(peer)
        if src in self.rank_nodes and dst in self.rank_nodes:
            return "hccs" if self.rank_nodes[src] == self.rank_nodes[dst] else "roce"
        classes = self.peer_classes if src == self.view.rank else ()
        return _peer_class_for(
            src,
            index,
            dst,
            node_ranks=self.view.node_ranks,
            peer_count=self.peer_count,
            classes=classes,
        )

    def _directed_link(self, rank):
        totals = {}
        labels = tuple(peer for peer in range(self.peer_count) if peer != rank)
        if self.route_matrix_bytes:
            # Statistical expectation uses the same directed matrix that drives
            # the native peer program.  It is not a second, independently
            # uniform distribution of the local receive volume.
            if 0 <= int(rank) < self.peer_count:
                for index, peer in enumerate(labels):
                    useful = self.payload_bytes(rank, peer)
                    if useful <= 0:
                        continue
                    klass = self._edge_class(rank, index, peer)
                    key = "link_hccs" if klass == "hccs" else "link_roce"
                    totals[key] = totals.get(key, Fraction(0)) + Fraction(useful)
            return totals or {"link": Fraction(0)}
        frames = self.frames_for(rank)
        packet = self.view.payload_packet_bytes
        for index, (peer, count) in enumerate(frames):
            klass = self._edge_class(rank, index, peer)
            key = "link_hccs" if klass == "hccs" else "link_roce"
            totals[key] = totals.get(key, Fraction(0)) + Fraction(count * packet)
        # A zero-frame rank still owns the named payload phase.  The empty
        # link ledger must not be mistaken for a declaration that the phase
        # has no peers at all; launch_native_cohort allocates the real edges.
        return totals or {"link": Fraction(0)}

    def _directed_count_link(self, rank):
        totals = {}
        if self.peer_count <= 1 or not 0 <= int(rank) < self.peer_count:
            return totals
        per_peer = Fraction(max(0, self.experts // self.peer_count) * EXPERT_STATUS_RECORD_BYTES)
        if per_peer <= 0:
            return totals
        labels = tuple(peer for peer in range(self.peer_count) if peer != int(rank))
        for index, peer in enumerate(labels):
            klass = self._edge_class(rank, index, peer)
            key = "link_hccs" if klass == "hccs" else "link_roce"
            totals[key] = totals.get(key, Fraction(0)) + per_peer
        return totals

    def link_by_phase_for(self, rank):
        links = {phase: dict(values)
                 for phase, values in self.link_by_phase.items()}
        if "count_status_exchange" in links:
            count_links = self._directed_count_link(rank)
            if count_links:
                links["count_status_exchange"] = count_links
        if self.route_matrix or self.route_matrix_bytes:
            links["payload_exchange"] = self._directed_link(rank)
        return links

    def work_for_rank(self, rank):
        work = dict(self._phase_work)
        for values in self.link_by_phase_for(rank).values():
            for pipe, amount in values.items():
                exact = _as_fraction(amount)
                if not exact:
                    continue
                key = self.names.get(pipe, pipe)
                work[key] = work.get(key, Fraction(0)) + exact
        return tuple(sorted(work.items()))

    @property
    def rank(self) -> int:
        return self.view.rank

    @property
    def node_ranks(self) -> int:
        return self.view.node_ranks

    @property
    def peer_labels(self) -> tuple[int, ...]:
        return self.view.peer_labels

    @property
    def peer_classes(self) -> tuple[tuple[int, str], ...]:
        return self.view.peer_classes

    @property
    def peer_weights(self) -> tuple[Fraction, ...]:
        return self.view.peer_weights

    def _peer_class(self, index: int, peer: int,
                    view: Mc2LaunchView | None = None) -> str:
        return (view or self.view).class_for(index, peer)

    @property
    def peer_class_basis(self) -> int:
        return int(bool(self.view.peer_classes) or
                   (self.view.rank >= 0 and self.view.node_ranks > 0))

    def _endpoint_moved(self, transport, phase, count, view=None):
        endpoints = {
            int(rank): str(name) for rank, name in
            (dict(transport or {}).get("destination_memory") or {}).items()
            if name}
        group = tuple(int(member) for member in ((transport or {}).get("group") or ()))
        if not endpoints or not group:
            return 0.0
        rank_value = int((view or self.view).rank)
        sources = tuple(int(item) for item in ((transport or {}).get("sources") or ()))
        if len(sources) == 1 and 0 <= sources[0] < len(group):
            destination_index = sources[0]
        elif 0 <= rank_value < len(group):
            destination_index = rank_value
        elif rank_value in group:
            destination_index = group.index(rank_value)
        else:
            return 0.0
        physical_dest = group[destination_index]
        if not endpoints.get(physical_dest):
            return 0.0
        total = 0.0
        for source_index in range(len(group)):
            if source_index == destination_index or group[source_index] == physical_dest:
                continue
            if count:
                useful = float(max(0, self.experts // len(group)) * EXPERT_STATUS_RECORD_BYTES)
            else:
                useful = self.payload_bytes(source_index, destination_index)
            if useful > 0:
                total += float(useful)
        return total

    def _endpoint_phase_program(self, program, transport, phase, count,
                                 view=None, resource_names=None):
        moved = self._endpoint_moved(transport, phase, count, view)
        if moved <= 0:
            return program, 0.0
        resource = dict(resource_names or {}).get(
            "hbm", self.names.get("hbm", "hbm"))
        return _subtract_endpoint_hbm(program, resource, moved)

    def _context(self, view=None):
        view = view or self.view
        names = dict(self.names)
        names.update(dict(view.resource_names))
        return RecipeContext(
            op=None, work={}, resource_names=names, rates=self.rates,
            core_count=self.aiv, resource_units=self.units, pool_names={},
            store_queues={}, service_modes=self.service_modes)
