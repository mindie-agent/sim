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
from .program import RecipeContext
from .mc2_program import (
    MIN_MC2_AIV_UNITS, Mc2LaunchView, _as_fraction, _peer_class_for,
    _resolve_aiv_units, _subtract_endpoint_hbm,
)
from .operators.mc2_fullmesh import EXPERT_STATUS_RECORD_BYTES



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
                 experts=0, transport_basis=None):
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
        source, destination = int(source), int(destination)
        if source == destination:
            return 0.0
        if self.route_matrix:
            count = (self.route_matrix[destination][source]
                     if self.route_reverse else self.route_matrix[source][destination])
            return float(count) * float(self.view.payload_packet_bytes)
        if self.route_matrix_bytes:
            return float(self.route_matrix_bytes[source][destination])
        ep = max(1, self.peer_count)
        return (self.uniform_payload_bytes / ep
                if ep > 1 else 0.0)

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
        group = tuple(transport.get("group") or ())
        sources = tuple(transport.get("sources") or ())
        if len(group) <= 1:
            return ()
        cells = []
        for source in sources:
            if not 0 <= source < len(group):
                continue
            if count:
                # Contiguous expert intervals: every remote peer receives its
                # own (experts/ep) status record, not the whole count on a ring
                # neighbour.  The directed arrivals stay independent.
                experts_per_peer = max(0, self.experts // len(group))
                payload = float(experts_per_peer * EXPERT_STATUS_RECORD_BYTES)
                if payload > 0:
                    for destination in range(len(group)):
                        if destination == source:
                            continue
                        cells.append((source, destination, group[source],
                                      group[destination], payload, 0))
                continue
            for destination in range(len(group)):
                if destination == source:
                    continue
                payload = self.payload_bytes(source, destination)
                if payload <= 0:
                    continue
                cells.append((source, destination, group[source],
                              group[destination], payload,
                              self.frame_count(source, destination)))
        return tuple(cells)

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
                    klass = _peer_class_for(
                        rank, index, peer, node_ranks=self.view.node_ranks,
                        peer_count=self.peer_count, classes=self.peer_classes)
                    key = "link_hccs" if klass == "hccs" else "link_roce"
                    totals[key] = totals.get(key, Fraction(0)) + Fraction(useful)
            return totals or {"link": Fraction(0)}
        frames = self.frames_for(rank)
        packet = self.view.payload_packet_bytes
        for index, (peer, count) in enumerate(frames):
            klass = _peer_class_for(
                rank, index, peer, node_ranks=self.view.node_ranks,
                peer_count=self.peer_count, classes=self.peer_classes)
            key = "link_hccs" if klass == "hccs" else "link_roce"
            totals[key] = totals.get(key, Fraction(0)) + Fraction(count * packet)
        # A zero-frame rank still owns the named payload phase.  The empty
        # link ledger must not be mistaken for a declaration that the phase
        # has no peers at all; launch_native_cohort allocates the real edges.
        return totals or {"link": Fraction(0)}

    def link_by_phase_for(self, rank):
        links = {phase: dict(values)
                 for phase, values in self.link_by_phase.items()}
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
        group = tuple((transport or {}).get("group") or ())
        destination = int((view or self.view).rank)
        if not endpoints or destination not in group or not endpoints.get(destination):
            return 0.0
        destination_index = group.index(destination)
        total = 0.0
        for source_index in range(len(group)):
            if source_index == destination_index:
                continue
            if count:
                useful = float(self.experts * EXPERT_STATUS_RECORD_BYTES / max(1, len(group)))
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
