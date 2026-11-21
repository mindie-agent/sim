"""Native participant launch views, resource bindings, and evidence summaries.

[Role]
  Bind native pipe names to physical stage/topology ports, construct immutable
  per-launch ``Mc2LaunchView`` snapshots, start native peer participants on
  the shared ``Engine``, and record bounded launch summaries.

[Invariants]
  * Cached native programs are never mutated; per-launch views are freshly
    constructed immutable snapshots.
  * Summary output is bounded by ``NATIVE_GROUP_SUMMARY_LIMIT``.

[Timeouts & Locks]
  Bounded in-memory resource lookups and dict reductions; no I/O or locks.

[Side-Effect Policy]
  Submits tasks only to ``self.engine`` and records bounded launch dictionaries.
"""
from __future__ import annotations

from mindie_sim.schema.common import ModelingError

NATIVE_GROUP_SUMMARY_LIMIT = 64


class _CohortNativeMixin:
    """Native participant launching, port binding, and bounded launch records."""

    def participant_peer_ready(self, replica, source_stage, ready_by_rank):
        """Actual peer ready Events for one participant, keyed by EP rank."""
        local_ranks = set(self.topology.ranks(replica, source_stage))
        for group in self.topology.groups("ep", replica, source_stage):
            if local_ranks & set(group):
                return {index: ready_by_rank[rank]
                        for index, rank in enumerate(group)
                        if rank not in local_ranks}
        return {rank: event for rank, event in ready_by_rank.items()
                if rank not in local_ranks}

    def participant_rank_index(self, replica, source_stage):
        """Local EP rank index for one participant (TP1/MC2 ingress)."""
        local_ranks = tuple(self.topology.ranks(replica, source_stage))
        for group in self.topology.groups("ep", replica, source_stage):
            for rank in local_ranks:
                if rank in group:
                    return group.index(rank)
        return None

    def _peer_ready_for(self, item, local_ready_by_rank, source_stage):
        return self.participant_peer_ready(
            item.replica, source_stage, local_ready_by_rank)

    def _topology_class_resource(self, rank, pipe):
        """Actual TX port for this rank/class, or None when not port-scoped."""
        rank = int(rank)
        kind = "roce" if pipe == "link_roce" else "hccs"
        links = list(getattr(self.topology, "links", {}).values())
        preferred = [link for link in links
                     if (kind == "roce") == ("roce" in link.name.lower()
                                             or "inter" in link.name.lower())]
        for link in preferred or links:
            if rank not in link.members:
                continue
            if link.semantics == "per_pair":
                peer = next((member for member in link.members
                             if member != rank), None)
                if peer is None:
                    continue
                return self.topology.resource(link, rank, peer)
            if link.semantics in {"per_port", "fully_connected", "mesh"}:
                return f"{self.topology.namespace}link:{link.name}:tx:port{rank}"
        return None

    def native_transport(self, op, replica):
        """Whole actual EP membership and source cells for this launch."""
        metadata = dict(op.metadata)
        source_stage = int(metadata.get("source_stage", op.stage))
        local_ranks = tuple(self.topology.ranks(replica, source_stage))
        groups = tuple(self.topology.groups("ep", replica, source_stage))
        group = next(
            (members for members in groups
             if set(members) & set(local_ranks)), ())
        if not group:
            group = local_ranks
        sources = tuple(index for index, rank in enumerate(group)
                        if rank in local_ranks)
        destination_memory = {
            int(rank): self.destination_window_resource(int(rank), source_stage)
            for rank in group}
        return {"group": tuple(group), "sources": sources,
                "route": self.topology.route_message,
                "destination_memory": destination_memory}

    def native_resource_names(self, context, program, *, rank=None,
                              replica=None):
        """Public launch-resource binding for canonical cohort drivers."""
        return self._native_resource_names(
            context, program, rank=rank, replica=replica)

    def _native_resource_names(self, context, program, *, rank=None,
                               replica=None):
        """Bind native pipe names to this rank's physical stage resources."""
        inner = getattr(program, "program", program)
        native_names = getattr(inner, "names", {}) or {}
        available = dict(getattr(context, "resource_names", {}) or {})
        bound = {str(role): str(physical)
                 for role, physical in available.items()}
        for pipe, native_name in native_names.items():
            physical = available.get(pipe) or available.get(native_name)
            if rank is not None and pipe in {"link", "link_hccs", "link_roce"}:
                route = self._topology_class_resource(rank, str(pipe))
                if route is not None:
                    physical = route
            if physical is not None:
                bound[str(pipe)] = str(physical)
        if rank is not None:
            for pipe in ("link", "link_hccs", "link_roce"):
                if pipe in bound:
                    continue
                route = self._topology_class_resource(rank, pipe)
                if route is not None:
                    bound[pipe] = route
        return bound

    @staticmethod
    def _build_launch_view(inner, rank_index, *, node_ranks=None,
                           peer_classes=None, peer_weights=None,
                           intra_latency_ms=None, inter_latency_ms=None,
                           resource_names=None):
        """Immutable per-launch rank/topology view; cached program untouched."""
        from mindie_sim.prediction.operators.mc2.mc2_program import Mc2LaunchView
        peer_count = int(getattr(inner, "peer_count", 0) or 0)
        if peer_count <= 0 or rank_index is None:
            return None
        rank = int(rank_index)
        if not 0 <= rank < peer_count:
            raise ModelingError(f"native_launch_rank_out_of_range:{rank}")
        source = getattr(inner, "view", None)
        labels = tuple(item for item in range(peer_count) if item != rank)
        classes = tuple(peer_classes or ())
        weights = tuple(peer_weights or ())
        if source is not None and source.rank == rank:
            if not classes:
                classes = tuple(source.peer_classes)
            if not weights:
                weights = tuple(source.peer_weights)
        node = int(node_ranks if node_ranks is not None else
                   (getattr(source, "node_ranks", 0) if source else 0))
        intra = (getattr(source, "intra_latency_ms", 0)
                 if intra_latency_ms is None and source
                 else (intra_latency_ms or 0))
        inter = (getattr(source, "inter_latency_ms", 0)
                 if inter_latency_ms is None and source
                 else (inter_latency_ms or 0))
        return Mc2LaunchView(
            rank=rank, peer_labels=labels, peer_classes=classes,
            peer_weights=weights, node_ranks=node,
            intra_latency_ms=intra, inter_latency_ms=inter,
            resource_names=tuple(sorted(
                (str(name), str(value))
                for name, value in dict(resource_names or {}).items())))

    @staticmethod
    def _phase_scope(program, mapping):
        """Normalise a flat peer map to MOE10's named-phase mapping."""
        if mapping is None:
            return None
        inner = getattr(program, "program", program)
        phases = getattr(inner, "phases", None)
        link_by_phase = getattr(inner, "link_by_phase", None)
        if not phases or not link_by_phase:
            return dict(mapping)
        names = {phase.name for phase in phases}
        if mapping and all(isinstance(key, str) and key in names
                           for key in mapping):
            return dict(mapping)
        peers = dict(mapping)
        return {phase.name: dict(peers) for phase in phases
                if phase.name in link_by_phase}

    def start_native_participant_with_handles(self, program, name, ready, *,
                                              owner="", local_ready=None,
                                              peer_ready=None, peer_send=None,
                                              rank_index=None, launch=None,
                                              source_send=None,
                                              send_placeholders=None,
                                              resource_names=None,
                                              transport=None):
        """Start a native peer program and return its actual phase handles."""
        from ..program import NativeProgram
        phase_ready = self._phase_scope(program, peer_ready)
        phase_send = self._phase_scope(program, peer_send)
        start = local_ready if local_ready is not None else ready
        if isinstance(program, NativeProgram):
            inner = program.program
            if launch is None and rank_index is not None:
                launch = self._build_launch_view(
                    inner, rank_index, resource_names=resource_names)
            if hasattr(inner, "start_with_handles"):
                return program.start_with_handles(
                    self.engine, name, start, peer_ready=phase_ready,
                    peer_send=phase_send, launch=launch,
                    source_send=source_send,
                    send_placeholders=send_placeholders,
                    transport=transport)
            if peer_send is not None:
                raise ModelingError(
                    "native_peer_send_requires_start_with_handles")
            return program.start(self.engine, name, start,
                                 peer_ready=phase_ready), {}
        event = self._start_engine_program(
            program, self.engine, name, ready, owner,
            local_ready=local_ready, peer_ready=peer_ready,
            peer_send=peer_send)
        return event, {}

    def start_native_participant(self, program, name, ready, *, owner="",
                                 local_ready=None, peer_ready=None,
                                 peer_send=None, rank_index=None,
                                 launch=None, source_send=None,
                                 send_placeholders=None, transport=None):
        """Start a lowered native program on this Execution's shared Engine."""
        event, _handles = self.start_native_participant_with_handles(
            program, name, ready, owner=owner, local_ready=local_ready,
            peer_ready=peer_ready, peer_send=peer_send,
            rank_index=rank_index, launch=launch, source_send=source_send,
            send_placeholders=send_placeholders, transport=transport)
        return event

    @staticmethod
    def _start_engine_program(program, engine, name, ready, owner,
                              local_ready=None, peer_ready=None,
                              peer_send=None, rank_index=None, launch=None,
                              source_send=None, send_placeholders=None,
                              resource_names=None, transport=None):
        """Start one lowered program on the caller's shared Engine."""
        from ..program import NativeProgram
        if isinstance(program, NativeProgram) and (
                peer_ready is not None or peer_send is not None):
            start = local_ready if local_ready is not None else ready
            inner = program.program
            if launch is None and rank_index is not None:
                launch = _CohortNativeMixin._build_launch_view(
                    inner, rank_index, resource_names=resource_names)
            phase_ready = _CohortNativeMixin._phase_scope(program, peer_ready)
            phase_send = _CohortNativeMixin._phase_scope(program, peer_send)
            if peer_send is not None:
                if not hasattr(program.program, "start_with_handles"):
                    raise ModelingError(
                        "native_peer_send_requires_start_with_handles")
                event, _handles = program.start_with_handles(
                    engine, name, start, peer_ready=phase_ready,
                    peer_send=phase_send, launch=launch,
                    source_send=source_send,
                    send_placeholders=send_placeholders,
                    transport=transport)
                return event
            return program.start(engine, name, start, peer_ready=phase_ready,
                                 launch=launch, source_send=source_send,
                                 transport=transport)
        start = ready
        if local_ready is not None and local_ready is not ready:
            start = engine.submit(name + ":cohort_gate", after=(ready,))
        if isinstance(program, NativeProgram):
            return program.start(engine, name, start, transport=transport)
        return program.start(engine, name, start, owner=owner)

    def _record_single_native(self, context, program, replica, name,
                              transport=None):
        """Retain an actual non-cohort native launch as public evidence."""
        from ..program import NativeProgram
        if not isinstance(program, NativeProgram):
            return
        inner = program.program
        if not hasattr(inner, "start_with_handles"):
            return
        key = context.op.op_id
        if key in self.native_group_launches:
            return
        meta = dict(context.op.metadata)
        raw = meta.get("mc2_phase", {})
        source_stage = int(meta.get("source_stage", context.op.stage))
        rank = (self.participant_rank_index(replica, source_stage)
                if replica is not None else None)
        if rank is None:
            rank = 0
        frames_for = getattr(inner, "frames_for", None)
        frames = (tuple(frames_for(rank)) if callable(frames_for) else
                  tuple(getattr(getattr(inner, "view", None),
                                "payload_frames", ()) or ()))
        basis_for = getattr(inner, "payload_basis_for", None)
        basis = (basis_for(rank) if callable(basis_for) else
                 str(getattr(getattr(inner, "view", None),
                             "payload_basis", "unknown")))
        packet = int(getattr(getattr(inner, "view", None),
                            "payload_packet_bytes", 0) or 0)
        links_for = getattr(inner, "link_by_phase_for", None)
        links = (links_for(rank) if callable(links_for)
                 else dict(getattr(inner, "link_by_phase", {}) or {}))
        work_for = getattr(inner, "work_for_rank", None)
        work = dict(work_for(rank) if callable(work_for)
                    else tuple(getattr(program, "work", ()) or ()))
        transport_details = {}
        transport_work = {}
        if transport is not None and hasattr(inner, "transport_account"):
            payload_account, count_account = inner.transport_account(transport)
            has_count = any(getattr(item, "name", "") == "count_status_exchange"
                            for item in getattr(inner, "phases", ()))
            for phase_name, account in (("payload_exchange", payload_account),
                                        ("count_status_exchange", count_account)):
                if phase_name == "count_status_exchange" and not has_count:
                    continue
                if not account.get("edges"):
                    continue
                transport_details[phase_name] = {
                    "frames": int(account.get("frames", 0)),
                    "edges": int(account.get("edges", 0)),
                    "tx_bytes": float(account.get("tx_bytes", 0.0)),
                    "rx_bytes": float(account.get("rx_bytes", 0.0)),
                    "basis": str(getattr(inner, "transport_basis", basis)),
                    "group": list(account.get("group") or ()),
                    "sources": list(account.get("sources") or ()),
                    "count": phase_name == "count_status_exchange",
                }
                for pipe, amount in account.get("work", {}).items():
                    transport_work[pipe] = (
                        transport_work.get(pipe, 0.0) + float(amount))
        if transport_work:
            work = {pipe: amount for pipe, amount in work.items()
                    if "link" not in str(pipe).lower()}
            for pipe, amount in transport_work.items():
                work[pipe] = work.get(pipe, 0.0) + amount
        total_edges = sum(item["edges"] for item in transport_details.values())
        transport_frames = sum(item["frames"] for item in transport_details.values())
        payload_edges = int(transport_details.get(
            "payload_exchange", {}).get("edges", 0))
        count_edges = int(transport_details.get(
            "count_status_exchange", {}).get("edges", 0))
        self.native_group_launches[key] = {
            "op_id": key, "kind": str(raw.get("kind", "")),
            "launch_mode": "single_program",
            "ranks": (rank,),
            "payload_basis": {name: str(basis)},
            "payload_packet_bytes": {name: packet},
            "payload_frames": {name: dict(frames)},
            "link_by_phase_by_rank": {
                name: {phase: {str(pipe): float(amount)
                               for pipe, amount in values.items()}
                       for phase, values in links.items()}},
            "work_by_rank": {name: tuple(sorted(work.items()))},
            "transport_by_rank": {name: transport_details},
            "transport_work_by_rank": {name: transport_work},
            "transport_tx_bytes": sum(
                item["tx_bytes"] for item in transport_details.values()),
            "transport_rx_bytes": sum(
                item["rx_bytes"] for item in transport_details.values()),
            "transport_frames": transport_frames,
            "transport_edges": payload_edges,
            "total_edges": total_edges,
            "payload_edges": payload_edges,
            "count_edges": count_edges,
            "directed_edges": (transport_details.get(
                "payload_exchange", {}).get("edges") or len(frames)),
            "directed_frames": (transport_details.get(
                "payload_exchange", {}).get("frames")
                or sum(int(count) for _peer, count in frames)),
            "protocol_edges": 0,
            "unresolved_edges": 0,
            "partial": (),
        }

    def native_group_summary(self) -> list:
        """Bounded actual launch evidence from the shared native cohorts."""
        groups = []
        for key in sorted(self.native_group_launches):
            item = self.native_group_launches[key]
            groups.append({
                "op_id": item["op_id"], "kind": item["kind"],
                "launch_mode": item.get("launch_mode", ""),
                "ranks": list(item["ranks"]),
                "payload_basis": dict(sorted(item["payload_basis"].items())),
                "payload_packet_bytes": dict(sorted(
                    item["payload_packet_bytes"].items())),
                "payload_frames": {
                    name: dict(sorted(frames.items()))
                    for name, frames in sorted(item["payload_frames"].items())},
                "link_by_phase_by_rank": {
                    name: {phase: {pipe: float(amount)
                                   for pipe, amount in values.items()}
                           for phase, values in sorted(phases.items())}
                    for name, phases in sorted(
                        item["link_by_phase_by_rank"].items())},
                "work_by_rank": {
                    name: [[str(pipe), float(amount)] for pipe, amount in work]
                    for name, work in sorted(item["work_by_rank"].items())},
                "transport_by_rank": {
                    name: {phase: dict(details)
                           for phase, details in sorted(phases.items())}
                    for name, phases in sorted(
                        item.get("transport_by_rank", {}).items())},
                "transport_work_by_rank": {
                    name: [[str(pipe), float(amount)]
                           for pipe, amount in sorted(work.items())]
                    for name, work in sorted(
                        item.get("transport_work_by_rank", {}).items())},
                "transport_tx_bytes": float(item.get("transport_tx_bytes", 0.0)),
                "transport_rx_bytes": float(item.get("transport_rx_bytes", 0.0)),
                "transport_frames": int(item.get("transport_frames", 0)),
                "transport_edges": int(item.get("transport_edges", 0)),
                "total_edges": int(item.get("total_edges", 0)),
                "payload_edges": int(item.get("payload_edges", 0)),
                "count_edges": int(item.get("count_edges", 0)),
                "directed_edges": item["directed_edges"],
                "directed_frames": item["directed_frames"],
                "protocol_edges": item["protocol_edges"],
                "unresolved_edges": item["unresolved_edges"],
                "partial": list(item["partial"]),
            })
            if len(groups) >= NATIVE_GROUP_SUMMARY_LIMIT:
                break
        return groups
