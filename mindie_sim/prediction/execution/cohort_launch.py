"""Two-phase native peer cohort launcher and rank-local Engine view.

[Role]
  Launch multi-rank native peer programs (e.g., MC2 / dispatch-combine cohorts)
  onto one shared ``Engine`` using prebound directed ``(src, dst, phase)``
  placeholder futures and shallow launch-local resource views.

[Invariants]
  * Cached ``NativeProgram`` objects are never mutated; each launch receives
    its own shallow resource view and ``Mc2LaunchView``.
  * Directed peer futures are resolved strictly by the sender's actual Engine
    send event; receivers never consume global phase barriers when directed
    edges exist.

[Timeouts & Locks]
  Bounded per-cohort future registries cleared upon final deferred start; no I/O.

[Side-Effect Policy]
  Submits tasks and resolves futures only on the caller-supplied Engine.
"""
from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
from copy import copy

from mindie_sim.schema.common import ModelingError


class _RankEngine:
    """Name-mapping view over the one shared Engine."""

    def __init__(self, base, resource_names):
        self.base = base
        self.resource_names = {str(key): str(value)
                               for key, value in dict(resource_names).items()}

    def _work(self, work):
        return {self.resource_names.get(str(key), str(key)): value
                for key, value in dict(work or {}).items()}

    def _keys(self, values):
        return {self.resource_names.get(str(key), str(key)): value
                for key, value in dict(values or {}).items()}

    def submit(self, name, work=None, **kwargs):
        return self.base.submit(
            name, self._work(work) if work is not None else None, **kwargs)

    def _prepare(self, name, work=None, *, latency=0.0, caps=None,
                 acquire=(), free=(), placement=None, coupled=False,
                 priority=0):
        return self.base._prepare(
            name, self._work(work) if work is not None else None,
            latency=latency, caps=self._keys(caps), acquire=acquire,
            free=free, placement=self._keys(placement), coupled=coupled,
            priority=priority)

    def _submit_prepared(self, prepared, name, **kwargs):
        return self.base._submit_prepared(prepared, name, **kwargs)

    def future(self, name):
        return self.base.future(name)

    def resolve(self, event):
        return self.base.resolve(event)

    def __getattr__(self, name):
        return getattr(self.base, name)


def _bind_native_resource_view(program, resource_names, engine):
    """Shallow launch-local resource view; cached program is not mutated."""
    if not resource_names:
        return program
    from ..program import NativeProgram
    inner = copy(program.program)
    names = dict(getattr(inner, "names", {}) or {})
    rates = dict(getattr(inner, "rates", {}) or {})
    units = dict(getattr(inner, "units", {}) or {})
    for pipe, physical in dict(resource_names).items():
        names[str(pipe)] = str(physical)
        resource = engine.resources.get(str(physical))
        if resource is not None:
            rates[str(pipe)] = resource.rate
            rates[str(physical)] = resource.rate
            units[str(pipe)] = resource.units
            units[str(physical)] = resource.units
    inner.names = names
    inner.rates = rates
    inner.units = units
    return NativeProgram(inner, program.work)


def launch_native_cohort(engine, launches, *, name_prefix="native_cohort",
                         owner="native_cohort", defer=False):
    """Two-phase launch of native peer programs on one shared Engine."""
    from mindie_sim.prediction.operators.mc2.mc2_program import Mc2LaunchView
    from ..program import NativeProgram

    def field(raw, key, default=None):
        return (raw.get(key, default) if isinstance(raw, _AbcMapping)
                else getattr(raw, key, default))

    infos, by_rank, names = [], {}, set()
    phases = None
    for raw in launches:
        program = field(raw, "program")
        if not isinstance(program, NativeProgram):
            raise ModelingError("native_cohort_requires_native_program")
        name = str(field(raw, "name") or "")
        if not name or name in names:
            raise ModelingError(f"native_cohort_launch_name_invalid:{name}")
        names.add(name)
        rank = field(raw, "rank_index")
        if rank is None:
            raise ModelingError(f"native_cohort_rank_missing:{name}")
        rank = int(rank)
        if rank in by_rank:
            raise ModelingError(f"native_cohort_rank_duplicate:{rank}")
        inner = program.program
        source_view = getattr(inner, "view", None)
        peer_count = int(field(raw, "peer_count")
                         or getattr(inner, "peer_count", 0) or 0)
        if peer_count <= 0:
            raise ModelingError(f"native_cohort_peer_count_missing:{name}")
        if not 0 <= rank < peer_count:
            raise ModelingError(f"native_cohort_rank_out_of_range:{name}:{rank}")
        labels = field(raw, "peer_labels")
        labels = (tuple(int(item) for item in labels)
                  if labels is not None else
                  tuple(item for item in range(peer_count) if item != rank))
        if len(set(labels)) != len(labels) or (
                rank in labels
                or any(not 0 <= item < peer_count for item in labels)):
            raise ModelingError(f"native_cohort_peer_labels_invalid:{name}")
        node_ranks = field(raw, "node_ranks")
        node = int(node_ranks if node_ranks is not None else
                   (getattr(source_view, "node_ranks", 0)
                    if source_view is not None else 0))
        peer_classes = field(raw, "peer_classes")
        classes = tuple(peer_classes or ())
        if not classes and source_view is not None and source_view.rank == rank:
            classes = tuple(source_view.peer_classes)
        peer_weights = field(raw, "peer_weights")
        weights = tuple(peer_weights or ())
        if not weights and source_view is not None and source_view.rank == rank:
            weights = tuple(source_view.peer_weights)
        intra = field(raw, "intra_latency_ms")
        inter = field(raw, "inter_latency_ms")
        resource_names = field(raw, "resource_names")
        frames_raw = field(raw, "payload_frames", None)
        if frames_raw is None:
            frames_for = getattr(inner, "frames_for", None)
            if callable(frames_for):
                frames_raw = frames_for(rank)
        if frames_raw is None and source_view is not None:
            frames_raw = getattr(source_view, "payload_frames", None)
        if isinstance(frames_raw, _AbcMapping):
            payload_frames = tuple(sorted(
                (int(peer), int(frames)) for peer, frames in frames_raw.items()
                if int(frames) > 0))
        elif frames_raw:
            payload_frames = tuple(sorted(
                (int(item[0]), int(item[1])) for item in frames_raw
                if int(item[1]) > 0))
        else:
            payload_frames = ()
        packet = field(raw, "payload_packet_bytes", None)
        if packet is None and source_view is not None:
            packet = getattr(source_view, "payload_packet_bytes", 0)
        payload_packet = int(packet or 0)
        basis_raw = field(raw, "payload_basis", None)
        if basis_raw is None:
            basis_for = getattr(inner, "payload_basis_for", None)
            if callable(basis_for):
                basis_raw = basis_for(rank)
        if basis_raw is None:
            basis_raw = (getattr(source_view, "payload_basis", "unknown")
                         if source_view is not None else "unknown")
        payload_basis = str(basis_raw)
        local_frames = field(raw, "payload_local_frames", None)
        if local_frames is None:
            local_for = getattr(inner, "local_frames_for", None)
            if callable(local_for):
                local_frames = local_for(rank)
        if local_frames is None and source_view is not None:
            local_frames = getattr(source_view, "payload_local_frames", 0)
        payload_local = int(local_frames or 0)
        view = Mc2LaunchView(
            rank=rank, peer_labels=labels, peer_classes=classes,
            peer_weights=weights, node_ranks=node,
            intra_latency_ms=(getattr(source_view, "intra_latency_ms", 0)
                              if intra is None and source_view is not None
                              else (intra or 0)),
            inter_latency_ms=(getattr(source_view, "inter_latency_ms", 0)
                              if inter is None and source_view is not None
                              else (inter or 0)),
            resource_names=tuple(sorted(
                (str(name), str(value))
                for name, value in dict(
                    resource_names
                    or (source_view.resource_names if source_view is not None
                        else ())).items())),
            payload_frames=payload_frames,
            payload_packet_bytes=payload_packet,
            payload_basis=payload_basis,
            payload_local_frames=payload_local)
        launch_phases = tuple(getattr(inner, "phases", ()))
        phase_names = tuple(getattr(phase, "name", str(phase))
                            for phase in launch_phases)
        if phases is None:
            phases = phase_names
        elif set(phases) != set(phase_names):
            raise ModelingError(
                f"native_cohort_phase_mismatch:{name}:"
                f"{sorted(set(phases) ^ set(phase_names))}")
        infos.append({
            "name": name, "rank": rank, "view": view,
            "program": program, "ready": field(raw, "ready"),
            "operator_id": str(field(raw, "operator_id", "mc2")),
            "resource_names": dict(resource_names or {}),
            "payload_frames": dict(payload_frames),
            "payload_packet_bytes": payload_packet,
            "payload_basis": payload_basis,
            "payload_local_frames": payload_local,
            "transport": field(raw, "transport", None),
        })
        by_rank[rank] = infos[-1]
    if not infos:
        raise ModelingError("native_cohort_empty")
    link_phases = tuple(
        phase for phase in (phases or ())
        if any(getattr(info["program"].program, "link_by_phase", {}).get(phase)
               for info in infos))
    for info in infos:
        info["program"] = _bind_native_resource_view(
            info["program"], info["resource_names"], engine)
    programs_by_rank = {info["rank"]: info["program"] for info in infos}
    names_by_rank = {info["rank"]: info["name"] for info in infos}
    unresolved, partial = [], []
    placeholders = {}
    for info in infos:
        transport = info.get("transport")
        inner = info["program"].program
        if transport is None or not hasattr(inner, "transport_cells"):
            continue
        for phase_name, count_phase in (
                ("payload_exchange", False), ("count_status_exchange", True)):
            if phase_name not in (phases or ()):
                continue
            for _si, di, _src, _dst, useful, _frames in inner.transport_cells(
                    transport, count=count_phase):
                if useful <= 0 or _si == di:
                    continue
                placeholders.setdefault(
                    (_si, di, phase_name),
                    engine.future(f"{name_prefix}:{phase_name}:{_si}->{di}"))
    protocol_edges = 0
    directed_edges = 0
    directed_frames = 0
    fullmesh_by_rank = {}
    payload_packet_by_rank = {}
    for info in infos:
        source_inner = info["program"].program
        view = info["view"]
        rank = int(info["rank"])
        source_phases = tuple(
            (str(phase), links) for phase, links
            in getattr(source_inner, "link_by_phase", {}).items() if links)
        fullmesh = tuple(sorted(
            phase for phase, _links in source_phases
            if phase != "payload_exchange"))
        fullmesh_by_rank[info["name"]] = fullmesh
        for phase, _links in source_phases:
            if phase != "payload_exchange":
                for peer in view.peer_labels:
                    if peer in programs_by_rank:
                        key = (rank, peer, phase)
                        if key not in placeholders:
                            placeholders[key] = engine.future(
                                f"{name_prefix}:{phase}:{rank}->{peer}")
                        protocol_edges += 1
                    else:
                        edge = (phase, rank, peer)
                        if edge not in unresolved:
                            unresolved.append(edge)
                continue
            frames = dict(info.get("payload_frames") or {})
            basis = str(info.get("payload_basis") or "unknown")
            packet = int(info.get("payload_packet_bytes") or 0)
            payload_packet_by_rank[info["name"]] = packet
            if frames:
                if packet <= 0:
                    partial.append({
                        "rank": rank, "name": info["name"],
                        "phase": "payload_exchange", "basis": basis,
                        "reason": "payload_packet_bytes_missing"})
                    continue
                for peer, whole_frames in frames.items():
                    peer = int(peer)
                    whole_frames = int(whole_frames)
                    if whole_frames <= 0:
                        continue
                    if peer not in view.peer_labels:
                        partial.append({
                            "rank": rank, "name": info["name"],
                            "phase": "payload_exchange", "peer": peer,
                            "basis": basis,
                            "reason": "destination_not_declared"})
                        continue
                    if peer in programs_by_rank:
                        key = (rank, peer, "payload_exchange")
                        if key not in placeholders:
                            placeholders[key] = engine.future(
                                f"{name_prefix}:payload_exchange:"
                                f"{rank}->{peer}")
                        directed_edges += 1
                        directed_frames += whole_frames
                    else:
                        edge = ("payload_exchange", rank, peer)
                        if edge not in unresolved:
                            unresolved.append(edge)
                continue
            if "zero" in basis.lower():
                continue
            partial.append({
                "rank": rank, "name": info["name"],
                "phase": "payload_exchange", "basis": basis,
                "reason": "payload_destinations_unknown"})
        if not any(phase == "payload_exchange"
                   for phase, _links in source_phases):
            frames = dict(info.get("payload_frames") or {})
            basis = str(info.get("payload_basis") or "unknown")
            if not frames and "zero" not in basis.lower():
                partial.append({
                    "rank": rank, "name": info["name"],
                    "phase": "payload_exchange", "basis": basis,
                    "reason": "payload_destinations_unknown"})

    def _declared_metrics(info):
        inner = info["program"].program
        transport = info.get("transport") or {}
        rank = int(info["rank"])
        payload, count = (inner.transport_account(transport)
                          if hasattr(inner, "transport_account") else ({}, {}))
        phases_set = {phase.name for phase in getattr(inner, "phases", ())}
        records = {}
        if payload:
            records["payload_exchange"] = {
                **payload,
                "basis": (inner.payload_basis_for(rank)
                          if hasattr(inner, "payload_basis_for")
                          else info["payload_basis"]),
                "count": False,
            }
        if count and "count_status_exchange" in phases_set:
            count_basis = (inner.payload_basis_for(rank)
                           if hasattr(inner, "payload_basis_for")
                           else info["payload_basis"])
            records["count_status_exchange"] = {
                **count,
                "basis": count_basis,
                "count": True,
            }
        links_for_rank = getattr(inner, "link_by_phase_for", None)
        links = (dict(links_for_rank(rank)) if callable(links_for_rank)
                 else dict(getattr(inner, "link_by_phase", {}) or {}))
        work_for_rank = getattr(inner, "work_for_rank", None)
        work = (tuple(work_for_rank(rank)) if callable(work_for_rank)
                else tuple(getattr(info["program"], "work", ()) or ()))
        transport_work = dict(payload.get("work", {}))
        if "count_status_exchange" in phases_set:
            for pipe, amount in (count.get("work", {}) or {}).items():
                transport_work[pipe] = transport_work.get(pipe, 0.0) + float(amount)
        return records, links, work, transport_work

    cohort = {}
    if defer:
        launch_by_rank = {info["rank"]: info for info in infos}
        parents, starts = {}, {}
        remaining_starts = [len(infos)]
        for info in infos:
            name = info["name"]
            parent = engine.future(name + ":program_complete")
            parents[name] = parent
            starts[name] = info["ready"]
            records, links, work, transport_work = _declared_metrics(info)
            cohort[info["rank"]] = {
                "parent": parent, "start": info["ready"],
                "operators": {}, "phases": {},
                "handles": {"deferred": True},
                "work": work,
                "link_by_phase": {
                    phase: {str(pipe): float(amount)
                            for pipe, amount in values.items()}
                    for phase, values in links.items()},
                "transport": records,
                "transport_work": transport_work,
            }

            def begin(_event, item=info, promise=parent):
                rank_engine = _RankEngine(engine, item["resource_names"])
                finish, _handles = item["program"].start_with_handles(
                    rank_engine, item["name"], item["ready"],
                    launch=item["view"], send_placeholders=placeholders,
                    transport=item.get("transport"))
                engine.submit(item["name"] + ":parent_finish", after=(finish,),
                              owner=owner,
                              on_complete=lambda _e: engine.resolve(promise))
                remaining_starts[0] -= 1
                if remaining_starts[0] == 0:
                    placeholders.clear()

            engine.submit(name + ":begin", after=(info["ready"],), owner=owner,
                          on_complete=begin)
        parents = {names_by_rank[rank]: item["parent"]
                   for rank, item in cohort.items()}
        starts = {names_by_rank[rank]: item["start"]
                  for rank, item in cohort.items()}
        operators = {names_by_rank[rank]: item["operators"]
                     for rank, item in cohort.items()}
        handles = {names_by_rank[rank]: item["handles"]
                   for rank, item in cohort.items()}
        phases_by_rank = {names_by_rank[rank]: item["phases"]
                          for rank, item in cohort.items()}
        work_by_rank = {names_by_rank[rank]: item["work"]
                        for rank, item in cohort.items()}
        link_by_phase_by_rank = {
            names_by_rank[rank]: item["link_by_phase"]
            for rank, item in cohort.items()}
        transport_by_rank = {
            names_by_rank[rank]: item["transport"]
            for rank, item in cohort.items()}
        transport_work_by_rank = {
            names_by_rank[rank]: item["transport_work"]
            for rank, item in cohort.items()}
        transport_tx = sum(
            float(phase.get("tx_bytes", 0.0))
            for details in transport_by_rank.values()
            for phase in details.values())
        transport_rx = sum(
            float(phase.get("rx_bytes", 0.0))
            for details in transport_by_rank.values()
            for phase in details.values())
        transport_frames = sum(
            int(phase.get("frames", 0))
            for details in transport_by_rank.values()
            for phase in details.values())
        transport_edges = sum(
            int(phase.get("edges", 0))
            for details in transport_by_rank.values()
            for phase in details.values())
        join = engine.submit(f"{name_prefix}:join", after=tuple(parents.values()),
                             owner=owner)
        return {
            "schema": "mindie-sim/native-cohort/3",
            "finishes": parents, "parents": parents, "join": join,
            "starts": starts, "operators": operators, "handles": handles,
            "phases_by_rank": phases_by_rank, "work_by_rank": work_by_rank,
            "link_by_phase_by_rank": link_by_phase_by_rank,
            "transport_by_rank": transport_by_rank,
            "transport_work_by_rank": transport_work_by_rank,
            "transport_tx_bytes": transport_tx,
            "transport_rx_bytes": transport_rx,
            "transport_frames": transport_frames,
            "transport_edges": transport_edges,
            "directed_edges": sum(
                int(value.get("edges", 0))
                for details in transport_by_rank.values()
                for phase, value in details.items()
                if isinstance(value, _AbcMapping) and not value.get("count")),
            "directed_frames": transport_frames,
            "rank_indices": {names_by_rank[rank]: rank for rank in cohort},
            "payload_basis": {
                names_by_rank[rank]: str(
                    getattr(launch_by_rank[rank]["program"].program,
                            "payload_basis_for",
                            lambda _r, info=launch_by_rank[rank]:
                            info["payload_basis"])(rank))
                for rank in cohort},
            "payload_packet_bytes": {
                names_by_rank[rank]: int(
                    getattr(launch_by_rank[rank]["program"].program, "view", None).payload_packet_bytes
                    if getattr(launch_by_rank[rank]["program"].program, "view", None) is not None else 0)
                for rank in cohort},
            "payload_frames": {
                names_by_rank[rank]: dict(
                    getattr(launch_by_rank[rank]["program"].program,
                            "frames_for", lambda _r: ())(rank))
                for rank in cohort},
            "protocol_edges": sum(
                1 for key in placeholders
                if key[2] not in {"payload_exchange"}),
            "unresolved_edges": len(unresolved),
            "partial": tuple(partial),
        }
    for info in infos:
        rank_engine = _RankEngine(engine, info["resource_names"])
        parent, launch_handles = info["program"].start_with_handles(
            rank_engine, info["name"], info["ready"],
            launch=info["view"], send_placeholders=placeholders,
            transport=info.get("transport"))
        op_id = info.get("operator_id", "mc2")
        operators = dict(launch_handles.get("operators") or {})
        if op_id != "mc2" and "mc2" in operators:
            operators = {op_id: operators["mc2"]}
        inner = getattr(info["program"], "program", None)
        work_for_rank = getattr(inner, "work_for_rank", None)
        work = (tuple(work_for_rank(info["rank"]))
                if callable(work_for_rank)
                else tuple(getattr(info["program"], "work", ()) or ()))
        links_for_rank = getattr(inner, "link_by_phase_for", None)
        links = (dict(links_for_rank(info["rank"]))
                 if callable(links_for_rank)
                 else dict(getattr(inner, "link_by_phase", {}) or {}))
        cohort[info["rank"]] = {
            "parent": parent,
            "start": launch_handles.get("start", info["ready"]),
            "operators": operators,
            "phases": dict(launch_handles.get("phase_complete") or {}),
            "handles": launch_handles,
            "work": work,
            "link_by_phase": {
                phase: {str(pipe): float(amount) for pipe, amount in values.items()}
                for phase, values in links.items()},
            "transport": dict(launch_handles.get("transport") or {}),
            "transport_work": dict(launch_handles.get("transport_work") or {}),
        }
    parents = {names_by_rank[rank]: item["parent"]
               for rank, item in cohort.items()}
    starts = {names_by_rank[rank]: item["start"]
              for rank, item in cohort.items()}
    operators = {names_by_rank[rank]: item["operators"]
                 for rank, item in cohort.items()}
    handles = {names_by_rank[rank]: item["handles"]
               for rank, item in cohort.items()}
    phases_by_rank = {names_by_rank[rank]: item["phases"]
                      for rank, item in cohort.items()}
    work_by_rank = {names_by_rank[rank]: item["work"]
                    for rank, item in cohort.items()}
    link_by_phase_by_rank = {
        names_by_rank[rank]: item["link_by_phase"]
        for rank, item in cohort.items()}
    transport_by_rank = {
        names_by_rank[rank]: item["transport"]
        for rank, item in cohort.items()}
    transport_work_by_rank = {
        names_by_rank[rank]: item["transport_work"]
        for rank, item in cohort.items()}
    transport_tx = sum(
        float(phase.get("tx_bytes", 0.0))
        for details in transport_by_rank.values()
        for phase in details.values())
    transport_rx = sum(
        float(phase.get("rx_bytes", 0.0))
        for details in transport_by_rank.values()
        for phase in details.values())
    transport_frames = sum(
        int(phase.get("frames", 0))
        for details in transport_by_rank.values()
        for phase in details.values())
    transport_edges = sum(
        int(phase.get("edges", 0))
        for details in transport_by_rank.values()
        for phase in details.values())
    join = engine.submit(f"{name_prefix}:join", after=tuple(parents.values()),
                         owner=owner)
    return {
        "schema": "mindie-sim/native-cohort/3",
        "finishes": parents, "parents": parents, "join": join,
        "starts": starts, "operators": operators, "handles": handles,
        "phases_by_rank": phases_by_rank, "work_by_rank": work_by_rank,
        "link_by_phase_by_rank": link_by_phase_by_rank,
        "transport_by_rank": transport_by_rank,
        "transport_work_by_rank": transport_work_by_rank,
        "transport_tx_bytes": transport_tx,
        "transport_rx_bytes": transport_rx,
        "transport_frames": transport_frames,
        "transport_edges": transport_edges,
        "by_rank": {rank: dict(item) for rank, item in cohort.items()},
        "names_by_rank": dict(names_by_rank),
        "phases": tuple(phases or ()), "link_phases": link_phases,
        "fullmesh_phases": fullmesh_by_rank,
        "payload_frames": {
            info["name"]: dict(info.get("payload_frames") or {})
            for info in infos},
        "payload_basis": {
            info["name"]: str(info.get("payload_basis") or "unknown")
            for info in infos},
        "payload_packet_bytes": payload_packet_by_rank,
        "partial": tuple(partial),
        "protocol_edges": protocol_edges,
        "directed_edges": directed_edges,
        "directed_frames": directed_frames,
        "unresolved_peers": tuple(sorted({peer for _, _, peer in unresolved})),
        "unresolved_edges": len(unresolved),
        "rank_indices": {info["name"]: info["rank"] for info in infos},
    }
