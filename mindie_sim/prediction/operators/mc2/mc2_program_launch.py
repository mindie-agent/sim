"""Engine launch schedule for MC2 native peer programs.

[Role]
  Start the selected native phase programs on the caller's shared Engine,
  binding peer readiness, send placeholders, launch handles and transport
  routes without creating a private Engine.

[Invariants]
  * Every phase keeps its own KernelProgram, owner placement and readiness.
  * Directed transport reserves every hop once with one arrival Event.
  * Retry/status branches execute only the declared phase DAG.
  * No precomputed duration is consulted; completion is the real Event.

[Timeouts & Locks]
  Bounded phase/task counts; no locks, sleeps or background tasks.

[Side-Effect Policy]
  Submits virtual tasks only to the caller-supplied Engine.
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from typing import Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.program import KernelProgram
from .mc2_program import Mc2LaunchView, _peer_class_for
from .mc2_program_state import Mc2ProgramState

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


class _Mc2Program(Mc2ProgramState):
    """Execute the selected native phase programs on the common Engine.

    Each source phase keeps its own KernelProgram, active owner placement and
    primitive/readiness dependencies.  Link bytes and peer propagation are
    separate declared services after the phase they belong to.
    """

    def _schedule_transport(self, engine, name, phase, prerequisite, view,
                            transport, *, count=False, placeholders=None):
        """Route this launch's whole-EP directed cells once, coupled TX+RX.

        One transmission reserves every physical hop capacity in ``hop.demand``
        (source TX and destination RX together) and has one arrival Event.  The
        local rank's send handle is the same first-hop service; no second wire
        reservation or second Engine is created.
        """
        from mindie_sim.prediction.communication import MessageFlow
        route = transport.get("route")
        group = tuple(transport.get("group") or ())
        local_indices = {int(item) for item in
                         (transport.get("sources") or ())}
        if not local_indices and 0 <= int(view.rank) < len(group):
            local_indices = {int(view.rank)}
        work = {}
        peer_events = {}
        parent_events = []
        incoming_arrivals = []
        frames = 0
        edges = 0
        tx_bytes = 0.0
        rx_bytes = 0.0
        useful_bytes = 0.0
        endpoint_memory = {
            int(rank): str(name) for rank, name in
            (transport.get("destination_memory") or {}).items() if name}
        endpoint_resources = set(endpoint_memory.values())
        for _si, _di, source, destination, useful, cell_frames in self.transport_cells(
                transport, count=count):
            if not callable(route) or useful <= 0:
                continue
            message = route(MessageFlow(source, destination, float(useful)))
            previous = (prerequisite,)
            first_service = None
            endpoint = endpoint_memory.get(int(destination))
            for hop_index, hop in enumerate(message.hops):
                demand = {str(name): float(amount)
                          for name, amount in hop.demand.items() if amount}
                src_rank = group[_si] if (group and 0 <= _si < len(group)) else source
                dst_rank = group[_di] if (group and 0 <= _di < len(group)) else destination
                if (endpoint
                        and hop_index == len(message.hops) - 1
                        and source != destination
                        and src_rank != dst_rank):
                    demand[endpoint] = demand.get(endpoint, 0.0) + float(useful)
                service = engine.submit(
                    f"{name}:{phase.name}:{source}->{destination}:"
                    f"hop{hop_index}:service",
                    work=demand, after=previous, coupled=True)
                if first_service is None:
                    first_service = service
                for key, amount in demand.items():
                    work[key] = work.get(key, 0.0) + amount
                    if key in endpoint_resources:
                        continue
                    lowered = key.lower()
                    if ":tx:" in lowered:
                        tx_bytes += amount
                    elif ":rx:" in lowered:
                        rx_bytes += amount
                    else:
                        tx_bytes += amount
                        rx_bytes += amount
                arrival = engine.submit(
                    f"{name}:{phase.name}:{source}->{destination}:"
                    f"hop{hop_index}:arrival",
                    latency=float(hop.latency_ms), after=(service,))
                previous = (arrival,)
            arrival = previous[0]
            final = arrival
            promise = dict(placeholders or {}).get(
                (_si, _di, phase.name))
            if promise is not None:
                final = engine.submit(
                    f"{name}:{phase.name}:{source}->{destination}:retire",
                    after=(arrival,), native_hook=True,
                    on_complete=lambda _event, item=promise:
                    engine.resolve(item))
            parent_events.append(final)
            if _di in local_indices and _si != _di:
                incoming_arrivals.append(final)
            frames += int(cell_frames)
            edges += 1
            useful_bytes += float(useful)
            if not count and (_si in local_indices or source in local_indices):
                peer_events[destination] = {
                    "send": first_service, "arrival": arrival,
                    "receive": None, "check": None,
                    "retire": (final if final is not arrival else None),
                    "class": view.class_for(
                        view.peer_labels.index(destination)
                        if destination in view.peer_labels else 0,
                        destination),
                    "pipe": None, "latency_ms": float(message.hops[-1].latency_ms)
                    if message.hops else 0.0,
                    "frames": int(cell_frames), "bytes": float(useful),
                    "send_placeholder": None, "remote_send": None,
                }
        # A local receive/check/copy must wait the actual directed arrivals
        # from every remote source, not just this rank's outgoing sends.  The
        # cohort launcher creates these futures before any program starts and
        # each sender resolves its own cell; no timestamp or global barrier is
        # synthesized here.
        incoming = []
        for peer in range(len(group)):
            for destination in sorted(local_indices):
                if peer == destination:
                    continue
                promise = dict(placeholders or {}).get(
                    (peer, destination, phase.name))
                if promise is not None:
                    incoming.append(promise)
        complete = engine.submit(
            f"{name}:{phase.name}:peer_complete",
            after=tuple(dict.fromkeys(tuple(parent_events) + tuple(incoming)))
            or (prerequisite,))
        return {
            "event": complete, "peer_events": peer_events, "work": work,
            "incoming_arrivals": tuple(incoming_arrivals),
            "frames": frames, "edges": edges, "tx_bytes": tx_bytes,
            "rx_bytes": rx_bytes, "useful_bytes": useful_bytes,
            "group": group, "sources": tuple(transport.get("sources") or ()),
            "basis": self.transport_basis, "count": bool(count),
        }

    def start(self, engine, name, ready, records=None, peer_ready=None,
              peer_send=None, launch=None, source_send=None,
              send_placeholders=None, transport=None):
        event, _handles = self.start_with_handles(
            engine, name, ready, records=records, peer_ready=peer_ready,
            peer_send=peer_send, launch=launch, source_send=source_send,
            send_placeholders=send_placeholders, transport=transport)
        return event

    def _with_route(self, view):
        if not self.route_matrix:
            return view
        return replace(
            view, payload_frames=self.frames_for(view.rank),
            payload_basis=self.payload_basis_for(view.rank),
            payload_local_frames=self.local_frames_for(view.rank))

    def _launch_view(self, launch):
        if launch is None:
            return self._with_route(self.view)
        if not isinstance(launch, Mc2LaunchView):
            raise ModelingError("mc2_launch_view_required")
        if len(launch.peer_labels) != max(0, self.peer_count - 1):
            raise ModelingError("mc2_launch_view_peer_count_mismatch")
        if launch.peer_weights:
            if (len(launch.peer_weights) != len(launch.peer_labels)
                    or sum(launch.peer_weights, Fraction(0)) != 1):
                raise ModelingError("mc2_launch_view_peer_weights_invalid")
        elif launch.peer_labels:
            launch = replace(
                launch,
                peer_weights=tuple(Fraction(1, len(launch.peer_labels))
                                   for _ in launch.peer_labels))
        return self._with_route(launch)

    def start_with_handles(self, engine, name, ready, records=None,
                           peer_ready=None, peer_send=None, launch=None,
                           source_send=None, send_placeholders=None,
                           transport=None):
        """Launch once and return launch-local peer/port handles.

        ``launch`` is an immutable :class:`Mc2LaunchView` for this rank and
        topology.  ``peer_ready``/``peer_send`` are phase-scoped maps.  A
        ``source_send=(src, dst, phase) -> Event`` map is consumed only for
        the matching source rank, destination peer and phase.
        """
        view = self._launch_view(launch)
        transport = dict(transport or {})
        events = {}
        handles = {"peer_events": {}, "phase_complete": {},
                   "peer_class": {}, "ports": {}, "launch": view,
                   "transport": {}, "transport_work": {},
                   "resource_names": dict(view.resource_names),
                   "payload_basis": view.payload_basis,
                   "peer_class_basis": self.peer_class_basis}
        ready_map = dict(peer_ready or {})
        send_map = dict(peer_send or {})
        source_map = dict(source_send or {})
        placeholder_map = dict(send_placeholders or {})
        phases = {phase.name for phase in self.phases}
        for label, table in (("peer_ready", ready_map),
                             ("peer_send", send_map)):
            if set(table) - phases:
                raise ModelingError(
                    f"mc2_{label}_phase_required:"
                    + ",".join(map(str, sorted(set(table) - phases))))
        def phase_map(table, phase):
            value = table.get(phase, {})
            if not isinstance(value, Mapping):
                raise ModelingError(f"mc2_{phase}_peer_map_invalid")
            unknown = set(value) - set(view.peer_labels)
            if unknown:
                raise ModelingError(
                    f"mc2_{phase}_peer_rank_not_participant:"
                    + ",".join(map(str, sorted(unknown))))
            return dict(value)
        local_count_event = None
        local_payload_event = None
        payload_transport_arrivals: tuple = ()
        for phase in self.phases:
            phase_ready, event = self._start_local_phase(
                engine, name, phase, ready, events, view, transport,
                placeholder_map, handles, local_count_event, local_payload_event,
                payload_transport_arrivals,
            )
            if phase.name == "count_status_exchange":
                local_count_event = event
            elif phase.name == "payload_exchange":
                local_payload_event = event
            link = self.link_by_phase.get(phase.name)
            ready_by_peer = phase_map(ready_map, phase.name)
            send_by_peer = phase_map(send_map, phase.name)
            event = self._start_phase_exchange(
                engine, name, phase, event, view, transport, link, handles,
                ready_by_peer, send_by_peer, source_map, placeholder_map,
            )
            if phase.name == "payload_exchange":
                payload_transport_arrivals = tuple(
                    (handles.get("_transport_arrivals") or {}).get("payload_exchange") or ()
                )
            # Authored per-owner primitive stages already execute above.
            # Do not replay the same inventory as an aggregate job.
            handles["phase_complete"].setdefault(phase.name, event)
            if records is not None:
                records.append((phase.name, phase_ready, event))
            events[phase.name] = event
        if not events:
            handles["start"] = ready
            handles["parent"] = ready
            handles["operators"] = {"mc2": {"start": ready, "complete": ready}}
            return ready, handles
        parent = engine.submit(f"{name}:complete", after=tuple(events.values()))
        handles["start"] = ready
        handles["parent"] = parent
        handles["operators"] = {"mc2": {"start": ready, "complete": parent}}
        return parent, handles

    def _has_local_payload(self, local_destinations: set[int]) -> bool:
        """Return True when local rank self-routes nonzero diagonal payload."""
        if self.route_matrix is not None:
            return any(self.local_frames_for(dst) > 0 for dst in local_destinations)
        if self.route_matrix_bytes is not None:
            return any(
                0 <= dst < len(self.route_matrix_bytes)
                and dst < len(self.route_matrix_bytes[dst])
                and float(self.route_matrix_bytes[dst][dst]) > 0.0
                for dst in local_destinations
            )
        return True

    def _receive_flag_attempts(
        self, phase, plan, program, events, view, transport, placeholder_map, handles,
        local_payload_event=None, payload_transport_arrivals=(),
    ):
        # The real MTE2/Compare success chain starts after the
        # selected inbound payload arrivals and local prerequisites.
        # Gather them before phase_ready/program.start so the actual
        # read is not launched early and followed by a fake scalar.
        if "receive_metadata" in events and "frame_payload" in events:
            local_deps = ("frame_payload", "receive_metadata")
        elif "return_read_frame" in events:
            local_deps = ("return_read_frame",)
        else:
            local_deps = phase.after
        deps = tuple(events[item] for item in local_deps)
        owners = tuple((plan.metadata or {}).get(
            "active_owner_domain") or ())
        local_destinations = {
            int(item) for item in
            (transport or {}).get("sources", ())}
        if not local_destinations:
            local_destinations = {int(view.rank)}
        incoming = ()
        if owners:
            incoming = tuple(dict.fromkeys(
                event_ for (source, destination, phase_name), event_
                in placeholder_map.items()
                if destination in local_destinations
                and phase_name == "payload_exchange"
                and source != destination))
            if payload_transport_arrivals:
                incoming = tuple(dict.fromkeys((*incoming, *payload_transport_arrivals)))
            if local_payload_event is not None and self._has_local_payload(local_destinations):
                incoming = tuple(dict.fromkeys((*incoming, local_payload_event)))
        # One genuinely source-shaped complete branch: combine
        # success-flag retry with exactly one active owner.  Other
        # owners/unsupported phases keep the existing causal wait.
        retry = bool(owners == (0,)
                     and any(str(stage.name).startswith("owner0:")
                             and str(stage.name).endswith(
                                 ("read_flags", "compare_flags"))
                             for stage in getattr(program, "stages", ())))
        init_program = read_compare_program = None
        success_sff_program = failure_sff_program = None
        if retry:
            by_name = {str(stage.name): stage
                       for stage in getattr(program, "stages", ())}
            init_stage = by_name.get("owner0:initialize_flags")
            read_stage = by_name.get("owner0:read_flags")
            compare_stage = by_name.get("owner0:compare_flags")
            success_sff = by_name.get(
                "owner0:compare_flags:after:primitive")
            required = (init_stage, read_stage, compare_stage,
                        success_sff)
            has_pipe_s = any("pipe_s" in str(role)
                             for role in self.rates)
            if (all(stage is not None for stage in required)
                    and (not any("pipe_s" in str(role)
                                 for role in success_sff.work)
                         or has_pipe_s)):
                init_stage = replace(init_stage, after=())
                read_stage = replace(read_stage, after=())
                compare_stage = replace(
                    compare_stage, after=(read_stage.name,))
                failure_work = {
                    str(role): (1.0 if "pipe_s" in str(role)
                                else float(amount))
                    for role, amount in dict(success_sff.work).items()}
                init_program = KernelProgram((init_stage,))
                read_compare_program = KernelProgram(
                    (read_stage, compare_stage))
                success_sff_program = KernelProgram((
                    replace(success_sff, after=()),))
                failure_sff_program = KernelProgram((
                    replace(success_sff, after=(),
                            work=failure_work),))
            else:
                retry = False
        if not retry and incoming:
            deps = tuple(dict.fromkeys(deps + incoming))
        handles["receive_flags_owners"] = owners
        handles["receive_flags_incoming"] = tuple(
            item.name for item in incoming)
        handles["receive_flags_retry"] = bool(retry)
        programs = (init_program, read_compare_program, success_sff_program,
                    failure_sff_program) if retry else None
        return deps, incoming, programs

    def _receive_count_specs(self, plan, program, view, transport, placeholder_map):
        count_specs = []
        active = tuple((plan.metadata or {}).get(
            "active_owner_domain") or ())
        by_name = {str(stage.name): stage
                   for stage in getattr(program, "stages", ())}
        count_local = {
            int(item) for item in
            (transport or {}).get("sources", ())}
        if not count_local:
            count_local = {int(view.rank)}
        count_cells = {
            (source, destination, pname): ev
            for (source, destination, pname), ev
            in placeholder_map.items()
            if pname == "count_status_exchange"
            and destination in count_local
            and source != destination}
        status_records = max(0, int(self.experts))
        quotient, remainder = (
            divmod(status_records, len(active)) if active else (0, 0))
        for index, owner in enumerate(active):
            start = (index * quotient
                     + min(index, remainder))
            end = start + quotient + int(index < remainder)
            required_sources = {
                row % max(1, int(self.peer_count))
                for row in range(start, end)}
            incoming = tuple(dict.fromkeys(
                ev for (source, destination, pname), ev
                in count_cells.items()
                if source in required_sources))
            prefix = f"owner{owner}:"
            owner_stages = [
                stage for stage in getattr(program, "stages", ())
                if str(stage.name).startswith(prefix)]
            read = next((stage for stage in owner_stages
                         if str(stage.name).endswith(":read_status")),
                        None)
            check = next((stage for stage in owner_stages
                          if str(stage.name).endswith(":check_status")),
                         None)
            if read is None or check is None:
                count_specs = []
                break
            read = replace(read, after=())
            check = replace(check, after=(read.name,))
            stale = {read.name, check.name}
            tail_stages = []
            for stage in owner_stages:
                if stage.name in stale:
                    continue
                after = tuple(name for name in stage.after
                              if name not in stale)
                tail_stages.append(replace(stage, after=after))
            if not tail_stages:
                count_specs = []
                break
            failed = KernelProgram((read, check))
            tail = KernelProgram(tuple(tail_stages))
            if not any(stage.work for stage in
                       (read, check, *tail.stages)):
                count_specs = []
                break
            count_specs.append((owner, incoming, failed, tail))
        return count_specs

    def _start_receive_count_retries(self, engine, name, phase_name, phase_ready, count_specs):
        count_done = engine.future(
            f"{name}:{phase_name}:success")
        remaining = [len(count_specs)]
        def owner_try(spec, after_event, *, phase_name=phase_name,
                      remaining=remaining, count_done=count_done):
            owner, incoming, failed, tail = spec
            attempt = [0]
            def finish(_event):
                remaining[0] -= 1
                if remaining[0] == 0:
                    engine.resolve(count_done)
            def start_gate(previous, *, owner=owner,
                           incoming=incoming, failed=failed,
                           tail=tail, attempt=attempt):
                def gate(_event, *, owner=owner,
                         incoming=incoming, failed=failed,
                         tail=tail, attempt=attempt):
                    complete = all(item.done for item in incoming)
                    read_event = failed.start(
                        engine,
                        f"{name}:{phase_name}:owner{owner}:"
                        f"failed{attempt[0]}",
                        _event)
                    def read_done(_read, *, owner=owner,
                                  complete=complete, tail=tail,
                                  attempt=attempt):
                        if complete:
                            final = tail.start(
                                engine,
                                f"{name}:{phase_name}:owner{owner}:"
                                "success", _read)
                            engine.submit(
                                f"{name}:{phase_name}:owner{owner}:"
                                "finish", after=(final,),
                                owner="receive_count_success",
                                on_complete=finish, native_hook=True)
                        else:
                            attempt[0] += 1
                            start_gate(_read)
                    engine.submit(
                        f"{name}:{phase_name}:owner{owner}:read_done",
                        after=(read_event,), owner="receive_count_try",
                        on_complete=read_done, native_hook=True)
                engine.submit(
                    f"{name}:{phase_name}:owner{owner}:gate{attempt[0]}",
                    after=(previous,), owner="receive_count_try",
                    on_complete=gate, native_hook=True)
            start_gate(after_event)
        for spec in count_specs:
            owner_try(spec, phase_ready)
        return count_done

    def _start_receive_flag_retries(
        self, engine, name, phase_name, phase_ready, incoming, programs,
    ):
        init_program, read_compare_program, success_sff_program, failure_sff_program = programs
        done = engine.future(f"{name}:{phase_name}:success")
        attempts = [0]
        def next_try(after_event, *, phase_name=phase_name,
                     init_program=init_program,
                     read_compare_program=read_compare_program,
                     success_sff_program=success_sff_program,
                     failure_sff_program=failure_sff_program,
                     required_incoming=incoming, done_event=done):
            attempt = attempts[0]
            init = init_program.start(
                engine, f"{name}:{phase_name}:failed{attempt}:init",
                after_event)
            def after_init(_event, *, phase_name=phase_name,
                           attempt=attempt,
                           init_program=init_program,
                           read_compare_program=read_compare_program,
                           success_sff_program=success_sff_program,
                           failure_sff_program=failure_sff_program,
                           required_incoming=required_incoming,
                           done_event=done_event):
                # Snapshot at the actual MTE2 read start, after this
                # try's initialize; never at pre-init try admission.
                complete = all(item.done for item in required_incoming)
                read_compare = read_compare_program.start(
                    engine,
                    f"{name}:{phase_name}:failed{attempt}:read_compare",
                    _event)
                def after_compare(_rc):
                    if complete:
                        sff = success_sff_program.start(
                            engine,
                            f"{name}:{phase_name}:success:sff", _rc)
                        engine.submit(
                            f"{name}:{phase_name}:success_finish",
                            after=(sff,),
                            owner="receive_flag_success", native_hook=True,
                            on_complete=lambda _done: engine.resolve(
                                done_event))
                    else:
                        attempts[0] += 1
                        sff = failure_sff_program.start(
                            engine,
                            f"{name}:{phase_name}:failed{attempt}:sff",
                            _rc)
                        next_try(sff)
                engine.submit(
                    f"{name}:{phase_name}:failed{attempt}:compare_done",
                    after=(read_compare,), owner="receive_flag_try",
                    on_complete=after_compare, native_hook=True)
            engine.submit(
                f"{name}:{phase_name}:failed{attempt}:init_done",
                after=(init,), owner="receive_flag_try",
                on_complete=after_init, native_hook=True)
        next_try(phase_ready)
        return done

    def _start_local_phase(
        self, engine, name, phase, ready, events, view, transport,
        placeholder_map, handles, local_count_event, local_payload_event=None,
        payload_transport_arrivals=(),
    ):
        plan = self.plans[phase.name]
        program = plan.program(self._context(view))
        if (phase.name in {"count_status_exchange", "payload_exchange"}
                and transport.get("destination_memory")):
            program, moved = self._endpoint_phase_program(
                program, transport, phase,
                phase.name == "count_status_exchange", view,
                self._context(view).resource_names)
        flag_programs = None
        if phase.name == "receive_flags":
            deps, incoming, flag_programs = self._receive_flag_attempts(
                phase, plan, program, events, view, transport, placeholder_map, handles,
                local_payload_event=local_payload_event,
                payload_transport_arrivals=payload_transport_arrivals,
            )
        else:
            deps = tuple(
                local_count_event
                if (
                    phase.name == "send_read_quantize"
                    and dependency == "count_status_exchange"
                    and local_count_event is not None
                )
                else events[dependency]
                for dependency in phase.after
            )

        count_specs = []
        if phase.name == "receive_counts":
            count_specs = self._receive_count_specs(plan, program, view, transport, placeholder_map)
            if count_specs and local_count_event is not None:
                deps = (local_count_event,)

        phase_ready = (engine.submit(f"{name}:{phase.name}:ready",
                                     after=deps) if deps else ready)
        if count_specs:
            event = self._start_receive_count_retries(
                engine, name, phase.name, phase_ready, count_specs,
            )
            handles["receive_counts_retry"] = True
        elif flag_programs is not None:
            event = self._start_receive_flag_retries(
                engine, name, phase.name, phase_ready, incoming, flag_programs,
            )
        else:
            event = program.start(engine, f"{name}:{phase.name}", phase_ready)
        return phase_ready, event

    def _start_frame_exchange(self, engine, name, phase, event, view, placeholder_map, handles):
        # Whole-frame projected token payload: exact selected
        # destinations.  The source transmission is one physical issue
        # per real edge; the reservation resolves only on the arrival
        # task after link/propagation, never at send.
        peer_events = []
        for peer, frames in view.payload_frames:
            if frames <= 0 or peer not in view.peer_labels:
                continue
            index = view.peer_labels.index(peer)
            klass = view.class_for(index, peer)
            pipe = "link_hccs" if klass == "hccs" else "link_roce"
            key = dict(view.resource_names).get(
                pipe, self.names.get(pipe, pipe))
            amount = _as_fraction(frames) * _as_fraction(
                view.payload_packet_bytes)
            placeholder = placeholder_map.get(
                (view.rank, peer, phase.name))
            send = engine.submit(
                f"{name}:{phase.name}:peer{peer}:send",
                work={key: float(amount)} if amount else {},
                after=(event,))
            arrival = engine.submit(
                f"{name}:{phase.name}:peer{peer}:arrival",
                latency=float(view.latency_for(klass)), after=(send,),
                native_hook=placeholder is not None,
                on_complete=(
                    (lambda _event, promise=placeholder:
                     engine.resolve(promise))
                    if placeholder is not None else None))
            peer_events.append(send)
            if placeholder is None:
                # No peer receive will consume this arrival (isolated
                # representative launch). The send is complete only
                # when its declared propagation has elapsed.
                peer_events.append(arrival)
            handles["peer_events"].setdefault(
                phase.name, {})[peer] = {
                    "send": send, "arrival": arrival,
                    "receive": None, "check": None, "retire": None,
                    "class": klass, "pipe": key,
                    "latency_ms": view.latency_for(klass),
                    "frames": frames, "bytes": amount,
                    "send_placeholder": placeholder,
                    "remote_send": None}
        for index, peer in enumerate(view.peer_labels):
            incoming = placeholder_map.get(
                (peer, view.rank, phase.name))
            if incoming is None:
                continue
            klass = view.class_for(index, peer)
            receive = engine.submit(
                f"{name}:{phase.name}:peer{peer}:receive",
                after=(event, incoming))
            check = engine.submit(
                f"{name}:{phase.name}:peer{peer}:check",
                after=(receive,))
            retire = engine.submit(
                f"{name}:{phase.name}:peer{peer}:retire",
                after=(check,))
            peer_events.append(retire)
            slot = handles["peer_events"].setdefault(
                phase.name, {}).setdefault(peer, {})
            slot.update({
                "receive": receive, "check": check, "retire": retire,
                "class": klass, "latency_ms": view.latency_for(klass),
                "remote_send": incoming})
        handles["payload_frames"] = dict(view.payload_frames)
        handles["payload_basis"] = view.payload_basis
        handles["payload_local_frames"] = view.payload_local_frames
        if peer_events:
            event = engine.submit(
                f"{name}:{phase.name}:peer_complete",
                after=tuple(peer_events))
        return event

    def _start_weighted_peer_exchange(
        self, engine, name, phase, event, view, link, handles, ready_by_peer,
        send_by_peer, source_map, placeholder_map,
    ):
        class_peers = {"hccs": [], "roce": []}
        for index, peer in enumerate(view.peer_labels):
            klass = view.class_for(index, peer)
            class_peers[klass].append((index, peer))
            handles["peer_class"][peer] = klass
        class_weight = {}
        for klass, members in class_peers.items():
            class_weight[klass] = sum(
                (view.peer_weights[index] for index, _ in members),
                Fraction(0))
        peer_events = []
        for klass, members in class_peers.items():
            if not members:
                continue
            pipe = "link_hccs" if klass == "hccs" else "link_roce"
            total = _as_fraction(link.get(pipe, 0))
            if (total <= 0 and "link" in link and
                    sum(1 for items in class_peers.values() if items) == 1):
                # A one-class synthetic supply declares only `link`.
                total = _as_fraction(link.get("link", 0))
            key = dict(view.resource_names).get(
                pipe, self.names.get(pipe, pipe))
            for index, peer in members:
                weight = (view.peer_weights[index] /
                          class_weight[klass]
                          if class_weight[klass] else Fraction(0))
                send = send_by_peer.get(peer)
                if send is None:
                    send = source_map.get(
                        (view.rank, peer, phase.name))
                placeholder = placeholder_map.get(
                    (view.rank, peer, phase.name))
                arrival = None
                if send is None:
                    work = ({key: float(total * weight)}
                            if total * weight else {})
                    send = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:send",
                        work=work, after=(event,))
                    arrival = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:arrival",
                        latency=float(view.latency_for(klass)),
                        after=(send,), native_hook=placeholder is not None,
                        on_complete=(
                            (lambda _event, _placeholder=placeholder:
                             engine.resolve(_placeholder))
                            if placeholder is not None else None))
                incoming = source_map.get(
                    (peer, view.rank, phase.name))
                if incoming is None:
                    incoming = placeholder_map.get(
                        (peer, view.rank, phase.name))
                receive = check = retire = None
                if incoming is not None:
                    receive_deps = (incoming,)
                    if peer in ready_by_peer:
                        receive_deps += (ready_by_peer[peer],)
                    receive = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:receive",
                        after=receive_deps)
                    check = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:check",
                        after=(receive,))
                    retire = engine.submit(
                        f"{name}:{phase.name}:peer{peer}:retire",
                        after=(check,))
                    peer_events.append(retire)
                peer_events.append(send)
                slot = handles["peer_events"].setdefault(
                    phase.name, {}).setdefault(peer, {})
                slot.update({
                    "send": send, "arrival": arrival,
                    "receive": receive, "check": check,
                    "retire": retire, "class": klass, "pipe": key,
                    "latency_ms": view.latency_for(klass),
                    "send_placeholder": placeholder,
                    "remote_send": incoming})
                handles["ports"][peer] = {
                    "source": view.rank, "peer": peer,
                    "class": klass, "pipe": key,
                    "latency_ms": view.latency_for(klass)}
        event = engine.submit(
            f"{name}:{phase.name}:peer_complete",
            after=tuple(peer_events))
        handles["phase_complete"][phase.name] = event
        return event

    def _start_phase_exchange(
        self, engine, name, phase, event, view, transport, link, handles,
        ready_by_peer, send_by_peer, source_map, placeholder_map,
    ):
        if (transport and callable(transport.get("route"))
                and phase.name in {"payload_exchange", "count_status_exchange"}
                and (self.has_payload_policy
                     or phase.name == "count_status_exchange")):
            details = self._schedule_transport(
                engine, name, phase, event, view, transport,
                count=phase.name == "count_status_exchange",
                placeholders=placeholder_map)
            event = details["event"]
            handles["transport"][phase.name] = {
                key: value for key, value in details.items()
                if key not in {"event", "peer_events", "incoming_arrivals"}}
            if details.get("incoming_arrivals"):
                handles.setdefault("_transport_arrivals", {})[phase.name] = details["incoming_arrivals"]
            for key, amount in details["work"].items():
                handles["transport_work"][key] = (
                    handles["transport_work"].get(key, 0.0) + amount)
            if details["peer_events"]:
                handles["peer_events"][phase.name] = dict(
                    details["peer_events"])
        elif phase.name == "payload_exchange" and view.payload_frames:
            event = self._start_frame_exchange(
                engine, name, phase, event, view, placeholder_map, handles,
            )
        elif (link and view.peer_labels and
              not (phase.name == "payload_exchange" and
                   (self.exact_payload or
                    view.payload_basis == "unknown"))):
            event = self._start_weighted_peer_exchange(
                engine, name, phase, event, view, link, handles, ready_by_peer,
                send_by_peer, source_map, placeholder_map,
            )
        elif phase.name == "payload_exchange":
            # Idle/zero-source rank: it has no outbound payload wire, but
            # active peers can still send outward payload to it.  Consume
            # only the matching incoming (peer, self, phase) placeholders;
            # do not invent a local send.
            incoming_events = []
            for index, peer in enumerate(view.peer_labels):
                incoming = placeholder_map.get(
                    (peer, view.rank, phase.name))
                if incoming is None:
                    continue
                klass = view.class_for(index, peer)
                latency = view.latency_for(klass)
                receive = engine.submit(
                    f"{name}:{phase.name}:peer{peer}:receive",
                    after=(event, incoming))
                check = engine.submit(
                    f"{name}:{phase.name}:peer{peer}:check",
                    after=(receive,))
                retire = engine.submit(
                    f"{name}:{phase.name}:peer{peer}:retire",
                    after=(check,))
                incoming_events.append(retire)
                handles["peer_events"].setdefault(
                    phase.name, {})[peer] = {
                        "send": None, "receive": receive,
                        "check": check, "retire": retire,
                        "class": klass, "pipe": None,
                        "latency_ms": latency,
                        "send_placeholder": None,
                        "remote_send": incoming}
            if incoming_events:
                event = engine.submit(
                    f"{name}:{phase.name}:peer_complete",
                    after=tuple(incoming_events))
        elif link:
            link_work = {}
            for pipe, amount in link.items():
                key = dict(view.resource_names).get(
                    pipe, self.names.get(pipe, pipe))
                link_work[key] = link_work.get(key, Fraction(0)) + amount
            event = engine.submit(
                f"{name}:{phase.name}:link",
                work={pipe: float(amount)
                      for pipe, amount in link_work.items() if amount},
                after=(event,))
            if view.intra_latency_ms:
                event = engine.submit(
                    f"{name}:{phase.name}:propagation",
                    latency=float(view.intra_latency_ms), after=(event,))
        return event
