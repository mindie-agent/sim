"""Finite direct-peer transport without receive-forward round barriers.

[Role] Schedule explicitly selected independent remote-window peer messages on
  the existing topology and Engine, including asymmetric cross-DP cohorts and
  topology-declared full-mesh ReduceScatter/AllGather phases.
[Invariants] Each directed cell is routed and framed once. A declared mesh
  phase issues one equal payload to every directed peer concurrently behind a
  single physical phase startup; AllReduce keeps dependent ReduceScatter then
  AllGather phases, so its span is two peer services rather than 2*(W-1) ring
  rounds. A sender owns a
  declared finite number of outstanding slots.  When no finite window is
  declared, the topology-derived maximum is the remote-peer count (one
  destination slot per peer); the actual finite vendor buffer window may be
  smaller and is an explicit unknown.  An optional chunk pipeline keeps
  per-hop order while overlapping the next chunk's transmission; a finite
  per-peer receive credit is reserved before send and released on arrival.
  Missing chunk/startup/credit inputs remain unknowns, never fitted defaults.
  A slot is reused only after remote arrival. No peer waits for an unrelated
  prior peer's receive. Rank completion includes every required incoming and
  outgoing message. Local quantization, packing and expert computation remain
  separate caller work.
[Timeouts & Locks] O(ranks * outstanding slots) in-flight state; bounded chunk
  count per message; virtual events.
[Side-Effect Policy] Engine tasks only; no I/O or hardware execution.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import ClassVar, Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.collectives import (PairwiseCollective, PairwiseGroup,
                           _joint_complete, _ready_by_rank,
                           plan_options as collective_plan_options)
from mindie_sim.prediction.communication import CommunicationWork, MessageFlow
from mindie_sim.prediction.topology import (
    DEFAULT_MAX_PHYSICAL_CHUNKS, TransportOptions, chunk_count, chunk_size_at, chunk_sizes,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
US_PER_MS = 1000.0
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="hicann/hccl",
        revision="c2d6790464654ccf58aca39448de4b65fdc7b184",
        relative_path="src/ops/all_reduce/algorithm/template/aicpu/ins_temp_all_reduce_mesh_1D_two_shot.cc",
        source_sha256="00a81cd91df30b50132b5eb22328d7c468a16a185432d8649c76ad8e14434784",
    ),
)



def resolve_transport_options(recipe, op):
    """Resolve the selected recipe's transport options once for both consumers.

    T04 can call this at ``Execution._context``/joint-cohort binding and pass
    the result to ``Topology.joint_plan`` and ``Topology.accounting``.
    SIM-STRUCTURE consumes the same object rather than re-reading metadata.
    """
    raw = collective_plan_options(recipe, op)
    return TransportOptions.from_mapping(raw)


@dataclass(frozen=True)
class DirectPeerGroup(PairwiseGroup):
    # None means the physical bound is the actual remote peer count: one
    # outstanding destination slot per peer, not a fitted vendor window.
    max_inflight_per_rank: int | None = None
    chunk_bytes: float | None = None
    hop_startup_us: float = 0.0
    receive_capacity_bytes_per_peer: float | None = None
    receive_buffer_id: str | None = None
    max_physical_chunks: int | None = DEFAULT_MAX_PHYSICAL_CHUNKS
    receive_capacity_bytes: tuple[tuple[int, float], ...] = ()
    receive_buffer_ids: tuple[tuple[int, str], ...] = ()

    def __post_init__(self):
        if self.max_physical_chunks is not None and (
                isinstance(self.max_physical_chunks, bool)
                or not isinstance(self.max_physical_chunks, int)
                or self.max_physical_chunks < 1):
            raise ModelingError(
                "direct_peer_max_physical_chunks_requires_positive_integer")
        if self.chunk_bytes is not None:
            if isinstance(self.chunk_bytes, bool):
                raise ModelingError(
                    "direct_peer_chunk_bytes_requires_positive_number")
            value = float(self.chunk_bytes)
            if not math.isfinite(value) or value <= 0:
                raise ModelingError("direct_peer_chunk_bytes_requires_positive_number")
        if (not isinstance(self.hop_startup_us, (int, float))
                or isinstance(self.hop_startup_us, bool)
                or not math.isfinite(float(self.hop_startup_us))
                or float(self.hop_startup_us) < 0):
            raise ModelingError("direct_peer_hop_startup_requires_nonnegative_number")
        if self.receive_buffer_id is not None and not str(
                self.receive_buffer_id):
            raise ModelingError("direct_peer_receive_buffer_id_invalid")
        if self.receive_capacity_bytes_per_peer is not None:
            if isinstance(self.receive_capacity_bytes_per_peer, bool):
                raise ModelingError(
                    "direct_peer_receive_capacity_requires_positive_number")
            value = float(self.receive_capacity_bytes_per_peer)
            if not math.isfinite(value) or value <= 0:
                raise ModelingError(
                    "direct_peer_receive_capacity_requires_positive_number")
        for _rank, raw_cap in self.receive_capacity_bytes:
            if isinstance(raw_cap, bool):
                raise ModelingError(
                    "direct_peer_receive_capacity_requires_positive_number")
            value = float(raw_cap)
            if not math.isfinite(value) or value <= 0:
                raise ModelingError(
                    "direct_peer_receive_capacity_requires_positive_number")
        for _rank, raw_buf in self.receive_buffer_ids:
            if not str(raw_buf):
                raise ModelingError("direct_peer_receive_buffer_id_invalid")
        if (self.receive_capacity_bytes_per_peer is not None
                and self.receive_capacity_bytes):
            raise ModelingError("direct_peer_receive_capacity_conflict")

    def _receive_pool_spec(self, source: int, destination: int) -> tuple[float | None, str | None]:
        rank_cap = next(
            (float(val) for r, val in self.receive_capacity_bytes if int(r) == destination),
            None,
        )
        rank_buf = next(
            (str(val) for r, val in self.receive_buffer_ids if int(r) == destination),
            None,
        )
        if rank_cap is not None:
            buf_id = rank_buf or self.receive_buffer_id or f"{self.topology.namespace}rank:{destination}"
            return rank_cap, f"transport:rx:{buf_id}"
        if self.receive_capacity_bytes_per_peer is not None:
            identity = (
                rank_buf
                or self.receive_buffer_id
                or f"{self.topology.namespace}peer:{source}>{destination}"
            )
            return float(self.receive_capacity_bytes_per_peer), f"transport:rx:{identity}"
        return None, None

    def start_participants(self, engine, name, ready_by_rank, *, owner="",
                           release_events=None):
        initial = _ready_by_rank(self.ranks, ready_by_rank)
        count = len(self.ranks)
        if count <= 1:
            return initial
        if (self.chunk_bytes is not None or self.hop_startup_us
                or self.receive_capacity_bytes
                or self.receive_capacity_bytes_per_peer is not None
                or release_events is not None):
            return self._start_chunked(
                engine, name, initial, owner=owner,
                release_events=release_events)
        volumes = dict(self.volumes)
        peer = self.peer_traffic
        # Count obligations without keeping a second dense traffic matrix.
        expected_receive = [0] * count
        for src in range(count):
            for dst in range(count):
                if src != dst and self._payload(src, dst, volumes, peer):
                    expected_receive[dst] += 1
        window = (self.max_inflight_per_rank if self.max_inflight_per_rank is not None
                  else max(1, count - 1))
        complete = {r: engine.future(f"{name}:rank{r}:complete") for r in self.ranks}
        issued = [0] * count
        outstanding = [0] * count
        received = [0] * count
        explicit_events = {}
        for key, value in (release_events or {}).items():
            events = (tuple(value) if isinstance(value, (tuple, list))
                      else (value,))
            if isinstance(key, tuple):
                pair = (int(key[0]), int(key[1]))
                explicit_events[pair] = events
            else:
                explicit_events[("dest", int(key))] = events
        release_cursor = {}

        def retire(index):
            if (issued[index] == count - 1 and not outstanding[index]
                    and received[index] == expected_receive[index]
                    and initial[self.ranks[index]].done):
                event = complete[self.ranks[index]]
                if not event.done:
                    engine.resolve(event)

        def issue(index):
            while (issued[index] < count - 1
                   and outstanding[index] < window):
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

    def _start_peer_message(
        self, engine, name, initial, owner, index, destination_index, payload,
        release_chunk, finish_message, *, has_explicit_release=False,
    ):
        """[Cohesive-Pipeline] Schedule one direct-peer collective chunk transfer across link resources."""
        source, destination = self.ranks[index], self.ranks[destination_index]
        prefix = f"{name}:rank{source}:peer{destination}"
        capacity, pool = self._receive_pool_spec(source, destination)
        total = chunk_count(float(payload), self.chunk_bytes)
        if (self.max_physical_chunks is not None
                and total > self.max_physical_chunks):
            raise ModelingError(
                "transport_chunk_expansion_unbounded:"
                f"{total}>{self.max_physical_chunks}")
        largest = chunk_size_at(float(payload), self.chunk_bytes, 0)
        if capacity is not None and largest > float(capacity) + 1e-9:
            raise ModelingError(
                "direct_peer_chunk_exceeds_receive_capacity:"
                f"{largest}>{capacity}")
        shared_pool = bool(
            self.receive_capacity_bytes
            or self.receive_buffer_ids
            or self.receive_buffer_id
        )
        atomic_message_credit = (
            pool is not None
            and capacity is not None
            and shared_pool
            and total > 1
            and has_explicit_release
            and float(payload) <= float(capacity) + 1e-9
        )
        if capacity is None or atomic_message_credit:
            chunk_window = None
        else:
            chunk_window = max(
                1, int(float(capacity) // largest + 1e-9))
        services = {}
        remaining = [total]
        arrived_chunks = [0]
        issued_chunks = [0]
        hop0_done = [0]
        # Local indices only index the peer matrix.  The consumption
        # binder, pool identity and release lookup all use physical ranks.
        pair = (source, destination)
        ready = engine.submit(
            prefix + ":ready", after=(initial[source], initial[destination]),
            owner=owner,
            metadata={"source_rank": source, "destination_rank": destination,
                      "transport_schedule": "chunked_direct_peer",
                      "chunk_bytes": self.chunk_bytes,
                      "receive_capacity_bytes_per_peer":
                          self.receive_capacity_bytes_per_peer,
                      **({"receive_capacity_bytes": capacity}
                         if capacity is not None and self.receive_capacity_bytes
                         else {})})
        if pool is not None and capacity is not None:
            if pool in engine.pools:
                if not math.isclose(
                        engine.pools[pool],
                        float(capacity),
                        rel_tol=1e-12, abs_tol=1e-9):
                    raise ModelingError(
                        f"direct_peer_receive_capacity_conflict:{pool}")
            else:
                engine.extend(pools={pool: float(capacity)})
        msg_allocation = None
        message_ready = ready
        if atomic_message_credit:
            msg_allocation = Allocation(
                f"{prefix}:credit", pool, float(payload), (0,))
            message_ready = engine.submit(
                f"{prefix}:grant", after=(ready,),
                acquire=(msg_allocation,), owner=owner)

        def arrived(_event, chunk_index, prefix, allocation, hop_count):
            remaining[0] -= 1
            arrived_chunks[0] += 1
            if hop_count == 0:
                hop0_done[0] += 1
            if atomic_message_credit:
                if remaining[0] == 0:
                    release_chunk(prefix, pair, msg_allocation, _event)
            else:
                release_chunk(prefix, pair, allocation, _event)
            pump()
            if remaining[0]:
                return
            finish_message(index, destination_index)

        def on_hop0_done(_event, chunk_index):
            hop0_done[0] += 1
            if chunk_index + 1 < total:
                pump()

        def schedule_chunk(chunk_index):
            useful = chunk_size_at(
                float(payload), self.chunk_bytes, chunk_index)
            chunk = self.topology.route_message(MessageFlow(
                source, destination, useful)).hops
            dependencies = [message_ready]
            previous_services = services.get(chunk_index - 1)
            if previous_services is not None:
                dependencies.append(previous_services[0])
            allocation = None
            if pool is not None and not atomic_message_credit:
                allocation = Allocation(
                    f"{prefix}:chunk{chunk_index}:credit", pool,
                    useful, (0,))
                gate = engine.submit(
                    f"{prefix}:chunk{chunk_index}:grant",
                    after=tuple(dependencies), acquire=(allocation,),
                    owner=owner)
            else:
                gate = engine.submit(
                    f"{prefix}:chunk{chunk_index}:gate",
                    after=tuple(dependencies), owner=owner)
            previous = gate
            startup_ms = float(self.hop_startup_us) / US_PER_MS
            has_tx = False
            has_rx = False
            if chunk_index == 0 and startup_ms:
                route_resources = {
                    res for hop in chunk
                    for res in (hop.resource, *hop.additional_resources)
                }
                has_tx = any(
                    res.endswith(f":tx:port{source}") for res in route_resources)
                has_rx = any(
                    res.endswith(f":rx:port{destination}") for res in route_resources)
            for hop_index, hop in enumerate(chunk):
                dependencies = [previous]
                if previous_services is not None:
                    dependencies.append(previous_services[hop_index])
                if chunk_index == 0 and startup_ms:
                    if has_tx and hop_index == 0:
                        inject = engine.submit(
                            f"{prefix}:inject:startup",
                            after=tuple(dependencies),
                            latency=startup_ms,
                            owner=owner,
                            metadata={"startup_boundary": "source_injection_once"})
                        dependencies = [inject]
                    startup = engine.submit(
                        f"{prefix}:hop{hop_index}:startup",
                        after=tuple(dependencies),
                        latency=startup_ms,
                        owner=owner,
                        metadata={"startup_boundary": "per_hop_protocol_once"})
                    dependencies = [startup]
                service = engine.submit(
                    f"{prefix}:chunk{chunk_index}:hop{hop_index}:service",
                    hop.demand, after=tuple(dependencies), owner=owner,
                    coupled=True,
                    on_complete=(
                        (lambda event, ci=chunk_index: on_hop0_done(event, ci))
                        if hop_index == 0
                        else None))
                arrival = engine.submit(
                    f"{prefix}:chunk{chunk_index}:hop{hop_index}:arrival",
                    after=(service,), latency=hop.latency_ms, owner=owner)
                services.setdefault(
                    chunk_index, [None] * len(chunk))[hop_index] = service
                previous = arrival
            if chunk_index == 0 and startup_ms and has_rx:
                previous = engine.submit(
                    f"{prefix}:receive:startup", after=(previous,),
                    latency=startup_ms, owner=owner,
                    metadata={"startup_boundary": "destination_receipt_once"})
            received_event = engine.submit(
                f"{prefix}:chunk{chunk_index}:received", after=(previous,),
                owner=owner,
                on_complete=lambda event, ci=chunk_index, p=prefix,
                a=allocation, hc=len(chunk): arrived(event, ci, p, a, hc))

        def pump():
            if chunk_window is None:
                limit = min(total, hop0_done[0] + 1)
            else:
                limit = min(total, arrived_chunks[0] + chunk_window + 1)
            while issued_chunks[0] < limit:
                index_to_issue = issued_chunks[0]
                issued_chunks[0] += 1
                schedule_chunk(index_to_issue)
                if index_to_issue >= 2:
                    services.pop(index_to_issue - 2, None)

        pump()

    def _start_chunked(self, engine, name, initial, *, owner="",
                       release_events=None):
        """Chunk-pipelined peer messages with optional receive credits.

        Chunks of one peer message are ordered on each hop but pipeline across
        hops.  A finite receiver credit is reserved before the first hop and
        released when the chunk arrives; the next queued chunk is then issued.
        Credits are per directed peer channel, never a collective-wide barrier.
        """
        count = len(self.ranks)
        volumes = dict(self.volumes)
        peer = self.peer_traffic
        expected_receive = [0] * count
        for src in range(count):
            for dst in range(count):
                if src != dst and self._payload(src, dst, volumes, peer):
                    expected_receive[dst] += 1
        window = (self.max_inflight_per_rank
                  if self.max_inflight_per_rank is not None else max(1, count - 1))
        complete = {r: engine.future(f"{name}:rank{r}:complete") for r in self.ranks}
        issued = [0] * count
        outstanding = [0] * count
        received = [0] * count
        explicit_events = {}
        for key, value in (release_events or {}).items():
            events = (tuple(value) if isinstance(value, (tuple, list))
                      else (value,))
            if isinstance(key, tuple):
                pair = (int(key[0]), int(key[1]))
                explicit_events[pair] = events
            else:
                explicit_events[("dest", int(key))] = events
        release_cursor = {}

        def retire(index):
            if (issued[index] == count - 1 and not outstanding[index]
                    and received[index] == expected_receive[index]
                    and initial[self.ranks[index]].done):
                event = complete[self.ranks[index]]
                if not event.done:
                    engine.resolve(event)

        def issue(index):
            while (issued[index] < count - 1
                   and outstanding[index] < window):
                offset = issued[index] + 1
                issued[index] += 1
                destination_index = (index + offset) % count
                payload = self._payload(index, destination_index, volumes, peer)
                if not payload:
                    continue
                outstanding[index] += 1
                pair = (self.ranks[index], self.ranks[destination_index])
                has_explicit_release = (
                    pair in explicit_events
                    or ("dest", pair[1]) in explicit_events
                )
                self._start_peer_message(
                    engine, name, initial, owner, index, destination_index, payload,
                    release_chunk, finish_message,
                    has_explicit_release=has_explicit_release,
                )
            retire(index)

        def finish_message(index, destination_index):
            outstanding[index] -= 1
            received[destination_index] += 1
            retire(destination_index)
            issue(index)

        def release_chunk(prefix, pair, allocation, arrival):
            _, destination = pair
            events = explicit_events.get(
                pair, explicit_events.get(("dest", destination), ()))
            explicit = (pair in explicit_events
                        or ("dest", destination) in explicit_events)
            cursor = release_cursor.get(pair, 0)
            while cursor < len(events) and events[cursor].done:
                cursor += 1
            release_cursor[pair] = cursor
            if explicit and cursor >= len(events):
                return
            after = (arrival,)
            if cursor < len(events):
                after = (arrival, events[cursor])
            if allocation is not None:
                engine.submit(
                    prefix + ":release", after=after,
                    free=(allocation,), owner=owner)

        for index, rank in enumerate(self.ranks):
            engine.submit(f"{name}:rank{rank}:entry", after=(initial[rank],),
                          owner=owner, on_complete=lambda _, i=index: issue(i))
        return complete

    def _payload(self, source, destination, volumes, peer):
        return (peer.payload(source, destination) if peer is not None
                else volumes[self.ranks[source]] / (len(self.ranks) - 1))


@dataclass(frozen=True)
class DirectMeshPhase:
    """One full-mesh peer phase: every directed pair carries the same payload.

    All W*(W-1) directed messages of a phase are issued together behind one
    declared phase startup; each directed peer owns its own capacity, so the
    phase span is the single-message service plus propagation, not W-1 rounds.
    """

    ranks: tuple[int, ...]
    payload_bytes: float
    topology: object = field(repr=False, compare=False)
    phase: str = "peer_phase"
    startup_ms: float = 0.0

    def message_flows(self):
        if not self.payload_bytes or len(self.ranks) <= 1:
            return
        for source in self.ranks:
            for destination in self.ranks:
                if source != destination:
                    yield MessageFlow(source, destination, self.payload_bytes)

    def routed_messages(self):
        for flow in self.message_flows():
            yield self.topology.route_message(flow)

    def start_participants(self, engine, name, ready_by_rank, *, owner=""):
        initial = _ready_by_rank(self.ranks, ready_by_rank)
        if not self.payload_bytes or len(self.ranks) <= 1:
            return initial
        # One physical phase startup, paid once by the whole cohort.  Distinct
        # rank readiness stays a real dependency of that startup gate.
        gate = engine.submit(
            f"{name}:startup", after=tuple(dict.fromkeys(initial.values())),
            latency=float(self.startup_ms), owner=owner,
            metadata={"transport_schedule": "direct_mesh",
                      "phase": self.phase,
                      "directed_peer_phase": True})
        received = {rank: [] for rank in self.ranks}
        for route in self.routed_messages():
            source, destination = route.flow.source, route.flow.destination
            prefix = f"{name}:rank{source}:peer{destination}"
            previous = engine.submit(
                prefix + ":ready", after=(gate,), owner=owner,
                metadata={"source_rank": source, "destination_rank": destination,
                          "transport_schedule": "direct_mesh",
                          "phase": self.phase})
            for hop_index, hop in enumerate(route.hops):
                service = engine.submit(
                    f"{prefix}:hop{hop_index}:service", hop.demand,
                    after=(previous,), owner=owner, coupled=True)
                previous = engine.submit(
                    f"{prefix}:hop{hop_index}:arrival", after=(service,),
                    latency=hop.latency_ms, owner=owner)
            received[source].append(previous)
            received[destination].append(previous)
        return {
            rank: (engine.submit(f"{name}:rank{rank}:complete",
                                 after=tuple(parts), owner=owner)
                   if parts else initial[rank])
            for rank, parts in received.items()
        }

    def start(self, engine, name, ready, *, owner=""):
        return _joint_complete(
            engine, name,
            self.start_participants(
                engine, name, {rank: ready for rank in self.ranks},
                owner=owner).values(),
            ready, owner)


@dataclass(frozen=True)
class DirectMeshGroup:
    """A topology-declared full mesh with dependent physical phases."""

    ranks: tuple[int, ...]
    volumes: tuple[tuple[int, float], ...]
    phases: tuple[DirectMeshPhase, ...]
    topology: object = field(repr=False, compare=False)

    def phase(self, name: str) -> DirectMeshPhase:
        for phase in self.phases:
            if phase.phase == name:
                return phase
        raise ModelingError(f"direct_mesh_phase_unknown:{name}")

    def start_participants(self, engine, name, ready_by_rank, *, owner=""):
        events = _ready_by_rank(self.ranks, ready_by_rank)
        for index, phase in enumerate(self.phases):
            # A later phase starts only after every rank's prior phase.  The
            # phase startup gate is that physical dependency, not an extra
            # socket round or a ring barrier.
            events = phase.start_participants(
                engine, f"{name}:phase{index}", events, owner=owner)
        return events

    def start(self, engine, name, ready, *, owner=""):
        return _joint_complete(
            engine, name,
            self.start_participants(
                engine, name, {rank: ready for rank in self.ranks},
                owner=owner).values(),
            ready, owner)


@dataclass(frozen=True)
class DirectMeshPlanPhase:
    """Executable view of one named phase across every declared mesh group."""

    plan: object = field(repr=False, compare=False)
    phase: str = "peer_phase"

    def start(self, engine, name, ready, *, owner=""):
        parts = [
            group.phase(self.phase).start(
                engine, f"{name}:group{index}", ready, owner=owner)
            for index, group in enumerate(self.plan.groups)
            if len(group.ranks) > 1
        ]
        if not parts:
            return ready
        return engine.submit(f"{name}:complete", after=tuple(parts),
                             owner=owner)


@dataclass(frozen=True)
class DirectMeshPlan:
    """Dependent ReduceScatter/AllGather phases over declared directed pairs."""

    family: str
    groups: tuple[DirectMeshGroup, ...]
    #: Marks the plan consumed by the topology collective recipe.
    mesh_algorithm: ClassVar[str] = "direct_mesh"

    @property
    def ranks(self):
        return tuple(sorted({rank for group in self.groups
                             for rank in group.ranks}))

    @property
    def phase_names(self):
        return (tuple(phase.phase for phase in self.groups[0].phases)
                if self.groups else ())

    def phase_program(self, name: str):
        for group in self.groups:
            group.phase(name)
        return DirectMeshPlanPhase(self, name)

    def start_participants(self, engine, name, ready_by_rank, *, owner=""):
        _ready_by_rank(self.ranks, ready_by_rank)
        events = {}
        for index, group in enumerate(self.groups):
            part = group.start_participants(
                engine, f"{name}:group{index}",
                {rank: ready_by_rank[rank] for rank in group.ranks},
                owner=owner)
            overlap = set(events) & set(part)
            if overlap:
                raise ModelingError(
                    "direct_mesh_collective_rank_accounted_twice:"
                    + ",".join(map(str, sorted(overlap))))
            events.update(part)
        if set(events) != set(self.ranks):
            raise ModelingError("direct_mesh_collective_group_completion_missing")
        return events

    def start(self, engine, name, ready, *, owner=""):
        return _joint_complete(
            engine, name,
            self.start_participants(
                engine, name, {rank: ready for rank in self.ranks},
                owner=owner).values(),
            ready, owner)

    @staticmethod
    def _accumulate_hop_resources(hop, useful, wire, transactions, frames):
        for resource in hop.demand:
            useful[resource] = useful.get(resource, 0.0) + hop.useful_bytes
            wire[resource] = wire.get(resource, 0.0) + hop.wire_bytes
            previous_tx = transactions.get(resource, 0)
            if hop.transactions is None or previous_tx is None:
                transactions[resource] = None
            else:
                transactions[resource] = previous_tx + hop.transactions
            if resource not in frames:
                frames[resource] = hop.transaction_frame
            elif frames[resource] != hop.transaction_frame:
                frames[resource] = None

    def accounting(self):
        """Physical wire of the selected phases, with no aggregate setup term.

        Phase startup and hop propagation are real program events, so the
        ledger reports ``setup_ms=0`` instead of re-adding a serial estimate.
        """
        useful_by_rank, useful, wire = {}, {}, {}
        transactions, frames = {}, {}
        messages, routed_useful, physical_wire = 0, 0.0, 0.0
        max_path_latency = 0.0
        for group in self.groups:
            for rank, volume in group.volumes:
                useful_by_rank[rank] = useful_by_rank.get(rank, 0.0) + volume
            for phase in group.phases:
                for route in phase.routed_messages():
                    messages += 1
                    path_latency = 0.0
                    for hop in route.hops:
                        routed_useful += hop.useful_bytes
                        physical_wire += hop.wire_bytes
                        self._accumulate_hop_resources(
                            hop, useful, wire, transactions, frames
                        )
                        path_latency += hop.latency_ms
                    max_path_latency = max(max_path_latency, path_latency)
        uniform_frames = {
            resource: frame for resource, frame in frames.items()
            if frame is not None
        }
        return CommunicationWork(
            useful_by_rank, useful, wire, transactions, uniform_frames, messages,
            0.0, routed_useful, physical_wire, max_path_latency)

    def wire_by_resource(self):
        return dict(self.accounting().wire_by_resource)

    def validate_path_model(self, path_model):
        if path_model not in {"direct", None}:
            raise ModelingError(
                f"direct_mesh_collective_path_model_unsupported:{path_model}")
        for group in self.groups:
            for phase in group.phases:
                for route in phase.routed_messages():
                    if len(route.hops) != 1:
                        raise ModelingError(
                            "direct_mesh_collective_requires_one_hop_peer")


@dataclass(frozen=True)
class DirectPeerCollective(PairwiseCollective):
    """Declared coalesced peer window; not an inferred vendor token scheduler."""

    max_inflight_per_rank: int | None = field(default=None, kw_only=True)
    chunk_bytes: float | None = field(default=None, kw_only=True)
    hop_startup_us: float = field(default=0.0, kw_only=True)
    receive_capacity_bytes_per_peer: float | None = field(
        default=None, kw_only=True)
    receive_buffer_id: str | None = field(default=None, kw_only=True)
    max_physical_chunks: int | None = field(default=DEFAULT_MAX_PHYSICAL_CHUNKS, kw_only=True)
    algorithm: ClassVar[str] = "direct_peer"

    def __post_init__(self):
        if self.max_inflight_per_rank is not None and (
                not isinstance(self.max_inflight_per_rank, int)
                or isinstance(self.max_inflight_per_rank, bool)
                or self.max_inflight_per_rank < 1):
            raise ModelingError("direct_peer_window_requires_positive_integer")
        if self.chunk_bytes is not None:
            if isinstance(self.chunk_bytes, bool):
                raise ModelingError(
                    "direct_peer_chunk_bytes_requires_positive_number")
            value = float(self.chunk_bytes)
            if not math.isfinite(value) or value <= 0:
                raise ModelingError("direct_peer_chunk_bytes_requires_positive_number")
        if (not isinstance(self.hop_startup_us, (int, float))
                or isinstance(self.hop_startup_us, bool)
                or not math.isfinite(float(self.hop_startup_us))
                or float(self.hop_startup_us) < 0):
            raise ModelingError(
                "direct_peer_hop_startup_requires_nonnegative_number")
        if self.receive_buffer_id is not None and not str(
                self.receive_buffer_id):
            raise ModelingError("direct_peer_receive_buffer_id_invalid")
        if self.receive_capacity_bytes_per_peer is not None:
            if isinstance(self.receive_capacity_bytes_per_peer, bool):
                raise ModelingError(
                    "direct_peer_receive_capacity_requires_positive_number")
            value = float(self.receive_capacity_bytes_per_peer)
            if not math.isfinite(value) or value <= 0:
                raise ModelingError(
                    "direct_peer_receive_capacity_requires_positive_number")

    def resolve_transport(self, op=None):
        return TransportOptions.from_mapping(self.plan_options)

    @property
    def plan_options(self):
        result = {"max_inflight_per_rank": self.max_inflight_per_rank}
        if self.chunk_bytes is not None:
            result["chunk_bytes"] = self.chunk_bytes
        if self.hop_startup_us:
            result["hop_startup_us"] = self.hop_startup_us
        if self.receive_capacity_bytes_per_peer is not None:
            result["receive_capacity_bytes_per_peer"] = (
                self.receive_capacity_bytes_per_peer)
        if self.receive_buffer_id is not None:
            result["receive_buffer_id"] = self.receive_buffer_id
        result["max_physical_chunks"] = self.max_physical_chunks
        return result

    def transport_unknowns(self):
        unknowns = {}
        if self.max_inflight_per_rank is None:
            unknowns["finite_vendor_peer_window"] = (
                "not_declared; topology-derived remote-peer bound used")
        if self.chunk_bytes is None:
            unknowns["transfer_chunk_bytes"] = (
                "not_declared; one peer message per hop")
        if not self.hop_startup_us:
            unknowns["protocol_startup_us"] = (
                "not_declared; no startup gate submitted")
        if self.receive_capacity_bytes_per_peer is None:
            unknowns["receive_capacity_bytes_per_peer"] = (
                "not_declared; no finite receive credit")
        return unknowns

    def accounting_description(self, accounting):
        result = accounting.to_dict()
        serial = result.pop("setup_ms")
        return {**result, "setup_ms": 0.0,
                "unused_serial_round_estimate_ms": serial,
                "latency_model": (
                    "chunk_pipeline_with_receive_credits"
                    if self.chunk_bytes is not None
                    or self.receive_capacity_bytes_per_peer is not None
                    else "per_hop_arrival_in_finite_direct_peer_window"),
                **self.plan_options,
                "unknowns": self.transport_unknowns(),
                "coverage": "transport_only; caller_local_work_separate"}
