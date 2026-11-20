"""Chunked flow framing, receive-credit pools and directed transport execution.

[Role]
  Frame logical point-to-point flows into bounded physical chunks, track
  receiver-credit residency on the shared Engine, and compute shared vs
  isolated transport lower bounds without mutating topology declarations.

[Invariants]
  * Chunk framing never enlarges a declared chunk to fit a task cap; exceeding
    ``max_physical_chunks`` raises ``transport_chunk_expansion_unbounded``.
  * Receive slots are real Engine pools acquired only after both endpoints are
    ready.

[Timeouts & Locks]
  Bounded chunk count and lazy ``ChunkSequence`` indexing; no threads, sleeps,
  locks or I/O.

[Side-Effect Policy]
  Submits tasks and pools only to the caller-supplied Engine.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

from mindie_sim.schema.common import ModelingError
from ..communication import MessageFlow, RoutedMessage

MS_PER_S = 1000.0
DEFAULT_MAX_PHYSICAL_CHUNKS = 4096
BOTTLENECK_ABS_TOL = 1e-15


def _transport_ready(required, ready_by_rank):
    """Exact participant coverage; a transport never invents a ready event."""
    required, supplied = set(required), set(ready_by_rank)
    if required != supplied:
        missing = ",".join(map(str, sorted(required - supplied)))
        unexpected = ",".join(map(str, sorted(supplied - required)))
        raise ModelingError(
            f"transport_ready_coverage_mismatch:missing={missing}:"
            f"unexpected={unexpected}")
    return {rank: ready_by_rank[rank] for rank in sorted(required)}


def chunk_sizes(payload: float, chunk_bytes: float | None,
                max_physical_chunks: int | None = DEFAULT_MAX_PHYSICAL_CHUNKS):
    """Split a payload into the declared physical chunks, or fail explicitly."""
    if not payload:
        return ()
    if chunk_bytes is None or float(chunk_bytes) >= float(payload):
        return (float(payload),)
    size = float(chunk_bytes)
    count = max(1, math.ceil(float(payload) / size - 1e-12))
    if max_physical_chunks is not None:
        if (isinstance(max_physical_chunks, bool)
                or not isinstance(max_physical_chunks, int)
                or max_physical_chunks < 1):
            raise ModelingError(
                f"chunk_max_physical_chunks_invalid:{max_physical_chunks}")
        if count > max_physical_chunks:
            raise ModelingError(
                "transport_chunk_expansion_unbounded:"
                f"{count}>{max_physical_chunks}")
    remaining, result = float(payload), []
    while remaining > 1e-9:
        current = min(size, remaining)
        result.append(current)
        remaining -= current
    return tuple(result)


def chunk_count(payload: float, chunk_bytes: float | None) -> int:
    """Number of physical frames without materializing their tuple."""
    if not payload:
        return 0
    if chunk_bytes is None or float(chunk_bytes) >= float(payload):
        return 1
    return max(1, math.ceil(float(payload) / float(chunk_bytes) - 1e-12))


def chunk_size_at(payload: float, chunk_bytes: float | None, index: int) -> float:
    """One physical frame size, including the tail, in O(1)."""
    total = float(payload)
    count = chunk_count(total, chunk_bytes)
    if not 0 <= index < count:
        raise IndexError(index)
    if chunk_bytes is None or float(chunk_bytes) >= total:
        return total
    size = float(chunk_bytes)
    if index == count - 1:
        return total - size * (count - 1)
    return size


@dataclass(frozen=True)
class ChunkSequence(Sequence):
    """Lazy physical frames for one logical flow."""

    topology: object = field(repr=False, compare=False)
    flow: MessageFlow
    chunk_bytes: float | None
    max_physical_chunks: int | None = DEFAULT_MAX_PHYSICAL_CHUNKS

    def __len__(self):
        count = chunk_count(self.flow.payload_bytes, self.chunk_bytes)
        if self.max_physical_chunks is not None and count > self.max_physical_chunks:
            raise ModelingError(
                "transport_chunk_expansion_unbounded:"
                f"{count}>{self.max_physical_chunks}")
        return count

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        size = chunk_size_at(
            self.flow.payload_bytes, self.chunk_bytes, index)
        return self.topology.route_message(MessageFlow(
            self.flow.source, self.flow.destination, size,
            self.flow.messages))


@dataclass(frozen=True)
class TransportOptions:
    """One resolved transport option set shared by execution and the ledger."""

    chunk_bytes: float | None = None
    hop_startup_us: float = 0.0
    receive_capacity_bytes: tuple[tuple[int, float], ...] = ()
    receive_buffer_ids: tuple[tuple[int, str], ...] = ()
    receive_capacity_bytes_per_peer: float | None = None
    receive_buffer_id: str | None = None
    max_inflight_per_rank: int | None = None
    max_physical_chunks: int | None = DEFAULT_MAX_PHYSICAL_CHUNKS

    def __post_init__(self):
        if self.chunk_bytes is not None:
            if isinstance(self.chunk_bytes, bool):
                raise ModelingError("transport_chunk_bytes_invalid")
            value = float(self.chunk_bytes)
            if not math.isfinite(value) or value <= 0:
                raise ModelingError(f"transport_chunk_bytes_invalid:{self.chunk_bytes}")
        if (not isinstance(self.hop_startup_us, (int, float))
                or isinstance(self.hop_startup_us, bool)
                or not math.isfinite(float(self.hop_startup_us))
                or float(self.hop_startup_us) < 0):
            raise ModelingError(
                f"transport_hop_startup_invalid:{self.hop_startup_us}")
        if self.receive_capacity_bytes_per_peer is not None:
            if isinstance(self.receive_capacity_bytes_per_peer, bool):
                raise ModelingError(
                    "direct_peer_receive_capacity_requires_positive_number")
            value = float(self.receive_capacity_bytes_per_peer)
            if not math.isfinite(value) or value <= 0:
                raise ModelingError(
                    "direct_peer_receive_capacity_requires_positive_number")
        if self.receive_buffer_id is not None and not str(
                self.receive_buffer_id):
            raise ModelingError("transport_receive_buffer_id_invalid")
        if self.max_inflight_per_rank is not None:
            if (isinstance(self.max_inflight_per_rank, bool)
                    or not isinstance(self.max_inflight_per_rank, int)
                    or self.max_inflight_per_rank < 1):
                raise ModelingError(
                    "direct_peer_window_requires_positive_integer")
        if self.max_physical_chunks is not None and (
                isinstance(self.max_physical_chunks, bool)
                or not isinstance(self.max_physical_chunks, int)
                or self.max_physical_chunks < 1):
            raise ModelingError(
                f"transport_max_physical_chunks_invalid:{self.max_physical_chunks}")

    @classmethod
    def from_mapping(cls, raw):
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        if not isinstance(raw, Mapping):
            raise ModelingError("transport_options_must_be_mapping")
        allowed = {"chunk_bytes", "hop_startup_us", "receive_capacity_bytes",
                   "receive_buffer_ids", "receive_capacity_bytes_per_peer",
                   "receive_buffer_id", "max_physical_chunks",
                   "max_inflight_per_rank"}
        unknown = set(raw) - allowed
        if unknown:
            raise ModelingError(
                f"transport_option_unknown:{sorted(unknown)[0]}")
        capacity = raw.get("receive_capacity_bytes", ())
        if isinstance(capacity, Mapping):
            capacity = tuple(sorted((int(rank), float(value))
                                    for rank, value in capacity.items()))
        else:
            capacity = tuple(sorted((int(rank), float(value))
                                    for rank, value in capacity))
        buffers = raw.get("receive_buffer_ids", ())
        if isinstance(buffers, Mapping):
            buffers = tuple(sorted((int(rank), str(value))
                                   for rank, value in buffers.items()))
        else:
            buffers = tuple(sorted((int(rank), str(value))
                                   for rank, value in buffers))
        return cls(
            chunk_bytes=raw.get("chunk_bytes"),
            hop_startup_us=raw.get("hop_startup_us", 0.0),
            receive_capacity_bytes=capacity,
            receive_buffer_ids=buffers,
            receive_capacity_bytes_per_peer=raw.get(
                "receive_capacity_bytes_per_peer"),
            receive_buffer_id=raw.get("receive_buffer_id"),
            max_inflight_per_rank=raw.get("max_inflight_per_rank"),
            max_physical_chunks=raw.get(
                "max_physical_chunks", DEFAULT_MAX_PHYSICAL_CHUNKS),
        )

    def plan_kwargs(self):
        result = {"max_physical_chunks": self.max_physical_chunks}
        if self.chunk_bytes is not None:
            result["chunk_bytes"] = self.chunk_bytes
        if self.hop_startup_us:
            result["hop_startup_us"] = self.hop_startup_us
        if self.receive_capacity_bytes:
            result["receive_capacity_bytes"] = dict(self.receive_capacity_bytes)
        if self.receive_buffer_ids:
            result["receive_buffer_ids"] = dict(self.receive_buffer_ids)
        if self.receive_capacity_bytes_per_peer is not None:
            result["receive_capacity_bytes_per_peer"] = (
                self.receive_capacity_bytes_per_peer)
        if self.receive_buffer_id is not None:
            result["receive_buffer_id"] = self.receive_buffer_id
        result["max_inflight_per_rank"] = self.max_inflight_per_rank
        return result


@dataclass(frozen=True)
class FlowPath:
    """One logical message and its already-framed transfer chunks."""

    flow: MessageFlow
    chunks: Sequence[RoutedMessage]

    @property
    def useful_bytes(self) -> float:
        return float(self.flow.useful_bytes)

    @property
    def hop_count(self) -> int:
        return len(self.chunks[0].hops) if self.chunks else 0


class TransportRun:
    """Live completion/credit state for one :class:`TransportPlan` start."""

    def __init__(self, plan, completions, stats, started_ms):
        self.plan = plan
        self.completions = tuple(completions)
        self._stats = stats
        self.started_ms = started_ms

    def report(self) -> dict:
        stats = self._stats
        times = {path.flow: event.time for path, event in
                 zip(self.plan.flows, self.completions)}
        complete = [value for value in times.values() if value is not None]
        span_ms = max(complete) if complete else self.started_ms
        duration = max(0.0, span_ms - self.started_ms)
        bound = self.plan.lower_bound()
        resident = stats["resident"]
        peak = stats["peak_resident"]
        byte_us = dict(stats["resident_byte_us"])
        residency_end = max(
            [span_ms, *stats["last_resident_change"].values()])
        residency_window = max(0.0, residency_end - self.started_ms)
        for destination, amount in resident.items():
            byte_us[destination] = byte_us.get(destination, 0.0) + max(
                0.0, span_ms - stats["last_resident_change"].get(
                    destination, self.started_ms)) * MS_PER_S * amount
        return {
            "boundary": "directed_flow_transport",
            "flows": [
                {
                    "source": path.flow.source,
                    "destination": path.flow.destination,
                    "useful_bytes": path.flow.useful_bytes,
                    "chunks": len(path.chunks),
                    "hops": path.hop_count,
                    "complete_ms": event.time,
                }
                for path, event in zip(self.plan.flows, self.completions)
            ],
            "span_ms": span_ms,
            "duration_ms": duration,
            "started_ms": self.started_ms,
            "service_lower_bound_ms": bound["service_ms"],
            "per_flow_isolated_service_ms": bound["per_flow_isolated_ms"],
            "path_startup_us": bound["path_startup_us"],
            "hop_startup_boundary": (
                "first_chunk_each_hop_once" if self.plan.hop_startup_us
                else "not_declared"),
            "shared_resource_work_bytes": bound["shared_resource_work_bytes"],
            "endpoint_bottleneck_work_bytes": bound["endpoint_bottleneck_work_bytes"],
            "bottleneck_resource_work_bytes": bound[
                "bottleneck_resource_work_bytes"],
            "capacity_work_bytes": bound["capacity_work_bytes"],
            "bottleneck_resources": bound["bottleneck_resources"],
            "chunks": stats["chunks"],
            "active_chunks_at_end": stats.get("active_chunks", 0),
            "peak_active_chunks": stats.get("peak_active_chunks", 0),
            "peak_active_chunks_by_flow": {
                int(index): value for index, value in
                stats.get("peak_active_by_flow", {}).items()},
            "credit_waits": stats["credit_waits"],
            "receiver_capacity_bytes": dict(self.plan.receive_capacity),
            "receiver_buffer_ids": {
                int(rank): value for rank, value in self.plan.receive_buffer_ids},
            "receiver_reserved_peak_bytes": {
                int(key): value for key, value in stats["peak_granted"].items()},
            "receiver_resident_peak_bytes": {
                int(key): value for key, value in peak.items()},
            "receiver_resident_bytes_at_end": {
                int(key): value for key, value in resident.items()},
            "receiver_resident_byte_us": {
                int(key): value for key, value in byte_us.items()},
            "residency_window_ms": residency_window,
            "receiver_mean_resident_bytes": ({
                str(dest): (value / (residency_window * MS_PER_S)
                            if residency_window else 0.0)
                for dest, value in byte_us.items()}),
            "credit_wait_ms": {
                int(key): value for key, value in stats["credit_wait_ms"].items()},
            "service_vs_span": "lower_bound_excludes_credit_waits_and_endpoint_pipeline",
            "unknowns": dict(self.plan.unknowns),
        }


@dataclass(frozen=True)
class TransportPlan:
    """Chunk-pipelined directed flows over one declared physical topology."""

    topology: object = field(repr=False, compare=False)
    flows: tuple[FlowPath, ...]
    chunk_bytes: float | None = None
    max_physical_chunks: int | None = DEFAULT_MAX_PHYSICAL_CHUNKS
    receive_capacity: tuple[tuple[int, float], ...] = ()
    receive_buffer_ids: tuple[tuple[int, str], ...] = ()
    hop_startup_us: float = 0.0
    unknowns: tuple[tuple[str, str], ...] = ()

    @property
    def ranks(self):
        return tuple(sorted({path.flow.source for path in self.flows}
                            | {path.flow.destination for path in self.flows}))

    def resources(self):
        """Topology links/ports; endpoint_rates already bind these resources."""
        return tuple(self.topology.resources())

    def _capacity(self, destination: int):
        for rank, value in self.receive_capacity:
            if rank == destination:
                return value
        return None

    def _buffer_id(self, destination: int) -> str:
        for rank, value in self.receive_buffer_ids:
            if rank == destination:
                return value
        return f"{self.topology.namespace}rank:{destination}"

    def _port_stages(self, path: FlowPath) -> int:
        """Source injection + link hops + destination receipt for one path."""
        if not path.chunks:
            return 0
        source, destination = path.flow.source, path.flow.destination
        names = {
            name
            for hop in path.chunks[0].hops
            for name in (hop.resource, *hop.additional_resources)
        }
        return (path.hop_count
                + int(any(name.endswith(f":tx:port{source}") for name in names))
                + int(any(name.endswith(f":rx:port{destination}") for name in names)))

    @property
    def stage_count(self) -> int:
        """Maximum route-stage count across flows."""
        return max((self._port_stages(path) for path in self.flows), default=0)

    def resource_work(self):
        """Return useful, wire and transaction totals per capacity resource."""
        useful, wire, transactions = defaultdict(float), defaultdict(float), defaultdict(int)
        frames = {}
        for path in self.flows:
            for chunk in path.chunks:
                for hop in chunk.hops:
                    useful[hop.resource] += hop.useful_bytes
                    wire[hop.resource] += hop.wire_bytes
                    for name in hop.additional_resources:
                        useful[name] += hop.useful_bytes
                        wire[name] += hop.wire_bytes
                    if hop.transactions is not None:
                        transactions[hop.resource] += hop.transactions
                    if hop.transaction_frame is not None:
                        frames[hop.resource] = hop.transaction_frame
        return {
            "useful": dict(useful), "wire": dict(wire),
            "transactions": dict(transactions), "frames": frames,
        }

    def lower_bound(self):
        """Aggregate shared-resource bound and per-flow isolated bounds."""
        rates = dict(self.topology._resource_rates or {
            resource.name: resource.rate for resource in self.topology.resources()})
        work = self.resource_work()
        wire = work["wire"]
        service = max((amount / rates[name] for name, amount in wire.items()
                       if name in rates), default=0.0)
        bottleneck = tuple(sorted(
            name for name, amount in wire.items()
            if name in rates and math.isclose(
                amount / rates[name], service,
                rel_tol=1e-12, abs_tol=BOTTLENECK_ABS_TOL)))
        per_flow = []
        for path in self.flows:
            flow_wire = defaultdict(float)
            for chunk in path.chunks:
                for hop in chunk.hops:
                    flow_wire[hop.resource] += hop.wire_bytes
                    for name in hop.additional_resources:
                        flow_wire[name] += hop.wire_bytes
            isolated = max((amount / rates[name]
                            for name, amount in flow_wire.items() if name in rates),
                           default=0.0)
            per_flow.append({
                "source": path.flow.source,
                "destination": path.flow.destination,
                "isolated_service_ms": isolated,
                "service_bytes": sum(flow_wire.values()),
                "hops": path.hop_count,
            })
        path_startup = max(
            (self._port_stages(path) * float(self.hop_startup_us)
             for path in self.flows),
            default=0.0)
        endpoint_names = {
            name for name in bottleneck
            if ":tx:port" in name or ":rx:port" in name or "endpoint:" in name
        }
        link_bottlenecks = tuple(
            name for name in bottleneck if name not in endpoint_names)
        return {
            "service_ms": service,
            "bottleneck_resources": bottleneck,
            "bottleneck_resource_work_bytes": {
                name: wire[name] for name in bottleneck},
            "shared_resource_work_bytes": sum(
                wire[name] for name in
                (link_bottlenecks if link_bottlenecks else bottleneck)),
            "endpoint_bottleneck_work_bytes": sum(
                wire[name] for name in endpoint_names),
            "capacity_work_bytes": sum(wire.values()),
            "per_flow_isolated_ms": per_flow,
            "path_startup_us": path_startup,
            "boundary": "aggregate_resource_work_excludes_startup_and_waits",
        }

    def _flow_ports(self, path: FlowPath):
        if not path.chunks:
            return False, False
        names = {
            name
            for hop in path.chunks[0].hops
            for name in (hop.resource, *hop.additional_resources)
        }
        source, destination = path.flow.source, path.flow.destination
        return (any(name.endswith(f":tx:port{source}") for name in names),
                any(name.endswith(f":rx:port{destination}") for name in names))

    def _ensure_receive_pool(self, engine, destination: int):
        for rank, capacity in self.receive_capacity:
            if rank != destination:
                continue
            if capacity <= 0:
                raise ModelingError(
                    f"transport_receive_capacity_zero:{destination}")
            pool = f"transport:rx:{self._buffer_id(destination)}"
            if pool in engine.pools:
                if not math.isclose(engine.pools[pool], capacity,
                                    rel_tol=1e-12, abs_tol=1e-9):
                    raise ModelingError(
                        f"transport_receive_capacity_conflict:{destination}:"
                        f"{engine.pools[pool]}!={capacity}")
            else:
                engine.extend(pools={pool: capacity})
            return pool
        return None

    def start(self, engine, name, ready_by_rank, *, release_events=None,
              owner="", start_ms=None):
        """Submit physical chunks with per-ready admission and Engine credits."""
        from ..engine import Allocation

        required = self.ranks
        initial = _transport_ready(required, ready_by_rank)
        releases = {}
        for rank, value in (release_events or {}).items():
            releases[int(rank)] = (tuple(value)
                                   if isinstance(value, (tuple, list))
                                   else (value,))
        completions = [engine.future(f"{name}:flow{index}:complete")
                       for index in range(len(self.flows))]
        stats = {
            "chunks": 0, "credit_waits": 0,
            "credit_wait_ms": defaultdict(float),
            "resident": defaultdict(float),
            "peak_resident": defaultdict(float),
            "peak_granted": defaultdict(float),
            "resident_byte_us": defaultdict(float),
            "last_resident_change": {},
        }
        started = (min((initial[rank].time or 0.0 for rank in required),
                       default=0.0) if start_ms is None else float(start_ms))
        for path in self.flows:
            stats["last_resident_change"].setdefault(path.flow.destination, started)
        pools = {destination: self._ensure_receive_pool(engine, destination)
                 for destination, _ in self.receive_capacity}
        expected = [len(path.chunks) for path in self.flows]
        arrived_count = [0] * len(self.flows)
        completed = [False] * len(self.flows)
        service_events = [{} for _ in self.flows]
        release_cursor = defaultdict(int)
        issued = [0] * len(self.flows)
        active = [0] * len(self.flows)
        stats["active_chunks"] = 0
        stats["peak_active_chunks"] = 0
        stats["peak_active_by_flow"] = {}
        window = []
        for path in self.flows:
            capacity = self._capacity(path.flow.destination)
            if capacity is None:
                window.append(max(1, path.hop_count))
            else:
                largest = min(
                    float(self.chunk_bytes or path.flow.payload_bytes),
                    float(path.flow.payload_bytes))
                window.append(max(1, int(float(capacity) // largest + 1e-9)))

        def note_residency(destination, now):
            last = stats["last_resident_change"].get(destination, started)
            delta = max(0.0, float(now) - float(last))
            stats["resident_byte_us"][destination] += (
                delta * MS_PER_S * stats["resident"][destination])
            stats["last_resident_change"][destination] = now
            current = stats["resident"][destination]
            if current > stats["peak_resident"][destination]:
                stats["peak_resident"][destination] = current

        def add_resident(destination, useful, now):
            note_residency(destination, now)
            stats["resident"][destination] += useful
            note_residency(destination, now)

        def remove_resident(destination, useful, now):
            note_residency(destination, now)
            stats["resident"][destination] = max(
                0.0, stats["resident"][destination] - useful)
            note_residency(destination, now)

        def complete_flow(flow_index):
            if completed[flow_index]:
                return
            completed[flow_index] = True
            event = completions[flow_index]
            if not event.done:
                engine.resolve(event)

        def arrived(flow_index, destination, useful, prefix, allocation,
                    received_event):
            add_resident(destination, useful, engine.now)
            arrived_count[flow_index] += 1
            active[flow_index] -= 1
            stats["active_chunks"] -= 1
            schedule_release(prefix, destination, useful, received_event,
                             allocation)
            pump(flow_index)
            if arrived_count[flow_index] == expected[flow_index]:
                complete_flow(flow_index)

        def track_grant(destination, source_event, destination_event):
            pool = pools.get(destination)
            if pool is None:
                return
            granted = engine.allocated.get(pool, 0.0)
            if granted > stats["peak_granted"][destination]:
                stats["peak_granted"][destination] = granted
            ready_at = max(source_event.time or 0.0,
                           destination_event.time or 0.0)
            wait = max(0.0, engine.now - ready_at)
            if wait > 1e-12:
                stats["credit_waits"] += 1
                if wait > stats["credit_wait_ms"][destination]:
                    stats["credit_wait_ms"][destination] = wait

        def schedule_release(prefix, destination, useful, arrival, allocation):
            events = releases.get(destination, ())
            explicit = destination in releases
            cursor = release_cursor[destination]
            while cursor < len(events) and events[cursor].done:
                cursor += 1
            release_cursor[destination] = cursor
            if explicit and cursor >= len(events):
                return
            after = [arrival]
            if cursor < len(events):
                after.append(events[cursor])
            callback = (
                (lambda _event, d=destination, u=useful:
                 remove_resident(d, u, engine.now)))
            if allocation is not None:
                engine.submit(
                    prefix + ":release", after=tuple(after),
                    free=(allocation,), owner=owner, on_complete=callback)
            else:
                engine.submit(
                    prefix + ":release", after=tuple(after),
                    owner=owner, on_complete=callback)

        def issue_chunk(flow_index, chunk_index):
            path = self.flows[flow_index]
            chunk = path.chunks[chunk_index]
            destination = path.flow.destination
            useful = float(chunk.flow.useful_bytes)
            prefix = f"{name}:flow{flow_index}:chunk{chunk_index}"
            previous_services = service_events[flow_index].get(
                chunk_index - 1) if chunk_index else None
            dependencies = [initial[path.flow.source], initial[destination]]
            if previous_services is not None:
                dependencies.append(previous_services[0])
            allocation = None
            pool = pools.get(destination)
            if pool is not None:
                allocation = Allocation(
                    prefix + ":credit", pool, useful, (0,))
                grant = engine.submit(
                    prefix + ":grant", after=tuple(dependencies),
                    acquire=(allocation,), owner=owner,
                    on_complete=lambda _event, d=destination,
                    s=initial[path.flow.source], r=initial[destination]:
                        track_grant(d, s, r))
                dependencies = [grant]
            else:
                dependencies = [engine.submit(
                    prefix + ":gate", after=tuple(dependencies), owner=owner)]
            has_tx, has_rx = self._flow_ports(path)
            previous = dependencies[0]
            chunk_service = [None] * len(chunk.hops)
            for hop_index, hop in enumerate(chunk.hops):
                hop_dependencies = [previous]
                if previous_services is not None:
                    hop_dependencies.append(previous_services[hop_index])
                startup = float(self.hop_startup_us)
                if chunk_index == 0 and startup:
                    if has_tx and hop_index == 0:
                        gate = engine.submit(
                            prefix + ":inject:startup",
                            after=tuple(hop_dependencies),
                            latency=startup / MS_PER_S, owner=owner,
                            metadata={"startup_boundary": "source_injection_once"})
                        hop_dependencies = [gate]
                    gate = engine.submit(
                        prefix + f":hop{hop_index}:startup",
                        after=tuple(hop_dependencies),
                        latency=startup / MS_PER_S, owner=owner,
                        metadata={"startup_boundary": "per_hop_protocol_once"})
                    hop_dependencies = [gate]
                service = engine.submit(
                    prefix + f":hop{hop_index}:service", hop.demand,
                    after=tuple(hop_dependencies), owner=owner, coupled=True,
                    on_complete=(
                        (lambda _event, fi=flow_index: pump(fi))
                        if hop_index == 0
                        and chunk_index + 1 < expected[flow_index]
                        else None))
                arrival = engine.submit(
                    prefix + f":hop{hop_index}:arrival", after=(service,),
                    latency=hop.latency_ms, owner=owner)
                chunk_service[hop_index] = service
                previous = arrival
            service_events[flow_index][chunk_index] = tuple(chunk_service)
            if chunk_index >= 2:
                service_events[flow_index].pop(chunk_index - 2, None)
            if chunk_index == 0 and float(self.hop_startup_us) and has_rx:
                previous = engine.submit(
                    prefix + ":receive:startup", after=(previous,),
                    latency=float(self.hop_startup_us) / MS_PER_S, owner=owner,
                    metadata={"startup_boundary": "destination_receipt_once"})
            engine.submit(
                prefix + ":received", after=(previous,), owner=owner,
                on_complete=lambda event, fi=flow_index, d=destination,
                value=useful, p=prefix, a=allocation:
                    arrived(fi, d, value, p, a, event))
            active[flow_index] += 1
            stats["active_chunks"] += 1
            if stats["active_chunks"] > stats["peak_active_chunks"]:
                stats["peak_active_chunks"] = stats["active_chunks"]
            by_flow = stats["peak_active_by_flow"]
            by_flow[flow_index] = max(by_flow.get(flow_index, 0),
                                      active[flow_index])
            stats["chunks"] += 1

        def pump(flow_index):
            if expected[flow_index] == 0:
                return
            limit = min(expected[flow_index],
                        arrived_count[flow_index] + window[flow_index] + 1)
            while issued[flow_index] < limit:
                chunk_index = issued[flow_index]
                issued[flow_index] += 1
                issue_chunk(flow_index, chunk_index)

        for flow_index, path in enumerate(self.flows):
            if not path.chunks:
                engine.submit(
                    f"{name}:flow{flow_index}:empty",
                    after=(initial[path.flow.source],
                           initial[path.flow.destination]),
                    owner=owner,
                    on_complete=lambda _event, fi=flow_index: complete_flow(fi))
                continue
            pump(flow_index)

        return TransportRun(self, completions, stats, started)
