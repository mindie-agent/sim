"""Route communication through actual shared physical links.

[Role] Map collective, peer and point-to-point flows to declared link, port
  and endpoint service, including a bounded chunk/credit transport schedule.
[Invariants] A route must connect both endpoints and obey the declared
  die/package/node scope. Shared fabric capacity is shared by every flow;
  full-duplex ports charge source Tx and destination Rx; per-pair links own
  independent directed capacities. Logical message bytes and
  transaction-rounded physical wire bytes remain distinct. A multi-hop flow is
  framed at every hop, never payload service multiplied by hop count. An
  aggregate resource lower bound is reported separately from the Engine span.
  Chunk pipeline, receive credits/residency and one declared per-hop startup
  are actual tasks; missing parameters stay explicit unknowns. Joint plans
  cover complete physical groups, account each rank once and never merge
  unrelated groups.
[Timeouts & Locks] Finite rank graph, bounded route cache and bounded chunk
  count; no network I/O.
[Side-Effect Policy] Only caches paths and submits caller-owned Engine tasks.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math

from mindie_sim.schema.common import ModelingError
from .communication import CommunicationWork, MessageFlow, RoutedHop, RoutedMessage
from .engine import Resource
from .peer_traffic import PeerTraffic


_PORT_SEMANTICS = {"per_port", "fully_connected", "mesh"}
# A cross-node fabric is not an on-node hop, even when its member list
# happens to name every rank of one node.  "inter" alone is not enough:
# an intra-package link can mention dies without being scale-out.
_CROSS_NODE_MARKERS = (
    "inter_node", "inter_pod", "roce", "rdma", "scaleout", "scale_out",
)
# Package/node markers are matched only after the cross-node markers: the
# substring "node" also occurs in "inter_node".  A link without a marker keeps
# its historical "unknown" scope and may connect any declared members.
_PACKAGE_MARKERS = ("intra_package", "package", "die_to_die")
_NODE_MARKERS = ("intra_node", "node_mesh")


def _link_text(link) -> str:
    return (f"{getattr(link, 'channel', '')} {getattr(link, 'name', '')}"
            .lower().replace("-", "_"))


def _is_cross_node_link(link) -> bool:
    text = _link_text(link)
    return any(marker in text for marker in _CROSS_NODE_MARKERS)


def _link_scope(link) -> str:
    """Classify a declared link by its own name, never by a rank-count guess.

    ``inter_node``/RoCE/superpod links are scale-out links even when their
    member list happens to contain every rank of one node.  Package and node
    markers order the on-node hierarchy.  An unmarked link is ``unknown`` and
    remains routable; the deployment owns that ambiguity.
    """
    text = _link_text(link)
    if any(marker in text for marker in _CROSS_NODE_MARKERS):
        return "cross_node"
    if any(marker in text for marker in _PACKAGE_MARKERS):
        return "package"
    if any(marker in text for marker in _NODE_MARKERS):
        return "node"
    return "unknown"


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
                max_physical_chunks: int | None = 4096):
    """Split a payload into the declared physical chunks, or fail explicitly.

    A chunk boundary is a physical framing/credit boundary.  This function
    never enlarges a declared chunk to fit a task budget: rounding,
    transactions and receive-credit units must stay those of the declared
    chunk.  The hard cap only prevents a small chunk on a huge payload from
    materializing an unbounded task graph; exceeding it is an explicit
    ``transport_chunk_expansion_unbounded`` report, not silent aggregation.
    """
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
    """Lazy physical frames for one logical flow.

    The plan keeps only the payload/chunk recipe; a frame and its routed hops
    are built on demand.  Iterating a report or lower bound sees every frame,
    but live execution never stores the whole history.
    """

    topology: object = field(repr=False, compare=False)
    flow: MessageFlow
    chunk_bytes: float | None
    max_physical_chunks: int | None = 4096

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
    """One resolved transport option set shared by execution and the ledger.

    T04 resolves this once from the selected recipe (``collective_options``) and
    passes the same object to ``Topology.joint_plan`` and
    ``Topology.accounting``.  SIM-STRUCTURE must not re-read a different
    metadata source for the same invocation.
    """

    chunk_bytes: float | None = None
    hop_startup_us: float = 0.0
    receive_capacity_bytes: tuple[tuple[int, float], ...] = ()
    receive_buffer_ids: tuple[tuple[int, str], ...] = ()
    receive_capacity_bytes_per_peer: float | None = None
    receive_buffer_id: str | None = None
    max_inflight_per_rank: int | None = None
    max_physical_chunks: int | None = 4096

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
            max_physical_chunks=raw.get("max_physical_chunks", 4096),
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
        # Preserve explicit None (remote-peer-count) rather than letting a
        # plan entry default it to 1.
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
        # Carry the last resident segment to the observed completion.  A
        # consumer that never fires is visible as bytes still resident.
        residency_end = max(
            [span_ms, *stats["last_resident_change"].values()])
        residency_window = max(0.0, residency_end - self.started_ms)
        for destination, amount in resident.items():
            byte_us[destination] = byte_us.get(destination, 0.0) + max(
                0.0, span_ms - stats["last_resident_change"].get(
                    destination, self.started_ms)) * 1000.0 * amount
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
                str(dest): (value / (residency_window * 1000.0)
                            if residency_window else 0.0)
                for dest, value in byte_us.items()}),
            "credit_wait_ms": {
                int(key): value for key, value in stats["credit_wait_ms"].items()},
            "service_vs_span": "lower_bound_excludes_credit_waits_and_endpoint_pipeline",
            "unknowns": dict(self.plan.unknowns),
        }


@dataclass(frozen=True)
class TransportPlan:
    """Chunk-pipelined directed flows over one declared physical topology.

    The plan keeps three measurements separate:

    * aggregate per-resource work gives a shared-link service lower bound;
    * per-flow isolated service is the widest single-resource bound for that
      flow, not a sum over its hops;
    * the Engine ``start`` span includes the actual chunk pipeline, receiver
      credits/residency and one declared protocol startup per hop.
    """

    topology: object = field(repr=False, compare=False)
    flows: tuple[FlowPath, ...]
    chunk_bytes: float | None = None
    max_physical_chunks: int | None = 4096
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
                amount / rates[name], service, rel_tol=1e-12, abs_tol=1e-15)))
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
            # A shared forward plane and an endpoint port can both bind the
            # same span.  Report the forward-plane work separately; summing
            # both made H4's heterogeneous 2400 B forward look like 3600 B.
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
        """Submit physical chunks with per-ready admission and Engine credits.

        A finite receive slot is an Engine pool.  A chunk acquires its
        allocation only after both source and destination are ready, so a
        late-ready flow cannot reserve capacity from a ready flow, and two
        plans on the same Engine share the same destination pool.  Arrival and
        release update residency at the event time in B*us; immediate release
        therefore reports zero resident area.
        """
        from .engine import Allocation

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
                # Without a declared receive slot, keep one chunk per path
                # stage active (plus one look-ahead) so a multi-hop pipeline
                # stays full without materializing later chunks.
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
                delta * 1000.0 * stats["resident"][destination])
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
                # The consumer sequence is exhausted.  Past drain events do
                # not authorize future arrivals: keep the allocation and
                # resident bytes until a new consumer is declared.
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
                            latency=startup / 1000.0, owner=owner,
                            metadata={"startup_boundary": "source_injection_once"})
                        hop_dependencies = [gate]
                    gate = engine.submit(
                        prefix + f":hop{hop_index}:startup",
                        after=tuple(hop_dependencies),
                        latency=startup / 1000.0, owner=owner,
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
                    latency=float(self.hop_startup_us) / 1000.0, owner=owner,
                    metadata={"startup_boundary": "destination_receipt_once"})
            received = engine.submit(
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
            # The first-hop service callback above issues the next chunk only
            # up to the current receiver window; no historical chunk is kept.

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


STARTUP_PEER_PHASE_DELTA = 1
STARTUP_ONE = 1
STARTUP_ZERO_FLOAT = 0.0
DEFAULT_MAX_PHYSICAL_CHUNKS = 4096
STARTUP_ZERO = 0


class Topology:
    def __init__(self, machine, parallel, namespace="", *,
                 collective_startup_mode="legacy"):
        self.machine, self.parallel = machine, parallel
        self.namespace = namespace
        self.collective_startup_mode = collective_startup_mode
        self.collective_protocols = dict(
            getattr(machine, "collective_protocols", ()) or ())
        self.links = {link.name: link for link in machine.links}
        self.routes = {}
        width = max(1, int(getattr(machine, "ranks_per_node", 1) or 1))
        self._ranks_per_node = width
        self._dies_per_package = max(
            1, int(getattr(machine, "dies_per_package", 1) or 1))
        # Placement is a fact from each device record.  Rank 0 is not a
        # representative for the first device in every declaration.
        self._rank_node = {}
        self._rank_package = {}
        self._rank_device = {}
        for index, die in enumerate(machine.dies):
            rank = int(die.rank)
            self._rank_device[rank] = index
            self._rank_node[rank] = int(getattr(
                die, "node", rank // width))
            self._rank_package[rank] = int(getattr(
                die, "package", index // self._dies_per_package))
        self._link_scope = {name: _link_scope(link)
                            for name, link in self.links.items()}
        self._resource_cache = None
        self._resource_rates = None
        # Exact resolved plan options keyed by the caller's op identity.  This
        # keeps execution and the public ledger on the same binding without a
        # second IR or a metadata re-parse.  Explicit transport arguments take
        # precedence; T04/SIM-STRUCTURE can also pass the object directly.
        self._transport_bindings = {}

    def startup_descriptor(self, op, group_size):
        """Resolve this collective's declared startup model once."""
        from .collective_startup import resolve_collective_startup

        metadata = dict(op.metadata)
        family = str(metadata.get("collective", op.family)).strip().lower()
        return resolve_collective_startup(
            family=family, group_size=int(group_size), metadata=metadata,
            runtime_mode=self.collective_startup_mode,
            topology_protocols=self.collective_protocols)

    def location(self, rank: int) -> dict:
        """Return the declared die/package/node placement of one rank."""
        key = int(rank)
        if key not in self._rank_node:
            raise ModelingError(f"physical_rank_unknown:{rank}")
        return {"rank": key, "device": self._rank_device.get(key, key),
                "package": self._rank_package[key],
                "node": self._rank_node[key]}

    def link_scope(self, name: str) -> str:
        return self._link_scope.get(name, "unknown")

    def _endpoint_rate(self, link, rank: int) -> float:
        """One fact source: Link.endpoint_rates is already bytes/ms.

        The schema normalizes the declaration once.  A missing rank keeps the
        link's shared/port rate; a declared rank owns its tx and rx port rate
        independently of the link's shared forward rate.
        """
        value = link.endpoint_capacity_bytes_per_ms(int(rank))
        return (float(value) if value is not None
                else float(link.capacity_bytes_per_ms))

    def _endpoint_resource(self, link, direction: str, rank: int):
        return Resource(
            f"{self.namespace}link:{link.name}:{direction}:port{rank}",
            self._endpoint_rate(link, rank), "bytes")

    def _has_endpoint_rate(self, link, rank: int) -> bool:
        return link.endpoint_capacity_bytes_per_ms(int(rank)) is not None

    def resources(self):
        if self._resource_cache is not None:
            return self._resource_cache
        result = []
        for link in self.links.values():
            endpoint_ranks = sorted({int(rank) for rank, _ in link.endpoint_rates})
            if link.semantics == "per_pair":
                result.extend(Resource(
                    f"{self.namespace}link:{link.name}:pair{source}>{destination}",
                    link.capacity_bytes_per_ms, "bytes",
                ) for source in link.members for destination in link.members if source != destination)
                result.extend(self._endpoint_resource(link, direction, rank)
                              for rank in endpoint_ranks
                              for direction in ("tx", "rx"))
            elif link.semantics in _PORT_SEMANTICS:
                # Per-port semantics already expose every placement's tx/rx;
                # endpoint_rates changes those resources, never a second copy.
                result.extend(self._endpoint_resource(link, direction, rank)
                              for rank in link.members
                              for direction in ("tx", "rx"))
            else:
                result.append(Resource(
                    f"{self.namespace}link:{link.name}",
                    link.capacity_bytes_per_ms, "bytes"))
                result.extend(self._endpoint_resource(link, direction, rank)
                              for rank in endpoint_ranks
                              for direction in ("tx", "rx"))
            if link.switch_capacity_bytes_per_ms is not None:
                result.append(Resource(f"{self.namespace}link:{link.name}:switch_plane",
                                       link.switch_capacity_bytes_per_ms, "bytes"))
        self._resource_cache = tuple(result)
        self._resource_rates = {resource.name: resource.rate
                                for resource in self._resource_cache}
        return self._resource_cache

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
                scope = self._link_scope.get(link.name, "unknown")
                for neighbor in link.members:
                    if neighbor in seen:
                        continue
                    if not self._scope_admits(scope, rank, neighbor):
                        continue
                    seen.add(neighbor)
                    queue.append((neighbor, path + ((link.name, rank, neighbor),)))
        raise ValueError(f"physical_route_missing:{source}->{destination}")

    def _scope_admits(self, scope: str, left: int, right: int) -> bool:
        if scope == "package":
            return self._same_package(left, right)
        if scope == "node":
            return self._same_node(left, right)
        if scope == "cross_node":
            # A scale-out fabric is not an on-node hop.  Its members list may
            # still name every rank of one node.
            return not self._same_node(left, right)
        return True

    def _same_node(self, left: int, right: int) -> bool:
        if left in self._rank_node and right in self._rank_node:
            return self._rank_node[left] == self._rank_node[right]
        width = self._ranks_per_node
        return int(left) // width == int(right) // width

    def _same_package(self, left: int, right: int) -> bool:
        if left in self._rank_package and right in self._rank_package:
            return self._rank_package[left] == self._rank_package[right]
        width = self._dies_per_package
        return int(left) // width == int(right) // width

    def ranks(self, replica, stage):
        p = self.parallel
        width = p.tp * p.pcp
        start = (replica * p.pp + stage) * width
        return tuple(device.rank for device in self.machine.dies[start:start + width])

    def groups(self, axis, replica, stage, *, metadata=None):
        """Rank axes are independent; EP absorbs PCP copies only when it spans them."""
        p = self.parallel
        sources = self.ranks(replica, stage)
        if axis == "local_tp":
            from .local_tp_groups import local_tp_groups

            declared = dict(metadata or {})
            width = int(declared.get("local_tp_width", STARTUP_ZERO) or STARTUP_ZERO)
            if width <= STARTUP_ZERO:
                raise ModelingError("local_tp_group_requires_declared_width")
            all_groups = local_tp_groups(self.machine, p, stage=stage, width=width)
            selected = tuple(group for group in all_groups
                             if any(rank in sources for rank in group))
            return selected or all_groups
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
            if p.ep_spans_dp:
                if p.ep_spans_pcp:
                    count = p.ep_dp_width
                    first = replica // count * count
                    return (tuple(rank for r in range(first, first + count)
                                  for rank in self.ranks(r, stage)),)
                count = p.ep // p.tp
                first = replica // count * count
                return tuple(
                    tuple(rank for r in range(first, first + count)
                          for rank in self.ranks(r, stage)[copy * p.tp:(copy + 1) * p.tp])
                    for copy in range(p.pcp)
                )
            if p.ep_spans_pcp:
                sources = self.ranks(replica, stage)
                return tuple(sources[i:i + p.ep] for i in range(0, len(sources), p.ep))
            return tuple(sources[i:i + p.ep] for i in range(0, len(sources), p.ep))
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
        alltoall = str(family).strip().lower() in {"alltoall", "alltoallv"}
        if alltoall:
            # Validate explicit steps before any zero-volume early return.
            for group in self.groups(axis, replica, source_stage, metadata=metadata):
                self.startup_descriptor(op, len(group))
        if "peer_traffic" in metadata:
            if family not in {"alltoall", "alltoallv"}:
                raise ModelingError(
                    "peer_traffic_requires_peer_collective")
            groups = tuple(self.groups(axis, replica, source_stage, metadata=metadata))
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
                rounds = self.startup_descriptor(op, len(group)).startup_rounds
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
        for group in self.groups(axis, replica, source_stage, metadata=metadata):
            if len(group) <= 1:
                continue
            if alltoall:
                rounds = self.startup_descriptor(op, len(group)).startup_rounds
            else:
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
        else:
            # A shared/direct/pair forward plane still has placement-specific
            # injection and receipt services when the deployment declares them.
            if self._has_endpoint_rate(link, source):
                names.append(f"{self.namespace}link:{link.name}:tx:port{source}")
            if self._has_endpoint_rate(link, destination):
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

    def transport_plan(self, flows, *, chunk_bytes=None, receive_capacity_bytes=None,
                       hop_startup_us=STARTUP_ZERO_FLOAT, max_physical_chunks=DEFAULT_MAX_PHYSICAL_CHUNKS,
                       transport=None, receive_buffer_id=None):
        """Plan directed flows with actual routes, chunks and receive credits.

        ``hop_startup_us`` is an explicit, already-normalized protocol service
        boundary (typically from the T06 resource-service record).  It is paid
        once per hop and never added again to the wire service.  A missing
        declaration is recorded as unknown instead of being filled with a
        fitted or default total duration.
        ``transport`` is the single resolved :class:`TransportOptions`; when
        supplied it is authoritative and cannot be combined with independent
        chunk/startup/credit arguments.
        """
        if transport is not None:
            options = TransportOptions.from_mapping(transport)
            if (chunk_bytes is not None or receive_capacity_bytes is not None
                    or receive_buffer_id is not None
                    or float(hop_startup_us or 0.0)
                    or max_physical_chunks != 4096):
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

        capacity = []
        if receive_capacity_bytes is not None:
            if isinstance(receive_capacity_bytes, Mapping):
                items = receive_capacity_bytes.items()
            else:
                items = ((rank, receive_capacity_bytes)
                         for rank in {flow.destination for flow in flows})
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
                         for rank in {flow.destination for flow in flows})
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
        for flow in flows:
            if not isinstance(flow, MessageFlow):
                raise ModelingError("transport_flow_must_be_message_flow")
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
                # A receive slot smaller than one declared physical chunk is
                # not a chunk-size hint; the declared framing cannot be sent.
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

    def joint_plan(self, op, participants, *, algorithm="ring", max_inflight_per_rank=STARTUP_ONE,
                   chunk_bytes=None, hop_startup_us=STARTUP_ZERO_FLOAT,
                   receive_capacity_bytes_per_peer=None,
                   receive_buffer_id=None,
                   max_physical_chunks=DEFAULT_MAX_PHYSICAL_CHUNKS, transport=None):
        """Bind complete cross-DP cohorts without inferring a runtime algorithm."""
        from .collectives import PairwiseGroup, PairwisePlan, RingGroup, RingPlan

        if transport is not None:
            options = TransportOptions.from_mapping(transport)
            if (chunk_bytes is not None or receive_capacity_bytes_per_peer is not None
                    or receive_buffer_id is not None
                    or max_inflight_per_rank != 1
                    or float(hop_startup_us or 0.0)
                    or max_physical_chunks != 4096):
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
        if len(self._transport_bindings) >= 8192:
            self._transport_bindings.clear()
        self._transport_bindings[id(op)] = resolved
        self._transport_bindings[getattr(op, "op_id", id(op))] = resolved
        if algorithm not in {"ring", "pairwise", "direct_peer", "startup_waves"}:
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
            rounds = max(len(ranks) - STARTUP_PEER_PHASE_DELTA, STARTUP_ZERO)
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
                from .direct_peer import DirectPeerGroup
                groups.append(DirectPeerGroup(
                    ranks, tuple(totals), rounds, self, peer_traffic=peer,
                    max_inflight_per_rank=max_inflight_per_rank,
                    chunk_bytes=chunk_bytes,
                    hop_startup_us=hop_startup_us,
                    receive_capacity_bytes_per_peer=
                        receive_capacity_bytes_per_peer,
                    receive_buffer_id=receive_buffer_id,
                    max_physical_chunks=max_physical_chunks))
            elif algorithm == "startup_waves":
                from .collective_waves import WaveGroup
                wave_rounds = (startup.startup_rounds
                               if startup is not None else rounds)
                groups.append(WaveGroup(
                    ranks, tuple(totals), wave_rounds, self, peer_traffic=peer,
                    max_inflight_per_rank=max_inflight_per_rank))
            else:
                groups.append(PairwiseGroup(
                    ranks, tuple(totals), rounds, self, peer_traffic=peer))
        if algorithm == "startup_waves":
            from .collective_waves import WavePlan
            return WavePlan(
                family, tuple(groups),
                startup_rounds=(startup.startup_rounds
                                if startup is not None else None))
        return PairwisePlan(
            family, tuple(groups),
            startup_rounds=(startup.startup_rounds if startup is not None else None),
            startup_application=("physical_program_owns_arrivals"
                                 if alltoall else "startup_model"))

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

    def _accounting_flows(self, flow, chunk_bytes, max_physical_chunks=4096):
        """Physical framing units for the public byte ledger.

        The ledger must use the same declared chunk and expansion policy as
        execution.  A chunk is never enlarged to save tasks; a too-small
        expansion limit is reported by :func:`chunk_sizes` instead of changing
        the wire account.
        """
        if chunk_bytes is None:
            return (flow,)
        sizes = chunk_sizes(
            flow.payload_bytes, chunk_bytes, max_physical_chunks)
        return tuple(MessageFlow(flow.source, flow.destination, size,
                                 flow.messages) for size in sizes)

    def accounting(self, op, replica, *, volume=None, transport=None,
                   recipe=None):
        """Account the one selected physical plan.

        ``transport`` is the resolved :class:`TransportOptions` (or its
        mapping).  T04 resolves it once from the selected recipe and passes the
        same object to execution and this ledger.  ``recipe`` is a convenience
        for callers that only have the selected recipe and op; both are
        resolved through the same function.  Metadata is only the fallback for
        a native network op that carries its own explicit transport declaration.
        """
        if transport is None and recipe is not None:
            from .direct_peer import resolve_transport_options
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
                "transport_max_physical_chunks", 4096)
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
        return CommunicationWork(
            useful_by_rank, useful, wire, transactions, transaction_frames, count, latency,
            routed_useful, physical_wire, max_path_latency,
        )

    def demand(self, op, replica, *, volume=None, transport=None, recipe=None):
        work = self.accounting(
            op, replica, volume=volume, transport=transport, recipe=recipe)
        return dict(work.wire_by_resource), work.setup_ms
