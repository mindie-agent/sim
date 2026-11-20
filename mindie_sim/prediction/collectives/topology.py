"""Route communication through declared physical links and shared port cosets.

[Role]
  Map collective, peer and point-to-point flows to declared link, port, pair,
  and shared physical port-coset Engine resources with explicit scope
  precedence (``package`` -> ``node`` -> ``cross_node``).

[Invariants]
  * Package locality compares ``(node_id, package_id)`` tuples, never bare
    ``package_id`` (so repeated package IDs on distinct nodes never collide).
  * Explicit ``link.scope`` takes precedence over name-marker fallback; any
    unknown explicit scope raises ``ModelingError("link_scope_unknown:...")``
    instead of silently admitting traffic.
  * Distinct links/cosets sharing the same physical
    ``(node, physical_endpoint, rail, tx_or_rx)`` port contend on one shared
    Engine resource row; distinct rails overlap cleanly.
  * Logical message bytes and transaction-rounded wire bytes remain distinct.

[Timeouts & Locks]
  Finite rank graph, bounded route cache, and bounded chunk count; no I/O.

[Side-Effect Policy]
  Caches bounded route tables and submits tasks only to caller-owned Engines.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
import math

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware_links import parse_collective_bands
from ..communication import CommunicationWork, MessageFlow, RoutedHop, RoutedMessage
from ..engine import Resource
from .plans import (
    DIRECT_MESH_FAMILIES,
    DIRECT_MESH_MAX_PEERS,
    DIRECT_MESH_PHASES,
    _TopologyPlansMixin,
    _declared_peer_traffic,
    _peer_declared_volume,
    _peer_totals_match,
)
from .ports import (
    PairKey,
    PortKey,
    build_shared_port_resources,
    find_shared_port_keys,
    link_pair_key,
    link_port_key,
)
from .transport import (
    DEFAULT_MAX_PHYSICAL_CHUNKS,
    MS_PER_S,
    ChunkSequence,
    FlowPath,
    TransportOptions,
    TransportPlan,
    TransportRun,
    _transport_ready,
    chunk_count,
    chunk_size_at,
    chunk_sizes,
)

MAX_ROUTE_CACHE_ENTRIES = 8192
UNDECLARED_SCOPE_PRIORITY = 3
_SCOPE_PRIORITY = {
    "package": 0,
    "node": 1,
    "cross_node": 2,
    "any": UNDECLARED_SCOPE_PRIORITY,
    "undeclared": UNDECLARED_SCOPE_PRIORITY,
}
_ROUTABLE_EXPLICIT_SCOPES = frozenset({"package", "node", "cross_node", "any"})
_NON_FABRIC_EXPLICIT_SCOPES = frozenset({"host", "device_local"})

_PORT_SEMANTICS = {"per_port", "fully_connected", "mesh"}
_CROSS_NODE_MARKERS = (
    "inter_node", "inter_pod", "roce", "rdma", "scaleout", "scale_out",
)
_PACKAGE_MARKERS = ("intra_package", "package", "die_to_die")
_NODE_MARKERS = ("intra_node", "intra_4p", "intra_8p", "node_mesh")


def _link_text(link) -> str:
    return (f"{getattr(link, 'channel', '')} {getattr(link, 'name', '')}"
            .lower().replace("-", "_"))


def _is_cross_node_link(link) -> bool:
    explicit = getattr(link, "scope", None)
    if explicit is not None:
        return str(explicit).strip().lower() == "cross_node"
    text = _link_text(link)
    return any(marker in text for marker in _CROSS_NODE_MARKERS)


def _link_scope(link, channel_decl: Mapping | None = None) -> str:
    """Resolve link scope from explicit ``link.scope`` or channel metadata.

    Rejects unknown explicit scopes on either the link or its declared
    ``transfer_channels`` entry; falls back to channel/name markers only when
    neither declares ``scope``.
    """
    explicit = getattr(link, "scope", None)
    if explicit is None and channel_decl is not None:
        explicit = channel_decl.get("scope")
    if explicit is not None:
        normalized = str(explicit).strip().lower()
        if normalized in _ROUTABLE_EXPLICIT_SCOPES or normalized in _NON_FABRIC_EXPLICIT_SCOPES:
            return normalized
        raise ModelingError(f"link_scope_unknown:{getattr(link, 'name', '')}:{explicit}")
    text = _link_text(link)
    if any(marker in text for marker in _CROSS_NODE_MARKERS):
        return "cross_node"
    if any(marker in text for marker in _PACKAGE_MARKERS):
        return "package"
    if any(marker in text for marker in _NODE_MARKERS):
        return "node"
    return "undeclared"


@dataclass(frozen=True)
class TopologyCommunicationWork(CommunicationWork):
    """Communication ledger that also publishes each resource's capacity."""

    capacity_by_resource: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self):
        result = super().to_dict()
        for name, row in result["resources"].items():
            capacity = self.capacity_by_resource.get(name)
            if capacity is not None:
                row["peak_bytes_per_ms"] = float(capacity)
        return result


class Topology(_TopologyPlansMixin):
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
        self._channel_decls: dict[str, dict] = {}
        self._channel_bands: dict[str, tuple] = {}
        self._channel_rail: dict[str, int | str] = {}
        self._channel_startup_ms = {}
        channels = getattr(machine, "transfer_channels", ()) or ()
        entries = channels.items() if isinstance(channels, Mapping) else channels
        for name, items in entries:
            decl = dict(items)
            ch_name = str(name)
            self._channel_decls[ch_name] = decl
            raw_bands = decl.get("protocol_bands")
            if raw_bands:
                self._channel_bands[ch_name] = parse_collective_bands(raw_bands)
            if decl.get("rail_id") not in (None, "", 0):
                self._channel_rail[ch_name] = decl["rail_id"]
            value = decl.get("alpha_us")
            if value is None:
                continue
            try:
                amount = float(value)
            except (TypeError, ValueError):
                raise ModelingError(
                    f"transport_channel_alpha_invalid:{name}") from None
            if not math.isfinite(amount) or amount < 0:
                raise ModelingError(f"transport_channel_alpha_invalid:{name}")
            self._channel_startup_ms[ch_name] = amount / MS_PER_S
        self._link_scope = {
            name: _link_scope(link, self._channel_decls.get(link.channel))
            for name, link in self.links.items()
        }
        self._ordered_links = tuple(sorted(
            self.links.values(),
            key=lambda item: (
                _SCOPE_PRIORITY.get(
                    self._link_scope[item.name],
                    UNDECLARED_SCOPE_PRIORITY),
                len(item.members),
                -item.capacity_bytes_per_ms,
                item.name,
            ),
        ))
        self._shared_port_rates, self._link_shared_ports = find_shared_port_keys(
            self.links.values(),
            self._node_of,
            port_semantics=_PORT_SEMANTICS,
        )
        self._resource_cache = None
        self._resource_rates = None
        self._transport_bindings = {}
        self._mesh_plans = {}

    def _node_of(self, rank: int) -> int:
        key = int(rank)
        if key in self._rank_node:
            return self._rank_node[key]
        return key // self._ranks_per_node

    def _package_key(self, rank: int) -> tuple[int, int]:
        """Return ``(node_id, package_id)`` so package IDs never collide across nodes."""
        key = int(rank)
        node = self._node_of(key)
        if key in self._rank_package:
            return (node, self._rank_package[key])
        return (node, key // self._dies_per_package)

    def link_protocol_bands(self, link_name: str):
        """Return piecewise ``CollectiveBandSpec``s declared on link or channel."""
        link = self.links[link_name]
        if link.protocol_bands:
            return link.protocol_bands
        return self._channel_bands.get(link.channel, ())

    def port_key(self, link, rank: int, direction: str) -> PortKey:
        """Canonical physical ``PortKey`` for ``(link, rank, direction)``."""
        key = link_port_key(link, int(rank), self._node_of(rank), direction)
        if key.rail == 0 and link.channel in self._channel_rail:
            return PortKey(
                node=key.node,
                physical_endpoint=key.physical_endpoint,
                rail=self._channel_rail[link.channel],
                direction=key.direction,
            )
        return key

    def pair_key(self, link, source: int, destination: int) -> PairKey:
        """Canonical physical ``PairKey`` for ``(link, source, destination)``."""
        key = link_pair_key(link, int(source), int(destination))
        if key.rail == 0 and link.channel in self._channel_rail:
            return PairKey(
                fabric=key.fabric,
                physical_src=key.physical_src,
                physical_dst=key.physical_dst,
                rail=self._channel_rail[link.channel],
            )
        return key

    def scope_for_group(self, ranks) -> str:
        """Return tightest physical scope spanning all ranks in ``ranks``."""
        unique = tuple(int(r) for r in ranks)
        if len(unique) <= 1:
            return "package"
        nodes = {self._node_of(r) for r in unique}
        if len(nodes) > 1:
            return "cross_node"
        packages = {self._package_key(r) for r in unique}
        return "package" if len(packages) == 1 else "node"

    def startup_descriptor(self, op, group_size):
        """Resolve this collective's declared startup model once."""
        from ..collective_startup import resolve_collective_startup

        metadata = dict(op.metadata)
        family = str(metadata.get("collective", op.family)).strip().lower()
        return resolve_collective_startup(
            family=family, group_size=int(group_size), metadata=metadata,
            runtime_mode=self.collective_startup_mode,
            topology_protocols=self.collective_protocols)

    def direct_mesh_groups(self, op, replica):
        """Declared single-node full all-to-all pair mesh for this invocation."""
        metadata = dict(op.metadata)
        family = str(metadata.get("collective", op.family)).strip().lower()
        if family not in DIRECT_MESH_FAMILIES or "peer_traffic" in metadata:
            return family, ()
        axis = metadata.get("group_axis", "tp")
        try:
            stage = int(metadata.get("source_stage", op.stage))
        except (TypeError, ValueError):
            return family, ()
        if not 0 <= stage < self.parallel.pp:
            return family, ()
        phases = DIRECT_MESH_PHASES[family]
        local = set(self.ranks(replica, stage))
        groups = tuple(self.groups(axis, replica, stage, metadata=metadata))
        if not groups:
            return family, ()
        for ranks in groups:
            if not set(ranks) <= local:
                return family, ()
            if not 2 <= len(ranks) <= DIRECT_MESH_MAX_PEERS:
                return family, ()
            nodes = {self._node_of(rank) for rank in ranks}
            if len(nodes) != 1:
                return family, ()
            rounds = len(phases) * (len(ranks) - 1)
            if metadata.get("steps", rounds) != rounds:
                return family, ()
            for source in ranks:
                for destination in ranks:
                    if source == destination:
                        continue
                    try:
                        path = self.route(source, destination)
                    except ValueError:
                        return family, ()
                    if len(path) != 1:
                        return family, ()
                    if self.links[path[0][0]].semantics != "per_pair":
                        return family, ()
        return family, groups

    def declares_direct_mesh(self, op, replica) -> bool:
        """True when this invocation declares the direct K_8 pair mesh."""
        return bool(self.direct_mesh_groups(op, replica)[1])

    def bind_direct_mesh(self, op, replica, plan) -> None:
        """Publish the one mesh plan selected for an invocation."""
        if len(self._mesh_plans) >= MAX_ROUTE_CACHE_ENTRIES:
            self._mesh_plans.clear()
        self._mesh_plans[(str(getattr(op, "op_id", "")), int(replica))] = plan

    def bound_mesh_plan(self, op, replica):
        if not self._mesh_plans:
            return None
        return self._mesh_plans.get(
            (str(getattr(op, "op_id", "")), int(replica)))

    def mesh_phase_startup_ms(self, links) -> float:
        """Declared per-phase protocol startup; unknown stays zero."""
        return max((self._channel_startup_ms.get(str(link.channel), 0.0)
                    for link in links), default=0.0)

    def _work_with_capacity(self, work) -> TopologyCommunicationWork:
        rates = self._resource_rates
        if rates is None:
            self.resources()
            rates = self._resource_rates
        return TopologyCommunicationWork(
            work.useful_by_rank, work.useful_by_resource, work.wire_by_resource,
            work.transactions_by_resource, work.transaction_frames, work.messages,
            work.setup_ms, work.routed_useful_bytes, work.physical_wire_bytes,
            work.path_latency_ms,
            capacity_by_resource={
                name: rates[name] for name in work.wire_by_resource
                if name in rates})

    def location(self, rank: int) -> dict:
        """Return the declared die/package/node placement of one rank."""
        key = int(rank)
        if key not in self._rank_node:
            raise ModelingError(f"physical_rank_unknown:{rank}")
        return {"rank": key, "device": self._rank_device.get(key, key),
                "package": self._rank_package[key],
                "node": self._rank_node[key]}

    def link_scope(self, name: str) -> str:
        if name not in self._link_scope:
            raise ModelingError(f"link_unknown:{name}")
        return self._link_scope[name]

    def _endpoint_rate(self, link, rank: int) -> float:
        value = link.endpoint_capacity_bytes_per_ms(int(rank))
        return (float(value) if value is not None
                else float(link.capacity_bytes_per_ms))

    def _endpoint_resource(self, link, direction: str, rank: int):
        return Resource(
            f"{self.namespace}link:{link.name}:{direction}:port{rank}",
            self._endpoint_rate(link, rank), "bytes")

    def _has_endpoint_rate(self, link, rank: int) -> bool:
        return link.endpoint_capacity_bytes_per_ms(int(rank)) is not None

    def _build_resources(self, active: frozenset[int] | None = None):
        result = []
        included_shared_ports: dict[PortKey, float] = {}
        for link in self.links.values():
            endpoint_ranks = sorted({int(rank) for rank, _ in link.endpoint_rates})
            if active is not None and not (set(link.members) & active):
                continue
            if link.semantics == "per_pair":
                result.extend(Resource(
                    f"{self.namespace}link:{link.name}:pair{source}>{destination}",
                    link.capacity_bytes_per_ms, "bytes",
                ) for source in link.members for destination in link.members
                    if source != destination
                    and (active is None or source in active or destination in active))
                result.extend(self._endpoint_resource(link, direction, rank)
                              for rank in endpoint_ranks
                              if active is None or rank in active
                              for direction in ("tx", "rx"))
            elif link.semantics in _PORT_SEMANTICS:
                result.extend(self._endpoint_resource(link, direction, rank)
                              for rank in link.members
                              for direction in ("tx", "rx"))
                for rank in link.members:
                    for direction in ("tx", "rx"):
                        shared_key = self._link_shared_ports.get(
                            (link.name, int(rank), direction)
                        )
                        if shared_key is not None:
                            included_shared_ports[shared_key] = (
                                self._shared_port_rates[shared_key]
                            )
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
        if included_shared_ports:
            result.extend(
                build_shared_port_resources(
                    included_shared_ports, namespace=self.namespace
                )
            )
        return tuple(result)

    def resources(self):
        if self._resource_cache is not None:
            return self._resource_cache
        self._resource_cache = self._build_resources()
        self._resource_rates = {resource.name: resource.rate
                                for resource in self._resource_cache}
        return self._resource_cache

    def resources_for_ranks(self, ranks):
        """Link resources touching one replica, not the whole declared cluster."""
        active = frozenset(int(rank) for rank in ranks)
        cache = getattr(self, "_rank_resource_cache", None)
        if cache is None:
            cache = {}
            self._rank_resource_cache = cache
        if active not in cache:
            cache[active] = self._build_resources(active)
        return cache[active]

    def route(self, source, destination):
        key = source, destination
        if key in self.routes:
            return self.routes[key]
        queue, seen = deque([(source, ())]), {source}
        while queue:
            rank, path = queue.popleft()
            if rank == destination:
                if len(self.routes) < MAX_ROUTE_CACHE_ENTRIES:
                    self.routes[key] = path
                return path
            for link in self._ordered_links:
                if rank not in link.members:
                    continue
                scope = self._link_scope[link.name]
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
            return not self._same_node(left, right)
        if scope in {"any", "undeclared"}:
            return True
        if scope in _NON_FABRIC_EXPLICIT_SCOPES:
            return False
        raise ModelingError(f"link_scope_unknown:{scope}")

    def _same_node(self, left: int, right: int) -> bool:
        return self._node_of(left) == self._node_of(right)

    def _same_package(self, left: int, right: int) -> bool:
        return self._package_key(left) == self._package_key(right)

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
            from ..local_tp_groups import local_tp_groups

            declared = dict(metadata or {})
            width = int(declared.get("local_tp_width", 0) or 0)
            if width <= 0:
                raise ModelingError("local_tp_group_requires_declared_width")
            all_groups = local_tp_groups(self.machine, p, stage=stage, width=width)
            selected = tuple(group for group in all_groups
                             if any(rank in sources for rank in group))
            return selected or all_groups
        if axis == "cp":
            if p.pcp > 1:
                transposed = tuple(
                    sources[copy * p.tp + lane]
                    for lane in range(p.tp)
                    for copy in range(p.pcp)
                )
                return tuple(
                    transposed[i:i + p.cp] for i in range(0, len(transposed), p.cp)
                )
            return tuple(sources[i:i + p.cp] for i in range(0, len(sources), p.cp))
        if axis == "dp":
            return tuple(
                tuple(self.ranks(r, stage)[idx] for r in range(p.dp))
                for idx in range(len(sources))
            )
        if axis == "etp":
            if p.ep_spans_dp:
                return tuple((rank,) for rank in sources)
            return tuple(sources[base + i:base + p.tp:p.ep]
                         for base in range(0, len(sources), p.tp) for i in range(p.ep))
        if axis == "pcp":
            return tuple(sources[i::p.tp] for i in range(p.tp))
        if axis == "ep":
            if p.rank_table is not None:
                members = set(sources)
                return tuple(tuple(group) for group in p.rank_table
                             if members.intersection(group))
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
            tx_shared = self._link_shared_ports.get((link.name, int(source), "tx"))
            if tx_shared is not None:
                names.append(tx_shared.resource_name(self.namespace))
            rx_shared = self._link_shared_ports.get((link.name, int(destination), "rx"))
            if rx_shared is not None:
                names.append(rx_shared.resource_name(self.namespace))
        else:
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
                transactions, frame, link.propagation_us / MS_PER_S,
                self.capacity_resources(link, source, destination)[1:],
            ))
        return RoutedMessage(flow, tuple(hops))


__all__ = [
    "ChunkSequence",
    "DEFAULT_MAX_PHYSICAL_CHUNKS",
    "DIRECT_MESH_FAMILIES",
    "DIRECT_MESH_MAX_PEERS",
    "DIRECT_MESH_PHASES",
    "FlowPath",
    "Topology",
    "TopologyCommunicationWork",
    "TransportOptions",
    "TransportPlan",
    "TransportRun",
    "_PORT_SEMANTICS",
    "_SCOPE_PRIORITY",
    "_declared_peer_traffic",
    "_is_cross_node_link",
    "_link_scope",
    "_link_text",
    "_peer_declared_volume",
    "_peer_totals_match",
    "_transport_ready",
    "chunk_count",
    "chunk_size_at",
    "chunk_sizes",
]
