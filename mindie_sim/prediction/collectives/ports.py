"""Canonical physical pair, port coset, and host root-complex resource keys.

[Role]
  Define immutable physical contention keys for directed fabric pairs
  ``(fabric, physical_src, physical_dst, rail)``, full-duplex physical ports
  ``(node, physical_endpoint, rail, tx_or_rx)``, and host bus root complexes
  ``(node, root_complex_id, direction)``. Map coset flows onto shared Engine
  resource rows while preserving disjoint-rail parallelism.

[Invariants]
  * ``communicator_id`` and ``coset_id`` are attribution/rendezvous metadata
    only; two cosets touching the same ``PortKey`` contend on one Engine row.
  * Distinct physical rails on the same node/endpoint emit distinct ``PortKey``
    rows and overlap without artificial serialization.
  * Host bus contention builds one ``RootComplexKey`` resource per unique
    ``(node, root_complex_id, direction)`` across all active dies; a solo die
    keeps its full solo ceiling without pre-division by configured die count.

[Timeouts & Locks]
  Pure in-memory key normalization and resource table construction; no I/O.

[Side-Effect Policy]
  Stateless key/resource builders with no global mutable state.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError
from ..engine import Resource

GBPS_TO_BYTES_PER_MS = 1_000_000.0
VALID_PORT_DIRECTIONS = frozenset({"tx", "rx"})
VALID_HOST_DIRECTIONS = frozenset({"h2d", "d2h"})


@dataclass(frozen=True, slots=True)
class PairKey:
    """Directed physical fabric pair key: ``(fabric, src, dst, rail)``."""

    fabric: str
    physical_src: int
    physical_dst: int
    rail: int | str = 0

    def __post_init__(self):
        if not self.fabric:
            raise ModelingError("pair_key_fabric_empty")
        if self.physical_src == self.physical_dst:
            raise ModelingError(f"pair_key_self_loop:{self.physical_src}")

    def resource_name(self, namespace: str = "") -> str:
        return (
            f"{namespace}pair:{self.fabric}:r{self.rail}:"
            f"{self.physical_src}>{self.physical_dst}"
        )


@dataclass(frozen=True, slots=True)
class PortKey:
    """Full-duplex physical port key: ``(node, endpoint, rail, tx_or_rx)``."""

    node: int
    physical_endpoint: str
    rail: int | str
    direction: str

    def __post_init__(self):
        if self.direction not in VALID_PORT_DIRECTIONS:
            raise ModelingError(f"port_key_direction_invalid:{self.direction}")
        if not self.physical_endpoint:
            raise ModelingError("port_key_endpoint_empty")

    def resource_name(self, namespace: str = "") -> str:
        return (
            f"{namespace}port:n{self.node}:{self.physical_endpoint}:"
            f"r{self.rail}:{self.direction}"
        )


@dataclass(frozen=True, slots=True)
class RootComplexKey:
    """Shared host bus root-complex key: ``(node, root_complex_id, dir)``."""

    node: int
    root_complex_id: str
    direction: str

    def __post_init__(self):
        if self.direction not in VALID_HOST_DIRECTIONS:
            raise ModelingError(
                f"root_complex_key_direction_invalid:{self.direction}"
            )
        if not self.root_complex_id:
            raise ModelingError("root_complex_id_empty")

    def resource_name(self, namespace: str = "") -> str:
        return (
            f"{namespace}host_rc:n{self.node}:{self.root_complex_id}:"
            f"{self.direction}"
        )


def link_port_key(link, rank: int, node: int, direction: str) -> PortKey:
    """Resolve the canonical ``PortKey`` for one ``(link, rank, direction)``."""
    endpoint = (
        link.physical_endpoint_id(rank)
        if hasattr(link, "physical_endpoint_id")
        else f"rank{int(rank)}"
    )
    rail = getattr(link, "rail_id", 0)
    return PortKey(
        node=int(node),
        physical_endpoint=str(endpoint),
        rail=rail,
        direction=direction,
    )


def link_pair_key(link, src: int, dst: int) -> PairKey:
    """Resolve the canonical ``PairKey`` for one directed hop."""
    fabric = str(getattr(link, "channel", "") or getattr(link, "name", "fabric"))
    rail = getattr(link, "rail_id", 0)
    return PairKey(
        fabric=fabric,
        physical_src=int(src),
        physical_dst=int(dst),
        rail=rail,
    )


def find_shared_port_keys(links, rank_node_lookup, *, port_semantics: set[str]):
    """Identify physical ``PortKey``s shared across two or more port links.

    Returns ``(shared_rates, link_port_map)`` where ``shared_rates`` maps each
    multi-link ``PortKey`` to its physical port capacity (bytes/ms), and
    ``link_port_map`` maps ``(link.name, rank, direction)`` to that shared
    ``PortKey``.
    """
    port_links: dict[PortKey, list[tuple[object, int]]] = defaultdict(list)
    for link in links:
        if link.semantics not in port_semantics:
            continue
        explicit_port_identity = bool(getattr(link, "endpoint_ids", ())) or (
            getattr(link, "rail_id", 0) not in (0, "")
        )
        if not explicit_port_identity:
            continue
        for rank in link.members:
            node = int(rank_node_lookup(rank))
            for direction in ("tx", "rx"):
                key = link_port_key(link, rank, node, direction)
                port_links[key].append((link, int(rank)))

    shared_rates: dict[PortKey, float] = {}
    link_port_map: dict[tuple[str, int, str], PortKey] = {}
    for key, entries in port_links.items():
        unique_links = {link.name for link, _ in entries}
        if len(unique_links) < 2:
            continue
        rates = [
            float(link.capacity_at(rank) if hasattr(link, "capacity_at")
                  else link.capacity_bytes_per_ms)
            for link, rank in entries
        ]
        shared_rates[key] = min(rates)
        for link, rank in entries:
            link_port_map[(link.name, rank, key.direction)] = key
    return shared_rates, link_port_map


def build_shared_port_resources(
    shared_rates: dict[PortKey, float], *, namespace: str = ""
) -> tuple[Resource, ...]:
    """Emit one Engine ``Resource`` per shared physical ``PortKey``."""
    return tuple(
        Resource(key.resource_name(namespace), rate, "bytes")
        for key, rate in sorted(
            shared_rates.items(), key=lambda item: item[0].resource_name(namespace)
        )
    )


def root_complex_resources_for_dies(
    machine, dies, *, namespace: str = ""
) -> tuple[Resource, ...]:
    """Build shared host root-complex Engine resources for every unique RC.

    Creates one ``(node, root_complex_id, direction)`` resource per unique
    ``(die.node, die.root_complex_id)`` present in ``dies`` for ``h2d`` and
    ``d2h``. Never collapses distinct nodes onto ``dies[0].node`` and never
    pre-divides solo die ceilings by configured die count.
    """
    if machine is None or not dies:
        return ()
    seen: set[tuple[int, str]] = set()
    ordered_rcs: list[tuple[int, str]] = []
    for die in dies:
        node = int(getattr(die, "node", 0))
        rc_id = str(getattr(die, "root_complex_id", "") or "rc0")
        pair = (node, rc_id)
        if pair not in seen:
            seen.add(pair)
            ordered_rcs.append(pair)

    resources: list[Resource] = []
    for direction in ("h2d", "d2h"):
        ceiling_gbps = (
            machine.host_root_complex_ceiling_gbps(direction)
            if hasattr(machine, "host_root_complex_ceiling_gbps")
            else None
        )
        if ceiling_gbps is None or ceiling_gbps <= 0:
            continue
        rate_bytes_per_ms = float(ceiling_gbps) * GBPS_TO_BYTES_PER_MS
        for node, rc_id in ordered_rcs:
            key = RootComplexKey(
                node=node, root_complex_id=rc_id, direction=direction
            )
            resources.append(
                Resource(key.resource_name(namespace), rate_bytes_per_ms, "bytes")
            )
    return tuple(resources)


def host_transfer_demand(
    machine, die, direction: str, payload_bytes: float, *,
    namespace: str = "", rank_resource: str | None = None,
) -> dict[str, float]:
    """Build coupled per-die solo + shared root-complex Engine demand dict."""
    if direction not in VALID_HOST_DIRECTIONS:
        raise ModelingError(f"host_transfer_direction_invalid:{direction}")
    payload = float(payload_bytes)
    if payload <= 0.0:
        return {}
    demand: dict[str, float] = {}
    if rank_resource:
        demand[rank_resource] = payload
    ceiling_gbps = (
        machine.host_root_complex_ceiling_gbps(direction)
        if machine is not None and hasattr(machine, "host_root_complex_ceiling_gbps")
        else None
    )
    if ceiling_gbps is not None and ceiling_gbps > 0:
        node = int(getattr(die, "node", 0))
        rc_id = str(getattr(die, "root_complex_id", "") or "rc0")
        key = RootComplexKey(node=node, root_complex_id=rc_id, direction=direction)
        demand[key.resource_name(namespace)] = payload
    return demand
