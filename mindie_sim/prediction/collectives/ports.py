"""Canonical physical pair, port coset, and host root-complex resource keys.

[Role]
  Define immutable physical contention keys for directed fabric pairs
  ``(fabric, physical_src, physical_dst, rail)``, full-duplex physical ports
  ``(node, physical_endpoint, rail, tx_or_rx)``, and host bus root complexes
  ``(node, root_complex_id, direction)``. Map coset flows onto shared Engine
  resource rows while preserving disjoint-rail parallelism.

[Invariants]
  * ``communicator_id`` and ``coset_id`` are attribution/rendezvous metadata
    only; two cosets touching the same ``PortKey`` or physical directional die
    injection/SDMA ceiling contend on one Engine row.
  * Distinct physical rails on the same node/endpoint emit distinct ``PortKey``
    rows and overlap without artificial serialization.
  * Host bus contention binds one physical directional solo resource
    ``host_solo:{h2d,d2h}:d{rank}`` per die plus one ``RootComplexKey`` resource
    per unique ``(node, root_complex_id, direction)`` across participating dies;
    a solo die keeps its full solo ceiling without pre-division by configured
    die count.

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


@dataclass(frozen=True, slots=True)
class SwitchKey:
    """Shared physical switch-plane key: ``(node_or_none, switch_id)``."""

    node: int | None
    switch_id: str

    def __post_init__(self):
        if not self.switch_id:
            raise ModelingError("switch_key_id_empty")

    def resource_name(self, namespace: str = "") -> str:
        prefix = f"n{self.node}:" if self.node is not None else "global:"
        return f"{namespace}switch:{prefix}{self.switch_id}"


def _resolved_endpoint_id(link, rank: int, channel_decl=None) -> tuple[str, bool]:
    ep_map = dict(getattr(link, "endpoint_ids", ()) or ())
    if int(rank) in ep_map:
        return str(ep_map[int(rank)]), True
    if channel_decl is not None:
        raw_eps = channel_decl.get("endpoint_ids") or ()
        ch_map = (
            {int(k): str(v) for k, v in raw_eps.items()}
            if isinstance(raw_eps, dict)
            else {int(k): str(v) for k, v in raw_eps}
        )
        if int(rank) in ch_map:
            return ch_map[int(rank)], True
    return f"rank{int(rank)}", False


def _resolved_rail_id(link, channel_decl=None) -> int | str:
    rail = getattr(link, "rail_id", 0)
    if rail not in (0, "", None):
        return rail
    if channel_decl is not None:
        ch_rail = channel_decl.get("rail_id", 0)
        if ch_rail not in (0, "", None):
            return ch_rail
    return 0


def link_port_key(link, rank: int, node: int, direction: str, *, channel_decl=None) -> PortKey:
    """Resolve the canonical ``PortKey`` for one ``(link, rank, direction)``."""
    endpoint, _ = _resolved_endpoint_id(link, int(rank), channel_decl)
    rail = _resolved_rail_id(link, channel_decl)
    return PortKey(
        node=int(node),
        physical_endpoint=str(endpoint),
        rail=rail,
        direction=direction,
    )


def link_pair_key(link, src: int, dst: int, *, channel_decl=None) -> PairKey:
    """Resolve the canonical ``PairKey`` for one directed hop."""
    fabric = str(getattr(link, "channel", "") or getattr(link, "name", "fabric"))
    rail = _resolved_rail_id(link, channel_decl)
    return PairKey(
        fabric=fabric,
        physical_src=int(src),
        physical_dst=int(dst),
        rail=rail,
    )


def find_shared_port_keys(
    links, rank_node_lookup, *, port_semantics: set[str], channel_decls=None
):
    """Identify physical ``PortKey``s shared by 2+ ``(link, rank)`` endpoints.

    Catches BOTH two ranks on ONE link sharing the same physical endpoint/rail
    AND two distinct links/communicators touching the same physical port.
    """
    decls = channel_decls or {}
    port_entries: dict[PortKey, list[tuple[object, int]]] = defaultdict(list)
    for link in links:
        is_port_sem = link.semantics in port_semantics
        is_pair_sem = link.semantics == "per_pair"
        is_shared_sem = link.semantics == "shared"
        if not (is_port_sem or is_pair_sem or is_shared_sem):
            continue
        ch_decl = decls.get(getattr(link, "channel", ""))
        rail = _resolved_rail_id(link, ch_decl)
        has_ep_map = bool(getattr(link, "endpoint_ids", ())) or bool(
            ch_decl and ch_decl.get("endpoint_ids")
        )
        has_ep_rates = bool(getattr(link, "endpoint_rates", ()))
        explicit_port_identity = has_ep_map or (rail not in (0, "")) or (is_pair_sem and has_ep_rates)
        if not explicit_port_identity:
            continue
        for rank in link.members:
            if is_shared_sem:
                _, explicit_ep = _resolved_endpoint_id(link, int(rank), ch_decl)
                if not (explicit_ep or rail not in (0, "")):
                    continue
            node = int(rank_node_lookup(rank))
            for direction in ("tx", "rx"):
                key = link_port_key(link, rank, node, direction, channel_decl=ch_decl)
                port_entries[key].append((link, int(rank)))

    shared_rates: dict[PortKey, float] = {}
    link_port_map: dict[tuple[str, int, str], PortKey] = {}
    for key, entries in port_entries.items():
        if len(entries) < 2:
            continue
        rates = []
        for link, rank in entries:
            ep_cap = (
                link.endpoint_capacity_bytes_per_ms(rank)
                if hasattr(link, "endpoint_capacity_bytes_per_ms")
                else None
            )
            rates.append(float(ep_cap if ep_cap is not None else link.capacity_bytes_per_ms))
        shared_rates[key] = min(rates)
        for link, rank in entries:
            link_port_map[(link.name, rank, key.direction)] = key
    return shared_rates, link_port_map


def find_shared_switch_keys(links, rank_node_lookup, *, channel_decls=None):
    """Identify physical ``SwitchKey``s shared across links declaring switch capacity."""
    decls = channel_decls or {}
    sw_entries: dict[SwitchKey, list[object]] = defaultdict(list)
    for link in links:
        cap = getattr(link, "switch_capacity_bytes_per_ms", None)
        if cap is None:
            continue
        ch_decl = decls.get(getattr(link, "channel", ""))
        sw_id = str(getattr(link, "switch_id", "") or (ch_decl.get("switch_id") if ch_decl else "") or "")
        if not sw_id:
            continue
        nodes = {int(rank_node_lookup(r)) for r in link.members} if link.members else set()
        scope_node = next(iter(nodes)) if len(nodes) == 1 else None
        sw_entries[SwitchKey(scope_node, sw_id)].append(link)

    shared_rates: dict[SwitchKey, float] = {}
    link_switch_map: dict[str, SwitchKey] = {}
    for key, entries in sw_entries.items():
        if len(entries) < 2:
            continue
        shared_rates[key] = min(float(link.switch_capacity_bytes_per_ms) for link in entries)
        for link in entries:
            link_switch_map[link.name] = key
    return shared_rates, link_switch_map


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


def build_shared_switch_resources(
    shared_rates: dict[SwitchKey, float], *, namespace: str = ""
) -> tuple[Resource, ...]:
    """Emit one Engine ``Resource`` per shared physical ``SwitchKey``."""
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


def die_injection_resource_name(
    rank: int, direction: str, *, namespace: str = ""
) -> str:
    """Canonical physical directional die-injection resource name."""
    if direction not in VALID_PORT_DIRECTIONS:
        raise ModelingError(f"port_key_direction_invalid:{direction}")
    return f"{namespace}ub_inj:{direction}:r{int(rank)}"


def sdma_flow_resource_name(
    rank: int, direction: str, *, namespace: str = ""
) -> str:
    """Canonical physical directional per-die SDMA flow-cap resource name."""
    if direction not in VALID_PORT_DIRECTIONS:
        raise ModelingError(f"port_key_direction_invalid:{direction}")
    return f"{namespace}sdma_flow:{direction}:r{int(rank)}"


def host_solo_resource_name(
    die_or_rank, direction: str, *, namespace: str = ""
) -> str:
    """Canonical physical directional per-die host DMA resource name."""
    if direction not in VALID_HOST_DIRECTIONS:
        raise ModelingError(f"host_transfer_direction_invalid:{direction}")
    rank = (
        int(die_or_rank)
        if isinstance(die_or_rank, int) and not isinstance(die_or_rank, bool)
        else int(getattr(die_or_rank, "rank", 0))
    )
    return f"{namespace}host_solo:{direction}:d{rank}"


def host_solo_resources_for_dies(
    machine, dies, *, namespace: str = ""
) -> tuple[Resource, ...]:
    """Build per-die directional ``host_solo:{h2d,d2h}:d{rank}`` resources."""
    if machine is None or not dies or not hasattr(machine, "host_solo_bw_gbps"):
        return ()
    seen: set[int] = set()
    ordered_ranks: list[int] = []
    for die in dies:
        rank = int(getattr(die, "rank", 0))
        if rank not in seen:
            seen.add(rank)
            ordered_ranks.append(rank)

    resources: list[Resource] = []
    for direction in ("h2d", "d2h"):
        solo_gbps = machine.host_solo_bw_gbps(direction)
        if solo_gbps is None or float(solo_gbps) <= 0.0:
            continue
        rate_bytes_per_ms = float(solo_gbps) * GBPS_TO_BYTES_PER_MS
        for rank in ordered_ranks:
            resources.append(
                Resource(
                    host_solo_resource_name(rank, direction, namespace=namespace),
                    rate_bytes_per_ms,
                    "bytes",
                )
            )
    return tuple(resources)


def host_dma_resources_for_dies(
    machine, dies, *, namespace: str = ""
) -> tuple[Resource, ...]:
    """Build unified per-die solo + shared root-complex host DMA resources."""
    die_tuple = tuple(dies) if dies is not None else ()
    return (
        *host_solo_resources_for_dies(machine, die_tuple, namespace=namespace),
        *root_complex_resources_for_dies(machine, die_tuple, namespace=namespace),
    )


def host_transfer_demand(
    machine, die, direction: str, payload_bytes: float, *,
    namespace: str = "", rank_resource: str | None = None, dies=None,
) -> dict[str, float]:
    """Build coupled per-die solo + shared root-complex Engine demand dict.

    Bills ``1x`` ``payload_bytes`` onto each participating die's physical
    directional solo channel ``host_solo:{h2d,d2h}:d{rank}`` (or an explicit
    caller ``rank_resource`` override) and accumulates ``payload_bytes`` per
    participating die onto each die's
    ``RootComplexKey(node, root_complex_id, direction)``.
    """
    if direction not in VALID_HOST_DIRECTIONS:
        raise ModelingError(f"host_transfer_direction_invalid:{direction}")
    payload = float(payload_bytes)
    if payload <= 0.0:
        return {}
    participants = tuple(dies) if dies is not None else ((die,) if die is not None else ())
    demand: dict[str, float] = {}
    if rank_resource is not None:
        if rank_resource:
            demand[rank_resource] = payload
    elif participants:
        solo_gbps = (
            machine.host_solo_bw_gbps(direction)
            if machine is not None and hasattr(machine, "host_solo_bw_gbps")
            else None
        )
        if machine is None or (solo_gbps is not None and float(solo_gbps) > 0.0):
            for participant in participants:
                solo_name = host_solo_resource_name(
                    participant, direction, namespace=namespace
                )
                demand[solo_name] = demand.get(solo_name, 0.0) + payload
    ceiling_gbps = (
        machine.host_root_complex_ceiling_gbps(direction)
        if machine is not None and hasattr(machine, "host_root_complex_ceiling_gbps")
        else None
    )
    if ceiling_gbps is not None and ceiling_gbps > 0 and participants:
        for participant in participants:
            node = int(getattr(participant, "node", 0))
            rc_id = str(getattr(participant, "root_complex_id", "") or "rc0")
            key = RootComplexKey(node=node, root_complex_id=rc_id, direction=direction)
            rc_name = key.resource_name(namespace)
            demand[rc_name] = demand.get(rc_name, 0.0) + payload
    return demand
