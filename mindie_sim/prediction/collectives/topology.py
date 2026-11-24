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
from dataclasses import dataclass, field, replace
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
    SwitchKey,
    build_shared_port_resources,
    build_shared_switch_resources,
    find_shared_port_keys,
    find_shared_switch_keys,
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
NS_PER_US_TOPO = 1000.0
P15_PRIMITIVE_ID = "P15_rts_stars"
P16_PRIMITIVE_ID = "P16_multistream_core_control"
P17_PRIMITIVE_ID = "P17_aicpu_metadata_launch"
P18_PRIMITIVE_ID = "P18_hccl_collective_band"
P19_PRIMITIVE_ID = "P19_direct_peer_window"
P20_PRIMITIVE_ID = "P20_host_dma_sync"
P19_UNCOALESCED_OP_RATIO = 1.5
STARS_SYNC_EVENTS_PER_STEP = 2
DUPLEX_SERIAL_OPS = 2.0
ALLREDUCE_DEVICE_PHASES = 2
_FAMILY_ALIASES = {"all_reduce": "allreduce", "all_gather": "allgather"}
_SCOPE_PRIORITY = {
    "package": 0,
    "node": 1,
    "cross_node": 2,
    "any": UNDECLARED_SCOPE_PRIORITY,
}
_ROUTABLE_EXPLICIT_SCOPES = frozenset({"package", "node", "cross_node", "any"})
_NON_FABRIC_EXPLICIT_SCOPES = frozenset({"host", "device_local"})

_PORT_SEMANTICS = {"per_port", "fully_connected", "mesh"}


def _link_text(link) -> str:
    del link
    raise NotImplementedError(
        "LEGACY_PURGED: deleted substring link-name guessing helper _link_text; "
        "read explicit typed LinkSpec.scope (S5-045)"
    )


def _is_cross_node_link(link) -> bool:
    explicit = getattr(link, "scope", None)
    if explicit is not None:
        return str(explicit).strip().lower() == "cross_node"
    raise NotImplementedError(
        "LEGACY_PURGED: deleted substring link-name guessing; require explicit typed "
        "LinkSpec.scope on every physical link (S5-045)"
    )


def _link_scope(link, channel_decl: Mapping | None = None) -> str:
    """Resolve link scope strictly from explicit ``link.scope`` or channel metadata."""
    explicit = getattr(link, "scope", None)
    if explicit is None and channel_decl is not None:
        explicit = channel_decl.get("scope")
    if explicit is not None:
        normalized = str(explicit).strip().lower()
        if normalized in _ROUTABLE_EXPLICIT_SCOPES or normalized in _NON_FABRIC_EXPLICIT_SCOPES:
            return normalized
        raise ModelingError(f"link_scope_unknown:{getattr(link, 'name', '')}:{explicit}")
    raise NotImplementedError(
        "LEGACY_PURGED: deleted substring link-name guessing and 'undeclared' pass-through; "
        "require explicit typed LinkSpec.scope (S5-045)"
    )


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
                 collective_startup_mode="legacy", primitive_table=None):
        self.machine, self.parallel = machine, parallel
        self.namespace = namespace
        self.collective_startup_mode = collective_startup_mode
        self.collective_protocols = dict(
            getattr(machine, "collective_protocols", ()) or ())
        self.links = {link.name: link for link in machine.links}
        self.p18_startup_us = 0.0
        self.p18_eager_startup_us = 0.0
        self._p18_link_bytes_per_ms = 0.0
        self._p18_cap_bytes_per_ms = 0.0
        self._p18_measured_scope = ""
        self._p18_measured_collective = ""
        self._p18_measured_world_size = 0
        self._has_primitive_table = bool(primitive_table is not None and hasattr(primitive_table, "cell"))
        self._p15_aiv_first_issue_us = 0.0
        self._p15_issue_interval_us = 0.0
        self._p15_host_submit_us = 0.0
        self._p16_stream_event_us = 0.0
        self._p17_eager_sync_us = 0.0
        self._p20_pinned_d2h_startup_us = 0.0
        self._p19_duplex_blocked_us = 0.0
        self._p19_duplex_eager_us = 0.0
        self._p19_duplex_bytes_per_ms = 0.0
        self._p19_uncoalesced_duplex = False
        self._vector_invocation_us = self._extract_vector_invocation_us(machine)
        if self._has_primitive_table:
            self._load_calibrated_comm_primitives(primitive_table)
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
        if self._p18_link_bytes_per_ms > 0.0:
            if not self._p18_measured_scope:
                raise NotImplementedError(
                    "LEGACY_PURGED: deleted covert _dies_per_package >= 2 P18 scope fingerprint; "
                    "bind P18 scope explicitly from Hardware/PrimitiveTable (S5-044)"
                )
            p18_scope = self._p18_measured_scope
            for name, link in list(self.links.items()):
                if (
                    self._link_scope.get(name) == p18_scope
                    and link.capacity_bytes_per_ms > self._p18_link_bytes_per_ms
                    and (
                        self._p18_cap_bytes_per_ms <= 0.0
                        or link.capacity_bytes_per_ms <= self._p18_cap_bytes_per_ms
                    )
                ):
                    self.links[name] = replace(link, capacity_bytes_per_ms=self._p18_link_bytes_per_ms)
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
            channel_decls=self._channel_decls,
        )
        self._shared_switch_rates, self._link_shared_switches = find_shared_switch_keys(
            self.links.values(),
            self._node_of,
            channel_decls=self._channel_decls,
        )
        self._resource_cache = None
        self._resource_rates = None
        self._transport_bindings = {}
        self._mesh_plans = {}
        self._oneshot_plans = {}

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

    @staticmethod
    def _extract_vector_invocation_us(machine) -> float:
        dies = getattr(machine, "dies", ()) or ()
        if not dies:
            return 0.0
        ki = dict(getattr(dies[0], "kernel_invocations", ()) or ())
        vec = ki.get("vector")
        if vec is None:
            return 0.0
        val = float(getattr(vec, "minimum_ms", 0.0) or 0.0) * MS_PER_S
        return val if val > 0.0 else 0.0

    @staticmethod
    def _sealed_metric_p50(cell, metric_name: str) -> float:
        if cell is None:
            return 0.0
        band = (getattr(cell, "metrics", None) or {}).get(metric_name)
        if band is None or band.p50 is None:
            return 0.0
        val = float(band.p50)
        return val if val > 0.0 else 0.0

    @staticmethod
    def _sealed_p19_diag_p50(p19_params: Mapping, metric_name: str) -> float:
        entry = (p19_params.get("hccl_duplex_diagnostics") or {}).get(metric_name)
        if not isinstance(entry, Mapping):
            return 0.0
        diag = entry.get("diagnostic") or {}
        band = entry.get("band") or {}
        if diag.get("status") != "calibrated" or band.get("p50") is None:
            return 0.0
        val = float(band["p50"])
        return val if val > 0.0 else 0.0

    def _load_calibrated_comm_primitives(self, primitive_table) -> None:
        def _cell(pid: str):
            try:
                return primitive_table.cell(pid)
            except ModelingError:
                return None

        p18 = _cell(P18_PRIMITIVE_ID)
        if p18 is not None:
            p18_params = p18.parameters or {}
            self._p18_measured_scope = str(p18_params.get("measured_scope") or "")
            raw_coll = str(p18_params.get("measured_collective") or "all_reduce").strip().lower()
            self._p18_measured_collective = _FAMILY_ALIASES.get(raw_coll, raw_coll)
            self._p18_measured_world_size = int(p18_params.get("measured_world_size") or 0)
            lb = self._sealed_metric_p50(p18, "link_bytes_per_s")
            if lb > 0.0:
                self._p18_link_bytes_per_ms = lb / MS_PER_S
                band = (p18.metrics or {}).get("link_bytes_per_s")
                cap = float(getattr(band, "certified_upper_capacity", 0.0) or 0.0)
                if cap > 0.0:
                    self._p18_cap_bytes_per_ms = cap / MS_PER_S
            sb = self._sealed_metric_p50(p18, "startup_blocked_ns")
            if sb > 0.0:
                self.p18_startup_us = sb / NS_PER_US_TOPO
            se = self._sealed_metric_p50(p18, "startup_ns")
            if se > 0.0:
                self.p18_eager_startup_us = se / NS_PER_US_TOPO
        p15 = _cell(P15_PRIMITIVE_ID)
        self._p15_aiv_first_issue_us = self._sealed_metric_p50(p15, "aiv_first_issue_ns") / NS_PER_US_TOPO
        self._p15_issue_interval_us = self._sealed_metric_p50(p15, "issue_interval_ns") / NS_PER_US_TOPO
        self._p15_host_submit_us = self._sealed_metric_p50(p15, "host_submit_ns") / NS_PER_US_TOPO
        p16 = _cell(P16_PRIMITIVE_ID)
        self._p16_stream_event_us = self._sealed_metric_p50(p16, "stream_event_ns") / NS_PER_US_TOPO
        p17 = _cell(P17_PRIMITIVE_ID)
        p20 = _cell(P20_PRIMITIVE_ID)
        eager_sync = self._sealed_metric_p50(p17, "eager_d2h_sync_ns") or self._sealed_metric_p50(p20, "eager_d2h_sync_ns")
        self._p17_eager_sync_us = eager_sync / NS_PER_US_TOPO
        self._p20_pinned_d2h_startup_us = self._sealed_metric_p50(p20, "pinned_d2h_startup_ns") / NS_PER_US_TOPO
        p19 = _cell(P19_PRIMITIVE_ID)
        if p19 is not None:
            p19_params = p19.parameters or {}
            self._p19_duplex_blocked_us = self._sealed_p19_diag_p50(p19_params, "hccl_duplex_4k_blocked_ns") / NS_PER_US_TOPO
            self._p19_duplex_eager_us = self._sealed_p19_diag_p50(p19_params, "hccl_duplex_4k_ns") / NS_PER_US_TOPO
            p19_bw = self._sealed_p19_diag_p50(p19_params, "hccl_duplex_bytes_per_s")
            if p19_bw > 0.0:
                self._p19_duplex_bytes_per_ms = p19_bw / MS_PER_S
            self._p19_uncoalesced_duplex = bool(
                self.p18_eager_startup_us > 0.0
                and self._p19_duplex_eager_us > self.p18_eager_startup_us * P19_UNCOALESCED_OP_RATIO
            )

    def _matches_p18_scope(self, group: tuple[int, ...]) -> bool:
        p18_scope = self._p18_measured_scope or ("package" if self._dies_per_package >= 2 else "node")
        same_pkg = len({self._package_key(r) for r in group}) == 1
        same_node = len({self._node_of(r) for r in group}) == 1
        if p18_scope == "package":
            return same_pkg
        if p18_scope == "node":
            return same_node and not (self._dies_per_package >= 2 and same_pkg)
        return True

    def p18_matched_startup_us(self, collective: str, group: tuple[int, ...], *, stack: str) -> float:
        """Return P18 startup only when collective, world size, route scope, and stack all match."""
        if not group or len(group) < 2:
            return 0.0
        if self._p18_measured_collective and collective != self._p18_measured_collective:
            return 0.0
        if self._p18_measured_world_size > 0 and len(group) != self._p18_measured_world_size:
            return 0.0
        if not self._matches_p18_scope(group):
            return 0.0
        if stack == "eager":
            return float(self.p18_eager_startup_us)
        return float(self.p18_startup_us)

    def p19_matched_duplex_startup_us(self, collective: str, group: tuple[int, ...], *, stack: str) -> float:
        """Return sealed P19 HCCL duplex startup when the physical scope matches."""
        if not self._has_primitive_table or not group or len(group) < 2:
            return 0.0
        if not self._matches_p18_scope(group):
            return 0.0
        raw_us = self._p19_duplex_eager_us if stack == "eager" else self._p19_duplex_blocked_us
        if raw_us <= 0.0:
            return 0.0
        extra_peers = max(0, len(group) - 2)
        if stack == "eager":
            extra_peer_us = float(extra_peers) * self._p15_host_submit_us
        else:
            extra_peer_us = float(extra_peers) * (
                self._p15_issue_interval_us + STARS_SYNC_EVENTS_PER_STEP * self._p16_stream_event_us
            )
        if not self._p19_uncoalesced_duplex:
            # Coalesced batch_isend_irecv enqueues 2 P2P descriptors vs 1 collective op.
            single_op_us = max(0.0, raw_us - self._p15_host_submit_us) if stack == "eager" else raw_us
            return float(single_op_us + extra_peer_us)
        per_op_us = raw_us / DUPLEX_SERIAL_OPS
        barrier_us = (
            STARS_SYNC_EVENTS_PER_STEP * self._p16_stream_event_us
            + self._p15_issue_interval_us
            + self._vector_invocation_us
        )
        reduce_us = self._vector_invocation_us if (collective == "allreduce" and stack != "eager") else 0.0
        return float(per_op_us + barrier_us + reduce_us + extra_peer_us)

    def collective_device_step_us(
        self, collective: str, group: tuple[int, ...], *,
        alpha_us: float, stack: str, phase_count: int = 1, total_rounds: int = 1,
        startup_scope: str = "step",
    ) -> float:
        """Derive collective startup from sealed P15/P16/P17/P20 STARS/AIV steps."""
        if not self._has_primitive_table or not group or len(group) < 2:
            return 0.0
        ev = self._p16_stream_event_us
        first = self._p15_aiv_first_issue_us or self._p15_issue_interval_us
        issue = self._p15_issue_interval_us
        vec = self._vector_invocation_us
        if ev <= 0.0 and first <= 0.0 and vec <= 0.0:
            return 0.0
        min_ar_rounds = ALLREDUCE_DEVICE_PHASES if (collective == "allreduce" and len(group) == 2) else 1
        eff_rounds = max(int(total_rounds), int(phase_count), min_ar_rounds)
        scope = str(startup_scope or "step").strip().lower()
        if scope == "phase":
            entry_us = STARS_SYNC_EVENTS_PER_STEP * ev + first + issue
            round_comm_us = STARS_SYNC_EVENTS_PER_STEP * ev
            alpha_total_us = float(max(1, int(phase_count))) * float(alpha_us)
        elif scope == "collective":
            entry_us = STARS_SYNC_EVENTS_PER_STEP * ev + first + issue
            round_comm_us = STARS_SYNC_EVENTS_PER_STEP * ev
            alpha_total_us = float(alpha_us)
        else:
            entry_us = (
                STARS_SYNC_EVENTS_PER_STEP * ev
                + first
                + issue
                + float(alpha_us)
            )
            round_comm_us = STARS_SYNC_EVENTS_PER_STEP * ev + float(alpha_us)
            alpha_total_us = 0.0
        kernel_us = vec * float(ALLREDUCE_DEVICE_PHASES if collective == "allreduce" else 1)
        total_us = entry_us + float(eff_rounds) * round_comm_us + alpha_total_us + kernel_us
        if stack == "eager":
            host_sync_us = self._p17_eager_sync_us or self._p20_pinned_d2h_startup_us
            eff_phases = max(1, int(phase_count))
            rendezvous_depth = max(1, math.ceil(math.log2(len(group))))
            total_us += (
                float(len(group) - 1) * self._p15_host_submit_us
                + float(eff_phases + rendezvous_depth) * host_sync_us
            )
        return float(total_us)

    def has_calibrated_p18_allreduce_rate(self) -> bool:
        """True when sealed P18 link capacity was calibrated on 2-phase AllReduce."""
        return bool(self._p18_link_bytes_per_ms > 0.0 and self._p18_measured_collective == "allreduce")

    def p18_allreduce_rate_applies(
        self,
        collective: str,
        group: tuple[int, ...],
        *,
        algorithm: str = "",
        scope: str = "",
    ) -> bool:
        """True only when P18 AllReduce rate calibration priced this group's scope."""
        del algorithm
        if not self.has_calibrated_p18_allreduce_rate():
            return False
        if collective not in {"allreduce", "reduce_scatter"}:
            return False
        if not group or len(group) < 2:
            return False
        resolved_scope = scope or self.scope_for_group(group)
        if resolved_scope == "cross_node" and self._p18_measured_scope in {"", "package", "node"}:
            return False
        if resolved_scope in {"package", "node"}:
            return True
        return bool(self._matches_p18_scope(group))

    def single_peer_flow_cap_bytes_per_ms(self, link_name: str) -> float | None:
        """Per-die SDMA rate cap bound explicitly from Hardware/PrimitiveTable declarations."""
        del link_name
        raise NotImplementedError(
            "LEGACY_PURGED: deleted covert _dies_per_package >= 2 per-peer flow cap; "
            "bind per-peer flow ceilings explicitly from Hardware/PrimitiveTable (S5-044)"
        )

    def link_protocol_bands(self, link_name: str):
        """Return piecewise ``CollectiveBandSpec``s declared on link or channel."""
        link = self.links[link_name]
        if link.protocol_bands:
            return link.protocol_bands
        ch_bands = self._channel_bands.get(link.channel, ())
        if not ch_bands:
            return ()
        ch_decl = self._channel_decls.get(link.channel)
        if ch_decl is not None:
            raw_alpha = ch_decl.get("latency_us", ch_decl.get("alpha_us"))
            if raw_alpha is not None and not math.isclose(
                float(link.latency_us), float(raw_alpha), rel_tol=1e-6, abs_tol=1e-9
            ):
                return ()
        return ch_bands

    def port_key(self, link, rank: int, direction: str) -> PortKey:
        """Canonical physical ``PortKey`` for ``(link, rank, direction)``."""
        return link_port_key(
            link, int(rank), self._node_of(rank), direction,
            channel_decl=self._channel_decls.get(link.channel),
        )

    def pair_key(self, link, source: int, destination: int) -> PairKey:
        """Canonical physical ``PairKey`` for ``(link, source, destination)``."""
        return link_pair_key(
            link, int(source), int(destination),
            channel_decl=self._channel_decls.get(link.channel),
        )

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
        from ..collectives.startup import resolve_collective_startup

        metadata = dict(op.metadata)
        family = str(metadata.get("collective", op.family)).strip().lower()
        return resolve_collective_startup(
            family=family, group_size=int(group_size), metadata=metadata,
            runtime_mode=self.collective_startup_mode,
            topology_protocols=self.collective_protocols)

    def direct_mesh_groups(self, op, replica):
        """Declared single-node full all-to-all pair mesh for this invocation."""
        metadata = dict(op.metadata)
        raw_family = str(metadata.get("collective", op.family)).strip().lower()
        family = _FAMILY_ALIASES.get(raw_family, raw_family)
        if family not in DIRECT_MESH_FAMILIES or "peer_traffic" in metadata:
            return family, ()
        axis = metadata.get("group_axis", "tp")
        stage = int(metadata.get("source_stage", op.stage))
        if not 0 <= stage < self.parallel.pp:
            raise ModelingError(f"direct_mesh_source_stage_out_of_range:{stage}:pp={self.parallel.pp}")
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
                    path = self.route(source, destination)
                    if len(path) != 1:
                        return family, ()
                    if self.links[path[0][0]].semantics != "per_pair":
                        return family, ()
        return family, groups

    def declares_direct_mesh(self, op, replica) -> bool:
        """True when this invocation declares the direct K_8 pair mesh without calibrated bands."""
        _, groups = self.direct_mesh_groups(op, replica)
        if not groups:
            return False
        if self._has_primitive_table:
            from .protocol import _candidate_bands, _infer_group_scope
            if all(_candidate_bands(ranks, self, _infer_group_scope(ranks, self)) for ranks in groups):
                return False
        return True

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

    @staticmethod
    def _oneshot_key(op, replica) -> tuple[object, int]:
        op_id = str(getattr(op, "op_id", ""))
        return (op_id if op_id else id(op), int(replica))

    def bind_oneshot_plan(self, op, replica, plan, *, volume=None) -> None:
        """Publish the executed one-shot plan for an invocation."""
        if len(self._oneshot_plans) >= MAX_ROUTE_CACHE_ENTRIES:
            self._oneshot_plans.clear()
        bound_vol = None if volume is None else float(volume)
        self._oneshot_plans[self._oneshot_key(op, replica)] = (plan, bound_vol)

    def clear_oneshot_plan(self, op, replica) -> None:
        if self._oneshot_plans:
            self._oneshot_plans.pop(self._oneshot_key(op, replica), None)

    def bound_oneshot_plan(self, op, replica, *, volume=None):
        if not self._oneshot_plans:
            return None
        entry = self._oneshot_plans.get(self._oneshot_key(op, replica))
        if entry is None:
            return None
        plan, bound_vol = entry
        if (volume is not None and bound_vol is not None
                and not math.isclose(bound_vol, float(volume), rel_tol=1e-12, abs_tol=1e-9)):
            return None
        return plan

    def mesh_phase_startup_ms(self, links) -> float:
        """Declared per-phase protocol startup; unknown stays zero."""
        return max((self._channel_startup_ms.get(str(link.channel), 0.0)
                    for link in links), default=0.0)

    def _oneshot_startup_ms(self) -> float:
        """Hardware-derived vector launch floor for a single-round one-shot collective.

        Derived strictly from the SKU's declared ``kernel_invocations["vector"]``
        minimum launch floor (and calibrated L0 ``P15``/``P16`` sync primitives
        when loaded) without chip-name branching or double-counting link
        propagation latency ``hop.latency_ms``.
        """
        if self._has_primitive_table and (
                self._p15_aiv_first_issue_us > 0.0 or self._p16_stream_event_us > 0.0):
            launch_us = (
                max(self._vector_invocation_us, self._p15_aiv_first_issue_us)
                + STARS_SYNC_EVENTS_PER_STEP * self._p16_stream_event_us
            )
            return launch_us / MS_PER_S
        return self._vector_invocation_us / MS_PER_S

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
        included_shared_switches: dict[SwitchKey, float] = {}
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
                for rank in link.members:
                    for direction in ("tx", "rx"):
                        shared_key = self._link_shared_ports.get(
                            (link.name, int(rank), direction)
                        )
                        if shared_key is not None:
                            included_shared_ports[shared_key] = (
                                self._shared_port_rates[shared_key]
                            )
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
                sw_key = self._link_shared_switches.get(link.name)
                if sw_key is not None:
                    included_shared_switches[sw_key] = self._shared_switch_rates[sw_key]
                else:
                    result.append(Resource(f"{self.namespace}link:{link.name}:switch_plane",
                                           link.switch_capacity_bytes_per_ms, "bytes"))
        if included_shared_ports:
            result.extend(
                build_shared_port_resources(
                    included_shared_ports, namespace=self.namespace
                )
            )
        if included_shared_switches:
            result.extend(
                build_shared_switch_resources(
                    included_shared_switches, namespace=self.namespace
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
        if scope == "any":
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
        declared = dict(metadata or {})
        if axis == "local_tp":
            from ..sharding.local_tp import local_tp_groups

            width = int(declared.get("local_tp_width", 0) or 0)
            stride = int(declared.get("local_tp_stride", 1) or 1)
            if width <= 0:
                raise ModelingError("local_tp_group_requires_declared_width")
            all_groups = local_tp_groups(
                self.machine, p, stage=stage, width=width, stride=stride)
            selected = tuple(group for group in all_groups
                             if any(rank in sources for rank in group))
            if not selected:
                raise ModelingError(f"local_tp_group_unmatched:replica={replica}:stage={stage}")
            return selected
        if axis in {"cp", "dcp"}:
            pcp_w = int(declared.get("dcp_pcp_factor") or p.dcp_pcp_factor)
            tp_w = int(declared.get("dcp_tp_gather") or p.dcp_tp_gather)
            dp_w = int(declared.get("dcp_dp_width") or p.dcp_dp_width)
            if dp_w > 1:
                first = (replica // dp_w) * dp_w
                return tuple(
                    tuple(
                        rank
                        for r in range(first, first + dp_w)
                        for copy in range(copy0, copy0 + pcp_w)
                        for rank in self.ranks(r, stage)[copy * p.tp + i:copy * p.tp + i + tp_w]
                    )
                    for copy0 in range(0, p.pcp, pcp_w)
                    for i in range(0, p.tp, tp_w)
                )
            if p.pcp > 1:
                return tuple(
                    tuple(
                        sources[copy * p.tp + lane]
                        for lane in range(i, i + tp_w)
                        for copy in range(copy0, copy0 + pcp_w)
                    )
                    for i in range(0, p.tp, tp_w)
                    for copy0 in range(0, p.pcp, pcp_w)
                )
            return tuple(sources[i:i + tp_w] for i in range(0, len(sources), tp_w))
        if axis == "dcp_tp":
            width = int(declared.get("dcp_tp_gather") or p.dcp_tp_gather)
            return tuple(
                sources[copy * p.tp + i:copy * p.tp + i + width]
                for copy in range(p.pcp)
                for i in range(0, p.tp, width)
            )
        if axis == "dcp_pcp":
            width = int(declared.get("dcp_pcp_factor") or p.dcp_pcp_factor)
            return tuple(
                sources[lane::p.tp][i:i + width]
                for lane in range(p.tp)
                for i in range(0, p.pcp, width)
            )
        if axis == "dcp_dp":
            count = int(declared.get("dcp_dp_width") or p.dcp_dp_width)
            first = (replica // count) * count
            return tuple(
                tuple(self.ranks(r, stage)[idx] for r in range(first, first + count))
                for idx in range(len(sources))
            )
        if axis == "dcp_a2a":
            tp_w = int(declared.get("dcp_tp_gather") or p.dcp_tp_gather)
            dp_w = int(declared.get("dcp_dp_width") or p.dcp_dp_width)
            if dp_w <= 1:
                return tuple(
                    sources[copy * p.tp + i:copy * p.tp + i + tp_w]
                    for copy in range(p.pcp)
                    for i in range(0, p.tp, tp_w)
                )
            first = (replica // dp_w) * dp_w
            return tuple(
                tuple(
                    rank
                    for r in range(first, first + dp_w)
                    for rank in self.ranks(r, stage)[copy * p.tp + i:copy * p.tp + i + tp_w]
                )
                for copy in range(p.pcp)
                for i in range(0, p.tp, tp_w)
            )
        if axis == "dp":
            first = (replica // p.dp) * p.dp
            return tuple(
                tuple(self.ranks(r, stage)[idx] for r in range(first, first + p.dp))
                for idx in range(len(sources))
            )
        if axis == "etp":
            from ..sharding.local_tp import expert_tp_groups

            return expert_tp_groups(self, replica=replica, stage=stage)
        if axis == "pcp":
            return tuple(sources[i::p.tp] for i in range(p.tp))
        if axis == "ep":
            if p.rank_table is not None:
                members = set(sources)
                return tuple(tuple(group) for group in p.rank_table
                             if members.intersection(group))
            ep_tp_w = max(1, p.tp // p.etp)
            ep_spans_pcp = p.pcp > 1 and p.ep > p.dp * ep_tp_w
            ep_intra_w = ep_tp_w * p.pcp if ep_spans_pcp else ep_tp_w
            ep_spans_dp = p.ep > ep_intra_w
            if ep_spans_dp or (ep_spans_pcp and p.etp > 1):
                count = max(1, p.ep // ep_intra_w)
                first = (replica // count) * count
                if ep_spans_pcp:
                    if p.etp <= 1:
                        return (tuple(rank for r in range(first, first + count)
                                      for rank in self.ranks(r, stage)),)
                    return tuple(
                        tuple(
                            rank
                            for r in range(first, first + count)
                            for copy in range(p.pcp)
                            for rank in self.ranks(r, stage)[copy * p.tp + off:copy * p.tp + off + ep_tp_w]
                        )
                        for off in range(0, p.tp, ep_tp_w)
                    )
                return tuple(
                    tuple(
                        rank
                        for r in range(first, first + count)
                        for rank in self.ranks(r, stage)[copy * p.tp + off:copy * p.tp + off + ep_tp_w]
                    )
                    for copy in range(p.pcp)
                    for off in range(0, p.tp, ep_tp_w)
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
            if "destination_stage" in metadata:
                stage = int(metadata["destination_stage"])
            else:
                stage = int(op.stage) + 1
            if not 0 <= stage < p.pp or stage == int(op.stage):
                raise ModelingError(
                    f"pp_p2p_invalid_terminal_stage:source={op.stage}:dest={stage}:pp={p.pp}"
                )
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
        else:
            if self._has_endpoint_rate(link, source):
                names.append(f"{self.namespace}link:{link.name}:tx:port{source}")
            if self._has_endpoint_rate(link, destination):
                names.append(f"{self.namespace}link:{link.name}:rx:port{destination}")
        tx_shared = self._link_shared_ports.get((link.name, int(source), "tx"))
        if tx_shared is not None:
            names.append(tx_shared.resource_name(self.namespace))
        rx_shared = self._link_shared_ports.get((link.name, int(destination), "rx"))
        if rx_shared is not None:
            names.append(rx_shared.resource_name(self.namespace))
        if link.switch_capacity_bytes_per_ms is not None:
            sw_key = self._link_shared_switches.get(link.name)
            if sw_key is not None:
                names.append(sw_key.resource_name(self.namespace))
            else:
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
