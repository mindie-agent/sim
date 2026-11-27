"""Device-independent resource descriptions.

[Role] Describe compute rates, memory, buffers, submission, datapath capability
  declarations and physical links for one or more placement dies.
[Invariants] Vendor names are labels. The declared resource graph, rates and
  storage capacities determine execution. Precision peaks require a declared
  native compute mode. Shared L2 enforces ``buffer_units.l2 == 1``. Host
  transfer channels expose solo endpoint ceilings and physical root-complex
  ceilings separately, never pre-dividing a solo transfer by configured die
  count when sibling dies are idle.
[Timeouts & Locks] Pure metadata parsing, no driver calls or device contexts.
[Side-Effect Policy] Immutable input records.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence, Set as AbstractSet
from copy import deepcopy
from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType
from typing import Any

from .common import ModelingError, _positive_float
from .compute_arch import ComputeArchSpec
from .compute_peak_aliases import (
    matrix_peak, validate_compute_peak_aliases,
)
from .device_input import (
    DispatchSyncSpec,
    MemoryAccessSpec,
    SubtileWaveSpec,
    machine_device_defaults,
    merge_device_inputs,
    parse_dispatch_sync,
    parse_memory_access,
    parse_subtile_waves,
    resolve_device_placements,
    validate_hbm_rw_alpha,
)
from .hardware_links import (
    GBPS_TO_BYTES_PER_MS,
    Link,
    _LINK_BANDWIDTH_KEYS,
    _PRESERVED_TOPOLOGY_KEYS,
    _collective_protocols,
    _link_bandwidth,
    _network_override_keys,
    _neutral_transfer_channel_links,
    apply_network_overrides,
    calibration_warnings,
)
from .invocation import InvocationTiming
from .parallel import DieIndex, GlobalRank, ParallelSpec, RankCoord, ReplicaLocalRank
from .physical_mechanisms import (
    PhysicalMechanismSpec, default_physical_mechanisms,
    parse_physical_mechanisms,
)

BITS_PER_BYTE = 8
DEFAULT_ROOT_COMPLEX_ID = "rc0"
_HOST_DMA_SCOPES = frozenset({"host", "device_local"})
_HOST_DMA_CHANNEL_TAGS = ("host", "h2d", "d2h", "pcie", "dma")

NATIVE_WORK_UNITS = {
    **dict.fromkeys(("hbm", "l2", "mte1", "mte2", "mte2_aic", "mte2_aiv", "mte3",
                     "fixpipe", "host_memory", "link"), "bytes"),
    "cube": "flops", "vector": "ops", "host_vector": "ops", "host": "ms",
    "scalar_store_issue": "stores",
}

_HOST_CHANNEL_ALIASES = {"host_to_device": "h2d", "device_to_host": "d2h"}
_QUANTITY_SCOPES = {
    "memory": "per_device",
    "buffers": "per_owner",
    "compute_peaks": "device_total",
    "rates": "device_total",
    "resource_units": "owners_per_device",
    "buffer_units": "owners_per_device",
}


def _optional_pos_rate(value: Any, label: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, f"{label}_invalid", allow_zero=False)


def _channel_scope_map(
    transfer_channels: Mapping[str, Any] | Sequence[tuple[str, Any]],
) -> dict[str, str]:
    entries = (
        transfer_channels.items()
        if isinstance(transfer_channels, Mapping)
        else transfer_channels
    )
    scopes: dict[str, str] = {}
    for channel_name, raw_items in entries:
        record = dict(raw_items) if isinstance(raw_items, (Mapping, Sequence)) else {}
        raw_scope = record.get("scope")
        if raw_scope is not None:
            scopes[str(channel_name)] = str(raw_scope).strip().lower()
    return scopes


def _is_host_or_dma_link(
    link: Link,
    channel_scopes: Mapping[str, str] | None = None,
) -> bool:
    scope = str(link.scope or "").strip().lower()
    if not scope and channel_scopes is not None:
        scope = channel_scopes.get(str(link.channel), "")
    if scope in _HOST_DMA_SCOPES:
        return True
    channel = str(link.channel or "").strip().lower()
    return any(tag in channel for tag in _HOST_DMA_CHANNEL_TAGS)


def _induced_links_for(
    links: Sequence[Link],
    selected_ranks: AbstractSet[GlobalRank],
    *,
    transfer_channels: Mapping[str, Any] | Sequence[tuple[str, Any]] = (),
) -> tuple[Link, ...]:
    selected = {int(rank) for rank in selected_ranks}
    channel_scopes = _channel_scope_map(transfer_channels) if transfer_channels else {}
    induced: list[Link] = []
    for link in links:
        kept = tuple(int(member) for member in link.members if int(member) in selected)
        if not kept:
            continue
        if len(kept) < 2 and not (
            len(link.members) == 1 or _is_host_or_dma_link(link, channel_scopes)
        ):
            continue
        if kept == link.members:
            induced.append(link)
            continue
        kept_set = set(kept)
        induced.append(replace(
            link,
            members=kept,
            endpoint_rates=tuple(
                (int(rank), rate)
                for rank, rate in link.endpoint_rates
                if int(rank) in kept_set
            ),
            endpoint_ids=tuple(
                (int(rank), endpoint_id)
                for rank, endpoint_id in link.endpoint_ids
                if int(rank) in kept_set
            ),
        ))
    return tuple(induced)


@dataclass(frozen=True)
class Device:
    rank: int
    memory_bytes: int
    compute_peaks: tuple[tuple[str, float], ...]
    rates: tuple[tuple[str, float], ...]
    buffers: tuple[tuple[str, int], ...] = ()
    compute_units: int = 1
    launch_us: float = 0
    node: int = 0
    package: int = 0
    resource_units: tuple[tuple[str, int], ...] = ()
    buffer_units: tuple[tuple[str, int], ...] = ()
    service_modes: tuple[tuple[str, tuple[str, float]], ...] = ()
    work_units: tuple[tuple[str, str], ...] = ()
    kernel_invocations: tuple[tuple[str, InvocationTiming], ...] = ()
    store_queues: tuple[tuple[str, tuple[tuple[str, float], ...]], ...] = ()
    compute_arch: ComputeArchSpec = field(default_factory=ComputeArchSpec)
    linked_rate_overrides: tuple[str, ...] = ()
    instruction_mode_overrides: tuple[str, ...] = ()
    instruction_unit_overrides: tuple[str, ...] = ()
    sku: str = ""
    arch_family: str = ""
    compute_peak_aliases: tuple[tuple[str, str], ...] = ()
    physical_mechanisms: PhysicalMechanismSpec = field(
        default_factory=default_physical_mechanisms)
    root_complex_id: str = DEFAULT_ROOT_COMPLEX_ID
    simt_ub_carveout_bytes: int = 0
    subtile_wave: tuple[SubtileWaveSpec, ...] = ()
    dispatch_sync: DispatchSyncSpec | None = None
    hbm_rw_alpha: float = 0.0
    hbm_read_bytes_per_s: float | None = None
    hbm_write_bytes_per_s: float | None = None
    memory_access: MemoryAccessSpec | None = None
    observed_aicore_freq_mhz: float | None = None

    @property
    def capability(self) -> ComputeArchSpec:
        return self.compute_arch

    @property
    def resource_signature(self) -> tuple[Any, ...]:
        """Hashable physical identity without rank/node/package labels."""
        arch = self.compute_arch
        return (self.sku, self.arch_family, self.memory_bytes, self.compute_peaks,
                self.compute_peak_aliases,
                self.rates, self.buffers, self.compute_units, self.resource_units,
                self.buffer_units, self.service_modes, self.work_units,
                self.launch_us,
                tuple((name, timing._cache_sig)
                      for name, timing in self.kernel_invocations),
                self.store_queues,
                arch.preset,
                arch.clock_hz,
                tuple((item.name, item.unit, item.owner_axis, item.per_owner_rate,
                       item.scale_with, item.resource, item.clock_hz,
                       item.cycles_per_unit)
                      for item in arch.instruction_services),
                tuple((item.name, item.bits, item.roles, item.alignment_bytes)
                      for item in arch.representations),
                tuple((item.name, item.kind, item.operands, item.unit,
                       item.accumulators, item.implementations)
                      for item in arch.native_modes),
                tuple((item.source, item.target, item.implementation, item.lossless)
                      for item in arch.casts),
                tuple((item.name, item.source, item.target, item.roles, item.unit,
                       item.executor, item.reach, item.buffer, item.sync,
                       item.alignment_bytes, item.representations)
                      for item in arch.paths),
                tuple((item.name, item.version, item.provides, item.conditions)
                      for item in arch.runtimes),
                self.physical_mechanisms.signature,
                self.simt_ub_carveout_bytes, self.subtile_wave, self.dispatch_sync,
                self.hbm_rw_alpha, self.hbm_read_bytes_per_s,
                self.hbm_write_bytes_per_s, self.memory_access,
                self.observed_aicore_freq_mhz)

    @property
    def hbm_bytes(self) -> int:
        return self.memory_bytes

    @property
    def cube_flops_per_s(self) -> float:
        return self.peak("bf16")

    @property
    def aic_count(self) -> int:
        return self.compute_units

    def peak(self, dtype: str) -> float:
        return matrix_peak(dtype, self.compute_peaks, self.compute_peak_aliases, self.compute_arch)

    def units(self, resource: str) -> int:
        declared = dict(self.resource_units)
        if resource in declared:
            return declared[resource]
        if resource == "l2" and (
            "l2" in dict(self.buffer_units) or "l2" in dict(self.buffers)
        ):
            return dict(self.buffer_units).get("l2", 1)
        return self.compute_units

    def pool_units(self, pool: str) -> int:
        return dict(self.buffer_units).get(pool, self.compute_units)

    def owner_count(self, resource: str) -> int:
        """Owners of ``resource`` on this device (never a package total)."""
        return self.units(resource)

    def total_rate(self, resource: str) -> float:
        """Declared device-total service of one resource, in its own unit."""
        return dict(self.rates)[resource]

    def owner_rate(self, resource: str) -> float:
        """Per-owner service; idle siblings do not lend their capacity."""
        count = self.owner_count(resource)
        if count < 1:
            raise ModelingError(f"resource_owner_count_nonpositive:{resource}")
        return self.total_rate(resource) / count

    def total_buffer_bytes(self, pool: str) -> int:
        """Per-owner pool capacity times its declared owner count."""
        return self.pool_capacity(pool) * self.pool_units(pool)

    def pool_capacity(self, pool: str) -> int:
        """Declared capacity of one pool instance (per owner)."""
        try:
            return dict(self.buffers)[pool]
        except KeyError as exc:
            raise ModelingError(f"buffer_pool_undeclared:{pool}") from exc

    @staticmethod
    def quantity_scope(quantity: str) -> str:
        """Return the declared accounting scope of a device quantity."""
        try:
            return _QUANTITY_SCOPES[quantity]
        except KeyError as exc:
            raise ModelingError(f"device_quantity_scope_unknown:{quantity}") from exc

    def to_dict(self) -> dict[str, Any]:
        derived_rates = {s.name for s in self.compute_arch.instruction_services
                         if s.has_linked_supply
                         and s.name not in self.linked_rate_overrides}
        derived_modes = {s.name for s in self.compute_arch.instruction_services
                         if s.resource and s.name not in self.instruction_mode_overrides}
        declared_pipes = {s.name: s.resource for s in self.compute_arch.instruction_services}
        derived_units = {s.name for s in self.compute_arch.instruction_services
                         if s.has_linked_supply
                         and s.name not in self.instruction_unit_overrides}
        result = {"rank": self.rank, "memory_bytes": self.memory_bytes,
                  "compute_peaks": dict(self.compute_peaks),
                  "rates": {k: v for k, v in self.rates if k not in derived_rates},
                  "buffers": dict(self.buffers), "compute_units": self.compute_units,
                  "launch_us": self.launch_us, "node": self.node, "package": self.package,
                  "resource_units": {k: v for k, v in self.resource_units if k not in derived_units},
                  "buffer_units": dict(self.buffer_units),
                  **({"sku": self.sku} if self.sku else {}),
                  **({"arch_family": self.arch_family} if self.arch_family else {}),
                  "service_modes": {name: {"resource": role,
                                           **({"rate": rate} if name not in derived_modes else {})}
                                    for name, (role, rate) in self.service_modes
                                    if name not in derived_modes or role != declared_pipes[name]},
                  **({"compute_arch": self.compute_arch.to_dict()}
                     if self.compute_arch.declared else {}),
                  **({"compute_peak_aliases": dict(self.compute_peak_aliases)}
                     if self.compute_peak_aliases else {}),
                  "physical_mechanisms": self.physical_mechanisms.to_dict(),
                  **({"work_units": dict(self.work_units)} if self.work_units else {}),
                  **({"store_queues": {name: dict(queue) for name, queue in self.store_queues}}
                     if self.store_queues else {}),
                  **({"kernel_invocations": {key: timing.to_dict()
                                            for key, timing in self.kernel_invocations}}
                     if self.kernel_invocations else {})}
        if self.root_complex_id != DEFAULT_ROOT_COMPLEX_ID:
            result["root_complex_id"] = self.root_complex_id
        if self.simt_ub_carveout_bytes:
            result["simt_ub_carveout_bytes"] = self.simt_ub_carveout_bytes
        if self.subtile_wave:
            result["subtile_wave"] = [spec.to_dict() for spec in self.subtile_wave]
        if self.dispatch_sync is not None:
            result["dispatch_sync"] = self.dispatch_sync.to_dict()
        if self.hbm_rw_alpha:
            result["hbm_rw_alpha"] = self.hbm_rw_alpha
        if self.hbm_read_bytes_per_s is not None:
            result["hbm_read_bytes_per_s"] = self.hbm_read_bytes_per_s
        if self.hbm_write_bytes_per_s is not None:
            result["hbm_write_bytes_per_s"] = self.hbm_write_bytes_per_s
        if self.memory_access is not None:
            result["memory_access"] = self.memory_access.to_dict()
        if self.observed_aicore_freq_mhz is not None:
            result["observed_aicore_freq_mhz"] = self.observed_aicore_freq_mhz
            # The serialized supplies already use this clock.
            result["aicore_freq_mhz"] = self.observed_aicore_freq_mhz
        return result


@dataclass(frozen=True)
class PhysicalRankMap:
    """Logical placement order and the node inventory indexed by physical rank."""

    parallel: ParallelSpec
    logical_to_physical: tuple[int, ...]
    _inventory: Mapping[int, Device] = field(repr=False)
    _links: tuple[Link, ...] = field(default=(), repr=False, compare=False)
    _transfer_channels: tuple[tuple[str, tuple[tuple[str, object], ...]], ...] = field(
        default=(), repr=False, compare=False,
    )
    _die_index_by_rank: Mapping[int, DieIndex] = field(
        init=False, repr=False, compare=False,
    )
    _physical_to_logical: Mapping[int, int] = field(
        init=False, repr=False, compare=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_die_index_by_rank",
            MappingProxyType({
                int(rank): DieIndex(idx)
                for idx, rank in enumerate(self._inventory)
            }),
        )
        object.__setattr__(
            self,
            "_physical_to_logical",
            MappingProxyType({
                int(rank): idx
                for idx, rank in enumerate(self.logical_to_physical)
            }),
        )

    @property
    def active_ranks(self) -> frozenset[GlobalRank]:
        return frozenset(GlobalRank(int(rank)) for rank in self.logical_to_physical)

    @property
    def spare_ranks(self) -> frozenset[GlobalRank]:
        active = {int(rank) for rank in self.logical_to_physical}
        return frozenset(
            GlobalRank(int(rank))
            for rank in self._inventory.keys()
            if int(rank) not in active
        )

    def die_for_rank(self, rank: int) -> Device:
        try:
            return self._inventory[int(rank)]
        except KeyError as exc:
            raise ModelingError(f"hardware_placement_unknown:{rank}") from exc

    def stage_ranks(self, replica: int, stage: int = 0) -> tuple[GlobalRank, ...]:
        p = self.parallel
        if not (0 <= replica < p.external_dp * p.dp and 0 <= stage < p.pp):
            raise ModelingError(f"placement_coordinate_unknown:{replica}:{stage}")
        width = p.tp * p.pcp
        start = (replica * p.pp + stage) * width
        return tuple(
            GlobalRank(int(rank))
            for rank in self.logical_to_physical[start:start + width]
        )

    def ep_group_ranks(
        self, replica: int, stage: int = 0, *, lane: int = 0,
    ) -> tuple[GlobalRank, ...]:
        """Return the EP group containing one lane of the replica and stage."""
        p = self.parallel
        local = self.stage_ranks(replica, stage)
        if not 0 <= lane < len(local):
            raise ModelingError(f"placement_lane_unknown:{lane}")
        if p.rank_table is not None:
            for group in p.rank_table:
                if local[lane] in group:
                    return tuple(GlobalRank(int(rank)) for rank in group)
            raise ModelingError(f"rank_table_rank_unassigned:{local[lane]}")
        coords = p.ep_group_coords(replica, stage, ReplicaLocalRank(lane))
        stage_cache: dict[tuple[int, int], tuple[GlobalRank, ...]] = {
            (replica, stage): local,
        }
        ranks: list[GlobalRank] = []
        for rep, stg, local_lane in coords:
            key = (rep, stg)
            stage_tuple = stage_cache.get(key)
            if stage_tuple is None:
                stage_tuple = self.stage_ranks(rep, stg)
                stage_cache[key] = stage_tuple
            ranks.append(stage_tuple[int(local_lane)])
        return tuple(ranks)

    def coord_for(
        self,
        replica: int,
        stage: int,
        local_rank: ReplicaLocalRank | int,
    ) -> RankCoord:
        """Project logical ``(replica, stage, local_rank)`` to a typed ``RankCoord``."""
        p = self.parallel
        local = self.stage_ranks(replica, stage)
        lane = int(local_rank)
        if not 0 <= lane < len(local):
            raise ModelingError(f"placement_lane_unknown:{lane}")
        global_rank = local[lane]
        die_index = self._die_index_by_rank.get(int(global_rank))
        if die_index is None:
            raise ModelingError(f"hardware_placement_unknown:{global_rank}")
        pcp_lane, tp_lane = divmod(lane, p.tp)
        if p.cp <= 1:
            dcp_lane = 0
        else:
            pcp_f = p.dcp_pcp_factor
            tp_g = p.dcp_tp_gather
            dp_w = p.dcp_dp_width
            dp_i = replica % p.dp
            if dp_w > 1:
                dcp_lane = (
                    (dp_i % dp_w) * (pcp_f * tp_g)
                    + (pcp_lane % pcp_f) * tp_g
                    + (tp_lane % tp_g)
                ) % p.cp
            else:
                dcp_lane = ((tp_lane % tp_g) * pcp_f + (pcp_lane % pcp_f)) % p.cp
        if p.ep <= 1:
            ep_rank = 0
        else:
            ep_group = self.ep_group_ranks(replica, stage, lane=lane)
            ep_rank = ep_group.index(global_rank)
        return RankCoord(
            global_rank=global_rank,
            die_index=die_index,
            replica=int(replica),
            stage=int(stage),
            local_rank=ReplicaLocalRank(lane),
            tp_lane=tp_lane,
            pcp_lane=pcp_lane,
            dcp_lane=dcp_lane,
            ep_rank=ep_rank,
        )

    def coord_for_physical_rank(self, rank: GlobalRank) -> RankCoord:
        """Project one active physical ``GlobalRank`` back to its ``RankCoord``."""
        rank_int = int(rank)
        logical_idx = self._physical_to_logical.get(rank_int)
        if logical_idx is None:
            raise ModelingError(f"hardware_placement_unknown:{rank_int}")
        p = self.parallel
        width = p.tp * p.pcp
        stage_slot, lane = divmod(logical_idx, width)
        replica, stage = divmod(stage_slot, p.pp)
        return self.coord_for(replica, stage, ReplicaLocalRank(lane))

    def physical_ranks_for_cohort(
        self, group: Sequence[int],
    ) -> tuple[GlobalRank, ...]:
        """Return actual physical ``GlobalRank`` IDs covering ``group`` replicas."""
        p = self.parallel
        seen: set[int] = set()
        ranks: list[GlobalRank] = []
        for replica in sorted({int(rep) for rep in group}):
            for stage in range(p.pp):
                for rank in self.stage_ranks(replica, stage):
                    rank_int = int(rank)
                    if rank_int not in seen:
                        seen.add(rank_int)
                        ranks.append(GlobalRank(rank_int))
        return tuple(ranks)

    def induced_links_for(
        self,
        selected_ranks: AbstractSet[GlobalRank],
        *,
        links: Sequence[Link] | None = None,
    ) -> tuple[Link, ...]:
        """Project links onto ``selected_ranks``, rewriting surviving members and endpoints."""
        source_links = self._links if links is None else links
        return _induced_links_for(
            source_links,
            selected_ranks,
            transfer_channels=self._transfer_channels,
        )


@dataclass(frozen=True)
class Hardware:
    name: str
    dies: tuple[Device, ...]
    links: tuple[Link, ...] = ()
    arch_family: str = "generic"
    hardware_key: str = ""
    ranks_per_node: int = 1
    dies_per_package: int = 1
    sku: str = ""
    assumptions: tuple[tuple[str, str], ...] = ()
    transfer_channels: tuple[tuple[str, tuple[tuple[str, object], ...]], ...] = field(
        default=(), compare=False, repr=False
    )
    topology_options: tuple[tuple[str, object], ...] = field(
        default=(), compare=False, repr=False
    )
    source: Mapping[str, object] = field(
        default_factory=dict, compare=False, repr=False
    )
    collective_protocols: tuple[tuple[str, str], ...] = ()

    @property
    def homogeneous(self) -> bool:
        return len({device.resource_signature for device in self.dies}) <= 1

    def _require_homogeneous(self, query: str) -> None:
        if not self.homogeneous:
            raise ModelingError(f"hardware_resources_are_per_placement:{query}")

    @property
    def compute_peaks(self) -> tuple[tuple[str, float], ...]:
        self._require_homogeneous("compute_peaks")
        return self.dies[0].compute_peaks

    def per_placement_peaks(self, dtype: str) -> dict[int, float]:
        return {device.rank: device.peak(dtype) for device in self.dies}

    def peak_flops_for(self, dtype: str) -> float:
        self._require_homogeneous("peak_flops_for")
        return min(device.peak(dtype) for device in self.dies)

    def declared_peak_key(self, dtype: str) -> str | None:
        return dtype if dtype in dict(self.compute_peaks) else None

    def package_memory_bytes(self, package: int, *, node: int = 0) -> int:
        """Total HBM in the package with this node-local identity."""
        return sum(device.memory_bytes for device in self.dies
                   if (device.node, device.package) == (node, package))

    def node_memory_bytes(self, node: int) -> int:
        return sum(device.memory_bytes for device in self.dies
                   if device.node == node)

    def placement_sku(self, rank: int) -> str:
        for device in self.dies:
            if device.rank == rank:
                return device.sku
        raise ModelingError(f"hardware_placement_unknown:{rank}")

    @property
    def node_peak_bw_gbps(self) -> dict[str, float]:
        """Declared per-root-complex / host-fabric directional bandwidth ceilings."""
        result: dict[str, float] = {}
        for channel, entries in self.transfer_channels:
            value = dict(entries).get("node_peak_bw_gbps")
            if value is not None:
                result[str(channel)] = _positive_float(
                    value, f"transfer_channel_node_peak_invalid:{channel}")
        return result

    def channel_node_peak_bw_gbps(self, channel: str) -> float | None:
        """Return the declared root-complex directional ceiling (GB/s)."""
        key = _HOST_CHANNEL_ALIASES.get(str(channel), str(channel))
        return self.node_peak_bw_gbps.get(key)

    def host_root_complex_ceiling_gbps(self, channel: str) -> float | None:
        """Root-complex directional ceiling (GB/s) for Engine contention rows."""
        return self.channel_node_peak_bw_gbps(channel)

    def host_solo_bw_gbps(self, channel: str) -> float | None:
        """Per-die solo endpoint ceiling (GB/s) without idle-die division."""
        key = _HOST_CHANNEL_ALIASES.get(str(channel), str(channel))
        entries = dict(self.transfer_channels).get(key)
        if entries is None:
            return None
        record = dict(entries)
        if "effective_uni_bw_gbps" in record and record["effective_uni_bw_gbps"] is None:
            return None
        limits = [_positive_float(record[k], f"transfer_channel_bw_invalid:{channel}")
                  for k in ("solo_uni_bw_gbps", "solo_effective_uni_bw_gbps")
                  if record.get(k) is not None]
        if any(record.get(k) is not None for k in _LINK_BANDWIDTH_KEYS):
            _, effective, _ = _link_bandwidth(record, channel)
            limits.append(effective / GBPS_TO_BYTES_PER_MS)
        return min(limits, default=None)

    def effective_host_channel_bw_gbps(
        self, channel: str, *, active_ranks_on_node: int | None = None,
    ) -> float | None:
        """Per-rank host channel ceiling; never pre-divides by idle configured dies.

        When ``active_ranks_on_node`` is omitted, returns the solo endpoint
        ceiling bounded by the root-complex ceiling so concurrent Engine flows
        perform actual contention without double-derating solo transfers.
        """
        per_die = self.host_solo_bw_gbps(channel)
        if per_die is None:
            return None
        node_peak = self.channel_node_peak_bw_gbps(channel)
        if node_peak is None:
            return float(per_die)
        if active_ranks_on_node is None:
            return min(float(per_die), float(node_peak))
        active_ranks = max(1, int(active_ranks_on_node))
        return min(float(per_die), float(node_peak) / float(active_ranks))

    def calibration_summary(self, rank: int = 0) -> dict[str, object]:
        device = next((item for item in self.dies if item.rank == rank), None)
        if device is None and self.dies:
            if rank != 0:
                raise ModelingError(f"hardware_placement_unknown:{rank}")
            device = self.dies[0]
        device_sku = (device.sku if device is not None else "") or self.sku or self.name
        source_map = dict(self.source or {})
        raw_cal = source_map.get("calibration")
        cal: dict[str, Any] = dict(raw_cal) if isinstance(raw_cal, Mapping) else {}
        if (device is not None and not self.homogeneous
                and device.sku and device.sku != self.sku and not cal):
            from mindie_sim.backends.ascend import profiles as ascend_profiles
            prof = ascend_profiles().get(device.sku)
            if isinstance(prof, Mapping):
                prof_source = prof.get("source")
                prof_cal = (
                    prof_source.get("calibration")
                    if isinstance(prof_source, Mapping) else None
                )
                if isinstance(prof_cal, Mapping):
                    cal = dict(prof_cal)
        calibrated_raw = cal.get("calibrated")
        if calibrated_raw is None and device is not None:
            hierarchy_prov = dict(device.physical_mechanisms.memory_hierarchy.provenance)
            calibrated_raw = bool(hierarchy_prov.get("calibrated", False))
        calibrated = bool(calibrated_raw)
        rate_provenance = str(
            cal.get("rate_provenance")
            or ("silicon_calibrated" if calibrated else "declared_nominal")
        )
        anchor_sku = str(
            cal.get("anchor_sku") or cal.get("source_sku") or device_sku or "declared"
        )
        return {
            "calibrated": calibrated,
            "rate_provenance": rate_provenance,
            "anchor_sku": anchor_sku,
        }

    def hardware_context(self, rank: int = 0) -> dict[str, Any]:
        device = next((item for item in self.dies if item.rank == rank), None)
        if device is None:
            raise ModelingError(f"hardware_placement_unknown:{rank}")
        cal = self.calibration_summary(rank=rank)
        return {
            "schema": "mindie-sim.hardware-context/1",
            "name": self.name, "sku": self.sku,
            "arch_family": self.arch_family, "homogeneous": self.homogeneous,
            "hardware_calibration": cal,
            "warnings": calibration_warnings(cal),
            "placement": {"rank": device.rank, "node": device.node,
                          "package": device.package,
                          "root_complex_id": device.root_complex_id},
            "device_sku": device.sku,
            "device_arch_family": device.arch_family,
            "memory_bytes": device.memory_bytes,
            "compute_units": device.compute_units,
            **({"observed_aicore_freq_mhz": device.observed_aicore_freq_mhz}
               if device.observed_aicore_freq_mhz is not None else {}),
            "compute_peaks": dict(device.compute_peaks),
            "compute_peak_aliases": dict(device.compute_peak_aliases),
            "rates": dict(device.rates),
            "resource_units": dict(device.resource_units),
            "buffers": dict(device.buffers),
            "buffer_units": dict(device.buffer_units),
            "service_modes": {
                name: {"resource": role, "rate": rate}
                for name, (role, rate) in device.service_modes},
            "work_units": dict(device.work_units),
            "launch_us": device.launch_us,
            "capability": device.compute_arch.to_dict(),
            **({"node_peak_bw_gbps": self.node_peak_bw_gbps}
               if self.node_peak_bw_gbps else {}),
        }

    def induced_links_for(self, selected_ranks: AbstractSet[GlobalRank]) -> tuple[Link, ...]:
        """Project hardware links onto ``selected_ranks``."""
        return _induced_links_for(
            self.links,
            selected_ranks,
            transfer_channels=self.transfer_channels,
        )

    def validate_placement(self, parallel: ParallelSpec) -> PhysicalRankMap:
        if parallel.world_size > len(self.dies):
            raise ModelingError("placement_exceeds_physical_devices")
        inventory = {device.rank: device for device in self.dies}
        if len(inventory) != len(self.dies):
            raise ModelingError("device_rank_duplicate")
        parallel.validate_expert_placement()
        active = tuple(inventory)[:parallel.world_size]
        placement = PhysicalRankMap(
            parallel,
            active,
            MappingProxyType(inventory),
            _links=self.links,
            _transfer_channels=self.transfer_channels,
        )
        if parallel.rank_table is not None:
            positions = {rank: pos for group in parallel.rank_table for pos, rank in enumerate(group)}
            unknown = positions.keys() - placement.active_ranks
            if unknown:
                raise ModelingError(f"rank_table_member_outside_placement:{sorted(unknown)}")
            width = parallel.tp * parallel.pcp
            coordinates = {rank: divmod(index // width, parallel.pp)
                           for index, rank in enumerate(active)}
            for group in parallel.rank_table:
                domains = {(replica // parallel.dp, stage)
                           for replica, stage in (coordinates[rank] for rank in group)}
                if len(domains) != 1:
                    raise ModelingError("rank_table_crosses_pipeline_or_replica")
            if parallel.etp > 1:
                for start in range(0, len(active), parallel.tp):
                    counts = Counter(positions[r] for r in active[start:start + parallel.tp] if r in positions)
                    if any(count != parallel.etp for count in counts.values()):
                        raise ModelingError("rank_table_expert_tp_width_mismatch")
        return placement

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> Hardware:
        from mindie_sim.backends import materialize
        raw_in = _normalize_core_aliases(raw)
        backend = raw_in.get("backend")
        raw_records = list(raw_in.get("devices", raw_in.get("dies", ())) or ())
        raw = materialize(raw_in)
        _validate_transfer_channels(raw)
        count = int(raw.get("num_devices", len(raw.get("devices", raw.get("dies", ()))) or 1))
        ranks_per_node = int(raw.get("ranks_per_node", count))
        package = int(raw.get("dies_per_package", 1))
        records = list(raw.get("devices", raw.get("dies", ({},) * count)))
        if len(records) != count:
            raise ModelingError("num_devices_differs_from_explicit_device_records")
        if raw_records and len(raw_records) > count:
            raise ModelingError("num_devices_differs_from_explicit_device_records")
        placements = resolve_device_placements(
            raw_records or records, count, ranks_per_node, package)
        machine_explicit = merge_device_inputs(
            machine_device_defaults(raw_in), raw_in.get("device", {}))
        devices = []
        for index in range(count):
            explicit_record = (raw_records[index]
                               if index < len(raw_records)
                               and isinstance(raw_records[index], Mapping) else {})
            devices.append(_parse_placement_device(
                raw, raw_in, backend, machine_explicit, explicit_record, placements[index]))
        ranks = tuple(device.rank for device in devices)
        if len(set(ranks)) != count:
            raise ModelingError("device_rank_duplicate")
        if "links" in raw:
            raw_links = list(raw["links"])
            links = tuple(Link.from_mapping(link, default_members=ranks) for link in raw_links)
            override_keys = _network_override_keys(raw) if "links" in raw_in else set()
            if override_keys:
                updated_links = []
                for raw_link, link in zip(raw_links, links):
                    link_keys = set(raw_link.keys()) if isinstance(raw_link, Mapping) else set()
                    filtered: dict[str, Any] = {}
                    for key in override_keys:
                        if key.endswith("_bw_util"):
                            if not ({"bw_utilization", "bw_ratio", "utilization"} & link_keys):
                                filtered[key] = raw[key]
                        elif key.endswith("_latency_us"):
                            if not ({"latency_us", "alpha_us"} & link_keys):
                                filtered[key] = raw[key]
                    updated_links.append(
                        apply_network_overrides(
                            (link,), filtered, gbps_to_bytes_per_ms=GBPS_TO_BYTES_PER_MS,
                        )[0]
                        if filtered else link
                    )
                links = tuple(updated_links)
        else:
            links = _neutral_transfer_channel_links(raw, placements)
        for link in links:
            if link.semantics not in {"shared", "per_port", "per_pair", "direct", "fully_connected", "mesh"}:
                raise ModelingError(f"link_semantics_unknown:{link.semantics}")
            if not set(link.members).issubset(ranks):
                raise ModelingError(f"link_member_unknown:{link.name}")
        device_families = {device.arch_family for device in devices}
        machine_family = (device_families.pop() if len(device_families) == 1
                          else "mixed")
        device_skus = {device.sku for device in devices if device.sku}
        machine_sku = (device_skus.pop() if len(device_skus) == 1
                       else "mixed" if device_skus else raw.get("sku", ""))
        return cls(
            name=raw.get("name", raw.get("sku", "declared")),
            dies=tuple(devices), links=links,
            arch_family=machine_family,
            hardware_key=raw.get("hardware_key", raw.get("name", "")),
            ranks_per_node=ranks_per_node, dies_per_package=package,
            sku=machine_sku,
            assumptions=tuple(sorted(raw.get("assumptions", {}).items())),
            transfer_channels=tuple(sorted(
                (str(k), tuple(sorted(v.items())))
                for k, v in (raw.get("transfer_channels") or {}).items()
                if isinstance(v, Mapping)
            )),
            topology_options=tuple(sorted(
                (str(k), raw[k]) for k in _PRESERVED_TOPOLOGY_KEYS
                if raw.get(k) is not None
            )),
            source=deepcopy(dict(raw.get("source") or {})),
            collective_protocols=_collective_protocols(
                raw.get("collective_protocols", ())),
        )

    @property
    def physical_ranks_per_node(self) -> int:
        raw_val = dict(self.topology_options).get("physical_ranks_per_node")
        if isinstance(raw_val, (int, float, str)) and not isinstance(raw_val, bool):
            return int(raw_val)
        return int(self.ranks_per_node)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "architecture": self.arch_family,
                "hardware_key": self.hardware_key,
                "sku": self.sku,
                "devices": [device.to_dict() for device in self.dies],
                "links": [link.to_dict() for link in self.links],
                "ranks_per_node": self.ranks_per_node,
                "dies_per_package": self.dies_per_package,
                "assumptions": dict(self.assumptions),
                **dict(self.topology_options),
                **({"transfer_channels": {
                    name: dict(items) for name, items in self.transfer_channels}}
                   if self.transfer_channels else {}),
                **({"source": deepcopy(dict(self.source))} if self.source else {}),
                **({"collective_protocols": dict(self.collective_protocols)}
                   if self.collective_protocols else {})}


from .datapath import (
    has_core_override as _has_core_override,
    normalize_core_aliases as _normalize_core_aliases,
    resolve_device_compute_arch as _resolve_device_compute_arch,
    scale_explicit_placement_device as _scale_explicit_placement_device,
)


def _validate_transfer_channels(raw: Mapping[str, Any]) -> None:
    allowed_semantics = {
        "shared", "per_port", "per_pair", "direct", "fully_connected",
        "mesh", "switched", "unknown", "none",
    }
    for name, channel in (raw.get("transfer_channels") or {}).items():
        if not isinstance(channel, Mapping):
            continue
        solo = _optional_pos_rate(channel.get("solo_effective_uni_bw_gbps"),
                                  f"transfer_channel_solo_bw:{name}")
        if solo is not None:
            peak = _optional_pos_rate(channel.get("uni_peak_bw_gbps"),
                                      f"transfer_channel_uni_peak_bw:{name}")
            if peak is not None and solo > peak:
                raise ModelingError(f"transfer_channel_solo_exceeds_uni_peak:{name}")
        for field_name in ("channel_semantics", "semantics"):
            if field_name in channel and channel[field_name] not in allowed_semantics:
                raise ModelingError(
                    f"transfer_channel_semantics_unsupported:{name}:{channel[field_name]}")
        if "scope" in channel and channel["scope"] is not None:
            from .hardware_links import normalize_link_scope
            normalize_link_scope(channel["scope"])


def _materialize_placement_device(
    raw: Mapping[str, Any],
    raw_in: Mapping[str, Any],
    backend: Any,
    machine_explicit: Mapping[str, Any],
    explicit_record: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    from mindie_sim.backends import materialize

    record_fields = {
        key: value for key, value in explicit_record.items()
        if key not in ("rank", "node", "package", "devices", "dies",
                       "sku", "backend")
    }
    record_device = record_fields.pop("device", {})
    explicit_decl = merge_device_inputs(
        machine_explicit, record_fields, record_device)
    record_backend = explicit_record.get("backend", backend)
    record_sku = (explicit_record.get("sku")
                  or raw_in.get("sku") or raw.get("sku", ""))
    rec_base_aic = rec_base_aiv = 0
    if backend:
        local = materialize({
            "backend": record_backend, "sku": record_sku,
            "num_devices": 1, "device": explicit_decl,
        })
        data = dict(local.get("device", {}))
        if not data:
            raise ModelingError(
                f"hardware_placement_sku_unmaterialized:{record_sku}")
        if _has_core_override(explicit_decl):
            bare = dict(materialize({
                "backend": record_backend, "sku": record_sku,
                "num_devices": 1}).get("device", {}))
            rec_base_aic = int(bare.get("compute_units", 0) or 0)
            rec_base_aiv = int((bare.get("resource_units") or {}).get(
                "vector", rec_base_aic * 2) or (rec_base_aic * 2))
        arch_family_local = str(local.get(
            "architecture", data.get("arch_family", "")))
    else:
        data = dict(explicit_decl)
        arch_family_local = str(data.get("arch_family") or "generic")
        if data.get("observed_aicore_freq_mhz") is not None:
            from mindie_sim.backends.ascend import apply_observed_aicore_clock, profiles
            profile = profiles().get(record_sku)
            if profile is not None and profile["architecture"] == arch_family_local:
                declaration = (raw.get("source") or {}).get("declaration", {})
                reference_mhz = (declaration.get("configured_clock_mhz")
                                 if record_sku == raw.get("sku") else None)
                data = apply_observed_aicore_clock(
                    data, profile, reference_mhz=reference_mhz)
    if explicit_record.get("root_complex_id") is not None:
        data["root_complex_id"] = explicit_record["root_complex_id"]
    data["sku"] = str(record_sku or data.get("sku", ""))
    data["arch_family"] = arch_family_local
    raw_cu = data.get("compute_units")
    if raw_cu is not None and (isinstance(raw_cu, bool) or int(raw_cu) < 1):
        raise ModelingError("compute_units_must_be_positive")
    new_aic = int(raw_cu or 1)
    if rec_base_aic > 0:
        _scale_explicit_placement_device(
            data, explicit_decl, new_aic, rec_base_aic, rec_base_aiv)
    arch_family = str(data.get("arch_family")
                      or raw.get("architecture") or "generic")
    arch_input = data.get("compute_arch") or {}
    if (not backend and arch_family == "A5" and isinstance(arch_input, Mapping)
            and not arch_input.get("native_compute")):
        from mindie_sim.backends.ascend import registered_matrix_capability
        capability = registered_matrix_capability(record_sku, arch_family, arch_input)
        if capability:
            data = merge_device_inputs({"compute_arch": capability}, data)
    return data, arch_family


def _parse_placement_device(
    raw: Mapping[str, Any],
    raw_in: Mapping[str, Any],
    backend: Any,
    machine_explicit: Mapping[str, Any],
    explicit_record: Mapping[str, Any],
    placement: tuple[int, int, int],
) -> Device:
    data, arch_family = _materialize_placement_device(
        raw, raw_in, backend, machine_explicit, explicit_record)
    arch = _resolve_device_compute_arch(data, raw, arch_family)
    peaks = dict(data.get("compute_peaks", {}))
    if arch.has_capabilities() and peaks and not arch.has_native_modes():
        raise ModelingError(
            "compute_peaks_without_native_capability:"
            f"{data.get('sku', arch_family)}")
    aliases = data.get("compute_peak_aliases") or {}
    if arch.has_native_modes():
        arch.validate_peaks(peaks, aliases=aliases)
    else:
        validate_compute_peak_aliases(peaks, aliases, arch)
    arch_rates, arch_units, arch_work_units, arch_modes = arch.supplies(
        int(data.get("compute_units", 1)), data.get("resource_units", {}),
        data.get("rates", {}))
    memory = int(data.get("memory_bytes", 0))
    rates = {key: float(value)
             for key, value in {**arch_rates, **data.get("rates", {})}.items()}
    if memory <= 0 or not peaks or "hbm" not in rates or "vector" not in rates:
        raise ModelingError("device_requires_memory_compute_vector_and_memory_rates")
    if any(not math.isfinite(value) or value <= 0
           for value in (*peaks.values(), *rates.values())):
        raise ModelingError("device_rates_must_be_positive")
    buffers = dict(data.get("buffers", {}))
    units = {**arch_units, **data.get("resource_units", {})}
    if any(value < 1 for value in units.values()):
        raise ModelingError("resource_units_must_be_positive")
    buffer_units = dict(data.get("buffer_units", {}))
    if "l2" in buffers:
        declared_l2_units = buffer_units.get("l2")
        if declared_l2_units is not None and int(declared_l2_units) != 1:
            raise ModelingError("shared_l2_buffer_requires_one_die_owner")
        buffer_units["l2"] = 1
    modes = {**arch_modes, **{
        name: {**arch_modes.get(name, {}), **mode}
        for name, mode in data.get("service_modes", {}).items()}}
    for name, mode in modes.items():
        if (name in rates or mode["resource"] not in rates
                or not math.isfinite(mode["rate"]) or mode["rate"] <= 0):
            raise ModelingError(f"invalid_physical_service_mode:{name}")
    work_units = {**arch_work_units, **data.get("work_units", {})}
    for role, unit in work_units.items():
        if role in NATIVE_WORK_UNITS and unit != NATIVE_WORK_UNITS[role]:
            raise ModelingError(f"native_work_unit_conflict:{role}")
    mechanisms = parse_physical_mechanisms(
        data.get("physical_mechanisms"),
        fallback_l2_bytes=int(buffers.get("l2", 0)),
        compute_arch=arch, buffer_units=buffer_units)
    simt_carveout = int(data.get("simt_ub_carveout_bytes", 0) or 0)
    if simt_carveout < 0:
        raise ModelingError(f"simt_ub_carveout_bytes_negative:{simt_carveout}")
    subtile_wave = parse_subtile_waves(data.get("subtile_wave"))
    dispatch_sync = parse_dispatch_sync(data.get("dispatch_sync"))
    hbm_rw_alpha = validate_hbm_rw_alpha(data.get("hbm_rw_alpha", 0.0))
    hbm_read = _optional_pos_rate(data.get("hbm_read_bytes_per_s"), "hbm_read_bytes_per_s")
    hbm_write = _optional_pos_rate(data.get("hbm_write_bytes_per_s"), "hbm_write_bytes_per_s")
    memory_access = parse_memory_access(data.get("memory_access"))
    observed_freq = _optional_pos_rate(data.get("observed_aicore_freq_mhz"),
                                       "observed_aicore_freq_mhz")
    rank, node, package_id = placement
    return Device(
        rank=rank, memory_bytes=memory,
        compute_peaks=tuple(sorted(peaks.items())),
        rates=tuple(sorted(rates.items())),
        buffers=tuple(sorted(buffers.items())),
        compute_units=int(data.get("compute_units", 1) or 1),
        launch_us=float(data.get("launch_us", 0)),
        node=node, package=package_id,
        resource_units=tuple(sorted(units.items())),
        buffer_units=tuple(sorted(buffer_units.items())),
        service_modes=tuple(sorted((name, (mode["resource"], float(mode["rate"])))
                                   for name, mode in modes.items())),
        work_units=tuple(sorted(work_units.items())),
        kernel_invocations=tuple(sorted(
            (key, InvocationTiming.from_mapping(value))
            for key, value in data.get("kernel_invocations", {}).items())),
        store_queues=tuple(sorted((name, tuple(sorted(queue.items())))
                                  for name, queue in data.get("store_queues", {}).items())),
        compute_arch=arch,
        compute_peak_aliases=tuple(sorted(dict(aliases).items())),
        linked_rate_overrides=tuple(sorted(
            s.name for s in arch.instruction_services
            if s.has_linked_supply
            and s.name in data.get("rates", {}))),
        instruction_mode_overrides=tuple(sorted(
            s.name for s in arch.instruction_services
            if s.resource and "rate" in data.get("service_modes", {}).get(s.name, {}))),
        instruction_unit_overrides=tuple(sorted(
            s.name for s in arch.instruction_services
            if s.has_linked_supply
            and s.name in data.get("resource_units", {}))),
        sku=str(data.get("sku", "")),
        arch_family=arch_family,
        physical_mechanisms=mechanisms,
        root_complex_id=str(data.get("root_complex_id") or DEFAULT_ROOT_COMPLEX_ID),
        simt_ub_carveout_bytes=simt_carveout,
        subtile_wave=subtile_wave,
        dispatch_sync=dispatch_sync,
        hbm_rw_alpha=hbm_rw_alpha,
        hbm_read_bytes_per_s=hbm_read,
        hbm_write_bytes_per_s=hbm_write,
        memory_access=memory_access,
        observed_aicore_freq_mhz=observed_freq,
    )


def parse_hardware(raw: Mapping[str, Any]) -> Hardware:
    """Parse a hardware declaration, including top-level network overrides."""
    if not isinstance(raw, Mapping):
        raise ModelingError("hardware_mapping_required")
    return Hardware.from_mapping(raw)
