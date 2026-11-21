"""Explicit hardware datapath and ISA capability view derived from ComputeArchSpec.

[Role] Derive immutable per-die datapath capabilities, AIC/AIV owner pairing,
  usable local data routes, SIMT UB carveout, sub-tile waves, STARS dispatch
  sync, directional HBM bus specs and upstream source citations strictly from
  declared ``ComputeArchSpec`` and ``Device`` records.
[Invariants] Zero SKU-name, ``arch_family``, ``compute_peaks`` or
  ``service_modes`` capability sniffing. Direct routes (``l0c->ub``,
  ``ub->l1``) exist only when explicitly declared on ``ComputeArchSpec.paths``;
  missing physical routes raise ``UnmodeledCapabilityError``. Representative
  multi-die stages require homogeneous ``compute_arch`` and datapath signatures.
[Timeouts & Locks] Pure metadata derivation over finite device tuples; no I/O
  or locks.
[Side-Effect Policy] Returns new frozen records; never mutates input devices.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NewType

from .common import ModelingError
from .compute_arch import ComputeArchSpec, DataPath
from .device_input import (
    DeviceInput,
    DispatchSyncSpec,
    MemoryAccessSpec,
    StarsSpec,
    SubtileWaveSpec,
)
from .hardware import Device, Link
from .hardware_links import CollectiveBandSpec, validate_protocol_bands

ArchId = NewType("ArchId", str)
DieSpec = Device

GENERIC_ARCH_ID = ArchId("generic")
L0C_TO_UB_OUTPUT_PATH = "l0c_to_ub"
BITS_PER_BYTE = 8
DEFAULT_ELEMENT_BITS = 16
PAIR_ARITY = 2
TILE_3D_ARITY = 3


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Verified upstream source file provenance for a modeled mechanism."""

    repository: str
    revision: str
    relative_path: str
    source_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "repository": self.repository,
            "revision": self.revision,
            "relative_path": self.relative_path,
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True, slots=True)
class ComputeArchIdentity:
    """Explicit compute architecture identity bound to its declared spec."""

    arch_id: ArchId
    declared: ComputeArchSpec | None


@dataclass(frozen=True, slots=True)
class StarsQueueSpec:
    """Declared hardware STARS descriptor queue capability on one die."""

    resource: str
    slots_per_die: int
    descriptor_path: DataPath


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """Resolved physical data route between two on-chip or GM buffers."""

    source: str
    destination: str
    path: DataPath
    representation: str = ""
    owner_group: Any = None
    tile: Any = None


@dataclass(frozen=True, slots=True)
class DatapathCapabilitySpec:
    """Immutable per-die hardware datapath and owner-pairing capability view."""

    compute_arch: ComputeArchIdentity
    aic_count: int
    aiv_count: int
    aic_aiv_pairs: tuple[tuple[int, tuple[int, ...]], ...]
    usable_paths: tuple[DataPath, ...]
    matrix_output_paths: frozenset[str]
    ub_peer_edges: tuple[tuple[int, int], ...]
    stars: StarsQueueSpec | None
    simt_ub_carveout_bytes: int = 0
    subtile_wave: tuple[SubtileWaveSpec, ...] = ()
    dispatch_sync: DispatchSyncSpec | None = None
    hbm_rw_alpha: float = 0.0
    hbm_read_bytes_per_s: float = 0.0
    hbm_write_bytes_per_s: float = 0.0
    memory_access: MemoryAccessSpec | None = None
    buffers: tuple[tuple[str, int], ...] = ()
    compute_peaks: tuple[tuple[str, float], ...] = ()
    compute_peak_aliases: tuple[tuple[str, str], ...] = ()

    @property
    def arch_id(self) -> ArchId:
        return self.compute_arch.arch_id

    def cube_peak(self, dtype: str) -> float:
        """Return the declared matrix peak (FLOP/s) for ``dtype`` without profile guessing."""
        from .compute_peak_aliases import resolve_compute_peak_name

        peaks = dict(self.compute_peaks)
        supply = dict(self.compute_peak_aliases)
        dtype_aliases = {
            "bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
            "fp8_e4m3": "fp8", "fp8_e5m2": "fp8",
        }
        canonical = dtype_aliases.get(dtype, dtype)
        arch = self.compute_arch.declared
        if arch is None or not (
            arch.supports_native("matrix", dtype) or arch.supports_native("matrix", canonical)
        ):
            raise ModelingError(f"fuse_region_matrix_precision_unsupported:{dtype}")
        native = resolve_compute_peak_name(dtype, peaks, supply) if supply else dtype
        name = native if native in peaks else dtype_aliases.get(native, canonical)
        if name in peaks and peaks[name] > 0.0:
            return float(peaks[name])
        raise ModelingError(f"fuse_region_compute_precision_requires_context:{dtype}")

    @property
    def supported_output_paths(self) -> tuple[str, ...]:
        """Sorted projection of declared supported matrix output paths."""
        return tuple(sorted(self.matrix_output_paths))

    def has_path(self, source: str, destination: str, /) -> bool:
        """True iff a directional route ``source -> destination`` is declared."""
        return any(
            path.source == source and path.target == destination
            for path in self.usable_paths
        )

    @property
    def has_l0c_to_ub(self) -> bool:
        return self.has_path("l0c", "ub")

    @property
    def has_ub_to_l1(self) -> bool:
        return self.has_path("ub", "l1")

    def _route_err(self, code: str, message: str):
        from mindie_sim.public.errors import UnmodeledCapabilityError
        return UnmodeledCapabilityError(
            capability="datapath_route", entry="DatapathCapabilitySpec.route",
            code=code, message=message,
        )

    def _validate_route_owner(self, path: DataPath, src: str, dst: str, owner_group: Any) -> None:
        aic_to_aivs = {int(aic): frozenset(int(a) for a in aivs) for aic, aivs in self.aic_aiv_pairs}
        named_pair: tuple[int, int] | None = None
        if isinstance(owner_group, Mapping):
            if "aic" in owner_group or "aiv" in owner_group:
                named_pair = (owner_group.get("aic"), owner_group.get("aiv"))
                pair = named_pair
            else:
                src_v, dst_v = owner_group.get("src"), owner_group.get("dst")
                pair = (src_v, dst_v)
        elif isinstance(owner_group, (tuple, list)):
            pair = tuple(owner_group)
        elif isinstance(owner_group, int) and not isinstance(owner_group, bool):
            limit = self.aic_count if path.executor == "aic" else self.aiv_count
            if not (0 <= owner_group < limit):
                raise self._route_err(
                    f"datapath_route_owner_unreachable:{src}->{dst}:{owner_group}",
                    f"owner {owner_group} outside [0, {limit}) for {path.executor}",
                )
            return
        else:
            raise self._route_err(f"datapath_route_owner_invalid:{src}->{dst}", f"invalid owner_group {owner_group!r}")
        if set(path.reach) == {"aic", "aiv"}:
            if len(pair) != PAIR_ARITY or any(isinstance(v, bool) or not isinstance(v, int) for v in pair):
                raise self._route_err(
                    f"datapath_route_owner_invalid:{src}->{dst}",
                    f"cross-core path {src}->{dst} requires (src_core, dst_core) ints",
                )
            if named_pair is not None:
                aic_id, aiv_id = named_pair
            else:
                aic_id, aiv_id = (pair[1], pair[0]) if src == "ub" else (pair[0], pair[1])
            if not (0 <= aic_id < self.aic_count and aiv_id in aic_to_aivs.get(aic_id, frozenset())):
                raise self._route_err(
                    f"datapath_route_owner_unreachable:{src}->{dst}:{aic_id}->{aiv_id}",
                    f"arch {self.arch_id} AIC {aic_id} cannot reach AIV {aiv_id}",
                )
            return
        limit = self.aic_count if path.executor == "aic" else self.aiv_count
        if not pair or any(isinstance(v, bool) or not isinstance(v, int) or not (0 <= v < limit) for v in pair):
            raise self._route_err(
                f"datapath_route_owner_unreachable:{src}->{dst}",
                f"owner_group {pair!r} outside [0, {limit}) for {path.executor}",
            )

    def _validate_route_tile(
        self, path: DataPath, src: str, dst: str, rep: str, elem_bits: int, tile: Any
    ) -> None:
        align = max(1, int(path.alignment_bytes or 1))
        if isinstance(tile, int) and not isinstance(tile, bool):
            if tile <= 0 or tile % align != 0:
                raise self._route_err(
                    f"datapath_route_tile_invalid:{src}->{dst}:{tile}",
                    f"tile bytes {tile} must be positive multiple of {align}",
                )
            tile_bytes = tile
        elif isinstance(tile, (tuple, list)) and tile:
            if any(isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in tile):
                raise self._route_err(
                    f"datapath_route_tile_invalid:{src}->{dst}",
                    f"tile dimensions {tile!r} must be positive integers",
                )
            if (src == "l0c" or dst == "l0c") and self.subtile_wave:
                wave = next((w for w in self.subtile_wave if w.native_mode == rep), self.subtile_wave[0])
                if (len(tile) < PAIR_ARITY
                        or tile[0] % wave.independently_issuable_m != 0
                        or tile[1] % wave.tile_n != 0
                        or (len(tile) >= TILE_3D_ARITY and tile[PAIR_ARITY] % wave.tile_k != 0)):
                    raise self._route_err(
                        f"datapath_route_tile_invalid:{src}->{dst}:{tuple(tile)}",
                        f"tile {tuple(tile)} violates subtile alignment ({wave.independently_issuable_m},{wave.tile_n})",
                    )
            elems = 1
            for dim in tile:
                elems *= dim
            total_bits = elems * elem_bits
            if total_bits % BITS_PER_BYTE != 0 or (total_bits // BITS_PER_BYTE) % align != 0:
                raise self._route_err(
                    f"datapath_route_tile_invalid:{src}->{dst}:{tuple(tile)}",
                    f"tile {tuple(tile)} ({total_bits} bits) violates {align}B alignment",
                )
            tile_bytes = total_bits // BITS_PER_BYTE
        else:
            raise self._route_err(f"datapath_route_tile_invalid:{src}->{dst}", f"unsupported tile {tile!r}")
        buffers = dict(self.buffers)
        for buf_name in (src, dst):
            if buf_name in buffers and tile_bytes > self.usable_bytes(buf_name):
                raise self._route_err(
                    f"datapath_route_tile_exceeds_buffer:{buf_name}:{tile_bytes}",
                    f"tile {tile_bytes}B exceeds usable {buf_name} capacity",
                )

    def route(
        self, src: str, dst: str, representation: str = "",
        owner_group: Any = None, tile: Any = None,
    ) -> RoutePlan:
        """Resolve a physical directed path or raise ``UnmodeledCapabilityError``."""
        rep = str(representation or "").strip().lower()
        declared_arch = self.compute_arch.declared
        registry = ({item.name: item for item in declared_arch.representations}
                    if declared_arch is not None else {})
        elem_bits = DEFAULT_ELEMENT_BITS
        if rep:
            if registry and rep not in registry:
                try:
                    rep = declared_arch.representation(rep).name
                except ModelingError:
                    raise self._route_err(
                        f"datapath_route_representation_unknown:{rep}",
                        f"arch {self.arch_id} does not declare representation {rep!r}",
                    ) from None
            if rep in registry:
                elem_bits = int(registry[rep].bits)
        candidates = [p for p in self.usable_paths if p.source == src and p.target == dst]
        for path in candidates:
            if rep and ((registry and not path.supports(rep, registry))
                        or (not registry and path.representations and rep not in path.representations)):
                continue
            if owner_group is not None:
                self._validate_route_owner(path, src, dst, owner_group)
            if tile is not None:
                self._validate_route_tile(path, src, dst, rep, elem_bits, tile)
            return RoutePlan(
                source=src, destination=dst, path=path,
                representation=rep, owner_group=owner_group, tile=tile,
            )
        raise self._route_err(
            f"datapath_route_unsupported:{src}->{dst}",
            f"arch {self.arch_id} does not declare directed route {src}->{dst} (representation={rep or 'any'})",
        )

    def usable_bytes(
        self,
        pool: str,
        owner: Any = None,
        kernel: Any = None,
    ) -> int:
        """Return usable per-owner bytes for ``pool`` after SIMT UB carveout."""
        del owner
        pool_key = str(pool).strip().lower()
        capacity = dict(self.buffers).get(pool_key)
        if capacity is None:
            raise ModelingError(f"datapath_buffer_pool_undeclared:{pool_key}")
        is_simt = False
        if isinstance(kernel, bool):
            is_simt = kernel
        elif isinstance(kernel, str):
            is_simt = "simt" in kernel.lower()
        elif isinstance(kernel, Mapping):
            is_simt = bool(kernel.get("simt")) or "simt" in str(kernel.get("abi") or "").lower()
        elif kernel is not None:
            is_simt = bool(getattr(kernel, "simt", False)) or "simt" in str(
                getattr(kernel, "abi", "") or getattr(kernel, "name", "")
            ).lower()
        if pool_key == "ub" and is_simt and self.simt_ub_carveout_bytes > 0:
            return max(0, int(capacity) - int(self.simt_ub_carveout_bytes))
        return max(0, int(capacity))

    @property
    def stars_spec(self) -> StarsSpec | None:
        """Return declared ``StarsSpec`` from ``dispatch_sync`` if present."""
        return None if self.dispatch_sync is None else self.dispatch_sync.stars

    @property
    def signature(self) -> tuple:
        """Hashable structural capability identity for cache keys."""
        return (
            self.compute_arch.arch_id,
            self.aic_count,
            self.aiv_count,
            self.aic_aiv_pairs,
            tuple((p.name, p.source, p.target, p.unit, p.executor, p.sync)
                  for p in self.usable_paths),
            tuple(sorted(self.matrix_output_paths)),
            self.ub_peer_edges,
            (None if self.stars is None else (
                self.stars.resource, self.stars.slots_per_die,
                self.stars.descriptor_path.name)),
            self.simt_ub_carveout_bytes,
            self.subtile_wave,
            self.dispatch_sync,
            self.hbm_rw_alpha,
            self.hbm_read_bytes_per_s,
            self.hbm_write_bytes_per_s,
            self.memory_access,
            self.buffers,
            self.compute_peaks,
        )


def _arch_identity(spec: ComputeArchSpec) -> ComputeArchIdentity:
    declared = spec if (spec.preset or spec.has_capabilities()
                        or spec.instruction_services) else None
    arch_id = ArchId(spec.preset) if spec.preset else GENERIC_ARCH_ID
    return ComputeArchIdentity(arch_id=arch_id, declared=declared)


def _aiv_count(die: Device) -> int:
    resource_units = dict(die.resource_units)
    if "vector" in resource_units:
        return int(resource_units["vector"])
    if "vector" in dict(die.rates):
        return int(die.compute_units)
    return 0


def _aic_aiv_pairs(aic_count: int, aiv_count: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if aic_count < 1 or aiv_count < 1 or aiv_count % aic_count != 0:
        return ()
    ratio = aiv_count // aic_count
    return tuple(
        (aic, tuple(range(aic * ratio, (aic + 1) * ratio)))
        for aic in range(aic_count)
    )


def _matrix_output_paths(die: Device, usable_paths: tuple[DataPath, ...]) -> frozenset[str]:
    declared = frozenset(die.physical_mechanisms.matrix.supported_output_paths)
    if not usable_paths:
        return declared
    has_l0c_ub = any(p.source == "l0c" and p.target == "ub" for p in usable_paths)
    if L0C_TO_UB_OUTPUT_PATH in declared and not has_l0c_ub:
        return declared - {L0C_TO_UB_OUTPUT_PATH}
    return declared


def _stars_queue(die: Device, usable_paths: tuple[DataPath, ...]) -> StarsQueueSpec | None:
    queues = dict(die.store_queues)
    raw = queues.get("stars_descriptor_queue")
    if not raw:
        return None
    mapping = dict(raw)
    slots = int(mapping.get("slots_per_die", 0))
    if slots < 1:
        return None
    descriptor = next((p for p in usable_paths if p.sync == "event"), None)
    if descriptor is None:
        return None
    return StarsQueueSpec(
        resource=str(mapping.get("resource", "stars_descriptor_queue")),
        slots_per_die=slots,
        descriptor_path=descriptor,
    )


def datapath_for_die(
    die: Device,
    /,
    *,
    ub_peer_edges: tuple[tuple[int, int], ...] = (),
) -> DatapathCapabilitySpec:
    """Derive immutable datapath capabilities strictly from ``die.compute_arch`` and ``die``."""
    spec = die.compute_arch
    identity = _arch_identity(spec)
    aic_count = int(die.compute_units)
    aiv_count = _aiv_count(die)
    usable_paths = tuple(spec.paths)
    hbm_rate = float(dict(die.rates).get("hbm", 0.0) or 0.0)
    hbm_read = float(die.hbm_read_bytes_per_s) if die.hbm_read_bytes_per_s is not None else hbm_rate
    hbm_write = float(die.hbm_write_bytes_per_s) if die.hbm_write_bytes_per_s is not None else hbm_rate
    return DatapathCapabilitySpec(
        compute_arch=identity,
        aic_count=aic_count,
        aiv_count=aiv_count,
        aic_aiv_pairs=_aic_aiv_pairs(aic_count, aiv_count),
        usable_paths=usable_paths,
        matrix_output_paths=_matrix_output_paths(die, usable_paths),
        ub_peer_edges=tuple(ub_peer_edges),
        stars=_stars_queue(die, usable_paths),
        simt_ub_carveout_bytes=int(die.simt_ub_carveout_bytes),
        subtile_wave=tuple(die.subtile_wave),
        dispatch_sync=die.dispatch_sync,
        hbm_rw_alpha=float(die.hbm_rw_alpha),
        hbm_read_bytes_per_s=hbm_read,
        hbm_write_bytes_per_s=hbm_write,
        memory_access=die.memory_access,
        buffers=tuple(die.buffers),
        compute_peaks=tuple((str(k), float(v)) for k, v in die.compute_peaks),
        compute_peak_aliases=tuple((str(k), str(v)) for k, v in die.compute_peak_aliases),
    )


def device_input_for_die(
    die: Device,
    /,
    *,
    links: Sequence[Link] = (),
    transfer_channels: Mapping[str, object] | Sequence[tuple[str, object]] = (),
    provenance: Mapping[str, object] | None = None,
) -> DeviceInput:
    """Build a typed ``DeviceInput`` normalized view for one physical die."""
    edges = ub_peer_edges_for_rank(die.rank, links, transfer_channels)
    datapath = datapath_for_die(die, ub_peer_edges=edges)
    entries = (
        tuple((str(k), tuple(sorted(dict(v).items()))) for k, v in transfer_channels.items())
        if isinstance(transfer_channels, Mapping)
        else tuple((str(k), tuple(v)) for k, v in transfer_channels)
    )
    return DeviceInput(
        compute_arch=die.compute_arch,
        datapath=datapath,
        kernel_invocations=tuple(die.kernel_invocations),
        transfer_channels=entries,
        rates=tuple(die.rates),
        units=tuple(die.resource_units),
        buffers=tuple(die.buffers),
        peaks=tuple(die.compute_peaks),
        provenance=tuple(sorted((str(k), v) for k, v in dict(provenance or {}).items())),
    )


def _ub_channel_names(
    transfer_channels: Mapping[str, object] | Sequence[tuple[str, object]] = (),
) -> tuple[frozenset[str], bool]:
    if not transfer_channels:
        return frozenset(), False
    entries = (
        transfer_channels.items()
        if isinstance(transfer_channels, Mapping)
        else transfer_channels
    )
    ub: set[str] = set()
    has_entries = False
    for channel, spec in entries:
        has_entries = True
        mapping = dict(spec) if isinstance(spec, (Mapping, tuple, list)) else {}
        protocol = str(mapping.get("protocol") or "").strip().upper()
        if protocol == "UB" or protocol.startswith("UB_"):
            ub.add(str(channel))
    return frozenset(ub), has_entries


def ub_peer_edges_for_rank(
    rank: int,
    links: Sequence[Link] = (),
    transfer_channels: Mapping[str, object] | Sequence[tuple[str, object]] = (),
) -> tuple[tuple[int, int], ...]:
    """Derive sorted directed UB direct-mesh peer edges ``(rank, peer)`` from ``links``."""
    if not links:
        return ()
    target = int(rank)
    ub_channels, _has_channels = _ub_channel_names(transfer_channels)
    edges: set[tuple[int, int]] = set()
    for link in links:
        if target not in link.members:
            continue
        chan_low = str(link.channel).lower()
        name_low = str(link.name).lower()
        is_ub = (
            link.channel in ub_channels
            or chan_low == "ub"
            or chan_low.startswith("ub_")
            or name_low.startswith("ub_")
        )
        if not is_ub:
            continue
        for member in link.members:
            peer = int(member)
            if peer != target:
                edges.add((target, peer))
    return tuple(sorted(edges))


def _arch_capability_key(spec: ComputeArchSpec) -> tuple:
    return (
        spec.preset,
        tuple(
            (s.name, s.unit, s.owner_axis, s.per_owner_rate, s.scale_with,
             s.resource, s.clock_hz, s.cycles_per_unit)
            for s in spec.instruction_services
        ),
        tuple((r.name, r.bits, r.roles, r.alignment_bytes) for r in spec.representations),
        tuple((m.name, m.kind, m.operands, m.unit, m.accumulators, m.implementations)
              for m in spec.native_modes),
        tuple((c.source, c.target, c.implementation, c.lossless) for c in spec.casts),
        tuple((p.name, p.source, p.target, p.roles, p.unit, p.executor, p.reach,
               p.buffer, p.sync, p.alignment_bytes, p.representations)
              for p in spec.paths),
    )


def _ub_relative_connectivity(
    rank: int,
    edges: tuple[tuple[int, int], ...],
    stage_ranks: frozenset[int],
) -> tuple[int, int]:
    in_stage = sum(1 for src, dst in edges if src == rank and dst in stage_ranks)
    out_stage = sum(1 for src, dst in edges if src == rank and dst not in stage_ranks)
    return (in_stage, out_stage)


def _homogeneity_key(
    spec: DatapathCapabilitySpec,
    rank: int,
    stage_ranks: frozenset[int],
) -> tuple:
    return (
        spec.compute_arch.arch_id,
        spec.aic_count,
        spec.aiv_count,
        spec.aic_aiv_pairs,
        tuple((p.name, p.source, p.target, p.unit, p.executor, p.sync)
              for p in spec.usable_paths),
        tuple(sorted(spec.matrix_output_paths)),
        _ub_relative_connectivity(rank, spec.ub_peer_edges, stage_ranks),
        (None if spec.stars is None else (
            spec.stars.resource, spec.stars.slots_per_die,
            spec.stars.descriptor_path.name)),
        spec.simt_ub_carveout_bytes,
        spec.subtile_wave,
        spec.dispatch_sync,
        spec.hbm_rw_alpha,
        spec.hbm_read_bytes_per_s,
        spec.hbm_write_bytes_per_s,
        spec.memory_access,
    )


def datapath_for_devices(
    devices: Sequence[Device],
    /,
    *,
    links: Sequence[Link] = (),
    transfer_channels: Mapping[str, object] | Sequence[tuple[str, object]] = (),
) -> tuple[ComputeArchSpec, DatapathCapabilitySpec]:
    """Validate homogeneous ``compute_arch``, owner counts, and datapath across stage dies."""
    dies = tuple(devices)
    if not dies:
        raise ModelingError("stage_devices_empty")
    stage_ranks = frozenset(int(d.rank) for d in dies)
    first_arch = dies[0].compute_arch
    first_key = _arch_capability_key(first_arch)
    first_edges = ub_peer_edges_for_rank(dies[0].rank, links, transfer_channels)
    first_datapath = datapath_for_die(dies[0], ub_peer_edges=first_edges)
    first_dp_key = _homogeneity_key(first_datapath, int(dies[0].rank), stage_ranks)
    for die in dies[1:]:
        if _arch_capability_key(die.compute_arch) != first_key:
            raise ModelingError(
                "stage_devices_heterogeneous_compute_arch:"
                f"{first_arch.preset}:{die.compute_arch.preset}"
            )
        other_edges = ub_peer_edges_for_rank(die.rank, links, transfer_channels)
        other_datapath = datapath_for_die(die, ub_peer_edges=other_edges)
        if _homogeneity_key(other_datapath, int(die.rank), stage_ranks) != first_dp_key:
            raise ModelingError(
                "stage_devices_heterogeneous_datapath:"
                f"{first_datapath.arch_id}:{other_datapath.arch_id}"
            )
    return first_arch, first_datapath


_CAPABILITY_REQUIRED_FAMILIES = frozenset({"A2", "A3", "A5"})


def resolve_device_compute_arch(data: dict, raw: Mapping, arch_family: str) -> ComputeArchSpec:
    """Resolve and attach registered native compute_arch capabilities for Ascend families."""
    arch = ComputeArchSpec.from_mapping(
        data.get("compute_arch", {}),
        source=f"compute_arch:{data.get('sku', arch_family)}")
    if arch_family in _CAPABILITY_REQUIRED_FAMILIES and not arch.has_native_modes():
        from mindie_sim.backends import registered_capability
        resolved = registered_capability(
            str(data.get("sku") or raw.get("sku") or ""), arch_family)
        if resolved is None:
            raise ModelingError(
                "hardware_native_capability_required:"
                f"{arch_family}:{data.get('sku', '')}")
        existing = data.get("compute_arch") or {}
        if not isinstance(existing, Mapping):
            existing = {}
        merged = {**existing, **resolved}
        merged["preset"] = resolved["preset"]
        for field_name in ("instruction_services", "representations",
                           "native_compute", "casts", "paths", "runtimes"):
            if field_name in existing and field_name in resolved:
                merged[field_name] = {**resolved[field_name], **existing[field_name]}
        arch = ComputeArchSpec.from_mapping(
            merged, source=f"registered_capability:{arch_family}")
    return arch


def scale_explicit_placement_device(
    data: dict, explicit_decl: Mapping,
    new_aic: int, rec_base_aic: int, rec_base_aiv: int,
) -> None:
    """Proportionally scale non-overridden rates/units for an explicit core-count override."""
    scale_aic = new_aic / rec_base_aic
    explicit_units = {str(key) for key in explicit_decl.get("resource_units", {})}
    cur_units = dict(data.get("resource_units", {}))
    new_aiv = int(cur_units.get("vector", rec_base_aiv) or rec_base_aiv)
    scale_aiv = (new_aiv / rec_base_aiv) if rec_base_aiv > 0 else 1.0
    if scale_aic == 1.0 and scale_aiv == 1.0:
        return
    explicit_peaks = {str(key) for key in explicit_decl.get("compute_peaks", {})}
    data["compute_peaks"] = {
        k: (float(v) if k in explicit_peaks else float(v) * scale_aic)
        for k, v in data.get("compute_peaks", {}).items()
    }
    aic_u = {"cube", "mte1", "mte2_aic", "fixpipe"}
    data["resource_units"] = {
        k: (int(v) if k in explicit_units else max(
            1, int(round(int(v) * (scale_aic if k in aic_u else scale_aiv)))))
        for k, v in cur_units.items()
    }
    explicit_buf_u = {str(key) for key in explicit_decl.get("buffer_units", {})}
    explicit_modes = {
        str(name) for name, mode in explicit_decl.get("service_modes", {}).items()
        if isinstance(mode, Mapping) and "rate" in mode
    }
    aic_b = {"l1", "l0a", "l0b", "l0c"}
    data["buffer_units"] = {
        k: (int(v) if k in explicit_buf_u else max(
            1, int(round(int(v) * (scale_aic if k in aic_b
                                   else (scale_aiv if k == "ub" else 1.0))))))
        for k, v in data.get("buffer_units", {}).items()
    }
    explicit_rates = {str(key) for key in explicit_decl.get("rates", {})}
    aic_r = {"mte1", "mte2_aic", "fixpipe"}
    aiv_r = {"vector", "mte2_aiv", "mte3", "reg_data_writes",
             "reg_ub_accesses", "reg_ub_stores", "reg_alu_lanes",
             "reg_special_lanes", "reg_reduce_lanes", "reg_shuffle_lanes",
             "scalar_store_issue"}
    scaled_r = {
        k: (float(v) * scale_aic if k in aic_r and k not in explicit_rates
            else (float(v) * scale_aiv if k in aiv_r and k not in explicit_rates
                  else float(v)))
        for k, v in data.get("rates", {}).items()
    }
    if "mte2" in scaled_r and "mte2" not in explicit_rates:
        scaled_r["mte2"] = (
            min(scaled_r["mte2_aic"], scaled_r["mte2_aiv"])
            if "mte2_aic" in scaled_r and "mte2_aiv" in scaled_r
            else scaled_r["mte2"] * min(scale_aic, scale_aiv))
    data["rates"] = scaled_r
    data["service_modes"] = {
        k: ({**m, "rate": float(m["rate"]) * (scale_aic if m.get("resource") in aic_r else scale_aiv)}
            if k not in explicit_modes and "rate" in m
            and m.get("resource") in (aic_r | aiv_r)
            and m.get("resource") not in explicit_rates
            else dict(m))
        for k, m in data.get("service_modes", {}).items()
    }


def _alias_cores(mapping: Mapping) -> dict:
    out = dict(mapping)
    if "compute_units" not in out and "aic_count" not in out:
        for key in ("num_cores", "cores"):
            if out.get(key) is not None:
                out["compute_units"] = int(out[key])
                break
    return out


def normalize_core_aliases(raw: Mapping) -> dict:
    """Normalize top-level and per-device ``num_cores``/``cores`` aliases."""
    out = _alias_cores(raw)
    if isinstance(out.get("device"), Mapping):
        out["device"] = _alias_cores(out["device"])
    for key in ("devices", "dies"):
        if isinstance(out.get(key), (list, tuple)):
            items = []
            for item in out[key]:
                if isinstance(item, Mapping):
                    rec = _alias_cores(item)
                    if isinstance(rec.get("device"), Mapping):
                        rec["device"] = _alias_cores(rec["device"])
                    items.append(rec)
                else:
                    items.append(item)
            out[key] = items
    return out


def has_core_override(declaration: Mapping) -> bool:
    return bool(declaration.get("compute_units") or declaration.get("resource_units"))


__all__ = [
    "ArchId",
    "CollectiveBandSpec",
    "ComputeArchIdentity",
    "DatapathCapabilitySpec",
    "DeviceInput",
    "DieSpec",
    "DispatchSyncSpec",
    "GENERIC_ARCH_ID",
    "MemoryAccessSpec",
    "RoutePlan",
    "SourceRef",
    "StarsQueueSpec",
    "StarsSpec",
    "SubtileWaveSpec",
    "datapath_for_devices",
    "datapath_for_die",
    "device_input_for_die",
    "ub_peer_edges_for_rank",
    "validate_protocol_bands",
]
