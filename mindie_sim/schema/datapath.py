"""Explicit hardware datapath and ISA capability view derived from ComputeArchSpec.

[Role] Derive immutable per-die datapath capabilities, AIC/AIV owner pairing,
  usable local data routes and upstream source citations strictly from declared
  ``ComputeArchSpec`` and ``Device`` records.
[Invariants] Zero SKU-name, ``arch_family``, ``compute_peaks`` or
  ``service_modes`` capability sniffing. Direct routes (``l0c->ub``,
  ``ub->l1``) exist only when explicitly declared on ``ComputeArchSpec.paths``.
  Representative multi-die stages require homogeneous ``compute_arch`` and
  datapath declarations.
[Timeouts & Locks] Pure metadata derivation over finite device tuples; no I/O
  or locks.
[Side-Effect Policy] Returns new frozen records; never mutates input devices.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NewType

from .common import ModelingError
from .compute_arch import ComputeArchSpec, DataPath
from .hardware import Device, Link

ArchId = NewType("ArchId", str)
DieSpec = Device

GENERIC_ARCH_ID = ArchId("generic")
L0C_TO_UB_OUTPUT_PATH = "l0c_to_ub"


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

    @property
    def arch_id(self) -> ArchId:
        return self.compute_arch.arch_id

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
    return DatapathCapabilitySpec(
        compute_arch=identity,
        aic_count=aic_count,
        aiv_count=aiv_count,
        aic_aiv_pairs=_aic_aiv_pairs(aic_count, aiv_count),
        usable_paths=usable_paths,
        matrix_output_paths=_matrix_output_paths(die, usable_paths),
        ub_peer_edges=tuple(ub_peer_edges),
        stars=_stars_queue(die, usable_paths),
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
    """Return ``(in_stage_peer_count, out_of_stage_peer_count)`` for homogeneity check."""
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


__all__ = [
    "ArchId",
    "ComputeArchIdentity",
    "DatapathCapabilitySpec",
    "DieSpec",
    "GENERIC_ARCH_ID",
    "SourceRef",
    "StarsQueueSpec",
    "datapath_for_devices",
    "datapath_for_die",
    "ub_peer_edges_for_rank",
]
