"""Normalize explicit device declarations and Vol-2 hardware capability specs.

[Role] Merge profile, machine and per-device inputs in physical resource units,
  resolve explicit placement tuples, and parse typed ``SubtileWaveSpec``,
  ``StarsSpec``, ``DispatchSyncSpec``, ``MemoryAccessSpec`` and ``DeviceInput``
  records.
[Invariants] A later declaration overrides an inherited value. Within one
  declaration, neutral fields override native aliases. Partial resource maps
  and typed physical-mechanism sections preserve inherited siblings; rates
  never scale with a model label. All capability dimensions, queue depths and
  bandwidths are strictly validated against finite physical domains.
[Timeouts & Locks] Bounded metadata arithmetic; no waits or locks.
[Side-Effect Policy] Returns new mappings and frozen dataclasses without
  mutating caller inputs.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from .common import ModelingError, _positive_float


BYTES_PER_GIB = 1024 * 1024 * 1024
ASCEND_SECTOR_BYTES = 32

STARS_RELEASE_POLICIES = frozenset({
    "descriptor_ready",
    "stream_predecessor_done",
})

_MAP_FIELDS = (
    "compute_peaks", "rates", "buffers", "resource_units", "buffer_units",
    "work_units", "kernel_invocations", "store_queues",
)
_PHYSICAL_MECHANISM_SECTIONS = (
    "memory_hierarchy", "matrix", "attention", "indexer", "vector", "collective",
)
_ARCH_FIELDS = ("architecture", "arch_family")
_RATE_FIELDS = {
    "hbm_bw_bytes_per_s": ("hbm",),
    "vector_ops_per_s": ("vector",),
    "mte_bw_bytes_per_s": ("mte2", "mte2_aic", "mte2_aiv", "mte3"),
    "mte1_bw_bytes_per_s": ("mte1",),
    "mte2_bw_bytes_per_s": ("mte2",),
    "mte2_aic_bw_bytes_per_s": ("mte2_aic",),
    "mte2_aiv_bw_bytes_per_s": ("mte2_aiv",),
    "mte3_bw_bytes_per_s": ("mte3",),
    "fixpipe_bw_bytes_per_s": ("fixpipe",),
}
_RATE_ALIASES = {"memory": "hbm", "read": "mte2", "write": "mte3"}
_POOLS = ("l1", "l0a", "l0b", "l0c", "ub")
_CAPABILITY_MAP_FIELDS = (
    "instruction_services", "representations", "native_compute", "casts",
    "paths", "runtimes",
)
_DATAPATH_DEVICE_FIELDS = (
    "simt_ub_carveout_bytes", "subtile_wave", "dispatch_sync",
    "hbm_rw_alpha", "hbm_read_bytes_per_s", "hbm_write_bytes_per_s",
    "memory_access", "root_complex_id",
)


def _req_pos_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"{label}_positive_int_required:{value}")
    return int(value)


def _req_nonneg_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"{label}_nonneg_int_required:{value}")
    return int(value)


def _req_nonneg_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelingError(f"{label}_nonneg_float_required:{value}")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ModelingError(f"{label}_nonneg_float_required:{value}")
    return parsed


def _req_nonempty_str(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(f"{label}_nonempty_str_required:{value}")
    return value.strip()


@dataclass(frozen=True, slots=True)
class SubtileWaveSpec:
    """Declared sub-tile wave quantization geometry for one native mode."""

    native_mode: str
    tile_m: int
    tile_n: int
    tile_k: int
    independently_issuable_m: int
    owner_groups: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "native_mode", _req_nonempty_str(self.native_mode, "subtile_wave.native_mode"))
        object.__setattr__(self, "tile_m", _req_pos_int(self.tile_m, "subtile_wave.tile_m"))
        object.__setattr__(self, "tile_n", _req_pos_int(self.tile_n, "subtile_wave.tile_n"))
        object.__setattr__(self, "tile_k", _req_pos_int(self.tile_k, "subtile_wave.tile_k"))
        object.__setattr__(
            self,
            "independently_issuable_m",
            _req_pos_int(self.independently_issuable_m, "subtile_wave.independently_issuable_m"),
        )
        object.__setattr__(self, "owner_groups", _req_pos_int(self.owner_groups, "subtile_wave.owner_groups"))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> SubtileWaveSpec:
        if not isinstance(raw, Mapping):
            raise ModelingError("subtile_wave_mapping_required")
        return cls(
            native_mode=raw.get("native_mode", ""),
            tile_m=raw.get("tile_m"),
            tile_n=raw.get("tile_n"),
            tile_k=raw.get("tile_k"),
            independently_issuable_m=raw.get("independently_issuable_m"),
            owner_groups=raw.get("owner_groups"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_mode": self.native_mode,
            "tile_m": self.tile_m,
            "tile_n": self.tile_n,
            "tile_k": self.tile_k,
            "independently_issuable_m": self.independently_issuable_m,
            "owner_groups": self.owner_groups,
        }


@dataclass(frozen=True, slots=True)
class StarsSpec:
    """Device STARS hardware descriptor-issue FIFO queue specification."""

    queue_scope: str
    queue_depth: int
    lanes: int
    issue_interval_us: float
    first_issue_us: float
    release_policy: str
    evidence_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "queue_scope", _req_nonempty_str(self.queue_scope, "stars.queue_scope"))
        object.__setattr__(self, "queue_depth", _req_pos_int(self.queue_depth, "stars.queue_depth"))
        object.__setattr__(self, "lanes", _req_pos_int(self.lanes, "stars.lanes"))
        object.__setattr__(self, "issue_interval_us", _req_nonneg_float(self.issue_interval_us, "stars.issue_interval_us"))
        object.__setattr__(self, "first_issue_us", _req_nonneg_float(self.first_issue_us, "stars.first_issue_us"))
        policy = _req_nonempty_str(self.release_policy, "stars.release_policy")
        if policy not in STARS_RELEASE_POLICIES:
            raise ModelingError(f"stars_release_policy_unsupported:{policy}")
        object.__setattr__(self, "release_policy", policy)
        object.__setattr__(self, "evidence_id", _req_nonempty_str(self.evidence_id, "stars.evidence_id"))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StarsSpec:
        if not isinstance(raw, Mapping):
            raise ModelingError("stars_spec_mapping_required")
        return cls(
            queue_scope=raw.get("queue_scope", "die"),
            queue_depth=raw.get("queue_depth"),
            lanes=raw.get("lanes"),
            issue_interval_us=raw.get("issue_interval_us", 0.0),
            first_issue_us=raw.get("first_issue_us", 0.0),
            release_policy=raw.get("release_policy", "descriptor_ready"),
            evidence_id=raw.get("evidence_id", "undeclared"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "queue_scope": self.queue_scope,
            "queue_depth": self.queue_depth,
            "lanes": self.lanes,
            "issue_interval_us": self.issue_interval_us,
            "first_issue_us": self.first_issue_us,
            "release_policy": self.release_policy,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True, slots=True)
class DispatchSyncSpec:
    """Device STARS descriptor queue and host D2H/submission sync bindings."""

    stars: StarsSpec
    host_submit_service: str
    d2h_channel: str
    split_host_service: str

    def __post_init__(self) -> None:
        if not isinstance(self.stars, StarsSpec):
            raise ModelingError("dispatch_sync_stars_required")
        object.__setattr__(
            self, "host_submit_service", _req_nonempty_str(self.host_submit_service, "dispatch_sync.host_submit_service")
        )
        object.__setattr__(self, "d2h_channel", _req_nonempty_str(self.d2h_channel, "dispatch_sync.d2h_channel"))
        object.__setattr__(
            self, "split_host_service", _req_nonempty_str(self.split_host_service, "dispatch_sync.split_host_service")
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> DispatchSyncSpec:
        if not isinstance(raw, Mapping):
            raise ModelingError("dispatch_sync_mapping_required")
        stars_raw = raw.get("stars")
        stars = stars_raw if isinstance(stars_raw, StarsSpec) else StarsSpec.from_mapping(stars_raw)
        return cls(
            stars=stars,
            host_submit_service=raw.get("host_submit_service", "host"),
            d2h_channel=raw.get("d2h_channel", "d2h"),
            split_host_service=raw.get("split_host_service", "host"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stars": self.stars.to_dict(),
            "host_submit_service": self.host_submit_service,
            "d2h_channel": self.d2h_channel,
            "split_host_service": self.split_host_service,
        }


@dataclass(frozen=True, slots=True)
class MemoryAccessBurstPenalty:
    """Burst-turnaround byte penalty for one mutually exclusive stride domain."""

    domain: str
    min_stride_bytes: int
    max_stride_bytes: int | None
    penalty_bytes: float
    evidence_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", _req_nonempty_str(self.domain, "burst_turn_penalty.domain"))
        min_s = _req_nonneg_int(self.min_stride_bytes, "burst_turn_penalty.min_stride_bytes")
        object.__setattr__(self, "min_stride_bytes", min_s)
        if self.max_stride_bytes is not None:
            max_s = _req_pos_int(self.max_stride_bytes, "burst_turn_penalty.max_stride_bytes")
            if max_s <= min_s:
                raise ModelingError(f"burst_turn_penalty_stride_range_invalid:{min_s}:{max_s}")
            object.__setattr__(self, "max_stride_bytes", max_s)
        object.__setattr__(self, "penalty_bytes", _req_nonneg_float(self.penalty_bytes, "burst_turn_penalty.penalty_bytes"))
        object.__setattr__(self, "evidence_id", _req_nonempty_str(self.evidence_id, "burst_turn_penalty.evidence_id"))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MemoryAccessBurstPenalty:
        if not isinstance(raw, Mapping):
            raise ModelingError("burst_turn_penalty_mapping_required")
        return cls(
            domain=raw.get("domain", "contiguous"),
            min_stride_bytes=raw.get("min_stride_bytes", 0),
            max_stride_bytes=raw.get("max_stride_bytes"),
            penalty_bytes=raw.get("penalty_bytes", 0.0),
            evidence_id=raw.get("evidence_id", "undeclared"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "min_stride_bytes": self.min_stride_bytes,
            "max_stride_bytes": self.max_stride_bytes,
            "penalty_bytes": self.penalty_bytes,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True, slots=True)
class MemoryAccessSpec:
    """DRAM sector granularity and burst-turnaround penalty table."""

    sector_bytes: int = ASCEND_SECTOR_BYTES
    burst_turn_penalty: tuple[MemoryAccessBurstPenalty, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sector_bytes", _req_pos_int(self.sector_bytes, "memory_access.sector_bytes"))
        penalties = tuple(
            item if isinstance(item, MemoryAccessBurstPenalty) else MemoryAccessBurstPenalty.from_mapping(item)
            for item in self.burst_turn_penalty
        )
        ordered = sorted(penalties, key=lambda item: (item.min_stride_bytes, item.domain))
        for left, right in zip(ordered, ordered[1:]):
            if left.max_stride_bytes is None or right.min_stride_bytes < left.max_stride_bytes:
                raise ModelingError(
                    f"burst_turn_penalty_domains_overlap:{left.domain}:{right.domain}"
                )
        object.__setattr__(self, "burst_turn_penalty", tuple(ordered))

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MemoryAccessSpec:
        if not isinstance(raw, Mapping):
            raise ModelingError("memory_access_mapping_required")
        return cls(
            sector_bytes=raw.get("sector_bytes", ASCEND_SECTOR_BYTES),
            burst_turn_penalty=tuple(raw.get("burst_turn_penalty") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector_bytes": self.sector_bytes,
            "burst_turn_penalty": [item.to_dict() for item in self.burst_turn_penalty],
        }


@dataclass(frozen=True, slots=True)
class DeviceInput:
    """Normalized typed device capability and supply summary."""

    compute_arch: Any
    datapath: Any
    kernel_invocations: tuple[tuple[str, Any], ...]
    transfer_channels: tuple[tuple[str, tuple[tuple[str, Any], ...]], ...]
    rates: tuple[tuple[str, float], ...] = ()
    units: tuple[tuple[str, int], ...] = ()
    buffers: tuple[tuple[str, int], ...] = ()
    peaks: tuple[tuple[str, float], ...] = ()
    provenance: tuple[tuple[str, Any], ...] = ()


def parse_subtile_waves(raw: Sequence[Any] | None) -> tuple[SubtileWaveSpec, ...]:
    """Parse and verify unique native_mode sub-tile wave declarations."""
    if not raw:
        return ()
    specs = tuple(
        item if isinstance(item, SubtileWaveSpec) else SubtileWaveSpec.from_mapping(item)
        for item in raw
    )
    seen: set[str] = set()
    for spec in specs:
        if spec.native_mode in seen:
            raise ModelingError(f"subtile_wave_duplicate_native_mode:{spec.native_mode}")
        seen.add(spec.native_mode)
    return specs


def parse_dispatch_sync(raw: Any) -> DispatchSyncSpec | None:
    """Parse an optional ``device.dispatch_sync`` mapping."""
    if raw is None:
        return None
    if isinstance(raw, DispatchSyncSpec):
        return raw
    return DispatchSyncSpec.from_mapping(raw)


def parse_memory_access(raw: Any) -> MemoryAccessSpec | None:
    """Parse an optional ``device.memory_access`` mapping."""
    if raw is None:
        return None
    if isinstance(raw, MemoryAccessSpec):
        return raw
    return MemoryAccessSpec.from_mapping(raw)


def validate_hbm_rw_alpha(value: Any) -> float:
    """Validate finite HBM read/write turnaround alpha in ``[0, 1]``."""
    alpha = _req_nonneg_float(value, "hbm_rw_alpha")
    if alpha > 1.0:
        raise ModelingError(f"hbm_rw_alpha_exceeds_unity:{alpha}")
    return alpha


def machine_device_defaults(raw):
    """Extract common device supply; rank/node/package remain placement fields."""
    fields = {
        *_MAP_FIELDS, *_ARCH_FIELDS, "service_modes", "compute_arch",
        "physical_mechanisms", *_RATE_FIELDS, *_DATAPATH_DEVICE_FIELDS,
        "memory_bytes", "hbm_gib", "hbm_bytes", "cube_flops_per_s",
        "compute_units", "aic_count", "aiv_count", "launch_us",
        *(pool + "_bytes" for pool in _POOLS),
    }
    return {key: value for key, value in raw.items() if key in fields}


def _normalize(raw):
    data = dict(raw)
    if "architecture" in data:
        architecture = data.pop("architecture")
        if architecture is not None:
            data["arch_family"] = architecture
    rates = {}
    for field_name, roles in _RATE_FIELDS.items():
        if field_name in data:
            value = data.pop(field_name)
            rates.update((role, value) for role in roles)
    explicit = data.get("rates", {})
    rates.update((_RATE_ALIASES[key], value) for key, value in explicit.items()
                 if key in _RATE_ALIASES)
    rates.update((key, value) for key, value in explicit.items()
                 if key not in _RATE_ALIASES)
    if rates or "rates" in data:
        data["rates"] = rates

    if "cube_flops_per_s" in data:
        value = data.pop("cube_flops_per_s")
        data["compute_peaks"] = {
            "bf16": value, "fp16": value, **data.get("compute_peaks", {}),
        }
    memory = {}
    if "hbm_gib" in data:
        memory["memory_bytes"] = float(data.pop("hbm_gib")) * BYTES_PER_GIB
    if "hbm_bytes" in data:
        memory["memory_bytes"] = data.pop("hbm_bytes")
    data = {**memory, **data}
    if "aic_count" in data:
        count = data.pop("aic_count")
        data.setdefault("compute_units", count)
    if "num_cores" in data or "cores" in data:
        count = data.pop("num_cores", None)
        cores = data.pop("cores", None)
        count = count if count is not None else cores
        if count is not None:
            data.setdefault("compute_units", count)
    if "aiv_count" in data:
        count = data.pop("aiv_count")
        data["resource_units"] = {"vector": count, **data.get("resource_units", {})}
    buffers = {pool: data.pop(pool + "_bytes")
               for pool in _POOLS if pool + "_bytes" in data}
    if buffers:
        data["buffers"] = {**buffers, **data.get("buffers", {})}
    return data


def _deep_merge(base, patch):
    """Merge nested mappings recursively; lists and scalars replace."""
    result = dict(base) if isinstance(base, Mapping) else {}
    for key, value in dict(patch).items():
        previous = result.get(key)
        result[key] = (_deep_merge(previous, value)
                       if isinstance(previous, Mapping) and isinstance(value, Mapping)
                       else value)
    return result


def _merge_physical_mechanisms(base, declared):
    """Merge one typed-mechanism patch section-wise over the inherited record."""
    if not isinstance(declared, Mapping):
        return declared
    inherited = base if isinstance(base, Mapping) else {}
    merged = dict(inherited)
    for section in _PHYSICAL_MECHANISM_SECTIONS:
        if section not in declared:
            continue
        patch = declared[section]
        if patch is None:
            merged[section] = None
        elif section == "matrix":
            merged[section] = _merge_matrix(inherited.get(section), patch)
        else:
            merged[section] = _deep_merge(inherited.get(section, {}), patch)
    for key, value in declared.items():
        if key not in _PHYSICAL_MECHANISM_SECTIONS:
            merged[key] = value
    return merged


def _merge_matrix(inherited, declared):
    """Merge matrix fields, replacing the dtype schedule as one declaration."""
    if not isinstance(declared, Mapping):
        return declared
    merged = dict(inherited) if isinstance(inherited, Mapping) else {}
    for key, value in declared.items():
        if key == "micro_tiles_by_dtype" and isinstance(value, Mapping):
            merged[key] = dict(value)
        else:
            merged[key] = value
    return merged


def _merge_service_modes(base, declared, ratios):
    """Keep inherited mode/base supply ratios under explicit rate overrides."""
    merged = {name: dict(mode) for name, mode in base.items()}
    for name, mode in merged.items():
        if "rate" in mode and mode.get("resource") in ratios:
            merged[name] = {**mode, "rate": mode["rate"] * ratios[mode["resource"]]}
    for name, mode in (declared or {}).items():
        merged[name] = {**merged.get(name, {}), **mode}
    return merged


def merge_device_inputs(*declarations):
    """Canonicalize each layer before later aliases override earlier defaults."""
    result = {}
    for raw in declarations:
        data = _normalize(raw)
        changed_preset = False
        if "compute_arch" in data:
            previous_arch = result.get("compute_arch", {})
            current_arch = data["compute_arch"]
            changed_preset = bool(current_arch.get("preset")
                                  and previous_arch.get("preset")
                                  and current_arch["preset"]
                                  != previous_arch["preset"])
            base = {} if changed_preset else previous_arch
            merged_arch = {**base, **current_arch}
            for field_name in _CAPABILITY_MAP_FIELDS:
                if field_name in base and field_name in current_arch:
                    merged_arch[field_name] = {**base[field_name], **current_arch[field_name]}
                elif field_name in base and field_name not in current_arch:
                    merged_arch[field_name] = base[field_name]
            data["compute_arch"] = merged_arch
        if changed_preset and isinstance(result.get("physical_mechanisms"), Mapping):
            inherited = dict(result["physical_mechanisms"])
            inherited.pop("matrix", None)
            result["physical_mechanisms"] = inherited
        if "physical_mechanisms" in data and data["physical_mechanisms"] is None:
            if result.get("physical_mechanisms") is not None:
                data.pop("physical_mechanisms")
        if "dispatch_sync" in data and isinstance(data["dispatch_sync"], Mapping):
            if isinstance(result.get("dispatch_sync"), Mapping):
                data["dispatch_sync"] = _deep_merge(
                    result["dispatch_sync"], data["dispatch_sync"]
                )
        previous = result.get("rates", {})
        ratios = {}
        for role, new_rate in data.get("rates", {}).items():
            old_rate = previous.get(role)
            if (isinstance(old_rate, (int, float)) and isinstance(new_rate, (int, float))
                    and not isinstance(old_rate, bool) and not isinstance(new_rate, bool)
                    and old_rate > 0 and new_rate > 0):
                ratios[role] = new_rate / old_rate
        if ratios or "service_modes" in data:
            data["service_modes"] = _merge_service_modes(
                result.get("service_modes", {}), data.get("service_modes"), ratios)
        for field_name in _MAP_FIELDS:
            if field_name in data:
                if field_name == "store_queues":
                    data[field_name] = {**result.get(field_name, {}), **{
                        name: {**result.get(field_name, {}).get(name, {}), **queue}
                        for name, queue in data[field_name].items()}}
                else:
                    data[field_name] = {**result.get(field_name, {}), **data[field_name]}
        if "physical_mechanisms" in data:
            data["physical_mechanisms"] = _merge_physical_mechanisms(
                result.get("physical_mechanisms"), data["physical_mechanisms"])
        result.update(data)
    return result


def resolve_device_placements(records, count, ranks_per_node, dies_per_package):
    """Resolve one ``(rank, node, package)`` fact for every placement index."""
    records = list(records or ())
    width = max(1, int(ranks_per_node))
    per_package = max(1, int(dies_per_package))
    packages_per_node = max(1, math.ceil(width / per_package))
    bases, offsets, placements = {}, {}, []
    for index in range(int(count)):
        record = (records[index] if index < len(records)
                  and isinstance(records[index], Mapping) else {})
        rank = int(record.get("rank", index))
        node = int(record.get("node", index // width))
        if node not in bases:
            bases[node] = len(bases) * packages_per_node
            offsets[node] = 0
        package = record.get("package")
        if package is None:
            package = bases[node] + offsets[node] // per_package
            offsets[node] += 1
        placements.append((rank, node, int(package)))
    return tuple(placements)


def placement_package_groups(placements):
    """Actual rank groups sharing one ``(node, package)``."""
    groups, positions = [], {}
    for first, (rank, node, package) in enumerate(placements):
        key = (node, package)
        if key not in positions:
            positions[key] = len(groups)
            groups.append((first, []))
        groups[positions[key]][1].append(rank)
    return [(first, members) for first, members in groups]


def placement_node_groups(placements, width=None):
    """Actual ranks per node, optionally chunked into structural tiers."""
    nodes, groups = {}, []
    for first, (rank, node, _package) in enumerate(placements):
        nodes.setdefault(node, []).append((first, rank))
    for members in nodes.values():
        step = len(members) if width is None else max(1, int(width))
        for start in range(0, len(members), step):
            chunk = members[start:start + step]
            groups.append((chunk[0][0], [rank for _, rank in chunk]))
    return groups


def placement_node_order(placements):
    """``(index, rank)`` pairs grouped by node, preserving explicit rank order."""
    ordered, nodes = [], {}
    for first, (rank, node, _package) in enumerate(placements):
        if node not in nodes:
            nodes[node] = len(ordered)
            ordered.append([])
        ordered[nodes[node]].append((first, rank))
    return [item for group in ordered for item in group]


def placement_node_order_groups(placements, size=None):
    """Actual ranks grouped by node, optionally chunked into SuperPod domains."""
    ordered = placement_node_order(placements)
    if size is None:
        return [(0, [rank for _, rank in ordered])] if ordered else []
    return [(ordered[start][0], [rank for _, rank in ordered[start:start + size]])
            for start in range(0, len(ordered), size)]


def resolve_same_pod_latency(raw, preset, declared, fallback):
    """Latency precedence independent of the same-pod bandwidth declaration."""
    if {"latency_us", "alpha_us"} & declared.get("d2d_inter_node", set()):
        return fallback
    for source in (raw, preset):
        for key in ("superpod_latency_us", "inter_node_latency_us"):
            if source.get(key) is not None:
                return _positive_float(source[key], f"{key}_invalid", allow_zero=True)
    return fallback


def resolve_same_pod_tier(combined, raw, preset, declared, explicit,
                          bw_caller, inter_tier, inter_latency):
    """Same-pod peak/util/capacity/latency, independent of any link cut."""
    peak, util, effective = inter_tier
    if not explicit:
        if raw.get("superpod_bw_gbps") is not None:
            peak = _positive_float(raw["superpod_bw_gbps"],
                                   "superpod_bw_gbps_invalid", allow_zero=True)
            effective = peak * util
        elif (not bw_caller
              and combined.get("superpod_bw_gbps") is not None):
            peak = _positive_float(combined["superpod_bw_gbps"],
                                   "superpod_bw_gbps_invalid", allow_zero=True)
            effective = peak * util
    return (peak, util, effective,
            resolve_same_pod_latency(raw, preset, declared, inter_latency))


__all__ = [
    "ASCEND_SECTOR_BYTES",
    "BYTES_PER_GIB",
    "DeviceInput",
    "DispatchSyncSpec",
    "MemoryAccessBurstPenalty",
    "MemoryAccessSpec",
    "STARS_RELEASE_POLICIES",
    "StarsSpec",
    "SubtileWaveSpec",
    "_deep_merge",
    "machine_device_defaults",
    "merge_device_inputs",
    "parse_dispatch_sync",
    "parse_memory_access",
    "parse_subtile_waves",
    "placement_node_groups",
    "placement_node_order",
    "placement_node_order_groups",
    "placement_package_groups",
    "resolve_device_placements",
    "resolve_same_pod_latency",
    "resolve_same_pod_tier",
    "validate_hbm_rw_alpha",
]
