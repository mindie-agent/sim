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

from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Mapping

from .common import ModelingError, _positive_float
from .compute_arch import ComputeArchSpec
from .compute_peak_aliases import (
    resolve_compute_peak_name, validate_compute_peak_aliases,
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
    _PRESERVED_TOPOLOGY_KEYS,
    _collective_protocols,
    _network_override_keys,
    _neutral_transfer_channel_links,
    _transfer_channel,
    apply_network_overrides,
    calibration_warnings,
)
from .invocation import InvocationTiming
from .physical_mechanisms import (
    PhysicalMechanismSpec, default_physical_mechanisms,
    parse_physical_mechanisms,
)

BITS_PER_BYTE = 8
DEFAULT_ROOT_COMPLEX_ID = "rc0"

NATIVE_WORK_UNITS = {
    **dict.fromkeys(("hbm", "l2", "mte1", "mte2", "mte2_aic", "mte2_aiv", "mte3",
                     "fixpipe", "host_memory", "link"), "bytes"),
    "cube": "flops", "vector": "ops", "host_vector": "ops", "host": "ms",
    "scalar_store_issue": "stores",
}

_CAPABILITY_REQUIRED_FAMILIES = frozenset({"A2", "A3", "A5"})
_HOST_CHANNEL_ALIASES = {"host_to_device": "h2d", "device_to_host": "d2h"}
_SOLO_BW_FIELDS = (
    "solo_effective_uni_bw_gbps", "effective_uni_bw_gbps",
    "calibrated_uni_bw_gbps", "derived_uni_bw_gbps", "uni_peak_bw_gbps",
)
_QUANTITY_SCOPES = {
    "memory": "per_device",
    "buffers": "per_owner",
    "compute_peaks": "device_total",
    "rates": "device_total",
    "resource_units": "owners_per_device",
    "buffer_units": "owners_per_device",
}


def _optional_pos_rate(value, label: str) -> float | None:
    if value is None:
        return None
    return _positive_float(value, f"{label}_invalid", allow_zero=False)


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

    @property
    def capability(self) -> ComputeArchSpec:
        return self.compute_arch

    @property
    def resource_signature(self):
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
                self.hbm_write_bytes_per_s, self.memory_access)

    @property
    def hbm_bytes(self):
        return self.memory_bytes

    @property
    def cube_flops_per_s(self):
        return self.peak("bf16")

    @property
    def aic_count(self):
        return self.compute_units

    def peak(self, dtype):
        peaks = dict(self.compute_peaks)
        supply = dict(self.compute_peak_aliases)
        aliases = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
                   "fp8_e4m3": "fp8", "fp8_e5m2": "fp8"}
        native = (resolve_compute_peak_name(dtype, peaks, supply)
                  if supply else dtype)
        name = native if native in peaks else aliases.get(native, native)
        if self.compute_arch.has_native_modes():
            self.compute_arch.native_mode("matrix", native)
        if name not in peaks:
            raise ModelingError(f"compute_precision_rate_required:{dtype}")
        return peaks[name]

    def units(self, resource):
        return dict(self.resource_units).get(resource, self.compute_units)

    def pool_units(self, pool):
        return dict(self.buffer_units).get(pool, self.compute_units)

    def owner_count(self, resource):
        """Owners of ``resource`` on this device (never a package total)."""
        return self.units(resource)

    def total_rate(self, resource):
        """Declared device-total service of one resource, in its own unit."""
        return dict(self.rates)[resource]

    def owner_rate(self, resource):
        """Per-owner service; idle siblings do not lend their capacity."""
        count = self.owner_count(resource)
        if count < 1:
            raise ModelingError(f"resource_owner_count_nonpositive:{resource}")
        return self.total_rate(resource) / count

    def total_buffer_bytes(self, pool):
        """Per-owner pool capacity times its declared owner count."""
        return self.pool_capacity(pool) * self.pool_units(pool)

    def pool_capacity(self, pool):
        """Declared capacity of one pool instance (per owner)."""
        try:
            return dict(self.buffers)[pool]
        except KeyError as exc:
            raise ModelingError(f"buffer_pool_undeclared:{pool}") from exc

    @staticmethod
    def quantity_scope(quantity):
        """Return the declared accounting scope of a device quantity."""
        try:
            return _QUANTITY_SCOPES[quantity]
        except KeyError as exc:
            raise ModelingError(f"device_quantity_scope_unknown:{quantity}") from exc

    def to_dict(self):
        derived_rates = {s.name for s in self.compute_arch.instruction_services
                         if (s.scale_with or s.clock_hz is not None)
                         and s.name not in self.linked_rate_overrides}
        derived_modes = {s.name for s in self.compute_arch.instruction_services
                         if s.resource and s.name not in self.instruction_mode_overrides}
        declared_pipes = {s.name: s.resource for s in self.compute_arch.instruction_services}
        derived_units = {s.name for s in self.compute_arch.instruction_services
                         if (s.scale_with or s.clock_hz is not None)
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
                     if self.compute_arch.instruction_services
                     or self.compute_arch.has_capabilities() else {}),
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
        return result


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
    def compute_peaks(self):
        return self.dies[0].compute_peaks

    def per_placement_peaks(self, dtype):
        return {device.rank: device.peak(dtype) for device in self.dies}

    def peak_flops_for(self, dtype):
        self._require_homogeneous("peak_flops_for")
        return min(device.peak(dtype) for device in self.dies)

    def declared_peak_key(self, dtype):
        return dtype if dtype in dict(self.compute_peaks) else None

    def package_memory_bytes(self, package: int) -> int:
        return sum(device.memory_bytes for device in self.dies
                   if device.package == package)

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
        for field_name in _SOLO_BW_FIELDS:
            candidate = record.get(field_name)
            if candidate is not None:
                return _positive_float(candidate, f"transfer_channel_bw_invalid:{channel}")
        return None

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
        cal = dict(source_map.get("calibration") or {})
        if (device is not None and not self.homogeneous
                and device.sku and device.sku != self.sku and not cal):
            from mindie_sim.backends.ascend import profiles as ascend_profiles
            prof = ascend_profiles().get(device.sku)
            if isinstance(prof, Mapping):
                cal = dict((prof.get("source") or {}).get("calibration") or {})
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

    def hardware_context(self, rank: int = 0) -> dict:
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

    def validate_placement(self, parallel):
        if parallel.world_size > len(self.dies):
            raise ModelingError("placement_exceeds_physical_devices")
        if len({device.rank for device in self.dies}) != len(self.dies):
            raise ModelingError("device_rank_duplicate")

    @classmethod
    def from_mapping(cls, raw: Mapping):
        from mindie_sim.backends import materialize
        raw_in = _normalize_core_aliases(raw)
        backend = raw_in.get("backend")
        raw_records = list(raw_in.get("devices", raw_in.get("dies", ())) or ())
        raw = materialize(raw_in)
        allowed_semantics = {
            "shared", "per_port", "per_pair", "direct", "fully_connected",
            "mesh", "switched", "unknown", "none",
        }
        for name, channel in (raw.get("transfer_channels") or {}).items():
            if not isinstance(channel, Mapping):
                continue
            for field_name in ("channel_semantics", "semantics"):
                if field_name in channel and channel[field_name] not in allowed_semantics:
                    raise ModelingError(
                        f"transfer_channel_semantics_unsupported:{name}:{channel[field_name]}")
            if "scope" in channel and channel["scope"] is not None:
                from .hardware_links import normalize_link_scope
                normalize_link_scope(channel["scope"])
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
            for key in ("rank", "node", "package", "root_complex_id"):
                value = explicit_record.get(key)
                if value is not None:
                    data[key] = value
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
            arch = _resolve_device_compute_arch(data, raw, arch_family)
            peaks = dict(data.get("compute_peaks", {}))
            if arch.has_capabilities() and peaks and not arch.has_native_modes():
                raise ModelingError(
                    "compute_peaks_without_native_capability:"
                    f"{data.get('sku', arch_family)}")
            raw_aliases = data.get("compute_peak_aliases")
            if raw_aliases is None:
                raw_aliases = (explicit_record.get("compute_peak_aliases")
                               or raw_in.get("compute_peak_aliases") or {})
            aliases = dict(raw_aliases) if isinstance(raw_aliases, Mapping) else raw_aliases
            validate_compute_peak_aliases(peaks, aliases, arch)
            if arch.has_native_modes():
                arch.validate_peaks(peaks, aliases=aliases)
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
            rank, node, package_id = placements[index]
            devices.append(Device(
                rank=int(data.get("rank", rank)), memory_bytes=memory,
                compute_peaks=tuple(sorted(peaks.items())),
                rates=tuple(sorted(rates.items())),
                buffers=tuple(sorted(buffers.items())),
                compute_units=int(data.get("compute_units", 1) or 1),
                launch_us=float(data.get("launch_us", 0)),
                node=int(data.get("node", node)),
                package=int(data.get("package", package_id)),
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
                compute_peak_aliases=tuple(sorted(dict(
                    raw_aliases if isinstance(raw_aliases, Mapping) else
                    tuple(raw_aliases or ())).items())),
                linked_rate_overrides=tuple(sorted(
                    s.name for s in arch.instruction_services
                    if (s.scale_with or s.clock_hz is not None)
                    and s.name in data.get("rates", {}))),
                instruction_mode_overrides=tuple(sorted(
                    s.name for s in arch.instruction_services
                    if s.resource and "rate" in data.get("service_modes", {}).get(s.name, {}))),
                instruction_unit_overrides=tuple(sorted(
                    s.name for s in arch.instruction_services
                    if (s.scale_with or s.clock_hz is not None)
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
            ))
        links = tuple(Link.from_mapping(link, default_members=range(len(devices)))
                      for link in raw.get("links", ()))
        if not backend or _network_override_keys(raw):
            scale = (GBPS_TO_BYTES_PER_MS / BITS_PER_BYTE
                     if backend == "ascend" and not raw.get("topology_preset")
                     else GBPS_TO_BYTES_PER_MS)
            links = apply_network_overrides(links, raw, gbps_to_bytes_per_ms=scale)
        for link in links:
            if link.semantics not in {"shared", "per_port", "per_pair", "direct", "fully_connected", "mesh"}:
                raise ModelingError(f"link_semantics_unknown:{link.semantics}")
            if not math.isfinite(link.capacity_bytes_per_ms) or link.capacity_bytes_per_ms <= 0:
                raise ModelingError(f"link_rate_invalid:{link.name}")
            if not set(link.members) <= {device.rank for device in devices}:
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

    def to_dict(self):
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


def parse_hardware(raw: Mapping) -> Hardware:
    """Parse a hardware declaration, including top-level network overrides."""
    if not isinstance(raw, Mapping):
        raise ModelingError("hardware_mapping_required")
    if raw.get("backend") or raw.get("links"):
        return Hardware.from_mapping(raw)
    links = _neutral_transfer_channel_links(raw)
    if not links:
        return Hardware.from_mapping(raw)
    return Hardware.from_mapping({**raw, "links": links})
