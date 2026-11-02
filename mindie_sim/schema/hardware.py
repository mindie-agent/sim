"""Device-independent resource descriptions.

[Role] Describe compute rates, memory, buffers, submission and physical links.
[Invariants] Vendor names are labels. The declared resource graph, rates and
  storage capacities determine execution. Precision peaks are never inferred
  from a vendor or a model name. A link's effective capacity is always
  ``peak_capacity * bw_utilization``; serialized links omit the utilization
  keys when utilization is unity so calibration identities stay stable.
[Timeouts & Locks] Pure metadata parsing, no driver calls or device contexts.
[Side-Effect Policy] Immutable input records.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Mapping

from .common import ModelingError, _positive_float, _unknown
from .device_input import machine_device_defaults, merge_device_inputs
from .invocation import InvocationTiming
from .compute_arch import ComputeArchSpec


NATIVE_WORK_UNITS = {
    **dict.fromkeys(("hbm", "mte1", "mte2", "mte2_aic", "mte2_aiv", "mte3",
                     "fixpipe", "host_memory", "link"), "bytes"),
    "cube": "flops", "vector": "ops", "host_vector": "ops", "host": "ms",
    "scalar_store_issue": "stores",
}


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
        aliases = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
                   "fp8_e4m3": "fp8", "fp8_e5m2": "fp8"}
        name = dtype if dtype in peaks else aliases.get(dtype, dtype)
        if name not in peaks:
            raise ModelingError(f"compute_precision_rate_required:{dtype}")
        return peaks[name]

    def units(self, resource):
        return dict(self.resource_units).get(resource, self.compute_units)

    def pool_units(self, pool):
        return dict(self.buffer_units).get(pool, self.compute_units)

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
        return {"rank": self.rank, "memory_bytes": self.memory_bytes,
                "compute_peaks": dict(self.compute_peaks),
                # Linked supplies remain declarations on round trip. Emitting
                # their derived values as explicit overrides freezes a sweep.
                "rates": {k: v for k, v in self.rates if k not in derived_rates},
                "buffers": dict(self.buffers), "compute_units": self.compute_units,
                "launch_us": self.launch_us, "node": self.node, "package": self.package,
                "resource_units": {k: v for k, v in self.resource_units if k not in derived_units},
                "buffer_units": dict(self.buffer_units),
                "service_modes": {name: {"resource": role,
                                         **({"rate": rate} if name not in derived_modes else {})}
                                  for name, (role, rate) in self.service_modes
                                  if name not in derived_modes or role != declared_pipes[name]},
                **({"compute_arch": self.compute_arch.to_dict()}
                   if self.compute_arch.instruction_services else {}),
                **({"work_units": dict(self.work_units)} if self.work_units else {}),
                **({"store_queues": {name: dict(queue) for name, queue in self.store_queues}}
                   if self.store_queues else {}),
                **({"kernel_invocations": {key: timing.to_dict()
                                          for key, timing in self.kernel_invocations}}
                   if self.kernel_invocations else {})}


@dataclass(frozen=True)
class Link:
    name: str
    capacity_bytes_per_ms: float
    members: tuple[int, ...]
    latency_us: float = 0
    channel: str = "fabric"
    semantics: str = "shared"
    transaction_bytes: int | None = None
    alignment_bytes: int = 1
    protocol_overhead_bytes: int = 0
    switch_hops: int = 0
    switch_hop_latency_us: float = 0
    switch_capacity_bytes_per_ms: float | None = None
    bw_utilization: float = 1.0
    peak_capacity_bytes_per_ms: float | None = None

    def __post_init__(self):
        for field_name, minimum in (("transaction_bytes", 1), ("alignment_bytes", 1),
                                    ("protocol_overhead_bytes", 0)):
            value = getattr(self, field_name)
            if field_name == "transaction_bytes" and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ModelingError(f"link_{field_name}_invalid:{self.name}")
        if self.transaction_bytes is None and (self.alignment_bytes != 1 or self.protocol_overhead_bytes):
            raise ModelingError(f"link_framing_requires_transaction_bytes:{self.name}")
        if (not isinstance(self.switch_hops, int) or isinstance(self.switch_hops, bool)
                or self.switch_hops < 0 or not math.isfinite(self.switch_hop_latency_us)
                or self.switch_hop_latency_us < 0):
            raise ModelingError(f"link_switch_latency_invalid:{self.name}")
        if (self.switch_capacity_bytes_per_ms is not None
                and (not math.isfinite(self.switch_capacity_bytes_per_ms)
                     or self.switch_capacity_bytes_per_ms <= 0)):
            raise ModelingError(f"link_switch_capacity_invalid:{self.name}")
        utilization = self.bw_utilization
        if (not isinstance(utilization, (int, float)) or isinstance(utilization, bool)
                or not math.isfinite(float(utilization)) or float(utilization) <= 0):
            raise ModelingError(f"link_bw_utilization_invalid:{self.name}")
        if float(utilization) > 1.0:
            raise ModelingError("link_bw_utilization_exceeds_unity")
        if self.peak_capacity_bytes_per_ms is not None:
            peak = self.peak_capacity_bytes_per_ms
            if (not isinstance(peak, (int, float)) or isinstance(peak, bool)
                    or not math.isfinite(float(peak)) or float(peak) <= 0):
                raise ModelingError(f"link_peak_capacity_invalid:{self.name}")

    @property
    def propagation_us(self):
        return self.latency_us + self.switch_hops * self.switch_hop_latency_us

    @property
    def effective_peak_capacity_bytes_per_ms(self) -> float:
        if self.peak_capacity_bytes_per_ms is not None:
            return float(self.peak_capacity_bytes_per_ms)
        return float(self.capacity_bytes_per_ms)

    def to_dict(self):
        result = {"name": self.name, "capacity_bytes_per_ms": self.capacity_bytes_per_ms,
                  "members": self.members, "latency_us": self.latency_us,
                  "channel": self.channel, "semantics": self.semantics,
                  **({"switch_hops": self.switch_hops,
                      "switch_hop_latency_us": self.switch_hop_latency_us}
                     if self.switch_hops or self.switch_hop_latency_us else {}),
                  **({"switch_capacity_bytes_per_ms": self.switch_capacity_bytes_per_ms}
                     if self.switch_capacity_bytes_per_ms is not None else {}),
                  **({"transaction_bytes": self.transaction_bytes,
                      "alignment_bytes": self.alignment_bytes,
                      "protocol_overhead_bytes": self.protocol_overhead_bytes}
                     if self.transaction_bytes is not None else {})}
        # Utilization and peak are only serialized when they add information.
        # Existing calibration identities hash this dictionary, so unity must
        # keep the historical key set byte-for-byte.
        if not math.isclose(self.bw_utilization, 1.0):
            result["bw_utilization"] = self.bw_utilization
            result["peak_capacity_bytes_per_ms"] = self.effective_peak_capacity_bytes_per_ms
        return result

    @classmethod
    def from_mapping(cls, value: Mapping, *, default_members=()) -> "Link":
        if not isinstance(value, Mapping):
            raise ModelingError("link_mapping_required")
        allowed = {
            "name", "capacity_bytes_per_ms", "capacity_gbps", "gbps", "members",
            "latency_us", "channel", "semantics", "transport_semantics",
            "transaction_bytes", "alignment_bytes", "protocol_overhead_bytes",
            "switch_hops", "switch_hop_latency_us", "switch_capacity_bytes_per_ms",
            "bw_utilization", "bw_ratio", "utilization",
            "peak_capacity_bytes_per_ms", "protocol", "transport_mode", "notes",
            "bandwidth_provenance", "channel_semantics", "alpha_us",
            "uni_peak_bw_gbps", "bi_peak_bw_gbps", "calibrated_uni_bw_gbps",
            "calibrated_bi_bw_gbps", "derived_uni_bw_gbps", "derived_bi_bw_gbps",
            "effective_uni_bw_gbps", "effective_bi_bw_gbps",
            "pageable_uni_bw_gbps",
        }
        _unknown(value, allowed, "link")
        name = value.get("name", "link")
        if not isinstance(name, str) or not name:
            raise ModelingError(f"link_name_invalid:{name}")
        utilization = 1.0
        for alias in ("bw_utilization", "bw_ratio", "utilization"):
            if alias not in value:
                continue
            candidate = _positive_float(value[alias], "link_bw_utilization_invalid",
                                        allow_zero=False)
            if candidate > 1.0:
                raise ModelingError("link_bw_utilization_exceeds_unity")
            utilization = candidate
            break
        peak = value.get("peak_capacity_bytes_per_ms")
        if peak is not None:
            peak = _positive_float(peak, f"link_peak_capacity_invalid:{name}")
        if "capacity_bytes_per_ms" in value:
            capacity = _positive_float(value["capacity_bytes_per_ms"],
                                       f"link_rate_invalid:{name}")
            if peak is None and not math.isclose(utilization, 1.0):
                # The declared byte capacity is already the effective value.
                peak = capacity / utilization
        elif "capacity_gbps" in value or "gbps" in value:
            rate = value.get("capacity_gbps", value.get("gbps"))
            declared_peak = _positive_float(rate, f"link_rate_invalid:{name}") * 1e6
            peak = declared_peak if peak is None else peak
            capacity = declared_peak * utilization
        elif peak is not None:
            capacity = peak * utilization
        else:
            raise ModelingError(f"link_rate_invalid:{name}")
        semantics = value.get("semantics", value.get("transport_semantics", "shared"))
        semantics = {"switched": "per_port"}.get(semantics, semantics)
        return cls(
            name=name, capacity_bytes_per_ms=capacity,
            members=tuple(value.get("members", default_members)),
            latency_us=float(value.get("latency_us", 0)),
            channel=value.get("channel", "fabric"), semantics=semantics,
            transaction_bytes=value.get("transaction_bytes"),
            alignment_bytes=value.get("alignment_bytes", 1),
            protocol_overhead_bytes=value.get("protocol_overhead_bytes", 0),
            switch_hops=value.get("switch_hops", 0),
            switch_hop_latency_us=value.get("switch_hop_latency_us", 0),
            switch_capacity_bytes_per_ms=value.get("switch_capacity_bytes_per_ms"),
            bw_utilization=utilization,
            peak_capacity_bytes_per_ms=peak,
        )


_NETWORK_OVERRIDE_CHANNELS = {
    "inter_node": frozenset({"d2d_inter_node", "inter_node"}),
    "intra_node": frozenset({"d2d_intra_node", "intra_node"}),
    "intra_package": frozenset({"d2d_intra_package", "intra_package"}),
}
_NETWORK_OVERRIDE_SUFFIXES = ("bw_gbps", "bw_bytes_per_s", "bw_util", "latency_us")
_PRESERVED_TOPOLOGY_KEYS = (
    "intra_bw_4p_gbps", "intra_bw_8p_gbps", "superpod_size", "max_dies",
    "superpod_bw_gbps", "inter_pod_bw_gbps", "superpod_latency_us",
    "inter_pod_latency_us", "topology_preset",
)


def _network_override_keys(raw: Mapping) -> set[str]:
    return {f"{kind}_{suffix}" for kind in _NETWORK_OVERRIDE_CHANNELS
            for suffix in _NETWORK_OVERRIDE_SUFFIXES if f"{kind}_{suffix}" in raw}


def apply_network_overrides(links, raw: Mapping, *,
                            gbps_to_bytes_per_ms: float = 1e6):
    """Apply machine-level bandwidth, utilization and latency overrides.

    ``gbps_to_bytes_per_ms`` is the unit conversion for the ``*_bw_gbps``
    fields. Ascend's legacy convenience fields declare decimal Gb/s and pass a
    smaller scale from the adapter; neutral link declarations use GB/s.
    """
    keys = _network_override_keys(raw)
    if not keys:
        return tuple(links)
    result = list(links)
    for kind, channels in _NETWORK_OVERRIDE_CHANNELS.items():
        bw_key, bytes_key, util_key, latency_key = (
            f"{kind}_{suffix}" for suffix in _NETWORK_OVERRIDE_SUFFIXES
        )
        if not any(key in raw for key in (bw_key, bytes_key, util_key, latency_key)):
            continue
        if raw.get(bytes_key) is not None:
            bandwidth = _positive_float(raw[bytes_key], f"{bytes_key}_invalid",
                                        allow_zero=False) * 1e-3
        elif raw.get(bw_key) is not None:
            kind_scale = 1e6 if kind == "intra_package" else gbps_to_bytes_per_ms
            bandwidth = _positive_float(raw[bw_key], f"{bw_key}_invalid",
                                        allow_zero=False) * kind_scale
        else:
            bandwidth = None
        utilization = None
        if raw.get(util_key) is not None:
            utilization = _positive_float(raw[util_key], f"{util_key}_invalid",
                                          allow_zero=False)
            if utilization > 1.0:
                raise ModelingError("link_bw_utilization_exceeds_unity")
        latency = None
        if raw.get(latency_key) is not None:
            latency = _positive_float(raw[latency_key], f"{latency_key}_invalid",
                                      allow_zero=True)
        for index, link in enumerate(result):
            if link.channel not in channels:
                continue
            link_util = link.bw_utilization if utilization is None else utilization
            if bandwidth is None:
                peak = link.effective_peak_capacity_bytes_per_ms
            else:
                peak = bandwidth
            capacity = peak * link_util
            link_latency = link.latency_us if latency is None else latency
            result[index] = replace(
                link, capacity_bytes_per_ms=capacity, latency_us=link_latency,
                bw_utilization=link_util,
                peak_capacity_bytes_per_ms=(peak if not math.isclose(link_util, 1.0)
                                            else None),
            )
    return tuple(result)


def _transfer_channel(channel: Mapping, name: str = "transfer") -> Link:
    """Build one link from a transfer-channel declaration.

    Channel ``gbps`` retains the historical GB/s meaning; ``bw_utilization``
    (or its aliases) scales it to the effective capacity while the declared
    peak remains available for diagnostics.
    """
    if not isinstance(channel, Mapping):
        raise ModelingError(f"transfer_channel_record_invalid:{name}")
    allowed = {
        "channel", "channel_name", "gbps", "capacity_bytes_per_ms",
        "capacity_gbps", "members", "latency_us", "channel_semantics",
        "semantics", "transport_semantics", "transaction_bytes",
        "alignment_bytes", "protocol_overhead_bytes", "switch_hops",
        "switch_hop_latency_us", "switch_capacity_bytes_per_ms",
        "bw_utilization", "bw_ratio", "utilization",
        "peak_capacity_bytes_per_ms", "effective_uni_bw_gbps",
        "uni_peak_bw_gbps", "bi_peak_bw_gbps", "calibrated_uni_bw_gbps",
        "calibrated_bi_bw_gbps", "derived_uni_bw_gbps", "derived_bi_bw_gbps",
        "effective_bi_bw_gbps", "pageable_uni_bw_gbps", "alpha_us",
        "protocol", "transport_mode", "notes", "bandwidth_provenance",
    }
    _unknown(channel, allowed, f"transfer_channel.{name}")
    utilization = 1.0
    for alias in ("bw_utilization", "bw_ratio", "utilization"):
        if alias in channel:
            utilization = _positive_float(channel[alias],
                                          "transfer_channel_bw_utilization_invalid",
                                          allow_zero=False)
            if utilization > 1.0:
                raise ModelingError("transfer_channel_bw_utilization_invalid")
            break
    if channel.get("gbps") is not None:
        peak = _positive_float(channel.get("gbps"),
                               "transfer_channel_requires_positive_gbps") * 1e6
    elif channel.get("peak_capacity_bytes_per_ms") is not None:
        peak = _positive_float(channel["peak_capacity_bytes_per_ms"],
                               "transfer_channel_requires_positive_gbps")
    elif channel.get("capacity_bytes_per_ms") is not None:
        peak = (_positive_float(channel["capacity_bytes_per_ms"],
                                "transfer_channel_requires_positive_gbps")
                / utilization)
    elif channel.get("capacity_gbps") is not None:
        peak = (_positive_float(channel["capacity_gbps"],
                                "transfer_channel_requires_positive_gbps")
                * 1e6 / utilization)
    elif channel.get("effective_uni_bw_gbps") is not None:
        peak = (_positive_float(channel["effective_uni_bw_gbps"],
                                "transfer_channel_requires_positive_gbps")
                * 1e6 / utilization)
    elif channel.get("uni_peak_bw_gbps") is not None:
        peak = (_positive_float(channel["uni_peak_bw_gbps"],
                                "transfer_channel_requires_positive_gbps")
                * 1e6)
    else:
        raise ModelingError("transfer_channel_requires_positive_gbps")
    effective = peak * utilization
    channel_name = channel.get("channel", channel.get("channel_name", name))
    semantics = channel.get("semantics", channel.get("channel_semantics",
                                                      channel.get("transport_semantics", "shared")))
    semantics = {"switched": "per_port"}.get(semantics, semantics)
    return Link(
        name=str(channel_name), capacity_bytes_per_ms=effective,
        members=tuple(channel.get("members", ())),
        latency_us=float(channel.get("latency_us", channel.get("alpha_us", 0))),
        channel=str(channel_name), semantics=semantics,
        transaction_bytes=channel.get("transaction_bytes"),
        alignment_bytes=channel.get("alignment_bytes", 1),
        protocol_overhead_bytes=channel.get("protocol_overhead_bytes", 0),
        switch_hops=channel.get("switch_hops", 0),
        switch_hop_latency_us=channel.get("switch_hop_latency_us", 0),
        switch_capacity_bytes_per_ms=channel.get("switch_capacity_bytes_per_ms"),
        bw_utilization=utilization,
        peak_capacity_bytes_per_ms=peak,
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

    @property
    def compute_peaks(self):
        return self.dies[0].compute_peaks

    def peak_flops_for(self, dtype):
        return min(device.peak(dtype) for device in self.dies)

    def declared_peak_key(self, dtype):
        return dtype if dtype in dict(self.compute_peaks) else None

    def validate_placement(self, parallel):
        if parallel.world_size > len(self.dies):
            raise ModelingError("placement_exceeds_physical_devices")
        if len({device.rank for device in self.dies}) != len(self.dies):
            raise ModelingError("device_rank_duplicate")

    @classmethod
    def from_mapping(cls, raw: Mapping):
        from mindie_sim.backends import materialize
        backend = raw.get("backend")
        raw = materialize(raw)
        count = int(raw.get("num_devices", len(raw.get("devices", raw.get("dies", ()))) or 1))
        ranks_per_node = int(raw.get("ranks_per_node", count))
        package = int(raw.get("dies_per_package", 1))
        inherited = machine_device_defaults(raw)
        defaults = merge_device_inputs(inherited, raw.get("device", {}))
        records = raw.get("devices", raw.get("dies", ({},) * count))
        if len(records) != count:
            raise ModelingError("num_devices_differs_from_explicit_device_records")
        devices = []
        for index, record in enumerate(records):
            data = merge_device_inputs(defaults, record)
            arch = ComputeArchSpec.from_mapping(data.get("compute_arch", {}))
            arch_rates, arch_units, arch_work_units, arch_modes = arch.supplies(
                int(data.get("compute_units", 1)), data.get("resource_units", {}),
                data.get("rates", {}))
            memory = int(data.get("memory_bytes", 0))
            peaks = dict(data.get("compute_peaks", {}))
            rates = {**arch_rates, **data.get("rates", {})}
            rates = {key: float(value) for key, value in rates.items()}
            if memory <= 0 or not peaks or "hbm" not in rates or "vector" not in rates:
                raise ModelingError("device_requires_memory_compute_vector_and_memory_rates")
            if any(not math.isfinite(value) or value <= 0 for value in (*peaks.values(), *rates.values())):
                raise ModelingError("device_rates_must_be_positive")
            buffers = dict(data.get("buffers", {}))
            units = {**arch_units, **data.get("resource_units", {})}
            if any(value < 1 for value in units.values()):
                raise ModelingError("resource_units_must_be_positive")
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
            devices.append(Device(
                rank=int(data.get("rank", index)), memory_bytes=memory,
                compute_peaks=tuple(sorted(peaks.items())), rates=tuple(sorted(rates.items())),
                buffers=tuple(sorted(buffers.items())),
                compute_units=int(data.get("compute_units", 1) or 1),
                launch_us=float(data.get("launch_us", 0)),
                node=int(data.get("node", index // ranks_per_node)),
                package=int(data.get("package", index // package)),
                resource_units=tuple(sorted(units.items())),
                buffer_units=tuple(sorted(data.get("buffer_units", {}).items())),
                service_modes=tuple(sorted((name, (mode["resource"], float(mode["rate"])))
                                           for name, mode in modes.items())),
                work_units=tuple(sorted(work_units.items())),
                kernel_invocations=tuple(sorted(
                    (key, InvocationTiming.from_mapping(value))
                    for key, value in data.get("kernel_invocations", {}).items())),
                store_queues=tuple(sorted((name, tuple(sorted(queue.items())))
                                          for name, queue in data.get("store_queues", {}).items())),
                compute_arch=arch,
                linked_rate_overrides=tuple(sorted(s.name for s in arch.instruction_services
                    if (s.scale_with or s.clock_hz is not None) and s.name in data.get("rates", {}))),
                instruction_mode_overrides=tuple(sorted(s.name for s in arch.instruction_services
                    if s.resource and "rate" in data.get("service_modes", {}).get(s.name, {}))),
                instruction_unit_overrides=tuple(sorted(s.name for s in arch.instruction_services
                    if (s.scale_with or s.clock_hz is not None)
                    and s.name in data.get("resource_units", {}))),
            ))
        links = tuple(Link.from_mapping(link, default_members=range(len(devices)))
                      for link in raw.get("links", ()))
        if not backend or _network_override_keys(raw):
            # Compiled backends strip convenience units when generating links;
            # pre-materialized link declarations still honor explicit overrides.
            scale = (1e6 / 8 if backend == "ascend" and not raw.get("topology_preset")
                     else 1e6)
            links = apply_network_overrides(links, raw, gbps_to_bytes_per_ms=scale)
        for link in links:
            if link.semantics not in {"shared", "per_port", "per_pair", "direct", "fully_connected", "mesh"}:
                raise ModelingError(f"link_semantics_unknown:{link.semantics}")
            if not math.isfinite(link.capacity_bytes_per_ms) or link.capacity_bytes_per_ms <= 0:
                raise ModelingError(f"link_rate_invalid:{link.name}")
            if not set(link.members) <= {device.rank for device in devices}:
                raise ModelingError(f"link_member_unknown:{link.name}")
        return cls(name=raw.get("name", raw.get("sku", "declared")), dies=tuple(devices), links=links,
                   arch_family=raw.get("architecture", raw.get("arch_family", "generic")),
                   hardware_key=raw.get("hardware_key", raw.get("name", "")),
                   ranks_per_node=ranks_per_node, dies_per_package=package, sku=raw.get("sku", ""),
                   assumptions=tuple(sorted(raw.get("assumptions", {}).items())),
                   transfer_channels=tuple(sorted(
                       (str(k), tuple(sorted(v.items())))
                       for k, v in (raw.get("transfer_channels") or {}).items()
                       if isinstance(v, Mapping)
                   )),
                   topology_options=tuple(sorted(
                       (str(k), raw[k]) for k in _PRESERVED_TOPOLOGY_KEYS
                       if raw.get(k) is not None
                   )))

    def to_dict(self):
        return {"name": self.name, "architecture": self.arch_family, "hardware_key": self.hardware_key,
                "sku": self.sku,
                "devices": [device.to_dict() for device in self.dies],
                "links": [link.to_dict() for link in self.links], "ranks_per_node": self.ranks_per_node,
                "dies_per_package": self.dies_per_package, "assumptions": dict(self.assumptions)}


def _neutral_transfer_channel_links(raw: Mapping) -> list[dict]:
    declaration = raw.get("transfer_channels")
    if not isinstance(declaration, Mapping):
        return []
    links = []
    for name, channel in declaration.items():
        if not isinstance(channel, Mapping):
            continue
        try:
            link = _transfer_channel(channel, str(name))
        except ModelingError:
            # Registered channels without a declared rate stay metadata.
            continue
        record = link.to_dict()
        record["name"] = str(name)
        if not link.members:
            record.pop("members", None)
        links.append(record)
    return links


def parse_hardware(raw: Mapping) -> Hardware:
    """Parse a hardware declaration, including top-level network overrides.

    Neutral declarations may list ``transfer_channels`` instead of explicit
    ``links``. Compiling backends own their channel registries; this helper
    materializes only channels that already declare a usable rate so a standalone
    hardware document remains inspectable without importing a vendor adapter.
    """
    if not isinstance(raw, Mapping):
        raise ModelingError("hardware_mapping_required")
    if raw.get("backend") or raw.get("links"):
        return Hardware.from_mapping(raw)
    links = _neutral_transfer_channel_links(raw)
    if not links:
        return Hardware.from_mapping(raw)
    return Hardware.from_mapping({**raw, "links": links})
