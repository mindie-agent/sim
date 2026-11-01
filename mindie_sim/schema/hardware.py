"""Device-independent resource descriptions.

[Role] Describe compute rates, memory, buffers, submission and physical links.
[Invariants] Vendor names are labels. The declared resource graph, rates and
  storage capacities determine execution. Precision peaks are never inferred
  from a vendor or a model name.
[Timeouts & Locks] Pure metadata parsing, no driver calls or device contexts.
[Side-Effect Policy] Immutable input records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

from .common import ModelingError
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

    @property
    def propagation_us(self):
        return self.latency_us + self.switch_hops * self.switch_hop_latency_us

    def to_dict(self):
        return {"name": self.name, "capacity_bytes_per_ms": self.capacity_bytes_per_ms,
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
        links = tuple(Link(
            name=link["name"],
            capacity_bytes_per_ms=float(link.get("capacity_bytes_per_ms",
                                                 link.get("capacity_gbps", 0) * 1e6)),
            members=tuple(link.get("members", range(len(devices)))),
            latency_us=float(link.get("latency_us", 0)), channel=link.get("channel", "fabric"),
            semantics={"switched": "per_port"}.get(
                link.get("semantics", link.get("transport_semantics", "shared")),
                link.get("semantics", link.get("transport_semantics", "shared"))),
            transaction_bytes=link.get("transaction_bytes"),
            alignment_bytes=link.get("alignment_bytes", 1),
            protocol_overhead_bytes=link.get("protocol_overhead_bytes", 0),
            switch_hops=link.get("switch_hops", 0),
            switch_hop_latency_us=link.get("switch_hop_latency_us", 0),
            switch_capacity_bytes_per_ms=link.get("switch_capacity_bytes_per_ms"),
        ) for link in raw.get("links", ()))
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
                   )))

    def to_dict(self):
        return {"name": self.name, "architecture": self.arch_family, "hardware_key": self.hardware_key,
                "sku": self.sku,
                "devices": [device.to_dict() for device in self.dies],
                "links": [link.to_dict() for link in self.links], "ranks_per_node": self.ranks_per_node,
                "dies_per_package": self.dies_per_package, "assumptions": dict(self.assumptions)}
