"""Device-independent resource descriptions.

[Role] Describe compute rates, memory, buffers, submission and physical links.
[Invariants] Vendor names are labels. The declared resource graph, rates and
  storage capacities determine execution. Precision peaks are never inferred
  from a vendor or a model name. A declared peak is a native-compute supply:
  an architecture capability must name the native operand/accumulator mode
  before that peak is accepted, so calibration cannot enable an unsupported
  native format. Representation storage/transfer, cast, native compute and
  runtime implementation remain separate declarations. A link's effective
  capacity is always ``peak_capacity * bw_utilization``; serialized links omit
  the utilization keys when utilization is unity so calibration identities stay
  stable.
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
from .physical_mechanisms import (PhysicalMechanismSpec, default_physical_mechanisms,
                                  parse_physical_mechanisms)

ZERO = 0
ONE = 1
from .compute_peak_aliases import (
    native_compute_peak_items, resolve_compute_peak_name,
    validate_compute_peak_aliases,
)


NATIVE_WORK_UNITS = {
    **dict.fromkeys(("hbm", "l2", "mte1", "mte2", "mte2_aic", "mte2_aiv", "mte3",
                     "fixpipe", "host_memory", "link"), "bytes"),
    "cube": "flops", "vector": "ops", "host_vector": "ops", "host": "ms",
    "scalar_store_issue": "stores",
}

#: Architecture families whose registered profiles carry a concrete
#: representation/native-compute capability. A bare peak dictionary is not
#: accepted for these families because it could silently enable a format.
_CAPABILITY_REQUIRED_FAMILIES = frozenset({"A2", "A3", "A5"})


_QUANTITY_SCOPES = {
    "memory": "per_device",
    "buffers": "per_owner",
    "compute_peaks": "device_total",
    "rates": "device_total",
    "resource_units": "owners_per_device",
    "buffer_units": "owners_per_device",
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
    sku: str = ""
    arch_family: str = ""
    #: Declared reporting/supply aliases (``mxfp4 -> fp8_e4m3``).  These are
    #: never native operand modes and must carry the target's exact peak.
    compute_peak_aliases: tuple[tuple[str, str], ...] = ()
    physical_mechanisms: PhysicalMechanismSpec = field(
        default_factory=default_physical_mechanisms)

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
                self.physical_mechanisms.signature)

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
            # A rate is not a capability. Registered architectures must name a
            # native operand/accumulator mode before its peak can price work.
            # An alias resolves to its native target; it never becomes one.
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
        return {"rank": self.rank, "memory_bytes": self.memory_bytes,
                "compute_peaks": dict(self.compute_peaks),
                # Linked supplies remain declarations on round trip. Emitting
                # their derived values as explicit overrides freezes a sweep.
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
    endpoint_rates: tuple[tuple[int, float], ...] = ()

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
        seen = set()
        for rank, rate in self.endpoint_rates:
            if (isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
                    or rank in seen):
                raise ModelingError(f"link_endpoint_rank_invalid:{self.name}")
            if (isinstance(rate, bool) or not isinstance(rate, (int, float))
                    or not math.isfinite(float(rate)) or float(rate) <= 0):
                raise ModelingError(f"link_endpoint_rate_invalid:{self.name}")
            seen.add(rank)
        if seen and not seen <= set(self.members):
            raise ModelingError(f"link_endpoint_member_unknown:{self.name}")

    @property
    def propagation_us(self):
        return self.latency_us + self.switch_hops * self.switch_hop_latency_us

    @property
    def effective_peak_capacity_bytes_per_ms(self) -> float:
        if self.peak_capacity_bytes_per_ms is not None:
            return float(self.peak_capacity_bytes_per_ms)
        return float(self.capacity_bytes_per_ms)

    def endpoint_capacity_bytes_per_ms(self, rank: int) -> float | None:
        """Declared receive/send service of one placement's endpoint port."""
        return dict(self.endpoint_rates).get(rank)

    def capacity_at(self, rank: int) -> float:
        """Bottleneck service when one placement uses this link.

        A shared link keeps its shared capacity; a declared endpoint port can be
        slower for this placement. Other members are not silently assigned the
        source placement's port rate.
        """
        endpoint = self.endpoint_capacity_bytes_per_ms(rank)
        if endpoint is None:
            return float(self.capacity_bytes_per_ms)
        return min(float(self.capacity_bytes_per_ms), endpoint)

    def to_dict(self):
        result = {"name": self.name, "capacity_bytes_per_ms": self.capacity_bytes_per_ms,
                  "members": self.members, "latency_us": self.latency_us,
                  "channel": self.channel, "semantics": self.semantics,
                  **({"switch_hops": self.switch_hops,
                      "switch_hop_latency_us": self.switch_hop_latency_us}
                     if self.switch_hops or self.switch_hop_latency_us else {}),
                  **({"switch_capacity_bytes_per_ms": self.switch_capacity_bytes_per_ms}
                     if self.switch_capacity_bytes_per_ms is not None else {}),
                  **({"endpoint_rates": {str(rank): rate
                                         for rank, rate in self.endpoint_rates}}
                     if self.endpoint_rates else {}),
                  **({"transaction_bytes": self.transaction_bytes,
                      "alignment_bytes": self.alignment_bytes,
                      "protocol_overhead_bytes": self.protocol_overhead_bytes}
                     if self.transaction_bytes is not None else {})}
        # Utilization and peak are only serialized when they add information.
        # Existing calibration identities hash this dictionary, so unity must
        # keep the historical key set byte-for-byte.
        if not math.isclose(self.bw_utilization, 1.0):
            result["bw_utilization"] = self.bw_utilization
        if self.peak_capacity_bytes_per_ms is not None:
            # Keep an explicitly declared peak even at unity utilization; it is
            # local declaration data, not a derived duplicate.
            result["peak_capacity_bytes_per_ms"] = self.peak_capacity_bytes_per_ms
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
            "pageable_uni_bw_gbps", "endpoint_rates",
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
        endpoint_raw = value.get("endpoint_rates", ())
        if isinstance(endpoint_raw, Mapping):
            endpoint_items = endpoint_raw.items()
        elif isinstance(endpoint_raw, (list, tuple)):
            endpoint_items = endpoint_raw
        else:
            raise ModelingError(f"link_endpoint_rates_invalid:{name}")
        endpoint_rates = []
        for item in endpoint_items:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                rank, rate = item
            elif isinstance(endpoint_raw, Mapping):
                rank, rate = item
            else:
                raise ModelingError(f"link_endpoint_rates_invalid:{name}")
            endpoint_rates.append((int(rank), float(rate)))
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
            endpoint_rates=tuple(sorted(endpoint_rates)),
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


COLLECTIVE_PROTOCOL_KEYS = frozenset({"alltoall", "alltoallv"})
COLLECTIVE_PROTOCOL_VALUES = frozenset({"ring", "log2", "concurrent"})


def _collective_protocols(value) -> tuple[tuple[str, str], ...]:
    """Validate a hardware-declared AllToAll startup protocol map."""
    if value is None or value == ():
        return ()
    if not isinstance(value, Mapping):
        raise ModelingError("collective_protocols_mapping_required")
    result = {}
    for key, item in value.items():
        name = str(key).strip().lower()
        mode = str(item).strip().lower() if isinstance(item, str) else ""
        if name not in COLLECTIVE_PROTOCOL_KEYS:
            raise ModelingError(f"collective_protocol_unknown:{name}")
        if mode not in COLLECTIVE_PROTOCOL_VALUES:
            raise ModelingError(f"collective_protocol_unsupported:{name}:{item}")
        result[name] = mode
    return tuple(sorted(result.items()))


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
    #: Declared AllToAll/AllToAllV startup protocols.  Empty preserves the
    #: legacy per-profile behavior; no SKU or latency value implies a protocol.
    collective_protocols: tuple[tuple[str, str], ...] = ()

    @property
    def homogeneous(self) -> bool:
        """True only when every placement declares the same physical resources."""
        return len({device.resource_signature for device in self.dies}) <= 1

    def _require_homogeneous(self, query: str) -> None:
        if not self.homogeneous:
            raise ModelingError(f"hardware_resources_are_per_placement:{query}")

    @property
    def compute_peaks(self):
        """Legacy common projection; each placement still owns its own peaks."""
        return self.dies[0].compute_peaks

    def per_placement_peaks(self, dtype):
        """Peak per rank, never one SKU's value broadcast to another rank."""
        return {device.rank: device.peak(dtype) for device in self.dies}

    def peak_flops_for(self, dtype):
        self._require_homogeneous("peak_flops_for")
        return min(device.peak(dtype) for device in self.dies)

    def declared_peak_key(self, dtype):
        return dtype if dtype in dict(self.compute_peaks) else None

    def package_memory_bytes(self, package: int) -> int:
        """Capacity reachable from one package's declared placements."""
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

    def hardware_context(self, rank: int = ZERO) -> dict:
        """Minimal neutral hardware context for independent operator/native use.

        It carries only the actual placement's facts and the derived
        homogeneity flag; it contains no model, workload, request, duration or
        device measurement. A consumer must not replace the per-rank values
        with a machine-wide first-die projection.
        """
        device = next((item for item in self.dies if item.rank == rank), None)
        if device is None:
            raise ModelingError(f"hardware_placement_unknown:{rank}")
        return {
            "schema": "mindie-sim.hardware-context/1",
            "name": self.name, "sku": self.sku,
            "arch_family": self.arch_family, "homogeneous": self.homogeneous,
            "placement": {"rank": device.rank, "node": device.node,
                          "package": device.package},
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
            for field in ("channel_semantics", "semantics"):
                if field in channel and channel[field] not in allowed_semantics:
                    raise ModelingError(
                        f"transfer_channel_semantics_unsupported:{name}:{channel[field]}")
        count = int(raw.get("num_devices", len(raw.get("devices", raw.get("dies", ()))) or 1))
        ranks_per_node = int(raw.get("ranks_per_node", count))
        package = int(raw.get("dies_per_package", 1))
        records = list(raw.get("devices", raw.get("dies", ({},) * count)))
        if len(records) != count:
            raise ModelingError("num_devices_differs_from_explicit_device_records")
        if raw_records and len(raw_records) > count:
            raise ModelingError("num_devices_differs_from_explicit_device_records")
        # One normalized explicit declaration per placement, in the existing
        # precedence order: machine public fields -> machine device -> record
        # fields -> record device. It is formed before any SKU facts are
        # inherited and is consumed by both the owning adapter and the supply
        # scaling below, so a nested explicit value keeps one identity.
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
                # The per-layer normalization above already produced the one
                # canonical identity. Do not re-run a cross-layer or-chain.
                arch_family_local = str(data.get("arch_family") or "generic")
            for key in ("rank", "node", "package"):
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
                scale_aic = new_aic / rec_base_aic
                explicit_units = _explicit_unit_keys(explicit_decl)
                cur_units = dict(data.get("resource_units", {}))
                new_aiv = int(cur_units.get("vector", rec_base_aiv) or rec_base_aiv)
                scale_aiv = (new_aiv / rec_base_aiv) if rec_base_aiv > 0 else 1.0
                if scale_aic != 1.0 or scale_aiv != 1.0:
                    explicit_peaks = _explicit_peak_keys(explicit_decl)
                    data["compute_peaks"] = {
                        k: (float(v) if k in explicit_peaks else float(v) * scale_aic)
                        for k, v in data.get("compute_peaks", {}).items()
                    }
                    aic_u = {"cube", "mte1", "mte2_aic", "fixpipe"}
                    data["resource_units"] = {
                        k: (int(v) if k in explicit_units else max(1, int(round(int(v) * (scale_aic if k in aic_u else scale_aiv)))))
                        for k, v in cur_units.items()
                    }
                    explicit_buf_u, explicit_modes = _explicit_buf_and_mode_keys(explicit_decl)
                    aic_b = {"l1", "l0a", "l0b", "l0c"}
                    data["buffer_units"] = {
                        k: (int(v) if k in explicit_buf_u else max(1, int(round(int(v) * (scale_aic if k in aic_b else (scale_aiv if k == "ub" else 1.0))))))
                        for k, v in data.get("buffer_units", {}).items()
                    }
                    explicit_rates = _explicit_rate_keys(explicit_decl)
                    aic_r = {"mte1", "mte2_aic", "fixpipe"}
                    aiv_r = {"vector", "mte2_aiv", "mte3", "reg_data_writes", "reg_ub_accesses", "reg_ub_stores",
                             "reg_alu_lanes", "reg_special_lanes", "reg_reduce_lanes", "reg_shuffle_lanes", "scalar_store_issue"}
                    scaled_r = {
                        k: (float(v) * scale_aic if k in aic_r and k not in explicit_rates
                            else (float(v) * scale_aiv if k in aiv_r and k not in explicit_rates else float(v)))
                        for k, v in data.get("rates", {}).items()
                    }
                    if "mte2" in scaled_r and "mte2" not in explicit_rates:
                        scaled_r["mte2"] = (min(scaled_r["mte2_aic"], scaled_r["mte2_aiv"])
                                            if "mte2_aic" in scaled_r and "mte2_aiv" in scaled_r
                                            else scaled_r["mte2"] * min(scale_aic, scale_aiv))
                    data["rates"] = scaled_r
                    data["service_modes"] = {
                        k: ({**m, "rate": float(m["rate"]) * (scale_aic if m.get("resource") in aic_r else scale_aiv)}
                            if k not in explicit_modes and "rate" in m and m.get("resource") in (aic_r | aiv_r) and m.get("resource") not in explicit_rates
                            else dict(m))
                        for k, m in data.get("service_modes", {}).items()
                    }
            arch_family = str(data.get("arch_family")
                              or raw.get("architecture") or "generic")
            arch = ComputeArchSpec.from_mapping(
                data.get("compute_arch", {}),
                source=f"compute_arch:{data.get('sku', arch_family)}")
            peaks = dict(data.get("compute_peaks", {}))
            if (arch_family in _CAPABILITY_REQUIRED_FAMILIES
                    and not arch.has_native_modes()):
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
                # Preserve a legacy serializer's real instruction services and
                # explicit overrides; add only the missing architecture
                # capability. Never replace the whole declaration with the
                # preset, which would drop linked rates/modes.
                merged = {**existing, **resolved}
                merged["preset"] = resolved["preset"]
                for field in ("instruction_services", "representations",
                              "native_compute", "casts", "paths", "runtimes"):
                    if field in existing and field in resolved:
                        merged[field] = {**resolved[field], **existing[field]}
                arch = ComputeArchSpec.from_mapping(
                    merged, source=f"registered_capability:{arch_family}")
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
                # Calibration and explicit overrides go through this same parse:
                # a rate for an undeclared native mode is rejected, not priced.
                arch.validate_peaks(peaks, aliases=aliases)
            arch_rates, arch_units, arch_work_units, arch_modes = arch.supplies(
                int(data.get("compute_units", 1)), data.get("resource_units", {}),
                data.get("rates", {}))
            memory = int(data.get("memory_bytes", 0))
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
            buffer_units = dict(data.get("buffer_units", {}))
            if "l2" in buffers:
                declared_l2_units = buffer_units.get("l2")
                if (declared_l2_units is not None
                        and int(declared_l2_units) != ONE):
                    raise ModelingError("shared_l2_buffer_requires_one_die_owner")
                buffer_units["l2"] = ONE
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
                fallback_l2_bytes=int(buffers.get("l2", ZERO)))
            devices.append(Device(
                rank=int(data.get("rank", index)), memory_bytes=memory,
                compute_peaks=tuple(sorted(peaks.items())), rates=tuple(sorted(rates.items())),
                buffers=tuple(sorted(buffers.items())),
                compute_units=int(data.get("compute_units", ONE) or ONE),
                launch_us=float(data.get("launch_us", ZERO)),
                node=int(data.get("node", index // ranks_per_node)),
                package=int(data.get("package", index // package)),
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
                linked_rate_overrides=tuple(sorted(s.name for s in arch.instruction_services
                    if (s.scale_with or s.clock_hz is not None) and s.name in data.get("rates", {}))),
                instruction_mode_overrides=tuple(sorted(s.name for s in arch.instruction_services
                    if s.resource and "rate" in data.get("service_modes", {}).get(s.name, {}))),
                instruction_unit_overrides=tuple(sorted(s.name for s in arch.instruction_services
                    if (s.scale_with or s.clock_hz is not None)
                    and s.name in data.get("resource_units", {}))),
                sku=str(data.get("sku", "")),
                arch_family=arch_family,
                physical_mechanisms=mechanisms,
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
        device_families = {device.arch_family for device in devices}
        machine_family = (device_families.pop() if len(device_families) == 1
                          else "mixed")
        device_skus = {device.sku for device in devices if device.sku}
        machine_sku = (device_skus.pop() if len(device_skus) == 1
                       else "mixed" if device_skus else raw.get("sku", ""))
        return cls(name=raw.get("name", raw.get("sku", "declared")), dies=tuple(devices), links=links,
                   arch_family=machine_family,
                   hardware_key=raw.get("hardware_key", raw.get("name", "")),
                   ranks_per_node=ranks_per_node, dies_per_package=package, sku=machine_sku,
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
                   collective_protocols=_collective_protocols(
                       raw.get("collective_protocols", ())))

    def to_dict(self):
        return {"name": self.name, "architecture": self.arch_family, "hardware_key": self.hardware_key,
                "sku": self.sku,
                "devices": [device.to_dict() for device in self.dies],
                "links": [link.to_dict() for link in self.links], "ranks_per_node": self.ranks_per_node,
                "dies_per_package": self.dies_per_package, "assumptions": dict(self.assumptions),
                **({"transfer_channels": {
                    name: dict(items) for name, items in self.transfer_channels}}
                   if self.transfer_channels else {}),
                **({"collective_protocols": dict(self.collective_protocols)}
                   if self.collective_protocols else {})}


def _alias_cores(mapping: Mapping) -> dict:
    out = dict(mapping)
    if "compute_units" not in out and "aic_count" not in out:
        for key in ("num_cores", "cores"):
            if out.get(key) is not None:
                out["compute_units"] = int(out[key])
                break
    return out


def _alias_record(record: Mapping) -> dict:
    out = _alias_cores(record)
    if isinstance(out.get("device"), Mapping):
        out["device"] = _alias_cores(out["device"])
    return out


def _normalize_core_aliases(raw: Mapping) -> dict:
    out = _alias_cores(raw)
    if isinstance(out.get("device"), Mapping):
        out["device"] = _alias_cores(out["device"])
    for key in ("devices", "dies"):
        if isinstance(out.get(key), (list, tuple)):
            out[key] = [_alias_record(d) if isinstance(d, Mapping) else d
                        for d in out[key]]
    return out


def _has_core_override(declaration: Mapping) -> bool:
    """True when the normalized placement declaration sets owner geometry."""
    return bool(declaration.get("compute_units")
                or declaration.get("resource_units"))


def _explicit_peak_keys(declaration: Mapping) -> set[str]:
    """Peak keys explicitly declared by this placement, not inherited."""
    return {str(key) for key in declaration.get("compute_peaks", {})}


def _explicit_buf_and_mode_keys(
        declaration: Mapping) -> tuple[set[str], set[str]]:
    buffer_units = {str(key) for key in declaration.get("buffer_units", {})}
    modes = {str(name) for name, mode in declaration.get("service_modes", {}).items()
             if isinstance(mode, Mapping) and "rate" in mode}
    return buffer_units, modes


def _explicit_unit_keys(declaration: Mapping) -> set[str]:
    return {str(key) for key in declaration.get("resource_units", {})}


def _explicit_rate_keys(declaration: Mapping) -> set[str]:
    return {str(key) for key in declaration.get("rates", {})}


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
