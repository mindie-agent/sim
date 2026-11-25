"""Link, protocol-band and transfer-channel schema helpers for ``Hardware``.

[Role] Parse, validate and serialize physical interconnect ``Link`` records,
  piecewise ``CollectiveBandSpec`` tables, machine bandwidth/latency overrides,
  transfer-channel projections, and collective startup protocol maps.
[Invariants] A link's effective capacity is always ``peak_capacity * bw_utilization``.
  Serialized links omit optional utilization/rail/band/scope keys at default values
  so historical calibration identities remain byte-for-byte stable. Explicit link
  scopes are validated against ``VALID_LINK_SCOPES``; piecewise protocol bands
  under a fixed selector are mutually exclusive over ``[payload_min_inclusive,
  payload_max_exclusive)``.
[Timeouts & Locks] Pure in-memory metadata validation; no I/O, driver calls, or locks.
[Side-Effect Policy] Returns immutable ``Link`` and ``CollectiveBandSpec`` records.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

from .common import ModelingError, _positive_float, _unknown

GBPS_TO_BYTES_PER_MS = 1_000_000.0
BYTES_PER_S_TO_BYTES_PER_MS = 1.0 / 1000.0

VALID_LINK_SCOPES = frozenset({"package", "node", "cross_node", "host"})
VALID_BAND_STARTUP_SCOPES = frozenset({"collective", "phase", "hop"})

_NETWORK_OVERRIDE_CHANNELS = {
    "inter_node": frozenset({"d2d_inter_node", "inter_node"}),
    "intra_node": frozenset({"d2d_intra_node", "intra_node"}),
    "intra_package": frozenset({"d2d_intra_package", "intra_package"}),
}
_NETWORK_OVERRIDE_SUFFIXES = ("bw_gbps", "bw_bytes_per_s", "bw_util", "latency_us")
_PRESERVED_TOPOLOGY_KEYS = (
    "intra_bw_4p_gbps", "intra_bw_8p_gbps", "superpod_size", "max_dies",
    "superpod_bw_gbps", "inter_pod_bw_gbps", "superpod_latency_us",
    "inter_pod_latency_us", "topology_preset", "physical_ranks_per_node",
)
_BAND_WILDCARDS = frozenset({"*", "any", "all"})
_BAND_STACK_WILDCARDS = frozenset({"*", "default"})

COLLECTIVE_PROTOCOL_KEYS = frozenset({"alltoall", "alltoallv"})
COLLECTIVE_PROTOCOL_VALUES = frozenset({"ring", "log2", "concurrent"})
UNCALIBRATED_SKU_WARNING = "uncalibrated_physics_scaled_sku"


def _band_pos_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"collective_band_invalid:{name}:{value}")
    return int(value)


def _band_nonneg_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ModelingError(f"collective_band_invalid:{name}:{value}")
    return int(value)


def _band_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(f"collective_band_invalid:{name}:{value}")
    return value.strip()


@dataclass(frozen=True, slots=True)
class CollectiveBandSpec:
    """Piecewise HCCL collective protocol/framing/startup band specification."""

    collective: str
    stack_id: str
    topology_scope: str
    group_min: int
    group_max: int
    expansion_mode: str
    payload_min_inclusive: int
    payload_max_exclusive: int | None
    algorithm: str
    protocol: str
    startup_scope: str
    startup_us: float
    payload_bytes_per_frame: int
    wire_bytes_per_frame: int
    credit_bytes: int
    evidence_id: str

    def __post_init__(self) -> None:
        raw_coll = _band_str(self.collective, "collective").lower()
        object.__setattr__(self, "collective", "all" if raw_coll in _BAND_WILDCARDS else raw_coll)
        raw_stack = _band_str(self.stack_id, "stack_id")
        object.__setattr__(self, "stack_id", "default" if raw_stack.lower() in _BAND_STACK_WILDCARDS else raw_stack)
        scope = _band_str(self.topology_scope, "topology_scope").lower()
        if scope in _BAND_WILDCARDS:
            scope = "any"
        if scope not in VALID_LINK_SCOPES and scope != "any":
            raise ModelingError(f"collective_band_topology_scope_unknown:{scope}")
        object.__setattr__(self, "topology_scope", scope)
        gmin = _band_pos_int(self.group_min, "group_min")
        gmax = _band_pos_int(self.group_max, "group_max")
        if gmax < gmin:
            raise ModelingError(f"collective_band_group_range_invalid:{gmin}:{gmax}")
        object.__setattr__(self, "group_min", gmin)
        object.__setattr__(self, "group_max", gmax)
        raw_exp = _band_str(self.expansion_mode, "expansion_mode").lower()
        object.__setattr__(self, "expansion_mode", "*" if raw_exp in _BAND_WILDCARDS else raw_exp)
        pmin = _band_nonneg_int(self.payload_min_inclusive, "payload_min_inclusive")
        object.__setattr__(self, "payload_min_inclusive", pmin)
        if self.payload_max_exclusive is not None:
            pmax = _band_pos_int(self.payload_max_exclusive, "payload_max_exclusive")
            if pmax <= pmin:
                raise ModelingError(f"collective_band_payload_range_invalid:{pmin}:{pmax}")
            object.__setattr__(self, "payload_max_exclusive", pmax)
        object.__setattr__(self, "algorithm", _band_str(self.algorithm, "algorithm").lower())
        object.__setattr__(self, "protocol", _band_str(self.protocol, "protocol"))
        sscope = _band_str(self.startup_scope, "startup_scope").lower()
        if sscope not in VALID_BAND_STARTUP_SCOPES:
            raise ModelingError(f"collective_band_startup_scope_unknown:{sscope}")
        object.__setattr__(self, "startup_scope", sscope)
        startup = _positive_float(self.startup_us, "collective_band_startup_invalid", allow_zero=True)
        object.__setattr__(self, "startup_us", startup)
        pframe = _band_pos_int(self.payload_bytes_per_frame, "payload_bytes_per_frame")
        wframe = _band_pos_int(self.wire_bytes_per_frame, "wire_bytes_per_frame")
        if wframe < pframe:
            raise ModelingError(f"collective_band_wire_frame_smaller_than_payload:{wframe}<{pframe}")
        object.__setattr__(self, "payload_bytes_per_frame", pframe)
        object.__setattr__(self, "wire_bytes_per_frame", wframe)
        object.__setattr__(self, "credit_bytes", _band_nonneg_int(self.credit_bytes, "credit_bytes"))
        object.__setattr__(self, "evidence_id", _band_str(self.evidence_id, "evidence_id"))

    @property
    def selector_key(self) -> tuple[str, str, str, int, int, str]:
        return (
            self.collective, self.stack_id, self.topology_scope,
            self.group_min, self.group_max, self.expansion_mode,
        )

    def matches_payload(self, api_bytes_per_rank: int) -> bool:
        if api_bytes_per_rank < self.payload_min_inclusive:
            return False
        if self.payload_max_exclusive is not None and api_bytes_per_rank >= self.payload_max_exclusive:
            return False
        return True

    def wire_bytes_for(self, payload_bytes: float) -> tuple[float, int]:
        """Frame ``payload_bytes`` into exact protocol frames and wire bytes."""
        if payload_bytes <= 0.0:
            return 0.0, 0
        frames = max(1, math.ceil(float(payload_bytes) / float(self.payload_bytes_per_frame)))
        return float(frames * self.wire_bytes_per_frame), frames

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CollectiveBandSpec:
        if not isinstance(raw, Mapping):
            raise ModelingError("collective_band_mapping_required")
        return cls(
            collective=raw.get("collective", "all"),
            stack_id=raw.get("stack_id", "default"),
            topology_scope=raw.get("topology_scope", "any"),
            group_min=raw.get("group_min", 2),
            group_max=raw.get("group_max", 1024),
            expansion_mode=raw.get("expansion_mode", "*"),
            payload_min_inclusive=raw.get("payload_min_inclusive", 0),
            payload_max_exclusive=raw.get("payload_max_exclusive"),
            algorithm=raw.get("algorithm", "ring"),
            protocol=raw.get("protocol", "standard"),
            startup_scope=raw.get("startup_scope", "collective"),
            startup_us=raw.get("startup_us", 0.0),
            payload_bytes_per_frame=raw.get("payload_bytes_per_frame", 1024),
            wire_bytes_per_frame=raw.get("wire_bytes_per_frame", 1024),
            credit_bytes=raw.get("credit_bytes", 0),
            evidence_id=raw.get("evidence_id", "undeclared"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "collective": self.collective,
            "stack_id": self.stack_id,
            "topology_scope": self.topology_scope,
            "group_min": self.group_min,
            "group_max": self.group_max,
            "expansion_mode": self.expansion_mode,
            "payload_min_inclusive": self.payload_min_inclusive,
            "payload_max_exclusive": self.payload_max_exclusive,
            "algorithm": self.algorithm,
            "protocol": self.protocol,
            "startup_scope": self.startup_scope,
            "startup_us": self.startup_us,
            "payload_bytes_per_frame": self.payload_bytes_per_frame,
            "wire_bytes_per_frame": self.wire_bytes_per_frame,
            "credit_bytes": self.credit_bytes,
            "evidence_id": self.evidence_id,
        }


def validate_protocol_bands(bands: Sequence[CollectiveBandSpec]) -> tuple[CollectiveBandSpec, ...]:
    """Validate that piecewise payload intervals under each selector are disjoint."""
    grouped: dict[tuple[str, str, str, int, int, str], list[CollectiveBandSpec]] = {}
    parsed = tuple(
        item if isinstance(item, CollectiveBandSpec) else CollectiveBandSpec.from_mapping(item)
        for item in bands
    )
    for band in parsed:
        grouped.setdefault(band.selector_key, []).append(band)
    for key, items in grouped.items():
        ordered = sorted(items, key=lambda b: b.payload_min_inclusive)
        for left, right in zip(ordered, ordered[1:]):
            if left.payload_max_exclusive is None or right.payload_min_inclusive < left.payload_max_exclusive:
                raise ModelingError(
                    f"collective_band_payload_intervals_overlap:{key[0]}:"
                    f"{left.payload_min_inclusive}:{left.payload_max_exclusive}:"
                    f"{right.payload_min_inclusive}"
                )
    return parsed


def normalize_link_scope(scope: Any, name: str = "") -> str:
    """Validate and normalize a declared link/channel scope."""
    normalized = str(scope).strip().lower()
    if normalized not in VALID_LINK_SCOPES:
        prefix = f"{name}:" if name else ""
        raise ModelingError(f"link_scope_unknown:{prefix}{scope}")
    return normalized


def parse_collective_bands(raw: Sequence[Any] | None) -> tuple[CollectiveBandSpec, ...]:
    """Parse and validate a sequence of ``CollectiveBandSpec`` records."""
    if not raw:
        return ()
    return validate_protocol_bands(raw)


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
    scope: str | None = None
    rail_id: int | str = 0
    switch_id: str = ""
    root_complex_id: str = ""
    endpoint_ids: tuple[tuple[int, str], ...] = ()
    protocol_bands: tuple[CollectiveBandSpec, ...] = ()
    protocol: str = ""

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
        if self.switch_capacity_bytes_per_ms is not None:
            _positive_float(self.switch_capacity_bytes_per_ms, f"link_switch_capacity_invalid:{self.name}")
        if _positive_float(self.bw_utilization, f"link_bw_utilization_invalid:{self.name}") > 1.0:
            raise ModelingError("link_bw_utilization_exceeds_unity")
        _positive_float(self.capacity_bytes_per_ms, f"link_rate_invalid:{self.name}")
        if self.peak_capacity_bytes_per_ms is not None:
            peak = _positive_float(self.peak_capacity_bytes_per_ms, f"link_peak_capacity_invalid:{self.name}")
            if self.capacity_bytes_per_ms > peak:
                raise ModelingError(f"link_effective_capacity_exceeds_peak:{self.name}")
        if self.scope is not None:
            object.__setattr__(self, "scope", normalize_link_scope(self.scope, self.name))
        if isinstance(self.rail_id, bool):
            raise ModelingError(f"link_rail_id_invalid:{self.name}:{self.rail_id}")
        if isinstance(self.rail_id, int):
            if self.rail_id < 0:
                raise ModelingError(f"link_rail_id_invalid:{self.name}:{self.rail_id}")
        elif isinstance(self.rail_id, str):
            stripped = self.rail_id.strip()
            if not stripped:
                object.__setattr__(self, "rail_id", 0)
            else:
                object.__setattr__(self, "rail_id", stripped)
        else:
            raise ModelingError(f"link_rail_id_invalid:{self.name}:{self.rail_id}")
        seen = set()
        for rank, rate in self.endpoint_rates:
            if (isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
                    or rank in seen):
                raise ModelingError(f"link_endpoint_rank_invalid:{self.name}")
            _positive_float(rate, f"link_endpoint_rate_invalid:{self.name}")
            seen.add(rank)
        if seen and not seen <= set(self.members):
            raise ModelingError(f"link_endpoint_member_unknown:{self.name}")
        if self.protocol_bands:
            object.__setattr__(self, "protocol_bands", validate_protocol_bands(self.protocol_bands))

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

    def physical_endpoint_id(self, rank: int) -> str:
        """Declared physical endpoint ID for ``rank``, defaulting to rank port."""
        return dict(self.endpoint_ids).get(int(rank), f"rank{int(rank)}")

    def capacity_at(self, rank: int) -> float:
        """Bottleneck service when one placement uses this link."""
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
        if not math.isclose(self.bw_utilization, 1.0):
            result["bw_utilization"] = self.bw_utilization
        if self.peak_capacity_bytes_per_ms is not None:
            result["peak_capacity_bytes_per_ms"] = self.peak_capacity_bytes_per_ms
        if self.scope is not None:
            result["scope"] = self.scope
        if self.rail_id:
            result["rail_id"] = self.rail_id
        if self.switch_id:
            result["switch_id"] = self.switch_id
        if self.root_complex_id:
            result["root_complex_id"] = self.root_complex_id
        if self.endpoint_ids:
            result["endpoint_ids"] = {str(rank): ep for rank, ep in self.endpoint_ids}
        if self.protocol:
            result["protocol"] = self.protocol
        if self.protocol_bands:
            result["protocol_bands"] = [band.to_dict() for band in self.protocol_bands]
        return result

    @classmethod
    def from_mapping(cls, value: Mapping, *, default_members=()) -> Link:
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
            "pageable_uni_bw_gbps", "node_peak_bw_gbps", "solo_effective_uni_bw_gbps",
            "endpoint_rates", "scope", "rail_id", "switch_id", "root_complex_id",
            "endpoint_ids", "protocol_bands", "solo_uni_bw_gbps", "fabric",
        }
        _unknown(value, allowed, "link")
        name = value.get("name", "link")
        if not isinstance(name, str) or not name:
            raise ModelingError(f"link_name_invalid:{name}")
        peak, capacity, utilization = _link_bandwidth(value, name)
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
        ep_ids_raw = value.get("endpoint_ids", ())
        if isinstance(ep_ids_raw, Mapping):
            ep_ids = tuple(sorted((int(rk), str(ep)) for rk, ep in ep_ids_raw.items()))
        elif isinstance(ep_ids_raw, (list, tuple)):
            ep_ids = tuple(sorted((int(rk), str(ep)) for rk, ep in ep_ids_raw))
        else:
            raise ModelingError(f"link_endpoint_ids_invalid:{name}")
        semantics = value.get("semantics", value.get("channel_semantics", value.get("transport_semantics", "shared")))
        semantics = {"switched": "per_port"}.get(semantics, semantics)
        bands = parse_collective_bands(value.get("protocol_bands"))
        return cls(
            name=name, capacity_bytes_per_ms=capacity,
            members=tuple(value.get("members", default_members)),
            latency_us=float(value.get("latency_us", value.get("alpha_us", 0))),
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
            scope=value.get("scope"),
            rail_id=0 if value.get("rail_id") is None else value["rail_id"],
            switch_id=str(value.get("switch_id") or ""),
            root_complex_id=str(value.get("root_complex_id") or ""),
            endpoint_ids=ep_ids,
            protocol_bands=bands, protocol=str(value.get("protocol") or ""),
        )


def _network_override_keys(raw: Mapping) -> set[str]:
    return {f"{kind}_{suffix}" for kind in _NETWORK_OVERRIDE_CHANNELS
            for suffix in _NETWORK_OVERRIDE_SUFFIXES if f"{kind}_{suffix}" in raw}


def apply_network_overrides(links, raw: Mapping, *,
                            gbps_to_bytes_per_ms: float = GBPS_TO_BYTES_PER_MS):
    """Apply machine-level bandwidth, utilization and latency overrides."""
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
            bandwidth = (_positive_float(raw[bytes_key], f"{bytes_key}_invalid",
                                         allow_zero=False)
                         * BYTES_PER_S_TO_BYTES_PER_MS)
        elif raw.get(bw_key) is not None:
            kind_scale = (GBPS_TO_BYTES_PER_MS if kind == "intra_package"
                          else gbps_to_bytes_per_ms)
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
            peak = (link.effective_peak_capacity_bytes_per_ms
                    if bandwidth is None else bandwidth)
            capacity = peak * link_util
            link_latency = link.latency_us if latency is None else latency
            result[index] = replace(
                link, capacity_bytes_per_ms=capacity, latency_us=link_latency,
                bw_utilization=link_util,
                peak_capacity_bytes_per_ms=(peak if not math.isclose(link_util, 1.0)
                                            else None),
            )
    return tuple(result)


_CHANNEL_BANDWIDTH_TIERS = (
    "effective_uni_bw_gbps", "calibrated_uni_bw_gbps", "derived_uni_bw_gbps", "uni_peak_bw_gbps",
)
_LINK_BANDWIDTH_KEYS = (
    *_CHANNEL_BANDWIDTH_TIERS, "gbps", "capacity_gbps",
    "capacity_bytes_per_ms", "peak_capacity_bytes_per_ms",
)


def _channel_rate_gbps(record):
    """Effective/calibrated/derived service is already derated; only a peak needs utilization."""
    if "effective_uni_bw_gbps" in record:
        return record["effective_uni_bw_gbps"]
    for key in _CHANNEL_BANDWIDTH_TIERS[1:-1]:
        if record.get(key) is not None:
            return record[key]
    return None


def _link_bandwidth(record, name):
    util = next((record[key] for key in ("bw_utilization", "bw_ratio", "utilization")
                 if key in record), None)
    utilization = 1.0 if util is None else _positive_float(util, "link_bw_utilization_invalid")
    if utilization > 1:
        raise ModelingError("link_bw_utilization_exceeds_unity")
    peak = record.get("peak_capacity_bytes_per_ms")
    if peak is not None:
        peak = _positive_float(peak, f"link_peak_capacity_invalid:{name}")
    for key in ("capacity_gbps", "gbps", "uni_peak_bw_gbps"):
        if record.get(key) is not None:
            declared = _positive_float(record[key], f"link_peak_capacity_invalid:{name}") * GBPS_TO_BYTES_PER_MS
            if peak is not None and not math.isclose(peak, declared):
                raise ModelingError(f"link_bandwidth_declarations_conflict:{name}")
            peak = declared
            break
    capacity = record.get("capacity_bytes_per_ms")
    if capacity is not None:
        capacity = _positive_float(capacity, f"link_rate_invalid:{name}")
    effective = _channel_rate_gbps(record)
    if capacity is None and effective is not None:
        capacity = _positive_float(effective, f"link_rate_invalid:{name}") * GBPS_TO_BYTES_PER_MS
    if capacity is None:
        if peak is None or ("effective_uni_bw_gbps" in record and effective is None):
            raise ModelingError(f"link_rate_invalid:{name}")
        capacity = peak * utilization
    elif peak is None:
        if util is not None or effective is not None:
            peak = capacity / utilization
    else:
        if capacity > peak:
            raise ModelingError(f"link_effective_capacity_exceeds_peak:{name}")
        if util is not None and not math.isclose(capacity, peak * utilization):
            raise ModelingError(f"link_bandwidth_declarations_conflict:{name}")
        utilization = capacity / peak
    return peak, capacity, utilization


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


def _neutral_transfer_channel_links(raw, placements):
    """Generate only physical D2D links, respecting package/node membership."""
    from .device_input import placement_node_groups, placement_package_groups

    channels = raw.get("transfer_channels") or {}
    nodes = placement_node_groups(placements)
    groups = {
        "d2d_intra_package": placement_package_groups(placements),
        "d2d_intra_node": nodes,
        "d2d_inter_node": [(0, [rank for rank, _, _ in placements])] if len(nodes) > 1 else [],
    }
    links = []
    for name, domains in groups.items():
        if name not in channels:
            continue
        channel = channels[name]
        if not isinstance(channel, Mapping):
            raise ModelingError(f"transfer_channel_record_invalid:{name}")
        if ("effective_uni_bw_gbps" in channel and channel["effective_uni_bw_gbps"] is None
                or any(key in channel for key in _CHANNEL_BANDWIDTH_TIERS)
                and all(channel.get(key) is None for key in _CHANNEL_BANDWIDTH_TIERS)):
            continue
        kind = name.removeprefix("d2d_")
        record = dict(channel)
        if not {"latency_us", "alpha_us"} & record.keys() and raw.get(f"{kind}_latency_us") is not None:
            record["latency_us"] = raw[f"{kind}_latency_us"]
        if not set(_LINK_BANDWIDTH_KEYS) & record.keys():
            if raw.get(f"{kind}_bw_bytes_per_s") is not None:
                record["peak_capacity_bytes_per_ms"] = raw[f"{kind}_bw_bytes_per_s"] * BYTES_PER_S_TO_BYTES_PER_MS
            elif raw.get(f"{kind}_bw_gbps") is not None:
                record["capacity_gbps"] = raw[f"{kind}_bw_gbps"]
        if (not {"bw_utilization", "bw_ratio", "utilization"} & record.keys()
                and _channel_rate_gbps(record) is None and raw.get(f"{kind}_bw_util") is not None):
            record["bw_utilization"] = raw[f"{kind}_bw_util"]
        channel_name = record.pop("channel_name", name)
        record.setdefault("channel", channel_name)
        ranks = tuple(rank for rank, _, _ in placements)
        link = Link.from_mapping({**record, "name": name}, default_members=ranks)
        if not set(link.members).issubset(ranks):
            raise ModelingError(f"link_member_unknown:{name}")
        for index, domain in domains:
            local = set(domain)
            members = tuple(rank for rank in link.members if rank in local)
            if len(members) < 2:
                continue
            label = name if len(domains) == 1 else f"{name}:{index}"
            links.append(replace(link, name=label, members=members,
                                 endpoint_ids=tuple(x for x in link.endpoint_ids if x[0] in local),
                                 endpoint_rates=tuple(x for x in link.endpoint_rates if x[0] in local)))
    return tuple(links)


def calibration_warnings(calibration: Mapping | None, existing=()) -> list[str]:
    """Append advisory warning when hardware rate provenance is uncalibrated."""
    out = [str(item) for item in (existing or ()) if item]
    if isinstance(calibration, Mapping) and not bool(calibration.get("calibrated", False)):
        if UNCALIBRATED_SKU_WARNING not in out:
            out.append(UNCALIBRATED_SKU_WARNING)
    return out


def combine_calibrations(by_pool: Mapping[str, Mapping]) -> dict:
    """Combine per-pool hardware calibration summaries for disaggregated runs."""
    items = [dict(v) for v in by_pool.values() if isinstance(v, Mapping)]
    calibrated = bool(items) and all(bool(item.get("calibrated", False)) for item in items)
    provenances = sorted({str(item.get("rate_provenance", "")) for item in items if item.get("rate_provenance")})
    anchors = sorted({str(item.get("anchor_sku", "")) for item in items if item.get("anchor_sku")})
    return {
        "calibrated": calibrated,
        "rate_provenance": "+".join(provenances) if provenances else "declared_nominal",
        "anchor_sku": "+".join(anchors) if anchors else "declared",
        "pools": {str(k): dict(v) for k, v in by_pool.items() if isinstance(v, Mapping)},
    }


__all__ = [
    "BYTES_PER_S_TO_BYTES_PER_MS",
    "COLLECTIVE_PROTOCOL_KEYS",
    "COLLECTIVE_PROTOCOL_VALUES",
    "CollectiveBandSpec",
    "GBPS_TO_BYTES_PER_MS",
    "Link",
    "UNCALIBRATED_SKU_WARNING",
    "VALID_BAND_STARTUP_SCOPES",
    "VALID_LINK_SCOPES",
    "_PRESERVED_TOPOLOGY_KEYS",
    "_collective_protocols",
    "_network_override_keys",
    "_neutral_transfer_channel_links",
    "apply_network_overrides",
    "calibration_warnings",
    "combine_calibrations",
    "parse_collective_bands",
    "validate_protocol_bands",
]
