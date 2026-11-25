"""Policy validation, SKU host-DMA rate binding, and offload resources.

[Role] Validate offload policies without guessed external SSD or flat host-DDR
  defaults, bind per-SKU calibrated L0 ``P20`` / hardware PCIe & DDR rates,
  and submit shared FIFO transfer work on the causal engine.
[Invariants] The scheduler's Memory owns placement when available. External SSD
  offload fails closed with ``missing_required_scenario_fields:ssd_bw_gbps`` if
  the caller omits ``ssd_bw_gbps``. Host DDR/PCIe rates derive strictly from
  explicit declarations, calibrated L0 ``P20_host_dma_sync``, or registered SKU
  ``transfer_channels`` (zero flat ``32 GB/s`` or ``200 GB/s`` fallbacks).
[Timeouts & Locks] In-memory policy/primitive lookup and event submission. No
  network, device context, waits, or locks.
[Side-Effect Policy] Caches read-only L0 primitive lookups and execution Memory;
  submits transfer events on the caller's Engine.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Any

from mindie_sim.backends.ascend import DERIVED_ASCEND_SKUS
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.collectives.ports import (
    VALID_HOST_DIRECTIONS,
    host_dma_resources_for_dies,
    host_transfer_demand,
)
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError


_DEFAULT_DDR_BANDWIDTH_UTILIZATION = 0.70
_DEFAULT_DDR_LATENCY_US = 5.0
_DEFAULT_SSD_BANDWIDTH_UTILIZATION = 0.70
_DEFAULT_SSD_LATENCY_US = 80.0
_PREFETCH_VERSIONS = ("v1", "v2", "v3", "v4")
BYTES_PER_GB = 1_000_000_000.0
BYTES_PER_MS_PER_GBPS = 1_000_000.0
NS_PER_US = 1000.0
HOST_BF16_BYTES = 2.0
P20_CACHE_MAXSIZE = 64


def missing_scenario_error(fields: Sequence[str]) -> ModelingError:
    """Canonical Domain-1 fail-fast error for omitted user scenario inputs."""
    ordered = ",".join(str(field) for field in fields)
    return ModelingError(
        f"missing_required_scenario_fields:{ordered} — specify explicitly or ask the user"
    )


@lru_cache(maxsize=P20_CACHE_MAXSIZE)
def _cached_p20_metrics(sku: str) -> dict[str, float | None]:
    """Read calibrated L0 ``P20_host_dma_sync`` rates for one SKU if registered."""
    if not sku:
        return {}
    try:
        from mindie_sim.primitives.loader import load_primitives
        table = load_primitives(sku)
        cell = table.cell("P20_host_dma_sync")
    except (ModelingError, OSError, KeyError, ValueError):
        return {}
    out: dict[str, float | None] = {}
    for metric, key in (
        ("host_h2d_bytes_per_s", "h2d_bps"),
        ("host_d2h_bytes_per_s", "d2h_bps"),
        ("pinned_h2d_startup_ns", "h2d_startup_ns"),
        ("pinned_d2h_startup_ns", "d2h_startup_ns"),
    ):
        try:
            band = cell.band(metric)
            if band.p50 is not None and float(band.p50) > 0.0:
                out[key] = float(band.p50)
        except ModelingError:
            continue
    for metric, key in (
        ("host_rc_h2d_bytes_per_s", "host_rc_h2d_bps"),
        ("host_rc_d2h_bytes_per_s", "host_rc_d2h_bps"),
    ):
        try:
            band = cell.band(metric)
            val = band.p50 if (band.p50 is not None and float(band.p50) > 0.0) else band.certified_upper_capacity
            if val is not None and float(val) > 0.0:
                out[key] = float(val)
        except ModelingError:
            continue
    return out


def _channel_bw_gbps(channel: Mapping[str, Any] | None) -> tuple[float | None, bool]:
    """Extract (bandwidth_gbps, is_effective) from one transfer-channel mapping."""
    if not isinstance(channel, Mapping):
        return None, False
    for key, is_eff in (
        ("calibrated_uni_bw_gbps", True),
        ("solo_effective_uni_bw_gbps", True),
        ("effective_uni_bw_gbps", True),
        ("derived_uni_bw_gbps", True),
        ("uni_peak_bw_gbps", False),
    ):
        val = channel.get(key)
        if val is not None and not isinstance(val, bool) and float(val) > 0.0:
            return float(val), is_eff
    return None, False


def _profile_channel_default_gbps(sku: str, channel_name: str) -> float | None:
    """Return the registered Ascend profile's baseline effective BW for override detection."""
    if not sku:
        return None
    try:
        from mindie_sim.backends.ascend import profiles as ascend_profiles
        prof = ascend_profiles().get(sku)
    except (ImportError, OSError, KeyError, ValueError):
        return None
    if not isinstance(prof, Mapping):
        return None
    ch = (prof.get("transfer_channels") or {}).get(channel_name)
    if not isinstance(ch, Mapping):
        return None
    val = ch.get("effective_uni_bw_gbps")
    return float(val) if (val is not None and not isinstance(val, bool) and float(val) > 0.0) else None


def _as_channel_dict(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        try:
            return {str(k): v for k, v in dict(raw).items()}
        except (TypeError, ValueError):
            return None
    return None


def _extract_machine_host_fields(
    machine: Any,
) -> tuple[str, dict[str, float], dict[str, dict[str, Any]]]:
    """Extract ``(sku, device_rates, transfer_channels)`` from Hardware or dict."""
    if isinstance(machine, Mapping):
        sku = str(machine.get("sku") or "")
        dev = machine.get("device") if isinstance(machine.get("device"), Mapping) else {}
        if not dev and isinstance(machine.get("devices"), Sequence) and machine["devices"]:
            first = machine["devices"][0]
            dev = first if isinstance(first, Mapping) else {}
        rates = {
            str(k): float(v)
            for k, v in dict(dev.get("rates") or {}).items()
            if v is not None and not isinstance(v, bool)
        }
        raw_tc = machine.get("transfer_channels") or {}
        tc: dict[str, dict[str, Any]] = {}
        for k, v in dict(raw_tc).items():
            parsed = _as_channel_dict(v)
            if parsed is not None:
                tc[str(k)] = parsed
        return sku, rates, tc
    sku = str(getattr(machine, "sku", "") or "")
    dies = tuple(getattr(machine, "dies", ()) or ())
    rates = {
        str(k): float(v)
        for k, v in dict(getattr(dies[0], "rates", ()) or ()).items()
        if v is not None and not isinstance(v, bool)
    } if dies else {}
    raw_tc = getattr(machine, "transfer_channels", ()) or ()
    tc = {}
    for k, v in dict(raw_tc).items():
        parsed = _as_channel_dict(v)
        if parsed is not None:
            tc[str(k)] = parsed
    return sku, rates, tc


def _channel_is_caller_override(sku: str, ch_name: str, ch: Mapping[str, Any] | None) -> bool:
    """True when the machine's transfer channel differs from the registered profile default."""
    if not isinstance(ch, Mapping):
        return False
    eff = ch.get("effective_uni_bw_gbps")
    if eff is None or isinstance(eff, bool) or float(eff) <= 0.0:
        return False
    default_eff = _profile_channel_default_gbps(sku, ch_name)
    if default_eff is None:
        return True
    return not math.isclose(float(eff), default_eff, rel_tol=1e-6, abs_tol=1e-6)


def resolve_sku_host_dma_rates(machine: Any) -> dict[str, Any]:
    """Resolve SKU-bound PCIe H2D/D2H and Host DDR rates without flat fallbacks.

    Precedence:
      1. Explicit ``device.rates`` (``h2d``, ``d2h``, ``host_memory``, ``host_vector``)
      2. Caller-overridden ``transfer_channels["h2d"|"d2h"]``
      3. Calibrated L0 ``P20_host_dma_sync`` (``host_h2d_bytes_per_s``, ``host_d2h_bytes_per_s``,
         ``host_rc_h2d_bytes_per_s``, ``pinned_*_startup_ns``)
      4. Registered SKU ``transfer_channels["h2d"|"d2h"]`` (PCIe Gen4/Gen5 & ``node_peak_bw_gbps``)
    """
    sku, dev_rates, tc = _extract_machine_host_fields(machine)
    # Derived SKUs inherit their base host channel and P20 calibration; price them identically.
    channel_sku = str(DERIVED_ASCEND_SKUS.get(sku, {}).get("base", sku))
    p20 = _cached_p20_metrics(channel_sku)
    h2d_ch = tc.get("h2d")
    d2h_ch = tc.get("d2h")

    h2d_bps: float | None = None
    h2d_source = "missing"
    h2d_effective = False
    if dev_rates.get("h2d", 0.0) > 0.0:
        h2d_bps, h2d_source, h2d_effective = dev_rates["h2d"], "explicit_device", True
    elif _channel_is_caller_override(channel_sku, "h2d", h2d_ch):
        bw, eff = _channel_bw_gbps(h2d_ch)
        if bw is not None:
            h2d_bps, h2d_source, h2d_effective = bw * BYTES_PER_GB, "explicit_transfer_channel", eff
    elif p20.get("h2d_bps"):
        h2d_bps, h2d_source, h2d_effective = float(p20["h2d_bps"]), "l0_p20_calibrated", True
    else:
        bw, eff = _channel_bw_gbps(h2d_ch)
        if bw is not None:
            h2d_bps, h2d_source, h2d_effective = bw * BYTES_PER_GB, "sku_transfer_channel", eff

    d2h_bps: float | None = None
    d2h_source = "missing"
    d2h_effective = False
    if dev_rates.get("d2h", 0.0) > 0.0:
        d2h_bps, d2h_source, d2h_effective = dev_rates["d2h"], "explicit_device", True
    elif _channel_is_caller_override(channel_sku, "d2h", d2h_ch):
        bw, eff = _channel_bw_gbps(d2h_ch)
        if bw is not None:
            d2h_bps, d2h_source, d2h_effective = bw * BYTES_PER_GB, "explicit_transfer_channel", eff
    elif p20.get("d2h_bps"):
        d2h_bps, d2h_source, d2h_effective = float(p20["d2h_bps"]), "l0_p20_calibrated", True
    else:
        bw, eff = _channel_bw_gbps(d2h_ch)
        if bw is not None:
            d2h_bps, d2h_source, d2h_effective = bw * BYTES_PER_GB, "sku_transfer_channel", eff

    host_mem_bps: float | None = None
    host_mem_source = "missing"
    if dev_rates.get("host_memory", 0.0) > 0.0:
        host_mem_bps, host_mem_source = dev_rates["host_memory"], "explicit_device"
    elif p20.get("host_rc_h2d_bps"):
        host_mem_bps, host_mem_source = float(p20["host_rc_h2d_bps"]), "l0_p20_rc"
    else:
        for ch in (h2d_ch, d2h_ch):
            node_peak = (ch or {}).get("node_peak_bw_gbps") if isinstance(ch, Mapping) else None
            if node_peak is not None and not isinstance(node_peak, bool) and float(node_peak) > 0.0:
                host_mem_bps, host_mem_source = float(node_peak) * BYTES_PER_GB, "sku_transfer_channel"
                break

    host_vec_ops: float | None = None
    if dev_rates.get("host_vector", 0.0) > 0.0:
        host_vec_ops = dev_rates["host_vector"]
    elif host_mem_bps is not None:
        host_vec_ops = host_mem_bps / HOST_BF16_BYTES

    h2d_lat_us: float | None = None
    if p20.get("h2d_startup_ns"):
        h2d_lat_us = float(p20["h2d_startup_ns"]) / NS_PER_US
    elif isinstance(h2d_ch, Mapping) and h2d_ch.get("alpha_us") is not None:
        h2d_lat_us = float(h2d_ch["alpha_us"])

    d2h_lat_us: float | None = None
    if p20.get("d2h_startup_ns"):
        d2h_lat_us = float(p20["d2h_startup_ns"]) / NS_PER_US
    elif isinstance(d2h_ch, Mapping) and d2h_ch.get("alpha_us") is not None:
        d2h_lat_us = float(d2h_ch["alpha_us"])

    d2h_host_mem_bps: float | None = None
    if dev_rates.get("host_memory", 0.0) > 0.0:
        d2h_host_mem_bps = dev_rates["host_memory"]
    elif p20.get("host_rc_d2h_bps"):
        d2h_host_mem_bps = float(p20["host_rc_d2h_bps"])
    elif isinstance(d2h_ch, Mapping):
        node_peak = d2h_ch.get("node_peak_bw_gbps")
        if node_peak is not None and not isinstance(node_peak, bool) and float(node_peak) > 0.0:
            d2h_host_mem_bps = float(node_peak) * BYTES_PER_GB
    if d2h_host_mem_bps is None:
        d2h_host_mem_bps = host_mem_bps

    ddr_candidates = [v for v in (h2d_bps, host_mem_bps) if v is not None and v > 0.0]
    ddr_eff_gbps = (min(ddr_candidates) / BYTES_PER_GB) if ddr_candidates else None
    d2h_candidates = [v for v in (d2h_bps, d2h_host_mem_bps) if v is not None and v > 0.0]
    ddr_d2h_eff_gbps = (min(d2h_candidates) / BYTES_PER_GB) if d2h_candidates else None
    return {
        "sku": sku,
        "h2d_bps": h2d_bps,
        "d2h_bps": d2h_bps,
        "host_memory_bps": host_mem_bps,
        "host_vector_ops_per_s": host_vec_ops,
        "h2d_latency_us": h2d_lat_us,
        "d2h_latency_us": d2h_lat_us,
        "h2d_source": h2d_source,
        "d2h_source": d2h_source,
        "host_memory_source": host_mem_source,
        "h2d_is_effective": h2d_effective,
        "d2h_is_effective": d2h_effective,
        "ddr_effective_bw_gbps": ddr_eff_gbps,
        "ddr_d2h_effective_bw_gbps": ddr_d2h_eff_gbps,
    }


@dataclass(frozen=True)
class OffloadPolicy:
    """Validated host-DDR/SSD offload and prefetch policy (zero guessed SSD/DDR BW)."""

    expert_offload_enabled: bool = False
    force_expert_offload: bool = False
    expert_ddr_ratio: float = 0.0
    ddr_bw_gbps: float | None = None
    ddr_bw_util: float = _DEFAULT_DDR_BANDWIDTH_UTILIZATION
    ddr_latency_us: float = _DEFAULT_DDR_LATENCY_US
    ssd_bw_gbps: float | None = None
    ssd_bw_util: float = _DEFAULT_SSD_BANDWIDTH_UTILIZATION
    ssd_latency_us: float = _DEFAULT_SSD_LATENCY_US
    kvc_ddr_ratio: float = 0.0
    kvc_ssd_ratio: float = 0.0
    kvc_prefetch_version: str = "v3"
    mtp_ffn_hit_rate: float = 0.0
    mtp_kvc_hit_rate: float = 0.0
    ddr_h2d_bw_gbps: float | None = None
    ddr_d2h_bw_gbps: float | None = None
    ddr_h2d_bw_util: float | None = None
    ddr_d2h_bw_util: float | None = None
    ddr_h2d_latency_us: float | None = None
    ddr_d2h_latency_us: float | None = None

    def __post_init__(self) -> None:
        for name in ("expert_offload_enabled", "force_expert_offload"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"offload_policy_boolean_required:{name}")
        for name in ("expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio",
                     "mtp_ffn_hit_rate", "mtp_kvc_hit_rate"):
            _ratio(name, getattr(self, name))
        for name in ("ddr_bw_util", "ssd_bw_util"):
            _utilization(name, getattr(self, name))
        for name in ("ddr_h2d_bw_util", "ddr_d2h_bw_util"):
            if getattr(self, name) is not None:
                _utilization(name, getattr(self, name))
        for name in ("ddr_bw_gbps", "ddr_h2d_bw_gbps", "ddr_d2h_bw_gbps", "ssd_bw_gbps"):
            if getattr(self, name) is not None:
                _positive(name, getattr(self, name))
        for name in ("ddr_latency_us", "ssd_latency_us"):
            _nonnegative(name, getattr(self, name))
        for name in ("ddr_h2d_latency_us", "ddr_d2h_latency_us"):
            if getattr(self, name) is not None:
                _nonnegative(name, getattr(self, name))
        if self.kvc_prefetch_version not in _PREFETCH_VERSIONS:
            raise ValueError(f"offload_policy_invalid_prefetch_version:{self.kvc_prefetch_version}")
        if self.kvc_ddr_ratio + self.kvc_ssd_ratio > 1.0 + 1e-12:
            raise ValueError("offload_policy_kvc_ratios_exceed_unity")
        if self.kvc_ssd_ratio > 0.0 and self.ssd_bw_gbps is None:
            raise missing_scenario_error(("ssd_bw_gbps",))

    def ddr_bandwidth_gbps(self, direction: str) -> float | None:
        """Resolved DDR bandwidth (GB/s) for one host DMA direction."""
        if direction not in VALID_HOST_DIRECTIONS:
            raise ModelingError(f"host_transfer_direction_invalid:{direction}")
        specific = self.ddr_h2d_bw_gbps if direction == "h2d" else self.ddr_d2h_bw_gbps
        return self.ddr_bw_gbps if specific is None else specific

    def ddr_effective_rate_bytes_per_ms(self, direction: str) -> float | None:
        """Resolved effective DDR transfer rate (bytes/ms) for one direction."""
        bw = self.ddr_bandwidth_gbps(direction)
        if bw is None:
            return None
        specific_util = self.ddr_h2d_bw_util if direction == "h2d" else self.ddr_d2h_bw_util
        util = self.ddr_bw_util if specific_util is None else specific_util
        return float(bw) * float(util) * BYTES_PER_MS_PER_GBPS

    def ddr_startup_latency_us(self, direction: str) -> float:
        """Resolved DDR startup latency (us) for one host DMA direction."""
        if direction not in VALID_HOST_DIRECTIONS:
            raise ModelingError(f"host_transfer_direction_invalid:{direction}")
        specific = self.ddr_h2d_latency_us if direction == "h2d" else self.ddr_d2h_latency_us
        return float(self.ddr_latency_us if specific is None else specific)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None = None) -> "OffloadPolicy":
        """Build a policy with strict ratio, bandwidth, and enum validation."""
        if mapping is None:
            return cls()
        if isinstance(mapping, cls):
            return mapping
        if not isinstance(mapping, Mapping):
            raise ValueError("offload_policy_mapping_required")
        unknown = set(mapping) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError("offload_policy_unknown_fields:" + ",".join(sorted(unknown)))
        return cls(**dict(mapping))


def _ratio(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"offload_policy_invalid_ratio:{name}:{value}")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"offload_policy_invalid_ratio:{name}:{value}")
    return result


def _positive(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"offload_policy_must_be_positive:{name}:{value}")
    return float(value)


def _utilization(name: str, value: Any) -> float:
    result = _positive(name, value)
    if result > 1.0:
        raise ValueError(f"offload_policy_invalid_utilization:{name}:{value}")
    return result


def _nonnegative(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or float(value) < 0.0:
        raise ValueError(f"offload_policy_must_be_nonnegative:{name}:{value}")
    return float(value)


def _resolve_policy(policy: OffloadPolicy | Mapping[str, Any] | None) -> OffloadPolicy:
    if policy is None:
        return OffloadPolicy()
    if isinstance(policy, OffloadPolicy):
        return policy
    return OffloadPolicy.from_mapping(policy)


def resolve_case_offload_policy(case: CaseSpec, memory_policy: Any = None) -> OffloadPolicy | None:
    """Resolve active offload policy bound to the SKU's calibrated PCIe/DDR rates."""
    raw_meta = dict(case.runtime.metadata).get("offload", {})
    if raw_meta is None:
        raw_meta = {}
    if not isinstance(raw_meta, Mapping):
        raise ValueError("memory_offload_metadata_mapping_required")
    merged: dict[str, Any] = dict(raw_meta)
    mem_cfg = dict(case.runtime.memory or {})
    for key in ("expert_ddr_ratio", "kvc_ddr_ratio", "kvc_ssd_ratio"):
        if mem_cfg.get(key, 0.0):
            merged[key] = mem_cfg[key]
        if memory_policy is not None and getattr(memory_policy, key, 0.0):
            merged[key] = getattr(memory_policy, key)
    if not merged:
        return None
    h2d_needed = (
        float(merged.get("expert_ddr_ratio", 0.0) or 0.0) > 0.0
        or float(merged.get("kvc_ddr_ratio", 0.0) or 0.0) > 0.0
        or bool(merged.get("force_expert_offload", False))
    )
    d2h_needed = float(merged.get("kvc_ddr_ratio", 0.0) or 0.0) > 0.0
    common_bw = merged.get("ddr_bw_gbps")
    need_sku_h2d = h2d_needed and common_bw is None and merged.get("ddr_h2d_bw_gbps") is None
    need_sku_d2h = d2h_needed and common_bw is None and merged.get("ddr_d2h_bw_gbps") is None
    if need_sku_h2d or need_sku_d2h:
        caller_set_dir_bw = (
            merged.get("ddr_h2d_bw_gbps") is not None
            or merged.get("ddr_d2h_bw_gbps") is not None
        )
        host_rates = resolve_sku_host_dma_rates(case.machine)
        sku_h2d_gbps = host_rates.get("ddr_effective_bw_gbps")
        sku_d2h_gbps = host_rates.get("ddr_d2h_effective_bw_gbps")
        has_h2d = sku_h2d_gbps is not None and float(sku_h2d_gbps) > 0.0
        has_d2h = sku_d2h_gbps is not None and float(sku_d2h_gbps) > 0.0
        if need_sku_h2d and not has_h2d:
            field = "ddr_h2d_bw_gbps" if has_d2h else "ddr_bw_gbps"
            raise ModelingError(f"missing_required_hardware_host_dma_rate:{field}")
        if need_sku_d2h and not has_d2h:
            field = "ddr_d2h_bw_gbps" if has_h2d else "ddr_bw_gbps"
            raise ModelingError(f"missing_required_hardware_host_dma_rate:{field}")
        h2d_eff = bool(host_rates.get("h2d_is_effective"))
        d2h_eff = bool(host_rates.get("d2h_is_effective"))
        if need_sku_h2d:
            merged["ddr_h2d_bw_gbps"] = float(sku_h2d_gbps)
            if "ddr_bw_util" not in merged and "ddr_h2d_bw_util" not in merged:
                merged["ddr_h2d_bw_util"] = 1.0 if h2d_eff else _DEFAULT_DDR_BANDWIDTH_UTILIZATION
            if ("ddr_latency_us" not in merged
                    and "ddr_h2d_latency_us" not in merged
                    and host_rates.get("h2d_latency_us") is not None):
                merged["ddr_h2d_latency_us"] = float(host_rates["h2d_latency_us"])
        if need_sku_d2h:
            merged["ddr_d2h_bw_gbps"] = float(sku_d2h_gbps)
            if "ddr_bw_util" not in merged and "ddr_d2h_bw_util" not in merged:
                merged["ddr_d2h_bw_util"] = 1.0 if d2h_eff else _DEFAULT_DDR_BANDWIDTH_UTILIZATION
            if ("ddr_latency_us" not in merged
                    and "ddr_d2h_latency_us" not in merged
                    and host_rates.get("d2h_latency_us") is not None):
                merged["ddr_d2h_latency_us"] = float(host_rates["d2h_latency_us"])
        if not caller_set_dir_bw and need_sku_h2d:
            merged["ddr_bw_gbps"] = float(sku_h2d_gbps)
            if "ddr_bw_util" not in merged and h2d_eff and (not need_sku_d2h or d2h_eff):
                merged["ddr_bw_util"] = 1.0
            if "ddr_latency_us" not in merged and host_rates.get("h2d_latency_us") is not None:
                merged["ddr_latency_us"] = float(host_rates["h2d_latency_us"])
    policy = OffloadPolicy.from_mapping(merged)
    if (policy.expert_ddr_ratio <= 0.0 and policy.kvc_ddr_ratio <= 0.0
            and policy.kvc_ssd_ratio <= 0.0 and not policy.force_expert_offload):
        return None
    return policy


def offload_resources(policy: OffloadPolicy | None, namespace: str, dp: int, pp: int) -> list[Any]:
    """Build full-duplex host DDR (H2D/D2H) and SSD transfer resources for Execution."""
    if policy is None:
        return []
    from mindie_sim.prediction.engine import Resource
    h2d_needed = (
        policy.expert_ddr_ratio > 0.0
        or policy.kvc_ddr_ratio > 0.0
        or policy.force_expert_offload
    )
    d2h_needed = policy.kvc_ddr_ratio > 0.0
    h2d_rate = policy.ddr_effective_rate_bytes_per_ms("h2d")
    d2h_rate = policy.ddr_effective_rate_bytes_per_ms("d2h")
    if h2d_needed and h2d_rate is None:
        field = "ddr_h2d_bw_gbps" if d2h_rate is not None else "ddr_bw_gbps"
        raise ModelingError(f"missing_required_hardware_host_dma_rate:{field}")
    if d2h_needed and d2h_rate is None:
        field = "ddr_d2h_bw_gbps" if h2d_rate is not None else "ddr_bw_gbps"
        raise ModelingError(f"missing_required_hardware_host_dma_rate:{field}")
    if policy.kvc_ssd_ratio > 0.0 and policy.ssd_bw_gbps is None:
        raise missing_scenario_error(("ssd_bw_gbps",))
    resources = []
    for replica in range(max(1, dp)):
        for stage in range(max(1, pp)):
            if h2d_rate is not None:
                resources.append(Resource(
                    f"{namespace}offload_ddr_h2d@r{replica}:s{stage}",
                    h2d_rate, "bytes", "fifo"))
            if d2h_rate is not None:
                resources.append(Resource(
                    f"{namespace}offload_ddr_d2h@r{replica}:s{stage}",
                    d2h_rate, "bytes", "fifo"))
            if policy.ssd_bw_gbps is not None:
                ssd_rate = policy.ssd_bw_gbps * policy.ssd_bw_util * BYTES_PER_MS_PER_GBPS
                resources.append(Resource(f"{namespace}offload_ssd@r{replica}:s{stage}",
                                          ssd_rate, "bytes", "fifo"))
    return resources


def _ensure_host_dma_resources(execution: Any, machine: Any, dies: Sequence[Any]) -> None:
    """Register physical directional host solo and RC Engine rows if missing."""
    if machine is None or not dies:
        return
    host_res = host_dma_resources_for_dies(
        machine, dies, namespace=execution.namespace
    )
    missing = [res for res in host_res if res.name not in execution.engine.resources]
    if missing:
        execution.engine.extend(missing)


def _die_ddr_resource_name(execution: Any, replica: int, stage: int, direction: str, rank_id: int) -> str:
    return f"{execution.namespace}offload_ddr_{direction}@r{replica}:s{stage}:d{rank_id}"


def _ensure_die_ddr_resources(
    execution: Any, replica: int, stage: int, direction: str, dies: Sequence[Any],
) -> None:
    """Register per-die directional FIFO DDR ports so distinct dies never serialize."""
    if not dies:
        return
    stage_res_name = f"{execution.namespace}offload_ddr_{direction}@r{replica}:s{stage}"
    stage_res = execution.engine.resources.get(stage_res_name)
    rate = float(stage_res.rate) if stage_res is not None else None
    if rate is None:
        policy = getattr(execution, "offload_policy", None)
        rate = policy.ddr_effective_rate_bytes_per_ms(direction) if policy is not None else None
    if rate is None or rate <= 0.0:
        return
    from mindie_sim.prediction.engine import Resource
    missing = []
    seen: set[str] = set()
    for idx, die in enumerate(dies):
        rank_id = int(getattr(die, "rank", idx))
        name = _die_ddr_resource_name(execution, replica, stage, direction, rank_id)
        if name not in execution.engine.resources and name not in seen:
            seen.add(name)
            missing.append(Resource(name, rate, "bytes", "fifo"))
    if missing:
        execution.engine.extend(missing)


def _host_ddr_work(
    execution: Any,
    replica: int,
    stage: int,
    direction: str,
    ddr_bytes: float,
    *,
    rank_ddr_bytes: Mapping[int | None, float] | None = None,
) -> dict[str, float]:
    """Build coupled directional offload DDR + physical host_solo + RC + HBM work.

    For uniform per-rank payloads across ``stage_dies``, charges ``1x`` payload
    per die's directional DDR port and ``host_solo:{direction}:d{rank}``, ``1x``
    onto HBM, and sums per-die payloads onto each die's
    ``RootComplexKey(node, root_complex_id, direction)``. On multi-die stages,
    each die owns an independent directional FIFO DDR port so distinct dies
    transfer concurrently up to their shared Root Complex ceiling.
    """
    if direction not in VALID_HOST_DIRECTIONS:
        raise ModelingError(f"host_transfer_direction_invalid:{direction}")
    ddr_res = f"{execution.namespace}offload_ddr_{direction}@r{replica}:s{stage}"
    hbm_res = execution.stage_names[replica, stage]["hbm"]
    machine = getattr(getattr(execution, "case", None), "machine", None)
    stage_dies = tuple(getattr(execution, "stage_dies", {}).get((replica, stage), ()) or ())
    if machine is None or not stage_dies:
        return {ddr_res: float(ddr_bytes), hbm_res: float(ddr_bytes)}
    multi_die = len(stage_dies) > 1

    if not rank_ddr_bytes or (len(rank_ddr_bytes) == 1 and None in rank_ddr_bytes):
        _ensure_host_dma_resources(execution, machine, stage_dies)
        demand = host_transfer_demand(
            machine, stage_dies[0], direction, float(ddr_bytes),
            namespace=execution.namespace, dies=stage_dies,
        )
        if multi_die:
            _ensure_die_ddr_resources(execution, replica, stage, direction, stage_dies)
            for idx, die in enumerate(stage_dies):
                rank_id = int(getattr(die, "rank", idx))
                demand[_die_ddr_resource_name(execution, replica, stage, direction, rank_id)] = float(ddr_bytes)
        else:
            demand[ddr_res] = float(ddr_bytes)
        demand[hbm_res] = float(ddr_bytes)
        return demand

    unranked = float(rank_ddr_bytes.get(None, 0.0))
    die_by_rank = {int(getattr(die, "rank", idx)): die for idx, die in enumerate(stage_dies)}
    all_dies_by_rank = {
        int(getattr(die, "rank", idx)): die
        for idx, die in enumerate(getattr(machine, "dies", ()) or ())
    }
    per_die: dict[int, tuple[Any, float]] = {}
    if unranked > 0.0:
        for idx, die in enumerate(stage_dies):
            rank_id = int(getattr(die, "rank", idx))
            per_die[rank_id] = (die, unranked)
    for rank, amount in rank_ddr_bytes.items():
        if rank is None or float(amount) <= 0.0:
            continue
        rank_int = int(rank)
        die = die_by_rank.get(rank_int)
        if die is None and 0 <= rank_int < len(stage_dies):
            die = stage_dies[rank_int]
        if die is None:
            die = all_dies_by_rank.get(rank_int, stage_dies[0])
        rank_id = int(getattr(die, "rank", 0))
        prev_die, prev_amt = per_die.get(rank_id, (die, 0.0))
        per_die[rank_id] = (prev_die, prev_amt + float(amount))

    participants = tuple(die for die, _ in per_die.values())
    _ensure_host_dma_resources(execution, machine, participants)
    if multi_die:
        _ensure_die_ddr_resources(execution, replica, stage, direction, participants)
    demand: dict[str, float] = {}
    for rank_id, (die, amount) in per_die.items():
        part = host_transfer_demand(
            machine, die, direction, amount, namespace=execution.namespace,
        )
        for key, val in part.items():
            demand[key] = demand.get(key, 0.0) + val
        if multi_die:
            die_res = _die_ddr_resource_name(execution, replica, stage, direction, rank_id)
            demand[die_res] = demand.get(die_res, 0.0) + amount
    solo_bytes = max((amt for _, amt in per_die.values()), default=float(ddr_bytes))
    if not multi_die:
        demand[ddr_res] = solo_bytes
    demand[hbm_res] = solo_bytes
    return demand


def _submit_offload_transfer(
    engine: Any, name: str, work: Mapping[str, float], *,
    after: tuple[Any, ...], latency_us: float, owner: str,
    coupled: bool = False,
) -> Any:
    startup_ms = float(latency_us) / NS_PER_US
    if startup_ms > 0.0:
        start = engine.submit(f"{name}:startup", after=after, latency=startup_ms, owner=owner)
        return engine.submit(name, work, after=(start,), owner=owner, coupled=coupled)
    return engine.submit(name, work, after=after, owner=owner, coupled=coupled)


def _offload_memory_for(execution: Any) -> Memory:
    """The one Memory shared with the service Scheduler when available."""
    memory = getattr(execution, "memory", None)
    if memory is None:
        memory = getattr(execution, "_offload_memory", None)
    if memory is None:
        memory = Memory(execution.case)
    execution._offload_memory = memory
    return memory
