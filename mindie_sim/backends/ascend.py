"""Ascend capability input adapter.

[Role] Materialize SKU metadata into neutral devices, buffers and links.
[Invariants] The simulator sees explicit rates. Nominal vector issue and launch
  assumptions are exposed and can be overridden; they are never measurements.
  Machine transfer declarations merge by channel over the registered profile;
  explicit machine links replace generated links. Transfer-channel ``*_bw_gbps``
  and convenience ``inter_node_bw_gbps``/``intra_node_bw_gbps`` fields use a
  single canonical GB/s unit regardless of whether a ``topology_preset`` is set.
  Multi-tier 4P/8P and SuperPod links preserve every declared boundary; a
  measured channel rate resolves each generated tier once and a unity
  bandwidth utilization emits exactly the historical link key set.
[Timeouts & Locks] Bounded local metadata reads; no device context.
[Side-Effect Policy] No external writes or probes.
"""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

from mindie_sim.schema.common import ModelingError, _positive_float
from mindie_sim.schema.device_input import (
    _deep_merge as _merge_nested, machine_device_defaults, merge_device_inputs,
)
from .ascend_channels import (
    DIRECT_MESH_RANKS,
    _CONVENIENCE_FIELDS,
    _apply_convenience,
    _caller_bandwidth_channels,
    _generated_links,
    _merge_channels,
    _merge_mapping,
    default_ranks_per_node,
)
from .instruction_presets import expand_instruction_presets

# Named topology preset constants (preserving public preset contract).
# Utilization factors below are explicit nominal preset deratings applied ONLY
# to uncalibrated nominal peaks when no effective/calibrated channel bandwidth
# is supplied in ``profiles/ascend.json`` or L0 ``P18_hccl_collective_band``
# (see ``_ASSUMPTIONS["transfer_bandwidth_precedence"]``).
A2_STD_RANKS_PER_NODE = 8
A2_ROCE_PEAK_GBPS = 25.0
#: Nominal 200GbE RoCE wire-framing derating when uncalibrated (profiles/ascend.json).
A2_ROCE_UTIL = 0.80
A2_INTER_LATENCY_US = 10.0

A3_DIES_PER_PACKAGE = 2
A3_NODE_RANKS_COMPACT = 8
A3_NODE_RANKS_FULL = 16
A3_SUPERPOD_PACKAGES = 384
A3_SUPERPOD_RANKS = A3_SUPERPOD_PACKAGES * A3_DIES_PER_PACKAGE
A3_INTRA_PEAK_GBPS = 196.0
A3_INTRA_BI_PEAK_GBPS = 2 * A3_INTRA_PEAK_GBPS
#: Nominal A3 intra-node HCCS protocol framing derating when uncalibrated.
A3_INTRA_UTIL = 0.80
#: Nominal A3 multi-tier SuperPod UB/HCCS fabric derating when uncalibrated.
A3_SUPERPOD_UTIL = 0.70
A3_ROCE_PEAK_GBPS = 25.0
#: Nominal A3 inter-node RoCE wire-framing derating when uncalibrated.
A3_ROCE_UTIL = 0.80
A3_INTRA_LATENCY_US = 1.5
A3_INTER_NODE_LATENCY_US = 5.0
A3_SUPERPOD_LATENCY_US = 3.5
A3_INTER_POD_LATENCY_US = 10.0

A5_POD_RANKS_PER_NODE = 64
A5_POD_SUPERPOD_RANKS = 8192
A5_POD_INTRA_BW_GBPS = 800.0
#: Nominal A5 Pod intra-domain UB-Direct framing derating when uncalibrated.
A5_POD_INTRA_UTIL = 0.85
A5_POD_INTER_POD_BW_GBPS = 100.0
#: Nominal A5 inter-pod fabric derating when uncalibrated.
A5_POD_INTER_UTIL = 0.80
A5_POD_INTER_LATENCY_US = 4.5

A5_SERVER_RANKS_PER_NODE = 8
A5_SERVER_SUBGROUP_BW_GBPS = 560.0
A5_SERVER_NODE_BW_GBPS = 640.0
#: Nominal A5 Server intra-node UB-Direct framing derating when uncalibrated.
A5_SERVER_INTRA_UTIL = 0.85
A5_SERVER_INTER_BW_GBPS = 200.0
#: Nominal A5 Server inter-node RoCE/UB derating when uncalibrated.
A5_SERVER_INTER_UTIL = 0.80

A5_PCIE_RANKS_PER_NODE = 16
A5_PCIE_QUAD_BW_GBPS = 288.0
A5_PCIE_OCTA_BW_GBPS = 128.0

_BYTES_PER_GIB = 1024 ** 3
A5_DT_425_AIC_CORES = 28
A5_DT_486_AIC_CORES = 32
A5_DT_547_AIC_CORES = 36
A5_DT_96_GIB = 96
A5_DT_144_GIB = 144

DERIVED_ASCEND_SKUS: dict[str, dict[str, object]] = {
    "Ascend950DT_425_96": {
        "base": "Ascend950DT_9582",
        "aic_count": A5_DT_425_AIC_CORES,
        "memory_bytes": A5_DT_96_GIB * _BYTES_PER_GIB,
    },
    "Ascend950DT_486_96": {
        "base": "Ascend950DT_9582",
        "aic_count": A5_DT_486_AIC_CORES,
        "memory_bytes": A5_DT_96_GIB * _BYTES_PER_GIB,
    },
    "Ascend950DT_547_96": {
        "base": "Ascend950DT_9582",
        "aic_count": A5_DT_547_AIC_CORES,
        "memory_bytes": A5_DT_96_GIB * _BYTES_PER_GIB,
    },
    "Ascend950DT_547_144": {
        "base": "Ascend950DT_9582",
        "aic_count": A5_DT_547_AIC_CORES,
        "memory_bytes": A5_DT_144_GIB * _BYTES_PER_GIB,
    },
}
A5_PCIE_NODE_BW_GBPS = 116.0
A5_PCIE_ROCE_PEAK_GBPS = 50.0
A5_PCIE_INTRA_LATENCY_US = 2.0
A5_PCIE_INTER_LATENCY_US = 8.0

_PRESERVED_TOPOLOGY_FIELDS = (
    "intra_bw_4p_gbps", "intra_bw_8p_gbps", "superpod_size", "max_dies",
    "superpod_bw_gbps", "inter_pod_bw_gbps", "superpod_latency_us",
    "inter_pod_latency_us", "topology_preset",
)
_TOPOLOGY_FIELDS = _CONVENIENCE_FIELDS | set(_PRESERVED_TOPOLOGY_FIELDS)
_PRESET_ARCH_PREFIXES = {"a2": "a2_", "a3": "a3_", "a5": "a5_"}


_ASSUMPTIONS = {
    "transfer_bandwidth_units": (
        "All *_bw_gbps convenience, preset, transfer_channels, and "
        "Link.capacity_gbps fields use canonical GB/s; *_bw_bytes_per_s fields "
        "use B/s and win over same-layer GB/s"
    ),
    "transfer_bandwidth_precedence": (
        "explicit machine.links > explicit transfer_channels channel bandwidth > "
        "convenience bytes_per_s > same-layer Gb/s > profile effective; within a "
        "channel explicit effective > explicit calibrated > explicit derived > "
        "explicit uni_peak, and null preserves unknown/disabled semantics. An "
        "effective/calibrated value resolves the tier's service capacity once: "
        "preset utilization derates only a nominal peak, and 4P/8P links inherit "
        "the measured node capacity"
    ),
    "transfer_channel_scope": (
        "Only d2d_intra_package, d2d_intra_node and d2d_inter_node generate "
        "device graph links. Other registered channels remain metadata; they "
        "do not override host/offload services or the P/D connector. Generated "
        "links retain per_port capacity unless semantics is explicitly declared."
    ),
    "topology_tiers": (
        "topology_preset supplies defaults only for fields not set explicitly; "
        "4P/8P subgroups and SuperPod domains are structural link boundaries, "
        "and bandwidth utilization scales the declared peak into service capacity"
    ),
}

ASCEND_TOPOLOGY_PRESETS: dict[str, dict[str, object]] = {
    "a2_standard": {
        "ranks_per_node": A2_STD_RANKS_PER_NODE, "dies_per_package": 1,
        "inter_node_bw_gbps": A2_ROCE_PEAK_GBPS,
        "inter_node_bw_util": A2_ROCE_UTIL,
        "inter_node_latency_us": A2_INTER_LATENCY_US,
    },
    "a3_8p": {
        "ranks_per_node": A3_NODE_RANKS_COMPACT,
        "dies_per_package": A3_DIES_PER_PACKAGE,
        "intra_bw_4p_gbps": A3_INTRA_PEAK_GBPS,
        "intra_node_bw_gbps": A3_INTRA_PEAK_GBPS,
        "intra_node_bw_util": A3_INTRA_UTIL,
        "inter_node_bw_gbps": A3_ROCE_PEAK_GBPS,
        "inter_node_bw_util": A3_ROCE_UTIL,
        "intra_node_latency_us": A3_INTRA_LATENCY_US,
        "inter_node_latency_us": A3_INTER_NODE_LATENCY_US,
    },
    "a3_16p": {
        "ranks_per_node": A3_NODE_RANKS_FULL,
        "dies_per_package": A3_DIES_PER_PACKAGE,
        "intra_bw_4p_gbps": A3_INTRA_PEAK_GBPS,
        "intra_bw_8p_gbps": A3_INTRA_PEAK_GBPS,
        "intra_node_bw_gbps": A3_INTRA_PEAK_GBPS,
        "intra_node_bw_util": A3_INTRA_UTIL,
        "inter_node_bw_gbps": A3_ROCE_PEAK_GBPS,
        "inter_node_bw_util": A3_ROCE_UTIL,
        "intra_node_latency_us": A3_INTRA_LATENCY_US,
        "inter_node_latency_us": A3_INTER_NODE_LATENCY_US,
    },
    "a3_superpod": {
        "ranks_per_node": A3_NODE_RANKS_FULL,
        "dies_per_package": A3_DIES_PER_PACKAGE,
        "superpod_size": A3_SUPERPOD_RANKS,
        "intra_node_bw_gbps": A3_INTRA_PEAK_GBPS,
        "intra_node_bw_util": A3_INTRA_UTIL,
        "superpod_bw_gbps": A3_INTRA_PEAK_GBPS,
        "inter_pod_bw_gbps": A3_ROCE_PEAK_GBPS,
        "inter_node_bw_util": A3_SUPERPOD_UTIL,
        "intra_node_latency_us": A3_INTRA_LATENCY_US,
        "inter_node_latency_us": A3_SUPERPOD_LATENCY_US,
        "inter_pod_latency_us": A3_INTER_POD_LATENCY_US,
    },
    "a5_pod": {
        "collective_protocols": {"alltoall": "concurrent", "alltoallv": "concurrent"},
        "ranks_per_node": A5_POD_RANKS_PER_NODE, "dies_per_package": 1,
        "superpod_size": A5_POD_SUPERPOD_RANKS,
        "intra_node_bw_gbps": A5_POD_INTRA_BW_GBPS,
        "intra_node_bw_util": A5_POD_INTRA_UTIL,
        "superpod_bw_gbps": A5_POD_INTRA_BW_GBPS,
        "inter_pod_bw_gbps": A5_POD_INTER_POD_BW_GBPS,
        "inter_node_bw_util": A5_POD_INTER_UTIL,
        "intra_node_latency_us": A3_INTRA_LATENCY_US,
        "inter_node_latency_us": A5_POD_INTER_LATENCY_US,
        "inter_pod_latency_us": A3_INTER_POD_LATENCY_US,
    },
    "a5_server": {
        "ranks_per_node": A5_SERVER_RANKS_PER_NODE, "dies_per_package": 1,
        "intra_bw_4p_gbps": A5_SERVER_SUBGROUP_BW_GBPS,
        "intra_bw_8p_gbps": A5_SERVER_NODE_BW_GBPS,
        "intra_node_bw_gbps": A5_SERVER_NODE_BW_GBPS,
        "intra_node_bw_util": A5_SERVER_INTRA_UTIL,
        "inter_node_bw_gbps": A5_SERVER_INTER_BW_GBPS,
        "inter_node_bw_util": A5_SERVER_INTER_UTIL,
        "intra_node_latency_us": A3_INTRA_LATENCY_US,
        "inter_node_latency_us": A3_INTER_NODE_LATENCY_US,
    },
    "a5_pcie_16p": {
        "ranks_per_node": A5_PCIE_RANKS_PER_NODE, "dies_per_package": 1,
        "intra_bw_4p_gbps": A5_PCIE_QUAD_BW_GBPS,
        "intra_bw_8p_gbps": A5_PCIE_OCTA_BW_GBPS,
        "intra_node_bw_gbps": A5_PCIE_NODE_BW_GBPS,
        "intra_node_bw_util": A5_SERVER_INTRA_UTIL,
        "inter_node_bw_gbps": A5_PCIE_ROCE_PEAK_GBPS,
        "inter_node_bw_util": A3_ROCE_UTIL,
        "intra_node_latency_us": A5_PCIE_INTRA_LATENCY_US,
        "inter_node_latency_us": A5_PCIE_INTER_LATENCY_US,
    },
}


@lru_cache(maxsize=1)
def profiles():
    with Path(__file__).with_name("profiles").joinpath("ascend.json").open(encoding="utf-8") as stream:
        return json.load(stream)


def registered_matrix_capability(sku, arch_family, declared):
    """Recover SKU-specific matrix modes for a neutral profile declaration."""
    profile = profiles().get(sku)
    if profile is None or profile["architecture"] != arch_family:
        return {}
    arch = profile["device"]["compute_arch"]
    if declared.get("preset") not in (None, "", arch["preset"]):
        return {}
    return {key: arch[key] for key in ("preset", "representations", "native_compute")
            if key in arch}


def _preset(raw):
    name = raw.get("topology_preset")
    if name is None:
        return dict(raw), {}
    if not isinstance(name, str) or name not in ASCEND_TOPOLOGY_PRESETS:
        raise ModelingError(f"ascend_topology_preset_unknown:{name}")
    preset = deepcopy(ASCEND_TOPOLOGY_PRESETS[name])
    merged = {**preset, **{key: value for key, value in raw.items()
                          if key != "topology_preset"}}
    merged["topology_preset"] = name
    return merged, preset


_MHZ_TO_HZ = 1_000_000.0


def apply_observed_aicore_clock(device, profile, *, reference_mhz=None, explicit=None):
    """Convert clock-bound rates from their declared clock to an observed clock."""
    observed = device.get("observed_aicore_freq_mhz")
    if observed is None:
        return device
    observed = _positive_float(observed, "observed_aicore_freq_mhz_invalid")
    nominal = (reference_mhz if reference_mhz is not None
               else profile["source"]["declaration"]["configured_clock_mhz"])
    reference = (nominal if explicit is not None
                 else device.get("aicore_freq_mhz", nominal))
    reference = _positive_float(reference, "aicore_freq_mhz_invalid")
    ratio = observed / reference
    clock = {"observed_aicore_freq_mhz": observed, "aicore_freq_mhz": observed}
    explicit = explicit or {}
    explicit_arch = explicit.get("compute_arch", {})
    if ratio == 1:
        if "clock_hz" not in explicit_arch:
            data = deepcopy(device)
            data.update(clock)
            data.setdefault("compute_arch", {})["clock_hz"] = observed * _MHZ_TO_HZ
            return data
        return {**device, **clock}
    data = deepcopy(device)
    data.update(clock)
    if "clock_hz" not in explicit_arch:
        data.setdefault("compute_arch", {})["clock_hz"] = observed * _MHZ_TO_HZ
    explicit_rates = explicit.get("rates", {})
    clock_rates = set(profile["device"]["rates"]) - {
        "hbm", "scalar_store_issue",
    }
    for name, peak in data.get("compute_peaks", {}).items():
        if name not in explicit.get("compute_peaks", {}):
            data["compute_peaks"][name] = peak * ratio
    for name, rate in data.get("rates", {}).items():
        if name in clock_rates and name not in explicit_rates:
            data["rates"][name] = rate * ratio
    for name, mode in data.get("service_modes", {}).items():
        resource = mode.get("resource")
        if (resource in clock_rates and resource not in explicit_rates
                and name not in {"mte2_aic_dram", "mte2_aiv_dram"}
                and "rate" not in explicit.get("service_modes", {}).get(name, {})):
            mode["rate"] *= ratio
    for name, service in data.get("compute_arch", {}).get("instruction_services", {}).items():
        if (name in explicit_arch.get("instruction_services", {})
                or name == "gm_hbm_cacheline_128b"
                or (name == "vbitsort32" and "clock_hz" in explicit_arch)):
            continue
        if "clock_hz" in service:
            service["clock_hz"] *= ratio
        elif "per_owner_rate" in service and "scale_with" not in service:
            service["per_owner_rate"] *= ratio
    return data


def _materialize_device(device, profile, sku, explicit):
    materialized = expand_instruction_presets(device, profile, sku)
    decl_mhz = profile.get("source", {}).get("declaration", {}).get("configured_clock_mhz")
    if decl_mhz is not None:
        materialized = deepcopy(materialized)
        arch = materialized.setdefault("compute_arch", {})
        if arch.get("clock_hz") is None:
            arch["clock_hz"] = float(decl_mhz) * _MHZ_TO_HZ
    return apply_observed_aicore_clock(materialized, profile, explicit=explicit)


def materialize(raw):
    if raw.get("sku_profile"):
        raise ModelingError("sku_profile_unsupported:declare_device_rates_peaks_and_buffers")
    if raw["sku"] not in profiles():
        raise ModelingError(f"sku_not_registered:{raw['sku']}:declare_device_geometry")
    merged, preset = _preset(raw)
    preset_active = bool(preset)
    profile = profiles()[raw["sku"]]
    arch_low = str(profile.get("architecture") or "").strip().lower()
    preset_name = merged.get("topology_preset")
    if preset_name and arch_low in _PRESET_ARCH_PREFIXES:
        if not str(preset_name).startswith(_PRESET_ARCH_PREFIXES[arch_low]):
            raise ModelingError(
                f"ascend_topology_preset_incompatible:{raw['sku']}:{preset_name}"
            )
    explicit_device = merge_device_inputs(machine_device_defaults(merged), merged.get("device", {}))
    device = merge_device_inputs(profile["device"], explicit_device)
    channels, explicit, declared = _merge_channels(merged, profile)
    bw_caller = _caller_bandwidth_channels(raw)
    _apply_convenience(raw, preset, channels, explicit, declared, preset_active)
    data = {**merged, "device": _materialize_device(device, profile, raw["sku"], explicit_device),
            "transfer_channels": channels,
            "architecture": profile["architecture"]}
    for field_name in ("devices", "dies"):
        if field_name in merged:
            data[field_name] = [_materialize_device(
                merge_device_inputs(device, record), profile, raw["sku"],
                merge_device_inputs(explicit_device, record))
                for record in merged[field_name]]
    count = int(merged.get("num_devices", len(merged.get(
        "devices", merged.get("dies", ()))) or 1))
    phys_rpn = int(
        raw.get("physical_ranks_per_node")
        or preset.get("ranks_per_node")
        or profile.get("physical_ranks_per_node")
        or profile.get("ranks_per_node")
        or default_ranks_per_node(raw["sku"])
    )
    per_node = int(merged.get("ranks_per_node", min(count, phys_rpn)))
    phys_rpn = max(phys_rpn, per_node)
    per_package = int(merged.get("dies_per_package", profile["dies_per_package"]))
    data.update(
        ranks_per_node=per_node,
        physical_ranks_per_node=phys_rpn,
        dies_per_package=per_package,
    )
    if "collective_protocols" not in raw:
        is_ub_domain = (
            preset_name == "a5_pod"
            or (not preset and per_node <= DIRECT_MESH_RANKS and count <= per_node)
        )
        if "links" in raw or not is_ub_domain:
            data.pop("collective_protocols", None)
        elif raw["sku"].startswith("Ascend950"):
            data["collective_protocols"] = {
                "alltoall": "concurrent", "alltoallv": "concurrent"}
    if "links" not in raw:
        data["links"] = _generated_links(
            merged, raw, preset, channels, count, per_package, per_node,
            explicit, bw_caller, declared)
        optical = channels["d2d_inter_node"]
        if (merged.get("topology_preset") == "a3_superpod"
                and optical["protocol"] == "HCCS_OPTICAL_SUPERPOD"
                and "bi_peak_bw_gbps" not in declared.get("d2d_inter_node", ())):
            optical["bi_peak_bw_gbps"] = max(
                A3_INTRA_BI_PEAK_GBPS, optical["uni_peak_bw_gbps"])
            optical["effective_bi_bw_gbps"] = (
                optical["bi_peak_bw_gbps"] * optical["bw_utilization"])
        for tier, key in (("d2d_intra_4p", "intra_bw_4p_gbps"),
                          ("d2d_intra_8p", "intra_bw_8p_gbps")):
            if tier in explicit and raw.get(key) is None:
                data[key] = channels[tier].get("uni_peak_bw_gbps")
    profile_source = deepcopy(profile.get("source") or {})
    caller_source = dict(raw.get("source") or {})
    caller_cal = dict(caller_source.get("calibration") or {})
    calibration = dict(profile_source.get("calibration") or {})
    calibrated = bool(calibration.get("calibrated", False))
    rate_provenance = calibration.get("rate_provenance") or (
        "silicon_calibrated" if calibrated else "declared_nominal")
    source_sku = str(calibration.get("source_sku") or raw["sku"])
    anchor_sku = str(calibration.get("anchor_sku") or source_sku)
    calibration.update(calibrated=bool(calibrated),
                       rate_provenance=rate_provenance,
                       source_sku=source_sku,
                       anchor_sku=anchor_sku)
    calibration = _merge_nested(calibration, caller_cal)
    if (caller_cal.get("calibrated") is not None
            and "rate_provenance" not in caller_cal):
        calibration["rate_provenance"] = (
            "silicon_calibrated" if bool(caller_cal["calibrated"])
            else "declared_nominal")
    if "anchor_sku" not in caller_cal and "source_sku" in caller_cal:
        calibration["anchor_sku"] = str(caller_cal["source_sku"])
    data["source"] = _merge_mapping(
        profile_source, {**caller_source, "calibration": calibration})
    data["assumptions"] = {
        **profile["assumptions"], **raw.get("assumptions", {}), **_ASSUMPTIONS}
    for field_name in _CONVENIENCE_FIELDS:
        if "links" not in raw or field_name not in raw:
            data.pop(field_name, None)
    return data


def normalize_ascend_machine(raw):
    """Public adapter entry point for an Ascend machine declaration."""
    return materialize(raw)
