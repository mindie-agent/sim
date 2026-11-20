"""Ascend capability input adapter.

[Role] Materialize SKU metadata into neutral devices, buffers and links.
[Invariants] The simulator sees explicit rates. Nominal vector issue and launch
  assumptions are exposed and can be overridden; they are never measurements.
  Machine transfer declarations merge by channel over the registered profile;
  explicit machine links replace generated links. Transfer-channel ``*_bw_gbps``
  values retain the historical GB/s meaning. The convenience
  ``inter_node_bw_gbps``/``intra_node_bw_gbps`` fields are decimal Gb/s unless a
  ``topology_preset`` is active, in which case they are GB/s like the preset.
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

from mindie_sim.schema.common import ModelingError
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
)
from .instruction_presets import expand_instruction_presets

# Named topology preset constants (preserving public preset contract):
A2_STD_RANKS_PER_NODE = 8
A2_INTRA_PEAK_GBPS = 196.0
A2_INTRA_UTIL = 0.70
A2_ROCE_PEAK_GBPS = 25.0
A2_ROCE_UTIL = 0.80
A2_INTRA_LATENCY_US = 2.0
A2_INTER_LATENCY_US = 10.0

A3_DIES_PER_PACKAGE = 2
A3_NODE_RANKS_COMPACT = 8
A3_NODE_RANKS_FULL = 16
A3_SUPERPOD_RANKS = 384
A3_INTRA_PEAK_GBPS = 392.0
A3_INTRA_UTIL = 0.80
A3_SUPERPOD_UTIL = 0.70
A3_ROCE_PEAK_GBPS = 50.0
A3_ROCE_UTIL = 0.80
A3_INTRA_LATENCY_US = 1.5
A3_INTER_NODE_LATENCY_US = 5.0
A3_SUPERPOD_LATENCY_US = 3.5
A3_INTER_POD_LATENCY_US = 10.0

A5_POD_RANKS_PER_NODE = 64
A5_POD_SUPERPOD_RANKS = 8192
A5_POD_INTRA_BW_GBPS = 800.0
A5_POD_INTRA_UTIL = 0.85
A5_POD_INTER_POD_BW_GBPS = 100.0
A5_POD_INTER_UTIL = 0.80
A5_POD_INTER_LATENCY_US = 4.5

A5_SERVER_RANKS_PER_NODE = 8
A5_SERVER_SUBGROUP_BW_GBPS = 560.0
A5_SERVER_NODE_BW_GBPS = 640.0
A5_SERVER_INTRA_UTIL = 0.85
A5_SERVER_INTER_BW_GBPS = 200.0
A5_SERVER_INTER_UTIL = 0.80

A5_PCIE_RANKS_PER_NODE = 16
A5_PCIE_QUAD_BW_GBPS = 288.0
A5_PCIE_OCTA_BW_GBPS = 128.0
A5_PCIE_NODE_BW_GBPS = 116.0
A5_PCIE_INTRA_LATENCY_US = 2.0
A5_PCIE_INTER_LATENCY_US = 8.0

_PRESERVED_TOPOLOGY_FIELDS = (
    "intra_bw_4p_gbps", "intra_bw_8p_gbps", "superpod_size", "max_dies",
    "superpod_bw_gbps", "inter_pod_bw_gbps", "superpod_latency_us",
    "inter_pod_latency_us", "topology_preset",
)
_TOPOLOGY_FIELDS = _CONVENIENCE_FIELDS | set(_PRESERVED_TOPOLOGY_FIELDS)
_ASSUMPTIONS = {
    "transfer_bandwidth_units": (
        "inter_node_bw_gbps/intra_node_bw_gbps are decimal Gb/s and divided by 8 "
        "unless a topology_preset is selected; preset bandwidths, "
        "intra_package_bw_gbps, intra_bw_4p/8p_gbps, superpod_bw_gbps and "
        "inter_pod_bw_gbps are GB/s; inter_node_bw_bytes_per_s/ "
        "intra_node_bw_bytes_per_s are B/s and win over same-layer Gb/s; "
        "transfer_channels.*_bw_gbps and Link.capacity_gbps keep the historical "
        "GB/s meaning"
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
        "intra_node_bw_gbps": A2_INTRA_PEAK_GBPS,
        "intra_node_bw_util": A2_INTRA_UTIL,
        "inter_node_bw_gbps": A2_ROCE_PEAK_GBPS,
        "inter_node_bw_util": A2_ROCE_UTIL,
        "intra_node_latency_us": A2_INTRA_LATENCY_US,
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
        "inter_node_bw_gbps": A3_ROCE_PEAK_GBPS,
        "inter_node_bw_util": A3_ROCE_UTIL,
        "intra_node_latency_us": A5_PCIE_INTRA_LATENCY_US,
        "inter_node_latency_us": A5_PCIE_INTER_LATENCY_US,
    },
}


@lru_cache(maxsize=1)
def profiles():
    with Path(__file__).with_name("profiles").joinpath("ascend.json").open(encoding="utf-8") as stream:
        return json.load(stream)


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


def materialize(raw):
    if raw.get("sku_profile"):
        raise ModelingError("sku_profile_unsupported:declare_device_rates_peaks_and_buffers")
    if raw["sku"] not in profiles():
        raise ModelingError(f"sku_not_registered:{raw['sku']}:declare_device_geometry")
    merged, preset = _preset(raw)
    preset_active = bool(preset)
    profile = profiles()[raw["sku"]]
    machine_defaults = machine_device_defaults(merged)
    device = merge_device_inputs(profile["device"], machine_defaults, merged.get("device", {}))
    channels, explicit, declared = _merge_channels(merged, profile)
    bw_caller = _caller_bandwidth_channels(raw)
    _apply_convenience(raw, preset, channels, explicit, declared, preset_active)
    data = {**merged, "device": expand_instruction_presets(device, profile, raw["sku"]),
            "transfer_channels": channels,
            "architecture": profile["architecture"]}
    for field_name in ("devices", "dies"):
        if field_name in merged:
            data["device"] = device
            data[field_name] = [expand_instruction_presets(
                merge_device_inputs(device, record), profile, raw["sku"])
                for record in merged[field_name]]
    count = int(merged.get("num_devices", len(merged.get(
        "devices", merged.get("dies", ()))) or 1))
    per_node = int(merged.get("ranks_per_node", count))
    per_package = int(merged.get("dies_per_package", profile["dies_per_package"]))
    data.update(ranks_per_node=per_node, dies_per_package=per_package)
    if "collective_protocols" not in raw:
        domain = int(merged.get("superpod_size", DIRECT_MESH_RANKS))
        if "links" in raw or count > domain:
            data.pop("collective_protocols", None)
        elif (raw["sku"].startswith("Ascend950") and not preset
              and per_node <= DIRECT_MESH_RANKS and count <= per_node):
            data["collective_protocols"] = {
                "alltoall": "concurrent", "alltoallv": "concurrent"}
    if "links" not in raw:
        data["links"] = _generated_links(
            merged, raw, preset, channels, count, per_package, per_node,
            explicit, bw_caller, declared)
        for tier, key in (("d2d_intra_4p", "intra_bw_4p_gbps"),
                          ("d2d_intra_8p", "intra_bw_8p_gbps")):
            if tier in explicit and raw.get(key) is None:
                data[key] = channels[tier].get("uni_peak_bw_gbps")
    profile_source = deepcopy(profile.get("source") or {})
    caller_source = dict(raw.get("source") or {})
    caller_cal = dict(caller_source.get("calibration") or {})
    profile_hierarchy = (((profile["device"].get("physical_mechanisms") or {})
                          .get("memory_hierarchy") or {})
                         .get("provenance") or {})
    calibration = dict(profile_source.get("calibration") or {})
    calibrated = calibration.get("calibrated")
    if calibrated is None:
        calibrated = bool(profile_hierarchy.get("calibrated", False))
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
