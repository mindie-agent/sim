"""Ascend capability input adapter.

[Role] Materialize SKU metadata into neutral devices, buffers and links.
[Invariants] The simulator sees explicit rates. Nominal vector issue and launch
  assumptions are exposed and can be overridden; they are never measurements.
  Machine transfer declarations merge by channel over the registered profile;
  explicit machine links replace generated links. Transfer-channel ``*_bw_gbps``
  values retain the historical GB/s meaning. The convenience
  ``inter_node_bw_gbps``/``intra_node_bw_gbps`` fields are decimal Gb/s.
[Timeouts & Locks] Bounded local metadata reads; no device context.
[Side-Effect Policy] No external writes or probes.
"""
from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
from collections.abc import Mapping

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.device_input import machine_device_defaults, merge_device_inputs
from .instruction_presets import expand_instruction_presets


_LINK_FIELDS = ("transaction_bytes", "alignment_bytes", "protocol_overhead_bytes",
                "latency_us", "semantics", "switch_hops", "switch_hop_latency_us",
                "switch_capacity_bytes_per_ms")
_BW_FIELDS = ("effective_uni_bw_gbps", "calibrated_uni_bw_gbps",
              "derived_uni_bw_gbps", "uni_peak_bw_gbps")
_LINK_SEMANTICS = {"shared", "per_port", "per_pair", "direct", "fully_connected",
                   "mesh", "switched"}
_CONVENIENCE = {
    "d2d_inter_node": ("inter_node_bw_bytes_per_s", "inter_node_bw_gbps"),
    "d2d_intra_node": ("intra_node_bw_bytes_per_s", "intra_node_bw_gbps"),
}
_ASSUMPTIONS = {
    "transfer_bandwidth_units": (
        "inter_node_bw_gbps/intra_node_bw_gbps are decimal Gb/s and divided by 8; "
        "inter_node_bw_bytes_per_s/intra_node_bw_bytes_per_s are B/s and win over "
        "same-layer Gb/s; transfer_channels.*_bw_gbps and Link.capacity_gbps keep "
        "the historical GB/s meaning"
    ),
    "transfer_bandwidth_precedence": (
        "explicit machine.links > explicit transfer_channels channel bandwidth > "
        "convenience bytes_per_s > same-layer Gb/s > profile effective; within a "
        "channel explicit effective > explicit calibrated > explicit derived > "
        "explicit uni_peak, and null preserves unknown/disabled semantics"
    ),
    "transfer_channel_scope": (
        "Only d2d_intra_package, d2d_intra_node and d2d_inter_node generate "
        "device graph links. Other registered channels remain metadata; they "
        "do not override host/offload services or the P/D connector. Generated "
        "links retain per_port capacity unless semantics is explicitly declared."
    ),
}


@lru_cache(maxsize=1)
def profiles():
    with Path(__file__).with_name("profiles").joinpath("ascend.json").open() as stream:
        return json.load(stream)


def _number(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        raise ModelingError(f"transfer_bandwidth_invalid:{name}:{value}")
    return float(value)


def _channel_entries(raw):
    declaration = raw.get("transfer_channels")
    if declaration is None:
        return ()
    if isinstance(declaration, Mapping):
        entries = list(declaration.items())
    elif isinstance(declaration, list):
        entries, seen = [], set()
        for item in declaration:
            if not isinstance(item, Mapping) or "channel" not in item:
                raise ModelingError("transfer_channel_record_requires_channel")
            name = item["channel"]
            if not isinstance(name, str):
                raise ModelingError(f"transfer_channel_name_invalid:{name}")
            if name in seen:
                raise ModelingError(f"transfer_channel_duplicate:{name}")
            seen.add(name)
            entries.append((name, {key: value for key, value in item.items()
                                   if key != "channel"}))
    else:
        raise ModelingError("transfer_channels_requires_mapping_or_list")
    for name, patch in entries:
        if not isinstance(name, str):
            raise ModelingError(f"transfer_channel_name_invalid:{name}")
        if not isinstance(patch, Mapping):
            raise ModelingError(f"transfer_channel_record_invalid:{name}")
    return entries


def _merge_channels(raw, profile):
    base = profile["transfer_channels"]
    allowed = {"channel", *_LINK_FIELDS}
    for channel in base.values():
        allowed.update(channel)
    patches, explicit = [], set()
    for name, declaration in _channel_entries(raw):
        if name not in base:
            raise ModelingError(f"transfer_channel_unknown:{name}")
        patch = dict(declaration)
        if "channel" in patch and patch["channel"] != name:
            raise ModelingError(f"transfer_channel_name_mismatch:{name}")
        unknown = set(patch) - allowed
        if unknown:
            field = sorted(unknown)[0]
            raise ModelingError(f"transfer_channel_field_unknown:{name}:{field}")
        for field in ("channel_semantics", "semantics"):
            if field in patch and patch[field] not in _LINK_SEMANTICS:
                raise ModelingError(
                    f"transfer_channel_semantics_unsupported:{name}:{patch[field]}")
        if set(patch) & set(_BW_FIELDS):
            explicit.add(name)
        patches.append((name, patch))

    channels = deepcopy(base)
    # Per-port service includes independent full-duplex transmit and receive
    # ports. Optional switch-plane supply is a separate shared constraint.
    for declaration in channels.values():
        declaration["semantics"] = "per_port"
    for name, patch in patches:
        channels[name] = _merge_mapping(channels[name], patch)
        if "semantics" in patch or "channel_semantics" in patch:
            channels[name]["semantics"] = patch.get("semantics", patch.get("channel_semantics"))
        if name in explicit:
            if "effective_uni_bw_gbps" in patch:
                value, provenance = patch["effective_uni_bw_gbps"], "explicit_effective"
            else:
                value, provenance = None, "explicit_unknown"
                for field, label in (
                        ("calibrated_uni_bw_gbps", "explicit_calibrated"),
                        ("derived_uni_bw_gbps", "explicit_derived"),
                        ("uni_peak_bw_gbps", "explicit_peak")):
                    if patch.get(field) is not None:
                        value, provenance = patch[field], label
                        break
            if value is not None:
                value = _number(f"{name}:{provenance}", value)
            channels[name]["effective_uni_bw_gbps"] = value
            channels[name]["bandwidth_provenance"] = provenance
    return channels, explicit


def _merge_mapping(base, patch):
    result = deepcopy(base)
    for key, value in patch.items():
        result[key] = (_merge_mapping(result[key], value)
                       if isinstance(result.get(key), Mapping) and isinstance(value, Mapping)
                       else deepcopy(value))
    return result


def _apply_convenience(raw, channels, explicit):
    for channel, (bytes_field, gbps_field) in _CONVENIENCE.items():
        if channel in explicit:
            continue
        if raw.get(bytes_field) is not None:
            value = _number(bytes_field, raw[bytes_field])
            channel_value, provenance = value / 1e9, "explicit_bytes_per_s"
        elif raw.get(gbps_field) is not None:
            value = _number(gbps_field, raw[gbps_field])
            channel_value, provenance = value / 8, "explicit_gbps"
        else:
            continue
        channels[channel] = {**channels[channel],
                             "effective_uni_bw_gbps": channel_value,
                             "bandwidth_provenance": provenance}


def _generated_links(channels, count, per_package, per_node):
    links = []
    for channel, size in (("d2d_intra_package", per_package),
                          ("d2d_intra_node", per_node),
                          ("d2d_inter_node", count)):
        if size <= 1 or channel == "d2d_inter_node" and count <= per_node:
            continue
        declaration = channels[channel]
        capacity = declaration.get("effective_uni_bw_gbps")
        if not capacity:
            continue
        value = _number(f"{channel}:effective_uni_bw_gbps", capacity)
        if not value:
            continue
        semantics = declaration["semantics"]
        latency = declaration.get("latency_us", declaration.get("alpha_us", 0.0))
        for first in range(0, count, size):
            members = list(range(first, min(first + size, count)))
            if len(members) <= 1:
                continue
            link = {"name": f"{channel}:{first}", "members": members,
                    "capacity_bytes_per_ms": value * 1e6,
                    "latency_us": latency, "channel": channel,
                    "semantics": semantics}
            for field in ("transaction_bytes", "alignment_bytes",
                          "protocol_overhead_bytes", "switch_hops",
                          "switch_hop_latency_us", "switch_capacity_bytes_per_ms"):
                if declaration.get(field) is not None:
                    link[field] = declaration[field]
            links.append(link)
    return links


def materialize(raw):
    if raw.get("sku_profile"):
        raise ModelingError("sku_profile_unsupported:declare_device_rates_peaks_and_buffers")
    if raw["sku"] not in profiles():
        raise ModelingError(f"sku_not_registered:{raw['sku']}:declare_device_geometry")
    profile = profiles()[raw["sku"]]
    machine_defaults = machine_device_defaults(raw)
    device = merge_device_inputs(profile["device"], machine_defaults, raw.get("device", {}))
    channels, explicit = _merge_channels(raw, profile)
    _apply_convenience(raw, channels, explicit)
    data = {**raw, "device": expand_instruction_presets(device, profile, raw["sku"]),
            "transfer_channels": channels,
            "architecture": profile["architecture"]}
    for field in ("devices", "dies"):
        if field in raw:
            # Each record expands after inheritance, including an explicit []
            # that disables a preset. An expanded default would restore it.
            data["device"] = device
            data[field] = [expand_instruction_presets(
                merge_device_inputs(device, record), profile, raw["sku"])
                for record in raw[field]]
    count = int(raw.get("num_devices", 1))
    per_node = int(raw.get("ranks_per_node", count))
    per_package = int(raw.get("dies_per_package", profile["dies_per_package"]))
    data.update(ranks_per_node=per_node, dies_per_package=per_package)
    if "links" not in raw:
        data["links"] = _generated_links(channels, count, per_package, per_node)
    data["assumptions"] = {**profile["assumptions"],
                           **raw.get("assumptions", {}), **_ASSUMPTIONS}
    return data
