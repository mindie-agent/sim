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
  unity bandwidth utilization emits exactly the historical link key set.
[Timeouts & Locks] Bounded local metadata reads; no device context.
[Side-Effect Policy] No external writes or probes.
"""
from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path
from collections.abc import Mapping

from mindie_sim.schema.common import ModelingError, _positive_float
from mindie_sim.schema.device_input import machine_device_defaults, merge_device_inputs
from .instruction_presets import expand_instruction_presets


_LINK_FIELDS = ("transaction_bytes", "alignment_bytes", "protocol_overhead_bytes",
                "latency_us", "semantics", "switch_hops", "switch_hop_latency_us",
                "switch_capacity_bytes_per_ms")
_BW_FIELDS = ("effective_uni_bw_gbps", "calibrated_uni_bw_gbps",
              "derived_uni_bw_gbps", "uni_peak_bw_gbps")
_UTIL_FIELDS = ("bw_utilization", "bw_ratio", "utilization")
_GENERATED_CHANNELS = {"d2d_intra_4p", "d2d_intra_8p", "d2d_inter_pod"}
_LINK_SEMANTICS = {"shared", "per_port", "per_pair", "direct", "fully_connected",
                   "mesh", "switched"}
_CONVENIENCE = {
    "inter_node": ("d2d_inter_node", "inter_node_bw_bytes_per_s",
                   "inter_node_bw_gbps", "inter_node_bw_util",
                   "inter_node_latency_us"),
    "intra_node": ("d2d_intra_node", "intra_node_bw_bytes_per_s",
                   "intra_node_bw_gbps", "intra_node_bw_util",
                   "intra_node_latency_us"),
    "intra_package": ("d2d_intra_package", "intra_package_bw_bytes_per_s",
                      "intra_package_bw_gbps", "intra_package_bw_util",
                      "intra_package_latency_us"),
}
_CONVENIENCE_FIELDS = {
    "inter_node_bw_gbps", "intra_node_bw_gbps", "intra_package_bw_gbps",
    "inter_node_bw_util", "intra_node_bw_util", "intra_package_bw_util",
    "inter_node_latency_us", "intra_node_latency_us", "intra_package_latency_us",
    "inter_node_bw_bytes_per_s", "intra_node_bw_bytes_per_s",
    "intra_package_bw_bytes_per_s",
}
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
        "explicit uni_peak, and null preserves unknown/disabled semantics"
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
        "ranks_per_node": 8, "dies_per_package": 1,
        "intra_node_bw_gbps": 196.0, "intra_node_bw_util": 0.70,
        "inter_node_bw_gbps": 25.0, "inter_node_bw_util": 0.80,
        "intra_node_latency_us": 2.0, "inter_node_latency_us": 10.0,
    },
    "a3_8p": {
        "ranks_per_node": 8, "dies_per_package": 2,
        "intra_bw_4p_gbps": 392.0,
        "intra_node_bw_gbps": 392.0, "intra_node_bw_util": 0.80,
        "inter_node_bw_gbps": 50.0, "inter_node_bw_util": 0.80,
        "intra_node_latency_us": 1.5, "inter_node_latency_us": 5.0,
    },
    "a3_16p": {
        "ranks_per_node": 16, "dies_per_package": 2,
        "intra_bw_4p_gbps": 392.0, "intra_bw_8p_gbps": 392.0,
        "intra_node_bw_gbps": 392.0, "intra_node_bw_util": 0.80,
        "inter_node_bw_gbps": 50.0, "inter_node_bw_util": 0.80,
        "intra_node_latency_us": 1.5, "inter_node_latency_us": 5.0,
    },
    "a3_superpod": {
        "ranks_per_node": 16, "dies_per_package": 2, "superpod_size": 384,
        "intra_node_bw_gbps": 392.0, "intra_node_bw_util": 0.80,
        "superpod_bw_gbps": 392.0, "inter_pod_bw_gbps": 50.0,
        "inter_node_bw_util": 0.70,
        "intra_node_latency_us": 1.5, "inter_node_latency_us": 3.5,
        "inter_pod_latency_us": 10.0,
    },
    "a5_pod": {
        "ranks_per_node": 64, "dies_per_package": 2, "superpod_size": 8192,
        "intra_node_bw_gbps": 800.0, "intra_node_bw_util": 0.85,
        "superpod_bw_gbps": 800.0, "inter_pod_bw_gbps": 100.0,
        "inter_node_bw_util": 0.80,
        "intra_node_latency_us": 1.5, "inter_node_latency_us": 4.5,
        "inter_pod_latency_us": 10.0,
    },
    "a5_server": {
        "ranks_per_node": 8, "dies_per_package": 2,
        "intra_bw_4p_gbps": 560.0, "intra_bw_8p_gbps": 640.0,
        "intra_node_bw_gbps": 640.0, "intra_node_bw_util": 0.85,
        "inter_node_bw_gbps": 200.0, "inter_node_bw_util": 0.80,
        "intra_node_latency_us": 1.5, "inter_node_latency_us": 5.0,
    },
    "a5_pcie_16p": {
        "ranks_per_node": 16, "dies_per_package": 2,
        "intra_bw_4p_gbps": 288.0, "intra_bw_8p_gbps": 128.0,
        "intra_node_bw_gbps": 116.0, "intra_node_bw_util": 0.85,
        "inter_node_bw_gbps": 50.0, "inter_node_bw_util": 0.80,
        "intra_node_latency_us": 2.0, "inter_node_latency_us": 8.0,
    },
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
    allowed = {"channel", *_LINK_FIELDS, *_UTIL_FIELDS,
               "peak_capacity_bytes_per_ms", "channel_semantics"}
    for channel in base.values():
        allowed.update(channel)
    patches, explicit, declared = [], set(), {}
    for name, declaration in _channel_entries(raw):
        if name not in base:
            if name in _GENERATED_CHANNELS:
                # A materialized or resized machine recreates these structural links.
                continue
            raise ModelingError(f"transfer_channel_unknown:{name}")
        patch = dict(declaration)
        if "channel" in patch and patch["channel"] != name:
            raise ModelingError(f"transfer_channel_name_mismatch:{name}")
        unknown = set(patch) - allowed
        if unknown:
            field = sorted(unknown)[0]
            raise ModelingError(f"transfer_channel_field_unknown:{name}:{field}")
        if "links" not in raw:
            for field in ("channel_semantics", "semantics"):
                if field in patch and patch[field] not in _LINK_SEMANTICS:
                    raise ModelingError(
                        f"transfer_channel_semantics_unsupported:{name}:{patch[field]}")
        if set(patch) & set(_BW_FIELDS):
            explicit.add(name)
        declared[name] = set(patch)
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
    return channels, explicit, declared


def _merge_mapping(base, patch):
    result = deepcopy(base)
    for key, value in patch.items():
        result[key] = (_merge_mapping(result[key], value)
                       if isinstance(result.get(key), Mapping) and isinstance(value, Mapping)
                       else deepcopy(value))
    return result


def _channel_util(declaration):
    for alias in _UTIL_FIELDS:
        if alias in declaration and declaration[alias] is not None:
            util = _positive_float(declaration[alias],
                                   "transfer_channel_bw_utilization_invalid",
                                   allow_zero=False)
            if util > 1.0:
                raise ModelingError("transfer_channel_bw_utilization_invalid")
            return util
    return 1.0


def _finalize_channel_utilization(channels):
    for channel, declaration in channels.items():
        util = _channel_util(declaration)
        if math.isclose(util, 1.0):
            declaration.pop("bw_utilization", None)
            declaration.pop("bw_ratio", None)
            declaration.pop("utilization", None)
            declaration.pop("peak_capacity_bytes_per_ms", None)
            continue
        peak_bytes = declaration.get("peak_capacity_bytes_per_ms")
        if peak_bytes is not None:
            peak = _positive_float(
                peak_bytes, f"transfer_channel_requires_positive_gbps:{channel}") / 1e6
        else:
            if declaration.get("bandwidth_provenance") == "explicit_effective":
                # A channel patch's effective value is the peak for its own
                # utilization declaration; the inherited profile peak is stale.
                peak = declaration.get("effective_uni_bw_gbps")
            else:
                peak = declaration.get("uni_peak_bw_gbps")
            if peak is None:
                peak = declaration.get("effective_uni_bw_gbps")
            if peak is None:
                raise ModelingError(f"transfer_channel_requires_positive_gbps:{channel}")
            peak = _positive_float(peak, f"transfer_channel_requires_positive_gbps:{channel}")
        declaration["bw_utilization"] = util
        declaration["uni_peak_bw_gbps"] = peak
        declaration["peak_capacity_bytes_per_ms"] = peak * 1e6
        declaration["effective_uni_bw_gbps"] = peak * util


def _topology_bandwidth(raw, kind, preset_active):
    channel, bytes_field, gbps_field, util_field, latency_field = _CONVENIENCE[kind]
    peak = None
    if raw.get(bytes_field) is not None:
        peak = _number(bytes_field, raw[bytes_field]) / 1e9
    elif raw.get(gbps_field) is not None:
        value = _number(gbps_field, raw[gbps_field])
        # The historical convenience fields are decimal Gb/s. Preset values
        # and the newer package field are declared GB/s.
        peak = value if (preset_active or kind == "intra_package") else value / 8
    util = None
    if raw.get(util_field) is not None:
        util = _positive_float(raw[util_field], f"{util_field}_invalid",
                               allow_zero=False)
        if util > 1.0:
            raise ModelingError("link_bw_utilization_exceeds_unity")
    latency = None
    if raw.get(latency_field) is not None:
        latency = _positive_float(raw[latency_field], f"{latency_field}_invalid",
                                  allow_zero=True)
    return channel, peak, util, latency


def _apply_convenience(raw, channels, explicit, declared, preset_active):
    for kind in _CONVENIENCE:
        channel, peak, util, latency = _topology_bandwidth(raw, kind, preset_active)
        declaration = channels[channel]
        fields = declared.get(channel, set())
        if peak is not None and channel not in explicit:
            declaration["uni_peak_bw_gbps"] = peak
            declaration["effective_uni_bw_gbps"] = peak
            declaration["bandwidth_provenance"] = (
                "explicit_bytes_per_s" if raw.get(_CONVENIENCE[kind][1]) is not None
                else "explicit_gbps")
        if util is not None and not (set(_UTIL_FIELDS) & fields):
            declaration["bw_utilization"] = util
            if "effective_uni_bw_gbps" in fields:
                # An explicit effective channel value is the peak for this
                # utilization override; the profile's older peak is stale.
                peak_value = declaration.get("effective_uni_bw_gbps")
            else:
                peak_value = (declaration.get("uni_peak_bw_gbps")
                              or declaration.get("effective_uni_bw_gbps"))
            if peak_value is not None:
                declaration["uni_peak_bw_gbps"] = float(peak_value)
                declaration["peak_capacity_bytes_per_ms"] = float(peak_value) * 1e6
        if latency is not None and "latency_us" not in fields and "alpha_us" not in fields:
            declaration["latency_us"] = latency
    _finalize_channel_utilization(channels)


def _tier_bandwidth(channels, channel):
    declaration = channels[channel]
    util = _channel_util(declaration)
    capacity = declaration.get("effective_uni_bw_gbps")
    if not capacity:
        return None
    if math.isclose(util, 1.0):
        # Explicit effective channel values win over an inherited profile peak.
        return float(capacity), util, float(capacity)
    peak = declaration.get("uni_peak_bw_gbps")
    if peak is None and declaration.get("peak_capacity_bytes_per_ms") is not None:
        peak = float(declaration["peak_capacity_bytes_per_ms"]) / 1e6
    if peak is None:
        peak = float(capacity) / util
    return float(peak), util, float(capacity)


def _link_record(name, members, channel, declaration, peak, util, latency):
    capacity = float(peak) * 1e6 * float(util)
    link = {"name": name, "members": list(members),
            "capacity_bytes_per_ms": capacity,
            "latency_us": latency, "channel": channel,
            "semantics": declaration["semantics"]}
    if not math.isclose(util, 1.0):
        link["bw_utilization"] = util
        link["peak_capacity_bytes_per_ms"] = float(peak) * 1e6
    for field in ("transaction_bytes", "alignment_bytes",
                  "protocol_overhead_bytes", "switch_hops",
                  "switch_hop_latency_us", "switch_capacity_bytes_per_ms"):
        if declaration.get(field) is not None:
            link[field] = declaration[field]
    return link


def _group_links(channels, channel, groups, peak=None, util=None, latency=None):
    declaration = channels[channel]
    if peak is None:
        tier = _tier_bandwidth(channels, channel)
        if tier is None:
            return []
        peak, util, _ = tier
    if util is None:
        util = _channel_util(declaration)
    if latency is None:
        latency = declaration.get("latency_us", declaration.get("alpha_us", 0.0))
    links = []
    for first, members in groups:
        if len(members) <= 1:
            continue
        links.append(_link_record(f"{channel}:{first}", members, channel,
                                  declaration, peak, util, latency))
    return links


def _generated_links(raw, channels, count, per_package, per_node):
    links = []
    if per_package > 1 and count > 1:
        groups = [(first, list(range(first, min(first + per_package, count))))
                  for first in range(0, count, per_package)]
        links.extend(_group_links(channels, "d2d_intra_package", groups))

    four = raw.get("intra_bw_4p_gbps")
    eight = raw.get("intra_bw_8p_gbps")
    four = _number("intra_bw_4p_gbps", four) if four is not None else None
    eight = _number("intra_bw_8p_gbps", eight) if eight is not None else None
    node_tier = _tier_bandwidth(channels, "d2d_intra_node")
    if per_node > 1 and node_tier is not None:
        node_peak, node_util, _ = node_tier
        node_decl = channels["d2d_intra_node"]
        node_latency = node_decl.get("latency_us", node_decl.get("alpha_us", 0.0))
        if four is not None and per_node > 4:
            groups = [(first + offset, list(range(first + offset,
                                                  min(first + offset + 4, first + per_node, count))))
                      for first in range(0, count, per_node)
                      for offset in range(0, per_node, 4)]
            channels["d2d_intra_4p"] = {**channels["d2d_intra_node"]}
            links.extend(_group_links(channels, "d2d_intra_4p", groups,
                                      peak=four, util=node_util,
                                      latency=node_latency))
        if eight is not None and per_node > 8:
            groups = [(first + offset, list(range(first + offset,
                                                  min(first + offset + 8, first + per_node, count))))
                      for first in range(0, count, per_node)
                      for offset in range(0, per_node, 8)]
            # The 8P tier is a distinct channel label so routing can prefer it
            # over the full node while sharing the same physical declaration.
            eight_channel = "d2d_intra_8p"
            channel_decl = {**channels["d2d_intra_node"]}
            channels[eight_channel] = channel_decl
            links.extend(_group_links(channels, eight_channel, groups,
                                      peak=eight, util=node_util,
                                      latency=node_latency))
        node_peak_eff = node_peak
        if per_node <= 4 and four is not None:
            node_peak_eff = four
        elif per_node <= 8 and eight is not None:
            node_peak_eff = eight
        groups = [(first, list(range(first, min(first + per_node, count))))
                  for first in range(0, count, per_node)]
        links.extend(_group_links(channels, "d2d_intra_node", groups,
                                  peak=node_peak_eff,
                                  util=node_util,
                                  latency=node_latency))

    superpod_size = raw.get("superpod_size", raw.get("max_dies"))
    if superpod_size is not None:
        if isinstance(superpod_size, bool):
            raise ModelingError("ascend_superpod_size_invalid")
        superpod_size = int(superpod_size)
        if superpod_size <= 0:
            raise ModelingError("ascend_superpod_size_invalid")
        if superpod_size < per_node:
            raise ModelingError("ascend_superpod_size_smaller_than_node")
    inter_tier = _tier_bandwidth(channels, "d2d_inter_node")
    if per_node > 0 and count > per_node and inter_tier is not None:
        inter_peak, util, _ = inter_tier
        inter_decl = channels["d2d_inter_node"]
        latency = inter_decl.get("latency_us", inter_decl.get("alpha_us", 0.0))
        if superpod_size is not None:
            same_peak = raw.get("superpod_bw_gbps")
            same_peak = (_number("superpod_bw_gbps", same_peak)
                         if same_peak is not None else inter_peak)
            same_latency = raw.get("superpod_latency_us")
            same_latency = (_positive_float(same_latency,
                                            "superpod_latency_us_invalid",
                                            allow_zero=True)
                            if same_latency is not None else latency)
            groups = [(first, list(range(first, min(first + superpod_size, count))))
                      for first in range(0, count, superpod_size)]
            links.extend(_group_links(channels, "d2d_inter_node", groups,
                                      peak=same_peak, util=util,
                                      latency=same_latency))
            if superpod_size < count:
                pod_peak = raw.get("inter_pod_bw_gbps")
                pod_peak = (_number("inter_pod_bw_gbps", pod_peak)
                            if pod_peak is not None else inter_peak)
                pod_latency = raw.get("inter_pod_latency_us")
                pod_latency = (_positive_float(pod_latency,
                                               "inter_pod_latency_us_invalid",
                                               allow_zero=True)
                               if pod_latency is not None else latency)
                channels["d2d_inter_pod"] = {**inter_decl}
                links.extend(_group_links(channels, "d2d_inter_pod",
                                          [(0, list(range(count)))],
                                          peak=pod_peak, util=util,
                                          latency=pod_latency))
        else:
            links.extend(_group_links(channels, "d2d_inter_node",
                                      [(0, list(range(count)))],
                                      peak=inter_peak, util=util,
                                      latency=latency))
    return links


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
    _apply_convenience(merged, channels, explicit, declared, preset_active)
    data = {**merged, "device": expand_instruction_presets(device, profile, raw["sku"]),
            "transfer_channels": channels,
            "architecture": profile["architecture"]}
    for field in ("devices", "dies"):
        if field in merged:
            # Each record expands after inheritance, including an explicit []
            # that disables a preset. An expanded default would restore it.
            data["device"] = device
            data[field] = [expand_instruction_presets(
                merge_device_inputs(device, record), profile, raw["sku"])
                for record in merged[field]]
    count = int(merged.get("num_devices", 1))
    per_node = int(merged.get("ranks_per_node", count))
    per_package = int(merged.get("dies_per_package", profile["dies_per_package"]))
    data.update(ranks_per_node=per_node, dies_per_package=per_package)
    if "links" not in raw:
        data["links"] = _generated_links(merged, channels, count, per_package, per_node)
    data["assumptions"] = {**profile["assumptions"],
                           **raw.get("assumptions", {}), **_ASSUMPTIONS}
    # The adapter has resolved convenience bandwidths into transfer_channels and
    # generated links. Dropping convenience fields when links were generated
    # keeps a neutral re-parse from applying a second unit conversion, while
    # preserving explicit convenience overrides on pre-materialized link lists.
    if "links" not in raw:
        for field in _CONVENIENCE_FIELDS:
            data.pop(field, None)
    return data


def normalize_ascend_machine(raw):
    """Public adapter entry point for an Ascend machine declaration."""
    return materialize(raw)
