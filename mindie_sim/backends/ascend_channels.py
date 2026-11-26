"""Transfer-channel merging and multi-tier link generation for Ascend SKUs.

[Role]
  Merge caller ``transfer_channels`` and convenience bandwidth overrides over a
  registered Ascend profile, resolve effective utilization once per channel, and
  emit intra-package, 4P/8P, intra-node, SuperPoD, and inter-pod ``Link`` dicts
  with explicit scope, rail/switch/root-complex, and piecewise protocol bands.

[Invariants]
  * Explicit caller channel rates resolve service capacity once without double
    derating by preset utilization.
  * Nodes wider than ``DIRECT_MESH_RANKS`` default ``d2d_intra_node`` to
    ``per_port`` unless the caller explicitly declares channel semantics.
  * Optional ``node_peak_bw_gbps``, ``solo_effective_uni_bw_gbps``, explicit
    ``scope``, and ``protocol_bands`` metadata are preserved across merges.

[Timeouts & Locks]
  Bounded in-memory dictionary merges and group generation; no I/O or locks.

[Side-Effect Policy]
  Mutates only fresh deep-copied channel dictionaries returned to ``materialize``.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

from mindie_sim.schema.common import ModelingError, _positive_float
from mindie_sim.schema.device_input import (
    placement_node_groups, placement_node_order_groups, placement_package_groups,
    resolve_device_placements, resolve_same_pod_tier,
)

DIRECT_MESH_RANKS = 8
A3_FULL_NODE_RANKS = 16
SUBGROUP_QUAD_DIE_RANKS = 4
SUBGROUP_OCTA_DIE_RANKS = 8
BITS_PER_BYTE = 8
GBPS_TO_BYTES_PER_MS = 1_000_000.0
BYTES_PER_S_TO_GBPS = 1_000_000_000.0
_COMPACT_A3_SKUS = frozenset({"Ascend910_9381", "Ascend910_9382"})


def default_ranks_per_node(sku: str) -> int:
    """Return canonical physical dies-per-node for an Ascend SKU."""
    name = str(sku or "").strip()
    if name in _COMPACT_A3_SKUS:
        return DIRECT_MESH_RANKS
    return A3_FULL_NODE_RANKS if name.startswith("Ascend910_93") else DIRECT_MESH_RANKS

_LINK_FIELDS = (
    "transaction_bytes", "alignment_bytes", "protocol_overhead_bytes",
    "latency_us", "semantics", "switch_hops", "switch_hop_latency_us",
    "switch_capacity_bytes_per_ms",
    "scope", "rail_id", "switch_id", "root_complex_id", "endpoint_ids",
    "protocol_bands",
)
_BW_FIELDS = ("effective_uni_bw_gbps", "calibrated_uni_bw_gbps",
              "derived_uni_bw_gbps", "uni_peak_bw_gbps")
_UTIL_FIELDS = ("bw_utilization", "bw_ratio", "utilization")
_EFFECTIVE_RATE_PROVENANCE = frozenset({
    "explicit_effective", "explicit_calibrated", "explicit_derived"})
_GENERATED_CHANNELS = {"d2d_intra_4p", "d2d_intra_8p", "d2d_inter_pod"}
_TIER_OVERRIDE_FIELDS = frozenset({
    *_BW_FIELDS, *_UTIL_FIELDS, "peak_capacity_bytes_per_ms",
    "latency_us", "alpha_us",
})
_TIER_SEEDS = {
    "d2d_intra_4p": "d2d_intra_node",
    "d2d_intra_8p": "d2d_intra_node",
    "d2d_inter_pod": "d2d_inter_node",
}
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
_PASSTHROUGH_LINK_FIELDS = (
    "transaction_bytes", "alignment_bytes", "protocol_overhead_bytes",
    "switch_hops", "switch_hop_latency_us", "switch_capacity_bytes_per_ms",
)


def _number(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        raise ModelingError(f"transfer_bandwidth_invalid:{name}:{value}")
    return float(value)


def _merge_mapping(base, patch):
    result = deepcopy(base)
    for key, value in patch.items():
        result[key] = (_merge_mapping(result[key], value)
                       if isinstance(result.get(key), Mapping)
                       and isinstance(value, Mapping)
                       else deepcopy(value))
    return result


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


def _clear_utilization(declaration):
    for alias in _UTIL_FIELDS:
        declaration.pop(alias, None)
    declaration.pop("peak_capacity_bytes_per_ms", None)


def _merge_channels(raw, profile):
    base = deepcopy(profile["transfer_channels"])
    allowed = {"channel", *_LINK_FIELDS, *_UTIL_FIELDS,
               "peak_capacity_bytes_per_ms", "channel_semantics",
               "node_peak_bw_gbps", "solo_effective_uni_bw_gbps",
               "fabric"}
    for channel in base.values():
        allowed.update(channel)
    patches, explicit, declared = [], set(), {}
    for name, declaration in _channel_entries(raw):
        patch = dict(declaration)
        if name not in base and name in _GENERATED_CHANNELS:
            if not (set(patch) & _TIER_OVERRIDE_FIELDS):
                continue
            base[name] = deepcopy(base[_TIER_SEEDS[name]])
            base[name]["channel"] = name
        if name not in base:
            raise ModelingError(f"transfer_channel_unknown:{name}")
        if name in _GENERATED_CHANNELS and patch.get("channel") == _TIER_SEEDS[name]:
            patch["channel"] = name
        if "channel" in patch and patch["channel"] != name:
            raise ModelingError(f"transfer_channel_name_mismatch:{name}")
        unknown = set(patch) - allowed
        if unknown:
            field_name = sorted(unknown)[0]
            raise ModelingError(f"transfer_channel_field_unknown:{name}:{field_name}")
        for field_name in ("channel_semantics", "semantics"):
            if patch.get(field_name) in {"unknown", "none"}:
                patch.pop(field_name)
            elif field_name in patch and patch[field_name] not in _LINK_SEMANTICS:
                raise ModelingError(
                    f"transfer_channel_semantics_unsupported:{name}:{patch[field_name]}")
        if (any(patch.get(field_name) is not None for field_name in _BW_FIELDS)
                or patch.get("peak_capacity_bytes_per_ms") is not None):
            explicit.add(name)
        declared[name] = set(patch)
        patches.append((name, patch))

    channels = deepcopy(base)
    for declaration in channels.values():
        profile_sem = (declaration.get("semantics")
                       or declaration.get("channel_semantics"))
        declaration["semantics"] = ("per_pair" if profile_sem == "per_pair"
                                    else "per_port")
    for name, patch in patches:
        channels[name] = _merge_mapping(channels[name], patch)
        if "semantics" in patch or "channel_semantics" in patch:
            channels[name]["semantics"] = patch.get("semantics", patch.get("channel_semantics"))
        if name in explicit:
            effective = patch.get("effective_uni_bw_gbps")
            nominal = patch.get("uni_peak_bw_gbps")
            coherent = (effective is not None and nominal is not None
                        and math.isclose(_channel_util(patch) * nominal,
                                         effective))
            if coherent:
                value, provenance = nominal, "explicit_peak"
            elif effective is not None:
                value, provenance = effective, "explicit_effective"
            else:
                value, provenance = None, "explicit_unknown"
                for field_name, label in (
                        ("calibrated_uni_bw_gbps", "explicit_calibrated"),
                        ("derived_uni_bw_gbps", "explicit_derived"),
                        ("uni_peak_bw_gbps", "explicit_peak")):
                    if patch.get(field_name) is not None:
                        value, provenance = patch[field_name], label
                        break
                if value is None and patch.get("peak_capacity_bytes_per_ms") is not None:
                    value = (_number(f"{name}:explicit_peak",
                                     patch["peak_capacity_bytes_per_ms"])
                             / GBPS_TO_BYTES_PER_MS)
                    provenance = "explicit_peak"
            if value is not None:
                value = _number(f"{name}:{provenance}", value)
                if provenance == "explicit_peak":
                    channels[name]["uni_peak_bw_gbps"] = value
                    channels[name]["peak_capacity_bytes_per_ms"] = value * GBPS_TO_BYTES_PER_MS
            channels[name]["effective_uni_bw_gbps"] = value
            channels[name]["bandwidth_provenance"] = provenance
    node_ranks = int(
        raw.get("physical_ranks_per_node")
        or raw.get("ranks_per_node")
        or profile.get("physical_ranks_per_node")
        or profile.get("ranks_per_node")
        or default_ranks_per_node(str(raw.get("sku", "")))
    )
    if node_ranks > DIRECT_MESH_RANKS:
        declared_semantics = declared.get("d2d_intra_node", set())
        if not ({"semantics", "channel_semantics"} & declared_semantics):
            channels["d2d_intra_node"].update(
                channel_semantics="per_port", semantics="per_port")
    return channels, explicit, declared


def _is_measured_effective(declaration):
    return (declaration.get("bandwidth_provenance")
            in _EFFECTIVE_RATE_PROVENANCE)


def _finalize_channel_utilization(channels, explicit=(), declared=None):
    declared = declared or {}
    for channel, declaration in channels.items():
        fields = declared.get(channel, set())
        util_declared = bool(set(_UTIL_FIELDS) & fields)
        if channel in explicit and _is_measured_effective(declaration) and not util_declared:
            effective = declaration.get("effective_uni_bw_gbps")
            if effective is None:
                raise ModelingError(
                    f"transfer_channel_requires_positive_gbps:{channel}")
            effective = _positive_float(
                effective, f"transfer_channel_requires_positive_gbps:{channel}")
            _clear_utilization(declaration)
            declaration["uni_peak_bw_gbps"] = effective
            declaration["peak_capacity_bytes_per_ms"] = effective * GBPS_TO_BYTES_PER_MS
            continue
        util = _channel_util(declaration)
        if math.isclose(util, 1.0):
            if (util_declared and channel not in explicit
                    and not _is_measured_effective(declaration)):
                peak_val = declaration.get("uni_peak_bw_gbps")
                if peak_val is not None:
                    declaration["effective_uni_bw_gbps"] = _positive_float(
                        peak_val, f"transfer_channel_requires_positive_gbps:{channel}")
            _clear_utilization(declaration)
            continue
        if _is_measured_effective(declaration):
            peak = declaration.get("effective_uni_bw_gbps")
        else:
            peak_bytes = declaration.get("peak_capacity_bytes_per_ms")
            if peak_bytes is not None:
                peak = (_positive_float(
                    peak_bytes, f"transfer_channel_requires_positive_gbps:{channel}")
                        / GBPS_TO_BYTES_PER_MS)
            else:
                peak = declaration.get("uni_peak_bw_gbps")
                if peak is None:
                    peak = declaration.get("effective_uni_bw_gbps")
        if peak is None:
            raise ModelingError(f"transfer_channel_requires_positive_gbps:{channel}")
        peak = _positive_float(peak, f"transfer_channel_requires_positive_gbps:{channel}")
        declaration["bw_utilization"] = util
        declaration["uni_peak_bw_gbps"] = peak
        declaration["peak_capacity_bytes_per_ms"] = peak * GBPS_TO_BYTES_PER_MS
        declaration["effective_uni_bw_gbps"] = peak * util


def _topology_bandwidth(raw, kind, preset_active):
    del preset_active
    channel, bytes_field, gbps_field, util_field, latency_field = _CONVENIENCE[kind]
    peak = None
    if raw.get(bytes_field) is not None:
        peak = _number(bytes_field, raw[bytes_field]) / BYTES_PER_S_TO_GBPS
    elif raw.get(gbps_field) is not None:
        peak = _number(gbps_field, raw[gbps_field])
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


def _convenience_values(raw, preset, kind, preset_active):
    _, raw_peak, raw_util, raw_latency = _topology_bandwidth(
        raw, kind, preset_active)
    _, preset_peak, preset_util, preset_latency = _topology_bandwidth(
        preset, kind, True)
    return (
        raw_peak if raw_peak is not None else preset_peak,
        raw_util if raw_util is not None else preset_util,
        raw_latency if raw_latency is not None else preset_latency,
    )


def _caller_bandwidth_channels(raw):
    bw_fields = {*_BW_FIELDS, "peak_capacity_bytes_per_ms"}
    channels = {name for name, patch in _channel_entries(raw)
                if set(patch) & bw_fields}
    for channel, *fields in _CONVENIENCE.values():
        if any(raw.get(field_name) is not None for field_name in fields[:2]):
            channels.add(channel)
    return channels


def _apply_convenience(raw, preset, channels, explicit, declared, preset_active):
    for kind in _CONVENIENCE:
        channel, bytes_field, gbps_field, _, _ = _CONVENIENCE[kind]
        peak, util, latency = _convenience_values(
            raw, preset, kind, preset_active)
        declaration = channels[channel]
        fields = declared.get(channel, set())
        if peak is not None and channel not in explicit:
            declaration["uni_peak_bw_gbps"] = peak
            declaration["effective_uni_bw_gbps"] = peak
            declaration["bandwidth_provenance"] = (
                "explicit_bytes_per_s" if raw.get(bytes_field) is not None
                else ("explicit_gbps" if raw.get(gbps_field) is not None
                      else "preset_gbps"))
        measured = channel in explicit and _is_measured_effective(declaration)
        if (util is not None and not (set(_UTIL_FIELDS) & fields)
                and not measured):
            declaration["bw_utilization"] = util
            if "effective_uni_bw_gbps" in fields:
                peak_value = declaration.get("effective_uni_bw_gbps")
            else:
                peak_value = (declaration.get("uni_peak_bw_gbps")
                              or declaration.get("effective_uni_bw_gbps"))
            if peak_value is not None:
                declaration["uni_peak_bw_gbps"] = float(peak_value)
                declaration["peak_capacity_bytes_per_ms"] = (
                    float(peak_value) * GBPS_TO_BYTES_PER_MS)
        if latency is not None and "latency_us" not in fields and "alpha_us" not in fields:
            declaration["latency_us"] = latency
    _finalize_channel_utilization(channels, explicit=explicit, declared=declared)


def _tier_bandwidth(channels, channel):
    declaration = channels[channel]
    util = _channel_util(declaration)
    capacity = declaration.get("effective_uni_bw_gbps")
    if not capacity:
        return None
    if math.isclose(util, 1.0):
        return float(capacity), util, float(capacity)
    peak = declaration.get("uni_peak_bw_gbps")
    if peak is None and declaration.get("peak_capacity_bytes_per_ms") is not None:
        peak = float(declaration["peak_capacity_bytes_per_ms"]) / GBPS_TO_BYTES_PER_MS
    if peak is None:
        peak = float(capacity) / util
    return float(peak), util, float(capacity)


def _link_record(name, members, channel, declaration, peak, util, latency,
                 capacity=None):
    resolved = (float(capacity) * GBPS_TO_BYTES_PER_MS if capacity is not None
                else float(peak) * GBPS_TO_BYTES_PER_MS * float(util))
    link = {"name": name, "members": list(members),
            "capacity_bytes_per_ms": resolved,
            "latency_us": latency, "channel": channel,
            "semantics": declaration["semantics"]}
    if not math.isclose(util, 1.0):
        link["bw_utilization"] = util
        link["peak_capacity_bytes_per_ms"] = float(peak) * GBPS_TO_BYTES_PER_MS
    for field_name in _PASSTHROUGH_LINK_FIELDS:
        if declaration.get(field_name) is not None:
            link[field_name] = deepcopy(declaration[field_name])
    return link


def _declaration_latency(declaration, fallback=0.0):
    return declaration.get("latency_us", declaration.get("alpha_us", fallback))


def _group_links(channels, channel, groups, peak=None, util=None, latency=None,
                 capacity=None):
    declaration = channels[channel]
    if peak is None:
        tier = _tier_bandwidth(channels, channel)
        if tier is None:
            return []
        peak, util, effective = tier
        if capacity is None:
            capacity = effective
    if util is None:
        util = _channel_util(declaration)
    if latency is None:
        latency = _declaration_latency(declaration)
    links = []
    for first, members in groups:
        if len(members) <= 1:
            continue
        links.append(_link_record(f"{channel}:{first}", members, channel,
                                  declaration, peak, util, latency,
                                  capacity=capacity))
    return links


def _sync_tier_channel(declaration, peak, util, effective, latency):
    declaration["uni_peak_bw_gbps"] = float(peak)
    declaration["effective_uni_bw_gbps"] = float(effective)
    declaration["latency_us"] = float(latency)
    if math.isclose(util, 1.0):
        _clear_utilization(declaration)
    else:
        declaration["bw_utilization"] = float(util)
        declaration["peak_capacity_bytes_per_ms"] = float(peak) * GBPS_TO_BYTES_PER_MS


def _subgroup_tier(channels, raw, preset, declared, explicit, name, key,
                   node_tier, node_latency, node_caller):
    node_peak, node_util, node_effective = node_tier
    fields = declared.get(name, set())
    latency = (_declaration_latency(channels[name], node_latency)
               if name in declared else node_latency)
    if name in explicit:
        peak, util, effective = _tier_bandwidth(channels, name)
        return peak, util, effective, latency
    util_declared = bool(set(_UTIL_FIELDS) & fields)
    tier_util = _channel_util(channels[name]) if util_declared else node_util
    if raw.get(key) is not None:
        peak = _number(key, raw[key])
        return peak, tier_util, peak * tier_util, latency
    if node_caller:
        effective = node_peak * tier_util if util_declared else node_effective
        return node_peak, tier_util, effective, latency
    peak = preset.get(key)
    if peak is not None:
        peak = _number(key, peak)
        return peak, tier_util, peak * tier_util, latency
    if name in declared:
        return node_peak, tier_util, node_peak * tier_util, latency
    return None


OPTICAL_SMALL_FRAME_BYTES = 4096
OPTICAL_MID_FRAME_BYTES = 65536
OPTICAL_LARGE_FRAME_BYTES = 262144
OPTICAL_SMALL_MAX_EXCLUSIVE = 65537
OPTICAL_MID_MAX_EXCLUSIVE = 2097152
OPTICAL_MID_CREDIT_BYTES = 524288
OPTICAL_LARGE_CREDIT_BYTES = 2097152
OPTICAL_MIN_GROUP = 2
# CloudMatrix SuperPoD optical HCCS is a 1-hop switched fabric: every ring step crosses the
# optical switch once, so the declared alpha is paid once per phase, never per ring hop.
OPTICAL_STARTUP_SCOPE = "phase"
OPTICAL_SUPERPOD_EVIDENCE_ID = (
    "campaign-202611-a2-a3/fabric_and_host_bus_train_summary.json#a3_superpod_optical_hccs"
)
A5_SUPERPOD_UB_EVIDENCE_ID = "a5-cann93/ub_superpod_optical"
A5_PCIE_GEN5_EVIDENCE_ID = "a5-cann93/pcie_gen5_16p"
_PROTOCOL_PIN_FIELDS = frozenset({"protocol", "protocol_bands", "transport_mode"})
_BAND_TIERS = (
    ("small", 0, OPTICAL_SMALL_MAX_EXCLUSIVE, OPTICAL_SMALL_FRAME_BYTES, OPTICAL_MID_FRAME_BYTES),
    ("mid", OPTICAL_SMALL_MAX_EXCLUSIVE, OPTICAL_MID_MAX_EXCLUSIVE, OPTICAL_MID_FRAME_BYTES, OPTICAL_MID_CREDIT_BYTES),
    ("large", OPTICAL_MID_MAX_EXCLUSIVE, None, OPTICAL_LARGE_FRAME_BYTES, OPTICAL_LARGE_CREDIT_BYTES),
)


def _a5_superpod_ub_bands(superpod_size: int, latency_us: float) -> list[dict]:
    max_group = max(OPTICAL_MIN_GROUP, int(superpod_size))
    return [
        {
            "collective": "all",
            "stack_id": "default",
            "topology_scope": "cross_node",
            "group_min": OPTICAL_MIN_GROUP,
            "group_max": max_group,
            "expansion_mode": "concurrent",
            "payload_min_inclusive": lo,
            "payload_max_exclusive": hi,
            "algorithm": "ring",
            "protocol": f"ub_superpod_{tier}",
            "startup_scope": OPTICAL_STARTUP_SCOPE,
            "startup_us": float(latency_us),
            "payload_bytes_per_frame": frame_bytes,
            "wire_bytes_per_frame": frame_bytes,
            "credit_bytes": credit_bytes,
            "evidence_id": A5_SUPERPOD_UB_EVIDENCE_ID,
        }
        for tier, lo, hi, frame_bytes, credit_bytes in _BAND_TIERS
    ]


def _a5_pcie_bands(per_node: int, latency_us: float) -> list[dict]:
    max_group = max(OPTICAL_MIN_GROUP, int(per_node))
    return [
        {
            "collective": "all",
            "stack_id": "default",
            "topology_scope": "node",
            "group_min": OPTICAL_MIN_GROUP,
            "group_max": max_group,
            "expansion_mode": "ring",
            "payload_min_inclusive": lo,
            "payload_max_exclusive": hi,
            "algorithm": "ring",
            "protocol": f"pcie_gen5_{tier}",
            "startup_scope": "hop",
            "startup_us": float(latency_us),
            "payload_bytes_per_frame": frame_bytes,
            "wire_bytes_per_frame": frame_bytes,
            "credit_bytes": credit_bytes,
            "evidence_id": A5_PCIE_GEN5_EVIDENCE_ID,
        }
        for tier, lo, hi, frame_bytes, credit_bytes in _BAND_TIERS
    ]


def _a3_superpod_optical_bands(superpod_size: int, latency_us: float) -> list[dict]:
    max_group = max(OPTICAL_MIN_GROUP, int(superpod_size))
    tiers = (
        ("small", 0, OPTICAL_SMALL_MAX_EXCLUSIVE, OPTICAL_SMALL_FRAME_BYTES, OPTICAL_MID_FRAME_BYTES),
        ("mid", OPTICAL_SMALL_MAX_EXCLUSIVE, OPTICAL_MID_MAX_EXCLUSIVE, OPTICAL_MID_FRAME_BYTES, OPTICAL_MID_CREDIT_BYTES),
        ("large", OPTICAL_MID_MAX_EXCLUSIVE, None, OPTICAL_LARGE_FRAME_BYTES, OPTICAL_LARGE_CREDIT_BYTES),
    )
    bands: list[dict] = []
    for tier, lo, hi, frame_bytes, credit_bytes in tiers:
        common = {
            "stack_id": "default",
            "topology_scope": "cross_node",
            "group_min": OPTICAL_MIN_GROUP,
            "group_max": max_group,
            "payload_min_inclusive": lo,
            "payload_max_exclusive": hi,
            "protocol": f"hccs_optical_{tier}",
            "startup_scope": OPTICAL_STARTUP_SCOPE,
            "startup_us": float(latency_us),
            "payload_bytes_per_frame": frame_bytes,
            "wire_bytes_per_frame": frame_bytes,
            "credit_bytes": credit_bytes,
            "evidence_id": OPTICAL_SUPERPOD_EVIDENCE_ID,
        }
        bands.append({
            "collective": "all",
            "expansion_mode": "ring",
            "algorithm": "ring",
            **common,
        })
        for coll in ("alltoall", "alltoallv"):
            bands.append({
                "collective": coll,
                "expansion_mode": "aiv",
                "algorithm": "direct_mesh",
                **common,
            })
    return bands


def _apply_superpod_inter_node_protocol(combined, raw, channels, declared, superpod_size, same_latency):
    preset_name = raw.get("topology_preset")
    sku_name = str(combined.get("sku", ""))
    is_a3_superpod = (
        preset_name == "a3_superpod"
        or (sku_name.startswith("Ascend910_93") and preset_name is None and combined.get("superpod_size") is not None)
    )
    is_a5_pod = (
        preset_name == "a5_pod"
        or (sku_name.startswith("Ascend950") and preset_name is None and raw.get("superpod_size") is not None)
    )
    if _PROTOCOL_PIN_FIELDS & declared.get("d2d_inter_node", set()):
        return
    if is_a3_superpod:
        channels["d2d_inter_node"].update(
            protocol="HCCS_OPTICAL_SUPERPOD", transport_mode="optical_hccs",
            rail_id="hccs_optical_r0", switch_id="superpod_sw0",
            scope="cross_node", alpha_us=same_latency,
            protocol_bands=_a3_superpod_optical_bands(superpod_size, same_latency),
        )
    elif is_a5_pod:
        channels["d2d_inter_node"].update(
            protocol="UB_SUPERPOD", transport_mode="ub_optical",
            rail_id="ub_pod_r0", switch_id="ub_pod_sw0",
            scope="cross_node", alpha_us=same_latency,
            protocol_bands=_a5_superpod_ub_bands(superpod_size, same_latency),
        )


def _pod_node_width(combined, raw, preset, per_node: int) -> int:
    """Resolve physical node die capacity for SuperPoD node-count grouping."""
    if raw.get("ranks_per_node") is not None:
        return int(per_node)
    phys_rpn = int(
        raw.get("physical_ranks_per_node")
        or preset.get("ranks_per_node")
        or combined.get("physical_ranks_per_node")
        or default_ranks_per_node(str(combined.get("sku", "")))
    )
    return max(phys_rpn, int(per_node))


def _superpod_groups(placements, node_width: int, superpod_size: int):
    """Group nodes into SuperPoD domains while preserving sparse multi-node boundaries."""
    if node_width > 0 and superpod_size >= node_width and superpod_size % node_width == 0:
        nodes_per_pod = max(1, superpod_size // node_width)
        node_groups = placement_node_groups(placements)
        if all(len(members) <= node_width for _, members in node_groups):
            return [
                (
                    node_groups[start][0],
                    [rank for _, members in node_groups[start:start + nodes_per_pod]
                     for rank in members],
                )
                for start in range(0, len(node_groups), nodes_per_pod)
            ]
    return placement_node_order_groups(placements, superpod_size)


def _inter_node_links(combined, raw, preset, channels, placements, count, per_node, explicit, bw_caller, declared):
    del count
    superpod_size = combined.get("superpod_size", combined.get("max_dies"))
    if superpod_size is not None:
        if isinstance(superpod_size, bool):
            raise ModelingError("ascend_superpod_size_invalid")
        superpod_size = int(superpod_size)
        if superpod_size <= 0:
            raise ModelingError("ascend_superpod_size_invalid")
        if superpod_size < per_node:
            raise ModelingError("ascend_superpod_size_smaller_than_node")
    inter_tier = _tier_bandwidth(channels, "d2d_inter_node") if "d2d_inter_node" in channels else None
    orig_inter_decl = deepcopy(channels["d2d_inter_node"]) if "d2d_inter_node" in channels else None
    same_peak = same_util = same_capacity = same_latency = None
    if superpod_size is not None and inter_tier is not None:
        same_peak, same_util, same_capacity, same_latency = resolve_same_pod_tier(
            combined, raw, preset, declared, "d2d_inter_node" in explicit,
            "d2d_inter_node" in bw_caller, inter_tier,
            _declaration_latency(channels["d2d_inter_node"]))
        channels["d2d_inter_node"].update(
            uni_peak_bw_gbps=same_peak, bw_utilization=same_util,
            effective_uni_bw_gbps=same_capacity,
            peak_capacity_bytes_per_ms=same_peak * GBPS_TO_BYTES_PER_MS,
            latency_us=same_latency)
        _apply_superpod_inter_node_protocol(combined, raw, channels, declared, superpod_size, same_latency)
    spans_nodes = len({node for _, node, _ in placements}) > 1
    if not (per_node > 0 and spans_nodes and inter_tier is not None):
        return []
    links = []
    inter_peak, inter_util, inter_effective = inter_tier
    inter_decl = orig_inter_decl or channels["d2d_inter_node"]
    inter_latency = _declaration_latency(inter_decl)
    if superpod_size is not None:
        pod_groups = _superpod_groups(
            placements, _pod_node_width(combined, raw, preset, per_node), superpod_size,
        )
        links.extend(_group_links(
            channels, "d2d_inter_node", pod_groups,
            peak=same_peak, util=same_util, latency=same_latency, capacity=same_capacity))
        if len(pod_groups) > 1:
            pod_fields = declared.get("d2d_inter_pod", set())
            if "d2d_inter_pod" in explicit:
                pod_peak, pod_util, pod_effective = _tier_bandwidth(channels, "d2d_inter_pod")
            else:
                pod_peak = _number("inter_pod_bw_gbps", raw.get(
                    "inter_pod_bw_gbps", preset.get("inter_pod_bw_gbps", inter_peak)))
                if set(_UTIL_FIELDS) & pod_fields:
                    pod_util = _channel_util(channels["d2d_inter_pod"])
                else:
                    pod_util = _positive_float(
                        raw.get("inter_node_bw_util", preset.get("inter_node_bw_util", 1.0)),
                        "inter_node_bw_util_invalid", allow_zero=False)
                pod_effective = pod_peak * pod_util
            if {"latency_us", "alpha_us"} & pod_fields:
                pod_latency = _declaration_latency(channels["d2d_inter_pod"], inter_latency)
            else:
                pod_latency = _positive_float(
                    raw.get("inter_pod_latency_us", preset.get("inter_pod_latency_us", inter_latency)),
                    "inter_pod_latency_us_invalid", allow_zero=True)
            pod_decl = {**inter_decl, "channel": "d2d_inter_pod"}
            if pod_decl.get("rail_id"):
                pod_decl["rail_id"] = f"{pod_decl['rail_id']}_pod"
            channels.setdefault("d2d_inter_pod", pod_decl)
            _sync_tier_channel(channels["d2d_inter_pod"], pod_peak, pod_util, pod_effective, pod_latency)
            links.extend(_group_links(
                channels, "d2d_inter_pod", placement_node_order_groups(placements),
                peak=pod_peak, util=pod_util, latency=pod_latency, capacity=pod_effective))
    else:
        links.extend(_group_links(
            channels, "d2d_inter_node", placement_node_order_groups(placements),
            peak=inter_peak, util=inter_util, latency=inter_latency, capacity=inter_effective))
    return links


def _generated_links(combined, raw, preset, channels, count, per_package,
                     per_node, explicit, bw_caller, declared):
    placements = resolve_device_placements(
        raw.get("devices", raw.get("dies", ())) or (), count, per_node, per_package)
    links = []
    if per_package > 1 and count > 1:
        links.extend(_group_links(
            channels, "d2d_intra_package",
            placement_package_groups(placements)))

    node_tier = (_tier_bandwidth(channels, "d2d_intra_node")
                 if "d2d_intra_node" in channels else None)
    node_groups = placement_node_groups(placements)
    has_shared_node = any(len(members) > 1 for _, members in node_groups)
    if (per_node > 1 or has_shared_node) and node_tier is not None:
        node_peak, node_util, node_effective = node_tier
        node_decl = channels["d2d_intra_node"]
        node_latency = _declaration_latency(node_decl)
        if (raw.get("topology_preset") == "a5_pcie_16p"
                and not (_PROTOCOL_PIN_FIELDS & declared.get("d2d_intra_node", set()))):
            node_decl.update(
                protocol="PCIe_Gen5", transport_mode="pcie_p2p", rail_id="pcie_r0",
                scope="node", alpha_us=node_latency,
                protocol_bands=_a5_pcie_bands(per_node, node_latency),
            )
        node_caller = "d2d_intra_node" in bw_caller
        quad = _subgroup_tier(
            channels, raw, preset, declared, explicit, "d2d_intra_4p",
            "intra_bw_4p_gbps", node_tier, node_latency, node_caller)
        octa = _subgroup_tier(
            channels, raw, preset, declared, explicit, "d2d_intra_8p",
            "intra_bw_8p_gbps", node_tier, node_latency, node_caller)
        if quad is not None and per_node > SUBGROUP_QUAD_DIE_RANKS:
            quad_decl = {**node_decl, "channel": "d2d_intra_4p"}
            if quad_decl.get("rail_id"):
                quad_decl["rail_id"] = f"{quad_decl['rail_id']}_4p"
            channels.setdefault("d2d_intra_4p", quad_decl)
            peak, util, effective, latency = quad
            _sync_tier_channel(channels["d2d_intra_4p"], peak, util, effective, latency)
            links.extend(_group_links(
                channels, "d2d_intra_4p",
                placement_node_groups(placements, SUBGROUP_QUAD_DIE_RANKS),
                peak=peak, util=util, latency=latency, capacity=effective))
        if octa is not None and per_node > SUBGROUP_OCTA_DIE_RANKS:
            octa_decl = {**node_decl, "channel": "d2d_intra_8p"}
            if octa_decl.get("rail_id"):
                octa_decl["rail_id"] = f"{octa_decl['rail_id']}_8p"
            channels.setdefault("d2d_intra_8p", octa_decl)
            peak, util, effective, latency = octa
            _sync_tier_channel(channels["d2d_intra_8p"], peak, util, effective, latency)
            links.extend(_group_links(
                channels, "d2d_intra_8p",
                placement_node_groups(placements, SUBGROUP_OCTA_DIE_RANKS),
                peak=peak, util=util, latency=latency, capacity=effective))
        node_peak_eff, node_util_eff, node_capacity = node_peak, node_util, node_effective
        if per_node <= SUBGROUP_QUAD_DIE_RANKS and quad is not None:
            node_peak_eff, node_util_eff, node_capacity, node_latency = quad
        elif per_node <= SUBGROUP_OCTA_DIE_RANKS and octa is not None:
            node_peak_eff, node_util_eff, node_capacity, node_latency = octa
        links.extend(_group_links(
            channels, "d2d_intra_node", node_groups,
            peak=node_peak_eff, util=node_util_eff, latency=node_latency,
            capacity=node_capacity))

    links.extend(_inter_node_links(
        combined, raw, preset, channels, placements, count, per_node, explicit, bw_caller, declared,
    ))
    return links
