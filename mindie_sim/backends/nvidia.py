"""NVIDIA public-SKU capability input adapter.

[Role] Materialize public NVIDIA SKU metadata into neutral devices and links.
[Invariants] Only publicly released H20, H800, H200 and B300 profiles are
  registered. Device rates come from the declared HBM bandwidth, tensor-core
  peaks and FP32 vector rate; no vendor branch enters the execution engine.
  ``*_bw_gbps`` values are GB/s, matching the historical transfer-channel
  convention, and unity utilization reproduces the historical link key set.
[Timeouts & Locks] Bounded local JSON read; no device context or driver call.
[Side-Effect Policy] No external writes, probes or weight loading.
"""
from functools import lru_cache
import json
import math
from pathlib import Path
from collections.abc import Mapping

from mindie_sim.schema.common import ModelingError, _positive_float
from mindie_sim.schema.device_input import machine_device_defaults, merge_device_inputs


_TB = 1e12
_NETWORK_FIELDS = {
    "inter_node_bw_gbps", "intra_node_bw_gbps", "intra_package_bw_gbps",
    "inter_node_bw_util", "intra_node_bw_util", "intra_package_bw_util",
    "inter_node_latency_us", "intra_node_latency_us", "intra_package_latency_us",
    "inter_node_bw_bytes_per_s", "intra_node_bw_bytes_per_s",
    "intra_package_bw_bytes_per_s",
    "topology_preset", "intra_bw_4p_gbps", "intra_bw_8p_gbps",
    "superpod_size", "max_dies", "superpod_bw_gbps", "inter_pod_bw_gbps",
    "superpod_latency_us", "inter_pod_latency_us",
}
_ASSUMPTIONS = {
    "profile": "registered public NVIDIA SKU geometry and declared rates",
    "hbm_units": "hbm_gb is GiB and hbm_bw_gbps is decimal GB/s",
    "compute_peaks": (
        "cube_fp16_tflops, cube_fp8_tflops and cube_int4_tflops are the "
        "publicly quoted dense tensor-core rates; vector_fp32_tflops is the "
        "declared FP32 vector rate"
    ),
    "links": (
        "intra_node_bw_gbps and inter_node_bw_gbps are GB/s; per-port links "
        "carry independent transmit and receive service, and bandwidth "
        "utilization scales the declared peak"
    ),
}


@lru_cache(maxsize=1)
def profiles():
    with Path(__file__).with_name("profiles").joinpath("nvidia.json").open() as stream:
        return json.load(stream)


def _positive(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0):
        raise ModelingError(f"nvidia_geometry_invalid:{name}:{value}")
    return float(value)


def _device(profile):
    peaks = {
        "bf16": _positive("cube_fp16_tflops", profile["cube_fp16_tflops"]) * _TB,
        "fp16": _positive("cube_fp16_tflops", profile["cube_fp16_tflops"]) * _TB,
        "fp8": _positive("cube_fp8_tflops", profile["cube_fp8_tflops"]) * _TB,
        "int8": _positive("cube_fp8_tflops", profile["cube_fp8_tflops"]) * _TB,
        "fp32": _positive("vector_fp32_tflops", profile["vector_fp32_tflops"]) * _TB,
    }
    if profile.get("cube_int4_tflops") is not None:
        peaks["int4"] = _positive("cube_int4_tflops",
                                  profile["cube_int4_tflops"]) * _TB
    buffers = {"l2": int(_positive("l2_mb", profile["l2_mb"]) * 1024**2)}
    return {
        "memory_bytes": int(_positive("hbm_gb", profile["hbm_gb"]) * 1024**3),
        "compute_peaks": peaks,
        "rates": {
            "hbm": _positive("hbm_bw_gbps", profile["hbm_bw_gbps"]) * 1e9,
            "vector": _positive("vector_fp32_tflops",
                                profile["vector_fp32_tflops"]) * _TB,
        },
        "buffers": buffers,
        "compute_units": int(_positive("sm_count", profile["sm_count"])),
        "resource_units": {"vector": int(_positive("sm_count", profile["sm_count"]))},
        "launch_us": float(profile.get("launch_us", 0.0)),
    }


def _channel(name, gbps, latency_us=0.0):
    value = _positive(f"{name}_bw_gbps", gbps)
    return {
        "channel": name,
        "protocol": "NVLINK" if name != "d2d_inter_node" else "RDMA",
        "uni_peak_bw_gbps": value,
        "effective_uni_bw_gbps": value,
        "alpha_us": float(latency_us),
        "latency_us": float(latency_us),
        "channel_semantics": "per_port",
        "semantics": "per_port",
        "bandwidth_provenance": "declared_peak",
    }


def _channels(profile):
    channels = {
        "d2d_intra_node": _channel("d2d_intra_node",
                                   profile["intra_node_bw_gbps"]),
        "d2d_inter_node": _channel("d2d_inter_node",
                                   profile["inter_node_bw_gbps"]),
    }
    if profile.get("intra_package_bw_gbps") is not None:
        channels["d2d_intra_package"] = _channel(
            "d2d_intra_package", profile["intra_package_bw_gbps"])
    return channels


def _number(name, value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0):
        raise ModelingError(f"transfer_bandwidth_invalid:{name}:{value}")
    return float(value)


def _merge_channels(profile, explicit):
    channels = _channels(profile)
    if not explicit:
        return channels, {}
    if not isinstance(explicit, Mapping):
        raise ModelingError("transfer_channels_invalid")
    declared = {}
    for channel, raw_decl in explicit.items():
        if not isinstance(raw_decl, Mapping):
            raise ModelingError(f"transfer_channel_invalid:{channel}")
        declared[channel] = set(raw_decl)
        base = dict(channels.get(channel, {
            "channel": channel, "protocol": "CUSTOM",
            "channel_semantics": "per_port", "semantics": "per_port",
            "latency_us": 0.0, "alpha_us": 0.0,
        }))
        patch = dict(raw_decl)
        if "channel_semantics" in patch and "semantics" not in patch:
            patch["semantics"] = patch["channel_semantics"]
        if "semantics" in patch and "channel_semantics" not in patch:
            patch["channel_semantics"] = patch["semantics"]
        if "alpha_us" in patch and "latency_us" not in patch:
            patch["latency_us"] = patch["alpha_us"]
        if "latency_us" in patch:
            lat = _positive_float(patch["latency_us"], f"transfer_channel_latency_invalid:{channel}", allow_zero=True)
            patch["latency_us"] = lat
            patch["alpha_us"] = lat
        util = 1.0
        for uf in ("bw_utilization", "bw_ratio", "utilization"):
            if patch.get(uf) is not None:
                util = _positive_float(patch[uf], "transfer_channel_bw_utilization_invalid", allow_zero=False)
                if util > 1.0:
                    raise ModelingError("transfer_channel_bw_utilization_invalid")
                break
        if "effective_uni_bw_gbps" in patch and patch["effective_uni_bw_gbps"] is not None:
            peak = _positive_float(patch["effective_uni_bw_gbps"], f"transfer_channel_bw_invalid:{channel}")
        elif "peak_capacity_bytes_per_ms" in patch and patch["peak_capacity_bytes_per_ms"] is not None:
            peak = _positive_float(patch["peak_capacity_bytes_per_ms"], f"transfer_channel_bw_invalid:{channel}") / 1e6
        else:
            peak = None
            for bw_field in ("calibrated_uni_bw_gbps", "derived_uni_bw_gbps", "uni_peak_bw_gbps"):
                if patch.get(bw_field) is not None:
                    peak = _positive_float(patch[bw_field], f"transfer_channel_bw_invalid:{channel}")
                    break
            if peak is None:
                peak = float(base.get("uni_peak_bw_gbps", base.get("effective_uni_bw_gbps", 0.0)))
        base.update(patch)
        base["uni_peak_bw_gbps"] = peak
        base["effective_uni_bw_gbps"] = peak * util
        base["bw_utilization"] = util
        if not math.isclose(util, 1.0):
            base["peak_capacity_bytes_per_ms"] = peak * 1e6
        channels[channel] = base
    return channels, declared


def _apply_overrides(raw, channels, declared=None):
    declared = declared or {}
    bw_fields = {"peak_capacity_bytes_per_ms", "uni_peak_bw_gbps",
                 "calibrated_uni_bw_gbps", "derived_uni_bw_gbps",
                 "effective_uni_bw_gbps", "capacity_bytes_per_ms"}
    for kind, channel in (("inter_node", "d2d_inter_node"),
                          ("intra_node", "d2d_intra_node"),
                          ("intra_package", "d2d_intra_package")):
        declaration = channels.get(channel)
        if declaration is None:
            continue
        ch_fields = declared.get(channel, set())
        bytes_field = f"{kind}_bw_bytes_per_s"
        gbps_field = f"{kind}_bw_gbps"
        util_field = f"{kind}_bw_util"
        latency_field = f"{kind}_latency_us"
        if raw.get(bytes_field) is not None and not (ch_fields & bw_fields):
            peak = _number(bytes_field, raw[bytes_field]) / 1e9
        elif raw.get(gbps_field) is not None and not (ch_fields & bw_fields):
            peak = _number(gbps_field, raw[gbps_field])
        else:
            peak = declaration["uni_peak_bw_gbps"]
        util = float(declaration.get("bw_utilization", 1.0))
        if raw.get(util_field) is not None and not ({"bw_utilization", "bw_ratio", "utilization"} & ch_fields):
            util = _positive_float(raw[util_field], f"{util_field}_invalid",
                                   allow_zero=False)
            if util > 1.0:
                raise ModelingError("link_bw_utilization_exceeds_unity")
        latency = declaration.get("latency_us", 0.0)
        if raw.get(latency_field) is not None and not ({"latency_us", "alpha_us"} & ch_fields):
            latency = _positive_float(raw[latency_field],
                                      f"{latency_field}_invalid", allow_zero=True)
        declaration["uni_peak_bw_gbps"] = peak
        declaration["effective_uni_bw_gbps"] = peak * util
        declaration["latency_us"] = latency
        declaration["bw_utilization"] = util
        if not math.isclose(util, 1.0):
            declaration["peak_capacity_bytes_per_ms"] = peak * 1e6
    return channels


def _links(channels, count, per_package, per_node):
    links = []

    def add(channel, groups):
        declaration = channels[channel]
        util = float(declaration.get("bw_utilization", 1.0))
        peak = declaration["uni_peak_bw_gbps"]
        latency = declaration.get("latency_us", declaration.get("alpha_us", 0.0))
        semantics = declaration.get("semantics", declaration.get("channel_semantics", "per_port"))
        for first, members in groups:
            if len(members) <= 1:
                continue
            link = {
                "name": f"{channel}:{first}", "members": members,
                "capacity_bytes_per_ms": peak * 1e6 * util,
                "latency_us": latency, "channel": channel,
                "semantics": semantics,
            }
            if not math.isclose(util, 1.0):
                link["bw_utilization"] = util
                link["peak_capacity_bytes_per_ms"] = peak * 1e6
            for field in ("transaction_bytes", "alignment_bytes",
                          "protocol_overhead_bytes", "switch_hops",
                          "switch_hop_latency_us", "switch_capacity_bytes_per_ms"):
                if declaration.get(field) is not None:
                    link[field] = declaration[field]
            links.append(link)

    if per_package > 1 and "d2d_intra_package" in channels:
        add("d2d_intra_package", [
            (first, list(range(first, min(first + per_package, count))))
            for first in range(0, count, per_package)])
    if per_node > 1:
        add("d2d_intra_node", [
            (first, list(range(first, min(first + per_node, count))))
            for first in range(0, count, per_node)])
    if count > per_node:
        add("d2d_inter_node", [(0, list(range(count)))])
    return links


def materialize(raw):
    if raw.get("sku") not in profiles():
        raise ModelingError(f"sku_not_registered:{raw.get('sku')}:declare_device_geometry")
    profile = profiles()[raw["sku"]]
    defaults = machine_device_defaults(raw)
    device = merge_device_inputs(_device(profile), defaults, raw.get("device", {}))
    merged_channels, declared = _merge_channels(profile, raw.get("transfer_channels"))
    channels = _apply_overrides(raw, merged_channels, declared)
    data = {**raw, "device": device, "transfer_channels": channels,
            "architecture": profile["arch_family"]}
    for field in ("devices", "dies"):
        if field in raw:
            data["device"] = device
            data[field] = [merge_device_inputs(device, record)
                           for record in raw[field]]
    count = int(raw.get("num_devices", 1))
    per_node = int(raw.get("ranks_per_node", profile["ranks_per_node"]))
    per_package = int(raw.get("dies_per_package", profile["dies_per_package"]))
    data.update(ranks_per_node=per_node, dies_per_package=per_package)
    if "links" not in raw:
        data["links"] = _links(channels, count, per_package, per_node)
        for field in _NETWORK_FIELDS:
            data.pop(field, None)
    data["assumptions"] = {**profile.get("assumptions", {}),
                           **_ASSUMPTIONS, **raw.get("assumptions", {})}
    return data


def normalize_nvidia_machine(raw):
    """Public adapter entry point for an NVIDIA machine declaration."""
    return materialize(raw)
