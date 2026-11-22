"""Agent-first custom hardware templates, builders, and a private SKU registry.

[Role]
  Export neutral structural hardware archetypes, materialize complete machine
  dictionaries from the handful of specifications an agent actually knows, and
  keep an in-memory or untracked-file registry of private custom SKUs.

[Invariants]
  * Public templates contain neutral structural defaults only; no confidential
    SKU identity or measured private parameter is embedded in this repository.
  * Every builder result parses through ``Hardware.from_mapping``. Memory is
    decimal GB, rates are SI units, link capacity is bytes/ms, and all units are
    documented explicitly in exported templates.
  * Registered profiles keep caller-supplied values and the chosen archetype.
    Derived values are recomputed by the builder, so later key-spec overrides
    still scale dependent rates.
  * Environment-file discovery reads only the declared path and never
    overwrites an explicitly registered in-memory SKU.

[Timeouts & Locks]
  Bounded local metadata reads (under 50 MiB); no network, device context,
  subprocess, wait, or lock is used.

[Side-Effect Policy]
  Export, build, parse, and resolve operations are pure. Registry mutations
  happen only through explicit register/load/unregister/clear calls.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
import math
import os
from pathlib import Path
from typing import Any

from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.compute_arch import ComputeArchSpec
from mindie_sim.schema.hardware import Hardware
from mindie_sim.hardware_derivation import derive_hardware_sku

from .custom_hardware_templates import (
    HARDWARE_ARCHETYPE_TEMPLATES,
    _TEMPLATE_SKU,
    _merge_inputs,
    _plain_mapping,
    export_hardware_sku_template,
)


_MAX_METADATA_BYTES = 50 * 1024**2


_CUSTOM_SKU_REGISTRY: dict[str, dict[str, Any]] = {}
_CUSTOM_SKU_CANONICAL: dict[str, str] = {}


def _positive_float(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0.0):
        raise ModelingError(f"custom_hardware_positive_required:{name}:{value}")
    return float(value)


def _nonnegative_float(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) < 0.0):
        raise ModelingError(f"custom_hardware_nonnegative_required:{name}:{value}")
    return float(value)


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ModelingError(f"custom_hardware_positive_integer_required:{name}:{value}")
    return int(value)


def _utilization(value: Any, name: str) -> float:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0):
        raise ModelingError(f"custom_hardware_utilization_out_of_range:{name}:{value}")
    return float(value)


def _rate_from(
    explicit: Mapping[str, Any], merged: Mapping[str, Any], gbps_key: str,
    bytes_key: str, fallback: float,
) -> float:
    if gbps_key in explicit and explicit[gbps_key] is not None:
        return _positive_float(explicit[gbps_key], gbps_key) * 1e9
    if bytes_key in explicit and explicit[bytes_key] is not None:
        return _positive_float(explicit[bytes_key], bytes_key)
    return fallback


_FORMAT_BITS = {
    "bf16": 16, "fp16": 16, "fp32": 32, "hf32": 32, "fp8": 8,
    "fp8_e4m3": 8, "fp8_e5m2": 8, "int8": 8, "int4": 4,
}


def _neutral_capability(peaks: Mapping[str, Any]) -> dict[str, Any]:
    """Declare exactly the synthetic native modes this template's peaks imply.

    A neutral template is not a vendor SKU. It gets native modes for the
    formats it already prices, plus payload representations; it gets no cast
    rules, data paths or runtime implementations unless the caller declares
    them explicitly.
    """
    representations: dict[str, Any] = {}
    native_compute: dict[str, Any] = {}
    def provenance() -> dict[str, Any]:
        return {
            "kind": "neutral_custom_template",
            "basis": "synthetic rates declared by this template; no vendor SKU, "
                     "measured capability or runtime implementation is implied",
        }
    needs_int_accumulator = False
    for name in sorted(peaks):
        bits = _FORMAT_BITS.get(name)
        if bits is None:
            raise ModelingError(f"custom_hardware_unknown_peak_format:{name}")
        integer = name.startswith("int")
        accumulator = "int32" if integer else "fp32"
        needs_int_accumulator = needs_int_accumulator or integer
        roles = ["storage", "transfer", "operand"]
        if not integer:
            roles.append("accumulator")
        representations[name] = {
            "bits": bits, "roles": roles,
            "alignment_bytes": max(1, bits // 8),
            "provenance": provenance(),
        }
        native_compute[f"{name}_x_{name}_acc_{accumulator}"] = {
            "kind": "matrix", "operands": [name],
            "accumulators": [accumulator], "unit": "cube",
            "provenance": provenance(),
        }
    if needs_int_accumulator:
        representations.setdefault("int32", {
            "bits": 32, "roles": ["storage", "transfer", "accumulator"],
            "alignment_bytes": 4, "provenance": provenance(),
        })
    return {"representations": representations, "native_compute": native_compute}


def _validate_fields(merged: Mapping[str, Any], explicit: Mapping[str, Any]) -> None:
    for key in ("hbm_gb", "hbm_bw_gbps", "cube_fp16_tflops", "vector_fp32_tflops"):
        _positive_float(merged.get(key), key)
    for key in ("cube_fp8_tflops", "cube_int4_tflops", "fp4_tflops",
                "mte1_bw_gbps", "mte2_bw_gbps", "mte2_aic_bw_gbps",
                "mte2_aiv_bw_gbps", "mte3_bw_gbps", "fixpipe_bw_gbps"):
        if explicit.get(key) is not None:
            _positive_float(explicit[key], key)
    for key in ("aic_count", "aiv_count", "dies_per_package", "ranks_per_node"):
        _positive_int(merged.get(key), key)
    for key in ("l2_mb", "ub_kb", "l1_kb", "l0a_kb", "l0b_kb", "l0c_kb"):
        _positive_float(merged.get(key), key)
    for key in ("intra_package_latency_us", "intra_node_latency_us",
                "inter_node_latency_us", "inter_pod_latency_us", "launch_us"):
        if merged.get(key) is not None:
            _nonnegative_float(merged[key], key)
    for key in ("intra_package_bw_util", "intra_node_bw_util", "inter_node_bw_util"):
        if key != "intra_package_bw_util" or merged.get(key) is not None:
            _utilization(merged.get(key), key)
    superpod = merged.get("superpod_size", merged.get("max_dies"))
    if superpod is not None:
        _positive_int(superpod, "superpod_size")
    for key in ("intra_package_bw_gbps", "intra_bw_4p_gbps", "intra_bw_8p_gbps",
                "intra_node_bw_gbps", "superpod_bw_gbps", "inter_node_bw_gbps",
                "inter_pod_bw_gbps"):
        value = merged.get(key)
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, (int, float))
                                  or not math.isfinite(float(value))
                                  or float(value) < 0.0):
            raise ModelingError(f"custom_hardware_bandwidth_invalid:{key}:{value}")
    capability = explicit.get("capability")
    if capability is not None:
        if not isinstance(capability, Mapping):
            raise ModelingError("custom_hardware_capability_must_be_mapping")
        ComputeArchSpec.from_mapping(dict(capability),
                                     source="custom_hardware_capability")


def _derived_device_fields(
    merged: Mapping[str, Any], explicit: Mapping[str, Any],
) -> dict[str, Any]:
    hbm_bytes_per_s = float(merged["hbm_bw_gbps"]) * 1e9
    fp16_flops = float(merged["cube_fp16_tflops"]) * 1e12
    fp8_explicit = explicit.get("cube_fp8_tflops")
    fp8_flops = (float(fp8_explicit) * 1e12 if fp8_explicit is not None
                 else 2.0 * fp16_flops)
    int4_tflops = next((float(explicit[key]) for key in
                        ("cube_int4_tflops", "fp4_tflops")
                        if explicit.get(key) is not None), None)
    peaks = {
        "bf16": fp16_flops, "fp16": fp16_flops,
        "fp32": fp16_flops / 4.0, "hf32": fp16_flops / 2.0,
        "fp8": fp8_flops, "fp8_e4m3": fp8_flops, "fp8_e5m2": fp8_flops,
        "int8": fp8_flops,
    }
    if int4_tflops is not None:
        peaks["int4"] = int4_tflops * 1e12
    mte1 = _rate_from(explicit, merged, "mte1_bw_gbps", "mte1_bw_bytes_per_s", 2.0 * hbm_bytes_per_s)
    mte2_common = (float(explicit["mte2_bw_gbps"]) * 1e9 if explicit.get("mte2_bw_gbps") is not None
                   else explicit.get("mte2_bw_bytes_per_s"))
    mte2_aic = _rate_from(explicit, merged, "mte2_aic_bw_gbps", "mte2_aic_bw_bytes_per_s",
                          float(mte2_common) if mte2_common is not None else hbm_bytes_per_s)
    mte2_aiv = _rate_from(explicit, merged, "mte2_aiv_bw_gbps", "mte2_aiv_bw_bytes_per_s",
                          float(mte2_common) if mte2_common is not None else 2.0 * hbm_bytes_per_s)
    return {
        "compute_peaks": peaks,
        "rates": {
            "hbm": hbm_bytes_per_s, "vector": float(merged["vector_fp32_tflops"]) * 1e12,
            "mte1": mte1, "mte2": float(mte2_common) if mte2_common is not None else mte2_aic,
            "mte2_aic": mte2_aic, "mte2_aiv": mte2_aiv,
            "mte3": _rate_from(explicit, merged, "mte3_bw_gbps", "mte3_bw_bytes_per_s", 2.0 * hbm_bytes_per_s),
            "fixpipe": _rate_from(explicit, merged, "fixpipe_bw_gbps", "fixpipe_bw_bytes_per_s",
                                  2.0 * hbm_bytes_per_s),
        },
    }


def _link(
    channel: str, members: Sequence[int], bw_gbps: float, utilization: float,
    latency_us: float, *, name_prefix: str | None = None,
    semantics: str = "shared",
) -> dict[str, Any]:
    util = float(utilization)
    peak = float(bw_gbps) * 1e6
    capacity = peak * util
    prefix = name_prefix or channel
    record: dict[str, Any] = {
        "name": f"{prefix}:{members[0]}",
        "members": tuple(int(rank) for rank in members),
        "capacity_bytes_per_ms": capacity,
        "latency_us": float(latency_us),
        "channel": channel,
        "semantics": semantics,
    }
    if not math.isclose(util, 1.0):
        record["bw_utilization"] = util
        record["peak_capacity_bytes_per_ms"] = peak
    return record


def _append_group_links(
    links: list[dict[str, Any]], *, channel: str, group_size: int,
    bw_gbps: Any, utilization: float, latency_us: float, num_devices: int,
    name_prefix: str | None = None, semantics: str = "shared",
) -> None:
    if group_size <= 1 or bw_gbps is None or float(bw_gbps) <= 0.0:
        return
    for start in range(0, num_devices, group_size):
        members = tuple(range(start, min(start + group_size, num_devices)))
        if len(members) > 1:
            links.append(_link(channel, members, float(bw_gbps), utilization,
                               latency_us, name_prefix=name_prefix,
                               semantics=semantics))


def _materialize_links(
    merged: Mapping[str, Any], *, num_devices: int, dies_per_package: int,
    ranks_per_node: int,
) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    intra_util = float(merged["intra_node_bw_util"])
    inter_util = float(merged["inter_node_bw_util"])
    node_latency = float(merged.get("inter_node_latency_us", 0.0))
    intra_latency = float(merged.get("intra_node_latency_us", 0.0))
    if dies_per_package > 1 and merged.get("intra_package_bw_gbps") is not None:
        pkg_util = float(merged.get("intra_package_bw_util") or 1.0)
        _append_group_links(
            links, channel="intra_package", group_size=dies_per_package,
            bw_gbps=merged.get("intra_package_bw_gbps"), utilization=pkg_util,
            latency_us=float(merged.get("intra_package_latency_us", 0.0)),
            num_devices=num_devices, semantics="direct",
        )
    if merged.get("intra_bw_4p_gbps") is not None and dies_per_package > 0:
        _append_group_links(
            links, channel="intra_4p", group_size=4 * dies_per_package,
            bw_gbps=merged.get("intra_bw_4p_gbps"), utilization=intra_util,
            latency_us=intra_latency, num_devices=num_devices,
        )
    if merged.get("intra_bw_8p_gbps") is not None and dies_per_package > 0:
        _append_group_links(
            links, channel="intra_8p", group_size=8 * dies_per_package,
            bw_gbps=merged.get("intra_bw_8p_gbps"), utilization=intra_util,
            latency_us=intra_latency, num_devices=num_devices,
        )
    if merged.get("intra_node_bw_gbps") is not None:
        _append_group_links(
            links, channel="intra_node", group_size=ranks_per_node,
            bw_gbps=merged.get("intra_node_bw_gbps"), utilization=intra_util,
            latency_us=intra_latency, num_devices=num_devices,
        )
    superpod = merged.get("superpod_size", merged.get("max_dies"))
    superpod_size = int(superpod) if superpod is not None else 0
    if superpod_size > 0:
        _append_group_links(
            links, channel="superpod", group_size=min(superpod_size, num_devices),
            bw_gbps=merged.get("superpod_bw_gbps"), utilization=inter_util,
            latency_us=node_latency, num_devices=num_devices,
        )
        if num_devices > superpod_size:
            _append_group_links(
                links, channel="inter_pod", group_size=num_devices,
                bw_gbps=merged.get("inter_pod_bw_gbps"), utilization=inter_util,
                latency_us=float(merged.get("inter_pod_latency_us", node_latency)),
                num_devices=num_devices,
            )
        else:
            _append_group_links(
                links, channel="inter_node", group_size=num_devices,
                bw_gbps=merged.get("inter_node_bw_gbps"), utilization=inter_util,
                latency_us=node_latency, num_devices=num_devices,
                name_prefix="z_fallback:inter_node",
            )
    else:
        # ``z_fallback:`` keeps intra-node and superpod links first for equal-member
        # routing while still declaring the inter-node fabric for cross-node cases.
        _append_group_links(
            links, channel="inter_node", group_size=num_devices,
            bw_gbps=merged.get("inter_node_bw_gbps"), utilization=inter_util,
            latency_us=node_latency, num_devices=num_devices,
            name_prefix="z_fallback:inter_node",
        )
    return [link for link in links if link["capacity_bytes_per_ms"] > 0.0]


def _device_mapping(
    merged: Mapping[str, Any], explicit: Mapping[str, Any],
) -> dict[str, Any]:
    derived = _derived_device_fields(merged, explicit)
    capability = explicit.get("capability")
    if capability is None:
        capability = merged.get("capability")
    if capability is None:
        compute_arch = _neutral_capability(derived["compute_peaks"])
    elif not isinstance(capability, Mapping):
        raise ModelingError("custom_hardware_capability_must_be_mapping")
    else:
        compute_arch = dict(capability)
    if compute_arch.get("clock_hz") is None:
        if explicit.get("clock_hz") is not None:
            compute_arch["clock_hz"] = _positive_float(explicit["clock_hz"], "clock_hz")
        elif explicit.get("clock_mhz") is not None:
            compute_arch["clock_hz"] = _positive_float(explicit["clock_mhz"], "clock_mhz") * 1e6
        elif merged.get("clock_hz") is not None:
            compute_arch["clock_hz"] = _positive_float(merged["clock_hz"], "clock_hz")
        elif merged.get("clock_mhz") is not None:
            compute_arch["clock_hz"] = _positive_float(merged["clock_mhz"], "clock_mhz") * 1e6
    aiv = int(merged["aiv_count"])
    aic = int(merged["aic_count"])
    buffers = {
        "l2": int(round(float(merged["l2_mb"]) * 1024**2)),
        "l1": int(round(float(merged["l1_kb"]) * 1024)),
        "l0a": int(round(float(merged["l0a_kb"]) * 1024)),
        "l0b": int(round(float(merged["l0b_kb"]) * 1024)),
        "l0c": int(round(float(merged["l0c_kb"]) * 1024)),
        "ub": int(round(float(merged["ub_kb"]) * 1024)),
    }
    return {
        "memory_bytes": int(round(float(merged["hbm_gb"]) * 1e9)),
        "compute_peaks": derived["compute_peaks"],
        "rates": derived["rates"],
        "buffers": buffers,
        "compute_units": aic,
        "resource_units": {"vector": aiv, "mte1": aic, "mte2": aic,
                           "mte2_aic": aic, "mte2_aiv": aiv, "mte3": aiv,
                           "fixpipe": aic},
        "buffer_units": {"l2": 1, "l1": aic, "l0a": aic, "l0b": aic,
                         "l0c": aic, "ub": aiv},
        "launch_us": float(merged.get("launch_us", 0.0) or 0.0),
        "compute_arch": compute_arch,
    }


def parse_hardware(machine: Mapping[str, Any]) -> Hardware:
    """Parse a materialized machine mapping through the neutral schema."""
    if not isinstance(machine, Mapping):
        raise ModelingError("custom_hardware_machine_must_be_mapping")
    return Hardware.from_mapping(machine)


def build_custom_hardware_machine(
    spec: Mapping[str, Any], *, num_devices: int = 8,
    archetype: str | None = None, **overrides: Any,
) -> dict[str, Any]:
    """Materialize a complete neutral machine mapping from partial key specs."""
    resolved, merged, explicit = _merge_inputs(spec, archetype, overrides)
    _positive_int(num_devices, "num_devices")
    _validate_fields(merged, explicit)
    count = int(num_devices)
    dies_per_package = int(merged["dies_per_package"])
    ranks_per_node = int(merged["ranks_per_node"])
    launched = float(merged.get("launch_us", 0.0) or 0.0)
    device = _device_mapping(merged, explicit)
    device["launch_us"] = launched
    dies = [
        {
            "rank": index,
            "node": index // ranks_per_node,
            "package": index // dies_per_package,
        }
        for index in range(count)
    ]
    links = _materialize_links(
        merged, num_devices=count, dies_per_package=dies_per_package,
        ranks_per_node=ranks_per_node,
    )
    sku = str(merged.get("sku") or _TEMPLATE_SKU)
    hardware_key = str(merged.get("hardware_key") or sku)
    assumptions = {
        "custom_hardware": f"neutral structural template {resolved}",
        "units": "hbm_gb is decimal GB; *_bw_gbps is decimal GB/s; "
                 "*_tflops is 10^12 FLOP/s; latencies are microseconds",
        "derived": "cube_fp8 defaults to twice cube_fp16; MTE supplies scale "
                   "with HBM when omitted",
        "capability": "native modes cover exactly the synthetic peaks unless "
                      "the caller supplies an explicit capability",
        "topology": "links are declared shared group capacities; equal-member "
                    "routing prefers intra-node before inter-node",
    }
    declared_assumptions = merged.get("assumptions")
    if isinstance(declared_assumptions, Mapping):
        assumptions.update({str(key): str(value)
                            for key, value in declared_assumptions.items()})
    machine: dict[str, Any] = {
        "name": sku,
        "sku": sku,
        "hardware_key": hardware_key,
        "arch_family": str(merged["arch_family"]),
        "architecture": str(merged["arch_family"]),
        "archetype": resolved,
        "num_devices": count,
        "ranks_per_node": ranks_per_node,
        "dies_per_package": dies_per_package,
        "device": device,
        "dies": dies,
        "links": links,
        "assumptions": assumptions,
    }
    superpod = merged.get("superpod_size", merged.get("max_dies"))
    if superpod is not None:
        machine["superpod_size"] = int(superpod)
        machine["max_dies"] = int(superpod)
    parse_hardware(machine)
    return machine


def _canonical_for(name: Any) -> str | None:
    if not isinstance(name, str):
        return None
    stripped = name.strip()
    for key in (stripped, stripped.lower()):
        canonical = _CUSTOM_SKU_CANONICAL.get(key)
        if canonical is not None:
            return canonical
    return None


def _remove_canonical(canonical: str) -> bool:
    removed = False
    for key, mapped in list(_CUSTOM_SKU_CANONICAL.items()):
        if mapped == canonical:
            _CUSTOM_SKU_CANONICAL.pop(key, None)
            _CUSTOM_SKU_REGISTRY.pop(key, None)
            removed = True
    if _CUSTOM_SKU_REGISTRY.pop(canonical, None) is not None:
        removed = True
    _CUSTOM_SKU_CANONICAL.pop(canonical, None)
    return removed


def _extract_aliases(spec: Mapping[str, Any], aliases: Sequence[str]) -> tuple[str, ...]:
    declared = spec.get("aliases", ())
    if declared is None:
        declared = ()
    if isinstance(declared, str) or not isinstance(declared, Sequence):
        raise ModelingError(f"custom_hardware_aliases_invalid:{declared}")
    combined = list(aliases or ()) + list(declared)
    result, seen = [], set()
    for alias in combined:
        if not isinstance(alias, str) or not alias.strip():
            raise ModelingError(f"custom_hardware_alias_invalid:{alias}")
        value = alias.strip()
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _normalize_profile(
    sku: str, spec: Mapping[str, Any], *, archetype: str | None,
    aliases: Sequence[str],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    if not isinstance(sku, str) or not sku.strip():
        raise ModelingError(f"custom_hardware_sku_invalid:{sku}")
    if not isinstance(spec, Mapping):
        raise ModelingError(f"custom_hardware_spec_must_be_mapping:{sku}")
    raw = _plain_mapping(spec, f"spec:{sku}")
    raw.pop("aliases", None)
    resolved, merged, explicit = _merge_inputs(
        raw, archetype, {}, prefer_spec_archetype=True,
    )
    _validate_fields(merged, explicit)
    profile = dict(raw)
    profile["sku"] = sku.strip()
    profile["archetype"] = resolved
    return profile, _extract_aliases(spec, aliases)


def _store_profile(profile: Mapping[str, Any], aliases: Sequence[str]) -> None:
    canonical = str(profile["sku"])
    _remove_canonical(canonical)
    stored = deepcopy(dict(profile))
    _CUSTOM_SKU_REGISTRY[canonical] = stored
    _CUSTOM_SKU_CANONICAL[canonical] = canonical
    keys = {canonical, canonical.lower()}
    for alias in aliases:
        keys.add(str(alias))
        keys.add(str(alias).lower())
    for key in keys:
        if key:
            _CUSTOM_SKU_REGISTRY[key] = stored
            _CUSTOM_SKU_CANONICAL[key] = canonical


def register_custom_hardware_sku(
    sku: str, spec: Mapping[str, Any], *,
    archetype: str = "ascend_dual_die_ub", aliases: Sequence[str] = (),
) -> dict[str, Any]:
    """Validate, normalize, and register one private custom SKU."""
    profile, all_aliases = _normalize_profile(
        sku, spec, archetype=archetype, aliases=aliases,
    )
    _store_profile(profile, all_aliases)
    return deepcopy(profile)


def unregister_custom_hardware_sku(sku: str) -> bool:
    """Remove one canonical SKU and every alias registered for it."""
    canonical = _canonical_for(sku)
    if canonical is None:
        return False
    return _remove_canonical(canonical)


def clear_custom_hardware_skus() -> None:
    """Drop every in-memory custom SKU (environment files remain untouched)."""
    _CUSTOM_SKU_REGISTRY.clear()
    _CUSTOM_SKU_CANONICAL.clear()


def _read_sku_file(path: str | Path) -> dict[str, Mapping[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise ModelingError(f"custom_hardware_sku_file_not_found:{target}")
    if target.stat().st_size > _MAX_METADATA_BYTES:
        raise ModelingError(f"custom_hardware_sku_file_too_large:{target}")
    try:
        with target.open(encoding="utf-8") as stream:
            data = json.load(stream)
    except TimeoutError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelingError(f"custom_hardware_sku_file_unreadable:{target}:{error}") from error
    if not isinstance(data, Mapping):
        raise ModelingError(f"custom_hardware_sku_file_must_be_mapping:{target}")
    entries: Any = data["skus"] if "skus" in data else data
    if not isinstance(entries, Mapping):
        raise ModelingError(f"custom_hardware_skus_record_must_be_mapping:{target}")
    result: dict[str, Mapping[str, Any]] = {}
    for name, raw in entries.items():
        if str(name).startswith("_"):
            continue
        if not isinstance(name, str) or not name.strip():
            raise ModelingError(f"custom_hardware_sku_name_invalid:{name}")
        if not isinstance(raw, Mapping):
            raise ModelingError(f"custom_hardware_sku_spec_invalid:{name}")
        result[name.strip()] = raw
    return result


def _load_file(
    path: str | Path, *, overwrite: bool,
) -> dict[str, dict[str, Any]]:
    entries = _read_sku_file(path)
    loaded: dict[str, dict[str, Any]] = {}
    for name, raw in entries.items():
        if not overwrite and _canonical_for(name) is not None:
            continue
        profile, all_aliases = _normalize_profile(
            name, raw, archetype=None, aliases=(),
        )
        _store_profile(profile, all_aliases)
        loaded[name] = deepcopy(profile)
    return loaded


def load_custom_hardware_skus_from_file(path: str | Path) -> dict[str, dict[str, Any]]:
    """Register every SKU in a private JSON file and return the loaded map."""
    return _load_file(path, overwrite=True)


def _maybe_load_environment_skus() -> None:
    path = os.environ.get("MINDIE_SIM_CUSTOM_SKUS")
    if not path:
        return
    try:
        _load_file(path, overwrite=False)
    except TimeoutError:
        raise
    except (OSError, ValueError, TypeError):
        # A missing or malformed optional environment file must not break an explicit registry lookup.
        return


def list_custom_hardware_skus() -> dict[str, dict[str, Any]]:
    """Return all registered names and aliases, including env-file discovery."""
    _maybe_load_environment_skus()
    return deepcopy(_CUSTOM_SKU_REGISTRY)


def resolve_custom_hardware_sku(
    sku_or_alias: str,
) -> dict[str, Any] | None:
    """Resolve a private SKU name or alias, including env-file discovery."""
    if not isinstance(sku_or_alias, str) or not sku_or_alias.strip():
        return None
    _maybe_load_environment_skus()
    canonical = _canonical_for(sku_or_alias)
    if canonical is None:
        return None
    profile = _CUSTOM_SKU_REGISTRY.get(canonical)
    return deepcopy(profile) if profile is not None else None


def compose_custom_case(
    model: str | Mapping[str, Any],
    custom_sku: str | Mapping[str, Any],
    *,
    tp: int = 8, dp: int = 1, pp: int = 1, ep: int = 1, cp: int = 1,
    pcp: int = 1, prompt_tokens: int = 1024, output_tokens: int = 128,
    num_requests: int = 1, max_num_seqs: int = 32,
    max_num_batched_tokens: int = 4096, max_model_len: int = 8192,
    async_scheduling: bool = False, quantization: str | None = None,
    **hardware_overrides: Any,
) -> dict[str, Any]:
    """Compose and validate a complete runnable case for a private custom SKU."""
    if isinstance(custom_sku, Mapping):
        spec: Mapping[str, Any] = custom_sku
    elif isinstance(custom_sku, str):
        resolved = resolve_custom_hardware_sku(custom_sku)
        if resolved is None:
            raise ModelingError(f"unknown_custom_hardware_sku:{custom_sku}")
        spec = resolved
    else:
        raise ModelingError(f"custom_hardware_sku_or_mapping_required:{custom_sku}")
    if "num_devices" in hardware_overrides:
        raise ModelingError(
            "custom_hardware_num_devices_is_derived_from_parallelism"
        )
    num_devices = int(dp) * int(pp) * int(tp) * int(pcp)
    machine = build_custom_hardware_machine(
        spec, num_devices=num_devices, **hardware_overrides,
    )
    # ``compose_case`` validates hardware SKUs against public profiles. Feed it
    # the already-built neutral resources under a registered placeholder name,
    # then replace the machine with the private SKU identity.
    compose_hardware = {**machine, "sku": "Ascend910_9392"}
    from mindie_sim.compose import compose_case

    case = compose_case(
        model=model,
        hardware=compose_hardware,
        tp=int(tp), dp=int(dp), ep=int(ep), pp=int(pp), pcp=int(pcp),
        dcp=int(cp), num_devices=num_devices,
        quant=quantization,
        prompt_tokens=int(prompt_tokens), output_tokens=int(output_tokens),
        concurrency=int(num_requests),
        max_num_seqs=int(max_num_seqs),
        max_num_batched_tokens=int(max_num_batched_tokens),
        max_model_len=int(max_model_len),
        async_scheduling=bool(async_scheduling),
    )
    case["machine"] = machine
    parse_case(case)
    return case


__all__ = [
    "HARDWARE_ARCHETYPE_TEMPLATES",
    "export_hardware_sku_template",
    "build_custom_hardware_machine",
    "parse_hardware",
    "register_custom_hardware_sku",
    "unregister_custom_hardware_sku",
    "clear_custom_hardware_skus",
    "list_custom_hardware_skus",
    "load_custom_hardware_skus_from_file",
    "resolve_custom_hardware_sku",
    "compose_custom_case",
    "derive_hardware_sku",
]
