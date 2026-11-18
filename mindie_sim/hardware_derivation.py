"""Pure one-pass derivation of a parameterized hardware SKU.

[Role]
  Produce a new immutable ``Hardware`` declaration from a registered base SKU or
  an explicit declaration, scaling only the resources whose declared ownership
  says they are aggregate per-owner supplies.  No registry installation, file
  write or device access occurs here.

[Invariants]
  * Device count, ranks, nodes, packages, links, topology, capabilities, clocks,
    buffers-per-owner and dtype names are preserved unless explicitly overridden.
  * Native matrix peaks scale by ``new_AIC/base_AIC``; AIC-owned owner counts,
    supplies, buffer unit counts and instruction services scale with AIC.
    Vector/AIV-owned supplies and UB unit counts scale with AIV.  Per-core
    buffers, HBM bandwidth and host/network/fixed control do not auto-scale.
  * Linked instruction rates are removed before materialization so the schema
    recomputes them once from the scaled owner declarations; explicit independent
    overrides are preserved and recorded.
  * A5-capable derivatives receive the validated ``mxfp4 -> fp8_e4m3`` supply
    alias when the base declares native E4M3 but no native MXFP4.

[Timeouts & Locks]
  Bounded in-memory mapping work; no I/O, locks or device contexts.

[Side-Effect Policy]
  Returns a new object and performs no registry or filesystem mutation.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.hardware_efficiency import (
    AXIS_AIC_COUNT, AXIS_HBM_PEAK, EfficiencyCurve, apply_mte2_ceiling,
    coerce_curve,
)


ZERO: float = 0.0
ONE: float = 1.0
FIRST = 0
DEFAULT_DEVICE_COUNT = 1
TOLERANCE: float = 1e-9
AIC_OWNER_UNITS = frozenset({"mte1", "mte2_aic", "fixpipe"})
AIV_OWNER_UNITS = frozenset({
    "vector", "mte2_aiv", "mte3", "pipe_s_issue", "pipe_v_issue",
    "pipe_v_lanes", "reg_alu_lanes", "reg_data_writes", "reg_reduce_lanes",
    "reg_shuffle_lanes", "reg_special_lanes", "reg_ub_accesses",
    "reg_ub_stores", "scalar_store_issue", "ub_read_bytes", "ub_write_bytes",
})
AIC_RATE_KEYS = frozenset({"mte1", "mte2_aic", "fixpipe"})
AIV_RATE_KEYS = frozenset({
    "vector", "mte2_aiv", "mte3", "pipe_s_issue", "pipe_v_issue",
    "pipe_v_lanes", "reg_alu_lanes", "reg_data_writes", "reg_reduce_lanes",
    "reg_shuffle_lanes", "reg_special_lanes", "reg_ub_accesses",
    "reg_ub_stores", "scalar_store_issue", "ub_read_bytes", "ub_write_bytes",
})
#: Service modes whose declared rate is one HBM read stream.  They scale with
#: ``hbm_peak_bytes_per_s`` even though their resource name is not ``hbm``.
DRAM_BOUND_MODES = frozenset({"mte2_aic_dram", "mte2_aiv_dram"})
AIC_BUFFER_UNITS = frozenset({"l1", "l0a", "l0b", "l0c"})
AIV_BUFFER_UNITS = frozenset({"ub"})
A5_FAMILY = "A5"
MXFP4_ALIAS = "mxfp4"
FP8_E4M3 = "fp8_e4m3"
FP8_GENERIC = "fp8"
DEFAULT_SKU = "derived-sku"
HARDWARE_KEYS = (
    "aic_count", "aiv_count", "memory_bytes", "hbm_peak_bytes_per_s",
    "hbm_efficiency", "mte2_ceiling_bytes_per_s", "mte2_efficiency",
)


def _load_base(base: Hardware | Mapping[str, Any] | str) -> Hardware:
    if isinstance(base, Hardware):
        return base
    if isinstance(base, Mapping):
        return Hardware.from_mapping(dict(base))
    if isinstance(base, str) and base:
        return Hardware.from_mapping({"backend": "ascend", "sku": base,
                                      "num_devices": DEFAULT_DEVICE_COUNT})
    raise ModelingError(f"hardware_derivation_base_invalid:{base!r}")


def _finite_positive(value: Any, reason: str) -> float:
    if isinstance(value, bool):
        raise ModelingError(reason)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError(reason) from exc
    if not math.isfinite(result) or result <= ZERO:
        raise ModelingError(reason)
    return result


def _positive_count(value: Any, reason: str) -> int:
    if isinstance(value, bool):
        raise ModelingError(reason)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError(reason) from exc
    if result < ONE or float(value) != float(result):
        raise ModelingError(reason)
    return result


def _exact_scaled(value: int, factor: float, reason: str) -> int:
    scaled = float(value) * float(factor)
    nearest = int(round(scaled))
    if abs(scaled - nearest) > TOLERANCE * max(ONE, abs(scaled)):
        raise ModelingError(reason)
    if nearest < ONE:
        raise ModelingError(reason)
    return nearest


def _resolve_efficiency(
    value: Any, *, axis_hint: str, aic_count: int, hbm_peak: float,
) -> tuple[float, dict[str, str]]:
    if value is None:
        return ONE, {}
    curve = coerce_curve(value, axis_hint=axis_hint)
    assert curve is not None
    if curve.axis == AXIS_AIC_COUNT:
        x = float(aic_count)
    elif curve.axis == AXIS_HBM_PEAK:
        x = float(hbm_peak)
    else:  # pragma: no cover - guarded by EfficiencyCurve
        raise ModelingError(f"efficiency_curve_axis_unsupported:{curve.axis}")
    resolved = curve.resolve(x)
    provenance = {
        "derivation.efficiency_axis": curve.axis,
        "derivation.efficiency_x": repr(x),
        "derivation.efficiency_y": repr(resolved),
        "derivation.efficiency_knots": repr([list(knot) for knot in curve.knots]),
    }
    if curve.provenance:
        provenance["derivation.efficiency_provenance"] = curve.provenance
    return resolved, provenance


def _scale_owner_units(units: dict[str, Any], s_c: float, s_v: float) -> dict[str, Any]:
    result = dict(units)
    for name, value in list(result.items()):
        if name in AIC_OWNER_UNITS:
            result[name] = max(ONE, int(round(int(value) * s_c)))
        elif name in AIV_OWNER_UNITS:
            result[name] = max(ONE, int(round(int(value) * s_v)))
    return result


def _scale_rates(rates: dict[str, Any], s_c: float, s_v: float) -> dict[str, Any]:
    result = dict(rates)
    for name in list(result):
        if name in AIC_RATE_KEYS:
            result[name] = float(result[name]) * s_c
        elif name in AIV_RATE_KEYS and name in result:
            result[name] = float(result[name]) * s_v
    return result


def _strip_derived_instruction_rates(
    data: dict[str, Any], arch: Any, assumptions: dict[str, str],
) -> None:
    """Let the schema recompute aggregate per-owner supplies exactly once."""
    rates = dict(data.get("rates", {}))
    units = dict(data.get("resource_units", {}))
    services = {item.name: item for item in getattr(arch, "instruction_services", ())}
    for name, service in services.items():
        if name not in rates:
            continue
        count = int(units.get(service.owner_axis, data.get("compute_units", DEFAULT_DEVICE_COUNT)) or DEFAULT_DEVICE_COUNT)
        canonical = float(service.per_owner_rate) * count
        current = float(rates[name])
        if service.clock_hz is not None or abs(current - canonical) <= TOLERANCE * max(ONE, abs(canonical)):
            rates.pop(name)
            assumptions[f"derivation.recomputed_service.{name}"] = "from_owner_supply"
        else:
            assumptions[f"derivation.explicit_service_override.{name}"] = "preserved"
    data["rates"] = rates
    modes = {name: dict(mode) for name, mode in data.get("service_modes", {}).items()}
    for name, mode in list(modes.items()):
        service = services.get(name)
        if service is None or "rate" not in mode:
            continue
        count = int(units.get(service.owner_axis, data.get("compute_units", DEFAULT_DEVICE_COUNT)) or DEFAULT_DEVICE_COUNT)
        canonical = float(service.per_owner_rate) * count
        if service.clock_hz is not None or abs(float(mode["rate"]) - canonical) <= TOLERANCE * max(ONE, abs(canonical)):
            mode.pop("rate")
            assumptions[f"derivation.recomputed_mode.{name}"] = "from_owner_supply"
        else:
            assumptions[f"derivation.explicit_mode_override.{name}"] = "preserved"
    data["service_modes"] = modes


def derive_hardware_sku(
    base: Hardware | Mapping[str, Any] | str, *, sku: str,
    aic_count: int | None = None, aiv_count: int | None = None,
    memory_bytes: int | None = None,
    hbm_peak_bytes_per_s: float | None = None,
    hbm_efficiency: float | EfficiencyCurve | None = None,
    mte2_ceiling_bytes_per_s: float | None = None,
    mte2_efficiency: float | EfficiencyCurve | None = None,
    collective_protocols: Mapping[str, str] | None = None,
) -> Hardware:
    """Derive one immutable SKU declaration from ``base``."""
    if not isinstance(sku, str) or not sku.strip():
        raise ModelingError("hardware_derivation_sku_required")
    base_hardware = _load_base(base)
    if not base_hardware.dies:
        raise ModelingError("hardware_derivation_requires_devices")
    if (aic_count is not None or aiv_count is not None) and not base_hardware.homogeneous:
        raise ModelingError(
            "hardware_derivation_requires_per_device_core_mapping")
    data = base_hardware.to_dict()
    devices = list(data.get("devices", ()))
    core_override = aic_count is not None or aiv_count is not None
    base_aic = int(devices[FIRST].get("compute_units", ONE) or ONE)
    base_aiv = int((devices[FIRST].get("resource_units") or {}).get("vector", base_aic))
    new_aic = base_aic if aic_count is None else _positive_count(
        aic_count, "hardware_derivation_aic_count_invalid")
    s_c = float(new_aic) / float(base_aic) if core_override else float(ONE)
    if aiv_count is None:
        new_aiv = (_exact_scaled(base_aiv, s_c,
                                 "hardware_derivation_aiv_count_not_integral")
                   if core_override else base_aiv)
    else:
        new_aiv = _positive_count(aiv_count, "hardware_derivation_aiv_count_invalid")
    s_v = float(new_aiv) / float(base_aiv) if core_override else float(ONE)
    assumptions: dict[str, str] = dict(data.get("assumptions", {}))
    assumptions.update({
        "derivation.base": base_hardware.sku or base_hardware.name,
        "derivation.sku": sku,
        "derivation.aic_scale": repr(s_c),
        "derivation.aiv_scale": repr(s_v),
    })
    if core_override or base_hardware.homogeneous:
        assumptions["derivation.aic_count"] = str(new_aic)
        assumptions["derivation.aiv_count"] = str(new_aiv)
    raw_peak = None if hbm_peak_bytes_per_s is None else _finite_positive(
        hbm_peak_bytes_per_s, "hardware_derivation_hbm_peak_invalid")
    ceiling = None if mte2_ceiling_bytes_per_s is None else _finite_positive(
        mte2_ceiling_bytes_per_s, "hardware_derivation_mte2_ceiling_invalid")
    hbm_eta, hbm_provenance = _resolve_efficiency(
        hbm_efficiency, axis_hint=AXIS_HBM_PEAK, aic_count=new_aic,
        hbm_peak=(raw_peak if raw_peak is not None
                  else float(devices[FIRST].get("rates", {}).get("hbm", ONE))))
    assumptions.update(hbm_provenance)
    if hbm_efficiency is not None:
        assumptions["derivation.hbm_efficiency"] = repr(hbm_eta)
    mte2_eta, mte2_provenance = _resolve_efficiency(
        mte2_efficiency, axis_hint=AXIS_AIC_COUNT, aic_count=new_aic,
        hbm_peak=(raw_peak if raw_peak is not None else ONE))
    assumptions.update({key.replace("efficiency", "mte2_efficiency", ONE): value
                        for key, value in mte2_provenance.items()})
    for device_index, device in enumerate(devices):
        device_arch = base_hardware.dies[device_index].compute_arch
        dev_aic = int(device.get("compute_units", ONE) or ONE)
        dev_aiv = int((device.get("resource_units") or {}).get("vector", dev_aic))
        device["compute_units"] = new_aic if core_override else dev_aic
        units = _scale_owner_units(
            dict(device.get("resource_units", {})), s_c, s_v)
        units["vector"] = new_aiv if core_override else dev_aiv
        device["resource_units"] = units
        buffers = dict(device.get("buffer_units", {}))
        for name, value in list(buffers.items()):
            if name in AIC_BUFFER_UNITS:
                buffers[name] = max(ONE, int(round(int(value) * s_c)))
            elif name in AIV_BUFFER_UNITS:
                buffers[name] = max(ONE, int(round(int(value) * s_v)))
        device["buffer_units"] = buffers
        aliases = dict(device.get("compute_peak_aliases", {}))
        alias_keys = set(aliases)
        peaks = {name: (float(value) * s_c if name not in alias_keys else float(value))
                 for name, value in device.get("compute_peaks", {}).items()}
        for name, target in list(aliases.items()):
            if target in peaks:
                peaks[name] = peaks[target]
        if (base_hardware.arch_family == A5_FAMILY
                and MXFP4_ALIAS not in peaks and MXFP4_ALIAS not in aliases):
            target = FP8_E4M3 if FP8_E4M3 in peaks else (
                FP8_GENERIC if FP8_GENERIC in peaks else None)
            if target is not None:
                aliases[MXFP4_ALIAS] = target
                peaks[MXFP4_ALIAS] = peaks[target]
                assumptions["derivation.mxfp4_supply_alias"] = target
        device["compute_peak_aliases"] = aliases
        device["compute_peaks"] = peaks
        rates = _scale_rates(dict(device.get("rates", {})), s_c, s_v)
        if memory_bytes is not None:
            capacity = _positive_count(memory_bytes, "hardware_derivation_memory_invalid")
            device["memory_bytes"] = capacity
        if raw_peak is not None:
            rates["hbm"] = raw_peak * hbm_eta
            assumptions["derivation.hbm_peak_bytes_per_s"] = repr(raw_peak)
            assumptions["derivation.hbm_rate_bytes_per_s"] = repr(rates["hbm"])
        elif hbm_efficiency is not None and "hbm" in rates:
            rates["hbm"] = float(rates["hbm"]) * hbm_eta
            assumptions["derivation.hbm_rate_bytes_per_s"] = repr(rates["hbm"])
        if mte2_efficiency is not None:
            for name in ("mte2_aic", "mte2_aiv"):
                if name in rates:
                    rates[name] = float(rates[name]) * mte2_eta
        if ceiling is not None:
            rates = apply_mte2_ceiling(rates, ceiling)
            assumptions["derivation.mte2_ceiling_bytes_per_s"] = repr(ceiling)
        if "mte2_aic" in rates and "mte2_aiv" in rates:
            rates["mte2"] = min(float(rates["mte2_aic"]), float(rates["mte2_aiv"]))
        original_hbm = dict(base_hardware.dies[device_index].rates)["hbm"]
        hbm_scale = float(rates["hbm"]) / float(original_hbm)
        # A service mode's duration is work / mode_rate.  Scaling only the
        # underlying HBM resource would leave an explicit mode rate unchanged,
        # so carry the die's effective bandwidth ratio exactly once.
        for name, mode in device.get("service_modes", {}).items():
            if ((mode.get("resource") == "hbm" or name in DRAM_BOUND_MODES)
                    and mode.get("rate") is not None):
                mode["rate"] = float(mode["rate"]) * hbm_scale
        device["rates"] = rates
        _strip_derived_instruction_rates(device, device_arch, assumptions)
        device["sku"] = sku
    data["name"] = sku
    data["sku"] = sku
    data["hardware_key"] = sku
    data["assumptions"] = assumptions
    if collective_protocols is not None:
        data["collective_protocols"] = dict(collective_protocols)
    return Hardware.from_mapping(data)


def derived_machine(machine: Any, overrides: Mapping[str, Any]) -> Any:
    """Apply only explicitly declared hardware overrides to a base machine.

    An omitted field keeps the parsed machine declaration.  A nonempty scalar
    override set derives one new immutable ``Hardware``; it never installs a
    registry entry or mutates the caller's machine.
    """
    kwargs = {key: overrides[key] for key in HARDWARE_KEYS if key in overrides}
    if not kwargs:
        return machine
    sku = str(overrides.get("sku") or getattr(machine, "sku", "") or DEFAULT_SKU)
    return derive_hardware_sku(machine, sku=sku, **kwargs)
