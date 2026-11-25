"""Hardware SKU derivation and utilization efficiency curve interpolation.

[Role] Define monotone efficiency curves and derive scaled/what-if hardware
  SKU profiles from calibrated baseline Ascend dies.
[Invariants] Derived hardware preserves physical unit consistency and non-negative rates.
[Timeouts & Locks] Pure arithmetic derivation; no I/O or locks.
[Side-Effect Policy] Pure computation."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware


AXIS_AIC_COUNT = "aic_count"
AXIS_HBM_PEAK = "hbm_peak_bytes_per_s"
AXES = frozenset({AXIS_AIC_COUNT, AXIS_HBM_PEAK})
EXTRAPOLATION_CLAMP = "clamp"
EXTRAPOLATIONS = frozenset({EXTRAPOLATION_CLAMP})


@dataclass(frozen=True)
class EfficiencyCurve:
    """One declared efficiency response ``y(x)`` with linear interpolation."""

    axis: str
    knots: tuple[tuple[float, float], ...]
    extrapolation: str = EXTRAPOLATION_CLAMP
    provenance: str = ""

    def __post_init__(self) -> None:
        axis = str(self.axis).strip().lower()
        if axis not in AXES:
            raise ModelingError(f"efficiency_curve_axis_unsupported:{self.axis}")
        extrapolation = str(self.extrapolation).strip().lower()
        if extrapolation not in EXTRAPOLATIONS:
            raise ModelingError(
                f"efficiency_curve_extrapolation_unsupported:{self.extrapolation}")
        knots = tuple((float(x), float(y)) for x, y in self.knots)
        if not knots:
            raise ModelingError("efficiency_curve_requires_knots")
        previous: float | None = None
        for x, y in knots:
            if not math.isfinite(x) or x < 0.0:
                raise ModelingError(f"efficiency_curve_knot_x_invalid:{x}")
            if not math.isfinite(y) or y <= 0.0 or y > 1.0:
                raise ModelingError(f"efficiency_curve_knot_y_invalid:{y}")
            if previous is not None and x <= previous:
                raise ModelingError("efficiency_curve_knots_not_increasing")
            previous = x
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "extrapolation", extrapolation)
        object.__setattr__(self, "knots", knots)

    def resolve(self, x: float) -> float:
        """Linear interpolation with explicitly declared clamp extrapolation."""
        value = float(x)
        if not math.isfinite(value) or value < 0.0:
            raise ModelingError(f"efficiency_curve_x_invalid:{x}")
        knots = self.knots
        if value <= knots[0][0]:
            if self.extrapolation == EXTRAPOLATION_CLAMP:
                return knots[0][1]
            raise ModelingError(f"efficiency_curve_below_range:{value}")
        if value >= knots[-1][0]:
            if self.extrapolation == EXTRAPOLATION_CLAMP:
                return knots[-1][1]
            raise ModelingError(f"efficiency_curve_above_range:{value}")
        for index in range(1, len(knots)):
            left_x, left_y = knots[index - 1]
            right_x, right_y = knots[index]
            if value <= right_x:
                span = right_x - left_x
                if span <= 0.0:
                    return right_y
                return left_y + (right_y - left_y) * (value - left_x) / span
        return knots[-1][1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "knots": [list(knot) for knot in self.knots],
            "extrapolation": self.extrapolation,
            **({"provenance": self.provenance} if self.provenance else {}),
        }


def coerce_curve(value: Any, *, axis_hint: str = "") -> EfficiencyCurve | None:
    """Accept an ``EfficiencyCurve``, a mapping declaration, or ``None``."""
    if value is None:
        return None
    if isinstance(value, EfficiencyCurve):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # A scalar efficiency is a constant curve, not a fitted trend.
        return EfficiencyCurve(axis_hint or AXIS_AIC_COUNT,
                               ((0.0, float(value)),))
    if isinstance(value, dict):
        axis = value.get("axis", axis_hint)
        knots = value.get("knots", ())
        return EfficiencyCurve(
            axis=axis, knots=tuple(tuple(item) for item in knots),
            extrapolation=value.get("extrapolation", EXTRAPOLATION_CLAMP),
            provenance=str(value.get("provenance", "")),
        )
    raise ModelingError(f"efficiency_curve_invalid:{value!r}")


def apply_mte2_ceiling(
    supplies: dict[str, float], ceiling: float | None,
) -> dict[str, float]:
    """Scale split MTE2 supplies under one shared coupling ceiling."""
    result = dict(supplies)
    if ceiling is None:
        return result
    cap = float(ceiling)
    if not math.isfinite(cap) or cap <= 0.0:
        raise ModelingError(f"mte2_ceiling_invalid:{ceiling}")
    keys = [name for name in ("mte2_aic", "mte2_aiv") if name in result]
    if not keys:
        if "mte2" in result:
            result["mte2"] = min(result["mte2"], cap)
        return result
    total = sum(max(0.0, float(result[name])) for name in keys)
    if total > cap:
        factor = cap / total
        for name in keys:
            result[name] = float(result[name]) * factor
    if "mte2" in result:
        result["mte2"] = min(float(result["mte2"]), cap)
    return result


DEFAULT_DEVICE_COUNT = 1
TOLERANCE: float = 1e-9
HBM_CACHELINE_SERVICE = "gm_hbm_cacheline_128b"
HBM_CACHELINE_BYTES = 128
AIC_OWNER_UNITS = frozenset({"mte1", "mte2_aic", "fixpipe"})
AIV_OWNER_UNITS = frozenset({
    "vector", "mte2_aiv", "mte3", "pipe_s_issue", "pipe_v_issue",
    "pipe_v_lanes", "reg_alu_lanes", "reg_data_writes", "reg_reduce_lanes",
    "reg_shuffle_lanes", "reg_special_lanes", "reg_ub_accesses",
    "reg_ub_stores", "scalar_store_issue", "ub_read_bytes", "ub_write_bytes",
    HBM_CACHELINE_SERVICE,
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
    if not math.isfinite(result) or result <= 0.0:
        raise ModelingError(reason)
    return result


def _positive_count(value: Any, reason: str) -> int:
    if isinstance(value, bool):
        raise ModelingError(reason)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError(reason) from exc
    if result < 1 or float(value) != float(result):
        raise ModelingError(reason)
    return result


def _exact_scaled(value: int, factor: float, reason: str) -> int:
    scaled = float(value) * float(factor)
    nearest = int(round(scaled))
    if abs(scaled - nearest) > TOLERANCE * max(1.0, abs(scaled)):
        raise ModelingError(reason)
    if nearest < 1:
        raise ModelingError(reason)
    return nearest


def _resolve_efficiency(
    value: Any, *, axis_hint: str, aic_count: int, hbm_peak: float,
) -> tuple[float, dict[str, str]]:
    if value is None:
        return 1.0, {}
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
            result[name] = max(1, int(round(int(value) * s_c)))
        elif name in AIV_OWNER_UNITS:
            result[name] = max(1, int(round(int(value) * s_v)))
    return result


def _scale_rates(rates: dict[str, Any], s_c: float, s_v: float) -> dict[str, Any]:
    result = dict(rates)
    for name in list(result):
        if name in AIC_RATE_KEYS:
            result[name] = float(result[name]) * s_c
        elif name in AIV_RATE_KEYS and name in result:
            result[name] = float(result[name]) * s_v
    return result


def _base_die_hbm_peak(
    device: Mapping[str, Any], base_assumptions: Mapping[str, str],
) -> float:
    current_hbm = float((device.get("rates") or {}).get("hbm", 1.0))
    raw_peak = base_assumptions.get("derivation.hbm_peak_bytes_per_s")
    raw_rate = base_assumptions.get("derivation.hbm_rate_bytes_per_s")
    if raw_peak is not None and raw_rate is not None:
        try:
            peak_val = float(raw_peak)
            rate_val = float(raw_rate)
        except ValueError:
            return current_hbm
        if (math.isfinite(peak_val) and peak_val > 0.0
                and math.isclose(current_hbm, rate_val)):
            return peak_val
    return current_hbm


def _strip_derived_instruction_rates(
    data: dict[str, Any],
    base_die: Any,
    assumptions: dict[str, str],
    *,
    preserved_rates: frozenset[str] = frozenset(),
) -> None:
    """Let the schema recompute aggregate per-owner supplies exactly once."""
    rates = dict(data.get("rates", {}))
    units = dict(data.get("resource_units", {}))
    base_rates = dict(base_die.rates)
    base_units = dict(base_die.resource_units)
    linked_overrides = frozenset(base_die.linked_rate_overrides) | preserved_rates
    unit_overrides = frozenset(base_die.instruction_unit_overrides)
    for service in getattr(base_die.compute_arch, "instruction_services", ()):
        name = service.name
        if service.resource is not None:
            continue
        base_axis_owners = int(
            base_units.get(service.owner_axis, base_die.compute_units) or DEFAULT_DEVICE_COUNT
        )
        if name in units and name not in unit_overrides and int(base_units.get(name, 0)) == base_axis_owners:
            units.pop(name)
        if name not in rates:
            continue
        if name in linked_overrides:
            assumptions[f"derivation.explicit_service_override.{name}"] = "preserved"
            continue
        base_canonical = float(service.per_owner_rate) * base_axis_owners
        new_owners = int(
            units.get(service.owner_axis, data.get("compute_units", DEFAULT_DEVICE_COUNT)) or DEFAULT_DEVICE_COUNT
        )
        new_canonical = float(service.per_owner_rate) * new_owners
        base_current = float(base_rates.get(name, rates[name]))
        current = float(rates[name])
        if (abs(base_current - base_canonical) <= TOLERANCE * max(1.0, abs(base_canonical))
                or abs(current - new_canonical) <= TOLERANCE * max(1.0, abs(new_canonical))):
            rates.pop(name)
            assumptions[f"derivation.recomputed_service.{name}"] = "from_owner_supply"
        else:
            assumptions[f"derivation.explicit_service_override.{name}"] = "preserved"
    data["resource_units"] = units
    data["rates"] = rates


def _derive_single_device(
    device: dict[str, Any],
    base_die: Any,
    *,
    arch_family: str,
    sku: str,
    core_override: bool,
    new_aic: int,
    new_aiv: int,
    s_c: float,
    s_v: float,
    memory_bytes: int | None,
    raw_peak: float | None,
    die_hbm_peak: float,
    record_hbm_assumptions: bool,
    hbm_efficiency: Any,
    hbm_eta: float,
    mte2_efficiency: Any,
    mte2_eta: float,
    ceiling: float | None,
    assumptions: dict[str, str],
) -> None:
    """Apply scaled core, buffer, peak, HBM/MTE2 rate and cacheline derivation to one die."""
    dev_aic = int(device.get("compute_units", 1) or 1)
    dev_aiv = int((device.get("resource_units") or {}).get("vector", dev_aic))
    device["compute_units"] = new_aic if core_override else dev_aic
    units = _scale_owner_units(dict(device.get("resource_units", {})), s_c, s_v)
    units["vector"] = new_aiv if core_override else dev_aiv
    buffers = dict(device.get("buffer_units", {}))
    for name, value in list(buffers.items()):
        if name in AIC_BUFFER_UNITS:
            buffers[name] = max(1, int(round(int(value) * s_c)))
        elif name in AIV_BUFFER_UNITS:
            buffers[name] = max(1, int(round(int(value) * s_v)))
    device["buffer_units"] = buffers

    aliases = dict(device.get("compute_peak_aliases", {}))
    alias_keys = set(aliases)
    peaks = {
        name: (float(value) * s_c if name not in alias_keys else float(value))
        for name, value in device.get("compute_peaks", {}).items()
    }
    for name, target in list(aliases.items()):
        if target in peaks:
            peaks[name] = peaks[target]
    if arch_family == A5_FAMILY and MXFP4_ALIAS not in peaks and MXFP4_ALIAS not in aliases:
        target = FP8_E4M3 if FP8_E4M3 in peaks else (FP8_GENERIC if FP8_GENERIC in peaks else None)
        if target is not None:
            aliases[MXFP4_ALIAS] = target
            peaks[MXFP4_ALIAS] = peaks[target]
            assumptions["derivation.mxfp4_supply_alias"] = target
    device["compute_peak_aliases"] = aliases
    device["compute_peaks"] = peaks

    base_rates = dict(base_die.rates)
    rates = _scale_rates(dict(device.get("rates", {})), s_c, s_v)
    if memory_bytes is not None:
        device["memory_bytes"] = _positive_count(memory_bytes, "hardware_derivation_memory_invalid")
    if raw_peak is not None:
        rates["hbm"] = raw_peak * hbm_eta
        if record_hbm_assumptions:
            assumptions["derivation.hbm_peak_bytes_per_s"] = repr(raw_peak)
            assumptions["derivation.hbm_rate_bytes_per_s"] = repr(rates["hbm"])
    elif hbm_efficiency is not None and "hbm" in rates:
        rates["hbm"] = float(rates["hbm"]) * hbm_eta
        if record_hbm_assumptions:
            assumptions["derivation.hbm_peak_bytes_per_s"] = repr(die_hbm_peak)
            assumptions["derivation.hbm_rate_bytes_per_s"] = repr(rates["hbm"])
    mte2_preserved: set[str] = set()
    if mte2_efficiency is not None or ceiling is not None:
        for name in ("mte2_aic", "mte2_aiv", "mte2"):
            if name not in rates and name in base_rates:
                scale = s_c if name in AIC_RATE_KEYS else (s_v if name in AIV_RATE_KEYS else 1.0)
                rates[name] = float(base_rates[name]) * scale
                mte2_preserved.add(name)
    if mte2_efficiency is not None:
        for name in ("mte2_aic", "mte2_aiv", "mte2"):
            if name in rates:
                rates[name] = float(rates[name]) * mte2_eta
    if ceiling is not None:
        rates = apply_mte2_ceiling(rates, ceiling)
        assumptions["derivation.mte2_ceiling_bytes_per_s"] = repr(ceiling)
    if "mte2_aic" in rates and "mte2_aiv" in rates:
        split_min = min(float(rates["mte2_aic"]), float(rates["mte2_aiv"]))
        if "mte2" not in base_rates:
            rates["mte2"] = split_min
        elif math.isclose(
            float(base_rates["mte2"]),
            min(float(base_rates.get("mte2_aic", float("inf"))),
                float(base_rates.get("mte2_aiv", float("inf")))),
        ):
            rates["mte2"] = split_min
        else:
            rates["mte2"] = min(float(rates["mte2"]), split_min)

    hbm_scale = float(rates["hbm"]) / float(base_rates["hbm"])
    for name, mode in device.get("service_modes", {}).items():
        resource = mode.get("resource")
        if mode.get("rate") is None:
            continue
        if resource == "hbm":
            mode["rate"] = float(mode["rate"]) * hbm_scale
        elif name in DRAM_BOUND_MODES:
            scaled = float(mode["rate"]) * hbm_scale
            if resource in rates:
                scaled = min(scaled, float(rates[resource]))
            mode["rate"] = scaled
        elif resource in ("mte2_aic", "mte2_aiv", "mte2") and resource in rates and resource in base_rates:
            mode["rate"] = float(mode["rate"]) * (float(rates[resource]) / float(base_rates[resource]))
        elif resource in AIC_RATE_KEYS:
            mode["rate"] = float(mode["rate"]) * s_c
        elif resource in AIV_RATE_KEYS:
            mode["rate"] = float(mode["rate"]) * s_v

    services = (device.get("compute_arch") or {}).get("instruction_services") or {}
    cacheline = services.get(HBM_CACHELINE_SERVICE)
    if isinstance(cacheline, dict) and cacheline.get("clock_hz") is None:
        if HBM_CACHELINE_SERVICE not in base_die.instruction_unit_overrides:
            units.pop(HBM_CACHELINE_SERVICE, None)
        owners = max(1, int(units.get(HBM_CACHELINE_SERVICE, units.get("vector", new_aiv))))
        cacheline["per_owner_rate"] = float(rates["hbm"]) / (HBM_CACHELINE_BYTES * owners)
        if HBM_CACHELINE_SERVICE not in base_die.linked_rate_overrides:
            rates.pop(HBM_CACHELINE_SERVICE, None)
            assumptions[f"derivation.recomputed_service.{HBM_CACHELINE_SERVICE}"] = "from_owner_supply"
        cacheline_mode = (device.get("service_modes") or {}).get(HBM_CACHELINE_SERVICE)
        if isinstance(cacheline_mode, dict) and HBM_CACHELINE_SERVICE not in base_die.instruction_mode_overrides:
            cacheline_mode.pop("rate", None)
    device["resource_units"] = units
    device["rates"] = rates
    _strip_derived_instruction_rates(
        device, base_die, assumptions,
        preserved_rates=frozenset(mte2_preserved),
    )
    device["sku"] = sku


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
        raise ModelingError("hardware_derivation_requires_per_device_core_mapping")
    data = base_hardware.to_dict()
    devices = list(data.get("devices", ()))
    core_override = aic_count is not None or aiv_count is not None
    base_aic = int(devices[0].get("compute_units", 1) or 1)
    base_aiv = int((devices[0].get("resource_units") or {}).get("vector", base_aic))
    new_aic = base_aic if aic_count is None else _positive_count(
        aic_count, "hardware_derivation_aic_count_invalid")
    s_c = float(new_aic) / float(base_aic) if core_override else 1.0
    if aiv_count is None:
        new_aiv = (_exact_scaled(base_aiv, s_c, "hardware_derivation_aiv_count_not_integral")
                   if core_override else base_aiv)
    else:
        new_aiv = _positive_count(aiv_count, "hardware_derivation_aiv_count_invalid")
    s_v = float(new_aiv) / float(base_aiv) if core_override else 1.0
    base_assumptions: dict[str, str] = dict(data.get("assumptions", {}))
    assumptions: dict[str, str] = dict(base_assumptions)
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
    record_hbm_assumptions = raw_peak is not None or base_hardware.homogeneous
    for device_index, device in enumerate(devices):
        die_hbm_peak = (
            raw_peak if raw_peak is not None
            else _base_die_hbm_peak(device, base_assumptions)
        )
        die_aic = new_aic if core_override else int(device.get("compute_units", 1) or 1)
        hbm_eta, hbm_provenance = _resolve_efficiency(
            hbm_efficiency, axis_hint=AXIS_HBM_PEAK, aic_count=die_aic,
            hbm_peak=die_hbm_peak)
        mte2_eta, mte2_provenance = _resolve_efficiency(
            mte2_efficiency, axis_hint=AXIS_AIC_COUNT, aic_count=die_aic,
            hbm_peak=die_hbm_peak)
        if device_index == 0:
            assumptions.update(hbm_provenance)
            if hbm_efficiency is not None:
                assumptions["derivation.hbm_efficiency"] = repr(hbm_eta)
            assumptions.update({
                key.replace("efficiency", "mte2_efficiency", 1): value
                for key, value in mte2_provenance.items()
            })
        _derive_single_device(
            device, base_hardware.dies[device_index],
            arch_family=base_hardware.arch_family, sku=sku,
            core_override=core_override, new_aic=new_aic, new_aiv=new_aiv,
            s_c=s_c, s_v=s_v, memory_bytes=memory_bytes,
            raw_peak=raw_peak, die_hbm_peak=die_hbm_peak,
            record_hbm_assumptions=record_hbm_assumptions and device_index == 0,
            hbm_efficiency=hbm_efficiency, hbm_eta=hbm_eta,
            mte2_efficiency=mte2_efficiency, mte2_eta=mte2_eta,
            ceiling=ceiling, assumptions=assumptions,
        )
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
