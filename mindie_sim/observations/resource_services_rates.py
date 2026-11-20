"""Declared nominal rates for normalized resource services.

[Role]
  Validate rate declarations and apply explicitly declared unit conversion.
[Invariants]
  Rate scope, hardware and work unit are checked before a rate can bind.
[Timeouts & Locks]
  In-memory validation. No I/O or locks.
[Side-Effect Policy]
  Returns a scalar rate without modifying the supplied declaration.
"""

import math
from typing import Mapping


_RATE_SCOPES = frozenset({"aggregate", "per_owner"})


def _text(value, name, *, default=None):
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"resource_service_requires_nonempty_{name}")
    return value.strip()


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"resource_service_{name}_requires_number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"resource_service_{name}_requires_finite")
    if positive and result <= 0:
        raise ValueError(f"resource_service_{name}_requires_positive")
    if result < 0:
        raise ValueError(f"resource_service_{name}_requires_nonnegative")
    return result


def resolve_nominal_rate(resource, nominal_rates, *, work_unit, hardware=None,
                         rate_scope="aggregate"):
    """Return one hardware-nominal rate, requiring its actual unit and scope.

    A bare number is rejected: it would silently reinterpret ``device.rates``
    in the caller's ``work_unit``/``rate_scope``.  A typed mapping must declare
    ``rate``, the hardware's actual ``work_unit`` and ``rate_scope``.  A unit
    conversion is accepted only when ``conversion`` explicitly declares the
    target unit and the source-units-per-target-unit factor; no implicit
    bytes-to-rows or aggregate-to-per-owner conversion is made.
    """
    if nominal_rates is None:
        return None
    if not isinstance(nominal_rates, Mapping):
        raise ValueError("resource_service_nominal_rates_requires_mapping")
    raw = nominal_rates.get(resource)
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise ValueError(
            "resource_service_bare_nominal_rate_requires_explicit_unit_and_scope")
    unknown = set(raw) - {
        "rate", "work_unit", "rate_scope", "hardware", "conversion", "source"}
    if unknown:
        raise ValueError("resource_service_nominal_rate_unknown_fields:"
                         + ",".join(sorted(str(item) for item in unknown)))
    if "work_unit" not in raw:
        raise ValueError(
            "resource_service_nominal_rate_requires_explicit_work_unit")
    if "rate_scope" not in raw:
        raise ValueError(
            "resource_service_nominal_rate_requires_explicit_rate_scope")
    source_unit = _text(raw.get("work_unit"), "nominal_rate_work_unit")
    declared_scope = _text(raw.get("rate_scope"), "nominal_rate_scope")
    if declared_scope not in _RATE_SCOPES:
        raise ValueError("resource_service_nominal_rate_scope_invalid")
    if declared_scope != rate_scope:
        raise ValueError("resource_service_nominal_rate_scope_mismatch")
    declared_hardware = raw.get("hardware")
    if hardware and declared_hardware and str(declared_hardware) != str(hardware):
        raise ValueError("resource_service_nominal_rate_hardware_mismatch")
    rate = _number(raw.get("rate"), "nominal_rate", positive=True)
    conversion = raw.get("conversion")
    if source_unit == work_unit:
        if conversion is not None:
            raise ValueError(
                "resource_service_nominal_rate_conversion_not_needed")
        return rate
    if not isinstance(conversion, Mapping):
        raise ValueError("resource_service_nominal_rate_unit_mismatch")
    unknown = set(conversion) - {
        "target_work_unit", "source_units_per_target_unit"}
    if unknown:
        raise ValueError("resource_service_nominal_rate_conversion_unknown_fields:"
                         + ",".join(sorted(str(item) for item in unknown)))
    target_unit = _text(conversion.get("target_work_unit"),
                        "conversion_target_work_unit")
    if target_unit != work_unit:
        raise ValueError("resource_service_nominal_rate_unit_mismatch")
    factor = _number(conversion.get("source_units_per_target_unit"),
                     "conversion_factor", positive=True)
    return rate / factor
