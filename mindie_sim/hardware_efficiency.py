"""Validated efficiency curves for pure hardware-SKU derivation.

[Role]
  Represent an explicitly declared efficiency response along one physical axis
  (AIC owner count or HBM peak supply) and resolve it without fitted defaults.
  Also provide the shared-ceiling arithmetic used by the split MTE2 pipes.

[Invariants]
  * Knot x values are finite, nonnegative and strictly increasing; y values are
    finite in ``(0, 1]``.  No model/hardware-name default curve exists.
  * Interpolation is linear between knots; only an explicit ``clamp``
    extrapolation extends an endpoint value.
  * A shared ceiling is a coupling cap: it is never handed at full value to two
    independently summed pipes.  The unified fallback is ``min(aic, aiv)``.

[Timeouts & Locks]
  Pure arithmetic; no I/O or locks.

[Side-Effect Policy]
  Read-only immutable value objects.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from mindie_sim.schema.common import ModelingError


ZERO: float = 0.0
ONE: float = 1.0
FIRST = 0
LAST = -1
INT_ONE = 1
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
            if not math.isfinite(x) or x < ZERO:
                raise ModelingError(f"efficiency_curve_knot_x_invalid:{x}")
            if not math.isfinite(y) or y <= ZERO or y > ONE:
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
        if not math.isfinite(value) or value < ZERO:
            raise ModelingError(f"efficiency_curve_x_invalid:{x}")
        knots = self.knots
        if value <= knots[FIRST][FIRST]:
            if self.extrapolation == EXTRAPOLATION_CLAMP:
                return knots[FIRST][INT_ONE]
            raise ModelingError(f"efficiency_curve_below_range:{value}")
        if value >= knots[LAST][FIRST]:
            if self.extrapolation == EXTRAPOLATION_CLAMP:
                return knots[LAST][INT_ONE]
            raise ModelingError(f"efficiency_curve_above_range:{value}")
        for index in range(INT_ONE, len(knots)):
            left_x, left_y = knots[index - INT_ONE]
            right_x, right_y = knots[index]
            if value <= right_x:
                span = right_x - left_x
                if span <= ZERO:
                    return right_y
                return left_y + (right_y - left_y) * (value - left_x) / span
        return knots[LAST][INT_ONE]

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
                               ((ZERO, float(value)),))
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
    if not math.isfinite(cap) or cap <= ZERO:
        raise ModelingError(f"mte2_ceiling_invalid:{ceiling}")
    keys = [name for name in ("mte2_aic", "mte2_aiv") if name in result]
    if not keys:
        if "mte2" in result:
            result["mte2"] = min(result["mte2"], cap)
        return result
    total = sum(max(ZERO, float(result[name])) for name in keys)
    if total > cap:
        factor = cap / total
        for name in keys:
            result[name] = float(result[name]) * factor
    if "mte2" in result:
        result["mte2"] = min(float(result["mte2"]), cap)
    return result
