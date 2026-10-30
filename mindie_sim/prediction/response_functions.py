"""Pure physical-coordinate response functions shared by execution and scoring.

[Role] Evaluate declared affine or one-dimensional piecewise linear responses.
[Invariants] Knots are supplied parameters, never learned from query timings.
  Interpolation covers one coordinate; other identity dimensions remain caller
  conditions. Extrapolation is explicit and never clamps an invalid duration.
[Timeouts & Locks] Bounded parameter tables and binary search; no waits or locks.
[Side-Effect Policy] No I/O, fitting, global mutation or device execution.
"""
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
import math

from mindie_sim.schema.common import ModelingError


def nonnegative(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ModelingError("service_regime_requires_finite_nonnegative_parameters")
    return float(value)


@dataclass(frozen=True)
class ResponseValue:
    duration_us: float | None
    basis: str


@dataclass(frozen=True)
class ResponseFunction:
    kind: str
    fixed_us: float = 0
    terms: tuple[Mapping, ...] = ()
    coordinate: str = ""
    knots: tuple[tuple[float, float], ...] = ()
    outside: str = "analytic"
    coordinates: tuple[float, ...] = ()

    @classmethod
    def from_mapping(cls, raw, coordinate_names):
        """Validate declared parameters without consuming any observation rows."""
        kind = raw.get("kind", "affine")
        if kind == "affine":
            if set(raw) - {"kind", "fixed_us", "terms"}:
                raise ModelingError("service_regime_response_unknown_fields")
            fixed = nonnegative(raw.get("fixed_us", 0))
            terms = []
            for term in raw.get("terms", ()):
                if set(term) - {"coordinate", "us_per_unit", "scale_with"}:
                    raise ModelingError("service_regime_term_unknown_fields")
                if term["coordinate"] not in coordinate_names:
                    raise ModelingError("service_regime_term_requires_declared_coordinate")
                nonnegative(term["us_per_unit"])
                scaling = term.get("scale_with")
                if scaling:
                    if set(scaling) - {"resource", "reference_rate", "dtype"} or not scaling.get("resource"):
                        raise ModelingError("service_regime_invalid_rate_scaling")
                    if nonnegative(scaling["reference_rate"]) == 0:
                        raise ModelingError("service_regime_reference_rate_must_be_positive")
                terms.append({**term, **({"scale_with": dict(scaling)} if scaling else {})})
            return cls(kind, fixed, tuple(terms))
        if kind != "piecewise_linear":
            raise ModelingError("service_regime_response_kind_unknown:" + kind)
        if set(raw) - {"kind", "coordinate", "knots", "outside"}:
            raise ModelingError("service_regime_response_unknown_fields")
        coordinate = raw["coordinate"]
        if coordinate not in coordinate_names:
            raise ModelingError("service_regime_curve_requires_declared_coordinate")
        knots = tuple(tuple(nonnegative(value) for value in pair) for pair in raw["knots"])
        if len(knots) < 2 or any(len(pair) != 2 for pair in knots):
            raise ModelingError("service_regime_curve_requires_coordinate_duration_pairs")
        xs = tuple(pair[0] for pair in knots)
        if any(right <= left for left, right in zip(xs, xs[1:])):
            raise ModelingError("service_regime_curve_requires_strictly_ordered_coordinates")
        outside = raw.get("outside", "analytic")
        if outside not in {"analytic", "linear", "error"}:
            raise ModelingError("service_regime_curve_outside_requires_analytic_linear_or_error")
        return cls(kind, coordinate=coordinate, knots=knots, outside=outside, coordinates=xs)

    def evaluate(self, values, *, rate_for=None):
        """Return the numerical response, not an Engine completion prediction."""
        if self.kind == "affine":
            duration = self.fixed_us
            for term in self.terms:
                time = values[term["coordinate"]] * term["us_per_unit"]
                if term.get("scale_with"):
                    if rate_for is None:
                        raise ModelingError("service_regime_scaled_response_requires_physical_rates")
                    time *= term["scale_with"]["reference_rate"] / rate_for(term["scale_with"])
                duration += time
            return ResponseValue(nonnegative(duration), "affine")
        x = nonnegative(values[self.coordinate])
        right = bisect_left(self.coordinates, x)
        if right < len(self.knots) and self.coordinates[right] == x:
            return ResponseValue(self.knots[right][1], "anchor")
        outside = right == 0 or right == len(self.knots)
        if outside and self.outside == "analytic":
            return ResponseValue(None, "analytic_outside_knots")
        if outside and self.outside == "error":
            raise ModelingError("service_regime_curve_coordinate_outside_knots:" + str(x))
        right = min(max(right, 1), len(self.knots) - 1)
        x0, y0 = self.knots[right - 1]
        x1, y1 = self.knots[right]
        duration = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        if duration < 0 or not math.isfinite(duration):
            raise ModelingError("service_regime_curve_extrapolation_invalid_duration")
        return ResponseValue(duration, "extrapolated" if outside else "interpolated")
