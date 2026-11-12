"""Pure physical-coordinate response functions shared by execution and scoring.

[Role] Evaluate physical supply responses and declared interpolation surfaces.
[Invariants] Knots are supplied parameters, never learned from query timings.
  Every coordinate, scaled coordinate, coefficient, knot, grid value and rate
  is finite and nonnegative; booleans are never numbers and rates are strictly
  positive before division. A grid requires every corner. Extrapolation never
  clamps an invalid duration. Response dependencies, knot support and cell
  breakpoints are exposed for static coverage compilation; continuity never
  implies global monotonicity.
[Timeouts & Locks] Bounded parameter tables and binary search; no waits or locks.
[Side-Effect Policy] No I/O, fitting, global mutation or device execution.
"""
from bisect import bisect_left
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
import math

from mindie_sim.schema.common import ModelingError


def nonnegative(value):
    """Return a finite nonnegative float; booleans and non-numbers are invalid."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ModelingError("service_regime_requires_finite_nonnegative_parameters")
    return float(value)


def positive(value):
    """Return a finite strictly positive rate; zero and negatives are invalid."""
    result = nonnegative(value)
    if result == 0:
        raise ModelingError("service_regime_requires_positive_rate")
    return result


def scaled_coordinate(value, scale=1):
    """Apply an explicit recipe scale without silently overflowing to infinity."""
    result = nonnegative(value) * nonnegative(scale)
    if not math.isfinite(result):
        raise ModelingError("service_regime_coordinate_scaling_overflow")
    return result


def _duration(value):
    result = nonnegative(value)
    if not math.isfinite(result):
        raise ModelingError("service_regime_response_not_finite")
    return result


@dataclass(frozen=True)
class ResponseValue:
    duration_us: float | None
    basis: str
    issued_work: float | None = None
    rate_flops_per_s: float | None = None
    active_owners: int | None = None
    stage: str | None = None


@dataclass(frozen=True)
class ResponseFunction:
    kind: str
    fixed_us: float = 0
    terms: tuple[Mapping, ...] = ()
    coordinate: str = ""
    knots: tuple[tuple[float, float], ...] = ()
    outside: str = "analytic"
    coordinates: tuple[float, ...] = ()
    grid: tuple[tuple[str, tuple[float, ...]], ...] = ()
    values_us: tuple[float, ...] = ()
    strides: tuple[int, ...] = ()
    loads: tuple[Mapping, ...] = ()
    resource: str = ""
    dtype: str = ""
    scope: str = ""
    issue: Mapping | None = None
    owner_grid: Mapping | None = None
    owner_coordinate: bool = False

    @property
    def varying_coordinates(self):
        if self.kind == "piecewise_linear":
            return {self.coordinate}
        if self.kind == "multilinear":
            return {name for name, _ in self.grid}
        if self.kind == "measured_dual_load":
            return {"m", "n", "k"} | ({"owners"} if self.owner_coordinate else set())
        return {term["coordinate"] for term in self.terms}

    @property
    def dependencies(self):
        """Physical coordinates the response actually consumes."""
        return frozenset(self.varying_coordinates)

    @property
    def breakpoints(self):
        """Declared cell breakpoints per varying coordinate (empty if analytic)."""
        if self.kind == "piecewise_linear":
            return {self.coordinate: self.coordinates}
        if self.kind == "multilinear":
            return {name: points for name, points in self.grid}
        return {}

    @property
    def knot_support(self):
        """Calibrated knot interval per coordinate; analytic kinds are unbounded."""
        return {name: (points[0], points[-1])
                for name, points in self.breakpoints.items() if points}

    @classmethod
    def from_mapping(cls, raw, coordinate_names):
        """Validate declared parameters without consuming any observation rows."""
        if not isinstance(raw, Mapping):
            raise ModelingError("service_regime_response_requires_mapping")
        if not isinstance(coordinate_names, Mapping):
            raise ModelingError("service_regime_coordinates_require_mapping")
        kind = raw.get("kind", "affine")
        if not isinstance(kind, str):
            raise ModelingError("service_regime_response_kind_unknown")
        if kind == "measured_dual_load":
            if set(raw) - {"kind", "fixed_us", "resource", "dtype", "loads",
                           "scope", "issue", "owner_grid"}:
                raise ModelingError("service_regime_response_unknown_fields")
            for name in ("m", "n", "k"):
                if name not in coordinate_names:
                    raise ModelingError(
                        "measured_dual_load_requires_declared_m_n_k_coordinates")
            owner_coordinate = "owners" in coordinate_names
            owner_grid = raw.get("owner_grid")
            if owner_grid is not None and (not isinstance(owner_grid, Mapping)
                  or set(owner_grid) != {"tile_m", "tile_n", "cores"}
                  or any(isinstance(owner_grid.get(key), bool)
                         or not isinstance(owner_grid.get(key), int)
                         or owner_grid[key] <= 0
                         for key in ("tile_m", "tile_n", "cores"))):
                raise ModelingError("measured_dual_load_invalid_owner_grid")
            resource = raw.get("resource")
            if not isinstance(resource, str) or not resource:
                raise ModelingError("measured_dual_load_requires_resource")
            dtype = raw.get("dtype")
            if resource == "cube" and not dtype:
                raise ModelingError("measured_dual_load_cube_requires_dtype")
            issue = raw.get("issue")
            if (not isinstance(issue, Mapping)
                    or set(issue) - {"native_type", "dtype", "output_dtype",
                                     "m_alignment", "alignment_basis", "source_path",
                                     "source_sha256", "source_revision", "scope", "n", "k",
                                     "weight_layout", "wrapper_source_path",
                                     "wrapper_source_sha256", "wrapper_source_revision",
                                     "wrapper_symbol", "deployment_image_id",
                                     "source_scope", "source_basis"}
                    or not isinstance(issue.get("m_alignment"), int)
                    or isinstance(issue.get("m_alignment"), bool)
                    or issue["m_alignment"] <= 0
                    or not isinstance(issue.get("n"), int)
                    or not isinstance(issue.get("k"), int)
                    or issue["n"] <= 0 or issue["k"] <= 0):
                raise ModelingError("measured_dual_load_requires_certified_issue_rule")
            declared = raw.get("loads")
            if not isinstance(declared, (list, tuple)) or len(declared) != 2:
                raise ModelingError("measured_dual_load_requires_two_measured_loads")
            loads, ids = [], set()
            for load in declared:
                if (not isinstance(load, Mapping)
                        or set(load) - {"id", "active_owners", "rate_flops_per_s",
                                        "measured_issued_flops", "measured_logical_flops",
                                        "event_count", "median_aic_mac_time_ns",
                                        "median_aic_mac_ratio", "median_mte2_time_ns",
                                        "median_elapsed_ns", "cube_utilization_pct",
                                        "limiting_pipe", "provenance"}):
                    raise ModelingError("measured_dual_load_invalid_load_record")
                load_id = load.get("id")
                if load_id not in {"saturated", "underfilled"} or load_id in ids:
                    raise ModelingError("measured_dual_load_requires_saturated_and_underfilled")
                ids.add(load_id)
                owners = load.get("active_owners")
                if (isinstance(owners, bool) or not isinstance(owners, int)
                        or owners <= 0):
                    raise ModelingError("measured_dual_load_requires_positive_active_owners")
                rate = positive(load.get("rate_flops_per_s"))
                issued = positive(load.get("measured_issued_flops"))
                loads.append({**load, "active_owners": owners,
                              "rate_flops_per_s": rate,
                              "measured_issued_flops": issued})
            if ids != {"saturated", "underfilled"}:
                raise ModelingError("measured_dual_load_requires_saturated_and_underfilled")
            if int(loads[0]["active_owners"]) == int(loads[1]["active_owners"]):
                raise ModelingError("measured_dual_load_requires_distinct_owner_counts")
            return cls(kind, nonnegative(raw.get("fixed_us", 0)),
                       loads=tuple(loads), resource=str(resource),
                       dtype=str(dtype or ""), scope=str(raw.get("scope", "")),
                       issue=dict(issue),
                       owner_grid=dict(owner_grid) if owner_grid is not None else None,
                       owner_coordinate=owner_coordinate)
        if kind == "roofline":
            if set(raw) - {"kind", "fixed_us", "services"}:
                raise ModelingError("service_regime_response_unknown_fields")
            services = raw.get("services")
            if not isinstance(services, (list, tuple)) or not services:
                raise ModelingError("service_regime_roofline_requires_service")
            resolved = []
            for term in services:
                if not isinstance(term, Mapping):
                    raise ModelingError("service_regime_roofline_requires_coordinate_and_resource")
                if (set(term) - {"coordinate", "resource", "dtype", "utilization"}
                        or term.get("coordinate") not in coordinate_names
                        or not term.get("resource")):
                    raise ModelingError("service_regime_roofline_requires_coordinate_and_resource")
                utilization = nonnegative(term.get("utilization", 1))
                if not 0 < utilization <= 1:
                    raise ModelingError("service_regime_roofline_requires_positive_utilization_and_precision")
                if term["resource"] == "cube" and not term.get("dtype"):
                    raise ModelingError("service_regime_roofline_requires_positive_utilization_and_precision")
                resolved.append({**term, "utilization": utilization})
            return cls(kind, nonnegative(raw.get("fixed_us", 0)), tuple(resolved))
        if kind == "affine":
            if set(raw) - {"kind", "fixed_us", "terms"}:
                raise ModelingError("service_regime_response_unknown_fields")
            fixed = nonnegative(raw.get("fixed_us", 0))
            declared = raw.get("terms", ())
            if not isinstance(declared, (list, tuple)):
                raise ModelingError("service_regime_affine_terms_require_sequence")
            terms = []
            for term in declared:
                if not isinstance(term, Mapping) or set(term) - {"coordinate", "us_per_unit", "scale_with"}:
                    raise ModelingError("service_regime_term_unknown_fields")
                if term.get("coordinate") not in coordinate_names:
                    raise ModelingError("service_regime_term_requires_declared_coordinate")
                nonnegative(term.get("us_per_unit"))
                scaling = term.get("scale_with")
                if scaling:
                    if (not isinstance(scaling, Mapping)
                            or set(scaling) - {"resource", "reference_rate", "dtype"}
                            or not scaling.get("resource")
                            or "reference_rate" not in scaling):
                        raise ModelingError("service_regime_invalid_rate_scaling")
                    if positive(scaling["reference_rate"]) == 0:
                        raise ModelingError("service_regime_reference_rate_must_be_positive")
                terms.append({**term, **({"scale_with": dict(scaling)} if scaling else {})})
            return cls(kind, fixed, tuple(terms))
        if kind == "multilinear":
            if set(raw) - {"kind", "grid", "values_us", "outside"}:
                raise ModelingError("service_regime_response_unknown_fields")
            declared = raw.get("grid")
            if not isinstance(declared, (list, tuple)) or not declared:
                raise ModelingError("service_regime_grid_requires_declared_coordinates")
            grid = []
            for axis in declared:
                if (not isinstance(axis, Mapping)
                        or set(axis) != {"coordinate", "knots"}
                        or axis.get("coordinate") not in coordinate_names):
                    raise ModelingError("service_regime_grid_requires_declared_coordinates")
                grid.append((axis["coordinate"], _ordered_knots(axis["knots"])))
            grid = tuple(grid)
            if len({name for name, _ in grid}) != len(grid):
                raise ModelingError("service_regime_grid_requires_declared_coordinates")
            raw_values = raw.get("values_us")
            if not isinstance(raw_values, (list, tuple)):
                raise ModelingError("service_regime_grid_requires_every_corner")
            values = tuple(nonnegative(value) for value in raw_values)
            shape = tuple(len(points) for _, points in grid)
            if len(values) != math.prod(shape):
                raise ModelingError("service_regime_grid_requires_every_corner")
            strides = tuple(math.prod(shape[index + 1:]) for index in range(len(shape)))
            return cls(kind, outside=_outside(raw), grid=grid, values_us=values, strides=strides)
        if kind != "piecewise_linear":
            raise ModelingError("service_regime_response_kind_unknown:" + kind)
        if set(raw) - {"kind", "coordinate", "knots", "outside"}:
            raise ModelingError("service_regime_response_unknown_fields")
        coordinate = raw.get("coordinate")
        if coordinate not in coordinate_names:
            raise ModelingError("service_regime_curve_requires_declared_coordinate")
        knots = _pairs(raw.get("knots"))
        xs = _ordered_knots(pair[0] for pair in knots)
        return cls(kind, coordinate=coordinate, knots=knots, outside=_outside(raw), coordinates=xs)

    def evaluate(self, values, *, rate_for=None, resource_for=None):
        """Return the numerical response, not an Engine completion prediction."""
        if not isinstance(values, Mapping):
            raise ModelingError("service_regime_response_requires_coordinate_values")
        if self.kind == "measured_dual_load":
            m = _value(values, "m")
            n = _value(values, "n")
            k = _value(values, "k")
            explicit_owners = None
            if "owners" in values:
                owners_value = values.get("owners")
                if (isinstance(owners_value, bool)
                        or not isinstance(owners_value, (int, float))
                        or not math.isfinite(float(owners_value))
                        or float(owners_value) != int(owners_value)
                        or int(owners_value) <= 0):
                    raise ModelingError("measured_dual_load_requires_positive_active_owners")
                explicit_owners = int(owners_value)
            issue = self.issue or {}
            if float(n) != float(issue.get("n")) or float(k) != float(issue.get("k")):
                return ResponseValue(None, "measured_load_geometry_mismatch")
            alignment = int(issue["m_alignment"])
            aligned_m = math.ceil(float(m) / alignment) * alignment
            issued = 2.0 * aligned_m * float(n) * float(k)
            if not math.isfinite(issued) or issued <= 0:
                raise ModelingError("measured_dual_load_issued_work_invalid")
            under = next((item for item in self.loads
                          if item["id"] == "underfilled"), None)
            saturated = next((item for item in self.loads
                              if item["id"] == "saturated"), None)
            load = None
            if (under is not None
                    and int(round(issued)) == int(under["measured_issued_flops"])):
                load = under
            else:
                full_work = (int(saturated["measured_issued_flops"])
                             if saturated is not None else 0)
                issued_int = int(round(issued))
                if (full_work > 0 and issued_int >= full_work
                        and issued_int % full_work == 0):
                    load = saturated
            if load is None:
                return ResponseValue(None, "measured_load_unknown_issue_class")
            owners = int(load["active_owners"])
            if explicit_owners is not None and explicit_owners != owners:
                return ResponseValue(None, "measured_load_unknown_active_owners")
            duration = self.fixed_us + issued * 1e6 / float(load["rate_flops_per_s"])
            if not math.isfinite(duration):
                raise ModelingError("measured_dual_load_service_overflow")
            return ResponseValue(
                _duration(duration), "measured_" + str(load["id"]),
                issued_work=issued, rate_flops_per_s=float(load["rate_flops_per_s"]),
                active_owners=owners, stage="mac",
            )
        if self.kind == "roofline":
            if rate_for is None:
                raise ModelingError("service_regime_scaled_response_requires_physical_rates")
            service = {}
            for term in self.terms:
                resource = resource_for(term) if resource_for is not None else term["resource"]
                rate = positive(rate_for(term))
                duration = _value(values, term["coordinate"]) * 1e6 / (rate * term["utilization"])
                if not math.isfinite(duration):
                    raise ModelingError("service_regime_roofline_service_overflow")
                service[resource] = service.get(resource, 0) + duration
            # Distinct instruction modes and compute precisions can consume
            # the same physical pipe; their service adds before pipe overlap.
            duration = self.fixed_us + max(service.values())
            return ResponseValue(_duration(duration), "physical_supply")
        if self.kind == "affine":
            duration = self.fixed_us
            for term in self.terms:
                time = _value(values, term["coordinate"]) * term["us_per_unit"]
                if term.get("scale_with"):
                    if rate_for is None:
                        raise ModelingError("service_regime_scaled_response_requires_physical_rates")
                    rate = positive(rate_for(term["scale_with"]))
                    time *= term["scale_with"]["reference_rate"] / rate
                if not math.isfinite(time) or time < 0:
                    raise ModelingError("service_regime_affine_response_invalid_term")
                duration += time
                if not math.isfinite(duration):
                    raise ModelingError("service_regime_affine_response_overflow")
            return ResponseValue(_duration(duration), "affine")
        if self.kind == "multilinear":
            brackets, outside, anchor = [], False, True
            for name, points in self.grid:
                x = nonnegative(_value(values, name))
                right = bisect_left(points, x)
                if right < len(points) and points[right] == x:
                    brackets.append(((right, 1.0),))
                    continue
                anchor = False
                outside |= right == 0 or right == len(points)
                right = min(max(right, 1), len(points) - 1)
                weight = (x - points[right - 1]) / (points[right] - points[right - 1])
                if not math.isfinite(weight):
                    raise ModelingError("service_regime_grid_cell_weight_invalid")
                brackets.append(((right - 1, 1 - weight), (right, weight)))
            if outside and self.outside == "analytic":
                return ResponseValue(None, "analytic_outside_knots")
            if outside and self.outside == "error":
                raise ModelingError("service_regime_grid_coordinate_outside_knots")
            duration = math.fsum(
                self.values_us[sum(index * stride for (index, _), stride in zip(corner, self.strides))]
                * math.prod(weight for _, weight in corner) for corner in product(*brackets))
            if duration < 0 or not math.isfinite(duration):
                raise ModelingError("service_regime_curve_extrapolation_invalid_duration")
            return ResponseValue(duration, "extrapolated" if outside else "anchor" if anchor else "interpolated")
        x = nonnegative(_value(values, self.coordinate))
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


def _value(values, name):
    if name not in values:
        raise ModelingError("service_regime_response_missing_coordinate:" + name)
    return nonnegative(values[name])


def _pairs(values):
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ModelingError("service_regime_curve_requires_coordinate_duration_pairs")
    pairs = []
    for pair in values:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ModelingError("service_regime_curve_requires_coordinate_duration_pairs")
        pairs.append((nonnegative(pair[0]), nonnegative(pair[1])))
    return tuple(pairs)


def _ordered_knots(values):
    if values is None or isinstance(values, (str, bytes)):
        raise ModelingError("service_regime_curve_requires_strictly_ordered_coordinates")
    try:
        points = tuple(nonnegative(value) for value in values)
    except TypeError as exc:
        raise ModelingError("service_regime_curve_requires_strictly_ordered_coordinates") from exc
    if len(points) < 2 or any(right <= left for left, right in zip(points, points[1:])):
        raise ModelingError("service_regime_curve_requires_strictly_ordered_coordinates")
    return points


def _outside(raw):
    outside = raw.get("outside", "analytic")
    if outside not in {"analytic", "linear", "error"}:
        raise ModelingError("service_regime_curve_outside_requires_analytic_linear_or_error")
    return outside
