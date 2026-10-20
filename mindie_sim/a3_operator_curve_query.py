"""Positive physical prediction policy and robust fit primitives for curves.

[Role]
  Own the single prediction policy used by the analysis, the artifact export and
  the operational query, plus the bounded robust-fit primitives that policy
  needs (Huber affine segment, interior-bracket local segment and the guarded
  monotone 2-D log-plane).  Resolve a pre-use axis value against fitted support:
  measured segment, unresolved transition bracket, local interior resolution,
  guarded 2-D interpolation, or labelled extrapolation.

[Invariants]
  * A segment publishes a non-decreasing cost: a measured negative slope is
    reported as measured, then projected to a constant level, and the published
    segment knots are projected onto the non-decreasing cone by isotonic
    regression (pool-adjacent-violators) before publication.  A decreasing
    published cost is therefore impossible; every projection is labelled with
    its measured slope and its size.
  * A transition bracket keeps honest endpoint bounds unless >= 2 measured
    points lie strictly inside it; only then is a local segment fitted, and its
    result carries the distinct ``transition_resolved_local`` status.
  * A predicted time is missing rather than clamped when it is non-positive or
    non-finite.  Outliers are reported, never silently dropped.
  * Robust fits use Huber loss with an IRLS scale from the residual MAD; every
    fit reports its residuals, outliers, iteration count and raw (unprojected)
    slope so a weak fit cannot masquerade as a strong one.
  * 2-D coverage is interpolation only: the query must be bracketed in both
    fixed dimensions, the log-plane must be monotone in both, at least three
    measured curves must participate, and the query axis must be interior to
    every participating curve.  No 2-D extrapolation exists.

[Timeouts & Locks]
  Pure bounded arithmetic; no I/O, device, subprocess, lock or sleep.

[Side-Effect Policy]
  No side effects.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

DEFAULT_HUBER_ITERATIONS = 40
HUBER_DELTA_SIGMA = 1.345
OUTLIER_SIGMA = 3.0
# A fitted slope is reported as non-monotone measurement when the fitted drop
# over the segment exceeds this fraction of the segment level.
MAX_RELATIVE_DROP = 0.10
# A 2-D estimate is published only while its relative error bound is below the
# point estimate itself: at >=100 % the interval includes zero and the number
# carries no usable information.
# A segment is projected onto a constant level instead of publishing a
# decreasing cost.  The projection is labelled, never silent.
PROJECTION_CONSTANT_LEVEL = "constant_level"
MIN_2D_NODES = 3
MAX_2D_RMS_LOG_RESIDUAL = 0.35
MAX_2D_RELATIVE_UNCERTAINTY = 1.0
MIN_R2_FOR_STRONG_FIT = 0.5
MAX_RELATIVE_RESIDUAL_FOR_STRONG_FIT = 0.15


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    return float(ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0)


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = _median(values)
    return _median([abs(float(value) - center) for value in values])


def _p90(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return ordered[index]


def robust_scale(values: Sequence[float]) -> float:
    """Normal-consistent robust standard deviation (1.4826 * MAD)."""
    return 1.4826 * _mad(values)


def _ols(xs: Sequence[float], ys: Sequence[float]) -> dict[str, float] | None:
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 1e-15:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x
    sse = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    sst = sum((y - mean_y) ** 2 for y in ys)
    return {
        "slope_ms_per_unit": slope, "intercept_ms": intercept, "sse": sse,
        "sst": sst, "n": float(n), "x_min": min(xs), "x_max": max(xs),
        "mean_y": mean_y,
    }


def r2_from(sse: float, sst: float, ys: Sequence[float]) -> float:
    """Coefficient of determination with an explicit degenerate-support rule.

    A constant series has no variance to explain, so it is a perfect fit of
    itself; only a spread that is resolvable against the mean level counts as
    variance.
    """
    mean_y = sum(ys) / len(ys) if ys else 0.0
    floor = max(1e-18, 1e-9 * len(ys) * mean_y * mean_y)
    if sst <= floor:
        return 1.0
    return max(0.0, min(1.0, 1.0 - sse / sst))


def _huber_weights(residuals: Sequence[float], delta: float) -> list[float]:
    if delta <= 0:
        return [1.0] * len(residuals)
    return [1.0 if abs(r) <= delta else delta / abs(r) for r in residuals]


def _weighted_line(xs: Sequence[float], ys: Sequence[float],
                   weights: Sequence[float]) -> tuple[float, float] | None:
    sw = sum(weights)
    if sw <= 0:
        return None
    mean_x = sum(w * x for w, x in zip(weights, xs)) / sw
    mean_y = sum(w * y for w, y in zip(weights, ys)) / sw
    sxx = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
    if sxx <= 1e-15:
        return None
    sxy = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys))
    slope = sxy / sxx
    return slope, mean_y - slope * mean_x


def robust_affine(xs: Sequence[float], ys: Sequence[float], *,
                  iterations: int = DEFAULT_HUBER_ITERATIONS) -> dict[str, Any] | None:
    """Huber-loss affine fit with IRLS, plus the raw OLS slope for reporting.

    The MAD-derived scale is recomputed between iterations so a single gross
    outlier cannot inflate the scale and hide itself.  Residuals, outliers and
    the raw slope are returned; nothing is deleted.
    """
    raw = _ols(xs, ys)
    if raw is None:
        return None
    residuals = [y - (raw["intercept_ms"] + raw["slope_ms_per_unit"] * x)
                 for x, y in zip(xs, ys)]
    scale = max(robust_scale(residuals), 1e-12)
    delta = HUBER_DELTA_SIGMA * scale
    slope, intercept = raw["slope_ms_per_unit"], raw["intercept_ms"]
    used = 0
    for used in range(1, max(1, iterations) + 1):
        residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        weights = _huber_weights(residuals, delta)
        fitted = _weighted_line(xs, ys, weights)
        if fitted is None:
            break
        new_slope, new_intercept = fitted
        converged = (abs(new_slope - slope) <= 1e-15 + 1e-6 * abs(slope)
                     and abs(new_intercept - intercept) <= 1e-15 + 1e-6 * abs(intercept))
        slope, intercept = new_slope, new_intercept
        scale = max(robust_scale(residuals), 1e-12)
        delta = HUBER_DELTA_SIGMA * scale
        if converged:
            break
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    scale = robust_scale(residuals)
    level = _median(ys)
    # A perfect (or near-perfect) fit collapses the MAD scale to ~0 and would
    # label every ordinary point an outlier.  Floor the outlier scale at a small
    # fraction of the response so only gross deviations are reported.
    outlier_sigma = max(scale, 1e-6 * abs(level))
    outliers = [
        {"x": float(x), "residual_ms": float(r), "median_ms": float(y)}
        for x, r, y in zip(xs, residuals, ys)
        if abs(r) > OUTLIER_SIGMA * outlier_sigma
    ]
    span = max(xs) - min(xs)
    drop_fraction = (-slope * span) / level if level > 0 else 0.0
    return {
        "slope_ms_per_unit": slope, "intercept_ms": intercept,
        "slope_raw_ms_per_unit": raw["slope_ms_per_unit"],
        "intercept_raw_ms": raw["intercept_ms"],
        "sse": sum(r * r for r in residuals), "sst": raw["sst"],
        "r2": r2_from(sum(r * r for r in residuals), raw["sst"], ys),
        "r2_raw": r2_from(raw["sse"], raw["sst"], ys),
        "n": len(xs), "x_min": min(xs), "x_max": max(xs),
        "huber_delta_ms": delta, "iterations": used,
        "residual_scale_ms": scale,
        "residual_median_abs_ms": _median([abs(r) for r in residuals]),
        "residual_max_abs_ms": max((abs(r) for r in residuals), default=0.0),
        "residual_p90_abs_ms": _p90([abs(r) for r in residuals]),
        "outlier_sigma_ms": outlier_sigma,
        "relative_residual": (scale / level) if level > 0 else 0.0,
        "level_ms": level,
        "relative_drop_fraction": max(0.0, drop_fraction),
        "measured_negative_slope": bool(slope < 0),
        "measured_negative_drop_exceeds_tolerance": bool(drop_fraction > MAX_RELATIVE_DROP),
        "outliers": outliers,
        "residuals": [{"x": float(x), "residual_ms": float(r)} for x, r in zip(xs, residuals)],
    }


def _segment_value(segment: Mapping[str, Any], x: float) -> float:
    if segment.get("projection") == PROJECTION_CONSTANT_LEVEL:
        if segment.get("level_ms") is not None:
            return float(segment["level_ms"])
    return float(segment["intercept_ms"]) + float(segment["slope_ms_per_unit"]) * x


def _segment_uncertainty(segment: Mapping[str, Any]) -> float | None:
    if segment.get("uncertainty_ms") is not None:
        return float(segment["uncertainty_ms"])
    if segment.get("residual_scale_ms") is not None:
        return float(segment["residual_scale_ms"])
    return None


def _fit_quality(fit: Mapping[str, Any]) -> dict[str, Any]:
    quality = fit.get("fit_quality")
    return dict(quality) if isinstance(quality, Mapping) else {}


def _segment_status(fit: Mapping[str, Any], segment: Mapping[str, Any]) -> str:
    """Publication status of a segment: strong, projected, or weak.

    A segment whose own residual spread already exceeds the strong-fit bound is
    weak regardless of how well the rest of the curve fits, because the query
    lands on *this* segment.  A measured negative slope is reported as projected
    onto a constant level, never as an increasing trend it cannot support.
    """
    quality = _fit_quality(fit)
    segment_relative = float(segment.get("relative_residual") or 0.0)
    if segment_relative > MAX_RELATIVE_RESIDUAL_FOR_STRONG_FIT:
        return "weak_fit"
    if segment.get("projection") == PROJECTION_CONSTANT_LEVEL:
        if quality.get("status") == "weak":
            return "weak_fit"
        return "plateau_projected"
    if quality.get("status") == "weak":
        return "weak_fit"
    return "ok"


def _bracket_support(fit: Mapping[str, Any], low: float, high: float) -> list[list[float]]:
    support = fit.get("bracket_support")
    points: list[list[float]] = []
    if isinstance(support, Mapping):
        for key, values in support.items():
            try:
                start, end = (float(part) for part in str(key).split(","))
            except (TypeError, ValueError):
                continue
            if abs(start - low) <= 1e-12 and abs(end - high) <= 1e-12:
                for value in values or ():
                    try:
                        x, y = float(value[0]), float(value[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if low < x < high:
                        points.append([x, y])
    elif isinstance(support, (list, tuple)):
        for item in support:
            if not isinstance(item, Mapping):
                continue
            try:
                start, end = float(item["x_min"]), float(item["x_max"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(start - low) > 1e-12 or abs(end - high) > 1e-12:
                continue
            for value in item.get("points") or ():
                try:
                    x, y = float(value[0]), float(value[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if low < x < high:
                    points.append([x, y])
    unique = {(round(x, 12), round(y, 12)): [x, y] for x, y in points}
    return [unique[key] for key in sorted(unique)]


def local_bracket_fit(points: Sequence[Sequence[float]], x: float, *,
                      lower_bound: float | None = None,
                      upper_bound: float | None = None) -> dict[str, Any] | None:
    """Robust local segment through >= 2 measured points strictly in a bracket.

    Uncertainty is the robust residual scale of the interior support plus a
    quarter of the unresolved bracket width scaled by 2/n_interior, so a
    2-point support keeps a quarter of the bracket disagreement instead of
    pretending the local line is exact.
    """
    clean = sorted((float(p[0]), float(p[1])) for p in points if float(p[1]) > 0)
    if len(clean) < 2:
        return None
    xs = [item[0] for item in clean]
    ys = [item[1] for item in clean]
    fit = robust_affine(xs, ys, iterations=DEFAULT_HUBER_ITERATIONS) if len(clean) >= 3 else None
    if fit is None:
        slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
        intercept = ys[0] - slope * xs[0]
        residuals = [y - (intercept + slope * x_value) for x_value, y in clean]
        fit = {
            "slope_ms_per_unit": slope, "intercept_ms": intercept,
            "slope_raw_ms_per_unit": slope, "intercept_raw_ms": intercept,
            "residual_scale_ms": robust_scale(residuals),
            "residual_median_abs_ms": _median([abs(r) for r in residuals]),
            "residual_max_abs_ms": max((abs(r) for r in residuals), default=0.0),
            "relative_residual": 0.0, "n": len(clean),
            "x_min": xs[0], "x_max": xs[-1],
            "outliers": [], "residuals": [],
            "r2": 1.0, "measured_negative_slope": bool(slope < 0),
        }
    prediction = float(fit["intercept_ms"]) + float(fit["slope_ms_per_unit"]) * float(x)
    if prediction <= 0 or not math.isfinite(prediction):
        return None
    bracket_width = 0.0
    if lower_bound is not None and upper_bound is not None:
        bracket_width = max(0.0, float(upper_bound) - float(lower_bound))
    scale = float(fit.get("residual_scale_ms") or 0.0)
    uncertainty = scale + 0.25 * bracket_width * (2.0 / len(clean))
    return {
        "prediction_ms": prediction,
        "uncertainty_ms": uncertainty,
        "slope_ms_per_unit": float(fit["slope_ms_per_unit"]),
        "intercept_ms": float(fit["intercept_ms"]),
        "support": [[item[0], item[1]] for item in clean],
        "support_count": len(clean),
        "residual_scale_ms": scale,
        "residual_max_abs_ms": fit.get("residual_max_abs_ms"),
        "outliers": fit.get("outliers") or [],
        "monotone_projection": (
            PROJECTION_CONSTANT_LEVEL if float(fit["slope_ms_per_unit"]) < 0 else "none"),
    }


def monotone_log_plane(points: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Robust plane ``ln cost = a + b ln n + c ln k`` over bracketing curves.

    ``points`` carry ``u``, ``v``, ``cost_ms`` and optionally
    ``residual_scale_ms``.  The plane is the 2-D interpolation used by guarded
    coverage: at least ``MIN_2D_NODES`` measured curves, and the fit must be
    monotone in both coordinates (b >= 0 and c >= 0) or it is refused.
    """
    clean = [
        {
            "u": math.log(float(item["u"])), "v": math.log(float(item["v"])),
            "y": math.log(float(item["cost_ms"])),
            "cost_ms": float(item["cost_ms"]),
            "u_raw": float(item["u"]), "v_raw": float(item["v"]),
            "residual_scale_ms": float(item.get("residual_scale_ms") or 0.0),
            "source": item.get("source"),
        }
        for item in points
        if float(item.get("u") or 0) > 0 and float(item.get("v") or 0) > 0
        and float(item.get("cost_ms") or 0) > 0
    ]
    if len(clean) < MIN_2D_NODES:
        return None
    if len({(round(item["u_raw"], 12), round(item["v_raw"], 12)) for item in clean}) < MIN_2D_NODES:
        return None
    # Robust weights from the OLS plane residual scale (Huber reweighting).
    design = [[1.0, item["u"], item["v"]] for item in clean]
    targets = [item["y"] for item in clean]
    weights = [1.0] * len(clean)
    coefficients = _weighted_plane(design, targets, weights)
    if coefficients is None:
        return None
    for _ in range(DEFAULT_HUBER_ITERATIONS):
        residuals = [item["y"] - sum(c * d for c, d in zip(coefficients, row))
                     for item, row in zip(clean, design)]
        scale = robust_scale(residuals)
        delta = HUBER_DELTA_SIGMA * max(scale, 1e-12)
        weights = _huber_weights(residuals, delta)
        updated = _weighted_plane(design, targets, weights)
        if updated is None:
            break
        coefficients = updated
    residuals = [item["y"] - sum(c * d for c, d in zip(coefficients, row))
                 for item, row in zip(clean, design)]
    scale = robust_scale(residuals)
    rms = math.sqrt(sum(r * r for r in residuals) / len(residuals))
    slopes = {"n": float(coefficients[1]), "k": float(coefficients[2])}
    return {
        "intercept": float(coefficients[0]),
        "slopes": slopes, "in_log_space": True,
        "rms_log_residual": rms,
        "residual_scale_log": scale,
        "monotone": bool(slopes["n"] >= -1e-12 and slopes["k"] >= -1e-12),
        "nodes": [
            {"n": item["u_raw"], "k": item["v_raw"], "cost_ms": item["cost_ms"],
             "ln_residual": residual, "source": item["source"]}
            for item, residual in zip(clean, residuals)
        ],
        "node_count": len(clean),
        "iterations": DEFAULT_HUBER_ITERATIONS,
    }


def _weighted_plane(design: Sequence[Sequence[float]], targets: Sequence[float],
                    weights: Sequence[float]) -> list[float] | None:
    size = len(design[0])
    matrix = [[0.0] * size for _ in range(size)]
    vector = [0.0] * size
    for row, target, weight in zip(design, targets, weights):
        for i in range(size):
            vector[i] += weight * row[i] * target
            for j in range(size):
                matrix[i][j] += weight * row[i] * row[j]
    # Gaussian elimination with partial pivoting on a 3x3 normal system.
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(matrix[r][column]))
        if abs(matrix[pivot][column]) <= 1e-18:
            return None
        if pivot != column:
            matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
            vector[column], vector[pivot] = vector[pivot], vector[column]
        for r in range(column + 1, size):
            factor = matrix[r][column] / matrix[column][column]
            for c in range(column, size):
                matrix[r][c] -= factor * matrix[column][c]
            vector[r] -= factor * vector[column]
    solution = [0.0] * size
    for r in range(size - 1, -1, -1):
        total = vector[r] - sum(matrix[r][c] * solution[c] for c in range(r + 1, size))
        solution[r] = total / matrix[r][r]
    return solution


def predict_2d(plane: Mapping[str, Any], n: float, k: float) -> dict[str, Any] | None:
    if not plane.get("monotone"):
        return None
    value = float(plane["intercept"]) + float(plane["slopes"]["n"]) * math.log(float(n)) \
        + float(plane["slopes"]["k"]) * math.log(float(k))
    prediction = math.exp(value)
    if prediction <= 0 or not math.isfinite(prediction):
        return None
    # Structural disagreement: the weighted spread of the measured neighbourhood
    # around the interpolated value.  A single "nearest" node is ambiguous when
    # several corners are equidistant (and picking one arbitrary corner would
    # inflate the band), so the spread over every participating node is used.
    nodes = list(plane.get("nodes") or ())
    weights = []
    for node in nodes:
        distance = (abs(math.log(float(node["n"]) / float(n)))
                    + abs(math.log(float(node["k"]) / float(k))))
        weights.append(1.0 / (distance + 0.25))
    total = sum(weights)
    displacement = (
        math.sqrt(sum(w * (float(node["cost_ms"]) - prediction) ** 2
                      for node, w in zip(nodes, weights)) / total)
        if total > 0 else 0.0)
    nearest = min(
        nodes,
        key=lambda node: (abs(math.log(float(node["n"]) / float(n)))
                          + abs(math.log(float(node["k"]) / float(k)))), default=None)
    return {
        "prediction_ms": prediction,
        "rms_log_residual": float(plane.get("rms_log_residual") or 0.0),
        "displacement_ms": displacement,
        "nearest_node": nearest,
    }


def _predict_query(fit: Mapping[str, Any], x: float) -> dict[str, Any]:
    """Positive physical prediction with explicit transition/extrapolation policy."""
    segments = list(fit.get("segments") or ())
    if not segments:
        return {"prediction_ms": None, "region": "unsupported",
                "status": "no_fitted_segments", "support": None}
    try:
        value = float(x)
    except (TypeError, ValueError):
        return {"prediction_ms": None, "region": "unsupported",
                "status": "nonfinite_query", "support": None}
    if not math.isfinite(value):
        return {"prediction_ms": None, "region": "unsupported",
                "status": "nonfinite_query", "support": None}
    first, last = segments[0], segments[-1]
    low, high = float(first["x_min"]), float(last["x_max"])
    for index, segment in enumerate(segments):
        start = float(segment["x_min"])
        end = float(segment["x_max"])
        if start <= value <= end:
            prediction = _segment_value(segment, value)
            quality = _fit_quality(fit)
            if prediction < 0 or not math.isfinite(prediction):
                return {"prediction_ms": None, "region": "nonphysical_interpolation",
                        "status": "nonphysical_prediction", "support": [start, end],
                        "lower_bound_ms": 0.0,
                        "upper_bound_ms": max(0.0, _segment_value(segment, start),
                                              _segment_value(segment, end))}
            return {
                "prediction_ms": prediction,
                "region": "exact" if value in {start, end} else "interpolation",
                "status": _segment_status(fit, segment), "support": [start, end],
                "segment_index": index,
                "coverage": "exact" if value in {start, end} else "interpolated_1d",
                "uncertainty_ms": _segment_uncertainty(segment),
                "support_points": int(segment.get("n") or 0),
                "monotone_projection": segment.get("projection") or "none",
                "monotone_source": segment.get("monotone_source"),
                "fit_quality": quality or None,
            }
        if value < start and index > 0:
            previous = segments[index - 1]
            left = _segment_value(previous, float(previous["x_max"]))
            right = _segment_value(segment, start)
            lower, upper = sorted((max(0.0, left), max(0.0, right)))
            bracket_width = upper - lower
            if bracket_width <= max(1e-12, 1e-9 * max(1.0, upper)):
                # Both neighbouring segments publish the same level (a measured
                # plateau projected to a constant), so the bracket carries no
                # disagreement and must not hide a usable number behind an
                # "unresolved" label.
                statuses = {_segment_status(fit, previous), _segment_status(fit, segment)}
                status = "ok" if statuses == {"ok"} else (
                    "weak_fit" if "weak_fit" in statuses else "plateau_projected")
                return {
                    "prediction_ms": upper, "region": "transition_collapsed",
                    "status": status, "coverage": "interpolated_1d",
                    "support": [float(previous["x_max"]), start],
                    "lower_bound_ms": lower, "upper_bound_ms": upper,
                    "uncertainty_ms": max(
                        _segment_uncertainty(previous) or 0.0,
                        _segment_uncertainty(segment) or 0.0),
                    "monotone_projection": (
                        PROJECTION_CONSTANT_LEVEL
                        if PROJECTION_CONSTANT_LEVEL in {
                            previous.get("projection"), segment.get("projection")}
                        else "none"),
                    "fit_quality": _fit_quality(fit) or None,
                    "reason": "adjacent segments publish the same level; bracket collapsed",
                }
            interior = _bracket_support(fit, float(previous["x_max"]), start)
            if len(interior) >= 2:
                local = local_bracket_fit(interior, value,
                                          lower_bound=lower, upper_bound=upper)
                if local is not None:
                    return {
                        "prediction_ms": local["prediction_ms"],
                        "region": "transition_local",
                        "status": "transition_resolved_local",
                        "coverage": "interpolated_local",
                        "support": [float(previous["x_max"]), start],
                        "lower_bound_ms": lower, "upper_bound_ms": upper,
                        "uncertainty_ms": max(float(local["uncertainty_ms"]),
                                              float(local.get("residual_scale_ms") or 0.0)),
                        "local_fit": local,
                        "fit_quality": _fit_quality(fit) or None,
                        "reason": "resolved from >=2 measured points strictly inside the bracket",
                    }
            return {
                "prediction_ms": None,
                "region": "transition_bracket",
                "status": "unresolved_transition",
                "coverage": "interval_only",
                "support": [float(previous["x_max"]), start],
                "lower_bound_ms": lower, "upper_bound_ms": upper,
                "uncertainty_ms": bracket_width,
                "interior_support_count": len(interior),
                "reason": "no train support inside the transition bracket; endpoint bounds only",
            }
    if value < low:
        segment = first
        prediction = _segment_value(segment, value)
        region, distance = "extrapolation", low - value
    elif value > high:
        segment = last
        prediction = _segment_value(segment, value)
        region, distance = "extrapolation", value - high
    else:  # pragma: no cover - covered by the segment loop
        return {"prediction_ms": None, "region": "unsupported",
                "status": "no_segment_match", "support": [low, high]}
    if prediction < 0 or not math.isfinite(prediction):
        return {"prediction_ms": None, "region": "nonphysical_extrapolation",
                "status": "nonphysical_prediction", "support": [low, high],
                "distance_axis_units": distance, "lower_bound_ms": 0.0,
                "upper_bound_ms": max(0.0, _segment_value(segment, low if value < low else high))}
    return {
        "prediction_ms": prediction, "region": region, "status": "extrapolation",
        "coverage": "extrapolated",
        "support": [low, high], "distance_axis_units": distance,
        "uncertainty_ms": abs(float(segment.get("slope_ms_per_unit") or 0.0)) * distance,
        "uncertainty_plus_residual_ms": (
            abs(float(segment.get("slope_ms_per_unit") or 0.0)) * distance
            + (_segment_uncertainty(segment) or 0.0)),
        "monotone_projection": segment.get("projection") or "none",
        "fit_quality": _fit_quality(fit) or None,
    }


def _predict(fit: Mapping[str, Any], x: float) -> float | None:
    return _predict_query(fit, x)["prediction_ms"]


__all__ = [
    "robust_affine", "robust_scale", "r2_from", "local_bracket_fit",
    "monotone_log_plane", "predict_2d", "_predict", "_predict_query",
    "_segment_value", "_bracket_support", "MAX_RELATIVE_DROP",
    "MIN_2D_NODES", "MAX_2D_RMS_LOG_RESIDUAL", "MAX_2D_RELATIVE_UNCERTAINTY",
    "PROJECTION_CONSTANT_LEVEL",
    "MIN_R2_FOR_STRONG_FIT", "MAX_RELATIVE_RESIDUAL_FOR_STRONG_FIT",
]
