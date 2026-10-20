"""Positive physical prediction/query policy for fitted latency curves.

[Role]
  Resolve a pre-use axis value against a fitted piecewise curve: exact endpoint,
  interpolation, unresolved transition bracket, or labelled extrapolation.

[Invariants]
  * A transition bracket has no point prediction; it exposes positive endpoint
    bounds and unresolved support instead of back-extrapolating a narrow segment.
  * Non-positive/non-finite interpolation or extrapolation is missing, never
    clamped into false confidence.
  * Analysis and operational forecast both call this single policy.

[Timeouts & Locks]
  Pure bounded arithmetic; no I/O or locks.

[Side-Effect Policy]
  No side effects.
"""
from __future__ import annotations

import math
from typing import Any, Mapping


def _segment_value(segment: Mapping[str, Any], x: float) -> float:
    return float(segment["intercept_ms"]) + float(segment["slope_ms_per_unit"]) * x


def _predict_query(fit: Mapping[str, Any], x: float) -> dict[str, Any]:
    """Positive physical prediction with explicit transition/extrapolation policy.

    * inside a fitted segment: exact endpoint or linear interpolation;
    * inside a transition bracket between segments: no point prediction; return
      explicit lower/upper bounds and unresolved support;
    * outside the train envelope: bounded linear extrapolation from the nearest
      segment, labelled extrapolation; non-positive results are nonphysical and
      returned as missing, never clamped to fake confidence.
    """
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
            if prediction < 0 or not math.isfinite(prediction):
                return {"prediction_ms": None, "region": "nonphysical_interpolation",
                        "status": "nonphysical_prediction", "support": [start, end],
                        "lower_bound_ms": 0.0,
                        "upper_bound_ms": max(0.0, _segment_value(segment, start),
                                              _segment_value(segment, end))}
            return {
                "prediction_ms": prediction,
                "region": "exact" if value in {start, end} else "interpolation",
                "status": "ok", "support": [start, end],
                "segment_index": index,
                "uncertainty_ms": None,
                "support_points": int(segment.get("n") or 0),
            }
        if value < start and index > 0:
            previous = segments[index - 1]
            left = _segment_value(previous, float(previous["x_max"]))
            right = _segment_value(segment, start)
            lower, upper = sorted((max(0.0, left), max(0.0, right)))
            return {
                "prediction_ms": None,
                "region": "transition_bracket",
                "status": "unresolved_transition",
                "support": [float(previous["x_max"]), start],
                "lower_bound_ms": lower, "upper_bound_ms": upper,
                "uncertainty_ms": upper - lower,
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
        "support": [low, high], "distance_axis_units": distance,
        "uncertainty_ms": abs(float(segment["slope_ms_per_unit"])) * distance,
    }


def _predict(fit: Mapping[str, Any], x: float) -> float | None:
    return _predict_query(fit, x)["prediction_ms"]


