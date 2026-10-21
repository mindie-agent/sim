"""Raw-sample curve analysis for the operator shape-linearity campaign.

[Role]
  Join a frozen campaign manifest to worker results, preserve raw trials,
  compute per-point median/spread and independent trial counts, fit piecewise
  affine regimes on train points only, evaluate heldout points separately,
  report local secants/residuals, launch-floor and saturation hints, and list
  pending/missing coverage.  Never pool different dtype/layout/recipe/backend/
  protocol conditions into one average.

[Invariants]
  * A group is one condition, one varied axis, one protocol/observed backend.
  * Fit inputs are train points only; heldout points are evaluation-only and
    root owns acceptance thresholds.
  * Repeats within a trial are not independent cases.  Independent N counts
    completed trials, not event iterations.
  * Pending/failed/unmeasured cases remain in coverage with reasons; they are
    never removed from the denominator or replaced by zero.
  * Segments carry at least three points whenever the curve can be split; the
    segment count is chosen by BIC on a robust (Huber) loss and every candidate
    is reported.  Residuals and outliers stay visible; no point is dropped.
  * A measured negative slope is reported as measured and projected onto a
    constant level; segment knots are isotonically projected before publication,
    so a published curve never decreases with the work axis.  Every projection
    reports the measured slope and its size, and a weak fit is labelled weak
    rather than presented as ``ok``.
  * Affine intercept is labelled constant cost, not proven launch latency;
    plateau is labelled a hint, not proven hardware saturation.
  * No source SHA/version is an admission gate; manifest/results paths are
    data references only.

[Timeouts & Locks]
  Pure local JSON/arithmetic bounded by the number of manifest requests.
  No device, network, subprocess, lock or sleep.

[Side-Effect Policy]
  Import writes nothing.  The CLI writes only the explicit output directory.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .a3_operator_campaign import load_manifest
from .a3_operator_curve_axis import fixed_shape as _axis_fixed_shape
from .a3_operator_curve_protocol import protocol_signature
from .a3_operator_curve_query import (
    MAX_RELATIVE_DROP, MAX_RELATIVE_RESIDUAL_FOR_STRONG_FIT,
    MIN_R2_FOR_STRONG_FIT, PROJECTION_CONSTANT_LEVEL,
    _predict, _predict_query, robust_affine,
)

# A segment is only published as a fitted trend when at least three independent
# points support it; curves with 2-5 points fit one segment because a 2-point
# segment has r2 == 1 by construction and cannot be validated.
DEFAULT_MIN_SEGMENT_POINTS = 3
DEFAULT_MAX_SEGMENTS = 3
DEFAULT_JUMP_THRESHOLD = 0.15


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _mad(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    center = _median(values)
    return float(statistics.median([abs(value - center) for value in values]))


def _p90(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(0.9 * len(ordered)) - 1))
    return float(ordered[index])


def _protocol_signature(protocol: Mapping[str, Any], observed: Mapping[str, Any]) -> str:
    """Single canonical protocol signature (see curve_protocol helper)."""
    return protocol_signature(protocol, observed)


def _result_points(manifest: Mapping[str, Any], results: Mapping[str, Any]) -> tuple[
    dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]
]:
    requests = {str(item.get("measurement_id") or item.get("request_key")): item
                for item in manifest.get("requests", ())}
    points_by_measurement: dict[tuple[str, str, str], dict[str, Any]] = {}
    seen_rows: set[str] = set()
    unmeasured: list[dict[str, Any]] = []
    counts = {"completed": 0, "pending": 0, "failed": 0, "dry_run_validated": 0,
              "unknown": 0, "duplicate_rows": 0, "derived_excluded": 0}
    for result in results.get("results", ()):
        row_identity = _canonical({
            "measurement_id": result.get("measurement_id") or result.get("request_key"),
            "run_id": result.get("run_id"),
            "status": result.get("status"),
            "trials": result.get("trials"),
            "raw": result.get("raw_samples_ms"),
            "correctness": result.get("correctness"),
        })
        if row_identity in seen_rows:
            counts["duplicate_rows"] += 1
            continue
        seen_rows.add(row_identity)
        midpoint = str(result.get("measurement_id") or result.get("request_key"))
        request = requests.get(midpoint)
        status = str(result.get("status", "unknown"))
        if status == "completed":
            correctness = result.get("correctness") or {}
            correctness_status = str(correctness.get("status", "not_applicable"))
            if correctness_status not in {"passed", "not_applicable"}:
                counts["failed"] += 1
                unmeasured.append({
                    "measurement_id": midpoint, "condition_key": result.get("condition_key"),
                    "status": "failed_correctness",
                    "reason": correctness.get("reason") or f"correctness_status:{correctness_status}",
                    "required_backend": None, "split": result.get("split"),
                })
                continue
            source_class = str(
                result.get("source_class") or request.get("source_class") or "unspecified"
            ) if request else "unspecified"
            explicit_not_measured = bool(
                result.get("not_a3_measured")
                or (result.get("environment") or {}).get("not_a3_measured")
                or ((request.get("environment") or {}) if request else {}).get("not_a3_measured")
            )
            if explicit_not_measured or source_class in {
                "derived", "synthetic", "cpu", "cpu_reference", "cpu_reference_runner",
            } or source_class not in {"measured", "observed", "unspecified"}:
                counts["derived_excluded"] += 1
                unmeasured.append({
                    "measurement_id": midpoint,
                    "condition_key": result.get("condition_key"),
                    "status": "derived_or_unmeasured_excluded",
                    "reason": f"source_class={source_class}; explicit not-measured={explicit_not_measured}",
                    "required_backend": "A3 measured/observed source evidence",
                    "split": result.get("split"),
                })
                continue
            if request is None:
                counts["unknown"] += 1
                unmeasured.append({
                    "measurement_id": midpoint,
                    "status": "orphan_result_without_manifest_request",
                    "reason": "result does not match a frozen request",
                    "required_backend": None,
                })
                continue
            protocol = dict(result.get("protocol") or request.get("protocol") or {})
            observed = dict(result.get("observed_context") or {})
            signature = _protocol_signature(protocol, observed)
            run_id = str(result.get("run_id") or request.get("run_id") or "")
            point_identity = (midpoint, run_id, signature)
            trials = result.get("trials") or []
            trial_map: dict[str, dict[str, Any]] = {}
            for index, trial in enumerate(trials):
                trial_id = str(trial.get("trial_index", index))
                trial_map.setdefault(trial_id, trial)
            if not trial_map and result.get("raw_samples_ms"):
                trial_map["0"] = {"trial_index": 0,
                                  "raw_samples_ms": list(result.get("raw_samples_ms") or ()),
                                  "median_ms": _median([float(v) for v in result.get("raw_samples_ms")])}
            if not trial_map:
                counts["failed"] += 1
                unmeasured.append({
                    "measurement_id": midpoint, "condition_key": request.get("condition_key"),
                    "status": "failed_no_raw_samples", "reason": "completed result has no raw trial",
                    "required_backend": None, "split": request.get("split"),
                })
                continue
            if point_identity in points_by_measurement:
                point = points_by_measurement[point_identity]
                for trial_id, trial in trial_map.items():
                    point["trial_map"].setdefault((run_id, trial_id), trial)
            else:
                point_shape = dict(request.get("point", {}).get("shape") or {})
                axis = str(request.get("point", {}).get("axis") or "elements")
                fixed_shape = _axis_fixed_shape(axis, point_shape)
                if axis == "q" and request.get("condition", {}).get("shape_kind") in {
                        "attention", "attention_softmax"}:
                    fixed_shape.pop("kv", None)
                point = {
                    "measurement_id": midpoint,
                    "condition_key": str(request.get("condition_key")),
                    "condition_shape": _canonical(request.get("condition", {}).get("shape", {})),
                    "fixed_shape": _canonical(fixed_shape),
                    "axis": axis,
                    "x": float(request.get("point", {}).get("value", 0.0)),
                    "shape": request.get("point", {}).get("shape"),
                    "split": str(request.get("split", "train")),
                    "hardware_key": str(request.get("hardware_key") or ""),
                    "source_class": source_class,
                    "source": result.get("source") or request.get("source"),
                    "run_id": run_id,
                    "execution_kind": request.get("execution_kind"),
                    "worker_recipe": request.get("worker_recipe"),
                    "backend": result.get("backend"),
                    "observed_api": observed.get("observed_api"),
                    "protocol_signature": signature,
                    "trial_map": trial_map,
                    "analytic_ms": (
                        float(result["analytic_ms"]) if result.get("analytic_ms") is not None
                        else float(request["analytic_ms"]) if request.get("analytic_ms") is not None
                        else None
                    ),
                    "work": dict(request.get("work") or {}),
                    "correctness": correctness,
                    "extrapolation": bool(request.get("extrapolation")),
                }
                point["trial_map"] = {(run_id, trial_id): trial
                                      for trial_id, trial in trial_map.items()}
                points_by_measurement[point_identity] = point
        else:
            if status == "dry_run_validated":
                counts["dry_run_validated"] += 1
            elif status in {"pending", "pending_backend"}:
                counts["pending"] += 1
            elif status == "failed":
                counts["failed"] += 1
            else:
                counts["unknown"] += 1
            unmeasured.append({
                "measurement_id": result.get("measurement_id"),
                "condition_key": result.get("condition_key"),
                "execution_kind": result.get("execution_kind"),
                "worker_recipe": result.get("worker_recipe"),
                "split": result.get("split"),
                "status": status,
                "reason": result.get("reason") or "no completed samples",
                "required_backend": result.get("required_backend"),
            })
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for point in points_by_measurement.values():
        trial_items = list(point.pop("trial_map").values())
        trial_medians, raw = [], []
        for trial in trial_items:
            raw_values = [float(value) for value in (trial.get("raw_samples_ms") or ())]
            raw.extend(raw_values)
            median = trial.get("median_ms")
            if median is None and raw_values:
                median = _median(raw_values)
            if median is not None:
                trial_medians.append(float(median))
        if not raw:
            raw = trial_medians
        if not raw:
            counts["failed"] += 1
            unmeasured.append({
                "measurement_id": point["measurement_id"],
                "condition_key": point["condition_key"],
                "status": "failed_no_raw_samples",
                "reason": "unique trials have no raw sample", "required_backend": None,
                "split": point["split"],
            })
            continue
        point.update({
            "trial_medians_ms": trial_medians,
            "trial_count": len(trial_medians),
            "raw_sample_count": len(raw),
            "median_ms": _median(trial_medians),
            "mad_ms": _mad(trial_medians) if len(trial_medians) > 1 else _mad(raw),
            "min_ms": min(raw), "max_ms": max(raw),
            "p90_ms": _p90(trial_medians) if trial_medians else _p90(raw),
            "raw_p90_ms": _p90(raw),
        })
        median_ms = point["median_ms"]
        work = point["work"]
        point["rate"] = (float(work.get("rate_work", 0.0)) / median_ms
                         if work.get("rate_work") and median_ms > 0 else None)
        point["work_unit"] = work.get("unit")
        point["traffic_bytes"] = work.get("traffic_bytes")
        group = (point["condition_key"], point["axis"], point["protocol_signature"],
                 point["condition_shape"], point["fixed_shape"], point["hardware_key"],
                 str(point.get("source_class")))
        groups.setdefault(group, []).append(point)
    counts["completed"] = len(points_by_measurement)
    return groups, unmeasured, counts


def _merge_same_x(points: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[float]]:
    """Merge repeated measurements only within the same precommitted split."""
    merged: dict[tuple[float, str], dict[str, Any]] = {}
    for point in points:
        key = (float(point["x"]), str(point["split"]))
        if key not in merged:
            merged[key] = {**point, "trial_medians_ms": list(point["trial_medians_ms"])}
        else:
            current = merged[key]
            current["trial_medians_ms"].extend(point["trial_medians_ms"])
            current["trial_count"] = len(current["trial_medians_ms"])
            current["raw_sample_count"] += int(point["raw_sample_count"])
            current["median_ms"] = _median(current["trial_medians_ms"])
            current["mad_ms"] = _mad(current["trial_medians_ms"])
            current["min_ms"] = min(current["min_ms"], point["min_ms"])
            current["max_ms"] = max(current["max_ms"], point["max_ms"])
            current["p90_ms"] = _p90(current["trial_medians_ms"])
    train_x = {float(p["x"]) for p in merged.values() if p["split"] == "train"}
    held_x = {float(p["x"]) for p in merged.values() if p["split"] != "train"}
    overlap = sorted(train_x & held_x)
    return list(merged.values()), overlap


def _effective_segment_min(n: int, min_points: int) -> int:
    """Minimum support per segment, honest about small curves.

    Three points per segment whenever the curve can be split at all
    (``n >= 2 * min_points``); otherwise the whole curve is one segment, because
    a two-point segment is not evidence of a regime.
    """
    if n >= 2 * max(2, int(min_points)):
        return max(2, int(min_points))
    return max(2, n)


def _huber_loss(fit: Mapping[str, Any]) -> float:
    """Bounded Huber loss of a fitted segment (reported, never a silent drop)."""
    delta = float(fit.get("huber_delta_ms") or 0.0)
    total = 0.0
    for item in fit.get("residuals") or ():
        residual = abs(float(item["residual_ms"]))
        if delta <= 0:
            total += 0.0
        elif residual <= delta:
            total += 0.5 * residual * residual
        else:
            total += delta * (residual - 0.5 * delta)
    return total


def _fit_piecewise(points: Sequence[Mapping[str, Any]], *, max_segments: int,
                   min_points: int) -> dict[str, Any] | None:
    """Robust piecewise-affine fit with a documented segment-count criterion.

    * Segment model: Huber-loss affine fit (IRLS, MAD scale), not OLS, so one
      gross repeat cannot tilt a segment.
    * Segment count: dynamic program over minimum robust loss, selected by BIC
      on that robust loss (``criterion="bic_on_robust_huber_loss"``); every
      candidate count is reported so the choice is auditable.
    * A measured negative slope is reported as measured and then projected onto
      a constant level, so a published curve never decreases with the probed
      work axis.  The projection is labelled, never silent.
    * Residuals, outliers and raw slopes are kept for every segment.
    """
    ordered = sorted(points, key=lambda item: (float(item["x"]), float(item["median_ms"])))
    xs = [float(item["x"]) for item in ordered]
    ys = [float(item["median_ms"]) for item in ordered]
    n = len(xs)
    if n < 2 or len(set(xs)) < 2:
        return None
    segment_min = _effective_segment_min(n, min_points)
    segment_limit = max(1, min(int(max_segments), n // segment_min))
    fits: dict[tuple[int, int], dict[str, Any]] = {}
    loss: dict[tuple[int, int], float] = {}
    for start in range(n):
        for end in range(start + segment_min, n + 1):
            fit = robust_affine(xs[start:end], ys[start:end])
            if fit is None:
                continue
            fits[(start, end)] = fit
            loss[(start, end)] = _huber_loss(fit)
    if not fits:
        return None
    mean_y = sum(ys) / n
    loss_floor = max(1e-18, 1e-12 * n * mean_y * mean_y)

    def _bic(segments: int, total_loss: float) -> float:
        return n * math.log(max(loss_floor, total_loss) / n) + (2 * segments) * math.log(max(2, n))

    dp = [[float("inf")] * (n + 1) for _ in range(segment_limit + 1)]
    back = [[-1] * (n + 1) for _ in range(segment_limit + 1)]
    dp[0][0] = 0.0
    for segments in range(1, segment_limit + 1):
        for end in range(1, n + 1):
            for start in range(0, end):
                if dp[segments - 1][start] == float("inf"):
                    continue
                if (start, end) not in loss:
                    continue
                candidate = dp[segments - 1][start] + loss[(start, end)]
                if candidate < dp[segments][end]:
                    dp[segments][end] = candidate
                    back[segments][end] = start
    candidates = []
    best_segments = None
    best_bic = float("inf")
    for segments in range(1, segment_limit + 1):
        total = dp[segments][n]
        if total == float("inf"):
            continue
        bic = _bic(segments, total)
        candidates.append({"segments": segments, "robust_loss": total, "bic": bic})
        if bic < best_bic:
            best_bic, best_segments = bic, segments
    if not best_segments:
        return None
    bounds = [n]
    cursor = n
    for segments in range(best_segments, 0, -1):
        cursor = back[segments][cursor]
        bounds.append(cursor)
    bounds.reverse()
    segment_items: list[dict[str, Any]] = []
    previous_published_end: float | None = None
    monotone_breaks = 0
    for left, right in zip(bounds[:-1], bounds[1:]):
        fit = fits.get((left, right))
        if fit is None:
            continue
        start_x, end_x = float(fit["x_min"]), float(fit["x_max"])
        slope = float(fit["slope_ms_per_unit"])
        intercept = float(fit["intercept_ms"])
        sxx = sum((x - sum(xs[left:right]) / (right - left)) ** 2 for x in xs[left:right])
        scale = float(fit["residual_scale_ms"])
        standard_error = (scale / math.sqrt(sxx)) if sxx > 0 and scale > 0 else 0.0
        drop = float(fit["relative_drop_fraction"])
        projected = bool(
            slope < 0 and (drop > MAX_RELATIVE_DROP or abs(slope) > 2.0 * standard_error)
        )
        level = float(fit["level_ms"])
        if projected and previous_published_end is not None:
            # Keep the published curve non-decreasing across the segment join.
            level = max(level, previous_published_end)
        if projected:
            published_end = level
            uncertainty = max(scale, 0.5 * (
                max(ys[left:right]) - min(ys[left:right])))
            start_value = published_end = level
        else:
            start_value = intercept + slope * start_x
            published_end = intercept + slope * end_x
            uncertainty = scale
            if previous_published_end is not None and start_value < previous_published_end:
                monotone_breaks += 1
        item = {
            "x_min": start_x, "x_max": end_x,
            "slope_ms_per_unit": 0.0 if projected else slope,
            "intercept_ms": level if projected else intercept,
            "n": right - left, "r2": float(fit["r2"]),
            "sse": float(fit["sse"]),
            "slope_raw_ms_per_unit": float(fit.get("slope_raw_ms_per_unit") or slope),
            "intercept_raw_ms": float(fit.get("intercept_raw_ms") or intercept),
            "slope_fitted_ms_per_unit": slope,
            "huber_delta_ms": float(fit.get("huber_delta_ms") or 0.0),
            "huber_iterations": int(fit.get("iterations") or 0),
            "residual_scale_ms": scale,
            "residual_median_abs_ms": float(fit.get("residual_median_abs_ms") or 0.0),
            "residual_p90_abs_ms": float(fit.get("residual_p90_abs_ms") or 0.0),
            "residual_max_abs_ms": float(fit.get("residual_max_abs_ms") or 0.0),
            "relative_residual": float(fit.get("relative_residual") or 0.0),
            "relative_drop_fraction": drop,
            "standard_error_ms_per_unit": standard_error,
            "measured_negative_slope": bool(slope < 0),
            "negative_slope_significant": bool(slope < 0 and abs(slope) > 2.0 * standard_error),
            "projection": PROJECTION_CONSTANT_LEVEL if projected else "none",
            "level_ms": level if projected else None,
            "published_start_ms": start_value,
            "published_end_ms": published_end,
            "uncertainty_ms": uncertainty,
            "outliers": list(fit.get("outliers") or ()),
            "outlier_count": len(fit.get("outliers") or ()),
            "publishable": True,
        }
        segment_items.append(item)
        previous_published_end = published_end
    if not segment_items:
        return None
    # Monotone publication by isotonic regression on the segment knots.
    #
    # Two independently fitted segments can meet with the later one below the
    # earlier one (measurement noise, or a real transient drop the layer must not
    # claim as a trend).  The published knot values are projected onto the
    # non-decreasing cone with pool-adjacent-violators, which is the least-squares
    # monotone fit of those knots: it neither over-corrects a join upward (a rigid
    # shift) nor publishes a decreasing cost.  Each segment is then re-expressed
    # through its two projected knots; the change is reported and widens the
    # segment's uncertainty instead of being hidden.
    knots = []
    for index, item in enumerate(segment_items):
        weight = max(1.0, float(item["n"]))
        knots.append({"index": index, "edge": "start", "x": float(item["x_min"]),
                      "y": float(item["published_start_ms"]), "weight": weight})
        knots.append({"index": index, "edge": "end", "x": float(item["x_max"]),
                      "y": float(item["published_end_ms"]), "weight": weight})
    tolerance = max(1e-12, 1e-9 * max(knot["y"] for knot in knots))
    blocks: list[dict[str, Any]] = []
    for knot in knots:
        blocks.append({"weight": knot["weight"],
                       "weighted": knot["weight"] * knot["y"],
                       "knots": [knot]})
        while len(blocks) > 1:
            left, right = blocks[-2], blocks[-1]
            left_level = left["weighted"] / left["weight"]
            right_level = right["weighted"] / right["weight"]
            if left_level <= right_level + tolerance:
                break
            blocks.pop()
            blocks.pop()
            blocks.append({"weight": left["weight"] + right["weight"],
                           "weighted": left["weighted"] + right["weighted"],
                           "knots": left["knots"] + right["knots"]})
    projected: dict[tuple[int, str], float] = {}
    for block in blocks:
        level = block["weighted"] / block["weight"]
        for knot in block["knots"]:
            projected[(knot["index"], knot["edge"])] = level
    for index, item in enumerate(segment_items):
        start_value = projected[(index, "start")]
        end_value = projected[(index, "end")]
        original_start = float(item["published_start_ms"])
        original_end = float(item["published_end_ms"])
        change = max(abs(start_value - original_start), abs(end_value - original_end))
        span = float(item["x_max"]) - float(item["x_min"])
        if change > tolerance:
            item["monotone_projection_ms"] = change
            item["uncertainty_ms"] = max(float(item["uncertainty_ms"] or 0.0), change)
        item["published_start_ms"] = start_value
        item["published_end_ms"] = end_value
        if span <= 0:
            continue
        if abs(end_value - start_value) <= tolerance:
            if item["projection"] == PROJECTION_CONSTANT_LEVEL:
                # Already projected from a measured negative slope.
                item["monotone_source"] = "negative_slope"
                item["level_ms"] = start_value
            elif change > tolerance:
                # PAVA pooled the knots into one level: publish it, but label why.
                item["projection"] = PROJECTION_CONSTANT_LEVEL
                item["monotone_source"] = "isotonic_pooling"
                item["level_ms"] = start_value
            else:
                # A genuinely horizontal measured line is not a projection.
                item["projection"] = "none"
                item["level_ms"] = None
            item["slope_ms_per_unit"] = 0.0
            item["intercept_ms"] = start_value
        else:
            slope = (end_value - start_value) / span
            item["slope_ms_per_unit"] = slope
            item["intercept_ms"] = start_value - slope * float(item["x_min"])
            if change > tolerance:
                item["monotone_tilt"] = True
                item["monotone_source"] = "isotonic_tilt"
    brackets = []
    for left, right in zip(segment_items[:-1], segment_items[1:]):
        brackets.append([float(left["x_max"]), float(right["x_min"])])
    published = []
    for x, y in zip(xs, ys):
        predicted = _predict({"segments": segment_items}, x)
        published.append(abs(predicted - y) if predicted is not None else None)
    scored = [value for value in published if value is not None]
    sse_published = sum(value * value for value in scored) if len(scored) == n else None
    sst = sum((y - mean_y) ** 2 for y in ys)
    relative_residual = (sum(scored) / len(scored)) / mean_y if scored and mean_y > 0 else 0.0
    quality_status = (
        "weak" if relative_residual > MAX_RELATIVE_RESIDUAL_FOR_STRONG_FIT else "strong"
    )
    projected_segments = [
        index for index, item in enumerate(segment_items)
        if item["projection"] == PROJECTION_CONSTANT_LEVEL
    ]
    return {
        "kind": "piecewise" if len(segment_items) > 1 else "linear",
        "segments": segment_items,
        "transition_brackets": brackets,
        "bracket_support": {f"{lo},{hi}": [] for lo, hi in brackets},
        "bic": best_bic,
        "robust_loss": dp[best_segments][n],
        "train_points": n,
        "selection": {
            "criterion": "bic_on_robust_huber_loss",
            "max_segments": segment_limit,
            "min_segment_points_requested": int(min_points),
            "min_segment_points_effective": segment_min,
            "chosen_segments": best_segments,
            "candidates": candidates,
        },
        "robust": {
            "loss": "huber",
            "delta_sigma": 1.345,
            "outlier_sigma": 3.0,
            "note": "IRLS with a MAD scale per segment; raw OLS slope retained for audit",
        },
        "fit_quality": {
            "status": quality_status,
            "published_r2": (
                None if sse_published is None
                else 1.0 - sse_published / sst if sst > 0 else 1.0
            ),
            "published_relative_residual": relative_residual,
            "projected_segments": projected_segments,
            "monotone_breaks": monotone_breaks,
            "outlier_count": sum(item["outlier_count"] for item in segment_items),
            "min_segment_points_effective": segment_min,
            "r2_floor_for_strong_fit": MIN_R2_FOR_STRONG_FIT,
            "relative_residual_floor_for_strong_fit": MAX_RELATIVE_RESIDUAL_FOR_STRONG_FIT,
        },
    }



def _secants(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(points, key=lambda item: float(item["x"]))
    result = []
    for left, right in zip(ordered[:-1], ordered[1:]):
        dx = float(right["x"]) - float(left["x"])
        if dx <= 0:
            continue
        dy = float(right["median_ms"]) - float(left["median_ms"])
        result.append({
            "from_x": float(left["x"]), "to_x": float(right["x"]), "dx": dx,
            "dy_ms": dy, "secant_ms_per_unit": dy / dx,
            "from_median_ms": float(left["median_ms"]),
            "to_median_ms": float(right["median_ms"]),
            "relative_jump": dy / max(1e-12, float(left["median_ms"])),
        })
    return result


def _jumps(points: Sequence[Mapping[str, Any]], threshold: float) -> list[dict[str, Any]]:
    result = []
    for secant in _secants(points):
        if abs(secant["relative_jump"]) >= threshold:
            result.append({
                "from_x": secant["from_x"], "to_x": secant["to_x"],
                "delta_ms": secant["dy_ms"],
                "relative": secant["relative_jump"],
                "diagnostic": "adjacent timing jump; not used to choose the fit",
            })
    return result


def _heldout_evaluation(fit: Mapping[str, Any], train: Sequence[Mapping[str, Any]],
                        heldout: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not heldout:
        return {"status": "no_heldout_points", "count": 0}
    rows = []
    for point in sorted(heldout, key=lambda item: float(item["x"])):
        x = float(point["x"])
        query = _predict_query(fit, x)
        actual = float(point["median_ms"])
        row = {"x": x, "actual_median_ms": actual,
               "predicted_ms": query.get("prediction_ms"),
               "region": query.get("region"), "prediction_status": query.get("status"),
               "support": query.get("support"),
               "lower_bound_ms": query.get("lower_bound_ms"),
               "upper_bound_ms": query.get("upper_bound_ms"),
               "uncertainty_ms": query.get("uncertainty_ms")}
        if query.get("prediction_ms") is not None:
            predicted = float(query["prediction_ms"])
            row["ape"] = abs(predicted - actual) / max(1e-12, actual)
            row["signed_error_ms"] = predicted - actual
        rows.append(row)
    scored = [row for row in rows if "ape" in row]
    apes = [row["ape"] for row in scored]
    return {
        "status": "reported_not_an_acceptance_gate",
        "count": len(rows), "scored_count": len(scored),
        "unresolved_count": len(rows) - len(scored),
        "median_ape": _median(apes) if apes else None,
        "p90_ape": _p90(apes) if apes else None,
        "max_ape": max(apes) if apes else None,
        "median_signed_bias_ms": _median([row["signed_error_ms"] for row in scored]) if scored else None,
        "interpolation_count": sum(1 for row in rows if row["region"] == "interpolation"),
        "exact_count": sum(1 for row in rows if row["region"] == "exact"),
        "transition_count": sum(1 for row in rows if row["region"] == "transition_bracket"),
        "extrapolation_count": sum(1 for row in rows if row["region"] == "extrapolation"),
        "nonphysical_count": sum(1 for row in rows if "nonphysical" in str(row["region"])),
        "points": rows,
        "note": "fit used train points only; transition brackets are unresolved, not clamped",
    }


def _fit_summary(group: Mapping[str, Any], points: Sequence[Mapping[str, Any]],
                 *, max_segments: int, min_points: int, jump_threshold: float,
                 split_overlap: Sequence[float] = ()) -> dict[str, Any]:
    train = [point for point in points if point["split"] == "train"]
    heldout = [point for point in points if point["split"] != "train"]
    fit = None if split_overlap else _fit_piecewise(
        train, max_segments=max_segments, min_points=min_points)
    summary: dict[str, Any] = {
        "condition_key": group["condition_key"], "axis": group["axis"],
        "condition_shape": group.get("condition_shape"),
        "fixed_shape": group.get("fixed_shape"),
        "hardware_key": group.get("hardware_key"),
        "source_class": group.get("source_class"),
        "protocol_signature": group["signature"],
        "split_overlap_x": list(split_overlap),
        "status": "rejected_split_overlap" if split_overlap else "fitted",
        "execution_kind": points[0].get("execution_kind") if points else None,
        "worker_recipe": points[0].get("worker_recipe") if points else None,
        "observed_api": points[0].get("observed_api") if points else None,
        "independent_train_points": len(train),
        "independent_heldout_points": len(heldout),
        "raw_train_samples": sum(point["raw_sample_count"] for point in train),
        "raw_heldout_samples": sum(point["raw_sample_count"] for point in heldout),
        "independent_train_trials": sum(point["trial_count"] for point in train),
        "independent_heldout_trials": sum(point["trial_count"] for point in heldout),
        "train_points": [_point_summary(point) for point in sorted(train, key=lambda item: item["x"])],
        "heldout_points": [_point_summary(point) for point in sorted(heldout, key=lambda item: item["x"])],
        "local_secants_train": _secants(train),
        "train_jumps": _jumps(train, jump_threshold),
        "heldout_jumps": _jumps(heldout, jump_threshold),
        "fit": fit,
        "heldout_evaluation": (
            {"status": "rejected_split_overlap", "count": len(heldout)}
            if split_overlap else
            _heldout_evaluation(fit, train, heldout) if fit and train else {
                "status": "not_fitted_insufficient_train_points", "count": len(heldout),
            }
        ),
        "work_rate": _work_rate(points),
    }
    if fit:
        first = fit["segments"][0]
        last = fit["segments"][-1]
        smallest = min(train, key=lambda item: float(item["x"]))
        intercept_ratio = float(first["intercept_ms"]) / max(1e-12, float(smallest["median_ms"]))
        summary["launch_floor_hint"] = {
            "constant_cost_ms": float(first["intercept_ms"]),
            "relative_to_smallest_train_median": intercept_ratio,
            "hint": bool(intercept_ratio > 0.15 and float(first["intercept_ms"]) > 0),
            "note": "affine intercept is constant cost in this regime, not proven launch latency",
        }
        secants = summary["local_secants_train"]
        saturation = bool(
            len(fit["segments"]) > 1
            and abs(float(last["slope_ms_per_unit"])) <= 0.2 * max(
                abs(float(first["slope_ms_per_unit"])), 1e-12)
            and len(secants) >= 3
        )
        summary["saturation_hint"] = {
            "hint": saturation,
            "first_slope": float(first["slope_ms_per_unit"]),
            "last_slope": float(last["slope_ms_per_unit"]),
            "note": "slope/plateau observation only; no hardware saturation claim without counters",
        }
        residuals = []
        for point in train:
            predicted = _predict(fit, float(point["x"]))
            if predicted is not None:
                residuals.append(abs(predicted - float(point["median_ms"])))
        summary["train_residuals_ms"] = {
            "median_abs": _median(residuals), "max_abs": max(residuals) if residuals else 0.0,
            "count": len(residuals),
        }
        summary["fit_quality"] = dict(fit.get("fit_quality") or {})
        summary["fit_selection"] = dict(fit.get("selection") or {})
        summary["transition_brackets"] = list(fit.get("transition_brackets") or ())
        summary["outliers"] = [
            {"segment": index, "x": item["x"], "residual_ms": item["residual_ms"]}
            for index, segment in enumerate(fit["segments"])
            for item in segment.get("outliers") or ()
        ]
    return summary


def _point_summary(point: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "x": point["x"], "shape": point.get("shape"),
        "split": point["split"], "trial_count": point["trial_count"],
        "raw_sample_count": point["raw_sample_count"],
        "median_ms": point["median_ms"], "mad_ms": point["mad_ms"],
        "min_ms": point["min_ms"], "max_ms": point["max_ms"], "p90_ms": point["p90_ms"],
        "analytic_ms": point["analytic_ms"], "rate": point["rate"],
        "work_unit": point["work_unit"], "traffic_bytes": point["traffic_bytes"],
        "extrapolation": point["extrapolation"],
        # Every raw trial median is retained: fitting may not discard evidence.
        "trial_medians_ms": list(point.get("trial_medians_ms") or ()),
        "raw_samples_ms": list(point.get("raw_samples_ms") or ()),
    }


def _work_rate(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rates = [(float(point["rate"]), point.get("work_unit"), point.get("traffic_bytes"))
             for point in points if point.get("rate") is not None]
    if not rates:
        return {"status": "no_work_rate"}
    values = [item[0] for item in rates]
    return {
        "unit": rates[0][1], "median_work_per_ms": _median(values),
        "min": min(values), "max": max(values), "count": len(values),
        "traffic_provenance": "logical bytes from the campaign work estimate, not measured counters",
    }


from .a3_operator_curves_report import _coverage, _markdown


def analyze(manifest: Mapping[str, Any], results: Mapping[str, Any], *,
            max_segments: int = DEFAULT_MAX_SEGMENTS,
            min_segment_points: int = DEFAULT_MIN_SEGMENT_POINTS,
            jump_threshold: float = DEFAULT_JUMP_THRESHOLD) -> dict[str, Any]:
    groups, unmeasured, counts = _result_points(manifest, results)
    analyses = []
    overlaps = []
    for (condition_key, axis, signature, condition_shape, fixed_shape, hardware_key,
         source_class), raw_points in sorted(groups.items()):
        merged, split_overlap = _merge_same_x(raw_points)
        if split_overlap:
            overlaps.append({"condition_key": condition_key, "axis": axis,
                             "x": list(split_overlap)})
        analyses.append(_fit_summary(
            {"condition_key": condition_key, "axis": axis, "signature": signature,
             "condition_shape": condition_shape, "fixed_shape": fixed_shape,
             "hardware_key": hardware_key, "source_class": source_class},
            merged, max_segments=max_segments, min_points=min_segment_points,
            jump_threshold=jump_threshold, split_overlap=split_overlap,
        ))
    measured_request_ids = {
        str(item.get("measurement_id") or item.get("request_key"))
        for item in manifest.get("requests", ())
        if str(item.get("measurement_id") or item.get("request_key")) in {
            str(result.get("measurement_id")) for result in results.get("results", ())
            if result.get("status") == "completed"
        }
    }
    unselected = [
        {
            "condition_key": request.get("condition_key"),
            "family": request.get("op_family"),
            "phase": request.get("phase"),
            "recipe": request.get("recipe"),
            "execution_kind": request.get("execution_kind"),
            "split": request.get("split"),
            "reason": "selected_in_manifest_but_no_completed_result",
        }
        for request in manifest.get("requests", ())
        if str(request.get("measurement_id") or request.get("request_key")) not in measured_request_ids
        and not any(item.get("measurement_id") == request.get("measurement_id")
                    for item in unmeasured)
    ]
    coverage = _coverage(manifest, analyses, unmeasured, counts, unselected)
    coverage["split_overlaps"] = overlaps
    coverage["split_overlap_count"] = len(overlaps)
    return {
        "campaign_id": manifest.get("campaign_id"),
        "data_references": {
            "manifest": "manifest.json", "results": "results.json",
            "note": "paths are data references; no source SHA/version admission gate",
        },
        "status": (
            "dry_run_no_measured_samples" if results.get("status") == "dry_run"
            else "derived_only_no_a3_measurement"
            if counts.get("derived_excluded") and not analyses
            else "unverified_source_analysis"
            if analyses and any(group.get("source_class") == "unspecified" for group in analyses)
            else "measured_analysis" if analyses
            else "no_measured_samples"
        ),
        "fit_policy": {
            "train_only": True, "heldout_evaluation_only": True,
            "no_pooling": "groups keep condition, axis, protocol and observed API separate",
            "max_segments": max_segments, "min_segment_points": min_segment_points,
            "min_segment_points_effective": (
                "3 points per segment whenever n >= 2*min_segment_points; a curve with "
                "fewer points fits one segment because a 2-point segment has r2 == 1 by "
                "construction and cannot be validated"
            ),
            "segment_model": "Huber-loss affine (IRLS, MAD scale) per segment",
            "segment_count_criterion": "bic_on_robust_huber_loss; all candidate counts reported",
            "monotone_policy": (
                "a measured negative slope is reported as measured and projected onto a "
                "constant level; a published curve never decreases with the work axis"
            ),
            "outliers": "reported per segment; never silently dropped",
            "jump_threshold": jump_threshold,
            "root_owns_acceptance": True,
            "observed_context_use": "used only to keep measured backend/protocol groups separate; it never selects a train regime from heldout outcomes",
        },
        "summary": {
            "groups": len(analyses),
            "completed_results": counts["completed"],
            "pending_results": counts["pending"],
            "failed_results": counts["failed"],
            "dry_run_validated": counts["dry_run_validated"],
            "unknown_results": counts["unknown"],
            "duplicate_result_rows_ignored": counts.get("duplicate_rows", 0),
            "derived_or_unmeasured_excluded": counts.get("derived_excluded", 0),
            "source_class_counts": {
                str(cls): sum(1 for group in analyses if group.get("source_class") == cls)
                for cls in sorted({str(group.get("source_class")) for group in analyses})
            },
            "split_overlap_groups": len(overlaps),
            "independent_condition_axis_groups": len(analyses),
            "weak_fit_groups": sum(
                1 for group in analyses
                if (group.get("fit_quality") or {}).get("status") == "weak"),
            "projected_segments": sum(
                len((group.get("fit_quality") or {}).get("projected_segments") or ())
                for group in analyses),
            "outliers_reported": sum(
                len(group.get("outliers") or ()) for group in analyses),
            "raw_measured_samples": sum(
                sum(point["raw_sample_count"] for point in group["train_points"] + group["heldout_points"])
                for group in analyses
            ),
        },
        "analyses": analyses,
        "unmeasured": unmeasured + unselected,
        "coverage": coverage,
    }


def adapt_linearity_split_results(*args: Any, **kwargs: Any) -> Any:
    from .a3_operator_curves_adapter import adapt_linearity_split_results as _fn
    return _fn(*args, **kwargs)


def analyze_linearity_files(*args: Any, **kwargs: Any) -> Any:
    from .a3_operator_curves_adapter import analyze_linearity_files as _fn
    return _fn(*args, **kwargs)


def analyze_campaign_files(*args: Any, **kwargs: Any) -> Any:
    from .a3_operator_curves_adapter import analyze_campaign_files as _fn
    return _fn(*args, **kwargs)


def linearity_calibration_groups(*args: Any, **kwargs: Any) -> Any:
    from .a3_operator_curves_adapter import linearity_calibration_groups as _fn
    return _fn(*args, **kwargs)


def normalized_conditions(*args: Any, **kwargs: Any) -> Any:
    from .a3_operator_curves_adapter import normalized_conditions as _fn
    return _fn(*args, **kwargs)


def analyze_files(manifest_path: str | Path, results_path: str | Path, *,
                  output_dir: str | Path | None = None,
                  max_segments: int = DEFAULT_MAX_SEGMENTS) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    results = json.loads(Path(results_path).expanduser().read_text(encoding="utf-8"))
    analysis = analyze(manifest, results, max_segments=max_segments)
    analysis["data_references"]["manifest"] = str(manifest_path)
    analysis["data_references"]["results"] = str(results_path)
    if output_dir is not None:
        root = Path(output_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        # Write the numerical artifact first.  Markdown is a secondary view and
        # must never leave a half-written analysis on a formatting edge case.
        (root / "analysis.json").write_text(
            json.dumps(analysis, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        try:
            markdown = _markdown(analysis)
        except Exception as exc:  # noqa: BLE001 - secondary view must not erase JSON
            markdown = f"# Operator shape-linearity analysis\n\nMarkdown rendering failed: {exc}\n"
        (root / "analysis.md").write_text(markdown, encoding="utf-8")
        (root / "normalized-conditions.json").write_text(
            json.dumps(normalized_conditions(manifest, results), indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (root / "measurement-index.json").write_text(json.dumps({
            "campaign_id": manifest.get("campaign_id"),
            "manifest": str(manifest_path), "results": str(results_path),
            "analysis": "analysis.json",
            "groups": [
                {
                    "condition_key": group["condition_key"], "axis": group["axis"],
                    "protocol_signature": group["protocol_signature"],
                    "train": group["independent_train_points"],
                    "heldout": group["independent_heldout_points"],
                }
                for group in analysis["analyses"]
            ],
        }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return analysis


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze operator campaign raw samples")
    parser.add_argument("--manifest")
    parser.add_argument("--results")
    parser.add_argument("--input", help="linearity split-results JSON for --adapter linearity")
    parser.add_argument("--adapter", choices=("worker", "linearity", "campaign"), default="worker")
    parser.add_argument("--output-dir", default="examples/operator-campaign/analysis")
    parser.add_argument("--max-segments", type=int, default=DEFAULT_MAX_SEGMENTS)
    args = parser.parse_args(argv)
    if args.adapter == "linearity":
        if not args.input:
            parser.error("--adapter linearity requires --input")
        analysis = analyze_linearity_files(args.input, output_dir=args.output_dir,
                                           max_segments=args.max_segments)
    elif args.adapter == "campaign":
        if not args.manifest or not args.results:
            parser.error("--adapter campaign requires --manifest and --results")
        analysis = analyze_campaign_files(args.manifest, args.results,
                                          output_dir=args.output_dir,
                                          max_segments=args.max_segments)
    else:
        if not args.manifest or not args.results:
            parser.error("worker adapter requires --manifest and --results")
        analysis = analyze_files(args.manifest, args.results, output_dir=args.output_dir,
                                 max_segments=args.max_segments)
    sys.stdout.write(json.dumps(analysis["summary"], indent=1, sort_keys=True) + "\n")
    return 1 if analysis["coverage"]["failed_results"] else 0


__all__ = [
    "analyze", "analyze_files", "normalized_conditions",
    "adapt_linearity_split_results", "analyze_linearity_files",
    "analyze_campaign_files", "linearity_calibration_groups", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
