"""Coverage-layer tests for the A3 operator curve queries.

These tests exercise the real export/analyze/query path (no mocks) for the
behaviour the accuracy round added:

* hardware-series isolation (an A3 curve must never answer an A2 query);
* interval vs local transition resolution from measured interior points;
* robust Huber fitting with the three-point segment minimum and reported
  outliers;
* the monotone publication rule (a measured negative slope is projected, never
  published as ``ok`` with a decreasing cost);
* the guarded monotone 2-D coverage (accept, reject, no extrapolation);
* coverage accounting semantics (numeric vs production-eligible vs refused);
* leakage rule L4 (a result row cannot promote itself into training);
* the gap-fill upgrade path (interior points resolve a bracket after re-fit).
"""
from __future__ import annotations

import copy
import itertools

import pytest

from mindie_sim.a3_operator_curve_forecast import (
    export_curve_artifact, query_curve_artifact,
)
from mindie_sim.a3_operator_curve_protocol import hardware_compatible, hardware_series
from mindie_sim.a3_operator_curve_query import _predict, _predict_query
from mindie_sim.a3_operator_curves import analyze
from mindie_sim.a3_operator_curves_adapter import adapt_linearity_split_results

PROTOCOL = {
    "timer": "device_event", "warmup": 10, "iterations": 30, "trials": 1,
    "preallocated": True, "cache_policy": "same_buffers_reused",
    "timing_scope": "device_event_warm",
}


_COUNTER = itertools.count()


def _case(x, y, *, axis="m", fixed=None, scope="scope-a", family="gemm",
          device="SyntheticDevice", split="train", layout="contiguous",
          dtype="bf16", features=None, protocol=None):
    fixed = {"n": 128, "k": 128} if fixed is None else dict(fixed)
    shape = {axis: x, **fixed}
    # Unique per case: the adapter keys requests by case_id.
    request = {
        "case_id": f"{scope}-{axis}-{x}-{next(_COUNTER)}",
        "op_family": family, "shape": shape,
        "dtype": dtype, "layout": layout, "features": dict(features or {}),
        "coverage_scope": scope, "sweeps": [{"axis": axis}],
    }
    return {
        "request": request, "campaign_split": split,
        "production_result": {
            "status": "completed", "source_class": "measured", "run_id": "synthetic",
            "raw_samples_ms": [y] * 3, "duration_ms": y,
            "environment": {"device_name": device},
            "protocol": dict(protocol or PROTOCOL),
        },
    }


def _artifact(cases):
    manifest, results = adapt_linearity_split_results({"cases": list(cases)})
    return export_curve_artifact(manifest, results)


def _curve_query(curve, shape, **overrides):
    physical = curve["physical"]
    query = {
        "hardware_key": physical["hardware_key"],
        "family": physical["family"], "recipe": physical["recipe"],
        "execution_kind": physical["execution_kind"],
        "worker_recipe": physical["worker_recipe"],
        "backend": physical["backend"],
        "target_match": physical["target_match"],
        "coverage_scope": physical["coverage_scope"],
        "layout": physical["layout"], "quantization": physical["quantization"],
        "precision": physical["precision"], "features": physical["features"],
        "axis": curve["axis"], "shape": shape, "protocol": curve["protocol"],
    }
    query.update(overrides)
    return query


def _merged(*artifacts):
    curves = [curve for artifact in artifacts for curve in artifact["curves"]]
    return {"campaign_id": "merged", "curve_count": len(curves), "curves": curves,
            "source_evidence": {"heldout_values_included": False}}


# --------------------------------------------------------------------------
# Hardware identity
# --------------------------------------------------------------------------


def test_hardware_series_is_part_of_physical_identity():
    assert hardware_series("Ascend910_9362") == "a3"
    assert hardware_series("Ascend910B4") == "a2"
    assert hardware_series("Ascend910B4-1") == "a2"
    assert hardware_series("OtherDevice") is None
    assert hardware_compatible("Ascend910_9362", "Ascend910_9362")
    assert not hardware_compatible("Ascend910_9362", "Ascend910B4")
    assert not hardware_compatible("Ascend910_9362", "")


def test_a3_curve_never_answers_a2_query_and_vice_versa():
    a3 = _artifact([_case(x, 1.0 + 0.001 * x, device="Ascend910_9362")
                    for x in (8, 16, 32, 64)])
    curve = a3["curves"][0]
    assert curve["physical"]["hardware_series"] == "a3"
    good = query_curve_artifact(a3, _curve_query(curve, {"m": 24, "n": 128, "k": 128}))
    assert good["status"] == "ok"
    assert good["hardware_series"] == "a3"

    a2 = _artifact([_case(x, 1.0 + 0.001 * x, device="Ascend910B4")
                    for x in (8, 16, 32, 64)])
    a2_curve = a2["curves"][0]
    assert a2_curve["physical"]["hardware_series"] == "a2"
    # A3 query against an A2-only artifact, and A2 query against A3-only.
    assert query_curve_artifact(a3, _curve_query(a2_curve, {
        "m": 24, "n": 128, "k": 128}))["status"] == "unsupported"
    assert query_curve_artifact(a2, _curve_query(curve, {
        "m": 24, "n": 128, "k": 128}))["status"] == "unsupported"
    # A declared series that contradicts the key never wildcards.
    crossed = _curve_query(curve, {"m": 24, "n": 128, "k": 128},
                           hardware_series="a2")
    assert query_curve_artifact(a3, crossed)["status"] == "unsupported"
    # Both series together: each query still resolves to its own board only.
    both = _merged(a3, a2)
    assert query_curve_artifact(both, _curve_query(curve, {
        "m": 24, "n": 128, "k": 128}))["matched_curve"] == curve["curve_id"]
    assert query_curve_artifact(both, _curve_query(a2_curve, {
        "m": 24, "n": 128, "k": 128}))["matched_curve"] == a2_curve["curve_id"]


def test_curve_identity_changes_with_hardware_recipe_precision_layout():
    base = [_case(x, 1.0 + 0.001 * x) for x in (8, 16, 32)]
    other_recipe = [_case(x, 1.0 + 0.001 * x) for x in (8, 16, 32)]
    for case in other_recipe:
        case["request"]["recipe"] = "different_recipe"
    other_precision = [_case(x, 1.0 + 0.001 * x, dtype="fp32",
                             features={"weight_dtype": "fp32",
                                       "activation_dtype": "fp32",
                                       "compute_dtype": "fp32"})
                       for x in (8, 16, 32)]
    other_layout = [_case(x, 1.0 + 0.001 * x, layout="strided")
                    for x in (8, 16, 32)]
    artifacts = [_artifact(base), _artifact(other_recipe),
                 _artifact(other_precision), _artifact(other_layout)]
    ids = {artifact["curves"][0]["curve_id"] for artifact in artifacts}
    assert len(ids) == 4


# --------------------------------------------------------------------------
# Transition brackets
# --------------------------------------------------------------------------


def test_transition_bracket_keeps_interval_without_interior_measurements():
    artifact = _artifact([_case(x, y) for x, y in
                          ((1, 1.0), (8, 1.0), (16, 3.0),
                           (129, 5.0), (512, 10.0), (513, 10.0))])
    curve = artifact["curves"][0]
    assert curve["transition_brackets"], curve["segments"]
    result = query_curve_artifact(artifact, _curve_query(
        curve, {"m": 32, "n": 128, "k": 128}))
    assert result["status"] == "transition_unresolved"
    assert result["prediction_ms"] is None
    assert result["coverage"] == "interval_only"
    assert result["lower_bound_ms"] is not None
    assert result["upper_bound_ms"] >= result["lower_bound_ms"]
    assert result["uncertainty_ms"] == pytest.approx(
        result["upper_bound_ms"] - result["lower_bound_ms"])


def test_two_interior_points_resolve_the_bracket_locally():
    base = _artifact([_case(x, y) for x, y in
                      ((1, 1.0), (8, 1.0), (16, 3.0),
                       (129, 5.0), (512, 10.0), (513, 10.0))])
    # A gap-fill series under the same physical contract (a different declared
    # probe description) measured two points strictly inside the bracket.
    filler = _artifact([_case(x, y, scope="gapfill-series") for x, y in
                        ((32, 1.6), (64, 2.2))])
    merged = _merged(base, filler)
    curve = base["curves"][0]
    # x=100 is inside the base curve's bracket [16, 129] but outside the
    # gap-fill series' own support, so only the interior points can answer it.
    result = query_curve_artifact(merged, _curve_query(
        curve, {"m": 100, "n": 128, "k": 128}))
    assert result["matched_curve"] == curve["curve_id"]
    assert result["status"] == "transition_resolved_local"
    assert result["region"] == "transition_local"
    assert result["prediction_ms"] is not None and result["prediction_ms"] > 0
    assert 1.0 < result["prediction_ms"] < 10.0
    assert result["uncertainty_ms"] is not None
    assert result["transition_bracket_resolution"]["interior_support_count"] >= 2
    assert "gapfill-series" not in str(result["matched_curve"])
    local = result["local_fit"]
    assert local["support_count"] >= 2
    assert local["slope_ms_per_unit"] > 0
    assert local["support"] == sorted(local["support"])


def test_local_resolution_requires_two_measured_interior_points():
    base = _artifact([_case(x, y) for x, y in
                      ((1, 1.0), (8, 1.0), (16, 3.0),
                       (129, 5.0), (512, 10.0), (513, 10.0))])
    single = _artifact([_case(x, y, scope="gapfill-single") for x, y in
                        ((48, 2.0), (1024, 20.0))])
    merged = _merged(base, single)
    result = query_curve_artifact(merged, _curve_query(
        base["curves"][0], {"m": 32, "n": 128, "k": 128}))
    assert result["prediction_ms"] is None
    assert result["status"] == "transition_unresolved"


# --------------------------------------------------------------------------
# Robust fitting
# --------------------------------------------------------------------------


def _piecewise_manifest(xs, ys, *, outlier=None, split="train"):
    cases = []
    for x, y in zip(xs, ys):
        cases.append(_case(x, y, scope="fit-series", split=split))
    if outlier is not None:
        cases.append(_case(outlier[0], outlier[1], scope="fit-series", split=split))
    return adapt_linearity_split_results({"cases": cases})


def test_robust_fit_recovers_known_piecewise_line_with_one_outlier():
    xs = [1, 2, 4, 8, 16, 32, 64, 128]
    ys = [2.0 + 0.5 * x if x <= 8 else 2.0 + 4.0 + 0.02 * (x - 8) for x in xs]
    manifest, results = _piecewise_manifest(xs, ys, outlier=(100, 40.0))
    analysis = analyze(manifest, results)
    group = analysis["analyses"][0]
    fit = group["fit"]
    assert fit["selection"]["criterion"] == "bic_on_robust_huber_loss"
    assert fit["selection"]["min_segment_points_effective"] == 3
    assert all(segment["n"] >= 3 for segment in fit["segments"])
    slopes = [segment["slope_fitted_ms_per_unit"] for segment in fit["segments"]]
    assert slopes[0] == pytest.approx(0.5, abs=0.02)
    assert slopes[-1] == pytest.approx(0.02, abs=0.005)
    # The gross outlier is reported, not silently dropped, and the fit ignores it.
    reported = [item for segment in fit["segments"] for item in segment["outliers"]]
    assert [item["x"] for item in reported] == [100.0]
    assert group["independent_train_points"] == len(xs) + 1
    assert fit["fit_quality"]["outlier_count"] == 1
    assert group["fit_quality"]["outlier_count"] == 1
    # All raw samples are still present in the analysis product.
    assert sum(point["raw_sample_count"] for point in group["train_points"]) == 3 * (len(xs) + 1)
    assert all(point["trial_medians_ms"] for point in group["train_points"])


def test_three_point_segment_minimum_and_small_curve_single_segment():
    # Five points cannot support two three-point segments: one segment is fitted.
    five = _artifact([_case(x, 1.0 + 0.01 * x, scope="five")
                      for x in (1, 2, 4, 8, 16)])
    fit = five["curves"][0]["segments"]
    assert len(fit) == 1
    assert five["curves"][0]["fit_settings"]["selection"]["chosen_segments"] == 1
    # Six points with a regime change split 3+3 and every segment has 3 points.
    six = _artifact([_case(x, y, scope="six") for x, y in
                     ((1, 2.0), (2, 3.0), (4, 5.0),
                      (64, 40.0), (128, 80.0), (256, 160.0))])
    curve = six["curves"][0]
    assert len(curve["segments"]) == 2
    assert [segment["n"] for segment in curve["segments"]] == [3, 3]
    assert curve["transition_brackets"] == [[4.0, 64.0]]


def test_published_curve_never_decreases_with_work_axis():
    # A measured negative slope over a wide range (the old add/copy failure).
    xs = [5120, 6144, 8192, 16384, 65536, 262144]
    ys = [0.0345, 0.0330, 0.0310, 0.0280, 0.0250, 0.0219]
    artifact = _artifact([_case(x, y, axis="elements", fixed={}, scope="copy")
                          for x, y in zip(xs, ys)])
    curve = artifact["curves"][0]
    segments = curve["segments"]
    assert segments[0]["measured_negative_slope"] is True
    assert segments[0]["slope_fitted_ms_per_unit"] < 0
    assert segments[0]["projection"] == "constant_level"
    assert segments[0]["slope_ms_per_unit"] == 0.0
    values = []
    for x in (6000, 7000, 9000, 20000, 100000):
        query = query_curve_artifact(artifact, _curve_query(
            curve, {"elements": x}))
        assert query["prediction_ms"] is not None
        assert query["status"] != "ok"
        assert query["status"] in {"plateau_projected", "weak_fit"}
        assert query["monotone_projection"] == "constant_level"
        values.append(query["prediction_ms"])
    assert values == sorted(values)
    assert max(values) - min(values) == pytest.approx(0.0, abs=1e-12)
    assert curve["fit_quality"]["projected_segments"]


def test_monotone_projection_is_applied_across_segment_joins():
    xs = [1, 2, 4, 8, 16, 32, 64, 128]
    ys = [5.0, 4.6, 4.2, 4.0, 4.0, 4.0, 4.0, 4.0]
    artifact = _artifact([_case(x, y, scope="decreasing") for x, y in zip(xs, ys)])
    curve = artifact["curves"][0]
    published = [
        _predict({"segments": curve["segments"]}, x) for x in xs
    ]
    assert all(value is not None for value in published)
    assert published == sorted(published)
    assert curve["segments"][0]["level_ms"] == pytest.approx(published[-1], abs=1e-9)


# --------------------------------------------------------------------------
# Guarded 2-D coverage
# --------------------------------------------------------------------------


def _two_dimensional_artifact():
    # A true log-plane in (N, K) at fixed m: cost = c0 * sqrt(N/128) * sqrt(K/128),
    # so the guarded monotone interpolation has a consistent neighbourhood.
    cases = []
    for n, k in ((128, 128), (128, 512), (512, 128), (512, 512)):
        base = 0.02 * (n / 128.0) ** 0.5 * (k / 128.0) ** 0.5
        for m in (8, 16, 32, 64):
            cases.append(_case(m, base * (1.0 + 0.01 * m),
                               fixed={"n": n, "k": k}, scope="grid"))
    return _artifact(cases)


def test_two_dimensional_coverage_accepts_bracketed_query():
    artifact = _two_dimensional_artifact()
    reference = artifact["curves"][0]
    result = query_curve_artifact(artifact, _curve_query(
        reference, {"m": 32, "n": 256, "k": 256}))
    assert result["status"] == "interpolated_2d"
    assert result["coverage"] == "interpolated_2d"
    assert result["region"] == "interpolation_2d"
    assert result["prediction_ms"] is not None and result["prediction_ms"] > 0
    assert result["uncertainty_ms"] is not None and result["uncertainty_ms"] > 0
    fit = result["two_dimensional_fit"]
    assert fit["model"] == "monotone_log_bilinear"
    assert fit["monotone"] is True
    assert fit["slopes"]["n"] >= 0 and fit["slopes"]["k"] >= 0
    assert result["support"]["node_count"] >= 3
    assert result["production_eligible"] is False
    # The value is bracketed by the measured corner cost levels at m=32.
    corners = [segment["x_min"] for segment in reference["segments"]]
    assert corners


def test_two_dimensional_coverage_rejects_insufficient_or_out_of_envelope():
    artifact = _two_dimensional_artifact()
    reference = artifact["curves"][0]
    # Only two bracketing curves in K: refuses (< 3 measured curves).
    few = _artifact([_case(m, 0.02 * (1 + 0.01 * m), fixed={"n": 512, "k": 256},
                           scope="few") for m in (8, 16, 32, 64)])
    merged = _merged(artifact, few)
    no_cell = _merged(artifact, _artifact([
        _case(m, 0.02, fixed={"n": 600, "k": 600}, scope="far")
        for m in (8, 16, 32, 64)]))
    # n=600/k=600 is not bracketed above by any measured curve.
    assert query_curve_artifact(no_cell, _curve_query(
        merged["curves"][0], {"m": 32, "n": 700, "k": 700}))["status"] == "unsupported"
    # Outside the measured axis envelope: no 2-D extrapolation.
    assert query_curve_artifact(artifact, _curve_query(
        reference, {"m": 4096, "n": 256, "k": 256}))["status"] in {
            "unsupported", "extrapolation"}
    # Degenerate cell (a single measured (n, k) node) stays unsupported.
    single = _artifact([_case(m, 0.02 * (1 + 0.01 * m), scope="single")
                        for m in (8, 16, 32, 64)])
    assert query_curve_artifact(single, _curve_query(
        single["curves"][0], {"m": 32, "n": 256, "k": 256}))["status"] == "unsupported"


def test_two_dimensional_coverage_rejects_disagreeing_neighbourhood():
    # Neighbours that disagree by ~100 % cannot support a point estimate: the
    # uncertainty band would exceed the value, so the query is refused.
    cases = []
    for n, k, base in ((128, 128, 0.01), (128, 512, 0.02),
                       (512, 128, 0.40), (512, 512, 0.05)):
        for m in (8, 16, 32, 64):
            cases.append(_case(m, base * (1.0 + 0.01 * m),
                               fixed={"n": n, "k": k}, scope="disagree"))
    artifact = _artifact(cases)
    reference = artifact["curves"][0]
    result = query_curve_artifact(artifact, _curve_query(
        reference, {"m": 32, "n": 256, "k": 256}))
    assert result["status"] == "unsupported"


def test_two_dimensional_coverage_rejects_non_monotone_support():
    cases = []
    # Cost must increase with N and K; this cell decreases in N at fixed K.
    for n, k, base in ((128, 128, 0.10), (128, 512, 0.12),
                       (512, 128, 0.02), (512, 512, 0.15)):
        for m in (8, 16, 32, 64):
            cases.append(_case(m, base * (1.0 + 0.01 * m),
                               fixed={"n": n, "k": k}, scope="non-monotone"))
    artifact = _artifact(cases)
    reference = artifact["curves"][0]
    result = query_curve_artifact(artifact, _curve_query(
        reference, {"m": 32, "n": 256, "k": 256}))
    assert result["status"] == "unsupported"


# --------------------------------------------------------------------------
# Coverage accounting
# --------------------------------------------------------------------------


def _coverage(artifact, queries):
    counts = {"numeric": 0, "production_eligible": 0, "unsupported": 0, "statuses": {}}
    for query in queries:
        result = query_curve_artifact(artifact, query)
        status = str(result.get("status"))
        counts["statuses"][status] = counts["statuses"].get(status, 0) + 1
        if result.get("prediction_ms") is not None:
            counts["numeric"] += 1
            if status in {"ok", "extrapolation"}:
                counts["production_eligible"] += 1
        else:
            counts["unsupported"] += 1
    counts["conditions"] = len(queries)
    counts["coverage"] = counts["numeric"] / len(queries)
    return counts


def test_coverage_accounting_separates_numeric_from_production_eligible():
    artifact = _artifact([_case(x, y) for x, y in
                          ((1, 1.0), (8, 1.0), (16, 3.0),
                           (129, 5.0), (512, 10.0), (513, 10.0))])
    curve = artifact["curves"][0]
    queries = [
        _curve_query(curve, {"m": 4, "n": 128, "k": 128}),      # interpolation
        _curve_query(curve, {"m": 32, "n": 128, "k": 128}),     # transition
        _curve_query(curve, {"m": 1024, "n": 128, "k": 128}),   # extrapolation
    ]
    counts = _coverage(artifact, queries)
    assert counts["conditions"] == 3
    assert counts["numeric"] == 2
    assert counts["production_eligible"] == 2
    assert counts["unsupported"] == 1
    assert counts["statuses"]["transition_unresolved"] == 1
    assert counts["coverage"] == pytest.approx(2 / 3)

    # A weak fit is numeric coverage but never production-eligible.
    weak = _artifact([_case(x, y, axis="elements", fixed={}, scope="weak")
                      for x, y in ((5120, 0.020), (16384, 0.019),
                                   (65536, 0.041), (262144, 0.018))])
    weak_curve = weak["curves"][0]
    weak_counts = _coverage(weak, [_curve_query(weak_curve, {"elements": 32768})])
    assert weak_counts["numeric"] == 1
    assert weak_counts["production_eligible"] == 0
    assert weak_counts["statuses"]["weak_fit"] == 1


# --------------------------------------------------------------------------
# Fit quality / leakage / gap-fill upgrade
# --------------------------------------------------------------------------


def test_heldout_result_cannot_promote_itself_into_training():
    cases = [_case(x, 1.0 + 0.01 * x, scope="split") for x in (8, 16, 32, 64)]
    cases.append(_case(1024, 999.0, scope="split", split="heldout"))
    manifest, results = adapt_linearity_split_results({"cases": cases})
    # A heldout row that claims ``train`` must not enter the fit.
    for row in results["results"]:
        if row["measurement_id"].endswith("1024"):
            row["split"] = "train"
    artifact = export_curve_artifact(manifest, results)
    curve = artifact["curves"][0]
    assert 1024.0 not in curve["train_support"]["x_values"]
    assert artifact["split_excluded_results"]
    assert all(item["reason"] == "frozen_split_not_train"
               for item in artifact["split_excluded_results"])


def test_gapfill_interior_points_upgrade_after_refit_without_code_change():
    before = _artifact([_case(x, y) for x, y in
                        ((1, 1.0), (8, 1.0), (16, 3.0),
                         (129, 5.0), (512, 10.0), (513, 10.0))])
    curve = before["curves"][0]
    shape = {"m": 48, "n": 128, "k": 128}
    assert query_curve_artifact(before, _curve_query(
        curve, shape))["status"] == "transition_unresolved"
    # New A3 gap-fill measurements: 32/64/96 strictly inside the [16, 129] gap,
    # consistent with the bracket (3.0 at x=16 to 5.0 at x=129).
    gapfill = [_case(x, 3.0 + (5.0 - 3.0) * (x - 16) / (129 - 16),
                     scope="gap-fill-20261021")
               for x in (32, 64, 96)]
    after = _artifact([_case(x, y) for x, y in
                       ((1, 1.0), (8, 1.0), (16, 3.0),
                        (129, 5.0), (512, 10.0), (513, 10.0))] + gapfill)
    after_curve = after["curves"][0]
    result = query_curve_artifact(after, _curve_query(after_curve, shape))
    assert result["prediction_ms"] is not None
    assert result["status"] in {"ok", "transition_resolved_local"}
    assert result["matched_curve"] == after_curve["curve_id"]
    assert 3.0 <= result["prediction_ms"] <= 5.0


def test_query_policy_reports_outliers_and_never_drops_points():
    xs = [1, 2, 4, 8, 16, 32]
    ys = [1.0 + 0.1 * x for x in xs]
    manifest, results = _piecewise_manifest(xs, ys, outlier=(12, 99.0))
    analysis = analyze(manifest, results)
    group = analysis["analyses"][0]
    assert group["fit_quality"]["outlier_count"] == 1
    assert [item["x"] for item in group["outliers"]] == [12.0]
    assert group["fit_selection"]["chosen_segments"] >= 1
    assert group["train_residuals_ms"]["max_abs"] > 0
    # The point is still present as a support point and in the raw trials.
    assert 12.0 in [point["x"] for point in group["train_points"]]
    assert analysis["summary"]["outliers_reported"] == 1
    # The published value at the outlier is not driven by it.
    fit = group["fit"]
    assert _predict(fit, 12.0) < 10.0
    assert _predict_query(fit, 12.0)["prediction_ms"] is not None


# --------------------------------------------------------------------------
# Round-2: protocol series separation and declaration/attestation identity
# --------------------------------------------------------------------------

BATCHED_PROTOCOL = {
    "timer": "device_event", "warmup": 10, "iterations": 30, "trials": 2,
    "preallocated": True, "cache_policy": "same_buffers_reused",
    "timing_scope": "device_event_warm_operator_batched_amortized",
    "pre_soak_scope": "continuous_same_kernel_before_each_trial_not_counted",
}


def _batched_case(x, y, *, scope="batched-series", device="Ascend910_9362",
                  fixed=None, axis="m", dtype="bf16", features=None):
    return _case(x, y, axis=axis, fixed=fixed, scope=scope, device=device,
                 dtype=dtype, features=features, protocol=BATCHED_PROTOCOL)


def test_protocol_series_never_answer_each_other():
    per_call = _artifact([_case(x, 1.0 + 0.01 * x, device="Ascend910_9362")
                          for x in (8, 16, 32, 64)])
    batched = _artifact([_batched_case(x, 2.0 + 0.01 * x)
                         for x in (8, 16, 32, 64)])
    merged = _merged(per_call, batched)
    assert len({curve["protocol"]["timing_scope"] for curve in merged["curves"]}) == 2
    per_call_curve, batched_curve = per_call["curves"][0], batched["curves"][0]
    # Each series answers its own protocol and only its own.
    a = query_curve_artifact(merged, _curve_query(
        per_call_curve, {"m": 24, "n": 128, "k": 128}))
    b = query_curve_artifact(merged, _curve_query(
        batched_curve, {"m": 24, "n": 128, "k": 128}))
    assert a["matched_curve"] == per_call_curve["curve_id"]
    assert b["matched_curve"] == batched_curve["curve_id"]
    cross = _curve_query(per_call_curve, {"m": 24, "n": 128, "k": 128},
                         protocol=dict(BATCHED_PROTOCOL))
    assert query_curve_artifact({"curves": [per_call_curve]}, cross)["status"] == "unsupported"
    back = _curve_query(batched_curve, {"m": 24, "n": 128, "k": 128},
                        protocol=dict(PROTOCOL))
    assert query_curve_artifact({"curves": [batched_curve]}, back)["status"] == "unsupported"


def test_attested_precision_accepts_unattested_and_refuses_contradiction():
    # The measured worker attests only input/compute/accumulation (a fused op
    # does not record weight/output formats); the declaration states more.
    artifact = _artifact([_case(x, 1.0 + 0.01 * x, scope="attestation")
                          for x in (8, 16, 32)])
    curve = artifact["curves"][0]
    base = _curve_query(curve, {"m": 24, "n": 128, "k": 128})
    richer = copy.deepcopy(base)
    richer["precision"] = {**base["precision"], "input_dtype": "bf16",
                           "output_dtype": "bf16", "weight_dtype": "bf16",
                           "compute_dtype": "bf16", "quantization": "none"}
    accepted = query_curve_artifact(artifact, richer)
    assert accepted["status"] == "ok"
    attested = accepted["physical_attestation"]["precision"]
    assert "output_dtype" in attested["unattested_by_curve"]
    # A fact the curve *does* attest and the query contradicts refuses.
    contradicted = copy.deepcopy(base)
    contradicted["precision"] = {**base["precision"], "input_dtype": "fp32"}
    assert query_curve_artifact(artifact, contradicted)["status"] == "unsupported"
    # A quantized query is not answered by an unquantized curve.
    quantized = copy.deepcopy(base)
    quantized["quantization"] = "fp8_e4m3"
    assert query_curve_artifact(artifact, quantized)["status"] == "unsupported"


def test_declared_and_attested_recipe_label_is_reported_not_gated():
    artifact = _artifact([_case(x, 1.0 + 0.01 * x, scope="recipe-label")
                          for x in (8, 16, 32)])
    curve = artifact["curves"][0]
    curve["physical"]["recipe"] = "catalog_campaign_label"
    query = _curve_query(curve, {"m": 24, "n": 128, "k": 128})
    query["recipe"] = "deployed_worker_label"
    result = query_curve_artifact(artifact, query)
    assert result["status"] == "ok"
    assert result["match_scope"] == "physical_cell_scope_agnostic"
    assert result["physical_attestation"]["recipe_measured"] == "catalog_campaign_label"
    assert result["physical_attestation"]["recipe_declared"] == "deployed_worker_label"
    # If only one side names a recipe the executed primitive cannot be verified.
    silent = copy.deepcopy(query)
    silent["recipe"] = None
    assert query_curve_artifact(artifact, silent)["status"] == "unsupported"


def test_stored_protocol_is_renormalized_for_older_artifacts():
    # An artifact written before the canonical filter drops batch-size
    # attestation still compares on the declared contract.
    artifact = _artifact([_case(x, 1.0 + 0.01 * x, scope="renormalize")
                          for x in (8, 16, 32)])
    curve = artifact["curves"][0]
    curve["protocol"] = {**curve["protocol"],
                         "repeats_per_sample": 57,
                         "warmup_events_total": 20,
                         "clock_gate": {"target_mhz": 1650}}
    query = _curve_query(curve, {"m": 24, "n": 128, "k": 128})
    query["protocol"].pop("repeats_per_sample", None)
    result = query_curve_artifact(artifact, query)
    assert result["status"] == "ok"
    assert result["matched_curve"] == curve["curve_id"]
