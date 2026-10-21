"""Public train-only curve artifact export and pre-use physical query API.

[Role]
  Export the existing product piecewise curve fits as a serializable,
  train-only artifact and answer pre-use physical-condition queries from that
  artifact.  This is the operational bridge between campaign measurements and a
  future public cost/predict consumer; it does not read or fit heldout data.

[Invariants]
  * Artifact export filters to precommitted train requests/results and never
    copies heldout observations into parameters, support, lineage or summaries.
    The manifest's frozen split is authoritative; a result row may confirm it
    but can never override it (a heldout row claiming ``train`` is excluded and
    reported, not fitted).
  * Matching requires complete hardware (key *and* series), precision, layout,
    quantization, feature, recipe/execution, target-match and protocol identity.
    Unknown or incompatible fields never wildcard.
  * Curve identity hashes hardware key/series, family, recipe, precision,
    layout, quantization, physical features, protocol signature and axis, so a
    curve can never be confused with a different physical contract.
  * `target_match=false` curves can answer physical candidates but never an
    exact production match.  Hardware-only transfer requires explicit caller
    authorization and remains labelled non-production.
  * Transition brackets are unresolved support; no point prediction is invented
    and no negative time is returned or clamped into false confidence.  A
    bracket resolves locally only from >= 2 measured points strictly inside it
    (own curve or an identical physical contract's curve), and says so with
    ``transition_resolved_local`` and a derived uncertainty.
  * Multiple exact matches are resolved deterministically only when every
    candidate shares the same physical contract (they differ in the free-text
    probe description and/or the sampled axis points); the resolution rule, the
    candidates and their spread are reported.  Physically different candidates
    stay `ambiguous`.
  * `coverage_scope` is a declared-probe description and `recipe` is the
    campaign's label for the executed primitive, not physical facts: an exact
    identity match always wins, and a query that declares a different label is
    answered only from the physically identical fixed-shape cell with
    ``match_scope="physical_cell_scope_agnostic"`` and both labels reported.
    The executed primitive (worker recipe, backend, execution kind), precision,
    layout, quantization, features, protocol and hardware still gate, as does a
    precision fact the curve attests and the query contradicts.  A query fact
    the curve does not attest is reported as unattested, never assumed.
  * Primitive, fused and reference-decomposition curves are separate.

[Timeouts & Locks]
  Pure JSON/curve arithmetic bounded by artifact size; no device/network/lock.

[Side-Effect Policy]
  In-memory APIs write nothing.  The optional CLI writes only explicit paths.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .a3_operator_curve_axis import (
    axis_value as _axis_value, fixed_shape as _axis_fixed_shape,
    query_shape as _query_shape,
)
from .a3_operator_curve_protocol import (
    canonical_protocol, hardware_compatible, hardware_series, protocol_signature,
)
from .a3_operator_curve_query import (
    MAX_2D_RELATIVE_UNCERTAINTY, MAX_2D_RMS_LOG_RESIDUAL, MIN_2D_NODES,
    _predict_query, monotone_log_plane, predict_2d,
)
from .a3_operator_curves import _canonical, _protocol_signature, analyze
from .a3_operator_curves_adapter import (
    adapt_campaign_manifest_results, adapt_linearity_split_results,
)

_PROVENANCE_FEATURE_KEYS = {"origin", "source_query_context"}
# Descriptions that identify *how a probe series was declared*, not the physics
# it measured.  Two curves that differ only here measure the same physical
# contract; exact matching still requires equality, but *support* may be pooled
# for interior-bracket resolution and guarded 2-D coverage.
_SUPPORT_NEUTRAL_KEYS = {"coverage_scope", "target_match"}
_NUMERIC_STATUSES = {"ok", "transition_resolved_local", "plateau_projected",
                     "weak_fit", "extrapolation", "interpolated_2d"}


def _train_only(manifest: Mapping[str, Any], results: Mapping[str, Any]) -> tuple[
    dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:
    """Filter to the precommitted train split; the manifest split is authoritative.

    Leakage rule L4: a result row can never promote itself into training.  Rows
    whose declared split disagrees with the frozen manifest split, or that carry
    no train declaration at all, are excluded and reported.
    """
    split_by_id = {
        str(item.get("measurement_id") or item.get("request_key")): str(item.get("split", ""))
        for item in manifest.get("requests", ())
    }
    requests = [dict(item) for item in manifest.get("requests", ())
                if str(item.get("split", "")) == "train"]
    result_rows = []
    excluded: list[dict[str, Any]] = []
    for item in results.get("results", ()):
        midpoint = str(item.get("measurement_id") or item.get("request_key"))
        frozen_split = split_by_id.get(midpoint)
        declared = str(item.get("split") or "")
        if frozen_split is None:
            excluded.append({"measurement_id": midpoint,
                             "reason": "result_without_frozen_request"})
            continue
        if frozen_split != "train":
            excluded.append({"measurement_id": midpoint, "reason": "frozen_split_not_train",
                             "frozen_split": frozen_split})
            continue
        if declared and declared != frozen_split:
            excluded.append({"measurement_id": midpoint,
                             "reason": "result_split_conflicts_with_frozen_split",
                             "frozen_split": frozen_split, "declared_split": declared})
            continue
        result_rows.append(dict(item))
    return (
        {**dict(manifest), "requests": requests},
        {**dict(results), "results": result_rows},
        excluded,
    )


def _physical_features(condition: Mapping[str, Any]) -> dict[str, Any]:
    features = condition.get("features") or {}
    return {str(key): value for key, value in features.items()
            if key not in _PROVENANCE_FEATURE_KEYS}


def _protocol_key(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Single canonical pre-use protocol view (shared helper)."""
    return canonical_protocol(protocol)


def _request_index(manifest: Mapping[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for request in manifest.get("requests", ()):
        point = request.get("point") or {}
        key = (str(request.get("condition_key")), str(point.get("axis") or "elements"))
        index.setdefault(key, []).append(dict(request))
    for key in index:
        index[key].sort(key=lambda item: str(item.get("measurement_id")
                                             or item.get("request_key") or ""))
    return index


def _group_result_candidates(manifest: Mapping[str, Any],
                             results: Mapping[str, Any]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    request_by_id = {
        str(item.get("measurement_id") or item.get("request_key")): dict(item)
        for item in manifest.get("requests", ())
    }
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for result in results.get("results", ()):
        midpoint = str(result.get("measurement_id") or result.get("request_key"))
        request = request_by_id.get(midpoint)
        if not request:
            continue
        axis = str((request.get("point") or {}).get("axis") or "elements")
        candidates.setdefault((str(request.get("condition_key")), axis), []).append({
            "request": request, "result": dict(result),
            "signature": _protocol_signature(result.get("protocol") or {},
                                             result.get("observed_context") or {}),
            "order": (midpoint, str(result.get("run_id") or "")),
        })
    for key in candidates:
        # Deterministic ordering: never depend on result-file row order.
        candidates[key].sort(key=lambda item: item["order"])
    return candidates


def _segment_residuals(segments: Sequence[Mapping[str, Any]],
                       train_points: Sequence[Mapping[str, Any]]) -> list[float]:
    residuals: list[float] = []
    for segment in segments:
        start, end = float(segment["x_min"]), float(segment["x_max"])
        segment_points = [point for point in train_points
                          if start <= float(point["x"]) <= end]
        for point in segment_points:
            predicted = (float(segment["intercept_ms"])
                         + float(segment["slope_ms_per_unit"]) * float(point["x"]))
            residuals.append(abs(predicted - float(point["median_ms"])))
    return residuals


def export_curve_artifact(manifest: Mapping[str, Any], results: Mapping[str, Any], *,
                          max_segments: int = 3, min_segment_points: int = 3,
                          jump_threshold: float = 0.15) -> dict[str, Any]:
    """Build a serializable train-only curve artifact from campaign products."""
    train_manifest, train_results, split_excluded = _train_only(manifest, results)
    analysis = analyze(
        train_manifest, train_results, max_segments=max_segments,
        min_segment_points=min_segment_points, jump_threshold=jump_threshold,
    )
    request_index = _request_index(train_manifest)
    result_candidates = _group_result_candidates(train_manifest, train_results)
    curves = []
    excluded = []
    for group in analysis.get("analyses", ()):
        fit = group.get("fit")
        condition_key = str(group["condition_key"])
        axis = str(group["axis"])
        if not fit:
            excluded.append({"condition_key": condition_key, "axis": axis,
                             "reason": group.get("status") or "no_fitted_curve"})
            continue
        candidates = request_index.get((condition_key, axis), [])
        if not candidates:
            excluded.append({"condition_key": condition_key, "axis": axis,
                             "reason": "no_train_request_for_group"})
            continue
        hardware_key = str(group.get("hardware_key") or candidates[0].get("hardware_key") or "")
        group_signature = str(group.get("protocol_signature") or "")
        matched = [
            item for item in result_candidates.get((condition_key, axis), ())
            if str(item["request"].get("hardware_key") or "") == hardware_key
            and str(item.get("signature")) == group_signature
        ]
        if not matched:
            excluded.append({"condition_key": condition_key, "axis": axis,
                             "reason": "no_result_matched_group_protocol_signature"})
            continue
        request = matched[0]["request"]
        matched_result = matched[0].get("result") or {}
        protocol_key = _protocol_key(
            matched_result.get("protocol") or request.get("protocol") or {})
        condition = request.get("condition") or {}
        point = request.get("point") or {}
        shape = dict(point.get("shape") or {})
        fixed_shape = _axis_fixed_shape(axis, shape)
        physical = {
            "family": request.get("op_family") or condition.get("family"),
            "recipe": request.get("recipe") or condition.get("recipe"),
            "execution_kind": request.get("execution_kind") or condition.get("execution_kind"),
            "worker_recipe": request.get("worker_recipe") or condition.get("worker_recipe"),
            "backend": request.get("backend") or condition.get("backend"),
            "target_match": (request.get("target_match")
                             if "target_match" in request else condition.get("target_match")),
            "coverage_scope": request.get("coverage_scope"),
            "precision": dict(condition.get("precision") or {}),
            "layout": request.get("layout") or condition.get("layout"),
            "quantization": (request.get("quantization")
                             or (request.get("condition") or {}).get("quantization")
                             or condition.get("quantization")
                             or (condition.get("precision") or {}).get("quantization")),
            "features": _physical_features(condition),
            "hardware_key": hardware_key,
            "hardware_series": hardware_series(hardware_key),
            "source_class": group.get("source_class") or request.get("source_class"),
        }
        segments = []
        for segment in fit["segments"]:
            item = dict(segment)
            start, end = float(item["x_min"]), float(item["x_max"])
            segment_points = [point for point in group["train_points"]
                              if start <= float(point["x"]) <= end]
            residuals = [
                abs((float(item["intercept_ms"]) + float(item["slope_ms_per_unit"]) * float(p["x"]))
                    - float(p["median_ms"]))
                for p in segment_points
            ]
            item["residual_median_abs_ms"] = _median_or_none(residuals)
            item["residual_max_abs_ms"] = max(residuals) if residuals else None
            item["support_point_count"] = len(segment_points)
            segments.append(item)
        residuals = _segment_residuals(fit["segments"], group["train_points"])
        train_x = sorted(float(point["x"]) for point in group["train_points"])
        # Identity is the complete physical contract: hardware key *and* series,
        # family, recipe, precision, layout, quantization, features, axis and the
        # canonical protocol signature.  Two curves that differ in any of them are
        # different curves even if they share a condition key.
        curve_identity = _canonical({
            "condition_key": condition_key, "axis": axis,
            "hardware_key": hardware_key,
            "hardware_series": physical["hardware_series"],
            "family": physical["family"], "recipe": physical["recipe"],
            "execution_kind": physical["execution_kind"],
            "worker_recipe": physical["worker_recipe"],
            "backend": physical["backend"], "target_match": physical["target_match"],
            "precision": physical["precision"], "layout": physical["layout"],
            "quantization": physical["quantization"], "features": physical["features"],
            "protocol_key": protocol_key,
            "protocol_signature": protocol_signature(
                matched_result.get("protocol") or request.get("protocol") or {},
                matched_result.get("observed_context") or {}),
            "analysis_protocol_signature": group_signature,
        })
        curves.append({
            "curve_id": "curve:" + hashlib.sha1(curve_identity.encode("utf-8")).hexdigest()[:20],
            "condition_key": condition_key,
            "axis": axis,
            "fixed_shape": fixed_shape,
            "condition_shape": group.get("condition_shape"),
            "physical": physical,
            "protocol": protocol_key,
            "request_protocol": _protocol_key(request.get("protocol") or {}),
            "protocol_source": "matched_result_group",
            "analysis_protocol_signature": group.get("protocol_signature"),
            "fit_settings": {
                "kind": fit.get("kind"), "bic": fit.get("bic"),
                "train_points": fit.get("train_points"),
                "selection": fit.get("selection"),
                "fit_quality": fit.get("fit_quality"),
                "robust": fit.get("robust"),
            },
            "train_support": {
                "x_min": min(train_x) if train_x else None,
                "x_max": max(train_x) if train_x else None,
                "x_values": train_x,
                "point_count": len(group.get("train_points", ())),
                "independent_trial_count": int(group.get("independent_train_trials") or 0),
            },
            "segments": segments,
            "transition_brackets": fit.get("transition_brackets") or [],
            "bracket_support": fit.get("bracket_support") or {},
            "fit_quality": fit.get("fit_quality"),
            "uncertainty": {
                "residual_median_abs_ms": _median_or_none(residuals),
                "residual_max_abs_ms": max(residuals) if residuals else None,
            },
            "train_points": [
                {"x": point.get("x"), "median_ms": point.get("median_ms"),
                 "mad_ms": point.get("mad_ms"), "trial_count": point.get("trial_count"),
                 "raw_sample_count": point.get("raw_sample_count"),
                 "trial_medians_ms": list(point.get("trial_medians_ms") or ()),
                 "raw_samples_ms": list(point.get("raw_samples_ms") or ())}
                for point in group.get("train_points", ())
            ],
            "lineage": {
                "source_case_count": sum(len(item.get("source_cases") or ())
                                         for item in candidates),
                "models": sorted({source.get("model") for item in candidates
                                  for source in (item.get("source_cases") or ())
                                  if source.get("model")}),
                "condition_keys": sorted({str(item.get("condition_key"))
                                          for item in candidates}),
            },
        })
    artifact = {
        "campaign_id": manifest.get("campaign_id"),
        "fit_policy": {
            "train_only": True, "max_segments": max_segments,
            "min_segment_points": min_segment_points,
            "split_authority": "frozen manifest split; a result row cannot promote itself into train",
            "segment_model": "Huber-loss affine (IRLS, MAD scale)",
            "segment_count_criterion": "bic_on_robust_huber_loss",
            "monotone_policy": (
                "measured negative slope is reported, then projected to a constant "
                "level; a published curve never decreases with the work axis"),
            "transition_brackets": (
                "unresolved; resolved locally only from >=2 measured points strictly "
                "inside the bracket"),
            "extrapolation": "nearest segment, positive result only, labelled extrapolation",
            "two_dimensional_coverage": (
                ">=3 measured curves bracketing in both fixed dimensions, monotone "
                "log-plane, no extrapolation; labelled interpolated_2d"),
        },
        "source_evidence": {
            "source_classes": analysis.get("summary", {}).get("source_class_counts", {}),
            "run_ids": sorted({str(item.get("run_id")) for item in train_results.get("results", ())
                               if item.get("run_id")}),
            "hardware_keys": sorted({str(item.get("hardware_key")) for item in train_manifest.get("requests", ())
                                     if item.get("hardware_key")}),
            "heldout_values_included": False,
        },
        "curve_count": len(curves),
        "curves": curves,
        "excluded": excluded,
        "split_excluded_results": split_excluded,
    }
    return artifact


def _median_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return float(ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0)


def _dict_strict_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    keys = {key for key, value in left.items() if value is not None}
    keys |= {key for key, value in right.items() if value is not None}
    return all(left.get(key) == right.get(key) for key in keys)


def _quantization_key(value: Any) -> str:
    """Normalize a quantization declaration; absence means "none recorded"."""
    text = str(value or "").strip().lower()
    return "none" if text in {"", "none", "unquantized", "fp32", "bf16"} else text


def _quantization_compatible(curve_value: Any, query_value: Any) -> bool:
    """A curve that attests no quantization is compatible with a "none" query.

    Absence is "no quantization recorded here", which agrees with a query that
    claims none and contradicts a query that claims fp8/int8/... (a quantized
    query must be answered by a curve that attests the same quantization).
    """
    curve = _quantization_key(curve_value)
    query = _quantization_key(query_value)
    if curve == query:
        return True
    return curve == "none" and query == "none"


_FORMAT_ALIASES = {
    "activation_format": "input_dtype",
    "weight_format": "weight_dtype",
    "compute_format": "compute_dtype",
}


def _without_format_aliases(values: Mapping[str, Any]) -> dict[str, Any]:
    """Drop format fields that only repeat the corresponding dtype spelling.

    ``activation_format``/``weight_format``/``compute_format`` are dtype
    spellings of the sibling ``*_dtype`` fields.  When they repeat it they add no
    fact, and a declaration that omits them is not making a weaker claim.  When
    they differ (for example a fp8 weight with an e4m3 block format) both stay
    and must match.
    """
    clean = dict(values or {})
    for format_key, dtype_key in _FORMAT_ALIASES.items():
        if format_key in clean and clean.get(dtype_key) is not None \
                and clean[format_key] == clean[dtype_key]:
            clean.pop(format_key)
    return clean


def _dict_attested(curve_values: Mapping[str, Any],
                   query_values: Mapping[str, Any]) -> tuple[bool, dict[str, list[str]]]:
    """Attestation-aware comparison of a partially attested physical dict.

    A measured curve attests the facts its worker recorded; a fresh declaration
    may state more (for example an accumulation dtype the worker never wrote
    down) or fewer.  The comparison refuses a contradiction and refuses a curve
    fact the query does not declare (that would be an unverified wildcard), but a
    query fact the curve does not attest is allowed and reported, never hidden.
    """
    curve = _without_format_aliases(curve_values)
    query = _without_format_aliases(query_values)
    keys = set(curve) | set(query)
    mismatched = sorted(
        key for key in keys
        if curve.get(key) is not None and query.get(key) is not None
        and curve[key] != query[key])
    undeclared_by_query = sorted(
        key for key in keys
        if curve.get(key) is not None and query.get(key) is None)
    unattested_by_curve = sorted(
        key for key in keys
        if query.get(key) is not None and curve.get(key) is None)
    return (not mismatched and not undeclared_by_query), {
        "mismatched": mismatched,
        "undeclared_by_query": undeclared_by_query,
        "unattested_by_curve": unattested_by_curve,
    }


def _matches(curve: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    physical = curve.get("physical") or {}
    curve_hardware = str(physical.get("hardware_key") or "")
    query_hardware = str(query.get("hardware_key") or "")
    # Hardware identity is key *and* series: an A3 curve can never answer an A2
    # query, and a look-alike key cannot wildcard another board.
    if not hardware_compatible(curve_hardware, query_hardware):
        return False
    query_series = query.get("hardware_series")
    if query_series is not None:
        curve_series = (physical.get("hardware_series")
                        or hardware_series(curve_hardware))
        if str(query_series) != str(curve_series):
            return False
    for key in ("family", "recipe", "execution_kind", "worker_recipe", "backend",
                "target_match", "coverage_scope", "layout", "quantization"):
        if physical.get(key) != query.get(key):
            return False
    if not _dict_strict_equal(physical.get("precision") or {}, query.get("precision") or {}):
        return False
    if not _dict_strict_equal(physical.get("features") or {}, query.get("features") or {}):
        return False
    # The stored protocol is a canonical view baked at export time; re-normalize
    # it so an artifact written under an earlier canonicalization still compares
    # on the same declared contract (filtering is idempotent).
    if (_canonical(_protocol_key(curve.get("protocol") or {}))
            != _canonical(_protocol_key(query.get("protocol") or {}))):
        return False
    shape = query.get("shape") or {}
    axis = str(query.get("axis") or curve.get("axis"))
    if axis != str(curve.get("axis")):
        return False
    fixed, value, reason = _query_shape(curve, shape)
    if reason != "ok" or fixed is None or value is None:
        return False
    return True


def _support_compatible(curve: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    """Same physical contract, ignoring the free-text probe description.

    Exact matching stays strict about ``coverage_scope``/``target_match``.  This
    weaker relation is only used to *share measured support*: interior-bracket
    points and the guarded 2-D neighbourhood of curves that measure the same
    hardware, family, recipe, precision, layout, quantization, features and
    protocol.  Nothing is transferred across a different physical contract.
    """
    physical = curve.get("physical") or {}
    if not hardware_compatible(physical.get("hardware_key"), query.get("hardware_key")):
        return False
    # A declared series that contradicts the key still refuses: it is physical
    # identity, not a label, and it must be enforced on every match tier.
    query_series = query.get("hardware_series")
    if query_series is not None:
        curve_series = (physical.get("hardware_series")
                        or hardware_series(physical.get("hardware_key")))
        if str(query_series) != str(curve_series):
            return False
    # The *executed* primitive identifies the physical contract: worker recipe,
    # backend and execution kind.  ``recipe`` is the campaign's declaration label
    # for that primitive.  Two declared labels may differ for the same executed
    # API (a catalog recipe name vs the deployed worker's name) and are reported;
    # if only one side names a recipe the executed primitive cannot be verified,
    # so the query refuses instead of wildcarding.
    for key in ("family", "execution_kind", "worker_recipe", "backend"):
        if physical.get(key) != query.get(key):
            return False
    if physical.get("layout") != query.get("layout"):
        return False
    if not _quantization_compatible(physical.get("quantization"),
                                    query.get("quantization")):
        return False
    curve_recipe = physical.get("recipe")
    query_recipe = query.get("recipe")
    if (curve_recipe in (None, "")) != (query_recipe in (None, "")):
        return False
    # Precision may be partially attested by the worker (a fused op may not
    # record an output/weight dtype); features are behavioral (mask, epsilon,
    # rotary mode) and stay strict in both directions.
    compatible, _ = _dict_attested(physical.get("precision") or {},
                                   query.get("precision") or {})
    if not compatible:
        return False
    if not _dict_strict_equal(physical.get("features") or {}, query.get("features") or {}):
        return False
    return (_canonical(_protocol_key(curve.get("protocol") or {}))
            == _canonical(_protocol_key(query.get("protocol") or {})))


def _interior_support(curve: Mapping[str, Any], compatible: Sequence[Mapping[str, Any]],
                      low: float, high: float) -> list[list[float]]:
    """Measured points strictly inside a bracket, from the same physical contract."""
    points: list[list[float]] = []
    own = curve.get("bracket_support") or {}
    for values in own.values() if isinstance(own, Mapping) else ():
        for value in values or ():
            try:
                x, y = float(value[0]), float(value[1])
            except (TypeError, ValueError, IndexError):
                continue
            if low < x < high:
                points.append([x, y])
    fixed = _canonical(dict(curve.get("fixed_shape") or {}))
    for other in compatible:
        if str(other.get("axis")) != str(curve.get("axis")):
            continue
        if _canonical(dict(other.get("fixed_shape") or {})) != fixed:
            continue
        for point in other.get("train_points") or ():
            try:
                x, y = float(point["x"]), float(point["median_ms"])
            except (KeyError, TypeError, ValueError):
                continue
            if low < x < high:
                points.append([x, y])
    unique = {(round(x, 12), round(y, 12)): [x, y] for x, y in points}
    return [unique[key] for key in sorted(unique)]


def _query_result(curve: Mapping[str, Any], query: Mapping[str, Any],
                  match_scope: str, *, transfer_authorized: bool = False,
                  compatible: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    shape = query.get("shape") or {}
    axis = str(curve.get("axis"))
    value = _axis_value(axis, shape)
    if value is None:
        return {"status": "unsupported", "prediction_ms": None,
                "reason": f"unsupported_or_invalid_axis:{axis}",
                "match_scope": match_scope, "production_match": False}
    x = float(value)
    base_fit = {
        "segments": curve.get("segments") or [],
        "fit_quality": curve.get("fit_quality"),
        "bracket_support": curve.get("bracket_support") or {},
    }
    prediction = _predict_query(base_fit, x)
    resolved = None
    if str(prediction.get("region")) == "transition_bracket":
        low, high = (float(part) for part in prediction.get("support") or (0.0, 0.0))
        interior = _interior_support(curve, compatible, low, high)
        if len(interior) >= 2:
            augmented = {**base_fit,
                         "bracket_support": {f"{low},{high}": interior}}
            candidate = _predict_query(augmented, x)
            if str(candidate.get("status")) == "transition_resolved_local":
                resolved = {
                    "interior_support_count": len(interior),
                    "interior_support_from_curves": sorted({
                        str(item.get("curve_id")) for item in compatible
                        if str(item.get("axis")) == str(curve.get("axis"))
                        and _canonical(dict(item.get("fixed_shape") or {}))
                        == _canonical(dict(curve.get("fixed_shape") or {}))
                    }),
                }
                prediction = candidate
    physical = curve.get("physical") or {}
    target_match = physical.get("target_match")
    production_match = bool(
        match_scope == "exact_production" and target_match is True
        and prediction.get("region") in {"exact", "interpolation"}
        and str(physical.get("source_class") or "") in {"measured", "observed"}
    )
    status = prediction.get("status")
    if prediction.get("prediction_ms") is None:
        status = "transition_unresolved" if status == "unresolved_transition" else "missing_prediction"
    return {
        "status": status,
        "prediction_ms": prediction.get("prediction_ms"),
        "region": prediction.get("region"),
        "coverage": prediction.get("coverage") or (
            "none" if prediction.get("prediction_ms") is None else "interpolated_1d"),
        "matched_curve": curve.get("curve_id"),
        "condition_key": curve.get("condition_key"),
        "axis": axis,
        "x": x,
        "match_scope": match_scope,
        "production_match": production_match,
        "production_eligible": bool(status in {"ok", "extrapolation"}),
        "target_match": target_match,
        "source_class": physical.get("source_class"),
        "protocol": curve.get("protocol"),
        "request_protocol": curve.get("request_protocol"),
        "hardware_key": physical.get("hardware_key"),
        "hardware_series": (physical.get("hardware_series")
                            or hardware_series(physical.get("hardware_key"))),
        "requested_hardware_key": query.get("hardware_key"),
        "support": prediction.get("support"),
        "lower_bound_ms": prediction.get("lower_bound_ms"),
        "upper_bound_ms": prediction.get("upper_bound_ms"),
        "uncertainty_ms": prediction.get("uncertainty_ms"),
        "monotone_projection": prediction.get("monotone_projection"),
        "monotone_source": prediction.get("monotone_source"),
        "monotone_projection_ms": (
            (curve.get("segments") or [{}])[prediction["segment_index"]].get(
                "monotone_projection_ms")
            if prediction.get("segment_index") is not None
            and prediction["segment_index"] < len(curve.get("segments") or ()) else None),
        "fit_quality": curve.get("fit_quality"),
        "physical_attestation": {
            "recipe_declared": query.get("recipe"),
            "recipe_measured": physical.get("recipe"),
            "precision": _dict_attested(physical.get("precision") or {},
                                        query.get("precision") or {})[1],
            "features": _dict_attested(physical.get("features") or {},
                                       query.get("features") or {})[1],
        },
        "transition_bracket_resolution": resolved,
        "local_fit": prediction.get("local_fit"),
        "train_support": curve.get("train_support"),
        "transition_brackets": curve.get("transition_brackets"),
        "training_uncertainty": curve.get("uncertainty"),
        "matched_segment": (
            curve.get("segments", [])[prediction["segment_index"]]
            if prediction.get("segment_index") is not None
            and prediction["segment_index"] < len(curve.get("segments") or ()) else None
        ),
        "transfer_authorized": bool(transfer_authorized),
        "reason": prediction.get("reason"),
    }


def load_curve_artifact(path: str | Path) -> dict[str, Any]:
    artifact = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(artifact, Mapping):
        raise ValueError("curve_artifact_invalid")
    curves = artifact.get("curves")
    if not isinstance(curves, list):
        raise ValueError("curve_artifact_missing_curves")
    for curve in curves:
        if not isinstance(curve, Mapping):
            raise ValueError("curve_artifact_invalid_curve")
        for key in ("condition_key", "axis", "fixed_shape", "physical", "segments"):
            if key not in curve:
                raise ValueError(f"curve_artifact_missing:{key}")
    return dict(artifact)


def query_curve_artifact(artifact: Mapping[str, Any], query: Mapping[str, Any], *,
                         allow_primitive_transfer: bool = False) -> dict[str, Any]:
    """Answer a pre-use physical query; no observed result is accepted."""
    if not isinstance(query, Mapping):
        return {"status": "unsupported",
                "prediction_ms": None, "reason": "query_invalid"}
    observed = {key for key in ("actual_median_ms", "duration_ms", "raw_samples_ms",
                                "predicted_ms", "results", "trials") if key in query}
    if observed:
        return {"status": "unsupported",
                "prediction_ms": None,
                "reason": "query_contains_observed_outcome:" + ",".join(sorted(observed))}
    if not str(query.get("hardware_key") or ""):
        return {"status": "unsupported",
                "prediction_ms": None, "reason": "hardware_unknown",
                "match_scope": "none", "production_match": False}
    if str(query.get("axis") or "") == "dtype_probe":
        return {"status": "unsupported", "prediction_ms": None,
                "reason": "non_numeric_dtype_probe_insufficient_curve",
                "match_scope": "none", "production_match": False}
    for key in ("hardware_key", "family", "recipe", "execution_kind", "worker_recipe",
                "backend", "target_match", "layout", "quantization", "precision",
                "features", "axis", "shape", "protocol"):
        if key not in query:
            return {"status": "unsupported",
                    "prediction_ms": None, "reason": f"query_missing:{key}"}
    curves = list(artifact.get("curves") or ())
    exact = [curve for curve in curves if _matches(curve, query)]
    compatible = [curve for curve in curves if _support_compatible(curve, query)]
    if exact:
        # A physical condition can be covered by more than one declared probe
        # series.  Resolution runs over the whole *cell* (same axis, same fixed
        # shape, same target scope) so every series' own query selects the same
        # curve deterministically instead of each series answering with itself
        # and the caller declaring ambiguity.
        cell = _cell_candidates(exact[0], compatible)
        if len(exact) > 1 or len(cell) > 1:
            resolved = _resolve_compatible_matches(cell, query, compatible)
            if resolved is not None:
                return resolved
        if len(exact) > 1:
            return {"status": "ambiguous",
                    "prediction_ms": None,
                    "matches": [curve.get("curve_id") for curve in exact],
                    "reason": "multiple_curves_match_physical_query"}
        curve = exact[0]
        target_match = (curve.get("physical") or {}).get("target_match")
        scope = "exact_production" if target_match is True else "exact_physical_candidate"
        return _query_result(curve, query, scope, compatible=compatible)
    # The query's fixed-shape cell can exist under a different *declared probe
    # description*.  ``coverage_scope`` is a free-text statement of how a probe
    # series was declared, not a physical fact: a fresh heldout condition can
    # only carry a heldout label, never a training label.  Exact identity still
    # wins; when it does not exist the cell is answered from the physically
    # identical series and the scope mismatch is reported, never hidden.
    cell = _scope_agnostic_cell(query, compatible)
    if cell:
        resolved = _resolve_compatible_matches(cell, query, compatible)
        if resolved is not None:
            resolved["match_scope"] = "physical_cell_scope_agnostic"
            resolved["declared_coverage_scope"] = query.get("coverage_scope")
            resolved["measured_coverage_scopes"] = sorted({
                str((curve.get("physical") or {}).get("coverage_scope"))
                for curve in cell})
            resolved["scope_agnostic"] = True
            resolved["production_match"] = False
            return resolved
    two_dimensional = _query_two_dimensional(compatible, query)
    if two_dimensional is not None:
        return two_dimensional
    return {"status": "unsupported",
            "prediction_ms": None, "reason": "no_compatible_curve",
            "match_scope": "none", "production_match": False,
            "coverage": "none",
            "diagnostics": _rejection_diagnostics(curves, query, compatible)}


def _scope_agnostic_cell(query: Mapping[str, Any],
                         compatible: Sequence[Mapping[str, Any]],
                         ) -> list[Mapping[str, Any]]:
    """Curves of the physically identical fixed-shape cell, any scope label."""
    axis = str(query.get("axis") or "")
    shape = dict(query.get("shape") or {})
    if not axis or axis not in shape:
        return []
    fixed = _canonical({key: value for key, value in shape.items() if key != axis})
    cell = []
    for curve in compatible:
        if str(curve.get("axis")) != axis:
            continue
        if _canonical(dict(curve.get("fixed_shape") or {})) != fixed:
            continue
        # ``target_match`` is a production-match flag (a measurement declares
        # False, a fresh condition declares "exact_primitive"); it is not a
        # physical fact, and a scope-agnostic answer is never a production match.
        cell.append(curve)
    return cell


_RESOLUTION_RULE = (
    "numeric prediction before transition bracket, then more support points in the "
    "enclosing segment, then more independent trials, then more train points, then "
    "the lexicographically smallest curve id; candidates must share the full "
    "physical contract and differ only in the free-text probe description"
)


def _cell_candidates(
    reference: Mapping[str, Any], compatible: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Curves of the same physical contract in the same fixed-shape cell."""
    axis = str(reference.get("axis"))
    fixed = _canonical(dict(reference.get("fixed_shape") or {}))
    target = (reference.get("physical") or {}).get("target_match")
    cell = []
    for curve in compatible:
        if str(curve.get("axis")) != axis:
            continue
        if _canonical(dict(curve.get("fixed_shape") or {})) != fixed:
            continue
        if (curve.get("physical") or {}).get("target_match") != target:
            continue
        cell.append(curve)
    return cell


def _resolve_compatible_matches(
    exact: Sequence[Mapping[str, Any]], query: Mapping[str, Any],
    compatible: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Deterministically pick one of several *physically identical* curves.

    Two probe series can cover the same physical condition with different axis
    points; refusing every such query as ``ambiguous`` discarded measured
    evidence.  Resolution is allowed only when all candidates share the same
    physical contract (hardware/family/recipe/precision/layout/quantization/
    features/protocol/axis/fixed shape) and is reported with the candidates and
    their spread.  Physically different candidates remain ambiguous.
    """
    if not all(_support_compatible(curve, query) for curve in exact):
        return None
    fixed = {_canonical(dict(curve.get("fixed_shape") or {})) for curve in exact}
    axes = {str(curve.get("axis")) for curve in exact}
    if len(fixed) != 1 or len(axes) != 1:
        return None
    results = []
    for curve in exact:
        target_match = (curve.get("physical") or {}).get("target_match")
        scope = "exact_production" if target_match is True else "exact_physical_candidate"
        results.append(_query_result(curve, query, scope, compatible=compatible))
    def _rank(item: Mapping[str, Any]) -> tuple[int, int, int, int, int, str]:
        region = str(item.get("region"))
        if item.get("prediction_ms") is None:
            resolution = 2  # honest interval
        elif region in {"exact", "interpolation", "transition_local",
                        "transition_collapsed"}:
            resolution = 0  # measured/interpolated support at this point
        elif region == "extrapolation":
            resolution = 3  # labelled extrapolation is the weakest numeric answer
        else:
            resolution = 2
        return (
            resolution,
            0 if item.get("target_match") == target_match else 1,
            -int((item.get("matched_segment") or {}).get("n") or 0),
            -int((item.get("train_support") or {}).get("point_count") or 0),
            -int((item.get("train_support") or {}).get("independent_trial_count") or 0),
            str(item.get("matched_curve") or ""),
        )

    target_match = query.get("target_match")
    chosen = sorted(results, key=_rank)[0]
    values = {str(item.get("matched_curve")): item.get("prediction_ms")
              for item in results if item.get("prediction_ms") is not None}
    spread = (max(values.values()) - min(values.values())) if len(values) > 1 else 0.0
    chosen = dict(chosen)
    chosen["curve_selection"] = {
        "rule": _RESOLUTION_RULE,
        "candidates": [str(item.get("matched_curve")) for item in results],
        "candidate_values_ms": values,
        "candidate_statuses": {str(item.get("matched_curve")): item.get("status")
                              for item in results},
        "spread_ms": spread,
        "compatible_physical_identity": True,
    }
    if chosen.get("prediction_ms") is not None and spread > 0:
        chosen["uncertainty_ms"] = max(float(chosen.get("uncertainty_ms") or 0.0), spread)
    return chosen


_TWO_DIMENSIONAL_NEIGHBOURHOODS = (6, 8, 12)


def _query_two_dimensional(
    compatible: Sequence[Mapping[str, Any]], query: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Guarded monotone 2-D coverage from bracketing measured curves.

    The query's exact (fixed-dimension) curve does not exist.  Take the nearest
    measured curves of the same physical contract, require that the set brackets
    the query in *both* fixed dimensions, require the query axis value to be
    interior to every participating curve (no axis extrapolation), fit the
    monotone log-bilinear plane and refuse if it is non-monotone or its residual
    is not bounded.  Anything else stays unsupported.
    """
    shape = query.get("shape") or {}
    axis = str(query.get("axis") or "")
    if not axis or axis not in shape:
        return None
    try:
        x = float(shape[axis])
    except (TypeError, ValueError):
        return None
    nodes = []
    for curve in compatible:
        if str(curve.get("axis")) != axis:
            continue
        fixed = dict(curve.get("fixed_shape") or {})
        if len(fixed) != 2 or axis in fixed:
            continue
        keys = sorted(fixed)
        try:
            query_u, query_v = float(shape[keys[0]]), float(shape[keys[1]])
            node_u, node_v = float(fixed[keys[0]]), float(fixed[keys[1]])
        except (KeyError, TypeError, ValueError):
            continue
        if min(query_u, query_v, node_u, node_v) <= 0:
            continue
        prediction = _predict_query({"segments": curve.get("segments") or []}, x)
        if (prediction.get("prediction_ms") is None
                or str(prediction.get("region")) not in {"exact", "interpolation"}):
            continue
        nodes.append({
            "u": node_u, "v": node_v, "cost_ms": float(prediction["prediction_ms"]),
            "residual_scale_ms": float(prediction.get("uncertainty_ms") or 0.0),
            "source": curve.get("curve_id"), "axis_key": keys,
            "coverage_1d": prediction.get("coverage"),
        })
    if len(nodes) < MIN_2D_NODES:
        return None
    query_u, query_v = float(shape[nodes[0]["axis_key"][0]]), float(shape[nodes[0]["axis_key"][1]])
    ranked = sorted(
        nodes,
        key=lambda node: (abs(math.log(node["u"] / query_u)) + abs(math.log(node["v"] / query_v)),
                          str(node["source"])),
    )
    refused = None
    # Neighbourhood size: the smallest bracketing set whose plane is monotone
    # with a bounded residual.  Widening is allowed for *fit-quality* failures
    # (a wider baseline can stabilise a monotone surface) but never for an
    # uncertainty failure: evaluating a larger neighbourhood only because the
    # first answer had a wide band would be answer-driven selection.
    selected = None
    plane = None
    rms = None
    for size in _TWO_DIMENSIONAL_NEIGHBOURHOODS:
        candidate = ranked[:size]
        if len(candidate) < MIN_2D_NODES:
            continue
        us = [node["u"] for node in candidate]
        vs = [node["v"] for node in candidate]
        if not (min(us) <= query_u <= max(us) and min(vs) <= query_v <= max(vs)):
            refused = "query_not_bracketed_in_both_fixed_dimensions"
            continue
        candidate_plane = monotone_log_plane(candidate)
        if candidate_plane is None:
            refused = "two_dimensional_support_not_identifiable"
            continue
        if not candidate_plane.get("monotone"):
            refused = "non_monotone_two_dimensional_support"
            continue
        candidate_rms = float(candidate_plane.get("rms_log_residual") or 0.0)
        if candidate_rms > MAX_2D_RMS_LOG_RESIDUAL:
            refused = "two_dimensional_residual_exceeds_bound"
            continue
        selected, plane, rms = candidate, candidate_plane, candidate_rms
        break
    if selected is None:
        return _two_dimensional_refusal(
            refused or "two_dimensional_support_not_identifiable")
    us = [node["u"] for node in selected]
    vs = [node["v"] for node in selected]
    predicted = predict_2d(plane, query_u, query_v)
    if predicted is None:
        return _two_dimensional_refusal("nonphysical_two_dimensional_prediction")
    # Leave-one-out error of the plane on its own measured nodes: the honest
    # interpolation error, unlike the neighbourhood's vertical spread, which is
    # mostly the surface's natural gradient across the cell.
    loo = []
    if len(selected) > MIN_2D_NODES:
        for held_out in selected:
            subset = [node for node in selected if node is not held_out]
            sub_plane = monotone_log_plane(subset)
            if sub_plane is None or not sub_plane.get("monotone"):
                continue
            sub_prediction = predict_2d(sub_plane, held_out["u"], held_out["v"])
            if sub_prediction is None or sub_prediction["prediction_ms"] <= 0:
                continue
            loo.append(abs(math.log(sub_prediction["prediction_ms"]
                                    / float(held_out["cost_ms"]))))
    loo_rms = math.sqrt(sum(value * value for value in loo) / len(loo)) if loo else rms
    relative = max(rms, loo_rms, 0.02)
    uncertainty = float(predicted["prediction_ms"]) * relative
    if relative > MAX_2D_RELATIVE_UNCERTAINTY:
        return _two_dimensional_refusal(
            "two_dimensional_uncertainty_exceeds_bound")
    return {
        "status": "interpolated_2d",
        "prediction_ms": float(predicted["prediction_ms"]),
        "region": "interpolation_2d",
        "coverage": "interpolated_2d",
        "axis": axis,
        "x": x,
        "fixed_coordinates": {selected[0]["axis_key"][0]: query_u,
                              selected[0]["axis_key"][1]: query_v},
        "support": {"bracket_n": [min(us), max(us)],
                    "bracket_k": [min(vs), max(vs)],
                    "node_count": len(selected)},
        "support_nodes": plane.get("nodes"),
        "two_dimensional_fit": {
            "model": "monotone_log_bilinear",
            "slopes": plane.get("slopes"),
            "rms_log_residual": rms,
            "loo_rms_log_error": loo_rms,
            "loo_holdouts": len(loo),
            "monotone": True,
            "neighbourhood_size": len(selected),
            "neighbourhood_ladder": list(_TWO_DIMENSIONAL_NEIGHBOURHOODS),
            "neighbourhood_spread_ms": float(predicted.get("displacement_ms") or 0.0),
        },
        "uncertainty_ms": uncertainty,
        "relative_uncertainty": relative,
        "match_scope": "interpolated_2d",
        "production_match": False,
        "production_eligible": False,
        "reason": ("no exact (N,K) curve; monotone 2-D interpolation inside the "
                   "measured envelope"),
        "transfer_authorized": False,
    }


def _two_dimensional_refusal(reason: str) -> dict[str, Any]:
    return {"status": "unsupported", "prediction_ms": None,
            "reason": reason, "match_scope": "none",
            "production_match": False, "coverage": "none"}




def _rejection_diagnostics(
    curves: Sequence[Mapping[str, Any]], query: Mapping[str, Any],
    compatible: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Why a query got no curve: counts, not a guess."""
    shape = query.get("shape") or {}
    axis = str(query.get("axis") or "")
    same_family = [curve for curve in curves
                   if (curve.get("physical") or {}).get("family") == query.get("family")]
    reasons: dict[str, int] = {}
    for curve in curves:
        physical = curve.get("physical") or {}
        if not hardware_compatible(physical.get("hardware_key"), query.get("hardware_key")):
            reasons["hardware_mismatch"] = reasons.get("hardware_mismatch", 0) + 1
        elif _canonical(curve.get("protocol") or {}) != _canonical(
                _protocol_key(query.get("protocol") or {})):
            reasons["protocol_mismatch"] = reasons.get("protocol_mismatch", 0) + 1
        elif _canonical(dict(curve.get("fixed_shape") or {})) != _canonical(
                {key: int(value) for key, value in shape.items() if key != axis}):
            reasons["fixed_shape_not_measured"] = reasons.get("fixed_shape_not_measured", 0) + 1
        elif str(curve.get("axis")) != axis:
            reasons["axis_mismatch"] = reasons.get("axis_mismatch", 0) + 1
        else:
            reasons["other_physical_mismatch"] = reasons.get("other_physical_mismatch", 0) + 1
    return {
        "curves_in_artifact": len(curves),
        "same_family_curves": len(same_family),
        "support_compatible_curves": len(compatible),
        "rejection_counts": reasons,
        "note": "counts are diagnostics only; no curve is chosen from them",
    }


def export_from_paths(manifest_path: str | Path, results_path: str | Path, *,
                      adapter: str = "worker", output: str | Path | None = None,
                      max_segments: int = 3) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
    results = json.loads(Path(results_path).expanduser().read_text(encoding="utf-8"))
    if adapter == "linearity":
        manifest, results = adapt_linearity_split_results(manifest)
    elif adapter == "campaign":
        manifest, results = adapt_campaign_manifest_results(manifest, results)
    artifact = export_curve_artifact(manifest, results, max_segments=max_segments)
    if output is not None:
        Path(output).expanduser().write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n",
                                             encoding="utf-8")
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train-only curve artifact export/query")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("--manifest")
    export.add_argument("--results")
    export.add_argument("--input", help="single linearity split-results JSON")
    export.add_argument("--adapter", choices=("worker", "linearity", "campaign"), default="worker")
    export.add_argument("--output", required=True)
    export.add_argument("--max-segments", type=int, default=3)
    query = sub.add_parser("query")
    query.add_argument("--artifact", required=True)
    query.add_argument("--query", required=True)
    query.add_argument("--output", default=None)
    query.add_argument("--allow-primitive-transfer", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "export":
        if args.adapter == "linearity":
            if not args.input:
                parser.error("--adapter linearity requires --input")
            artifact = export_from_paths(args.input, args.input, adapter="linearity",
                                         output=args.output, max_segments=args.max_segments)
        else:
            if not args.manifest or not args.results:
                parser.error("export requires --manifest and --results")
            artifact = export_from_paths(args.manifest, args.results, adapter=args.adapter,
                                         output=args.output, max_segments=args.max_segments)
        sys.stdout.write(json.dumps({"status": "written", "curves": artifact["curve_count"],
                                     "output": args.output}, indent=1) + "\n")
        return 0
    artifact = load_curve_artifact(args.artifact)
    query_payload = json.loads(Path(args.query).expanduser().read_text(encoding="utf-8"))
    result = query_curve_artifact(artifact, query_payload,
                                  allow_primitive_transfer=args.allow_primitive_transfer)
    if args.output:
        Path(args.output).expanduser().write_text(
            json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(result, indent=1, sort_keys=True) + "\n")
    return 0


__all__ = [
    "export_curve_artifact", "load_curve_artifact", "query_curve_artifact",
    "export_from_paths", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
