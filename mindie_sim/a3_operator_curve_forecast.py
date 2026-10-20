"""Public train-only curve artifact export and pre-use physical query API.

[Role]
  Export the existing product piecewise curve fits as a serializable,
  train-only artifact and answer pre-use physical-condition queries from that
  artifact.  This is the operational bridge between campaign measurements and a
  future public cost/predict consumer; it does not read or fit heldout data.

[Invariants]
  * Artifact export filters to precommitted train requests/results and never
    copies heldout observations into parameters, support, lineage or summaries.
  * Matching requires complete hardware, precision, layout, quantization,
    feature, recipe/execution, target-match and protocol identity.  Unknown or
    incompatible fields never wildcard.
  * `target_match=false` curves can answer physical candidates but never an
    exact production match.  Hardware-only transfer requires explicit caller
    authorization and remains labelled non-production.
  * Transition brackets are unresolved support; no point prediction is invented
    and no negative time is returned or clamped into false confidence.
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
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .a3_operator_curve_axis import (
    axis_value as _axis_value, fixed_shape as _axis_fixed_shape,
    query_shape as _query_shape,
)
from .a3_operator_curve_protocol import canonical_protocol
from .a3_operator_curve_query import _predict_query
from .a3_operator_curves import _canonical, _protocol_signature, analyze
from .a3_operator_curves_adapter import (
    adapt_campaign_manifest_results, adapt_linearity_split_results,
)

_PROVENANCE_FEATURE_KEYS = {"origin", "source_query_context"}
def _train_only(manifest: Mapping[str, Any], results: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    split_by_id = {
        str(item.get("measurement_id") or item.get("request_key")): str(item.get("split", "train"))
        for item in manifest.get("requests", ())
    }
    requests = [dict(item) for item in manifest.get("requests", ())
                if str(item.get("split", "train")) == "train"]
    result_rows = []
    for item in results.get("results", ()):
        midpoint = str(item.get("measurement_id") or item.get("request_key"))
        split = str(item.get("split") or split_by_id.get(midpoint, "train"))
        if split == "train":
            result_rows.append(dict(item))
    return (
        {**dict(manifest), "requests": requests},
        {**dict(results), "results": result_rows},
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
    return index


def _select_request(requests: Sequence[Mapping[str, Any]],
                    hardware_key: str, protocol_key: Mapping[str, Any]) -> Mapping[str, Any]:
    for request in requests:
        request_hardware = str(request.get("hardware_key") or "")
        request_protocol = _protocol_key(request.get("protocol") or {})
        if request_hardware == hardware_key and _canonical(request_protocol) == _canonical(protocol_key):
            return request
    for request in requests:
        if str(request.get("hardware_key") or "") == hardware_key:
            return request
    return requests[0]


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
        })
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
                          max_segments: int = 3, min_segment_points: int = 2,
                          jump_threshold: float = 0.15) -> dict[str, Any]:
    """Build a serializable train-only curve artifact from campaign products."""
    train_manifest, train_results = _train_only(manifest, results)
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
            "quantization": request.get("quantization") or condition.get("quantization")
            or (condition.get("precision") or {}).get("quantization"),
            "features": _physical_features(condition),
            "hardware_key": hardware_key,
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
        curve_identity = _canonical({
            "condition_key": condition_key, "axis": axis, "hardware_key": hardware_key,
            "protocol_key": protocol_key, "physical": physical,
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
            "fit_settings": {"kind": fit.get("kind"), "bic": fit.get("bic"),
                             "train_points": fit.get("train_points")},
            "train_support": {
                "x_min": min(train_x) if train_x else None,
                "x_max": max(train_x) if train_x else None,
                "x_values": train_x,
                "point_count": len(group.get("train_points", ())),
                "independent_trial_count": int(group.get("independent_train_trials") or 0),
            },
            "segments": segments,
            "transition_brackets": fit.get("transition_brackets") or [],
            "uncertainty": {
                "residual_median_abs_ms": _median_or_none(residuals),
                "residual_max_abs_ms": max(residuals) if residuals else None,
            },
            "train_points": [
                {"x": point.get("x"), "median_ms": point.get("median_ms"),
                 "mad_ms": point.get("mad_ms"), "trial_count": point.get("trial_count"),
                 "raw_sample_count": point.get("raw_sample_count")}
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
            "transition_brackets": "unresolved; no point prediction",
            "extrapolation": "nearest segment, positive result only, labelled extrapolation",
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


def _matches(curve: Mapping[str, Any], query: Mapping[str, Any]) -> bool:
    physical = curve.get("physical") or {}
    curve_hardware = str(curve.get("physical", {}).get("hardware_key") or "")
    query_hardware = str(query.get("hardware_key") or "")
    if not curve_hardware or not query_hardware:
        return False
    if curve_hardware != query_hardware:
        return False
    for key in ("family", "recipe", "execution_kind", "worker_recipe", "backend",
                "target_match", "coverage_scope", "layout", "quantization"):
        if physical.get(key) != query.get(key):
            return False
    if not _dict_strict_equal(physical.get("precision") or {}, query.get("precision") or {}):
        return False
    if not _dict_strict_equal(physical.get("features") or {}, query.get("features") or {}):
        return False
    if _canonical(curve.get("protocol") or {}) != _canonical(_protocol_key(query.get("protocol") or {})):
        return False
    shape = query.get("shape") or {}
    axis = str(query.get("axis") or curve.get("axis"))
    if axis != str(curve.get("axis")):
        return False
    fixed, value, reason = _query_shape(curve, shape)
    if reason != "ok" or fixed is None or value is None:
        return False
    return True


def _query_result(curve: Mapping[str, Any], query: Mapping[str, Any],
                  match_scope: str, *, transfer_authorized: bool = False) -> dict[str, Any]:
    shape = query.get("shape") or {}
    axis = str(curve.get("axis"))
    value = _axis_value(axis, shape)
    if value is None:
        return {"status": "unsupported", "prediction_ms": None,
                "reason": f"unsupported_or_invalid_axis:{axis}",
                "match_scope": match_scope, "production_match": False}
    x = float(value)
    prediction = _predict_query({"segments": curve.get("segments") or []}, x)
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
        "matched_curve": curve.get("curve_id"),
        "condition_key": curve.get("condition_key"),
        "axis": axis,
        "x": x,
        "match_scope": match_scope,
        "production_match": production_match,
        "target_match": target_match,
        "source_class": physical.get("source_class"),
        "protocol": curve.get("protocol"),
        "request_protocol": curve.get("request_protocol"),
        "hardware_key": physical.get("hardware_key"),
        "requested_hardware_key": query.get("hardware_key"),
        "support": prediction.get("support"),
        "lower_bound_ms": prediction.get("lower_bound_ms"),
        "upper_bound_ms": prediction.get("upper_bound_ms"),
        "uncertainty_ms": prediction.get("uncertainty_ms"),
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
    if len(exact) == 1:
        curve = exact[0]
        target_match = (curve.get("physical") or {}).get("target_match")
        scope = "exact_production" if target_match is True else "exact_physical_candidate"
        return _query_result(curve, query, scope)
    if len(exact) > 1:
        return {"status": "ambiguous",
                "prediction_ms": None, "matches": [curve.get("curve_id") for curve in exact],
                "reason": "multiple_curves_match_physical_query"}
    return {"status": "unsupported",
            "prediction_ms": None, "reason": "no_compatible_curve",
            "match_scope": "none", "production_match": False}


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
