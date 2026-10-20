"""Import/adapt measured linearity split files into campaign artifacts.

[Role]
  Convert the public production split-results cases[].production_result format
  into the campaign manifest/results/analysis/calibration views.  Also export
  the independent reviewer's pre-use normalized condition rows.

[Invariants]
  * The measured-median ``analytic_ms`` legacy alias is never used as an
  analytic baseline.
  * Thirty event repeats are one independent measurement, not 30 trials.
  * Query context is pre-use only; observed kernel context is excluded from the
    normalized reviewer input.
  * Split labels are preserved; no heldout value chooses a fit.

[Timeouts & Locks]
  Pure local JSON/arithmetic; no device/lock/sleep.

[Side-Effect Policy]
  Functions return in-memory records; the CLI caller writes only its explicit
  output directory.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .a3_operator_curve_axis import (
    fixed_shape as _axis_fixed_shape, normalize_shape as _normalize_series_shape,
    validate_series as _validate_series,
)
from .a3_operator_curve_protocol import protocol_key, source_derived_protocol
from .a3_operator_curves import (
    DEFAULT_MAX_SEGMENTS, _canonical, _median, _protocol_signature,
    analyze_files, load_manifest,
)

def adapt_linearity_split_results(payload: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt a measured linearity split file (train or heldout) to this schema.

    The adapter is intentionally split-agnostic: callers pass either split, and
    the split label stays in the rows.  It never fills an analytic baseline from
    the measured median and treats the 30 event repeats as one independent
    measurement (repeats are not independent trials).
    """
    requests, results = [], []
    for index, case in enumerate(payload.get("cases", ())):
        request = case.get("request") or {}
        production = case.get("production_result") or {}
        shape = {str(k): int(v) for k, v in (request.get("shape") or {}).items()}
        sweeps = request.get("sweeps") or ()
        raw_axis = str((sweeps[0] or {}).get("axis")) if sweeps else (
            next(iter(shape)) if len(shape) == 1 else "elements")
        axis = raw_axis.lower() if raw_axis.lower() in shape else raw_axis
        value = int(shape.get(axis, request.get("axis_value", 1)))
        fixed = {key: val for key, val in shape.items() if key != axis}
        features = dict(request.get("features") or {})
        family = str(request.get("op_family") or "unknown")
        dtype = str(request.get("dtype") or "")
        layout = str(request.get("layout") or "")
        coverage_scope = str(request.get("coverage_scope") or (
            "diagnostic_resource_probe" if family in {"add", "copy"}
            else "native_operator_semantic"))
        identity = _canonical({"family": family, "dtype": dtype, "layout": layout,
                               "coverage_scope": coverage_scope,
                               "features": features, "fixed": fixed, "axis": axis})
        condition_key = "lin:" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]
        split = str(case.get("campaign_split") or production.get("split") or "train")
        midpoint = f"lin:{request.get('case_id') or index}"
        protocol = source_derived_protocol(production, request)
        query_context = {
            "family": family, "phase": request.get("phase", ""),
            "recipe": request.get("recipe", ""), "execution_kind": request.get("execution_kind", ""),
            "worker_recipe": request.get("worker_recipe", ""),
            "requested_backend": request.get("backend", ""),
            "dtype": dtype, "weight_dtype": features.get("weight_dtype"),
            "output_dtype": features.get("activation_dtype"),
            "compute_dtype": features.get("compute_dtype"),
            "accumulation_dtype": features.get("accumulation_dtype"),
            "layout": layout, "quantization": features.get("quantization"),
            "parallel": request.get("parallel"),
            "protocol": protocol,
        }
        run_id = str(production.get("run_id") or request.get("run_id") or "")
        environment = dict(production.get("environment") or {})
        hardware_key = str(request.get("hardware_key") or environment.get("device_name") or "")
        query_context["hardware_key"] = hardware_key or None
        requests.append({
            "measurement_id": midpoint, "request_key": midpoint, "run_id": run_id,
            "hardware_key": hardware_key,
            "source_class": production.get("source_class"),
            "source": production.get("source"),
            "environment": environment,
            "timing_scope": production.get("timing_scope"),
            "condition_key": condition_key, "op_family": family,
            "coverage_scope": coverage_scope,
            "native_precision_claim": family not in {"add", "copy"} and dtype == "bf16",
            "phase": request.get("phase", ""), "recipe": request.get("recipe", ""),
            "split": split, "execution_kind": request.get("execution_kind"),
            "worker_recipe": request.get("worker_recipe"),
            "backend": request.get("backend"),
            "dtype": dtype, "layout": layout, "features": features,
            "condition": {"shape": fixed, "layout": layout,
                          "coverage_scope": coverage_scope,
                          "quantization": features.get("quantization"),
                          "precision": {
                "input_dtype": dtype, "weight_dtype": features.get("weight_dtype"),
                "output_dtype": features.get("activation_dtype"),
                "compute_dtype": features.get("compute_dtype"),
                "accumulation_dtype": features.get("accumulation_dtype"),
                "quantization": features.get("quantization"),
                "weight_format": features.get("weight_dtype"),
                "activation_format": features.get("activation_dtype"),
                "compute_format": features.get("compute_dtype"),
            }},
            "point": {"axis": axis, "value": value, "shape": shape},
            "query_context": query_context, "protocol": protocol,
            "analytic_ms": None, "work": None,
            "source_cases": [],
        })
        raw = [float(v) for v in (production.get("raw_samples_ms") or ())]
        duration = production.get("duration_ms")
        if duration is None and raw:
            duration = _median(raw)
        results.append({
            "measurement_id": midpoint, "request_key": midpoint,
            "condition_key": condition_key, "status": production.get("status", "unknown"),
            "duration_ms": duration, "raw_samples_ms": raw, "analytic_ms": None,
            "split": split, "execution_kind": request.get("execution_kind"),
            "worker_recipe": request.get("worker_recipe"), "backend": request.get("backend"),
            "run_id": run_id, "hardware_key": hardware_key,
            "source_class": production.get("source_class"),
            "source": production.get("source"),
            "not_a3_measured": bool(
                (production.get("environment") or {}).get("not_a3_measured")
                or str(production.get("source_class") or "") not in {"measured", "observed"}),
            "revision": production.get("revision", ""),
            "timing_scope": production.get("timing_scope"),
            "environment": dict(production.get("environment") or {}),
            "protocol": protocol,
            "observed_context": {},
            "correctness": {"status": "not_applicable",
                            "reason": "split file does not carry a correctness record"},
            "trials": ([{"trial_index": 0, "median_ms": duration, "raw_samples_ms": raw}]
                       if raw else []),
        })
    manifest = {
        "campaign_id": str(payload.get("schema", "imported-linearity")),
        "status": "imported",
        "requests": requests,
        "inventory": [],
        "models": {},
        "coverage": {"selected_train": sum(1 for r in requests if r["split"] == "train"),
                     "selected_heldout": sum(1 for r in requests if r["split"] != "train"),
                     "pending_conditions": 0, "derivation_failures": []},
        "pending_requests": [],
    }
    not_measured = any(bool(item.get("not_a3_measured")) for item in results)
    result_payload = {
        "status": str(payload.get("split", "imported")),
        "results": results,
        "not_a3_measured": not_measured,
        "source_classes": sorted({str(item.get("source_class") or "missing") for item in results}),
    }
    return manifest, result_payload


def _calibration_case(case: Mapping[str, Any], index: int) -> tuple[dict[str, Any], bool]:
    request = case.get("request") or {}
    production = case.get("production_result") or {}
    shape = {str(k): int(v) for k, v in (request.get("shape") or {}).items()}
    family = str(request.get("op_family") or "unknown")
    environment = dict(production.get("environment") or {})
    source_class = str(production.get("source_class") or request.get("source_class") or "")
    not_a3 = bool(environment.get("not_a3_measured")
                  or source_class not in {"measured", "observed"})
    case_id = request.get("case_id") or f"case-{index}"
    protocol = source_derived_protocol(production, request)
    item = {
        "case_id": case_id,
        "status": production.get("status", "unknown"),
        "split": case.get("campaign_split") or production.get("split") or "train",
        "kind": "matmul" if family == "gemm" else family,
        "raw_device_ms_per_op": [float(v) for v in (production.get("raw_samples_ms") or ())],
        "duration_ms": production.get("duration_ms"),
        "dtype": request.get("dtype"),
        "layout": request.get("layout"),
        "features": request.get("features"),
        "source_family": family,
        "source_recipe": request.get("recipe"),
        "source_class": source_class or "unspecified",
        "source": production.get("source") or request.get("source"),
        "run_id": production.get("run_id") or request.get("run_id"),
        "hardware_key": request.get("hardware_key") or environment.get("device_name"),
        "environment": environment,
        "protocol": protocol,
        "not_a3_measured": not_a3,
        "coverage_scope": str(request.get("coverage_scope") or (
            "diagnostic_resource_probe" if family in {"add", "copy"}
            else "native_operator_semantic")),
        "native_precision_claim": family not in {"add", "copy"} and str(request.get("dtype")) == "bf16",
        "call_path": production.get("source"),
        "calibratable": not not_a3,
    }
    if family == "gemm":
        item["shape_mkn"] = [int(shape.get("m", 0)), int(shape.get("k", 0)),
                             int(shape.get("n", 0))]
    else:
        item["shape"] = [int(shape.get("elements", 0))]
    return item, not_a3


def linearity_calibration_groups(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Group calibration cases by true source/hardware/run/protocol identity.

    Non-measured/CPU/derived cases are never put in an importable ``cases``
    list.  They are preserved verbatim in ``excluded_cases`` so the core
    importer cannot promote them to A3 measured evidence.
    """
    groups: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    for index, case in enumerate(payload.get("cases", ())):
        item, not_a3 = _calibration_case(case, index)
        if not_a3:
            excluded.append({**item, "exclusion_reason": "not_a3_measured_or_unsupported_source"})
            continue
        key = _canonical({
            "environment": item["environment"], "protocol": item["protocol"],
            "hardware_key": item["hardware_key"], "source_class": item["source_class"],
            "run_id": item["run_id"],
        })
        group = groups.setdefault(key, {
            "status": "calibration_export",
            "environment": item["environment"],
            "protocol": item["protocol"],
            "hardware_key": item["hardware_key"],
            "source_class": item["source_class"],
            "run_id": item["run_id"],
            "cases": [],
            "provenance": {
                "source": item["source"], "source_class": item["source_class"],
                "run_id": item["run_id"], "hardware_key": item["hardware_key"],
                "note": "per-case identity is preserved; core importer must not replace it",
            },
        })
        group["cases"].append(item)
    return {
        "groups": [groups[key] for key in sorted(groups)],
        "excluded_cases": excluded,
        "note": "import each group separately; excluded CPU/derived evidence is not importable",
    }




def analyze_linearity_files(input_path: str | Path, *, output_dir: str | Path,
                            max_segments: int = DEFAULT_MAX_SEGMENTS) -> dict[str, Any]:
    payload = json.loads(Path(input_path).expanduser().read_text(encoding="utf-8"))
    manifest, results = adapt_linearity_split_results(payload)
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapted-manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (root / "adapted-results.json").write_text(
        json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    grouped = linearity_calibration_groups(payload)
    (root / "calibration-groups.json").write_text(
        json.dumps(grouped, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (root / "calibration-excluded-cases.json").write_text(
        json.dumps({"excluded_cases": grouped["excluded_cases"]},
                   indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return analyze_files(root / "adapted-manifest.json", root / "adapted-results.json",
                         output_dir=root, max_segments=max_segments)


def _campaign_series(shape: Mapping[str, int], origins: Sequence[Mapping[str, Any]]) -> tuple[str, int, dict[str, int], dict[str, Any]]:
    origin = dict(origins[0]) if origins else {}
    raw_axis = str(origin.get("axis") or "")
    if raw_axis == "tokens":
        axis = "rows" if "rows" in shape else "tokens"
    elif raw_axis in {"width_neighbor", "width"}:
        axis = "row_bytes" if "row_bytes" in shape else "width"
    elif raw_axis in shape:
        axis = raw_axis
    else:
        axis = next(iter(shape)) if shape else "elements"
    value = int(shape.get(axis, 0))
    fixed = {key: int(val) for key, val in shape.items() if key != axis}
    return axis, value, fixed, origin


def _series_identity_features(axis: str, features: Mapping[str, Any],
                              value: int | None = None) -> dict[str, Any]:
    """Remove point-varying routing facts from a derived-axis series identity."""
    result = dict(features or {})
    if axis == "tokens_per_expert":
        routing = dict(result.get("routing") or {})
        counts = [int(v) for v in (routing.pop("local_expert_counts", ()) or ())]
        if counts:
            if value and value > 0 and all(count % value == 0 for count in counts):
                routing["local_expert_counts"] = [
                    count // int(value) for count in counts]
            else:
                routing["local_expert_counts"] = counts
        for key in ("routing_tokens", "local_routing_tokens", "local_padded_tokens"):
            routing.pop(key, None)
        if routing:
            result["routing"] = routing
        else:
            result.pop("routing", None)
    return result


def adapt_campaign_manifest_results(manifest: Mapping[str, Any],
                                    results: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Adapt a native wave manifest/results pair using predeclared origins.

    The source manifest point carries only the full shape; the true series axis
    and anchor width are in ``origins[].axis``/``anchor_width``.  This adapter
    reconstructs fixed axes and series identities instead of producing one
    ``elements`` group per physical point.
    """
    results_by_measurement: dict[str, list[dict[str, Any]]] = {}
    seen_result_rows: set[str] = set()
    for item in results.get("results", ()):
        mid = str(item.get("measurement_id") or item.get("request_key"))
        row_identity = _canonical({
            "measurement_id": mid, "run_id": item.get("run_id"),
            "status": item.get("status"), "trials": item.get("trials"),
            "raw": item.get("raw_samples_ms"), "correctness": item.get("correctness"),
        })
        if row_identity in seen_result_rows:
            continue
        seen_result_rows.add(row_identity)
        results_by_measurement.setdefault(mid, []).append(dict(item))
    requests, adapted_results = [], []
    for request in manifest.get("requests", ()):
        midpoint = str(request.get("measurement_id") or request.get("request_key"))
        point = dict(request.get("point") or {})
        shape = {str(k): int(v) for k, v in (point.get("shape") or {}).items()}
        origins = [dict(item) for item in (request.get("origins") or ())]
        declared_axis = str(point.get("axis") or "")
        if declared_axis:
            axis = declared_axis
            value = int(point.get("value"))
            valid, reason = _validate_series(axis, shape, value)
            if not valid and axis not in {"dtype_probe"}:
                raise ValueError(f"{midpoint}: {reason}")
            shape = _normalize_series_shape(axis, value, shape)
            fixed = _axis_fixed_shape(axis, shape)
            origin = dict(origins[0]) if origins else {}
        else:
            axis, value, fixed, origin = _campaign_series(shape, origins)
        condition = dict(request.get("condition") or {})
        precision = dict(condition.get("precision") or {})
        source_features = dict(condition.get("features") or {})
        family = str(condition.get("family") or request.get("op_family") or "unknown")
        worker_recipe = str(request.get("worker_recipe") or condition.get("worker_recipe") or "")
        execution_kind = str(request.get("execution_kind") or condition.get("execution_kind") or "")
        target_match = request.get("target_match")
        if target_match is None:
            target_match = condition.get("target_match")
        if "target_match" in request:
            target_match = request.get("target_match")  # preserve explicit false
        coverage_scope = str((request.get("query_context") or {}).get("scope")
                             or request.get("coverage_scope") or "")
        series_features = _series_identity_features(axis, source_features, value)
        identity = _canonical({
            "family": family, "worker_recipe": worker_recipe, "execution_kind": execution_kind,
            "precision": precision, "layout": condition.get("layout"),
            "features": series_features, "axis": axis, "fixed": fixed,
            "target_match": target_match, "coverage_scope": coverage_scope,
        })
        condition_key = "wave2:" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]
        query_context = dict(request.get("query_context") or {})
        query_context.update({
            "family": family, "worker_recipe": worker_recipe,
            "execution_kind": execution_kind, "axis": axis,
            "anchor_width": origin.get("anchor_width"),
            "anchor_family": origin.get("anchor_family"),
            "dtype": precision.get("input_dtype"),
            "layout": condition.get("layout"), "precision": precision,
            "features": series_features, "target_match": target_match,
            "coverage_scope": coverage_scope,
        })
        protocol = dict(request.get("protocol") or {})
        run_id = str(request.get("run_id") or results.get("run_id") or "")
        work_raw = dict(request.get("work") or {})
        elements = int(work_raw.get("logical_input_elements") or 0)
        if not elements:
            elements = max(1, value * max([v for k, v in fixed.items()] or [1]))
        work = {"rate_work": float(elements), "unit": "elements",
                "traffic_provenance": "logical_input_elements from source manifest; not measured counters"}
        adapted_condition = {
            "condition_key": condition_key, "family": family,
            "recipe": request.get("recipe") or condition.get("recipe") or worker_recipe,
            "phase": request.get("phase") or condition.get("phase") or "operator",
            "execution_kind": execution_kind, "worker_recipe": worker_recipe,
            "backend": request.get("backend") or condition.get("backend"),
            "status": condition.get("status") or "executable",
            "shape_kind": condition.get("shape_kind") or family,
            "shape": fixed, "precision": precision, "layout": condition.get("layout"),
            "quantization": precision.get("quantization"),
            "features": {**series_features, "origin": origin,
                         "source_query_context": query_context},
            "target_match": target_match, "coverage_scope": coverage_scope,
            "boundaries": [], "protocol": protocol,
        }
        source_cases = [{
            "model": model, "origin": origin, "source_row_ids": origin.get("source_row_ids", []),
            "exact_source_row_ids": origin.get("exact_source_row_ids", []),
        } for model in (origin.get("models") or [])]
        result_list = results_by_measurement.get(midpoint, [])
        result_for_request = result_list[0] if result_list else {}
        requested_hardware_key = str(request.get("hardware_key") or "")
        observed_hardware_key = str(
            (result_for_request.get("environment") or {}).get("device_name") or "")
        hardware_key = observed_hardware_key or requested_hardware_key
        adapted_request = {
            "measurement_id": midpoint, "request_key": midpoint,
            "condition_key": condition_key, "op_family": family,
            "hardware_key": hardware_key,
            "requested_hardware_key": requested_hardware_key,
            "observed_hardware_key": observed_hardware_key,
            "phase": adapted_condition["phase"], "recipe": adapted_condition["recipe"],
            "split": str(request.get("split", "train")), "execution_kind": execution_kind,
            "worker_recipe": worker_recipe, "backend": adapted_condition["backend"],
            "dtype": precision.get("input_dtype"), "layout": condition.get("layout"),
            "condition": adapted_condition,
            "point": {"axis": axis, "value": value, "shape": shape,
                      "semantic_shape": [shape[k] for k in shape],
                      "source_axis": str(point.get("axis") or axis),
                      "split": str(request.get("split", "train"))},
            "query_context": query_context, "protocol": protocol,
            "run_id": run_id,
            "run_ids": sorted({str(item.get("run_id") or run_id) for item in result_list}),
            "target_match": target_match, "coverage_scope": coverage_scope,
            "source_class": result_for_request.get("source_class"),
            "analytic_ms": None,
            "work": work, "source_cases": source_cases,
            "origins": origins, "split_reasons": request.get("split_reasons") or [],
        }
        requests.append(adapted_request)
        for result in result_list:
            adapted_results.append({
                **result, "condition_key": condition_key,
                "execution_kind": execution_kind, "worker_recipe": worker_recipe,
                "backend": adapted_condition["backend"],
                "split": str(request.get("split", result.get("split", "train"))),
                "run_id": result.get("run_id") or run_id,
                "hardware_key": hardware_key,
                "requested_hardware_key": requested_hardware_key,
                "observed_hardware_key": observed_hardware_key,
                "target_match": target_match, "coverage_scope": coverage_scope,
                "condition": adapted_condition,
                "protocol": dict(result.get("protocol") or protocol),
                "analytic_ms": None,
            })
    adapted_manifest = {
        "campaign_id": str(manifest.get("run_id") or results.get("run_id") or "native-wave2"),
        "status": "adapted",
        "requests": requests,
        "inventory": [],
        "models": {},
        "coverage": {
            "selected_train": sum(1 for item in requests if item["split"] == "train"),
            "selected_heldout": sum(1 for item in requests if item["split"] != "train"),
            "pending_conditions": 0, "derivation_failures": [],
            "source_excluded_rows": len(manifest.get("excluded_rows") or ()),
            "series_count": len({item["condition_key"] for item in requests}),
        },
        "pending_requests": [],
        "source_manifest": {
            "run_id": manifest.get("run_id"), "scope": manifest.get("scope"),
            "split_policy": manifest.get("split_policy"),
            "protocol": manifest.get("protocol"),
            "excluded_rows": manifest.get("excluded_rows") or [],
        },
    }
    adapted_results_payload = {
        "status": results.get("status", "completed"),
        "results": adapted_results,
        "run_id": results.get("run_id") or manifest.get("run_id"),
        "protocol": results.get("protocol") or manifest.get("protocol") or {},
        "environment": results.get("environment") or {},
        "not_a3_measured": bool(results.get("not_a3_measured")),
    }
    return adapted_manifest, adapted_results_payload


def campaign_query_from_request(request: Mapping[str, Any],
                               protocol: Mapping[str, Any] | None = None,
                               hardware_key: str | None = None) -> dict[str, Any]:
    """Build a source-derived query from an original campaign request."""
    condition = dict(request.get("condition") or {})
    point = dict(request.get("point") or {})
    axis = str(point.get("axis") or "")
    shape = dict(point.get("shape") or {})
    value = int(point.get("value") or 0)
    if not axis or not shape:
        raise ValueError("campaign_query_missing_declared_axis_or_shape")
    source_features = {key: value for key, value in (condition.get("features") or {}).items()
                       if key not in {"origin", "source_query_context"}}
    features = _series_identity_features(axis, source_features, value)
    precision = dict(condition.get("precision") or {})
    family = request.get("op_family") or condition.get("family")
    coverage_scope = str((request.get("query_context") or {}).get("scope")
                         or request.get("coverage_scope") or (
                             "diagnostic_resource_probe" if family in {"add", "copy"}
                             else "native_operator_semantic"))
    query_protocol = dict(protocol or request.get("protocol") or {})
    if query_protocol.get("warmup_scope") == "before_each_trial":
        warmup = int(query_protocol.get("warmup", 10))
        trials = int(query_protocol.get("trials", 2))
        query_protocol.setdefault("warmup_iterations_per_trial", warmup)
        query_protocol.setdefault("warmup_events_total", warmup * trials)
    return {
        "hardware_key": str(hardware_key or request.get("hardware_key") or ""),
        "requested_hardware_key": request.get("hardware_key") or "",
        "family": family,
        "recipe": request.get("recipe") or condition.get("recipe"),
        "execution_kind": request.get("execution_kind") or condition.get("execution_kind"),
        "worker_recipe": request.get("worker_recipe") or condition.get("worker_recipe"),
        "backend": request.get("backend") or condition.get("backend"),
        "target_match": (request.get("target_match")
                         if "target_match" in request else condition.get("target_match")),
        "coverage_scope": coverage_scope,
        "layout": request.get("layout") or condition.get("layout"),
        "quantization": precision.get("quantization"),
        "precision": precision,
        "features": features,
        "axis": axis, "shape": shape,
        "protocol": protocol_key(query_protocol),
    }


def analyze_campaign_files(manifest_path: str | Path, results_path: str | Path, *,
                           output_dir: str | Path,
                           max_segments: int = DEFAULT_MAX_SEGMENTS) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).expanduser().read_text(encoding="utf-8"))
    results = json.loads(Path(results_path).expanduser().read_text(encoding="utf-8"))
    adapted_manifest, adapted_results = adapt_campaign_manifest_results(manifest, results)
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapted-manifest.json").write_text(
        json.dumps(adapted_manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (root / "adapted-results.json").write_text(
        json.dumps(adapted_results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    (root / "source-evidence.json").write_text(json.dumps({
        "source_manifest": str(manifest_path), "source_results": str(results_path),
        "run_id": manifest.get("run_id"),
        "scope": manifest.get("scope"), "split_policy": manifest.get("split_policy"),
        "environment": results.get("environment"), "protocol": results.get("protocol"),
    }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    excluded_cases = []
    result_by_id = {str(item.get("measurement_id")): item for item in adapted_results["results"]}
    for request in adapted_manifest["requests"]:
        result = result_by_id.get(str(request["measurement_id"])) or {}
        excluded_cases.append({
            "case_id": request["measurement_id"],
            "kind": "excluded_native_semantic",
            "status": result.get("status", "pending"),
            "split": request["split"],
            "source_family": request["op_family"],
            "source_recipe": request.get("recipe"),
            "worker_recipe": request.get("worker_recipe"),
            "shape": request["point"]["shape"],
            "axis": request["point"]["axis"],
            "anchor_width": (request.get("origins") or [{}])[0].get("anchor_width"),
            "source_class": result.get("source_class"),
            "source": result.get("source"),
            "run_id": result.get("run_id") or request.get("run_id"),
            "target_match": request.get("target_match"),
            "coverage_scope": request.get("coverage_scope"),
            "precision": (request.get("condition") or {}).get("precision"),
            "physical_features": {
                key: value for key, value in
                ((request.get("condition") or {}).get("features") or {}).items()
                if key not in {"origin", "source_query_context"}
            },
            "hardware_key": (result.get("environment") or {}).get("device_name")
            or (request.get("origins") or [{}])[0].get("hardware_key"),
            "environment": result.get("environment"),
            "protocol": result.get("protocol") or request.get("protocol"),
            "not_a3_measured": result.get("not_a3_measured"),
            "calibratable": False,
            "exclusion_reason": "native primitive semantics are not legacy calibration matmul/add/copy rows",
        })
    (root / "native-calibration-excluded-cases.json").write_text(json.dumps({
        "excluded_cases": excluded_cases,
        "note": "not imported as legacy calibration rows; provenance retained for a native-aware core path",
    }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return analyze_files(root / "adapted-manifest.json", root / "adapted-results.json",
                         output_dir=root, max_segments=max_segments)


def normalized_conditions(manifest: Mapping[str, Any],
                          results: Mapping[str, Any],
                          predictions: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Export the independent reviewer's pre-use condition shape.

    This is an adapter for the prospective acceptance script, not a second
    fitter.  Query context contains only pre-use facts; observed kernel/profiler
    context stays in the worker result and is never used to choose a regime.
    """
    results_by_measurement: dict[str, list[dict[str, Any]]] = {}
    for item in results.get("results", ()):
        mid = str(item.get("measurement_id") or item.get("request_key"))
        results_by_measurement.setdefault(mid, []).append(dict(item))
    rows = []
    seen_identity: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    for request in manifest.get("requests", ()):
        midpoint = str(request.get("measurement_id") or request.get("request_key"))
        result_list = results_by_measurement.get(midpoint, [])
        result = result_list[0] if result_list else {}
        condition = request.get("condition") or {}
        precision = condition.get("precision") or {}
        point = request.get("point") or {}
        query_context = {
            "family": request.get("op_family"),
            "phase": request.get("phase"),
            "recipe": request.get("recipe"),
            "execution_kind": request.get("execution_kind"),
            "worker_recipe": request.get("worker_recipe"),
            "requested_backend": request.get("backend"),
            "hardware_key": request.get("hardware_key"),
            "target_match": request.get("target_match"),
            "dtype": precision.get("input_dtype") or request.get("dtype"),
            "weight_dtype": precision.get("weight_dtype"),
            "output_dtype": precision.get("output_dtype"),
            "compute_dtype": precision.get("compute_dtype"),
            "accumulation_dtype": precision.get("accumulation_dtype"),
            "layout": request.get("layout"),
            "quantization": condition.get("quantization"),
            "target_match": request.get("target_match"),
            "coverage_scope": request.get("coverage_scope"),
            "physical_features": {
                key: value for key, value in (condition.get("features") or {}).items()
                if key not in {"origin", "source_query_context"}
            },
            "parallel": (condition.get("features") or {}).get("parallel"),
            "protocol": protocol_key(request.get("protocol") or {}),
        }
        identity = _canonical({"query_context": query_context, "shape": point.get("shape")})
        split = str(request.get("split", "train"))
        if identity in seen_identity:
            if seen_identity[identity] == split:
                continue  # exact duplicate within the same precommitted split
            conflicts.append({
                "identity": identity, "condition_id": midpoint, "split": split,
                "previous_split": seen_identity[identity],
                "reason": "same physical query_context+shape appears in both splits",
            })
            # Keep the conflicting row so the reviewer/root can reject the leak;
            # never silently drop the heldout side.
        else:
            seen_identity[identity] = split
        samples = []
        seen_trials: set[tuple[str, str]] = set()
        for candidate in result_list:
            candidate_run = str(candidate.get("run_id") or request.get("run_id") or "")
            if candidate.get("status") == "completed":
                trials = candidate.get("trials") or ()
                if not trials and candidate.get("duration_ms") is not None:
                    trials = ({"trial_index": 0, "median_ms": candidate.get("duration_ms")},)
                for index, trial in enumerate(trials):
                    trial_id = f"{candidate_run}:{trial.get('trial_index', index)}"
                    key = (candidate_run, str(trial.get("trial_index", index)))
                    if key in seen_trials:
                        continue
                    seen_trials.add(key)
                    samples.append({
                        "trial_id": trial_id, "status": "completed",
                        "duration_ms": trial.get("median_ms"),
                    })
            else:
                trial_id = f"{candidate_run}:0"
                key = (candidate_run, "0")
                if key in seen_trials:
                    continue
                seen_trials.add(key)
                samples.append({
                    "trial_id": trial_id, "status": str(candidate.get("status", "pending")),
                    "duration_ms": None,
                })
        if not samples:
            samples.append({
                "trial_id": f"{request.get('run_id') or 'unknown'}:0",
                "status": "pending", "duration_ms": None,
            })
        work = request.get("work") or {}
        prediction = {}
        if predictions:
            prediction = dict(predictions.get(midpoint) or predictions.get(
                str(request.get("condition_key"))) or {})
        rows.append({
            "condition_id": midpoint,
            "split": split,
            "source_class": result.get("source_class") or request.get("source_class"),
            "source": result.get("source") or request.get("source"),
            "run_id": result.get("run_id") or request.get("run_id"),
            "run_ids": sorted({str(item.get("run_id") or request.get("run_id") or "")
                               for item in result_list}),
            "execution_evidence": [{
                "source_class": item.get("source_class") or request.get("source_class"),
                "source": item.get("source") or request.get("source"),
                "run_id": item.get("run_id") or request.get("run_id"),
                "environment": item.get("environment") or request.get("environment"),
                "protocol": item.get("protocol") or request.get("protocol"),
            } for item in result_list],
            "axis": point.get("axis"),
            "axis_value": point.get("value"),
            "regime": {
                **{key: query_context[key] for key in (
                    "family", "phase", "recipe", "execution_kind", "worker_recipe",
                    "requested_backend", "dtype", "weight_dtype", "compute_dtype",
                    "layout", "quantization", "parallel", "protocol",
                )},
                "axis": point.get("axis"),
                "fixed_shape": {key: value for key, value in (point.get("shape") or {}).items()
                                if key != point.get("axis")},
            },
            "query_context": query_context,
            "shape": point.get("shape"),
            "samples": samples,
            "predictions": {
                "analytic_ms": prediction.get("analytic_ms", next(
                    (item.get("analytic_ms") for item in result_list
                     if item.get("analytic_ms") is not None),
                    request.get("analytic_ms"))),
                "calibrated_ms": prediction.get("calibrated_ms"),
                "source": prediction.get("source", "caller_frozen_public_prediction"),
            },
            "work": work.get("rate_work"),
            "work_unit": work.get("unit"),
            "traffic_provenance": "logical work estimate; not measured counters",
            "holdout_stratum": "extrapolation" if request.get("extrapolation") else "anchor_or_interpolation",
        })
    return {
        "campaign_id": manifest.get("campaign_id"),
        "plateau_relative_tolerance": 0.15,
        "valid": not conflicts,
        "conflicts": conflicts,
        "conditions": rows,
        "note": "pre-use query_context only; observed_context is intentionally excluded; split conflicts are preserved, not dropped",
    }


