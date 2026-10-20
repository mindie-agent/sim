"""Public campaign <-> native FA/RoPE attention adapter.

[Role]
  Adapt campaign requests with explicit attention/rope geometry to the
  accepted standalone attention manifest/plan/runner, and convert its
  three-clock results back into campaign-shaped results for curve analysis.

[Invariants]
  * The accepted standalone `a3_operator_worker_attention` implementation is
    reused through its public manifest/plan/run API; no tensor/math code is
    copied, monkeypatched or routed through the generic event-only loop.
  * FA/RoPE cases carry explicit normalized shape, mask/GQA/scale/layout and
    half-RoPE/positions semantics. Missing facts become precise pending fields,
    never inferred defaults.
  * All cases are target_match=false; split is copied exactly.
  * Device results retain cpu-submit, device-event and wall-complete clocks,
    per-trial raw samples, native allocation/auxiliary facts and failures.  The
    curve-facing device interval is explicitly inclusive of host gaps and
    native output extraction; it is never labelled pure device service.
  * Dry-run imports no torch and writes nothing.

[Timeouts & Locks]
  Pure validation/conversion; real dispatch delegates to the standalone
  attention worker's cooperative budgets and supervisor contract.

[Side-Effect Policy]
  In-memory APIs no I/O.  `run_campaign_attention` writes only the explicit
  manifest/results paths and delegates device execution to sim_audit.
"""
from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from .a3_operator_curve_axis import fixed_shape as _axis_fixed_shape

CAMPAIGN_ATTENTION_PROTOCOL = {
    "timer": "device_event",
    "timing_scope": "device_event_block_interval_inclusive_host_gap_native_allocation",
    "measurement_scope": "inclusive_event_interval_with_host_gap_and_native_output_extraction",
    "interval_scope": "device_event_block_interval_inclusive_host_gap_native_allocation",
    "warmup_scope": "before_each_trial",
}
_OPERATOR_FAMILIES = {
    "fusion_attention": "fusion_attention",
    "npu_fusion_attention": "fusion_attention",
    "attention_qk": "fusion_attention",
    "attention_pv": "fusion_attention",
    "rotary_mul": "rotary_mul",
    "npu_rotary_mul": "rotary_mul",
    "rope": "rotary_mul",
}


def _manifest_module():
    from . import a3_operator_worker_attention as attention
    return attention


def _source_contract_complete(source: Mapping[str, Any]) -> bool:
    lines = source.get("schema_lines")
    return (isinstance(lines, Mapping)
            and all(key in lines for key in ("npu_fusion_attention", "npu_rotary_mul"))
            and all(source.get(key) for key in (
                "native_functions_yaml", "sha256", "scope")))


def _request_id(request: Mapping[str, Any]) -> str:
    return str(request.get("request_key") or request.get("measurement_id")
               or request.get("condition_key") or "request")


def _protocol_from_request(request: Mapping[str, Any],
                           template: Mapping[str, Any]) -> dict[str, Any]:
    protocol = dict(request.get("protocol") or {})
    base = dict(template.get("protocol") or {})
    return {
        "warmup_iterations": int(protocol.get("warmup", base.get("warmup_iterations", 10))),
        "trials": int(protocol.get("trials", base.get("trials", 2))),
        "samples_per_trial": int(protocol.get("iterations", base.get("samples_per_trial", 30))),
        "seed": int(protocol.get("seed", base.get("seed", 20261021))),
        "reference": dict(base.get("reference") or {"atol": 0.02, "rtol": 0.02}),
        "budgets_seconds": dict(base.get("budgets_seconds") or {
            "case": 24, "setup": 60, "global_work": 240, "cleanup": 60}),
        "supervisor_seconds_min": float(base.get("supervisor_seconds_min", 370)),
    }


def _unsupported_semantics(request: Mapping[str, Any]) -> list[str]:
    condition = request.get("condition") or {}
    metadata = dict((condition.get("features") or {}).get("metadata") or {})
    case = request.get("attention_case") if isinstance(request.get("attention_case"), Mapping) else {}
    merged = {**metadata, **dict(case.get("source") or {}), **dict(case)}
    issues = []
    rotary_mode = merged.get("rotary_mode") or request.get("rotary_mode")
    if rotary_mode not in (None, "half"):
        issues.append(f"rotary_mode {rotary_mode!r} unsupported (only half)")
    attention = str(merged.get("attention") or merged.get("attention_type") or "").lower()
    if attention in {"mla", "dsa", "sparse", "paged", "page_attention"}:
        issues.append(f"attention semantics {attention!r} unsupported")
    primitive = str(merged.get("primitive_scope") or "").lower()
    if any(token in primitive for token in ("mrope", "mla", "paged", "sparse", "fp8")):
        if "not" not in primitive and "no " not in primitive:
            issues.append("primitive scope includes unsupported mRoPE/MLA/paged/sparse/FP8")
    precision = dict(condition.get("precision") or {})
    quant = str(precision.get("quantization") or "none")
    if quant not in {"none", "bf16", "bfloat16", ""}:
        issues.append(f"quantized attention/rope semantics {quant!r} unsupported")
    return sorted(set(issues))


def _missing_case_fields(request: Mapping[str, Any]) -> list[str]:
    case = request.get("attention_case")
    if isinstance(case, Mapping) and case:
        return []
    condition = request.get("condition") or {}
    metadata = dict((condition.get("features") or {}).get("metadata") or {})
    shape = dict(request.get("point", {}).get("shape") or request.get("shape") or {})
    family = str(request.get("op_family") or condition.get("family") or "")
    operator = _OPERATOR_FAMILIES.get(family)
    missing = []
    if operator is None:
        missing.append("operator fusion_attention/rotary_mul")
    if operator == "fusion_attention":
        for key in ("batch", "q_heads", "kv_heads", "q_tokens", "kv_tokens", "qk_dim", "v_dim"):
            if key not in shape and key not in metadata:
                missing.append(f"shape.{key}")
        for key in ("mask", "scale"):
            if key not in metadata and key not in request:
                missing.append(key)
        if str(request.get("layout") or condition.get("layout")) != "BNSD":
            missing.append("layout BNSD")
        for key in ("model", "source_scope", "primitive_scope",
                    "topology", "global_q_heads", "global_kv_heads", "head_dim"):
            if key not in metadata:
                missing.append(f"source.{key}")
    elif operator == "rotary_mul":
        for key in ("batch", "seq", "heads", "dim"):
            if key not in shape and key not in metadata:
                missing.append(f"shape.{key}")
        for key in ("rotary_mode", "positions"):
            if key not in metadata and key not in request:
                missing.append(key)
        if str(request.get("layout") or condition.get("layout")) != "BSND":
            missing.append("layout BSND")
        for key in ("model", "source_scope", "primitive_scope",
                    "partial_rotary_factor", "full_head_dim", "rotated_dim"):
            if key not in metadata:
                missing.append(f"source.{key}")
    if "data" not in request and "data" not in metadata:
        missing.append("data.seed/low/high")
    if request.get("target_match") is not False:
        missing.append("target_match=false")
    return sorted(set(missing))


def _case_from_request(request: Mapping[str, Any]) -> dict[str, Any] | None:
    case = request.get("attention_case")
    if isinstance(case, Mapping) and case:
        candidate = dict(case)
        if candidate.get("target_match") is not False:
            return None
        return candidate
    missing = _missing_case_fields(request)
    if missing:
        return None
    condition = request.get("condition") or {}
    metadata = dict((condition.get("features") or {}).get("metadata") or {})
    shape = dict(request.get("point", {}).get("shape") or request.get("shape") or {})
    family = str(request.get("op_family") or condition.get("family") or "")
    operator = _OPERATOR_FAMILIES[family]
    source = {key: metadata.get(key) for key in (
        "model", "topology", "global_q_heads", "global_kv_heads", "head_dim",
        "kv_heads_replicated", "source_scope", "primitive_scope",
        "partial_rotary_factor", "full_head_dim", "rotated_dim") if metadata.get(key) is not None}
    source.setdefault("target_match", False)
    common = {
        "case_id": _request_id(request).replace(":", "-")[:128],
        "operator": operator, "dtype": "bfloat16",
        "layout": "BNSD" if operator == "fusion_attention" else "BSND",
        "shape": {key: int(shape[key]) for key in (
            ("batch", "q_heads", "kv_heads", "q_tokens", "kv_tokens", "qk_dim", "v_dim")
            if operator == "fusion_attention" else ("batch", "seq", "heads", "dim"))
            if key in shape},
        "data": dict(metadata.get("data") or request.get("data") or {
            "seed": int(metadata.get("seed", 20261021)), "low": -1.0, "high": 1.0}),
        "source": source, "target_match": False,
        "split": str(request.get("split", "train")),
    }
    if operator == "fusion_attention":
        common.update({"mask": metadata.get("mask") or request.get("mask"),
                       "scale": float(metadata.get("scale") or request.get("scale"))})
    else:
        common.update({"rotary_mode": metadata.get("rotary_mode") or request.get("rotary_mode"),
                       "positions": dict(metadata.get("positions") or request.get("positions") or {}),
                       "inv_freq_base": float(metadata.get("inv_freq_base", 10000.0))})
    return common


def _series_identity(case: Mapping[str, Any]) -> tuple[str, str, int, dict[str, Any],
                                                          dict[str, Any], str]:
    """Return operator, canonical axis, value, full shape, fixed shape, reason."""
    operator = str(case["operator"])
    source = dict(case.get("source") or {})
    declared_axis = str(source.get("axis") or "")
    shape = {str(k): int(v) for k, v in dict(case.get("shape") or {}).items()}
    if operator == "fusion_attention":
        if declared_axis == "q_kv":
            axis, value = "q_kv", shape["q_tokens"]
            if shape["kv_tokens"] != value:
                raise ValueError("q_kv requires q_tokens == kv_tokens")
            shape = {**shape, "q": value, "kv": value}
            shape.pop("q_tokens", None); shape.pop("kv_tokens", None)
        elif declared_axis in {"kv", "q"}:
            axis = declared_axis
            value = shape["kv_tokens"] if axis == "kv" else shape["q_tokens"]
            shape = {**shape, "q": shape["q_tokens"], "kv": shape["kv_tokens"]}
            shape.pop("q_tokens", None); shape.pop("kv_tokens", None)
        else:
            raise ValueError(f"unsupported_attention_axis:{declared_axis}")
        context = (
            f"{operator}:{case.get('layout')}:{case.get('dtype')}:{axis}:"
            f"{shape.get('batch')}:{shape.get('q_heads')}:{shape.get('kv_heads')}:"
            f"{shape.get('qk_dim')}:{shape.get('v_dim')}:{case.get('mask')}:"
            f"{case.get('scale')}")
    elif operator == "rotary_mul":
        if declared_axis not in {"seq", "tokens"}:
            raise ValueError(f"unsupported_rope_axis:{declared_axis}")
        axis = "tokens"
        value = shape["seq"]
        shape = {"tokens": value, "batch": shape["batch"],
                 "heads": shape["heads"], "width": shape["dim"]}
        context = (
            f"{operator}:{case.get('layout')}:{case.get('dtype')}:{axis}:"
            f"{shape.get('batch')}:{shape.get('heads')}:{shape.get('width')}:"
            f"{case.get('rotary_mode')}:{source.get('positions_start')}:"
            f"{case.get('inv_freq_base')}")
    else:
        raise ValueError(f"unsupported_attention_operator:{operator}")
    fixed = _axis_fixed_shape(axis, shape)
    return operator, axis, int(value), shape, fixed, context


def campaign_requests_from_attention(manifest: Mapping[str, Any], *,
                                     hardware_key: str = "") -> list[dict[str, Any]]:
    """Create direct campaign requests with source-declared derived axes."""
    requests = []
    base_protocol = {
        **CAMPAIGN_ATTENTION_PROTOCOL,
        "warmup": int(manifest.get("protocol", {}).get("warmup_iterations", 10)),
        "iterations": int(manifest.get("protocol", {}).get("samples_per_trial", 30)),
        "trials": int(manifest.get("protocol", {}).get("trials", 2)),
        "preallocated": False,
        "cache_policy": "one_attention_stream_reused_events_allocating_native_outputs",
        "seed": int(manifest.get("protocol", {}).get("seed", 20261021)),
    }
    for case in manifest.get("cases", ()):
        operator, axis, value, point_shape, fixed, series_context = _series_identity(case)
        source = dict(case.get("source") or {})
        if operator == "fusion_attention":
            features = {
                "mask": case.get("mask"), "scale": case.get("scale"),
                "batch": int(point_shape["batch"]), "q_heads": int(point_shape["q_heads"]),
                "kv_heads": int(point_shape["kv_heads"]),
                "qk_dim": int(point_shape["qk_dim"]), "v_dim": int(point_shape["v_dim"]),
                "layout": "BNSD", "source_axis": str(source.get("axis") or ""),
                "native_compute_precision": "backend_default_not_observed",
            }
            family, recipe = "attention_qk", "npu_fusion_attention"
            backend = "torch_npu.npu_fusion_attention"
            shape_kind = "attention"
        else:
            features = {
                "rotary_mode": case.get("rotary_mode"),
                "positions_start": int((case.get("positions") or {}).get("start", 1)),
                "positions_rule": "end_equals_sequence",
                "inv_freq_base": float(case.get("inv_freq_base", 10000.0)),
                "batch": int(point_shape["batch"]), "heads": int(point_shape["heads"]),
                "width": int(point_shape["width"]),
                "layout": "BSND", "source_axis": str(source.get("axis") or ""),
                "native_compute_precision": "backend_default_not_observed",
            }
            family, recipe = "rope", "npu_rotary_mul"
            backend = "torch_npu.npu_rotary_mul"
            shape_kind = "vector"
        precision = {
            "input_dtype": "bfloat16", "output_dtype": "bfloat16",
            "compute_dtype": "backend_default", "compute_format": "backend_default",
            "accumulation_dtype": "backend_default", "quantization": "none",
            "weight_format": None, "weight_dtype": None,
            "activation_format": "bfloat16", "quant_output_dtype": "bfloat16",
            "weight_block": [0, 0], "expert_block": 0, "scale_format": "",
            "scale_bytes": 0.0, "scale_shape": [], "scale_axis": "",
            "row_width": 0, "kv_dtype": "",
        }
        identity = json.dumps({
            "operator": operator, "family": family, "recipe": recipe,
            "backend": backend, "execution_kind": "fused_kernel",
            "layout": case.get("layout"), "dtype": case.get("dtype"),
            "axis": axis, "fixed": fixed, "features": features,
            "precision": precision, "protocol": base_protocol,
            "hardware_key": hardware_key,
            "target_match": False, "coverage_scope": "native_primitive_attention_rope_capability",
        }, sort_keys=True, separators=(",", ":"))
        condition_key = "attn:" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:20]
        condition = {
            "condition_key": condition_key, "family": family, "recipe": recipe,
            "phase": "operator", "execution_kind": "fused_kernel",
            "worker_recipe": recipe, "backend": backend,
            "target_match": False, "status": "conditional_backend",
            "reason": "standalone accepted native capability; exact case attached",
            "required_backend": "A3 native attention worker",
            "shape_kind": shape_kind, "shape": fixed, "precision": precision,
            "layout": "contiguous", "quantization": "none",
            "coverage_scope": "native_primitive_attention_rope_capability",
            "features": features, "state": {}, "boundaries": [], "expansion": {},
            "protocol": base_protocol, "sources": {},
        }
        requests.append({
            "request_key": f"attention:{case['case_id']}",
            "measurement_id": f"attention:{case['case_id']}",
            "condition_key": condition_key, "op_family": family,
            "phase": "operator", "recipe": recipe, "shape": point_shape,
            "dtype": "bfloat16", "layout": "contiguous",
            "condition": condition,
            "point": {"axis": axis, "value": value, "shape": point_shape,
                      "source_axis": str(source.get("axis") or ""),
                      "source_value": source.get("value")},
            "split": str(case.get("split", "train")),
            "execution_kind": "fused_kernel", "worker_recipe": recipe,
            "backend": backend, "target_match": False,
            "coverage_scope": "native_primitive_attention_rope_capability",
            "attention_case": dict(case), "protocol": base_protocol,
            "hardware_key": hardware_key, "analytic_ms": None,
            "query_context": {
                "axis": axis, "declared_axis": str(source.get("axis") or ""),
                "shape_coupling": "q=kv=x" if axis == "q_kv"
                else "seq->tokens" if operator == "rotary_mul" else "single_axis",
                "series_context": series_context, "features": features,
                "precision": precision, "scope": "native_attention_capability",
            },
            "source_cases": [{"model": source.get("model"),
                              "source_row_ids": source.get("source_row_ids") or [],
                              "source_configs": source.get("source_configs") or []}],
            "work": None,
        })
    return requests


def attention_query_from_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Build a public query from the original campaign request, not artifact rows."""
    condition = dict(request.get("condition") or {})
    point = dict(request.get("point") or {})
    features = {key: value for key, value in (condition.get("features") or {}).items()
                if key not in {"origin", "source_query_context"}}
    protocol = dict(request.get("protocol") or {})
    protocol.setdefault("warmup_iterations_per_trial", protocol.get("warmup", 10))
    return {
        "hardware_key": request.get("hardware_key") or "",
        "family": request.get("op_family") or condition.get("family"),
        "recipe": request.get("recipe") or condition.get("recipe"),
        "execution_kind": request.get("execution_kind") or condition.get("execution_kind"),
        "worker_recipe": request.get("worker_recipe") or condition.get("worker_recipe"),
        "backend": request.get("backend") or condition.get("backend"),
        "target_match": request.get("target_match", condition.get("target_match")),
        "coverage_scope": request.get("coverage_scope"),
        "layout": request.get("layout") or condition.get("layout"),
        "quantization": request.get("quantization") or condition.get("quantization"),
        "precision": dict(condition.get("precision") or {}),
        "features": features,
        "axis": point.get("axis"),
        "shape": dict(point.get("shape") or {}),
        "protocol": protocol,
    }


def attention_manifest_from_campaign(requests: Sequence[Mapping[str, Any]],
                                     template: Mapping[str, Any], *,
                                     manifest_id: str = "a3-campaign-attention") -> dict[str, Any]:
    """Build/validate an attention manifest from campaign requests.

    Returns ``{"manifest": ..., "pending": [...]}``.  Pending entries list the
    exact missing facts; unsupported composites are never inferred.
    """
    cases, pending = [], []
    protocols = []
    protocol_canonical = None
    for request in requests:
        unsupported = _unsupported_semantics(request)
        missing = _missing_case_fields(request)
        if unsupported or missing:
            pending.append({"request_key": _request_id(request), "status": "pending",
                            "reason": "unsupported_native_semantics" if unsupported
                            else "missing_attention_case_fields",
                            "missing_fields": sorted(set(missing + unsupported))})
            continue
        case = _case_from_request(request)
        if case is None:
            pending.append({"request_key": _request_id(request), "status": "pending",
                            "missing_fields": ["target_match=false"]})
            continue
        request_protocol = _protocol_from_request(request, template)
        serialized = json.dumps(request_protocol, sort_keys=True, separators=(",", ":"))
        if protocol_canonical is not None and serialized != protocol_canonical:
            pending.append({"request_key": _request_id(request), "status": "pending",
                            "reason": "attention_manifest_single_protocol",
                            "missing_fields": ["matching protocol for one manifest"]})
            continue
        protocol_canonical = serialized
        cases.append(case)
        protocols.append(request_protocol)
    source_contract = dict(template.get("source_contract") or {})
    attribution_complete = _source_contract_complete(source_contract)
    attribution_optional = not attribution_complete
    protocol = protocols[0] if protocols else dict(template.get("protocol") or {})
    trials = int(protocol.get("trials", 2))
    samples = int(protocol.get("samples_per_trial", 30))
    manifest = {
        "kind": "a3_attention_rope_manifest",
        "manifest_id": manifest_id,
        "description": "campaign attention adapter; target_match=false primitive cases",
        "protocol": protocol,
        "expectations": {"cases": len(cases),
                         "planned_raw_blocks": len(cases) * trials * samples},
        "cases": cases,
    }
    if attribution_complete:
        manifest["source_contract"] = source_contract
    module = _manifest_module()
    case_errors = []
    normalized = []
    for index, case in enumerate(cases):
        errors = []
        try:
            checked = module._case(case, index, errors)
        except Exception as exc:  # noqa: BLE001 - adapter boundary
            checked, errors = None, [f"adapter_case_exception:{type(exc).__name__}:{exc}"]
        if checked is None or errors:
            case_errors.append({"case_id": case.get("case_id", index),
                                "errors": list(errors)})
        else:
            normalized.append(checked)
    if case_errors:
        pending.append({"request_key": "manifest", "status": "pending",
                        "reason": "attention_case_physical_invalid",
                        "case_errors": case_errors})
    if cases and not case_errors:
        try:
            module.validate_manifest(manifest)
        except Exception as exc:
            pending.append({"request_key": "manifest", "status": "pending",
                            "reason": f"attention_manifest_invalid:{type(exc).__name__}:{exc}"})
    return {"manifest": manifest, "pending": pending,
            "ready_cases": len(cases), "requested_cases": len(requests),
            "attribution_optional": attribution_optional,
            "physical_cases": normalized}


def dry_run_campaign_attention(requests: Sequence[Mapping[str, Any]],
                               template: Mapping[str, Any], *,
                               manifest_id: str = "a3-campaign-attention") -> dict[str, Any]:
    built = attention_manifest_from_campaign(requests, template, manifest_id=manifest_id)
    module = _manifest_module()
    if built["ready_cases"] and not built["pending"]:
        try:
            plan = module.build_plan(built["manifest"])
            valid = True
            validation = "stdlib_only_no_torch_standalone_validate"
        except Exception as exc:
            plan = None
            valid = False
            validation = f"invalid:{type(exc).__name__}:{exc}"
    else:
        plan = None
        valid = False
        validation = "stdlib_only_no_torch_no_ready_cases"
    return {"dry_run": True, "requires_device": False, "writes_results": False,
            "valid": valid and not built["pending"], "validation": validation,
            "plan": plan, "pending": built["pending"],
            "ready_cases": built["ready_cases"],
            "requested_cases": built["requested_cases"],
            "attribution_optional": built.get("attribution_optional", False)}


def run_campaign_attention(requests: Sequence[Mapping[str, Any]],
                           template: Mapping[str, Any], *, manifest_path: str | Path,
                           results_path: str | Path, run_id: str,
                           device: str = "npu:0", allow_overwrite: bool = False,
                           manifest_id: str = "a3-campaign-attention") -> dict[str, Any]:
    built = attention_manifest_from_campaign(requests, template, manifest_id=manifest_id)
    if built["pending"] or not built["ready_cases"]:
        return {"status": "pending", "pending": built["pending"],
                "results_path": str(results_path), "device": device,
                "dry_run": False, "executed": False}
    target = Path(manifest_path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(built["manifest"], indent=1, sort_keys=True) + "\n",
                      encoding="utf-8")
    module = _manifest_module()
    doc, digest, path = module.load_manifest(target)
    plan = module.build_plan(doc)
    code = module.run_manifest(plan, path, digest, run_id, device, results_path,
                               allow_overwrite=allow_overwrite)
    return {"status": "completed" if code == 0 else "failed",
            "returncode": code, "manifest_path": str(target),
            "results_path": str(results_path), "device": device,
            "dry_run": False, "executed": True}


def attention_results_to_campaign(attention_results: Mapping[str, Any],
                                  requests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Convert accepted three-clock attention results to campaign results."""
    by_id = {str(request.get("attention_case", {}).get("case_id", "")): request
             for request in requests}
    rows = []
    for case in attention_results.get("cases", ()):
        case_id = str(case.get("case_id"))
        request = by_id.get(case_id, {})
        condition_key = str(request.get("condition_key") or f"attn:{case_id}")
        trials = []
        all_device, all_cpu, all_wall = [], [], []
        for trial in case.get("trials", ()):
            samples = list(trial.get("samples") or ())
            device = [float(s["device_interval_ms"]) for s in samples
                      if s.get("device_interval_ms") is not None]
            cpu = [float(s["cpu_submit_ms"]) for s in samples
                   if s.get("cpu_submit_ms") is not None]
            wall = [float(s["wall_complete_ms"]) for s in samples
                    if s.get("wall_complete_ms") is not None]
            trial_median = float(statistics.median(device)) if device else None
            trials.append({
                "trial_index": trial.get("trial_index"),
                "status": trial.get("status"),
                "median_ms": trial_median,
                "raw_samples_ms": device,
                "cpu_submit_samples_ms": cpu,
                "wall_complete_samples_ms": wall,
                "warmup": trial.get("warmup"),
                "events_reused": all(bool(s.get("events_reused")) for s in samples) if samples else None,
                "cpu_submit_scope": samples[0].get("cpu_submit_scope") if samples else None,
                "auxiliary_lifetime": samples[0].get("auxiliary_lifetime") if samples else None,
            })
            all_device.extend(device); all_cpu.extend(cpu); all_wall.extend(wall)
        measured = [t for t in trials if t["status"] == "measured" and t["median_ms"] is not None]
        duration = float(statistics.median(all_device)) if all_device else None
        status = "completed" if case.get("status") == "measured" else str(case.get("status", "failed"))
        protocol = dict(request.get("protocol") or {})
        protocol.update({
            **CAMPAIGN_ATTENTION_PROTOCOL,
            "warmup": int(attention_results.get("protocol", {}).get("warmup_iterations", 10)),
            "warmup_iterations_per_trial": int(
                attention_results.get("protocol", {}).get("warmup_iterations", 10)),
            "iterations": int(attention_results.get("protocol", {}).get("samples_per_trial", 30)),
            "trials": int(attention_results.get("protocol", {}).get("trials", 2)),
            "preallocated": False,
            "cache_policy": "one_attention_stream_reused_events_allocating_native_outputs",
        })
        rows.append({
            "measurement_id": request.get("measurement_id") or case_id,
            "request_key": request.get("request_key") or case_id,
            "condition_key": condition_key, "status": status,
            "source": "a3_operator_worker_attention",
            "source_class": "measured" if status == "completed" else "not_measured",
            "run_id": str(attention_results.get("run_id") or ""),
            "split": str(request.get("split", case.get("split", "train"))),
            "execution_kind": "fused_kernel", "worker_recipe": request.get("worker_recipe"),
            "backend": request.get("backend"), "target_match": False,
            "duration_ms": duration if status == "completed" else None,
            "raw_samples_ms": all_device if status == "completed" else [],
            "trials": trials, "protocol": protocol,
            "observed_context": {
                "observed": status == "completed",
                "observed_api": (case.get("native") or {}).get("observed_api"),
                "native": case.get("native"), "allocation": case.get("allocation"),
                "clocks": {"cpu_submit_ms": all_cpu, "device_interval_ms": all_device,
                           "wall_complete_ms": all_wall},
                "interval_scope": CAMPAIGN_ATTENTION_PROTOCOL["interval_scope"],
                "inclusive_host_gap": True,
                "native_output_allocation": True,
                "output_lifetime": (case.get("native") or {}).get(
                    "retained_policy", "last_output_only"),
                "not_pure_device_service": True,
            },
            "correctness": case.get("correctness"), "error": case.get("error"),
            "details": case.get("details") or case.get("geometry"),
            "analytic_ms": None, "environment": attention_results.get("environment"),
        })
    return {
        "run_id": attention_results.get("run_id"),
        "source_class": "measured",
        "target_match": False,
        "not_a3_measured": False,
        "results": rows,
        "cases": rows,
        "summary": {
            "cases": len(rows),
            "completed": sum(1 for row in rows if row["status"] == "completed"),
            "failed": sum(1 for row in rows if row["status"] == "failed"),
            "pending": sum(1 for row in rows if row["status"] == "pending"),
            "raw_device_samples": sum(len(row["raw_samples_ms"]) for row in rows),
            "interval_scope": CAMPAIGN_ATTENTION_PROTOCOL["interval_scope"],
            "not_pure_device_service": True,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Campaign/native attention adapter")
    sub = parser.add_subparsers(dest="command", required=True)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--requests", required=True)
    dry.add_argument("--template", required=True)
    dry.add_argument("--manifest-id", default="a3-campaign-attention")
    conv = sub.add_parser("convert-results")
    conv.add_argument("--attention-results", required=True)
    conv.add_argument("--requests", required=True)
    conv.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "dry-run":
        requests = json.loads(Path(args.requests).expanduser().read_text(encoding="utf-8"))
        template = json.loads(Path(args.template).expanduser().read_text(encoding="utf-8"))
        payload = dry_run_campaign_attention(requests, template, manifest_id=args.manifest_id)
        print(json.dumps(payload, indent=1, sort_keys=True))
        return 0 if payload["valid"] else 1
    results = json.loads(Path(args.attention_results).expanduser().read_text(encoding="utf-8"))
    requests = json.loads(Path(args.requests).expanduser().read_text(encoding="utf-8"))
    payload = attention_results_to_campaign(results, requests)
    Path(args.output).expanduser().write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "output": args.output,
                      "summary": payload["summary"]}, indent=1))
    return 0


__all__ = [
    "CAMPAIGN_ATTENTION_PROTOCOL",
    "campaign_requests_from_attention", "attention_query_from_request",
    "attention_manifest_from_campaign",
    "dry_run_campaign_attention", "run_campaign_attention",
    "attention_results_to_campaign", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
