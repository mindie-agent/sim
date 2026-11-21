"""Training-only primitive and resource service regression.

[Role]
  Aggregate repeated trials and estimate startup and steady-state response.
[Invariants]
  Training trials identify parameters. Heldout trials only evaluate them.
  Mixed or whole-step timing never identifies an independent resource rate.
[Timeouts & Locks]
  In-memory calculation. No I/O or locks.
[Side-Effect Policy]
  Returns fitted models and separate heldout estimates.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import json
import math
from statistics import median

from mindie_sim.observations.resource_services import (
    normalize_resource_service,
    resolve_nominal_rate,
)
from mindie_sim.prediction.services import ServiceCurve


_FORBIDDEN_SERVICE_BOUNDARIES = frozenset({
    "step", "body", "request", "serving", "e2e", "engine_step", "body_step",
    "whole_step", "whole_body",
})
_TARGET_KEYS = frozenset({
    "target_ms", "target_us", "target_duration_ms", "target_duration_us",
    "body_residual_ms", "step_residual_ms", "residual_ms",
})
_FORBIDDEN_WINDOWS = frozenset({
    "step", "body", "serving", "e2e", "engine_step", "body_step",
    "whole_step", "whole_body", "request",
})
_JOINT_ONLY_WINDOWS = frozenset({"joint", "mixed", "overlapping"})
_FIXED_TERM_TOLERANCE_US = 1e-9


def nonnegative_affine(points):
    """Identify startup + work/rate, constraining startup to be nonnegative."""
    n = len(points)
    if n < 2 or len({x for x, _ in points}) < 2:
        raise ValueError("rate_fit_requires_training_trials_at_distinct_work_sizes")
    if any(not math.isfinite(v) or v <= 0 for point in points for v in point):
        raise ValueError("primitive_fit_requires_positive_finite_work_and_duration")
    x = sum(p[0] for p in points) / n
    y = sum(p[1] for p in points) / n
    variance = sum((p[0] - x) ** 2 for p in points)
    slope = sum((a - x) * (b - y) for a, b in points) / variance
    intercept = y - slope * x
    if intercept < 0:
        intercept = 0
        slope = sum(a * b for a, b in points) / sum(a ** 2 for a, _ in points)
    if slope <= 0:
        raise ValueError("observations_do_not_identify_a_positive_service_rate")
    return intercept, 1 / slope


def fit_service_curve(packet):
    groups = defaultdict(lambda: defaultdict(list))
    repeats = {}
    training_sources = set()
    heldout = []
    for sample in packet["samples"]:
        work, duration = sample["work"], sample["duration_ms"]
        if not all(math.isfinite(v) and v > 0 for v in (work, duration)):
            raise ValueError("primitive_fit_requires_positive_finite_work_and_duration")
        key = sample["run_id"], sample["repeat_index"], work
        record = duration, sample["split"]
        if key in repeats:
            if repeats[key] != record:
                raise ValueError("conflicting_primitive_repeat_or_split")
            continue
        repeats[key] = record
        if sample["split"] == "train":
            groups[work][sample["run_id"]].append(duration)
            training_sources.add(sample["source"])
        elif sample["split"] == "heldout":
            heldout.append(sample)
        else:
            raise ValueError("service_sample_split_requires_train_or_heldout")
    knots = tuple((work, median(median(values) for values in runs.values()))
                  for work, runs in sorted(groups.items()))
    startup, rate = nonnegative_affine(knots)
    # A multi-regime ladder (small OneShot plateau + asymptotic large-payload
    # slope) can put the affine intercept above the smallest measured duration.
    # The curve startup must remain below every served duration, so cap it at
    # the tightest per-knot residual. The affine reference keeps the raw fit.
    curve_startup = startup
    if curve_startup >= min(t for _, t in knots):
        curve_startup = max(0.0, min(t - x / rate for x, t in knots))
    model = ServiceCurve(
        packet["resource"], packet["work_unit"], packet["identity"], packet["boundary"],
        packet["active_units"], curve_startup, knots, {
            **packet.get("provenance", {}),
            "training_sources": sorted(training_sources),
            "training_trials": sum(len(runs) for runs in groups.values()),
            "training_samples": sum(len(values) for runs in groups.values() for values in runs.values()),
            "aggregation": "median_of_run_medians_at_each_work",
            "startup_estimator": "nonnegative_affine_on_training_knots",
        })
    rows = []
    for sample in heldout:
        estimate = model.estimate(sample["work"])
        predicted = estimate.duration_ms if estimate else None
        rows.append({
            "work": sample["work"], "observed_ms": sample["duration_ms"],
            "predicted_ms": predicted, "source": sample["source"], "run_id": sample["run_id"],
            "repeat_index": sample["repeat_index"],
            "basis": estimate.basis if estimate else "outside_domain",
            "relative_error": predicted / sample["duration_ms"] - 1 if estimate else None,
        })
    return {"model": model.to_dict(), "model_id": model.model_id,
            "affine_reference": {"startup_ms": startup, "units_per_ms": rate},
            "heldout": rows}


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"resource_service_fit_{name}_requires_number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"resource_service_fit_{name}_requires_positive_finite")
    return result


def _nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"resource_service_fit_{name}_requires_number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"resource_service_fit_{name}_requires_nonnegative_finite")
    return result


def _packet_resources(packet):
    raw = packet.get("resources")
    if raw is None:
        raw = [packet["resource"]] if packet.get("resource") else None
    elif isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("resource_service_fit_requires_a_resource_list")
    resources = tuple(str(item).strip() for item in raw)
    if any(not item for item in resources) or len(set(resources)) != len(resources):
        raise ValueError("resource_service_fit_requires_distinct_resources")
    return resources


def _sample_record(sample):
    if not isinstance(sample, Mapping):
        raise ValueError("resource_service_fit_sample_requires_mapping")
    for key in _TARGET_KEYS:
        if sample.get(key) is not None:
            raise ValueError("resource_service_fit_rejects_target_duration:" + key)
    work = _positive(sample.get("work"), "work")
    has_us = sample.get("duration_us") is not None
    has_ms = sample.get("duration_ms") is not None
    if has_us and has_ms:
        raise ValueError("resource_service_fit_sample_requires_one_duration_unit")
    if has_us:
        duration_us = _positive(sample["duration_us"], "duration_us")
    elif has_ms:
        duration_us = _positive(sample["duration_ms"], "duration_ms") * 1000
    else:
        raise ValueError("resource_service_fit_sample_requires_duration")
    run_id = sample.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("resource_service_fit_sample_requires_run_id")
    repeat_index = sample.get("repeat_index", 0)
    if isinstance(repeat_index, bool) or not isinstance(repeat_index, int):
        raise ValueError("resource_service_fit_sample_repeat_index_requires_integer")
    if repeat_index < 0:
        raise ValueError("resource_service_fit_sample_repeat_index_requires_nonnegative")
    split = sample.get("split")
    if split not in {"train", "heldout"}:
        raise ValueError("resource_service_fit_sample_split_requires_train_or_heldout")
    source = sample.get("source")
    if source is not None and not isinstance(source, str):
        raise ValueError("resource_service_fit_sample_source_requires_text")
    return {
        "work": work,
        "duration_us": duration_us,
        "run_id": run_id.strip(),
        "repeat_index": repeat_index,
        "split": split,
        "source": source or run_id.strip(),
    }


def _training_knots(samples):
    groups = defaultdict(lambda: defaultdict(list))
    repeats = {}
    training_sources = set()
    training_runs = set()
    heldout = []
    training_samples = 0
    for sample in samples:
        key = sample["run_id"], sample["repeat_index"], sample["work"]
        record = sample["duration_us"], sample["split"]
        if key in repeats:
            if repeats[key] != record:
                raise ValueError("conflicting_resource_service_repeat_or_split")
            continue
        repeats[key] = record
        if sample["split"] == "train":
            groups[sample["work"]][sample["run_id"]].append(sample["duration_us"])
            training_sources.add(sample["source"])
            training_runs.add(sample["run_id"])
            training_samples += 1
        else:
            heldout.append(sample)
    if not groups:
        raise ValueError("resource_service_fit_requires_training_samples")
    knots = tuple(
        (work, median(median(values) for values in runs.values()))
        for work, runs in sorted(groups.items())
    )
    run_work_cells = sum(len(runs) for runs in groups.values())
    return {
        "knots": knots,
        "heldout": heldout,
        "training_sources": sorted(training_sources),
        "training_runs": len(training_runs),
        "training_run_work_cells": run_work_cells,
        "training_samples": training_samples,
    }


def _cap_fixed(fixed_us, rate_per_us, knots):
    if fixed_us < 0:
        fixed_us = 0.0
    capped = fixed_us >= min(duration for _, duration in knots)
    if capped:
        fixed_us = max(0.0, min(duration - work / rate_per_us
                               for work, duration in knots))
    return fixed_us, capped


def _declared_fixed(knots, effective_rate):
    candidates = [duration - work * 1e6 / effective_rate
                  for work, duration in knots]
    fixed_us = median(candidates)
    if fixed_us < -_FIXED_TERM_TOLERANCE_US:
        raise ValueError(
            "resource_service_observations_contradict_declared_rate")
    return max(0.0, fixed_us)


def fit_resource_service(packet, *, nominal_rates=None, hardware=None):
    """Fit one fixed term plus steady-state rate from independent trials.

    A declared ``supply`` fixes the rate and calibrates only the fixed term from
    one or more work sizes.  Otherwise at least two distinct training work sizes
    are required to identify startup and rate.  A joint/mixed window is fitted
    only as a joint-phase service; it never emits an internal resource rate.
    Sample ``work`` uses the declared ``rate_scope`` (aggregate or per-owner);
    it is not silently converted between the two scopes.
    """
    if not isinstance(packet, Mapping):
        raise ValueError("resource_service_fit_requires_mapping")
    for key in _TARGET_KEYS:
        if packet.get(key) is not None:
            raise ValueError("resource_service_fit_rejects_target_duration:" + key)
    samples = packet.get("samples")
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("resource_service_fit_requires_samples")
    records = [_sample_record(sample) for sample in samples]

    scope = str(packet.get("scope") or "").strip()
    if scope not in {"independent_resource", "joint_phase"}:
        raise ValueError("resource_service_fit_scope_unknown")
    resources = _packet_resources(packet)
    if scope == "independent_resource" and len(resources) != 1:
        raise ValueError(
            "resource_service_mixed_window_cannot_identify_independent_resource")
    phase = str(packet.get("phase") or "").strip()
    if not phase:
        raise ValueError("resource_service_fit_requires_phase")
    work_unit = str(packet.get("work_unit") or "").strip()
    if not work_unit:
        raise ValueError("resource_service_fit_requires_work_unit")
    boundary = str(packet.get("boundary") or "").strip()
    if boundary in _FORBIDDEN_SERVICE_BOUNDARIES:
        raise ValueError(
            "resource_service_fit_boundary_must_be_resource_phase_or_device_kernel")
    if boundary not in {"resource_phase", "device_kernel"}:
        raise ValueError(
            "resource_service_fit_boundary_must_be_resource_phase_or_device_kernel")

    provenance = packet.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        raise ValueError("resource_service_fit_provenance_requires_mapping")
    provenance = dict(provenance)
    for key in ("window", "observation_kind", "boundary_kind", "scope"):
        declared = str(provenance.get(key, "")).strip().lower()
        if declared in _FORBIDDEN_WINDOWS:
            raise ValueError(
                "resource_service_fit_rejects_step_or_body_window:" + key)
        if scope == "independent_resource" and declared in _JOINT_ONLY_WINDOWS:
            raise ValueError(
                "resource_service_mixed_window_cannot_identify_independent_resource:"
                + key)

    conditions = packet.get("conditions") or {}
    if not isinstance(conditions, Mapping):
        raise ValueError("resource_service_fit_conditions_requires_mapping")
    conditions = dict(conditions)
    if hardware:
        declared_hardware = conditions.get("hardware")
        if declared_hardware and str(declared_hardware) != str(hardware):
            raise ValueError("resource_service_hardware_mismatch")
        conditions["hardware"] = str(hardware)

    active_units = packet.get("active_units", 1)
    if isinstance(active_units, bool) or not isinstance(active_units, int) \
            or active_units < 1:
        raise ValueError("resource_service_fit_active_units_requires_positive_integer")
    rate_scope = str(packet.get("rate_scope", "aggregate")).strip()
    if rate_scope not in {"aggregate", "per_owner"}:
        raise ValueError("resource_service_fit_rate_scope_unknown")

    grouped = _training_knots(records)
    knots = grouped["knots"]
    declared_supply = packet.get("supply")
    declared_response = _nonnegative(packet.get("response_us", 0), "response_us")
    declared_startup = packet.get("startup_us")
    if declared_startup is not None:
        declared_startup = _nonnegative(declared_startup, "startup_us")
    affine_reference = None
    startup_capped = False
    nominal_rate = None
    nominal_multiplier = None
    declared_supply_kind = None
    if declared_supply is not None:
        if declared_startup is not None:
            raise ValueError(
                "resource_service_fit_supply_and_startup_are_mutually_exclusive")
        if not isinstance(declared_supply, Mapping):
            raise ValueError("resource_service_fit_supply_requires_mapping")
        declared_supply_kind = str(declared_supply.get("kind") or "").strip()
        rate_record = {
            "id": str(packet.get("id") or packet.get("service_id")
                      or "resource-service"),
            "scope": scope,
            "resources": list(resources),
            "phase": phase,
            "work_unit": work_unit,
            "boundary": boundary,
            "startup_us": 0,
            "response_us": 0,
            "active_units": active_units,
            "rate_scope": rate_scope,
            "supply": dict(declared_supply),
            "conditions": conditions,
            "identification": {"status": "declared",
                               "resource_identified": False},
        }
        rate_service = normalize_resource_service(
            rate_record, nominal_rates=nominal_rates, hardware=hardware)
        effective_rate = rate_service.effective_rate
        nominal_rate = rate_service.nominal_rate
        nominal_multiplier = rate_service.nominal_multiplier
        fixed_us = _declared_fixed(knots, effective_rate)
        rate_source = "declared"
        response_us = declared_response
        if response_us > fixed_us + 1e-9:
            raise ValueError(
                "resource_service_fit_response_exceeds_observed_fixed_term")
        startup_us = max(0.0, fixed_us - response_us)
        separable = bool(packet.get("startup_response_separable",
                                    response_us > 0))
    elif declared_startup is not None:
        fixed_us = declared_startup + declared_response
        service_work = [(work, duration - fixed_us)
                        for work, duration in knots]
        if any(service_us <= 0 for _, service_us in service_work):
            raise ValueError(
                "resource_service_observations_contradict_declared_startup")
        slope_us_per_work = (
            sum(work * service_us for work, service_us in service_work)
            / sum(work * work for work, _ in service_work))
        if slope_us_per_work <= 0:
            raise ValueError(
                "observations_do_not_identify_a_positive_service_rate")
        rate_per_us = 1 / slope_us_per_work
        effective_rate = rate_per_us * 1e6
        if nominal_rates is not None and len(resources) == 1:
            nominal_rate = resolve_nominal_rate(
                resources[0], nominal_rates, work_unit=work_unit,
                hardware=hardware, rate_scope=rate_scope)
        nominal_multiplier = (effective_rate / nominal_rate
                              if nominal_rate is not None else None)
        affine_reference = {
            "fixed_us": fixed_us,
            "fixed_source": "declared",
            "rate_per_us": rate_per_us,
            "rate_per_s": effective_rate,
        }
        rate_source = "fitted_with_declared_fixed"
        startup_us = declared_startup
        response_us = declared_response
        separable = True
    else:
        if len(knots) < 2:
            raise ValueError(
                "resource_service_fit_requires_two_work_sizes_or_a_declared_rate")
        affine_fixed, rate_per_us = nonnegative_affine(knots)
        effective_rate = rate_per_us * 1e6
        fixed_us, startup_capped = _cap_fixed(affine_fixed, rate_per_us, knots)
        if nominal_rates is not None and len(resources) == 1:
            nominal_rate = resolve_nominal_rate(
                resources[0], nominal_rates, work_unit=work_unit,
                hardware=hardware, rate_scope=rate_scope)
        nominal_multiplier = (effective_rate / nominal_rate
                              if nominal_rate is not None else None)
        affine_reference = {
            "fixed_us": affine_fixed,
            "rate_per_us": rate_per_us,
            "rate_per_s": effective_rate,
        }
        rate_source = "fitted"
        response_us = declared_response
        if response_us > fixed_us + 1e-9:
            raise ValueError(
                "resource_service_fit_response_exceeds_observed_fixed_term")
        startup_us = max(0.0, fixed_us - response_us)
        separable = bool(packet.get("startup_response_separable",
                                    response_us > 0))

    identification = packet.get("identification") or {}
    if not isinstance(identification, Mapping):
        raise ValueError("resource_service_fit_identification_requires_mapping")
    identification = dict(identification)
    fitted_rate = rate_source in {"fitted", "fitted_with_declared_fixed"}
    if "resource_identified" in identification:
        resource_identified = bool(identification["resource_identified"])
        if resource_identified and not fitted_rate:
            raise ValueError(
                "resource_service_declared_rate_cannot_be_reported_as_identified")
    else:
        resource_identified = (scope == "independent_resource" and fitted_rate)
    if scope == "joint_phase" and resource_identified:
        raise ValueError("joint_service_cannot_identify_an_internal_resource")
    status = identification.get("status")
    if status is None:
        if scope == "joint_phase":
            status = "joint_identified"
        elif declared_supply_kind == "nominal":
            status = "nominal_specification"
        elif resource_identified:
            status = "identified"
        elif rate_source == "declared":
            status = "partially_identified"
        else:
            status = "declared_effective_service"
    train_work = [work for work, _ in knots]
    heldout_work = sorted({sample["work"] for sample in grouped["heldout"]})
    fixed_source = "declared" if declared_startup is not None else "fitted"
    if declared_startup is not None:
        included_fixed = (
            "startup and response are declared separately; no fixed term is "
            "fitted from the observation")
    elif separable:
        included_fixed = (
            "declared response is separated; the remaining observed fixed term "
            "may include startup, occupancy or tail effects and is counted once")
    else:
        included_fixed = (
            "startup, occupancy and tail effects are combined in the observed "
            "fixed term and are counted once")
    if scope == "joint_phase":
        parameter_class = "joint_phase_service"
    elif declared_supply_kind == "nominal":
        parameter_class = "nominal_specification"
    else:
        parameter_class = "effective_resource_service"
    if scope == "joint_phase":
        rate_interpretation = (
            "joint_phase_service_rate_not_an_isolated_internal_resource_rate")
    elif declared_supply_kind == "nominal":
        rate_interpretation = (
            "declared_nominal_rate_with_observed_fixed_term_not_a_calibrated_service")
    elif resource_identified:
        rate_interpretation = "identified_effective_resource_service"
    else:
        rate_interpretation = (
            "declared_effective_resource_service_not_identified_as_isolated_peak")
    fit_report = {
        "rate_source": rate_source,
        "declared_supply_kind": declared_supply_kind,
        "fixed_source": fixed_source,
        "fixed_us": fixed_us,
        "startup_us": startup_us,
        "response_us": response_us,
        "effective_rate": effective_rate,
        "rate_scope": rate_scope,
        "nominal_rate": nominal_rate,
        "nominal_multiplier": nominal_multiplier,
        "affine_reference": affine_reference,
        "startup_capped": startup_capped,
        "training_work": train_work,
        "training_runs": grouped["training_runs"],
        "training_run_work_cells": grouped["training_run_work_cells"],
        "training_samples": grouped["training_samples"],
        "aggregation": "median_of_run_medians_at_each_work",
        "included_fixed_effects": included_fixed,
        "residual_us": {
            "max_abs": max(abs(duration - (fixed_us + work * 1e6 / effective_rate))
                           for work, duration in knots),
            "mean_abs": sum(abs(duration - (fixed_us + work * 1e6 / effective_rate))
                            for work, duration in knots) / len(knots),
        },
    }
    if declared_supply is not None:
        service_supply = dict(declared_supply)
        if nominal_rate is not None and service_supply.get("nominal_rate") is None:
            service_supply["nominal_rate"] = nominal_rate
    else:
        service_supply = {
            "kind": "absolute_rate",
            "rate": effective_rate,
            **({"nominal_rate": nominal_rate} if nominal_rate is not None else {}),
        }
    service_record = {
        "id": str(packet.get("id") or packet.get("service_id")
                  or "resource-service"),
        "scope": scope,
        "resources": list(resources),
        "phase": phase,
        "work_unit": work_unit,
        "boundary": boundary,
        "startup_us": startup_us,
        "response_us": response_us,
        "active_units": active_units,
        "rate_scope": rate_scope,
        "supply": service_supply,
        "conditions": conditions,
        "identification": {
            **identification,
            "status": status,
            "resource_identified": resource_identified,
            "rate_source": rate_source,
            "fixed_identified": declared_startup is None,
            "fixed_source": fixed_source,
            "startup_response_separable": separable,
            "fixed_components": (
                "declared_startup_and_response"
                if declared_startup is not None else
                "startup_plus_declared_response"
                if separable else "combined_fixed"),
            "train_work": train_work,
            "heldout_work": heldout_work,
            "training_work_range": [min(train_work), max(train_work)],
            "formula_domain": "all_nonnegative_work",
            "concurrency_scope": identification.get(
                "concurrency_scope", "isolated_invocation"),
            "included_fixed_effects": included_fixed,
        },
        "provenance": {
            **provenance,
            "fit": {**dict(provenance.get("fit") or {}), **fit_report},
            "training_sources": grouped["training_sources"],
            "parameter_class": parameter_class,
            **({"rate_interpretation": rate_interpretation}
               if "rate_interpretation" not in provenance else {}),
            "normalized_rate_interpretation": rate_interpretation,
            "concurrency": "not_fitted; one invocation boundary",
            "included_effects": included_fixed,
        },
    }
    normalized = normalize_resource_service(service_record, hardware=hardware)
    heldout_rows = []
    for sample in grouped["heldout"]:
        prediction = normalized.prediction(sample["work"])
        predicted_us = prediction["duration_us"]
        heldout_rows.append({
            "work": sample["work"],
            "observed_us": sample["duration_us"],
            "predicted_us": predicted_us,
            "relative_error": predicted_us / sample["duration_us"] - 1,
            "source": sample["source"],
            "run_id": sample["run_id"],
            "repeat_index": sample["repeat_index"],
            "basis": prediction["basis"],
            "inside_training_range": prediction["inside_training_range"],
            "extrapolated": prediction["extrapolated"],
        })
    return {
        "schema": "mindie-sim.resource-service-fit/1",
        "service": normalized.to_dict(),
        "model_id": normalized.model_id,
        "formula": normalized.formula,
        "fit": fit_report,
        "heldout": heldout_rows,
        "normalization": normalized.normalization,
        "source": (provenance.get("source") or provenance.get("sources")
                   or packet.get("source")),
    }
