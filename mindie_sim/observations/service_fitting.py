"""Train primitive service curves and normalized resource services.

[Role] Aggregate repeated primitive trials and fit reusable work-response
  models, including one normalized resource/phase rate with a fixed term.
[Invariants] Only training trials supply knots/startup/rate. One packet fixes
  the hardware, implementation, layout, units and measurement boundary. A
  mixed or joint window is never promoted to an independent resource rate; a
  whole-step/body residual is rejected. Startup/response and rate are priced
  once, and new work sizes evaluate the same formula without a per-shape table.
[Timeouts & Locks] Bounded explicit metadata input; no subprocesses or locks.
[Side-Effect Policy] Returns a model and separate validation; never runs a device.
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
    if fixed_us < -1e-9:
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


_ROLE_IN_SAMPLE = "in_sample"
_ROLE_FROZEN_TRANSFER = "frozen_training_function_transfer_to_new_capture"
_ROLE_UNDECLARED = "not_declared"
_ROLE_ALIASES = {
    _ROLE_IN_SAMPLE: _ROLE_IN_SAMPLE,
    _ROLE_FROZEN_TRANSFER: "frozen_transfer",
    _ROLE_UNDECLARED: "not_declared",
}
_ROLE_SCOPES = frozenset({"independent_resource", "joint_phase"})
_PHASE_WINDOWS = frozenset({"resource_phase", "resource", "phase"})
_JOINT_ROLE_WINDOWS = frozenset({"joint", "mixed", "overlapping", "joint_phase"})
_FORBIDDEN_ROLE_WINDOWS = frozenset({
    "step", "body", "request", "serving", "e2e", "engine_step", "body_step",
    "whole_step", "whole_body",
})
_ROLE_NEEDED_EVIDENCE = (
    "scope", "resources", "phase", "work", "work_unit",
    "resource_duration_us", "owner_units", "boundary",
)
_ROLE_SERVICE_BOUNDARIES = frozenset({"resource_phase", "device_kernel"})


def load_role_ledger(path):
    """Read a JSONL cohort ledger without opening any referenced capture."""
    import json
    from pathlib import Path
    records = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))
    return tuple(records)


def calibration_role_partitions(records):
    """Partition historical cohorts by their declared role, without relabeling.

    ``in_sample`` alone is eligible for parameter fitting.  Frozen transfer
    cohorts remain independent transfer evidence; ``not_declared`` rows are
    neither training nor a holdout.
    """
    if not isinstance(records, (list, tuple)):
        raise ValueError("calibration_role_records_require_sequence")
    parts = {"in_sample": [], "frozen_transfer": [], "not_declared": []}
    seen = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("calibration_role_record_requires_mapping")
        cohort_id = str(record.get("cohort_id") or "").strip()
        if not cohort_id:
            raise ValueError("calibration_role_record_requires_cohort_id")
        if cohort_id in seen:
            raise ValueError("calibration_role_record_duplicate_cohort_id")
        seen.add(cohort_id)
        role = str(record.get("role") or "").strip()
        if role not in _ROLE_ALIASES:
            raise ValueError("calibration_role_unknown:" + role)
        parts[_ROLE_ALIASES[role]].append(record)
    counts = {key: len(value) for key, value in parts.items()}
    return {
        "in_sample": tuple(parts["in_sample"]),
        "frozen_transfer": tuple(parts["frozen_transfer"]),
        "not_declared": tuple(parts["not_declared"]),
        "counts": counts,
        "total": sum(counts.values()),
        "policy": ("in_sample_only_for_fit; frozen_transfer_independent; "
                   "not_declared_neither_training_nor_holdout"),
    }


def _canonical(value):
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _role_conditions(record, *, conditions, hardware):
    """Merge record-level and call-level conditions, rejecting conflicts."""
    raw = record.get("conditions")
    if raw is None:
        declared = {}
    elif not isinstance(raw, Mapping):
        raise ValueError("resource_service_role_record_conditions_requires_mapping")
    else:
        declared = dict(raw)
    merged = dict(declared)
    if hardware:
        declared_hardware = declared.get("hardware")
        if declared_hardware and str(declared_hardware) != str(hardware):
            raise ValueError("resource_service_role_record_hardware_mismatch")
        merged["hardware"] = str(hardware)
    if conditions is not None:
        if not isinstance(conditions, Mapping):
            raise ValueError("resource_service_role_record_conditions_requires_mapping")
        for path, expected in conditions.items():
            if path in declared and _canonical(declared[path]) != _canonical(expected):
                raise ValueError(
                    "resource_service_role_record_condition_conflict:" + str(path))
            merged[path] = expected
    return declared, merged


def _role_resources(record):
    raw = record.get("resources")
    if raw is None and record.get("resource") is not None:
        raw = [record["resource"]]
    elif isinstance(raw, str):
        raw = [raw]
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("resource_service_role_record_resources_invalid")
    values = tuple(str(item).strip() for item in raw)
    if any(not item for item in values) or len(set(values)) != len(values):
        raise ValueError("resource_service_role_record_resources_invalid")
    return values


def _role_contract(record, *, resource, phase, work_unit, boundary,
                   rate_scope, active_units, conditions, hardware, scope):
    """Extract one source observation contract exactly as declared.

    ``state`` is one of ``candidate``, ``joint``, ``excluded`` or ``gap``.
    The caller parameter is a selector and a source-level declaration; it is
    never silently substituted for a missing per-record contract field.
    """
    cohort_id = str(record["cohort_id"])
    declared = {
        "cohort_id": cohort_id,
        "case_id": record.get("case_id"),
        "source_path": (record.get("source_path") or record.get("source")),
        "locator": (record.get("locator") or record.get("example_locator")
                    or record.get("capture_locator")),
        "scope": record.get("scope"),
        "resources": _role_resources(record),
        "resource": record.get("resource"),
        "phase": record.get("phase"),
        "work_unit": record.get("work_unit"),
        "work": record.get("work"),
        "duration_us": record.get("duration_us"),
        "resource_duration_us": record.get("resource_duration_us"),
        "boundary": record.get("boundary"),
        "rate_scope": record.get("rate_scope"),
        "active_units": (record.get("active_units")
                         if record.get("active_units") is not None else
                         record.get("owner_units")
                         if record.get("owner_units") is not None else
                         record.get("owners")),
        "conditions": record.get("conditions") or {},
        "provenance": record.get("provenance") or {},
        "window": (record.get("window")
                   or (record.get("provenance") or {}).get("window")),
    }
    base = {
        "cohort_id": cohort_id,
        "case_id": record.get("case_id"),
        "source_path": declared["source_path"],
        "declared": declared,
    }
    scope_decl = str(record.get("scope") or "").strip()
    missing = []
    if scope_decl not in _ROLE_SCOPES:
        missing.append("scope")
        scope_decl = None
    resources = declared["resources"]
    if resources is None:
        missing.append("resources")
    elif scope == "independent_resource" and resource not in resources:
        return {**base, "state": "excluded", "reason": "resource_not_selected"}
    phase_decl = str(record.get("phase") or "").strip()
    if not phase_decl:
        missing.append("phase")
    elif phase_decl != phase:
        return {**base, "state": "excluded", "reason": "phase_not_selected"}
    unit_decl = str(record.get("work_unit") or "").strip()
    if not unit_decl:
        missing.append("work_unit")
    elif unit_decl != work_unit:
        raise ValueError(
            "resource_service_role_record_work_unit_mismatch:"
            + unit_decl + "!=" + work_unit)
    work = record.get("work")
    if work is None:
        missing.append("work")
    provenance = record.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        raise ValueError("resource_service_role_record_provenance_requires_mapping")
    provenance = dict(provenance)
    window = (provenance.get("window") or provenance.get("observation_kind")
              or record.get("window"))
    if window is not None:
        window = str(window).strip().lower()
    if window in _FORBIDDEN_ROLE_WINDOWS:
        return {**base, "state": "gap", "reason":
                "whole_step_or_body_window_is_not_a_service_phase",
                "missing": ["resource_phase_window"]}
    declared_duration = record.get("resource_duration_us")
    duration_field = None
    if declared_duration is not None:
        duration_selector = declared_duration
        duration_field = "resource_duration_us"
    elif record.get("duration_us") is not None:
        if scope_decl == "joint_phase":
            duration_selector = record["duration_us"]
            duration_field = "duration_us_joint_window"
        elif scope_decl == "independent_resource" and window in _PHASE_WINDOWS:
            duration_selector = record["duration_us"]
            duration_field = "duration_us_resource_phase_alias"
        else:
            duration_selector = None
            missing.append("resource_duration_us")
    else:
        duration_selector = None
        missing.append("resource_duration_us")
    if scope_decl == "independent_resource" and window in _JOINT_ROLE_WINDOWS:
        return {**base, "state": "gap",
                "reason": "mixed_window_declared_as_independent_resource",
                "missing": ["independent_resource_window"]}
    declared_boundary = str(record.get("boundary") or "").strip()
    if declared_boundary and declared_boundary not in _ROLE_SERVICE_BOUNDARIES:
        return {**base, "state": "gap", "reason": "boundary_invalid",
                "missing": ["boundary"]}
    if boundary is not None:
        if declared_boundary and declared_boundary != boundary:
            raise ValueError(
                "resource_service_role_record_boundary_conflict:"
                + declared_boundary + "!=" + boundary)
        selected_boundary = boundary
    else:
        selected_boundary = declared_boundary or "resource_phase"
    record_rate_scope = record.get("rate_scope")
    if record_rate_scope is not None:
        record_rate_scope = str(record_rate_scope).strip()
        if record_rate_scope not in {"aggregate", "per_owner"}:
            return {**base, "state": "gap", "reason": "rate_scope_invalid",
                    "missing": ["rate_scope"]}
    if (rate_scope is not None and record_rate_scope is not None
            and record_rate_scope != rate_scope):
        raise ValueError(
            "resource_service_role_record_owner_scope_conflict:rate_scope")
    selected_rate_scope = record_rate_scope or rate_scope or "aggregate"
    record_units = (record.get("active_units")
                    if record.get("active_units") is not None else
                    record.get("owner_units")
                    if record.get("owner_units") is not None else
                    record.get("owners"))
    if record_units is not None:
        if isinstance(record_units, bool) or not isinstance(record_units, int) \
                or record_units < 1:
            return {**base, "state": "gap", "reason": "active_units_invalid",
                    "missing": ["active_units"]}
    if (active_units is not None and record_units is not None
            and int(record_units) != int(active_units)):
        raise ValueError(
            "resource_service_role_record_owner_scope_conflict:active_units")
    selected_units = int(record_units or active_units or 1)
    declared_conditions, merged_conditions = _role_conditions(
        record, conditions=conditions, hardware=hardware)
    if missing:
        return {**base, "state": "gap",
                "reason": "missing_explicit_resource_phase_work",
                "missing": missing}
    if resources is not None and scope_decl == "independent_resource" \
            and tuple(resources) != (resource,):
        return {**base, "state": "gap",
                "reason": "independent_resource_requires_exactly_the_selected_resource",
                "missing": ["resources"]}
    contract = {
        "cohort_id": cohort_id,
        "scope": scope_decl,
        "resources": tuple(resources),
        "phase": phase_decl,
        "work_unit": unit_decl,
        "work": work,
        "duration_us": duration_selector,
        "duration_field": duration_field,
        "boundary": selected_boundary,
        "rate_scope": selected_rate_scope,
        "active_units": selected_units,
        "conditions": merged_conditions,
        "declared_conditions": declared_conditions,
        "provenance": provenance,
        "window": window,
        "supply": record.get("supply", provenance.get("supply")),
        "startup_us": (record.get("startup_us")
                       if record.get("startup_us") is not None
                       else provenance.get("startup_us")),
        "source_path": declared["source_path"],
        "locator": declared["locator"],
        "case_id": record.get("case_id"),
    }
    state = "joint" if scope_decl == "joint_phase" else "candidate"
    return {**base, "state": state, "contract": contract, "declared": declared}


def _role_gap_row(extracted, *, reason=None, missing=None):
    return {
        "cohort_id": extracted["cohort_id"],
        "case_id": extracted.get("case_id"),
        "source_path": extracted.get("source_path"),
        "reason": reason or extracted.get("reason")
        or "missing_explicit_resource_phase_work",
        "missing": list(missing if missing is not None
                        else extracted.get("missing") or ()),
        "declared": extracted.get("declared"),
    }


def _role_joint_evidence(contract):
    return {
        "cohort_id": contract["cohort_id"],
        "case_id": contract.get("case_id"),
        "scope": contract["scope"],
        "resources": list(contract["resources"]),
        "phase": contract["phase"],
        "work_unit": contract["work_unit"],
        "work": contract["work"],
        "duration_us": contract["duration_us"],
        "duration_field": contract["duration_field"],
        "boundary": contract["boundary"],
        "rate_scope": contract["rate_scope"],
        "active_units": contract["active_units"],
        "conditions": contract["conditions"],
        "window": contract["window"],
        "provenance": contract["provenance"],
        "source_path": contract["source_path"],
        "locator": contract["locator"],
        "resource_identified": False,
    }


def _role_service_source(contract):
    return {
        "cohort_id": contract["cohort_id"],
        "case_id": contract.get("case_id"),
        "source_path": contract["source_path"],
        "locator": contract.get("locator"),
        "window": contract.get("window"),
        "duration_field": contract.get("duration_field"),
        "work_unit": contract.get("work_unit"),
        "rate_scope": contract.get("rate_scope"),
        "active_units": contract.get("active_units"),
    }


def fit_resource_service_from_role_records(
        records, *, resource, phase, work_unit, boundary=None,
        rate_scope=None, active_units=None, conditions=None, hardware=None,
        scope="independent_resource", supply=None, startup_us=None,
        response_us=0):
    """Fit one normalized service only from explicit in-sample source work.

    The caller's ``resource/phase/work_unit`` select one homogeneous source
    observation contract; they do not fill in a missing record field.  Each
    training record keeps its declared scope, resource set, phase, window,
    work unit, owner scope, boundary, conditions and provenance.  A joint or
    mixed window is retained as joint evidence and is never promoted to an
    independent resource rate.  ``duration_us`` aliases
    ``resource_duration_us`` only when the source declares the same resource
    phase window; otherwise the record stays underdetermined.

    Frozen transfer and not-declared records are preserved separately and
    never enter the fit.  The fit itself still runs through
    ``fit_resource_service -> normalize_resource_service``.
    """
    partition = calibration_role_partitions(records)
    resource = str(resource).strip()
    phase = str(phase).strip()
    work_unit = str(work_unit).strip()
    if not resource or not phase or not work_unit:
        raise ValueError("role_service_fit_requires_resource_phase_work_unit")
    scope = str(scope).strip()
    if scope not in _ROLE_SCOPES:
        raise ValueError("role_service_fit_scope_unknown")
    rate_scope = None if rate_scope is None else str(rate_scope).strip()
    if rate_scope is not None and rate_scope not in {"aggregate", "per_owner"}:
        raise ValueError("role_service_fit_rate_scope_unknown")
    boundary = None if boundary is None else str(boundary).strip()
    if boundary is not None and boundary not in _ROLE_SERVICE_BOUNDARIES:
        raise ValueError("role_service_fit_boundary_invalid")
    candidates = []
    joint = []
    underdetermined = []
    excluded = []
    participation = []
    for record in partition["in_sample"]:
        extracted = _role_contract(
            record, resource=resource, phase=phase, work_unit=work_unit,
            boundary=boundary, rate_scope=rate_scope,
            active_units=active_units, conditions=conditions,
            hardware=hardware, scope=scope)
        if extracted["state"] == "excluded":
            excluded.append({
                "cohort_id": extracted["cohort_id"],
                "case_id": extracted.get("case_id"),
                "source_path": extracted.get("source_path"),
                "reason": extracted["reason"],
                "declared": extracted.get("declared"),
            })
            participation.append({
                "cohort_id": extracted["cohort_id"],
                "case_id": extracted.get("case_id"),
                "status": "excluded",
                "reason": extracted["reason"],
            })
            continue
        if extracted["state"] == "joint":
            contract = extracted["contract"]
            joint.append(_role_joint_evidence(contract))
            if scope == "independent_resource":
                underdetermined.append(_role_gap_row(
                    extracted, reason="joint_window_not_independent_resource",
                    missing=["independent_resource_measurement"]))
                participation.append({
                    "cohort_id": extracted["cohort_id"],
                    "case_id": extracted.get("case_id"),
                    "status": "joint_evidence_only",
                    "reason": "joint_window_not_independent_resource",
                    "scope": "joint_phase",
                })
            else:
                candidates.append(contract)
                participation.append({
                    "cohort_id": extracted["cohort_id"],
                    "case_id": extracted.get("case_id"),
                    "status": "pending_training",
                    "scope": "joint_phase",
                })
            continue
        if extracted["state"] == "gap":
            underdetermined.append(_role_gap_row(extracted))
            participation.append({
                "cohort_id": extracted["cohort_id"],
                "case_id": extracted.get("case_id"),
                "status": "underdetermined",
                "reason": extracted.get("reason"),
                "missing": list(extracted.get("missing") or ()),
            })
            continue
        if scope == "joint_phase":
            underdetermined.append(_role_gap_row(
                extracted, reason="independent_resource_not_a_joint_phase",
                missing=["joint_phase_measurement"]))
            participation.append({
                "cohort_id": extracted["cohort_id"],
                "case_id": extracted.get("case_id"),
                "status": "excluded",
                "reason": "independent_resource_not_a_joint_phase",
            })
            continue
        candidates.append(extracted["contract"])
        participation.append({
            "cohort_id": extracted["cohort_id"],
            "case_id": extracted.get("case_id"),
            "status": "pending_training",
            "scope": extracted["contract"]["scope"],
        })

    base = {
        "schema": "mindie-sim.role-service-fit/2",
        "role_counts": partition["counts"],
        "total": partition["total"],
        "training_ids": [record["cohort_id"]
                         for record in partition["in_sample"]],
        "frozen_transfer_ids": [record["cohort_id"]
                                for record in partition["frozen_transfer"]],
        "not_declared_ids": [record["cohort_id"]
                            for record in partition["not_declared"]],
        "excluded": excluded,
        "underdetermined": underdetermined,
        "joint_evidence": joint,
        "training_participation": participation,
        "policy": partition["policy"],
        "scope": scope,
        "needed_evidence": list(_ROLE_NEEDED_EVIDENCE),
        "identifiable_combination": (
            "fixed_us + work/effective_rate; from a whole-call total only "
            "this linear combination is identifiable"),
    }
    def _finalize_participation(status, reason=None):
        for item in participation:
            if item["status"] == "pending_training":
                item["status"] = status
                if reason is not None:
                    item["reason"] = reason

    if not candidates:
        _finalize_participation("not_fitted")
        if scope == "independent_resource" and joint:
            reason = "joint_window_not_independent_resource"
        elif scope == "joint_phase" and not joint:
            reason = "no_in_sample_joint_phase_with_explicit_work"
        else:
            reason = "no_in_sample_record_with_explicit_resource_phase_work"
        return {**base, "status": "underdetermined", "reason": reason,
                "service": None}

    # The selected records must describe one physical packet.  A different
    # window/unit/owner/condition set is not silently averaged into it.
    for key in ("work_unit", "boundary", "rate_scope", "active_units",
                "conditions"):
        values = {_canonical(item[key]) for item in candidates}
        if len(values) != 1:
            raise ValueError(
                "resource_service_role_record_contract_not_homogeneous:" + key)
    if scope == "joint_phase":
        resource_sets = {tuple(item["resources"]) for item in candidates}
        if len(resource_sets) != 1:
            raise ValueError(
                "resource_service_role_record_contract_not_homogeneous:resources")
    windows = {item["window"] for item in candidates}
    if len(windows) != 1:
        raise ValueError(
            "resource_service_role_record_contract_not_homogeneous:window")
    common_window = next(iter(windows))
    if scope == "independent_resource" and common_window in _JOINT_ROLE_WINDOWS:
        raise ValueError(
            "resource_service_role_record_contract_not_homogeneous:window")
    declared_supplies = {_canonical(item["supply"])
                         for item in candidates
                         if item.get("supply") is not None}
    if supply is not None and declared_supplies and \
            _canonical(supply) not in declared_supplies:
        raise ValueError("resource_service_role_record_supply_conflict")
    if len(declared_supplies) > 1:
        raise ValueError("resource_service_role_record_supply_conflict")
    if supply is None and declared_supplies:
        supply = candidates[0]["supply"]
    declared_startups = {_canonical(item["startup_us"])
                         for item in candidates
                         if item.get("startup_us") is not None}
    if startup_us is not None and declared_startups and \
            _canonical(startup_us) not in declared_startups:
        raise ValueError("resource_service_role_record_startup_conflict")
    if len(declared_startups) > 1:
        raise ValueError("resource_service_role_record_startup_conflict")
    if startup_us is None and declared_startups:
        startup_us = candidates[0]["startup_us"]

    distinct_work = {item["work"] for item in candidates}
    if len(distinct_work) < 2 and supply is None and startup_us is None:
        for item in candidates:
            underdetermined.append({
                "cohort_id": item["cohort_id"],
                "case_id": item.get("case_id"),
                "source_path": item["source_path"],
                "reason": "fit_requires_two_work_sizes_or_declared_rate",
                "missing": ["second_training_work", "declared_supply"],
                "declared": None,
            })
        _finalize_participation(
            "not_fitted", "fit_requires_two_work_sizes_or_declared_rate")
        return {**base, "status": "underdetermined",
                "reason": "fit_requires_two_work_sizes_or_declared_rate",
                "service": None}

    resources = ([resource] if scope == "independent_resource"
                 else list(candidates[0]["resources"]))
    packet = {
        "id": f"role-fit:{resource}:{phase}",
        "scope": scope,
        "resources": resources,
        "phase": phase,
        "work_unit": work_unit,
        "boundary": candidates[0]["boundary"],
        "rate_scope": candidates[0]["rate_scope"],
        "active_units": candidates[0]["active_units"],
        "conditions": candidates[0]["conditions"],
        "samples": [{
            "work": item["work"],
            "duration_us": item["duration_us"],
            "run_id": item["cohort_id"],
            "repeat_index": 0,
            "split": "train",
            "source": str(item["source_path"] or item["cohort_id"]),
        } for item in candidates],
        "provenance": {
            "window": common_window or (
                "mixed" if scope == "joint_phase" else "resource_phase"),
            "observation_kind": (
                "joint_phase" if scope == "joint_phase"
                else "explicit_resource_phase"),
            "boundary_kind": candidates[0]["boundary"],
            "scope": "independent" if scope == "independent_resource" else "joint",
            "role_ledger": {
                "training_role": _ROLE_IN_SAMPLE,
                "role_counts": dict(partition["counts"]),
                "training_sources": [_role_service_source(item)
                                      for item in candidates],
            },
        },
    }
    if supply is not None:
        packet["supply"] = dict(supply)
    if startup_us is not None:
        packet["startup_us"] = startup_us
        packet["response_us"] = response_us
    if scope == "joint_phase":
        packet["identification"] = {"resource_identified": False,
                                    "status": "joint_identified"}
    try:
        fitted = fit_resource_service(packet, hardware=hardware)
    except ValueError as error:
        text = str(error)
        if "requires_two_work_sizes_or_a_declared_rate" in text:
            _finalize_participation(
                "not_fitted", "fit_requires_two_work_sizes_or_declared_rate")
            return {**base, "status": "underdetermined",
                    "reason": "fit_requires_two_work_sizes_or_declared_rate",
                    "service": None}
        raise
    service = dict(fitted["service"])
    provenance = dict(service.get("provenance") or {})
    provenance.update({
        "training_role": "in_sample",
        "role_counts": dict(partition["counts"]),
        "frozen_transfer_role": "independent_transfer_not_in_fit",
        "not_declared_role": "excluded_neither_training_nor_holdout",
        "role_sources": [_role_service_source(item) for item in candidates],
        "source_contract": {
            "scope": scope,
            "resources": resources,
            "phase": phase,
            "work_unit": work_unit,
            "boundary": candidates[0]["boundary"],
            "rate_scope": candidates[0]["rate_scope"],
            "active_units": candidates[0]["active_units"],
            "conditions": candidates[0]["conditions"],
            "window": common_window,
            "duration_fields": sorted({
                item["duration_field"] for item in candidates
                if item.get("duration_field")}),
        },
    })
    service["provenance"] = provenance
    _finalize_participation("trained")
    return {
        **base,
        "status": "fitted",
        "service": service,
        "fit": fitted.get("fit"),
        "formula": fitted.get("formula"),
    }


ACTUAL_EVIDENCE_SCHEMA = "mindie-sim.actual-resource-evidence/1"

_CAPTURE_DTYPE_BYTES = {
    "DT_BF16": 2, "DT_FLOAT16": 2, "DT_FLOAT": 4, "DT_INT8": 1,
    "DT_UINT8": 1, "DT_FLOAT8_E4M3FN": 1, "DT_FLOAT8_E8M0": 1,
    "DT_INT64": 8, "DT_INT32": 4,
}
_CAPTURE_DTYPE_ALIASES = {
    "DT_BF16": "bf16", "DT_FLOAT16": "fp16", "DT_FLOAT": "fp32",
    "DT_INT8": "int8", "DT_UINT8": "uint8", "DT_FLOAT8_E4M3FN": "fp8_e4m3",
    "DT_FLOAT8_E8M0": "fp8_e8m0", "DT_INT64": "int64", "DT_INT32": "int32",
}
_AIC_PIPES = ("mac", "scalar", "mte1", "mte2", "mte3", "fixpipe")
_AIV_PIPES = ("vec", "scalar", "mte2", "mte3")
_PIPE_TO_OWNER_PHASE = {
    "mac": ("cube", "compute"),
    "mte1": ("mte1", "l1_to_l0"),
    "mte2": ("mte2", "load"),
    "mte3": ("mte3", "store"),
    "fixpipe": ("fixpipe", "epilogue"),
    "scalar": ("scalar", "scalar"),
    "vec": ("vector", "compute"),
}


def _capture_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"N/A", "None"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _capture_int(value):
    number = _capture_float(value)
    return None if number is None else int(number)


def load_native_cohort_validation(path, *, expected_sha256=None):
    """Read the T03 native cohort validation CSV and its source hash.

    The file is already the normalised T03 output; this function never rescans
    a capture and never uses a filename as evidence.
    """
    import csv
    import hashlib
    from pathlib import Path
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise ValueError("native_cohort_validation_source_missing:" + str(source))
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    sha = digest.hexdigest()
    if expected_sha256 and sha != str(expected_sha256):
        raise ValueError(
            "native_cohort_validation_sha256_mismatch:"
            + sha + "!=" + str(expected_sha256))
    rows = []
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(dict(row))
    by_cohort = {}
    for row in rows:
        cohort_id = str(row.get("cohort_id") or "").strip()
        if not cohort_id:
            raise ValueError("native_cohort_validation_row_requires_cohort_id")
        if cohort_id in by_cohort:
            raise ValueError(
                "native_cohort_validation_duplicate_cohort_id:" + cohort_id)
        by_cohort[cohort_id] = row
    counts = {}
    for row in rows:
        key = (str(row.get("calibration_relation") or ""),
               str(row.get("contract_match_status") or ""))
        counts[key] = counts.get(key, 0) + 1
    return {
        "schema": "mindie-sim.native-cohort-validation/1",
        "path": str(source),
        "sha256": sha,
        "bytes": source.stat().st_size,
        "rows": rows,
        "by_cohort": by_cohort,
        "count": len(rows),
        "counts": {f"{role}:{status}": value
                   for (role, status), value in sorted(counts.items())},
    }


def read_capture_locators(capture_path, locators, *, with_sha256=False):
    """Stream one capture and return only the explicitly requested CSV lines."""
    import csv
    import hashlib
    from pathlib import Path
    path = Path(capture_path).expanduser().resolve()
    if not path.is_file():
        raise ValueError("capture_source_missing:" + str(path))
    requested = sorted({int(item) for item in locators})
    if not requested:
        raise ValueError("capture_locator_lines_required")
    if requested[0] < 2:
        raise ValueError("capture_locator_line_is_one_based_data_line")
    found = {}
    rows_scanned = 0
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "Task ID" not in reader.fieldnames:
            raise ValueError("capture_source_requires_kernel_details_header")
        for row in reader:
            rows_scanned += 1
            if reader.line_num in requested:
                found[reader.line_num] = dict(row)
    sha = None
    if with_sha256:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        sha = digest.hexdigest()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha,
        "requested_lines": requested,
        "rows": found,
        "missing_lines": [line for line in requested if line not in found],
        "rows_scanned": rows_scanned,
    }


def capture_image_identity(capture_path):
    """Read the same-capture image identity from its retained side files."""
    import hashlib
    import re
    from pathlib import Path
    source = Path(capture_path).expanduser().resolve()
    directory = source.parent
    checks = []
    readme = directory / "README.md"
    if readme.is_file():
        text = readme.read_text(encoding="utf-8", errors="replace")
        match = re.search(
            r"image\s+`([0-9a-fA-F]{12,64})`(?:\s+\(([^)]+)\))?", text)
        if match:
            digest = match.group(1).lower()
            return {
                "image_id": digest,
                "revision": (match.group(2) or "").strip() or None,
                "source": str(readme),
                "raw": match.group(0),
                "sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
            }
        checks.append({"source": str(readme), "status": "image_not_found"})
    env = directory / "session.env"
    if env.is_file():
        text = env.read_text(encoding="utf-8", errors="replace")
        match = re.search(r"(?m)^image=sha256:([0-9a-fA-F]{12,64})\s*$", text)
        tag = re.search(r"(?m)^image_tags=(.+)$", text)
        if match:
            digest = match.group(1).lower()
            return {
                "image_id": digest,
                "revision": None,
                "image_tags": (tag.group(1).strip() if tag else None),
                "source": str(env),
                "raw": match.group(0),
                "sha256": hashlib.sha256(env.read_bytes()).hexdigest(),
            }
        checks.append({"source": str(env), "status": "image_not_found"})
    return {"image_id": None, "revision": None, "source": None,
            "checks": checks, "reason": "capture_image_identity_not_declared"}


_CAPTURE_ZERO = 0
_CAPTURE_ONE = 1
_CAPTURE_MATMUL_ARITY = 2
_CAPTURE_QBMM_ARITY = 4
_CAPTURE_NATIVE_ARITY = {
    "MatMulV3": _CAPTURE_MATMUL_ARITY,
    "QuantBatchMatmulV3": _CAPTURE_QBMM_ARITY,
}


def _capture_binary_contract(image_id, native_type):
    """Record caller-supplied image provenance without prefix dispatch.

    A short image prefix is not a program identity: the observation contract
    keeps the declared digest and reports binary equivalence as undeclared
    unless the caller supplies an exact identity.  No image-to-adapter table
    or prefix expansion exists here.
    """
    from mindie_sim.observations.resource_services import identity_digest
    digest = identity_digest(image_id or "")
    return {
        "native_type": native_type,
        "observed_image_id": digest or None,
        "binary_match": False,
        "status": ("caller_supplied_image_identity" if digest
                   else "image_identity_not_declared"),
    }


def parse_capture_abi(row):
    """Parse raw shapes/dtypes/formats into the ABI-signature order."""
    def clean(value):
        return str(value or "").strip().strip("\"").strip("'").strip()

    def split(value):
        return [item.strip() for item in clean(value).split(";") if item.strip()]

    def shaped(values):
        parsed = []
        for item in values:
            try:
                parsed.append([int(part) for part in item.split(",") if part])
            except ValueError:
                parsed.append(None)
        return parsed

    inputs = split(row.get("Input Shapes"))
    input_dtypes = [clean(item) for item in split(row.get("Input Data Types"))]
    input_formats = [clean(item) for item in split(row.get("Input Formats"))]
    outputs = split(row.get("Output Shapes"))
    output_dtypes = [clean(item) for item in split(row.get("Output Data Types"))]
    output_formats = [clean(item) for item in split(row.get("Output Formats"))]
    signature = "|".join([
        ";".join(inputs), ";".join(input_dtypes), ";".join(input_formats),
        ";".join(outputs), ";".join(output_dtypes), ";".join(output_formats),
    ])
    return {
        "input_shapes": shaped(inputs),
        "input_dtypes": input_dtypes,
        "input_formats": input_formats,
        "output_shapes": shaped(outputs),
        "output_dtypes": output_dtypes,
        "output_formats": output_formats,
        "abi_signature": signature,
    }


def _capture_geometry(native_type, inputs, outputs):
    """Derive M/K/N from the captured ABI shapes, not from an image table."""
    first, second = list(inputs[_CAPTURE_ZERO]), list(inputs[_CAPTURE_ONE])
    if len(first) < _CAPTURE_MATMUL_ARITY or len(second) < _CAPTURE_MATMUL_ARITY:
        raise ValueError("capture_abi_requires_rank_two_matmul_operands")
    rows = _CAPTURE_ONE
    for dim in first[:-_CAPTURE_ONE]:
        rows *= int(dim)
    inner = int(first[-_CAPTURE_ONE])
    if native_type == "MatMulV3":
        columns = int(second[-_CAPTURE_ONE])
        declared_inner = int(second[_CAPTURE_ZERO])
        if declared_inner != inner and columns == inner:
            columns, inner = declared_inner, int(second[-_CAPTURE_ONE])
    else:
        declared_inner = int(second[_CAPTURE_ZERO])
        columns = (int(second[-_CAPTURE_ONE]) if declared_inner == inner
                   else declared_inner)
    return {
        "m": int(rows), "k": int(inner), "n": int(columns),
        "macs": int(rows) * int(inner) * int(columns),
    }


def _capture_tensor(name, shape, dtype, layout=None):
    return {
        "name": name,
        "shape": [int(item) for item in shape],
        "dtype": _CAPTURE_DTYPE_ALIASES.get(dtype, str(dtype).lower()),
        **({"layout": layout} if layout else {}),
    }


def _capture_bytes(shapes, dtypes):
    total = _CAPTURE_ZERO
    for shape, dtype in zip(shapes, dtypes):
        width = _CAPTURE_DTYPE_BYTES.get(dtype)
        if width is None or not shape:
            continue
        elements = _CAPTURE_ONE
        for dim in shape:
            elements *= int(dim)
        total += elements * width
    return total


def bind_capture_native_plan(row, *, image_id=None):
    """Bind one captured ABI row to the canonical declared call geometry.

    Shapes and dtype widths come only from the captured row; the returned
    declaration carries no measured duration, no image-prefix program dispatch
    and no ``predict_abi`` pricing.  Missing or malformed geometry stays an
    explicit unsupported result.
    """
    native_type = str(row.get("Type") or "")
    abi = parse_capture_abi(row)
    inputs = abi["input_shapes"]
    outputs = abi["output_shapes"]
    version = ({"image_id": str(image_id)} if image_id
               else {"source": "captured_abi"})
    binary = _capture_binary_contract(image_id, native_type)
    result = {
        "native_type": native_type,
        "entry_adapter": None,
        "binary": binary,
        "version": version,
        "status": "not_attempted",
        "reasons": [],
        "source_role_status": "not_identified_from_capture",
        "dispatch_status": "not_established_from_capture",
    }
    arity = _CAPTURE_NATIVE_ARITY.get(native_type)
    parsed = list(inputs[:arity]) if arity else []
    if (arity is None or len(inputs) < arity or not outputs
            or any(shape is None for shape in parsed + list(outputs[:_CAPTURE_ONE]))):
        result["status"] = "unsupported"
        result["reasons"] = ["no_canonical_geometry_for_capture_family"]
        return result
    try:
        geometry = _capture_geometry(native_type, inputs, outputs)
    except (TypeError, ValueError, IndexError) as error:
        result["status"] = "adapter_error"
        result["reasons"] = [type(error).__name__ + ":" + str(error)]
        return result
    input_bytes = _capture_bytes(inputs, abi["input_dtypes"])
    output_bytes = _capture_bytes(outputs, abi["output_dtypes"])
    issued = {
        "macs": geometry["macs"],
        "mte2_bytes": input_bytes,
        "mte3_bytes": output_bytes,
    }
    tensors = [
        _capture_tensor(f"input{index}", shape, abi["input_dtypes"][index]
                        if index < len(abi["input_dtypes"]) else "")
        for index, shape in enumerate(inputs)]
    tensors.extend(
        _capture_tensor(f"output{index}", shape,
                        abi["output_dtypes"][index]
                        if index < len(abi["output_dtypes"]) else "")
        for index, shape in enumerate(outputs))
    selected_call = {
        "api": native_type,
        "arity": arity,
        "shapes": {"inputs": inputs, "outputs": outputs},
        "dtypes": {"inputs": abi["input_dtypes"],
                   "outputs": abi["output_dtypes"]},
        "tensors": tensors,
    }
    owner_phases = {
        "cube": {
            "owner": "cube", "phase": "matmul", "resource": "cube",
            "declared_logical_macs": geometry["macs"],
            "declared_padded_macs": None,
            "declared_bytes": None, "issued_macs": None, "issued_bytes": None,
            "work_basis": "declared_ABI_geometry_not_measured_dispatch",
        },
        "mte2": {
            "owner": "mte2", "phase": "load", "resource": "mte2",
            "declared_logical_macs": None, "declared_padded_macs": None,
            "declared_bytes": input_bytes, "issued_macs": None,
            "issued_bytes": None,
            "work_basis": "declared_ABI_geometry_not_measured_dispatch",
        },
        "mte3": {
            "owner": "mte3", "phase": "store", "resource": "mte3",
            "declared_logical_macs": None, "declared_padded_macs": None,
            "declared_bytes": output_bytes, "issued_macs": None,
            "issued_bytes": None,
            "work_basis": "declared_ABI_geometry_not_measured_dispatch",
        },
    }
    descriptor = {
        "selected_call": selected_call,
        "geometry": geometry,
        "issued": issued,
        "kernels": list(owner_phases.values()),
    }
    result.update({
        "status": "predicted",
        "record_status": "predicted",
        "geometry": geometry,
        "work": {
            "work_unit": "macs",
            "logical_work": geometry["macs"],
            "logical_bytes": input_bytes + output_bytes,
            "issued_work": None,
            "issued_work_status": "requires_selected_dispatch_and_tiling",
        },
        "selected_call": selected_call,
        "operator_plan": descriptor,
        "issued": issued,
        "kernels": list(owner_phases.values()),
        "owner_phases": owner_phases,
        "plan_basis": "declared_ABI_geometry_not_measured_dispatch",
        "tail_status": "not_exposed_at_descriptor_boundary",
    })
    result["source_established"] = bool(binary.get("binary_match"))
    return result


def capture_measurement(row):
    """Expose the raw measured counters with their actual unit and limits.

    The kernel_details CSV contains one row per device kernel.  The PMU pipe
    columns are busy intervals inside that same kernel, normalised by the
    AIC/AIV total time; they overlap.  The CSV does not declare whether the
    scalar is an average over active cores or a cumulative interval, and it
    has no per-owner or per-block counter.  ``Block Num`` is therefore kept as
    a block count, not an effective-owner count.
    """
    accelerator = str(row.get("Accelerator Core") or "")
    aicore = _capture_float(row.get("aicore_time(us)"))
    aiv = _capture_float(row.get("aiv_time(us)"))
    aic = {name: _capture_float(row.get(f"aic_{name}_time(us)"))
           for name in _AIC_PIPES}
    aiv_pipes = {name: _capture_float(row.get(f"aiv_{name}_time(us)"))
                 for name in _AIV_PIPES}
    duration = _capture_float(row.get("Duration(us)"))
    checks = []
    if duration is None or duration <= 0:
        raise ValueError("capture_measurement_requires_positive_duration")
    checks.append({"name": "duration_positive", "ok": duration > 0,
                   "detail": f"{duration} us"})
    ratio_checks = []
    for name, value in aic.items():
        if not value:
            continue
        ratio = _capture_float(row.get(f"aic_{name}_ratio"))
        computed = value / aicore if aicore else None
        ok = (ratio is None or computed is None
              or abs(computed - ratio) <= max(5e-4, abs(ratio) * 0.02))
        ratio_checks.append({"pipe": name, "time_us": value,
                             "aicore_time_us": aicore,
                             "computed_ratio": computed,
                             "declared_ratio": ratio, "ok": ok})
    for name, value in aiv_pipes.items():
        if not value:
            continue
        ratio = _capture_float(row.get(f"aiv_{name}_ratio"))
        computed = value / aiv if aiv else None
        ok = (ratio is None or computed is None
              or abs(computed - ratio) <= max(5e-4, abs(ratio) * 0.02))
        ratio_checks.append({"pipe": "aiv_" + name, "time_us": value,
                             "aiv_time_us": aiv,
                             "computed_ratio": computed,
                             "declared_ratio": ratio, "ok": ok})
    bad_ratios = [item for item in ratio_checks if not item["ok"]]
    checks.append({"name": "pmu_ratios_match_total_time",
                   "ok": not bad_ratios,
                   "detail": f"{len(ratio_checks) - len(bad_ratios)}/"
                             f"{len(ratio_checks)} checked"})
    aic_active = ({name: value for name, value in aic.items() if value}
                  if accelerator in {"AI_CORE", "MIX_AIC", "MIX_AIV"} else {})
    aiv_active = ({name: value for name, value in aiv_pipes.items() if value}
                  if accelerator in {"AI_VECTOR_CORE", "MIX_AIC", "MIX_AIV"}
                  else {})
    total_time = max(value for value in (aicore or 0.0, aiv or 0.0))
    pipe_sum = sum(aic_active.values()) + sum(aiv_active.values())
    overlaps = bool(total_time and pipe_sum > total_time + 1e-6)
    checks.append({"name": "pipe_times_overlap_or_sum_at_most_total",
                   "ok": True,
                   "detail": ("overlapping; not additive" if overlaps
                              else "within total window")})
    block_num = _capture_int(row.get("Block Num"))
    mix_block_num = _capture_int(row.get("Mix Block Num"))
    checks.append({"name": "block_num_is_not_owner_count", "ok": True,
                   "detail": (f"Block Num={block_num}; it is a block count, "
                              "not a measured active-owner count")})
    return {
        "boundary": "device_kernel",
        "window": "one_kernel_invocation",
        "units": {"duration": "us", "pipe_times": "us",
                  "ratios": "fraction", "cycles": "count"},
        "kernel_duration_us": duration,
        "aicore_time_us": aicore,
        "aiv_time_us": aiv,
        "accelerator_core": accelerator or None,
        "aic_pipe_busy_us": aic_active,
        "aiv_pipe_busy_us": aiv_active,
        "pipe_busy_us": {
            **{f"aic_{name}": value for name, value in aic_active.items()},
            **{f"aiv_{name}": value for name, value in aiv_active.items()},
        },
        "pipe_ratio_checks": ratio_checks,
        "pipe_busy_sum_us": pipe_sum,
        "pipe_times_overlap": overlaps,
        "pipe_times_additive": False,
        "aggregation_semantics": (
            "not_declared_by_kernel_details_csv; pipe times are per-event busy "
            "intervals normalised by the AIC/AIV total time; average-vs-"
            "cumulative and active-owner aggregation are not declared"),
        "effective_owners": None,
        "owner_aggregation": "not_declared",
        "block_num": block_num,
        "mix_block_num": mix_block_num,
        "block_num_is_owner": False,
        "issued_mac": None,
        "issued_bytes": None,
        "issued_work_status": "not_measured_by_capture",
        "scheduling": {
            "device_id": row.get("Device_id"),
            "model_id": row.get("Model ID"),
            "task_id": row.get("Task ID"),
            "stream_id": row.get("Stream ID"),
            "start_us": _capture_float(row.get("Start Time(us)")),
            "end_us": (None if _capture_float(row.get("Start Time(us)")) is None
                       else _capture_float(row.get("Start Time(us)")) + duration),
            "wait_us": _capture_float(row.get("Wait Time(us)")),
            "op_state": row.get("OP State"),
            "kernel_name": row.get("Name"),
            "kernel_type": row.get("Type"),
        },
        "checks": checks,
    }


def logical_capture_work(row):
    """Derive only source-declared logical work from the raw ABI shapes.

    MACs are ``M*N*K`` for a dense matrix call.  The logical tensor footprint
    is the call-boundary operand/output byte count.  Neither is claimed to be
    issued work; issued work requires the selected native program/dispatch.
    """
    abi = parse_capture_abi(row)
    native_type = str(row.get("Type") or "")
    inputs = abi["input_shapes"]
    input_dtypes = abi["input_dtypes"]
    outputs = abi["output_shapes"]
    output_dtypes = abi["output_dtypes"]
    if native_type in {"MatMulV3", "QuantBatchMatmulV3"} and len(inputs) >= 2 \
            and inputs[0] and inputs[1] and outputs and outputs[0]:
        first, second, output = inputs[0], inputs[1], outputs[0]
        if len(first) != 2 or len(second) != 2 or len(output) != 2:
            return {"native_type": native_type, "work_unit": None,
                    "logical_work": None,
                    "reason": "logical_matrix_work_requires_rank_two_ABI"}
        m, n = int(output[0]), int(output[1])
        orientation = None
        if first[0] == m:
            k = int(first[1])
            orientation = "x1_mk"
        elif first[1] == m:
            k = int(first[0])
            orientation = "x1_km"
        else:
            return {"native_type": native_type, "work_unit": None,
                    "logical_work": None,
                    "reason": "logical_matrix_work_x1_does_not_match_output_m"}
        if second[0] == n and second[1] == k:
            b_layout = "n_k"
        elif second[0] == k and second[1] == n:
            b_layout = "k_n"
        else:
            return {"native_type": native_type, "work_unit": None,
                    "logical_work": None,
                    "reason": "logical_matrix_work_x2_does_not_match_output_n_and_k"}
        macs = m * n * k
        bytes_by_tensor = {}
        total_bytes = 0
        for index, (shape, dtype) in enumerate(zip(inputs, input_dtypes)):
            width = _CAPTURE_DTYPE_BYTES.get(dtype)
            if width is None or not shape:
                continue
            count = 1
            for dim in shape:
                count *= int(dim)
            bytes_by_tensor[f"input{index}"] = count * width
            total_bytes += count * width
        for index, (shape, dtype) in enumerate(zip(outputs, output_dtypes)):
            width = _CAPTURE_DTYPE_BYTES.get(dtype)
            if width is None or not shape:
                continue
            count = 1
            for dim in shape:
                count *= int(dim)
            bytes_by_tensor[f"output{index}"] = count * width
            total_bytes += count * width
        return {
            "native_type": native_type,
            "work_unit": "mac",
            "logical_work": int(macs),
            "formula": "M*N*K",
            "m": m, "n": n, "k": k,
            "x1_orientation": orientation,
            "x2_layout": b_layout,
            "logical_bytes": int(total_bytes),
            "logical_bytes_by_tensor": bytes_by_tensor,
            "logical_work_status": "derived_from_raw_ABI_shapes",
            "issued_work": None,
            "issued_work_status": "requires_selected_native_program_and_dispatch",
        }
    if native_type in {"RmsNorm", "SwiGlu", "DynamicMxQuant", "HcPost",
                       "Slice", "ScatterPaKvCache"} and inputs:
        shape = max(inputs, key=lambda item: (len(item or []), item or []))
        if shape:
            count = 1
            for dim in shape:
                count *= int(dim)
            total_bytes = 0
            for shapes, dtypes in ((inputs, input_dtypes),
                                   (outputs, output_dtypes)):
                for item, dtype in zip(shapes, dtypes):
                    width = _CAPTURE_DTYPE_BYTES.get(dtype)
                    if width is None or not item:
                        continue
                    elements = 1
                    for dim in item:
                        elements *= int(dim)
                    total_bytes += elements * width
            return {
                "native_type": native_type,
                "work_unit": "elements",
                "logical_work": int(count),
                "formula": "max_input_element_count",
                "logical_bytes": int(total_bytes),
                "logical_work_status": "derived_from_raw_ABI_shapes",
                "issued_work": None,
                "issued_work_status": "requires_selected_native_program_and_dispatch",
            }
    return {
        "native_type": native_type,
        "work_unit": None,
        "logical_work": None,
        "reason": "logical_work_not_derivable_from_capture_family",
        "issued_work": None,
    }


def join_capture_measurement_and_plan(measurement, plan):
    """Pair measured pipe times with the selected plan's owner/phase work."""
    if plan.get("status") != "predicted":
        return {
            "status": "unmatched",
            "reason": "native_plan_not_predicted:" + ",".join(
                str(item) for item in plan.get("reasons") or ()),
            "owner_phases": {},
            "resource_identified": False,
        }
    owner_phases = plan.get("owner_phases") or {}
    joins = []
    active = {}
    for name, value in (measurement.get("aic_pipe_busy_us") or {}).items():
        active[f"aic_{name}"] = value
    for name, value in (measurement.get("aiv_pipe_busy_us") or {}).items():
        active[f"aiv_{name}"] = value
    for pipe, measured_us in sorted(active.items()):
        owner_phase = (_PIPE_TO_OWNER_PHASE.get(pipe.removeprefix("aic_"))
                       if pipe.startswith("aic_") else None)
        if not owner_phase:
            joins.append({
                "pipe": pipe, "measured_us": measured_us,
                "owner": None, "phase": None,
                "issued_work": None, "issued_unit": None,
                "declared_work": None, "declared_unit": None,
                "status": "pipe_not_in_selected_native_plan",
            })
            continue
        owner, phase = owner_phase
        plan_owner = owner_phases.get(owner)
        if not plan_owner:
            joins.append({
                "pipe": pipe, "measured_us": measured_us,
                "owner": owner, "phase": phase,
                "issued_work": None, "issued_unit": None,
                "status": "owner_not_in_plan",
            })
            continue
        if owner == "cube":
            declared = plan_owner.get("declared_padded_macs")
            declared_unit = "mac"
            declared_basis = "declared_padded_16_alignment"
        else:
            declared = plan_owner.get("declared_bytes")
            declared_unit = "bytes"
            declared_basis = "declared_call_boundary_tensor_footprint"
        joins.append({
            "pipe": pipe, "measured_us": measured_us,
            "owner": owner, "phase": plan_owner.get("phase"),
            "issued_work": None,
            "issued_unit": None,
            "issued_status": "not_measured_by_capture",
            "declared_work": declared,
            "declared_unit": declared_unit if declared is not None else None,
            "declared_basis": declared_basis,
            "declared_source": plan.get("plan_basis"),
            "diagnostic_us_per_declared_unit": (
                measured_us / declared if declared else None),
            "training_eligible": False,
            "status": ("measured_pipe_with_declared_plan_work"
                       if declared is not None
                       else "plan_owner_has_no_declared_work"),
        })
    return {
        "status": "paired_diagnostic",
        "joins": joins,
        "unmapped_pipes": sorted(
            item["pipe"] for item in joins
            if item.get("status") == "pipe_not_in_selected_native_plan"),
        "owner_aggregation": measurement.get("owner_aggregation"),
        "resource_identified": False,
        "issued_work_measured": False,
        "reason": "measured_pipe_times_are_overlapping_joint_window_and_"
                  "owner_aggregation_is_not_declared",
        "training_eligible": False,
    }


def _capture_record_cohort(row):
    import json as _json
    fields = {}
    for key in ("contract_kind", "contract_match_status", "contract_checks",
                "contract_notes", "family_role", "family_status",
                "archive_event_count", "archive_mean_us", "cohort_window",
                "global_signature_count", "global_signature_mean_us",
                "full_cohort_window_reproduced", "clock_domain", "step_state",
                "locator_status", "locator_notes", "ambiguities"):
        value = row.get(key)
        if key in {"contract_checks", "contract_notes", "cohort_window",
                   "clock_domain", "step_state", "locator_notes",
                   "ambiguities"}:
            try:
                fields[key] = _json.loads(value) if value else None
            except (TypeError, ValueError):
                fields[key] = value
        else:
            fields[key] = value
    return fields


def _capture_locator(row):
    return {
        "csv_line": _capture_int(row.get("example_locator_csv_line")),
        "row_index": _capture_int(row.get("example_locator_row_index")),
        "task_id": row.get("example_locator_task_id"),
        "stream_id": row.get("example_locator_stream_id"),
        "device_id": row.get("example_locator_device_id"),
        "raw_start_us": _capture_float(row.get("raw_start_us")),
        "raw_duration_us": _capture_float(row.get("raw_duration_us")),
        "raw_end_us": _capture_float(row.get("raw_end_us")),
        "raw_op_type": row.get("raw_op_type"),
        "raw_input_shapes": row.get("raw_input_shapes"),
        "raw_output_shapes": row.get("raw_output_shapes"),
        "abi_signature": row.get("abi_signature"),
        "source_path": row.get("source_path"),
    }


def _locator_checks(raw, locator):
    checks = []
    for key, raw_key in (("device_id", "Device_id"),
                         ("task_id", "Task ID"),
                         ("stream_id", "Stream ID")):
        expected = str(locator.get(key) or "")
        actual = str(raw.get(raw_key) or "")
        checks.append({"name": key, "ok": expected == actual,
                       "expected": expected, "actual": actual})
    start = _capture_float(raw.get("Start Time(us)"))
    duration = _capture_float(raw.get("Duration(us)"))
    expected_start = locator.get("raw_start_us")
    expected_duration = locator.get("raw_duration_us")
    checks.append({
        "name": "start_time",
        "ok": (start is not None and expected_start is not None
               and abs(start - expected_start) <= 1e-6),
        "expected": expected_start, "actual": start,
    })
    checks.append({
        "name": "duration",
        "ok": (duration is not None and expected_duration is not None
               and abs(duration - expected_duration) <= 1e-6),
        "expected": expected_duration, "actual": duration,
    })
    abi = parse_capture_abi(raw)["abi_signature"]
    expected_abi = str(locator.get("abi_signature") or "")
    expected_sections = expected_abi.split("|")
    actual_sections = abi.split("|")
    tokens_match = (len(expected_sections) == len(actual_sections)
                    and all(
                        [token for token in expected.split(";") if token]
                        == [token for token in actual.split(";") if token]
                        for expected, actual in zip(expected_sections,
                                                    actual_sections)))
    checks.append({
        "name": "abi_signature", "ok": tokens_match,
        "expected": expected_abi, "actual": abi,
        "slot_compaction": tokens_match and abi != expected_abi,
        "detail": ("present_tensor_tokens_match"
                   + (";raw_field_compacts_empty_optional_slots"
                      if tokens_match and abi != expected_abi else "")),
    })
    return checks


def build_actual_evidence(role_records, native_validation, *,
                          with_sha256=False, max_cohorts=None,
                          role="in_sample"):
    """Join role ledger -> T03 validation -> same-boundary raw PMU rows.

    The function reads only ``in_sample`` captures.  Frozen-transfer and
    not-declared cohorts are registered but their raw captures are not opened.
    Every join check, participation record, exclusion, duplicate counter and
    source hash is returned so the training set is reproducible.
    """
    import hashlib
    if isinstance(native_validation, Mapping):
        native_rows = list(native_validation.get("rows") or ())
        native_meta = {key: native_validation.get(key)
                       for key in ("path", "sha256", "bytes", "count",
                                   "counts")}
    else:
        native_rows = list(native_validation)
        native_meta = {"path": None, "sha256": None, "bytes": None,
                       "count": len(native_rows), "counts": {}}
    native_by_cohort = {str(row.get("cohort_id")): row
                        for row in native_rows}
    partition = calibration_role_partitions(role_records)
    selected = list(partition["in_sample"])
    if max_cohorts is not None:
        selected = selected[:int(max_cohorts)]
    registration = {
        "role_counts": dict(partition["counts"]),
        "total": partition["total"],
        "in_sample_ids": [record["cohort_id"]
                          for record in partition["in_sample"]],
        "frozen_transfer_ids": [record["cohort_id"]
                                for record in partition["frozen_transfer"]],
        "not_declared_ids": [record["cohort_id"]
                            for record in partition["not_declared"]],
        "policy": partition["policy"],
        "read_capture_roles": [role],
    }
    excluded = []
    for record in partition["frozen_transfer"]:
        excluded.append({
            "cohort_id": record["cohort_id"],
            "role": record.get("role"),
            "reason": "role_frozen_transfer_not_training",
            "source_path": record.get("source_path"),
        })
    for record in partition["not_declared"]:
        excluded.append({
            "cohort_id": record["cohort_id"],
            "role": record.get("role"),
            "reason": "role_not_declared_neither_training_nor_holdout",
            "source_path": record.get("source_path"),
        })
    joined = []
    groups_by_capture = {}
    for record in selected:
        cohort_id = record["cohort_id"]
        row = native_by_cohort.get(cohort_id)
        if row is None:
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "native_validation_row_missing",
                "source_path": record.get("source_path"),
            })
            continue
        if str(row.get("calibration_relation") or "") != _ROLE_IN_SAMPLE:
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "native_validation_role_mismatch",
                "declared_role": row.get("calibration_relation"),
                "source_path": record.get("source_path"),
            })
            continue
        if str(row.get("case_id") or "") != str(record.get("case_id") or ""):
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "case_id_mismatch",
                "ledger_case": record.get("case_id"),
                "native_case": row.get("case_id"),
                "source_path": record.get("source_path"),
            })
            continue
        if str(row.get("source_path") or "") != str(record.get("source_path") or ""):
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "source_path_mismatch",
                "ledger_source": record.get("source_path"),
                "native_source": row.get("source_path"),
            })
            continue
        locator = _capture_locator(row)
        if not locator["csv_line"]:
            excluded.append({
                "cohort_id": cohort_id, "role": record.get("role"),
                "reason": "locator_line_missing",
                "source_path": row.get("source_path"),
            })
            continue
        groups_by_capture.setdefault(locator["source_path"], []).append(
            (record, row, locator))
    capture_files = {}
    raw_by_key = {}
    for source_path, items in groups_by_capture.items():
        lines = [item[2]["csv_line"] for item in items]
        try:
            raw = read_capture_locators(source_path, lines,
                                        with_sha256=with_sha256)
        except ValueError as error:
            for record, row, locator in items:
                excluded.append({
                    "cohort_id": record["cohort_id"],
                    "role": record.get("role"),
                    "reason": "capture_unavailable",
                    "detail": str(error),
                    "source_path": source_path,
                })
            continue
        capture_files[source_path] = {
            "path": raw["path"], "bytes": raw["bytes"],
            "sha256": raw["sha256"], "rows_scanned": raw["rows_scanned"],
            "image": capture_image_identity(source_path),
        }
        for record, row, locator in items:
            raw_row = raw["rows"].get(locator["csv_line"])
            if raw_row is None:
                excluded.append({
                    "cohort_id": record["cohort_id"],
                    "role": record.get("role"),
                    "reason": "locator_row_missing",
                    "source_path": source_path,
                    "csv_line": locator["csv_line"],
                })
                continue
            raw_by_key[record["cohort_id"]] = raw_row
    for record in selected:
        cohort_id = record["cohort_id"]
        row = native_by_cohort.get(cohort_id)
        if row is None or cohort_id not in raw_by_key:
            continue
        raw_row = raw_by_key[cohort_id]
        locator = _capture_locator(row)
        image = capture_files.get(locator["source_path"], {}).get("image") or {}
        contract = _capture_record_cohort(row)
        measurement = capture_measurement(raw_row)
        work = logical_capture_work(raw_row)
        checks = _locator_checks(raw_row, locator)
        locator_verified = all(item["ok"] for item in checks)
        abi_slot_compaction = any(
            item.get("slot_compaction") for item in checks)
        exact = str(row.get("contract_match_status") or "") == "exact"
        plan = bind_capture_native_plan(raw_row,
                                        image_id=image.get("image_id"))
        pairing = join_capture_measurement_and_plan(measurement, plan)
        missing = []
        if not locator_verified:
            missing.append("raw_locator_and_abi_signature_verified")
        if not exact:
            missing.append("exact_geometry_contract")
        if str(row.get("full_cohort_window_reproduced") or "") != "True":
            missing.append("reproduced_full_cohort_window")
        if work.get("logical_work") is None:
            missing.append("logical_physical_work")
        if plan.get("status") != "predicted":
            missing.append("selected_native_program")
            missing.extend(
                "native_plan:" + str(item) for item in plan.get("reasons") or ())
        else:
            missing.append("dispatch_and_tiling_identity")
            missing.append("issued_work_measured")
        missing.append("owner_aggregation_and_active_owner_count")
        missing.append("independent_pipe_measurement")
        eligible = False
        if not locator_verified:
            reason = "raw_locator_or_abi_signature_mismatch"
        elif not exact:
            reason = "geometry_contract_not_exact"
        elif work.get("logical_work") is None:
            reason = "logical_physical_work_not_derivable"
        elif plan.get("status") != "predicted":
            reason = "selected_native_program_not_established"
        elif not plan.get("source_established"):
            missing.append("same_binary_source_identity")
            reason = "binary_source_not_established"
        else:
            reason = "measured_pipe_aggregation_and_dispatch_not_established"
        joined.append({
            "cohort_id": cohort_id,
            "case_id": record.get("case_id"),
            "role": record.get("role"),
            "source_path": locator["source_path"],
            "locator": locator,
            "native_validation": contract,
            "capture_image": image,
            "locator_checks": checks,
            "locator_verified": locator_verified,
            "abi_slot_compaction": abi_slot_compaction,
            "raw_measurement": measurement,
            "logical_work": work,
            "native_plan": plan,
            "measurement_plan_join": pairing,
            "training_eligibility": {
                "eligible": eligible,
                "reason": reason,
                "missing": list(dict.fromkeys(missing)),
            },
            "duplicates": {
                "source": "T03 native_cohort_validation global_signature_count",
                "global_signature_count": _capture_int(
                    row.get("global_signature_count")),
                "archive_event_count": _capture_int(
                    row.get("archive_event_count")),
                "global_signature_mean_us": _capture_float(
                    row.get("global_signature_mean_us")),
                "example_row_is_one_occurrence": True,
            },
        })
    groups = {}
    for item in joined:
        plan = item["native_plan"]
        key = (str(item["capture_image"].get("image_id")),
               str(item["raw_measurement"]["scheduling"].get("kernel_type")),
               str(item["native_validation"].get("contract_match_status")))
        group = groups.setdefault(key, {
            "image_id": key[0], "native_type": key[1],
            "contract_match_status": key[2], "cohorts": [],
        })
        group["cohorts"].append(item["cohort_id"])
    for group in groups.values():
        group["count"] = len(group["cohorts"])
        group["fit_eligible"] = False
        group["missing"] = [
            "independent_pipe_work_and_owner_aggregation",
            "same_dispatch_tiling_identity",
            "at_least_two_independent_work_sizes_for_one_resource",
        ]
    pairs = []
    by_type = {}
    for item in joined:
        by_type.setdefault(
            (item["source_path"],
             item["raw_measurement"]["scheduling"].get("kernel_type")),
            []).append(item)
    for (source_path, native_type), items in sorted(
            by_type.items(), key=lambda pair: (str(pair[0][0]), str(pair[0][1]))):
        if len(items) < 2:
            continue
        first, second = items[0], items[1]
        same_work = (
            first["logical_work"].get("logical_work")
            == second["logical_work"].get("logical_work"))
        pairs.append({
            "source_path": source_path,
            "native_type": native_type,
            "cohort_ids": [first["cohort_id"], second["cohort_id"]],
            "csv_lines": [first["locator"]["csv_line"],
                          second["locator"]["csv_line"]],
            "task_ids": [first["locator"]["task_id"],
                         second["locator"]["task_id"]],
            "stream_ids": [first["locator"]["stream_id"],
                           second["locator"]["stream_id"]],
            "logical_work": [first["logical_work"].get("logical_work"),
                             second["logical_work"].get("logical_work")],
            "block_num": [
                first["raw_measurement"].get("block_num"),
                second["raw_measurement"].get("block_num")],
            "kernel_duration_us": [
                first["raw_measurement"].get("kernel_duration_us"),
                second["raw_measurement"].get("kernel_duration_us")],
            "same_work": same_work,
            "same_owner_condition": False,
            "two_point_fit_eligible": False,
            "reason": ("overlapping_joint_kernel_window; different_work_and_"
                       "owner_conditions; measured_pipe_aggregation_not_declared"),
        })
    counts = {}
    for item in joined:
        key = item["native_validation"].get("contract_match_status")
        counts[key] = counts.get(key, 0) + 1
    reproduced = sum(
        1 for item in joined
        if item["native_validation"].get("full_cohort_window_reproduced")
        in {"True", True})
    training_inputs = []
    for item in joined:
        if item["training_eligibility"]["eligible"]:
            contract = {
                "cohort_id": item["cohort_id"],
                "role": item["role"],
                "scope": "independent_resource",
                "resources": [item["native_plan"].get("owner_phases", {}).get(
                    "mte2", {}).get("resource", "mte2")],
                "phase": "load",
                "work_unit": "bytes",
                "work": item["native_plan"].get("issued", {}).get("mte2_bytes"),
                "duration_us": item["raw_measurement"].get("kernel_duration_us"),
                "boundary": "device_kernel",
                "provenance": {"window": "resource_phase"},
            }
            training_inputs.append(contract)
    result = {
        "schema": ACTUAL_EVIDENCE_SCHEMA,
        "role_registration": registration,
        "native_validation": native_meta,
        "captures": capture_files,
        "participation": joined,
        "excluded": excluded,
        "counts": {
            "in_sample_joined": len(joined),
            "in_sample_excluded": len(excluded) - len(
                partition["frozen_transfer"]) - len(partition["not_declared"]),
            "contract_match_status": counts,
            "reproduced_full_cohort_window": reproduced,
            "not_reproduced_full_cohort_window": len(joined) - reproduced,
            "abi_slot_compaction": sum(
                1 for item in joined if item.get("abi_slot_compaction")),
        },
        "groups": list(groups.values()),
        "real_pairs": pairs,
        "training_inputs": training_inputs,
        "service": None,
        "fit": {
            "status": "underdetermined",
            "reason": ("no_in_sample_observation_has_an_independent_pipe_"
                       "measurement_with_declared_owner_aggregation_and_"
                       "verified_dispatch"),
            "training_inputs": len(training_inputs),
            "needed_evidence": [
                "selected_native_program_and_same_dispatch_tiling",
                "measured_issued_work_per_owner",
                "declared_owner_aggregation_and_active_owner_count",
                "independent_resource_pipe_measurement",
                "resource_duration_us_for_one_phase",
            ],
        },
        "policy": {
            "read_capture_roles": ["in_sample"],
            "frozen_transfer_capture_read": False,
            "not_declared_capture_read": False,
            "sealed_holdout_read": False,
            "cpu_simulated_duration_in_training": False,
            "target_duration_in_training": False,
            "logical_work_is_issued_work": False,
            "block_num_is_owner_count": False,
            "overlapping_pipe_times_added": False,
        },
    }
    canonical = json.loads(json.dumps(result, sort_keys=True,
                                      separators=(",", ":"),
                                      allow_nan=False))
    result["canonical_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")).hexdigest()
    return result


def fit_actual_training_evidence(role_path, native_validation_path, *,
                                 with_sha256=False, max_cohorts=None):
    """Load the real role/native evidence and return one reproducible report."""
    import hashlib
    from pathlib import Path
    role_path = Path(role_path).expanduser().resolve()
    if not role_path.is_file():
        raise ValueError("role_ledger_source_missing:" + str(role_path))
    records = load_role_ledger(role_path)
    validation = load_native_cohort_validation(native_validation_path)
    evidence = build_actual_evidence(
        records, validation, with_sha256=with_sha256,
        max_cohorts=max_cohorts)
    evidence["role_ledger"] = {
        "path": str(role_path),
        "sha256": hashlib.sha256(role_path.read_bytes()).hexdigest(),
        "bytes": role_path.stat().st_size,
        "count": len(records),
    }
    canonical = json.loads(json.dumps(
        evidence, sort_keys=True, separators=(",", ":"), allow_nan=False))
    evidence["canonical_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")).hexdigest()
    if evidence["training_inputs"]:
        fitted = fit_resource_service_from_role_records(
            evidence["training_inputs"], resource="mte2", phase="load",
            work_unit="bytes", scope="independent_resource")
        evidence["fit"] = {
            "status": fitted.get("status"),
            "service": fitted.get("service"),
            "fit": fitted.get("fit"),
            "reason": fitted.get("reason"),
            "training_participation": fitted.get("training_participation"),
        }
    return evidence


__all__ = [
    "ACTUAL_EVIDENCE_SCHEMA",
    "build_actual_evidence",
    "bind_capture_native_plan",
    "capture_image_identity",
    "capture_measurement",
    "fit_actual_training_evidence",
    "fit_resource_service_from_role_records",
    "join_capture_measurement_and_plan",
    "load_native_cohort_validation",
    "load_role_ledger",
    "logical_capture_work",
    "parse_capture_abi",
    "read_capture_locators",
]
