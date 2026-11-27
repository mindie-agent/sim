"""Cohort accuracy verification and simulated-vs-observed cohort comparison.

[Role] Evaluate cohort prediction accuracy and align simulated structure cohorts
  against observed profiling captures.
[Invariants] Reports honest error metrics with zero silent masking of misses.
[Timeouts & Locks] Pure read-only analysis over supplied cohorts.
[Side-Effect Policy] Read-only."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math


ACCURACY_POLICY = "mindie-sim.cohort-accuracy/1"
_THRESHOLD_US = 50.0
_RELATIVE_TOLERANCE = 0.15
_ABSOLUTE_TOLERANCE_US = 15.0

NATIVE_COMPARISON_SCHEMA = "mindie-profiler-native-comparison/1"
NATIVE_ACCEPTANCE_POLICY = "mindie-sim.native-acceptance/3"
_NATIVE_WORK_ONLY_STATUS = "work_only"
_NATIVE_SCORED_STATUSES = ("within_tolerance", "outside_tolerance")
_NATIVE_WORK_ONLY_REASON = "prediction_work_only_no_duration"
_NATIVE_RELATIVE_TOLERANCE = 0.10
_NATIVE_ABSOLUTE_TOLERANCE_MS = 0.0
_NATIVE_FLOAT_TOLERANCE = 1e-9
_NATIVE_ABSOLUTE_EPSILON = 1e-12


def accuracy_policy() -> dict:
    """Return the report-level policy definition, independent of prediction."""
    return {
        "version": ACCURACY_POLICY,
        "observed_statistic": "complete_event_mean",
        "time_unit": "us",
        "eligible_status": "compared",
        "relative_15pct": {
            "observed_us_gt": _THRESHOLD_US,
            "tolerance": {"unit": "ratio", "value": _RELATIVE_TOLERANCE},
            "pass_status": "within_relative_tolerance",
        },
        "absolute_15us": {
            "observed_us_lte": _THRESHOLD_US,
            "tolerance": {"unit": "us", "value": _ABSOLUTE_TOLERANCE_US},
            "pass_status": "within_launch_floor_tolerance",
        },
        "fail_status": "outside_tolerance",
        "tolerance_inclusive": True,
        "prediction_adjustment": "none",
        "interpretation": "conditional_per_call_comparison; not_native_dispatch_certification",
    }


def accuracy_fields(row: Mapping) -> dict:
    """Recompute additive fields from a current or saved comparison row.

    Existing accuracy fields and precomputed residuals are not inputs. Callers
    may merge the result into a copy of the row without changing its raw values.
    """
    duration = (row.get("observed") or {}).get("duration_ns") or {}
    mean_ns = duration.get("mean")
    predicted_ms = (row.get("prediction") or {}).get("per_call_ms")
    observed = None if mean_ns is None else mean_ns / 1000
    predicted = None if predicted_ms is None else predicted_ms * 1000
    difference = None if observed is None or predicted is None else predicted - observed
    status = row.get("status")
    fields = {
        "observed_per_call_us": observed,
        "predicted_per_call_us": predicted,
        "per_call_diff_us": difference,
        "abs_error_per_call_us": None if difference is None else abs(difference),
        "accuracy_tier": "reference_only" if status == "reference_only" else "unscored",
        "accuracy_status": "reference_only" if status == "reference_only" else "unscored",
        "accuracy_pass": None,
        "accuracy_tolerance": None,
        "accuracy_policy": ACCURACY_POLICY,
    }
    if status != "compared":
        return fields
    if (observed is None or predicted is None or observed <= 0 or predicted < 0
            or not math.isfinite(observed) or not math.isfinite(predicted)):
        raise ValueError("compared_cohort_requires_finite_per_call_times")
    if observed > _THRESHOLD_US:
        tier, unit, tolerance = "relative_15pct", "ratio", _RELATIVE_TOLERANCE
        error = abs(difference) / observed
    else:
        tier, unit, tolerance = "absolute_15us", "us", _ABSOLUTE_TOLERANCE_US
        error = abs(difference)
    passed = error <= tolerance
    fields.update(
        accuracy_tier=tier,
        accuracy_status=(
            "outside_tolerance" if not passed else
            "within_relative_tolerance" if tier == "relative_15pct" else
            "within_launch_floor_tolerance"),
        accuracy_pass=passed,
        accuracy_tolerance={"unit": unit, "value": tolerance},
    )
    return fields


def native_acceptance_policy() -> dict:
    """Fixed per-item criterion for the profiler's independent native comparison.

    The profiler owns the formal pairing and comparison result.  This consumer
    reads ``tolerance.formal`` and ``formal_allowed_error_ms``; a diagnostic
    tolerance is retained beside the formal result and never changes a verdict.
    Legacy 15%/15 us tiers and in-sample mean reproductions are not accepted.
    """
    return {
        "version": NATIVE_ACCEPTANCE_POLICY,
        "source_schema": NATIVE_COMPARISON_SCHEMA,
        "formal_tolerance_field": "tolerance.formal",
        "formal_sample_field": "formal_allowed_error_ms",
        "diagnostic_tolerance_field": "tolerance.diagnostic",
        "diagnostic_sample_fields": [
            "diagnostic_allowed_error_ms", "diagnostic_within",
        ],
        "diagnostic_effect": (
            "retained_for_diagnosis_only; never_changes_formal_verdict_or_status"),
        "observed_statistic": "retained_per_item_observed_ms_with_every_sample",
        "time_unit": "ms",
        "denominator": "observed_ms",
        "scored_item_statuses": list(_NATIVE_SCORED_STATUSES),
        "work_only_item_status": _NATIVE_WORK_ONLY_STATUS,
        "duration_available_field": "comparison.duration_available",
        "duration_denominator": "within_tolerance+outside_tolerance_items",
        "paired_item_set": "all_abi_boundary_paired_items_including_work_only",
        "work_only_evidence": (
            "predicted duration and numerical error stay unavailable; observation, "
            "work and unresolved reason are retained"),
        "relative_tolerance": _NATIVE_RELATIVE_TOLERANCE,
        "absolute_tolerance_ms": _NATIVE_ABSOLUTE_TOLERANCE_MS,
        "item_pass": (
            "every_retained_sample_formal_within_tolerance_on_duration_available_items"),
        "status": (
            "source_status_authoritative; failed stays failed; partial/inconclusive "
            "map to incomplete; item-only within never becomes passed"),
        "coverage": (
            "source coverage, truncated flag, partial flag, window and source identity "
            "are retained"),
        "aggregation": "no_median_no_mean_no_family_average",
        "excluded_evidence": [
            "historical_absolute_15us_pass",
            "conditional_relative_15pct_pass",
            "in_sample_reproduction",
            "mean_reproduction",
            "training_sample_resplit",
        ],
        "prediction_adjustment": "none",
        "interpretation": "independent_native_slice_only; not_a_heldout_generalization_claim",
    }


def _native_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_must_be_finite_number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name}_must_be_finite_number")
    return value


def _native_positive(value, name):
    value = _native_number(value, name)
    if value <= 0:
        raise ValueError(f"{name}_must_be_positive")
    return value


def _native_bool(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name}_must_be_boolean")
    return value


def _native_mapping(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name}_requires_object")
    return value


def _native_unknown_keys(value, allowed, name):
    for key in value:
        if key not in allowed:
            raise ValueError(f"{name}_unknown_field:{key}")


def _native_close(left, right):
    return math.isclose(left, right, rel_tol=_NATIVE_FLOAT_TOLERANCE,
                        abs_tol=_NATIVE_ABSOLUTE_EPSILON)


def _native_within(relative):
    return abs(relative) <= _NATIVE_RELATIVE_TOLERANCE + _NATIVE_ABSOLUTE_EPSILON


def _native_diagnostic_tolerance(tolerance):
    """Keep diagnostic values beside, never inside, the formal criterion."""
    raw = tolerance.get("diagnostic")
    if raw is None:
        return None
    diagnostic = _native_mapping(raw, "native_comparison.tolerance.diagnostic")
    _native_unknown_keys(
        diagnostic, {"relative", "absolute_ms", "denominator", "application"},
        "native_comparison.tolerance.diagnostic")
    relative = _native_number(
        diagnostic.get("relative"), "native_comparison.tolerance.diagnostic.relative")
    absolute = _native_number(
        diagnostic.get("absolute_ms", 0.0),
        "native_comparison.tolerance.diagnostic.absolute_ms")
    if relative < 0 or absolute < 0:
        raise ValueError("native_comparison_diagnostic_tolerance_must_be_nonnegative")
    denominator = diagnostic.get("denominator", "observed_ms")
    if denominator != "observed_ms":
        raise ValueError("native_comparison_diagnostic_denominator_must_be_observed_ms")
    application = diagnostic.get("application")
    if application is not None and not isinstance(application, str):
        raise ValueError("native_comparison_diagnostic_application_must_be_string")
    return {
        "relative": relative, "absolute_ms": absolute,
        "denominator": denominator, "application": application,
    }


def _native_tolerance(report):
    tolerance = _native_mapping(report.get("tolerance"),
                                "native_comparison.tolerance")
    _native_unknown_keys(
        tolerance, {"formal", "diagnostic", "applies_to"},
        "native_comparison.tolerance")
    formal = _native_mapping(tolerance.get("formal"),
                             "native_comparison.tolerance.formal")
    _native_unknown_keys(
        formal, {"relative", "absolute_ms", "denominator", "policy"},
        "native_comparison.tolerance.formal")
    relative = _native_number(
        formal.get("relative"), "native_comparison.tolerance.formal.relative")
    if not _native_close(relative, _NATIVE_RELATIVE_TOLERANCE):
        raise ValueError("native_comparison_formal_relative_tolerance_must_stay_0.10")
    absolute = _native_number(
        formal.get("absolute_ms", 0.0),
        "native_comparison.tolerance.formal.absolute_ms")
    if not _native_close(absolute, _NATIVE_ABSOLUTE_TOLERANCE_MS):
        raise ValueError("native_comparison_formal_absolute_tolerance_must_stay_0.0")
    if formal.get("denominator") != "observed_ms":
        raise ValueError("native_comparison_formal_denominator_must_be_observed_ms")
    policy = formal.get("policy")
    if policy is not None and not isinstance(policy, str):
        raise ValueError("native_comparison_formal_policy_must_be_string")
    if tolerance.get("applies_to") != "paired_comparable_items_only":
        raise ValueError("native_comparison_scope_must_be_paired_comparable_items")
    return ({
        "relative": relative, "absolute_ms": absolute,
        "denominator": "observed_ms", "policy": policy,
    }, _native_diagnostic_tolerance(tolerance))


def _native_coverage(report):
    coverage = _native_mapping(report.get("coverage"),
                               "native_comparison.coverage")
    truncated = _native_bool(coverage.get("truncated"),
                             "native_comparison.coverage.truncated")
    partial = _native_bool(coverage.get("partial_coverage"),
                           "native_comparison.coverage.partial_coverage")
    window = coverage.get("window")
    if window is not None and not isinstance(window, Mapping):
        raise ValueError("native_comparison_coverage_window_must_be_object_or_null")
    windows = coverage.get("windows")
    if windows is not None and not isinstance(windows, (list, tuple)):
        raise ValueError("native_comparison_coverage_windows_must_be_list_or_null")
    source_identity = coverage.get("source_identity")
    if source_identity is not None and not isinstance(source_identity, Mapping):
        raise ValueError("native_comparison_source_identity_must_be_object_or_null")
    observation_counts = coverage.get("observation_counts")
    if observation_counts is not None and not isinstance(observation_counts, Mapping):
        raise ValueError("native_comparison_observation_counts_must_be_object_or_null")
    reconciliation = _native_mapping(
        coverage.get("reconciliation"), "native_comparison.coverage.reconciliation")
    reconciled = _native_bool(
        reconciliation.get("reconciled"),
        "native_comparison.coverage.reconciliation.reconciled")
    subset = _native_bool(
        reconciliation.get("subset"),
        "native_comparison.coverage.reconciliation.subset")
    reconciliation_truncated = _native_bool(
        reconciliation.get("truncated"),
        "native_comparison.coverage.reconciliation.truncated")
    if reconciliation_truncated is not truncated:
        raise ValueError("native_comparison_coverage_truncated_flags_differ")
    if not reconciled and not partial:
        raise ValueError("native_comparison_unreconciled_coverage_cannot_pass")
    if truncated and not partial:
        raise ValueError("native_comparison_truncated_coverage_must_be_partial")
    scope = reconciliation.get("scope")
    if scope is not None and not isinstance(scope, Mapping):
        raise ValueError("native_comparison_reconciliation_scope_must_be_object_or_null")
    parent_capture = reconciliation.get("parent_capture")
    if parent_capture is not None and not isinstance(parent_capture, str):
        raise ValueError("native_comparison_parent_capture_must_be_string_or_null")
    for key in ("declared_window_counts", "actual_window_counts"):
        values = reconciliation.get(key)
        if values is not None:
            if not isinstance(values, Mapping):
                raise ValueError(f"native_comparison_reconciliation_{key}_must_be_object")
            for window_id, count in values.items():
                if not isinstance(window_id, str) or isinstance(count, bool) \
                        or not isinstance(count, int) or count < 0:
                    raise ValueError(f"native_comparison_reconciliation_{key}_invalid")
    for key in ("missing_declared_window_ids", "missing_window_ids",
                "unexpected_window_ids"):
        values = reconciliation.get(key)
        if values is not None and not isinstance(values, (list, tuple)):
            raise ValueError(f"native_comparison_reconciliation_{key}_must_be_list")
    for key in ("declared_events", "actual_events", "missing_events",
                "unexpected_events"):
        value = reconciliation.get(key)
        if value is not None and (isinstance(value, bool)
                                  or not isinstance(value, int) or value < 0):
            raise ValueError(f"native_comparison_reconciliation_{key}_invalid")
    return {
        "truncated": truncated, "partial_coverage": partial,
        "window": deepcopy(window), "windows": deepcopy(windows),
        "source_identity": deepcopy(source_identity),
        "observation_counts": deepcopy(observation_counts),
        "reconciliation": deepcopy(reconciliation),
        "reconciled": reconciled, "subset": subset,
        "scope": deepcopy(scope), "parent_capture": parent_capture,
    }


def _native_sample(sample, predicted, index, diagnostic):
    if not isinstance(sample, Mapping):
        raise ValueError(f"native_sample[{index}]_requires_object")
    name = f"native_sample[{index}]"
    observed = _native_positive(sample.get("observed_ms"), name + ".observed_ms")
    relative = predicted / observed - 1
    reported = _native_number(sample.get("relative_error"), name + ".relative_error")
    if not _native_close(relative, reported):
        raise ValueError(f"{name}_relative_error_inconsistent_with_values")
    formal_allowed = _native_number(
        sample.get("formal_allowed_error_ms"), name + ".formal_allowed_error_ms")
    if not _native_close(formal_allowed, _NATIVE_RELATIVE_TOLERANCE * observed):
        raise ValueError(f"{name}_formal_tolerance_is_not_fixed_10pct_of_observed")
    signed = _native_number(
        sample.get("signed_residual_ms", predicted - observed), name + ".signed_residual_ms")
    if not _native_close(signed, predicted - observed):
        raise ValueError(f"{name}_signed_residual_inconsistent_with_values")
    absolute = _native_number(
        sample.get("absolute_residual_ms", abs(predicted - observed)),
        name + ".absolute_residual_ms")
    if not _native_close(absolute, abs(predicted - observed)):
        raise ValueError(f"{name}_absolute_residual_inconsistent_with_values")
    within = _native_within(relative)
    flag = sample.get("within_tolerance")
    if not isinstance(flag, bool) or flag is not within:
        raise ValueError(f"{name}_verdict_must_follow_formal_observed_denominator")
    if "denominator" in sample and sample["denominator"] != "observed_ms":
        raise ValueError(f"{name}_denominator_must_be_observed_ms")
    result = deepcopy(sample)
    result.update({
        "observed_ms": observed, "relative_error": relative,
        "signed_residual_ms": predicted - observed,
        "absolute_residual_ms": abs(predicted - observed),
        "formal_allowed_error_ms": formal_allowed, "within_tolerance": within,
    })
    has_diagnostic = ("diagnostic_allowed_error_ms" in sample
                      or "diagnostic_within" in sample)
    if diagnostic is None:
        if has_diagnostic:
            raise ValueError(f"{name}_has_diagnostic_fields_without_diagnostic_tolerance")
        return result
    if "diagnostic_allowed_error_ms" not in sample or "diagnostic_within" not in sample:
        raise ValueError(f"{name}_diagnostic_fields_missing")
    diagnostic_allowed = _native_number(
        sample["diagnostic_allowed_error_ms"], name + ".diagnostic_allowed_error_ms")
    expected = diagnostic["absolute_ms"] + diagnostic["relative"] * observed
    if not _native_close(diagnostic_allowed, expected):
        raise ValueError(f"{name}_diagnostic_allowed_error_inconsistent_with_tolerance")
    diagnostic_within = _native_bool(
        sample["diagnostic_within"], name + ".diagnostic_within")
    expected_diagnostic = abs(predicted - observed) <= diagnostic_allowed
    if diagnostic_within is not expected_diagnostic:
        raise ValueError(f"{name}_diagnostic_verdict_inconsistent_with_allowed_error")
    result.update({
        "diagnostic_allowed_error_ms": diagnostic_allowed,
        "diagnostic_within": diagnostic_within,
    })
    return result


def _native_item_common(item, index):
    """Validate the facts shared by work-only and duration-scored items."""
    if not isinstance(item, Mapping):
        raise ValueError(f"native_item[{index}]_requires_object")
    name = f"native_item[{index}]"
    item_id = item.get("item_id")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(f"{name}_requires_item_id")
    identity = item.get("identity")
    boundary = item.get("boundary")
    if not isinstance(identity, Mapping):
        raise ValueError(f"{name}_requires_identity")
    if not identity.get("native_operator") or not identity.get("native_abi_sha256"):
        raise ValueError(f"{name}_requires_native_operator_and_abi")
    if not isinstance(boundary, Mapping) or not boundary.get("kind"):
        raise ValueError(f"{name}_requires_boundary_kind")
    identity_abi = identity["native_abi_sha256"]
    observation_abi = item.get("observation_abi_digest")
    prediction_abi = item.get("prediction_abi_digest")
    for field, value in (("observation_abi_digest", observation_abi),
                         ("prediction_abi_digest", prediction_abi)):
        if value is not None:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name}_{field}_must_be_nonempty_string")
            if value != identity_abi:
                raise ValueError(f"{name}_{field}_differs_from_identity_abi")
    if observation_abi is not None and prediction_abi is not None \
            and observation_abi != prediction_abi:
        raise ValueError(f"{name}_observation_and_prediction_abi_differ")
    comparison = item.get("comparison")
    if not isinstance(comparison, Mapping):
        raise ValueError(f"{name}_requires_comparison")
    status = item.get("status")
    if status not in (*_NATIVE_SCORED_STATUSES, _NATIVE_WORK_ONLY_STATUS):
        raise ValueError(f"{name}_unknown_status:{status}")
    duration_available = comparison.get("duration_available")
    if duration_available is not None and not isinstance(duration_available, bool):
        raise ValueError(f"{name}_comparison.duration_available_must_be_boolean")
    observed = _native_positive(item.get("observed_ms"), name + ".observed_ms")
    gaps = item.get("condition_gaps") or ()
    if not isinstance(gaps, (list, tuple)):
        raise ValueError(f"{name}_condition_gaps_must_be_a_list")
    count = item.get("observation_samples")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(f"{name}_observation_samples_must_be_positive_integer")
    observation_ids = item.get("observation_ids")
    if observation_ids is not None and (
            not isinstance(observation_ids, (list, tuple))
            or len(observation_ids) != count):
        raise ValueError(f"{name}_observation_ids_inconsistent_with_observation_samples")
    selected_range = item.get("observation_range_ms")
    if selected_range is not None:
        if not isinstance(selected_range, (list, tuple)) or len(selected_range) != 2:
            raise ValueError(f"{name}_observation_range_ms_must_be_two_numbers")
        low = _native_number(selected_range[0], name + ".observation_range_ms[0]")
        high = _native_number(selected_range[1], name + ".observation_range_ms[1]")
        if low > high:
            raise ValueError(f"{name}_observation_range_ms_is_reversed")
        selected_range = [low, high]
    return {
        "item": item, "name": name, "comparison": comparison, "status": status,
        "duration_available": duration_available, "observed": observed,
        "condition_gaps": list(gaps), "observation_samples": count,
        "observation_ids": observation_ids, "observation_range_ms": selected_range,
    }


def _native_work_only_item(common, _diagnostic):
    """Retain an ABI pair whose prediction has explicit work but no duration.

    No predicted duration, formal error, sample verdict or within/outside status
    is invented.  The caller excludes this item from the duration denominator.
    """
    item, name = common["item"], common["name"]
    comparison = common["comparison"]
    if common["duration_available"] is not False:
        raise ValueError(f"{name}_work_only_requires_duration_available_false")
    predicted = item.get("predicted_ms")
    if predicted is not None:
        raise ValueError(f"{name}_work_only_predicted_ms_must_stay_null")
    for field in ("relative_error", "formal_allowed_error_ms", "denominator",
                  "within_tolerance", "signed_residual_ms", "absolute_residual_ms",
                  "worst_absolute_relative_error", "outside_sample_ids"):
        if comparison.get(field) is not None and comparison.get(field) not in ((), []):
            raise ValueError(f"{name}_work_only_forbids_scored_field:{field}")
    raw_samples = comparison.get("sample_verdicts")
    if not isinstance(raw_samples, (list, tuple)):
        raise ValueError(f"{name}_work_only_sample_verdicts_must_be_a_list")
    if raw_samples:
        raise ValueError(f"{name}_work_only_must_not_retain_sample_verdicts")
    work = comparison.get("work")
    if not isinstance(work, Mapping) or not work:
        raise ValueError(f"{name}_work_only_requires_explicit_work")
    cube = work.get("cube")
    if cube is not None:
        if not isinstance(cube, Mapping):
            raise ValueError(f"{name}_work_only_cube_work_must_be_object")
        macs = _native_number(cube.get("macs"), name + ".comparison.work.cube.macs")
        if macs < 0:
            raise ValueError(f"{name}_work_only_cube_macs_must_be_nonnegative")
    reported_duration = comparison.get("observation_duration_ms")
    if reported_duration is not None:
        reported_duration = _native_number(
            reported_duration, name + ".comparison.observation_duration_ms")
        if not _native_close(reported_duration, common["observed"]):
            raise ValueError(f"{name}_work_only_observation_duration_inconsistent")
    result = deepcopy(item)
    result.update({
        "observed_ms": common["observed"], "predicted_ms": None,
        "duration_available": False, "status": _NATIVE_WORK_ONLY_STATUS,
        "samples": [], "sample_verdicts": [], "work": deepcopy(work),
        "observation_samples": common["observation_samples"],
        "observation_range_ms": common["observation_range_ms"],
        "condition_gaps": common["condition_gaps"],
    })
    return result


def _native_scored_item(common, diagnostic):
    """Validate a numeric prediction only after a real duration is available."""
    item, name = common["item"], common["name"]
    comparison = common["comparison"]
    if common["duration_available"] is False:
        raise ValueError(f"{name}_scored_item_requires_duration_available")
    predicted = _native_number(item.get("predicted_ms"), name + ".predicted_ms")
    if predicted < 0:
        raise ValueError(f"{name}.predicted_ms_must_be_nonnegative")
    observed = common["observed"]
    relative = predicted / observed - 1
    reported = _native_number(
        comparison.get("relative_error"), name + ".comparison.relative_error")
    if not _native_close(relative, reported):
        raise ValueError(f"{name}_comparison_relative_error_inconsistent_with_values")
    formal_allowed = _native_number(
        comparison.get("formal_allowed_error_ms"),
        name + ".comparison.formal_allowed_error_ms")
    if not _native_close(formal_allowed, _NATIVE_RELATIVE_TOLERANCE * observed):
        raise ValueError(f"{name}_formal_tolerance_is_not_fixed_10pct_of_observed")
    if comparison.get("denominator") != "observed_ms":
        raise ValueError(f"{name}_denominator_must_be_observed_ms")
    formal_policy = comparison.get("formal_policy")
    if formal_policy is not None and not isinstance(formal_policy, str):
        raise ValueError(f"{name}_formal_policy_must_be_string")
    raw_samples = comparison.get("sample_verdicts")
    if not isinstance(raw_samples, (list, tuple)) or not raw_samples:
        raise ValueError(f"{name}_requires_retained_sample_verdicts")
    samples = [_native_sample(sample, predicted, position, diagnostic)
               for position, sample in enumerate(raw_samples)]
    within = all(sample["within_tolerance"] for sample in samples)
    flag = comparison.get("within_tolerance")
    if not isinstance(flag, bool) or flag is not within:
        raise ValueError(f"{name}_verdict_cannot_hide_a_formal_sample_failure")
    signed = _native_number(comparison.get("signed_residual_ms", predicted - observed),
                            name + ".comparison.signed_residual_ms")
    if not _native_close(signed, predicted - observed):
        raise ValueError(f"{name}_comparison_signed_residual_inconsistent_with_values")
    absolute = _native_number(
        comparison.get("absolute_residual_ms", abs(predicted - observed)),
        name + ".comparison.absolute_residual_ms")
    if not _native_close(absolute, abs(predicted - observed)):
        raise ValueError(f"{name}_comparison_absolute_residual_inconsistent_with_values")
    worst = max(abs(sample["relative_error"]) for sample in samples)
    if "worst_absolute_relative_error" in comparison:
        reported_worst = _native_number(
            comparison["worst_absolute_relative_error"],
            name + ".comparison.worst_absolute_relative_error")
        if not _native_close(reported_worst, worst):
            raise ValueError(f"{name}_worst_sample_error_inconsistent")
    status = common["status"]
    expected = "within_tolerance" if within else "outside_tolerance"
    if status != expected:
        raise ValueError(f"{name}_status_must_match_every_formal_sample_verdict")
    observed_samples = [sample["observed_ms"] for sample in samples]
    low, high = min(observed_samples), max(observed_samples)
    selected_range = common["observation_range_ms"]
    if selected_range is not None and (
            not _native_close(selected_range[0], low)
            or not _native_close(selected_range[1], high)):
        raise ValueError(f"{name}_observation_range_inconsistent_with_samples")
    if not (low - _NATIVE_ABSOLUTE_EPSILON <= observed
            <= high + _NATIVE_ABSOLUTE_EPSILON):
        raise ValueError(f"{name}_observed_ms_outside_retained_sample_range")
    count = common["observation_samples"]
    if count != len(samples):
        raise ValueError(f"{name}_observation_samples_inconsistent_with_verdicts")
    observation_ids = common["observation_ids"]
    if observation_ids is not None and len(observation_ids) != len(samples):
        raise ValueError(f"{name}_observation_ids_inconsistent_with_verdicts")
    result = deepcopy(item)
    result.update({
        "observed_ms": observed, "predicted_ms": predicted,
        "duration_available": True,
        "relative_error": relative, "signed_residual_ms": predicted - observed,
        "absolute_residual_ms": abs(predicted - observed),
        "formal_allowed_error_ms": formal_allowed, "formal_policy": formal_policy,
        "within_tolerance": within, "observation_range_ms": [low, high],
        "observation_samples": count, "samples": samples,
        "condition_gaps": common["condition_gaps"], "status": status,
    })
    return result


def _native_item(item, index, diagnostic):
    common = _native_item_common(item, index)
    if common["status"] == _NATIVE_WORK_ONLY_STATUS:
        return _native_work_only_item(common, diagnostic)
    return _native_scored_item(common, diagnostic)


def _native_count(counts, key):
    value = counts.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"native_comparison_count_mismatch:{key}")
    return value


def _native_counts(report, items, unpaired_observations, unpaired_predictions,
                   ambiguous, coverage):
    counts = report.get("counts")
    if not isinstance(counts, Mapping):
        raise ValueError("native_comparison_requires_counts")
    within = sum(1 for item in items if item["status"] == "within_tolerance")
    outside = sum(1 for item in items if item["status"] == "outside_tolerance")
    work_only = sum(1 for item in items if item["status"] == _NATIVE_WORK_ONLY_STATUS)
    if within + outside + work_only != len(items):
        raise ValueError("native_comparison_unknown_item_status")
    duration_denominator = within + outside
    gaps = sum(bool(item["condition_gaps"]) for item in items)
    expected = {
        "paired": len(items), "within_tolerance": within, "outside_tolerance": outside,
        "unpaired_observations": len(unpaired_observations),
        "unpaired_predictions": len(unpaired_predictions), "ambiguous": len(ambiguous),
        "items_with_condition_gaps": gaps,
    }
    for key, value in expected.items():
        if _native_count(counts, key) != value:
            raise ValueError(f"native_comparison_count_mismatch:{key}")
    if counts.get("work_only") is None:
        if work_only:
            raise ValueError("native_comparison_count_mismatch:work_only")
    elif _native_count(counts, "work_only") != work_only:
        raise ValueError("native_comparison_count_mismatch:work_only")
    for key in ("truncated", "partial_coverage"):
        flag = _native_bool(counts.get(key),
                            f"native_comparison.counts.{key}")
        if flag is not coverage[key]:
            raise ValueError(f"native_comparison_count_mismatch:{key}")
    samples = sum(len(item["samples"]) for item in items)
    observed_rows = sum(int(item["observation_samples"]) for item in items)
    if "observation_verdicts" in counts \
            and _native_count(counts, "observation_verdicts") != samples:
        raise ValueError("native_comparison_count_mismatch:observation_verdicts")
    sample_within = sum(1 for item in items for sample in item["samples"]
                        if sample["within_tolerance"])
    if "observations_within_tolerance" in counts \
            and _native_count(counts, "observations_within_tolerance") != sample_within:
        raise ValueError("native_comparison_count_mismatch:observations_within_tolerance")
    if "observations_outside_tolerance" in counts \
            and _native_count(counts, "observations_outside_tolerance") \
            != samples - sample_within:
        raise ValueError("native_comparison_count_mismatch:observations_outside_tolerance")
    if "observations" in counts and all(
            isinstance(row.get("observation_samples"), int)
            and not isinstance(row.get("observation_samples"), bool)
            for row in unpaired_observations):
        expected_observed = observed_rows + sum(
            int(row["observation_samples"]) for row in unpaired_observations)
        if _native_count(counts, "observations") != expected_observed:
            raise ValueError("native_comparison_count_mismatch:observations")
    for key in ("family_aggregates_used", "name_only_matches", "fused_members_allocated"):
        if counts.get(key, 0) != 0:
            raise ValueError(f"native_comparison_forbids:{key}")
    return {
        "paired_items": len(items), "within": within, "outside": outside,
        "work_only": work_only,
        "duration_denominator_items": duration_denominator,
        "condition_gap_items_in_denominator": gaps,
        "observation_samples": samples, "observed_rows": observed_rows,
        "unpaired_observations": len(unpaired_observations),
        "unpaired_predictions": len(unpaired_predictions),
        "ambiguous": len(ambiguous),
        "truncated": coverage["truncated"],
        "partial_coverage": coverage["partial_coverage"],
        "source_counts": deepcopy(dict(counts)),
    }


def native_acceptance(report) -> dict:
    """Validate a profiler native comparison and retain its formal result.

    The comparison owner performs pairing and the formal comparison once.  This
    consumer reads ``tolerance.formal`` and each sample's
    ``formal_allowed_error_ms``; it never substitutes diagnostic tolerance,
    re-prices, medians, or promotes an item-only all-within set to passed.
    """
    if not isinstance(report, Mapping):
        raise ValueError("native_comparison_requires_object")
    if report.get("schema") != NATIVE_COMPARISON_SCHEMA:
        raise ValueError("native_comparison_schema_mismatch")
    source_status = report.get("status")
    if not isinstance(source_status, str) or not source_status:
        raise ValueError("native_comparison_requires_status")
    formal, diagnostic = _native_tolerance(report)
    coverage = _native_coverage(report)
    boundary_policy = _native_mapping(
        report.get("boundary_policy"), "native_comparison.boundary_policy")
    interpretation = report.get("interpretation")
    if interpretation is not None and not isinstance(interpretation, str):
        raise ValueError("native_comparison_interpretation_must_be_string")
    raw_items = report.get("items")
    if not isinstance(raw_items, (list, tuple)):
        raise ValueError("native_comparison_requires_items")
    items = [_native_item(item, index, diagnostic)
             for index, item in enumerate(raw_items)]
    seen = set()
    for item in items:
        if item["item_id"] in seen:
            raise ValueError("native_comparison_duplicate_item_id")
        seen.add(item["item_id"])
    for key in ("unpaired_observations", "unpaired_predictions", "ambiguous",
                "failure_matrix"):
        if not isinstance(report.get(key, ()), (list, tuple)):
            raise ValueError(f"native_comparison_{key}_must_be_a_list")
    non_claims = report.get("non_claims", ())
    if not isinstance(non_claims, (list, tuple)):
        raise ValueError("native_comparison_non_claims_must_be_a_list")
    unpaired_observations = deepcopy(list(report.get("unpaired_observations", ())))
    unpaired_predictions = deepcopy(list(report.get("unpaired_predictions", ())))
    ambiguous = deepcopy(list(report.get("ambiguous", ())))
    failure_matrix = deepcopy(list(report.get("failure_matrix", ())))
    denominators = _native_counts(
        report, items, unpaired_observations, unpaired_predictions, ambiguous, coverage)
    for row in failure_matrix:
        if not isinstance(row, Mapping) or not isinstance(row.get("item_ids", ()),
                                                          (list, tuple)):
            raise ValueError("native_comparison_failure_matrix_row_invalid")
    outside = denominators["outside"]
    if outside and not failure_matrix:
        raise ValueError("native_comparison_outside_items_require_failure_matrix")
    if denominators["work_only"] and not any(
            row.get("reason") == _NATIVE_WORK_ONLY_REASON
            for row in failure_matrix):
        raise ValueError("native_comparison_work_only_reason_missing")
    if coverage["partial_coverage"] and not any(
            row.get("reason") == "partial_observation_window_coverage"
            for row in failure_matrix):
        raise ValueError("native_comparison_partial_coverage_reason_missing")
    expected_source_status = (
        "failed" if outside else
        "inconclusive" if (not items or denominators["work_only"]
                            or unpaired_observations
                            or unpaired_predictions or ambiguous) else
        "partial" if coverage["partial_coverage"] else "passed")
    if source_status != expected_source_status:
        raise ValueError(
            "native_comparison_source_status_inconsistent_with_counts:"
            f"{source_status}!={expected_source_status}")
    status = {"failed": "failed", "passed": "passed"}.get(source_status, "incomplete")
    return {
        "schema": NATIVE_ACCEPTANCE_POLICY,
        "source_schema": report["schema"],
        "source_status": source_status,
        "status": status,
        "policy": native_acceptance_policy(),
        "tolerance": {
            "formal": formal, "diagnostic": diagnostic,
            "applies_to": "paired_comparable_items_only",
        },
        "coverage": deepcopy(coverage),
        "truncated": coverage["truncated"],
        "partial_coverage": coverage["partial_coverage"],
        "window": deepcopy(coverage["window"]),
        "windows": deepcopy(coverage["windows"]),
        "source_identity": deepcopy(coverage["source_identity"]),
        "observation_counts": deepcopy(coverage["observation_counts"]),
        "coverage_reconciled": coverage["reconciled"],
        "subset_scope": deepcopy(coverage["scope"]),
        "parent_capture": coverage["parent_capture"],
        "boundary_policy": deepcopy(boundary_policy),
        "source_interpretation": interpretation,
        "counts": {**denominators, "source_status": source_status},
        "source_counts": denominators["source_counts"],
        "items": items,
        "unpaired_observations": unpaired_observations,
        "unpaired_predictions": unpaired_predictions,
        "ambiguous": ambiguous,
        "failure_matrix": failure_matrix,
        "non_claims": list(non_claims),
        "interpretation": [
            "ABI/boundary pairing is retained for every item, including work-only items.",
            "Only items with a real predicted duration enter the formal duration denominator; "
            "work-only items keep observation and work facts and no numerical error.",
            "Every retained sample must pass its formal allowed error; neither median nor family mean can turn an outside item into a pass.",
            "Diagnostic tolerance is retained beside the formal verdict and never changes that verdict.",
            "Source partial/truncated/inconclusive coverage maps to incomplete; an item-only all-within set cannot become passed.",
            "Legacy absolute_15us, in-sample and mean reproductions are not accepted as this round's evidence.",
        ],
    }


import argparse
from collections import Counter
from collections.abc import Mapping
import json
import math
from pathlib import Path

from .prediction.graph_analysis import graph_costs
from .prediction.operator_analysis import operator_costs, target_graphs
from .prediction.program import RecipeBook
from .prediction.structure import parse_case


SCHEMA = "mindie-sim.cohort-comparison/1"
MAX_METADATA_BYTES = 32 * 1024 * 1024
MAX_COHORTS = 10_000


def _read(value):
    if isinstance(value, (str, Path)):
        with Path(value).open("rb") as stream:
            data = stream.read(MAX_METADATA_BYTES + 1)
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("cohort_metadata_exceeds_32_mib")
        value = json.loads(data)
    if not isinstance(value, Mapping):
        raise ValueError("cohort_metadata_requires_object")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _member_ids(target, kind):
    return target["node_ids"] if kind == "subgraph" else [target["node_id"]]


def _check_placement(cohort, selected, structure):
    placement = cohort.get("placement", {})
    symmetric = placement.get("basis") == "single_rank_symmetric_dp"
    replicas = sorted({target.get("replica") for target in selected}, key=str)
    if not symmetric:
        if len(replicas) != 1 or replicas[0] != cohort.get("replica"):
            raise ValueError("cohort_target_placement_mismatch")
        return 1
    execution = structure["execution"]
    if (execution.get("replica_axis") != "data_parallel"
            or set(replicas) != set(execution.get("replicas", ()))
            or placement.get("replica_candidates") != replicas or len(replicas) < 2
            or cohort.get("replica") is not None
            or placement.get("replica_binding_implied") is not False
            or not cohort.get("rank_id") or not cohort.get("device_id")):
        raise ValueError("cohort_symmetric_dp_placement_invalid")
    local = {}
    for target in selected:
        key = target.get("subgraph_id", target.get("node_id"))
        prefix = f"rank:{target['replica']}/"
        if not key.startswith(prefix):
            raise ValueError("cohort_symmetric_dp_target_identity_invalid")
        local.setdefault(target["replica"], set()).add(
            (key[len(prefix):], target.get("layer"), tuple(target.get("layer_ids", ()))))
    if any(value != local[replicas[0]] for value in local.values()):
        raise ValueError("cohort_symmetric_dp_candidates_incomplete")
    if cohort.get("declared_candidate_instance_count") != len(selected):
        raise ValueError("cohort_candidate_instance_count_mismatch")
    return len(replicas)


def _check_cohort(cohort, window, structure, targets):
    if cohort.get("schema_version") != "mindie-profile-cohort/1":
        raise ValueError("cohort_schema_mismatch")
    if cohort.get("graph_id") != structure["graph_id"]:
        raise ValueError("cohort_graph_id_mismatch")
    ids = cohort.get("target_ids", ())
    if not ids or len(set(ids)) != len(ids) or any(key not in targets for key in ids):
        raise ValueError("cohort_target_ids_unknown_or_duplicated")
    if str(cohort.get("rank_id")) != str(window["rank_id"]):
        raise ValueError("cohort_capture_rank_mismatch")
    kind = cohort.get("target_kind")
    if kind not in {"node", "subgraph"}:
        raise ValueError("cohort_target_kind_invalid")
    selected = [targets[key] for key in ids]
    replica_count = _check_placement(cohort, selected, structure)
    if cohort.get("declared_instance_count") != len(ids) // replica_count:
        raise ValueError("cohort_declared_instance_count_mismatch")
    if _canonical(cohort.get("resource_boundary")) != _canonical(
            selected[0].get("resource_boundary")):
        raise ValueError("cohort_resource_boundary_mismatch")
    if _canonical(cohort.get("contract_scope")) != _canonical(
            selected[0].get("contract_scope")):
        raise ValueError("cohort_contract_scope_mismatch")
    placement = (cohort.get("replica"), cohort.get("stage"), cohort.get("phase"))
    seen = set()
    for target in selected:
        if target.get("event_matchable") is False:
            raise ValueError("cohort_target_is_internal_to_selected_native_fusion")
        if ("subgraph_id" in target) != (kind == "subgraph"):
            raise ValueError("cohort_target_boundaries_differ")
        if tuple(target.get(key) for key in ("stage", "phase")) != placement[1:]:
            raise ValueError("cohort_target_placement_mismatch")
        if (not cohort.get("physical_signature")
                or target.get("physical_signature") != cohort["physical_signature"]
                or _canonical(target.get("physical_work")) !=
                _canonical(cohort.get("physical_work_per_instance"))
                or target.get("contract") is None
                or _canonical(target["contract"]) != _canonical(cohort.get("contract"))):
            raise ValueError("cohort_physical_or_tensor_contract_mismatch")
        coverage = target.get("coverage")
        is_composite_shared = isinstance(coverage, Mapping) and bool(
            coverage.get("composite_shared") or coverage.get("kind") == "composite_shared"
        )
        if isinstance(coverage, Mapping) and coverage.get("kind") not in {
            "complete_subgraph", "complete_semantic_subgraph", "composite_shared",
        }:
            raise ValueError("cohort_incomplete_native_boundary")
        members = set(_member_ids(target, kind))
        if members & seen and not is_composite_shared:
            raise ValueError("cohort_candidate_boundaries_overlap")
        seen.update(members)
    expected_phase = structure["execution"]["phase"]
    draft_prepare = bool(structure["execution"].get("draft_prepare", False))
    valid_phases = {expected_phase}
    if expected_phase == "decode":
        valid_phases.add("draft")
    elif expected_phase == "prefill" and draft_prepare:
        valid_phases.update({"draft", "draft_prefill"})
    if (replica_count == 1 and placement[0] is None
            or placement[1] is None or placement[2] not in valid_phases):
        raise ValueError("cohort_placement_or_phase_unspecified")
    count, complete, clipped = (cohort.get(key) for key in (
        "event_count", "complete_event_count", "clipped_event_count"))
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (count, complete, clipped)) or count != complete + clipped):
        raise ValueError("cohort_event_counts_inconsistent")
    return selected


def _check_symmetric_hardware(case, cohort):
    placement = cohort.get("placement", {})
    if placement.get("basis") != "single_rank_symmetric_dp":
        return
    if cohort["physical_work_per_instance"].get("link_bytes"):
        raise ValueError("symmetric_dp_transport_requires_explicit_replica")
    parallel = case.runtime.parallel
    width = parallel.tp * parallel.pcp
    declarations = []
    for replica in placement["replica_candidates"]:
        start = (replica * parallel.pp + cohort["stage"]) * width
        declarations.append([
            {key: value for key, value in die.to_dict().items()
             if key not in {"rank", "node", "package"}}
            for die in case.machine.dies[start:start + width]
        ])
    if any(value != declarations[0] for value in declarations[1:]):
        raise ValueError("symmetric_dp_hardware_differs")


def _program_ids(target, kind):
    boundary = target.get("resource_boundary") if kind == "subgraph" else None
    if boundary:
        identifiers = boundary.get("operator_ids", ())
        if not identifiers:
            raise ValueError("cohort_resource_boundary_has_no_operators")
        prefix = f"rank:{target['replica']}/"
        return [key if key.startswith(prefix) else prefix + key for key in identifiers], boundary
    return _member_ids(target, kind), None


def _predict(case, target, kind, by_id, book, *, evaluation=None):
    raw_identifiers, selected_boundary = _program_ids(target, kind)
    identifiers = list(dict.fromkeys(raw_identifiers))
    if any(key not in by_id for key in identifiers):
        raise ValueError("cohort_target_physical_program_unavailable")
    ops = tuple(by_id[key] for key in identifiers)
    replica = target["replica"]
    physical_work = target["physical_work"]
    one_program = (len(ops) == 1 and physical_work.get("launches") == 1
                   and ops[0].source_boundary != "call")
    if kind == "node" and len(ops) == 1:
        result = operator_costs(case, ops, recipes=book, replica=replica, evaluation=evaluation)
        if result["operator_occurrences"] != 1:
            raise ValueError("cohort_node_expands_to_multiple_device_programs")
        row = result["operators"][0]
        duration, serviced = row["duration_ms"], row["serviced"]
        parameters = row["basis"]
    else:
        local = {op.op_id for op in ops}
        external = {dep: 0 for op in ops for dep in op.deps if dep not in local}
        result = graph_costs(
            case, {replica: ops}, recipes=book, external_inputs=external,
            include_submission=False, isolated=False, trace_limit=0, evaluation=evaluation)
        duration = result["duration_ms"]
        serviced = {item["resource"]: item["work"] for item in result["resources"]}
        parameters = result["parameters"]
    return {
        "boundary": "semantic_device_program" if one_program else "unfused_subgraph_reference",
        "representative_target_id": target.get("subgraph_id", target.get("node_id")),
        "representative_operator_ids": identifiers,
        "representative_selection_basis": "equivalent_declared_program; not_capture_occurrence_binding",
        "per_call_ms": duration, "physical_work_per_instance": physical_work,
        "serviced": serviced, "parameters": parameters,
        "resource_boundary": selected_boundary,
        "single_device_program": one_program,
        "assumptions": [
            "External inputs are ready; no host submission or unrelated graph contention is simulated.",
            "The case supplies work, precision, layout and resource bindings independently of measurements.",
            "Unobserved fields excluded by the tensor contract remain declared conditions.",
        ],
    }


def _error(predicted, observed):
    return {"signed_residual_ms": predicted - observed,
            "absolute_residual_ms": abs(predicted - observed),
            "relative_error": (predicted / observed - 1) if observed > 0 else None}


def _comparison(cohort, prediction):
    scope = cohort.get("contract_scope")
    coverage = scope.get("prediction_coverage", {}) if isinstance(scope, Mapping) else {}
    if coverage.get("kind") == "partial_program":
        return "reference_only", coverage["reason"], None
    if not prediction["single_device_program"]:
        return "reference_only", "unfused_subgraph_is_not_the_observed_native_kernel", None
    if cohort["clipped_event_count"]:
        return "unscored", "window_contains_clipped_events", None
    if not cohort["complete_event_count"]:
        return "unscored", "no_complete_events", None
    duration = cohort.get("duration_ns", {})
    actual = duration.get("mean")
    if actual is None or not math.isfinite(actual) or actual <= 0:
        return "unscored", "complete_event_mean_unavailable", None
    forecast = prediction["per_call_ms"]
    observed_sum = cohort["activity_sum_ns"] / 1_000_000
    comparison = {
        "per_call_mean": _error(forecast, actual / 1_000_000),
        "activity_sum": _error(forecast * cohort["event_count"], observed_sum),
        "interpretation": "conditional_descriptive_comparison; no_occurrence_or_accuracy_certification",
    }
    if duration.get("p50") is not None:
        comparison["per_call_p50"] = _error(forecast, duration["p50"] / 1_000_000)
    return "compared", "same_declared_single_program_boundary", comparison


def compare_cohorts(case, mapping_report, *, primitive_table=None):
    """Price each homogeneous candidate set once without choosing a layer.

    The report is produced by ``mindie_profiler.structure_mapping``. The case
    must independently regenerate the report's complete structure identity.
    No profiler import or raw capture is required by this consumer.
    """
    report = _read(mapping_report)
    if "structure_mapping" in report:
        report = _read(report["structure_mapping"])
    if report.get("schema_version") != "mindie-profile-structure-mapping/1":
        raise ValueError("expected_profiler_structure_mapping_report")
    case = parse_case(_read(case) if isinstance(case, (str, Path, Mapping)) else case)
    execution = report.get("declared_execution", {})
    required = ("phase", "tokens", "context", "sequences", "scope")
    if any(key not in execution for key in required):
        raise ValueError("cohort_report_lacks_execution_geometry")
    geometry = {key: execution[key] for key in required}
    if "apply_body_defaults" in execution:
        geometry["apply_body_defaults"] = execution["apply_body_defaults"]
    if execution.get("draft_prepare"):
        geometry["draft_prepare"] = True
    if execution.get("media_tokens_per_sequence") is not None:
        geometry["media_tokens_per_sequence"] = execution["media_tokens_per_sequence"]
    manifest_geometry = dict(geometry)
    projection = execution.get("replica_projection") or {}
    selected_replicas = projection.get("selected")
    if selected_replicas:
        # Reproduce the export's projected digest before pricing the complete cohort.
        manifest_geometry["replicas"] = selected_replicas
    from .structure.physical import select_structure_targets
    wanted = {key for window in report.get("windows", ())
              for cohort in window.get("cohorts", ()) for key in cohort.get("target_ids", ())}
    manifest = select_structure_targets(
        case, wanted, **manifest_geometry,
        expand_calls=execution.get("expand_calls", True))
    if manifest["graph_id"] != report.get("graph_id"):
        raise ValueError("cohort_case_graph_id_mismatch:regenerate_mapping_for_this_case")
    if report.get("observed_phase") not in (None, geometry["phase"]):
        raise ValueError("cohort_observed_phase_mismatch")
    targets = manifest["targets"]
    # Validate candidates before releasing the export and building the executable DAG.
    checked, needed = {}, set()
    for window in report.get("windows", ()):
        for cohort in window.get("cohorts", ()):
            if len(checked) >= MAX_COHORTS:
                raise ValueError("cohort_comparison_exceeds_10000_groups")
            selected = _check_cohort(cohort, window, manifest, targets)
            checked[id(cohort)] = selected[0]
            identifiers, _ = _program_ids(selected[0], cohort["target_kind"])
            needed.update(identifiers)
    manifest = {key: manifest[key] for key in ("graph_id", "execution")}
    del targets
    selected = None
    graphs = target_graphs(case, **geometry)
    from .prediction.body_defaults import consume_target_declaration
    case = consume_target_declaration(case, graphs)
    book = RecipeBook.from_mapping(case.runtime.kernels)
    by_id = {f"rank:{replica}/{op.op_id}": op
             for replica, operators in graphs.items() for op in book.expand(operators)
             if f"rank:{replica}/{op.op_id}" in needed}
    del graphs
    from .prediction.evaluation import EvaluationContext
    eval_ctx = EvaluationContext.create(case, book, primitive_table=primitive_table)
    cache, windows, counts = {}, [], Counter()
    total = 0
    for window in report.get("windows", ()):
        comparisons = []
        for cohort in window.get("cohorts", ()):
            total += 1
            if total > MAX_COHORTS:
                raise ValueError("cohort_comparison_exceeds_10000_groups")
            selected = checked[id(cohort)]
            key = (cohort["physical_signature"], cohort["replica"], cohort["stage"], cohort["target_kind"])
            row = {
                "cohort_id": cohort["cohort_id"], "target_ids": cohort["target_ids"],
                "target_kind": cohort["target_kind"], "native_operator_type": cohort["native_operator_type"],
                "physical_signature": cohort["physical_signature"], "replica": cohort["replica"],
                "stage": cohort["stage"], "phase": cohort["phase"],
                "rank_id": cohort["rank_id"], "device_id": cohort.get("device_id"),
                "contract": cohort["contract"], "contract_scope": cohort.get("contract_scope"),
                "resource_boundary": cohort.get("resource_boundary"),
                "structural_classification": cohort.get("structural_classification"),
                "declared_instance_count": cohort["declared_instance_count"],
                "declared_candidate_instance_count": cohort.get(
                    "declared_candidate_instance_count", len(cohort["target_ids"])),
                "declared_layer_count": cohort["declared_layer_count"], "placement": cohort.get("placement"),
                "observed": {key: cohort.get(key) for key in (
                    "event_count", "complete_event_count", "clipped_event_count", "duration_ns",
                    "activity_sum_ns", "complete_activity_sum_ns", "busy_union_ns", "status_counts")},
                "multiplicity": dict(cohort["multiplicity"]),
                "occurrence_binding_implied": False, "prediction": None, "comparison": None,
            }
            try:
                _check_symmetric_hardware(case, cohort)
                if key not in cache:
                    cache[key] = _predict(case, selected, cohort["target_kind"], by_id, book, evaluation=eval_ctx)
                prediction = cache[key]
            except ValueError as exc:
                row.update(status="prediction_failed", reason=str(exc))
            else:
                # Equal-cost cohorts can still refer to disjoint layers.
                identifiers, boundary = _program_ids(selected, cohort["target_kind"])
                prediction = {
                    **prediction,
                    "representative_target_id": selected.get("subgraph_id", selected.get("node_id")),
                    "representative_operator_ids": identifiers, "resource_boundary": boundary,
                }
                scope = cohort.get("contract_scope")
                coverage = scope.get("prediction_coverage", {}) if isinstance(scope, Mapping) else {}
                if coverage.get("kind") == "partial_program":
                    prediction.update(boundary="partial_program_reference", prediction_coverage=coverage)
                per_call = prediction["per_call_ms"]
                row["prediction"] = {
                    **prediction,
                    "observed_call_count_total_ms": per_call * cohort["event_count"],
                    "per_declared_step_ms": per_call * cohort["declared_instance_count"],
                    "step_total_basis": "one_invocation_per_declared_instance; not_observed_step_count",
                    "instance_scope": cohort["multiplicity"].get("instance_scope", "single_declared_replica"),
                }
                steps = row["multiplicity"].get("steps_in_window")
                if steps is not None:
                    expected = steps * cohort["declared_instance_count"]
                    row["multiplicity"].update(
                        expected_calls=expected, call_count_difference=cohort["event_count"] - expected)
                status, reason, comparison = _comparison(cohort, prediction)
                row.update(status=status, reason=reason, comparison=comparison)
            row.update(accuracy_fields(row))
            counts[row["status"]] += 1
            comparisons.append(row)
        windows.append({
            "rank_id": window["rank_id"], "step": window.get("step"),
            "source_identity": window.get("source_identity"),
            "window": window["window"], "rank_scope": window.get("rank_scope"),
            "cohort_excluded_event_counts": window.get("cohort_excluded_event_counts", {}),
            "cohorts": comparisons,
        })
    return {
        "schema_version": SCHEMA, "graph_id": manifest["graph_id"], "accuracy_policy": accuracy_policy(),
        "declared_execution": manifest["execution"],
        "hardware_declaration": case.machine.to_dict(), "declared_kernel_bindings": dict(case.runtime.kernels),
        "structure_source": report.get("structure_source"),
        "mapping_coverage": {key: report.get(key) for key in (
            "event_window_count", "status_counts", "cohort_event_window_count",
            "cohort_status", "cohort_event_fraction")},
        "status_counts": dict(counts), "distinct_priced_programs": len(cache), "windows": windows,
        "semantics": {
            "join": "same_tensor_contract_and_physical_work_candidate_set; no_layer_choice",
            "multiplicity": "window_counts_are_not_steps; repeated_windows_are_not_independent_runs",
            "timing": "per_call_device_program_and_activity_sum; never_wall_time_or_busy_union_error",
            "subgraphs": "unfused_multiple_programs_are_references_until_a_fused_resource_boundary_is_selected",
            "aggregation": "no_cross_rank_window_or_overlapping_declaration_totals",
            "calibration": "comparator_does_not_fit_parameters;explicit_observable_service_calibration_is_supported",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args(argv)
    result = compare_cohorts(args.case, args.mapping)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
