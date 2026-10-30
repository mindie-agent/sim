"""Per-call accuracy policy for saved conditional cohort comparisons.

[Role] Add accuracy fields without invoking prediction or reading captures.
[Invariants] Tier selection uses the observed complete-event mean. Predictions
  remain unchanged; references and unscored rows never enter pass denominators.
[Timeouts & Locks] Constant work per row; no waits or locks.
[Side-Effect Policy] Pure projection of supplied mappings; no mutation or I/O.
"""
from collections.abc import Mapping
import math


ACCURACY_POLICY = "mindie-sim.cohort-accuracy/1"
_THRESHOLD_US = 50.0
_RELATIVE_TOLERANCE = 0.15
_ABSOLUTE_TOLERANCE_US = 15.0


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
