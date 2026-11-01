"""Train primitive service curves independently of their heldout evaluation.

[Role] Aggregate repeated primitive trials and fit reusable work-response models.
[Invariants] Only training trials supply knots/startup. One packet fixes the
  hardware, implementation, layout, units and measurement boundary.
[Timeouts & Locks] Bounded explicit metadata input; no subprocesses or locks.
[Side-Effect Policy] Returns a model and separate validation; never runs a device.
"""
from __future__ import annotations

from collections import defaultdict
import math
from statistics import median

from mindie_sim.prediction.services import ServiceCurve


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
