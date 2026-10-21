"""Fitted physical (roofline) calibration for A2/A3 operator durations.

[Role]
  Fit and serve a versioned, provenance-bearing physical cost model per
  ``(hardware_key, family, recipe, protocol_signature)``: a wave-quantized
  roofline with fitted tile geometry, launch intercept, cube / memory
  efficiency and an optional fitted L2-residency split.  This is the physical
  layer that ``PredictiveCostModel`` consumes above the analytic formula; it
  replaces the fixed 64x128 tiling, the constant AIC count, the multiplicative
  calibration factor and the invented launch floor of ``a3_cost.py``
  (see ``.workstreams/code-audit/formula-weaknesses.md`` W1/W2/W8/W9).

[Invariants]
  * Fitting reads train observations only.  Any observation whose ``split`` is
    not ``train`` is rejected before a parameter is estimated, and a declared
    heldout guard (condition keys plus physical identities) is asserted
    disjoint from the training set.
  * Observations are never pooled across ``protocol_signature`` or
    ``hardware_key``.  An ambiguous identity is a rejection with a reason, never
    a silent merge; a family below the minimum sample count is fitted at a
    coarser level that stays inside one protocol, or is declared unmodeled.
  * Every fitted value lives inside a declared physical bound (``eta_*`` in
    ``(0, 1]``, launch in ``LAUNCH_BOUNDS_MS``, ``block_*`` from the declared
    grid, ``l2_usable`` at most the declared L2 capacity) and is stored next to
    its sample count, train metrics and train-only CV metrics.
  * The L2 bandwidth is the only fitted rate allowed above the HBM peak, and
    only when the fitted blend beats the HBM-only model on train-only CV by
    ``MIN_CV_IMPROVEMENT_PCT``; otherwise ``memory_model`` is ``"hbm"``.
  * Prediction never returns a number for an unmodeled family / recipe / dtype,
    or for a machine whose hardware key contradicts the artifact: it returns
    ``None`` and an explicit reason (``predict_op_explained``).
  * Model terms are shape-derived, never per-shape corrections: the only
    per-family degrees of freedom are tile geometry, launch, the efficiencies
    and the L2 split.

[Timeouts & Locks]
  Pure in-memory arithmetic with bounded loops: no device, subprocess, network,
  lock, sleep or filesystem access.  Fitting is O(tile grid x memory candidates
  x CV folds x observations) with every loop count fixed by this module's
  declared constants.

[Side-Effect Policy]
  Importing and fitting write nothing and mutate no environment.  The
  ``__main__`` CLI reads the paths it is given and writes only ``--output``.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "SCHEMA_VERSION", "OBSERVATION_SCHEMA", "MODEL_KINDS",
    "BLOCK_M_CHOICES", "BLOCK_N_CHOICES", "LAUNCH_BOUNDS_MS", "ETA_BOUNDS",
    "MODEL_FORMS", "FAMILY_ALIASES", "ACCEPTED_TIMING_SCOPES", "M_BAND_LIMITS",
    "MIN_SAMPLES_PER_FIT", "PhysicalCalibrationError", "MachineLimits",
    "Observation", "FamilyParameters", "OpPrediction", "HeldoutGuard",
    "PhysicalCalibration", "fit_calibration", "read_observations_json", "read_campaign_results",
    "read_sqlite_observations", "observation_identity", "canonical_shape",
]

SCHEMA_VERSION = "mindie-sim.physical-calibration/1"
OBSERVATION_SCHEMA = "mindie-sim.observations/1"

# Tile geometry grid searched per fit (W1: fixed 64x128x64 is structural).
BLOCK_M_CHOICES = (32, 64, 128, 256)
BLOCK_N_CHOICES = (64, 128, 256, 512)
# Launch intercept: physically 0..60 us (W8: 20 us was fabricated for every SKU).
LAUNCH_BOUNDS_MS = (0.0, 0.060)
# Efficiency bounds; eta = 1 is the declared peak, never exceeded on HBM.
ETA_BOUNDS = (0.05, 1.0)
# L2 usable-capacity candidates as a fraction of the declared L2 capacity.
L2_USABLE_FRACTIONS = (0.0625, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1.0)
# Fitted L2 read bandwidth, as a multiple of the declared HBM peak.
L2_BW_MULTIPLE_BOUNDS = (1.0, 4.0)
# `l2_usable` and `eta_memory` trade off on a flat plateau once every binding row
# is fully resident; prefer the smallest resident prefix whose train WAPE is
# within this relative tolerance of the best, so the split stays identifiable.
L2_USABLE_SELECTION_TOLERANCE = 0.05

# Additive pipeline fill for short-k GEMMs (W3): declared k threshold and bounds.
K_FILL_MAX_K = 256.0
K_FILL_BOUNDS_MS = (0.0, 0.2)
# Memory-side n-alignment penalty: extra full-bandwidth passes paid when
# n % block_n != 0 (misaligned column tiles stream at a fraction of peak).
ALIGN_PENALTY_BOUNDS = (0.0, 4.0)
# A tail term must buy this much train-only CV before it is kept.
MIN_TAIL_IMPROVEMENT_PCT = 0.5
# A cross-scope shared fit must buy this much pooled train-only CV over the
# per-scope independent fits before it is adopted.
MIN_SHARED_SCOPE_IMPROVEMENT_PCT = 1.0

HUBER_K = 1.345
IRLS_ITERATIONS = 12
MIN_SAMPLES_PER_FIT = 6
CV_FOLDS = 5
# Declared issue ceiling per AIV (4 issue slots x 128 lanes): the physical upper
# bound for the fitted effective vector throughput.
VEC_ISSUE_MAX_PER_AIV = 512.0
# Slowest vector throughput the fit may claim, as a fraction of that ceiling.
VEC_THROUGHPUT_MIN_FRACTION = 1.0 / 64.0
# Profile interval: the parameter range whose train WAPE stays within this
# relative tolerance of the best value, all other parameters refitted.  A 25%
# band is a practical-equivalence width for a corpus whose own repeat spread is
# 10-40%, not a sampling confidence interval.
CI_TOLERANCE = 0.25
CI_SCAN_POINTS = 9
# The L2 blend must earn its two extra parameters on the regime where it can act.
MIN_CV_IMPROVEMENT_PCT = 1.0
MAX_FULL_CV_DEGRADATION_PCT = 1.0
# Timing scopes are physical-protocol identities: a batched interval and a
# per-call interval never share a launch intercept, so they never share a fit.
ACCEPTED_TIMING_SCOPES = (
    "device_event_warm",
    "device_event_warm_operator",
    "device_event_warm_per_call",
    "device_event_warm_batched",
    "device_event_warm_operator_batched_amortized",
    "device_event_warm_operator_per_call",
)
MIN_CACHE_PROBE_SAMPLES = 8
# Cache-probe subset: outcome-free, shape-only (working set fits the declared L2
# and the op is not compute-bound, so a residency split can change the answer).
CACHE_PROBE_MAX_INTENSITY = 32.0
# A rate is identified only when at least this many binding rows constrain it.
MIN_IDENTIFYING_ROWS = 3
MIN_IDENTIFYING_SHARE = 0.10
# Above this train-only CV WAPE a family fit is reported but flagged low_confidence.
MAX_TRUSTED_CV_WAPE_PCT = 25.0
MAX_TRUSTED_P90_APE_PCT = 60.0
ACCEPTED_SOURCE_CLASSES = ("measured",)
# family -> model kind.  Only families with measured train support are fitted;
# a family absent from the artifact is unmodeled, never extrapolated.
# Declared alias from an acceptance family name to a fitted family.  The A2 split
# declaration names its softmax family ``attention_softmax`` while the measured
# rows are ``torch_softmax`` at the same ``tokens x width`` geometry; the alias is
# explicit, recorded in the artifact and reported in every prediction reason.
FAMILY_ALIASES = {"attention_softmax": "softmax"}

MODEL_KINDS = {
    "gemm": "gemm",
    "copy": "vector",
    "add": "vector",
    "rmsnorm": "vector",
    "swiglu": "vector",
    "rope": "vector",
    "softmax": "vector",
    "masked_softmax": "vector",
    "quantize": "vector",
    "cast": "vector",
}
DTYPE_BYTES = {"bf16": 2.0, "bfloat16": 2.0, "fp16": 2.0, "float16": 2.0, "fp32": 4.0, "float32": 4.0}


class PhysicalCalibrationError(ValueError):
    """Raised for an inconsistent artifact, observation document or fit request."""


# ---------------------------------------------------------------------------
# Canonical identity helpers
# ---------------------------------------------------------------------------
def _number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return ("%.9g" % float(value))


def canonical_shape(shape: Mapping[str, Any]) -> str:
    """Stable text identity of a shape mapping; ``None`` entries are dropped."""
    parts = []
    for key in ("m", "n", "k", "elements"):
        if key in shape and shape[key] is not None:
            parts.append("%s=%s" % (key, _number(float(shape[key]))))
    return ",".join(parts)


def observation_identity(family: str, shape: Mapping[str, Any]) -> str:
    """Physical identity used for heldout exclusion: ``family|canonical_shape``."""
    return "%s|%s" % (str(family), canonical_shape(shape))


def _stable_digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_float(value: Any, name: str, *, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PhysicalCalibrationError("not_a_number:%s=%r" % (name, value))
    out = float(value)
    if not math.isfinite(out):
        raise PhysicalCalibrationError("not_finite:%s" % name)
    return out


def _ceil_div(total: float, width: float) -> int:
    if width <= 0:
        raise PhysicalCalibrationError("non_positive_tile")
    return max(1, int(math.ceil(max(0.0, total) / width)))


# ---------------------------------------------------------------------------
# Machine limits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MachineLimits:
    """Per-die peak rates the model divides by; every field names its unit."""

    hardware_key: str
    arch_family: str
    aic_count: int
    aiv_count: int
    clock_mhz: float
    sustained_clock_mhz: float
    clock_source: str
    ops_per_cycle_per_aic: float
    peak_flops: float
    peak_bw_bytes_per_s: float
    l2_bytes: int
    vec_elements_per_cycle_per_aiv: float
    vec_peak_ops_per_s: float
    vec_max_ops_per_s: float
    evidence: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MachineLimits":
        if not isinstance(value, Mapping):
            raise PhysicalCalibrationError("machine_must_be_a_mapping")
        key = str(value.get("hardware_key") or "")
        arch = str(value.get("arch_family") or "")
        if value.get("clock_mhz") is not None:
            clock_mhz = _as_float(value.get("clock_mhz"), "clock_mhz")
        elif value.get("clock_hz") is not None:
            clock_mhz = _as_float(value.get("clock_hz"), "clock_hz") / 1e6
        else:
            raise PhysicalCalibrationError("machine_missing_clock")
        # The sustained (under-load) clock governs a warm repeated-call interval.
        sustained = _as_float(
            value.get("sustained_clock_mhz"), "sustained_clock_mhz", allow_none=True
        )
        if sustained is None:
            sustained = clock_mhz
            clock_source = "rated"
        else:
            clock_source = "sustained"
        declared_source = str(value.get("clock_source") or "")
        if declared_source:
            clock_source = declared_source
        if sustained > clock_mhz * 1.001:
            raise PhysicalCalibrationError("sustained_clock_above_rated")
        aic = int(_as_float(value.get("aic_count"), "aic_count"))
        aiv = int(_as_float(value.get("aiv_count", 0), "aiv_count", allow_none=True) or 0)
        ops_per_cycle = _as_float(value.get("ops_per_cycle"), "ops_per_cycle")
        declared_peak = _as_float(value.get("peak_flops"), "peak_flops", allow_none=True)
        peak = aic * ops_per_cycle * sustained * 1e6
        if declared_peak is not None and abs(declared_peak - peak) > 1e-3 * peak:
            raise PhysicalCalibrationError(
                "peak_flops_contradicts_components: declared=%r derivable=%r" % (declared_peak, peak)
            )
        bw = _as_float(value.get("peak_bw_bytes_per_s"), "peak_bw_bytes_per_s")
        if bw <= 0:
            raise PhysicalCalibrationError("non_positive_peak_bw")
        l2 = int(_as_float(value.get("l2_bytes"), "l2_bytes"))
        lanes = _as_float(
            value.get("vec_elements_per_cycle_per_aiv"), "vec_elements_per_cycle_per_aiv", allow_none=True
        )
        vec_peak = _as_float(value.get("vec_peak_ops_per_s"), "vec_peak_ops_per_s", allow_none=True)
        if lanes is None and vec_peak is not None and aiv > 0:
            lanes = vec_peak / (float(aiv) * sustained * 1e6)
        if lanes is None or aiv <= 0:
            raise PhysicalCalibrationError("vec_peak_underivable")
        vec_peak = float(aiv) * lanes * sustained * 1e6
        vec_max = float(aiv) * VEC_ISSUE_MAX_PER_AIV * sustained * 1e6
        declared_vec_max = _as_float(value.get("vec_max_ops_per_s"), "vec_max_ops_per_s", allow_none=True)
        if declared_vec_max is not None and declared_vec_max < vec_peak:
            raise PhysicalCalibrationError("vec_max_below_nominal_peak")
        evidence = value.get("evidence") or ()
        rows: list[tuple[str, str]] = []
        if isinstance(evidence, Mapping):
            rows = [(str(k), str(v)) for k, v in sorted(evidence.items())]
        elif isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
            for item in evidence:
                if isinstance(item, Mapping):
                    rows.append((str(item.get("field")), str(item.get("source"))))
                elif isinstance(item, Sequence) and len(item) == 2:
                    rows.append((str(item[0]), str(item[1])))
        return cls(
            hardware_key=key,
            arch_family=arch,
            aic_count=aic,
            aiv_count=aiv,
            clock_mhz=clock_mhz,
            sustained_clock_mhz=float(sustained),
            clock_source=clock_source,
            ops_per_cycle_per_aic=ops_per_cycle,
            peak_flops=peak,
            peak_bw_bytes_per_s=bw,
            l2_bytes=l2,
            vec_elements_per_cycle_per_aiv=float(lanes),
            vec_peak_ops_per_s=float(vec_peak),
            vec_max_ops_per_s=float(max(vec_max, declared_vec_max or 0.0)),
            evidence=tuple(rows),
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "hardware_key": self.hardware_key,
            "arch_family": self.arch_family,
            "aic_count": self.aic_count,
            "aiv_count": self.aiv_count,
            "clock_mhz": self.clock_mhz,
            "sustained_clock_mhz": self.sustained_clock_mhz,
            "clock_source": self.clock_source,
            "ops_per_cycle": self.ops_per_cycle_per_aic,
            "peak_flops": self.peak_flops,
            "peak_bw_bytes_per_s": self.peak_bw_bytes_per_s,
            "l2_bytes": self.l2_bytes,
            "vec_elements_per_cycle_per_aiv": self.vec_elements_per_cycle_per_aiv,
            "vec_peak_ops_per_s": self.vec_peak_ops_per_s,
            "vec_max_ops_per_s": self.vec_max_ops_per_s,
        }
        if self.evidence:
            out["evidence"] = [{"field": f, "source": s} for f, s in self.evidence]
        return out


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Observation:
    """One measured physical condition, already reduced to a target duration."""

    condition_key: str
    hardware_key: str
    arch_family: str
    family: str
    recipe: str
    protocol_signature: str
    split: str
    duration_ms: float
    flops: float
    hbm_bytes: float
    elements: float
    shape: Mapping[str, Any]
    dtype: str
    timing_scope: str
    source_class: str
    die: str
    run_id: str
    raw_event_count: int
    model_kind: str
    target_estimator: str = ""
    protocol_core_signature: str = ""
    package_tree: str = ""

    @property
    def identity(self) -> str:
        return observation_identity(self.family, self.shape)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, model_kind: str | None = None) -> "Observation":
        if not isinstance(value, Mapping):
            raise PhysicalCalibrationError("observation_must_be_a_mapping")
        family = str(value.get("family") or "").strip()
        if not family:
            raise PhysicalCalibrationError("observation_missing_family")
        kind = model_kind or MODEL_KINDS.get(family, "")
        shape = value.get("shape") or {}
        if not isinstance(shape, Mapping):
            raise PhysicalCalibrationError("observation_shape_must_be_a_mapping")
        shape = {k: v for k, v in shape.items() if v is not None}
        raw = value.get("raw_samples_ms") or ()
        raw_count = value.get("raw_event_count")
        if raw_count is None:
            raw_count = len(raw) if isinstance(raw, Sequence) else 0
        duration = _as_float(value.get("duration_ms"), "duration_ms")
        estimator = str(value.get("target_estimator") or "")
        trials = value.get("trial_medians_ms")
        if isinstance(trials, Sequence) and not isinstance(trials, (str, bytes)) and len(trials):
            medians = sorted(float(item) for item in trials if item is not None)
            if medians:
                # Acceptance target is the median of per-trial event medians; the
                # pooled repeat median is a weaker estimator of the same quantity.
                duration = medians[len(medians) // 2] if len(medians) % 2 else (
                    0.5 * (medians[len(medians) // 2 - 1] + medians[len(medians) // 2])
                )
                estimator = "median_of_trial_medians"
        if not estimator:
            estimator = "declared_duration_ms"
        if duration is None or duration <= 0:
            raise PhysicalCalibrationError("observation_duration_not_positive")
        flops = _as_float(value.get("flops"), "flops", allow_none=True) or 0.0
        hbm_bytes = _as_float(value.get("hbm_bytes"), "hbm_bytes", allow_none=True) or 0.0
        elements = _as_float(value.get("elements"), "elements", allow_none=True) or 0.0
        if kind == "gemm" and not (flops > 0 and hbm_bytes > 0):
            raise PhysicalCalibrationError("gemm_observation_missing_flops_or_bytes")
        if kind == "vector" and not (elements > 0 or hbm_bytes > 0):
            raise PhysicalCalibrationError("vector_observation_missing_elements_and_bytes")
        return cls(
            condition_key=str(value.get("condition_key") or ""),
            hardware_key=str(value.get("hardware_key") or ""),
            arch_family=str(value.get("arch_family") or ""),
            family=family,
            recipe=str(value.get("recipe") or ""),
            protocol_signature=str(value.get("protocol_signature") or ""),
            split=str(value.get("split") or ""),
            duration_ms=float(duration),
            flops=flops,
            hbm_bytes=hbm_bytes,
            elements=elements,
            shape=dict(shape),
            dtype=str(value.get("dtype") or value.get("compute_dtype") or ""),
            timing_scope=str(value.get("timing_scope") or ""),
            source_class=str(value.get("source_class") or ""),
            die=str(value.get("die") or ""),
            run_id=str(value.get("run_id") or ""),
            raw_event_count=int(raw_count or 0),
            model_kind=kind,
            target_estimator=estimator,
            protocol_core_signature=str(
                value.get("protocol_core_signature")
                or protocol_core_signature(value.get("protocol") or {})
                or str(value.get("protocol_signature") or "")
            ),
            package_tree=str(value.get("package_tree") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_key": self.condition_key,
            "hardware_key": self.hardware_key,
            "die": self.die,
            "arch_family": self.arch_family,
            "family": self.family,
            "recipe": self.recipe,
            "protocol_signature": self.protocol_signature,
            "run_id": self.run_id,
            "split": self.split,
            "shape": dict(self.shape),
            "elements": self.elements or None,
            "flops": self.flops,
            "hbm_bytes": self.hbm_bytes,
            "duration_ms": self.duration_ms,
            "dtype": self.dtype,
            "source_class": self.source_class,
            "timing_scope": self.timing_scope,
            "raw_event_count": self.raw_event_count,
            "target_estimator": self.target_estimator,
            "protocol_core_signature": self.protocol_core_signature,
            "package_tree": self.package_tree,
        }


# ---------------------------------------------------------------------------
# Heldout guard
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HeldoutGuard:
    """Declared heldout conditions that must never enter a fit."""

    condition_keys: frozenset[str]
    identities: frozenset[str]
    sources: tuple[str, ...]
    declared_conditions: int

    @classmethod
    def from_files(cls, paths: Iterable[str | Path]) -> "HeldoutGuard":
        keys: set[str] = set()
        identities: set[str] = set()
        used: list[str] = []
        for raw_path in paths:
            path = Path(raw_path)
            value = json.loads(path.read_text())
            found_keys, found_ids = _extract_heldout(value)
            if not found_ids and not found_keys:
                continue
            keys |= found_keys
            identities |= found_ids
            used.append(str(path))
        return cls(
            condition_keys=frozenset(keys),
            identities=frozenset(identities),
            sources=tuple(used),
            declared_conditions=len(identities | {k for k in keys}),
        )

    def overlap(self, observations: Sequence[Observation]) -> list[str]:
        hits = []
        for obs in observations:
            if obs.identity in self.identities or (obs.condition_key and obs.condition_key in self.condition_keys):
                hits.append("%s|%s" % (obs.family, canonical_shape(obs.shape)))
        return hits

    def to_dict(self, *, status: str, overlap: Sequence[str] = ()) -> dict[str, Any]:
        return {
            "status": status,
            "declared_conditions": self.declared_conditions,
            "identity_count": len(self.identities),
            "condition_key_count": len(self.condition_keys),
            "sources": list(self.sources),
            "overlap": list(overlap),
        }


def _extract_heldout(value: Any) -> tuple[set[str], set[str]]:
    """Collect heldout condition keys and physical identities from known schemas."""
    keys: set[str] = set()
    identities: set[str] = set()

    def take(family: Any, shape: Any, *key_values: Any) -> None:
        for item in key_values:
            if isinstance(item, str) and item:
                keys.add(item)
        if family and isinstance(shape, Mapping):
            identities.add(observation_identity(str(family), shape))

    if not isinstance(value, Mapping):
        return keys, identities
    # Frozen score rows: one row per (condition, variant).
    for row in value.get("rows") or ():
        if isinstance(row, Mapping):
            take(row.get("family"), row.get("physical_shape"), row.get("condition_id"))
    # Split declarations: families[].heldout[], new_heldout.points[], cross_sku_matched[].
    for family_block in value.get("families") or ():
        if isinstance(family_block, Mapping):
            name = family_block.get("family")
            if isinstance(name, str):
                take(name, None)
            for point in family_block.get("heldout") or ():
                if isinstance(point, Mapping):
                    take(point.get("family") or family_block.get("family"), point.get("shape"),
                         point.get("point_id"), point.get("geometry_key"))
    for section in ("new_heldout", "gap_fill_train"):
        block = value.get(section)
        if isinstance(block, Mapping):
            for point in block.get("points") or ():
                if isinstance(point, Mapping) and str(point.get("split") or "heldout") == "heldout":
                    take(point.get("family"), point.get("shape"), point.get("point_id"),
                         point.get("geometry_key"))
    for point in value.get("cross_sku_matched") or ():
        if isinstance(point, Mapping):
            take(point.get("family"), point.get("shape"), point.get("a3_geometry_key"),
                 point.get("a2_geometry_key"))
    # Campaign result files: cases[] / results[] with a request block.
    for case in value.get("cases") or ():
        if isinstance(case, Mapping):
            request = case.get("request") if isinstance(case.get("request"), Mapping) else case
            take(request.get("op_family") or request.get("family"), request.get("shape"),
                 request.get("case_id"), case.get("case_id"))
    for result in value.get("results") or ():
        if isinstance(result, Mapping):
            request = result.get("request") if isinstance(result.get("request"), Mapping) else result
            take(request.get("op_family") or request.get("family"),
                 request.get("shape") or result.get("shape"),
                 request.get("case_id"), result.get("case_id"))
    return keys, identities


# ---------------------------------------------------------------------------
# Family parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FamilyParameters:
    """Fitted parameters of one ``(family, recipe, protocol)`` model."""

    family: str
    recipe: str
    protocol_signature: str
    timing_scope: str
    model_kind: str
    model_form: str
    fit_level: str
    block_m: int | None
    block_n: int | None
    launch_ms: float
    eta_compute: float | None
    eta_memory: float | None
    vec_ops_per_s: float | None
    memory_model: str
    l2_usable_bytes: int
    l2_bw_bytes_per_s: float | None
    compute_dtype: str
    samples: int
    raw_event_count: int
    train_wape_pct: float
    cv_wape_pct: float | None
    median_ape_pct: float
    p90_ape_pct: float
    max_ape_pct: float
    protocol_core_signature: str = ""
    launch_ms_by_scope: Mapping[str, float] = field(default_factory=dict)
    scope_samples: Mapping[str, int] = field(default_factory=dict)
    tail: Mapping[str, bool] = field(default_factory=dict)
    k_fill: bool = False
    k_fill_ms: float | None = None
    align_penalty: float | None = None
    binding_share: Mapping[str, float] = field(default_factory=dict)
    binding_counts: Mapping[str, int] = field(default_factory=dict)
    identified: Mapping[str, bool] = field(default_factory=dict)
    pinned_parameters: tuple[str, ...] = ()
    intervals: Mapping[str, Any] = field(default_factory=dict)
    breakdown: Mapping[str, Any] = field(default_factory=dict)
    model_selection: Mapping[str, Any] = field(default_factory=dict)
    memory_model_selection: Mapping[str, Any] = field(default_factory=dict)
    quality: str = "trusted"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "family": self.family,
            "recipe": self.recipe,
            "protocol_signature": self.protocol_signature,
            "protocol_core_signature": self.protocol_core_signature,
            "timing_scope": self.timing_scope,
            "launch_ms_by_scope": dict(self.launch_ms_by_scope),
            "scope_samples": dict(self.scope_samples),
            "model_kind": self.model_kind,
            "model_form": self.model_form,
            "tail": dict(self.tail),
            "k_fill": self.k_fill,
            "k_fill_ms": self.k_fill_ms,
            "align_penalty": self.align_penalty,
            "fit_level": self.fit_level,
            "block_m": self.block_m,
            "block_n": self.block_n,
            "launch_ms": self.launch_ms,
            "eta_compute": self.eta_compute,
            "eta_memory": self.eta_memory,
            "vec_ops_per_s": self.vec_ops_per_s,
            "memory_model": self.memory_model,
            "l2_usable_bytes": self.l2_usable_bytes,
            "l2_bw_bytes_per_s": self.l2_bw_bytes_per_s,
            "compute_dtype": self.compute_dtype,
            "samples": self.samples,
            "raw_event_count": self.raw_event_count,
            "train_wape_pct": self.train_wape_pct,
            "cv_wape_pct": self.cv_wape_pct,
            "median_ape_pct": self.median_ape_pct,
            "p90_ape_pct": self.p90_ape_pct,
            "max_ape_pct": self.max_ape_pct,
            "binding_share": dict(self.binding_share),
            "binding_counts": dict(self.binding_counts),
            "identified": dict(self.identified),
            "pinned_parameters": list(self.pinned_parameters),
            "intervals": {key: list(value) for key, value in self.intervals.items()},
            "breakdown": {key: dict(value) for key, value in self.breakdown.items()},
            "model_selection": dict(self.model_selection),
            "memory_model_selection": dict(self.memory_model_selection),
            "quality": self.quality,
            "warnings": list(self.warnings),
        }
        return out

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FamilyParameters":
        if not isinstance(value, Mapping):
            raise PhysicalCalibrationError("family_parameters_must_be_a_mapping")
        try:
            return cls(
                family=str(value["family"]),
                recipe=str(value.get("recipe") or ""),
                protocol_signature=str(value.get("protocol_signature") or ""),
                protocol_core_signature=str(value.get("protocol_core_signature") or ""),
                timing_scope=str(value.get("timing_scope") or ""),
                launch_ms_by_scope={
                    str(k): float(v) for k, v in (value.get("launch_ms_by_scope") or {}).items()
                },
                scope_samples={
                    str(k): int(v) for k, v in (value.get("scope_samples") or {}).items()
                },
                model_kind=str(value["model_kind"]),
                model_form=str(value.get("model_form") or "wave_max"),
                tail={str(k): bool(v) for k, v in (value.get("tail") or {}).items()},
                k_fill=bool(value.get("k_fill") or False),
                k_fill_ms=None if value.get("k_fill_ms") is None else float(value["k_fill_ms"]),
                align_penalty=(
                    None if value.get("align_penalty") is None else float(value["align_penalty"])
                ),
                fit_level=str(value["fit_level"]),
                block_m=None if value.get("block_m") is None else int(value["block_m"]),
                block_n=None if value.get("block_n") is None else int(value["block_n"]),
                launch_ms=float(value["launch_ms"]),
                eta_compute=None if value.get("eta_compute") is None else float(value["eta_compute"]),
                eta_memory=None if value.get("eta_memory") is None else float(value["eta_memory"]),
                vec_ops_per_s=(
                    None if value.get("vec_ops_per_s") is None else float(value["vec_ops_per_s"])
                ),
                memory_model=str(value["memory_model"]),
                l2_usable_bytes=int(value.get("l2_usable_bytes") or 0),
                l2_bw_bytes_per_s=(
                    None if value.get("l2_bw_bytes_per_s") is None else float(value["l2_bw_bytes_per_s"])
                ),
                compute_dtype=str(value.get("compute_dtype") or "bf16"),
                samples=int(value.get("samples") or 0),
                raw_event_count=int(value.get("raw_event_count") or 0),
                train_wape_pct=float(value.get("train_wape_pct") or 0.0),
                cv_wape_pct=None if value.get("cv_wape_pct") is None else float(value["cv_wape_pct"]),
                median_ape_pct=float(value.get("median_ape_pct") or 0.0),
                p90_ape_pct=float(value.get("p90_ape_pct") or 0.0),
                max_ape_pct=float(value.get("max_ape_pct") or 0.0),
                binding_share=dict(value.get("binding_share") or {}),
                binding_counts={str(k): int(v) for k, v in (value.get("binding_counts") or {}).items()},
                identified=dict(value.get("identified") or {}),
                pinned_parameters=tuple(value.get("pinned_parameters") or ()),
                intervals=dict(value.get("intervals") or {}),
                breakdown=dict(value.get("breakdown") or {}),
                model_selection=dict(value.get("model_selection") or {}),
                memory_model_selection=dict(value.get("memory_model_selection") or {}),
                quality=str(value.get("quality") or "trusted"),
                warnings=tuple(value.get("warnings") or ()),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PhysicalCalibrationError("malformed_family_parameters:%s" % (exc,)) from exc


# ---------------------------------------------------------------------------
# Prediction result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class OpPrediction:
    """One predicted operator duration plus the reason it exists (or does not)."""

    duration_ms: float | None
    status: str
    reason: str
    family: str
    recipe: str
    protocol_signature: str = ""
    timing_scope: str = ""
    model_form: str = ""
    fit_level: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    breakdown: Mapping[str, float] = field(default_factory=dict)
    machine_source: str = ""
    machine_fields_from_artifact: tuple[str, ...] = ()

    @property
    def modeled(self) -> bool:
        return self.duration_ms is not None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status,
            "reason": self.reason,
            "duration_ms": self.duration_ms,
            "family": self.family,
            "recipe": self.recipe,
            "protocol_signature": self.protocol_signature,
            "timing_scope": self.timing_scope,
            "model_form": self.model_form,
            "fit_level": self.fit_level,
            "machine_source": self.machine_source,
        }
        if self.parameters:
            out["parameters"] = dict(self.parameters)
        if self.breakdown:
            out["breakdown"] = dict(self.breakdown)
        if self.machine_fields_from_artifact:
            out["machine_fields_from_artifact"] = list(self.machine_fields_from_artifact)
        return out


# ---------------------------------------------------------------------------
# Fitting machinery
# ---------------------------------------------------------------------------
def _gemm_geometry(obs: Observation, block_m: int, block_n: int, aic: int) -> tuple[int, int, float]:
    m = float(obs.shape.get("m") or 0.0)
    n = float(obs.shape.get("n") or 0.0)
    blocks = _ceil_div(m, block_m) * _ceil_div(n, block_n)
    waves = max(1, _ceil_div(blocks, aic))
    return blocks, waves, blocks / float(waves * max(1, aic))


def _memory_split(total_bytes: float, l2_usable: int, peak_bw: float) -> tuple[float, float]:
    """(hbm_part, l2_part) in bytes for the declared residency split."""
    if l2_usable <= 0:
        return total_bytes, 0.0
    l2 = min(total_bytes, float(l2_usable))
    return max(0.0, total_bytes - l2), l2


def _short_k(obs: Observation, threshold: float = K_FILL_MAX_K) -> bool:
    """True when a GEMM's k is short enough to pay a pipeline fill (W3)."""
    if obs.model_kind != "gemm":
        return False
    k = float(obs.shape.get("k") or 0.0)
    return 0.0 < k <= threshold


def _padded_flops(obs: Observation, block_m: int | None, tail: Mapping[str, bool]) -> float:
    """FLOPs the cube executes once a partial last row tile is padded."""
    m = float(obs.shape.get("m") or 0.0)
    n = float(obs.shape.get("n") or 0.0)
    k = float(obs.shape.get("k") or 0.0)
    if m <= 0 or n <= 0 or k <= 0:
        return obs.flops
    rows = _ceil_div(m, block_m) * block_m if (tail.get("m") and block_m) else m
    return 2.0 * rows * n * k


def _misaligned(obs: Observation, block_n: int | None) -> bool:
    """True when the column tile does not divide n (memory-side tail)."""
    if not block_n:
        return False
    n = float(obs.shape.get("n") or 0.0)
    if n <= 0:
        return False
    return int(n) % int(block_n) != 0


def _design_row(
    obs: Observation,
    machine: MachineLimits,
    form: str,
    memory_model: str,
    l2_usable: int,
    block_m: int | None,
    block_n: int | None,
    *,
    tail: Mapping[str, bool],
    launch_index: int,
    n_launch: int,
    k_fill: bool,
) -> dict[str, Any]:
    """One coefficient row for ``t = launch[scope] + additive + combine(compute, memory)``.

    The parameter vector is ``[launch_0 .. launch_{S-1}, rate_compute, rate_memory,
    (rate_l2), (k_fill)]``; the row stores its own offsets so scoring and fitting
    stay linear in the parameters for a fixed form.
    """
    to_ms = 1000.0
    n_rate = 3 if memory_model == "l2_blend" else 2
    i_rate = n_launch
    i_add = i_rate + n_rate
    compute_coef = 0.0
    geometry: dict[str, float] = {"wave_eff": 1.0, "blocks": 0.0, "waves": 0.0}
    if obs.model_kind == "gemm":
        wave_eff = 1.0
        if form == "wave_max":
            blocks, waves, wave_eff = _gemm_geometry(
                obs, int(block_m or 1), int(block_n or 1), machine.aic_count
            )
            geometry = {"blocks": float(blocks), "waves": float(waves), "wave_eff": wave_eff}
            flops = _padded_flops(obs, block_m, tail)
        else:
            flops = _padded_flops(obs, block_m, tail)
        geometry["flops_effective"] = flops
        if flops > 0:
            compute_coef = flops / (machine.peak_flops * max(wave_eff, 1e-9)) * to_ms
    else:
        if obs.elements > 0:
            # Normalised by the declared issue ceiling, so the fitted coordinate
            # is ceiling/throughput and the rate keeps a physical upper bound.
            compute_coef = obs.elements / machine.vec_max_ops_per_s * to_ms
    hbm_part, l2_part = _memory_split(obs.hbm_bytes, l2_usable, machine.peak_bw_bytes_per_s)
    memory_coef = [hbm_part / machine.peak_bw_bytes_per_s * to_ms]
    if memory_model == "l2_blend":
        memory_coef.append(l2_part / machine.peak_bw_bytes_per_s * to_ms)
    additive = []
    if k_fill:
        additive.append(1.0 if _short_k(obs) else 0.0)
    n_align = 1 if tail.get("n") else 0
    i_align = i_add + len(additive)
    align_coefs = [0.0] * len(memory_coef)
    if n_align and _misaligned(obs, block_n):
        # One extra full-bandwidth pass over the whole working set per unit of
        # the fitted penalty: memory = coef/eta_m + coef*p_align.
        align_coefs = list(memory_coef)
    return {
        "form": form,
        "launch_index": launch_index,
        "i_rate": i_rate,
        "i_add": i_add,
        "i_align": i_align,
        "n_align": n_align,
        "compute_coef": compute_coef,
        "memory_coef": memory_coef,
        "align_coefs": align_coefs,
        "additive": additive,
        "geometry": geometry,
        "target": obs.duration_ms,
    }


def _term_values(row: Mapping[str, Any], theta: Sequence[float]) -> tuple[float, float]:
    """(compute, memory) terms of one design row for the current parameters."""
    i = row["i_rate"]
    compute = row["compute_coef"] * theta[i]
    memory = sum(coef * theta[i + 1 + index] for index, coef in enumerate(row["memory_coef"]))
    if row.get("n_align"):
        memory += theta[row["i_align"]] * sum(row["align_coefs"])
    return compute, memory


def _evaluate(row: Mapping[str, Any], theta: Sequence[float]) -> float:
    """Predicted duration of one design row under the row's declared form."""
    compute, memory = _term_values(row, theta)
    base = theta[row["launch_index"]]
    for index, coefficient in enumerate(row["additive"]):
        base += theta[row["i_add"] + index] * coefficient
    if row.get("form", "wave_max") == "wave_max":
        return base + max(compute, memory)
    return base + compute + memory


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float] | None:
    n = len(vector)
    work = [list(matrix[i]) + [vector[i]] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < 1e-18:
            return None
        work[col], work[pivot] = work[pivot], work[col]
        inverse = 1.0 / work[col][col]
        for r in range(col + 1, n):
            factor = work[r][col] * inverse
            if factor:
                for c in range(col, n + 1):
                    work[r][c] -= factor * work[col][c]
    out = [0.0] * n
    for i in range(n - 1, -1, -1):
        acc = work[i][n] - sum(work[i][j] * out[j] for j in range(i + 1, n))
        out[i] = acc / work[i][i]
    return out


def _parameter_bounds(
    n_launch: int,
    kind: str,
    memory_model: str,
    k_fill: bool,
    n_align: bool = False,
    l2_multiple_bounds: Sequence[float] = L2_BW_MULTIPLE_BOUNDS,
) -> list[tuple[float, float]]:
    low, high = ETA_BOUNDS
    bounds = [LAUNCH_BOUNDS_MS] * max(1, n_launch)
    if kind == "gemm":
        bounds.append((1.0 / high, 1.0 / low))
    else:
        bounds.append((1.0, 1.0 / VEC_THROUGHPUT_MIN_FRACTION))
    bounds.append((1.0 / high, 1.0 / low))
    if memory_model == "l2_blend":
        bounds.append((1.0 / l2_multiple_bounds[1], 1.0 / l2_multiple_bounds[0]))
    if n_align:
        bounds.append(ALIGN_PENALTY_BOUNDS)
    if k_fill:
        bounds.append(K_FILL_BOUNDS_MS)
    return bounds


def _clamp(theta: Sequence[float], bounds: Sequence[tuple[float, float]]) -> list[float]:
    return [min(max(value, lo), hi) for value, (lo, hi) in zip(theta, bounds)]


def _initial_theta(bounds: Sequence[tuple[float, float]]) -> list[float]:
    # Start every launch at 0, every efficiency branch at eta = 0.5 (or the
    # vector ceiling fraction 2), the L2 rate at 2x peak and the fill at 0.
    start = []
    for lo, hi in bounds:
        if lo == 0.0:
            start.append(0.0)
        elif hi <= 1.0:
            start.append(0.5 * (lo + hi) if hi < 1.0 else 2.0)
        else:
            start.append(min(max(2.0, lo), hi))
    return _clamp(start, bounds)


def _robust_scale(residuals: Sequence[float]) -> float:
    if not residuals:
        return 1e-9
    ordered = sorted(residuals)
    mid = ordered[len(ordered) // 2]
    deviations = sorted(abs(r - mid) for r in residuals)
    mad = deviations[len(deviations) // 2]
    return max(1.4826 * mad, 1e-9)


def _irls_fit(
    rows: Sequence[Mapping[str, Any]],
    bounds: Sequence[tuple[float, float]],
    iterations: int = IRLS_ITERATIONS,
    pinned: Mapping[int, float] | None = None,
) -> list[float]:
    """Projected IRLS with Huber weights over the active (binding) branch.

    ``pinned`` fixes coordinates at declared values (used by the profile
    interval, which refits every other parameter at a fixed candidate rate).
    """
    theta = _initial_theta(bounds)
    n_params = len(bounds)
    if not rows:
        return theta
    pinned = dict(pinned or {})
    for index, value in pinned.items():
        theta[index] = min(max(value, bounds[index][0]), bounds[index][1])
    if len(pinned) >= n_params:
        return theta
    scale = 1.0
    for _ in range(iterations):
        predictions = [_evaluate(row, theta) for row in rows]
        residuals = [row["target"] - pred for row, pred in zip(rows, predictions)]
        scale = _robust_scale(residuals)
        weights = []
        for residual in residuals:
            magnitude = abs(residual)
            weights.append(1.0 if magnitude <= HUBER_K * scale else (HUBER_K * scale) / magnitude)
        matrix = [[0.0] * n_params for _ in range(n_params)]
        vector = [0.0] * n_params
        for row, weight in zip(rows, weights):
            compute, memory = _term_values(row, theta)
            active_memory = memory > compute
            design = [0.0] * n_params
            design[row["launch_index"]] = 1.0
            for index, coefficient in enumerate(row["additive"]):
                design[row["i_add"] + index] += coefficient
            if active_memory:
                for index, coefficient in enumerate(row["memory_coef"]):
                    design[row["i_rate"] + 1 + index] += coefficient
                if row.get("n_align"):
                    design[row["i_align"]] += sum(row["align_coefs"])
            else:
                design[row["i_rate"]] += row["compute_coef"]
            for i in range(n_params):
                if design[i] == 0.0:
                    continue
                vector[i] += weight * design[i] * row["target"]
                for j in range(i, n_params):
                    matrix[i][j] += weight * design[i] * design[j]
        for i in range(n_params):
            for j in range(i):
                matrix[i][j] = matrix[j][i]
        trace = sum(matrix[i][i] for i in range(n_params))
        ridge = max(trace, 1.0) * 1e-12
        for i in range(n_params):
            matrix[i][i] += ridge
        solved = _solve(matrix, vector)
        if solved is None:
            break
        theta = _clamp(solved, bounds)
        for index, value in pinned.items():
            theta[index] = value
    return theta


def _score(rows: Sequence[Mapping[str, Any]], theta: Sequence[float]) -> dict[str, Any]:
    targets = [row["target"] for row in rows]
    predictions = [_evaluate(row, theta) for row in rows]
    residuals = [pred - target for pred, target in zip(predictions, targets)]
    total = sum(targets)
    absolute = sum(abs(r) for r in residuals)
    apes = sorted(100.0 * abs(r) / target for r, target in zip(residuals, targets) if target > 0)
    return {
        "wape_pct": 100.0 * absolute / total if total > 0 else float("nan"),
        "median_ape_pct": _quantile(apes, 0.5),
        "p90_ape_pct": _quantile(apes, 0.9),
        "max_ape_pct": apes[-1] if apes else float("nan"),
        "signed_bias_pct": 100.0 * sum(residuals) / total if total > 0 else float("nan"),
    }


def _quantile(ordered: Sequence[float], fraction: float) -> float:
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(math.ceil(fraction * len(ordered))) - 1))
    return float(ordered[index])


def _binding_detail(rows: Sequence[Mapping[str, Any]], theta: Sequence[float]) -> dict[str, Any]:
    """Which term dominates per row, and whether each rate entered that term."""
    counts = {"compute": 0, "memory": 0, "memory_hbm_part": 0, "memory_l2_part": 0}
    for row in rows:
        compute, memory = _term_values(row, theta)
        if compute >= memory:
            counts["compute"] += 1
            continue
        counts["memory"] += 1
        if row["memory_coef"][0] > 0:
            counts["memory_hbm_part"] += 1
        if len(row["memory_coef"]) > 1 and row["memory_coef"][1] > 0:
            counts["memory_l2_part"] += 1
    total = max(1, len(rows))
    return {
        "share": {"compute": counts["compute"] / total, "memory": counts["memory"] / total},
        "counts": counts,
        "samples": len(rows),
    }


def _theta_to_parameters(
    theta: Sequence[float],
    kind: str,
    machine: MachineLimits,
    n_launch: int,
    memory_model: str,
    k_fill: bool,
    n_align: bool = False,
) -> dict[str, Any]:
    """Convert the fitted vector to physical rates (efficiencies / throughput)."""
    i_rate = n_launch
    i_mem = i_rate + 1
    launches = list(theta[:n_launch])
    eta_memory = 1.0 / theta[i_mem] if theta[i_mem] > 0 else ETA_BOUNDS[0]
    out: dict[str, Any] = {
        "launches_ms": launches,
        "launch_ms": launches[0] if launches else 0.0,
        "eta_compute": None,
        "eta_memory": eta_memory,
        "vec_ops_per_s": None,
        "l2_bw_bytes_per_s": None,
        "k_fill_ms": None,
        "align_penalty": None,
    }
    if kind == "gemm":
        out["eta_compute"] = 1.0 / theta[i_rate] if theta[i_rate] > 0 else ETA_BOUNDS[0]
    else:
        out["vec_ops_per_s"] = (
            machine.vec_max_ops_per_s / theta[i_rate] if theta[i_rate] > 0 else machine.vec_max_ops_per_s
        )
    if memory_model == "l2_blend":
        i_l2 = i_mem + 1
        if theta[i_l2] > 0:
            out["l2_bw_bytes_per_s"] = machine.peak_bw_bytes_per_s / theta[i_l2]
    n_mem = 3 if memory_model == "l2_blend" else 2
    if n_align:
        out["align_penalty"] = float(theta[i_mem + n_mem - 1])
    if k_fill:
        out["k_fill_ms"] = float(theta[-1])
    return out


def _fold_index(observations: Sequence[Observation], folds: int) -> list[int]:
    ordered = sorted(
        range(len(observations)),
        key=lambda i: (observations[i].condition_key, observations[i].identity, i),
    )
    assignment = [0] * len(observations)
    for slot, index in enumerate(ordered):
        assignment[index] = slot % folds
    return assignment


def _launch_indices(observations: Sequence[Observation], launch_scopes: Sequence[str]) -> list[int]:
    order = {scope: index for index, scope in enumerate(launch_scopes)}
    return [order.get(obs.timing_scope, 0) for obs in observations]


def _build_rows(
    observations: Sequence[Observation],
    machine: MachineLimits,
    *,
    form: str,
    memory_model: str,
    l2_usable: int,
    block_m: int | None,
    block_n: int | None,
    tail: Mapping[str, bool],
    launch_scopes: Sequence[str],
    k_fill: bool,
) -> list[dict[str, Any]]:
    indices = _launch_indices(observations, launch_scopes)
    return [
        _design_row(
            obs,
            machine,
            form,
            memory_model,
            l2_usable,
            block_m,
            block_n,
            tail=tail,
            launch_index=index,
            n_launch=max(1, len(launch_scopes)),
            k_fill=k_fill,
        )
        for obs, index in zip(observations, indices)
    ]


def _cross_validate(
    observations: Sequence[Observation],
    machine: MachineLimits,
    *,
    form: str,
    memory_model: str,
    l2_usable: int,
    block_m: int | None,
    block_n: int | None,
    tail: Mapping[str, bool],
    launch_scopes: Sequence[str],
    k_fill: bool,
    kind: str,
    folds: int,
) -> dict[str, Any]:
    """Train-only k-fold CV: refit on each complement, score the held-out fold."""
    if len(observations) < folds or folds < 2:
        return {"wape_pct": None, "folds": 0, "method": "insufficient_samples"}
    assignment = _fold_index(observations, folds)
    rows = _build_rows(
        observations,
        machine,
        form=form,
        memory_model=memory_model,
        l2_usable=l2_usable,
        block_m=block_m,
        block_n=block_n,
        tail=tail,
        launch_scopes=launch_scopes,
        k_fill=k_fill,
    )
    bounds = _parameter_bounds(
        max(1, len(launch_scopes)), kind, memory_model, k_fill, bool(tail.get("n"))
    )
    absolute = 0.0
    total = 0.0
    per_fold = []
    out_of_fold: list[tuple[int, float, float]] = []
    for fold in range(folds):
        train_index = [i for i, held in enumerate(assignment) if held != fold]
        test_index = [i for i, held in enumerate(assignment) if held == fold]
        if not train_index or not test_index:
            continue
        theta = _irls_fit([rows[i] for i in train_index], bounds)
        fold_abs = 0.0
        fold_total = 0.0
        for i in test_index:
            row = rows[i]
            prediction = _evaluate(row, theta)
            fold_abs += abs(prediction - row["target"])
            fold_total += row["target"]
            out_of_fold.append((i, prediction, row["target"]))
        absolute += fold_abs
        total += fold_total
        per_fold.append(
            {"fold": fold, "train": len(train_index), "test": len(test_index),
             "wape_pct": 100.0 * fold_abs / fold_total if fold_total > 0 else None}
        )
    return {
        "wape_pct": 100.0 * absolute / total if total > 0 else None,
        "folds": len(per_fold),
        "per_fold": per_fold,
        "method": "train_only_kfold",
        "grid_selected_on": "full_train",
        "out_of_fold": out_of_fold,
    }


def _memory_candidates(machine: MachineLimits) -> list[tuple[str, int]]:
    candidates: list[tuple[str, int]] = [("hbm", 0)]
    for fraction in L2_USABLE_FRACTIONS:
        usable = int(round(machine.l2_bytes * fraction))
        if usable > 0:
            candidates.append(("l2_blend", usable))
    return candidates


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[float, int, int]:
    # Strict train-WAPE order; ties prefer the coarser tile geometry.
    return (candidate["metrics"]["wape_pct"], candidate["block_m"] or 0, candidate["block_n"] or 0)


def _l2_bw_at_bound(theta: Sequence[float], n_launch: int, memory_model: str) -> bool:
    """True when the fitted L2 rate sits on a declared bound (not identified)."""
    if memory_model != "l2_blend":
        return False
    ratio = theta[n_launch + 2]
    lower = 1.0 / L2_BW_MULTIPLE_BOUNDS[1]
    upper = 1.0 / L2_BW_MULTIPLE_BOUNDS[0]
    return ratio <= lower * 1.01 or ratio >= upper * 0.99


def _select_best(
    candidates: Sequence[Mapping[str, Any]], identifying_threshold: int, n_launch: int
) -> Mapping[str, Any]:
    """Best candidate of one memory model for one form.

    A candidate whose L2 rate sits on a declared bound is a clipped parameter,
    not an estimate, so it is only kept when no interior solution fits within
    ``L2_USABLE_SELECTION_TOLERANCE`` of it.  Among equally good interior
    solutions, ``l2_usable`` and ``eta_memory`` trade off on a flat plateau once
    no binding row has an HBM part; there the smallest resident prefix is the
    conservative, identifiable claim.
    """
    best = min(candidates, key=_candidate_sort_key)
    if best["memory_model"] != "l2_blend" or not _l2_bw_at_bound(
        best["theta"], n_launch, best["memory_model"]
    ):
        return best
    limit = best["metrics"]["wape_pct"] * (1.0 + L2_USABLE_SELECTION_TOLERANCE)
    interior = [
        item
        for item in candidates
        if not _l2_bw_at_bound(item["theta"], n_launch, item["memory_model"])
        and item["metrics"]["wape_pct"] <= limit
    ]
    if not interior:
        return best
    identified = [
        item for item in interior if item["binding"]["counts"]["memory_hbm_part"] >= identifying_threshold
    ]
    if identified:
        return min(identified, key=_candidate_sort_key)
    return min(interior, key=lambda item: (item["l2_usable"],) + _candidate_sort_key(item))


def _difference(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return left - right


def _wape_from_triples(triples: Sequence[tuple[int, float, float]]) -> float | None:
    total = sum(target for _index, _prediction, target in triples)
    if total <= 0:
        return None
    return 100.0 * sum(abs(prediction - target) for _index, prediction, target in triples) / total


def _quality(metrics: Mapping[str, Any], cv_wape_pct: float | None) -> str:
    """Declared trust gate: reported either way, flagged when outside the bound."""
    if cv_wape_pct is None or cv_wape_pct > MAX_TRUSTED_CV_WAPE_PCT:
        return "low_confidence"
    if metrics["p90_ape_pct"] > MAX_TRUSTED_P90_APE_PCT:
        return "low_confidence"
    return "trusted"


def _cache_probe_flags(observations: Sequence[Observation], machine: MachineLimits) -> list[bool]:
    """Outcome-free subset where an L2 residency split can change the prediction."""
    flags = []
    for observation in observations:
        if observation.model_kind != "gemm":
            flags.append(observation.hbm_bytes <= machine.l2_bytes)
            continue
        intensity = observation.flops / max(1.0, observation.hbm_bytes)
        flags.append(
            observation.hbm_bytes <= machine.l2_bytes and intensity <= CACHE_PROBE_MAX_INTENSITY
        )
    return flags


# ---------------------------------------------------------------------------
# Reported breakdowns: M bands and dominant term
# ---------------------------------------------------------------------------
M_BAND_LIMITS = ((2, "m<=2"), (32, "m3-32"), (127, "m33-127"), (129, "m128-129"),
                 (512, "m130-512"), (None, "m>512"))


def _band_of(observation: Observation) -> str:
    if observation.model_kind != "gemm":
        return "vector"
    m = float(observation.shape.get("m") or 0.0)
    for limit, label in M_BAND_LIMITS:
        if limit is None or m <= limit:
            return label
    return "m>512"


def _breakdown(
    observations: Sequence[Observation],
    rows: Sequence[Mapping[str, Any]],
    theta: Sequence[float],
    out_of_fold: Sequence[tuple[int, float, float]],
) -> dict[str, Any]:
    """Train and out-of-fold WAPE by M band and by dominant term."""
    by_band: dict[str, dict[str, Any]] = {}
    by_term: dict[str, dict[str, Any]] = {}
    oof = {index: (prediction, target) for index, prediction, target in out_of_fold}
    for index, (observation, row) in enumerate(zip(observations, rows)):
        prediction = _evaluate(row, theta)
        target = row["target"]
        compute, memory = _term_values(row, theta)
        term = "compute" if compute >= memory else "memory"
        for table, key in ((by_band, _band_of(observation)), (by_term, term)):
            entry = table.setdefault(
                key, {"samples": 0, "train_abs": 0.0, "train_total": 0.0,
                      "cv_abs": 0.0, "cv_total": 0.0, "cv_samples": 0}
            )
            entry["samples"] += 1
            entry["train_abs"] += abs(prediction - target)
            entry["train_total"] += target
            if index in oof:
                held_out, held_target = oof[index]
                entry["cv_samples"] += 1
                entry["cv_abs"] += abs(held_out - held_target)
                entry["cv_total"] += held_target
    def finish(table: dict[str, dict[str, Any]]) -> dict[str, Any]:
        out = {}
        for key, entry in sorted(table.items()):
            out[key] = {
                "samples": entry["samples"],
                "train_wape_pct": (
                    100.0 * entry["train_abs"] / entry["train_total"] if entry["train_total"] else None
                ),
                "cv_samples": entry["cv_samples"],
                "cv_wape_pct": (
                    100.0 * entry["cv_abs"] / entry["cv_total"] if entry["cv_total"] else None
                ),
            }
        return out
    return {"by_m_band": finish(by_band), "by_dominant_term": finish(by_term)}


def _profile_interval(
    rows: Sequence[Mapping[str, Any]],
    bounds: Sequence[tuple[float, float]],
    theta: Sequence[float],
    index: int,
    best_wape: float,
    to_physical,
) -> list[float] | None:
    """Profile interval: the range whose refitted train WAPE stays in tolerance.

    Deterministic and outcome-free beyond the fitted data: the parameter is
    pinned on a log grid inside its declared bounds and everything else is
    refitted, so the reported interval is a plateau width, not a sampling CI.
    """
    low, high = bounds[index]
    if not (high > low):
        return None
    steps = CI_SCAN_POINTS
    accepted = []
    grid = [theta[index]]
    for step in range(steps + 1):
        fraction = step / float(steps)
        # Log grid for multiplicative rates, linear for additive intercepts.
        grid.append(low * (high / low) ** fraction if low > 0 else low + (high - low) * fraction)
    for value in grid:
        pinned = {index: value}
        candidate = _irls_fit(rows, bounds, iterations=max(3, IRLS_ITERATIONS // 4), pinned=pinned)
        wape = _score(rows, candidate)["wape_pct"]
        if wape <= best_wape * (1.0 + CI_TOLERANCE):
            accepted.append(value)
    if not accepted:
        return None
    endpoints = sorted((to_physical(min(accepted)), to_physical(max(accepted))))
    return [endpoints[0], endpoints[1]]


MODEL_FORMS = ("flat_additive", "wave_max")
FORM_DESCRIPTIONS = {
    "wave_max": "t = launch + max(flops_eff/(peak*eta*wave_eff), memory)",
    "flat_additive": "t = launch + flops_eff/(peak*eta) + memory",
}
TAIL_ORDER = ("m", "n", "k")


def _tile_choices(form: str, tail: Mapping[str, bool]) -> list[tuple[int | None, int | None]]:
    if form == "wave_max":
        return [(bm, bn) for bm in BLOCK_M_CHOICES for bn in BLOCK_N_CHOICES]
    block_m = None if not tail.get("m") else BLOCK_M_CHOICES
    block_n = None if not tail.get("n") else BLOCK_N_CHOICES
    if block_m is None and block_n is None:
        return [(None, None)]
    rows_m = block_m or (None,)
    rows_n = block_n or (None,)
    return [(bm, bn) for bm in rows_m for bn in rows_n]


def _fit_form(
    observations: Sequence[Observation],
    machine: MachineLimits,
    form: str,
    kind: str,
    folds: int,
    launch_scopes: Sequence[str],
    tail: Mapping[str, bool],
) -> dict[str, Any] | None:
    """Fit one model form: grid the tiles / resident prefix, then robust IRLS."""
    n_launch = max(1, len(launch_scopes))
    tile_choices = _tile_choices(form, tail)
    candidates: list[dict[str, Any]] = []
    for memory_model, l2_usable in _memory_candidates(machine):
        for k_fill in (False, True) if tail.get("k") else (False,):
            bounds = _parameter_bounds(n_launch, kind, memory_model, k_fill, bool(tail.get("n")))
            if k_fill and not any(_short_k(obs) for obs in observations):
                continue
            for block_m, block_n in tile_choices:
                rows = _build_rows(
                    observations,
                    machine,
                    form=form,
                    memory_model=memory_model,
                    l2_usable=l2_usable,
                    block_m=block_m,
                    block_n=block_n,
                    tail=tail,
                    launch_scopes=launch_scopes,
                    k_fill=k_fill,
                )
                theta = _irls_fit(rows, bounds)
                metrics = _score(rows, theta)
                candidates.append(
                    {
                        "form": form,
                        "tail": dict(tail),
                        "k_fill": k_fill,
                        "memory_model": memory_model,
                        "l2_usable": l2_usable,
                        "block_m": block_m,
                        "block_n": block_n,
                        "theta": theta,
                        "metrics": metrics,
                        "binding": _binding_detail(rows, theta),
                    }
                )
    if not candidates:
        return None
    identifying_threshold = max(
        MIN_IDENTIFYING_ROWS, int(math.ceil(MIN_IDENTIFYING_SHARE * len(observations)))
    )
    best_by_memory: dict[str, Mapping[str, Any]] = {}
    for memory_model in {candidate["memory_model"] for candidate in candidates}:
        best_by_memory[memory_model] = _select_best(
            [candidate for candidate in candidates if candidate["memory_model"] == memory_model],
            identifying_threshold,
            n_launch,
        )
    probe_flags = _cache_probe_flags(observations, machine)
    probe_index = {i for i, flag in enumerate(probe_flags) if flag}
    selection: dict[str, Any] = {
        "chosen_on": "train_only_cv",
        "margin_pct": MIN_CV_IMPROVEMENT_PCT,
        "max_full_cv_degradation_pct": MAX_FULL_CV_DEGRADATION_PCT,
        "cache_probe_samples": len(probe_index),
        "cache_probe_rule": (
            "hbm_bytes <= declared l2_bytes and flops/byte <= %.1f" % CACHE_PROBE_MAX_INTENSITY
        ),
    }
    for memory_model, candidate in best_by_memory.items():
        candidate["cv"] = _cross_validate(
            observations,
            machine,
            form=form,
            memory_model=memory_model,
            l2_usable=candidate["l2_usable"],
            block_m=candidate["block_m"],
            block_n=candidate["block_n"],
            tail=tail,
            launch_scopes=launch_scopes,
            k_fill=candidate["k_fill"],
            kind=kind,
            folds=folds,
        )
        out_of_fold = candidate["cv"].pop("out_of_fold", [])
        candidate["out_of_fold"] = out_of_fold
        candidate["probe_cv_wape_pct"] = _wape_from_triples(
            [triple for triple in out_of_fold if triple[0] in probe_index]
        )
        selection[memory_model] = {
            "train_wape_pct": candidate["metrics"]["wape_pct"],
            "cv_wape_pct": candidate["cv"].get("wape_pct"),
            "cache_probe_cv_wape_pct": candidate["probe_cv_wape_pct"],
            "block_m": candidate["block_m"],
            "block_n": candidate["block_n"],
            "l2_usable_bytes": candidate["l2_usable"] or None,
        }
    chosen = best_by_memory["hbm"]
    if "l2_blend" in best_by_memory:
        blend = best_by_memory["l2_blend"]
        hbm = best_by_memory["hbm"]
        full_gain = _difference(hbm["cv"].get("wape_pct"), blend["cv"].get("wape_pct"))
        probe_gain = _difference(hbm["probe_cv_wape_pct"], blend["probe_cv_wape_pct"])
        degrade = _difference(blend["cv"].get("wape_pct"), hbm["cv"].get("wape_pct"))
        selection["compared"] = True
        selection["cv_improvement_pct"] = full_gain
        selection["cache_probe_cv_improvement_pct"] = probe_gain
        selection["full_cv_degradation_pct"] = degrade
        if len(probe_index) >= MIN_CACHE_PROBE_SAMPLES:
            supported = (
                probe_gain is not None
                and probe_gain >= MIN_CV_IMPROVEMENT_PCT
                and (degrade is None or degrade <= MAX_FULL_CV_DEGRADATION_PCT)
            )
            rule = "cache_probe_cv_with_full_cv_guard"
        else:
            supported = full_gain is not None and full_gain >= MIN_CV_IMPROVEMENT_PCT
            rule = "full_cv_only_because_probe_subset_too_small"
        selection["decision_rule"] = rule
        if supported:
            chosen = blend
            selection["reason"] = "blend_supported_by_train_only_cv"
        else:
            selection["reason"] = "blend_not_supported_by_train_only_cv"
    else:
        selection["compared"] = False
        selection["reason"] = "blend_disabled_by_configuration"
    selection["chosen"] = chosen["memory_model"]
    return {
        "form": form,
        "chosen": chosen,
        "selection": selection,
        "cv_wape_pct": chosen["cv"].get("wape_pct"),
    }


def _fit_scope_group(
    observations: Sequence[Observation],
    machine: MachineLimits,
    *,
    family: str,
    recipe: str,
    protocol_signature: str,
    protocol_core_signature: str,
    timing_scope: str,
    launch_scopes: Sequence[str],
    fit_level: str,
    compute_dtype: str,
    folds: int,
) -> dict[str, Any]:
    """Select base form, tail terms and memory model by train-only CV."""
    kind = observations[0].model_kind
    # No tail term is assumed: m, n and k tails are forward-selected only if they
    # buy MIN_TAIL_IMPROVEMENT_PCT of train-only CV (padding a tile is physical,
    # but it trades off against eta, so it must be identified by the data).
    base_tail = {"m": False, "n": False, "k": False}
    fitted: dict[str, dict[str, Any]] = {}
    for form in MODEL_FORMS:
        result = _fit_form(observations, machine, form, kind, folds, launch_scopes, base_tail)
        if result is not None:
            fitted[form] = result
    form_selection: dict[str, Any] = {
        "chosen_on": "train_only_cv",
        "margin_pct": MIN_CV_IMPROVEMENT_PCT,
        "candidates": {
            form: {
                "description": FORM_DESCRIPTIONS[form],
                "train_wape_pct": result["chosen"]["metrics"]["wape_pct"],
                "cv_wape_pct": result["cv_wape_pct"],
                "memory_model": result["selection"]["chosen"],
                "block_m": result["chosen"]["block_m"],
                "block_n": result["chosen"]["block_n"],
            }
            for form, result in fitted.items()
        },
    }
    chosen_form = None
    for form in MODEL_FORMS:
        result = fitted.get(form)
        if result is None:
            continue
        if chosen_form is None:
            chosen_form = form
            continue
        incumbent = fitted[chosen_form]["cv_wape_pct"]
        challenger = result["cv_wape_pct"]
        if (
            incumbent is not None
            and challenger is not None
            and (incumbent - challenger) >= MIN_CV_IMPROVEMENT_PCT
        ):
            chosen_form = form
    assert chosen_form is not None
    form_selection["chosen"] = chosen_form
    form_selection["chosen_form_wape_pct"] = fitted[chosen_form]["cv_wape_pct"]
    if len(fitted) > 1:
        best_other = min(
            (result["cv_wape_pct"] for form, result in fitted.items()
             if form != chosen_form and result["cv_wape_pct"] is not None),
            default=None,
        )
        form_selection["best_rejected_cv_wape_pct"] = best_other
        form_selection["reason"] = (
            "chosen_form_declared_order"
            if best_other is None
            or (best_other - (fitted[chosen_form]["cv_wape_pct"] or 0.0)) < MIN_CV_IMPROVEMENT_PCT
            else "chosen_form_better_cv"
        )

    result = fitted[chosen_form]
    tail_selection: dict[str, Any] = {
        "margin_pct": MIN_TAIL_IMPROVEMENT_PCT,
        "trials": [
            {
                "tail": dict(base_tail),
                "cv_wape_pct": result["cv_wape_pct"],
                "train_wape_pct": result["chosen"]["metrics"]["wape_pct"],
                "adopted": True,
            }
        ],
        "chosen": dict(base_tail),
    }
    # Forward selection: each extra tail term must buy MIN_TAIL_IMPROVEMENT_PCT
    # of train-only CV.  The incumbent is always refitted with the tested tail.
    incumbent = result
    for tail_name in TAIL_ORDER:
        if kind != "gemm" and tail_name in ("m", "n"):
            continue
        trial_tail = dict(tail_selection["chosen"], **{tail_name: True})
        if trial_tail == tail_selection["chosen"]:
            continue
        if tail_name == "k" and not any(_short_k(obs) for obs in observations):
            tail_selection["trials"].append(
                {"tail": trial_tail, "cv_wape_pct": None, "adopted": False,
                 "reason": "no_short_k_observations"}
            )
            continue
        trial = _fit_form(observations, machine, chosen_form, kind, folds, launch_scopes, trial_tail)
        if trial is None:
            continue
        improved = (
            incumbent["cv_wape_pct"] is not None
            and trial["cv_wape_pct"] is not None
            and (incumbent["cv_wape_pct"] - trial["cv_wape_pct"]) >= MIN_TAIL_IMPROVEMENT_PCT
        )
        tail_selection["trials"].append(
            {
                "tail": dict(trial_tail),
                "cv_wape_pct": trial["cv_wape_pct"],
                "train_wape_pct": trial["chosen"]["metrics"]["wape_pct"],
                "adopted": bool(improved),
            }
        )
        if improved:
            incumbent = trial
            tail_selection["chosen"] = trial_tail
    result = incumbent

    chosen = result["chosen"]
    selection = dict(result["selection"])
    selection["tails"] = tail_selection["chosen"]
    theta = chosen["theta"]
    metrics = chosen["metrics"]
    n_launch = max(1, len(launch_scopes))
    parameters = _theta_to_parameters(
        theta, kind, machine, n_launch, chosen["memory_model"], chosen["k_fill"],
        bool(tail_selection["chosen"].get("n")),
    )
    counts = chosen["binding"]["counts"]
    n_align = bool(tail_selection["chosen"].get("n"))
    threshold = max(MIN_IDENTIFYING_ROWS, int(math.ceil(MIN_IDENTIFYING_SHARE * len(observations))))
    compute_name = "eta_compute" if kind == "gemm" else "vec_ops_per_s"
    identified = {
        "launch": True,
        compute_name: bool(counts["compute"] >= threshold),
        "eta_memory": bool(counts["memory_hbm_part"] >= threshold),
        "tile_geometry": chosen["block_m"] is not None or chosen["block_n"] is not None,
        "l2_split": bool(
            chosen["memory_model"] == "l2_blend" and counts["memory_l2_part"] >= threshold
        ),
        "k_fill": bool(chosen["k_fill"]),
    }
    warnings: list[str] = []
    # A rate that too few binding observations constrain is reported at the
    # declared ceiling instead of at an arbitrary solved value, and flagged.
    pinned: list[str] = []
    compute_rate = parameters["eta_compute"] if kind == "gemm" else parameters["vec_ops_per_s"]
    eta_memory = parameters["eta_memory"]
    if not identified[compute_name]:
        pinned.append(compute_name)
        warnings.append("%s_unidentified_pinned_to_declared_peak" % compute_name)
        compute_rate = 1.0 if kind == "gemm" else machine.vec_max_ops_per_s
    if not identified["eta_memory"]:
        pinned.append("eta_memory")
        warnings.append("eta_memory_unidentified_pinned_to_declared_peak")
        eta_memory = 1.0
    eta_compute = compute_rate if kind == "gemm" else None
    vec_ops_per_s = None if kind == "gemm" else compute_rate
    l2_bw = parameters["l2_bw_bytes_per_s"]
    if l2_bw is not None and l2_bw >= L2_BW_MULTIPLE_BOUNDS[1] * machine.peak_bw_bytes_per_s * 0.99:
        warnings.append("l2_bw_at_declared_cap")
    if l2_bw is not None and l2_bw <= L2_BW_MULTIPLE_BOUNDS[0] * machine.peak_bw_bytes_per_s * 1.01:
        warnings.append("l2_bw_at_declared_peak_blend_degenerates_to_hbm")
    for name, eta in (("eta_compute", eta_compute), ("eta_memory", eta_memory)):
        if eta is None or name in pinned:
            continue
        if eta <= ETA_BOUNDS[0] * 1.0000001:
            warnings.append("%s_at_lower_bound" % name)
        if eta >= 1.0 - 1e-9:
            warnings.append("%s_at_declared_peak" % name)
    launch_map = {
        scope: value for scope, value in zip(launch_scopes, parameters["launches_ms"])
    }
    scope_counts: dict[str, int] = {}
    for observation in observations:
        scope_counts[observation.timing_scope] = scope_counts.get(observation.timing_scope, 0) + 1
    for scope, count in sorted(scope_counts.items()):
        if count < MIN_IDENTIFYING_ROWS:
            warnings.append("scope_launch_weakly_identified:%s" % scope)
    for scope, value in launch_map.items():
        if value <= LAUNCH_BOUNDS_MS[0] + 1e-9:
            warnings.append("launch_at_lower_bound:%s" % scope)
            identified["launch"] = False
        elif value >= LAUNCH_BOUNDS_MS[1] - 1e-9:
            warnings.append("launch_at_upper_bound:%s" % scope)
            identified["launch"] = False
    if chosen["memory_model"] == "l2_blend" and chosen["l2_usable"] >= int(
        round(machine.l2_bytes * max(L2_USABLE_FRACTIONS))
    ):
        warnings.append("l2_usable_at_declared_capacity")
    if n_align and parameters["align_penalty"] is not None:
        if parameters["align_penalty"] <= ALIGN_PENALTY_BOUNDS[0] + 1e-9:
            warnings.append("align_penalty_at_lower_bound")
        if parameters["align_penalty"] >= ALIGN_PENALTY_BOUNDS[1] - 1e-9:
            warnings.append("align_penalty_at_upper_bound")
    if chosen["k_fill"] and parameters["k_fill_ms"] is not None:
        if parameters["k_fill_ms"] <= K_FILL_BOUNDS_MS[0] + 1e-9:
            warnings.append("k_fill_at_lower_bound")
            identified["k_fill"] = False
        if parameters["k_fill_ms"] >= K_FILL_BOUNDS_MS[1] - 1e-9:
            warnings.append("k_fill_at_upper_bound")
    if chosen["cv"].get("wape_pct") is not None and chosen["cv"]["wape_pct"] > MAX_TRUSTED_CV_WAPE_PCT:
        warnings.append("cv_wape_above_%.0f_pct" % MAX_TRUSTED_CV_WAPE_PCT)
    if metrics["p90_ape_pct"] > MAX_TRUSTED_P90_APE_PCT:
        warnings.append("p90_ape_above_%.0f_pct" % MAX_TRUSTED_P90_APE_PCT)
    if kind == "gemm" and not identified["eta_compute"]:
        warnings.append("compute_branch_weakly_identified")

    rows = _build_rows(
        observations,
        machine,
        form=chosen["form"],
        memory_model=chosen["memory_model"],
        l2_usable=chosen["l2_usable"],
        block_m=chosen["block_m"],
        block_n=chosen["block_n"],
        tail=tail_selection["chosen"],
        launch_scopes=launch_scopes,
        k_fill=chosen["k_fill"],
    )
    breakdown = _breakdown(observations, rows, theta, chosen["out_of_fold"])
    bounds = _parameter_bounds(
        n_launch, kind, chosen["memory_model"], chosen["k_fill"], n_align
    )
    i_rate = n_launch
    i_mem = i_rate + 1
    i_l2 = i_mem + 1
    n_mem = 3 if chosen["memory_model"] == "l2_blend" else 2
    i_align = i_mem + n_mem - 1
    intervals: dict[str, Any] = {}
    for scope_index, scope in enumerate(launch_scopes):
        interval = _profile_interval(
            rows, bounds, theta, scope_index, metrics["wape_pct"], lambda value: value
        )
        if interval is not None:
            intervals["launch_ms:%s" % scope] = interval
    rate_to_physical = (
        (lambda value: 1.0 / value) if kind == "gemm"
        else (lambda value: machine.vec_max_ops_per_s / value)
    )
    interval = _profile_interval(rows, bounds, theta, i_rate, metrics["wape_pct"], rate_to_physical)
    if interval is not None:
        intervals[compute_name] = interval
    interval = _profile_interval(rows, bounds, theta, i_mem, metrics["wape_pct"],
                                 lambda value: 1.0 / value)
    if interval is not None:
        intervals["eta_memory"] = interval
    if chosen["memory_model"] == "l2_blend":
        interval = _profile_interval(
            rows, bounds, theta, i_l2, metrics["wape_pct"],
            lambda value: machine.peak_bw_bytes_per_s / value,
        )
        if interval is not None:
            intervals["l2_bw_bytes_per_s"] = interval
    if n_align:
        interval = _profile_interval(rows, bounds, theta, i_align,
                                     metrics["wape_pct"], lambda value: value)
        if interval is not None:
            intervals["align_penalty"] = interval
    if chosen["k_fill"]:
        interval = _profile_interval(rows, bounds, theta, len(bounds) - 1,
                                     metrics["wape_pct"], lambda value: value)
        if interval is not None:
            intervals["k_fill_ms"] = interval
    return {
        "family": family,
        "recipe": recipe,
        "protocol_signature": protocol_signature,
        "protocol_core_signature": protocol_core_signature,
        "timing_scope": timing_scope,
        "launch_ms_by_scope": launch_map,
        "scope_samples": scope_counts,
        "model_kind": kind,
        "model_form": chosen["form"],
        "tail": tail_selection["chosen"],
        "k_fill": bool(chosen["k_fill"]),
        "align_penalty": parameters["align_penalty"],
        "fit_level": fit_level,
        "block_m": chosen["block_m"],
        "block_n": chosen["block_n"],
        "launch_ms": parameters["launch_ms"],
        "eta_compute": eta_compute,
        "eta_memory": eta_memory,
        "vec_ops_per_s": vec_ops_per_s,
        "k_fill_ms": parameters["k_fill_ms"],
        "memory_model": chosen["memory_model"],
        "l2_usable_bytes": chosen["l2_usable"] if chosen["memory_model"] == "l2_blend" else 0,
        "l2_bw_bytes_per_s": l2_bw,
        "compute_dtype": compute_dtype,
        "samples": len(observations),
        "raw_event_count": sum(obs.raw_event_count for obs in observations),
        "train_wape_pct": metrics["wape_pct"],
        "cv_wape_pct": chosen["cv"].get("wape_pct"),
        "median_ape_pct": metrics["median_ape_pct"],
        "p90_ape_pct": metrics["p90_ape_pct"],
        "max_ape_pct": metrics["max_ape_pct"],
        "binding_share": chosen["binding"]["share"],
        "binding_counts": chosen["binding"]["counts"],
        "identified": identified,
        "pinned_parameters": tuple(pinned),
        "intervals": intervals,
        "breakdown": breakdown,
        "memory_model_selection": selection,
        "model_selection": dict(form_selection, tails=tail_selection),
        "quality": _quality(metrics, chosen["cv"].get("wape_pct")),
        "warnings": tuple(warnings),
        "cv_detail": {"folds": chosen["cv"], "training_sample_count": len(observations)},
    }


# ---------------------------------------------------------------------------
# Public calibration object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PhysicalCalibration:
    """Versioned physical calibration artifact for one hardware key."""

    hardware_key: str
    arch_family: str
    machine: MachineLimits
    families: Mapping[str, FamilyParameters]
    fit_config: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    unmodeled: tuple[Mapping[str, Any], ...] = ()
    schema_version: str = SCHEMA_VERSION

    # -- construction -----------------------------------------------------
    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PhysicalCalibration":
        if not isinstance(value, Mapping):
            raise PhysicalCalibrationError("artifact_must_be_a_mapping")
        version = str(value.get("schema_version") or "")
        if version != SCHEMA_VERSION:
            raise PhysicalCalibrationError("unsupported_schema_version:%r" % (version,))
        hardware_key = str(value.get("hardware_key") or "")
        if not hardware_key:
            raise PhysicalCalibrationError("artifact_missing_hardware_key")
        machine_value = dict(value.get("machine") or {})
        machine_key = str(machine_value.get("hardware_key") or "")
        if machine_key and machine_key != hardware_key:
            raise PhysicalCalibrationError(
                "artifact_hardware_key_mismatch:%s!=%s" % (hardware_key, machine_key)
            )
        machine_value.setdefault("hardware_key", hardware_key)
        machine_value.setdefault("arch_family", value.get("arch_family") or "")
        machine = MachineLimits.from_mapping(machine_value)
        families_value = value.get("families") or {}
        if not isinstance(families_value, Mapping):
            raise PhysicalCalibrationError("artifact_families_must_be_a_mapping")
        families = {}
        for key, item in families_value.items():
            parameters = FamilyParameters.from_mapping(item)
            if parameters.model_kind not in ("gemm", "vector"):
                raise PhysicalCalibrationError("artifact_unknown_model_kind:%r" % (parameters.model_kind,))
            families[str(key)] = parameters
        return cls(
            hardware_key=hardware_key,
            arch_family=str(value.get("arch_family") or machine.arch_family),
            machine=machine,
            families=families,
            fit_config=dict(value.get("fit_config") or {}),
            provenance=dict(value.get("provenance") or {}),
            unmodeled=tuple(value.get("unmodeled") or ()),
            schema_version=version,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PhysicalCalibration":
        return cls.from_mapping(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hardware_key": self.hardware_key,
            "arch_family": self.arch_family,
            "machine": self.machine.to_dict(),
            "fit_config": dict(self.fit_config),
            "families": {key: value.to_dict() for key, value in sorted(self.families.items())},
            "provenance": dict(self.provenance),
            "unmodeled": [dict(item) for item in self.unmodeled],
        }

    # -- lookup -----------------------------------------------------------
    def parameters_for(
        self,
        family: str,
        recipe: str,
        *,
        protocol_signature: str = "",
        timing_scope: str = "",
    ) -> FamilyParameters | None:
        """Exact ``(family, recipe)`` match, else the pooled ``family`` fit.

        ``protocol_signature`` and ``timing_scope`` are hard filters: a query that
        cannot name its protocol is only answered when the artifact has exactly
        one candidate, never by borrowing another protocol's parameters.
        """
        exact = [
            item
            for item in self.families.values()
            if item.family == family and item.recipe == recipe
        ]
        pooled = [
            item
            for item in self.families.values()
            if item.family == family and item.recipe == "*"
        ]
        for group in (exact, pooled):
            if not group:
                continue
            candidates = self._filter_protocol(group, protocol_signature, timing_scope)
            if candidates:
                return candidates[0] if len(candidates) == 1 else None
        return None

    @staticmethod
    def _filter_protocol(
        group: Sequence[FamilyParameters], protocol_signature: str, timing_scope: str
    ) -> list[FamilyParameters]:
        candidates = list(group)
        if protocol_signature:
            candidates = [
                item
                for item in candidates
                if item.protocol_signature == protocol_signature
                or (item.fit_level == "family_scope_shared" and not item.protocol_signature)
            ]
        if timing_scope:
            # A shared-efficiency record answers only for the scopes it fitted a
            # launch intercept for; anything else is unmodeled, not borrowed.
            candidates = [
                item
                for item in candidates
                if item.timing_scope == timing_scope
                or (item.fit_level == "family_scope_shared" and timing_scope in item.launch_ms_by_scope)
            ]
        if not protocol_signature and not timing_scope and len(candidates) > 1:
            # Ambiguous protocol identity: refuse rather than guess.
            return []
        return candidates

    # -- prediction -------------------------------------------------------
    def predict_op(
        self,
        op: Mapping[str, Any],
        machine: Mapping[str, Any] | None = None,
        *,
        compute_dtype: str = "bf16",
    ) -> float | None:
        """Predicted duration in ms, or ``None`` when the op is unmodeled.

        The explicit reason for ``None`` (and the parameter provenance for a
        number) is available from :meth:`predict_op_explained`.
        """
        return self.predict_op_explained(op, machine, compute_dtype=compute_dtype).duration_ms

    def predict_op_explained(
        self,
        op: Mapping[str, Any],
        machine: Mapping[str, Any] | None = None,
        *,
        compute_dtype: str = "bf16",
    ) -> OpPrediction:
        if not isinstance(op, Mapping):
            raise PhysicalCalibrationError("op_must_be_a_mapping")
        family = str(op.get("family") or "")
        recipe = str(op.get("recipe") or "")
        signature = str(op.get("protocol_signature") or "")
        scope = str(op.get("timing_scope") or "")
        limits, source, from_artifact = self._resolve_machine(machine)
        mismatch = str((machine or {}).get("hardware_key") or "")
        if mismatch and mismatch != self.hardware_key:
            return self._unmodeled(op, "hardware_key_mismatch", source, from_artifact)
        parameters = self.parameters_for(
            family, recipe, protocol_signature=signature, timing_scope=scope
        )
        alias = ""
        if parameters is None and family in FAMILY_ALIASES:
            aliased = FAMILY_ALIASES[family]
            parameters = self.parameters_for(
                aliased, recipe, protocol_signature=signature, timing_scope=scope
            )
            if parameters is not None:
                alias = aliased
        if parameters is None:
            known = {item.family for item in self.families.values()}
            reason = "unmodeled_family" if family not in known else "unmodeled_recipe_or_protocol"
            return self._unmodeled(op, reason, source, from_artifact)
        dtype = str(op.get("dtype") or compute_dtype or parameters.compute_dtype)
        if dtype != parameters.compute_dtype:
            return self._unmodeled(
                op,
                "dtype_outside_fit:%s!=%s" % (dtype, parameters.compute_dtype),
                source,
                from_artifact,
                parameters,
            )
        shape = op.get("shape")
        shape = dict(shape) if isinstance(shape, Mapping) else {}
        for key in ("m", "n", "k", "elements"):
            if op.get(key) is not None and key not in shape:
                shape[key] = op[key]
        flops = _as_float(op.get("flops"), "flops", allow_none=True)
        hbm_bytes = _as_float(op.get("hbm_bytes"), "hbm_bytes", allow_none=True)
        elements = _as_float(op.get("elements"), "elements", allow_none=True)
        breakdown: dict[str, float] = {}
        if parameters.model_kind == "gemm":
            m = _as_float(shape.get("m"), "m", allow_none=True)
            n = _as_float(shape.get("n"), "n", allow_none=True)
            k = _as_float(shape.get("k"), "k", allow_none=True)
            if m is None or n is None or k is None:
                return self._unmodeled(op, "missing_shape_m_n_k", source, from_artifact, parameters)
            if flops is None:
                flops = 2.0 * m * n * k
            if hbm_bytes is None:
                hbm_bytes = (m * k + k * n + m * n) * DTYPE_BYTES.get(dtype, 2.0)
            wave_eff = 1.0
            shape_only = _ShapeOnly(family, shape, elements or 0.0, flops, hbm_bytes)
            effective_flops = _padded_flops(shape_only, parameters.block_m, parameters.tail or {})
            if parameters.model_form == "wave_max":
                blocks, waves, wave_eff = _gemm_geometry(
                    shape_only,
                    int(parameters.block_m or 1),
                    int(parameters.block_n or 1),
                    limits.aic_count,
                )
                breakdown["blocks"] = float(blocks)
                breakdown["waves"] = float(waves)
                breakdown["wave_eff"] = wave_eff
            breakdown["flops_effective"] = effective_flops
            compute_ms = (
                effective_flops
                / (limits.peak_flops * (parameters.eta_compute or 0.0) * max(wave_eff, 1e-9))
                * 1000.0
                if parameters.eta_compute
                else 0.0
            )
            breakdown["compute_ms"] = compute_ms
        else:
            if elements is None:
                # Accept the measured geometry directly (tokens x width, rows x
                # cols, heads x q x kv) instead of forcing the caller to reduce it.
                elements = _campaign_elements(shape)
            if elements is None:
                return self._unmodeled(op, "missing_elements", source, from_artifact, parameters)
            if hbm_bytes is None:
                hbm_bytes = elements * 2.0 * DTYPE_BYTES.get(dtype, 2.0)
            compute_ms = (
                elements / parameters.vec_ops_per_s * 1000.0
                if parameters.vec_ops_per_s
                else 0.0
            )
            breakdown["vector_ms"] = compute_ms
        memory_ms = self._memory_time(limits, parameters, hbm_bytes)
        if (
            parameters.tail.get("n")
            and parameters.align_penalty
            and parameters.block_n
            and shape.get("n") is not None
            and int(float(shape["n"])) % int(parameters.block_n) != 0
        ):
            # Extra full-bandwidth passes paid when the column tile does not divide n.
            align_ms = parameters.align_penalty * hbm_bytes / limits.peak_bw_bytes_per_s * 1000.0
            memory_ms += align_ms
            breakdown["align_penalty_ms"] = align_ms
        breakdown["memory_ms"] = memory_ms
        launch_ms = parameters.launch_ms
        if parameters.fit_level == "family_scope_shared":
            if not scope or scope not in parameters.launch_ms_by_scope:
                return self._unmodeled(
                    op, "unmodeled_timing_scope:%s" % (scope or "unspecified"),
                    source, from_artifact, parameters,
                )
            launch_ms = parameters.launch_ms_by_scope[scope]
        breakdown["launch_ms"] = launch_ms
        k_fill_ms = 0.0
        if parameters.k_fill and parameters.k_fill_ms:
            if float(shape.get("k") or 0.0) <= K_FILL_MAX_K:
                k_fill_ms = parameters.k_fill_ms
                breakdown["k_fill_ms"] = k_fill_ms
        if parameters.model_form == "wave_max":
            work_ms = max(compute_ms, memory_ms)
        else:
            work_ms = compute_ms + memory_ms
        breakdown["work_ms"] = work_ms
        return OpPrediction(
            duration_ms=launch_ms + k_fill_ms + work_ms,
            status="predicted",
            reason=(
                ("physical_calibration_via_family_alias:%s" % alias)
                if alias
                else (
                    "physical_calibration"
                    if parameters.quality == "trusted"
                    else "physical_calibration_low_confidence"
                )
            ),
            family=family,
            recipe=recipe,
            protocol_signature=parameters.protocol_signature,
            timing_scope=parameters.timing_scope,
            model_form=parameters.model_form,
            fit_level=parameters.fit_level,
            parameters={
                "block_m": parameters.block_m,
                "block_n": parameters.block_n,
                "launch_ms": parameters.launch_ms,
                "eta_compute": parameters.eta_compute,
                "eta_memory": parameters.eta_memory,
                "vec_ops_per_s": parameters.vec_ops_per_s,
                "memory_model": parameters.memory_model,
                "l2_usable_bytes": parameters.l2_usable_bytes,
                "l2_bw_bytes_per_s": parameters.l2_bw_bytes_per_s,
                "launch_ms_by_scope": dict(parameters.launch_ms_by_scope),
                "k_fill_ms": parameters.k_fill_ms,
                "align_penalty": parameters.align_penalty,
                "tail": dict(parameters.tail),
                "samples": parameters.samples,
                "cv_wape_pct": parameters.cv_wape_pct,
                "quality": parameters.quality,
                "family_alias_from": alias or None,
            },
            breakdown=breakdown,
            machine_source=source,
            machine_fields_from_artifact=from_artifact,
        )

    def _unmodeled(
        self,
        op: Mapping[str, Any],
        reason: str,
        source: str,
        from_artifact: Sequence[str],
        parameters: FamilyParameters | None = None,
    ) -> OpPrediction:
        return OpPrediction(
            duration_ms=None,
            status="unmodeled",
            reason=reason,
            family=str(op.get("family") or ""),
            recipe=str(op.get("recipe") or ""),
            protocol_signature=str(op.get("protocol_signature") or ""),
            timing_scope=str(op.get("timing_scope") or ""),
            model_form=parameters.model_form if parameters else "",
            fit_level=parameters.fit_level if parameters else "",
            machine_source=source,
            machine_fields_from_artifact=tuple(from_artifact),
        )

    # -- helpers ----------------------------------------------------------
    def _memory_time(
        self, limits: MachineLimits, parameters: FamilyParameters, hbm_bytes: float
    ) -> float:
        if hbm_bytes <= 0:
            return 0.0
        hbm_part, l2_part = _memory_split(
            hbm_bytes, parameters.l2_usable_bytes, limits.peak_bw_bytes_per_s
        )
        eta = parameters.eta_memory or ETA_BOUNDS[0]
        total = hbm_part / (limits.peak_bw_bytes_per_s * eta) * 1000.0
        if parameters.memory_model == "l2_blend" and l2_part > 0 and parameters.l2_bw_bytes_per_s:
            total += l2_part / parameters.l2_bw_bytes_per_s * 1000.0
        elif l2_part > 0:
            total += l2_part / (limits.peak_bw_bytes_per_s * eta) * 1000.0
        return total

    def _resolve_machine(
        self, machine: Mapping[str, Any] | None
    ) -> tuple[MachineLimits, str, tuple[str, ...]]:
        if not machine:
            return self.machine, "artifact", ()
        merged = self.machine.to_dict()
        fields_from_artifact = []
        key_map = {
            "peak_flops": "peak_flops",
            "peak_flops_per_s": "peak_flops",
            "peak_bw_bytes_per_s": "peak_bw_bytes_per_s",
            "hbm_bw_bytes_per_s": "peak_bw_bytes_per_s",
            "aic_count": "aic_count",
            "aiv_count": "aiv_count",
            "clock_mhz": "clock_mhz",
            "l2_bytes": "l2_bytes",
            "vec_peak_ops_per_s": "vec_peak_ops_per_s",
            "ops_per_cycle": "ops_per_cycle",
        }
        for source_key, artifact_key in key_map.items():
            if machine.get(source_key) is not None:
                merged[artifact_key] = machine[source_key]
            else:
                fields_from_artifact.append(artifact_key)
        if machine.get("hardware_key"):
            merged["hardware_key"] = str(machine["hardware_key"])
        try:
            limits = MachineLimits.from_mapping(merged)
        except PhysicalCalibrationError:
            return self.machine, "artifact", tuple(sorted(set(fields_from_artifact)))
        return limits, "call", tuple(sorted(set(fields_from_artifact)))

    def transfer_to(
        self,
        machine: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> "PhysicalCalibration":
        """Re-target the fitted efficiencies onto another machine (cross-SKU).

        Only *efficiencies* and the fitted L2 bandwidth cross the SKU boundary.
        The peaks come from the target machine, the resident prefix is capped by
        the target L2 capacity, and the launch intercept must be supplied per
        target from that target's own measurement -- a dispatch cost is a
        property of the protocol and the part, never of the source artifact.
        """
        limits = MachineLimits.from_mapping(machine)
        if limits.hardware_key == self.hardware_key:
            raise PhysicalCalibrationError("transfer_target_is_the_source_hardware_key")
        transferred: dict[str, FamilyParameters] = {}
        launch_sources = {}
        for target in targets:
            launch = _as_float(target.get("launch_ms"), "launch_ms")
            if launch is None or launch < 0:
                raise PhysicalCalibrationError("transfer_target_missing_launch_ms")
            source_key = str(target.get("source_family_key") or "")
            source = self.families.get(source_key)
            if source is None:
                raise PhysicalCalibrationError("transfer_source_family_unknown:%s" % source_key)
            key = str(target.get("family_key") or source_key)
            transferred[key] = dataclasses.replace(
                source,
                recipe=str(target.get("recipe") or source.recipe),
                protocol_signature=str(target.get("protocol_signature") or ""),
                timing_scope=str(target.get("timing_scope") or ""),
                fit_level="cross_sku_transfer",
                launch_ms=float(launch),
                l2_usable_bytes=int(min(source.l2_usable_bytes, limits.l2_bytes)),
                samples=0,
                raw_event_count=0,
                train_wape_pct=0.0,
                cv_wape_pct=None,
                median_ape_pct=0.0,
                p90_ape_pct=0.0,
                max_ape_pct=0.0,
                quality="transferred",
                warnings=tuple(
                    list(source.warnings)
                    + ["efficiencies_transferred_from:%s" % self.hardware_key,
                       "launch_from:%s" % str(target.get("launch_source") or "declared"),
                       "source_train_metrics_not_applicable"]
                ),
            )
            launch_sources[key] = str(target.get("launch_source") or "declared")
        if not transferred:
            raise PhysicalCalibrationError("transfer_without_targets")
        provenance = dict(self.provenance)
        provenance["transfer"] = {
            "source_hardware_key": self.hardware_key,
            "source_artifact_schema": self.schema_version,
            "target_hardware_key": limits.hardware_key,
            "launch_sources": launch_sources,
            "policy": (
                "efficiencies, optional L2 split and vector throughput transferred; peaks from "
                "the target machine; launch never transferred; resident prefix capped by the "
                "target L2 capacity"
            ),
            "fitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        provenance["source_runs"] = []
        provenance["condition_count"] = 0
        provenance["raw_event_count"] = 0
        provenance["training_hash"] = ""
        provenance["heldout_guard"] = {
            "status": "not_applicable",
            "note": "transfer artifact carries no target observations of its own",
        }
        return PhysicalCalibration(
            hardware_key=limits.hardware_key,
            arch_family=limits.arch_family or self.arch_family,
            machine=limits,
            families=transferred,
            fit_config=dict(self.fit_config, transfer=True),
            provenance=provenance,
            unmodeled=tuple(
                dict(item, reason="transferred_model_unmodeled") for item in self.unmodeled
            ),
            schema_version=self.schema_version,
        )

    def report(self) -> dict[str, Any]:
        rows = []
        for key, parameters in sorted(self.families.items()):
            rows.append(
                {
                    "key": key,
                    "family": parameters.family,
                    "recipe": parameters.recipe,
                    "timing_scope": parameters.timing_scope,
                    "fit_level": parameters.fit_level,
                    "model_kind": parameters.model_kind,
                    "model_form": parameters.model_form,
                    "tail": dict(parameters.tail),
                    "k_fill_ms": parameters.k_fill_ms,
                    "align_penalty": parameters.align_penalty,
                    "launch_ms_by_scope": dict(parameters.launch_ms_by_scope),
                    "scope_samples": dict(parameters.scope_samples),
                    "samples": parameters.samples,
                    "raw_event_count": parameters.raw_event_count,
                    "train_wape_pct": parameters.train_wape_pct,
                    "cv_wape_pct": parameters.cv_wape_pct,
                    "median_ape_pct": parameters.median_ape_pct,
                    "p90_ape_pct": parameters.p90_ape_pct,
                    "max_ape_pct": parameters.max_ape_pct,
                    "memory_model": parameters.memory_model,
                    "block_m": parameters.block_m,
                    "block_n": parameters.block_n,
                    "launch_ms": parameters.launch_ms,
                    "eta_compute": parameters.eta_compute,
                    "eta_memory": parameters.eta_memory,
                    "vec_ops_per_s": parameters.vec_ops_per_s,
                    "l2_usable_bytes": parameters.l2_usable_bytes,
                    "l2_bw_bytes_per_s": parameters.l2_bw_bytes_per_s,
                    "intervals": {key: list(value) for key, value in parameters.intervals.items()},
                    "breakdown": {key: dict(value) for key, value in parameters.breakdown.items()},
                    "model_selection": dict(parameters.model_selection),
                    "quality": parameters.quality,
                    "warnings": list(parameters.warnings),
                }
            )
        return {
            "schema_version": self.schema_version,
            "hardware_key": self.hardware_key,
            "arch_family": self.arch_family,
            "machine": self.machine.to_dict(),
            "families": rows,
            "unmodeled": [dict(item) for item in self.unmodeled],
            "provenance": dict(self.provenance),
            "fit_config": dict(self.fit_config),
        }


@dataclass(frozen=True)
class _ShapeOnly:
    """Minimal stand-in carrying the fields ``_gemm_geometry`` reads."""

    family: str
    shape: Mapping[str, Any]
    elements: float
    flops: float
    hbm_bytes: float
    model_kind: str = "gemm"


# ---------------------------------------------------------------------------
# Fit entry point
# ---------------------------------------------------------------------------
def _merge_repeat_conditions(
    observations: Sequence[Observation],
) -> tuple[list[Observation], dict[str, int]]:
    """One row per measured physical condition.

    Campaign directories can be mirrored (the same run appears under two names)
    and a condition can be re-measured.  Either way the target for a condition is
    a single number, so repeats are merged by median instead of being fed to the
    fit as independent rows (which would double-weight them).  Runs are told
    apart by ``condition_key`` when the runner provides one.
    """
    groups: dict[tuple[Any, ...], list[Observation]] = {}
    for observation in observations:
        if observation.condition_key:
            key = (observation.condition_key, observation.protocol_signature, observation.timing_scope)
        else:
            key = (
                observation.identity,
                observation.protocol_signature,
                observation.timing_scope,
                observation.package_tree,
                round(observation.duration_ms, 9),
            )
        groups.setdefault(key, []).append(observation)
    merged: list[Observation] = []
    duplicates = 0
    repeats = 0
    for key in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        group = groups[key]
        if len(group) == 1:
            merged.append(group[0])
            continue
        duplicates += len(group) - 1
        durations = sorted(item.duration_ms for item in group)
        middle = len(durations) // 2
        median = (
            durations[middle]
            if len(durations) % 2
            else 0.5 * (durations[middle - 1] + durations[middle])
        )
        if len({round(item.duration_ms, 9) for item in group}) == 1:
            # Byte-identical mirror of the same measurement, not a repeat.
            merged.append(group[0])
            continue
        repeats += 1
        merged.append(
            dataclasses.replace(
                group[0],
                duration_ms=median,
                raw_event_count=max(item.raw_event_count for item in group),
                target_estimator="repeat_condition_median",
            )
        )
    return merged, {
        "observations_deduplicated": duplicates,
        "repeat_conditions_merged": repeats,
        "conditions_after_merge": len(merged),
    }


def _count_by(observations: Sequence[Observation], key) -> dict[str, int]:
    counts: dict[str, int] = {}
    for observation in observations:
        value = key(observation)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _reject(rejections: dict[str, Any], reason: str, detail: str) -> None:
    entry = rejections.setdefault(reason, {"count": 0, "examples": []})
    entry["count"] += 1
    if len(entry["examples"]) < 5:
        entry["examples"].append(detail)


def fit_calibration(
    observations: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    machine: Mapping[str, Any],
    *,
    hardware_key: str = "",
    arch_family: str = "",
    compute_dtype: str = "bf16",
    family_kinds: Mapping[str, str] | None = None,
    min_samples: int = MIN_SAMPLES_PER_FIT,
    cv_folds: int = CV_FOLDS,
    accepted_package_trees: Sequence[str] = (),
    heldout_guard: HeldoutGuard | None = None,
    git_head: str | None = None,
    source_artifacts: Sequence[str] = (),
) -> PhysicalCalibration:
    """Fit the physical model on train observations only.

    ``observations`` is either the observation document
    (``{"schema_version": OBSERVATION_SCHEMA, "observations": [...]}``) or the
    bare list.  Non-train split, non-measured source, foreign hardware key,
    unsupported timing scope, empty protocol signature and missing shape are
    rejected with a counted reason before any parameter is estimated.
    """
    kind_table = dict(MODEL_KINDS)
    if family_kinds:
        kind_table.update({str(k): str(v) for k, v in family_kinds.items()})
    if isinstance(observations, Mapping):
        document = observations
        declared = str(document.get("schema_version") or "")
        if declared and declared != OBSERVATION_SCHEMA:
            raise PhysicalCalibrationError("unsupported_observation_schema:%r" % (declared,))
        rows = list(document.get("observations") or ())
    else:
        rows = list(observations)
    limits = MachineLimits.from_mapping(machine)
    target_key = hardware_key or limits.hardware_key
    target_arch = arch_family or limits.arch_family
    if not target_key:
        raise PhysicalCalibrationError("missing_hardware_key")

    rejections: dict[str, Any] = {}
    accepted: list[Observation] = []
    for row in rows:
        detail = ""
        try:
            family = str((row or {}).get("family") or "")
            if not family:
                raise PhysicalCalibrationError("missing_family")
            if family not in kind_table:
                _reject(rejections, "unsupported_family", family)
                continue
            observation = Observation.from_mapping(row, model_kind=kind_table[family])
            detail = observation.condition_key or observation.identity
        except PhysicalCalibrationError as exc:
            _reject(rejections, str(exc), str((row or {}).get("condition_key") or "?"))
            continue
        if observation.split != "train":
            _reject(rejections, "non_train_split:%s" % observation.split, detail)
            continue
        if observation.hardware_key and observation.hardware_key != target_key:
            _reject(rejections, "hardware_key_mismatch:%s" % observation.hardware_key, detail)
            continue
        if observation.source_class not in ACCEPTED_SOURCE_CLASSES:
            _reject(rejections, "non_measured_source_class:%s" % (observation.source_class or "unknown"), detail)
            continue
        if observation.timing_scope not in ACCEPTED_TIMING_SCOPES:
            _reject(rejections, "unsupported_timing_scope:%s" % (observation.timing_scope or "unknown"), detail)
            continue
        if not observation.protocol_signature:
            _reject(rejections, "ambiguous_protocol_signature", detail)
            continue
        if (
            accepted_package_trees
            and observation.package_tree
            and observation.package_tree not in accepted_package_trees
        ):
            # A run from a divergent mindie_sim package is not comparable data.
            _reject(
                rejections,
                "divergent_package_tree:%s" % observation.package_tree[:12],
                detail,
            )
            continue
        if observation.dtype and observation.dtype != compute_dtype:
            _reject(rejections, "dtype_outside_fit:%s" % observation.dtype, detail)
            continue
        accepted.append(observation)
    if not accepted:
        raise PhysicalCalibrationError("no_train_observations_after_rejection")
    accepted, merge_stats = _merge_repeat_conditions(accepted)
    guard_status = "absent"
    overlap: list[str] = []
    if heldout_guard is not None:
        overlap = heldout_guard.overlap(accepted)
        if overlap:
            raise PhysicalCalibrationError(
                "heldout_identity_in_training_set:%s" % (sorted(overlap)[:5],)
            )
        guard_status = "clean"

    # Per-scope fit identity: (family, recipe, protocol signature, timing scope).
    # Two timing scopes are different physical protocols (a batched interval
    # amortises dispatch), so the conservative baseline never shares them.
    groups: dict[tuple[str, str, str, str], list[Observation]] = {}
    for observation in accepted:
        groups.setdefault(
            (
                observation.family,
                observation.recipe,
                observation.protocol_signature,
                observation.timing_scope,
            ),
            [],
        ).append(observation)
    # Cross-scope identity: (family, protocol core signature).  The core excludes
    # the timing scope and the repeat settings, so a batched and a per-call
    # interval of the same physical protocol can share efficiencies while keeping
    # their own launch intercept.
    cores: dict[tuple[str, str], list[Observation]] = {}
    for observation in accepted:
        cores.setdefault(
            (observation.family, observation.protocol_core_signature), []
        ).append(observation)
    protocol_count = len({key[2] for key in groups})
    scope_count = len({key[3] for key in groups})
    protocol_of_family: dict[str, set[str]] = {}
    for family, _recipe, signature, scope in groups:
        protocol_of_family.setdefault(family, set()).add("%s|%s" % (signature, scope))

    families: dict[str, FamilyParameters] = {}
    unmodeled: list[dict[str, Any]] = []
    cv_details: dict[str, Any] = {}
    family_fits: dict[str, dict[str, Any]] = {}
    for (family, recipe, signature, scope), group in sorted(groups.items()):
        group_sorted = sorted(group, key=lambda item: (item.identity, item.condition_key))
        record: dict[str, Any] | None = None
        if len(group_sorted) >= min_samples:
            record = _fit_scope_group(
                group_sorted,
                limits,
                family=family,
                recipe=recipe,
                protocol_signature=signature,
                protocol_core_signature=group_sorted[0].protocol_core_signature,
                timing_scope=scope,
                launch_scopes=(scope,),
                fit_level="family_recipe_protocol",
                compute_dtype=compute_dtype,
                folds=cv_folds,
            )
        else:
            pooled = [
                item
                for item in accepted
                if item.family == family
                and item.protocol_signature == signature
                and item.timing_scope == scope
            ]
            if len(pooled) >= min_samples and len({item.recipe for item in pooled}) > 1:
                record = _fit_scope_group(
                    sorted(pooled, key=lambda item: (item.identity, item.condition_key)),
                    limits,
                    family=family,
                    recipe="*",
                    protocol_signature=signature,
                    protocol_core_signature=pooled[0].protocol_core_signature,
                    timing_scope=scope,
                    launch_scopes=(scope,),
                    fit_level="family_protocol",
                    compute_dtype=compute_dtype,
                    folds=cv_folds,
                )
            else:
                unmodeled.append(
                    {
                        "family": family,
                        "recipe": recipe,
                        "protocol_signature": signature,
                        "timing_scope": scope,
                        "reason": "insufficient_samples",
                        "samples": len(group_sorted),
                        "min_samples": min_samples,
                    }
                )
        if record is None:
            continue
        family_fits.setdefault(family, {"per_scope": [], "core": None})["per_scope"].append(record)
        family_fits[family]["core"] = record["protocol_core_signature"]

    # Cross-scope shared variant: one efficiency / tile / L2 set, one launch per
    # scope.  Adopted only when the pooled train-only CV beats the per-scope fits.
    for family, bundle in sorted(family_fits.items()):
        per_scope = bundle["per_scope"]
        core = bundle["core"]
        scopes = sorted({record["timing_scope"] for record in per_scope})
        if len(scopes) < 2:
            family_fits[family]["decision"] = {
                "compared": False,
                "reason": "single_timing_scope",
                "chosen": "per_scope_independent",
                "scopes": scopes,
            }
            continue
        union = sorted(
            [obs for obs in accepted if obs.family == family
             and obs.protocol_core_signature == core],
            key=lambda item: (item.timing_scope, item.identity, item.condition_key),
        )
        if len(union) < 2 * min_samples:
            family_fits[family]["decision"] = {
                "compared": False,
                "reason": "insufficient_samples_for_shared_fit",
                "chosen": "per_scope_independent",
                "scopes": scopes,
            }
            continue
        union_scopes = sorted({obs.timing_scope for obs in union})
        shared = _fit_scope_group(
            union,
            limits,
            family=family,
            recipe="*",
            protocol_signature="",
            protocol_core_signature=core,
            timing_scope="*",
            launch_scopes=union_scopes,
            fit_level="family_scope_shared",
            compute_dtype=compute_dtype,
            folds=cv_folds,
        )
        # The per-scope baseline is scored out-of-fold with the SAME fold
        # assignment as the shared model, so the two numbers are comparable.
        assignment = _fold_index(union, cv_folds)
        pooled_abs = 0.0
        pooled_total = 0.0
        for record in per_scope:
            scope_index = [
                i for i, obs in enumerate(union)
                if obs.timing_scope == record["timing_scope"]
            ]
            observations_scope = [union[i] for i in scope_index]
            folds_local = [assignment[i] for i in scope_index]
            rows = _build_rows(
                observations_scope,
                limits,
                form=record["model_form"],
                memory_model=record["memory_model"],
                l2_usable=record["l2_usable_bytes"],
                block_m=record["block_m"],
                block_n=record["block_n"],
                tail=record["tail"],
                launch_scopes=(record["timing_scope"],),
                k_fill=record["k_fill"],
            )
            bounds = _parameter_bounds(
                1, record["model_kind"], record["memory_model"], record["k_fill"],
                bool((record["tail"] or {}).get("n")),
            )
            for fold in range(cv_folds):
                train_rows = [row for row, held in zip(rows, folds_local) if held != fold]
                test_rows = [row for row, held in zip(rows, folds_local) if held == fold]
                if not train_rows or not test_rows:
                    continue
                theta = _irls_fit(train_rows, bounds)
                for row in test_rows:
                    prediction = _evaluate(row, theta)
                    pooled_abs += abs(prediction - row["target"])
                    pooled_total += row["target"]
        baseline_cv = 100.0 * pooled_abs / pooled_total if pooled_total else None
        gain = _difference(baseline_cv, shared["cv_wape_pct"])
        adopted = gain is not None and gain >= MIN_SHARED_SCOPE_IMPROVEMENT_PCT
        family_fits[family]["shared"] = shared
        family_fits[family]["decision"] = {
            "compared": True,
            "chosen": "shared_efficiencies_per_scope_launch" if adopted else "per_scope_independent",
            "scopes": union_scopes,
            "per_scope_pooled_cv_wape_pct": baseline_cv,
            "shared_cv_wape_pct": shared["cv_wape_pct"],
            "cv_improvement_pct": gain,
            "margin_pct": MIN_SHARED_SCOPE_IMPROVEMENT_PCT,
            "baseline_estimator": "per-scope model refit on 4/5 folds, scored on fold 0",
            "shared_scope_samples": shared["scope_samples"],
        }
        if adopted:
            family_fits[family]["per_scope"] = []

    for family, bundle in sorted(family_fits.items()):
        records = list(bundle["per_scope"])
        if bundle.get("shared") is not None and not records:
            records = [bundle["shared"]]
        for record in records:
            decisive = bundle.get("decision") or {}
            if decisive.get("compared"):
                record["memory_model_selection"] = dict(
                    record["memory_model_selection"], scope_selection=decisive
                )
            key = "%s/%s" % (record["family"], record["recipe"] or "*")
            if key in families:
                suffix = (record["protocol_signature"] or record["protocol_core_signature"]).replace(
                    "sha256:", ""
                )[:8]
                key = "%s#%s" % (key, suffix)
            cv_details[key] = record.pop("cv_detail", {})
            record.pop("_theta", None)
            families[key] = FamilyParameters.from_mapping(record)
    for family, signatures in sorted(protocol_of_family.items()):
        if len(signatures) > 1:
            unmodeled.append(
                {
                    "family": family,
                    "reason": "protocol_groups_kept_separate",
                    "protocol_signatures": sorted(signatures),
                    "note": (
                        "per-scope fits stay separate unless the shared-efficiency variant wins "
                        "on pooled train-only CV; a launch intercept is never transferred"
                    ),
                }
            )

    fitted_keys = {
        (item.family, item.recipe, item.protocol_signature, item.timing_scope)
        for item in families.values()
    }
    for family in sorted({item.family for item in accepted}):
        if not any(item[0] == family for item in fitted_keys):
            unmodeled.append(
                {"family": family, "reason": "no_family_fit", "note": "all groups below min_samples"}
            )

    training_rows = [
        {
            "condition_key": obs.condition_key,
            "identity": obs.identity,
            "family": obs.family,
            "recipe": obs.recipe,
            "protocol_signature": obs.protocol_signature,
            "duration_ms": round(obs.duration_ms, 9),
            "flops": obs.flops,
            "hbm_bytes": obs.hbm_bytes,
            "elements": obs.elements,
            "raw_event_count": obs.raw_event_count,
            "run_id": obs.run_id,
        }
        for obs in sorted(accepted, key=lambda item: (item.identity, item.condition_key))
    ]
    provenance = {
        "source_runs": sorted({obs.run_id for obs in accepted if obs.run_id}),
        "condition_count": len(accepted),
        "repeat_merge": merge_stats,
        "raw_event_count": sum(obs.raw_event_count for obs in accepted),
        "protocol_signature_count": protocol_count,
        "timing_scope_count": scope_count,
        "package_trees": {
            tree or "unknown": count
            for tree, count in sorted(_count_by(accepted, lambda obs: obs.package_tree).items())
        },
        "accepted_package_trees": list(accepted_package_trees),
        "scope_samples": {
            scope: count
            for scope, count in sorted(_count_by(accepted, lambda obs: obs.timing_scope).items())
        },
        "timing_scopes": sorted({obs.timing_scope for obs in accepted if obs.timing_scope}),
        "hardware_keys": sorted({obs.hardware_key for obs in accepted if obs.hardware_key}),
        "dies": sorted({obs.die for obs in accepted if obs.die}),
        "training_hash": _stable_digest(training_rows),
        "training_hash_scope": "condition_key|identity|family|recipe|protocol|duration|flops|bytes|elements|raw_event_count|run_id",
        "git_head": git_head,
        "fitted_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "split_policy": "train_only",
        "target_estimators": sorted({obs.target_estimator for obs in accepted if obs.target_estimator}),
        "target_estimator_note": (
            "median_of_trial_medians is the acceptance target; pooled_repeat_median is a weaker "
            "estimator of the same quantity and is recorded as a limitation"
            if "pooled_repeat_median" in {obs.target_estimator for obs in accepted}
            else "median_of_trial_medians or declared_duration_ms"
        ),
        "heldout_guard": (
            heldout_guard.to_dict(status=guard_status, overlap=overlap)
            if heldout_guard is not None
            else {"status": "absent", "warning": "no heldout guard declared; only split=train enforced"}
        ),
        "rejections": {
            "total": sum(entry["count"] for entry in rejections.values()),
            "by_reason": {reason: entry["count"] for reason, entry in sorted(rejections.items())},
            "examples": {reason: entry["examples"] for reason, entry in sorted(rejections.items())},
        },
        "source_artifacts": list(source_artifacts),
        "generator": "mindie_sim.a3_physical_calibration",
        "generator_schema": SCHEMA_VERSION,
        "cv_detail": cv_details,
    }
    fit_config = {
        "block_m_choices": list(BLOCK_M_CHOICES),
        "block_n_choices": list(BLOCK_N_CHOICES),
        "launch_bounds_ms": list(LAUNCH_BOUNDS_MS),
        "eta_bounds": list(ETA_BOUNDS),
        "l2_usable_fractions": list(L2_USABLE_FRACTIONS),
        "l2_bw_multiple_bounds": list(L2_BW_MULTIPLE_BOUNDS),
        "huber_k": HUBER_K,
        "irls_iterations": IRLS_ITERATIONS,
        "min_samples": min_samples,
        "cv_folds": cv_folds,
        "cv_method": "train_only_kfold",
        "selection_criterion": "train_wape_pct",
        "memory_model_rule": "blend requires >=%.1f pct train-only CV improvement" % MIN_CV_IMPROVEMENT_PCT,
        "family_aliases": dict(FAMILY_ALIASES),
        "model_forms": list(MODEL_FORMS),
        "tail_terms": ["m", "n", "k"],
        "min_tail_improvement_pct": MIN_TAIL_IMPROVEMENT_PCT,
        "shared_scope_min_improvement_pct": MIN_SHARED_SCOPE_IMPROVEMENT_PCT,
        "k_fill_max_k": K_FILL_MAX_K,
        "k_fill_bounds_ms": list(K_FILL_BOUNDS_MS),
        "model_form_descriptions": dict(FORM_DESCRIPTIONS),
        "model_selection_margin_pct": MIN_CV_IMPROVEMENT_PCT,
        "cache_probe_min_samples": MIN_CACHE_PROBE_SAMPLES,
        "min_identifying_rows": MIN_IDENTIFYING_ROWS,
        "min_identifying_share": MIN_IDENTIFYING_SHARE,
        "max_trusted_cv_wape_pct": MAX_TRUSTED_CV_WAPE_PCT,
        "max_trusted_p90_ape_pct": MAX_TRUSTED_P90_APE_PCT,
        "ci_tolerance": CI_TOLERANCE,
        "vec_issue_max_per_aiv": VEC_ISSUE_MAX_PER_AIV,
        "vec_throughput_min_fraction": VEC_THROUGHPUT_MIN_FRACTION,
        "accepted_timing_scopes": list(ACCEPTED_TIMING_SCOPES),
        "protocol_core_fields": [
            "timer", "preallocated", "cache_policy", "compute_dtype", "weight_dtype",
            "activation_dtype", "accumulation_dtype", "quantization", "layout",
        ],
        "protocol_core_defaults": {
            "compute_dtype": "bf16",
            "weight_dtype": "bf16",
            "activation_dtype": "bf16",
            "accumulation_dtype": "backend_default",
            "quantization": "none",
            "layout": "contiguous",
            "cache_policy": DECLARED_CACHE_POLICY,
        },
        "protocol_core_excluded": [
            "timing_scope", "measurement_scope", "warmup", "warmup_scope", "iterations", "trials",
        ],
        "accepted_source_classes": list(ACCEPTED_SOURCE_CLASSES),
        "compute_dtype": compute_dtype,
    }
    return PhysicalCalibration(
        hardware_key=target_key,
        arch_family=target_arch,
        machine=limits,
        families=families,
        fit_config=fit_config,
        provenance=provenance,
        unmodeled=tuple(unmodeled),
    )


# ---------------------------------------------------------------------------
# Observation readers
# ---------------------------------------------------------------------------
def read_observations_json(path: str | Path) -> dict[str, Any]:
    """Read an observation document and validate its declared schema."""
    value = json.loads(Path(path).read_text())
    if isinstance(value, Mapping):
        declared = str(value.get("schema_version") or "")
        if declared and declared != OBSERVATION_SCHEMA:
            raise PhysicalCalibrationError("unsupported_observation_schema:%r" % (declared,))
        return dict(value)
    if isinstance(value, list):
        return {"schema_version": OBSERVATION_SCHEMA, "observations": value}
    raise PhysicalCalibrationError("observation_document_must_be_mapping_or_list")


_VECTOR_BYTES_PER_ELEMENT = {
    # Declared logical byte conventions for the vector families: tensors read and
    # written per element, times the dtype size (see the readers).
    "copy": 2.0,
    "cast": 2.0,
    "rmsnorm": 2.0,
    "softmax": 2.0,
    "masked_softmax": 2.0,
    "quantize": 2.0,
    "rope": 2.0,
    "add": 3.0,
    "swiglu": 3.0,
}
_SQLITE_COLUMNS = (
    "sample_key, op_family, recipe, shape_json, dtype, split, duration_median, measurement_count,"
    " run_ids_json, source_classes_json, timing_scope, protocol_json, hardware_key, environment_json"
)


def read_sqlite_observations(
    path: str | Path,
    *,
    hardware_key: str = "",
    arch_family: str = "",
    include_splits: Sequence[str] = ("train",),
    die: str = "",
) -> dict[str, Any]:
    """Fallback reader for a ``sample_statistics`` calibration store.

    Duration target is ``duration_median`` (median of per-trial event medians is
    not available in this store); ``raw_samples_ms`` and ``raw_event_count``
    come from the ``measurements`` table, so the artifact stays traceable to raw
    events.  Logical bytes follow the declared per-family convention: a GEMM
    touches ``m*k + k*n + m*n`` elements, ``copy`` reads and writes one array,
    ``add`` reads two and writes one.
    """
    connection = sqlite3.connect("file:%s?mode=ro" % Path(path), uri=True)
    try:
        rows = connection.execute(
            "select %s from sample_statistics order by sample_key" % _SQLITE_COLUMNS
        ).fetchall()
        raw_by_sample: dict[str, list[float]] = {}
        for sample_key, duration in connection.execute(
            "select sample_key, duration_ms from measurements order by sample_key, repeat_index"
        ):
            raw_by_sample.setdefault(sample_key, []).append(float(duration))
    finally:
        connection.close()
    observations = []
    skipped = {"split": 0, "family": 0, "shape": 0, "duration": 0}
    for row in rows:
        (sample_key, op_family, recipe, shape_json, dtype, split, duration_median,
         measurement_count, run_ids_json, source_classes_json, timing_scope,
         protocol_json, row_hardware_key, environment_json) = row
        if include_splits and split not in include_splits:
            skipped["split"] += 1
            continue
        if op_family not in MODEL_KINDS:
            skipped["family"] += 1
            continue
        kind = MODEL_KINDS[op_family]
        shape = json.loads(shape_json)
        itemsize = DTYPE_BYTES.get(str(dtype).lower(), 2.0)
        shape = {k: float(v) for k, v in shape.items() if v is not None}
        if kind == "gemm":
            m, n, k = shape.get("m"), shape.get("n"), shape.get("k")
            if not (m and n and k):
                skipped["shape"] += 1
                continue
            flops = 2.0 * m * n * k
            hbm_bytes = (m * k + k * n + m * n) * itemsize
            elements = None
        else:
            elements = shape.get("elements")
            if not elements:
                skipped["shape"] += 1
                continue
            flops = 0.0
            hbm_bytes = elements * itemsize * _VECTOR_BYTES_PER_ELEMENT[op_family]
        duration = float(duration_median) if duration_median else 0.0
        if duration <= 0:
            skipped["duration"] += 1
            continue
        protocol = json.loads(protocol_json or "{}")
        environment = json.loads(environment_json or "{}")
        sources = json.loads(source_classes_json or "[]")
        run_ids = json.loads(run_ids_json or "[]")
        observations.append(
            {
                "condition_key": sample_key,
                "hardware_key": str(row_hardware_key or hardware_key),
                "die": str(environment.get("die") or die or ""),
                "arch_family": arch_family,
                "family": op_family,
                "recipe": str(recipe or ""),
                "protocol_signature": protocol_signature(protocol),
                "run_id": str(run_ids[0]) if run_ids else "",
                "split": split,
                "shape": shape,
                "elements": elements,
                "flops": flops,
                "hbm_bytes": hbm_bytes,
                "duration_ms": duration,
        "target_estimator": "pooled_repeat_median",
                "raw_samples_ms": raw_by_sample.get(sample_key, []),
                "raw_event_count": int(measurement_count or 0),
                "dtype": str(dtype or ""),
                "source_class": str(sources[0]) if sources else "unknown",
                "timing_scope": str(timing_scope or ""),
                "protocol": protocol,
                "protocol_core_signature": protocol_core_signature(protocol),
            }
        )
    if not hardware_key:
        hardware_key = next((item["hardware_key"] for item in observations if item["hardware_key"]), "")
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "source": "sqlite:%s" % Path(path).name,
        "source_path": str(Path(path)),
        "hardware_key": hardware_key,
        "arch_family": arch_family,
        "skipped": skipped,
        "observations": observations,
    }


DECLARED_CACHE_POLICY = "same_preallocated_buffers_reused_no_cache_flush"

_DTYPE_ALIASES = {
    "bfloat16": "bf16", "bf16": "bf16", "float16": "fp16", "fp16": "fp16",
    "float32": "fp32", "fp32": "fp32", "float8": "fp8", "fp8": "fp8",
}


def _normalise_dtype(value: Any) -> str:
    text = str(value or "").strip().lower()
    return _DTYPE_ALIASES.get(text, text)


def protocol_core(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Timing-scope-independent protocol identity.

    A batched interval and a per-call interval of the same kernel differ in
    *dispatch treatment*, not in the physical protocol, so the scope, the
    measurement-scope prose and the repeat settings (warmup / iterations /
    trials) are excluded here; they are reported separately and each scope keeps
    its own launch intercept.  Missing dtype keys normalise to the declared
    defaults that the A2/A3 runners attest (bf16 / none / contiguous).
    """
    if not isinstance(protocol, Mapping) or not protocol:
        return {}
    return {
        "timer": str(protocol.get("timer") or ""),
        "preallocated": bool(protocol.get("preallocated", False)),
        # The old A3 store predates the explicit cache_policy field; every run in
        # this programme was measured under the declared policy below, so a
        # missing key normalises to it instead of splitting the protocol identity.
        "cache_policy": str(protocol.get("cache_policy") or DECLARED_CACHE_POLICY),
        "compute_dtype": _normalise_dtype(
            protocol.get("requested_compute_dtype") or protocol.get("compute_dtype") or "bf16"
        ),
        "weight_dtype": _normalise_dtype(protocol.get("weight_dtype") or "bf16"),
        "activation_dtype": _normalise_dtype(protocol.get("activation_dtype") or "bf16"),
        "accumulation_dtype": str(
            protocol.get("requested_accumulation_dtype")
            or protocol.get("accumulation_dtype")
            or "backend_default"
        ),
        "quantization": str(protocol.get("quantization") or "none"),
        "layout": str(protocol.get("layout") or "contiguous"),
    }


def protocol_core_signature(protocol: Mapping[str, Any]) -> str:
    """Stable digest of the scope-independent protocol identity."""
    core = protocol_core(protocol)
    if not core:
        return ""
    return "sha256:%s" % _stable_digest(core)[:16]


def protocol_signature(protocol: Mapping[str, Any]) -> str:
    """Stable digest of the timing-affecting protocol fields.

    Two conditions with different timing identity (timer, scope, warmup,
    iterations, trials, allocation, dtype, cache policy) must never share a fit.
    """
    if not isinstance(protocol, Mapping) or not protocol:
        return ""
    keys = (
        "timer", "timing_scope", "measurement_scope", "iterations", "trials", "warmup",
        "warmup_scope", "preallocated", "cache_policy", "requested_compute_dtype",
        "weight_dtype", "activation_dtype", "accumulation_dtype", "quantization", "layout",
    )
    selected = {key: protocol[key] for key in keys if key in protocol}
    return "sha256:%s" % _stable_digest(selected)[:16]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _read_git_head(repository: str | Path | None) -> str | None:
    """Resolve HEAD from ``.git`` files only (no subprocess, no network).

    Handles a linked worktree, where ``.git`` is a file containing
    ``gitdir: <path>`` rather than a directory.
    """
    if not repository:
        return None
    root = Path(repository)
    entry = root / ".git"
    try:
        if entry.is_file():
            text = entry.read_text().strip()
            if not text.startswith("gitdir:"):
                return None
            git_dir = Path(text[len("gitdir:"):].strip())
            if not git_dir.is_absolute():
                git_dir = (root / git_dir).resolve()
        elif entry.is_dir():
            git_dir = entry
        else:
            return None
        head = (git_dir / "HEAD").read_text().strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head[len("ref:"):].strip()
        search = [git_dir]
        common = git_dir / "commondir"
        if common.is_file():
            shared = Path(common.read_text().strip())
            search.append(shared if shared.is_absolute() else (git_dir / shared).resolve())
        for directory in search:
            ref_file = directory / ref
            if ref_file.is_file():
                return ref_file.read_text().strip() or None
            packed = directory / "packed-refs"
            if packed.is_file():
                for line in packed.read_text().splitlines():
                    if line and not line.startswith(("#", "^")):
                        sha, _, name = line.partition(" ")
                        if name.strip() == ref:
                            return sha.strip()
        return None
    except OSError:
        return None


def _load_observations(args: Any) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    sources: list[str] = []
    for path in args.observations or ():
        document = read_observations_json(path)
        rows.extend(document.get("observations") or ())
        sources.append("json:%s" % path)
    for path in getattr(args, "observations_campaign", None) or ():
        document = read_campaign_results(
            path,
            hardware_key=args.hardware_key or "",
            arch_family=args.arch_family or "",
        )
        rows.extend(document.get("observations") or ())
        sources.append("campaign:%s" % path)
    for path in args.observations_sqlite or ():
        document = read_sqlite_observations(
            path,
            hardware_key=args.hardware_key or "",
            arch_family=args.arch_family or "",
        )
        rows.extend(document.get("observations") or ())
        sources.append("sqlite:%s" % path)
    return rows, sources


def _command_fit(args: Any) -> dict[str, Any]:
    rows, sources = _load_observations(args)
    if not rows:
        raise PhysicalCalibrationError("no_observations_loaded")
    machine = json.loads(Path(args.machine).read_text())
    guard = HeldoutGuard.from_files(args.heldout_guard) if args.heldout_guard else None
    calibration = fit_calibration(
        {"schema_version": OBSERVATION_SCHEMA, "observations": rows},
        machine,
        hardware_key=args.hardware_key or "",
        arch_family=args.arch_family or "",
        compute_dtype=args.compute_dtype,
        min_samples=args.min_samples,
        cv_folds=args.cv_folds,
        accepted_package_trees=args.accept_package_tree or (),
        heldout_guard=guard,
        git_head=args.git_head or _read_git_head(args.repo_root),
        source_artifacts=sources + list(args.source_artifact or ()),
    )
    payload = calibration.to_dict()
    if args.output:
        output = Path(args.output)
        if output.parent and not output.parent.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {
        "status": "fitted",
        "output": str(args.output) if args.output else None,
        "hardware_key": calibration.hardware_key,
        "families": len(calibration.families),
        "provenance": calibration.provenance,
        "report": calibration.report(),
    }


def _command_predict(args: Any) -> dict[str, Any]:
    calibration = PhysicalCalibration.load(args.artifact)
    op = json.loads(Path(args.op).read_text())
    if "op" in op and isinstance(op["op"], Mapping):
        op = op["op"]
    machine = json.loads(Path(args.machine).read_text()) if args.machine else None
    prediction = calibration.predict_op_explained(op, machine, compute_dtype=args.compute_dtype)
    return prediction.to_dict()


def _command_report(args: Any) -> dict[str, Any]:
    calibration = PhysicalCalibration.load(args.artifact)
    return calibration.report()


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        prog="mindie_sim.a3_physical_calibration",
        description="Fit / query the physical (roofline) calibration artifact.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="fit an artifact from train observations")
    fit.add_argument("--observations", action="extend", nargs="+", default=[],
                     help="mindie-sim.observations/1 JSON document (repeatable)")
    fit.add_argument("--observations-sqlite", action="extend", nargs="+", default=[],
                     help="sample_statistics calibration store (repeatable fallback reader)")
    fit.add_argument("--observations-campaign", action="extend", nargs="+", default=[],
                     help="operator-campaign results.json (repeatable; batched and per-call scopes kept apart)")
    fit.add_argument("--machine", required=True, help="machine limits JSON")
    fit.add_argument("--output", default="", help="artifact output path")
    fit.add_argument("--accept-package-tree", action="extend", nargs="+", default=[],
                     help="only accept observations measured by these mindie_sim tree sha256 values")
    fit.add_argument("--heldout-guard", action="extend", nargs="+", default=[],
                     help="declared heldout split declaration / score / results JSON")
    fit.add_argument("--hardware-key", default="")
    fit.add_argument("--arch-family", default="")
    fit.add_argument("--compute-dtype", default="bf16")
    fit.add_argument("--min-samples", type=int, default=MIN_SAMPLES_PER_FIT)
    fit.add_argument("--cv-folds", type=int, default=CV_FOLDS)
    fit.add_argument("--source-artifact", action="extend", nargs="+", default=[])
    fit.add_argument("--git-head", default="")
    fit.add_argument("--repo-root", default=".")
    fit.set_defaults(handler=_command_fit)

    predict = sub.add_parser("predict", help="predict one operator duration")
    predict.add_argument("--artifact", required=True)
    predict.add_argument("--op", required=True, help="operator mapping JSON")
    predict.add_argument("--machine", default="")
    predict.add_argument("--compute-dtype", default="bf16")
    predict.set_defaults(handler=_command_predict)

    report = sub.add_parser("report", help="print the artifact metrics and provenance")
    report.add_argument("--artifact", required=True)
    report.set_defaults(handler=_command_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import sys

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = args.handler(args)
    except PhysicalCalibrationError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0




def _campaign_family(recipe: str, shape: Mapping[str, Any], kind: str) -> str:
    """Map a campaign recipe/shape onto a modelled operator family."""
    token = ("%s %s %s" % (kind, recipe, shape)).lower()
    if {"m", "n", "k"} <= set(shape):
        return "gemm"
    aliases = (
        ("rmsnorm", ("rms_norm", "rmsnorm")),
        ("swiglu", ("swiglu", "sigmoid_mul", "silu_mul")),
        # Checked before the generic softmax alias: a masked softmax is a distinct
        # kernel with its own mask traffic, so it keeps its own family fit.
        ("masked_softmax", ("masked_softmax", "scaled_masked_softmax")),
        ("softmax", ("softmax",)),
        ("quantize", ("quant",)),
        ("rope", ("rope", "rotary")),
        ("copy", ("copy",)),
        ("add", ("add", "residual")),
        ("cast", ("cast", "to_", "dtype_convert")),
    )
    for family, needles in aliases:
        if any(needle in token for needle in needles):
            return family
    return kind or recipe or "unknown"


def _campaign_elements(shape: Mapping[str, Any]) -> float | None:
    """Element count of a non-GEMM shape: ``elements`` or ``tokens x width``."""
    if shape.get("elements") is not None:
        return float(shape["elements"])
    if shape.get("tokens") is not None and shape.get("width") is not None:
        return float(shape["tokens"]) * float(shape["width"])
    if shape.get("rows") is not None and shape.get("cols") is not None:
        return float(shape["rows"]) * float(shape["cols"])
    if all(shape.get(key) is not None for key in ("heads", "q", "kv")):
        # Score-matrix work of a (masked) softmax: heads x q x kv elements.
        return float(shape["heads"]) * float(shape["q"]) * float(shape["kv"])
    return None


def read_campaign_results(
    path: str | Path,
    *,
    hardware_key: str = "",
    arch_family: str = "",
    include_splits: Sequence[str] = ("train",),
) -> dict[str, Any]:
    """Read an operator-campaign ``results.json`` (the A2/A3 runner schema).

    Emits the batched and the per-call interval of the same condition as two
    observations with distinct ``timing_scope`` values, because they are
    different physical protocols: a batched interval amortises dispatch over
    ``repeats_per_sample`` calls, a per-call interval includes it.  A2 runs
    report both (``duration_ms`` batched, ``per_call_median_ms`` per call) and
    carry ``trial_medians_ms``; A3 gap-fill runs report
    ``device_event_warm_operator_batched_amortized`` with two trials.

    Logical bytes follow the same declared convention as the other readers:
    a GEMM touches ``m*k + k*n + m*n`` elements, and each vector family reads /
    writes ``_VECTOR_BYTES_PER_ELEMENT`` tensors per element.
    """
    value = json.loads(Path(path).read_text())
    results = value.get("results") or value.get("cases") or []
    document_protocol = value.get("protocol") or {}
    document_environment = value.get("environment") if isinstance(value.get("environment"), Mapping) else {}
    package_tree = str(
        (document_protocol.get("package_provenance") or {}).get("tree_sha256")
        or document_environment.get("mindie_sim_tree_sha256")
        or ""
    )
    observations = []
    skipped = {"split": 0, "shape": 0, "duration": 0, "family": 0, "failed_case": 0}
    for item in results:
        if not isinstance(item, Mapping):
            skipped["shape"] += 1
            continue
        if str(item.get("split") or "train") not in include_splits:
            skipped["split"] += 1
            continue
        if str(item.get("status") or "completed") != "completed":
            # A failed case (NaN output, dispatch error) is not a measurement.
            skipped["failed_case"] = skipped.get("failed_case", 0) + 1
            continue
        details = item.get("details") if isinstance(item.get("details"), Mapping) else {}
        request = item.get("request") if isinstance(item.get("request"), Mapping) else {}
        shape_source = details.get("shape") or request.get("shape") or {}
        raw_shape = dict(shape_source) if isinstance(shape_source, Mapping) else {}
        protocol = dict(document_protocol)
        protocol.update(item.get("protocol") or {})
        recipe = str(
            details.get("kind") or item.get("worker_recipe") or protocol.get("worker_recipe") or ""
        )
        kind = str(item.get("execution_kind") or protocol.get("execution_kind") or "primitive")
        family = _campaign_family(recipe, raw_shape, kind)
        if family not in MODEL_KINDS:
            skipped["family"] += 1
            continue
        # Propagate a die identity from the protocol block when the item has one.
        die = ""
        identity = protocol.get("die_identity") or item.get("die_identity")
        if isinstance(identity, Mapping):
            die = "phy%s" % identity.get("phy_id") if identity.get("phy_id") is not None else str(
                identity.get("logical") or ""
            )
        dtype = str(
            protocol.get("requested_compute_dtype")
            or details.get("dtype")
            or item.get("dtype")
            or "bf16"
        )
        itemsize = DTYPE_BYTES.get(dtype.lower(), 2.0)
        if family == "gemm":
            shape = {
                key: float(raw_shape[key])
                for key in ("m", "n", "k")
                if raw_shape.get(key) is not None
            }
            m, n, k = shape.get("m"), shape.get("n"), shape.get("k")
            if not (m and n and k):
                skipped["shape"] += 1
                continue
            flops = 2.0 * m * n * k
            hbm_bytes = (m * k + k * n + m * n) * itemsize
            elements = None
        else:
            elements = _campaign_elements(raw_shape)
            declared_read = details.get("actual_read_bytes")
            declared_write = details.get("actual_write_bytes")
            declared_bytes = None
            if declared_read is not None or declared_write is not None:
                declared_bytes = float(declared_read or 0.0) + float(declared_write or 0.0)
            if not elements and declared_bytes:
                # Byte-view references report bytes, not elements; express the
                # element count in the fitted dtype so the vector term stays
                # comparable across families.
                elements = declared_bytes / itemsize
            if not elements:
                skipped["shape"] += 1
                continue
            shape = {"elements": float(elements)}
            flops = 0.0
            hbm_bytes = (
                declared_bytes
                if declared_bytes
                else elements * itemsize * _VECTOR_BYTES_PER_ELEMENT.get(family, 2.0)
            )
        base = {
            "hardware_key": hardware_key,
            "arch_family": arch_family,
            "die": die,
            "family": family,
            "recipe": recipe,
            "run_id": str(item.get("run_id") or value.get("run_id") or ""),
            "split": str(item.get("split") or "train"),
            "shape": shape,
            "elements": elements,
            "flops": flops,
            "hbm_bytes": hbm_bytes,
            "dtype": dtype,
            "source_class": str(item.get("source_class") or protocol.get("source_class") or "unknown"),
            "package_tree": package_tree,
        }
        condition_key = str(
            item.get("condition_key") or item.get("case_id") or item.get("measurement_id") or ""
        )
        campaign = item.get("campaign") if isinstance(item.get("campaign"), Mapping) else {}
        clock = campaign.get("clock_in_window") if isinstance(campaign.get("clock_in_window"), Mapping) else {}
        declared_scope = str(
            item.get("timing_scope")
            or protocol.get("timing_mode")
            or protocol.get("timing_scope")
            or "device_event_warm_batched"
        )
        batched_duration = item.get("duration_ms", item.get("batched_per_call_median_ms"))
        if batched_duration and float(batched_duration) > 0:
            batched_protocol = dict(protocol, timing_scope=declared_scope)
            observations.append(
                dict(
                    base,
                    condition_key="%s|batched" % condition_key,
                    protocol_signature=protocol_signature(batched_protocol),
                    protocol_core_signature=protocol_core_signature(batched_protocol),
                    duration_ms=float(batched_duration),
                    trial_medians_ms=item.get("trial_medians_ms"),
                    raw_samples_ms=item.get("raw_samples_ms"),
                    raw_event_count=int(item.get("trials_count") or 0)
                    or len(item.get("raw_samples_ms") or ())
                    or int(protocol.get("executed_trials") or protocol.get("trials") or 1),
                    timing_scope=declared_scope,
                    target_estimator="median_of_trial_medians",
                    protocol=batched_protocol,
                    observed_clock={"cur_freq_mhz": clock.get("cur_freq_mhz"),
                                    "rated_freq_mhz": clock.get("rated_freq_mhz"),
                                    "loaded_freq_floor_mhz": campaign.get("loaded_freq_floor_mhz")},
                )
            )
        else:
            skipped["duration"] += 1
        per_call = item.get("per_call_median_ms")
        if per_call and float(per_call) > 0:
            per_call_protocol = dict(protocol, timing_scope="device_event_warm_per_call")
            per_call_protocol.pop("timing_mode", None)
            diagnostics = [float(x) for x in (item.get("per_call_diagnostic_ms") or ())]
            observations.append(
                dict(
                    base,
                    condition_key="%s|per_call" % condition_key,
                    protocol_signature=protocol_signature(per_call_protocol),
                    protocol_core_signature=protocol_core_signature(per_call_protocol),
                    duration_ms=float(per_call),
                    raw_samples_ms=diagnostics,
                    raw_event_count=len(diagnostics) or 1,
                    timing_scope="device_event_warm_per_call",
                    target_estimator="declared_duration_ms",
                    protocol=per_call_protocol,
                    derived_from="per_call_diagnostic_ms",
                    observed_clock={"cur_freq_mhz": clock.get("cur_freq_mhz"),
                                    "rated_freq_mhz": clock.get("rated_freq_mhz"),
                                    "loaded_freq_floor_mhz": campaign.get("loaded_freq_floor_mhz")},
                )
            )
    if not hardware_key:
        hardware_key = str(document_environment.get("device_name") or "")
    return {
        "schema_version": OBSERVATION_SCHEMA,
        "source": "campaign_results:%s" % Path(path).name,
        "source_path": str(Path(path)),
        "hardware_key": hardware_key,
        "arch_family": arch_family,
        "protocol": document_protocol,
        "package_tree": package_tree,
        "skipped": skipped,
        "observations": observations,
    }


if __name__ == "__main__":
    raise SystemExit(main())
