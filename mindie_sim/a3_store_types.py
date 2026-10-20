"""Calibration sample keys, timing-scope normalization and record type."""
from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


def canonical_timing_scope(value: Any) -> str:
    """Map free-form worker prose to a stable compatibility class.

    Detailed timer/warmup/cache wording stays in the protocol; only genuinely
    compatible classes may share a calibration grouping key.
    """
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if "host" in text or "wall" in text:
        return "host_wall"
    if "cpu" in text or "python" in text:
        return "python_cpu_reference_only"
    if ("device" in text or "event" in text or "npu" in text) and "warm" in text:
        return "device_event_warm"
    if "communication" in text or "lane" in text:
        return "communication_lane"
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _semantic_key(payload: Mapping[str, Any]) -> str:
    metadata = {
        "revision", "commit", "git_commit", "source_sha", "source", "created_at",
        "updated_at", "run_id", "case_id", "repeat_index", "details",
    }
    clean = {key: value for key, value in payload.items() if key not in metadata}
    return "sha256:" + hashlib.sha256(_json(clean).encode("utf-8")).hexdigest()


def _measurement_key(run_id: str, case_id: str, repeat_index: int, sample_key: str) -> str:
    if run_id and case_id:
        return "sha256:" + hashlib.sha256(
            f"{run_id}|{case_id}|{int(repeat_index)}|{sample_key}".encode("utf-8")
        ).hexdigest()
    return "sha256:" + hashlib.sha256(f"{sample_key}|{uuid.uuid4()}".encode("utf-8")).hexdigest()


def _percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


@dataclass(frozen=True)
class SampleRecord:
    id: int
    measurement_key: str
    sample_key: str
    hardware_key: str
    op_family: str
    phase: str
    recipe: str
    shape: Mapping[str, float]
    dtype: str
    layout: str
    features: Mapping[str, Any]
    competition: Mapping[str, Any]
    duration_ms: float
    analytic_ms: float
    source: str
    source_class: str
    synthetic: bool
    revision: str
    run_id: str
    case_id: str
    repeat_index: int
    split: str
    timing_scope: str
    protocol: Mapping[str, Any]
    environment: Mapping[str, Any]
    details: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "measurement_key": self.measurement_key,
            "sample_key": self.sample_key, "hardware_key": self.hardware_key,
            "op_family": self.op_family, "phase": self.phase,
            "recipe": self.recipe, "shape": dict(self.shape),
            "dtype": self.dtype, "layout": self.layout,
            "features": dict(self.features), "competition": dict(self.competition),
            "duration_ms": self.duration_ms, "analytic_ms": self.analytic_ms,
            "source": self.source, "source_class": self.source_class,
            "synthetic": self.synthetic, "revision": self.revision,
            "run_id": self.run_id, "case_id": self.case_id,
            "repeat_index": self.repeat_index, "split": self.split,
            "timing_scope": self.timing_scope,
            "protocol": dict(self.protocol), "environment": dict(self.environment),
            "details": dict(self.details),
        }

    def as_observation(self) -> CalibrationObservation:
        return CalibrationObservation(
            op_family=self.op_family, phase=self.phase, recipe=self.recipe,
            shape=tuple(sorted((str(k), float(v)) for k, v in self.shape.items())),
            duration_ms=self.duration_ms, analytic_ms=self.analytic_ms,
            source_class=self.source_class, source=self.source,
            revision=self.revision, machine_key=self.hardware_key,
            synthetic=self.synthetic,
            metadata=tuple(sorted((str(k), str(v)) for k, v in self.competition.items())),
            dtype=self.dtype, layout=self.layout,
            features=tuple(sorted((str(k), str(v)) for k, v in self.features.items())),
        )




__all__ = [
    "canonical_timing_scope", "SampleRecord", "_utc_now", "_json",
    "_semantic_key", "_measurement_key", "_percentile",
]
