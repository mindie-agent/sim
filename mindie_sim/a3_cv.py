"""Measured paired C/V stream calibration over a shared lane.

Kept separate from communication calibration; both execute the common resource
engine and preserve raw protocol semantics at ingestion.

[Invariants]
  * Hardware identity is an exact join: a keyed query only sees conditions
    measured on that same machine key.  An unbound condition is refused (and
    counted in the result's ``identity`` block) instead of being pooled across
    SKUs.
  * No trained condition means the analytic HBM/vector estimate with its reason,
    never a calibrated zero.

[Timeouts & Locks]
  Pure in-memory arithmetic with one bounded resource-engine run; no I/O,
  locks, device or sleep.

[Side-Effect Policy]
  No filesystem/network/global writes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

from .a3_resources import ResourceEngine, ResourceSpec
from .a3_types import A3InputError, MachineSpec

if TYPE_CHECKING:
    from .a3_calibration_store import SQLiteCalibrationStore


@dataclass(frozen=True)
class CvCondition:
    mode: str
    v_elements: int
    requested_v_delay_ms: float
    c_shape_mkn: tuple[int, int, int]
    dtype: str
    c_repeats: int
    v_repeats: int
    c_isolated_ms: float
    v_isolated_ms: float
    span_ms: float
    c_slowdown: float
    v_slowdown: float
    samples: int
    hardware_key: str = ""
    timing_scope: str = ""


class CvCalibration:
    """Measured paired-stream lane work and shared-lane contention model."""

    def __init__(self, conditions: Sequence[CvCondition]) -> None:
        self.conditions = list(conditions)

    @staticmethod
    def _identity_matches(item: CvCondition, hardware_key: str) -> bool:
        """Exact machine join.  An unkeyed query is an explicit fleet query."""
        key = str(hardware_key or "").strip()
        if not key:
            return True
        identity = str(item.hardware_key or "").strip()
        return bool(identity) and identity == key

    def _identity_audit(
        self, shape: tuple[int, int, int], hardware_key: str,
    ) -> dict[str, Any] | None:
        """Conditions for this C-shape refused for the queried machine."""
        key = str(hardware_key or "").strip()
        if not key:
            return None
        scope = [item for item in self.conditions if item.c_shape_mkn == shape]
        refused = [item for item in scope if not self._identity_matches(item, key)]
        if not refused:
            return None
        unbound = [item for item in refused if not str(item.hardware_key or "").strip()]
        return {
            "machine_key": key,
            "refused_conditions": {
                "count": len(refused),
                "unbound": len(unbound),
                "mismatched": len(refused) - len(unbound),
                "examples": [
                    {
                        "mode": item.mode,
                        "c_shape_mkn": list(item.c_shape_mkn),
                        "v_elements": int(item.v_elements),
                        "hardware_key": str(item.hardware_key or ""),
                        "reason": (
                            "hardware_key_unbound"
                            if not str(item.hardware_key or "").strip()
                            else "hardware_key_mismatch"
                        ),
                    }
                    for item in refused[:8]
                ],
            },
        }

    @classmethod
    def from_store(cls, store: SQLiteCalibrationStore) -> "CvCalibration":
        rows = [item for item in store.statistics(split="train") if item["phase"] == "concurrent"]
        records: dict[tuple[Any, ...], dict[str, Any]] = {}
        for row in rows:
            features = row["features"]
            competition = row["competition"]
            c_shape = tuple(int(v) for v in competition.get("c_shape_mkn", (0, 0, 0)))
            key = (
                str(features.get("mode", competition.get("mode", "unknown"))),
                int(competition.get("v_elements", 0)),
                float(competition.get("requested_v_delay_ms", 0.0)),
                c_shape,
                int(competition.get("c_repeats", 1)),
                int(competition.get("v_repeats", 1)),
                str(row.get("hardware_key", "")),
                str(row.get("timing_scope", "")),
                str(row.get("dtype", "")),
            )
            record = records.setdefault(key, {"streams": {}, "samples": 0})
            stream = str(features.get("stream", ""))
            record["streams"][stream] = float(row["duration_median"])
            record["samples"] = max(record["samples"], int(row["calibratable_count"]))
            record["hardware_key"] = str(row.get("hardware_key", ""))
            record["timing_scope"] = str(row.get("timing_scope", ""))
            record["dtype"] = str(row.get("dtype", ""))
        isolated_c = {
            (values["streams"].get("C", 0.0), key[1], key[3], key[6], key[7], key[8])
            for key, values in records.items() if key[0] == "c_only"
        }
        isolated_v = {
            (values["streams"].get("V", 0.0), key[1], key[3], key[6], key[7], key[8])
            for key, values in records.items() if key[0] == "v_only"
        }
        c_by_v = {
            (size, shape, hw, timing, dtype): value
            for value, size, shape, hw, timing, dtype in isolated_c
        }
        v_by_v = {
            (size, shape, hw, timing, dtype): value
            for value, size, shape, hw, timing, dtype in isolated_v
        }
        conditions = []
        for key, values in sorted(records.items()):
            mode, v_elements, delay, c_shape, c_repeats, v_repeats, hw, timing, dtype = key
            iso_key = (v_elements, c_shape, hw, timing, dtype)
            c_iso = c_by_v.get(iso_key, values["streams"].get("C", 0.0))
            v_iso = v_by_v.get(iso_key, values["streams"].get("V", 0.0))
            c_dur = values["streams"].get("C", c_iso)
            v_dur = values["streams"].get("V", v_iso)
            span = values["streams"].get("span", max(c_dur, v_dur))
            conditions.append(CvCondition(
                mode=mode, v_elements=v_elements, requested_v_delay_ms=delay,
                c_shape_mkn=c_shape, dtype=dtype or "bf16",
                c_repeats=c_repeats, v_repeats=v_repeats,
                c_isolated_ms=c_iso, v_isolated_ms=v_iso, span_ms=span,
                c_slowdown=c_dur / c_iso if c_iso > 0 else 1.0,
                v_slowdown=v_dur / v_iso if v_iso > 0 else 1.0,
                samples=int(values["samples"]),
                hardware_key=hw, timing_scope=timing,
            ))
        return cls(conditions)

    def _isolated_value(
        self, field: str, mode: str, c_shape: tuple[int, int, int], size: int,
        *, hardware_key: str = "", timing_scope: str = "", dtype: str = "",
    ) -> float:
        points = sorted(
            (item.v_elements, getattr(item, field)) for item in self.conditions
            if item.mode == mode and item.c_shape_mkn == c_shape
            and getattr(item, field) > 0
            # Exact machine identity; an unbound condition is never pooled.
            and self._identity_matches(item, hardware_key)
            and (not timing_scope or not item.timing_scope or item.timing_scope == timing_scope)
            and (not dtype or not item.dtype or item.dtype == dtype)
        )
        if not points:
            return 0.0
        if len(points) == 1:
            return points[0][1]
        for point_size, value in points:
            if point_size == size:
                return value
        if size <= points[0][0]:
            return points[0][1] * size / points[0][0] if points[0][0] > 0 else points[0][1]
        if size >= points[-1][0]:
            (x0, y0), (x1, y1) = points[-2], points[-1]
            return y0 + (y1 - y0) * (size - x0) / (x1 - x0) if x1 != x0 else y1
        lower, upper = points[0], points[-1]
        for left, right in zip(points, points[1:]):
            if left[0] <= size <= right[0]:
                lower, upper = left, right
                break
        if lower[1] <= 0 or upper[1] <= 0:
            return lower[1] + (upper[1] - lower[1]) * (size - lower[0]) / (upper[0] - lower[0])
        # log-log interpolation in size
        return math.exp(
            math.log(lower[1]) + (
                math.log(upper[1]) - math.log(lower[1])
            ) * (math.log(size + 1) - math.log(lower[0] + 1))
            / (math.log(upper[0] + 1) - math.log(lower[0] + 1))
        )

    def _analytic_cv(
        self, shape: tuple[int, int, int], v_elements: int,
        requested_v_delay_ms: float, c_reps: int, v_reps: int,
        dtype: str, machine: MachineSpec | None,
    ) -> dict[str, Any]:
        from .a3_real_data import baseline_copy, baseline_gemm

        request = {
            "op_family": "concurrent_cv", "phase": "concurrent",
            "recipe": "analytic_cv_shared_lane",
            "shape": {"c_m": shape[0], "c_n": shape[1], "c_k": shape[2],
                      "v_elements": int(v_elements)},
            "features": {"stream": "CV", "dtype": dtype},
        }
        if machine is None:
            return {
                "status": "uncalibrated_needs_machine",
                "reason": "c_shape_not_trained_and_no_machine",
                "c_shape_mkn": list(shape), "span_ms": None,
                "measurement_request": request,
            }
        c_ms = baseline_gemm(machine, shape[0], shape[1], shape[2]) * max(1, c_reps)
        v_ms = baseline_copy(machine, int(v_elements)) * max(1, v_reps)
        engine = ResourceEngine({"lane": ResourceSpec("lane", 1.0)})
        engine.add_task("all_reduce", {"lane": c_ms})
        engine.add_task("p2p", {"lane": v_ms},
                        release_ms=max(0.0, float(requested_v_delay_ms)))
        result = engine.run()
        return {
            "status": "uncalibrated_analytic_cv",
            "reason": "missing_trained_cv_condition; analytic HBM/vector estimate",
            "c_shape_mkn": list(shape), "dtype": dtype,
            "c_ms": result["completed_at_ms"]["all_reduce"],
            "v_ms": result["completed_at_ms"]["p2p"],
            "span_ms": result["makespan_ms"],
            "measurement_request": request,
            "assumptions": ["declared machine peaks/bandwidth", "no heldout value"],
        }

    def predict(
        self, *, c_shape_mkn: Sequence[int], v_elements: int,
        requested_v_delay_ms: float = 0.0,
        machine: MachineSpec | None = None,
        c_repetitions: int | None = None, v_repetitions: int | None = None,
        dtype: str = "bf16",
        hardware_key: str = "",
        timing_scope: str = "",
    ) -> dict[str, Any]:
        shape = tuple(int(v) for v in c_shape_mkn)
        if len(shape) != 3:
            raise A3InputError("cv_c_shape_invalid")
        if machine is not None and not hardware_key:
            hardware_key = machine.hardware_key
        identity = self._identity_audit(shape, hardware_key)
        trained = [
            item for item in self.conditions if item.c_shape_mkn == shape
            # Exact machine identity; a `:`-prefix match used to let one machine
            # borrow another's concurrency measurement.
            and self._identity_matches(item, hardware_key)
            and (not timing_scope or not item.timing_scope or item.timing_scope == timing_scope)
            and (not dtype or not item.dtype or item.dtype == dtype)
        ]

        def finish(result: dict[str, Any]) -> dict[str, Any]:
            if identity is None:
                return result
            out = dict(result)
            out["identity"] = identity
            if not str(out.get("status", "")).startswith("calibrated"):
                out["identity_reason"] = (
                    "measured_conditions_exist_only_for_another_machine_key"
                )
            return out

        if not trained:
            reference = None
            c_reps = int(c_repetitions or 1)
            v_reps = int(v_repetitions or 1)
            return finish(self._analytic_cv(
                shape, v_elements, requested_v_delay_ms, c_reps, v_reps,
                dtype, machine,
            ))
        cv_conditions = [item for item in trained if item.mode == "cv"]
        reference = min(cv_conditions, key=lambda item: abs(math.log((item.v_elements + 1) / (v_elements + 1)))) if cv_conditions else None
        c_reps = int(c_repetitions if c_repetitions is not None else (reference.c_repeats if reference else 1))
        v_reps = int(v_repetitions if v_repetitions is not None else (reference.v_repeats if reference else 1))
        c_iso = self._isolated_value(
            "c_isolated_ms", "c_only", shape, v_elements,
            hardware_key=hardware_key, timing_scope=timing_scope, dtype=dtype,
        )
        v_iso = self._isolated_value(
            "v_isolated_ms", "v_only", shape, v_elements,
            hardware_key=hardware_key, timing_scope=timing_scope, dtype=dtype,
        )
        if c_iso <= 0 or v_iso <= 0:
            return finish(self._analytic_cv(
                shape, v_elements, requested_v_delay_ms, c_reps, v_reps,
                dtype, machine,
            ))
        # Fit joint span scale from measured CV conditions against shared-lane base.
        scales = []
        for item in cv_conditions:
            base_engine = ResourceEngine({"lane": ResourceSpec("lane", 1.0)})
            base_engine.add_task("c", {"lane": item.c_isolated_ms * item.c_repeats})
            base_engine.add_task("v", {"lane": item.v_isolated_ms * item.v_repeats})
            base = base_engine.run()["makespan_ms"]
            if base > 0:
                scales.append((item.v_elements, item.span_ms * max(item.c_repeats, item.v_repeats) / base))
        if scales:
            scales.sort()
            if len(scales) == 1:
                scale = scales[0][1]
            else:
                (x0, y0), (x1, y1) = scales[0], scales[-1]
                scale = y0 + (y1 - y0) * (v_elements - x0) / (x1 - x0) if x1 != x0 else y0
        else:
            scale = 1.0
        engine = ResourceEngine({"lane": ResourceSpec("lane", 1.0)})
        engine.add_task("all_reduce", {"lane": c_iso * c_reps * scale})
        engine.add_task("p2p", {"lane": v_iso * v_reps * scale},
                        release_ms=max(0.0, float(requested_v_delay_ms)))
        result = engine.run()
        completed = result["completed_at_ms"]
        v_start = max(0.0, float(requested_v_delay_ms))
        v_duration = max(0.0, completed["p2p"] - v_start)
        return finish({
            "status": "calibrated_cv_shared_lane",
            "condition": {
                "c_shape_mkn": list(shape), "dtype": dtype,
                "c_repetitions": c_reps, "v_repetitions": v_reps,
                "v_elements": int(v_elements),
                "nearest_cv_v_elements": reference.v_elements if reference else None,
                "samples": reference.samples if reference else 0,
            },
            "slowdowns": {"joint_scale": scale},
            "c_ms": completed["all_reduce"],
            "v_ms": v_duration,
            "span_ms": result["makespan_ms"],
            "per_repetition": {
                "c_ms": completed["all_reduce"] / max(1, c_reps),
                "v_ms": v_duration / max(1, v_reps),
                "span_ms": result["makespan_ms"] / max(1, c_reps),
            },
            "span_total_ms": result["makespan_ms"],
            "note": "Measured isolated lane curves and paired span; actual offset scheduled by ResourceEngine.",
        })


__all__ = ["CvCondition", "CvCalibration"]
