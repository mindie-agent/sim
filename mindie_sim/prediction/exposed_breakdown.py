"""Pure four-clock max-plus exposed-time attribution.

[Role]
  Represent the same realized causal graph under four ordered service-class
  eliminations and reduce terminal clocks into a conserving four-bucket
  breakdown: computation first, then memory residual, then communication
  residual.  This is a schedule attribution, not a counterfactual speedup.

[Invariants]
  * A clock is ``(all, without_communication, compute, cube)``; joins are
    componentwise maxima and advances add one node's four nonnegative times.
  * ``all >= without_communication >= compute >= cube >= 0`` for every clock.
  * Terminal buckets are ``cube=D3, vector=D2-D3, memory=D1-D2,
    communication=D0-D1`` and telescope to the executed boundary.
  * Percentages are computed once with ``math.fsum``; only the final residual is
    assigned to one deterministic bucket.  Zero-time results are all zeros and
    available, never NaN.

[Timeouts & Locks]
  Pure arithmetic; no I/O or locks.

[Side-Effect Policy]
  Returns plain dictionaries; no global state.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError


ZERO: float = 0.0
FULL_PERCENT: float = 100.0
ATTRIBUTION_PERCENT_TOLERANCE: float = 1e-9
CLOCK_ULP_MULTIPLIER: int = 32
CLOCK_ABS_TOLERANCE_MS: float = 1e-12
METHOD = "ordered_critical_span_v1"
OVERLAP_PRIORITY = ("cube", "vector", "memory", "communication")
BUCKETS = OVERLAP_PRIORITY


@dataclass(frozen=True)
class ExposedClocks:
    all_ms: float
    without_communication_ms: float
    compute_ms: float
    cube_ms: float

    def __post_init__(self) -> None:
        values = (self.all_ms, self.without_communication_ms,
                  self.compute_ms, self.cube_ms)
        if any(not math.isfinite(value) or value < ZERO for value in values):
            raise ModelingError(f"exposed_clock_invalid:{values}")
        if not (self.all_ms >= self.without_communication_ms
                >= self.compute_ms >= self.cube_ms):
            raise ModelingError(f"exposed_clock_order_invalid:{values}")


def join_clocks(predecessors: Sequence[ExposedClocks]) -> ExposedClocks:
    """Componentwise maximum of every predecessor clock."""
    if not predecessors:
        return ExposedClocks(ZERO, ZERO, ZERO, ZERO)
    return ExposedClocks(
        max(item.all_ms for item in predecessors),
        max(item.without_communication_ms for item in predecessors),
        max(item.compute_ms for item in predecessors),
        max(item.cube_ms for item in predecessors),
    )


def advance_clocks(
    predecessors: Sequence[ExposedClocks], duration: ExposedClocks,
) -> ExposedClocks:
    """Series composition: join predecessor clocks, then add this node."""
    joined = join_clocks(predecessors)
    return ExposedClocks(
        joined.all_ms + duration.all_ms,
        joined.without_communication_ms + duration.without_communication_ms,
        joined.compute_ms + duration.compute_ms,
        joined.cube_ms + duration.cube_ms,
    )


def roofline_clocks(
    *, cube_ms: float, vector_ms: float, memory_ms: float,
    communication_ms: float,
) -> ExposedClocks:
    """Independent concurrent Roofline service node with ordered elimination."""
    values = (cube_ms, vector_ms, memory_ms, communication_ms)
    if any(not math.isfinite(float(value)) or float(value) < ZERO for value in values):
        raise ModelingError(f"roofline_clock_invalid:{values}")
    cube, vector, memory, communication = (float(value) for value in values)
    return ExposedClocks(
        max(cube, vector, memory, communication),
        max(cube, vector, memory),
        max(cube, vector),
        cube,
    )


def cube_clocks(duration_ms: float) -> ExposedClocks:
    return ExposedClocks(duration_ms, duration_ms, duration_ms, duration_ms)


def vector_clocks(duration_ms: float) -> ExposedClocks:
    return ExposedClocks(duration_ms, duration_ms, duration_ms, ZERO)


def memory_clocks(duration_ms: float) -> ExposedClocks:
    return ExposedClocks(duration_ms, duration_ms, ZERO, ZERO)


def communication_clocks(duration_ms: float) -> ExposedClocks:
    return ExposedClocks(duration_ms, ZERO, ZERO, ZERO)


def control_clocks(duration_ms: float) -> ExposedClocks:
    """Host launch/control is a vector/control service node."""
    return vector_clocks(duration_ms)


def clock_reconciliation_tolerance(duration_ms: float) -> float:
    if duration_ms <= ZERO:
        return CLOCK_ABS_TOLERANCE_MS
    return max(CLOCK_ULP_MULTIPLIER * math.ulp(float(duration_ms)),
               CLOCK_ABS_TOLERANCE_MS)


def _clip(value: float, tolerance: float) -> float:
    if value < ZERO:
        if value < -tolerance:
            raise ModelingError(f"exposed_attribution_negative:{value}")
        return ZERO
    return value


def _residual_bucket(values: Mapping[str, float]) -> str:
    for name in BUCKETS:
        if values[name] > ZERO:
            return name
    return BUCKETS[ZERO]


def finish_breakdown(
    clocks: ExposedClocks, *, duration_ms: float, boundary: str,
    control_ms: float = ZERO, opaque_control_ms: float = ZERO,
) -> dict[str, Any]:
    """Reduce terminal clocks to the public conserving breakdown."""
    duration = float(duration_ms)
    if not math.isfinite(duration) or duration < ZERO:
        raise ModelingError(f"exposed_attribution_duration_invalid:{duration_ms}")
    tolerance = clock_reconciliation_tolerance(duration)
    if duration <= ZERO:
        milliseconds = dict.fromkeys(BUCKETS, ZERO)
        return {
            "exposed_breakdown_ms": milliseconds,
            "exposed_breakdown_pct": dict.fromkeys(BUCKETS, ZERO),
            "exposed_breakdown_basis": {
                "method": METHOD,
                "available": True,
                "boundary": str(boundary),
                "duration_ms": ZERO,
                "overlap_priority": list(OVERLAP_PRIORITY),
                "vector_control_ms": ZERO,
                "opaque_control_ms": ZERO,
                "counterfactual_speedup": False,
            },
        }
    terminal = _clip(clocks.all_ms, tolerance)
    if abs(terminal - duration) > tolerance:
        raise ModelingError(
            f"exposed_attribution_clock_mismatch:{terminal}!={duration}")
    without_communication = _clip(clocks.without_communication_ms, tolerance)
    compute = _clip(clocks.compute_ms, tolerance)
    cube = _clip(clocks.cube_ms, tolerance)
    if not (terminal >= without_communication >= compute >= cube >= ZERO):
        raise ModelingError("exposed_attribution_clock_order_invalid")
    values = {
        "cube": cube,
        "vector": max(ZERO, compute - cube),
        "memory": max(ZERO, without_communication - compute),
        "communication": max(ZERO, terminal - without_communication),
    }
    total = math.fsum(values.values())
    if abs(total - duration) > tolerance:
        raise ModelingError(
            f"exposed_attribution_clock_mismatch:{total}!={duration}")
    vector = values["vector"]
    control = min(max(ZERO, float(control_ms)), vector)
    opaque = min(max(ZERO, float(opaque_control_ms)), vector)
    percentages = {
        name: values[name] / duration * FULL_PERCENT for name in BUCKETS
    }
    residual = FULL_PERCENT - math.fsum(percentages.values())
    percentages[_residual_bucket(values)] += residual
    if abs(math.fsum(percentages.values()) - FULL_PERCENT) > ATTRIBUTION_PERCENT_TOLERANCE:
        raise ModelingError("exposed_attribution_percentage_mismatch")
    return {
        "exposed_breakdown_ms": {name: float(values[name]) for name in BUCKETS},
        "exposed_breakdown_pct": {
            name: float(percentages[name]) for name in BUCKETS},
        "exposed_breakdown_basis": {
            "method": METHOD,
            "available": True,
            "boundary": str(boundary),
            "duration_ms": duration,
            "overlap_priority": list(OVERLAP_PRIORITY),
            "vector_control_ms": control,
            "opaque_control_ms": opaque,
            "counterfactual_speedup": False,
        },
    }


def unavailable_breakdown(reason: str, *, boundary: str) -> dict[str, Any]:
    """Operator-only report without an executed positive device-step boundary."""
    return {
        "exposed_breakdown_ms": None,
        "exposed_breakdown_pct": None,
        "exposed_breakdown_basis": {
            "method": METHOD,
            "available": False,
            "reason": str(reason),
            "boundary": str(boundary),
            "duration_ms": None,
            "overlap_priority": list(OVERLAP_PRIORITY),
            "vector_control_ms": None,
            "opaque_control_ms": None,
            "counterfactual_speedup": False,
        },
    }
