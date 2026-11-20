"""Independent analytical recurrences and float64 ULP helpers for Engine tests.

[Role] Supply closed-form lower bounds, ring/pipeline recurrences, and exact
  IEEE-754 ULP comparisons derived from mathematical definitions rather than
  ``mindie_sim`` internals, so runtime/collective/operator tests check physics
  against an independent oracle.
[Invariants] Pure stdlib math; never imports ``mindie_sim`` execution modules.
[Timeouts & Locks] Pure in-memory arithmetic; no I/O, sleeps, or locks.
[Side-Effect Policy] Side-effect-free.
"""
from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Mapping


def _ordered_f64_bits(value: float) -> int:
    raw = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    if raw & (1 << 63):
        return (1 << 64) - raw
    return raw | (1 << 63)


def ulp_distance(lhs: float, rhs: float) -> int:
    """IEEE-754 binary64 ULP distance (``+0.0`` and ``-0.0`` compare as 0 ULP)."""
    if math.isnan(lhs) or math.isnan(rhs):
        raise ValueError(f"cannot compute ULP distance on NaN: {lhs!r}, {rhs!r}")
    if lhs == rhs:
        return 0
    if math.isinf(lhs) or math.isinf(rhs):
        return 1 << 62
    return abs(_ordered_f64_bits(lhs) - _ordered_f64_bits(rhs))


def roofline_lower_bound(demands: Mapping[str, float], rates: Mapping[str, float]) -> float:
    """Max single-resource service time ``max_r(work_r / rate_r)`` over positive demands."""
    lower = 0.0
    for resource, work in demands.items():
        if work <= 0.0:
            continue
        rate = rates[resource]
        if rate <= 0.0:
            raise ValueError(f"non-positive rate {rate!r} for positive demand on {resource!r}")
        lower = max(lower, float(work) / float(rate))
    return lower


def steady_two_stage_makespan(tiles: int, stage_a: float, stage_b: float) -> float:
    """Exact makespan of an uncontended two-stage double-buffered pipeline over ``tiles``."""
    count = int(tiles)
    if count <= 0:
        return 0.0
    first, second = float(stage_a), float(stage_b)
    return first + (count - 1) * max(first, second) + second


def ring_collective_volume_per_rank(payload_bytes: int, group_size: int, *, two_phase: bool) -> float:
    """Per-rank wire bytes for ring ReduceScatter/AllGather (1x) or AllReduce (2x)."""
    ranks = int(group_size)
    if ranks <= 1 or payload_bytes <= 0:
        return 0.0
    factor = 2.0 if two_phase else 1.0
    return factor * float(payload_bytes) * (ranks - 1) / ranks


def serial_sum(durations: Iterable[float]) -> float:
    """High-precision sum ``math.fsum`` of sequential non-overlapping intervals."""
    return math.fsum(float(d) for d in durations)
