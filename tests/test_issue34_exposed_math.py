"""Four-clock max-plus exposed-time arithmetic and conservation oracles."""
from __future__ import annotations

import math

import pytest

from mindie_sim.prediction.exposed_breakdown import (
    ATTRIBUTION_PERCENT_TOLERANCE, ExposedClocks, advance_clocks,
    communication_clocks, control_clocks, cube_clocks, finish_breakdown,
    join_clocks, memory_clocks, roofline_clocks, unavailable_breakdown,
    vector_clocks,
)
from mindie_sim.schema.common import ModelingError


def breakdown(clocks, duration, **kwargs):
    return finish_breakdown(clocks, duration_ms=duration, boundary="device_step",
                            **kwargs)


def test_section_table_examples():
    cases = [
        (roofline_clocks(cube_ms=10, vector_ms=0, memory_ms=4, communication_ms=0),
         10, (10, 0, 0, 0)),
        (roofline_clocks(cube_ms=4, vector_ms=0, memory_ms=10, communication_ms=0),
         10, (4, 0, 6, 0)),
        (advance_clocks((cube_clocks(4),), vector_clocks(3)), 7, (4, 3, 0, 0)),
        (join_clocks((cube_clocks(4), vector_clocks(7))), 7, (4, 3, 0, 0)),
        (join_clocks((communication_clocks(3),
                      roofline_clocks(cube_ms=5, vector_ms=0, memory_ms=7,
                                      communication_ms=0))),
         7, (5, 0, 2, 0)),
        (join_clocks((communication_clocks(10),
                      roofline_clocks(cube_ms=5, vector_ms=0, memory_ms=7,
                                      communication_ms=0))),
         10, (5, 0, 2, 3)),
        (advance_clocks((communication_clocks(3),),
                        roofline_clocks(cube_ms=5, vector_ms=0, memory_ms=7,
                                        communication_ms=0)),
         10, (5, 0, 2, 3)),
        (advance_clocks((control_clocks(2),),
                        roofline_clocks(cube_ms=5, vector_ms=0, memory_ms=3,
                                        communication_ms=0)),
         7, (5, 2, 0, 0)),
    ]
    for clocks, duration, expected in cases:
        result = breakdown(clocks, duration)
        values = result["exposed_breakdown_ms"]
        assert tuple(values[name] for name in
                     ("cube", "vector", "memory", "communication")) == pytest.approx(expected)


def test_componentwise_join_wins_after_communication_elimination():
    left = ExposedClocks(all_ms=10, without_communication_ms=8, compute_ms=8,
                         cube_ms=2)
    right = ExposedClocks(all_ms=9, without_communication_ms=9, compute_ms=4,
                          cube_ms=4)
    joined = join_clocks((left, right))
    assert joined == ExposedClocks(10, 9, 8, 4)


def test_positive_breakdown_conserves_milliseconds_and_percentages():
    clocks = roofline_clocks(cube_ms=3.0, vector_ms=5.0, memory_ms=7.0,
                             communication_ms=11.0)
    result = breakdown(clocks, 11.0, control_ms=2.0, opaque_control_ms=1.0)
    values = result["exposed_breakdown_ms"]
    percentages = result["exposed_breakdown_pct"]
    assert math.fsum(values.values()) == pytest.approx(11.0)
    assert math.fsum(percentages.values()) == pytest.approx(100.0,
                                                           abs=ATTRIBUTION_PERCENT_TOLERANCE)
    assert all(value >= 0 for value in values.values())
    assert result["exposed_breakdown_basis"]["vector_control_ms"] == 2.0
    assert result["exposed_breakdown_basis"]["opaque_control_ms"] == 1.0
    assert result["exposed_breakdown_basis"]["counterfactual_speedup"] is False


def test_zero_time_and_operator_only_results_are_explicit():
    zero = breakdown(ExposedClocks(0, 0, 0, 0), 0)
    assert set(zero["exposed_breakdown_ms"].values()) == {0}
    assert zero["exposed_breakdown_basis"]["available"] is True
    unavailable = unavailable_breakdown("operator_only", boundary="operator_sum")
    assert unavailable["exposed_breakdown_ms"] is None
    assert unavailable["exposed_breakdown_basis"]["available"] is False
    assert unavailable["exposed_breakdown_basis"]["reason"] == "operator_only"


def test_boundary_and_order_mismatches_are_errors_not_scaled():
    with pytest.raises(ModelingError, match="clock_mismatch"):
        breakdown(cube_clocks(5), 6)
    with pytest.raises(ModelingError, match="clock_order_invalid"):
        ExposedClocks(1, 2, 0, 0)
    with pytest.raises(ModelingError, match="clock_invalid"):
        ExposedClocks(-1, 0, 0, 0)
    clocks = roofline_clocks(cube_ms=2, vector_ms=0, memory_ms=3,
                             communication_ms=0)
    with pytest.raises(ModelingError, match="clock_mismatch"):
        breakdown(clocks, 4)


def test_percent_residual_is_assigned_deterministically():
    clocks = ExposedClocks(3.0, 3.0, 3.0, 3.0)
    first = breakdown(clocks, 3.0)
    second = breakdown(clocks, 3.0)
    assert first == second
    assert first["exposed_breakdown_pct"]["cube"] == pytest.approx(100.0)
