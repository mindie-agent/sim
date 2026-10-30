"""Independent numerical controls for finite serialization and credit return."""
from itertools import product

import pytest

from tools.finite_transfer_oracle import finite_transfer, homogeneous_duration


def test_credit_window_has_fill_tail_and_response_not_just_a_rate_cap():
    # Service 1ms, response 4ms, two credits: response times 5, 6, 10, 11, 15.
    result = finite_transfer(50, 10, 10, 2, 4)
    assert result["duration_ms"] == 15
    assert result["credit_stall_ms"] == 6
    assert result["serialization_ms"] == 5
    assert finite_transfer(10, 10, 10, 2, 4)["duration_ms"] == 5
    # Five credits cover the 1ms service +4ms response round trip.
    assert finite_transfer(50, 10, 10, 5, 4)["duration_ms"] == 9


def test_partial_final_transaction_has_its_own_byte_boundary():
    rounded = finite_transfer(41, 10, 10, 2, 4)
    exact = finite_transfer(41, 10, 10, 2, 4, rounded_tail=False)
    assert rounded["transactions"] == exact["transactions"] == 5
    assert rounded["charged_bytes"] == 50
    assert exact["charged_bytes"] == 41
    assert rounded["duration_ms"] == 15
    assert exact["duration_ms"] == pytest.approx(14.1)
    assert exact["credit_stall_ms"] == pytest.approx(6)


def test_recurrence_matches_closed_form_across_credit_and_bandwidth_transitions():
    for count, service, credits, response in product(
        (0, 1, 2, 5, 17), (.125, 1, 8), (1, 2, 5), (0, .5, 4, 20),
    ):
        result = finite_transfer(count * 16, 16, 16 / service, credits, response)
        expected = homogeneous_duration(count, service, credits, response)
        assert result["duration_ms"] == pytest.approx(expected)
        assert result["charged_bytes"] == count * 16
