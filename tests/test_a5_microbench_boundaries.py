"""Exact arithmetic against retained real A5 profiler rows, without device access."""
from pathlib import Path

import pytest

from tools.analyze_a5_microbench_boundaries import analyze_window, nanoseconds


FIXTURES = Path(__file__).parent / "fixtures/a5_microbench_20261030"


def test_real_matmul_rows_do_not_gain_negative_gaps_from_epoch_float_rounding():
    report = analyze_window(FIXTURES / "cube_matmul_m1_k4096_n4096.csv", 4)
    stream = report["streams"][0]
    assert nanoseconds("1793294469786372.021\t") == 1793294469786372021
    assert stream["negative_gap_pairs"] == 0
    assert stream["legacy_float_negative_gap_pairs"] == 1
    assert stream["signed_end_to_start_gap"]["p50_us"] == .022
    assert stream["first_transition_gap_us"] == 84.448
    assert stream["duration_sum_us"] == 50.254
    assert stream["first_start_to_last_end_us"] == 134.746
    assert stream["duration_sum_over_window_span"] == pytest.approx(50.254 / 134.746)


def test_real_vector_rows_retain_first_transition_and_device_work_separately():
    report = analyze_window(FIXTURES / "vector_add_t1.csv", 4)
    stream = report["streams"][0]
    assert report["physical_rows"] == 4
    assert report["device_kernel_duration"]["min_us"] == 1.326
    assert stream["first_transition_gap_us"] == 85.347
    assert stream["signed_end_to_start_gap"]["n"] == 3
    assert stream["after_first_transition_gap_diagnostic"]["n"] == 2
    assert stream["after_first_transition_gap_diagnostic"]["mean_us"] == (4.542 + 3.337) / 2
    assert stream["duration_sum_us"] == pytest.approx(2.134 + 1.523 + 1.477 + 1.326)
    assert report["kernel_descriptors"]["Block Num"] == ["8"]
    assert report["kernel_descriptors"]["Accelerator Core"] == ["AI_VECTOR_CORE"]
