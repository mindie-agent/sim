"""Real useful-work Cube rows preserve device/cadence and mixed-core boundaries."""
from pathlib import Path

import pytest

from tools.analyze_a5_microbench_boundaries import analyze_window


FIXTURES = Path(__file__).parent / "fixtures/a5_tiny_cube_20261030"


def test_bf16_device_minimum_excludes_the_large_first_submission_gap():
    report = analyze_window(FIXTURES / "bf16_matmulv3_m1_k64_n64.csv", 4)
    stream = report["streams"][0]
    assert report["device_kernel_duration"]["min_us"] == 1.691
    assert stream["first_transition_gap_us"] == 69.581
    assert stream["duration_sum_us"] == 8.091
    assert stream["first_start_to_last_end_us"] == 92.107
    assert report["kernel_descriptors"]["Block Num"] == ["1"]
    assert report["kernel_descriptors"]["Mix Block Num"] == ["0"]


def test_mixed_quantized_kernel_is_one_task_and_exact_gap_remains_positive():
    report = analyze_window(FIXTURES / "int8_quantbatchmatmulv3_m1_k512_n512.csv", 3)
    stream = report["streams"][0]
    assert report["physical_rows"] == 3
    assert stream["negative_gap_pairs"] == 0
    assert stream["legacy_float_negative_gap_pairs"] == 1
    assert stream["first_transition_gap_us"] == .022
    assert report["duration_sum_us"] == pytest.approx(5.795 + 6.679 + 6.151)
    descriptors = report["kernel_descriptors"]
    assert descriptors["Accelerator Core"] == ["MIX_AIC"]
    assert descriptors["Block Num"] == ["32"]
    assert descriptors["Mix Block Num"] == ["64"]
    assert descriptors["Input Data Types"] == ["INT8;INT8;FLOAT;FLOAT"]
    assert descriptors["Output Data Types"] == ["DT_BF16"]
