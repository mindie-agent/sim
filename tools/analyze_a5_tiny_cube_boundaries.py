#!/usr/bin/env python3
"""Audit retained useful-work A5 Cube captures without importing their producer.

[Role] Preserve physical kernel identity and distinguish inclusive invocation
duration from same-stream cadence, empty kernels, and Host submission.
[Invariants] Integer nanosecond subtraction; no residual fitting or inferred
HBM/L2 byte counts. Minima outside the observed domain are declared assumptions.
[Timeouts & Locks] Local bounded CSV reads, 50 rows/window, streaming hashes.
[Side-Effect Policy] Read local captures; write only the requested audit report.
"""
from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path

from tools.analyze_a5_microbench_boundaries import (
    analyze_window, file_identity, prior_measurements,
)


def review(capture: Path, repository: Path) -> dict:
    summary_path = capture / "benchmark_summary.json"
    original = json.loads(summary_path.read_text())
    rows = []
    for case in original["cases"]:
        folder = capture / case["case_id"]
        report = analyze_window(folder / "kernel_details.csv",
                                original["protocol"]["profiled_iters"])
        with (folder / "op_statistic.csv").open(newline="") as handle:
            statistics = list(csv.DictReader(handle))
        total_ns = sum(Decimal(row["Total Time(us)"]) * 1000 for row in statistics)
        report.update({
            "case_id": case["case_id"], "shape": case["shape"],
            "family": case["family"], "logical_weight_bytes": case["weight_matrix_bytes"],
            "hbm_or_l2_traffic_measured": False,
            "op_statistic": {"source": file_identity(folder / "op_statistic.csv"),
                             "rows": statistics,
                             "count_matches_kernel_rows": sum(int(r["Count"]) for r in statistics)
                             == report["physical_rows"],
                             "sum_matches_kernel_durations": total_ns
                             == Decimal(str(report["duration_sum_us"])) * 1000},
            "producer_float_parser": {key: case[key] for key in (
                "start_interval_us", "inter_kernel_gap_raw_us", "overlapping_pairs",
                "device_busy_fraction")},
        })
        rows.append(report)
    return {
        "schema": 1, "status": "independently_reviewed_local_capture", "split": "unassigned",
        "method": "Decimal us to integer ns, then same-stream subtraction; no device execution",
        "sources": [file_identity(summary_path),
                    file_identity(capture / "cfg/t6a_kernel_floor_tiny_cube.py"),
                    file_identity(capture / "report.md")],
        "analyzer": file_identity(Path(__file__).resolve()),
        "reported_environment": {k: original[k] for k in
                                 ("host", "device", "versions", "finished_utc")},
        "protocol": {**original["protocol"], "activities": ["NPU"],
                     "host_submit_timer": False, "event_timer": False,
                     "execution": "single-process eager Python, same operands, fresh returned output per call",
                     "numeric_reference_check": False,
                     "quant_probe": "Selected w8a8_pertoken; CPU copy verifies completion, not numeric correctness",
                     "warmups_in_csv": False, "discarded_profiled_rows": 0,
                     "cache_policy": "20 warmups and identical operands; cache residency/hit rate unmeasured"},
        "summary": {
            "conditions": len(rows), "physical_kernel_rows": sum(r["physical_rows"] for r in rows),
            "same_stream_pairs": sum(s["start_cadence"]["n"] for r in rows for s in r["streams"]),
            "exact_negative_gap_pairs": sum(s["negative_gap_pairs"] for r in rows for s in r["streams"]),
            "legacy_float_negative_gap_pairs": sum(s["legacy_float_negative_gap_pairs"] for r in rows for s in r["streams"]),
            "failures_reported_by_producer": original["failures"],
        },
        "cases": rows, "prior_independent_measurements": prior_measurements(repository),
        "parameter_recommendations": {
            "ordinary_nonempty_cube": {
                "minimum_us": 1.578, "startup_us": 0,
                "basis": "engineering_extrapolation_from_smallest_observed_BF16_library_MatMulV3",
                "observed_scope": "Ascend950DT_9582, M1 K64 N64, ND BF16 inputs/output, 1 AIC, warm eager calls",
                "not_a_universal_hardware_lower_bound": True,
                "explicit_empty_or_custom_program_classes_keep_precedence": True,
            },
            "w8a8_per_token_mixed_bf16_output": {
                "minimum_us": 3.037, "startup_us": 0,
                "basis": "engineering_extrapolation_from_smallest_observed_QuantBatchMatmulV3_call",
                "observed_scope": "Ascend950DT_9582, M1 K128 N128, ND INT8 x INT8, FP32 channel/token scales, BF16 output, 8 AIC+16 AIV",
                "exclude": ["INT32 output", "weight-only", "W4", "MXFP8", "grouped GEMM", "NZ layout", "other SKU"],
                "selection": "explicit implementation/quantization/output contract, never all quantized matrices",
            },
            "composition": "max(program duration, selected minimum) once per physical invocation; startup zero",
            "mixed_kernel": "One MIX_AIC row includes AIC/AIV overlap and epilogue; never sum AIC and AIV times or two minima",
            "shape_specific_controls": "P50 values may define explicit measured program classes, not generic startup",
            "host_submission": "Unidentified here; retain separate direct-ACL evidence and explicit host overrides",
            "larger_shapes": "5.322/8.379us INT8 minima include changing grid and useful work; not launch intercepts",
            "runtime_changes": False, "body_residual_fit": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    args.output.write_text(json.dumps(review(args.capture, args.repository), indent=2) + "\n")


if __name__ == "__main__":
    main()
