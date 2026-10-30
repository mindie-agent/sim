#!/usr/bin/env python3
"""Review the retained A5 eager microbench without launching its producer.

[Role] Reconcile device task durations and same-stream cadence from local CSVs.
[Invariants] Subtract integer nanosecond stamps before float conversion; retain
all profiled calls; never identify host-only time from a device timeline.
[Timeouts & Locks] No subprocess/network/locks. CSV working set is bounded by
the producer's declared calls per window; hashes stream in 64 KiB chunks.
[Side-Effect Policy] Read captures and write only the explicitly requested report.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from decimal import Decimal
from pathlib import Path


IDENTITY_FIELDS = ("Device_id", "Model ID", "Stream ID", "Task ID")
DESCRIPTORS = ("Name", "Type", "Accelerator Core", "Block Num", "Mix Block Num",
               "Input Shapes", "Input Data Types", "Input Formats", "Output Shapes",
               "Output Data Types", "Output Formats")
PIPE_FIELDS = ("aicore_time(us)", "aic_mac_ratio", "aic_mte1_ratio", "aic_mte2_ratio",
               "aiv_time(us)", "aiv_vec_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio")


def nanoseconds(value: str) -> int:
    value_ns = Decimal(value.strip()) * 1000
    if value_ns != value_ns.to_integral_value():
        raise ValueError("Capture contains sub-nanosecond precision; do not truncate")
    return int(value_ns)


def stats_ns(values: list[int]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    def percentile(q: float) -> float:
        index = (len(ordered) - 1) * q
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return (ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)) / 1000
    return {"n": len(values), "min_us": ordered[0] / 1000,
            "p10_us": percentile(.1), "p50_us": percentile(.5),
            "mean_us": sum(values) / len(values) / 1000,
            "p90_us": percentile(.9), "max_us": ordered[-1] / 1000}


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def analyze_window(path: Path, expected_calls: int) -> dict:
    streams: dict[tuple[str, str], list[tuple[int, int]]] = {}
    identities = set()
    descriptors = {key: set() for key in DESCRIPTORS}
    pipes = {key: [] for key in PIPE_FIELDS}
    waits = []
    with path.open(newline="") as handle:
        for count, row in enumerate(csv.DictReader(handle), 1):
            if count > expected_calls:
                raise ValueError(f"{path}: more kernel rows than declared one-kernel calls")
            start = nanoseconds(row["Start Time(us)"])
            duration = nanoseconds(row["Duration(us)"])
            identity = tuple(row[key].strip() for key in IDENTITY_FIELDS) + (start,)
            if identity in identities:
                raise ValueError(f"{path}: duplicate physical kernel identity {identity}")
            identities.add(identity)
            stream = (row["Device_id"].strip(), row["Stream ID"].strip())
            streams.setdefault(stream, []).append((start, duration))
            for key in descriptors:
                descriptors[key].add(row[key].strip())
            for key in pipes:
                value = row[key].strip()
                if value not in {"", "N/A"}:
                    pipes[key].append(float(value))
            waits.append(nanoseconds(row["Wait Time(us)"]))
    if len(identities) != expected_calls:
        raise ValueError(f"{path}: expected {expected_calls} kernels, found {len(identities)}")
    durations, stream_reports = [], []
    for (device, stream), events in sorted(streams.items()):
        events.sort()
        durations.extend(duration for _, duration in events)
        intervals = [b[0] - a[0] for a, b in zip(events, events[1:])]
        gaps = [interval - events[i][1] for i, interval in enumerate(intervals)]
        # Reproduce only the old parser's precision error for explicit comparison.
        float_gaps = [(float(Decimal(b[0]) / 1000) - float(Decimal(a[0]) / 1000))
                      - a[1] / 1000 for a, b in zip(events, events[1:])]
        span = events[-1][0] + events[-1][1] - events[0][0]
        stream_reports.append({
            "device": device, "stream": stream, "rows": len(events),
            "first_start_ns": str(events[0][0]), "last_start_ns": str(events[-1][0]),
            "duration_sum_us": sum(duration for _, duration in events) / 1000,
            "first_start_to_last_end_us": span / 1000,
            "duration_sum_over_window_span": sum(duration for _, duration in events) / span,
            "start_cadence": stats_ns(intervals), "signed_end_to_start_gap": stats_ns(gaps),
            "negative_gap_pairs": sum(value < 0 for value in gaps),
            "legacy_float_negative_gap_pairs": sum(value < 0 for value in float_gaps),
            "first_transition_gap_us": gaps[0] / 1000 if gaps else None,
            "after_first_transition_gap_diagnostic": stats_ns(gaps[1:]),
        })
    return {"raw": file_identity(path), "physical_rows": len(identities),
            "kernel_descriptors": {k: sorted(v) for k, v in descriptors.items()},
            "device_kernel_duration": stats_ns(durations),
            "duration_sum_us": sum(durations) / 1000,
            "exported_wait_field": stats_ns(waits), "streams": stream_reports,
            "pipe_metrics_mean": {k: statistics.fmean(v) for k, v in pipes.items() if v}}


def prior_measurements(root: Path) -> dict:
    empty_path = root / "tests/fixtures/verification/a5-empty-programs.json"
    tiny_path = root / "tests/fixtures/verification/a5-tiny-programs.json"
    empty = json.loads(empty_path.read_text())
    tiny = json.loads(tiny_path.read_text())
    return {
        "empty_source": file_identity(empty_path), "tiny_source": file_identity(tiny_path),
        "empty_programs": [{"kernel": row["kernel"], "blocks": row["blocks"],
                            "steady_device": row["steady_launches"]}
                           for row in empty["independent_review"]["conditions"]],
        "tiny_aiv_programs": [
            {"shape": row["shape"], "blocks": row["blocks"],
             "steady_device": row["steady_launches"],
             "profiled_direct_acl_host_submit_us": row["host_submit_us_per_launch"],
             "profiled_event_span_per_call_us": row["event_batch_span_divided_by_50_us"]}
            for row in tiny["independent_review"]["conditions"]],
        "tiny_aic_status": tiny["cube"]["status"],
    }


def review(capture: Path, repository: Path) -> dict:
    summary_path = capture / "benchmark_summary.json"
    original = json.loads(summary_path.read_text())
    rows = []
    for case in original["cases"]:
        row = analyze_window(capture / case["case_id"] / "kernel_details.csv",
                             original["protocol"]["profiled_iters"])
        row.update({"case_id": case["case_id"], "family": case["family"],
                    "op": case["op"], "tokens": case["tokens"]})
        row["original_float_parser"] = {key: case[key] for key in (
            "start_interval_us", "inter_kernel_gap_raw_us", "overlapping_pairs",
            "device_busy_fraction")}
        if case["family"] == "cube":
            n = int(case["op"].rsplit("n", 1)[1])
            row["logical_operand_work"] = {
                "M": case["tokens"], "K": 4096, "N": n,
                "weight_bytes_bf16": 4096 * n * 2,
                "hbm_traffic_measured": False,
                "note": "Same B reused for warmup and every call; cache residency unknown"}
        rows.append(row)
    exact_negative = sum(s["negative_gap_pairs"] for row in rows for s in row["streams"])
    legacy_negative = sum(s["legacy_float_negative_gap_pairs"] for row in rows for s in row["streams"])
    return {
        "schema": 1, "status": "independently_reviewed_local_capture", "split": "unassigned",
        "method": "Decimal us to integer ns; subtract before float conversion; stream-local adjacency",
        "sources": [file_identity(summary_path),
                    file_identity(capture / "cfg/t1a_kernel_floor.py"),
                    file_identity(capture / "report.md"),
                    file_identity(capture / "logs/t1a_kernel_floor.run.log")],
        "analyzer": file_identity(Path(__file__).resolve()),
        "reported_environment": {key: original[key] for key in
                                 ("host", "device", "versions", "finished_utc")},
        "protocol": {**original["protocol"],
                     "activities": ["NPU"], "host_submit_timer": False,
                     "execution": "single process eager Python loop; fresh output each call",
                     "concurrency_note": "tokens are rows, not serving concurrency; no graph replay",
                     "warmups_present_in_csv": False, "all_profiled_rows_retained": True},
        "summary": {"conditions": len(rows), "physical_kernel_rows": sum(r["physical_rows"] for r in rows),
                    "same_stream_pairs": sum(s["start_cadence"]["n"] for r in rows for s in r["streams"]),
                    "exact_negative_gap_pairs": exact_negative,
                    "legacy_float_negative_gap_pairs": legacy_negative,
                    "failures_reported_by_producer": original["failures"]},
        "cases": rows, "prior_independent_measurements": prior_measurements(repository),
        "parameter_conclusions": {
            "device_invocation": "No universal A5 floor identified; retained exact program classes remain valid",
            "new_useful_programs": "Complete-kernel inclusive durations, not additive startup measurements",
            "host_submission": "Not identified by this capture; prior direct ACL submit timer is a different path",
            "matmul": "Nonempty operand-transfer work; neither empty Cube floor nor measured HBM byte volume",
            "gap": "Same-stream timeline spacing, not unique attribution to host/dispatch/device idle",
            "calibration_or_validation": "No fitted parameters or accuracy admission; observations unassigned",
            "runtime_changes": False},
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
