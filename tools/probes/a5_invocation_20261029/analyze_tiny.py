"""[Role] Independently analyze the retained tiny AIV control.
[Invariants] Fixed producer order binds shapes; warmup is removed per shape.
  Profiler, event intervals, host submission and numerical checks retain run IDs.
[Timeouts & Locks] Local scalar CSV/log parsing only; no locks or subprocesses.
[Side-Effect Policy] Prints JSON; does not launch kernels or change calibration.
"""

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from analyze import distribution


def analyze(capture, profiler_log):
    with capture.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    with profiler_log.open() as stream:
        records = [json.loads(line) for line in stream if line.startswith("{")]
    precheck = next(row for row in records if row["kind"] == "precheck")
    batches = [row for row in records if row["kind"] == "device_event_batch"]
    numeric = [row for row in records if row["kind"] == "numeric_check"]
    if ([row["shape"] for row in batches] != [[64], [256], [1024]]
            or [row["shape"] for row in numeric] != [[64], [256], [1024]]
            or any(not row["passed"] for row in numeric)):
        raise ValueError("producer order or numeric check does not match")
    if len(rows) != 510:
        raise ValueError("profile does not contain all expected launches")
    identities = {
        tuple(row[key] for key in
              ("Device_id", "Stream ID", "Task ID", "Task Start Time(us)"))
        for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError("duplicate physical rows")
    for row in rows:
        if (row["Op Name"] != "tiny_aiv_add" or row["Task Type"] != "AI_VECTOR_CORE"
                or row["Block Num"] != "1" or float(row["aiv_total_cycles"]) <= 0
                or float(row["aic_total_cycles"]) != 0):
            raise ValueError("device counters contradict the AIV-only control")
    rows.sort(key=lambda row: Decimal(row["Task Start Time(us)"].strip()))
    conditions = []
    for index, batch in enumerate(batches):
        if batch["warmup"] != 20 or batch["repetitions"] != 50:
            raise ValueError("producer repetition protocol changed")
        subset = rows[index * 170:(index + 1) * 170]
        durations = [float(row["Task Duration(us)"]) for row in subset]
        conditions.append({
            "shape": batch["shape"],
            "blocks": 1,
            "task_type": "AI_VECTOR_CORE",
            "all_launches": distribution(durations),
            "warmup_launches": 20,
            "steady_launches": distribution(durations[20:]),
            "steady_core_time": distribution([
                float(row["aiv_time(us)"]) for row in subset[20:]
            ]),
            "event_batch_span_divided_by_50_us": batch["batch_us"],
            "host_submit_us_per_launch": batch["host_submit_us_per_launch"],
            "numeric_check": numeric[index],
        })
    with capture.open("rb") as stream:
        capture_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    with profiler_log.open("rb") as stream:
        log_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "run_id": precheck["run_id"],
        "split": "unassigned",
        "profiler": {"path": str(capture), "sha256": capture_sha},
        "producer": {"path": str(profiler_log), "sha256": log_sha},
        "shape_binding": "fixed producer order; exactly 170 chronological launches per shape",
        "physical_unique_rows": len(identities),
        "conditions": conditions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("profiler_log", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.capture, args.profiler_log), indent=2))
