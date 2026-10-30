"""[Role] Recompute the retained empty-kernel probe from device profiler rows.
[Invariants] Device duration, core counters and event batches stay separate.
  Fifty warmup launches are excluded independently for each kernel/block pair.
  No observed duration becomes a simulator default or whole-model correction.
[Timeouts & Locks] Local files only; 3,300 scalar samples; no locks or waits.
[Side-Effect Policy] Prints JSON; never initializes a device or edits raw data.
"""

import argparse
import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def distribution(values):
    ordered = sorted(values)

    def quantile(fraction):
        index = fraction * (len(ordered) - 1)
        lower = int(index)
        remainder = index - lower
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] * (1 - remainder) + ordered[upper] * remainder

    return {
        "n": len(ordered),
        "min_us": min(ordered),
        "p10_us": quantile(0.1),
        "p50_us": median(ordered),
        "p90_us": quantile(0.9),
        "p99_us": quantile(0.99),
        "max_us": max(ordered),
        "mean_us": mean(ordered),
    }


def analyze(path):
    groups = {}
    identities = set()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            identity = tuple(row[key] for key in
                             ("Device_id", "Stream ID", "Task ID", "Task Start Time(us)"))
            if identity in identities:
                raise ValueError("duplicate physical profiler row")
            identities.add(identity)
            key = row["Op Name"], int(row["Block Num"])
            groups.setdefault(key, []).append(row)
    if len(identities) != 3300 or len(groups) != 6:
        raise ValueError("capture does not match the recorded six-condition protocol")
    conditions = []
    expected = {
        "empty_aiv": ("AI_VECTOR_CORE", "aiv_total_cycles", "aic_total_cycles", "aiv_time(us)"),
        "empty_aic": ("AI_CORE", "aic_total_cycles", "aiv_total_cycles", "aicore_time(us)"),
    }
    for (kernel, blocks), rows in sorted(groups.items()):
        task_type, active_cycles, inactive_cycles, core_time = expected[kernel]
        if blocks not in (1, 8, 64) or len(rows) != 550:
            raise ValueError("missing launch rows")
        if any(row["Task Type"] != task_type for row in rows):
            raise ValueError("actual task type does not match the requested core")
        if any(float(row[active_cycles]) <= 0 or float(row[inactive_cycles]) != 0
               for row in rows):
            raise ValueError("core counters contradict an isolated AIV/AIC launch")
        rows.sort(key=lambda row: Decimal(row["Task Start Time(us)"].strip()))
        durations = [float(row["Task Duration(us)"]) for row in rows]
        steady = rows[50:]
        conditions.append({
            "kernel": kernel,
            "blocks": blocks,
            "task_type": task_type,
            "all_launches": distribution(durations),
            "warmup_launches": 50,
            "steady_launches": distribution(durations[50:]),
            "steady_batch_medians_us": [
                median(durations[start:start + 100]) for start in range(50, 550, 100)
            ],
            "steady_core_time": distribution([float(row[core_time]) for row in steady]),
            "positive_active_core_cycles": len(rows),
            "zero_other_core_cycles": len(rows),
        })
    with path.open("rb") as stream:
        source_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "boundary": "profiler_device_task_duration",
        "path": str(path),
        "sha256": source_hash,
        "unique_physical_rows": len(identities),
        "split": "unassigned",
        "warmup_rule": "first 50 chronological launches per kernel/block condition",
        "conditions": conditions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.capture), indent=2))
