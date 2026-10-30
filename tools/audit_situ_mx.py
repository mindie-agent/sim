"""Stream retained SituMxQuant task and PMU diagnostics with the profiler parser.

[Role] Keep original task-duration and pipe-counter boundaries distinct.
[Invariants] No timing or counter becomes a simulation parameter; no event is
  divided into invented steps. Counter sums are not added to task duration.
[Timeouts & Locks] At most 500000 capture rows; parser deadline 45 seconds.
[Side-Effect Policy] Read an existing capture and write one compact report.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time


def audit(source):
    from mindie_profiler.normalized import kernel_rows

    started = time.monotonic()
    groups = {}
    for row in kernel_rows(source, timeout_s=45, max_rows=500_000):
        if row.kind != "SituMxQuant":
            continue
        key = str((row.input_shapes, row.output_shapes, row.input_dtypes,
                   row.output_dtypes, row.input_formats, row.output_formats))
        group = groups.setdefault(key, {
            "events": 0, "blocks": Counter(), "streams": Counter(),
            "samples": defaultdict(list), "first_identity": row.identity,
            "input_shapes": row.input_shapes, "output_shapes": row.output_shapes,
        })
        group["events"] += 1
        group["blocks"][row.blocks] += 1
        group["streams"][row.stream] += 1
        group["samples"]["task_duration_us"].append(row.duration_ns / 1000)
        for name, value in row.counters_us.items():
            group["samples"][name].append(value)
        if row.event.wait_ns is not None:
            group["samples"]["task_wait_us"].append(row.event.wait_ns / 1000)
    rows = []
    for signature, group in groups.items():
        samples = group.pop("samples")
        group["statistics"] = {
            name: {"present": len(values), "min": min(values),
                   "p50": statistics.median(values), "mean": statistics.fmean(values),
                   "max": max(values), "sum": sum(values)}
            for name, values in samples.items()
        }
        rows.append({"signature": signature, **group})
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "source": str(source), "source_sha256": digest.hexdigest(),
        "source_size_bytes": source.stat().st_size,
        "rows": rows, "wall_seconds": time.monotonic() - started,
        "scope": "whole_capture_descriptor_activity; no_step_or_phase_assignment",
        "timing_policy": {
            "primary": "Task Duration(us) retained without adjustment",
            "pmu": "separate exported counters; no additive interpretation or "
                   "scheduling-latency derivation without exporter semantics",
            "invocation": "no floor inferred from task/core counter difference",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.capture)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "groups": len(report["rows"]),
                      "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
