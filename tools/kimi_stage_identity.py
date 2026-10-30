"""Inspect retained KDA occurrence identifiers without inferring native stages.

[Role] Stream one CSV and project selected rows through the profiler decoder.
[Invariants] No stage is inferred from counts, order, duration or pipe activity.
  Missing wait time remains None; it is never converted into zero waiting.
[Timeouts & Locks] At most 25 seconds, 1M CSV rows and 1104 retained KDA rows.
[Side-Effect Policy] Read-only capture; write one requested evidence JSON.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import time

if __package__:
    from .model_convergence import worker_usage
else:
    from model_convergence import worker_usage


def identity(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest, "size_bytes": path.stat().st_size}


def inspect(capture):
    from mindie_profiler import events, normalized, sources_csv, sources_db

    started = time.monotonic()
    schema = sources_csv.probe_csv(capture)
    columns = {name: index for index, name in enumerate(schema.header)}
    stage_columns = [name for name in schema.header
                     if re.search(r"stage|tiling|attribute|attrs", name, re.I)]
    fields = (
        "Device_id", "Model ID", "Task ID", "Stream ID", "Context ID",
        "Op Name", "OP Type", "OP State", "Task Type", "Block Num",
        "Mix Block Num", "HF32 Eligible",
    )
    values = {name: Counter() for name in (*fields, *stage_columns) if name in columns}
    descriptor_keys = Counter()
    configurations = Counter()
    durations = []
    waits = Counter()
    first, last = [], deque(maxlen=4)
    count, raw_rows = 0, 0
    first_start, final_end = None, None
    digest = hashlib.sha256()
    stats = events.QueryStats(rank_id="", window_label="kda_stage_audit", budget=1_000_000)

    def lines(stream):
        for line in stream:
            digest.update(line)
            yield line.decode("utf-8-sig")

    with Path(capture).open("rb") as stream:
        reader = csv.reader(lines(stream))
        next(reader)
        for index, cells in enumerate(reader):
            raw_rows += 1
            if raw_rows > 1_000_000 or time.monotonic() - started > 25:
                raise TimeoutError("kda_stage_audit_scan_budget_exceeded")
            if cells[columns["OP Type"]] != "ChunkKdaFwd":
                continue
            if count >= 1104:
                raise ValueError("kda_stage_audit_more_than_1104_selected_rows")
            event = sources_csv.decode_csv_row(
                cells, index, schema, rank_id="", rank_index=None, stats=stats,
                strict=True, include_zero_duration=True)
            row = normalized.row_from_event(event)
            count += 1
            durations.append(row.duration_ns)
            waits[event.wait_status] += 1
            first_start = row.start_ns if first_start is None else min(first_start, row.start_ns)
            final_end = row.end_ns if final_end is None else max(final_end, row.end_ns)
            selected = {name: cells[columns[name]] for name in values}
            for name, value in selected.items():
                values[name][value] += 1
            shape_key = json.dumps({
                name: getattr(row, name) for name in (
                    "input_shapes", "output_shapes", "input_dtypes", "output_dtypes",
                    "input_formats", "output_formats")
            }, sort_keys=True)
            descriptor_keys[hashlib.sha256(shape_key.encode()).hexdigest()] += 1
            config = tuple(selected[name] for name in (
                "Task Type", "Block Num", "Mix Block Num", "OP State", "HF32 Eligible"))
            configurations[config] += 1
            record = {
                "csv_line": row.line, **selected,
                "event_id": row.event_id, "start_ns": row.start_ns,
                "end_ns": row.end_ns, "duration_ms": row.duration_ns / 1e6,
                "wait_ns": event.wait_ns, "wait_status": event.wait_status,
            }
            if len(first) < 12:
                first.append(record)
            last.append(record)
    field_inventory = {}
    for name, counts in values.items():
        field_inventory[name] = {
            "distinct_values": len(counts), "empty_rows": counts.get("", 0),
            "values": dict(counts) if len(counts) <= 16 else None,
            "first_distinct_values": list(counts)[:12] if len(counts) > 16 else None,
        }
    if not count:
        raise ValueError("no_chunk_kda_fwd_rows_in_capture")
    known_labels = [name for name in values["Op Name"]
                    if re.search(r"gate_prepare|post_wu|fwd_h|finalize", name, re.I)]
    report = {
        "schema": "kimi-kda-stage-identity-audit/1",
        "status": "completed",
        "source": {"path": schema.path, "sha256": digest.hexdigest(),
                   "size_bytes": schema.size_bytes},
        "header": list(schema.header), "header_sha256": schema.header_sha256,
        "raw_rows_scanned": raw_rows, "selected_rows": count,
        "selected_family": "ChunkKdaFwd", "field_inventory": field_inventory,
        "wait_status_counts": dict(waits),
        "descriptor_signature_counts": dict(descriptor_keys),
        "launch_configurations": [
            {**dict(zip(("task_type", "block_num", "mix_block_num", "op_state", "hf32"), key)),
             "count": value} for key, value in configurations.items()
        ],
        "selected_activity": {
            "sum_ms": sum(durations) / 1e6,
            "mean_ms": statistics.mean(durations) / 1e6,
            "p50_ms": statistics.median(durations) / 1e6,
            "first_start_ns": first_start, "last_end_ns": final_end,
            "scope": "native task activity from a mixed capture; not complete-call or model-step duration",
        },
        "examples": {"first_12": first, "last_4": list(last)},
        "stage_identity": {
            "stage_or_tiling_or_attr_columns": stage_columns,
            "known_source_stage_labels_in_native_names": known_labels,
            "status": ("candidate_fields_require_source_binding" if stage_columns or known_labels
                       else "not_recorded_in_selected_export"),
            "complete_calls": None, "stage_population_counts": None,
            "reason": ("Candidate stage fields are retained, but their mapping has not been established."
                       if stage_columns or known_labels else
                       "No stage scalar, stage-specific native label or tiling-key-to-stage mapping is recorded."),
        },
        "limitations": [
            "Model/Context/Task/Stream identifiers preserve occurrence identity; no source mapping establishes them as native stage identifiers.",
            "No grouping by event count, ordering, duration cluster or PMU activity establishes complete calls.",
            "The source-conditioned wrapper stage ledger is a separate declaration, not captured stage evidence.",
        ],
        "producer": identity(__file__),
        "parser_sources": [identity(module.__file__)
                           for module in (events, normalized, sources_csv, sources_db)],
        "argv": sys.argv[1:], "wall_seconds": time.monotonic() - started,
        "peak_rss_mib": worker_usage().get("peak_rss_mib"),
        "budgets": {"seconds": 25, "raw_rows": 1_000_000, "selected_rows": 1104},
        "full_capture_passes": 1,
    }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    result = inspect(args.capture)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "raw_rows_scanned", "selected_rows", "selected_activity",
        "stage_identity", "wall_seconds", "peak_rss_mib")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
