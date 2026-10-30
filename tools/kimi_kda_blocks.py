"""Inspect source-compatible KDA call blocks in a retained native event stream.

[Role] Pair observed consecutive runs with explicit external semantic boundaries.
[Invariants] Stage labels are conditional on pinned source order and block
  closure, never on duration clusters or total event count divided by four.
  Missing wait remains None. Mixed-capture activity is not a model-step span.
[Timeouts & Locks] One pass; 25 s, 1M raw rows, 1104 KDA events; bounded neighbors.
[Side-Effect Policy] Reads capture/source text only; writes one requested report.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

if __package__:
    from .kimi_stage_identity import identity
    from .model_convergence import worker_usage
else:
    from kimi_stage_identity import identity
    from model_convergence import worker_usage


STAGES = ("GATE_PREPARE", "POST_WU", "FWD_H", "FINALIZE")
SOURCE_FILES = (
    ("vllm_ascend/ops/kimi_kda.py", [[460, 521], [623, 632], [746, 753]]),
    ("csrc/attention/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.cpp",
     [[650, 669], [770, 800]]),
    ("csrc/attention/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp", [[15, 70]]),
)


def compact(event, line):
    return {
        "line": line, "name": event.name, "kind": event.op_type,
        "task_id": event.task_id, "stream": event.stream_id,
        "start_ns": event.start_ns, "end_ns": event.end_ns,
        "duration_ms": event.duration_ns / 1e6,
        "wait_ns": event.wait_ns, "wait_status": event.wait_status,
    }


def summary(values):
    return {"count": len(values), "sum_ms": sum(values),
            "mean_ms": statistics.mean(values) if values else None,
            "p50_ms": statistics.median(values) if values else None,
            "min_ms": min(values) if values else None,
            "max_ms": max(values) if values else None}


def inspect(capture, source_root, *, stream_id="61", neighbors=24):
    from mindie_profiler import events, normalized, sources_csv

    started = time.monotonic()
    schema = sources_csv.probe_csv(capture)
    cols = schema.columns
    digest = hashlib.sha256()
    stats = events.QueryStats(rank_id="", window_label="kda_blocks", budget=1_000_000)
    history, global_history = deque(maxlen=neighbors), deque(maxlen=2)
    runs, active, pending = [], None, None
    raw_rows, stream_rows, kda_rows, order_errors, overlaps = 0, 0, 0, 0, 0
    prior_start, prior_end = None, None
    descriptors = {}

    def finish(run):
        before, after = run.pop("before"), run.pop("after")
        kinds_before = [item["kind"] for item in before]
        kinds_after = [item["kind"] for item in after]
        anchors = {
            "preceding_short_convolution": any("CausalConv1d" in kind for kind in kinds_before),
            "preceding_qk_normalization": any("l2norm" in kind for kind in kinds_before),
            "following_gated_normalization": any("layer_norm_gated" in kind for kind in kinds_after),
            "following_output_projection": "MatmulReduceScatterV2" in kinds_after,
        }
        sequence = run["events"]
        signatures = set(item["descriptor_key"] for item in sequence)
        descriptor = descriptors[next(iter(signatures))]
        shapes = descriptor["input_shapes"]
        source_split_geometry = (
            len(shapes) == 10 and shapes[0] == [1, 8192, 12, 128]
            and shapes[8] == [2] and shapes[9] == [256]
            and descriptor["input_dtypes"][0] == "DT_BF16")
        monotonic = all(a["end_ns"] <= b["start_ns"]
                        for a, b in zip(sequence, sequence[1:]))
        closed = bool(before and after)
        compatible = (len(sequence) == 4 and len(signatures) == 1 and source_split_geometry
                      and monotonic and closed and all(anchors.values()))
        run.update(
            observed_run_length=len(sequence), closed_by_non_kda_events=closed,
            internal_time_ordered_nonoverlapping=monotonic,
            descriptor_uniform=len(signatures) == 1,
            source_split_geometry_compatible=source_split_geometry,
            semantic_boundary_anchors=anchors,
            preceding_names=[item["name"] for item in before],
            following_names=[item["name"] for item in after],
            nearest_before=before[-1] if before else None,
            nearest_after=after[0] if after else None,
            status="source_compatible_conditional" if compatible else "unresolved",
            activity_ms=sum(item["duration_ms"] for item in sequence),
            span_ms=(sequence[-1]["end_ns"] - sequence[0]["start_ns"]) / 1e6,
        )
        for position, event in enumerate(sequence):
            event["observed_position"] = position
            event["source_conditioned_stage"] = STAGES[position] if compatible else None
        if len(runs) < 4:
            run["boundary_event_examples"] = {"before": before, "after": after}
        runs.append(run)

    def lines(handle):
        for line in handle:
            digest.update(line)
            yield line.decode("utf-8-sig")

    with Path(capture).open("rb") as handle:
        reader = csv.reader(lines(handle))
        next(reader)
        for index, cells in enumerate(reader):
            raw_rows += 1
            if raw_rows > 1_000_000 or time.monotonic() - started > 25:
                raise TimeoutError("kda_call_block_audit_budget_exceeded")
            global_item = {"line": index + 2, "name": cells[cols["name"]],
                           "kind": cells[cols["type"]], "stream": cells[cols["stream_id"]]}
            if cells[cols["stream_id"]] != stream_id:
                global_history.append(global_item)
                continue
            event = sources_csv.decode_csv_row(
                cells, index, schema, rank_id="", rank_index=None, stats=stats,
                strict=True, include_zero_duration=True)
            stream_rows += 1
            order_errors += prior_start is not None and event.start_ns < prior_start
            overlaps += prior_end is not None and event.start_ns < prior_end
            prior_start, prior_end = event.start_ns, event.end_ns
            item = compact(event, index + 2)
            if event.op_type == "ChunkKdaFwd":
                kda_rows += 1
                if kda_rows > 1104:
                    raise ValueError("kda_call_block_audit_selected_row_budget_exceeded")
                if pending is not None:
                    finish(pending)
                    pending = None
                if active is None:
                    active = {"id": f"run{len(runs)}", "before": list(history),
                              "global_before": list(global_history), "after": [], "events": []}
                row = normalized.row_from_event(event)
                descriptor = {key: getattr(row, key) for key in (
                    "input_shapes", "output_shapes", "input_dtypes", "output_dtypes",
                    "input_formats", "output_formats")}
                descriptor = json.loads(json.dumps(descriptor))
                key = hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
                descriptors[key] = descriptor
                item["descriptor_key"] = key
                active["events"].append(item)
            else:
                if active is not None:
                    pending, active = active, None
                if pending is not None:
                    pending["after"].append(item)
                    if len(pending["after"]) == neighbors:
                        finish(pending)
                        pending = None
            history.append(item)
            global_history.append(global_item)
    if active is not None:
        finish(active)
    if pending is not None:
        finish(pending)
    compatible = [run for run in runs if run["status"] == "source_compatible_conditional"]
    positional = {}
    for position, stage in enumerate(STAGES):
        positional[stage] = summary([run["events"][position]["duration_ms"] for run in compatible])
    return {
        "schema": "kimi-kda-call-blocks/1", "status": "completed",
        "source": {"path": schema.path, "size_bytes": schema.size_bytes,
                   "sha256": digest.hexdigest()},
        "selected_stream": stream_id, "raw_rows_scanned": raw_rows,
        "stream_rows": stream_rows, "kda_rows": kda_rows,
        "stream_start_order_violations": order_errors, "stream_interval_overlaps": overlaps,
        "consecutive_run_lengths": dict(Counter(run["observed_run_length"] for run in runs)),
        "observed_runs": len(runs), "source_compatible_closed_runs": len(compatible),
        "unresolved_run_ids": [run["id"] for run in runs if run not in compatible],
        "activity": summary([event["duration_ms"] for run in runs for event in run["events"]]),
        "closed_run_activity": summary([run["activity_ms"] for run in compatible]),
        "closed_run_spans": summary([run["span_ms"] for run in compatible]),
        "conditional_stage_activity": positional, "descriptors": descriptors, "runs": runs,
        "pinned_source": {
            "revision": "ae923905948f360f4b7ab06df52d0789a0c9316e",
            "stage_order": list(STAGES), "stage_attribute_observed": False,
            "sources": [{**identity(Path(source_root) / relative), "lines": ranges}
                        for relative, ranges in SOURCE_FILES],
        },
        "comparison_scope": "source-compatible conditional four-stage core block; not exact binary or direct stage-attribute identity",
        "assumptions": [
            "Observed block order follows the pinned Ascend950 split-stage wrapper (chunk64 and present cu_seqlens).",
            "Each block is a maximal observed consecutive KDA run, with short-convolution/qk-normalization before and gated-normalization/output-projection after.",
            "Adjacent semantic names establish source-compatible boundaries, not captured tensor pointers.",
            "A model layer/step identity is not inferred; activity is distinct from the complete model critical path.",
        ],
        "producer": identity(__file__),
        "parser_sources": [identity(module.__file__) for module in (events, normalized, sources_csv)],
        "budgets": {"seconds": 25, "raw_rows": 1_000_000, "kda_rows": 1104,
                    "neighbors_per_boundary": neighbors},
        "full_capture_passes": 1, "wall_seconds": time.monotonic() - started,
        "peak_rss_mib": worker_usage().get("peak_rss_mib"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    result = inspect(args.capture, args.source_root)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in (
        "observed_runs", "consecutive_run_lengths", "source_compatible_closed_runs",
        "stream_start_order_violations", "stream_interval_overlaps",
        "conditional_stage_activity", "wall_seconds", "peak_rss_mib")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
