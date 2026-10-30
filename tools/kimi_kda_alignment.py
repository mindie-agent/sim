"""Compare independently closed KDA blocks with retained semantic predictions.

[Role] Join native occurrence blocks to an explicit four-stage core declaration.
[Invariants] Native launch statistics remain separate from complete-call sums.
  Source order is a conditional interpretation, never an observed stage scalar.
  Durations/counts enter only this comparison, never prediction generation.
[Timeouts & Locks] Bounded metadata inputs, no subprocesses or locks.
[Side-Effect Policy] Reads retained results and writes one requested report.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

if __package__:
    from .model_scoreboard_kimi import difference, read_json, source
else:
    from model_scoreboard_kimi import difference, read_json, source


def compare(blocks, prediction):
    if prediction["status"] != "completed" or prediction["mode"] != "kda":
        raise ValueError("kda_alignment_requires_completed_kda_prediction")
    if prediction["runtime_before"] != prediction["runtime_after"]:
        raise ValueError("kda_alignment_prediction_runtime_changed")
    declared = prediction["rows"][0]
    core = declared["prediction"]
    descriptors = declared["scenario_descriptors"]
    stages = core["stage_predictions"]
    names = [stage["stage"] for stage in stages]
    if names != blocks["pinned_source"]["stage_order"]:
        raise ValueError("kda_alignment_source_stage_order_differs")
    selected, stage_sums, activity, span, skipped = [], [0.0] * len(stages), 0.0, 0.0, []
    for run in blocks["runs"]:
        events = run["events"]
        compatible = (
            run["closed_by_non_kda_events"] and run["internal_time_ordered_nonoverlapping"]
            and run["descriptor_uniform"] and run["source_split_geometry_compatible"]
            and all(run["semantic_boundary_anchors"].values()) and len(events) == len(stages)
            and all(blocks["descriptors"][event["descriptor_key"]] == descriptors for event in events)
            and [event["source_conditioned_stage"] for event in events] == names
        )
        if not compatible:
            skipped.append(run["id"])
            continue
        selected.append(run["id"])
        for index, event in enumerate(events):
            stage_sums[index] += event["duration_ms"]
            activity += event["duration_ms"]
        span += (events[-1]["end_ns"] - events[0]["start_ns"]) / 1e6
    if not selected:
        return {"status": "incomparable", "reason": "no_closed_source_compatible_descriptor_matched_blocks",
                "unpaired_run_ids": skipped}
    count = len(selected)
    declared_tokens = core["stage_ledger"]["geometry"]["tokens"]
    chunk = core["stage_ledger"]["geometry"]["chunk_size"]
    chunks = core["stage_ledger"]["geometry"]["total_chunks"]
    predicted = core["per_call_ms"] * count
    comparisons = []
    for stage, observed_sum in zip(stages, stage_sums):
        estimate = stage["predicted_semantic_projection_ms"]
        comparisons.append({
            "stage": stage["stage"], "conditional_native_events": count,
            "observed_activity_sum_ms": observed_sum,
            "observed_mean_ms": observed_sum / count,
            "predicted_semantic_projection_ms": estimate,
            "predicted_activity_sum_ms": estimate * count,
            "sum_error": difference(estimate * count, observed_sum),
            "semantic_work": stage["semantic_work"],
            "native_mapping_status": stage["native_mapping_status"],
            "source_unresolved_work": stage["source_unresolved_work"],
        })
    comparisons.sort(key=lambda row: row["sum_error"]["absolute_gap_ms"], reverse=True)
    source_stages = [stage["stage"] for stage in stages
                     if stage["native_mapping_status"] == "source_phase_not_capture_binary_identity"]
    return {
        "schema": "kimi-kda-conditional-call-alignment/1",
        "status": "conditional_closed_call_blocks",
        "boundary": "four-stage native core; wrapper copies and external normalization excluded",
        "geometry": {
            **core["stage_ledger"]["geometry"],
            "descriptor_shapes_match": True,
            "declared_tokens": declared_tokens,
            "source_valid_token_interval": [(chunks - 1) * chunk + 1, declared_tokens],
            "valid_token_basis": "one sequence; chunk_indices has 2*chunks entries; cu_seqlens values absent",
            "scenario": "full descriptor extent selected explicitly; not a measured cu_seqlens value",
        },
        "matched_blocks": count, "unpaired_run_ids": skipped,
        "coverage": {"observed_runs": len(blocks["runs"]), "matched": count,
                     "unpaired": len(skipped)},
        "observed_activity_sum_ms": activity, "observed_block_span_sum_ms": span,
        "observed_mean_block_activity_ms": activity / count,
        "predicted_core_ms": core["per_call_ms"],
        "predicted_activity_sum_ms": predicted,
        "sum_error": difference(predicted, activity),
        "block_span_error": difference(predicted, span),
        "stages": comparisons,
        "stage_projection_sum_equals_core": math.isclose(
            sum(stage["predicted_semantic_projection_ms"] for stage in stages),
            core["per_call_ms"], rel_tol=1e-12),
        "runtime": prediction["runtime_before"],
        "kda_implementation": core.get("kda_implementation"),
        "source_selected_stages": source_stages,
        "raw_native_launch_statistics": blocks["activity"],
        "source_condition": blocks["pinned_source"],
        "assumptions": blocks["assumptions"] + [
            f"{count} calls are recovered from closed consecutive blocks, not native_event_count/4.",
            "Stage order is inferred conditionally from independent source and surrounding operators; stage attributes and capture binary identity remain unknown.",
            f"The {declared_tokens}-token scenario is the upper endpoint permitted by {chunks} source chunks; the captured cu_seqlens values are absent.",
            ("Source-selected phases: " + ", ".join(source_stages)
             + "; all other phases retain generic semantic work. Capture binary identity is not established."
             if source_stages else
             "The projection is a generic semantic mechanism, not reconstructed native tiling, scratch traffic or per-core ownership."),
            "Complete-call comparisons do not reclassify native per-launch mean/P50 as complete-call statistics.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    result = compare(read_json(args.blocks), read_json(args.prediction))
    result["inputs"] = {"blocks": source(args.blocks), "prediction": source(args.prediction)}
    result["producer"] = source(__file__)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result.get(key) for key in (
        "status", "matched_blocks", "observed_activity_sum_ms",
        "predicted_activity_sum_ms", "sum_error")}))
    return 0 if result["status"] == "conditional_closed_call_blocks" else 2


if __name__ == "__main__":
    raise SystemExit(main())
