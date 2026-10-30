"""Source-bound activation and grouped-matrix observation adapters.

[Role] Normalize retained call, burst, replay and kernel-sum observations.
[Invariants] No fit, timing correction, implicit split or boundary conversion.
  Failed calls remain conditions without samples. Producer hashes bind the
  interpretation; capture hashes identify samples and actual acquisition runs.
[Timeouts & Locks] Streaming JSON/CSV; one call condition or bounded step ledger.
[Side-Effect Policy] Read-only source access; never imports a hardware producer.
"""
from collections import Counter
from itertools import groupby
import json
import math
from pathlib import Path

import ijson
from ijson.common import ObjectBuilder

from .kernels import kernel_rows
from .store import digest_file, json_items


def _checked(root, reference):
    path = Path(root) / reference["path"]
    if digest_file(path) != reference["sha256"]:
        raise ValueError(f"capture_source_hash_mismatch:{path}")
    return path


def _fields(path, wanted):
    result, builder, selected, depth = {}, None, None, 0
    with path.open("rb") as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if builder is None:
                if prefix not in wanted:
                    continue
                if event not in {"start_map", "start_array"}:
                    result[prefix] = value
                    continue
                selected, builder = prefix, ObjectBuilder()
            builder.event(event, value)
            if event in {"start_map", "start_array"}:
                depth += 1
            elif event in {"end_map", "end_array"}:
                depth -= 1
                if not depth:
                    result[selected], builder = builder.value, None
    return result


def _identity(manifest, header, *, level, boundary, implementation, phase,
              shape, dtype, layout, execution):
    return {
        "level": level, "boundary": boundary,
        "hardware": {**manifest["hardware"], "device": header["device"]},
        "implementation": {
            "producer_sha256": manifest["producer"]["sha256"],
            "software": {key: header[key] for key in ("torch", "torch_npu", "cann") if key in header},
            **implementation,
        },
        "phase": phase, "shape": shape, "dtype": dtype, "layout": layout,
        "execution": execution,
    }


def _condition(capture, path, pointer, identity, metadata, durations, *,
               pointers=None, samples_per_trial=None, status="measured"):
    samples = []
    for index, duration in enumerate(durations):
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"capture_event_duration_invalid:{path}#{pointer}:{index}")
        location = pointers[index] if pointers else f"{pointer}/{index}"
        sample_metadata = {
            **metadata, "condition_id": f"{capture['sha256']}:{pointer}", "repeat": index,
        }
        if samples_per_trial:
            sample_metadata.update(trial=index // samples_per_trial,
                                   repeat=index % samples_per_trial)
        samples.append({
            "sample_id": f"{capture['sha256']}:{location}",
            "identity": identity, "duration_ms": duration,
            "run_id": capture["sha256"], "split": "unassigned",
            "source": f"{path}#{location}", "metadata": sample_metadata,
        })
    return {
        "condition_id": f"{capture['sha256']}:{pointer}",
        "source": f"{path}#{pointer}", "identity": identity, "status": status,
        "metadata": metadata, "samples": samples,
    }


def _activation(manifest, capture, path):
    header = _fields(path, {"schema", "device", "visible", "torch", "torch_npu", "canaries"})
    if header["schema"] != "a3-activation-measure/1":
        raise ValueError("activation_capture_schema_mismatch")
    with path.open("rb") as stream:
        for key, row in ijson.kvitems(stream, "cells", use_float=True):
            variant = manifest["variants"][row["variant"]]
            prefix = "/cells/" + key.replace("~", "~0").replace("/", "~1")
            metadata = {
                "condition_label": key, "variant": row["variant"],
                "context": {name: row[name] for name in ("hidden", "batch", "seq", "scope")},
                "context_role": "producer_labels_not_additional_tensor_dimensions",
                "canary": header["canaries"].get(row["variant"]),
                "canary_drift_pct": row.get("canary_drift_pct"),
                "source_status": row["status"],
                "training_use": "not_assigned_for_current_prediction; historical_fit_reuse_not_excluded",
                "limitations": manifest["limitations"],
            }
            base = dict(
                level=variant["level"], implementation={"api": variant["api"], "variant": row["variant"]},
                phase="decode" if row["scope"] == "decode_step" else "prefill",
                shape={"tokens": row["tokens"], "intermediate": row["intermediate"]},
                dtype=row["dtype"], layout=variant["layout"],
            )
            execution = {
                "mode": "eager", "timer": "npu_events", "invocations": 1,
                "input_values": "zeros", "output_lifetime": "discard_after_call",
                "warmup_calls": 10, "timer_interleaving": "event_call_event",
            }
            identity = _identity(manifest, header, boundary="npu_event_call", execution=execution, **base)
            if row["status"] != "measured":
                yield _condition(capture, path, prefix, identity,
                                 {**metadata, "error": row.get("error")}, (), status=row["status"])
                continue
            passes = row["per_pass_ms"]
            samples = [value for values in passes for value in values]
            if samples != row["samples_ms"] or len(samples) != row["n"]:
                raise ValueError(f"activation_trial_samples_conflict:{key}")
            pointers = [f"{prefix}/per_pass_ms/{trial}/{repeat}"
                        for trial, values in enumerate(passes) for repeat in range(len(values))]
            yield _condition(capture, path, prefix + "/per_pass_ms", identity, metadata,
                             samples, pointers=pointers, samples_per_trial=len(passes[0]))
            count = row["burst_ops"]
            if len(row["burst_samples_ms"]) != row["burst_reps"]:
                raise ValueError(f"activation_burst_samples_conflict:{key}")
            burst_identity = _identity(
                manifest, header, boundary="npu_event_burst",
                execution={**execution, "invocations": count, "warmup_calls": 3,
                           "timer_interleaving": "event_calls_event"}, **{**base, "level": "subgraph"})
            yield _condition(
                capture, path, prefix + "/burst_samples_ms", burst_identity,
                {**metadata, "recorded_units": "batch_elapsed_ms_divided_by_invocations",
                 "duration_conversion": "recorded_per_call_ms_times_burst_ops",
                 "recorded_rounding_ms": 0.000001},
                [value * count for value in row["burst_samples_ms"]])


def _grouped(manifest, capture, path):
    header = _fields(path, {"schema", "device", "torch", "torch_npu", "cann", "timing"})
    if header["schema"] != "a5-qwen35-grouped-graph-probe/1":
        raise ValueError("grouped_capture_schema_mismatch")
    for index, row in enumerate(json_items(path, "rows.item")):
        prefix = f"/rows/{index}"
        metadata = {
            "condition_label": row["label"], "projection_label": row.get("projection"),
            "canary": row.get("canary"), "source_status": row["status"],
            "training_use": "not_assigned_for_current_prediction",
            "limitations": manifest["limitations"],
        }
        if row["status"] != "measured":
            yield _condition(capture, path, prefix, {},
                             {**metadata, "error": row.get("error"), "shape": row.get("shape")},
                             (), status=row["status"])
            continue
        shape = row["shape"]
        counts = [shape["tokens_per_expert"]] * shape["groups"]
        counts += [0] * (shape["experts"] - shape["groups"])
        if sum(counts) != shape["tokens"]:
            raise ValueError("grouped_producer_histogram_conflict")
        base = dict(
            implementation={"api": row["contract"]["api"], "contract": row["contract"]},
            phase="primitive",
            shape={"tokens": shape["tokens"], "k": shape["k"], "n": shape["n"],
                   "experts": shape["experts"], "expert_counts": counts},
            dtype={"input": "int8", "weight": "int8", "scale": "float32",
                   "per_token_scale": "float32", "output": "bfloat16"},
            layout={"input": "ND", "weight": "FRACTAL_NZ", "output": "ND"},
        )
        for field, mode, invocations in (
            ("eager_per_call", "eager", 1), ("graph_replay_1call", "graph", 1),
            ("graph_replay_8calls", "graph", 8),
        ):
            execution = {
                "mode": mode, "timer": "npu_events", "invocations": invocations,
                "warmup_calls": header["timing"]["warmup"], "activation_quant": "outside_timed_region",
                "output_lifetime": "captured_static_buffers" if mode == "graph" else "discard_after_call",
                "capture_warmup_calls": 3 if mode == "graph" else 0,
                "dependency": "serial",
            }
            identity = _identity(
                manifest, header, level="subgraph" if mode == "graph" else "operator",
                boundary="npu_event_graph_replay" if mode == "graph" else "npu_event_call",
                execution=execution, **base)
            timing = row.get(field)
            if timing is None:
                yield _condition(capture, path, prefix + "/" + field, identity,
                                 {**metadata, "error": row.get("graph_error")},
                                 (), status="unavailable")
                continue
            if (len(timing["raw_samples_ms"]) != timing["sample_count"]
                    or timing["sample_count"] != header["timing"]["trials"] * header["timing"]["samples_per_trial"]):
                raise ValueError("grouped_trial_samples_conflict")
            yield _condition(capture, path, prefix + "/" + field + "/raw_samples_ms", identity,
                             metadata, timing["raw_samples_ms"],
                             samples_per_trial=header["timing"]["samples_per_trial"])


def _csv_rows(path):
    previous = None
    # A marker defines a boundary even when it carries no measured duration.
    for row in kernel_rows(path, include_zero_duration=True):
        if previous is not None and row.start_ns < previous:
            raise ValueError(f"capture_csv_start_times_not_monotone:{path}:{row.line}")
        previous = row.start_ns
        yield row


def _ledger():
    return {"duration_ns": 0, "calls": Counter(), "source_preview": [], "source_count": 0,
            "contaminated": False, "devices": set()}


def _timestamp_group(manifest, rows):
    ledger, markers = _ledger(), 0
    for row in rows:
        name = row.name
        if name == manifest["marker"]:
            markers += 1
            continue
        grouped = name in manifest["kernels"]
        if not grouped and name != manifest["attention_kernel"]:
            continue
        if not row.input_shapes or not row.input_shapes[0]:
            raise ValueError(f"capture_tensor_shape_missing:{row.source}:{row.line}")
        tokens = row.input_shapes[0][0]
        limit = manifest["max_grouped_tokens"] if grouped else manifest["max_attention_query_tokens"]
        if tokens > limit:
            ledger["contaminated"] = True
        if not grouped:
            continue
        event = row.event
        signature = {
            "kernel": name, "input_shapes": event.input_shapes.strip('"'),
            "input_dtypes": event.input_dtypes, "input_formats": event.input_formats,
            "output_shapes": event.output_shapes.strip('"'),
            "output_dtypes": event.output_dtypes, "output_formats": event.output_formats,
        }
        ledger["calls"][json.dumps(signature, sort_keys=True)] += 1
        ledger["duration_ns"] += row.duration_ns
        if len(ledger["source_preview"]) < 5:
            ledger["source_preview"].append(row.line)
        ledger["source_count"] += 1
        ledger["devices"].add(event.device_id)
    if markers > 1:
        raise ValueError("served_grouped_duplicate_step_marker_timestamp")
    return ledger, markers


def _served_condition(manifest, capture, path, ledger, index, begin, end):
    identity = {
        "level": "subgraph", "boundary": "profiled_kernel_sum",
        "hardware": {**manifest["hardware"], "profile_device_ids": sorted(ledger["devices"])},
        "implementation": {
            "kernels": manifest["kernels"], "runtime": manifest["runtime"],
            "producer_sha256": manifest["producer"]["sha256"],
        },
        "phase": "decode",
        "shape": {"calls": [dict(json.loads(signature), occurrences=count)
                           for signature, count in sorted(ledger["calls"].items())]},
        "dtype": "per_tensor_in_call_descriptors", "layout": "per_tensor_in_call_descriptors",
        "execution": {"mode": "FULL_DECODE_ONLY", "aggregation": "sum_of_selected_kernel_durations",
                      "step_marker": manifest["marker"]},
    }
    metadata = {
        "condition_label": capture["label"], "step": index, "start_ns": begin, "end_ns": end,
        "source_line_preview": ledger["source_preview"], "source_line_count": ledger["source_count"],
        "training_use": "not_assigned_for_current_prediction",
        "unresolved_semantics": ["group_list_tensor_values", "loaded_kernel_binary_hash"],
        "limitations": manifest["limitations"],
        "selection": "exclude_intervals_with_prefill_sized_grouped_or_attention_calls",
    }
    return _condition(
        capture, path, f"/intervals/{index}", identity, metadata,
        () if ledger["contaminated"] else (ledger["duration_ns"] / 1e6,),
        status="prefill_contaminated" if ledger["contaminated"] else "measured")


def _served_grouped(manifest, capture, path):
    """Keep one active interval; equal-start kernels belong after that marker."""
    current, begin, markers = _ledger(), None, 0
    for start, rows in groupby(_csv_rows(path), key=lambda row: row.start_ns):
        pending, marker = _timestamp_group(manifest, rows)
        if marker:
            if begin is not None and current["calls"]:
                yield _served_condition(manifest, capture, path, current, markers - 1, begin, start)
            begin, current, markers = start, _ledger(), markers + 1
        if begin is not None:
            current["calls"].update(pending["calls"])
            current["duration_ns"] += pending["duration_ns"]
            current["devices"].update(pending["devices"])
            current["contaminated"] |= pending["contaminated"]
            current["source_preview"].extend(pending["source_preview"][:5 - len(current["source_preview"])])
            current["source_count"] += pending["source_count"]
    if markers < 2:
        raise ValueError("served_grouped_step_markers_missing")


ADAPTERS = {
    "activation_calls": _activation,
    "grouped_calls": _grouped,
    "served_grouped_kernels": _served_grouped,
}


def iter_conditions(manifest, *, root="."):
    """Read original conditions under a source-pinned interpretation manifest."""
    if manifest["schema"] != "mindie-sim.capture-protocol/1":
        raise ValueError("capture_protocol_schema_unknown")
    _checked(root, manifest["producer"])
    for reference in manifest.get("supporting_sources", ()):
        _checked(root, reference)
    adapter = ADAPTERS[manifest["adapter"]]
    for capture in manifest["captures"]:
        yield from adapter(manifest, capture, _checked(root, capture))


def iter_observations(manifest, *, root="."):
    """Yield the existing accuracy sample contract, with unassigned original use."""
    for condition in iter_conditions(manifest, root=root):
        yield from condition["samples"]


def capture_inventory(manifest, *, root=".", preview_limit=5):
    conditions, samples, boundaries, statuses = 0, 0, Counter(), Counter()
    identities, runs, sources, previews = set(), set(), set(), []
    for condition in iter_conditions(manifest, root=root):
        conditions += 1
        statuses[condition["status"]] += 1
        identity = condition["identity"]
        boundaries[identity.get("boundary", "unknown")] += 1
        if identity:
            identities.add(json.dumps(identity, sort_keys=True))
        samples += len(condition["samples"])
        for sample in condition["samples"]:
            runs.add(sample["run_id"])
            sources.add(sample["source"].split("#")[0])
        if len(previews) < preview_limit:
            previews.append({key: value for key, value in condition.items() if key != "samples"})
    return {
        "schema": "mindie-sim.capture-inventory/1",
        "adapter": manifest["adapter"], "source_captures": len(manifest["captures"]),
        "timing_conditions": conditions, "distinct_identities": len(identities),
        "event_or_kernel_sum_samples": samples, "acquisition_runs": len(runs),
        "conditions_by_boundary": dict(boundaries), "condition_statuses": dict(statuses),
        "sample_splits": {"unassigned": samples}, "condition_preview": previews,
        "sources": sorted(sources), "limitations": manifest["limitations"],
    }
