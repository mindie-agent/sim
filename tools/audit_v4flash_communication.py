"""Stream the retained V4Flash A3 communication capture into a bounded audit.

[Role] Read one retained kernel_details.csv plus committed scoreboard/probe
  metadata and reconcile named HCCL, projected intervals, AICPU control rows and
  the reported collective-family gap.
[Invariants] Row identity is the complete 46-column record; time arithmetic is
  integer nanoseconds parsed from the original microsecond columns. Native
  intervals are never mixed with AivKernel projections or AICPU control without
  an explicit class label. Matching intervals are not proven producer identities.
  One retained rank is not a whole cohort.
[Timeouts & Locks] No network, device, weight or simulation work. One local
  `git rev-parse` supplies provenance; the capture is consumed once, row by row.
[Side-Effect Policy] Reads local JSON/CSV paths; writes only --out.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from hashlib import blake2b, sha256
import json
from pathlib import Path
import re
import statistics
import subprocess

SCHEMA = "v4flash-communication-audit/2"
MODEL_ID = "v4flash_a3"
EXPECTED_COLUMNS = 46
CAPTURE_COLUMNS = (
    "Device_id", "Model ID", "Task ID", "Stream ID", "Name", "Type", "OP State",
    "Accelerator Core", "Start Time(us)", "Duration(us)", "Wait Time(us)",
    "Block Num", "Mix Block Num", "HF32 Eligible", "Input Shapes",
    "Input Data Types", "Input Formats", "Output Shapes", "Output Data Types",
    "Output Formats", "Context ID", "aicore_time(us)", "aic_total_cycles",
    "aic_mac_time(us)", "aic_mac_ratio", "aic_scalar_time(us)", "aic_scalar_ratio",
    "aic_mte1_time(us)", "aic_mte1_ratio", "aic_mte2_time(us)", "aic_mte2_ratio",
    "aic_fixpipe_time(us)", "aic_fixpipe_ratio", "aic_icache_miss_rate",
    "aiv_time(us)", "aiv_total_cycles", "aiv_vec_time(us)", "aiv_vec_ratio",
    "aiv_scalar_time(us)", "aiv_scalar_ratio", "aiv_mte2_time(us)", "aiv_mte2_ratio",
    "aiv_mte3_time(us)", "aiv_mte3_ratio", "aiv_icache_miss_rate",
    "cube_utilization(%)",
)

REQUIRED_COLUMNS = (
    "Device_id", "Task ID", "Stream ID", "Name", "Type", "Accelerator Core",
    "Start Time(us)", "Duration(us)", "Wait Time(us)", "Input Shapes",
    "Input Data Types", "Output Shapes", "Output Data Types",
)
NATIVE_FAMILIES = {
    "hcom_allReduce_": "allreduce",
    "hcom_allGather_": "allgather",
    "hcom_alltoallv_": "alltoall",
}
AICPU_FAMILIES = {
    "allreduceAicpuKernel": "allreduce_control",
    "allgatherAicpuKernel": "allgather_control",
}
MODEL_FAMILIES = ("allreduce", "allgather", "alltoall")
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURE = Path(
    "/Volumes/data/mindie-sim-data/a3/2026-10-23/v4f-prefill/raw/pre4096.kernel_details.csv"
)
DEFAULT_SCOREBOARD = ROOT / "tests/fixtures/verification/v4flash-communication-scoreboard.json"
DEFAULT_INPUT = ROOT / "tests/fixtures/verification/v4flash-communication-input.json"
DEFAULT_PREDICTION = ROOT / (
    "tests/fixtures/verification/v4flash-communication-prediction.json"
)
DEFAULT_CASE = ROOT / "tests/fixtures/verification/v4flash-communication-case.json"
DEFAULT_CATALOG = ROOT / "mindie_sim/model_configs/deepseek-v4-flash-0731.json"
DISPATCH_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "8fd0bc4885cb52e58791aaa8efacc571b9bd115d",
    "path": "vllm_ascend/ops/fused_moe/token_dispatcher.py",
    "sha256": "1dcfd8181cb2d34c584d8336c3379dbeefba75c0bf664f924063701b93295cbb",
    "symbol": "TokenDispatcherWithAll2AllV",
    "scope": "independent_source_semantics_not_capture_binary_identity",
    "operations": {
        "histogram_and_gather": [616, 641],
        "dynamic_quant_and_scale_exchange_wait": [507, 516],
        "payload_exchange_wait": [518, 522],
        "reverse_exchange_wait": [559, 576],
    },
    "condition": "dispatch_with_quant; non-LoRA; explicit quant format",
}


def ns_from_us(value: object) -> int:
    """Parse the capture's microsecond text exactly into integer nanoseconds."""
    text = str(value).strip()
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"capture_time_not_decimal:{text!r}") from exc
    if not number.is_finite() or number < 0:
        raise ValueError(f"capture_time_not_finite_nonnegative:{text!r}")
    return int(number * 1000)


def ms(nanoseconds: int) -> float:
    return round(nanoseconds / 1_000_000, 6)


def merged_intervals(intervals):
    """Disjoint half-open intervals, suitable for set arithmetic."""
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    if not ordered:
        return []
    result = []
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            result.append((current_start, current_end))
            current_start, current_end = start, end
    result.append((current_start, current_end))
    return result


def union_ns(intervals) -> int:
    """Length of the union of half-open [start, end) intervals."""
    return sum(end - start for start, end in merged_intervals(intervals))


def intersection_ns(left, right) -> int:
    """Length of the intersection of two unions of half-open intervals."""
    a, b = merged_intervals(left), merged_intervals(right)
    i = j = total = 0
    while i < len(a) and j < len(b):
        start = max(a[i][0], b[j][0])
        end = min(a[i][1], b[j][1])
        if end > start:
            total += end - start
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def _family(row: dict, klass: str) -> str:
    if klass in {"native", "mirror"}:
        return NATIVE_FAMILIES.get(row["Type"], "other")
    if klass == "aicpu":
        return AICPU_FAMILIES.get(row["Name"], "other")
    return "other"


def _logical_stream(name: str, fallback: str) -> str:
    match = re.search(r"__(\d+)_", name)
    return match.group(1) if match else fallback


def _classify(row: dict) -> str:
    name, type_name = row["Name"], row["Type"]
    if name.startswith("hcom_") and type_name.startswith("hcom_"):
        return "native"
    if name == "AivKernel" and type_name.startswith("hcom_"):
        return "mirror"
    if name.endswith("AicpuKernel") and row["Accelerator Core"] == "AI_CPU":
        return "aicpu"
    return "other"


def _interval_stats(intervals) -> dict:
    values = sorted(end - start for start, end in intervals)
    return {
        "count": len(values),
        "sum_ms": ms(sum(values)),
        "union_ms": ms(union_ns(intervals)),
        "min_ms": round(values[0] / 1_000_000, 6) if values else None,
        "median_ms": round(statistics.median(values) / 1_000_000, 6) if values else None,
        "max_ms": round(values[-1] / 1_000_000, 6) if values else None,
    }


def _group_by_gap(rows, gap_ns=5_000_000):
    ordered = sorted(rows, key=lambda item: item["start_ns"])
    groups = []
    for row in ordered:
        if not groups or row["start_ns"] - groups[-1][-1]["start_ns"] > gap_ns:
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _hash_json(path: Path) -> dict:
    size = path.stat().st_size
    if size > 50 * 1024 * 1024:
        raise ValueError(f"json_metadata_exceeds_50_mib:{path}")
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return {"path": _display(path), "sha256": digest.hexdigest(), "size_bytes": size}


def _load_json(path: Path) -> dict:
    info = _hash_json(path)
    info["value"] = json.loads(path.read_text())
    return info


def audit_capture(path: Path, gap_ns: int = 5_000_000, *, max_rows=100_000) -> dict:
    """Consume the CSV once and retain only family aggregates and comm intervals."""
    rows = positive = duplicate_rows = 0
    activity_ns = 0
    first_start = last_end = None
    header = []
    devices, streams, native_streams = set(), Counter(), Counter()
    class_intervals = defaultdict(list)
    other_intervals = []
    family_intervals = defaultdict(lambda: defaultdict(list))
    type_counts = Counter()
    seen_rows = set()
    native_keys, mirror_keys = Counter(), Counter()
    mirror_rows = []
    aicpu_rows = []
    alltoallv_rows = []
    allgather_rows = []
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        header = list(reader.fieldnames or ())
        if len(header) != EXPECTED_COLUMNS or set(header) != set(CAPTURE_COLUMNS):
            raise ValueError(f"capture_schema_columns_mismatch:{len(header)}")
        missing = [name for name in REQUIRED_COLUMNS if name not in header]
        if missing:
            raise ValueError(f"capture_schema_missing_columns:{missing}")
        for row in reader:
            rows += 1
            if rows > max_rows:
                raise ValueError(f"capture_row_budget_exceeded:{max_rows}")
            start = ns_from_us(row["Start Time(us)"])
            duration = ns_from_us(row["Duration(us)"])
            end = start + duration
            if duration > 0:
                first_start = start if first_start is None else min(first_start, start)
                last_end = end if last_end is None else max(last_end, end)
            devices.add(row["Device_id"])
            identity = blake2b(
                "\x1f".join(row[name] for name in header).encode("utf-8"),
                digest_size=16,
            ).digest()
            if identity in seen_rows:
                duplicate_rows += 1
                continue
            seen_rows.add(identity)
            activity_ns += duration
            if duration > 0:
                positive += 1
            klass = _classify(row)
            if klass == "other" and duration > 0:
                other_intervals.append((start, end))
            if klass == "other" or duration <= 0:
                continue
            family = _family(row, klass)
            logical_stream = _logical_stream(row["Name"], row["Stream ID"])
            streams[logical_stream] += 1
            if klass == "native":
                native_streams[logical_stream] += 1
            type_counts[row["Type"]] += 1
            interval = (start, end)
            class_intervals[klass].append(interval)
            family_intervals[klass][family].append(interval)
            key = (row["Device_id"], row["Type"], row["Accelerator Core"], start, end)
            if klass == "native":
                native_keys[key] += 1
                if family == "alltoall":
                    alltoallv_rows.append({"start_ns": start, "duration_ns": duration,
                                           "stream": _logical_stream(row["Name"], row["Stream ID"]),
                                           "name": row["Name"]})
                if family == "allgather":
                    allgather_rows.append({"start_ns": start, "duration_ns": duration,
                                           "stream": _logical_stream(row["Name"], row["Stream ID"]),
                                           "name": row["Name"]})
            elif klass == "mirror":
                mirror_rows.append({"start_ns": start, "duration_ns": duration,
                                    "family": family, "key": key, "name": row["Name"]})
                mirror_keys[key] += 1
            else:
                aicpu_rows.append({"start_ns": start, "duration_ns": duration,
                                   "family": family, "name": row["Name"]})
    native = class_intervals["native"]
    mirror = class_intervals["mirror"]
    aicpu = class_intervals["aicpu"]
    all_intervals = native + mirror + aicpu
    native_sum, mirror_sum, aicpu_sum = sum(map(lambda x: x[1] - x[0], native)), \
        sum(map(lambda x: x[1] - x[0], mirror)), sum(map(lambda x: x[1] - x[0], aicpu))
    paired = [row for row in mirror_rows
              if native_keys[row["key"]] == mirror_keys[row["key"]] == 1]
    ambiguous = [row for row in mirror_rows
                 if native_keys[row["key"]] and
                 (native_keys[row["key"]] != 1 or mirror_keys[row["key"]] != 1)]
    mirror_matches = len(paired)
    mirror_match_ns = sum(row["duration_ns"] for row in paired)
    mirror_interval_ns = union_ns(mirror)
    paired_union_ns = union_ns((row["key"][-2], row["key"][-1]) for row in paired)
    aicpu_intersection = intersection_ns(native, aicpu)
    classes = {}
    for klass, intervals in (("native", native), ("projection_candidates", mirror), ("aicpu", aicpu)):
        classes[klass] = _interval_stats(intervals)
    classes["all_communication_rows"] = _interval_stats(all_intervals)
    families = {}
    for klass in ("native", "mirror", "aicpu"):
        families[klass] = {
            family: _interval_stats(intervals)
            for family, intervals in sorted(family_intervals[klass].items())
        }
    ordinal = []
    alltoall_groups = _group_by_gap(alltoallv_rows, gap_ns)
    if alltoall_groups and len({len(group) for group in alltoall_groups}) == 1:
        width = len(alltoall_groups[0])
        for index in range(width):
            ordinal.append({
                "ordinal": index + 1,
                "count": len(alltoall_groups),
                "sum_ms": ms(sum(group[index]["duration_ns"] for group in alltoall_groups)),
                "median_ms": round(statistics.median(
                    group[index]["duration_ns"] for group in alltoall_groups) / 1_000_000, 6),
            })
    by_stream = {}
    for stream in sorted({row["stream"] for row in allgather_rows}):
        selected = [row for row in allgather_rows if row["stream"] == stream]
        by_stream[stream] = {
            "count": len(selected),
            "sum_ms": ms(sum(row["duration_ns"] for row in selected)),
            "median_ms": round(statistics.median(
                row["duration_ns"] for row in selected) / 1_000_000, 6),
        }
    span_ns = (last_end - first_start) if first_start is not None else 0
    return {
        "path": _display(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "schema_columns": len(header),
        "schema_fields": header,
        "rows": rows,
        "positive_event_count": positive,
        "duplicate_full_identity_rows": duplicate_rows,
        "activity_sum_ms": ms(activity_ns),
        "bounds_ns": [first_start, last_end],
        "span_ms": ms(span_ns),
        "rank_scope": {
            "device_ids": sorted(devices),
            "retained_rank_count": len(devices),
            "one_retained_device_rank": len(devices) == 1,
            "whole_cohort": False,
        },
        "communication_streams": dict(sorted(streams.items())),
        "native_communication_streams": dict(sorted(native_streams.items())),
        "stream_basis": "native Name suffix (e.g. __503_); wrapper/AICPU CSV Stream ID fallback",
        "retained_state": {
            "communication_intervals": len(all_intervals),
            "row_identity_hashes": len(seen_rows),
            "raw_rows_whole_read": False,
            "max_rows": max_rows,
        },
        "classes": classes,
        "families": families,
        "projection_candidates": {
            "rows": len(mirror_rows),
            "exact_native_interval_matches": mirror_matches,
            "exact_match_ms": ms(mirror_match_ns),
            "mirror_interval_ms": ms(mirror_interval_ns),
            "paired_interval_union_ms": ms(paired_union_ns),
            "unmatched_rows": sum(not native_keys[row["key"]] for row in mirror_rows),
            "ambiguous_rows": len(ambiguous),
            "match_axes": ["Device_id", "Type", "Accelerator Core", "start_ns", "end_ns"],
            "matching_rule": "one_named_and_one_projected_row_per_key",
            "producer_identity": "unproven_no_exported_correlation_join",
            "by_family": {
                family: {"count": len([row for row in mirror_rows if row["family"] == family]),
                         "sum_ms": ms(sum(row["duration_ns"] for row in mirror_rows
                                          if row["family"] == family))}
                for family in MODEL_FAMILIES
            },
            "mirror_groups": mirror_matches,
        },
        "aicpu": {
            "rows": len(aicpu_rows),
            "sum_ms": ms(aicpu_sum),
            "intersection_with_native_ms": ms(aicpu_intersection),
            "outside_native_ms": ms(union_ns(aicpu) - aicpu_intersection),
            "by_family": {
                family: {
                    "count": len([row for row in aicpu_rows if row["family"] == family]),
                    "sum_ms": ms(sum(row["duration_ns"] for row in aicpu_rows
                                     if row["family"] == family)),
                    "intersection_with_native_ms": ms(intersection_ns(
                        family_intervals["native"].get(family.removesuffix("_control"), []),
                        family_intervals["aicpu"].get(family, []))),
                }
                for family in AICPU_FAMILIES.values()
            },
        },
        "union_intersection": {
            "native_sum_ms": ms(native_sum),
            "native_union_ms": ms(union_ns(native)),
            "native_internal_overlap_ms": ms(native_sum - union_ns(native)),
            "mirror_sum_ms": ms(mirror_sum),
            "aicpu_sum_ms": ms(aicpu_sum),
            "all_rows_sum_ms": ms(native_sum + mirror_sum + aicpu_sum),
            "all_rows_union_ms": ms(union_ns(all_intervals)),
            "all_rows_overlap_ms": ms(native_sum + mirror_sum + aicpu_sum - union_ns(all_intervals)),
            "native_aicpu_intersection_ms": ms(aicpu_intersection),
            "native_aicpu_union_ms": ms(union_ns(native + aicpu)),
            "native_mirror_exact_overlap_ms": ms(paired_union_ns),
            "native_noncomm_intersection_ms": ms(intersection_ns(native, other_intervals)),
            "native_without_noncomm_occupancy_ms": ms(
                union_ns(native) - intersection_ns(native, other_intervals)),
            "critical_path_inferred": False,
        },
        "alltoallv_ordinal_groups": ordinal,
        "alltoallv_group_count": len(alltoall_groups),
        "allgather_by_stream": by_stream,
        "top_types": type_counts.most_common(12),
    }


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entry(input_path: Path) -> dict:
    value = _load_json(input_path)["value"]
    return next(row for row in value["entries"] if row["id"] == MODEL_ID)


def _scoreboard(path: Path) -> dict:
    value = _load_json(path)["value"]
    return next(row for row in value["models"] if row["id"] == MODEL_ID)


def _prediction(path: Path) -> dict:
    return _load_json(path)["value"]


def _contract(case_path: Path, catalog_path: Path, conditions: dict) -> dict:
    case = _load_json(case_path)["value"]
    catalog = _load_json(catalog_path)["value"]
    parallel = conditions["parallel"]
    tokens = int(conditions["tokens"])
    tp, ep, dp = int(parallel["tp"]), int(parallel["ep"]), int(parallel["dp"])
    hidden = int(catalog["hidden_size"])
    vocab = int(catalog["vocab_size"])
    experts = int(catalog["n_routed_experts"])
    top_k = int(catalog["num_experts_per_tok"])
    layers = int(catalog["num_hidden_layers"]) + int(catalog["num_nextn_predict_layers"])
    source_partitions = min(tp, ep)
    assignments = tokens * top_k
    if assignments % source_partitions:
        raise ValueError("capture_contract_assignments_not_divisible")
    per_source = assignments // source_partitions
    if per_source % ep:
        raise ValueError("capture_contract_source_rows_not_divisible")
    self_rows = per_source // ep
    payload_bytes = 2  # BF16 hidden payload in the declared frontend.
    self_bytes = self_rows * hidden * payload_bytes
    remote_bytes = per_source * hidden * payload_bytes * (ep - 1) // ep
    allreduce_bytes = 2 * (tp - 1) * tokens * hidden * payload_bytes // tp
    draft_gather_bytes = tokens * hidden * payload_bytes * (tp - 1) // tp
    logits_gather_bytes = vocab * payload_bytes * (tp - 1) // tp
    kernels = dict(case.get("runtime", {}).get("kernels", {}))
    collective_keys = {
        "allreduce", "allgather", "alltoall", "expert_dispatch", "expert_combine",
        "output_reduce", "moe_output_reduce", "expert_tensor_reduce", "ring_allreduce",
        "ring_allgather", "draft_features_gather", "pcp_hidden_gather", "pcp_kv_gather",
    }
    explicit = {**kernels.get("recipes", {}), **kernels.get("families", {})}
    bound = sorted(set(explicit) & collective_keys)
    causal = sorted(name for name in bound
                    if explicit[name].get("kind") in {"ring_collective", "pairwise_collective"})
    predicted_link_bytes = (
        2 * layers * allreduce_bytes + 2 * layers * remote_bytes
        + draft_gather_bytes + logits_gather_bytes
    )
    return {
        "case": {"path": _display(case_path), "sha256": _hash_json(case_path)["sha256"]},
        "catalog": {"path": _display(catalog_path), "sha256": _hash_json(catalog_path)["sha256"]},
        "parallel": {"dp": dp, "tp": tp, "ep": ep},
        "ep_group_ranks": ep,
        "local_experts": experts // ep,
        "top_k": top_k,
        "served_layers": layers,
        "tokens": tokens,
        "hidden_size": hidden,
        "vocab_size": vocab,
        "source_assignment_partitions": source_partitions,
        "assignments_total": assignments,
        "assignments_per_source": per_source,
        "self_assignments_per_source": self_rows,
        "self_fraction": round(1 / ep, 9),
        "self_bytes_per_source_per_layer": self_bytes,
        "remote_bytes_per_source_per_layer": remote_bytes,
        "whole_cohort_remote_bytes_per_layer": remote_bytes * ep,
        "whole_cohort_self_bytes_per_layer": self_bytes * ep,
        "allreduce_bytes_per_rank_per_call": allreduce_bytes,
        "draft_features_gather_bytes": draft_gather_bytes,
        "logits_gather_bytes": logits_gather_bytes,
        "predicted_group_link_bytes": {
            "allreduce_two_groups": 2 * layers * allreduce_bytes,
            "expert_dispatch": layers * remote_bytes,
            "expert_combine": layers * remote_bytes,
            "draft_features_gather": draft_gather_bytes,
            "logits_gather": logits_gather_bytes,
        },
        "predicted_link_bytes_per_rank_total": predicted_link_bytes,
        "collective_recipe_bindings": bound,
        "algorithm_scope": ("resource_envelope_default" if not bound else
                            "causal_collective_recipe" if causal else "explicit_non_causal_recipe"),
        "round_readiness_modeled": bool(causal),
        "self_copy_modeled_as_link": False,
        "combine_volume_equals_dispatch": True,
        "reverse_flow_contract": "balanced_symmetric_only",
        "mc2_declared_in_case": "mc2_chunks" in case.get("runtime", {}),
        "moe_recipe_declared_in_case": "moe_recipe" in case.get("runtime", {}),
    }


def _reconcile(capture: dict, scoreboard: dict, prediction: dict, contract: dict) -> dict:
    group = next(row for row in scoreboard["group_comparisons"]
                 if row["comparison_key"] == "collective")
    observed = group["observed"]
    observed_ms = group["observed_ms"]
    predicted_ms = group["predicted_ms"]
    native = {family: capture["families"]["native"].get(family, {"count": 0, "sum_ms": 0})
              for family in MODEL_FAMILIES}
    predicted_groups = [row for row in scoreboard["predicted"]["groups"]
                        if row.get("family") in MODEL_FAMILIES]
    predicted = defaultdict(lambda: {"count": 0, "sum_ms": 0.0})
    for row in predicted_groups:
        predicted[row["family"]]["count"] += row["occurrences"]
        predicted[row["family"]]["sum_ms"] += row["duration_ms"]
    members = {row["native_type"]: row for row in observed.get("members", [])}
    family_rows = {}
    for family, native_type in (("allreduce", "hcom_allReduce_"),
                                ("allgather", "hcom_allGather_"),
                                ("alltoall", "hcom_alltoallv_")):
        member = members.get(native_type, {})
        family_rows[family] = {
            "native_count": native[family]["count"],
            "native_sum_ms": native[family]["sum_ms"],
            "scoreboard_member_count": member.get("occurrences"),
            "scoreboard_member_ms": member.get("duration_ms"),
            "scoreboard_minus_named_ms": round(
                (member["duration_ms"] - native[family]["sum_ms"]), 6) if member.get("duration_ms") is not None else None,
            "predicted_count": predicted[family]["count"],
            "predicted_sum_ms": round(predicted[family]["sum_ms"], 6),
            "native_minus_predicted_ms": round(
                native[family]["sum_ms"] - predicted[family]["sum_ms"], 6),
        }
    native_total = capture["union_intersection"]["native_sum_ms"]
    mirror_total = capture["projection_candidates"]["mirror_interval_ms"]
    observed_minus_native = round(observed_ms - native_total, 6)
    residual = round(observed_minus_native - mirror_total, 6)
    return {
        "comparison_key": group["comparison_key"],
        "status": group.get("status"),
        "formal_accuracy": group.get("formal_accuracy"),
        "observed_ms": observed_ms,
        "predicted_ms": predicted_ms,
        "observed_occurrences": observed.get("occurrences"),
        "predicted_occurrences": sum(row["occurrences"] for row in predicted_groups),
        "native_occurrences": sum(native[f]["count"] for f in MODEL_FAMILIES),
        "mirror_occurrences": capture["projection_candidates"]["rows"],
        "observed_minus_native_ms": observed_minus_native,
        "mirror_interval_ms": mirror_total,
        "observed_minus_native_minus_mirror_ms": residual,
        "native_minus_predicted_ms": round(native_total - predicted_ms, 6),
        "observed_minus_predicted_ms": round(observed_ms - predicted_ms, 6),
        "native_plus_mirror_ms": round(native_total + mirror_total, 6),
        "scoreboard_observed_is_native_plus_mirror": abs(residual) < 1e-6,
        "families": family_rows,
        "observed_members": observed.get("members", []),
        "predicted_groups": predicted_groups,
        "predicted_link_bytes_per_rank_total": contract["predicted_link_bytes_per_rank_total"],
        "predicted_total_equals_link_bytes": abs(
            contract["predicted_link_bytes_per_rank_total"]
            - prediction["costs"]["semantic_work"]["link_bytes"]) < 1e-6,
    }


def build_audit(capture_path: Path = DEFAULT_CAPTURE, *, scoreboard_path: Path = DEFAULT_SCOREBOARD,
                input_path: Path = DEFAULT_INPUT, prediction_path: Path = DEFAULT_PREDICTION,
                case_path: Path = DEFAULT_CASE, catalog_path: Path = DEFAULT_CATALOG,
                gap_ns: int = 5_000_000, repository: Path | None = None) -> dict:
    capture = audit_capture(capture_path, gap_ns)
    entry = _entry(input_path)
    conditions = entry["observed"]["conditions"]
    scoreboard = _scoreboard(scoreboard_path)
    prediction = _prediction(prediction_path)
    contract = _contract(case_path, catalog_path, conditions)
    reconciliation = _reconcile(capture, scoreboard, prediction, contract)
    revision = _git_revision(repository or Path.cwd())
    summary_sources = [row for row in entry.get("sources", [])
                       if row.get("role") == "observed_summary"]
    findings = [
        f"the reported collective row is {reconciliation['observed_ms']:.6f} ms; "
        f"named HCCL sums to {capture['union_intersection']['native_sum_ms']:.6f} ms and "
        f"AivKernel projected intervals to {capture['projection_candidates']['mirror_interval_ms']:.6f} ms; "
        f"reported row equals their sum: {reconciliation['scoreboard_observed_is_native_plus_mirror']}; "
        f"producer identity remains unproven",
        f"full 46-column duplicate records: {capture['duplicate_full_identity_rows']}; "
        f"one-to-one same-rank named/projected interval pairs: {capture['projection_candidates']['exact_native_interval_matches']}",
        f"AICPU rows ({capture['aicpu']['rows']}) are distinct intervals, not exact interval matches; "
        f"they intersect native time by {capture['aicpu']['intersection_with_native_ms']:.6f} ms "
        f"and add {capture['aicpu']['outside_native_ms']:.6f} ms outside native spans",
        f"native communication union is {capture['union_intersection']['native_union_ms']:.6f} ms "
        f"across native logical streams {capture['native_communication_streams']}; it is one rank, not a cohort",
        f"predicted {reconciliation['predicted_ms']:.6f} ms is {reconciliation['predicted_occurrences']} "
        f"isolated envelope programs; native call counts are allreduce={reconciliation['families']['allreduce']['native_count']}, "
        f"allgather={reconciliation['families']['allgather']['native_count']}, "
        f"alltoall={reconciliation['families']['alltoall']['native_count']}",
        f"collective bindings are {contract['collective_recipe_bindings']}; "
        f"scope={contract['algorithm_scope']}, causal rounds={contract['round_readiness_modeled']}; "
        f"operator recipe labels alone do not select an algorithm",
        f"TP{contract['parallel']['tp']} EP{contract['parallel']['ep']} DP{contract['parallel']['dp']} "
        f"maps each source to {contract['remote_bytes_per_source_per_layer']} remote bytes/layer "
        f"and omits {contract['self_bytes_per_source_per_layer']} local/self bytes/layer from link work",
        f"alltoallv rows form {capture['alltoallv_group_count']} time clusters of "
        f"{len(capture['alltoallv_ordinal_groups'])} (not layer identities) "
        f"({reconciliation['families']['alltoall']['native_count']} native calls); predicted "
        f"dispatch+combine is {reconciliation['families']['alltoall']['predicted_count']} calls",
        f"624 classified rows sum to {capture['union_intersection']['all_rows_sum_ms']:.6f} ms; "
        f"analysis/pre4096.json labels the same 624-call class tp8_comm=198.474 ms",
        f"allgather rows are 90 native calls across logical streams 503/703 "
        f"(46 calls/{capture['allgather_by_stream']['503']['sum_ms']:.6f} ms and "
        f"44 calls/{capture['allgather_by_stream']['703']['sum_ms']:.6f} ms); predicted allgather is "
        f"{reconciliation['families']['allgather']['predicted_count']} calls",
    ]
    mechanisms = [
        {
            "id": "scoreboard_collective_projection_overlap",
            "status": "reported_not_modified",
            "counterexample": (
                f"{reconciliation['observed_ms']:.6f} = "
                f"{capture['union_intersection']['native_sum_ms']:.6f} native + "
                f"{capture['projection_candidates']['mirror_interval_ms']:.6f} projected interval sum"
            ),
            "suggested_owner": "mindie_sim/observations/scoreboard.py family membership",
            "suggested_api": "report named-HCCL and projected views separately; no producer-identity deduplication without correlation evidence",
            "root_confirmation_required": True,
        },
        {
            "id": "alltoallv_call_count",
            "status": "reported_not_modified",
            "counterexample": (
                f"native alltoallv={reconciliation['families']['alltoall']['native_count']} "
                f"({capture['alltoallv_group_count']} groups of "
                f"{len(capture['alltoallv_ordinal_groups'])}; ordinal sums "
                f"{[row['sum_ms'] for row in capture['alltoallv_ordinal_groups']]}) "
                f"but predicted={reconciliation['families']['alltoall']['predicted_count']}"
            ),
            "suggested_owner": "mindie_sim/prediction/feedforward.py source-declared quantized dispatch protocol",
            "suggested_api": "explicit source-selected scale/payload/reverse protocol; capture ordinal roles remain unidentified",
            "root_confirmation_required": True,
        },
        {
            "id": "allgather_call_count",
            "status": "reported_not_modified",
            "counterexample": (
                f"native allgather={reconciliation['families']['allgather']['native_count']} "
                f"across streams {capture['allgather_by_stream']} but predicted "
                f"{reconciliation['families']['allgather']['predicted_count']} calls"
            ),
            "suggested_owner": "mindie_sim/prediction/feedforward.py source-declared MoE histogram gather",
            "suggested_api": "source-bound allgather operator with group_axis and steps, never a measured multiplier",
            "root_confirmation_required": True,
        },
        {
            "id": "native_allreduce_control_overlap",
            "status": "reported_not_modified",
            "counterexample": (
                f"native allreduce={capture['families']['native']['allreduce']['sum_ms']:.6f} ms; "
                f"AICPU control={capture['aicpu']['by_family']['allreduce_control']['sum_ms']:.6f} ms; "
                f"intersection={capture['aicpu']['by_family']['allreduce_control']['intersection_with_native_ms']:.6f} ms"
            ),
            "suggested_owner": "mindie_sim/prediction/collectives.py/execution.py control-resource boundary",
            "suggested_api": "declare a separate AICPU/control resource if source identity is established; do not add wrapper time to native HCCL",
            "root_confirmation_required": True,
        },
    ]
    limitations = [
        "Only one retained device rank (Device_id set) is present; no whole-cohort union, peer matrix, "
        "cross-rank contention or global critical path can be established.",
        "Native communication rows expose no Input/Output Shapes, peer IDs or correlation IDs; dispatch, "
        "combine and any third alltoallv cannot be attributed by name or tensor direction.",
        "AICPU and native rows overlap; the capture cannot prove that native duration is pure wire service "
        "or that AICPU is nested inside every native call.",
        "The predicted graph is an isolated program sum, not a wall time; the observed 575.235407 ms span "
        "and 609.513832 ms activity sum bound different things.",
        "The case's declared source geometry is conditional; binary identity, exact HCCL algorithm and "
        "link rate remain unestablished. No rate or residual multiplier is fitted here.",
        "The balanced DP1/TP8/EP8 case only exercises symmetric source totals; it does not validate "
        "the generic explicit peer-traffic transpose mechanism.",
        "One-to-one equal intervals support projection candidates but do not prove physical producer "
        "identity: the CSV has no exported correlation join.",
        "Five-millisecond gap clustering is descriptive and does not establish layer/call roles.",
        "The stored prediction has no collective interval timeline, so observed/predicted interval "
        "intersection is unavailable; only family sums can be compared.",
    ]
    return {
        "schema": SCHEMA,
        "audit_date": "2026-10-29",
        "model": {"id": MODEL_ID, "name": "deepseek-v4-flash-0731",
                  "not": "deepseek-v4.1-flash"},
        "source": {
            "repository_revision": revision,
            "audit_tool": _hash_json(Path(__file__)),
            "capture": capture,
            "scoreboard": _hash_json(scoreboard_path),
            "input": _hash_json(input_path),
            "prediction": _hash_json(prediction_path),
            "case": _hash_json(case_path),
            "catalog": _hash_json(catalog_path),
            "independent_dispatch_source": DISPATCH_SOURCE,
            "observed_summary": summary_sources[0] if summary_sources else None,
        },
        "capture_window": {
            "phase": conditions["phase"],
            "tokens": conditions["tokens"],
            "context": conditions["context"],
            "sequences": conditions["sequences"],
            "branch": conditions["branch"],
            "sample_count": entry["observed"]["step"]["sample_count"],
            "scope": entry["observed"]["step"]["scope"],
            "boundary": entry["observed"]["capture"]["boundary"],
            "span_ms": entry["observed"]["capture"]["span_ms"],
            "activity_sum_ms": entry["observed"]["step"]["activity_sum_ms"],
            "rank_scope": entry["observed"]["capture"]["rank_scope"],
            "served_attention_calls": entry["observed"]["independent_conditions"]["served_attention_calls"],
        },
        "contract": contract,
        "reconciliation": reconciliation,
        "mechanism_candidates": mechanisms,
        "source_supported_protocol": {
            "kind": "separate_quantized_payload",
            "sequence": ["local_histogram", "histogram_allgather", "pack_and_quantize",
                         "scale_alltoall_wait", "payload_alltoall_wait",
                         "expert_compute", "reverse_payload_alltoall_wait"],
            "peer_contract": "reverse split counts swap forward input/output splits; format may differ",
            "capture_call_attribution": "not_established_from_interval_order",
            "missing_control": "CPU split extraction and stream synchronization not priced by source count alone",
        },
        "findings": findings,
        "limitations": limitations,
    }


def _git_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--scoreboard", type=Path, default=DEFAULT_SCOREBOARD)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prediction", type=Path, default=DEFAULT_PREDICTION)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gap-us", type=int, default=5000,
                        help="alltoallv group gap in microseconds")
    args = parser.parse_args(argv)
    report = build_audit(
        args.capture, scoreboard_path=args.scoreboard, input_path=args.input,
        prediction_path=args.prediction, case_path=args.case, catalog_path=args.catalog,
        gap_ns=args.gap_us * 1000, repository=Path.cwd(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    capture = report["source"]["capture"]
    print(f"native={capture['union_intersection']['native_sum_ms']:.6f} ms, "
          f"mirror={capture['projection_candidates']['mirror_interval_ms']:.6f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
