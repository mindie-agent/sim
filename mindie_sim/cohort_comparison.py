"""Compare measured repeated-structure cohorts with independent sim programs.

[Role] Consume profiler cohort aggregates without requiring layer occurrence binding.
[Invariants] Tensor and physical identities must agree before pricing. Observed
  durations never select targets or enter prediction. Window multiplicity does
  not establish step count, and activity sums are not wall-clock latency.
[Timeouts & Locks] Bounded JSON inputs and one finite declared graph; no locks.
[Side-Effect Policy] The CLI reads local metadata and writes its requested report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
import math
from pathlib import Path

from .cohort_accuracy import accuracy_fields, accuracy_policy
from .prediction.graph_analysis import graph_costs
from .prediction.operator_analysis import operator_costs, target_graphs
from .prediction.program import RecipeBook
from .prediction.structure import parse_case


SCHEMA = "mindie-sim.cohort-comparison/1"
MAX_METADATA_BYTES = 32 * 1024 * 1024
MAX_COHORTS = 10_000


def _read(value):
    if isinstance(value, (str, Path)):
        with Path(value).open("rb") as stream:
            data = stream.read(MAX_METADATA_BYTES + 1)
        if len(data) > MAX_METADATA_BYTES:
            raise ValueError("cohort_metadata_exceeds_32_mib")
        value = json.loads(data)
    if not isinstance(value, Mapping):
        raise ValueError("cohort_metadata_requires_object")
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _member_ids(target, kind):
    return target["node_ids"] if kind == "subgraph" else [target["node_id"]]


def _check_placement(cohort, selected, structure):
    placement = cohort.get("placement", {})
    symmetric = placement.get("basis") == "single_rank_symmetric_dp"
    replicas = sorted({target.get("replica") for target in selected}, key=str)
    if not symmetric:
        if len(replicas) != 1 or replicas[0] != cohort.get("replica"):
            raise ValueError("cohort_target_placement_mismatch")
        return 1
    execution = structure["execution"]
    if (execution.get("replica_axis") != "data_parallel"
            or set(replicas) != set(execution.get("replicas", ()))
            or placement.get("replica_candidates") != replicas or len(replicas) < 2
            or cohort.get("replica") is not None
            or placement.get("replica_binding_implied") is not False
            or not cohort.get("rank_id") or not cohort.get("device_id")):
        raise ValueError("cohort_symmetric_dp_placement_invalid")
    local = {}
    for target in selected:
        key = target.get("subgraph_id", target.get("node_id"))
        prefix = f"rank:{target['replica']}/"
        if not key.startswith(prefix):
            raise ValueError("cohort_symmetric_dp_target_identity_invalid")
        local.setdefault(target["replica"], set()).add(
            (key[len(prefix):], target.get("layer"), tuple(target.get("layer_ids", ()))))
    if any(value != local[replicas[0]] for value in local.values()):
        raise ValueError("cohort_symmetric_dp_candidates_incomplete")
    if cohort.get("declared_candidate_instance_count") != len(selected):
        raise ValueError("cohort_candidate_instance_count_mismatch")
    return len(replicas)


def _check_cohort(cohort, window, structure, targets):
    if cohort.get("schema_version") != "mindie-profile-cohort/1":
        raise ValueError("cohort_schema_mismatch")
    if cohort.get("graph_id") != structure["graph_id"]:
        raise ValueError("cohort_graph_id_mismatch")
    ids = cohort.get("target_ids", ())
    if not ids or len(set(ids)) != len(ids) or any(key not in targets for key in ids):
        raise ValueError("cohort_target_ids_unknown_or_duplicated")
    if str(cohort.get("rank_id")) != str(window["rank_id"]):
        raise ValueError("cohort_capture_rank_mismatch")
    kind = cohort.get("target_kind")
    if kind not in {"node", "subgraph"}:
        raise ValueError("cohort_target_kind_invalid")
    selected = [targets[key] for key in ids]
    replica_count = _check_placement(cohort, selected, structure)
    if cohort.get("declared_instance_count") != len(ids) // replica_count:
        raise ValueError("cohort_declared_instance_count_mismatch")
    if _canonical(cohort.get("resource_boundary")) != _canonical(
            selected[0].get("resource_boundary")):
        raise ValueError("cohort_resource_boundary_mismatch")
    if _canonical(cohort.get("contract_scope")) != _canonical(
            selected[0].get("contract_scope")):
        raise ValueError("cohort_contract_scope_mismatch")
    placement = (cohort.get("replica"), cohort.get("stage"), cohort.get("phase"))
    seen = set()
    for target in selected:
        if target.get("event_matchable") is False:
            raise ValueError("cohort_target_is_internal_to_selected_native_fusion")
        if ("subgraph_id" in target) != (kind == "subgraph"):
            raise ValueError("cohort_target_boundaries_differ")
        if tuple(target.get(key) for key in ("stage", "phase")) != placement[1:]:
            raise ValueError("cohort_target_placement_mismatch")
        if (not cohort.get("physical_signature")
                or target.get("physical_signature") != cohort["physical_signature"]
                or _canonical(target.get("physical_work")) !=
                _canonical(cohort.get("physical_work_per_instance"))
                or target.get("contract") is None
                or _canonical(target["contract"]) != _canonical(cohort.get("contract"))):
            raise ValueError("cohort_physical_or_tensor_contract_mismatch")
        if isinstance(target.get("coverage"), Mapping) and target["coverage"].get("kind") not in {
            "complete_subgraph", "complete_semantic_subgraph",
        }:
            raise ValueError("cohort_incomplete_native_boundary")
        members = set(_member_ids(target, kind))
        if members & seen:
            raise ValueError("cohort_candidate_boundaries_overlap")
        seen.update(members)
    if (replica_count == 1 and placement[0] is None
            or placement[1] is None or placement[2] != structure["execution"]["phase"]):
        raise ValueError("cohort_placement_or_phase_unspecified")
    count, complete, clipped = (cohort.get(key) for key in (
        "event_count", "complete_event_count", "clipped_event_count"))
    if (any(not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (count, complete, clipped)) or count != complete + clipped):
        raise ValueError("cohort_event_counts_inconsistent")
    return selected


def _check_symmetric_hardware(case, cohort):
    placement = cohort.get("placement", {})
    if placement.get("basis") != "single_rank_symmetric_dp":
        return
    if cohort["physical_work_per_instance"].get("link_bytes"):
        raise ValueError("symmetric_dp_transport_requires_explicit_replica")
    parallel = case.runtime.parallel
    width = parallel.tp * parallel.pcp
    declarations = []
    for replica in placement["replica_candidates"]:
        start = (replica * parallel.pp + cohort["stage"]) * width
        declarations.append([
            {key: value for key, value in die.to_dict().items()
             if key not in {"rank", "node", "package"}}
            for die in case.machine.dies[start:start + width]
        ])
    if any(value != declarations[0] for value in declarations[1:]):
        raise ValueError("symmetric_dp_hardware_differs")


def _program_ids(target, kind):
    boundary = target.get("resource_boundary") if kind == "subgraph" else None
    if boundary:
        identifiers = boundary.get("operator_ids", ())
        if not identifiers:
            raise ValueError("cohort_resource_boundary_has_no_operators")
        prefix = f"rank:{target['replica']}/"
        return [key if key.startswith(prefix) else prefix + key for key in identifiers], boundary
    return _member_ids(target, kind), None


def _predict(case, target, kind, by_id, book):
    identifiers, selected_boundary = _program_ids(target, kind)
    if any(key not in by_id for key in identifiers):
        raise ValueError("cohort_target_physical_program_unavailable")
    ops = tuple(by_id[key] for key in identifiers)
    replica = target["replica"]
    physical_work = target["physical_work"]
    one_program = (len(ops) == 1 and physical_work.get("launches") == 1
                   and ops[0].source_boundary != "call")
    if kind == "node" and len(ops) == 1:
        result = operator_costs(case, ops, recipes=book, replica=replica)
        if result["operator_occurrences"] != 1:
            raise ValueError("cohort_node_expands_to_multiple_device_programs")
        row = result["operators"][0]
        duration, serviced = row["duration_ms"], row["serviced"]
        parameters = row["basis"]
    else:
        local = {op.op_id for op in ops}
        external = {dep: 0 for op in ops for dep in op.deps if dep not in local}
        result = graph_costs(
            case, {replica: ops}, recipes=book, external_inputs=external,
            include_submission=False, isolated=False, trace_limit=0)
        duration = result["duration_ms"]
        serviced = {item["resource"]: item["work"] for item in result["resources"]}
        parameters = result["parameters"]
    return {
        "boundary": "semantic_device_program" if one_program else "unfused_subgraph_reference",
        "representative_target_id": target.get("subgraph_id", target.get("node_id")),
        "representative_operator_ids": identifiers,
        "representative_selection_basis": "equivalent_declared_program; not_capture_occurrence_binding",
        "per_call_ms": duration, "physical_work_per_instance": physical_work,
        "serviced": serviced, "parameters": parameters,
        "resource_boundary": selected_boundary,
        "single_device_program": one_program,
        "assumptions": [
            "External inputs are ready; no host submission or unrelated graph contention is simulated.",
            "The case supplies work, precision, layout and resource bindings independently of measurements.",
            "Unobserved fields excluded by the tensor contract remain declared conditions.",
        ],
    }


def _error(predicted, observed):
    return {"signed_residual_ms": predicted - observed,
            "absolute_residual_ms": abs(predicted - observed),
            "relative_error": (predicted / observed - 1) if observed > 0 else None}


def _comparison(cohort, prediction):
    scope = cohort.get("contract_scope")
    coverage = scope.get("prediction_coverage", {}) if isinstance(scope, Mapping) else {}
    if coverage.get("kind") == "partial_program":
        return "reference_only", coverage["reason"], None
    if not prediction["single_device_program"]:
        return "reference_only", "unfused_subgraph_is_not_the_observed_native_kernel", None
    if cohort["clipped_event_count"]:
        return "unscored", "window_contains_clipped_events", None
    if not cohort["complete_event_count"]:
        return "unscored", "no_complete_events", None
    duration = cohort.get("duration_ns", {})
    actual = duration.get("mean")
    if actual is None or not math.isfinite(actual) or actual <= 0:
        return "unscored", "complete_event_mean_unavailable", None
    forecast = prediction["per_call_ms"]
    observed_sum = cohort["activity_sum_ns"] / 1_000_000
    comparison = {
        "per_call_mean": _error(forecast, actual / 1_000_000),
        "activity_sum": _error(forecast * cohort["event_count"], observed_sum),
        "interpretation": "conditional_descriptive_comparison; no_occurrence_or_accuracy_certification",
    }
    if duration.get("p50") is not None:
        comparison["per_call_p50"] = _error(forecast, duration["p50"] / 1_000_000)
    return "compared", "same_declared_single_program_boundary", comparison


def compare_cohorts(case, mapping_report):
    """Price each homogeneous candidate set once without choosing a layer.

    The report is produced by ``mindie_profiler.structure_mapping``. The case
    must independently regenerate the report's complete structure identity.
    No profiler import or raw capture is required by this consumer.
    """
    report = _read(mapping_report)
    if "structure_mapping" in report:
        report = _read(report["structure_mapping"])
    if report.get("schema_version") != "mindie-profile-structure-mapping/1":
        raise ValueError("expected_profiler_structure_mapping_report")
    case = parse_case(_read(case) if isinstance(case, (str, Path, Mapping)) else case)
    execution = report.get("declared_execution", {})
    required = ("phase", "tokens", "context", "sequences", "scope")
    if any(key not in execution for key in required):
        raise ValueError("cohort_report_lacks_execution_geometry")
    geometry = {key: execution[key] for key in required}
    if "apply_body_defaults" in execution:
        geometry["apply_body_defaults"] = execution["apply_body_defaults"]
    from .structure_selection import select_structure_targets
    wanted = {key for window in report.get("windows", ())
              for cohort in window.get("cohorts", ()) for key in cohort.get("target_ids", ())}
    manifest = select_structure_targets(
        case, wanted, **geometry, expand_calls=execution.get("expand_calls", True))
    if manifest["graph_id"] != report.get("graph_id"):
        raise ValueError("cohort_case_graph_id_mismatch:regenerate_mapping_for_this_case")
    if report.get("observed_phase") not in (None, geometry["phase"]):
        raise ValueError("cohort_observed_phase_mismatch")
    targets = manifest["targets"]
    # Validate every candidate before dropping metadata. Keeping the full export
    # beside a second full executable DAG multiplied memory by the cohort size.
    checked, needed = {}, set()
    for window in report.get("windows", ()):
        for cohort in window.get("cohorts", ()):
            if len(checked) >= MAX_COHORTS:
                raise ValueError("cohort_comparison_exceeds_10000_groups")
            selected = _check_cohort(cohort, window, manifest, targets)
            checked[id(cohort)] = selected[0]
            identifiers, _ = _program_ids(selected[0], cohort["target_kind"])
            needed.update(identifiers)
    manifest = {key: manifest[key] for key in ("graph_id", "execution")}
    del targets
    selected = None
    graphs = target_graphs(case, **geometry)
    from .prediction.body_defaults import consume_target_declaration
    case = consume_target_declaration(case, graphs)
    book = RecipeBook.from_mapping(case.runtime.kernels)
    by_id = {f"rank:{replica}/{op.op_id}": op
             for replica, operators in graphs.items() for op in book.expand(operators)
             if f"rank:{replica}/{op.op_id}" in needed}
    del graphs
    cache, windows, counts = {}, [], Counter()
    total = 0
    for window in report.get("windows", ()):
        comparisons = []
        for cohort in window.get("cohorts", ()):
            total += 1
            if total > MAX_COHORTS:
                raise ValueError("cohort_comparison_exceeds_10000_groups")
            selected = (checked[id(cohort)],)
            key = (cohort["physical_signature"], cohort["replica"], cohort["stage"],
                   cohort["target_kind"])
            row = {
                "cohort_id": cohort["cohort_id"], "target_ids": cohort["target_ids"],
                "target_kind": cohort["target_kind"], "native_operator_type": cohort["native_operator_type"],
                "physical_signature": cohort["physical_signature"],
                "replica": cohort["replica"], "stage": cohort["stage"], "phase": cohort["phase"],
                "rank_id": cohort["rank_id"], "device_id": cohort.get("device_id"),
                "contract": cohort["contract"], "contract_scope": cohort.get("contract_scope"),
                "resource_boundary": cohort.get("resource_boundary"),
                "structural_classification": cohort.get("structural_classification"),
                "declared_instance_count": cohort["declared_instance_count"],
                "declared_candidate_instance_count": cohort.get(
                    "declared_candidate_instance_count", len(cohort["target_ids"])),
                "declared_layer_count": cohort["declared_layer_count"],
                "placement": cohort.get("placement"),
                "observed": {key: cohort.get(key) for key in (
                    "event_count", "complete_event_count", "clipped_event_count", "duration_ns",
                    "activity_sum_ns", "complete_activity_sum_ns", "busy_union_ns", "status_counts")},
                "multiplicity": dict(cohort["multiplicity"]), "occurrence_binding_implied": False,
                "prediction": None, "comparison": None,
            }
            try:
                _check_symmetric_hardware(case, cohort)
                if key not in cache:
                    cache[key] = _predict(case, selected[0], cohort["target_kind"], by_id, book)
                prediction = cache[key]
            except ValueError as exc:
                row.update(status="prediction_failed", reason=str(exc))
            else:
                # Equal physical programs share a cost, not an occurrence identity.
                # Host evidence can narrow equal-work cohorts to disjoint layers.
                identifiers, boundary = _program_ids(selected[0], cohort["target_kind"])
                prediction = {
                    **prediction,
                    "representative_target_id": selected[0].get(
                        "subgraph_id", selected[0].get("node_id")),
                    "representative_operator_ids": identifiers,
                    "resource_boundary": boundary,
                }
                scope = cohort.get("contract_scope")
                coverage = scope.get("prediction_coverage", {}) if isinstance(scope, Mapping) else {}
                if coverage.get("kind") == "partial_program":
                    prediction = {
                        **prediction, "boundary": "partial_program_reference",
                        "prediction_coverage": coverage,
                    }
                per_call = prediction["per_call_ms"]
                row["prediction"] = {
                    **prediction,
                    "observed_call_count_total_ms": per_call * cohort["event_count"],
                    "per_declared_step_ms": per_call * cohort["declared_instance_count"],
                    "step_total_basis": "one_invocation_per_declared_instance; not_observed_step_count",
                    "instance_scope": cohort["multiplicity"].get(
                        "instance_scope", "single_declared_replica"),
                }
                steps = row["multiplicity"].get("steps_in_window")
                if steps is not None:
                    expected = steps * cohort["declared_instance_count"]
                    row["multiplicity"].update(
                        expected_calls=expected, call_count_difference=cohort["event_count"] - expected)
                status, reason, comparison = _comparison(cohort, prediction)
                row.update(status=status, reason=reason, comparison=comparison)
            row.update(accuracy_fields(row))
            counts[row["status"]] += 1
            comparisons.append(row)
        windows.append({
            "rank_id": window["rank_id"], "step": window.get("step"),
            "source_identity": window.get("source_identity"),
            "window": window["window"], "rank_scope": window.get("rank_scope"),
            "cohort_excluded_event_counts": window.get("cohort_excluded_event_counts", {}),
            "cohorts": comparisons,
        })
    return {
        "schema_version": SCHEMA, "graph_id": manifest["graph_id"],
        "accuracy_policy": accuracy_policy(),
        "declared_execution": manifest["execution"],
        "hardware_declaration": case.machine.to_dict(),
        "declared_kernel_bindings": dict(case.runtime.kernels),
        "structure_source": report.get("structure_source"),
        "mapping_coverage": {key: report.get(key) for key in (
            "event_window_count", "status_counts", "cohort_event_window_count",
            "cohort_status", "cohort_event_fraction")},
        "status_counts": dict(counts), "distinct_priced_programs": len(cache), "windows": windows,
        "semantics": {
            "join": "same_tensor_contract_and_physical_work_candidate_set; no_layer_choice",
            "multiplicity": "window_counts_are_not_steps; repeated_windows_are_not_independent_runs",
            "timing": "per_call_device_program_and_activity_sum; never_wall_time_or_busy_union_error",
            "subgraphs": "unfused_multiple_programs_are_references_until_a_fused_resource_boundary_is_selected",
            "aggregation": "no_cross_rank_window_or_overlapping_declaration_totals",
            "calibration": "comparator_does_not_fit_parameters;explicit_observable_service_calibration_is_supported",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", type=Path)
    parser.add_argument("mapping", type=Path)
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args(argv)
    result = compare_cohorts(args.case, args.mapping)
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
