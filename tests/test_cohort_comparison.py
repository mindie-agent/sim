"""Retained native rows flow through the real profiler cohort producer into sim."""
from copy import deepcopy
from dataclasses import replace
from itertools import islice
import json
from pathlib import Path

import pytest

from mindie_sim.cohort_accuracy import accuracy_fields, accuracy_policy
from mindie_sim.cohort_comparison import compare_cohorts
from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.structure import export_op_structure, export_structure
from test_prediction_system import case as small_case


ROOT = Path(__file__).resolve().parents[1]


def _captured_report(case, capture, kind, *, phase="decode", tokens=1, repeat=2,
                     clip=False, rank_id="rank_0", structure=None, host_targets=None,
                     source_row=0, context=0):
    import mindie_profiler
    from mindie_profiler.events import Observation, QueryStats, SourceRef, Window
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.structure_mapping import map_structure

    path = Path(mindie_profiler.__file__).resolve().parents[1] / "tests/fixtures/fusion_contracts" / capture
    original = next(islice(
        (row.event for row in kernel_rows(path, rank_id=rank_id) if row.kind == kind),
        source_row, None))
    # Controlled repeats retain the original descriptor; they test aggregation,
    # not an asserted historical step count or collected per-layer annotation.
    events = [replace(original, start_ns=index * 10_000_000,
                      end_ns=index * 10_000_000 + original.duration_ns * (index + 1),
                      task_id=str(index + 1),
                      locator={**original.locator, "csv_line": index + 2})
              for index in range(repeat)]
    window = Window(1 if clip else 0, events[-1].end_ns, step=19)
    stat = path.stat()
    observation = Observation(
        rank_id, 0, window, None, events, SourceRef(
            rank_id=rank_id, rank_index=0, kind="csv", path=str(path), basename=path.name,
            sha256=None, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
            schema_version="retained_capture", schema_variant="kernel_details",
        ), QueryStats(rank_id, "controlled_repetitions"),
    )
    if structure is None:
        structure = export_structure(case, phase=phase, tokens=tokens, context=context, scope="body")
    options = {}
    if host_targets is not None:
        assert len(host_targets) == len(events)
        scopes, calls = [], []
        for index, (event, target) in enumerate(zip(events, host_targets)):
            name = f"controlled.host.scope{index}"
            scopes.append({"name": name, "node_ids": [target]})
            calls.append({
                "device_task": {
                    "task_id": event.task_id, "stream_id": event.stream_id,
                    "device_id": event.device_id, "start_ns": event.start_ns,
                    "end_ns": event.end_ns,
                    "global_task_id": event.locator.get("task_global_id"),
                },
                "global_task_id_unique": True,
                "association_basis": {"pytorch": "connection_ids"},
                "association_unique": {"pytorch": True},
                "host_stack": [{"name": name, "table": "PYTORCH_API"}],
            })
        options = {
            "bindings": {
                "schema_version": "mindie-profile-bindings/1",
                "graph_id": structure["graph_id"], "host_scopes": scopes,
            },
            "correlations": [{
                "schema_version": "mindie-profiler-correlation/1", "rank_id": rank_id,
                "source_identity": {
                    "path": observation.source.path,
                    "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                },
                "correlated_calls": calls,
            }],
        }
    return map_structure([observation], structure, mapping_limit=0, **options), events


def _norm_case():
    case = small_case()
    case["model"].update(hidden_size=6144, num_heads=48, num_kv_heads=8, head_dim=128)
    return case


@pytest.mark.parametrize("observed_us,predicted_us,tier,passed", [
    (50, 65, "absolute_15us", True),
    (50, 65.001, "absolute_15us", False),
    (50, 35, "absolute_15us", True),
    (50, 34.999, "absolute_15us", False),
    (50.001, 65.001, "relative_15pct", False),
    (100, 115, "relative_15pct", True),
    (100, 85, "relative_15pct", True),
    (100, 115.001, "relative_15pct", False),
    (100, 84.999, "relative_15pct", False),
    (200, 218, "relative_15pct", True),
    (10, 24, "absolute_15us", True),
    (10, 0, "absolute_15us", True),
    (50, 0, "absolute_15us", False),
])
def test_accuracy_uses_observed_mean_tier_and_inclusive_tolerance_without_a_floor(
        observed_us, predicted_us, tier, passed):
    row = {
        "status": "compared",
        "observed": {"duration_ns": {"mean": observed_us * 1000}},
        "prediction": {"per_call_ms": predicted_us / 1000},
    }
    original = deepcopy(row)
    fields = accuracy_fields(row)
    assert fields["accuracy_tier"] == tier
    assert fields["accuracy_pass"] is passed
    assert fields["accuracy_status"] == (
        "outside_tolerance" if not passed else
        "within_relative_tolerance" if tier == "relative_15pct" else
        "within_launch_floor_tolerance")
    assert fields["accuracy_tolerance"] == (
        {"unit": "ratio", "value": 0.15} if tier == "relative_15pct"
        else {"unit": "us", "value": 15.0})
    assert fields["observed_per_call_us"] == observed_us
    assert fields["predicted_per_call_us"] == pytest.approx(predicted_us)
    assert fields["per_call_diff_us"] == pytest.approx(predicted_us - observed_us)
    assert fields["abs_error_per_call_us"] == pytest.approx(abs(predicted_us - observed_us))
    assert row == original


@pytest.mark.parametrize("status", ["reference_only", "unscored"])
def test_accuracy_excludes_ineligible_boundaries_even_when_times_are_equal(status):
    row = {
        "status": status,
        "observed": {"duration_ns": {"mean": 100_000}},
        "prediction": {"per_call_ms": 0.1},
    }
    fields = accuracy_fields(row)
    assert fields["accuracy_tier"] == status
    assert fields["accuracy_status"] == status
    assert fields["accuracy_pass"] is None and fields["accuracy_tolerance"] is None
    assert fields["observed_per_call_us"] == fields["predicted_per_call_us"] == 100
    assert fields["per_call_diff_us"] == 0
    assert fields["abs_error_per_call_us"] == 0


def test_repeated_real_norms_compare_without_choosing_layers_or_inventing_steps(profiler_api):
    case = _norm_case()
    mapping, events = _captured_report(case, "a3_glm53_decode.csv", "RmsNorm")
    assert mapping["status_counts"] == {"ambiguous": 2}
    assert mapping["event_mappings"] == []  # Cohorts do not depend on the retained sample budget.
    observed = mapping["windows"][0]["cohorts"]
    assert len(observed) == 1 and observed[0]["declared_instance_count"] > 1
    result = compare_cohorts(case, {
        "schema_version": "mindie-profile-summary/1", "structure_mapping": mapping,
    })
    row = result["windows"][0]["cohorts"][0]
    assert row["status"] == "compared"
    assert not row["occurrence_binding_implied"]
    assert row["multiplicity"]["steps_in_window"] is None
    assert "expected_calls" not in row["multiplicity"]
    assert row["observed"]["activity_sum_ns"] == sum(event.duration_ns for event in events)
    predicted = row["prediction"]["per_call_ms"]
    assert row["prediction"]["observed_call_count_total_ms"] == predicted * 2
    assert row["prediction"]["per_declared_step_ms"] == predicted * row["declared_instance_count"]
    assert row["comparison"]["activity_sum"]["signed_residual_ms"] == pytest.approx(
        predicted * 2 - sum(event.duration_ns for event in events) / 1_000_000)
    assert row["observed"]["status_counts"] == {"ambiguous": 2}
    assert result["windows"][0]["source_identity"] == mapping["windows"][0]["source_identity"]
    assert result["declared_execution"] == mapping["declared_execution"]
    assert row["contract_scope"] == observed[0]["contract_scope"]
    assert result["accuracy_policy"] == accuracy_policy()
    assert row["observed_per_call_us"] == observed[0]["duration_ns"]["mean"] / 1000
    assert row["predicted_per_call_us"] == predicted * 1000
    assert row["per_call_diff_us"] == row["predicted_per_call_us"] - row["observed_per_call_us"]
    assert row["abs_error_per_call_us"] == abs(row["per_call_diff_us"])
    saved = json.loads(json.dumps(row))
    original = deepcopy(saved)
    saved.update(accuracy_tier="outdated_policy", accuracy_pass=False)
    refreshed = accuracy_fields(saved)
    assert saved["observed"] == original["observed"]
    assert saved["prediction"] == original["prediction"]
    assert saved["comparison"] == original["comparison"]
    assert refreshed == {key: row[key] for key in refreshed}


def test_single_capture_rank_folds_symmetric_dp_without_multiplying_instances(profiler_api):
    case = small_case(dp=4, ep=4)
    case["model"].update(hidden_size=6144, num_heads=48, num_kv_heads=8, head_dim=128,
                         num_experts=4, top_k=1, moe_intermediate_size=64)
    mapping, events = _captured_report(case, "a3_glm53_decode.csv", "RmsNorm")
    assert mapping["status_counts"] == {"ambiguous": 2}
    group = mapping["windows"][0]["cohorts"][0]
    assert group["replica"] is None
    assert group["placement"]["replica_candidates"] == [0, 1, 2, 3]
    assert group["placement"]["basis"] == "single_rank_symmetric_dp"
    assert group["declared_candidate_instance_count"] == len(group["target_ids"])
    assert group["declared_instance_count"] * 4 == len(group["target_ids"])
    assert group["multiplicity"]["calls_per_declared_instance"] == (
        len(events) / group["declared_instance_count"])
    assert group["multiplicity"]["steps_in_window"] is None
    result = compare_cohorts(case, mapping)
    row = result["windows"][0]["cohorts"][0]
    assert row["status"] == "compared" and row["replica"] is None
    assert row["observed"]["activity_sum_ns"] == sum(event.duration_ns for event in events)
    assert row["prediction"]["observed_call_count_total_ms"] == row["prediction"]["per_call_ms"] * 2
    assert row["prediction"]["per_declared_step_ms"] == (
        row["prediction"]["per_call_ms"] * group["declared_instance_count"])
    assert not row["occurrence_binding_implied"]
    wrong = deepcopy(mapping)
    wrong["windows"][0]["cohorts"][0]["declared_instance_count"] *= 4
    with pytest.raises(ValueError, match="cohort_declared_instance_count_mismatch"):
        compare_cohorts(case, wrong)
    from mindie_sim.prediction.structure import parse_case

    parsed = parse_case(case)
    slower = replace(parsed.machine.dies[-1], rates=tuple(
        (key, rate / 2) for key, rate in parsed.machine.dies[-1].rates))
    heterogeneous = replace(parsed, machine=replace(
        parsed.machine, dies=(*parsed.machine.dies[:-1], slower)))
    failed = compare_cohorts(heterogeneous, mapping)["windows"][0]["cohorts"][0]
    assert failed["status"] == "prediction_failed"
    assert failed["reason"] == "symmetric_dp_hardware_differs"
    assert failed["observed"]["event_count"] == len(events)
    assert failed["accuracy_tier"] == "unscored"
    assert failed["accuracy_status"] == "unscored"
    assert failed["accuracy_pass"] is None and failed["predicted_per_call_us"] is None
    assert failed["per_call_diff_us"] is None and failed["abs_error_per_call_us"] is None


def test_real_kimi_decode_matrix_crosses_semantic_recipes_without_choosing_a_replica(profiler_api):
    case = json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text())
    mapping, events = _captured_report(
        case, "kimi_matmul_decode.csv", "MatMulV3", source_row=1, context=32768)
    group = mapping["windows"][0]["cohorts"][0]
    assert mapping["status_counts"] == {"ambiguous": 2}
    assert group["declared_candidate_instance_count"] == group["declared_instance_count"] * 4
    recipes = {node["recipe"] for node in export_structure(
        case, phase="decode", tokens=1, context=32768, scope="body")["nodes"]
        if node["node_id"] in group["target_ids"]}
    assert {"linear_q", "output_gate", "dense_gate_up"} <= recipes
    result = compare_cohorts(case, mapping)
    row = result["windows"][0]["cohorts"][0]
    assert row["status"] == "compared"
    assert row["replica"] is None and row["placement"]["replica_candidates"] == [0, 1, 2, 3]
    assert row["observed"]["activity_sum_ns"] == sum(event.duration_ns for event in events)
    assert row["prediction"]["per_call_ms"] > 0
    assert row["prediction"]["observed_call_count_total_ms"] == row["prediction"]["per_call_ms"] * 2
    assert mapping["windows"][0]["nodes"] == mapping["windows"][0]["layers"] == []


def test_clipped_cohort_keeps_complete_statistics_but_has_no_error_score(profiler_api):
    case = _norm_case()
    mapping, _ = _captured_report(case, "a3_glm53_decode.csv", "RmsNorm", clip=True)
    result = compare_cohorts(case, mapping)
    row = result["windows"][0]["cohorts"][0]
    assert row["observed"]["complete_event_count"] == 1
    assert row["observed"]["clipped_event_count"] == 1
    assert row["prediction"]["per_call_ms"] > 0
    assert row["status"] == "unscored" and row["comparison"] is None
    assert row["reason"] == "window_contains_clipped_events"
    assert row["accuracy_tier"] == "unscored"
    assert row["accuracy_status"] == "unscored"
    assert row["accuracy_pass"] is None and row["accuracy_tolerance"] is None
    assert row["per_call_diff_us"] == (
        row["prediction"]["per_call_ms"] * 1000 - row["observed"]["duration_ns"]["mean"] / 1000)


def test_w4a8_real_grouped_cohort_retains_totals_but_not_full_program_error(profiler_api):
    case = json.loads((ROOT / "examples/cases/glm53-a3-w4a8.json").read_text())
    mapping, events = _captured_report(
        case, "a3_glm53_prefill.csv", "GroupedMatmul", phase="prefill", tokens=4096)
    assert mapping["status_counts"] == {"ambiguous": 2}
    cohorts = mapping["windows"][0]["cohorts"]
    assert len(cohorts) == 1 and cohorts[0]["declared_instance_count"] == 75
    result = compare_cohorts(case, mapping)
    row = result["windows"][0]["cohorts"][0]
    assert row["status"] == "reference_only"
    assert row["reason"] == "quantized_epilogue_unmodeled"
    assert row["comparison"] is None
    assert row["prediction"]["boundary"] == "partial_program_reference"
    assert row["prediction"]["per_call_ms"] > 0
    assert row["observed"]["activity_sum_ns"] == sum(event.duration_ns for event in events)
    assert row["observed"]["duration_ns"]["mean"] == sum(event.duration_ns for event in events) / 2
    assert row["observed"]["status_counts"] == {"ambiguous": 2}
    assert not row["occurrence_binding_implied"]
    assert row["accuracy_tier"] == "reference_only"
    assert row["accuracy_status"] == "reference_only"
    assert row["accuracy_pass"] is None and row["accuracy_tolerance"] is None
    assert row["observed_per_call_us"] > 0 and row["predicted_per_call_us"] > 0


def test_source_declared_w8a8_epilogue_still_compares_real_dense_cohort(profiler_api):
    case = json.loads((ROOT / "examples/cases/glm53-a3-w4a8.json").read_text())
    mapping, _ = _captured_report(
        case, "a3_glm53_prefill.csv", "QuantBatchMatmulV3", phase="prefill", tokens=4096)
    result = compare_cohorts(case, mapping)
    row = result["windows"][0]["cohorts"][0]
    assert row["status"] == "compared"
    assert row["comparison"]["per_call_mean"] is not None
    assert row["prediction"]["physical_work_per_instance"]["vector_ops"] > 0
    assert "prediction_coverage" not in row["contract_scope"]


def test_declared_case_identity_and_physical_work_cannot_be_replaced_by_matching_shapes(profiler_api):
    case = _norm_case()
    mapping, _ = _captured_report(case, "a3_glm53_decode.csv", "RmsNorm")
    changed = deepcopy(case)
    changed["model"]["layers"] += 1
    with pytest.raises(ValueError, match="cohort_case_graph_id_mismatch"):
        compare_cohorts(changed, mapping)
    corrupt = deepcopy(mapping)
    corrupt["windows"][0]["cohorts"][0]["physical_work_per_instance"]["vector_ops"] += 1
    with pytest.raises(ValueError, match="cohort_physical_or_tensor_contract_mismatch"):
        compare_cohorts(case, corrupt)


def test_same_shape_different_rank_windows_are_not_summed_as_elapsed_time(profiler_api):
    case = _norm_case()
    mapping, _ = _captured_report(case, "a3_glm53_decode.csv", "RmsNorm")
    other, _ = _captured_report(case, "a3_glm53_decode.csv", "RmsNorm", rank_id="rank_1")
    mapping["windows"].append(other["windows"][0])
    result = compare_cohorts(case, mapping)
    assert len(result["windows"]) == 2
    assert result["distinct_priced_programs"] == 1
    assert result["status_counts"] == {"compared": 2}
    assert "duration_ms" not in result and "activity_sum_ms" not in result


def test_cached_cost_keeps_each_host_bound_cohorts_own_representative(profiler_api):
    case = _norm_case()
    declared = export_structure(case, phase="decode", tokens=1, scope="body")
    first = next(node for node in declared["nodes"] if node["family"] == "rmsnorm")
    second = next(node for node in declared["nodes"]
                  if node["layer"] != first["layer"]
                  and node["physical_signature"] == first["physical_signature"])
    targets = [first["node_id"], second["node_id"]]
    mapping, _ = _captured_report(
        case, "a3_glm53_decode.csv", "RmsNorm",
        structure=declared, host_targets=targets)
    assert mapping["status_counts"] == {"bound": 2}
    result = compare_cohorts(case, mapping)
    rows = result["windows"][0]["cohorts"]
    assert len(rows) == 2 and result["distinct_priced_programs"] == 1
    assert rows[0]["prediction"]["per_call_ms"] == rows[1]["prediction"]["per_call_ms"]
    for row, target in zip(rows, targets):
        assert row["target_ids"] == [target]
        assert row["prediction"]["representative_target_id"] == target
        assert row["prediction"]["representative_operator_ids"] == [target]
        assert row["prediction"]["resource_boundary"] == row["resource_boundary"]


def test_real_hyper_connection_is_only_an_unfused_reference_until_resource_selection(profiler_api):
    case = json.loads((ROOT / "examples/cases/v4f-a3-w8a8-tp8.json").read_text())
    mapping, _ = _captured_report(
        case, "a3_v4f.csv", "HcPre", phase="prefill", tokens=4096, repeat=1)
    assert mapping["windows"][0]["cohorts"]
    result = compare_cohorts(case, mapping)
    row = result["windows"][0]["cohorts"][0]
    assert row["target_kind"] == "subgraph"
    assert row["status"] == "reference_only"
    assert row["comparison"] is None
    assert row["prediction"]["boundary"] == "unfused_subgraph_reference"
    assert row["prediction"]["physical_work_per_instance"]["launches"] > 1
    assert row["prediction"]["per_call_ms"] > 0


def test_selected_native_fusion_prices_the_physical_program_without_binding_a_layer(profiler_api):
    case = json.loads((ROOT / "examples/cases/v4f-a3-w8a8-tp8.json").read_text())
    case["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})["native_fusion"] = {
        "implementation": "declared_native_fusion",
        "options": {"roles": ["hc_pre"], "traffic_profile": "hc_pre_m_k_split_175ef4d4"},
    }
    mapping, _ = _captured_report(
        case, "a3_v4f.csv", "HcPre", phase="prefill", tokens=4096)
    assert mapping["status_counts"] == {"ambiguous": 2}
    assert mapping["windows"][0]["cohorts"]
    result = compare_cohorts(case, mapping)
    row = result["windows"][0]["cohorts"][0]
    assert row["status"] == "compared"
    assert row["target_kind"] == "subgraph" and row["declared_instance_count"] > 1
    assert row["prediction"]["boundary"] == "semantic_device_program"
    assert row["prediction"]["physical_work_per_instance"]["launches"] == 1
    assert row["prediction"]["resource_boundary"]["kind"] == "selected_native_fusion"
    assert row["prediction"]["per_call_ms"] > 0
    assert not row["occurrence_binding_implied"]
    declared = export_structure(case, phase="prefill", tokens=4096, scope="body")
    target = next(group for group in declared["subgraphs"]
                  if group["subgraph_id"] == row["prediction"]["representative_target_id"])
    assert len(target["node_ids"]) > 1
    assert row["prediction"]["representative_operator_ids"] == target["resource_boundary"]["operator_ids"]
    internal = [node for node in declared["nodes"] if node["node_id"] in target["node_ids"]]
    assert internal and all(node["event_matchable"] is False for node in internal)
    assert set(target["node_ids"]).isdisjoint(row["prediction"]["representative_operator_ids"])
    assert row["prediction"]["physical_work_per_instance"]["hbm_bytes"] < sum(
        node["physical_work"]["hbm_bytes"] for node in internal)


def test_equal_tensor_work_with_different_explicit_recipes_is_not_a_cohort(profiler_api):
    case = _norm_case()
    original = next(op for op in target_graphs(
        case, phase="decode", tokens=1, scope="body")[0] if op.family == "rmsnorm")
    operators = [
        replace(original, op_id="norm_a", deps=(), layer=0, recipe="envelope_norm"),
        replace(original, op_id="norm_b", deps=(), layer=1, recipe="tiled_norm"),
    ]
    kernels = {"recipes": {
        "envelope_norm": {"kind": "envelope"},
        "tiled_norm": {"kind": "tiled_pointwise", "input_bytes_per_element": 2,
                       "outputs": [{"bytes_per_element": 2}], "parameter_bytes_per_col": 2,
                       "tile_rows": 1, "tile_cols": 6144},
    }}
    declared = export_op_structure(
        operators, model=case["model"], kernels=kernels,
        execution={"phase": "decode", "tokens": 1, "context": 0,
                   "sequences": 1, "scope": "body"})
    first, second = declared["nodes"]
    assert first["contract"] == second["contract"]
    assert first["physical_work"] == second["physical_work"]
    costs = operator_costs(case, operators, recipes=RecipeBook.from_mapping(kernels))["operators"]
    assert costs[0]["duration_ms"] != costs[1]["duration_ms"]
    mapping, _ = _captured_report(
        case, "a3_glm53_decode.csv", "RmsNorm", structure=declared)
    assert mapping["status_counts"] == {"ambiguous": 2}
    assert mapping["windows"][0]["cohorts"] == []
    assert mapping["windows"][0]["cohort_excluded_event_counts"] == {
        "candidate_physical_work_differs": 2}


def test_same_recipe_name_with_different_binding_changes_identity_and_rejects_old_mapping(
    profiler_api,
):
    envelope = _norm_case()
    envelope["runtime"]["kernels"] = {"recipes": {"rmsnorm": {"kind": "envelope"}}}
    tiled = deepcopy(envelope)
    tiled["runtime"]["kernels"]["recipes"]["rmsnorm"] = {
        "kind": "tiled_pointwise", "input_bytes_per_element": 2,
        "outputs": [{"bytes_per_element": 2}], "parameter_bytes_per_col": 2,
        "tile_rows": 1, "tile_cols": 6144,
    }
    first = export_structure(envelope, phase="decode", tokens=1, scope="body")
    second = export_structure(tiled, phase="decode", tokens=1, scope="body")
    norm = next(node for node in first["nodes"] if node["family"] == "rmsnorm")
    rebound = next(node for node in second["nodes"] if node["node_id"] == norm["node_id"])
    assert norm["contract"] == rebound["contract"]
    assert norm["physical_work"] == rebound["physical_work"]
    assert first["graph_id"] != second["graph_id"]
    assert norm["physical_signature"] != rebound["physical_signature"]
    op = next(op for op in target_graphs(
        envelope, phase="decode", tokens=1, scope="body")[0] if op.op_id == norm["op_id"])
    costs = [operator_costs(
        case, [op], recipes=RecipeBook.from_mapping(case["runtime"]["kernels"]))["operator_sum_ms"]
        for case in (envelope, tiled)]
    assert costs[0] != costs[1]
    mapping, _ = _captured_report(
        envelope, "a3_glm53_decode.csv", "RmsNorm", structure=first)
    with pytest.raises(ValueError, match="cohort_case_graph_id_mismatch"):
        compare_cohorts(tiled, mapping)
