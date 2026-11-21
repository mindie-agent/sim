"""
[Role] Verify model scoreboard arithmetic, retained refreshes and coverage inventory.
[Invariants] Preserve comparison boundaries and keep capture identity out of timing denominators.
[Timeouts & Locks] Use bounded analytical predictions and small local fixtures.
[Side-Effect Policy] Write only temporary reports. No device or network use.
"""
from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
import json
import pytest
from _support.captures import REPO_ROOT, fixture_json, fixture_path
from _support.memory_cases import draft_case
from mindie_sim.coverage import comparison_coverage, inventory_coverage, record_linkage
from mindie_sim.observations.scoreboard import gap, model_scoreboard, prediction_from_probe, render_markdown
from mindie_sim.observations.scoreboard_accuracy import (
    annotate_call, annotate_kimi_report, render_kimi_markdown, step_fields,
)
from tools.model_scoreboard_kimi import comparison_row, prediction_packet, predict_worker

pytestmark = pytest.mark.tier0

RECURRENT_CASE = "kimi-k3-a5.json"

def prefill_entry():
    entries = fixture_json("verification", "scoreboard-input.json")["entries"]
    return next(entry for entry in entries if entry["id"] == "glm53_a3")

def test_prefill_gap_uses_step_span_with_overlapping_activity():
    entry = prefill_entry()
    pair = model_scoreboard([entry])["models"][0]["step_comparison"]
    observed, predicted = (entry[side]["step"]["duration_ms"] for side in ("observed", "predicted"))
    assert entry["observed"]["step"]["activity_sum_ms"] > observed
    assert pair["status"] == "conditional" and pair["formal_accuracy"] == "not_assessed"
    assert pair["signed_gap_ms"] == pytest.approx(predicted - observed)
    campaign = model_scoreboard(fixture_json("verification", "scoreboard-input.json")["entries"])
    assert campaign["coverage"]["unavailable_or_incomparable"] > 0
    audit = entry["observed"]["communication_audit"]["overlap"]
    rendered = render_markdown(campaign)
    assert f"Cross-domain intersection: {audit['native_aicpu_intersection_ms']:.6f} ms" in rendered

@pytest.mark.parametrize("field, value, reason", [
    ("context", None, "unknown_condition:context"),
    ("branch", "speculative_verify_and_draft", "condition_mismatch:branch"),
    ("parallel", {"tp": None}, "unknown_condition:parallel.tp"),
])
def test_step_accuracy_requires_matching_capture_conditions(field, value, reason):
    entry = prefill_entry()
    entry["observed"]["conditions"][field] = value
    pair = model_scoreboard([entry])["models"][0]["step_comparison"]
    assert pair["absolute_gap_ms"] is None
    assert reason in pair["reasons"]

def descriptor(identity, observed_us, predicted_us, count):
    observed, predicted = observed_us / 1000, predicted_us / 1000
    return {
        "id": identity, "family": "GroupedMatmul", "status": "conditional_reference",
        "occurrences": count, "observed_mean_ms": observed,
        "predicted_ms": predicted, "observed_sum_ms": observed * count,
        "predicted_sum_ms": predicted * count,
        "sum_error": gap(predicted * count, observed * count),
    }

def mixed_report():
    rows = [descriptor("small", 10, 24, 1000), descriptor("large", 100, 80, 700)]
    return {
        "schema": "mindie-sim.kimi-family-scoreboard/2", "rows": rows,
        "family_totals": [{
            "family": "GroupedMatmul", "status": "conditional_reference",
            "descriptor_groups": 2, "occurrences": 1700,
            "observed_sum_ms": 80, "predicted_sum_ms": 80,
            "sum_error": gap(80, 80), "descriptor_absolute_residual_sum_ms": 28,
        }],
        "grouped_reference_domains": [],
    }

def test_mixed_descriptor_mean_cannot_hide_large_failure_or_cumulative_cancellation():
    original = mixed_report()
    preserved = deepcopy(original)
    report = annotate_kimi_report(original)
    total = report["family_totals"][0]
    assert total["descriptive_tier"] is total["accuracy_pass"] is None
    assert total["observed_per_call_us"] == total["predicted_per_call_us"]
    diagnostic = total["descriptor_diagnostics"]
    assert diagnostic["statuses"] == {
        "within_launch_floor_tolerance": 1, "outside_tolerance": 1,
    }
    assert diagnostic["signed_cumulative_gap_ms"] == 0
    assert diagnostic["absolute_descriptor_cumulative_gap_ms"] == 28
    assert original == preserved
    assert annotate_kimi_report(report) == report
    rendered = render_kimi_markdown(report)
    assert "outside_tolerance" in rendered and "0.000000 / 28.000000" in rendered

def test_unresolved_native_phase_has_no_per_call_pair_or_zero_residual():
    original = mixed_report()
    row = original["rows"][0]
    row.update(status="not_comparable", predicted_sum_ms=None, sum_error=None)
    original["rows"] = [row]
    total = original["family_totals"][0]
    total.update(status="not_comparable", descriptor_groups=1, predicted_sum_ms=None,
                 sum_error=None, descriptor_absolute_residual_sum_ms=None)
    report = annotate_kimi_report(original)
    assert report["rows"][0]["predicted_per_call_us"] is None
    assert report["rows"][0]["descriptive_tier"] is None
    assert report["family_totals"][0]["descriptor_diagnostics"]["signed_cumulative_gap_ms"] is None

@pytest.mark.parametrize("lowering", [None, "vllm_ascend_kda_chunk_state"])
def test_recurrent_prediction_inputs_exclude_observed_counts_and_times(lowering):
    observed = fixture_json("verification", "kimi-native-descriptors.json")
    case = json.loads((REPO_ROOT / "examples/cases" / RECURRENT_CASE).read_text())
    if lowering:
        case["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})["kda"] = lowering
    source = next(row for row in observed["groups"] if row["kind"] == "ChunkKdaFwd")
    def packet(row):
        return prediction_packet("kda", {"groups": [row]}, case, {}, context=32768,
                                 kda_tokens=8192, matrix_recipe="steady_matmul")

    before = packet(source)
    assert packet({**source, "count": 100, "activity_sum_ms": 1000, "mean_ms": 10}) == before
    predicted = predict_worker(before)["rows"][0]
    if lowering:
        assert predicted["prediction"]["kda_implementation"] == lowering
    row = comparison_row(predicted, source, "kda")
    assert row["status"] == "not_comparable"
    stages = row["stage_predictions"]
    assert sum(stage["predicted_semantic_projection_ms"] for stage in stages) == pytest.approx(
        row["predicted_ms"])
    assert all(stage["observed_stage_ms"] is None for stage in stages)

def test_retained_family_mapping_conserves_all_costs(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    from refresh_model_scoreboard import family_mapping

    report = fixture_json("verification", "qwen38-family-costs.json")
    result = family_mapping(report)
    assert result["classified_ms"] + result["unassigned_ms"] == pytest.approx(
        report["costs"]["operator_sum_ms"])
    assignments = result["assignments"]
    assert len(assignments) == len(report["costs"]["families"])
    by_recipe = {row["recipe"]: row["semantic_family"] for row in assignments}
    assert by_recipe["flash_attention"] == "attention"
    assert by_recipe["recurrent_scan0"] == "gdn"
    assert by_recipe["linear_qkvz"] == "matmul"

def test_compact_publication_preserves_failed_latest_and_archived_selection(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    from retain_body_scoreboard import retain

    source = fixture_path("body-scoreboard-latest-timeout.json")
    full = source.read_bytes()
    archive = tmp_path / "full.json"
    archive.write_bytes(full)
    compact = json.loads(full)
    del compact["current_selection_policy"], compact["coverage_status"]
    compact.update(publication_format="mindie-priority-scoreboard-compact/1",
                   full_evidence={"path": str(archive), "sha256": sha256(full).hexdigest(),
                                  "size_bytes": len(full)})
    path = tmp_path / "compact.json"
    path.write_text(json.dumps(compact))
    expected, actual = retain(source), retain(path)
    for key in ("selection_policy", "coverage_status", "rows", "latest_attempt_counts"):
        assert actual[key] == expected[key]
    completed, latest = actual["rows"][0], actual["latest_attempts"][0]
    assert completed["case_id"] == latest["case_id"] and completed["strict_body_15pct"]
    assert latest["comparison"]["predicted_ms"] is None
    assert latest["prediction_failure"]["reason"] == "body_prediction_timeout"
    archive.write_bytes(full + b" ")
    with pytest.raises(ValueError, match="identity_mismatch"):
        retain(path)

def test_scheduled_scoreboard_prices_draft_sampling_and_commit(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    from model_scoreboard import execution_identity, scheduled_probe

    case = json.loads((REPO_ROOT / "examples/cases/qwen35-a3-tp4.json").read_text())
    case["model"]["layers"] = 4
    case["runtime"].setdefault("kernels", {}).setdefault("programs", {})["matmul"] = {"kind": "phase_matmul"}
    spec = {
        "case": case, "driver": "scheduled_iteration", "proposal_tokens": 1,
        "geometry": {"phase": "decode", "tokens": 1, "context": 16, "sequences": 1, "scope": "model"},
        "max_operators": 30_000, "top": 5,
    }
    probe = scheduled_probe(spec)
    assert {"sampling", "acceptance", "token_commit"} <= set(probe["iteration"]["families"])
    assert probe["graph"]["live_tasks"] == probe["graph"]["live_programs"] == 0
    conditions = execution_identity(spec, probe)["conditions"]
    assert conditions["tokens"] == 2 and conditions["branch"] == "full_speculative_step"

def test_scheduled_prefill_runs_draft_preparation_to_completion(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    from model_scoreboard import execution_identity, scheduled_probe

    parsed = draft_case()
    case = parsed.to_dict()
    case["model"]["auxiliary"] = asdict(parsed.model.auxiliary)
    spec = {
        "case": case, "driver": "scheduled_prefill", "proposal_tokens": 0, "draft_layers": 1,
        "geometry": {"phase": "prefill", "tokens": 4, "context": 0, "sequences": 1, "scope": "model"},
        "max_operators": 1000, "top": 5,
    }
    probe = scheduled_probe(spec)
    assert probe["iteration"]["phase_occurrences"]["draft_prefill"] > 0
    assert probe["iteration"]["plans"][0]["requests"][0]["draft_prepare"] is True
    assert "acceptance" not in probe["iteration"]["families"]
    assert probe["graph"]["live_tasks"] == probe["graph"]["live_programs"] == 0
    conditions = execution_identity(spec, probe)["conditions"]
    assert conditions["branch"] == "target_plus_mtp_prefill" and conditions["tokens"] == 4
    spec["draft_layers"] = 3
    with pytest.raises(ValueError, match="draft_layer_count_mismatch"):
        scheduled_probe(spec)

def step_entry():
    # Raw pf4k_b1.csv.gz: 2,392 positive tasks, one p4096/o1/c1 request.
    # The graph value is the independently executed phase probe retained with
    # this scoreboard; observed event spans never populate model parameters.
    conditions = {
        "model": "GLM-5.3-W4A8", "hardware": "Ascend910_9362",
        "parallel": {"tp": 8, "dp": 1, "ep": 1, "pp": 1},
        "phase": "prefill", "tokens": 4096, "context": 0, "sequences": 1,
        "precision": "W4A8 routed / W8A8 attention", "branch": "no_mtp",
    }
    return {
        "id": "glm53", "observed": {
            "conditions": conditions,
            "step": {"duration_ms": 1680.733399, "boundary": "profiled_task_span",
                     "comparison_scope": "single_prefill_iteration"},
        },
        "predicted": {
            "conditions": deepcopy(conditions),
            "step": {"duration_ms": 786.5277604589537, "boundary": "device_graph",
                     "comparison_scope": "single_prefill_iteration"},
        },
        "boundary_relation": {
            "observed": "profiled_task_span", "predicted": "device_graph",
            "assumptions": ["Observed runtime helpers remain inside the full window."],
        },
        "assumptions": ["Balanced routing and serial matrix phases are explicit approximations."],
    }

def test_group_sum_mean_or_occurrence_scope_mismatch_is_not_a_residual():
    entry = step_entry()
    observed = {
        "name": "attention", "comparison_key": "attention",
        "identity": {"shape": [4096, 8, 256], "dtype": "bf16"},
        "boundary": "kernel_activity_sum", "scope": "one_prefill",
        "statistic": "sum", "occurrences": 10, "duration_ms": 100,
    }
    entry["observed"]["groups"] = [observed]
    for delta in (
        {"statistic": "mean", "occurrences": 1, "duration_ms": 10},
        {"occurrences": 9}, {"scope": "whole_capture"},
    ):
        entry["predicted"]["groups"] = [{**observed, **delta}]
        group = model_scoreboard([entry])["models"][0]["group_comparisons"][0]
        assert group["reason"] == "group_statistical_scope_mismatch"
        assert "signed_gap_ms" not in group
    entry["predicted"]["groups"] = [{**observed, "duration_ms": 90}]
    group = model_scoreboard([entry])["models"][0]["group_comparisons"][0]
    assert group["signed_gap_ms"] == -10
    for delta, reason in (
        ({"identity": {"shape": [4096, 8, 256], "dtype": "fp16"}}, "group_contract_mismatch"),
        ({"boundary": "host_wrapper"}, "group_boundary_mismatch"),
    ):
        entry["predicted"]["groups"] = [{**observed, **delta}]
        group, = model_scoreboard([entry])["models"][0]["group_comparisons"]
        assert group["reason"] == reason and "signed_gap_ms" not in group
    entry["predicted"]["groups"] = []
    group, = model_scoreboard([entry])["models"][0]["group_comparisons"]
    assert group["status"] == "unavailable" and group["predicted"] is None

def test_executed_probe_identity_cannot_be_relabelled_to_different_geometry(monkeypatch):
    root = REPO_ROOT
    monkeypatch.syspath_prepend(str(root / "tools"))
    from model_convergence import run_probe
    from model_scoreboard import execution_identity

    spec = {
        "case": json.loads((root / "examples/quickstart.json").read_text()),
        "geometry": {"phase": "prefill", "tokens": 2, "context": 0,
                     "sequences": 1, "scope": "model"},
        "level": "graph", "max_operators": 1000, "top": 5,
    }
    probe = run_probe(spec)
    probe["execution_identity"] = execution_identity(spec, probe)
    actual = probe["execution_identity"]["conditions"]
    assert actual["tokens"] == 2
    claimed = {**actual, "tokens": 256}
    predicted = prediction_from_probe(
        probe, conditions=claimed, comparison_scope="single_prefill_iteration")
    entry = step_entry()
    entry["observed"]["conditions"] = claimed
    entry["predicted"] = predicted
    pair = model_scoreboard([entry])["models"][0]["step_comparison"]
    assert pair["status"] == "incomparable"
    assert pair["reasons"] == ["prediction_label_mismatch:tokens"]
    assert pair["predicted_ms"] == probe["graph"]["duration_ms"]
    del probe["execution_identity"]
    missing = prediction_from_probe(
        probe, conditions=actual, comparison_scope="single_prefill_iteration")
    assert missing["identity_issues"] == ["prediction_execution_identity_not_retained"]

def test_family_ranges_keep_native_grouping_and_failed_endpoint_coverage():
    entry = step_entry()
    entry["observed"]["conditions"]["context"] = None
    entry["observed"]["context_range"] = [16, 32]
    entry["observed"]["family_groups"] = [{
        "name": "matrix", "duration_ms": 10, "statistic": "mean_of_step_activity_sums",
        "occurrences": 7,
    }, {
        "name": "mixed_communication", "duration_ms": 20,
        "statistic": "mean_of_step_activity_sums", "comparable": False,
        "reason": "native_and_control_wrappers_cannot_be_added_as_link_work",
    }]
    low, high = deepcopy(entry["predicted"]), deepcopy(entry["predicted"])
    for scenario, context, duration in ((low, 16, 4), (high, 32, 6)):
        scenario["conditions"]["context"] = context
        scenario["groups"] = [{
            "name": "dense_gate_up", "family": "gemm", "duration_ms": duration,
            "occurrences": 2, "boundary": "sum_of_isolated_device_programs",
        }, {
            "name": "allreduce", "family": "allreduce", "duration_ms": 3,
            "occurrences": 1, "boundary": "sum_of_isolated_device_programs",
        }]
    entry["predicted"] = {"scenarios": [low, high]}
    high["assumptions"] = ["Package sources changed during this probe."]
    entry["family_bindings"] = [{
        "id": name, "observed_group": observed, "predicted_groups": [predicted],
        "assumptions": ["Native calls and analytic programs have different grouping."],
        "boundary_assumptions": ["Activity sums are compared with isolated program sums."],
    } for name, observed, predicted in (
        ("matrix", "matrix", "dense_gate_up"),
        ("collective", "mixed_communication", "allreduce"))]
    report = model_scoreboard([entry])
    rows = {row["comparison_key"]: row for row in report["models"][0]["group_comparisons"]}
    assert rows["matrix"]["predicted_range_ms"] == [4, 6]
    assert rows["matrix"]["signed_gap_range_ms"] == [-6, -4]
    assert rows["matrix"]["absolute_gap_range_ms"] == [4, 6]
    assert rows["collective"]["status"] == "incomparable"
    assert rows["collective"]["predicted_range_ms"] == [3, 3]
    assert rows["collective"]["signed_gap_ms"] is None
    rendered = render_markdown(report)
    assert "Observed runtime helpers remain inside the full window." in rendered
    assert "Context 32: Package sources changed during this probe." in rendered
    entry["observed"]["conditions"]["context"] = 16
    mismatched = model_scoreboard([entry])["models"][0]["step_comparison"]
    assert mismatched["status"] == "incomparable" and mismatched["predicted_ms"] is None
    entry["observed"]["conditions"]["context"] = None
    # A real failed endpoint must not look like a complete zero-width interval.
    entry["predicted"]["scenarios"][1] = {
        "conditions": high["conditions"], "status": "failed",
        "unavailable_reasons": ["prediction_timeout:40s"],
    }
    partial = model_scoreboard([entry])
    leader = partial["models"][0]["predicted_top"][0]
    assert leader["scenario_coverage"] == "1/2"
    assert leader["statistic"] == "partial_context_predictions_missing_endpoint"
    assert "| 1/2 |" in render_markdown(partial)
    assert "Context 32: prediction_timeout:40s" in render_markdown(partial)
    assert all(row["signed_gap_ms"] is None
               for row in partial["models"][0]["group_comparisons"])

@pytest.mark.parametrize("observed_us,predicted_us,status", [
    (50, 65, "within_launch_floor_tolerance"),
    (50, 65.001, "outside_tolerance"),
    (50.001, 65.001, "outside_tolerance"),
    (100, 115, "within_relative_tolerance"),
])
def test_homogeneous_group_uses_real_sum_count_and_shared_threshold(
        observed_us, predicted_us, status):
    entry = step_entry()
    row = {
        "name": "norm", "comparison_key": "norm",
        "identity": {"shape": [1, 1024], "dtype": "bf16"},
        "boundary": "complete_native_event", "scope": "window",
        "statistic": "sum", "occurrences": 20, "duration_ms": observed_us * 20 / 1000,
    }
    entry["observed"]["groups"] = [row]
    entry["predicted"]["groups"] = [{**row, "duration_ms": predicted_us * 20 / 1000}]
    original = deepcopy(entry)
    report = model_scoreboard([entry])
    group = report["models"][0]["group_comparisons"][0]
    assert group["observed_per_call_us"] == pytest.approx(observed_us)
    assert group["predicted_per_call_us"] == pytest.approx(predicted_us)
    assert group["per_call_diff_us"] == pytest.approx(predicted_us - observed_us)
    assert group["abs_error_per_call_us"] == pytest.approx(abs(predicted_us - observed_us))
    assert group["descriptive_tier"]["accuracy_status"] == status
    # A descriptive scoreboard cannot promote itself into scored native cohorts.
    assert group["accuracy_pass"] is None
    assert group["accuracy_status"] == "reference_only"
    assert group["signed_gap_ms"] == pytest.approx((predicted_us - observed_us) * 20 / 1000)
    assert entry == original
    markdown = render_markdown(report)
    assert status in markdown and "outside_tolerance" in markdown

def test_small_call_allowance_does_not_change_whole_step_or_endpoint_criterion():
    entry = step_entry()
    entry["observed"]["step"]["duration_ms"] = 0.04
    entry["predicted"]["step"]["duration_ms"] = 0.05
    call = annotate_call({
        "status": "conditional_closed_call_blocks", "closed_block_comparison": True,
        "occurrences": 10, "observed_sum_ms": 0.4, "predicted_sum_ms": 0.5,
    })
    assert call["descriptive_tier"]["accuracy_status"] == "within_launch_floor_tolerance"
    report = model_scoreboard([entry])
    step = report["models"][0]["step_comparison"]
    assert step["step_accuracy_pass"] is False
    assert step["absolute_gap_percent"] == pytest.approx(25)
    ranged = step_fields({
        "status": "conditional_range", "scenarios": [
            {"status": "conditional", "absolute_gap_percent": 14},
            {"status": "conditional", "absolute_gap_percent": 16},
        ],
    })
    assert ranged["step_accuracy_pass"] is False
    assert [row["step_accuracy_pass"] for row in ranged["step_accuracy_endpoints"]] == [True, False]

def test_missing_count_and_coarse_native_program_mismatch_do_not_invent_per_call():
    row = annotate_call({
        "status": "conditional_reference", "descriptor_groups": 1,
        "observed_mean_ms": 0.01, "predicted_weighted_mean_ms": 0.02,
        "observed_sum_ms": 100, "predicted_sum_ms": 200,
    })
    assert row["observed_per_call_us"] is row["predicted_per_call_us"] is None
    assert row["per_call_diff_us"] is row["descriptive_tier"] is None
    assert row["per_call_ineligible_reason"] == "retained_call_count_unavailable"
    entry = step_entry()
    entry["observed"]["family_groups"] = [{
        "name": "matrix", "duration_ms": 10, "occurrences": 7,
        "statistic": "mean_of_step_activity_sums",
    }]
    entry["predicted"]["groups"] = [{
        "name": "projection", "duration_ms": 2, "occurrences": 2,
    }]
    entry["family_bindings"] = [{
        "id": "matrix", "observed_group": "matrix", "predicted_groups": ["projection"],
        "assumptions": ["Native and program boundaries differ."],
        "boundary_assumptions": ["Whole-family activity diagnostic only."],
    }]
    group = model_scoreboard([entry])["models"][0]["group_comparisons"][0]
    assert group["signed_gap_ms"] == -8
    assert group["observed_per_call_us"] is group["predicted_per_call_us"] is None
    assert group["descriptive_tier"] is None
    assert group["per_call_ineligible_reason"] == "native_and_program_grouping_not_equivalent"

def write_csv(path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns.split())
        writer.writerows(rows)
    return path

def test_inventory_reconciles_capture_manifest_with_record_recovery(tmp_path):
    summary = tmp_path / "inventory.json"
    inventory = {"ledger_total_records": 3, "a5_records": 2, "a5_unique_event_captures": 1,
                 "a5_duplicate_extra_records": 1, "a5_duplicate_groups": 1,
                 "native": {"archive_cases": 1, "validation_rows": 2}}
    summary.write_text(json.dumps(inventory))
    scope = write_csv(
        tmp_path / "scope.csv",
        "audit_id hardware_tier suite phase capture_id duplicate_group_id duplicate_group_size duplicate_role in_a5_scope",
        [("direct", "A5", "controlled", "decode", "capture-0", "set", 2, "first_reference", True),
         ("alias", "A5", "controlled", "decode", "capture-0", "set", 2, "duplicate_reference", True),
         ("other", "A3", "controlled", "decode", "capture-1", "other", 1, "first_reference", False)])
    captures = {"capture_count": 2, "tier_capture_counts": {"A5": 1, "A3": 1}}
    manifest = {"capture_count": 2, "total_lines": 2, "shard_files": ["captures.jsonl"]}
    counts = dict(paired_items=2, duration_denominator_items=1, within=1, outside=0, work_only=1)
    for evidence_column in ("current_evidence_kind", "record_grain"):
        records = write_csv(
            tmp_path / "records.csv",
            "hardware_tier full_per_call_10pct_validated per_step_runtime_state_restored "
            f"context_explicitly_unobserved configuration_recovery_status {evidence_column} native_compared_cohorts",
            [("A5", False, True, False, "structured", "partial_native_contract_cohort_mean", 2),
             ("A5", False, False, True, "metadata", "inventory_only", 0),
             ("A3", False, False, True, "metadata", "inventory_only", 0)],
        )
        report = inventory_coverage(summary, captures, ledger=scope, records=records, captures_manifest=manifest)
        ledger, recovery = report["ledger"], report["a5_record_recovery"]
        assert ledger["total_records"] == 3 and ledger["a5_capture_count"] == 1
        assert ledger["capture_count_source"] == "captures_manifest"
        assert ledger["a5_native_case_count"] == 1 and ledger["a5_native_cohort_records"] == 2
        identity = report["identity_ledger"]
        assert identity["captures"]["records_per_capture_histogram"] == {"1": 1, "2": 1}
        assert identity["a5_captures"]["distinct"] == identity["duplicate_extra_records"] == 1
        assert recovery["flags"]["full_per_call_10pct_validated"] == {"false": 2}
        assert recovery["flags"]["source_binding_ambiguous"] == {"not_declared": 2}
        assert recovery["native_evidence_records"] == 1 and recovery["native_compared_cohorts_total"] == 2
        assert report["unrecovered"]["context_unobserved_records"] == 1
        assert report["sources"]["inventory_summary"]["sha256"] == sha256(summary.read_bytes()).hexdigest()
        coverage = comparison_coverage(report, {"counts": counts})
        assert coverage["comparison"]["duration_denominator_items"] == 1
        assert coverage["validated_fraction_of_ledger"] is None
    with pytest.raises(ValueError, match="inventory_ledger_row_count_mismatch"):
        inventory_coverage({**inventory, "ledger_total_records": 4}, ledger=scope)

@pytest.fixture
def linkage_inputs(tmp_path):
    raw, alias, indexed, sidecar = (str(tmp_path / label) for label in ("raw", "alias", "indexed", "sidecar"))
    rows = []
    for record, source, capture, path, role, derived, phase, case in (
        ("direct", "source-0", "capture-0", raw, "first_reference", False, "prefill", "native"),
        ("derived", "source-0", "capture-0", raw, "duplicate_reference", True, "prefill", ""),
        ("other", "source-1", "capture-1", sidecar, "first_reference", False, "decode", ""),
    ):
        edge = {"source_id": source, "capture_id": capture, "event_source": path, "raw_path": path}
        ids = json.dumps([source])
        duplicates = [edge, edge] if derived else [edge]
        rows.append((record, "A5", "controlled", phase, path, "" if derived else source, ids, ids,
                     json.dumps([capture]), json.dumps(duplicates), derived, "", source,
                     1 if record == "other" else 2, role, case))
    dispositions = write_csv(
        tmp_path / "dispositions.csv",
        "audit_id hardware_tier suite phase primary_raw_source primary_source_id source_ids event_source_ids "
        "evidence_capture_ids source_statuses derived derived_key source_set_group_id source_set_group_size "
        "source_set_role native_case_id", rows)
    record_sources = write_csv(
        tmp_path / "sources.csv", "audit_id source_id raw_path canonical_path event_source",
        [("direct", "source-0", raw, alias, ""), ("unrelated", "source-2", "", "", "")])
    registry = write_csv(
        tmp_path / "registry.csv", "source_id canonical_path event_sources record_ids",
        [("source-0", alias, json.dumps([raw, ""]), '["direct","derived"]')])
    capture_index = write_csv(
        tmp_path / "captures.csv", "capture_id event_source tiers record_count",
        [("capture-0", indexed, '["A5"]', 2), ("capture-1", "", '["A5"]', 1)])
    return (dict(dispositions=dispositions, record_sources=record_sources, source_registry=registry,
                 captures_index=capture_index, captures_manifest={"capture_count": 2}), alias, indexed)

@pytest.mark.parametrize("reference, status, reason", [
    ("observation", "matched", None),
    ("window", "matched", None),
    ("capture_index", "matched", None),
    ("source_registry", "matched", None),
    ("foreign_path", "unmatched", "source_not_in_ledger"),
    ("phase", "conflict", "phase_conflict"),
    ("capture", "conflict", "declared_capture_conflicts_with_source"),
    ("unregistered_source", "unmatched", "source_edge_has_no_capture"),
])
def test_record_linkage_uses_actual_edges_without_changing_duration_denominator(
        linkage_inputs, reference, status, reason):
    inputs, alias, indexed = linkage_inputs
    source_edges = {
        "observation": [{"observation_id": "obs", "source_id": "source-0", "capture_id": "capture-0"}],
        "window": {"window": {"source_id": "source-0"}},
        "capture_index": {"item": [{"source_path": indexed}]},
        "source_registry": {"source_edges": {"item": {"source_path": alias}}},
        "foreign_path": {"item": {"source_path": alias + "-missing"}},
        "phase": {"item": {"source_id": "source-1"}},
        "capture": {"item": {"source_id": "source-0", "capture_id": "capture-1"}},
        "unregistered_source": {"item": {"source_id": "source-2"}},
    }
    acceptance = {
        "counts": {"paired_items": 1, "within": 1, "duration_denominator_items": 1},
        "source_identity": {"path": alias},
        "items": [{"item_id": "item", "request_id": "request", "identity": {"phase": "prefill"},
                   "observation_ids": [] if reference == "window" else ["obs"],
                   "observation_window_ids": ["window"] if reference == "observation" else [],
                   "samples": [{"observation_id": "obs", "window_id": "window"}, {}]}],
    }
    linkage = record_linkage(acceptance, source_edges=source_edges[reference], **inputs)
    item, = linkage["items"]
    assert item["status"] == status and item["reason"] == reason
    assert item["observation_ids"] == ["obs"] and item["observation_window_ids"] == ["window"]
    evidence = linkage["independent_evidence"]
    assert evidence["precision_denominator_paired_items"] == 1
    if status == "matched":
        assert item["selected_source_edges"] == [{"source_id": "source-0", "capture_id": "capture-0"}]
        assert item["matched_audit_ids"] == ["derived", "direct"]
        assert evidence["independent_capture_count"] == evidence["independent_source_set_count"] == 1
        assert evidence["derived_records"] == evidence["duplicate_reference_records"] == 1
        assert evidence["native_case_ids"] == ["native"]
    else:
        assert evidence["matched_audit_records"] == evidence["independent_capture_count"] == 0

def test_source_phase_calls_and_reference_do_not_replace_missing_step():
    entry = step_entry()
    reference = {**deepcopy(entry), "id": "neighbour", "role": "reference"}
    del entry["observed"]["step"]
    del reference["boundary_relation"]
    stages = fixture_json("verification", "kda-prepare-stages.json")["stages"]
    stage = next(row for row in stages if row["stage"] == "GATE_PREPARE")
    obs, pred, calls = (stage[key] for key in ("observed_sum_ms", "compact_predicted_sum_ms", "observed_calls"))
    entry["call_comparisons"] = [{
        "family": "closed stage", "status": "conditional_source_phase", "closed_block_comparison": True,
        "occurrences": calls, "observed_mean_ms": obs / calls, "observed_sum_ms": obs,
        "predicted_sum_ms": pred, "sum_error": gap(pred, obs),
        "assumption": "Only the retained stage is compared.",
        "descriptor_absolute_residual_sum_ms": abs(pred - obs), "semantic_reference_ms": pred,
    }]
    report = model_scoreboard([entry, reference])
    assert report["coverage"] == {"models": 1, "conditional_step_comparisons": 0,
                                  "unavailable_or_incomparable": 1}
    row, = report["models"][0]["call_comparisons"]
    assert row["per_call_diff_us"] == pytest.approx((pred - obs) * 1000 / calls)
    assert row["descriptive_tier"]["accuracy_pass"] is False and row["accuracy_pass"] is None
    rendered = render_markdown(report)
    assert "timing_boundary_relation_not_declared" in rendered
    assert "complete semantic reference:" in rendered and "outside_tolerance" in rendered
