"""Additive compact graph/body output keeps exact per-replica identity.

[Role] Prove the compact result is an emission reduction, not a second
  evaluator, on checked-in representative A3 body cases.
[Invariants] Same finite graph, same readiness/completion observer, same
  duration/work. Structural per-replica operator identity, counts, and source
  ready/complete boundaries are reconstructed from the full timeline. No
  synthetic time histogram is materialized.
[Timeouts & Locks] Small one/two-rank A3 graphs and one short body forward.
[Side-Effect Policy] Read-only case load and in-memory simulation.
"""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.prediction.body import body_forward
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import target_graphs
from test_graph_analysis import case as generic_case, op


ROOT = Path(__file__).resolve().parents[1]
A3_FIXTURE = ROOT / "examples/cases/glm-5.3.json"


def glm51_a3(*, layers=1):
    return compose_case(
        "glm-5.1", "a3", tp=1, dp=1, ep=1, pcp=1, layers=layers,
        prompt_tokens=8, output_tokens=1, max_num_batched_tokens=16,
        max_num_seqs=2, max_model_len=256,
    )


def a3_fixture(*, layers=2):
    raw = json.loads(A3_FIXTURE.read_text())
    raw["model"] = {**raw["model"], "layers": layers}
    return raw


def groups(full):
    grouped = {}
    for row in full["timeline"]:
        identity = f"s{row['stage']}:{row['op_id']}"
        grouped.setdefault(identity, {}).setdefault(row["replica"], []).append(row)
    return grouped


def assert_compact_equals_full(full, compact):
    assert compact["result_mode"] == "compact"
    assert "timeline" not in compact and "events" not in compact
    assert compact["timeline_omitted"] is True
    assert compact["events_omitted"] is True
    assert "by_window" not in compact["profile"]
    assert "window_ms" not in compact["profile"]
    for key in ("duration_ms", "complete_ms", "input_ready_ms",
                "operator_occurrences", "replicas", "replay_scope",
                "replay_scope_by_replica", "resources", "execution",
                "parameters"):
        assert compact[key] == full[key], key

    expected = groups(full)
    assert set(compact["profile"]["by_operator"]) == set(expected)
    for identity, by_rank in expected.items():
        item = compact["profile"]["by_operator"][identity]
        rows = [row for values in by_rank.values() for row in values]
        assert item["operator_events"] == len(rows)
        assert item["ready_ms"] == min(row["ready_ms"] for row in rows)
        assert item["complete_ms"] == max(row["complete_ms"] for row in rows)
        assert item["elapsed_ms"] == sum(row["elapsed_ms"] for row in rows)
        assert set(item["replicas"]) == set(by_rank)
        for rank, rank_rows in by_rank.items():
            actual = item["replicas"][rank]
            assert actual["operator_events"] == len(rank_rows)
            assert actual["ready_ms"] == min(row["ready_ms"] for row in rank_rows)
            assert actual["complete_ms"] == max(row["complete_ms"] for row in rank_rows)
            assert actual["elapsed_ms"] == sum(row["elapsed_ms"] for row in rank_rows)
            # Rank-specific declared identity, not another rank's label.
            assert (actual["family"], actual["layer"], actual["stream"],
                    actual["phase"], actual["program"], actual["recipe"],
                    actual["source_boundary"]) == (
                rank_rows[0]["family"], rank_rows[0]["layer"],
                rank_rows[0]["stream"], rank_rows[0]["phase"],
                rank_rows[0]["program"], rank_rows[0]["recipe"],
                rank_rows[0]["source_boundary"])
            assert all((row["layer"], row["stream"]) ==
                       (actual["layer"], actual["stream"]) for row in rank_rows)

    for family, family_rows in {
            family: [row for row in full["timeline"] if row["family"] == family]
            for family in {row["family"] for row in full["timeline"]}}.items():
        summary = compact["profile"]["by_family"][family]
        assert summary["operator_events"] == len(family_rows)
        assert summary["elapsed_ms"] == sum(row["elapsed_ms"] for row in family_rows)
        for rank in {row["replica"] for row in family_rows}:
            rank_rows = [row for row in family_rows if row["replica"] == rank]
            assert summary["replicas"][rank]["operator_events"] == len(rank_rows)
            assert summary["replicas"][rank]["elapsed_ms"] == sum(
                row["elapsed_ms"] for row in rank_rows)

    assert compact["event_profile"]["completed_physical_events"] == \
        full["execution"]["completed_tasks"]
    assert compact["event_profile"]["trace_retained"] == len(full["events"])
    assert compact["event_profile"]["trace_dropped"] == \
        full["execution"]["trace_dropped"]
    assert compact["event_profile"]["trace_events"] == (
        len(full["events"]) + full["execution"]["trace_dropped"])


@pytest.mark.parametrize("raw", [
    pytest.param(glm51_a3(), id="glm51-a3-body"),
    pytest.param(a3_fixture(), id="checked-in-a3-glm53-fixture"),
])
def test_compact_equals_full_per_family_operator_rank_and_source_window(raw):
    graphs = target_graphs(raw, phase="decode", tokens=8, context=0,
                           sequences=1, scope="body")
    full = graph_costs(raw, graphs, external_inputs={"analysis_input": 0},
                       isolated=False, trace_limit=4096)
    compact = graph_costs(raw, graphs, external_inputs={"analysis_input": 0},
                          isolated=False, trace_limit=4096, compact=True)
    assert_compact_equals_full(full, compact)


def test_compact_keeps_heterogeneous_rank_identity_and_has_no_histogram():
    raw = generic_case(dp=2)
    first = op("same_id", memory=64, stream="computeA")
    second = replace(first, stream="computeB", layer=7)
    options = {"external_inputs": {}, "isolated": False, "trace_limit": 10}
    full = graph_costs(raw, {0: [first], 1: [second]}, **options)
    compact = graph_costs(raw, {0: [first], 1: [second]}, compact=True, **options)
    item = compact["profile"]["by_operator"]["s0:same_id"]
    assert (item["replicas"][0]["layer"], item["replicas"][0]["stream"]) == (0, "computeA")
    assert (item["replicas"][1]["layer"], item["replicas"][1]["stream"]) == (7, "computeB")
    assert "layer" not in item and "stream" not in item
    assert "by_window" not in compact["profile"]
    assert compact["event_profile"]["completed_physical_events"] > 0
    zero = graph_costs(raw, {0: [first], 1: [second]}, compact=True,
                       external_inputs={}, isolated=False, trace_limit=0)
    assert zero["event_profile"]["completed_physical_events"] > 0
    assert zero["event_profile"]["trace_events"] == 0
    with pytest.raises(TypeError):
        graph_costs(raw, {0: [first], 1: [second]}, compact=True,
                    compact_window_ms=1.0, **options)


def test_compact_multi_rank_keeps_each_replica_count_and_time():
    raw = compose_case(
        "glm-5.1", "a3", tp=1, dp=2, ep=2, pcp=1, layers=4,
        prompt_tokens=8, output_tokens=1, max_num_batched_tokens=16,
        max_num_seqs=2, max_model_len=256,
    )
    graphs = target_graphs(raw, phase="decode", tokens=8, context=0,
                           sequences=1, scope="body")
    full = graph_costs(raw, graphs, external_inputs={"analysis_input": 0},
                       isolated=False, trace_limit=1024)
    compact = graph_costs(raw, graphs, external_inputs={"analysis_input": 0},
                          isolated=False, trace_limit=1024, compact=True)
    assert_compact_equals_full(full, compact)
    assert set(compact["profile"]["by_rank"]) == {0, 1}
    for item in compact["profile"]["by_operator"].values():
        assert sum(rank["operator_events"] for rank in item["replicas"].values()) \
            == item["operator_events"]


def test_body_forward_compact_keeps_same_body_span_and_profile():
    raw = glm51_a3()
    default = body_forward(raw, phase="decode", tokens=8, context=0,
                           sequences=1, event_limit=512)
    compact = body_forward(raw, phase="decode", tokens=8, context=0,
                           sequences=1, event_limit=512, compact=True)
    assert compact["body_forward_ms"] == default["body_forward_ms"]
    assert compact["geometry"] == default["geometry"]
    assert compact["capacity"] == default["capacity"]
    assert compact["execution"] == default["execution"]
    assert compact["duration_ms"] == default["duration_ms"]
    assert compact["profile"]["operator_events"] == len(default["timeline"])
    assert compact["profile"]["by_family"] == {
        family: {
            "operator_events": len(rows),
            "elapsed_ms": sum(row["elapsed_ms"] for row in rows),
            "replicas": compact["profile"]["by_family"][family]["replicas"],
        }
        for family, rows in {
            family: [row for row in default["timeline"] if row["family"] == family]
            for family in {row["family"] for row in default["timeline"]}}.items()
    }


def test_compact_isolated_reduction_preserves_weighted_sum_and_bound():
    raw = a3_fixture()
    graphs = target_graphs(raw, phase="decode", tokens=8, context=0,
                           sequences=1, scope="body")
    full = graph_costs(raw, graphs, external_inputs={"analysis_input": 0},
                       trace_limit=0)
    compact = graph_costs(raw, graphs, external_inputs={"analysis_input": 0},
                          trace_limit=0, compact=True)
    assert compact["isolated"]["operator_sum_ms"] == full["isolated"]["operator_sum_ms"]
    assert compact["isolated"]["operators_omitted"] is True
    assert "operators" not in compact["isolated"]
    assert compact["isolated"]["by_family"] == {
        family: {
            "occurrences": sum(row["occurrences"] for row in full["isolated"]["operators"]
                               if row["operator"]["family"] == family),
            "total_ms": sum(row["total_ms"] for row in full["isolated"]["operators"]
                            if row["operator"]["family"] == family),
        }
        for family in {row["operator"]["family"]
                       for row in full["isolated"]["operators"]}
    }
    assert compact["bounds"] == full["bounds"]
    assert compact["duration_ms"] == full["duration_ms"]


def test_compact_publishes_selected_matrix_wrapper_kind_without_second_op():
    raw = generic_case()
    wrapper = {
        "native_kind": "GroupedMatmul",
        "source_role": "single_group_dense_wrapper",
        "source": {"basis": "declared_source_wrapper_not_capture_dispatch_proof"},
    }
    selected = replace(
        op("latent", flops=4e6), family="projection", recipe="latent_projection",
        metadata=(("matrix_wrapper", wrapper),))
    compact = graph_costs(raw, {0: [selected]}, compact=True,
                          isolated=False, trace_limit=0)
    item = compact["profile"]["by_operator"]["s0:latent"]
    rank = item["replicas"][0]
    assert item["operator_events"] == 1
    assert rank["native_kind"] == "GroupedMatmul"
    assert rank["source_role"] == "single_group_dense_wrapper"
    assert rank["source_basis"] == "declared_source_wrapper_not_capture_dispatch_proof"
    assert compact["execution"]["completed_tasks"] >= 1

    ordinary = replace(selected, metadata=())
    plain = graph_costs(raw, {0: [ordinary]}, compact=True,
                        isolated=False, trace_limit=0)
    plain_rank = plain["profile"]["by_operator"]["s0:latent"]["replicas"][0]
    assert "native_kind" not in plain_rank
    assert "source_role" not in plain_rank
    assert "source_basis" not in plain_rank
