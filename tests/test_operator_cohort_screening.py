"""Public isolated screening retains actual joint transports and source work."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, study, target_graphs
from mindie_sim.__main__ import main
from test_communication_topology import exchange
from test_joint_collective_execution import cohort_case


def test_isolated_screening_counts_repeated_joint_cohort_once_per_source():
    raw = cohort_case(3)
    graphs = {
        rank: tuple(replace(exchange(volume, axis="ep"), op_id=f"call{index}")
                    for index in range(2))
        for rank, volume in enumerate((0, 4, 8))
    }
    isolated = operator_costs(raw, graphs, input_ready_ms=7)
    graph = graph_costs(raw, graphs)
    assert isolated["operator_sum_ms"] == graph["isolated"]["operator_sum_ms"] == 40
    assert isolated["operator_occurrences"] == 6
    assert len(isolated["collective_cohorts"]) == 1
    assert isolated["collective_cohorts"][0]["occurrences"] == 2
    assert sum(sum(row["serviced"].values()) * row["occurrences"]
               for row in isolated["operators"]) == 24
    assert [row["communication"]["wire_bytes"] for row in isolated["operators"]] == [0, 4, 8]
    assert all(not row["service_before_input"] and row["live_tasks"] == 0
               for row in isolated["operators"])
    # The entire input mapping exists, but a counterpart call is missing.
    graphs[2] = graphs[2][:1]
    with pytest.raises(ValueError, match="collective_participants_mismatch:call1"):
        operator_costs(raw, graphs)


def test_public_study_and_operators_cli_share_complete_frontend_cohort(tmp_path):
    path = Path(__file__).parents[1] / "examples/cases/balanced-experts-control.json"
    raw = json.loads(path.read_text())
    probe = {"id": "prefill", "phase": "prefill", "tokens": 4}
    result = study([{"id": "balanced", "case": raw}], [probe])
    assert result["status"] == "completed"
    evaluation = result["evaluations"][0]
    graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=4))
    output = tmp_path / "operators.json"
    assert main(["operators", str(path), "--phase", "prefill", "--tokens", "4",
                 "--scope", "model", "-o", str(output)]) == 0
    cli = json.loads(output.read_text())
    assert evaluation["duration_ms"] == pytest.approx(graph["isolated"]["operator_sum_ms"])
    assert cli["operator_sum_ms"] == pytest.approx(evaluation["duration_ms"])
    assert cli["replicas"] == evaluation["replicas"] == [0, 1]
    assert cli["operator_occurrences"] == graph["operator_occurrences"]
    assert evaluation["graph"] is None
    assert any(row["program"] == "PairwiseCollective" for row in cli["operators"])
    assert sum(row["communication"]["wire_bytes"] * row["occurrences"]
               for row in cli["operators"] if row["program"] == "PairwiseCollective") == 1536


def test_study_does_not_rank_different_transport_isolation_scopes_as_equal_boundaries():
    path = Path(__file__).parents[1] / "examples/cases/balanced-experts-control.json"
    raw = json.loads(path.read_text())
    candidates = [
        {"id": "causal", "case": raw},
        {"id": "envelope", "case": raw, "overrides": {
            "runtime": {"kernels": {"families": {"alltoall": {"kind": "envelope"}}}}}},
        *({"id": role, "case": raw, "overrides": {
            "runtime": {"kernels": {"recipes": {role: {"kind": "envelope"}}}}}}
          for role in ("expert_dispatch", "expert_combine")),
    ]
    probes = [{"id": "p4", "phase": "prefill", "tokens": 4}]
    isolated = study(candidates, probes)
    assert isolated["status"] == "completed"
    assert {row["operator_isolation"] for row in isolated["evaluations"]} == {
        "per_replica", "complete_cohort", "mixed"}
    assert len(isolated["rankings"]) == 4
    assert all(row["status"] == "single_candidate" for row in isolated["rankings"])
    graph = study(candidates, probes, level="graph")
    assert graph["status"] == "completed"
    assert len(graph["rankings"]) == 1
    assert graph["rankings"][0]["status"] == "ranked"
    assert len(graph["rankings"][0]["candidates"]) == 4
    assert all(row["operator_isolation"] is None for row in graph["evaluations"])
