"""Public graph and serving controls for physical cross-DP collective plans."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs, predict, target_graphs
from mindie_sim.prediction.cost import primitive_key
from mindie_sim.prediction.primitives import PrimitiveStore
from mindie_sim.prediction.structure import parse_case
from test_communication_topology import exchange
from test_graph_analysis import op
from test_prediction_system import case


def cohort_case(count):
    raw = case(dp=count, ep=count)
    raw["model"].update(ffn="moe", num_experts=count * 2, top_k=2,
                        moe_intermediate_size=32)
    raw["machine"]["links"][0].update(
        semantics="per_pair", capacity_bytes_per_ms=1)
    raw["runtime"]["kernels"] = {"families": {
        "alltoall": {"kind": "pairwise_collective"},
        "expert_dispatch": {"kind": "pairwise_collective"},
        "expert_combine": {"kind": "pairwise_collective"},
    }}
    return raw


def wire(report):
    return {key: value for key, value in report["execution"]["serviced"].items()
            if key.startswith("link:") and value}


def test_joint_graph_keeps_idle_receives_local_entry_and_early_replica_completion():
    raw = cohort_case(3)
    graphs = {
        rank: (
            op("producer", flops=2e6 if rank == 1 else 0),
            replace(exchange(volume, axis="ep"), deps=("producer",)),
            op("consumer", flops=1e6 if rank == 0 else 0, deps=("exchange",)),
        )
        for rank, volume in enumerate((0, 4, 8))
    }
    report = graph_costs(raw, graphs, trace_limit=200)
    indexed = {row["id"]: row for row in report["timeline"]}
    # First round: 1->2 serves [2,4], 2->0 serves [0,4].
    # Second round: 1->0 serves [4,6], 2->1 serves [4,8].
    assert [indexed[f"r{r}:exchange"]["ready_ms"] for r in range(3)] == [0, 2, 0]
    assert [indexed[f"r{r}:exchange"]["complete_ms"] for r in range(3)] == [6, 8, 8]
    assert indexed["r0:consumer"]["complete_ms"] == 7
    assert report["duration_ms"] == 8
    assert not any("rendezvous_wait_ms" in row for row in report["timeline"])
    assert sum(wire(report).values()) == 12
    assert report["bounds"]["combined_ms"] <= report["duration_ms"]
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0

    isolated = report["isolated"]
    rows = [row for row in isolated["operators"] if row["program"] == "PairwiseCollective"]
    # Isolation removes the 2ms producer delay, so rank 0 completes at 4ms.
    assert [row["duration_ms"] for row in rows] == [4, 8, 8]
    assert [row["communication"]["wire_bytes"] for row in rows] == [0, 4, 8]
    assert isolated["collective_cohorts"][0]["duration_ms"] == 8
    assert isolated["operator_sum_ms"] == 23
    assert all(not row["service_delta"] for row in rows)
    assert not rows[0]["serviced"]


def test_primitive_minimum_delays_only_its_own_replica_completion():
    raw = cohort_case(2)
    idle, sender = exchange(0, axis="ep"), exchange(4, axis="ep")
    parsed = parse_case(raw)
    with PrimitiveStore(":memory:") as primitives:
        primitives.record(
            primitive_key(parsed, idle, "none"), duration_ms=20,
            run_id="local-completion-control", repeat_index=0,
            source="independent-test-constraint")
        report = graph_costs(
            raw, {0: (idle,), 1: (sender, op("after", flops=1e6, deps=("exchange",)))},
            primitives=primitives)
    indexed = {row["id"]: row for row in report["timeline"]}
    assert indexed["r0:exchange"]["complete_ms"] == 20
    assert indexed["r1:exchange"]["complete_ms"] == 4
    assert indexed["r1:after"]["complete_ms"] == 5
    assert report["duration_ms"] == 20
    assert sum(wire(report).values()) == 4
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


def test_complete_cohort_isolation_keeps_repeated_call_wire_once_per_source():
    raw = cohort_case(3)
    graphs = {
        rank: tuple(replace(exchange(volume, axis="ep"), op_id=f"call{index}")
                    for index in range(2))
        for rank, volume in enumerate((0, 4, 8))
    }
    report = graph_costs(raw, graphs)
    isolated = report["isolated"]
    assert len(isolated["collective_cohorts"]) == 1
    assert isolated["collective_cohorts"][0]["occurrences"] == 2
    assert [row["occurrences"] for row in isolated["operators"]] == [2, 2, 2]
    assert isolated["operator_sum_ms"] == 40
    assert sum(row["communication"]["wire_bytes"] * row["occurrences"]
               for row in isolated["operators"]) == sum(wire(report).values()) == 24
    assert report["bounds"]["combined_ms"] <= report["duration_ms"]
    with pytest.raises(ValueError, match="operator_collective_requires_complete_cohort"):
        operator_costs(raw, graphs[0])
    with pytest.raises(ValueError, match="graph_expert_collective_requires_complete_cohort"):
        graph_costs(raw, graphs[0])


def test_ordinary_frontend_graph_and_predict_bind_cross_dp_pairwise_programs():
    raw = cohort_case(2)
    raw["workload"]["requests"] = [
        {"id": "first", "prompt_tokens": 4, "output_tokens": 1},
        {"id": "second", "prompt_tokens": 4, "output_tokens": 1},
    ]
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="model")
    graph = graph_costs(raw, graphs)
    serving = predict(raw, event_limit=1000)
    assert serving["status"] == "modeled"
    assert graph["duration_ms"] > 0 and serving["metrics"]["makespan_ms"] > 0
    assert wire(graph) == wire(serving)
    assert any(row["program"] == "PairwiseCollective" for row in graph["timeline"])
    assert any(":cohort:group0:rank1:round0:hop0:service" in row["task"]
               for row in serving["events"])
    assert graph["execution"]["live_tasks"] == serving["execution"]["live_tasks"] == 0
    assert graph["execution"]["live_programs"] == serving["execution"]["live_programs"] == 0
    assert graph["bounds"]["combined_ms"] <= graph["duration_ms"] + 1e-9


def test_tp_sources_and_pipeline_stage_enter_the_joint_plan_exactly_once():
    raw = case(dp=2, tp=2, pp=2, ep=4)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2,
                        moe_intermediate_size=32)
    raw["machine"]["links"][0].update(
        semantics="per_pair", capacity_bytes_per_ms=1)
    raw["runtime"]["kernels"] = {
        "families": {"alltoall": {"kind": "pairwise_collective"}}}
    transfer = exchange(3, family="alltoall", axis="ep", stage=1)
    report = graph_costs(raw, {0: (transfer,), 1: (transfer,)})
    cohort = report["isolated"]["collective_cohorts"][0]
    # Stage 1 has ranks 2,3 on DP0 and 6,7 on DP1. Each source sends one
    # byte to each remote peer, with three independent 1ms rounds.
    assert cohort["rank_completion_ms"] == {"2": 3, "3": 3, "6": 3, "7": 3}
    assert report["duration_ms"] == 3
    assert sum(wire(report).values()) == 12
    assert [row["communication"]["wire_bytes"]
            for row in report["isolated"]["operators"]] == [6, 6]
    assert report["isolated"]["operator_sum_ms"] == 6
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
