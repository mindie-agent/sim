"""Graph composition checked against independent service and precedence oracles."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.__main__ import parser, run
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.program import RecipeBook, TransferRecipe
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def op(name, *, flops=0, memory=0, deps=(), stream="main", stage=0):
    return OpSpec(name, "test", "prefill", stage, 0, (), (1,), (1,), (1,), (1,),
                  flops=flops, hbm_bytes=memory, deps=deps, dtype="bf16", stream=stream)


def test_independent_streams_overlap_but_shared_memory_still_contends():
    # Both physical capacities are 1e6 units/ms; no hidden setup or calibration.
    compute = op("compute", flops=4e6)
    memory = op("copy", memory=3e6, stream="copy")
    parallel = graph_costs(case(), (compute, memory))
    serial = graph_costs(case(), (compute, replace(memory, stream="main")))
    assert parallel["duration_ms"] == 4
    assert serial["duration_ms"] == 7
    assert parallel["isolated"]["operator_sum_ms"] == serial["isolated"]["operator_sum_ms"] == 7
    assert parallel["bounds"]["precedence"]["duration_ms"] == 4
    assert serial["bounds"]["precedence"]["duration_ms"] == 7
    assert parallel["execution"]["serviced"] == serial["execution"]["serviced"]

    shared = graph_costs(case(), (memory, op("other_copy", memory=2e6, stream="other")))
    assert shared["duration_ms"] == 5
    assert shared["bounds"]["precedence"]["duration_ms"] == 3
    assert shared["bounds"]["resource_capacity_ms"] == 5
    assert shared["execution"]["serviced"]["memory@r0:s0"] == 5e6


def test_data_dependency_and_external_arrival_hold_all_resource_service():
    first = op("producer", flops=4e6, deps=("upstream",))
    second = op("consumer", memory=3e6, deps=("producer",), stream="copy")
    report = graph_costs(case(), (first, second), input_ready_ms=10,
                         external_inputs={"upstream": 12}, trace_limit=100)
    assert report["duration_ms"] == 9
    assert report["complete_ms"] == 19
    assert report["bounds"]["precedence"]["duration_ms"] == 9
    assert [row["ready_ms"] for row in report["timeline"]] == [12, 16]
    assert all(event["start_ms"] >= 12 for event in report["events"] if event["demand"])
    with pytest.raises(ValueError, match="graph_dependency_unavailable:producer:upstream"):
        graph_costs(case(), (first, second))
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


def test_device_boundary_excludes_host_cost_and_submitted_boundary_uses_same_execution():
    raw = case()
    raw["machine"]["device"]["launch_us"] = 1000
    raw["runtime"]["submission"] = {"scheduler_ms": 2, "completion_ms": .5}
    operators = (op("a", flops=4e6), op("b", flops=2e6, deps=("a",)))
    pure = graph_costs(raw, operators)
    submitted = graph_costs(raw, operators, include_submission=True)
    assert pure["duration_ms"] == 6
    assert submitted["duration_ms"] == 9.5
    parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed))
    step = execution.add(IterationWork(operators, IterationPlan(()), 1, 0, 0, None,
                                       "prefill", "test"), 0, 0)
    execution.engine.advance(until=(step.completion,))
    assert submitted["duration_ms"] == step.completion.time
    assert submitted["execution"]["serviced"] == execution.engine.serviced
    assert pure["execution"]["serviced"]["host@r0:s0"] == 0
    assert submitted["execution"]["serviced"]["host@r0:s0"] == 2
    assert submitted["execution"]["serviced"]["scheduler@r0"] == 2


@pytest.mark.parametrize("scope", ("body", "model"))
def test_real_frontend_preserves_join_barriers_and_model_boundary(scope):
    raw = case(pp=2, tp=2)
    graphs = target_graphs(raw, phase="prefill", tokens=3, scope=scope)
    inputs = {"analysis_input": 5} if scope == "body" else None
    report = graph_costs(raw, graphs, external_inputs=inputs, input_ready_ms=5)
    assert report["duration_ms"] > 0
    assert report["bounds"]["combined_ms"] <= report["duration_ms"] + 1e-9
    assert not any(row["family"] in {"sampling", "token_commit"} for row in report["timeline"])
    if scope == "model":
        assert any(row["op_id"] == "iteration_input" for row in report["timeline"])
        assert any(row["family"] == "embedding" for row in report["timeline"])
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    actual = report["execution"]["serviced"]
    summed = {}
    for row in report["isolated"]["operators"]:
        for resource, amount in row["serviced"].items():
            summed[resource] = summed.get(resource, 0) + amount * row["occurrences"]
    assert {key: amount for key, amount in actual.items() if amount} == pytest.approx(
        {key: amount for key, amount in summed.items() if amount})


def test_expert_cohort_waits_for_peer_inputs_and_charges_all_senders():
    raw = case(dp=2, ep=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    send = replace(op("exchange", deps=("local",)), family="alltoall", link_bytes=1000,
                   metadata=(("group_axis", "ep"),))
    graphs = {0: (op("local", flops=4e6), send),
              1: (op("local", flops=1e6), send)}
    report = graph_costs(raw, graphs)
    assert report["duration_ms"] == 6  # max(4, 1) + (1000+1000)/1000
    assert report["execution"]["serviced"]["link:fabric"] == 2000
    assert [row["complete_ms"] for row in report["timeline"]
            if row["op_id"] == "exchange"] == [6, 6]
    exchanges = [row for row in report["timeline"] if row["op_id"] == "exchange"]
    assert [row["ready_ms"] for row in exchanges] == [4, 4]
    assert [row["local_ready_ms"] for row in exchanges] == [4, 1]
    assert report["bounds"]["combined_ms"] <= report["duration_ms"]
    with pytest.raises(ValueError, match="graph_expert_collective_requires_complete_cohort"):
        graph_costs(raw, graphs[0])


def test_model_graph_includes_complete_ep_cohort_without_a_serving_scheduler():
    raw = case(dp=2, ep=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    graphs = target_graphs(raw, phase="decode", tokens=1, context=8, scope="model")
    assert set(graphs) == {0, 1}
    report = graph_costs(raw, graphs)
    assert report["duration_ms"] > 0
    assert report["bounds"]["combined_ms"] <= report["duration_ms"] + 1e-9
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert report["execution"]["serviced"]["link:fabric"] > 0


def test_expert_cohort_keeps_transfer_startup_and_per_invocation_supply():
    raw = case(dp=2, ep=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    raw["machine"]["links"][0]["latency_us"] = 10000
    exchange = replace(op("exchange"), family="alltoall", link_bytes=1000,
                       metadata=(("group_axis", "ep"),))
    graphs = {0: (exchange,), 1: (exchange,)}
    report = graph_costs(raw, graphs)
    assert report["duration_ms"] == 12  # startup 10 + both senders' 2ms service
    assert report["bounds"]["combined_ms"] <= report["duration_ms"]
    raw["machine"]["links"][0]["latency_us"] = 0
    recipe = TransferRecipe(transaction_bytes=10, outstanding=1, response_ms=1)
    limited = graph_costs(raw, graphs, recipes=RecipeBook(families={"alltoall": recipe}))
    # Two senders share 1000B/ms. Each of 100 transactions uses .02ms
    # wire service followed by a 1ms response before its one slot is returned.
    assert limited["duration_ms"] == pytest.approx(102)
    assert limited["bounds"]["combined_ms"] <= limited["duration_ms"]
    assert limited["execution"]["serviced"]["link:fabric"] == 2000


def test_finite_transfer_uses_link_frames_without_reframing_wire_payload():
    raw = case(tp=2)
    raw["machine"]["links"][0].update(
        transaction_bytes=16, protocol_overhead_bytes=4)
    exchange = replace(op("exchange"), family="alltoall", link_bytes=1)
    recipes = RecipeBook(families={"alltoall": TransferRecipe(
        transaction_bytes=16, outstanding=1, response_ms=1)})
    result = graph_costs(raw, (exchange,), recipes=recipes)
    # Two one-byte messages become two 20-byte frames. Each returns one credit.
    assert result["execution"]["serviced"]["link:fabric"] == 40
    assert result["duration_ms"] == pytest.approx(2.04)
    default_granule = graph_costs(raw, (exchange,), recipes=RecipeBook(
        families={"alltoall": TransferRecipe(outstanding=1, response_ms=1)}))
    assert default_granule["duration_ms"] == result["duration_ms"]
    with pytest.raises(ValueError, match="transfer_granule_conflicts_with_link"):
        graph_costs(raw, (exchange,), recipes=RecipeBook(families={
            "alltoall": TransferRecipe(transaction_bytes=8, outstanding=1, response_ms=1)}))


def test_graph_cli_runs_public_model_path_without_sampling_or_host_work():
    args = parser().parse_args(["graph", "examples/quickstart.json", "--phase", "decode",
                               "--tokens", "1", "--context", "16", "--skip-isolated"])
    report = run(args)
    assert report["duration_ms"] > 0
    assert any(row["family"] == "embedding" for row in report["timeline"])
    assert not any(row["family"] == "sampling" for row in report["timeline"])
    assert all(amount == 0 for name, amount in report["execution"]["serviced"].items()
               if name.startswith(("host@", "scheduler@")))


def test_target_graph_retains_frontend_capture_scope_and_padding():
    raw = case()
    raw["machine"]["device"]["launch_us"] = 1000
    raw["runtime"].update(graph_mode="auto", graph_capture_keys=[4], graph_capture_mode="full")
    graphs = target_graphs(raw, phase="decode", tokens=1)
    captured = graph_costs(raw, graphs, include_submission=True)
    eager = graph_costs(raw, graphs, include_submission=True, replay_scope="none")
    assert captured["replay_scope"] == "full"
    assert captured["execution"]["serviced"]["host@r0:s0"] == 1
    assert eager["execution"]["serviced"]["host@r0:s0"] > 1
    assert captured["execution"]["work"] == eager["execution"]["work"]
    assert captured["duration_ms"] < eager["duration_ms"]


def test_media_geometry_is_explicit_and_preserves_actual_frontend_tower_work():
    raw = json.loads(Path("examples/cases/qwen3.8-27b-media.json").read_text())
    with pytest.raises(ValueError, match="analysis_requires_explicit_media_tokens_per_sequence"):
        target_graphs(raw, phase="prefill", tokens=5)
    graphs = target_graphs(raw, phase="prefill", tokens=5, media_tokens_per_sequence=4)
    language = target_graphs(raw, phase="prefill", tokens=5, media_tokens_per_sequence=0)
    parsed = parse_case(raw)
    actual = Frontend()(IterationPlan((RequestView("r0", 5, 0, True, "prefill",
                                                  draft_prepare=False),)),
                        parsed.model, parsed.runtime, parsed.machine)
    expected = [op for op in actual.ops if op.family not in {"sampling", "token_commit"}]
    assert sum(op.flops for op in graphs[0]) == sum(op.flops for op in expected)
    assert sum(op.hbm_bytes for op in graphs[0]) == sum(op.hbm_bytes for op in expected)
    assert sum(op.flops for op in graphs[0]) > sum(op.flops for op in language[0])


def test_inactive_model_layers_do_not_expand_per_core_programs():
    raw = case()
    cores, layers = 64, 24
    raw["machine"]["device"]["compute_units"] = cores
    parsed = parse_case(raw)
    shape = (128, 128 * cores, 128)
    matrix = replace(
        op("matrix", flops=2 * shape[0] * shape[1] * shape[2]),
        local_shape=shape, semantic_shape=shape, padded_shape=shape, launch_shape=shape,
        metadata=(("program_kind", "matmul"),),
    )
    operators = tuple(replace(matrix, op_id=f"matrix{index}") for index in range(layers))
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    step = execution.add(
        IterationWork(operators, IterationPlan(()), 1, 0, 0, None, "prefill", "test"),
        0, 7,
    )
    # Only graph readiness/continuation handles are needed before any input.
    # An eager expansion creates thousands of idle per-core pipeline tasks.
    assert execution.engine.summary()["live_tasks"] < 4 * layers + 10
    execution.engine.advance(horizon=6)
    assert not any(execution.engine.serviced.values())
    execution.engine.advance(until=(step.completion,))
    assert step.completion.time == pytest.approx(7 + layers * matrix.flops / 1e6)
    assert execution.engine.serviced["matrix@r0:s0"] == pytest.approx(
        layers * matrix.flops / 1e6)
    assert execution.engine.peak_tasks < 12 * cores + 4 * layers
    assert execution.engine.summary()["live_tasks"] == 0
    assert execution.engine.summary()["live_programs"] == 0
