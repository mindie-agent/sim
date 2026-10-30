"""Integer expert tails conserve banks, scales, invocations and cohort rows."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs, predict, target_graphs, target_operators
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.grouped_rows import row_classes, split_context
from mindie_sim.prediction.matrix_steady import SteadyMatmul
from mindie_sim.prediction.structure import parse_case
from test_balanced_experts import balanced_case, groups


def test_balanced_default_honors_declared_matrix_program_without_recipe_overrides():
    raw = balanced_case(tokens=5)
    raw["runtime"]["kernels"]["programs"] = {"matmul": {"kind": "steady_matmul"}}
    case = parse_case(raw)
    op = groups(target_operators(case, phase="decode", tokens=5))[0]
    execution = Execution(case, CostModel(case))
    _, recipe = execution.prepare(op)
    assert isinstance(recipe.matrix, SteadyMatmul)
    result = operator_costs(case, (op,))["operators"][0]
    assert result["duration_ms"] > 0 and not result["live_tasks"]


@pytest.mark.parametrize("kind", ("mapped_matmul", "steady_matmul", "phase_matmul",
                                  "hierarchical_matmul", "tiled_matmul"))
def test_mixed_rows_execute_one_kernel_and_preserve_all_service(kind):
    raw = balanced_case(tokens=5)
    raw["runtime"]["kernels"]["recipes"] = {
        name: {"kind": kind} for name in ("balanced_grouped_gate_up", "balanced_grouped_down")}
    raw["machine"]["device"]["compute_peaks"]["int8"] = 1e9
    case = parse_case(raw)
    case = replace(case, model=replace(case.model, quant=QuantSpec(
        "declared-int8", expert_weight_format="int8")))
    operators = groups(target_operators(case, phase="decode", tokens=5, context=16))
    assert len(operators) == 2
    for op in operators:
        metadata = dict(op.metadata)
        assert row_classes(op) == ((1, 6), (2, 2))
        assert metadata["matrix_mapping_domain"] == "integer_expert_groups"
        result = operator_costs(case, (op,), input_ready_ms=7)["operators"][0]
        assert result["program"] == "IntegerGroupedMatmul"
        assert not result["service_before_input"] and not result["live_tasks"]
        assert result["serviced"]["memory@r0:s0"] == pytest.approx(op.hbm_bytes)
        assert result["serviced"]["matrix@r0:s0"] == pytest.approx(op.flops / 1e6)
        execution = Execution(case, CostModel(case))
        prepared, recipe = execution.prepare(op)
        context = execution._context(prepared, 0, "none", recipe=recipe)
        sub = [split_context(context, rows, count).op for rows, count in row_classes(op)]
        for attr in ("flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes"):
            assert sum(getattr(part, attr) for part in sub) == pytest.approx(getattr(op, attr))
        assert [dict(part.metadata)["packed_weight_bytes"] for part in sub] == pytest.approx(
            [metadata["packed_weight_bytes"] * 6 / 8, metadata["packed_weight_bytes"] * 2 / 8])
        assert [dict(part.metadata)["matrix_epilogue"]["row_scale_bytes"] for part in sub] == [24, 16]


def test_group_tail_pays_device_startup_once_and_preserves_input_readiness():
    raw = balanced_case(tokens=5)
    raw["runtime"]["kernels"]["recipes"] = {"balanced_grouped_gate_up": {"kind": "steady_matmul"}}
    op = groups(target_operators(raw, phase="decode", tokens=5))[0]
    base = operator_costs(raw, (op,))["operators"][0]
    raw["runtime"]["kernels"]["invocations"] = [{
        "match": {"recipe": "balanced_grouped_gate_up"},
        "startup_ms": 2, "minimum_ms": 0}]
    timed = operator_costs(raw, (op,), input_ready_ms=10)["operators"][0]
    assert timed["duration_ms"] == pytest.approx(base["duration_ms"] + 2)
    assert timed["serviced"] == base["serviced"]
    assert not timed["service_before_input"]


def test_row_class_preserves_canonical_extended_resource_partition():
    raw = balanced_case(tokens=5)
    raw["machine"]["device"]["rates"]["mte1"] = 1e9
    case = parse_case(raw)
    op = groups(target_operators(case, phase="decode", tokens=5))[0]
    op = replace(op, metadata=op.metadata + (("resource_work", {"mte1": 100, "vector": 100}),))
    execution = Execution(case, CostModel(case))
    prepared, recipe = execution.prepare(op)
    context = execution._context(prepared, 0, "none", recipe=recipe)
    # The canonical native vector includes the explicitly added physical work.
    # Resource roles that duplicate OpSpec fields are normalized by Work.apply.
    parts = [split_context(context, rows, count) for rows, count in row_classes(op)]
    assert sum(dict(part.op.metadata)["resource_work"]["mte1"] for part in parts) == 100
    assert sum(part.op.vector_ops for part in parts) == op.vector_ops + 100
    assert all("mte1_bytes" not in dict(part.op.metadata) for part in parts)


def test_cross_dp_capture_routes_all_padded_rows_and_keeps_empty_source_cohort():
    raw = balanced_case(dp=2, tp=2, ep=4, tokens=1)
    raw["runtime"].update(graph_mode="auto", graph_capture_keys=[4], graph_capture_mode="full")
    graphs = target_graphs(raw, phase="decode", tokens=1, scope="body")
    for ops in graphs.values():
        record = dict(groups(ops)[0].metadata)["routing_realization"]
        assert record["global_tokens"] == 8
        assert sum(record["global_expert_rows"]) == 16
    result = graph_costs(raw, graphs, external_inputs={"analysis_input": 0}, isolated=False)
    assert result["execution"]["live_tasks"] == 0
    raw["workload"]["requests"] = [{"id": "one", "prompt_tokens": 1, "output_tokens": 2}]
    serving = predict(raw)
    assert serving["status"] == "modeled"
    assert serving["execution"]["live_tasks"] == serving["execution"]["live_programs"] == 0


@pytest.mark.parametrize("batch", (1, 2, 4, 8, 16, 32, 64, 128))
@pytest.mark.parametrize("query", (1, 4, 5, 8))
def test_arbitrary_integer_balance_with_group_limited_router_executes(batch, query):
    raw = balanced_case(tp=4, ep=4, experts=24)
    case = parse_case(raw)
    layer = replace(case.model.layer_plans[0], router_groups=4, router_group_topk=1)
    case = replace(case, model=replace(case.model, layer_plans=(layer,)))
    result = predict(case, scope="body", tokens=query, sequences=batch, context=64)
    assert result["body_forward_ms"] > 0
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    operators = target_graphs(case, phase="decode", tokens=query, sequences=batch, scope="body")[0]
    record = dict(groups(operators)[0].metadata)["routing_realization"]
    assert sum(record["global_expert_rows"]) == batch * query * 2
    assert record["rank_assignment_rows"][0] == max(record["rank_assignment_rows"])
