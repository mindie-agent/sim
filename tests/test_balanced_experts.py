"""Balanced expert arithmetic, cohort ownership and ordinary-path controls."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs, predict, target_graphs, target_operators
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def balanced_case(*, tp=1, dp=1, ep=1, experts=8, tokens=32):
    raw = case(tp=tp, dp=dp, ep=ep, prompt=tokens, output=1)
    raw["model"].update(layers=1, ffn="moe", num_experts=experts,
                        top_k=2, moe_intermediate_size=64)
    raw["runtime"].update(routing={"policy": "balanced"},
                          max_num_batched_tokens=max(16, tokens))
    raw["runtime"]["kernels"] = {
        "families": {"alltoall": {"kind": "pairwise_collective"}}}
    return raw


def groups(ops):
    return [op for op in ops if op.family == "grouped_gemm"]


def test_eight_equal_experts_keep_one_grouped_boundary_and_all_tp_work():
    raw = balanced_case(tp=4)
    ops = target_operators(raw, phase="prefill", tokens=32)
    expert_ops = groups(ops)
    assert len(expert_ops) == 2
    assert sum(op.flops for op in expert_ops) == 196608
    assert sum(op.flops for op in expert_ops) * 4 == 786432
    for op in expert_ops:
        meta = dict(op.metadata)
        assert meta["matrix_groups"] == 8
        assert meta["routing_realization"]["local_expert_rows"] == (8,) * 8
        assert meta["matrix_mapping_domain"] == "equal_expert_groups"
        assert meta["routing_realization"]["token_identities"] == "not_declared"
    report = operator_costs(raw, expert_ops)
    assert report["operator_occurrences"] == 2
    assert all(row["program"] == "MappedMatmul" for row in report["operators"])
    # This fixture declares 1e9 FLOP/s and exposes normalized matrix service.
    assert sum(sum(value for key, value in row["serviced"].items() if key.startswith("matrix@"))
               for row in report["operators"]) == pytest.approx(196608 / 1e6)
    assert all(not row["service_before_input"] and not row["live_tasks"]
               for row in report["operators"])


def test_sparse_balance_reads_only_two_live_experts_and_preserves_empty_ids():
    raw = balanced_case(tokens=1)
    ops = groups(target_operators(raw, phase="decode", tokens=1, context=16))
    assert len(ops) == 2
    for op in ops:
        meta = dict(op.metadata)
        assert meta["matrix_groups"] == 2
        assert meta["routing_realization"]["global_expert_rows"] == (1, 1, 0, 0, 0, 0, 0, 0)
        assert meta["packed_weight_bytes"] == 2 * op.local_shape[1] * op.local_shape[2] * 2
        assert op.local_shape[0] == 2


def test_cross_dp_uses_one_cohort_and_partitions_replicated_tp_sources():
    raw = balanced_case(dp=4, tp=2, ep=4, tokens=4)
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="body")
    assert tuple(graphs) == (0, 1)
    for ops in graphs.values():
        for op in groups(ops):
            meta = dict(op.metadata)
            assert meta["matrix_groups"] == 2
            assert op.local_shape[0] == 4
            assert meta["routing_realization"]["global_expert_rows"] == (2,) * 8
        exchanges = [op for op in ops if op.family == "alltoall"]
        assert len(exchanges) == 2
        assert all(op.link_bytes == 192 for op in exchanges)
    report = graph_costs(raw, graphs, external_inputs={"analysis_input": 0})
    exchanges = [row for row in report["isolated"]["operators"]
                 if row["operator"]["family"] == "alltoall"]
    # Four physical sources, each sends 3 remote assignments of 64 bytes.
    assert sum(row["communication"]["wire_bytes"] * row["occurrences"]
               for row in exchanges) == 2 * 768
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert report["bounds"]["combined_ms"] <= report["duration_ms"] + 1e-9


def test_balanced_graph_and_serving_execute_the_same_full_cohort():
    raw = balanced_case(dp=2, tp=2, ep=4, tokens=4)
    raw["workload"]["requests"] = [
        {"id": name, "prompt_tokens": 4, "output_tokens": 1} for name in ("first", "second")]
    graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=4))
    serving = predict(raw)
    wire = lambda result: {key: value for key, value in result["execution"]["serviced"].items()
                           if key.startswith("link:")}
    assert serving["status"] == "modeled"
    assert wire(graph) == wire(serving)
    assert graph["execution"]["live_tasks"] == serving["execution"]["live_tasks"] == 0


def test_quantized_balanced_experts_retain_payload_and_scale_producers():
    parsed = parse_case(balanced_case(tokens=4))
    parsed = replace(parsed, model=replace(
        parsed.model, quant=QuantSpec("declared-int8", expert_weight_format="int8")))
    ops = target_operators(parsed, phase="prefill", tokens=4)
    by_id = {op.op_id: op for op in ops}
    for op in groups(ops):
        meta = dict(op.metadata)
        producer = by_id[op.deps[0]]
        rows, _, k = op.local_shape
        assert meta["program_kind"] == "matmul"
        assert producer.family == "quantize"
        assert producer.mte2_bytes == rows * k * 2
        assert producer.mte3_bytes == rows * k + 4 * rows
        # The catalog retains scale+offset storage, unlike the historical
        # no-offset source API. Output casts and two scales are now explicit.
        assert meta["matrix_epilogue"]["row_scale_bytes"] == 4 * rows
        assert meta["matrix_epilogue"]["column_scale_bytes"] == 8 * op.local_shape[1] * 8
        assert op.vector_ops == 4 * rows * op.local_shape[1]


def test_remainder_rank_skew_and_idle_sources_keep_integer_counts():
    mixed = groups(target_operators(balanced_case(tokens=5), phase="prefill", tokens=5))
    assert dict(mixed[0].metadata)["matrix_group_rows"] == (2, 2, 1, 1, 1, 1, 1, 1)
    skew = groups(target_operators(balanced_case(tp=4, ep=4, tokens=1), phase="decode", tokens=1))
    record = dict(skew[0].metadata)["routing_realization"]
    assert record["rank_assignment_rows"] == (1, 1, 0, 0)
    assert record["representative_rank"] == 0
    parsed = parse_case(balanced_case(dp=2, ep=2, tokens=4))
    views = (RequestView("only-active", 4, 0, True, "prefill"),)
    work = Frontend()(IterationPlan(views), parsed.model, parsed.runtime, parsed.machine,
                      global_tokens=4, global_views=views)
    assert sum(dict(groups(work.ops)[0].metadata)["matrix_group_rows"]) == 4
    with pytest.raises(ValueError, match="complete_dp_cohort"):
        target_operators(parsed, phase="prefill", tokens=4)
