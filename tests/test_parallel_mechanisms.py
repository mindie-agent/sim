"""Brute-force context ownership and executable communication controls."""
from dataclasses import replace

import pytest

from mindie_sim import Case, predict
from mindie_sim.catalog.context_parallel import split_causal_pairs, split_pcp_pairs, pcp_rank_chunk_ranges
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.feedforward import RoutingPolicy
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def test_context_partition_matches_every_visible_query_key_pair():
    for query, context, degree, block, window in (
        (1, 0, 4, 128, 0), (9, 11, 3, 2, 5), (17, 8, 4, 3, 0), (3, 129, 2, 128, 0),
    ):
        actual = split_causal_pairs(query, context, degree, block, window=window)
        expected = [0] * degree
        for q in range(context, context + query):
            for k in range(max(0, q - window + 1) if window else 0, q + 1):
                expected[k // block % degree] += 1
        assert list(actual.per_rank) == expected
        assert actual.total == sum(expected)


def test_pcp_zigzag_partitions_queries_and_preserves_causal_triangle():
    for query in (1, 5, 16, 19):
        degree, context = 3, 7
        split = split_pcp_pairs(query, context, degree)
        pairs, positions = [], []
        for rank in range(degree):
            local = [q for start, end in pcp_rank_chunk_ranges(query, degree, rank) for q in range(start, end)]
            positions += local
            pairs.append(sum(context + q + 1 for q in local))
        assert sorted(positions) == list(range(query))
        assert tuple(pairs) == split.per_rank
        assert sum(pairs) == query * context + query * (query + 1) / 2


def test_pcp_public_program_splits_queries_and_adds_real_kv_collectives():
    raw = case(prompt=16, output=2)
    raw["machine"]["num_devices"] = 2
    raw["machine"]["links"][0]["members"] = [0, 1]
    raw["runtime"]["pcp"] = 2
    parsed = Case.from_mapping(raw)
    plan = IterationPlan((RequestView("r", 16, 0, True, "prefill"),))
    work = Frontend()(plan, parsed.model, parsed.runtime, parsed.machine)
    attn = [op for op in work.ops if op.family == "attention"]
    # A symmetric zigzag division of the sixteen query rows gives 68 pairs.
    assert sum(op.flops for op in attn) == 2 * 4 * 68 * 4 * 8
    assert any(op.recipe == "pcp_kv_gather" and op.link_bytes > 0 for op in work.ops)
    norm = next(op for op in work.ops if op.op_id == "final_norm")
    selection = next(op for op in work.ops if op.op_id == "select_hidden")
    gather = next(op for op in work.ops if op.op_id == "pcp_hidden_gather")
    assert norm.local_shape == (8, parsed.model.hidden_size)
    assert selection.local_shape == (1, parsed.model.hidden_size)
    assert selection.deps == (norm.op_id,) and gather.deps == (selection.op_id,)
    result = predict(parsed)
    assert result["status"] == "modeled"
    assert result["execution"]["live_programs"] == result["execution"]["live_tasks"] == 0


def test_cp_reads_whole_owned_blocks_and_window_tile_union():
    from mindie_sim.prediction.attention import AttentionPolicy
    raw = case(tp=4, prompt=6, output=1)
    raw["model"]["num_kv_heads"] = 1
    raw["runtime"].update(cp=2, kv_block_size=2)
    parsed = Case.from_mapping(raw)
    model = replace(parsed.model, layer_plans=tuple(
        replace(layer, window=5) for layer in parsed.model.layer_plans))
    view = RequestView("r", 6, 3, True, "prefill")
    work = Frontend(attention_policy=AttentionPolicy(query_tile=3))(
        IterationPlan((view,)), model, parsed.runtime, parsed.machine)
    attention = next(op for op in work.ops if op.family == "attention")
    per_rank = [0, 0]
    for tile_start in (3, 6):
        visible = set()
        for query in range(tile_start, tile_start + 3):
            visible.update(range(max(0, query - 4), query + 1))
        for key in visible:
            per_rank[key // 2 % 2] += 1
    input_bytes = 6 * 1 * 8 * 2
    expected_kv = max(per_rank) * 2 * 1 * 8 * 2
    assert attention.mte2_bytes == expected_kv + input_bytes


def test_routing_assignment_conservation_and_empty_expert_weight_reads():
    parsed = Case.from_mapping(case())
    layer = replace(parsed.model.layer_plans[0], num_experts=4, top_k=2)
    policy = RoutingPolicy()
    rows, active = policy.local(parsed.model, layer, ParallelSpec(), 3)
    assert rows == 6
    assert active == pytest.approx(4 * (1 - .5 ** 3))
    assert policy.local(parsed.model, layer, ParallelSpec(), 0) == (0, 0)
    weighted = replace(parsed.model, expert_routing=(.9, .05, .04, .01))
    assert policy.local(weighted, layer, ParallelSpec(), 7)[0] == pytest.approx(14)
    with pytest.raises(ValueError, match="routing_support"):
        policy.local(replace(parsed.model, expert_routing=(1, 0, 0, 0)), layer, ParallelSpec(), 7)


def test_mc2_executes_finite_chunks_and_conserves_wire_volume():
    raw = case(tp=2, ep=2, prompt=16, output=1)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    plain = predict(raw, event_limit=2000)
    raw["runtime"].update(moe_recipe="mc2", mc2_chunks=4)
    chunked = predict(raw, event_limit=2000)
    assert chunked["status"] == "modeled"
    assert plain["execution"]["work"]["link_bytes"] == chunked["execution"]["work"]["link_bytes"]
    assert plain["execution"]["work"]["flops"] == chunked["execution"]["work"]["flops"]
    assert plain["metrics"] != chunked["metrics"]
    assert chunked["execution"]["live_tasks"] == chunked["execution"]["live_programs"] == 0
    assert any(".chunk3." in event["task"] for event in chunked["events"])
