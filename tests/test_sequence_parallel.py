"""Independent contracts for the source-selected Kimi K3 sequence-parallel path."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, predict
from mindie_sim.catalog.ops import comm_op, vector_op
from mindie_sim.prediction.sequence_parallel import (
    DECLARATION,
    allgather,
    close_output,
    input_shard,
    reduce_scatter,
    select,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.workload import RequestView
from test_prediction_system import case


def source_case(*, tp=2, ep=None, dp=1):
    ep = tp if ep is None else ep
    raw = case(dp=dp, tp=tp, ep=ep, prompt=8)
    raw["model"].update(
        ffn="moe", num_experts=8, top_k=2, moe_intermediate_size=32,
        num_heads=tp, num_kv_heads=max(1, tp // 2), head_dim=8,
    )
    parsed = parse_case(raw)
    model = replace(parsed.model, layer_plans=tuple(
        replace(layer, residual_block_size=2, hyper_connections=False, hc_mult=0)
        for layer in parsed.model.layer_plans))
    runtime = replace(parsed.runtime, kernels={"sequence_parallel": dict(DECLARATION)})
    return replace(parsed, model=model, runtime=runtime)


def views(*lengths, phase="prefill"):
    return tuple(RequestView(f"r{index}", length, 0, False, phase)
                 for index, length in enumerate(lengths))


def plan_for(*, tp=2, lengths=(8,), ep=None, dp=1, global_tokens=None, phase="prefill"):
    parsed = source_case(tp=tp, ep=ep, dp=dp)
    batch = views(*lengths, phase=phase)
    plan = select(parsed.runtime, parsed.model, batch,
                  tokens=sum(lengths), global_tokens=global_tokens)
    return parsed, batch, plan


@pytest.mark.parametrize("tp,dtype", [(2, "bf16"), (4, "fp32"), (8, "int8")])
def test_independent_ag_and_rs_wire_recompute_across_tp_widths(tp, dtype):
    parsed, batch, plan = plan_for(tp=tp, lengths=(2 * tp,))
    width = 32
    local, global_rows = 2, 2 * tp
    expected = global_rows * width * dtype_bytes(dtype) * (tp - 1) / tp
    requests = tuple(view.request_id for view in batch)
    gathered = allgather(
        plan=plan, op_id="ag", deps=("shard",), stage=0, layer=0,
        requests=requests, phase="prefill", width=width, dtype=dtype)
    scattered = reduce_scatter(
        plan=plan, op_id="rs", deps=("attention",), stage=0, layer=0,
        requests=requests, phase="prefill", width=width, dtype=dtype)

    # Each physical rank sends this payload in both directions of the path.
    assert gathered.link_bytes == pytest.approx(expected)
    assert scattered.link_bytes == pytest.approx(expected)
    assert gathered.metadata is not None and scattered.metadata is not None
    ag_meta, rs_meta = dict(gathered.metadata), dict(scattered.metadata)
    assert ag_meta["input_shape"] == rs_meta["output_shape"] == (local, width)
    assert ag_meta["output_shape"] == rs_meta["input_shape"] == (global_rows, width)
    assert ag_meta["group_axis"] == rs_meta["group_axis"] == "tp"
    assert ag_meta["steps"] == rs_meta["steps"] == tp - 1
    assert ag_meta["source_partitions"] == 1
    assert gathered.local_shape == (global_rows, width)
    assert scattered.local_shape == (local, width)
    assert gathered.dtype == scattered.dtype == dtype

    # The ordinary topology accounts the same per-rank payload independently.
    topo = Topology(parsed.machine, parsed.runtime.parallel)
    ag_work = topo.accounting(gathered, 0)
    rs_work = topo.accounting(scattered, 0)
    assert set(ag_work.useful_by_rank) == set(rs_work.useful_by_rank) == set(range(tp))
    assert all(value == pytest.approx(expected)
               for value in ag_work.useful_by_rank.values())
    assert all(value == pytest.approx(expected)
               for value in rs_work.useful_by_rank.values())


def test_global_and_local_matrix_norm_dimensions_come_from_the_plan():
    _, _, plan = plan_for(tp=4, lengths=(16,))
    parsed = source_case(tp=4)
    dense_layer = replace(parsed.model.layer_plans[0], ffn="dense")
    moe_layer = replace(parsed.model.layer_plans[1], ffn="moe")
    assert plan.ffn_tokens(dense_layer) == plan.tokens_global == 16
    assert plan.ffn_tokens(moe_layer) == plan.tokens_local == 4
    assert plan.attention_tokens == 16
    assert plan.normalize_tokens == plan.attn_res_tokens == plan.router_tokens == 4
    assert plan.cohort_tokens == 16
    assert plan.source_partitions == 1


def test_multi_request_total_can_be_divisible_when_no_sequence_is():
    _, batch, plan = plan_for(tp=2, lengths=(3, 3, 1, 1))
    assert [view.q_len for view in batch] == [3, 3, 1, 1]
    assert all(view.q_len % 2 for view in batch)
    assert plan.tokens_global == 8
    assert plan.tokens_local == 4
    assert plan.tokens_global == sum(view.q_len for view in batch)


def test_decode_enters_the_same_branch_and_rejects_only_unsupported_rows():
    _, batch, plan = plan_for(tp=2, lengths=(1, 1, 1, 1), phase="decode")
    assert plan.phases == ("decode",)
    assert plan.tokens_global == 4 and plan.tokens_local == 2
    with pytest.raises(ModelingError, match="tokens_divisible_by_tp"):
        plan_for(tp=2, lengths=(1, 1, 1), phase="decode")


def test_public_serving_non_draft_decode_keeps_the_single_committed_token():
    parsed = source_case(tp=2)
    declaration = {**DECLARATION, "padding": "right_to_tp"}
    parsed = replace(parsed, runtime=replace(parsed.runtime, kernels={
        "sequence_parallel": declaration}))
    result = predict(parsed, trace_limit=4)
    row = result["requests"][0]
    assert row["status"] == "served"
    assert row["generated_tokens"] == 3 and row["iterations"] == 3
    assert row["itl"]["count"] == 2 and row["finish_ms"] > row["first_token_abs_ms"]
    assert [step["valid_tokens"] for step in result["traces"]] == [8, 1, 1]
    assert row["speculative"]["proposed_tokens"] == 0
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


def test_cross_dp_cohort_keeps_unequal_totals_and_requires_complete_total():
    _, batch, plan = plan_for(
        tp=2, ep=4, dp=2, lengths=(6,), global_tokens=10)
    assert plan.tokens_global == 6
    assert plan.tokens_local == 3
    # Ten unique cohort tokens are not forced to 2 * this replica's six rows.
    assert plan.cohort_tokens == 10
    assert plan.ep == 4 and plan.dp == 2

    parsed = source_case(tp=2, ep=4, dp=2)
    with pytest.raises(ModelingError, match="cross_dp_requires_complete_cohort_tokens"):
        select(parsed.runtime, parsed.model, batch, tokens=6)
    with pytest.raises(ModelingError, match="cohort_tokens_below_local_full"):
        select(parsed.runtime, parsed.model, batch, tokens=6, global_tokens=4)
    with pytest.raises(ModelingError, match="cohort_tokens_not_divisible_by_tp"):
        select(parsed.runtime, parsed.model, batch, tokens=6, global_tokens=9)

    local_cohort = source_case(tp=2, ep=2)
    with pytest.raises(ModelingError, match="outside_ep_cohort"):
        select(local_cohort.runtime, local_cohort.model, batch,
               tokens=6, global_tokens=8)


def test_idle_dp_source_keeps_unique_expert_cohort_without_empty_tp_collective():
    parsed = source_case(tp=2, ep=4, dp=2)
    active = select(parsed.runtime, parsed.model, views(8),
                    tokens=8, global_tokens=8)
    idle = select(parsed.runtime, parsed.model, (), tokens=0, global_tokens=8)
    assert idle.tokens_global == idle.tokens_local == 0
    assert active.cohort_tokens == idle.cohort_tokens == 8
    assert sum(plan.tokens_local * plan.tp for plan in (active, idle)) == 8
    assert active.source_partitions == idle.source_partitions == 1
    for builder in (allgather, reduce_scatter):
        with pytest.raises(ModelingError, match="empty_replica_has_no_tp_collective"):
            builder(plan=idle, op_id="empty", deps=("input",), stage=0,
                    layer=0, requests=(), phase="prefill", width=32, dtype="bf16")


def test_actual_kimi_case_retains_tp_shards_and_unique_cross_dp_rows():
    path = (Path(__file__).resolve().parents[1] / "examples" / "cases"
            / "kimi-k3-a5-current-checkpoint-conditional.json")
    raw = json.loads(path.read_text())
    raw["runtime"].setdefault("kernels", {})["sequence_parallel"] = dict(DECLARATION)
    parsed = parse_case(raw)
    plans = tuple(select(parsed.runtime, parsed.model, views(rows) if rows else (),
                         tokens=rows, global_tokens=32768)
                  for rows in (8192, 16384, 8192, 0))
    assert (plans[0].tp, plans[0].ep, plans[0].dp) == (8, 32, 4)
    assert [plan.tokens_local for plan in plans] == [1024, 2048, 1024, 0]
    assert sum(plan.tokens_local * plan.tp for plan in plans) == 32768
    assert {plan.cohort_tokens for plan in plans} == {32768}
    assert {plan.source_partitions for plan in plans} == {1}


@pytest.mark.parametrize("field,value,match", [
    ("pp", 2, "requires_pp1"),
    ("cp", 2, "requires_cp1"),
    ("pcp", 2, "requires_pcp1"),
])
def test_source_selected_branch_rejects_unsupported_parallel_axes(field, value, match):
    parsed = source_case(tp=2)
    changes = {field: value}
    if field == "pcp":
        changes["cp_plan"] = "pcp"
    runtime = replace(parsed.runtime, parallel=replace(
        parsed.runtime.parallel, **changes))
    with pytest.raises(ModelingError, match=match):
        select(runtime, parsed.model, views(8), tokens=8)


def test_declaration_and_structural_domain_are_exact():
    parsed = source_case(tp=2)
    wrong = dict(DECLARATION, padding="pad")
    with pytest.raises(ModelingError, match="option_unsupported:padding"):
        select(replace(parsed.runtime, kernels={"sequence_parallel": wrong}),
               parsed.model, views(8), tokens=8)
    with pytest.raises(ModelingError, match="declaration_unknown:extra"):
        select(replace(parsed.runtime, kernels={
            "sequence_parallel": dict(DECLARATION, extra=True)}),
            parsed.model, views(8), tokens=8)

    no_residual = replace(parsed.model, layer_plans=tuple(
        replace(layer, residual_block_size=0) for layer in parsed.model.layer_plans))
    with pytest.raises(ModelingError, match="attn_residual_block_size"):
        select(parsed.runtime, no_residual, views(8), tokens=8)
    hyper = replace(parsed.model, layer_plans=tuple(
        replace(layer, hyper_connections=True) for layer in parsed.model.layer_plans))
    with pytest.raises(ModelingError, match="hyperconnections_unsupported"):
        select(parsed.runtime, hyper, views(8), tokens=8)
    draft = replace(parsed.model, layer_plans=(
        replace(parsed.model.layer_plans[0], scope="draft"),))
    with pytest.raises(ModelingError, match="requires_body_scope"):
        select(parsed.runtime, draft, views(8), tokens=8)


def test_padding_speculation_and_partial_rows_never_silently_disable():
    parsed = source_case(tp=2)
    with pytest.raises(ModelingError, match="no_padding_domain"):
        select(parsed.runtime, parsed.model,
               (replace(views(8)[0], padding=True),), tokens=8)
    with pytest.raises(ModelingError, match="speculative_field_unsupported"):
        select(parsed.runtime, parsed.model,
               (replace(views(8)[0], draft_proposal_tokens=1),), tokens=8)
    with pytest.raises(ModelingError, match="tokens_must_equal_sum_q_len"):
        select(parsed.runtime, parsed.model, views(8), tokens=9)
    with pytest.raises(ModelingError, match="requires_positive_q_len"):
        select(parsed.runtime, parsed.model, views(0), tokens=8)
    with pytest.raises(ModelingError, match="tokens_divisible_by_tp"):
        select(parsed.runtime, parsed.model, views(7), tokens=7)


def test_input_shard_is_a_zero_payload_view_with_full_rows_as_input():
    _, batch, plan = plan_for(tp=2, lengths=(8,))
    shard = input_shard(
        plan=plan, op_id="shard", deps=("full_embedding",), stage=0, layer=0,
        requests=tuple(view.request_id for view in batch), phase="prefill",
        width=32, dtype="bf16")
    assert shard.family == "barrier" and shard.link_bytes == 0
    metadata = dict(shard.metadata)
    assert metadata["shard_materialization"] == "view"
    assert metadata["physical_payload_bytes"] == 0
    assert metadata["zero_payload_reason"] == "declared_view_no_padding"
    assert metadata["input_shape"] == (8, 32)
    assert metadata["output_shape"] == (4, 32)
    assert shard.deps == ("full_embedding",)
    assert metadata["input_dependency"] == "full_embedding"
    assert metadata["source_assignment_partitions"] == 1
    with pytest.raises(ModelingError, match="requires_full_embedding_dependency"):
        input_shard(
            plan=plan, op_id="bad", deps=(), stage=0, layer=0,
            requests=("r0",), phase="prefill", width=32, dtype="bf16")


def test_final_aux_channels_scale_payload_and_true_shape():
    _, batch, plan = plan_for(tp=4, lengths=(16,))
    packed = allgather(
        plan=plan, op_id="final_packed", deps=("final",), stage=0, layer=-1,
        requests=tuple(view.request_id for view in batch), phase="prefill",
        width=32, dtype="bf16", channel_multiplier=3)
    metadata = dict(packed.metadata)
    assert metadata["input_shape"] == (4, 96)
    assert metadata["output_shape"] == (16, 96)
    expected = 16 * 96 * dtype_bytes("bf16") * 3 / 4
    assert packed.link_bytes == pytest.approx(expected)


def test_close_output_replaces_only_the_exact_tp_allreduce_tail():
    _, batch, plan = plan_for(tp=4, lengths=(16,))
    requests = tuple(view.request_id for view in batch)
    ag = allgather(
        plan=plan, op_id="mlp_ag", deps=("mlp_shard",), stage=0, layer=0,
        requests=requests, phase="prefill", width=32, dtype="bf16")
    allreduce = comm_op(
        op_id="mlp_reduce", deps=(ag.op_id,), stage=0, layer=0,
        requests=requests, family="allreduce", phase="prefill",
        recipe="ring_allreduce", link_bytes=2 * 3 / 4 * 16 * 32 * 2,
        shape=(16, 32), metadata=(("group_axis", "tp"), ("steps", 6)))
    allreduce = replace(allreduce, stream="aux")
    ops, last = close_output(
        (ag, allreduce), allreduce.op_id, plan=plan, width=32, dtype="bf16")
    assert last != allreduce.op_id
    assert [op.family for op in ops] == ["allgather", "reduce_scatter"]
    assert ops[-1].deps == (ag.op_id,)
    assert ops[-1].stream == "aux"
    assert ops[-1].link_bytes == pytest.approx(ag.link_bytes)
    assert not any(op.family == "allreduce" for op in ops)

    with pytest.raises(ModelingError, match="requires_exact_tail"):
        close_output((allreduce, ag), allreduce.op_id,
                     plan=plan, width=32, dtype="bf16")
    non_tp = replace(allreduce, metadata=(("group_axis", "ep"),))
    with pytest.raises(ModelingError, match="requires_tp_allreduce_tail"):
        close_output((non_tp,), non_tp.op_id, plan=plan, width=32, dtype="bf16")
    two_deps = replace(allreduce, deps=(ag.op_id, "other"))
    with pytest.raises(ModelingError, match="requires_sole_producer"):
        close_output((two_deps,), two_deps.op_id, plan=plan, width=32, dtype="bf16")


def test_ordinary_graph_executes_dependencies_and_releases_input_ready_state():
    parsed, batch, plan = plan_for(tp=4, lengths=(16,))
    requests = tuple(view.request_id for view in batch)
    shard = input_shard(
        plan=plan, op_id="shard", deps=("full_embedding",), stage=0, layer=0,
        requests=requests, phase="prefill", width=32, dtype="bf16")
    ag = allgather(
        plan=plan, op_id="attention_ag", deps=(shard.op_id,), stage=0, layer=0,
        requests=requests, phase="prefill", width=32, dtype="bf16")
    attention = vector_op(
        op_id="attention", deps=(ag.op_id,), stage=0, layer=0,
        requests=requests, elements=16 * 32, ops_per_element=2, dtype="bf16",
        family="rmsnorm", phase="prefill", recipe="rmsnorm", shape=(16, 32))
    rs = reduce_scatter(
        plan=plan, op_id="attention_rs", deps=(attention.op_id,), stage=0, layer=0,
        requests=requests, phase="prefill", width=32, dtype="bf16")
    consumer = vector_op(
        op_id="consumer", deps=(rs.op_id,), stage=0, layer=0,
        requests=requests, elements=4 * 32, ops_per_element=1, dtype="bf16",
        family="rmsnorm", phase="prefill", recipe="rmsnorm", shape=(4, 32))

    ring = replace(parsed.runtime, kernels={
        "sequence_parallel": dict(DECLARATION),
        "families": {
            "allgather": {"kind": "ring_collective"},
            "reduce_scatter": {"kind": "ring_collective"},
        },
    })
    report = graph_costs(
        replace(parsed, runtime=ring), (shard, ag, attention, rs, consumer),
        external_inputs={"full_embedding": 3}, trace_limit=200)
    rows = {row["id"]: row for row in report["timeline"]}
    assert rows["r0:shard"]["ready_ms"] >= 3
    assert rows["r0:attention_ag"]["ready_ms"] >= rows["r0:shard"]["complete_ms"]
    assert rows["r0:attention"]["ready_ms"] >= rows["r0:attention_ag"]["complete_ms"]
    assert rows["r0:attention_rs"]["ready_ms"] >= rows["r0:attention"]["complete_ms"]
    assert rows["r0:consumer"]["ready_ms"] >= rows["r0:attention_rs"]["complete_ms"]
    assert report["execution"]["live_tasks"] == 0
    assert report["execution"]["live_programs"] == 0

    topo = Topology(parsed.machine, parsed.runtime.parallel)
    expected = (sum(topo.accounting(ag, 0).wire_by_resource.values())
                + sum(topo.accounting(rs, 0).wire_by_resource.values()))
    assert report["execution"]["serviced"]["link:fabric"] == pytest.approx(expected)
    assert report["execution"]["work"]["link_bytes"] == pytest.approx(
        ag.link_bytes + rs.link_bytes)
