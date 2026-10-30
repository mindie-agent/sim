"""Ordinary MLA decode caller, source launch boundaries and batch work."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def mla_case(*, splits=3):
    raw = case(prompt=4, output=3)
    raw["model"].update(
        layers=1, attention="mla", num_kv_heads=1, head_dim=48,
        mla={"q_lora_rank": 32, "kv_lora_rank": 256, "qk_nope_head_dim": 16,
             "qk_rope_head_dim": 32, "v_head_dim": 16})
    raw["runtime"]["kernels"] = {"lowerings": {"mla_decode": {
        "implementation": "vllm_triton_mla_decode_tuple",
        "options": {"num_kv_splits": splits},
    }}}
    return raw


def test_batched_decode_keeps_caller_materializations_and_two_kernel_join():
    raw = mla_case()
    graphs = target_graphs(raw, phase="decode", tokens=1, sequences=2, context=16)
    ops = {op.op_id: op for op in graphs[0]}
    packed = ops["s0.l0.mla_query_concat"]
    assert packed.deps == ("s0.l0.q_absorb", "s0.l0.rotary")
    assert packed.mte2_bytes == packed.mte3_bytes == 2 * 4 * (256 + 32) * 2
    assert ops["s0.l0.mla_output_zero"].mte3_bytes == 2 * 4 * 256 * 2
    assert ops["s0.l0.mla_lse_zero"].mte3_bytes == 2 * 4 * 2
    call = ops["s0.l0.latent_attention_decode"]
    assert call.source_boundary == "call"
    assert set(call.deps) == {
        packed.op_id, "s0.l0.mla_output_zero", "s0.l0.mla_lse_zero", "s0.l0.kv_store"}
    assert dict(call.metadata)["seq_lens"] == (17, 17)
    assert dict(call.metadata)["semantic_metadata"]["logical_pairs"] == 34
    assert ops["s0.l0.v_expand"].deps == (call.op_id,)
    report = graph_costs(raw, graphs, input_ready_ms=7, trace_limit=1000)
    calls = [row for row in report["timeline"] if row["source_boundary"] == "kernel"]
    assert [row["op_id"].rsplit("/", 1)[-1] for row in calls] == [
        "stage1_grouped_decode", "stage2_split_merge"]
    assert calls[0]["id"] in calls[1]["predecessors"]
    joined = next(row for row in report["timeline"] if row["op_id"] == call.op_id)
    expanded = next(row for row in report["timeline"] if row["op_id"] == "s0.l0.v_expand")
    assert joined["complete_ms"] == calls[1]["complete_ms"]
    assert expanded["ready_ms"] >= joined["complete_ms"]
    assert all(row["start_ms"] >= 7 for row in report["events"] if row["demand"])
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert not any(report["execution"]["live_allocated"].values())


def test_split_tiles_and_scratch_follow_integer_contexts_not_semantic_pair_count():
    first, second = mla_case(splits=1), mla_case(splits=3)
    reports = []
    for raw in (first, second):
        ops = target_operators(raw, phase="decode", tokens=1, context=16)
        reports.append(operator_costs(raw, ops))
    phases = []
    for report in reports:
        phases.append({row["operator"]["metadata"]["kernel_phase"]: row["operator"]
                       for row in report["operators"] if row["operator"].get("source_boundary")})
    # Seventeen keys give one key tile for one split and three restarted tiles
    # for three splits. The MMA executes 16 head lanes and 32 key lanes.
    one_tile = 2 * 16 * 32 * (256 + 32 + 256)
    assert phases[0]["stage1"]["flops"] == one_tile
    assert phases[1]["stage1"]["flops"] == 3 * one_tile
    assert phases[1]["stage1"]["mte3_bytes"] == 3 * 4 * (256 + 1) * 4
    assert phases[1]["stage2"]["mte2_bytes"] == 4 * 4 + 3 * 4 * (256 + 1) * 4
    assert phases[1]["stage2"]["mte3_bytes"] == 4 * (256 + 1) * 2
    assert phases[1]["stage1"]["metadata"]["semantic_metadata"]["logical_pairs"] == 17


def test_decode_selection_works_in_serving_while_prefill_remains_analytical():
    raw = mla_case()
    prefill = target_operators(raw, phase="prefill", tokens=4)
    assert not any(op.plan_required for op in prefill)
    result = predict(raw)
    assert result["requests"][0]["generated_tokens"] == 3
    assert result["execution"]["recipes"]["triton_decode_grouped_mla"] == 4
    assert result["execution"]["recipes"]["mla_query_concat"] == 2
    assert result["parameters"]["calibration"] == "none"


def test_ragged_batch_keeps_each_requests_split_tails_at_equal_total_context():
    parsed = parse_case(mla_case(splits=2))
    flops = []
    for lengths in ((1, 33), (17, 17)):
        views = tuple(RequestView(str(i), 1, length - 1, True, "decode")
                      for i, length in enumerate(lengths))
        work = Frontend()(IterationPlan(views), parsed.model, parsed.runtime, parsed.machine)
        call = next(op for op in work.ops if op.source_boundary == "call")
        assert dict(call.metadata)["seq_lens"] == lengths
        stage1 = operator_costs(parsed, (call,))["operators"][0]["operator"]
        flops.append(stage1["flops"])
    # 1 key has one nonempty split; 33 keys restart two tiles. Two sequences
    # of 17 keys restart four tiles despite the same 34 logical key positions.
    one_tile = 2 * 16 * 32 * (256 + 32 + 256)
    assert flops == [3 * one_tile, 4 * one_tile]


def test_local_heads_and_empty_splits_keep_the_source_dispatch():
    raw = mla_case(splits=8)
    raw["runtime"]["tp"] = 2
    raw["machine"]["num_devices"] = 2
    raw["machine"]["links"][0]["members"] = [0, 1]
    ops = target_operators(raw, phase="decode", tokens=1, context=0)
    call = next(op for op in ops if op.source_boundary == "call")
    plan = dict(call.metadata)["plan"]
    assert plan["heads"] == 2 and plan["nonempty_splits"] == (1,)
    report = operator_costs(raw, (call,))
    stage1, stage2 = [row["operator"] for row in report["operators"]]
    assert stage1["mte3_bytes"] == 2 * (256 + 1) * 4
    assert stage2["mte2_bytes"] == 2 * 4 + 2 * (256 + 1) * 4
    assert stage1["metadata"]["plan"]["stage1_programs"] == 8


def test_source_decode_does_not_erase_sparse_or_padded_or_mixed_geometry():
    parsed = parse_case(mla_case())
    sparse = replace(parsed.model.layer_plans[0], window=8)
    with pytest.raises(ModelingError, match="dense_latent_cache"):
        target_operators(replace(parsed, model=replace(parsed.model, layer_plans=(sparse,))),
                         phase="decode", tokens=1, context=16)
    padded = replace(parsed, runtime=replace(parsed.runtime, graph_buckets=(4,)))
    with pytest.raises(ModelingError, match="unpadded_single_token"):
        predict(padded)
    mixed = IterationPlan((RequestView("a", 1, 16, True, "decode"),
                           RequestView("b", 3, 0, False, "prefill")))
    with pytest.raises(ModelingError, match="unpadded_single_token"):
        Frontend()(mixed, parsed.model, parsed.runtime, parsed.machine)
    with pytest.raises(ModelingError, match="unpadded_single_token"):
        target_operators(parsed, phase="decode", tokens=2, context=16)
