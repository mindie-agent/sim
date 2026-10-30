"""Sequence shards preserve global expert work and physical return traffic."""
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.cohort_traffic import bind_cohort_traffic
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.sequence_parallel import DECLARATION
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def small_case():
    raw = case(dp=2, tp=2, ep=4)
    raw["model"].update(
        ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32,
        num_shared_experts=1,
        layer_plans=[
            {"index": 0, "attention": "gqa", "ffn": "dense",
             "num_heads": 4, "num_kv_heads": 2, "head_dim": 8,
             "intermediate_size": 64, "residual_block_size": 2},
            {"index": 1, "attention": "gqa", "ffn": "moe", "num_experts": 4,
             "num_heads": 4, "num_kv_heads": 2, "head_dim": 8,
             "top_k": 2, "moe_intermediate_size": 32, "num_shared_experts": 1,
             "residual_block_size": 2},
        ])
    raw["runtime"].update(graph_mode="none", routing={"policy": "balanced"},
                          kernels={"sequence_parallel": dict(DECLARATION)})
    return parse_case(raw)


def one(ops, recipe):
    values = [op for op in ops if op.recipe == recipe]
    assert len(values) == 1
    return values[0]


@pytest.mark.parametrize("counts", [(4, 4), (8, 0)])
def test_full_cohort_work_survives_sequence_shards_and_idle_source(counts):
    parsed = small_case()
    views = tuple((RequestView(f"r{i}", count, 0, True, "prefill"),) if count else ()
                  for i, count in enumerate(counts))
    global_views = tuple(view for group in views for view in group)
    graphs = {}
    for replica, local in enumerate(views):
        work = Frontend()(IterationPlan(local), parsed.model, parsed.runtime, parsed.machine,
                          global_tokens=sum(counts), global_views=global_views)
        graphs[replica] = tuple(op for op in work.ops
                                if op.family not in {"sampling", "token_commit", "acceptance"})
    graphs = bind_cohort_traffic(parsed, graphs)
    for replica, ops in graphs.items():
        full, local = counts[replica], counts[replica] // 2
        grouped = [op for op in ops if op.family == "grouped_gemm"]
        assert len(grouped) == 2
        assert all(op.local_shape[0] == sum(counts) * 2 // 4 for op in grouped)
        dispatch = one(ops, "expert_dispatch")
        assert dict(dispatch.metadata)["source_assignment_partitions"] == 1
        assert dispatch.link_bytes == local * 2 * 32 * 2 * 3 / 4
        combine = one(ops, "expert_combine")
        assert dict(combine.metadata)["peer_traffic"]["direction"] == "transpose"
        if not full:
            assert combine.link_bytes > 0
            assert not any(op.family in {"allgather", "reduce_scatter"} for op in ops)
            assert not any(dict(op.metadata).get("ffn_category") == "shared_expert"
                           for op in ops)
            continue
        assert not any(op.family == "allreduce" for op in ops)
        shared = [op for op in ops if dict(op.metadata).get("ffn_category") == "shared_expert"]
        assert all(op.local_shape[0] == full for op in shared)
        router = one(ops, "router")
        assert router.local_shape[0] == local
        dense = [op for op in ops if op.recipe == "dense_gate_up"]
        assert len(dense) == 2 and all(op.local_shape[0] == full for op in dense)
        source = {op.op_id: op for op in ops}
        for op in ops:
            if op.family == "qkv_projection":
                assert op.local_shape[0] == full
                assert source[op.deps[0]].family == "allgather"
        assert sum(op.family == "reduce_scatter" for op in ops) == 4
    result = graph_costs(parsed, graphs)
    assert result["duration_ms"] > 0
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert not any(result["execution"]["live_allocated"].values())


def test_real_kimi_prefill_wrapper_uses_local_rows_without_shrinking_attention_or_experts():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/cases/"
                     "kimi-k3-a5-current-checkpoint-conditional.json").read_text())
    raw["runtime"]["graph_mode"] = "eager"
    raw["runtime"]["kernels"] = {
        "sequence_parallel": dict(DECLARATION),
        "lowerings": {"matrix_wrapper": {
            "implementation": "vllm_ascend_mxfp8_single_group_eager",
            "options": {"projection_recipes": ["latent_projection"]},
        }},
    }
    graphs = target_graphs(raw, phase="prefill", tokens=8192, scope="model")
    assert len(graphs) == 4
    for ops in graphs.values():
        latent = [op for op in ops if op.recipe == "latent_projection"]
        assert len(latent) == 184
        assert all(op.local_shape[0] == 1024 for op in latent)
        assert all("matrix_wrapper" in dict(op.metadata) for op in latent)
        grouped = [op for op in ops if op.family == "grouped_gemm"]
        assert all(op.local_shape[0] == 16384 for op in grouped)
        attention = [op for op in ops if op.family == "attention"]
        assert attention and all(op.local_shape[0] == 8192 for op in attention)
        shared = [op for op in ops
                  if dict(op.metadata).get("ffn_category") == "shared_expert"]
        assert shared and all(op.local_shape[0] == 8192 for op in shared)
