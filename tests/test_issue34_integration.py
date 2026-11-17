"""Compact end-to-end composition, prediction and capacity agreement."""
from __future__ import annotations

import json

from mindie_sim.compose import compose_case
from mindie_sim.prediction import predict
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import capacity_window
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case
from mindie_sim.schema.operators import IterationWork, OpSpec


def linear_work(plan, model, runtime, machine, **_):
    tokens = sum(view.q_len for view in plan.requests)
    shape = (tokens,)
    op = OpSpec("work", "vector", "body", 0, 0,
                tuple(view.request_id for view in plan.requests),
                shape, shape, shape, shape, flops=tokens * 1.0e6,
                vector_ops=tokens * 1.0e6, dtype="bf16")
    return IterationWork((op,), plan, 1, tokens, tokens, None, "body", "issue34")


def composed():
    model = {
        "name": "issue34-compact", "layers": 2, "hidden_size": 32,
        "num_heads": 4, "num_kv_heads": 2, "head_dim": 8,
        "intermediate_size": 64, "vocab_size": 128,
        "num_experts": 4, "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
    }
    hardware = {"sku": "Ascend950DT_9582", "num_devices": 4,
                "ranks_per_node": 4}
    return compose_case(
        model, hardware, tp=1, dp=4, ep=4, num_devices=4, ranks_per_node=4,
        prompt_tokens=16, output_tokens=2, concurrency=2, prefix_tokens=8,
        o_proj_tp=4, linear_proj_tp=4, lm_head_tp=4,
        redundant_experts_per_rank=1, recover_ssm=True,
        prefix_cache_mode="shared", prefix_read_mode="auto",
        collective_startup_mode="log2", matrix_compute_policy="auto",
        homogeneous_graph_reuse="auto",
    )


def test_compose_parse_predict_capacity_agree_without_new_engine():
    raw = composed()
    case = parse_case(raw)
    assert case.runtime.parallel.o_proj_tp == 4
    assert case.runtime.parallel.redundant_experts_per_rank == 1
    assert case.runtime.recover_ssm is True
    encoded = json.dumps(case.to_dict())
    assert Case.from_mapping(json.loads(encoded)).to_dict() == case.to_dict()
    result = predict(case, lower=linear_work)
    assert result["status"] in {"modeled", "infeasible"}
    assert result["execution"]["work"]["flops"] > 0
    memory = Memory(case)
    window = capacity_window(case, phase="decode", context=32, sequences=2,
                             memory=memory)
    assert window["schema"] == "mindie-sim.capacity-window/1"
    assert window["stages"][0]["peak_bytes"] > 0
    assert window["stages"][0]["supply_bytes"] > 0
    assert window["fits"] in {True, False}
