"""Independent TP/PP arithmetic for shared-KV public HF graph execution."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology

from test_prediction_system import case


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize("phase,query,context", [("prefill", 3, 0), ("decode", 1, 16)])
def test_shared_kv_tp_pp_preserves_payload_weights_and_real_ready(
        tp, pp, phase, query, context):
    raw = case(tp=tp, pp=pp, prompt=3, output=2)
    raw["model"] = {
        "model_type": "llama", "hidden_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "intermediate_size": 64, "vocab_size": 128,
        "kv_source_layer_ids": [0, 0], "qk_norm": True,
    }
    parsed = parse_case(raw)
    graphs = target_graphs(raw, phase=phase, tokens=query, context=context)
    ops = graphs[0]
    heads, kv_heads = 4 // tp, max(1, 2 // tp)
    projected = [op for op in ops if op.family == "qkv_projection"]
    assert [op.flops for op in projected] == [
        2 * query * 32 * (heads + 2 * kv_heads) * 8,
        2 * query * 32 * heads * 8,
    ]
    weights = [
        {term.name: term for term in layer_weight_terms(layer, parsed.model)}
        for layer in parsed.model.layer_plans
    ]
    assert weights[0]["k_proj"].logical_parameters == 2 * 8 * 32
    assert weights[0]["v_proj"].logical_parameters == 2 * 8 * 32
    assert not {"k_proj", "v_proj", "k_norm"} & weights[1].keys()
    stores = [op for op in ops if op.recipe == "kv_store"]
    assert len(stores) == 1
    assert stores[0].mte3_bytes == query * 2 * kv_heads * 8 * 2
    cache = [
        layer_cache_components(parsed.model, layer, parsed.runtime.parallel,
                               context + query, capacity_padded=False)
        for layer in parsed.model.layer_plans
    ]
    assert sum(cache[0].values()) == (context + query) * 2 * kv_heads * 8 * 2
    assert cache[1] == {}
    consumer = next(op for op in ops if op.recipe == "shared_cache_dependency")
    transfers = [op for op in ops if op.recipe == "shared_kv_transfer"]
    if pp == 1:
        assert not transfers
        assert stores[0].op_id in consumer.deps
    else:
        assert len(transfers) == 1
        transfer = transfers[0]
        assert transfer.deps == (stores[0].op_id,)
        assert transfer.op_id in consumer.deps
        expected_rank_bytes = (context + query) * 2 * kv_heads * 8 * 2
        assert transfer.link_bytes == expected_rank_bytes
        topology = Topology(parsed.machine, parsed.runtime.parallel)
        flows = [flow for flow, _ in topology.messages(transfer, 0, transfer.link_bytes)]
        assert len(flows) == tp
        assert sum(flow.payload_bytes for flow in flows) == tp * expected_rank_bytes

    report = graph_costs(parsed, graphs, input_ready_ms=7, isolated=False)
    timeline = {row["op_id"]: row for row in report["timeline"]}
    assert all(row["ready_ms"] >= 7 for row in timeline.values())
    assert timeline[consumer.op_id]["ready_ms"] >= timeline[stores[0].op_id]["complete_ms"]
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert not any(report["execution"]["live_allocated"].values())
    if pp > 1:
        assert report["model_coverage"] == "partial"
        assert report["degraded_submodules"]


@pytest.mark.parametrize("owners", [[0, 0], {"0": 0, "1": 0}])
def test_standard_family_consumer_alias_removes_duplicate_kv_work(owners):
    raw = case(prompt=3, output=2)
    raw["model"] = {
        "model_type": "qwen3", "hidden_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "intermediate_size": 64, "vocab_size": 128,
        "kv_consumer_to_owner": owners,
    }
    parsed = parse_case(raw)
    assert [layer.kv_owner_layer_id for layer in parsed.model.layer_plans] == [0, 0]
    graph = target_graphs(raw, phase="decode", tokens=1, context=16)[0]
    assert sum(op.recipe == "kv_store" for op in graph) == 1
    assert [op.local_shape[1] for op in graph if op.family == "qkv_projection"] == [64, 32]
    result = graph_costs(raw, graph, isolated=False)
    assert result["duration_ms"] > 0 and result["execution"]["live_tasks"] == 0


@pytest.mark.parametrize("fixture,family", [
    ("deepseek-v2-lite", "deepseek"), ("glm4-0414", "glm"),
])
def test_unknown_same_family_and_changed_activation_remain_partial(fixture, family):
    path = Path(__file__).parent / "fixtures/hf-standard-families" / (fixture + ".json")
    config = json.loads(path.read_text())
    config.pop("rope_scaling", None)
    config.pop("partial_rotary_factor", None)
    config.update(model_type=family + "_future_custom", architectures=["Custom" + family],
                  attention_bias=False, hidden_act="gelu")
    raw = case(prompt=1, output=1)
    raw["model"] = config
    parsed = parse_case(raw)
    gaps = dict(parsed.model.metadata)["unmodeled_features"]
    assert dict(parsed.model.metadata)["model_coverage"] == "partial"
    assert any("gelu" in gap and "proxy" in gap for gap in gaps)
    # Known matrix/FFN work remains available even though the changed family
    # and activation do not justify a complete checkpoint prediction.
    graphs = target_graphs(raw, phase="decode", tokens=1, context=8)
    assert sum(op.flops for op in graphs[0]) > 0
