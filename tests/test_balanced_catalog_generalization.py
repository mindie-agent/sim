"""Real catalog FFN geometry under integer forced balancing.

[Role] Cover routed catalog structures without reducing their real dimensions.
[Invariants] Pure metadata/Engine controls; no model weights, measured timing
  inputs, fabricated token IDs or layer-count extrapolation. Full-model serving
  and accuracy acceptance belong to the bounded integration report.
[Timeouts & Locks] In-memory integer sweeps and a finite number of FFN graphs.
[Side-Effect Policy] Read installed catalog metadata; no processes or network.
"""
from collections import Counter
from functools import lru_cache
import math

import pytest

from mindie_sim import graph_costs
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.feedforward import alltoall, dense
from mindie_sim.prediction.routing_realizations import select_realization
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import lower_implementations


MODELS = (
    "qwen3.5-35b-a3b", "qwen3.6-27b", "qwen3.8-27b", "glm-5.3",
    "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4.1", "kimi-k3", "minimax-m3",
)


@lru_cache(maxsize=18)
def catalog_case(name, sku="Ascend910_9362"):
    routed = any(plan.ffn == "moe" for plan in load_model(name).layer_plans)
    return parse_case({
        "model": {"preset": name, "quantization": "w8a8"},
        "machine": {"backend": "ascend", "sku": sku, "num_devices": 8},
        "runtime": {"tp": 8, "ep": 8 if routed else 1,
                    "routing": {"policy": "balanced"},
                    "kernels": {"programs": {"matmul": {"kind": "steady_matmul"}}}},
        "workload": {"requests": [{"id": "component", "prompt_tokens": 1,
                                      "output_tokens": 1}]},
    })


def representatives(model):
    """Only merge identical FFN structures; retain all original layer members."""
    fields = tuple(name for name in model.layer_plans[0].__dataclass_fields__
                   if any(term in name for term in ("expert", "router", "shared", "ffn", "situ"))
                   or name in {"moe_intermediate_size", "intermediate_size", "scope"})
    groups = {}
    for layer in model.layer_plans:
        key = tuple(getattr(layer, name) for name in fields)
        groups.setdefault(key, []).append(layer)
    return tuple((layers[0], tuple(layers)) for layers in groups.values())


def realization(case, layer, tokens):
    return select_realization(case.runtime, layer, phase="decode", tokens=tokens,
                              valid_tokens=tokens)


def ffn_ops(case, layer, tokens):
    routing = realization(case, layer, tokens) if layer.ffn == "moe" else None
    function = alltoall if layer.ffn == "moe" else dense
    ops, _ = function(
        prefix="component", plan=layer, deps="input", stage=0, requests=("component",),
        tokens=tokens, model=case.model, parallel=case.runtime.parallel,
        mode="decode", runtime=case.runtime, routing_realization=routing)
    return lower_implementations(ops, case.runtime, case.model.activation_dtype,
                                 local_tokens=tokens, resource_fusion=False)


@pytest.mark.parametrize("name", MODELS)
def test_real_routed_catalog_layers_cover_every_batch_and_query_integer(name):
    case = catalog_case(name)
    groups = representatives(case.model)
    assert sum(len(members) for _, members in groups) == case.model.layers
    routed = [(layer, members) for layer, members in groups if layer.ffn == "moe"]
    for layer, members in routed:
        for batch in range(1, 129):
            for query in (1, 4, 5, 8):
                tokens = batch * query
                route = realization(case, layer, tokens)
                assert sum(route.expert_rows) == tokens * layer.top_k
                assert len(route.expert_rows) == layer.num_experts
                counts = Counter(route.expert_rows)
                assert max(counts) - min(counts) <= 1
                if tokens * layer.top_k < layer.num_experts:
                    assert counts[0] == layer.num_experts - tokens * layer.top_k
                assert all(rows <= tokens for rows in route.expert_rows)
                ranks = tuple(sum(route.expert_rows[rank::case.runtime.parallel.ep])
                              for rank in range(case.runtime.parallel.ep))
                assert sum(ranks) == tokens * layer.top_k
                assert route.rows == ranks[0] == max(ranks)
                assert route.active == sum(count > 0 for count in route.local_expert_rows)
    if not routed:
        # Dense catalog siblings remain dense; balancing must not manufacture
        # experts merely to satisfy a routing coverage sweep.
        ops = ffn_ops(case, groups[0][0], 1)
        assert any(op.flops > 0 for op in ops)
        assert not any(op.family == "grouped_gemm" for op in ops)


@pytest.mark.parametrize("name", tuple(name for name in MODELS if any(
    layer.ffn == "moe" for layer in load_model(name).layer_plans)))
def test_real_expert_width_weight_banks_and_row_scales_survive_lowering(name):
    case = catalog_case(name)
    for layer, _ in representatives(case.model):
        if layer.ffn != "moe":
            continue
        # Sparse, remainder and populated distributions exercise all real FFN
        # structures without putting complete multi-layer graphs in unit tests.
        for tokens in (1, 5, 128 * 5):
            route = realization(case, layer, tokens)
            ops = ffn_ops(case, layer, tokens)
            matrices = [op for op in ops if op.family == "grouped_gemm"]
            assert len(matrices) == 2
            latent = layer.expert_latent_size or case.model.hidden_size
            width = math.ceil(layer.moe_intermediate_size / case.runtime.parallel.etp)
            assert [op.local_shape for op in matrices] == [
                (route.rows, 2 * width, latent), (route.rows, latent, width)]
            by_id = {op.op_id: op for op in ops}
            for op in matrices:
                metadata = dict(op.metadata)
                rows, n, k = op.local_shape
                assert op.flops == 2 * rows * n * k
                assert metadata["matrix_group_rows"] == route.local_expert_rows
                assert metadata["packed_weight_bytes"] == pytest.approx(
                    route.active * case.model.quant.matrix_bytes(n, k, "routed_expert"))
                producer = by_id[op.deps[0]]
                assert producer.family == "quantize"
                assert producer.mte3_bytes == rows * k + 4 * rows
                assert metadata["matrix_epilogue"]["row_scale_bytes"] == 4 * rows


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("sku", ("Ascend910_9362", "Ascend950DT_9582"))
def test_real_ffn_component_runs_on_common_engine_without_balanced_recipe_override(name, sku):
    case = catalog_case(name, sku)
    candidates = representatives(case.model)
    layer = next((plan for plan, _ in candidates if plan.ffn == "moe"), candidates[0][0])
    ops = tuple(ffn_ops(case, layer, 5))
    result = graph_costs(case, {0: ops}, external_inputs={"input": 0})
    assert result["duration_ms"] > 0
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert result["bounds"]["combined_ms"] <= result["duration_ms"] + 1e-8
