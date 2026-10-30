"""Catalog coverage is retained as structured data through real public costs."""
from copy import deepcopy
import json

from mindie_sim import graph_costs, operator_costs, target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.model import ModelSpec
from test_prediction_system import case


def test_catalog_partial_metadata_survives_normalization_and_real_programs():
    raw = case()
    gaps = ["int8_sparse_kv_source_unmodeled", "native_dispatch_protocol_unbound"]
    degraded = ["sparse_attention_analytic"]
    raw["model"] = {
        "catalog_key": "glm-5.3-flash",
        "name": "declared-capture-model",
        "metadata": {
            "model_coverage": "partial",
            "unmodeled_features": gaps,
            "degraded_submodules": degraded,
            "source": {"identity": "capture-launch", "dimensions": [24, 4096]},
            "name": {"checkpoint": "declared-capture-model"},
        },
    }
    original = deepcopy(raw)
    parsed = parse_case(raw)
    assert parsed.model.layers == 45
    metadata = dict(parsed.model.metadata)
    assert metadata["source"] == original["model"]["metadata"]["source"]
    assert metadata["name"] == {"checkpoint": "declared-capture-model"}
    assert metadata["unmodeled_features"] == gaps
    restored = ModelSpec.from_mapping(parsed.model.to_dict())
    assert json.dumps(restored.to_dict(), sort_keys=True) == json.dumps(
        parsed.model.to_dict(), sort_keys=True)

    # Price a real catalog norm rather than fabricating an operator whose
    # physical boundary cannot exercise catalog/frontend integration.
    operators = target_operators(parsed, phase="decode", tokens=1, context=2048)
    norm = next(op for op in operators if op.family == "rmsnorm")
    costs = operator_costs(parsed, (norm,))
    graph = graph_costs(parsed, (norm,),
                        external_inputs={parent: 0 for parent in norm.deps})
    assert costs["operator_sum_ms"] > 0
    assert graph["duration_ms"] > 0
    assert graph["execution"]["live_tasks"] == 0
    for result in (costs, graph):
        assert result["model_coverage"] == "partial"
        assert result["unmodeled_features"] == gaps
        assert result["degraded_submodules"] == degraded
    assert raw == original

    # Parsed declarations own their metadata, including nested contracts.
    gaps.append("subsequent_input_mutation")
    raw["model"]["metadata"]["source"]["dimensions"].append(1)
    assert dict(parsed.model.metadata)["unmodeled_features"] == original["model"]["metadata"]["unmodeled_features"]
    assert dict(parsed.model.metadata)["source"] == original["model"]["metadata"]["source"]
