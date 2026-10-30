"""Partial structural knowledge survives candidate and deployment composition."""
from copy import deepcopy
import json

import pytest

from mindie_sim import predict, study
from mindie_sim.model_catalog import CONFIG_DIR
from mindie_sim.schema.model import ModelSpec
from test_composable_models import config, run_case
from test_prediction_deployment import pd_case
from test_study import PROBES


def test_hf_overrides_preserve_layers_and_unknown_semantics_in_rankings():
    raw = run_case(config(custom_module={"order": 1}))
    original = deepcopy(raw)
    result = study([
        {"id": "base", "case": raw},
        {"id": "memory", "case": raw, "overrides": {
            "machine": {"device": {"rates": {"memory": 2e9}}}}},
        {"id": "other-module", "case": raw, "overrides": {
            "model": {"custom_module": {"order": 2}}}},
        {"id": "deeper", "case": raw, "overrides": {
            "model": {"num_hidden_layers": 3}}},
    ], [PROBES[0]], level="graph", serving=True)
    assert result["status"] == "completed" and raw == original
    base, faster, custom, deeper = result["evaluations"]
    assert base["operators"]["work"] == faster["operators"]["work"]
    assert base["workload_key"] == faster["workload_key"]
    assert faster["duration_ms"] < base["duration_ms"]
    assert custom["workload_key"] != base["workload_key"]
    assert deeper["operators"]["work"]["flops"] > base["operators"]["work"]["flops"]
    assert len(result["rankings"]) == 3
    assert result["coverage_summary"] == {"complete_candidates": 0, "partial_candidates": 4}
    for candidate, evaluation in zip(result["candidates"], result["evaluations"]):
        for scope in (candidate, evaluation, candidate["serving"]):
            assert scope["model_coverage"] == "partial"
            assert "unrecognized_config_field:custom_module" in scope["unmodeled_features"]


def test_disaggregated_prediction_keeps_partial_model_scope():
    raw = pd_case()
    raw["model"] = config(custom_module={"order": 1})
    result = predict(raw)
    assert result["status"] == "modeled"
    assert result["requests"][0]["status"] == "served"
    assert result["model_coverage"] == "partial"
    assert "unrecognized_config_field:custom_module" in result["unmodeled_features"]
    assert result["execution"]["live_tasks"] == 0


@pytest.mark.parametrize("wrapper", [None, "hf_config", "text_config"])
def test_hf_layer_alias_override_changes_real_work_through_wrappers(wrapper):
    model = config()
    if wrapper:
        model = {wrapper: model}
    result = study([
        {"id": "base", "case": run_case(model)},
        {"id": "deeper", "case": run_case(model),
         "overrides": {"model": {"num_layers": 3}}},
    ], [PROBES[0]])
    base, deeper = result["evaluations"]
    assert result["status"] == "completed"
    assert deeper["operators"]["work"]["flops"] > base["operators"]["work"]["flops"]
    assert deeper["workload_key"] != base["workload_key"]


def test_native_hybrid_adapter_retains_custom_fields_and_unknown_quant_proxy():
    raw = json.loads((CONFIG_DIR / "qwen3.5-35b-a3b.json").read_text())
    text = raw.get("text_config", raw)
    text.update(custom_module={"order": 1}, attention_bias=True,
                quantization_config={"quant_method": "future-packing"})
    model = ModelSpec.from_mapping(raw)
    metadata = dict(model.metadata)
    assert model.layer_plans and model.quant.weight_format == "bf16"
    assert metadata["unrecognized_config_fields"]["custom_module"] == {"order": 1}
    assert any(gap.startswith("projection_bias_not_in_native_layer_plan")
               for gap in metadata["unmodeled_features"])
    assert any("future-packing" in gap for gap in metadata["degraded_submodules"])
    assert ModelSpec.from_mapping(model.to_dict()) == model
