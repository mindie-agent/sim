"""Parsed HF source facts reach canonical FeatureConfig metadata exactly."""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
from mindie_sim.schema.model import ModelSpec


def _model(metadata=(), norm_eps=1e-6):
    raw = dict(name="source-meta", layers=1, hidden_size=64, num_heads=4,
               num_kv_heads=2, head_dim=16, intermediate_size=64,
               vocab_size=128, layer_plans=[dict(index=0, attention="gqa",
                                                 ffn="dense")],
               norm_eps=norm_eps, metadata=metadata)
    return ModelSpec.from_mapping(raw)


def test_literal_mrope_and_eps_are_promoted_without_defaults():
    rope = {"mrope_section": [11, 11, 10], "mrope_interleaved": True,
            "partial_rotary_factor": 0.25, "rope_theta": 10_000_000,
            "rope_type": "default"}
    model = _model((("unconsumed_config_fields",
                     {"rope_parameters": rope,
                      "partial_rotary_factor": 0.25}),))
    meta = dict(FeatureConfig.from_model(model).metadata)
    assert meta["rope_parameters"]["mrope_section"] == (11, 11, 10)
    assert meta["rope_parameters"]["mrope_interleaved"] is True
    assert meta["mrope_section"] == (11, 11, 10)
    assert meta["mrope_interleaved"] is True
    assert meta["partial_rotary_factor"] == 0.25
    assert meta["rope_theta"] == 10_000_000
    assert meta["norm_eps"] == 1e-6


def test_missing_mrope_fields_stay_absent_and_explicit_false_stays_false():
    model = _model((("unconsumed_config_fields", {}),))
    missing = dict(FeatureConfig.from_model(model).metadata)
    assert "rope_parameters" not in missing
    assert "mrope_section" not in missing
    assert "mrope_interleaved" not in missing
    assert "partial_rotary_factor" not in missing
    assert "routed_scaling_factor" not in missing
    explicit = _model((("unconsumed_config_fields", {
        "rope_parameters": {"mrope_section": [2, 2, 2],
                            "mrope_interleaved": False}}),))
    meta = dict(FeatureConfig.from_model(explicit).metadata)
    assert meta["mrope_interleaved"] is False
    assert "partial_rotary_factor" not in meta


def test_actual_glm_and_kimi_routed_scaling_factor_is_not_a_default():
    import json
    from pathlib import Path
    from mindie_sim.prediction.structure import parse_case
    root = Path(__file__).parents[1]
    values = {}
    for relative in ("examples/cases/kimi-a5-paged-host-20261030/c1.json",
                     "examples/cases/glm53-a5.json"):
        case = parse_case(json.loads((root / relative).read_text()))
        features = FeatureConfig.from_runtime(case.runtime, case.model)
        values[relative] = dict(features.metadata).get("routed_scaling_factor")
    assert values["examples/cases/kimi-a5-paged-host-20261030/c1.json"] == 1.0
    assert values["examples/cases/glm53-a5.json"] == 2.5
