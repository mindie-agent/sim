"""Ordinary frontend selection keeps dense work while exporting the source ABI."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs, target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure_contracts import predicted_contract


def test_eager_wrapper_uses_public_lowering_and_preserves_executed_matrix_costs():
    root = Path(__file__).resolve().parents[1]
    raw = json.loads((root / "examples/cases/"
                     "kimi-k3-a5-current-checkpoint-conditional.json").read_text())
    raw["runtime"]["graph_mode"] = "eager"
    baseline = parse_case(raw)
    raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})["matrix_wrapper"] = {
        "implementation": "vllm_ascend_mxfp8_single_group_eager",
        "options": {"projection_recipes": ["latent_projection"]},
    }
    selected = parse_case(raw)
    before = target_operators(baseline, phase="decode", tokens=1, context=2048)
    after = target_operators(selected, phase="decode", tokens=1, context=2048)
    assert len(before) == len(after)
    pairs = []
    for original, wrapped in zip(before, after):
        assert replace(wrapped, metadata=original.metadata) == original
        if original.recipe == "latent_projection":
            assert predicted_contract(original.to_dict())[0] == "matrix"
            assert predicted_contract(wrapped.to_dict())[0] == "grouped_matrix"
            pairs.append((original, wrapped))
        else:
            assert "matrix_wrapper" not in dict(wrapped.metadata)
    assert pairs
    prior = operator_costs(baseline, tuple(pair[0] for pair in pairs[:2]))
    current = operator_costs(selected, tuple(pair[1] for pair in pairs[:2]))
    assert current["operator_sum_ms"] == pytest.approx(prior["operator_sum_ms"])
    assert [row["serviced"] for row in current["operators"]] == [
        row["serviced"] for row in prior["operators"]]
