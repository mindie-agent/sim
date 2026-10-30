"""Source GDN convolution keeps active tensors separate from native index ABI."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim.structure import export_structure
from mindie_sim.structure_contracts import short_conv_active_contract


@pytest.mark.parametrize("model,tp,dim", [("qwen36", 8, 1280), ("qwen38", 4, 2560)])
def test_source_decode_convolution_exports_all_declared_layer_tensor_boundaries(model, tp, dim):
    # Corresponding retained A5 c4 captures expose x=[16,D], w=[4,D],
    # state-bank=[capacity,6,D], y=[16,D], all BF16. Capacity is not a Case fact.
    root = Path(__file__).resolve().parents[1]
    case = json.loads((root / "examples/cases/high-concurrency-20261030/source-gdn"
                       / f"{model}-a5-tp{tp}-c4.json").read_text())
    graph = export_structure(case, phase="decode", tokens=4, context=4104,
                             sequences=4, scope="body")
    nodes = [row for row in graph["nodes"] if row["family"] == "short_convolution"]
    assert len(nodes) == 48
    expected = ["causal_conv1d_active", "CausalConv1d", [[16, dim], "bf16", "ND"],
                [[4, dim], "bf16", "ND"], [[], None, None],
                [[6, dim], "bf16", "ND"], [[16, dim], "bf16", "ND"]]
    assert all(row["contract"] == expected for row in nodes)
    op = nodes[0]["operator"]
    changed = deepcopy(op)
    changed["metadata"]["state_history_slots"] = 7
    assert short_conv_active_contract(changed) != short_conv_active_contract(op)
    changed = deepcopy(op)
    changed["metadata"]["cache_index_dtype"] = "int64"
    assert short_conv_active_contract(changed) == short_conv_active_contract(op)
    assert nodes[0]["physical_work"] is not None
