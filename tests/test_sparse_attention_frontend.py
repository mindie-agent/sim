"""Indexed MLA does not inherit dense query-tile KV sharing."""
import pytest

from mindie_sim import target_operators
from mindie_sim.prediction.structure import parse_case
from test_prediction_system import case


def test_glm_prefill_prices_independent_query_selections_in_the_ordinary_graph():
    raw = case(tp=8)
    raw["model"] = {"catalog_key": "GLM-5.3-W4A8", "layers": 1}
    parsed = parse_case(raw)
    op = next(op for op in target_operators(parsed, phase="prefill", tokens=4096)
              if op.recipe == "latent_attention0")
    meta = dict(op.metadata)
    selected = sum(min(q, 2048) for q in range(1, 4097))
    assert selected == 6292480
    assert meta["indexed_gather_rows"] == meta["kv_sweeps"] == selected
    assert meta["logical_pairs"] == selected
    assert meta["selection_index_read_bytes"] == 4 * selected
    query_transfer = 4096 * 8 * (512 + 64 + 512) * 2
    assert op.mte2_bytes == pytest.approx(selected * (512 + 64) * 2
                                         + 4 * selected + query_transfer / 2)
    assert op.flops == 2 * selected * 8 * (512 + 64 + 512)
