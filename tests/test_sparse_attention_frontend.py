"""Indexed MLA does not inherit dense query-tile KV sharing."""
from dataclasses import replace

import pytest

from mindie_sim import target_operators
from mindie_sim.prediction.index_pool import selection
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import RequestView
from test_index_pool import case as pool_case
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


def test_index_pool_prices_unpooled_causal_tail_row():
    pmodel = parse_case(pool_case()).model
    original = pmodel.layer_plans[3]
    pool_reads = []
    for pool in (2, 4):
        pplan = replace(original, index_kpool=pool, index_topk=4 * pool)
        pviews = (RequestView("r0", 1, 4 * pool - 1, False, "decode"),)
        b = Builder("l3", pplan, 0, pviews, pmodel, "decode")
        selection(b, query="q", weights="w", cache="k", views=pviews,
                  plan=pplan, quant=pmodel.quant, padded_tokens=1)
        pool_reads.append({op.recipe: op for op in b.ops}["index_pool_topk"].mte2_bytes)
    assert pool_reads[0] == pytest.approx(pool_reads[1])

    tviews = (RequestView("r0", 1, 0, False, "prefill"),)
    btail = Builder("l3", original, 0, tviews, pmodel, "prefill")
    selection(btail, query="q", weights="w", cache="k", views=tviews,
              plan=original, quant=pmodel.quant, padded_tokens=1)
    packed = {op.recipe: op for op in btail.ops}["index_topk"]
    assert dict(packed.metadata)["causal_tail_rows"] == 1
    assert packed.vector_ops > 0
    assert packed.mte3_bytes > 0
