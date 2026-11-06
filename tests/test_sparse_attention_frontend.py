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
    assert op.hbm_bytes == pytest.approx(op.mte2_bytes + op.mte3_bytes)
    assert op.flops == 2 * selected * 8 * (512 + 64 + 512)


def test_glm_lightning_indexer_distinguishes_prefill_and_mtp_decode_work():
    raw = case(tp=8)
    raw["model"] = {"catalog_key": "GLM-5.3-W4A8", "layers": 1}
    parsed = parse_case(raw)
    decode_ops = {
        op.recipe: op
        for op in target_operators(parsed, phase="decode", tokens=6, context=4096, sequences=2)
    }
    prefill_ops = {
        op.recipe: op
        for op in target_operators(parsed, phase="prefill", tokens=4096, context=0, sequences=1)
    }
    assert decode_ops["index_scores"].hbm_bytes > 0
    assert decode_ops["index_topk"].vector_ops > 48 * 2048 * 66
    assert prefill_ops["index_scores"].flops == 2 * sum(range(1, 4097)) * 32 * 128
    assert prefill_ops["index_topk"].vector_ops > sum(min(q, 2048) for q in range(1, 4097)) * 11


def test_sparse_indexer_handles_empty_mixed_and_candidate_filtered_views():
    from mindie_sim.model_catalog import load_model
    from mindie_sim.prediction.mla import mla
    from mindie_sim.schema.parallel import ParallelSpec
    from mindie_sim.schema.workload import RequestView

    def run_mla(model, plan, views, *, mode="decode", tp=8, candidate_ids=None):
        q = sum(v.q_len for v in views)
        res = mla(
            prefix=f"s0.l{plan.index}", plan=plan, deps="in", stage=0, views=views,
            model=model, quant=model.quant, parallel=ParallelSpec(tp=tp),
            mode=mode, padded_tokens=max(1, q),
            cache_ids={plan.kv_owner_layer_id: "kv0"}, topk_ids={},
            index_cache_ids={plan.index_k_owner_layer_id: "idx0"},
            candidate_ids=candidate_ids or {plan.candidate_source_layer_id: "cand0"},
            stage_of=lambda _layer: 0)
        return {op.recipe: op for op in res.ops}

    glm = load_model("glm-5.3")
    plan = glm.layer_plans[0]
    empty_by_recipe = run_mla(glm, plan, ())
    assert empty_by_recipe["index_scores"].vector_ops == 0
    assert empty_by_recipe["index_topk"].vector_ops == 0
    assert empty_by_recipe["index_topk"].mte2_bytes == 0

    mixed_by_recipe = run_mla(
        glm, plan,
        (RequestView("d0", 1, 4096, False, "decode"),
         RequestView("p0", 1, 0, False, "prefill")),
        mode="prefill")
    assert mixed_by_recipe["index_topk"].vector_ops > 48 * 2048 * 66

    v41 = load_model("deepseek-v4.1-flash")
    v41_by_recipe = run_mla(
        v41, v41.layer_plans[24],
        (RequestView("r0", 64, 65536, False, "prefill"),),
        mode="prefill", tp=1)
    assert v41_by_recipe["index_topk"].mte2_bytes == pytest.approx(
        v41_by_recipe["index_scores"].mte3_bytes)

    from dataclasses import replace
    from test_index_pool import case as pool_case
    from mindie_sim.prediction.lowering import Builder
    from mindie_sim.prediction.index_pool import selection

    views_decode = (RequestView("r0", 1, 4096, False, "decode"),)
    topk_reads = []
    for kind in ("int8", "bf16"):
        quant = replace(glm.quant, index_dtype=kind,
                        index_scale_dtype="fp16" if kind == "int8" else "none")
        res = mla(
            prefix="s0.l0", plan=plan, deps="in", stage=0, views=views_decode,
            model=replace(glm, quant=quant), quant=quant, parallel=ParallelSpec(tp=8),
            mode="decode", padded_tokens=1, cache_ids={0: "kv0"}, topk_ids={},
            index_cache_ids={0: "idx0"}, candidate_ids={}, stage_of=lambda _layer: 0)
        by_recipe = {op.recipe: op for op in res.ops}
        topk_reads.append(by_recipe["index_topk"].mte2_bytes)
    assert topk_reads[0] == pytest.approx(topk_reads[1])

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
