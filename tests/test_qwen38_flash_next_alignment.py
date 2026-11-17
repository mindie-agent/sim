"""End-to-end Qwen4Exp (qwen3.8-flash-next) and cross-catalog alignment tests.

[Role]
  Verify Qwen4Exp 16-prime PLE tables, low-rank gated HyperConnection, weight-free
  4:1 QSA indexer, 12-slot Qwen4KVCacheLayout pool, sparse-GQA index_k cache/decode,
  Gemma-4 K=V cache row width, and built-in kimi-k3.1 catalog resolution on Ascend.

[Invariants]
  * PLE 1-based layer_id=2 maps to 0-based layer 1 with 320,001,536 x 160 rows (~102.40 GB BF16).
  * Low-rank HC emits zero Sinkhorn or FP32 full-stream cast ops.
  * Real Ascend SKUs execute predict, efficiency, SLA, and body graph_costs without error.

[Timeouts & Locks]
  Bounded CPU-only tests (< 2.5s); no NPU contexts, network, or locks.

[Side-Effect Policy]
  Pure in-memory case composition and evaluation; zero host config mutation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mindie_sim import (
    analyze_network_efficiency,
    compose_case,
    graph_costs,
    predict,
    solve_sla_sizing,
    target_graphs,
)
from mindie_sim.catalog.cache import layer_cache_components, request_cache_breakdown
from mindie_sim.catalog.weights import (
    cache_row_bytes_by_layer,
    engram_table_terms,
    layer_weight_terms,
    stage_weight_bytes,
)
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.parallel import ParallelSpec

ROOT = Path(__file__).resolve().parents[1]


def test_qwen38_flash_next_ple_weights_sharding_quant_offload_and_text_only():
    model = load_model("qwen3.8-flash-next")
    assert model.engram is not None
    table = model.engram.tables[0]
    assert (table.layer_id, table.rows, table.width) == (1, 320_001_536, 160)
    assert model.layer_plans[1].engram and not model.layer_plans[2].engram

    terms = engram_table_terms(model)
    assert len(terms) == 1 and terms[0].shard == "out_tp"
    bf16_total = table.rows * table.width * 2
    assert 102.40e9 <= bf16_total <= 102.41e9

    tp1 = stage_weight_bytes(model, ParallelSpec(tp=1, ep=1))[0]
    tp8 = stage_weight_bytes(model, ParallelSpec(tp=8, ep=8))[0]
    tp8_offload = stage_weight_bytes(
        model, ParallelSpec(tp=8, ep=8), include_engram_tables=False
    )[0]
    assert pytest.approx(tp8 - tp8_offload, rel=1e-4) == bf16_total / 8
    assert tp1 > tp8

    raw_cfg = json.loads(
        (ROOT / "mindie_sim/catalog/configs/qwen3.8-flash-next.json").read_text(encoding="utf-8")
    )
    int8_cfg = json.loads(json.dumps(raw_cfg))
    int8_cfg["text_config"]["engram_storage_mode"] = "device_int8_fp32"
    mxfp8_cfg = json.loads(json.dumps(raw_cfg))
    mxfp8_cfg["text_config"]["engram_storage_mode"] = "host_fp8_e8m0"
    lm_only_cfg = json.loads(json.dumps(raw_cfg))
    lm_only_cfg["language_model_only"] = True

    int8_m = parse_case(compose_case(int8_cfg, hardware="a3", tp=8, ep=8, num_devices=8,
                                  va_recommended=False)).model
    mxfp8_m = parse_case(compose_case(mxfp8_cfg, hardware="a5", tp=8, ep=8, num_devices=8,
                                  va_recommended=False)).model
    lm_only_m = parse_case(compose_case(lm_only_cfg, hardware="a3", tp=8, ep=8, num_devices=8,
                                  va_recommended=False)).model
    base_m = parse_case(compose_case("qwen3.8-flash-next", hardware="a3", tp=8, ep=8,
                                  num_devices=8, va_recommended=False)).model

    w_base = stage_weight_bytes(base_m, ParallelSpec(tp=8, ep=8))[0]
    w_int8 = stage_weight_bytes(int8_m, ParallelSpec(tp=8, ep=8))[0]
    w_mxfp8 = stage_weight_bytes(mxfp8_m, ParallelSpec(tp=8, ep=8))[0]
    w_lm_only = stage_weight_bytes(lm_only_m, ParallelSpec(tp=8, ep=8))[0]
    assert w_mxfp8 < w_int8 < w_base
    assert w_lm_only < w_base


def test_qwen38_flash_next_qsa_indexer_and_lowrank_hc_invariants():
    model = load_model("qwen3.8-flash-next")
    qsa_plan = next(p for p in model.layer_plans if p.indexer)
    names = {t.name for t in layer_weight_terms(qsa_plan, model)}
    assert {"index_qk_proj", "indexer_q_norm", "indexer_k_norm"} <= names
    assert names.isdisjoint({"indexer_weights", "indexer_pool_gate", "indexer_pool_ape"})
    assert {"hc_attn_norm", "hc_attn_down_inject", "hc_attn_up"} <= names
    assert names.isdisjoint({"input_norm", "post_attn_norm", "hc_attn_fn"})

    case = compose_case(
        "Qwen3.8-Flash-Next-w8a8-mtp",
        hardware="a3", tp=8, ep=8, num_devices=8, quant="w8a8",
        layers=4, prompt_tokens=32, output_tokens=1,
    )
    graphs = target_graphs(case, phase="decode", tokens=1, context=4096, scope="body")
    ops = graphs[0]
    recipes = {op.recipe for op in ops}
    assert {"grouped_gemma_rmsnorm", "hc_lowrank_down_inject", "hc_lowrank_up",
            "hc_gate_mix", "hc_combine", "ple_ngram_ids", "ple_kv_proj",
            "ple_gate", "ple_short_conv"} <= recipes
    assert recipes.isdisjoint({"hc_sinkhorn", "hc_values_fp32"})

    cost = graph_costs(case, graphs, external_inputs={"analysis_input": 0})
    assert cost["duration_ms"] > 0


def test_qwen4_kv_pool_sparse_gqa_index_cache_gemma_keqv_and_kimi_k31():
    qwen = load_model("qwen3.8-flash-next")
    breakdown = request_cache_breakdown(qwen, ParallelSpec(tp=8, ep=8), 4096)
    pool = breakdown["qwen4_exp_pool"]
    assert (pool["slot_count"], pool["gdn_state_groups"], pool["raw_blocks"]) == (12, 3, 1)
    assert pool["private_ple_bytes"] > 0 and breakdown["by_component"]["ple_conv_state"] > 0

    for key in ("qwen3.8-flash-next", "minimax-m3", "minimax-m3.1-flash", "step-5-preview"):
        m = load_model(key)
        sparse_plan = next(p for p in m.layer_plans if p.owns_index_k)
        comp = layer_cache_components(m, sparse_plan, ParallelSpec(tp=8, ep=8), 4096)
        assert comp.get("index_k", 0.0) > 0.0, f"{key} missing index_k cache"

    gemma = load_model("gemma-4-26b-a4b")
    row_map = dict(cache_row_bytes_by_layer(gemma))
    keqv_layer = next(p for p in gemma.layer_plans if p.k_eq_v)
    assert row_map[keqv_layer.index] == keqv_layer.num_kv_heads * keqv_layer.head_dim * 2

    k31 = parse_case(compose_case(
        "kimi-k3.1", hardware="a5", tp=8, dp=4, ep=32, dcp=8,
        num_devices=32, block_size=1536, max_model_len=32768,
        layers=8, prompt_tokens=64, output_tokens=1,
    ))
    assert sum(p.indexer for p in k31.model.layer_plans) == 2


def test_ascend_five_entrypoint_smoke_matrix():
    qwen = compose_case(
        "qwen3.8-flash-next", hardware="a3", quant="w4a8",
        tp=8, ep=8, num_devices=8, layers=4, prompt_tokens=32, output_tokens=1,
    )
    pred = predict(qwen)
    assert pred["status"] == "modeled" and pred["metrics"]
    eff = analyze_network_efficiency(qwen, phase="decode", tokens=1, context=512, sequences=2)
    assert isinstance(eff, dict) and eff
    sla = solve_sla_sizing(
        qwen, sla_ms=200.0, context=512, max_sequences_per_dp=2,
        scenarios=({"name": "1x", "compute_scale": 1.0, "hbm_scale": 1.0},),
        b1_analysis=eff,
    )
    assert sla["scenarios"]

    for key, hw, quant in (
        ("qwen3.8-flash-next", "a5", "w8a8_mxfp8"),
        ("minimax-m3.1-flash", "a5", "w8a8_mxfp8"),
        ("gemma-4-26b-a4b", "a3", "bf16"),
    ):
        case = compose_case(
            key, hardware=hw, quant=quant, tp=8, ep=8, num_devices=8,
            layers=6, prompt_tokens=32, output_tokens=1,
        )
        graphs = target_graphs(case, phase="decode", tokens=1, context=512, scope="body")
        assert graph_costs(case, graphs, external_inputs={"analysis_input": 0})["duration_ms"] > 0
