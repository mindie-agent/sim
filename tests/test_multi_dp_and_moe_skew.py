"""Tier 0 unit tests for multi-DP barrier skew, MoE routing imbalance, and chunked prefill interference.

[Role] Verify axiomatic conservation and zero-drift properties of
  analyze_multi_dp_barrier_skew, analyze_moe_routing_imbalance, and
  analyze_chunked_prefill_decode_interference across uniform and skewed loads.
[Invariants] Pure in-memory Tier 0 execution (< 5s total); zero subprocesses,
  zero network, zero sleep; exact 0.00% baseline drift at uniform DP context,
  1.0x expert imbalance factor, and 0 prefill chunk tokens.
[Timeouts & Locks] In-memory representative MoE and dense geometries only.
[Side-Effect Policy] No file writes, device contexts, or global state mutation.
"""
from __future__ import annotations

import math
import pytest

from mindie_sim.prediction.efficiency import analyze_network_efficiency
from mindie_sim.prediction.skew_analysis import (
    analyze_chunked_prefill_decode_interference,
    analyze_moe_routing_imbalance,
    analyze_multi_dp_barrier_skew,
)


def _make_moe_case(*, layers: int = 2, dp: int = 1, tp: int = 4, ep: int = 4) -> dict:
    """Build an in-memory hybrid MLA + MoE case on Ascend910_9362."""
    return {
        "machine": {
            "backend": "ascend",
            "sku": "Ascend910_9362",
            "num_devices": dp * tp,
            "ranks_per_node": dp * tp,
            "device": {
                "rates": {"native_call_service": 1e9},
                "resource_units": {"native_call_service": 1},
                "work_units": {"native_call_service": "busy_ns"},
            },
        },
        "model": {
            "name": "tier0-skew-mla-moe",
            "layers": layers,
            "hidden_size": 4096,
            "num_heads": 32,
            "num_kv_heads": 1,
            "head_dim": 128,
            "intermediate_size": 2048,
            "vocab_size": 64000,
            "attention": "mla",
            "ffn": "moe",
            "activation_dtype": "bf16",
            "weight_dtype": "int8",
            "kv_dtype": "bf16",
            "compute_dtype": "bf16",
            "quantization": "w8a8",
            "num_experts": 16,
            "top_k": 2,
            "moe_intermediate_size": 1024,
            "num_shared_experts": 1,
            "mla": {
                "q_lora_rank": 1024,
                "kv_lora_rank": 512,
                "qk_nope_head_dim": 128,
                "qk_rope_head_dim": 64,
                "v_head_dim": 128,
            },
        },
        "runtime": {
            "tp": tp,
            "dp": dp,
            "ep": ep,
            "pp": 1,
            "max_model_len": 131072,
            "max_num_seqs": 128,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.90,
            "kv_block_size": 128,
        },
        "workload": {
            "requests": [
                {
                    "request_id": "req0",
                    "prompt_tokens": 4096,
                    "output_tokens": 16,
                    "arrival_ms": 0.0,
                }
            ]
        },
    }


def _make_dense_case(*, layers: int = 2, dp: int = 1, tp: int = 2) -> dict:
    """Build an in-memory dense GQA case without MoE EP coupling."""
    return {
        "machine": {
            "backend": "ascend",
            "sku": "Ascend910_9362",
            "num_devices": dp * tp,
            "ranks_per_node": dp * tp,
        },
        "model": {
            "name": "tier0-skew-dense",
            "layers": layers,
            "hidden_size": 4096,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "intermediate_size": 11008,
            "vocab_size": 32000,
            "attention": "gqa",
            "ffn": "dense",
            "activation_dtype": "bf16",
            "weight_dtype": "bf16",
            "kv_dtype": "bf16",
            "compute_dtype": "bf16",
        },
        "runtime": {
            "tp": tp,
            "dp": dp,
            "ep": 1,
            "pp": 1,
            "max_model_len": 131072,
            "max_num_seqs": 128,
            "max_num_batched_tokens": 8192,
            "gpu_memory_utilization": 0.90,
            "kv_block_size": 128,
        },
        "workload": {
            "requests": [
                {
                    "request_id": "req0",
                    "prompt_tokens": 4096,
                    "output_tokens": 16,
                    "arrival_ms": 0.0,
                }
            ]
        },
    }


def test_multi_dp_barrier_skew_uniform_and_skewed():
    """Verify 0.00% drift at uniform DP load and positive AllToAllV barrier wait under context skew."""
    case = _make_moe_case(layers=2, dp=4, tp=1, ep=4)
    base_4k = analyze_network_efficiency(
        case, phase="decode", tokens=1, context=4096, sequences=4, scope="body"
    )

    uniform = analyze_multi_dp_barrier_skew(
        case, dp_contexts=[4096, 4096, 4096, 4096], dp_sequences=4, phase="decode", tokens=1
    )
    assert uniform["schema"] == "mindie-sim.multi-dp-skew/1"
    assert uniform["ep_coupled"] is True
    assert uniform["num_dp_replicas"] == 4
    assert uniform["max_barrier_wait_ms"] == 0.0
    assert uniform["mean_barrier_wait_ms"] == 0.0
    assert uniform["skew_slowdown_ratio"] == 1.0
    assert uniform["kv_balance_speedup_ratio"] == 1.0
    assert math.isclose(uniform["coupled_step_ms"], base_4k["graph_duration_ms"], rel_tol=1e-12)
    assert math.isclose(uniform["kv_balanced_step_ms"], base_4k["graph_duration_ms"], rel_tol=1e-12)
    for rep in uniform["replicas"]:
        assert rep["barrier_wait_ms"] == 0.0
        assert math.isclose(
            rep["attention_and_local_ms"] + rep["shared_ep_and_ffn_ms"],
            rep["standalone_step_ms"],
            rel_tol=1e-12,
        )

    skewed = analyze_multi_dp_barrier_skew(
        case, dp_contexts=[4096, 4096, 4096, 131072], dp_sequences=4, phase="decode", tokens=1
    )
    assert skewed["ep_coupled"] is True
    assert skewed["straggler_replica"] == 3
    assert skewed["max_barrier_wait_ms"] > 0.0
    assert skewed["mean_barrier_wait_ms"] > 0.0
    assert skewed["coupled_step_ms"] > skewed["kv_balanced_step_ms"]
    assert skewed["skew_slowdown_ratio"] > 1.0
    assert skewed["kv_balance_speedup_ratio"] > 1.0
    assert skewed["replicas"][3]["barrier_wait_ms"] == 0.0
    for r in (0, 1, 2):
        assert math.isclose(
            skewed["replicas"][r]["barrier_wait_ms"],
            skewed["max_barrier_wait_ms"],
            rel_tol=1e-12,
        )


def test_multi_dp_barrier_skew_independent_vs_partitioned_ep_cohorts():
    """Verify independent EP groups (ep <= tp) never wait across DP, and sub-cohorts synchronize locally."""
    # DP=2, TP=4, EP=4 -> ep_spans_dp is False, cohorts are [[0], [1]] (no cross-DP barrier)
    indep_ep_case = _make_moe_case(layers=2, dp=2, tp=4, ep=4)
    indep_res = analyze_multi_dp_barrier_skew(
        indep_ep_case, dp_contexts=[4096, 131072], dp_sequences=4, phase="decode", tokens=1
    )
    assert indep_res["ep_coupled"] is False
    assert indep_res["cohorts"] == [[0], [1]]
    assert indep_res["max_barrier_wait_ms"] == 0.0
    assert indep_res["mean_barrier_wait_ms"] == 0.0

    # DP=4, TP=2, EP=4 -> ep_spans_dp is True, cohorts are [[0, 1], [2, 3]]
    sub_cohort_case = _make_moe_case(layers=2, dp=4, tp=2, ep=4)
    sub_res = analyze_multi_dp_barrier_skew(
        sub_cohort_case,
        dp_contexts=[4096, 4096, 4096, 131072],
        dp_sequences=4,
        phase="decode",
        tokens=1,
    )
    assert sub_res["ep_coupled"] is True
    assert sub_res["cohorts"] == [[0, 1], [2, 3]]
    assert sub_res["replicas"][0]["barrier_wait_ms"] == 0.0
    assert sub_res["replicas"][1]["barrier_wait_ms"] == 0.0
    assert sub_res["replicas"][2]["barrier_wait_ms"] > 0.0
    assert sub_res["replicas"][3]["barrier_wait_ms"] == 0.0

    # Mismatched dp_contexts length (including parallel.dp == 1) raises ValueError
    with pytest.raises(ValueError, match="dp_contexts_length_must_match_parallel_dp"):
        analyze_multi_dp_barrier_skew(sub_cohort_case, dp_contexts=[4096, 4096])
    with pytest.raises(ValueError, match="dp_contexts_length_must_match_parallel_dp"):
        analyze_multi_dp_barrier_skew(_make_moe_case(layers=2, dp=1, tp=4, ep=4), dp_contexts=[4096, 8192])

    dense = _make_dense_case()
    with pytest.raises(ValueError):
        analyze_multi_dp_barrier_skew(dense, dp_contexts=[])
    with pytest.raises(ValueError):
        analyze_multi_dp_barrier_skew(dense, dp_contexts=[-1])
    with pytest.raises(ValueError):
        analyze_multi_dp_barrier_skew(dense, dp_contexts=[4096], dp_sequences=[1, 2])
    with pytest.raises(ValueError):
        analyze_multi_dp_barrier_skew(dense, dp_contexts=[4096], dp_sequences=0)


def test_moe_routing_imbalance_rank_level_and_greedy_eplb():
    """Verify strided rank-level ownership, exact Engine agreement, and constructive greedy EPLB redistribution."""
    case = _make_moe_case(layers=2, dp=1, tp=4, ep=4)

    bal = analyze_moe_routing_imbalance(
        case, expert_imbalance_factor=1.0, eplb_redundant_experts=0
    )
    assert bal["schema"] == "mindie-sim.moe-imbalance/1"
    assert bal["effective_imbalance_factor"] == 1.0
    assert bal["eplb_effective_imbalance_factor"] == 1.0
    assert bal["imbalanced_step_ms"] == bal["balanced_step_ms"]
    assert bal["eplb_mitigated_step_ms"] == bal["balanced_step_ms"]
    assert bal["imbalance_slowdown_ratio"] == 1.0
    assert bal["eplb_speedup_ratio"] == 1.0
    assert bal["eplb_extra_weight_bytes_per_die"] == 0.0
    assert bal["routed_moe_share_of_step"] > 0.0

    # Strided ownership (e % 4): [8, 0, 0, 0] * 4 puts all 32 tokens on Rank 0!
    strided_all_r0 = [8, 0, 0, 0] * 4
    r0_only = analyze_moe_routing_imbalance(case, expert_token_counts=strided_all_r0)
    assert r0_only["rank_token_loads"] == [32.0, 0.0, 0.0, 0.0]
    assert math.isclose(r0_only["effective_imbalance_factor"], 4.0, rel_tol=1e-12)

    # Sparse active experts: exact agreement with analyze_network_efficiency on the Engine
    sparse_counts = [16, 0, 0, 0, 0, 8, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4]
    sparse_case = _make_moe_case(layers=1, dp=1, tp=4, ep=4)
    sparse_res = analyze_moe_routing_imbalance(
        sparse_case, sequences=16, context=128, expert_token_counts=sparse_counts
    )
    sparse_case["model"]["expert_routing"] = sparse_counts
    eng_sparse = analyze_network_efficiency(
        sparse_case, sequences=16, context=128, scope="body", run_graph=False
    )
    assert math.isclose(sparse_res["imbalanced_step_ms"], eng_sparse["graph_duration_ms"], rel_tol=1e-12)

    # Skewed strided rank loads ([200, 40, 40, 40]) mitigated by 2 greedy EPLB redundant experts -> [115, 82.5, 82.5, 40]
    skewed_counts = [85, 10, 10, 10, 85, 10, 10, 10, 15, 10, 10, 10, 15, 10, 10, 10]
    eplb_res = analyze_moe_routing_imbalance(
        case, expert_token_counts=skewed_counts, eplb_redundant_experts=2
    )
    assert eplb_res["rank_token_loads"] == [200.0, 40.0, 40.0, 40.0]
    assert math.isclose(eplb_res["effective_imbalance_factor"], 2.5, rel_tol=1e-12)
    assert eplb_res["eplb_rank_token_loads"] == [115.0, 82.5, 82.5, 40.0]
    assert math.isclose(eplb_res["eplb_effective_imbalance_factor"], 115.0 / 80.0, rel_tol=1e-12)
    assert eplb_res["balanced_step_ms"] < eplb_res["eplb_mitigated_step_ms"] < eplb_res["imbalanced_step_ms"]
    assert eplb_res["eplb_extra_weight_bytes_per_die"] > 0.0

    clamped = analyze_moe_routing_imbalance(case, expert_imbalance_factor=10.0)
    assert clamped["effective_imbalance_factor"] == 4.0

    with pytest.raises(ValueError):
        analyze_moe_routing_imbalance(case, expert_imbalance_factor=0.9)
    with pytest.raises(ValueError):
        analyze_moe_routing_imbalance(case, eplb_redundant_experts=-1)
    with pytest.raises(ValueError):
        analyze_moe_routing_imbalance(case, expert_token_counts=[10, 20, 30, 40])
    with pytest.raises(ValueError):
        analyze_moe_routing_imbalance(case, expert_token_counts=[0] * 16)


def test_chunked_prefill_decode_interference_monotonicity_and_sla():
    """Verify exact 0.0 inflation at C=0, Engine-scheduled mixed graph monotonicity, and SLA selection."""
    case = _make_moe_case(layers=2, dp=1, tp=4, ep=4)
    chunks = (0, 512, 1024, 2048, 4096, 8192)
    res = analyze_chunked_prefill_decode_interference(
        case,
        decode_sequences=16,
        decode_context=32768,
        prefill_chunk_tokens=chunks,
        prefill_context=0,
        sla_tpot_ms=10.0,
    )
    assert res["schema"] == "mindie-sim.chunked-prefill-interference/1"
    assert len(res["chunks"]) == len(chunks)

    c0 = res["chunks"][0]
    assert c0["prefill_chunk_tokens"] == 0
    assert c0["hybrid_step_ms"] == res["pure_decode_step_ms"]
    assert c0["decode_itl_inflation_ms"] == 0.0
    assert c0["decode_itl_inflation_ratio"] == 1.0
    assert c0["meets_tpot_sla"] is True

    hybrid_times = [row["hybrid_step_ms"] for row in res["chunks"]]
    for prev_t, next_t in zip(hybrid_times, hybrid_times[1:]):
        assert next_t > prev_t

    expected_max_chunk = max(
        (row["prefill_chunk_tokens"] for row in res["chunks"] if row["hybrid_step_ms"] <= 10.0),
        default=0,
    )
    assert res["max_sla_compliant_chunk_tokens"] == expected_max_chunk
    assert 0 < expected_max_chunk < 8192

    tight = analyze_chunked_prefill_decode_interference(
        case,
        decode_sequences=16,
        decode_context=32768,
        prefill_chunk_tokens=(512, 1024),
        sla_tpot_ms=0.01,
    )
    assert tight["max_sla_compliant_chunk_tokens"] == 0

    model_scope_res = analyze_chunked_prefill_decode_interference(
        _make_dense_case(layers=1, dp=1, tp=2),
        decode_sequences=1,
        decode_context=128,
        prefill_chunk_tokens=(0, 4),
        scope="model",
    )
    assert model_scope_res["chunks"][1]["hybrid_step_ms"] > model_scope_res["chunks"][0]["hybrid_step_ms"]

    with pytest.raises(ValueError):
        analyze_chunked_prefill_decode_interference(case, prefill_chunk_tokens=())
    with pytest.raises(ValueError):
        analyze_chunked_prefill_decode_interference(case, prefill_chunk_tokens=(-512,))
    with pytest.raises(ValueError):
        analyze_chunked_prefill_decode_interference(case, sla_tpot_ms=0.0)
