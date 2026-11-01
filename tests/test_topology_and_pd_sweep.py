"""Tier 0 unit tests for axiomatic multi-topology and disaggregated P/D Pareto solvers.

[Role] Verify deterministic (DP, TP, EP, PP, PCP) candidate enumeration,
  multi-topology hardware SLA/VRAM grid sweeps with Pareto dominance, and
  disaggregated Prefill/Decode instance ratio sweeps with physical KV handoff
  conservation.
[Invariants] Pure in-memory Tier 0 execution (< 5s total); zero subprocesses,
  zero network, zero sleep; exact conservation of KV handoff bytes and strict
  multi-objective Pareto non-dominance without heuristic weights.
[Timeouts & Locks] In-memory representative Ascend SKU and model geometries only.
[Side-Effect Policy] No file writes, device contexts, or global state mutation.
"""
from __future__ import annotations

from copy import deepcopy
import math
import pytest

from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.structure import parse_case
from mindie_sim.sweep import (
    compute_pareto_frontier,
    enumerate_parallel_candidates,
    sweep_pd_disaggregation_ratios,
    sweep_topology_grid,
)


def _make_dense_gqa_case(*, layers: int = 2, dp: int = 1, tp: int = 4) -> dict:
    """Build an in-memory dense GQA case on Ascend910_9362."""
    return {
        "machine": {
            "backend": "ascend",
            "sku": "Ascend910_9362",
            "num_devices": dp * tp,
            "ranks_per_node": dp * tp,
        },
        "model": {
            "name": "tier0-dense-gqa",
            "layers": layers,
            "hidden_size": 4096,
            "num_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "intermediate_size": 11008,
            "vocab_size": 64000,
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
                {"request_id": "req0", "prompt_tokens": 2048, "output_tokens": 16, "arrival_ms": 0.0}
            ]
        },
    }


def _make_mla_moe_case(*, layers: int = 2, dp: int = 1, tp: int = 4, ep: int = 4) -> dict:
    """Build an in-memory MLA + MoE case on Ascend910_9362."""
    return {
        "machine": {
            "backend": "ascend",
            "sku": "Ascend910_9362",
            "num_devices": dp * tp,
            "ranks_per_node": dp * tp,
        },
        "model": {
            "name": "tier0-mla-moe",
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
                {"request_id": "req0", "prompt_tokens": 2048, "output_tokens": 16, "arrival_ms": 0.0}
            ]
        },
    }


def test_enumerate_parallel_candidates_dense_and_moe_8_and_16_devices():
    """Verify deterministic candidate generation on dense and MoE models for 8 and 16 devices."""
    dense_case = parse_case(_make_dense_gqa_case())
    for n_dev, expected_tps in ((8, [1, 2, 4, 8]), (16, [1, 2, 4, 8, 16])):
        cands = enumerate_parallel_candidates(dense_case, num_devices=n_dev)
        assert [c["tp"] for c in cands] == expected_tps
        for c in cands:
            assert c["dp"] * c["tp"] * c["pp"] * c["pcp"] == n_dev
            assert c["ep"] == 1
            assert c["pp"] == 1
            assert c["pcp"] == 1

    # allow_pp=True and allow_pcp=True on dense GQA (layers=3 uneven PP2, num_devices=8)
    dense_3l = parse_case(_make_dense_gqa_case(layers=3))
    pp_pcp_cands = enumerate_parallel_candidates(
        dense_3l, num_devices=8, allow_pp=True, allow_pcp=True, candidate_tps=[2]
    )
    assert any(c["pp"] == 2 for c in pp_pcp_cands)
    assert any(c["pcp"] == 2 for c in pp_pcp_cands)
    assert all(c["dp"] * c["tp"] * c["pp"] * c["pcp"] == 8 for c in pp_pcp_cands)

    # Disallowing DP restricts to pure TP = num_devices
    no_dp_8 = enumerate_parallel_candidates(dense_case.model, num_devices=8, allow_dp=False)
    assert no_dp_8 == [{"dp": 1, "tp": 8, "ep": 1, "pp": 1, "pcp": 1}]

    # MoE model (16 routed experts) on 8 and 16 devices: includes all divisors of dp*tp dividing 16
    moe_case = parse_case(_make_mla_moe_case())
    moe_8 = enumerate_parallel_candidates(moe_case, num_devices=8)
    assert all(c["dp"] * c["tp"] * c["pp"] * c["pcp"] == 8 for c in moe_8)
    # For tp=2, dp=4 (dp*tp=8, experts=16) -> ep in [1, 2, 4, 8] (including ep=4!)
    tp2_eps = [c["ep"] for c in moe_8 if c["tp"] == 2]
    assert tp2_eps == [1, 2, 4, 8]
    # For tp=8, dp=1 -> ep in [1, 2, 4, 8]
    tp8_eps = [c["ep"] for c in moe_8 if c["tp"] == 8]
    assert tp8_eps == [1, 2, 4, 8]

    moe_16 = enumerate_parallel_candidates(moe_case, num_devices=16)
    assert all(c["dp"] * c["tp"] * c["pp"] * c["pcp"] == 16 for c in moe_16)
    # For tp=4, dp=4 on 16 experts -> ep in [1, 2, 4, 8, 16]
    tp4_eps_16 = [c["ep"] for c in moe_16 if c["tp"] == 4]
    assert tp4_eps_16 == [1, 2, 4, 8, 16]

    # Head divisibility filtering: 12 attention heads, 4 KV heads on 8 devices excludes tp=8
    cands_12h = enumerate_parallel_candidates(
        {"num_attention_heads": 12, "num_key_value_heads": 4, "n_routed_experts": 0},
        num_devices=8,
    )
    assert [c["tp"] for c in cands_12h] == [1, 2, 4]

    # MLA replicate mode permits tp=8 even when num_attention_heads=12
    cands_mla_rep = enumerate_parallel_candidates(
        {"num_attention_heads": 12, "num_key_value_heads": 1, "n_routed_experts": 8, "mla_replicate": True},
        num_devices=8,
        candidate_tps=[4, 8],
    )
    assert {c["tp"] for c in cands_mla_rep} == {4, 8}


def test_sweep_topology_grid_pareto_frontier_node_topology_and_infeasible_honesty():
    """Verify topology grid sweep metrics, node topology preservation, Pareto frontier, and error honesty."""
    raw_moe = _make_mla_moe_case(layers=2, dp=1, tp=4, ep=4)
    original = deepcopy(raw_moe)
    candidates = [
        {"dp": 4, "tp": 2, "ep": 4, "pp": 1, "pcp": 1},
        {"dp": 2, "tp": 4, "ep": 4, "pp": 1, "pcp": 1},
        {"dp": 1, "tp": 8, "ep": 8, "pp": 1, "pcp": 1},
        # Structurally invalid candidate: ep=32 exceeds dp*tp=8 and 16 experts
        {"dp": 1, "tp": 8, "ep": 32, "pp": 1, "pcp": 1},
    ]
    report = sweep_topology_grid(
        raw_moe,
        candidates=candidates,
        num_devices=8,
        sla_ms=15.0,
        context=4096,
        phase="decode",
        scope="body",
        output_budget=256,
        max_sequences_per_dp=16,
    )
    assert raw_moe == original
    assert report["schema"] == "mindie-sim.topology-sweep/1"
    assert report["num_devices"] == 8
    assert report["search_domain"]["num_devices"] == 8
    assert report["search_domain"]["candidate_count"] == 4
    assert len(report["candidates"]) == 4

    feasible_rows = [r for r in report["candidates"] if r["feasible"]]
    infeasible_rows = [r for r in report["candidates"] if r["status"] == "infeasible"]
    assert len(feasible_rows) == 3
    assert len(infeasible_rows) == 1
    assert infeasible_rows[0]["parallel"]["ep"] == 32
    assert infeasible_rows[0]["reason"]
    assert infeasible_rows[0]["pareto_optimal"] is False

    # Base case had ranks_per_node=4; scaling to 8, 12, or 16 devices preserves ranks_per_node=4!
    assert all(r["ranks_per_node"] == 4 for r in feasible_rows)
    scale12_rep = sweep_topology_grid(
        _make_dense_gqa_case(layers=2, dp=2, tp=4),
        candidates=[{"dp": 3, "tp": 4, "ep": 1, "pp": 1, "pcp": 1}],
        num_devices=12,
        sla_ms=15.0,
        context=4096,
        max_sequences_per_dp=4,
    )
    assert scale12_rep["candidates"][0]["ranks_per_node"] == 8
    scale16_rep = sweep_topology_grid(
        raw_moe,
        candidates=[{"dp": 4, "tp": 4, "ep": 4, "pp": 1, "pcp": 1}],
        num_devices=16,
        sla_ms=15.0,
        context=4096,
        max_sequences_per_dp=4,
    )
    assert scale16_rep["candidates"][0]["ranks_per_node"] == 4

    # Preserves custom link bandwidth overrides (including unmaterialized inter_node_bw_gbps) whether passed as a dict or parsed Case
    slow_link = _make_dense_gqa_case(layers=1, dp=1, tp=2)
    slow_link["machine"]["intra_node_bw_gbps"] = 10.0
    slow_link["machine"]["inter_node_bw_gbps"] = 8.0
    rep_from_dict = sweep_topology_grid(
        slow_link, candidates=[{"dp": 1, "tp": 4, "ep": 1, "pp": 1, "pcp": 1}], num_devices=4
    )
    rep_from_case = sweep_topology_grid(
        parse_case(slow_link), candidates=[{"dp": 1, "tp": 4, "ep": 1, "pp": 1, "pcp": 1}], num_devices=4
    )
    assert math.isclose(
        rep_from_dict["candidates"][0]["step_ms_at_b1"],
        rep_from_case["candidates"][0]["step_ms_at_b1"],
        rel_tol=1e-12,
    )
    assert math.isclose(
        rep_from_dict["candidates"][0]["tps_feasible"],
        rep_from_case["candidates"][0]["tps_feasible"],
        rel_tol=1e-12,
    )

    # Weight bytes per die must decrease monotonically as TP/EP increases from 2 -> 4 -> 8
    by_tp = {r["parallel"]["tp"]: r for r in feasible_rows}
    assert by_tp[8]["weight_bytes_per_die"] < by_tp[4]["weight_bytes_per_die"] < by_tp[2]["weight_bytes_per_die"]

    for r in feasible_rows:
        assert r["status"] == "feasible"
        assert r["b_sla"] >= 1
        assert r["b_vram_max"] >= 1
        assert r["b_feasible"] == min(r["b_sla"], r["b_vram_max"])
        assert r["total_feasible_concurrency"] == r["b_feasible"] * r["parallel"]["dp"]
        assert math.isclose(r["tps_per_die"], r["tps_feasible"] / 8.0, rel_tol=1e-9)
        assert 0.0 < r["network_mbu"] <= 1.0
        assert 0.0 < r["network_cube_mfu"] <= 1.0
        assert 0.0 <= r["network_vec_mfu"] <= 1.0

    # Verify axiomatic Pareto frontier properties
    frontier = report["pareto_frontier"]
    assert len(frontier) >= 1
    for a in frontier:
        assert a["pareto_optimal"] is True
        for b in frontier:
            if a is b:
                continue
            dominates = (
                b["tps_per_die"] >= a["tps_per_die"]
                and b["total_feasible_concurrency"] >= a["total_feasible_concurrency"]
                and b["step_ms_at_b1"] <= a["step_ms_at_b1"]
                and (
                    b["tps_per_die"] > a["tps_per_die"]
                    or b["total_feasible_concurrency"] > a["total_feasible_concurrency"]
                    or b["step_ms_at_b1"] < a["step_ms_at_b1"]
                )
            )
            assert not dominates

    # Auto-enumeration (candidates=None) and SLA-exceeded status check
    dense_1l = _make_dense_gqa_case(layers=1, dp=1, tp=4)
    auto_rep = sweep_topology_grid(
        dense_1l, num_devices=8, sla_ms=0.001, context=1024, max_sequences_per_dp=4
    )
    assert [r["parallel"]["tp"] for r in auto_rep["candidates"]] == [1, 2, 4, 8]
    assert all(r["status"] == "sla_or_vram_exceeded" and not r["feasible"] for r in auto_rep["candidates"])
    assert auto_rep["pareto_frontier"] == []


def test_sweep_pd_disaggregation_ratios_kv_conservation_handoff_tpot_and_oom():
    """Verify P/D ratio sweep across total_dies=16, KV conservation, p_dp/output_tokens-1 RPS, handoff TPOT, and OOM."""
    # Use dp=2, tp=2 (4 dies per prefill instance) to verify p_dp=2 multiplicity!
    prefill_case = _make_dense_gqa_case(layers=2, dp=2, tp=2)
    decode_case = _make_dense_gqa_case(layers=2, dp=1, tp=4)
    decode_case["runtime"]["max_num_seqs"] = 8

    pd_report = sweep_pd_disaggregation_ratios(
        prefill_case,
        decode_case,
        total_dies=16,
        prompt_tokens=1024,
        output_tokens=256,
        ttft_sla_ms=2000.0,
        tpot_sla_ms=25.0,
        kv_transfer_bw_gbps=100.0,
    )
    assert pd_report["schema"] == "mindie-sim.pd-sweep/1"
    assert pd_report["total_dies"] == 16

    # Verify p_dp=2 multiplicity in prefill_instance RPS and (output_tokens - 1) in decode_instance RPS
    p_inst = pd_report["prefill_instance"]
    d_inst = pd_report["decode_instance"]
    assert p_inst["hbm_ok"] is True
    assert math.isclose(p_inst["rps_per_instance"], (2 * 1000.0) / p_inst["step_ms"], rel_tol=1e-12)
    expected_d_rps = (d_inst["b_feasible"] * 1 * 1.0 * 1000.0) / (d_inst["step_ms_feasible"] * 255.0)
    assert math.isclose(d_inst["rps_per_instance"], expected_d_rps, rel_tol=1e-12)

    # Verify exact physical KV handoff bytes conservation and handoff contribution to effective_tpot_ms
    parsed_dec = parse_case(decode_case)
    expected_kv_bytes = (
        sum(Memory(parsed_dec).cache_bytes(1024, padded=False))
        * parsed_dec.runtime.parallel.tp
        * parsed_dec.runtime.parallel.pcp
    )
    expected_xfer_ms = expected_kv_bytes / (100.0 * 1e6)
    assert math.isclose(pd_report["kv_bytes_per_request"], expected_kv_bytes, rel_tol=1e-12)
    assert math.isclose(pd_report["kv_transfer_ms_per_request"], expected_xfer_ms, rel_tol=1e-12)
    assert math.isclose(
        pd_report["effective_tpot_ms"],
        pd_report["decode_step_ms"] + expected_xfer_ms / 255.0,
        rel_tol=1e-12,
    )
    assert math.isclose(
        pd_report["first_decode_itl_ms"],
        expected_xfer_ms + pd_report["decode_step_ms"],
        rel_tol=1e-12,
    )

    ratios = {r["pd_ratio"]: r for r in pd_report["candidates"]}
    assert {"1P:1D", "1P:2D", "1P:3D", "2P:1D", "2P:2D", "3P:1D"} <= set(ratios.keys())
    for r in pd_report["candidates"]:
        assert r["used_dies"] == r["n_prefill"] * 4 + r["n_decode"] * 4 <= 16
        assert r["feasible"] is True
        assert r["infeasible_reason"] is None
        assert math.isclose(r["effective_tpot_ms"], pd_report["effective_tpot_ms"], rel_tol=1e-12)
        assert math.isclose(r["first_decode_itl_ms"], pd_report["first_decode_itl_ms"], rel_tol=1e-12)
        assert math.isclose(
            r["sustainable_rps"],
            min(r["prefill_rps"], r["kv_transfer_rps"], r["decode_rps"]),
            rel_tol=1e-12,
        )
        assert math.isclose(r["sustainable_tps"], r["sustainable_rps"] * 256.0, rel_tol=1e-12)

    assert len(pd_report["pareto_frontier"]) >= 1
    assert pd_report["best_configuration"] in pd_report["pareto_frontier"]
    assert ratios["3P:1D"]["bottleneck_stage"] == "decode"

    # Moderate KV bandwidth (0.05 GB/s -> ~167.8ms transfer, +0.66ms TPOT): feasible, bottleneck="kv_transfer"
    kv_bound_feasible = sweep_pd_disaggregation_ratios(
        prefill_case, decode_case, total_dies=16, prompt_tokens=1024, output_tokens=256,
        ttft_sla_ms=2000.0, tpot_sla_ms=25.0, kv_transfer_bw_gbps=0.05,
    )
    assert all(r["feasible"] and r["bottleneck_stage"] == "kv_transfer" for r in kv_bound_feasible["candidates"])

    # Severe KV bandwidth throttling (0.001 GB/s -> ~8388ms transfer, +32.9ms TPOT > 25ms SLA): infeasible!
    kv_tpot_exceeded = sweep_pd_disaggregation_ratios(
        prefill_case, decode_case, total_dies=16, prompt_tokens=1024, output_tokens=256,
        ttft_sla_ms=2000.0, tpot_sla_ms=25.0, kv_transfer_bw_gbps=0.001,
    )
    assert all(
        not r["feasible"] and r["infeasible_reason"] == "effective_tpot_sla_exceeded"
        for r in kv_tpot_exceeded["candidates"]
    )

    # Prefill HBM OOM: reserve 64 GiB per rank so prefill cannot fit in HBM
    oom_prefill = deepcopy(prefill_case)
    oom_prefill["runtime"]["memory"] = {"workspace_bytes": 64 * 1024**3}
    oom_report = sweep_pd_disaggregation_ratios(
        oom_prefill, decode_case, total_dies=16, prompt_tokens=1024, output_tokens=256,
    )
    assert oom_report["prefill_instance"]["hbm_ok"] is False
    assert all(
        not r["feasible"] and r["infeasible_reason"] == "prefill_hbm_oom"
        for r in oom_report["candidates"]
    )

    # Decode max_model_len validation: prompt + output exceeding decode max_model_len is infeasible
    short_decode = deepcopy(decode_case)
    short_decode["workload"]["requests"][0]["prompt_tokens"] = 128
    short_decode["runtime"]["max_model_len"] = 1024
    short_rep = sweep_pd_disaggregation_ratios(
        prefill_case, short_decode, total_dies=16, prompt_tokens=128, output_tokens=2048,
    )
    assert all(
        not r["feasible"] and r["infeasible_reason"] == "decode_max_model_len_exceeded"
        for r in short_rep["candidates"]
    )
    oom_decode = deepcopy(decode_case)
    oom_decode["runtime"]["memory"] = {"workspace_bytes": 64 * 1024**3}
    out1_report = sweep_pd_disaggregation_ratios(
        prefill_case, oom_decode, total_dies=16, prompt_tokens=1024, output_tokens=1,
    )
    assert out1_report["kv_bytes_per_request"] == 0.0
    assert out1_report["kv_transfer_ms_per_request"] == 0.0
    assert out1_report["decode_step_ms"] == 0.0
    assert all(r["feasible"] and r["bottleneck_stage"] == "prefill" for r in out1_report["candidates"])

    # Source KV retention bound: when prefill max_num_seqs=1, prefill and transfer cannot overlap on a single replica
    one_slot_prefill = deepcopy(prefill_case)
    one_slot_prefill["runtime"]["max_num_seqs"] = 1
    one_slot_rep = sweep_pd_disaggregation_ratios(
        one_slot_prefill, decode_case, total_dies=16, prompt_tokens=1024, output_tokens=256,
        kv_transfer_bw_gbps=0.05,
    )
    p_step = one_slot_rep["prefill_instance"]["step_ms"]
    xfer_ms = one_slot_rep["kv_transfer_ms_per_request"]
    assert math.isclose(
        one_slot_rep["prefill_instance"]["rps_per_instance"],
        (2 * 1000.0) / (p_step + xfer_ms),
        rel_tol=1e-9,
    )


def test_compute_pareto_frontier_strict_dominance():
    """Verify mathematical Pareto dominance on synthetic rows."""
    rows = [
        {"id": "a", "feasible": True, "tps_per_die": 100.0, "total_feasible_concurrency": 32, "step_ms_at_b1": 10.0},
        {"id": "b", "feasible": True, "tps_per_die": 90.0, "total_feasible_concurrency": 32, "step_ms_at_b1": 11.0},
        {"id": "c", "feasible": True, "tps_per_die": 80.0, "total_feasible_concurrency": 64, "step_ms_at_b1": 12.0},
        {"id": "d", "feasible": False, "tps_per_die": 500.0, "total_feasible_concurrency": 128, "step_ms_at_b1": 1.0},
    ]
    frontier = compute_pareto_frontier(rows)
    assert [r["id"] for r in frontier] == ["a", "c"]
    assert rows[1]["pareto_optimal"] is False
    assert rows[3]["pareto_optimal"] is False
