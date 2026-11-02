"""Tier 0 unit tests for AFD disaggregation, M-step pipeline and Pareto solver.

[Role] Verify cross-pool link short-board selection and overrides, the structural
  Attention/FFN stage split, the serial and M-step pipeline equations, the AF
  dispatch/combine byte and latency accounting, and the SLA-constrained AFD
  die-count Pareto solver.
[Invariants] Stage totals conserve body operator work, AF byte volume is
  batch_tokens * hidden_size * dtype_bytes per layer, dispatch/combine timing is
  link latency plus wire time, pipeline duration never exceeds the serial step,
  and Pareto rows are pairwise non-dominated without heuristic weights.
[Timeouts & Locks] Deterministic in-memory Tier 0 execution only; no device
  contexts, network, subprocesses, sleeps, file writes or global mutation.
[Side-Effect Policy] Fixtures build fresh dictionaries per test; production
  inputs are asserted to remain unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from mindie_sim.prediction.afd import (
    effective_pool_link,
    predict_afd,
    solve_afd_pareto,
)
from mindie_sim.prediction.efficiency import analyze_network_efficiency
from mindie_sim.prediction.structure import parse_case
from mindie_sim.sweep import solve_afd_pareto as exported_solve_afd_pareto


@dataclass(frozen=True)
class _MockLink:
    channel: str
    capacity_bytes_per_ms: float
    latency_us: float
    effective_peak_capacity_bytes_per_ms: float | None = None
    bw_utilization: float = 1.0
    name: str = ""


@dataclass(frozen=True)
class _MockHardware:
    links: tuple[_MockLink, ...]
    name: str = "mock"


def _make_dense_gqa_case(
    *, layers: int = 2, dp: int = 1, tp: int = 4,
    num_devices: int | None = None, ranks_per_node: int | None = None,
) -> dict:
    """Build an in-memory dense GQA case on Ascend910_9362."""
    devices = dp * tp if num_devices is None else num_devices
    return {
        "machine": {
            "backend": "ascend",
            "sku": "Ascend910_9362",
            "num_devices": devices,
            "ranks_per_node": devices if ranks_per_node is None else ranks_per_node,
        },
        "model": {
            "name": "afd-dense-gqa",
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


def _make_mla_moe_case(*, layers: int = 2, dp: int = 2, tp: int = 4, ep: int = 4) -> dict:
    """Build an in-memory MLA + MoE case on Ascend910_9362."""
    return {
        "machine": {
            "backend": "ascend",
            "sku": "Ascend910_9362",
            "num_devices": dp * tp,
            "ranks_per_node": dp * tp,
        },
        "model": {
            "name": "afd-mla-moe",
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


def test_effective_pool_link_short_board_uses_min_bandwidth_and_max_latency():
    """Verify min effective/peak and max latency across heterogeneous pools."""
    attn = _MockHardware((
        _MockLink("d2d_intra_node", 400e6, 2.0, 400e6, 0.8),
    ))
    ffn = _MockHardware((
        _MockLink("d2d_intra_node", 200e6, 5.0, 200e6, 0.7),
    ))
    info = effective_pool_link(attn, ffn, pool_link_tier="intra")
    assert info["pool_link_tier"] == "intra"
    assert info["effective_gbps"] == pytest.approx(140.0)
    assert info["peak_gbps"] == pytest.approx(200.0)
    assert info["effective_bytes_per_ms"] == pytest.approx(140e6)
    assert info["latency_us"] == pytest.approx(5.0)
    assert info["latency_ms"] == pytest.approx(0.005)
    assert info["effective_gbps"] == pytest.approx(min(400.0 * 0.8, 200.0 * 0.7))
    assert info["latency_us"] == pytest.approx(max(2.0, 5.0))

    # A link may declare its capacity as an already-effective rate instead of
    # a raw peak; both declaration forms must produce the same short board.
    attn_effective = _MockHardware((
        _MockLink("d2d_intra_node", 320e6, 2.0, 400e6, 0.8),
    ))
    ffn_effective = _MockHardware((
        _MockLink("d2d_intra_node", 140e6, 5.0, 200e6, 0.7),
    ))
    assert effective_pool_link(
        attn_effective, ffn_effective, pool_link_tier="intra") == info

    # Some declarations name the effective rate in the peak slot; the effective
    # rate must stay the same short board rather than being discounted twice.
    attn_rerated = _MockHardware((
        _MockLink("d2d_intra_node", 400e6, 2.0, 320e6, 0.8),
    ))
    ffn_rerated = _MockHardware((
        _MockLink("d2d_intra_node", 200e6, 5.0, 140e6, 0.7),
    ))
    rerated = effective_pool_link(attn_rerated, ffn_rerated, pool_link_tier="intra")
    assert rerated["effective_gbps"] == pytest.approx(140.0)
    assert rerated["latency_us"] == pytest.approx(5.0)

    # Single-node machines have no inter_node link, so the inter tier falls
    # back to the physical intra_node fabric.
    assert effective_pool_link(attn, ffn, pool_link_tier="inter") == {
        **info, "pool_link_tier": "inter"}


def test_effective_pool_link_real_ascend_links_and_tier_fallback():
    """Verify short-board extraction from materialized Ascend physical links."""
    multi_node = parse_case(_make_dense_gqa_case(tp=8, num_devices=8, ranks_per_node=4))
    intra = effective_pool_link(multi_node.machine, multi_node.machine, pool_link_tier="intra")
    inter = effective_pool_link(multi_node.machine, multi_node.machine, pool_link_tier="inter")
    assert intra["effective_gbps"] == pytest.approx(196.0)
    assert intra["latency_us"] == pytest.approx(2.5)
    assert inter["effective_gbps"] == pytest.approx(25.0)
    assert inter["latency_us"] == pytest.approx(5.0)

    single_node = parse_case(_make_dense_gqa_case(tp=4, num_devices=4, ranks_per_node=4))
    fallback = effective_pool_link(single_node.machine, single_node.machine, pool_link_tier="inter")
    assert fallback["effective_gbps"] == pytest.approx(196.0)
    assert fallback["latency_us"] == pytest.approx(2.5)


def test_effective_pool_link_overrides_and_validation():
    """Verify gbps/bw_utilization/latency overrides and argument validation."""
    attn = _MockHardware((_MockLink("d2d_intra_node", 400e6, 2.0, 400e6, 0.8),))
    ffn = _MockHardware((_MockLink("d2d_intra_node", 200e6, 5.0, 200e6, 0.7),))

    # Pure bandwidth override keeps the selected short-board utilization (0.7).
    gbps_only = effective_pool_link(attn, ffn, pool_link_tier="intra", link_override={"gbps": 100.0})
    assert gbps_only["peak_gbps"] == pytest.approx(100.0)
    assert gbps_only["effective_gbps"] == pytest.approx(70.0)
    assert gbps_only["latency_us"] == pytest.approx(5.0)

    # Pure utilization override applies to the short-board peak.
    util_only = effective_pool_link(attn, ffn, pool_link_tier="intra", link_override={"bw_utilization": 0.5})
    assert util_only["peak_gbps"] == pytest.approx(200.0)
    assert util_only["effective_gbps"] == pytest.approx(100.0)

    all_overrides = effective_pool_link(
        attn, ffn, pool_link_tier="intra",
        link_override={"gbps": 100.0, "bw_utilization": 0.5, "latency_us": 20.0},
    )
    assert all_overrides["effective_gbps"] == pytest.approx(50.0)
    assert all_overrides["peak_gbps"] == pytest.approx(100.0)
    assert all_overrides["latency_us"] == pytest.approx(20.0)
    assert all_overrides["latency_ms"] == pytest.approx(0.020)

    with pytest.raises(ValueError):
        effective_pool_link(attn, ffn, pool_link_tier="rack")
    with pytest.raises(ValueError):
        effective_pool_link(attn, ffn, pool_link_tier="intra", link_override={"bw_utilization": 0.0})
    with pytest.raises(ValueError):
        effective_pool_link(attn, ffn, pool_link_tier="intra", link_override={"bw_utilization": 1.5})
    with pytest.raises(ValueError):
        effective_pool_link(attn, ffn, pool_link_tier="intra", link_override={"gbps": 0.0})
    with pytest.raises(ValueError):
        effective_pool_link(attn, ffn, pool_link_tier="intra", link_override={"latency_us": -1.0})


def test_predict_afd_stage_split_conserves_body_work():
    """Verify every body operator belongs to exactly one pool for dense GQA and MLA MoE."""
    for case in (_make_dense_gqa_case(tp=4), _make_mla_moe_case(dp=2, tp=4, ep=4)):
        parsed = parse_case(case)
        report = predict_afd(case, case, m_steps=1, tokens=1, context=64, sequences=32)
        seqs_per_dp = 32 // parsed.runtime.parallel.dp
        attention = analyze_network_efficiency(
            parsed, phase="decode", tokens=1, context=64, sequences=seqs_per_dp,
            scope="body", run_graph=False,
        )
        assert report["schema"] == "mindie-sim.afd-prediction/1"
        assert report["attn_stage_ms"] > 0.0
        assert report["ffn_stage_ms"] > 0.0
        # No body operator is dropped or double-counted: the two pool totals
        # conserve the independently evaluated serial operator work.
        assert report["attn_stage_ms"] + report["ffn_stage_ms"] == pytest.approx(
            attention["operator_sum_ms"], rel=1e-12)
        assert report["attn_stage_ms"] < attention["operator_sum_ms"]
        assert report["ffn_stage_ms"] < attention["operator_sum_ms"]
        assert report["pipeline"]["serial_step_ms"] == pytest.approx(
            report["attn_stage_ms"] + report["ffn_stage_ms"] + report["af_comm_ms"], rel=1e-12)

        model_report = predict_afd(
            case, case, m_steps=1, tokens=1, context=64, sequences=32, scope="model")
        full_model = analyze_network_efficiency(
            parsed, phase="decode", tokens=1, context=64, sequences=seqs_per_dp,
            scope="model", run_graph=False,
        )
        assert model_report["attn_stage_ms"] > report["attn_stage_ms"]
        assert model_report["attn_stage_ms"] + model_report["ffn_stage_ms"] == pytest.approx(
            full_model["operator_sum_ms"], rel=1e-12)


def test_predict_afd_m_step_pipeline_af_comm_and_link_degradation():
    """Verify serial, M=2 and M>=3 pipeline equations plus degraded link behavior."""
    attn_case = _make_dense_gqa_case(layers=2, tp=4)
    ffn_case = _make_dense_gqa_case(layers=2, tp=8)
    common = dict(tokens=64, context=2048, sequences=60)
    serial = predict_afd(attn_case, ffn_case, m_steps=1, **common)
    partial = predict_afd(attn_case, ffn_case, m_steps=2, **common)
    pipelined = predict_afd(attn_case, ffn_case, m_steps=3, **common)

    s_attn_ms, s_ffn_ms, s_comm_ms = (
        serial["attn_stage_ms"], serial["ffn_stage_ms"], serial["af_comm_ms"])
    assert serial["pipeline"]["mode"] == "serial"
    assert serial["pipeline"]["fill_drain_bubble_ms"] == 0.0
    assert serial["step_duration_ms"] == pytest.approx(s_attn_ms + s_ffn_ms + s_comm_ms, rel=1e-12)

    p_attn_ms, p_ffn_ms, p_comm_ms = (
        partial["attn_stage_ms"], partial["ffn_stage_ms"], partial["af_comm_ms"])
    assert partial["pipeline"]["mode"] == "partial_overlap"
    partial_steady = max(p_attn_ms + p_ffn_ms, p_comm_ms)
    assert partial["pipeline"]["steady_state_ms"] == pytest.approx(partial_steady, rel=1e-12)
    assert partial["pipeline"]["fill_drain_bubble_ms"] == pytest.approx(
        (p_attn_ms + p_ffn_ms + p_comm_ms - partial_steady) / 2.0, rel=1e-12)
    assert partial["step_duration_ms"] == pytest.approx(
        partial_steady + partial["pipeline"]["fill_drain_bubble_ms"], rel=1e-12)

    attn_ms, ffn_ms, comm_ms = (
        pipelined["attn_stage_ms"], pipelined["ffn_stage_ms"], pipelined["af_comm_ms"])
    assert pipelined["pipeline"]["mode"] == "full_overlap"
    steady = max(attn_ms, ffn_ms, comm_ms)
    assert pipelined["pipeline"]["steady_state_ms"] == pytest.approx(steady, rel=1e-12)
    assert pipelined["pipeline"]["fill_drain_bubble_ms"] == pytest.approx(
        (attn_ms + ffn_ms + comm_ms - steady) / 3.0, rel=1e-12)
    assert pipelined["step_duration_ms"] == pytest.approx(
        steady + pipelined["pipeline"]["fill_drain_bubble_ms"], rel=1e-12)
    assert pipelined["step_duration_ms"] < partial["step_duration_ms"] < serial["step_duration_ms"]

    # Independent AF byte/latency accounting from the effective link and dtypes.
    link = pipelined["pool_link"]
    batch_tokens, hidden = 64 * 60, 4096
    dispatch_bytes = batch_tokens * hidden * 2.0
    combine_bytes = batch_tokens * hidden * 2.0
    dispatch_layer_ms = 3 * link["latency_ms"] + dispatch_bytes / link["effective_bytes_per_ms"]
    combine_layer_ms = 3 * link["latency_ms"] + combine_bytes / link["effective_bytes_per_ms"]
    assert pipelined["af_comm"]["dispatch_bytes_per_layer"] == pytest.approx(dispatch_bytes)
    assert pipelined["af_comm"]["combine_bytes_per_layer"] == pytest.approx(combine_bytes)
    assert pipelined["af_comm"]["dispatch_ms_per_layer"] == pytest.approx(dispatch_layer_ms, rel=1e-12)
    assert pipelined["af_comm"]["combine_ms_per_layer"] == pytest.approx(combine_layer_ms, rel=1e-12)
    assert pipelined["af_comm"]["af_dispatch_ms"] == pytest.approx(dispatch_layer_ms, rel=1e-12)
    assert pipelined["af_comm"]["af_combine_ms"] == pytest.approx(combine_layer_ms, rel=1e-12)
    assert pipelined["af_dispatch_ms"] == pytest.approx(dispatch_layer_ms, rel=1e-12)
    assert pipelined["af_combine_ms"] == pytest.approx(combine_layer_ms, rel=1e-12)
    assert pipelined["af_comm_ms"] == pytest.approx(
        (dispatch_layer_ms + combine_layer_ms) * parse_case(attn_case).model.layers, rel=1e-12)

    # Throughput and die accounting are direct functions of the step.
    total_dies = pipelined["attn_pool"]["dies"] + pipelined["ffn_pool"]["dies"]
    assert total_dies == 12
    assert pipelined["throughput"]["total_dies"] == total_dies
    expected_tps = batch_tokens * 1000.0 / pipelined["step_duration_ms"]
    assert pipelined["tokens_per_sec"] == pytest.approx(expected_tps, rel=1e-12)
    assert pipelined["tokens_per_sec_per_total_die"] == pytest.approx(expected_tps / total_dies, rel=1e-12)

    # A degraded 5 GB/s link dominates both stages and flips the feasibility guard.
    slow = predict_afd(
        attn_case, ffn_case, m_steps=3, **common,
        link_override={"gbps": 5.0},
    )
    assert pipelined["comm_hidden_ok"] is True
    assert slow["af_comm_ms"] > max(slow["attn_stage_ms"], slow["ffn_stage_ms"])
    assert slow["comm_hidden_ok"] is False
    assert slow["step_duration_ms"] > pipelined["step_duration_ms"]

    # Explicit dispatch dtype changes only the dispatch byte volume.
    fp32_dispatch = predict_afd(
        attn_case, ffn_case, m_steps=3, **common,
        af_dispatch_dtype="fp32",
    )
    assert fp32_dispatch["af_comm"]["dispatch_bytes_per_layer"] == pytest.approx(2 * dispatch_bytes)
    assert fp32_dispatch["af_comm"]["combine_bytes_per_layer"] == pytest.approx(combine_bytes)

    # SLA and balance constraints are exact comparisons against the report.
    assert pipelined["meets_tpot_sla"] is True
    tight = predict_afd(
        attn_case, ffn_case, m_steps=3, **common,
        sla_tpot_ms=pipelined["step_duration_ms"] / 2.0,
    )
    assert tight["meets_tpot_sla"] is False
    assert math.isclose(
        pipelined["compute_balance_gap"],
        abs(attn_ms - ffn_ms) / max(attn_ms, ffn_ms, 1e-12),
        rel_tol=1e-12,
    )

    for kwargs in (
        {"m_steps": 0}, {"tokens": 0}, {"context": -1}, {"sequences": 0},
        {"sla_tpot_ms": 0.0}, {"phase": "serving"}, {"scope": "layer"},
    ):
        with pytest.raises(ValueError):
            predict_afd(attn_case, ffn_case, **kwargs)


def test_solve_afd_pareto_grid_sla_filter_and_recommendation():
    """Verify the (attn_dies, ffn_dies, m_steps) grid, SLA filter and Pareto frontier."""
    assert exported_solve_afd_pareto is solve_afd_pareto
    base = _make_dense_gqa_case(layers=2, tp=2)
    base["runtime"]["max_num_seqs"] = 512
    base["runtime"]["max_num_batched_tokens"] = 65536
    report = solve_afd_pareto(
        base, base,
        attn_die_counts=(2, 4),
        ffn_die_counts=(2, 4),
        m_steps_options=(1, 3),
        sequences=120,
        context=4096,
        link_override={"gbps": 10.0},
        sla_tpot_ms=50.0,
    )
    assert report["schema"] == "mindie-sim.afd-pareto/1"
    expected_grid = {
        (attn, ffn, m)
        for attn in (2, 4)
        for ffn in (2, 4)
        for m in (1, 3)
    }
    observed_grid = {
        (row["attn_dies"], row["ffn_dies"], row["m_steps"])
        for row in report["candidates"]
    }
    assert observed_grid == expected_grid
    assert report["search_domain"]["candidate_count"] == len(report["candidates"]) == 8

    steps_by_pool = {}
    for row in report["candidates"]:
        assert row["total_dies"] == row["attn_dies"] + row["ffn_dies"]
        assert row["meets_tpot_sla"] is True
        assert row["feasible"] is True
        assert row["status"] == "feasible"
        assert row["infeasible_reason"] is None
        assert row["tokens_per_sec"] == pytest.approx(
            120 * 1000.0 / row["step_duration_ms"], rel=1e-12)
        assert row["tokens_per_sec_per_total_die"] == pytest.approx(
            row["tokens_per_sec"] / row["total_dies"], rel=1e-12)
        steps_by_pool[(row["attn_dies"], row["ffn_dies"], row["m_steps"])] = row["step_duration_ms"]
    for (attn, ffn, m_steps), step_ms in steps_by_pool.items():
        if m_steps == 1:
            assert steps_by_pool[(attn, ffn, 3)] < step_ms

    frontier = report["pareto_frontier"]
    recommended = report["recommended"]
    assert len(frontier) >= 1
    assert recommended is not None
    assert recommended in frontier
    assert recommended["tokens_per_sec_per_total_die"] == max(
        row["tokens_per_sec_per_total_die"] for row in report["candidates"])
    for candidate in report["candidates"]:
        dominated = any(
            other is not candidate
            and other["tokens_per_sec_per_total_die"] >= candidate["tokens_per_sec_per_total_die"]
            and other["step_duration_ms"] <= candidate["step_duration_ms"]
            and (
                other["tokens_per_sec_per_total_die"] > candidate["tokens_per_sec_per_total_die"]
                or other["step_duration_ms"] < candidate["step_duration_ms"]
            )
            for other in report["candidates"]
        )
        assert candidate["pareto_optimal"] is (candidate in frontier)
        assert candidate["pareto_optimal"] is not dominated

    # An impossible SLA keeps every evaluated row and honestly reports no frontier.
    tight = solve_afd_pareto(
        base, base,
        attn_die_counts=(2, 4),
        ffn_die_counts=(2, 4),
        m_steps_options=(1, 3),
        sequences=32,
        context=64,
        sla_tpot_ms=1e-9,
    )
    assert all(row["meets_tpot_sla"] is False for row in tight["candidates"])
    assert all(row["status"] == "sla_exceeded" for row in tight["candidates"])
    assert tight["pareto_frontier"] == []
    assert tight["recommended"] is None

    with pytest.raises(ValueError):
        solve_afd_pareto(base, base, attn_die_counts=(), m_steps_options=(1, 3))
    with pytest.raises(ValueError):
        solve_afd_pareto(base, base, attn_die_counts=(2,), m_steps_options=(0,))


def test_solve_afd_pareto_scales_ffn_expert_parallelism():
    """Verify the FFN pool scales EP while the Attention pool keeps EP at one."""
    base = _make_mla_moe_case(layers=2, dp=1, tp=2, ep=2)
    report = solve_afd_pareto(
        base, base,
        attn_die_counts=(2, 4),
        ffn_die_counts=(2, 4, 8),
        m_steps_options=(1, 3),
        sequences=16,
        context=32,
        sla_tpot_ms=50.0,
    )
    assert len(report["candidates"]) == 2 * 3 * 2
    assert all(row["status"] == "feasible" for row in report["candidates"])
    assert all(row["attn_parallel"]["ep"] == 1 for row in report["candidates"])
    assert any(row["ffn_parallel"]["ep"] > 1 for row in report["candidates"])
    assert max(row["ffn_parallel"]["ep"] for row in report["candidates"]) == 8
    assert report["pareto_frontier"]
    assert report["recommended"] in report["pareto_frontier"]


def test_afd_request_latency_sla_and_pool_memory_feasibility():
    """Verify TPOT SLA checks full request latency and Pareto filters OOM pools."""
    attn_case = _make_dense_gqa_case(layers=2, tp=4)
    ffn_case = _make_dense_gqa_case(layers=2, tp=8)
    pipelined = predict_afd(attn_case, ffn_case, m_steps=3, tokens=1, context=64, sequences=64)
    assert pipelined["request_latency_ms"] == pytest.approx(pipelined["pipeline"]["serial_step_ms"])
    assert pipelined["step_duration_ms"] < pipelined["request_latency_ms"]

    mid_sla = 0.5 * (pipelined["step_duration_ms"] + pipelined["request_latency_ms"])
    mid_report = predict_afd(
        attn_case, ffn_case, m_steps=3, tokens=1, context=64, sequences=64,
        sla_tpot_ms=mid_sla,
    )
    assert mid_report["meets_tpot_sla"] is False

    oom_attn = _make_dense_gqa_case(layers=64, tp=2)
    oom_ffn = _make_dense_gqa_case(layers=64, tp=8)
    oom_report = solve_afd_pareto(
        oom_attn, oom_ffn,
        attn_die_counts=(2,), ffn_die_counts=(8,), m_steps_options=(3,),
        sequences=128, context=65536, sla_tpot_ms=1000.0,
    )
    assert oom_report["candidates"][0]["feasible"] is False
    assert oom_report["candidates"][0]["status"] == "oom"
    assert oom_report["pareto_frontier"] == []


def test_afd_rejects_mismatched_pool_models_and_checks_role_specific_memory():
    """Verify Attention/FFN pools must share model geometry and only count resident weights."""
    attn_case = _make_mla_moe_case(layers=2, dp=1, tp=2, ep=2)
    mismatched = _make_mla_moe_case(layers=4, dp=1, tp=2, ep=2)
    with pytest.raises(ValueError, match="afd_model_contract_mismatch"):
        predict_afd(attn_case, mismatched)
    with pytest.raises(ValueError, match="afd_model_contract_mismatch"):
        solve_afd_pareto(attn_case, mismatched, attn_die_counts=(2,), ffn_die_counts=(2,))


def test_afd_single_sequence_m_step_invariance_switch_hops_and_structure_block():
    """Verify sequences=1 clamps effective M-steps, switch hops add latency, and structure_block wins."""
    from mindie_sim.prediction.afd import _operator_pool

    attn_case = _make_dense_gqa_case(layers=2, tp=4)
    ffn_case = _make_dense_gqa_case(layers=2, tp=8)
    s1_m1 = predict_afd(attn_case, ffn_case, m_steps=1, tokens=1, context=64, sequences=1)
    s1_m30 = predict_afd(attn_case, ffn_case, m_steps=30, tokens=1, context=64, sequences=1)
    assert s1_m30["pipeline"]["effective_m_steps"] == 1
    assert s1_m30["step_duration_ms"] == pytest.approx(s1_m1["step_duration_ms"], rel=1e-12)
    assert s1_m30["tokens_per_sec"] == pytest.approx(s1_m1["tokens_per_sec"], rel=1e-12)

    hw_hops = {"name": "hop_hw", "links": [{
        "name": "inter_node:0", "channel": "inter_node", "capacity_bytes_per_ms": 50e6,
        "latency_us": 2.0, "switch_hops": 3, "switch_hop_latency_us": 1.5,
    }]}
    link = effective_pool_link(hw_hops, hw_hops, pool_link_tier="inter")
    assert link["latency_us"] == pytest.approx(2.0 + 3 * 1.5)

    assert _operator_pool(
        {"op_id": "s0.l0.expert_attention_helper", "family": "vector",
         "recipe": "helper", "structure_block": "attention"},
        "ffn",
    ) == ("attention", "attention")


