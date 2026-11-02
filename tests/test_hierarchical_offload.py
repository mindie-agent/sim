"""Unit tests for hierarchical expert/KV offload, prefetch masking, and memory.

[Role]
  Establish the conditional expert-DDR activation contract, the auto-sizing
  invariant that converts an HBM deficit into a feasible resident plan, the
  monotonic effects of MTP hit rates, the v1-v4 KV prefetch mask ordering, and
  the backward-compatible resident-memory accounting in ``capacity.py``.

[Invariants]
  * A disabled policy leaves the baseline step, exposed time, and HBM bytes
    untouched, and ``Memory.to_dict()`` gains no extra key at zero ratios.
  * Auto-sized expert offload is the minimum ratio that clears the deficit for
    a single-PP-stage rank and makes ``Memory`` feasible.
  * Raising either MTP hit rate never raises transfer bytes or time.
  * v1 -> v4 prefetch masks are non-decreasing, so exposed KV offload time is
    non-increasing; each report equals baseline plus its own exposed time.
  * Synthetic link objects prove the effective/peak bandwidth helpers keep the
    effective value and expose the peak value via ``getattr`` fallback.

[Timeouts & Locks]
  Pure in-memory tests on four-layer catalog cases (< 30s); no device, network,
  subprocess, or file locks.

[Side-Effect Policy]
  No filesystem writes; composed cases are deep-copied before mutation.
"""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from dataclasses import replace

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.prediction.capacity import Memory, MemoryPolicy
from mindie_sim.prediction.efficiency import (
    _resolve_link_bw_bytes_per_ms,
    _resolve_link_peak_bw_bytes_per_ms,
)
from mindie_sim.prediction.offload import (
    OffloadPolicy,
    analyze_kv_cache_offload,
    prepare_expert_offload,
)
from mindie_sim.prediction.structure import parse_case


def _case(**overrides) -> dict:
    base = dict(
        model="qwen3.5-35b-a3b", hardware="a3", tp=4, dp=2, ep=8, layers=4,
        prompt_tokens=2048, output_tokens=64, concurrency=1,
    )
    base.update(overrides)
    return compose_case(**base)


def _tight_case(gpu_memory_utilization: float = 0.03, **overrides) -> dict:
    case = _case(**overrides)
    case["runtime"]["gpu_memory_utilization"] = gpu_memory_utilization
    case["runtime"]["max_model_len"] = 131072
    return case


def test_offload_policy_defaults_and_strict_validation() -> None:
    policy = OffloadPolicy()
    assert policy.expert_offload_enabled is False
    assert policy.force_expert_offload is False
    assert policy.expert_ddr_ratio == 0.0
    assert policy.ddr_bw_gbps == pytest.approx(200.0)
    assert policy.ddr_bw_util == pytest.approx(0.70)
    assert policy.ddr_latency_us == pytest.approx(5.0)
    assert policy.ssd_bw_gbps == pytest.approx(14.0)
    assert policy.ssd_bw_util == pytest.approx(0.70)
    assert policy.ssd_latency_us == pytest.approx(80.0)
    assert policy.kvc_ddr_ratio == 0.0
    assert policy.kvc_ssd_ratio == 0.0
    assert policy.kvc_prefetch_version == "v3"
    assert policy.mtp_ffn_hit_rate == 0.0
    assert policy.mtp_kvc_hit_rate == 0.0

    rebuilt = OffloadPolicy.from_mapping({
        "expert_offload_enabled": True,
        "force_expert_offload": True,
        "expert_ddr_ratio": 0.25,
        "ddr_bw_gbps": 100.0,
        "ddr_bw_util": 0.5,
        "ddr_latency_us": 1.0,
        "ssd_bw_gbps": 10.0,
        "ssd_bw_util": 0.6,
        "ssd_latency_us": 10.0,
        "kvc_ddr_ratio": 0.3,
        "kvc_ssd_ratio": 0.2,
        "kvc_prefetch_version": "v4",
        "mtp_ffn_hit_rate": 0.4,
        "mtp_kvc_hit_rate": 0.6,
    })
    assert rebuilt.expert_ddr_ratio == pytest.approx(0.25)
    assert rebuilt.kvc_prefetch_version == "v4"
    assert rebuilt.kvc_ddr_ratio + rebuilt.kvc_ssd_ratio == pytest.approx(0.5)
    assert OffloadPolicy.from_mapping(rebuilt) is rebuilt
    assert OffloadPolicy.from_mapping(None) == OffloadPolicy()
    assert OffloadPolicy.from_mapping({"expert_ddr_ratio": 0}) == OffloadPolicy()

    invalid = [
        {"expert_ddr_ratio": -0.01},
        {"expert_ddr_ratio": 1.01},
        {"ddr_bw_util": 1.5},
        {"ssd_bw_util": 0.0},
        {"mtp_ffn_hit_rate": -0.1},
        {"mtp_kvc_hit_rate": 2.0},
        {"ddr_bw_gbps": 0.0},
        {"ssd_bw_gbps": -1.0},
        {"ddr_latency_us": -1.0},
        {"kvc_prefetch_version": "v5"},
        {"expert_offload_enabled": 1},
        {"unknown_field": True},
        {"kvc_ddr_ratio": 0.7, "kvc_ssd_ratio": 0.4},
    ]
    for mapping in invalid:
        with pytest.raises(ValueError):
            OffloadPolicy.from_mapping(mapping)
    with pytest.raises(ValueError):
        OffloadPolicy(expert_ddr_ratio=1.5)
    with pytest.raises(ValueError):
        OffloadPolicy(kvc_ddr_ratio=0.8, kvc_ssd_ratio=0.8)


def test_expert_offload_inactive_when_hbm_is_sufficient() -> None:
    case = _case()
    result = prepare_expert_offload(case)
    assert result["schema"] == "mindie-sim.expert-offload/1"
    assert result["activated"] is False
    assert result["effective_expert_ddr_ratio"] == 0.0
    assert result["hbm_saved_bytes_per_rank"] == 0.0
    assert result["ddr_transfer_ms"] == 0.0
    assert result["exposed_expert_offload_ms"] == 0.0
    assert result["offloaded_step_ms"] == pytest.approx(result["baseline_step_ms"])
    assert result["feasible_after_offload"] is True
    assert result["hbm_deficit_bytes"] == 0.0

    enabled = prepare_expert_offload(case, {"expert_offload_enabled": True})
    assert enabled["activated"] is False
    assert enabled["exposed_expert_offload_ms"] == 0.0


def test_expert_offload_forced_activation_accounts_hbm_and_time() -> None:
    case = _case()
    policy = OffloadPolicy(force_expert_offload=True, expert_ddr_ratio=0.5)
    result = prepare_expert_offload(case, policy)
    assert result["activated"] is True
    assert result["activation_reason"] == "forced"
    assert result["effective_expert_ddr_ratio"] == pytest.approx(0.5)
    assert result["routed_expert_weight_bytes_per_rank"] > 0.0
    assert result["hbm_saved_bytes_per_rank"] == pytest.approx(
        0.5 * result["routed_expert_weight_bytes_per_rank"]
    )
    assert result["ddr_bytes_total"] > 0.0
    assert result["ddr_transfer_ms"] > 0.0
    assert result["mask_window_ms"] > 0.0
    assert result["mask_window_ms"] <= result["baseline_step_ms"]
    assert result["exposed_expert_offload_ms"] == pytest.approx(
        max(0.0, result["ddr_transfer_ms"] - result["mask_window_ms"])
    )
    assert result["offloaded_step_ms"] == pytest.approx(
        result["baseline_step_ms"] + result["exposed_expert_offload_ms"]
    )
    assert result["feasible_after_offload"] is True
    assert result["num_moe_layers"] == 4


def test_expert_offload_auto_sizing_makes_hbm_feasible() -> None:
    case = _tight_case()
    parsed = parse_case(case)
    assert Memory(parsed).to_dict()["status"] == "oom"

    result = prepare_expert_offload(case, {"expert_offload_enabled": True})
    assert result["activated"] is True
    assert result["activation_reason"] == "hbm_exceeded"
    ratio = result["effective_expert_ddr_ratio"]
    assert 0.0 < ratio <= 1.0
    assert result["hbm_deficit_bytes"] > 0.0
    assert result["hbm_saved_bytes_per_rank"] > 0.0
    assert result["feasible_after_offload"] is True
    assert result["net_hbm_saved_bytes_per_rank"] >= result["hbm_deficit_bytes"] - 1e-6
    assert ratio > result["hbm_deficit_bytes"] / result["routed_expert_weight_bytes_per_rank"]

    offloaded = Memory(parsed, MemoryPolicy(expert_ddr_ratio=ratio))
    report = offloaded.to_dict()
    assert report["status"] == "feasible"
    assert report["weights_bytes_per_stage_rank"] == pytest.approx(
        [value - ratio * routed for value, routed in zip(
            offloaded.gross_weights, offloaded.routed_expert_bytes)]
    )


def test_mtp_ffn_hit_rate_monotonically_reduces_expert_transfer() -> None:
    case = _case()
    results = [
        prepare_expert_offload(case, OffloadPolicy(
            force_expert_offload=True, expert_ddr_ratio=0.5, mtp_ffn_hit_rate=hit))
        for hit in (0.0, 0.5, 1.0)
    ]
    bytes_seen = [row["ddr_bytes_total"] for row in results]
    time_seen = [row["ddr_transfer_ms"] for row in results]
    assert bytes_seen[0] > bytes_seen[1] > bytes_seen[2] >= 0.0
    assert time_seen[0] > time_seen[1] > time_seen[2] >= 0.0
    assert all(row["mask_window_ms"] == pytest.approx(results[0]["mask_window_ms"])
               for row in results)


def test_kv_offload_prefetch_versions_are_monotone() -> None:
    case = _case()
    reports = {}
    for version in ("v1", "v2", "v3", "v4"):
        reports[version] = analyze_kv_cache_offload(
            case,
            OffloadPolicy(kvc_ddr_ratio=0.5, kvc_ssd_ratio=0.1,
                          kvc_prefetch_version=version),
            context=32768, sequences=32,
        )

    for version, report in reports.items():
        assert report["schema"] == "mindie-sim.kv-offload/1"
        assert report["activated"] is True
        assert report["kvc_prefetch_version"] == version
        assert report["kvc_hbm_ratio"] == pytest.approx(0.4)
        assert report["total_kv_bytes"] == pytest.approx(
            report["hbm_resident_kv_bytes"] + report["ddr_kv_bytes"] + report["ssd_kv_bytes"]
        )
        assert report["t_ssd_ms"] > report["t_ddr_ms"] > 0.0
        assert report["exposed_kv_offload_ms"] == pytest.approx(
            max(0.0, report["transfer_ms"] - report["mask_window_ms"])
        )
        assert report["offloaded_step_ms"] == pytest.approx(
            report["baseline_step_ms"] + report["exposed_kv_offload_ms"]
        )
        assert report["prefetch_masks_ms"][version] == pytest.approx(report["mask_window_ms"])

    v1, v2, v3, v4 = (reports[key] for key in ("v1", "v2", "v3", "v4"))
    assert v1["mask_window_ms"] == 0.0
    assert v2["mask_window_ms"] > v1["mask_window_ms"]
    assert v3["mask_window_ms"] > v2["mask_window_ms"]
    assert v4["mask_window_ms"] == pytest.approx(v4["baseline_step_ms"])
    exposed = [row["exposed_kv_offload_ms"] for row in (v1, v2, v3, v4)]
    assert exposed[0] > exposed[1] > exposed[2] > exposed[3] >= 0.0


def test_kv_offload_mtp_hit_rate_monotonically_reduces_transfer() -> None:
    case = _case()
    reports = [
        analyze_kv_cache_offload(
            case,
            OffloadPolicy(kvc_ddr_ratio=0.5, kvc_ssd_ratio=0.1,
                          kvc_prefetch_version="v3", mtp_kvc_hit_rate=hit),
            context=32768, sequences=32,
        )
        for hit in (0.0, 0.5, 1.0)
    ]
    assert [row["ddr_transfer_bytes"] for row in reports] == sorted(
        [row["ddr_transfer_bytes"] for row in reports], reverse=True)
    assert [row["ssd_transfer_bytes"] for row in reports] == sorted(
        [row["ssd_transfer_bytes"] for row in reports], reverse=True)
    assert reports[0]["t_ddr_ms"] > reports[1]["t_ddr_ms"] > reports[2]["t_ddr_ms"] >= 0.0
    assert reports[0]["t_ssd_ms"] > reports[1]["t_ssd_ms"] > reports[2]["t_ssd_ms"] >= 0.0
    assert reports[0]["exposed_kv_offload_ms"] > reports[1]["exposed_kv_offload_ms"] > reports[2]["exposed_kv_offload_ms"] >= 0.0
    assert all(row["mask_window_ms"] == pytest.approx(reports[0]["mask_window_ms"])
               for row in reports)


def test_kv_offload_default_policy_is_inactive() -> None:
    report = analyze_kv_cache_offload(_case())
    assert report["activated"] is False
    assert report["ddr_kv_bytes"] == 0.0
    assert report["ssd_kv_bytes"] == 0.0
    assert report["t_ddr_ms"] == 0.0
    assert report["t_ssd_ms"] == 0.0
    assert report["exposed_kv_offload_ms"] == 0.0
    assert report["offloaded_step_ms"] == pytest.approx(report["baseline_step_ms"])


def test_memory_default_dict_is_unchanged_and_offload_metadata_applies() -> None:
    case = _case()
    parsed = parse_case(case)
    baseline = Memory(parsed).to_dict()
    zero_policy = Memory(parsed, MemoryPolicy(
        expert_ddr_ratio=0.0, kvc_ddr_ratio=0.0, kvc_ssd_ratio=0.0)).to_dict()
    assert baseline == zero_policy
    assert "offload" not in baseline

    tight = _tight_case()
    auto = prepare_expert_offload(tight, {"expert_offload_enabled": True})
    ratio = auto["effective_expert_ddr_ratio"]
    tight["runtime"]["metadata"]["offload"] = {
        "expert_ddr_ratio": ratio,
        "kvc_ddr_ratio": 0.5,
    }
    loaded = Memory(parse_case(tight)).to_dict()
    assert loaded["status"] == "feasible"
    assert "offload" in loaded
    assert loaded["offload"]["expert_ddr_ratio"] == pytest.approx(ratio)
    assert loaded["offload"]["kvc_ddr_ratio"] == pytest.approx(0.5)
    assert loaded["offload"]["kvc_hbm_ratio"] == pytest.approx(0.5)
    assert loaded["weights_bytes_per_stage_rank"][0] < loaded["offload"]["gross_weights_bytes_per_stage_rank"][0]


def test_memory_kvc_ratio_scales_resize_and_capacity_is_bounded() -> None:
    case = _tight_case(gpu_memory_utilization=0.048, layers=8)
    parsed = parse_case(case)
    auto = prepare_expert_offload(case, {"expert_offload_enabled": True}, sequences=20)
    ratio = auto["effective_expert_ddr_ratio"]

    no_kvc = Memory(parsed, MemoryPolicy(expert_ddr_ratio=ratio))
    with_kvc = Memory(parsed, MemoryPolicy(expert_ddr_ratio=ratio, kvc_ddr_ratio=0.5))
    assert no_kvc.to_dict()["status"] == "feasible"
    assert with_kvc.to_dict()["status"] == "feasible"

    context = 131072
    raw = no_kvc.cache_bytes(context)
    scaled = with_kvc.cache_bytes(context)
    assert scaled == raw
    capacity = with_kvc.capacity[0][0]
    assert raw[0] > capacity, "the test context must not fit without KV tiering"
    assert scaled[0] * 0.5 <= capacity
    assert no_kvc.resize(0, "req", context) is False
    assert with_kvc.resize(0, "req", context) is True
    assert with_kvc.live[0][0] == pytest.approx(raw[0] * 0.5)


def test_link_effective_and_peak_bandwidth_helpers() -> None:
    case = _case()
    parsed = parse_case(case)
    effective = min(link.capacity_bytes_per_ms for link in parsed.machine.links)
    assert _resolve_link_bw_bytes_per_ms(parsed, "body") == pytest.approx(effective)
    assert _resolve_link_peak_bw_bytes_per_ms(parsed, "model") == pytest.approx(effective)

    synthetic = replace(parsed, machine=replace(parsed.machine, links=(
        SimpleNamespace(capacity_bytes_per_ms=100.0, effective_peak_capacity_bytes_per_ms=250.0),
        SimpleNamespace(capacity_bytes_per_ms=150.0, effective_peak_capacity_bytes_per_ms=300.0),
    )))
    assert _resolve_link_bw_bytes_per_ms(synthetic, "body") == pytest.approx(100.0)
    assert _resolve_link_peak_bw_bytes_per_ms(synthetic, "body") == pytest.approx(250.0)


def test_predict_schedules_offload_transfers_and_active_expert_sampling() -> None:
    from mindie_sim.prediction.api import predict
    from mindie_sim.prediction.offload import _expected_active_experts

    assert _expected_active_experts(8, 1, 8) == pytest.approx(8.0)

    base = _case(prompt_tokens=16, output_tokens=2)
    base_report = predict(base)

    with_offload = _case(prompt_tokens=16, output_tokens=2)
    with_offload["runtime"]["metadata"]["offload"] = {
        "kvc_ddr_ratio": 0.5,
        "kvc_ssd_ratio": 0.25,
        "kvc_prefetch_version": "v1",
        "ssd_bw_gbps": 0.1,
    }
    offload_report = predict(with_offload)
    assert offload_report["metrics"]["e2e_ms"]["mean"] > base_report["metrics"]["e2e_ms"]["mean"]


def test_predict_kv_prefetch_v1_through_v4_ordering_and_startup_latency() -> None:
    from mindie_sim.prediction.api import predict
    from mindie_sim.prediction.disaggregated import predict_pd

    latencies = {}
    for ver in ("v1", "v2", "v3", "v4"):
        c = _case(prompt_tokens=16, output_tokens=2)
        c["runtime"]["metadata"]["offload"] = {
            "kvc_ddr_ratio": 0.5,
            "kvc_prefetch_version": ver,
            "ddr_bw_gbps": 0.05,
            "ddr_latency_us": 10.0,
        }
        latencies[ver] = predict(c)["metrics"]["e2e_ms"]["mean"]
    assert latencies["v1"] >= latencies["v2"] >= latencies["v3"] >= latencies["v4"]

    high_lat = _case(prompt_tokens=16, output_tokens=2)
    high_lat["runtime"]["metadata"]["offload"] = {
        "kvc_ddr_ratio": 0.5,
        "kvc_prefetch_version": "v1",
        "ddr_bw_gbps": 0.05,
        "ddr_latency_us": 5000.0,
    }
    assert predict(high_lat)["metrics"]["e2e_ms"]["mean"] > latencies["v1"]

    p_pool = _case(prompt_tokens=16, output_tokens=2)
    d_pool = _case(prompt_tokens=16, output_tokens=2)
    pd_raw = {
        "model": p_pool["model"],
        "workload": p_pool["workload"],
        "connector": {"bytes_per_s": 1e12},
        "pools": {
            "prefill": {"machine": p_pool["machine"], "runtime": p_pool["runtime"]},
            "decode": {"machine": d_pool["machine"], "runtime": d_pool["runtime"]},
        },
    }
    pd_base = predict_pd(pd_raw)
    pd_off_raw = deepcopy(pd_raw)
    pd_off_raw["pools"]["decode"]["runtime"]["metadata"]["offload"] = {
        "kvc_ddr_ratio": 0.5, "ddr_bw_gbps": 0.05, "ddr_latency_us": 2000.0,
    }
    pd_with_off = predict_pd(pd_off_raw)
    assert pd_with_off["metrics"]["e2e_ms"]["mean"] > pd_base["metrics"]["e2e_ms"]["mean"]


def test_round3_single_layer_staging_infeasibility_source_grouped_gemm_and_hybrid_recurrent() -> None:
    from mindie_sim.prediction.api import predict

    one_layer = _case(layers=1, tp=1, dp=1, ep=1, prompt_tokens=16, output_tokens=2)
    one_layer["runtime"]["gpu_memory_utilization"] = 0.005
    prep_1l = prepare_expert_offload(
        one_layer, {"expert_offload_enabled": True, "expert_ddr_ratio": 1.0},
        tokens=256, sequences=64,
    )
    assert prep_1l["feasible_after_offload"] is False
    assert Memory(parse_case(one_layer), MemoryPolicy(expert_ddr_ratio=1.0), workspace_tokens=16384).to_dict()["status"] == "oom"

    src_base = _case(layers=1, quant="bf16", prompt_tokens=8, output_tokens=1)
    src_base["runtime"]["kernels"] = {"grouped_gemm": "ascendc_grouped_nd_bf16"}
    src_off = deepcopy(src_base)
    src_off["runtime"]["metadata"]["offload"] = {
        "force_expert_offload": True, "expert_ddr_ratio": 0.5,
        "ddr_bw_gbps": 0.05, "ddr_latency_us": 1000.0,
    }
    assert predict(src_off)["metrics"]["e2e_ms"]["mean"] > predict(src_base)["metrics"]["e2e_ms"]["mean"]

    hyb_base = _case(layers=1, prompt_tokens=8, output_tokens=2)
    hyb_off = deepcopy(hyb_base)
    hyb_off["runtime"]["metadata"]["offload"] = {
        "kvc_ddr_ratio": 0.5, "ddr_bw_gbps": 0.05, "ddr_latency_us": 1000.0,
    }
    assert predict(hyb_off)["metrics"]["e2e_ms"]["mean"] > predict(hyb_base)["metrics"]["e2e_ms"]["mean"]


