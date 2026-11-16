"""Hierarchical expert/KV offload through canonical Memory and prediction.

[Role]
  Establish the conditional expert-DDR policy contract, the resident-memory
  accounting in ``Capacity.Memory``, and the actual transfer/predict effects of
  a declared offload policy.

[Invariants]
  * A disabled policy leaves the baseline ``Memory.to_dict()`` untouched and
    gains no ``offload`` key at zero ratios.
  * An explicit expert ratio reduces resident weight bytes by the declared
    share while staging remains allocated and recorded.
  * KV tiering keeps the logical cache extent but bounds live HBM residency
    so ``resize`` can admit a context that would otherwise not fit.
  * ``predict`` emits the declared expert-DDR transfer event and never charges
    an unselected transfer.

[Timeouts & Locks]
  Pure in-memory tests on four-layer catalog cases (< 30s); no device, network,
  subprocess or file locks.

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
from mindie_sim.prediction.offload import OffloadPolicy
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









def test_memory_default_dict_is_unchanged_and_offload_metadata_applies() -> None:
    case = _case()
    parsed = parse_case(case)
    baseline = Memory(parsed).to_dict()
    zero_policy = Memory(parsed, MemoryPolicy(
        expert_ddr_ratio=0.0, kvc_ddr_ratio=0.0, kvc_ssd_ratio=0.0)).to_dict()
    assert baseline == zero_policy
    assert "offload" not in baseline

    tight = _tight_case()
    ratio = 0.75
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
    ratio = 0.5

    no_kvc = Memory(parsed, MemoryPolicy(expert_ddr_ratio=ratio))
    with_kvc = Memory(parsed, MemoryPolicy(expert_ddr_ratio=ratio, kvc_ddr_ratio=0.5))
    assert no_kvc.to_dict()["status"] == "feasible"
    assert with_kvc.to_dict()["status"] == "feasible"

    context = 327680
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

    base = _case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2)
    base_report = predict(base)

    with_offload = _case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2)
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
        c = _case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2)
        c["runtime"]["metadata"]["offload"] = {
            "kvc_ddr_ratio": 0.5,
            "kvc_prefetch_version": ver,
            "ddr_bw_gbps": 0.05,
            "ddr_latency_us": 10.0,
        }
        latencies[ver] = predict(c)["metrics"]["e2e_ms"]["mean"]
    assert latencies["v1"] >= latencies["v2"] >= latencies["v3"] >= latencies["v4"]

    high_lat = _case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2)
    high_lat["runtime"]["metadata"]["offload"] = {
        "kvc_ddr_ratio": 0.5,
        "kvc_prefetch_version": "v1",
        "ddr_bw_gbps": 0.05,
        "ddr_latency_us": 5000.0,
    }
    assert predict(high_lat)["metrics"]["e2e_ms"]["mean"] > latencies["v1"]

    p_pool = _case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2)
    d_pool = _case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2)
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
    memory = Memory(parse_case(one_layer), MemoryPolicy(expert_ddr_ratio=1.0),
                    workspace_tokens=16384).to_dict()
    assert memory["status"] == "oom"
    assert memory["offload"]["expert_ddr_ratio"] == pytest.approx(1.0)

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


def test_memory_expert_offload_reduces_resident_weights_and_records_share():
    base = Memory(parse_case(_case())).to_dict()
    offloaded = Memory(parse_case(_case()), MemoryPolicy(
        expert_ddr_ratio=0.5)).to_dict()
    assert offloaded["weights_bytes_per_stage_rank"][0] < (
        offloaded["offload"]["gross_weights_bytes_per_stage_rank"][0])
    assert offloaded["weights_bytes_per_stage_rank"][0] < (
        base["weights_bytes_per_stage_rank"][0])
    saved = offloaded["offload"]["expert_hbm_saved_bytes_per_stage_rank"][0]
    assert saved == pytest.approx(
        offloaded["offload"]["routed_expert_bytes_per_stage_rank"][0] * 0.5)
    headroom = offloaded["physical_rank_headroom_bytes_per_replica_stage"]
    assert all(value >= 0 for replica in headroom for stage in replica
               for value in stage)


def test_predict_emits_declared_expert_ddr_transfer():
    from mindie_sim.prediction.api import predict

    base = _case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2)
    offload = _case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2)
    offload["runtime"]["metadata"]["offload"] = {
        "expert_offload_enabled": True, "force_expert_offload": True,
        "expert_ddr_ratio": 0.5, "ddr_bw_gbps": 0.05, "ddr_latency_us": 1000.0,
    }
    report = predict(offload)
    events = [event for event in report["events"]
              if event["task"].endswith(":expert_ddr_offload")]
    assert events, "a selected expert offload must transfer on the shared engine"
    assert any(event["demand"] for event in events)
    memory = report["memory"]["offload"]
    assert memory["expert_ddr_ratio"] == pytest.approx(0.5)
    assert memory["expert_hbm_saved_bytes_per_stage_rank"][0] > 0
    assert not any(event["task"].endswith(":expert_ddr_offload")
                   for event in predict(base)["events"])
