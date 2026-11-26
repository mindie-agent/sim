"""Hierarchical KV/state offload tiers and bandwidth feedback.
[Role]
  Pin the host-DRAM/SSD/MemFabric tier contract end to end: policy validation,
  resident-weight accounting, KV-tier capacity admission, actual producer-bound
  state-access projection, prefetch/writeback event ordering, sparse-MLA
  offload placement and the calibrated MemFabric copy/exchange kernels.
[Invariants]
  * A disabled policy leaves the baseline memory view untouched; an explicit
    expert ratio reduces resident bytes by that share while staging remains.
  * Tier bytes are projected from actual occurrence-bound access rows, once per
    physical producer, and never from model layer counts or call ids.
  * DDR/SSD/HBM transfers are real Engine events ordered after their source or
    grant and before their reader; padding has no state access.
  * Sparse-MLA offload keeps only index state on device, enforces host-DRAM
    capacity, and emits H2D misses only for long contexts.
  * MemFabric copy, exchange and gather costs track the calibrated tables.
[Timeouts & Locks] In-memory Engine fixtures; no threads or I/O. [Side-Effect Policy]
  No filesystem, environment or network access; cases are deep-copied before mutation.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
import gc
import math
import weakref
from types import SimpleNamespace
import pytest
from _support.memory_cases import SPARSE_OFFLOAD_SKU, draft_case, linear_work, sparse_mla_case
from mindie_sim.catalog.draft_geometry import native_draft_layer_plans
from mindie_sim import predict
from mindie_sim.compose import compose_case
from mindie_sim.prediction.capacity import Memory, MemoryPolicy
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.disaggregated import predict_pd
from mindie_sim.prediction.efficiency import (
    _resolve_link_bw_bytes_per_ms, _resolve_link_peak_bw_bytes_per_ms,
)
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.offload import (
    OffloadPolicy, StateAccess, _accesses_from_ops, _expected_active_experts,
    bind_offload_access_events, project_state_tiers, release_offload_access,
    schedule_draft_kv_writeback, schedule_kv_iteration_offload,
    schedule_kv_layer_offload, schedule_kv_layer_writeback,
)
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.scheduler import Scheduler
from mindie_sim.prediction.operators.sparse_indexer.sparse_kv_offload import copy_sfa_pool_capacity
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
pytestmark = pytest.mark.tier0
TIER = OffloadPolicy(kvc_ddr_ratio=0.5, kvc_ssd_ratio=0.25, ddr_bw_gbps=200.0, ssd_bw_gbps=14.0)
INVALID_POLICIES = ({"expert_ddr_ratio": -0.01}, {"expert_ddr_ratio": 1.01}, {"ddr_bw_util": 1.5},
                    {"ssd_bw_util": 0.0}, {"ddr_bw_gbps": 0.0}, {"ddr_latency_us": -1.0},
                    {"kvc_prefetch_version": "v5"}, {"expert_offload_enabled": 1},
                    {"unknown_field": True}, {"kvc_ddr_ratio": 0.7, "kvc_ssd_ratio": 0.4})
def catalog_case(**overrides):
    base = dict(model="qwen3.5-35b-a3b", hardware="a3", tp=4, dp=2, ep=8,
                layers=4, prompt_tokens=2048, output_tokens=64, concurrency=1)
    base.update(overrides)
    return compose_case(**base)
def tight_case(gpu_memory_utilization=0.032, **overrides):
    case = catalog_case(**overrides); case["runtime"]["gpu_memory_utilization"] = gpu_memory_utilization; case["runtime"]["max_model_len"] = 131072
    return case
def with_offload(case_spec, *, ddr=0.5, ssd=0.25, ddr_bw_gbps=200.0, ssd_bw_gbps=14.0):
    cfg = {"kvc_ddr_ratio": ddr, "kvc_ssd_ratio": ssd, **({"ddr_bw_gbps": ddr_bw_gbps} if ddr > 0 and ddr_bw_gbps is not None else {}), **({"ssd_bw_gbps": ssd_bw_gbps} if ssd > 0 else {})}
    return replace(case_spec, runtime=replace(case_spec.runtime, metadata=tuple(dict(case_spec.runtime.metadata).items()) + (("offload", cfg),)))
def execution(case_spec):
    return Execution(case_spec, CostModel(case_spec), include_serving_io=False)
def work(views=None, ops=()):
    views = views or (RequestView("r", 1, 4, True, "decode"),)
    return SimpleNamespace(plan=IterationPlan(tuple(views), iteration=0), ops=tuple(ops))
def access(*, bytes, rank=None, occurrence="occ", request="r", scope="target",
           direction="write", storage="gqa_kv", layer=0):
    return StateAccess(
        occurrence=occurrence, producer=occurrence, request=request, stage=0,
        layer=layer, direction=direction, position="cache", output=storage,
        storage=storage, span=((0, 1),), rank=rank, scope=scope, bytes=bytes)
def serviced(case_execution, events):
    case_execution.engine.advance(until=tuple(events), first=False)
    return dict(case_execution.engine.serviced)
def memfabric_case():
    return parse_case({
        "machine": {"backend": "ascend", "sku": SPARSE_OFFLOAD_SKU, "num_devices": 1},
        "model": {"name": "memfabric-geometry", "layers": 1, "hidden_size": 512,
                  "num_heads": 8, "num_kv_heads": 8, "head_dim": 64,
                  "intermediate_size": 1024, "vocab_size": 1024},
        "workload": {"requests": [{"prompt_tokens": 16, "output_tokens": 1}]},
        "runtime": {},
    })
def test_offload_policy_defaults_and_rebuild():
    policy = OffloadPolicy()
    assert (policy.expert_offload_enabled, policy.force_expert_offload,
            policy.expert_ddr_ratio, policy.kvc_ddr_ratio, policy.kvc_ssd_ratio) == (
        False, False, 0.0, 0.0, 0.0)
    assert policy.ddr_bw_gbps is None and (policy.ddr_bw_util, policy.ddr_latency_us) == pytest.approx((0.70, 5.0))
    assert policy.ssd_bw_gbps is None and (policy.ssd_bw_util, policy.ssd_latency_us) == pytest.approx((0.70, 80.0))
    with pytest.raises(ValueError, match="missing_required_scenario_fields:ssd_bw_gbps"):
        OffloadPolicy(kvc_ssd_ratio=0.25)
    assert (policy.kvc_prefetch_version, policy.mtp_ffn_hit_rate,
            policy.mtp_kvc_hit_rate) == ("v3", 0.0, 0.0)
    rebuilt = OffloadPolicy.from_mapping({
        "expert_offload_enabled": True, "force_expert_offload": True,
        "expert_ddr_ratio": 0.25, "ddr_bw_gbps": 100.0, "ddr_bw_util": 0.5,
        "ddr_latency_us": 1.0, "ssd_bw_gbps": 10.0, "ssd_bw_util": 0.6,
        "ssd_latency_us": 10.0, "kvc_ddr_ratio": 0.3, "kvc_ssd_ratio": 0.2,
        "kvc_prefetch_version": "v4", "mtp_ffn_hit_rate": 0.4, "mtp_kvc_hit_rate": 0.6})
    assert rebuilt.expert_ddr_ratio == pytest.approx(0.25); assert rebuilt.kvc_ddr_ratio + rebuilt.kvc_ssd_ratio == pytest.approx(0.5); assert rebuilt.kvc_prefetch_version == "v4"; assert OffloadPolicy.from_mapping(rebuilt) is rebuilt; assert OffloadPolicy.from_mapping(None) == OffloadPolicy(); assert OffloadPolicy.from_mapping({"expert_ddr_ratio": 0}) == OffloadPolicy()
@pytest.mark.parametrize("mapping", INVALID_POLICIES)
def test_offload_policy_rejects_invalid_ratios_and_unknown_fields(mapping):
    with pytest.raises(ValueError):
        OffloadPolicy.from_mapping(mapping)
    with pytest.raises(ValueError):
        OffloadPolicy(expert_ddr_ratio=1.5)
    with pytest.raises(ValueError):
        OffloadPolicy(kvc_ddr_ratio=0.8, kvc_ssd_ratio=0.8)
def test_memory_offload_metadata_and_resident_weight_shares():
    parsed = parse_case(catalog_case()); baseline = Memory(parsed).to_dict()
    zero = Memory(parsed, MemoryPolicy(expert_ddr_ratio=0.0, kvc_ddr_ratio=0.0,
                                      kvc_ssd_ratio=0.0)).to_dict()
    assert baseline == zero and "offload" not in baseline
    tight = tight_case(); tight["runtime"]["metadata"]["offload"] = {"expert_ddr_ratio": 0.75, "kvc_ddr_ratio": 0.5}; loaded = Memory(parse_case(tight)).to_dict()
    assert loaded["status"] == "feasible" and "offload" in loaded; assert loaded["offload"]["expert_ddr_ratio"] == pytest.approx(0.75); assert loaded["offload"]["kvc_ddr_ratio"] == pytest.approx(0.5); assert loaded["offload"]["kvc_hbm_ratio"] == pytest.approx(0.5); assert loaded["weights_bytes_per_stage_rank"][0] < (
        loaded["offload"]["gross_weights_bytes_per_stage_rank"][0])
    base = Memory(parse_case(catalog_case())).to_dict()
    offloaded = Memory(parse_case(catalog_case()), MemoryPolicy(
        expert_ddr_ratio=0.5)).to_dict()
    assert offloaded["weights_bytes_per_stage_rank"][0] < (
        offloaded["offload"]["gross_weights_bytes_per_stage_rank"][0])
    assert offloaded["weights_bytes_per_stage_rank"][0] < (
        base["weights_bytes_per_stage_rank"][0])
    saved = offloaded["offload"]["expert_hbm_saved_bytes_per_stage_rank"][0]
    assert saved == pytest.approx(
        offloaded["offload"]["routed_expert_bytes_per_stage_rank"][0] * 0.5)
    headroom = offloaded["physical_rank_headroom_bytes_per_replica_stage"]
    assert all(value >= 0 for replica in headroom for stage in replica for value in stage)
def test_kvc_ratio_scales_resize_and_capacity_is_bounded():
    parsed = parse_case(tight_case(gpu_memory_utilization=0.038, layers=8, mtp_proposal_tokens=0)); no_kvc = Memory(parsed, MemoryPolicy(expert_ddr_ratio=0.75)); with_kvc = Memory(parsed, MemoryPolicy(expert_ddr_ratio=0.75, kvc_ddr_ratio=0.5))
    assert no_kvc.to_dict()["status"] == with_kvc.to_dict()["status"] == "feasible"
    context = 2 * 327680; raw, scaled = no_kvc.cache_bytes(context), with_kvc.cache_bytes(context)
    assert scaled == raw
    capacity = with_kvc.capacity[0][0]
    assert raw[0] > capacity and scaled[0] * 0.5 <= capacity; assert no_kvc.resize(0, "req", context) is False; assert with_kvc.resize(0, "req", context) is True; assert with_kvc.live[0][0] == pytest.approx(raw[0] * 0.5)
def test_link_effective_and_peak_bandwidth_helpers():
    parsed = parse_case(catalog_case())
    assert _resolve_link_bw_bytes_per_ms(parsed, "body") == pytest.approx(122e6)
    assert _resolve_link_peak_bw_bytes_per_ms(parsed, "model") == pytest.approx(196e6)
    synthetic = replace(parsed, machine=replace(parsed.machine, links=(
        SimpleNamespace(capacity_bytes_per_ms=100.0, effective_peak_capacity_bytes_per_ms=250.0),
        SimpleNamespace(capacity_bytes_per_ms=150.0, effective_peak_capacity_bytes_per_ms=300.0))))
    assert _resolve_link_bw_bytes_per_ms(synthetic, "body") == pytest.approx(100.0); assert _resolve_link_peak_bw_bytes_per_ms(synthetic, "body") == pytest.approx(250.0)
def test_predict_emits_declared_offload_transfers_and_prefetch_order():
    assert _expected_active_experts(8, 1, 8) == pytest.approx(8.0)
    base = catalog_case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2); base_report = predict(base); offload = deepcopy(base); offload["runtime"]["metadata"]["offload"] = {
        "kvc_ddr_ratio": 0.5, "kvc_ssd_ratio": 0.25,
        "kvc_prefetch_version": "v1", "ssd_bw_gbps": 0.1}
    assert predict(offload)["metrics"]["e2e_ms"]["mean"] > (
        base_report["metrics"]["e2e_ms"]["mean"])
    latencies = {}
    for version in ("v1", "v2", "v3", "v4"):
        case = catalog_case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2)
        case["runtime"]["metadata"]["offload"] = {
            "kvc_ddr_ratio": 0.5, "kvc_prefetch_version": version,
            "ddr_bw_gbps": 0.05, "ddr_latency_us": 10.0}
        latencies[version] = predict(case)["metrics"]["e2e_ms"]["mean"]
    assert latencies["v1"] >= latencies["v2"] >= latencies["v3"] >= latencies["v4"]
    high_latency = catalog_case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2); high_latency["runtime"]["metadata"]["offload"] = {
        "kvc_ddr_ratio": 0.5, "kvc_prefetch_version": "v1",
        "ddr_bw_gbps": 0.05, "ddr_latency_us": 5000.0}
    assert predict(high_latency)["metrics"]["e2e_ms"]["mean"] > latencies["v1"]
    expert = catalog_case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2); expert["runtime"]["metadata"]["offload"] = {
        "expert_offload_enabled": True, "force_expert_offload": True,
        "expert_ddr_ratio": 0.5, "ddr_bw_gbps": 0.05, "ddr_latency_us": 1000.0}
    report = predict(expert, event_limit=20000); events = [event for event in report["events"] if event["task"].endswith(":expert_ddr_offload")]
    assert events and any(event["demand"] for event in events); assert report["memory"]["offload"]["expert_hbm_saved_bytes_per_stage_rank"][0] > 0; assert not any(event["task"].endswith(":expert_ddr_offload")
                   for event in predict(base)["events"])
def test_pd_decode_pool_offload_and_single_layer_staging():
    prefill = catalog_case(layers=1, tp=2, dp=1, ep=2, prompt_tokens=16, output_tokens=2); decode = deepcopy(prefill)
    pd_raw = {"model": prefill["model"], "workload": prefill["workload"],
              "connector": {"bytes_per_s": 1e12},
              "pools": {"prefill": {"machine": prefill["machine"], "runtime": prefill["runtime"]},
                        "decode": {"machine": decode["machine"], "runtime": decode["runtime"]}}}
    pd_base = predict_pd(pd_raw); pd_offload = deepcopy(pd_raw); pd_offload["pools"]["decode"]["runtime"]["metadata"]["offload"] = {
        "kvc_ddr_ratio": 0.5, "ddr_bw_gbps": 0.05, "ddr_latency_us": 2000.0}
    assert predict_pd(pd_offload)["metrics"]["e2e_ms"]["mean"] > pd_base["metrics"]["e2e_ms"]["mean"]
    one_layer = catalog_case(layers=1, tp=1, dp=1, ep=1, prompt_tokens=16, output_tokens=2); one_layer["runtime"]["gpu_memory_utilization"] = 0.005
    memory = Memory(parse_case(one_layer), MemoryPolicy(expert_ddr_ratio=1.0),
                    workspace_tokens=16384).to_dict()
    assert memory["status"] == "oom"; assert memory["offload"]["expert_ddr_ratio"] == pytest.approx(1.0)
    grouped = catalog_case(layers=1, quant="bf16", prompt_tokens=8, output_tokens=1); grouped["runtime"]["kernels"] = {"grouped_gemm": "ascendc_grouped_nd_bf16"}; grouped_offload = deepcopy(grouped); grouped_offload["runtime"]["metadata"]["offload"] = {
        "force_expert_offload": True, "expert_ddr_ratio": 0.5,
        "ddr_bw_gbps": 0.05, "ddr_latency_us": 1000.0}
    assert predict(grouped_offload)["metrics"]["e2e_ms"]["mean"] > (
        predict(grouped)["metrics"]["e2e_ms"]["mean"])
    hybrid = catalog_case(layers=1, prompt_tokens=8, output_tokens=2); hybrid_offload = deepcopy(hybrid); hybrid_offload["runtime"]["metadata"]["offload"] = {
        "kvc_ddr_ratio": 0.5, "ddr_bw_gbps": 0.05, "ddr_latency_us": 1000.0}
    assert predict(hybrid_offload)["metrics"]["e2e_ms"]["mean"] > (
        predict(hybrid)["metrics"]["e2e_ms"]["mean"])
def test_state_tier_projection_uses_actual_occurrence_rows():
    payload = {"occurrence": "dcp:rank0", "producer": "dcp:rank0", "request": "r",
               "stage": 0, "layer": 0, "direction": "write", "rank": 0,
               "span": [[0, 1]], "storage": "gqa_kv", "scope": "target", "bytes": 32}
    projected, = project_state_tiers((payload,), TIER)
    assert projected["span"] == ((0, 1),); assert (projected["ddr_bytes"], projected["ssd_bytes"], projected["hbm_bytes"]) == (
        16.0, 8.0, 24.0)
    assert len({projected["span"], ((0, 1),)}) == 1
    rank0 = access(bytes=32, rank=0, occurrence="dcp:rank0"); rank1 = access(bytes=32, rank=1, occurrence="dcp:rank1"); rows = project_state_tiers((rank0, rank1), TIER)
    assert [(row["rank"], row["ddr_bytes"], row["ssd_bytes"], row["hbm_bytes"])
            for row in rows] == [(0, 16.0, 8.0, 24.0), (1, 16.0, 8.0, 24.0)]
    assert rank0.key != rank1.key
    case_spec = with_offload(parse_case(_raw_case()))
    for item in (rank0, rank1):
        case_execution = execution(case_spec)
        events = schedule_kv_layer_writeback(
            case_execution, "dcp", 0, 0, 0, work(),
            case_execution.engine.submit("producer", latency=1.0), state_accesses=(item,))
        result = serviced(case_execution, events)
        assert result["offload_ddr_d2h@r0:s0"] == pytest.approx(16.0); assert result["offload_ssd@r0:s0"] == pytest.approx(8.0); assert result["memory@r0:s0"] == pytest.approx(24.0)
    target = access(bytes=256, occurrence="target:cache_store"); draft = access(bytes=192, occurrence="draft:cache_store", scope="draft"); projected = project_state_tiers((target, draft), TIER)
    assert sum(row["ddr_bytes"] for row in projected) == pytest.approx(224.0); assert sum(row["ssd_bytes"] for row in projected) == pytest.approx(112.0); assert sum(row["hbm_bytes"] for row in projected) == pytest.approx(336.0)
    tail = access(bytes=1024, occurrence="index_pool_tail_store", storage="index_pool_tail"); projected, = project_state_tiers((tail,), TIER)
    assert (projected["ddr_bytes"], projected["ssd_bytes"], projected["hbm_bytes"]) == (
        512.0, 256.0, 768.0)
    alias = replace(access(bytes=0), direction="alias"); zero, = project_state_tiers((alias,), TIER)
    assert (zero["ddr_bytes"], zero["ssd_bytes"], zero["hbm_bytes"]) == (0.0, 0.0, 0.0)
def _raw_case(output=1, **kwargs):
    from _support.memory_cases import case
    return case(output=output, **kwargs)
def _gdn_op():
    return SimpleNamespace(
        op_id="gdn:recurrent", family="recurrent_vector", stage=0, layer=0,
        requests=("pad", "r"), mte3_bytes=0.0, hbm_bytes=0.0,
        metadata=(("heads", 8), ("dk", 128), ("dv", 128), ("state_element_bytes", 4.0),
                  ("gdn_state_contract", {
                      "phase": "decode", "direction": "token", "target_kind": "recurrent",
                      "read_versions": (None, 16), "checkpoints": (0, 1),
                      "publish_versions": (None, 17), "targets": ("", "r"),
                      "state_layout": "V,K"})))
def _index_tail_op():
    return SimpleNamespace(
        op_id="index_pool_tail_store", family="cache_store", stage=0, layer=0,
        requests=("r",), mte3_bytes=1024.0, hbm_bytes=2048.0,
        metadata=(("cache_kind", "index_pool_tail"), ("retained_rows", 1),
                  ("index_width", 128), ("cache_visible", True)))
def test_state_access_discovery_uses_physical_contracts():
    op = _gdn_op(); pad = RequestView("pad", 1, 16, True, "decode", padding=True); real = RequestView("r", 1, 16, True, "decode"); accesses = _accesses_from_ops(SimpleNamespace(plan=IterationPlan((pad, real)), ops=(op,)), (op,))
    assert len(accesses) == 2
    read_access = next(item for item in accesses if item.direction == "read"); write_access = next(item for item in accesses if item.direction == "write")
    assert (read_access.request, read_access.bytes) == ("r", 524288.0); assert (write_access.request, write_access.bytes) == ("r", 524288.0)
    for item in (read_access, write_access):
        projected, = project_state_tiers((item,), TIER)
        assert (projected["ddr_bytes"], projected["ssd_bytes"], projected["hbm_bytes"]) == (
            262144.0, 131072.0, 393216.0)
    assert len(project_state_tiers((read_access, write_access), TIER,
                                   expected_bytes=1048576.0)) == 2
    with pytest.raises(ValueError, match="total_bytes_mismatch"):
        project_state_tiers((read_access, write_access), TIER, expected_bytes=1024.0)
    swapped_contract = dict(dict(op.metadata)["gdn_state_contract"])
    swapped_contract.update(read_versions=(16, None), checkpoints=(1, 0),
                            publish_versions=(17, None), targets=("r", ""))
    swapped = SimpleNamespace(**{**op.__dict__, "requests": ("r", "pad"),
                                 "metadata": tuple({**dict(op.metadata),
                                                    "gdn_state_contract": swapped_contract}.items())})
    swapped_accesses = _accesses_from_ops(
        SimpleNamespace(plan=IterationPlan((real, pad)), ops=(swapped,)), (swapped,))
    assert all(item.request == "r" and item.bytes == 524288.0 for item in swapped_accesses)
    tail = _index_tail_op()
    discovered = _accesses_from_ops(
        SimpleNamespace(plan=IterationPlan((RequestView("r", 1, 4, True, "decode"),)),
                        ops=(tail,)), (tail,))
    assert len(discovered) == 1 and discovered[0].bytes == 1024.0; assert discovered[0].storage == "index_pool_tail"
    aggregate = SimpleNamespace(**{**op.__dict__, "metadata": op.metadata + (
        ("state_read_bytes", 524288.0), ("state_write_bytes", 524288.0))})
    assert len(_accesses_from_ops(
        SimpleNamespace(plan=IterationPlan((pad, real)), ops=(aggregate,)), (aggregate,))) == 2
    bad = SimpleNamespace(**{**op.__dict__, "metadata": op.metadata + (("state_read_bytes", 1.0),)})
    with pytest.raises(ValueError, match="state_access_contract_total_mismatch"):
        _accesses_from_ops(SimpleNamespace(plan=IterationPlan((pad, real)), ops=(bad,)), (bad,))
def test_offload_uses_scheduler_memory_and_releases_registry():
    case_spec = with_offload(parse_case(_raw_case())); memory, case_execution = Memory(case_spec), execution(case_spec); scheduler = Scheduler(case_spec, list(case_spec.workload.requests), memory); scheduler.bind_execution(case_execution)
    assert scheduler.memory is memory and case_execution.memory is memory
    item = access(bytes=32, occurrence="shared-memory:writer"); producer = case_execution.engine.submit("producer", latency=1.0); same_work = work()
    schedule_kv_layer_writeback(case_execution, "same", 0, 0, 0, same_work, producer,
                                state_accesses=(item,))
    assert case_execution._offload_memory is memory; assert scheduler.grant_state([("shared-grant", 4096.0)]) is True; assert memory.granted[0][0] == 4096.0; assert release_offload_access(case_execution, work()) == 0; assert release_offload_access(case_execution, same_work) == 1
    for index in range(32):
        item = access(bytes=1024, occurrence="ring", storage="index_pool_tail")
        producer = case_execution.engine.submit(f"ring:{index}", latency=1.0)
        ring_work = work()
        events = schedule_kv_layer_writeback(
            case_execution, f"ring-{index}", 0, 0, 0, ring_work, producer,
            state_accesses=(item,))
        assert len(events) == 2; assert schedule_kv_iteration_offload(
            case_execution, f"ring-fb-{index}", 0, ring_work, producer, (),
            {"ring": producer}, scheduled_layers={(0, 0)},
            state_accesses=(item,)) == ()
        assert release_offload_access(case_execution, ring_work) == 1
        case_execution.engine.advance(until=events, first=False)
    assert not case_execution._offload_access_batches; assert not getattr(case_execution, "_offload_actual_accesses", {})
    second_case = with_offload(parse_case(_raw_case())); memory_a, memory_b = Memory(second_case), Memory(second_case); execution_a, execution_b = execution(second_case), execution(second_case); scheduler_a = Scheduler(second_case, list(second_case.workload.requests), memory_a); scheduler_b = Scheduler(second_case, list(second_case.workload.requests), memory_b); scheduler_a.bind_execution(execution_a); scheduler_b.bind_execution(execution_b)
    assert scheduler_b.grant_state([("only-b", 4096.0)]) is True; assert memory_a.granted[0][0] == 0.0 and memory_b.granted[0][0] == 4096.0
    event = execution_a.engine.submit("a", latency=1.0)
    schedule_kv_layer_writeback(execution_a, "a", 0, 0, 0, work(), event,
                                state_accesses=(access(bytes=32, occurrence="a"),))
    assert execution_a._offload_memory is memory_a; assert execution_b._offload_memory is memory_b and memory_a.granted[0][0] == 0.0
    refs = []
    for _ in range(32):
        parsed = parse_case(_raw_case())
        memory = Memory(parsed)
        scheduler = Scheduler(parsed, list(parsed.workload.requests), memory)
        refs.append((weakref.ref(parsed), weakref.ref(memory)))
    del scheduler, parsed, memory; gc.collect()
    assert all(case_ref() is None and memory_ref() is None for case_ref, memory_ref in refs)
def test_issued_registry_splits_writers_and_ignores_old_layer_hints():
    case_spec = with_offload(parse_case(_raw_case())); case_execution = execution(case_spec); a, b = access(bytes=32, occurrence="writer:A"), access(bytes=64, occurrence="writer:B"); event_a = case_execution.engine.submit("writer:A", latency=1.0); event_b = case_execution.engine.submit("writer:B", latency=2.0); current_work = work()
    first = schedule_kv_layer_writeback(
        case_execution, "one", 0, 0, 0, current_work, event_a, state_accesses=(a,))
    assert serviced(case_execution, first)["offload_ddr_d2h@r0:s0"] == pytest.approx(16.0); assert schedule_kv_iteration_offload(
        case_execution, "fallback", 0, current_work, event_a, (),
        {"writer:A": event_a}, scheduled_layers={(0, 0)}, state_accesses=(a, b)) == ()
    later = schedule_kv_iteration_offload(
        case_execution, "fallback2", 0, current_work, event_a, (),
        {"writer:A": event_a, "writer:B": event_b}, scheduled_layers={(0, 0)},
        state_accesses=(a, b))
    result = serviced(case_execution, later)
    assert result["offload_ddr_d2h@r0:s0"] == pytest.approx(16.0 + 32.0); assert result["offload_ssd@r0:s0"] == pytest.approx(8.0 + 16.0); assert len(later) == 2
    first_access = access(bytes=32, occurrence="call0"); second_access = access(bytes=32, occurrence="call0", direction="write"); work0, work1 = work(), work(); event0 = case_execution.engine.submit("writer:0", latency=1.0); event1 = case_execution.engine.submit("writer:1", latency=2.0)
    first = schedule_kv_layer_writeback(case_execution, "batch0", 0, 0, 0, work0,
                                        event0, state_accesses=(first_access,))
    assert release_offload_access(case_execution, work0) > 0
    second = schedule_kv_layer_writeback(case_execution, "batch1", 0, 0, 0, work1,
                                         event1, state_accesses=(second_access,))
    before_batches = case_execution.engine.serviced["offload_ddr_d2h@r0:s0"]
    assert serviced(case_execution, first + second)["offload_ddr_d2h@r0:s0"] - before_batches == pytest.approx(32.0)
    ring0 = access(bytes=1024, occurrence="ring", storage="index_pool_tail"); ring1 = access(bytes=1024, occurrence="ring", storage="index_pool_tail"); ring_work0, ring_work1 = work(), work(); before = serviced(case_execution, first + second)
    ring_events0 = schedule_kv_layer_writeback(
        case_execution, "ring0", 0, 0, 0, ring_work0, event0, state_accesses=(ring0,))
    ring_events1 = schedule_kv_layer_writeback(
        case_execution, "ring1", 0, 0, 0, ring_work1, event1, state_accesses=(ring1,))
    assert len(ring_events0) == len(ring_events1) == 2
    result = serviced(case_execution, ring_events0 + ring_events1)
    assert result["offload_ddr_d2h@r0:s0"] - before["offload_ddr_d2h@r0:s0"] == pytest.approx(1024.0); assert result["offload_ssd@r0:s0"] - before["offload_ssd@r0:s0"] == pytest.approx(512.0)
def test_prefetch_strategies_respect_source_and_grant_edges():
    case_spec = with_offload(parse_case(_raw_case())); case_execution = execution(case_spec); item = access(bytes=32, occurrence="fetch", direction="read"); input_ready = case_execution.engine.submit("input", latency=0.0); source_ready = case_execution.engine.submit("source", release=3.0); grant_ready = case_execution.engine.submit("grant", release=5.0); early = case_execution.engine.submit("early", latency=0.0); unrelated_late = case_execution.engine.submit("unrelated-late", release=10.0)
    for version in ("v1", "v2", "v3", "v4"):
        case_execution.offload_policy = replace(
            case_execution.offload_policy, kvc_prefetch_version=version)
        events = schedule_kv_layer_offload(
            case_execution, f"edge-{version}", 0, 0, 0, work(), input_ready,
            (input_ready, unrelated_late), {0: early}, state_accesses=(item,),
            source_ready=source_ready, grant_ready=grant_ready)
        assert events
        case_execution.engine.advance(until=(events[-1],), first=False)
        assert min(event.time for event in events) >= 5.0; assert max(event.time for event in events) < 10.0
        reader = case_execution.engine.submit(
            f"reader-{version}", latency=0.0, after=events)
        case_execution.engine.advance(until=(reader,), first=False)
        assert reader.time >= max(event.time for event in events)
    draft_execution = execution(case_spec)
    draft_execution.offload_policy = replace(
        draft_execution.offload_policy, kvc_prefetch_version="v1")
    read = access(bytes=64, occurrence="draft:read", scope="draft",
                  direction="read", storage="draft_kv", layer=-1)
    source = draft_execution.engine.submit("source", latency=1.0)
    fetch = schedule_kv_layer_offload(
        draft_execution, "draft", 0, 0, -1, work(), source, (source,), {},
        state_accesses=(read,))
    assert fetch
    reader = draft_execution.engine.submit("reader", latency=0.0, after=fetch); draft_execution.engine.advance(until=(reader,), first=False)
    assert fetch[0].time >= source.time and reader.time >= fetch[0].time; assert draft_execution.engine.serviced["offload_ddr_h2d@r0:s0"] == pytest.approx(32.0); assert draft_execution.engine.serviced["offload_ssd@r0:s0"] == pytest.approx(16.0)
def test_writer_binding_selects_actual_producer_output():
    case_spec = with_offload(parse_case(_raw_case())); case_execution = execution(case_spec); current_work = work()
    call0 = SimpleNamespace(op_id="call0", family="attention", stage=0, layer=0,
                            requests=("r",), mte3_bytes=0.0, hbm_bytes=0.0, metadata=())
    store_b = SimpleNamespace(op_id="storeB", family="state", stage=0, layer=0,
                              requests=("r",), mte3_bytes=0.0, hbm_bytes=0.0, metadata=())
    binding = StateAccess(occurrence="call0", producer="storeB", request="r", stage=0,
                          layer=0, direction="write", output="out", storage="kv/key",
                          span=((0, 1),), bytes=64.0)
    e1 = case_execution.engine.submit("call0", release=1.0); e_b = case_execution.engine.submit("storeB.out", release=5.0)
    assert schedule_kv_iteration_offload(
        case_execution, "only-call0", 0, current_work, e1, (call0, store_b),
        {"call0": e1}, state_accesses=(binding,)) == ()
    assert case_execution.engine.serviced.get("offload_ddr_d2h@r0:s0", 0.0) == 0.0; assert release_offload_access(case_execution, current_work) == 0
    events = schedule_kv_iteration_offload(
        case_execution, "bound", 0, current_work, e1, (call0, store_b),
        {"call0": e1}, state_accesses=(binding,), access_events={("storeB", "out"): e_b})
    assert events
    case_execution.engine.advance(until=events, first=False)
    assert min(event.time for event in events) >= 5.0
    result = dict(case_execution.engine.serviced)
    assert result["offload_ddr_d2h@r0:s0"] == pytest.approx(32.0); assert result["offload_ssd@r0:s0"] == pytest.approx(16.0); assert result["memory@r0:s0"] == pytest.approx(48.0); assert schedule_kv_iteration_offload(
        case_execution, "same-batch-repeat", 0, current_work, e1, (call0, store_b),
        {"storeB": e_b}, state_accesses=(binding,)) == ()
    assert release_offload_access(case_execution, current_work) == 1
    work2, access2 = work(), binding
    assert bind_offload_access_events(case_execution, work2, {access2.key: e_b}) == 1
    again = schedule_kv_iteration_offload(
        case_execution, "registered", 0, work2, e1, (call0, store_b),
        {"call0": e1}, state_accesses=(access2,))
    assert again
    case_execution.engine.advance(until=again, first=False)
    assert case_execution.engine.serviced["offload_ddr_d2h@r0:s0"] == pytest.approx(64.0); assert release_offload_access(case_execution, work2) == 1; assert not getattr(case_execution, "_offload_access_events", {})
def test_multiple_producers_padding_and_empty_accesses():
    case_spec = with_offload(parse_case(_raw_case())); case_execution = execution(case_spec); producer = case_execution.engine.submit("producer", latency=1.0)
    with pytest.raises(Exception, match="one_producer_event_per_call"):
        schedule_kv_layer_writeback(
            case_execution, "two", 0, 0, 0, work(), producer,
            state_accesses=(access(bytes=32, occurrence="target:a"),
                            access(bytes=32, occurrence="target:b")))
    padding = execution(case_spec); input_ready = padding.engine.submit("producer", latency=1.0)
    assert schedule_kv_layer_writeback(
        padding, "pad", 0, 0, 0, work(), input_ready, state_accesses=()) == ()
    assert padding.engine.serviced.get("offload_ddr_d2h@r0:s0", 0.0) == 0.0
    active = access(bytes=32, occurrence="active:cache_store", request="active"); active_execution = execution(case_spec)
    active_events = schedule_kv_layer_writeback(
        active_execution, "active", 0, 0, 0, work(),
        active_execution.engine.submit("producer", latency=1.0), state_accesses=(active,))
    assert serviced(active_execution, active_events)["offload_ddr_d2h@r0:s0"] == pytest.approx(16.0)
    physical = execution(case_spec); padding_view = RequestView("padding", 4, 0, False, "decode", padding=True)
    step = physical.add(linear_work(IterationPlan((padding_view,), iteration=0),
                                    case_spec.model, case_spec.runtime,
                                    case_spec.machine), 0, 0.0, emitted_tokens={})
    physical.engine.advance(until=(step.completion,), first=False)
    assert step.padded_tokens == 4 and step.operators == 1; assert physical.engine.serviced.get("offload_ddr_d2h@r0:s0", 0.0) == 0.0; assert physical.engine.serviced["matrix@r0:s0"] > 0
    empty_execution = execution(case_spec); input_ready = empty_execution.engine.submit("input", latency=1.0)
    assert schedule_kv_iteration_offload(
        empty_execution, "empty", 0, work(ops=(_index_tail_op(),)), input_ready, (), {}) == ()
    positive_execution = execution(case_spec); input_ready = positive_execution.engine.submit("input", latency=1.0)
    positive = work(ops=(SimpleNamespace(
        op_id="opaque:state_store", family="opaque", stage=0, layer=0, requests=("r",),
        mte3_bytes=32.0, hbm_bytes=64.0, metadata=(("state_write_bytes", 32.0),)),))
    producer = positive_execution.engine.submit("opaque:state_store", latency=1.0)
    events = schedule_kv_iteration_offload(
        positive_execution, "positive", 0, positive, input_ready, positive.ops,
        {"opaque:state_store": producer})
    assert serviced(positive_execution, events)["offload_ddr_d2h@r0:s0"] == pytest.approx(16.0)
    discovered = _accesses_from_ops(positive, positive.ops)
    assert len(discovered) == 1 and discovered[0].occurrence == "opaque:state_store"; assert discovered[0].bytes == 32.0
def test_target_and_draft_keep_separate_write_sets():
    target = access(bytes=256, occurrence="target:cache_store"); draft = access(bytes=192, occurrence="draft:cache_store", scope="draft"); projected = project_state_tiers((target, draft), TIER)
    assert sum(row["ddr_bytes"] for row in projected) == pytest.approx(224.0)
    assert sum(row["ssd_bytes"] for row in projected) == pytest.approx(112.0)
    assert sum(row["hbm_bytes"] for row in projected) == pytest.approx(336.0)
    case_execution = execution(with_offload(parse_case(_raw_case()))); target_producer = case_execution.engine.submit("target:store", release=1.0); draft_producer = case_execution.engine.submit("draft:store", release=2.0)
    target_events = schedule_kv_layer_writeback(
        case_execution, "i0", 0, 0, 0, work(), target_producer, state_accesses=(target,))
    draft_events = schedule_draft_kv_writeback(
        case_execution, "i0", 0, 0, work(), draft_producer, state_accesses=(draft,))
    assert all("draft_kv_writeback" not in event.name for event in target_events)
    assert all("draft_kv_writeback" in event.name for event in draft_events)
    result = serviced(case_execution, target_events + draft_events)
    assert result["offload_ddr_d2h@r0:s0"] == pytest.approx(224.0)
    assert result["offload_ssd@r0:s0"] == pytest.approx(112.0)
    assert result["memory@r0:s0"] == pytest.approx(336.0)
    assert target_events[0].time >= target_producer.time
    assert draft_events[0].time >= draft_producer.time and target.key != draft.key
    no_draft = execution(with_offload(parse_case(_raw_case()))); producer = no_draft.engine.submit("producer", latency=1.0)
    assert schedule_draft_kv_writeback(
        no_draft, "i1", 0, 0, work(), producer, state_accesses=()) == ()


def test_dead_work_bindings_are_reclaimed():
    case_execution = execution(with_offload(parse_case(_raw_case()))); transient = work(); event = case_execution.engine.submit("input", latency=1.0)
    schedule_kv_iteration_offload(case_execution, "transient", 0, transient,
                                  event, (), {})
    del transient; gc.collect(); replacement = work()
    schedule_kv_iteration_offload(case_execution, "replacement", 0, replacement,
                                  event, (), {})
    assert schedule_kv_iteration_offload(case_execution, "again", 0, replacement, event, (), {}) == ()
    assert all(reference() is not None for reference, _ in
               case_execution._offload_actual_accesses.values())

def test_real_draft_store_is_not_the_proposal_transfer():
    raw = _raw_case(output=2, pp=2)
    raw["runtime"]["metadata"] = {"draft_mode": "mtp", "draft_proposal_tokens": 2,
                                  "draft_accept_tokens": 1}
    base = parse_case(raw); auxiliary = draft_case().model.auxiliary
    draft = replace(auxiliary.mtp, layer_plans=native_draft_layer_plans(
        base.model, auxiliary.mtp))
    target = replace(base, model=replace(base.model, auxiliary=replace(auxiliary, mtp=draft)))
    view = RequestView("r", 1, 4, True, "decode", draft_proposal_tokens=2,
                       draft_accepted_tokens=1)
    from mindie_sim.prediction.frontend import Frontend
    frontend_work = Frontend()(IterationPlan((view,)), target.model, target.runtime, target.machine); accesses = _accesses_from_ops(frontend_work)
    draft_accesses = [item for item in accesses if item.scope == "draft"]
    assert draft_accesses and all(item.stage == 1 for item in draft_accesses); assert all(item.occurrence != "draft_proposal_transfer" for item in draft_accesses); assert any("cache" in item.occurrence for item in draft_accesses)
    case_spec = with_offload(target); case_execution = execution(case_spec)
    producer = case_execution.engine.submit("draft:last-store", latency=1.0)
    events = schedule_draft_kv_writeback(
        case_execution, "real", 0, 1, frontend_work, producer,
        state_accesses=tuple(draft_accesses),
        access_events={item.producer: producer for item in draft_accesses})
    assert events and all("draft_kv_writeback" in event.name for event in events); assert serviced(case_execution, events)["offload_ddr_d2h@r0:s1"] > 0
def _memfabric_op(name, recipe, direction, rows, entries, payload):
    return OpSpec(name, f"kv_offload_{direction}", "decode", 0, 0, ("r0",),
                  (rows, 576), (rows, 576), (rows, 576), (rows, 576), recipe=recipe,
                  dtype="bf16", source_boundary="kernel", hbm_bytes=payload,
                  mte2_bytes=payload, mte3_bytes=payload,
                  metadata=(("memfabric_call", {"direction": direction, "entries": entries,
                                                "payload_bytes": payload}),))
def test_memfabric_sparse_copy_d2h_and_h2d_scale_from_p20_primitives():
    case = memfabric_case()
    d2h_small = graph_costs(case, (_memfabric_op("d2h_s", "memfabric_sparse_copy_d2h", "d2h", 32, 64, float(32 * 1152)),), isolated=False)["duration_ms"]
    d2h_large = graph_costs(case, (_memfabric_op("d2h_l", "memfabric_sparse_copy_d2h", "d2h", 4096, 8192, float(4096 * 1152)),), isolated=False)["duration_ms"]
    h2d_small = graph_costs(case, (_memfabric_op("h2d_s", "memfabric_sparse_copy_h2d", "h2d", 32, 64, float(32 * 1152)),), isolated=False)["duration_ms"]
    h2d_large = graph_costs(case, (_memfabric_op("h2d_l", "memfabric_sparse_copy_h2d", "h2d", 4096, 8192, float(4096 * 1152)),), isolated=False)["duration_ms"]
    assert 0.0 < d2h_small < d2h_large and 0.0 < h2d_small < h2d_large
    # Explicit transfer_channels override scales DMA duration directly (zero hardcoded A5_GVA_CALIBRATION).
    fast_op = OpSpec("h2d_fast", "kv_offload_h2d", "decode", 0, 0, ("r0",), (4096, 576), (4096, 576), (4096, 576), (4096, 576),
                     recipe="memfabric_sparse_copy_h2d", dtype="bf16", source_boundary="kernel",
                     hbm_bytes=float(4096 * 1152), mte2_bytes=float(4096 * 1152), mte3_bytes=float(4096 * 1152),
                     metadata=(("memfabric_call", {"direction": "h2d", "entries": 8192, "payload_bytes": float(4096 * 1152),
                                                   "channel_bw_gbps": 200.0, "channel_alpha_us": 5.0}),))
    assert graph_costs(case, (fast_op,), isolated=False)["duration_ms"] < h2d_large
def test_memfabric_kv_exchange_uses_last_two_vector_owners():
    case = memfabric_case()
    durs = []
    for direction, pages in (("h2d", 1), ("h2d", 64), ("d2h", 1), ("d2h", 64)):
        payload = float(pages * 128 * 1152)
        op = OpSpec("kv_ex", "kv_offload_exchange", "prefill", 0, 0, ("r0",),
                    (pages, 128, 576), (pages, 128, 576), (pages, 128, 576),
                    (pages, 128, 576), recipe="memfabric_kv_exchange", dtype="bf16",
                    source_boundary="kernel", hbm_bytes=payload, mte2_bytes=payload,
                    mte3_bytes=payload,
                    metadata=(("memfabric_call", {"direction": direction, "pages": pages,
                                                  "payload_bytes": payload}),))
        result = graph_costs(case, (op,), isolated=True, trace_limit=64)
        durs.append(result["duration_ms"])
        execution_meta = dict(result["isolated"]["operators"][0]["operator"]["metadata"])[
            "program_execution"]
        assert tuple(execution_meta["active_aiv_owners"]) == (62, 63)
    assert durs[0] < durs[1] and durs[2] < durs[3]
def test_memfabric_entry_gather_scales_with_rows_and_wide_ub():
    case = memfabric_case()
    durs = []
    for rows in (64, 4096):
        payload = float(rows * 1024)
        op = OpSpec("eg", "kv_offload_gather", "decode", 0, 0, ("r0",),
                    (rows, 512), (rows, 512), (rows, 512), (rows, 512),
                    recipe="memfabric_entry_gather", dtype="bf16", source_boundary="kernel",
                    hbm_bytes=payload, mte2_bytes=payload, mte3_bytes=payload,
                    metadata=(("memfabric_call", {"rows": rows, "payload_bytes": payload}),))
        result = graph_costs(case, (op,), isolated=True)
        durs.append(result["duration_ms"])
        execution_meta = dict(result["isolated"]["operators"][0]["operator"]["metadata"])[
            "program_execution"]
        assert execution_meta["ub_bytes_per_owner"] == 122880
    assert 0.0 < durs[0] < durs[1]
def test_sparse_mla_offload_decode_and_host_dram_budget():
    from mindie_sim.prediction.implementations import select
    from mindie_sim.schema.common import ModelingError

    assert copy_sfa_pool_capacity(16) == 18; assert copy_sfa_pool_capacity(17) == 19
    with pytest.raises(ModelingError, match="implementation_not_registered:sparse_kv_offload:hisparse_kv_offload"):
        select({"sparse_kv_offload": "hisparse_kv_offload"}, "sparse_kv_offload")
    short_raw = sparse_mla_case(context_tokens=1024, offloading={
        "implementation": "vllm_ascend_sparse_kv_offload", "options": {"miss_rate": 0.2}})
    short = predict(short_raw, scope="body", phase="decode", tokens=1,
                    context=1024, sequences=[1])
    short_recipes = [op["recipe"] for op in short["timeline"]]
    assert short_recipes.count("memfabric_sparse_copy_d2h") == 2; assert "memfabric_sparse_copy_h2d" not in short_recipes; assert "fused_lightning_indexer_manage" not in short_recipes
    long_raw = sparse_mla_case(context_tokens=8192, offloading={
        "implementation": "vllm_ascend_sparse_kv_offload", "options": {"miss_rate": 0.2}})
    long = predict(long_raw, scope="body", phase="decode", tokens=1,
                   context=8192, sequences=[1])
    recipes = [op["recipe"] for op in long["timeline"]]
    assert recipes.count("memfabric_sparse_copy_d2h") == 2; assert recipes.count("fused_lightning_indexer_manage") == 1; assert recipes.count("memfabric_sparse_copy_h2d") == 2
    long_ops = target_graphs(parse_case(long_raw), phase="decode", tokens=1,
                             context=8192, sequences=[1], scope="body")[0]
    h2d_ops = [op for op in long_ops if op.recipe == "memfabric_sparse_copy_h2d"]; layer0 = dict(h2d_ops[0].metadata)
    layer1 = dict(h2d_ops[1].metadata)
    assert layer0["memfabric_call"]["tokens"] == 2048; assert layer0["shared_indexer_reuse"] is False and layer1["shared_indexer_reuse"] is True; assert layer1["skip_topk"] is True; assert layer0["hardware_support"]["memfabric_acc_offload"] == "host_gva_dma_modeled"; assert layer0["hardware_support"]["abi_verification"] == "verified_via_probed_abi_registry"; assert layer0["hardware_support"]["copy_sfa_attention"] == "analytical_model"
    base_raw = sparse_mla_case(context_tokens=4096); offload_raw = sparse_mla_case(context_tokens=4096, offloading={
        "implementation": "vllm_ascend_sparse_kv_offload",
        "options": {"dram_size_per_dp_gb": 0.02, "keep_device_kv_cache": False}})
    base_memory, offload_memory = Memory(parse_case(base_raw)), Memory(parse_case(offload_raw))
    assert sum(offload_memory.cache_bytes(4096)) == 4096 * 130; assert sum(base_memory.cache_bytes(4096)) == 4096 * (2 * 1152 + 130); assert offload_memory.host_dram_cache_bytes(4096) == 4096 * 2 * 1152; assert offload_memory.resize(0, "req_fits", 2048) is True; assert offload_memory.resize(0, "req_overflows_dram", 4096) is False
    summary = offload_memory.to_dict()["sparse_kv_offload"]
    assert summary["limiting_factor"] == "host_dram"; assert summary["peak_host_dram_bytes_per_replica"][0] == 2048 * 2 * 1152
def test_prefill_layerwise_and_pd_remote_handoff():
    prefill_raw = sparse_mla_case(context_tokens=1024, offloading={
        "implementation": "vllm_ascend_sparse_kv_offload",
        "options": {"layerwise_kv_offload": True, "num_shared_layers": 1}})
    prefill = predict(prefill_raw, scope="body", phase="prefill", tokens=1024,
                      context=0, sequences=[1024])
    kv_exchange = [op for op in prefill["timeline"] if op["recipe"] == "memfabric_kv_exchange"]
    assert len(kv_exchange) == 2 and all(op["stream"] == "offload" for op in kv_exchange)
    base_pd = sparse_mla_case(context_tokens=1024, tp=4)
    base_pd["pools"] = {"prefill": {"machine": deepcopy(base_pd["machine"]),
                                    "runtime": deepcopy(base_pd["runtime"])},
                        "decode": {"machine": deepcopy(base_pd["machine"]),
                                   "runtime": deepcopy(base_pd["runtime"])}}
    base_pd["connector"] = {"bytes_per_s": 50e9, "latency_us": 5.0}; offload_pd = deepcopy(base_pd)
    offload_pd["pools"]["decode"]["runtime"]["kernels"] = {"lowerings": {
        "sparse_kv_offload": {"implementation": "vllm_ascend_sparse_kv_offload",
                              "options": {"use_fused_overlap": True,
                                          "dram_size_per_dp_gb": 64.0}}}}
    base_report, offload_report = predict(base_pd), predict(offload_pd)
    assert base_report["status"] == offload_report["status"] == "modeled"; assert base_report["execution"]["handoff_bytes"] == 4 * 1024 * (2 * 1152 + 130); assert offload_report["execution"]["handoff_bytes"] == (
        4 * 1024 * 130 + 1 * 1024 * (2 * 1152))
    assert offload_report["execution"]["handoff_bytes"] < base_report["execution"]["handoff_bytes"]


def test_compose_accepts_full_window_speculative_requests_and_validates_draft_cap():
    import json
    from pathlib import Path
    from _support.axes import REPRESENTATIVE_SKUS
    from mindie_sim.compose import _resolve_mtp_policy
    from mindie_sim.model_catalog import load_model
    from mindie_sim.schema.common import ModelingError

    cfg_path = Path(__file__).resolve().parents[2] / "mindie_sim" / "model_configs" / "deepseek-v3.2.json"
    base_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    short_cfg = {**base_cfg, "num_hidden_layers": 4, "max_position_embeddings": 4096}
    short_target = load_model("deepseek-v3.2")

    # 4080 + 16 == 4096 fits the 4096 target horizon even with explicit mtp_proposal_tokens=3.
    composed = compose_case(
        short_cfg, hardware=REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, ep=1,
        prompt_tokens=4080, output_tokens=16, max_model_len=4096,
        mtp_proposal_tokens=3, mtp_acceptance_length=2.5,
    )
    assert composed["runtime"]["max_model_len"] == 4096
    assert composed["runtime"]["metadata"]["draft_proposal_tokens"] == 3

    # Inferred MTP stays active at exact 4096 boundary rather than disabling with max_len_bounded.
    prof_tokens, _accept, source = _resolve_mtp_policy(
        mtp_proposal_tokens=None, mtp_acceptance_length=None,
        raw_eager=False, production=True,
        va_profile={"mtp_proposal_tokens": 3},
        model_spec=short_target, prompt_tokens=4080, output_tokens=16,
        ttft_slo_ms=None, tpot_slo_ms=None, require_scenario=True,
        model_limit=4096, max_model_len=4096,
    )
    assert (prof_tokens, source) == (3, "profile")

    # Exceeding the actual target horizon (4081 + 16 = 4097 > 4096) still rejects.
    with pytest.raises(ModelingError, match="workload_exceeds_model_max_position_embeddings"):
        compose_case(
            short_cfg, hardware=REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, ep=1,
            prompt_tokens=4081, output_tokens=16, max_model_len=4096,
            mtp_proposal_tokens=3, mtp_acceptance_length=2.5,
        )

    # Independent draft-model context cap is checked against draft.max_position_embeddings separately.
    kimi_path = Path(__file__).resolve().parents[2] / "mindie_sim" / "model_configs" / "kimi-k3.json"
    kimi_cfg = json.loads(kimi_path.read_text(encoding="utf-8"))
    capped_cfg = deepcopy(kimi_cfg)
    capped_cfg["text_config"]["dspark_max_model_len"] = 4096
    ok_capped = compose_case(
        capped_cfg, hardware=REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, ep=1, layers=4,
        prompt_tokens=8192, output_tokens=16, mtp_proposal_tokens=3,
        mtp_acceptance_length=2.5, draft_max_model_len=4096,
    )
    assert ok_capped["runtime"]["metadata"]["draft_max_model_len"] == 4096
    with pytest.raises(ModelingError, match="draft_max_model_len_exceeds_draft_max_position_embeddings"):
        compose_case(
            capped_cfg, hardware=REPRESENTATIVE_SKUS["a5"], tp=1, dp=1, ep=1, layers=4,
            prompt_tokens=8192, output_tokens=16, mtp_proposal_tokens=3,
            mtp_acceptance_length=2.5, draft_max_model_len=8192,
        )


def test_derived_sku_host_dma_inherits_base_channel_and_calibration():
    from mindie_sim.backends.ascend import DERIVED_ASCEND_SKUS
    from mindie_sim.compose import _COMPOSE_OPTION_GROUPS, _materialize_machine
    from mindie_sim.prediction.offload import resolve_sku_host_dma_rates

    assert DERIVED_ASCEND_SKUS
    opts = dict.fromkeys(_COMPOSE_OPTION_GROUPS["machine_options"])
    for derived, spec in DERIVED_ASCEND_SKUS.items():
        own = resolve_sku_host_dma_rates(_materialize_machine(derived, required_devices=1, **opts)[2])
        base = resolve_sku_host_dma_rates(_materialize_machine(str(spec["base"]), required_devices=1, **opts)[2])
        assert own["sku"] == derived and own["h2d_source"] != "explicit_transfer_channel"
        for key in ("h2d_bps", "d2h_bps", "host_memory_bps", "h2d_source", "d2h_source", "h2d_latency_us"):
            assert own[key] == base[key], (derived, key)
