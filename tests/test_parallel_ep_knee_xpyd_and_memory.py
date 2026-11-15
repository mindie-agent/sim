"""Tier-0 behavioral tests for fabric, memory, MoE skew and parallel planning.

[Role] Exercise the next-release Pillar-4 modules as black boxes: the two-mode
  fabric, the six-component HBM ledger, EPLB/MTP physics, the EP knee,
  dual-phase context parallelism, xPyD pool sizing and SKU what-if scaling.
[Invariants] Every assertion is a physical inequality or conservation identity,
  never a string-grep or a hard-coded golden latency.
[Timeouts & Locks] No subprocess, network, sleep or device access.
[Side-Effect Policy] No workspace writes; fixtures build immutable Cases.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.fabric_topology import (CalibratedFabricProfile, FabricTier,
                                       HierarchicalFabricTree, project_case,
                                       scale_hardware)
from mindie_sim.memory_breakdown import COMPONENT_NAMES, analyze_memory_breakdown
from mindie_sim.moe_skew_eplb import (max_rank_tokens, mtp_batch_reversal,
                                      replicated_expert_bytes)
from mindie_sim.parallel_explorer import (DecodeCostModel, kv_transfer_contention,
                                          plan_context_parallel, solve_xpyd_pools,
                                          sweep_ep_knee)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.parallel import ParallelSpec


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    for name, sub in (("HOME", "home"), ("MINDIE_CONFIG_HOME", "config"),
                      ("MINDIE_DATA_HOME", "data"), ("MINDIE_CACHE_HOME", "cache"),
                      ("MINDIE_DIAGNOSTICS_ROOT", "diagnostics")):
        monkeypatch.setenv(name, str(root / sub))


def _case(*, sku="Ascend910_9362", topology="a3_16p", num_devices=64,
          ranks_per_node=16, layers=8, hidden=4096, heads=32, kv_heads=8,
          experts=32, top_k=8, moe_inter=1024, tp=1, dp=2, ep=2,
          max_seqs=256, max_model_len=32768, graph="none", inter_node=None):
    machine = {"backend": "ascend", "sku": sku, "num_devices": num_devices,
               "ranks_per_node": ranks_per_node, "topology_preset": topology}
    if inter_node is not None:
        machine.update(inter_node_bw_gbps=inter_node, inter_node_bw_util=1.0,
                       inter_node_latency_us=10.0)
    return parse_case({
        "machine": machine,
        "model": {"name": "tier0-parallel", "layers": layers, "hidden_size": hidden,
                  "num_heads": heads, "num_kv_heads": kv_heads, "head_dim": 128,
                  "intermediate_size": hidden * 2, "vocab_size": 32000,
                  "attention": "gqa", "ffn": "moe" if experts else "dense",
                  "num_experts": experts, "top_k": top_k,
                  "moe_intermediate_size": moe_inter,
                  "activation_dtype": "bf16", "weight_dtype": "bf16",
                  "kv_dtype": "bf16"},
        "runtime": {"tp": tp, "dp": dp, "ep": ep, "pp": 1,
                    "max_model_len": max_model_len, "max_num_seqs": max_seqs,
                    "max_num_batched_tokens": 8192, "gpu_memory_utilization": 0.9,
                    "kv_block_size": 128, "graph_capture_mode": graph},
        "workload": {"requests": [{"request_id": "r", "prompt_tokens": 1024,
                                   "output_tokens": 16, "arrival_ms": 0.0}]},
    })


@pytest.fixture(scope="module")
def moe_case():
    return _case()


@pytest.fixture(scope="module")
def a5_case():
    return _case(sku="Ascend950DT_9582", topology="a5_pod", num_devices=64,
                 ranks_per_node=64, experts=128, max_seqs=256)


@pytest.fixture(scope="module")
def cp_case():
    return _case(layers=4, hidden=2048, heads=16, kv_heads=4, experts=0,
                 top_k=0, moe_inter=0, tp=8, dp=1, ep=1, num_devices=16,
                 max_model_len=8192)


def _tree(intra_domain: int, pod_bw: float, *, oversub: float = 1.0) -> HierarchicalFabricTree:
    return HierarchicalFabricTree((
        FabricTier("intra_node_hccs", intra_domain, 1.5, 392.0),
        FabricTier("pod_roce", 64, 10.0, pod_bw, oversubscription=oversub),
    ))


def _tree_hccs() -> HierarchicalFabricTree:
    return HierarchicalFabricTree((
        FabricTier("intra_node_hccs", 8, 1.5, 392.0),
        FabricTier("superpod_hccs", 64, 3.5, 800.0),
    ))


def _knee(case, *, skew=1.0, fabric=None, mc2=False, eplb=False, redundant=0,
          batch=128, context=1024):
    return sweep_ep_knee(case, ep_candidates=(2, 4, 8, 16, 32), batch_sequences=batch,
                         context_tokens=context, routing_skew=skew, fabric=fabric,
                         mc2=mc2, hccl_buffsize_mb=64, enable_eplb=eplb,
                         eplb_redundant_experts=redundant)


def test_ep_knee_shifts_lower_with_routing_skew_and_slow_fabric(moe_case):
    fast = _tree_hccs()
    low = _knee(moe_case, skew=1.2, fabric=fast)
    high = _knee(moe_case, skew=1.5, fabric=fast)
    assert high["ep_star_latency"] <= low["ep_star_latency"]
    slow = _tree(8, 12.5, oversub=4.0)
    slow_knee = _knee(moe_case, skew=1.2, fabric=slow)
    assert slow_knee["ep_star_latency"] <= low["ep_star_latency"]
    for row in low["rows"]:
        if row["feasible"]:
            total = (row["t_attn_ms"] + row["t_expert_compute_ms"]
                     + row["t_expert_straggler_ms"] + row["t_dispatch_combine_ms"]
                     + row["t_non_moe_ms"])
            assert total <= row["step_ms"] + 1e-9
    assert max_rank_tokens(128, 8, 32, 1.5, num_experts=32) >= max_rank_tokens(
        128, 8, 32, 1.0, num_experts=32)


def test_ep_knee_eplb_and_mc2_reduce_exposed_cost(moe_case):
    fast = _tree_hccs()
    plain = _knee(moe_case, skew=1.5, fabric=fast)
    eplb = _knee(moe_case, skew=1.5, fabric=fast, eplb=True, redundant=2)
    plain_row = next(row for row in plain["rows"] if row["ep"] == 16)
    eplb_row = next(row for row in eplb["rows"] if row["ep"] == 16)
    assert eplb_row["t_expert_straggler_ms"] <= plain_row["t_expert_straggler_ms"]
    assert eplb_row["step_ms"] <= plain_row["step_ms"]
    slow = _tree(8, 12.5, oversub=4.0)
    no_mc2 = _knee(moe_case, skew=1.0, fabric=slow, mc2=False)
    with_mc2 = _knee(moe_case, skew=1.0, fabric=slow, mc2=True)
    a = next(row for row in no_mc2["rows"] if row["ep"] == 8)
    b = next(row for row in with_mc2["rows"] if row["ep"] == 8)
    assert b["raw_comm_ms"] <= a["raw_comm_ms"] or b["step_ms"] <= a["step_ms"]
    for report in (plain, eplb, no_mc2, with_mc2):
        assert report["ep_star"] is None or all(
            row["feasible"] for row in report["rows"] if row["ep"] == report["ep_star"])


def test_memory_oracle_catches_large_ep_and_graph_assassins():
    ep_case = _case(sku="Ascend910B4", topology="a2_standard", num_devices=64,
                    ranks_per_node=8, layers=8, hidden=2048, heads=16, kv_heads=4,
                    experts=128, top_k=4, moe_inter=512, tp=4, dp=16, ep=64,
                    max_seqs=64, max_model_len=8192, graph="full")
    full = analyze_memory_breakdown(
        ep_case, sequences=8, sequence_tokens=4096, hccl_buffsize_mb=256,
        graph_buckets=(1024, 4096, 8192, 16384),
        graph_activation_bytes_per_token=1 << 20)
    assert tuple(component.name for component in full.components) == COMPONENT_NAMES
    assert sum(component.bytes for component in full.components) == pytest.approx(
        full.peak_hbm_bytes, rel=1e-12)
    assert not full.oom_feasible
    assert full.naive_weights_kv_bytes <= full.budget_bytes
    assert full.peak_hbm_bytes > full.budget_bytes
    assert full.dominant_assassin in {"collective_static_buffers_bytes",
                                      "graph_capture_pool_bytes"}
    small = analyze_memory_breakdown(ep_case, sequences=8, sequence_tokens=4096,
                                     hccl_buffsize_mb=64)
    big = analyze_memory_breakdown(ep_case, sequences=8, sequence_tokens=4096,
                                   hccl_buffsize_mb=256)

    def comm(record):
        return dict((item.name, item.bytes) for item in record.components)[
            "collective_static_buffers_bytes"]
    assert comm(big) > comm(small) > 0.0
    graph_case = _case(sku="Ascend910B4", topology="a2_standard", num_devices=8,
                       ranks_per_node=8, layers=4, hidden=1024, heads=8, kv_heads=2,
                       experts=0, top_k=0, moe_inter=0, tp=4, dp=2, ep=1,
                       max_seqs=32, max_model_len=4096, graph="full")
    graph = analyze_memory_breakdown(
        graph_case, sequences=1, sequence_tokens=2048, hccl_buffsize_mb=8,
        graph_buckets=(1024, 4096, 8192, 16384),
        graph_activation_bytes_per_token=1 << 20)
    assert graph.naive_weights_kv_bytes < graph.budget_bytes
    assert not graph.oom_feasible
    assert graph.dominant_assassin == "graph_capture_pool_bytes"
    aux = analyze_memory_breakdown(ep_case, sequences=8, sequence_tokens=4096,
                                   hccl_buffsize_mb=64, enable_eplb=True,
                                   eplb_redundant_experts=4)
    aux_bytes = dict((item.name, item.bytes) for item in aux.components)["aux_weights_bytes"]
    assert aux_bytes > 0.0
    assert replicated_expert_bytes(ep_case, 4)["bytes_per_rank"] > 0.0


def test_mtp_batch_reversal_on_moe_decode():
    mtp_case = _case(dp=8, ep=8)
    model = DecodeCostModel.from_case(mtp_case, context_tokens=1024)
    closure = lambda batch: model.step_ms(batch, context=1024)
    result = mtp_batch_reversal(closure, draft_len=3, accept_rate=0.75, max_batch=512)
    assert result.b_reversal is not None
    assert result.speedup_at_one > 1.0
    assert 1 in {point[0] for point in result.curve}
    assert result.crossings and result.b_reversal == result.crossings[0]
    baseline = mtp_batch_reversal(closure, draft_len=0, max_batch=8)
    assert baseline.b_reversal is None and baseline.accepted_tokens_per_step == 1.0


def test_xpyd_joint_sizing_and_d_node_jitter(a5_case):
    common = dict(
        total_npus=64, isl=8192, osl=1024, ttft_slo_ms=3000.0,
        tpot_slo_ms=50.0, hccl_buffsize_mb=64, max_pool_instances=6,
        prefill_configs=({"tp": 2, "dp": 1, "ep": 1},),
        decode_configs=({"tp": 1, "dp": 2, "ep": 2},),
    )
    cold = solve_xpyd_pools(a5_case, prefix_hit_rate=0.0,
                            allow_unqualified=True, **common)
    warm = solve_xpyd_pools(a5_case, prefix_hit_rate=0.75,
                            allow_unqualified=True, **common)
    for report in (cold, warm):
        best = report["best_configuration"]
        assert best is not None and best["used_npus"] <= 64
        assert report["qualified"] is False and not best["route_qualified"]
        assert best["goodput_output_tps"] > 0.0
        assert best["goodput_rps"] * 1024 == pytest.approx(best["goodput_output_tps"])
        assert best["utilization"] < 1.0
        assert best["ttft_ms"] <= 3000.0 and best["tpot_ms"] <= 50.0
        assert best["d_hbm_jitter"] > 0.0
    assert warm["best_configuration"]["goodput_output_tps"] >= cold[
        "best_configuration"]["goodput_output_tps"]
    assert warm["best_configuration"]["n_prefill"] < cold["best_configuration"]["n_prefill"]
    handoff = kv_transfer_contention(
        512 * 1024 * 1024, n_prefill=28, n_decode=4, nic_bw_gbps=50.0,
        hbm_bw_bytes_per_s=4e12, p_world=1, d_world=2,
        decode_step_ms=10.0, osl=1024, requests_per_s=40.0)
    assert handoff["transfer_ms"] > 0.0 and handoff["jitter_ratio"] > 0.0
    assert handoff["tpot_jitter_ms"] > 10.0


def test_hardware_what_if_scaling_shifts_regimes_monotonically():
    case = _case(sku="Ascend910_9362", topology="a3_16p", num_devices=64,
                 ranks_per_node=16, layers=8, hidden=2048, heads=16, kv_heads=4,
                 experts=64, top_k=4, moe_inter=512, tp=1, dp=2, ep=2,
                 max_seqs=512, max_model_len=16384, inter_node=6.25)

    def step_at(target):
        rows = sweep_ep_knee(target, ep_candidates=(8, 16, 32), batch_sequences=256,
                             context_tokens=256, hccl_buffsize_mb=64,
                             mc2=True)["rows"]
        row = next(item for item in rows if item["ep"] == 32)
        return row["step_ms"], row["bottleneck"], row["t_dispatch_combine_ms"]

    base_step, base_regime, _ = step_at(case)
    hbm = step_at(scale_hardware(case, hbm_bw=2.0))
    cube = step_at(scale_hardware(case, cube_tflops=2.0))
    link = step_at(scale_hardware(case, link_bw=2.0))
    assert hbm[0] <= base_step and cube[0] <= base_step and link[0] <= base_step
    assert base_step - link[0] >= 0.0 and base_step - cube[0] >= 0.0
    scale = scale_hardware(case.machine, hbm_bw=2.0, cube_tflops=2.0,
                           vector_ops=2.0, link_bw=2.0, hbm_bytes=1.5,
                           l1_bytes=2.0, ub_bytes=2.0)
    old = case.machine.dies[0]
    new = scale.dies[0]
    assert new.memory_bytes == int(old.memory_bytes * 1.5)
    assert dict(new.rates)["hbm"] == pytest.approx(dict(old.rates)["hbm"] * 2.0)
    assert dict(new.buffers)["l1"] == dict(old.buffers)["l1"] * 2
    assert dict(new.buffers)["ub"] == dict(old.buffers)["ub"] * 2
    launched = replace(case.machine, dies=tuple(replace(die, launch_us=2.0)
                                                for die in case.machine.dies))
    assert scale_hardware(launched, launch_us=2.0).dies[0].launch_us == 4.0
    assert scale_hardware(case.machine) is case.machine
    for sku, expected in (("a2", 32.0), ("a3", 64.0), ("a5", 96.0)):
        projected = project_case(case, sku)
        assert projected.machine.dies[0].memory_bytes == int(expected * 2**30)


def test_dual_phase_pcp_dcp_and_mrv2_plan(cp_case):
    parallel = ParallelSpec(dp=1, tp=8, ep=1, pp=1, cp=2, pcp=2, cp_plan="pcp")
    plan = plan_context_parallel(cp_case.model, parallel, mrv2_pcp=False)
    assert plan.compatible
    assert plan.prefill_plan == "pcp" and plan.decode_plan == "dcp"
    assert "pcp_kv_gather" in plan.prefill_collectives
    assert "dcp_lse_merge" in plan.decode_collectives
    mrv2 = plan_context_parallel(cp_case.model, parallel, mrv2_pcp=True,
                                 attention_families=("dsa", "swa", "csa2"))
    assert not mrv2.compatible and "attention_family_mismatch" in (mrv2.reason or "")
    dsa_parallel = ParallelSpec(dp=1, tp=8, ep=1, pp=1, cp=2, pcp=1, cp_plan="dcp")
    broken = plan_context_parallel(cp_case.model, dsa_parallel, prefill_plan="dsa_cp")
    assert not broken.compatible
    assert "dsa_cp" in (broken.reason or "")


def test_fabric_two_modes_and_oversubscription(moe_case):
    one_to_one = _tree(16, 50.0, oversub=1.0)
    four_to_one = _tree(16, 50.0, oversub=4.0)
    payload = 8 * 1024 * 1024
    slow = four_to_one.collective("all_to_all_v", payload, 32)
    fast = one_to_one.collective("all_to_all_v", payload, 32)
    assert slow.total_us > fast.total_us
    assert slow.bottleneck_tier == "pod_roce"
    small = one_to_one.collective("all_to_all_v", payload // 8, 32)
    large = one_to_one.collective("all_to_all_v", payload, 32)
    assert large.total_us > small.total_us
    assert one_to_one.collective("all_to_all_v", payload, 2).total_us <= one_to_one.collective(
        "all_to_all_v", payload, 32).total_us
    profile = CalibratedFabricProfile.from_probe({"source": "unit-probe", "tiers": [{
        "name": "intra_node_hccs", "domain_ranks": 16, "points": [
            {"payload_bytes": 4096, "alpha_us": 3.0, "beta_ns_per_byte": 0.02,
             "congestion_factor": 1.0},
            {"payload_bytes": 1 << 20, "alpha_us": 5.0, "beta_ns_per_byte": 0.03,
             "congestion_factor": 1.2}]}]})
    mid = profile.collective("all_reduce", 1 << 16, 8)
    assert mid.method == "calibrated_piecewise_log2"
    assert 0.0 < mid.total_us < profile.collective("all_reduce", 1 << 20, 8).total_us
    hardware_tree = HierarchicalFabricTree.from_hardware(moe_case.machine)
    assert hardware_tree.dual_die_affinity
    assert hardware_tree.collective("all_reduce", 1 << 20, 32).total_us > 0.0
