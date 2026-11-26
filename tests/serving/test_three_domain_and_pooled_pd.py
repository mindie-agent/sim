"""Three-domain parameter law, SKU host-DMA binding, 3-tier pooled PD, and auto-optimal solver.

[Role] Verify Domain-1 fail-fast on omitted scenario/SLA/SSD fields, Domain-2
  default-ON model-intrinsic offload and production stack features (opt-out
  only) with SKU-bound PCIe/DDR bandwidths, shard-aware 3-tier pooled PD
  handoff conservation, and Domain-3 joint PD-vs-Colocated auto-optimal search.
[Invariants] Zero guessed prompt/output/TTFT/TPOT/SSD constants; zero flat
  32/64/100/200 GB/s fallback bandwidths; explicit opt-out and pinned parallel
  axes are always honored.
[Timeouts & Locks] Pure CPU tier-0 unit tests; no network, NPU, or file locks.
[Side-Effect Policy] Read-only over repository data; CLI tests write only to
  pytest ``tmp_path``.
"""
from __future__ import annotations

import json
import sys
import pytest

from mindie_sim import api as sdk_api
from mindie_sim.cli import _main as cli_main
from mindie_sim.compose import compose_case
from mindie_sim.compose.runtime import apply_va_recommended
from mindie_sim.prediction.disaggregated import compute_pd_handoff_plan
from mindie_sim.prediction.offload import (
    OffloadPolicy,
    resolve_case_offload_policy,
    resolve_sku_host_dma_rates,
)
from mindie_sim.feature_mechanisms import resolve_runtime_stack
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.sweep import (
    solve_optimal_deployment,
    sweep_pd_disaggregation_ratios,
    sweep_topologies,
)

pytestmark = pytest.mark.tier0


def test_domain_one_fails_fast_on_omitted_scenario_sla_and_ssd(tmp_path):
    with pytest.raises(ModelingError, match="missing_required_scenario_fields:prompt_tokens,output_tokens"):
        compose_case("qwen3-32b", "a3", tp=4, layers=2, require_scenario=True)
    with pytest.raises(ModelingError, match="missing_required_scenario_fields:output_tokens"):
        compose_case("qwen3-32b", "a3", tp=4, layers=2, prompt_tokens=1024)
    with pytest.raises(ModelingError, match="missing_required_scenario_fields:prompt_tokens"):
        compose_case("qwen3-32b", "a3", tp=4, layers=2, output_tokens=64)
    with pytest.raises(ModelingError, match="missing_required_scenario_fields:num_devices,prompt_tokens,output_tokens,tpot_slo_ms"):
        compose_case("qwen3-32b", "a3", parallel="auto")

    probe = compose_case("qwen3-32b", "a3", tp=2, layers=2)
    assert probe["runtime"]["metadata"]["scenario_declared"] is False
    assert probe["slo"] == {}
    with pytest.raises(ModelingError, match="missing_required_scenario_fields:prompt_tokens,output_tokens"):
        sdk_api.evaluate_service(probe)
    with pytest.raises(ModelingError, match="missing_required_scenario_fields:sla_ms,context,output_budget"):
        sweep_topologies(probe, num_devices=2)

    p = compose_case("qwen3-32b", "a3", tp=2, layers=2, prompt_tokens=128, output_tokens=32)
    with pytest.raises(ModelingError, match="missing_required_scenario_fields:prompt_tokens,output_tokens,ttft_sla_ms,tpot_sla_ms"):
        sweep_pd_disaggregation_ratios(p, p, total_dies=8)
    with pytest.raises(ModelingError, match="missing_required_scenario_fields:ssd_bw_gbps"):
        sweep_pd_disaggregation_ratios(
            p, p, total_dies=8, prompt_tokens=128, output_tokens=32,
            ttft_sla_ms=2000.0, tpot_sla_ms=50.0,
            pd_transfer_mode="pooled_dram_ssd", pool_ssd_ratio=0.5,
        )
    with pytest.raises(ModelingError, match="missing_required_scenario_fields:ssd_bw_gbps"):
        OffloadPolicy(kvc_ssd_ratio=0.25)

    out_path = tmp_path / "composed.json"
    assert cli_main(["compose", "--model", "qwen3-32b", "--hardware", "a3", "--tp", "2", "--layers", "2", "-o", str(out_path)]) == 2
    assert cli_main(["compose", "--model", "qwen3-32b", "--hardware", "a3", "--tp", "2", "--layers", "2", "--structure-only", "-o", str(out_path)]) == 0
    assert json.loads(out_path.read_text(encoding="utf-8"))["runtime"]["metadata"]["scenario_declared"] is False


def test_domain_two_defaults_intrinsic_offload_stack_features_and_opt_out():
    stack = resolve_runtime_stack("vllm-ascend")
    assert stack.name == "vllm-ascend"
    for model_key in ("deepseek-v4.1-flash", "qwen3.8-flash-next"):
        case = compose_case(
            model_key, "a5", tp=4, ep=4, layers=2, prompt_tokens=1024, output_tokens=64,
            exclude_optimizations=("cann_megamoe_fusion",),
        )
        rt = case["runtime"]
        meta = rt["metadata"]
        policy = meta["execution_policy"]
        assert rt["engram_cpu_offload"] is True
        assert rt["multistream_overlap_shared_expert"] is True
        assert rt["graph_capture_mode"] == "full"
        assert policy["engram_cpu_offload"] is True
        assert policy["engram_cpu_offload_source"] == "profile"
        assert policy["multistream_overlap_shared_expert"] is True
        assert policy["mtp_proposal_tokens"] == 0
        assert policy["mtp_source"] == "profile_none"

    flash_a3 = compose_case(
        "qwen3.8-flash-next", "a3", tp=4, ep=4, layers=2, prompt_tokens=1024, output_tokens=64,
        enable_shared_expert_dp=True,
    )
    assert flash_a3["runtime"]["metadata"]["enable_shared_expert_dp"] is True
    assert flash_a3["runtime"]["engram_cpu_offload"] is True
    assert flash_a3["runtime"]["multistream_overlap_shared_expert"] is True
    assert "sequence_parallel" in flash_a3["runtime"]["kernels"]

    opted_out = compose_case(
        "qwen3.8-flash-next", "a3", tp=4, ep=4, layers=2,
        prompt_tokens=1024, output_tokens=64,
        engram_cpu_offload=False,
        multistream_overlap_shared_expert=False,
        enable_shared_expert_dp=False,
        mtp_proposal_tokens=0,
        sp=False,
    )
    opt_rt = opted_out["runtime"]
    assert not opt_rt.get("engram_cpu_offload", False)
    assert not opt_rt.get("multistream_overlap_shared_expert", False)
    assert not opt_rt["metadata"].get("enable_shared_expert_dp", False)
    assert opt_rt["metadata"]["execution_policy"]["mtp_proposal_tokens"] == 0
    assert opt_rt["metadata"]["execution_policy"]["mtp_source"] == "explicit"

    eager = compose_case(
        "qwen3.8-flash-next", "a3", tp=4, ep=4, layers=2,
        prompt_tokens=1024, output_tokens=64, raw_eager=True,
    )
    assert not eager["runtime"].get("engram_cpu_offload", False)
    assert not eager["runtime"].get("multistream_overlap_shared_expert", False)
    assert eager["runtime"]["graph_capture_mode"] == "none"

    excluded = compose_case(
        "deepseek-v4.1-flash", "a3", tp=4, ep=4, layers=2,
        exclude_optimizations=("shared_expert_multistream_overlap",),
    )
    assert not excluded["runtime"].get("multistream_overlap_shared_expert", False)

    reapplied = apply_va_recommended(compose_case(
        "qwen3.8-flash-next", "a3", tp=4, ep=4, layers=2, va_recommended=False,
    ))
    assert reapplied["runtime"]["engram_cpu_offload"] is True
    assert reapplied["runtime"]["multistream_overlap_shared_expert"] is True

    # Guardrails: EP=1 does not auto-enable MSO; 1-layer dense-prefix slices skip unattachable auto-MTP.
    ep1_case = compose_case("qwen3.5-35b-a3b", "a3", tp=1, ep=1, layers=2, prompt_tokens=64, output_tokens=8)
    assert not ep1_case["runtime"].get("multistream_overlap_shared_expert", False)
    sliced_v32 = compose_case("deepseek-v3.2", "a3", tp=4, ep=1, layers=1, prompt_tokens=64, output_tokens=8)
    assert sliced_v32["runtime"]["metadata"]["execution_policy"]["mtp_proposal_tokens"] == 0
    assert sliced_v32["runtime"]["homogeneous_graph_reuse"] == "off"
    assert compose_case("deepseek-v3.2", "a3", tp=4, ep=1, layers=1, homogeneous_graph_reuse="auto")["runtime"]["homogeneous_graph_reuse"] == "auto"


def test_host_dma_and_offload_rates_bind_strictly_to_hardware_profile():
    cases = {
        fam: compose_case("qwen3.8-flash-next", fam, tp=4, ep=4, layers=2)
        for fam in ("a2", "a3", "a5")
    }
    h2d_rates = {fam: float(cases[fam]["machine"]["device"]["rates"]["h2d"]) for fam in cases}
    mem_rates = {fam: float(cases[fam]["machine"]["device"]["rates"]["host_memory"]) for fam in cases}
    # Distinct calibrated PCIe Gen4 vs Gen5 & DDR rates, never flat 32e9 / 64e9 fallbacks.
    assert h2d_rates["a2"] < h2d_rates["a5"] < h2d_rates["a3"]
    assert all(rate != pytest.approx(32e9) for rate in h2d_rates.values())
    assert mem_rates["a5"] > mem_rates["a2"]
    assert all(rate != pytest.approx(64e9) for rate in mem_rates.values())

    for fam, raw in cases.items():
        raw["runtime"]["metadata"]["offload"] = {"kvc_ddr_ratio": 0.5}
        resolved = resolve_case_offload_policy(parse_case(raw))
        assert resolved is not None and resolved.ddr_bw_gbps is not None
        expected = resolve_sku_host_dma_rates(raw["machine"])["ddr_effective_bw_gbps"]
        assert resolved.ddr_bw_gbps == pytest.approx(expected)


def test_cli_compose_preserves_production_defaults_and_opt_out(tmp_path):
    on_path = tmp_path / "on.json"
    off_path = tmp_path / "off.json"
    assert cli_main([
        "compose", "--model", "qwen3.8-flash-next", "--hardware", "a3",
        "--tp", "4", "--ep", "4", "--layers", "2",
        "--prompt-tokens", "512", "--output-tokens", "64", "-o", str(on_path),
    ]) == 0
    on_case = json.loads(on_path.read_text(encoding="utf-8"))
    assert on_case["runtime"]["engram_cpu_offload"] is True
    assert on_case["runtime"]["multistream_overlap_shared_expert"] is True
    assert on_case["runtime"]["graph_capture_mode"] == "full"

    assert cli_main([
        "compose", "--model", "qwen3.8-flash-next", "--hardware", "a3",
        "--tp", "4", "--ep", "4", "--layers", "2",
        "--prompt-tokens", "512", "--output-tokens", "64",
        "--no-engram-cpu-offload", "--no-multistream-overlap-shared-expert",
        "--mtp-tokens", "0", "-o", str(off_path),
    ]) == 0
    off_case = json.loads(off_path.read_text(encoding="utf-8"))
    assert not off_case["runtime"].get("engram_cpu_offload", False)
    assert not off_case["runtime"].get("multistream_overlap_shared_expert", False)
    assert off_case["runtime"]["metadata"]["execution_policy"]["mtp_proposal_tokens"] == 0


def test_handoff_plan_conserves_dcp_cohorts_and_gqa_replication():
    p_mla = compose_case("deepseek-v3.2", "a5", tp=1, dp=1, pcp=8, ep=8, layers=4, prompt_tokens=1024, output_tokens=64)
    d_tp8_cp1 = compose_case("deepseek-v3.2", "a5", tp=8, dp=1, dcp=1, ep=8, layers=4, prompt_tokens=1024, output_tokens=64)
    d_tp8_dcp8 = compose_case("deepseek-v3.2", "a5", tp=8, dp=1, dcp=8, ep=8, layers=4, prompt_tokens=1024, output_tokens=64)
    d_dp8_dcp8 = compose_case("deepseek-v3.2", "a5", tp=1, dp=8, dcp=8, ep=8, layers=4, prompt_tokens=1024, output_tokens=64)

    plan_rep = compute_pd_handoff_plan(p_mla, d_tp8_cp1, tokens=1024)
    plan_tp_dcp = compute_pd_handoff_plan(p_mla, d_tp8_dcp8, tokens=1024)
    plan_dp_dcp = compute_pd_handoff_plan(p_mla, d_dp8_dcp8, tokens=1024)

    # DP=8, TP=1, DCP=8 (dcp_dp_width=8) must conserve 100% of KV bytes identically to TP=8, DCP=8.
    assert plan_dp_dcp["total_transfer_bytes"] == pytest.approx(plan_tp_dcp["total_transfer_bytes"])
    assert plan_dp_dcp["unique_kv_bytes"] == pytest.approx(plan_rep["unique_kv_bytes"])
    assert len(plan_dp_dcp["destination_ranks"]) == 8
    # Unsharded TP=8 MLA replicates full KV to all 8 decode ranks, strictly exceeding DCP=8.
    assert plan_rep["total_transfer_bytes"] > plan_tp_dcp["total_transfer_bytes"]
    assert plan_rep["tail_layer_bytes"] == pytest.approx(plan_rep["total_transfer_bytes"] / 4.0)

    # GQA with 8 KV heads and decode TP=16 replicates each head twice -> total == 2 * unique.
    p_gqa = compose_case("qwen3-32b", "a5", tp=4, layers=2, prompt_tokens=1024, output_tokens=64)
    d_gqa = compose_case("qwen3-32b", "a5", tp=16, layers=2, prompt_tokens=1024, output_tokens=64)
    plan_gqa = compute_pd_handoff_plan(p_gqa, d_gqa, tokens=1024)
    assert plan_gqa["total_transfer_bytes"] == pytest.approx(2.0 * plan_gqa["unique_kv_bytes"])


def test_three_tier_pd_modes_link_derivation_and_prefix_hit():
    p = compose_case("qwen3-32b", "a5", tp=2, layers=4, ranks_per_node=8, prompt_tokens=2048, output_tokens=128)
    d = compose_case("qwen3-32b", "a5", tp=2, layers=4, ranks_per_node=8, prompt_tokens=2048, output_tokens=128)
    common = dict(
        total_dies=16, prompt_tokens=2048, output_tokens=128,
        ttft_sla_ms=5000.0, tpot_sla_ms=50.0,
    )
    direct = sweep_pd_disaggregation_ratios(p, d, pd_transfer_mode="direct_blocking_d2d", **common)
    overlap = sweep_pd_disaggregation_ratios(p, d, pd_transfer_mode="layerwise_overlap_d2d", **common)
    pooled_dram = sweep_pd_disaggregation_ratios(p, d, pd_transfer_mode="pooled_dram_ssd", **common)
    pooled_hit = sweep_pd_disaggregation_ratios(
        p, d, pd_transfer_mode="pooled_dram_ssd", pool_prefix_hit_ratio=0.5, **common,
    )
    pooled_ssd = sweep_pd_disaggregation_ratios(
        p, d, pd_transfer_mode="pooled_dram_ssd",
        pool_dram_ratio=0.5, pool_ssd_ratio=0.5, ssd_bw_gbps=7.0, **common,
    )
    assert direct["kv_link_source"] == "sku_link:d2d_inter_node"
    assert direct["kv_transfer_bw_gbps"] != pytest.approx(100.0)
    # Layerwise overlap hides interior layers behind prefill compute.
    assert overlap["exposed_kv_handoff_ms"] < direct["exposed_kv_handoff_ms"]
    assert overlap["first_decode_itl_ms"] < direct["first_decode_itl_ms"]
    # Prefix hit in 3-tier pool skips recomputing hit prefix tokens on prefill and boosts throughput.
    assert pooled_hit["prefill_instance"]["effective_prefill_tokens"] == 1024
    assert pooled_hit["prefill_instance"]["step_ms"] < pooled_dram["prefill_instance"]["step_ms"]
    assert pooled_hit["best_configuration"]["sustainable_tps"] > pooled_dram["best_configuration"]["sustainable_tps"]
    # SSD tier charges slower external SSD bandwidth & latency than DRAM-only pool.
    assert pooled_ssd["kv_transfer_ms_per_request"] > pooled_dram["kv_transfer_ms_per_request"]
    # Intra-node SuperNode cluster derives intra-node UB link rather than inter-node RoCE.
    p_pod = compose_case("qwen3-32b", "a5", tp=2, layers=2, topology_preset="a5_pod", prompt_tokens=1024, output_tokens=64)
    intra = sweep_pd_disaggregation_ratios(
        p_pod, p_pod, total_dies=8, prompt_tokens=1024, output_tokens=64,
        ttft_sla_ms=5000.0, tpot_sla_ms=50.0,
    )
    assert intra["kv_link_source"] == "sku_link:d2d_intra_node"
    assert intra["kv_transfer_bw_gbps"] > direct["kv_transfer_bw_gbps"]


def test_domain_three_auto_optimal_joint_pd_colocated_and_pinned_constraints():
    coloc_only = solve_optimal_deployment(
        "qwen3-32b", "a5",
        num_devices=4, layers=2,
        prompt_tokens=1024, output_tokens=64,
        ttft_sla_ms=5000.0, tpot_sla_ms=50.0,
        deployment="colocated",
        pinned_parallel={"tp": 2},
        max_sequences_per_dp=16,
    )
    assert coloc_only["schema"] == "mindie-sim.optimal-deployment/1"
    assert coloc_only["feasible"] is True
    assert coloc_only["winner_mode"] == "colocated"
    assert coloc_only["best_pd"] is None
    assert coloc_only["best_configuration"]["parallel"]["tp"] == 2
    assert coloc_only["deployment_outcome"]["status"] == "feasible"

    pd_only = solve_optimal_deployment(
        "qwen3-32b", "a5",
        num_devices=8, layers=2,
        prompt_tokens=1024, output_tokens=64,
        ttft_sla_ms=5000.0, tpot_sla_ms=50.0,
        deployment="pd_disaggregated",
        pd_transfer_mode="layerwise_overlap_d2d",
        pinned_prefill_parallel={"tp": 2},
        pinned_decode_parallel={"tp": 2},
        max_sequences_per_dp=16,
    )
    assert pd_only["winner_mode"] == "pd_disaggregated"
    assert pd_only["best_colocated"] is None
    assert pd_only["best_configuration"]["pd_transfer_mode"] == "layerwise_overlap_d2d"
    assert pd_only["best_configuration"]["prefill_parallel"]["tp"] == 2
    assert pd_only["best_configuration"]["decode_parallel"]["tp"] == 2

    auto_case = compose_case(
        "qwen3-32b", "a5",
        num_devices=4, layers=2,
        prompt_tokens=1024, output_tokens=64,
        ttft_slo_ms=5000.0, tpot_slo_ms=50.0,
        deployment="colocated",
    )
    opt_meta = auto_case["runtime"]["metadata"]["optimal_deployment"]
    assert opt_meta["feasible"] is True
    assert opt_meta["winner_mode"] == "colocated"


def test_kimi_k31_a5_fp8_canonicalization_pd_decode_workspace_and_scope_provenance():
    from mindie_sim.prediction import predict
    from mindie_sim.prediction.capacity import Memory

    # Live dogfooding regression: kimi-k3.1 on 950DT 96GiB SuperPod, 128K in / 1K out,
    # W4A8_MXFP + fp8 KV/Indexer + mtp=5 must not crash binding or falsely OOM workspace.
    raw = compose_case(
        "kimi-k3.1", "950dt-96g-pod",
        tp=8, dcp=8, ep=8, layers=4, role="pd_decode",
        prompt_tokens=131072, output_tokens=1024,
        mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
        kv_cache_dtype="fp8", indexer_kv_dtype="fp8",
        tpot_slo_ms=30.0,
    )
    assert raw["machine"]["sku"] == "Ascend950DT_9582"
    assert raw["model"]["quant"]["name"] == "mxfp4_routed_only"
    assert raw["model"]["quant"]["weight_format"] == "bf16"
    assert raw["model"]["quant"]["expert_weight_format"] == "mxfp4"
    assert raw["model"]["quant"]["kv_dtype"] == "fp8_e4m3"
    assert raw["model"]["quant"]["index_dtype"] == "fp8_e4m3"
    assert raw["runtime"]["max_num_batched_tokens"] <= 768
    mem = Memory(parse_case(raw))
    assert max(mem.workspace) < 16 * (1024 ** 3)
    assert all(cap > 0 for row in mem.capacity for cap in row)

    model_step = predict(raw, scope="model", phase="decode", tokens=6, context=131072, sequences=1)
    assert model_step["duration_ms"] > 0.0
    inc_model = model_step["included_components"]
    assert inc_model["target_layers"] == 4
    assert inc_model["draft_layers"] > 0
    assert inc_model["embedding_lm_head"] is True

    # Auto-routing geometry probes without explicit scope="body" never throws.
    auto_step = predict(raw, phase="decode", tokens=6, context=131072, sequences=1)
    assert auto_step["duration_ms"] > 0.0
    assert auto_step["included_components"]["embedding_lm_head"] is False
    assert auto_step["included_components"]["draft_layers"] == 0


def test_aliases_self_correcting_errors_catalog_cli_parity_and_oneshot_solve(tmp_path):
    from mindie_sim.cli.parser import build_parser
    from mindie_sim.compose.inputs import _resolve_hardware

    assert _resolve_hardware("950dt-96g")[0] == "Ascend950DT_9582"
    assert _resolve_hardware("950dt-96g-pod")[:3:2] == ("Ascend950DT_9582", "a5_pod")
    assert _resolve_hardware("a5-96g-pod")[:3:2] == ("Ascend950DT_9582", "a5_pod")
    assert _resolve_hardware("950dt-144g-pod")[:3:2] == ("Ascend950DT_9581", "a5_pod")
    assert _resolve_hardware("a3-16p")[:3:2] == ("Ascend910_9392", "a3_16p")
    assert _resolve_hardware("a3-pod")[:3:2] == ("Ascend910_9392", "a3_superpod")
    assert _resolve_hardware("a2-8p")[:3:2] == ("Ascend910B4", "a2_standard")

    c_mxfp = compose_case("qwen3-32b", "a5", tp=2, layers=2, quant="w4a8_mxfp", kv_cache_dtype="c8")
    assert c_mxfp["model"]["quant"]["weight_format"] == "fp8_e4m3"
    assert c_mxfp["model"]["quant"]["expert_weight_format"] == "mxfp4"
    assert c_mxfp["model"]["quant"]["kv_dtype"] == "int8"

    with pytest.raises(ModelingError, match=r"unknown_hardware_sku:950dt-bad.*GiB.*presets:"):
        compose_case("qwen3-32b", "950dt-bad", tp=1, layers=1)
    with pytest.raises(ModelingError, match=r"unknown_model:qwen3-32b-typo.*closest valid:"):
        compose_case("qwen3-32b-typo", "a3", tp=1, layers=1)
    with pytest.raises(ModelingError, match=r"invalid_quant_override:w4a99.*closest valid:"):
        compose_case("qwen3-32b", "a3", tp=1, layers=1, quant="w4a99")

    cat_out = tmp_path / "catalog.json"
    assert cli_main(["catalog", "--hardware", "--presets", "--quant", "-o", str(cat_out)]) == 0
    cat = json.loads(cat_out.read_text(encoding="utf-8"))
    assert any(row["sku"] == "Ascend950DT_9582" and row["hbm_gib"] == 96.0 for row in cat["hardware"])
    assert "a5_pod" in cat["presets"] and "mxfp4" in cat["quant"]["presets"]

    parser = build_parser()
    sw_default = parser.parse_args(["sweep", "case.json"])
    assert (sw_default.allow_pp, sw_default.allow_pcp, sw_default.allow_dcp, sw_default.two_stage) == (True, True, True, True)
    sw_off = parser.parse_args(["sweep", "case.json", "--no-pp", "--no-pcp", "--no-dcp", "--no-two-stage"])
    assert (sw_off.allow_pp, sw_off.allow_pcp, sw_off.allow_dcp, sw_off.two_stage) == (False, False, False, False)
    ft = parser.parse_args([
        "feature-tracks", "--model", "kimi-k3.1", "--hardware", "950dt-96g-pod",
        "--mtp-tokens", "5", "--mtp-accept", "3.5",
        "--kv-cache-dtype", "fp8", "--indexer-kv-dtype", "fp8",
        "--topology-preset", "a5_pod", "--linear-proj-tp", "8", "--o-proj-tp", "8",
    ])
    assert (ft.mtp_accept, ft.kv_cache_dtype, ft.indexer_kv_dtype, ft.linear_proj_tp) == (3.5, "fp8", "fp8", 8)

    # One-shot solve_deployment via SDK & CLI (explicit --deployment-mode; ttft_slo_ms=None allowed).
    sol_out = tmp_path / "solve.json"
    assert cli_main([
        "solve", "--model", "qwen3-32b", "--hardware", "950dt-96g-pod",
        "--num-devices", "4", "--layers", "2",
        "--deployment-mode", "colocated",
        "--prompt-tokens", "512", "--output-tokens", "64",
        "--tpot-slo-ms", "50.0", "--max-sequences-per-dp", "8",
        "-o", str(sol_out),
    ]) == 0
    sol = json.loads(sol_out.read_text(encoding="utf-8"))
    assert sol["schema"] == "mindie-sim.deployment-solution/1"
    assert sol["feasible"] is True
    assert sol["scenario"]["ttft_slo_ms"] is None
    assert sol["capacity_ledger"]["hbm_capacity_gib"] == pytest.approx(96.0)
    assert sol["recommended_deployment"]["winner_mode"] == "colocated"
    assert "shipping" in sol["four_track_waterfall"]["waterfall"]
    assert sol["bottleneck_roofline"]["top_families"]
    assert sol["bottleneck_roofline"]["included_components"]["target_layers"] == 2
    sdk_sol = sdk_api.solve_deployment(
        "qwen3-32b", "950dt-96g-pod", num_devices=4, layers=2,
        prompt_tokens=512, output_tokens=64, tpot_slo_ms=50.0,
        deployment="colocated", kv_pooling="off", max_sequences_per_dp=8,
    )
    assert sdk_sol["feasible"] is True


def test_kimi_k31_five_live_correctness_regressions():
    from types import SimpleNamespace
    from mindie_sim.prediction.efficiency import analyze_network_efficiency, solve_sla_sizing
    from mindie_sim.prediction.operators.linear_attention.common import (
        _resolve_recover_ssm_accepted_per_request,
        validate_contract,
    )
    from mindie_sim.prediction.results import StepRequest
    from mindie_sim.schema.operators import OpSpec
    from mindie_sim.schema.workload import RequestView

    # Bug #1: PP=4 x PCP=8 x EP=8 (VA Kimi-K3.1 Prefill) must not IndexError in MC2 transport.
    prefill = compose_case(
        "kimi-k3.1", "950dt-96g-pod",
        dp=1, pp=4, pcp=8, tp=1, ep=8, layers=4, role="pd_prefill",
        prompt_tokens=1024, output_tokens=64, ttft_slo_ms=5000.0,
    )
    p_eff = analyze_network_efficiency(
        prefill, phase="prefill", tokens=1024, context=0, sequences=1, scope="body",
    )
    assert p_eff["graph_duration_ms"] > 0.0

    # Bug #2: shipping tutorial_flags enable_fused_mc2=0 must stay on shipping only and never leak into target.
    common = dict(
        dp=1, tp=8, dcp=8, ep=8, layers=4, role="pd_decode",
        prompt_tokens=131072, output_tokens=1024,
        mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
    )
    c_merged = compose_case("kimi-k3.1", "950dt-96g-pod", feature_track="shipping", **common)
    c_ideal = compose_case("kimi-k3.1", "950dt-96g-pod", feature_track="target", **common)
    assert c_merged["runtime"]["metadata"]["additional_config"]["enable_fused_mc2"] == 0
    assert c_ideal["runtime"]["metadata"]["additional_config"].get("enable_fused_mc2", 0) != 0
    # kimi-k3 / kimi-k3.1 default native mxfp4_routed_only preserves BF16 exclusions;
    # explicit quant="w4a8_mxfp" yields MXFP8 linears + MXFP4 routed experts, and
    # explicit quant="mxfp4_all" overrides exclusions with MX scales.
    for q_override in (None, "mxfp4_routed_only"):
        c_ro = compose_case("kimi-k3.1", "950dt-96g-pod", quant=q_override, **common)
        assert c_ro["model"]["quant"]["name"] == "mxfp4_routed_only"
        assert {"attn", "mla", "kda", "shared_expert", "dense_mlp"} <= set(c_ro["model"]["quant"]["exclusions"])
    c_w4a8_mx = compose_case("kimi-k3.1", "950dt-96g-pod", quant="w4a8_mxfp", **common)
    assert c_w4a8_mx["model"]["quant"]["weight_format"] == "fp8_e4m3"
    assert c_w4a8_mx["model"]["quant"]["expert_weight_format"] == "mxfp4"
    c_k3 = compose_case("kimi-k3", "950dt-96g-pod", tp=8, ep=8, layers=4)
    assert c_k3["model"]["quant"]["name"] == "mxfp4_routed_only"
    c_all = compose_case("kimi-k3.1", "950dt-96g-pod", quant="mxfp4_all", **common)
    assert c_all["model"]["quant"]["activation_group_size"] == 32
    assert c_all["model"]["quant"]["activation_scale_format"] == "fp8_e8m0"
    assert not c_all["model"]["quant"]["exclusions"]
    # Target mla_dsa prolog: kimi-k3.1 has use_nope+qkv_a_gate_fused so it stays native_decomposed,
    # while deepseek-v3.2 fuses Target mla_dsa prolog into mla_prolog.native with index_q wired cleanly.
    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.structure import parse_case
    g_ideal = target_graphs(parse_case(c_ideal), phase="decode", tokens=6, context=131072, sequences=8, scope="model")[0]
    assert not any(op.op_id == "s0.l3.mla_prolog.native" for op in g_ideal)
    idx_q_k31 = next(op for op in g_ideal if op.op_id == "s0.l3.index_q")
    assert idx_q_k31.deps
    c_v32_ideal = compose_case("deepseek-v3.2", "a3", feature_track="target", tp=8, ep=8, layers=4, prompt_tokens=4096, output_tokens=64)
    g_v32 = target_graphs(parse_case(c_v32_ideal), phase="decode", tokens=1, context=4096, sequences=1, scope="body")[0]
    assert any(op.op_id == "s0.l0.mla_prolog.native" for op in g_v32)
    assert "s0.l0.mla_prolog.native" in next(op for op in g_v32 if op.op_id == "s0.l0.index_q").deps
    e_merged = analyze_network_efficiency(c_merged, phase="decode", tokens=6, context=131072, sequences=8, scope="body")
    e_ideal = analyze_network_efficiency(c_ideal, phase="decode", tokens=6, context=131072, sequences=8, scope="body")
    # With MC2 ⊥ MSO mutex enforced, c_merged runs discrete SiTU + MSO while c_ideal runs
    # routed MegaMoE without MSO; both must execute cleanly with valid positive duration.
    assert e_merged["graph_duration_ms"] > 0.0 and e_ideal["graph_duration_ms"] > 0.0
    assert c_ideal["runtime"]["metadata"]["moe_route"]["effective_multistream_overlap_shared_expert"] is False
    assert c_merged["runtime"]["metadata"]["moe_route"]["effective_multistream_overlap_shared_expert"] is True

    # #365 Item 2: mindie-sim solve parses optional --xlsx / --trace / --compare-track / --target-track flags.
    from mindie_sim.cli.parser import build_parser
    parsed_solve = build_parser().parse_args([
        "solve", "--model", "kimi-k3.1", "--hardware", "950dt-96g-pod",
        "--num-devices", "8", "--prompt-tokens", "131072", "--output-tokens", "1024",
        "--tpot-slo-ms", "30", "--xlsx", "out.xlsx", "--trace", "trace_view.json",
    ])
    assert (parsed_solve.xlsx, parsed_solve.trace, parsed_solve.compare_track, parsed_solve.target_track) == (
        "out.xlsx", "trace_view.json", "shipping", "target",
    )

    # Bug #3: solve_sla_sizing & sweep default to scope="model" conserving Target + Draft + Logits.
    sla_rep = solve_sla_sizing(
        c_merged, sla_ms=50.0, context=131072, output_budget=1024,
        max_sequences_per_dp=1, sensitivity="baseline",
    )
    assert sla_rep["scope"] == "model"
    assert sla_rep["timing_boundary"] == "full_model_step"
    assert sla_rep["included_components"]["draft_layers"] > 0
    assert sla_rep["included_components"]["embedding_lm_head"] is True

    # Bug #4: analyze_network_efficiency(scope="model") with DSpark conserves graph_duration_ms <= operator_sum_ms.
    e_model = analyze_network_efficiency(c_merged, phase="decode", tokens=6, context=131072, sequences=8, scope="model")
    assert e_model["graph_duration_ms"] <= e_model["operator_sum_ms"] + 1e-6

    # Bug #5: causal_conv1d & recover_ssm_commit validate cleanly from ctx.requests
    # when metadata omits lengths/acceptance, and spine evaluate_step runs both
    # raw_eager and unmerged_pr (recover_ssm=True, mtp_acceptance_length=3.5) on kimi-k3.1.
    conv_op = OpSpec(
        op_id="s0.l0.short_convolution", family="short_convolution", phase="decode",
        stage=0, layer=0, requests=("r0", "r1"),
        semantic_shape=(12, 128), local_shape=(12, 128), padded_shape=(12, 128), launch_shape=(12, 128),
        recipe="causal_conv1d", dtype="bf16",
        metadata=(("channels", 128), ("width", 4), ("mode", "incremental"), ("activation", "silu")),
    )
    conv_ctx = SimpleNamespace(
        requested_abi="",
        requests=(
            RequestView("r0", 6, 131584, True, "decode", draft_proposal_tokens=5, draft_accepted_tokens=3),
            RequestView("r1", 6, 131584, True, "decode", draft_proposal_tokens=5, draft_accepted_tokens=4),
        ),
    )
    contract = validate_contract(conv_op, conv_ctx)
    assert contract.lengths == (6, 6)
    with pytest.raises(ModelingError, match="recover_ssm_requires_integer_acceptance"):
        _resolve_recover_ssm_accepted_per_request({"accepted_tokens": 3.5}, conv_ctx, 6, 2)
    assert _resolve_recover_ssm_accepted_per_request({}, conv_ctx, 6, 2) == (3, 4)
    c_eager = compose_case("kimi-k3.1", "950dt-96g-pod", feature_track="eager", quant="mxfp4", **common)
    step_eval = sdk_api.evaluate_step(c_eager, StepRequest(phase="decode", tokens=6, context=131584, sequences=8, scope="model"))
    assert step_eval.span_ms > 0.0
    c_unmerged = compose_case("kimi-k3.1", "950dt-96g-pod", feature_track="proposed", **common)
    step_unmerged = sdk_api.evaluate_step(c_unmerged, StepRequest(phase="decode", tokens=6, context=131584, sequences=8, scope="model"))
    assert step_unmerged.span_ms > 0.0


def test_memory_hbm_pools_draft_cap_engram_offload_and_pd_runtime_sharding():
    """Verify HBM cache pool conservation, draft_max_model_len cap, Engram CPU offload, and PD runtime sharding."""
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.efficiency.sla import _vram_max_sequences_per_dp

    # 1) 128K Kimi-K3.1 950DT 96G (DP4 TP8 EP32 DCP8, K=5, FP8 KV/Indexer):
    #    target MLA KV + DSA Indexer K + KDA [K+1] SSM slots + Draft KV all accounted & conserved.
    base_k31 = dict(
        dp=4, tp=8, dcp=8, ep=32, role="pd_decode",
        prompt_tokens=131072, output_tokens=1024,
        mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
        kv_cache_dtype="fp8", indexer_kv_dtype="fp8",
        max_num_seqs=128, max_num_batched_tokens=1024,
    )
    c_k5_norec = parse_case(compose_case(
        "kimi-k3.1", "950dt-96g-pod", feature_track="shipping", recover_ssm=False, **base_k31,
    ))
    mem_k5 = Memory(c_k5_norec)
    seq_len = 131072 + 1024
    bd_k5 = mem_k5.cache_pool_breakdown(seq_len, padded=True)
    assert bd_k5["query_tokens"] == 6
    assert bd_k5["target_kv_bytes"] > 0.0
    assert bd_k5["indexer_k_bytes"] > 0.0
    assert bd_k5["recurrent_state_bytes"] > 0.0
    assert bd_k5["draft_kv_bytes"] > 0.0
    pool_sum = (
        bd_k5["target_kv_bytes"] + bd_k5["indexer_k_bytes"]
        + bd_k5["recurrent_state_bytes"] + bd_k5["draft_kv_bytes"]
        + bd_k5["draft_state_bytes"] + bd_k5["page_reserve_bytes"]
        + bd_k5["page_metadata_bytes"]
    )
    assert pool_sum == pytest.approx(bd_k5["total_bytes"])
    assert bd_k5["total_bytes"] == pytest.approx(sum(mem_k5.cache_bytes(seq_len, padded=True)))
    # K+1 speculative SSM state scales up with K=7 (recover_ssm=False) and drops sharply
    # under opt-in recover_ssm=True.
    c_k7 = parse_case(compose_case(
        "kimi-k3.1", "950dt-96g-pod", feature_track="shipping", recover_ssm=False,
        **{**base_k31, "mtp_proposal_tokens": 7},
    ))
    bd_k7 = Memory(c_k7).cache_pool_breakdown(seq_len, padded=True)
    assert bd_k7["query_tokens"] == 8
    assert bd_k7["recurrent_state_bytes"] > bd_k5["recurrent_state_bytes"]

    c_rec = parse_case(compose_case(
        "kimi-k3.1", "950dt-96g-pod", feature_track="shipping", recover_ssm=True, **base_k31,
    ))
    assert c_rec.runtime.recover_ssm is True
    bd_rec = Memory(c_rec).cache_pool_breakdown(seq_len, padded=True)
    assert bd_rec["recurrent_state_bytes"] < bd_k5["recurrent_state_bytes"]
    # Realistic 128K capacity on 950DT-96G with shipping recover_ssm=True (~24-36 seqs/DP, never 75),
    # strictly exceeding the non-recover_ssm K+1 baseline.
    max_seqs_shipping = _vram_max_sequences_per_dp(c_rec, context=131072, tokens_per_seq=6, output_budget=1024)
    max_seqs_norec = _vram_max_sequences_per_dp(c_k5_norec, context=131072, tokens_per_seq=6, output_budget=1024)
    assert 20 <= max_seqs_shipping <= 36
    assert 12 <= max_seqs_norec < max_seqs_shipping

    # 2) Non-paired_sfa draft KV cap (draft_max_model_len=4096) shrinks cache_bytes & conserves rank views.
    raw_uncapped = compose_case(
        "deepseek-v3.2", "950dt-96g-pod", layers=4,
        dp=1, tp=4, dcp=4, ep=4, role="pd_decode",
        prompt_tokens=65536, output_tokens=1024,
        mtp_proposal_tokens=3, mtp_acceptance_length=2.5,
    )
    raw_capped = compose_case(
        "deepseek-v3.2", "950dt-96g-pod", layers=4,
        dp=1, tp=4, dcp=4, ep=4, role="pd_decode",
        prompt_tokens=65536, output_tokens=1024,
        mtp_proposal_tokens=3, mtp_acceptance_length=2.5,
    )
    raw_capped["runtime"]["metadata"]["draft_max_model_len"] = 4096
    mem_uncapped = Memory(parse_case(raw_uncapped))
    mem_capped = Memory(parse_case(raw_capped))
    bd_uncapped = mem_uncapped.cache_pool_breakdown(65536, padded=True)
    bd_capped = mem_capped.cache_pool_breakdown(65536, padded=True)
    assert bd_capped["draft_context_cap"] == 4096
    assert bd_capped["draft_kv_bytes"] < bd_uncapped["draft_kv_bytes"]
    assert sum(mem_capped.cache_bytes(65536, padded=True)) < sum(mem_uncapped.cache_bytes(65536, padded=True))
    rank_views = mem_capped.rank_cache_allocation(65536)
    assert len(rank_views) == 4
    expected_capped = sum(mem_capped.cache_bytes(65536, padded=True))
    assert all(sum(view["allocated"]["total_bytes_per_stage_rank"]) == pytest.approx(expected_capped) for view in rank_views)

    # 3) Engram CPU offload excludes Engram tables from HBM weights and records engram_host_dram_bytes.
    raw_eng_off = compose_case(
        "deepseek-v4.1-flash", "950dt-96g-pod", layers=4, dp=1, tp=4, ep=4,
        prompt_tokens=4096, output_tokens=64, engram_cpu_offload=False,
    )
    raw_eng_on = compose_case(
        "deepseek-v4.1-flash", "950dt-96g-pod", layers=4, dp=1, tp=4, ep=4,
        prompt_tokens=4096, output_tokens=64, engram_cpu_offload=True,
    )
    mem_eng_off = Memory(parse_case(raw_eng_off))
    mem_eng_on = Memory(parse_case(raw_eng_on))
    assert sum(mem_eng_on.weights) < sum(mem_eng_off.weights)
    assert sum(mem_eng_on.engram_host_dram_bytes) > 0.0
    assert sum(mem_eng_off.engram_host_dram_bytes) == 0.0
    assert mem_eng_on.cache_pool_breakdown(4096)["engram_host_dram_bytes"] > 0.0

    # 4) PD handoff honors runtime dcp_indexer_sharded and excludes target speculative SSM slots from draft_kv_bytes.
    p_pd = compose_case(
        "kimi-k3.1", "950dt-96g-pod", layers=4, dp=1, pp=1, pcp=4, tp=1, ep=4,
        role="pd_prefill", prompt_tokens=4096, output_tokens=64, mtp_proposal_tokens=5,
        ttft_slo_ms=5000.0,
    )
    d_pd_rep = compose_case(
        "kimi-k3.1", "950dt-96g-pod", layers=4, dp=1, pp=1, tp=4, dcp=4, ep=4,
        role="pd_decode", prompt_tokens=4096, output_tokens=64, mtp_proposal_tokens=5,
    )
    d_pd_shd = compose_case(
        "kimi-k3.1", "950dt-96g-pod", layers=4, dp=1, pp=1, tp=4, dcp=4, ep=4,
        role="pd_decode", prompt_tokens=4096, output_tokens=64, mtp_proposal_tokens=5,
    )
    d_pd_rep["runtime"]["metadata"]["dcp_indexer_sharded"] = False
    d_pd_shd["runtime"]["metadata"]["dcp_indexer_sharded"] = True
    plan_rep = compute_pd_handoff_plan(p_pd, d_pd_rep, tokens=4096)
    plan_shd = compute_pd_handoff_plan(p_pd, d_pd_shd, tokens=4096)
    assert plan_shd["total_transfer_bytes"] < plan_rep["total_transfer_bytes"]
    d_mem_shd = Memory(parse_case(d_pd_shd))
    bd_unpad_q1 = d_mem_shd.cache_pool_breakdown(4096, padded=False, query_tokens=1)
    assert plan_shd["draft_kv_bytes"] == pytest.approx(bd_unpad_q1["draft_kv_bytes"] * 4.0)


def test_archetype_folding_zero_drift_on_unseen_irregular_hybrid_with_prime_quant_exclusions():
    from mindie_sim.prediction.evaluation import EvaluationContext
    from mindie_sim.prediction.execution.layer_runs import discover_layer_cohorts
    from mindie_sim.prediction.graph_analysis import graph_costs
    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.sweep import pd as pd_mod

    # Synthetic unseen 10-layer irregular hybrid (Dense+MoE+KDA+MLA) with prime-numbered per-layer
    # quant exclusions and category_overrides so cohort discovery preserves exact per-layer quant.
    raw = compose_case(
        "kimi-k3.1", "950dt-96g-pod",
        dp=1, tp=2, ep=2, pp=1, layers=10,
        prompt_tokens=2048, output_tokens=64, concurrency=2,
        quant="mxfp8", kv_cache_dtype="fp8_e4m3",
    )
    prime_exclusions = [f"model.layers.{idx}.self_attn" for idx in (2, 3, 5, 7)]
    raw["model"]["quant"]["exclusions"] = list(raw["model"]["quant"].get("exclusions", ())) + prime_exclusions
    raw["model"]["quant"]["category_overrides"] = {
        **dict(raw["model"]["quant"].get("category_overrides", {})),
        "model.layers.7.mlp": {"weight_format": "bf16"},
    }
    case = parse_case(raw)
    assert not hasattr(pd_mod, "build_archetype_representative_case")
    plan = discover_layer_cohorts(case)
    assert sum(c.count for c in plan.cohorts) == 10
    assert len(plan.cohorts) >= 5  # Mixed KDA/MLA/Dense/MoE + prime quant exclusions + layer-7 override.

    full_graphs = target_graphs(case, phase="decode", tokens=1, context=2048, sequences=2, scope="model")
    full_ctx = EvaluationContext.create(case)
    full_ms = float(graph_costs(
        case, full_graphs, isolated=False, evaluation=full_ctx, timeline_replicas=(0,),
    )["duration_ms"])
    folded_ms = pd_mod._archetype_weighted_step_ms(
        case, phase="decode", tokens=1, context=2048, sequences=2, scope="model", min_layers=2,
    )
    assert folded_ms == pytest.approx(full_ms, rel=1e-6, abs=1e-6)


def test_iso_sla_independent_before_after_solve_workspace_and_direct_d2d_ttft():
    from mindie_sim.sweep.pd import sweep_pd_disaggregation_ratios

    sol_mod = sdk_api.solve_deployment(
        "qwen3-32b", "950dt-96g-pod",
        num_devices=4, layers=2,
        prompt_tokens=512, output_tokens=64,
        tpot_slo_ms=50.0, deployment="colocated", max_sequences_per_dp=8,
        mode="modeling", compare_track="shipping", target_track="target",
    )
    assert sol_mod["mode"] == "modeling" and sol_mod["allow_unbound_formula"] is True
    iso = sol_mod["iso_sla_comparison"]
    assert iso["before"]["track"] == "shipping" and iso["after"]["track"] == "target"
    assert iso["before"]["feasible"] is True and iso["after"]["feasible"] is True
    assert iso["after"]["sustainable_tps_per_die"] >= iso["before"]["sustainable_tps_per_die"] - 1e-6
    assert sol_mod["capacity_ledger"]["workspace_tokens"] >= 1

    sol_sim = sdk_api.solve_deployment(
        "qwen3-32b", "950dt-96g-pod",
        num_devices=4, layers=2,
        prompt_tokens=512, output_tokens=64,
        tpot_slo_ms=50.0, deployment="colocated", max_sequences_per_dp=8,
        mode="simulation", compare_track="shipping", target_track="target",
    )
    assert sol_sim["mode"] == "simulation" and sol_sim["allow_unbound_formula"] is False
    assert sol_sim["iso_sla_comparison"]["after"]["track"] == "shipping"

    # #406 Item 13: direct_blocking_d2d TTFT check includes exposed KV transfer delay.
    prefill = compose_case("qwen3-32b", "a3", dp=1, tp=2, ep=1, layers=1, prompt_tokens=256, output_tokens=64)
    decode = compose_case("qwen3-32b", "a3", dp=1, tp=2, ep=1, layers=1, prompt_tokens=256, output_tokens=64)
    slow_kv = sweep_pd_disaggregation_ratios(
        prefill, decode, total_dies=4,
        prompt_tokens=256, output_tokens=64,
        ttft_sla_ms=50.0, tpot_sla_ms=200.0,
        pd_transfer_mode="direct_blocking_d2d",
        kv_transfer_bw_gbps=0.01,
    )
    assert slow_kv["prefill_instance"]["effective_ttft_ms"] > slow_kv["prefill_instance"]["step_ms"]
    assert all(not row["feasible"] and row["infeasible_reason"] == "prefill_ttft_sla_exceeded" for row in slow_kv["candidates"])


def _run_kimi_k31_32die_rss_probe(payload: dict[str, object]) -> dict[str, object]:
    """Isolated child-process target for 32-die 128K Decode Iso-SLA RSS gate."""
    import time
    from mindie_sim.solve import solve_deployment

    t0 = time.perf_counter()
    sol = solve_deployment(
        "kimi-k3.1", "950dt-96g-pod",
        num_devices=int(payload.get("num_devices", 32)),
        layers=int(payload["layers"]) if payload.get("layers") is not None else None,
        prompt_tokens=int(payload.get("prompt_tokens", 131072)),
        output_tokens=int(payload.get("output_tokens", 1024)),
        tpot_slo_ms=float(payload.get("tpot_slo_ms", 30.0)),
        deployment_mode=str(payload.get("deployment_mode", "pd_decode")),
        quant=str(payload.get("quant", "mxfp8_mxfp4")),
        mtp_tokens=int(payload.get("mtp_tokens", 0)),
        mtp_accept=(float(payload["mtp_accept"]) if payload.get("mtp_accept") is not None else None),
        kv_cache_dtype="fp8", indexer_kv_dtype="fp8",
        pinned_parallel=dict(payload["pinned_parallel"]) if payload.get("pinned_parallel") else None,
        mode=str(payload.get("mode", "modeling")),
        max_sequences_per_dp=payload.get("max_sequences_per_dp", 32),
    )
    elapsed = time.perf_counter() - t0
    return {
        "feasible": bool(sol["feasible"]),
        "elapsed_s": round(elapsed, 3),
        "deployment_mode": sol["deployment_mode"],
        "deployment_outcome": sol["deployment_outcome"],
        "winner": sol["recommended_deployment"].get("winner_mode"),
        "search_coverage": sol["recommended_deployment"].get("search_coverage"),
        "before": sol["iso_sla_comparison"]["before"],
        "after": sol["iso_sla_comparison"]["after"],
        "ledger": sol["capacity_ledger"],
    }


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc/self/status VmRSS/VmHWM required")
def test_kimi_k31_32die_solve_deployment_peak_rss_bounded():
    """Isolated Linux child RSS probe: 32-die 128K Decode solves with <250 MiB delta & <=512 MiB peak."""
    from _support.rss_probe import assert_rss_budget, measure_child_rss

    measurement = measure_child_rss(
        "serving.test_three_domain_and_pooled_pd:_run_kimi_k31_32die_rss_probe",
        {
            "num_devices": 32,
            "layers": 4,
            "prompt_tokens": 131072,
            "output_tokens": 1024,
            "tpot_slo_ms": 30.0,
            "deployment_mode": "pd_decode",
            "quant": "mxfp8_mxfp4",
            "mtp_tokens": 0,
            "mtp_accept": None,
            "pinned_parallel": {"dp": 4, "tp": 8, "ep": 32, "pp": 1, "dcp": 8},
            "mode": "simulation",
            "max_sequences_per_dp": 20,
        },
        timeout_s=120.0,
    )
    metrics = measurement.result
    print(
        f"[32DIE_RSS_GATE] pre_mib={measurement.pre_rss_mib:.2f} "
        f"hwm_mib={measurement.hwm_mib:.2f} delta_mib={measurement.delta_hwm_mib:.2f} "
        f"elapsed_s={metrics['elapsed_s']} "
        f"after_par={metrics['after']['parallel']} "
        f"after_b_feas={metrics['after']['b_feasible']} "
        f"after_b_vram={metrics['after']['b_vram_max']} "
        f"after_tpot_ms={metrics['after']['effective_tpot_ms']:.3f} "
        f"after_tps_die={metrics['after']['decode_tps_per_die']:.2f}"
    )
    assert_rss_budget(measurement, max_delta_mib=250.0, max_hwm_mib=512.0)
    assert metrics["feasible"] is True, f"32-die solve unexpectedly infeasible: {metrics}"
    assert metrics["deployment_mode"] == "pd_decode"
    assert metrics["deployment_outcome"]["status"] == "feasible"
    assert metrics["deployment_outcome"]["verified_ttft_ms"] is None
    assert metrics["after"]["b_feasible"] > 16, f"Artificially capped at <=16: {metrics}"
    assert metrics["after"]["b_vram_max"] >= metrics["after"]["b_feasible"]
    assert metrics["after"]["configured_max_num_seqs"] == 20
    assert metrics["after"]["effective_tpot_ms"] <= 30.0 + 1e-6
    assert metrics["ledger"]["role"] == "pd_decode"
    assert metrics["ledger"]["max_seqs_per_dp_vram"] == metrics["after"]["b_vram_max"]
    if metrics["after"]["b_feasible"] < metrics["after"]["b_vram_max"]:
        assert metrics["after"]["operating_regime"] != "HBM_CAPACITY_BOUND"


def test_iso_sla_target_graph_ranking_dcp_rejection_and_b_vram_binary_search(monkeypatch):
    """Verify target-graph After search, single-die TPS ranking, DCP rejection, and binary-search b_vram."""
    from mindie_sim.export import build_export_bundle
    from mindie_sim.model_catalog import load_model
    from mindie_sim.prediction.results import StepRequest
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.solve import solve_deployment
    from mindie_sim.sweep import pd as pd_mod

    # 1. Single-Die Decode TPS ranking: 900 TPS / 4 dies (225/die) MUST beat 1000 TPS / 8 dies (125/die).
    row_4die = {
        "deployment_mode": "colocated", "used_dies": 4, "parallel": {"dp": 1, "tp": 4},
        "feasible": True, "effective_tpot_ms": 20.0,
        "sustainable_tps": 900.0, "sustainable_tps_per_die": 225.0,
        "decode_tps_per_die": 225.0, "us_per_tok_die": 1e6 / 225.0,
        "e2e_total_tps": 800.0, "e2e_tps_per_die": 200.0,
    }
    row_8die = {
        "deployment_mode": "colocated", "used_dies": 8, "parallel": {"dp": 2, "tp": 4},
        "feasible": True, "effective_tpot_ms": 20.0,
        "sustainable_tps": 1000.0, "sustainable_tps_per_die": 125.0,
        "decode_tps_per_die": 125.0, "us_per_tok_die": 1e6 / 125.0,
        "e2e_total_tps": 880.0, "e2e_tps_per_die": 110.0,
    }
    fin = pd_mod._finalize_optimal_deployment(
        model_id="test", sku="a3", n_dev=8, prompt=512, output=64, ttft=5000.0, tpot=50.0,
        dep_norm="colocated", all_candidates=[row_8die, row_4die],
        coloc_entries=[(row_8die, {"id": 8}), (row_4die, {"id": 4})],
        pd_entries=[], compose_for=lambda *_a, **_kw: {},
    )
    assert fin["best_configuration"]["used_dies"] == 4
    assert [r["used_dies"] for r in fin["pareto_frontier"]] == [4]

    # Colocated candidate separates decode-step Single_Die_TPS (1e6/us_per_tok_die) from full-request E2E TPS.
    col_raw = compose_case("qwen3-32b", "a3", tp=2, layers=2, prompt_tokens=512, output_tokens=64)
    col_row = pd_mod._evaluate_colocated_candidate(
        col_raw, prompt_tokens=512, output_tokens=64, ttft_sla_ms=5000.0, tpot_sla_ms=50.0,
        mtp_tokens=0, max_sequences_per_dp=8,
    )
    assert col_row["feasible"] is True
    assert col_row["sustainable_tps_per_die"] == pytest.approx(1e6 / col_row["us_per_tok_die"], rel=1e-9)
    assert col_row["decode_tps_per_die"] == pytest.approx(col_row["sustainable_tps_per_die"], rel=1e-9)
    assert 0.0 < col_row["e2e_tps_per_die"] < col_row["decode_tps_per_die"]

    # 2. Pinned and enumerated illegal DCP configurations are rejected before feasible is computed.
    sfa_spec = load_model("deepseek-v3.2")
    gqa_spec = load_model("qwen3-32b")
    # PCP-only SFA (dcp=1) is valid, while dcp=4 violates SFA replicated-indexer {1, pcp, pcp*tp}.
    assert pd_mod._build_colocated_candidates(
        num_devices=8, is_moe=True, layers=4, rec_topo=None,
        pinned={"tp": 4, "pcp": 2, "dcp": 1}, model_spec=sfa_spec, prompt_tokens=4096,
    ) != []
    assert pd_mod._build_colocated_candidates(
        num_devices=8, is_moe=True, layers=4, rec_topo=None,
        pinned={"tp": 4, "pcp": 2, "dcp": 4}, model_spec=sfa_spec, prompt_tokens=4096,
    ) == []
    assert pd_mod._build_pd_stage_candidates(
        role="pd_decode", max_dies=8, is_moe=False, layers=4, prompt_tokens=4096,
        rec_topo=None, pinned={"tp": 4, "pcp": 2, "dcp": 4}, model_spec=gqa_spec,
    ) == []
    assert pd_mod._build_pd_stage_candidates(
        role="pd_prefill", max_dies=8, is_moe=True, layers=4, prompt_tokens=4096,
        rec_topo=None, pinned={"tp": 2, "pcp": 2, "dcp": 1}, model_spec=sfa_spec,
    ) != []
    bad_dcp_case = compose_case("deepseek-v3.2", "a3", tp=4, ep=4, dcp=2, layers=4, prompt_tokens=256, output_tokens=64)
    p_sfa_case = compose_case("deepseek-v3.2", "a3", tp=4, ep=4, layers=4, prompt_tokens=256, output_tokens=64)
    bad_col = pd_mod._evaluate_colocated_candidate(
        bad_dcp_case, prompt_tokens=256, output_tokens=64,
        ttft_sla_ms=5000.0, tpot_sla_ms=50.0, mtp_tokens=0,
    )
    assert bad_col["feasible"] is False and bad_col["dcp_codomain_valid"] is False
    assert bad_col["infeasible_reason"] is not None and "vllm_ascend_sfa_dcp_must_equal_tp" in bad_col["infeasible_reason"]
    bad_pd = sweep_pd_disaggregation_ratios(
        p_sfa_case, bad_dcp_case, total_dies=8, prompt_tokens=256, output_tokens=64,
        ttft_sla_ms=5000.0, tpot_sla_ms=50.0,
    )
    assert bad_pd["decode_instance"]["dcp_codomain_valid"] is False
    assert all(
        not r["feasible"] and r["infeasible_reason"] is not None
        and "vllm_ascend_sfa_dcp_must_equal_tp" in r["infeasible_reason"]
        for r in bad_pd["candidates"]
    )

    # 3. Binary-search _solve_b_vram_max avoids 99 -> 1 collapse and matches Sheet 3 max_hbm_concurrency_per_dp.
    parsed_col = parse_case(col_raw)
    fake_kv = 1000.0

    class _LinearWorkspaceMemory:
        def __init__(self, _parsed, *, workspace_tokens=1):
            ws = max(1, int(workspace_tokens))
            # At ws=1: 99*kv free (upper=99); at ws=99: 0 free; exact crossing at b=50 (100 - 50 >= 50).
            self._free = max(0.0, (100.0 - float(ws)) * fake_kv)
            self.capacity = [[self._free]]

        def dp_cohort_cache_bytes(self, _seq_len, padded=True):
            return [fake_kv]

        def resident_memory_receipt(self, *, sequence_tokens, batch_per_dp, query_tokens=1):
            from types import SimpleNamespace
            headroom = int(self._free - int(batch_per_dp) * fake_kv)
            return SimpleNamespace(dies=(SimpleNamespace(free_hbm_headroom_bytes=headroom),))

    with monkeypatch.context() as mp:
        mp.setattr(pd_mod, "Memory", _LinearWorkspaceMemory)
        _, _, b_exact = pd_mod._solve_b_vram_max(
            parsed_col, base_ws=1, decode_q=1, seq_len=576, cap_bound=256,
        )
    assert b_exact == 50

    dec_case = compose_case(
        "qwen3-32b", "a3", tp=2, layers=2, role="pd_decode",
        prompt_tokens=8192, output_tokens=2048, max_num_seqs=4096,
    )
    parsed_dec = parse_case(dec_case)
    _, _, sweep_b_vram = pd_mod._solve_b_vram_max(
        parsed_dec, base_ws=1, decode_q=1, seq_len=10240, cap_bound=int(parsed_dec.runtime.max_num_seqs),
    )
    s3_bundle = build_export_bundle(
        dec_case, request=StepRequest(phase="decode", tokens=1, context=8192, sequences=1, scope="model"),
        apply_target_envelope=False,
    )
    assert sweep_b_vram == s3_bundle.memory_ledger.max_hbm_concurrency_per_dp

    # 4. After Iso-SLA search evaluates target graph even when compare_track == target_track and unlocks larger B1*.
    orig_target_step = pd_mod._solve_target_step_ms
    target_calls: list[int] = []

    def _accelerated_target(*args, **kwargs):
        target_calls.append(1)
        return orig_target_step(*args, **kwargs) * 0.5

    probe_before = solve_deployment(
        "qwen3-32b", "a3", num_devices=2, layers=2, prompt_tokens=512, output_tokens=64,
        tpot_slo_ms=50.0, deployment="colocated", pinned_parallel={"tp": 2},
        max_sequences_per_dp=16, mode="simulation", compare_track="shipping", target_track="shipping",
    )
    b1_ms = probe_before["iso_sla_comparison"]["before"]["decode_step_ms"]
    tight_slo = max(0.05, b1_ms * 1.15)
    with monkeypatch.context() as mp:
        mp.setattr(pd_mod, "_solve_target_step_ms", _accelerated_target)
        sol_same_track = solve_deployment(
            "qwen3-32b", "a3", num_devices=2, layers=2, prompt_tokens=512, output_tokens=64,
            tpot_slo_ms=tight_slo, deployment="colocated", pinned_parallel={"tp": 2},
            max_sequences_per_dp=64, mode="modeling", compare_track="shipping", target_track="shipping",
        )
    assert len(target_calls) >= 1
    iso = sol_same_track["iso_sla_comparison"]
    assert iso["before"]["feasible"] is True and iso["after"]["feasible"] is True
    assert iso["after"]["b_feasible"] >= iso["before"]["b_feasible"]
    assert iso["after"]["sustainable_tps_per_die"] > iso["before"]["sustainable_tps_per_die"]


def test_explicit_rewrite_plan_no_clamping_unrounded_rank_and_prefill_export(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from mindie_sim.schema.common import ModelingError
    from mindie_sim.sweep import pd as pd_mod
    from mindie_sim.solve import solve_deployment

    # 1. Unrounded decode TPS ranking & zero E2E fallback when decode_step_ms > 0.
    r_lo = {"feasible": True, "decode_tps_per_die": 100.0000001, "used_dies": 2, "e2e_tps_per_die": 90.0, "effective_tpot_ms": 10.0}
    r_hi = {"feasible": True, "decode_tps_per_die": 100.0000004, "used_dies": 4, "e2e_tps_per_die": 90.0, "effective_tpot_ms": 10.0}
    assert pd_mod._single_die_decode_tps(r_hi) > pd_mod._single_die_decode_tps(r_lo)
    r_zero_dec = {"feasible": True, "decode_tps_per_die": 0.0, "us_per_tok_die": 0.0, "decode_step_ms": 5.0, "sustainable_tps_per_die": 88.0}
    assert pd_mod._single_die_decode_tps(r_zero_dec) == 0.0

    # 2. Typed draft metadata classification never misclassifies target ops containing '.draft.' in op_id.
    target_op = SimpleNamespace(op_id="target.l0.draft.proj", metadata=(("layer_scope", "target"),))
    draft_op = SimpleNamespace(op_id="mtp.step0.l0.proj", metadata=(("layer_scope", "draft"),))
    assert pd_mod._is_draft_op(target_op) is False
    assert pd_mod._is_draft_op(draft_op) is True

    # 3. Explicit rewrite_plan preserves regressions (no min(baseline, solved) clamp) and no Before fallback.
    dummy_plan = SimpleNamespace(edits=(SimpleNamespace(operation=None),))
    with monkeypatch.context() as mp:
        mp.setattr(pd_mod, "load_rewrite_plan", lambda _p: dummy_plan)
        mp.setattr(pd_mod, "_solve_target_step_ms", lambda *_a, **_kw: 999.0)
        res_reg = solve_deployment(
            "qwen3-32b", "a3", num_devices=2, layers=2, prompt_tokens=512, output_tokens=32,
            tpot_slo_ms=50.0, deployment="colocated", pinned_parallel={"tp": 2},
            max_sequences_per_dp=4, mode="simulation", rewrite_plan={"edits": [{"id": "reg"}]},
        )
    assert res_reg["iso_sla_comparison"]["before"]["feasible"] is True
    assert res_reg["iso_sla_comparison"]["after"]["feasible"] is False
    assert res_reg["feasible"] is False

    # 4. Strict rewrite-plan binding errors propagate out of solver instead of being swallowed.
    def _raise_unbound(*_a, **_kw):
        raise ModelingError("rewrite_fusion_recipe_unbound:custom:FusedOp")

    with monkeypatch.context() as mp:
        mp.setattr(pd_mod, "load_rewrite_plan", lambda _p: dummy_plan)
        mp.setattr(pd_mod, "_solve_target_step_ms", _raise_unbound)
        with pytest.raises(ModelingError, match="rewrite_fusion_recipe_unbound"):
            solve_deployment(
                "qwen3-32b", "a3", num_devices=2, layers=2, prompt_tokens=512, output_tokens=32,
                tpot_slo_ms=50.0, deployment="colocated", pinned_parallel={"tp": 2},
                max_sequences_per_dp=4, mode="simulation", rewrite_plan={"edits": [{"id": "fail"}]},
            )

    # 5. Prefill-only workload (output_tokens == 1) exports phase='prefill'.
    xlsx_prefill = tmp_path / "prefill_only.xlsx"
    res_prefill = solve_deployment(
        "qwen3-32b", "a3", num_devices=2, layers=2, prompt_tokens=512, output_tokens=1,
        tpot_slo_ms=50.0, ttft_slo_ms=5000.0, deployment="colocated", pinned_parallel={"tp": 2},
        mode="simulation", xlsx_path=xlsx_prefill,
    )
    assert res_prefill["feasible"] is True
    assert res_prefill["exports"]["phase"] == "prefill"
    assert res_prefill["exports"]["xlsx"]["phase"] == "prefill"


def test_iso_sla_colocated_prefill_token_cap_exact_cache_b1_and_prefill_dcp(monkeypatch):
    import math
    from mindie_sim.catalog.context_parallel import validate_vllm_ascend_dcp_codomain
    from mindie_sim.solve import _summarize_iso_sla_state
    from mindie_sim.sweep import pd as pd_mod

    # 1. Unconstrained TTFT still prices colocated finalist prefill so TTFT & e2e_total_tps stay physical.
    col_raw = compose_case("qwen3-32b", "a3", tp=2, layers=2, prompt_tokens=256, output_tokens=8)
    row = pd_mod._evaluate_colocated_candidate(
        col_raw, prompt_tokens=256, output_tokens=8,
        ttft_sla_ms=math.inf, tpot_sla_ms=100.0, mtp_tokens=0, max_sequences_per_dp=2,
    )
    assert row["feasible"] is True
    assert row["ttft_ms"] > 0.0
    assert row["e2e_total_tps"] > 0.0 and row["e2e_total_tps"] < row["sustainable_tps"]
    iso_state = _summarize_iso_sla_state(
        {"feasible": True, "winner_mode": "colocated", "best_configuration": row},
        track="merged", mode_label="before", num_devices=2,
    )
    assert iso_state["cluster_tps_per_purchased_die"] == pytest.approx(row["e2e_total_tps"] / 2.0, rel=1e-12)

    # 2. Decode SLA search caps batch by max_num_batched_tokens // q_tok while keeping b_vram_max physical,
    #    and returns exact step_ms_at_b1 even when the top batch fits on the first probe.
    cap_raw = compose_case(
        "deepseek-v3.2", "a5", tp=8, ep=8, quant="mxfp4", prompt_tokens=256, output_tokens=16,
        mtp_proposal_tokens=3, mtp_acceptance_length=2.5, max_num_seqs=64, max_num_batched_tokens=16,
    )
    parsed_cap = parse_case(cap_raw)
    seen_seqs: list[int] = []

    def _linear_step(_parsed, *, sequences, **_kw):
        seen_seqs.append(int(sequences))
        return 2.0 + 0.5 * float(sequences)

    with monkeypatch.context() as mp:
        mp.setattr(pd_mod, "_archetype_weighted_step_ms", _linear_step)
        sla_res = pd_mod._solve_sla_archetype_aware(
            parsed_cap, sla_ms=50.0, context=256, output_budget=16, max_sequences_per_dp=64,
        )
    scen = sla_res["scenarios"][0]
    assert scen["b_vram_max"] > 4  # physical HBM ceiling preserved
    assert scen["b_feasible"] == 4  # 16 batched tokens // (1 + 3) query tokens
    assert scen["step_ms_feasible"] == pytest.approx(4.0, rel=1e-9)
    assert scen["step_ms_at_b1"] == pytest.approx(2.5, rel=1e-9)
    assert 1 in seen_seqs and all(s <= 4 for s in seen_seqs)

    # 3. PD SLA cache keys use exact eff_sla_ms so tight/loose sub-0.01 ms budgets never collide.
    cache: dict[tuple[int, float], dict[str, object]] = {}
    d_parsed = parse_case(col_raw)

    def _step_near_10(_parsed, *, sequences, **_kw):
        return 10.003 if int(sequences) >= 2 else 9.0

    with monkeypatch.context() as mp:
        mp.setattr(pd_mod, "_archetype_weighted_step_ms", _step_near_10)
        looser = pd_mod._measure_decode_stage(
            d_parsed, prompt=256, output=101, eff_decode=256,
            tpot_sla_ms=10.004, exposed_kv_ms=0.0, decode_tokens=100,
            max_sequences_per_dp=2, cache=cache, cache_id=7,
        )
        tighter = pd_mod._measure_decode_stage(
            d_parsed, prompt=256, output=101, eff_decode=256,
            tpot_sla_ms=10.001, exposed_kv_ms=0.0, decode_tokens=100,
            max_sequences_per_dp=2, cache=cache, cache_id=7,
        )
    assert looser["tpot_ok"] is True and looser["b_feasible"] == 2
    assert tighter["tpot_ok"] is True and tighter["b_feasible"] == 1
    assert len(cache) == 2

    # 4. Prefill role enforces validate_vllm_ascend_dcp_codomain in candidate builder and fixed-case PD sweep.
    sfa_case = compose_case("deepseek-v3.2", "a3", tp=4, ep=4, layers=4, prompt_tokens=256, output_tokens=32)
    sfa_spec = parse_case(sfa_case).model
    assert pd_mod._build_pd_stage_candidates(
        role="pd_prefill", max_dies=8, is_moe=True, layers=4, prompt_tokens=256,
        rec_topo=None, pinned={"tp": 4, "pcp": 1, "dcp": 2}, model_spec=sfa_spec,
    ) == []
    bad_p = compose_case("deepseek-v3.2", "a3", tp=4, ep=4, dcp=2, layers=4, prompt_tokens=256, output_tokens=32)
    assert validate_vllm_ascend_dcp_codomain(sfa_spec, parse_case(bad_p).runtime.parallel)[0] is False
    bad_p_rep = sweep_pd_disaggregation_ratios(
        bad_p, sfa_case, total_dies=8, prompt_tokens=256, output_tokens=32,
        ttft_sla_ms=5000.0, tpot_sla_ms=50.0,
    )
    assert bad_p_rep["prefill_instance"]["dcp_codomain_valid"] is False
    assert bad_p_rep["best_configuration"] is None
    assert all(
        not r["feasible"] and str(r["infeasible_reason"]).startswith("prefill:")
        for r in bad_p_rep["candidates"]
    )

    # 5. PD pool summary carries prefill_kv_rps and e2e_total_tps from the winning ratio.
    sol_pd = solve_optimal_deployment(
        "qwen3-32b", "a3", num_devices=4, layers=2, prompt_tokens=256, output_tokens=32,
        ttft_sla_ms=5000.0, tpot_sla_ms=50.0, deployment="pd_disaggregated",
        pinned_prefill_parallel={"tp": 2}, pinned_decode_parallel={"tp": 2},
        max_sequences_per_dp=4,
    )
    assert sol_pd["feasible"] is True and isinstance(sol_pd["pd_pool_summary"], dict)
    pool = sol_pd["pd_pool_summary"]
    assert pool["e2e_total_tps"] == pytest.approx(sol_pd["best_configuration"]["e2e_total_tps"], rel=1e-9)
    assert pool["prefill_kv_rps"] > 0.0


def test_solver_full_grid_admission_explicit_mode_overlap_and_early_abort(monkeypatch):
    """Lock full-grid 32-die admission >16, explicit modes, overlap/K-rule, and zero-Engine OOM abort."""
    from mindie_sim.model_catalog import load_model
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.solve import solve_deployment
    from mindie_sim.sweep import pd as pd_mod

    # 1. Explicit deployment mode rejects omitted / "auto" and enforces role SLA boundaries.
    for bad_mode in (None, "auto", "pd"):
        with pytest.raises(ModelingError, match=r"deployment_mode_(?:required|invalid)"):
            solve_optimal_deployment(
                "qwen3-32b", "a5", num_devices=4, layers=1,
                prompt_tokens=256, output_tokens=32, ttft_sla_ms=5000.0, tpot_sla_ms=50.0,
                deployment=bad_mode,
            )
        with pytest.raises(ModelingError, match=r"deployment_mode_(?:required|invalid)"):
            solve_deployment(
                "qwen3-32b", "a5", num_devices=4, layers=1,
                prompt_tokens=256, output_tokens=32, tpot_slo_ms=50.0,
                deployment=bad_mode,
            )
    with pytest.raises(ModelingError, match="pd_decode_ttft_unavailable"):
        solve_deployment(
            "qwen3-32b", "a5", num_devices=4, layers=1,
            prompt_tokens=256, output_tokens=32, ttft_slo_ms=5000.0, tpot_slo_ms=50.0,
            deployment_mode="pd_decode",
        )
    with pytest.raises(ModelingError, match="pd_prefill_rejects_decode_sla"):
        solve_deployment(
            "qwen3-32b", "a5", num_devices=4, layers=1,
            prompt_tokens=256, output_tokens=32, ttft_slo_ms=5000.0, tpot_slo_ms=50.0,
            deployment_mode="pd_prefill",
        )
    pf_sol = solve_deployment(
        "qwen3-32b", "a5", num_devices=2, layers=1,
        prompt_tokens=256, output_tokens=32, ttft_slo_ms=5000.0,
        deployment_mode="pd_prefill", mode="simulation",
    )
    assert pf_sol["feasible"] is True
    assert pf_sol["deployment_mode"] == "pd_prefill"
    assert pf_sol["deployment_outcome"]["verified_tpot_ms"] is None
    assert pf_sol["recommended_deployment"]["best_configuration"]["effective_tpot_ms"] is None

    # 2. Full-grid candidate generation retains TP=4/8/16/32 and all local projection axes.
    kimi_spec = load_model("kimi-k3.1")
    grid = pd_mod._build_colocated_candidates(
        num_devices=32, is_moe=True, layers=kimi_spec.layers, rec_topo=None,
        pinned={}, model_spec=kimi_spec, prompt_tokens=131072, role="pd_decode",
    )
    full_budget_tps = {
        (c["dp"], c["tp"], c["ep"], c["dcp"])
        for c in grid if c["pp"] == 1 and c["dp"] * c["tp"] == 32
    }
    assert {(8, 4, 32, 4), (4, 8, 32, 8), (2, 16, 32, 16), (1, 32, 32, 32)}.issubset(full_budget_tps)
    k_a = pd_mod._candidate_dedup_key({"dp": 4, "tp": 8, "ep": 32, "pp": 1, "pcp": 1, "dcp": 8, "q_proj_tp": 4})
    k_b = pd_mod._candidate_dedup_key({"dp": 4, "tp": 8, "ep": 32, "pp": 1, "pcp": 1, "dcp": 8, "q_proj_tp": 8})
    assert k_a != k_b

    # 3. Physical admission unlocks >16 on 32-die 128K Decode (shipping K+1 snapshot ring: DP4xTP8xDCP8=29,
    #    DP2xTP16xDCP16=63; opt-in recover_ssm=True: DP4xTP8xDCP8 in 43..55, DP2xTP16xDCP16 in 100..120)
    #    and never mislabels a configured max_num_seqs cap as HBM_CAPACITY_BOUND.
    for dp, tp, dcp, lo_ship, hi_ship, lo_rec, hi_rec in (
        (4, 8, 8, 25, 35, 43, 55),
        (2, 16, 16, 55, 70, 100, 120),
    ):
        base_d = dict(
            dp=dp, tp=tp, ep=32, dcp=dcp, role="pd_decode",
            quant="mxfp8_mxfp4", prompt_tokens=131072, output_tokens=1024,
            mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
            kv_cache_dtype="fp8", indexer_kv_dtype="fp8",
            tpot_slo_ms=30.0,
        )
        raw_ship = compose_case("kimi-k3.1", "950dt-96g-pod", **base_d)
        raw_rec = compose_case("kimi-k3.1", "950dt-96g-pod", recover_ssm=True, **base_d)
        _, _, b_ship = pd_mod._solve_b_vram_max(
            parse_case(raw_ship), base_ws=6, decode_q=6, seq_len=131072 + 1024, cap_bound=None,
        )
        _, _, b_rec = pd_mod._solve_b_vram_max(
            parse_case(raw_rec), base_ws=6, decode_q=6, seq_len=131072 + 1024, cap_bound=None,
        )
        assert lo_ship <= b_ship <= hi_ship, f"DP{dp}xTP{tp}xDCP{dcp} shipping b_vram={b_ship} outside [{lo_ship}, {hi_ship}]"
        assert lo_rec <= b_rec <= hi_rec, f"DP{dp}xTP{tp}xDCP{dcp} recover_ssm b_vram={b_rec} outside [{lo_rec}, {hi_rec}]"
        assert b_rec > b_ship > 16
    regime_cap, lim_cap = pd_mod._classify_decode_regime(
        feasible=True, b_feasible=32, b_vram=49, configured_max_seqs=32,
        token_batch_limit=128, unused_sla_slack_ms=10.0,
    )
    assert regime_cap == pd_mod.REGIME_CONFIGURED_SEQ_CAP_BOUND and lim_cap == "runtime_max_seqs"

    # 4. PD layerwise overlap obeys causal publish/consume (<= transfer_ms) and honors P/D speculative K-rule.
    hold_1, exp_1 = pd_mod._compute_layerwise_overlap_ms(
        1.0, 10.0, per_layer_bytes={0: 1024.0}, num_layers=1,
    )
    assert hold_1 == pytest.approx(11.0) and exp_1 == pytest.approx(10.0) and exp_1 <= 10.0
    hold_4, exp_4 = pd_mod._compute_layerwise_overlap_ms(
        10.0, 10.0, per_layer_bytes={0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0}, num_layers=4,
    )
    assert hold_4 == pytest.approx(12.5) and 0.0 < exp_4 < 10.0
    dsr1_spec = load_model("deepseek-r1")
    assert pd_mod._resolve_pd_prefill_speculative_k("kimi-k3.1", kimi_spec, decode_mtp_tokens=5, hw_family="a5") == 5
    assert pd_mod._resolve_pd_prefill_speculative_k("deepseek-r1", dsr1_spec, decode_mtp_tokens=3, hw_family="a3") == 1
    assert pd_mod._resolve_pd_prefill_speculative_k("kimi-k3.1", kimi_spec, decode_mtp_tokens=0, hw_family="a5") == 0

    # 5. Static-infeasible early abort: zero Engine calls when all joint-PD pools fail static HBM admission.
    engine_calls: list[int] = []

    def _forbid_engine(*_args, **_kwargs):
        engine_calls.append(1)
        raise AssertionError("Engine must not be invoked when static HBM admission is 0")

    with monkeypatch.context() as mp:
        mp.setattr(pd_mod, "_archetype_weighted_step_ms", _forbid_engine)
        oom_pd = solve_optimal_deployment(
            "kimi-k3.1", "950dt-96g-pod",
            num_devices=32, prompt_tokens=131072, output_tokens=1024,
            ttft_sla_ms=5000.0, tpot_sla_ms=30.0,
            deployment_mode="pd_disaggregated",
            mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
            kv_cache_dtype="fp8", indexer_kv_dtype="fp8",
        )
    assert engine_calls == []
    assert oom_pd["feasible"] is False
    assert oom_pd["recommended_case"] is None
    assert oom_pd["deployment_outcome"]["status"] == "infeasible"
    assert oom_pd["deployment_outcome"]["feasible_b_per_dp"] == 0
    assert oom_pd["deployment_outcome"]["has_winner_case"] is False


def test_disaggregated_handoff_dcp_demand_host_only_receiver_and_explicit_owners():
    """DCP decode HBM demand charges per-rank max, host-only receivers skip HBM hold, and explicit owners bypass TP filter."""
    from _support.memory_cases import sparse_mla_case
    from _support.serving_cases import bind_pd_pool, linear_work, pd_model_case
    from mindie_sim.prediction.disaggregated.handoff import compute_pd_handoff_plan
    from mindie_sim.prediction.disaggregated.serving import predict_pd

    # 1. Idle DCP sibling die must not halve actual receiver HBM bandwidth charge or transfer elapsed time.
    for tp in (1, 2):
        raw = pd_model_case(8, 1)
        bind_pd_pool(raw, "decode", tuple(range(1, tp + 1)), tp, tp)
        decode_pool = raw["pools"]["decode"]
        decode_pool["runtime"]["kv_block_size"] = 128
        decode_pool["machine"]["device"]["rates"]["memory"] = 64_000
        raw["connector"]["link"]["capacity_bytes_per_ms"] = 1e9
        result = predict_pd(raw, lower=linear_work)
        handoff = result["execution"]["pd_handoffs"][0]
        start_ms = result["requests"][0]["first_token_abs_ms"]
        assert handoff["receive_capacity_bytes"] == {1: 512}
        assert handoff["memory_work"]["decode:memory@r0:s0"] == pytest.approx(512.0)
        assert handoff["complete_ms"] - start_ms >= 8.0 - 1e-6

    # 2. Sparse-KV host-only destination rank completes cleanly without raising missing HBM range errors.
    p_case = sparse_mla_case(context_tokens=8)
    d_case = sparse_mla_case(
        context_tokens=8,
        tp=2,
        offloading={
            "implementation": "vllm_ascend_sparse_kv_offload",
            "options": {"use_fused_copy_sfa": False, "dram_size_per_dp_gb": 64.0},
        },
    )
    d_case["runtime"].update(dcp=2, metadata={"enable_sfa_dcp_sharded_indexer": True})
    plan = compute_pd_handoff_plan(p_case, d_case, tokens=8)
    assert plan["destination_hbm_rank_bytes"] == {1: 1040}
    assert plan["destination_host_rank_bytes"] == {1: 9216, 2: 9216}
    pools = {role: {k: c[k] for k in ("machine", "runtime")} for role, c in (("prefill", p_case), ("decode", d_case))}
    served = predict_pd(dict(p_case, pools=pools, connector={"bytes_per_s": 50e9}))
    assert served["status"] == "modeled"

    # 3. Explicit rank_token_owners overrides TP-lane replication filtering when DCP == 1,
    #    handles tail growth beyond the last span, and bills prefill unicast fan-out HBM reads.
    raw_own = pd_model_case(8, 1)
    bind_pd_pool(raw_own, "prefill", (0, 1), 2, 1)
    raw_own["pools"]["prefill"]["runtime"]["metadata"] = {"rank_token_owners": [(0, 4), (4, 8)]}
    p_own, d_own = [
        parse_case(dict(model=raw_own["model"], workload=raw_own["workload"], **pool))
        for pool in raw_own["pools"].values()
    ]
    assert compute_pd_handoff_plan(p_own, d_own, tokens=8)["source_rank_bytes"] == {0: 256, 1: 256}
    assert compute_pd_handoff_plan(p_own, d_own, tokens=9)["source_rank_bytes"] == {0: 256, 1: 320}

    raw_fanout = pd_model_case(8, 1)
    bind_pd_pool(raw_fanout, "prefill", (0, 1), 2, 1)
    bind_pd_pool(raw_fanout, "decode", (2, 3), 2, 1)
    raw_fanout["pools"]["prefill"]["runtime"]["metadata"] = {"rank_token_owners": [(0, 4), (4, 8)]}
    fanout_res = predict_pd(raw_fanout, lower=linear_work)
    assert fanout_res["execution"]["pd_handoffs"][0]["memory_work"]["prefill:memory@r0:s0"] == pytest.approx(512.0)

    # 4. Non-inherited MLA draft replica grouping covers cross-DP DCP shards (tp=2, dp=dcp=3)
    #    and inherited draft handoff respects draft.dcp_sharded=False.
    from dataclasses import replace as _replace
    from _support.memory_cases import case as _mem_case
    from mindie_sim.catalog.draft_spec import DraftSpec
    from mindie_sim.catalog.types import AuxiliarySpec

    mla_draft = DraftSpec(
        kind="mtp", layers=1, hidden_size=64, intermediate_size=128,
        num_heads=4, num_kv_heads=1, head_dim=16, q_width=64, kv_width=16,
        attention_kind="mla", q_lora_rank=64, kv_lora_rank=512,
        qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=64, dcp_sharded=True,
    )
    raw_mp = _mem_case(dp=3, tp=2, pp=1)
    raw_mp["model"]["num_kv_heads"] = 1
    raw_mp["machine"] = dict(backend="ascend", sku="Ascend910B4", num_devices=6)
    raw_mp["runtime"].update(dcp=3, kv_block_size=1, metadata={"spec_method": "mtp", "mtp_proposal_tokens": 1, "cp_kv_cache_interleave_size": 1})
    raw_md = _mem_case(dp=1, tp=1, pp=1)
    raw_md["model"]["num_kv_heads"] = 1
    raw_md["machine"] = dict(backend="ascend", sku="Ascend910B4", num_devices=1)
    raw_md["runtime"].update(dcp=1, kv_block_size=1, metadata={"spec_method": "mtp", "mtp_proposal_tokens": 1, "cp_kv_cache_interleave_size": 1})
    p_mla = _replace(parse_case(raw_mp), model=_replace(parse_case(raw_mp).model, auxiliary=AuxiliarySpec(mtp=mla_draft)))
    d_mla = _replace(parse_case(raw_md), model=_replace(parse_case(raw_md).model, auxiliary=AuxiliarySpec(mtp=mla_draft)))
    assert {0, 2, 4}.issubset(set(compute_pd_handoff_plan(p_mla, d_mla, tokens=3)["source_rank_bytes"]))

    raw_ns = compose_case("glm-5.3", "a3", layers=1, tp=2, ep=1, dcp=2, block_size=1, mtp_proposal_tokens=1)
    raw_ns["runtime"].setdefault("metadata", {})["cp_kv_cache_interleave_size"] = 1
    p_sh = parse_case(raw_ns)
    p_ns = _replace(p_sh, model=_replace(p_sh.model, auxiliary=_replace(p_sh.model.auxiliary, mtp=_replace(p_sh.model.auxiliary.mtp, dcp_sharded=False))))
    h_ns = compute_pd_handoff_plan(p_ns, p_ns, tokens=1)
    h_sh = compute_pd_handoff_plan(p_sh, p_sh, tokens=1)
    assert sum(h_ns["destination_hbm_rank_bytes"].values()) > sum(h_sh["destination_hbm_rank_bytes"].values())


def test_pd_shared_l2_namespace_dcp_replica_scope_and_recoverssm_stride():
    """Lock SFA PD L2 namespace isolation, DCP-DP per-rank replica tagging, and RecoverSSM buffer stride."""
    from dataclasses import replace as _replace
    from types import SimpleNamespace
    from mindie_sim.catalog.speculative_state import MODE_REPLAY, speculative_state_plan
    from mindie_sim.prediction.disaggregated.handoff import compute_pd_handoff_plan
    from mindie_sim.prediction.operators.sharedkv.sfa_native import _resolve_engine_memory_view
    from mindie_sim.prediction.pd_connector.connector import _transfer_scope
    from mindie_sim.prediction.runtime.engine import Engine
    from mindie_sim.prediction.runtime.engine_types import Resource
    from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView
    from mindie_sim.schema.parallel import ParallelSpec

    # 1. Shared-engine PD: decode: die 2 selects its own namespace view instead of prefill: die 0 view.
    eng = Engine([Resource("prefill:hbm@r0:s0", 1.0), Resource("decode:hbm@r0:s0", 1.0)])
    prefill_view = AdmissionMemoryView({0: 64 * 1024 * 1024}, arch_id="npu_arch_3510", pool_namespace="prefill:")
    decode_view = AdmissionMemoryView({2: 64 * 1024 * 1024}, arch_id="npu_arch_3510", pool_namespace="decode:")
    eng.attach_admission_memory_view(prefill_view)
    ctx_dec = SimpleNamespace(admission_memory_view=decode_view, compute_arch=SimpleNamespace(arch_id="npu_arch_3510"))
    assert _resolve_engine_memory_view(eng, 2, 64 * 1024 * 1024, "decode:hbm@r0:s0", ctx_dec) is decode_view
    ctx_none = SimpleNamespace(admission_memory_view=None, compute_arch=SimpleNamespace(arch_id="npu_arch_3510"))
    isolated_dec = _resolve_engine_memory_view(eng, 2, 64 * 1024 * 1024, "decode:hbm@r0:s0", ctx_none)
    assert isolated_dec is not prefill_view and isolated_dec.pool_namespace == "decode:" and isolated_dec.has_die(2)
    # Preview / namespaced scratch engine keeps its attached scratch view without mutating live ctx_view.
    scratch_eng = Engine([Resource("hbm", 1.0)])
    scratch_view = AdmissionMemoryView({2: 64 * 1024 * 1024}, arch_id="npu_arch_3510", pool_namespace="decode:")
    scratch_eng.attach_admission_memory_view(scratch_view)
    assert _resolve_engine_memory_view(scratch_eng, 2, 64 * 1024 * 1024, "hbm", ctx_dec) is scratch_view
    ns_scratch_eng = Engine([Resource("decode:hbm@r0:s0", 1.0)])
    ns_scratch_eng.attach_admission_memory_view(scratch_view)
    assert _resolve_engine_memory_view(ns_scratch_eng, 2, 64 * 1024 * 1024, "decode:hbm@r0:s0", ctx_dec) is scratch_view

    # 2. DCP-DP handoff tags each physical rank with its owning DP replica (even with published component scope).
    from mindie_sim.prediction.disaggregated.handoff import _qualify_published_scope
    from mindie_sim.prediction.pd_connector import PDTransfer
    from mindie_sim.prediction.pd_connector.connector import _matching_transfer_indices

    p_dcp = parse_case(compose_case("deepseek-v3.2", "a5", tp=1, dp=4, dcp=4, ep=1, layers=1, prompt_tokens=128, output_tokens=8))
    d_dcp = parse_case(compose_case("deepseek-v3.2", "a5", tp=1, dp=4, dcp=4, ep=1, layers=1, prompt_tokens=128, output_tokens=8))
    plan_dcp = compute_pd_handoff_plan(p_dcp, d_dcp, tokens=128)
    assert plan_dcp["transfers"]
    src_reps = {_transfer_scope(t, "source")[0] for t in plan_dcp["transfers"]}
    dst_reps = {_transfer_scope(t, "destination")[0] for t in plan_dcp["transfers"]}
    assert src_reps == {0, 1, 2, 3} and dst_reps == {0, 1, 2, 3}
    for t in plan_dcp["transfers"]:
        assert f"prefill:r{t.source_rank}:s0:rank{t.source_rank}" == t.source_scope
        dst_owner = t.destination_rank - plan_dcp["destination_rank_offset"]
        assert f"decode:r{dst_owner}:s0:rank{t.destination_rank}" == t.destination_scope
        assert f":prefill:r{t.source_rank}:rank{t.source_rank}" in t.source_storage
        assert f":decode:r{dst_owner}:rank{t.destination_rank}" in t.destination_storage
    pub_flows = [
        PDTransfer(
            r, r + 2, 64, component="mla_kv", producer="kv", source_storage="kv_pool",
            source_scope=_qualify_published_scope("hbm", multi_replica=True, rep=r, stage=0, rank=r),
        )
        for r in (0, 1)
    ]
    assert _matching_transfer_indices(pub_flows, "source", (1, 0, "hbm")) == [1]

    # 3. RecoverSSM admission allocates physical shared buffer stride max(num_spec + 1, q) + kernel override.
    c_kda = parse_case(compose_case("glm-5.3-flash", "a5", layers=1, tp=1, ep=1, mtp_proposal_tokens=3, recover_ssm=True))
    kda_layer = next(lp for lp in c_kda.model.layer_plans if lp.attention == "kda")
    sp_q2 = speculative_state_plan(c_kda.model, kda_layer, ParallelSpec(tp=1), query_tokens=2, recover_ssm=True, capacity_padded=False, runtime=c_kda.runtime)
    sp_q4 = speculative_state_plan(c_kda.model, kda_layer, ParallelSpec(tp=1), query_tokens=4, recover_ssm=True, capacity_padded=False, runtime=c_kda.runtime)
    sp_q6 = speculative_state_plan(c_kda.model, kda_layer, ParallelSpec(tp=1), query_tokens=6, recover_ssm=True, capacity_padded=False, runtime=c_kda.runtime)
    assert sp_q2.mode == MODE_REPLAY and sp_q2.query_tokens == 2
    assert sp_q2.replay_bytes == pytest.approx(sp_q4.replay_bytes)
    assert sp_q6.replay_bytes == pytest.approx(sp_q4.replay_bytes * 1.5)
    rt_stride8 = _replace(c_kda.runtime, kernels={**dict(c_kda.runtime.kernels), "lowerings": {"recurrent_scan": {"options": {"record_stride": 8}}}})
    sp_s8 = speculative_state_plan(c_kda.model, kda_layer, ParallelSpec(tp=1), query_tokens=2, recover_ssm=True, capacity_padded=False, runtime=rt_stride8)
    assert sp_s8.replay_bytes == pytest.approx(sp_q4.replay_bytes * 2.0)
    rt_none = _replace(c_kda.runtime, metadata=tuple({**dict(c_kda.runtime.metadata), "spec_method": "none"}.items()))
    sp_none = speculative_state_plan(c_kda.model, kda_layer, ParallelSpec(tp=1), query_tokens=2, recover_ssm=True, capacity_padded=False, runtime=rt_none)
    assert sp_none.replay_bytes == pytest.approx(sp_q4.replay_bytes * 0.5)


def test_pd_prefill_prefix_cache_monotonically_reduces_ttft():
    """Standalone pd_prefill must honor prefix_tokens / prefix hit ratio and reduce TTFT monotonically."""
    from mindie_sim.sweep.pd import _evaluate_pd_prefill_candidate

    prompt_tokens = 2048
    output_tokens = 64
    ttft_sla_ms = 5000.0
    ratios = (0.0, 0.5, 0.9)
    rows = []
    for ratio in ratios:
        prefix_tokens = int(prompt_tokens * ratio)
        raw = compose_case(
            "qwen3-32b", "a3", dp=1, tp=2, ep=1, layers=2,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            prefix_tokens=prefix_tokens, role="pd_prefill", ttft_slo_ms=ttft_sla_ms,
        )
        rows.append(_evaluate_pd_prefill_candidate(
            raw, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            ttft_sla_ms=ttft_sla_ms, mtp_tokens=0,
        ))
    assert all(r["feasible"] for r in rows)
    assert rows[2]["ttft_ms"] < rows[1]["ttft_ms"] < rows[0]["ttft_ms"]
    assert rows[2]["prefill_rps_per_die"] > rows[1]["prefill_rps_per_die"] > rows[0]["prefill_rps_per_die"]

    # End-to-end solve_optimal_deployment honors both compose_overrides.prefix_tokens and pool_prefix_hit_ratio.
    common = dict(
        num_devices=2, prompt_tokens=prompt_tokens, output_tokens=output_tokens,
        ttft_sla_ms=ttft_sla_ms, deployment_mode="pd_prefill",
        layers=2, pinned_parallel={"dp": 1, "tp": 2, "ep": 1, "pp": 1},
        granted_workers=1,
    )
    sol_0 = solve_optimal_deployment("qwen3-32b", "a3", **common)
    sol_90_override = solve_optimal_deployment(
        "qwen3-32b", "a3", compose_overrides={"prefix_tokens": int(prompt_tokens * 0.9)}, **common,
    )
    sol_90_ratio = solve_optimal_deployment(
        "qwen3-32b", "a3", pool_prefix_hit_ratio=0.9, **common,
    )
    ttft_0 = float(sol_0["best_configuration"]["ttft_ms"])
    ttft_90_ov = float(sol_90_override["best_configuration"]["ttft_ms"])
    ttft_90_rat = float(sol_90_ratio["best_configuration"]["ttft_ms"])
    assert ttft_90_ov == pytest.approx(rows[2]["ttft_ms"], rel=1e-9)
    assert ttft_90_rat == pytest.approx(rows[2]["ttft_ms"], rel=1e-9)
    assert ttft_90_ov < ttft_0


