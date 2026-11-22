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
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.stack_tracks import resolve_runtime_stack
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
    with pytest.raises(ModelingError, match="missing_required_scenario_fields:num_devices,prompt_tokens,output_tokens,ttft_slo_ms,tpot_slo_ms"):
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
    for model_key in ("deepseek-v4.1-flash", "qwen3.8-flash-next"):
        defaults = stack.model_defaults[model_key]
        assert defaults["engram_cpu_offload"] is True
        assert defaults["multistream_overlap_shared_expert"] is True
        case = compose_case(model_key, "a5", tp=4, ep=4, layers=2, prompt_tokens=1024, output_tokens=64)
        rt = case["runtime"]
        meta = rt["metadata"]
        policy = meta["execution_policy"]
        assert rt["engram_cpu_offload"] is True
        assert rt["multistream_overlap_shared_expert"] is True
        assert rt["graph_capture_mode"] == "full"
        assert policy["engram_cpu_offload"] is True
        assert policy["engram_cpu_offload_source"] == "profile"
        assert policy["multistream_overlap_shared_expert"] is True
        assert policy["mtp_proposal_tokens"] > 0
        assert policy["mtp_source"] == "profile"

    flash_a3 = compose_case("qwen3.8-flash-next", "a3", tp=4, ep=4, layers=2, prompt_tokens=1024, output_tokens=64)
    assert flash_a3["runtime"]["metadata"]["enable_shared_expert_dp"] is True
    assert flash_a3["runtime"]["metadata"]["additional_config"]["enable_flashcomm1"] is True
    assert flash_a3["runtime"]["metadata"]["additional_config"]["enable_prefill_mc2"] is True

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
    solved = solve_optimal_deployment(
        "qwen3-32b", "a5",
        num_devices=8, layers=2,
        prompt_tokens=1024, output_tokens=128,
        ttft_sla_ms=5000.0, tpot_sla_ms=50.0,
        max_sequences_per_dp=16,
    )
    assert solved["schema"] == "mindie-sim.optimal-deployment/1"
    assert solved["feasible"] is True
    assert solved["winner_mode"] in {"pd_disaggregated", "colocated"}
    assert solved["best_pd"] is not None and solved["best_colocated"] is not None
    assert solved["best_configuration"]["sustainable_tps"] == pytest.approx(
        max(solved["best_pd"]["sustainable_tps"], solved["best_colocated"]["sustainable_tps"])
    )

    coloc_only = solve_optimal_deployment(
        "qwen3-32b", "a5",
        num_devices=4, layers=2,
        prompt_tokens=1024, output_tokens=64,
        ttft_sla_ms=5000.0, tpot_sla_ms=50.0,
        deployment="colocated",
        pinned_parallel={"tp": 2},
        max_sequences_per_dp=16,
    )
    assert coloc_only["winner_mode"] == "colocated"
    assert coloc_only["best_pd"] is None
    assert coloc_only["best_configuration"]["parallel"]["tp"] == 2

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
    )
    opt_meta = auto_case["runtime"]["metadata"]["optimal_deployment"]
    assert opt_meta["feasible"] is True
    assert opt_meta["winner_mode"] in {"pd_disaggregated", "colocated"}
    # Never silently falls back to naive 1x1x1x1 on a 4-die budget when larger topologies win.


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
    assert c_mxfp["model"]["quant"]["weight_format"] == "mxfp4"
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

    # One-shot solve_deployment via SDK & CLI (ttft_slo_ms=None allowed as unconstrained).
    sol_out = tmp_path / "solve.json"
    assert cli_main([
        "solve", "--model", "qwen3-32b", "--hardware", "950dt-96g-pod",
        "--num-devices", "4", "--layers", "2",
        "--prompt-tokens", "512", "--output-tokens", "64",
        "--tpot-slo-ms", "50.0", "--max-sequences-per-dp", "8",
        "-o", str(sol_out),
    ]) == 0
    sol = json.loads(sol_out.read_text(encoding="utf-8"))
    assert sol["schema"] == "mindie-sim.deployment-solution/1"
    assert sol["feasible"] is True
    assert sol["scenario"]["ttft_slo_ms"] is None
    assert sol["capacity_ledger"]["hbm_capacity_gib"] == pytest.approx(96.0)
    assert sol["recommended_deployment"]["winner_mode"] in {"pd_disaggregated", "colocated"}
    assert "current_merged" in sol["four_track_waterfall"]["waterfall"]
    assert sol["bottleneck_roofline"]["top_families"]
    assert sol["bottleneck_roofline"]["included_components"]["target_layers"] == 2
    sdk_sol = sdk_api.solve_deployment(
        "qwen3-32b", "950dt-96g-pod", num_devices=4, layers=2,
        prompt_tokens=512, output_tokens=64, tpot_slo_ms=50.0,
        kv_pooling="off", max_sequences_per_dp=8,
    )
    assert sdk_sol["feasible"] is True


def test_kimi_k31_five_live_correctness_regressions():
    from types import SimpleNamespace
    from mindie_sim.prediction.efficiency import analyze_network_efficiency, solve_sla_sizing
    from mindie_sim.prediction.operators.linear_attention.common import (
        _resolve_recover_ssm_accepted,
        validate_contract,
    )
    from mindie_sim.prediction.results import StepRequest
    from mindie_sim.schema.operators import OpSpec
    from mindie_sim.schema.workload import RequestView

    # Bug #1: PP=4 x PCP=8 x EP=8 (VA Kimi-K3.1 Prefill) must not IndexError in MC2 transport.
    prefill = compose_case(
        "kimi-k3.1", "950dt-96g-pod",
        dp=1, pp=4, pcp=8, tp=1, ep=8, layers=4, role="pd_prefill",
        prompt_tokens=1024, output_tokens=64,
    )
    p_eff = analyze_network_efficiency(
        prefill, phase="prefill", tokens=1024, context=0, sequences=1, scope="body",
    )
    assert p_eff["graph_duration_ms"] > 0.0

    # Bug #2: merged tutorial_flags enable_fused_mc2=0 must stay on merged only and never leak into expected_ideal.
    common = dict(
        dp=1, tp=8, dcp=8, ep=8, layers=4, role="pd_decode",
        prompt_tokens=131072, output_tokens=1024,
        mtp_proposal_tokens=5, mtp_acceptance_length=3.5,
    )
    c_merged = compose_case("kimi-k3.1", "950dt-96g-pod", feature_track="merged", **common)
    c_ideal = compose_case("kimi-k3.1", "950dt-96g-pod", feature_track="expected_ideal", **common)
    assert c_merged["runtime"]["metadata"]["additional_config"]["enable_fused_mc2"] == 0
    assert c_ideal["runtime"]["metadata"]["additional_config"].get("enable_fused_mc2", 0) != 0
    # #364 Item 1: kimi-k3 / kimi-k3.1 preserve native mxfp4_routed_only BF16 exclusions
    # even under quant="mxfp4"/"w4a8_mxfp"; explicit quant="mxfp4_all" overrides exclusions with MX scales.
    for q_override in (None, "mxfp4", "w4a8_mxfp", "mxfp4_routed_only"):
        c_ro = compose_case("kimi-k3.1", "950dt-96g-pod", quant=q_override, **common)
        assert c_ro["model"]["quant"]["name"] == "mxfp4_routed_only"
        assert {"attn", "mla", "kda", "shared_expert", "dense_mlp"} <= set(c_ro["model"]["quant"]["exclusions"])
    c_k3 = compose_case("kimi-k3", "950dt-96g-pod", tp=8, ep=8, layers=4)
    assert c_k3["model"]["quant"]["name"] == "mxfp4_routed_only"
    c_all = compose_case("kimi-k3.1", "950dt-96g-pod", quant="mxfp4_all", **common)
    assert c_all["model"]["quant"]["activation_group_size"] == 32
    assert c_all["model"]["quant"]["activation_scale_format"] == "fp8_e8m0"
    assert not c_all["model"]["quant"]["exclusions"]
    # #364 Item 2: expected_ideal fuses Target mla_dsa prolog into mla_prolog.native with index_q wired cleanly.
    from mindie_sim.prediction.operator_analysis import target_graphs
    from mindie_sim.prediction.structure import parse_case
    g_ideal = target_graphs(parse_case(c_ideal), phase="decode", tokens=6, context=131072, sequences=8, scope="model")[0]
    assert any(op.op_id == "s0.l3.mla_prolog.native" for op in g_ideal)
    idx_q_k31 = next(op for op in g_ideal if op.op_id == "s0.l3.index_q")
    assert "s0.l3.mla_prolog.native" in idx_q_k31.deps
    c_v32_ideal = compose_case("deepseek-v3.2", "a3", feature_track="expected_ideal", tp=8, ep=8, layers=4, prompt_tokens=4096, output_tokens=64)
    g_v32 = target_graphs(parse_case(c_v32_ideal), phase="decode", tokens=1, context=4096, sequences=1, scope="body")[0]
    assert any(op.op_id == "s0.l0.mla_prolog.native" for op in g_v32)
    assert "s0.l0.mla_prolog.native" in next(op for op in g_v32 if op.op_id == "s0.l0.index_q").deps
    e_merged = analyze_network_efficiency(c_merged, phase="decode", tokens=6, context=131072, sequences=8, scope="body")
    e_ideal = analyze_network_efficiency(c_ideal, phase="decode", tokens=6, context=131072, sequences=8, scope="body")
    assert e_ideal["graph_duration_ms"] < e_merged["graph_duration_ms"]

    # #365 Item 2: mindie-sim solve parses optional --xlsx / --trace / --compare-track / --target-track flags.
    from mindie_sim.cli.parser import build_parser
    parsed_solve = build_parser().parse_args([
        "solve", "--model", "kimi-k3.1", "--hardware", "950dt-96g-pod",
        "--num-devices", "8", "--prompt-tokens", "131072", "--output-tokens", "1024",
        "--tpot-slo-ms", "30", "--xlsx", "out.xlsx", "--trace", "trace_view.json",
    ])
    assert (parsed_solve.xlsx, parsed_solve.trace, parsed_solve.compare_track, parsed_solve.target_track) == (
        "out.xlsx", "trace_view.json", "unmerged_pr", "expected_ideal",
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
    assert _resolve_recover_ssm_accepted({"accepted_tokens": 3.5}, conv_ctx, 6) == 4
    assert _resolve_recover_ssm_accepted({}, conv_ctx, 6) == 4
    c_eager = compose_case("kimi-k3.1", "950dt-96g-pod", feature_track="raw_eager", quant="mxfp4", **common)
    step_eval = sdk_api.evaluate_step(c_eager, StepRequest(phase="decode", tokens=6, context=131584, sequences=8, scope="model"))
    assert step_eval.span_ms > 0.0
    c_unmerged = compose_case("kimi-k3.1", "950dt-96g-pod", feature_track="unmerged_pr", **common)
    step_unmerged = sdk_api.evaluate_step(c_unmerged, StepRequest(phase="decode", tokens=6, context=131584, sequences=8, scope="model"))
    assert step_unmerged.span_ms > 0.0
