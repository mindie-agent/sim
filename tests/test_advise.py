"""Tier 0/1 tests for causal advising, profile ingestion, and CLI wiring.

[Role]
  Verify the closed-world counterfactual advisor, four-dimensional attribution,
  calibration residuals, guardrail catalog, profile ingestion (Wave 5, C7,
  capture, delegated bridge), and ``advise``/``compose --from-profile`` CLI.

[Invariants]
  * All tests isolate ``MINDIE_*`` homes and ``$HOME`` under ``tmp_path``.
  * Predicted latency never changes when observed metadata is attached.
  * Every mutation candidate is a copy; caller records are unchanged.

[Timeouts & Locks]
  Pure in-memory / tmp_path tests; no NPU, network, subprocess, or locks.

[Side-Effect Policy]
  Writes only below pytest ``tmp_path``.
"""
from __future__ import annotations

import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest

from mindie_sim import advise_case, load_case_from_profile
from mindie_sim.__main__ import main
from mindie_sim.cli import main as cli_main
from mindie_sim.cli import parser, run
from mindie_sim.compose import compose_case

REAL_WAVE5 = Path(
    "/Volumes/data/mindie-sim-data/a2/heldout_engine_20261031/"
    "a2_real_dsv2lite_E64_K6_tp4ep4_decode_128k_c1"
)


@pytest.fixture(autouse=True)
def _isolated_homes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("MINDIE_CONFIG_HOME", "MINDIE_DATA_HOME", "MINDIE_DIAGNOSTICS_ROOT"):
        target = tmp_path / name.lower()
        target.mkdir()
        monkeypatch.setenv(name, str(target))
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


def _small_case(model: str = "qwen3.5-35b-a3b", hardware: str = "a3",
                layers: int = 2, prompt: int = 2048, output: int = 64,
                concurrency: int = 2, **kwargs) -> dict:
    return compose_case(
        model=model, hardware=hardware, tp=2, dp=1, ep=2, layers=layers,
        prompt_tokens=prompt, output_tokens=output, concurrency=concurrency, **kwargs,
    )


def _write_wave5(root: Path, *, tpot_ms: float = 12.0, step_ms: float = 14.0,
                 ttft_ms: float = 120.0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "concurrency": 2, "ttft_p50_ms": ttft_ms, "tpot_true_p50_ms": tpot_ms,
        "step_from_metrics_ms": step_ms, "itl_mean_p50_ms": step_ms - 1.0,
        "e2e_p50_ms": 500.0,
    }
    (root / "run_meta.json").write_text(json.dumps({
        "tag": "a3_synth_qwen35_L2_E32_K4_tp2ep2_decode_2k_c2",
        "npu_type": "Ascend910B_A3", "model_name": "qwen3.5-35b-a3b",
        "parallelism": {"tp": 2, "dp": 1, "ep": 2},
        "architecture_override": {
            "num_hidden_layers": 2, "num_routed_experts": 32, "num_experts_per_tok": 4,
        },
        "workload": {
            "mode": "decode", "input_len": 2048, "prefix_len": 1024,
            "output_len": 32, "concurrency": 2,
        },
        "metrics": {"summary": summary},
    }), encoding="utf-8")
    (root / "benchmark_summary.json").write_text(json.dumps({
        "summary": summary, "requests": [],
    }), encoding="utf-8")
    (root / "meta.env").write_text(
        "NPU_TYPE=Ascend910B_A3\nTP=2\nDP=1\nEP=2\nNUM_LAYERS=2\n"
        "NUM_EXPERTS=32\nTOP_K=4\nINPUT_LEN=2048\nPREFIX_LEN=1024\n"
        "OUTPUT_LEN=32\nCONCURRENCY=2\nMODE=decode\n", encoding="utf-8")
    (root / "cmd.sh").write_text(
        "#!/bin/bash\nvllm serve model --tensor-parallel-size 2 "
        "--max-num-seqs 16 --max-num-batched-tokens 8192 --max-model-len 4096 "
        "--gpu-memory-utilization 0.9\n", encoding="utf-8")
    (root / "op_statistic.csv").write_text(
        "Device_id,OP Type,Core Type,Count,Total Time(us),Ratio(%)\n"
        "0,AllReduce,MIX_AIC,10,100,20\n"
        "0,FusedInferAttentionScore,MIX_AIC,10,400,80\n", encoding="utf-8")
    (root / "kernel_details.csv").write_text(
        "Device_id,Step ID,Name,Start Time(us),Duration(us)\n"
        "0,1,matmul,0,100\n0,1,allreduce,100,20\n", encoding="utf-8")
    return root


def test_advise_baseline_attribution_ranking_and_no_mutation() -> None:
    case = _small_case()
    before = deepcopy(case)
    report = advise_case(case, context=2048, sequences=2, max_interventions=9)
    assert report["schema"] == "mindie-sim.advise/1"
    assert report["status"] == "completed"
    baseline = report["baseline"]
    assert set(baseline) == {
        "step_duration_ms", "operator_sum_ms", "tpot_p50_ms", "ttft_p50_ms",
        "sla_goodput_tok_s", "b_vram_max_sync", "b_vram_max_async", "dominant_bottleneck",
    }
    assert baseline["step_duration_ms"] > 0.0
    assert baseline["tpot_p50_ms"] > 0.0
    assert baseline["ttft_p50_ms"] > 0.0
    assert baseline["sla_goodput_tok_s"] >= 0.0
    assert baseline["b_vram_max_sync"] >= 1
    assert baseline["b_vram_max_async"] >= 1
    assert baseline["dominant_bottleneck"] in {
        "comm_bound", "launch_host_bound", "memory_kv_bound", "compute_bound"}

    attribution = report["attribution_4d"]
    assert set(attribution) == {"efficiency", "skew", "capacity", "stage"}
    efficiency = attribution["efficiency"]
    for key in ("cube_mfu", "vec_mfu", "mbu", "link_util", "cv_overlap",
                "roofline_efficiency", "breakdown_by_bound", "breakdown_by_kind",
                "dominant_bound"):
        assert key in efficiency
    assert 0.0 <= efficiency["mbu"] <= 1.0
    assert 0.0 <= efficiency["cube_mfu"] <= 1.0
    capacity = attribution["capacity"]
    for key in ("vram_max_sequences_per_dp_sync", "vram_max_sequences_per_dp_async",
                "weight_bytes_per_die", "kv_bytes_per_seq",
                "available_kv_capacity_bytes_per_die", "hbm_utilization_ratio",
                "capacity_headroom_bytes", "is_vram_feasible"):
        assert key in capacity
    assert capacity["weight_bytes_per_die"] > 0.0
    assert capacity["kv_bytes_per_seq"] > 0.0
    assert capacity["available_kv_capacity_bytes_per_die"] > 0.0
    stage = attribution["stage"]
    assert stage["pp_stages"] == 1
    assert stage["stage_layer_counts"] == [2]
    assert sum(stage[name] for name in ("attention_share", "moe_share", "comm_share",
                                        "norm_pointwise_share")) > 0.0
    skew = attribution["skew"]
    assert skew["num_dp_replicas"] == 1
    assert skew["has_routed_moe"] is True
    assert skew["moe_imbalance_slowdown_ratio"] >= 1.0

    assert report["observed_vs_predicted"] == {
        "available": False, "reason": "no_observed_profile_metadata"}
    recommendations = report["recommendations"]
    assert recommendations
    ids = [record["intervention_id"] for record in recommendations]
    assert len(ids) == len(set(ids))
    ordering = sorted(recommendations, key=lambda record: (
        0 if record["applicable_and_recommended"] else 1,
        -record["tpot_improvement_pct"], -record["sla_goodput_gain_pct"],
        -record["ttft_improvement_pct"], record["intervention_id"]))
    assert ids == [record["intervention_id"] for record in ordering]
    assert report["top_recommendation"] is not None
    assert report["top_recommendation"]["applicable_and_recommended"] is True
    assert case == before

    capped = advise_case(case, context=2048, sequences=2, max_interventions=2)
    assert len(capped["recommendations"]) == 2
    with pytest.raises(ValueError, match="max_interventions"):
        advise_case(case, max_interventions=-1)


def test_advise_observed_residuals_and_no_forward_leakage() -> None:
    case = _small_case()
    control = advise_case(case, context=2048, sequences=2, max_interventions=1)
    baseline = control["baseline"]
    observed_case = deepcopy(case)
    observed_case["runtime"]["metadata"] = {
        "observed_profile": {
            "observed_step_ms": baseline["step_duration_ms"] * 1.1,
            "observed_tpot_p50_ms": baseline["tpot_p50_ms"] * 1.1,
            "observed_ttft_p50_ms": baseline["ttft_p50_ms"] * 1.1,
            "observed_dominant_bottleneck": baseline["dominant_bottleneck"],
            "observed_share_by_bound": {baseline["dominant_bottleneck"]: 1.0},
        }
    }
    before = deepcopy(observed_case)
    report = advise_case(observed_case, context=2048, sequences=2, max_interventions=1)
    assert report["baseline"] == baseline
    residual = report["observed_vs_predicted"]
    assert residual["available"] is True
    assert residual["tpot_delta_pct"] == pytest.approx(10.0, abs=1e-6)
    assert residual["step_delta_pct"] == pytest.approx(10.0, abs=1e-6)
    assert residual["ttft_delta_pct"] == pytest.approx(10.0, abs=1e-6)
    assert residual["bottleneck_aligned"] is True
    assert residual["calibration_status"] == "calibrated"
    assert set(residual["share_residuals"]) == {
        "comm_bound", "launch_host_bound", "memory_kv_bound", "compute_bound"}
    for value in residual["share_residuals"].values():
        assert value["residual"] == pytest.approx(
            value["observed_share"] - value["predicted_share"], abs=1e-9)
    assert observed_case == before

    mapped = deepcopy(observed_case)
    mapped["runtime"]["metadata"]["observed_profile"]["observed_dominant_bottleneck"] = \
        "memory_bound"
    mapped_report = advise_case(mapped, context=2048, sequences=2, max_interventions=0)
    assert mapped_report["observed_vs_predicted"]["observed_dominant_bottleneck"] == \
        "memory_kv_bound"


def test_guardrails_across_catalog() -> None:
    # A5 AIV expansion is selectable and contends with Vector work.
    a5 = _small_case(hardware="a5")
    a5_report = advise_case(a5, context=2048, sequences=2, max_interventions=3)
    aiv = next(record for record in a5_report["recommendations"]
               if record["category"] == "hccl_op_expansion_mode")
    assert "aiv_expansion_vector_contention_risk" in {
        guardrail["id"] for guardrail in aiv["guardrails"]}

    # Tight A2 memory: ACLGraph capture exceeds KV headroom.
    tight = compose_case("deepseek-v2-lite", hardware="a2", tp=4, dp=1, ep=4,
                         layers=4, prompt_tokens=131072, output_tokens=32,
                         concurrency=5)
    tight["machine"]["device"]["memory_bytes"] = 4 * 1024**3
    tight["runtime"]["gpu_memory_utilization"] = 0.95
    tight_report = advise_case(tight, context=131072, sequences=5, max_interventions=1)
    assert "aclgraph_capture_memory_exceeds_hbm_headroom" in {
        guardrail["id"] for guardrail in tight_report["recommendations"][0]["guardrails"]}

    # Long-context KV quantization, delayed-release async, and missing draft
    # weights are flagged on an A2 MoE profile.
    long_context = compose_case("deepseek-v2-lite", hardware="a2", tp=4, dp=1, ep=4,
                                layers=8, prompt_tokens=131072, output_tokens=160,
                                concurrency=1)
    long_report = advise_case(long_context, context=131072, sequences=1,
                              max_interventions=9)
    by_category = {record["category"]: record for record in long_report["recommendations"]}
    assert "kv_quant_calibration_accuracy_guardrail" in {
        guardrail["id"] for guardrail in by_category["kv_quant"]["guardrails"]}
    assert "async_scheduling_delayed_kv_release_reduces_max_batch" in {
        guardrail["id"] for guardrail in by_category["async_scheduling"]["guardrails"]}
    assert "requires_draft_checkpoint_weights" in {
        guardrail["id"] for guardrail in by_category["speculative_mode"]["guardrails"]}

    # High-batch MTP verification is flagged as compute-bound.
    high_batch = _small_case(prompt=4096, concurrency=32)
    high_report = advise_case(high_batch, context=4096, sequences=32,
                              max_interventions=8)
    speculative = next(record for record in high_report["recommendations"]
                       if record["category"] == "speculative_mode")
    assert "speculative_compute_bound_at_high_batch" in {
        guardrail["id"] for guardrail in speculative["guardrails"]}


def test_from_profile_wave5_synthetic_and_cli(tmp_path: Path) -> None:
    profile = _write_wave5(tmp_path / "wave5")
    case = load_case_from_profile(profile)
    assert case["machine"]["sku"] == "Ascend910_9392"
    assert case["runtime"]["parallel"]["tp"] == 2
    assert case["runtime"]["parallel"]["ep"] == 2
    assert case["model"]["layers"] == 2
    assert case["model"]["num_experts"] == 32
    assert case["workload"]["requests"][0]["prompt_tokens"] == 2048
    assert case["workload"]["requests"][0]["prefix_tokens"] == 1024
    metadata = case["runtime"]["metadata"]
    assert metadata["observed_profile"]["observed_step_ms"] == pytest.approx(14.0)
    assert metadata["observed_profile"]["observed_tpot_p50_ms"] == pytest.approx(12.0)
    assert metadata["observed_profile"]["observed_ttft_p50_ms"] == pytest.approx(120.0)
    assert metadata["observed_profile"]["observed_dominant_bottleneck"] == "compute_bound"
    assert metadata["profiler_provenance"]["source_kind"] == "wave5_capture"
    assert metadata["profiler_provenance"]["source_sha256"]

    report = advise_case(from_profile=profile, context=2048, sequences=2,
                         max_interventions=2)
    assert report["observed_vs_predicted"]["available"] is True
    assert report["observed_vs_predicted"]["observed_tpot_p50_ms"] == pytest.approx(12.0)

    compose_path = tmp_path / "composed.json"
    assert main(["compose", "--from-profile", str(profile), "--tp", "2",
                 "--dp", "1", "--ep", "2", "--layers", "1", "-o",
                 str(compose_path)]) == 0
    overridden = json.loads(compose_path.read_text(encoding="utf-8"))
    assert overridden["model"]["layers"] == 1
    assert overridden["runtime"]["metadata"]["observed_profile"][
        "observed_tpot_p50_ms"] == pytest.approx(12.0)

    advise_path = tmp_path / "advised.json"
    assert main(["advise", "--from-profile", str(profile), "--context", "2048",
                 "--sequences", "2", "--max-interventions", "2", "-o",
                 str(advise_path)]) == 0
    advised = json.loads(advise_path.read_text(encoding="utf-8"))
    assert advised["schema"] == "mindie-sim.advise/1"
    assert advised["observed_vs_predicted"]["available"] is True

    assert main(["compose"]) == 2


def test_from_profile_c7_capture_and_single_files(tmp_path: Path) -> None:
    c7 = tmp_path / "c7"
    c7.mkdir()
    (c7 / "summary.json").write_text(json.dumps({
        "schema_version": "mindie-profile-summary/1",
        "scene": {"model": "qwen3.5-35b-a3b", "hardware": "a3", "tp": 2,
                  "dp": 1, "ep": 2, "phase": "decode"},
        "step_accounting": {"median_step_ms": 9.0},
        "semantic_block_breakdown": {
            "attention": {"block_union_ms": 60},
            "communication": {"block_union_ms": 20},
            "mte": {"block_union_ms": 20},
        },
        "window_accounting": {"observed_event_span_ms": 100.0},
    }), encoding="utf-8")
    c7_case = load_case_from_profile(c7 / "summary.json")
    c7_observed = c7_case["runtime"]["metadata"]["observed_profile"]
    assert c7_observed["observed_step_ms"] == pytest.approx(9.0)
    assert c7_observed["observed_dominant_bottleneck"] == "compute_bound"
    assert set(c7_observed["observed_share_by_bound"]) == {
        "comm_bound", "memory_kv_bound", "compute_bound"}

    capture = tmp_path / "capture"
    (capture / "rank_0").mkdir(parents=True)
    (capture / "capture.json").write_text(json.dumps({
        "scene": {"model": "deepseek-v2-lite", "hardware": "a2", "tp": 2,
                  "dp": 1, "ep": 2, "phase": "decode"},
        "workload": {"prompt_tokens": 2048, "output_tokens": 32, "concurrency": 2},
    }), encoding="utf-8")
    (capture / "provenance.json").write_text(json.dumps({"capture_id": "synthetic"}),
                                             encoding="utf-8")
    (capture / "rank_0" / "op_statistic.csv").write_text(
        "OP Type,Total Time(us)\nAllReduce,20\nAttention,80\n", encoding="utf-8")
    (capture / "rank_0" / "kernel_details.csv").write_text(
        "Step ID,Name,Start Time(us),Duration(us)\n1,Attention,0,80\n1,AllReduce,80,20\n",
        encoding="utf-8")
    capture_case = load_case_from_profile(capture)
    capture_observed = capture_case["runtime"]["metadata"]["observed_profile"]
    assert capture_observed["observed_step_ms"] == pytest.approx(0.1)
    assert capture_observed["observed_share_by_bound"]["compute_bound"] == pytest.approx(0.8)
    assert capture_case["runtime"]["metadata"]["profiler_provenance"][
        "source_kind"] == "profiler_capture"

    run_meta = tmp_path / "run_meta.json"
    run_meta.write_text(json.dumps({
        "tag": "a3_synth_qwen35_L2_E32_K4_tp2ep2_decode_2k_c2",
        "npu_type": "Ascend910C_A3", "model_name": "qwen3.5-35b-a3b",
        "parallelism": {"tp": 2, "dp": 1, "ep": 2},
        "architecture_override": {"num_hidden_layers": 2, "num_routed_experts": 32,
                                  "num_experts_per_tok": 4},
        "workload": {"mode": "prefill_chunked", "input_len": 2048,
                     "output_len": 1, "concurrency": 1},
        "metrics": {"summary": {"step_from_metrics_ms": 8.0, "ttft_p50_ms": 8.0}},
    }), encoding="utf-8")
    run_meta_case = load_case_from_profile(run_meta)
    assert run_meta_case["runtime"]["metadata"]["observed_profile"][
        "observed_step_ms"] == pytest.approx(8.0)

    attribution = tmp_path / "attribution"
    attribution.mkdir()
    (attribution / "attribution.json").write_text(json.dumps({
        "schema_version": "mindie-profile-summary/1",
        "scene": {"model": "qwen3.5-35b-a3b", "hardware": "a3", "tp": 2,
                  "dp": 1, "ep": 2, "phase": "decode"},
        "attribution": {"observed_step_ms": 11.0, "dominant_bottleneck": "communication",
                        "share_by_bound": {"link_bound": 0.4, "cube_bound": 0.6}},
    }), encoding="utf-8")
    attribution_case = load_case_from_profile(attribution / "attribution.json")
    attribution_observed = attribution_case["runtime"]["metadata"]["observed_profile"]
    assert attribution_observed["observed_step_ms"] == pytest.approx(11.0)
    assert attribution_observed["observed_dominant_bottleneck"] == "comm_bound"

    exported = _small_case(layers=1, prompt=1024, output=16, concurrency=1)
    exported["runtime"]["metadata"] = {
        "observed_profile": {"observed_tpot_p50_ms": 5.0,
                             "observed_dominant_bottleneck": "compute_bound"}}
    exported_path = tmp_path / "exported.json"
    exported_path.write_text(json.dumps(exported), encoding="utf-8")
    exported_case = load_case_from_profile(exported_path)
    assert exported_case["runtime"]["metadata"]["observed_profile"][
        "observed_tpot_p50_ms"] == pytest.approx(5.0)


def test_p2_review_regressions(tmp_path: Path) -> None:
    # 1. Pure-device cases do not fabricate host launch/scheduler cost; declared submission differentiates.
    pure_case = _small_case(layers=2, prompt=2048, output=64, concurrency=2)
    pure_rep = advise_case(pure_case, context=2048, sequences=2, max_interventions=9)
    pure_cat = {r["category"]: r for r in pure_rep["recommendations"]}
    assert pure_cat["async_scheduling"]["predicted"]["step_duration_ms"] == pytest.approx(
        pure_rep["baseline"]["step_duration_ms"]
    )
    case = _small_case(layers=2, prompt=2048, output=64, concurrency=2)
    case["runtime"]["submission"] = {
        "kernel_launch_ms": 0.02, "graph_replay_ms": 0.005, "scheduler_ms": 0.05,
    }
    report = advise_case(case, context=2048, sequences=2, max_interventions=9)
    by_cat = {r["category"]: r for r in report["recommendations"]}
    assert by_cat["execution_mode"]["predicted"]["step_duration_ms"] < report["baseline"]["step_duration_ms"]
    assert by_cat["async_scheduling"]["predicted"]["step_duration_ms"] < report["baseline"]["step_duration_ms"]
    assert by_cat["speculative_mode"]["predicted"]["step_duration_ms"] > report["baseline"]["step_duration_ms"]
    assert by_cat["speculative_mode"]["predicted"]["tpot_p50_ms"] != report["baseline"]["tpot_p50_ms"]

    # Async scheduling overlaps scheduler_ms with device work (max(work, sched)) rather than zeroing it.
    heavy_sched = deepcopy(case)
    heavy_sched["runtime"]["submission"]["scheduler_ms"] = 50.0
    heavy_rep = advise_case(heavy_sched, context=2048, sequences=2, max_interventions=9)
    heavy_async = next(r for r in heavy_rep["recommendations"] if r["category"] == "async_scheduling")
    assert heavy_async["predicted"]["step_duration_ms"] == pytest.approx(50.0)
    assert heavy_rep["baseline"]["step_duration_ms"] > 50.0

    # 2. KV quantization updates model.quant.kv_dtype and shifts KV capacity.
    kv_case = _small_case("deepseek-v2-lite", hardware="a2", layers=2, prompt=2048, output=64, concurrency=2)
    kv_case["runtime"]["max_num_seqs"] = 4096
    kv_case["runtime"]["gpu_memory_utilization"] = 0.08
    kv_rep = advise_case(kv_case, context=2048, sequences=2, max_interventions=9)
    kv_rec = next(r for r in kv_rep["recommendations"] if r["category"] == "kv_quant")
    assert kv_rec["predicted"]["b_vram_max_sync"] > kv_rep["baseline"]["b_vram_max_sync"]

    # 3. TTFT SLA violation sets sla_goodput_tok_s == 0.0.
    ttft_rep = advise_case(case, context=2048, sequences=2, sla_ttft_ms=0.001, max_interventions=1)
    assert ttft_rep["baseline"]["sla_goodput_tok_s"] == 0.0

    # 4. Unknown runtime keys and invalid draft_mode must raise rather than silently falling back.
    bad_case = _small_case(layers=1)
    bad_case["runtime"]["serving_floor_ms"] = 999.0
    with pytest.raises(Exception):
        advise_case(bad_case, max_interventions=1)
    bad_draft = _small_case(layers=1)
    bad_draft["runtime"]["mtp_acceptance_length"] = 9.0
    bad_draft["runtime"]["metadata"] = {"draft_mode": "mtp", "draft_proposal_tokens": 2}
    with pytest.raises(ValueError, match="draft_acceptance_exceeds_declared_proposals"):
        advise_case(bad_draft, max_interventions=1)

    # 5. Calibration geometry/hardware/boundary mismatch and insufficient latency evidence.
    obs_case = _small_case(layers=1, prompt=1024, output=32, concurrency=2)
    obs_case["runtime"]["metadata"] = {
        "observed_profile": {
            "observed_dominant_bottleneck": "compute_bound",
            "observed_share_by_bound": {"compute_bound": 1.0},
        }
    }
    insuff_rep = advise_case(obs_case, context=1024, sequences=2, max_interventions=0)
    assert insuff_rep["observed_vs_predicted"]["calibration_status"] == "insufficient_observed_metrics"
    obs_case["runtime"]["metadata"]["observed_profile"]["observed_tpot_p50_ms"] = 10.0
    obs_case["runtime"]["metadata"]["observed_profile"]["phase"] = "prefill"
    geom_rep = advise_case(obs_case, phase="decode", context=1024, sequences=2, max_interventions=0)
    assert geom_rep["observed_vs_predicted"]["calibration_status"] == "geometry_mismatch"
    obs_case["runtime"]["metadata"]["observed_profile"]["phase"] = "decode"
    obs_case["runtime"]["metadata"]["observed_profile"]["hardware"] = "Ascend910B4"
    hw_rep = advise_case(obs_case, phase="decode", context=1024, sequences=2, max_interventions=0)
    assert hw_rep["observed_vs_predicted"]["calibration_status"] == "geometry_mismatch"

    # 6. Native C7 fields, partial share preservation, and structural override preservation.
    c7_native = tmp_path / "c7_native.json"
    c7_native.write_text(json.dumps({
        "schema_version": "mindie-profile-summary/1",
        "scene": {"model": "qwen3.5-35b-a3b", "hardware": "a3", "tp": 2, "dp": 1, "ep": 2, "phase": "decode"},
        "workload": {"prompt_tokens": 2048, "prefix_tokens": 1024, "output_tokens": 32, "concurrency": 2},
        "window_accounting": {
            "device_busy_union_ms_p50": 100.0,
            "exposed_comm_ms_p50": 0.5,
            "host_gap_ms_p50": 10.0,
        },
        "semantic_block_breakdown": {
            "communication": {"wall_ms_p50": 0.5, "share_of_busy_pct": 0.5},
        },
    }), encoding="utf-8")
    c7_loaded = load_case_from_profile(c7_native, prefix_tokens=0, layers=2)
    assert c7_loaded["workload"]["requests"][0]["prefix_tokens"] == 0
    assert c7_loaded["runtime"]["metadata"]["observed_profile"]["observed_share_by_bound"]["comm_bound"] == pytest.approx(0.005)
    assert "observed_dominant_bottleneck" not in c7_loaded["runtime"]["metadata"]["observed_profile"]
    assert "observed_step_ms" not in c7_loaded["runtime"]["metadata"]["observed_profile"]

    custom_base = compose_case("qwen3.5-35b-a3b", hardware="a3", tp=2, dp=1, ep=2, layers=2,
                               num_devices=4, ranks_per_node=2)
    custom_base["runtime"]["splitting_ops"] = ["allreduce"]
    derived = load_case_from_profile(custom_base, layers=1)
    assert derived["machine"]["num_devices"] == 4
    assert derived["machine"]["ranks_per_node"] == 2
    assert derived["runtime"]["splitting_ops"] == ["allreduce"]

    # 7. Specific checkpoint aliases, default prefill TTFT parity, speculative acceptance preservation,
    #    cmd.sh preserving explicit graph_mode, and VRAM-infeasible intervention blocking.
    from mindie_sim.profile_ingest import _canonical_model
    assert _canonical_model("/home/weight/Qwen3.8-2.4T-A95B") == "qwen3.8-2.4t-a95b"
    assert _canonical_model("GLM-5.3-Flash-W4A8") == "glm-5.3-flash"
    prefill_rep = advise_case(pure_case, phase="prefill", sequences=1, max_interventions=0)
    decode_rep = advise_case(pure_case, phase="decode", sequences=1, max_interventions=0)
    assert prefill_rep["baseline"]["ttft_p50_ms"] == pytest.approx(decode_rep["baseline"]["ttft_p50_ms"])
    spec_prof = load_case_from_profile({
        "model_name": "qwen3.5-35b-a3b", "npu_type": "a3", "tp": 2, "dp": 1, "ep": 2, "layers": 1,
        "spec_k": 2, "mtp_acceptance_length": 2.5, "graph_mode": "full",
        "cmd_flags": {"--tensor-parallel-size": "2"},
    })
    assert spec_prof["runtime"]["mtp_acceptance_length"] == pytest.approx(2.5)
    assert spec_prof["runtime"]["graph_capture_mode"] == "full"
    infeasible = _small_case(layers=1, prompt=4096, output=32, concurrency=4)
    infeasible["machine"]["device"]["memory_bytes"] = 512 * 1024**2
    inf_rep = advise_case(infeasible, context=4096, sequences=4, max_interventions=9)
    assert all(not r["applicable_and_recommended"] for r in inf_rep["recommendations"])

    # 8. observed_profile model comparison, multi-DP speculative efficiency, prefill failure gating,
    #    quantization JSON extraction, gpu_memory_utilization override, and mtp_proposal_tokens=0.
    obs_mod_case = deepcopy(pure_case)
    obs_mod_case["runtime"]["metadata"] = {
        "observed_profile": {
            "model": "qwen3.5-35b-a3b",
            "observed_tpot_p50_ms": 20.0,
            "observed_dominant_bottleneck": "compute_bound",
        }
    }
    obs_rep = advise_case(obs_mod_case, sequences=1, max_interventions=0)
    assert obs_rep["observed_vs_predicted"]["available"] is True
    assert obs_rep["observed_vs_predicted"]["calibration_status"] != "geometry_mismatch"
    obs_mod_case["runtime"]["metadata"]["observed_profile"]["model"] = "deepseek-v3"
    assert advise_case(obs_mod_case, sequences=1, max_interventions=0)["observed_vs_predicted"]["calibration_status"] == "geometry_mismatch"

    dp_base = compose_case("qwen3.5-35b-a3b", hardware="a3", tp=1, dp=2, ep=2, layers=1,
                           prompt_tokens=1024, output_tokens=16, concurrency=2)
    dp_spec = load_case_from_profile(dp_base, mtp_proposal_tokens=2, mtp_acceptance_length=2.2)
    rep_no_spec = advise_case(dp_base, sequences=1, max_interventions=0)
    rep_spec = advise_case(dp_spec, sequences=1, max_interventions=0)
    assert rep_spec["baseline"]["operator_sum_ms"] > rep_no_spec["baseline"]["operator_sum_ms"]
    assert rep_spec["baseline"]["b_vram_max_sync"] <= rep_no_spec["baseline"]["b_vram_max_sync"]

    from mindie_sim import advise_catalog
    orig_analyze = advise_catalog._analyze
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            advise_catalog, "_analyze",
            lambda parsed, cfg, **kw: (_ for _ in ()).throw(ValueError("prefill_failed"))
            if kw.get("phase") == "prefill" else orig_analyze(parsed, cfg, **kw),
        )
        unbuf_rep = advise_case(pure_case, context=2048, sequences=1, max_interventions=2)
    piecewise_rec = next(r for r in unbuf_rep["recommendations"]
                         if r["intervention_id"] == "execution_mode.piecewise")
    assert piecewise_rec["applicable"] is False
    assert piecewise_rec["reason"] == "counterfactual_not_selectable:prefill_failed"

    quant_case = load_case_from_profile({
        "model_name": "qwen3.5-35b-a3b", "npu_type": "a3", "tp": 2, "dp": 1, "ep": 2,
        "layers": 1, "quantization": "w8a8",
    }, gpu_memory_utilization=0.85)
    assert quant_case["model"]["weight_dtype"] == "int8"
    assert quant_case["runtime"]["gpu_memory_utilization"] == pytest.approx(0.85)

    disabled_mtp = load_case_from_profile(dp_spec, mtp_proposal_tokens=0)
    assert "draft_mode" not in disabled_mtp["runtime"]["metadata"]
    assert "draft_proposal_tokens" not in disabled_mtp["runtime"]["metadata"]
    assert disabled_mtp["runtime"]["mtp_acceptance_length"] == pytest.approx(1.0)
    disabled_mtp_struct = load_case_from_profile(dp_spec, tp=2, dp=1, ep=2, mtp_proposal_tokens=0)
    assert "draft_mode" not in disabled_mtp_struct["runtime"]["metadata"]
    assert "draft_proposal_tokens" not in disabled_mtp_struct["runtime"]["metadata"]

    # 9. Heterogeneous request preservation, SLA probe error handling, prefill KV capacity,
    #    single completion_ms charge in speculative decode, CP/DCP/SP preservation,
    #    client boundary rejection, distinct model prefix check, custom GPU hardware overrides,
    #    sibling provenance hashing, and hardware override before inference.
    het_case = deepcopy(pure_case)
    het_case["workload"]["requests"] = [
        {"id": "r0", "arrival_ms": 0.0, "prompt_tokens": 32, "output_tokens": 16, "prefix_tokens": 0, "priority": 1},
        {"id": "r1", "arrival_ms": 100.0, "prompt_tokens": 128, "output_tokens": 32, "prefix_tokens": 16, "priority": 2},
    ]
    het_loaded = load_case_from_profile(het_case, layers=1)
    assert [r["prompt_tokens"] for r in het_loaded["workload"]["requests"]] == [32, 128]
    assert [r["arrival_ms"] for r in het_loaded["workload"]["requests"]] == [0.0, 100.0]
    assert [r["priority"] for r in het_loaded["workload"]["requests"]] == [1, 2]
    het_out = load_case_from_profile(het_case, output_tokens=64)
    assert [r["prompt_tokens"] for r in het_out["workload"]["requests"]] == [32, 128]
    assert [r["output_tokens"] for r in het_out["workload"]["requests"]] == [64, 64]

    prefill_kv = advise_case(pure_case, phase="prefill", context=2048, sequences=1, max_interventions=0)
    decode_kv = advise_case(pure_case, phase="decode", context=2048, sequences=1, max_interventions=0)
    assert prefill_kv["attribution_4d"]["capacity"]["kv_bytes_per_seq"] == pytest.approx(
        decode_kv["attribution_4d"]["capacity"]["kv_bytes_per_seq"]
    )

    comp_base = deepcopy(dp_spec)
    comp_base["runtime"]["submission"] = {"completion_ms": 10.0}
    comp_zero = deepcopy(dp_spec)
    comp_zero["runtime"]["submission"] = {"completion_ms": 0.0}
    rep_c10 = advise_case(comp_base, sequences=1, max_interventions=0)
    rep_c0 = advise_case(comp_zero, sequences=1, max_interventions=0)
    assert rep_c10["baseline"]["step_duration_ms"] - rep_c0["baseline"]["step_duration_ms"] == pytest.approx(10.0)

    cp_case = compose_case("deepseek-v2-lite", hardware="a3", tp=2, dp=1, ep=1, dcp=2, layers=1)
    cp_derived = load_case_from_profile(cp_case, dp=1)
    assert cp_derived["runtime"]["parallel"]["cp"] == 2
    sp_base = compose_case("kimi-k3", hardware="a3", tp=2, dp=1, ep=2, layers=2)
    sp_enabled = load_case_from_profile(sp_base, sp=True)
    assert "sequence_parallel" in sp_enabled["runtime"]["kernels"]
    sp_disabled = load_case_from_profile(sp_enabled, sp=False)
    assert "sequence_parallel" not in sp_disabled["runtime"]["kernels"]

    client_obs = deepcopy(pure_case)
    client_obs["runtime"]["serving_io"] = {"client_delivery_ms": 10000.0}
    client_obs["runtime"]["metadata"] = {
        "observed_profile": {
            "measurement_boundary": "client",
            "observed_tpot_p50_ms": 20.0,
            "observed_dominant_bottleneck": "compute_bound",
        }
    }
    assert advise_case(client_obs, sequences=1, max_interventions=0)["observed_vs_predicted"]["calibration_status"] == "geometry_mismatch"

    q38_obs = deepcopy(pure_case)
    q38_obs["runtime"]["metadata"] = {
        "observed_profile": {
            "model": "qwen3.8-27b",
            "observed_tpot_p50_ms": 20.0,
            "observed_dominant_bottleneck": "compute_bound",
        }
    }
    assert advise_case(q38_obs, sequences=1, max_interventions=0)["observed_vs_predicted"]["calibration_status"] == "geometry_mismatch"

    gpu_case = deepcopy(pure_case)
    gpu_case["machine"]["sku"] = "declared-gpu"
    gpu_case["machine"]["name"] = "declared-gpu"
    gpu_case["machine"]["hardware_key"] = "declared-gpu"
    gpu_derived = load_case_from_profile(gpu_case, layers=1)
    assert gpu_derived["machine"]["sku"] == "declared-gpu"
    gpu_derived["runtime"]["metadata"] = {
        "observed_profile": {
            "hardware": "declared-gpu",
            "observed_tpot_p50_ms": advise_case(gpu_derived, sequences=1, max_interventions=0)["baseline"]["tpot_p50_ms"],
            "observed_dominant_bottleneck": advise_case(gpu_derived, sequences=1, max_interventions=0)["baseline"]["dominant_bottleneck"],
        }
    }
    assert advise_case(gpu_derived, sequences=1, max_interventions=0)["observed_vs_predicted"]["calibration_status"] == "calibrated"

    sib_dir = _write_wave5(tmp_path / "sib_wave5", tpot_ms=10.0)
    c_sib1 = load_case_from_profile(sib_dir / "run_meta.json")
    (sib_dir / "benchmark_summary.json").write_text(
        json.dumps({"summary": {"concurrency": 2, "tpot_true_p50_ms": 99.0}, "requests": []}),
        encoding="utf-8",
    )
    c_sib2 = load_case_from_profile(sib_dir / "run_meta.json")
    assert c_sib1["runtime"]["metadata"]["profiler_provenance"]["source_sha256"] != c_sib2["runtime"]["metadata"]["profiler_provenance"]["source_sha256"]
    assert "benchmark_summary.json" in c_sib1["runtime"]["metadata"]["profiler_provenance"]["files"]

    hw_over = load_case_from_profile({"model_name": "deepseek-v2-lite", "hardware": "unknown-sku"}, hardware="a3", layers=1)
    assert hw_over["machine"]["sku"] == "Ascend910_9392"

    # P1: Multi-sequence decode TTFT evaluated at cfg.sequences, and b_sla capped by TTFT SLA
    ttft_case = deepcopy(case)
    ttft_case["runtime"]["max_num_seqs"] = 8
    t_s1 = advise_case(ttft_case, phase="decode", context=2048, sequences=1, max_interventions=1)["baseline"]["ttft_p50_ms"]
    t_s4 = advise_case(ttft_case, phase="decode", context=2048, sequences=4, max_interventions=1)["baseline"]["ttft_p50_ms"]
    assert t_s4 > t_s1
    assert advise_case(ttft_case, phase="decode", context=2048, sequences=4, sla_ttft_ms=(t_s1 + t_s4) / 2.0, max_interventions=1)["baseline"]["sla_goodput_tok_s"] == 0.0
    assert advise_case(ttft_case, phase="decode", context=2048, sequences=1, sla_ttft_ms=(t_s1 + t_s4) / 2.0, sla_tpot_ms=500.0, max_interventions=0)["baseline"]["sla_goodput_tok_s"] < advise_case(ttft_case, phase="decode", context=2048, sequences=1, sla_ttft_ms=5000.0, sla_tpot_ms=500.0, max_interventions=0)["baseline"]["sla_goodput_tok_s"]

    # P2 #1: Prefill capacity reserves both prompt and output KV tokens
    cap_case = compose_case("deepseek-v2-lite", hardware="a3", tp=1, dp=1, ep=1, layers=1, prompt_tokens=512, output_tokens=512, concurrency=1)
    assert advise_case(cap_case, phase="prefill", max_interventions=0)["attribution_4d"]["capacity"]["kv_bytes_per_seq"] == advise_case(cap_case, phase="decode", max_interventions=0)["attribution_4d"]["capacity"]["kv_bytes_per_seq"]
    assert advise_case(cap_case, phase="prefill", max_interventions=0)["baseline"]["b_vram_max_sync"] == advise_case(cap_case, phase="decode", max_interventions=0)["baseline"]["b_vram_max_sync"]

    # P2 #2 & #3: Extending workload assigns unique IDs; shorthand workload overrides work
    ext_reqs = load_case_from_profile(cap_case, concurrency=4)["workload"]["requests"]
    assert len({r["id"] for r in ext_reqs}) == 4
    short_case = deepcopy(cap_case)
    short_case["workload"] = {"num_requests": 2, "prompt_tokens": 512, "output_tokens": 64}
    short_over = load_case_from_profile(short_case, prompt_tokens=1024, concurrency=3)
    assert len(short_over["workload"]["requests"]) == 3 and short_over["workload"]["requests"][0]["prompt_tokens"] == 1024

    # P2 #4: mtp_proposal_tokens=0 disables all speculative policies
    spec_case = compose_case("qwen3.5-35b-a3b", hardware="a3", tp=2, dp=1, ep=2, layers=1, mtp_proposal_tokens=2, mtp_acceptance_length=2.2)
    spec_case["runtime"]["mtp_acceptance_by_output_tokens"] = {128: 2.2}
    spec_case["runtime"]["metadata"]["speculative"] = {"draft_len": 2, "accept_probs": [0.8, 0.6]}
    spec_off = load_case_from_profile(spec_case, mtp_proposal_tokens=0)
    assert "mtp_acceptance_by_output_tokens" not in spec_off["runtime"] and "speculative" not in spec_off["runtime"].get("metadata", {})
    assert advise_case(spec_off, max_interventions=0)["geometry"]["tokens"] == 1

    # P2 #5: Retired/unknown top-level fields on exported Case raise ModelingError
    with pytest.raises(Exception, match="structure_variant_retired"):
        load_case_from_profile({**cap_case, "structure_variant": "legacy"})

    # P2 #6 & #7: Execution mode mismatch invalidates calibration; separate from_profile merges provenance
    eager_obs = deepcopy(cap_case)
    eager_obs["runtime"]["graph_mode"] = eager_obs["runtime"]["graph_capture_mode"] = "none"
    eager_obs["runtime"]["metadata"] = {"execution_mode": "eager", "observed_profile": {"observed_tpot_p50_ms": 1.0, "observed_dominant_bottleneck": "compute_bound"}}
    assert advise_case(load_case_from_profile(eager_obs, graph_mode="piecewise"), max_interventions=0)["observed_vs_predicted"]["calibration_status"] == "geometry_mismatch"
    assert advise_case(cap_case, from_profile=sib_dir, max_interventions=0)["profile_provenance"].get("source_kind") == "wave5_capture"


def test_delegated_bridge_and_cli_shim(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = _small_case(layers=1, prompt=1024, output=16, concurrency=1)
    package = types.ModuleType("mindie_profiler")
    package.__path__ = []
    bridge = types.ModuleType("mindie_profiler.sim_bridge")
    bridge.export_sim_case = lambda source: base
    package.sim_bridge = bridge
    monkeypatch.setitem(sys.modules, "mindie_profiler", package)
    monkeypatch.setitem(sys.modules, "mindie_profiler.sim_bridge", bridge)
    delegated_before = deepcopy(base)
    case = load_case_from_profile(tmp_path / "missing")
    assert base == delegated_before
    assert case["model"]["layers"] == 1
    assert case["runtime"]["metadata"]["profiler_provenance"][
        "source_kind"] == "delegated_sim_bridge"

    assert cli_main is main
    parsed = parser().parse_args(["advise", "--from-profile", "missing",
                                  "--phase", "decode"])
    assert parsed.phase == "decode"
    assert callable(run)

    root = Path(__file__).resolve().parents[1] / "mindie_sim"
    for mod_name in ("advise.py", "advise_catalog.py", "advise_core.py", "advise_util.py",
                     "profile_ingest.py", "profile_observed.py", "cli.py"):
        lines = (root / mod_name).read_text(encoding="utf-8").splitlines()
        assert len(lines) < 800, f"{mod_name} has {len(lines)} lines"
        header = "\n".join(lines[:30])
        for marker in ("[Role]", "[Invariants]", "[Timeouts & Locks]", "[Side-Effect Policy]"):
            assert marker in header, f"{mod_name} missing {marker}"


def test_from_profile_real_wave5_when_present() -> None:
    if not REAL_WAVE5.is_dir():
        pytest.skip(f"real Wave 5 fixture not present: {REAL_WAVE5}")
    case = load_case_from_profile(REAL_WAVE5)
    assert case["machine"]["sku"] == "Ascend910B4"
    assert case["runtime"]["parallel"]["tp"] == 4
    assert case["runtime"]["parallel"]["ep"] == 4
    assert case["model"]["layers"] == 27
    assert case["model"]["num_experts"] == 64
    assert case["model"]["num_experts_per_tok"] == 6
    request = case["workload"]["requests"][0]
    assert request["prompt_tokens"] == 131072
    assert request["prefix_tokens"] == 130944
    assert request["output_tokens"] == 160
    observed = case["runtime"]["metadata"]["observed_profile"]
    assert observed["observed_tpot_p50_ms"] == pytest.approx(18.40853990998658)
    assert observed["observed_step_ms"] == pytest.approx(35.93128891851276)
    assert observed["observed_ttft_p50_ms"] == pytest.approx(483.9179515838623)
    assert observed["observed_dominant_bottleneck"] == "compute_bound"
    assert case["runtime"]["metadata"]["profiler_provenance"]["source_kind"] == "wave5_capture"
