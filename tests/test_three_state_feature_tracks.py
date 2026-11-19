"""Tier 0/1 verification for Two-Layer Pluggable Three-State Feature Tracks.

[Role]
  Pin ``kimi-k3.1`` paired-SFA page sizing, the monotone 12-model four-level
  waterfall, single-PR What-If attribution, zero-edit JSON stack onboarding,
  model-support honesty, and the offline stack-PR audit helper.

[Invariants]
  * ``merged`` is bitwise the documented vLLM-Ascend baseline; each subsequent
    level preserves the engine's physical work and only removes modeled traffic.
  * Waterfall ordering (``merged >= unmerged_pr >= expected_ideal >= 1.0x``)
    is asserted on real graph durations, never on stored expectations.
  * Built-in stacks are only audited ``vllm-ascend`` and framework-neutral
    ``raw_eager``; third-party stacks must declare real model support.
  * ``supported_models`` gates ``merged``, ``unmerged_pr_models`` additionally
    permits ``unmerged_pr``, and ``expected_ideal`` is never model-gated.
  * An explicit caller ``block_size`` is preserved so invalid pages still fail.

[Timeouts & Locks]
  Small layer counts and two-device-free cases; one bounded subprocess for the
  audit helper; no devices, network or long-running work.

[Side-Effect Policy]
  Writes only inside pytest's temporary directory.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mindie_sim import (
    ModelingError,
    compare_feature_tracks,
    compose_case,
    load_runtime_stack_manifest,
    parse_case,
)
from mindie_sim.prediction.capacity import Memory
from mindie_sim.stack_tracks import RUNTIME_STACKS

FRONTIER_12 = (
    "deepseek-v3.2",
    "deepseek-v4-10t",
    "deepseek-v4.1-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "qwen3.8-27b",
    "qwen3.8-flash-next",
    "kimi-k3",
    "kimi-k3.1",
    "minimax-m3.1-flash",
    "step-5-preview",
    "gemma-4-31b",
)
# Heterogeneous mixtures: a two-layer slice already contains a structural pair;
# a leading dense stack or a KDA/MLA period needs the first four layers.
FOUR_LAYER = frozenset({
    "deepseek-v3.2", "glm-5.3", "glm-5.3-flash", "kimi-k3", "kimi-k3.1",
    "minimax-m3.1-flash", "step-5-preview",
})


def _has_moe(model: str) -> bool:
    from mindie_sim.model_catalog import load_model
    return any(plan.ffn == "moe" for plan in load_model(model).layer_plans)


@pytest.mark.parametrize("tp", (1, 8, 16))
def test_kimi_k31_auto_paired_sfa_block_size_never_crashes_memory(tp: int):
    raw = compose_case("kimi-k3.1", hardware="a3", tp=tp, ep=tp, num_devices=tp,
                       layers=8)
    parsed = parse_case(raw)
    assert parsed.runtime.kv_block_size >= 544
    memory = Memory(parsed)
    if tp > 1:
        assert memory.capacity[0][0] > 0

    bad = parse_case(compose_case(
        "kimi-k3.1", hardware="a3", tp=8, ep=8, num_devices=8, layers=8,
        block_size=64,
    ))
    with pytest.raises(ModelingError,
                       match="paired_sfa_block_too_small_for_recurrent_page"):
        Memory(bad)


def test_explicit_caller_block_size_is_preserved_verbatim():
    raw = compose_case("qwen3.8-27b", hardware="a3", tp=8, ep=1, num_devices=8,
                       layers=2, block_size=256)
    assert raw["runtime"]["kv_block_size"] == 256


@pytest.mark.parametrize("model", FRONTIER_12)
def test_twelve_frontier_models_monotonic_waterfall(model: str):
    report = compare_feature_tracks(
        model,
        hardware="a3",
        tp=8,
        ep=8 if _has_moe(model) else 1,
        num_devices=8,
        layers=4 if model in FOUR_LAYER else 2,
        phase="decode",
        tokens=1,
        context=1024,
        sequences=2,
        ablate_prs=False,
    )
    waterfall = report["waterfall"]
    merged_ms = waterfall["current_merged"]["step_ms"]
    pr_ms = waterfall["current_unmerged_pr"]["step_ms"]
    ideal_ms = waterfall["expected_ideal"]["step_ms"]
    roof_1x = waterfall["hardware_roofline"]["physical_ideal_1p0x_ms"]

    assert report["monotonic_verified"] is True
    assert merged_ms + 1e-5 >= pr_ms >= ideal_ms - 1e-5
    assert ideal_ms + 1e-5 >= roof_1x
    assert len(waterfall["current_merged"]["active_mechanisms"]) > 0


def test_dsv41_mhc_pr22142_single_pr_what_if_and_hbm_reduction():
    report = compare_feature_tracks(
        "deepseek-v4.1-flash", hardware="a3",
        tp=8, ep=8, num_devices=8, layers=2,
        phase="decode", tokens=1, context=2048, sequences=2,
        ablate_prs=True,
    )
    waterfall = report["waterfall"]
    assert waterfall["current_unmerged_pr"]["step_ms"] < waterfall["current_merged"]["step_ms"]
    assert waterfall["expected_ideal"]["step_ms"] < waterfall["current_unmerged_pr"]["step_ms"]
    # PR-level eliminations reduce traffic. The ideal level additionally
    # materializes the dual BF16 payload that the router/shared branch reads,
    # so its HBM stays within the same bounded envelope instead of inventing a
    # saving by dropping a required output.
    assert waterfall["current_unmerged_pr"]["hbm_bytes"] < waterfall["current_merged"]["hbm_bytes"]
    mhc_rows = [
        row for row in waterfall["current_unmerged_pr"]["single_pr_ablation"]
        if row["mechanism_id"] == "mhc_boundary_fusion_overlap"
    ]
    assert mhc_rows and mhc_rows[0]["ref"] == "vllm-project/vllm-ascend#22142"
    assert mhc_rows[0]["delta_vs_merged_ms"] < 0.0


def test_custom_json_stack_manifest_onboarding_zero_python_edits(tmp_path: Path):
    manifest = {
        "schema": "mindie-sim.stack-manifest/1",
        "name": "custom-xllm-stack",
        "repository": "jd-opensource/xllm",
        "description": "Custom stack manifest loaded from JSON without Python edits",
        "bindings": [
            {"mechanism_id": "aclgraph_decode_capture", "state": "merged",
             "ref": "jd-opensource/xllm@main"},
            {"mechanism_id": "add_rmsnorm_quant_fusion", "state": "merged",
             "ref": "jd-opensource/xllm@main"},
            {"mechanism_id": "gmm_swiglu_quant_fusion", "state": "unmerged_pr",
             "ref": "jd-opensource/xllm#888"},
        ],
    }
    manifest_path = tmp_path / "xllm_stack.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    profile = load_runtime_stack_manifest(manifest_path)
    assert profile.name == "custom-xllm-stack"

    report = compare_feature_tracks(
        "deepseek-v3.2", hardware="a5", runtime_stack=manifest_path,
        tp=8, ep=8, num_devices=8, layers=4, sequences=2,
    )
    assert report["runtime_stack"] == "custom-xllm-stack"
    assert report["monotonic_verified"] is True
    assert any(
        item["ref"] == "jd-opensource/xllm#888"
        for item in report["waterfall"]["current_unmerged_pr"]["active_open_prs"]
    )


def test_builtin_runtime_stacks_are_only_audited_vllm_ascend_and_raw_eager():
    assert set(RUNTIME_STACKS) == {"vllm-ascend", "raw_eager"}


def test_stack_manifest_model_support_honesty(tmp_path: Path):
    manifest = {
        "schema": "mindie-sim.stack-manifest/1",
        "name": "custom-support-stack",
        "repository": "jd-opensource/xllm",
        "description": "Declares merged and open-PR model support explicitly",
        "supported_models": ["deepseek-v3.2"],
        "unmerged_pr_models": ["qwen3.8-flash-next"],
        "bindings": [
            {"mechanism_id": "aclgraph_decode_capture", "state": "merged",
             "ref": "jd-opensource/xllm@main"},
            {"mechanism_id": "add_rmsnorm_quant_fusion", "state": "merged",
             "ref": "jd-opensource/xllm@main"},
            {"mechanism_id": "gmm_swiglu_quant_fusion", "state": "unmerged_pr",
             "ref": "jd-opensource/xllm#888"},
        ],
    }
    manifest_path = tmp_path / "support_stack.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    profile = load_runtime_stack_manifest(manifest_path)
    assert profile.supported_models == ("deepseek-v3.2",)
    assert profile.unmerged_pr_models == ("qwen3.8-flash-next",)
    serialized = profile.to_dict()
    assert serialized["supported_models"] == ["deepseek-v3.2"]
    assert serialized["unmerged_pr_models"] == ["qwen3.8-flash-next"]
    roundtrip = load_runtime_stack_manifest(serialized)
    assert roundtrip.supported_models == profile.supported_models
    assert roundtrip.unmerged_pr_models == profile.unmerged_pr_models

    def _case(model: str, track: str) -> dict:
        return compose_case(
            model, hardware="a5", runtime_stack=manifest_path,
            feature_track=track, tp=8, ep=8, num_devices=8, layers=4,
        )

    for track in ("merged", "unmerged_pr", "expected_ideal"):
        report = _case("deepseek-v3.2", track)["runtime"]["metadata"][
            "feature_track_report"]
        assert report["stack"] == "custom-support-stack"
        assert report["feature_track"] == track

    with pytest.raises(ModelingError) as merged_error:
        _case("qwen3.8-flash-next", "merged")
    assert str(merged_error.value) == (
        "model_not_supported_by_runtime_stack:"
        "custom-support-stack:qwen3.8-flash-next:merged")
    for track in ("unmerged_pr", "expected_ideal"):
        report = _case("qwen3.8-flash-next", track)["runtime"]["metadata"][
            "feature_track_report"]
        assert report["feature_track"] == track

    for track in ("merged", "unmerged_pr"):
        with pytest.raises(ModelingError) as unsupported_error:
            _case("deepseek-v4.1-flash", track)
        assert str(unsupported_error.value) == (
            "model_not_supported_by_runtime_stack:"
            f"custom-support-stack:deepseek-v4.1-flash:{track}")
    ideal_report = _case("deepseek-v4.1-flash", "expected_ideal")["runtime"][
        "metadata"]["feature_track_report"]
    assert ideal_report["feature_track"] == "expected_ideal"


def test_audit_stack_prs_offline_fixture(tmp_path: Path):
    fixture = {
        "repository": "vllm-project/vllm-ascend",
        "open_prs": [
            {"number": 22142, "title": "[Performance] Fuse DeepSeek V4.1 mHC boundaries",
             "state": "OPEN", "labels": [{"name": "module:ops"}]},
            {"number": 99999, "title": "[Performance] Fuse brand-new speculative router",
             "state": "OPEN", "labels": [{"name": "module:ops"}]},
        ],
        "tracked_prs": {
            "22142": {"number": 22142,
                      "title": "[Performance] Fuse DeepSeek V4.1 mHC boundaries",
                      "state": "OPEN", "mergedAt": None},
            "21999": {"number": 21999, "title": "[Performance] Use a plain tensor add",
                      "state": "MERGED", "mergedAt": "2026-11-17T06:00:00Z"},
        },
    }
    fixture_path = tmp_path / "prs.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_stack_prs.py"
    proc = subprocess.run(
        [sys.executable, str(script), "--stack", "vllm-ascend",
         "--offline-fixture", str(fixture_path)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    out = json.loads(proc.stdout)
    assert any(row["pr_number"] == 21999 for row in out["ready_to_promote_to_merged"])
    assert any(row["pr_number"] == 99999
               for row in out["uncataloged_performance_candidates"])


def test_a5_cann93_megamoe_mhc_text_gated_residual_and_chunk32_tracks():
    # 1. MegaMoE(enable_fused_mc2=3) promoted to merged on A5 MXFP MoE.
    megamoe_case = compose_case(
        "qwen3.5-397b-a17b", hardware="a5", feature_track="merged",
        tp=8, ep=8, num_devices=8, layers=4,
    )
    merged_mechs = {
        row["id"]
        for row in megamoe_case["runtime"]["metadata"]["feature_track_report"]["active_mechanisms"]
    }
    assert "cann_megamoe_fusion" in merged_mechs
    assert megamoe_case["runtime"]["metadata"]["additional_config"]["enable_fused_mc2"] == 3

    # 2. #22142 (DeepSeek-V4.1-Flash mhc_boundary_fusion_overlap + A5 mHC) on A5.
    dsv41 = compare_feature_tracks(
        "deepseek-v4.1-flash", hardware="a5",
        tp=8, ep=8, num_devices=8, layers=2,
        phase="decode", tokens=1, context=2048, sequences=2, ablate_prs=True,
    )
    assert dsv41["monotonic_verified"] is True
    assert any(
        row["ref"] == "vllm-project/vllm-ascend#22142" and row["delta_vs_merged_ms"] < 0.0
        for row in dsv41["waterfall"]["current_unmerged_pr"]["single_pr_ablation"]
    )

    # 3. #21794 (Qwen4Exp text_gated_residual_fusion on A5 prefill M > 256, guarded on decode).
    qsa_prefill = compose_case(
        "qwen3.8-flash-next", hardware="a5", feature_track="unmerged_pr",
        _stack_phase="prefill", tp=8, ep=8, num_devices=8, layers=2,
    )
    qsa_decode = compose_case(
        "qwen3.8-flash-next", hardware="a5", feature_track="unmerged_pr",
        _stack_phase="decode", tp=8, ep=8, num_devices=8, layers=2,
    )
    assert qsa_prefill["runtime"]["metadata"].get("text_gated_residual") is True
    assert qsa_decode["runtime"]["metadata"].get("text_gated_residual") is not True

    # 4. #22147 (Chunk32 in unmerged_pr vs Chunk64 in merged on Hv/Hk<=4; Hv/Hk=8 falls back).
    qwen_gdn = compare_feature_tracks(
        "qwen3.8-27b", hardware="a5",
        tp=8, ep=1, num_devices=8, layers=2,
        phase="prefill", tokens=128, context=0, sequences=4, ablate_prs=False,
    )
    assert qwen_gdn["monotonic_verified"] is True
    merged_gdn = compose_case(
        "qwen3.8-27b", hardware="a5", feature_track="merged",
        _stack_phase="prefill", tp=8, ep=1, num_devices=8, layers=2,
    )
    pr_gdn = compose_case(
        "qwen3.8-27b", hardware="a5", feature_track="unmerged_pr",
        _stack_phase="prefill", tp=8, ep=1, num_devices=8, layers=2,
    )
    assert merged_gdn["runtime"]["attention_policy"]["recurrent_chunk"] == 64
    assert pr_gdn["runtime"]["attention_policy"]["recurrent_chunk"] == 32
    assert qwen_gdn["waterfall"]["current_unmerged_pr"]["step_ms"] < qwen_gdn["waterfall"]["current_merged"]["step_ms"]

    gqa8_pr = compose_case(
        "qwen3.8-2.4t-a95b", hardware="a5", feature_track="unmerged_pr",
        _stack_phase="prefill", tp=8, ep=8, num_devices=8, layers=2,
    )
    assert "gdn_prefill" not in (gqa8_pr["runtime"].get("kernels", {}).get("lowerings") or {})
