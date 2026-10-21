"""Targeted tests for the A3 operator shape-linearity campaign.

The tests use the real nine-model configs through the public lowering, but
small representative demand sets so they remain local and fast.  No torch or
NPU is imported, and no dry-run result is accepted as a measured sample.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from mindie_sim.a3_operator_campaign import (
    CampaignConfig, assign_splits, build_campaign, condition_points,
    extract_conditions, load_manifest, write_campaign,
)
from mindie_sim.a3_operator_curves import analyze
from mindie_sim.a3_operator_worker import run_manifest


def _topology(tp=8, ep=1, pp=1, dp=1, devices=8, name="t"):
    return {"name": name, "tp": tp, "ep": ep, "pp": pp, "dp": dp, "devices": devices}


def _config(models, *, budget=40, prefill=(128,), decode=({"batch": 8, "context": 1024},)):
    options = {}
    for key in models:
        options[key] = {
            "topologies": [_topology(tp=8 if key != "kimi-k3-dspark" else 1,
                                     ep=1, pp=1, devices=8)],
            "prefill_tokens": list(prefill),
            "decode_cases": [dict(item) for item in decode],
        }
    return CampaignConfig.from_mapping({
        "campaign_id": "test-campaign", "models": list(models),
        "model_options": options, "budget": budget,
    })


def _condition(**overrides):
    condition = {
        "condition_key": "cond:test", "family": "gemm",
        "recipe": "catalog_dense_gate_up", "phase": "prefill",
        "execution_kind": "primitive", "worker_recipe": "torch_mm_preallocated",
        "backend": "torch.mm", "target_match": "exact_primitive",
        "status": "executable", "reason": "test", "required_backend": "torch",
        "shape_kind": "gemm", "shape": {"m": 37, "n": 128, "k": 256},
        "semantic_shape": [37, 128, 256], "launch_shape": [37, 128, 256],
        "padded_shape": [48, 128, 256], "local_shape": [37, 128, 256],
        "precision": {
            "input_dtype": "bf16", "weight_dtype": "bf16", "output_dtype": "bf16",
            "compute_dtype": "bf16", "accumulation_dtype": "backend_default",
            "quantization": "none", "weight_format": "bf16",
            "activation_format": "bf16", "compute_format": "bf16",
            "weight_representation": "bf16", "quant_category": "",
            "packed_weight_bytes": 128 * 256 * 2.0, "dequant_before_compute": False,
            "weight_block": [0, 0], "expert_block": 32, "scale_format": "fp32",
            "scale_bytes": 4.0, "kv_dtype": "bf16",
        },
        "layout": "bf16", "quantization": "none",
        "features": {"metadata": {}, "parallel": {"tp": 1, "ep": 1, "pp": 1, "dp": 1},
                     "mode": "prefill", "graph_bucket": None, "graph_mode": "eager",
                     "kv_block_size": 128, "moe_recipe": "prefill_causal+dense", "routing": None},
        "state": {"stage": 0, "layer": 0, "requests": ["r0"], "phase": "prefill",
                  "mode": "prefill", "padded_tokens": 37, "valid_tokens": 37,
                  "graph_bucket": None, "architecture": "gqa", "attention": "gqa",
                  "ffn": "dense", "compress_ratio": 0, "window": 0, "indexer": False,
                  "num_heads": 8, "num_kv_heads": 1, "head_dim": 128,
                  "num_experts": 0, "top_k": 0, "moe_intermediate_size": 0,
                  "recurrent_state_elements": 0, "kv_cache_elements": 0, "scope": "body"},
        "boundaries": [{"dim": 0, "value": 16, "source": "lowering_padding_alignment",
                        "launch": 37, "padded": 48}],
        "expansion": {"ai_cpu": None, "aiv": None, "communication": None, "raw": {},
                      "note": "generic only"},
        "work": {"flops": 2.0 * 37 * 128 * 256, "hbm_bytes": (37 * 256 + 128 * 256 + 37 * 128) * 2.0,
                 "mte2_bytes": 0.0, "mte3_bytes": 0.0, "vector_ops": 0.0,
                 "dtype": "bf16", "semantic_shape": [37, 128, 256],
                 "launch_shape": [37, 128, 256]},
        "protocol": {"timer": "device_event", "warmup": 10, "iterations": 30,
                     "trials": 2, "preallocated": True},
        "sources": {},
    }
    condition.update(overrides)
    return condition


def test_real_catalog_extraction_dedup_and_full_inventory():
    config = _config(["qwen3.8-27b", "kimi-k3", "kimi-k3-dspark"], budget=50)
    extraction = extract_conditions(config)
    assert not extraction["failures"], extraction["failures"][:2]
    models = extraction["models"]
    assert set(models) == {"qwen3.8-27b", "kimi-k3", "kimi-k3-dspark"}
    assert models["qwen3.8-27b"]["body_layers"] == 64
    assert models["kimi-k3"]["body_layers"] == 93
    assert models["kimi-k3-dspark"]["body_layers"] == 5
    assert models["qwen3.8-27b"]["hidden_size"] == 5120
    assert models["kimi-k3"]["hidden_size"] == 7168

    conditions = extraction["conditions"]
    families = {condition["family"] for condition in conditions.values()}
    assert {"gemm", "rmsnorm", "swiglu", "elementwise", "moe_gate_up", "attention_qk"} <= families
    # Execution kinds are never collapsed into one another.
    rms = [c for c in conditions.values() if c["family"] == "rmsnorm"]
    assert {c["execution_kind"] for c in rms} == {"fused_kernel", "reference_decomposition"}
    assert any(c["status"] == "pending_backend" for c in conditions.values())

    gemms = [c for c in conditions.values() if c["family"] == "gemm" and c["shape_kind"] == "gemm"]
    assert gemms
    sample = max(gemms, key=lambda c: len(c["sources"]))
    assert set(sample["shape"]) == {"m", "n", "k"}
    assert sample["semantic_shape"] and sample["launch_shape"] and sample["padded_shape"]
    assert sample["precision"]["input_dtype"]
    assert sample["precision"]["weight_dtype"]
    assert sample["precision"]["compute_dtype"]
    assert sample["condition_key"].startswith("cond:")
    assert len(sample["sources"]) >= 1

    # Duplicate topology definitions must merge origins, not duplicate the
    # physical condition or lose the second topology/op lineage.
    duplicate = CampaignConfig.from_mapping({
        "models": ["qwen3.8-27b"],
        "model_options": {"qwen3.8-27b": {
            "topologies": [_topology(name="a"), _topology(name="b")],
            "prefill_tokens": [128], "decode_cases": [],
        }},
        "budget": 10,
    })
    deduped = extract_conditions(duplicate)
    topologies = {
        origin["topology"]
        for condition in deduped["conditions"].values()
        for origin in condition["sources"].values()
    }
    assert {"a", "b"} <= topologies
    assert len(deduped["conditions"]) == len(extract_conditions(CampaignConfig.from_mapping({
        "models": ["qwen3.8-27b"],
        "model_options": {"qwen3.8-27b": {
            "topologies": [_topology(name="a")],
            "prefill_tokens": [128], "decode_cases": [],
        }},
        "budget": 10,
    }))["conditions"])


def test_axis_ladder_boundary_legality_and_precommitted_splits():
    condition = _condition()
    points = condition_points(condition, _config(["qwen3.8-27b"]))
    shapes = {(point["axis"], point["value"]) for point in points}
    actual = [point for point in points if point["actual"]]
    assert len(actual) == 1 and actual[0]["split_hint"] == "train"
    assert ("m", 9) in shapes or ("m", 10) in shapes  # x0/4 ladder is legal/rounded
    assert ("n", 64) in shapes and ("k", 128) in shapes
    # T-2,T,T+2 train; T-1,T+1 heldout around the observed padding boundary.
    assert {("m", 14), ("m", 16), ("m", 18)} <= shapes
    assert {("m", 15), ("m", 17)} <= shapes
    splits = {}
    for point in points:
        point = {**point, "condition_keys": ["cond:test"],
                 "lineage": ["cond:test"]}
        splits[(point["axis"], point["value"], point["point_key"])] = point
    assigned = assign_splits(splits.values())
    by_axis_value = {(p["axis"], p["value"]): p for p in assigned}
    assert by_axis_value[("m", 14)]["split"] == "train"
    assert by_axis_value[("m", 16)]["split"] == "train"
    assert by_axis_value[("m", 18)]["split"] == "train"
    assert by_axis_value[("m", 15)]["split"] == "heldout"
    assert by_axis_value[("m", 17)]["split"] == "heldout"
    assert len({(p["axis"], p["value"], p["point_key"]) for p in assigned}) == len(assigned)

    # Attention query variation holds context fixed instead of silently
    # changing KV; prefill context 0 therefore couples q and kv.
    attention = _condition(
        family="attention_qk", recipe="catalog_gqa_qk", shape_kind="attention",
        shape={"q": 128, "kv": 128, "heads": 8, "dim": 64},
        semantic_shape=[128, 128, 8, 64], launch_shape=[128, 128, 8, 64],
        padded_shape=[128, 128, 8, 64], boundaries=[],
        worker_recipe="torch_attention_matmul", backend="torch.matmul",
    )
    attention_points = condition_points(attention, _config(["qwen3.8-27b"]))
    q32 = [p for p in attention_points if p["axis"] == "q" and p["value"] == 32][0]
    assert q32["shape"]["q"] == 32 and q32["shape"]["kv"] == 32
    q256 = [p for p in attention_points if p["axis"] == "q" and p["value"] == 256][0]
    assert q256["shape"]["kv"] == 256
    assert all(p["shape"]["q"] <= p["shape"]["kv"] for p in attention_points
               if p["axis"] in {"q", "kv"})


def test_global_train_wins_and_duplicate_holdout_is_not_leaked():
    condition = _condition()
    points = condition_points(condition, _config(["qwen3.8-27b"]))
    base = next(point for point in points if point["axis"] == "m" and point["value"] == 16)
    train_point = {**base, "condition_keys": ["cond:train"], "lineage": ["cond:train"],
                   "split_hint": "train", "split_group": "actual"}
    holdout_point = {**base, "point_key": base["point_key"],
                     "condition_keys": ["cond:holdout"], "lineage": ["cond:holdout"],
                     "split_hint": "heldout", "split_group": None}
    assigned = assign_splits([train_point, holdout_point])
    assert len(assigned) == 1
    assert assigned[0]["split"] == "train"
    assert set(assigned[0]["condition_keys"]) == {"cond:train", "cond:holdout"}
    assert set(assigned[0]["lineage"]) == {"cond:train", "cond:holdout"}


def test_budget_pending_and_inventory_are_explicit():
    manifest = build_campaign(_config(["qwen3.8-27b", "kimi-k3", "kimi-k3-dspark"], budget=12))
    assert len(manifest["requests"]) <= 12
    assert manifest["budget"] if "budget" in manifest else manifest["coverage"]["budget"]
    assert manifest["coverage"]["budget"]["limit"] == 12
    assert manifest["coverage"]["selected_train"] > 0
    assert manifest["coverage"]["selected_heldout"] > 0
    pending = manifest["pending_requests"]
    assert pending
    assert all(item["reason"] and item["required_backend"] for item in pending)
    inventory = manifest["inventory"]
    assert inventory
    families = {item["family"] for item in inventory}
    assert {"gemm", "rmsnorm", "swiglu", "moe_gate_up"} <= families
    assert all(item["collection_status"] for item in inventory)
    assert all(item["dimension_samples"] for item in inventory)
    # Every emitted family must be in inventory, including non-GEMM.
    emitted = set(manifest["coverage"]["all_families"])
    assert emitted <= families
    # No request may claim a result before timing.
    assert all(request["status"] == "pending" for request in manifest["requests"])
    assert all(request["request_status"] == "pending" for request in manifest["requests"])


def test_dry_run_is_not_measured_and_real_mode_requires_runner(tmp_path):
    manifest = build_campaign(_config(["qwen3.8-27b"], budget=8))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    results_path = tmp_path / "dry.json"
    payload = run_manifest(manifest_path, results_path, dry_run=True)
    assert payload["status"] == "dry_run"
    assert payload["not_a3_measured"] is True
    assert payload["summary"]["completed"] == 0
    assert payload["summary"]["measured_samples"] == 0
    assert all(item["status"] == "dry_run_validated" for item in payload["results"])
    assert all(item["source_class"] == "derived" for item in payload["results"])
    assert all(not item["raw_samples_ms"] for item in payload["results"])
    from mindie_sim.a3_types import A3InputError
    with pytest.raises(A3InputError):
        run_manifest(manifest_path, results_path, dry_run=False, runner=None)


def _synthetic_results(manifest, values, *, heldout_indices=(3,)):
    requests = []
    results = []
    for index, (x, median) in enumerate(values):
        split = "heldout" if index in heldout_indices else "train"
        request = {
            "measurement_id": f"m{index}", "request_key": f"m{index}",
            "condition_key": "cond:synthetic", "op_family": "gemm",
            "recipe": "catalog_dense_gate_up", "split": split,
            "execution_kind": "primitive", "worker_recipe": "torch_mm_preallocated",
            "backend": "torch.mm",
            "point": {"axis": "m", "value": x, "shape": {"m": x, "n": 128, "k": 256}},
            "analytic_ms": median,
            "work": {"rate_work": 2 * x * 128 * 256, "unit": "flops",
                     "traffic_bytes": x * 256 * 2},
            "protocol": {"timer": "device_event", "warmup": 10, "iterations": 30,
                         "trials": 2, "preallocated": True, "cache_policy": "reuse"},
        }
        requests.append(request)
        results.append({
            "measurement_id": f"m{index}", "request_key": f"m{index}",
            "condition_key": "cond:synthetic", "status": "completed",
            "execution_kind": "primitive", "worker_recipe": "torch_mm_preallocated",
            "backend": "torch.mm", "split": split, "source_class": "measured",
            "analytic_ms": median, "duration_ms": median,
            "raw_samples_ms": [median, median * 1.01, median * 0.99],
            "trials": [
                {"trial_index": 0, "median_ms": median, "raw_samples_ms": [median] * 3},
                {"trial_index": 1, "median_ms": median * 1.001, "raw_samples_ms": [median] * 3},
            ],
            "observed_context": {"observed_api": "torch.mm", "execution_kind": "primitive"},
            "protocol": request["protocol"],
            "correctness": {"status": "passed"},
        })
    manifest = {
        **manifest,
        "requests": requests,
        "inventory": [{"family": "gemm", "recipe": "catalog_dense_gate_up",
                       "collection_status": "selected"}],
        "models": {"synthetic": {"selected_points": len(values)}},
        "coverage": {"selected_train": len(values) - len(heldout_indices),
                     "selected_heldout": len(heldout_indices),
                     "pending_conditions": 0, "derivation_failures": []},
        "pending_requests": [],
    }
    return manifest, {"status": "completed", "results": results}


def test_curves_recover_piecewise_regime_train_only_and_missing_coverage():
    manifest = build_campaign(_config(["qwen3.8-27b"], budget=4))
    xs = [1, 2, 4, 8, 16, 32, 64]
    values = [(x, 2.0 + 0.5 * x if x <= 8 else 2.0 + 4.0 + 0.02 * (x - 8)) for x in xs]
    synthetic_manifest, results = _synthetic_results(manifest, values, heldout_indices=(3,))
    analysis = analyze(synthetic_manifest, results)
    assert analysis["status"] == "measured_analysis"
    group = analysis["analyses"][0]
    assert group["independent_train_points"] == 6
    assert group["independent_heldout_points"] == 1
    assert group["fit"]["kind"] == "piecewise"
    slopes = [segment["slope_ms_per_unit"] for segment in group["fit"]["segments"]]
    assert slopes[0] == pytest.approx(0.5, abs=2e-3)
    assert slopes[1] == pytest.approx(0.02, abs=2e-3)
    assert group["fit"]["transition_brackets"][0][0] <= 8 <= group["fit"]["transition_brackets"][0][1]
    assert group["launch_floor_hint"]["hint"] is True
    assert group["saturation_hint"]["hint"] is True
    heldout = group["heldout_evaluation"]
    assert heldout["count"] == 1
    assert heldout["scored_count"] == 0
    assert heldout["transition_count"] == 1
    assert heldout["points"][0]["predicted_ms"] is None
    assert heldout["points"][0]["region"] == "transition_bracket"
    assert heldout["points"][0]["lower_bound_ms"] is not None
    assert heldout["points"][0]["upper_bound_ms"] >= heldout["points"][0]["lower_bound_ms"]
    assert group["independent_train_trials"] == 12
    assert group["raw_train_samples"] == 36

    # Changing heldout outcomes cannot change the train-only fit.
    changed = copy.deepcopy(results)
    for result in changed["results"]:
        if result["measurement_id"] == "m3":
            result["duration_ms"] = 999.0
            result["raw_samples_ms"] = [999.0] * 3
            for trial in result["trials"]:
                trial["median_ms"] = 999.0
                trial["raw_samples_ms"] = [999.0] * 3
    changed_analysis = analyze(synthetic_manifest, changed)
    assert changed_analysis["analyses"][0]["fit"] == group["fit"]
    assert changed_analysis["analyses"][0]["heldout_evaluation"]["scored_count"] == 0

    # A pending result must remain visible with its reason/backend.
    pending = copy.deepcopy(changed)
    pending["results"].append({
        "measurement_id": "pending-1", "condition_key": "cond:synthetic",
        "status": "pending", "reason": "verified grouped kernel unavailable",
        "required_backend": "torch_npu.npu_grouped_matmul", "execution_kind": "fused_kernel",
        "worker_recipe": None, "split": "train",
    })
    pending_analysis = analyze(synthetic_manifest, pending)
    assert any(item.get("required_backend") == "torch_npu.npu_grouped_matmul"
               for item in pending_analysis["coverage"]["missing"])
    assert pending_analysis["summary"]["pending_results"] == 1


def test_no_pooling_across_dtype_or_protocol():
    manifest = build_campaign(_config(["qwen3.8-27b"], budget=4))
    base_values = [(1, 1.0), (2, 2.0), (4, 4.0), (8, 8.0)]
    first_manifest, first_results = _synthetic_results(manifest, base_values, heldout_indices=())
    second_manifest = copy.deepcopy(first_manifest)
    second_results = copy.deepcopy(first_results)
    for index, request in enumerate(second_manifest["requests"]):
        request["condition_key"] = "cond:synthetic-b"
        request["measurement_id"] = request["request_key"] = f"b{index}"
    for index, result in enumerate(second_results["results"]):
        result["condition_key"] = "cond:synthetic-b"
        result["measurement_id"] = result["request_key"] = f"b{index}"
        result["status"] = "completed"
        result["duration_ms"] *= 2.0
        result["raw_samples_ms"] = [value * 2.0 for value in result["raw_samples_ms"]]
        for trial in result["trials"]:
            trial["median_ms"] *= 2.0
            trial["raw_samples_ms"] = [value * 2.0 for value in trial["raw_samples_ms"]]
    combined = {
        "status": "completed",
        "results": first_results["results"] + second_results["results"],
    }
    combined_manifest = {
        **first_manifest,
        "requests": first_manifest["requests"] + second_manifest["requests"],
    }
    analysis = analyze(combined_manifest, combined)
    assert len(analysis["analyses"]) == 2
    keys = {group["condition_key"] for group in analysis["analyses"]}
    assert keys == {"cond:synthetic", "cond:synthetic-b"}
    slopes = {group["condition_key"]: group["fit"]["segments"][0]["slope_ms_per_unit"]
              for group in analysis["analyses"]}
    assert slopes["cond:synthetic"] == pytest.approx(1.0, abs=2e-3)
    assert slopes["cond:synthetic-b"] == pytest.approx(2.0, abs=2e-3)


def test_cli_plan_dry_run_and_analyze(tmp_path):
    from mindie_sim.a3_operator_campaign import main as campaign_main
    from mindie_sim.a3_operator_curves import main as curves_main
    from mindie_sim.a3_operator_worker import main as worker_main

    config_path = tmp_path / "config.json"
    config = _config(["qwen3.8-27b"], budget=6).to_dict()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "campaign"
    assert campaign_main(["plan", "--config", str(config_path),
                          "--output-dir", str(out), "--budget", "6"]) == 0
    manifest = load_manifest(out / "manifest.json")
    assert manifest["coverage"]["budget"]["limit"] == 6
    assert (out / "inventory.json").exists()
    assert (out / "coverage.md").exists()
    assert (out / "unmeasured.json").exists()
    results = tmp_path / "dry.json"
    assert worker_main(["--manifest", str(out / "manifest.json"),
                        "--results", str(results), "--dry-run"]) == 0
    payload = json.loads(results.read_text(encoding="utf-8"))
    assert payload["summary"]["completed"] == 0
    assert payload["summary"]["dry_run_validated"] == len(manifest["requests"])
    analysis_dir = tmp_path / "analysis"
    assert curves_main(["--manifest", str(out / "manifest.json"),
                        "--results", str(results),
                        "--output-dir", str(analysis_dir)]) == 0
    analysis = json.loads((analysis_dir / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["status"] == "dry_run_no_measured_samples"
    assert analysis["summary"]["raw_measured_samples"] == 0


def test_physical_point_identity_ignores_model_family_and_layer_origin():
    first = _condition(family="qkv_projection", recipe="gqa_qkv")
    second = _condition(family="gate_up_projection", recipe="catalog_dense_gate_up",
                        condition_key="cond:other")
    second["features"]["metadata"] = {"owner_layer": 42, "semantic_family": "different"}
    second["state"] = {**second["state"], "layer": 42}
    from mindie_sim.a3_operator_campaign import _point_key
    point = {"shape": {"m": 37, "n": 128, "k": 256},
             "semantic_shape": [37, 128, 256], "launch_shape": [37, 128, 256],
             "padded_shape": [48, 128, 256]}
    assert _point_key(first, point) == _point_key(second, point)
    assigned = assign_splits([
        {**point, "point_key": _point_key(first, point), "axis": "m", "value": 37,
         "actual": True, "boundary_probe": False, "extrapolation": False,
         "split_hint": "train", "split_group": "actual", "condition_keys": ["cond:a"],
         "lineage": ["a"]},
        {**point, "point_key": _point_key(second, point), "axis": "m", "value": 37,
         "actual": True, "boundary_probe": False, "extrapolation": False,
         "split_hint": "heldout", "split_group": None, "condition_keys": ["cond:b"],
         "lineage": ["b"]},
    ])
    assert len(assigned) == 1 and assigned[0]["split"] == "train"
    assert set(assigned[0]["condition_keys"]) == {"cond:a", "cond:b"}


def test_worker_check_rows_uses_already_selected_reference_rows():
    from types import SimpleNamespace
    from mindie_sim.a3_operator_worker import TorchNpuOperatorRunner

    runner = TorchNpuOperatorRunner.__new__(TorchNpuOperatorRunner)
    runner._f32_cpu = lambda value: value
    runner.torch = SimpleNamespace(
        allclose=lambda reference, actual, rtol, atol: bool(
            np.allclose(reference.value, actual.value, rtol=rtol, atol=atol)))
    import numpy as np

    class Fake:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=float)
        def __getitem__(self, index):
            return Fake(self.value[index])
        def detach(self): return self
        def float(self): return self
        def cpu(self): return self
        def __sub__(self, other): return Fake(self.value - other.value)
        def abs(self): return Fake(np.abs(self.value))
        def max(self): return Fake(np.max(self.value))
        def numel(self): return int(self.value.size)
        def item(self): return float(self.value)

    reference = Fake([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    rows = [[0.0, 0.0]] * 128
    rows[0], rows[64], rows[127] = [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]
    got = Fake(rows)
    check = runner._check_rows(reference, got, [0, 64, 127],
                               rtol=1e-3, atol=1e-3, name="row_alignment")
    assert check["status"] == "passed"
    assert check["rows"] == [0, 64, 127]


def test_analysis_rejects_overlap_duplicate_replay_and_failed_correctness(tmp_path):
    manifest = build_campaign(_config(["qwen3.8-27b"], budget=4))
    requests = []
    results = []
    for index, (x, split) in enumerate([(1.0, "train"), (2.0, "train"), (2.0, "heldout")]):
        request = {
            "measurement_id": f"o{index}", "request_key": f"o{index}",
            "condition_key": "cond:overlap", "op_family": "gemm",
            "recipe": "catalog_dense_gate_up", "split": split,
            "execution_kind": "primitive", "worker_recipe": "torch_mm_preallocated",
            "backend": "torch.mm",
            "condition": {"shape": {"m": 2, "n": 128, "k": 256}},
            "point": {"axis": "m", "value": x, "shape": {"m": x, "n": 128, "k": 256}},
            "analytic_ms": 1.0,
            "work": {"rate_work": x, "unit": "flops"},
            "protocol": {"timer": "device_event", "warmup": 10, "iterations": 30,
                         "trials": 2, "preallocated": True, "cache_policy": "reuse"},
        }
        requests.append(request)
        results.append({
            "measurement_id": f"o{index}", "request_key": f"o{index}",
            "condition_key": "cond:overlap", "status": "completed",
            "execution_kind": "primitive", "worker_recipe": "torch_mm_preallocated",
            "backend": "torch.mm", "split": split, "source_class": "measured",
            "analytic_ms": 1.0, "raw_samples_ms": [1.0, 1.0], "duration_ms": 1.0,
            "trials": [{"trial_index": 0, "median_ms": 1.0, "raw_samples_ms": [1.0]},
                       {"trial_index": 1, "median_ms": 1.0, "raw_samples_ms": [1.0]}],
            "observed_context": {"observed_api": "torch.mm", "execution_kind": "primitive"},
            "protocol": request["protocol"], "correctness": {"status": "passed"},
        })
    # Exact duplicate replay must not inflate independent N.
    results.append(copy.deepcopy(results[0]))
    # A failed correctness row must be counted as failed, never fitted.
    failed = copy.deepcopy(results[1])
    failed["measurement_id"] = failed["request_key"] = "failed"
    failed["correctness"] = {"status": "failed", "reason": "sample mismatch"}
    results.append(failed)
    manifest = {
        **manifest, "requests": requests,
        "inventory": [{"family": "gemm", "recipe": "catalog_dense_gate_up",
                       "collection_status": "selected"}],
        "models": {"synthetic": {"selected_points": 3}},
        "coverage": {"selected_train": 2, "selected_heldout": 1,
                     "pending_conditions": 0, "derivation_failures": []},
        "pending_requests": [],
    }
    analysis = analyze(manifest, {"status": "completed", "results": results})
    assert analysis["summary"]["duplicate_result_rows_ignored"] == 1
    assert analysis["summary"]["completed_results"] == 3
    assert analysis["summary"]["failed_results"] == 1
    assert analysis["coverage"]["split_overlap_count"] == 1
    group = analysis["analyses"][0]
    assert group["status"] == "rejected_split_overlap"
    assert group["fit"] is None
    assert group["split_overlap_x"] == [2.0]
    assert len(group["train_points"]) == 2
    # 1 train + 1 heldout must still write both artifacts without KeyError.
    small_manifest = {**manifest, "requests": requests[:1],
                      "coverage": {**manifest["coverage"], "selected_heldout": 1}}
    small_results = {"status": "completed", "results": [results[0], results[2]]}
    out = tmp_path / "small"
    from mindie_sim.a3_operator_curves import analyze_files
    small = analyze_files(tmp_path / "manifest.json", tmp_path / "results.json",
                          output_dir=out) if False else None
    # Direct analyzer path covers the small-case metric guard.
    small_analysis = analyze(small_manifest, small_results)
    assert small_analysis["analyses"][0]["heldout_evaluation"]["status"] in {
        "not_fitted_insufficient_train_points", "rejected_split_overlap",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(small_manifest), encoding="utf-8")
    (tmp_path / "results.json").write_text(json.dumps(small_results), encoding="utf-8")
    analyze_files(tmp_path / "manifest.json", tmp_path / "results.json", output_dir=out)
    assert (out / "analysis.json").exists()
    assert (out / "analysis.md").exists()


def test_normalized_conditions_export_is_pre_use_and_unique():
    from mindie_sim.a3_operator_curves import normalized_conditions
    manifest = build_campaign(_config(["qwen3.8-27b"], budget=6))
    rows = normalized_conditions(manifest, {"status": "dry_run", "results": []})["conditions"]
    assert rows
    identities = set()
    for row in rows:
        assert "observed_api" not in row["query_context"]
        assert row["query_context"]["phase"] in {"prefill", "decode"}
        identity = json.dumps([row["query_context"], row["shape"]], sort_keys=True)
        assert identity not in identities
        identities.add(identity)
        assert row["samples"]


def test_linearity_split_adapter_keeps_repeats_as_one_independent_measurement(tmp_path):
    from mindie_sim.a3_operator_curves import analyze_linearity_files
    payload = {
        "split": "train",
        "cases": [
            {
                "request": {
                    "case_id": f"c{index}", "op_family": "add",
                    "shape": {"elements": value}, "dtype": "bf16",
                    "layout": "contiguous",
                    "features": {"weight_dtype": "bf16", "activation_dtype": "bf16",
                                 "compute_dtype": "bf16",
                                 "accumulation_dtype": "backend_default",
                                 "quantization": "none"},
                    "sweeps": [{"axis": "elements"}],
                },
                "campaign_split": "train",
                "production_result": {
                    "status": "completed", "raw_samples_ms": [1.0] * 30,
                    "duration_ms": 1.0, "analytic_ms": 1.0,
                    "source_class": "measured", "timing_scope": "device_event_warm",
                },
            }
            for index, value in enumerate([1024, 2048, 4096])
        ],
    }
    input_path = tmp_path / "train.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    analysis = analyze_linearity_files(input_path, output_dir=tmp_path / "out")
    assert analysis["summary"]["completed_results"] == 3
    assert analysis["summary"]["raw_measured_samples"] == 90
    group = analysis["analyses"][0]
    assert group["independent_train_points"] == 3
    assert group["independent_train_trials"] == 3  # one trial per measurement, not 30
    assert group["raw_train_samples"] == 90
    assert group["fit"]["segments"][0]["slope_ms_per_unit"] == pytest.approx(0.0)
    assert (tmp_path / "out" / "adapted-manifest.json").exists()
    assert (tmp_path / "out" / "adapted-results.json").exists()
    calibration = json.loads((tmp_path / "out" / "calibration-groups.json").read_text())
    assert len(calibration["groups"]) == 1
    assert all(case["kind"] == "add" for case in calibration["groups"][0]["cases"])
    assert "analytic_ms" not in calibration["groups"][0]["cases"][0]


def test_complete_condition_persistence_and_pending_materialization(tmp_path):
    from mindie_sim.a3_operator_campaign import (
        condition_points, materialize_conditions,
    )
    config = CampaignConfig.from_mapping({
        "campaign_id": "persistence-test", "models": ["qwen3.8-27b"],
        "model_options": {"qwen3.8-27b": {
            "topologies": [_topology(tp=8, name="tp8")],
            "prefill_tokens": [128, 512],
            "decode_cases": [{"batch": 8, "context": 1024},
                             {"batch": 32, "context": 8192}],
        }},
        "budget": 4,
    })
    extraction = extract_conditions(config)
    manifest = build_campaign(config)
    paths = write_campaign(manifest, tmp_path / "campaign")
    from mindie_sim.a3_operator_campaign import load_conditions
    conditions = load_conditions(paths["conditions"])
    assert set(conditions) == set(extraction["conditions"])
    assert len(conditions) == len(manifest["conditions"])
    # The layer template normalizes actual layer ids; it is not 16/64 variants.
    templates = {origin["op_id_template"] for record in conditions.values()
                 for origin in record["sources"]}
    assert "s0.l{layer}.qkv" in templates or any(
        template.endswith(".l{layer}.qkv") for template in templates)
    assert not any(template.endswith(".l51.qkv") for template in templates)
    pending = [key for key, record in conditions.items()
               if record["status"] in {"pending_backend", "unsupported"}]
    assert pending
    for key in pending[:8]:
        materialized = condition_points(conditions[key], config)
        assert materialized
        assert all(point["expected_status"] in {"pending_backend", "unsupported"}
                   for point in materialized)
        requests = materialize_conditions(conditions, [key], config)
        assert requests and requests[0]["status"] == "pending_backend"
        assert requests[0]["condition_key"] == key
    # Selected source lineage is not cut at 12 origins.
    for request in manifest["requests"]:
        full = extraction["conditions"][request["condition_key"]]
        assert len(request["source_cases"]) == len(full["sources"])


def test_planner_copy_bytes_matches_uint8_worker_contract():
    condition = _condition(
        family="cache_store", recipe="catalog_mla_store", shape_kind="cache",
        shape={"rows": 4, "row_bytes": 128}, semantic_shape=[4, 128],
        launch_shape=[4, 128], padded_shape=[4, 128], boundaries=[],
        worker_recipe="copy_rows_reference",
    )
    points = condition_points(condition, _config(["qwen3.8-27b"]))
    actual = next(point for point in points if point["actual"])
    assert actual["work"]["hbm_bytes"] == 1024.0  # read 512 + write 512 bytes
    assert actual["work"]["traffic_bytes"] == 1024.0
    assert actual["work"]["rate_work"] == 512.0
    assert actual["working_set_bytes"] == 1024


def test_grouped_histogram_uses_actual_point_not_anchor():
    from mindie_sim.a3_operator_worker import _scaled_expert_counts
    counts, source = _scaled_expert_counts([4, 4, 4, 4], 4, 16)
    assert counts == [4, 4, 4, 4] and source == "anchor_actual_histogram"
    counts, source = _scaled_expert_counts([4, 4, 4, 4], 4, 32)
    assert sum(counts) == 32 and source.startswith("actual_point")
    counts, source = _scaled_expert_counts([8, 4, 2, 2], 8, 32)
    assert len(counts) == 8 and sum(counts) == 32
    assert source.startswith("actual_point")


def test_dynamic_quant_scale_broadcast_contract():
    from mindie_sim.a3_operator_worker import _quant_scale_mode
    assert _quant_scale_mode(8, 8) == "row_unsqueeze"
    assert _quant_scale_mode(1, 8) == "scalar"
    assert _quant_scale_mode(128, 8) is None


def test_independent_runs_same_trial_index_and_protocol_separation():
    manifest = build_campaign(_config(["qwen3.8-27b"], budget=4))
    request = {
        "measurement_id": "run-x", "request_key": "run-x", "condition_key": "cond:run",
        "op_family": "gemm", "recipe": "catalog_dense_gate_up", "split": "train",
        "execution_kind": "primitive", "worker_recipe": "torch_mm_preallocated",
        "backend": "torch.mm", "hardware_key": "Ascend910_9362",
        "condition": {"shape": {"m": 2, "n": 128, "k": 256}, "shape_kind": "gemm"},
        "point": {"axis": "m", "value": 2, "shape": {"m": 2, "n": 128, "k": 256}},
        "analytic_ms": 1.0, "work": {"rate_work": 1.0, "unit": "flops"},
        "protocol": {"timer": "device_event", "warmup": 10, "iterations": 30,
                     "trials": 2, "preallocated": True, "cache_policy": "reuse"},
    }
    def result(run_id, duration, iterations=30):
        return {
            "measurement_id": "run-x", "request_key": "run-x",
            "condition_key": "cond:run", "status": "completed",
            "run_id": run_id, "execution_kind": "primitive",
            "worker_recipe": "torch_mm_preallocated", "backend": "torch.mm",
            "split": "train", "analytic_ms": None, "source_class": "measured",
            "raw_samples_ms": [duration, duration], "duration_ms": duration,
            "trials": [{"trial_index": 0, "median_ms": duration,
                        "raw_samples_ms": [duration, duration]}],
            "observed_context": {"observed_api": "torch.mm", "execution_kind": "primitive"},
            "protocol": {**request["protocol"], "iterations": iterations},
            "correctness": {"status": "passed"},
        }
    manifest = {
        **manifest, "requests": [request],
        "inventory": [{"family": "gemm", "recipe": "catalog_dense_gate_up",
                       "collection_status": "selected"}],
        "models": {}, "coverage": {"selected_train": 1, "selected_heldout": 0,
                                   "pending_conditions": 0, "derivation_failures": []},
        "pending_requests": [],
    }
    # Two independent runs with the same trial_index 0 must both count.
    analysis = analyze(manifest, {"status": "completed", "results": [
        result("run-1", 1.0), result("run-2", 1.1),
    ]})
    group = analysis["analyses"][0]
    assert analysis["summary"]["completed_results"] == 2
    assert group["independent_train_points"] == 1
    assert group["independent_train_trials"] == 2
    assert group["train_points"][0]["p90_ms"] == pytest.approx(1.1, rel=1e-6)
    # A protocol change is a different group and cannot be mixed into the fit.
    changed = analyze(manifest, {"status": "completed", "results": [
        result("run-1", 1.0), result("run-2", 1.1, iterations=31),
    ]})
    assert changed["summary"]["completed_results"] == 2
    assert len(changed["analyses"]) == 2


def test_linearity_calibration_gemm_uses_mkn_order_and_no_measurement_alias():
    from mindie_sim.a3_operator_curves import linearity_calibration_groups
    payload = {"cases": [{
        "request": {"case_id": "g1", "op_family": "gemm",
                    "shape": {"m": 8, "n": 256, "k": 128},
                    "dtype": "bf16", "layout": "contiguous",
                    "features": {"weight_dtype": "bf16"}},
        "campaign_split": "train",
        "production_result": {"status": "completed", "raw_samples_ms": [1.0, 2.0],
                              "duration_ms": 1.5, "analytic_ms": 1.5,
                              "source_class": "measured",
                              "environment": {"device_name": "Ascend910_9362"}},
    }]}
    calibration = linearity_calibration_groups(payload)
    assert len(calibration["groups"]) == 1
    case = calibration["groups"][0]["cases"][0]
    assert case["kind"] == "matmul"
    assert case["shape_mkn"] == [8, 128, 256]  # importer expects M,K,N
    assert "analytic_ms" not in case
    assert case["coverage_scope"] == "native_operator_semantic"
    assert case["native_precision_claim"] is True


def test_derived_cpu_evidence_is_not_fitted_as_a3_measurement():
    manifest = build_campaign(_config(["qwen3.8-27b"], budget=4))
    values = [(1, 1.0), (2, 2.0), (3, 3.0)]
    synthetic_manifest, results = _synthetic_results(manifest, values, heldout_indices=())
    for result in results["results"]:
        result["source_class"] = "derived"
        result["environment"] = {"not_a3_measured": True}
    analysis = analyze(synthetic_manifest, results)
    assert analysis["status"] == "derived_only_no_a3_measurement"
    assert analysis["analyses"] == []
    assert analysis["summary"]["completed_results"] == 0
    assert analysis["summary"]["derived_or_unmeasured_excluded"] == 3
    assert any(item["status"] == "derived_or_unmeasured_excluded"
               for item in analysis["coverage"]["missing"])


def test_linearity_adapter_preserves_protocol_and_source_evidence(tmp_path):
    from mindie_sim.a3_operator_curves import analyze_linearity_files
    payload = {"cases": [{
        "request": {"case_id": "ev1", "op_family": "gemm",
                    "shape": {"m": 8, "n": 256, "k": 128},
                    "dtype": "bf16", "layout": "contiguous", "run_id": "run-9",
                    "features": {"weight_dtype": "bf16"}},
        "campaign_split": "train",
        "production_result": {
            "status": "completed", "raw_samples_ms": [1.0, 2.0],
            "duration_ms": 1.5, "analytic_ms": 1.5,
            "source": "managed_runner", "source_class": "measured",
            "run_id": "run-9", "timing_scope": "device_event_warm",
            "environment": {"device_name": "Ascend910_9362"},
            "protocol": {"timer": "device_event", "warmup": 10,
                         "iterations": 30, "preallocated": True,
                         "cache_policy": "reuse"},
        },
    }]}
    input_path = tmp_path / "train.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    analysis = analyze_linearity_files(input_path, output_dir=tmp_path / "out")
    manifest = json.loads((tmp_path / "out" / "adapted-manifest.json").read_text())
    results = json.loads((tmp_path / "out" / "adapted-results.json").read_text())
    request = manifest["requests"][0]
    result = results["results"][0]
    assert request["protocol"]["warmup"] == 10
    assert request["protocol"]["preallocated"] is True
    assert request["run_id"] == result["run_id"] == "run-9"
    assert result["source_class"] == "measured"
    assert result["environment"]["device_name"] == "Ascend910_9362"
    assert analysis["status"] == "measured_analysis"


def test_normalized_export_preserves_cross_split_conflict():
    from mindie_sim.a3_operator_curves import normalized_conditions
    manifest = build_campaign(_config(["qwen3.8-27b"], budget=4))
    base = dict(manifest["requests"][0])
    base["measurement_id"] = base["request_key"] = "a"
    base["split"] = "train"
    other = copy.deepcopy(base)
    other["measurement_id"] = other["request_key"] = "b"
    other["split"] = "heldout"
    manifest = {**manifest, "requests": [base, other]}
    exported = normalized_conditions(manifest, {"results": []})
    assert exported["valid"] is False
    assert exported["conflicts"]
    assert len(exported["conditions"]) == 2
    assert {row["split"] for row in exported["conditions"]} == {"train", "heldout"}


def test_native_softmax_geometry_matches_official_nchw_bounds():
    from mindie_sim.a3_operator_worker_native import _native_softmax_geometry
    assert _native_softmax_geometry(32, 32)
    assert _native_softmax_geometry(128, 512)
    assert not _native_softmax_geometry(1, 128)
    assert not _native_softmax_geometry(33, 128)
    assert not _native_softmax_geometry(8192, 128)


def test_native_wave2_adapter_uses_predeclared_origin_axis_and_fixed_shape(tmp_path):
    from mindie_sim.a3_operator_curves_adapter import (
        adapt_campaign_manifest_results, analyze_campaign_files,
    )
    requests, results = [], []
    def add(mid, shape, axis, anchor_width, value, duration):
        condition = {
            "family": "rmsnorm", "worker_recipe": "npu_rms_norm",
            "execution_kind": "fused_kernel", "status": "ready",
            "shape_kind": "rmsnorm",
            "precision": {"input_dtype": "bf16", "compute_dtype": "bf16",
                          "accumulation_dtype": "backend_default"},
            "layout": "contiguous", "features": {},
        }
        protocol = {"warmup": 10, "iterations": 30, "trials": 2,
                    "timer": "device_event", "seed": 20261021}
        request = {
            "request_key": mid, "measurement_id": mid, "condition_key": mid,
            "point": {"shape": shape}, "condition": condition,
            "worker_recipe": "npu_rms_norm", "execution_kind": "fused_kernel",
            "backend": "torch_npu_installed_api", "split": "train",
            "run_id": "run-wave2", "protocol": protocol,
            "query_context": {"scope": "native_primitive_at_source_declared_2d_geometry"},
            "origins": [{"anchor_family": "rmsnorm", "anchor_width": anchor_width,
                         "axis": axis, "models": ["qwen3.8-27b"],
                         "source_row_ids": ["r1"], "exact_source_row_ids": ["r1"]}],
            "work": {"logical_input_elements": shape.get("tokens", 1) * shape.get("width", 1)},
        }
        requests.append(request)
        results.append({
            "measurement_id": mid, "request_key": mid, "status": "completed",
            "condition_key": mid, "duration_ms": duration,
            "raw_samples_ms": [duration] * 30,
            "trials": [{"trial_index": i, "median_ms": duration,
                        "raw_samples_ms": [duration] * 30} for i in range(2)],
            "source_class": "measured", "source": "torch_npu_operator_worker",
            "run_id": "run-wave2", "split": "train",
            "environment": {"device_name": "Ascend910_9362"},
            "protocol": {**protocol, "preallocated": False},
            "observed_context": {"observed_api": "torch_npu.npu_rms_norm",
                                 "execution_kind": "fused_kernel"},
            "correctness": {"status": "passed"},
        })
    for tokens in [1, 2, 4]:
        shape = {"tokens": tokens, "width": 128}
        add(f"t{tokens}", shape, "tokens", 128, tokens, 1.0 + tokens)
    for width in [127, 129]:
        shape = {"tokens": 128, "width": width}
        add(f"w{width}", shape, "width_neighbor", width, width, 2.0)
    source_manifest = {
        "run_id": "run-wave2", "requests": requests,
        "excluded_rows": [{"reason": "unsupported"}],
    }
    source_results = {"status": "completed", "results": results, "run_id": "run-wave2"}
    adapted_manifest, adapted_results = adapt_campaign_manifest_results(
        source_manifest, source_results)
    assert len({r["condition_key"] for r in adapted_manifest["requests"]}) == 2
    assert {r["point"]["axis"] for r in adapted_manifest["requests"]} == {"tokens", "width"}
    analysis = analyze_campaign_files(
        tmp_path / "manifest.json", tmp_path / "results.json",
        output_dir=tmp_path / "out") if False else None
    # Direct analysis on the adapted products (CLI path is exercised below).
    from mindie_sim.a3_operator_curves import analyze
    analysis = analyze(adapted_manifest, adapted_results)
    assert analysis["summary"]["groups"] == 2
    counts = sorted(group["independent_train_points"] for group in analysis["analyses"])
    assert counts == [2, 3]
    assert all(group["axis"] in {"tokens", "width"} for group in analysis["analyses"])
    assert all(group["independent_train_points"] >= 2 for group in analysis["analyses"])
    # CLI product path writes analysis plus excluded native calibration evidence.
    manifest_path = tmp_path / "wave-manifest.json"
    results_path = tmp_path / "wave-results.json"
    manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")
    results_path.write_text(json.dumps(source_results), encoding="utf-8")
    from mindie_sim.a3_operator_curves_adapter import analyze_campaign_files
    product = analyze_campaign_files(manifest_path, results_path, output_dir=tmp_path / "product")
    assert product["summary"]["groups"] == 2
    assert (tmp_path / "product" / "native-calibration-excluded-cases.json").exists()
    assert (tmp_path / "product" / "normalized-conditions.json").exists()


def test_semantic_mappings_do_not_substitute_rms_or_softmax():
    config = CampaignConfig.from_mapping({
        "models": ["qwen3.8-27b", "glm-5.3", "deepseek-v4-flash-0731"],
        "model_options": {
            "qwen3.8-27b": {"topologies": [_topology(tp=8, name="q")],
                            "prefill_tokens": [128], "decode_cases": []},
            "glm-5.3": {"topologies": [_topology(tp=8, ep=8, pp=2, name="g")],
                        "prefill_tokens": [128], "decode_cases": []},
            "deepseek-v4-flash-0731": {"topologies": [_topology(tp=8, ep=8, name="d")],
                                       "prefill_tokens": [128], "decode_cases": []},
        },
        "budget": 10,
    })
    extraction = extract_conditions(config)
    conditions = extraction["conditions"]
    qk = [c for c in conditions.values() if c["family"] == "qk_norm"]
    assert qk and all(c["worker_recipe"] != "npu_rms_norm" for c in qk)
    assert all(c["status"] == "pending_backend" for c in qk)
    router = [c for c in conditions.values()
              if c["family"] == "router_softmax"
              and "topk" in str(c["features"]["metadata"].get("semantic_family"))]
    assert router and all(c["worker_recipe"] is None for c in router)
    quant = [c for c in conditions.values() if c["family"] == "quantize"]
    assert quant
    assert any(c["worker_recipe"] == "int8_per_row_quant_reference"
               for c in quant)
    assert all(c["worker_recipe"] != "npu_dynamic_quant" for c in quant)
    local = max(quant, key=lambda c: c["precision"].get("scale_bytes") or 0)
    assert local["precision"]["input_dtype"] == "bf16"
    assert local["precision"]["output_dtype"] == "int8"
    assert local["precision"]["scale_format"] == "fp16"
    assert local["precision"]["scale_bytes"] == 2.0
    assert local["precision"]["scale_axis"] == "per_row"
    assert local["precision"]["scale_shape"]


def test_normalized_export_preserves_independent_runs_and_trials():
    from mindie_sim.a3_operator_curves import normalized_conditions
    manifest = build_campaign(_config(["qwen3.8-27b"], budget=4))
    request = {
        "measurement_id": "m", "request_key": "m", "condition_key": "cond:m",
        "op_family": "gemm", "phase": "prefill", "recipe": "r", "split": "train",
        "execution_kind": "primitive", "worker_recipe": "torch_mm_preallocated",
        "backend": "torch.mm", "hardware_key": "Device-A",
        "condition": {"shape": {"m": 2, "n": 128, "k": 256},
                      "quantization": "none",
                      "precision": {"input_dtype": "bf16", "weight_dtype": "bf16",
                                    "output_dtype": "bf16", "compute_dtype": "bf16",
                                    "accumulation_dtype": "backend_default"}},
        "point": {"axis": "m", "value": 2, "shape": {"m": 2, "n": 128, "k": 256}},
        "query_context": {}, "protocol": {"timer": "device_event", "warmup": 10,
                                          "iterations": 30, "trials": 2,
                                          "preallocated": True},
        "run_id": "run-a", "analytic_ms": None, "work": None,
    }
    def result(run):
        return {
            "measurement_id": "m", "request_key": "m", "status": "completed",
            "run_id": run, "source_class": "measured", "source": "worker",
            "split": "train", "condition_key": "cond:m",
            "environment": {"device_name": "Device-A"},
            "protocol": request["protocol"],
            "trials": [{"trial_index": 0, "median_ms": 1.0},
                       {"trial_index": 1, "median_ms": 1.1}],
        }
    manifest = {**manifest, "requests": [request]}
    exported = normalized_conditions(manifest, {"results": [result("run-a"), result("run-b")]})
    row = exported["conditions"][0]
    assert [sample["trial_id"] for sample in row["samples"]] == [
        "run-a:0", "run-a:1", "run-b:0", "run-b:1"]
    assert row["run_ids"] == ["run-a", "run-b"]
    assert len(row["execution_evidence"]) == 2


def test_timing_scope_and_hardware_do_not_pool():
    manifest = build_campaign(_config(["qwen3.8-27b"], budget=4))
    base = {
        "condition_key": "cond:iso", "op_family": "gemm", "phase": "prefill",
        "recipe": "r", "split": "train", "execution_kind": "primitive",
        "worker_recipe": "torch_mm_preallocated", "backend": "torch.mm",
        "condition": {"shape": {"m": 2, "n": 128, "k": 256}},
        "point": {"axis": "m", "value": 2, "shape": {"m": 2, "n": 128, "k": 256}},
        "analytic_ms": None, "work": None,
    }
    requests, results = [], []
    for index, (hardware, scope, duration) in enumerate([
        ("Device-A", "per_operator", 1.0),
        ("Device-B", "per_operator", 100.0),
        ("Device-A", "total_repeated_loop", 30.0),
    ]):
        request = {**base, "measurement_id": f"i{index}", "request_key": f"i{index}",
                   "hardware_key": hardware,
                   "protocol": {"timer": "device_event", "warmup": 10, "iterations": 30,
                                "trials": 1, "preallocated": True,
                                "timing_scope": scope}}
        requests.append(request)
        results.append({
            "measurement_id": f"i{index}", "request_key": f"i{index}",
            "condition_key": "cond:iso", "status": "completed",
            "source_class": "measured", "run_id": f"run{index}", "split": "train",
            "duration_ms": duration, "raw_samples_ms": [duration],
            "trials": [{"trial_index": 0, "median_ms": duration,
                        "raw_samples_ms": [duration]}],
            "observed_context": {"observed_api": "torch.mm",
                                 "execution_kind": "primitive"},
            "protocol": request["protocol"], "correctness": {"status": "passed"},
        })
    manifest = {**manifest, "requests": requests,
                "coverage": {"selected_train": 3, "selected_heldout": 0,
                             "pending_conditions": 0, "derivation_failures": []},
                "inventory": [], "models": {}, "pending_requests": []}
    analysis = analyze(manifest, {"status": "completed", "results": results})
    assert len(analysis["analyses"]) == 3
    assert {group["hardware_key"] for group in analysis["analyses"]} == {
        "Device-A", "Device-B"}
    scopes = {group["protocol_signature"] for group in analysis["analyses"]}
    assert len(scopes) == 2


def test_grouped_histogram_conserves_tokens_and_experts_exactly():
    from mindie_sim.a3_operator_worker_recipes import _scaled_expert_counts
    counts, source = _scaled_expert_counts([], 5, 3)
    assert len(counts) == 5 and sum(counts) == 3 and counts == [1, 1, 1, 0, 0]
    assert sum(1 for value in counts if value == 0) == 2
    assert source.startswith("actual_point")
    counts, source = _scaled_expert_counts([1, 3, 0, 4], 4, 8)
    assert counts == [1, 3, 0, 4] and sum(counts) == 8
    assert source == "anchor_actual_histogram"
    counts, _ = _scaled_expert_counts([4, 4], 5, 7)
    assert len(counts) == 5 and sum(counts) == 7
    counts, _ = _scaled_expert_counts([], 4, 0)
    assert counts == [0, 0, 0, 0] and sum(counts) == 0
    from mindie_sim.a3_operator_worker_native import _validate_grouped_shapes
    from mindie_sim.a3_types import A3InputError
    assert _validate_grouped_shapes([(1, 128), (3, 128), (0, 128), (4, 128)],
                                    [1, 3, 0, 4], 128) == [
        [1, 128], [3, 128], [0, 128], [4, 128]]
    with pytest.raises(A3InputError):
        _validate_grouped_shapes([(1, 128), (3, 128), (0, 128)], [1, 3, 0, 4], 128)
    with pytest.raises(A3InputError):
        _validate_grouped_shapes([(1, 128), (3, 64)], [1, 3], 128)


def test_native_grouped_recipe_dispatches_and_shape_binding_exists():
    from types import SimpleNamespace
    from mindie_sim.a3_operator_worker_recipes import (
        RECIPE_BUILDERS, SUPPORTED_RECIPES, TorchNpuOperatorRunner,
    )
    from mindie_sim.a3_operator_worker_native import _shape
    assert _shape({"point": {"shape": {"q": 32, "kv": 32, "heads": 1}}})["q"] == 32
    assert "npu_grouped_matmul_list_bf16" in SUPPORTED_RECIPES
    assert RECIPE_BUILDERS["npu_grouped_matmul_list_bf16"] == "_grouped_matmul_inputs"
    dispatch = TorchNpuOperatorRunner._builder(
        None, {"worker_recipe": "npu_grouped_matmul_list_bf16"})
    assert dispatch["status"] == "dispatch_only"
    assert dispatch["device_access"] is False
    # The split left no unbound shape name in any of the three native methods.
    probe = SimpleNamespace(torch=None, torch_npu=SimpleNamespace())
    for method, request in (
        ("_masked_softmax_inputs", {"point": {"shape": {"q": 32, "kv": 32, "heads": 1}}}),
        ("_grouped_matmul_inputs", {"point": {"shape": {"routed_tokens": 3, "experts": 5,
                                                       "k": 128, "n": 128}},
                                     "condition": {"features": {}}}),
        ("_dynamic_quant_inputs", {"point": {"shape": {"tokens": 8, "width": 128}}}),
    ):
        from mindie_sim.a3_operator_worker_native import NativeRecipesMixin
        try:
            getattr(NativeRecipesMixin, method)(probe, request)
        except NameError as exc:
            pytest.fail(f"{method} still has unbound shape/name: {exc}")
        except Exception:
            pass  # expected missing fake torch attrs, not a NameError


def test_fa_and_rope_are_not_advertised_executable_without_builders():
    from mindie_sim.a3_operator_worker_recipes import SUPPORTED_RECIPES
    assert "npu_fusion_attention" not in SUPPORTED_RECIPES
    assert "npu_rotary_mul" not in SUPPORTED_RECIPES
    assert "npu_apply_rotary_pos_emb" not in SUPPORTED_RECIPES


def _gap_curve_artifact():
    from mindie_sim.a3_operator_curves_adapter import adapt_linearity_split_results
    from mindie_sim.a3_operator_curve_forecast import export_curve_artifact
    # Six train points split 3+3 under the three-point segment minimum, so the
    # unresolved gap sits between x=16 and x=129 and the heldout x=32 falls in
    # it.  (The fixture previously relied on 2-point segments ending at x=8.)
    cases = []
    for x, y, split in [(1, 1, "train"), (8, 1, "train"), (16, 3, "train"),
                        (129, 5, "train"), (512, 10, "train"), (513, 10, "train"),
                        (32, 1, "heldout")]:
        cases.append({
            "request": {"case_id": f"x{x}", "op_family": "gemm",
                        "shape": {"m": x, "n": 128, "k": 128}, "dtype": "bf16",
                        "layout": "contiguous",
                        # Complete precision attestation: the curve must be able to
                        # reject a contradicted dtype (see the identity test).
                        "features": {"weight_dtype": "bf16",
                                     "activation_dtype": "bf16",
                                     "compute_dtype": "bf16",
                                     "accumulation_dtype": "backend_default",
                                     "quantization": "none"},
                        "sweeps": [{"axis": "m"}]},
            "campaign_split": split,
            "production_result": {
                "status": "completed", "source_class": "measured", "run_id": "synthetic",
                "raw_samples_ms": [y] * 3, "duration_ms": y,
                "environment": {"device_name": "SyntheticDevice"},
                "protocol": {"timer": "device_event", "warmup": 10,
                             "iterations": 3, "preallocated": True},
            },
        })
    manifest, results = adapt_linearity_split_results({"cases": cases})
    return export_curve_artifact(manifest, results)


def _curve_query(curve, shape, **overrides):
    physical = curve["physical"]
    query = {
        "hardware_key": physical["hardware_key"],
        "family": physical["family"], "recipe": physical["recipe"],
        "execution_kind": physical["execution_kind"],
        "worker_recipe": physical["worker_recipe"],
        "backend": physical["backend"],
        "target_match": physical["target_match"],
        "coverage_scope": physical["coverage_scope"],
        "layout": physical["layout"], "quantization": physical["quantization"],
        "precision": physical["precision"], "features": physical["features"],
        "axis": curve["axis"], "shape": shape, "protocol": curve["protocol"],
    }
    query.update(overrides)
    return query


def test_curve_artifact_gap_policy_and_public_query():
    from mindie_sim.a3_operator_curve_forecast import query_curve_artifact
    artifact = _gap_curve_artifact()
    assert artifact["curve_count"] == 1
    assert artifact["source_evidence"]["heldout_values_included"] is False
    curve = artifact["curves"][0]
    assert 32.0 not in curve["train_support"]["x_values"]
    assert all(point["x"] != 32.0 for point in curve["train_points"])
    gap = query_curve_artifact(artifact, _curve_query(
        curve, {"m": 32, "n": 128, "k": 128}))
    assert gap["status"] == "transition_unresolved"
    assert gap["prediction_ms"] is None
    assert gap["lower_bound_ms"] is not None and gap["lower_bound_ms"] >= 0
    assert gap["upper_bound_ms"] >= gap["lower_bound_ms"]
    assert gap["production_match"] is False
    inside = query_curve_artifact(artifact, _curve_query(
        curve, {"m": 4, "n": 128, "k": 128}))
    assert inside["status"] == "ok"
    assert inside["prediction_ms"] > 0
    assert inside["region"] == "interpolation"
    outside = query_curve_artifact(artifact, _curve_query(
        curve, {"m": 1024, "n": 128, "k": 128}))
    assert outside["region"] == "extrapolation"
    assert outside["prediction_ms"] is not None and outside["prediction_ms"] >= 0


def test_curve_query_rejects_incompatible_identity_and_labels_transfer():
    from mindie_sim.a3_operator_curve_forecast import query_curve_artifact
    artifact = _gap_curve_artifact()
    curve = artifact["curves"][0]
    base = _curve_query(curve, {"m": 4, "n": 128, "k": 128})
    bad_dtype = copy.deepcopy(base)
    bad_dtype["precision"] = {**base["precision"], "output_dtype": "fp32"}
    assert query_curve_artifact(artifact, bad_dtype)["status"] == "unsupported"
    bad_layout = copy.deepcopy(base)
    bad_layout["layout"] = "strided"
    assert query_curve_artifact(artifact, bad_layout)["status"] == "unsupported"
    bad_feature = copy.deepcopy(base)
    bad_feature["features"] = {"eps": 1e-5}
    assert query_curve_artifact(artifact, bad_feature)["status"] == "unsupported"
    bad_recipe = copy.deepcopy(base)
    bad_recipe["recipe"] = "different_recipe"
    assert query_curve_artifact(artifact, bad_recipe)["status"] == "unsupported"
    bad_protocol = copy.deepcopy(base)
    bad_protocol["protocol"] = {**base["protocol"], "warmup": 11}
    assert query_curve_artifact(artifact, bad_protocol)["status"] == "unsupported"
    bad_backend = copy.deepcopy(base)
    bad_backend["backend"] = "wrong_backend"
    assert query_curve_artifact(artifact, bad_backend)["status"] == "unsupported"
    bad_hw = copy.deepcopy(base)
    bad_hw["hardware_key"] = "OtherDevice"
    assert query_curve_artifact(artifact, bad_hw)["status"] == "unsupported"
    transfer = query_curve_artifact(artifact, bad_hw, allow_primitive_transfer=True)
    assert transfer["status"] == "unsupported"
    assert transfer["reason"] == "no_compatible_curve"


def test_curve_artifact_preserves_native_reruns_and_target_false():
    from mindie_sim.a3_operator_curves_adapter import adapt_campaign_manifest_results
    from mindie_sim.a3_operator_curve_forecast import (
        export_curve_artifact, query_curve_artifact,
    )
    condition = {
        "family": "rmsnorm", "worker_recipe": "npu_rms_norm",
        "execution_kind": "fused_kernel", "status": "ready",
        "shape_kind": "rmsnorm", "precision": {
            "input_dtype": "bf16", "output_dtype": "bf16",
            "compute_dtype": "bf16", "accumulation_dtype": "backend_default"},
        "layout": "contiguous", "features": {"eps": 1e-6},
    }
    protocol = {"warmup": 10, "iterations": 30, "trials": 2,
                "timer": "device_event", "preallocated": False}
    requests = []
    results = []
    for x in (128, 256):
        midpoint = f"m{x}"
        requests.append({
            "request_key": midpoint, "measurement_id": midpoint, "condition_key": midpoint,
            "point": {"shape": {"tokens": x, "width": 512}},
            "condition": condition, "worker_recipe": "npu_rms_norm",
            "execution_kind": "fused_kernel", "backend": "torch_npu_installed_api",
            "split": "train", "run_id": "run-a", "target_match": False,
            "coverage_scope": "native_primitive_at_source_declared_2d_geometry",
            "protocol": protocol, "query_context": {"scope": "native"},
            "origins": [{"anchor_family": "rmsnorm", "anchor_width": 512,
                         "axis": "tokens", "models": ["qwen3.8-27b"]}],
            "work": {"logical_input_elements": x * 512},
        })
        for run, value in (("run-a", 1.0 + x), ("run-b", 1.1 + x)):
            results.append({
                "measurement_id": midpoint, "status": "completed",
                "source_class": "measured", "run_id": run, "split": "train",
                "trials": [{"trial_index": i, "median_ms": value,
                            "raw_samples_ms": [value] * 3} for i in range(2)],
                "observed_context": {"observed_api": "torch_npu.npu_rms_norm",
                                     "execution_kind": "fused_kernel"},
                "protocol": {**protocol, "preallocated": False},
                "environment": {"device_name": "Ascend910_9362"},
                "correctness": {"status": "passed"},
            })
    manifest = {"campaign_id": "native-rerun", "requests": requests,
                "coverage": {"selected_train": 2, "selected_heldout": 0,
                             "pending_conditions": 0, "derivation_failures": []},
                "inventory": [], "models": {}, "pending_requests": []}
    adapted_manifest, adapted_results = adapt_campaign_manifest_results(
        manifest, {"status": "completed", "results": results, "run_id": "run-a"})
    assert len(adapted_results["results"]) == 4
    from mindie_sim.a3_operator_curves_adapter import normalized_conditions
    normalized = normalized_conditions(adapted_manifest, adapted_results)
    assert all(row["query_context"]["target_match"] is False
               for row in normalized["conditions"])
    artifact = export_curve_artifact(adapted_manifest, adapted_results)
    assert artifact["curve_count"] == 1
    curve = artifact["curves"][0]
    assert curve["physical"]["target_match"] is False
    assert curve["train_support"]["independent_trial_count"] == 8
    assert all(point["trial_count"] == 4 for point in curve["train_points"])
    query = _curve_query(curve, {"tokens": 128, "width": 512})
    result = query_curve_artifact(artifact, query)
    assert result["status"] == "ok"
    assert result["match_scope"] == "exact_physical_candidate"
    assert result["production_match"] is False
    assert result["target_match"] is False


def test_query_policy_never_returns_negative_extrapolation():
    from mindie_sim.a3_operator_curve_query import _predict, _predict_query
    fit = {"segments": [{"x_min": 1.0, "x_max": 8.0, "n": 2,
                         "slope_ms_per_unit": -2.0, "intercept_ms": 10.0}]}
    result = _predict_query(fit, 100.0)
    assert result["prediction_ms"] is None
    assert result["region"] == "nonphysical_extrapolation"
    assert _predict(fit, 100.0) is None
    assert _predict_query(fit, 4.0)["prediction_ms"] == 2.0


def test_native_adapter_keeps_epsilon_and_output_dtype_physical_identity():
    from mindie_sim.a3_operator_curves_adapter import (
        adapt_campaign_manifest_results, normalized_conditions,
    )
    from mindie_sim.a3_operator_curves import analyze
    base = {
        "family": "rmsnorm", "worker_recipe": "npu_rms_norm",
        "execution_kind": "fused_kernel", "status": "ready",
        "shape_kind": "rmsnorm",
        "layout": "contiguous", "features": {"epsilon": 1e-6},
        "precision": {"input_dtype": "bf16", "output_dtype": "bf16",
                      "compute_dtype": "bf16", "accumulation_dtype": "backend_default"},
    }
    protocol = {"warmup": 10, "iterations": 30, "trials": 2,
                "timer": "device_event", "preallocated": False}
    requests, results = [], []
    for index, (eps, output) in enumerate(((1e-6, "bf16"), (1e-5, "bf16"),
                                           (1e-6, "fp32"))):
        midpoint = f"p{index}"
        condition = copy.deepcopy(base)
        condition["features"]["epsilon"] = eps
        condition["precision"]["output_dtype"] = output
        requests.append({
            "request_key": midpoint, "measurement_id": midpoint,
            "point": {"shape": {"tokens": 128, "width": 512}},
            "condition": condition, "worker_recipe": "npu_rms_norm",
            "execution_kind": "fused_kernel", "backend": "torch_npu_installed_api",
            "split": "train", "target_match": False,
            "coverage_scope": "native_primitive_at_source_declared_2d_geometry",
            "protocol": protocol, "query_context": {"scope": "native"},
            "origins": [{"anchor_family": "rmsnorm", "anchor_width": 512,
                         "axis": "tokens", "models": ["qwen3.8-27b"]}],
            "work": {"logical_input_elements": 128 * 512},
        })
        results.append({
            "measurement_id": midpoint, "status": "completed",
            "source_class": "measured", "run_id": f"run{index}", "split": "train",
            "trials": [{"trial_index": 0, "median_ms": 1.0,
                        "raw_samples_ms": [1.0] * 3}],
            "environment": {"device_name": "Ascend910_9362"},
            "protocol": protocol, "correctness": {"status": "passed"},
            "observed_context": {"observed_api": "torch_npu.npu_rms_norm"},
        })
    manifest = {"campaign_id": "native-physical", "requests": requests,
                "coverage": {"selected_train": 3, "selected_heldout": 0,
                             "pending_conditions": 0, "derivation_failures": []},
                "inventory": [], "models": {}, "pending_requests": []}
    adapted_manifest, adapted_results = adapt_campaign_manifest_results(
        manifest, {"status": "completed", "results": results})
    keys = [request["condition_key"] for request in adapted_manifest["requests"]]
    assert len(set(keys)) == 3
    analysis = analyze(adapted_manifest, adapted_results)
    assert len(analysis["analyses"]) == 3
    normalized = normalized_conditions(adapted_manifest, adapted_results)
    assert len(normalized["conditions"]) == 3


def test_runner_entry_binds_validation_before_device_boundary():
    from mindie_sim.a3_operator_worker_recipes import TorchNpuOperatorRunner
    request = {
        "request_key": "entry", "measurement_id": "entry",
        "condition_key": "cond:entry", "point": {"shape": {"m": 8, "n": 8, "k": 8}},
        "condition": {"status": "executable", "worker_recipe": "torch_mm_preallocated",
                      "execution_kind": "primitive"},
        "worker_recipe": "torch_mm_preallocated", "execution_kind": "primitive",
        "protocol": {"warmup": 1, "iterations": 1, "trials": 1},
    }
    runner = object.__new__(TorchNpuOperatorRunner)
    try:
        runner.run(request)
    except NameError as exc:
        pytest.fail(f"runner entry still unbound: {exc}")
    except AttributeError as exc:
        assert "torch" in str(exc).lower()
    pending = {**request, "condition": {"status": "pending_backend",
                                        "worker_recipe": None,
                                        "execution_kind": "primitive"}}
    result = runner.run(pending)
    assert result["status"] == "pending"
    assert result["source_class"] == "not_measured"


def test_curve_artifact_keeps_distinct_protocol_groups_and_curve_ids():
    from mindie_sim.a3_operator_curves_adapter import adapt_linearity_split_results
    from mindie_sim.a3_operator_curve_forecast import (
        export_curve_artifact, query_curve_artifact,
    )
    cases = []
    for index, (x, scope) in enumerate([
        (1, "per_operator"), (8, "per_operator"),
        (16, "total_repeated_loop"), (32, "total_repeated_loop"),
    ]):
        cases.append({
            "request": {"case_id": f"x{x}", "op_family": "gemm",
                        "shape": {"m": x, "n": 128, "k": 128}, "dtype": "bf16",
                        "layout": "contiguous", "features": {}, "sweeps": [{"axis": "m"}]},
            "campaign_split": "train",
            "production_result": {
                "status": "completed", "source_class": "measured", "run_id": f"r{index}",
                "raw_samples_ms": [1.0] * 3, "duration_ms": 1.0,
                "environment": {"device_name": "SyntheticDevice"},
                "protocol": {"timer": "device_event", "warmup": 10,
                             "iterations": 3, "preallocated": True,
                             "timing_scope": scope},
            },
        })
    manifest, results = adapt_linearity_split_results({"cases": cases})
    artifact = export_curve_artifact(manifest, results)
    assert artifact["curve_count"] == 2
    assert len({curve["curve_id"] for curve in artifact["curves"]}) == 2
    scopes = {curve["protocol"]["timing_scope"] for curve in artifact["curves"]}
    assert scopes == {"per_operator", "total_repeated_loop"}
    for curve in artifact["curves"]:
        result = query_curve_artifact(artifact, _curve_query(
            curve, {"m": 4, "n": 128, "k": 128}))
        assert result["status"] in {"ok", "transition_unresolved", "extrapolation"}
        assert result["matched_curve"] == curve["curve_id"]


def _graph_protocol_fixture():
    from mindie_sim.a3_operator_curves_adapter import adapt_campaign_manifest_results
    from mindie_sim.a3_operator_curve_forecast import export_curve_artifact
    base = {
        "timer": "device_event", "warmup": 10, "iterations": 30, "trials": 2,
        "preallocated": False,
        "cache_policy": "same_static_inputs; graph context sync+empty_cache before capture; every trial rewarm",
        "timing_scope": "device_event_block_interval_with_separate_host_clocks",
        "measurement_scope": "device_event_block_interval_with_separate_host_clocks",
        "device_interval_may_include_idle_gaps": True,
        "per_invocation_derived_only": True,
        "pool_sharing": {"shared": False, "policy": "none", "per_condition_graph": True},
        "output_policy": "native_default_outputs_only_no_out_overload",
        "output_lifetime": "retain_last_only",
    }
    modes = [
        ("eager_block", 0, 32, 0, "not_applicable_eager_block", False),
        ("graph_single_repeated", 1, 32, 32, "none", True),
        ("graph_block", 32, 1, 1, "none", True),
    ]
    requests, results = [], []
    for mode, capture, replay, host, pool_policy, per_graph in modes:
        for x in (8, 16, 32):
            midpoint = f"{mode}-{x}"
            protocol = {**base, "mode": mode, "api_invocations_per_block": 32,
                        "capture_invocations": capture,
                        "host_calls_per_block": host,
                        "replay_calls_per_block": replay,
                        "pool_sharing": {"shared": False, "policy": pool_policy,
                                         "per_condition_graph": per_graph}}
            condition = {
                "family": "rmsnorm", "recipe": "npu_rms_norm",
                "shape": {"width": 5120},
                "precision": {"input_dtype": "bf16", "output_dtype": "bf16",
                              "compute_dtype": "bf16"},
                "features": {"epsilon": 1e-6}, "layout": "contiguous",
                "quantization": "none",
            }
            requests.append({
                "measurement_id": midpoint, "request_key": midpoint,
                "condition_key": "physical-rms", "hardware_key": "Ascend910_9362",
                "op_family": "rmsnorm", "recipe": "npu_rms_norm",
                "execution_kind": "primitive", "worker_recipe": "npu_rms_norm",
                "backend": "torch_npu_installed_api", "target_match": False,
                "coverage_scope": "primitive_candidate", "condition": condition,
                "layout": "contiguous", "split": "train",
                "point": {"axis": "tokens", "value": x,
                          "shape": {"tokens": x, "width": 5120}},
                "protocol": protocol,
            })
            value = (1 + x / 100) * (modes.index((mode, capture, replay, host, pool_policy, per_graph)) + 1)
            results.append({
                "measurement_id": midpoint, "condition_key": "physical-rms",
                "status": "completed", "source_class": "measured", "run_id": "synthetic",
                "split": "train", "protocol": protocol,
                "correctness": {"status": "passed"},
                "trials": [{"trial_index": 0, "median_ms": value,
                            "raw_samples_ms": [value] * 3}],
            })
    manifest = {"campaign_id": "graph-protocol", "requests": requests,
                "coverage": {"selected_train": len(requests), "selected_heldout": 0,
                             "pending_conditions": 0, "derivation_failures": []},
                "inventory": [], "models": {}, "pending_requests": []}
    adapted_manifest, adapted_results = adapt_campaign_manifest_results(
        manifest, {"status": "completed", "results": results})
    return (export_curve_artifact(adapted_manifest, adapted_results),
            adapted_manifest, adapted_results)


def test_graph_modes_are_distinct_protocol_identity():
    from mindie_sim.a3_operator_curve_forecast import query_curve_artifact
    from mindie_sim.a3_operator_curve_protocol import canonical_protocol
    artifact, adapted_manifest, adapted_results = _graph_protocol_fixture()
    assert artifact["curve_count"] == 3
    assert len({curve["curve_id"] for curve in artifact["curves"]}) == 3
    modes = {curve["protocol"]["mode"] for curve in artifact["curves"]}
    assert modes == {"eager_block", "graph_single_repeated", "graph_block"}
    from mindie_sim.a3_operator_curves import analyze
    from mindie_sim.a3_operator_curves_adapter import normalized_conditions
    assert len(analyze(adapted_manifest, adapted_results)["analyses"]) == 3
    normalized = normalized_conditions(adapted_manifest, adapted_results)
    assert {row["query_context"]["protocol"]["mode"]
            for row in normalized["conditions"]} == modes
    for curve in artifact["curves"]:
        query = _curve_query(curve, {"tokens": 24, "width": 5120})
        result = query_curve_artifact(artifact, query)
        assert result["status"] == "ok"
        assert result["matched_curve"] == curve["curve_id"]
    eager = next(curve for curve in artifact["curves"]
                 if curve["protocol"]["mode"] == "eager_block")
    base = _curve_query(eager, {"tokens": 24, "width": 5120})
    for mode in ("graph_single_repeated", "graph_block"):
        changed = copy.deepcopy(base)
        target = next(curve for curve in artifact["curves"]
                      if curve["protocol"]["mode"] == mode)
        changed["protocol"] = copy.deepcopy(target["protocol"])
        assert query_curve_artifact(artifact, changed)["status"] == "ok"
        # A curve from one mode must never answer a different mode.
        assert query_curve_artifact(
            {"curves": [eager]}, changed)["status"] == "unsupported"
    changed_n = copy.deepcopy(base)
    changed_n["protocol"]["api_invocations_per_block"] = 128
    assert query_curve_artifact(artifact, changed_n)["status"] == "unsupported"
    changed_lifetime = copy.deepcopy(base)
    changed_lifetime["protocol"]["output_lifetime"] = "retain_all"
    assert query_curve_artifact(artifact, changed_lifetime)["status"] == "unsupported"
    missing_mode = copy.deepcopy(base)
    missing_mode["protocol"].pop("mode")
    assert query_curve_artifact(artifact, missing_mode)["status"] == "unsupported"
    # Outcome/address/run-id facts are provenance, not protocol applicability.
    noisy = copy.deepcopy(base)
    noisy["protocol"].update({
        "capture_unique_address_count": 999, "eager_unique_address_count": 999,
        "run_id": "another-run", "duration_ms": 123.0,
        "block_wall_ms": 9.9, "capture_time_ms": 0.2,
    })
    assert canonical_protocol(noisy["protocol"]) == canonical_protocol(base["protocol"])


def test_legacy_artifact_without_mode_does_not_match_graph_query():
    from mindie_sim.a3_operator_curve_forecast import (
        export_curve_artifact, query_curve_artifact,
    )
    from mindie_sim.a3_operator_curves_adapter import adapt_linearity_split_results
    cases = []
    for x in (8, 16, 32):
        cases.append({
            "request": {"case_id": f"l{x}", "op_family": "rmsnorm",
                        "shape": {"tokens": x, "width": 5120}, "dtype": "bf16",
                        "layout": "contiguous", "features": {}, "sweeps": [{"axis": "tokens"}]},
            "campaign_split": "train",
            "production_result": {
                "status": "completed", "source_class": "measured", "run_id": "legacy",
                "raw_samples_ms": [1.0 + x / 100] * 3,
                "duration_ms": 1.0 + x / 100,
                "environment": {"device_name": "Ascend910_9362"},
                "protocol": {"timer": "device_event", "warmup": 10,
                             "iterations": 30, "trials": 2, "preallocated": False,
                             "cache_policy": "legacy", "timing_scope": "device_event_warm"},
            },
        })
    manifest, results = adapt_linearity_split_results({"cases": cases})
    artifact = export_curve_artifact(manifest, results)
    curve = artifact["curves"][0]
    assert "mode" not in curve["protocol"]
    legacy_query = _curve_query(curve, {"tokens": 24, "width": 5120})
    assert query_curve_artifact(artifact, legacy_query)["status"] == "ok"
    graph_query = copy.deepcopy(legacy_query)
    graph_query["protocol"] = {**graph_query["protocol"], "mode": "graph_block",
                               "api_invocations_per_block": 32,
                               "capture_invocations": 32,
                               "host_calls_per_block": 1,
                               "replay_calls_per_block": 1,
                               "output_lifetime": "retain_last_only",
                               "pool_sharing": {"shared": False, "policy": "none",
                                                "per_condition_graph": True}}
    assert query_curve_artifact(artifact, graph_query)["status"] == "unsupported"


def test_precision_and_grouped_geometry_use_actual_op_facts():
    from types import SimpleNamespace
    from mindie_sim.a3_operator_campaign_conditions import (
        _actual_grouped_geometry, _precision,
    )
    quant = SimpleNamespace(scale_format="fp8_e8m0", scale_bytes=1,
                            weight_block=[128, 128], expert_block=32,
                            kv_dtype="bf16")
    model = SimpleNamespace(quant=quant, weight_dtype="bf16",
                            activation_dtype="bf16", compute_dtype="bf16",
                            quantization="fp8_dynamic+mxfp4_experts",
                            kv_dtype="bf16")
    softmax = SimpleNamespace(
        metadata={"input_format": "fp32", "output_format": "fp32",
                  "semantic_family": "attention_softmax"},
        dtype="fp32", flops=0.0, layout="contiguous")
    precision = _precision(softmax, model)
    assert precision["input_dtype"] == "fp32"
    assert precision["output_dtype"] == "fp32"
    assert precision["compute_dtype"] == "fp32"
    assert precision["quantization"] == "none"
    assert precision["weight_format"] is None
    assert precision["scale_format"] == ""
    assert precision["scale_bytes"] == 0
    assert precision["weight_block"] == [0, 0]
    assert precision["expert_block"] == 0
    assert precision["kv_dtype"] == ""

    grouped_op = SimpleNamespace(metadata={
        "group_size": 8, "expert_tokens": [32] * 8, "local_expert_start": 0,
    })
    work = SimpleNamespace(routing=SimpleNamespace(to_dict=lambda: {
        "routing_tokens": 1280, "local_experts": 16,
        "local_expert_counts": [32] * 16,
    }))
    grouped = _actual_grouped_geometry(grouped_op, work, 512)
    assert grouped["actual_experts"] == 8
    assert grouped["actual_routed_tokens"] == 256
    assert grouped["actual_expert_tokens"] == [32] * 8
    assert grouped["global_experts"] == 512
    assert grouped["rank_experts"] == 16
    assert grouped["rank_routed_tokens"] == 1280
    assert grouped["source"] == "actual_op_group_size_expert_tokens"


def test_grouped_sweep_histogram_conserves_actual_group():
    from mindie_sim.a3_operator_campaign_sweeps import _grouped_histogram
    condition = {
        "shape_kind": "moe_grouped", "shape": {"experts": 8, "routed_tokens": 256,
                                                "n": 4096, "k": 8192},
        "features": {"grouped_geometry": {
            "actual_expert_tokens": [32] * 8, "actual_experts": 8,
            "actual_routed_tokens": 256,
        }, "routing": {"local_expert_counts": [32] * 8}},
        "precision": {"input_dtype": "bf16", "output_dtype": "bf16",
                      "compute_dtype": "bf16", "weight_dtype": "bf16",
                      "quantization": "none", "weight_format": "bf16",
                      "accumulation_dtype": "backend_default"},
        "work": {"flops": 1.0, "hbm_bytes": 1.0, "mte2_bytes": 1.0,
                 "mte3_bytes": 1.0, "vector_ops": 0.0},
        "execution_kind": "fused_kernel",
        "worker_recipe": "npu_grouped_matmul_list_bf16",
        "backend": "torch_npu", "layout": "contiguous",
        "quantization": "none", "protocol": {"timer": "device_event"},
    }
    counts, source = _grouped_histogram(condition, condition["shape"])
    assert counts == [32] * 8 and sum(counts) == 256
    assert source == "anchor_actual_histogram"
    smaller = {**condition["shape"], "routed_tokens": 128}
    counts, source = _grouped_histogram(condition, smaller)
    assert len(counts) == 8 and sum(counts) == 128
    assert source.startswith("actual_point")
    # Point materialization must carry the exact conserved histogram.
    from mindie_sim.a3_operator_campaign_sweeps import _point_from_shape
    class Config:
        max_working_set_bytes = 10 ** 12
    point = _point_from_shape(condition, condition["shape"], axis="routed_tokens",
                              value=256, reason="anchor", split_hint="train",
                              split_group=None, actual=True, boundary_probe=False,
                              extrapolation=False, config=Config())
    assert point["expert_token_counts"] == [32] * 8
    assert sum(point["expert_token_counts"]) == point["shape"]["routed_tokens"]


def test_trial_schedule_warms_each_trial_without_fake_npu():
    from mindie_sim.a3_operator_worker_recipes import (
        TorchNpuOperatorRunner, _request_protocol,
    )
    from mindie_sim.a3_operator_campaign_conditions import DEFAULT_PROTOCOL
    from mindie_sim.a3_operator_curve_protocol import protocol_key
    runner = object.__new__(TorchNpuOperatorRunner)
    calls = []

    def invoke():
        calls.append("warmup")

    def measure(trial_index):
        calls.append(f"measure:{trial_index}")
        return [float(trial_index + 1)]

    scheduled = list(runner._trial_schedule(warmup=10, trials=2,
                                            invoke=invoke, measure=measure))
    assert [item[0] for item in scheduled] == [0, 1]
    assert calls == (["warmup"] * 10 + ["measure:0"] +
                     ["warmup"] * 10 + ["measure:1"])
    assert DEFAULT_PROTOCOL["warmup_scope"] == "before_each_trial"
    assert _request_protocol({"protocol": {"warmup": 10, "iterations": 30,
                                            "trials": 2}})["warmup_scope"] == "before_each_trial"
    assert protocol_key({"warmup_scope": "before_each_trial"}) != protocol_key(
        {"warmup_scope": "once_before_trials"})
    assert protocol_key({"warmup": 10}) != protocol_key(
        {"warmup": 10, "warmup_scope": "before_each_trial"})


def test_attention_campaign_adapter_dry_run_and_three_clocks():
    from mindie_sim.a3_operator_attention_adapter import (
        attention_manifest_from_campaign, attention_results_to_campaign,
        campaign_requests_from_attention, dry_run_campaign_attention,
    )
    template = json.loads(Path("examples/operator-attention/capability.manifest.json").read_text())
    for case in template["cases"]:
        source = case.setdefault("source", {})
        if case["operator"] == "fusion_attention":
            source.setdefault("axis", "q_kv" if case["shape"]["q_tokens"] == case["shape"]["kv_tokens"] else "kv")
            source["value"] = (case["shape"]["q_tokens"]
                               if source["axis"] == "q_kv" else case["shape"]["kv_tokens"])
        else:
            source.setdefault("axis", "seq")
            source["value"] = case["shape"]["seq"]
    requests = campaign_requests_from_attention(template, hardware_key="Ascend910_9362")
    assert len(requests) == 4
    built = attention_manifest_from_campaign(requests, template)
    assert built["manifest"]["kind"] == "a3_attention_rope_manifest"
    assert "schema_version" not in built["manifest"]
    dry = dry_run_campaign_attention(requests, template)
    assert dry["valid"] is True and dry["ready_cases"] == 4
    assert dry["plan"]["counts"]["planned_raw_blocks"] == 240
    assert dry["requires_device"] is False
    no_source = {key: value for key, value in template.items()
                 if key != "source_contract"}
    optional = dry_run_campaign_attention(requests, no_source)
    assert optional["valid"] is True
    assert optional["attribution_optional"] is True
    assert not optional["plan"].get("source_contract")
    partial = {**template, "source_contract": {"sha256": "0" * 64}}
    partial_dry = dry_run_campaign_attention(requests, partial)
    assert partial_dry["valid"] is True
    assert partial_dry["attribution_optional"] is True
    assert not partial_dry["plan"].get("source_contract")

    missing = dry_run_campaign_attention([{
        "request_key": "rope-generic", "op_family": "rope",
        "condition": {"family": "rope"}, "point": {"shape": {"elements": 8192}},
        "target_match": False,
    }], template)
    assert missing["valid"] is False
    assert missing["pending"][0]["missing_fields"]
    assert "shape.seq" in missing["pending"][0]["missing_fields"]
    mrope = copy.deepcopy(requests[2])
    mrope["attention_case"] = {**mrope["attention_case"], "rotary_mode": "mrope"}
    unsupported = dry_run_campaign_attention([mrope], template)
    assert unsupported["valid"] is False
    assert unsupported["pending"][0]["reason"] == "unsupported_native_semantics"

    case = requests[0]["attention_case"]
    results = {
        "run_id": "synthetic", "environment": {"device_name": "SyntheticNPU"},
        "protocol": {"warmup_iterations": 10, "trials": 2, "samples_per_trial": 2},
        "cases": [{
            "case_id": case["case_id"], "operator": case["operator"],
            "status": "measured", "split": "train", "target_match": False,
            "native": {"observed_api": "torch_npu.npu_fusion_attention",
                       "output0": {"dtype": "bfloat16", "shape": [1, 3, 32, 256]}},
            "correctness": {"status": "passed"}, "allocation": {"preallocated": False},
            "trials": [
                {"trial_index": 0, "status": "measured", "warmup": {"iterations": 10},
                 "samples": [
                     {"trial_index": 0, "sample_index": 0, "api_invocations": 1,
                      "cpu_submit_ms": 0.1, "device_interval_ms": 1.0,
                      "wall_complete_ms": 1.5, "events_reused": True,
                      "cpu_submit_scope": "native_api_call_only"},
                     {"trial_index": 0, "sample_index": 1, "api_invocations": 1,
                      "cpu_submit_ms": 0.1, "device_interval_ms": 1.2,
                      "wall_complete_ms": 1.7, "events_reused": True,
                      "cpu_submit_scope": "native_api_call_only"},
                 ]},
                {"trial_index": 1, "status": "measured", "warmup": {"iterations": 10},
                 "samples": [
                     {"trial_index": 1, "sample_index": 0, "api_invocations": 1,
                      "cpu_submit_ms": 0.1, "device_interval_ms": 2.0,
                      "wall_complete_ms": 2.5, "events_reused": True,
                      "cpu_submit_scope": "native_api_call_only"},
                 ]},
            ],
        }],
    }
    campaign = attention_results_to_campaign(results, [requests[0]])
    row = campaign["cases"][0]
    assert row["status"] == "completed"
    assert row["source_class"] == "measured"
    assert row["target_match"] is False
    assert len(row["trials"]) == 2
    assert row["raw_samples_ms"] == [1.0, 1.2, 2.0]
    assert row["observed_context"]["clocks"]["cpu_submit_ms"] == [0.1, 0.1, 0.1]
    assert row["observed_context"]["clocks"]["wall_complete_ms"] == [1.5, 1.7, 2.5]
    assert row["protocol"]["interval_scope"].endswith("native_allocation")
    assert row["observed_context"]["not_pure_device_service"] is True
    from mindie_sim.a3_operator_curves import analyze
    manifest = {"campaign_id": "attention-conversion", "requests": [requests[0]],
                "coverage": {"selected_train": 1, "selected_heldout": 0,
                             "pending_conditions": 0, "derivation_failures": []},
                "inventory": [], "models": {}, "pending_requests": []}
    analysis = analyze(manifest, campaign)
    assert len(analysis["analyses"]) == 1
    assert analysis["analyses"][0]["independent_train_trials"] == 2


def test_actual_catalog_grouped_and_softmax_primitive_facts():
    config = CampaignConfig.from_mapping({
        "models": ["qwen3.8-2.4t-a95b", "deepseek-v4-flash-0731"],
        "model_options": {
            "qwen3.8-2.4t-a95b": {
                "topologies": [{"name": "catalog_reference", "tp": 32, "ep": 32,
                                "pp": 4, "dp": 1, "devices": 128}],
                "prefill_tokens": [128], "decode_cases": [],
            },
            "deepseek-v4-flash-0731": {
                "topologies": [{"name": "catalog_reference", "tp": 8, "ep": 8,
                                "pp": 1, "dp": 1, "devices": 8}],
                "prefill_tokens": [128], "decode_cases": [],
            },
        },
        "budget": 20,
    })
    extraction = extract_conditions(config)
    assert not extraction["failures"], extraction["failures"][:2]
    grouped = [c for c in extraction["conditions"].values()
               if c["family"] == "moe_gate_up"
               and c["worker_recipe"] == "npu_grouped_matmul_list_bf16"]
    assert grouped
    condition = grouped[0]
    geometry = condition["features"]["grouped_geometry"]
    assert condition["shape"]["experts"] == geometry["actual_experts"] == 8
    assert condition["shape"]["routed_tokens"] == sum(geometry["actual_expert_tokens"]) == 256
    assert geometry["actual_expert_tokens"] == [32] * 8
    assert condition["shape"]["rank_experts"] == 16
    assert condition["shape"]["rank_routed_tokens"] == 1280
    assert condition["shape"]["global_experts"] == 512
    down = [c for c in extraction["conditions"].values()
            if c["family"] == "moe_down"
            and c["worker_recipe"] == "npu_grouped_matmul_list_bf16"]
    assert down
    assert down[0]["shape"]["experts"] == 8
    assert down[0]["shape"]["routed_tokens"] == 256
    assert down[0]["shape"]["rank_experts"] == 16
    assert sum(down[0]["features"]["grouped_geometry"]["actual_expert_tokens"]) == 256
    softmax = [c for c in extraction["conditions"].values()
               if c["family"] == "attention_softmax"
               and c["precision"]["input_dtype"] == "fp32"]
    assert softmax
    precision = softmax[0]["precision"]
    assert precision["compute_dtype"] == "fp32"
    assert precision["quantization"] == "none"
    assert precision["weight_format"] is None
    assert precision["scale_format"] == ""
    assert precision["weight_block"] == [0, 0]


def test_generic_scope_is_part_of_grouping_export_and_query():
    from mindie_sim.a3_operator_curves import analyze
    from mindie_sim.a3_operator_curve_forecast import (
        export_curve_artifact, query_curve_artifact,
    )
    from mindie_sim.a3_operator_curves_adapter import adapt_linearity_split_results
    scopes = ("plain_contiguous_bf16_shape_probe; not native kernel identity",
              "plain_bf16_M_axis_and_tile_neighbors")
    cases = []
    for scope_index, scope in enumerate(scopes):
        for x in (8, 16, 32):
            cases.append({
                "request": {"case_id": f"scope{scope_index}-{x}", "op_family": "gemm",
                            "shape": {"m": x, "n": 128, "k": 128}, "dtype": "bf16",
                            "layout": "contiguous",
                            "features": {"weight_dtype": "bf16",
                                         "activation_dtype": "bf16",
                                         "compute_dtype": "bf16",
                                         "accumulation_dtype": "backend_default",
                                         "quantization": "none"},
                            "coverage_scope": scope, "sweeps": [{"axis": "m"}]},
                "campaign_split": "train",
                "production_result": {
                    "status": "completed", "source_class": "measured",
                    "run_id": "scope-run", "raw_samples_ms": [1.0] * 3,
                    "duration_ms": 1.0,
                    "environment": {"device_name": "SyntheticDevice"},
                    "protocol": {"timer": "device_event", "warmup": 10,
                                 "iterations": 30, "trials": 1, "preallocated": True,
                                 "cache_policy": "reuse"},
                },
            })
    manifest, results = adapt_linearity_split_results({"cases": cases})
    assert len({request["condition_key"] for request in manifest["requests"]}) == 2
    assert len(analyze(manifest, results)["analyses"]) == 2
    artifact = export_curve_artifact(manifest, results)
    assert artifact["curve_count"] == 2
    assert {curve["physical"]["coverage_scope"] for curve in artifact["curves"]} == set(scopes)
    for curve in artifact["curves"]:
        query = _curve_query(curve, {"m": 12, "n": 128, "k": 128})
        exact = query_curve_artifact(artifact, query)
        assert exact["status"] == "ok"
        assert exact["match_scope"] == "exact_physical_candidate"
        # The same physical cell carries two declared probe series; resolution
        # is deterministic and reported, so both series' own queries agree.
        assert exact["curve_selection"]["candidates"]
        # A declared-probe description is not a physical fact.  A fresh heldout
        # condition can only carry a heldout label, so a scope that names no
        # training series is still answered from the physically identical cell,
        # labelled and with both scope labels reported (round-2 policy).
        crossed = copy.deepcopy(query)
        crossed["coverage_scope"] = "unrelated_scope_not_in_artifact"
        answer = query_curve_artifact(artifact, crossed)
        assert answer["status"] == "ok"
        assert answer["match_scope"] == "physical_cell_scope_agnostic"
        assert answer["scope_agnostic"] is True
        assert answer["declared_coverage_scope"] == "unrelated_scope_not_in_artifact"
        assert answer["measured_coverage_scopes"] == sorted(scopes)
        assert answer["production_match"] is False
        # Physical mismatches still refuse.
        other_dtype = copy.deepcopy(crossed)
        other_dtype["precision"] = {**crossed["precision"], "output_dtype": "fp32"}
        assert query_curve_artifact(artifact, other_dtype)["status"] == "unsupported"
        other_protocol = copy.deepcopy(crossed)
        other_protocol["protocol"] = {**crossed["protocol"], "warmup": 11}
        assert query_curve_artifact(artifact, other_protocol)["status"] == "unsupported"
        other_shape = copy.deepcopy(crossed)
        other_shape["shape"] = {"m": 12, "n": 256, "k": 128}
        assert query_curve_artifact(artifact, other_shape)["status"] == "unsupported"
    reversed_manifest, reversed_results = adapt_linearity_split_results(
        {"cases": list(reversed(cases))})
    reversed_artifact = export_curve_artifact(reversed_manifest, reversed_results)
    assert {curve["curve_id"] for curve in reversed_artifact["curves"]} == {
        curve["curve_id"] for curve in artifact["curves"]}


def test_native_export_uses_actual_result_protocol_identity():
    from mindie_sim.a3_operator_curves_adapter import adapt_campaign_manifest_results
    from mindie_sim.a3_operator_curve_forecast import (
        export_curve_artifact, query_curve_artifact,
    )
    shallow = {"timer": "device_event", "warmup": 10, "iterations": 30, "trials": 1,
               "preallocated": True, "cache_policy": "conflicting_request_cache"}
    full = {"timer": "device_event", "warmup": 10, "iterations": 30, "trials": 1,
            "preallocated": False,
            "cache_policy": "same_preallocated_buffers_reused_no_cache_flush",
            "timing_scope": "device_event_warm_operator"}
    condition = {
        "family": "rmsnorm", "worker_recipe": "npu_rms_norm",
        "execution_kind": "fused_kernel", "status": "ready",
        "precision": {"input_dtype": "bf16", "output_dtype": "bf16",
                      "compute_dtype": "bf16", "accumulation_dtype": "backend_default"},
        "layout": "contiguous", "features": {"epsilon": 1e-6},
    }
    requests, results = [], []
    for x in (8, 16):
        midpoint = f"native-protocol-{x}"
        requests.append({
            "measurement_id": midpoint, "request_key": midpoint,
            "condition_key": "native:protocol", "op_family": "rmsnorm",
            "point": {"axis": "tokens", "value": x,
                      "shape": {"tokens": x, "width": 128}},
            "condition": condition, "worker_recipe": "npu_rms_norm",
            "execution_kind": "fused_kernel", "backend": "torch_npu_installed_api",
            "split": "train", "protocol": shallow, "target_match": False,
            "coverage_scope": "native_primitive", "query_context": {"scope": "native"},
            "origins": [{"anchor_family": "rmsnorm", "anchor_width": 128,
                         "axis": "tokens"}],
            "work": {"logical_input_elements": x * 128},
        })
        results.append({
            "measurement_id": midpoint, "status": "completed",
            "source_class": "measured", "run_id": "native-run", "split": "train",
            "protocol": full, "trials": [{"trial_index": 0, "median_ms": 1.0,
                                          "raw_samples_ms": [1.0] * 3}],
            "observed_context": {"observed_api": "torch_npu.npu_rms_norm",
                                 "execution_kind": "fused_kernel"},
            "environment": {"device_name": "SyntheticNPU"},
            "correctness": {"status": "passed"},
        })
    adapted_manifest, adapted_results = adapt_campaign_manifest_results(
        {"requests": requests}, {"status": "completed", "results": results})
    artifact = export_curve_artifact(adapted_manifest, adapted_results)
    assert artifact["curve_count"] == 1
    curve = artifact["curves"][0]
    assert curve["protocol"] == full
    assert curve["request_protocol"] == shallow
    query = _curve_query(curve, {"tokens": 12, "width": 128})
    assert query_curve_artifact(artifact, query)["status"] == "ok"
    shallow_query = copy.deepcopy(query)
    shallow_query["protocol"] = shallow
    assert query_curve_artifact(artifact, shallow_query)["status"] == "unsupported"


def _attention_train_template():
    cases = []
    for q in (8, 16):
        cases.append({
            "case_id": f"fa-qkv-{q}", "operator": "fusion_attention",
            "dtype": "bfloat16", "layout": "BNSD",
            "shape": {"batch": 1, "q_heads": 2, "kv_heads": 1,
                      "q_tokens": q, "kv_tokens": q, "qk_dim": 64, "v_dim": 64},
            "mask": "causal_bottom_right", "scale": 0.125,
            "data": {"seed": 1, "low": -1.0, "high": 1.0},
            "source": {"model": "m", "source_scope": "s", "primitive_scope": "p",
                       "axis": "q_kv", "value": q, "topology": {"tp": 8},
                       "global_q_heads": 16, "global_kv_heads": 4, "head_dim": 64},
            "target_match": False, "split": "train"})
    for kv in (127, 128):
        cases.append({
            "case_id": f"fa-kv-{kv}", "operator": "fusion_attention",
            "dtype": "bfloat16", "layout": "BNSD",
            "shape": {"batch": 1, "q_heads": 2, "kv_heads": 1,
                      "q_tokens": 1, "kv_tokens": kv, "qk_dim": 64, "v_dim": 64},
            "mask": "full_visible", "scale": 0.125,
            "data": {"seed": 2, "low": -1.0, "high": 1.0},
            "source": {"model": "m", "source_scope": "s", "primitive_scope": "p",
                       "axis": "kv", "value": kv, "topology": {"tp": 8},
                       "global_q_heads": 16, "global_kv_heads": 4, "head_dim": 64},
            "target_match": False, "split": "train"})
    for seq in (8, 128):
        cases.append({
            "case_id": f"rope-{seq}", "operator": "rotary_mul",
            "dtype": "bfloat16", "layout": "BSND",
            "shape": {"batch": 1, "seq": seq, "heads": 2, "dim": 64},
            "rotary_mode": "half",
            "positions": {"start": 1, "end": seq}, "inv_freq_base": 10000.0,
            "data": {"seed": 3, "low": -1.0, "high": 1.0},
            "source": {"model": "m", "source_scope": "s", "primitive_scope": "p",
                       "axis": "seq", "value": seq, "partial_rotary_factor": 0.25,
                       "full_head_dim": 256, "rotated_dim": 64},
            "target_match": False, "split": "train"})
    return {"kind": "a3_attention_rope_manifest", "manifest_id": "series",
            "protocol": {"warmup_iterations": 10, "trials": 2,
                         "samples_per_trial": 30, "seed": 1,
                         "reference": {"atol": 0.02, "rtol": 0.02},
                         "budgets_seconds": {"case": 24, "setup": 60,
                                             "global_work": 240, "cleanup": 60},
                         "supervisor_seconds_min": 370},
            "cases": cases}


def test_attention_derived_axes_form_series_and_query_coupling():
    from mindie_sim.a3_operator_attention_adapter import (
        attention_query_from_request, attention_results_to_campaign,
        campaign_requests_from_attention,
    )
    from mindie_sim.a3_operator_curve_forecast import (
        export_curve_artifact, query_curve_artifact,
    )
    from mindie_sim.a3_operator_curves import analyze
    template = _attention_train_template()
    requests = campaign_requests_from_attention(template, hardware_key="SyntheticNPU")
    assert [r["point"]["axis"] for r in requests] == [
        "q_kv", "q_kv", "kv", "kv", "tokens", "tokens"]
    assert len({r["condition_key"] for r in requests}) == 3
    assert all(r["condition"]["precision"]["compute_dtype"] == "backend_default"
               for r in requests)
    attention_results = {
        "run_id": "r", "protocol": template["protocol"], "environment": {},
        "cases": [
            {"case_id": r["attention_case"]["case_id"],
             "operator": r["attention_case"]["operator"],
             "status": "measured", "target_match": False, "split": "train",
             "native": {}, "correctness": {"status": "passed"},
             "trials": [{"trial_index": 0, "status": "measured",
                         "warmup": {"iterations": 10},
                         "samples": [{"trial_index": 0, "sample_index": 0,
                                      "cpu_submit_ms": 0.1,
                                      "device_interval_ms": 1.0,
                                      "wall_complete_ms": 1.2,
                                      "events_reused": True}]}]}
            for r in requests],
    }
    converted = attention_results_to_campaign(attention_results, requests)
    assert len(analyze({"campaign_id": "a", "requests": requests},
                       converted)["analyses"]) == 3
    artifact = export_curve_artifact({"campaign_id": "a", "requests": requests},
                                     converted)
    assert artifact["curve_count"] == 3
    for request in requests:
        result = query_curve_artifact(
            artifact, attention_query_from_request(request))
        assert result["status"] == "ok", (request["point"], result)
    bad = attention_query_from_request(requests[0])
    bad["shape"] = {**bad["shape"], "kv": bad["shape"]["q"] + 1}
    assert query_curve_artifact(artifact, bad)["status"] == "unsupported"


def test_generic_derived_axes_and_queries_preserve_coupling():
    from mindie_sim.a3_operator_curves_adapter import (
        adapt_campaign_manifest_results, campaign_query_from_request,
    )
    from mindie_sim.a3_operator_curve_forecast import (
        export_curve_artifact, query_curve_artifact,
    )
    protocol = {"timer": "device_event", "warmup": 10, "iterations": 30,
                "trials": 2, "preallocated": True, "cache_policy": "reuse",
                "warmup_scope": "before_each_trial",
                "warmup_iterations_per_trial": 10, "warmup_events_total": 20}
    requests, results = [], []
    for value in (32, 64):
        mid = f"qkv-{value}"
        condition = {"family": "attention_softmax", "recipe": "masked",
                     "execution_kind": "fused_kernel", "worker_recipe": "softmax",
                     "backend": "torch_npu", "target_match": False,
                     "shape_kind": "attention_softmax", "shape": {"q": value, "kv": value, "heads": 2},
                     "precision": {"input_dtype": "bf16", "output_dtype": "bf16",
                                   "compute_dtype": "backend_default"},
                     "layout": "contiguous", "quantization": "none",
                     "features": {"mask": "causal", "scale": 1.0}}
        requests.append({"measurement_id": mid, "request_key": mid,
                         "condition_key": mid, "op_family": "attention_softmax",
                         "condition": condition, "worker_recipe": "softmax",
                         "execution_kind": "fused_kernel", "backend": "torch_npu",
                         "split": "train", "protocol": protocol,
                         "point": {"axis": "q_kv", "value": value,
                                   "shape": {"q": value, "kv": value, "heads": 2}},
                         "target_match": False, "coverage_scope": "probe",
                         "origins": [{"axis": "q_kv"}]})
        results.append({"measurement_id": mid, "condition_key": mid,
                        "status": "completed", "source_class": "measured",
                        "run_id": "r", "split": "train", "protocol": protocol,
                        "trials": [{"trial_index": 0, "median_ms": 1.0,
                                    "raw_samples_ms": [1.0] * 3}],
                        "environment": {"device_name": "SyntheticNPU"},
                        "correctness": {"status": "passed"}})
    for value in (1, 8):
        mid = f"moe-{value}"
        counts = [value] * 8
        condition = {"family": "moe_gate_up", "recipe": "grouped",
                     "execution_kind": "fused_kernel", "worker_recipe": "grouped",
                     "backend": "torch_npu", "target_match": False,
                     "shape_kind": "moe_grouped",
                     "shape": {"experts": 8, "routed_tokens": value * 8,
                               "k": 128, "n": 128},
                     "precision": {"input_dtype": "bf16", "output_dtype": "bf16",
                                   "compute_dtype": "bf16"},
                     "layout": "contiguous", "quantization": "none",
                     "features": {"routing": {"local_experts": 8,
                                              "local_expert_counts": counts}}}
        requests.append({"measurement_id": mid, "request_key": mid,
                         "condition_key": mid, "op_family": "moe_gate_up",
                         "condition": condition, "worker_recipe": "grouped",
                         "execution_kind": "fused_kernel", "backend": "torch_npu",
                         "split": "train", "protocol": protocol,
                         "point": {"axis": "tokens_per_expert", "value": value,
                                   "shape": {"experts": 8, "routed_tokens": value * 8,
                                             "k": 128, "n": 128}},
                         "target_match": False, "coverage_scope": "probe",
                         "origins": [{"axis": "tokens_per_expert"}]})
        results.append({"measurement_id": mid, "condition_key": mid,
                        "status": "completed", "source_class": "measured",
                        "run_id": "r", "split": "train", "protocol": protocol,
                        "trials": [{"trial_index": 0, "median_ms": 1.0,
                                    "raw_samples_ms": [1.0] * 3}],
                        "environment": {"device_name": "SyntheticNPU"},
                        "correctness": {"status": "passed"}})
    adapted, adapted_results = adapt_campaign_manifest_results(
        {"requests": requests}, {"results": results})
    assert [r["point"]["axis"] for r in adapted["requests"]] == [
        "q_kv", "q_kv", "tokens_per_expert", "tokens_per_expert"]
    assert adapted["requests"][0]["point"]["shape"]["q"] == 32
    assert adapted["requests"][3]["point"]["shape"]["routed_tokens"] == 64
    assert len({r["condition_key"] for r in adapted["requests"]}) == 2
    artifact = export_curve_artifact(adapted, adapted_results)
    assert artifact["curve_count"] == 2
    for request, result in zip(adapted["requests"], adapted_results["results"]):
        query = campaign_query_from_request(request, result["protocol"])
        assert query_curve_artifact(artifact, query)["status"] == "ok"
    bad = campaign_query_from_request(adapted["requests"][0])
    bad["shape"] = {**bad["shape"], "kv": bad["shape"]["q"] + 1}
    assert query_curve_artifact(artifact, bad)["status"] == "unsupported"
    bad = campaign_query_from_request(adapted["requests"][2])
    bad["shape"] = {**bad["shape"], "routed_tokens": 9}
    assert query_curve_artifact(artifact, bad)["status"] == "unsupported"


def test_unbalanced_grouped_histogram_is_not_wildcarded():
    from mindie_sim.a3_operator_curves_adapter import _series_identity_features
    balanced = {"routing": {"local_experts": 8,
                             "local_expert_counts": [2] * 8}}
    unbalanced = {"routing": {"local_experts": 8,
                               "local_expert_counts": [4, 4, 2, 2, 2, 2, 0, 0]}}
    first = _series_identity_features("tokens_per_expert", balanced, 2)
    second = _series_identity_features("tokens_per_expert", unbalanced, 2)
    assert first["routing"]["local_expert_counts"] == [1] * 8
    assert second["routing"]["local_expert_counts"] == [2, 2, 1, 1, 1, 1, 0, 0]
    assert first != second


def test_shared_source_protocol_parity_for_db_and_curve_export():
    from mindie_sim.a3_operator_curves_adapter import (
        _calibration_case, adapt_linearity_split_results,
    )
    from mindie_sim.a3_operator_curve_protocol import canonical_protocol
    case = {
        "request": {"case_id": "parity-real-shape", "op_family": "gemm",
                    "shape": {"m": 8, "n": 256, "k": 128}, "dtype": "bf16",
                    "layout": "contiguous", "features": {"weight_dtype": "bf16"},
                    "sweeps": [{"axis": "m"}]},
        "campaign_split": "train",
        "production_result": {
            "status": "completed", "source_class": "measured",
            "run_id": "parity-run", "environment": {"device_name": "Ascend910_9362"},
            "raw_samples_ms": [1.0] * 30, "duration_ms": 1.0,
            "timing_scope": "device_event_warm",
            "protocol": {"timer": "device_event", "warmup": 10,
                         "iterations": 30, "preallocated": True,
                         "measurement_scope": "inclusive",
                         "layout": "contiguous", "weight_dtype": "bf16"},
        },
    }
    db_case, not_a3 = _calibration_case(case, 0)
    assert not not_a3
    _manifest, results = adapt_linearity_split_results({"cases": [case]})
    assert canonical_protocol(db_case["protocol"]) == canonical_protocol(
        results["results"][0]["protocol"])
    assert db_case["protocol"]["timing_scope"] == "device_event_warm"
    assert db_case["protocol"]["trials"] == 1
    assert "cache_policy" not in db_case["protocol"]
    explicit = copy.deepcopy(case)
    explicit["production_result"]["protocol"]["trials"] = 3
    explicit_case, _ = _calibration_case(explicit, 0)
    assert explicit_case["protocol"]["trials"] == 3


def test_measured_hardware_uses_observed_environment_not_request_placeholder():
    from mindie_sim.a3_operator_curves_adapter import (
        adapt_campaign_manifest_results, campaign_query_from_request,
    )
    from mindie_sim.a3_operator_curve_forecast import (
        export_curve_artifact, query_curve_artifact,
    )
    protocol = {"timer": "device_event", "warmup": 10, "iterations": 30,
                "trials": 2, "preallocated": True, "cache_policy": "reuse",
                "warmup_scope": "before_each_trial",
                "warmup_iterations_per_trial": 10, "warmup_events_total": 20}
    requests, results = [], []
    for x in (8, 16):
        mid = f"hw-{x}"
        condition = {"family": "gemm", "recipe": "dense", "execution_kind": "primitive",
                     "worker_recipe": "torch.mm", "backend": "torch.mm",
                     "target_match": False, "shape_kind": "gemm",
                     "shape": {"m": x, "n": 128, "k": 128},
                     "precision": {"input_dtype": "bf16", "weight_dtype": "bf16",
                                   "output_dtype": "bf16", "compute_dtype": "bf16"},
                     "layout": "contiguous", "quantization": "none", "features": {}}
        requests.append({"measurement_id": mid, "request_key": mid,
                         "condition_key": mid, "op_family": "gemm",
                         "condition": condition, "worker_recipe": "torch.mm",
                         "execution_kind": "primitive", "backend": "torch.mm",
                         "split": "train", "protocol": protocol,
                         "hardware_key": "declared-a3-operator-campaign",
                         "point": {"axis": "m", "value": x,
                                   "shape": {"m": x, "n": 128, "k": 128}},
                         "target_match": False, "coverage_scope": "probe",
                         "origins": [{"axis": "m"}]})
        results.append({"measurement_id": mid, "condition_key": mid,
                        "status": "completed", "source_class": "measured",
                        "run_id": "r", "split": "train", "protocol": protocol,
                        "trials": [{"trial_index": 0, "median_ms": 1.0,
                                    "raw_samples_ms": [1.0] * 3}],
                        "environment": {"device_name": "Ascend910_9362"},
                        "correctness": {"status": "passed"}})
    manifest, result_payload = adapt_campaign_manifest_results(
        {"run_id": "r", "requests": requests}, {"results": results})
    assert manifest["requests"][0]["hardware_key"] == "Ascend910_9362"
    assert manifest["requests"][0]["requested_hardware_key"] == "declared-a3-operator-campaign"
    assert result_payload["results"][0]["hardware_key"] == "Ascend910_9362"
    artifact = export_curve_artifact(manifest, result_payload)
    source = requests[0]
    assert query_curve_artifact(
        artifact, campaign_query_from_request(
            source, hardware_key="Ascend910_9362"))["status"] == "ok"
    assert query_curve_artifact(
        artifact, campaign_query_from_request(source))["status"] == "unsupported"
    assert query_curve_artifact(
        artifact, campaign_query_from_request(
            source, hardware_key="otherNPU"))["status"] == "unsupported"
