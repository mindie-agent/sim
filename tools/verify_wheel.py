"""Exercise the installed public CLI outside the source checkout.

[Role] Verify wheel contents, entry points and observation interoperability.
[Invariants] Use real captures for imports; heldout fixtures never calibrate
  the serving examples. Installation must not resolve the source package.
[Timeouts & Locks] Environment setup <= 120 s; each CLI call <= 60 s.
[Side-Effect Policy] Creates a temporary virtualenv and the requested report.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile


def verify_dsq_arch35_caller(run, work):
    """Reuse the installed MC2 INT8 caller, retaining allocated vs valid rows."""
    raw = json.loads((work / "mc2-tp-int8-grouped-caller.json").read_text())
    raw["runtime"]["kernels"]["lowerings"]["grouped_int8_caller"] = {
        "implementation": "ascend_grouped_int32_dequant_swiglu",
        "options": {"dequant_program": "arch35_full_load", "input_capacity_rows": 128}}
    selected = work / "mc2-tp-int8-dsq-arch35.json"
    selected.write_text(json.dumps(raw) + "\n")
    args = [selected, "--scope", "body", "--phase", "decode", "--tokens", 4,
            "--sequences", 4, "--context", 128]
    result = run(["graph", *args, "--input-ready-ms", 7], "graph-source-dsq-arch35-caller")
    assert result["duration_ms"] > 0
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert not any(result["execution"]["live_allocated"].values())
    assert all(row["ready_ms"] >= 7 for row in result["timeline"])
    op = next(row["operator"] for row in result["isolated"]["operators"]
              if row["operator"]["recipe"] == "grouped_int32_dequant_swiglu")
    schedule = op["metadata"]["program_execution"]["grouped_dequant_schedule"]
    assert schedule["kind"] == "arch35_full_load"
    assert schedule["input_capacity_rows"] == 128 and schedule["valid_rows"] == 32
    assert op["local_shape"] == [32, 512] and schedule["host_special_group_cut"]
    assert op["metadata"]["activation_scale_shape"] == [32]
    assert op["mte3_bytes"] == 32 * (512 + 4)
    structure = run(args, "structure-source-dsq-arch35-allocated-abi", module="mindie_sim.structure")
    node = next(n for n in structure["nodes"] if n["recipe"] == "grouped_int32_dequant_swiglu")
    assert node["contract"][2][0][0] == [128, 1024]
    assert node["contract"][2][2][0] == [128]
    assert node["contract"][3][0][0] == [128, 512]


def verify_composable_paths(run, root, work):
    """Exercise raw official configs at small geometry through public CLIs."""
    base = json.loads((root / "examples/quickstart.json").read_text())
    base["workload"] = {"requests": [{"id": "small", "prompt_tokens": 4, "output_tokens": 1}]}
    base["runtime"]["graph_mode"] = "none"
    base["machine"]["device"]["compute_peaks"]["fp8_e4m3"] = 2e14

    def write(name, payload):
        path = work / (name + ".json")
        path.write_text(json.dumps(payload) + "\n")
        return path

    def small_config(directory, name):
        config = json.loads((root / "tests/fixtures" / directory / (name + ".json")).read_text())
        text = config.get("text_config", config)
        text.update(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, vocab_size=128,
                    head_dim=16 if name.startswith("qwen3") else 8)
        for key in ("n_routed_experts", "num_experts", "num_local_experts", "moe_num_experts"):
            if text.get(key):
                text[key] = 4
                text.update(num_experts_per_tok=2, moe_intermediate_size=32,
                            first_k_dense_replace=1)
        for key in ("n_group", "topk_group"):
            if key in text:
                text[key] = 1
        if "shared_expert_intermediate_size" in text:
            text["shared_expert_intermediate_size"] = 48
        if text.get("kv_lora_rank"):
            text.update(kv_lora_rank=16, qk_nope_head_dim=8,
                        qk_rope_head_dim=4, v_head_dim=8)
            if text.get("q_lora_rank") is not None:
                text["q_lora_rank"] = 16
        return config

    def finite(report):
        assert report["duration_ms"] > 0
        assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
        assert not any(report["execution"]["live_allocated"].values())

    controls = {}
    for directory, name in (
        ("standard-qwen-kimi", "qwen2-7b"),
        ("standard-qwen-kimi", "qwen3-30b-a3b"),
        ("hf-standard-families", "deepseek-v3"),
        ("hf-standard-families", "glm4-0414"),
        ("hf-standard-families", "glm45"),
        ("standard-qwen-kimi", "kimi-k2-instruct"),
    ):
        raw = deepcopy(base)
        raw["model"] = small_config(directory, name)
        path = write("hf-" + name, raw)
        graph = run(["graph", path, "--phase", "prefill", "--tokens", 3],
                    "graph-hf-" + name)
        finite(graph)
        rows = graph["isolated"]["operators"]
        operators = [row["operator"] for row in rows]
        assert sum(row["occurrences"] for row in rows
                   if row["operator"]["family"] == "attention") == 2
        assert any(op["family"] == "logits" for op in operators)
        assert any(op["flops"] > 0 for op in operators)
        if name.startswith("qwen"):
            qkv = [row for row in rows if row["operator"]["family"] == "qkv_projection"]
            width = 16 if name.startswith("qwen3") else 8
            assert sum(row["occurrences"] for row in qkv) == 2
            assert sum(row["operator"]["flops"] * row["occurrences"] for row in qkv) == (
                2 * 2 * 3 * 32 * 8 * width)
        controls[name] = raw, graph

    for name in ("minimax-m1-80k", "step3"):
        raw = deepcopy(base)
        config = small_config("composable-hf", name)
        text = config.get("text_config", config)
        if "attn_type_list" in text:
            text["attn_type_list"] = [0, 1]
        else:
            text.update(moe_layers_enum="1", moe_top_k=2, share_expert_dim=16, share_q_dim=16)
        raw["model"] = config
        path = write("hf-" + name, raw)
        graph = run(["graph", path, "--phase", "prefill", "--tokens", 3],
                    "graph-hf-partial-" + name)
        serving = run(["predict", path], "predict-hf-partial-" + name)
        finite(graph)
        assert serving["status"] == "modeled"
        assert serving["requests"][0]["status"] == "served"
        assert serving["execution"]["live_tasks"] == serving["execution"]["live_programs"] == 0
        assert graph["model_coverage"] == serving["model_coverage"] == "partial"
        assert graph["unmodeled_features"] and serving["unmodeled_features"]
        assert any(row["family"] == "barrier" for row in graph["timeline"])

    raw, baseline = controls["qwen3-30b-a3b"]
    partial = deepcopy(raw)
    partial["model"] = {"hf_config": {**raw["model"], "custom_component": {"order": 1}}}
    path = write("hf-override-study", {
        "level": "graph", "candidates": [
            {"id": "base", "case": partial},
            {"id": "deeper", "case": partial, "overrides": {"model": {"num_layers": 3}}},
            {"id": "other-component", "case": partial,
             "overrides": {"model": {"custom_component": {"order": 2}}}},
        ], "probes": [{"id": "p3", "phase": "prefill", "tokens": 3}],
    })
    result = run(["study", path], "study-hf-overrides-and-coverage")
    original, deeper, changed = result["evaluations"]
    assert result["status"] == "completed"
    assert deeper["operators"]["work"]["flops"] > original["operators"]["work"]["flops"]
    assert changed["operators"]["work"] == original["operators"]["work"]
    assert len({row["workload_key"] for row in result["evaluations"]}) == 3
    assert len(result["rankings"]) == 3
    assert all(row["model_coverage"] == "partial" for row in result["evaluations"])
    assert result["coverage_summary"] == {"complete_candidates": 0, "partial_candidates": 3}

    timed = deepcopy(raw)
    timed["machine"]["device"]["kernel_invocations"] = {
        "vector": {"minimum_ms": .0025, "provenance": {"basis": "synthetic_wheel_control"}},
        "cube": {"minimum_ms": .005, "provenance": {"basis": "synthetic_wheel_control"}},
    }
    path = write("hf-hardware-minimum", timed)
    isolated = run(["operators", path, "--phase", "prefill", "--tokens", 3],
                   "operators-hardware-kernel-minimum")
    constrained = run(["graph", path, "--phase", "prefill", "--tokens", 3],
                      "graph-hardware-kernel-minimum")
    prior = {row["operator"]["op_id"]: row for row in baseline["isolated"]["operators"]}
    for row in isolated["operators"]:
        op = row["operator"]
        before = prior[op["op_id"]]
        minimum = .005 if op["flops"] else .0025 if op["vector_ops"] else 0
        assert math.isclose(row["duration_ms"], max(before["duration_ms"], minimum),
                            rel_tol=1e-10, abs_tol=1e-12)
        assert row["serviced"] == before["serviced"]
    finite(constrained)
    assert constrained["duration_ms"] > baseline["duration_ms"]
    assert constrained["execution"]["work"] == baseline["execution"]["work"]
    assert constrained["parameters"]["invocation_models"]


def verify_source_boundaries(run, root, work):
    """Execute selected source programs through the installed public frontend."""
    def graph(raw, name, tokens=4):
        path = work / (name + ".json")
        path.write_text(json.dumps(raw) + "\n")
        result = run(["graph", path, "--phase", "prefill", "--tokens", tokens], name)
        assert result["duration_ms"] > 0
        assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
        assert not any(result["execution"]["live_allocated"].values())
        return result

    raw = json.loads((root / "examples/source-families/kda-state-control.json").read_text())
    raw["runtime"]["kernels"]["lowerings"]["kda"] = "vllm_ascend_kda_chunk_prepare_state"
    raw["machine"]["device"]["rates"]["reg_special_lanes"] = 8e11
    raw["machine"]["device"]["resource_units"].update(reg_special_lanes=4, mte3=4)
    kda = graph(raw, "graph-source-kda-prepare-compact", 128)
    assert any(row["operator"]["recipe"] == "vllm_ascend_kda_gate_prepare"
               and row["operator"]["source_boundary"] == "kernel"
               for row in kda["isolated"]["operators"])

    raw = json.loads((root / "examples/source-families/w4a8-msd-control.json").read_text())
    raw["model"]["num_shared_experts"] = 1
    raw["model"]["quant"].update(weight_format="int8", expert_weight_format="int8")
    raw["runtime"]["kernels"]["lowerings"] = {"shared_expert": "ascend_shared_gate_up_int32"}
    raw["runtime"]["kernels"]["moe_transport"] = {
        "kind": "separate_quantized_payload", "payload_dtype": "int8",
        "scale_dtype": "fp32", "scale_granularity": "row", "histogram_dtype": "int64",
    }
    shared = graph(raw, "graph-shared-int32-and-typed-transport")
    operators = [row["operator"] for row in shared["isolated"]["operators"]]
    assert any(op["metadata"].get("output_format") == "int32" for op in operators)
    assert any(op["family"] == "dequant_swiglu_quant" for op in operators)

    raw = json.loads((root / "tests/fixtures/verification/"
                     "v4flash-sharedkv-case.json").read_text())
    raw["model"]["layers"] = 3
    raw["runtime"]["pp_layer_partition"] = [3]
    raw["runtime"].pop("routing", None)
    raw["runtime"]["kernels"] = {
        "programs": {"matmul": {"kind": "steady_matmul"}},
        "lowerings": {"sharedkv_attention": {
            "implementation": "vllm_ascend_sharedkv_scfa",
            "options": {"owner_schedule": "one_query_per_core",
                        "selected_rows": "valid_prefix", "sinks_dtype": "fp32",
                        "metadata_s2_split": False, "return_softmax_lse": False},
        }},
    }
    # Exercise the installed SKU's nominal pipeline declaration, not injected
    # synthetic rates. This establishes executable supply, not measured accuracy.
    scfa = graph(raw, "graph-source-scfa-boundary")
    attention = [row for row in scfa["timeline"] if row["family"] == "attention"]
    assert len(attention) == 3
    assert sum(row["recipe"] == "vllm_ascend_sharedkv_scfa"
               and row["source_boundary"] == "kernel" for row in attention) == 1

    raw = json.loads((root / "examples/source-families/"
                     "indexed-mla-a3-control.json").read_text())
    indexed = graph(raw, "graph-source-indexed-mla-a3", 21)
    source = [row for row in indexed["timeline"]
              if row["recipe"] == "ascend_sparse_flash_attention_arch22"]
    assert len(source) == 1 and source[0]["source_boundary"] == "kernel"
    path = work / "graph-source-indexed-mla-a3.json"
    decode = run(["graph", path, "--phase", "decode", "--tokens", 1,
                  "--context", 4096], "graph-source-indexed-mla-a3-decode")
    assert decode["duration_ms"] > 0
    assert decode["execution"]["live_tasks"] == decode["execution"]["live_programs"] == 0
    assert not any(decode["execution"]["live_allocated"].values())
    assert any(row["recipe"] == "ascend_sparse_flash_attention_arch22"
               and row["source_boundary"] == "kernel" for row in decode["timeline"])

    raw = json.loads((root / "examples/source-families/"
                     "sequence-parallel-control.json").read_text())
    sequence = graph(raw, "graph-sequence-parallel-complete-cohort")
    assert any(row["family"] == "reduce_scatter" for row in sequence["timeline"])
    assert not any(row["family"] == "allreduce" for row in sequence["timeline"])

    raw = json.loads((root / "examples/quickstart.json").read_text())
    raw["runtime"]["kernels"]["recipes"] = {
        "flash_attention_prefill": {"kind": "staged_gqa_prefill"},
    }
    prefill = graph(raw, "graph-staged-gqa-prefill", 65)
    attention = [row for row in prefill["isolated"]["operators"]
                 if row["operator"]["family"] == "attention"]
    assert sum(row["occurrences"] for row in attention) == raw["model"]["layers"]
    assert all(row["operator"]["recipe"] == "flash_attention_prefill"
               for row in attention)

    raw = json.loads((root / "examples/source-families/"
                     "attnres-prefix-control.json").read_text())
    attnres = graph(raw, "graph-source-attnres-prefix")
    points = [row for row in attnres["isolated"]["operators"]
              if row["operator"]["recipe"] == "attn_res_fwd"]
    assert sum(row["occurrences"] for row in points) == 2 * raw["model"]["layers"] + 1
    assert all(row["operator"]["source_boundary"] == "kernel"
               and row["operator"]["flops"] == 0 for row in points)

    raw = json.loads((root / "examples/cases/"
                     "kimi-k3-a5-current-checkpoint-conditional.json").read_text())
    raw["model"]["layers"] = 2
    raw["runtime"]["pp_layer_partition"] = [2]
    raw["runtime"]["kernels"] = {
        "programs": {"matmul": {"kind": "steady_matmul"}},
        "lowerings": {"native_fusion": {
            "implementation": "declared_native_fusion",
            "options": {"roles": ["situ_quant"],
                        "program_profiles": {"situ_quant": "situ_mx_quant_arch35"}},
        }},
    }
    situ = graph(raw, "graph-source-situ-mx", 1)
    assert any(row["operator"]["recipe"] == "situ_mx_quant_arch35"
               and row["operator"]["source_boundary"] == "kernel"
               for row in situ["isolated"]["operators"])


def verify(wheel, root, output, *, profiler_wheel=None):
    root, wheel = Path(root).resolve(), Path(wheel).resolve()
    work = Path(tempfile.mkdtemp(prefix="sim-wheel-"))
    env = work / "env"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(env)],
                   check=True, timeout=60)
    python = next(path for path in (env / "bin/python", env / "Scripts/python.exe")
                  if path.is_file())
    subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel)],
                   check=True, timeout=120)
    checks = []

    def run(args, name, expected_code=0, *, module="mindie_sim"):
        result = subprocess.run(
            [str(python), "-I", "-m", module, *map(str, args)],
            cwd=work, text=True, capture_output=True, timeout=60)
        if result.returncode != expected_code:
            raise RuntimeError((name, result.stdout, result.stderr))
        checks.append({"command": name, "exit_code": result.returncode})
        return json.loads(result.stdout) if result.stdout.strip() and name != "help" else None

    cli = next(path for path in (env / "bin/mindie-sim", env / "Scripts/mindie-sim.exe")
               if path.is_file())
    subprocess.run([str(cli), "--help"], cwd=work, check=True, capture_output=True, timeout=30)
    checks.append({"command": "installed-entrypoint", "exit_code": 0})
    run(["--help"], "help")
    assert len(run(["catalog"], "catalog")) > 1
    run(["predict", root / "examples/quickstart.json", "-o", work / "prediction.json"], "predict")
    prediction = json.loads((work / "prediction.json").read_text())
    assert prediction["status"] == "modeled"
    assert prediction["metrics"]["output_tokens"] == 14
    body = run(["predict", root / "examples/quickstart.json", "--scope", "body",
                "--phase", "decode", "--tokens", 4, "--context", 2048,
                "--sequences", 2, "--event-limit", 0], "predict-body")
    assert body["body_forward_ms"] > 0 and body["geometry"]["valid_query_tokens"] == 8
    assert body["execution"]["submission"]["host_service_ms"] == 0
    pd_body = run(["predict", root / "examples/disaggregated.json", "--scope", "body",
                   "--tokens", 4, "--context", 2048, "--sequences", 2], "predict-pd-body")
    assert set(pd_body["body_forward_ms"]) == {"prefill", "decode"}
    assert all(value > 0 for value in pd_body["body_forward_ms"].values())
    # Real installed CLI: uneven forced expert banks and wide-head buffer
    # selection must compose without source imports or per-recipe workarounds.
    for sku in ("Ascend910_9362", "Ascend950DT_9582"):
        balanced = {
            "machine": {"backend": "ascend", "sku": sku, "num_devices": 1},
            "model": {"name": "integer-bank-control", "layers": 1, "hidden_size": 2048,
                      "num_heads": 4, "num_kv_heads": 2, "head_dim": 512,
                      "ffn": "moe", "num_experts": 24, "top_k": 7,
                      "moe_intermediate_size": 64, "intermediate_size": 64, "vocab_size": 128},
            "runtime": {"routing": {"policy": "balanced"},
                        "kernels": {"programs": {"matmul": {"kind": "steady_matmul"}}}},
            "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        }
        control_path = work / (sku + "-integer-body.json")
        control_path.write_text(json.dumps(balanced) + "\n")
        report = run(["predict", control_path, "--scope", "body", "--tokens", 5,
                      "--context", 256], "body-integer-banks-and-pipes-" + sku)
        programs = {row["program"] for row in report["timeline"]}
        assert {"IntegerGroupedMatmul", "StagedGQADecode"} <= programs
        assert report["body_forward_ms"] > 0
        assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
        assert report["execution"]["submission"]["host_service_ms"] == 0
        assert report["default_mechanisms"]["gqa_tiles"]["key_tile"] < 128
    verify_composable_paths(run, root, work)
    verify_source_boundaries(run, root, work)
    from verify_decode_mechanisms import verify_decode_mechanisms
    verify_decode_mechanisms(run, root, work)
    from verify_source_callers import verify_source_callers
    verify_source_callers(run, root, work)
    from verify_service_regimes import verify_service_regimes
    verify_service_regimes(run, root, work)
    verify_dsq_arch35_caller(run, work)
    from verify_whitebox_services import verify_whitebox_services
    verify_whitebox_services(run, root, work)
    sort32 = subprocess.run(
        [str(python), '-I', str(root / 'tools/verify_sort32_installed.py')],
        cwd=work, text=True, capture_output=True, timeout=30, check=True)
    sort_report = json.loads(sort32.stdout)
    assert Path(sort_report['package']).is_relative_to(env)
    assert len(sort_report['results']) == 9
    checks.append({'command': 'installed-sort32-owner-capacity-controls', 'exit_code': 0})
    for sku, native in (("Ascend910_9362", False), ("Ascend950DT_9582", False), ("Ascend950DT_9582", True)):
        source = root / "examples/cases/high-concurrency-20261030/source-gdn-native/qwen36-a5-tp8-c4.json"
        raw = json.loads(source.read_text())
        raw["machine"]["sku"], raw["model"]["layers"] = sku, 1
        if not native:
            raw["runtime"]["kernels"].pop("lowerings")
        path = work / f"gdn-body-{sku}-{native}.json"
        path.write_text(json.dumps(raw) + "\n")
        report = run(["predict", path, "--scope", "body", "--tokens", 4, "--context", 128,
                      "--sequences", 4], f"body-gdn-{sku}-{native}")
        recipes = {row["recipe"] for row in report["timeline"]}
        assert {"ascend_causal_conv1d_update", "ascend_gdn_decode_gating"} <= recipes
        if native:
            assert {"ascend_recurrent_gdn_fp32_state", "state_indices_contiguous"} <= recipes
        assert report["body_forward_ms"] > 0
        assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
        assert not any(report["execution"]["live_allocated"].values())
        assert report["execution"]["submission"]["host_service_ms"] == 0
    assert run(["predict", root / "examples/disaggregated.json"], "predict-pd")["status"] == "modeled"
    assert len(run(["inspect", root / "examples/quickstart.json"], "inspect")["operators"]) > 5
    operators = run(["operators", root / "examples/quickstart.json",
                     "--phase", "decode", "--tokens", 4, "--context", 16,
                     "--sequences", 2, "--scope", "model"], "operators")
    assert operators["operator_sum_ms"] > 0 and operators["operator_occurrences"] > 5
    assert any(row["operator"]["family"] == "logits" for row in operators["operators"])
    for scope in ("body", "model"):
        graph = run(["graph", root / "examples/quickstart.json",
                     "--phase", "decode", "--tokens", 1, "--context", 16,
                     "--scope", scope], "graph-" + scope)
        assert graph["duration_ms"] > 0
        assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0
        assert graph["bounds"]["combined_ms"] <= graph["duration_ms"] + 1e-9
    source_graph = run(["graph", root / "examples/source-families/gqa-norm-control.json",
                        "--phase", "prefill", "--tokens", 4], "graph-source-families")
    assert {row["family"] for row in source_graph["timeline"]
            if row["source_boundary"]} == {"rmsnorm", "attention"}
    for filename, phase, tokens, family in (
        ("mla-decode-control.json", "decode", 1, "attention"),
        ("dynamic-int8-control.json", "prefill", 3, "quantize"),
    ):
        graph = run(["graph", root / "examples/source-families" / filename,
                     "--phase", phase, "--tokens", tokens, "--context", 33],
                    "graph-" + family)
        assert any(row["family"] == family and row["source_boundary"]
                   for row in graph["timeline"])
        assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0
    kda_case = root / "examples/source-families/kda-state-control.json"
    kda_graph = run(["graph", kda_case, "--phase", "prefill", "--tokens", 65,
                     "--sequences", 2], "graph-kda-state-source")
    kda_rows = [row for row in kda_graph["isolated"]["operators"]
                if row["operator"]["recipe"] == "vllm_ascend_kda_fwd_h"]
    assert len(kda_rows) == 1
    assert kda_rows[0]["operator"]["metadata"]["lengths"] == [65, 65]
    assert kda_rows[0]["operator"]["flops"] == 4 * 130 * 2 * 128 * 128
    assert kda_rows[0]["program"] == "OperatorPlan"
    assert kda_graph["execution"]["live_tasks"] == kda_graph["execution"]["live_programs"] == 0
    kda_report = work / "kda-installed-call.json"
    subprocess.run(
        [str(python), "-I", str(root / "tools/verify_kda_state.py"), str(kda_case),
         "--rss-byte-factor", "1" if sys.platform == "darwin" else "1024",
         "-o", str(kda_report)], cwd=work, check=True, capture_output=True, timeout=30)
    verified_kda = json.loads(kda_report.read_text())
    assert verified_kda["bounded_active_state"]
    assert verified_kda["two_call_protocol"]["protocol"]["invocations"] == 2
    assert verified_kda["two_call_protocol"]["single_call"]["source_boundaries"] == {"kernel": 1}
    checks.append({"command": "call-kda-state-source-installed", "exit_code": 0})
    moe = root / "examples/source-families/moe-routing-control.json"
    moe_graph = run(["graph", moe, "--phase", "prefill", "--tokens", 4],
                    "graph-moe-align-combine")
    assert any(row["source_boundary"] and row["family"] == "moe_combine"
               for row in moe_graph["timeline"])
    assert moe_graph["execution"]["live_tasks"] == moe_graph["execution"]["live_programs"] == 0
    assert run(["predict", moe], "predict-moe-align-combine")["status"] == "modeled"
    for filename, recipe in (
        ("w4a8-msd-control.json", "ascend_w4a8_msd"),
        ("w4a8-down-msd-control.json", "ascend_w4a8_msd_grouped_down"),
    ):
        path = root / "examples/source-families" / filename
        graph = run(["graph", path, "--phase", "prefill", "--tokens", 4],
                    "graph-" + recipe)
        assert any(row["recipe"] == recipe and row["source_boundary"]
                   for row in graph["timeline"])
        assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0
        assert run(["predict", path], "predict-" + recipe)["status"] == "modeled"
    staged = root / "tests/fixtures/verification/qwen38-staged-attention-case.json"
    staged_graph = run([
        "graph", staged, "--phase", "decode", "--tokens", 1, "--context", 16384,
        "--skip-isolated",
    ], "graph-staged-gqa-decode")
    assert any(row["recipe"] == "flash_attention_decode"
               for row in staged_graph["timeline"])
    assert staged_graph["execution"]["live_tasks"] == staged_graph["execution"]["live_programs"] == 0
    sparse = root / "tests/fixtures/verification/glm53-indexed-attention-case.json"
    sparse_graph = run([
        "graph", sparse, "--phase", "prefill", "--tokens", 4, "--skip-isolated",
    ], "graph-indexed-attention-pipeline")
    assert any(row.get("program") == "indexed_attention_pipeline"
               for row in sparse_graph["parameters"]["matrix_mappings"])
    assert sparse_graph["execution"]["live_tasks"] == sparse_graph["execution"]["live_programs"] == 0
    mapped = json.loads((root / "examples/quickstart.json").read_text())
    mapped["runtime"].setdefault("kernels", {}).setdefault("programs", {})["matmul"] = {
        "kind": "mapped_matmul"}
    (work / "mapped.json").write_text(json.dumps(mapped) + "\n")
    mapped_graph = run(["graph", work / "mapped.json", "--phase", "prefill", "--tokens", 4],
                       "graph-mapped-matrix")
    assert mapped_graph["parameters"]["matrix_mappings"]
    assert mapped_graph["execution"]["live_tasks"] == mapped_graph["execution"]["live_programs"] == 0
    mapped_prediction = run(["predict", work / "mapped.json"], "predict-mapped-matrix")
    assert mapped_prediction["status"] == "modeled"
    assert mapped_prediction["parameters"]["matrix_mappings"]
    mapped["runtime"]["kernels"]["programs"]["matmul"] = {"kind": "phase_matmul"}
    (work / "phase.json").write_text(json.dumps(mapped) + "\n")
    phase_graph = run(["graph", work / "phase.json", "--phase", "prefill", "--tokens", 4],
                      "graph-phase-matrix")
    assert phase_graph["duration_ms"] > 0
    assert phase_graph["execution"]["live_tasks"] == phase_graph["execution"]["live_programs"] == 0
    assert phase_graph["bounds"]["combined_ms"] <= phase_graph["duration_ms"] + 1e-9
    assert phase_graph["parameters"]["matrix_mappings"]
    assert all(row["execution_abstraction"] == "serial_per_owner_resource_phases"
               for row in phase_graph["parameters"]["matrix_mappings"])
    detailed_work = mapped_graph["execution"]["serviced"]
    phase_work = phase_graph["execution"]["serviced"]
    assert detailed_work.keys() == phase_work.keys()
    assert all(math.isclose(detailed_work[key], phase_work[key], rel_tol=1e-12, abs_tol=1e-9)
               for key in detailed_work)
    mapped["runtime"]["kernels"]["programs"]["matmul"] = {"kind": "steady_matmul"}
    (work / "steady.json").write_text(json.dumps(mapped) + "\n")
    steady_graph = run(["graph", work / "steady.json", "--phase", "prefill", "--tokens", 4],
                       "graph-steady-matrix")
    assert steady_graph["execution"]["live_tasks"] == steady_graph["execution"]["live_programs"] == 0
    assert all(row["execution_abstraction"] == "closed_form_k_pipeline"
               for row in steady_graph["parameters"]["matrix_mappings"])
    assert all(math.isclose(detailed_work[key], steady_graph["execution"]["serviced"][key],
                           rel_tol=1e-12, abs_tol=1e-9) for key in detailed_work)
    mapped["runtime"]["kernels"]["invocations"] = [{
        "match": {"recipe": "rmsnorm"}, "startup_ms": 0.002, "minimum_ms": 0.004,
        "provenance": {"kind": "synthetic_interface_control"},
    }]
    (work / "invocation.json").write_text(json.dumps(mapped) + "\n")
    invocation = run(["graph", work / "invocation.json", "--phase", "prefill", "--tokens", 4],
                     "graph-device-invocation")
    assert invocation["parameters"]["invocation_models"][0]["invocations"] > 0
    assert invocation["duration_ms"] > steady_graph["duration_ms"]
    assert invocation["execution"]["serviced"] == steady_graph["execution"]["serviced"]
    integer = json.loads((root / "examples/quickstart.json").read_text())
    integer["model"]["quant"] = {"name": "dynamic", "weight_format": "int8"}
    integer["machine"]["device"]["compute_peaks"]["int8"] = 2e14
    integer["runtime"]["kernels"]["programs"]["matmul"] = {"kind": "steady_matmul"}
    (work / "integer-output.json").write_text(json.dumps(integer) + "\n")
    outputs = run(["operators", work / "integer-output.json", "--phase", "prefill",
                   "--tokens", 4], "operators-integer-output")
    scaled = [row["operator"] for row in outputs["operators"]
              if row["operator"]["metadata"].get("matrix_epilogue")]
    assert scaled and all(op["vector_ops"] == 4 * op["local_shape"][0] * op["local_shape"][1]
                          for op in scaled)
    flash = run([
        "graph", root / "tests/fixtures/verification/v4flash-mtp-case.json",
        "--phase", "prefill", "--tokens", 4, "--draft-prepare", "--skip-isolated",
    ], "graph-explicit-mtp1")
    attention = [row for row in flash["timeline"] if row["family"] == "attention"]
    assert len(attention) == 44
    assert sum(row["op_id"].startswith("draft_prefill.") for row in attention) == 1
    assert flash["execution"]["live_tasks"] == flash["execution"]["live_programs"] == 0
    balanced = json.loads(
        (root / "examples/cases/balanced-experts-control.json").read_text())
    (work / "balanced-cohort.json").write_text(json.dumps(balanced) + "\n")
    balanced_graph = run(
        ["graph", work / "balanced-cohort.json", "--phase", "prefill", "--tokens", 4],
        "graph-balanced-expert-cohort")
    balanced_serving = run(
        ["predict", work / "balanced-cohort.json"], "predict-balanced-expert-cohort")
    wire = lambda report: {key: value for key, value in report["execution"]["serviced"].items()
                           if key.startswith("link:")}
    assert balanced_graph["replicas"] == [0, 1]
    assert balanced_serving["status"] == "modeled"
    assert wire(balanced_graph) == wire(balanced_serving)
    # Four physical sources send three remote assignments each, 64B per
    # assignment; dispatch and combine each carry 768B across this cohort.
    assert sum(row["communication"]["wire_bytes"] * row["occurrences"]
               for row in balanced_graph["isolated"]["operators"]
               if row["program"] == "PairwiseCollective") == 1536
    assert sum(row["communication"]["wire_bytes"] * row["occurrences"]
               for row in balanced_graph["isolated"]["operators"]
               if row.get("communication")) == sum(wire(balanced_graph).values())
    assert balanced_graph["isolated"]["collective_cohorts"]
    for report in (balanced_graph, balanced_serving):
        assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert balanced_graph["bounds"]["combined_ms"] <= balanced_graph["duration_ms"] + 1e-9
    balanced_screen = run(
        ["operators", work / "balanced-cohort.json", "--phase", "prefill",
         "--tokens", 4, "--scope", "model"], "operators-joint-cohort")
    assert balanced_screen["replicas"] == [0, 1]
    assert math.isclose(balanced_screen["operator_sum_ms"],
                        balanced_graph["isolated"]["operator_sum_ms"], rel_tol=1e-12)
    assert balanced_screen["collective_cohorts"]
    (work / "cohort-study.json").write_text(json.dumps({
        "candidates": [{"id": "balanced", "case": str(work / "balanced-cohort.json")}],
        "probes": [{"id": "p4", "phase": "prefill", "tokens": 4}],
    }) + "\n")
    cohort_study = run(["study", work / "cohort-study.json"], "study-joint-cohort")
    assert cohort_study["status"] == "completed"
    assert math.isclose(cohort_study["evaluations"][0]["duration_ms"],
                        balanced_screen["operator_sum_ms"], rel_tol=1e-12)
    balanced["runtime"].update(dp=1, tp=4, ep=1)
    (work / "balanced-local.json").write_text(json.dumps(balanced) + "\n")
    balanced_ops = run(
        ["operators", work / "balanced-local.json", "--phase", "prefill", "--tokens", 32,
         "--scope", "model"],
        "operators-balanced-experts")
    experts = [row for row in balanced_ops["operators"]
               if row["operator"]["family"] == "grouped_gemm"]
    assert len(experts) == 2
    assert sum(row["operator"]["flops"] * row["occurrences"] for row in experts) == 196608
    assert all(row["program"] == "MappedMatmul"
               and row["operator"]["metadata"]["matrix_groups"] == 8 for row in experts)
    body_ops = run(
        ["operators", work / "balanced-local.json", "--phase", "prefill", "--tokens", 32,
         "--scope", "body"], "operators-balanced-experts-body-default")
    body_experts = [row for row in body_ops["operators"]
                    if row["operator"]["family"] == "grouped_gemm"]
    assert all(row["program"] == "SteadyMatmul" for row in body_experts)
    assert [row["operator"]["local_shape"] for row in body_experts] == [
        row["operator"]["local_shape"] for row in experts]
    assert sum(row["operator"]["flops"] * row["occurrences"] for row in body_experts) == 196608
    balanced["runtime"]["moe_recipe"] = "tp_join_reduce"
    (work / "late-reduce.json").write_text(json.dumps(balanced) + "\n")
    late = run(["graph", work / "late-reduce.json", "--phase", "prefill", "--tokens", 32],
               "graph-tp-expert-output-reduction")
    reductions = [row["operator"] for row in late["isolated"]["operators"]
                  if row["operator"]["recipe"] == "moe_output_reduce"]
    assert reductions and all(op["local_shape"] == [32, 32] for op in reductions)
    assert not any(row["operator"]["recipe"] == "expert_tensor_reduce"
                   for row in late["isolated"]["operators"])
    assert late["execution"]["live_tasks"] == late["execution"]["live_programs"] == 0
    for filename, declaration in (
        ("activation-nominal.json", "swiglu-separate.json"),
        ("gdn-historical-device.json", "gdn-gating.json"),
    ):
        call = run(["call", root / "examples/calls" / filename,
                    root / "examples/calls" / declaration], "call-" + declaration)
        assert call["boundary"] == "device_graph" and call["duration_ms"] > 0
        assert call["execution"]["live_tasks"] == call["execution"]["live_programs"] == 0
        assert call["parameters"]["calibration"] == "none"
    quantized = run([
        "call", root / "examples/calls/quantized-grouped-small.json",
        root / "examples/calls/quantized-grouped-small-call.json",
    ], "call-quantized-grouped")
    assert quantized["operator_occurrences"] == quantized["protocol"]["invocations"] == 8
    assert quantized["single_call"]["expanded_operators"] == 1
    assert quantized["execution"]["work"]["hbm_bytes"] == 63488
    assert quantized["execution"]["work"]["vector_ops"] == 12288
    assert quantized["execution"]["live_tasks"] == quantized["execution"]["live_programs"] == 0
    assert quantized["parameters"]["calibration"] == "none"
    source_grouped = run([
        "call", root / "examples/calls/quantized-grouped-small.json",
        root / "examples/calls/quantized-grouped-source-call.json",
    ], "call-quantized-grouped-source")
    assert source_grouped["operator_occurrences"] == 2
    assert source_grouped["single_call"]["source_boundaries"] == {"kernel": 1}
    assert source_grouped["execution"]["work"]["hbm_bytes"] == 76656
    assert source_grouped["execution"]["serviced"]["fixpipe@r0:s0"] == 16384
    assert math.isclose(source_grouped["execution"]["serviced"]["mte2_aiv@r0:s0"],
                        8288, rel_tol=1e-12)
    assert source_grouped["execution"]["live_tasks"] == 0
    assert source_grouped["execution"]["live_programs"] == 0
    assert not any(source_grouped["execution"]["live_allocated"].values())
    assert source_grouped["parameters"]["calibration"] == "none"
    scan = json.loads((root / "examples/calls/gdn-scan-prefill.json").read_text())
    scan["arguments"].update(batch=2, tokens=65, k_heads=2, v_heads=6)
    (work / "gdn-scan-prefill.json").write_text(json.dumps(scan) + "\n")
    assert run(["call", root / "examples/calls/gdn-historical-device.json",
                work / "gdn-scan-prefill.json", "-o", work / "gdn-scan-result.json"],
               "call-gdn-scan-prefill") is None
    scan_report = json.loads((work / "gdn-scan-result.json").read_text())
    assert scan_report["boundary"] == "device_graph" and scan_report["duration_ms"] > 0
    assert scan_report["operator_occurrences"] == 18
    assert scan_report["single_call"]["families"] == {
        "tensor_copy": 15, "gdn_prepare": 1, "gdn_state": 1, "gdn_output": 1}
    assert scan_report["execution"]["live_tasks"] == scan_report["execution"]["live_programs"] == 0
    assert scan_report["parameters"]["calibration"] == "none"
    ring = run(["graph", root / "examples/ring-collective.json",
                "--phase", "prefill", "--tokens", 4, "--scope", "model"], "graph-ring")
    ring_serving = run(["predict", root / "examples/ring-collective.json"], "predict-ring")
    assert wire(ring) == wire(ring_serving)
    assert any(row["program"] == "RingCollective" for row in ring["timeline"])
    study = run(["study", root / "examples/studies/resource-tradeoff.json"], "study")
    assert study["status"] == "completed" and study["summary"]["modeled"] == 12
    orders = {row["probe"]: [c["candidate"] for c in row["candidates"]]
              for row in study["rankings"]}
    assert orders["prefill32"][0] == "twice-memory-bandwidth"
    assert orders["prefill256"][0] == "twice-matrix-throughput"
    assert all(row["serving"]["status"] == "modeled" for row in study["candidates"])
    communication = run(
        ["study", root / "examples/studies/communication-tradeoff.json"],
        "study-finite-communication")
    assert communication["status"] == "completed"
    assert communication["summary"]["modeled"] == 6
    assert all(row["candidates"][0]["candidate"] == "eight-credits"
               for row in communication["rankings"])
    paged = json.loads((root / "examples/quickstart.json").read_text())
    paged["workload"] = {"requests": [{"id": "pages", "prompt_tokens": 4, "output_tokens": 1}]}
    paged["runtime"]["memory"] = {"derive_workspace": False, "cache_page_groups": [
        {"members": [{"layer": 0, "component": "gqa_kv"},
                     {"layer": 1, "component": "gqa_kv"}],
         "tokens_per_page": 3, "bytes_per_group_page": 3072},
        {"layers": [2, 3], "tokens_per_page": 5, "bytes_per_layer_page": 2560},
    ]}
    (work / "independent-pages.json").write_text(json.dumps(paged) + "\n")
    paged_result = run(["predict", work / "independent-pages.json"], "predict-independent-pages")
    assert paged_result["status"] == "modeled"
    assert paged_result["memory"]["peak_cache_bytes_per_replica_stage"] == [[11264]]
    assert paged_result["memory"]["live_cache_bytes_per_replica_stage"] == [[0]]
    queued = run(["operators", root / "examples/cases/qwen38-a5.json",
                  "--phase", "decode", "--tokens", 4, "--context", 16], "operators-queued-store")
    gating = next(row for row in queued["operators"]
                  if row["operator"]["family"] == "gdn_gating")
    assert gating["duration_ms"] > 0
    assert math.isclose(gating["serviced"]["scalar_store_issue@r0:s0"], 4 * 48, rel_tol=1e-12)
    assert gating["basis"]["declared_parameter_sources"]
    companion = None
    if profiler_wheel is not None:
        # All prediction checks above run without the optional observation
        # provider. Import checks below use an installed, identified wheel.
        from mindie_sim.observations.store import digest_file
        profiler_wheel = Path(profiler_wheel).resolve()
        subprocess.run(["uv", "pip", "install", "--python", str(python), str(profiler_wheel)],
                       check=True, timeout=120)
        companion = {"wheel": str(profiler_wheel), "sha256": digest_file(profiler_wheel)}
        checks.append({"command": "install-companion-profiler-wheel", "exit_code": 0})
    fixtures = root / "tests/fixtures/observations"
    rows = run(["profiler-rows", fixtures / "profiler.csv", "--limit", 2], "profiler-csv")
    assert len(rows) == 2
    variants = run(["profiler-rows", fixtures / "catalogue.sqlite", "--limit", 1], "profiler-sqlite")
    key = {
        "hardware": {"scope": "golden_capture_import_only"},
        "implementation": variants[0]["capture"], "kernel": variants[0]["signature"],
        "dtype": "raw_capture", "shape": variants[0]["context"], "scope": "device_kernel",
    }
    binding = {"variant": variants[0]["id"], "key": key, "run_id": "wheel-import", "split": "heldout"}
    (work / "bindings.jsonl").write_text(json.dumps(binding) + "\n")
    result = run(["bind-profiler", fixtures / "catalogue.sqlite", work / "primitives.sqlite",
                  work / "bindings.jsonl"], "bind-profiler")
    assert result["imported_samples"] > 0
    sample = {
        "key": {**key, "kernel": rows[0]["kernel"], "shape": rows[0]["input_shapes"]},
        "duration_ms": rows[0]["duration_ms"], "run_id": "wheel-csv-import",
        "repeat_index": rows[0]["line"], "source": f"{fixtures / 'profiler.csv'}#line={rows[0]['line']}",
        "split": "heldout",
    }
    (work / "samples.jsonl").write_text(json.dumps(sample) + "\n")
    imported = run(["calibrate", work / "primitives.sqlite", work / "samples.jsonl"], "calibrate")
    assert imported["heldout"]["repeats"] == result["imported_samples"] + 1
    heldout = run(["predict", root / "examples/quickstart.json", "--primitives",
                   work / "primitives.sqlite"], "predict-with-heldout-only")
    assert heldout["metrics"] == prediction["metrics"]
    comparison = run(["validate", work / "prediction.json", fixtures / "service.json",
                      "--boundary", "client_stream"], "validate")
    assert comparison["comparable"] is False
    inventory = run(["observations", work / "index.sqlite", "--scan", fixtures], "observations")
    indexed_csv = run(["observations", work / "index.sqlite", "--find", ".csv"],
                      "observations-find")
    assert {row["path"] for row in indexed_csv} == {
        str(path) for path in fixtures.rglob("*.csv")}
    assert all(row["kind"] == "profiler" for row in indexed_csv)
    assert inventory["profiler"]["files"] == len(indexed_csv)
    capture = next((root / "data/measurements/gpu").glob("*.stdout"))
    fit = run(["fit-rate", capture], "fit-rate")
    assert fit["train_trials"] == 15 and len(fit["heldout"]) == 5
    raw = json.loads(capture.read_text())
    retained = json.loads((root / "data/measurements/gpu/service-curve-validation-20261028.json").read_text())
    model = retained["calibration"]["model"]
    packet = {key: model[key] for key in
              ("resource", "work_unit", "identity", "boundary", "active_units")}
    packet["samples"] = [
        {"work": row["work_bytes"], "duration_ms": row["duration_ms"],
         "run_id": capture.name, "repeat_index": row["trial"],
         "source": f"{capture}#/samples/{index}", "split": row["split"]}
        for index, row in enumerate(raw["samples"])]
    (work / "service-packet.json").write_text(json.dumps(packet) + "\n")
    service = run(["fit-service", work / "service-packet.json"], "fit-service")
    assert len(service["heldout"]) == 5
    # This is an interface check using a real retained event, not an accuracy claim.
    from mindie_sim.observations.captures import iter_observations
    from mindie_sim.observations.store import digest_file
    fixture = fixtures / "capture_adapters"
    protocol = json.loads((root / "data/observations/protocols/grouped-call-protocol.json").read_text())
    protocol["producer"] = {"path": str(fixture / "grouped_producer.py.txt"),
                            "sha256": digest_file(fixture / "grouped_producer.py.txt")}
    protocol["supporting_sources"] = []
    protocol["captures"] = [{"path": str(fixture / "grouped_probe.json"),
                             "sha256": digest_file(fixture / "grouped_probe.json")}]
    observation = next(iter_observations(protocol))
    packet = {"predictions": [{"point_id": "interface-contract-only",
                               "identity": observation["identity"], "duration_ms": 1}],
              "observations": [observation]}
    (work / "accuracy-packet.json").write_text(json.dumps(packet) + "\n")
    accuracy = run(["accuracy", work / "accuracy-packet.json"], "accuracy")
    assert accuracy["coverage"]["paired_heldout_points"] == 0
    assert accuracy["coverage"]["excluded_observations"] == {"split_unassigned": 1}
    (work / "grouped-protocol.json").write_text(json.dumps(protocol) + "\n")
    grouped = run(["compare-calls", root / "examples/calls/quantized-grouped-small.json",
                   work / "grouped-protocol.json"], "compare-calls-quantized-grouped")
    assert grouped["summary"]["samples"] == 240
    assert grouped["summary"]["sample_roles"] == {"unassigned": 240}
    assert grouped["summary"]["unique_predictions"] == 6
    assert grouped["summary"]["condition_statuses"] == {"predicted": 6}
    assert grouped["validation"]["strict_pairs"] == 0
    assert {row["protocol"]["invocations"] for row in grouped["predictions"]} == {1, 8}
    protocol = json.loads((root / "data/observations/protocols/activation-call-protocol.json").read_text())
    protocol["producer"] = {"path": str(fixture / "activation_producer.py.txt"),
                            "sha256": digest_file(fixture / "activation_producer.py.txt")}
    protocol["supporting_sources"] = []
    protocol["captures"] = [
        {"path": str(fixture / name), "sha256": digest_file(fixture / name)}
        for name in ("activation_a.json", "activation_b.json")]
    (work / "activation-protocol.json").write_text(json.dumps(protocol) + "\n")
    aligned = run(["compare-calls", root / "examples/calls/activation-nominal.json",
                   work / "activation-protocol.json", "--include-submission"],
                  "compare-calls-activation")
    assert aligned["summary"]["samples"] == 270
    assert aligned["summary"]["sample_roles"] == {"unassigned": 270}
    assert aligned["validation"]["strict_pairs"] == 0
    assert {row["boundary"] for row in aligned["predictions"]} == {
        "device_graph", "submitted_device_graph"}
    gdn = run(["compare-calls", root / "examples/calls/gdn-historical-device.json",
               fixtures / "gdn-call-protocol.json", "--root", root],
              "compare-calls-gdn-partial", expected_code=2)
    assert gdn["summary"]["condition_statuses"] == {"predicted": 3, "unsupported": 2}
    assert gdn["summary"]["sample_roles"] == {"unassigned": 200}
    assert any(row["canary"].get("ok") is False for row in gdn["conditions"] if row["canary"])
    scoreboard = work / "installed-scoreboard.json"
    subprocess.run(
        [str(python), "-I", "-c",
         "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
         "runpy.run_path(sys.argv.pop(1),run_name='__main__')",
         str(root / "tools"), str(root / "tools/model_scoreboard.py"),
         str(root / "tests/fixtures/verification/scoreboard-input.json"),
         "-o", str(scoreboard)], cwd=work, check=True, capture_output=True, timeout=30)
    board = json.loads(scoreboard.read_text())
    assert len(board["models"]) == 6
    assert len(board["references"]) == 1
    assert any(row["step_comparison"]["status"] == "conditional_range" for row in board["models"])
    checks.append({"command": "installed-scoreboard-retained-observations", "exit_code": 0})
    location = subprocess.check_output(
        [str(python), "-I", "-c", "import mindie_sim; print(mindie_sim.__file__)"],
        cwd=work, text=True, timeout=30).strip()
    assert Path(location).is_relative_to(env)
    report = {
        "status": "passed", "environment": str(env), "import_path": location,
        "wheel": str(wheel), "wheel_sha256": digest_file(wheel),
        "companion_profiler": companion,
        "checks": checks,
        "scope": "Isolated installed wheel outside checkout; fixture imports check identity "
                 "translation and heldout separation, not serving calibration accuracy.",
    }
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel")
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--profiler-wheel",
                        help="local companion wheel for optional profiling import checks")
    args = parser.parse_args()
    print(json.dumps(verify(args.wheel, args.root, args.output,
                            profiler_wheel=args.profiler_wheel), indent=2))
