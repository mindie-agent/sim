"""Predict CLI integration with synthetic operator profiles, never an NPU run."""
import json
from pathlib import Path
import subprocess
import sys


def test_predict_opt_in_selects_plans_and_preserves_explicit_fallback(tmp_path):
    root = Path(__file__).resolve().parents[1]
    mixed = json.loads((root / "examples/operator-plans/mixed.json").read_text(encoding="utf-8"))
    # Reuse the one declared profile source instead of duplicating rates here.
    config = dict(mixed["estimators"], strict=False)
    config_path = tmp_path / "estimators.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    case = {
        "name": "synthetic-small-dense-operator-integration",
        "machine": mixed["machine"],
        "model": {
            "name": "synthetic-dense", "architecture": "gqa", "layers": 1,
            "hidden_size": 16, "num_heads": 2, "num_kv_heads": 1,
            "head_dim": 8, "intermediate_size": 32, "vocab_size": 64,
            "activation_dtype": "bf16", "weight_dtype": "bf16",
            "kv_dtype": "bf16", "compute_dtype": "bf16",
        },
        "workload": {"requests": [
            {"id": "one", "prompt_tokens": 16, "output_tokens": 3},
        ]},
        "runtime": {
            "tp": 1, "ep": 1, "dp": 1, "pp": 1,
            "max_num_batched_tokens": 32, "max_num_seqs": 1,
            "max_model_len": 64, "kv_block_size": 16,
            "graph_mode": "eager", "gpu_memory_utilization": .9,
        },
    }
    case_path = tmp_path / "case.json"
    result_path = tmp_path / "result.json"
    case_path.write_text(json.dumps(case), encoding="utf-8")
    run = subprocess.run([
        sys.executable, "-m", "mindie_sim", "predict", "--input", str(case_path),
        "--operator-config", str(config_path), "--output", str(result_path),
    ], cwd=root, text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stdout[-2000:] + run.stderr[-2000:]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "modeled"
    assert result["command"] == "predict"
    assert len(result["requests"]) == 1
    assert len(result["traces"]) == 3  # Prefill first token + two Decode tokens.
    operators = [op for step in result["traces"] for op in step["ops"]]
    selected = [op for op in operators
                if op.get("operator_estimation", {}).get("status") == "analytic_plan"]
    assert {op["execution_plan"]["backend"] for op in selected} >= {
        "vector_instruction", "gemm_finite_pipeline",
    }
    assert any(op.get("operator_estimation", {}).get("status") == "fallback"
               for op in operators)
    assert all(op["operator_estimation"]["not_npu_measured"] for op in selected)
    # Subsequent iterations depend on the prior iteration's completion, while
    # every execution plan is visible in the existing prediction trace.
    for previous, following in zip(result["traces"], result["traces"][1:]):
        assert previous["end_ms"] <= following["start_ms"]
    assert result["traces"][-1]["end_ms"] > 0
