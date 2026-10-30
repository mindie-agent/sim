"""Compare the frozen short-context parameter packet against long-context Body.

[Role] Reproduce the ten retained comparisons through ordinary predict(body).
[Invariants] Observation totals are read only after prediction. The pinned
  Profiler report identifies the measured boundary; no response is refit here.
[Timeouts & Locks] Local Git reads have 15-second limits; predictions run serially.
[Side-Effect Policy] Reads captures' retained reports and writes one local report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from mindie_sim import predict

REVISION = "58d75f6"


def read_revision(repo, path):
    value = subprocess.run(
        ["git", "show", f"{REVISION}:{path}"], cwd=repo, check=True,
        capture_output=True, text=True, timeout=15)
    return json.loads(value.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiler-root", type=Path, required=True)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    parameter_file = directory / "parameters.json"
    parameters = json.loads(parameter_file.read_text())["service_regimes"]
    report = read_revision(args.profiler_root, "examples/structure/results/decode.json")
    cases = []
    for observed in report["cases"]:
        name = observed["case_id"]
        if not name.startswith(("a3-qwen35-tp4-longctx", "a3-qwen36-tp4-longctx")):
            continue
        raw = read_revision(args.profiler_root,
                            "examples/structure/" + observed["model_case"]["path"])
        if "qwen35" in name:
            raw["model"]["quant"]["category_overrides"]["shared_expert"] = {
                "weight_format": "bf16", "activation_format": "bf16"}
            raw["runtime"]["moe_recipe"] = "tp_join_reduce"
        kernels = raw["runtime"].setdefault("kernels", {})
        kernels.setdefault("recipes", {})["flash_attention_decode"] = {
            "kind": "staged_gqa_decode", "target_cores": 20, "query_tile": 64,
            "key_tile": 128, "slots": 2, "kv_mapping": "fai_host"}
        kernels["service_regimes"] = kernels.get("service_regimes", []) + parameters
        if "qwen35" in name:
            kernels.setdefault("lowerings", {}).update({
                "group_list_prefix": "vllm_ascend_group_list_prefix",
                "native_fusion": {"implementation": "declared_native_fusion",
                                  "options": {"roles": ["grouped_gate_up_activation_quant"]}}})
        geometry = observed["geometry"]
        start = time.monotonic()
        result = predict(raw, scope="body", phase="decode",
                         tokens=geometry["tokens_per_sequence"],
                         context=geometry["context_tokens"],
                         sequences=geometry["sequences_per_replica"])
        measured = observed["comparison"]["observed_ms"]["mean"]
        estimated = result["body_forward_ms"]
        cases.append({"id": name, "geometry": geometry, "source": observed["source"],
                      "model_case": observed["model_case"], "observed_ms": measured,
                      "predicted_ms": estimated, "error": estimated / measured - 1,
                      "wall_seconds": time.monotonic() - start,
                      "input_sha256": hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(),
                      "execution": result["execution"]})
        print(name, estimated, measured, flush=True)
    runtime = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                             capture_output=True, text=True, timeout=15).stdout.strip()
    output = {"runtime_commit": runtime, "observation_report_revision": "profiler/" + REVISION,
              "parameter_sha256": hashlib.sha256(parameter_file.read_bytes()).hexdigest(),
              "boundary": "target_body_device_graph; conditional source alignment",
              "passed": sum(abs(row["error"]) <= .15 for row in cases),
              "cases": cases}
    (directory / "body-validation.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
