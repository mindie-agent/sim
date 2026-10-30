"""Run bounded full-catalog deployment and resource-sensitivity controls.

[Role] Exercise ordinary predict/predict_pd under explicit resource changes.
[Invariants] Full model geometry and chosen placement/precision remain fixed
  across a sweep. Faster unused resources may plateau. Failures stay visible.
[Timeouts & Locks] Serial workers, per-run and campaign wall budgets; no locks.
[Side-Effect Policy] Local child simulations and requested report writes only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "qwen35": "qwen3.5-35b-a3b", "qwen36": "qwen3.6-27b",
    "qwen38": "qwen3.8-27b", "glm53": "glm-5.3",
    "v4flash": "deepseek-v4-flash", "v4pro": "deepseek-v4-pro",
    "v41": "deepseek-v4.1", "kimi-k3": "kimi-k3", "minimax-m3": "minimax-m3",
}
HARDWARE = {"a3": ("Ascend910_9362", 16), "a5": ("Ascend950DT_9582", 8)}
AXES = ("network", "hbm", "aic", "aiv")


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def package_identity():
    import mindie_sim
    root = Path(mindie_sim.__file__).parent
    entries = []
    for path in sorted(root.rglob("*")):
        if path.suffix in {".py", ".json"}:
            entries.append((str(path.relative_to(root)), sha256(path.read_bytes()).hexdigest()))
    return {"sha256": digest(entries), "files": len(entries)}


def base_case(catalog, sku, per_node, *, nodes, tp, dp, ep, pp, quant, prompt=128):
    return {
        "machine": {"backend": "ascend", "sku": sku,
                    "num_devices": tp * dp * pp, "ranks_per_node": per_node},
        "model": {"preset": catalog, "quantization": quant},
        "workload": {"requests": [{"id": "r0", "prompt_tokens": prompt, "output_tokens": 2}]},
        "runtime": {"tp": tp, "dp": dp, "ep": ep, "pp": pp,
                    "max_num_batched_tokens": prompt, "max_num_seqs": 1,
                    "max_model_len": prompt + 2,
                    "kernels": {"programs": {"matmul": {"kind": "steady_matmul"}},
                                "recipes": {key: {"kind": "steady_matmul"} for key in
                                            ("balanced_grouped_gate_up", "balanced_grouped_down")}}},
    }


def plan_scheme(model, hardware, layout, deployment, *, prompt=128):
    from mindie_sim.prediction.capacity import Memory
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.model_catalog import load_model
    catalog, (sku, per_node) = MODELS[model], HARDWARE[hardware]
    attempts = []
    expert_model = bool(load_model(catalog).num_experts)
    # Try complete one/two/four-node placements without increasing SKU memory.
    nodes_list = (1,) if layout == "single" else (2, 4)
    for quant in ("w8a8", "w4a8", "bf16"):
        for nodes in nodes_list:
            ranks = nodes * per_node
            candidates = [(min(8, ranks), ranks // min(8, ranks), ranks, 1),
                          (min(8, ranks), 1, min(8, ranks), ranks // min(8, ranks))]
            if not expert_model:
                candidates = [(tp, dp, 1, pp) for tp, dp, ep, pp in reversed(candidates)]
            for tp, dp, ep, pp in dict.fromkeys(candidates):
                raw = base_case(catalog, sku, per_node, nodes=nodes, tp=tp, dp=dp,
                                ep=ep, pp=pp, quant=quant, prompt=prompt)
                try:
                    parsed = parse_case(raw)
                    ledger = Memory(parsed)
                    fits = ledger.resize(0, "r0", prompt + 2)
                    attempt = {"quantization": quant, "nodes": nodes,
                               "parallel": parsed.runtime.parallel.to_dict(),
                               "weights_bytes_per_rank_stage": list(ledger.weights),
                               "available_cache_bytes": ledger.capacity,
                               "status": "fits" if fits else "infeasible"}
                except ValueError as exc:
                    attempts.append({"quantization": quant, "nodes": nodes,
                                     "parallel": [tp, dp, ep, pp], "status": "unsupported",
                                     "reason": str(exc)})
                    continue
                attempts.append(attempt)
                if not fits:
                    continue
                if deployment == "pd":
                    raw["pools"] = {name: {"machine": deepcopy(raw["machine"]),
                                           "runtime": deepcopy(raw["runtime"])}
                                    for name in ("prefill", "decode")}
                    raw["connector"] = {"bytes_per_s": 25e9, "latency_us": 5}
                return {"status": "planned", "case": raw, "attempts": attempts,
                        "selected": attempt, "full_layers": parsed.model.layers,
                        "routing": "declared_probability_moments"}
    return {"status": "infeasible", "attempts": attempts,
            "reason": "no_supported_capacity_fit_in_declared_single_or_2_to_4_node_candidates"}


def scale_machine(machine, axis, factor):
    from mindie_sim.schema.hardware import Hardware
    effective = Hardware.from_mapping(machine)
    result = deepcopy(machine)
    rates = dict(effective.dies[0].rates)
    overrides = result.setdefault("device", {})
    if axis == "network":
        links = [link for link in effective.links if link.channel == "d2d_inter_node"]
        if links:
            # The convenient field is unambiguous B/s. No explicit channel patch
            # is present in these campaign cases to take precedence over it.
            result["inter_node_bw_bytes_per_s"] = links[0].capacity_bytes_per_ms * 1000 * factor
    else:
        roles = ({"hbm"} if axis == "hbm" else {"mte1", "mte2_aic", "fixpipe"}
                 if axis == "aic" else {"vector", "mte2_aiv", "mte3"} |
                 {role for role in rates if role.startswith("reg_")})
        overrides.setdefault("rates", {}).update({role: rates[role] * factor
                                                    for role in roles if role in rates})
        if axis == "aic":
            overrides["compute_peaks"] = {key: value * factor
                                           for key, value in effective.dies[0].compute_peaks}
    return result


def scaled_case(raw, axis, factor):
    result = deepcopy(raw)
    if "pools" in result:
        for pool in result["pools"].values():
            pool["machine"] = scale_machine(pool["machine"], axis, factor)
        if axis == "network":
            result["connector"]["bytes_per_s"] *= factor
    else:
        result["machine"] = scale_machine(result["machine"], axis, factor)
    return result


def effective_resources(raw):
    from mindie_sim.schema.hardware import Hardware
    pools = raw.get("pools", {"colocated": raw})
    result = {}
    for name, pool in pools.items():
        hardware = Hardware.from_mapping(pool["machine"])
        result[name] = {"device": hardware.dies[0].to_dict(),
                        "links": [link.to_dict() for link in hardware.links],
                        "ranks": len(hardware.dies), "ranks_per_node": hardware.ranks_per_node}
    if "connector" in raw:
        result["connector"] = deepcopy(raw["connector"])
    return result


def run_case(raw):
    from mindie_sim import predict
    started = time.monotonic()
    package_before = package_identity()
    report = predict(raw, event_limit=0, trace_limit=2)
    from tools.model_convergence import worker_usage
    execution = report["execution"]
    return {"status": report["status"], "elapsed_s": time.monotonic() - started,
            "case_sha256": digest(raw), "resources": effective_resources(raw),
            "semantic_input_sha256": digest({
                "model": raw["model"], "workload": raw["workload"], "runtime": raw.get("runtime"),
                "pool_runtime": {name: pool.get("runtime") for name, pool in raw.get("pools", {}).items()},
                "connector_latency_us": raw.get("connector", {}).get("latency_us", 0)}),
            "package_before": package_before, "package_after": package_identity(),
            "worker_memory": worker_usage(),
            "metrics": report["metrics"], "model_coverage": report.get("model_coverage"),
            "unmodeled_features": report.get("unmodeled_features", []),
            "requests": [{key: row.get(key) for key in ("status", "reason", "generated_tokens")}
                         for row in report["requests"]],
            "memory": report["memory"], "traces": report["traces"],
            "execution": {key: execution.get(key) for key in
                          ("live_tasks", "live_programs", "live_allocated", "serviced", "work",
                           "submission", "iterations", "handoffs", "handoff_bytes", "transfer_layout")}}


def compare_sweep(rows):
    if any(row.get("status") != "modeled" for row in rows):
        return {"status": "incomplete", "reason": "all_three_successful_predictions_required"}
    identities = {row.get("semantic_input_sha256") for row in rows}
    if len(identities) > 1:
        return {"status": "invalid", "reason": "semantic_input_changed_between_resource_factors"}
    if any(row.get("package_before") != row.get("package_after") for row in rows):
        return {"status": "invalid", "reason": "runtime_changed_during_execution"}
    if len({row.get("package_before", {}).get("sha256") for row in rows}) > 1:
        return {"status": "invalid", "reason": "runtime_changed_between_resource_factors"}
    if any(row["execution"].get("live_tasks") or row["execution"].get("live_programs")
           or any(request.get("status") != "served" for request in row["requests"])
           for row in rows):
        return {"status": "invalid", "reason": "unfinished_device_work_or_request"}
    if len({row["execution"].get("handoff_bytes") for row in rows}) > 1:
        return {"status": "invalid", "reason": "pd_handoff_volume_changed"}
    if any(row["execution"].get("work") != rows[1]["execution"].get("work") for row in rows):
        return {"status": "invalid", "reason": "semantic_work_changed"}
    values = [row["metrics"]["makespan_ms"] for row in rows]
    tolerance = max(values) * 1e-9 + 1e-9
    monotone = values[0] + tolerance >= values[1] and values[1] + tolerance >= values[2]
    strict = values[0] > values[1] + tolerance and values[1] > values[2] + tolerance
    return {"status": "strict_response" if monotone and strict else "plateau" if monotone else "regression",
            "factors": [.5, 1, 2], "makespan_ms": values,
            "relative_span": (values[0] - values[2]) / values[1] if values[1] else 0,
            "tolerance_ms": tolerance}


def worker(raw, timeout, source_root):
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--source-root", str(source_root)]
    started = time.monotonic()
    try:
        process = subprocess.run(command, input=json.dumps(raw), text=True, capture_output=True,
                                 timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "budget_s": timeout, "elapsed_s": time.monotonic() - started}
    if process.returncode:
        return {"status": "failed", "returncode": process.returncode,
                "stderr": process.stderr[-8000:], "elapsed_s": time.monotonic() - started}
    return json.loads(process.stdout)


def campaign(args):
    started = time.monotonic()
    report = {"schema": "mindie-sim.deployment-generalization/1", "package": package_identity(),
              "boundary": "public serving device_token_commit; complete catalog, analytical predictions",
              "producer_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
              "configuration": {"prompt_tokens": args.prompt_tokens, "output_tokens": 2,
                                "per_run_s": args.run_timeout, "total_s": args.budget,
                                "axes": args.axes, "factors": [.5, 1, 2]}, "schemes": []}
    for model in args.models:
        for hardware in args.hardware:
            for layout in args.layouts:
                for deployment in args.deployments:
                    entry = {"id": f"{model}-{hardware}-{layout}-{deployment}", "model": model,
                             "hardware": hardware, "layout": layout, "deployment": deployment}
                    entry.update(plan_scheme(model, hardware, layout, deployment, prompt=args.prompt_tokens))
                    report["schemes"].append(entry)
    # Baselines get priority over completing a sweep for only the first model.
    def invoke(raw):
        remaining = args.budget - (time.monotonic() - started)
        return (worker(raw, min(args.run_timeout, remaining), args.source_root) if remaining > .1
                else {"status": "not_run", "reason": "campaign_wall_budget_exhausted"})
    for entry in report["schemes"]:
        if entry["status"] == "planned":
            entry["baseline"] = invoke(entry["case"])
    for entry in report["schemes"]:
        if entry.get("baseline", {}).get("status") != "modeled":
            continue
        entry["sweeps"] = {}
        for axis in args.axes:
            slow = invoke(scaled_case(entry["case"], axis, .5))
            fast = invoke(scaled_case(entry["case"], axis, 2))
            entry["sweeps"][axis] = {"half": slow, "double": fast,
                                      **compare_sweep([slow, entry["baseline"], fast])}
    report["elapsed_s"] = time.monotonic() - started
    report["package_after"] = package_identity()
    report["source_stable"] = report["package"] == report["package_after"]
    report["summary"] = {"schemes": len(report["schemes"]),
                         "modeled_baselines": sum(s.get("baseline", {}).get("status") == "modeled"
                                                  for s in report["schemes"])}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--hardware", nargs="+", choices=HARDWARE, default=list(HARDWARE))
    parser.add_argument("--layouts", nargs="+", choices=("single", "multi"), default=["single", "multi"])
    parser.add_argument("--deployments", nargs="+", choices=("colocated", "pd"), default=["colocated", "pd"])
    parser.add_argument("--axes", nargs="*", choices=AXES, default=list(AXES))
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--run-timeout", type=float, default=30)
    parser.add_argument("--budget", type=float, default=600)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root))
    if args.worker:
        print(json.dumps(run_case(json.load(sys.stdin))))
        return 0
    result = campaign(args)
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    regressions = any(sweep["status"] in {"regression", "invalid"} for row in result["schemes"]
                      for sweep in row.get("sweeps", {}).values())
    incomplete = (not result["source_stable"] or any(
        row.get("baseline", {}).get("status") != "modeled" or
        any(sweep["status"] == "incomplete" for sweep in row.get("sweeps", {}).values())
        for row in result["schemes"]))
    return 2 if regressions or incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
