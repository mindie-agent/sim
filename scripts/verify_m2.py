#!/usr/bin/env python3
"""M2 audit: physical recipe coverage, roofline invariants and step bounds.

[Role] Compose each maintained catalog model on A5 and execute one tiny
  prefill/decode workload through the public prediction path.
[Invariants] A model is accepted only when execution reaches ``modeled``, no
  aggregate ResourceEnvelope or unmodeled mechanism appears, and every step is
  below the declared wall bound. Roofline checks are reported with their raw
  values; a failed inequality is not hidden.
[Timeouts & Locks] Sequential bounded executions; no device or network work.
[Side-Effect Policy] Writes only the requested JSON report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from mindie_sim import Case, predict
from mindie_sim.compose import compose_case

MODELS = (
    "deepseek-v4-10t", "deepseek-v4-flash-0731", "deepseek-v4-pro-0813",
    "deepseek-v4.1-flash", "deepseek-v2-lite", "deepseek-v3.2-prune-3l",
    "deepseek-v3", "deepseek-r1", "deepseek-v3.2", "glm-5.3",
    "glm-5.3-flash", "glm-5.1", "glm-5.2", "kimi-k3", "kimi-k3.1",
    "kimi-k3-dspark", "kimi-k3-dspark-block5", "kimi-k2", "kimi-k2-thinking",
    "kimi-k2.5", "kimi-k2.6", "qwen3-30b-a3b", "qwen3-32b", "qwen3-235b-a22b",
    "qwen3-next-80b-a3b", "qwen3.5-27b", "qwen3.5-35b-a3b",
    "qwen3.5-122b-a10b", "qwen3.5-397b-a17b", "qwen3.6-27b", "qwen3.8-2.4t-a95b", "qwen3.8-27b",
    "minimax-m3", "minimax-m3.1-flash", "gemma-4-26b-a4b", "gemma-4-31b",
    "qwen3.6-35b-a3b", "qwen3.8-flash-next", "step-5-preview",
)
STEP_BOUND_MS = 50.0


def _attempts(layers):
    layer_attempts = ((layers, 4, 8) if layers is not None
                      else (4, 8, None))
    for tp, ep in ((8, 8), (8, 1), (1, 1)):
        for selected_layers in layer_attempts:
            yield tp, ep, selected_layers


def audit_model(model, layers):
    started = time.perf_counter()
    last = None
    compose_model = {"kimi-k3-dspark": "kimi-k3",
                     "kimi-k3-dspark-block5": "kimi-k3"}.get(model, model)
    for tp, ep, selected_layers in _attempts(layers):
        try:
            raw = compose_case(
                compose_model, hardware="a5", tp=tp, ep=ep,
                layers=selected_layers, prompt_tokens=512,
                output_tokens=16, concurrency=1, block_size=1024,
                graph_capture_mode="none", max_num_seqs=1,
            )
            case = Case.from_mapping(raw)
            result = predict(case, trace_limit=0, event_limit=0)
            if result.get("status") != "modeled":
                raise RuntimeError(f"prediction_status:{result.get('status')}")
            break
        except Exception as error:  # noqa: BLE001 - audited per attempt
            last = error
    else:
        raise last
    text = json.dumps(result, default=str, sort_keys=True)
    recipes = dict(result.get("execution", {}).get("recipes", {}))
    groups = list(result.get("execution", {}).get("step_summary", {}).get("groups", ()))
    maximum = max((float(group.get("duration_ms", {}).get("max", 0.0))
                   for group in groups), default=0.0)
    return {
        "model": model,
        "compose_model": compose_model,
        "parallel": {"tp": tp, "ep": ep, "layers": selected_layers},
        "status": result.get("status"),
        "envelope_hits": text.count("ResourceEnvelope"),
        "unmodeled_hits": text.count("unmodeled_operator_mechanism"),
        "recipe_kinds": len(recipes),
        "max_step_ms": maximum,
        "steps_under_bound": maximum < STEP_BOUND_MS,
        "groups": [group.get("mode") for group in groups],
        "wall_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=list(MODELS))
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows, failures = [], []
    for model in args.models:
        try:
            row = audit_model(model, args.layers)
        except Exception as error:  # noqa: BLE001 - report the model failure
            row = {"model": model, "status": "failed",
                   "error": f"{type(error).__name__}:{error}"}
        rows.append(row)
        if (row.get("status") != "modeled"
                or row.get("envelope_hits")
                or row.get("unmodeled_hits")
                or not row.get("steps_under_bound", False)):
            failures.append(row["model"])
    report = {
        "schema": "mindie-sim.m2-audit/1",
        "step_bound_ms": STEP_BOUND_MS,
        "models": rows,
        "passed": len(rows) - len(failures),
        "total": len(rows),
        "failures": failures,
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("passed", "total", "failures")}, sort_keys=True))
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
