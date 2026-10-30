"""Run model graphs with physical dimensions intact, without accuracy claims.

[Role] Check the public pure-model path across supplied case files and phases.
[Invariants] Costs, precedence bounds and shared-resource service stay distinct.
  A completed structural case is not measured model timing acceptance.
[Timeouts & Locks] Sequential bounded input graphs; no device work or network.
[Side-Effect Policy] Writes only the requested compact JSON report.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

from mindie_sim import graph_costs, target_graphs
from mindie_sim.__main__ import read_json


def check(path, phase, tokens, context):
    case = read_json(path)
    started = time.perf_counter()
    graphs = target_graphs(case, phase=phase, tokens=tokens, context=context, scope="model")
    result = graph_costs(case, graphs, input_ready_ms=3)
    bounds = {**result["bounds"], "precedence": {
        key: value for key, value in result["bounds"]["precedence"].items() if key != "operators"}}
    state = result["execution"]
    # A bound violation diagnoses inconsistent graph/isolated execution; it is
    # not repaired by clipping the bound to the output duration.
    valid_bound = bounds["combined_ms"] <= result["duration_ms"] + 1e-8
    return {
        "case": str(path), "sha256": sha256(Path(path).read_bytes()).hexdigest(),
        "phase": phase, "tokens_per_sequence": tokens, "context": context,
        "replicas": result["replicas"], "replay_scope": result["replay_scope"],
        "operator_occurrences": result["operator_occurrences"],
        "duration_ms": result["duration_ms"],
        "isolated_sum_ms": result["isolated"]["operator_sum_ms"],
        "bounds": bounds, "live_tasks": state["live_tasks"],
        "live_programs": state["live_programs"], "recipes": state["recipes"],
        "passed": valid_bound and state["live_tasks"] == state["live_programs"] == 0,
        "wall_seconds": time.perf_counter() - started,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", nargs="+")
    parser.add_argument("--prefill-tokens", type=int, default=32)
    parser.add_argument("--decode-context", type=int, default=4096)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    rows = []
    for path in args.cases:
        for phase, tokens, context in (("prefill", args.prefill_tokens, 0),
                                        ("decode", 1, args.decode_context)):
            row = check(path, phase, tokens, context)
            rows.append(row)
            print(json.dumps({key: row[key] for key in
                              ("case", "phase", "duration_ms", "passed", "wall_seconds")}), flush=True)
    Path(args.output).write_text(json.dumps({
        "scope": "pure_model_structural_execution_no_accuracy_claim",
        "cases": rows, "passed": sum(row["passed"] for row in rows), "total": len(rows),
    }, indent=2, allow_nan=False) + "\n")
    return 0 if all(row["passed"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
