"""Compare explicit transport schedules on an ordinary model's EP cohort.

[Role] Retain a compact, reproducible network-only candidate comparison.
[Invariants] All candidates use identical payloads, topology, rates and link
  latency; local dispatch/quantization/combine work is excluded, not erased
  from the model. Predicted transport is not native-kernel measured accuracy.
[Timeouts & Locks] Finite target graph; only one dispatch/return pair executes.
[Side-Effect Policy] Read one case and write the requested report; no network.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

from mindie_sim import graph_costs, target_graphs


def compare(raw):
    graphs = target_graphs(raw, phase="decode", tokens=1, context=32768, scope="body")
    pair = {}
    for replica, ops in graphs.items():
        dispatch = next(op for op in ops if op.recipe == "expert_dispatch")
        combine = next(op for op in ops if op.recipe == "expert_combine"
                       and op.layer == dispatch.layer)
        pair[replica] = (replace(dispatch, deps=()),
                         replace(combine, deps=(dispatch.op_id,)))
    rows = []
    declarations = [
        ("aggregate_serial_estimate", {"kind": "envelope"}),
        ("pairwise", {"kind": "pairwise_collective"}),
        *((f"direct_window_{window}", {"kind": "direct_peer_collective",
                                      "max_inflight_per_rank": window})
          for window in (1, 8, 32)),
    ]
    for name, declaration in declarations:
        case = deepcopy(raw)
        case.setdefault("runtime", {}).setdefault("kernels", {}).setdefault("recipes", {}).update({
            "expert_dispatch": declaration, "expert_combine": declaration})
        started = time.monotonic()
        report = graph_costs(case, pair, include_submission=False)
        rows.append({
            "candidate": name, "recipe": declaration,
            "pair_graph_ms": report["duration_ms"],
            "execution": {key: report["execution"][key] for key in
                          ("serviced", "live_tasks", "live_programs", "peak_live_tasks")},
            "operators": [{key: op[key] for key in
                           ("replica", "program", "duration_ms", "communication")}
                          for op in report["isolated"]["operators"]],
            "elapsed_seconds": time.monotonic() - started,
        })
    expected_wire = rows[0]["execution"]["serviced"]
    import math
    for row in rows:
        actual = row["execution"]["serviced"]
        if any(not math.isclose(expected_wire.get(k, 0), actual.get(k, 0), rel_tol=1e-12,
                                abs_tol=1e-9) for k in expected_wire.keys() | actual.keys()):
            raise ValueError("candidate_changed_wire_work")
        if row["execution"]["live_tasks"] or row["execution"]["live_programs"]:
            raise ValueError("candidate_left_live_program")
    return {
        "schema": "mindie-sim.direct-peer-transport-control/1",
        "geometry": {"phase": "decode", "q": 1, "batch_per_replica": 1,
                     "context": 32768, "scope": "body"},
        "participants": list(pair),
        "operator_contracts": {str(replica): [op.to_dict() for op in ops]
                               for replica, ops in pair.items()},
        "candidates": rows,
        "invariant_checks": {"wire_conserved": True, "no_live_tasks_or_programs": True},
        "limitations": [
            "One ordinary dispatch/return pair; not complete native MC2 kernel or body latency.",
            "Window sizes are explicit transport candidates, not observed source tiling.",
            "Uniform peer assignment is a declared scenario, not measured top-k IDs.",
            "Count/status exchange, data-as-flag expansion, local materialization, and fused quant work are outside this transport-only comparison.",
            "A coalesced peer slot is conservatively held through remote arrival; native per-token MTE3 queue progress can differ.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    payload = args.case.read_bytes()
    report = compare(json.loads(payload))
    report["case"] = {"path": str(args.case), "sha256": hashlib.sha256(payload).hexdigest()}
    root = Path(__file__).resolve().parents[1]
    report["runtime_source_sha256"] = {
        name: hashlib.sha256((root / "mindie_sim/prediction" / name).read_bytes()).hexdigest()
        for name in ("direct_peer.py", "collectives.py", "topology.py", "execution.py",
                     "program.py", "operator_analysis.py")}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["candidates"]:
        print(row["candidate"], row["pair_graph_ms"])


if __name__ == "__main__":
    main()
