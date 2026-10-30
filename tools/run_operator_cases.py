"""Run source-derived extension cases without hardware or measured parameters.

[Role] Exercise the public isolated operator path for a named local extension.
[Invariants] Synthetic rates are structural controls, never accuracy evidence.
[Timeouts & Locks] Serial bounded cases; no device contexts or subprocesses.
[Side-Effect Policy] Reads case JSON and writes a requested diagnostic report.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import json
import math
from pathlib import Path

from mindie_sim.prediction.verification import verify_operator
from mindie_sim.prediction.instructions import REGISTER_RESOURCES


def control_case(*, split_copy=False, units=32, service_roles=None):
    rates = {"memory": 1e12, "vector": 1e12, "mte1": 1e12, "fixpipe": 1e12}
    rates.update({role: 1e12 for role in REGISTER_RESOURCES})
    if split_copy:
        rates.update(read=1e12, write=1e12)
    case = {
        "machine": {
            "name": "synthetic-contract-control", "num_devices": 1,
            "device": {"memory_bytes": 64 * 1024**3, "compute_units": units,
                       "compute_peaks": {"bf16": 1e14, "fp16": 1e14, "fp32": 2.5e13,
                                         "fp8": 2e14, "int8": 2e14, "tf32": 5e13},
                       "rates": rates, "resource_units": {
                           "vector": 2 * units, **{role: 2 * units for role in REGISTER_RESOURCES}},
                       "service_modes": {
                           "ub_copy": {"resource": "vector", "rate": 1e12},
                           "reg_issue_slots": {"resource": "vector", "rate": 1e12}}},
        },
        "model": {"name": "unlabeled-geometry", "layers": 1, "hidden_size": 8,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }
    device = case["machine"]["device"]
    for role, resource in (service_roles or {}).items():
        device["rates"].setdefault(resource, 1e12)
        device["resource_units"].setdefault(resource, units)
        device["service_modes"][role] = {"resource": resource, "rate": 1e12}
    return case


def run(module, cases):
    rows = []
    for case in cases:
        op = module.make_op(case["params"])
        selected = dict(op.metadata).get("dispatch", {})
        mismatches = {key: {"expected": value, "selected": selected.get(key)}
                      for key, value in case.get("expected_dispatch", {}).items()
                      if key not in selected or selected[key] != value}
        for split in (False, True):
            hardware = control_case(split_copy=split, units=case.get("units", 32),
                                    service_roles=getattr(module, "SERVICE_ROLES", {}))
            kernels = getattr(module, "kernels", None)
            report = verify_operator(hardware, op, module.build, kernels=kernels)
            renamed = replace(op, op_id="identity-control", layer=97, requests=("other",))
            control = verify_operator(hardware, renamed, module.build, kernels=kernels,
                                      input_ready_ms=1)
            same = math.isclose(report["result"]["duration_ms"], control["result"]["duration_ms"],
                                rel_tol=1e-9, abs_tol=1e-12)
            original, shifted = report["result"]["serviced"], control["result"]["serviced"]
            same = same and all(math.isclose(original.get(key, 0), shifted.get(key, 0),
                                            rel_tol=1e-9, abs_tol=1e-12)
                                for key in original.keys() | shifted.keys())
            if not same or mismatches or control["status"] != "passed":
                report["status"] = "failed"
            rows.append({"case": case["id"], "split": case["split"],
                         "copy_resources": split, "identity_invariant": same,
                         "delayed_input_obeyed": control["input_dependency_obeyed"],
                         "dispatch": selected, "dispatch_mismatches": mismatches, **report})
    return {"scope": "structural_contract_only_no_accuracy_claim",
            "status": "passed" if all(row["status"] == "passed" for row in rows) else "failed",
            "cases": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("module", help="Reviewed local Python module, not user runtime configuration")
    parser.add_argument("cases", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(importlib.import_module(args.module), json.loads(args.cases.read_text())["cases"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(result["cases"]),
                      "output": str(args.output)}))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
