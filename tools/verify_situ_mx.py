"""Reproduce complete SiTU/MX call controls on the ordinary frontend.

[Role] Compare explicit semantic, traffic-only and source-program selections.
[Invariants] All candidates retain the same declared tensor/scalar arguments.
  No observation duration, model-specific factor or device context is used.
[Timeouts & Locks] Two finite frontend probes and bounded small owner controls.
[Side-Effect Policy] Read one small case; write one optional compact report.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import time

from mindie_sim.prediction.family_situ_mx import (
    Config, PROFILE, SOURCE, make_op, recipe_book, tiling_for,
)
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.native_fusion import _groups, lower_native_fusions
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case


def compact(result):
    execution = result["execution"]
    return {
        "boundary": result["boundary"], "duration_ms": result["duration_ms"],
        "operator_occurrences": result["operator_occurrences"],
        "serviced": {k: v for k, v in execution["serviced"].items() if v},
        **{key: execution[key] for key in (
            "peak_live_tasks", "live_tasks", "live_programs",
            "peak_allocated_per_unit", "live_allocated")},
    }


def selected(case, mode):
    kernels = dict(case.runtime.kernels)
    lowerings = dict(kernels.get("lowerings", {}))
    lowerings.pop("native_fusion", None)
    if mode != "semantic":
        options = {"roles": ["situ_quant"]}
        if mode == "traffic_only":
            options["traffic_profile"] = "situ_mx_quant_arch35_2a9bd1b0"
        else:
            options["program_profiles"] = {"situ_quant": PROFILE}
        lowerings["native_fusion"] = {
            "implementation": "declared_native_fusion", "options": options}
    return replace(case, runtime=replace(
        case.runtime, kernels={**kernels, "lowerings": lowerings}))


def run(case_path):
    started = time.monotonic()
    content = case_path.read_bytes()
    case = parse_case(json.loads(content))
    comparisons = []
    for phase, tokens in (("decode", 1), ("prefill", 8192)):
        baseline = selected(case, "semantic")
        ops = target_operators(baseline, phase=phase, tokens=tokens, context=32768,
                               resource_fusion=False)
        declaration, members = next((d, m) for d, m in _groups(ops)
                                    if d["role"] == "situ_quant"
                                    and d["tensor_view"]["width"] == 768)
        identities = {op.op_id for op in members}
        external = {dep: 0 for op in members for dep in op.deps if dep not in identities}
        modes = {}
        for mode in ("semantic", "traffic_only", "source_program"):
            candidate = selected(case, mode)
            physical = lower_native_fusions(members, candidate.runtime)
            result = graph_costs(candidate, physical, external_inputs=external,
                                 isolated=False)
            modes[mode] = compact(result)
        comparisons.append({
            "phase": phase, "tensor_view": declaration["tensor_view"], "modes": modes,
            "scope": "one_declared_frontend_group_not_whole_model_or_measured_accuracy",
        })
    controls = []
    geometry = Geometry.from_devices(case.machine.dies[:case.runtime.parallel.tp])
    for rows, width in ((1, 65), (1, 257), (130, 768)):
        config = Config(rows, width, "fp8_e4m3", 4.0, 25.0)
        plan = tiling_for(config, geometry)
        result = graph_costs(case, (make_op(config),), recipes=recipe_book(), isolated=False)
        controls.append({
            "rows": rows, "width": width, "active_aiv": plan.used_cores,
            "factor_dim0": plan.factor_dim0, "factor_dim1": plan.factor_dim1,
            **compact(result),
        })
    package = Path(__file__).resolve().parents[1] / "mindie_sim"
    files = {str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(package.rglob("*.py"))}
    return {
        "schema": "mindie-sim.situ-mx-mechanism-controls/1",
        "case": str(case_path), "case_sha256": hashlib.sha256(content).hexdigest(),
        "source_identity": SOURCE, "package_python_sha256": files,
        "comparisons": comparisons, "controls": controls,
        "wall_seconds": time.monotonic() - started,
        "max_rss_native_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "limits": [
            "Source selection is conditional, not capture binary identity.",
            "Body resource time does not identify task scheduling or response latency.",
            "No observed duration or PMU diagnostic changes prediction inputs.",
            "Scalar issue, special instruction latency and DMA transactions are not inferred.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    report = run(args.case)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
        print(json.dumps({"output": str(args.output),
                          "wall_seconds": report["wall_seconds"]}))
    else:
        print(text)


if __name__ == "__main__":
    main()
