"""Refresh retained model-scoreboard predictions without importing observations.

[Role] Run bounded existing workers and retain their reports and family mapping.
[Invariants] Measured durations never enter worker inputs or recipe selection.
  Semantic family sums are diagnostic, not a decomposition of graph latency.
  Unclassified work and failed workers remain explicit.
[Timeouts & Locks] Sequential children; per-probe and total prediction budgets.
[Side-Effect Policy] Read declared cases; write requested manifest and reports.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import time

from model_scoreboard import read_json, run_prediction, source


MATRIX_FAMILIES = {
    "gemm", "qkv_projection", "output_projection", "gate_up_projection",
    "down_projection", "logits", "mla", "projection",
}
COLLECTIVE_FAMILIES = {
    "allreduce", "allgather", "alltoall", "alltoallv", "reducescatter",
    "reduce_scatter", "pp_p2p",
}
GDN_RECIPES = {
    "recurrent_scan0", "delta_gates", "short_convolution", "qk_l2norm",
    "linear_output_norm_gate", "gate_beta_contiguous", "gate_alpha_contiguous",
    "convolution_input_contiguous",
}
VECTOR_FAMILIES = {
    "quantize", "elementwise", "rmsnorm", "add_rmsnorm", "add_rmsnorm_quant",
    "swiglu", "routing", "topk", "sampling", "acceptance", "embedding", "rotary",
    "tensor_copy", "tensor_select", "cache_store", "mhc", "router", "zero_fill", "cast",
    "histogram", "dequant_swiglu_quant", "attn_res_prefix",
}


def classify(family, recipe, overrides):
    """Recipe overrides declare mixed native-call boundaries explicitly."""
    if recipe in overrides:
        return overrides[recipe]
    if family in COLLECTIVE_FAMILIES:
        return "collective"
    if family == "grouped_gemm" or recipe == "grouped_swiglu":
        return "grouped_gemm"
    if recipe in GDN_RECIPES or family in {"gdn_gating", "recurrent_vector"}:
        return "gdn"
    if family in MATRIX_FAMILIES:
        return "matmul"
    if family == "attention":
        return "attention"
    if family == "indexer" or recipe.startswith("index_"):
        return "indexer"
    if family in VECTOR_FAMILIES:
        return "vector"
    return None


def family_mapping(report, overrides=None):
    """Account for every isolated family exactly once, including unassigned work."""
    groups, assignments, unassigned = {}, [], []
    for row in report["costs"]["families"]:
        category = classify(row["family"], row["recipe"], overrides or {})
        item = {**row, "semantic_family": category}
        assignments.append(item)
        if category is None:
            unassigned.append(item)
        else:
            groups.setdefault(category, []).append(row["recipe"])
    return {
        "groups": {key: sorted(set(value)) for key, value in groups.items()},
        "assignments": assignments,
        "classified_ms": math.fsum(
            row["isolated_ms"] for row in assignments if row["semantic_family"] is not None),
        "unassigned_ms": math.fsum(row["isolated_ms"] for row in unassigned),
        "unassigned": unassigned,
        "isolated_sum_ms": report["costs"]["operator_sum_ms"],
        "interpretation": "Semantic inventory only; never added to graph or measured spans.",
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def refresh(manifest, output_manifest, prediction_dir, *, revision, timeout, total_timeout):
    manifest, output_manifest = Path(manifest).resolve(), Path(output_manifest).resolve()
    prediction_dir = Path(prediction_dir).resolve()
    prediction_dir.mkdir(parents=True, exist_ok=True)
    packet = deepcopy(read_json(manifest))
    input_source = source(manifest)
    started, runs = time.monotonic(), []
    for entry in packet["entries"]:
        prediction = entry.get("prediction_input")
        if not prediction:
            continue
        shared_case = ((manifest.parent / prediction["case"]).resolve()
                       if "case" in prediction else None)
        mapped = {}
        for index, scenario in enumerate(prediction.get("scenarios", [prediction])):
            declaration = {**prediction, **scenario}
            case_path = (manifest.parent / declaration["case"]).resolve()
            spec = {
                "case": read_json(case_path), "geometry": declaration["geometry"],
                "level": "graph", "max_operators": declaration.get("max_operators", 30000),
                "top": declaration.get("top", 30),
                **{key: declaration[key] for key in
                   ("driver", "proposal_tokens", "draft_layers") if key in declaration},
            }
            remaining = total_timeout - (time.monotonic() - started)
            result = (run_prediction(spec, timeout=min(timeout, remaining)) if remaining > 0
                      else {"status": "failed", "reason": "scoreboard_prediction_budget_exhausted"})
            output = prediction_dir / f"{entry['id']}-{index}-prediction.json"
            write_json(output, result)
            scenario["report"] = os.path.relpath(output, output_manifest.parent)
            if "case" in scenario:
                scenario["case"] = os.path.relpath(case_path, output_manifest.parent)
            row = {
                "id": entry["id"], "scenario": index, "geometry": spec["geometry"],
                "status": result["status"], "report": source(output),
                "case": source(case_path),
                **{key: result[key] for key in
                   ("runtime", "runtime_integrity", "execution_identity", "graph",
                    "resources", "worker_usage", "wall_seconds", "reason", "error") if key in result},
            }
            if result["status"] == "modeled":
                mapping = family_mapping(result, entry.get("family_recipe_groups"))
                row["family_mapping"] = mapping
                for family, recipes in mapping["groups"].items():
                    mapped.setdefault(family, set()).update(recipes)
            runs.append(row)
            print(json.dumps({key: row[key] for key in
                              ("id", "scenario", "status", "wall_seconds") if key in row}),
                  flush=True)
        if shared_case is not None:
            prediction["case"] = os.path.relpath(shared_case, output_manifest.parent)
        for binding in entry.get("family_bindings", ()):
            if binding["id"] in mapped:
                binding["predicted_groups"] = sorted(mapped[binding["id"]])
    execution = {
        "source_revision": revision, "input_manifest": input_source,
        "limits": {"per_probe_seconds": timeout, "total_prediction_seconds": total_timeout},
        "wall_seconds": time.monotonic() - started, "runs": runs,
        "refresh_driver": source(__file__),
    }
    write_json(output_manifest, packet)
    return execution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest")
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--execution-report", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--timeout", type=float, default=40)
    parser.add_argument("--total-timeout", type=float, default=150)
    args = parser.parse_args(argv)
    report = refresh(
        args.manifest, args.output_manifest, args.prediction_dir,
        revision=args.revision, timeout=args.timeout, total_timeout=args.total_timeout)
    write_json(args.execution_report, report)
    return 0 if all(row["status"] == "modeled" for row in report["runs"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
