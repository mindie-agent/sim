"""Retain source-caller AttnRes candidates and real isolated Engine costs.

[Role] Evaluate every caller-reachable scalar combination without timing input.
[Invariants] A source point histogram is not an observed event/layer binding.
  Capture extents can condition candidates but cannot select history or flags.
[Timeouts & Locks] Finite model layers and bounded per-owner plans; no devices.
[Side-Effect Policy] Reads small declarations and writes one requested report.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

from mindie_sim.prediction.family_attnres import (
    AttnResPrefixState, IMPLEMENTATION, SOURCE, attn_res_point,
)
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.structure import parse_case


AXES = ("tokens_local", "hidden", "valid_blocks", "bank_capacity",
        "block_token_stride", "fuse_add", "block_write_idx", "mix",
        "output_norm", "save_materialized", "optimize_prefill")


def caller_points(case, tokens, phase, optimize_prefill):
    state = AttnResPrefixState.start(
        model=case.model, runtime=case.runtime, ready="embedding_or_sp_shard",
        tokens=tokens, mode=phase, requests=(), optimize_prefill=optimize_prefill)
    points = []
    for layer in case.model.layer_plans:
        prefix = f"l{layer.index}"
        points.extend(state.prepare(prefix=prefix + ".attention", layer=layer, stage=0).ops)
        points.extend(state.attention_added(
            prefix=prefix + ".ffn", layer=layer, stage=0,
            attention_id=prefix + ".attention_result").ops)
        state.defer_mlp(prefix + ".mlp_result")
    points.extend(state.final(
        prefix="final", last_layer=case.model.layer_plans[-1], stage=0).ops)
    return points


def evaluate(case, tokens, phase, optimize_prefill):
    points = caller_points(case, tokens, phase, optimize_prefill)
    counts, representatives = Counter(), {}
    for point in points:
        view = dict(point.metadata)["attn_res_view"]
        key = tuple(view[axis] for axis in AXES)
        counts[key] += 1
        representatives.setdefault(key, point)
    # Auxiliary capture layers and the prefill optimization flag are absent
    # from historical descriptors. Preserve reachable alternatives without
    # inventing how many calls used them.
    for point in points:
        metadata = dict(point.metadata)
        view = metadata["attn_res_view"]
        cache_flags = (False, True) if phase == "prefill" else (False,)
        saves = (False, True) if metadata["structure_block"] == "attention" else (False,)
        for cache in cache_flags:
            for save in saves:
                alternative = attn_res_point(
                    {"prefix": point.op_id + ".candidate", "model": case.model,
                     "stage": point.stage, "layer": point.layer, "tokens": tokens,
                     "mode": phase},
                    prefix_id=metadata["attn_res_prefix_id"],
                    addend_id=metadata["attn_res_addend_id"],
                    bank_ids=metadata["attn_res_bank_ids"],
                    valid_blocks=view["valid_blocks"],
                    write_slot=view["block_write_idx"] if view["block_write_idx"] >= 0 else None,
                    output_norm=view["output_norm"], mix=view["mix"], materialize=save,
                    tokens_local=tokens, bank_capacity=view["bank_capacity"],
                    block_token_stride=view["block_token_stride"],
                    optimize_prefill=cache).ops[0]
                candidate_view = dict(alternative.metadata)["attn_res_view"]
                key = tuple(candidate_view[axis] for axis in AXES)
                representatives.setdefault(key, alternative)
    candidates = []
    for index, (key, point) in enumerate(representatives.items()):
        count = counts[key]
        row = operator_costs(case, [point], input_ready_ms=3)["operators"][0]
        operator = row["operator"]
        view = operator["metadata"]["attn_res_view"]
        candidates.append({
            "candidate_id": f"{phase}-T{tokens}-c{index}", "primary_call_count": count,
            "candidate_role": "primary_declared_caller" if count else
                              "conditional_aux_or_optimization_alternative_count_unknown",
            "flags": dict(zip(AXES, key)),
            "physical": {key: view[key] for key in (
                "resident", "used_cores", "tokens_per_core", "output_norm_cached",
                "ub_bytes_per_owner", "vector_work_basis")},
            "duration_ms": row["duration_ms"],
            "weighted_primary_activity_ms": count * row["duration_ms"],
            "work": {name: operator[name] for name in (
                "hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops", "flops")},
            "extended_work": operator["metadata"].get("resource_work", {}),
            "serviced": row["serviced"], "service_delta": row["service_delta"],
            "service_before_input": row["service_before_input"],
            "remaining_buffers": row["remaining_buffers"], "live_tasks": row["live_tasks"],
        })
    return {
        "phase": phase, "tokens_local": tokens, "source_call_count": len(points),
        "candidate_count": len(candidates),
        "predicted_isolated_activity_sum_ms": sum(
            row["weighted_primary_activity_ms"] for row in candidates),
        "candidates": candidates,
        "scope": "one_source_caller_forward_per_representative_rank; isolated_native_point_sum",
        "primary_flags": {"optimize_prefill": bool(optimize_prefill and phase == "prefill"),
                          "materialized_aux_capture": False},
        "excluded": ["attention", "ffn", "sp_collectives", "embedding",
                     "logits_and_final_norm", "host_dispatch"],
    }


def capture_candidates(evidence, scenarios):
    rows = []
    for descriptor in evidence["descriptors"]:
        classification = descriptor["classification"]
        tokens = classification["main_output_rows_if_source_empty_like_view_applies"]
        hidden = classification["parameter_hidden_extent"]
        compatible = [candidate["candidate_id"]
                      for scenario in scenarios for candidate in scenario["candidates"]
                      if candidate["flags"]["tokens_local"] == tokens
                      and candidate["flags"]["hidden"] == hidden]
        rows.append({
            "descriptor_id": descriptor["descriptor_id"],
            "observed_events": descriptor["event_count"],
            "observed_activity_sum_ms": descriptor["activity_sum_us"] / 1000,
            "conditional_candidates": compatible,
            "candidate_basis": "output_extent_under_retained_empty_like_view_assumption",
            "strict_contract_available": False,
            "unresolved": ["active_prefix_view", "bank_stride", "valid_blocks", "fuse_add",
                           "output_norm", "bank_write", "materialization", "optimization_flag",
                           "event_layer_step_identity", "loaded_binary_identity"],
        })
    return rows


def run(args):
    started = time.monotonic()
    raw = json.loads(args.case.read_text())
    case = parse_case(raw)
    kernels = {**case.runtime.kernels, "lowerings": {
        **case.runtime.kernels.get("lowerings", {}),
        "attention_residual": {"implementation": IMPLEMENTATION,
                               "options": {"optimize_prefill": args.optimize_prefill}},
    }}
    case = replace(case, runtime=replace(case.runtime, kernels=kernels))
    scenarios = [evaluate(case, tokens, "decode" if tokens == 1 else "prefill",
                          args.optimize_prefill) for tokens in args.tokens_local]
    package = Path(__file__).resolve().parents[1] / "mindie_sim"
    hashes = {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(package.rglob("*")) if path.suffix in {".py", ".json"}}
    report = {
        "schema": "attnres-source-candidate-verification/1", "source": SOURCE,
        "case_path": str(args.case.resolve()),
        "case_sha256": hashlib.sha256(args.case.read_bytes()).hexdigest(),
        "implementation": IMPLEMENTATION, "optimize_prefill": args.optimize_prefill,
        "runtime_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=package, text=True, timeout=5).strip(),
        "runtime_files_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "runtime_files": hashes, "scenarios": scenarios,
        "comparison_policy": "retain_all_source_reachable_candidates; no_timing_selected_flags",
    }
    if args.capture_evidence:
        evidence = json.loads(args.capture_evidence.read_text())
        report["capture_evidence"] = {
            "path": str(args.capture_evidence.resolve()),
            "sha256": hashlib.sha256(args.capture_evidence.read_bytes()).hexdigest(),
            "capture": evidence["capture"],
            "descriptors": capture_candidates(evidence, scenarios),
        }
    report["execution"] = {
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * args.rss_byte_factor,
        "rss_byte_factor": args.rss_byte_factor,
    }
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output), "scenarios": [{
            key: row[key] for key in ("tokens_local", "source_call_count", "candidate_count",
                                     "predicted_isolated_activity_sum_ms")}
            for row in scenarios], "execution": report["execution"],
    }))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--tokens-local", type=int, nargs="+", default=[1, 1024])
    parser.add_argument("--optimize-prefill", action="store_true")
    parser.add_argument("--capture-evidence", type=Path)
    parser.add_argument("--rss-byte-factor", type=int, default=1,
                        help="ru_maxrss bytes: 1 on macOS, 1024 on Linux.")
    parser.add_argument("-o", "--output", type=Path, required=True)
    run(parser.parse_args())
