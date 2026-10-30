"""Compare structural operators with raw, shape-resolved profiler kernels.

[Role] Reproduce operator-level differences without fitting their residual.
[Invariants] Hardware/software identity is caller-supplied provenance. A shape
  match compares an analytical program, not proof of a vendor tiling match.
  Missing raw shapes are never inferred from serving batch size.
[Timeouts & Locks] Streaming CSV, bounded distinct signatures and samples.
[Side-Effect Policy] Writes only the requested report.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from mindie_sim import Case
from mindie_sim.observations.kernels import PROFILER_API_REQUIREMENT, kernel_rows
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators


from mindie_sim.structure_contracts import (
    GDN_KINDS, comparison_scope, conv_contract, gdn_contract, predicted_contract,
)


def observed_contract(row):
    try:
        from mindie_profiler.tensor_contracts import observed_contract as captured_contract
    except ModuleNotFoundError as exc:
        if exc.name not in {"mindie_profiler", "mindie_profiler.tensor_contracts"}:
            raise
        raise ModuleNotFoundError(PROFILER_API_REQUIREMENT) from exc
    return captured_contract(row)


def capture_groups(path, *, start_line=2, end_line=None, sample_limit=100000):
    groups, total, unresolved = {}, 0, Counter()
    for row in kernel_rows(path):
        if row.line < start_line:
            continue
        if end_line is not None and row.line > end_line:
            break
        total += 1
        if total > sample_limit:
            raise ValueError("selected_capture_exceeds_sample_limit:select_a_bounded_line_window")
        contract = observed_contract(row)
        if contract is None:
            unresolved["missing_shapes" if row.input_shapes is None else "unmapped_contract"] += 1
        key = contract, row.signature
        if key not in groups:
            groups[key] = {"contract": contract, "kernel": row.name, "core": row.core,
                           "state": row.state, "blocks": row.blocks, "samples_us": [],
                           "input_shapes": row.input_shapes, "output_shapes": row.output_shapes,
                           "source_lines": [], "counters": defaultdict(list)}
        group = groups[key]
        group["samples_us"].append(row.duration_ns / 1000)
        if len(group["source_lines"]) < 3:
            group["source_lines"].append(row.line)
        for counter, value in row.counters_us.items():
            group["counters"][counter].append(value)
    return list(groups.values()), dict(unresolved)


def compare(case, path, *, phase, tokens, context, sequences=1, baseline=None,
            start_line=2, end_line=None, families=()):
    ops = target_operators(case, phase=phase, tokens=tokens, context=context, sequences=sequences)
    if families:
        ops = tuple(op for op in ops if op.family in families)
    prediction = operator_costs(case, ops)
    groups, unresolved = capture_groups(path, start_line=start_line, end_line=end_line)
    by_contract = defaultdict(list)
    for group in groups:
        if group["contract"] is not None:
            by_contract[group["contract"]].append(group)
    before = {}
    for row in (baseline or {}).get(phase, {}).get("operators", ()) if sequences == 1 else ():
        op = row["operator"]
        before[(op["recipe"], tuple(op["local_shape"]), op["dtype"])] = row["duration_ms"] * 1000
    comparisons, matched = [], set()
    predicted_calls = Counter()
    for item in prediction["operators"]:
        contract = predicted_contract(item["operator"])
        if contract is not None:
            predicted_calls[contract] += item["occurrences"]
    for item in prediction["operators"]:
        op = item["operator"]
        contract = predicted_contract(op)
        for group in by_contract.get(contract, ()):
            actual, forecast = median(group["samples_us"]), item["duration_ms"] * 1000
            matched.add(id(group))
            comparisons.append({
                "recipe": op["recipe"], "shape": op["local_shape"], "dtype": op["dtype"],
                "program": item["program"],
                "model_calls": item["occurrences"], "kernel": group["kernel"],
                "capture_calls": len(group["samples_us"]), "launch_blocks": group["blocks"],
                "contract_model_calls": predicted_calls[contract],
                "contract_capture_calls": sum(len(g["samples_us"]) for g in by_contract[contract]),
                "core": group["core"], "state": group["state"],
                "source_lines": group["source_lines"],
                "observed_median_us": actual, "predicted_us": forecast,
                "error_percent": (forecast / actual - 1) * 100,
                "before_us": before.get((op["recipe"], tuple(op["local_shape"]), op["dtype"])),
                "pipe_median_us": {k: median(v) for k, v in group["counters"].items()},
                "dispatch": op["metadata"].get("dispatch"),
                "declared_sequence_lengths": op["metadata"].get("lengths",
                                                               op["metadata"].get("sequence_tokens")),
                "observed_input_shapes": group["input_shapes"],
                "observed_output_shapes": group["output_shapes"],
                "declared_service": item["declared_service"],
                "serviced": item["serviced"],
                "remaining_buffers": item["remaining_buffers"],
                "match": comparison_scope(op),
            })
    unmatched = sorted(({
        "kernel": group["kernel"], "calls": len(group["samples_us"]),
        "sum_us": sum(group["samples_us"]), "source_lines": group["source_lines"],
        "reason": "missing_or_unmapped_contract" if group["contract"] is None else "no_matching_operator",
    } for group in groups if id(group) not in matched), key=lambda row: row["sum_us"], reverse=True)
    return {
        "schema": "mindie-sim.operator-comparison/1",
        "source": str(path), "line_window": [start_line, end_line],
        "model": case.model.name, "hardware": case.machine.hardware_key,
        "software": {"runtime": case.runtime.vllm_impl, "revision": case.runtime.vllm_version},
        "phase": phase, "tokens_per_sequence": tokens, "sequences": sequences, "context": context,
        "selected_families": list(families),
        "hardware_declaration": case.machine.to_dict(),
        "calibration": "comparison rows never enter prediction parameters",
        "declared_parameter_sources": dict(case.runtime.metadata).get("primitive_parameter_sources", ()),
        "operator_sum_ms": prediction["operator_sum_ms"],
        "before_operator_sum_ms": (baseline or {}).get(phase, {}).get("operator_sum_ms") if sequences == 1 else None,
        "baseline_revision": (baseline or {}).get("revision"),
        "boundary": prediction["boundary"], "comparisons": comparisons,
        "unresolved_rows": unresolved, "unmatched_kernels": unmatched,
        "matched_capture_kernel_sum_ms": sum(sum(g["samples_us"]) for g in groups if id(g) in matched) / 1000,
        "selected_capture_kernel_sum_ms": sum(sum(g["samples_us"]) for g in groups) / 1000,
        "serving_accuracy_accepted": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case")
    parser.add_argument("capture")
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument("--start-line", type=int, default=2)
    parser.add_argument("--end-line", type=int)
    parser.add_argument("--baseline")
    parser.add_argument("--family", action="append", default=[],
                        help="Price only this operator family; repeat for a bounded replay")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    case = Case.from_mapping(json.loads(Path(args.case).read_text()))
    baseline = json.loads(Path(args.baseline).read_text()) if args.baseline else None
    report = compare(case, args.capture, phase=args.phase, tokens=args.tokens, context=args.context,
                     sequences=args.sequences, baseline=baseline,
                     start_line=args.start_line, end_line=args.end_line, families=args.family)
    files = [Path(args.case), Path(__file__), *Path("mindie_sim").rglob("*.py"),
             Path("mindie_sim/backends/profiles/ascend.json")]
    report["code_and_input_sha256"] = {}
    for path in files:
        with path.open("rb") as stream:
            report["code_and_input_sha256"][str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"comparisons": len(report["comparisons"]),
                      "operator_sum_ms": report["operator_sum_ms"],
                      "unresolved_rows": report["unresolved_rows"]}))


if __name__ == "__main__":
    main()
