"""Verify a source-selected KDA state phase on an explicit case.

[Role] Run ordinary frontend selection and the shared graph/isolated APIs.
[Invariants] The verifier checks independent physical arithmetic and finite
  state, not measurement accuracy. No observed duration enters the simulation.
[Timeouts & Locks] Finite local CPU runs; caller supplies the process budget.
[Side-Effect Policy] Writes only the requested JSON report.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import resource
import time

from mindie_sim.prediction.family_kda_state import IMPLEMENTATION, SOURCE
from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.operators import kda_state
from mindie_sim.prediction.structure import parse_case


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def expected_work(tokens, heads, chunk=64, k=128, v=128):
    """Closed forms from initial H0 plus the C1/V1/C2/V2 GM intervals."""
    count = (tokens + chunk - 1) // chunk
    reads = heads * (6*tokens*k + 8*tokens*v + 10*count*k*v
                     + 4*count*k + 4*k*v)
    writes = heads * (2*tokens*k + 8*tokens*v + 10*count*k*v)
    fixpipe = heads * (4*tokens*v + 4*count*k*v)
    operand = heads * (6*tokens*k + 2*count*k*v)
    return {
        "hbm": reads + writes, "mte2": reads, "mte3": writes - fixpipe,
        "fixpipe": fixpipe, "mte1": operand,
        "matrix_flops": 4*tokens*heads*k*v,
    }


def run(case_path, tokens, rss_byte_factor):
    raw = json.loads(case_path.read_text())
    raw["runtime"].setdefault("kernels", {}).setdefault(
        "lowerings", {})["kda"] = IMPLEMENTATION
    case = parse_case(raw)
    layer = next(plan for plan in case.model.layer_plans if plan.attention == "kda")
    # Isolate an existing structural layer. Precision, placement and physical
    # resource declarations remain those of the supplied case.
    case = replace(
        case, model=replace(case.model, layers=1, layer_plans=(replace(layer, index=0),)),
        runtime=replace(case.runtime, pp_layer_partition=(1,)))
    module = Path(kda_state.__file__)
    before = hashlib.sha256(module.read_bytes()).hexdigest()
    rows = []
    for length in (512, tokens):
        started = time.perf_counter()
        ops = target_operators(case, phase="prefill", tokens=length,
                               sequences=1, scope="model")
        selected = [op for op in ops if op.recipe == kda_state.RECIPE]
        assert len(selected) == 1
        op = selected[0]
        meta = dict(op.metadata)
        report = graph_costs(
            case, selected, external_inputs={dep: 3 for dep in op.deps},
            input_ready_ms=3, trace_limit=0)
        isolated = report["isolated"]["operators"][0]
        expected = expected_work(length, meta["heads"])
        serviced = isolated["serviced"]
        assert math.isclose(serviced["memory@r0:s0"], expected["hbm"], rel_tol=1e-9)
        for role in ("mte1", "mte3", "fixpipe"):
            assert math.isclose(serviced[f"{role}@r0:s0"], expected[role], rel_tol=1e-9)
        reads = sum(value for key, value in serviced.items()
                    if key.startswith(("mte2@", "mte2_aic@", "mte2_aiv@")))
        assert math.isclose(reads, expected["mte2"], rel_tol=1e-9)
        assert math.isclose(isolated["operator"]["flops"],
                            expected["matrix_flops"], rel_tol=1e-9)
        for name, declared in isolated["declared_service"].items():
            assert math.isclose(serviced.get(name, 0), declared, rel_tol=1e-9, abs_tol=1e-12)
        execution = report["execution"]
        assert execution["live_tasks"] == execution["live_programs"] == 0
        assert not any(execution["live_allocated"].values())
        assert not isolated["service_before_input"]
        assert math.isclose(report["duration_ms"], isolated["duration_ms"], rel_tol=1e-9)
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_byte_factor
        assert rss <= 512 * 1024**2
        rows.append({
            "tokens": length, "heads": meta["heads"], "chunk": meta["chunk"],
            "expected_work": expected, "graph_duration_ms": report["duration_ms"],
            "isolated_duration_ms": isolated["duration_ms"],
            "serviced": serviced, "execution": execution,
            "input_dependency_obeyed": True,
            "operator": isolated["operator"], "parameters": report["parameters"],
            "wall_seconds": time.perf_counter() - started, "peak_rss_bytes": rss,
        })
    after = hashlib.sha256(module.read_bytes()).hexdigest()
    assert before == after
    assert rows[0]["execution"]["peak_live_tasks"] == rows[1]["execution"]["peak_live_tasks"]
    call = call_costs(case, (replace(op, deps=()),), invocations=2, input_ready_ms=3)
    assert math.isclose(call["duration_ms"], 2 * rows[-1]["isolated_duration_ms"], rel_tol=1e-9)
    assert call["single_call"]["source_boundaries"] == {"kernel": 1}
    assert call["execution"]["live_tasks"] == call["execution"]["live_programs"] == 0
    assert not any(call["execution"]["live_allocated"].values())
    final_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_byte_factor
    assert final_rss <= 512 * 1024**2
    return {
        "schema": "mindie-sim.kda-state-verification/1",
        "scope": "source_phase_resource_contract_not_accuracy",
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "case": str(case_path), "case_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "normalized_single_layer_case_sha256": digest(case.to_dict()),
        "source": SOURCE, "module_sha256": before,
        "input_ready_ms": 3, "bounded_active_state": True,
        "rss_byte_factor": rss_byte_factor, "runs": rows, "two_call_protocol": call,
        "final_peak_rss_bytes": final_rss,
        "limitations": [
            "Only FWD_H is source-selected; the full KDA call is not simulated here.",
            "Vector phases are AIV-pair envelopes; odd V1 tails are not lane-exact.",
            "L1 tail fill payload is recorded but unpriced; full chunks have none.",
            "Flag, queue and synchronization durations are not inferred.",
            "Per-owner initialization aggregates casts before the global H0 barrier.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--rss-byte-factor", type=int, required=True,
                        help="Native ru_maxrss to bytes: 1 on macOS, 1024 on Linux.")
    args = parser.parse_args()
    report = run(args.case, args.tokens, args.rss_byte_factor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output), "runs": [
            {key: row[key] for key in ("tokens", "isolated_duration_ms",
                                       "wall_seconds", "peak_rss_bytes")}
            for row in report["runs"]]}))


if __name__ == "__main__":
    main()
