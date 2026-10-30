"""Compare compact KDA owner runs with the retained expanded source control.

[Role] Execute source-selected stage geometry through the real shared Engine.
[Invariants] Only case declarations supply rates; observations are never read.
[Timeouts & Locks] Sequential bounded four-geometry local verification.
[Side-Effect Policy] Writes the requested JSON; no device or repository edits.
"""
import argparse
import json
from pathlib import Path
import time

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators import kda_prepare, kda_prepare_compact
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView

if __package__:
    from .model_convergence import _runtime_identity, worker_usage
else:
    from model_convergence import _runtime_identity, worker_usage


def run(case, length, compact):
    value = {"lengths": [length], "heads": 12, "key_heads": 12,
             "dk": 128, "dv": 128, "chunk": 64, "compact": compact}
    started = time.monotonic()
    op = kda_prepare.make_op(value)
    parsed = parse_case(case)
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(kda_prepare.build)})
    execution = Execution(parsed, CostModel(parsed), recipes=recipes,
                          include_submission=False)
    plan = IterationPlan((RequestView("probe", length, 0, True, "prefill"),))
    iteration = IterationWork((op,), plan, length, 1, length, None, "prefill", "one")
    scheduled = execution.add(iteration, 0, 3)
    execution.engine.advance(until=(scheduled.completion,))
    state = execution.engine.summary()
    return {
        "tokens": length, "heads": 12, "compact": compact,
        "duration_ms": execution.engine.now - 3, "input_ready_ms": 3,
        "wall_seconds": time.monotonic() - started,
        "serviced": {name: amount for name, amount in execution.engine.serviced.items() if amount},
        "completed_tasks": state["completed_tasks"],
        "peak_live_tasks": state["peak_live_tasks"],
        "live_tasks": state["live_tasks"], "live_programs": state["live_programs"],
        "live_allocated": {name: amount for name, amount in state["live_allocated"].items() if amount},
        "mapping": execution.cost.summary(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Prior Kimi scenario input, not its measured report")
    parser.add_argument("-o", "--output", required=True, type=Path)
    args = parser.parse_args()
    packet = json.loads(args.input.read_text())
    case = packet.get("case", packet)
    before = _runtime_identity()
    records = []
    for length in (64, 128, 512, 8192):
        kda_prepare_compact._NOMINAL.clear()
        expanded = run(case, length, False)
        cold = run(case, length, True)
        warm = run(case, length, True)
        names = expanded["serviced"].keys() | cold["serviced"].keys()
        delta = {name: cold["serviced"].get(name, 0) - expanded["serviced"].get(name, 0)
                 for name in names}
        records.append({
            "tokens": length, "expanded": expanded, "compact_cold": cold,
            "compact_warm": warm,
            "timing_difference_ms": cold["duration_ms"] - expanded["duration_ms"],
            "max_relative_service_difference": max(
                (abs(delta[name]) / max(1, abs(expanded["serviced"].get(name, 0)))
                 for name in names), default=0),
        })
    report = {
        "schema": "kda-prepare-compact-control/1", "source_input": str(args.input.resolve()),
        "status": "passed" if all(
            row["max_relative_service_difference"] < 1e-9
            and all(
                not row[mode]["live_tasks"] and not row[mode]["live_programs"]
                and not row[mode]["live_allocated"]
                for mode in ("expanded", "compact_cold", "compact_warm"))
            for row in records) else "failed",
        "runtime_before": before, "runtime_after": _runtime_identity(),
        "rows": records, "worker_usage": worker_usage(),
        "interpretation": [
            "Only source stage prediction, not measurement accuracy.",
            "Owner finite nominal schedules use the same Engine and declared rates.",
            "Concurrent owner or external resource contention uses coupled fluid progress.",
            "Input release is 3 ms; durations exclude that release delay.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{
        "tokens": row["tokens"], "expanded_ms": row["expanded"]["duration_ms"],
        "compact_ms": row["compact_cold"]["duration_ms"],
        "cold_s": row["compact_cold"]["wall_seconds"],
        "warm_s": row["compact_warm"]["wall_seconds"],
        "tasks": row["compact_warm"]["completed_tasks"],
        "work_error": row["max_relative_service_difference"],
    } for row in records], indent=2))
    if report["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
