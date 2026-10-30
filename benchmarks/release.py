"""CPU cost of the maintained engine and public serving API.

[Role] Benchmark finite pipelines, colocated serving and disaggregated serving.
[Invariants] Verify completion and conservation before reporting wall time.
  CPU microbenchmark results are not device performance or model accuracy.
[Timeouts & Locks] Bounded iteration counts; no devices, network or sleeps.
[Side-Effect Policy] Writes MINDIE_BENCH_RESULT_PATH only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

from mindie_sim import predict
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.program import PipelineProgram, Stage

ROOT = Path(__file__).resolve().parents[1]


def pipeline():
    engine = Engine([Resource("load", 1), Resource("compute", 1)])
    ready = engine.submit("ready")
    program = PipelineProgram(100, 2, lambda _: (
        Stage("load", {"load": 1}), Stage("compute", {"compute": 2}, after=("load",))))
    done = program.start(engine, "pipeline", ready)
    engine.advance(until=(done,))
    assert done.time == 201
    assert engine.serviced == {"load": 100, "compute": 200}
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return engine.summary()


def serving(path):
    case = json.loads((ROOT / path).read_text())

    def run():
        result = predict(case, trace_limit=0, event_limit=0)
        assert result["status"] == "modeled"
        assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
        assert result["metrics"]["output_tokens"] == 14
        for row in result["requests"]:
            assert math.isclose(row["e2e_ms"], row["ttft_ms"] + (row["output_tokens"] - 1) * row["tpot_ms"])
        if "pools" in case:
            assert result["execution"]["handoffs"] == 2
            assert result["execution"]["handoff_bytes"] > 0
        return result
    return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("all", "pipeline", "serving", "pd-serving"), default="all")
    args = parser.parse_args()
    output = os.environ.get("MINDIE_BENCH_RESULT_PATH")
    if not output:
        parser.error("MINDIE_BENCH_RESULT_PATH is required")
    cases = {"pipeline": (pipeline, 20), "serving": (serving("examples/quickstart.json"), 3),
             "pd-serving": (serving("examples/disaggregated.json"), 3)}
    metrics = []
    for name in cases if args.case == "all" else (args.case,):
        run, count = cases[name]
        run()
        start = time.perf_counter_ns()
        for _ in range(count):
            run()
        metrics.append({"name": name, "unit": "us/op", "iterations": count,
                        "value": (time.perf_counter_ns() - start) / count / 1000})
    Path(output).write_text(json.dumps({"status": "pass", "metrics": metrics}, indent=2) + "\n")


if __name__ == "__main__":
    main()
