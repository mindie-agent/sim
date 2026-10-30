"""Score frozen primitive parameters through the production execution engine.

[Role] Compare address-transfer programs with independent primitive observations.
[Invariants] Reads frozen parameters; does not optimize against validation rows.
  Queue service excludes the probe's seed load, conversion and kernel entry.
[Timeouts & Locks] Finite tile programs and bounded queue state, no device calls.
[Side-Effect Policy] Writes only the selected comparison report.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mindie_sim.catalog.ops import vector_op
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.pointwise import TiledPointwise
from mindie_sim.prediction.program import RecipeContext


def verify(protocol, source, output):
    parameters = json.loads(protocol.read_text())["candidate"]
    queue = {
        "outstanding": parameters["outstanding"], "granule_bytes": parameters["granule_bytes"],
        "response_ms": parameters["response_ns"] / 1e6,
        "row_control_ms": parameters["row_control_ns"] / 1e6,
    }
    results = []
    with source.open() as stream:
        for line in stream:
            sample = json.loads(line)
            if "durations_us" not in sample:
                continue
            units, rows, cols = sample["units"], sample["tokens"], sample["heads"]
            rates = {"hbm": 4e9, "vector": 1e20,
                     "scalar_store_issue": units * 1e6 / parameters["issue_ns"]}
            engine = Engine([Resource(key, value) for key, value in rates.items()])
            op = vector_op(
                op_id="primitive", deps=(), stage=0, layer=0, requests=(), elements=rows * cols,
                ops_per_element=0, dtype="bf16", family="store", phase="primitive", recipe="store",
                shape=(rows, cols))
            context = RecipeContext(op, {}, {key: key for key in rates}, rates, units,
                                    resource_units={key: units for key in rates})
            recipe = TiledPointwise(0, ({
                "bytes_per_element": 2, "resource": "scalar_store_issue", "queue": queue,
                "row_stride_bytes": sample["leading_dim"] * 2,
                "offset_bytes": sample["offset_elements"] * 2,
            },))
            done = recipe.lower(context).start(engine, "primitive", engine.submit("ready"))
            engine.advance(until=(done,))
            observed = statistics.median(sample["durations_us"])
            results.append({
                **sample, "predicted_us": done.time * 1000,
                "measured_median_us": observed,
                "error_pct": 100 * (done.time * 1000 / observed - 1),
                "engine": engine.summary(),
            })
    output.write_text(json.dumps({
        "source": str(source), "frozen_protocol": str(protocol),
        "scope": "modeled scalar transfer queue vs measured primitive event interval",
        "omitted_probe_stages": ["seed DMA and conversion", "kernel entry/exit"],
        "results": results,
    }, indent=2) + "\n")
    print(json.dumps({"cases": len(results),
                      "max_absolute_error_pct": max(abs(r["error_pct"]) for r in results),
                      "max_live_tasks": max(r["engine"]["peak_live_tasks"] for r in results)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("protocol", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.protocol, args.source, args.output)
