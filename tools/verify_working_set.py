"""Run finite execution workloads and record bounded state.

[Role] Check live-task cleanup, trace bounds and process peak resident memory.
[Invariants] Prompt/output lengths do not become retained execution history.
  RSS is cumulative in this process; this is not device accuracy evidence.
[Timeouts & Locks] Finite local CPU workloads; no locks, network or subprocesses.
[Side-Effect Policy] Writes only the requested JSON report.
"""
import argparse
import json
import math
from pathlib import Path
import resource
import time

from mindie_sim import predict
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.matrix import HierarchicalMatmul
from mindie_sim.prediction.program import PipelineProgram, RecipeContext, Stage
from mindie_sim.prediction.transfer import FiniteTransfer
from mindie_sim.prediction.transactions import QueuedStore
from mindie_sim.prediction.collectives import RingGroup
from mindie_sim.prediction.communication import MessageFlow, RoutedHop, RoutedMessage


CASES = ("long_pipeline", "scalar_queue", "finite_transfer", "ring_rounds",
         "matrix_refill", "long_decode", "long_prefill", "catalog_qwen35")


def matrix_refill(rss_byte_factor):
    m = n = 512
    cores, refill_slots, consumer_slots = 4, 2, 2
    macro, micro, panel_k, width = 256, 128, 128, 2
    pools = {
        "l1": refill_slots * (macro + macro) * panel_k * width,
        "l0a": consumer_slots * micro * panel_k * width,
        "l0b": consumer_slots * micro * panel_k * width,
        "l0c": macro * macro * 4,
    }
    rates = {"cube": cores * 2 * micro**2 * panel_k, "hbm": 1e9,
             "mte2": cores * 2 * macro * panel_k * width,
             "mte1": cores * 2 * micro * panel_k * width, "mte3": 1e9}
    units = {name: 1 if name == "hbm" else cores for name in rates}
    recipe = HierarchicalMatmul(macro, macro, panel_k, micro, micro,
                                refill_slots, consumer_slots)
    # Each owner retains two refill loads and at most two micro DAGs
    # (operand, matrix, optional final store, retire), plus bounded join tasks.
    task_bound = cores * (refill_slots + 4 * consumer_slots + 1) + 4
    runs = []
    for panels in (100, 10000):
        started = time.perf_counter()
        k = panels * panel_k
        op = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(),
                     m=m, n=n, k=k, quant=QuantSpec("bf16"),
                     category="dense_mlp", phase="prefill")
        engine = Engine([
            Resource(name, rate, "flops" if name == "cube" else "bytes",
                     units=units[name]) for name, rate in rates.items()
        ], pools=pools, pool_units={name: cores for name in pools}, trace_limit=8)
        context = RecipeContext(
            op, {"cube": op.flops}, {name: name for name in rates}, rates, cores,
            pool_names={name: name for name in pools}, pool_capacities=pools,
            resource_units=units)
        done = recipe.lower(context).start(engine, "matrix", engine.submit("ready"))
        engine.advance(until=(done,))
        summary = engine.summary()
        a, b, c = m * k * width, n * k * width, m * n * width
        # Dense geometry gives independent traffic counts at all three levels:
        # one ideal shared-cache HBM read; repeated private macro and micro copies.
        expected = {"cube": 2 * m * n * k, "hbm": a + b + c,
                    "mte2": a * math.ceil(n / macro) + b * math.ceil(m / macro),
                    "mte1": a * math.ceil(n / micro) + b * math.ceil(m / micro),
                    "mte3": c}
        assert summary["serviced"].keys() == expected.keys()
        for name, amount in expected.items():
            assert math.isclose(summary["serviced"][name], amount, rel_tol=1e-9), name
        assert summary["peak_live_tasks"] <= task_bound
        assert summary["live_tasks"] == summary["live_programs"] == 0
        assert not engine.allocations
        assert not any(summary["live_allocated"].values())
        assert summary["retained_trace_events"] <= 8
        for pool, amount in summary["peak_allocated"].items():
            assert 0 < amount <= pools[pool]
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_byte_factor
        assert rss_bytes <= 512 * 1024**2
        runs.append({
            "panels_per_owner": panels, "k": k, "duration_ms": done.time,
            "wall_seconds": time.perf_counter() - started,
            "peak_rss_bytes": rss_bytes, "expected_work": expected,
            "serviced": summary["serviced"], "completed_tasks": summary["completed_tasks"],
            "peak_live_tasks": summary["peak_live_tasks"],
            "peak_allocated_per_unit": summary["peak_allocated_per_unit"],
            "live_allocated": summary["live_allocated"],
            "live_allocation_count": len(engine.allocations),
            "live_tasks": summary["live_tasks"], "live_programs": summary["live_programs"],
            "retained_trace_events": summary["retained_trace_events"],
        })
    assert runs[0]["peak_live_tasks"] == runs[1]["peak_live_tasks"]
    assert runs[0]["peak_allocated_per_unit"] == runs[1]["peak_allocated_per_unit"]
    details = {
        "dtype": "bf16", "cores": cores, "macro_tile": [macro, macro, panel_k],
        "micro_tile": [micro, micro, panel_k],
        "refill_slots": refill_slots, "consumer_slots": consumer_slots,
        "resource_rates_per_ms": rates, "resource_units": units,
        "pool_capacity_bytes_per_unit": pools, "peak_live_tasks_bound": task_bound,
        "assumptions": [
            "Synthetic resource rates; mechanism boundedness, not hardware accuracy.",
            "Ideal shared-cache HBM reuse preserves every private macro/micro copy.",
            "RSS is the process cumulative peak, not a per-engine allocation.",
        ],
        "active_state_equal_across_k": True, "variants": runs,
    }
    return summary, {"m": m, "n": n, "panels_per_owner": [100, 10000]}, details


def verify(root, rss_byte_factor, only=None):
    rows = []
    for kind in CASES:
        if only and kind not in only:
            continue
        start = time.perf_counter()
        details = {}
        if kind == "long_pipeline":
            engine = Engine([Resource("memory", 1000), Resource("compute", 1000)], trace_limit=8)
            ready = engine.submit("ready")
            program = PipelineProgram(100000, 2, lambda i: (
                Stage("load", {"memory": 16}),
                Stage("compute", {"compute": 32}, after=("load",))))
            done = program.start(engine, "finite", ready)
            engine.advance(until=(done,))
            summary, tokens, workload = engine.summary(), None, "100000 tiles"
            assert summary["serviced"] == {"memory": 1600000, "compute": 3200000}
        elif kind == "scalar_queue":
            engine = Engine([Resource("issue", 1e6), Resource("memory", 1e9)], trace_limit=8)
            program = QueuedStore(25000, 4, 128, 2, 0, 1, "issue", 1e6, "memory",
                                  128, 8, 91.25e-6, 7e-6)
            done = program.start(engine, "finite_store", engine.submit("ready"))
            engine.advance(until=(done,))
            summary, tokens, workload = engine.summary(), None, "100000 scalar writes, 8 queue slots"
            assert math.isclose(summary["serviced"]["issue"], 100000, rel_tol=1e-12)
            assert summary["serviced"]["memory"] == 200000
        elif kind == "finite_transfer":
            engine = Engine([Resource("wire", 1000, "bytes")], trace_limit=8)
            program = FiniteTransfer({"wire": 1_000_000}, 10, 8, .2)
            done = program.start(engine, "finite_transfer", engine.submit("ready"))
            engine.advance(until=(done,))
            summary, tokens, workload = engine.summary(), None, "100000 transactions, 8 credits"
            # Independent homogeneous serial-server/credit-window closed form.
            expected = 100000 * .01 + .2 + ((100000 - 1) // 8) * (.2 - 7 * .01)
            assert math.isclose(done.time, expected, rel_tol=1e-9)
            assert summary["serviced"]["wire"] == 1_000_000
            assert summary["peak_live_tasks"] <= 2 * 8 + 2
        elif kind == "ring_rounds":
            rounds = 100000
            engine = Engine([Resource(f"edge{rank}", 1, "bytes") for rank in range(4)],
                            trace_limit=8)
            group = RingGroup(tuple(range(4)), rounds, tuple(
                RoutedMessage(MessageFlow(rank, (rank + 1) % 4, 1), (
                    RoutedHop(f"edge{rank}", 1, 1, None, None, 10 if rank == 0 else 0),
                )) for rank in range(4)))
            done = group.start(engine, "ring", engine.submit("ready"))
            engine.advance(until=(done,))
            sent, received = [0] * 4, [0] * 4
            for _ in range(rounds):
                sent = [max(sent[i], received[(i - 1) % 4]) + 1 for i in range(4)]
                received = [sent[i] + (10 if i == 0 else 0) for i in range(4)]
            assert done.time == max(received)
            summary, tokens, workload = engine.summary(), None, "100000 rounds, 4 ranks, unequal latency"
            assert summary["serviced"] == {f"edge{rank}": rounds for rank in range(4)}
        elif kind == "matrix_refill":
            summary, workload, details = matrix_refill(rss_byte_factor)
            tokens = None
        else:
            filename = "examples/cases/qwen35-a5.json" if kind == "catalog_qwen35" else "examples/quickstart.json"
            case = json.loads((root / filename).read_text())
            prompt, output = {"long_decode": (16, 2000), "long_prefill": (131072, 4),
                              "catalog_qwen35": (4096, 8)}[kind]
            workload = {"prompt": prompt, "output": output}
            case["workload"] = {"requests": [{"id": "bound", "prompt_tokens": prompt, "output_tokens": output}]}
            case["runtime"].update(max_model_len=prompt + output, max_num_batched_tokens=512)
            result = predict(case, trace_limit=1, event_limit=8)
            assert result["status"] == "modeled", result["requests"]
            summary, tokens = result["execution"], result["metrics"]["output_tokens"]
            assert tokens == output
        assert summary["live_tasks"] == summary["live_programs"] == 0
        assert summary["retained_trace_events"] <= 8
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_byte_factor
        assert rss_bytes <= 512 * 1024**2
        row = {
            "case": kind, "workload": workload, "wall_seconds": time.perf_counter() - start,
            "peak_rss_bytes": rss_bytes, "peak_live_tasks": summary["peak_live_tasks"],
            "live_tasks": summary["live_tasks"], "live_programs": summary["live_programs"],
            "retained_trace_events": summary["retained_trace_events"], "output_tokens": tokens,
        }
        if details:
            row["mechanism"] = details
        rows.append(row)
        print(json.dumps(row), flush=True)
    return {"scope": "Local process peak RSS, cumulative; mechanism boundedness only.",
            "rss_byte_factor": rss_byte_factor, "runs": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--rss-byte-factor", type=int, required=True,
                        help="Native ru_maxrss to bytes: 1 on macOS, 1024 on Linux.")
    parser.add_argument("--only", action="append", choices=CASES,
                        help="Run only this case; repeat to select multiple cases.")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    Path(args.output).write_text(json.dumps(
        verify(args.root, args.rss_byte_factor, args.only), indent=2) + "\n")
