"""Bounded production-Engine checks for conditional copy source-program supply.

[Role] Compare frozen PMU responses, owner counts, byte conservation and one Body.
[Invariants] Native durations and Body totals remain comparison-only. A joint
  busy bound is not whole-call accuracy; existing whole-call response is inclusive.
[Timeouts & Locks] One process per mode, independently bounded by 60 seconds.
[Side-Effect Policy] Local input reads and one explicit verification output.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARAMETERS = ROOT / "data/observations/a5-copy-joint-service-20261030/parameters.json"
BODY = ROOT / ("examples/cases/high-concurrency-20261030/glm-observable-service/"
               "a5_glm53flash_tp8_c16_p4k.json")


def bind(raw, packet, scale=1):
    raw = deepcopy(raw)
    arch = deepcopy(packet["compute_arch"])
    for item in arch["instruction_services"].values():
        item["per_owner_rate"] *= scale
    raw["machine"].setdefault("device", {})["compute_arch"] = arch
    raw["runtime"]["kernels"]["lowerings"]["conv_state_copy"]["options"][
        "joint_program_service"] = packet["joint_program_service"]
    return raw


def isolated(packet):
    from mindie_sim import operator_costs, target_graphs
    template = json.loads((ROOT / "examples/source-families/glm-routing-state-control.json").read_text())
    raw = bind(template, packet)
    rows, scans = [], []
    for batch in (1, 4, 8, 16, 32, 64):
        ops = target_graphs(raw, scope="body", phase="decode", tokens=4,
                            sequences=batch, context=4104)[0]
        selected = [next(op for op in ops if op.family == "conv_state_copy"
                         and dict(op.metadata)["direction"] == direction)
                    for direction in ("gather", "scatter")]
        for direction, op in zip(("gather", "scatter"), selected):
            row = operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]
            info = row["operator"]["metadata"]["program_execution"]
            observation = next(p for p in packet["observations"]
                               if p["batch"] == batch and p["direction"] == direction)
            pipes = packet["joint_program_service"]["directions"][direction]
            busy = {pipe: (p["owner_busy_ns"] + batch * 72 / 64 * p["program_busy_ns"]) / 1000
                    for pipe, p in pipes.items()}
            peak = {pipe: (p["owner_busy_ns"] + max(info["source_programs_per_owner"]) *
                          p["program_busy_ns"]) / 1000 for pipe, p in pipes.items()}
            assert not row["service_before_input"] and not row["live_tasks"]
            assert not any(row["remaining_buffers"].values())
            assert max(map(abs, row["service_delta"].values()), default=0) < 1e-5
            rows.append({"batch": batch, "direction": direction, "role": observation["split"],
                "program_count_histogram": dict(Counter(info["source_programs_per_owner"])),
                "source_programs": info["source_programs"], "metadata_loads": info["metadata_loads"],
                "read_bytes": info["payload_read_bytes"], "write_bytes": info["payload_write_bytes"],
                "observed_busy_us": observation["busy_us"], "mean_owner_joint_busy_us": busy,
                "peak_owner_joint_busy_us": peak,
                "mean_owner_busy_relative_error": {p: busy[p] / observation["busy_us"][p] - 1 for p in busy},
                "engine_lower_bound_us": row["duration_ms"] * 1000,
                "observed_whole_call_us": observation["duration_us"],
                "signed_whole_call_gap_us": row["duration_ms"] * 1000 - observation["duration_us"],
                "service_delta": row["service_delta"]})
            if batch == 16:
                controls = []
                for scale in (.5, 1, 2):
                    report = operator_costs(bind(template, packet, scale), (op,))["operators"][0]
                    controls.append({"scale": scale, "duration_us": report["duration_ms"] * 1000,
                                     "declared_service": report["declared_service"]})
                assert controls[0]["declared_service"] == controls[1]["declared_service"] == controls[2]["declared_service"]
                assert controls[0]["duration_us"] > controls[1]["duration_us"] > controls[2]["duration_us"]
                scans.append({"direction": direction, "controls": controls})
    return {"rows": rows, "supply_scans": scans}


def body(packet):
    from mindie_sim import predict
    source = json.loads(BODY.read_text())
    baseline = predict(source, scope="body", phase="decode", tokens=4, sequences=16, context=4104)
    raw = bind(source, packet)
    result = predict(raw, scope="body", phase="decode", tokens=4, sequences=16, context=4104)
    ex = result["execution"]
    assert not ex["live_tasks"] and not ex["live_programs"] and not any(ex["live_allocated"].values())
    return {"case": str(BODY.relative_to(ROOT)), "case_sha256": hashlib.sha256(BODY.read_bytes()).hexdigest(),
        "entrypoint": "predict(scope=body, phase=decode)", "layers": 45,
        "geometry": {"tokens": 4, "sequences": 16, "context": 4104},
        "duration_ms": result["duration_ms"], "existing_whole_call_responses_retained": True,
        "without_joint_busy_duration_ms": baseline["duration_ms"],
        "joint_busy_duration_delta_ms": result["duration_ms"] - baseline["duration_ms"],
        "execution": {k: ex[k] for k in ("work", "serviced", "live_tasks", "live_programs",
                      "live_allocated", "peak_live_tasks", "completed_tasks")},
        "parameters": compact_parameters(result["parameters"])}


def compact_parameters(parameters):
    return {"operator_occurrences": parameters["operator_occurrences"],
        "calibration": parameters["calibration"],
        "regime_selection": parameters.get("regime_selection", {}),
        "service_regimes": [{key: row[key] for key in
            ("id", "model_id", "queries", "response_us_min", "response_us_max")}
            for row in parameters.get("service_regimes", ())],
        "invocation_models": [{key: row[key] for key in
            ("id", "match", "startup_ms", "minimum_ms", "invocations")}
            for row in parameters.get("invocation_models", ())]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", choices=("isolated", "body"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rss-byte-factor", type=int, required=True,
                        help="Native ru_maxrss conversion: 1 on macOS, 1024 on Linux.")
    args = parser.parse_args()
    if args.child:
        started = time.perf_counter()
        packet = json.loads(PARAMETERS.read_text())
        result = isolated(packet) if args.child == "isolated" else body(packet)
        result["wall_seconds"] = time.perf_counter() - started
        result["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * args.rss_byte_factor
        assert result["peak_rss_bytes"] <= 512 * 1024 ** 2
        print(json.dumps(result))
        return
    result = {"schema": "mindie-sim.copy-joint-verification/1",
              "parameters_sha256": hashlib.sha256(PARAMETERS.read_bytes()).hexdigest(),
              "limits": {"seconds_per_child": 60, "peak_rss_bytes": 512 * 1024 ** 2},
              "rss_byte_factor": args.rss_byte_factor}
    for mode in ("isolated", "body"):
        run = subprocess.run([sys.executable, __file__, "--child", mode,
                              "--rss-byte-factor", str(args.rss_byte_factor)],
                             text=True, capture_output=True, timeout=60, check=True)
        result[mode] = json.loads(run.stdout)
        print(mode, result[mode]["wall_seconds"], result[mode]["peak_rss_bytes"], flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
