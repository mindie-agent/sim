"""Verify selected Kimi sequence shards on the ordinary frontend and Engine.

[Role] Reproduce local/global row, shared expert and complete cohort checks.
[Invariants] Preserve actual model geometry and explicit approximation choices.
  This is work/dependency acceptance, not a measured timing accuracy test.
[Timeouts & Locks] One isolated worker, 60 seconds by default; no locks.
[Side-Effect Policy] Reads committed or working-tree sources and writes only the
  requested report. No devices, model weights, remote services or automations.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import subprocess
import sys
import time


CASE_PATH = "examples/cases/kimi-k3-a5-current-checkpoint-conditional.json"


def source_hashes(root):
    return {str(path.relative_to(root)): sha256(path.read_bytes()).hexdigest()
            for path in sorted((root / "mindie_sim").rglob("*.py"))}


def make_case(root, balanced=False):
    from mindie_sim.prediction.sequence_parallel import DECLARATION
    from mindie_sim.prediction.structure import parse_case
    raw = json.loads((root / CASE_PATH).read_text())
    kernels = raw["runtime"].setdefault("kernels", {})
    kernels["sequence_parallel"] = dict(DECLARATION)
    kernels.setdefault("programs", {})["matmul"] = {"kind": "steady_matmul"}
    kernels.setdefault("families", {}).update({
        "grouped_gemm": {"kind": "steady_matmul" if balanced else "envelope"},
        "alltoall": {"kind": "pairwise_collective"},
        "allgather": {"kind": "ring_collective"},
        "reduce_scatter": {"kind": "ring_collective"},
    })
    if balanced:
        raw["runtime"]["routing"] = {"policy": "balanced"}
        kernels.setdefault("recipes", {}).update({
            "balanced_grouped_gate_up": {"kind": "steady_matmul"},
            "balanced_grouped_down": {"kind": "steady_matmul"},
        })
    return parse_case(raw)


def one(ops, predicate):
    found = [op for op in ops if predicate(op)]
    assert len(found) == 1, [(op.op_id, op.recipe) for op in found]
    return found[0]


def build(case, rows):
    from mindie_sim.prediction.cohort_traffic import bind_cohort_traffic
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.schema.workload import RequestView
    frontend = Frontend()
    graphs = {}
    for replica, count in enumerate(rows):
        views = ((RequestView(f"r{replica}", count, 0, False, "prefill"),)
                 if count else ())
        ops, _ = frontend.body(
            case.model, case.runtime, views, ready="analysis_input",
            tokens=count, mode="prefill", global_tokens=sum(rows))
        graphs[replica] = tuple(ops)
    return bind_cohort_traffic(case, graphs)


def inspect(case, graphs, rows):
    from mindie_sim.prediction.peer_traffic import PeerTraffic
    from mindie_sim.schema.common import dtype_bytes
    p, model = case.runtime.parallel, case.model
    layer = model.layer_plans[1]
    hidden, latent = model.hidden_size, layer.expert_latent_size
    width = layer.num_shared_experts * layer.moe_intermediate_size // p.tp
    byte = dtype_bytes(model.activation_dtype)
    total = sum(rows)
    expected_expert_rows = total * layer.top_k / p.ep
    evidence = []
    for replica, ops in graphs.items():
        local, full = rows[replica] // p.tp, rows[replica]
        ffn = [op for op in ops if op.layer == 1
               and dict(op.metadata).get("structure_block") == "ffn"]
        dispatch = one(ffn, lambda op: op.recipe == "expert_dispatch")
        combine = one(ffn, lambda op: op.recipe == "expert_combine")
        gate = one(ffn, lambda op: op.recipe in {
            "grouped_gate_up", "balanced_grouped_gate_up"})
        n, k = gate.local_shape[1:]
        actual_rows = gate.flops / (2 * n * k)
        assert math.isclose(actual_rows, expected_expert_rows, rel_tol=1e-12)
        forward_bytes = local * layer.top_k * latent * byte * (p.ep - 1) / p.ep
        assert dispatch.link_bytes == forward_bytes
        assert dict(dispatch.metadata)["source_assignment_partitions"] == 1
        forward = PeerTraffic.from_mapping(
            dict(dispatch.metadata)["peer_traffic"], p.ep)
        backward = PeerTraffic.from_mapping(
            dict(combine.metadata)["peer_traffic"], p.ep)
        for src in range(p.ep):
            for dst in range(p.ep):
                assert forward.payload(src, dst) == backward.payload(dst, src)
        incoming_rows = sum(forward.payload(src, replica * p.tp)
                            for src in range(p.ep)) / (latent * byte)
        local_assignments = local * layer.top_k / p.ep
        assert math.isclose(incoming_rows + local_assignments,
                            expected_expert_rows, rel_tol=1e-12)
        if full:
            router = one(ffn, lambda op: op.recipe == "router")
            assert router.local_shape[0] == local
            projections = [op for op in ffn if op.recipe == "latent_projection"]
            assert len(projections) == 2
            assert all(op.local_shape[0] == local for op in projections)
            shared = [op for op in ffn
                      if dict(op.metadata).get("ffn_category") == "shared_expert"]
            sg = one(shared, lambda op: dict(op.metadata).get("ffn_role") == "gate_up")
            sd = one(shared, lambda op: dict(op.metadata).get("ffn_role") == "down")
            assert sg.local_shape == (full, 2 * width, hidden)
            assert sd.local_shape == (full, hidden, width)
            by_id = {op.op_id: op for op in ffn}
            gather = by_id[sg.deps[0]]
            assert gather.family == "allgather"
            scatter = one(ffn, lambda op: op.family == "reduce_scatter"
                          and op.deps == (sd.op_id,))
            expected_wire = full * hidden * byte * (p.tp - 1) / p.tp
            assert gather.link_bytes == scatter.link_bytes == expected_wire
            join = one(ffn, lambda op: op.recipe == "expert_add")
            assert scatter.op_id in join.deps
            assert join.vector_ops == local * hidden
            assert not any(op.family == "allreduce" for op in ffn)
        else:
            assert not any(op.recipe in {"router", "latent_projection"} for op in ops)
            assert not any(dict(op.metadata).get("ffn_category") == "shared_expert"
                           for op in ops)
            assert not any(dict(op.metadata).get("group_axis") == "tp" for op in ops)
            assert dispatch.link_bytes == 0 and combine.link_bytes > 0
        evidence.append({
            "replica": replica, "full_rows": full, "local_rows": local,
            "expert_rows": actual_rows, "incoming_expert_rows": incoming_rows,
            "locally_routed_expert_rows": local_assignments,
            "forward_bytes": dispatch.link_bytes, "reverse_bytes": combine.link_bytes,
            "ffn_nodes": len(ffn), "body_nodes": len(ops),
        })
    physical_forward = p.tp * sum(item["forward_bytes"] for item in evidence)
    physical_reverse = p.tp * sum(item["reverse_bytes"] for item in evidence)
    expected_wire = total * layer.top_k * latent * byte * (p.ep - 1) / p.ep
    assert physical_forward == physical_reverse == expected_wire
    return {"replicas": evidence, "cohort_unique_rows": total,
            "physical_dispatch_bytes": physical_forward,
            "physical_combine_bytes": physical_reverse}


def execute_first_moe(case, graphs):
    from mindie_sim import graph_costs
    from mindie_sim.prediction.topology import Topology
    from mindie_sim.prediction.transforms import lower_implementations
    selected = {
        r: tuple(op for op in ops if op.layer == 1
                 and dict(op.metadata).get("structure_block") == "ffn")
        for r, ops in graphs.items()
    }
    external = {}
    for ops in selected.values():
        ids = {op.op_id for op in ops}
        external.update({dep: 11 for op in ops for dep in op.deps if dep not in ids})
    lowered = {r: lower_implementations(ops, case.runtime, case.model.activation_dtype)
               for r, ops in selected.items()}
    started = time.monotonic()
    report = graph_costs(case, lowered, external_inputs=external,
                        input_ready_ms=11, isolated=False, trace_limit=256)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert all(row["ready_ms"] >= 11 for row in report["timeline"])
    assert all(event["start_ms"] >= 11 for event in report["events"] if event["demand"])
    # All physical links, including every active source and the idle receiver,
    # must match the independently routed graph ledger.
    topology = Topology(case.machine, case.runtime.parallel)
    wire = {}
    for replica, ops in lowered.items():
        for op in ops:
            if not op.link_bytes:
                continue
            for name, value in topology.accounting(op, replica).wire_by_resource.items():
                wire[name] = wire.get(name, 0) + value
    actual = report["execution"]["serviced"]
    for name, amount in wire.items():
        assert math.isclose(actual[name], amount, rel_tol=1e-10, abs_tol=1e-5), (
            name, actual[name], amount)
    return {
        "boundary": "first_moe_ffn_complete_DP_cohort_actual_geometry",
        "wall_seconds": time.monotonic() - started,
        "duration_ms": report["duration_ms"],
        "operators": len(report["timeline"]),
        "completed_tasks": report["execution"].get("completed_tasks"),
        "peak_live_tasks": report["execution"].get("peak_live_tasks"),
        "live_tasks": report["execution"]["live_tasks"],
        "live_programs": report["execution"]["live_programs"],
        "physical_wire_by_resource": wire,
    }


def verify(root):
    from tools.model_convergence import worker_usage

    before = source_hashes(root)
    began = time.monotonic()
    case = make_case(root)
    graphs = build(case, (8192, 16384, 8192, 0))
    report = {
        "status": "completed",
        "boundary": "mechanism_conservation_not_measured_accuracy",
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5).strip(),
        "source_package_sha256": sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
        "case_source": {"path": CASE_PATH,
                        "sha256": sha256((root / CASE_PATH).read_bytes()).hexdigest()},
        "dirty": subprocess.check_output(
            ["git", "status", "--short"], cwd=root, text=True, timeout=5),
        "source_assumptions": {
            "shared_expert_layout": "tp_sharded_explicit_not_inferred_capture_dispatch",
            "matrix_recipe": "steady_matmul_dense_and_equal_groups;explicit_envelope_probability_aggregate",
            "full_geometry_scenario": "probability_moments_not_integer_expert_realization",
            "balanced_scenario": "separate32256token_cohort_equal576rows_per_expert",
        },
        "unequal": inspect(case, graphs, (8192, 16384, 8192, 0)),
        "first_moe_execution": execute_first_moe(case, graphs),
    }
    balanced = make_case(root, balanced=True)
    balanced_graphs = build(balanced, (8064, 16128, 8064, 0))
    report["balanced"] = inspect(balanced, balanced_graphs, (8064, 16128, 8064, 0))
    report["balanced_first_moe_execution"] = execute_first_moe(balanced, balanced_graphs)
    report["wall_seconds"] = time.monotonic() - began
    report["worker_usage"] = worker_usage()
    report["source_unchanged"] = before == source_hashes(root)
    if not report["source_unchanged"]:
        report.update(status="failed", reason="runtime_changed_during_acceptance")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.worker:
        sys.path.insert(0, str(root))
        report = verify(root)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "completed" else 1
    if args.timeout <= 0 or not math.isfinite(args.timeout):
        parser.error("--timeout must be positive and finite")
    command = [sys.executable, "-I", str(Path(__file__).resolve()),
               "--root", str(root), "--worker"]
    started = time.monotonic()
    try:
        result = subprocess.run(command, text=True, capture_output=True,
                                timeout=args.timeout, check=False)
    except subprocess.TimeoutExpired:
        report = {"status": "failed", "reason": "worker_timeout",
                  "budget_seconds": args.timeout,
                  "wall_seconds": time.monotonic() - started}
        code = 2
    else:
        if result.returncode:
            report = {"status": "failed", "reason": "worker_failed",
                      "returncode": result.returncode,
                      "stderr": result.stderr[-16000:],
                      "stdout": result.stdout[-16000:]}
            code = result.returncode
        else:
            report, code = json.loads(result.stdout), 0
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
