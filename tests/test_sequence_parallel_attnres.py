"""Combined source-selected SP and AttnRes on real Kimi graph boundaries."""
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import time

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.prediction.cohort_traffic import bind_cohort_traffic
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.peer_traffic import PeerTraffic
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.prediction.transforms import lower_implementations
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import RequestView


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = "examples/cases/kimi-k3-a5-sp-attnres-conditional.json"


def combined_case(*, layers=None, bounded=False):
    raw = json.loads((ROOT / CASE_PATH).read_text())
    if bounded:
        kernels = raw["runtime"]["kernels"]
        kernels.setdefault("programs", {})["matmul"] = {"kind": "steady_matmul"}
        kernels.setdefault("families", {}).update({
            "grouped_gemm": {"kind": "envelope"},
            "alltoall": {"kind": "pairwise_collective"},
            "allgather": {"kind": "ring_collective"},
            "reduce_scatter": {"kind": "ring_collective"},
        })
    case = parse_case(raw)
    if layers is not None:
        case = replace(
            case,
            model=replace(case.model, layers=layers, body_layers=layers,
                          layer_plans=case.model.layer_plans[:layers]),
            runtime=replace(case.runtime, pp_layer_partition=(layers,)),
        )
    return case


def one(ops, predicate):
    found = [op for op in ops if predicate(op)]
    assert len(found) == 1, [op.op_id for op in found]
    return found[0]


def unequal_graphs(case, rows):
    graphs = {}
    for replica, count in enumerate(rows):
        views = ((RequestView(f"r{replica}", count, 0, False, "prefill"),)
                 if count else ())
        ops, _ = Frontend().body(
            case.model, case.runtime, views, ready="analysis_input",
            tokens=count, mode="prefill", global_tokens=sum(rows))
        graphs[replica] = lower_implementations(
            ops, case.runtime, case.model.activation_dtype, local_tokens=count)
    return bind_cohort_traffic(case, graphs)


def check_structure(case, graphs, rows, *, phase="prefill"):
    """Recompute local, global and unique expert rows from the declaration."""
    parallel, model = case.runtime.parallel, case.model
    width = model.hidden_size
    expected_expert_rows = sum(rows) * model.layer_plans[1].top_k / parallel.ep
    result = []
    for replica, ops in graphs.items():
        local = rows[replica] // parallel.tp
        points = [op for op in ops if op.recipe == "attn_res_fwd"]
        if not local:
            assert not points
            assert not any(dict(op.metadata).get("group_axis") == "tp" for op in ops)
            assert not any(op.recipe in {"latent_projection", "router"} for op in ops)
            assert not any(op.op_id in {"sp_shard", "sp_output_gather"} for op in ops)
        else:
            assert len(points) == 2 * model.layers + 1
            assert all(op.local_shape == (local, width) for op in points)
            assert all(dict(op.metadata)["attn_res_view"]["tokens_local"] == local
                       for op in points)
            assert all(dict(op.metadata)["attn_res_view"]["optimize_prefill"]
                       == (phase == "prefill") for op in points)
            by_id = {op.op_id: op for op in ops}
            assert by_id["sp_shard"].deps == ("analysis_input",)
            assert points[0].deps == ("sp_shard",)
            assert by_id["sp_output_gather"].deps == ("final.attn_res_fwd",)
            assert by_id["sp_output_gather"].local_shape == (rows[replica], width)
            # Every residual point supplies the local input to its global
            # attention gather; the attention projection retains full rows.
            for layer in model.layer_plans:
                prefix = f"s0.l{layer.index}"
                gather = by_id[prefix + ".attention.sp_gather"]
                assert gather.deps == (prefix + ".attention.attn_res_fwd",)
                assert gather.local_shape == (rows[replica], width)
                projections = [op for op in ops if op.layer == layer.index
                               and dict(op.metadata).get("structure_block") == "attention"
                               and op.recipe in {"linear_q", "q_a"}]
                assert projections
                assert all(op.local_shape[0] == rows[replica] for op in projections)
                residual = by_id[prefix + ".ffn.attn_res_fwd"]
                scatter = one(ops, lambda op: op.layer == layer.index
                              and dict(op.metadata).get("structure_block") == "attention"
                              and op.family == "reduce_scatter")
                assert scatter.op_id in residual.deps
            latent = [op for op in ops if op.recipe == "latent_projection"]
            assert latent and all(op.local_shape[0] == local for op in latent)
            assert all(dict(op.metadata)["matrix_wrapper"]["group_list_values"] == (local,)
                       for op in latent)
        gates = [op for op in ops if op.recipe == "grouped_gate_up"]
        assert len(gates) == sum(layer.ffn == "moe" for layer in model.layer_plans)
        for gate in gates:
            _, n, k = gate.local_shape
            assert gate.flops / (2 * n * k) == pytest.approx(expected_expert_rows)
        dispatch = one(ops, lambda op: op.layer == 1 and op.recipe == "expert_dispatch")
        combine = one(ops, lambda op: op.layer == 1 and op.recipe == "expert_combine")
        assert dict(dispatch.metadata)["source_assignment_partitions"] == 1
        forward = PeerTraffic.from_mapping(dict(dispatch.metadata)["peer_traffic"], parallel.ep)
        backward = PeerTraffic.from_mapping(dict(combine.metadata)["peer_traffic"], parallel.ep)
        for source in range(parallel.ep):
            for destination in range(parallel.ep):
                assert forward.payload(source, destination) == backward.payload(destination, source)
        if not local:
            assert dispatch.link_bytes == 0 < combine.link_bytes
            assert expected_expert_rows > 0
        result.append({
            "replica": replica, "global_rows": rows[replica], "local_rows": local,
            "attnres_points": len(points), "expert_rows_per_layer": expected_expert_rows,
            "operators": len(ops), "dispatch_bytes": dispatch.link_bytes,
            "combine_bytes": combine.link_bytes,
        })
    return result


def execute(case, graphs):
    began = time.monotonic()
    report = graph_costs(case, graphs, external_inputs={"analysis_input": 13},
                        input_ready_ms=7, isolated=False, trace_limit=512)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert all(abs(amount) < 1e-6 for amount in report["execution"]["live_allocated"].values())
    assert all(row["ready_ms"] >= 13 for row in report["timeline"])
    assert all(event["start_ms"] >= 13 for event in report["events"] if event["demand"])
    timeline = {row["id"]: row for row in report["timeline"]}
    for replica, ops in graphs.items():
        for op in ops:
            row = timeline[f"r{replica}:{op.op_id}"]
            for dep in op.deps:
                if dep != "analysis_input":
                    previous = timeline[f"r{replica}:{dep}"]
                    assert row["ready_ms"] >= previous["complete_ms"] - 1e-10
    topology, wire = Topology(case.machine, case.runtime.parallel), {}
    for replica, ops in graphs.items():
        for op in ops:
            if op.link_bytes:
                for key, amount in topology.accounting(op, replica).wire_by_resource.items():
                    wire[key] = wire.get(key, 0) + amount
    actual = report["execution"]["serviced"]
    for key, amount in wire.items():
        assert math.isclose(actual[key], amount, rel_tol=1e-10, abs_tol=1e-5)
    return {
        "wall_seconds": time.monotonic() - began,
        "duration_ms": report["duration_ms"], "operators": len(report["timeline"]),
        "completed_tasks": report["execution"]["completed_tasks"],
        "peak_live_tasks": report["execution"]["peak_live_tasks"],
        "live_tasks": report["execution"]["live_tasks"],
        "live_programs": report["execution"]["live_programs"],
        "remaining_buffers": report["execution"]["live_allocated"],
        "serviced_wire_by_resource": wire,
        "external_input_ready_ms": 13,
    }


def test_full_93_layer_public_cohort_keeps_local_attnres_global_attention_and_experts():
    case = combined_case()
    assert case.model.layers == 93
    graphs = target_graphs(case, phase="prefill", tokens=8192, scope="body")
    evidence = check_structure(case, graphs, (8192,) * 4)
    assert {row["attnres_points"] for row in evidence} == {187}
    assert {row["local_rows"] for row in evidence} == {1024}
    assert {row["expert_rows_per_layer"] for row in evidence} == {16384}


def test_full_unequal_cohort_idle_replica_keeps_experts_without_empty_attnres_points():
    case = combined_case()
    rows = (8192, 16384, 8192, 0)
    evidence = check_structure(case, unequal_graphs(case, rows), rows)
    assert [row["attnres_points"] for row in evidence] == [187, 187, 187, 0]
    assert [row["local_rows"] for row in evidence] == [1024, 2048, 1024, 0]
    assert {row["expert_rows_per_layer"] for row in evidence} == {16384}


@pytest.mark.parametrize("unequal", [False, True])
def test_actual_complete_small_graph_preserves_bank_shard_and_transport_dependencies(unequal):
    case = combined_case(layers=2, bounded=True)
    rows = (8, 16, 8, 0) if unequal else (8,) * 4
    graphs = (unequal_graphs(case, rows) if unequal else
              target_graphs(case, phase="prefill", tokens=8, scope="body"))
    check_structure(case, graphs, rows)
    evidence = execute(case, graphs)
    assert evidence["completed_tasks"] > evidence["operators"]


def test_decode_keeps_sp_branch_but_does_not_claim_prefill_cache():
    case = combined_case(layers=2)
    graphs = target_graphs(case, phase="decode", tokens=1, sequences=8, context=64, scope="body")
    evidence = check_structure(case, graphs, (8,) * 4, phase="decode")
    assert {row["local_rows"] for row in evidence} == {1}


def test_idle_exception_does_not_hide_nonidle_unmatched_source_selection():
    case = combined_case(layers=2)
    ops, _ = Frontend().body(case.model, case.runtime, (), ready="analysis_input",
                            tokens=0, global_tokens=32, mode="prefill")
    original = deepcopy(ops)
    lower_implementations(ops, case.runtime, case.model.activation_dtype, local_tokens=0)
    assert ops == original
    with pytest.raises(ModelingError, match="matrix_wrapper_projection_recipes_unmatched"):
        lower_implementations(ops, case.runtime, case.model.activation_dtype, local_tokens=8)
