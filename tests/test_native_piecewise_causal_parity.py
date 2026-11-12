"""Native body DAG vs piecewise causal body on identical declarations.

[Role]
  Run the two maintained whole-body execution paths on the same ModelSpec,
  Hardware, parallel axes and per-request geometry, then compare the declared
  native units, their logical work, their stage/round footprint and their
  elapsed time.  The native path is ``prediction.body.body_forward`` (frontend
  OpSpec DAG -> shared execution engine).  The piecewise path is
  ``piecewise.model_pipeline.assemble_model(...).forward`` (compiled
  subgraphs -> contention composition).

[Invariants]
  * A unit required by the model structure appears with the same multiplicity
    in both paths (per replica, body scope only).
  * Logical work is conserved independently of tile padding: routed expert
    FLOPs use the expected rows ``tokens * top_k / ep``; a routed transport
    carries ``rows * row_bytes``; a TP allreduce carries the declared message.
  * Execution time follows the declared topology: a 16-rank TP group whose
    hardware declares 16 ranks per node has 30 intra-node ring rounds and 0
    inter-node rounds; each declared stage costs at least the larger of the
    declared link latency and the declared execution-class minimum.
  * One declared matrix-kernel completion floor is charged once per kernel,
    not once per row or tile.
  * Opaque case/cell/workload labels and inherited observation payloads do not
    change either path.
  * The A5 Qwen body is the large-family check; the two paths still expose a
    material difference there, so a structural pass is not inferred from one
    family.

[Timeouts & Locks]
  Pure CPU arithmetic.  The two A3/Qwen probes perform a finite number of
  model walks; no device, subprocess, network or lock is used.

[Side-Effect Policy]
  Read-only repository fixtures and in-memory records.  No filesystem writes.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import json
import math
from pathlib import Path

import pytest

from mindie_sim.prediction.body import body_forward
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.piecewise.compiler import PipeSupply
from mindie_sim.prediction.piecewise.geometry import RequestGeometry, WorkloadGeometry
from mindie_sim.prediction.piecewise.model_pipeline import assemble_model
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import FeatureConfig
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.hardware import Link
from mindie_sim.schema.invocation import InvocationTiming

ROOT = Path(__file__).resolve().parents[1]
A3_CASE = (ROOT / "examples/cases/high-concurrency-20261030"
           / "glm-kda-accepted-conditional/glm53flash-a3-b4-q6.json")
QWEN_CASE = ROOT / "examples/cases/qwen38-a5.json"
A5_MACHINE = ROOT / "examples/cases/glm53-a5.json"
PARITY = 0.10
_SERVICE_KEYS = ("service_regimes", "invocations", "primitives")


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _unobserved(raw: dict) -> dict:
    """Drop inherited measured devices before either path sees the case.

    The kept records are structural: model/layer plans, quant representation,
    machine rates/links, parallel axes and graph buckets.  Observed response
    tables and invocation minima are held out of both paths here.
    """
    payload = deepcopy(raw)
    runtime = payload.setdefault("runtime", {})
    kernels = runtime.get("kernels") or {}
    for key in _SERVICE_KEYS:
        kernels.pop(key, None)
    if kernels:
        runtime["kernels"] = kernels
    elif "kernels" in runtime:
        runtime.pop("kernels")
    return payload


def _a3_definition() -> dict:
    raw = _unobserved(_load(A3_CASE))
    raw["runtime"].update({"tp": 16, "dp": 1, "ep": 16, "pp": 1, "graph_mode": "none"})
    return raw


def _qwen_definition() -> dict:
    raw = _unobserved(_load(QWEN_CASE))
    # The 731 qwen38 cell runs on an A5 2-node rack; the checked-in single-die
    # profile has no inter-die routes.  Use the maintained 16-die A5 profile so
    # both paths resolve the same declared topology.
    raw["machine"] = deepcopy(_load(A5_MACHINE)["machine"])
    raw["runtime"].update({"tp": 2, "dp": 4, "ep": 1, "pp": 1, "graph_mode": "none"})
    return raw


def _case(raw: dict):
    return parse_case(raw)


@lru_cache(maxsize=4)
def _a3_result():
    case = _case(_a3_definition())
    return _native_and_piece(case, tokens=6, context=4096, sequences=4,
                             tp=16, dp=1, ep=16, pp=1)


@lru_cache(maxsize=4)
def _qwen_result():
    case = _case(_qwen_definition())
    return _native_and_piece(case, tokens=1, context=1024, sequences=1,
                             tp=2, dp=4, ep=1, pp=1)


def _native_and_piece(case, *, tokens: int, context: int, sequences: int,
                      tp: int, dp: int, ep: int, pp: int):
    native = body_forward(case, phase="decode", tokens=tokens, context=context,
                          sequences=sequences)
    supply = PipeSupply.from_hardware(case.machine)
    features = FeatureConfig.from_runtime(case.runtime, case.model)
    assert (features.tp, features.dp, features.ep, features.pp) == (tp, dp, ep, pp)
    pipeline = assemble_model(case.model, supply=supply, features=features)
    geometry = WorkloadGeometry(
        tuple(RequestGeometry(tokens, context, phase="decode") for _ in range(sequences)),
        phase="decode")
    prediction = pipeline.forward(geometry)
    body_rows = tuple(row for layer in prediction.layers for row in layer.rows)
    body_ms = float(sum(layer.duration_ms for layer in prediction.layers))
    return {
        "case": case,
        "native": native,
        "pipeline": pipeline,
        "prediction": prediction,
        "body_rows": body_rows,
        "body_ms": body_ms,
        "replicas": len(native["replicas"]),
        "tokens": tokens * sequences,
    }


def _native_timeline(result) -> list[dict]:
    return list(result["native"]["timeline"])


def _native_recipe_totals(result) -> dict[str, tuple[int, float]]:
    totals: dict[str, list] = defaultdict(lambda: [0, 0.0])
    replicas = max(1, result["replicas"])
    for event in _native_timeline(result):
        totals[event["recipe"]][0] += 1
        totals[event["recipe"]][1] += float(event["elapsed_ms"])
    return {name: (count // replicas, value / replicas)
            for name, (count, value) in totals.items()}


def _piece_name_totals(result) -> dict[str, tuple[int, float]]:
    totals: dict[str, list] = defaultdict(lambda: [0, 0.0])
    for row in result["body_rows"]:
        totals[row.cost.name][0] += 1
        totals[row.cost.name][1] += float(row.cost.duration_ms)
    return {name: (count, value) for name, (count, value) in totals.items()}


def _required_counts(case) -> dict[str, int]:
    plans = case.model.layer_plans
    layers = len(plans)
    moe = sum(1 for plan in plans if plan.ffn == "moe")
    return {
        "hc_projection": 2 * layers,
        "dispatch": moe,
        "combine": moe,
        "expert_gate_up": moe,
        "expert_down": moe,
        "tp_allreduce": 2 * layers,
    }


def _native_unit_counts(result) -> dict[str, int]:
    recipes = {
        "hc_projection": ("hc_projection",),
        "dispatch": ("expert_dispatch",),
        "combine": ("expert_combine",),
        "expert_gate_up": ("grouped_gate_up",),
        "expert_down": ("grouped_down",),
        "tp_allreduce": ("ring_allreduce", "linear_reduce", "output_reduce"),
    }
    totals = _native_recipe_totals(result)
    return {unit: sum(totals.get(name, (0, 0.0))[0] for name in names)
            for unit, names in recipes.items()}


def _piece_unit_counts(result) -> dict[str, int]:
    names = {
        "hc_projection": ("hc_projection",),
        "dispatch": ("dispatch",),
        "combine": ("combine",),
        "expert_gate_up": ("expert_gate_up",),
        "expert_down": ("expert_down",),
        "tp_allreduce": ("tp_allreduce",),
    }
    totals = _piece_name_totals(result)
    return {unit: sum(totals.get(name, (0, 0.0))[0] for name in candidates)
            for unit, candidates in names.items()}


def _native_specs(case, *, tokens: int, context: int, sequences: int):
    return tuple(target_operators(case, phase="decode", tokens=tokens,
                                  context=context, sequences=sequences))


def _derived(cost, key: str):
    return dict(cost.derived).get(key)


def _piece_row(result, *, layer: int, name: str, kind: str | None = None):
    for row in result["body_rows"]:
        if row.layer == layer and row.cost.name == name and (kind is None or row.kind == kind):
            return row
    raise AssertionError(f"piecewise_row_missing:{layer}:{name}:{kind}")


def _native_event(result, op_suffix: str, *, replica: int = 0):
    for event in _native_timeline(result):
        if event["replica"] == replica and event["op_id"].endswith(op_suffix):
            return event
    raise AssertionError(f"native_event_missing:{op_suffix}")


def _link_latency_ms(case) -> float:
    rates = dict(case.machine.dies[0].rates)
    for name in ("link_latency_ms", "link_latency_intra_ms"):
        if name in rates:
            return float(rates[name])
    # The maintained hardware records expose link latency on the supply, not
    # always as a device rate.
    return float(dict(PipeSupply.from_hardware(case.machine).invocations).get(
        "link_latency_ms", 0.0))


def _link_rate(case) -> float:
    return float(PipeSupply.from_hardware(case.machine).rate("link_hccs"))


def _hc_probe(case, *, floor_ms=None, mte_scale=1.0):
    """Re-run both paths with a declared completion floor or MTE2 rate change."""
    def die(item):
        invocations = dict(item.kernel_invocations)
        if floor_ms is not None:
            invocations["cube"] = InvocationTiming(
                startup_ms=0.0,
                minimum_ms=floor_ms,
                provenance={"basis": "independent_perturbation"},
            )
        rates = tuple(
            (name, value * mte_scale if name == "mte2_aic" else value)
            for name, value in item.rates
        )
        return replace(item, rates=rates,
                       kernel_invocations=tuple(invocations.items()))

    changed = replace(
        case, machine=replace(
            case.machine, dies=tuple(die(item) for item in case.machine.dies)))
    result = _native_and_piece(changed, tokens=6, context=4096, sequences=4,
                               tp=16, dp=1, ep=16, pp=1)
    native_ms = float(_native_event(result, ".attention.hc_project")["elapsed_ms"])
    piece_ms = float(_piece_row(
        result, layer=0, name="hc_projection", kind="R0").cost.duration_ms)
    return native_ms, piece_ms


def test_required_native_units_have_identical_multiplicity_on_both_paths():
    for result in (_a3_result(),):
        required = _required_counts(result["case"])
        assert _native_unit_counts(result) == required
        assert _piece_unit_counts(result) == required


def test_routed_expert_work_uses_expected_rows_not_active_tokens():
    result = _a3_result()
    case, tokens = result["case"], result["tokens"]
    plans = case.model.layer_plans
    top_k = int(case.model.top_k)
    ep = int(case.runtime.parallel.ep)
    hidden = int(case.model.hidden_size)
    plans = [plan for plan in plans if plan.ffn == "moe"]
    assert plans
    width = int(plans[0].moe_intermediate_size)
    rows = tokens * top_k / ep
    expected_flops = 2 * rows * 2 * width * hidden
    assert expected_flops == pytest.approx(402_653_184.0)

    native = [op for op in _native_specs(case, tokens=6, context=4096, sequences=4)
              if op.recipe == "grouped_gate_up"]
    assert native
    for op in native:
        assert op.flops == pytest.approx(expected_flops)
        assert dict(op.metadata)["expected_rows"] == pytest.approx(rows)

    for row in (item for item in result["body_rows"] if item.cost.name == "expert_gate_up"):
        assert _derived(row.cost, "flops") == pytest.approx(expected_flops)
        assert float(_derived(row.cost, "moe_rows")) == pytest.approx(rows)


@pytest.mark.parametrize("per_seq_tokens,ep", [(6, 16), (1, 16), (6, 8)])
def test_routed_dispatch_carries_the_declared_routed_rows(per_seq_tokens, ep):
    raw = _a3_definition()
    raw["runtime"].update({"tp": ep, "ep": ep})
    case = _case(raw)
    result = _native_and_piece(case, tokens=per_seq_tokens, context=4096, sequences=4,
                               tp=ep, dp=1, ep=ep, pp=1)
    tokens = result["tokens"]
    top_k = int(case.model.top_k)
    hidden = int(case.model.hidden_size)
    act_bytes = 2  # declared bf16 body activation

    # Each source rank owns tokens*top_k/ep routed rows.  A directed
    # all-to-all sends one row payload to each of the other ep-1 ranks; the
    # source's own share is not part of its send.  Keep the per-source and
    # group byte ledgers separate: OpSpec.link_bytes is per-source.
    local_rows = tokens * top_k / ep
    row_bytes = hidden * act_bytes
    local_payload = local_rows * row_bytes
    per_source_useful = local_payload * (ep - 1) / ep
    group_useful = per_source_useful * ep

    # Framing is per remote destination, from one source's share of its own
    # payload.  The source itself is not a destination.
    mtu = 4096
    remote_useful = local_payload / ep
    remote_wire = math.ceil(remote_useful / mtu) * mtu
    per_source_wire = (ep - 1) * remote_wire
    group_wire = per_source_wire * ep

    dispatch = _piece_row(result, layer=3, name="dispatch", kind="F2")
    assert int(_derived(dispatch.cost, "routed_rows")) == tokens * top_k
    assert float(_derived(dispatch.cost, "transport_row_bytes")) == row_bytes
    assert float(_derived(dispatch.cost, "peer_bytes")) == tokens * top_k * row_bytes

    native_dispatch = [op for op in _native_specs(case, tokens=per_seq_tokens,
                                                  context=4096, sequences=4)
                       if op.recipe == "expert_dispatch"]
    assert native_dispatch
    topology = Topology(case.machine, case.runtime.parallel)
    for op in native_dispatch:
        metadata = dict(op.metadata)
        assert float(metadata["steps"]) == ep - 1
        assert per_source_useful * 0.90 <= op.link_bytes <= per_source_useful * 1.10

        source_stage = int(metadata.get("source_stage", op.stage))
        flows = list(topology.messages(op, 0, op.link_bytes))
        by_source = defaultdict(float)
        source_flows = defaultdict(int)
        for flow, rounds in flows:
            assert rounds == ep - 1
            by_source[flow.source] += flow.useful_bytes
            source_flows[flow.source] += 1
        assert len(by_source) == ep
        assert set(by_source) == set(topology.ranks(0, source_stage))
        assert set(source_flows.values()) == {ep - 1}
        assert len(flows) == ep * (ep - 1)
        for source_bytes in by_source.values():
            assert per_source_useful * 0.90 <= source_bytes <= per_source_useful * 1.10
        assert group_useful * 0.90 <= sum(by_source.values()) <= group_useful * 1.10

    # `send_bytes` is the per-source framed ledger and `aggregate_wire_bytes`
    # is the group ledger.  Both must match the independently derived framed
    # bytes, not a whole-group payload copied into each source.
    piece_send = float(_derived(dispatch.cost, "send_bytes"))
    piece_group = float(_derived(dispatch.cost, "aggregate_wire_bytes"))
    problems = []
    if not per_source_wire * 0.90 <= piece_send <= per_source_wire * 1.10:
        problems.append(f"send_bytes={piece_send} expected={per_source_wire}")
    if not group_wire * 0.90 <= piece_group <= group_wire * 1.10:
        problems.append(
            f"aggregate_wire_bytes={piece_group} expected={group_wire}")
    assert not problems, f"tokens={per_seq_tokens} ep={ep}: " + "; ".join(problems)


def _tp_independent_ms(case) -> tuple[float, float, float]:
    ep = int(case.runtime.parallel.ep)
    ranks_per_node = int(case.machine.ranks_per_node)
    nodes = max(1, (ep + ranks_per_node - 1) // ranks_per_node)
    # All 16 TP ranks live on one node: one intra-node ring with R-1
    # reduce-scatter rounds and R-1 allgather rounds, no inter-node round.
    intra = 2 * (min(ep, ranks_per_node) - 1)
    inter = 2 * (nodes - 1) if nodes > 1 else 0
    rounds = intra + inter
    message = 24 * 4096 * 2
    ring_bytes = 2 * (ep - 1) / ep * message
    stage_min = float(dict(PipeSupply.from_hardware(case.machine).invocations)["vector"])
    per_round = max(0.0008, stage_min)
    independent = rounds * per_round + ring_bytes / 196_000_000.0
    return independent, ring_bytes, float(rounds)


def test_native_tp_allreduce_uses_declared_stage_rounds_and_minimum():
    result = _a3_result()
    independent, ring_bytes, rounds = _tp_independent_ms(result["case"])
    assert rounds == 30.0
    # 30 declared dependent rounds, each at least max(link latency 0.8 us,
    # declared vector execution-class minimum 2.5 us), plus one link body.
    assert independent == pytest.approx(0.075 + 368_640 / 196_000_000.0, abs=1e-12)
    event = _native_event(result, ".shared.tp")
    assert float(event["elapsed_ms"]) == pytest.approx(independent, rel=PARITY)


def test_piecewise_tp_allreduce_uses_declared_ranks_per_node():
    result = _a3_result()
    independent, ring_bytes, rounds = _tp_independent_ms(result["case"])
    row = _piece_row(result, layer=3, name="tp_allreduce", kind="F4")
    derived = dict(row.cost.derived)
    assert float(derived["link_bytes"]) == pytest.approx(ring_bytes)
    problems = []
    if float(derived["nodes"]) != 1.0:
        problems.append(f"nodes={derived['nodes']} but hardware ranks_per_node=16")
    elapsed = float(row.cost.duration_ms)
    if abs(elapsed - independent) > PARITY * independent:
        problems.append(f"elapsed={elapsed:.8f} independent={independent:.8f}")
    assert not problems, "; ".join(problems)


def test_unused_inter_node_latency_does_not_change_single_node_dispatch():
    base = _a3_result()
    case = base["case"]
    assert int(case.machine.ranks_per_node) == 16
    # All 16 ranks are declared on one node.  A copy of this declaration with
    # an additional inter-node link at 80 us must leave the intra-node
    # dispatch unchanged; both paths currently route the unused link.
    inter = Link(name="d2d_inter_node_probe", capacity_bytes_per_ms=196_000.0,
                 members=tuple(range(16)), latency_us=80.0,
                 channel="d2d_inter_node")
    changed_case = replace(
        case, machine=replace(case.machine, links=case.machine.links + (inter,)))
    changed = _native_and_piece(changed_case, tokens=6, context=4096, sequences=4,
                                tp=16, dp=1, ep=16, pp=1)
    base_native = float(_native_event(base, ".dispatch")["elapsed_ms"])
    changed_native = float(_native_event(changed, ".dispatch")["elapsed_ms"])
    base_piece = float(_piece_row(base, layer=3, name="dispatch", kind="F2").cost.duration_ms)
    changed_piece = float(_piece_row(changed, layer=3, name="dispatch", kind="F2").cost.duration_ms)
    problems = []
    if abs(changed_native - base_native) > PARITY * base_native:
        problems.append(f"native {base_native:.9f}->{changed_native:.9f}")
    if abs(changed_piece - base_piece) > PARITY * base_piece:
        problems.append(f"piece {base_piece:.9f}->{changed_piece:.9f}")
    assert not problems, "; ".join(problems)


def test_piecewise_routed_expert_elapsed_tracks_native_grouped_units():
    result = _a3_result()
    problems = []
    for native_recipe, piece_name in (("grouped_gate_up", "expert_gate_up"),
                                      ("grouped_down", "expert_down")):
        native = _native_recipe_totals(result).get(native_recipe, (0, 0.0))
        piece = _piece_name_totals(result).get(piece_name, (0, 0.0))
        assert native[0] == piece[0] == 42
        if abs(piece[1] - native[1]) > PARITY * native[1]:
            problems.append(f"{piece_name} {native[1]:.6f}->{piece[1]:.6f}")
    assert not problems, "; ".join(problems)


def test_piecewise_mla_attention_tracks_the_native_attention_unit():
    result = _a3_result()
    native = _native_recipe_totals(result).get("latent_attention_batch", (0, 0.0))
    piece = _piece_name_totals(result).get("mla_attention", (0, 0.0))
    assert native[0] == piece[0] == 11
    assert piece[1] == pytest.approx(native[1], rel=PARITY)


def test_piecewise_router_gemm_tracks_the_native_router_unit():
    result = _a3_result()
    native = _native_recipe_totals(result).get("router", (0, 0.0))
    piece = _piece_name_totals(result).get("router_gemm", (0, 0.0))
    assert native[0] == piece[0] == 42
    assert piece[1] == pytest.approx(native[1], rel=PARITY)


def test_piecewise_routed_expert_pipe_work_matches_expected_rows():
    result = _a3_result()
    logical = 402_653_184.0
    for name in ("expert_gate_up", "expert_down"):
        rows = [row for row in result["body_rows"] if row.cost.name == name]
        assert rows
        for row in rows:
            expected = logical if name == "expert_gate_up" else logical / 2
            assert _derived(row.cost, "flops") == pytest.approx(expected)
            issued = float(dict(row.cost.work)["cube"])
            assert issued == pytest.approx(_derived(row.cost, "flops"))


def test_piecewise_hc_pipe_work_uses_the_issued_rows_not_a_padded_bucket():
    result = _a3_result()
    logical = 18_874_368.0
    for row in (item for item in result["body_rows"] if item.cost.name == "hc_projection"):
        derived = dict(row.cost.derived)
        assert _derived(row.cost, "flops") == pytest.approx(logical)
        fractal_m = float(derived["fractal_m"])
        fractal_n = float(derived["fractal_n"])
        tail_rows = float(derived["tail_rows"])
        issued_rows = ((tail_rows + fractal_m - 1) // fractal_m) * fractal_m
        # The issued product may take the next declared 32x32 fractal tile, but
        # a 24-row tail cannot become a 128-row product.  The old 100,663,296
        # (128 rows) was outside this declared tensor-instruction bound; Q's
        # 33,554,432 (32 rows x 32 columns) is the physical tile.
        upper = 2.0 * issued_rows * fractal_n * 16384
        issued = float(dict(row.cost.work)["cube"])
        assert logical <= issued <= upper


def test_one_kernel_floor_is_charged_once_per_hc_projection():
    result = _a3_result()
    case = result["case"]
    rows = _required_counts(case)["hc_projection"]
    assert rows == 90
    logical = 2 * 24 * 24 * (4 * 4096)
    assert logical == 18_874_368

    native_ops = [item for item in _native_specs(
        case, tokens=6, context=4096, sequences=4)
        if item.recipe == "hc_projection"]
    assert native_ops
    for op in native_ops:
        assert op.flops == pytest.approx(logical)
        assert op.local_shape == (24, 24, 16384)
        assert op.dtype == "fp32"
    for row in (item for item in result["body_rows"] if item.cost.name == "hc_projection"):
        assert _derived(row.cost, "flops") == pytest.approx(logical)

    # One of the 20 AIC owners copies the 24x24x16384 FP32 operands:
    # (M + N) * K * 4 bytes.  That copy, not a 5 us empty-launch floor, is the
    # real lower bound for this ordinary HC projection.
    die = case.machine.dies[0]
    owners = int(die.units("mte2_aic"))
    assert owners == 20
    total_mte = float(dict(die.rates)["mte2_aic"])
    assert total_mte == pytest.approx(1.056e12)
    per_owner = total_mte / owners
    assert per_owner == pytest.approx(52.8e9)
    m, n, k = native_ops[0].local_shape
    operand_bytes = (m + n) * k * 4
    assert operand_bytes == 3_145_728
    copy_lower_ms = operand_bytes / per_owner * 1000.0
    assert copy_lower_ms == pytest.approx(0.05957818181818182, rel=1e-12)

    one = _native_event(result, ".attention.hc_project")
    piece = _piece_row(result, layer=0, name="hc_projection", kind="R0")
    native_ms = float(one["elapsed_ms"])
    piece_ms = float(piece.cost.duration_ms)

    def within_copy_bound(value):
        return copy_lower_ms * (1 - PARITY) <= value <= copy_lower_ms * (1 + PARITY)

    assert native_ms >= copy_lower_ms and piece_ms >= copy_lower_ms
    assert within_copy_bound(native_ms)
    assert within_copy_bound(piece_ms)

    # A missing physical copy (0.005 ms) and a per-owner or per-row floor
    # multiplication are rejected by the same copy-bound tolerance.
    assert not within_copy_bound(0.005)
    assert not within_copy_bound(owners * 0.005)
    assert not within_copy_bound(rows * 0.005)

    # Floors below the copy bound do not change the span; 100 us dominates
    # once and is not incremented by an extra 5 us or multiplied.
    for floor in (0.0, 0.010):
        floor_native, floor_piece = _hc_probe(case, floor_ms=floor)
        assert floor_native == pytest.approx(native_ms, rel=1e-9)
        assert floor_piece == pytest.approx(piece_ms, rel=1e-9)
    floor_native, floor_piece = _hc_probe(case, floor_ms=0.100)
    assert floor_native == pytest.approx(0.100, rel=1e-9, abs=1e-12)
    assert floor_piece == pytest.approx(0.100, rel=1e-9, abs=1e-12)

    # Independent owner-rate counterfactuals move the copy bound and both
    # observed spans together.  No native or piecewise elapsed value is used
    # as the expected bound.
    for scale in (0.5, 2.0):
        rate_native, rate_piece = _hc_probe(case, mte_scale=scale)
        rate_lower_ms = operand_bytes / (per_owner * scale) * 1000.0
        assert rate_native == pytest.approx(rate_lower_ms, rel=PARITY)
        assert rate_piece == pytest.approx(rate_lower_ms, rel=PARITY)


def test_a3_body_total_parity_on_matched_scope():
    a3 = _a3_result()
    a3_native = float(a3["native"]["body_forward_ms"])
    a3_piece = a3["body_ms"]
    assert a3_piece == pytest.approx(a3_native, rel=PARITY)


def test_qwen_a5_body_total_parity():
    qwen = _qwen_result()
    qwen_native = float(qwen["native"]["body_forward_ms"])
    qwen_piece = qwen["body_ms"]
    assert qwen_piece == pytest.approx(qwen_native, rel=PARITY)


def test_declared_kda_implementation_option_is_bound_on_both_paths():
    base = _a3_result()
    raw = _a3_definition()
    raw["runtime"]["kernels"]["lowerings"]["kda"]["options"]["use_beta_sigmoid"] = False
    changed = _native_and_piece(
        _case(raw), tokens=6, context=4096, sequences=4,
        tp=16, dp=1, ep=16, pp=1)
    native_delta = float(changed["native"]["body_forward_ms"]) - float(
        base["native"]["body_forward_ms"])
    piece_delta = changed["body_ms"] - base["body_ms"]
    assert native_delta > 0
    assert piece_delta == pytest.approx(native_delta, rel=PARITY)


def test_parity_source_declarations_carry_no_observed_timing_payload():
    for path in (A3_CASE, QWEN_CASE):
        kernels = _load(path).get("runtime", {}).get("kernels", {})
        assert not (set(kernels) & set(_SERVICE_KEYS))


def test_opaque_case_and_workload_labels_do_not_change_either_path():
    baseline = _a3_result()
    raw = _a3_definition()
    raw["name"] = "opaque-unrelated-label"
    metadata = raw.setdefault("runtime", {}).setdefault("metadata", {})
    metadata["capture_directory"] = "heldout/not/a/measurement"
    raw["model"].setdefault("metadata", {})["model_coverage"] = "renamed"
    raw["workload"]["requests"][0]["id"] = "not-the-baseline-request"
    changed = _case(raw)
    changed_result = _native_and_piece(changed, tokens=6, context=4096, sequences=4,
                                       tp=16, dp=1, ep=16, pp=1)
    assert float(changed_result["native"]["body_forward_ms"]) == pytest.approx(
        float(baseline["native"]["body_forward_ms"]), rel=0, abs=0)
    assert changed_result["body_ms"] == pytest.approx(baseline["body_ms"], rel=0, abs=0)
    assert _piece_unit_counts(changed_result) == _piece_unit_counts(baseline)


def test_declared_hardware_rates_remain_fixed_costs_on_both_paths():
    result = _a3_result()
    case = result["case"]
    machine = case.machine
    factor = 2.0
    scaled = replace(
        machine,
        dies=tuple(replace(
            die,
            rates=tuple((key, value * factor) for key, value in die.rates),
            compute_peaks=tuple((key, value * factor) for key, value in die.compute_peaks),
        ) for die in machine.dies),
        links=tuple(replace(link, capacity_bytes_per_ms=link.capacity_bytes_per_ms * factor)
                    for link in machine.links),
    )
    scaled_case = replace(case, machine=scaled)
    faster = _native_and_piece(scaled_case, tokens=6, context=4096, sequences=4,
                               tp=16, dp=1, ep=16, pp=1)
    assert float(faster["native"]["body_forward_ms"]) < float(result["native"]["body_forward_ms"])
    assert faster["body_ms"] < result["body_ms"]
