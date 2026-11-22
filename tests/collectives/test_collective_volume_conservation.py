"""Collective volume conservation across peers, plans, handoffs, and intervals.

[Role]
  Verify message/round framing, source volumes, directed peer/cohort traffic,
  caller partition, direct-peer windows, interval algebra, and PD handoffs.
[Invariants]
  * Peak wire volume is charged once per message/round; chunking conserves it.
  * One source partition owns the volume; directed cells own pairwise volume.
  * Windows/credits bound admission, never work; intervals are half-open and a
    self copy never enters the remote-link contract.
  * A whole-message grant is atomic; decode-complete releases consumer credit.
[Timeouts & Locks]
  Tier-0 in-memory Engines and fixture reads (<5 s); no locks, sleeps, threads
  or subprocesses.
[Side-Effect Policy]
  Read-only; builds fresh mappings and writes no repository state.
"""
from __future__ import annotations

import heapq
import importlib.util
import json
import random
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from mindie_sim import (
    compose_case, graph_costs, operator_costs, predict, predict_pd, target_graphs,
)
from mindie_sim.prediction.execution.cohort_traffic import bind_cohort_traffic
from mindie_sim.prediction.collectives import PairwiseCollective
from mindie_sim.prediction.operators.collectives.direct_peer import DirectPeerGroup
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.operators.collectives.hccl_alltoallv import AlltoAllVGroup
from mindie_sim.prediction.operators.collectives.hccl_ccu_two_die import TEMPLATE
from mindie_sim.prediction.operators.collectives.hccl_contracts import select as hccl_select
from mindie_sim.prediction.operators.collectives.hccl_graph import lower as hccl_lower
from mindie_sim.prediction.operators.moe_routing.moe_token_partition import ALLTOALL_KIND
from mindie_sim.prediction.pd_connector import PDConnector, PDHandoff, PDTransfer
from mindie_sim.prediction.collectives.peer_traffic import PeerTraffic
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

from _support.collective_cases import (
    case, exchange, per_pair_topology, ready_events, topology,
)
from _support.memory_cases import linear_work

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ((0, 0, 0), (2, 0, 2), (4, 4, 0))
TRANSPOSE = ((0, 2, 4), (0, 0, 4), (0, 2, 0))
EXCHANGE_RECIPES = ("expert_dispatch", "expert_combine")

_AUDIT_SPEC = importlib.util.spec_from_file_location(
    "audit_v4flash_communication", ROOT / "tools" / "audit_v4flash_communication.py")
AUDIT = importlib.util.module_from_spec(_AUDIT_SPEC)
_AUDIT_SPEC.loader.exec_module(AUDIT)

def _prefix(engine, prefix: str) -> float:
    return sum(value for name, value in engine.serviced.items() if name.startswith(prefix))

def _pd_case(requests=None) -> dict:
    raw = case(prompt=3, output=3, requests=requests)
    pools = {pool: deepcopy({"machine": raw["machine"], "runtime": raw["runtime"]}) for pool in ("prefill", "decode")}
    raw["pools"] = pools
    raw["connector"] = {"bytes_per_s": 128000, "latency_us": 0}
    return raw

def _peer_case(*, dp=1, tp=3) -> dict:
    raw = case(dp=dp, tp=tp, ep=3)
    raw["model"].update(ffn="moe", num_experts=6, top_k=1, moe_intermediate_size=32,
                        num_heads=6, num_kv_heads=3, hidden_size=48)
    raw["machine"]["links"] = [{"name": f"p{a}{b}", "members": [a, b],
                               "semantics": "per_pair", "capacity_bytes_per_ms": 1}
                              for a in range(3) for b in range(a + 1, 3)]
    return raw

def _matrix_of(traffic) -> tuple:
    return tuple(tuple(traffic.payload(src, dst) for dst in range(traffic.size)) for src in range(traffic.size))

def _directed_wire(matrix) -> dict:
    return {f"link:p{min(s, d)}{max(s, d)}:pair{s}>{d}": matrix[s][d]
            for s in range(len(matrix)) for d in range(len(matrix)) if matrix[s][d]}

def _window_arrivals(matrix, ready, window, capacity=1, latency=.005):
    count, completed = len(matrix), list(ready)
    for src in range(count):
        slots = [ready[src]] * min(window, count - 1)
        heapq.heapify(slots)
        for offset in range(1, count):
            dst = (src + offset) % count
            if matrix[src][dst]:
                end = max(heapq.heappop(slots), ready[dst]) + matrix[src][dst] / capacity + latency
                heapq.heappush(slots, end)
                completed[src] = max(completed[src], end)
                completed[dst] = max(completed[dst], end)
    return completed

def _source_case(*, dp=1, tp=1, ep=4, balanced=False) -> dict:
    raw = case(dp=dp, tp=tp, ep=ep, prompt=4)
    raw["model"].update(layers=1, ffn="moe", num_experts=8, top_k=2, num_heads=tp,
                        num_kv_heads=tp, moe_intermediate_size=64, head_dim=32 // tp)
    if balanced:
        raw["runtime"].update(routing={"policy": "balanced"}, max_num_batched_tokens=16)
    return raw

def _caller(raw: dict) -> dict:
    raw["runtime"]["kernels"] = { "moe_token_partition": {"kind": ALLTOALL_KIND, "layer_scopes": ["body"],
                                "gather_output_materialization": "per_rank_copy"},
        "moe_transport": {"kind": "separate_quantized_payload", "payload_dtype": "int8",
                          "scale_dtype": "fp32", "scale_granularity": "row",
                          "histogram_dtype": "int32", "layer_scopes": ["body"]}}
    return raw

def _ccu_case(preset: str) -> dict:
    raw = compose_case("qwen3.5-35b-a3b", hardware="Ascend950DT_9581",
                       topology_preset=preset, tp=8, dp=1, ep=8, layers=1,
                       prompt_tokens=8, output_tokens=1, concurrency=1)
    device = raw["machine"]["device"]
    device["rates"].update(hccl_control=1e9, ccu_copy=1e9, ccu_buffer_bytes=1e12, ccu_control=1e9)
    device["resource_units"].update(dict.fromkeys( ("hccl_control", "ccu_copy", "ccu_buffer_bytes", "ccu_control"), 2))
    raw["runtime"]["kernels"] = {"hccl": {"allgather": {
        "expansion": "CCU_MS", "template": TEMPLATE, "topology": "TWO_DIE_REGULAR",
        "signal_bytes": 8, "channel_die_map": [[None if src == dst else
            int(((dst - src) % 8) >= 4) for dst in range(8)] for src in range(8)]}}}
    return raw

def _leaf_contexts(topology, ranks):
    resources = list(topology.resources())
    bindings = {}
    for rank in ranks:
        rates = {"hbm": 1e9, "hccl_copy": 1e9, "hccl_control": 1e9}
        names = {role: f"{role}:{rank}" for role in rates}
        resources.extend(Resource(names[role], rate) for role, rate in rates.items())
        bindings[rank] = RecipeContext(None, {}, names, {names[role]: rate for role, rate in rates.items()}, 1,
            resource_units={name: 1 for name in names.values()})
    return resources, bindings

def test_peer_framing_is_charged_per_message_and_per_hop():
    raw = case(tp=4)
    raw["machine"]["links"][0].update(transaction_bytes=16, protocol_overhead_bytes=4)
    work = topology(raw).accounting(exchange(30), 0)
    assert work.to_dict()["useful_bytes"] == 120
    assert work.to_dict()["wire_bytes"] == 12 * 20
    row = operator_costs(raw, (exchange(30),))["operators"][0]
    assert row["duration_ms"] == pytest.approx(240 / 1000)
    analytic = topology(case(tp=4)).accounting(exchange(10), 0)
    assert analytic.transactions_by_resource == {"link:fabric": None}
    ring = case(tp=4)
    ring["machine"]["links"][0].update(transaction_bytes=16, protocol_overhead_bytes=4, latency_us=2)
    rounds = topology(ring).accounting(exchange(42, family="allreduce"), 0)
    assert (rounds.messages, rounds.wire_by_resource["link:fabric"]) == (24, 480)
    assert rounds.setup_ms == pytest.approx(6 * .002)
    report = graph_costs(ring, (exchange(42, family="allreduce"),))
    assert report["duration_ms"] == pytest.approx(.480 + .012)
    assert report["isolated"]["operators"][0]["communication"]["messages"] == 24
    raw = case(tp=4, prompt=4, output=1)
    raw["machine"]["links"][0].update(transaction_bytes=64, protocol_overhead_bytes=8)
    graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=4, scope="model"))
    served = predict(raw)
    expected = sum(row["communication"]["wire_bytes"] * row["occurrences"] for row in
                   graph["isolated"]["operators"] if "communication" in row)
    assert graph["execution"]["serviced"]["link:fabric"] == expected
    assert served["execution"]["serviced"]["link:fabric"] == expected
    assert served["execution"]["live_tasks"] == served["execution"]["live_programs"] == 0

SOURCE_CONTRACTS = ( (1, 4, 4, False, "probability_moments", 4, 96.0, 384.0, None),
    (1, 4, 4, True, "forced_balanced_integer_counts", 4, 96.0, 384.0, None),
    (2, 2, 4, False, "probability_moments", 2, 192.0, 384.0, None),
    (2, 2, 4, True, "forced_balanced_integer_counts", 2, 192.0, 384.0, None),
    (1, 8, 4, False, "probability_moments", 4, 96.0, 768.0, (384.0, 384.0)),
    (1, 8, 4, True, "forced_balanced_integer_counts", 4, 96.0, 768.0, (384.0, 384.0)),
)

@pytest.mark.parametrize("dp,tp,ep,balanced,basis,partitions,link_bytes,per_replica,group_totals", SOURCE_CONTRACTS)
def test_source_volume_contract_partitions_each_physical_source(
        dp, tp, ep, balanced, basis, partitions, link_bytes, per_replica, group_totals):
    raw = _source_case(dp=dp, tp=tp, ep=ep, balanced=balanced)
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="body")
    topo = topology(raw)
    for replica, ops in graphs.items():
        both = [op for op in ops if op.recipe in EXCHANGE_RECIPES]
        assert [op.link_bytes for op in both] == [link_bytes, link_bytes]
        for op in both:
            metadata = dict(op.metadata)
            assert (metadata["source_assignment_partitions"], metadata["traffic_model"],
                    metadata["routing_basis"]) == (partitions, "uniform_peer_payload", basis)
        work = topo.accounting(next(op for op in both if op.recipe == "expert_dispatch"), replica)
        assert work.to_dict()["useful_bytes"] == per_replica
        assert set(work.useful_by_rank.values()) == {link_bytes}
        if group_totals:
            assert [sum(work.useful_by_rank[rank] for rank in group)
                    for group in topo.groups("ep", 0, 0)] == list(group_totals)

CALLER_CASES = (("v4f-a3-w8a8-tp8", 4096, 8, 4096, 256), ("v4pro-a3-w4a8-tp16", 1024, 16, 7168, 384))

@pytest.mark.parametrize("name,rows,tp,hidden,experts", CALLER_CASES)
def test_caller_partition_reconstructs_local_rows_and_five_collectives( name, rows, tp, hidden, experts):
    raw = json.loads((ROOT / "examples/cases" / f"{name}.json").read_text())
    raw["model"]["layers"] = 1
    _caller(raw)
    graph = target_graphs(raw, phase="prefill", tokens=rows, scope="body")[0]
    ops = {op.op_id: op for op in graph}
    local, assignments = rows // tp, rows // tp * 6
    assert ops["s0.l0.topk"].local_shape == (local, experts)
    assert ops["s0.l0.topk"].deps == ("s0.l0.alltoall_logits_view",)
    assert ops["s0.l0.pack"].mte3_bytes == assignments * (hidden * 2 + 4)
    assert ops["s0.l0.unpack"].local_shape == (local, 6, hidden)
    hist, counts = ops["s0.l0.moe_histogram"], ops["s0.l0.moe_histogram_gather"]
    assert (hist.dtype, counts.link_bytes) == ("int32", (tp - 1) * experts * 4)
    quant = ops["s0.l0.moe_transport_quantize"]
    assert (quant.mte2_bytes, quant.mte3_bytes) == (assignments * hidden * 2, assignments * (hidden + 4))
    remote = (tp - 1) / tp
    scale, payload, back = (ops["s0.l0." + suffix] for suffix in
                            ("moe_scale_dispatch", "moe_payload_dispatch", "expert_combine"))
    assert [op.link_bytes for op in (scale, payload, back)] == [
        assignments * 4 * remote, assignments * hidden * remote, assignments * hidden * 2 * remote]
    assert ops["s0.l0.alltoall_tp_gather"].link_bytes == (tp - 1) * local * hidden * 2
    copies = [op for op in graph if ".alltoall_gather_copy" in op.op_id]
    assert len(copies) == tp and sum(op.mte2_bytes for op in copies) == rows * hidden * 2
    control = deepcopy(raw)
    del control["runtime"]["kernels"]["moe_token_partition"]
    previous = {op.op_id: op for op in target_graphs( control, phase="prefill", tokens=rows, scope="body")[0]}
    assert previous["s0.l0.expert_combine"].link_bytes == back.link_bytes

def test_peer_transpose_and_pairwise_recurrence_keep_directed_cells():
    traffic = PeerTraffic.from_mapping( {"kind": "uniform_peers", "source_bytes": [0, 4, 8]}, 3)
    assert _matrix_of(traffic) == MATRIX
    assert [traffic.row_total(rank) for rank in range(3)] == [0, 4, 8]
    reverse = traffic.transposed()
    assert _matrix_of(reverse) == TRANSPOSE
    assert [reverse.row_total(rank) for rank in range(3)] == [6, 4, 2]
    topo = topology(_peer_case())
    op = exchange(4, family="alltoallv", axis="tp", metadata=(
        ("peer_traffic", {"kind": "matrix", "payload_bytes": MATRIX}), ("peer_volume_basis", "group_mean")))
    work = topo.accounting(op, 0)
    assert work.useful_by_rank == {1: 4, 2: 8}
    assert work.wire_by_resource == _directed_wire(MATRIX)
    plan = topo.joint_plan(op, {0: 4}, algorithm="pairwise")
    assert plan.groups[0].volumes == ((0, 0), (1, 4), (2, 8))
    engine = Engine(topo.resources())
    events = plan.start_participants(engine, "mean", ready_events(engine, [0, 0, 0]))
    engine.advance(until=tuple(events.values()), first=False)
    assert [events[rank].time for rank in range(3)] == [4.0, 8.0, 8.0]
    assert {name: value for name, value in engine.serviced.items() if value} == _directed_wire(MATRIX)
    with pytest.raises(ModelingError, match="peer_traffic_source_volume_mismatch"):
        topo.joint_plan(replace(op, metadata=tuple(
            (key, value) for key, value in op.metadata if key != "peer_volume_basis")), {0: 4},
            algorithm="pairwise")

def test_cohort_binding_conserves_directed_dispatch_and_return():
    parsed = parse_case(_peer_case(dp=3, tp=1))
    graphs = {}
    for replica, volume in enumerate((0, 4, 8)):
        forward = replace(exchange(volume, axis="ep",
            metadata=(("collective_id", "body.dispatch"),)), op_id="body.dispatch")
        graphs[replica] = (forward, replace(exchange(volume, axis="ep", metadata=(
            ("collective_id", "body.combine"), ("reverse_of_collective", "body.dispatch"))),
            op_id="body.combine", deps=(forward.op_id,)))
    bound = bind_cohort_traffic(parsed, graphs)
    assert [ops[1].link_bytes for ops in bound.values()] == [6, 4, 2]
    report = graph_costs(parsed, graphs, input_ready_ms=7, recipes=RecipeBook(
        families={"alltoall": PairwiseCollective()}))
    expected = _directed_wire(MATRIX)
    for name, value in _directed_wire(TRANSPOSE).items():
        expected[name] = expected.get(name, 0) + value
    actual = {name: value for name, value in report["execution"]["serviced"].items() if value}
    assert actual == pytest.approx(expected)
    assert sum(actual.values()) == 24 and report["duration_ms"] == 14
    with pytest.raises(ModelingError, match="complete_dispatch_cohort"):
        bind_cohort_traffic(parsed, {0: graphs[0]})
    with pytest.raises(ValueError, match="complete_dispatch_cohort"):
        operator_costs(parsed, graphs[0])
    framing = per_pair_topology(3, capacity=1000, framing={"transaction_bytes": 16, "protocol_overhead_bytes": 4})
    cells = ((0, 17, 1), (18, 0, 0), (1, 17, 0))
    work = framing.accounting(exchange(18, family="alltoall", axis="tp", metadata=(
        ("peer_traffic", {"kind": "matrix", "payload_bytes": cells}),)), 0)
    assert (work.messages, work.to_dict()["wire_bytes"]) == (5, 160)
    with pytest.raises(ModelingError, match="peer_traffic_source_volume_mismatch"):
        per_pair_topology(3).accounting(exchange(12, metadata=(
            ("peer_traffic", {"kind": "matrix", "payload_bytes": MATRIX}),)), 0)
    with pytest.raises(ModelingError, match="explicit_peer_matrix"):
        per_pair_topology(3).accounting(exchange(2, family="alltoallv"), 0)

@pytest.mark.parametrize("window", (1, 7))
def test_direct_peer_window_matches_independent_slot_recurrence(window):
    rng = random.Random(291031)
    for count in (2, 5):
        matrix = tuple(tuple(0 if src == dst else rng.randrange(8) for dst in range(count)) for src in range(count))
        traffic = PeerTraffic.from_mapping({"kind": "matrix", "payload_bytes": matrix}, count)
        for peer in (traffic, traffic.transposed()):
            topo = per_pair_topology(count, latency_us=5)
            initial = [float(rng.randrange(5)) for _ in range(count)]
            volumes = {source: peer.row_total(source) for source in range(count)}
            group = DirectPeerGroup(tuple(range(count)), tuple(volumes.items()), count - 1, topo, peer, window)
            engine = Engine(topo.resources())
            events = group.start_participants(engine, "direct", ready_events(engine, initial))
            engine.advance(until=tuple(events.values()), first=False)
            assert [events[rank].time for rank in range(count)] == pytest.approx(
                _window_arrivals(_matrix_of(peer), initial, window))

def test_direct_peer_lazy_chunks_and_receive_pools():
    large = DirectPeerGroup((0, 1), ((0, 4097.0), (1, 4097.0)), 1, per_pair_topology(2, capacity=64), chunk_bytes=1,
                            receive_capacity_bytes_per_peer=64, max_physical_chunks=None)
    engine = Engine(large.topology.resources())
    done = large.start_participants( engine, "peer-large", {rank: engine.submit(f"r{rank}") for rank in (0, 1)})
    engine.advance(until=tuple(done.values()), first=False)
    assert _prefix(engine, "link:") == pytest.approx(2 * 4097)
    ranks = (2, 3)
    group = DirectPeerGroup(ranks, ((2, 256.0), (3, 256.0)), 1, per_pair_topology(4, capacity=64), chunk_bytes=64,
                            receive_capacity_bytes_per_peer=64)
    engine = Engine(group.topology.resources())
    ready = {rank: engine.submit(f"r{rank}") for rank in ranks}
    consumers = {pair: tuple(engine.submit(f"c{pair[0]}{pair[1]}{ms}", latency=ms)
                             for ms in (2, 4, 8, 10)) for pair in (ranks, tuple(reversed(ranks)))}
    done = group.start_participants(engine, "peer-pool", ready, release_events=consumers)
    engine.advance(until=tuple(done.values()) + tuple(
        event for sequence in consumers.values() for event in sequence), first=False)
    assert max(event.time for event in done.values()) == pytest.approx(9.0)
    assert _prefix(engine, "link:") == pytest.approx(512.0)
    assert all(engine.allocated[f"transport:rx:peer:{a}>{b}"] == pytest.approx(0.0) for a, b in consumers)

def test_interval_algebra_is_end_exclusive_and_excludes_self_copy():
    assert AUDIT.union_ns([(0, 10), (5, 15), (20, 30)]) == 25
    assert AUDIT.union_ns([(0, 10), (10, 20)]) == 20
    assert AUDIT.intersection_ns([(0, 10), (20, 30)], [(3, 12), (25, 40)]) == 12
    assert AUDIT.intersection_ns([(0, 10)], [(10, 20)]) == 0
    entry = json.loads((ROOT / AUDIT.DEFAULT_INPUT).read_text())
    conditions = next(row for row in entry["entries"] if row["id"] == AUDIT.MODEL_ID )["observed"]["conditions"]
    contract = AUDIT._contract(ROOT / AUDIT.DEFAULT_CASE, ROOT / AUDIT.DEFAULT_CATALOG, conditions)
    assert contract["source_assignment_partitions"] == 8
    assert contract["self_bytes_per_source_per_layer"] == 3_145_728
    assert contract["remote_bytes_per_source_per_layer"] == 22_020_096
    assert contract["self_copy_modeled_as_link"] is False

def test_ccu_leaf_placement_changes_route_and_sharing_domain():
    reports = {}
    for preset in ("a5_server", "a5_pod"):
        raw = _ccu_case(preset)
        parsed = parse_case(raw)
        selected = hccl_lower((OpSpec("ag", "allgather", "decode", 0, 0, (), (1,), (1,),
                                      (1,), (1,), link_bytes=210, metadata=(
                                          ("group_axis", "tp"), ("transport_hop_startup_us", 0.5))),),
                              parsed.runtime, parsed.model.activation_dtype)
        reports[preset] = graph_costs(raw, selected, input_ready_ms=7, trace_limit=20000)
    server = {name for name, _ in reports["a5_server"]["execution"]["serviced"].items() if name.startswith("link:")}
    pod = {name for name, _ in reports["a5_pod"]["execution"]["serviced"].items() if name.startswith("link:")}
    assert any("d2d_intra_4p" in name for name in server)
    assert not any("d2d_intra_4p" in name for name in pod)
    server_wire = sum(value for name, value in reports["a5_server"]["execution"]["serviced"].items()
                      if name.startswith("link:"))
    pod_wire = sum(value for name, value in reports["a5_pod"]["execution"]["serviced"].items()
                   if name.startswith("link:") and ":tx:port" in name)
    assert server_wire == pytest.approx(pod_wire)

def test_alltoallv_leaf_chunk_credit_and_consumer_release():
    count, cell = 4, 30
    matrix = tuple(tuple(0 if source == destination else cell for destination in range(count))
                   for source in range(count))
    topo = per_pair_topology(count, capacity=64)
    selection = hccl_select({"expansion": "AICPU_TS", "template": "mesh_write",
                             "dtype_bytes": 1}, collective="alltoallv", ranks=count,
                            payload_bytes=cell * (count - 1))
    def run(group, tag):
        resources, bindings = _leaf_contexts(topo, tuple(range(count)))
        engine = Engine(resources, trace_limit=5000)
        ready = {rank: engine.submit(f"{tag}:ready{rank}") for rank in range(count)}
        done = group.start_participants(engine, tag, ready, bindings)
        engine.advance(until=tuple(done.values()), first=False)
        return done, engine
    plan_done, plan_engine = run(AlltoAllVGroup( selection, matrix, tuple(range(count)), topo, chunk_bytes=16,
        hop_startup_us=1, receive_capacity_bytes_per_peer=16), "plan")
    tasks = [row["task"] for row in plan_engine.trace if ":transport:flow0:" in row["task"]]
    assert any(":chunk1:" in task for task in tasks)
    assert any(task.endswith(":chunk0:hop0:startup") for task in tasks)
    assert not any(":chunk1:hop0:startup" in task for task in tasks)
    assert all(event.done for event in plan_done.values())
    assert _prefix(plan_engine, "hbm:") == pytest.approx(16 * 30 * 3)

@pytest.mark.parametrize("empty_consumer", (False, True))
def test_alltoallv_leaf_consumer_release_holds_receive_pool(empty_consumer):
    topo = per_pair_topology(2, capacity=64)
    selection = hccl_select({"expansion": "AICPU_TS", "template": "mesh_write",
                             "dtype_bytes": 1}, collective="alltoallv", ranks=2,
                            payload_bytes=64)
    resources, bindings = _leaf_contexts(topo, (0, 1))
    engine = Engine(resources, trace_limit=1000)
    ready = {rank: engine.submit(f"ready{rank}") for rank in (0, 1)}
    releases = {rank: engine.future(f"release{rank}") for rank in (0, 1)}
    done = AlltoAllVGroup( selection, ((0, 64), (64, 0)), (0, 1), topo, chunk_bytes=32,
        receive_capacity_bytes_per_peer=64,
        receive_buffer_ids=((0, "channel0"), (1, "channel1"))).start_participants(
            engine, "hccl", ready, bindings,
            release_events={rank: (() if empty_consumer else (releases[rank],)) for rank in (0, 1)})
    engine.advance(until=tuple(done.values()), first=False)
    tasks = {row["task"] for row in engine.trace}
    assert {"transport:rx:channel0", "transport:rx:channel1"} <= set(engine.pools)
    if empty_consumer:
        assert all(f"hccl:{rank}->{other}:receive_release" not in tasks for rank, other in ((0, 1), (1, 0)))
        return
    consumers = tuple(engine.submit( f"consume{rank}", after=(done[rank],), latency=1,
        on_complete=lambda _event, target=releases[rank]: None
        if target.done else engine.resolve(target)) for rank in (0, 1))
    engine.advance(until=consumers, first=False)
    assert "hccl:0->1:receive_release" in {row["task"] for row in engine.trace}
    assert _prefix(engine, "link:") == pytest.approx(128.0)

def test_pd_whole_message_grants_are_atomic_and_avoid_deadlock():
    connector = PDConnector({"link": {"name": "handoff", "capacity_bytes_per_ms": 128,
                                      "latency_us": 0}, "chunk_bytes": 128})
    def consume(engine, done, releases):
        consumers = [engine.submit( f"consumer{index}", after=(event,), latency=1,
            on_complete=lambda _event, target=release: None
            if target.done else engine.resolve(target))
            for index, (event, release) in enumerate(zip(done, releases))]
        engine.advance(until=tuple(consumers), first=False)
        return [event.time for event in consumers]
    engine = Engine(connector.resources(), trace_limit=1000)
    source = engine.submit("source")
    grants = ({2: engine.submit("ready2", release=0), 3: engine.submit("ready3", release=10)},
              {2: engine.submit("ready2b", release=10), 3: engine.submit("ready3b", release=0)})
    releases = [engine.future("first_release"), engine.future("second_release")]
    def cross(name, src, grant, release):
        return connector.submit_handoff( engine, PDHandoff.build(
                name, (PDTransfer(src, 2, 128.0), PDTransfer(src, 3, 128.0)),
                receive_capacity_bytes={2: 128.0, 3: 128.0},
                receive_buffer_ids={2: "X", 3: "Y"}),
            source_ready_by_rank={src: source}, destination_ready_by_rank=grant,
            release_events={2: (release,), 3: (release,)})
    done = [cross("first", 0, grants[0], releases[0]), cross("second", 1, grants[1], releases[1])]
    assert consume(engine, done, releases) == pytest.approx([13.0, 16.0])
    assert [event.time for event in done] == pytest.approx([12.0, 15.0])
    transport = connector.summary()["transport"]
    assert transport["peak_reserved_bytes"]["2"] == pytest.approx(128.0)
    assert connector.wire_bytes == pytest.approx(512.0)
    pool = PDConnector({"link": {"name": "handoff", "capacity_bytes_per_ms": 128, "latency_us": 0}, "chunk_bytes": 128})
    engine = Engine(pool.resources(), trace_limit=1000)
    ready = {0: engine.submit("source"), 1: engine.submit("destination")}
    releases = [engine.future("first"), engine.future("second")]
    def pair(name, release):
        return pool.submit_handoff( engine, PDHandoff.build(name, (PDTransfer(0, 1, 256.0),),
                                    receive_capacity_bytes={1: 256.0},
                                    receive_buffer_ids={1: "pool"}),
            source_ready_by_rank={0: ready[0]},
            destination_ready_by_rank={1: ready[1]},
            release_events={1: (release,)})
    done = [pair("first", releases[0]), pair("second", releases[1])]
    assert consume(engine, done, releases) == pytest.approx([3.0, 6.0])
    assert [event.time for event in done] == pytest.approx([2.0, 5.0])
    transport = pool.summary()["transport"]
    assert transport["peak_reserved_bytes"]["1"] == pytest.approx(256.0)
    assert transport["reservation_waits"] >= 1

def _layout_case(*, pools) -> dict:
    raw = _pd_case([{"id": "r", "prompt_tokens": 1, "output_tokens": 3}])
    for pool, ranks, tp, pp in pools:
        machine = raw["pools"][pool]["machine"]
        machine.update(num_devices=len(ranks), devices=[{"rank": rank} for rank in ranks])
        machine["links"][0]["members"] = list(ranks)
        raw["pools"][pool]["runtime"].update(tp=tp, pp=pp)
    raw["connector"] = {"link": {"name": "handoff", "capacity_bytes_per_ms": 64,
                                 "latency_us": 0, "semantics": "per_pair"},
                        "chunk_bytes": 64}
    return raw

@pytest.mark.parametrize("kind,expected_payload", (("pp", 64.0), ("tp", 32.0)))
def test_pd_transfer_sources_follow_pp_tp_range_intersection(kind, expected_payload):
    layout = (("prefill", (0, 1), 1, 2), ("decode", (2,), 1, 1)) if kind == "pp" else (
        ("prefill", (0, 1), 2, 1), ("decode", (2,), 1, 1))
    result = predict_pd(_layout_case(pools=layout), lower=linear_work, event_limit=100000)
    handoff = result["execution"]["pd_handoffs"][0]
    assert (handoff["source_ranks"], handoff["destination_ranks"]) == ([0, 1], [2])
    assert all(row["payload_bytes"] == pytest.approx(expected_payload) for row in handoff["transfers"])
    assert sum(row["payload_bytes"] for row in handoff["transfers"]) == pytest.approx(128.0)
    if kind == "pp":
        assert {(row["source_rank"], row["source_stage"]) for row in handoff["transfers"]} == {(0, 0), (1, 1)}
    else:
        assert {row["source_layers"][0] for row in handoff["transfers"]} == {0, 1}
        assert all(row["source_stage"] == row["destination_stage"] == 0 for row in handoff["transfers"])
    assert sum(value for name, value in result["execution"]["serviced"].items()
               if name.startswith("pd:link:handoff")) == pytest.approx(128.0)
    assert [row["status"] for row in result["requests"]] == ["served"]
    renumbered = _layout_case(pools=(("prefill", (11, 5), 1, 2), ("decode", (23,), 1, 1)))
    handoff = predict_pd(renumbered, lower=linear_work, event_limit=100000)["execution"]["pd_handoffs"][0]
    assert handoff["source_ranks"] == [5, 11] and handoff["destination_ranks"] == [23]
    assert {(row["source_rank"], row["source_stage"]) for row in handoff["transfers"]} == {(11, 0), (5, 1)}

def test_pd_consumer_release_uses_decode_device_complete_not_host_tail():
    raw = _pd_case()
    raw["pools"]["decode"]["runtime"]["submission"] = {"completion_ms": 10}
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    rows = {row["task"]: row for row in result["events"]}
    consumer, host = rows["pd:r:consumer_boundary"], rows["decode:r0:i0:complete"]
    assert consumer["start_ms"] == pytest.approx( rows["decode:r0:i0:device_complete"]["end_ms"])
    assert consumer["start_ms"] < host["end_ms"] == pytest.approx(17.0)

def test_pd_connector_options_flat_alias_and_transport_fold():
    with pytest.raises(ModelingError, match="pd_connector_chunk_bytes_invalid:0"):
        PDConnector({"link": {"name": "h", "capacity_bytes_per_ms": 128}, "chunk_bytes": 0})
    with pytest.raises(ModelingError, match="pd_connector_receive_capacity_bytes_invalid:0"):
        PDConnector({"link": {"name": "h", "capacity_bytes_per_ms": 128}, "receive_capacity_bytes": 0})
    with pytest.raises(ModelingError, match="pd_connector_hop_startup_us_invalid"):
        PDConnector({"link": {"name": "h", "capacity_bytes_per_ms": 128}, "hop_startup_us": -1})
    flat = PDConnector({"bytes_per_s": 128000, "chunk_bytes": 128,
                        "receive_capacity_bytes": 256, "transaction_bytes": 256,
                        "protocol_overhead_bytes": 0})
    engine = Engine(flat.resources(), trace_limit=200)
    done = flat.submit(engine, "pd:flat", 256, {})
    engine.advance(until=(done,), first=False)
    assert flat.wire_bytes == pytest.approx(512.0)
    assert flat.summary()["transport"]["peak_reserved_bytes"]["1"] <= 256
    plain = PDConnector({"bytes_per_s": 128000, "chunk_bytes": 128, "receive_capacity_bytes": 256})
    engine = Engine(plain.resources(), trace_limit=200)
    for name in ("pd:plain", "pd:plain2"):
        done = plain.submit(engine, name, 256, {})
    engine.advance(until=(done,), first=False)
    held = plain.submit(engine, "pd:held", 256, {}, release_events={1: ()})
    engine.advance(until=(held,), first=False)
    assert plain.summary()["transactions"] is None
    blocked = PDConnector({"bytes_per_s": 128000, "chunk_bytes": 128, "receive_capacity_bytes": 128})
    engine = Engine(blocked.resources())
    with pytest.raises(ModelingError, match="per_chunk_consumer"):
        blocked.submit(engine, "pd:blocked", 256, {}, release_events={1: engine.future("release")})

@pytest.mark.parametrize("chunk,event_kind,released", (
    (128, "one", True), (128, "two", True), (64, "two", True), (64, "early", False)))
def test_pd_reserved_pool_holds_footprint_until_consumer_coverage(
        chunk, event_kind, released):
    connector = PDConnector({"bytes_per_s": 128000, "chunk_bytes": chunk,
                             "receive_capacity_bytes": 256,
                             "receive_buffer_id": "pool", "transaction_bytes": 256})
    engine = Engine(connector.resources(), trace_limit=1000)
    ready = {0: engine.submit("source"), 1: engine.submit("destination")}
    consumers = {
        "one": (engine.submit("one", latency=8),),
        "two": (engine.submit("two1", latency=8), engine.submit("two2", latency=16)),
        "early": (engine.submit("early", latency=0.1),),
    }[event_kind]
    done = connector.submit(engine, "pd:reserved", 256, {}, ready=ready[0],
                            destination_ready=ready[1], release_events={1: consumers})
    engine.advance(until=(done, *consumers), first=False)
    tasks = {row["task"] for row in engine.trace}
    assert ("pd:reserved:receive_release:r1" in tasks) is released
    assert connector.summary()["transport"]["peak_reserved_bytes"]["1"] == pytest.approx(256.0)
    connector.submit(engine, "pd:again", 256, {}, ready=ready[0],
                     destination_ready=ready[1], release_events={1: ()})
    conflicting = PDConnector({"bytes_per_s": 128000, "chunk_bytes": 128,
                               "receive_capacity_bytes": 128,
                               "receive_buffer_id": "pool"})
    with pytest.raises(ModelingError, match="pd_receive_pool_capacity_conflict"):
        conflicting.submit(engine, "pd:conflict", 128, {}, ready=ready[0],
                           destination_ready=ready[1], release_events={1: ()})

def test_pd_legacy_connector_charges_transactions_and_consumer_release():
    connector = PDConnector({"link": {"name": "h", "capacity_bytes_per_ms": 128, "transaction_bytes": 256,
                                      "protocol_overhead_bytes": 0}})
    engine = Engine(connector.resources(), trace_limit=100)
    first = connector.submit(engine, "pd:first", 384, {})
    release = engine.submit("release", latency=4)
    second = connector.submit(engine, "pd:second", 384, {}, release_events={1: (release,)})
    engine.advance(until=(first, second, release), first=False)
    assert connector.transactions == 4
    assert "pd:second:consume" in {row["task"] for row in engine.trace}

def test_pd_handoff_build_normalizes_explicit_pool_capacity_mapping():
    handoff = PDHandoff.build("r", (PDTransfer(0, 1, 64.0),), receive_pool_capacity_bytes={1: 128.0},
                              receive_buffer_ids={1: "pool"})
    assert handoff.receive_pool_capacity_bytes == ((1, 128.0),)
    assert handoff.receive_capacity_bytes == ((1, 64.0),)
    assert handoff.receive_buffer_ids == ((1, "pool"),)
