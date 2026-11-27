"""Collective readiness, round recurrences, startup waves, and hop startup laws.

[Role]
  Verify how a collective acquires its start: observed readiness is bound to
  real pipeline events (never to profiler chronology), graph execution uses
  the true predecessor, ring/pairwise rounds are causal recurrences, startup
  waves partition peer offsets, and declared startup protocols charge gates
  per hop/round instead of multiplying payload service.

[Invariants]
  * A readiness report may only describe an already-ready pipeline event; a
    local clock never advances, delays, or creates collective work.
  * A round is a predecessor-delivery recurrence, not a global barrier: a slow
    hop delays its own successor only, and total wire work is conserved.
  * Startup rounds resolve from explicit steps, operator metadata, runtime
    mode, or declared topology protocol; unknown/illegal values fail loudly.
  * Wave ranges partition every peer offset once and conserve peer bytes;
    a zero-payload participant still rendezvous.
  * Startup gates are distinct engine events and are never propagation.

[Timeouts & Locks]
  Tier-0, in-memory Engines with bounded traces (<5 s); no locks, sleeps,
  threads, subprocesses or device contexts.

[Side-Effect Policy]
  Read-only; every mutable case/report is a fresh mapping owned by this
  module, and no environment or filesystem state changes.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.catalog.ops import comm_op
from mindie_sim.prediction.collectives.readiness import (
    SCHEMA as READINESS_SCHEMA, bind_pipeline_ready, readiness_skew, start_with_pipeline,
)
from mindie_sim.prediction.collectives.startup import resolve_collective_startup
from mindie_sim.prediction.collectives.waves import WaveGroup, wave_offsets, wave_ranges
from mindie_sim.prediction.collectives import RingCollective, RingGroup
from mindie_sim.prediction.communication import MessageFlow, RoutedHop, RoutedMessage
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.execution import _collective_native_basis
from mindie_sim.prediction.operators.collectives.hccl_signals import SignalGroup
from mindie_sim.prediction.collectives.peer_traffic import PeerTraffic
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology, TransportOptions
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware, Link
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec

from _support.builders import hardware
from _support.collective_cases import case, cohort_case

pytestmark = pytest.mark.tier0

READINESS_ENTRY_NS = 1_000_000_000
READINESS_TYPED_BYTES = 10240
SLOW_RING_US = 10_000

def _report(*, collective=None, typed_bytes=READINESS_TYPED_BYTES, dtype="BFP16") -> dict:
    calls = []
    for rank in (0, 1):
        operation = {"dtype": dtype, "typed_bytes": typed_bytes}
        if collective is not None:
            operation["collective"] = collective
        calls.append({ "rank_id": f"rank_{rank}", "rank_index": rank, "operation": operation,
            "entry": {"start_ns": READINESS_ENTRY_NS + rank},
            "completion": {"end_ns": READINESS_ENTRY_NS + 9_000_000 + rank},
            "observed_context": {"entry_domain": "rank_local", "cross_rank_aligned": False},
            "clock": {"domain": "rank_local", "cross_rank_aligned": False},
            "dependencies": (), "waits": (),
            "unknowns": ("cross_rank_clock_not_established",)})
    return {"schema": READINESS_SCHEMA, "calls": calls}

def _ring_group() -> RingGroup:
    return RingGroup((0, 1), 1, ( RoutedMessage(MessageFlow(0, 1, READINESS_TYPED_BYTES),
                      (RoutedHop("wire", READINESS_TYPED_BYTES, READINESS_TYPED_BYTES, None, None, 0.0),)),
        RoutedMessage(MessageFlow(1, 0, READINESS_TYPED_BYTES), (RoutedHop("wire", READINESS_TYPED_BYTES,
                                 READINESS_TYPED_BYTES, None, None, 0.0),))))

def _run_pipeline(rank0_delay: float):
    engine = Engine([Resource("wire", 1_000_000_000, "bytes")], trace_limit=16)
    base = engine.submit("base")
    ready = {0: engine.submit("upstream0", after=(base,), latency=rank0_delay),
             1: engine.submit("upstream1", after=(base,), latency=0.0)}
    completions, context = start_with_pipeline(_report(), engine, _ring_group(), ready)
    engine.advance(until=tuple(completions.values()))
    return ({rank: event.time for rank, event in completions.items()}, context, dict(engine.summary()["serviced"]))

def _exchange(volume: float = 30.0, *, family: str = "alltoall", axis: str = "tp",
              stage: int = 0, steps=None, metadata=()) -> OpSpec:
    entries = [("group_axis", axis)]
    if steps is not None:
        entries.append(("steps", steps))
    return OpSpec("exchange", family, "prefill", stage, 0, (), (1,), (1,), (1,), (1,),
                  link_bytes=volume, metadata=tuple(entries) + tuple(metadata))

def _startup(group, *, steps=None, runtime="legacy", protocols=None, family="alltoall"):
    metadata = {} if steps is None else {"steps": steps}
    return resolve_collective_startup( family=family, group_size=group, metadata=metadata,
        runtime_mode=runtime, topology_protocols=protocols or {})

def _op(name, *, flops=0, deps=(), stream="main"):
    return OpSpec(name, "test", "prefill", 0, 0, (), (1,), (1,), (1,), (1,),
                  flops=flops, deps=deps, dtype="bf16", stream=stream)

def _gather(*, axis="tp", deps=("producer",)):
    return comm_op(op_id="ag", deps=deps, stage=0, layer=0, requests=(),
                   family="allgather", phase="prefill", recipe="ring_allgather",
                   link_bytes=READINESS_TYPED_BYTES, shape=(2, 5120),
                   metadata=(("collective", "allgather"), ("group_axis", axis)))

def _ring_case() -> dict:
    raw = case(tp=4, prompt=4, output=1)
    raw["machine"]["links"] = [ {"name": f"edge{rank}", "members": [rank, (rank + 1) % 4],
         "semantics": "per_pair", "capacity_bytes_per_ms": 1,
         "latency_us": SLOW_RING_US if rank == 0 else 0}
        for rank in range(4)]
    return raw

def _independent_ring_arrivals(rounds, service, latency):
    """Serial source service and predecessor delivery, with no simulator code."""
    sent = [0.0] * len(service)
    received = [0.0] * len(service)
    for _ in range(rounds):
        sent = [max(sent[index], received[(index - 1) % len(service)]) + service[index]
                for index in range(len(service))]
        received = [sent[index] + latency[index] for index in range(len(service))]
    return received

def _wire_total(report: dict) -> float:
    return sum(value for name, value in report["execution"]["serviced"].items() if name.startswith("link:"))

def test_pipeline_predecessor_controls_completion_not_observed_chronology():
    base, _, base_service = _run_pipeline(0.0)
    delayed, context, delayed_service = _run_pipeline(5.0)
    assert base_service == delayed_service == {"wire": 2 * READINESS_TYPED_BYTES}
    assert delayed[0] > base[0] + 4.9 and delayed[1] > base[1] + 4.9
    assert all(item["readiness_basis"] == "pipeline_engine_event" for item in context.values())
    assert context[0]["entry_ns"] == READINESS_ENTRY_NS

def test_readiness_binding_requires_coverage_and_typed_operation():
    engine = Engine([Resource("wire", 1_000_000_000, "bytes")], trace_limit=16)
    base = engine.submit("base")
    with pytest.raises(ModelingError, match="collective_readiness_ready_coverage"):
        bind_pipeline_ready(_report(), {0: base})
    untyped = _report()
    untyped["calls"][0]["operation"].pop("typed_bytes")
    with pytest.raises(ModelingError, match="collective_readiness_typed_admission"):
        bind_pipeline_ready(untyped, {0: base, 1: base})

    ready = {0: engine.submit("r0"), 1: engine.submit("r1")}
    bind_pipeline_ready(_report(collective="all_gather"), ready, expected_collective="all_gather",
                        expected_bytes=READINESS_TYPED_BYTES,
                        expected_dtype="bf16", expected_group_size=2)
    for field, value, reason in (("dtype", "F32", "collective_readiness_dtype_mismatch"),
                                 ("typed_bytes", 20480, "collective_readiness_bytes_mismatch")):
        bad = _report(collective="all_gather")
        bad["calls"][0]["operation"][field] = value
        with pytest.raises(ModelingError, match=reason):
            bind_pipeline_ready(bad, ready, expected_collective="all_gather", expected_bytes=READINESS_TYPED_BYTES,
                                expected_dtype="bf16")

def test_readiness_context_never_overrides_actual_graph_predecessor():
    raw = case(dp=1, tp=2)
    raw["runtime"]["metadata"] = {"collective_readiness": _report(collective="all_gather")}
    baseline = graph_costs(raw, {0: (_op("producer", flops=1e6), _gather())},
                           external_inputs={"input": 0.0}, include_submission=False)
    delayed = graph_costs( raw, {0: (_op("producer", flops=1e6, deps=("input",)), _gather())},
        external_inputs={"input": 5.0}, include_submission=False)
    before = {row["id"]: row for row in baseline["timeline"]}
    after = {row["id"]: row for row in delayed["timeline"]}
    assert after["r0:ag"]["ready_ms"] == before["r0:ag"]["ready_ms"] + 5.0
    assert after["r0:ag"]["complete_ms"] == before["r0:ag"]["complete_ms"] + 5.0
    assert baseline["execution"]["serviced"].get("link") == \
        delayed["execution"]["serviced"].get("link")
    observations = delayed["execution"]["collective_readiness_observations"]
    assert len(observations) == 1
    assert observations[0]["op_id"] == "ag"
    assert observations[0]["context"][0]["readiness_basis"] == "pipeline_engine_event"
    assert observations[0]["context"][0]["entry_ns"] == READINESS_ENTRY_NS

def test_cross_dp_distinct_participant_ready_is_retained():
    raw = cohort_case(2)
    raw["runtime"]["kernels"] = {"families": {"allgather": {"kind": "ring_collective"}}}
    raw["runtime"]["metadata"] = {"collective_readiness": _report(collective="all_gather")}
    report = graph_costs( raw, {0: (_op("producer", flops=1e6), _gather(axis="ep")),
              1: (_op("producer", flops=1e6, deps=("input",)), _gather(axis="ep"))},
        external_inputs={"input": 5.0}, include_submission=False)
    rows = {row["id"]: row for row in report["timeline"]}
    assert rows["r0:ag"]["ready_ms"] == 1.0
    assert rows["r1:ag"]["ready_ms"] == 6.0
    assert len(report["execution"]["collective_readiness_observations"]) == 1
    assert report["execution"]["collective_readiness_unresolved"] == []
    context = report["execution"]["collective_readiness_observations"][0]["context"]
    assert sorted(context) == [0, 1]

def test_wire_to_logical_input_basis_is_primitive_and_group_specific():
    gather = comm_op(op_id="ag4", deps=(), stage=0, layer=0, requests=(),
                     family="allgather", phase="prefill", recipe="ring_allgather",
                     link_bytes=30720, shape=(2, 5120),
                     metadata=(("collective", "allgather"), ("group_axis", "tp"), ("payload_dtype", "bf16")))
    scatter = comm_op(op_id="rs2", deps=(), stage=0, layer=0, requests=(), family="reduce_scatter", phase="prefill",
                      recipe="ring_reduce_scatter", link_bytes=5120, shape=(2, 5120),
                      metadata=(("collective", "reduce_scatter"), ("group_axis", "tp"), ("payload_dtype", "bf16")))
    assert _collective_native_basis(gather, 4)[1] == READINESS_TYPED_BYTES
    assert _collective_native_basis(scatter, 2)[1] == READINESS_TYPED_BYTES
    assert _collective_native_basis(gather, 4)[2] == "bf16"

def test_startup_protocol_resolves_rounds_and_rejects_undeclared():
    assert _startup(64).startup_rounds == 63
    assert _startup(128).startup_rounds == 127
    assert _startup(64, runtime="log2").startup_rounds == 6
    assert _startup(128, runtime="log2").startup_rounds == 7
    assert _startup(3, runtime="log2").startup_rounds == 2
    assert _startup(5, runtime="log2").startup_rounds == 3
    concurrent = _startup(8, runtime="concurrent")
    assert concurrent.startup_rounds == 1 and concurrent.peer_phases == 7
    declared = _startup(64, runtime="topology", protocols={"alltoall": "log2"})
    assert declared.origin == "topology_protocol"
    with pytest.raises(ModelingError, match="collective_startup_protocol_undeclared"):
        _startup(64, runtime="topology")
    with pytest.raises(ModelingError, match="collective_startup_family_unsupported"):
        resolve_collective_startup(family="allreduce", group_size=4, metadata={},
                                   runtime_mode="legacy", topology_protocols={})
    assert _startup(1).startup_rounds == 0 and _startup(1).peer_phases == 0

@pytest.mark.parametrize("bad", (0, 4, -1, True, 1.5, "2", float("nan")))
def test_invalid_steps_are_rejected_before_zero_volume(bad):
    with pytest.raises(ModelingError, match="communication_round_count_invalid"):
        _startup(4, steps=bad)
    with pytest.raises(ModelingError, match="communication_round_count_invalid"):
        Topology(hardware(4), ParallelSpec(tp=4)).accounting( _exchange(0.0, steps=bad), 0)

def test_steps_change_only_startup_and_conserve_peer_work():
    topology = Topology(hardware(4), ParallelSpec(tp=4))
    rows = [topology.accounting(_exchange(30.0, steps=steps), 0) for steps in (1, 2, 3)]
    legacy = topology.accounting(_exchange(30.0), 0)
    assert [sum(row.useful_by_rank.values()) for row in rows] == [120.0] * 3
    assert rows[0].wire_by_resource == rows[1].wire_by_resource == rows[2].wire_by_resource
    assert [row.messages for row in rows] == [12, 12, 12]
    assert rows[2].setup_ms >= rows[0].setup_ms
    assert legacy.setup_ms == pytest.approx(rows[2].setup_ms)
    single = Topology(hardware(1), ParallelSpec(tp=1)).accounting(_exchange(0.0), 0)
    assert single.messages == 0 and single.setup_ms == 0
    with pytest.raises(ModelingError, match="communication_round_count_invalid"):
        _startup(1, steps=1)

@pytest.mark.parametrize("family,rounds,expected", (
    ("allgather", 3, 13), ("reduce_scatter", 3, 13), ("allreduce", 6, 26)))
def test_ring_recurrences_do_not_create_a_global_barrier(family, rounds, expected):
    raw = _ring_case()
    report = graph_costs(raw, (_exchange(rounds, family=family),),
                         recipes=RecipeBook(families={family: RingCollective()}),
                         trace_limit=200)
    oracle = _independent_ring_arrivals(rounds, [1] * 4, [10, 0, 0, 0])
    assert report["duration_ms"] == max(oracle) == expected
    assert _wire_total(report) == 4 * rounds
    assert report["bounds"]["combined_ms"] <= report["duration_ms"]
    early = next(row for row in report["events"] if ":rank2:round1:hop0:service" in row["task"])
    late = next(row for row in report["events"] if ":rank0:round0:hop0:arrival" in row["task"])
    assert early["start_ms"] == 1 < late["end_ms"] == 11
    aggregate = graph_costs(raw, (_exchange(rounds, family=family),))
    assert aggregate["duration_ms"] == rounds * 11
    assert aggregate["execution"]["serviced"] == report["execution"]["serviced"]

def test_competing_collectives_share_links_without_one_global_barrier():
    raw = _ring_case()
    first = _exchange(3, family="allgather")
    second = replace(first, op_id="second", stream="other")
    report = graph_costs(raw, (first, second), recipes=RecipeBook(families={"allgather": RingCollective()}))
    assert report["duration_ms"] == max( _independent_ring_arrivals(3, [2] * 4, [10, 0, 0, 0])) == 16
    assert _wire_total(report) == 24
    assert [row["complete_ms"] for row in report["timeline"]] == [16, 16]
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0

def test_ring_framing_keeps_transaction_tail_and_latency_once():
    raw = case(tp=2)
    raw["machine"]["links"][0].update( semantics="per_pair", transaction_bytes=16, protocol_overhead_bytes=4,
        latency_us=1000)
    op = replace(_exchange(34, family="allreduce"), deps=("upstream",))
    report = graph_costs(raw, (op,), recipes=RecipeBook(families={ "allreduce": RingCollective()}), input_ready_ms=5,
        external_inputs={"upstream": 8}, trace_limit=100)
    assert report["duration_ms"] == pytest.approx(3 + 2 * 1.04)
    assert report["complete_ms"] == pytest.approx(10.08)
    assert _wire_total(report) == 160
    assert report["isolated"]["operators"][0]["communication"]["wire_bytes"] == 160
    assert all(row["start_ms"] >= 8 for row in report["events"] if row["demand"])

def test_declared_collective_kind_keeps_the_same_ring_framing():
    raw = case(tp=2)
    raw["machine"]["links"][0].update( semantics="per_pair", transaction_bytes=16, protocol_overhead_bytes=4)
    op = _exchange(34, family="transport", metadata=(("collective", "allreduce"),))
    report = graph_costs(raw, (op,), recipes=RecipeBook( families={"transport": RingCollective()}))
    communication = report["isolated"]["operators"][0]["communication"]
    assert communication["messages"] == 4
    assert communication["wire_bytes"] == 160
    assert sum(report["execution"]["serviced"].values()) == 160

def test_multihop_forwarding_conserves_hop_work():
    raw = case(tp=2)
    raw["machine"]["num_devices"] = 3
    raw["machine"]["links"] = [ {"name": "a", "members": [0, 2], "capacity_bytes_per_ms": 1000,
         "transaction_bytes": 16, "protocol_overhead_bytes": 4, "latency_us": 200},
        {"name": "b", "members": [2, 1], "capacity_bytes_per_ms": 1000,
         "transaction_bytes": 32, "protocol_overhead_bytes": 8, "latency_us": 300}]
    op = _exchange(34, family="allreduce")
    with pytest.raises(ValueError, match="multihop_requires_store_forward"):
        graph_costs(raw, (op,), recipes=RecipeBook( families={"allreduce": RingCollective()}))
    report = graph_costs(raw, (op,), recipes=RecipeBook(families={
        "allreduce": RingCollective(path_model="store_forward")}), trace_limit=200)
    assert report["duration_ms"] == pytest.approx(2 * (.04 + .2 + .04 + .3))
    assert report["execution"]["serviced"]["link:a"] == 160
    assert report["execution"]["serviced"]["link:b"] == 160
    stages = {row["task"].split("rank0:round0:")[-1]: row for row in report["events"]
              if ":rank0:round0:" in row["task"]}
    assert stages["hop1:service"]["start_ms"] == stages["hop0:arrival"]["end_ms"]

def test_public_ring_binding_reaches_graph_and_serving():
    raw = _ring_case()
    raw["runtime"]["kernels"] = {"families": {
        family: {"kind": "ring_collective"} for family in ("allreduce", "allgather")}}
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="model")
    graph = graph_costs(raw, graphs)
    serving = predict(raw, event_limit=1000)
    wire = lambda report: {name: value for name, value in report["execution"]["serviced"].items()
                           if name.startswith("link:")}
    assert wire(graph) == wire(serving)
    assert any(row["program"] == "RingCollective" for row in graph["timeline"])
    assert any(":round1:" in row["task"] for row in serving["events"])
    assert serving["status"] == "modeled"
    assert graph["execution"]["live_tasks"] == serving["execution"]["live_tasks"] == 0

def test_repeated_ring_steps_keep_active_messages_bounded():
    parsed = parse_case(_ring_case())
    topology = Topology(parsed.machine, parsed.runtime.parallel)
    group = topology.ring_plan(_exchange(3, family="allgather"), 0).groups[0]
    peaks = []
    for repetitions in (100, 1000):
        engine = Engine(topology.resources())
        complete = replace(group, rounds=repetitions).start( engine, "loop", engine.submit("ready"))
        engine.advance(until=(complete,))
        assert complete.time == max(_independent_ring_arrivals( repetitions, [1] * 4, [10, 0, 0, 0]))
        peaks.append(engine.peak_tasks)
        assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert peaks[0] == peaks[1]

def test_cross_replica_ring_executes_complete_cohort_once():
    raw = case(dp=2, ep=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    op = _exchange(2, family="allreduce", axis="ep")
    report = graph_costs(raw, {0: (op,), 1: (op,)}, recipes=RecipeBook(
        families={"allreduce": RingCollective()}), trace_limit=100)
    assert report["duration_ms"] == pytest.approx(.004)
    assert report["execution"]["serviced"]["link:fabric"] == 4
    assert report["isolated"]["operator_sum_ms"] == pytest.approx(.008)
    assert sum(row["communication"]["wire_bytes"] for row in report["isolated"]["operators"]) == 4
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    with pytest.raises(ValueError, match="requires_complete_cohort"):
        graph_costs(raw, (op,), recipes=RecipeBook( families={"allreduce": RingCollective()}))

def _wave_topology(ranks: int) -> Topology:
    return Topology(hardware(ranks), ParallelSpec(tp=ranks))

def _run_waves(ranks: int, rounds: int, volumes=None):
    topology = _wave_topology(ranks)
    volumes = volumes or tuple((rank, 30.0) for rank in range(ranks))
    group = WaveGroup(tuple(range(ranks)), volumes, rounds, topology)
    engine = Engine(topology.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in range(ranks)}
    done = group.start_participants(engine, f"waves{ranks}-{rounds}", ready)
    engine.advance(until=tuple(done.values()), first=False)
    return engine, done

def test_startup_phases_partition_every_offset_exactly_once():
    for size in (2, 3, 4, 5, 8):
        for rounds in range(1, size):
            ranges = wave_ranges(size, rounds)
            offsets = [offset for _wave, offset in wave_offsets(size, rounds)]
            assert offsets == list(range(1, size))
            assert len(ranges) == rounds
            assert all(start <= end for start, end in ranges)
            for left, right in zip(ranges, ranges[1:]):
                assert left[1] < right[0]
    with pytest.raises(ModelingError, match="communication_round_count_invalid"):
        wave_ranges(4, 4)

def test_waves_conserve_peer_bytes_and_bound_inflight():
    serial_engine, serial_done = _run_waves(4, 3)
    concurrent_engine, concurrent_done = _run_waves(4, 1)
    assert all(event.done for event in (*serial_done.values(), *concurrent_done.values()))
    serial_bytes = sum(value for name, value in serial_engine.serviced.items() if name.startswith("link:"))
    concurrent_bytes = sum(value for name, value in concurrent_engine.serviced.items() if name.startswith("link:"))
    assert serial_bytes == pytest.approx(concurrent_bytes) == pytest.approx(120.0)
    assert serial_engine.now == pytest.approx(concurrent_engine.now)

def test_startup_phases_preserve_reverse_variable_traffic():
    topology = _wave_topology(3)
    peer = PeerTraffic.from_mapping({ "kind": "matrix", "payload_bytes": [[0, 5, 7], [3, 0, 11], [13, 17, 0]],
        "direction": "forward"}, 3)
    volumes = tuple((rank, peer.row_total(rank)) for rank in range(3))
    group = WaveGroup(tuple(range(3)), volumes, 1, topology, peer_traffic=peer)
    engine = Engine(topology.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in range(3)}
    done = group.start_participants(engine, "reverse", ready)
    engine.advance(until=tuple(done.values()), first=False)
    assert all(event.done for event in done.values())
    assert sum(value for name, value in engine.serviced.items() if name.startswith("link:")) == pytest.approx(56.0)

def test_zero_payload_idle_peer_still_rendezvous():
    engine, done = _run_waves(3, 1, volumes=((0, 0.0), (1, 60.0), (2, 0.0)))
    assert all(event.done for event in done.values())
    assert engine.serviced.get("link:fabric", 0.0) == pytest.approx(60.0)

def test_startup_gate_is_a_real_event_and_unknown_is_explicit():
    report_doc = {"schema": READINESS_SCHEMA, "calls": [ {"rank_id": f"rank_{rank}", "rank_index": rank,
         "operation": {"dtype": "BFP16", "typed_bytes": 1024},
         "entry": {"start_ns": 1}, "completion": {"end_ns": 2},
         "observed_context": {}, "clock": {}, "dependencies": (), "waits": (),
         "unknowns": ()} for rank in (0, 1)]}

    def plan():
        return SimpleNamespace(start_participants=lambda engine, name, ready, owner="": {
            rank: engine.submit(f"{name}:{rank}", after=(event,))
            for rank, event in ready.items()})

    def run(startup_service=None):
        engine = Engine([], trace_limit=16)
        ready = {rank: engine.submit(f"r{rank}") for rank in (0, 1)}
        kwargs = {} if startup_service is None else {"startup_service": startup_service}
        completions, context = start_with_pipeline( report_doc, engine, plan(), ready, **kwargs)
        engine.advance(until=tuple(completions.values()), first=False)
        return completions, context

    base, _ = run()
    submitted, context = run({"startup_us": 2.0, "source": "declared-startup"})
    assert context[0]["protocol_startup"]["status"] == "submitted"
    assert context[0]["protocol_startup"]["startup_us"] == pytest.approx(2.0)
    assert submitted[0].time == pytest.approx(base[0].time + 0.002)
    skew = readiness_skew(context)
    assert skew["status"] == "pipeline_events"
    assert skew["skew_ms"] == pytest.approx(0.0)
    unknown, unknown_context = run()
    assert unknown_context[0]["protocol_startup"]["status"] == "unknown_not_submitted"
    assert unknown[0].time == pytest.approx(base[0].time)

def test_envelope_rounds_add_only_startup_not_wire_work():
    raw = case(tp=4)
    raw["machine"]["links"][0].update( semantics="per_pair", capacity_bytes_per_ms=1000, latency_us=1000)
    durations = []
    for family, extra in (("alltoall", ()), ("alltoallv", (("collective", "alltoall"),))):
        op = _exchange(30, family=family, steps=2, metadata=extra)
        result = graph_costs(raw, (op,))
        durations.append(result["duration_ms"])
        assert result["duration_ms"] == pytest.approx(.01 + 2)
        ledger = result["isolated"]["operators"][0]["communication"]
        assert ledger["setup_ms"] == 2
        assert ledger["wire_bytes"] == 120
    assert durations[0] == durations[1]

@pytest.mark.parametrize("kind,options,expected", ( ("pairwise_collective", {}, 3.03),
    ("direct_peer_collective", {"max_inflight_per_rank": 3}, 1.01)))
def test_explicit_exchange_programs_charge_their_own_hop_arrivals(kind, options, expected):
    raw = case(tp=4)
    raw["machine"]["links"][0].update( semantics="per_pair", capacity_bytes_per_ms=1000, latency_us=1000)
    raw["runtime"]["kernels"] = {"families": {"alltoall": {"kind": kind, **options}}}
    result = graph_costs(raw, (_exchange(30, steps=3),))
    assert result["duration_ms"] == pytest.approx(expected)
    assert _wire_total(result) == 120
    assert result["execution"]["live_tasks"] == 0

def test_signal_startup_is_one_event_per_hop_not_propagation():
    links = (Link("first", 1000, (0, 1), 1000.0, "fabric", "shared"),
             Link("second", 1000, (1, 2), 1000.0, "fabric", "shared"))
    hardware = Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950DT_9581", "num_devices": 3})
    topology = Topology(replace(hardware, links=links), ParallelSpec(tp=1))
    engine = Engine(topology.resources(), trace_limit=100)
    ready = engine.submit("ready")
    done = SignalGroup((0, 1, 2), topology, ((0, 2),), signal_bytes=0, hop_startup_us=2).start_participants(
                           engine, "signal", {rank: ready for rank in (0, 1, 2)})
    engine.advance(until=tuple(done.values()), first=False)
    assert done[2].time == pytest.approx(2 + 2 * 0.002)
    startup = [event for event in engine.trace if event["task"].endswith(":startup")]
    assert len(startup) == 2
    assert all(event["demand"] == {} for event in startup)

def test_transport_options_mapping_binds_every_declared_admission_field():
    options = TransportOptions.from_mapping({
        "chunk_bytes": 64, "hop_startup_us": 2, "receive_capacity_bytes": {1: 128},
        "receive_buffer_ids": {1: "pool"}, "receive_capacity_bytes_per_peer": 32,
        "receive_buffer_id": "peer", "max_inflight_per_rank": 2,
        "max_physical_chunks": 8})
    assert options.plan_kwargs() == {
        "max_physical_chunks": 8, "chunk_bytes": 64, "hop_startup_us": 2,
        "receive_capacity_bytes": {1: 128}, "receive_buffer_ids": {1: "pool"},
        "receive_capacity_bytes_per_peer": 32, "receive_buffer_id": "peer",
        "max_inflight_per_rank": 2}
    assert TransportOptions.from_mapping(None) == TransportOptions()
    assert TransportOptions().plan_kwargs() == {
        "max_physical_chunks": 4096, "max_inflight_per_rank": None}
    with pytest.raises(ModelingError, match="transport_option_unknown"):
        TransportOptions.from_mapping({"unknown": 1})

def test_transport_resource_work_counts_declared_transactions_and_frames():
    link = Link("f", 1000, (0, 1), 0.0, "fabric", "shared", transaction_bytes=64, protocol_overhead_bytes=0)
    topology = Topology(replace(hardware(2), links=(link,)), ParallelSpec(tp=1))
    plan = topology.transport_plan([MessageFlow(0, 1, 128)], chunk_bytes=64)
    work = plan.resource_work()
    assert work["transactions"] == {"link:f": 2}
    assert work["frames"] == {"link:f": (64, 64)}

def test_transport_receive_pool_conflict_is_explicit():
    link = Link("f", 1000, (0, 1), 0.0, "fabric", "shared")
    topology = Topology(replace(hardware(2), links=(link,)), ParallelSpec(tp=1))
    first = topology.transport_plan([MessageFlow(0, 1, 8)], receive_capacity_bytes={1: 64},
                                    receive_buffer_id={1: "shared"})
    second = topology.transport_plan([MessageFlow(0, 1, 8)], receive_capacity_bytes={1: 128},
                                     receive_buffer_id={1: "shared"})
    engine = Engine(first.resources())
    ready = {rank: engine.submit(f"r{rank}") for rank in first.ranks}
    first.start(engine, "a", ready)
    with pytest.raises(ModelingError, match="transport_receive_capacity_conflict"):
        second.start(engine, "b", ready)


def test_collective_algorithm_and_hccs_k8_alltoall_staging_independent_of_calibration():
    """Physical topology picks double_ring and hccs_k8 AllToAll HBM staging without primitive_table."""
    from mindie_sim import compose_case, parse_case
    from mindie_sim.prediction.collectives.protocol import HCCL_DEFAULT_BUFFSIZE_BYTES, lower_collective
    from mindie_sim.primitives.loader import load_primitives

    cs_a3 = parse_case(compose_case("qwen3.8-27b", hardware="Ascend910_9362", tp=4, ep=1, num_devices=4, layers=1, quant="none"))
    topo_uncal = Topology(cs_a3.machine, cs_a3.runtime.parallel, primitive_table=None)
    topo_cal = Topology(cs_a3.machine, cs_a3.runtime.parallel, primitive_table=load_primitives("Ascend910_9362"))
    low_uncal = lower_collective("allreduce", (0, 1, 2, 3), topo_uncal, 4 * 1024 * 1024)
    low_cal = lower_collective("allreduce", (0, 1, 2, 3), topo_cal, 4 * 1024 * 1024)
    assert low_uncal.band.algorithm == "double_ring" == low_cal.band.algorithm

    cs_a2 = parse_case(compose_case("qwen3.8-27b", hardware="Ascend910B4", tp=4, ep=1, num_devices=4, layers=1, quant="none"))
    topo_a2 = Topology(cs_a2.machine, cs_a2.runtime.parallel, primitive_table=None)
    low_a2a = lower_collective("alltoall", (0, 1, 2, 3), topo_a2, 4 * 1024 * 1024)
    assert low_a2a.local_hbm_bytes_per_rank == pytest.approx(4.0 * 4 * 1024 * 1024)
    low_direct = lower_collective("alltoall", (0, 1, 2, 3), topo_a2, HCCL_DEFAULT_BUFFSIZE_BYTES + 1024 * 1024)
    assert low_direct.local_hbm_bytes_per_rank == 0.0

    from mindie_sim import operator_costs
    from mindie_sim.prediction.call_builders import build_aclnn_call
    from mindie_sim.prediction.evaluation import EvaluationContext
    raw_ep4 = compose_case("deepseek-v2-lite", "Ascend910B4", tp=1, dp=4, pp=1, ep=4)
    op_a2a = replace(
        build_aclnn_call({"api": "aclnnMatmul", "dtype": "bf16", "m": 16, "n": 256, "k": 1024}).ops[0],
        op_id="a2a_iso", family="alltoall", recipe="alltoall", flops=0, matrix_flops=(),
        vector_ops=0, hbm_bytes=0, mte2_bytes=0, mte3_bytes=0,
        link_bytes=3 * 1048576, link_scope="ep", metadata=(("group_axis", "ep"), ("collective", "alltoall")),
    )
    ev_cal = EvaluationContext.create(parse_case(raw_ep4), primitive_table=load_primitives("Ascend910B4"))
    for ev in (None, ev_cal):
        serviced = operator_costs(raw_ep4, [op_a2a], evaluation=ev)["operators"][0]["serviced"]
        links = {k: v for k, v in serviced.items() if k.startswith("link:")}
        expected_links = {
            f"link:d2d_intra_node:0:pair{src}>{dst}"
            for peer in (1, 2, 3) for src, dst in ((0, peer), (peer, 0))
        }
        assert set(links) == expected_links and all(v == pytest.approx(1048576.0) for v in links.values())
        assert serviced.get("memory@r0:s0", 0.0) == pytest.approx(16 * 1048576.0)


def test_superpod_optical_alltoall_bands_and_startup_boundaries():
    """Verify A3 SuperPoD 32-die optical AllToAll/AllToAllV direct AIV band matching and route counts."""
    from _support.axes import REPRESENTATIVE_SKUS
    from mindie_sim.compose import compose_case
    from mindie_sim.prediction.collectives.protocol import lower_collective, select_band
    from mindie_sim.prediction.collectives.topology import Topology
    from mindie_sim.prediction.structure import parse_case

    raw = compose_case(
        "qwen3.5-397b-a17b", REPRESENTATIVE_SKUS["a3"], topology_preset="a3_superpod",
        tp=16, dp=2, ep=32, layers=1, quant="w8a8",
    )
    parsed = parse_case(raw)
    topo = Topology(parsed.machine, parsed.runtime.parallel, primitive_table=None)
    ep32 = tuple(range(32))

    # 1. Default and explicit AIV select direct_mesh with 1 phase startup (3.5 us) across all 3 band boundaries.
    for coll in ("alltoall", "alltoallv"):
        for exp in (None, "aiv"):
            for payload, proto in (
                (65536, "hccs_optical_small"),
                (65537, "hccs_optical_mid"),
                (2097151, "hccs_optical_mid"),
                (2097152, "hccs_optical_large"),
            ):
                band = select_band(coll, ep32, topo, payload, expansion_mode=exp)
                assert band.algorithm == "direct_mesh"
                assert band.expansion_mode == "aiv"
                assert band.protocol == proto
                assert band.startup_scope == "phase"
                assert band.startup_us == pytest.approx(3.5)

    # 2. Explicit ring retains 31 serialized rounds on balanced alltoall; forced-ring allreduce preserved.
    ring_band = select_band("alltoall", ep32, topo, 65536, expansion_mode="ring")
    assert ring_band.algorithm == "ring"
    low_ring = lower_collective("alltoall", ep32, topo, 65536, expansion_mode="ring")
    assert len(low_ring.phases) == 1 and low_ring.phases[0].rounds == 31
    ar_ring = select_band("allreduce", ep32, topo, 65536, expansion_mode="ring")
    assert ar_ring.algorithm == "ring"

    # 3. Balanced EP32 matrix routes 32 SIO, 448 intra-node HCCS, 512 optical pairs in 1 round (3.5 us startup).
    matrix = tuple(tuple(0 if r == c else 1024 for c in range(32)) for r in range(32))
    splits = tuple(sum(row) for row in matrix)
    low_ep = lower_collective(
        "alltoallv", ep32, topo, float(max(splits)),
        peer_matrix=matrix, send_splits=splits, recv_splits=splits,
    )
    assert low_ep.band.algorithm == "direct_mesh"
    assert len(low_ep.phases) == 1 and low_ep.phases[0].rounds == 1
    assert low_ep.phases[0].startup_us == pytest.approx(3.5)
    tier_counts = {"d2d_intra_package": 0, "d2d_intra_node": 0, "d2d_inter_node": 0}
    for tr in low_ep.phases[0].transfers:
        assert len(tr.hops) == 1
        ch = topo.links[tr.hops[0].link_name].channel
        tier_counts[ch] += 1
    assert tier_counts == {"d2d_intra_package": 32, "d2d_intra_node": 448, "d2d_inter_node": 512}

    # 4. Zero remote traffic yields 0 phases; both TP16 communicators stay intra-node (0 optical bytes).
    zero_m = tuple(tuple(0 for _ in range(32)) for _ in range(32))
    low_zero = lower_collective(
        "alltoallv", ep32, topo, 0.0,
        peer_matrix=zero_m, send_splits=(0,) * 32, recv_splits=(0,) * 32,
    )
    assert low_zero.phases == ()
    for tp_group in (tuple(range(16)), tuple(range(16, 32))):
        low_tp = lower_collective("allreduce", tp_group, topo, 65536)
        optical_transfers = [
            tr for ph in low_tp.phases for tr in ph.transfers
            for hop in tr.hops if topo.links[hop.link_name].channel == "d2d_inter_node"
        ]
        assert optical_transfers == []
