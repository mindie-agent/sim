"""A5 communication leaves on the common topology/resource program.

These tests establish independent oracles for the pieces added by this slice:

* a declared per-hop startup reaches the HCCL remote-notify leaf exactly once
  per hop and is not confused with propagation;
* the public A5 CCU two-die allgather leaf keeps the same useful bytes while
  its route/sharing domain and completion change with the placement;
* the HCCL AlltoAllV leaf can use the common transport plan for chunk,
  receive-credit and startup while preserving local HBM/copy/control work;
* the P/D connector can use the same plan, preserving wire framing and
  exposing receiver credit without inventing extra payload work.
"""
from dataclasses import replace

import pytest

from mindie_sim import compose_case, graph_costs, predict_pd
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.hccl_alltoallv import AlltoAllVGroup
from mindie_sim.prediction.hccl_contracts import select
from mindie_sim.prediction.hccl_graph import lower as hccl_lower
from mindie_sim.prediction.hccl_signals import SignalGroup
from mindie_sim.prediction.pd_connector import PDConnector, PDHandoff, PDTransfer
from mindie_sim.prediction.program import RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware, Link
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec

from test_collective_cohorts import per_pair_topology
from test_hccl_ccu_two_die import TEMPLATE, channel_map
from test_prediction_deployment import pd_case
from test_prediction_system import linear_work


def _a5_ccu_case(preset):
    raw = compose_case(
        "qwen3.5-35b-a3b", hardware="Ascend950DT_9581",
        topology_preset=preset, tp=8, dp=1, ep=8, layers=1,
        prompt_tokens=8, output_tokens=1, concurrency=1)
    device = raw["machine"]["device"]
    # This A5 CCU source leaf needs its declared endpoint services; the A5
    # profile does not publish CCU rates and must not have them inferred.
    device["rates"].update({"hccl_control": 1e9, "ccu_copy": 1e9,
                            "ccu_buffer_bytes": 1e12, "ccu_control": 1e9})
    device["resource_units"].update({"hccl_control": 2, "ccu_copy": 2,
                                    "ccu_buffer_bytes": 2, "ccu_control": 2})
    raw["runtime"]["kernels"] = {"hccl": {"allgather": {
        "expansion": "CCU_MS", "template": TEMPLATE,
        "topology": "TWO_DIE_REGULAR",
        "channel_die_map": channel_map(8), "signal_bytes": 8}}}
    return raw


def _allgather(volume, *, startup_us=None):
    metadata = (("group_axis", "tp"),)
    if startup_us is not None:
        metadata += (("transport_hop_startup_us", startup_us),)
    return OpSpec("ag", "allgather", "decode", 0, 0, (), (1,), (1,), (1,), (1,),
                  link_bytes=volume, metadata=metadata)


def _a5_ccu_report(raw, *, startup_us=None):
    parsed = parse_case(raw)
    selected = hccl_lower((_allgather(7 * 30, startup_us=startup_us),),
                          parsed.runtime, parsed.model.activation_dtype)
    return graph_costs(raw, selected, input_ready_ms=7, trace_limit=20000)


def test_signal_startup_is_one_per_hop_and_not_another_propagation():
    links = (Link("first", 1000, (0, 1), 1000.0, "fabric", "shared"),
             Link("second", 1000, (1, 2), 1000.0, "fabric", "shared"))
    hardware = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950DT_9581", "num_devices": 3})
    topology = Topology(replace(hardware, links=links), ParallelSpec(tp=1))
    engine = Engine(topology.resources(), trace_limit=100)
    ready = engine.submit("ready")
    group = SignalGroup((0, 1, 2), topology, ((0, 2),), signal_bytes=0,
                        hop_startup_us=2)
    done = group.start_participants(
        engine, "signal", {rank: ready for rank in (0, 1, 2)})
    engine.advance(until=tuple(done.values()), first=False)

    # Two propagation hops are 2 ms; the declared startup is two 2 us gates,
    # not a third propagation delay and not a call-total multiplier.
    assert done[2].time == pytest.approx(2 + 2 * 0.002)
    startup = [event for event in engine.trace
               if event["task"].endswith(":startup")]
    assert len(startup) == 2
    assert all(event["demand"] == {} for event in startup)


def test_public_a5_ccu_leaf_changes_route_sharing_and_completion_with_placement():
    reports = {}
    for preset in ("a5_server", "a5_pod"):
        raw = _a5_ccu_case(preset)
        report = _a5_ccu_report(raw, startup_us=0.5)
        reports[preset] = report
        assert report["execution"]["live_tasks"] == 0
    server = {name for name in reports["a5_server"]["execution"]["serviced"]
              if name.startswith("link:")}
    pod = {name for name in reports["a5_pod"]["execution"]["serviced"]
           if name.startswith("link:")}
    assert any("d2d_intra_4p" in name for name in server)
    assert any("d2d_intra_node" in name for name in server)
    assert not any("d2d_intra_4p" in name for name in pod)
    assert any("d2d_intra_node" in name for name in pod)
    # The same complete allgather payload is conserved, but its physical route
    # and shared-domain service differ under the A5 placement.
    server_wire = sum(value for name, value in
                      reports["a5_server"]["execution"]["serviced"].items()
                      if name.startswith("link:"))
    # A per-port link owns the same payload on its TX and RX endpoints; the
    # per-pair route owns one directed pair. Compare one direction.
    pod_wire = sum(value for name, value in
                   reports["a5_pod"]["execution"]["serviced"].items()
                   if name.startswith("link:") and ":tx:port" in name)
    assert server_wire == pytest.approx(pod_wire)
    assert reports["a5_server"]["duration_ms"] != reports["a5_pod"]["duration_ms"]
    # The declared T06 startup is a separate gate, never folded into link work.
    assert any(event["task"].endswith(":startup")
               for event in reports["a5_server"]["events"])


def test_alltoallv_leaf_uses_common_plan_for_chunk_credit_and_startup():
    count, cell = 4, 30
    matrix = tuple(tuple(0 if src == dst else cell for dst in range(count))
                   for src in range(count))
    topology = per_pair_topology(count, capacity=64, latency_us=0)
    selection = select({"expansion": "AICPU_TS", "template": "mesh_write",
                        "dtype_bytes": 1}, collective="alltoallv",
                       ranks=count, payload_bytes=cell * (count - 1))

    def contexts():
        resources = list(topology.resources())
        result = {}
        for rank in range(count):
            rates = {"hbm": 1e9, "hccl_copy": 1e9, "hccl_control": 1e9}
            names = {role: f"{role}:{rank}" for role in rates}
            resources.extend(Resource(names[role], rate) for role, rate in rates.items())
            result[rank] = RecipeContext(
                None, {}, names, {names[role]: rate for role, rate in rates.items()},
                1, resource_units={name: 1 for name in names.values()})
        return resources, result

    def run(group, tag):
        resources, bindings = contexts()
        engine = Engine(resources, trace_limit=5000)
        ready = {rank: engine.submit(f"{tag}:ready{rank}") for rank in range(count)}
        done = group.start_participants(engine, tag, ready, bindings)
        engine.advance(until=tuple(done.values()), first=False)
        assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
        return done, engine

    legacy_group = AlltoAllVGroup(selection, matrix, tuple(range(count)), topology)
    legacy_done, legacy_engine = run(legacy_group, "legacy")
    plan_group = AlltoAllVGroup(
        selection, matrix, tuple(range(count)), topology,
        chunk_bytes=16, hop_startup_us=1,
        receive_capacity_bytes_per_peer=16)
    plan_done, plan_engine = run(plan_group, "plan")

    # Local HBM/copy/control work is independent of the transport option.
    for role in ("hbm", "hccl_copy", "hccl_control"):
        assert sum(value for name, value in plan_engine.serviced.items()
                   if name.startswith(role + ":")) == pytest.approx(
                       sum(value for name, value in legacy_engine.serviced.items()
                           if name.startswith(role + ":")))
    # Chunks and per-hop startup are real tasks; the plan is not a duration
    # multiplier over the legacy per-peer service.
    plan_tasks = [row for row in plan_engine.trace if ":transport:flow0:" in row["task"]]
    assert any(":chunk1:" in row["task"] for row in plan_tasks)
    assert any(row["task"].endswith(":chunk0:hop0:startup") for row in plan_tasks)
    assert not any(":chunk1:hop0:startup" in row["task"] for row in plan_tasks)
    assert max(event.time for event in plan_done.values()) > max(
        event.time for event in legacy_done.values())
    # Each directed payload is still charged once; chunking does not add bytes.
    legacy_wire = sum(value for name, value in legacy_engine.serviced.items()
                      if name.startswith("link:"))
    plan_wire = sum(value for name, value in plan_engine.serviced.items()
                    if name.startswith("link:"))
    assert plan_wire == pytest.approx(legacy_wire)


@pytest.mark.parametrize("empty_consumer", (False, True))
def test_alltoallv_real_buffer_and_consumer_release_reach_t09_plan(empty_consumer):
    count, cell = 2, 64
    matrix = tuple(tuple(0 if src == dst else cell for dst in range(count))
                   for src in range(count))
    topology = per_pair_topology(count, capacity=64, latency_us=0)
    selection = select({"expansion": "AICPU_TS", "template": "mesh_write",
                        "dtype_bytes": 1}, collective="alltoallv",
                       ranks=count, payload_bytes=cell)
    resources = list(topology.resources())
    contexts = {}
    for rank in range(count):
        rate = 1e9
        names = {"hbm": f"hbm:{rank}"}
        resources.append(Resource(names["hbm"], rate))
        contexts[rank] = RecipeContext(
            None, {}, names, {names["hbm"]: rate}, 1,
            resource_units={names["hbm"]: 1})
    engine = Engine(resources, trace_limit=1000)
    ready = {rank: engine.submit(f"ready{rank}") for rank in range(count)}
    release = {rank: engine.future(f"release{rank}") for rank in range(count)}
    group = AlltoAllVGroup(
        selection, matrix, tuple(range(count)), topology,
        chunk_bytes=32, receive_capacity_bytes_per_peer=64,
        receive_buffer_ids=((0, "channel0"), (1, "channel1")))
    done = group.start_participants(
        engine, "hccl", ready, contexts,
        release_events={rank: (() if empty_consumer else (release[rank],))
                        for rank in range(count)})
    engine.advance(until=tuple(done.values()), first=False)
    if empty_consumer:
        tasks = {row["task"] for row in engine.trace}
        assert all(f"hccl:{rank}->{other}:receive_release" not in tasks
                   for rank, other in ((0, 1), (1, 0)))
        assert {"transport:rx:channel0", "transport:rx:channel1"} <= set(engine.pools)
        return
    consumers = tuple(
        engine.submit(f"consume{rank}", after=(done[rank],), latency=1,
                      on_complete=lambda _event, target=release[rank]:
                          None if target.done else engine.resolve(target))
        for rank in range(count))
    engine.advance(until=consumers, first=False)
    assert engine.summary()["live_tasks"] == 0
    assert all(event.done for event in done.values())
    # Real buffers appear in the T09 receive pools and release events were
    # consumed rather than replaced by arrival consumption.
    assert {"transport:rx:channel0", "transport:rx:channel1"} <= set(engine.pools)
    assert engine.pools["transport:rx:channel0"] == pytest.approx(64.0)
    assert engine.pools["transport:rx:channel1"] == pytest.approx(64.0)
    assert sum(value for name, value in engine.serviced.items()
               if name.startswith("link:")) == pytest.approx(128.0)
    assert "hccl:0->1:receive_release" in {row["task"] for row in engine.trace}


def test_declared_transport_options_reject_nonphysical_values():
    link = {"name": "handoff", "capacity_bytes_per_ms": 128}
    for raw in ({"chunk_bytes": 0}, {"receive_capacity_bytes": 0},
                {"hop_startup_us": -1}):
        with pytest.raises(ModelingError):
            PDConnector({"link": link, **raw})
    selection = select({"expansion": "AICPU_TS", "template": "mesh_write",
                        "dtype_bytes": 1}, collective="alltoallv", ranks=2,
                       payload_bytes=1)
    topology = per_pair_topology(2, capacity=64)
    with pytest.raises(ModelingError):
        AlltoAllVGroup(selection, ((0, 0), (0, 0)), (0, 1), topology,
                       chunk_bytes=-1)


@pytest.mark.parametrize("options", ({}, {"chunk_bytes": 128}))
def test_pd_common_ready_boundary_cannot_delete_work(options):
    link = {"name": "handoff", "capacity_bytes_per_ms": 128,
            "latency_us": 1000}

    def complete(*, source=None, destination=None):
        connector = PDConnector({"link": link, **options})
        engine = Engine(connector.resources())
        ready = (None if source is None else
                 engine.submit("source_ready", release=source))
        grant = (None if destination is None else
                 engine.submit("destination_ready", release=destination))
        done = connector.submit(engine, "pd:control", 128, {},
                                ready=ready, destination_ready=grant)
        engine.advance(until=(done,), first=False)
        return done.time

    # Implicit and explicit zero-time readiness are physically equivalent.
    assert complete() == pytest.approx(2.0)
    assert complete(source=0) == pytest.approx(2.0)
    # Overall readiness shift preserves the same propagation and service.
    assert complete(source=5) == pytest.approx(7.0)
    # A late destination cannot complete before both endpoints are ready.
    assert complete(source=0, destination=9) == pytest.approx(11.0)


def test_pd_exhausted_consumer_prefix_holds_future_arrivals():
    connector = PDConnector({
        "link": {"name": "handoff", "capacity_bytes_per_ms": 64,
                 "latency_us": 0},
        "chunk_bytes": 64, "receive_capacity_bytes": 256})
    engine = Engine(connector.resources(), trace_limit=1000)
    source = engine.submit("source")
    destination = engine.submit("destination")
    early = (engine.submit("consume1", latency=1.25),
             engine.submit("consume2", latency=2.25),
             engine.submit("tail_consume", latency=5.0))
    done = connector.submit(
        engine, "pd:hold", 256, {}, ready=source, destination_ready=destination,
        release_events={1: early})
    engine.advance(until=(done,), first=False)
    rows = {row["task"]: row for row in engine.trace}
    # The two early consumers drain only their arrived chunks; 128 B remains
    # held at 4.5 ms and the allocation cannot be released by exhaustion.
    assert done.time == pytest.approx(4.0)
    assert "pd:hold:receive_release:r1" not in rows
    transport = connector.summary()["transport"]
    assert transport["resident_bytes"]["1"] == pytest.approx(128.0)
    assert transport["peak_reserved_bytes"]["1"] == pytest.approx(256.0)
    engine.advance(until=(early[2],), first=False)
    rows = {row["task"]: row for row in engine.trace}
    assert rows["pd:hold:receive_release:r1"]["start_ms"] == pytest.approx(5.0)
    assert connector.summary()["transport"]["resident_bytes"]["1"] == pytest.approx(0.0)
    successor_consumer = engine.submit("successor_consume", latency=10.0)
    successor = connector.submit(
        engine, "pd:successor", 256, {}, ready=source,
        destination_ready=destination,
        release_events={1: (successor_consumer,)})
    engine.advance(until=(successor,), first=False)
    rows = {row["task"]: row for row in engine.trace}
    assert rows["pd:successor:receive_grant"]["start_ms"] == pytest.approx(5.0)
    assert successor.time == pytest.approx(9.0)


def test_pd_explicit_empty_consumer_holds_allocation():
    connector = PDConnector({
        "link": {"name": "handoff", "capacity_bytes_per_ms": 64,
                 "latency_us": 0},
        "chunk_bytes": 64, "receive_capacity_bytes": 256})
    engine = Engine(connector.resources(), trace_limit=1000)
    source = engine.submit("source")
    destination = engine.submit("destination")
    done = connector.submit(
        engine, "pd:empty", 256, {}, ready=source,
        destination_ready=destination, release_events={1: ()})
    engine.advance(until=(done,), first=False)
    tasks = {row["task"] for row in engine.trace}
    assert done.time == pytest.approx(4.0)
    assert "pd:empty:receive_release:r1" not in tasks
    assert connector.summary()["transport"]["peak_reserved_bytes"]["1"] == pytest.approx(256.0)


def test_pd_512_pool_admits_two_256_handoffs_concurrently():
    connector = PDConnector({
        "link": {"name": "handoff", "capacity_bytes_per_ms": 128,
                 "latency_us": 0},
        "chunk_bytes": 128, "receive_capacity_bytes": 512})
    engine = Engine(connector.resources(), trace_limit=1000)
    ready = {0: engine.submit("source"), 1: engine.submit("destination")}
    first_release = engine.future("first_release")
    second_release = engine.future("second_release")
    first = connector.submit(
        engine, "pd:first", 256, {}, ready=ready[0],
        destination_ready=ready[1], release_events={1: (first_release,)})
    second = connector.submit(
        engine, "pd:second", 256, {}, ready=ready[0],
        destination_ready=ready[1], release_events={1: (second_release,)})
    first_consumer = engine.submit(
        "first_consumer", after=(first,), latency=1,
        on_complete=lambda _event: None if first_release.done else engine.resolve(first_release))
    second_consumer = engine.submit(
        "second_consumer", after=(second,), latency=1,
        on_complete=lambda _event: None if second_release.done else engine.resolve(second_release))
    engine.advance(until=(first_consumer, second_consumer), first=False)
    assert first.time == pytest.approx(4.0)
    assert second.time == pytest.approx(4.0)
    assert first_consumer.time == pytest.approx(5.0)
    assert second_consumer.time == pytest.approx(5.0)
    assert connector.summary()["transport"]["peak_reserved_bytes"]["1"] == pytest.approx(512.0)


def test_pd_submit_whole_message_reservation_matches_two_handoff_control():
    connector = PDConnector({
        "link": {"name": "handoff", "capacity_bytes_per_ms": 128,
                 "latency_us": 0},
        "chunk_bytes": 128, "receive_capacity_bytes": 256})
    engine = Engine(connector.resources(), trace_limit=1000)
    ready = {0: engine.submit("source"), 1: engine.submit("destination")}
    first_release = engine.future("first_release")
    second_release = engine.future("second_release")
    first = connector.submit(
        engine, "pd:first", 256, {}, ready=ready[0],
        destination_ready=ready[1], release_events={1: (first_release,)})
    second = connector.submit(
        engine, "pd:second", 256, {}, ready=ready[0],
        destination_ready=ready[1], release_events={1: (second_release,)})
    first_consumer = engine.submit(
        "first_consumer", after=(first,), latency=1,
        on_complete=lambda _event: None if first_release.done else engine.resolve(first_release))
    second_consumer = engine.submit(
        "second_consumer", after=(second,), latency=1,
        on_complete=lambda _event: None if second_release.done else engine.resolve(second_release))
    engine.advance(until=(first_consumer, second_consumer), first=False)
    assert first.time == pytest.approx(2.0)
    assert second.time == pytest.approx(5.0)
    assert first_consumer.time == pytest.approx(3.0)
    assert second_consumer.time == pytest.approx(6.0)
    assert connector.wire_bytes == pytest.approx(512.0)
    assert connector.summary()["transport"]["peak_reserved_bytes"]["1"] == pytest.approx(256.0)


def test_pd_cross_destination_whole_message_grant_is_atomic():
    connector = PDConnector({
        "link": {"name": "handoff", "capacity_bytes_per_ms": 128,
                 "latency_us": 0},
        "chunk_bytes": 128})
    engine = Engine(connector.resources(), trace_limit=1000)
    source = engine.submit("source")
    ready = {2: engine.submit("ready2", release=0),
             3: engine.submit("ready3", release=10)}
    other_ready = {2: engine.submit("ready2b", release=10),
                   3: engine.submit("ready3b", release=0)}
    first_release = engine.future("first_release")
    second_release = engine.future("second_release")

    first_handoff = PDHandoff.build(
        "first",
        (PDTransfer(0, 2, 128.0), PDTransfer(0, 3, 128.0)),
        receive_capacity_bytes={2: 128.0, 3: 128.0},
        receive_buffer_ids={2: "X", 3: "Y"})
    second_handoff = PDHandoff.build(
        "second",
        (PDTransfer(1, 2, 128.0), PDTransfer(1, 3, 128.0)),
        receive_capacity_bytes={2: 128.0, 3: 128.0},
        receive_buffer_ids={2: "X", 3: "Y"})
    first = connector.submit_handoff(
        engine, first_handoff, source_ready_by_rank={0: source},
        destination_ready_by_rank=ready,
        release_events={2: (first_release,), 3: (first_release,)})
    second = connector.submit_handoff(
        engine, second_handoff, source_ready_by_rank={1: source},
        destination_ready_by_rank=other_ready,
        release_events={2: (second_release,), 3: (second_release,)})
    first_consumer = engine.submit(
        "first_consumer", after=(first,), latency=1,
        on_complete=lambda _event: None if first_release.done else engine.resolve(first_release))
    second_consumer = engine.submit(
        "second_consumer", after=(second,), latency=1,
        on_complete=lambda _event: None if second_release.done else engine.resolve(second_release))
    engine.advance(until=(first_consumer, second_consumer), first=False)
    assert first.time == pytest.approx(12.0)
    assert first_consumer.time == pytest.approx(13.0)
    assert second.time == pytest.approx(15.0)
    assert second_consumer.time == pytest.approx(16.0)
    assert connector.wire_bytes == pytest.approx(512.0)
    transport = connector.summary()["transport"]
    assert transport["peak_reserved_bytes"]["2"] == pytest.approx(128.0)
    assert transport["peak_reserved_bytes"]["3"] == pytest.approx(128.0)


def test_pd_whole_message_receive_reservation_avoids_partial_credit_deadlock():
    connector = PDConnector({
        "link": {"name": "handoff", "capacity_bytes_per_ms": 128,
                 "latency_us": 0},
        "chunk_bytes": 128})
    engine = Engine(connector.resources(), trace_limit=1000)
    ready = {0: engine.submit("source"), 1: engine.submit("destination")}
    first_release = engine.future("first_release")
    second_release = engine.future("second_release")

    def handoff(name):
        return PDHandoff.build(
            name, (PDTransfer(0, 1, 256.0),),
            receive_capacity_bytes={1: 256.0},
            receive_buffer_ids={1: "pool"})

    first = connector.submit_handoff(
        engine, handoff("first"), source_ready_by_rank={0: ready[0]},
        destination_ready_by_rank={1: ready[1]},
        release_events={1: (first_release,)})
    second = connector.submit_handoff(
        engine, handoff("second"), source_ready_by_rank={0: ready[0]},
        destination_ready_by_rank={1: ready[1]},
        release_events={1: (second_release,)})
    first_consumer = engine.submit(
        "first_consumer", after=(first,), latency=1,
        on_complete=lambda _event: None if first_release.done else engine.resolve(first_release))
    second_consumer = engine.submit(
        "second_consumer", after=(second,), latency=1,
        on_complete=lambda _event: None if second_release.done else engine.resolve(second_release))
    engine.advance(until=(first_consumer, second_consumer), first=False)

    # One whole-message reservation is held for each 256 B handoff.  The
    # second grant waits for the first consumer, not for a partial chunk that
    # can never become a complete handoff.
    assert first.time == pytest.approx(2.0)
    assert second.time == pytest.approx(5.0)
    assert first_consumer.time == pytest.approx(3.0)
    assert second_consumer.time == pytest.approx(6.0)
    assert connector.wire_bytes == pytest.approx(512.0)
    rows = {row["task"]: row for row in engine.trace}
    assert rows["pd:first:receive_grant"]["start_ms"] == pytest.approx(0.0)
    assert rows["pd:first:receive_release:r1"]["start_ms"] == pytest.approx(3.0)
    assert rows["pd:second:receive_grant"]["start_ms"] == pytest.approx(3.0)
    assert rows["pd:second:receive_release:r1"]["start_ms"] == pytest.approx(6.0)
    transport = connector.summary()["transport"]
    assert transport["peak_reserved_bytes"]["1"] == pytest.approx(256.0)
    assert transport["reservation_waits"] >= 1


@pytest.mark.parametrize("options", ({}, {"chunk_bytes": 128}))
def test_pd_memory_waits_both_ready_events_and_shares_actual_hbm(options):
    connector = PDConnector({
        "link": {"name": "handoff", "capacity_bytes_per_ms": 128,
                 "latency_us": 1000}, **options})
    engine = Engine((*connector.resources(), Resource("hbm", 10)),
                    trace_limit=200)
    source = engine.submit("source_ready", release=0)
    destination = engine.submit("destination_ready", release=9)
    competitor = engine.submit("hbm_competitor", {"hbm": 50}, release=9)
    done = connector.submit(engine, "pd:memory_bound", 128, {"hbm": 100},
                            ready=source, destination_ready=destination)
    engine.advance(until=(done, competitor), first=False)
    rows = {row["task"]: row for row in engine.trace}
    memory = rows["pd:memory_bound:memory"]
    # Legacy pays its propagation stage after the common boundary (start 10);
    # transport memory starts with the boundary (start 9).  Both wait for the
    # destination grant, and the competitor shares the same real HBM resource.
    assert memory["start_ms"] == pytest.approx(9 if options else 10)
    assert memory["end_ms"] > memory["start_ms"]
    assert done.time == pytest.approx(memory["end_ms"])
    assert rows["hbm_competitor"]["end_ms"] > 9


def test_pd_single_chunk_transport_preserves_legacy_framing_work():
    link = {"name": "handoff", "capacity_bytes_per_ms": 128,
            "latency_us": 1000, "transaction_bytes": 256,
            "protocol_overhead_bytes": 64, "alignment_bytes": 256}
    def run(connector):
        engine = Engine((*connector.resources(), Resource("hbm", 1000)),
                        trace_limit=100)
        done = connector.submit(engine, "pd:whole", 384, {"hbm": 17})
        engine.advance(until=(done,), first=False)
        row = next(row for row in engine.trace if row["task"] == "pd:whole")
        return done, engine, row

    legacy = PDConnector({"link": link})
    done, legacy_engine, legacy_row = run(legacy)
    transport = PDConnector({"link": link, "chunk_bytes": 384})
    planned, transport_engine, transport_row = run(transport)
    assert planned.time == pytest.approx(done.time) == pytest.approx(6.0)
    for engine, row in ((legacy_engine, legacy_row),
                        (transport_engine, transport_row)):
        assert engine.serviced["hbm"] == pytest.approx(17)
        assert row["metadata"]["useful_bytes"] == 384
        assert row["metadata"]["wire_bytes"] == 640
        assert row["metadata"]["transactions"] == 2
        if row["metadata"].get("transport"):
            assert sum(1 for event in engine.trace
                       if event["task"].endswith(":hop0:arrival")) == 1
        else:
            assert sum(1 for event in engine.trace
                       if event["task"] == "pd:whole:latency") == 1
    assert transport.wire_bytes == pytest.approx(legacy.wire_bytes) == 640.0
    assert transport.transactions == legacy.transactions == 2


def test_public_pd_uses_real_source_grant_and_consumer_boundaries():
    result = predict_pd(pd_case(), lower=linear_work, event_limit=100000)
    rows = {row["task"]: row for row in result["events"]}
    assert rows["pd:r:destination_grant"]["metadata"]["boundary"] == "pd_destination_admission"
    assert rows["pd:r:ready"]["metadata"]["boundary"] == "pd_common_source_destination_ready"
    assert rows["pd:r:ready"]["metadata"]["source_ready"].startswith("prefill:")
    assert "batch_device_complete" in rows["pd:r:ready"]["metadata"]["source_ready"]
    assert rows["pd:r:ready"]["metadata"]["destination_ready"] == "pd:r:destination_grant"
    assert rows["pd:r:consumer_boundary"]["metadata"]["boundary"] == "pd_decode_consumer_release"
    assert rows["pd:r:consume"]["metadata"]["boundary"] == "pd_consumer_release"
    # Consumer release is after transfer completion, not folded into it.
    assert rows["pd:r:consumer_boundary"]["start_ms"] > rows["pd:r"]["end_ms"]


def test_pd_connector_chunk_credit_and_startup_preserve_wire_and_work():
    link = {"name": "handoff", "capacity_bytes_per_ms": 128,
            "latency_us": 1000, "transaction_bytes": 256,
            "protocol_overhead_bytes": 64, "alignment_bytes": 256}
    legacy = PDConnector({"link": link})
    engine = Engine(legacy.resources())
    done = legacy.submit(engine, "pd:legacy", 384, {"memory": 0})
    engine.advance(until=(done,))
    assert done.time == pytest.approx(6.0)
    assert legacy.wire_bytes == pytest.approx(640.0)

    connector = PDConnector({"link": link, "chunk_bytes": 128,
                             "receive_capacity_bytes": 128,
                             "hop_startup_us": 2})
    engine = Engine(connector.resources())
    done = connector.submit(engine, "pd:plan", 384, {"memory": 0})
    engine.advance(until=(done,))
    # Three 128 B chunks each frame as one 256 B transaction.  A finite
    # receive credit serializes the next chunk behind the previous arrival;
    # the declared startup is one gate on the first chunk of the hop.
    assert connector.wire_bytes == pytest.approx(960.0)
    assert connector.summary()["transport"]["chunks"] == 3
    assert connector.summary()["transport"]["credit_waits"] >= 1
    assert connector.summary()["transport"]["peak_reserved_bytes"]["1"] <= 128
    assert done.time == pytest.approx(10.5 + 0.002)


def test_pd_receiver_residency_holds_arrived_bytes_until_the_declared_consumer():
    link = {"name": "handoff", "capacity_bytes_per_ms": 128,
            "latency_us": 1000}
    connector = PDConnector({"link": link, "chunk_bytes": 128,
                             "receive_capacity_bytes": 384})
    engine = Engine(connector.resources())
    ready = engine.submit("ready")
    consumes = tuple(engine.submit(f"decode_consume{index}", latency=20 + 5 * index)
                     for index in range(3))
    done = connector.submit(engine, "pd:resident", 384, {"memory": 0},
                            ready=ready, release_events={1: consumes})
    engine.advance(until=(done,), first=False)
    transport = connector.summary()["transport"]
    assert transport["peak_reserved_bytes"]["1"] == pytest.approx(384)
    assert transport["peak_resident_bytes"]["1"] == pytest.approx(384)
    # Consumer release is a separate boundary, not an extra handoff duration.
    assert done.time == pytest.approx(4.0)
    engine.advance(until=tuple(consumes), first=False)
    assert engine.summary()["live_tasks"] == 0


def test_pd_concurrent_handoffs_use_the_shared_engine_receive_pool():
    connector = PDConnector({
        "link": {"name": "handoff", "capacity_bytes_per_ms": 128,
                 "latency_us": 0},
        "chunk_bytes": 128, "receive_capacity_bytes": 384})
    engine = Engine(connector.resources(), trace_limit=1000)
    first = connector.submit(engine, "pd:first", 384, {})
    second = connector.submit(engine, "pd:second", 384, {})
    engine.advance(until=(first, second), first=False)
    transport = connector.summary()["transport"]
    assert transport["reserved_peak_basis"] == "engine_pool_peak_granted"
    pool = next(name for name in engine.pools if name.startswith("transport:rx:"))
    assert transport["peak_reserved_bytes"]["1"] == pytest.approx(
        engine.peak_allocated[pool])
    assert transport["peak_reserved_bytes"]["1"] <= 384
    assert transport["residency_peak_basis"] == "max_single_handoff_not_concurrent_sum"


def test_pd_small_receive_credit_rejects_a_single_post_handoff_consumer():
    connector = PDConnector({
        "link": {"name": "handoff", "capacity_bytes_per_ms": 128},
        "chunk_bytes": 128, "receive_capacity_bytes": 128})
    engine = Engine(connector.resources())
    release = engine.future("decode_consume")
    with pytest.raises(ModelingError, match="per_chunk_consumer"):
        connector.submit(engine, "pd:deadlock", 384, {}, release_events={1: release})


def test_legacy_pd_connector_options_keep_public_pd_link_identity():
    raw = pd_case()
    raw["connector"].update({"chunk_bytes": 64, "receive_capacity_bytes": 384,
                             "hop_startup_us": 1})
    result = predict_pd(raw, lower=linear_work)
    assert result["execution"]["serviced"]["pd_link"] == pytest.approx(384)
    assert result["execution"]["connector"]["transport"]["chunks"] > 1
    assert result["execution"]["connector"]["link"]["name"] == "connector"


def test_public_pd_finite_receive_credit_is_shared_and_does_not_deadlock():
    raw = pd_case()
    raw["connector"] = {
        "link": {"name": "handoff", "capacity_bytes_per_ms": 128},
        "chunk_bytes": 128, "receive_capacity_bytes": 384, "hop_startup_us": 1}
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    transport = result["execution"]["connector"]["transport"]
    assert result["execution"]["handoff_bytes"] == 384
    assert transport["chunks"] == 3
    assert transport["peak_reserved_bytes"]["1"] <= 384
    assert transport["peak_resident_bytes"]["1"] <= 384
    rows = {row["task"]: row for row in result["events"]}
    assert rows["pd:r:consumer_boundary"]["metadata"]["boundary"] == "pd_decode_consumer_release"

    raw["connector"]["receive_capacity_bytes"] = 256
    # C is preserved independently of F now.  A whole-message grant whose
    # F=384 cannot fit the declared C=256 is refused before any issue instead
    # of silently building a 384 B pool.
    with pytest.raises(ModelingError, match="per_chunk_consumer"):
        predict_pd(raw, lower=linear_work, event_limit=100000)


def _disjoint_pd_case(requests=None):
    raw = pd_case(requests)
    for pool, ranks in (("prefill", (0, 1)), ("decode", (2, 3))):
        machine = raw["pools"][pool]["machine"]
        machine["num_devices"] = 2
        machine["devices"] = [{"rank": rank} for rank in ranks]
        machine["links"][0]["members"] = list(ranks)
        raw["pools"][pool]["runtime"]["tp"] = 2
    raw["connector"] = {
        "link": {"name": "handoff", "capacity_bytes_per_ms": 128},
        "chunk_bytes": 128, "receive_capacity_bytes": 384, "hop_startup_us": 1}
    return raw


def test_public_pd_two_concurrent_requests_complete_under_shared_limits():
    raw = _disjoint_pd_case([
        {"id": "a", "prompt_tokens": 2, "output_tokens": 3},
        {"id": "b", "prompt_tokens": 2, "output_tokens": 3}])
    raw["connector"]["receive_capacity_bytes"] = 256
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    assert result["status"] == "modeled"
    assert [row["status"] for row in result["requests"]] == ["served", "served"]
    assert len(result["execution"]["pd_handoffs"]) == 2
    assert {row["request_id"] for row in result["execution"]["pd_handoffs"]} == {
        "a", "b"}
    rows = {row["task"]: row for row in result["events"]}
    for request in ("a", "b"):
        assert f"pd:{request}:receive_grant" in rows
        assert f"pd:{request}:receive_release:r2" in rows
        assert f"pd:{request}:consumer_boundary:r2" in rows


def test_public_pd_quantized_kv_keeps_storage_and_versions():
    raw = _disjoint_pd_case()
    raw["model"]["kv_dtype"] = "int8"
    raw["connector"]["receive_capacity_bytes"] = 384
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    handoff = result["execution"]["pd_handoffs"][0]
    assert [row["status"] for row in result["requests"]] == ["served"]
    assert handoff["storage_holds"] == {
        "2": "pd:kv:r:decode:r0:rank2",
        "3": "pd:kv:r:decode:r0:rank3"}
    assert all(row["component"] == "gqa_kv" for row in handoff["transfers"])
    assert all(row["source_version"] >= 1 and row["destination_version"] >= 1
               for row in handoff["transfers"])
    assert all(row["payload_bytes"] == pytest.approx(48.0)
               for row in handoff["transfers"])


def test_public_pd_shared_384_pool_accepts_mixed_footprints():
    raw = _disjoint_pd_case([
        {"id": "a", "prompt_tokens": 2, "output_tokens": 3},
        {"id": "b", "prompt_tokens": 3, "output_tokens": 3}])
    raw["connector"]["shared_receive_buffer"] = True
    raw["connector"]["receive_capacity_bytes"] = 384
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    assert [row["status"] for row in result["requests"]] == ["served", "served"]
    first, second = result["execution"]["pd_handoffs"]
    assert first["receive_pool_capacity_bytes"] == second["receive_pool_capacity_bytes"]
    assert first["receive_pool_capacity_bytes"]["2"] == pytest.approx(384.0)
    assert first["receive_capacity_bytes"]["2"] < second["receive_capacity_bytes"]["2"]
    rows = {row["task"]: row for row in result["events"]}
    for request in ("a", "b"):
        assert rows[f"pd:{request}:receive_grant"]["metadata"]["pool_capacity_bytes"]["2"] == pytest.approx(384.0)
        assert rows[f"pd:{request}:memory"]["start_ms"] >= rows[f"pd:{request}:receive_grant"]["start_ms"]
    assert sum(value for name, value in result["execution"]["serviced"].items()
               if name.startswith("pd:link:handoff")) == pytest.approx(640.0)


def test_public_pd_shared_receive_buffer_admits_successor_after_release():
    raw = _disjoint_pd_case([
        {"id": "a", "prompt_tokens": 3, "output_tokens": 3},
        {"id": "b", "prompt_tokens": 3, "output_tokens": 3}])
    raw["connector"]["shared_receive_buffer"] = True
    raw["connector"]["receive_capacity_bytes"] = 256
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    assert [row["status"] for row in result["requests"]] == ["served", "served"]
    first, second = result["execution"]["pd_handoffs"]
    assert first["receive_buffer_ids"] == second["receive_buffer_ids"] == {
        "2": "connector:rank2", "3": "connector:rank3"}
    rows = {row["task"]: row for row in result["events"]}
    release = rows["pd:a:receive_release:r2"]["start_ms"]
    assert rows["pd:b:receive_grant"]["start_ms"] >= release
    assert rows["pd:b:receive_grant"]["start_ms"] < result["requests"][1]["finish_ms"]
    assert rows["pd:b:memory"]["start_ms"] >= rows["pd:b:receive_grant"]["start_ms"]
    assert result["execution"]["serviced"]["pd:link:handoff"] == pytest.approx(768.0)


def _layout_pd_case(kind):
    raw = pd_case([{"id": "r", "prompt_tokens": 1, "output_tokens": 3}])
    if kind == "pp":
        layout = (("prefill", (0, 1), 1, 2), ("decode", (2,), 1, 1))
    else:
        layout = (("prefill", (0, 1), 2, 1), ("decode", (2,), 1, 1))
    for pool, ranks, tp, pp in layout:
        machine = raw["pools"][pool]["machine"]
        machine["num_devices"] = len(ranks)
        machine["devices"] = [{"rank": rank} for rank in ranks]
        machine["links"][0]["members"] = list(ranks)
        raw["pools"][pool]["runtime"].update(tp=tp, pp=pp)
    raw["connector"] = {
        "link": {"name": "handoff", "capacity_bytes_per_ms": 64,
                 "latency_us": 0, "semantics": "per_pair"},
        "chunk_bytes": 64}
    return raw


@pytest.mark.parametrize("kind", ("pp", "tp"))
def test_pd_transfer_sources_follow_actual_pp_tp_range_intersection(kind):
    result = predict_pd(_layout_pd_case(kind), lower=linear_work,
                        event_limit=100000)
    handoff = result["execution"]["pd_handoffs"][0]
    transfers = handoff["transfers"]
    assert handoff["source_ranks"] == [0, 1]
    assert handoff["destination_ranks"] == [2]
    expected_payload = 64.0 if kind == "pp" else 32.0
    assert all(row["payload_bytes"] == pytest.approx(expected_payload)
               for row in transfers)
    assert sum(row["payload_bytes"] for row in transfers) == pytest.approx(128.0)
    assert {row["source_rank"] for row in transfers} == {0, 1}
    if kind == "pp":
        assert {(row["source_rank"], row["source_stage"],
                 row["destination_stage"]) for row in transfers} == {
            (0, 0, 0), (1, 1, 0)}
    else:
        assert {row["source_layers"][0] for row in transfers} == {0, 1}
        assert all(row["source_stage"] == 0 and row["destination_stage"] == 0
                   for row in transfers)
    assert sum(value for name, value in result["execution"]["serviced"].items()
               if name.startswith("pd:link:handoff")) == pytest.approx(128.0)
    assert [row["status"] for row in result["requests"]] == ["served"]


def test_pd_pp_source_ranges_survive_noncontiguous_rank_renumbering():
    raw = _layout_pd_case("pp")
    for pool, ranks in (("prefill", (11, 5)), ("decode", (23,))):
        machine = raw["pools"][pool]["machine"]
        machine["devices"] = [{"rank": rank} for rank in ranks]
        machine["links"][0]["members"] = list(ranks)
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    handoff = result["execution"]["pd_handoffs"][0]
    assert handoff["source_ranks"] == [5, 11]
    assert handoff["destination_ranks"] == [23]
    assert {(row["source_rank"], row["source_stage"]) for row in
            handoff["transfers"]} == {(11, 0), (5, 1)}
    assert all(row["destination_rank"] == 23 for row in handoff["transfers"])
    assert sum(row["payload_bytes"] for row in handoff["transfers"]) == 128.0


def _tp4_pd_case(prefill_tp, decode_tp, prefill_ranks, decode_ranks):
    raw = pd_case([{"id": "r", "prompt_tokens": 1, "output_tokens": 3}])
    raw["model"].update(layers=1, hidden_size=64, num_heads=4,
                        num_kv_heads=4, head_dim=16, intermediate_size=64,
                        vocab_size=128)
    for pool, ranks, tp in (("prefill", prefill_ranks, prefill_tp),
                            ("decode", decode_ranks, decode_tp)):
        machine = raw["pools"][pool]["machine"]
        machine["num_devices"] = len(ranks)
        machine["devices"] = [{"rank": rank} for rank in ranks]
        machine["links"][0]["members"] = list(ranks)
        raw["pools"][pool]["runtime"].update(tp=tp, pp=1)
    raw["connector"] = {
        "link": {"name": "handoff", "capacity_bytes_per_ms": 64,
                 "latency_us": 0, "semantics": "per_pair"},
        "chunk_bytes": 64}
    return raw


@pytest.mark.parametrize("kind", ("tp4to2", "tp2to4"))
def test_pd_tp_head_range_intersection_is_not_cartesian(kind):
    if kind == "tp4to2":
        raw = _tp4_pd_case(4, 2, (0, 1, 2, 3), (4, 5))
        expected_source_heads = {(0, 0, 1), (1, 1, 2),
                                 (2, 2, 3), (3, 3, 4)}
        expected_destination_heads = {(4, 0, 2), (5, 2, 4)}
    else:
        raw = _tp4_pd_case(2, 4, (10, 11), (20, 21, 22, 23))
        expected_source_heads = {(10, 0, 2), (11, 2, 4)}
        expected_destination_heads = {(20, 0, 1), (21, 1, 2),
                                      (22, 2, 3), (23, 3, 4)}
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    handoff = result["execution"]["pd_handoffs"][0]
    transfers = handoff["transfers"]
    assert len(transfers) == 4
    assert all(row["payload_bytes"] == pytest.approx(64.0)
               for row in transfers)
    assert sum(row["payload_bytes"] for row in transfers) == pytest.approx(256.0)
    assert {(row["source_rank"], *row["source_heads"]) for row in transfers} == (
        expected_source_heads)
    assert {(row["destination_rank"], *row["destination_heads"])
            for row in transfers} == expected_destination_heads
    assert all(row["component"] == "gqa_kv" for row in transfers)
    assert all(row["source_token_span"] == [0, 1]
               and row["destination_token_span"] == [0, 1]
               for row in transfers)
    assert all(row["producer"].startswith("prefill:")
               and row["output"].startswith("decode:") for row in transfers)
    assert sum(value for name, value in result["execution"]["serviced"].items()
               if name.startswith("pd:link:handoff")) == pytest.approx(256.0)


def test_public_pd_disjoint_pools_bind_actual_ranks_chunks_and_buffers():
    result = predict_pd(_disjoint_pd_case(), lower=linear_work,
                        event_limit=100000)
    binding = result["execution"]["memory_binding"]
    assert binding["same_engine"] is True
    assert binding["pools"]["prefill"] != binding["pools"]["decode"]
    connector = result["execution"]["connector"]
    assert connector["handoff_ranks"] == [0, 1, 2, 3]
    assert connector["handoff_flows"] == 4
    assert connector["handoff_receive_capacity_bytes"] == {
        "2": 192.0, "3": 192.0}
    rows = {row["task"]: row for row in result["events"]}
    assert "pd:r:pair0>2:ready" in rows
    assert "pd:r:pair1>3:ready" in rows
    assert ("pd:r:destination_grant:r2" in rows
            and "pd:r:destination_grant:r3" in rows)
    assert rows["pd:r:destination_grant:r2"]["metadata"]["receive_buffer_id"] == (
        "pd:kv:r:decode:r0:rank2")
    assert rows["pd:r:destination_grant:r3"]["metadata"]["receive_buffer_id"] == (
        "pd:kv:r:decode:r0:rank3")
    assert all(any(f":flow{index}:chunk0:" in task for task in rows)
               for index in range(4))
    assert rows["pd:r"]["metadata"]["flows"] == 4
    assert rows["pd:r"]["metadata"]["receive_buffer_ids"] == {
        2: "pd:kv:r:decode:r0:rank2", 3: "pd:kv:r:decode:r0:rank3"}
    assert result["execution"]["handoff_bytes"] == 384
    assert result["execution"]["serviced"]["pd:link:handoff"] == 384
    handoff = result["execution"]["pd_handoffs"][0]
    assert handoff["transport_binding"] == "actual_rank"
    assert handoff["source_ranks"] == [0, 1]
    assert handoff["destination_ranks"] == [2, 3]
    assert [row["destination_rank"] for row in handoff["transfers"]] == [2, 3, 2, 3]
    assert all(row["component"] for row in handoff["transfers"])
    assert all(row["source_layers"] for row in handoff["transfers"])
    assert all(row["source_version"] >= 0 and row["destination_version"] >= 0
               for row in handoff["transfers"])
    assert set(handoff["storage_holds"]) == {"2", "3"}
    assert handoff["receive_buffer_ids"] == {
        "2": "pd:kv:r:decode:r0:rank2", "3": "pd:kv:r:decode:r0:rank3"}
    serviced = result["execution"]["serviced"]
    assert serviced["prefill:memory@r0:s0"] == 192
    assert serviced["decode:memory@r0:s0"] == 192


def test_pd_consumer_release_uses_decode_device_complete_not_host_tail():
    raw = pd_case()
    raw["pools"]["decode"]["runtime"]["submission"] = {"completion_ms": 10}
    result = predict_pd(raw, lower=linear_work, event_limit=100000)
    rows = {row["task"]: row for row in result["events"]}
    device = rows["decode:r0:i0:device_complete"]
    consumer = rows["pd:r:consumer_boundary"]
    host_completion = rows["decode:r0:i0:complete"]
    assert consumer["start_ms"] == pytest.approx(device["end_ms"])
    assert consumer["start_ms"] < host_completion["end_ms"]
    # The host tail is still visible as the decode step completion, but it did
    # not hold the receive credit or delay the consumer release.
    assert host_completion["end_ms"] == pytest.approx(17.0)


def test_public_pd_connector_options_keep_handoff_conservation_and_finish_causal():
    raw = pd_case()
    raw["connector"] = {"link": {"name": "handoff", "capacity_bytes_per_ms": 128},
                        "chunk_bytes": 128, "hop_startup_us": 1}
    baseline = predict_pd(pd_case(), lower=linear_work)
    planned = predict_pd(raw, lower=linear_work)
    assert planned["execution"]["handoff_bytes"] == baseline["execution"]["handoff_bytes"]
    assert planned["execution"]["connector"]["transport"]["chunks"] > 1
    assert planned["requests"][0]["finish_ms"] > baseline["requests"][0]["finish_ms"]
