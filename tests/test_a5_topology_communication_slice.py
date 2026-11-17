"""A5/T09 physical topology, sharing, chunk/credit and native-call controls.

These tests establish independent oracles, not snapshots of a fitted model:

* H4's shared-forward work gives a 40 us lower bound and the full span keeps
  endpoint and chunk-pipeline effects separate.
* The same traffic and one-hop route changes service when the Rank/die/node
  placement changes the sharing domain.
* A multi-hop transfer is not priced as payload service times hop count.
* A finite receive credit/residency changes completion without inventing work.
* The public graph and native HCCL lowering reach the same transport.
"""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim import compose_case, graph_costs, operator_costs
from mindie_sim.prediction.collective_readiness import (
    SCHEMA as READINESS_SCHEMA, readiness_skew, start_with_pipeline,
)
from mindie_sim.prediction.communication import MessageFlow
from mindie_sim.prediction.direct_peer import DirectPeerCollective, DirectPeerGroup
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.hccl_graph import NETWORK as HCCL_NETWORK, lower as hccl_lower
from mindie_sim.prediction.mc2_graph import NETWORK as MC2_NETWORK, recipes as mc2_recipes, select as mc2_select
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import (
    Topology, TransportOptions, chunk_sizes,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware, Link
from mindie_sim.schema.parallel import ParallelSpec

from test_communication_topology import exchange
from test_hccl_source_graph import declared as declared_hccl
from test_prediction_system import case


def _machine(links, *, devices=2, ranks_per_node=None):
    base = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend910B1",
        "num_devices": devices,
        "ranks_per_node": ranks_per_node or devices,
    })
    return replace(base, links=tuple(links))


def _ready(engine, count, starts=None):
    starts = starts or [0.0] * count
    return {rank: engine.submit(f"ready{rank}", release=starts[rank])
            for rank in range(count)}


def _run(plan, *, capacity_events=None, starts=None, name="transport"):
    engine = Engine(plan.resources())
    required = plan.ranks
    ready = {rank: engine.submit(f"ready{rank}",
                                 release=(starts or {}).get(rank, 0.0))
             for rank in required}
    run = plan.start(engine, name, ready, release_events=capacity_events)
    engine.advance(until=run.completions, first=False)
    return run.report(), engine


def _h4_plan(semantics="shared", *, endpoint_rates=None, capacity=None,
             chunk=300, reverse=False):
    rates = endpoint_rates or {0: 120_000, 1: 120_000,
                               2: 120_000, 3: 120_000}
    endpoint = tuple(sorted((int(rank), float(rate))
                            for rank, rate in rates.items()))
    link = Link("shared_forward", 60_000, (0, 1, 2, 3), 2.0,
                "fabric", semantics, endpoint_rates=endpoint)
    topology = Topology(_machine((link,), devices=4), ParallelSpec(tp=1))
    flows = ([MessageFlow(0, 2, 1200), MessageFlow(2, 0, 1200)]
             if reverse else
             [MessageFlow(0, 2, 1200), MessageFlow(1, 3, 1200)])
    return topology.transport_plan(
        flows, chunk_bytes=chunk, hop_startup_us=2,
        receive_capacity_bytes=capacity)


def test_h4_same_direction_shared_lower_bound_is_not_the_full_span():
    shared, _ = _run(_h4_plan("shared"))
    disjoint, _ = _run(_h4_plan("per_pair"))
    reverse, _ = _run(_h4_plan("per_pair", reverse=True))

    # H4 frozen contract: two same-direction flows, 2 * 1200 B on one
    # 60,000 B/ms (60 B/us) forward plane -> 40 us lower bound.
    assert shared["service_lower_bound_ms"] == pytest.approx(0.04)
    assert shared["shared_resource_work_bytes"] == pytest.approx(2400.0)
    assert shared["path_startup_us"] == pytest.approx(6.0)
    # The span includes the six declared startup us and the actual chunk
    # pipeline; it must not be replaced by the 40 us lower bound.
    assert shared["span_ms"] == pytest.approx(0.046, rel=1e-6)

    # Independent per-pair forward directions: 1200 B / 60,000 B/ms each.
    assert disjoint["service_lower_bound_ms"] == pytest.approx(0.02)
    assert disjoint["span_ms"] < shared["span_ms"]
    assert reverse["service_lower_bound_ms"] == pytest.approx(0.02)
    reverse_resources = {
        name for name in reverse["bottleneck_resources"]
        if ":pair" in name
    }
    shared_resources = {
        name for name in shared["bottleneck_resources"]
        if name == "link:shared_forward"
    }
    assert len(reverse_resources) == 2
    assert shared_resources == {"link:shared_forward"}


def test_h4_heterogeneous_destination_port_binds_its_own_placement():
    report, _ = _run(_h4_plan(
        "shared", endpoint_rates={0: 120_000, 1: 120_000,
                                  2: 30_000, 3: 120_000}))
    by_flow = {(row["source"], row["destination"]): row
               for row in report["per_flow_isolated_service_ms"]}
    # 1200 B / 30,000 B/ms at rank 2's receive port.
    assert by_flow[(0, 2)]["isolated_service_ms"] == pytest.approx(0.04)
    assert by_flow[(1, 3)]["isolated_service_ms"] == pytest.approx(0.02)
    assert report["service_lower_bound_ms"] == pytest.approx(0.04)
    assert report["shared_resource_work_bytes"] == pytest.approx(2400.0)
    assert report["endpoint_bottleneck_work_bytes"] == pytest.approx(1200.0)


def test_multihop_chunks_pipeline_without_payload_times_hop_count():
    links = (
        Link("first", 1000, (0, 1), 1000.0, "fabric", "shared"),
        Link("second", 1000, (1, 2), 1000.0, "fabric", "shared"),
    )
    topology = Topology(_machine(links, devices=3), ParallelSpec(tp=1))
    flow = MessageFlow(0, 2, 3000)
    whole = topology.transport_plan([flow])
    pipelined = topology.transport_plan([flow], chunk_bytes=1000)

    whole_report, _ = _run(whole)
    pipe_report, _ = _run(pipelined)
    # Two 1000 B/us links carry 3000 B each: the widest resource bound is 3 ms,
    # not 3 hops * 3 ms = 9 ms.
    assert whole_report["service_lower_bound_ms"] == pytest.approx(3.0)
    assert pipe_report["service_lower_bound_ms"] == pytest.approx(3.0)
    assert pipe_report["chunks"] == 3
    # A full message pays 3 ms + 1 us on each hop; chunks overlap the next
    # hop's service with the following chunk's transmission.
    assert whole_report["span_ms"] == pytest.approx(8.0)
    assert pipe_report["span_ms"] < whole_report["span_ms"]
    assert (pipe_report["capacity_work_bytes"]
            == whole_report["capacity_work_bytes"] == pytest.approx(6000.0))


def test_a5_placement_changes_route_and_span_for_identical_traffic():
    def machine(node_of, split):
        devices = [{"rank": rank, "node": node_of(rank), "package": rank // 2}
                   for rank in range(8)]
        links = [
            {"name": "d2d_intra_node:0",
             "members": [0, 1, 2, 3] if split else list(range(8)),
             "capacity_bytes_per_ms": 800e6, "latency_us": 1.5,
             "channel": "d2d_intra_node", "semantics": "per_port"},
            *([{"name": "d2d_intra_node:1", "members": [4, 5, 6, 7],
                "capacity_bytes_per_ms": 800e6, "latency_us": 1.5,
                "channel": "d2d_intra_node", "semantics": "per_port"}]
              if split else []),
            {"name": "d2d_inter_node:0", "members": list(range(8)),
             "capacity_bytes_per_ms": 50e6, "latency_us": 4.5,
             "channel": "d2d_inter_node", "semantics": "per_port"},
        ]
        return Hardware.from_mapping({
            "backend": "ascend", "sku": "Ascend950DT_9581",
            "num_devices": 8, "ranks_per_node": 4,
            "devices": devices, "links": links,
        })

    split_topology = Topology(machine(lambda rank: rank // 4, True),
                              ParallelSpec(tp=8))
    flat = Topology(machine(lambda rank: 0, False), ParallelSpec(tp=8))
    assert split_topology.location(5)["node"] == 1
    assert flat.location(5)["node"] == 0
    assert [hop[0] for hop in split_topology.route(0, 5)] == ["d2d_inter_node:0"]
    assert [hop[0] for hop in flat.route(0, 5)] == ["d2d_intra_node:0"]

    flows = [MessageFlow(0, 5, 1024 * 1024), MessageFlow(5, 0, 1024 * 1024)]
    split_report, _ = _run(split_topology.transport_plan(flows))
    flat_report, _ = _run(flat.transport_plan(flows))
    assert split_report["service_lower_bound_ms"] > flat_report["service_lower_bound_ms"]
    assert split_report["span_ms"] > flat_report["span_ms"]



def test_receive_credit_residency_changes_completion_without_new_work():
    baseline_full = _h4_plan("shared", capacity=None)
    baseline_plan = replace(baseline_full, flows=(baseline_full.flows[0],))
    baseline, _ = _run(baseline_plan)
    assert baseline["credit_waits"] == 0

    full = _h4_plan("shared", capacity={2: 600})
    plan = replace(full, flows=(full.flows[0],))
    engine = Engine(plan.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in plan.ranks}
    consume = engine.submit("decode_consume", latency=10)
    run = plan.start(engine, "credit", ready,
                     release_events={2: (consume,)})
    engine.advance(until=run.completions, first=False)
    report = run.report()
    assert report["credit_waits"] >= 1
    assert report["span_ms"] > baseline["span_ms"]
    assert all(value <= 600 for value in
               report["receiver_reserved_peak_bytes"].values())
    assert all(value <= 600 for value in
               report["receiver_resident_peak_bytes"].values())
    assert report["receiver_resident_byte_us"][2] > 0
    # Work is conserved; only the admission/residency relation changed.
    assert (report["capacity_work_bytes"]
            == baseline["capacity_work_bytes"])


def test_public_a5_graph_same_traffic_different_sharing_domains():
    reports = {}
    for preset in ("a5_server", "a5_pod"):
        raw = compose_case(
            "qwen3.5-35b-a3b", hardware="Ascend950DT_9581",
            topology_preset=preset, tp=8, dp=1, ep=8, layers=1,
            prompt_tokens=8, output_tokens=1, concurrency=1)
        report = graph_costs(
            raw, (exchange(8 * 1024 * 1024, family="allreduce"),))
        reports[preset] = report
        assert report["execution"]["live_tasks"] == 0
    server = {name for name in reports["a5_server"]["execution"]["serviced"]
              if name.startswith("link:")}
    pod = {name for name in reports["a5_pod"]["execution"]["serviced"]
           if name.startswith("link:")}
    # The same 8-Rank allreduce is served by 4P and node domains on the
    # server preset, but only by the UB node domain on the pod preset.
    assert any("intra_4p" in name for name in server)
    assert any("intra_node" in name for name in server)
    assert not any("intra_4p" in name for name in pod)
    # The server preset uses directed per-pair UB resources; the pod preset
    # exposes each rank endpoint as one TX and one RX port.  Both carry the
    # same 8 MB per-rank payload once in one direction.
    assert all(":pair" in name for name in server)
    assert any(":tx:port" in name for name in pod)
    server_wire = sum(value for name, value in
                      reports["a5_server"]["execution"]["serviced"].items()
                      if name.startswith("link:"))
    pod_wire = sum(value for name, value in
                   reports["a5_pod"]["execution"]["serviced"].items()
                   if name.startswith("link:") and ":tx:port" in name)
    assert server_wire == pytest.approx(pod_wire)
    assert reports["a5_server"]["duration_ms"] != reports["a5_pod"]["duration_ms"]


def test_public_direct_peer_graph_exposes_chunk_pipeline_and_unknowns():
    raw = compose_case(
        "qwen3.5-35b-a3b", hardware="Ascend950DT_9581",
        topology_preset="a5_server", tp=8, dp=1, ep=8, layers=1,
        prompt_tokens=8, output_tokens=1, concurrency=1)
    raw["runtime"]["kernels"] = {"families": {"alltoall": {
        "kind": "direct_peer_collective", "max_inflight_per_rank": 7,
        "chunk_bytes": 65536,
        "receive_capacity_bytes_per_peer": 2 * 1024 * 1024}}}
    report = graph_costs(
        raw, (exchange(8 * 1024 * 1024),), isolated=True)
    row = next(row for row in report["isolated"]["operators"]
               if row.get("communication"))
    assert row["communication"]["latency_model"] == (
        "chunk_pipeline_with_receive_credits")
    assert row["communication"]["chunk_bytes"] == 65536
    assert row["communication"]["unknowns"] == {
        "protocol_startup_us": "not_declared; no startup gate submitted"}
    assert report["execution"]["live_tasks"] == 0
    # Each directed pair carries one 8 MB payload; M2 charges the pair once
    # instead of separate TX and RX port aliases.
    assert sum(value for name, value in report["execution"]["serviced"].items()
               if name.startswith("link:")) == pytest.approx(
                   8 * 8 * 1024 * 1024)


def test_native_hccl_lowering_uses_declared_transport_chunks():
    raw = declared_hccl("AIV", "mesh_oneshot", ranks=4)
    parsed = parse_case(raw)
    baseline = replace(
        exchange(12 * 1024 * 1024, family="allreduce", axis="tp"), deps=(),
        metadata=(("group_axis", "tp"),))
    chunked = replace(
        baseline, metadata=(("group_axis", "tp"),
                            ("transport_chunk_bytes", 65536)))
    reports = {}
    names = {}
    for label, op in (("baseline", baseline), ("chunked", chunked)):
        selected = hccl_lower((op,), parsed.runtime,
                              parsed.model.activation_dtype)
        network = next(item for item in selected
                       if item.recipe == HCCL_NETWORK)
        topology = Topology(parsed.machine, parsed.runtime.parallel)
        ledger = topology.accounting(network, 0)
        # Independent physical-chunk expansion against the public ledger.
        expected_wire = 0.0
        expected_messages = 0
        declared_chunk = dict(network.metadata).get("transport_chunk_bytes")
        for flow, _rounds in topology.messages(
                network, 0, volume=network.link_bytes):
            for size in chunk_sizes(flow.payload_bytes, declared_chunk):
                message = topology.route_message(MessageFlow(
                    flow.source, flow.destination, size, flow.messages))
                expected_wire += sum(hop.wire_bytes for hop in message.hops)
                expected_messages += flow.messages
        assert ledger.physical_wire_bytes == pytest.approx(expected_wire)
        assert ledger.messages == expected_messages
        report = graph_costs(raw, selected, input_ready_ms=0, trace_limit=100000)
        reports[label] = report
        names[label] = [row.get("task") or row.get("name") or ""
                        for row in report["events"]]
    assert not any("chunk" in name for name in names["baseline"])
    assert any("chunk" in name for name in names["chunked"])
    assert (sum(value for name, value in
                reports["chunked"]["execution"]["serviced"].items()
                if name.startswith("link:"))
            == pytest.approx(sum(value for name, value in
                                 reports["baseline"]["execution"]["serviced"].items()
                                 if name.startswith("link:"))))


def test_readiness_startup_gate_is_a_real_engine_event_and_unknown_is_explicit():
    report_doc = {"schema": READINESS_SCHEMA, "calls": [
        {"rank_id": "rank_0", "rank_index": 0,
         "operation": {"dtype": "BFP16", "typed_bytes": 1024},
         "entry": {"start_ns": 1}, "completion": {"end_ns": 2},
         "observed_context": {}, "clock": {}, "dependencies": (), "waits": (),
         "unknowns": ()},
        {"rank_id": "rank_1", "rank_index": 1,
         "operation": {"dtype": "BFP16", "typed_bytes": 1024},
         "entry": {"start_ns": 1}, "completion": {"end_ns": 2},
         "observed_context": {}, "clock": {}, "dependencies": (), "waits": (),
         "unknowns": ()},
    ]}

    def plan():
        return SimpleNamespace(start_participants=lambda engine, name, ready,
                               owner="": {
            rank: engine.submit(f"{name}:{rank}", after=(event,))
            for rank, event in ready.items()})

    base_engine = Engine([], trace_limit=16)
    base = plan().start_participants(base_engine, "base", _ready(base_engine, 2))
    base_engine.advance(until=tuple(base.values()), first=False)

    engine = Engine([], trace_limit=16)
    ready = _ready(engine, 2)
    completions, context = start_with_pipeline(
        report_doc, engine, plan(), ready,
        startup_service={"startup_us": 2.0, "source": "T06-test"})
    engine.advance(until=tuple(completions.values()), first=False)
    assert context[0]["protocol_startup"]["status"] == "submitted"
    assert context[0]["protocol_startup"]["startup_us"] == pytest.approx(2.0)
    assert completions[0].time == pytest.approx(base[0].time + 0.002)
    skew = readiness_skew(context)
    assert skew["status"] == "pipeline_events"
    assert skew["skew_ms"] == pytest.approx(0.0)

    unknown_engine = Engine([], trace_limit=16)
    unknown_ready = _ready(unknown_engine, 2)
    unknown_completions, unknown_context = start_with_pipeline(
        report_doc, unknown_engine, plan(), unknown_ready)
    unknown_engine.advance(until=tuple(unknown_completions.values()), first=False)
    assert unknown_context[0]["protocol_startup"]["status"] == (
        "unknown_not_submitted")
    assert unknown_completions[0].time == pytest.approx(base[0].time)


def test_mc2_native_selection_preserves_explicit_chunk_and_records_unknowns():
    raw = {"native_moe_protocol": {
        "kind": "arch35_fullmesh", "max_inflight_per_rank": 2,
        "chunk_bytes": 1024, "hop_startup_us": 1.0,
        "receive_capacity_bytes_per_peer": 4096}}
    selected = mc2_select(raw)
    assert selected["chunk_bytes"] == 1024
    recipe = mc2_recipes(raw)[MC2_NETWORK]
    assert isinstance(recipe, DirectPeerCollective)
    assert recipe.plan_options["chunk_bytes"] == 1024
    assert recipe.plan_options["receive_capacity_bytes_per_peer"] == 4096
    assert recipe.transport_unknowns() == {}
    with pytest.raises(ModelingError, match="chunk_bytes"):
        mc2_select({"native_moe_protocol": {
            "kind": "arch35_fullmesh", "chunk_bytes": 0}})
    baseline = mc2_recipes({"native_moe_protocol": {
        "kind": "arch35_fullmesh"}})[MC2_NETWORK]
    assert "transfer_chunk_bytes" in baseline.transport_unknowns()



def test_pd_handoff_credits_bound_receiver_residency_before_decode_consume():
    base = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend950DT_9581",
        "num_devices": 2, "ranks_per_node": 2})
    link = Link("pd_connector", 100e6, (0, 1), 5.0, "fabric", "per_port")
    topology = Topology(replace(base, links=(link,)), ParallelSpec(tp=1))
    flow = MessageFlow(0, 1, 8 * 1024 * 1024)
    baseline, _ = _run(topology.transport_plan([flow], chunk_bytes=1024 * 1024))
    assert baseline["credit_waits"] == 0

    plan = topology.transport_plan(
        [flow], chunk_bytes=1024 * 1024,
        receive_capacity_bytes={1: 2 * 1024 * 1024})
    engine = Engine(plan.resources())
    ready = _ready(engine, 2)
    consumes = [engine.submit(f"decode_consume{index}", latency=10 * (index + 1))
                for index in range(4)]
    run = plan.start(engine, "pd", ready, release_events={1: tuple(consumes)})
    engine.advance(until=run.completions, first=False)
    report = run.report()
    assert report["credit_waits"] >= 1
    assert report["span_ms"] > baseline["span_ms"]
    assert all(value <= 2 * 1024 * 1024 for value in
               report["receiver_reserved_peak_bytes"].values())
    assert report["receiver_mean_resident_bytes"]["1"] > 0
    assert report["capacity_work_bytes"] == baseline["capacity_work_bytes"]


def test_nonuniform_moe_peer_chunks_conserve_each_directed_cell():
    from mindie_sim.prediction.peer_traffic import PeerTraffic
    from test_collective_cohorts import per_pair_topology

    count = 4
    matrix = ((0, 9, 3, 20), (7, 0, 11, 5), (2, 13, 0, 4), (17, 1, 6, 0))
    traffic = PeerTraffic.from_mapping(
        {"kind": "matrix", "payload_bytes": matrix}, count)
    topology = per_pair_topology(count, latency_us=5)
    volumes = {rank: traffic.row_total(rank) for rank in range(count)}
    for chunk in (None, 4):
        group = DirectPeerGroup(
            tuple(range(count)), tuple(volumes.items()), count - 1, topology,
            traffic, 2, chunk, 0.0, 64)
        engine = Engine(topology.resources())
        events = group.start_participants(
            engine, "moe", _ready(engine, count))
        engine.advance(until=tuple(events.values()), first=False)
        assert engine.summary()["live_tasks"] == 0
        assert sum(engine.serviced.values()) == pytest.approx(sum(map(sum, matrix)))
        assert all(event.done for event in events.values())




def test_declared_physical_chunks_keep_public_framing_and_wire():
    raw = case(pp=2)
    raw["machine"]["links"][0].update(
        transaction_bytes=64, protocol_overhead_bytes=0,
        capacity_bytes_per_ms=1_000_000)
    op = exchange(128, family="pp_p2p", axis="pp",
                  metadata=(("transport_chunk_bytes", 1),))
    parsed = parse_case(raw)
    topology = Topology(parsed.machine, parsed.runtime.parallel)
    work = topology.accounting(op, 0)
    # 128 declared 1 B chunks, each one 64 B transaction: 8192 wire bytes.
    assert work.useful_by_rank == {0: 128}
    assert work.messages == 128
    assert work.physical_wire_bytes == 8192
    assert 128 in work.transactions_by_resource.values()

    link = Link("f", 1_000_000, (0, 1), 0.0, "fabric", "shared",
                transaction_bytes=64)
    small = Topology(_machine((link,)), ParallelSpec(tp=1))
    plan = small.transport_plan([MessageFlow(0, 1, 128)], chunk_bytes=1)
    assert plan.lower_bound()["capacity_work_bytes"] == 8192
    report, _ = _run(plan)
    assert report["chunks"] == 128
    assert report["chunks"] == 128  # not silently aggregated to 64


def test_more_than_64_physical_chunks_keep_exact_work_and_fail_loudly():
    links = (
        Link("first", 1000, (0, 1), 0.0, "fabric", "shared",
             transaction_bytes=64),
        Link("second", 1000, (1, 2), 0.0, "fabric", "shared",
             transaction_bytes=64),
    )
    topology = Topology(_machine(links, devices=3), ParallelSpec(tp=1))
    plan = topology.transport_plan([MessageFlow(0, 2, 256)], chunk_bytes=1)
    # 256 physical chunks * 64 B transaction on each of two hops.
    assert plan.lower_bound()["capacity_work_bytes"] == 2 * 256 * 64
    report, _ = _run(plan)
    assert report["chunks"] == 256
    assert report["service_lower_bound_ms"] == pytest.approx(256 * 64 / 1000)
    with pytest.raises(ModelingError, match="transport_chunk_expansion_unbounded"):
        topology.transport_plan([MessageFlow(0, 2, 8192)], chunk_bytes=1)


def test_small_receive_credit_is_a_real_admission_bound_not_a_cap():
    link = Link("f", 60_000, (0, 1), 0.0, "fabric", "shared")
    topology = Topology(_machine((link,)), ParallelSpec(tp=1))
    plan = topology.transport_plan(
        [MessageFlow(0, 1, 128)], chunk_bytes=1,
        receive_capacity_bytes={1: 64})
    engine = Engine(plan.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in plan.ranks}
    run = plan.start(engine, "small", ready)
    engine.advance(until=run.completions, first=False)
    report = run.report()
    assert report["chunks"] == 128
    assert report["receiver_reserved_peak_bytes"][1] <= 64
    # Immediate consumption reports no resident area.
    assert report["receiver_resident_byte_us"][1] == pytest.approx(0.0)
    with pytest.raises(ModelingError, match="transport_receive_capacity_zero"):
        zero = topology.transport_plan(
            [MessageFlow(0, 1, 1)], receive_capacity_bytes={1: 0})
        zero.start(Engine(zero.resources()), "zero", {
            rank: engine.submit(f"z{rank}") for rank in zero.ranks})


def test_public_operator_consumes_link_endpoint_rates():
    durations = []
    for rate in (120_000, 30_000):
        raw = case(tp=2)
        raw["machine"]["links"][0].update(
            semantics="per_port", capacity_bytes_per_ms=60_000,
            latency_us=2, endpoint_rates={1: rate})
        raw["runtime"]["kernels"] = {"families": {"alltoall": {
            "kind": "direct_peer_collective", "max_inflight_per_rank": 1}}}
        row = operator_costs(raw, (exchange(1200),))["operators"][0]
        durations.append(row["duration_ms"])
        parsed = parse_case(raw)
        topology = Topology(parsed.machine, parsed.runtime.parallel)
        topology.resources()
        assert topology._resource_rates["link:fabric:rx:port1"] == rate
    # The slower placement receive port must change the ordinary prediction.
    assert durations[1] > durations[0]


def test_late_ready_chunk_does_not_starve_ready_flow_or_oversell_receiver():
    link = Link("f", 60_000, (0, 1, 2), 0.0, "fabric", "shared")
    topology = Topology(_machine((link,), devices=3), ParallelSpec(tp=1))
    for flows in (
            [MessageFlow(0, 2, 64), MessageFlow(1, 2, 64)],
            [MessageFlow(1, 2, 64), MessageFlow(0, 2, 64)]):
        plan = topology.transport_plan(flows, receive_capacity_bytes={2: 64})
        engine = Engine(plan.resources())
        ready = {
            0: engine.submit("r0", release=100.0),
            1: engine.submit("r1"),
            2: engine.submit("r2"),
        }
        run = plan.start(engine, "late", ready)
        engine.advance(until=run.completions, first=False)
        by_source = {item["source"]: item["complete_ms"]
                     for item in run.report()["flows"]}
        assert by_source[1] < 1.0
        assert by_source[0] > 99.0

    # Two independent plans on one Engine share the same 64 B destination pool.
    plan = topology.transport_plan([MessageFlow(0, 2, 64)],
                                   receive_capacity_bytes={2: 64})
    engine = Engine(plan.resources())
    ready = {rank: engine.submit(f"p{rank}") for rank in plan.ranks}
    first = plan.start(engine, "first", ready)
    second = plan.start(engine, "second", ready)
    engine.advance(until=first.completions + second.completions, first=False)
    assert engine.peak_allocated["transport:rx:rank:2"] <= 64
    assert first.completions[0].done and second.completions[0].done


def test_immediate_release_has_zero_resident_area_and_delayed_uses_b_us():
    link = Link("f", 60_000, (0, 1), 0.0, "fabric", "shared")
    topology = Topology(_machine((link,)), ParallelSpec(tp=1))
    plan = topology.transport_plan([MessageFlow(0, 1, 64)], chunk_bytes=64,
                                   receive_capacity_bytes={1: 64})
    immediate, _ = _run(plan)
    assert immediate["receiver_resident_byte_us"][1] == pytest.approx(0.0)
    assert immediate["receiver_resident_bytes_at_end"][1] == pytest.approx(0.0)

    engine = Engine(plan.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in plan.ranks}
    consume = engine.submit("consume", latency=10)
    run = plan.start(engine, "held", ready, release_events={1: (consume,)})
    engine.advance(until=run.completions + (consume,), first=False)
    report = run.report()
    # 64 B held about 10 ms is 64 * 10,000 us = 640,000 B*us.
    assert report["receiver_resident_byte_us"][1] == pytest.approx(
        640_000, rel=0.02)
    assert report["receiver_mean_resident_bytes"]["1"] == pytest.approx(
        64, rel=0.02)
    assert report["residency_window_ms"] == pytest.approx(10, rel=0.02)


def test_runtime_chunk_one_drives_operator_and_cohort_ledgers():
    from mindie_sim.prediction.direct_peer import DirectPeerCollective, resolve_transport_options
    from test_prediction_system import case as base_case

    def raw_case(dp):
        raw = base_case(tp=2, dp=dp)
        raw["machine"]["links"][0].update(
            transaction_bytes=64, protocol_overhead_bytes=0,
            capacity_bytes_per_ms=1_000_000)
        raw["runtime"]["kernels"] = {"families": {"alltoall": {
            "kind": "direct_peer_collective",
            "max_inflight_per_rank": 1, "chunk_bytes": 1}}}
        return raw

    op = exchange(128)
    for dp in (1, 2):
        raw = raw_case(dp)
        if dp == 2:
            report = operator_costs(raw, {0: (op,), 1: (op,)})
        else:
            report = operator_costs(raw, (op,))
        for row in report["operators"]:
            communication = row["communication"]
            assert communication["wire_bytes"] == 16_384
            assert row["serviced"]["link:fabric"] == 16_384
            assert communication["resources"]["link:fabric"]["wire_bytes"] == (
                16_384)
        assert all(row["serviced"]["link:fabric"] == 16_384
                   for row in report["operators"])

    recipe = DirectPeerCollective(chunk_bytes=1)
    options = resolve_transport_options(recipe, op)
    assert options.chunk_bytes == 1


def test_tail_and_per_hop_transactions_match_execution_and_ledger():
    raw = case(pp=3)
    raw["model"]["layers"] = 3
    raw["machine"]["links"] = [
        {"name": "first", "members": [0, 1], "capacity_bytes_per_ms": 1000,
         "transaction_bytes": 64, "semantics": "shared"},
        {"name": "second", "members": [1, 2], "capacity_bytes_per_ms": 1000,
         "transaction_bytes": 32, "semantics": "shared"},
    ]
    op = exchange(
        130, family="pp_p2p", axis="pp",
        metadata=(("destination_stage", 2), ("transport_chunk_bytes", 64)))
    parsed = parse_case(raw)
    topology = Topology(parsed.machine, parsed.runtime.parallel)
    ledger = topology.accounting(op, 0)
    # 64 + 64 + 2 B chunks: 3 frames on each hop, with different hop frames.
    assert ledger.messages == 3
    # first hop: 3 x 64 B; second hop frames each 64 B chunk as two 32 B
    # transactions and the 2 B tail as one 32 B transaction.
    assert ledger.physical_wire_bytes == 3 * 64 + (2 * 64 + 32)
    assert sum(ledger.wire_by_resource.values()) == 3 * 64 + (2 * 64 + 32)
    plan = topology.transport_plan(
        [MessageFlow(0, 2, 130)], chunk_bytes=64)
    assert plan.lower_bound()["capacity_work_bytes"] == 3 * 64 + (2 * 64 + 32)
    assert plan.lower_bound()["service_ms"] == pytest.approx(3 * 64 / 1000)


def test_namespaced_receive_buffers_are_independent_and_explicit_shared_id():
    link = Link("f", 60_000, (0, 1), 0.0, "fabric", "shared")
    base = _machine((link,))
    a = Topology(base, ParallelSpec(tp=1), "a:")
    b = Topology(base, ParallelSpec(tp=1), "b:")
    pa = a.transport_plan([MessageFlow(0, 1, 64)],
                          receive_capacity_bytes={1: 64})
    pb = b.transport_plan([MessageFlow(0, 1, 64)],
                          receive_capacity_bytes={1: 128})
    engine = Engine(pa.resources() + pb.resources())
    ready = {rank: engine.submit(f"n{rank}") for rank in pa.ranks}
    ra = pa.start(engine, "a", ready)
    rb = pb.start(engine, "b", ready)
    engine.advance(until=ra.completions + rb.completions, first=False)
    assert ra.completions[0].done and rb.completions[0].done
    assert "transport:rx:a:rank:1" in engine.pools
    assert "transport:rx:b:rank:1" in engine.pools

    shared_a = a.transport_plan(
        [MessageFlow(0, 1, 64)], receive_capacity_bytes={1: 64},
        receive_buffer_id={1: "physical-shared"})
    shared_b = b.transport_plan(
        [MessageFlow(0, 1, 64)], receive_capacity_bytes={1: 64},
        receive_buffer_id={1: "physical-shared"})
    engine2 = Engine(shared_a.resources() + shared_b.resources())
    ready2 = {rank: engine2.submit(f"m{rank}") for rank in shared_a.ranks}
    r1 = shared_a.start(engine2, "a", ready2)
    r2 = shared_b.start(engine2, "b", ready2)
    engine2.advance(until=r1.completions + r2.completions, first=False)
    assert engine2.peak_allocated["transport:rx:physical-shared"] <= 64
    with pytest.raises(ModelingError, match="transport_receive_capacity_conflict"):
        conflicting = b.transport_plan(
            [MessageFlow(0, 1, 64)], receive_capacity_bytes={1: 128},
            receive_buffer_id={1: "physical-shared"})
        conflicting.start(engine2, "c", ready2)


def test_two_stage_consume_releases_each_arrival_and_holds_b_us():
    link = Link("f", 64, (0, 1), 0.0, "fabric", "shared")
    topology = Topology(_machine((link,)), ParallelSpec(tp=1))
    plan = topology.transport_plan(
        [MessageFlow(0, 1, 128)], chunk_bytes=64,
        receive_capacity_bytes={1: 64})
    engine = Engine(plan.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in plan.ranks}
    consume2 = engine.submit("consume2", latency=2)
    consume4 = engine.submit("consume4", latency=4)
    run = plan.start(engine, "consume", ready,
                     release_events={1: (consume2, consume4)})
    engine.advance(until=run.completions + (consume2, consume4),
                   first=False)
    report = run.report()
    assert run.completions[0].time == pytest.approx(3.0)
    assert report["receiver_resident_byte_us"][1] == pytest.approx(
        128_000, rel=0.02)
    assert report["receiver_resident_bytes_at_end"][1] == pytest.approx(0.0)


def test_direct_peer_receive_credit_uses_one_shared_engine_allocation():
    links = (Link("f", 64, (0, 1), 0.0, "fabric", "shared"),)
    topology = Topology(_machine(links), ParallelSpec(tp=1), "peer:")
    group = DirectPeerGroup(
        (0, 1), ((0, 64.0), (1, 64.0)), 1, topology,
        chunk_bytes=64, receive_capacity_bytes_per_peer=64)
    engine = Engine(topology.resources())
    ready = {rank: engine.submit(f"p{rank}") for rank in (0, 1)}
    first = group.start_participants(engine, "one", ready)
    second = group.start_participants(engine, "two", ready)
    engine.advance(until=tuple(first.values()) + tuple(second.values()),
                   first=False)
    assert engine.peak_allocated["transport:rx:peer:peer:0>1"] <= 64
    assert engine.peak_allocated["transport:rx:peer:peer:1>0"] <= 64
    assert all(event.done for event in (*first.values(), *second.values()))


def test_transport_options_preserve_peer_window_and_physical_expansion_limit():
    from mindie_sim.prediction.direct_peer import DirectPeerCollective, resolve_transport_options

    raw = case(tp=3)
    raw["model"].update(num_heads=6, num_kv_heads=3)
    raw["machine"]["links"] = [{
        "name": "f", "members": [0, 1, 2], "capacity_bytes_per_ms": 64,
        "semantics": "per_pair", "latency_us": 0}]
    parsed = parse_case(raw)
    topology = Topology(parsed.machine, parsed.runtime.parallel)
    op = exchange(128, metadata=(("steps", 2),))
    expected = {2: 1.0, None: 1.0, 1: 2.0}
    for window, span in expected.items():
        recipe = DirectPeerCollective(
            max_inflight_per_rank=window, chunk_bytes=64)
        options = resolve_transport_options(recipe, op)
        assert options.max_inflight_per_rank == window
        assert options.plan_kwargs()["max_inflight_per_rank"] == window
        plan = topology.joint_plan(
            op, {0: 128}, algorithm="direct_peer", transport=options)
        group = plan.groups[0]
        assert group.max_inflight_per_rank == window
        engine = Engine(topology.resources())
        ready = engine.submit("ready")
        events = plan.start_participants(
            engine, "window", {rank: ready for rank in plan.ranks})
        engine.advance(until=tuple(events.values()), first=False)
        assert max(event.time for event in events.values()) == pytest.approx(span)
        assert sum(engine.serviced.values()) == pytest.approx(384.0)

    # The ledger uses the same expansion limit as execution.
    opts = TransportOptions(chunk_bytes=1, max_physical_chunks=8192)
    tail_raw = case(pp=2)
    tail_raw["machine"]["links"][0].update(
        transaction_bytes=64, protocol_overhead_bytes=0,
        capacity_bytes_per_ms=1_000_000)
    tail_op = exchange(
        4097, family="pp_p2p", axis="pp",
        metadata=(("transport_chunk_bytes", 1),))
    tail_topology = Topology(
        parse_case(tail_raw).machine, parse_case(tail_raw).runtime.parallel)
    ledger = tail_topology.accounting(tail_op, 0, transport=opts)
    assert ledger.physical_wire_bytes == 4097 * 64
    plan = tail_topology.transport_plan(
        [MessageFlow(0, 1, 4097)], transport=TransportOptions(
            chunk_bytes=1, max_physical_chunks=8192,
            receive_capacity_bytes=((1, 4097 * 64),)))
    assert plan.lower_bound()["capacity_work_bytes"] == 4097 * 64
    large_engine = Engine(plan.resources())
    large_ready = {rank: large_engine.submit(f"large{rank}")
                   for rank in plan.ranks}
    large_run = plan.start(large_engine, "large", large_ready)
    large_engine.advance(until=large_run.completions, first=False)
    assert large_engine.serviced["link:fabric"] == pytest.approx(4097 * 64)
    assert large_run.report()["chunks"] == 4097

    from mindie_sim.prediction.hccl_graph import SourceMeshPeers
    from mindie_sim.prediction.hccl_contracts import select as hccl_select
    selection = hccl_select(
        {"expansion": "AICPU_TS", "template": "mesh_write",
         "dtype_bytes": 1}, collective="alltoallv", ranks=3,
        payload_bytes=128)
    hccl_op = SimpleNamespace(metadata=(
        ("hccl_selection", selection.to_dict()),))
    hccl_options = SourceMeshPeers(path_model="store_forward").resolve_transport(
        hccl_op)
    assert hccl_options.max_inflight_per_rank == 2


def test_explicit_consumer_exhaustion_holds_future_arrivals():
    link = Link("f", 64, (0, 1), 0.0, "fabric", "shared")
    topology = Topology(_machine((link,)), ParallelSpec(tp=1))
    plan = topology.transport_plan(
        [MessageFlow(0, 1, 256)], chunk_bytes=64,
        receive_capacity_bytes={1: 64})
    engine = Engine(plan.resources())
    ready = {rank: engine.submit(f"r{rank}") for rank in plan.ranks}
    consume2 = engine.submit("c2", latency=2)
    consume4 = engine.submit("c4", latency=4)
    run = plan.start(engine, "exhaust", ready,
                     release_events={1: (consume2, consume4)})
    engine.advance(horizon=6.5, first=False)
    report = run.report()
    # The third block arrives at 5 ms after the explicit consumers are gone:
    # it must stay resident and block the fourth block's credit.
    assert report["chunks"] == 4
    assert sum(engine.serviced.values()) == pytest.approx(192.0)
    assert report["receiver_resident_bytes_at_end"][1] == pytest.approx(64.0)
    assert report["receiver_reserved_peak_bytes"][1] == pytest.approx(64.0)
    assert not run.completions[0].done


def test_full_consumer_sequence_releases_each_block_and_zeroes_residency():
    link = Link("f", 64, (0, 1), 0.0, "fabric", "shared")
    topology = Topology(_machine((link,)), ParallelSpec(tp=1))
    plan = topology.transport_plan(
        [MessageFlow(0, 1, 256)], chunk_bytes=64,
        receive_capacity_bytes={1: 64})
    engine = Engine(plan.resources())
    ready = {rank: engine.submit(f"r{rank}") for rank in plan.ranks}
    events = tuple(engine.submit(f"consume{ms}", latency=ms)
                   for ms in (2, 4, 8, 10))
    run = plan.start(engine, "full", ready,
                     release_events={1: events})
    engine.advance(until=run.completions + events, first=False)
    report = run.report()
    assert run.completions[0].time == pytest.approx(9.0)
    assert report["receiver_resident_bytes_at_end"][1] == pytest.approx(0.0)
    assert report["receiver_resident_byte_us"][1] == pytest.approx(
        384_000, rel=0.02)


def test_consumer_before_arrival_does_not_release_future_data():
    link = Link("f", 64, (0, 1), 0.0, "fabric", "shared")
    topology = Topology(_machine((link,)), ParallelSpec(tp=1))
    plan = topology.transport_plan(
        [MessageFlow(0, 1, 64)], chunk_bytes=64,
        receive_capacity_bytes={1: 64})
    engine = Engine(plan.resources())
    ready = {rank: engine.submit(f"r{rank}") for rank in plan.ranks}
    early = engine.submit("early_consume", latency=0.5)
    run = plan.start(engine, "early", ready, release_events={1: (early,)})
    engine.advance(horizon=2.0, first=False)
    report = run.report()
    # The consumer fired before the block arrived; it must not consume it.
    assert report["receiver_resident_bytes_at_end"][1] == pytest.approx(64.0)
    assert engine.allocated["transport:rx:rank:1"] == pytest.approx(64.0)


@pytest.mark.parametrize("ranks", [(2, 3), (3, 2), (1, 3)])
def test_direct_peer_consumer_exhaustion_holds_pool_credit(ranks):
    from test_collective_cohorts import per_pair_topology

    topology = per_pair_topology(4, capacity=64, latency_us=0)
    group = DirectPeerGroup(
        ranks, tuple((rank, 256.0) for rank in ranks), 1, topology,
        chunk_bytes=64, receive_capacity_bytes_per_peer=64)
    engine = Engine(topology.resources())
    ready = {rank: engine.submit(f"r{rank}") for rank in ranks}
    events = {}
    for pair in (ranks, tuple(reversed(ranks))):
        events[pair] = (
            engine.submit(f"c{pair[0]}{pair[1]}2", latency=2),
            engine.submit(f"c{pair[0]}{pair[1]}4", latency=4),
        )
    done = group.start_participants(
        engine, "peer-consumer", ready, release_events=events)
    engine.advance(horizon=6.5, first=False)
    assert sum(value for name, value in engine.serviced.items()
               if name.startswith("link:")) == pytest.approx(384.0)
    for source, destination in events:
        assert engine.allocated[
            f"transport:rx:peer:{source}>{destination}"] == pytest.approx(64.0)
    assert not any(event.done for event in done.values())


def test_direct_peer_full_consumer_sequence_releases_pools():
    from test_collective_cohorts import per_pair_topology

    ranks = (2, 3)
    topology = per_pair_topology(4, capacity=64, latency_us=0)
    group = DirectPeerGroup(
        ranks, ((2, 256.0), (3, 256.0)), 1, topology,
        chunk_bytes=64, receive_capacity_bytes_per_peer=64)
    engine = Engine(topology.resources())
    ready = {rank: engine.submit(f"r{rank}") for rank in ranks}
    events = {}
    for pair in (ranks, tuple(reversed(ranks))):
        events[pair] = tuple(
            engine.submit(f"c{pair[0]}{pair[1]}{ms}", latency=ms)
            for ms in (2, 4, 8, 10))
    done = group.start_participants(
        engine, "peer-full", ready, release_events=events)
    all_events = tuple(done.values()) + tuple(
        event for sequence in events.values() for event in sequence)
    engine.advance(until=all_events, first=False)
    assert max(event.time for event in done.values()) == pytest.approx(9.0)
    assert sum(value for name, value in engine.serviced.items()
               if name.startswith("link:")) == pytest.approx(512.0)
    assert all(engine.allocated[
        f"transport:rx:peer:{source}>{destination}"] == pytest.approx(0.0)
        for source, destination in events)


@pytest.mark.parametrize("mode", ["empty", "early", "default"])
def test_direct_peer_consumer_modes_distinguish_empty_early_and_arrival(mode):
    from test_collective_cohorts import per_pair_topology

    ranks = (2, 3)
    topology = per_pair_topology(4, capacity=64, latency_us=0)
    group = DirectPeerGroup(
        ranks, ((2, 256.0), (3, 256.0)), 1, topology,
        chunk_bytes=64, receive_capacity_bytes_per_peer=64)
    engine = Engine(topology.resources())
    ready = {rank: engine.submit(f"r{rank}") for rank in ranks}
    if mode == "empty":
        events = {pair: () for pair in (ranks, tuple(reversed(ranks)))}
    elif mode == "early":
        events = {
            pair: (engine.submit(f"early{pair[0]}", latency=0.5),)
            for pair in (ranks, tuple(reversed(ranks)))}
    else:
        events = None
    done = group.start_participants(
        engine, f"peer-{mode}", ready, release_events=events)
    if mode == "default":
        engine.advance(until=tuple(done.values()), first=False)
        assert max(event.time for event in done.values()) == pytest.approx(4.0)
        assert sum(value for name, value in engine.serviced.items()
                   if name.startswith("link:")) == pytest.approx(512.0)
        assert all(engine.allocated[
            f"transport:rx:peer:{source}>{destination}"] == pytest.approx(0.0)
            for source, destination in (ranks, tuple(reversed(ranks))))
    else:
        engine.advance(horizon=6.5, first=False)
        assert sum(value for name, value in engine.serviced.items()
                   if name.startswith("link:")) == pytest.approx(128.0)
        assert engine.allocated[
            f"transport:rx:peer:{ranks[0]}>{ranks[1]}"] == pytest.approx(64.0)
        assert not any(event.done for event in done.values())


def test_joint_plan_nonzero_cross_replica_group_uses_physical_consumers():
    from test_communication_topology import topology as topology_from_raw

    raw = case(dp=4, tp=2, ep=4)
    raw["model"].update(
        ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    raw["machine"]["links"] = [{
        "name": "f", "members": list(range(8)),
        "capacity_bytes_per_ms": 64, "semantics": "per_pair",
        "latency_us": 0}]
    topology = topology_from_raw(raw)
    op = exchange(768, family="alltoall", axis="ep")
    plan = topology.joint_plan(
        op, {0: 768, 1: 768, 2: 768, 3: 768},
        algorithm="direct_peer", chunk_bytes=64,
        max_inflight_per_rank=3, receive_capacity_bytes_per_peer=64)
    assert [group.ranks for group in plan.groups] == [
        (0, 1, 2, 3), (4, 5, 6, 7)]
    group = plan.groups[1]
    ranks = group.ranks
    engine = Engine(topology.resources())
    ready = {rank: engine.submit(f"r{rank}") for rank in ranks}
    events = {
        (source, destination): (
            engine.submit(f"c{source}{destination}2", latency=2),
            engine.submit(f"c{source}{destination}4", latency=4))
        for source in ranks for destination in ranks if source != destination}
    done = group.start_participants(
        engine, "cross-replica", ready, release_events=events)
    engine.advance(horizon=6.5, first=False)
    assert sum(value for name, value in engine.serviced.items()
               if name.startswith("link:")) == pytest.approx(2304.0)
    assert all(engine.allocated[
        f"transport:rx:peer:{source}>{destination}"] == pytest.approx(64.0)
        for source, destination in events)
    assert not any(event.done for event in done.values())


def test_large_message_streams_bounded_chunks_without_history():
    link = Link("f", 64, (0, 1), 0.0, "fabric", "shared",
                transaction_bytes=64)
    topology = Topology(_machine((link,)), ParallelSpec(tp=1))
    plan = topology.transport_plan(
        [MessageFlow(0, 1, 4097)], chunk_bytes=1,
        max_physical_chunks=None, receive_capacity_bytes={1: 64})
    assert len(plan.flows[0].chunks) == 4097
    engine = Engine(plan.resources())
    ready = {rank: engine.submit(f"r{rank}") for rank in plan.ranks}
    run = plan.start(engine, "stream", ready)
    engine.advance(until=run.completions, first=False)
    report = run.report()
    assert report["chunks"] == 4097
    assert engine.serviced["link:f"] == pytest.approx(4097 * 64)
    # The declared 64 B slot keeps 64 arrived chunks resident plus at most
    # one look-ahead chunk; the full 4097-frame history is never live.
    assert report["peak_active_chunks"] <= 65
    assert report["peak_active_chunks"] < report["chunks"]
    assert report["active_chunks_at_end"] == 0

    tail = topology.transport_plan(
        [MessageFlow(0, 1, 4097.5)], chunk_bytes=64,
        max_physical_chunks=None, receive_capacity_bytes={1: 64})
    assert len(tail.flows[0].chunks) == 65
    tail_engine = Engine(tail.resources())
    tail_ready = {rank: tail_engine.submit(f"t{rank}") for rank in tail.ranks}
    tail_run = tail.start(tail_engine, "tail", tail_ready)
    tail_engine.advance(until=tail_run.completions, first=False)
    assert tail_engine.serviced["link:f"] == pytest.approx(65 * 64)


def test_direct_peer_large_message_uses_lazy_chunks():
    from test_collective_cohorts import per_pair_topology

    topology = per_pair_topology(2, capacity=64, latency_us=0)
    group = DirectPeerGroup(
        (0, 1), ((0, 4097.0), (1, 4097.0)), 1, topology,
        chunk_bytes=1, receive_capacity_bytes_per_peer=64,
        max_physical_chunks=None)
    engine = Engine(topology.resources())
    ready = {rank: engine.submit(f"r{rank}") for rank in (0, 1)}
    done = group.start_participants(engine, "peer-large", ready)
    engine.advance(until=tuple(done.values()), first=False)
    assert sum(value for name, value in engine.serviced.items()
               if name.startswith("link:")) == pytest.approx(2 * 4097)
    assert engine.summary()["live_tasks"] == 0


def test_shared_and_independent_receive_pools_admit_successors():
    link = Link("f", 64, (0, 1, 2), 0.0, "fabric", "shared")
    topology = Topology(_machine((link,), devices=3), ParallelSpec(tp=1))
    shared_a = topology.transport_plan(
        [MessageFlow(0, 1, 64)], chunk_bytes=64,
        receive_capacity_bytes={1: 64})
    shared_b = topology.transport_plan(
        [MessageFlow(2, 1, 64)], chunk_bytes=64,
        receive_capacity_bytes={1: 64})
    engine = Engine(shared_a.resources())
    ready0 = engine.submit("ready0")
    ready15 = engine.submit("ready15", latency=1.5)
    consume2 = engine.submit("consumeA2", latency=2)
    consume4 = engine.submit("consumeB4", latency=4)
    run_a = shared_a.start(
        engine, "A", {0: ready0, 1: ready0},
        release_events={1: (consume2,)})
    run_b = shared_b.start(
        engine, "B", {2: ready15, 1: ready0},
        release_events={1: (consume4,)})
    engine.advance(
        until=run_a.completions + run_b.completions
        + (consume2, consume4), first=False)
    assert run_a.completions[0].time == pytest.approx(1.0)
    assert run_b.completions[0].time == pytest.approx(3.0)
    assert engine.peak_allocated["transport:rx:rank:1"] <= 64
    assert engine.allocated["transport:rx:rank:1"] == pytest.approx(0.0)
    assert run_b.report()["credit_wait_ms"][1] == pytest.approx(0.5)

    independent = topology.transport_plan(
        [MessageFlow(2, 0, 64)], chunk_bytes=64,
        receive_capacity_bytes={0: 64},
        receive_buffer_id={0: "independent-B"})
    engine2 = Engine(independent.resources())
    ready15b = engine2.submit("ready15b", latency=1.5)
    ready0b = engine2.submit("ready0b")
    independent_run = independent.start(
        engine2, "B-independent", {2: ready15b, 0: ready0b})
    engine2.advance(until=independent_run.completions, first=False)
    assert independent_run.completions[0].time == pytest.approx(2.5)
