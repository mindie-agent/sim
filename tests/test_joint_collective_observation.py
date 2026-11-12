"""Normal serving keeps per-rank producer readiness through the rendezvous."""
import json

import pytest

from mindie_sim import predict
from test_joint_collective_execution import cohort_case


def _case(peer_tokens, *, profile=None):
    raw = cohort_case(2)
    raw["machine"]["links"][0].update(capacity_bytes_per_ms=1e9)
    raw["workload"]["requests"] = [
        {"id": "early", "prompt_tokens": 4, "output_tokens": 1},
        {"id": "peer", "prompt_tokens": peer_tokens, "output_tokens": 1},
    ]
    if profile is not None:
        raw["runtime"].setdefault("metadata", {})[
            "collective_participant_profile"] = profile
    return raw


def _run(peer_tokens, *, event_limit=64, profile=None):
    out = predict(_case(peer_tokens, profile=profile), event_limit=event_limit,
                  trace_limit=1)
    execution = out["execution"]
    return out, execution.get("joint_collective_observations", []), execution


def _first_dispatch(observations):
    return next(item for item in observations
                if item["op_id"] == "s0.l0.dispatch")


def test_joint_trace_is_bounded_and_contains_numbers_not_events():
    out, observations, execution = _run(32, event_limit=3)
    assert out["status"] == "modeled"
    assert len(observations) == 3
    stats = execution["joint_collective_stats"]
    assert stats["observations"] > stats["retained"] == 3
    # The returned trace is JSON data, not live Event handles.
    json.dumps(observations)
    for item in observations:
        assert set(item["ready_ms_by_rank"]) == set(item["completion_ms_by_rank"])
        assert all(isinstance(value, (int, float))
                   for value in item["ready_ms_by_rank"].values())
        assert all(isinstance(value, (int, float))
                   for value in item["completion_ms_by_rank"].values())
        assert item["source"]["resolved"] is False


def test_absent_trace_limit_keeps_the_existing_summary_shape():
    out, observations, execution = _run(32, event_limit=0)
    assert observations == []
    assert "joint_collective_observations" not in execution
    assert "joint_collective_stats" not in execution


def test_peer_rank_work_delays_the_next_result_through_rendezvous():
    equal, equal_observations, _ = _run(4)
    unequal, unequal_observations, _ = _run(32)
    equal_first = _first_dispatch(equal_observations)
    unequal_first = _first_dispatch(unequal_observations)

    # "early" keeps exactly the same four-token request and the same local
    # producer work in both runs.  Only its peer's genuine model work differs.
    assert equal_first["ready_ms_by_rank"][0] == pytest.approx(
        unequal_first["ready_ms_by_rank"][0])
    assert equal_first["ready_spread_ms"] == pytest.approx(0.0)
    assert unequal_first["ready_spread_ms"] > 0
    assert unequal_first["critical_rank"] == 1
    assert unequal_first["rendezvous_ready_ms"] == pytest.approx(
        unequal_first["ready_ms_by_rank"][1])
    assert unequal_first["completion_ms_by_rank"][0] == pytest.approx(
        unequal_first["completion_ms_by_rank"][1])
    assert unequal_first["completion_ms_by_rank"][0] > equal_first["completion_ms_by_rank"][0]

    equal_ttft = {row["request_id"]: row["ttft_ms"] for row in equal["requests"]}
    unequal_ttft = {row["request_id"]: row["ttft_ms"] for row in unequal["requests"]}
    assert unequal_ttft["early"] > equal_ttft["early"]
    assert unequal_ttft["peer"] > unequal_ttft["early"]


def test_swapping_peer_work_swaps_the_critical_rank_and_delayed_result():
    equal, _, _ = _run(4)
    left, left_observations, _ = _run(32)
    right_raw = _case(4)
    right_raw["workload"]["requests"][0]["prompt_tokens"] = 32
    right = predict(right_raw, event_limit=64, trace_limit=1)
    left_first = _first_dispatch(left_observations)
    right_first = _first_dispatch(
        right["execution"]["joint_collective_observations"])
    assert left_first["critical_rank"] == 1
    assert right_first["critical_rank"] == 0
    equal_ttft = {row["request_id"]: row["ttft_ms"] for row in equal["requests"]}
    left_ttft = {row["request_id"]: row["ttft_ms"] for row in left["requests"]}
    right_ttft = {row["request_id"]: row["ttft_ms"] for row in right["requests"]}
    # The same four-token baseline item is delayed by the slow peer in both
    # arrangements, so the symmetric swap reproduces the same delay.
    assert left_ttft["early"] > equal_ttft["early"]
    assert right_ttft["peer"] > equal_ttft["peer"]
    assert right_ttft["peer"] == pytest.approx(left_ttft["early"])


def test_participant_profile_only_annotates_and_never_prices_observed_clocks():
    profile = {
        "schema": "mindie-sim/collective-participant-profile/1",
        "source": {"capture_id": "real-source"},
        "calls": [{
            "ordinal": 1, "primitive": "all_to_all",
            "native_name": "hcom_alltoallv__369_1_1",
            "group_size": 2, "dtype": None,
            "ranks": [{"rank_id": 9, "role": "primary"},
                      {"rank_id": 4, "role": "idle"}],
        }],
    }
    baseline, baseline_observations, _ = _run(32)
    annotated, annotated_observations, execution = _run(32, profile=profile)
    assert annotated["execution"]["time_ms"] == pytest.approx(
        baseline["execution"]["time_ms"])
    assert annotated["metrics"] == baseline["metrics"]
    first = _first_dispatch(annotated_observations)
    assert first["source"]["resolved"] is True
    assert first["source"]["ordinal"] == 1
    assert first["source"]["native_name"] == "hcom_alltoallv__369_1_1"
    # The genuine source has no declared dtype; keep the row honest rather
    # than inventing one or rejecting the correlation.
    assert first["source"]["dtype"] is None
    assert first["source"]["dtype_basis"] == "underdetermined_source"
    # The annotation is separate from the predicted boundaries.
    assert first["ready_ms_by_rank"] == pytest.approx(
        _first_dispatch(baseline_observations)["ready_ms_by_rank"])
    assert execution["joint_collective_stats"]["observations"] >= 1


def test_common_barrier_retains_local_rank_readiness_and_wait():
    raw = _case(32)
    raw["runtime"]["kernels"] = {}
    out = predict(raw, event_limit=64, trace_limit=1)
    observations = out["execution"]["joint_collective_observations"]
    first = _first_dispatch(observations)
    assert first["barrier_ms"] == pytest.approx(
        max(first["ready_ms_by_rank"].values()))
    assert first["rendezvous_ready_ms"] == pytest.approx(first["barrier_ms"])
    assert first["ready_spread_ms"] > 0
    early = next(item for item in first["participants"] if item["replica"] == 0)
    peer = next(item for item in first["participants"] if item["replica"] == 1)
    assert early["wait_ms"][0] > 0
    assert peer["wait_ms"][1] == pytest.approx(0.0)
    assert early["parent_complete_ms"] == pytest.approx(
        peer["parent_complete_ms"])


def test_participant_profile_rejects_declared_mismatch_but_keeps_absent_unknowns():
    from mindie_sim.prediction.participant_profile import normalise, select_call
    profile = normalise({
        "schema": "mindie-sim/collective-participant-profile/1",
        "calls": [{"ordinal": 1, "primitive": "all_to_all",
                   "native_name": "hcom_alltoallv__369_1_1",
                   "group_size": 2, "dtype": "float32",
                   "ranks": [{"rank_id": 9, "role": "primary"}]}],
    })
    assert select_call(profile, primitive="all_to_all", group_size=2,
                       dtype="float32")["resolved"] is True
    mismatch = select_call(profile, primitive="all_to_all", group_size=2,
                           dtype="fp8_e4m3")
    assert mismatch["resolved"] is False
    assert "source_dtype_mismatch" in mismatch["reason"]
    unknown = select_call(profile, primitive="all_to_all", group_size=2,
                          dtype=None)
    assert unknown["resolved"] is True


def test_participant_profile_does_not_change_aggregate_barrier_timing():
    profile = {
        "schema": "mindie-sim/collective-participant-profile/1",
        "calls": [{
            "ordinal": 1, "primitive": "all_to_all",
            "native_name": "hcom_alltoallv__369_1_1",
            "group_size": 2, "dtype": None,
            "ranks": [{"rank_id": 9, "role": "primary"},
                      {"rank_id": 4, "role": "idle"}],
        }],
    }
    raw = _case(32)
    raw["runtime"]["kernels"] = {}
    baseline = predict(raw, event_limit=64, trace_limit=1)
    annotated_raw = _case(32, profile=profile)
    annotated_raw["runtime"]["kernels"] = {}
    annotated = predict(annotated_raw, event_limit=64, trace_limit=1)
    assert annotated["execution"]["time_ms"] == pytest.approx(
        baseline["execution"]["time_ms"])
    assert annotated["metrics"] == baseline["metrics"]
    first = _first_dispatch(
        annotated["execution"]["joint_collective_observations"])
    assert first["barrier_ms"] == pytest.approx(
        _first_dispatch(baseline["execution"]["joint_collective_observations"])[
            "barrier_ms"])
    assert first["source"]["resolved"] is True


def test_padded_idle_keeps_physical_graph_without_a_logical_request():
    from mindie_sim.prediction.body import body_forward
    raw = cohort_case(2)
    raw["runtime"]["graph_mode"] = "full"
    raw["runtime"]["graph_capture_mode"] = "full"
    raw["runtime"]["graph_capture_keys"] = [1, 2, 4]
    out = body_forward(raw, phase="decode", tokens=1, context=4,
                       sequences=[1, 0], event_limit=64, compact=True,
                       padded_idle=True)
    assert out["geometry"]["valid_query_tokens_by_replica"] == {0: 1, 1: 0}
    assert out["geometry"]["graph_query_tokens_by_replica"] == {0: 1, 1: 1}
    assert out["geometry"]["padded_query_tokens_by_replica"] == {0: 1, 1: 1}
    assert out["profile"]["by_rank"][1]["operator_events"] > 0
    first = _first_dispatch(
        out["execution"]["joint_collective_observations"])
    assert first["ready_ms_by_rank"][1] > 0
    assert first["ready_ms_by_rank"][0] > 0


def test_native_peer_start_uses_local_ready_and_peer_events():
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.piecewise.compiler import NativeProgram
    from mindie_sim.prediction.program import KernelProgram, Stage

    class _PeerProgram:
        rank = 0
        peer_labels = (1,)

        def __init__(self):
            self.supplied = None
            self.events = {}

        def start(self, engine, name, ready, peer_ready=None, **kwargs):
            self.supplied = dict(peer_ready or {})
            send = engine.submit(name + ":send", after=(ready,))
            receive = engine.submit(name + ":receive",
                                    after=(send, peer_ready[1]))
            finish = engine.submit(name + ":retire", after=(receive,))
            self.events = {"send": send, "receive": receive, "finish": finish}
            return finish

    engine = Engine([])
    ready = engine.submit("ready", release=10.0)
    local = engine.submit("local", release=5.0)
    peer = engine.submit("peer", release=20.0)
    inner = _PeerProgram()
    execution = object.__new__(Execution)
    execution.engine = engine
    finish = execution.start_native_participant(
        NativeProgram(inner, (), 0), "native", ready,
        local_ready=local, peer_ready={1: peer})
    engine.advance(until=(finish,))
    assert inner.supplied == {1: peer}
    assert inner.events["send"].time == pytest.approx(5.0)
    assert inner.events["receive"].time == pytest.approx(20.0)
    assert finish.time == pytest.approx(20.0)

    # A non-native aggregate program keeps the common cohort barrier.
    engine = Engine([Resource("work", 1.0)], trace_limit=10)
    ready = engine.submit("ready", release=10.0)
    local = engine.submit("local", release=5.0)
    execution.engine = engine
    finish = execution.start_native_participant(
        KernelProgram((Stage("stage", work={"work": 1.0}),)), "aggregate",
        ready, local_ready=local, peer_ready={1: peer})
    engine.advance(until=(finish,))
    assert finish.time == pytest.approx(11.0)
    assert any(row["task"] == "aggregate:cohort_gate" and row["end_ms"] == 10.0
               for row in engine.trace)


def test_native_cohort_binds_actual_sender_send_and_keeps_cached_program():
    from mindie_sim.prediction.engine import Engine
    from mindie_sim.prediction.execution import launch_native_cohort
    from mindie_sim.prediction.piecewise.compiler import NativeProgram
    from mindie_sim.prediction.piecewise.families_moe_mc2 import Mc2LaunchView

    class _Phase:
        name = "count_status_exchange"
        after = ()

    class _Inner:
        phases = (_Phase(),)
        link_by_phase = {"count_status_exchange": {"link": 1}}
        peer_count = 2
        view = Mc2LaunchView(rank=0, peer_labels=(1,), node_ranks=2,
                             intra_latency_ms=0, inter_latency_ms=0)

        def start_with_handles(self, engine, name, ready, peer_ready=None,
                               peer_send=None, launch=None, source_send=None,
                               send_placeholders=None):
            peer = launch.peer_labels[0]
            placeholders = dict(send_placeholders or {})
            own = placeholders.get((launch.rank, peer, "count_status_exchange"))
            incoming = placeholders.get((peer, launch.rank, "count_status_exchange"))
            send = engine.submit(
                name + ":send", after=(ready,),
                on_complete=((lambda _event, promise=own:
                              engine.resolve(promise)) if own is not None
                             else None))
            receive = engine.submit(
                name + ":receive", after=(send, incoming))
            complete = engine.submit(name + ":complete", after=(receive,))
            return complete, {
                "peer_events": {"count_status_exchange": {peer: {
                    "send": send, "receive": receive}}},
                "phase_complete": {"count_status_exchange": receive},
            }

    inner = _Inner()
    program = NativeProgram(inner, (), 0)
    original_rank, original_labels = inner.view.rank, inner.view.peer_labels

    def run(rank1_ready):
        engine = Engine([], trace_limit=64)
        ready0 = engine.submit("ready0", release=0.0)
        ready1 = engine.submit("ready1", release=rank1_ready)
        result = launch_native_cohort(engine, [
            {"program": program, "name": "r0", "ready": ready0,
             "rank_index": 0, "peer_count": 2},
            {"program": program, "name": "r1", "ready": ready1,
             "rank_index": 1, "peer_count": 2},
        ])
        engine.advance(until=(result["join"],))
        p0 = result["handles"]["r0"]["peer_events"]["count_status_exchange"][1]
        p1 = result["handles"]["r1"]["peer_events"]["count_status_exchange"][0]
        return {
            "r0_send": p0["send"].time, "r0_receive": p0["receive"].time,
            "r1_send": p1["send"].time, "r1_receive": p1["receive"].time,
            "r0_parent": result["parents"]["r0"].time,
            "r1_parent": result["parents"]["r1"].time,
            "join": result["join"].time,
            "unresolved": result["unresolved_edges"],
        }

    early = run(1.0)
    late = run(6.0)
    assert early["r0_receive"] == pytest.approx(1.0)
    assert late["r0_receive"] == pytest.approx(6.0)
    assert early["r1_receive"] == pytest.approx(1.0)
    assert late["r1_receive"] == pytest.approx(6.0)
    assert late["r0_parent"] == pytest.approx(6.0)
    assert late["join"] == pytest.approx(6.0)
    assert early["unresolved"] == late["unresolved"] == 0
    # The shared cached native program is never mutated by either launch.
    assert inner.view.rank == original_rank
    assert inner.view.peer_labels == original_labels


def test_native_cohort_does_not_invent_idle_phase_send():
    from mindie_sim.prediction.engine import Engine
    from mindie_sim.prediction.execution import launch_native_cohort
    from mindie_sim.prediction.piecewise.compiler import NativeProgram
    from mindie_sim.prediction.piecewise.families_moe_mc2 import Mc2LaunchView

    class _Phase:
        name = "count_status_exchange"
        after = ()

    class _Active:
        phases = (_Phase(),)
        link_by_phase = {"count_status_exchange": {"link": 1}}
        peer_count = 2
        view = Mc2LaunchView(rank=0, peer_labels=(1,), node_ranks=2,
                             intra_latency_ms=0, inter_latency_ms=0)

        def start_with_handles(self, engine, name, ready, peer_ready=None,
                               peer_send=None, launch=None, source_send=None,
                               send_placeholders=None):
            peer = launch.peer_labels[0]
            placeholders = dict(send_placeholders or {})
            own = placeholders.get((launch.rank, peer, "count_status_exchange"))
            incoming = placeholders.get((peer, launch.rank, "count_status_exchange"))
            send = engine.submit(
                name + ":send", after=(ready,),
                on_complete=((lambda _event, promise=own:
                              engine.resolve(promise)) if own is not None
                             else None))
            deps = (send, incoming) if incoming is not None else (send,)
            receive = engine.submit(name + ":receive", after=deps)
            complete = engine.submit(name + ":complete", after=(receive,))
            return complete, {"peer_events": {"count_status_exchange": {peer: {
                "send": send, "receive": receive}}},
                "phase_complete": {"count_status_exchange": receive}}

    class _Idle:
        phases = (_Phase(),)
        link_by_phase = {}
        peer_count = 2
        view = Mc2LaunchView(rank=1, peer_labels=(0,), node_ranks=2,
                             intra_latency_ms=0, inter_latency_ms=0)

        def start_with_handles(self, engine, name, ready, peer_ready=None,
                               peer_send=None, launch=None, source_send=None,
                               send_placeholders=None):
            complete = engine.submit(name + ":complete", after=(ready,))
            return complete, {"peer_events": {},
                              "phase_complete": {"count_status_exchange": complete}}

    engine = Engine([], trace_limit=64)
    active = NativeProgram(_Active(), (), 0)
    idle = NativeProgram(_Idle(), (), 0)
    result = launch_native_cohort(engine, [
        {"program": active, "name": "r0",
         "ready": engine.submit("ready0", release=0.0),
         "rank_index": 0, "peer_count": 2,
         "fullmesh_phases": ("count_status_exchange",)},
        {"program": idle, "name": "r1",
         "ready": engine.submit("ready1", release=1.0),
         "rank_index": 1, "peer_count": 2},
    ])
    engine.advance(until=(result["join"],))
    assert not engine._tasks
    assert not engine._promises
    assert result["unresolved_edges"] == 0
    item = result["handles"]["r0"]["peer_events"]["count_status_exchange"][1]
    assert item["receive"].time is not None
    assert result["parents"]["r1"].time == pytest.approx(1.0)


def test_native_cohort_binds_moe16_payload_frames_to_remote_receive():
    from fractions import Fraction
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.execution import launch_native_cohort
    from mindie_sim.prediction.piecewise.compiler import NativeProgram
    from mindie_sim.prediction.piecewise.families_moe_mc2 import Mc2LaunchView

    class _Phase:
        def __init__(self, name):
            self.name = name
            self.after = ()

    class _FrameInner:
        phases = (_Phase("count_status_exchange"),
                  _Phase("payload_exchange"))
        link_by_phase = {
            "count_status_exchange": {"link": 1},
            "payload_exchange": {"link": 1},
        }
        peer_count = 4
        view = Mc2LaunchView(
            rank=0, peer_labels=(1, 2, 3), node_ranks=2,
            intra_latency_ms=Fraction("0.0018"),
            inter_latency_ms=Fraction("0.005"))

        def start_with_handles(self, engine, name, ready, peer_ready=None,
                               peer_send=None, launch=None, source_send=None,
                               send_placeholders=None):
            placeholders = dict(send_placeholders or {})
            count_done = engine.submit(
                f"{name}:count_status_exchange:phase", after=(ready,))
            payload_done = engine.submit(
                f"{name}:payload_exchange:phase", after=(ready,))
            count_sends = []
            for peer in launch.peer_labels:
                own = placeholders.get(
                    (launch.rank, peer, "count_status_exchange"))
                count_sends.append(engine.submit(
                    f"{name}:count_status_exchange:peer{peer}:send",
                    work={"link": 1}, after=(count_done,),
                    on_complete=((lambda _event, promise=own:
                                  engine.resolve(promise))
                                 if own is not None else None)))
            payload_sends = []
            receives = []
            if launch.payload_frames:
                for peer, frames in launch.payload_frames:
                    amount = frames * launch.payload_packet_bytes
                    own = placeholders.get(
                        (launch.rank, peer, "payload_exchange"))
                    send = engine.submit(
                        f"{name}:payload_exchange:peer{peer}:send",
                        work={"link": amount}, after=(payload_done,))
                    arrival = engine.submit(
                        f"{name}:payload_exchange:peer{peer}:arrival",
                        latency=float(launch.latency_for(
                            launch.class_for(
                                launch.peer_labels.index(peer), peer))),
                        after=(send,),
                        on_complete=((lambda _event, promise=own:
                                      engine.resolve(promise))
                                     if own is not None else None))
                    payload_sends.append((peer, send, arrival))
            else:
                for peer in launch.peer_labels:
                    incoming = placeholders.get(
                        (peer, launch.rank, "payload_exchange"))
                    if incoming is None:
                        continue
                    receive = engine.submit(
                        f"{name}:payload_exchange:peer{peer}:receive",
                        after=(payload_done, incoming))
                    receives.append((peer, receive))
            complete = engine.submit(
                f"{name}:complete",
                after=(count_done, payload_done, *count_sends,
                       *(send for _peer, send, _arrival in payload_sends),
                       *(receive for _peer, receive in receives)))
            peer_events = {
                "count_status_exchange": {
                    peer: {"send": event}
                    for peer, event in zip(launch.peer_labels, count_sends)},
                "payload_exchange": {
                    peer: {"send": send, "arrival": arrival}
                    for peer, send, arrival in payload_sends},
            }
            for peer, receive in receives:
                peer_events["payload_exchange"].setdefault(
                    peer, {})["receive"] = receive
            return complete, {
                "phase_complete": {
                    "count_status_exchange": count_done,
                    "payload_exchange": payload_done},
                "peer_events": peer_events,
            }

    program = NativeProgram(_FrameInner(), (), 0)
    engine = Engine([Resource("link", 1.0, "work", "fair", 64)],
                    trace_limit=128)
    launches = []
    for rank in range(4):
        launch = {
            "program": program, "name": f"r{rank}",
            "ready": engine.submit(f"ready{rank}", release=0.0),
            "rank_index": rank, "peer_count": 4, "node_ranks": 2,
            "resource_names": {"link_hccs": "link", "link_roce": "link"},
            "payload_packet_bytes": 6144,
            "payload_basis": ("actual" if rank == 0 else
                              "actual_zero" if rank in (1, 2) else
                              "unknown"),
        }
        if rank == 0:
            launch["payload_frames"] = ((1, 4), (2, 2))
        else:
            launch["payload_frames"] = ()
        launches.append(launch)
    result = launch_native_cohort(engine, launches)
    engine.advance(until=(result["join"],))
    assert not engine._tasks
    assert not engine._promises
    assert result["directed_edges"] == 2
    assert result["directed_frames"] == 6
    assert result["payload_frames"]["r0"] == {1: 4, 2: 2}
    assert result["payload_basis"]["r3"] == "unknown"
    assert any(item.get("rank") == 3 for item in result["partial"])
    r1 = result["handles"]["r1"]["peer_events"]["payload_exchange"][0]
    assert r1["receive"].time > result["handles"]["r0"]["peer_events"][
        "payload_exchange"][1]["send"].time
