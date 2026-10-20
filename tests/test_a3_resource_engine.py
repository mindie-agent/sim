"""Behavioral tests for the persistent resource engine.

These are normalized arithmetic/timeline fixtures, not A3 measurements.  They
exercise capacity sharing, admission versus service start, release events,
background carry-over, zero-work dependency cascades, finite PP and delta
outputs through the public engine surface.
"""
from __future__ import annotations

import math

import pytest

from mindie_sim.a3_resource_engine import ResourceEngine, ResourceError, ResourceSpec
from mindie_sim.a3_resources import simulate_microbatch_pipeline


def _trace(result):
    return {item["task_id"]: item for item in result["trace"]}


def _served_by_task_and_resource(result):
    served: dict[tuple[str, str], float] = {}
    for event in result["trace"]:
        for item in event["service"]:
            key = (event["task_id"], item["resource"])
            served[key] = served.get(key, 0.0) + (
                item["end_ms"] - item["start_ms"]
            ) * item["rate"]
    return served


def test_dynamic_equal_share_recomputes_at_release_and_conserves_work():
    engine = ResourceEngine({"memory": ResourceSpec("memory", 1.0)})
    engine.add_task("a", {"memory": 6.0})
    engine.add_task("b", {"memory": 6.0}, release_ms=3.0)

    result = engine.run()
    trace = _trace(result)

    # b's release dilutes a from rate 1 to rate 1/2; b completes after a.
    assert result["completed_at_ms"] == {"a": 9.0, "b": 12.0}
    assert trace["a"]["start_ms"] == 0.0
    assert trace["b"]["ready_ms"] == 3.0
    assert trace["b"]["start_ms"] == 3.0
    assert result["serviced"]["memory"] == pytest.approx(12.0)

    # Derive every task's service directly from its reported intervals; this
    # catches a timeline that reports completion without conserving work.
    arithmetic = _served_by_task_and_resource(result)
    assert arithmetic[("a", "memory")] == pytest.approx(6.0)
    assert arithmetic[("b", "memory")] == pytest.approx(6.0)

    intervals = trace["a"]["service"]
    assert intervals == pytest.approx([
        {"resource": "memory", "start_ms": 0.0, "end_ms": 3.0, "rate": 1.0},
        {"resource": "memory", "start_ms": 3.0, "end_ms": 9.0, "rate": 0.5},
    ])


def test_fifo_enqueue_priority_and_admission_versus_actual_service_start():
    engine = ResourceEngine({"lane": ResourceSpec("lane", 1.0, policy="fifo")})
    engine.add_task("a", {"lane": 5.0})
    engine.add_task("b", {"lane": 4.0})
    engine.add_task("c", {"lane": 2.0}, release_ms=2.0, priority=-1)

    result = engine.run()
    trace = _trace(result)

    # a holds the FIFO lane first, c preempts at its release, then b is served
    # despite having been admissible since t=0.
    assert result["completed_at_ms"] == {"a": 7.0, "b": 11.0, "c": 4.0}
    assert trace["b"]["ready_ms"] == 0.0
    assert trace["b"]["start_ms"] == 7.0
    assert trace["b"]["end_ms"] == 11.0
    assert trace["c"]["ready_ms"] == 2.0
    assert trace["c"]["start_ms"] == 2.0
    assert trace["b"]["service"][0]["start_ms"] == 7.0
    assert result["serviced"]["lane"] == pytest.approx(11.0)


def test_zero_work_cascade_same_loop_and_earliest_until_any_target():
    engine = ResourceEngine({"cube": ResourceSpec("cube", 1.0)})
    engine.add_task("producer", {"cube": 3.0})
    engine.add_task("z1", {}, dependencies=("producer",))
    engine.add_task("z2", {}, dependencies=("z1",))
    engine.add_task("released_zero", {}, dependencies=("producer",), release_ms=3.0)
    engine.add_task("late", {"cube": 10.0}, release_ms=50.0)

    first = engine.run(until=("z2", "late"), until_any=True)
    trace = _trace(first)

    assert first["target_completed_at_ms"] == {"z2": 3.0}
    assert trace["producer"]["end_ms"] == 3.0
    assert trace["z1"]["start_ms"] == trace["z1"]["end_ms"] == 3.0
    assert trace["z2"]["start_ms"] == trace["z2"]["end_ms"] == 3.0
    assert trace["released_zero"]["end_ms"] == 3.0
    assert "late" not in first["completed_at_ms"]
    assert first["remaining_tasks"] == ["late"]

    drained = engine.run()
    assert drained["completed_at_ms"]["late"] == 60.0
    assert drained["makespan_ms"] == 60.0


def test_future_start_services_pending_background_through_idle_gap():
    engine = ResourceEngine({
        "cube": ResourceSpec("cube", 1.0),
        "hbm": ResourceSpec("hbm", 1.0),
    })
    engine.add_task("f0", {"cube": 10.0})
    engine.add_task("prepare", {"hbm": 20.0})
    first = engine.run(until=("f0",))
    assert first["resume_ms"] == 10.0
    assert first["remaining_tasks"] == ["prepare"]

    engine.add_task(
        "f1", {"hbm": 10.0}, release_ms=100.0, dependencies=("f0",),
    )
    result = engine.run(start_ms=100.0, compact=True)

    # The clock advance must first finish the already-active HBM work, then
    # wait for f1's release.  It may not jump to 100 with old work untouched.
    assert result["completed_at_ms"] == {"prepare": 20.0, "f1": 110.0}
    assert result["new_completed_at_ms"] == {"prepare": 20.0, "f1": 110.0}
    assert result["resume_ms"] == 110.0


def test_future_start_does_not_skip_intermediate_releases():
    engine = ResourceEngine({"r": ResourceSpec("r", 1.0)})
    engine.add_task("early", {"r": 1.0}, release_ms=5.0)
    engine.add_task("late", {"r": 1.0}, release_ms=20.0, dependencies=("early",))

    result = engine.run(start_ms=20.0)
    assert result["completed_at_ms"] == {"early": 6.0, "late": 21.0}
    assert result["resume_ms"] == 21.0


def test_shared_hbm_contention_conserves_capacity_and_reports_pushout():
    engine = ResourceEngine([
        ResourceSpec("cube", 1.0),
        ResourceSpec("mte", 1.0),
        ResourceSpec("hbm", 1.0),
    ])
    engine.add_task("forward0", {"cube": 10.0, "hbm": 10.0})
    engine.add_task("prepare1", {"mte": 8.0, "hbm": 8.0})
    engine.add_task(
        "forward1", {"cube": 5.0}, dependencies=("forward0", "prepare1"),
    )

    result = engine.run()
    assert result["completed_at_ms"] == {
        "forward0": 18.0, "prepare1": 16.0, "forward1": 23.0,
    }
    assert result["serviced"]["hbm"] == pytest.approx(18.0)
    assert result["serviced"]["cube"] == pytest.approx(15.0)
    assert result["serviced"]["mte"] == pytest.approx(8.0)
    assert result["completed_at_ms"]["forward0"] - 10.0 == pytest.approx(8.0)
    assert result["completed_at_ms"]["prepare1"] - result["completed_at_ms"]["forward0"] == pytest.approx(-2.0)

    arithmetic = _served_by_task_and_resource(result)
    assert arithmetic[("forward0", "hbm")] == pytest.approx(10.0)
    assert arithmetic[("prepare1", "hbm")] == pytest.approx(8.0)
    assert arithmetic[("forward0", "cube")] == pytest.approx(10.0)
    assert arithmetic[("forward1", "cube")] == pytest.approx(5.0)


def _register_equivalence_dag(engine: ResourceEngine) -> list[str]:
    engine.add_task("read0", {"hbm": 6.0})
    engine.add_task("compute0", {"cube": 8.0}, dependencies=("read0",))
    engine.add_task("read1", {"hbm": 6.0}, release_ms=5.0)
    engine.add_task("compute1", {"cube": 3.0}, dependencies=("read1",))
    engine.add_task(
        "join", {}, dependencies=("compute0", "compute1"),
    )
    engine.add_task("tail", {"hbm": 4.0}, release_ms=10.0, dependencies=("join",))
    return ["read0", "compute0", "read1", "compute1", "join", "tail"]


def test_incremental_until_calls_equal_one_shot_after_full_registration():
    resources = {"hbm": ResourceSpec("hbm", 1.0), "cube": ResourceSpec("cube", 1.0)}
    one_shot = ResourceEngine(resources)
    incremental = ResourceEngine(resources)
    order = _register_equivalence_dag(one_shot)
    _register_equivalence_dag(incremental)

    expected = one_shot.run()
    cursor = 0
    for task_id in order:
        partial = incremental.run(until=(task_id,), compact=True, cursor=cursor)
        cursor = partial["trace_cursor"]
    actual = incremental.run()

    assert actual["completed_at_ms"] == expected["completed_at_ms"]
    assert actual["serviced"] == pytest.approx(expected["serviced"])
    one_shot_trace = _trace(expected)
    incremental_trace = _trace(actual)
    for task_id in order:
        assert incremental_trace[task_id]["start_ms"] == pytest.approx(
            one_shot_trace[task_id]["start_ms"]
        )
        assert incremental_trace[task_id]["end_ms"] == pytest.approx(
            one_shot_trace[task_id]["end_ms"]
        )


def test_progressive_add_is_monotonic_and_does_not_invent_past_service():
    engine = ResourceEngine({"cube": ResourceSpec("cube", 1.0)})
    engine.add_task("f0", {"cube": 10.0})
    assert engine.run(until=("f0",))["target_completed_at_ms"]["f0"] == 10.0

    engine.add_task("f1", {"cube": 5.0}, dependencies=("f0",))
    second = engine.run(until=("f1",))
    trace = _trace(second)
    assert second["target_completed_at_ms"]["f1"] == 15.0
    assert trace["f1"]["start_ms"] == 10.0
    assert second["completed_at_ms"]["f1"] >= 10.0


def test_compact_delta_cursor_scopes_trace_completions_and_deadlines():
    engine = ResourceEngine({"x": ResourceSpec("x", 1.0)})
    engine.add_task("token", {"x": 5.0}, deadline_ms=4.0)
    engine.add_task("background", {"x": 10.0}, deadline_ms=20.0)

    first = engine.run(until=("token",), compact=True)
    assert first["trace_cursor"] == 1
    assert [item["task_id"] for item in first["trace"]] == ["token"]
    assert first["completed_at_ms"] == {"token": 10.0}
    assert first["deadline_slack_ms"] == {"token": pytest.approx(-6.0)}
    assert first["deadline_misses"] == ["token"]

    second = engine.run(compact=True, cursor=first["trace_cursor"])
    assert second["trace_start_cursor"] == 1
    assert second["trace_cursor"] == 2
    assert [item["task_id"] for item in second["trace"]] == ["background"]
    assert second["completed_at_ms"] == {"background": 15.0}
    assert second["deadline_slack_ms"] == {"background": pytest.approx(5.0)}
    assert second["deadline_misses"] == []
    # The previous task's service is not restarted.
    assert second["trace"][0]["start_ms"] == 0.0

    third = engine.run(compact=True, cursor=second["trace_cursor"])
    assert third["trace"] == []
    assert third["new_completed_at_ms"] == {}
    assert third["drain_complete"] is True

    full = engine.run()
    assert [item["task_id"] for item in full["trace"]] == ["token", "background"]
    assert full["completed_at_ms"] == {"background": 15.0, "token": 10.0}
    assert full["deadline_slack_ms"] == {
        "token": pytest.approx(-6.0), "background": pytest.approx(5.0),
    }


def test_incremental_compact_serial_chain_returns_linear_history_once():
    count = 3000
    engine = ResourceEngine({"x": ResourceSpec("x", 1.0)})
    returned = 0
    for index in range(count):
        task_id = f"t{index}"
        dependencies = (f"t{index - 1}",) if index else ()
        engine.add_task(task_id, {"x": 1.0}, dependencies=dependencies)
        result = engine.run(until=(task_id,), compact=True)
        returned += len(result["trace"])
        assert result["trace_cursor"] == index + 1
    assert returned == count
    assert len(engine.trace) == count
    assert engine.completed_at[f"t{count - 1}"] == float(count)

    # Full export still contains the whole graph, never a truncated prefix.
    full = engine.run()
    assert len(full["trace"]) == count
    assert len(full["completed_at_ms"]) == count
    assert full["makespan_ms"] == float(count)


def test_finite_pp_schedule_and_delayed_competing_jobs():
    plain = simulate_microbatch_pipeline([3.0, 5.0], 4)
    transferred = simulate_microbatch_pipeline([3.0, 5.0], 4, transfer_ms=1.0)
    assert plain["first_token_ms"] == pytest.approx(8.0)
    assert plain["last_token_ms"] == pytest.approx(23.0)
    assert plain["makespan_ms"] == pytest.approx(23.0)
    assert transferred["first_token_ms"] == pytest.approx(9.0)
    assert transferred["last_token_ms"] == pytest.approx(24.0)
    assert transferred["makespan_ms"] == pytest.approx(24.0)
    assert len(transferred["timeline"]) == 8
    # Finite pipeline is not total-work/PP (32/2) and not a single-stage sum.
    assert plain["makespan_ms"] < sum([3.0, 5.0]) * 4

    competing = ResourceEngine({"mem": ResourceSpec("mem", 1.0)})
    competing.add_task("a", {"mem": 6.0})
    competing.add_task("b", {"mem": 6.0}, release_ms=5.0)
    competing.add_task("consumer", {}, dependencies=("a", "b"))
    result = competing.run()
    assert result["completed_at_ms"] == {"a": 7.0, "b": 12.0, "consumer": 12.0}
    assert result["serviced"]["mem"] == pytest.approx(12.0)


def test_tiny_positive_demand_is_real_work_and_horizon_is_bounded():
    tiny = ResourceEngine({"x": ResourceSpec("x", 1.0)})
    tiny.add_task("tiny", {"x": 5e-13})
    result = tiny.run()
    assert result["completed_at_ms"]["tiny"] == pytest.approx(5e-13)
    assert result["serviced"]["x"] == pytest.approx(5e-13)

    horizon = ResourceEngine({"x": ResourceSpec("x", 1.0)})
    horizon.add_task("long", {"x": 10.0})
    with pytest.raises(ResourceError, match="resource_horizon_exceeded"):
        horizon.run(horizon_ms=5.0)

    with pytest.raises(ResourceError, match="task_dependency_unknown"):
        horizon.add_task("bad", {}, dependencies=("missing",))
    with pytest.raises(ResourceError, match="resource_unknown"):
        horizon.add_task("bad_reserve", {}, reservations={"missing": 1.0})
    with pytest.raises(ResourceError, match="task_reservation_invalid"):
        horizon.add_task("bad_reserve_value", {}, reservations={"x": -1.0})
    with pytest.raises(ResourceError, match="task_reservation_mode_invalid"):
        horizon.add_task(
            "bad_reserve_mode", {}, reservations={"x": 0.1},
            reservation_mode="peak",
        )
    with pytest.raises(ResourceError, match="until_task_unknown"):
        horizon.run(until=("missing",))
    with pytest.raises(ResourceError, match="trace_cursor_invalid"):
        horizon.run(cursor=999)


def test_dynamic_resource_addition_services_new_scope_without_defaults():
    engine = ResourceEngine({"base": ResourceSpec("base", 1.0)})
    engine.resources["prep"] = ResourceSpec("prep", 1.0, "ms")
    engine.serviced["prep"] = 0.0
    engine.add_task("base", {"base": 1.0})
    engine.add_task("prep", {"prep": 4.0}, release_ms=2.0)

    result = engine.run()
    assert result["completed_at_ms"] == {"base": 1.0, "prep": 6.0}
    assert result["serviced"] == {"base": 1.0, "prep": 4.0}
    assert result["resources"]["prep"]["unit"] == "ms"

    # a3_async adds both maps, but the engine also tolerates a new serviced
    # counter being created only when work is actually served.
    tolerant = ResourceEngine({"base": ResourceSpec("base", 1.0)})
    tolerant.resources["late"] = ResourceSpec("late", 1.0, "ms")
    tolerant.add_task("late", {"late": 3.0})
    assert tolerant.run()["serviced"]["late"] == pytest.approx(3.0)


def test_future_start_services_future_release_gap_with_no_active_work():
    engine = ResourceEngine({"r": ResourceSpec("r", 1.0)})
    engine.add_task("f0", {"r": 10.0})
    assert engine.run(until=("f0",))["resume_ms"] == 10.0

    engine.add_task("background", {"r": 20.0}, release_ms=50.0)
    engine.add_task("forward", {"r": 5.0}, release_ms=100.0)
    result = engine.run(start_ms=100.0, compact=True)

    # At frontier 10 there is no active work.  The advance must still hit the
    # release at 50, service background 50..70, and only then wait for 100.
    assert result["completed_at_ms"] == {
        "background": 70.0, "forward": 105.0,
    }
    assert result["resume_ms"] == 105.0


def test_until_any_deltas_accumulate_and_engine_completed_map_is_authoritative():
    engine = ResourceEngine({
        "x": ResourceSpec("x", 1.0),
        "y": ResourceSpec("y", 1.0),
    })
    engine.add_task("a1", {"x": 1.0})
    engine.add_task("b", {"y": 2.0})
    engine.add_task("a2", {"x": 1.0}, release_ms=100.0)

    first = engine.run(until=("a1", "b"), until_any=True, compact=True)
    assert first["new_completed_at_ms"] == {"a1": 1.0}
    assert first["completed_at_ms"] == {"a1": 1.0}
    assert [item["task_id"] for item in first["trace"]] == ["a1"]

    second = engine.run(
        until=("a2", "b"), until_any=True, compact=True,
        cursor=first["trace_cursor"],
    )
    assert second["new_completed_at_ms"] == {"b": 2.0}
    assert second["completed_at_ms"] == {"b": 2.0}
    assert [item["task_id"] for item in second["trace"]] == ["b"]
    # A1 is not in second's delta; callers must accumulate deltas or consult
    # the engine's authoritative full completed map.
    assert "a1" not in second["completed_at_ms"]
    assert engine.completed_at == {"a1": 1.0, "b": 2.0}
    assert engine.completed_ms("a1") == 1.0
    assert engine.completed_ms("never") is None
    accumulated = {**first["new_completed_at_ms"], **second["new_completed_at_ms"]}
    assert accumulated == {"a1": 1.0, "b": 2.0}

    third = engine.run(compact=True, cursor=second["trace_cursor"])
    assert third["new_completed_at_ms"] == {"a2": 101.0}
    assert third["completed_at_ms"] == {"a2": 101.0}
    assert third["drain_complete"] is True


def test_optional_reservation_is_residency_not_service_work():
    engine = ResourceEngine({
        "vector": ResourceSpec("vector", 2.0),
        "wait": ResourceSpec("wait", 1.0),
    })
    engine.add_task(
        "aiv", {"vector": 1.0, "wait": 10.0},
        reservations={"vector": 1.0},
    )
    engine.add_task("other", {"vector": 3.0})

    result = engine.run()
    trace = _trace(result)

    # Fixed allocation is the default: AIV's vector work is capped at its one
    # reserved slot, so other work gets the remaining slot and finishes at 3 ms.
    # AIV keeps the slot until its wait work ends at 10 ms.
    assert result["completed_at_ms"]["aiv"] == pytest.approx(10.0)
    assert result["completed_at_ms"]["other"] == pytest.approx(3.0)
    assert result["serviced"]["vector"] == pytest.approx(4.0)
    assert result["serviced"]["wait"] == pytest.approx(10.0)
    assert trace["aiv"]["residency"] == [{
        "resource": "vector", "start_ms": 0.0, "end_ms": 10.0, "amount": 1.0,
    }]
    assert trace["aiv"]["reservation_mode"] == "fixed"
    assert "residency" not in trace["other"]

    control = ResourceEngine({
        "vector": ResourceSpec("vector", 2.0),
        "wait": ResourceSpec("wait", 1.0),
    })
    control.add_task("aiv", {"vector": 1.0, "wait": 10.0})
    control.add_task("other", {"vector": 3.0})
    assert control.run()["completed_at_ms"]["other"] == pytest.approx(2.0)

    minimum = ResourceEngine({
        "vector": ResourceSpec("vector", 2.0),
        "wait": ResourceSpec("wait", 1.0),
    })
    minimum.add_task(
        "aiv", {"vector": 1.0, "wait": 10.0},
        reservations={"vector": 1.0}, reservation_mode="minimum",
    )
    minimum.add_task("other", {"vector": 3.0})
    minimum_result = minimum.run()
    assert minimum_result["completed_at_ms"]["other"] == pytest.approx(10.0 / 3.0)
    assert minimum_result["serviced"]["vector"] == pytest.approx(4.0)


def test_simultaneous_legal_reservations_queue_and_release_capacity():
    engine = ResourceEngine({
        "vector": ResourceSpec("vector", 10.0),
        "link": ResourceSpec("link", 1.0),
        "side": ResourceSpec("side", 1.0),
    })
    engine.add_task("a", {"link": 2.0}, reservations={"vector": 6.0})
    engine.add_task("b", {"link": 2.0}, reservations={"vector": 6.0})
    # A task with no reservation keeps running while b waits for its allocation.
    engine.add_task("c", {"side": 2.0})

    result = engine.run()
    trace = _trace(result)
    assert result["completed_at_ms"] == {"a": 2.0, "b": 4.0, "c": 2.0}
    assert result["serviced"]["link"] == pytest.approx(4.0)
    assert result["serviced"]["side"] == pytest.approx(2.0)
    assert result["serviced"]["vector"] == pytest.approx(0.0)
    assert trace["b"]["start_ms"] == 2.0
    assert trace["b"]["residency"] == [{
        "resource": "vector", "start_ms": 2.0, "end_ms": 4.0, "amount": 6.0,
    }]

    invalid = ResourceEngine({"vector": ResourceSpec("vector", 10.0)})
    with pytest.raises(ResourceError, match="resource_reservation_exceeds_capacity"):
        invalid.add_task("too_big", {}, reservations={"vector": 11.0})
    assert invalid.run()["drain_complete"] is True
